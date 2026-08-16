// The embodied demo: train a stochastic-program CONTROLLER by gradient descent and verify it
// against the provable optimum — control cost is where our E_task framing meets this layer.
//
// System: scalar linear plant x' = x + 0.1 u, 40 steps, x0 = 1.5, target 0.
// Policy gate: u ~ N(k * (0 - x), 0.4^2) — one trainable parameter, the gain k.
// Cost: sum_t [ x_t^2 + 0.2 u_t^2 ]  (state error + control effort, the actuation-proxy term).
//
// For a linear plant + Gaussian policy the expected cost has an exact deterministic recursion in
// (mean, variance), so the best stationary gain k* is computable to machine precision WITHOUT
// sampling. The stochastic-program training must land on it.
//
// run: cargo run --release --example lqr_energy

use ferrotherm::program::{Gate, Program, State};

const A: f64 = 1.0;
const B: f64 = 0.1;
const Q: f64 = 1.0;
const R: f64 = 0.2;
const SG: f64 = 0.4;
const T: usize = 40;
const X0: f64 = 1.5;

/// Exact expected cost of stationary gain k: propagate mean and variance of x, accumulate
/// E[q x^2 + r u^2] with u = -k x + SG eps  =>  E[u^2] = k^2 E[x^2] + SG^2.
fn exact_cost(k: f64) -> f64 {
    let (mut m, mut v) = (X0, 0.0f64);
    let mut cost = 0.0;
    for _ in 0..T {
        let ex2 = m * m + v;
        cost += Q * ex2 + R * (k * k * ex2 + SG * SG);
        let c = A - B * k;
        m *= c;
        v = c * c * v + B * B * SG * SG;
    }
    cost
}

fn main() {
    // ---- closed-form reference: golden-section search on the exact cost ----
    let (mut lo, mut hi) = (0.0f64, 9.0f64);
    let phi = (5.0f64.sqrt() - 1.0) / 2.0;
    for _ in 0..80 {
        let c = hi - phi * (hi - lo);
        let d = lo + phi * (hi - lo);
        if exact_cost(c) < exact_cost(d) {
            hi = d;
        } else {
            lo = c;
        }
    }
    let k_star = (lo + hi) / 2.0;
    let c_star = exact_cost(k_star);

    // ---- the stochastic program: T repetitions of (Err ; CtrlGauss ; Lin), cost via aux wires ----
    // wires: reals[0]=x, reals[1]=err, reals[2]=u, reals[3]=accumulated cost
    let mut gates = Vec::new();
    for _ in 0..T {
        gates.push(Gate::Err { err: 1, x: 0, tgt: 0.0 });
        gates.push(Gate::CtrlGauss { u: 2, err: 1, p_k: 0, sigma: SG });
        gates.push(Gate::CostQuad { acc: 3, x: 0, u: 2, q: Q, r: R }); // stage cost BEFORE the step
        gates.push(Gate::Lin { x: 0, u: 2, a: A, b: B });
    }
    let prog = Program { gates, graphs: vec![], n_params: 1 };
    let init = State { bits: vec![], reals: vec![X0, 0.0, 0.0, 0.0] };
    let loss = |s: &State| s.reals[3];

    // ---- verify the program's expected cost against the exact recursion at a probe gain ----
    let probe_k = 2.0;
    let (_, mean_probe) = prog.reinforce_grad(&init, &[probe_k], &loss, 100_000, 0xCAB);
    let exact_probe = exact_cost(probe_k);
    println!("probe k = {probe_k}: program E[cost] = {mean_probe:.3}, exact recursion = {exact_probe:.3}");
    let probe_ok = (mean_probe - exact_probe).abs() / exact_probe < 0.01;
    println!("  agreement: {}\n", if probe_ok { "ok (<1%)" } else { "FAIL" });

    // ---- train the gain by REINFORCE ----
    let mut k = 0.3f64;
    let iters = 250;
    for it in 0..iters {
        let (grad, _) = prog.reinforce_grad(&init, &[k], &loss, 4_000, 0x11E + it as u64);
        let lr = 0.02 / (1.0 + it as f64 / 80.0);
        k -= lr * grad[0];
    }
    let c_trained = exact_cost(k); // evaluate the LEARNED gain on the exact recursion
    println!("trained by stochastic-program gradient descent (REINFORCE, 250 iters):");
    println!("  k trained = {k:.3}    k* exact = {k_star:.3}");
    println!("  expected cost at k trained = {c_trained:.3}   at k* = {c_star:.3}   excess = {:.2}%",
             100.0 * (c_trained - c_star) / c_star);
    println!("  (control-effort term R*E[sum u^2] is the actuation proxy: what the arm would pay for this policy)");

    let ok = probe_ok && (k - k_star).abs() < 0.2 && (c_trained - c_star) / c_star < 0.01;
    println!("\n  verdict: {}", if ok {
        "PASS — trained stochastic controller matches the provable optimum"
    } else {
        "FAIL"
    });
    std::process::exit(if ok { 0 } else { 1 });
}
