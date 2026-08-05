// Differentiating THROUGH a Gibbs kernel — the bridge between the program layer and the sampler.
//
// A K-sweep Glauber kernel has an exactly tractable trajectory log-density (every spin update is
// a Bernoulli with known probability), so REINFORCE through it is unbiased with no approximation.
// Verify the score against central finite differences with common random numbers at three bias
// points, then train the biases to shape the sampled distribution and report the loss drop.
//
// Loss: L(s) = ((sum_i s_i)^2) / n  — favors zero-magnetization states; training the biases of a
// ferromagnetically coupled ring must fight the coupling to balance the spins.
//
// run: cargo run --release --example gibbs_grad

use ferrotherm::ising::ring;
use ferrotherm::program::{Force, Gate, Program, State};
use ferrotherm::rng::Pcg;

fn main() {
    let n = 6usize;
    let g = ring(n, 0.5, 0.0);
    let prog = Program {
        gates: vec![Gate::GibbsK { g: 0, sweeps: 4, beta: 0.8, p_h0: 0 }],
        graphs: vec![g],
        n_params: n,
    };
    let init = State { bits: vec![1; n], reals: vec![] };
    let loss = |s: &State| {
        let m: i64 = s.bits.iter().map(|&v| v as i64).sum();
        (m * m) as f64 / n as f64
    };

    // ---- 1. gradient cross-check at three bias points ----
    println!("REINFORCE-through-Gibbs vs finite differences (common random numbers), 6-node ring:");
    println!("  {:>28} {:>12} {:>12} {:>9}", "bias point", "REINFORCE", "FD referee", "|delta|");
    let mut worst: f64 = 0.0;
    for (tag, h) in [
        ("h = 0", vec![0.0; n]),
        ("h = +0.3 uniform", vec![0.3; n]),
        ("h = alternating +-0.4", (0..n).map(|i| if i % 2 == 0 { 0.4 } else { -0.4 }).collect()),
    ] {
        let (grf, _) = prog.reinforce_grad(&init, &h, &loss, 300_000, 0x9B5);
        // check two representative coordinates against FD
        for j in [0usize, 3] {
            let gfd = prog.fd_grad(j, 0.05, &init, &h, &loss, 300_000, 0xFD & j as u64 ^ 0xABC);
            let d = (grf[j] - gfd).abs();
            worst = worst.max(d);
            println!("  {:>28} {:>12.4} {:>12.4} {:>9.4}   (coord {})", tag, grf[j], gfd, d, j);
        }
    }
    let grad_ok = worst < 0.05;
    println!("  gradient check: {}\n", if grad_ok { "PASS" } else { "FAIL" });

    // ---- 2. train the biases to reshape the sampled distribution ----
    let mut h = vec![0.0; n];
    let mut before = 0.0;
    let mut after = 0.0;
    let iters = 60;
    for it in 0..iters {
        let (grad, mean_l) = prog.reinforce_grad(&init, &h, &loss, 20_000, 0x7A1 + it as u64);
        if it == 0 {
            before = mean_l;
        }
        after = mean_l;
        for j in 0..n {
            h[j] -= 0.25 * grad[j];
        }
    }
    // held-out evaluation of the trained program
    let eval = |h: &[f64], seed: u64| {
        let mut acc = 0.0;
        let m = 50_000;
        for e in 0..m {
            let mut rng = Pcg::new(seed, e as u64);
            let st = prog.run(&init, &mut rng, None, Force::None, h, None, None);
            acc += loss(&st);
        }
        acc / m as f64
    };
    let l0 = eval(&vec![0.0; n], 0xE0);
    let l1 = eval(&h, 0xE1);
    println!("training the Gibbs biases against E[(sum s)^2/n] (ferromagnetic ring resists it):");
    println!("  E[L] untrained = {l0:.3}   trained = {l1:.3}   (training-curve {before:.3} -> {after:.3})");
    println!("  learned biases: {:?}", h.iter().map(|v| (v * 100.0).round() / 100.0).collect::<Vec<_>>());
    let train_ok = l1 < 0.75 * l0;
    println!(
        "  verdict: {}",
        if grad_ok && train_ok {
            "PASS — the sampler is a differentiable program component"
        } else {
            "FAIL"
        }
    );
    std::process::exit(if grad_ok && train_ok { 0 } else { 1 });
}
