//! Training through the dynamics, and what the integrator costs the training signal.
//!
//! ```text
//!   cargo run --release --example pathwise
//! ```
//!
//! `docs/ABSORPTION.md` lists differentiable simulation as the second thing this crate needed in
//! order to contain the coupled-oscillator programme: their models are trained by sending a
//! gradient back through the integrator, and every estimator here before `pathwise` differentiated
//! a log-probability instead.
//!
//! Section 1 checks the gradient against central differences, because a gradient nobody checked is
//! a gradient of whatever function was implemented. Section 2 is the finding: a step size chosen by
//! watching the trajectory is not a step size the gradient agrees with. Section 3 uses the gradient
//! to fit a coupling, which is the point of having it.
use ferrotherm::kuramoto::Kuramoto;
use ferrotherm::pathwise::{
    euler_adjoint, feature_cotangent, features_of, integrator_gap, trajectory,
};
use ferrotherm::rng::Pcg;
use ferrotherm::round::sum_up;

/// A dense asymmetric system with natural frequencies — the published shape, at a size that fits
/// on a page.
fn fixture(n: usize, seed: u64, scale: f64) -> Kuramoto {
    let mut rng = Pcg::new(seed, 0x9A7_4EED);
    let mut k = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            if i != j {
                k[i * n + j] = scale * (2.0 * rng.f64() - 1.0);
            }
        }
    }
    let omega: Vec<f64> = (0..n).map(|_| 0.6 * (2.0 * rng.f64() - 1.0)).collect();
    Kuramoto::new(omega, k).expect("well formed")
}

/// A linear loss on the readout features, and its cotangent.
fn loss(sys: &Kuramoto, phi0: &[f64], dt: f64, steps: usize, g: &[f64]) -> f64 {
    let tape = trajectory(sys, phi0, dt, steps);
    let u = features_of(tape.last().expect("a tape carries its initial state"));
    let terms: Vec<f64> = (0..u.len()).map(|i| g[i] * u[i]).collect();
    sum_up(&terms)
}

fn main() {
    // ---- 1. the gradient is the gradient -------------------------------------------------
    println!("== 1. the adjoint against central differences ==");
    let n = 4;
    let sys = fixture(n, 4242, 1.4);
    let (dt, steps) = (0.02f64, 25usize);
    let phi0: Vec<f64> = (0..n).map(|i| 0.3 * (i as f64 + 1.0)).collect();
    let g: Vec<f64> = (0..2 * n).map(|i| 0.2 + 0.11 * i as f64).collect();
    let tape = trajectory(&sys, &phi0, dt, steps);
    let end = tape.last().expect("tape").clone();
    let grads = euler_adjoint(&sys, &tape, dt, &feature_cotangent(&end, &g));
    let h = 1e-6;
    let mut worst = 0.0f64;
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let base: Vec<f64> = (0..n * n).map(|a| sys.coupling(a / n, a % n)).collect();
            let mut up = base.clone();
            let mut down = base;
            up[i * n + j] += h;
            down[i * n + j] -= h;
            let fd = (loss(
                &Kuramoto::new(sys.omega().to_vec(), up).expect("well formed"),
                &phi0,
                dt,
                steps,
                &g,
            ) - loss(
                &Kuramoto::new(sys.omega().to_vec(), down).expect("well formed"),
                &phi0,
                dt,
                steps,
                &g,
            )) / (2.0 * h);
            worst = worst.max((grads.d_k[i * n + j] - fd).abs());
        }
    }
    println!("  {} coupling entries, worst |adjoint - finite difference| = {worst:.3e}", n * (n - 1));
    println!("  (the forward map is pinned to the two-oscillator closed form in the tests first:");
    println!("   a gradient check alone validates the derivative of whatever was implemented)");

    // ---- 2. what the integrator costs the gradient ---------------------------------------
    println!("\n== 2. Euler against a fourth-order reference, in the trajectory and the gradient ==");
    println!("  each error relative to the size of the quantity it belongs to");
    let cases: [(&str, usize, u64, f64, usize); 5] = [
        ("n=4  K~1.4   20 steps", 4, 2026, 1.4, 20),
        ("n=4  K~1.4  100 steps", 4, 2026, 1.4, 100),
        ("n=8  K~1.4   50 steps", 8, 7, 1.4, 50),
        ("n=4  K~3.0   50 steps", 4, 11, 3.0, 50),
        ("n=6  K~0.5  200 steps", 6, 5, 0.5, 200),
    ];
    let (mut above, mut total, mut biggest) = (0usize, 0usize, 0.0f64);
    for (label, n, seed, scale, steps) in cases {
        let sys = fixture(n, seed, scale);
        let phi0: Vec<f64> = (0..n).map(|i| 0.4 * (i as f64 + 1.0)).collect();
        let g: Vec<f64> = (0..2 * n).map(|i| 0.5 + 0.2 * i as f64).collect();
        println!("  -- {label} --");
        println!("  {:>8} {:>12} {:>12} {:>9}", "dt", "phase rel", "grad rel", "penalty");
        for dt in [0.2f64, 0.1, 0.05, 0.025, 0.0125] {
            let gap = integrator_gap(&sys, &phi0, dt, steps, 64, &g);
            let p = gap.gradient_penalty();
            total += 1;
            if p > 1.0 {
                above += 1;
            }
            biggest = biggest.max(p);
            println!(
                "  {:>8.4} {:>12.3e} {:>12.3e} {:>8.2}x",
                dt, gap.phase_relative, gap.grad_relative, p
            );
        }
    }
    println!();
    println!("  the gradient's relative error exceeds the trajectory's in {above} of {total},");
    println!("  by as much as {biggest:.0}x. The exceptions are the smallest, most weakly coupled");
    println!("  system at its finest steps, where both errors are already under a percent --");
    println!("  they are kept because the temptation is to state this as a theorem, and it is not.");
    println!("  What it is: the two errors are not proxies for one another, and dt is a trained");
    println!("  parameter in disguise. A model fitted through a coarse Euler map absorbed that");
    println!("  map's error into its weights, and silicon has no such error to cancel.");

    // ---- 3. the gradient is usable -------------------------------------------------------
    println!("\n== 3. fitting a coupling with it ==");
    let target = fixture(4, 909, 1.0);
    let phi0: Vec<f64> = (0..4).map(|i| 0.25 * (i as f64 + 1.0)).collect();
    let want = features_of(
        trajectory(&target, &phi0, 0.05, 40).last().expect("tape"),
    );
    // Start from no coupling at all, keeping the target's natural frequencies.
    let mut k = vec![0.0f64; 16];
    let mut report = Vec::new();
    for step in 0..=400 {
        let sys = Kuramoto::new(target.omega().to_vec(), k.clone()).expect("well formed");
        let tape = trajectory(&sys, &phi0, 0.05, 40);
        let end = tape.last().expect("tape").clone();
        let u = features_of(&end);
        let residual: Vec<f64> = (0..8).map(|i| u[i] - want[i]).collect();
        let sq: Vec<f64> = residual.iter().map(|r| r * r).collect();
        if step % 100 == 0 {
            report.push((step, sum_up(&sq)));
        }
        if step == 400 {
            break;
        }
        // d(sum r^2)/du = 2r, which is the feature cotangent of a squared-error loss.
        let cot: Vec<f64> = residual.iter().map(|r| 2.0 * r).collect();
        let grads = euler_adjoint(&sys, &tape, 0.05, &feature_cotangent(&end, &cot));
        for a in 0..16 {
            k[a] -= 0.35 * grads.d_k[a];
        }
        for a in 0..4 {
            k[a * 4 + a] = 0.0; // the diagonal does not act and the constructor refuses it
        }
    }
    for (step, l) in &report {
        println!("  step {step:>4}: squared feature error {l:.3e}");
    }
    println!("  fitted from a coupling of zero, by the gradient of section 1 alone.");
}
