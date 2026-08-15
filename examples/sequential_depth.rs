//! What does a sampling controller's accuracy cost, and which half of that cost can hardware buy down?
//!
//! TR-2026-40 argues that a sampling controller's wall-clock is set by its **sequential depth**
//! rather than by its total sample count, because parallel lanes divide the samples and cannot
//! divide a refinement chain. That argument is only worth making with a number attached, and the
//! number has to be scored against something better than a rival heuristic.
//!
//! MPPI on a linear-quadratic plant has an exact oracle: the discrete algebraic Riccati equation
//! gives the optimal cost in closed form, so "within 10% of optimal" is a statement about truth
//! rather than about a baseline someone chose. This probe sweeps the two knobs that buy accuracy
//! and separates them by what a machine can do with them:
//!
//! - **rollouts** are independent. Thousands of lanes evaluate them at once, so their cost is
//!   throughput, and throughput is what the last two decades of hardware bought.
//! - **iters** are refinement passes. Each one perturbs the mean the previous one produced, so
//!   they cannot overlap. Their cost is latency, and latency is what parallel hardware leaves
//!   untouched.
//!
//! The output is the frontier: for each accuracy target, the cheapest configuration that reaches
//! it, and how much of that cost is sequential. Run:
//!
//! ```text
//! cargo run --release --example sequential_depth
//! ```

use ferrotherm::mppi::{Lqr, Mppi, System};

/// Median over seeds. A single seed on a stochastic controller reports the seed, not the setting;
/// the spread is printed alongside so a difference smaller than it is not read as a difference.
fn score(m: &Mppi, s: &System, seeds: &[u64]) -> (f64, f64, f64) {
    let mut v: Vec<f64> = seeds.iter().map(|&sd| m.run(s, 1.0, 200, sd)).collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (v[v.len() / 2], v[0], v[v.len() - 1])
}

fn main() {
    // A stable plant and an unstable one. The unstable one is the interesting case: a controller
    // that is merely close on a self-correcting plant has not been tested.
    for (label, s) in [
        ("stable   a=0.9", System { a: 0.9, b: 1.0, q: 1.0, r: 0.5 }),
        ("unstable a=1.1", System { a: 1.1, b: 1.0, q: 1.0, r: 0.5 }),
    ] {
        let l = Lqr::solve(&s);
        let opt = l.cost_to_go(1.0);
        println!("\n=== {label} ===");
        println!("exact optimum (Riccati): {opt:.4}\n");

        let seeds: Vec<u64> = (1..=7).collect();
        let mut rows: Vec<(usize, usize, f64, f64)> = Vec::new(); // (rollouts, iters, excess%, spread%)

        for &k in &[50usize, 200, 800, 3200] {
            for &it in &[1usize, 2, 4, 8, 16] {
                let m = Mppi { horizon: 5, rollouts: k, sigma: 0.2, lambda: 0.3, iters: it };
                let (med, lo, hi) = score(&m, &s, &seeds);
                rows.push((k, it, (med - opt) / opt * 100.0, (hi - lo) / opt * 100.0));
            }
        }

        println!("{:>9} {:>6} {:>10} {:>9}   {:>12} {:>12}",
                 "rollouts", "iters", "excess %", "spread %", "total draws", "seq depth");
        for &(k, it, ex, sp) in &rows {
            println!("{k:>9} {it:>6} {ex:>9.2}% {sp:>8.2}%   {:>12} {it:>12}", k * it);
        }

        // The frontier: cheapest way to reach each accuracy target, by each of the two costs.
        println!("\n  target      min total draws (cfg)        min seq depth (cfg)");
        for &target in &[25.0f64, 15.0, 10.0] {
            let ok: Vec<_> = rows.iter().filter(|r| r.2 <= target).collect();
            if ok.is_empty() {
                println!("  <={target:>4.0}%      not reached in this sweep      not reached");
                continue;
            }
            let by_draws = ok.iter().min_by_key(|r| r.0 * r.1).unwrap();
            let by_depth = ok.iter().min_by_key(|r| r.1).unwrap();
            println!("  <={:>4.0}%   {:>8} ({} x {})            {:>3} ({} x {})",
                     target, by_draws.0 * by_draws.1, by_draws.0, by_draws.1,
                     by_depth.1, by_depth.0, by_depth.1);
        }

        // The load-bearing comparison: hold accuracy fixed and ask what each knob is worth.
        // If depth is doing the work, adding rollouts at fixed depth stalls while adding depth
        // at fixed rollouts keeps paying.
        println!("\n  holding one knob fixed:");
        let at = |k: usize, it: usize| rows.iter().find(|r| r.0 == k && r.1 == it).map(|r| r.2).unwrap();
        println!("    rollouts 50 -> 3200 at iters=1  : {:>6.2}% -> {:>6.2}%  ({:>5.1} pts)",
                 at(50, 1), at(3200, 1), at(50, 1) - at(3200, 1));
        println!("    iters 1 -> 16 at rollouts=50    : {:>6.2}% -> {:>6.2}%  ({:>5.1} pts)",
                 at(50, 1), at(50, 16), at(50, 1) - at(50, 16));
        println!("    iters 1 -> 16 at rollouts=3200  : {:>6.2}% -> {:>6.2}%  ({:>5.1} pts)",
                 at(3200, 1), at(3200, 16), at(3200, 1) - at(3200, 16));
    }

    println!("\nRollouts are independent and divide across lanes. Iters are a chain and do not.");
    println!("Whichever knob carries the accuracy is the one that decides what hardware can buy.");
}
