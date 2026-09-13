#![allow(missing_docs)]
// WHAT A LARGE PENALTY COSTS IN MIXING -- the other half of the penalty question, exactly.
//
// `penalty_exact` settled the equilibrium half: at fixed temperature the Boltzmann mass on the
// optimal tours can only RISE with the penalty A, so "the smallest feasible penalty is best" is
// false about mass. If it is true about anything it is true about sweeps: a large A makes every
// infeasible state a high wall, and a sampler that starts anywhere must climb over walls to reach
// the tours. That is a statement about the kernel's dynamics, and on a four-city model it is
// exactly computable:
//
//   * the exact integrated autocorrelation time of the energy under sequential Gibbs at each A;
//   * the exact number of sweeps from the UNIFORM start until the mass on the optimal tours is
//     within 1% (relative) of its equilibrium value, by pushing the uniform distribution through
//     the kernel with `autocorr::apply_distribution`.
//
// Together with `penalty_exact`'s P_GS(A) they give the number a finite-budget sampler actually
// pays: the sweeps to reach a given fraction of the optimal-tour mass at each penalty. Whether a
// small A wins on that -- and by how much against what it loses in equilibrium mass -- is the
// measured answer.
//
// Count-based throughout; valid on a busy machine.
//
// run: cargo run --release --example penalty_mixing

use ferrotherm::autocorr::{apply_distribution, boltzmann, spins, tau_int_exact, Kernel};
use ferrotherm::graph::Graph;
use ferrotherm::npising::Tsp;
use ferrotherm::rng::Pcg;

struct Instance {
    a_crit: f64,
    max_w: f64,
    threshold: f64,
    t_star: f64,
    /// Per state: feasible and optimal.
    optimal: Vec<bool>,
}

fn analyse(n: usize, weights: &[f64]) -> Instance {
    let one = Tsp::with_weights(n, weights, 1.0, 1.0).expect("valid");
    let two = Tsp::with_weights(n, weights, 1.0, 2.0).expect("valid");
    let ns = one.graph().n;
    let m = 1usize << ns;
    let mut pw = Vec::with_capacity(m);
    let mut feasible = Vec::with_capacity(m);
    let mut t_star = f64::INFINITY;
    for x in 0..m {
        let s = spins(x, ns);
        let h1 = one.graph().energy(&s) + one.offset();
        let h2 = two.graph().energy(&s) + two.offset();
        let p = h2 - h1;
        let w = h1 - p;
        let f = one.decode(&s).is_ok();
        if f {
            t_star = t_star.min(w);
        }
        pw.push((p, w));
        feasible.push(f);
    }
    let mut a_crit = 0.0f64;
    for (k, &(p, w)) in pw.iter().enumerate() {
        if !feasible[k] && p > 0.0 {
            a_crit = a_crit.max((t_star - w) / p);
        }
    }
    let optimal: Vec<bool> = pw.iter().zip(&feasible).map(|(&(_, w), &f)| f && (w - t_star).abs() < 1e-9).collect();
    let max_w = weights.iter().copied().filter(|w| w.is_finite()).fold(0.0f64, f64::max);
    let threshold = Tsp::new(n, weights).expect("valid").threshold();
    Instance { a_crit, max_w, threshold, t_star, optimal }
}

/// Sweeps from the uniform start until the mass on the optimal tours is within `rel` of its
/// equilibrium value, under sequential Gibbs at `beta`; `cap` if never.
fn sweeps_to_mass(g: &Graph, beta: f64, optimal: &[bool], rel: f64, cap: usize) -> (usize, f64) {
    let m = 1usize << g.n;
    let pi = boltzmann(g, beta).expect("small");
    let target: f64 = pi.iter().zip(optimal).filter(|(_, o)| **o).map(|(p, _)| p).sum();
    let mut mu = vec![1.0 / m as f64; m];
    for k in 1..=cap {
        mu = apply_distribution(g, beta, Kernel::SequentialGibbs, &mu);
        let mass: f64 = mu.iter().zip(optimal).filter(|(_, o)| **o).map(|(p, _)| p).sum();
        if (mass - target).abs() <= rel * target {
            return (k, target);
        }
    }
    (cap, target)
}

fn main() {
    let n = 4usize;
    let instances = 12u64;
    let beta = 1.0;
    let (rel, cap) = (0.01, 3_000usize);
    println!("WHAT A PENALTY COSTS IN SWEEPS -- exact, sequential Gibbs on all 2^16 states of four-city TSPs\n");
    println!("  penalties  just above A_crit; Lucas's max(W); the provable threshold; four times it");
    println!("  columns    P_GS at equilibrium; exact tau_int of the energy (sweeps); sweeps from uniform to within {}% of P_GS (cap {cap})\n", rel * 100.0);
    println!(
        "  {:>4}   {:>22}   {:>22}   {:>22}   {:>22}",
        "inst", "A_crit+: P_GS tau K1%", "max(W): P_GS tau K1%", "thresh: P_GS tau K1%", "4x thr: P_GS tau K1%"
    );
    let mut sums = [[0.0f64; 3]; 4];
    let mut wins = [0usize; 4];
    for seed in 0..instances {
        let mut rng = Pcg::new(seed, 0x7E5);
        let mut w = vec![0.0f64; n * n];
        for u in 0..n {
            for v in (u + 1)..n {
                let e = 1.0 + (rng.f64() * 9.0).floor().min(8.0);
                w[u * n + v] = e;
                w[v * n + u] = e;
            }
        }
        let inst = analyse(n, &w);
        let penalties = [inst.a_crit * 1.05 + 1e-9, inst.max_w, inst.threshold, 4.0 * inst.threshold];
        let mut cells = Vec::new();
        let mut best_k = usize::MAX;
        let mut best_idx = 0;
        for (i, &a) in penalties.iter().enumerate() {
            let model = Tsp::with_weights(n, &w, 1.0, a).expect("valid");
            let g = model.graph();
            let tau = tau_int_exact(g, beta, Kernel::SequentialGibbs, |s| g.energy(s), 1e-10, 40_000)
                .expect("small");
            let (k, p_gs) = sweeps_to_mass(g, beta, &inst.optimal, rel, cap);
            cells.push(format!("{p_gs:>6.3} {:>7.1} {k:>6}", tau.tau_int));
            sums[i][0] += p_gs;
            sums[i][1] += tau.tau_int;
            sums[i][2] += k as f64;
            if k < best_k {
                best_k = k;
                best_idx = i;
            }
        }
        wins[best_idx] += 1;
        println!("  {seed:>4}   {}   (T* {}, A_crit {:.2}, max W {}, thr {:.0})", cells.join("   "), inst.t_star, inst.a_crit, inst.max_w, inst.threshold);
    }
    let f = instances as f64;
    println!("\n  means over {instances} instances:");
    for (i, name) in ["just above A_crit", "max(W)", "provable threshold", "4x threshold"].iter().enumerate() {
        println!(
            "    {name:<20} P_GS {:.3}   tau_int {:.1} sweeps   sweeps to 1% of P_GS {:.0}   fastest on {} instances",
            sums[i][0] / f,
            sums[i][1] / f,
            sums[i][2] / f,
            wins[i]
        );
    }
    println!("\n  WHAT THE TABLE SAYS.\n");
    println!("  P_GS rises with A -- that is the equilibrium half, a theorem. tau_int and the sweeps to reach the");
    println!("  optimal-tour mass are the dynamics half: if they rise with A faster than P_GS does, a small");
    println!("  penalty wins for a sampler with a sweep budget, and the crossover budget is where the two");
    println!("  curves meet. Every number is exact for its model; a row at the cap is a bound, not a value.");
}
