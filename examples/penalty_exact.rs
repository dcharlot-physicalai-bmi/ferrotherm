#![allow(missing_docs)]
// THE EXACT CRITICAL PENALTY OF A REDUCTION, OVER MANY INSTANCES -- and what a penalty above it
// costs in ground-state probability.
//
// Lucas (2014) reduces the travelling salesman to spins with a penalty `A` on every violated
// constraint and states the sufficient condition `0 < B max(W) < A`. `npising` proves the looser
// `A > B * (upper bound on the optimal tour)` and measured, on one four-city instance, an exact
// critical penalty of 4.5 against a `max(W)` of 6 -- so the stated condition is not tight there.
// Two questions with answers on models small enough to enumerate:
//
//   1. Over random instances, where does the exact critical penalty A_crit sit relative to
//      Lucas's `max(W)`? Is `max(W)` ever INSUFFICIENT (a ground state that is not a tour), and
//      how far above A_crit does it usually sit?
//   2. Ayodele (2022) and folk practice say a penalty as small as possible -- even below the
//      sufficient bound -- gives the best ground-state probability. Exactly, at a fixed beta, how
//      does the Boltzmann mass on the optimal tours move as A rises from A_crit through max(W)
//      to the provable threshold? Is the smallest feasible penalty the best one?
//
// A_crit is exact: over all 2^16 states of a four-city model, with the penalty count P(x) and
// tour cost W(x) of every state read off from two enumerations at two penalties, A_crit is the
// largest (T* - W(x)) / P(x) over infeasible states -- the penalty at which the best infeasible
// state ties the optimal tour. Ground-state probabilities are exact Boltzmann sums.
//
// Count-based throughout; valid on a busy machine.
//
// run: cargo run --release --example penalty_exact

use ferrotherm::autocorr::spins;
use ferrotherm::npising::Tsp;
use ferrotherm::rng::Pcg;

struct Instance {
    n_spins: usize,
    /// Per state: (penalty count P, tour cost W, feasible).
    table: Vec<(f64, f64, bool)>,
    t_star: f64,
    a_crit: f64,
    max_w: f64,
    threshold: f64,
}

fn analyse(n: usize, weights: &[f64]) -> Instance {
    // H_A(x) = A P(x) + W(x): two penalties give P and W by subtraction, exactly (integer weights).
    let one = Tsp::with_weights(n, weights, 1.0, 1.0).expect("valid");
    let two = Tsp::with_weights(n, weights, 1.0, 2.0).expect("valid");
    let n_spins = one.graph().n;
    let m = 1usize << n_spins;
    let mut table = Vec::with_capacity(m);
    let mut t_star = f64::INFINITY;
    for x in 0..m {
        let s = spins(x, n_spins);
        let h1 = one.graph().energy(&s) + one.offset();
        let h2 = two.graph().energy(&s) + two.offset();
        let p = h2 - h1;
        let w = h1 - p;
        let feasible = one.decode(&s).is_ok();
        if feasible {
            t_star = t_star.min(w);
        }
        table.push((p, w, feasible));
    }
    // The cheapest tour by brute force over permutations is the oracle for T*.
    let mut best = f64::INFINITY;
    let mut perm: Vec<usize> = (0..n).collect();
    fn permute(k: usize, perm: &mut Vec<usize>, tsp: &Tsp, best: &mut f64) {
        if k == perm.len() {
            if let Some(c) = tsp.tour_cost(perm) {
                *best = best.min(c);
            }
            return;
        }
        for i in k..perm.len() {
            perm.swap(k, i);
            permute(k + 1, perm, tsp, best);
            perm.swap(k, i);
        }
    }
    permute(0, &mut perm, &one, &mut best);
    assert!((best - t_star).abs() < 1e-9, "the spin model's best tour {t_star} must be the permutation optimum {best}");
    let mut a_crit = 0.0f64;
    for &(p, w, feasible) in &table {
        if !feasible && p > 0.0 {
            a_crit = a_crit.max((t_star - w) / p);
        }
    }
    let max_w = weights.iter().copied().filter(|w| w.is_finite()).fold(0.0f64, f64::max);
    let threshold = Tsp::new(n, weights).expect("valid").threshold();
    Instance { n_spins, table, t_star, a_crit, max_w, threshold }
}

/// Exact Boltzmann mass on the optimal tours at penalty `a` and inverse temperature `beta`.
fn ground_mass(inst: &Instance, a: f64, beta: f64) -> f64 {
    let energies: Vec<f64> = inst.table.iter().map(|&(p, w, _)| a * p + w).collect();
    let emin = energies.iter().copied().fold(f64::INFINITY, f64::min);
    let mut z = 0.0;
    let mut mass = 0.0;
    for (k, &e) in energies.iter().enumerate() {
        let b = (-beta * (e - emin)).exp();
        z += b;
        let (_, w, feasible) = inst.table[k];
        if feasible && (w - inst.t_star).abs() < 1e-9 {
            mass += b;
        }
    }
    mass / z
}

fn main() {
    let n = 4usize;
    let instances = 300u64;
    let beta = 1.0;
    println!("THE EXACT CRITICAL PENALTY OF LUCAS'S TSP REDUCTION, OVER {instances} RANDOM FOUR-CITY INSTANCES\n");
    println!("  weights   integers drawn uniformly from 1..=9, symmetric, complete graph; B = 1");
    println!("  A_crit    the largest (T* - W(x)) / P(x) over infeasible x, from all 2^16 states, exact");
    println!("  P_GS      exact Boltzmann mass on the optimal tours at beta = {beta}, as a function of A\n");
    let mut ratios: Vec<f64> = Vec::new();
    let mut insufficient = 0usize;
    let mut half_tstar_violations = 0usize;
    let (mut best_is_smallest, mut best_is_maxw, mut best_is_thresh) = (0usize, 0usize, 0usize);
    let mut sum_mass = [0.0f64; 4];
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
        ratios.push(inst.a_crit / inst.max_w);
        if inst.a_crit > inst.max_w {
            insufficient += 1;
        }
        if inst.a_crit > 0.5 * inst.t_star {
            half_tstar_violations += 1;
        }
        // Ground-state mass just above the critical penalty, at Lucas's bound, and at the
        // provable threshold this crate defaults to.
        let a_small = inst.a_crit * 1.001 + 1e-9;
        let masses = [
            ground_mass(&inst, a_small, beta),
            ground_mass(&inst, inst.max_w, beta),
            ground_mass(&inst, inst.threshold, beta),
            ground_mass(&inst, 4.0 * inst.threshold, beta),
        ];
        for (k, m) in masses.iter().enumerate() {
            sum_mass[k] += m;
        }
        let best = masses.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if (masses[0] - best).abs() < 1e-12 {
            best_is_smallest += 1;
        } else if (masses[1] - best).abs() < 1e-12 {
            best_is_maxw += 1;
        } else if (masses[2] - best).abs() < 1e-12 {
            best_is_thresh += 1;
        }
        if seed < 5 {
            println!(
                "  instance {seed}: T* = {}, max(W) = {}, A_crit = {:.3} ({:.2} x max W), threshold {:.1}; \
                 P_GS at A_crit+ {:.4}, at max W {:.4}, at threshold {:.4}, at 4x threshold {:.4}  [{} spins]",
                inst.t_star, inst.max_w, inst.a_crit, inst.a_crit / inst.max_w, inst.threshold,
                masses[0], masses[1], masses[2], masses[3], inst.n_spins
            );
        }
    }
    ratios.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q = |f: f64| ratios[((ratios.len() - 1) as f64 * f) as usize];
    println!("\n  A_crit / max(W) over {instances} instances: min {:.3}, 10% {:.3}, median {:.3}, 90% {:.3}, max {:.3}",
             ratios[0], q(0.1), q(0.5), q(0.9), ratios[ratios.len() - 1]);
    println!("  instances where Lucas's max(W) is INSUFFICIENT (A_crit > max W): {insufficient} of {instances}");
    println!("  instances where A_crit exceeds T*/2 (the bound npising's wave-4 note proved): {half_tstar_violations} of {instances}");
    let inst_f = instances as f64;
    println!("\n  mean P_GS at beta {beta}:  just above A_crit {:.4}   at max(W) {:.4}   at the provable threshold {:.4}   at 4x threshold {:.4}",
             sum_mass[0] / inst_f, sum_mass[1] / inst_f, sum_mass[2] / inst_f, sum_mass[3] / inst_f);
    println!("  best of the three feasible penalties: smallest {best_is_smallest}, max(W) {best_is_maxw}, threshold {best_is_thresh}  (ties counted for the smaller)");
    println!("\n  WHAT THE TABLE SAYS.\n");
    println!("  A_crit is exact and instance-specific; the distribution of A_crit / max(W) is how far Lucas's");
    println!("  stated bound sits from tight, and the INSUFFICIENT count is whether it ever fails. The P_GS");
    println!("  columns are the exact price of a penalty above the critical one: every unit of A above A_crit");
    println!("  lifts every infeasible state further, which sharpens the landscape, and whether that helps or");
    println!("  hurts the mass on the optimal tours at a fixed temperature is the measured column, not a rule.");
}
