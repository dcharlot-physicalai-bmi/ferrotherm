//! Learning theory as oracles: what the samplers must reproduce, in closed form.
//!
//! The founding discipline of this crate is the Onsager check -- a closed form the sampler must
//! hit before it is trusted. This example extends it to the statistical mechanics of learning:
//! the mean-field family against exact ln Z (one of them a theorem), belief propagation exact on a
//! tree and degrading with loops, and the Hopfield memory against Curie-Weiss and the
//! Amit-Gutfreund-Sompolinsky replica theory, capacity included.
//!
//! usage: cargo run --release --example learning_theory

use ferrotherm::exact::Elimination;
use ferrotherm::free_energy::exact_log_z;
use ferrotherm::graph::GraphBuilder;
use ferrotherm::hopfield::{ags_rs, ags_zero_t, capacity_zero_t, curie_weiss_m, hebbian, random_patterns, retrieval_overlap};
use ferrotherm::ising;
use ferrotherm::meanfield::{belief_propagation, naive_mean_field, tap};
use ferrotherm::rng::Pcg;

fn main() {
    // ---- 1. the mean-field family against exact ln Z ---------------------------------------
    println!("--- ln Z: exact vs mean field (a lower bound, always), TAP, Bethe");
    println!("{:<28} {:>6} {:>9} {:>9} {:>9} {:>9}", "model", "beta", "exact", "MF bound", "TAP", "Bethe");
    let mut rng = Pcg::new(3, 0);
    let mut gb = GraphBuilder::new(20);
    for i in 1..20 {
        let parent = (rng.f64() * i as f64) as usize;
        gb.couple(parent, i, 2.0 * rng.f64() - 1.0);
    }
    for i in 0..20 {
        gb.bias(i, rng.f64() - 0.5);
    }
    let tree = gb.build();
    let rows: Vec<(&str, ferrotherm::graph::Graph, f64)> = vec![
        ("random tree n=20", tree, 0.9),
        ("4x4 torus, disordered", ising::lattice2d(4, 1.0), 0.3),
        ("4x4 torus, ordered", ising::lattice2d(4, 1.0), 0.5),
        ("ring n=16 h=0.2", ising::ring(16, 1.0, 0.2), 1.0),
    ];
    for (name, g, beta) in &rows {
        let exact = if g.n <= 24 { exact_log_z(g, *beta) } else { Elimination::default().log_partition(g, *beta).unwrap().log_z.unwrap() };
        let mf = naive_mean_field(g, *beta, 5000, 0.3);
        let tp = tap(g, *beta, 5000, 0.5);
        let bp = belief_propagation(g, *beta, 5000, 0.5);
        println!(
            "{:<28} {:>6.2} {:>9.4} {:>9.4} {:>9.4} {:>9.4}{}",
            name, beta, exact, mf.log_z, tp.log_z, bp.log_z,
            if !bp.converged(1e-8) { "  (BP did not converge)" } else { "" }
        );
    }
    println!("  the MF column never exceeds exact (Gibbs-Bogoliubov); Bethe equals exact on the tree\n  and drifts on the torus once beta passes 0.44 -- loops reinforce, and BP cannot see them.\n");

    // ---- 2. Hopfield: one pattern is Curie-Weiss -------------------------------------------
    println!("--- Hopfield, one pattern, N=256: the overlap solves m = tanh(beta m)");
    println!("{:<8} {:>12} {:>20}", "beta", "Curie-Weiss", "sampled overlap");
    let pats = random_patterns(256, 1, 1);
    let g = hebbian(&pats);
    for beta in [1.2, 1.5, 2.0, 3.0] {
        let got = retrieval_overlap(&g, &pats[0], beta, 200, 2000, 9);
        println!("{:<8.2} {:>12.4} {:>13.4} +- {:.4}", beta, curie_weiss_m(beta), got.value, got.stderr);
    }
    println!();

    // ---- 3. Hopfield at finite load: the AGS replica theory and the capacity -----------------
    println!("--- Hopfield at load alpha = P/N, beta = 2 (T = 0.5), N = 1000");
    println!("{:<8} {:>4} {:>10} {:>20}", "alpha", "P", "AGS m", "sampled overlap");
    for (alpha, p, seed) in [(0.02, 20usize, 2u64), (0.10, 100, 4), (0.30, 300, 5)] {
        let rs = ags_rs(alpha, 2.0);
        let pats = random_patterns(1000, p, seed);
        let g = hebbian(&pats);
        let got = retrieval_overlap(&g, &pats[0], 2.0, 100, 300, seed + 10);
        println!(
            "{:<8.2} {:>4} {:>10} {:>13.4} +- {:.4}",
            alpha, p, if rs.m > 0.05 { format!("{:.4}", rs.m) } else { "none".into() }, got.value, got.stderr
        );
    }
    // alpha = 0.05 is the instructive one: the retrieval state exists (AGS m = 0.904) but above
    // alpha ~ 0.05 it is METASTABLE, and whether a finite-N chain stays in it for 400 sweeps
    // depends on the pattern set. So it is reported per set, with a colder control.
    for beta in [2.0, 4.0] {
        let rs = ags_rs(0.05, beta);
        let finals: Vec<f64> = (3..8u64)
            .map(|seed| {
                let pats = random_patterns(1000, 50, seed);
                retrieval_overlap(&hebbian(&pats), &pats[0], beta, 100, 300, seed + 10).value
            })
            .collect();
        let held = finals.iter().filter(|&&m| m > 0.8).count();
        println!(
            "{:<8.2} {:>4} {:>10.4}   held in {held} of 5 pattern sets (beta {beta}); overlaps {:?}",
            0.05, 50, rs.m, finals.iter().map(|m| (m * 100.0).round() / 100.0).collect::<Vec<_>>()
        );
    }
    println!("  The replica theory is a statement about N -> infinity and about which states are\n  \
              free-energy minima; a finite-N chain can leave a metastable one, and at alpha = 0.05,\n  \
              T = 0.5 it sometimes does within the run. Where it stays, the overlap is the theory's.");
    let ac = capacity_zero_t();
    let below = ags_zero_t(ac - 0.002).unwrap();
    println!(
        "\n  capacity at T = 0 from the crate's own replica-symmetric numerics: alpha_c = {ac:.4}\n  \
         (Amit-Gutfreund-Sompolinsky: 0.138); overlap just below it {:.3}, then none -- first order.\n  \
         Replica symmetry is an approximation near alpha_c and every N is finite; the figures above\n  \
         are the theory's, not measurements of a machine.",
        below.m
    );
}
