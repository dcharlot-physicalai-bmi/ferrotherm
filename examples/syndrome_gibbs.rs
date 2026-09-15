//! An LDPC code decoded three ways on the same syndromes -- the exact hard posterior, belief
//! propagation, and the relaxed C-body model a Gibbs sampler can hold -- with the relaxed
//! chain's autocorrelation time beside its bit error, across the check strength.
//!
//! ```text
//!   cargo run --release --example syndrome_gibbs
//! ```
//!
//! Counts over sampled noise patterns; each pattern's decoders are exact enumerations and dense
//! solves, so the channel is the only randomness.

use ferrotherm::pdit::{stationary_of, tau_int_of};
use ferrotherm::syndrome::{
    bp_marginals, bsc, decisions, hard_marginals, relaxed_graph, relaxed_marginals,
    relaxed_sweep_kernel, Code,
};

fn errors(decoded: &[u8], noise: &[u8]) -> f64 {
    decoded.iter().zip(noise).filter(|(a, b)| a != b).count() as f64 / noise.len() as f64
}

fn main() {
    let code = Code::regular(10, 3, 5, 11).expect("a (3,5) code on 10 bits");
    println!(
        "(3,5)-regular LDPC code, n = {}, m = {}, rate {:.3}, Tanner graph {}",
        code.n(),
        code.m(),
        code.rate(),
        if code.tanner_is_forest() {
            "a forest"
        } else {
            "cyclic"
        }
    );
    let patterns = 30u64;
    for &p in &[0.05f64, 0.1] {
        println!("-- channel p = {p} --");
        let (mut hard_ber, mut bp_ber) = (0.0, 0.0);
        let mut noises = Vec::new();
        for k in 0..patterns {
            let noise = bsc(&vec![0u8; code.n()], p, 500 + k);
            let syndrome = code.syndrome(&noise).expect("length");
            let hard = hard_marginals(&code, &syndrome, p).expect("small");
            hard_ber += errors(&decisions(&hard), &noise);
            let bp = bp_marginals(&code, &syndrome, p, 40);
            bp_ber += errors(&decisions(&bp), &noise);
            noises.push((noise, syndrome));
        }
        println!(
            "  hard posterior: bit error {:.4}   belief propagation (40 rounds): {:.4}",
            hard_ber / patterns as f64,
            bp_ber / patterns as f64
        );
        println!(
            "  {:>6} {:>10} {:>12} {:>12}",
            "gamma", "BER relax", "tau_int(sw)", "tau_int(fl)"
        );
        for gamma in [0.5f64, 1.0, 2.0, 4.0, 8.0] {
            let mut relaxed_ber = 0.0;
            let mut tau_sum = 0.0;
            let mut tau_count = 0usize;
            for (k, (noise, syndrome)) in noises.iter().enumerate() {
                let marg = relaxed_marginals(&code, syndrome, p, gamma).expect("small");
                relaxed_ber += errors(&decisions(&marg), noise);
                // The chain's autocorrelation time on every pattern: a 1024-state solve each.
                if k < patterns as usize {
                    let g = relaxed_graph(&code, syndrome, p, gamma);
                    let kernel = relaxed_sweep_kernel(&g).expect("twelve bits");
                    let states = 1usize << code.n();
                    if let Ok(pi) = stationary_of(&kernel, states) {
                        let first: Vec<f64> = (0..states)
                            .map(|x| {
                                if (x >> (code.n() - 1)) & 1 == 1 {
                                    1.0
                                } else {
                                    -1.0
                                }
                            })
                            .collect();
                        if let Ok(tau) = tau_int_of(&kernel, states, &pi, &first) {
                            tau_sum += tau;
                            tau_count += 1;
                        }
                    }
                }
            }
            let tau = if tau_count > 0 {
                tau_sum / tau_count as f64
            } else {
                f64::NAN
            };
            println!(
                "  {gamma:>6.1} {:>10.4} {:>12.2} {:>12.1}",
                relaxed_ber / patterns as f64,
                tau,
                tau * code.n() as f64
            );
        }
    }
    println!();
    println!("tau_int is of the first noise spin, in sweeps (sw) and single-spin flips (fl), averaged over the patterns.");
}
