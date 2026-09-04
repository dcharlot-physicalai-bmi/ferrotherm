//! Free energy, certified: what a sampler owes, and the bound it can prove it paid.
//!
//! Every other certificate in this crate says a chain mixed. This one says what the distribution
//! IS: `ln Z(β)`, the number that turns sampled statistics into normalised probabilities and an
//! energy-based model's marginals into a likelihood. Three estimators, each carrying the guarantee
//! it actually has, checked against exact oracles at every size the oracles reach:
//!
//!   * annealed importance sampling -- an UNCONDITIONAL high-probability lower bound (Markov on an
//!     unbiased estimator; mixing changes the variance, never the expectation);
//!   * reverse AIS -- the mirror upper bound, conditional on starting from the target;
//!   * thermodynamic integration -- a two-sided bracket from one monotonicity fact,
//!     d<E>/dβ = -Var(E) <= 0, widened by each rung's own error bar.
//!
//! Counts, not durations: sweeps and runs are the cost, and they are the same on every machine.
//!
//! usage: cargo run --release --example free_energy

use ferrotherm::exact::Elimination;
use ferrotherm::free_energy::{
    ais, linear_ladder, onsager_log_z_density, reverse_ais, ring_log_z, thermodynamic_integration, Sandwich,
};
use ferrotherm::gibbs::Sampler;
use ferrotherm::ising;
use ferrotherm::popanneal::{run as popanneal, Params};

fn main() {
    // ---- 1. a chain with a closed form: every estimator against the transfer matrix -----------
    let (n, j, h, beta) = (16usize, 1.0, 0.15, 0.9);
    let g = ising::ring(n, j, h);
    let truth = ring_log_z(n, j, h, beta);
    let ladder = linear_ladder(beta, 64);
    let fwd = ais(&g, &ladder, 2, 128, 1);
    // Starting states for the reverse run: a long chain at the target, thinned. The upper bound is
    // as good as these are equilibrated, which is why the chain's length is printed beside it.
    let starts: Vec<Vec<i8>> = {
        let mut s = Sampler::new(&g, beta, 11);
        s.sweeps(2_000, None);
        (0..128)
            .map(|_| {
                s.sweeps(50, None);
                s.read_all(None)
            })
            .collect()
    };
    let rev = reverse_ais(&g, &ladder, 2, &starts, 2);
    let sw = Sandwich::new(&fwd, &rev, 0.01);
    let ti = thermodynamic_integration(&g, &linear_ladder(beta, 32), 200, 2_000, 3.0, 3);
    let pa = popanneal(&g, &Params::linear_from_zero(512, 2, beta, 64), 4);

    println!("--- ring n={n} J={j} h={h} beta={beta}: ln Z, exact by transfer matrix = {truth:.4}");
    println!("{:<28} {:>10} {:>22} {:>8}", "estimator", "ln Z", "bound / bracket", "cost");
    println!(
        "{:<28} {:>10.4} {:>22} {:>8}",
        "AIS (64 rungs x 2 sweeps)", fwd.log_z,
        format!(">= {:.3} @ 99%", fwd.lower_bound(0.01)), format!("{} runs", fwd.log_weights.len())
    );
    println!(
        "{:<28} {:>10.4} {:>22} {:>8}",
        "reverse AIS (2000+50/draw)", rev.log_z,
        format!("<= {:.3} @ 99%*", rev.upper_bound(0.01)), format!("{} runs", rev.log_weights.len())
    );
    println!(
        "{:<28} {:>10.4} {:>22} {:>8}",
        "sandwich", 0.5 * (sw.lower + sw.upper),
        format!("[{:.3}, {:.3}] @ 98%", sw.lower, sw.upper), ""
    );
    println!(
        "{:<28} {:>10.4} {:>22} {:>8}",
        "TI (32 rungs x 2000 draws)", ti.midpoint(),
        format!("[{:.3}, {:.3}] @ 3se", ti.lower_widened, ti.upper_widened), "64k sw"
    );
    println!(
        "{:<28} {:>10.4} {:>22} {:>8}",
        "population annealing", pa.ln_z, "(no bound: SMC point est.)", "512 x 64"
    );
    println!(
        "  weight ess: forward {:.1} of {}, reverse {:.1} of {}   (* conditional on the starts)",
        fwd.ess, fwd.log_weights.len(), rev.ess, rev.log_weights.len()
    );
    println!("  truth inside the sandwich: {}   inside the TI bracket: {}\n",
        sw.contains(truth), ti.lower_widened <= truth && truth <= ti.upper_widened);

    // ---- 1b. the same curve for free, out of an optimisation run ------------------------------
    //
    // Parallel tempering already holds a chain at every rung of a ladder; until the observer was
    // added, free_energy drew its own and the optimiser's samples were discarded. Now one run
    // answers both questions, and the recording is bit-identical to the unobserved loop.
    {
        use ferrotherm::tempering::parallel_tempering_observed;
        let ladder: Vec<f64> = (0..24).map(|k| beta * k as f64 / 23.0).collect();
        let (res, traces) = parallel_tempering_observed(&g, &ladder, 3000, 2, 300, 5, None);
        let th = traces.thermodynamics(n, 3.0).unwrap();
        println!("--- the same ln Z out of a parallel-tempering run that was optimising anyway");
        println!("  best energy found          {:.4}", res.best_e);
        println!("  ln Z at the top rung       {:.4} +- {:.4}   (transfer matrix {truth:.4})", th.top().log_z, th.top().stderr);
        println!("  entropy there              {:.4} nats", th.top().entropy);
        println!("  heat capacity there        {:.4}", th.top().heat_capacity);
        println!("  the traces cost one energy evaluation per replica per round, which the");
        println!("  best-tracking loop was already paying.\n");
    }

    // ---- 2. past enumeration: a 6x6 torus against exact elimination ----------------------------
    let g = ising::lattice2d(6, 1.0);
    let beta = 0.4;
    let elim = Elimination::default().log_partition(&g, beta).unwrap().log_z.unwrap();
    let a = ais(&g, &linear_ladder(beta, 64), 2, 128, 7);
    println!("--- 6x6 torus at beta={beta} (36 spins: enumeration refuses, elimination does not)");
    println!("  elimination (exact)   {elim:.4}");
    println!("  AIS                   {:.4}   >= {:.3} @ 99%, ess {:.1}\n", a.log_z, a.lower_bound(0.01), a.ess);

    // ---- 3. the thermodynamic limit: a torus against Onsager ---------------------------------
    // lattice2d is periodic, so finite-size corrections to the infinite-lattice density are
    // exponentially small away from criticality (beta_c = 0.4407); at beta = 0.3 a 12x12 torus is
    // already within the estimator's own error of the closed form.
    let (l, beta) = (12usize, 0.3);
    let g = ising::lattice2d(l, 1.0);
    let onsager = onsager_log_z_density(beta, 1.0, 512);
    let a = ais(&g, &linear_ladder(beta, 64), 2, 128, 9);
    let ti = thermodynamic_integration(&g, &linear_ladder(beta, 32), 200, 2_000, 3.0, 10);
    let nn = (l * l) as f64;
    println!("--- {l}x{l} torus at beta={beta}: ln Z per spin against Onsager's closed form");
    println!("  Onsager (infinite lattice)  {onsager:.5}");
    println!("  AIS / N                     {:.5}   (>= {:.5} @ 99%)", a.log_z / nn, a.lower_bound(0.01) / nn);
    println!("  TI bracket / N              [{:.5}, {:.5}]", ti.lower_widened / nn, ti.upper_widened / nn);
    println!("  |AIS/N - Onsager| = {:.2e}; the residual is finite size plus estimator noise, and\n  \
              the bound is a bound on the TORUS's ln Z, which Onsager only approximates.",
        (a.log_z / nn - onsager).abs());
}
