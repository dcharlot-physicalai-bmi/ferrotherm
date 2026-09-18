// `missing_docs` is denied workspace-wide and is right to be: it guards the API surface, and
// every public item in every library here carries a doc. An EXAMPLE has no API surface -- it is
// a program, and its helpers are private to it -- so the lint has nothing to guard and asks for
// doc comments on `fn main`'s scaffolding instead. Scoped off here rather than weakened there.
#![allow(missing_docs)]
// The diagnostic that cannot report its own failure, and what it costs us.
//
// `floors::cost_per_effective_sample` divides a run's sweeps by `2 tau_int` to get the number of
// independent samples it bought, and `tau_int` comes from ONE chain's own trace. That number is a
// property of the timescales the trace CONTAINS. It is silent about the ones the chain never
// sampled.
//
// So a chain that never leaves the basin it started in fluctuates only inside that basin, its
// autocorrelation decays fast, and `tau_int` comes back SMALL -- which prices the run as though it
// produced MANY independent samples, at a LOW cost per sample. The error flatters the run, on the
// one number this crate exists to report honestly.
//
// This is a textbook pathology and that is exactly why it is used here: below the critical
// temperature a zero-field Ising ferromagnet has `<m> = 0` EXACTLY, by the symmetry between `+s`
// and `-s`, and everyone knows to watch `|m|` instead. The point is not that the observable is
// tricky. The point is what the DIAGNOSTIC says while the answer is wrong.
//
// Two instruments in this crate would both have caught it, and neither is wired to the pricing:
// `rhat` (several chains from dispersed starts, Vehtari et al. 2021) and `cftp` (coupling from the
// past, which returns the burn-in that was actually required rather than the one assumed).
//
// run: cargo run --release --example burnin

use ferrotherm::{certify, cftp, gibbs, ising, rhat};

/// Mean spin of a state.
fn magnetization(s: &[i8]) -> f64 {
    s.iter().map(|&v| f64::from(v)).sum::<f64>() / s.len() as f64
}

/// One chain from a fixed start, returning its magnetization trace after `burn_in` sweeps.
fn trace(g: &ferrotherm::graph::Graph, beta: f64, seed: u64, start: i8, burn_in: usize, draws: usize) -> Vec<f64> {
    let mut s = gibbs::Sampler::new(g, beta, seed);
    s.s = vec![start; g.n];
    s.sweeps(burn_in, None);
    let mut out = Vec::with_capacity(draws);
    for _ in 0..draws {
        s.sweep(None);
        out.push(magnetization(&s.read_all(None)));
    }
    out
}

fn main() {
    let l = 8usize;
    let g = ising::lattice2d(l, 1.0);
    let n = g.n;
    let beta_c = (1.0 + 2.0_f64.sqrt()).ln() / 2.0;
    let beta = 0.60;
    let burn_in = 200usize;
    let draws = 4_000usize;

    println!("A CHAIN THAT CANNOT NOTICE IT HAS NOT CONVERGED");
    println!("{l}x{l} Ising ferromagnet, n = {n}, zero field");
    println!("beta = {beta:.4}, above beta_c = {beta_c:.4} -- the ordered phase");
    println!();
    println!("The exact answer is not estimated here. In zero field every state and its global");
    println!("flip have the same energy, so the Boltzmann distribution is symmetric and");
    println!("  <m> = 0  EXACTLY, at every size and every beta.");
    println!();

    let up = trace(&g, beta, 0xB0_1CE, 1, burn_in, draws);
    let mean = up.iter().sum::<f64>() / up.len() as f64;
    let tau = certify::tau_int(&up);
    let var = up.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / (up.len() - 1) as f64;
    let ess = up.len() as f64 / (2.0 * tau).max(1.0);
    let stderr = (var / ess).sqrt();

    println!("ONE CHAIN, started all-up, {burn_in} sweeps of burn-in, {draws} draws:");
    println!("  <m>                 {mean:+.6} +- {stderr:.6}");
    println!("  tau_int             {tau:.3}          <- small: reads as well mixed");
    println!("  effective samples   {ess:.0}");
    println!("  truth               {:+.6}", 0.0);
    if stderr > 0.0 {
        println!("  the truth sits      {:.0} standard errors outside the interval", (mean - 0.0).abs() / stderr);
    }
    println!();
    println!("Nothing in that block is wrong about the trace. It is a correct summary of a chain");
    println!("that never crossed the barrier, and every number in it is confident.");
    println!();

    // Instrument one: several chains from dispersed starts.
    let chains: Vec<Vec<f64>> = vec![
        trace(&g, beta, 0x1111, 1, burn_in, draws),
        trace(&g, beta, 0x2222, -1, burn_in, draws),
        trace(&g, beta, 0x3333, 1, burn_in, draws),
        trace(&g, beta, 0x4444, -1, burn_in, draws),
    ];
    let r = rhat::split_rhat(&chains);
    println!("INSTRUMENT ONE -- rhat, four chains from dispersed starts (two up, two down):");
    println!("  split R-hat         {r:.3}          <- Vehtari et al. 2021 refuse above 1.01");
    println!("  verdict             {}", if r > 1.01 { "REFUSED -- the chains do not agree" } else { "accepted" });
    println!();

    // Instrument two: coupling from the past, which certifies the burn-in instead of assuming it.
    println!("INSTRUMENT TWO -- cftp, which returns the burn-in actually required:");
    match cftp::Perfect::new(&g, beta) {
        Ok(p) => match p.draw(7, None) {
            Ok(d) => {
                println!("  coalesced_at        {} sweeps", d.coalesced_at);
                println!("  burn-in used        {burn_in} sweeps");
                println!(
                    "  verdict             {}",
                    if (d.coalesced_at as f64) > burn_in as f64 {
                        "REFUSED -- the burn-in was shorter than coalescence required"
                    } else {
                        "adequate"
                    }
                );
            }
            Err(e) => {
                println!("  REFUSED: {e:?}");
                println!("  cftp declines to certify this chain at all. Every other sampler here");
                println!("  returns a confident number for it.");
            }
        },
        Err(e) => println!("  cftp cannot run on this model: {e:?}"),
    }
    println!();
    println!("WHAT IT COSTS US.");
    println!("  tau_int here is {tau:.2} -- not merely unalarming, but close to the ideal 1.0 that");
    println!("  says every draw is independent. The diagnostic returns its BEST score on a chain");
    println!("  that sampled the wrong distribution.");
    println!();
    println!("  `floors::cost_per_effective_sample` used to consume that and nothing else, so it");
    println!("  priced this run at {ess:.0} independent samples. The chain drew none: every draw");
    println!("  came from one basin, which is a conditional distribution, not the target. Joules");
    println!("  per effective sample is not understated here -- it has no denominator. You cannot");
    println!("  price per sample of a distribution you did not sample.");
    println!();
    println!("  cftp doubles its lookback, so a coalescence reported at 2^k means the true");
    println!("  requirement lies in (2^(k-1), 2^k]. That is the burn-in the pricing was assuming");
    println!("  when it was handed {burn_in}.");
    println!();
    println!("  tau_int is not a convergence diagnostic. It summarises the timescales a trace");
    println!("  already contains, and a barrier never crossed contributes none.");
}
