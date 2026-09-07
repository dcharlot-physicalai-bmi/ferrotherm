//! Critical slowing down, measured: single-spin Gibbs against Swendsen–Wang and Wolff.
//!
//! `cluster.rs` claims a cluster update fixes the thing a single-spin sampler is worst at. This
//! measures it instead of citing it.
//!
//! At the critical point of the 2D Ising model, `beta_c = ln(1 + sqrt 2) / 2 = 0.44069`, the
//! correlation length diverges with the lattice, and a single-spin sampler has to move a domain of
//! linear size `L` one spin at a time against the surface tension holding it together. The
//! integrated autocorrelation time of the magnetisation then grows as `L^z`. Away from `beta_c`
//! there is nothing to see: correlations are bounded by a fixed length and every method looks fine.
//! That is why this runs AT the critical point, and why `examples/onsager.rs` — which steps around
//! it — cannot answer this question.
//!
//! Two costs are reported, because they are different questions:
//!
//!   - **per sweep**, which is what the literature quotes and what `z` is defined against;
//!   - **per spin visited**, from the ledger, which is what the machine pays.
//!
//! The second exists because a sweep is a convention and the two moves do not mean the same thing
//! by it. A Wolff sweep here is one cluster flip, which on a large lattice at `beta_c` touches a
//! large fraction of the spins — so counting sweeps would flatter it, and counting visits will not.
//! This module has already been wrong once in the other direction: an earlier `wolff_sweep` derived
//! its step count from the cluster sizes it built, which made the number of steps a function of the
//! state and biased the sample. Measurement units are not a presentation choice here.
//!
//! Run with `cargo run --release --example critical_slowdown`.

use ferrotherm::certify::tau_int;
use ferrotherm::cluster::{Sampler, Update};
use ferrotherm::graph::Graph;
use ferrotherm::ledger::Ledger;

/// `ln(1 + sqrt 2) / 2`, where the 2D Ising model orders.
const BETA_C: f64 = 0.440_686_793_509_771_5;

/// Absolute magnetisation per spin, the observable whose autocorrelation defines `z`.
fn abs_m(s: &[i8]) -> f64 {
    (s.iter().map(|&x| f64::from(x)).sum::<f64>() / s.len() as f64).abs()
}

/// What one method did on one lattice.
struct Run {
    /// Integrated autocorrelation time in sweeps, by Sokal's automatic windowing.
    tau_sweeps: f64,
    /// The same, in units of spins visited, which is what the hardware is billed for.
    tau_visits: f64,
    /// Mean `|m|`, so a method that mixes fast and samples the wrong thing is visible.
    m: f64,
}

fn measure_gibbs(g: &Graph, draws: usize, burn: usize) -> Run {
    let mut s = ferrotherm::gibbs::Sampler::new(g, BETA_C, 17);
    let mut l = Ledger::default();
    s.sweeps(burn, None);
    let mut trace = Vec::with_capacity(draws);
    for _ in 0..draws {
        s.sweeps(1, Some(&mut l));
        trace.push(abs_m(&s.read_all(None)));
    }
    let tau = tau_int(&trace);
    Run {
        tau_sweeps: tau,
        // A Gibbs sweep visits every spin by construction, so the ledger's per-sweep visit count is
        // n -- taken from the ledger anyway rather than assumed, so the two rows are commensurable.
        tau_visits: tau * (l.samples as f64 / draws as f64),
        m: trace.iter().sum::<f64>() / draws as f64,
    }
}

fn measure_cluster(g: &Graph, update: Update, draws: usize, burn: usize) -> Run {
    let mut c = Sampler::new(g, BETA_C, 17).expect("a ferromagnet is balanced and unbiased");
    let mut l = Ledger::default();
    for _ in 0..burn {
        c.sweep_with(update, None);
    }
    let mut trace = Vec::with_capacity(draws);
    for _ in 0..draws {
        c.sweep_with(update, Some(&mut l));
        trace.push(abs_m(&c.state()));
    }
    let tau = tau_int(&trace);
    Run {
        tau_sweeps: tau,
        tau_visits: tau * (l.samples as f64 / draws as f64),
        m: trace.iter().sum::<f64>() / draws as f64,
    }
}

/// Least-squares slope of `ln tau` against `ln L`, which is the dynamic exponent `z`.
fn exponent(sizes: &[usize], taus: &[f64]) -> f64 {
    let n = sizes.len() as f64;
    let x: Vec<f64> = sizes.iter().map(|&l| (l as f64).ln()).collect();
    let y: Vec<f64> = taus.iter().map(|t| t.ln()).collect();
    let (mx, my) = (x.iter().sum::<f64>() / n, y.iter().sum::<f64>() / n);
    let num: f64 = x.iter().zip(&y).map(|(a, b)| (a - mx) * (b - my)).sum();
    let den: f64 = x.iter().map(|a| (a - mx).powi(2)).sum();
    num / den
}

fn main() {
    let sizes = [8usize, 12, 16, 24, 32];
    let draws = 40_000;
    let burn = 4_000;

    println!("critical slowing down at beta_c = {BETA_C:.6}");
    println!("tau_int of |m| by Sokal windowing, {draws} draws after {burn} burn-in sweeps\n");
    println!(
        "{:>4}  {:>26}  {:>26}  {:>26}",
        "L", "single-spin Gibbs", "Swendsen-Wang", "Wolff (1 cluster/sweep)"
    );
    println!(
        "{:>4}  {:>10} {:>9} {:>5}  {:>10} {:>9} {:>5}  {:>10} {:>9} {:>5}",
        "", "tau/sweep", "tau/visit", "<|m|>", "tau/sweep", "tau/visit", "<|m|>", "tau/sweep",
        "tau/visit", "<|m|>"
    );

    let mut taus: [Vec<f64>; 3] = [vec![], vec![], vec![]];
    let mut visits: [Vec<f64>; 3] = [vec![], vec![], vec![]];
    for &l in &sizes {
        let g = ferrotherm::ising::lattice2d(l, 1.0);
        let runs = [
            measure_gibbs(&g, draws, burn),
            measure_cluster(&g, Update::SwendsenWang, draws, burn),
            measure_cluster(&g, Update::Wolff, draws, burn),
        ];
        print!("{l:>4}  ");
        for (k, r) in runs.iter().enumerate() {
            print!("{:>10.2} {:>9.0} {:>5.3}  ", r.tau_sweeps, r.tau_visits, r.m);
            taus[k].push(r.tau_sweeps);
            visits[k].push(r.tau_visits);
        }
        println!();
    }

    println!();
    for (k, name) in ["single-spin Gibbs", "Swendsen-Wang", "Wolff"].iter().enumerate() {
        println!(
            "{name:<18} z = {:.2} per sweep,  {:.2} per spin visited",
            exponent(&sizes, &taus[k]),
            exponent(&sizes, &visits[k])
        );
    }
    println!(
        "\nThe literature value for single-spin dynamics in 2D is z ~ 2.17, and ~0.25 for cluster\n\
         updates. The per-visit column is the one that says what the hardware pays: a cluster sweep\n\
         costs more than a Gibbs sweep, and the exponent is what decides whether that is worth it."
    );
}
