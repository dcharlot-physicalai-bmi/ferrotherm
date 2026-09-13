#![allow(missing_docs)]
// HOW FAR IS SOKAL'S tau_int FROM THE EXACT ONE -- the instrument behind every ESS in this crate.
//
// Every "joules per independent sample", every "flips per independent sample", every
// `Finding::Undermixed`, and both tables in `informed.rs` divide by an effective sample size that
// comes from `certify::tau_int`: Sokal's automatic windowing over an empirical autocorrelation,
// truncated at the first window W >= 5 tau. That estimator has a known bias -- it truncates a tail
// that is positive, and on a trace not much longer than tau it sees the slow mode only partly --
// and nothing here has ever measured the bias against a value that does not itself come from a
// trace.
//
// THE ORACLE. On a model small enough to enumerate, the chromatic Gibbs sweep is an explicit
// linear operator P on the 2^n states: a product over colour classes of independent single-site
// heat-bath kernels, in the order `gibbs::Sampler::sweep` applies them. The stationary
// autocovariance of the energy at lag k is then EXACTLY
//
//     C(k) = sum_x pi(x) e(x) (P^k e)(x),     e = E - <E>_pi,
//
// and tau_exact = 1/2 + sum_{k>=1} C(k)/C(0), summed until the tail is below floating point. No
// sampling, no windowing, no trace: linear algebra on pi and P. Then the SAME kernel is run as a
// chain, its energy trace handed to `certify::tau_int`, and the ratio tau_sokal / tau_exact
// reported as a function of the trace length in units of tau. The question with an answer:
//
//     at how many tau of trace does Sokal's estimate land within 10% of the truth, and on which
//     side of it does it sit before that?
//
// If the ratio sits well below 1 at the lengths this crate's examples actually run, every ESS
// they print is too large by that factor, and every joules-per-independent-sample too small.
//
// Count-based throughout; valid on a busy machine.
//
// run: cargo run --release --example tau_exactness

use ferrotherm::autocorr::{tau_int_fundamental, Kernel};
use ferrotherm::certify::tau_int;
use ferrotherm::fft::autocovariance;
use ferrotherm::gibbs::Sampler;
use ferrotherm::graph::{Graph, GraphBuilder};
use ferrotherm::rng::Pcg;

/// A w x h open grid with random +-1 couplings and small random fields: frustrated, bipartite,
/// and small enough that 2^n states are cheap.
fn grid_glass(w: usize, h: usize, seed: u64) -> Graph {
    let mut rng = Pcg::new(seed, 0x6A);
    let n = w * h;
    let mut b = GraphBuilder::new(n);
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            if x + 1 < w {
                b.couple(i, i + 1, if rng.f64() < 0.5 { -1.0 } else { 1.0 });
            }
            if y + 1 < h {
                b.couple(i, i + w, if rng.f64() < 0.5 { -1.0 } else { 1.0 });
            }
        }
    }
    for i in 0..n {
        b.bias(i, (rng.f64() - 0.5) * 0.4);
    }
    b.build()
}

/// Exact `tau_int` of the energy under the chromatic Gibbs sweep, by the fundamental matrix:
/// one dense solve, no lags, no truncation, at any temperature (`autocorr::tau_int_fundamental`).
/// The first version of this example carried its own lag-summing operator; at `beta = 2` that sum
/// needs some `5e7` lags of a 4,096-state operator and never returned.
fn tau_exact(g: &Graph, beta: f64) -> f64 {
    tau_int_fundamental(g, beta, Kernel::ChromaticGibbs, |s| g.energy(s)).expect("12 spins, energy varies").tau_int
}

/// This keeps summing past Sokal's window for as long as the empirical autocorrelation is still
/// resolvably positive: above twice its own noise, `sqrt((1 + 2 tau(k)) / L)` (Bartlett's large-lag
/// standard error for a stationary sequence), and above a floor of 0.01. A single-mode chain is
/// left exactly where Sokal left it, because at `W = 5 tau` its `rho` is `e^-5`, under both.
fn tau_extended(trace: &[f64]) -> f64 {
    let n = trace.len();
    if n < 16 {
        return f64::NAN;
    }
    let mean = trace.iter().sum::<f64>() / n as f64;
    let var = trace.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
    if var <= 0.0 {
        return f64::INFINITY;
    }
    let max_lag = (n / 4).max(1);
    // Every lag at once: this sum runs far past Sokal's window on a long trace, where lag by lag
    // is O(L) each and the run of 2026-09-13 never finished.
    let cov = autocovariance(trace, max_lag);
    let mut tau = 0.5f64;
    let mut closed = false;
    for k in 1..=max_lag {
        let rho = cov[k] / var;
        if !closed && (k as f64) >= 5.0 * tau.max(0.5) {
            closed = true;
        }
        if closed {
            let noise = ((1.0 + 2.0 * tau) / n as f64).sqrt();
            if rho < (2.0 * noise).max(0.01) {
                break;
            }
        }
        tau += rho;
    }
    tau.max(0.5)
}

/// A third opinion with no window at all: batch means. With `b` batches of length `L/b`,
/// `Var(batch mean) ~ Var(x) * 2 tau / (L/b)`, so `tau = (L/b) * Var(batch means) / (2 Var(x))`.
/// Unbiased only when the batch is much longer than the slow mode, so it reads LOW on a short
/// trace -- which is why it is a check and not the estimator.
fn tau_batch(trace: &[f64], b: usize) -> f64 {
    let n = trace.len();
    let len = n / b;
    if len < 2 {
        return f64::NAN;
    }
    let mean = trace.iter().sum::<f64>() / n as f64;
    let var = trace.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
    if var <= 0.0 {
        return f64::INFINITY;
    }
    let means: Vec<f64> =
        (0..b).map(|i| trace[i * len..(i + 1) * len].iter().sum::<f64>() / len as f64).collect();
    let vb = means.iter().map(|m| (m - mean).powi(2)).sum::<f64>() / (b as f64 - 1.0);
    (len as f64 * vb / (2.0 * var)).max(0.5)
}

struct Estimates {
    sokal: (f64, f64),
    extended: (f64, f64),
    batch: (f64, f64),
}

fn mean_sd(v: &[f64]) -> (f64, f64) {
    let m = v.iter().sum::<f64>() / v.len() as f64;
    let sd = (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64).sqrt();
    (m, sd)
}

/// All three estimators on `reps` independent traces of `len` sweeps from the same kernel, after
/// a burn-in of `burn` sweeps. Means and standard deviations over the reps.
fn tau_sampled(g: &Graph, beta: f64, len: usize, burn: usize, reps: u64) -> Estimates {
    let (mut s_v, mut e_v, mut b_v) = (Vec::new(), Vec::new(), Vec::new());
    for r in 0..reps {
        let mut s = Sampler::new(g, beta, 1000 + r);
        s.sweeps(burn, None);
        let mut trace = Vec::with_capacity(len);
        for _ in 0..len {
            s.sweep(None);
            trace.push(g.energy(&s.s));
        }
        s_v.push(tau_int(&trace));
        e_v.push(tau_extended(&trace));
        b_v.push(tau_batch(&trace, 20));
    }
    Estimates { sokal: mean_sd(&s_v), extended: mean_sd(&e_v), batch: mean_sd(&b_v) }
}

fn main() {
    let (w, h) = (4usize, 3usize);
    let g = grid_glass(w, h, 7);
    let n = g.n;
    println!("SOKAL'S tau_int AGAINST THE EXACT tau_int OF THE SAME KERNEL\n");
    println!(
        "  model    {w}x{h} open grid, {n} spins, random +-1 couplings, fields in [-0.2, 0.2]; {} colour classes",
        g.classes.len()
    );
    println!("  oracle   autocorr::tau_int_fundamental on all 2^{n} states: one dense solve per temperature, no lags");
    println!("  chain    gibbs::Sampler, the same sweep, traces of L sweeps after a burn-in of 20 tau, 16 reps\n");
    println!("  estimators   S = certify::tau_int (Sokal, window at 5 tau); X = the same sum extended past the");
    println!("               window while rho stays above twice its noise; B = batch means, 20 batches.");
    println!("               Each cell is estimate / exact, mean over 16 reps, with sd/mean in brackets.\n");
    let mults = [30.0f64, 100.0, 1_000.0, 10_000.0];
    // The longest trace this run will draw, per rep: 2^26 sweeps (a 2^27-point transform, 2 GB).
    // Cells past it print '-' rather than being quietly skipped.
    let cap = 1usize << 26;
    println!("  cap      traces above {cap} sweeps are not drawn and print '-'\n");
    println!(
        "  {:>5} {:>12}   {}",
        "beta",
        "tau exact",
        mults.iter().map(|m| format!("{:>26}", format!("L = {m:.0} tau: S / X / B"))).collect::<Vec<_>>().join("  ")
    );
    let mut worst: Vec<(f64, &str, f64)> = Vec::new(); // (mult, estimator, worst |ratio - 1|)
    for &beta in &[0.5f64, 1.0, 1.5, 2.0] {
        let te = tau_exact(&g, beta);
        let mut cells = Vec::new();
        for &mlt in &mults {
            let len = ((mlt * te).ceil() as usize).max(64);
            if len > cap {
                cells.push(format!("{:>8} {:>8} {:>8}", "-", "-", "-"));
                continue;
            }
            let burn = ((20.0 * te).ceil() as usize).max(16);
            let e = tau_sampled(&g, beta, len, burn, 16);
            let f = |(m, sd): (f64, f64)| format!("{:.2}({:.0}%)", m / te, 100.0 * sd / m.max(1e-300));
            cells.push(format!("{:>8} {:>8} {:>8}", f(e.sokal), f(e.extended), f(e.batch)));
            for (name, (m, _)) in [("S", e.sokal), ("X", e.extended), ("B", e.batch)] {
                worst.push((mlt, name, (m / te - 1.0).abs()));
            }
        }
        println!("  {beta:>5.2} {te:>12.2}   {}", cells.join("  "));
    }
    println!("\n  WHAT THE TABLE SAYS.\n");
    println!("  A cell below 1 UNDERSTATES the true autocorrelation time at that trace length, so every ESS");
    println!("  derived from it is too large by the inverse and every cost per independent sample too small.");
    println!("  The exact column is linear algebra on the transition operator and pi: no sampling error, no");
    println!("  window. An S cell that stays flat as L grows is not a short-trace bias -- it is the window");
    println!("  closing on a fast mode and never summing a slow one, and no trace length repairs it.");
    println!("\n  Worst |estimate/exact - 1| over the five temperatures:");
    for &mlt in &mults {
        let row: Vec<String> = ["S", "X", "B"]
            .iter()
            .map(|name| {
                let w = worst
                    .iter()
                    .filter(|(m, n, _)| *m == mlt && n == name)
                    .map(|(_, _, r)| *r)
                    .fold(0.0f64, f64::max);
                format!("{name} {:>4.0}%", 100.0 * w)
            })
            .collect();
        println!("    L = {mlt:>5.0} tau:   {}", row.join("   "));
    }
    println!("\n  This crate's examples typically hold 4,000 draws; at a tau of 100 draws that is L = 40 tau.");
    println!("  Read the column nearest that for what those examples' ESS figures were worth.");
}
