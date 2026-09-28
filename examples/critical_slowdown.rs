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
//! THE ESTIMATOR AND THE LENGTH, and why both changed on 2026-09-28. Until then this ran ONE seed
//! (17) at 40,000 draws and read `tau` with Sokal's window. At `L = 32` under Gibbs that is about
//! 300 autocorrelation times, and across 24 seeds at that length the window read a median of 122.8
//! with a standard deviation of 45: the published 100.85 was seed 17, at about the 17th
//! percentile. So now every `tau` is [`ferrotherm::certify::tau_estimate`] -- Geyer's initial
//! monotone sequence with long-batch means beside it, the larger carried -- over five seeds at
//! `200 L^2` draws, which is at least 1,000 autocorrelation times at every size here, and every
//! cell prints the median with the range over seeds and its own `draws / tau`. Geyer's value alone
//! and Sokal's window are printed beside the carried one, so the change of estimator is visible
//! in the output rather than asserted in a comment. `z` is fitted per seed as well as to the
//! medians, because a single seed's `z` is what was published and the spread is what says whether
//! a second decimal is there to be quoted.
//!
//! Run with `cargo run --release --example critical_slowdown`.

use ferrotherm::certify::{tau_estimate, tau_int_sokal};
use ferrotherm::cluster::{Sampler, Update};
use ferrotherm::graph::Graph;
use ferrotherm::ledger::Ledger;

/// `ln(1 + sqrt 2) / 2`, where the 2D Ising model orders.
const BETA_C: f64 = 0.440_686_793_509_771_5;

/// Seeds, the first of them the one the single-seed table was published from.
const SEEDS: [u64; 5] = [17, 18, 19, 20, 21];

/// Lattice sizes.
const SIZES: [usize; 5] = [8, 12, 16, 24, 32];

/// Burn-in sweeps before recording.
const BURN: usize = 4_000;

/// Draws at size `l`: `200 L^2`, and never fewer than the 40,000 the first table used. At
/// `z` near 2.1 and `tau(8)` near 6.5 sweeps that is at least 1,000 `tau` at every size.
fn draws(l: usize) -> usize {
    (200 * l * l).max(40_000)
}

/// Absolute magnetisation per spin, the observable whose autocorrelation defines `z`.
fn abs_m(s: &[i8]) -> f64 {
    (s.iter().map(|&x| f64::from(x)).sum::<f64>() / s.len() as f64).abs()
}

/// What one method did on one lattice with one seed.
#[derive(Clone, Copy)]
struct Run {
    /// Integrated autocorrelation time in sweeps, as `certify::tau_estimate` carries it.
    tau: f64,
    /// Geyer's initial monotone sequence alone.
    geyer: f64,
    /// Sokal's window, what this example measured with until 2026-09-28.
    sokal: f64,
    /// Spins visited per sweep, from the ledger: what converts `tau` to what the hardware pays.
    visits: f64,
    /// Mean `|m|`, so a method that mixes fast and samples the wrong thing is visible.
    m: f64,
    /// Draws recorded.
    draws: usize,
}

fn finish(trace: &[f64], ledger: &Ledger) -> Run {
    let est = tau_estimate(&[trace]);
    Run {
        tau: est.tau,
        geyer: est.geyer,
        sokal: tau_int_sokal(trace),
        visits: ledger.samples as f64 / trace.len() as f64,
        m: trace.iter().sum::<f64>() / trace.len() as f64,
        draws: trace.len(),
    }
}

fn measure_gibbs(g: &Graph, draws: usize, seed: u64) -> Run {
    let mut s = ferrotherm::gibbs::Sampler::new(g, BETA_C, seed);
    let mut l = Ledger::default();
    s.sweeps(BURN, None);
    let mut trace = Vec::with_capacity(draws);
    for _ in 0..draws {
        s.sweeps(1, Some(&mut l));
        trace.push(abs_m(&s.read_all(None)));
    }
    // A Gibbs sweep visits every spin by construction, so the ledger's per-sweep visit count is
    // n -- taken from the ledger anyway rather than assumed, so the rows are commensurable.
    finish(&trace, &l)
}

fn measure_cluster(g: &Graph, update: Update, draws: usize, seed: u64) -> Run {
    let mut c = Sampler::new(g, BETA_C, seed).expect("a ferromagnet is balanced and unbiased");
    let mut l = Ledger::default();
    for _ in 0..BURN {
        c.sweep_with(update, None);
    }
    let mut trace = Vec::with_capacity(draws);
    for _ in 0..draws {
        c.sweep_with(update, Some(&mut l));
        trace.push(abs_m(&c.state()));
    }
    finish(&trace, &l)
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

/// Median, minimum and maximum.
fn spread(v: &[f64]) -> (f64, f64, f64) {
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    let k = s.len();
    let med = if k % 2 == 1 { s[k / 2] } else { 0.5 * (s[k / 2 - 1] + s[k / 2]) };
    (med, s[0], s[k - 1])
}

fn main() {
    const NAMES: [&str; 3] = ["single-spin Gibbs", "Swendsen-Wang", "Wolff (1 cluster/sweep)"];
    println!("critical slowing down at beta_c = {BETA_C:.6}");
    println!(
        "tau_int of |m| by certify::tau_estimate (Geyer's sequence, long-batch means beside it, the\n\
         larger carried), {} seeds, 200 L^2 draws (at least 40,000) after {BURN} burn-in sweeps\n",
        SEEDS.len()
    );

    // runs[method][size][seed], every seed on its own thread: the result does not depend on it.
    let mut runs: Vec<Vec<Vec<Run>>> = vec![vec![Vec::new(); SIZES.len()]; 3];
    for (si, &l) in SIZES.iter().enumerate() {
        let g = ferrotherm::ising::lattice2d(l, 1.0);
        let n = draws(l);
        let per_seed: Vec<[Run; 3]> = std::thread::scope(|scope| {
            let handles: Vec<_> = SEEDS
                .iter()
                .map(|&seed| {
                    let g = &g;
                    scope.spawn(move || {
                        [
                            measure_gibbs(g, n, seed),
                            measure_cluster(g, Update::SwendsenWang, n, seed),
                            measure_cluster(g, Update::Wolff, n, seed),
                        ]
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().expect("a seed's run panicked")).collect()
        });
        for r in per_seed {
            for k in 0..3 {
                runs[k][si].push(r[k]);
            }
        }
    }

    // Medians of the carried tau, per sweep and per spin visited, for the fits and the ratios.
    let mut med_sweep = [[0.0f64; SIZES.len()]; 3];
    let mut med_visit = [[0.0f64; SIZES.len()]; 3];
    for (k, name) in NAMES.iter().enumerate() {
        println!("{name}");
        println!(
            "{:>4} {:>8}  {:>26}  {:>8}  {:>8}  {:>9}  {:>10}  {:>6}",
            "L", "draws", "tau/sweep median [range]", "Geyer", "Sokal", "draws/tau", "tau/visit", "<|m|>"
        );
        for (si, &l) in SIZES.iter().enumerate() {
            let rs = &runs[k][si];
            let taus: Vec<f64> = rs.iter().map(|r| r.tau).collect();
            let (tm, tlo, thi) = spread(&taus);
            let (gm, _, _) = spread(&rs.iter().map(|r| r.geyer).collect::<Vec<_>>());
            let (sm, _, _) = spread(&rs.iter().map(|r| r.sokal).collect::<Vec<_>>());
            let (vm, _, _) = spread(&rs.iter().map(|r| r.tau * r.visits).collect::<Vec<_>>());
            let (mm, _, _) = spread(&rs.iter().map(|r| r.m).collect::<Vec<_>>());
            let min_ratio = rs.iter().map(|r| r.draws as f64 / r.tau).fold(f64::INFINITY, f64::min);
            med_sweep[k][si] = tm;
            med_visit[k][si] = vm;
            println!(
                "{l:>4} {:>8}  {:>8.2} [{tlo:>7.2}..{thi:>7.2}]  {gm:>8.2}  {sm:>8.2}  {min_ratio:>9.0}  {vm:>10.0}  {mm:>6.3}",
                rs[0].draws, tm
            );
        }
        let per_seed_z: Vec<f64> = (0..SEEDS.len())
            .map(|j| exponent(&SIZES, &(0..SIZES.len()).map(|si| runs[k][si][j].tau).collect::<Vec<_>>()))
            .collect();
        let (zm, zlo, zhi) = spread(&per_seed_z);
        println!(
            "     z = {:.2} per sweep from the medians ({zm:.2} median of per-seed fits, range {zlo:.2}..{zhi:.2}),  {:.2} per spin visited\n",
            exponent(&SIZES, &med_sweep[k]),
            exponent(&SIZES, &med_visit[k])
        );
    }

    let last = SIZES.len() - 1;
    let ratio = |si: usize| med_visit[0][si] / med_visit[2][si];
    println!(
        "At L = {} an independent sample costs {:.0} spin visits under Gibbs and {:.0} under Wolff\n\
         (medians over seeds): {:.0}x, widening as L^{:.2} over L = {}..{}.",
        SIZES[last],
        med_visit[0][last],
        med_visit[2][last],
        ratio(last),
        exponent(&SIZES, &(0..SIZES.len()).map(ratio).collect::<Vec<_>>()),
        SIZES[0],
        SIZES[last]
    );
    println!(
        "\nThe literature value for single-spin dynamics in 2D is z ~ 2.17, and ~0.25 for cluster\n\
         updates. The per-visit column is the one that says what the hardware pays: a cluster sweep\n\
         costs more than a Gibbs sweep, and the exponent is what decides whether that is worth it."
    );
}
