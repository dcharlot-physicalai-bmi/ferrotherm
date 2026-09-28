#![allow(missing_docs)]
// Does weighting the proposal by the energy actually mix faster? Measured, flips against flips.
//
// THE CLAIM UNDER TEST is the one Zanella (JASA 2020) and Grathwohl et al. (ICML 2021) make for
// locally-informed proposals: choosing WHICH move to make by how much it would change the target
// beats choosing uniformly, by enough to pay for the bookkeeping. Every other single-flip sampler
// in this crate picks uniformly or in order, so if the claim holds here it is a gap worth having
// closed, and if it does not, that is worth writing down too.
//
// THE CONTROL THAT MAKES IT A MEASUREMENT. A locally-informed step flips ONE spin; a Gibbs sweep
// updates n. Comparing steps to sweeps would report the ratio n and nothing else. Both are
// therefore counted in FLIPS, and the per-flip cost is the same order UP TO A LOGARITHM: flipping
// k moves the field at its neighbours and nowhere else, so an informed step reweighs O(deg) sites
// and draws the next site from a Fenwick tree in O(log n).
//
// THAT WAS NOT TRUE WHEN THIS TABLE WAS FIRST PRINTED. Until 2026-09-13 the site was drawn by a
// linear scan of all n weights, so every "flip" in the informed columns did O(n) work while this
// comment charged it O(deg) -- on this 256-spin fixture, some seventy reweighs' worth. The claim
// that flips were the same order of work was wrong, and so, it turned out, was the claim this
// comment used to make next: that the tau_int-in-flips numbers were "correct as a statement about
// the chain's law". They were Sokal-window readings of 4,000-draw chains -- at beta = 2 the Gibbs
// arm was about 37 of its own autocorrelation times long, at beta = 4 about 11 -- and the window
// truncates a slow mode by an amount that differs between kernels (on a 3x3 glass at beta = 1 it
// read an informed Barker chain's energy at 0.42 of the exact value and the chromatic sweep's at
// 0.55). So every cold cell was a lower bound, by an unknown and arm-dependent factor, and the
// ratios between arms were ratios of lower bounds. `informed_scaling` measures how the per-flip
// advantage moves with n, which is what decides whether it survives either cost model.
//
// WHAT IT MEASURES NOW (2026-09-28). tau_int is `certify::tau_estimate` -- Geyer's sequence with
// long-batch means beside it, the larger carried -- and every chain is extended, doubling from the
// old 4,000 draws, until it is at least `certify::RESOLVED_TAUS` of its own autocorrelation times
// long or reaches CAP draws. A cell whose chain hit the cap first is printed with `>=`: it is a
// lower bound and says so. The last column group prints the shortest chain, in its own tau, of
// the four seeds behind each cell.
//
// WHERE IT SHOULD HELP AND WHERE IT SHOULD NOT. An informed proposal has something to say only when
// the weights are UNEQUAL. On a flat or high-temperature landscape every flip is about as good as
// every other, the proposal is nearly uniform, and this is Gibbs with extra arithmetic. The sweep
// below is over inverse temperature for that reason: the interesting column is the cold end.

use ferrotherm::certify::{tau_estimate, RESOLVED_TAUS};
use ferrotherm::gibbs::Sampler;
use ferrotherm::graph::{Graph, GraphBuilder};
use ferrotherm::informed::{Balance, Informed};
use ferrotherm::rng::Pcg;

/// Draws every chain starts with: the whole budget of the table this replaced.
const START: usize = 4_000;

/// The most draws any chain is extended to: 2^20, a quarter of a billion flips at n = 256.
const CAP: usize = 1 << 20;

/// A frustrated ring with chords and fields — sparse, and not solvable by inspection.
fn frustrated(n: usize, seed: u64) -> Graph {
    let mut rng = Pcg::new(seed, 0xF5);
    let mut b = GraphBuilder::new(n);
    for i in 0..n {
        b.couple(i, (i + 1) % n, if rng.f64() < 0.5 { -1.0 } else { 1.0 });
    }
    for _ in 0..n / 4 {
        let (i, j) = ((rng.f64() * n as f64) as usize % n, (rng.f64() * n as f64) as usize % n);
        if i != j {
            b.couple(i, j, if rng.f64() < 0.5 { -1.0 } else { 1.0 });
        }
    }
    for i in 0..n {
        b.bias(i, (rng.f64() - 0.5) * 0.4);
    }
    b.build()
}

/// Draw from `step` -- one call, one draw -- doubling the trace from `START` until it is at least
/// `RESOLVED_TAUS` of its own autocorrelation times long or reaches `CAP`. Returns
/// `(tau in draws, draws, resolved)`.
fn resolved(mut step: impl FnMut() -> f64) -> (f64, usize, bool) {
    let mut trace: Vec<f64> = (0..START).map(|_| step()).collect();
    loop {
        let est = tau_estimate(&[&trace]);
        let ok = est.tau.is_finite() && !est.lower_bound();
        if ok || trace.len() >= CAP {
            return (est.tau, trace.len(), ok);
        }
        let more = trace.len();
        trace.extend((0..more).map(|_| step()));
    }
}

/// `tau_int` of the energy trace in units of FLIPS, the draws it took, and whether it resolved.
fn gibbs_tau(g: &Graph, beta: f64, seed: u64) -> (f64, usize, bool) {
    let mut s = Sampler::new(g, beta, seed);
    s.sweeps(START / 10, None); // burn in
    let (tau, draws, ok) = resolved(|| {
        s.sweeps(1, None);
        g.energy(&s.s)
    });
    // tau in sweeps -> tau in flips, since one sweep is n flips.
    (tau * g.n as f64, draws, ok)
}

fn informed_tau(g: &Graph, beta: f64, seed: u64, b: Balance) -> (f64, usize, bool, f64) {
    let mut it = Informed::new(g, beta, seed).with_balance(b);
    it.steps(START * g.n / 10);
    let (tau, draws, ok) = resolved(|| {
        it.steps(g.n); // one draw per n flips, the same opportunity a sweep gets
        it.energy()
    });
    (tau * g.n as f64, draws, ok, it.acceptance())
}

fn main() {
    let n = 256;
    let seeds = 4u64;
    println!(
        "Locally-informed proposals against Gibbs, on a frustrated ring with chords\n\n\
         {n} spins, {seeds} seeds, tau_int in FLIPS (lower is better), mean over seeds. One draw per\n\
         {n} flips on both arms, which is the same order of work. Every chain runs from {START}\n\
         draws, doubling, until it is {RESOLVED_TAUS:.0} of its own tau long (or {CAP} draws: `>=`).\n"
    );
    println!(
        "{:>6} {:>12} {:>12} {:>12} {:>12} {:>10}   {:>28}",
        "beta", "gibbs", "sqrt", "barker", "metropolis", "sqrt acc", "shortest chain, in its tau"
    );

    for &beta in &[0.2f64, 0.5, 1.0, 2.0, 4.0] {
        // Every (seed, arm) chain on its own thread; the result does not depend on it.
        let cells: Vec<[(f64, usize, bool, f64); 4]> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..seeds)
                .map(|seed| {
                    scope.spawn(move || {
                        let g = frustrated(n, seed);
                        let (t, d, ok) = gibbs_tau(&g, beta, seed);
                        let mut out = [(t, d, ok, f64::NAN); 4];
                        for (k, b) in [Balance::Sqrt, Balance::Barker, Balance::Metropolis].iter().enumerate() {
                            out[k + 1] = informed_tau(&g, beta, seed, *b);
                        }
                        out
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().expect("a chain panicked")).collect()
        });
        let mut row = [0.0f64; 4];
        let mut short = [f64::INFINITY; 4];
        let mut all_ok = [true; 4];
        let mut acc = 0.0;
        for c in &cells {
            for k in 0..4 {
                row[k] += c[k].0 / seeds as f64;
                short[k] = short[k].min(c[k].1 as f64 * n as f64 / c[k].0);
                all_ok[k] &= c[k].2;
            }
            acc += c[1].3 / seeds as f64;
        }
        let cell = |k: usize| {
            if all_ok[k] { format!("{:.0}", row[k]) } else { format!(">={:.0}", row[k]) }
        };
        println!(
            "{beta:>6.1} {:>12} {:>12} {:>12} {:>12} {acc:>10.3}   {:>6.0} {:>6.0} {:>6.0} {:>6.0}",
            cell(0), cell(1), cell(2), cell(3), short[0], short[1], short[2], short[3]
        );
    }
    println!(
        "\nThe informed columns are worth their bookkeeping only where they beat the first one.\n\
         An acceptance near 1.0 means the proposal landscape barely moved -- which is the regime\n\
         where this has nothing to offer, and is why it is printed beside the times."
    );
}
