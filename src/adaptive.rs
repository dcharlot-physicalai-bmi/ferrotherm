//! Ladders that fix themselves — adaptive parallel tempering over beta, and over (beta, coupling).
//!
//! [`crate::tempering::parallel_tempering`] already REPORTS the thing that decides whether a run
//! meant anything: `swap_rates`, the acceptance of each adjacent pair. A pair at zero is a wall the
//! replicas never cross, so the cold end is an independent short anneal and the ladder was
//! decoration. The diagnostic has been there since the beginning and nothing ever acted on it —
//! a user was told their ladder was broken and left to fix it by retyping numbers.
//!
//! # The rule, and why it is this one
//!
//! Place the betas so **every adjacent pair accepts at the same rate**. That is the standard
//! prescription and it follows from what a ladder is for: a state has to random-walk from the hot
//! end to the cold end and back, and a walk on a chain is slowest at its narrowest link. Equalising
//! the links maximises the round-trip rate for a fixed number of replicas.
//!
//! The move is a fixed point iteration on that condition. Between runs, the ladder is redrawn so
//! that the CUMULATIVE acceptance is spread evenly over the replicas: read the measured rates as
//! distances, and re-space the betas along the resulting axis. Nothing here needs a model of the
//! density of states, which is the usual apparatus and the usual place to be wrong.
//!
//! # The second axis, and the honest reason it exists
//!
//! [`adapt_2d`] tempers over `(beta, scale)`, where `scale` multiplies every coupling. Beta alone
//! flattens a landscape by warming it; scaling the couplings flattens it while keeping the fields
//! intact, which is a different path between the same two ends. On a model with strong fields and
//! strong couplings the two are not interchangeable — warming enough to cross a coupling barrier
//! also erases the field information that says which side to land on.
//!
//! # What this is, measured
//!
//! **`examples/adaptive_ladder` measures both claims, and only the mechanism survives.** Acceptance
//! spread falls on every family tried, by a lot — so the respacing does what it says. The energies
//! do not move: on a 16×16 glass, −360.3 against −360.3 inside a between-seed spread of 6 to 8. On
//! these families this is a diagnostic that can now fix itself rather than a faster optimiser, and
//! it cannot save a ladder whose range needs more replicas than it was given.
//!
//! The second axis did not earn its replicas anywhere in that table, including on the fielded model
//! it was predicted to help. [`adapt_2d`] stays because the move is correct and the refusal it
//! carries is worth having. It is not a recommendation.

use crate::gibbs::Sampler;
use crate::graph::{Graph, GraphBuilder};
use crate::rng::Pcg;
use crate::tempering::TemperingResult;

/// How an adaptive run is scheduled.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Params {
    /// Replicas in the ladder. Two is the minimum that can swap at all.
    pub replicas: usize,
    /// Adaptation rounds. Each one runs the ladder, measures acceptance, and redraws it.
    pub epochs: usize,
    /// Swap attempts per epoch.
    pub rounds: usize,
    /// Sweeps between swap attempts.
    pub swap_every: usize,
    /// Hot end. The ladder's ends are FIXED: adaptation moves the interior only, because the ends
    /// are the physics the caller asked for and moving them answers a different question.
    pub beta_min: f64,
    /// Cold end.
    pub beta_max: f64,
}

impl Default for Params {
    fn default() -> Self {
        Params {
            replicas: 8,
            epochs: 6,
            rounds: 200,
            swap_every: 4,
            beta_min: 0.05,
            beta_max: 4.0,
        }
    }
}

/// What an adaptive run did, including the evidence that it helped or did not.
#[derive(Clone, Debug)]
pub struct Outcome {
    pub best: Vec<i8>,
    pub best_e: f64,
    /// The ladder it finished with.
    pub betas: Vec<f64>,
    /// Acceptance per adjacent pair in the LAST epoch.
    pub swap_rates: Vec<f64>,
    /// Acceptance spread — max minus min — at each epoch, first to last.
    ///
    /// This says whether adaptation did anything: a ladder is healthy when every pair accepts
    /// alike, so this falling is the mechanism working, and it staying flat means the geometric
    /// ladder was already even and the epochs were spent for nothing.
    ///
    /// **NEVER READ IT ALONE.** A spread near zero means every pair accepts ALIKE, not that every
    /// pair accepts. A ladder whose range needs more replicas than it was given has every pair at
    /// zero, which is perfectly even and scores BETTER on this column than a healthy ladder with
    /// one weak link. `examples/adaptive_ladder` shows exactly that: at four replicas over a wide
    /// range the spread falls from 0.07 to 0.01 while the worst pair stays at 0.000 throughout.
    /// Read it with the minimum of [`Outcome::swap_rates`], which is the number that can tell a
    /// dead ladder from a healthy one. This is the same shape of error as a frozen chain returning
    /// a small autocorrelation time.
    pub spread: Vec<f64>,
}

/// Run parallel tempering, redrawing the ladder between epochs so every pair accepts alike.
pub fn adapt(g: &Graph, p: &Params, seed: u64) -> Outcome {
    let r = p.replicas.max(2);
    let mut betas = crate::tempering::geometric_ladder(p.beta_min, p.beta_max, r);
    let mut spread = Vec::with_capacity(p.epochs.max(1));
    let mut last = TemperingResult { best: vec![1; g.n], best_e: f64::INFINITY, swap_rates: vec![] };

    for epoch in 0..p.epochs.max(1) {
        let out = crate::tempering::parallel_tempering(
            g,
            &betas,
            p.rounds,
            p.swap_every,
            seed ^ ((epoch as u64) << 17),
            None,
        );
        let hi = out.swap_rates.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let lo = out.swap_rates.iter().cloned().fold(f64::INFINITY, f64::min);
        spread.push(hi - lo);
        if out.best_e < last.best_e {
            last.best = out.best.clone();
            last.best_e = out.best_e;
        }
        last.swap_rates = out.swap_rates.clone();
        // The last epoch's measurement is the report, not an input: redrawing after it would
        // return a ladder whose rates were never measured.
        if epoch + 1 < p.epochs.max(1) {
            betas = respace(&betas, &out.swap_rates);
        }
    }

    Outcome { best: last.best, best_e: last.best_e, betas, swap_rates: last.swap_rates, spread }
}

/// [`adapt`], with the final epoch observed so the free-energy curve comes out beside the answer.
///
/// The traces are recorded on the ladder that was actually measured — the last epoch's, after all
/// respacing — because a curve built on a ladder the run then abandoned would describe nothing
/// that happened. The earlier epochs are adaptation and are not recorded.
///
/// The ladder is geometric between `beta_min` and `beta_max`, so it does not contain `β = 0` and
/// the traces carry free-energy DIFFERENCES rather than an absolute `ln Z`; see
/// [`crate::tempering::LadderTraces::log_z_differences`]. Whether the respacing makes those
/// differences better conditioned is a measurement, and `examples/adaptive_free_energy` makes it.
pub fn adapt_observed(g: &Graph, p: &Params, seed: u64) -> (Outcome, crate::tempering::LadderTraces) {
    let r = p.replicas.max(2);
    let epochs = p.epochs.max(1);
    let mut betas = crate::tempering::geometric_ladder(p.beta_min, p.beta_max, r);
    let mut spread = Vec::with_capacity(epochs);
    let mut last = TemperingResult { best: vec![1; g.n], best_e: f64::INFINITY, swap_rates: vec![] };
    let mut traces = crate::tempering::LadderTraces {
        betas: betas.clone(),
        energies: vec![Vec::new(); r],
        round_trips: 0,
        round_trip_time: None,
    };

    for epoch in 0..epochs {
        let final_epoch = epoch + 1 == epochs;
        let out = if final_epoch {
            let (out, tr) = crate::tempering::parallel_tempering_observed(
                g,
                &betas,
                p.rounds,
                p.swap_every,
                p.rounds / 4,
                seed ^ ((epoch as u64) << 17),
                None,
            );
            traces = tr;
            out
        } else {
            crate::tempering::parallel_tempering(g, &betas, p.rounds, p.swap_every, seed ^ ((epoch as u64) << 17), None)
        };
        let hi = out.swap_rates.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let lo = out.swap_rates.iter().cloned().fold(f64::INFINITY, f64::min);
        spread.push(hi - lo);
        if out.best_e < last.best_e {
            last.best = out.best.clone();
            last.best_e = out.best_e;
        }
        last.swap_rates = out.swap_rates.clone();
        if !final_epoch {
            betas = respace(&betas, &out.swap_rates);
        }
    }

    (Outcome { best: last.best, best_e: last.best_e, betas, swap_rates: last.swap_rates, spread }, traces)
}

/// Redraw a ladder so acceptance is spread evenly across it.
///
/// A pair that accepts RARELY is too far apart and needs its neighbours pulled in; a pair that
/// accepts almost always is closer than it needs to be and is spending a replica. Treating
/// `1/(rate + floor)` as the "length" of each gap and then re-spacing the interior betas at equal
/// cumulative length does both at once, and needs no density of states.
///
/// The ends never move. They are what the caller asked for.
pub fn respace(betas: &[f64], rates: &[f64]) -> Vec<f64> {
    let r = betas.len();
    if r < 3 || rates.len() + 1 != r {
        return betas.to_vec();
    }
    // A floor, because a pair that accepted ZERO times would otherwise have infinite length and
    // collapse every other gap to nothing -- turning one broken pair into a broken ladder.
    const FLOOR: f64 = 0.02;
    let len: Vec<f64> = rates.iter().map(|&x| 1.0 / (x.max(0.0) + FLOOR)).collect();
    let total: f64 = len.iter().sum();
    if !total.is_finite() || total <= 0.0 {
        return betas.to_vec();
    }

    // Walk the old ladder in log-beta, which is the scale a geometric ladder is even on and the
    // scale acceptance varies smoothly in. Interpolating in beta itself crowds the hot end.
    let lo = betas[0].ln();
    let hi = betas[r - 1].ln();
    let mut cum = vec![0.0; r];
    for i in 0..r - 1 {
        cum[i + 1] = cum[i] + len[i];
    }
    let mut out = vec![betas[0]; r];
    out[r - 1] = betas[r - 1];
    for k in 1..r - 1 {
        let target = total * k as f64 / (r - 1) as f64;
        // Which old gap does this cumulative position fall in?
        let mut i = 0;
        while i + 2 < r && cum[i + 1] < target {
            i += 1;
        }
        let span = cum[i + 1] - cum[i];
        let frac = if span > 0.0 { (target - cum[i]) / span } else { 0.0 };
        let a = betas[i].ln();
        let b = betas[i + 1].ln();
        out[k] = (a + (b - a) * frac).exp();
    }
    // Monotone by construction above, but float arithmetic near-equal gaps can tie. A ladder that
    // is not strictly increasing makes `delta_beta` zero or negative and the swap criterion
    // meaningless, so nudge rather than return something that samples wrongly.
    for k in 1..r {
        if out[k] <= out[k - 1] {
            out[k] = out[k - 1] * (1.0 + 1e-9);
        }
    }
    let _ = (lo, hi);
    out
}

/// A copy of `g` with every coupling multiplied by `scale`, fields untouched.
///
/// This is the second tempering axis. Warming a model flattens couplings AND fields together;
/// scaling the couplings flattens the couplings alone, so a replica can cross a coupling barrier
/// while still being told by its fields which side to land on.
pub fn scaled(g: &Graph, scale: f64) -> Graph {
    let mut b = GraphBuilder::new(g.n);
    for i in 0..g.n {
        for k in g.offset[i]..g.offset[i + 1] {
            let j = g.nbr[k] as usize;
            if j > i {
                b.couple(i, j, g.w[k] * scale);
            }
        }
    }
    for (i, &h) in g.h.iter().enumerate() {
        if h != 0.0 {
            b.bias(i, h);
        }
    }
    b.build()
}

/// Two-dimensional tempering over `(beta, scale)`.
///
/// Replicas sit on a `betas.len()` x `scales.len()` grid and swap with their neighbours along both
/// axes. A swap along the beta axis is the ordinary replica exchange; a swap along the scale axis
/// exchanges states between two DIFFERENT graphs, so its criterion carries the energy of each state
/// under each graph rather than a single delta-beta times delta-E.
///
/// Returns the best state found under the ORIGINAL model — that is, at `scale = 1`, which must be
/// one of `scales` or the answer is about a model the caller did not ask about.
pub fn adapt_2d(
    g: &Graph,
    betas: &[f64],
    scales: &[f64],
    rounds: usize,
    swap_every: usize,
    seed: u64,
) -> Result<Outcome, String> {
    if betas.len() < 2 {
        return Err("a ladder needs at least two betas".into());
    }
    if scales.is_empty() {
        return Err("give at least one coupling scale; [1.0] is ordinary tempering".into());
    }
    if !scales.iter().any(|&s| (s - 1.0).abs() < 1e-12) {
        return Err(
            "`scales` must contain 1.0: the answer is about the model as given, and a grid that \
             never visits it reports the best state of a DIFFERENT model"
                .into(),
        );
    }
    let (nb, ns) = (betas.len(), scales.len());
    let graphs: Vec<Graph> = scales.iter().map(|&s| scaled(g, s)).collect();
    let at = |bi: usize, si: usize| si * nb + bi;

    let mut reps: Vec<Sampler> = Vec::with_capacity(nb * ns);
    for si in 0..ns {
        for bi in 0..nb {
            reps.push(Sampler::new(&graphs[si], betas[bi], seed ^ ((at(bi, si) as u64) * 0x9E37)));
        }
    }
    let mut rng = Pcg::new(seed ^ 0x2D2D, 7);
    let mut attempts = vec![0u64; nb.saturating_sub(1)];
    let mut accepts = vec![0u64; nb.saturating_sub(1)];
    // The physical row is the one at scale 1; its best is the answer.
    let phys = scales.iter().position(|&s| (s - 1.0).abs() < 1e-12).unwrap();
    let mut best = reps[at(nb - 1, phys)].s.clone();
    let mut best_e = g.energy(&best);

    for round in 0..rounds {
        // Each replica already holds its own graph from construction; nothing rebinds it here.
        for rep in reps.iter_mut() {
            for _ in 0..swap_every.max(1) {
                rep.sweep(None);
            }
        }
        for si in 0..ns {
            for bi in 0..nb {
                let e = g.energy(&reps[at(bi, si)].s);
                if e < best_e {
                    best_e = e;
                    best = reps[at(bi, si)].s.clone();
                }
            }
        }

        // Beta axis, alternating pairs, within each scale row.
        let start = round % 2;
        for si in 0..ns {
            for bi in (start..nb.saturating_sub(1)).step_by(2) {
                let (a, b) = (at(bi, si), at(bi + 1, si));
                let ea = graphs[si].energy(&reps[a].s);
                let eb = graphs[si].energy(&reps[b].s);
                let arg = (betas[bi + 1] - betas[bi]) * (eb - ea);
                attempts[bi] += 1;
                if arg >= 0.0 || rng.f64() < arg.exp() {
                    accepts[bi] += 1;
                    swap_states(&mut reps, a, b);
                }
            }
        }
        // Scale axis. The two replicas obey DIFFERENT Hamiltonians, so the acceptance is
        // exp(beta * [ (E_a(x_a) + E_b(x_b)) - (E_a(x_b) + E_b(x_a)) ]) with a shared beta -- each
        // state is scored under both graphs. Using a single delta here would be the ordinary
        // replica-exchange formula applied where it does not hold.
        let sstart = (round / 2) % 2;
        for bi in 0..nb {
            for si in (sstart..ns.saturating_sub(1)).step_by(2) {
                let (a, b) = (at(bi, si), at(bi, si + 1));
                let (ga, gb) = (&graphs[si], &graphs[si + 1]);
                let arg = betas[bi]
                    * (ga.energy(&reps[a].s) + gb.energy(&reps[b].s)
                        - ga.energy(&reps[b].s)
                        - gb.energy(&reps[a].s));
                if arg >= 0.0 || rng.f64() < arg.exp() {
                    swap_states(&mut reps, a, b);
                }
            }
        }
    }

    let rates: Vec<f64> = (0..nb.saturating_sub(1))
        .map(|i| accepts[i] as f64 / attempts[i].max(1) as f64)
        .collect();
    let hi = rates.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let lo = rates.iter().cloned().fold(f64::INFINITY, f64::min);
    Ok(Outcome {
        best,
        best_e,
        betas: betas.to_vec(),
        swap_rates: rates,
        spread: vec![hi - lo],
    })
}

fn swap_states(reps: &mut [Sampler], a: usize, b: usize) {
    let (lo, hi) = if a < b { (a, b) } else { (b, a) };
    let (l, r) = reps.split_at_mut(hi);
    core::mem::swap(&mut l[lo].s, &mut r[0].s);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ising::lattice2d;

    /// THE CLAIM IS THAT THE SPREAD FALLS, so that is what this asserts -- on a ladder deliberately
    /// built to be uneven, because a ladder that was already even proves nothing about adaptation.
    ///
    /// A wide beta range over a lattice gives adjacent pairs wildly different acceptance: the hot
    /// pairs swap almost always and the cold pairs almost never.
    #[test]
    fn adaptation_evens_out_a_ladder_that_started_uneven() {
        let g = lattice2d(12, 1.0);
        let p = Params {
            replicas: 8,
            epochs: 6,
            rounds: 300,
            swap_every: 2,
            beta_min: 0.02,
            beta_max: 6.0,
        };
        let out = adapt(&g, &p, 11);
        assert_eq!(out.spread.len(), p.epochs);
        assert!(
            out.spread[p.epochs - 1] < out.spread[0] * 0.75,
            "the spread must fall, and it went {:?}",
            out.spread
        );
        // The ends are the caller's physics and adaptation does not get to change them.
        assert!((out.betas[0] - p.beta_min).abs() < 1e-12);
        assert!((out.betas[out.betas.len() - 1] - p.beta_max).abs() < 1e-12);
        // A ladder must stay strictly increasing or the swap criterion is meaningless.
        assert!(out.betas.windows(2).all(|w| w[1] > w[0]), "{:?}", out.betas);
    }

    /// AN EVEN SPREAD IS NOT A HEALTHY LADDER, and this pins the trap rather than trusting the
    /// docstring that warns about it.
    ///
    /// Four replicas over beta 0.02 to 8 on a glass is a range that needs more rungs than it was
    /// given. Adaptation cannot fix that -- no placement of two interior betas makes those gaps
    /// crossable -- but the SPREAD still falls, because every pair converges on the same rate of
    /// nearly zero. All-dead is perfectly even. A caller reading only the spread would call this
    /// the healthiest ladder in the file.
    #[test]
    fn a_uniformly_dead_ladder_has_a_small_spread_and_is_not_healthy() {
        let mut rng = Pcg::new(4, 0xDEAD_1ADD);
        let l = 12usize;
        let mut b = GraphBuilder::new(l * l);
        for y in 0..l {
            for x in 0..l {
                let i = y * l + x;
                let s = |r: &mut Pcg| if r.f64() < 0.5 { 1.0 } else { -1.0 };
                b.couple(i, y * l + (x + 1) % l, s(&mut rng));
                b.couple(i, ((y + 1) % l) * l + x, s(&mut rng));
            }
        }
        let g = b.build();
        let p = Params {
            replicas: 4,
            epochs: 5,
            rounds: 200,
            swap_every: 2,
            beta_min: 0.02,
            beta_max: 8.0,
        };
        let out = adapt(&g, &p, 3);
        let worst = out.swap_rates.iter().cloned().fold(f64::INFINITY, f64::min);
        assert!(worst < 0.02, "this ladder is meant to be dead: {:?}", out.swap_rates);
        assert!(
            *out.spread.last().unwrap() <= out.spread[0] + 1e-12,
            "and its spread does not rise: {:?}",
            out.spread
        );
        // The two together are the point: even, and dead. Either number alone lies.
        assert!(
            *out.spread.last().unwrap() < 0.2,
            "a dead ladder looks EVEN, which is why the spread must not be read alone: {:?}",
            out.spread
        );
    }

    /// Respacing is a pure function and its two extremes are worth pinning, because both are
    /// silent when wrong: it must not move a ladder that is already even, and it must pull in the
    /// neighbours of a pair that never accepts.
    #[test]
    fn respacing_leaves_an_even_ladder_alone_and_closes_a_dead_gap() {
        let even = crate::tempering::geometric_ladder(0.1, 4.0, 6);
        let same = respace(&even, &[0.4; 5]);
        for (a, b) in even.iter().zip(&same) {
            assert!((a - b).abs() < 1e-9, "an even ladder must not move: {even:?} -> {same:?}");
        }

        // One dead pair in the middle: its neighbours must close in around it.
        let rates = vec![0.6, 0.6, 0.0, 0.6, 0.6];
        let moved = respace(&even, &rates);
        assert!(moved.windows(2).all(|w| w[1] > w[0]), "{moved:?}");
        assert!((moved[0] - even[0]).abs() < 1e-12 && (moved[5] - even[5]).abs() < 1e-12);
        let gap = |v: &[f64], i: usize| v[i + 1].ln() - v[i].ln();
        assert!(
            gap(&moved, 2) < gap(&even, 2),
            "the dead gap must narrow: {:.4} was {:.4}",
            gap(&moved, 2),
            gap(&even, 2)
        );
    }

    /// A ladder too short to have an interior has nothing to move, and must come back untouched
    /// rather than panicking on an empty range.
    #[test]
    fn a_two_rung_ladder_is_returned_unchanged() {
        let two = vec![0.5, 2.0];
        assert_eq!(respace(&two, &[0.3]), two);
        // A rates array of the wrong length is a caller error, and returning the input unchanged
        // is the only thing that cannot silently sample the wrong distribution.
        let four = crate::tempering::geometric_ladder(0.1, 4.0, 4);
        assert_eq!(respace(&four, &[0.3]), four);
    }

    /// Scaling touches couplings and leaves fields alone, which is the entire reason the second
    /// axis is a different move from raising beta.
    #[test]
    fn scaling_moves_couplings_and_not_fields() {
        let mut b = GraphBuilder::new(3);
        b.couple(0, 1, 2.0);
        b.couple(1, 2, -1.0);
        b.bias(0, 0.5);
        b.bias(2, -1.5);
        let g = b.build();
        let s = scaled(&g, 0.25);
        assert_eq!(s.n, g.n);
        assert_eq!(s.n_edges, g.n_edges);
        assert_eq!(s.h, g.h, "fields must be untouched");
        let all_up = vec![1i8; 3];
        // E = -sum h s - sum J s s. With every spin +1: couplings contribute -(2.0 - 1.0) = -1.0
        // at scale 1 and -0.25 at scale 0.25; fields contribute -(0.5 - 1.5) = 1.0 either way.
        assert!((g.energy(&all_up) - 0.0).abs() < 1e-12, "{}", g.energy(&all_up));
        assert!((s.energy(&all_up) - 0.75).abs() < 1e-12, "{}", s.energy(&all_up));
    }

    /// A 2D grid that never visits the model as given is refused, because its answer would be the
    /// best state of a different model and nothing in the reply would say so.
    #[test]
    fn a_grid_that_never_reaches_the_real_model_is_refused() {
        let g = lattice2d(6, 1.0);
        let b = crate::tempering::geometric_ladder(0.2, 3.0, 4);
        let e = adapt_2d(&g, &b, &[0.5, 0.8], 10, 2, 1).unwrap_err();
        assert!(e.contains("1.0"), "{e}");
        assert!(adapt_2d(&g, &b, &[], 10, 2, 1).is_err());
        assert!(adapt_2d(&g, &[1.0], &[1.0], 10, 2, 1).is_err());
        // And the ordinary case still runs: one scale is plain tempering.
        assert!(adapt_2d(&g, &b, &[1.0], 20, 2, 1).is_ok());
    }

    /// The answer must be a state of the ORIGINAL model, scored under the original model.
    #[test]
    fn the_2d_answer_is_about_the_model_as_given() {
        let g = lattice2d(8, 1.0);
        let b = crate::tempering::geometric_ladder(0.1, 3.0, 5);
        let out = adapt_2d(&g, &b, &[0.4, 0.7, 1.0], 300, 2, 5).unwrap();
        assert_eq!(out.best.len(), g.n);
        assert!((g.energy(&out.best) - out.best_e).abs() < 1e-9, "energy must be the real one");
        // A 2D ladder over a ferromagnet at these betas should find something well below zero.
        assert!(out.best_e < -0.8 * g.n_edges as f64, "best {} on {} edges", out.best_e, g.n_edges);
    }
}
