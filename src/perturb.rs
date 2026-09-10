//! Perturb-and-MAP — Gumbel noise turns a maximiser into a sampler, and its expected maximum into
//! `log Z`.
//!
//! Give every state its own zero-mean Gumbel and add it to the score `theta(s) = -beta E(s)`. Then
//!
//! ```text
//!   argmax_s { theta(s) + gamma(s) }  ~  Gibbs(beta)     exactly
//!   E[ max_s { theta(s) + gamma(s) } ] = log Z           exactly, with variance pi^2/6
//! ```
//!
//! [`perturb_and_map`] does that literally, at one Gumbel per state, so it stops where enumeration
//! does. What carries past it is that LOW-ORDER perturbations — one Gumbel per spin per value
//! rather than per state — keep a MAP solver in the loop and still bound `log Z`
//! (Hazan & Jaakkola, *On the partition function and random maximum a-posteriori perturbations*,
//! ICML 2012):
//!
//! * perturbing **every** spin's field gives an upper bound, [`perturbed_map_upper`], which is an
//!   expectation and so arrives as an estimate with a standard error;
//! * perturbing **one** spin gives a lower bound, and there the expectation has a closed form —
//!   `log sum_{s_k} exp max_{rest} theta` — so it costs two MAP solves and no sampling at all.
//!
//! [`map_ladder`] is that closed form for any chosen set of spins: sum exactly over the set,
//! maximise over the rest, and pair it with the counting bound `+ (n - |S|) log 2` above. Both ends
//! of the ladder are exact — the empty set is the MAP value, the full set is `log Z` twice — so the
//! bracket is checkable against [`crate::exact::Elimination`] rather than asserted.
//!
//! The Euler–Mascheroni shift that makes the noise zero-mean moves the expected maximum, never the
//! argmax: it is the same constant on every candidate.

use crate::exact::{Elimination, TooWide};
use crate::graph::{Graph, GraphBuilder, rescaled};
use crate::rng::Pcg;
use crate::samples::{ENUMERATION_LIMIT, Refused};

/// The Euler–Mascheroni constant, which is the mean of a standard Gumbel.
pub const EULER_MASCHERONI: f64 = 0.5772156649015329;

/// The most spins [`map_ladder`] will sum over, since it costs `2^|S|` MAP solves.
const LADDER_LIMIT: usize = 20;

/// One zero-mean Gumbel deviate, `-ln(-ln U) - EULER_MASCHERONI`.
///
/// Zero-mean rather than standard: the shift is what makes `E[max]` equal `log Z` instead of
/// `log Z + EULER_MASCHERONI`.
#[must_use]
pub fn gumbel(r: &mut Pcg) -> f64 {
    // `Pcg::f64` is uniform on [0, 1), and a zero would make this `-inf`. Redrawing keeps the
    // deviate exactly Gumbel; clamping would put an atom at the largest representable value.
    let mut u = r.f64();
    while u <= 0.0 {
        u = r.f64();
    }
    -(-u.ln()).ln() - EULER_MASCHERONI
}

/// A full-order perturb-and-MAP draw.
#[derive(Clone, Debug)]
pub struct Draw {
    /// The argmax, which is an exact Gibbs sample at the beta it was drawn at.
    pub state: Vec<i8>,
    /// The perturbed maximum: mean `log Z`, variance `pi^2/6`.
    pub value: f64,
}

/// An estimate of `log Z` with the spread of the draws behind it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LogZEstimate {
    /// The mean, in nats.
    pub value: f64,
    /// Standard error of that mean.
    pub stderr: f64,
    /// Sample variance of the perturbed maxima — `pi^2/6` for a full-order perturbation, whatever
    /// the model, and `n pi^2/6` for an all-dimensions perturbation of an uncoupled one.
    pub variance: f64,
    /// Draws averaged.
    pub draws: usize,
}

/// Mean and spread of `vals`, with the `n - 1` denominator.
fn estimate(vals: &[f64]) -> LogZEstimate {
    let n = vals.len();
    let mean = vals.iter().sum::<f64>() / n as f64;
    let variance = if n > 1 {
        vals.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / (n - 1) as f64
    } else {
        0.0
    };
    LogZEstimate { value: mean, stderr: (variance / n as f64).sqrt(), variance, draws: n }
}

/// One exact Gibbs sample, by perturbing every state's score and taking the argmax.
///
/// # Errors
///
/// [`Refused::TooLargeToEnumerate`] past [`ENUMERATION_LIMIT`]: full-order perturbation needs one
/// Gumbel per state, so it visits `2^n` of them.
pub fn perturb_and_map(g: &Graph, beta: f64, r: &mut Pcg) -> Result<Draw, Refused> {
    if g.n > ENUMERATION_LIMIT {
        return Err(Refused::TooLargeToEnumerate { spins: g.n, limit: ENUMERATION_LIMIT });
    }
    let m = 1usize << g.n;
    let mut s = vec![-1i8; g.n];
    let (mut best, mut arg) = (f64::NEG_INFINITY, 0usize);
    for mask in 0..m {
        for b in 0..g.n {
            s[b] = if mask >> b & 1 == 1 { 1 } else { -1 };
        }
        let v = -beta * g.energy(&s) + gumbel(r);
        if v > best {
            best = v;
            arg = mask;
        }
    }
    for b in 0..g.n {
        s[b] = if arg >> b & 1 == 1 { 1 } else { -1 };
    }
    Ok(Draw { state: s, value: best })
}

/// Average `draws` full-order perturbed maxima: an unbiased estimator of `log Z`.
///
/// Unbiased for any model and any beta, because the maximum of independently Gumbel-perturbed
/// scores is itself Gumbel with location `log Z`. The energies are computed once and reused across
/// draws, so the cost is `draws * 2^n` Gumbels rather than `draws * 2^n` energy evaluations.
///
/// # Errors
///
/// [`Refused::TooLargeToEnumerate`], as [`perturb_and_map`].
///
/// # Panics
///
/// If `draws` is zero — an average needs something to average.
pub fn gumbel_log_z(g: &Graph, beta: f64, draws: usize, seed: u64) -> Result<LogZEstimate, Refused> {
    assert!(draws > 0, "an average needs at least one draw");
    if g.n > ENUMERATION_LIMIT {
        return Err(Refused::TooLargeToEnumerate { spins: g.n, limit: ENUMERATION_LIMIT });
    }
    let m = 1usize << g.n;
    let mut s = vec![-1i8; g.n];
    let mut theta = Vec::with_capacity(m);
    for mask in 0..m {
        for b in 0..g.n {
            s[b] = if mask >> b & 1 == 1 { 1 } else { -1 };
        }
        theta.push(-beta * g.energy(&s));
    }

    let mut r = Pcg::new(seed, 0);
    let mut vals = Vec::with_capacity(draws);
    for _ in 0..draws {
        let mut best = f64::NEG_INFINITY;
        for &t in &theta {
            let v = t + gumbel(&mut r);
            if v > best {
                best = v;
            }
        }
        vals.push(best);
    }
    Ok(estimate(&vals))
}

/// A deterministic two-sided bracket on `log Z`, from MAP solves alone.
#[derive(Clone, Debug)]
pub struct MapLadder {
    /// Spins summed exactly. The rest were maximised over.
    pub summed: Vec<usize>,
    /// Never above `log Z`. Accumulated through [`crate::round`] so that stays true after rounding.
    pub lower: f64,
    /// Never below `log Z`, being `lower` plus `(n - |summed|) log 2`.
    pub upper: f64,
    /// MAP solves performed, which is `2^summed.len()`.
    pub solves: usize,
}

impl MapLadder {
    /// Bracket width in nats, which is `(n - |summed|) log 2` up to rounding.
    #[must_use]
    pub fn width(&self) -> f64 {
        self.upper - self.lower
    }

    /// Whether the bracket contains a given `log Z`.
    #[must_use]
    pub fn contains(&self, log_z: f64) -> bool {
        self.lower <= log_z && log_z <= self.upper
    }
}

/// A few ulps below `x`, absorbing the rounding of the `exp` and `ln` that produced it.
fn slack_down(x: f64) -> f64 {
    x.next_down().next_down().next_down()
}

/// A few ulps above `x`, for the same reason as [`slack_down`].
fn slack_up(x: f64) -> f64 {
    x.next_up().next_up().next_up()
}

/// The model with some spins held, as a graph over the same nodes — held ones isolated and
/// unbiased, so they cost nothing whichever way a solver leaves them — plus the constant energy the
/// held terms contribute.
fn condition(g: &Graph, fixed: &[Option<i8>]) -> (Graph, f64) {
    let mut b = GraphBuilder::new(g.n);
    let mut offset = 0.0;
    for i in 0..g.n {
        match fixed[i] {
            Some(v) => offset -= g.h[i] * f64::from(v),
            None => b.bias(i, g.h[i]),
        }
    }
    for i in 0..g.n {
        for k in g.offset[i]..g.offset[i + 1] {
            let j = g.nbr[k] as usize;
            // Each undirected edge sits in both CSR rows; fold it once.
            if j <= i {
                continue;
            }
            let w = g.w[k];
            match (fixed[i], fixed[j]) {
                (Some(a), Some(c)) => offset -= w * f64::from(a) * f64::from(c),
                (Some(a), None) => b.bias(j, w * f64::from(a)),
                (None, Some(c)) => b.bias(i, w * f64::from(c)),
                (None, None) => b.couple(i, j, w),
            }
        }
    }
    (b.build(), offset)
}

/// Sum `log Z` exactly over `sites`, maximise over the rest, and bracket `log Z` between that and
/// the counting bound above it.
///
/// `log sum_{x_S} exp max_{x_-S} theta(x)` is below `log Z` because a maximum is below a
/// log-sum-exp, and within `(n - |S|) log 2` of it because that log-sum-exp has `2^(n-|S|)` terms.
/// With one site it is exactly Hazan & Jaakkola's single-dimension perturbation bound, whose
/// expectation over the Gumbel has this closed form; with every site it is `log Z` itself.
///
/// # Errors
///
/// [`TooWide`] when the elimination order for a conditioned model exceeds `el.max_width`.
/// Conditioning only removes edges, so a model whose own `log_partition` fits cannot be refused
/// here.
///
/// # Panics
///
/// If `beta` is negative or not finite, or `sites` repeats an index, names one past the last spin,
/// or holds more than 20 — the cost is `2^sites.len()` MAP solves.
pub fn map_ladder(
    g: &Graph,
    beta: f64,
    sites: &[usize],
    el: &Elimination,
) -> Result<MapLadder, TooWide> {
    assert!(beta.is_finite() && beta >= 0.0, "beta must be finite and non-negative; got {beta}");
    assert!(
        sites.len() <= LADDER_LIMIT,
        "summing over {} spins is 2^{} MAP solves; the limit is {LADDER_LIMIT}",
        sites.len(),
        sites.len()
    );
    let mut fixed: Vec<Option<i8>> = vec![None; g.n];
    for &i in sites {
        assert!(i < g.n, "site {i} is past the last of {} spins", g.n);
        assert!(fixed[i].is_none(), "site {i} is listed twice");
        fixed[i] = Some(1);
    }

    let k = sites.len();
    let m = 1usize << k;
    let mut scores = Vec::with_capacity(m);
    for mask in 0..m {
        for (b, &i) in sites.iter().enumerate() {
            fixed[i] = Some(if mask >> b & 1 == 1 { 1 } else { -1 });
        }
        let (red, offset) = condition(g, &fixed);
        let e = el.ground_state(&red)?.ground_energy.expect("min-sum was run, so it reports one");
        scores.push(-beta * (offset + e));
    }

    // Log-sum-exp with the maximum shifted out, so one term is exactly 1.0 and the sum cannot
    // round to zero. The terms are summed in both directions: down for the bound that must stay
    // below, up for the one that must stay above.
    let mx = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let terms: Vec<f64> = scores.iter().map(|&v| (v - mx).exp()).collect();
    let lower = slack_down(mx + crate::round::sum_down(&terms).ln());
    let top = slack_up(mx + crate::round::sum_up(&terms).ln());
    let upper = slack_up(top + (g.n - k) as f64 * std::f64::consts::LN_2);

    Ok(MapLadder { summed: sites.to_vec(), lower, upper, solves: m })
}

/// The tightest single-spin bound: [`map_ladder`] over each spin in turn, the best kept.
///
/// `2n` MAP solves. Which spin is best is a property of the instance, so it is searched rather than
/// guessed.
///
/// # Errors
///
/// [`TooWide`], as [`map_ladder`].
///
/// # Panics
///
/// If the model has no spins, since then there is no single spin to pick.
pub fn best_single_site(g: &Graph, beta: f64, el: &Elimination) -> Result<MapLadder, TooWide> {
    assert!(g.n > 0, "a single-spin bound needs a spin");
    let mut best: Option<MapLadder> = None;
    for i in 0..g.n {
        let l = map_ladder(g, beta, &[i], el)?;
        if best.as_ref().is_none_or(|b| l.lower > b.lower) {
            best = Some(l);
        }
    }
    Ok(best.expect("the model has at least one spin"))
}

/// Hazan & Jaakkola's all-dimensions upper bound on `log Z`, estimated by `draws` MAP solves.
///
/// The quantity estimated is `E[ max_s { -beta E(s) + sum_i gamma_i(s_i) } ]`, which is never below
/// `log Z` and equals it exactly when the model has no couplings. A per-spin perturbation
/// `gamma_i(s_i)` splits as `a_i + b_i s_i`, so each draw is a MAP solve on the same couplings with
/// the fields shifted by `b_i` — one call to an exact solver, not a search over states.
///
/// The **mean** is what bounds. A finite sample of it can land below `log Z`, which is why
/// [`LogZEstimate::stderr`] comes with it.
///
/// # Errors
///
/// [`TooWide`], as [`map_ladder`].
///
/// # Panics
///
/// If `draws` is zero, or `beta` is negative or not finite.
pub fn perturbed_map_upper(
    g: &Graph,
    beta: f64,
    draws: usize,
    seed: u64,
    el: &Elimination,
) -> Result<LogZEstimate, TooWide> {
    assert!(draws > 0, "an average needs at least one draw");
    assert!(beta.is_finite() && beta >= 0.0, "beta must be finite and non-negative; got {beta}");

    // `theta(s) = -beta E(s)` is the energy of the beta-scaled model, negated, so the perturbed
    // maximum is a MINIMUM of that model's energy with the fields shifted. Scale once; only the
    // fields move between draws.
    let mut work = rescaled(g, beta);
    let mut r = Pcg::new(seed, 0);
    let mut consts = vec![0.0f64; g.n];
    let mut vals = Vec::with_capacity(draws);
    for _ in 0..draws {
        for i in 0..g.n {
            let (gp, gm) = (gumbel(&mut r), gumbel(&mut r));
            consts[i] = 0.5 * (gp + gm);
            work.h[i] = beta * g.h[i] + 0.5 * (gp - gm);
        }
        let e = el.ground_state(&work)?.ground_energy.expect("min-sum was run, so it reports one");
        vals.push(consts.iter().sum::<f64>() - e);
    }
    Ok(estimate(&vals))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{free_energy, ising};
    use std::f64::consts::{LN_2, PI};

    /// Variance of a standard Gumbel, and of any maximum of independently Gumbel-perturbed scores.
    const GUMBEL_VAR: f64 = PI * PI / 6.0;

    fn glass(n: usize, seed: u64) -> Graph {
        let mut r = Pcg::new(seed, 11);
        let mut b = GraphBuilder::new(n);
        for i in 0..n {
            b.bias(i, 1.6 * r.f64() - 0.8);
        }
        for i in 0..n {
            for j in i + 1..n {
                if r.f64() < 0.55 {
                    b.couple(i, j, 1.6 * r.f64() - 0.8);
                }
            }
        }
        b.build()
    }

    fn mask_of(s: &[i8]) -> usize {
        s.iter().enumerate().fold(0usize, |m, (b, &v)| if v > 0 { m | 1 << b } else { m })
    }

    /// The primitive's own two moments, against their closed forms.
    ///
    /// This is the test that catches a missing Euler–Mascheroni shift, and it is worth having
    /// separately: a standard Gumbel would leave every `log Z` estimate in this module exactly
    /// 0.577 nats high, which on a model whose bound is loose by more than that looks like a
    /// perfectly healthy upper bound.
    #[test]
    fn a_gumbel_deviate_has_mean_zero_and_variance_pi_squared_over_six() {
        let mut r = Pcg::new(3, 1);
        let n = 200_000;
        let v: Vec<f64> = (0..n).map(|_| gumbel(&mut r)).collect();
        let est = estimate(&v);
        // Four standard errors of the mean, 1.2825/sqrt(n).
        assert!(est.value.abs() < 4.0 * est.stderr, "mean {} (se {})", est.value, est.stderr);
        assert!(
            (est.variance - GUMBEL_VAR).abs() < 0.05,
            "variance {} against pi^2/6 = {GUMBEL_VAR}",
            est.variance
        );
    }

    /// Conditioning is an exact rewrite of the energy, not an approximation of it.
    ///
    /// Everything in [`map_ladder`] rests on this: if the held terms were folded into the wrong
    /// neighbour or double-counted, every bound would still be a smooth-looking number.
    #[test]
    fn conditioning_reproduces_the_full_energy_of_every_consistent_state() {
        let n = 7;
        let g = glass(n, 12);
        let mut fixed: Vec<Option<i8>> = vec![None; n];
        fixed[1] = Some(1);
        fixed[4] = Some(-1);
        fixed[5] = Some(1);
        let (red, offset) = condition(&g, &fixed);
        let mut checked = 0;
        for mask in 0..(1usize << n) {
            let s: Vec<i8> = (0..n).map(|b| if mask >> b & 1 == 1 { 1 } else { -1 }).collect();
            if fixed.iter().enumerate().any(|(i, f)| f.is_some_and(|v| v != s[i])) {
                continue;
            }
            assert!(
                (offset + red.energy(&s) - g.energy(&s)).abs() < 1e-12,
                "state {mask}: {} + {} != {}",
                offset,
                red.energy(&s),
                g.energy(&s)
            );
            checked += 1;
        }
        assert_eq!(checked, 1 << (n - 3), "every consistent state, and only those");
    }

    /// Both ends of the ladder are closed forms, so both are checked as equalities.
    ///
    /// Empty set: the lower bound IS the MAP value `-beta E_0` and the upper is that plus
    /// `n log 2`. Full set: both ends are `log Z`. Oracles are `Elimination::ground_state` and
    /// `Elimination::log_partition`, not another bound.
    #[test]
    fn the_ladder_ends_on_the_map_value_and_on_log_z() {
        let n = 8;
        let g = glass(n, 5);
        let el = Elimination::default();
        let beta = 0.9;
        let log_z = el.log_partition(&g, beta).unwrap().log_z.unwrap();
        let e0 = el.ground_state(&g).unwrap().ground_energy.unwrap();

        let none = map_ladder(&g, beta, &[], &el).unwrap();
        assert_eq!(none.solves, 1);
        assert!((none.lower - (-beta * e0)).abs() < 1e-12, "empty lower {}", none.lower);
        assert!(
            (none.upper - (-beta * e0 + n as f64 * LN_2)).abs() < 1e-12,
            "empty upper {}",
            none.upper
        );

        let all: Vec<usize> = (0..n).collect();
        let full = map_ladder(&g, beta, &all, &el).unwrap();
        assert_eq!(full.solves, 1 << n);
        assert!((full.lower - log_z).abs() < 1e-9, "full lower {} vs {log_z}", full.lower);
        assert!((full.upper - log_z).abs() < 1e-9, "full upper {} vs {log_z}", full.upper);
    }

    /// Every rung brackets the exact `log Z`, and adding a spin can only narrow the bracket.
    ///
    /// The tolerance is the ORACLE's: `log_partition` accumulates its own log-sum-exp and is good
    /// to about 1e-14, and at the top rung the bound equals it. The bound's own directed rounding
    /// is three ulps and is what keeps the inequality from failing on the rounding of `exp`.
    #[test]
    fn every_rung_brackets_log_z_and_the_bracket_only_narrows() {
        let el = Elimination::default();
        for seed in 0..4u64 {
            for &beta in &[0.0, 0.4, 1.3] {
                let n = 7;
                let g = glass(n, seed);
                let log_z = el.log_partition(&g, beta).unwrap().log_z.unwrap();
                let mut prev: Option<MapLadder> = None;
                for k in 0..=n {
                    let sites: Vec<usize> = (0..k).collect();
                    let l = map_ladder(&g, beta, &sites, &el).unwrap();
                    assert!(
                        l.lower <= log_z + 1e-9,
                        "seed {seed} beta {beta} k {k}: lower {} above log Z {log_z}",
                        l.lower
                    );
                    assert!(
                        l.upper >= log_z - 1e-9,
                        "seed {seed} beta {beta} k {k}: upper {} below log Z {log_z}",
                        l.upper
                    );
                    assert!(l.contains(log_z) || l.width() < 1e-9);
                    if let Some(p) = &prev {
                        assert!(l.lower >= p.lower - 1e-12, "k {k}: lower fell");
                        assert!(l.upper <= p.upper + 1e-12, "k {k}: upper rose");
                    }
                    prev = Some(l);
                }
                // At beta zero every state weighs one, so the ladder is exact at every rung.
                if beta == 0.0 {
                    let mid = map_ladder(&g, beta, &[0, 3], &el).unwrap();
                    assert!((mid.lower - 2.0 * LN_2).abs() < 1e-12);
                    assert!((mid.upper - n as f64 * LN_2).abs() < 1e-12);
                }
            }
        }
    }

    /// The mean of the full-order perturbed maximum is `log Z` and its variance is `pi^2/6`.
    ///
    /// Both are closed forms — the second one *independent of the model*, which is the sharper
    /// check of the two: a perturbation applied at the wrong scale (beta folded in twice, say)
    /// would still average to something plausible but could not have unit-scale Gumbel spread.
    /// The `log Z` oracle is the transfer matrix, not an enumeration of the same states.
    #[test]
    fn the_full_order_perturbed_maximum_has_mean_log_z_and_gumbel_spread() {
        let (n, j, h, beta) = (6, 0.7, 0.3, 1.1);
        let g = ising::ring(n, j, h);
        let log_z = free_energy::ring_log_z(n, j, h, beta);
        let est = gumbel_log_z(&g, beta, 20_000, 7).unwrap();
        assert_eq!(est.draws, 20_000);
        assert!(
            (est.value - log_z).abs() < 4.0 * est.stderr,
            "estimate {} +- {} against transfer-matrix log Z {log_z}",
            est.value,
            est.stderr
        );
        assert!(
            (est.variance - GUMBEL_VAR).abs() < 0.1,
            "variance {} against pi^2/6 = {GUMBEL_VAR}",
            est.variance
        );
    }

    /// The argmax is the exact Boltzmann distribution, checked against enumeration.
    ///
    /// This compares a sampler against a CLOSED FORM over every state, not against another
    /// sampler. The draws are independent — nothing is a chain here — so the expected total
    /// variation for a correct sampler is about `sum_s sqrt(p_s (1-p_s) / (2 pi N))`, which is
    /// 0.0035 at these settings; the threshold is three times that.
    #[test]
    fn the_argmax_is_distributed_exactly_boltzmann() {
        let n = 4;
        let g = glass(n, 3);
        let beta = 0.8;
        let truth = ising::exact_boltzmann(&g, beta);
        let draws = 200_000;
        let mut r = Pcg::new(2024, 5);
        let mut count = vec![0u32; 1 << n];
        for _ in 0..draws {
            let d = perturb_and_map(&g, beta, &mut r).unwrap();
            count[mask_of(&d.state)] += 1;
        }
        let emp: Vec<f64> = count.iter().map(|&c| f64::from(c) / f64::from(draws)).collect();
        let tv = ising::tv(&emp, &truth);
        assert!(tv < 0.011, "total variation {tv} from the exact Boltzmann law");
    }

    /// The Gumbel-max argmax IS the exponential race, state for state, on the same uniforms.
    ///
    /// `theta + (-ln(-ln u))` is a strictly increasing transform of `-((-ln u) e^{-theta})`, so the
    /// two constructions must pick the same index — an algebraic identity, checked with no
    /// statistics at all. It pins the direction of both logarithms: flip either and the argmax
    /// becomes an argmin, which a distributional test would only notice as "the wrong shape".
    #[test]
    fn the_gumbel_argmax_is_the_exponential_race() {
        let mut r = Pcg::new(99, 4);
        for trial in 0..2_000 {
            let m = 12;
            let theta: Vec<f64> = (0..m).map(|_| 6.0 * r.f64() - 3.0).collect();
            let u: Vec<f64> = (0..m).map(|_| r.f64().max(f64::MIN_POSITIVE)).collect();

            let mut by_gumbel = (0usize, f64::NEG_INFINITY);
            let mut by_race = (0usize, f64::INFINITY);
            for i in 0..m {
                let v = theta[i] - (-u[i].ln()).ln();
                if v > by_gumbel.1 {
                    by_gumbel = (i, v);
                }
                let t = -u[i].ln() * (-theta[i]).exp();
                if t < by_race.1 {
                    by_race = (i, t);
                }
            }
            assert_eq!(by_gumbel.0, by_race.0, "trial {trial}");
        }
    }

    /// With nothing coupled, the all-dimensions bound is `log Z` exactly — mean and variance both.
    ///
    /// The maximum then splits into `n` independent one-spin maxima, so the estimator is unbiased
    /// for `sum_i log(2 cosh(beta h_i))` and its variance is exactly `n pi^2/6`. That makes an
    /// otherwise-stochastic bound checkable against a closed form, and it is the case that catches
    /// a beta the field was never multiplied by and a dropped `a_i` constant. Both survive a "the
    /// bound is above `log Z`" test, because being loose is what an upper bound is allowed to be.
    ///
    /// Exchanging `gamma_i(+1)` with `gamma_i(-1)` is NOT such a defect and nothing here catches
    /// it: the two are i.i.d., so the swap is a relabelling of the same distribution.
    #[test]
    fn the_all_dimensions_bound_is_exactly_log_z_when_nothing_is_coupled() {
        let n = 6;
        let mut r = Pcg::new(4, 4);
        let mut b = GraphBuilder::new(n);
        // Fields wide enough that beta is VISIBLE. At |h| < 0.5 the beta-scaled and unscaled
        // closed forms differ by 0.11 nats and four standard errors is 0.14, so a run that never
        // multiplied the field by beta passed this test; at |h| < 2 the gap is 1.8 nats.
        for i in 0..n {
            b.bias(i, 4.0 * r.f64() - 2.0);
        }
        let g = b.build();
        assert_eq!(g.n_edges, 0, "the closed form needs an uncoupled model");
        let beta = 1.2;
        let truth: f64 = (0..n).map(|i| (2.0 * (beta * g.h[i]).cosh()).ln()).sum();

        let el = Elimination::default();
        let est = perturbed_map_upper(&g, beta, 12_000, 17, &el).unwrap();
        assert!(
            (est.value - truth).abs() < 4.0 * est.stderr,
            "estimate {} +- {} against sum_i log 2cosh(beta h_i) = {truth}",
            est.value,
            est.stderr
        );
        let want = n as f64 * GUMBEL_VAR;
        assert!((est.variance - want).abs() < 0.9, "variance {} against {want}", est.variance);
    }

    /// The two low-order bounds land on the right sides of the exact `log Z`.
    ///
    /// The single-spin bound is deterministic, so it is asserted outright. The all-dimensions
    /// bound is an expectation, so what is asserted is that the estimate clears `log Z` by four of
    /// its own standard errors — a finite sample of a valid upper bound may dip below, and saying
    /// so is the difference between a bound and an estimate.
    #[test]
    fn the_all_dimensions_bound_is_above_log_z_and_the_single_spin_bound_below_it() {
        let (n, j, h, beta) = (10, 1.0, 0.2, 0.8);
        let g = ising::ring(n, j, h);
        let log_z = free_energy::ring_log_z(n, j, h, beta);
        let el = Elimination::default();

        let lo = best_single_site(&g, beta, &el).unwrap();
        assert_eq!(lo.summed.len(), 1);
        assert_eq!(lo.solves, 2);
        assert!(lo.lower <= log_z + 1e-9, "single-spin lower {} above {log_z}", lo.lower);
        assert!(lo.upper >= log_z - 1e-9, "single-spin upper {} below {log_z}", lo.upper);
        // It must also beat the zero-order bound, or the extra spin bought nothing.
        let map_only = map_ladder(&g, beta, &[], &el).unwrap();
        assert!(lo.lower > map_only.lower, "{} did not improve on {}", lo.lower, map_only.lower);

        let up = perturbed_map_upper(&g, beta, 4_000, 31, &el).unwrap();
        assert!(
            up.value - 4.0 * up.stderr > log_z,
            "all-dimensions estimate {} +- {} does not clear log Z {log_z}",
            up.value,
            up.stderr
        );
    }

    /// Full-order perturbation refuses what it cannot enumerate rather than approximating it.
    #[test]
    fn full_order_perturbation_refuses_a_model_it_cannot_enumerate() {
        let g = ising::ring(ENUMERATION_LIMIT + 1, 1.0, 0.0);
        let mut r = Pcg::new(1, 1);
        let want =
            Refused::TooLargeToEnumerate { spins: ENUMERATION_LIMIT + 1, limit: ENUMERATION_LIMIT };
        assert_eq!(perturb_and_map(&g, 1.0, &mut r).unwrap_err(), want);
        assert_eq!(gumbel_log_z(&g, 1.0, 4, 1).unwrap_err(), want);
    }
}
