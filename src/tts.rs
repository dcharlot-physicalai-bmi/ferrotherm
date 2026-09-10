//! Time-to-solution — the metric the Ising-machine field reports, and the one this crate did not.
//!
//! # Why a solver's best energy is not a result
//!
//! Every solver here returns the best state it found for a budget it was given. That number cannot
//! be compared with anything: a different machine, a different budget or a different seed makes it
//! a different number, and two solvers that both "found the optimum" may differ by orders of
//! magnitude in how often and how fast. The field settled on a metric that removes all three —
//! **TTS**, the expected work to find the optimum at least once with probability `s`, minimised
//! over the run length:
//!
//! ```text
//!   TTS(t) = t · ln(1 − s) / ln(1 − p(t))            TTS = min over t of TTS(t)
//! ```
//!
//! `p(t)` is the probability that one run of length `t` succeeds, estimated over a seed ensemble.
//! `s` is conventionally 0.99, which is why the number is often written R99.
//!
//! # The minimisation is the whole point, and reporting at one length is the usual error
//!
//! `TTS(t)` is not monotone. A run too short almost never succeeds, so the restarts multiply; a run
//! too long spends most of its budget after the answer was already found. The minimum sits between,
//! and **quoting `TTS` at a single arbitrary `t` overstates it, sometimes by a lot**. Rønnow et al.
//! (*Defining and detecting quantum speedup*, Science 345:420, 2014) made this the standard, and it
//! is the step that makes the metric comparable between machines rather than between budgets.
//! [`Tts::curve`] carries every length measured, so a reader can see the minimum instead of
//! trusting it.
//!
//! # Two degenerate cases that a naive implementation gets wrong
//!
//! **`p ≥ s`.** The formula gives `ln(1−s)/ln(1−p) < 1` — fewer than one run. You cannot do fewer
//! than one run: a single run at `p = 0.999` already clears a 0.99 target, so `TTS = t`, not
//! `0.43 t`. Written naively this reports a machine as faster than it can possibly be, and it is
//! worst exactly where a solver is doing well.
//!
//! **`p = 0`.** No estimate exists. `ln(1−0) = 0` and the quotient is infinite, which is the honest
//! answer — but it must be reported as "not solved at this length" rather than as a number, because
//! an infinity that reaches a table becomes a plot with a missing point and no explanation.
//!
//! # The interval, because twenty seeds is not a probability
//!
//! `p̂ = successes / trials` from twenty runs has a standard error near 0.11 at `p̂ = 0.5`, and TTS
//! inherits it. [`Tts::interval`] is the Wilson score interval on `p` mapped through the formula —
//! Wilson rather than the normal approximation because the normal one is worst at small `p`, which
//! is the regime a hard instance lives in and where it can produce a NEGATIVE lower bound.

/// One measured point: a run length, and how often it succeeded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Trial {
    /// Run length, in whatever unit the solver's budget is counted in — spin proposals here.
    pub length: u64,
    /// Runs that reached the target.
    pub successes: usize,
    /// Runs attempted.
    pub trials: usize,
}

/// What TTS came to, and everything needed to disbelieve it.
#[derive(Clone, Debug, PartialEq)]
pub struct Tts {
    /// The run length that minimises TTS.
    pub best_length: u64,
    /// TTS at that length, in the same unit as `length`. `None` when no length ever succeeded.
    pub tts: Option<f64>,
    /// Success probability at `best_length`.
    pub p_success: f64,
    /// A Wilson interval on TTS at `best_length`, at 95%. Wide is informative, not a failure.
    pub interval: Option<(f64, f64)>,
    /// The target probability, conventionally 0.99. Must be strictly inside `(0, 1)`.
    pub confidence: f64,
    /// `(length, p, TTS)` at every length measured, so the minimum can be seen rather than trusted.
    pub curve: Vec<(u64, f64, Option<f64>)>,
}

/// Expected runs to succeed once with probability `s`, given per-run probability `p`.
///
/// Returns `None` for `p == 0` — no finite number of runs achieves the target — and clamps at one
/// run for `p >= s`, which is the case a naive formula reports as a fraction of a run.
#[must_use]
pub fn runs_needed(p: f64, s: f64) -> Option<f64> {
    if !(0.0..=1.0).contains(&p) || !(s > 0.0 && s < 1.0) {
        return None;
    }
    if p <= 0.0 {
        return None;
    }
    if p >= s {
        // One run already clears the target. The formula's answer here is below 1, and a run count
        // below one is not a smaller amount of work, it is a wrong reading of the question.
        return Some(1.0);
    }
    Some((1.0 - s).ln() / (1.0 - p).ln())
}

/// The Wilson score interval for a binomial proportion, at `z` standard normal deviates.
///
/// Chosen over the normal approximation because the latter is worst at small `p` — where a hard
/// instance lives — and can put the lower end BELOW ZERO, which then maps to a negative TTS.
#[must_use]
pub fn wilson(successes: usize, trials: usize, z: f64) -> (f64, f64) {
    if trials == 0 {
        return (0.0, 1.0);
    }
    let n = trials as f64;
    let phat = successes as f64 / n;
    let z2 = z * z;
    let denom = 1.0 + z2 / n;
    let centre = (phat + z2 / (2.0 * n)) / denom;
    let half = (z / denom) * (phat * (1.0 - phat) / n + z2 / (4.0 * n * n)).sqrt();
    ((centre - half).max(0.0), (centre + half).min(1.0))
}

/// Time-to-solution over a set of measured run lengths, minimised over length.
///
/// # Errors
///
/// Returns `None` when `trials` is empty or `confidence` is not in `(0, 1)`.
#[must_use]
pub fn tts(trials: &[Trial], confidence: f64) -> Option<Tts> {
    if trials.is_empty() || !(confidence > 0.0 && confidence < 1.0) {
        return None;
    }
    let mut curve = Vec::with_capacity(trials.len());
    for t in trials {
        let p = if t.trials == 0 { 0.0 } else { t.successes as f64 / t.trials as f64 };
        let v = runs_needed(p, confidence).map(|r| r * t.length as f64);
        curve.push((t.length, p, v));
    }
    // The minimum over lengths that produced one. A length that never succeeded is not a large
    // TTS to be beaten, it is an absent measurement, and averaging it in as infinity would make
    // the minimum depend on which hopeless lengths happened to be tried.
    let best = curve
        .iter()
        .enumerate()
        .filter_map(|(i, (_, _, v))| v.map(|v| (i, v)))
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(core::cmp::Ordering::Equal));

    let (bi, bv) = match best {
        Some(x) => (x.0, Some(x.1)),
        // Nothing succeeded anywhere: report the longest length tried, so the caller can see how
        // far the search got before giving up.
        None => (curve.len() - 1, None),
    };
    let t = trials[bi];
    let p = curve[bi].1;
    let interval = bv.and_then(|_| {
        let (lo, hi) = wilson(t.successes, t.trials, 1.96);
        // TTS falls as p rises, so the interval flips: the high p gives the low TTS.
        let a = runs_needed(hi, confidence).map(|r| r * t.length as f64);
        let b = runs_needed(lo, confidence).map(|r| r * t.length as f64);
        match (a, b) {
            (Some(a), Some(b)) => Some((a.min(b), a.max(b))),
            _ => None,
        }
    });
    Some(Tts {
        best_length: t.length,
        tts: bv,
        p_success: p,
        interval,
        confidence,
        curve,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The formula against hand-computable values, and the two degenerate cases.
    #[test]
    fn runs_needed_matches_the_closed_form_and_clamps_where_it_must() {
        // p = 0.5, s = 0.99: ln(0.01)/ln(0.5) = 6.6438...
        let r = runs_needed(0.5, 0.99).unwrap();
        assert!((r - (0.01f64).ln() / (0.5f64).ln()).abs() < 1e-12, "{r}");
        assert!((r - 6.643_856).abs() < 1e-5, "{r}");

        // p exactly at the target: one run, not the 1.0 the formula happens to give.
        assert!((runs_needed(0.99, 0.99).unwrap() - 1.0).abs() < 1e-12);
        // p ABOVE the target: still one run. The formula gives 0.43 here, which would report
        // less work than a single run costs.
        let naive = (0.01f64).ln() / (1.0 - 0.999f64).ln();
        assert!(naive < 1.0, "the fixture must exercise the clamp: {naive}");
        assert!((runs_needed(0.999, 0.99).unwrap() - 1.0).abs() < 1e-12);
        // Never succeeded: no finite number of runs achieves the target.
        assert_eq!(runs_needed(0.0, 0.99), None);
    }

    /// TTS(t) is not monotone, so the metric is the MINIMUM over run length. A synthetic curve with
    /// a known minimum in the interior is what says the search finds it rather than taking an end.
    #[test]
    fn the_minimum_over_run_length_is_found_rather_than_an_endpoint() {
        // p(t) chosen so TTS(t) = t * ln(0.01)/ln(1-p) dips at t = 400.
        let trials = vec![
            Trial { length: 100, successes: 1, trials: 100 },   // p=0.01, huge TTS
            Trial { length: 200, successes: 10, trials: 100 },  // p=0.10
            Trial { length: 400, successes: 60, trials: 100 },  // p=0.60  <- the dip
            Trial { length: 800, successes: 80, trials: 100 },  // p=0.80, but twice the length
            Trial { length: 3200, successes: 95, trials: 100 }, // p=0.95, eight times
        ];
        let t = tts(&trials, 0.99).unwrap();
        assert_eq!(t.best_length, 400, "the minimum is interior: {:?}", t.curve);
        assert!((t.tts.unwrap() - 2010.35).abs() < 0.1, "TTS at the dip: {:?}", t.tts);
        // Every length must appear, so a reader can see the curve rather than trust the minimum.
        assert_eq!(t.curve.len(), trials.len());
        // And quoting the LONGEST length instead would overstate it, which is the usual error.
        let at_longest = t.curve.last().unwrap().2.unwrap();
        assert!(at_longest > 2.4 * t.tts.unwrap(), "{at_longest} against {:?}", t.tts);
    }

    /// A length that never succeeded is an absent measurement, not a large TTS. Including it as
    /// infinity would make the minimum depend on which hopeless lengths happened to be tried.
    #[test]
    fn a_length_that_never_succeeded_is_absent_rather_than_infinite() {
        let trials = vec![
            Trial { length: 10, successes: 0, trials: 50 },
            Trial { length: 1000, successes: 25, trials: 50 },
        ];
        let t = tts(&trials, 0.99).unwrap();
        assert_eq!(t.best_length, 1000);
        assert_eq!(t.curve[0].2, None, "a zero-success length has no TTS");
        assert!(t.tts.is_some());

        // Nothing succeeded anywhere: no TTS at all, and the longest length tried is reported.
        let none = vec![Trial { length: 7, successes: 0, trials: 20 }];
        let t = tts(&none, 0.99).unwrap();
        assert_eq!(t.tts, None);
        assert_eq!(t.best_length, 7);
    }

    /// The Wilson interval must stay inside `[0, 1]` at the extremes, which is exactly where the
    /// normal approximation goes negative and maps to a negative TTS.
    #[test]
    fn the_wilson_interval_stays_in_range_where_the_normal_one_does_not() {
        let (lo, hi) = wilson(0, 20, 1.96);
        assert!(lo >= 0.0 && hi <= 1.0 && hi > 0.0, "({lo}, {hi})");
        // The normal approximation at 0 successes gives a zero-width interval at zero, and at 1
        // success it goes negative: p̂ ± z·sqrt(p̂(1-p̂)/n) = 0.05 − 1.96·0.0487 = −0.045.
        let phat: f64 = 1.0 / 20.0;
        let naive_lo = phat - 1.96 * (phat * (1.0 - phat) / 20.0).sqrt();
        assert!(naive_lo < 0.0, "the fixture must exercise the difference: {naive_lo}");
        let (lo, hi) = wilson(1, 20, 1.96);
        assert!(lo > 0.0 && hi < 1.0, "({lo}, {hi})");
        assert!(lo <= phat && phat <= hi, "the interval must contain the estimate");

        // Ordering: TTS falls as p rises, so the interval on TTS flips relative to the one on p.
        let t = tts(&[Trial { length: 100, successes: 10, trials: 40 }], 0.99).unwrap();
        let (a, b) = t.interval.unwrap();
        assert!(a < t.tts.unwrap() && t.tts.unwrap() < b, "{a} {:?} {b}", t.tts);
    }

    /// Refusals rather than a number nobody can read.
    #[test]
    fn malformed_input_is_refused() {
        assert!(tts(&[], 0.99).is_none());
        let one = [Trial { length: 1, successes: 1, trials: 1 }];
        assert!(tts(&one, 1.0).is_none(), "a confidence of 1 needs infinitely many runs");
        assert!(tts(&one, 0.0).is_none());
        assert_eq!(runs_needed(1.5, 0.99), None, "a probability above one is not one");
        assert_eq!(wilson(0, 0, 1.96), (0.0, 1.0), "no trials means no information");
    }
}
