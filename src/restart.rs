//! When to stop and start again: the expected work of a restart strategy from a run-length
//! distribution, the optimal fixed cutoff, and Luby's universal sequence — with the closed forms
//! that pin each of them.
//!
//! # Why this exists
//!
//! [`crate::rld`] measures a solver's run-length distribution, `F(t) = P(solved within t steps)`,
//! and [`crate::tts`] turns it into a time to solution at a confidence. Neither says what a RUN
//! should do: a heuristic whose success is front-loaded should be cut off and restarted, one whose
//! success is memoryless gains nothing from any restart, and one whose curve is unknown in advance
//! can still be run under Luby, Sinclair and Zuckerman's universal schedule (1993) at a bounded cost
//! over the best fixed cutoff in hindsight. Every quantity here is a functional of `F`, so every
//! test is a closed form.
//!
//! # The closed forms
//!
//! Restarting every `t` steps costs an expected `E[W_t] = sum_{k < t} (1 - F(k)) / F(t)`: the
//! expected steps of one attempt, over the probability an attempt succeeds. For a memoryless
//! (Bernoulli) solver with per-step success `p`, `F(k) = 1 - (1 - p)^k` and `E[W_t] = 1 / p` for
//! EVERY cutoff — restarts neither help nor hurt, which is what memoryless means. For a two-mode
//! solver that succeeds at the first step with probability `q` and otherwise never, the optimal
//! cutoff is 1 and its expected work `1 / q`, while any cutoff `t` costs `(1 + (t - 1)(1 - q)) / q`.
//! Luby's sequence is `1, 1, 2, 1, 1, 2, 4, 1, 1, 2, 1, 1, 2, 4, 8, ...`, `u(i) = 2^{k-1}` when
//! `i = 2^k - 1` and `u(i - 2^{k-1} + 1)` when `2^{k-1} <= i < 2^k - 1`.

/// The `i`-th term of Luby's universal restart sequence, `i >= 1`:
/// `1, 1, 2, 1, 1, 2, 4, 1, 1, 2, 1, 1, 2, 4, 8, ...`.
#[must_use]
pub fn luby(i: u64) -> u64 {
    debug_assert!(i >= 1, "Luby's sequence is indexed from 1");
    let mut i = i.max(1);
    loop {
        // The smallest k with 2^k - 1 >= i, i.e. k = ceil(log2(i + 1)): the bit length of i + 1,
        // less one when i + 1 is itself a power of two. (The first version took the bit length
        // unconditionally, which sent i = 1 to k = 2, subtracted 1, and looped on i = 0 forever.)
        let n = i + 1;
        let k = u64::from(64 - n.leading_zeros()) - u64::from(n.is_power_of_two());
        let block = (1u64 << k) - 1;
        if i == block {
            return 1u64 << (k - 1);
        }
        i -= (1u64 << (k - 1)) - 1;
    }
}

/// Luby's sequence scaled by `unit` steps: the cutoff of the `i`-th run.
#[must_use]
pub fn luby_cutoff(i: u64, unit: u64) -> u64 {
    luby(i) * unit
}

/// Geometrically growing cutoffs `base, base f, base f^2, ...` (rounded up), `n` of them.
#[must_use]
pub fn geometric_cutoffs(base: u64, factor: f64, n: usize) -> Vec<u64> {
    (0..n).map(|i| (base as f64 * factor.powi(i as i32)).ceil().max(1.0) as u64).collect()
}

/// Expected total work of restarting every `cutoff` steps under a success curve `f`, where
/// `f[k]` is the probability of success within `k` steps (`f[0] = 0`): the expected steps of one
/// attempt over the probability an attempt succeeds. `+inf` when `f[cutoff]` is 0.
///
/// # Panics
///
/// If `cutoff` is 0 or beyond the curve.
#[must_use]
pub fn expected_work(f: &[f64], cutoff: usize) -> f64 {
    assert!(cutoff >= 1 && cutoff < f.len(), "cutoff must be in 1..{}", f.len());
    let success = f[cutoff];
    if success <= 0.0 {
        return f64::INFINITY;
    }
    // A loop rather than a closure: the mutation suite's rows split on '|', so the line a row
    // names cannot hold a closure's parameter list.
    let mut steps = 0.0;
    for p in &f[..cutoff] {
        steps += 1.0 - p;
    }
    steps / success
}

/// The fixed cutoff that minimises [`expected_work`] over the curve, and that minimum.
#[must_use]
pub fn optimal_cutoff(f: &[f64]) -> (usize, f64) {
    (1..f.len())
        .map(|t| (t, expected_work(f, t)))
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(core::cmp::Ordering::Equal))
        .unwrap_or((1, f64::INFINITY))
}

/// Expected total work of Luby's schedule with `unit` steps per unit, under the curve `f`, summed
/// over the first `runs` runs: each run `i` succeeds with probability `f[min(cutoff_i, len - 1)]`
/// and costs its expected steps otherwise. Returns the expected work over the runs and the
/// probability the schedule has not succeeded by the end of them.
#[must_use]
pub fn luby_expected_work(f: &[f64], unit: u64, runs: u64) -> (f64, f64) {
    let mut alive = 1.0;
    let mut work = 0.0;
    for i in 1..=runs {
        let t = (luby_cutoff(i, unit) as usize).min(f.len() - 1);
        let steps: f64 = f[..t].iter().map(|p| 1.0 - p).sum();
        work += alive * steps;
        alive *= 1.0 - f[t];
    }
    (work, alive)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The sequence, term by term, and the block-sum identity `sum_{i <= 2^k - 1} u(i) = k 2^{k-1}`.
    #[test]
    fn lubys_sequence_is_the_published_one_and_its_blocks_sum_as_they_must() {
        let want = [1u64, 1, 2, 1, 1, 2, 4, 1, 1, 2, 1, 1, 2, 4, 8, 1, 1, 2, 1, 1, 2, 4, 1, 1, 2, 1, 1, 2, 4, 8, 16];
        for (i, &w) in want.iter().enumerate() {
            assert_eq!(luby(i as u64 + 1), w, "term {}", i + 1);
        }
        for k in 1..=10u64 {
            let sum: u64 = (1..=((1u64 << k) - 1)).map(luby).sum();
            assert_eq!(sum, k * (1u64 << (k - 1)), "block sum at k = {k}");
        }
    }

    /// A memoryless solver: every cutoff costs exactly `1 / p`, to rounding.
    #[test]
    fn a_memoryless_solver_gains_nothing_from_any_cutoff() {
        let p = 0.013f64;
        let f: Vec<f64> = (0..2000).map(|k| 1.0 - (1.0 - p).powi(k)).collect();
        for t in [1usize, 2, 7, 50, 400, 1999] {
            let w = expected_work(&f, t);
            assert!((w - 1.0 / p).abs() < 1e-9 / p, "cutoff {t}: {w} vs 1/p = {}", 1.0 / p);
        }
    }

    /// A front-loaded solver -- success at the first step with probability `q`, never after --
    /// has optimal cutoff 1 at work `1 / q`, and any longer cutoff costs `(1 + (t - 1)(1 - q)) / q`.
    #[test]
    fn a_front_loaded_solver_wants_the_shortest_cutoff() {
        let q = 0.2f64;
        let f: Vec<f64> = (0..100).map(|k| if k == 0 { 0.0 } else { q }).collect();
        let (t, w) = optimal_cutoff(&f);
        assert_eq!(t, 1);
        assert!((w - 1.0 / q).abs() < 1e-12);
        for t in [2usize, 5, 30] {
            let want = (1.0 + (t as f64 - 1.0) * (1.0 - q)) / q;
            assert!((expected_work(&f, t) - want).abs() < 1e-12, "cutoff {t}");
        }
        // Luby at unit 1 succeeds in the end and costs more than the optimal cutoff, less than a
        // cutoff of 30.
        let (work, alive) = luby_expected_work(&f, 1, 200);
        assert!(alive < 1e-12, "Luby must succeed: {alive}");
        assert!(work > 1.0 / q && work < expected_work(&f, 30), "Luby work {work}");
    }
}
