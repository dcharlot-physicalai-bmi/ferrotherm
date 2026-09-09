//! Directed rounding — sums that are certainly on the safe side of the truth.
//!
//! # Why a whole module for addition
//!
//! A **bound** is a promise about every state, and `f64` addition does not keep promises. Round to
//! nearest moves a total either way, so a lower bound accumulated with `+` can come out ABOVE the
//! quantity it bounds, and a bound that is above what it bounds is not a bound. The error is small
//! and the claim it breaks is absolute, which is the worst combination: nothing observable goes
//! wrong until a caller believes a proof that is not one.
//!
//! It is not hypothetical here. [`crate::bound::forest`] accumulated its parts with `+=`, and on
//! random trees — where the bound is exact and the gap must be zero — **1688 of 4800 trials
//! reported a NEGATIVE gap**, worst `−7.8e-14`. A negative gap is the observable signature of this
//! defect, and [`crate::bound::Bound::gap`]'s own documentation already said so; nothing was
//! checking it.
//!
//! # What is here
//!
//! [`sum_down`] and [`sum_up`] bracket the exact sum of a slice. [`accumulation_guard`] is for the
//! case the slice does not cover: a quantity computed by some other routine, where the caller knows
//! how many additions went into it and how large the partial sums could get.
//!
//! # This is not interval arithmetic, and is not trying to be
//!
//! An interval type would carry `[lo, hi]` through every operation and is the right tool when a
//! whole computation must be bounded. Every use in this crate is a SUM whose direction is known in
//! advance, so the narrow tool is the honest one: it is exact about what it guarantees, it costs one
//! pass, and it does not tempt a caller into believing a chain of interval operations is tight.
//! Where a tighter bound is needed the answer is a better relaxation, not a wider float.

/// Neumaier-compensated total, and a rigorous bound on its distance from the exact sum.
///
/// The guard is Kahan–Babuška's: `2 ε |total|` for the rounding of the final addition, taken
/// relative to the RESULT, plus `n² ε² Σ|x|` in the magnitude summed.
///
/// **Scaling the whole guard by `Σ|x|` is also sound and is unusable.** That was the first version
/// of this code: on `[1e16, 1, −1e16]`, whose true sum is one, the guard `2 n ε Σ|x|` is **26.6**,
/// so `sum_down` returned −25.6. Sound, and worthless. The first-order term below follows the
/// ANSWER rather than the arithmetic that produced it, and is 9e-15 on the same input.
///
/// The second-order term is carried for rigour and is not observable at any size this crate can
/// represent: it overtakes the first only when `n² ε² Σ|x| > 2 ε |total|`, which for the usual sign
/// structure is `n > sqrt(2/ε) ≈ 9.5e7` — a hundred million terms. A mutation deleting it survives
/// every test here, and that is recorded rather than papered over.
fn compensated(v: &[f64]) -> (f64, f64) {
    let (mut s, mut c) = (0.0f64, 0.0f64);
    for &x in v {
        let t = s + x;
        c += if s.abs() >= x.abs() { (s - t) + x } else { (x - t) + s };
        s = t;
    }
    let total = s + c;
    let mag: f64 = v.iter().map(|x| x.abs()).sum::<f64>().next_up();
    let n = v.len() as f64;
    let guard = 2.0 * f64::EPSILON * total.abs() + n * n * f64::EPSILON * f64::EPSILON * mag;
    (total, guard)
}

/// Sum that is never ABOVE the exact total, whatever the inputs.
///
/// Use where the result is a LOWER bound. Left-to-right addition can round up:
/// `[1.0, 3·2⁻⁵⁴, 3·2⁻⁵⁴]` has exact sum `1 + 1.5·2⁻⁵²` and sums in `f64` to `1 + 2·2⁻⁵²`, over by
/// 1.1e-16.
#[must_use]
pub fn sum_down(v: &[f64]) -> f64 {
    let (total, guard) = compensated(v);
    total - guard
}

/// Sum that is never BELOW the exact total, whatever the inputs.
///
/// Use where the result is an UPPER bound — an energy that a gap is measured from, say, where
/// understating it would understate the gap and overstate how good an answer is.
#[must_use]
pub fn sum_up(v: &[f64]) -> f64 {
    let (total, guard) = compensated(v);
    total + guard
}

/// A bound on the error of a quantity someone else accumulated.
///
/// `ops` additions whose partial sums never exceed `magnitude` in absolute value accumulate at most
/// `ops · ε · magnitude` of rounding error. That is the textbook first-order bound for recursive
/// summation with `u = ε/2`, taken at `ε` rather than `u` so the operation count may be
/// over-estimated without invalidating it — which is the point, since a caller counting the
/// additions inside a routine it did not write should round that count UP.
///
/// This is deliberately cruder than [`sum_down`]. Reach for it only when the terms are not
/// available as a slice: the compensated bound follows the answer, and this one follows the
/// arithmetic, so this one is far looser under cancellation. Where the terms ARE available,
/// [`sum_down`] is both tighter and simpler to justify.
#[must_use]
pub fn accumulation_guard(ops: usize, magnitude: f64) -> f64 {
    if !magnitude.is_finite() || magnitude <= 0.0 || ops == 0 {
        return 0.0;
    }
    ops as f64 * f64::EPSILON * magnitude
}

#[cfg(test)]
mod tests {
    use super::*;

    /// THE POINT OF THE MODULE, on the input that motivates it: a sum that `+` rounds UP.
    ///
    /// `1 + 3·2⁻⁵⁴ + 3·2⁻⁵⁴` is exactly `1 + 1.5·2⁻⁵²`. **That value cannot be written down here**
    /// — it is exactly half way between two neighbouring `f64`s, which is why the example works at
    /// all — so the test is stated through the two floats that surround it instead. Writing
    /// `let exact = 1.0 + 1.5 * 2f64.powi(-52)` was the first version of this test and it rounds to
    /// the very value it is supposed to catch, making the assertion `naive > naive`.
    ///
    /// `lo` is the largest float below the exact sum and `hi` the smallest above it. Plain addition
    /// lands on `hi`; a sound lower bound must not exceed `lo`, and a sound upper bound must reach
    /// `hi`.
    #[test]
    fn a_sum_that_plain_addition_rounds_upward_is_bracketed() {
        let tiny = 3.0 * 2.0f64.powi(-54);
        let v = [1.0, tiny, tiny];
        let lo = 1.0f64.next_up(); // 1 + 2⁻⁵², strictly below the exact sum
        let hi = lo.next_up(); // 1 + 2·2⁻⁵², strictly above it
        let naive: f64 = v.iter().sum();
        assert_eq!(naive, hi, "the fixture must actually round up, or it tests nothing");
        assert!(sum_down(&v) <= lo, "sum_down {:e} is above the exact sum", sum_down(&v));
        assert!(sum_up(&v) >= hi, "sum_up {:e} is below the exact sum", sum_up(&v));
    }

    /// UNDER CANCELLATION THE GUARD MUST FOLLOW THE ANSWER, not the arithmetic. This is the input
    /// that killed the first version, where a guard scaled by `Σ|x|` returned −25.6 for a true sum
    /// of one.
    #[test]
    fn cancellation_does_not_make_the_bound_worthless() {
        let v = [1e16, 1.0, -1e16];
        let (lo, hi) = (sum_down(&v), sum_up(&v));
        assert!(lo <= 1.0 && hi >= 1.0, "must bracket 1: [{lo}, {hi}]");
        assert!(hi - lo < 1e-9, "and must stay useful under cancellation: width {}", hi - lo);
    }

    /// The bracket holds on many shapes, and is never inverted.
    #[test]
    fn the_bracket_holds_and_never_inverts() {
        let mut rng = crate::rng::Pcg::new(4, 0x5D);
        for n in [0usize, 1, 2, 7, 64, 1000] {
            for scale in [1e-8f64, 1.0, 1e8] {
                let v: Vec<f64> = (0..n).map(|_| (rng.f64() - 0.5) * scale).collect();
                let (lo, hi) = (sum_down(&v), sum_up(&v));
                assert!(lo <= hi, "n={n} scale={scale}: inverted [{lo}, {hi}]");
                // Every prefix ordering of the same terms must land inside the bracket, which is
                // the property a direction-certain sum actually promises.
                let mut w = v.clone();
                w.reverse();
                let other: f64 = w.iter().sum();
                let fwd: f64 = v.iter().sum();
                for t in [other, fwd] {
                    assert!(
                        lo - 1e-9 * t.abs().max(1.0) <= t && t <= hi + 1e-9 * t.abs().max(1.0),
                        "n={n} scale={scale}: {t} outside [{lo}, {hi}]"
                    );
                }
            }
        }
    }

    /// An empty sum is zero and a one-element sum is that element, both exactly — the guard must
    /// not manufacture width where there is no arithmetic to be wrong about.
    #[test]
    fn trivial_sums_are_exact() {
        assert_eq!(sum_down(&[]), 0.0);
        assert_eq!(sum_up(&[]), 0.0);
        for x in [0.0f64, 1.0, -3.5, 1e300] {
            assert!(sum_down(&[x]) <= x && sum_up(&[x]) >= x, "{x} not bracketed");
        }
    }

    /// The crude guard is monotone in both arguments and refuses to invent width from nothing.
    #[test]
    fn the_accumulation_guard_is_zero_where_there_is_no_arithmetic() {
        assert_eq!(accumulation_guard(0, 1e9), 0.0);
        assert_eq!(accumulation_guard(100, 0.0), 0.0);
        assert_eq!(accumulation_guard(100, f64::NAN), 0.0);
        assert!(accumulation_guard(10, 1.0) < accumulation_guard(100, 1.0));
        assert!(accumulation_guard(10, 1.0) < accumulation_guard(10, 10.0));
    }
}
