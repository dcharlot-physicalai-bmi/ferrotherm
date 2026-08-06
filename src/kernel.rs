//! The update. One implementation, every caller.
//!
//! A binary stochastic neuron is resampled from a sigmoid of its local field:
//!
//! ```text
//!     P(s_i = +1 | rest) = sigma( 2 beta f_i ),   f_i = sum_j J_ij s_j + h_i
//! ```
//!
//! That single line is the whole computation this crate performs, and before this module it was
//! written out six times across five files in three different spellings of beta -- including one
//! that had no beta at all, because it had been folded into the weights upstream. Six copies of one
//! equation is six chances for them to disagree, and the one without beta cannot be annealed
//! without rewriting every weight.
//!
//! So: nothing in this crate re-derives the update. Samplers decide *which* site to visit and *in
//! what order*; the arithmetic of visiting one lives here.
//!
//! Beta is always a parameter and never a weight. Annealing changes a number, never a program.

use crate::rng::Pcg;

/// Probability that site `i` lands on +1, given its local field and inverse temperature.
///
/// `field` is `sum_j J_ij s_j + h_i` with beta *excluded*; beta is applied here.
#[inline]
pub fn p_up(field: f64, beta: f64) -> f64 {
    // Written as the logistic of 2*beta*f. The factor of two is the gap between the two states of
    // a +/-1 spin: flipping s_i changes the energy by 2*f_i, not f_i. Dropping it is the classic
    // way to sample a distribution at half the temperature you meant.
    1.0 / (1.0 + (-2.0 * beta * field).exp())
}

/// Draw the new value of a site.
#[inline]
pub fn draw(field: f64, beta: f64, rng: &mut Pcg) -> i8 {
    if rng.f64() < p_up(field, beta) {
        1
    } else {
        -1
    }
}

/// Contribution to `d log p(s_new) / d h_i` from this update.
///
/// This is what makes a program of these updates differentiable, and it is the reason the
/// differentiable-program layer used to carry its own copy of the sweep: it needs the score
/// alongside the draw. Exposing it here means it can have the draw *and* the shared kernel.
#[inline]
pub fn score_dh(field: f64, beta: f64, s_new: i8) -> f64 {
    // log p(s) = log sigma(2 beta f s)  =>  d/dh = 2 beta s sigma(-2 beta f s)
    let s = s_new as f64;
    let arg = 2.0 * beta * field * s;
    2.0 * beta * s / (1.0 + arg.exp())
}

/// Energy change if site `i` were to flip from its current value, given its local field.
#[inline]
pub fn delta_e(field: f64, s_i: i8) -> f64 {
    2.0 * field * s_i as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_field_is_a_fair_coin() {
        assert_eq!(p_up(0.0, 1.0), 0.5);
        assert_eq!(p_up(0.0, 100.0), 0.5);
    }

    #[test]
    fn beta_is_a_parameter_not_a_weight() {
        // The whole point of the module: scaling the field and scaling beta are the same thing, so
        // annealing never needs to touch a weight.
        for &(f, b) in &[(0.3, 2.0), (-1.2, 0.5), (2.5, 0.25)] {
            let via_beta = p_up(f, b);
            let via_weights = p_up(f * b, 1.0);
            assert!((via_beta - via_weights).abs() < 1e-15, "f={f} beta={b}");
        }
    }

    #[test]
    fn colder_is_more_decided() {
        let mut last = p_up(0.7, 0.1);
        for &b in &[0.5, 1.0, 2.0, 8.0] {
            let p = p_up(0.7, b);
            assert!(p > last, "beta {b} should be more decided than the last");
            last = p;
        }
        assert!(p_up(0.7, 1e3) > 0.999999);
    }

    #[test]
    fn saturates_without_overflow() {
        // exp() of a large negative argument is the failure mode here; it must not produce NaN.
        for &(f, b) in &[(1e6, 1.0), (-1e6, 1.0), (1.0, 1e6), (1.0, -1e6)] {
            let p = p_up(f, b);
            assert!(p.is_finite() && (0.0..=1.0).contains(&p), "f={f} beta={b} -> {p}");
        }
    }

    #[test]
    fn score_matches_a_numerical_derivative() {
        // If this drifts from p_up, the gradient estimators are silently wrong.
        let (beta, h) = (0.8, 1e-6);
        for &f in &[-1.5, -0.3, 0.0, 0.4, 2.2] {
            for &s in &[1i8, -1] {
                let lp = |x: f64| {
                    let p = p_up(x, beta);
                    (if s > 0 { p } else { 1.0 - p }).ln()
                };
                let numeric = (lp(f + h) - lp(f - h)) / (2.0 * h);
                let analytic = score_dh(f, beta, s);
                assert!(
                    (numeric - analytic).abs() < 1e-6,
                    "f={f} s={s}: numeric {numeric} vs analytic {analytic}"
                );
            }
        }
    }

    #[test]
    fn draw_reproduces_the_probability() {
        let mut rng = Pcg::new(42, 1);
        let (f, beta) = (0.35, 1.3);
        let n = 400_000;
        let ups = (0..n).filter(|_| draw(f, beta, &mut rng) > 0).count();
        let got = ups as f64 / n as f64;
        let want = p_up(f, beta);
        // three sigma of a binomial at this n
        let tol = 3.0 * (want * (1.0 - want) / n as f64).sqrt();
        assert!((got - want).abs() < tol, "got {got}, want {want}, tol {tol}");
    }

    #[test]
    fn delta_e_agrees_with_the_acceptance_ratio() {
        // Detailed balance ties the two together: p(+1)/p(-1) = exp(-beta * dE(-1 -> +1)).
        let (f, beta) = (0.6, 0.9);
        let p = p_up(f, beta);
        let ratio = p / (1.0 - p);
        let de = delta_e(f, -1); // energy change flipping -1 to +1
        assert!((ratio - (-beta * de).exp()).abs() < 1e-12);
    }
}
