//! Thermodynamic length — the Fisher metric on the inverse-temperature axis, and the ladder that
//! places rungs at equal length along it.
//!
//! Salamon & Berry (1983); Crooks, "Measuring thermodynamic length", PRL 99:100602 (2007).
//!
//! # The geometry, in one line
//!
//! A family of Boltzmann distributions parameterised by `β` is a curve in distribution space, and
//! the Fisher information of that family with respect to `β` is exactly `Var(E)`. So the curve has
//! a metric, its line element is `sqrt(Var(E)) dβ`, and the distance between two temperatures is
//!
//! ```text
//!   L(β₀ → β₁) = ∫ sqrt(Var_β(E)) dβ
//! ```
//!
//! That integral is what a replica has to cross to get from one rung to the next, and it is what
//! sets the swap acceptance: for adjacent rungs separated by `δL`, the Gaussian estimate is
//! `erfc(δL/2)` ([`expected_acceptance`], derived in its doc). **Equal length is therefore equal
//! acceptance**, which is the condition [`crate::adaptive`] chases by measuring swap rates and
//! iterating, and which [`crate::free_energy::linear_ladder`] and
//! [`crate::tempering::geometric_ladder`] do not aim at at all — they are arithmetic and geometric
//! in `β`, and neither has looked at the model.
//!
//! # What is here
//!
//! [`exact_dos`] enumerates the spectrum, [`variance`] evaluates the metric on it, and
//! [`LengthProfile`] integrates it and inverts the integral, which is all a ladder is:
//! [`equal_length_ladder`] is `beta_at(k·L/K)`. [`LengthProfile::from_traces`] does the same from
//! measured energies when the model is too large to enumerate.
//!
//! [`exact_swap_acceptance`] computes the acceptance of a rung pair exactly, by the same rule
//! [`crate::tempering::parallel_tempering`] samples with, so the claim that equal length gives
//! uniform acceptance can be checked against an oracle as well as measured.
//!
//! # Where equal length stops being equal acceptance, measured
//!
//! `erfc(δL/2)` is a Gaussian statement, and its error is first order in `δL` with a coefficient set
//! by how far from Gaussian the energy distribution is. On a 13-spin model with 8,192 distinct
//! levels that coefficient is 0.07; on a 4×4 lattice, whose spectrum is 15 levels four apart, it is
//! 0.55 — and at the frozen end of a ladder, where two or three levels carry all the weight, equal
//! length over-accepts outright: the exact acceptances of an 8-rung equal-length ladder on that
//! lattice over `β ∈ [0, 3]` are `0.63, 0.63, 0.62, 0.61, 0.63, 0.72, 0.88`. That is a spread of
//! 0.27 against the linear ladder's 0.97 on the same span, which is the claim; it is not
//! uniformity.

use crate::graph::Graph;
use crate::samples::{Refused, enumerate};
use crate::wanglandau::Dos;

/// Grid points [`equal_length_ladder`] integrates the metric on.
///
/// The rule is trapezoidal, so its error falls as `1/grid²`; a thousand points puts a ladder's rung
/// positions well inside the noise of any swap rate measured from them.
pub const DEFAULT_GRID: usize = 1025;

/// The exact density of states by enumeration: every distinct energy with its count.
///
/// A [`Dos`] with `steps = 0`, because no walk happened — this is the object
/// [`crate::wanglandau`] estimates, computed instead.
///
/// # Errors
///
/// [`Refused::TooLargeToEnumerate`] past [`crate::samples::ENUMERATION_LIMIT`].
pub fn exact_dos(g: &Graph) -> Result<Dos, Refused> {
    let energies = enumerate(g, 0.0)?.energies().to_vec();
    // A level is "the same energy" within a billionth of the model's own scale, which is
    // `crate::wanglandau::Wl`'s rule; two states this close are one level to any sampler.
    let scale =
        g.w.iter().map(|x| x.abs()).sum::<f64>() / 2.0 + g.h.iter().map(|x| x.abs()).sum::<f64>();
    let quantum = scale.max(1.0) * 1e-9;
    let mut keyed: Vec<(i64, f64)> = energies.iter().map(|&e| ((e / quantum).round() as i64, e)).collect();
    keyed.sort_by_key(|a| a.0);
    let (mut energy, mut log_g) = (Vec::new(), Vec::new());
    let mut i = 0;
    while i < keyed.len() {
        let mut j = i;
        while j < keyed.len() && keyed[j].0 == keyed[i].0 {
            j += 1;
        }
        energy.push(keyed[i].1);
        log_g.push(((j - i) as f64).ln());
        i = j;
    }
    Ok(Dos { energy, log_g, steps: 0 })
}

/// Normalised probability of each level at `beta`.
fn probabilities(dos: &Dos, beta: f64) -> Vec<f64> {
    let w: Vec<f64> = dos.log_g.iter().zip(&dos.energy).map(|(lg, e)| lg - beta * e).collect();
    let m = w.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !m.is_finite() {
        return vec![0.0; w.len()];
    }
    let z: f64 = w.iter().map(|x| (x - m).exp()).sum();
    w.iter().map(|x| (x - m).exp() / z).collect()
}

/// `⟨E⟩` at `beta`, from the density of states.
#[must_use]
pub fn mean_energy(dos: &Dos, beta: f64) -> f64 {
    probabilities(dos, beta).iter().zip(&dos.energy).map(|(p, e)| p * e).sum()
}

/// `Var_β(E)` — the Fisher information of the Boltzmann family with respect to `β`, and the square
/// of the metric's line element.
///
/// Two passes, `Σ p (E − ⟨E⟩)²` rather than `⟨E²⟩ − ⟨E⟩²`. The one-pass form cancels catastrophically
/// exactly where a ladder's cold end lives: at `βh = 20` on six free spins the true variance is
/// `1.02e-16` while `⟨E⟩²` is 36, so `⟨E²⟩ − ⟨E⟩²` returns **exactly zero** while this returns the
/// truth to a relative `4e-16`. There is a test that pins it.
#[must_use]
pub fn variance(dos: &Dos, beta: f64) -> f64 {
    let p = probabilities(dos, beta);
    let m: f64 = p.iter().zip(&dos.energy).map(|(p, e)| p * e).sum();
    p.iter().zip(&dos.energy).map(|(p, e)| p * (e - m) * (e - m)).sum::<f64>().max(0.0)
}

/// `sqrt(Var_β(E))` — the metric's line element, the speed at which length accumulates in `β`.
#[must_use]
pub fn speed(dos: &Dos, beta: f64) -> f64 {
    variance(dos, beta).sqrt()
}

/// The metric sampled on a `β` grid, with its running integral: everything a ladder needs.
///
/// The fields are private because `cumulative` must be the integral of `speeds` — a caller who set
/// them independently would get a ladder built on a curve that is not the one it reports.
#[derive(Clone, Debug)]
pub struct LengthProfile {
    betas: Vec<f64>,
    speeds: Vec<f64>,
    cumulative: Vec<f64>,
}

impl LengthProfile {
    /// A profile from a sampled metric: `speeds[i] = sqrt(Var(E))` at `betas[i]`.
    ///
    /// The running integral is trapezoidal, which is what makes this usable from measured
    /// variances at a handful of rungs as well as from a fine exact grid.
    ///
    /// # Errors
    ///
    /// A message when the two slices disagree in length, hold fewer than two points, are not
    /// finite, are not strictly increasing in `β`, or carry a negative speed.
    pub fn from_speed(betas: &[f64], speeds: &[f64]) -> Result<LengthProfile, String> {
        if betas.len() != speeds.len() {
            return Err(format!("{} betas against {} speeds", betas.len(), speeds.len()));
        }
        if betas.len() < 2 {
            return Err("a profile needs at least two grid points to integrate between".into());
        }
        if betas.windows(2).any(|w| !(w[1] > w[0])) || betas.iter().any(|b| !b.is_finite()) {
            return Err("the grid must be finite and strictly increasing in beta".into());
        }
        if speeds.iter().any(|v| !(*v >= 0.0) || !v.is_finite()) {
            return Err("a speed is sqrt(Var(E)) and cannot be negative, infinite or NaN".into());
        }
        let mut cumulative = vec![0.0; betas.len()];
        for i in 1..betas.len() {
            cumulative[i] = cumulative[i - 1] + 0.5 * (speeds[i] + speeds[i - 1]) * (betas[i] - betas[i - 1]);
        }
        Ok(LengthProfile { betas: betas.to_vec(), speeds: speeds.to_vec(), cumulative })
    }

    /// The metric of `dos` on a uniform grid of `grid` points spanning `beta_lo ..= beta_hi`.
    ///
    /// # Panics
    ///
    /// If `grid` is below 2 or the span is not increasing and finite.
    #[must_use]
    pub fn from_dos(dos: &Dos, beta_lo: f64, beta_hi: f64, grid: usize) -> LengthProfile {
        assert!(grid >= 2, "a profile needs at least two grid points");
        assert!(beta_hi > beta_lo && beta_lo.is_finite() && beta_hi.is_finite(), "bad span");
        let k = (grid - 1) as f64;
        let betas: Vec<f64> = (0..grid)
            .map(|i| if i + 1 == grid { beta_hi } else { beta_lo + (beta_hi - beta_lo) * i as f64 / k })
            .collect();
        let speeds: Vec<f64> = betas.iter().map(|&b| speed(dos, b)).collect();
        LengthProfile::from_speed(&betas, &speeds).expect("a uniform grid of a real metric")
    }

    /// [`LengthProfile::from_dos`] on the enumerated spectrum of `g`.
    ///
    /// # Errors
    ///
    /// [`Refused::TooLargeToEnumerate`] past [`crate::samples::ENUMERATION_LIMIT`].
    ///
    /// # Panics
    ///
    /// If `grid` is below 2 or the span is not increasing and finite.
    pub fn exact(g: &Graph, beta_lo: f64, beta_hi: f64, grid: usize) -> Result<LengthProfile, Refused> {
        Ok(LengthProfile::from_dos(&exact_dos(g)?, beta_lo, beta_hi, grid))
    }

    /// A profile from measured energy traces — `(β, energies)` pairs, the shape
    /// [`crate::tempering::LadderTraces::as_pairs`] returns.
    ///
    /// The speed at each rung is the sample standard deviation of that rung's trace, so this is the
    /// same geometry read off a run instead of an enumeration. It is an ESTIMATE: a trace whose
    /// chain has not decorrelated understates the variance, and understating the variance shortens
    /// the length and spaces the rungs too far apart.
    ///
    /// # Errors
    ///
    /// A message when there are fewer than two rungs, a rung has fewer than two samples, or the
    /// rungs are not strictly increasing in `β`.
    pub fn from_traces(traces: &[(f64, Vec<f64>)]) -> Result<LengthProfile, String> {
        if traces.iter().any(|(_, e)| e.len() < 2) {
            return Err("a sample variance needs at least two draws at every rung".into());
        }
        let betas: Vec<f64> = traces.iter().map(|(b, _)| *b).collect();
        let speeds: Vec<f64> = traces
            .iter()
            .map(|(_, e)| {
                let n = e.len() as f64;
                let m = e.iter().sum::<f64>() / n;
                (e.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (n - 1.0)).sqrt()
            })
            .collect();
        LengthProfile::from_speed(&betas, &speeds)
    }

    /// The grid this was sampled on.
    #[must_use]
    pub fn betas(&self) -> &[f64] {
        &self.betas
    }

    /// `sqrt(Var(E))` at each grid point.
    #[must_use]
    pub fn speeds(&self) -> &[f64] {
        &self.speeds
    }

    /// Length accumulated from the hot end to each grid point.
    #[must_use]
    pub fn cumulative(&self) -> &[f64] {
        &self.cumulative
    }

    /// The total thermodynamic length of the span.
    ///
    /// # Panics
    ///
    /// Never: a profile always holds at least two points.
    #[must_use]
    pub fn length(&self) -> f64 {
        *self.cumulative.last().unwrap()
    }

    /// Length from the hot end to `beta`, by linear interpolation on the grid; clamped at the ends.
    ///
    /// A NaN `beta` is at the hot end, not past the cold one: `total_cmp` sorts NaN above every
    /// number, so letting it reach the search would read one past the grid.
    #[must_use]
    pub fn length_at(&self, beta: f64) -> f64 {
        let last = self.betas.len() - 1;
        if !(beta > self.betas[0]) {
            return 0.0;
        }
        if beta >= self.betas[last] {
            return self.length();
        }
        let i = match self.betas.binary_search_by(|b| b.total_cmp(&beta)) {
            Ok(i) => return self.cumulative[i],
            Err(i) => i - 1,
        };
        let span = self.betas[i + 1] - self.betas[i];
        let frac = if span > 0.0 { (beta - self.betas[i]) / span } else { 0.0 };
        self.cumulative[i] + frac * (self.cumulative[i + 1] - self.cumulative[i])
    }

    /// The `β` at length `l` from the hot end — the inverse of [`LengthProfile::length_at`],
    /// clamped to the span.
    #[must_use]
    pub fn beta_at(&self, l: f64) -> f64 {
        let last = self.betas.len() - 1;
        if !(l > 0.0) {
            return self.betas[0];
        }
        if l >= self.length() {
            return self.betas[last];
        }
        let i = match self.cumulative.binary_search_by(|c| c.total_cmp(&l)) {
            Ok(i) => return self.betas[i],
            Err(i) => i - 1,
        };
        let span = self.cumulative[i + 1] - self.cumulative[i];
        let frac = if span > 0.0 { (l - self.cumulative[i]) / span } else { 0.0 };
        self.betas[i] + frac * (self.betas[i + 1] - self.betas[i])
    }

    /// A ladder of `rungs` betas at equal thermodynamic length, ends pinned to the span.
    ///
    /// Ties are spread rather than returned: a stretch of `β` where the metric vanishes has no
    /// length to divide, and two equal rungs make `Δβ = 0`, which is a swap that always accepts and
    /// a [`crate::free_energy`] ladder that panics. Such rungs are spaced uniformly in `β` instead,
    /// which is the only information left there.
    ///
    /// # Panics
    ///
    /// If `rungs` is below 2.
    #[must_use]
    pub fn ladder(&self, rungs: usize) -> Vec<f64> {
        assert!(rungs >= 2, "a ladder needs two rungs to have a step");
        let k = (rungs - 1) as f64;
        let total = self.length();
        let mut out: Vec<f64> = (0..rungs).map(|i| self.beta_at(total * i as f64 / k)).collect();
        out[0] = self.betas[0];
        out[rungs - 1] = *self.betas.last().unwrap();
        spread_ties(&mut out);
        out
    }

    /// The thermodynamic length of each adjacent pair of `betas` — what
    /// [`expected_acceptance`] turns into a predicted swap rate.
    #[must_use]
    pub fn rung_lengths(&self, betas: &[f64]) -> Vec<f64> {
        betas.windows(2).map(|w| self.length_at(w[1]) - self.length_at(w[0])).collect()
    }
}

/// Make a non-decreasing ladder strictly increasing by spacing each tied run uniformly in `β`.
///
/// Terminates and always succeeds because the last entry is strictly the largest: every interior
/// rung is `beta_at(l)` for `l < L`, and that is strictly below `beta_hi` — so a tied run always
/// has something above it to be spread against.
fn spread_ties(out: &mut [f64]) {
    let n = out.len();
    let mut i = 1;
    while i + 1 < n {
        if out[i] > out[i - 1] {
            i += 1;
            continue;
        }
        // The run [i, j) is tied to out[i-1], and out[j] is the first value strictly above it.
        let mut j = i;
        while j + 1 < n && out[j] <= out[i - 1] {
            j += 1;
        }
        let (lo, hi) = (out[i - 1], out[j]);
        let steps = (j - i + 1) as f64;
        for (t, k) in (i..j).enumerate() {
            out[k] = lo + (hi - lo) * (t + 1) as f64 / steps;
        }
        i = j;
    }
}

/// The equal-thermodynamic-length ladder of `g` across `beta_lo ..= beta_hi`, by enumeration.
///
/// The geometric counterpart of [`crate::free_energy::linear_ladder`] and
/// [`crate::tempering::geometric_ladder`]: those space rungs by arithmetic on `β`, this spaces them
/// by the distance the model actually has between them.
///
/// # Errors
///
/// [`Refused::TooLargeToEnumerate`] past [`crate::samples::ENUMERATION_LIMIT`]; use
/// [`LengthProfile::from_traces`] there.
///
/// # Panics
///
/// If `rungs` is below 2 or the span is not increasing and finite.
pub fn equal_length_ladder(g: &Graph, beta_lo: f64, beta_hi: f64, rungs: usize) -> Result<Vec<f64>, Refused> {
    Ok(LengthProfile::exact(g, beta_lo, beta_hi, DEFAULT_GRID)?.ladder(rungs))
}

/// The exact swap acceptance between two independent equilibrium replicas at `beta_a` and
/// `beta_b`, from the density of states.
///
/// ```text
///   A = Σ_{a,b} p_a(β_a) p_b(β_b) · min(1, exp((β_b − β_a)(E_b − E_a)))
/// ```
///
/// the same rule [`crate::tempering::parallel_tempering`] draws against, summed over levels instead
/// of sampled. Linear in the level count, not quadratic: the levels are sorted, so the `min` splits
/// at `E_b = E_a` and each half telescopes — and the exponentials are accumulated relative to the
/// running level, so nothing overflows at large `β` where the raw form does.
///
/// Clamped into `[0, 1]`: this is a probability, and equal rungs accept always, which the sum
/// reaches as `1 + 9e-16`.
#[must_use]
pub fn exact_swap_acceptance(dos: &Dos, beta_a: f64, beta_b: f64) -> f64 {
    if beta_b < beta_a {
        return exact_swap_acceptance(dos, beta_b, beta_a);
    }
    let d = beta_b - beta_a;
    let p = probabilities(dos, beta_a);
    let q = probabilities(dos, beta_b);
    let e = &dos.energy;
    let n = e.len();
    // Suffix sums of q: the states the swap accepts outright, E_b >= E_a.
    let mut suffix = vec![0.0; n + 1];
    for i in (0..n).rev() {
        suffix[i] = suffix[i + 1] + q[i];
    }
    // acc = Σ_{b<a} q_b exp(d (E_b − E_{a−1})), every exponent <= 0.
    let mut acc = 0.0;
    let mut total = 0.0;
    for a in 0..n {
        let below = if a == 0 { 0.0 } else { (d * (e[a - 1] - e[a])).exp() * acc };
        total += p[a] * (suffix[a] + below);
        acc = below + q[a];
    }
    total.clamp(0.0, 1.0)
}

/// The Gaussian estimate of a rung pair's swap acceptance from its thermodynamic length:
/// `erfc(δL/2)`.
///
/// With `Δ = Δβ(E_j − E_i)` Gaussian, `Var(Δ) = 2Δβ²σ²` and `⟨Δ⟩ = −Δβ²σ² = −Var(Δ)/2` — the
/// fluctuation relation — so `E[min(1, e^Δ)] = 2Φ(−sqrt(Var Δ)/2) = erfc(Δβσ/2)`, and `Δβσ` is
/// `δL`. This is why equal length is equal acceptance; [`exact_swap_acceptance`] is the same number
/// without the Gaussian assumption.
#[must_use]
pub fn expected_acceptance(delta_length: f64) -> f64 {
    1.0 - crate::hopfield::erf(delta_length * 0.5)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::free_energy::linear_ladder;
    use crate::graph::GraphBuilder;
    use crate::ising;
    use crate::tempering::parallel_tempering;

    /// `n` uncoupled spins in a field `h`: `E = −h Σ s_i`, every moment in closed form.
    fn free_spins(n: usize, h: f64) -> Graph {
        let mut gb = GraphBuilder::new(n);
        for i in 0..n {
            gb.bias(i, h);
        }
        gb.build()
    }

    /// The Gudermannian, `gd(x) = ∫₀ˣ sech u du = arcsin(tanh x)`, and its inverse.
    fn gd(x: f64) -> f64 {
        x.tanh().asin()
    }
    fn gd_inv(y: f64) -> f64 {
        y.sin().atanh()
    }

    /// The metric against the closed form on a model whose every moment is known.
    ///
    /// Independent spins: `⟨E⟩ = −n h tanh(βh)`, `Var(E) = n h² sech²(βh)`, `ln Z = n ln(2 cosh βh)`.
    #[test]
    fn the_metric_is_the_closed_form_on_free_spins() {
        let (n, h) = (6usize, 0.7);
        let dos = exact_dos(&free_spins(n, h)).unwrap();
        assert_eq!(dos.energy.len(), n + 1, "a free-spin spectrum has n+1 levels");
        for k in 0..12 {
            let beta = 0.05 + 0.3 * k as f64;
            let (bh, nn) = (beta * h, n as f64);
            let want_var = nn * h * h / (bh.cosh() * bh.cosh());
            let want_mean = -nn * h * bh.tanh();
            let want_lz = nn * (2.0 * bh.cosh()).ln();
            assert!((variance(&dos, beta) - want_var).abs() < 1e-12 * want_var.max(1.0), "beta {beta}");
            assert!((mean_energy(&dos, beta) - want_mean).abs() < 1e-12, "beta {beta}");
            assert!((dos.log_z(beta) - want_lz).abs() < 1e-12, "beta {beta}");
        }
    }

    /// THE ONE-PASS VARIANCE IS RUBBLE AT THE COLD END, AND THIS IS WHERE.
    ///
    /// At `βh = 20` the true variance is `1.02e-16` while `⟨E⟩²` is 36, so `⟨E²⟩ − ⟨E⟩²` subtracts
    /// two numbers that agree past the last bit and returns **exactly zero** — a metric of zero,
    /// a length that stops accumulating, and a ladder that puts its last rung anywhere. The
    /// two-pass form is exact to a relative `1e-12` at the same point, and this test fails if
    /// anyone rewrites it the short way.
    #[test]
    fn the_variance_survives_where_cancellation_would_destroy_it() {
        let (n, h) = (6usize, 1.0);
        let dos = exact_dos(&free_spins(n, h)).unwrap();
        let one_pass = |beta: f64| {
            let p = probabilities(&dos, beta);
            let m1: f64 = p.iter().zip(&dos.energy).map(|(p, e)| p * e).sum();
            let m2: f64 = p.iter().zip(&dos.energy).map(|(p, e)| p * e * e).sum();
            m2 - m1 * m1
        };
        for &beta in &[10.0, 15.0, 20.0, 30.0] {
            let want = n as f64 * h * h / (beta * h).cosh().powi(2);
            let got = variance(&dos, beta);
            assert!((got - want).abs() < 1e-12 * want, "beta {beta}: {got:e} against {want:e}");
        }
        // 1.02e-16 and 2.10e-25 are the truths at these two betas, and the short form returns zero
        assert!(variance(&dos, 20.0) > 1e-16 && variance(&dos, 30.0) > 2e-25);
        assert_eq!(one_pass(20.0), 0.0, "the one-pass form is supposed to be destroyed here");
        assert_eq!(one_pass(30.0), 0.0);
        // and it is already visibly wrong well before that
        let want15 = n as f64 * h * h / 15.0f64.cosh().powi(2);
        assert!((one_pass(15.0) - want15).abs() > 1e-3 * want15, "one-pass at beta 15");
    }

    /// `Var(E) = −d⟨E⟩/dβ = d²lnZ/dβ²` — the identity that makes the Fisher information the metric.
    ///
    /// Checked by central differences on a frustrated model with irrational couplings, where no
    /// symmetry can make it hold by accident.
    #[test]
    fn the_metric_is_the_second_derivative_of_log_z() {
        let mut gb = GraphBuilder::new(10);
        for i in 0..10 {
            gb.couple(i, (i + 1) % 10, if i % 3 == 0 { -1.3 } else { 0.9 });
            gb.couple(i, (i + 3) % 10, 0.41 * (i as f64).sin());
            gb.bias(i, 0.2 * (i as f64).cos());
        }
        let dos = exact_dos(&gb.build()).unwrap();
        let d = 1e-4;
        for k in 0..8 {
            let beta = 0.1 + 0.35 * k as f64;
            let var = variance(&dos, beta);
            let dmean = -(mean_energy(&dos, beta + d) - mean_energy(&dos, beta - d)) / (2.0 * d);
            let d2lz = (dos.log_z(beta + d) - 2.0 * dos.log_z(beta) + dos.log_z(beta - d)) / (d * d);
            assert!((var - dmean).abs() < 1e-5 * var.max(1.0), "beta {beta}: {var} vs -dE/dbeta {dmean}");
            assert!((var - d2lz).abs() < 1e-4 * var.max(1.0), "beta {beta}: {var} vs d2lnZ {d2lz}");
        }
    }

    /// The length of a free-spin model is the Gudermannian, and the equal-length ladder inverts it.
    ///
    /// `speed = sqrt(n)·h·sech(βh)`, so `L(0→β) = sqrt(n)·gd(βh)` exactly, and the rung at fraction
    /// `k/K` of the length sits at `β_k = gd⁻¹(k/K · gd(βh))/h` — independent of `n`, which is the
    /// closed form this ladder must reproduce.
    #[test]
    fn the_equal_length_ladder_is_the_inverse_gudermannian_on_free_spins() {
        let (n, h, top) = (4usize, 0.8, 3.0);
        let prof = LengthProfile::exact(&free_spins(n, h), 0.0, top, 100_001).unwrap();
        let want_l = (n as f64).sqrt() * gd(top * h);
        assert!((prof.length() - want_l).abs() < 1e-8, "L = {} against {want_l}", prof.length());

        let rungs = 9;
        let got = prof.ladder(rungs);
        for k in 0..rungs {
            let want = gd_inv(k as f64 / (rungs - 1) as f64 * gd(top * h)) / h;
            assert!((got[k] - want).abs() < 1e-7, "rung {k}: {} against {want}", got[k]);
        }
        assert_eq!(got[0], 0.0);
        assert_eq!(got[rungs - 1], top);
    }

    /// Equal length means equal length: every rung of the ladder spans `L/K`, measured back off the
    /// profile the ladder came from.
    #[test]
    fn every_rung_of_an_equal_length_ladder_is_the_same_length() {
        let g = ising::lattice2d(4, 1.0);
        let prof = LengthProfile::exact(&g, 0.0, 2.5, DEFAULT_GRID).unwrap();
        for rungs in [3usize, 6, 12] {
            let ladder = prof.ladder(rungs);
            let want = prof.length() / (rungs - 1) as f64;
            for (i, dl) in prof.rung_lengths(&ladder).iter().enumerate() {
                assert!((dl - want).abs() < 1e-6 * want, "rungs {rungs}, pair {i}: {dl} against {want}");
            }
            assert!(ladder.windows(2).all(|w| w[1] > w[0]), "a ladder must increase strictly");
        }
    }

    /// The linear-time acceptance equals the defining double sum, term for term.
    #[test]
    fn the_streaming_acceptance_is_the_double_sum() {
        let g = ising::ring(8, 1.0, 0.3);
        let dos = exact_dos(&g).unwrap();
        for &(ba, bb) in &[(0.0, 0.4), (0.2, 0.9), (1.0, 1.05), (0.5, 4.0), (2.0, 2.0)] {
            let p = probabilities(&dos, ba);
            let q = probabilities(&dos, bb);
            let mut brute = 0.0;
            for a in 0..dos.energy.len() {
                for b in 0..dos.energy.len() {
                    let arg = (bb - ba) * (dos.energy[b] - dos.energy[a]);
                    brute += p[a] * q[b] * if arg >= 0.0 { 1.0 } else { arg.exp() };
                }
            }
            let got = exact_swap_acceptance(&dos, ba, bb);
            assert!((got - brute).abs() < 1e-12, "({ba},{bb}): {got} against {brute}");
            assert!((0.0..=1.0).contains(&got));
        }
        // identical rungs always swap, and the order of the pair cannot matter
        assert!((exact_swap_acceptance(&dos, 0.7, 0.7) - 1.0).abs() < 1e-12);
        let (x, y) = (exact_swap_acceptance(&dos, 0.3, 1.4), exact_swap_acceptance(&dos, 1.4, 0.3));
        assert_eq!(x.to_bits(), y.to_bits());
    }

    /// `erfc(δL/2)` is FIRST ORDER in the rung length, and this measures the order.
    ///
    /// The prediction drops the third cumulant, so its error should be linear in `δL`: halving the
    /// rung length must halve the gap to the exact acceptance. Measured on a 13-spin model with
    /// 8,192 distinct levels, where the energy distribution is close to Gaussian, the ratio is
    /// 0.51 each time it is refined and the residual reaches 0.0024. The published `erfc` values
    /// pin the special function itself.
    #[test]
    fn the_gaussian_prediction_is_first_order_in_the_rung_length() {
        assert_eq!(expected_acceptance(0.0), 1.0);
        assert!((expected_acceptance(2.0) - 0.157_299_207_050_285_13).abs() < 1e-12, "erfc(1)");
        assert!((expected_acceptance(1.0) - 0.479_500_122_186_953_5).abs() < 1e-12, "erfc(1/2)");

        let mut gb = GraphBuilder::new(13);
        let mut r = crate::rng::Pcg::new(7, 1);
        for i in 0..13 {
            for j in i + 1..13 {
                if r.f64() < 0.45 {
                    gb.couple(i, j, 2.0 * r.f64() - 1.0);
                }
            }
            gb.bias(i, 0.5 * r.f64() - 0.25);
        }
        let dos = exact_dos(&gb.build()).unwrap();
        assert_eq!(dos.energy.len(), 8192, "every state at its own energy: a dense spectrum");
        let prof = LengthProfile::from_dos(&dos, 0.0, 2.0, DEFAULT_GRID);
        let worst: Vec<(f64, f64)> = [16usize, 32, 64, 128]
            .iter()
            .map(|&rungs| {
                let ladder = prof.ladder(rungs);
                let dl = prof.rung_lengths(&ladder);
                let err = ladder
                    .windows(2)
                    .zip(&dl)
                    .map(|(w, &l)| (exact_swap_acceptance(&dos, w[0], w[1]) - expected_acceptance(l)).abs())
                    .fold(0.0f64, f64::max);
                (prof.length() / (rungs - 1) as f64, err)
            })
            .collect();
        for w in worst.windows(2) {
            let ratio = w[1].1 / w[0].1;
            assert!((0.4..0.6).contains(&ratio), "first order in dL, got a ratio of {ratio:.3}: {worst:?}");
        }
        assert!(worst.last().unwrap().1 < 0.005, "the residual at the finest ladder: {worst:?}");
    }

    /// **THE CLAIM, AGAINST AN EXACT ORACLE.**
    ///
    /// Equal thermodynamic length gives more uniform swap acceptance than equal `β`, computed
    /// exactly for both ladders from the enumerated spectrum — no sampling anywhere, so nothing here
    /// depends on a seed.
    #[test]
    fn equal_length_beats_linear_on_exact_acceptance() {
        let g = ising::lattice2d(4, 1.0);
        let dos = exact_dos(&g).unwrap();
        let top = 3.0;
        for rungs in [6usize, 8, 12] {
            let geo = equal_length_ladder(&g, 0.0, top, rungs).unwrap();
            let lin = linear_ladder(top, rungs);
            let rates = |l: &[f64]| -> Vec<f64> {
                l.windows(2).map(|w| exact_swap_acceptance(&dos, w[0], w[1])).collect()
            };
            let spread = |r: &[f64]| {
                r.iter().copied().fold(f64::NEG_INFINITY, f64::max) - r.iter().copied().fold(f64::INFINITY, f64::min)
            };
            let (rg, rl) = (rates(&geo), rates(&lin));
            let (sg, sl) = (spread(&rg), spread(&rl));
            // Measured ratios are 0.23 to 0.35 across these rung counts; the bar is 0.45, which is
            // above every one of them and still far below "no improvement".
            assert!(sg < 0.45 * sl, "rungs {rungs}: equal-length spread {sg:.4} against linear {sl:.4}");
            let worst = |r: &[f64]| r.iter().copied().fold(f64::INFINITY, f64::min);
            assert!(worst(&rg) > worst(&rl) + 0.1, "rungs {rungs}: worst pair {:.4} vs {:.4}", worst(&rg), worst(&rl));
        }
    }

    /// **THE CLAIM, MEASURED THROUGH [`crate::tempering`].**
    ///
    /// The same two ladders, run as actual parallel tempering, over eight seeds. What is asserted is
    /// the mean spread of the measured `swap_rates` — and, because a measurement of a known quantity
    /// is worth more than a comparison, that the measured rates land on the exact acceptance the
    /// spectrum predicts for those same rungs.
    #[test]
    fn equal_length_beats_linear_on_measured_swap_rates() {
        let g = ising::lattice2d(4, 1.0);
        let dos = exact_dos(&g).unwrap();
        let (top, rungs, rounds) = (3.0, 8usize, 4000);
        let geo = equal_length_ladder(&g, 0.0, top, rungs).unwrap();
        let lin = linear_ladder(top, rungs);

        let spread = |r: &[f64]| {
            r.iter().copied().fold(f64::NEG_INFINITY, f64::max) - r.iter().copied().fold(f64::INFINITY, f64::min)
        };
        let (mut sg, mut sl, mut drift) = (0.0, 0.0, 0.0f64);
        let seeds = 8u64;
        for seed in 0..seeds {
            let og = parallel_tempering(&g, &geo, rounds, 2, seed, None);
            let ol = parallel_tempering(&g, &lin, rounds, 2, seed, None);
            sg += spread(&og.swap_rates) / seeds as f64;
            sl += spread(&ol.swap_rates) / seeds as f64;
            for (i, &r) in og.swap_rates.iter().enumerate() {
                drift = drift.max((r - exact_swap_acceptance(&dos, geo[i], geo[i + 1])).abs());
            }
        }
        assert!(sg < 0.3 * sl, "measured spread: equal-length {sg:.4} against linear {sl:.4}");
        // The rungs of a tempering ladder are not quite independent equilibrium replicas -- the
        // swaps themselves correlate them -- so this is a tolerance, not an identity. It is what
        // says the measurement and the oracle are describing the same experiment.
        assert!(drift < 0.05, "measured rates should sit on the exact acceptance: worst gap {drift:.4}");
    }

    /// The measured route recovers the exact metric — **on the rungs it measured it on**.
    ///
    /// [`LengthProfile::from_traces`] reads the metric off a tempering run instead of an
    /// enumeration, and the oracle is the exact metric integrated the same way on the same twelve
    /// rungs: 5.821, matched to within 3% at every seed.
    ///
    /// The oracle is NOT the fine-grid length, which is 4.995. Twelve rungs of trapezoid over a
    /// metric that peaks at the transition overstates it by 17%, and that error is quadrature, not
    /// sampling — a caller who wants the true length of a span must refine the grid, and no number
    /// of samples at twelve rungs will do it. The second assertion is what keeps the two apart.
    #[test]
    fn the_measured_metric_reproduces_the_exact_one_on_the_same_rungs() {
        let g = ising::lattice2d(4, 1.0);
        let dos = exact_dos(&g).unwrap();
        let ladder = equal_length_ladder(&g, 0.0, 2.0, 12).unwrap();
        let on_rungs: Vec<f64> = ladder.iter().map(|&b| speed(&dos, b)).collect();
        let oracle = LengthProfile::from_speed(&ladder, &on_rungs).unwrap().length();
        for seed in 0..4u64 {
            let (_, traces) =
                crate::tempering::parallel_tempering_observed(&g, &ladder, 3000, 2, 500, seed, None);
            let got = LengthProfile::from_traces(&traces.as_pairs()).unwrap().length();
            assert!((got - oracle).abs() < 0.06 * oracle, "seed {seed}: {got:.3} against {oracle:.3}");
        }
        let fine = LengthProfile::from_dos(&dos, 0.0, 2.0, DEFAULT_GRID).length();
        assert!(oracle - fine > 0.1 * fine, "the coarse trapezoid is the larger error: {oracle:.3} vs {fine:.3}");
    }

    /// A metric that vanishes still yields a strictly increasing ladder.
    ///
    /// A dead span has no length to divide, so every rung lands on the same `β` and the ladder is a
    /// wall of ties — `Δβ = 0` is a swap that always accepts and a [`crate::free_energy`] ladder
    /// that panics. The tied run is spaced uniformly in `β` instead, which is the only information
    /// left. A live stretch followed by a dead one keeps its rungs in the live part, which is the
    /// right answer and not a tie at all.
    #[test]
    fn a_dead_metric_is_spaced_uniformly_rather_than_tied() {
        let betas = [0.0, 0.5, 1.0, 1.5, 2.0];
        let dead = LengthProfile::from_speed(&betas, &[0.0; 5]).unwrap();
        assert_eq!(dead.length(), 0.0);
        let ladder = dead.ladder(6);
        assert!(ladder.windows(2).all(|w| w[1] > w[0]), "{ladder:?}");
        for (k, b) in ladder.iter().enumerate() {
            assert!((b - 2.0 * k as f64 / 5.0).abs() < 1e-12, "uniform in beta: {ladder:?}");
        }

        let half = LengthProfile::from_speed(&betas, &[2.0, 2.0, 0.0, 0.0, 0.0]).unwrap();
        let ladder = half.ladder(6);
        assert!(ladder.windows(2).all(|w| w[1] > w[0]), "{ladder:?}");
        assert_eq!((ladder[0], ladder[5]), (0.0, 2.0));
        // the metric dies at beta 1.0, and every rung but the pinned cold end is below it
        assert!(ladder[4] < 1.0, "every rung with length in it is in the live stretch: {ladder:?}");
    }

    /// The profile refuses what it cannot integrate, rather than returning a curve.
    #[test]
    fn a_profile_refuses_a_grid_it_cannot_integrate() {
        assert!(LengthProfile::from_speed(&[0.0, 1.0], &[1.0]).unwrap_err().contains("against"));
        assert!(LengthProfile::from_speed(&[0.0], &[1.0]).unwrap_err().contains("two grid points"));
        assert!(LengthProfile::from_speed(&[1.0, 0.0], &[1.0, 1.0]).unwrap_err().contains("increasing"));
        assert!(LengthProfile::from_speed(&[0.0, 1.0], &[1.0, -1.0]).unwrap_err().contains("negative"));
        assert!(LengthProfile::from_speed(&[0.0, 1.0], &[1.0, f64::NAN]).unwrap_err().contains("negative"));
        assert!(LengthProfile::from_traces(&[(0.0, vec![1.0]), (1.0, vec![1.0, 2.0])])
            .unwrap_err()
            .contains("two draws"));
    }

    /// Enumeration has a limit and this says so rather than trying.
    #[test]
    fn a_model_too_large_to_enumerate_is_refused() {
        let g = ising::lattice2d(6, 1.0);
        assert_eq!(
            exact_dos(&g).unwrap_err(),
            Refused::TooLargeToEnumerate { spins: 36, limit: crate::samples::ENUMERATION_LIMIT }
        );
        assert!(equal_length_ladder(&g, 0.0, 1.0, 4).is_err());
    }

    /// The interpolation and its inverse are inverses, on the grid and between it.
    #[test]
    fn length_at_and_beta_at_invert_each_other() {
        let prof = LengthProfile::exact(&ising::ring(10, 1.0, 0.2), 0.1, 2.4, 513).unwrap();
        for k in 0..40 {
            let beta = 0.1 + 2.3 * k as f64 / 39.0;
            let back = prof.beta_at(prof.length_at(beta));
            assert!((back - beta).abs() < 1e-9, "beta {beta} came back {back}");
        }
        assert_eq!(prof.beta_at(-1.0), 0.1);
        assert_eq!(prof.beta_at(1e9), 2.4);
        assert_eq!(prof.length_at(0.0), 0.0);
        assert_eq!(prof.length_at(9.0), prof.length());
        // NaN sorts above every number under `total_cmp`, so an unguarded search would read past
        // the grid and panic. Both ends of the pair send it to the hot end instead.
        assert_eq!(prof.length_at(f64::NAN), 0.0);
        assert_eq!(prof.beta_at(f64::NAN), 0.1);
    }
}
