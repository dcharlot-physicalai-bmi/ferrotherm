//! The in-fabric phase generator — a coupled-oscillator generative model with **no digital
//! decoder**, and therefore one whose substrate efficiency can reach its total.
//!
//! # Why this module exists
//!
//! The published coupled-oscillator image generators are two machines in a trenchcoat: a phase
//! system that does the interesting physics, and a conventional convolutional decoder of tens of
//! millions of parameters that turns the phase readout into pixels. [`crate::precision`] prices that
//! arrangement and the answer is not close — the decoder is over 99.9% of the energy, so making the
//! oscillator half *free* improves the whole by under a percent. A thousand-fold substrate claim
//! attached to that architecture is a claim about the component that was never the cost.
//!
//! This module is the architecture with the decoder removed. Output units are Gaussian and live in
//! the same energy function as the phases, so a sample is one draw from one model rather than a
//! transport step followed by a neural network:
//!
//! ```text
//!   E(x, theta) = 1/2 x^T A x - b^T x - x^T C u(theta) - sum_{i<j} S_ij cos(theta_i - theta_j)
//! ```
//!
//! with `x` in `R^m` the outputs, `theta` in `T^n` the phases, `u(theta)` the `2n` readout features
//! `(cos theta_1, sin theta_1, ...)` the oscillator fabric already produces, `A` symmetric positive
//! definite, and `S` the symmetric phase coupling that [`crate::kuramoto`] extracts from any
//! oscillator network.
//!
//! # Both conditionals are exactly samplable, which is the whole design
//!
//! **Outputs given phases** are Gaussian: precision `beta A`, mean `A^-1 (b + C u(theta))`. One
//! Cholesky, one triangular solve, no approximation — [`PhaseGen::sample_outputs`].
//!
//! **A phase given everything else** is **von Mises**. Collecting every term containing `theta_i`:
//!
//! ```text
//!   -cos(theta_i) [ (x^T C)_{2i}   + sum_j S_ij cos(theta_j) ]
//!   -sin(theta_i) [ (x^T C)_{2i+1} + sum_j S_ij sin(theta_j) ]
//!   = -R_i cos(theta_i - phi_i),
//! ```
//!
//! so the conditional is `vM(phi_i, beta R_i)` with `R_i` the magnitude of that two-component field
//! and `phi_i` its direction — [`PhaseGen::phase_field`], sampled exactly by Best & Fisher's
//! rejection method (*Efficient simulation of the von Mises distribution*, Appl. Statist. 28:152,
//! 1979). Its normaliser is `2 pi I_0(beta R)` in closed form.
//!
//! A von Mises conditional is the circular heat bath, and on the `q`-point grid it is *exactly* the
//! clock-model heat bath this crate already samples. So the continuous lane and the categorical lane
//! are the same sampler at two resolutions, with [`crate::kuramoto::Kuramoto::covering_gap`] as the
//! stated distance between them.
//!
//! # What this buys that the decoder architecture cannot have
//!
//! An **exact oracle for the whole generator**. The outputs are Gaussian, so they integrate out in
//! closed form, leaving a marginal over phases alone:
//!
//! ```text
//!   E_eff(theta) = -1/2 (b + C u)^T A^-1 (b + C u) - sum_{i<j} S_ij cos(theta_i - theta_j)
//! ```
//!
//! ([`PhaseGen::effective_energy`]). Restricted to the `q`-point grid this is enumerable, so
//! [`PhaseGen::enumerate_grid`] returns the exact distribution and the exact `ln Z` of a complete
//! generative model — against which the sampler is scored in
//! `block_gibbs_reproduces_the_enumerated_generator`. No published generative model in this field
//! has an exact partition function, because a convolutional decoder has none; this one does, because
//! every block of it was chosen so that it would.
//!
//! That is the absorption in its constructive form. The rival architecture is a transport map with
//! no invariant measure, no temperature and no likelihood. This one is an energy-based model with
//! all three, it runs on the same fabric, its phase block is the same physics, and it can be
//! certified.

use crate::continuous::{cholesky, log_det, solve};
use crate::ledger::Ledger;
use crate::rng::Pcg;
use crate::round::sum_up;

/// Two pi.
const TAU: f64 = core::f64::consts::TAU;

/// What can be wrong with a generator's description.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    /// A matrix was not the shape its dimensions require.
    Shape {
        /// Which matrix.
        what: &'static str,
    },
    /// The output precision matrix is not positive definite, so the outputs have no distribution.
    NotPositiveDefinite,
    /// `q^n` exceeds what may be enumerated.
    TooBig {
        /// Grid points per phase.
        q: usize,
        /// Phases.
        n: usize,
    },
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Error::Shape { what } => write!(f, "{what} is not the shape its dimensions require"),
            Error::NotPositiveDefinite => write!(
                f,
                "the output precision matrix is not positive definite, so the Gaussian block has no \
                 distribution and nothing below it is defined"
            ),
            Error::TooBig { q, n } => {
                write!(f, "enumerating {q}^{n} phase states is refused; a partial enumeration is not an oracle")
            }
        }
    }
}

impl core::error::Error for Error {}

/// The largest number of grid states [`PhaseGen::enumerate_grid`] will visit.
pub const ENUMERATION_LIMIT: usize = 1 << 20;

/// A joint model of Gaussian outputs and coupled phases.
#[derive(Clone, Debug)]
pub struct PhaseGen {
    m: usize,
    n: usize,
    a: Vec<f64>,
    b: Vec<f64>,
    c: Vec<f64>,
    s: Vec<f64>,
}

impl PhaseGen {
    /// Build from the output precision `a` (`m * m`, symmetric positive definite), the output bias
    /// `b` (`m`), the phase-to-output coupling `c` (`m * 2n`, row-major) and the phase coupling `s`
    /// (`n * n`, symmetrised on the way in).
    ///
    /// # Errors
    ///
    /// [`Error::Shape`] for any wrong dimension, [`Error::NotPositiveDefinite`] when `a` has no
    /// Cholesky factor.
    pub fn new(
        m: usize,
        n: usize,
        a: Vec<f64>,
        b: Vec<f64>,
        c: Vec<f64>,
        s: Vec<f64>,
    ) -> Result<PhaseGen, Error> {
        if a.len() != m * m {
            return Err(Error::Shape { what: "the output precision matrix" });
        }
        if b.len() != m {
            return Err(Error::Shape { what: "the output bias" });
        }
        if c.len() != m * 2 * n {
            return Err(Error::Shape { what: "the phase-to-output coupling" });
        }
        if s.len() != n * n {
            return Err(Error::Shape { what: "the phase coupling" });
        }
        if cholesky(&a, m).is_none() {
            return Err(Error::NotPositiveDefinite);
        }
        // Symmetrise rather than trust: the energy only sees the symmetric part, and a caller who
        // passed an asymmetric matrix should get the model the energy describes, not a silently
        // different one. The asymmetric half belongs in `crate::kuramoto`, which has a name for it.
        let mut sym = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                sym[i * n + j] = if i == j { 0.0 } else { 0.5 * (s[i * n + j] + s[j * n + i]) };
            }
        }
        Ok(PhaseGen { m, n, a, b, c, s: sym })
    }

    /// Output count.
    #[must_use]
    pub fn m(&self) -> usize {
        self.m
    }

    /// Phase count.
    #[must_use]
    pub fn n(&self) -> usize {
        self.n
    }

    /// The readout features `u(theta) = (cos theta_0, sin theta_0, ...)`, length `2n`.
    ///
    /// The same `2n` numbers an oscillator fabric puts on its pins, which is why this model needs no
    /// decoder to consume them.
    ///
    /// # Panics
    ///
    /// If `phi` is not length `n`.
    #[must_use]
    pub fn features(&self, phi: &[f64]) -> Vec<f64> {
        assert_eq!(phi.len(), self.n, "phase vector must have one entry per oscillator");
        let mut u = Vec::with_capacity(2 * self.n);
        for t in phi {
            u.push(t.cos());
            u.push(t.sin());
        }
        u
    }

    /// The joint energy.
    ///
    /// # Panics
    ///
    /// If either vector is the wrong length.
    #[must_use]
    pub fn energy(&self, x: &[f64], phi: &[f64]) -> f64 {
        assert_eq!(x.len(), self.m, "output vector must be length {}", self.m);
        let u = self.features(phi);
        let mut terms = Vec::new();
        for i in 0..self.m {
            for j in 0..self.m {
                terms.push(0.5 * x[i] * self.a[i * self.m + j] * x[j]);
            }
            terms.push(-self.b[i] * x[i]);
            for k in 0..(2 * self.n) {
                terms.push(-x[i] * self.c[i * 2 * self.n + k] * u[k]);
            }
        }
        for i in 0..self.n {
            for j in (i + 1)..self.n {
                terms.push(-self.s[i * self.n + j] * (phi[i] - phi[j]).cos());
            }
        }
        sum_up(&terms)
    }

    /// The linear term of the output conditional, `b + C u(theta)`.
    ///
    /// # Panics
    ///
    /// If `phi` is the wrong length.
    #[must_use]
    pub fn output_field(&self, phi: &[f64]) -> Vec<f64> {
        let u = self.features(phi);
        (0..self.m)
            .map(|i| {
                let mut t: Vec<f64> =
                    (0..(2 * self.n)).map(|k| self.c[i * 2 * self.n + k] * u[k]).collect();
                t.push(self.b[i]);
                sum_up(&t)
            })
            .collect()
    }

    /// The mean of the output conditional, `A^-1 (b + C u)`.
    ///
    /// # Panics
    ///
    /// If `phi` is the wrong length or the precision matrix has lost positive definiteness.
    #[must_use]
    pub fn output_mean(&self, phi: &[f64]) -> Vec<f64> {
        let g = self.output_field(phi);
        solve(&self.a, self.m, &g).expect("positive definiteness was checked at construction")
    }

    /// Draw the outputs exactly from their Gaussian conditional, charging one device read apiece.
    ///
    /// `x = mu + beta^{-1/2} L^{-T} z` with `A = L L^T` and `z` standard normal — an exact draw, not
    /// a Metropolis step.
    ///
    /// # Panics
    ///
    /// If `phi` is the wrong length or `beta` is not positive.
    #[must_use]
    pub fn sample_outputs(
        &self,
        phi: &[f64],
        beta: f64,
        rng: &mut Pcg,
        ledger: Option<&mut Ledger>,
    ) -> Vec<f64> {
        assert!(beta > 0.0 && beta.is_finite(), "beta must be positive and finite, got {beta}");
        let m = self.m;
        let l = cholesky(&self.a, m).expect("positive definiteness was checked at construction");
        let mu = self.output_mean(phi);
        let z: Vec<f64> = (0..m).map(|_| gaussian(rng)).collect();
        // Back-substitute L^T y = z.
        let mut y = vec![0.0; m];
        for i in (0..m).rev() {
            let mut acc: Vec<f64> = ((i + 1)..m).map(|k| -l[k * m + i] * y[k]).collect();
            acc.push(z[i]);
            y[i] = sum_up(&acc) / l[i * m + i];
        }
        let scale = beta.sqrt().recip();
        let out: Vec<f64> = (0..m).map(|i| mu[i] + scale * y[i]).collect();
        if let Some(led) = ledger {
            led.reads += m as u64;
        }
        out
    }

    /// The two-component field on phase `i`, returned as `(R, phi)` — magnitude and direction.
    ///
    /// The conditional of `theta_i` is `vM(phi, beta R)`, whose density is
    /// `exp(beta R cos(theta - phi)) / (2 pi I_0(beta R))`.
    ///
    /// # Panics
    ///
    /// If either vector is the wrong length.
    #[must_use]
    pub fn phase_field(&self, i: usize, x: &[f64], phi: &[f64]) -> (f64, f64) {
        assert_eq!(x.len(), self.m, "output vector must be length {}", self.m);
        assert_eq!(phi.len(), self.n, "phase vector must have one entry per oscillator");
        let mut gc: Vec<f64> = Vec::new();
        let mut gs: Vec<f64> = Vec::new();
        for r in 0..self.m {
            gc.push(x[r] * self.c[r * 2 * self.n + 2 * i]);
            gs.push(x[r] * self.c[r * 2 * self.n + 2 * i + 1]);
        }
        for j in 0..self.n {
            if j == i {
                continue;
            }
            gc.push(self.s[i * self.n + j] * phi[j].cos());
            gs.push(self.s[i * self.n + j] * phi[j].sin());
        }
        let c = sum_up(&gc);
        let s = sum_up(&gs);
        (c.hypot(s), s.atan2(c))
    }

    /// One block-Gibbs sweep: draw every phase from its von Mises conditional, then the outputs from
    /// their Gaussian one.
    ///
    /// Charges one device sample per phase update and one read per output drawn.
    ///
    /// # Panics
    ///
    /// If either vector is the wrong length or `beta` is not positive.
    pub fn sweep(
        &self,
        beta: f64,
        x: &mut Vec<f64>,
        phi: &mut [f64],
        rng: &mut Pcg,
        mut ledger: Option<&mut Ledger>,
    ) {
        for i in 0..self.n {
            let (r, mu) = self.phase_field(i, x, phi);
            phi[i] = sample_von_mises(mu, beta * r, rng);
        }
        if let Some(l) = ledger.as_deref_mut() {
            l.samples += self.n as u64;
        }
        *x = self.sample_outputs(phi, beta, rng, ledger);
    }

    /// The exact marginal energy over phases, with the outputs integrated out.
    ///
    /// `-1/2 (b + C u)^T A^-1 (b + C u) - sum_{i<j} S_ij cos(theta_i - theta_j)`. The Gaussian
    /// integral is exact, so this is not a bound or an approximation: the phase marginal of the
    /// joint model is `exp(-beta E_eff)` up to the constant that
    /// [`PhaseGen::output_log_partition_constant`] returns.
    ///
    /// # Panics
    ///
    /// If `phi` is the wrong length.
    #[must_use]
    pub fn effective_energy(&self, phi: &[f64]) -> f64 {
        let g = self.output_field(phi);
        let ainv_g =
            solve(&self.a, self.m, &g).expect("positive definiteness was checked at construction");
        let quad: Vec<f64> = (0..self.m).map(|i| -0.5 * g[i] * ainv_g[i]).collect();
        let mut terms = quad;
        for i in 0..self.n {
            for j in (i + 1)..self.n {
                terms.push(-self.s[i * self.n + j] * (phi[i] - phi[j]).cos());
            }
        }
        sum_up(&terms)
    }

    /// The phase-independent constant the output integral contributes to `ln Z`:
    /// `(m/2) ln(2 pi / beta) - (1/2) ln det A`.
    ///
    /// # Panics
    ///
    /// If `beta` is not positive.
    #[must_use]
    pub fn output_log_partition_constant(&self, beta: f64) -> f64 {
        assert!(beta > 0.0 && beta.is_finite(), "beta must be positive and finite, got {beta}");
        let ld = log_det(&self.a, self.m).expect("positive definiteness was checked at construction");
        0.5 * self.m as f64 * (TAU / beta).ln() - 0.5 * ld
    }

    /// The exact distribution of the phase marginal on the `q`-point grid, and its `ln Z`.
    ///
    /// The oracle. `p` is indexed with phase 0 fastest, matching [`crate::potts::Potts::index_of`].
    ///
    /// # Errors
    ///
    /// [`Error::TooBig`] when `q^n` exceeds [`ENUMERATION_LIMIT`].
    ///
    /// # Panics
    ///
    /// If `q` is below 2 or `beta` is not positive.
    pub fn enumerate_grid(&self, q: usize, beta: f64) -> Result<Enumerated, Error> {
        assert!(q >= 2, "a grid needs at least 2 points, got {q}");
        assert!(beta > 0.0 && beta.is_finite(), "beta must be positive and finite, got {beta}");
        let total = u32::try_from(self.n)
            .ok()
            .and_then(|e| q.checked_pow(e))
            .filter(|&t| t <= ENUMERATION_LIMIT)
            .ok_or(Error::TooBig { q, n: self.n })?;
        let mut energies = Vec::with_capacity(total);
        let mut phi = vec![0.0; self.n];
        for index in 0..total {
            let mut rest = index;
            for v in &mut phi {
                *v = TAU * (rest % q) as f64 / q as f64;
                rest /= q;
            }
            energies.push(self.effective_energy(&phi));
        }
        let min = energies.iter().copied().fold(f64::INFINITY, f64::min);
        let w: Vec<f64> = energies.iter().map(|e| (-beta * (e - min)).exp()).collect();
        let z = sum_up(&w);
        let p = w.into_iter().map(|v| v / z).collect();
        Ok(Enumerated {
            q,
            n: self.n,
            beta,
            log_z: z.ln() - beta * min + self.output_log_partition_constant(beta),
            p,
            energies,
        })
    }
}

/// The exact phase marginal on a grid.
#[derive(Clone, Debug)]
pub struct Enumerated {
    /// Grid points per phase.
    pub q: usize,
    /// Phases.
    pub n: usize,
    /// The inverse temperature.
    pub beta: f64,
    /// `ln Z` of the **whole** model, outputs included.
    pub log_z: f64,
    /// Probability of every grid state, phase 0 fastest.
    pub p: Vec<f64>,
    /// Effective energy of every grid state, in the same order.
    pub energies: Vec<f64>,
}

/// A standard normal by Box–Muller, from the crate's generator.
fn gaussian(rng: &mut Pcg) -> f64 {
    let u1 = rng.f64().max(f64::MIN_POSITIVE);
    let u2 = rng.f64();
    (-2.0 * u1.ln()).sqrt() * (TAU * u2).cos()
}

/// An exact draw from `vM(mu, kappa)` by Best & Fisher's rejection method.
///
/// Falls back to the uniform law below `kappa = 1e-8`, where the acceptance constants lose their
/// significance and the distribution is uniform to well past `f64` resolution anyway.
///
/// # Panics
///
/// If `kappa` is negative or not finite.
#[must_use]
pub fn sample_von_mises(mu: f64, kappa: f64, rng: &mut Pcg) -> f64 {
    assert!(kappa >= 0.0 && kappa.is_finite(), "concentration must be finite and non-negative");
    if kappa < 1e-8 {
        return (mu + TAU * rng.f64()).rem_euclid(TAU);
    }
    let a = 1.0 + (1.0 + 4.0 * kappa * kappa).sqrt();
    let b = (a - (2.0 * a).sqrt()) / (2.0 * kappa);
    let r = (1.0 + b * b) / (2.0 * b);
    // Bounded rather than unbounded: Best & Fisher accept with probability above 0.65 at every
    // concentration, so 1,000 attempts fail with probability below 1e-450. Returning the mean on
    // exhaustion keeps the stream length independent of its own contents, which is what makes a
    // seeded run reproducible across platforms.
    for _ in 0..1_000 {
        let u1 = rng.f64();
        let u2 = rng.f64();
        let z = (core::f64::consts::PI * u1).cos();
        let f = (1.0 + r * z) / (r + z);
        let c = kappa * (r - f);
        let accept = c * (2.0 - c) - u2 > 0.0 || (c / u2).ln() + 1.0 - c >= 0.0;
        if accept {
            let sign = if rng.f64() > 0.5 { 1.0 } else { -1.0 };
            return (mu + sign * f.clamp(-1.0, 1.0).acos()).rem_euclid(TAU);
        }
    }
    mu.rem_euclid(TAU)
}

/// The modified Bessel function `I_0(x)`, by its power series.
///
/// Present so that the von Mises normaliser is available in closed form, which is what makes the
/// phase block's conditional density checkable rather than merely samplable.
///
/// # Why the series and not an asymptotic form
///
/// Every term of `I_0(x) = sum_k (x^2/4)^k / (k!)^2` is **positive**, so the sum has no
/// cancellation at any argument and `f64` carries it to full relative accuracy wherever it does not
/// overflow — which is past `x = 700`, far beyond any concentration a phase conditional produces.
/// The usual polynomial asymptotic is there to save time, not accuracy, and it costs accuracy: an
/// earlier draft of this function used one and returned `I_0(20)` three orders of magnitude wrong,
/// which the table test below caught. A closed form that is only approximately closed is not what
/// this module needs from it.
///
/// # Panics
///
/// If `|x|` exceeds 700, where the series overflows `f64` and the caller needs a scaled form.
#[must_use]
pub fn bessel_i0(x: f64) -> f64 {
    let ax = x.abs();
    assert!(ax <= 700.0, "I_0 overflows f64 past |x| = 700; got {x}");
    let mut term = 1.0;
    let mut acc = 1.0;
    for k in 1..1_000 {
        term *= (ax / (2.0 * f64::from(k))).powi(2);
        acc += term;
        if term < 1e-18 * acc {
            break;
        }
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two outputs, two phases, everything coupled.
    fn fixture() -> PhaseGen {
        let a = vec![2.0, 0.3, 0.3, 1.5];
        let b = vec![0.2, -0.1];
        // 2 outputs x 4 features.
        let c = vec![0.9, 0.1, -0.4, 0.5, 0.2, -0.7, 0.6, 0.3];
        let s = vec![0.0, 0.8, 0.8, 0.0];
        PhaseGen::new(2, 2, a, b, c, s).expect("fixture is well formed")
    }

    #[test]
    fn a_non_positive_definite_precision_is_refused() {
        let a = vec![1.0, 2.0, 2.0, 1.0];
        let r = PhaseGen::new(2, 1, a, vec![0.0, 0.0], vec![0.0; 4], vec![0.0]);
        assert_eq!(r.unwrap_err(), Error::NotPositiveDefinite);
    }

    #[test]
    fn the_output_conditional_has_the_mean_and_covariance_the_closed_form_states() {
        let g = fixture();
        let phi = [0.4, 1.7];
        let beta = 1.3;
        let mu = g.output_mean(&phi);
        let mut rng = Pcg::new(5150, 2);
        let draws = 200_000;
        let mut s0 = 0.0;
        let mut s1 = 0.0;
        let mut q00 = 0.0;
        let mut q11 = 0.0;
        let mut q01 = 0.0;
        for _ in 0..draws {
            let x = g.sample_outputs(&phi, beta, &mut rng, None);
            s0 += x[0];
            s1 += x[1];
            q00 += x[0] * x[0];
            q11 += x[1] * x[1];
            q01 += x[0] * x[1];
        }
        let n = f64::from(draws);
        let (m0, m1) = (s0 / n, s1 / n);
        assert!((m0 - mu[0]).abs() < 0.01, "mean {m0} != {}", mu[0]);
        assert!((m1 - mu[1]).abs() < 0.01, "mean {m1} != {}", mu[1]);
        // Covariance is (beta A)^-1.
        let inv = crate::continuous::inverse(&[2.0, 0.3, 0.3, 1.5], 2).expect("invertible");
        let want = |i: usize, j: usize| inv[i * 2 + j] / beta;
        assert!((q00 / n - m0 * m0 - want(0, 0)).abs() < 0.02);
        assert!((q11 / n - m1 * m1 - want(1, 1)).abs() < 0.02);
        assert!((q01 / n - m0 * m1 - want(0, 1)).abs() < 0.02);
    }

    #[test]
    fn the_phase_conditional_is_exactly_the_von_mises_the_field_names() {
        // The algebraic claim, checked against the energy itself: moving one phase changes the
        // energy by exactly -R[cos(theta' - phi) - cos(theta - phi)].
        let g = fixture();
        let x = [0.7, -0.3];
        let phi = [0.4, 1.7];
        let (r, mu) = g.phase_field(0, &x, &phi);
        for t in [0.0, 0.9, 2.2, 5.5] {
            let mut p1 = phi;
            p1[0] = t;
            let mut p2 = phi;
            p2[0] = t + 0.31;
            let de = g.energy(&x, &p2) - g.energy(&x, &p1);
            let want = -r * ((p2[0] - mu).cos() - (p1[0] - mu).cos());
            assert!((de - want).abs() < 1e-12, "energy change {de} != von Mises form {want}");
        }
    }

    #[test]
    fn the_von_mises_sampler_matches_its_own_density() {
        // Histogram against exp(kappa cos(t - mu)) / (2 pi I0(kappa)), with the normaliser computed
        // rather than fitted -- so the Bessel routine is under test at the same time.
        let mut rng = Pcg::new(11, 7);
        let kappa = 2.0;
        let mu = 1.0;
        let bins = 24;
        let mut hist = vec![0u64; bins];
        let draws = 400_000u64;
        for _ in 0..draws {
            let t = sample_von_mises(mu, kappa, &mut rng);
            hist[((t / TAU) * bins as f64) as usize % bins] += 1;
        }
        let width = TAU / bins as f64;
        let norm = TAU * bessel_i0(kappa);
        for k in 0..bins {
            let centre = (k as f64 + 0.5) * width;
            let want = (kappa * (centre - mu).cos()).exp() / norm * width;
            let got = hist[k] as f64 / draws as f64;
            assert!(
                (got - want).abs() < 0.004,
                "bin {k}: sampled {got}, density says {want}"
            );
        }
    }

    #[test]
    fn bessel_i0_matches_known_values() {
        // Abramowitz & Stegun table values, and the series/asymptotic crossover.
        assert!((bessel_i0(0.0) - 1.0).abs() < 1e-15);
        assert!((bessel_i0(1.0) - 1.266_065_877_75).abs() < 1e-9);
        assert!((bessel_i0(5.0) - 27.239_871_823_6).abs() < 1e-6);
        assert!((bessel_i0(20.0) / 43_558_282.56 - 1.0).abs() < 1e-4);
    }

    #[test]
    fn the_effective_energy_is_the_exact_gaussian_marginal() {
        // Independent path: for one output the integral is elementary, so compare the closed form
        // against quadrature over x.
        let a = vec![1.7];
        let b = vec![0.35];
        let c = vec![0.8, -0.2];
        let s = vec![0.0];
        let g = PhaseGen::new(1, 1, a, b, c, s).expect("well formed");
        let beta = 1.1;
        for t in [0.0, 0.7, 2.4, 4.9] {
            let phi = [t];
            // Numerically integrate exp(-beta E(x, phi)) over x on a wide grid.
            let lo = -12.0;
            let hi = 12.0;
            let steps = 400_000;
            let h = (hi - lo) / steps as f64;
            let mut acc = Vec::with_capacity(steps);
            for k in 0..steps {
                let x = lo + (k as f64 + 0.5) * h;
                acc.push((-beta * g.energy(&[x], &phi)).exp() * h);
            }
            let quad = sum_up(&acc).ln();
            let closed = -beta * g.effective_energy(&phi) + g.output_log_partition_constant(beta);
            assert!(
                (quad - closed).abs() < 1e-6,
                "quadrature {quad} != closed form {closed} at theta = {t}"
            );
        }
    }

    #[test]
    fn block_gibbs_reproduces_the_enumerated_generator() {
        // The certificate this architecture has and a decoder architecture cannot: the sampler is
        // scored against the exact distribution of the whole generative model.
        let g = fixture();
        let beta = 1.0;
        let q = 16;
        let exact = g.enumerate_grid(q, beta).expect("small enough");
        let mut rng = Pcg::new(2026, 5);
        let mut phi = vec![0.0; 2];
        let mut x = vec![0.0; 2];
        let mut led = Ledger::default();
        for _ in 0..2_000 {
            g.sweep(beta, &mut x, &mut phi, &mut rng, Some(&mut led));
        }
        let draws = 300_000u64;
        let mut hist = vec![0u64; q * q];
        for _ in 0..draws {
            g.sweep(beta, &mut x, &mut phi, &mut rng, Some(&mut led));
            let a0 = ((phi[0] / TAU) * q as f64).round() as usize % q;
            let a1 = ((phi[1] / TAU) * q as f64).round() as usize % q;
            hist[a1 * q + a0] += 1;
        }
        let got: Vec<f64> = hist.iter().map(|&c| c as f64 / draws as f64).collect();
        let tv = 0.5
            * sum_up(
                &got.iter().zip(exact.p.iter()).map(|(a, b)| (a - b).abs()).collect::<Vec<_>>(),
            );
        // The continuous sampler is binned onto the grid, so the comparison carries the covering
        // error of a 16-point grid as well as sampling noise. Both are small and the tolerance says
        // which regime this is: a broken sampler misses by tenths, not hundredths.
        assert!(tv < 0.05, "total variation {tv} against the exact generator is too large");
        assert_eq!(led.samples, (2_000 + draws) * 2);
        assert_eq!(led.reads, (2_000 + draws) * 2);
    }

    #[test]
    fn enumeration_refuses_a_grid_it_cannot_hold() {
        let g = fixture();
        assert_eq!(g.enumerate_grid(2048, 1.0).unwrap_err(), Error::TooBig { q: 2048, n: 2 });
    }
}
