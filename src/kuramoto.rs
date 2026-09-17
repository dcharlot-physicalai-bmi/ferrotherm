//! The general Kuramoto system, its Hodge decomposition, and the exact reduction that puts it
//! inside this crate.
//!
//! # What this is and why it is not [`crate::oim`]
//!
//! [`crate::oim`] is an oscillator **Ising machine**: symmetric `J`, no natural frequencies, a
//! sub-harmonic pump that binarises the phases, and — asserted by construction — a Lyapunov
//! function it can only descend. It is a solver.
//!
//! This module is the system that solver is a special case of:
//!
//! ```text
//!   dtheta_i/dt  =  omega_i  +  sum_j K_ij sin(theta_j - theta_i)
//! ```
//!
//! with `K` **not assumed symmetric** and `omega` **not assumed zero**. That is the object trained
//! by the published coupled-oscillator generative models — a dense `K`, a learned `omega`, a fixed
//! number of explicit Euler steps from random initial phases, and a readout of
//! `(cos theta, sin theta)`. Dropping either assumption destroys the Lyapunov function, and with it
//! every guarantee the rest of this crate is built on. What this module does is say exactly what is
//! lost, exactly what survives, and exactly what the surviving part is worth in this crate's own
//! units.
//!
//! # The decomposition, which is the whole content
//!
//! Split the coupling into its symmetric and antisymmetric parts, `S = (K + K^T)/2` and
//! `A = (K - K^T)/2`. The drift field on the `n`-torus then separates into three pieces that cannot
//! be converted into one another:
//!
//! 1. **Gradient.** The `S` part is exactly `-dE/dtheta` for
//!    `E(theta) = -sum_{i<j} S_ij cos(theta_i - theta_j)` — the **XY model** energy, with `S` as its
//!    couplings. [`Kuramoto::xy_energy`] is that `E` and [`Kuramoto::xy_grad`] is that gradient; the
//!    identity is asserted, not asserted-about.
//! 2. **Solenoidal.** The `A` part has an antisymmetric Jacobian contribution, so it is curl and
//!    nothing else. It admits no potential at all.
//! 3. **Harmonic.** `omega` is constant, hence curl-free — but a constant field on a torus is not
//!    the gradient of any *periodic* function unless it vanishes. It is the harmonic one-form, and
//!    what it produces is winding.
//!
//! The test of membership is exact and pointwise: the drift is a gradient field **iff** the
//! Jacobian is symmetric everywhere, and
//!
//! ```text
//!   dF_i/dtheta_j = K_ij cos(theta_j - theta_i),     dF_j/dtheta_i = K_ji cos(theta_j - theta_i)
//! ```
//!
//! share the cosine. So the Jacobian is symmetric everywhere iff `K_ij = K_ji` for every pair —
//! [`Kuramoto::jacobian_asymmetry`] measures the violation and
//! `the_drift_is_a_gradient_exactly_when_the_coupling_is_symmetric` checks both directions.
//!
//! # The finding this module exists to state
//!
//! It is tempting to read the antisymmetric part as *non-reversible acceleration* — the well-known
//! trick of adding a divergence-free drift to a Langevin sampler, which preserves the target and
//! strictly improves the asymptotic variance (Hwang, Hwang & Sheu, *Accelerating Gaussian
//! diffusions*, Ann. Appl. Probab. 3:897, 1993). It is not that trick, and the difference is
//! measurable here.
//!
//! The `A` part **is** divergence-free — `sum_i dF_{A,i}/dtheta_i = -sum_ij A_ij cos(theta_j -
//! theta_i) = 0`, because `A` is antisymmetric and the cosine matrix is symmetric, and
//! [`Kuramoto::solenoidal_divergence`] returns that zero to rounding. Note that the *full* drift is
//! not: the same contraction against the symmetric `S` does not vanish, which is why
//! [`Kuramoto::divergence`] is a separate function returning a separate number.
//!
//! But preserving the **Boltzmann** measure of `E` needs `div(pi F) = 0`, which expands to
//! `pi (div F - beta grad E . F)`, and for the solenoidal part the surviving term
//! `grad E . F_A` is generically nonzero: [`Kuramoto::gibbs_defect`] is that term, and
//! `an_antisymmetric_coupling_is_divergence_free_but_not_gibbs_preserving` exhibits phases where it
//! is a third of the coupling scale rather than a rounding artefact.
//!
//! So an asymmetrically-coupled Kuramoto system has, in general, **no stationary distribution this
//! crate can name** — not the Boltzmann measure of its own symmetric part, not anything else in
//! closed form. The drift that *does* preserve that measure is `A grad E` rather than the coupling
//! asymmetry, because `div(pi A grad E) = pi tr(A H)` and the trace of an antisymmetric matrix
//! against the symmetric Hessian vanishes identically. That construction lives in
//! [`crate::nonrev`], which is where the acceleration actually is.
//!
//! # The reduction, which is what makes the rest of the crate apply
//!
//! Restrict the phases to the `q`-point grid `theta_i = 2 pi a_i / q`. Then
//! `cos(theta_i - theta_j) = cos(2 pi (a_i - a_j) / q)`, which is exactly the pair term of the
//! **clock model** already in [`crate::potts`] under [`crate::potts::Interaction::Clock`]. So
//! [`Kuramoto::to_clock`] is not an approximation of the symmetric part; on the grid it reproduces
//! the XY energy **to machine precision**, and the test says so with a tolerance of `1e-12`.
//!
//! What the grid costs is stated separately and as a bound. Rounding any continuum configuration to
//! the grid moves each phase by at most `pi/q`, hence each phase difference by at most `2 pi / q`,
//! and the cosine is 1-Lipschitz, so
//!
//! ```text
//!   |E(theta) - E(round_q(theta))|  <=  (2 pi / q) * sum_{i<j} |S_ij|   =:  epsilon(q)
//! ```
//!
//! ([`Kuramoto::covering_gap`]). Two Boltzmann measures whose energies differ by at most `epsilon`
//! pointwise satisfy `|ln Z1 - ln Z2| <= beta epsilon` and `KL <= 2 beta epsilon` — a three-line
//! argument written out at [`Kuramoto::quantisation_kl_bound`]. Since `epsilon(q)` falls like
//! `1/q = 2^-b` in the readout precision `b`, **the KL cost of emulating a continuous phase by `b`
//! bits falls exponentially in `b`**. That is the rate–distortion statement that turns "their
//! variable is continuous and yours is discrete" from an objection into an exchange rate, and
//! [`crate::precision`] is where it is priced in joules.
//!
//! # Dense coupling is a physical claim, and this module makes its cost visible
//!
//! `K` here is dense and row-major, because the published models are dense. That is deliberate and
//! the arithmetic is the point: `n = 16384` all-to-all is `2.68e8` couplings, **2.1 GB** in `f64`
//! ([`dense_coupling_bytes`]), and no fabric of coupled physical oscillators implements it. A
//! substrate claim built on a dense `K` is a claim about a machine that has to be sparsified before
//! it can exist, and the tax for doing so is what [`crate::sparsify`] and [`crate::embed`] already
//! measure.

use crate::ledger::Ledger;
use crate::potts::{Interaction, Potts, PottsBuilder};
use crate::rng::Pcg;
use crate::round::sum_up;

/// Two pi, to the precision of the type.
const TAU: f64 = core::f64::consts::TAU;

/// What can be wrong with a coupled-oscillator system's description.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Error {
    /// Fewer than two oscillators: there is nothing to couple.
    TooSmall {
        /// The count supplied.
        n: usize,
    },
    /// The coupling matrix is not `n * n`.
    ShapeMismatch {
        /// Oscillator count.
        n: usize,
        /// Length of the coupling slice supplied.
        got: usize,
    },
    /// The natural-frequency vector is not length `n`.
    OmegaMismatch {
        /// Oscillator count.
        n: usize,
        /// Length of the frequency slice supplied.
        got: usize,
    },
    /// A coupling or frequency was not finite.
    NotFinite {
        /// Which field carried it.
        what: &'static str,
    },
    /// A self-coupling `K_ii` was nonzero.
    ///
    /// `sin(theta_i - theta_i)` is zero, so the diagonal contributes nothing to the dynamics — but
    /// it contributes to `sum |S_ij|`, and therefore to every bound in this module. A nonzero
    /// diagonal is rejected rather than ignored, because a bound inflated by a term that does not
    /// act is a bound that hides the thing it was computed to expose.
    NonzeroDiagonal {
        /// The offending index.
        i: usize,
    },
    /// An Euler step larger than the descent-lemma limit was requested of a gradient system.
    StepTooLarge {
        /// The step requested.
        dt: f64,
        /// The largest step for which monotone descent is guaranteed.
        limit: f64,
    },
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Error::TooSmall { n } => {
                write!(f, "a coupled system needs at least 2 oscillators, got {n}")
            }
            Error::ShapeMismatch { n, got } => {
                write!(f, "coupling matrix must be {n}x{n} = {} entries, got {got}", n * n)
            }
            Error::OmegaMismatch { n, got } => {
                write!(f, "natural frequencies must be length {n}, got {got}")
            }
            Error::NotFinite { what } => write!(f, "{what} contained a non-finite value"),
            Error::NonzeroDiagonal { i } => write!(
                f,
                "K[{i}][{i}] is nonzero; a self-coupling does not act on the dynamics but does \
                 inflate every bound in this module, so it is refused rather than ignored"
            ),
            Error::StepTooLarge { dt, limit } => write!(
                f,
                "dt = {dt} exceeds the descent limit 1/L = {limit}; a step this large is not a \
                 descent and the caller should say so explicitly"
            ),
        }
    }
}

impl core::error::Error for Error {}

/// Bytes an all-to-all `f64` coupling matrix occupies at `n` oscillators.
///
/// Present because the number is an argument. At the `n = 16384` of a published dense
/// coupled-oscillator generator this is 2,147,483,648 bytes of couplings alone — which is the
/// reason a physical fabric will be sparse, and therefore the reason the embedding tax applies.
#[must_use]
pub fn dense_coupling_bytes(n: usize) -> u128 {
    (n as u128) * (n as u128) * 8
}

/// A system of `n` phase oscillators with arbitrary (possibly asymmetric) coupling.
#[derive(Clone, Debug)]
pub struct Kuramoto {
    n: usize,
    omega: Vec<f64>,
    k: Vec<f64>,
}

impl Kuramoto {
    /// Build from natural frequencies and a dense row-major coupling matrix, `k[i * n + j] = K_ij`.
    ///
    /// # Errors
    ///
    /// [`Error::TooSmall`], [`Error::ShapeMismatch`], [`Error::OmegaMismatch`],
    /// [`Error::NotFinite`] or [`Error::NonzeroDiagonal`] as each describes.
    pub fn new(omega: Vec<f64>, k: Vec<f64>) -> Result<Kuramoto, Error> {
        let n = omega.len();
        if n < 2 {
            return Err(Error::TooSmall { n });
        }
        if k.len() != n * n {
            return Err(Error::ShapeMismatch { n, got: k.len() });
        }
        if omega.iter().any(|v| !v.is_finite()) {
            return Err(Error::NotFinite { what: "omega" });
        }
        if k.iter().any(|v| !v.is_finite()) {
            return Err(Error::NotFinite { what: "K" });
        }
        for i in 0..n {
            if k[i * n + i] != 0.0 {
                return Err(Error::NonzeroDiagonal { i });
            }
        }
        Ok(Kuramoto { n, omega, k })
    }

    /// A symmetric, zero-frequency system: the gradient case, and the one [`crate::oim`] solves.
    ///
    /// # Errors
    ///
    /// As [`Kuramoto::new`].
    pub fn gradient(k: Vec<f64>, n: usize) -> Result<Kuramoto, Error> {
        Kuramoto::new(vec![0.0; n], k)
    }

    /// Oscillator count.
    #[must_use]
    pub fn n(&self) -> usize {
        self.n
    }

    /// Natural frequencies.
    #[must_use]
    pub fn omega(&self) -> &[f64] {
        &self.omega
    }

    /// The coupling `K_ij`.
    ///
    /// # Panics
    ///
    /// If either index is out of range.
    #[must_use]
    pub fn coupling(&self, i: usize, j: usize) -> f64 {
        assert!(i < self.n && j < self.n, "index out of range for {} oscillators", self.n);
        self.k[i * self.n + j]
    }

    /// The symmetric part `S = (K + K^T) / 2`, row-major.
    #[must_use]
    pub fn symmetric_part(&self) -> Vec<f64> {
        let n = self.n;
        let mut s = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                s[i * n + j] = 0.5 * (self.k[i * n + j] + self.k[j * n + i]);
            }
        }
        s
    }

    /// The antisymmetric part `A = (K - K^T) / 2`, row-major.
    #[must_use]
    pub fn antisymmetric_part(&self) -> Vec<f64> {
        let n = self.n;
        let mut a = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                a[i * n + j] = 0.5 * (self.k[i * n + j] - self.k[j * n + i]);
            }
        }
        a
    }

    /// `max |K_ij - K_ji| / 2` — the size of the solenoidal part of the field.
    ///
    /// Zero exactly when the drift is the gradient of [`Kuramoto::xy_energy`].
    #[must_use]
    pub fn asymmetry(&self) -> f64 {
        let n = self.n;
        let mut m = 0.0f64;
        for i in 0..n {
            for j in (i + 1)..n {
                m = m.max(0.5 * (self.k[i * n + j] - self.k[j * n + i]).abs());
            }
        }
        m
    }

    /// `max |omega_i|` — the size of the harmonic part of the field.
    #[must_use]
    pub fn winding(&self) -> f64 {
        self.omega.iter().fold(0.0f64, |m, v| m.max(v.abs()))
    }

    /// Whether the drift is exactly a gradient field: symmetric coupling and no natural frequency.
    ///
    /// Exact equality, not a tolerance. A system that is a gradient field to within `1e-12` is not
    /// a gradient field, and the guarantees that follow from being one do not degrade gracefully.
    #[must_use]
    pub fn is_gradient(&self) -> bool {
        self.asymmetry() == 0.0 && self.winding() == 0.0
    }

    /// The XY energy of the symmetric part: `-sum_{i<j} S_ij cos(theta_i - theta_j)`.
    ///
    /// # Panics
    ///
    /// If `phi` is not length `n`.
    #[must_use]
    pub fn xy_energy(&self, phi: &[f64]) -> f64 {
        assert_eq!(phi.len(), self.n, "phase vector must have one entry per oscillator");
        let n = self.n;
        let mut terms = Vec::with_capacity(n * (n - 1) / 2);
        for i in 0..n {
            for j in (i + 1)..n {
                let s = 0.5 * (self.k[i * n + j] + self.k[j * n + i]);
                terms.push(-s * (phi[i] - phi[j]).cos());
            }
        }
        sum_up(&terms)
    }

    /// `dE/dtheta_i` for the XY energy above, written into `out`.
    ///
    /// # Panics
    ///
    /// If either slice is the wrong length.
    pub fn xy_grad(&self, phi: &[f64], out: &mut [f64]) {
        assert_eq!(phi.len(), self.n, "phase vector must have one entry per oscillator");
        assert_eq!(out.len(), self.n, "gradient buffer must have one entry per oscillator");
        let n = self.n;
        for i in 0..n {
            let mut g = 0.0;
            for j in 0..n {
                if j == i {
                    continue;
                }
                let s = 0.5 * (self.k[i * n + j] + self.k[j * n + i]);
                g += s * (phi[i] - phi[j]).sin();
            }
            out[i] = g;
        }
    }

    /// The full drift `F_i = omega_i + sum_j K_ij sin(theta_j - theta_i)`, written into `out`.
    ///
    /// # Panics
    ///
    /// If either slice is the wrong length.
    pub fn drift(&self, phi: &[f64], out: &mut [f64]) {
        assert_eq!(phi.len(), self.n, "phase vector must have one entry per oscillator");
        assert_eq!(out.len(), self.n, "drift buffer must have one entry per oscillator");
        let n = self.n;
        for i in 0..n {
            let mut f = self.omega[i];
            for j in 0..n {
                if j == i {
                    continue;
                }
                f += self.k[i * n + j] * (phi[j] - phi[i]).sin();
            }
            out[i] = f;
        }
    }

    /// The Jacobian `dF_i/dtheta_j` at `phi`, row-major.
    ///
    /// # Panics
    ///
    /// If `phi` is not length `n`.
    #[must_use]
    pub fn jacobian(&self, phi: &[f64]) -> Vec<f64> {
        assert_eq!(phi.len(), self.n, "phase vector must have one entry per oscillator");
        let n = self.n;
        let mut jac = vec![0.0; n * n];
        for i in 0..n {
            let mut diag = 0.0;
            for j in 0..n {
                if j == i {
                    continue;
                }
                let c = (phi[j] - phi[i]).cos();
                jac[i * n + j] = self.k[i * n + j] * c;
                diag -= self.k[i * n + j] * c;
            }
            jac[i * n + i] = diag;
        }
        jac
    }

    /// The Hessian of [`Kuramoto::xy_energy`] at `phi`, row-major.
    ///
    /// `d2E/dtheta_i dtheta_j = -S_ij cos(theta_i - theta_j)` off the diagonal, and the negated row
    /// sum on it. Symmetric by construction — which is the property
    /// [`crate::nonrev::trace_against`] consumes to prove the skew drift preserves the Boltzmann
    /// measure.
    ///
    /// # Panics
    ///
    /// If `phi` is not length `n`.
    #[must_use]
    pub fn xy_hessian(&self, phi: &[f64]) -> Vec<f64> {
        assert_eq!(phi.len(), self.n, "phase vector must have one entry per oscillator");
        let n = self.n;
        let mut h = vec![0.0; n * n];
        for i in 0..n {
            let mut diag = 0.0;
            for j in 0..n {
                if j == i {
                    continue;
                }
                let s = 0.5 * (self.k[i * n + j] + self.k[j * n + i]);
                let c = s * (phi[i] - phi[j]).cos();
                h[i * n + j] = -c;
                diag += c;
            }
            h[i * n + i] = diag;
        }
        h
    }

    /// `max |J_ij - J_ji|` at `phi`: zero everywhere iff the drift is a gradient field.
    ///
    /// # Panics
    ///
    /// If `phi` is not length `n`.
    #[must_use]
    pub fn jacobian_asymmetry(&self, phi: &[f64]) -> f64 {
        let n = self.n;
        let jac = self.jacobian(phi);
        let mut m = 0.0f64;
        for i in 0..n {
            for j in (i + 1)..n {
                m = m.max((jac[i * n + j] - jac[j * n + i]).abs());
            }
        }
        m
    }

    /// `div F = sum_i dF_i/dtheta_i` at `phi`, for the **full** drift.
    ///
    /// Generally nonzero, and it is worth being precise about why, because the obvious shortcut is
    /// wrong. The Jacobian diagonal sums to `-sum_ij K_ij cos(theta_j - theta_i)`; splitting `K`
    /// into `S + A`, the `A` contraction vanishes against the symmetric cosine matrix but the `S`
    /// contraction does not. So it is the **solenoidal part alone** that is divergence-free —
    /// [`Kuramoto::solenoidal_divergence`] — and a gradient flow is not divergence-free at all,
    /// since `div(-grad E) = -tr H`.
    ///
    /// # Panics
    ///
    /// If `phi` is not length `n`.
    #[must_use]
    pub fn divergence(&self, phi: &[f64]) -> f64 {
        let n = self.n;
        let jac = self.jacobian(phi);
        let diag: Vec<f64> = (0..n).map(|i| jac[i * n + i]).collect();
        sum_up(&diag)
    }

    /// `div F_A` at `phi` — the divergence of the solenoidal part alone.
    ///
    /// Identically zero, at every `phi` and for every coupling: it contracts the antisymmetric `A`
    /// against the symmetric matrix `cos(theta_j - theta_i)`. Returned rather than asserted because
    /// the *value* is the evidence, and because the contrast with [`Kuramoto::divergence`] on the
    /// same instance is what separates "this part is a circulation" from "this field preserves
    /// volume".
    ///
    /// # Panics
    ///
    /// If `phi` is not length `n`.
    #[must_use]
    pub fn solenoidal_divergence(&self, phi: &[f64]) -> f64 {
        assert_eq!(phi.len(), self.n, "phase vector must have one entry per oscillator");
        let n = self.n;
        let a = self.antisymmetric_part();
        let mut terms = Vec::with_capacity(n * n);
        for i in 0..n {
            for j in 0..n {
                if j == i {
                    continue;
                }
                terms.push(-a[i * n + j] * (phi[j] - phi[i]).cos());
            }
        }
        sum_up(&terms)
    }

    /// `grad E . F_A` at `phi` — the obstruction to the antisymmetric part preserving the Boltzmann
    /// measure of [`Kuramoto::xy_energy`].
    ///
    /// Stationarity of `pi = e^{-beta E}` under a drift `F` requires `div(pi F) = 0`, which is
    /// `pi (div F - beta grad E . F)`. The first term vanishes ([`Kuramoto::divergence`]); this is
    /// the second, up to the factor `beta`. A nonzero value is a proof that the system does **not**
    /// sample the Boltzmann distribution of its own symmetric part.
    ///
    /// # Panics
    ///
    /// If `phi` is not length `n`.
    #[must_use]
    pub fn gibbs_defect(&self, phi: &[f64]) -> f64 {
        let n = self.n;
        let mut grad = vec![0.0; n];
        self.xy_grad(phi, &mut grad);
        let a = self.antisymmetric_part();
        let mut terms = Vec::with_capacity(n);
        for i in 0..n {
            let mut f = 0.0;
            for j in 0..n {
                if j == i {
                    continue;
                }
                f += a[i * n + j] * (phi[j] - phi[i]).sin();
            }
            terms.push(grad[i] * f);
        }
        sum_up(&terms)
    }

    /// A Gershgorin bound on the spectral radius of the XY Hessian: `2 max_i sum_j |S_ij|`.
    ///
    /// The Hessian of `E` has `d2E/dtheta_i dtheta_j = -S_ij cos(theta_i - theta_j)` off the
    /// diagonal and `sum_{j != i} S_ij cos(theta_i - theta_j)` on it, so every row sums in magnitude
    /// to at most twice the row's coupling mass, uniformly in `theta`. An explicit Euler step of at
    /// most `1/L` therefore decreases `E` at every step by the descent lemma.
    #[must_use]
    pub fn lipschitz(&self) -> f64 {
        let n = self.n;
        let mut m = 0.0f64;
        for i in 0..n {
            let mut row = 0.0;
            for j in 0..n {
                if j == i {
                    continue;
                }
                row += (0.5 * (self.k[i * n + j] + self.k[j * n + i])).abs();
            }
            m = m.max(2.0 * row);
        }
        m
    }

    /// One explicit Euler step of the **full** drift, charging one device sample per oscillator.
    ///
    /// The integrator the published models use, at the step size they use it at. It is not refused
    /// here for being large, because a non-gradient system has no descent property for a large step
    /// to break — see [`Kuramoto::step_descent`] for the case where there is one to protect.
    ///
    /// # Panics
    ///
    /// If `phi` is not length `n`.
    pub fn step_euler(&self, phi: &mut [f64], dt: f64, ledger: Option<&mut Ledger>) {
        let n = self.n;
        let mut f = vec![0.0; n];
        self.drift(phi, &mut f);
        for i in 0..n {
            phi[i] = (phi[i] + dt * f[i]).rem_euclid(TAU);
        }
        if let Some(l) = ledger {
            l.samples += n as u64;
        }
    }

    /// One Euler step that is **guaranteed** to decrease [`Kuramoto::xy_energy`].
    ///
    /// Refuses a step above `1/L` rather than taking it, on the same reasoning [`crate::oim`] uses:
    /// a machine that is not a descent is a different machine, and the caller should have to say so.
    ///
    /// # Errors
    ///
    /// [`Error::StepTooLarge`] when `dt` exceeds `1 / lipschitz()`, or [`Error::NotFinite`] when the
    /// system is not a gradient field at all — in which case there is no energy to descend.
    ///
    /// # Panics
    ///
    /// If `phi` is not length `n`.
    pub fn step_descent(
        &self,
        phi: &mut [f64],
        dt: f64,
        ledger: Option<&mut Ledger>,
    ) -> Result<(), Error> {
        if !self.is_gradient() {
            return Err(Error::NotFinite { what: "a non-gradient system has no energy to descend" });
        }
        let l = self.lipschitz();
        let limit = if l > 0.0 { 1.0 / l } else { f64::INFINITY };
        if dt > limit {
            return Err(Error::StepTooLarge { dt, limit });
        }
        let n = self.n;
        let mut g = vec![0.0; n];
        self.xy_grad(phi, &mut g);
        for i in 0..n {
            phi[i] = (phi[i] - dt * g[i]).rem_euclid(TAU);
        }
        if let Some(led) = ledger {
            led.samples += n as u64;
        }
        Ok(())
    }

    /// Kuramoto's order parameter `(r, psi)`: `r e^{i psi} = (1/n) sum_j e^{i theta_j}`.
    ///
    /// # Panics
    ///
    /// If `phi` is not length `n`.
    #[must_use]
    pub fn order_parameter(&self, phi: &[f64]) -> (f64, f64) {
        assert_eq!(phi.len(), self.n, "phase vector must have one entry per oscillator");
        let c: Vec<f64> = phi.iter().map(|t| t.cos()).collect();
        let s: Vec<f64> = phi.iter().map(|t| t.sin()).collect();
        let cm = sum_up(&c) / self.n as f64;
        let sm = sum_up(&s) / self.n as f64;
        (cm.hypot(sm), sm.atan2(cm))
    }

    /// The `q`-state clock model whose energy is this system's XY energy restricted to the
    /// `q`-point phase grid.
    ///
    /// Not an approximation on the grid: with `theta_i = 2 pi a_i / q`,
    /// `cos(theta_i - theta_j) = cos(2 pi (a_i - a_j) / q)` is the clock pair term exactly, so
    /// `to_clock(q).energy(a)` equals `xy_energy(grid_phases(a, q))` to rounding. What the grid
    /// costs, relative to the continuum, is [`Kuramoto::covering_gap`].
    ///
    /// The antisymmetric part and the natural frequencies are **dropped**, because neither has any
    /// representation in an energy function. A caller that needs them needs [`crate::nonrev`].
    ///
    /// # Panics
    ///
    /// If `q` is below 2.
    #[must_use]
    pub fn to_clock(&self, q: usize) -> Potts {
        assert!(q >= 2, "a clock model needs at least 2 states, got {q}");
        let n = self.n;
        let mut b = PottsBuilder::new(q, n, Interaction::Clock);
        for i in 0..n {
            for j in (i + 1)..n {
                let s = 0.5 * (self.k[i * n + j] + self.k[j * n + i]);
                if s != 0.0 {
                    b.couple(i, j, s);
                }
            }
        }
        b.build()
    }

    /// Total coupling mass of the symmetric part, `sum_{i<j} |S_ij|`.
    #[must_use]
    pub fn coupling_mass(&self) -> f64 {
        let n = self.n;
        let mut terms = Vec::with_capacity(n * (n - 1) / 2);
        for i in 0..n {
            for j in (i + 1)..n {
                terms.push((0.5 * (self.k[i * n + j] + self.k[j * n + i])).abs());
            }
        }
        sum_up(&terms)
    }

    /// The worst energy error a `q`-point phase grid can cause: `(2 pi / q) * sum_{i<j} |S_ij|`.
    ///
    /// Rounding a phase moves it by at most `pi/q`, a phase *difference* by at most `2 pi / q`, and
    /// the cosine is 1-Lipschitz. An upper bound, not an estimate, and
    /// `the_covering_gap_bounds_the_measured_quantisation_error` checks it against the measured
    /// worst case over random configurations.
    ///
    /// # Panics
    ///
    /// If `q` is below 2.
    #[must_use]
    pub fn covering_gap(&self, q: usize) -> f64 {
        assert!(q >= 2, "a clock model needs at least 2 states, got {q}");
        (TAU / q as f64) * self.coupling_mass()
    }

    /// How many nats of KL a `q`-point readout can cost, at inverse temperature `beta`.
    ///
    /// If two energies differ by at most `epsilon` pointwise then their partition functions differ
    /// by at most a factor `e^{beta epsilon}`, so `|ln Z1 - ln Z2| <= beta epsilon`; and
    /// `KL(p1 || p2) = beta <E2 - E1>_1 + ln(Z2/Z1) <= 2 beta epsilon`. With
    /// `epsilon = covering_gap(q)` falling like `1/q`, the cost of `b = log2 q` bits of phase
    /// precision falls like `2^-b`.
    ///
    /// This is the exchange rate between a continuous dynamical variable and a categorical one, and
    /// it is the reason a continuous-state machine has no information-theoretic advantage that `b`
    /// bits cannot buy. [`crate::precision`] converts it to joules.
    ///
    /// # Panics
    ///
    /// If `q` is below 2.
    #[must_use]
    pub fn quantisation_kl_bound(&self, beta: f64, q: usize) -> f64 {
        2.0 * beta * self.covering_gap(q)
    }
}

/// The phases a clock-model state stands for: `theta_i = 2 pi a_i / q`.
///
/// # Panics
///
/// If `q` is below 2 or any state is not below `q`.
#[must_use]
pub fn grid_phases(a: &[u8], q: usize) -> Vec<f64> {
    assert!(q >= 2, "a clock model needs at least 2 states, got {q}");
    a.iter()
        .map(|&v| {
            assert!(usize::from(v) < q, "state {v} is not below q = {q}");
            TAU * f64::from(v) / q as f64
        })
        .collect()
}

/// Round continuum phases to the nearest `q`-point grid state.
///
/// # Panics
///
/// If `q` is below 2 or above 256.
#[must_use]
pub fn round_to_grid(phi: &[f64], q: usize) -> Vec<u8> {
    assert!((2..=256).contains(&q), "grid size must be in 2..=256, got {q}");
    phi.iter()
        .map(|&t| {
            let scaled = t.rem_euclid(TAU) / TAU * q as f64;
            let r = scaled.round() as usize % q;
            r as u8
        })
        .collect()
}

/// The locked phase difference of a two-oscillator system, or `None` when it cannot lock.
///
/// For `dtheta_1 = omega_1 + k sin(theta_2 - theta_1)` and its partner, the difference obeys
/// `d(delta)/dt = d_omega - 2 k sin(delta)`, so a fixed point exists iff `|d_omega| <= 2 k` and sits
/// at `asin(d_omega / 2k)`. A closed form, and the oracle this module's dynamics are checked
/// against.
#[must_use]
pub fn two_oscillator_locked(d_omega: f64, k: f64) -> Option<f64> {
    if k <= 0.0 || d_omega.abs() > 2.0 * k {
        return None;
    }
    Some((d_omega / (2.0 * k)).asin())
}

/// The drift period of an unlocked two-oscillator system: `2 pi / sqrt(d_omega^2 - 4 k^2)`.
///
/// `None` below threshold, where the system locks instead and the period is infinite.
#[must_use]
pub fn two_oscillator_period(d_omega: f64, k: f64) -> Option<f64> {
    let disc = d_omega * d_omega - 4.0 * k * k;
    (disc > 0.0).then(|| TAU / disc.sqrt())
}

/// A deterministic phase-to-feature map: random initial phases, a fixed number of Euler steps, a
/// `(cos, sin)` readout.
///
/// This is the generative architecture of the published coupled-oscillator models, minus their
/// decoder, and stating it that way is the point. Noise enters **once**, at `t = 0`, as the initial
/// phases; the dynamics that follow are deterministic. So the object is a *pushforward* of the
/// uniform measure on the torus — the same information-theoretic animal as a generator network or a
/// one-step flow — and not a sampler of any distribution the system defines. It has no invariant
/// measure, carries no temperature, and admits no distribution certificate, which is why nothing in
/// [`crate::certify`] applies to it and why this crate reports it as a transport map rather than as
/// a sampler.
///
/// The two facts that make it useful here anyway: it is exactly reproducible from a seed, and every
/// step and every readout lands on the [`Ledger`], so its energy can be priced on any device model
/// in [`crate::ledger`] — including one with an analogue-to-digital readout cost, which is what
/// [`crate::precision`] supplies.
#[derive(Clone, Debug)]
pub struct Pushforward<'k> {
    sys: &'k Kuramoto,
    dt: f64,
    steps: usize,
}

impl<'k> Pushforward<'k> {
    /// A map that integrates `steps` Euler steps of size `dt`.
    #[must_use]
    pub fn new(sys: &'k Kuramoto, dt: f64, steps: usize) -> Pushforward<'k> {
        Pushforward { sys, dt, steps }
    }

    /// The uniform initial phases a seed produces.
    #[must_use]
    pub fn seed_phases(&self, seed: u64) -> Vec<f64> {
        let mut rng = Pcg::new(seed, 0x00B1_A5E5);
        (0..self.sys.n()).map(|_| rng.f64() * TAU).collect()
    }

    /// Run the map and return the terminal phases, charging the ledger for every update.
    #[must_use]
    pub fn phases(&self, seed: u64, mut ledger: Option<&mut Ledger>) -> Vec<f64> {
        let mut phi = self.seed_phases(seed);
        for _ in 0..self.steps {
            self.sys.step_euler(&mut phi, self.dt, ledger.as_deref_mut());
        }
        phi
    }

    /// Run the map and return the `2n` readout features `(cos theta_i, sin theta_i)`, charging one
    /// device read per oscillator.
    ///
    /// One read per **oscillator**, not per feature: the cosine and sine are two coordinates of one
    /// phase, and a fabric that reports a phase reports it once. Charging two would double the one
    /// cost this architecture is most exposed to.
    #[must_use]
    pub fn features(&self, seed: u64, mut ledger: Option<&mut Ledger>) -> Vec<f64> {
        let phi = self.phases(seed, ledger.as_deref_mut());
        if let Some(l) = ledger {
            l.reads += self.sys.n() as u64;
        }
        let mut out = Vec::with_capacity(2 * phi.len());
        for t in &phi {
            out.push(t.cos());
            out.push(t.sin());
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A small asymmetric system with nonzero frequencies, deterministic in its literals.
    fn fixture() -> Kuramoto {
        // 3 oscillators. K is deliberately not symmetric: K_01 = 1.0 but K_10 = 0.25.
        let k = vec![
            0.0, 1.0, -0.5, //
            0.25, 0.0, 0.75, //
            -0.5, 0.75, 0.0, //
        ];
        Kuramoto::new(vec![0.1, -0.2, 0.05], k).expect("fixture is well formed")
    }

    fn symmetric_fixture() -> Kuramoto {
        let k = vec![
            0.0, 1.0, -0.5, //
            1.0, 0.0, 0.75, //
            -0.5, 0.75, 0.0, //
        ];
        Kuramoto::gradient(k, 3).expect("fixture is well formed")
    }

    #[test]
    fn a_nonzero_diagonal_is_refused_rather_than_ignored() {
        let k = vec![0.5, 1.0, 1.0, 0.0];
        assert_eq!(Kuramoto::new(vec![0.0, 0.0], k).unwrap_err(), Error::NonzeroDiagonal { i: 0 });
    }

    #[test]
    fn the_symmetric_drift_is_exactly_minus_the_xy_gradient() {
        // The identity the whole decomposition rests on, checked term by term.
        let sys = symmetric_fixture();
        let phi = [0.3, 1.1, -0.7];
        let mut f = vec![0.0; 3];
        let mut g = vec![0.0; 3];
        sys.drift(&phi, &mut f);
        sys.xy_grad(&phi, &mut g);
        for i in 0..3 {
            assert!(
                (f[i] + g[i]).abs() < 1e-15,
                "drift {} is not minus gradient {} at site {i}",
                f[i],
                g[i]
            );
        }
    }

    #[test]
    fn the_drift_is_a_gradient_exactly_when_the_coupling_is_symmetric() {
        // Both directions. Symmetric K: the Jacobian is symmetric at every phase tried.
        let sym = symmetric_fixture();
        for t in [0.0, 0.4, 1.3, 2.9] {
            let phi = [t, 2.0 * t, -t];
            assert!(
                sym.jacobian_asymmetry(&phi) < 1e-15,
                "symmetric coupling produced an asymmetric Jacobian at t = {t}"
            );
        }
        // Asymmetric K: the Jacobian is NOT symmetric, and the violation is the size of A.
        let asym = fixture();
        let phi = [0.0, 0.0, 0.0];
        // At equal phases every cosine is 1, so |J_ij - J_ji| = |K_ij - K_ji| = 2 * asymmetry.
        assert!((asym.jacobian_asymmetry(&phi) - 2.0 * asym.asymmetry()).abs() < 1e-15);
        assert!(!asym.is_gradient());
    }

    #[test]
    fn an_antisymmetric_coupling_is_divergence_free_but_not_gibbs_preserving() {
        // The finding, in two halves. The SOLENOIDAL divergence vanishes identically...
        let sys = fixture();
        for t in [0.2, 1.0, 2.5] {
            let phi = [t, -0.5 * t, 1.7 * t];
            assert!(
                sys.solenoidal_divergence(&phi).abs() < 1e-14,
                "the solenoidal divergence should vanish identically, got {} at t = {t}",
                sys.solenoidal_divergence(&phi)
            );
        }
        // ...while the full drift's divergence does not, because the symmetric part contracts
        // against the cosine matrix without cancelling. Stating this separately matters: "the
        // antisymmetric part is divergence-free" is true and "the field is divergence-free" is not,
        // and conflating them is how an asymmetric coupling gets mistaken for a measure-preserving
        // perturbation.
        let phi = [0.3, 1.4, -0.8];
        assert!(
            sys.divergence(&phi).abs() > 1e-3,
            "the full divergence should not vanish, got {}",
            sys.divergence(&phi)
        );
        // ...and yet the Boltzmann measure of the symmetric part is not preserved, because the
        // second term of div(pi F) does not vanish. If it did, an asymmetrically coupled oscillator
        // network would sample its own XY model, and it does not.
        // A vector field's defect may vanish at isolated points, so the claim is that it is not
        // identically zero -- scanned, not sampled at one lucky phase.
        let mut worst = 0.0f64;
        for a in 0..12 {
            for b in 0..12 {
                let phi = [0.0, TAU * f64::from(a) / 12.0, TAU * f64::from(b) / 12.0];
                worst = worst.max(sys.gibbs_defect(&phi).abs());
            }
        }
        assert!(worst > 1e-2, "expected a substantial Gibbs defect somewhere, worst was {worst}");
    }

    #[test]
    fn a_symmetric_system_descends_its_energy_at_the_lipschitz_step() {
        let sys = symmetric_fixture();
        let dt = 1.0 / sys.lipschitz();
        let mut phi = vec![0.9, -2.2, 1.4];
        let mut last = sys.xy_energy(&phi);
        for step in 0..200 {
            sys.step_descent(&mut phi, dt, None).expect("gradient system at the limit step");
            let e = sys.xy_energy(&phi);
            assert!(e <= last + 1e-12, "energy rose at step {step}: {last} -> {e}");
            last = e;
        }
    }

    #[test]
    fn a_step_above_the_descent_limit_is_refused() {
        let sys = symmetric_fixture();
        let limit = 1.0 / sys.lipschitz();
        let mut phi = vec![0.1, 0.2, 0.3];
        match sys.step_descent(&mut phi, 2.0 * limit, None) {
            Err(Error::StepTooLarge { .. }) => {}
            other => panic!("expected refusal, got {other:?}"),
        }
    }

    #[test]
    fn two_oscillators_lock_exactly_where_the_closed_form_says() {
        // Below threshold: the simulated difference converges on asin(d_omega / 2k).
        let k = 1.0;
        let d_omega = 1.2;
        let want = two_oscillator_locked(d_omega, k).expect("below threshold");
        let sys = Kuramoto::new(vec![0.0, d_omega], vec![0.0, k, k, 0.0]).expect("well formed");
        let mut phi = vec![0.0, 0.0];
        for _ in 0..200_000 {
            sys.step_euler(&mut phi, 1e-3, None);
        }
        let mut delta = (phi[1] - phi[0]).rem_euclid(TAU);
        if delta > core::f64::consts::PI {
            delta -= TAU;
        }
        assert!((delta - want).abs() < 1e-6, "locked at {delta}, closed form says {want}");
        // At threshold and above there is no fixed point at all.
        assert!(two_oscillator_locked(2.0 * k + 1e-9, k).is_none());
        assert!(two_oscillator_period(2.5, 1.0).is_some());
        assert!(two_oscillator_period(1.2, 1.0).is_none());
    }

    #[test]
    fn the_clock_model_reproduces_the_xy_energy_on_the_grid_exactly() {
        // Not an approximation: on grid phases the two energies are the same number.
        let sys = symmetric_fixture();
        let q = 12;
        let clock = sys.to_clock(q);
        for a0 in 0..q {
            for a1 in 0..q {
                for a2 in 0..q {
                    let a = [a0 as u8, a1 as u8, a2 as u8];
                    let phi = grid_phases(&a, q);
                    let e_clock = clock.energy(&a).expect("valid state");
                    let e_xy = sys.xy_energy(&phi);
                    assert!(
                        (e_clock - e_xy).abs() < 1e-12,
                        "clock {e_clock} != XY {e_xy} at {a:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn the_covering_gap_bounds_the_measured_quantisation_error() {
        let sys = symmetric_fixture();
        let mut rng = Pcg::new(7, 1);
        for &q in &[4_usize, 8, 16, 64] {
            let bound = sys.covering_gap(q);
            let mut worst = 0.0f64;
            for _ in 0..2_000 {
                let phi: Vec<f64> = (0..3).map(|_| rng.f64() * TAU).collect();
                let a = round_to_grid(&phi, q);
                let gap = (sys.xy_energy(&phi) - sys.xy_energy(&grid_phases(&a, q))).abs();
                worst = worst.max(gap);
            }
            assert!(worst <= bound, "measured {worst} exceeded the bound {bound} at q = {q}");
            // And the bound is not vacuous: it is within an order of magnitude of what happens.
            assert!(worst > 0.02 * bound, "bound {bound} is loose against measured {worst}");
        }
    }

    #[test]
    fn the_quantisation_cost_halves_with_every_extra_bit() {
        // The rate-distortion claim, as arithmetic: doubling q halves the KL ceiling.
        let sys = symmetric_fixture();
        let beta = 2.0;
        let a = sys.quantisation_kl_bound(beta, 16);
        let b = sys.quantisation_kl_bound(beta, 32);
        assert!((a / b - 2.0).abs() < 1e-12, "{a} / {b} should be exactly 2");
    }

    #[test]
    fn the_pushforward_is_reproducible_and_charges_what_it_does() {
        let sys = fixture();
        let map = Pushforward::new(&sys, 0.01, 50);
        let mut l1 = Ledger::default();
        let mut l2 = Ledger::default();
        let f1 = map.features(11, Some(&mut l1));
        let f2 = map.features(11, Some(&mut l2));
        assert_eq!(f1, f2, "the same seed must give the same features, bit for bit");
        assert_eq!(l1, l2);
        // 50 steps x 3 oscillators updated, 3 phases read out.
        assert_eq!(l1.samples, 150);
        assert_eq!(l1.reads, 3);
        assert_ne!(map.features(12, None), f1, "a different seed must move the output");
    }

    #[test]
    fn the_order_parameter_is_one_when_locked_and_zero_when_spread() {
        let sys = symmetric_fixture();
        let (r, _) = sys.order_parameter(&[0.5, 0.5, 0.5]);
        assert!((r - 1.0).abs() < 1e-15, "identical phases must give r = 1, got {r}");
        let third = TAU / 3.0;
        let (r, _) = sys.order_parameter(&[0.0, third, 2.0 * third]);
        assert!(r < 1e-15, "evenly spread phases must give r = 0, got {r}");
    }

    #[test]
    fn dense_coupling_at_the_published_scale_is_two_gigabytes() {
        // The number that says a dense K is not a fabric.
        assert_eq!(dense_coupling_bytes(16_384), 2_147_483_648);
    }
}
