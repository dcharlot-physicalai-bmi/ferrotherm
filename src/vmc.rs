//! Variational Monte Carlo with a neural-network wavefunction: the restricted Boltzmann machine as
//! an ansatz for the ground state of the transverse-field Ising model.
//!
//! Carleo & Troyer, *Solving the quantum many-body problem with artificial neural networks*,
//! Science **355**:602 (2017).
//!
//! # Why this belongs in a thermodynamic-sampling crate
//!
//! Everything else here samples a DIAGONAL energy. `E(s) = -sum J s s - sum h s` assigns a number
//! to each configuration and a sampler visits configurations in proportion to `exp(-beta E)`.
//! Add a transverse field and that stops being possible: the Hamiltonian
//!
//! ```text
//!   H = - sum_(i,j) J_ij Z_i Z_j  -  sum_i h_i Z_i  -  Gamma sum_i X_i
//! ```
//!
//! has off-diagonal matrix elements, its ground state is a superposition, and no configuration has
//! a well-defined energy any more. What replaces the energy is the **local energy** of a trial
//! wavefunction,
//!
//! ```text
//!   E_loc(s) = sum_s' H_(s s') psi(s') / psi(s)
//! ```
//!
//! whose average under `|psi|^2` is the variational energy `<psi|H|psi> / <psi|psi>`. That average
//! is a Monte Carlo problem over exactly the state space [`crate::gibbs`] already walks, with the
//! Metropolis ratio `|psi(s')/psi(s)|^2` in place of the Boltzmann one -- so the sampler is the
//! same machine and only the weight changed.
//!
//! Carleo & Troyer's contribution was the ansatz: take the RBM this crate already trains seven ways
//! in [`crate::ebm`], and read it as an amplitude rather than a probability,
//!
//! ```text
//!   psi(s) = exp(sum_i a_i s_i) * prod_j 2 cosh(b_j + sum_i W_ij s_i)
//! ```
//!
//! One hidden unit per visible spin already captures ground states that a product (mean-field)
//! state cannot -- which is the point, and is asserted here rather than claimed, against exact
//! diagonalisation.
//!
//! # Real parameters, and what complex ones would add
//!
//! Every parameter in [`Rbm`] is real, so `psi(s) > 0` for every configuration. That is not a
//! restriction for this Hamiltonian and the reason is worth stating: every off-diagonal element of
//! `H` above is `-Gamma`, so for `Gamma > 0` the matrix has non-positive off-diagonals -- it is
//! **stoquastic** -- and Perron-Frobenius says its ground state can be chosen strictly positive.
//! A strictly positive ansatz is therefore complete for the GROUND STATE of any transverse-field
//! Ising model, frustrated couplings included, and the sign of `Gamma` does not change that: with
//! `U = prod_i Z_i`, `U H(Gamma) U^dagger = H(-Gamma)`, so the two spectra are identical and only
//! the ground state's signs differ.
//!
//! Complex parameters buy three things this module does not do: excited states and any state with a
//! sign or phase structure, real-time evolution (Carleo & Troyer's `t-VMC`, where the phase IS the
//! dynamics), and non-stoquastic Hamiltonians such as an `XY` or fermionic model where the sign
//! problem is intrinsic. Extending [`Rbm`] to complex parameters means carrying `log psi` as a
//! complex number and taking the real part in [`gradient`]; nothing else in the algorithm changes.
//!
//! # What is checked, and against what
//!
//! The variational principle is the one thing a method like this GUARANTEES: for every parameter
//! setting, `<psi|H|psi>/<psi|psi> >= E_0`. A single sign error in [`Tfim::local_energy`] destroys
//! it, and a run whose local energy is wrong still descends smoothly and still reports a number --
//! just a number that is no longer a variational energy, and that lands on whichever side of `E_0`
//! the error happens to fall. So [`Tfim::variational_energy_exact`] computes the true quantity by
//! enumeration, deterministically, accumulated through [`crate::round`] so the claim is not undone
//! by the last bit, and the tests assert it against [`Tfim::ground_energy`] at **every step of a
//! run**. The Monte Carlo estimate is deliberately NOT asserted that way: it is a random variable
//! and may sit below `E_0` by chance, so asserting on it would be either flaky or vacuous.
//!
//! The sharper check is the one that needs no averaging at all. An exact eigenstate has
//! `E_loc(s) = E` at EVERY configuration, so on an uncoupled model -- where a product ansatz IS the
//! ground state, in closed form -- the local energy must be flat to rounding across all `2^n`
//! configurations. Flipping the sign of the transverse term in [`Tfim::local_energy`] moves that
//! test by 15.09 against a tolerance of 1e-13, and takes four others red with it.
//!
//! ```
//! use ferrotherm::{graph::GraphBuilder, vmc};
//!
//! // Two spins, ferromagnetically coupled, in a transverse field: E_0 = -sqrt(J^2 + 4 Gamma^2).
//! let mut gb = GraphBuilder::new(2);
//! gb.couple(0, 1, 1.0);
//! let g = gb.build();
//! let h = vmc::Tfim::new(&g, 0.5);
//!
//! let closed_form = -(1.0f64 + 4.0 * 0.25).sqrt();
//! let ed = h.ground_energy().unwrap();
//! assert!((ed.energy - closed_form).abs() < 1e-12);
//!
//! let mut psi = vmc::Rbm::new(2, 2, 0.1, 7);
//! let p = vmc::Params { steps: 40, samples: 400, ..vmc::Params::default() };
//! let run = vmc::optimize(&h, &mut psi, &p, 11).unwrap();
//! // Variational, so never below the truth.
//! assert!(h.variational_energy_exact(&psi).unwrap() >= ed.energy - 1e-12);
//! assert!(run.acceptance > 0.0);
//! ```

use crate::graph::Graph;
use crate::linalg::jacobi_eig;
use crate::rng::Pcg;
use crate::round::{sum_down, sum_up};

/// The largest system the enumerating routines here will touch.
///
/// [`Tfim::variational_energy_exact`] holds two `2^n` arrays and [`Tfim::ground_energy`] holds a
/// Krylov basis of [`KRYLOV`] of them, which at sixteen spins is 33 MB -- large, bounded, and the
/// point at which "small enough to check exactly" stops being true.
pub const MAX_EXACT_SPINS: usize = 16;

/// Krylov dimension used by [`Tfim::ground_energy`]. Below `2^n` this is a Lanczos approximation
/// and [`Ground::residual`] reports how good; at or above it the subspace is the whole space and
/// the answer is exact to rounding.
pub const KRYLOV: usize = 64;

/// Blocks used for the blocked standard error of a Monte Carlo average; see [`Estimate::stderr`].
const BLOCKS: usize = 32;

/// Stream for the parameter initialiser. The walker, the initialiser and the Lanczos start vector
/// must not share draws, or two runs differing only in their initial parameters would also walk
/// differently.
const STREAM_INIT: u64 = 0xCA7;
/// Stream for the Metropolis walker.
const STREAM_WALK: u64 = 0x1A57;
/// Stream for the Lanczos start vector, seeded from a constant so an exact answer is exact.
const STREAM_LANCZOS: u64 = 0x1A0C;

/// Why a variational Monte Carlo call could not answer.
#[derive(Clone, Debug, PartialEq)]
pub enum VmcError {
    /// Past [`MAX_EXACT_SPINS`], where a `2^n` amplitude vector stops fitting.
    TooLarge {
        /// Spins asked for.
        n: usize,
        /// The cap, which is [`MAX_EXACT_SPINS`].
        max: usize,
    },
    /// The ansatz has a different number of visible units than the model has spins.
    SizeMismatch {
        /// Visible units in the ansatz.
        visible: usize,
        /// Spins in the model.
        spins: usize,
    },
    /// A run was asked for zero samples per step, so no average exists.
    NoSamples,
    /// The ansatz has no visible units, so there is nothing to sample.
    Empty,
    /// The stochastic-reconfiguration metric was not positive definite.
    ///
    /// `S + diag_shift I` is positive definite for any positive shift, so in practice this means a
    /// non-finite entry reached the metric -- which is [`VmcError::Diverged`] one step earlier.
    NotPositiveDefinite {
        /// The pivot that failed, as a parameter index.
        index: usize,
    },
    /// A parameter or an energy stopped being finite.
    Diverged {
        /// The optimisation step it happened on.
        step: usize,
    },
}

impl core::fmt::Display for VmcError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            VmcError::TooLarge { n, max } => write!(
                f,
                "{n} spins is past the cap of {max} for an exact routine; 2^{n} amplitudes do not \
                 fit"
            ),
            VmcError::SizeMismatch { visible, spins } => write!(
                f,
                "the ansatz has {visible} visible units and the model has {spins} spins; one \
                 visible unit per spin is the whole correspondence"
            ),
            VmcError::NoSamples => {
                write!(f, "zero samples per step: an average over nothing is not an energy")
            }
            VmcError::Empty => write!(f, "the ansatz has no visible units"),
            VmcError::NotPositiveDefinite { index } => write!(
                f,
                "the reconfiguration metric failed at parameter {index}; with a positive diagonal \
                 shift that means a non-finite entry reached it"
            ),
            VmcError::Diverged { step } => {
                write!(f, "a parameter or an energy stopped being finite at step {step}")
            }
        }
    }
}

impl std::error::Error for VmcError {}

/// `ln(2 cosh x)`, written so it does not overflow.
///
/// `2 cosh(x)` is `e^x + e^-x`, which is `inf` in `f64` past `x = 710` -- and `x` here is a hidden
/// unit's field, which grows with both the weights and the system size. `|x| + ln(1 + e^-2|x|)` is
/// the same number and is finite everywhere.
#[inline]
fn ln2cosh(x: f64) -> f64 {
    let a = x.abs();
    a + (-2.0 * a).exp().ln_1p()
}

/// One standard normal draw, by Box-Muller.
fn gauss(rng: &mut Pcg) -> f64 {
    // `1 - f64()` is in (0, 1] and never zero, so the logarithm is always finite. `f64()` itself
    // CAN return zero, and `ln(0)` would put an infinity in the initial parameters.
    let u1 = 1.0 - rng.f64();
    let u2 = rng.f64();
    (-2.0 * u1.ln()).sqrt() * (core::f64::consts::TAU * u2).cos()
}

/// A restricted Boltzmann machine read as a wavefunction:
/// `psi(s) = exp(sum_i a_i s_i) * prod_j 2 cosh(b_j + sum_i W_ij s_i)`.
///
/// The hidden units are summed out analytically -- that sum is where the `2 cosh` comes from -- so
/// there is no hidden state to sample and `psi` is a closed-form function of the visible spins.
/// The parameters live in one flat vector so that a gradient, a metric and an update are ordinary
/// dense linear algebra: `a` first, then `b`, then `W` row-major with `W_ij` at
/// `visible + hidden + j * visible + i`.
#[derive(Clone, Debug)]
pub struct Rbm {
    visible: usize,
    hidden: usize,
    p: Vec<f64>,
}

impl Rbm {
    /// An ansatz with every parameter drawn from `N(0, sigma^2)`.
    ///
    /// A zero initialisation is NOT a neutral choice and is the reason `sigma` has no default:
    /// with `W = 0` every `theta_j` is zero, every `tanh(theta_j)` is zero, and so is every
    /// derivative with respect to `b` and `W`. The hidden units would never move, and the machine
    /// would silently be a product state for the whole run.
    #[must_use]
    pub fn new(visible: usize, hidden: usize, sigma: f64, seed: u64) -> Self {
        let mut rng = Pcg::new(seed, STREAM_INIT);
        let np = visible + hidden + hidden * visible;
        let p = (0..np).map(|_| sigma * gauss(&mut rng)).collect();
        Rbm { visible, hidden, p }
    }

    /// The product (mean-field) ansatz: no hidden units, every `a_i` zero.
    ///
    /// `psi(s) = 1` for every configuration, the uniform superposition -- which is the exact ground
    /// state of a model with no couplings and no longitudinal field, and a sensible start
    /// everywhere else. This is the control [`Rbm::new`] is measured against: a product state is
    /// exactly the class of wavefunction that cannot represent entanglement, so the difference
    /// between the two optima is what the hidden units bought.
    #[must_use]
    pub fn product(visible: usize) -> Self {
        Rbm { visible, hidden: 0, p: vec![0.0; visible] }
    }

    /// Visible units, one per spin.
    #[must_use]
    pub fn visible(&self) -> usize {
        self.visible
    }

    /// Hidden units. Zero is a product state.
    #[must_use]
    pub fn hidden(&self) -> usize {
        self.hidden
    }

    /// Parameters in the flat layout this module optimises: `a`, then `b`, then `W` row-major.
    #[must_use]
    pub fn params(&self) -> &[f64] {
        &self.p
    }

    /// Mutable access to the same flat vector, for a caller setting an ansatz by hand.
    pub fn params_mut(&mut self) -> &mut [f64] {
        &mut self.p
    }

    /// Parameter count, `visible + hidden + hidden * visible`.
    #[must_use]
    pub fn n_params(&self) -> usize {
        self.p.len()
    }

    #[inline]
    fn a(&self) -> &[f64] {
        &self.p[..self.visible]
    }

    #[inline]
    fn b(&self) -> &[f64] {
        &self.p[self.visible..self.visible + self.hidden]
    }

    #[inline]
    fn w(&self) -> &[f64] {
        &self.p[self.visible + self.hidden..]
    }

    /// The hidden fields `theta_j = b_j + sum_i W_ij s_i`, which everything else is built from.
    ///
    /// # Panics
    ///
    /// If `s` is shorter than [`Rbm::visible`] -- an ansatz applied to the wrong model.
    #[must_use]
    pub fn theta(&self, s: &[i8]) -> Vec<f64> {
        let mut out = vec![0.0; self.hidden];
        self.theta_into(s, &mut out);
        out
    }

    fn theta_into(&self, s: &[i8], out: &mut [f64]) {
        assert!(s.len() >= self.visible, "an ansatz applied to a state of the wrong size");
        for j in 0..self.hidden {
            let row = &self.w()[j * self.visible..(j + 1) * self.visible];
            let mut t = self.b()[j];
            for i in 0..self.visible {
                t += row[i] * f64::from(s[i]);
            }
            out[j] = t;
        }
    }

    /// `ln psi(s)`. The logarithm, not the amplitude: `psi` itself overflows on any real system.
    ///
    /// # Panics
    ///
    /// If `s` is shorter than [`Rbm::visible`].
    #[must_use]
    pub fn log_psi(&self, s: &[i8]) -> f64 {
        let theta = self.theta(s);
        self.log_psi_with(s, &theta)
    }

    fn log_psi_with(&self, s: &[i8], theta: &[f64]) -> f64 {
        let mut l = 0.0;
        for i in 0..self.visible {
            l += self.a()[i] * f64::from(s[i]);
        }
        for j in 0..self.hidden {
            l += ln2cosh(theta[j]);
        }
        l
    }

    /// `ln(psi(s') / psi(s))` where `s'` is `s` with spin `i` flipped.
    ///
    /// The ratio is what both the Metropolis test and the local energy need, and it is computed
    /// from the flip alone rather than from two evaluations of [`Rbm::log_psi`]: `O(hidden)` work
    /// instead of `O(hidden * visible)`, and no cancellation between two large logarithms.
    ///
    /// # Panics
    ///
    /// If `s` is shorter than [`Rbm::visible`], or `i` is not a visible unit.
    #[must_use]
    pub fn log_ratio_flip(&self, s: &[i8], i: usize) -> f64 {
        let theta = self.theta(s);
        self.log_ratio_with(s, &theta, i)
    }

    #[inline]
    fn log_ratio_with(&self, s: &[i8], theta: &[f64], i: usize) -> f64 {
        let si = f64::from(s[i]);
        let mut d = -2.0 * self.a()[i] * si;
        for j in 0..self.hidden {
            let wji = self.w()[j * self.visible + i];
            d += ln2cosh(theta[j] - 2.0 * wji * si) - ln2cosh(theta[j]);
        }
        d
    }

    /// Flip spin `i` and carry `theta` with it, in `O(hidden)`.
    #[inline]
    fn flip(&self, s: &mut [i8], theta: &mut [f64], i: usize) {
        let si = f64::from(s[i]);
        for j in 0..self.hidden {
            theta[j] -= 2.0 * self.w()[j * self.visible + i] * si;
        }
        s[i] = -s[i];
    }

    /// The log-derivatives `O_k(s) = d ln psi(s) / d p_k`, in the flat parameter layout.
    ///
    /// These are what make the gradient a COVARIANCE rather than a derivative of anything sampled:
    /// `dE/dp_k = 2 (<O_k E_loc> - <O_k><E_loc>)`, so the energy's gradient is estimable from the
    /// same walk that estimates the energy.
    ///
    /// # Panics
    ///
    /// If `s` is shorter than [`Rbm::visible`].
    #[must_use]
    pub fn derivatives(&self, s: &[i8]) -> Vec<f64> {
        let theta = self.theta(s);
        let mut out = vec![0.0; self.n_params()];
        self.derivatives_into(s, &theta, &mut out);
        out
    }

    fn derivatives_into(&self, s: &[i8], theta: &[f64], out: &mut [f64]) {
        let (v, hd) = (self.visible, self.hidden);
        for i in 0..v {
            out[i] = f64::from(s[i]);
        }
        for j in 0..hd {
            let tj = theta[j].tanh();
            out[v + j] = tj;
            for i in 0..v {
                out[v + hd + j * v + i] = tj * f64::from(s[i]);
            }
        }
    }
}

/// The transverse-field Ising Hamiltonian
/// `H = -sum J_ij Z_i Z_j - sum h_i Z_i - Gamma sum_i X_i`.
///
/// The diagonal is [`Graph::energy`] unchanged -- the classical model IS the diagonal of the
/// quantum one, with this crate's sign convention `E = -J s s - h s` -- so a `Tfim` is a borrowed
/// [`Graph`] plus one number.
pub struct Tfim<'g> {
    /// The couplings and longitudinal fields, which are the diagonal of `H`.
    pub g: &'g Graph,
    /// The transverse field `Gamma`. The spectrum is even in it: conjugating by `prod_i Z_i` flips
    /// every `X_i` and leaves every `Z_i`, so `H(-Gamma)` and `H(Gamma)` are unitarily equivalent.
    pub gamma: f64,
}

/// What a Lanczos diagonalisation found.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ground {
    /// The smallest Ritz value. A rigorous **upper** bound on `E_0`: it is the minimum of the
    /// Rayleigh quotient over a subspace, and the minimum over a subspace cannot beat the minimum
    /// over the whole space.
    pub energy: f64,
    /// `||H y - energy * y||` for the Ritz vector `y`, which bounds the distance from
    /// [`Ground::energy`] to the nearest eigenvalue of `H`.
    pub residual: f64,
    /// Krylov steps taken. At `2^n` the subspace exhausted the space and the answer is exact to
    /// rounding.
    pub steps: usize,
}

impl<'g> Tfim<'g> {
    /// The model `g` in a transverse field of `gamma`.
    #[must_use]
    pub fn new(g: &'g Graph, gamma: f64) -> Self {
        Tfim { g, gamma }
    }

    /// `E_loc(s) = sum_s' H_(s s') psi(s') / psi(s)`.
    ///
    /// Two terms, and the whole method rests on the second: the diagonal `H_ss` is the classical
    /// energy, and `X_i` connects `s` to exactly one other configuration, the one with spin `i`
    /// flipped, with matrix element `-Gamma`. So
    ///
    /// ```text
    ///   E_loc(s) = E(s) - Gamma sum_i psi(s^i) / psi(s)
    /// ```
    ///
    /// and `n + 1` evaluations answer a sum that formally runs over all `2^n` configurations --
    /// which is the entire reason variational Monte Carlo is possible.
    ///
    /// # Panics
    ///
    /// If `s` is shorter than the model's spin count.
    #[must_use]
    pub fn local_energy(&self, psi: &Rbm, s: &[i8]) -> f64 {
        let theta = psi.theta(s);
        self.local_energy_with(psi, s, &theta)
    }

    fn local_energy_with(&self, psi: &Rbm, s: &[i8], theta: &[f64]) -> f64 {
        let mut e = self.g.energy(s);
        for i in 0..self.g.n {
            e -= self.gamma * psi.log_ratio_with(s, theta, i).exp();
        }
        e
    }

    /// `<psi|H|psi> / <psi|psi>` by enumeration: the variational energy with no sampling error.
    ///
    /// This is the quantity the variational principle is ABOUT, and it is deterministic, so it is
    /// what a test can assert on. The Monte Carlo estimate in [`Run::history`] is an estimate OF
    /// this number and may sit either side of it.
    ///
    /// Accumulated through [`crate::round`], and the direction is chosen so the returned value is
    /// never BELOW the exact ratio: an upper bound that drifts down is not an upper bound, and this
    /// crate has already shipped one of those. The numerator is bounded above by
    /// [`crate::round::sum_up`]; the denominator is bracketed, and which end of the bracket
    /// maximises the quotient depends on the numerator's sign, which is handled explicitly. What is
    /// NOT claimed is a bound on the per-term `exp` and `tanh` -- those carry their own last-bit
    /// error, and pretending otherwise would be the same overreach in a different place.
    ///
    /// # Errors
    ///
    /// [`VmcError::TooLarge`] past [`MAX_EXACT_SPINS`], and [`VmcError::SizeMismatch`] if the
    /// ansatz was built for a different model.
    pub fn variational_energy_exact(&self, psi: &Rbm) -> Result<f64, VmcError> {
        let n = self.g.n;
        if n > MAX_EXACT_SPINS {
            return Err(VmcError::TooLarge { n, max: MAX_EXACT_SPINS });
        }
        if psi.visible != n {
            return Err(VmcError::SizeMismatch { visible: psi.visible, spins: n });
        }
        let dim = 1usize << n;
        let mut s = vec![0i8; n];
        let mut lw = vec![0.0f64; dim];
        let mut el = vec![0.0f64; dim];
        let mut mx = f64::NEG_INFINITY;
        for (x, item) in lw.iter_mut().enumerate() {
            for i in 0..n {
                s[i] = if x >> i & 1 == 1 { 1 } else { -1 };
            }
            let theta = psi.theta(&s);
            // The weight is |psi|^2, so twice the log amplitude.
            *item = 2.0 * psi.log_psi_with(&s, &theta);
            el[x] = self.local_energy_with(psi, &s, &theta);
            if *item > mx {
                mx = *item;
            }
        }
        // Shift so the largest weight is exactly one: the shift cancels in the ratio and keeps
        // every exponent at or below zero, which is what stops a large ansatz overflowing to inf.
        let w: Vec<f64> = lw.iter().map(|v| (v - mx).exp()).collect();
        let num: Vec<f64> = w.iter().zip(&el).map(|(wi, ei)| wi * ei).collect();
        Ok(ratio_up(sum_up(&num), sum_down(&w), sum_up(&w)))
    }

    /// The exact ground energy, by Lanczos on the `2^n`-dimensional Hamiltonian.
    ///
    /// The oracle every variational claim in this module is measured against, and deliberately not
    /// built out of any of the machinery above: it forms `H` as an operator on amplitudes and
    /// diagonalises it, sharing nothing with the ansatz, the sampler or the local energy.
    ///
    /// The tridiagonal eigenproblem is handed to [`crate::linalg::jacobi_eig`], an existing module
    /// with its own tests, rather than to a second eigensolver written here.
    ///
    /// # Errors
    ///
    /// [`VmcError::TooLarge`] past [`MAX_EXACT_SPINS`], [`VmcError::Empty`] for a model with no
    /// spins.
    pub fn ground_energy(&self) -> Result<Ground, VmcError> {
        let n = self.g.n;
        if n == 0 {
            return Err(VmcError::Empty);
        }
        if n > MAX_EXACT_SPINS {
            return Err(VmcError::TooLarge { n, max: MAX_EXACT_SPINS });
        }
        let dim = 1usize << n;
        let mut s = vec![0i8; n];
        let mut diag = vec![0.0f64; dim];
        for (x, d) in diag.iter_mut().enumerate() {
            for i in 0..n {
                s[i] = if x >> i & 1 == 1 { 1 } else { -1 };
            }
            *d = self.g.energy(&s);
        }
        // The spectrum is even in Gamma (conjugate by the product of all Z), so working with the
        // magnitude costs nothing and buys the Perron-Frobenius argument for the start vector.
        Ok(lanczos(&diag, self.gamma.abs(), n, KRYLOV.min(dim)))
    }
}

/// Upper bound on `num / den` from a bound on the numerator and a bracket on the denominator.
///
/// `den > 0` throughout this module -- it is a sum of squared amplitudes with the largest term
/// normalised to one. Which end of the denominator's bracket maximises the quotient depends on the
/// numerator's SIGN, and a variational energy is usually negative, so this is not academic: taking
/// `num_hi / den_lo` unconditionally would return a number BELOW the exact ratio whenever the
/// numerator is negative, which is the wrong side for an upper bound.
fn ratio_up(num_hi: f64, den_lo: f64, den_hi: f64) -> f64 {
    if num_hi >= 0.0 { num_hi / den_lo.max(f64::MIN_POSITIVE) } else { num_hi / den_hi }
}

/// `H v` for `H = diag - gamma * (sum of single-spin flips)`, in place of a stored matrix.
fn apply_h(diag: &[f64], gamma: f64, n: usize, v: &[f64], out: &mut [f64]) {
    for (x, o) in out.iter_mut().enumerate() {
        let mut acc = diag[x] * v[x];
        for i in 0..n {
            acc -= gamma * v[x ^ (1usize << i)];
        }
        *o = acc;
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Lanczos with full reorthogonalisation, returning the lowest Ritz pair's energy and residual.
fn lanczos(diag: &[f64], gamma: f64, n: usize, max_steps: usize) -> Ground {
    let dim = diag.len();
    let mut rng = Pcg::new(1, STREAM_LANCZOS);
    // A STRICTLY POSITIVE start vector, which is a guarantee rather than an almost-sure statement:
    // with gamma >= 0 every off-diagonal element of H is <= 0, so H is stoquastic, its ground state
    // can be chosen strictly positive, and two strictly positive vectors cannot be orthogonal. A
    // signed random vector is non-orthogonal to the ground state only with probability one, which
    // is a different and weaker thing to rest an oracle on.
    let mut v: Vec<f64> = (0..dim).map(|_| 0.5 + rng.f64()).collect();
    let nrm = dot(&v, &v).sqrt();
    for x in &mut v {
        *x /= nrm;
    }
    // An upper bound on the norm of H, used only to decide when a residual counts as zero.
    let scale = diag.iter().fold(0.0f64, |m, d| m.max(d.abs())) + n as f64 * gamma.abs();
    let floor = 1e-13 * scale.max(1.0);

    let mut basis: Vec<Vec<f64>> = Vec::with_capacity(max_steps);
    let mut alpha: Vec<f64> = Vec::with_capacity(max_steps);
    let mut beta: Vec<f64> = Vec::with_capacity(max_steps);
    let mut w = vec![0.0f64; dim];
    for _ in 0..max_steps {
        basis.push(v.clone());
        let k = basis.len() - 1;
        apply_h(diag, gamma, n, &basis[k], &mut w);
        alpha.push(dot(&w, &basis[k]));
        // Twice, because one pass of classical Gram-Schmidt loses orthogonality exactly where it
        // matters -- and without orthogonality Lanczos reports spurious copies of the eigenvalue it
        // has already found, which looks like convergence and is not.
        for _ in 0..2 {
            for u in &basis {
                let c = dot(&w, u);
                for (wx, ux) in w.iter_mut().zip(u) {
                    *wx -= c * ux;
                }
            }
        }
        let nb = dot(&w, &w).sqrt();
        if nb <= floor {
            break;
        }
        beta.push(nb);
        for (vx, wx) in v.iter_mut().zip(&w) {
            *vx = wx / nb;
        }
    }

    let m = alpha.len();
    let mut t = vec![0.0f64; m * m];
    for k in 0..m {
        t[k * m + k] = alpha[k];
        if k + 1 < m {
            t[k * m + k + 1] = beta[k];
            t[(k + 1) * m + k] = beta[k];
        }
    }
    let vecs = jacobi_eig(&mut t, m);
    let mut best = 0;
    for c in 1..m {
        if t[c * m + c] < t[best * m + best] {
            best = c;
        }
    }
    // The standard Lanczos residual: the norm of H y - theta y is the last beta times the last
    // component of the Ritz vector in the Krylov basis, with no need to form y itself.
    let tail = if beta.len() >= m { beta[m - 1].abs() } else { 0.0 };
    Ground { energy: t[best * m + best], residual: tail * vecs[(m - 1) * m + best].abs(), steps: m }
}

/// A Monte Carlo average and how well it is known.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Estimate {
    /// The sample mean, accumulated through [`crate::round::sum_up`].
    pub mean: f64,
    /// Standard error of [`Estimate::mean`], by BLOCKING: the samples are cut into consecutive
    /// blocks and the error is taken from the spread of the block means.
    ///
    /// A Markov chain's samples are correlated, so the textbook `sigma / sqrt(N)` understates the
    /// error by the square root of the autocorrelation time -- often by a factor of several, which
    /// is enough to make a wrong answer look significant. Blocking absorbs correlations shorter
    /// than a block. It does NOT absorb correlations longer than one, so this remains a lower bound
    /// on the true error when the chain is badly stuck, and is reported as such.
    pub stderr: f64,
    /// Samples behind the average.
    pub samples: usize,
}

fn estimate(e: &[f64]) -> Estimate {
    let n = e.len();
    let mean = sum_up(e) / n as f64;
    let stderr = if n >= 2 * BLOCKS {
        let len = n / BLOCKS;
        let means: Vec<f64> = (0..BLOCKS)
            .map(|b| e[b * len..(b + 1) * len].iter().sum::<f64>() / len as f64)
            .collect();
        let mb = means.iter().sum::<f64>() / BLOCKS as f64;
        let var = means.iter().map(|m| (m - mb) * (m - mb)).sum::<f64>() / (BLOCKS - 1) as f64;
        (var / BLOCKS as f64).sqrt()
    } else if n >= 2 {
        let var = e.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / (n - 1) as f64;
        (var / n as f64).sqrt()
    } else {
        // One sample says nothing about its own spread, and zero would be a lie about that.
        f64::INFINITY
    };
    Estimate { mean, stderr, samples: n }
}

impl Estimate {
    /// The mean plus `sigmas` standard errors -- the number to quote when a claim must not be
    /// beaten by sampling noise.
    #[must_use]
    pub fn upper(&self, sigmas: f64) -> f64 {
        self.mean + sigmas * self.stderr
    }
}

/// How a step is taken from an estimated gradient.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Optimizer {
    /// Plain stochastic gradient descent: `p -= lr * dE/dp`.
    ///
    /// The gradient is in PARAMETER space, where a step of fixed size means nothing in particular:
    /// two parameters can change the wavefunction by wildly different amounts per unit of
    /// themselves, and the steepest descent direction is then not the direction that improves the
    /// state fastest.
    Sgd,
    /// Stochastic reconfiguration (Sorella 1998; Carleo & Troyer's optimiser).
    ///
    /// Solves `S d = dE/dp` for the step, where `S_kl = <O_k O_l> - <O_k><O_l>` is the covariance
    /// of the log-derivatives -- the Fubini-Study metric of the variational manifold, which for a
    /// normalised real ansatz is the Fisher information of `|psi|^2` up to a factor. That makes the
    /// step a distance in WAVEFUNCTION space rather than in parameter space: it is imaginary-time
    /// evolution projected onto the manifold, and it is natural-gradient descent under a different
    /// name.
    Sr {
        /// Absolute shift added to the metric's diagonal before solving.
        ///
        /// `S` is singular whenever two parameters move the wavefunction the same way, which an
        /// over-parameterised ansatz guarantees, and is only ever estimated from a finite sample.
        /// The shift is what makes the solve well-posed; large values interpolate back towards
        /// plain gradient descent, which is the honest way to read it.
        diag_shift: f64,
    },
}

/// How a run is run.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Params {
    /// Optimisation steps.
    pub steps: usize,
    /// Configurations averaged per step.
    pub samples: usize,
    /// Sweeps discarded at the start of each step, after the parameters moved.
    pub therm_sweeps: usize,
    /// Sweeps between kept configurations. A sweep is one proposed flip per visible unit.
    pub sweeps_per_sample: usize,
    /// Step size.
    pub learning_rate: f64,
    /// Which step to take.
    pub optimizer: Optimizer,
    /// Also record the EXACT variational energy of the parameters at every step, by enumeration.
    ///
    /// `2^n` work per step, so a diagnostic for small systems -- and the only way to watch the
    /// variational principle hold step by step, since the Monte Carlo estimate is a random variable
    /// and may fall below `E_0` by chance.
    pub track_exact: bool,
}

impl Default for Params {
    fn default() -> Self {
        Params {
            steps: 200,
            samples: 1000,
            therm_sweeps: 50,
            sweeps_per_sample: 1,
            learning_rate: 0.05,
            optimizer: Optimizer::Sr { diag_shift: 0.01 },
            track_exact: false,
        }
    }
}

/// What a run produced.
#[derive(Clone, Debug)]
pub struct Run {
    /// The Monte Carlo energy at each step, measured with the parameters that step STARTED from.
    pub history: Vec<Estimate>,
    /// The exact variational energy of the same parameters, when [`Params::track_exact`] asked for
    /// it; empty otherwise. Paired index by index with [`Run::history`].
    pub exact: Vec<f64>,
    /// Fraction of proposed single-spin flips that were accepted, over the whole run.
    ///
    /// Worth reading: an acceptance near zero means the chain never moved and every average is one
    /// configuration repeated, which no number of samples fixes.
    pub acceptance: f64,
}

impl Run {
    /// The last step's estimate, or `None` for a run of no steps.
    #[must_use]
    pub fn last(&self) -> Option<Estimate> {
        self.history.last().copied()
    }

    /// The first step whose exact variational energy came within `tol` of `target`, or `None` if no
    /// step did. Always `None` when [`Params::track_exact`] was off, since nothing was recorded.
    ///
    /// This is how convergence SPEED is compared between optimisers without comparing noise: the
    /// exact variational energy is a deterministic function of the parameters, so two runs differ
    /// in it only where the optimiser actually behaved differently.
    #[must_use]
    pub fn first_within(&self, target: f64, tol: f64) -> Option<usize> {
        self.exact.iter().position(|&e| e - target <= tol)
    }
}

/// One Metropolis sweep: one proposed single-spin flip per visible unit. Returns acceptances.
fn sweep(psi: &Rbm, s: &mut [i8], theta: &mut [f64], rng: &mut Pcg) -> u64 {
    let nv = psi.visible;
    let mut acc = 0;
    for _ in 0..nv {
        let i = (rng.next_u32() as usize) % nv;
        let lr = psi.log_ratio_with(s, theta, i);
        // The target is |psi|^2, so the acceptance ratio is exp(2 * ln(psi'/psi)). Taking the
        // uphill case first saves the draw AND keeps exp() away from overflow.
        if lr >= 0.0 || rng.f64() < (2.0 * lr).exp() {
            psi.flip(s, theta, i);
            acc += 1;
        }
    }
    acc
}

/// The energy gradient from a sampled batch: `dE/dp_k = 2 (<O_k E_loc> - <O_k><E_loc>)`.
///
/// The covariance form is what makes this estimable at all. `E = <psi|H|psi>/<psi|psi>` depends on
/// its parameters through both the amplitude and the normalisation, and those two contributions
/// combine into exactly this covariance -- so the term that would need the derivative of a
/// partition function cancels, and a walk that estimates the energy estimates its gradient for
/// free. The factor of two is the two places each real parameter enters, and it is the difference
/// between a correct step size and half of one.
///
/// `o_bar` and `oe` are the per-parameter averages of `O_k` and `O_k E_loc`; `e_bar` is the average
/// local energy.
#[must_use]
pub fn gradient(o_bar: &[f64], oe: &[f64], e_bar: f64) -> Vec<f64> {
    o_bar.iter().zip(oe).map(|(ok, oek)| 2.0 * (oek - ok * e_bar)).collect()
}

/// In-place Cholesky solve of a symmetric positive-definite system.
fn cholesky_solve(a: &mut [f64], n: usize, rhs: &[f64]) -> Result<Vec<f64>, VmcError> {
    for k in 0..n {
        let mut d = a[k * n + k];
        for m in 0..k {
            d -= a[k * n + m] * a[k * n + m];
        }
        if !(d > 0.0) || !d.is_finite() {
            return Err(VmcError::NotPositiveDefinite { index: k });
        }
        let dk = d.sqrt();
        a[k * n + k] = dk;
        for i in (k + 1)..n {
            let mut v = a[i * n + k];
            for m in 0..k {
                v -= a[i * n + m] * a[k * n + m];
            }
            a[i * n + k] = v / dk;
        }
    }
    let mut y = rhs.to_vec();
    for i in 0..n {
        for m in 0..i {
            y[i] -= a[i * n + m] * y[m];
        }
        y[i] /= a[i * n + i];
    }
    for i in (0..n).rev() {
        for m in (i + 1)..n {
            y[i] -= a[m * n + i] * y[m];
        }
        y[i] /= a[i * n + i];
    }
    Ok(y)
}

/// Optimise `psi` towards the ground state of `h`, in place.
///
/// Each step walks `|psi|^2` with Metropolis, averages the local energy and the log-derivatives
/// over that walk, forms the gradient by [`gradient`], and takes a step -- plain for
/// [`Optimizer::Sgd`], through the metric for [`Optimizer::Sr`].
///
/// # Errors
///
/// [`VmcError::Empty`] for an ansatz with no visible units, [`VmcError::SizeMismatch`] if it was
/// built for a different model, [`VmcError::NoSamples`] for a zero sample count,
/// [`VmcError::Diverged`] if a parameter or an energy stops being finite, and
/// [`VmcError::NotPositiveDefinite`] if the reconfiguration metric cannot be factorised.
/// [`Params::track_exact`] additionally propagates [`VmcError::TooLarge`].
pub fn optimize(h: &Tfim, psi: &mut Rbm, p: &Params, seed: u64) -> Result<Run, VmcError> {
    let nv = psi.visible;
    if nv == 0 {
        return Err(VmcError::Empty);
    }
    if nv != h.g.n {
        return Err(VmcError::SizeMismatch { visible: nv, spins: h.g.n });
    }
    if p.samples == 0 {
        return Err(VmcError::NoSamples);
    }
    let np = psi.n_params();
    let sr = matches!(p.optimizer, Optimizer::Sr { .. });

    let mut rng = Pcg::new(seed, STREAM_WALK);
    let mut s: Vec<i8> = (0..nv).map(|_| rng.spin(0.5)).collect();
    let mut theta = psi.theta(&s);

    let mut history = Vec::with_capacity(p.steps);
    let mut exact = Vec::new();
    let mut o = vec![0.0f64; np];
    let mut o_bar = vec![0.0f64; np];
    let mut oe = vec![0.0f64; np];
    let mut moment = if sr { vec![0.0f64; np * np] } else { Vec::new() };
    let mut e_vals = Vec::with_capacity(p.samples);
    let (mut proposed, mut accepted) = (0u64, 0u64);

    for step in 0..p.steps {
        if p.track_exact {
            exact.push(h.variational_energy_exact(psi)?);
        }
        o_bar.fill(0.0);
        oe.fill(0.0);
        moment.fill(0.0);
        e_vals.clear();

        for _ in 0..p.therm_sweeps {
            accepted += sweep(psi, &mut s, &mut theta, &mut rng);
            proposed += nv as u64;
        }
        for _ in 0..p.samples {
            for _ in 0..p.sweeps_per_sample {
                accepted += sweep(psi, &mut s, &mut theta, &mut rng);
                proposed += nv as u64;
            }
            let e = h.local_energy_with(psi, &s, &theta);
            psi.derivatives_into(&s, &theta, &mut o);
            e_vals.push(e);
            for k in 0..np {
                o_bar[k] += o[k];
                oe[k] += o[k] * e;
            }
            if sr {
                for k in 0..np {
                    for l in k..np {
                        moment[k * np + l] += o[k] * o[l];
                    }
                }
            }
        }

        let inv = 1.0 / p.samples as f64;
        for k in 0..np {
            o_bar[k] *= inv;
            oe[k] *= inv;
        }
        let est = estimate(&e_vals);
        if !est.mean.is_finite() {
            return Err(VmcError::Diverged { step });
        }
        let grad = gradient(&o_bar, &oe, est.mean);
        history.push(est);

        let delta = match p.optimizer {
            Optimizer::Sgd => grad,
            Optimizer::Sr { diag_shift } => {
                let mut metric = vec![0.0f64; np * np];
                for k in 0..np {
                    for l in k..np {
                        let v = moment[k * np + l] * inv - o_bar[k] * o_bar[l];
                        metric[k * np + l] = v;
                        metric[l * np + k] = v;
                    }
                    metric[k * np + k] += diag_shift;
                }
                cholesky_solve(&mut metric, np, &grad)?
            }
        };
        for (pk, dk) in psi.p.iter_mut().zip(&delta) {
            *pk -= p.learning_rate * dk;
        }
        if !psi.p.iter().all(|v| v.is_finite()) {
            return Err(VmcError::Diverged { step });
        }
        // The parameters moved, so the cached hidden fields belong to the previous ansatz.
        psi.theta_into(&s, &mut theta);
    }

    let acceptance = if proposed == 0 { 0.0 } else { accepted as f64 / proposed as f64 };
    Ok(Run { history, exact, acceptance })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exact::Elimination;
    use crate::graph::GraphBuilder;
    use crate::ising::tv;

    /// Uncoupled spins carrying the given longitudinal fields.
    fn field_only(h: &[f64]) -> Graph {
        let mut gb = GraphBuilder::new(h.len());
        for (i, &hi) in h.iter().enumerate() {
            gb.bias(i, hi);
        }
        gb.build()
    }

    /// An open chain with uniform coupling.
    fn chain(n: usize, j: f64) -> Graph {
        let mut gb = GraphBuilder::new(n);
        for i in 0..n - 1 {
            gb.couple(i, i + 1, j);
        }
        gb.build()
    }

    /// Four spins, every pair coupled, mixed signs and a longitudinal field on every site.
    ///
    /// Frustrated and asymmetric on purpose: a uniform ferromagnet is symmetric enough that several
    /// wrong signs still produce a plausible number, and this is not.
    fn frustrated4() -> Graph {
        let mut gb = GraphBuilder::new(4);
        for (a, b, j) in
            [(0, 1, 1.0), (0, 2, -0.6), (0, 3, 0.35), (1, 2, 0.8), (1, 3, -0.45), (2, 3, 0.7)]
        {
            gb.couple(a, b, j);
        }
        for (i, hi) in [0.25, -0.4, 0.1, 0.55].into_iter().enumerate() {
            gb.bias(i, hi);
        }
        gb.build()
    }

    /// `E_0 = -sum_i sqrt(h_i^2 + Gamma^2)` for uncoupled spins: each site is a two-by-two problem
    /// `[[-h, -G], [-G, h]]` and the sites do not talk.
    fn non_interacting_energy(h: &[f64], gamma: f64) -> f64 {
        -h.iter().map(|&hi| hi.hypot(gamma)).sum::<f64>()
    }

    /// The product ansatz that IS the ground state of an uncoupled model, in closed form.
    ///
    /// The single-site ground vector of `[[-h, -G], [-G, h]]` has amplitude ratio
    /// `up/down = G / (sqrt(h^2 + G^2) - h)`, and the product ansatz produces `exp(2 a_i)` for that
    /// ratio, so `a_i` is half its logarithm.
    fn product_optimum(h: &[f64], gamma: f64) -> Vec<f64> {
        h.iter().map(|&hi| 0.5 * (gamma / (hi.hypot(gamma) - hi)).ln()).collect()
    }

    /// Spins of the configuration `x` writes in binary, bit set meaning `+1`.
    fn state(x: usize, n: usize) -> Vec<i8> {
        (0..n).map(|i| if x >> i & 1 == 1 { 1 } else { -1 }).collect()
    }

    /// ORACLE: the closed-form spectrum of the transverse-field Ising model, on the systems where it
    /// can be written down.
    ///
    /// ```text
    ///   one spin       H = -h Z - G X               E_0 = -sqrt(h^2 + G^2)
    ///   two spins      H = -J Z Z - G (X + X)       E_0 = -sqrt(J^2 + 4 G^2)
    ///   n uncoupled    H = -sum h Z - G sum X       E_0 = -sum sqrt(h_i^2 + G^2)
    /// ```
    ///
    /// The two-spin case is the one that pins the FACTOR on the transverse field. In the symmetric
    /// subspace spanned by `(uu + dd)` and `(ud + du)` the Hamiltonian is `[[-J, -2G], [-2G, J]]`,
    /// whose eigenvalues are `+-sqrt(J^2 + 4 G^2)`: an assembly that used `G/2` or `2G` still passes
    /// the one-spin case and fails here.
    #[test]
    fn exact_diagonalisation_matches_the_closed_form_tfim_ground_energies() {
        let one = field_only(&[0.7]);
        let e = Tfim::new(&one, 1.3).ground_energy().unwrap();
        let want = 0.7f64.hypot(1.3);
        assert!((e.energy + want).abs() < 1e-12, "one spin: {} vs {}", e.energy, -want);

        let mut gb = GraphBuilder::new(2);
        gb.couple(0, 1, 0.9);
        let two = gb.build();
        let e = Tfim::new(&two, 0.6).ground_energy().unwrap();
        // 0.9^2 + 4 * 0.6^2 = 2.25, so this instance is exactly -1.5.
        assert!((e.energy + 1.5).abs() < 1e-12, "two spins: {}", e.energy);

        let hs = [0.3, -0.7, 1.1, 0.0, 0.45];
        let g = field_only(&hs);
        let e = Tfim::new(&g, 0.8).ground_energy().unwrap();
        let want = non_interacting_energy(&hs, 0.8);
        assert!((e.energy - want).abs() < 1e-12, "five uncoupled: {} vs {want}", e.energy);
        assert!(e.residual < 1e-10, "an exhausted Krylov space must leave no residual: {}", e.residual);
    }

    /// ORACLE: [`crate::exact::Elimination`], an existing module with its own.
    ///
    /// At `Gamma = 0` the Hamiltonian is diagonal and its ground energy IS the classical ground
    /// energy of the same graph, so the two modules must agree exactly. What this checks is the
    /// thing no amount of transverse-field testing reveals: that the diagonal was assembled from
    /// [`Graph::energy`] in the basis the flips are taken in. A transposed or reversed bit order
    /// would leave every closed form above intact -- they are all symmetric under relabelling --
    /// and would break here, where the couplings are not.
    #[test]
    fn exact_diagonalisation_matches_the_elimination_oracle_with_the_field_off() {
        let g = frustrated4();
        let ed = Tfim::new(&g, 0.0).ground_energy().unwrap();
        let classical = Elimination::default().ground_state(&g).unwrap().ground_energy.unwrap();
        assert!((ed.energy - classical).abs() < 1e-12, "{} vs {classical}", ed.energy);
    }

    /// ORACLE: the closed-form single-spin ground state, through the ZERO-VARIANCE PROPERTY.
    ///
    /// `H psi = E psi` read component by component says `E_loc(s) = E` for EVERY configuration, not
    /// on average. So on an uncoupled model, where the exact ground state IS a product state and
    /// [`product_optimum`] writes it down, all `2^5` local energies must equal
    /// `-sum_i sqrt(h_i^2 + Gamma^2)` to rounding.
    ///
    /// This is the sharpest test in the module and the reason it is not an averaged one: a wrong
    /// sign, a factor of two or a dropped term in [`Tfim::local_energy`] makes the spread non-zero
    /// at every single configuration, while an average over `|psi|^2` would hide much of it behind
    /// cancellation.
    #[test]
    fn the_local_energy_is_flat_at_the_closed_form_product_ground_state() {
        let hs = [0.3, -0.7, 1.1, 0.0, 0.45];
        let gamma = 0.8;
        let g = field_only(&hs);
        let h = Tfim::new(&g, gamma);
        let want = non_interacting_energy(&hs, gamma);

        let mut psi = Rbm::product(hs.len());
        psi.params_mut().copy_from_slice(&product_optimum(&hs, gamma));

        let mut worst = 0.0f64;
        for x in 0..(1usize << hs.len()) {
            let s = state(x, hs.len());
            worst = worst.max((h.local_energy(&psi, &s) - want).abs());
        }
        assert!(worst < 1e-13, "local energy is not flat: worst deviation {worst:e}");

        // And therefore the variational energy is the exact one, with no sampling involved.
        let var = h.variational_energy_exact(&psi).unwrap();
        assert!((var - want).abs() < 1e-13, "variational {var} vs exact {want}");
    }

    /// ORACLE: central finite differences of [`Rbm::log_psi`], which shares no line with the
    /// analytic derivative path.
    ///
    /// `O_k` is what turns a walk into a gradient, and it is the one quantity here with no physical
    /// invariant to catch it: a wrong `O_k` still produces a variational energy that is a valid
    /// upper bound, still decreasing, just down a direction that is not the gradient.
    #[test]
    fn the_log_derivatives_match_central_differences_of_log_psi() {
        let mut psi = Rbm::new(4, 3, 0.7, 17);
        let s = [1i8, -1, -1, 1];
        let got = psi.derivatives(&s);
        let eps = 1e-6;
        for k in 0..psi.n_params() {
            let keep = psi.params()[k];
            psi.params_mut()[k] = keep + eps;
            let up = psi.log_psi(&s);
            psi.params_mut()[k] = keep - eps;
            let down = psi.log_psi(&s);
            psi.params_mut()[k] = keep;
            let fd = (up - down) / (2.0 * eps);
            assert!((got[k] - fd).abs() < 1e-9, "parameter {k}: analytic {} against {fd}", got[k]);
        }
    }

    /// ORACLE: `|psi|^2` enumerated over all `2^6` configurations, against the walk that is supposed
    /// to sample it.
    ///
    /// The Metropolis acceptance is `exp(2 ln(psi'/psi))`, and the factor of two is the whole
    /// content: dropping it samples `|psi|` instead of `|psi|^2`, which is a perfectly well-behaved
    /// distribution that no run would ever complain about and every average would be taken under.
    #[test]
    fn the_metropolis_walk_reproduces_the_enumerated_psi_squared() {
        let n = 6;
        let psi = Rbm::new(n, 4, 0.6, 3);
        let dim = 1usize << n;
        let mut want: Vec<f64> = (0..dim).map(|x| (2.0 * psi.log_psi(&state(x, n))).exp()).collect();
        let z: f64 = want.iter().sum();
        for v in &mut want {
            *v /= z;
        }

        let mut rng = Pcg::new(5, 99);
        let mut s: Vec<i8> = (0..n).map(|_| rng.spin(0.5)).collect();
        let mut theta = psi.theta(&s);
        for _ in 0..1000 {
            sweep(&psi, &mut s, &mut theta, &mut rng);
        }
        let draws = 400_000;
        let mut got = vec![0.0; dim];
        for _ in 0..draws {
            sweep(&psi, &mut s, &mut theta, &mut rng);
            let x = (0..n).filter(|&i| s[i] > 0).fold(0usize, |m, i| m | 1 << i);
            got[x] += 1.0;
        }
        for v in &mut got {
            *v /= f64::from(draws);
        }
        let d = tv(&got, &want);
        // The statistical floor for 64 bins at this count is about 0.005; sampling |psi| instead of
        // |psi|^2 with these weights is 0.19, thirty times the threshold.
        assert!(d < 0.01, "total variation from the enumerated |psi|^2 is {d}");
    }

    /// ORACLE: [`Tfim::variational_energy_exact`], the same quantity enumerated rather than sampled.
    ///
    /// A learning rate of zero makes this a pure measurement: one batch, one fixed ansatz, no
    /// update. Both halves of the claim matter -- the estimate must land inside its own error bar,
    /// AND the error bar must be small, or the first half would be satisfied by any implementation
    /// that reported enough uncertainty.
    #[test]
    fn the_monte_carlo_energy_agrees_with_the_enumerated_variational_energy() {
        let g = chain(6, 1.0);
        let h = Tfim::new(&g, 1.0);
        let mut psi = Rbm::new(6, 4, 0.3, 23);
        let p = Params {
            steps: 1,
            samples: 20_000,
            therm_sweeps: 200,
            sweeps_per_sample: 2,
            learning_rate: 0.0,
            optimizer: Optimizer::Sgd,
            track_exact: true,
        };
        let run = optimize(&h, &mut psi, &p, 31).unwrap();
        let est = run.history[0];
        let truth = run.exact[0];
        assert!(est.stderr < 0.05, "the error bar must be tight enough to test with: {}", est.stderr);
        assert!(
            (est.mean - truth).abs() <= 4.0 * est.stderr,
            "sampled {} +- {}, enumerated {truth}",
            est.mean,
            est.stderr
        );
    }

    /// ORACLE: exact diagonalisation, asserted at EVERY step of a run rather than at the end.
    ///
    /// The variational principle is the one thing this method guarantees, and it is guaranteed
    /// pointwise: no parameter setting, however bad, can put `<psi|H|psi>/<psi|psi>` below `E_0`.
    /// So a run that dips below it has not found a better answer, it has a broken local energy --
    /// and the end-of-run number would look like success.
    ///
    /// Asserted on the EXACT variational energy of each step's parameters, not on the Monte Carlo
    /// estimate. The estimate is a random variable and may legitimately sit below `E_0`; asserting
    /// on it would be flaky, and loosening the assertion until it was not would make it vacuous.
    #[test]
    fn every_step_stays_above_exact_diagonalisation_on_a_frustrated_tfim() {
        let g = frustrated4();
        let h = Tfim::new(&g, 0.7);
        let ed = h.ground_energy().unwrap();
        let mut psi = Rbm::new(4, 4, 0.2, 12);
        let p = Params {
            steps: 120,
            samples: 500,
            therm_sweeps: 30,
            sweeps_per_sample: 2,
            learning_rate: 0.05,
            optimizer: Optimizer::Sr { diag_shift: 0.01 },
            track_exact: true,
        };
        let run = optimize(&h, &mut psi, &p, 77).unwrap();
        assert_eq!(run.exact.len(), p.steps);
        for (k, &e) in run.exact.iter().enumerate() {
            assert!(
                e >= ed.energy - 1e-12,
                "step {k} reported {e}, below the exact ground energy {}",
                ed.energy
            );
        }
        let last = h.variational_energy_exact(&psi).unwrap();
        let rel = (last - ed.energy).abs() / ed.energy.abs();
        assert!(rel < 1e-3, "converged to {last} against {} ({rel:e} relative)", ed.energy);
    }

    /// ORACLE: `-sum_i sqrt(h_i^2 + Gamma^2)` and [`product_optimum`], both closed forms.
    ///
    /// REACHES, not approaches. On an uncoupled model the product ansatz contains the exact ground
    /// state, and there the local energy has zero variance -- so the sampled gradient becomes
    /// exactly zero at the optimum however few samples were drawn, and the optimiser stops on the
    /// answer rather than rattling around it. The parameters are checked as well as the energy: an
    /// energy can be right for the wrong state, and the energy is flat to second order in the
    /// parameters so it is the weaker of the two claims.
    #[test]
    fn a_product_ansatz_reaches_the_closed_form_energy_of_a_non_interacting_model() {
        let hs = [0.3, -0.7, 1.1, 0.0, 0.45];
        let gamma = 0.8;
        let g = field_only(&hs);
        let h = Tfim::new(&g, gamma);
        let want = non_interacting_energy(&hs, gamma);

        let mut psi = Rbm::product(hs.len());
        let p = Params {
            steps: 200,
            samples: 500,
            therm_sweeps: 40,
            sweeps_per_sample: 2,
            learning_rate: 0.1,
            optimizer: Optimizer::Sr { diag_shift: 0.01 },
            track_exact: true,
        };
        let run = optimize(&h, &mut psi, &p, 3).unwrap();
        let got = h.variational_energy_exact(&psi).unwrap();
        assert!((got - want).abs() < 1e-12, "reached {got}, exact is {want}");
        for (k, a) in product_optimum(&hs, gamma).into_iter().enumerate() {
            assert!(
                (psi.params()[k] - a).abs() < 1e-4,
                "parameter {k} is {} and the closed form is {a}",
                psi.params()[k]
            );
        }
        let lowest = run.exact.iter().copied().fold(f64::INFINITY, f64::min);
        assert!(lowest >= want - 1e-12, "a step reached {lowest}, below the exact {want}");
    }

    /// ORACLE: exact diagonalisation, as the target both optimisers are timed against.
    ///
    /// A COMPARATIVE claim: stochastic reconfiguration must reach one percent of `E_0` in fewer
    /// steps than plain gradient descent does on the same instance, the same ansatz, the same seed
    /// and the same learning rate. If `Optimizer::Sr` were silently the `Sgd` code path -- a
    /// metric that came out proportional to the identity, a solve that returned its right-hand side
    /// -- the two counts would be equal and this fails.
    ///
    /// Gradient descent is given the ADVANTAGE of a learning-rate sweep and the best count it
    /// achieves anywhere in that sweep, because the obvious objection to the comparison is that
    /// reconfiguration is just a larger effective step. Measured here, at a fixed seed: SGD needs
    /// 133, 51, 31 and 17 steps at rates 0.02 to 0.2, and reconfiguration needs 5 at 0.05.
    ///
    /// Speed is read off [`Tfim::variational_energy_exact`] rather than off the sampled energy, so
    /// the comparison is between two optimisers and not between two noise realisations.
    #[test]
    fn reconfiguration_needs_fewer_steps_than_the_best_sgd_against_exact_diagonalisation() {
        let g = chain(6, 1.0);
        let h = Tfim::new(&g, 1.0);
        let ed = h.ground_energy().unwrap();
        let tol = 0.01 * ed.energy.abs();
        let base = Params {
            steps: 200,
            samples: 500,
            therm_sweeps: 40,
            sweeps_per_sample: 2,
            learning_rate: 0.05,
            optimizer: Optimizer::Sgd,
            track_exact: true,
        };

        let mut best_sgd = usize::MAX;
        for lr in [0.02, 0.05, 0.1, 0.2] {
            let mut psi = Rbm::new(6, 6, 0.1, 4);
            let p = Params { learning_rate: lr, ..base };
            let run = optimize(&h, &mut psi, &p, 9).unwrap();
            if let Some(k) = run.first_within(ed.energy, tol) {
                best_sgd = best_sgd.min(k);
            }
        }
        assert!(best_sgd < usize::MAX, "no gradient-descent rate converged, so nothing is compared");

        let mut psi = Rbm::new(6, 6, 0.1, 4);
        let p = Params { steps: 60, optimizer: Optimizer::Sr { diag_shift: 0.01 }, ..base };
        let run = optimize(&h, &mut psi, &p, 9).unwrap();
        let sr = run.first_within(ed.energy, tol).expect("reconfiguration did not converge");
        assert!(sr < best_sgd, "reconfiguration took {sr} steps, the best gradient descent {best_sgd}");

        let got = h.variational_energy_exact(&psi).unwrap();
        let rel = (got - ed.energy).abs() / ed.energy.abs();
        assert!(rel < 1e-4, "reconfiguration settled at {got} against {} ({rel:e})", ed.energy);
    }

    /// ORACLE: exact diagonalisation, and the product ansatz as the control.
    ///
    /// Carleo & Troyer's claim in one assertion: the hidden units buy representational power a
    /// product state does not have. Same Hamiltonian, same optimiser, same seed, same budget -- the
    /// only difference is whether there are hidden units. The mean-field optimum on this chain is
    /// -6.901 and the network reaches -7.2961 against an exact -7.29623, so the hidden units close
    /// 99.98% of a gap that is 5% of the energy.
    ///
    /// This is also the test that fails if the `W` derivatives are dropped: the machine would still
    /// optimise its visible biases and land exactly on the mean-field number.
    #[test]
    fn hidden_units_beat_the_best_product_state_against_exact_diagonalisation() {
        let g = chain(6, 1.0);
        let h = Tfim::new(&g, 1.0);
        let ed = h.ground_energy().unwrap();
        let p = Params {
            steps: 200,
            samples: 600,
            therm_sweeps: 40,
            sweeps_per_sample: 2,
            learning_rate: 0.05,
            optimizer: Optimizer::Sr { diag_shift: 0.01 },
            track_exact: true,
        };

        let mut mean_field = Rbm::product(6);
        optimize(&h, &mut mean_field, &p, 21).unwrap();
        let e_mf = h.variational_energy_exact(&mean_field).unwrap();

        let mut net = Rbm::new(6, 6, 0.1, 4);
        optimize(&h, &mut net, &p, 21).unwrap();
        let e_nn = h.variational_energy_exact(&net).unwrap();

        assert!(e_mf >= ed.energy, "the product optimum {e_mf} is below the exact {}", ed.energy);
        assert!(e_nn >= ed.energy, "the network optimum {e_nn} is below the exact {}", ed.energy);
        assert!(e_nn < e_mf - 0.3, "network {e_nn} did not beat mean field {e_mf}");
        let closed = (e_mf - e_nn) / (e_mf - ed.energy);
        assert!(closed > 0.99, "the hidden units closed only {closed} of the mean-field gap");
    }

    /// Two runs from one seed are the same run, parameter for parameter.
    ///
    /// The crate's headline promise, and here it is also what makes every comparative claim above
    /// mean anything: a step count that moved between runs would compare noise.
    #[test]
    fn a_run_is_reproducible_from_its_seed() {
        let g = chain(5, 0.8);
        let h = Tfim::new(&g, 0.9);
        let p = Params { steps: 20, samples: 200, ..Params::default() };
        let mut a = Rbm::new(5, 3, 0.2, 8);
        let mut b = Rbm::new(5, 3, 0.2, 8);
        let ra = optimize(&h, &mut a, &p, 55).unwrap();
        let rb = optimize(&h, &mut b, &p, 55).unwrap();
        assert_eq!(a.params(), b.params());
        assert_eq!(ra.acceptance, rb.acceptance);
        assert_eq!(ra.history[19].mean, rb.history[19].mean);
    }

    /// Every refusal is a typed variant that says what was wrong, and prints as a sentence.
    #[test]
    fn refusals_are_typed_and_name_what_was_wrong() {
        let g = chain(4, 1.0);
        let h = Tfim::new(&g, 0.5);
        let p = Params::default();

        let mut wrong = Rbm::new(5, 2, 0.1, 1);
        assert_eq!(
            optimize(&h, &mut wrong, &p, 1).unwrap_err(),
            VmcError::SizeMismatch { visible: 5, spins: 4 }
        );
        assert_eq!(h.variational_energy_exact(&wrong), Err(VmcError::SizeMismatch {
            visible: 5,
            spins: 4
        }));

        let mut psi = Rbm::new(4, 2, 0.1, 1);
        let none = Params { samples: 0, ..p };
        assert_eq!(optimize(&h, &mut psi, &none, 1).unwrap_err(), VmcError::NoSamples);

        let empty = GraphBuilder::new(0).build();
        assert_eq!(Tfim::new(&empty, 0.5).ground_energy(), Err(VmcError::Empty));

        let big = chain(MAX_EXACT_SPINS + 1, 1.0);
        assert_eq!(
            Tfim::new(&big, 0.5).ground_energy(),
            Err(VmcError::TooLarge { n: MAX_EXACT_SPINS + 1, max: MAX_EXACT_SPINS })
        );
        assert!(
            VmcError::NoSamples.to_string().contains("not an energy"),
            "{}",
            VmcError::NoSamples
        );
    }
}
