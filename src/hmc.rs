//! Hamiltonian Monte Carlo and NUTS — the gradient-based samplers for this crate's continuous side.
//!
//! [`crate::multiflip`] brought informed, Langevin-style proposals to the SPIN side. The continuous
//! side had none: [`crate::continuous`] samples a Gaussian–Bernoulli machine coordinate by
//! coordinate, [`crate::tla`] integrates an Ornstein–Uhlenbeck network, [`crate::lrw`] walks an SDE
//! on a lattice. Every one of those moves one coordinate at a time or follows the physics of the
//! model it is given; none of them uses `∇U` to propose a long, distant, *accepted* move.
//!
//! That is what is here:
//!
//! * [`leapfrog`] — the Störmer–Verlet step for `H(q, p) = U(q) + ½ pᵀ M⁻¹ p`, half a kick, a
//!   drift, half a kick. Volume preserving and reversible, both exactly in exact arithmetic, and
//!   on a dyadic fixture — where the `f64` operations are themselves exact — the tests assert both
//!   with `assert_eq!` rather than a tolerance.
//! * [`Hmc`] — draw a momentum, integrate `L` steps, accept on `exp(−ΔH)`
//!   (Duane, Kennedy, Pendleton and Roweth, *Hybrid Monte Carlo*, Physics Letters B 195:216, 1987;
//!   Neal, *MCMC using Hamiltonian dynamics*, Handbook of MCMC ch. 5, 2011).
//! * [`DualAveraging`] — Nesterov dual averaging on the log step size, driving the average
//!   acceptance to a target (Hoffman and Gelman, JMLR 15:1593, 2014, Algorithm 5).
//! * [`Nuts`] — the no-U-turn sampler: recursive doubling of the trajectory until it doubles back
//!   on itself, with slice sampling over the states it visited (Hoffman and Gelman, Algorithm 3;
//!   dual averaging as their Algorithm 6).
//!
//! # Why the discretisation is allowed to be wrong
//!
//! Leapfrog does not conserve `H`. It is allowed not to, because it has the two properties the
//! Metropolis correction needs and no more: the map is **volume preserving** (its Jacobian is
//! exactly 1, being a composition of shears) and **reversible** (negate the momentum, integrate
//! the same number of steps, negate again, and you are back where you started). Those two make
//! `min(1, exp(−ΔH))` an exactly valid acceptance probability for ANY step size, so the sampler is
//! exact even where the integrator is bad — it just rejects more.
//!
//! **Volume preservation and reversibility do not pin the integrator down, and this was measured
//! rather than argued.** `p −= ε g(q); q += ε M⁻¹ p; p −= ε g(q)` — leapfrog with the halves
//! dropped from both kicks — is still a symmetric composition of shears, so it is still volume
//! preserving and still reversible, and it integrates the wrong dynamics. Building it and running
//! the suite: the Jacobian determinant is still **exactly 1.0**, the dyadic round trip is still
//! **bit-for-bit**, and the NUTS invariance test on the logistic still passes, because a wrong
//! integrator inside a correct Metropolis correction is a slow sampler and not a biased one. What
//! goes red is the closed form below.
//!
//! # The closed form that does pin it down
//!
//! On `U(q) = ½ a q²` the leapfrog map is linear, `M = [[c, ε], [−(εa/2)(1+c), c]]` with
//! `c = 1 − ε²a/2`, and a linear map with equal diagonal entries has exactly one invariant
//! quadratic form up to scale: `S = [[−m₂₁, 0], [0, m₁₂]]`. Written as an energy that is
//!
//! ```text
//!   H̃(q, p) = ½ p² + ½ a_eff q²,      a_eff = a (1 − ε² a / 4)
//! ```
//!
//! and leapfrog conserves it **exactly**, at every step size, for as long as you integrate. Two
//! consequences this module's tests are built on:
//!
//! * `ΔH = H − H̃` is `(ε² a² / 8) q²`, so along any trajectory
//!   `ΔH = (ε² a² / 8) (q_L² − q_0²)` — a closed form for the quantity the Metropolis test
//!   consumes, for any `L`, checked against the implementation in
//!   `the_energy_error_matches_its_closed_form_on_a_quadratic`.
//! * UNCORRECTED leapfrog with full momentum refresh has `exp(−H̃)` as its exact stationary
//!   distribution, so it samples `q` with variance `1 / (a(1 − ε²a/4))` rather than `1/a`. At
//!   `a = 1, ε = 1` that is **4/3, not 1** — a 33% error in the variance from dropping one
//!   accept/reject line. [`Hmc::correct`] exists so that contrast can be measured rather than
//!   asserted; it is the only way to prove the correction is wired at all.
//!
//! # The u-turn criterion
//!
//! Doubling stops when the trajectory's two ends start approaching each other, which for the
//! leftmost and rightmost states `(q⁻, p⁻)` and `(q⁺, p⁺)` is `(q⁺ − q⁻) · M⁻¹p⁻ < 0` or
//! `(q⁺ − q⁻) · M⁻¹p⁺ < 0`. It is checked at every subtree as well as at the top, which is what
//! makes the stopping rule symmetric under reversal, and therefore what makes the whole thing
//! reversible. The velocity `M⁻¹p` rather than the momentum `p` is the mass-matrix generalisation;
//! with unit mass they are the same vector and the same criterion the paper prints.
//!
//! [`Nuts::u_turn`] turns it off, leaving fixed-depth doubling — still a valid sampler, simply one
//! that always pays `2^max_depth` gradients. `the_u_turn_criterion_fires_and_shortens_the_trajectory`
//! measures both arms, because a test that only watched the criterion-on arm would pass for an
//! implementation whose criterion never fires.
//!
//! # What this module does NOT round directionally
//!
//! `ΔH` feeds a Metropolis ratio, and a Metropolis ratio is two-sided: accumulating it through
//! [`crate::round::sum_down`] would bias every acceptance in one direction and change the
//! distribution being sampled. Directed rounding belongs to quantities whose direction is a
//! promise, and there is exactly one of those here — [`delta_h_guard`], which bounds how much of a
//! reported `ΔH` could be floating-point noise, and which is a bound, so it goes through
//! [`crate::round::accumulation_guard`].
//!
//! # Where this lands, and where it does not
//!
//! Rust only, deliberately and for now. The C ABI, and everything hanging off it, is a SPIN
//! surface: `ft_*` takes graphs, states and ledgers, and a gradient sampler takes a callback into
//! the caller's own `U` and `∇U`, which is a different kind of boundary and a different lifetime
//! question. Exporting it as it stands would mean either a function-pointer ABI nothing in this
//! tree exercises or a target enum that reaches only the two built-ins here — and a binding that
//! can call one hard-coded Gaussian is a declaration, not a surface. The honest form is a
//! gradient-callback ABI designed as one, and until that exists this paragraph is the gap.
//!
//! ```
//! use ferrotherm::hmc::{Nuts, Quadratic};
//!
//! // N(0, A⁻¹) with A = [[2, 1], [1, 2]], so Var(q₀) = 2/3.
//! let t = Quadratic::new(2, vec![2.0, 1.0, 1.0, 2.0], vec![0.0, 0.0]).unwrap();
//! let mut s = Nuts::new(&t, &[0.0, 0.0], 0.6, 11).unwrap();
//! for _ in 0..200 {
//!     s.draw();
//! }
//! let mut m2 = 0.0;
//! for _ in 0..4000 {
//!     let q = s.draw();
//!     m2 += q[0] * q[0];
//! }
//! let var = m2 / 4000.0;
//! assert!((var - 2.0 / 3.0).abs() < 0.1, "Var(q0) = {var}, want 2/3");
//! ```

use crate::rng::Pcg;

/// How far `H` may rise above the trajectory's starting value before the draw is called divergent.
///
/// Hoffman and Gelman's `Δ_max`, at the 1000 their paper fixes it to. It is a diagnostic threshold
/// and not a tuning knob: a trajectory that has gained a thousand nats of energy has left the
/// typical set, and continuing to double it wastes gradients on states that will never be selected.
pub const DIVERGENCE_LIMIT: f64 = 1000.0;

/// Deepest doubling [`Nuts`] will accept, `2^24` leapfrog steps.
///
/// The tree holds `2^depth` states and its slice counter is a `u64`, so the cap is about overflow
/// as much as about patience: a depth this crate cannot count is a depth it must refuse rather
/// than silently wrap.
pub const MAX_DEPTH_CAP: usize = 24;

/// Smallest step size the adaptation will propose.
const MIN_STEP: f64 = 1e-10;

/// Largest step size the adaptation will propose.
const MAX_STEP: f64 = 1e10;

/// A continuous distribution, as the potential a Hamiltonian sampler needs and its gradient.
///
/// `π(q) ∝ exp(−U(q))`. Only differences of `U` are ever used, so an additive constant — the one
/// nobody can normalise — does not matter and does not have to be found.
pub trait Target {
    /// Number of coordinates. Every slice this trait sees has exactly this length.
    fn dim(&self) -> usize;

    /// `U(q) = −ln π(q)`, up to an additive constant.
    fn potential(&self, q: &[f64]) -> f64;

    /// `∇U(q)`, written into `out`, which is [`Target::dim`] long.
    fn grad(&self, q: &[f64], out: &mut [f64]);
}

/// Why a sampler refused its arguments.
///
/// Every variant names the value it actually saw. A step size of zero is not quietly replaced by
/// something workable, because a chain that ran with a step size nobody chose is a chain whose
/// numbers nobody can reproduce from what they wrote down.
#[derive(Clone, Debug, PartialEq)]
pub enum Invalid {
    /// A step size that is not a finite, strictly positive number.
    StepSize {
        /// The step size as given.
        eps: f64,
    },
    /// A vector whose length disagrees with the target's dimension.
    Shape {
        /// Which vector.
        what: &'static str,
        /// Length given.
        given: usize,
        /// Length required.
        want: usize,
    },
    /// An inverse mass that is not a finite, strictly positive number.
    Mass {
        /// The coordinate.
        index: usize,
        /// The entry of `M⁻¹` exactly as the caller gave it, not the mass derived from it: an
        /// error names what it SAW, and a zero reported back as an infinite mass is a third value
        /// nobody typed.
        inv_mass: f64,
    },
    /// A starting coordinate that is not finite.
    Start {
        /// The coordinate.
        index: usize,
        /// The value as given.
        value: f64,
    },
    /// The potential is not finite at the starting point, so no trajectory can be scored from it.
    StartPotential {
        /// `U(q₀)` as the target returned it.
        potential: f64,
    },
    /// A target acceptance outside the open interval `(0, 1)`.
    TargetAcceptance {
        /// The target as given.
        delta: f64,
    },
    /// A tree depth past [`MAX_DEPTH_CAP`].
    Depth {
        /// The depth as given.
        depth: usize,
        /// The cap.
        cap: usize,
    },
    /// The quadratic form of a [`Quadratic`] target is not positive definite, so it is not a
    /// normalisable density and has no mean or covariance to sample.
    NotPositiveDefinite,
    /// The step-size search doubled or halved its way to the iteration cap without bracketing an
    /// acceptance of one half.
    NoReasonableStep {
        /// The last step size tried.
        eps: f64,
        /// Doublings or halvings spent.
        tries: usize,
    },
}

impl core::fmt::Display for Invalid {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Invalid::StepSize { eps } => {
                write!(f, "step size {eps} is not a finite positive number")
            }
            Invalid::Shape { what, given, want } => {
                write!(f, "{what} has length {given}, and the target has dimension {want}")
            }
            Invalid::Mass { index, inv_mass } => write!(
                f,
                "inverse mass {inv_mass} at coordinate {index} is not a finite positive number, \
                 and the momentum's standard deviation is one over its square root"
            ),
            Invalid::Start { index, value } => {
                write!(f, "starting coordinate {index} is {value}, which is not finite")
            }
            Invalid::StartPotential { potential } => write!(
                f,
                "the target returned U(q0) = {potential}: a trajectory scored from a non-finite \
                 energy accepts or rejects on a comparison with NaN, which is neither"
            ),
            Invalid::TargetAcceptance { delta } => write!(
                f,
                "target acceptance {delta} is outside (0, 1), so no step size can attain it"
            ),
            Invalid::Depth { depth, cap } => {
                write!(f, "tree depth {depth} is past the cap of {cap} ({} states)", 1u64 << cap)
            }
            Invalid::NotPositiveDefinite => write!(
                f,
                "the quadratic form is not positive definite, so exp(-U) is not integrable and has \
                 no covariance to recover"
            ),
            Invalid::NoReasonableStep { eps, tries } => write!(
                f,
                "the step-size search reached {eps:e} after {tries} halvings or doublings without \
                 crossing an acceptance of one half"
            ),
        }
    }
}

impl core::error::Error for Invalid {}

/// `½ pᵀ M⁻¹ p`, the kinetic energy of a momentum under a diagonal mass.
#[must_use]
pub fn kinetic(p: &[f64], inv_mass: &[f64]) -> f64 {
    let mut k = 0.0;
    for i in 0..p.len() {
        k += inv_mass[i] * p[i] * p[i];
    }
    0.5 * k
}

/// `H(q, p) = U(q) + ½ pᵀ M⁻¹ p`, the energy the Metropolis correction is a ratio of.
#[must_use]
pub fn hamiltonian<T: Target + ?Sized>(t: &T, q: &[f64], p: &[f64], inv_mass: &[f64]) -> f64 {
    t.potential(q) + kinetic(p, inv_mass)
}

/// One Störmer–Verlet step of size `eps`, in place.
///
/// Half a kick `p −= (ε/2) ∇U(q)`, a drift `q += ε M⁻¹ p`, half a kick again. A negative `eps`
/// integrates backwards, which is how [`Nuts`] grows a trajectory leftwards and how reversibility
/// is tested.
///
/// Allocates one gradient buffer per call. The samplers in this module keep their own and do not
/// go through here; this is the surface for a caller who wants the map itself.
pub fn leapfrog<T: Target + ?Sized>(
    t: &T,
    q: &mut [f64],
    p: &mut [f64],
    inv_mass: &[f64],
    eps: f64,
) {
    let mut g = vec![0.0; q.len()];
    leapfrog_with(t, q, p, inv_mass, eps, &mut g);
}

/// [`leapfrog`] against a caller-owned gradient buffer.
fn leapfrog_with<T: Target + ?Sized>(
    t: &T,
    q: &mut [f64],
    p: &mut [f64],
    inv_mass: &[f64],
    eps: f64,
    g: &mut [f64],
) {
    let half = 0.5 * eps;
    t.grad(q, g);
    for i in 0..p.len() {
        p[i] -= half * g[i];
    }
    for i in 0..q.len() {
        q[i] += eps * inv_mass[i] * p[i];
    }
    t.grad(q, g);
    for i in 0..p.len() {
        p[i] -= half * g[i];
    }
}

/// A bound on how much of a reported `ΔH` is floating-point noise rather than physics.
///
/// The kinetic half of `H` is `dim` squarings summed, on each end of the trajectory, and the
/// difference costs one more addition: `2·dim + 3` additions whose partial sums never exceed
/// `magnitude`. That is [`crate::round::accumulation_guard`]'s hypothesis exactly, and the count is
/// rounded UP for the same reason that function asks for it to be.
///
/// **The target's own rounding is not covered and cannot be from here** — `U` is code this module
/// did not write and cannot count the additions inside. Pass a `magnitude` that bounds the whole
/// energy, potential included, and read the result as the guard on the arithmetic this module
/// performs on top of whatever `U` returned.
///
/// This is the module's only directed-rounding claim, and it deliberately does not touch the
/// acceptance test: subtracting a guard from `ΔH` before `exp(−ΔH)` would tilt every accept/reject
/// the same way and sample a distribution nobody asked for.
#[must_use]
pub fn delta_h_guard(dim: usize, magnitude: f64) -> f64 {
    crate::round::accumulation_guard(2 * dim + 3, magnitude)
}

// ---------------------------------------------------------------------------------------------
// Targets that can be checked
// ---------------------------------------------------------------------------------------------

/// `U(q) = ½ qᵀA q − bᵀq`, whose density is the Gaussian with mean `A⁻¹b` and covariance `A⁻¹`.
///
/// The one target whose every moment is a closed form, which is why the sampler is checked against
/// it. The mean and covariance are not recomputed here: the test takes them from
/// [`crate::continuous::Gbm::exact_gaussian`], which carries its own verification against Gaussian
/// elimination, so the oracle is a module this one did not write.
#[derive(Clone, Debug)]
pub struct Quadratic {
    /// `A`, row-major `n × n`, symmetric positive definite.
    pub a: Vec<f64>,
    /// `b`, length `n`. The mean is `A⁻¹b`, so `b = 0` centres the target at the origin.
    pub b: Vec<f64>,
    n: usize,
}

impl Quadratic {
    /// A quadratic target over `n` coordinates.
    ///
    /// # Errors
    ///
    /// [`Invalid::Shape`] if `a` is not `n × n` or `b` is not `n` long, and
    /// [`Invalid::NotPositiveDefinite`] if `a` has no Cholesky factor — checked through
    /// [`crate::continuous::cholesky`], because a target whose `exp(−U)` does not integrate has no
    /// distribution for a sampler to be right about.
    pub fn new(n: usize, a: Vec<f64>, b: Vec<f64>) -> Result<Quadratic, Invalid> {
        if a.len() != n * n {
            return Err(Invalid::Shape { what: "a", given: a.len(), want: n * n });
        }
        if b.len() != n {
            return Err(Invalid::Shape { what: "b", given: b.len(), want: n });
        }
        if crate::continuous::cholesky(&a, n).is_none() {
            return Err(Invalid::NotPositiveDefinite);
        }
        Ok(Quadratic { a, b, n })
    }
}

impl Target for Quadratic {
    fn dim(&self) -> usize {
        self.n
    }

    fn potential(&self, q: &[f64]) -> f64 {
        let n = self.n;
        let mut e = 0.0;
        for i in 0..n {
            let mut row = 0.0;
            for k in 0..n {
                row += self.a[i * n + k] * q[k];
            }
            e += 0.5 * row * q[i] - self.b[i] * q[i];
        }
        e
    }

    fn grad(&self, q: &[f64], out: &mut [f64]) {
        let n = self.n;
        for i in 0..n {
            let mut row = 0.0;
            for k in 0..n {
                row += self.a[i * n + k] * q[k];
            }
            out[i] = row - self.b[i];
        }
    }
}

/// A product of standard logistic coordinates: `U(q) = Σ [q_i + 2 ln(1 + e^{−q_i})]`.
///
/// Here because its **cumulative distribution function is elementary** — `F(q) = 1/(1 + e^{−q})`,
/// with an exact inverse — so a bin of the real line has an exactly known probability and an exact
/// draw can be made without any sampler at all. A Gaussian would need `erf`, which `std` does not
/// have, and a quadrature oracle is a weaker thing to check against than a closed form.
///
/// It is also not Gaussian, so the leapfrog error varies over the state space and the Metropolis
/// correction has real work to do — unlike [`Quadratic`], where `ΔH` has the tidy closed form this
/// module's docs open with.
#[derive(Clone, Copy, Debug)]
pub struct Logistic {
    d: usize,
}

impl Logistic {
    /// A product of `d` independent standard logistic coordinates.
    #[must_use]
    pub fn new(d: usize) -> Logistic {
        Logistic { d }
    }

    /// `F(q) = 1/(1 + e^{−q})`, exact and elementary.
    #[must_use]
    pub fn cdf(q: f64) -> f64 {
        if q >= 0.0 { 1.0 / (1.0 + (-q).exp()) } else { q.exp() / (1.0 + q.exp()) }
    }

    /// `F⁻¹(u) = ln(u / (1 − u))`, for `u` in `(0, 1)`.
    ///
    /// An exact draw from the target, which is what makes a one-step invariance test possible:
    /// start from the distribution itself and see whether one transition leaves it there.
    #[must_use]
    pub fn quantile(u: f64) -> f64 {
        (u / (1.0 - u)).ln()
    }
}

/// `ln(1 + e^x)`, without overflowing for large `x` or losing the small-`x` end.
fn softplus(x: f64) -> f64 {
    if x > 0.0 { x + (-x).exp().ln_1p() } else { x.exp().ln_1p() }
}

impl Target for Logistic {
    fn dim(&self) -> usize {
        self.d
    }

    fn potential(&self, q: &[f64]) -> f64 {
        let mut u = 0.0;
        for i in 0..self.d {
            u += q[i] + 2.0 * softplus(-q[i]);
        }
        u
    }

    fn grad(&self, q: &[f64], out: &mut [f64]) {
        for i in 0..self.d {
            out[i] = (0.5 * q[i]).tanh();
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Step-size adaptation
// ---------------------------------------------------------------------------------------------

/// Nesterov dual averaging on `ln ε`, driving the average acceptance to a target.
///
/// Hoffman and Gelman, JMLR 15:1593, Algorithm 5, at the constants their paper fixes:
/// `γ = 0.05`, `t₀ = 10`, `κ = 0.75`, and a shrinkage point `μ = ln(10 ε₀)` that pulls the
/// iterates toward step sizes LARGER than the one the search started from.
///
/// Two step sizes come out and they are not interchangeable. [`DualAveraging::update`] returns the
/// one the next warmup iteration should run at — noisy on purpose, since it is still exploring.
/// [`DualAveraging::averaged`] returns the smoothed `ε̄` that sampling should use, and is the only
/// one of the two that has converged to anything.
#[derive(Clone, Debug)]
pub struct DualAveraging {
    mu: f64,
    delta: f64,
    gamma: f64,
    t0: f64,
    kappa: f64,
    h_bar: f64,
    log_eps_bar: f64,
    m: u64,
}

impl DualAveraging {
    /// An adaptation started at `eps0`, aiming at an average acceptance of `delta`.
    ///
    /// `delta = 0.8` is the usual choice for NUTS and `0.65` for fixed-length HMC; both come from
    /// the same paper's own tuning study.
    ///
    /// # Errors
    ///
    /// [`Invalid::StepSize`] for a non-positive or non-finite `eps0`, and
    /// [`Invalid::TargetAcceptance`] for a `delta` outside `(0, 1)` — an acceptance of exactly 1 is
    /// attained only in the limit `ε → 0`, so it names no step size.
    pub fn new(eps0: f64, delta: f64) -> Result<DualAveraging, Invalid> {
        if !(eps0.is_finite() && eps0 > 0.0) {
            return Err(Invalid::StepSize { eps: eps0 });
        }
        if !(delta > 0.0 && delta < 1.0) {
            return Err(Invalid::TargetAcceptance { delta });
        }
        Ok(DualAveraging {
            mu: (10.0 * eps0).ln(),
            delta,
            gamma: 0.05,
            t0: 10.0,
            kappa: 0.75,
            h_bar: 0.0,
            log_eps_bar: 0.0,
            m: 0,
        })
    }

    /// Fold in one iteration's acceptance statistic and return the step size for the next.
    ///
    /// The statistic is HMC's `min(1, exp(−ΔH))` or NUTS's `α / n_α` over the last doubling, not
    /// the accept/reject bit: a Bernoulli carries the same mean with far more variance, and the
    /// whole point of dual averaging is to spend few iterations.
    ///
    /// A non-finite statistic counts as an acceptance of zero — that is what a divergence is —
    /// rather than poisoning `H̄` with a NaN that never washes out. The returned step size is
    /// clamped to `[1e-10, 1e10]` so that a transient excursion cannot hand the sampler a zero or
    /// an infinity.
    pub fn update(&mut self, accept_stat: f64) -> f64 {
        let a = if accept_stat.is_finite() { accept_stat.clamp(0.0, 1.0) } else { 0.0 };
        self.m += 1;
        let m = self.m as f64;
        let eta = 1.0 / (m + self.t0);
        self.h_bar = (1.0 - eta) * self.h_bar + eta * (self.delta - a);
        let log_eps = self.mu - m.sqrt() / self.gamma * self.h_bar;
        let w = m.powf(-self.kappa);
        self.log_eps_bar = w * log_eps + (1.0 - w) * self.log_eps_bar;
        log_eps.exp().clamp(MIN_STEP, MAX_STEP)
    }

    /// The smoothed step size `ε̄`, which is the one to sample with once warmup ends.
    ///
    /// Before the first [`DualAveraging::update`] this is `exp(0) = 1`, the paper's initialisation
    /// of `ln ε̄₀`, and it means nothing: read it after warmup, not before.
    #[must_use]
    pub fn averaged(&self) -> f64 {
        self.log_eps_bar.exp().clamp(MIN_STEP, MAX_STEP)
    }

    /// Iterations folded in so far.
    #[must_use]
    pub fn iterations(&self) -> u64 {
        self.m
    }

    /// The target acceptance this adaptation is aiming at.
    #[must_use]
    pub fn target_acceptance(&self) -> f64 {
        self.delta
    }
}

/// Hoffman and Gelman's Algorithm 4: double or halve a step size until one leapfrog step crosses an
/// acceptance of one half.
///
/// A starting point for [`DualAveraging`], whose shrinkage target `μ = ln(10 ε₀)` is defined
/// relative to it. It is a heuristic and is documented as one; what it guarantees is only that the
/// returned step size is on the correct side of catastrophe, which is enough to start from.
///
/// # Errors
///
/// [`Invalid::Shape`] if `q` or `inv_mass` disagrees with the target's dimension,
/// [`Invalid::StartPotential`] if `U(q)` is not finite there, and [`Invalid::NoReasonableStep`] if
/// 100 doublings or halvings pass without a crossing — which on a target whose gradient is wrong
/// by a sign is exactly what happens, and is worth an error rather than a shrug.
pub fn reasonable_step_size<T: Target + ?Sized>(
    t: &T,
    q: &[f64],
    inv_mass: &[f64],
    seed: u64,
) -> Result<f64, Invalid> {
    let n = t.dim();
    if q.len() != n {
        return Err(Invalid::Shape { what: "q", given: q.len(), want: n });
    }
    if inv_mass.len() != n {
        return Err(Invalid::Shape { what: "inv_mass", given: inv_mass.len(), want: n });
    }
    let u0 = t.potential(q);
    if !u0.is_finite() {
        return Err(Invalid::StartPotential { potential: u0 });
    }
    let mut rng = Pcg::new(seed, 0x4C46);
    let p0: Vec<f64> = (0..n).map(|i| standard_normal(&mut rng) / inv_mass[i].sqrt()).collect();
    let h0 = u0 + kinetic(&p0, inv_mass);
    let mut eps = 1.0f64;
    let mut g = vec![0.0; n];
    // `log_ratio` is `-ΔH`: positive means the step gained probability.
    let trial = |eps: f64, g: &mut Vec<f64>| -> f64 {
        let (mut q1, mut p1) = (q.to_vec(), p0.clone());
        leapfrog_with(t, &mut q1, &mut p1, inv_mass, eps, g);
        let h1 = t.potential(&q1) + kinetic(&p1, inv_mass);
        if h1.is_finite() { h0 - h1 } else { f64::NEG_INFINITY }
    };
    let mut log_ratio = trial(eps, &mut g);
    let grow = log_ratio > -core::f64::consts::LN_2;
    for tries in 0..100usize {
        if grow != (log_ratio > -core::f64::consts::LN_2) {
            return Ok(eps);
        }
        eps *= if grow { 2.0 } else { 0.5 };
        if !(eps.is_finite() && eps > 0.0) {
            return Err(Invalid::NoReasonableStep { eps, tries });
        }
        log_ratio = trial(eps, &mut g);
    }
    Err(Invalid::NoReasonableStep { eps, tries: 100 })
}

/// One standard normal from two uniforms, by Box–Muller.
///
/// The cosine half only, matching [`crate::continuous`]: keeping the sine would halve the uniforms
/// spent and would put a cached value in the sampler's state, which is one more thing that has to
/// be right for "same seed, same draws" to hold.
fn standard_normal(rng: &mut Pcg) -> f64 {
    let u1 = rng.f64().max(f64::MIN_POSITIVE);
    let u2 = rng.f64();
    (-2.0 * u1.ln()).sqrt() * (core::f64::consts::TAU * u2).cos()
}

/// The checked pieces every sampler needs before it can take a step.
struct Setup {
    /// The starting state, copied.
    q: Vec<f64>,
    /// `M⁻¹`, defaulted to ones only when the caller asked for the default.
    inv_mass: Vec<f64>,
    /// `sqrt(M)`, the standard deviation each momentum coordinate is drawn at.
    sd: Vec<f64>,
}

/// Check and unpack the pieces every sampler needs, once.
fn setup<T: Target + ?Sized>(
    t: &T,
    q0: &[f64],
    eps: f64,
    inv_mass: Option<&[f64]>,
) -> Result<Setup, Invalid> {
    let n = t.dim();
    if !(eps.is_finite() && eps > 0.0) {
        return Err(Invalid::StepSize { eps });
    }
    if q0.len() != n {
        return Err(Invalid::Shape { what: "q0", given: q0.len(), want: n });
    }
    for (i, &v) in q0.iter().enumerate() {
        if !v.is_finite() {
            return Err(Invalid::Start { index: i, value: v });
        }
    }
    let inv = match inv_mass {
        None => vec![1.0; n],
        Some(m) => {
            if m.len() != n {
                return Err(Invalid::Shape { what: "inv_mass", given: m.len(), want: n });
            }
            for (i, &v) in m.iter().enumerate() {
                if !(v.is_finite() && v > 0.0) {
                    return Err(Invalid::Mass { index: i, inv_mass: v });
                }
            }
            m.to_vec()
        }
    };
    let u0 = t.potential(q0);
    if !u0.is_finite() {
        return Err(Invalid::StartPotential { potential: u0 });
    }
    let sd: Vec<f64> = inv.iter().map(|&v| 1.0 / v.sqrt()).collect();
    Ok(Setup { q: q0.to_vec(), inv_mass: inv, sd })
}

// ---------------------------------------------------------------------------------------------
// Hamiltonian Monte Carlo
// ---------------------------------------------------------------------------------------------

/// Fixed-length Hamiltonian Monte Carlo: refresh the momentum, integrate `L` leapfrog steps, accept
/// on `exp(−ΔH)`.
///
/// Exact for any step size the integrator does not blow up at, because the correction is exact —
/// see the module docs for what happens when it is removed, which [`Hmc::correct`] can do.
pub struct Hmc<'t, T: Target + ?Sized> {
    target: &'t T,
    q: Vec<f64>,
    p: Vec<f64>,
    grad: Vec<f64>,
    inv_mass: Vec<f64>,
    sd: Vec<f64>,
    eps: f64,
    steps: usize,
    correct: bool,
    rng: Pcg,
    proposed: u64,
    accepted: u64,
    gradients: u64,
    accept_stat: f64,
    last_delta_h: f64,
}

impl<'t, T: Target + ?Sized> Hmc<'t, T> {
    /// A sampler at `q0` with step size `eps` and `steps` leapfrog steps per draw, unit mass.
    ///
    /// # Errors
    ///
    /// [`Invalid::StepSize`], [`Invalid::Shape`], [`Invalid::Start`] or
    /// [`Invalid::StartPotential`], each naming the value it saw.
    pub fn new(
        target: &'t T,
        q0: &[f64],
        eps: f64,
        steps: usize,
        seed: u64,
    ) -> Result<Hmc<'t, T>, Invalid> {
        Self::with_mass(target, q0, eps, steps, None, seed)
    }

    /// A sampler with a diagonal mass, given as its INVERSE `M⁻¹` — the form every formula uses.
    ///
    /// # Errors
    ///
    /// As [`Hmc::new`], plus [`Invalid::Mass`] for an entry that is not finite and positive.
    pub fn with_mass(
        target: &'t T,
        q0: &[f64],
        eps: f64,
        steps: usize,
        inv_mass: Option<&[f64]>,
        seed: u64,
    ) -> Result<Hmc<'t, T>, Invalid> {
        let s = setup(target, q0, eps, inv_mass)?;
        let n = s.q.len();
        Ok(Hmc {
            target,
            q: s.q,
            p: vec![0.0; n],
            grad: vec![0.0; n],
            inv_mass: s.inv_mass,
            sd: s.sd,
            eps,
            steps,
            correct: true,
            rng: Pcg::new(seed, 0x484D),
            proposed: 0,
            accepted: 0,
            gradients: 0,
            accept_stat: 0.0,
            last_delta_h: 0.0,
        })
    }

    /// Whether the Metropolis correction is applied. `true` is the sampler; `false` is a
    /// measurement instrument.
    ///
    /// With `false` every proposal is kept, which makes this uncorrected leapfrog and NOT a sampler
    /// for the target: on `U = ½ a q²` it samples a Gaussian of variance `1/(a(1 − ε²a/4))`
    /// exactly, which the module docs derive and
    /// `the_metropolis_correction_is_what_makes_the_variance_right` measures. It is public because
    /// a test that only runs the corrected arm passes for an implementation that never looks at
    /// `ΔH`.
    pub fn correct(&mut self, on: bool) {
        self.correct = on;
    }

    /// Replace the step size.
    ///
    /// # Errors
    ///
    /// [`Invalid::StepSize`] for a value that is not finite and positive.
    pub fn set_step_size(&mut self, eps: f64) -> Result<(), Invalid> {
        if !(eps.is_finite() && eps > 0.0) {
            return Err(Invalid::StepSize { eps });
        }
        self.eps = eps;
        Ok(())
    }

    /// One draw. The returned state is the chain's next sample, accepted or repeated.
    pub fn draw(&mut self) -> &[f64] {
        for i in 0..self.p.len() {
            self.p[i] = self.sd[i] * standard_normal(&mut self.rng);
        }
        let h0 = self.target.potential(&self.q) + kinetic(&self.p, &self.inv_mass);
        let q0 = self.q.clone();
        for _ in 0..self.steps {
            leapfrog_with(
                self.target,
                &mut self.q,
                &mut self.p,
                &self.inv_mass,
                self.eps,
                &mut self.grad,
            );
        }
        self.gradients += 2 * self.steps as u64;
        let h1 = self.target.potential(&self.q) + kinetic(&self.p, &self.inv_mass);
        let dh = h1 - h0;
        self.last_delta_h = dh;
        let a = if dh.is_nan() { 0.0 } else { (-dh).exp().min(1.0) };
        self.accept_stat = a;
        self.proposed += 1;
        if !self.correct || self.rng.f64() < a {
            self.accepted += 1;
        } else {
            self.q = q0;
        }
        &self.q
    }

    /// Run `iters` draws adapting the step size by dual averaging, then set the smoothed `ε̄` and
    /// return it.
    ///
    /// # Errors
    ///
    /// [`Invalid::TargetAcceptance`] for a `delta` outside `(0, 1)`. The step sizes the adaptation
    /// proposes are clamped before they are applied, so no error comes back from those.
    pub fn warmup(&mut self, iters: usize, delta: f64) -> Result<f64, Invalid> {
        let mut da = DualAveraging::new(self.eps, delta)?;
        for _ in 0..iters {
            self.draw();
            let next = da.update(self.accept_stat);
            self.set_step_size(next)?;
        }
        let bar = da.averaged();
        self.set_step_size(bar)?;
        Ok(bar)
    }

    /// The current state.
    #[must_use]
    pub fn state(&self) -> &[f64] {
        &self.q
    }

    /// Move the chain to `q`, discarding the current state.
    ///
    /// # Errors
    ///
    /// [`Invalid::Shape`], [`Invalid::Start`] or [`Invalid::StartPotential`].
    pub fn set_state(&mut self, q: &[f64]) -> Result<(), Invalid> {
        self.q = setup(self.target, q, self.eps, Some(&self.inv_mass))?.q;
        Ok(())
    }

    /// Fraction of proposals accepted so far, or `None` before the first draw.
    #[must_use]
    pub fn acceptance(&self) -> Option<f64> {
        (self.proposed > 0).then(|| self.accepted as f64 / self.proposed as f64)
    }

    /// `min(1, exp(−ΔH))` for the most recent proposal — the statistic dual averaging consumes.
    #[must_use]
    pub fn accept_stat(&self) -> f64 {
        self.accept_stat
    }

    /// `ΔH` for the most recent proposal, signed: positive is energy the integrator invented.
    #[must_use]
    pub fn last_delta_h(&self) -> f64 {
        self.last_delta_h
    }

    /// The current step size.
    #[must_use]
    pub fn step_size(&self) -> f64 {
        self.eps
    }

    /// Gradient evaluations spent, which is the currency a gradient sampler is actually billed in.
    #[must_use]
    pub fn gradients(&self) -> u64 {
        self.gradients
    }
}

// ---------------------------------------------------------------------------------------------
// NUTS
// ---------------------------------------------------------------------------------------------

/// One subtree, as Hoffman and Gelman's `BuildTree` returns it.
struct Sub {
    q_minus: Vec<f64>,
    p_minus: Vec<f64>,
    q_plus: Vec<f64>,
    p_plus: Vec<f64>,
    /// The state this subtree offers to the progressive selection.
    q_prop: Vec<f64>,
    /// States in the slice, `n'`.
    n: u64,
    /// `s'`: whether the subtree may still be extended.
    ok: bool,
    /// `Σ min(1, exp(−ΔH))` over every state built, for dual averaging.
    alpha: f64,
    /// How many states that sum has in it.
    n_alpha: u64,
}

/// The no-U-turn sampler: double the trajectory until it turns back on itself.
///
/// Hoffman and Gelman, JMLR 15:1593, Algorithm 3 — the memory-efficient recursion, with slice
/// sampling over the trajectory and the progressive `min(1, n'/n)` selection that lets a doubling
/// be discarded without storing it. Algorithm 6's acceptance statistic is carried alongside so
/// [`Nuts::warmup`] can adapt the step size from it.
pub struct Nuts<'t, T: Target + ?Sized> {
    target: &'t T,
    q: Vec<f64>,
    grad: Vec<f64>,
    inv_mass: Vec<f64>,
    sd: Vec<f64>,
    eps: f64,
    max_depth: usize,
    u_turn: bool,
    rng: Pcg,
    depth: usize,
    leapfrogs: u64,
    total_leapfrogs: u64,
    diverged: bool,
    accept_stat: f64,
    draws: u64,
}

impl<'t, T: Target + ?Sized> Nuts<'t, T> {
    /// A sampler at `q0` with step size `eps` and unit mass.
    ///
    /// # Errors
    ///
    /// [`Invalid::StepSize`], [`Invalid::Shape`], [`Invalid::Start`] or
    /// [`Invalid::StartPotential`].
    pub fn new(target: &'t T, q0: &[f64], eps: f64, seed: u64) -> Result<Nuts<'t, T>, Invalid> {
        Self::with_mass(target, q0, eps, None, seed)
    }

    /// A sampler with a diagonal mass, given as its INVERSE `M⁻¹`.
    ///
    /// # Errors
    ///
    /// As [`Nuts::new`], plus [`Invalid::Mass`].
    pub fn with_mass(
        target: &'t T,
        q0: &[f64],
        eps: f64,
        inv_mass: Option<&[f64]>,
        seed: u64,
    ) -> Result<Nuts<'t, T>, Invalid> {
        let s = setup(target, q0, eps, inv_mass)?;
        let n = s.q.len();
        Ok(Nuts {
            target,
            q: s.q,
            grad: vec![0.0; n],
            inv_mass: s.inv_mass,
            sd: s.sd,
            eps,
            max_depth: 10,
            u_turn: true,
            rng: Pcg::new(seed, 0x4E55),
            depth: 0,
            leapfrogs: 0,
            total_leapfrogs: 0,
            diverged: false,
            accept_stat: 0.0,
            draws: 0,
        })
    }

    /// Set the deepest doubling a draw may reach, `2^depth` leapfrog steps.
    ///
    /// # Errors
    ///
    /// [`Invalid::Depth`] past [`MAX_DEPTH_CAP`].
    pub fn set_max_depth(&mut self, depth: usize) -> Result<(), Invalid> {
        if depth > MAX_DEPTH_CAP {
            return Err(Invalid::Depth { depth, cap: MAX_DEPTH_CAP });
        }
        self.max_depth = depth;
        Ok(())
    }

    /// Whether the u-turn criterion stops the doubling. `true` is the sampler.
    ///
    /// With `false` every draw doubles to [`Nuts::max_depth`] unless it diverges. That is still a
    /// valid transition — fixed-depth doubling is reversible on its own — and it is here so that
    /// `the_u_turn_criterion_fires_and_shortens_the_trajectory` can measure a criterion that fires
    /// against one that cannot, rather than asserting that a number is small and hoping.
    pub fn u_turn(&mut self, on: bool) {
        self.u_turn = on;
    }

    /// Replace the step size.
    ///
    /// # Errors
    ///
    /// [`Invalid::StepSize`] for a value that is not finite and positive.
    pub fn set_step_size(&mut self, eps: f64) -> Result<(), Invalid> {
        if !(eps.is_finite() && eps > 0.0) {
            return Err(Invalid::StepSize { eps });
        }
        self.eps = eps;
        Ok(())
    }

    /// Move the chain to `q`, discarding the current state.
    ///
    /// # Errors
    ///
    /// [`Invalid::Shape`], [`Invalid::Start`] or [`Invalid::StartPotential`].
    pub fn set_state(&mut self, q: &[f64]) -> Result<(), Invalid> {
        self.q = setup(self.target, q, self.eps, Some(&self.inv_mass))?.q;
        Ok(())
    }

    /// `true` when the two ends of the trajectory have started approaching each other.
    fn turned(&self, qm: &[f64], pm: &[f64], qp: &[f64], pp: &[f64]) -> bool {
        if !self.u_turn {
            return false;
        }
        let (mut back, mut fwd) = (0.0, 0.0);
        for i in 0..qm.len() {
            let d = qp[i] - qm[i];
            back += d * self.inv_mass[i] * pm[i];
            fwd += d * self.inv_mass[i] * pp[i];
        }
        back < 0.0 || fwd < 0.0
    }

    /// `BuildTree`, recursively.
    fn build(&mut self, q: &[f64], p: &[f64], log_u: f64, v: f64, j: usize, h0: f64) -> Sub {
        if j == 0 {
            let (mut q1, mut p1) = (q.to_vec(), p.to_vec());
            leapfrog_with(
                self.target,
                &mut q1,
                &mut p1,
                &self.inv_mass,
                v * self.eps,
                &mut self.grad,
            );
            self.leapfrogs += 1;
            let h = self.target.potential(&q1) + kinetic(&p1, &self.inv_mass);
            let finite = h.is_finite();
            let n = u64::from(finite && log_u <= -h);
            let ok = finite && log_u < -h + DIVERGENCE_LIMIT;
            if !ok {
                self.diverged = true;
            }
            let alpha = if finite { (h0 - h).exp().min(1.0) } else { 0.0 };
            return Sub {
                q_minus: q1.clone(),
                p_minus: p1.clone(),
                q_plus: q1.clone(),
                p_plus: p1.clone(),
                q_prop: q1,
                n,
                ok,
                alpha,
                n_alpha: 1,
            };
        }
        let mut sub = self.build(q, p, log_u, v, j - 1, h0);
        if !sub.ok {
            return sub;
        }
        let (from_q, from_p) = if v < 0.0 {
            (sub.q_minus.clone(), sub.p_minus.clone())
        } else {
            (sub.q_plus.clone(), sub.p_plus.clone())
        };
        let other = self.build(&from_q, &from_p, log_u, v, j - 1, h0);
        let total = sub.n + other.n;
        if total > 0 && self.rng.f64() * (total as f64) < other.n as f64 {
            sub.q_prop = other.q_prop;
        }
        if v < 0.0 {
            sub.q_minus = other.q_minus;
            sub.p_minus = other.p_minus;
        } else {
            sub.q_plus = other.q_plus;
            sub.p_plus = other.p_plus;
        }
        sub.n = total;
        sub.alpha += other.alpha;
        sub.n_alpha += other.n_alpha;
        sub.ok = other.ok
            && !self.turned(&sub.q_minus, &sub.p_minus, &sub.q_plus, &sub.p_plus);
        sub
    }

    /// One draw.
    pub fn draw(&mut self) -> &[f64] {
        let n = self.q.len();
        let p0: Vec<f64> =
            (0..n).map(|i| self.sd[i] * standard_normal(&mut self.rng)).collect();
        let h0 = self.target.potential(&self.q) + kinetic(&p0, &self.inv_mass);
        // The slice variable, in logs: ln u with u ~ Uniform(0, exp(-H0)).
        let log_u = -h0 + self.rng.f64().max(f64::MIN_POSITIVE).ln();
        let (mut q_minus, mut q_plus) = (self.q.clone(), self.q.clone());
        let (mut p_minus, mut p_plus) = (p0.clone(), p0);
        let mut q_new = self.q.clone();
        let mut n_slice = 1u64;
        let mut ok = h0.is_finite();
        let mut depth = 0usize;
        self.leapfrogs = 0;
        self.diverged = false;
        while ok && depth < self.max_depth {
            let v = if self.rng.f64() < 0.5 { -1.0 } else { 1.0 };
            let (from_q, from_p) = if v < 0.0 {
                (q_minus.clone(), p_minus.clone())
            } else {
                (q_plus.clone(), p_plus.clone())
            };
            let sub = self.build(&from_q, &from_p, log_u, v, depth, h0);
            if v < 0.0 {
                q_minus = sub.q_minus;
                p_minus = sub.p_minus;
            } else {
                q_plus = sub.q_plus;
                p_plus = sub.p_plus;
            }
            if sub.ok && sub.n > 0 && self.rng.f64() * (n_slice as f64) < sub.n as f64 {
                q_new = sub.q_prop;
            }
            self.accept_stat = sub.alpha / sub.n_alpha as f64;
            n_slice += sub.n;
            ok = sub.ok && !self.turned(&q_minus, &p_minus, &q_plus, &p_plus);
            depth += 1;
        }
        self.depth = depth;
        self.total_leapfrogs += self.leapfrogs;
        self.draws += 1;
        self.q = q_new;
        &self.q
    }

    /// Run `iters` draws adapting the step size by dual averaging, then set the smoothed `ε̄` and
    /// return it.
    ///
    /// # Errors
    ///
    /// [`Invalid::TargetAcceptance`] for a `delta` outside `(0, 1)`.
    pub fn warmup(&mut self, iters: usize, delta: f64) -> Result<f64, Invalid> {
        let mut da = DualAveraging::new(self.eps, delta)?;
        for _ in 0..iters {
            self.draw();
            let next = da.update(self.accept_stat);
            self.set_step_size(next)?;
        }
        let bar = da.averaged();
        self.set_step_size(bar)?;
        Ok(bar)
    }

    /// The current state.
    #[must_use]
    pub fn state(&self) -> &[f64] {
        &self.q
    }

    /// Doublings the most recent draw performed. `2^depth` is its trajectory length.
    #[must_use]
    pub fn depth(&self) -> usize {
        self.depth
    }

    /// The deepest doubling a draw may reach.
    #[must_use]
    pub fn max_depth(&self) -> usize {
        self.max_depth
    }

    /// Leapfrog steps the most recent draw spent.
    #[must_use]
    pub fn leapfrogs(&self) -> u64 {
        self.leapfrogs
    }

    /// Leapfrog steps over every draw so far, which is twice the gradient bill.
    #[must_use]
    pub fn total_leapfrogs(&self) -> u64 {
        self.total_leapfrogs
    }

    /// Whether the most recent draw met a state past [`DIVERGENCE_LIMIT`].
    ///
    /// A divergence is not a failure of the sampler, it is the sampler reporting that the
    /// integrator broke. A run with divergences has explored less than its draw count says.
    #[must_use]
    pub fn diverged(&self) -> bool {
        self.diverged
    }

    /// `α / n_α` over the last doubling — Algorithm 6's acceptance statistic.
    #[must_use]
    pub fn accept_stat(&self) -> f64 {
        self.accept_stat
    }

    /// The current step size.
    #[must_use]
    pub fn step_size(&self) -> f64 {
        self.eps
    }

    /// Draws taken.
    #[must_use]
    pub fn draws(&self) -> u64 {
        self.draws
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `A` for a well-conditioned quadratic target, and the same matrix a `Gbm` would hold.
    fn spd2() -> (usize, Vec<f64>, Vec<f64>) {
        (2, vec![2.0, 1.0, 1.0, 2.0], vec![0.5, -1.0])
    }

    /// The 2n x 2n Jacobian of one leapfrog step, EXACTLY, by applying the map to basis vectors.
    ///
    /// Legitimate only because the map is linear, which it is when `b = 0`: the gradient is `A q`,
    /// so the step is a composition of linear shears and its image of a basis vector IS the
    /// corresponding column. No finite differences, no step-size choice, no truncation error.
    fn jacobian(t: &Quadratic, inv_mass: &[f64], eps: f64) -> Vec<Vec<f64>> {
        let n = t.dim();
        let mut cols = Vec::with_capacity(2 * n);
        for c in 0..2 * n {
            let (mut q, mut p) = (vec![0.0; n], vec![0.0; n]);
            if c < n {
                q[c] = 1.0;
            } else {
                p[c - n] = 1.0;
            }
            leapfrog(t, &mut q, &mut p, inv_mass, eps);
            let mut col = q;
            col.extend_from_slice(&p);
            cols.push(col);
        }
        cols
    }

    /// Determinant of a small matrix given as COLUMNS, by Gaussian elimination with partial
    /// pivoting.
    fn det(cols: &[Vec<f64>]) -> f64 {
        let n = cols.len();
        let mut m = vec![vec![0.0; n]; n];
        for (c, col) in cols.iter().enumerate() {
            for r in 0..n {
                m[r][c] = col[r];
            }
        }
        let mut d = 1.0;
        for k in 0..n {
            let mut piv = k;
            for r in (k + 1)..n {
                if m[r][k].abs() > m[piv][k].abs() {
                    piv = r;
                }
            }
            if m[piv][k] == 0.0 {
                return 0.0;
            }
            if piv != k {
                m.swap(piv, k);
                d = -d;
            }
            d *= m[k][k];
            for r in (k + 1)..n {
                let f = m[r][k] / m[k][k];
                for c in k..n {
                    m[r][c] -= f * m[k][c];
                }
            }
        }
        d
    }

    /// ORACLE: the 2x2 leapfrog map on a dyadic fixture, where every operation is EXACT.
    ///
    /// `a = 1`, `ε = 1/2`: the map is `[[7/8, 1/2], [−15/32, 7/8]]` and its determinant is
    /// `49/64 + 15/64 = 1` with no rounding anywhere, so the assertion is `assert_eq!` and not a
    /// tolerance. Volume preservation is one of the two properties that make `min(1, exp(−ΔH))` a
    /// valid acceptance probability; the other is reversibility, tested beside this.
    ///
    /// The 6-dimensional case beside it is the same statement where the determinant has to be
    /// computed rather than read, so it carries the elimination's own rounding and nothing more.
    ///
    /// The two exact COLUMNS are asserted as well as the determinant, and they are the half of this
    /// test that bites: dropping the half from both kicks leaves a map whose determinant is still
    /// exactly 1.0, measured. A determinant is a one-number summary of a map, and one number does
    /// not identify a map.
    #[test]
    fn leapfrog_is_volume_preserving_oracle_the_exact_determinant_of_its_linear_map() {
        let t = Quadratic::new(1, vec![1.0], vec![0.0]).unwrap();
        let j = jacobian(&t, &[1.0], 0.5);
        assert_eq!(j[0], vec![0.875, -0.46875], "image of (q, p) = (1, 0)");
        assert_eq!(j[1], vec![0.5, 0.875], "image of (q, p) = (0, 1)");
        assert_eq!(det(&j), 1.0, "a dyadic fixture leaves nothing to round");

        // Three coupled coordinates and a non-unit diagonal mass, so the shears actually mix.
        let a = vec![2.0, -1.0, 0.0, -1.0, 2.0, -1.0, 0.0, -1.0, 2.0];
        let t3 = Quadratic::new(3, a, vec![0.0; 3]).unwrap();
        for eps in [0.125f64, 0.25, 0.5, 0.75] {
            let d = det(&jacobian(&t3, &[1.0, 0.5, 2.0], eps));
            assert!((d - 1.0).abs() < 1e-14, "eps {eps}: det = {d}, off by {:e}", d - 1.0);
        }
    }

    /// ORACLE: reversibility, as a BITWISE round trip on a fixture whose arithmetic is exact.
    ///
    /// Integrate `L` steps, negate the momentum, integrate `L` more, negate again: in exact
    /// arithmetic that is the identity, and on dyadic inputs (`a = 1`, `ε = 1/2`, `q₀ = 1`,
    /// `p₀ = 1/2`) the floating-point arithmetic IS exact, so the round trip is bit-for-bit and the
    /// assertion is `assert_eq!`.
    ///
    /// The second half is the same claim where the arithmetic is not exact — a coupled quadratic
    /// and a logistic, 40 steps out and 40 back — and there the round trip is NOT bit-exact and
    /// cannot be: `fl(fl(a + d) − d)` is not `a` in general, `1 + 2⁻⁵² ± 2⁻⁵³` being the smallest
    /// counterexample, both roundings going to even. The measured worst error over those six cases
    /// is **8.9e-16**, four units in the last place at a state of size 2, and the tolerance is set
    /// an order above it rather than at a round number pulled from nowhere.
    ///
    /// Reversibility is what a wrong HALF-step breaks. Volume preservation does not: `p −= ε g; q
    /// += ε p; p −= ε g` is still a symmetric composition of shears. See the module docs.
    #[test]
    fn leapfrog_is_reversible_oracle_a_bitwise_round_trip_on_a_dyadic_fixture() {
        let t = Quadratic::new(1, vec![1.0], vec![0.0]).unwrap();
        let (q0, p0) = (vec![1.0f64], vec![0.5f64]);
        let (mut q, mut p) = (q0.clone(), p0.clone());
        for _ in 0..6 {
            leapfrog(&t, &mut q, &mut p, &[1.0], 0.5);
        }
        assert_ne!(q, q0, "the fixture must actually move, or the round trip tests nothing");
        p[0] = -p[0];
        for _ in 0..6 {
            leapfrog(&t, &mut q, &mut p, &[1.0], 0.5);
        }
        p[0] = -p[0];
        assert_eq!(q, q0, "reversed trajectory did not land on q0 exactly");
        assert_eq!(p, p0, "reversed trajectory did not land on p0 exactly");

        // A negative step size must be the inverse map, which is how NUTS grows leftwards.
        let (mut q, mut p) = (vec![0.75f64], vec![-0.25f64]);
        leapfrog(&t, &mut q, &mut p, &[1.0], 0.5);
        leapfrog(&t, &mut q, &mut p, &[1.0], -0.5);
        assert_eq!(q, vec![0.75], "forward then backward is not the identity");
        assert_eq!(p, vec![-0.25]);

        let (n, a, b) = spd2();
        let quad = Quadratic::new(n, a, b).unwrap();
        let logi = Logistic::new(3);
        let inv2 = [1.0, 0.4];
        let inv3 = [1.0, 1.0, 1.0];
        let mut worst = 0.0f64;
        for eps in [0.05f64, 0.2, 0.4] {
            let (mut q, mut p) = (vec![0.3, -0.7], vec![1.1, 0.2]);
            let (q0, p0) = (q.clone(), p.clone());
            for _ in 0..40 {
                leapfrog(&quad, &mut q, &mut p, &inv2, eps);
            }
            for _ in 0..40 {
                leapfrog(&quad, &mut q, &mut p, &inv2, -eps);
            }
            for i in 0..2 {
                worst = worst.max((q[i] - q0[i]).abs()).max((p[i] - p0[i]).abs());
            }

            let (mut q, mut p) = (vec![1.0, -2.0, 0.5], vec![0.2, 1.3, -0.4]);
            let (q0, p0) = (q.clone(), p.clone());
            for _ in 0..40 {
                leapfrog(&logi, &mut q, &mut p, &inv3, eps);
            }
            for _ in 0..40 {
                leapfrog(&logi, &mut q, &mut p, &inv3, -eps);
            }
            for i in 0..3 {
                worst = worst.max((q[i] - q0[i]).abs()).max((p[i] - p0[i]).abs());
            }
        }
        assert!(worst < 1e-14, "worst round-trip error {worst:e} over 80 steps");
    }

    /// ORACLE: the closed-form invariant of the leapfrog map on a quadratic.
    ///
    /// A linear map with equal diagonal entries has exactly one invariant quadratic form up to
    /// scale, and for leapfrog on `U = ½ a q²` it is `H̃ = ½p² + ½ a(1 − ε²a/4) q²`. That is an
    /// EXACT conservation law for the discrete map — not an `O(ε²)` statement — so it holds at
    /// every step size, including ones far too large to sample with.
    ///
    /// This is the test that a wrong half-step cannot survive: kicking by `ε` instead of `ε/2`
    /// leaves a map that is still volume preserving and still reversible, but conserves
    /// `a(1 − ε²a/2)` instead, which is a different number at every `ε`.
    #[test]
    fn the_leapfrog_map_conserves_the_closed_form_shadow_hamiltonian() {
        for &a in &[0.25f64, 1.0, 3.0] {
            let t = Quadratic::new(1, vec![a], vec![0.0]).unwrap();
            for &eps in &[0.1f64, 0.5, 1.0, 1.5] {
                if eps * a.sqrt() >= 2.0 {
                    continue; // past the stability limit the orbit is unbounded, not periodic
                }
                let a_eff = a * (1.0 - eps * eps * a / 4.0);
                let (mut q, mut p) = (vec![1.3f64], vec![-0.6f64]);
                let h_tilde = |q: f64, p: f64| 0.5 * p * p + 0.5 * a_eff * q * q;
                let h0 = h_tilde(q[0], p[0]);
                let mut worst = 0.0f64;
                for _ in 0..400 {
                    leapfrog(&t, &mut q, &mut p, &[1.0], eps);
                    worst = worst.max((h_tilde(q[0], p[0]) - h0).abs() / h0);
                }
                assert!(
                    worst < 1e-12,
                    "a {a} eps {eps}: shadow energy drifted {worst:e} in 400 steps"
                );
            }
        }
    }

    /// ORACLE: `ΔH = (ε²a²/8)(q_L² − q_0²)`, a closed form the sampler never sees.
    ///
    /// It follows from the shadow Hamiltonian above: `H − H̃ = (ε²a²/8) q²` everywhere, and `H̃` is
    /// conserved, so the energy error of a whole trajectory is a function of its endpoints alone,
    /// for ANY number of steps. The Metropolis correction eats this number, and this is the check
    /// that the number it eats is the right one.
    #[test]
    fn the_energy_error_matches_its_closed_form_on_a_quadratic() {
        let mut rng = Pcg::new(0xD17A, 3);
        for &a in &[0.5f64, 1.0, 2.0] {
            let t = Quadratic::new(1, vec![a], vec![0.0]).unwrap();
            for &eps in &[0.2f64, 0.7, 1.1] {
                for l in [1usize, 3, 17] {
                    let q0 = (rng.f64() - 0.5) * 4.0;
                    let p0 = (rng.f64() - 0.5) * 4.0;
                    let (mut q, mut p) = (vec![q0], vec![p0]);
                    let h0 = hamiltonian(&t, &q, &p, &[1.0]);
                    for _ in 0..l {
                        leapfrog(&t, &mut q, &mut p, &[1.0], eps);
                    }
                    let dh = hamiltonian(&t, &q, &p, &[1.0]) - h0;
                    let want = eps * eps * a * a / 8.0 * (q[0] * q[0] - q0 * q0);
                    assert!(
                        (dh - want).abs() < 1e-12 * (1.0 + want.abs()),
                        "a {a} eps {eps} L {l}: dH {dh} vs closed form {want}"
                    );
                }
            }
        }
    }

    /// Mean of a slice and the standard error of that mean ACROSS independent chains.
    fn mean_and_se(x: &[f64]) -> (f64, f64) {
        let r = x.len() as f64;
        let m = x.iter().sum::<f64>() / r;
        let v = x.iter().map(|&v| (v - m) * (v - m)).sum::<f64>() / (r - 1.0);
        (m, (v / r).sqrt())
    }

    /// ORACLE: `continuous::Gbm::exact_gaussian`, which is itself checked against Gaussian
    /// elimination and closed-form `ln Z` in its own module.
    ///
    /// The mean is `A⁻¹b` and the covariance is `A⁻¹`; the sampler must find both at every step
    /// size, since the Metropolis correction is exact for all of them. Error bars come from twelve
    /// INDEPENDENT chains, so autocorrelation is inside the spread rather than assumed away — and
    /// the spread itself is asserted to be small, because a test whose error bars are wide enough
    /// passes for anything.
    #[test]
    fn hmc_recovers_the_gaussian_law_oracle_continuous_exact_gaussian() {
        let (n, a, b) = spd2();
        let gbm = crate::continuous::Gbm::gaussian(n, a.clone(), b.clone());
        let (want_mean, want_cov, _) = gbm.exact_gaussian(1.0).unwrap();
        let t = Quadratic::new(n, a, b).unwrap();

        // The two potentials must be the same function, or the oracle is for another model.
        for q in [[0.0, 0.0], [1.5, -0.5], [-2.0, 3.0]] {
            let mine = t.potential(&q);
            let theirs = gbm.energy(&q, &[]);
            assert!((mine - theirs).abs() < 1e-12, "U {mine} vs Gbm::energy {theirs}");
        }

        const CHAINS: usize = 12;
        const DRAWS: usize = 4000;
        for &eps in &[0.15f64, 0.35, 0.6] {
            let mut m0 = [0.0; CHAINS];
            let mut m1 = [0.0; CHAINS];
            let mut c00 = [0.0; CHAINS];
            let mut c01 = [0.0; CHAINS];
            let mut c11 = [0.0; CHAINS];
            for (c, seed) in (0..CHAINS).zip(1000u64..) {
                let mut s = Hmc::new(&t, &[0.0, 0.0], eps, 8, seed).unwrap();
                for _ in 0..500 {
                    s.draw();
                }
                let (mut s0, mut s1, mut s00, mut s01, mut s11) = (0.0, 0.0, 0.0, 0.0, 0.0);
                for _ in 0..DRAWS {
                    let q = s.draw();
                    s0 += q[0];
                    s1 += q[1];
                    s00 += q[0] * q[0];
                    s01 += q[0] * q[1];
                    s11 += q[1] * q[1];
                }
                let d = DRAWS as f64;
                m0[c] = s0 / d;
                m1[c] = s1 / d;
                c00[c] = s00 / d - m0[c] * m0[c];
                c01[c] = s01 / d - m0[c] * m1[c];
                c11[c] = s11 / d - m1[c] * m1[c];
                assert!(
                    s.acceptance().unwrap() > 0.6,
                    "eps {eps}: acceptance {:?} is too low to be measuring anything",
                    s.acceptance()
                );
            }
            let checks: [(&str, &[f64], f64); 5] = [
                ("mean0", &m0, want_mean[0]),
                ("mean1", &m1, want_mean[1]),
                ("cov00", &c00, want_cov[0]),
                ("cov01", &c01, want_cov[1]),
                ("cov11", &c11, want_cov[3]),
            ];
            for (name, samples, exact) in checks {
                let (est, se) = mean_and_se(samples);
                assert!(se < 0.02, "eps {eps} {name}: error bar {se:.4} is too wide to bite");
                assert!(
                    (est - exact).abs() < 4.0 * se,
                    "eps {eps} {name}: {est:.5} vs exact {exact:.5}, {:.2} error bars away",
                    (est - exact).abs() / se
                );
            }
        }
    }

    /// ASYMMETRIC ORACLE: the correction is load-bearing, and the number it is worth is a closed
    /// form.
    ///
    /// Uncorrected leapfrog with a full momentum refresh has `exp(−H̃)` as its EXACT stationary
    /// distribution, because the map preserves volume and conserves `H̃`, and the refresh draws `p`
    /// from the Gaussian `H̃` already factorises into. So at `a = 1, ε = 1` it samples `q` with
    /// variance `1/(1 − ε²/4) = 4/3`, not 1.
    ///
    /// Both arms are asserted. A test that only checked the corrected arm would pass for a sampler
    /// that never computes `ΔH` at all — because at a small step size the two arms agree, which is
    /// exactly why this one runs at `ε = 1`.
    ///
    /// **`L` MAY NOT BE A MULTIPLE OF THREE HERE, AND THE FIRST VERSION OF THIS TEST USED 3.** At
    /// `a = 1, ε = 1` the leapfrog map is a rotation through `arccos(1 − ε²a/2) = π/3` in the shadow
    /// metric, so three steps are exactly half a period and send `(q, p)` to `(−q, −p)`. Started
    /// from `q = 0` the chain then never leaves it: measured variance 0.00000, `ΔH` exactly 0,
    /// acceptance exactly 1.000, in both arms, for 20,000 draws. Every assertion below except the
    /// error-bar guard passed on that chain, which is why the guard is there.
    #[test]
    fn the_metropolis_correction_is_what_makes_the_variance_right() {
        let a = 1.0f64;
        let eps = 1.0f64;
        let t = Quadratic::new(1, vec![a], vec![0.0]).unwrap();
        let want_uncorrected = 1.0 / (a * (1.0 - eps * eps * a / 4.0));
        assert!((want_uncorrected - 4.0 / 3.0).abs() < 1e-15, "the fixture is the 4/3 case");

        const CHAINS: usize = 12;
        const DRAWS: usize = 20000;
        let run = |correct: bool| -> (f64, f64) {
            let mut var = [0.0f64; CHAINS];
            for (c, seed) in (0..CHAINS).zip(7000u64..) {
                let mut s = Hmc::new(&t, &[0.0], eps, 4, seed).unwrap();
                s.correct(correct);
                for _ in 0..1000 {
                    s.draw();
                }
                let (mut s1, mut s2) = (0.0, 0.0);
                for _ in 0..DRAWS {
                    let q = s.draw()[0];
                    s1 += q;
                    s2 += q * q;
                }
                let m = s1 / DRAWS as f64;
                var[c] = s2 / DRAWS as f64 - m * m;
            }
            mean_and_se(&var)
        };

        let (on, se_on) = run(true);
        let (off, se_off) = run(false);
        assert!(
            se_on > 0.0 && se_off > 0.0,
            "twelve chains agreed to the last bit, so the fixture is a fixed point and not a chain"
        );
        assert!(se_on < 0.01 && se_off < 0.01, "error bars {se_on:.4} / {se_off:.4} too wide");
        assert!(
            (on - 1.0 / a).abs() < 4.0 * se_on,
            "corrected variance {on:.4} vs exact 1.0 ({:.1} error bars)",
            (on - 1.0 / a).abs() / se_on
        );
        assert!(
            (off - want_uncorrected).abs() < 4.0 * se_off,
            "uncorrected variance {off:.4} vs shadow closed form {want_uncorrected:.4} ({:.1} \
             error bars)",
            (off - want_uncorrected).abs() / se_off
        );
        // And the thing that FAILS: the uncorrected arm is not the target, by a mile.
        assert!(
            (off - 1.0 / a).abs() > 20.0 * se_off,
            "uncorrected variance {off:.4} is indistinguishable from 1.0, so the contrast is \
             unearned"
        );
    }

    /// Exact deciles of the standard logistic, from its closed-form quantile function.
    fn logistic_deciles() -> Vec<f64> {
        (1..10).map(|k| Logistic::quantile(f64::from(k) / 10.0)).collect()
    }

    /// Which decile a value falls in.
    fn decile(edges: &[f64], q: f64) -> usize {
        edges.iter().take_while(|&&e| q > e).count()
    }

    /// ORACLE: the standard logistic's closed-form CDF, `F(q) = 1/(1 + e^{−q})`.
    ///
    /// The state space is cut at the EXACT deciles, so each of the ten cells has probability
    /// exactly 1/10 and the stationary distribution of the discretised chain is known with no
    /// quadrature, no special function and no reference implementation.
    ///
    /// Two statements, and they fail differently:
    ///
    /// * **invariance** — start 120,000 chains from EXACT draws (`F⁻¹(u)`), take ONE NUTS
    ///   transition each, and the cell frequencies must still be 1/10. The starts are independent,
    ///   so the error bar is the binomial one and nothing has to be assumed about mixing.
    /// * **the long run** — sixteen chains of 25,000 draws from a cold start must land on the same
    ///   ten numbers. This is the arm that catches a bias too small for one step to show.
    #[test]
    fn nuts_leaves_the_logistic_invariant_oracle_its_closed_form_cdf() {
        let t = Logistic::new(1);
        let edges = logistic_deciles();
        // The oracle itself: quantile and CDF must be inverses, and the cells equiprobable.
        for (k, &e) in edges.iter().enumerate() {
            let want = (k + 1) as f64 / 10.0;
            assert!((Logistic::cdf(e) - want).abs() < 1e-15, "decile {k}: F = {}", Logistic::cdf(e));
        }

        const ONE_STEP: usize = 120_000;
        let mut cells = [0u64; 10];
        let mut rng = Pcg::new(0x10617, 21);
        let mut s = Nuts::new(&t, &[0.0], 0.7, 4242).unwrap();
        s.set_max_depth(6).unwrap();
        for _ in 0..ONE_STEP {
            let start = Logistic::quantile(rng.f64().clamp(1e-12, 1.0 - 1e-12));
            s.set_state(&[start]).unwrap();
            let q = s.draw()[0];
            cells[decile(&edges, q)] += 1;
        }
        let se = (0.1f64 * 0.9 / ONE_STEP as f64).sqrt();
        for (k, &c) in cells.iter().enumerate() {
            let f = c as f64 / ONE_STEP as f64;
            assert!(
                (f - 0.1).abs() < 4.0 * se,
                "one step from stationarity moved cell {k} to {f:.5} ({:.1} binomial error bars)",
                (f - 0.1).abs() / se
            );
        }

        const CHAINS: usize = 16;
        const DRAWS: usize = 25_000;
        let mut freq = [[0.0f64; CHAINS]; 10];
        for (c, seed) in (0..CHAINS).zip(300u64..) {
            let mut s = Nuts::new(&t, &[3.0], 0.7, seed).unwrap();
            s.set_max_depth(6).unwrap();
            let mut hit = [0u64; 10];
            for _ in 0..1000 {
                s.draw();
            }
            for _ in 0..DRAWS {
                let q = s.draw()[0];
                hit[decile(&edges, q)] += 1;
            }
            for k in 0..10 {
                freq[k][c] = hit[k] as f64 / DRAWS as f64;
            }
        }
        for (k, f) in freq.iter().enumerate() {
            let (est, se) = mean_and_se(f);
            assert!(se < 0.004, "cell {k}: chain-to-chain error bar {se:.5} is too wide to bite");
            assert!(
                (est - 0.1).abs() < 4.0 * se,
                "cell {k}: long-run frequency {est:.5} vs exact 0.1 ({:.1} error bars)",
                (est - 0.1).abs() / se
            );
        }
    }

    /// ASYMMETRIC: the u-turn criterion fires, and the same sampler without it runs longer.
    ///
    /// On `U = ½q²` the orbit has period `2π`, so a trajectory turns back on itself after about
    /// `π/ε` steps. At `ε = 0.25` that is ~13 steps, depth 4 — well short of the depth-8 cap, so a
    /// bounded trajectory here is the CRITERION stopping it and not the cap.
    ///
    /// The second arm is what makes this a test rather than an observation: with the criterion off
    /// every draw must reach depth 8, which is 256 leapfrog steps against the handful the criterion
    /// allows. An implementation whose criterion never fires passes the first arm and fails this
    /// one; one that always fires fails the first.
    #[test]
    fn the_u_turn_criterion_fires_and_shortens_the_trajectory() {
        let t = Quadratic::new(1, vec![1.0], vec![0.0]).unwrap();
        const DRAWS: usize = 400;

        let mut on = Nuts::new(&t, &[0.0], 0.25, 99).unwrap();
        on.set_max_depth(8).unwrap();
        let mut deepest = 0usize;
        for _ in 0..DRAWS {
            on.draw();
            deepest = deepest.max(on.depth());
            assert!(!on.diverged(), "a stable step size must not diverge");
        }
        let steps_on = on.total_leapfrogs() as f64 / DRAWS as f64;
        assert!(deepest < 8, "the deepest draw reached the CAP ({deepest}), so nothing was proved");
        assert!(deepest >= 2, "the criterion fired instantly ({deepest}), which is its own bug");

        let mut off = Nuts::new(&t, &[0.0], 0.25, 99).unwrap();
        off.set_max_depth(8).unwrap();
        off.u_turn(false);
        for _ in 0..DRAWS {
            off.draw();
            assert_eq!(off.depth(), 8, "without the criterion every draw must reach the cap");
            assert_eq!(off.leapfrogs(), 255, "depth 8 is 2^8 - 1 leapfrog steps");
        }
        let steps_off = off.total_leapfrogs() as f64 / DRAWS as f64;
        assert!(
            steps_off > 8.0 * steps_on,
            "criterion on {steps_on:.1} steps/draw, off {steps_off:.1}: not a contrast"
        );
    }

    /// The divergence flag is a report, and it must fire on a step size that deserves it and stay
    /// silent on one that does not.
    #[test]
    fn a_step_size_past_the_stability_limit_is_reported_as_a_divergence() {
        let t = Quadratic::new(1, vec![1.0], vec![0.0]).unwrap();
        // Stability needs eps < 2/sqrt(a) = 2, and MERELY unstable is not enough: at eps = 4 the
        // measured divergence count is 0 of 200, because the u-turn criterion fires after the very
        // first step of an orbit that oscillates, and one step has not gained a thousand nats yet.
        // The flag is about ENERGY, not about stability, and the fixture has to supply energy.
        let mut wild = Nuts::new(&t, &[2.0], 20.0, 5).unwrap();
        wild.set_max_depth(6).unwrap();
        let mut hits = 0;
        for _ in 0..200 {
            wild.draw();
            hits += usize::from(wild.diverged());
        }
        assert_eq!(hits, 200, "draws at eps = 20 from q = 2 that were reported divergent");

        let mut calm = Nuts::new(&t, &[0.0], 0.3, 5).unwrap();
        calm.set_max_depth(6).unwrap();
        for _ in 0..200 {
            calm.draw();
            assert!(!calm.diverged(), "eps = 0.3 must never divergence-flag on a unit Gaussian");
        }
    }

    /// `E[min(1, exp(−ΔH))]` for one leapfrog step on `U = ½q²`, by deterministic midpoint
    /// quadrature of the CLOSED FORM `ΔH = (ε²/8)(q'² − q²)`, `q' = (1 − ε²/2) q + ε p`.
    ///
    /// No sampler, no RNG, no chain — just a two-dimensional Gaussian integral over the plane,
    /// truncated at 7 sigma where the weight is 1e-11. This is the oracle the adaptation is judged
    /// against.
    fn quadrature_accept_rate(eps: f64) -> f64 {
        const M: usize = 1400;
        const LIM: f64 = 7.0;
        let h = 2.0 * LIM / M as f64;
        let c = 1.0 - 0.5 * eps * eps;
        let lambda = eps * eps / 8.0;
        let phi: Vec<(f64, f64)> = (0..M)
            .map(|k| {
                let x = -LIM + (k as f64 + 0.5) * h;
                (x, (-0.5 * x * x).exp() / (core::f64::consts::TAU).sqrt())
            })
            .collect();
        let mut total = 0.0;
        for &(q, wq) in &phi {
            let mut inner = 0.0;
            for &(p, wp) in &phi {
                let qn = c * q + eps * p;
                let dh = lambda * (qn * qn - q * q);
                inner += wp * (-dh).exp().min(1.0);
            }
            total += wq * inner;
        }
        total * h * h
    }

    /// ORACLE: the step size that the closed-form acceptance integral says hits the target, found
    /// by bisection on a quadrature this module's samplers never touch.
    ///
    /// Dual averaging is then run blind — 3000 warmup draws of one-step HMC on the unit Gaussian,
    /// target 0.75 — and its smoothed `ε̄` has to land on that step size. What is being checked is
    /// not that the adaptation converges to SOMETHING but that it converges to the RIGHT thing,
    /// which needs a source for the right thing that is not the adaptation.
    #[test]
    fn dual_averaging_finds_the_step_size_the_acceptance_integral_names() {
        let delta = 0.75f64;
        // The quadrature must be a decreasing function that brackets the target, or bisection is
        // solving nothing.
        let (lo_eps, hi_eps) = (0.05f64, 1.9f64);
        let (lo_a, hi_a) = (quadrature_accept_rate(lo_eps), quadrature_accept_rate(hi_eps));
        assert!(lo_a > delta && hi_a < delta, "acceptance {lo_a:.4}..{hi_a:.4} does not bracket");
        let (mut lo, mut hi) = (lo_eps, hi_eps);
        for _ in 0..18 {
            let mid = 0.5 * (lo + hi);
            if quadrature_accept_rate(mid) > delta {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let want = 0.5 * (lo + hi);

        let t = Quadratic::new(1, vec![1.0], vec![0.0]).unwrap();
        let mut s = Hmc::new(&t, &[0.0], 0.1, 1, 2026).unwrap();
        let got = s.warmup(3000, delta).unwrap();
        assert!(
            (got - want).abs() / want < 0.08,
            "dual averaging settled on {got:.4}; the acceptance integral says {want:.4}"
        );
        assert!((s.step_size() - got).abs() < 1e-15, "warmup must leave the chain at eps-bar");

        // And the adapted chain must actually accept at the rate it was asked for.
        let mut check = Hmc::new(&t, &[0.0], got, 1, 7).unwrap();
        for _ in 0..40_000 {
            check.draw();
        }
        let rate = check.acceptance().unwrap();
        assert!((rate - delta).abs() < 0.02, "adapted chain accepts at {rate:.4}, asked {delta}");
    }

    /// The step-size search brackets, rather than returning whatever it started at.
    #[test]
    fn the_reasonable_step_size_search_lands_where_one_step_is_roughly_even_money() {
        for &a in &[0.01f64, 1.0, 100.0] {
            let t = Quadratic::new(1, vec![a], vec![0.0]).unwrap();
            let eps = reasonable_step_size(&t, &[0.0], &[1.0], 4).unwrap();
            let rate = quadrature_accept_rate(eps * a.sqrt());
            assert!(
                (0.2..0.95).contains(&rate),
                "a {a}: search returned eps {eps:e}, whose acceptance is {rate:.3}"
            );
        }
    }

    /// Every refusal is typed and names the value it saw. A default here would be a chain nobody
    /// can reproduce from what they wrote down.
    #[test]
    fn bad_arguments_are_typed_errors_naming_what_they_saw() {
        let t = Quadratic::new(1, vec![1.0], vec![0.0]).unwrap();
        assert_eq!(
            Hmc::new(&t, &[0.0], 0.0, 4, 1).err(),
            Some(Invalid::StepSize { eps: 0.0 })
        );
        assert_eq!(
            Hmc::new(&t, &[0.0], f64::NAN, 4, 1).err().map(|e| matches!(e, Invalid::StepSize { .. })),
            Some(true)
        );
        assert_eq!(
            Nuts::new(&t, &[0.0, 1.0], 0.5, 1).err(),
            Some(Invalid::Shape { what: "q0", given: 2, want: 1 })
        );
        assert_eq!(
            Nuts::new(&t, &[f64::INFINITY], 0.5, 1).err(),
            Some(Invalid::Start { index: 0, value: f64::INFINITY })
        );
        assert_eq!(
            Hmc::with_mass(&t, &[0.0], 0.5, 4, Some(&[0.0]), 1).err(),
            Some(Invalid::Mass { index: 0, inv_mass: 0.0 })
        );
        assert_eq!(
            Hmc::with_mass(&t, &[0.0], 0.5, 4, Some(&[1.0, 1.0]), 1).err(),
            Some(Invalid::Shape { what: "inv_mass", given: 2, want: 1 })
        );
        assert_eq!(
            DualAveraging::new(0.5, 1.0).err(),
            Some(Invalid::TargetAcceptance { delta: 1.0 })
        );
        let mut s = Nuts::new(&t, &[0.0], 0.5, 1).unwrap();
        assert_eq!(
            s.set_max_depth(MAX_DEPTH_CAP + 1).err(),
            Some(Invalid::Depth { depth: MAX_DEPTH_CAP + 1, cap: MAX_DEPTH_CAP })
        );
        assert_eq!(
            Quadratic::new(2, vec![1.0, 2.0, 2.0, 1.0], vec![0.0, 0.0]).err(),
            Some(Invalid::NotPositiveDefinite)
        );
        assert_eq!(
            Quadratic::new(2, vec![1.0, 0.0], vec![0.0, 0.0]).err(),
            Some(Invalid::Shape { what: "a", given: 2, want: 4 })
        );
        // Every one of them prints the value, not a category.
        let msg = Invalid::Mass { index: 3, inv_mass: -2.5 }.to_string();
        assert!(msg.contains("-2.5") && msg.contains('3'), "{msg}");
    }

    /// Same seed, same draws — for both samplers, and a different seed must actually differ.
    #[test]
    fn the_samplers_are_deterministic_by_seed() {
        let (n, a, b) = spd2();
        let t = Quadratic::new(n, a, b).unwrap();
        let run_hmc = |seed: u64| {
            let mut s = Hmc::new(&t, &[0.0, 0.0], 0.4, 5, seed).unwrap();
            (0..200).map(|_| s.draw().to_vec()).collect::<Vec<_>>()
        };
        assert_eq!(run_hmc(5), run_hmc(5));
        assert_ne!(run_hmc(5), run_hmc(6));

        let run_nuts = |seed: u64| {
            let mut s = Nuts::new(&t, &[0.0, 0.0], 0.4, seed).unwrap();
            (0..200).map(|_| s.draw().to_vec()).collect::<Vec<_>>()
        };
        assert_eq!(run_nuts(5), run_nuts(5));
        assert_ne!(run_nuts(5), run_nuts(6));
    }

    /// The one bound this module makes: it is monotone, zero where there is no arithmetic, and far
    /// below the threshold it must never be confused with.
    #[test]
    fn the_energy_error_guard_is_a_bound_and_not_a_threshold() {
        assert_eq!(delta_h_guard(4, 0.0), 0.0);
        assert!(delta_h_guard(4, 1.0) < delta_h_guard(40, 1.0));
        assert!(delta_h_guard(4, 1.0) < delta_h_guard(4, 10.0));
        // A divergence is a thousand nats. The guard on a hundred-dimensional model at an energy
        // scale of 1e6 is 4.5e-8, ten orders below it, so rounding can never be read as one.
        assert!(delta_h_guard(100, 1e6) < DIVERGENCE_LIMIT * 1e-6, "{}", delta_h_guard(100, 1e6));
    }

    /// A diagonal mass rescales the geometry and must leave the ANSWER alone: sampling
    /// `N(0, A⁻¹)` with `M = A` is preconditioned HMC, and it has to find the same covariance.
    ///
    /// `L = 5` and not 8. With `M = A` every mode has unit frequency, so the leapfrog rotation is
    /// `arccos(1 − ε²/2) = 0.4027` rad and `L = 8` puts the chain at `3.22` rad — within 2% of half
    /// a period, an autocorrelation of `−0.996`, and a measured `Var(q₀)` of 0.10426 against an
    /// exact 1/9 = 0.11111 after 60,000 draws. Nothing about the sampler was wrong; the fixture was
    /// resonant, and a single chain has no way to say so. Chain-to-chain error bars do.
    #[test]
    fn a_diagonal_mass_changes_the_cost_and_not_the_distribution() {
        let a = vec![9.0, 0.0, 0.0, 0.25];
        let t = Quadratic::new(2, a.clone(), vec![0.0, 0.0]).unwrap();
        let gbm = crate::continuous::Gbm::gaussian(2, a, vec![0.0, 0.0]);
        let (_, want_cov, _) = gbm.exact_gaussian(1.0).unwrap();

        const CHAINS: usize = 8;
        const DRAWS: usize = 20_000;
        let mut v0 = [0.0f64; CHAINS];
        let mut v1 = [0.0f64; CHAINS];
        for (c, seed) in (0..CHAINS).zip(77u64..) {
            // M = A, so M^-1 = diag(1/9, 4).
            let mut s =
                Hmc::with_mass(&t, &[0.0, 0.0], 0.4, 5, Some(&[1.0 / 9.0, 4.0]), seed).unwrap();
            for _ in 0..2000 {
                s.draw();
            }
            let (mut s00, mut s11) = (0.0, 0.0);
            for _ in 0..DRAWS {
                let q = s.draw();
                s00 += q[0] * q[0];
                s11 += q[1] * q[1];
            }
            v0[c] = s00 / DRAWS as f64;
            v1[c] = s11 / DRAWS as f64;
            assert!(s.acceptance().unwrap() > 0.9, "preconditioning should accept freely");
        }
        for (name, samples, exact) in
            [("Var(q0)", &v0, want_cov[0]), ("Var(q1)", &v1, want_cov[3])]
        {
            let (est, se) = mean_and_se(samples);
            assert!(se / exact < 0.01, "{name}: error bar {se:.5} is too wide to bite");
            assert!(
                (est - exact).abs() < 4.0 * se,
                "{name}: {est:.5} vs exact {exact:.5} ({:.2} error bars)",
                (est - exact).abs() / se
            );
        }
    }
}
