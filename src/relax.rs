//! Relaxation gradient estimators for discrete variables — the Concrete family, and the unbiased
//! estimator that buys its bias back.
//!
//! A discrete objective `L(theta) = E_{s ~ q_theta}[f(s)]` over spins has no pathwise derivative:
//! `s` is a step function of its noise, so there is nothing to differentiate through. The field
//! built two answers on top of that fact and this module holds both, together with the oracle that
//! tells them apart.
//!
//! * **The relaxation** (Jang, Gu & Poole, *Categorical Reparameterization with Gumbel-Softmax*,
//!   ICLR 2017, arXiv:1611.01144; Maddison, Mnih & Teh, *The Concrete Distribution*, ICLR 2017,
//!   arXiv:1611.00712 — the same object, found twice, in the same week). Replace the argmax by a
//!   softmax at temperature `tau`, and the step function becomes smooth. [`gumbel_softmax_grad`].
//! * **Straight-through** (Hinton, 2012 lecture 15b; Bengio, Léonard & Courville,
//!   arXiv:1308.3432, 2013). Keep the hard sample in the forward pass and pretend, in the backward
//!   pass, that it was the relaxation. [`straight_through_grad`].
//! * **REBAR** (Tucker, Mnih, Maddison, Lawson & Sohl-Dickstein, `NeurIPS` 2017, arXiv:1703.07370),
//!   the estimator that uses the relaxation as a CONTROL VARIATE for the score function rather
//!   than as a substitute for the objective, and is therefore unbiased at every temperature.
//!   [`rebar_grad`]. RELAX (Grathwohl, Choi, Wu, Roeder & Duvenaud, ICLR 2018, arXiv:1711.00123)
//!   is the same estimator with the control variate replaced by a learned network; **the control
//!   variate here is FIXED** — `eta * f(relaxed)` at temperature `lambda`, with `eta` and `lambda`
//!   given by the caller — because unbiasedness holds for any fixed control variate and is the
//!   property this module is verified on. A learned `c_phi` would change the variance and not the
//!   mean.
//!
//! # What is actually different about these, in one line each
//!
//! | estimator | forward pass | gradient is OF | unbiased for `E[f(s)]` |
//! |---|---|---|---|
//! | [`gumbel_softmax_grad`] | relaxed `f(s_hat)` | the relaxed objective | **no**, at any `tau > 0` |
//! | [`straight_through_grad`] | discrete `f(s)` | nothing in particular | **no** |
//! | [`rebar_grad`] | discrete `f(s)` | the discrete objective | **yes**, at every `eta`, `lambda` |
//!
//! That is why [`GradEstimate`] carries `value` AND `discrete_value`: the first is the objective
//! the returned gradient differentiates, the second is `f` at the discrete draw. They are the same
//! float for straight-through and REBAR and they are not for the relaxation, and asserting that
//! difference is how the tests here prove a relaxation was implemented rather than the exact thing.
//!
//! # The parameterisation
//!
//! `theta[i]` is the LOGIT of spin `i` being up: `P(s_i = +1) = sigma(theta_i)`, independently per
//! site. That is deliberately the same parameterisation as [`crate::program::Gate::PNot`] reached
//! from an all-down start, so a [`crate::program::Program`] of `PNot` gates is the SAME
//! distribution and its `reinforce_grad` is a second, independently-written estimator of the same
//! quantity — which is what the variance comparison here is run against.
//!
//! # The closed form that referees everything
//!
//! For a MULTILINEAR objective — and [`Multilinear`], the Ising energy read off the corners, is
//! one — independence gives `E[f(s)] = f(m)` with `m_i = E[s_i] = 2 sigma(theta_i) - 1`, so
//!
//! ```text
//!   dL/dtheta_i = df/dx_i (m) * 2 sigma(theta_i) sigma(-theta_i)
//! ```
//!
//! exactly, in `O(edges)`, with no enumeration and no sampling — [`multilinear_grad`].
//! [`exact_grad`] instead sums `q(s) f(s) d log q / d theta` over all `2^n` states, which is a
//! different computation of the same number; the two agree to 1e-13, and it is the closed form the
//! estimators are judged against.
//!
//! # Nothing here is a bound
//!
//! Every number this module returns is an estimate with an error bar, never a certificate, so the
//! sums accumulate in plain arithmetic and [`crate::round`] does not appear. The draws are
//! independent by construction — one fresh RNG stream per sample, no chain — so the error bar is
//! `sqrt(var/N)` with `N` the sample count, and the effective-sample-size correction that
//! [`crate::samples`] applies to chains has nothing to correct here.

use crate::graph::Graph;
use crate::rng::Pcg;
use crate::samples::ENUMERATION_LIMIT;

/// Why a relaxation estimator refused to run.
#[derive(Clone, Debug, PartialEq)]
pub enum Refused {
    /// The temperature was zero, negative, or not finite.
    BadTemperature {
        /// The temperature asked for.
        tau: f64,
    },
    /// A logit was not finite, so the site has no distribution to sample or differentiate.
    BadLogit {
        /// Which site.
        index: usize,
        /// The value that was passed.
        theta: f64,
    },
    /// The control-variate scale was not finite.
    BadScale {
        /// The scale asked for.
        eta: f64,
    },
    /// There are no variables to estimate a gradient for.
    Empty,
    /// Zero samples were asked for, and a mean of nothing is not a number.
    NoSamples,
    /// The logit vector and the objective disagree about how many variables there are.
    Arity {
        /// Logits supplied.
        logits: usize,
        /// Variables the objective reads.
        objective: usize,
    },
    /// The exact gradient enumerates `2^n` states and `n` is past the limit.
    TooLargeToEnumerate {
        /// Variables in the model.
        spins: usize,
        /// The largest that will be enumerated.
        limit: usize,
    },
}

impl core::fmt::Display for Refused {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Refused::BadTemperature { tau } => write!(
                f,
                "temperature {tau} is not positive and finite: the Concrete relaxation divides by \
                 it, and at tau = 0 the map it defines is the step function the relaxation exists \
                 to avoid. Pass a small positive tau -- the bias falls with it and the variance \
                 rises"
            ),
            Refused::BadLogit { index, theta } => write!(
                f,
                "logit {index} is {theta}, which is not finite: P(s = +1) = sigma(theta) needs a \
                 real number, and an infinite logit is a fixed variable rather than a random one \
                 -- fix it out of the model instead"
            ),
            Refused::BadScale { eta } => write!(
                f,
                "control-variate scale {eta} is not finite; REBAR is unbiased for every finite \
                 eta, including 0, which turns it into plain single-sample REINFORCE"
            ),
            Refused::Empty => {
                write!(f, "no logits were given, so there is no gradient to estimate")
            }
            Refused::NoSamples => {
                write!(f, "zero samples were asked for, and the mean of no samples is not a number")
            }
            Refused::Arity { logits, objective } => write!(
                f,
                "{logits} logits were given for an objective over {objective} variables; the \
                 relaxation feeds one relaxed coordinate per logit into the objective, so the two \
                 counts are the same number or the call is a mistake"
            ),
            Refused::TooLargeToEnumerate { spins, limit } => write!(
                f,
                "the exact gradient sums over 2^{spins} states and the limit is 2^{limit}; for a \
                 multilinear objective use `multilinear_grad`, which is exact and O(edges)"
            ),
        }
    }
}

impl core::error::Error for Refused {}

/// The logistic function, `1 / (1 + e^-x)`.
#[inline]
fn sigma(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// `sigma'(x) = sigma(x) sigma(-x)`, written as the product so it is symmetric in `x` and never
/// rounds negative at either tail.
#[inline]
fn dsigma(x: f64) -> f64 {
    sigma(x) * sigma(-x)
}

/// One standard logistic deviate, `ln u - ln(1 - u)`.
///
/// This is the noise the BINARY Concrete is built on, and it is a Gumbel construction in disguise:
/// the difference of two independent standard Gumbels is exactly standard logistic, so
/// `argmax(theta + G_1, G_0)` and `sign(theta + L)` are the same draw. Drawing the difference
/// directly costs one uniform instead of two and is exact rather than a two-step reconstruction of
/// itself. Its variance is `pi^2/3`, which is the closed form
/// `the_logistic_noise_has_the_variance_the_closed_form_says` checks.
///
/// A zero uniform would give `-inf`; it is redrawn rather than clamped, which keeps the deviate
/// exactly logistic instead of putting an atom at the largest representable value.
#[must_use]
pub fn logistic(r: &mut Pcg) -> f64 {
    let mut u = r.f64();
    while u <= 0.0 {
        u = r.f64();
    }
    u.ln() - (-u).ln_1p()
}

/// Something a relaxation can be pushed through: a real-valued function of `n` real coordinates
/// that agrees with the discrete objective on the corners of `[-1, 1]^n`.
///
/// The trait carries the GRADIENT, and that is the point of it being a trait rather than a closure.
/// Every estimator here needs `f` defined and differentiable strictly INSIDE the cube — at points
/// no discrete state ever visits — and an objective that only knows how to score `{-1, +1}^n` has
/// nothing for the relaxation family to do. Making that a type-level requirement stops the mistake
/// of relaxing an objective that was never extended.
pub trait Objective {
    /// How many coordinates [`Objective::value`] and [`Objective::grad`] read.
    fn arity(&self) -> usize;
    /// The objective at a relaxed point, `x` in `[-1, 1]^arity`.
    fn value(&self, x: &[f64]) -> f64;
    /// `d value / d x` at `x`, written into `out` (length `arity`).
    fn grad(&self, x: &[f64], out: &mut [f64]);
}

/// The multilinear extension of an Ising energy: the same polynomial, read off the corners.
///
/// `E(x) = - sum_edges J_ij x_i x_j - sum_i h_i x_i`, which on `x` in `{-1, +1}^n` is exactly
/// [`Graph::energy`] — the same additions in the same order, so the two agree BIT FOR BIT and the
/// test asserting it uses `assert_eq!` rather than a tolerance. Off the corners it is the unique
/// multilinear interpolant, and its gradient is minus the local field, `-`[`Graph::field`].
///
/// Multilinear is not an accident of convenience. A [`Graph`] has no self-couplings, so no `x_i^2`
/// term exists, and for independent coordinates that makes `E[f(s)] = f(E[s])` — the closed form
/// this module's estimators are referee'd against.
pub struct Multilinear<'a> {
    g: &'a Graph,
}

impl<'a> Multilinear<'a> {
    /// The multilinear extension of `g`'s energy.
    #[must_use]
    pub fn new(g: &'a Graph) -> Self {
        Multilinear { g }
    }

    /// The graph being extended.
    #[must_use]
    pub fn graph(&self) -> &Graph {
        self.g
    }
}

impl Objective for Multilinear<'_> {
    fn arity(&self) -> usize {
        self.g.n
    }

    fn value(&self, x: &[f64]) -> f64 {
        // Deliberately the same loop shape as `Graph::energy`, so a corner evaluates to the same
        // bits rather than merely to the same number.
        let mut e = 0.0;
        for i in 0..self.g.n {
            let xi = x[i];
            e -= self.g.h[i] * xi;
            for k in self.g.offset[i]..self.g.offset[i + 1] {
                let j = self.g.nbr[k] as usize;
                if j > i {
                    e -= self.g.w[k] * xi * x[j];
                }
            }
        }
        e
    }

    fn grad(&self, x: &[f64], out: &mut [f64]) {
        for i in 0..self.g.n {
            let mut f = self.g.h[i];
            for k in self.g.offset[i]..self.g.offset[i + 1] {
                f += self.g.w[k] * x[self.g.nbr[k] as usize];
            }
            out[i] = -f;
        }
    }
}

/// A Gumbel-softmax / Concrete draw on the `k`-simplex: `softmax((log alpha + G) / tau)`.
///
/// The categorical form the two 2017 papers are written in. `logits` are unnormalised log-weights;
/// the returned vector is strictly inside the simplex and sums to one. Its ARGMAX is an exact
/// categorical draw from `softmax(logits)` at every temperature — the temperature divides the
/// perturbed scores and so cannot reorder them — which is the Gumbel-max trick, and is what makes
/// the straight-through variant's forward pass exact.
///
/// The Gumbels come from [`crate::perturb::gumbel`], which is zero-mean; the Euler–Mascheroni shift
/// is the same constant on every class and so cancels in both the softmax and the argmax.
///
/// # Errors
///
/// [`Refused::Empty`] for no classes, [`Refused::BadTemperature`] for a non-positive or
/// non-finite `tau`, [`Refused::BadLogit`] for a non-finite logit.
pub fn gumbel_softmax(logits: &[f64], tau: f64, r: &mut Pcg) -> Result<Vec<f64>, Refused> {
    if logits.is_empty() {
        return Err(Refused::Empty);
    }
    check_tau(tau)?;
    check_logits(logits)?;
    let z: Vec<f64> = logits.iter().map(|&l| (l + crate::perturb::gumbel(r)) / tau).collect();
    let mx = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut y: Vec<f64> = z.iter().map(|&v| (v - mx).exp()).collect();
    let total: f64 = y.iter().sum();
    for v in &mut y {
        *v /= total;
    }
    Ok(y)
}

/// The binary Concrete as a relaxed SPIN: `2 sigma(z / tau) - 1` in `(-1, +1)`, for a perturbed
/// score `z = theta + logistic noise`.
///
/// `sign(z)` is the discrete draw and does not depend on `tau`, so the relaxation's SIGN marginal
/// is exactly right at every temperature and only its magnitude softens. That is the coupling the
/// `tau -> 0` convergence test uses: the relaxed and the discrete draw are functions of the SAME
/// noise, so `E|s_hat - s|` bounds the Wasserstein distance between their laws.
#[must_use]
pub fn relaxed_spin(z: f64, tau: f64) -> f64 {
    2.0 * sigma(z / tau) - 1.0
}

/// `d relaxed_spin / dz`, which is `2 sigma'(z / tau) / tau`.
#[inline]
fn drelaxed(z: f64, tau: f64) -> f64 {
    2.0 * dsigma(z / tau) / tau
}

/// Which Jacobian the straight-through estimator pretends the sampler had.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Through {
    /// The binary-Concrete Jacobian at temperature `tau` — Jang et al.'s ST-Gumbel-Softmax: hard
    /// forward, relaxed backward.
    Concrete {
        /// Relaxation temperature, used for the backward pass only.
        tau: f64,
    },
    /// The identity, which is straight-through as originally stated (Hinton 2012; Bengio et al.
    /// 2013): the sampler is treated as if it had been `s = theta`. It has no temperature, so
    /// unlike [`Through::Concrete`] there is no knob that makes its bias small.
    Identity,
}

/// The fixed control variate REBAR subtracts: `eta * f(relaxed at lambda)`.
///
/// Unbiasedness holds for EVERY finite `eta` and every positive `lambda`, which is the whole point
/// of the construction — the control variate is subtracted inside the score-function term and added
/// back as a reparameterisation gradient, so its expectation cancels exactly. `eta` and `lambda`
/// therefore move the variance and nothing else, and both ends of that are asserted here.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ControlVariate {
    /// Scale on the relaxed objective. `0.0` removes the control variate entirely, leaving plain
    /// single-sample REINFORCE with no baseline.
    pub eta: f64,
    /// Temperature of the relaxation used as the control variate.
    pub lambda: f64,
}

impl ControlVariate {
    /// A control variate of scale `eta` at temperature `lambda`.
    ///
    /// # Errors
    ///
    /// [`Refused::BadScale`] for a non-finite `eta`, [`Refused::BadTemperature`] for a
    /// non-positive or non-finite `lambda`.
    pub fn new(eta: f64, lambda: f64) -> Result<Self, Refused> {
        if !eta.is_finite() {
            return Err(Refused::BadScale { eta });
        }
        check_tau(lambda)?;
        Ok(ControlVariate { eta, lambda })
    }

    /// No control variate: REBAR degenerates to single-sample REINFORCE without a baseline.
    #[must_use]
    pub fn none() -> Self {
        ControlVariate { eta: 0.0, lambda: 1.0 }
    }
}

/// A gradient estimate, with the spread of the samples behind it.
#[derive(Clone, Debug)]
pub struct GradEstimate {
    /// Sample mean of the per-sample estimates, one entry per logit.
    pub grad: Vec<f64>,
    /// Standard error of each coordinate of [`GradEstimate::grad`], `sqrt(var / samples)`. The
    /// draws are independent, so the sample count IS the effective sample count here.
    pub stderr: Vec<f64>,
    /// Sample variance of ONE sample's estimate, per coordinate. This is the number to compare
    /// estimators on: it does not move with the sample count.
    pub variance: Vec<f64>,
    /// Mean of the objective THIS ESTIMATOR'S GRADIENT DIFFERENTIATES: the relaxed objective for
    /// [`gumbel_softmax_grad`], the discrete one for the other two.
    pub value: f64,
    /// Standard error of [`GradEstimate::value`].
    pub value_stderr: f64,
    /// Mean of `f` at the DISCRETE sample drawn on each step, whatever the gradient refers to.
    /// Equal to [`GradEstimate::value`] bit for bit except in the relaxation.
    pub discrete_value: f64,
    /// Samples averaged.
    pub samples: usize,
}

impl GradEstimate {
    /// Summed per-coordinate variance of one sample — the single number estimators are fairly
    /// compared on, since a per-coordinate comparison can be won on one coordinate and lost on the
    /// rest.
    #[must_use]
    pub fn total_variance(&self) -> f64 {
        self.variance.iter().sum()
    }
}

/// Welford accumulator: mean and variance in one pass, without forming `E[x^2] - E[x]^2`.
struct Moments {
    n: usize,
    mean: Vec<f64>,
    m2: Vec<f64>,
}

impl Moments {
    fn new(k: usize) -> Self {
        Moments { n: 0, mean: vec![0.0; k], m2: vec![0.0; k] }
    }

    fn push(&mut self, x: &[f64]) {
        self.n += 1;
        let n = self.n as f64;
        for i in 0..self.mean.len() {
            let d = x[i] - self.mean[i];
            self.mean[i] += d / n;
            self.m2[i] += d * (x[i] - self.mean[i]);
        }
    }

    /// Sample variance with the `n - 1` denominator; zero for a single sample.
    fn variance(&self) -> Vec<f64> {
        if self.n > 1 {
            self.m2.iter().map(|&v| v / (self.n - 1) as f64).collect()
        } else {
            vec![0.0; self.mean.len()]
        }
    }
}

fn check_tau(tau: f64) -> Result<(), Refused> {
    if tau > 0.0 && tau.is_finite() { Ok(()) } else { Err(Refused::BadTemperature { tau }) }
}

fn check_logits(theta: &[f64]) -> Result<(), Refused> {
    for (index, &t) in theta.iter().enumerate() {
        if !t.is_finite() {
            return Err(Refused::BadLogit { index, theta: t });
        }
    }
    Ok(())
}

fn check_call<O: Objective>(theta: &[f64], f: &O, samples: usize) -> Result<(), Refused> {
    if theta.is_empty() {
        return Err(Refused::Empty);
    }
    if theta.len() != f.arity() {
        return Err(Refused::Arity { logits: theta.len(), objective: f.arity() });
    }
    if samples == 0 {
        return Err(Refused::NoSamples);
    }
    check_logits(theta)
}

/// The perturbed scores and the discrete draw they imply: `z_i = theta_i + L_i`, `s_i = sign(z_i)`.
///
/// Every estimator in this module starts here and draws its noise in this order, so at a fixed seed
/// they all see the SAME discrete sample on every step. That is common random numbers BETWEEN
/// estimators, and it is what makes the variance comparison a comparison of estimators rather than
/// of seeds.
fn perturb(theta: &[f64], r: &mut Pcg, z: &mut [f64], s: &mut [f64]) {
    for i in 0..theta.len() {
        z[i] = theta[i] + logistic(r);
        s[i] = if z[i] > 0.0 { 1.0 } else { -1.0 };
    }
}

/// `d log q(s_i) / d theta_i` for `P(s_i = +1) = sigma(theta_i)`: `s_i sigma(-theta_i s_i)`.
///
/// Identical to the score [`crate::program::Gate::PNot`] accumulates, which is not a coincidence —
/// it is the same distribution written twice, and the tests here rely on that.
#[inline]
fn score(theta: f64, s: f64) -> f64 {
    s * sigma(-theta * s)
}

/// The Gumbel-softmax / Concrete gradient estimator: differentiate `f` through the relaxation.
///
/// Per sample, with `z_i = theta_i + L_i` and `s_hat_i = 2 sigma(z_i / tau) - 1`,
///
/// ```text
///   g_i = df/dx_i(s_hat) * d s_hat_i/d theta_i ,   d s_hat_i/d theta_i = 2 sigma'(z_i/tau)/tau
/// ```
///
/// This is an UNBIASED estimator of the gradient of the RELAXED objective `E[f(s_hat)]` and a
/// BIASED estimator of the gradient of `E[f(s)]`, which is the trade the method is named for. The
/// bias falls with `tau` and the variance rises like `1/tau`; both directions are measured in
/// `the_relaxation_trades_bias_for_variance_along_the_temperature_ladder`.
///
/// # Errors
///
/// [`Refused::Empty`], [`Refused::Arity`], [`Refused::NoSamples`], [`Refused::BadLogit`],
/// [`Refused::BadTemperature`].
pub fn gumbel_softmax_grad<O: Objective>(
    theta: &[f64],
    tau: f64,
    f: &O,
    samples: usize,
    seed: u64,
) -> Result<GradEstimate, Refused> {
    check_call(theta, f, samples)?;
    check_tau(tau)?;
    let n = theta.len();
    let (mut z, mut s) = (vec![0.0; n], vec![0.0; n]);
    let (mut relaxed, mut df) = (vec![0.0; n], vec![0.0; n]);
    let mut est = vec![0.0; n];
    let mut mom = Moments::new(n);
    let mut vals = Moments::new(2);
    for k in 0..samples {
        let mut r = Pcg::new(seed, k as u64);
        perturb(theta, &mut r, &mut z, &mut s);
        for i in 0..n {
            relaxed[i] = relaxed_spin(z[i], tau);
        }
        f.grad(&relaxed, &mut df);
        for i in 0..n {
            est[i] = df[i] * drelaxed(z[i], tau);
        }
        mom.push(&est);
        vals.push(&[f.value(&relaxed), f.value(&s)]);
    }
    Ok(finish(mom, &vals, samples))
}

/// The straight-through estimator: the discrete sample goes forward, a relaxed Jacobian comes back.
///
/// Per sample, with `s_i = sign(theta_i + L_i)` the exact discrete draw,
///
/// ```text
///   g_i = df/dx_i(s) * J_i ,      J_i = 2 sigma'(z_i/tau)/tau   or   1
/// ```
///
/// according to [`Through`]. The forward pass is exactly the discrete objective — no relaxation
/// enters `value` — and the backward pass is a substitution with no expectation identity behind it,
/// so this is biased and stays biased. It is here because it is what the discrete-variable
/// literature is overwhelmingly built on in practice, and because having it beside [`rebar_grad`]
/// is what makes that bias a number rather than a caveat.
///
/// # Errors
///
/// [`Refused::Empty`], [`Refused::Arity`], [`Refused::NoSamples`], [`Refused::BadLogit`], and
/// [`Refused::BadTemperature`] from [`Through::Concrete`].
pub fn straight_through_grad<O: Objective>(
    theta: &[f64],
    through: Through,
    f: &O,
    samples: usize,
    seed: u64,
) -> Result<GradEstimate, Refused> {
    check_call(theta, f, samples)?;
    if let Through::Concrete { tau } = through {
        check_tau(tau)?;
    }
    let n = theta.len();
    let (mut z, mut s) = (vec![0.0; n], vec![0.0; n]);
    let mut df = vec![0.0; n];
    let mut est = vec![0.0; n];
    let mut mom = Moments::new(n);
    let mut vals = Moments::new(2);
    for k in 0..samples {
        let mut r = Pcg::new(seed, k as u64);
        perturb(theta, &mut r, &mut z, &mut s);
        f.grad(&s, &mut df);
        for i in 0..n {
            let jac = match through {
                Through::Concrete { tau } => drelaxed(z[i], tau),
                Through::Identity => 1.0,
            };
            est[i] = df[i] * jac;
        }
        mom.push(&est);
        let v = f.value(&s);
        vals.push(&[v, v]);
    }
    Ok(finish(mom, &vals, samples))
}

/// REBAR: the relaxation as a control variate, so the estimator stays unbiased.
///
/// Per sample, with `z = theta + L`, `s = sign(z)` the discrete draw, `r(.)` the relaxation at
/// `lambda`, and `z_tilde ~ p(z | s)` the relaxation CONDITIONED on that draw,
///
/// ```text
///   g_i = [ f(s) - eta f(r(z_tilde)) ] d log q(s_i)/d theta_i
///       + eta df/dx_i(r(z))       dr/dz(z_i)
///       - eta df/dx_i(r(z_tilde)) dr/dz(z_tilde_i) dz_tilde_i/dtheta_i
/// ```
///
/// The first line is the score function with the control variate subtracted; the second and third
/// add its expectation back as reparameterisation gradients, which is exactly why the bias cancels
/// for any `eta` and any `lambda`. The third term's `dz_tilde/dtheta` is the part that is easy to
/// drop: `z_tilde` depends on `theta` TWICE, once explicitly and once through the conditional
/// draw's own `sigma(theta)`, and omitting the second dependence leaves an estimator that still
/// runs, still has low variance, and is no longer unbiased.
///
/// The conditional draw, from `v ~ U(0,1)` with `p = sigma(theta)` and `q = sigma(-theta)`:
///
/// ```text
///   s = +1:  u = q + p v,   dz_tilde/dtheta = 1 - q/u
///   s = -1:  u = q v,       dz_tilde/dtheta = 1 - p/(1 - u)
///   z_tilde = theta + ln u - ln(1 - u)
/// ```
///
/// which is the uniform that produced `z` restricted to the half-line the draw landed in, so
/// `sign(z_tilde) = s` by construction.
///
/// # Errors
///
/// [`Refused::Empty`], [`Refused::Arity`], [`Refused::NoSamples`], [`Refused::BadLogit`],
/// [`Refused::BadScale`], [`Refused::BadTemperature`].
pub fn rebar_grad<O: Objective>(
    theta: &[f64],
    cv: ControlVariate,
    f: &O,
    samples: usize,
    seed: u64,
) -> Result<GradEstimate, Refused> {
    check_call(theta, f, samples)?;
    let ControlVariate { eta, lambda } = cv;
    if !eta.is_finite() {
        return Err(Refused::BadScale { eta });
    }
    check_tau(lambda)?;
    let n = theta.len();
    let (mut z, mut s) = (vec![0.0; n], vec![0.0; n]);
    let (mut zt, mut dzt) = (vec![0.0; n], vec![0.0; n]);
    let (mut r_un, mut r_cond) = (vec![0.0; n], vec![0.0; n]);
    let (mut df_un, mut df_cond) = (vec![0.0; n], vec![0.0; n]);
    let mut est = vec![0.0; n];
    let mut mom = Moments::new(n);
    let mut vals = Moments::new(2);
    for k in 0..samples {
        let mut r = Pcg::new(seed, k as u64);
        perturb(theta, &mut r, &mut z, &mut s);
        for i in 0..n {
            let mut v = r.f64();
            while v <= 0.0 {
                v = r.f64();
            }
            let (p, q) = (sigma(theta[i]), sigma(-theta[i]));
            let u = if s[i] > 0.0 { q + p * v } else { q * v };
            zt[i] = theta[i] + u.ln() - (-u).ln_1p();
            dzt[i] = if s[i] > 0.0 { 1.0 - q / u } else { 1.0 - p / (1.0 - u) };
            r_un[i] = relaxed_spin(z[i], lambda);
            r_cond[i] = relaxed_spin(zt[i], lambda);
        }
        let f_hard = f.value(&s);
        let f_cond = f.value(&r_cond);
        f.grad(&r_un, &mut df_un);
        f.grad(&r_cond, &mut df_cond);
        for i in 0..n {
            est[i] = (f_hard - eta * f_cond) * score(theta[i], s[i])
                + eta * df_un[i] * drelaxed(z[i], lambda)
                - eta * df_cond[i] * drelaxed(zt[i], lambda) * dzt[i];
        }
        mom.push(&est);
        vals.push(&[f_hard, f_hard]);
    }
    Ok(finish(mom, &vals, samples))
}

/// Pack the accumulators into the public estimate.
fn finish(mom: Moments, vals: &Moments, samples: usize) -> GradEstimate {
    let variance = mom.variance();
    let sn = samples as f64;
    let stderr = variance.iter().map(|v| (v / sn).sqrt()).collect();
    GradEstimate {
        grad: mom.mean,
        stderr,
        variance,
        value: vals.mean[0],
        value_stderr: (vals.variance()[0] / sn).sqrt(),
        discrete_value: vals.mean[1],
        samples,
    }
}

/// `E[f(s)]` under independent spins, by enumerating all `2^n` states. The referee for `value`.
///
/// # Errors
///
/// [`Refused::Empty`], [`Refused::Arity`], [`Refused::BadLogit`], and
/// [`Refused::TooLargeToEnumerate`] past [`crate::samples::ENUMERATION_LIMIT`].
pub fn exact_value<O: Objective>(theta: &[f64], f: &O) -> Result<f64, Refused> {
    Ok(enumerate(theta, f)?.0)
}

/// The EXACT gradient of `E_{s ~ q_theta}[f(s)]`, by enumerating all `2^n` states.
///
/// `dL/dtheta_i = sum_s q(s) f(s) s_i sigma(-theta_i s_i)` — no sampling, no relaxation, and no
/// line of code shared with any estimator in this module. It is the oracle the estimators are
/// judged against, and it is itself judged against the `O(edges)` closed form
/// [`multilinear_grad`].
///
/// # Errors
///
/// [`Refused::Empty`], [`Refused::Arity`], [`Refused::BadLogit`], and
/// [`Refused::TooLargeToEnumerate`] past [`crate::samples::ENUMERATION_LIMIT`].
pub fn exact_grad<O: Objective>(theta: &[f64], f: &O) -> Result<Vec<f64>, Refused> {
    Ok(enumerate(theta, f)?.1)
}

fn enumerate<O: Objective>(theta: &[f64], f: &O) -> Result<(f64, Vec<f64>), Refused> {
    if theta.is_empty() {
        return Err(Refused::Empty);
    }
    if theta.len() != f.arity() {
        return Err(Refused::Arity { logits: theta.len(), objective: f.arity() });
    }
    check_logits(theta)?;
    let n = theta.len();
    if n > ENUMERATION_LIMIT {
        return Err(Refused::TooLargeToEnumerate { spins: n, limit: ENUMERATION_LIMIT });
    }
    let mut grad = vec![0.0; n];
    let mut mean = 0.0;
    let mut x = vec![0.0; n];
    for mask in 0..1usize << n {
        let mut q = 1.0;
        for i in 0..n {
            x[i] = if mask >> i & 1 == 1 { 1.0 } else { -1.0 };
            q *= sigma(theta[i] * x[i]);
        }
        let fv = f.value(&x);
        mean += q * fv;
        for i in 0..n {
            grad[i] += q * fv * score(theta[i], x[i]);
        }
    }
    Ok((mean, grad))
}

/// The `O(edges)` closed form for the gradient of a MULTILINEAR objective under independent spins.
///
/// `E[f(s)] = f(m)` with `m_i = 2 sigma(theta_i) - 1`, so
/// `dL/dtheta_i = df/dx_i(m) * 2 sigma'(theta_i)`. Exact, with no enumeration — and therefore
/// usable as an oracle well past the `2^n` wall, which is what makes it the referee rather than
/// [`exact_grad`].
///
/// **Valid only for a multilinear `f`.** A [`Graph`] has no self-couplings, so [`Multilinear`] is
/// one; an objective with an `x_i^2` term is not, and this returns a wrong answer for it rather
/// than an error, because nothing in the type system distinguishes the two.
///
/// # Errors
///
/// [`Refused::Empty`], [`Refused::Arity`], [`Refused::BadLogit`].
pub fn multilinear_grad<O: Objective>(theta: &[f64], f: &O) -> Result<Vec<f64>, Refused> {
    if theta.is_empty() {
        return Err(Refused::Empty);
    }
    if theta.len() != f.arity() {
        return Err(Refused::Arity { logits: theta.len(), objective: f.arity() });
    }
    check_logits(theta)?;
    let m: Vec<f64> = theta.iter().map(|&t| 2.0 * sigma(t) - 1.0).collect();
    let mut df = vec![0.0; theta.len()];
    f.grad(&m, &mut df);
    for i in 0..theta.len() {
        df[i] *= 2.0 * dsigma(theta[i]);
    }
    Ok(df)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::GraphBuilder;
    use crate::program::{Force, Gate, Program, State};

    /// A small dense spin glass: enumerable, frustrated, biased, and with an O(1) energy scale, so
    /// nothing here is testing overflow by accident.
    fn glass(n: usize, seed: u64) -> Graph {
        let mut r = Pcg::new(seed, 17);
        let mut b = GraphBuilder::new(n);
        for i in 0..n {
            b.bias(i, 0.8 * (r.f64() - 0.5));
            for j in (i + 1)..n {
                if r.f64() < 0.6 {
                    b.couple(i, j, 1.4 * (r.f64() - 0.5));
                }
            }
        }
        b.build()
    }

    /// Asymmetric logits — deliberately not centred on zero, because a symmetric `theta` with a
    /// symmetric objective can make a biased estimator's bias cancel and the headline test vacuous.
    fn logits(n: usize, seed: u64) -> Vec<f64> {
        let mut r = Pcg::new(seed, 3);
        (0..n).map(|_| 1.6 * (r.f64() - 0.4)).collect()
    }

    /// The objective plus a constant. Still multilinear (a constant is degree zero), so the closed
    /// form still applies, and the exact gradient is unchanged because `sum_s q(s) score(s) = 0`.
    ///
    /// It exists to separate the estimators that carry a baseline from the ones that do not: a
    /// constant is invisible to the gradient and enormous to the variance of a bare score function.
    struct Shifted<'a> {
        inner: Multilinear<'a>,
        c: f64,
    }

    impl Objective for Shifted<'_> {
        fn arity(&self) -> usize {
            self.inner.arity()
        }
        fn value(&self, x: &[f64]) -> f64 {
            self.inner.value(x) + self.c
        }
        fn grad(&self, x: &[f64], out: &mut [f64]) {
            self.inner.grad(x, out);
        }
    }

    /// The worst per-coordinate deviation from `want`, in units of the estimate's own standard
    /// error. This is the CLT yardstick every convergence assertion here is stated in.
    fn worst_sigma(e: &GradEstimate, want: &[f64]) -> f64 {
        (0..want.len()).map(|i| (e.grad[i] - want[i]).abs() / e.stderr[i]).fold(0.0f64, f64::max)
    }

    fn worst_abs(got: &[f64], want: &[f64]) -> f64 {
        (0..want.len()).map(|i| (got[i] - want[i]).abs()).fold(0.0f64, f64::max)
    }

    /// ORACLE: the multilinear closed form, `E[f(s)] = f(m)`, computed in `O(edges)` with no
    /// enumeration — a completely different computation from the `2^n` sum in [`exact_grad`].
    ///
    /// This is the referee the estimator tests are written against, so it is the first thing that
    /// has to be true. Measured worst deviation over four sizes: 1.2e-16.
    #[test]
    fn the_enumerated_gradient_matches_the_multilinear_closed_form() {
        for (n, gs, ls) in [(4usize, 1u64, 2u64), (8, 7, 11), (10, 3, 5), (12, 21, 8)] {
            let g = glass(n, gs);
            let f = Multilinear::new(&g);
            let th = logits(n, ls);
            let enumerated = exact_grad(&th, &f).unwrap();
            let closed = multilinear_grad(&th, &f).unwrap();
            let w = worst_abs(&enumerated, &closed);
            assert!(w < 1e-13, "n={n}: enumeration and closed form differ by {w:e}");

            // And the same for the objective itself: E[f(s)] must be f(E[s]).
            let m: Vec<f64> = th.iter().map(|&t| 2.0 * sigma(t) - 1.0).collect();
            let d = (exact_value(&th, &f).unwrap() - f.value(&m)).abs();
            assert!(d < 1e-13, "n={n}: E[f] and f(E[s]) differ by {d:e}");

            // A non-trivial gradient, or the agreement above means nothing.
            let scale = closed.iter().fold(0.0f64, |a, b| a.max(b.abs()));
            assert!(scale > 0.02, "n={n}: the fixture's gradient is {scale:e}, too flat to test");
        }
    }

    /// ORACLE: [`Graph::energy`] and [`Graph::field`], which have their own verification elsewhere
    /// in this crate. The extension must agree on every corner BIT FOR BIT, not approximately — a
    /// relaxation that disagreed with the discrete objective at the corners would be relaxing a
    /// different problem.
    #[test]
    fn the_multilinear_extension_equals_graph_energy_on_every_corner() {
        let g = glass(10, 4);
        let f = Multilinear::new(&g);
        let mut x = vec![0.0; g.n];
        let mut s = vec![0i8; g.n];
        let mut out = vec![0.0; g.n];
        for mask in 0..1usize << g.n {
            for i in 0..g.n {
                s[i] = if mask >> i & 1 == 1 { 1 } else { -1 };
                x[i] = f64::from(s[i]);
            }
            assert_eq!(f.value(&x), g.energy(&s), "corner {mask} disagrees with Graph::energy");
            f.grad(&x, &mut out);
            for i in 0..g.n {
                assert_eq!(out[i], -g.field(i, &s), "corner {mask} site {i}: grad is not -field");
            }
        }
    }

    /// THE HEADLINE, and it is a pair: the same oracle, the same seed, the same sample count, and
    /// opposite verdicts.
    ///
    /// ORACLE: [`multilinear_grad`], the `O(edges)` closed form — independent of every line of the
    /// estimators.
    ///
    /// REBAR is unbiased, so its mean must land inside the CLT band: five standard errors is
    /// `p = 6e-7` per coordinate, 5e-6 over the eight. Measured worst: **1.42 sigma**.
    ///
    /// The Concrete relaxation at `tau = 1` is BIASED, and asserting that is what proves a
    /// relaxation was implemented rather than the exact thing wearing its name. Measured worst
    /// bias: **0.115, which is 267 standard errors** — not a test that could pass on noise, and not
    /// a tolerance that could be widened into agreement.
    #[test]
    fn rebar_converges_to_the_closed_form_gradient_and_gumbel_softmax_does_not() {
        let g = glass(8, 7);
        let f = Multilinear::new(&g);
        let th = logits(8, 11);
        let exact = multilinear_grad(&th, &f).unwrap();
        let n = 200_000;

        let rebar = rebar_grad(&th, ControlVariate::new(1.0, 0.5).unwrap(), &f, n, 5).unwrap();
        let w = worst_sigma(&rebar, &exact);
        assert!(w < 5.0, "REBAR is off the exact gradient by {w:.2} standard errors");
        // ... and its forward value is the discrete objective, also inside the CLT band.
        let ev = exact_value(&th, &f).unwrap();
        let dv = (rebar.value - ev).abs() / rebar.value_stderr;
        assert!(dv < 5.0, "REBAR's value is {dv:.2} standard errors off E[f] = {ev}");

        let gs = gumbel_softmax_grad(&th, 1.0, &f, n, 5).unwrap();
        let bias = worst_abs(&gs.grad, &exact);
        let sig = worst_sigma(&gs, &exact);
        assert!(
            sig > 50.0 && bias > 0.05,
            "the relaxation must be BIASED at tau = 1: worst bias {bias:.5} ({sig:.1} sigma). An \
             implementation that agrees here is not a relaxation"
        );

        // The two saw the same discrete draws -- common random numbers, same seed -- so the
        // disagreement is the estimator and not the sample.
        assert_eq!(
            gs.discrete_value, rebar.discrete_value,
            "the two estimators did not draw the same discrete samples at the same seed"
        );
    }

    /// The trade the relaxation is named for, measured in both directions on the same seed.
    ///
    /// ORACLE: [`multilinear_grad`] again, for the bias half. Measured, N = 100 000, seed 5:
    ///
    /// | tau | worst bias | total variance |
    /// |---|---|---|
    /// | 1.0 | 0.115 | 0.21 |
    /// | 0.5 | 0.051 | 0.84 |
    /// | 0.2 | 0.012 | 3.21 |
    /// | 0.1 | 0.004 | 7.32 |
    ///
    /// Asserting only that the bias falls would pass for an implementation whose variance also
    /// fell, which would be a better estimator than exists; the point of the method is that the
    /// second column is the price of the first.
    #[test]
    fn the_relaxation_trades_bias_for_variance_along_the_temperature_ladder() {
        let g = glass(8, 7);
        let f = Multilinear::new(&g);
        let th = logits(8, 11);
        let exact = multilinear_grad(&th, &f).unwrap();
        let mut prev: Option<(f64, f64)> = None;
        for tau in [1.0, 0.5, 0.2, 0.1] {
            let e = gumbel_softmax_grad(&th, tau, &f, 100_000, 5).unwrap();
            let bias = worst_abs(&e.grad, &exact);
            let var = e.total_variance();
            if let Some((pb, pv)) = prev {
                assert!(bias < pb, "tau={tau}: bias {bias:.5} did not fall below {pb:.5}");
                assert!(var > pv, "tau={tau}: variance {var:.4} did not rise above {pv:.4}");
            }
            prev = Some((bias, var));
        }
        let (bias, var) = prev.unwrap();
        assert!(bias < 0.01, "the coldest rung should be nearly unbiased, got {bias:.5}");
        assert!(var > 5.0, "and should have paid for it in variance, got {var:.4}");
    }

    /// (2) As `tau -> 0` the relaxation converges to the discrete draw IN DISTRIBUTION.
    ///
    /// Asserted through a coupling, which is stronger than a histogram and needs no binning: both
    /// are functions of the SAME logistic deviate, so `E|s_hat - s|` is an upper bound on the
    /// Wasserstein-1 distance between their laws, and a bound that goes to zero IS convergence in
    /// distribution.
    ///
    /// ORACLE: that bound has a closed-form limit. `|s_hat - s| = 2 sigma(-|z|/tau)` exactly, and
    /// `z` has the logistic density `sigma(z - theta) sigma(theta - z)`, which at zero is
    /// `sigma'(theta)`; `integral of 2 sigma(-|w|) dw = 4 ln 2`; so
    ///
    /// ```text
    ///   E|s_hat - s| / tau  ->  4 ln(2) sigma'(theta)  =  0.634325   at theta = 0.6
    /// ```
    ///
    /// MEASURED down the ladder, 100 000 draws each: 0.4779, 0.5727, 0.6177, **0.6255, 0.6299,
    /// 0.6358, 0.6417** — within 1.4% of the closed form from `tau = 0.1` down, and 25% away at
    /// `tau = 1`, which is the whole content of the word "converges".
    ///
    /// The asymmetric half: the SIGN is exactly the discrete draw at EVERY temperature, including
    /// `tau = 1` where the coupling error is 0.47. Convergence in distribution is a limit; the sign
    /// marginal is not, and an implementation that reached the limit by softening the sign as well
    /// would fail this line while passing the one above.
    #[test]
    fn at_zero_temperature_the_relaxation_converges_to_the_discrete_draw() {
        let theta = 0.6f64;
        let m = 100_000;
        let limit = 4.0 * 2.0f64.ln() * dsigma(theta);
        let mut prev = f64::INFINITY;
        let mut coldest = 0.0;
        let mut saturated_at = Vec::new();
        for tau in [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01] {
            let mut r = Pcg::new(99, 1);
            let (mut coupling, mut mean_relaxed, mut up, mut corner) = (0.0, 0.0, 0usize, 0usize);
            for _ in 0..m {
                let z = theta + logistic(&mut r);
                let hard = if z > 0.0 { 1.0 } else { -1.0 };
                let soft = relaxed_spin(z, tau);
                coupling += (soft - hard).abs();
                mean_relaxed += soft;
                up += usize::from(soft > 0.0);
                corner += usize::from(soft.abs() >= 1.0);
                assert_eq!(
                    soft > 0.0,
                    hard > 0.0,
                    "tau={tau}: the relaxation changed the sign of the draw"
                );
            }
            coupling /= m as f64;
            mean_relaxed /= m as f64;
            assert!(coupling < prev, "tau={tau}: coupling {coupling:.6} did not fall");
            let ratio = coupling / tau;
            assert!(ratio < 1.1 * limit, "tau={tau}: coupling/tau {ratio:.4} is not O(tau)");
            if tau <= 0.1 {
                assert!(
                    (ratio / limit - 1.0).abs() < 0.03,
                    "tau={tau}: coupling/tau {ratio:.4} against the closed form {limit:.6}"
                );
            } else if tau >= 0.5 {
                // ... and the limit is a LIMIT: at tau = 0.5 and above the coupling is nowhere
                // near it, by 10% and 25%.
                assert!(
                    ratio < 0.95 * limit,
                    "tau={tau}: coupling/tau {ratio:.4} already reached the closed form {limit:.6}"
                );
            }
            prev = coupling;
            coldest = mean_relaxed;
            saturated_at.push(corner);

            // The sign marginal is exact at EVERY temperature: P(s_hat > 0) = sigma(theta), and
            // four standard deviations of Binomial(m, sigma(theta)) is the whole allowance.
            let p = sigma(theta);
            let dev = (up as f64 - m as f64 * p).abs();
            let tol = 4.0 * (m as f64 * p * (1.0 - p)).sqrt();
            assert!(dev < tol, "tau={tau}: {up} of {m} up, off by {dev:.0} (tol {tol:.0})");
        }
        assert!(prev < 0.01, "the coldest rung is still {prev:.6} from the discrete draw");
        // And the first moment has converged to the discrete one, `E[s] = 2 sigma(theta) - 1`.
        let m1 = 2.0 * sigma(theta) - 1.0;
        assert!(
            (coldest - m1).abs() < 0.01,
            "at tau = 0.01 the relaxed mean is {coldest:.5}, discrete mean {m1:.5}"
        );
        // The convergence is a LIMIT in exact arithmetic and an ARRIVAL in f64: at tau = 1 not one
        // draw of 100 000 reaches the corner, and at tau = 0.01 most of them do, because
        // `sigma(z/tau)` rounds to exactly 1 once `|z|/tau` passes ~37. Worth asserting rather than
        // discovering: below that temperature the relaxed draw IS the discrete draw, bit for bit,
        // and its gradient is exactly zero.
        assert_eq!(saturated_at[0], 0, "tau = 1 should never reach the corner in f64");
        assert!(
            saturated_at[6] > m / 2,
            "tau = 0.01 should saturate to the corner; only {} of {m} did",
            saturated_at[6]
        );
    }

    /// Straight-through's forward pass is the DISCRETE sample, exactly — that is the whole claim of
    /// the method, and it is the one thing about it that is exactly true.
    ///
    /// Asserted bit for bit against [`rebar_grad`] at the same seed, which draws the same discrete
    /// samples from the same stream. The relaxation, on the same seed and the same draws, reports a
    /// DIFFERENT forward value, because its gradient refers to a different objective. Both halves
    /// are needed: the equality alone would pass for three estimators that were secretly one.
    #[test]
    fn the_straight_through_forward_pass_is_exactly_the_discrete_sample() {
        let g = glass(8, 7);
        let f = Multilinear::new(&g);
        let th = logits(8, 11);
        let n = 50_000;
        let st = straight_through_grad(&th, Through::Concrete { tau: 1.0 }, &f, n, 5).unwrap();
        let rb = rebar_grad(&th, ControlVariate::new(1.0, 0.5).unwrap(), &f, n, 5).unwrap();
        let gs = gumbel_softmax_grad(&th, 1.0, &f, n, 5).unwrap();

        assert_eq!(st.value, st.discrete_value, "straight-through's forward pass is not discrete");
        assert_eq!(rb.value, rb.discrete_value, "REBAR's forward pass is not discrete");
        assert_eq!(st.value, rb.value, "the same seed gave two different discrete forward passes");
        assert_eq!(gs.discrete_value, rb.discrete_value, "the discrete draws are not shared");
        assert!(
            (gs.value - gs.discrete_value).abs() > 0.01,
            "the relaxed objective {} and the discrete one {} are the same number, so nothing was \
             relaxed",
            gs.value,
            gs.discrete_value
        );

        // And straight-through is biased -- measured worst bias 0.090, 164 standard errors, with
        // the identity Jacobian three times worse again at 0.286.
        let exact = multilinear_grad(&th, &f).unwrap();
        let sig = worst_sigma(&st, &exact);
        assert!(sig > 20.0, "straight-through must be biased; it is within {sig:.1} sigma");
        let id = straight_through_grad(&th, Through::Identity, &f, n, 5).unwrap();
        assert!(
            worst_abs(&id.grad, &exact) > worst_abs(&st.grad, &exact),
            "the identity Jacobian should be further from the truth than the Concrete one"
        );
    }

    /// (3) VARIANCE, against [`crate::program::Program::reinforce_grad`] — a REINFORCE written
    /// independently of this module, on the same distribution, the same objective and the same
    /// seeds.
    ///
    /// The `PNot` program from an all-down start IS `q_theta`: gate `i` flips spin `i` with
    /// probability `sigma(theta_i)`, so `P(s_i = +1) = sigma(theta_i)`. Checked here rather than
    /// assumed, because a variance comparison between two different distributions measures nothing.
    ///
    /// MEASURED, per-sample total variance over 400 batches of 512, seeds 9000..9400:
    ///
    /// | estimator | variance | biased |
    /// |---|---|---|
    /// | `program::reinforce_grad` (batch-mean baseline) | 5.41 | at O(1/B) — see the next test |
    /// | `rebar_grad`, eta = 1, lambda = 0.5 | 3.16 | no |
    /// | `gumbel_softmax_grad`, tau = 1 | 0.25 | **yes** |
    ///
    /// So the direction is: REBAR is 1.7x quieter than REINFORCE here, and the relaxation is 21x
    /// quieter than REINFORCE and wrong. That ordering is the measurement and not an assumption —
    /// the control variate could have lost, and on an objective whose mean is far from zero the
    /// baseline REINFORCE carries would shrink the first row instead (see the constant-shift test,
    /// where a bare score function goes up 143x and both baselined estimators do not move).
    #[test]
    fn rebar_is_quieter_than_program_reinforce_and_the_relaxation_is_quieter_still() {
        let n = 8;
        let g = glass(n, 7);
        let f = Multilinear::new(&g);
        let th = logits(n, 11);
        let gates: Vec<Gate> = (0..n).map(|i| Gate::PNot { bit: i, p_theta: i }).collect();
        let prog = Program { gates, graphs: Vec::new(), n_params: n };
        let init = State { bits: vec![-1i8; n], reals: Vec::new() };
        let loss = |s: &State| g.energy(&s.bits);

        // The program and this module are the same distribution, or the comparison is meaningless.
        let draws = 100_000;
        let mut up = vec![0u32; n];
        for e in 0..draws {
            let mut r = Pcg::new(1234, e as u64);
            let st = prog.run(&init, &mut r, None, Force::None, &th, None, None);
            for i in 0..n {
                up[i] += u32::from(st.bits[i] == 1);
            }
        }
        for i in 0..n {
            let p = sigma(th[i]);
            let dev = (f64::from(up[i]) - f64::from(draws) * p).abs();
            let tol = 4.0 * (f64::from(draws) * p * (1.0 - p)).sqrt();
            assert!(dev < tol, "site {i}: the program's marginal is not sigma(theta), off {dev:.0}");
        }

        let (b, reps) = (512usize, 400usize);
        let cv = ControlVariate::new(1.0, 0.5).unwrap();
        let mut acc = [Moments::new(n), Moments::new(n), Moments::new(n)];
        for rep in 0..reps {
            let seed = 9_000 + rep as u64;
            let (r, _) = prog.reinforce_grad(&init, &th, &loss, b, seed);
            acc[0].push(&r);
            acc[1].push(&rebar_grad(&th, cv, &f, b, seed).unwrap().grad);
            acc[2].push(&gumbel_softmax_grad(&th, 1.0, &f, b, seed).unwrap().grad);
        }
        let per_sample: Vec<f64> =
            acc.iter().map(|m| m.variance().iter().sum::<f64>() * b as f64).collect();
        let (v_reinforce, v_rebar, v_gs) = (per_sample[0], per_sample[1], per_sample[2]);
        assert!(
            v_rebar * 1.25 < v_reinforce,
            "measured REBAR {v_rebar:.4} against REINFORCE {v_reinforce:.4}; the direction this \
             test records is REBAR quieter"
        );
        assert!(v_gs * 5.0 < v_rebar, "measured relaxation {v_gs:.4} against REBAR {v_rebar:.4}");

        // The cheap one is the wrong one: the same batches, 21x quieter, and off the exact answer.
        let exact = multilinear_grad(&th, &f).unwrap();
        let se_gs = (acc[2].variance().iter().sum::<f64>() / reps as f64).sqrt();
        assert!(
            worst_abs(&acc[2].mean, &exact) > 10.0 * se_gs,
            "the relaxation's mean over {reps} batches should still be resolvably off the truth"
        );
    }

    /// A constant added to the objective changes NO gradient — `sum_s q(s) d log q/d theta = 0` —
    /// and it is the sharpest test of which estimators carry a baseline.
    ///
    /// MEASURED, `c = 20`, per-sample total variance:
    ///
    /// | estimator | c = 0 | c = 20 |
    /// |---|---|---|
    /// | `rebar_grad`, eta = 1 | 2.948 | 2.948 |
    /// | `rebar_grad`, eta = 0 (bare score function) | 5.22 | **749.4** |
    ///
    /// At `eta = 1` the constant cancels inside `f(s) - eta f(r(z~))` and the estimate is unmoved to
    /// ten digits; at `eta = 0` there is nothing to cancel against and the variance goes up 143x.
    /// Both stay unbiased, which is the other half: the control variate moved the variance and not
    /// the mean.
    #[test]
    fn a_constant_shift_is_invisible_to_the_gradient_and_enormous_to_a_bare_score_function() {
        let g = glass(8, 7);
        let th = logits(8, 11);
        let n = 100_000;
        let (cv1, cv0) = (ControlVariate::new(1.0, 0.5).unwrap(), ControlVariate::none());
        let mut var1 = [0.0f64; 2];
        let mut var0 = [0.0f64; 2];
        for (k, c) in [0.0f64, 20.0].into_iter().enumerate() {
            let f = Shifted { inner: Multilinear::new(&g), c };
            let exact = multilinear_grad(&th, &f).unwrap();
            let a = rebar_grad(&th, cv1, &f, n, 5).unwrap();
            let b = rebar_grad(&th, cv0, &f, n, 5).unwrap();
            assert!(worst_sigma(&a, &exact) < 5.0, "c={c}: REBAR eta=1 drifted");
            assert!(worst_sigma(&b, &exact) < 5.0, "c={c}: REBAR eta=0 drifted");
            var1[k] = a.total_variance();
            var0[k] = b.total_variance();
        }
        let ratio1 = var1[1] / var1[0];
        assert!(
            (ratio1 - 1.0).abs() < 1e-6,
            "the control variate should absorb a constant exactly; variance moved by {ratio1}"
        );
        let ratio0 = var0[1] / var0[0];
        assert!(
            ratio0 > 50.0,
            "a bare score function must pay for the constant; its variance moved by only {ratio0}"
        );
    }

    /// [`crate::program::Program::reinforce_grad`] returns the exact gradient SHRUNK BY
    /// `1 - 1/episodes`, and this module's estimator does not.
    ///
    /// Its baseline is the mean of the SAME batch it is subtracted from, so the baseline is
    /// correlated with each episode's own score: `E[mean * score_e] = (1/N) E[L_e score_e]`, which
    /// leaves `(1 - 1/N) * grad`. A leave-one-out baseline would not do this.
    ///
    /// MEASURED at B = 8 over 25 000 batches: every coordinate is within 1.9 standard errors of
    /// `(1 - 1/8) * exact` and as far as **21.4 standard errors** from `exact`. It is 12.5% at
    /// B = 8, 0.2% at B = 512, and it is why the comparison above is a variance comparison rather
    /// than a mean one. Recorded here rather than fixed: `program` is not this module.
    #[test]
    fn program_reinforce_is_the_exact_gradient_shrunk_by_one_over_the_batch() {
        let n = 8;
        let g = glass(n, 7);
        let f = Multilinear::new(&g);
        let th = logits(n, 11);
        let exact = multilinear_grad(&th, &f).unwrap();
        let gates: Vec<Gate> = (0..n).map(|i| Gate::PNot { bit: i, p_theta: i }).collect();
        let prog = Program { gates, graphs: Vec::new(), n_params: n };
        let init = State { bits: vec![-1i8; n], reals: Vec::new() };
        let loss = |s: &State| g.energy(&s.bits);

        let (b, reps) = (8usize, 25_000usize);
        let mut acc = Moments::new(n);
        for rep in 0..reps {
            let (r, _) = prog.reinforce_grad(&init, &th, &loss, b, 500_000 + rep as u64);
            acc.push(&r);
        }
        let var = acc.variance();
        let shrink = 1.0 - 1.0 / b as f64;
        let mut worst_raw = 0.0f64;
        for i in 0..n {
            let se = (var[i] / reps as f64).sqrt();
            let d_shrunk = (acc.mean[i] - exact[i] * shrink).abs() / se;
            assert!(
                d_shrunk < 4.0,
                "site {i}: {:.5} is {d_shrunk:.2} standard errors from (1-1/B) * exact",
                acc.mean[i]
            );
            worst_raw = worst_raw.max((acc.mean[i] - exact[i]).abs() / se);
        }
        assert!(
            worst_raw > 5.0,
            "the shrinkage must be RESOLVED, or this test records nothing: worst {worst_raw:.2}"
        );

        // The same total sample count, this module's estimator, and no shrinkage.
        let rb = rebar_grad(&th, ControlVariate::new(1.0, 0.5).unwrap(), &f, reps * b, 77).unwrap();
        let sig = worst_sigma(&rb, &exact);
        assert!(sig < 5.0, "REBAR should hit the unshrunk gradient: {sig:.2} sigma");
    }

    /// ORACLE: the logistic distribution's closed-form variance, `pi^2/3`.
    ///
    /// The whole binary Concrete rests on this noise being logistic — it is what makes
    /// `sign(theta + L)` a Bernoulli(`sigma(theta)`) draw — so the noise is checked against the
    /// distribution's own moments rather than against itself.
    #[test]
    fn the_logistic_noise_has_the_variance_the_closed_form_says() {
        let m = 400_000;
        let mut r = Pcg::new(2024, 9);
        let mut acc = Moments::new(1);
        for _ in 0..m {
            acc.push(&[logistic(&mut r)]);
        }
        let var = acc.variance()[0];
        let want = core::f64::consts::PI * core::f64::consts::PI / 3.0;
        // The sample variance of a logistic has sd = var * sqrt((2 + kurt)/m) with excess kurtosis
        // 1.2; four of those is the allowance.
        let tol = 4.0 * want * (3.2 / m as f64).sqrt();
        assert!((var - want).abs() < tol, "variance {var:.5}, closed form {want:.5}");
        let se = (var / m as f64).sqrt();
        assert!(acc.mean[0].abs() < 4.0 * se, "mean {:.5} is not zero", acc.mean[0]);
    }

    /// ORACLE: the softmax itself. The argmax of a Gumbel-perturbed score is a draw from
    /// `softmax(logits)` — the Gumbel-max trick — at EVERY temperature, because `tau` divides all
    /// the perturbed scores and so cannot reorder them.
    ///
    /// The two temperatures are the asymmetric part: a relaxation that had leaked its temperature
    /// into the argmax would match at one of them and not the other.
    #[test]
    fn the_categorical_relaxation_argmaxes_to_an_exact_softmax_draw() {
        let logits = [0.9f64, -0.4, 1.7, 0.1];
        let mx = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let ex: Vec<f64> = logits.iter().map(|l| (l - mx).exp()).collect();
        let total: f64 = ex.iter().sum();
        let m = 100_000;
        for tau in [0.1f64, 5.0] {
            let mut r = Pcg::new(31, 2);
            let mut counts = [0u32; 4];
            for _ in 0..m {
                let y = gumbel_softmax(&logits, tau, &mut r).unwrap();
                let s: f64 = y.iter().sum();
                assert!((s - 1.0).abs() < 1e-12, "tau={tau}: the draw sums to {s}");
                assert!(y.iter().all(|&v| v > 0.0 && v <= 1.0), "tau={tau}: left the simplex");
                if tau > 1.0 {
                    // Hot, so no class rounds to the corner; cold, and the winner does.
                    assert!(y.iter().all(|&v| v < 1.0), "tau={tau}: a hot draw hit the corner");
                }
                let mut best = 0;
                for k in 1..4 {
                    if y[k] > y[best] {
                        best = k;
                    }
                }
                counts[best] += 1;
            }
            for k in 0..4 {
                let p = ex[k] / total;
                let dev = (f64::from(counts[k]) - f64::from(m) * p).abs();
                let tol = 4.0 * (f64::from(m) * p * (1.0 - p)).sqrt();
                assert!(dev < tol, "tau={tau} class {k}: off by {dev:.0} draws (tol {tol:.0})");
            }
        }
    }

    /// The two-class Concrete and the binary Concrete are the same law by two different noises —
    /// two Gumbels in the first, their logistic difference in the second.
    ///
    /// Checked as a mean over independent streams, which is the only honest form: the routes cannot
    /// agree draw by draw because they consume different numbers of uniforms.
    #[test]
    fn the_two_class_concrete_is_the_binary_concrete() {
        let (theta, tau, m) = (0.7f64, 0.6f64, 200_000);
        let mut r1 = Pcg::new(77, 4);
        let mut a = Moments::new(1);
        for _ in 0..m {
            let y = gumbel_softmax(&[theta, 0.0], tau, &mut r1).unwrap();
            a.push(&[2.0 * y[0] - 1.0]);
        }
        let mut r2 = Pcg::new(78, 5);
        let mut b = Moments::new(1);
        for _ in 0..m {
            b.push(&[relaxed_spin(theta + logistic(&mut r2), tau)]);
        }
        let se = ((a.variance()[0] + b.variance()[0]) / m as f64).sqrt();
        let d = (a.mean[0] - b.mean[0]).abs();
        assert!(
            d < 4.0 * se,
            "categorical route {:.6}, binary route {:.6}, {:.2} standard errors apart",
            a.mean[0],
            b.mean[0],
            d / se
        );
    }

    /// Every refusal is reachable, names what it saw, and says something.
    #[test]
    fn an_input_the_estimator_cannot_understand_is_a_typed_error() {
        let g = glass(4, 1);
        let f = Multilinear::new(&g);
        let th = logits(4, 2);
        assert_eq!(
            gumbel_softmax_grad(&th, 0.0, &f, 10, 1).unwrap_err(),
            Refused::BadTemperature { tau: 0.0 }
        );
        assert_eq!(
            straight_through_grad(&th, Through::Concrete { tau: -1.0 }, &f, 10, 1).unwrap_err(),
            Refused::BadTemperature { tau: -1.0 }
        );
        assert_eq!(
            rebar_grad(&th, ControlVariate::none(), &f, 0, 1).unwrap_err(),
            Refused::NoSamples
        );
        assert_eq!(
            rebar_grad(&[0.1, 0.2], ControlVariate::none(), &f, 10, 1).unwrap_err(),
            Refused::Arity { logits: 2, objective: 4 }
        );
        assert_eq!(exact_grad(&[], &f).unwrap_err(), Refused::Empty);
        assert_eq!(
            exact_grad(&[f64::INFINITY, 0.0, 0.0, 0.0], &f).unwrap_err(),
            Refused::BadLogit { index: 0, theta: f64::INFINITY }
        );
        assert_eq!(
            ControlVariate::new(f64::INFINITY, 1.0).unwrap_err(),
            Refused::BadScale { eta: f64::INFINITY }
        );
        assert_eq!(
            ControlVariate::new(1.0, -1.0).unwrap_err(),
            Refused::BadTemperature { tau: -1.0 }
        );
        assert_eq!(gumbel_softmax(&[], 1.0, &mut Pcg::new(1, 1)).unwrap_err(), Refused::Empty);

        let big = glass(ENUMERATION_LIMIT + 1, 6);
        let fb = Multilinear::new(&big);
        let wide = vec![0.1; big.n];
        assert_eq!(
            exact_value(&wide, &fb).unwrap_err(),
            Refused::TooLargeToEnumerate { spins: ENUMERATION_LIMIT + 1, limit: ENUMERATION_LIMIT }
        );
        // ... and the closed form still answers at that size, which is what it is for.
        assert_eq!(multilinear_grad(&wide, &fb).unwrap().len(), big.n);

        for e in [
            Refused::BadTemperature { tau: 0.0 },
            Refused::BadLogit { index: 1, theta: f64::NAN },
            Refused::BadScale { eta: f64::NAN },
            Refused::Empty,
            Refused::NoSamples,
            Refused::Arity { logits: 1, objective: 2 },
            Refused::TooLargeToEnumerate { spins: 30, limit: 20 },
        ] {
            assert!(e.to_string().len() > 40, "{e:?} does not explain itself");
        }
    }

    /// Same seed, same numbers, every time — and different seeds actually differ, which is the half
    /// a determinism test usually forgets.
    #[test]
    fn the_estimators_are_deterministic_by_seed() {
        let g = glass(6, 2);
        let f = Multilinear::new(&g);
        let th = logits(6, 9);
        let cv = ControlVariate::new(0.8, 0.4).unwrap();
        let a = rebar_grad(&th, cv, &f, 2_000, 4).unwrap();
        let b = rebar_grad(&th, cv, &f, 2_000, 4).unwrap();
        let c = rebar_grad(&th, cv, &f, 2_000, 5).unwrap();
        assert_eq!(a.grad, b.grad);
        assert_eq!(a.value, b.value);
        assert_ne!(a.grad, c.grad);
        for tau in [0.3f64, 1.0] {
            let x = gumbel_softmax_grad(&th, tau, &f, 2_000, 4).unwrap();
            let y = gumbel_softmax_grad(&th, tau, &f, 2_000, 4).unwrap();
            assert_eq!(x.grad, y.grad);
        }
    }

    /// A one-sample estimate has a mean and no spread, and the module must say so rather than
    /// divide by `n - 1`.
    #[test]
    fn a_single_sample_has_no_error_bar_rather_than_an_infinite_one() {
        let g = glass(5, 3);
        let f = Multilinear::new(&g);
        let th = logits(5, 7);
        let e = rebar_grad(&th, ControlVariate::none(), &f, 1, 12).unwrap();
        assert_eq!(e.samples, 1);
        assert!(e.grad.iter().all(|v| v.is_finite()));
        assert!(e.stderr.iter().all(|&v| v == 0.0), "a lone sample reported a spread");
        assert_eq!(e.total_variance(), 0.0);
        assert_eq!(e.value_stderr, 0.0);
    }
}
