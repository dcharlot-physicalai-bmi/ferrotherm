//! Expectation propagation for the binary pairwise model, with Gaussian site approximations.
//!
//! Minka, *Expectation Propagation for Approximate Bayesian Inference*, UAI 2001, is the
//! algorithm: approximate every factor of an intractable product by one drawn from a tractable
//! exponential family, and fit each approximate factor in turn against the EXACT factor *in the
//! context of all the others*, by matching moments. Opper & Winther, *Expectation Consistent
//! Approximate Inference*, JMLR 6:2177 (2005), is the treatment of an Ising model that this
//! module implements — the version in which the tractable family is a Gaussian and the exact
//! factors are the spins' own two-point measures.
//!
//! # What is approximated, and what is kept exactly
//!
//! Write the model in this crate's convention, `E(s) = −Σ_{i<j} J_ij s_i s_j − Σ_i h_i s_i`, as a
//! product of a coupling factor and one measure per site:
//!
//! ```text
//!   p(s) ∝ exp(β(½ sᵀJ s + hᵀs)) · Π_i t_i(s_i),      t_i(s_i) = δ(s_i−1) + δ(s_i+1)
//! ```
//!
//! The coupling factor is quadratic, so it is already in the Gaussian family and is kept EXACTLY —
//! including the fields. What is approximated is the thing that makes the model discrete: each
//! site's two-point measure is replaced by a Gaussian site
//!
//! ```text
//!   t̃_i(s_i) ∝ exp(−½ λ_i s_i² + γ_i s_i)
//! ```
//!
//! and the product of the coupling factor with all the sites is a Gaussian `q` with precision
//! `A = diag(λ) − βJ` and mean `m = A⁻¹ b`, `b = βh + γ`.
//!
//! That is the whole difference from [`crate::meanfield`]. Naive mean field approximates `p` by a
//! product of independent spins and keeps nothing of the correlations; EP keeps the full `n × n`
//! covariance `A⁻¹` and approximates only the discreteness. The cost is the same `O(n³)` linear
//! algebra per sweep that the covariance implies, and the tests measure what it buys.
//!
//! # One EP sweep
//!
//! For each site, in the natural parameters `(precision, field)` of the Gaussian family:
//!
//! 1. **Cavity by division.** `q`'s marginal at `i` has variance `Σ_ii` and mean `m_i`, hence
//!    natural parameters `(1/Σ_ii, m_i/Σ_ii)`. Dividing out `t̃_i` SUBTRACTS its parameters:
//!    `λ_∖i = 1/Σ_ii − λ_i`, `γ_∖i = m_i/Σ_ii − γ_i`.
//! 2. **Moment matching against the exact site.** The tilted distribution `t_i(s) · exp(γ_∖i s −
//!    ½λ_∖i s²)` is, for a spin, a two-point law with `P(+1)/P(−1) = exp(2γ_∖i)` — the `s² = 1` of
//!    a spin cancels the quadratic term entirely. So the tilted mean is `tanh(γ_∖i)` and, because
//!    `⟨s²⟩ = 1` exactly, the tilted variance is `1 − tanh²(γ_∖i)`, with no approximation anywhere
//!    in either.
//! 3. **Update by division again.** The new site is the tilted marginal divided by the cavity:
//!    `λ_i ← 1/v̂ − λ_∖i`, `γ_i ← m̂/v̂ − γ_∖i`.
//! 4. **Damping**, in the natural parameters. Undamped EP has two ways to fail and damping fixes
//!    one of them: measured, a 4x4 torus at `β = 0.5` cycles forever at damping 0.5 and settles in
//!    16 sweeps at 0.8, while a frustrated 12-spin model at `β = 2.2` diverges at every damping
//!    tried up to 0.9999.
//!
//! # The cavity precision is NEGATIVE here, and that is the physics
//!
//! For any positive-definite `A`, `(A⁻¹)_ii ≥ 1/A_ii`, with equality exactly when row `i` has no
//! off-diagonal entry. Since `A_ii = λ_i`, the cavity precision `1/Σ_ii − λ_i` is `≤ 0` at every
//! site of a coupled model and `= 0` at an uncoupled one. An improper cavity would be fatal in
//! Gaussian-process EP, where the prior supplies its own precision; it is harmless here, because
//! the exact site measure is supported on two points and normalises the tilted law whatever the
//! sign of its quadratic term. The negative correction is the Onsager reaction in another
//! notation, and the `the_cavity_precision_is_never_positive` test asserts the inequality rather
//! than trusting this paragraph.
//!
//! # The free energy is NOT a bound
//!
//! [`Ep::log_z`] is the expectation-consistent free energy
//!
//! ```text
//!   ln Z_EP = ln Z_site + ln Z_gauss − ln Z_ref
//! ```
//!
//! — the stationary value of Opper & Winther's three-term functional, where `Z_site` is the
//! product of tilted normalisers, `Z_gauss` is the Gaussian `q`'s normaliser, and `Z_ref` removes
//! the Gaussian overlap that the other two both describe, so it is counted once. It is EXACT where
//! EP is exact (no couplings, or a Gaussian target — both are tests here) and an approximation
//! otherwise.
//!
//! It is **not a bound**, and this module therefore does not accumulate it through
//! [`crate::round`]. Directed rounding buys certainty about the last bit of a quantity whose
//! second digit is already an approximation, and writing `sum_down` around an estimate would
//! dress it as a proof. [`crate::meanfield::gibbs_bogoliubov`] is the bound in this crate, and it
//! is a bound because Jensen's inequality says so at every `m`; nothing of the kind is available
//! here, since the expectation-consistent free energy is a SADDLE point of its functional rather
//! than a maximum.
//!
//! What can honestly be said is what was measured. Against exhaustive enumeration over 66
//! converged runs — 24 sparse frustrated graphs at two betas, plus ferromagnetic and
//! antiferromagnetic rings, a 4x4 torus and a 4x3 grid at `β` from 0.1 to 0.8 — `ln Z_EP` came
//! out BELOW `ln Z` all 66 times, by between 1.6e-4 and 1.34. That is an observation about those
//! instances, not a theorem, and no test in this module asserts it as one.

use crate::graph::Graph;

/// One exact factor of the model, and therefore one thing to moment-match against.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Site {
    /// The Ising measure, `s_i ∈ {−1, +1}`. Every site of a [`Graph`] is one of these.
    Spin,
    /// A Gaussian factor `exp(−(s − mean)² / (2 · variance))` on a continuous `s`.
    ///
    /// Not a thing this crate's models contain, and present for a specific reason: with every site
    /// Gaussian the whole target is a multivariate Gaussian, EP's approximating family contains the
    /// truth, and EP must reproduce it EXACTLY — marginals and free energy alike. That is the only
    /// oracle for the cavity/moment-matching/free-energy algebra that does not involve running the
    /// algebra, which is why the variant exists and why
    /// the `ep_is_exact_on_a_gaussian_target` test checks it against a closed-form matrix
    /// inverse written out by hand.
    Gaussian {
        /// Centre of the factor.
        mean: f64,
        /// Width of the factor, strictly positive.
        variance: f64,
    },
}

/// How the iteration is run.
#[derive(Clone, Debug, PartialEq)]
pub struct Params {
    /// Sweeps before the run is refused as non-convergent.
    pub max_sweeps: usize,
    /// Converged when no natural parameter moved by more than this in a sweep.
    ///
    /// A threshold on `(λ, γ)` rather than on the marginals, deliberately: the marginals are a
    /// `tanh` of the parameters and saturate, so two visibly identical magnetisations can sit on
    /// parameters that are still travelling.
    ///
    /// Measured RELATIVE to each parameter's own size, `|Δx| / (1 + |x|)`, because an absolute
    /// threshold is not scale invariant and this crate has been bitten by that before. Site
    /// precisions reach 5.5e3 on a model with `βh = 5`, the cavity subtracts two quantities of
    /// that size, and the resulting noise floor is about `ε·λ` — measured, an absolute residual
    /// of 5.5e-9 that twenty thousand sweeps did not move. A tolerance below the floor does not
    /// make a run careful; it makes convergence unreachable.
    pub tol: f64,
    /// Fraction of the OLD natural parameters kept each sweep, in `[0, 1)`. `0.0` is undamped.
    pub damping: f64,
}

impl Default for Params {
    fn default() -> Self {
        Params { max_sweeps: 500, tol: 1e-11, damping: 0.5 }
    }
}

/// What an EP run produced, at its fixed point.
#[derive(Clone, Debug)]
pub struct Ep {
    /// Inverse temperature the run was at.
    pub beta: f64,
    /// Approximate marginals `⟨s_i⟩`, taken from the TILTED distribution at each site.
    ///
    /// The tilted mean and the Gaussian `q`'s mean coincide at a fixed point — that is what moment
    /// matching means — and the tilted one is reported because it is the one that cannot leave
    /// `(−1, 1)`: it is a `tanh`. `q`'s mean is a linear solve and would report a magnetisation
    /// past one on a model EP handles badly, which is a wrong answer that looks like a number.
    /// The `the_two_means_agree_at_the_fixed_point` test holds them against each other.
    pub m: Vec<f64>,
    /// Tilted variance at each site, `1 − m_i²` for a spin.
    pub variance: Vec<f64>,
    /// Diagonal of the Gaussian covariance `A⁻¹`, which equals [`Ep::variance`] at a fixed point.
    pub sigma: Vec<f64>,
    /// The Gaussian `q`'s own mean, `A⁻¹b`, computed by the Cholesky solve rather than from any
    /// cavity.
    ///
    /// Carried beside [`Ep::m`] because their difference is the only local diagnostic EP offers:
    /// moment matching makes them equal at a fixed point, so what is left is how far the reported
    /// answer is from one. It is not clamped to `(−1, 1)` and, on a model EP fits badly, it can
    /// leave that interval — which is the reason it is not what [`Ep::m`] reports.
    pub gaussian_mean: Vec<f64>,
    /// The expectation-consistent free energy as `ln Z`. **An approximation, not a bound** — see
    /// the module note.
    pub log_z: f64,
    /// Cavity precision per site; `≤ 0` always, and `0` exactly at an uncoupled site.
    pub cavity_precision: Vec<f64>,
    /// Cavity field per site. The marginals are its `tanh`, so this is the EP analogue of the
    /// local field in [`crate::meanfield`].
    pub cavity_field: Vec<f64>,
    /// Sweeps actually run.
    pub sweeps: usize,
    /// Largest natural-parameter move on the last sweep, which is below `Params::tol`.
    pub residual: f64,
}

/// Why an EP run was refused.
///
/// Every variant is a case where the iteration cannot honestly produce marginals; none of them is
/// recoverable by returning the last iterate, which would be a wrong answer wearing the shape of a
/// right one.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EpError {
    /// The sweeps ran out with the parameters still moving.
    ///
    /// EP is a fixed-point iteration with no descent guarantee. On a strongly coupled model it
    /// cycles; on a strongly frustrated one it diverges. More sweeps fix neither, and more damping
    /// fixes only the first: measured, a 4x4 ferromagnetic torus at `β = 0.5` cycles forever at
    /// damping 0.5 and converges in 16 sweeps at 0.8, while a frustrated 12-spin instance at
    /// `β = 2.2` was still refused after 100,000 sweeps at every damping from 0.6 to 0.9999.
    DidNotConverge {
        /// Sweeps run before giving up.
        sweeps: usize,
        /// Largest natural-parameter move on the last of them.
        residual: f64,
        /// The tolerance it had to beat.
        tol: f64,
    },
    /// The site precisions stopped making `A = diag(λ) − βJ` positive definite, so the Gaussian
    /// `q` does not exist and neither does its covariance.
    Indefinite {
        /// Sweep the factorisation failed on.
        sweep: usize,
        /// Row whose Cholesky pivot was not positive.
        site: usize,
        /// The pivot itself.
        pivot: f64,
    },
    /// A site's tilted distribution is not normalisable: its precision came out non-positive.
    ///
    /// Only a [`Site::Gaussian`] can produce one: its factor is itself a Gaussian, so it cannot
    /// rescue a negative cavity precision the way a two-point measure can.
    ///
    /// **As shipped, no call to [`run_with_sites`] reaches this.** A Gaussian site starts at its
    /// own precision `1/v` and the update returns `1/v` again at every sweep, so its tilted
    /// precision is identically `1/Σ_ii`, which is positive whenever the Gaussian exists at all.
    /// The variant is the guard on that division rather than a failure anyone has seen, and
    /// `an_improper_tilt_is_refused_rather_than_divided_by` exercises it at the division itself.
    /// It is reachable the moment a site is initialised anywhere other than its own parameters,
    /// which an earlier version of this module did — and it refused a three-site chain EP is
    /// supposed to solve exactly, with a tilted precision of -0.40.
    ImproperTilt {
        /// Sweep it happened on.
        sweep: usize,
        /// The site.
        site: usize,
        /// Tilted precision, which is not positive.
        precision: f64,
    },
    /// A site saturated: its tilted variance underflowed, so the Gaussian family cannot represent
    /// it and `1/v̂` is not a number.
    ///
    /// `tanh` reaches `1.0` in `f64` at a field of about 19.1, and a site pinned that hard has an
    /// exactly zero tilted variance. The honest report is that the approximating family ran out,
    /// not a precision of infinity propagated into a matrix.
    Saturated {
        /// Sweep it happened on.
        sweep: usize,
        /// The site.
        site: usize,
        /// Cavity field that saturated it.
        field: f64,
    },
}

impl core::fmt::Display for EpError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match *self {
            EpError::DidNotConverge { sweeps, residual, tol } => write!(
                f,
                "expectation propagation did not converge in {sweeps} sweeps: the largest natural \
                 parameter still moved by {residual:e} against a tolerance of {tol:e}. EP has no \
                 descent guarantee and oscillates on strongly coupled models; raise the damping \
                 before raising the sweep count"
            ),
            EpError::Indefinite { sweep, site, pivot } => write!(
                f,
                "on sweep {sweep} the site precisions left A = diag(lambda) - beta*J indefinite: \
                 row {site} has Cholesky pivot {pivot:e}. The Gaussian approximation does not \
                 exist, so neither does the covariance the cavity is built from"
            ),
            EpError::ImproperTilt { sweep, site, precision } => write!(
                f,
                "on sweep {sweep} the tilted distribution at site {site} had precision \
                 {precision:e} and cannot be normalised; a Gaussian site cannot absorb a cavity \
                 this improper"
            ),
            EpError::Saturated { sweep, site, field } => write!(
                f,
                "on sweep {sweep} site {site} saturated at a cavity field of {field}: its tilted \
                 variance underflowed to zero, and a Gaussian site of infinite precision is not a \
                 member of the family"
            ),
        }
    }
}

impl core::error::Error for EpError {}

/// `ln(2 cosh x)`, without overflowing at `|x| > 710`.
///
/// `2cosh(x) = e^{|x|}(1 + e^{−2|x|})`, so the log is `|x| + ln1p(e^{−2|x|})` and the exponential
/// that could overflow never happens. The naive `(2.0 * x.cosh()).ln()` is `inf` from `|x| = 711`,
/// and EP reaches fields that large on a model with strong couplings.
fn log_two_cosh(x: f64) -> f64 {
    let a = x.abs();
    a + (-2.0 * a).exp().ln_1p()
}

/// Moments of a tilted distribution: an exact site against a Gaussian cavity.
struct Tilt {
    mean: f64,
    var: f64,
    log_z: f64,
}

impl Site {
    /// The mean, variance and log-normaliser of `site(s) · exp(field·s − ½·precision·s²)`.
    ///
    /// `Err` carries the offending tilted precision when that product is not normalisable. A
    /// [`Site::Spin`] never is — its measure lives on two points, so the quadratic term is a
    /// constant and the sign of `precision` cannot matter — and a [`Site::Gaussian`] is whenever
    /// its own precision fails to outweigh a negative cavity.
    fn tilt(self, precision: f64, field: f64) -> Result<Tilt, f64> {
        match self {
            Site::Spin => {
                // s² = 1 on the support, so the quadratic term is the constant exp(−precision/2)
                // and leaves the SHAPE alone: the tilted law is two points with log-odds 2·field.
                // Both moments are therefore exact, not approximations of an integral.
                let mean = field.tanh();
                Ok(Tilt {
                    mean,
                    var: 1.0 - mean * mean,
                    log_z: log_two_cosh(field) - 0.5 * precision,
                })
            }
            Site::Gaussian { mean: mu, variance: v } => {
                let p = precision + 1.0 / v;
                if !(p > 0.0) {
                    return Err(p);
                }
                let q = field + mu / v;
                Ok(Tilt {
                    mean: q / p,
                    var: 1.0 / p,
                    log_z: 0.5 * (core::f64::consts::TAU / p).ln() + q * q / (2.0 * p)
                        - mu * mu / (2.0 * v),
                })
            }
        }
    }
}

/// Site `i`'s cavity, in natural parameters: the Gaussian `q` with its own site divided out.
///
/// `q`'s marginal at `i` is `(1/sigma, mean/sigma)` in natural form, and division in an
/// exponential family is SUBTRACTION of natural parameters — so this one line is what makes the
/// algorithm expectation propagation rather than a fixed-point iteration on marginals. Written
/// once and called from both the sweep and the final assembly, so the two cannot drift apart.
///
/// The returned precision is `<= 0` at every coupled site; see the module note on why that is the
/// Onsager reaction rather than a defect.
fn cavity(sigma: f64, mean: f64, lam: f64, gam: f64) -> (f64, f64) {
    (1.0 / sigma - lam, mean / sigma - gam)
}

/// The Gaussian `q`'s diagonal covariance, mean, and the two scalars its normaliser needs.
struct Moments {
    /// `diag(A⁻¹)`.
    sigma: Vec<f64>,
    /// `A⁻¹ b`.
    mean: Vec<f64>,
    /// `Σ_i ln L_ii`, which is `½ ln det A`.
    half_log_det: f64,
    /// `bᵀ A⁻¹ b`.
    quad: f64,
}

/// Solve the Gaussian at site precisions `lam` and site fields `gam`.
///
/// `Err(i)` carries the row whose Cholesky pivot was not positive, together with the pivot.
fn gaussian_moments(
    g: &Graph,
    beta: f64,
    lam: &[f64],
    gam: &[f64],
) -> Result<Moments, (usize, f64)> {
    let n = g.n;
    let mut a = vec![0.0f64; n * n];
    for i in 0..n {
        a[i * n + i] = lam[i];
        for e in g.offset[i]..g.offset[i + 1] {
            // Each undirected edge appears in both CSR rows, so both triangles are filled.
            a[i * n + g.nbr[e] as usize] = -beta * g.w[e];
        }
    }
    // Cholesky, lower triangular, into `l`.
    let mut l = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[i * n + j];
            for k in 0..j {
                s -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                if !(s > 0.0) {
                    return Err((i, s));
                }
                l[i * n + i] = s.sqrt();
            } else {
                l[i * n + j] = s / l[j * n + j];
            }
        }
    }
    // Inverse of L, also lower triangular. diag(A⁻¹) = diag(L⁻ᵀ L⁻¹) is then a column norm, which
    // is the whole reason the inverse is formed: the alternative is n right-hand sides.
    let mut li = vec![0.0f64; n * n];
    for j in 0..n {
        li[j * n + j] = 1.0 / l[j * n + j];
        for i in (j + 1)..n {
            let mut s = 0.0;
            for k in j..i {
                s += l[i * n + k] * li[k * n + j];
            }
            li[i * n + j] = -s / l[i * n + i];
        }
    }
    let mut sigma = vec![0.0f64; n];
    for i in 0..n {
        let mut s = 0.0;
        for k in i..n {
            s += li[k * n + i] * li[k * n + i];
        }
        sigma[i] = s;
    }
    // b = βh + γ, then m = A⁻¹ b by forward and back substitution.
    let b: Vec<f64> = (0..n).map(|i| beta * g.h[i] + gam[i]).collect();
    let mut y = vec![0.0f64; n];
    for i in 0..n {
        let mut s = b[i];
        for k in 0..i {
            s -= l[i * n + k] * y[k];
        }
        y[i] = s / l[i * n + i];
    }
    let mut mean = vec![0.0f64; n];
    for i in (0..n).rev() {
        let mut s = y[i];
        for k in (i + 1)..n {
            s -= l[k * n + i] * mean[k];
        }
        mean[i] = s / l[i * n + i];
    }
    let half_log_det: f64 = (0..n).map(|i| l[i * n + i].ln()).sum();
    let quad: f64 = (0..n).map(|i| b[i] * mean[i]).sum();
    Ok(Moments { sigma, mean, half_log_det, quad })
}

/// Run EP with the Ising measure at every site — the ordinary call.
///
/// # Errors
///
/// [`EpError::DidNotConverge`] when the sweeps run out, [`EpError::Indefinite`] when the site
/// precisions stop admitting a Gaussian, and [`EpError::Saturated`] when a site pins so hard its
/// tilted variance underflows. [`EpError::ImproperTilt`] cannot occur for spins.
///
/// # Panics
///
/// If `damping` is outside `[0, 1)` or `beta` is not finite.
pub fn run(g: &Graph, beta: f64, p: &Params) -> Result<Ep, EpError> {
    let sites = vec![Site::Spin; g.n];
    run_with_sites(g, beta, &sites, p)
}

/// Run EP with an arbitrary exact factor at each site.
///
/// See [`Site::Gaussian`] for why a non-spin site exists at all.
///
/// # Errors
///
/// As [`run`], plus [`EpError::ImproperTilt`] where a Gaussian site meets a cavity it cannot
/// normalise.
///
/// # Panics
///
/// If `sites` is not one per node, if `damping` is outside `[0, 1)`, if `beta` is not finite, or
/// if a [`Site::Gaussian`] has a non-positive variance.
pub fn run_with_sites(g: &Graph, beta: f64, sites: &[Site], p: &Params) -> Result<Ep, EpError> {
    assert_eq!(sites.len(), g.n, "one site factor per node");
    assert!((0.0..1.0).contains(&p.damping), "damping must be in [0, 1); got {}", p.damping);
    assert!(beta.is_finite(), "beta must be finite; got {beta}");
    for (i, s) in sites.iter().enumerate() {
        if let Site::Gaussian { variance, .. } = *s {
            assert!(variance > 0.0, "site {i} has variance {variance}, which is not a width");
        }
    }
    let n = g.n;
    // Each site starts at its own natural parameters, and a spin gets a diagonal margin on top.
    //
    // A spin's bare measure has mean 0 and second moment 1, so its moment-matched Gaussian is
    // precision 1 and field 0. Adding the absolute row sum makes `diag(λ) − βJ` strictly
    // diagonally dominant, so Gershgorin puts every eigenvalue above zero and sweep 0 cannot fail
    // for want of a starting guess.
    //
    // A GAUSSIAN SITE GETS NO SUCH MARGIN, and that is not an oversight. Its precision `1/v` is
    // fixed by the target rather than chosen by us, and inflating it makes the cavity precision
    // `1/Σ_ii − λ_i` so negative that the site's own `1/v` can no longer outweigh it — measured,
    // a tilted precision of −0.40 on the three-site chain in the tests, which is an
    // `ImproperTilt` refusal on sweep 0 of a target EP is supposed to solve exactly. Starting a
    // Gaussian site at its own parameters starts it at the fixed point, which is the right answer
    // to "where does this site end up" when the approximating family contains the truth.
    let mut lam: Vec<f64> = (0..n)
        .map(|i| match sites[i] {
            Site::Spin => {
                let row: f64 =
                    (g.offset[i]..g.offset[i + 1]).map(|e| (beta * g.w[e]).abs()).sum();
                1.0 + row
            }
            Site::Gaussian { variance, .. } => 1.0 / variance,
        })
        .collect();
    let mut gam: Vec<f64> = sites
        .iter()
        .map(|s| match *s {
            Site::Spin => 0.0,
            Site::Gaussian { mean, variance } => mean / variance,
        })
        .collect();

    let mut residual = f64::INFINITY;
    let mut swept = 0usize;
    for sweep in 0..p.max_sweeps {
        let mom = gaussian_moments(g, beta, &lam, &gam)
            .map_err(|(site, pivot)| EpError::Indefinite { sweep, site, pivot })?;
        residual = 0.0;
        for i in 0..n {
            let (prec_cav, field_cav) = cavity(mom.sigma[i], mom.mean[i], lam[i], gam[i]);
            let t = sites[i]
                .tilt(prec_cav, field_cav)
                .map_err(|precision| EpError::ImproperTilt { sweep, site: i, precision })?;
            if !(t.var > 0.0) || !t.var.is_finite() {
                return Err(EpError::Saturated { sweep, site: i, field: field_cav });
            }
            let fresh_lam = 1.0 / t.var - prec_cav;
            let fresh_gam = t.mean / t.var - field_cav;
            let next_lam = p.damping * lam[i] + (1.0 - p.damping) * fresh_lam;
            let next_gam = p.damping * gam[i] + (1.0 - p.damping) * fresh_gam;
            // RELATIVE to each parameter's own size, with a floor of one.
            //
            // An absolute threshold is not reachable here and that is arithmetic, not impatience.
            // Site precisions run to thousands on a strongly biased model — 5.5e3 at `βh = 5` —
            // and the cavity is recovered by SUBTRACTING two quantities of that size, so the
            // sweep-to-sweep noise floor is `ε·λ`, about 1e-12 relative. Measured: that model sat
            // at an absolute residual of 5.5e-9 after twenty thousand sweeps, moving nowhere. An
            // absolute tolerance below the floor does not make a run careful, it makes it
            // impossible — the same lesson `linalg::jacobi_eig` records about thresholds that do
            // not scale with their matrix.
            let scale = |x: f64, y: f64| (x - y).abs() / (1.0 + y.abs().max(x.abs()));
            residual = residual.max(scale(next_lam, lam[i])).max(scale(next_gam, gam[i]));
            lam[i] = next_lam;
            gam[i] = next_gam;
        }
        swept = sweep + 1;
        if residual < p.tol {
            return finish(g, beta, sites, &lam, &gam, swept, residual);
        }
    }
    Err(EpError::DidNotConverge { sweeps: swept, residual, tol: p.tol })
}

/// Assemble the answer at the converged parameters.
///
/// Solved once more rather than reusing the last sweep's moments, so every reported quantity —
/// marginals, cavity, covariance, free energy — belongs to the SAME `(λ, γ)`. Reusing the previous
/// solve would report a cavity taken from parameters the run no longer holds, and the
/// moment-consistency test would then be checking two things that agree by construction.
fn finish(
    g: &Graph,
    beta: f64,
    sites: &[Site],
    lam: &[f64],
    gam: &[f64],
    sweeps: usize,
    residual: f64,
) -> Result<Ep, EpError> {
    let n = g.n;
    let mom = gaussian_moments(g, beta, lam, gam)
        .map_err(|(site, pivot)| EpError::Indefinite { sweep: sweeps, site, pivot })?;
    let mut m = vec![0.0f64; n];
    let mut variance = vec![0.0f64; n];
    let mut cavity_precision = vec![0.0f64; n];
    let mut cavity_field = vec![0.0f64; n];
    // ln Z_site: the tilted normalisers, one per site.
    let mut log_z_site = 0.0f64;
    for i in 0..n {
        let (prec_cav, field_cav) = cavity(mom.sigma[i], mom.mean[i], lam[i], gam[i]);
        let t = sites[i]
            .tilt(prec_cav, field_cav)
            .map_err(|precision| EpError::ImproperTilt { sweep: sweeps, site: i, precision })?;
        m[i] = t.mean;
        variance[i] = t.var;
        cavity_precision[i] = prec_cav;
        cavity_field[i] = field_cav;
        log_z_site += t.log_z;
    }
    // ln Z_gauss − ln Z_ref. Both carry (n/2)ln(2π) and it cancels, which is why neither appears:
    //   ln Z_gauss = (n/2)ln(2π) − ½ln det A + ½ bᵀA⁻¹b
    //   ln Z_ref   = Σ_i [ (1/2)ln(2π Σ_ii) + m_i²/(2 Σ_ii) ]
    // Z_ref is the Gaussian that the site term and the coupling term BOTH already describe, and
    // subtracting it once is what stops the overlap being counted twice.
    let mut log_z = log_z_site - mom.half_log_det + 0.5 * mom.quad;
    for i in 0..n {
        log_z -= 0.5 * mom.sigma[i].ln() + mom.mean[i] * mom.mean[i] / (2.0 * mom.sigma[i]);
    }
    Ok(Ep {
        beta,
        m,
        variance,
        sigma: mom.sigma,
        gaussian_mean: mom.mean,
        log_z,
        cavity_precision,
        cavity_field,
        sweeps,
        residual,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::GraphBuilder;
    use crate::rng::Pcg;

    /// Exact marginals and `ln Z` by visiting all `2^n` states, straight off [`Graph::energy`].
    ///
    /// Deliberately not [`crate::exact::Elimination`] or [`crate::free_energy::exact_log_z`]: this
    /// is six lines, it shares nothing with the module under test, and a reader can check it
    /// against the definition of a Boltzmann average without leaving the file.
    fn enumerate(g: &Graph, beta: f64) -> (Vec<f64>, f64) {
        assert!(g.n <= 20);
        let mut num = vec![0.0f64; g.n];
        let mut z = 0.0f64;
        let mut s = vec![-1i8; g.n];
        for mask in 0..(1usize << g.n) {
            for b in 0..g.n {
                s[b] = if mask >> b & 1 == 1 { 1 } else { -1 };
            }
            let w = (-beta * g.energy(&s)).exp();
            z += w;
            for i in 0..g.n {
                num[i] += w * f64::from(s[i]);
            }
        }
        (num.iter().map(|x| x / z).collect(), z.ln())
    }

    /// A sparse frustrated instance: random +-1 couplings on a random graph, small fields.
    fn frustrated(n: usize, edges: usize, seed: u64) -> Graph {
        let mut rng = Pcg::new(seed, 0x1D);
        let mut gb = GraphBuilder::new(n);
        for i in 1..n {
            let parent = (rng.f64() * i as f64) as usize;
            gb.couple(parent, i, if rng.f64() < 0.5 { -1.0 } else { 1.0 });
        }
        for _ in 0..edges.saturating_sub(n - 1) {
            let a = (rng.f64() * n as f64) as usize;
            let b = (rng.f64() * n as f64) as usize;
            if a != b {
                gb.couple(a, b, if rng.f64() < 0.5 { -1.0 } else { 1.0 });
            }
        }
        for i in 0..n {
            gb.bias(i, 0.4 * (2.0 * rng.f64() - 1.0));
        }
        gb.build()
    }

    /// ORACLE: the closed form `tanh(β h_i)`, and `ln Z = Σ ln 2cosh(β h_i)`.
    ///
    /// With no couplings the Gaussian family contains the truth exactly — every site's tilted law
    /// IS its marginal — so this is an identity rather than an approximation, and the tolerances
    /// are as tight as the cancellation inside the cavity allows. That cancellation is real and
    /// grows with `β|h|`: the cavity field is recovered as `m_i/Σ_ii − γ_i`, and both terms are
    /// about `m/(1−m²)`, which is 256 at `β|h| = 3.45`. Measured worst over the three betas below
    /// — marginals 2.2e-16, `ln Z` 2.8e-14, cavity precision 5.7e-14, cavity field 1.6e-14 — and
    /// each assertion sits about one order above its measurement, not four.
    #[test]
    fn ep_is_exact_on_an_uncoupled_model_oracle_tanh_closed_form() {
        let mut rng = Pcg::new(7, 0);
        for beta in [0.4f64, 1.0, 2.3] {
            let n = 9;
            let mut gb = GraphBuilder::new(n);
            for i in 0..n {
                gb.bias(i, 3.0 * rng.f64() - 1.5);
            }
            let g = gb.build();
            let out = run(&g, beta, &Params { damping: 0.0, ..Params::default() }).unwrap();
            let mut worst = 0.0f64;
            for i in 0..n {
                worst = worst.max((out.m[i] - (beta * g.h[i]).tanh()).abs());
            }
            assert!(worst < 1e-15, "beta {beta}: marginal off the closed form by {worst:e}");
            let truth: f64 = (0..n).map(|i| (2.0 * (beta * g.h[i]).cosh()).ln()).sum();
            assert!(
                (out.log_z - truth).abs() < 1e-13,
                "beta {beta}: ln Z {} vs closed form {truth}",
                out.log_z
            );
            // And the uncoupled cavity is the whole field, with no reaction term to remove.
            for i in 0..n {
                assert!(
                    out.cavity_precision[i].abs() < 1e-12,
                    "site {i}: an uncoupled cavity must be flat, not {:e}",
                    out.cavity_precision[i]
                );
                assert!((out.cavity_field[i] - beta * g.h[i]).abs() < 1e-13, "site {i}");
            }
        }
    }

    /// ORACLE: a hand-written `3 x 3` matrix inverse, by Cramer's rule, in this test.
    ///
    /// With Gaussian sites the target is a genuine multivariate Gaussian, EP's family contains it,
    /// and EP must return the truth: the exact marginals of `N(A⁻¹b, A⁻¹)` and the exact log
    /// normaliser. The oracle shares no code with the module — the cofactors below are written out
    /// for the tridiagonal `A` and can be checked by hand — so this pins the cavity algebra, the
    /// moment matching AND all three terms of the free energy at once. It is the only test here
    /// that can do that, because it is the only target EP is supposed to nail.
    #[test]
    fn ep_is_exact_on_a_gaussian_target() {
        for (j1, j2, beta) in [(0.4f64, -0.3f64, 1.0f64), (0.9, 0.7, 0.5), (-0.5, 0.2, 1.7)] {
            let mut gb = GraphBuilder::new(3);
            gb.couple(0, 1, j1);
            gb.couple(1, 2, j2);
            gb.bias(0, 0.3);
            gb.bias(1, -0.6);
            gb.bias(2, 0.15);
            let g = gb.build();
            let sites = [
                Site::Gaussian { mean: 0.5, variance: 0.8 },
                Site::Gaussian { mean: -0.2, variance: 1.3 },
                Site::Gaussian { mean: 0.9, variance: 0.45 },
            ];
            let out =
                run_with_sites(&g, beta, &sites, &Params { damping: 0.0, ..Params::default() })
                    .unwrap();

            // A = diag(1/v) − βJ for this chain, and b = βh + μ/v.
            let (v0, v1, v2) = (0.8f64, 1.3f64, 0.45f64);
            let (mu0, mu1, mu2) = (0.5f64, -0.2f64, 0.9f64);
            let (a1, a2, a3) = (1.0 / v0, 1.0 / v1, 1.0 / v2);
            let (c1, c2) = (beta * j1, beta * j2);
            let b = [
                beta * 0.3 + mu0 / v0,
                beta * -0.6 + mu1 / v1,
                beta * 0.15 + mu2 / v2,
            ];
            let det = a1 * a2 * a3 - a1 * c2 * c2 - c1 * c1 * a3;
            // adj(A), symmetric tridiagonal, by cofactors.
            let (adj11, adj12, adj13) = (a2 * a3 - c2 * c2, c1 * a3, c1 * c2);
            let (adj22, adj23, adj33) = (a1 * a3, a1 * c2, a1 * a2 - c1 * c1);
            let want = [
                (adj11 * b[0] + adj12 * b[1] + adj13 * b[2]) / det,
                (adj12 * b[0] + adj22 * b[1] + adj23 * b[2]) / det,
                (adj13 * b[0] + adj23 * b[1] + adj33 * b[2]) / det,
            ];
            for i in 0..3 {
                assert!(
                    (out.m[i] - want[i]).abs() < 1e-13,
                    "j=({j1},{j2}) beta={beta} site {i}: EP {} vs closed form {}",
                    out.m[i],
                    want[i]
                );
            }
            let want_var = [adj11 / det, adj22 / det, adj33 / det];
            for i in 0..3 {
                assert!(
                    (out.variance[i] - want_var[i]).abs() < 1e-13,
                    "site {i}: EP variance {} vs closed form {}",
                    out.variance[i],
                    want_var[i]
                );
            }
            // ln Z of the Gaussian target, written from its definition:
            //   Z = ∫ Π exp(−(s−μ)²/2v) · exp(β(½sᵀJs + hᵀs)) ds
            let quad: f64 = (0..3).map(|i| b[i] * want[i]).sum();
            let offset = mu0 * mu0 / (2.0 * v0) + mu1 * mu1 / (2.0 * v1) + mu2 * mu2 / (2.0 * v2);
            let want_lz =
                1.5 * core::f64::consts::TAU.ln() - 0.5 * det.ln() + 0.5 * quad - offset;
            assert!(
                (out.log_z - want_lz).abs() < 1e-12,
                "j=({j1},{j2}) beta={beta}: EP ln Z {} vs closed form {want_lz}",
                out.log_z
            );
        }
    }

    /// ORACLE: exhaustive enumeration of all `2^12` states — and EP must BEAT naive mean field on
    /// the same instance, at the same beta, on both marginals and `ln Z`.
    ///
    /// The comparison is the point. An implementation that quietly degenerated into mean field
    /// would still pass "close to the truth" at any plausible tolerance; it cannot pass "strictly
    /// closer than mean field on every one of these instances". Measured, worst marginal error
    /// over the six seeds: EP 0.0061 to 0.0148, mean field 0.25 to 0.82 — a factor of 32 to 97.
    /// In `ln Z`: EP 0.026 to 0.082, mean field 0.86 to 1.23.
    #[test]
    fn ep_beats_naive_mean_field_oracle_exhaustive_enumeration() {
        let beta = 0.35;
        for seed in 0..6u64 {
            let g = frustrated(12, 20, seed);
            let (marg, log_z) = enumerate(&g, beta);
            let out = run(&g, beta, &Params::default()).unwrap();
            let mf = crate::meanfield::naive_mean_field(&g, beta, 5000, 0.5);
            assert!(mf.converged(1e-10), "seed {seed}: the control must converge too");

            let err = |m: &[f64]| {
                (0..g.n).map(|i| (m[i] - marg[i]).abs()).fold(0.0f64, f64::max)
            };
            let (e_ep, e_mf) = (err(&out.m), err(&mf.m));
            assert!(
                e_ep < e_mf,
                "seed {seed}: EP marginal error {e_ep:e} is not below mean field's {e_mf:e}"
            );
            let (z_ep, z_mf) = ((out.log_z - log_z).abs(), (mf.log_z - log_z).abs());
            assert!(
                z_ep < z_mf,
                "seed {seed}: EP ln Z error {z_ep:e} is not below mean field's {z_mf:e}"
            );
            // Absolute, so a regression that degrades BOTH cannot hide behind the comparison.
            assert!(e_ep < 0.1, "seed {seed}: EP marginal error {e_ep:e} is too large outright");
        }
    }

    /// EP failing to converge must be a typed error, not a wrong answer.
    ///
    /// The instance is a 4x4 ferromagnetic torus at `β = 0.5`, past Onsager's `β_c = 0.4407` for
    /// `J = 1`: below the transition the model is two states and one unimodal Gaussian cannot
    /// straddle them, so lightly damped EP cycles rather than settling. Twenty thousand sweeps
    /// leave the relative residual at 8.7e-2, so this is a limit cycle and not a budget that ran
    /// out — which is why the cap here is small and the assertion is on the RESIDUAL rather than
    /// on the sweep count alone.
    ///
    /// The second half is what keeps the error message honest: the same instance at the same beta
    /// converges in 16 sweeps at damping 0.8. This refusal is a property of the iteration, not of
    /// the model.
    #[test]
    fn a_run_that_does_not_converge_is_a_typed_error() {
        let g = crate::ising::lattice2d(4, 1.0);
        let p = Params { max_sweeps: 2000, tol: 1e-11, damping: 0.5 };
        match run(&g, 0.5, &p) {
            Err(EpError::DidNotConverge { sweeps, residual, tol }) => {
                assert_eq!(sweeps, 2000, "it must spend its whole budget before refusing");
                assert!(
                    residual > 1e-3,
                    "residual {residual:e} is small, so this instance is merely slow and the test \
                     is no longer about a limit cycle"
                );
                assert!(residual > tol);
                let text = EpError::DidNotConverge { sweeps, residual, tol }.to_string();
                assert!(text.contains("did not converge"), "{text}");
            }
            other => panic!("expected a non-convergence error, got {other:?}"),
        }
        let damped = run(&g, 0.5, &Params { max_sweeps: 2000, tol: 1e-11, damping: 0.8 })
            .expect("damping is the documented remedy for a cycle");
        assert!(damped.sweeps < 100, "and it is a fast remedy: {} sweeps", damped.sweeps);

        // The remedy has a limit, which the message also says. A frustrated instance at beta 2.2
        // is refused at every damping tried: measured over 100,000 sweeps, a relative residual of
        // 1.3e-3 at damping 0.6, 3.5e-5 at 0.95, 1.2e-7 at 0.999 and 4.1e-5 at 0.9999. Note that
        // it does not FALSELY converge as its parameters blow up, which is the failure a relative
        // residual has to be checked for.
        for damping in [0.6f64, 0.95] {
            let hard = frustrated(12, 24, 3);
            let out = run(&hard, 2.2, &Params { max_sweeps: 500, tol: 1e-11, damping });
            assert!(
                matches!(out, Err(EpError::DidNotConverge { .. })),
                "damping {damping}: expected divergence, got {out:?}"
            );
        }
        // That same frustrated instance at beta 0.5 converges, so the refusal is about the regime.
        assert!(run(&frustrated(12, 24, 3), 0.5, &Params::default()).is_ok());
    }

    /// A model whose site precisions stop admitting a Gaussian is refused, not factorised anyway.
    ///
    /// The same torus one step colder. A Cholesky that runs past a non-positive pivot returns
    /// `NaN`s, and `NaN` compares false against every tolerance a downstream test could apply, so
    /// the pivot check is the difference between a refusal and a silent pass.
    #[test]
    fn a_gaussian_that_does_not_exist_is_refused_at_the_pivot() {
        let g = crate::ising::lattice2d(4, 1.0);
        match run(&g, 0.6, &Params { max_sweeps: 2000, tol: 1e-11, damping: 0.5 }) {
            Err(EpError::Indefinite { site, pivot, .. }) => {
                assert!(site < g.n);
                assert!(pivot <= 0.0, "an indefinite report needs a non-positive pivot: {pivot}");
            }
            other => panic!("expected an indefinite report, got {other:?}"),
        }
    }

    /// A site the Gaussian family cannot hold is a typed error, not an infinite precision.
    ///
    /// `tanh` reaches exactly `1.0` in `f64` near 19.1, at which point the tilted variance is
    /// exactly zero and its reciprocal is `inf`. Propagated into the matrix that gives a `NaN`
    /// covariance and a `NaN` free energy — a number-shaped answer to a question the family
    /// cannot answer.
    #[test]
    fn a_saturated_site_is_a_typed_error() {
        let mut gb = GraphBuilder::new(2);
        gb.couple(0, 1, 0.1);
        gb.bias(0, 25.0);
        gb.bias(1, 25.0);
        match run(&gb.build(), 1.0, &Params::default()) {
            Err(EpError::Saturated { site, field, .. }) => {
                assert!(site < 2);
                assert_eq!(field.tanh(), 1.0, "the field must really saturate f64 tanh");
            }
            other => panic!("expected a saturation report, got {other:?}"),
        }
        // The same shape at a field the family CAN hold is fine, so this is about saturation and
        // not about strong fields. It needs the sweeps: at h = 5 the site precisions reach 5.5e3,
        // and the convergence threshold is ABSOLUTE (see `Params::tol`), so this model has to be
        // iterated relatively much harder than a soft one before it is accepted.
        let mut gb = GraphBuilder::new(2);
        gb.couple(0, 1, 0.1);
        gb.bias(0, 5.0);
        gb.bias(1, 5.0);
        let mild = run(&gb.build(), 1.0, &Params { max_sweeps: 20_000, ..Params::default() });
        assert!(mild.is_ok(), "a field of 5 is inside the family: {mild:?}");
    }

    /// The division that builds a tilted distribution refuses a non-positive precision.
    ///
    /// Asserted at the division rather than through a run, because no run reaches it — see
    /// [`EpError::ImproperTilt`]. A spin is shown to be immune in the same test, which is the
    /// asymmetry the whole module rests on: a two-point measure normalises any quadratic tilt and
    /// a Gaussian one does not.
    #[test]
    fn an_improper_tilt_is_refused_rather_than_divided_by() {
        let gauss = Site::Gaussian { mean: 0.4, variance: 1.0 };
        assert_eq!(gauss.tilt(-4.0, 0.3).err(), Some(-3.0), "precision -4 + 1/v is -3");
        assert_eq!(gauss.tilt(-1.0, 0.3).err(), Some(0.0), "exactly zero is not normalisable");
        let ok = gauss.tilt(-0.75, 0.3).expect("0.25 is a precision");
        assert!((ok.var - 4.0).abs() < 1e-15);
        // A spin takes the same improper cavity without complaint, and its moments do not move
        // with the quadratic term at all: s^2 = 1 on its support.
        let want = 0.3f64.tanh();
        for precision in [-4.0f64, -1.0, 0.0, 3.0] {
            let t = Site::Spin.tilt(precision, 0.3).expect("a spin normalises it");
            assert_eq!(t.mean, want, "precision {precision} moved a spin's mean");
            assert_eq!(t.var, 1.0 - want * want);
        }
    }

    /// For positive-definite `A`, `(A⁻¹)_ii ≥ 1/A_ii` — so the cavity precision is never positive,
    /// and is zero exactly where the site has no couplings.
    ///
    /// A linear-algebra fact, independent of EP, and the one that makes an "improper" cavity the
    /// normal case here rather than a bug to be clamped away.
    #[test]
    fn the_cavity_precision_is_never_positive() {
        let g = frustrated(10, 16, 11);
        let out = run(&g, 0.3, &Params::default()).unwrap();
        for i in 0..g.n {
            assert!(
                out.cavity_precision[i] <= 0.0,
                "site {i}: cavity precision {} is positive",
                out.cavity_precision[i]
            );
            assert!(out.cavity_precision[i] < -1e-12, "site {i} is coupled, so it must be strict");
        }
        // An isolated node in an otherwise coupled graph gets exactly zero.
        let mut gb = GraphBuilder::new(4);
        gb.couple(0, 1, 0.7);
        gb.couple(1, 2, -0.5);
        gb.bias(3, 0.6);
        let mixed = gb.build();
        let out = run(&mixed, 0.8, &Params::default()).unwrap();
        assert!(out.cavity_precision[3].abs() < 1e-15, "an isolated site has no reaction term");
    }

    /// Moment consistency: at a fixed point the tilted mean and variance equal the Gaussian's.
    ///
    /// This is what "expectation consistent" names, and it is the one property the two reported
    /// means are ALLOWED to share. They are computed by different routes — one a `tanh` of the
    /// cavity field, the other a Cholesky solve — so agreement to 1e-10 is a statement about the
    /// fixed point rather than about the code.
    #[test]
    fn the_two_means_agree_at_the_fixed_point() {
        for seed in 0..4u64 {
            let g = frustrated(11, 18, seed);
            let out = run(&g, 0.3, &Params::default()).unwrap();
            for i in 0..g.n {
                assert!(
                    (out.m[i] - out.gaussian_mean[i]).abs() < 1e-9,
                    "seed {seed} site {i}: tilted {} vs Gaussian {}",
                    out.m[i],
                    out.gaussian_mean[i]
                );
                assert!(
                    (out.variance[i] - out.sigma[i]).abs() < 1e-9,
                    "seed {seed} site {i}: tilted variance {} vs Gaussian {}",
                    out.variance[i],
                    out.sigma[i]
                );
            }
        }
    }

    /// `ln(2 cosh x)` must survive the fields EP actually reaches, where the textbook form is `inf`.
    #[test]
    fn the_log_cosh_helper_does_not_overflow_where_cosh_does() {
        assert!((2.0f64 * 800.0f64.cosh()).ln().is_infinite(), "the naive form must really fail");
        assert!((log_two_cosh(800.0) - 800.0).abs() < 1e-12);
        assert!((log_two_cosh(0.0) - 2.0f64.ln()).abs() < 1e-15);
        // Symmetric, and equal to the textbook form wherever the textbook form exists.
        for x in [-3.25f64, -0.5, 0.125, 4.0, 20.0] {
            assert!((log_two_cosh(x) - log_two_cosh(-x)).abs() < 1e-15);
            assert!((log_two_cosh(x) - (2.0 * x.cosh()).ln()).abs() < 1e-13, "at {x}");
        }
    }

    /// A model with nothing in it is not a special case that panics or divides by zero.
    #[test]
    fn an_empty_model_has_a_zero_free_energy() {
        let g = GraphBuilder::new(0).build();
        let out = run(&g, 1.0, &Params::default()).unwrap();
        assert!(out.m.is_empty());
        assert_eq!(out.log_z, 0.0);
    }

    /// A single spin with a field: `ln Z = ln 2cosh(βh)` and `⟨s⟩ = tanh(βh)`, exactly.
    #[test]
    fn one_spin_is_the_two_state_system() {
        let mut gb = GraphBuilder::new(1);
        gb.bias(0, 0.75);
        let g = gb.build();
        let out = run(&g, 1.3, &Params::default()).unwrap();
        let x: f64 = 1.3 * 0.75;
        assert!((out.m[0] - x.tanh()).abs() < 1e-16);
        assert!((out.log_z - (2.0 * x.cosh()).ln()).abs() < 1e-15);
    }
}
