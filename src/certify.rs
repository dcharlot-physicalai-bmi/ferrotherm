//! Proof that a sampler did what it claimed.
//!
//! Every commercial machine in this field returns "best found". Not one of them — American,
//! Japanese or Chinese — tells you at what temperature it actually sampled, how many of its samples
//! were independent, or how far its distribution sat from the one you asked for. This module is
//! that missing answer, and it is the thing the rest of the stack exists to make trustworthy.
//!
//! A [`Certificate`] is computed **from samples alone**, not from the sampler's own account of
//! itself. That is deliberate: a sampler cannot certify itself any more than a witness can
//! corroborate their own testimony, and taking only the samples means a deliberately broken sampler
//! can be handed to the same function and caught. It is, and the tests do exactly that.
//!
//! # What it measures
//!
//! - **`beta_eff`** — the inverse temperature the samples were actually drawn at, as a
//!   pseudolikelihood maximum-likelihood estimate. If a site is +1 with frequency `sigma(2 beta f)`
//!   given its local field `f`, then fitting one parameter to every (field, spin) pair observed
//!   recovers `beta`. A sampler running at the wrong temperature cannot hide from this.
//! - **`tau_int` and `ess`** — integrated autocorrelation time (Geyer's initial monotone sequence,
//!   cross-checked by long-batch means, the larger carried; see [`tau_estimate`]), and the
//!   resulting count of genuinely independent samples, never more than the draws. Ten thousand
//!   correlated draws are not ten thousand samples, and reporting them as such is the most common
//!   quiet lie in MCMC. Until 2026-09-28 this was Sokal's automatic window, which read the time
//!   more than 10% low in 26 of 151 exactly-known cells and passed a 3x3 glass at `beta = 1` clean
//!   in 12 of 16 seeds with its effective sample size overstated 1.72 to 1.99 times.
//! - **`tv_exact`** — where the model is small enough to enumerate, total variation distance from
//!   the true Boltzmann distribution, always alongside
//! - **`noise_floor`** — the TV that finite sampling alone produces. A distance below the floor is
//!   agreement, not accuracy, and this module refuses to let the two be confused.
//!
//! Findings are a list. An empty list is the only thing that means "passed".

use crate::graph::Graph;

/// Something wrong with a run, in the sampler's own output.
#[derive(Clone, Debug, PartialEq)]
pub enum Finding {
    /// The temperature the samples were drawn at is not the temperature that was requested.
    BetaMismatch {
        /// The inverse temperature asked for.
        requested: f64,
        /// The one the samples are consistent with.
        effective: f64,
        /// 95% interval on `effective`, widened for autocorrelation.
        ci: (f64, f64),
    },
    /// Successive samples are too correlated for the count to mean what it says.
    Undermixed {
        /// Integrated autocorrelation time, in draws.
        tau_int: f64,
        /// Effective sample size these draws are worth.
        ess: f64,
        /// Draws taken.
        draws: usize,
    },
    /// The distribution is measurably not the Boltzmann distribution, beyond sampling noise.
    AboveNoiseFloor {
        /// Measured total-variation distance from the exact distribution.
        tv: f64,
        /// The distance finite sampling alone produces. Never quote a `tv` below this.
        floor: f64,
    },
    /// The chain was still drifting: early samples do not look like late ones.
    NotConverged {
        /// Mean energy over the first half of the run.
        early: f64,
        /// Mean over the second half.
        late: f64,
        /// Standard error the gap is judged against.
        sigma: f64,
    },
    /// Too few samples to say anything. Reported rather than guessed at.
    TooFewSamples {
        /// Draws taken, which is too few to estimate anything here.
        draws: usize,
    },
    /// Geyer's initial sequence stopped before a slow mode that long-batch means still see: the
    /// overlapping-batch-means estimate at `N / 20` came out more than twice as large. The larger
    /// value is the one the certificate's `tau_int` and `ess` carry, and even that is a lower bound
    /// on the truth -- see [`tau_int`] for the measurement behind this, and for why a
    /// non-reversible chain (every fixed-order sweep in this crate) can do this and a reversible
    /// one cannot.
    TauTruncated {
        /// What Geyer's initial monotone sequence returned, in draws (the larger over the traces).
        geyer: f64,
        /// What overlapping batch means at batch length `N / 20` returned, in draws.
        batch: f64,
    },
    /// Geyer's initial sequence summed to zero or less on a trace, so it has NO value there: the
    /// draws are strongly antithetic (a lag-one autocorrelation near `-1`) or the chain is not
    /// reversible. Such a sum is a failure of the estimator, not an effective sample size of
    /// `N log10 N`, which is what capping it would report. The certificate carries the long-batch
    /// value for that trace instead, never below `1/2`, so `ess` stays at or below `draws`.
    TauUnresolved {
        /// Geyer's sum on the trace where it failed, in draws: zero or negative.
        geyer: f64,
        /// The overlapping-batch-means value carried in its place (before the `1/2` floor).
        batch: f64,
    },
    /// The chain is too short, measured in its own autocorrelation times, for `tau_int` to be more
    /// than a LOWER bound -- and `ess` therefore more than an UPPER bound. Below
    /// [`RESOLVED_TAUS`] autocorrelation times every single-chain estimator in this crate reads low
    /// on a chain with a slow mode: measured over 16 seeds on five slow-mode cells, Geyer's median
    /// read 0.41 to 0.68 of the exact value at 100 `tau` and 0.74 to 0.99 at 1,000.
    TauLowerBound {
        /// Integrated autocorrelation time carried, in draws: a lower bound.
        tau_int: f64,
        /// Effective sample size these draws are worth at most.
        ess: f64,
        /// Draws taken.
        draws: usize,
    },
}

impl core::fmt::Display for Finding {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Finding::BetaMismatch { requested, effective, ci } => write!(
                f,
                "sampled at beta {effective:.4} (95% CI {:.4}..{:.4}), not the requested {requested:.4}",
                ci.0, ci.1
            ),
            Finding::Undermixed { tau_int, ess, draws } => write!(
                f,
                "draws are correlated: tau_int {tau_int:.1}, so {draws} draws are worth about \
                 {ess:.0} independent samples; thin the chain or run longer"
            ),
            Finding::AboveNoiseFloor { tv, floor } => write!(
                f,
                "total variation {tv:.4} exceeds the {floor:.4} sampling-noise floor, so the \
                 difference is real rather than finite-sample scatter"
            ),
            Finding::NotConverged { early, late, sigma } => write!(
                f,
                "the chain was still moving: early draws average {early:.4} and late ones \
                 {late:.4}, a gap of {sigma:.1} standard errors; burn in for longer"
            ),
            Finding::TooFewSamples { draws } => {
                write!(
                    f,
                    "{draws} draws is too few to certify anything: the sampling-noise floor for a \
                     state space this large reaches or exceeds 1, which is the most a total \
                     variation can be, so a distributional comparison here cannot distinguish a \
                     good sampler from pure noise. Draw more, or certify a smaller model"
                )
            }
            Finding::TauTruncated { geyer, batch } => write!(
                f,
                "the autocorrelation sum stopped early: Geyer's tau_int {geyer:.1} against a \
                 long-batch-means {batch:.1}, so a slow mode is being missed; the larger value is \
                 used and is itself a lower bound. Run much longer, or compute tau exactly on a \
                 model small enough to enumerate"
            ),
            Finding::TauUnresolved { geyer, batch } => write!(
                f,
                "Geyer's autocorrelation sum came to {geyer:.3}, zero or less, so it has no value \
                 on this trace (strongly antithetic draws, or a chain that is not reversible); the \
                 long-batch-means {batch:.3} is carried instead, never below 1/2. Thinning by an \
                 even number of sweeps turns an alternating chain into a positively correlated one"
            ),
            Finding::TauLowerBound { tau_int, ess, draws } => write!(
                f,
                "{draws} draws are only {:.0} autocorrelation times, too few for tau_int \
                 {tau_int:.2} to be more than a lower bound or ess {ess:.0} more than an upper \
                 one: below {RESOLVED_TAUS:.0} single-chain estimators read low on a slow mode. \
                 Run longer",
                *draws as f64 / tau_int
            ),
        }
    }
}

/// What a run actually did.
#[derive(Clone, Debug)]
pub struct Certificate {
    /// States the certificate was computed from.
    pub draws: usize,
    /// The inverse temperature the caller ASKED for, against which `beta_eff` is judged.
    pub beta_requested: f64,
    /// Pseudolikelihood MLE of the inverse temperature the samples came from.
    pub beta_eff: f64,
    /// 95% interval for `beta_eff`, widened for autocorrelation.
    pub beta_ci: (f64, f64),
    /// Integrated autocorrelation time, in draws, in the convention `1/2 + sum_k rho(k)`: `0.5`
    /// means consecutive draws are independent, and it is never reported below that. See
    /// [`tau_estimate`] for how it is measured.
    pub tau_int: f64,
    /// Effective sample size, `draws / (2 tau_int)` -- how many independent draws these are worth.
    /// Never more than `draws`.
    pub ess: f64,
    /// TV from the exact Boltzmann distribution, where enumeration was possible.
    pub tv_exact: Option<f64>,
    /// TV that finite sampling alone produces. Never quote a distance below this.
    pub noise_floor: Option<f64>,
    /// Empty means the run is sound as far as this can tell.
    pub findings: Vec<Finding>,
}

impl Certificate {
    #[must_use = "a certificate with findings is a certificate that failed; ignoring this is reporting a sound run that was not one"]
    /// Whether the run is sound as far as this could tell: no findings.
    pub fn passed(&self) -> bool {
        self.findings.is_empty()
    }
}

impl core::fmt::Display for Certificate {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        writeln!(
            f,
            "draws {}  beta {:.4} (asked {:.4}, CI {:.4}..{:.4})  tau_int {:.1}  ess {:.0}",
            self.draws, self.beta_eff, self.beta_requested, self.beta_ci.0, self.beta_ci.1,
            self.tau_int, self.ess
        )?;
        if let (Some(tv), Some(fl)) = (self.tv_exact, self.noise_floor) {
            writeln!(f, "tv {tv:.4} against a {fl:.4} noise floor")?;
        }
        if self.findings.is_empty() {
            write!(f, "PASSED")
        } else {
            for x in &self.findings {
                writeln!(f, "FINDING: {x}")?;
            }
            Ok(())
        }
    }
}

/// Estimate the inverse temperature the samples were drawn at, and the variance of that estimate.
///
/// Maximum pseudolikelihood: every site of every sample contributes one logistic observation with
/// feature `2 f_i` and label `s_i`, and the log-likelihood in beta is concave.
///
/// # The variance is a sandwich, and the textbook Fisher form is wrong here
///
/// Pseudolikelihood is a **composite** likelihood — a product of conditionals that is not the
/// likelihood of anything. For a real likelihood the asymptotic variance is the inverse Hessian
/// `H^-1`; for a composite one it is the Godambe sandwich `H^-1 J H^-1`, where `J` is the variance
/// of the score. The two agree only when the terms being multiplied are independent, and here they
/// emphatically are not: the `n` conditionals from one configuration all read the same spins, so
/// their scores are positively correlated and `H^-1` under-states the variance.
///
/// This returned `H` and the caller took `sqrt(1/H)`, and the consequence was measurable. Certifying
/// EXACT independent draws — where the sampler cannot be wrong, because there is no sampler —
/// produced a `BetaMismatch` finding on 12% of runs for `ring(10, 1.0, 0.0)`, 18% for its
/// antiferromagnet and 8% for a fixture at `beta = 0.6`, against the 5% a 95% interval is allowed.
/// `beta_eff` itself was unbiased to about 0.02%, so the point estimate was never the problem: the
/// interval was too narrow by half, and the instrument was calling correct samplers broken.
///
/// So `J` is estimated by clustering on the configuration — `J = sum_d (sum_i x(y - p))^2`, one term
/// per draw — which is exactly the dependence the naive form ignores. Correlation BETWEEN draws is a
/// separate matter and is handled by the caller's autocorrelation inflation.
///
/// Returns `(beta, variance_of_beta)`.
fn fit_beta(g: &Graph, samples: &[Vec<i8>]) -> (f64, f64) {
    // Precompute (field, spin) once; the fit visits them many times. `starts` records where each
    // configuration's observations begin, because the sandwich needs to know which observations
    // came from the same spins.
    let mut obs: Vec<(f64, f64)> = Vec::with_capacity(samples.len() * g.n);
    let mut starts: Vec<usize> = Vec::with_capacity(samples.len() + 1);
    for s in samples {
        starts.push(obs.len());
        for i in 0..g.n {
            let f = g.field(i, s);
            if f != 0.0 {
                obs.push((2.0 * f, if s[i] > 0 { 1.0 } else { 0.0 }));
            }
        }
    }
    starts.push(obs.len());
    if obs.is_empty() {
        return (f64::NAN, f64::INFINITY); // every field was zero: the data says nothing about beta
    }

    // d(log L)/d(beta), which is strictly decreasing because the log-likelihood is concave.
    let d1 = |b: f64| -> f64 {
        obs.iter().map(|&(x, y)| x * (y - 1.0 / (1.0 + (-b * x).exp()))).sum()
    };

    // Bisection rather than Newton. Newton is faster and wrong here: far from the optimum the
    // logistic saturates, the second derivative underflows, and the step diverges -- which it did,
    // pinning uniform noise at the clamp instead of reporting its true beta of zero. Monotonicity
    // makes bracketing exact, so the robust method is also the correct one.
    const LIM: f64 = 60.0;
    let (mut lo, mut hi) = (-LIM, LIM);
    let (flo, fhi) = (d1(lo), d1(hi));
    if flo <= 0.0 || fhi >= 0.0 {
        // No sign change: the likelihood is maximised at the edge, which means the fields separate
        // the spins perfectly. Report the bound rather than a fabricated interior value.
        let beta = if fhi >= 0.0 { LIM } else { -LIM };
        return (beta, f64::INFINITY);
    }
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if d1(mid) > 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
        if hi - lo < 1e-12 {
            break;
        }
    }
    let beta = 0.5 * (lo + hi);

    // H: the Hessian of the pseudo-log-likelihood, the "bread" of the sandwich.
    let h: f64 = obs
        .iter()
        .map(|&(x, _)| {
            let p = 1.0 / (1.0 + (-beta * x).exp());
            x * x * p * (1.0 - p)
        })
        .sum();
    if !(h > 0.0) {
        return (beta, f64::INFINITY);
    }

    // J: the variance of the score, clustered by configuration -- the "meat". The scores sum to
    // zero at the optimum, so this is a variance about a known mean and needs no centring.
    let clusters = starts.len() - 1;
    let j: f64 = starts
        .windows(2)
        .map(|w| {
            obs[w[0]..w[1]]
                .iter()
                .map(|&(x, y)| x * (y - 1.0 / (1.0 + (-beta * x).exp())))
                .sum::<f64>()
                .powi(2)
        })
        .sum();

    // With one cluster there is no spread to estimate from, and a sandwich built on a single term
    // would report a variance of whatever that term happened to be. Fall back to the naive form and
    // let the caller's inflation be the only widening, which is the conservative direction.
    let var = if clusters >= 2 { j / (h * h) } else { 1.0 / h };
    (beta, var)
}

/// Integrated autocorrelation time of a scalar trace: Geyer's initial monotone sequence over the
/// trace's own autocorrelation.
///
/// `tau_int = 1/2 + sum_k rho(k)` in this crate's convention, so INDEPENDENT DRAWS HAVE
/// `tau_int = 1/2` and an effective sample size of `N / (2 tau_int) = N`. The empirical
/// autocorrelation, from [`crate::fft::autocovariance`] in the divided-by-`N` form Stan uses, is
/// summed in adjacent pairs `rho(2m) + rho(2m + 1)` while the pair sums stay positive, each pair
/// clipped to the one before it so the sequence is non-increasing ([`geyer_initial_monotone`], the
/// loop [`crate::rhat::ess`] runs over several chains, run here over one; Geyer, "Practical Markov
/// Chain Monte Carlo", Statistical Science 7(4), 1992).
///
/// Returns `NaN` under sixteen draws, `+inf` for a constant trace (a chain that never moved), and
/// `NaN` when the pair sums total zero or less: that is a FAILURE of the estimator, not a value,
/// and an effective sample size read off it would be infinite. Otherwise the value is never
/// below `1/2`, so an effective sample size built on it never exceeds the draws.
///
/// # Why Geyer and not Sokal's window, which this was until 2026-09-28
///
/// Sokal's window ([`tau_int_sokal`]) stops summing at the first lag `W >= 5 tau(W)`, and on a
/// chain whose autocorrelation is a large fast mode beside a small slow one it stops before the
/// slow mode is summed -- with NO sampling noise involved. Applied to EXACT autocorrelation
/// sequences from [`crate::autocorr::apply`] on seven enumerable fixtures, ten kernels and two or
/// three temperatures each, and counted the way [`certify`] carries it (the larger of energy and
/// magnetisation), the window read more than 10% low in 26 of 151 cells: `-92%` on the 3x3 glass at
/// `beta = 1.6` under SCA, `-60%` under the chromatic sweep at the same temperature, `-45%` at
/// `beta = 1` (2.98 against an exact 5.42 for the energy). A consumer that reads energy alone had it
/// worse: the 12-spin glass of `examples/tau_exactness.rs` reads 1.94 against 33.24, `-94%`.
///
/// Geyer's sequence is the right estimator for a REVERSIBLE kernel: its autocorrelation is
/// `sum_i a_i lambda_i^k` over real eigenvalues, so every pair sum is
/// `sum_i a_i lambda_i^(2m) (1 + lambda_i) >= 0` and the sequence is summed until the noise, not
/// until a window rule, ends it. On the exact sequences of reversible kernels it read within
/// `1e-4` on all 312 converged cells, including every synchronous and SCA cell with negative
/// eigenvalues. On 224 simulated traces at least 2,000 exact `tau` long (8 seeds; the chromatic
/// sweep, a sequential sweep and informed samplers), the count reading below 0.8 of the exact value
/// was 103 for Sokal's window, 57 for what the certificate used to carry, and 23 for this.
///
/// # And why it is not the whole certificate: NON-REVERSIBLE chains
///
/// A fixed-order sweep -- the chromatic sweep [`crate::gibbs::Sampler`] runs, the sequential sweep,
/// a lifted chain, a renewal process -- is a PRODUCT of reversible updates and is not itself
/// reversible. Its autocorrelation can rotate, pair sums can go negative, and then Geyer stops
/// early. It usually over-reads (the chromatic sweep's antithetic magnetisation on an
/// antiferromagnetic ring, `+29%` to `+99.8%`), but not always: on closed forms
/// `rho(k) = a r^k cos(w k) + (1 - a) lambda^k`, a fast rotation beside a slow real mode, it reads
/// `-94.6%` to `-99.4%`, and on an in-tree sequential sweep of `ring(8, -1, 0)` at `beta = 2`, with
/// `f = m + 0.001 m_stag`, `-7.7%`. So [`tau_estimate`] -- the one entry point every error bar and
/// certificate in the crate goes through -- carries this value beside a long-batch-means value
/// ([`tau_int_obm`] at `N / 20`), which needs no reversibility, and takes the LARGER. A certificate
/// is computed from samples alone and cannot know which kernel drew them, so the cross-check
/// applies to every chain, reversible or not.
///
/// # Every estimate here is a LOWER bound on a chain too short for it
///
/// Over 16 seeds on five slow-mode cells, this estimator's median read 0.41 to 0.68 of the exact
/// value at 100 `tau` of trace and 0.74 to 0.99 at 1,000. [`certify`] says so with
/// [`Finding::TauLowerBound`] below [`RESOLVED_TAUS`].
#[must_use]
pub fn tau_int(trace: &[f64]) -> f64 {
    let raw = tau_int_geyer_raw(trace);
    if raw.is_nan() || raw.is_infinite() {
        raw
    } else if raw > 0.0 {
        raw.max(0.5)
    } else {
        f64::NAN // a failure of the estimator, not a value -- see `TauUnresolved`
    }
}

/// Geyer's sum over a trace, unfloored: `NaN` under sixteen draws, `+inf` for a constant trace,
/// and otherwise whatever the initial monotone sequence totals -- which can be below `1/2` on an
/// antithetic trace and zero or less where the estimator has failed.
fn tau_int_geyer_raw(trace: &[f64]) -> f64 {
    let n = trace.len();
    if n < 16 {
        return f64::NAN;
    }
    let mean = trace.iter().sum::<f64>() / n as f64;
    let var = trace.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
    if var <= 0.0 {
        return f64::INFINITY; // a constant trace never decorrelates
    }
    // Per-lag autocovariances divided by the pairs that formed them, taken to Stan's biased form
    // (divided by N), as `rhat::ess` does, so the far lags are shrunk rather than amplified.
    let cov = crate::fft::autocovariance(trace, n - 1);
    let c0 = cov[0];
    if !(c0 > 0.0) {
        return f64::INFINITY;
    }
    let nf = n as f64;
    geyer_initial_monotone(n - 2, |k| cov[k] * (nf - k as f64) / nf / c0)
}

/// Geyer's initial monotone sequence over an autocorrelation given lag by lag, UNFLOORED, in this
/// crate's convention: `-1/2 + sum_m G_m` with `G_m = rho(2m) + rho(2m + 1)`, `rho(0) = 1`, summed
/// while `G_m > 0` and with each `G_m` clipped to the one before it, reading no lag past `max_lag`.
///
/// The same loop [`crate::rhat::ess`] runs over the multi-chain autocorrelation, taken out so it can
/// be measured on an autocorrelation known EXACTLY -- the way [`sokal_window`] is -- and so the two
/// cannot drift apart. A reversible kernel has `G_m >= 0` at every `m` and this is its exact
/// integrated time; a non-reversible one can have a negative pair before its slow mode, and there
/// this stops early (see [`tau_int`]). The result can be below `1/2` (an antithetic sequence) and
/// can be zero or negative, which no chain's integrated time is: callers treat that as a failure.
#[must_use]
pub fn geyer_initial_monotone(max_lag: usize, mut rho: impl FnMut(usize) -> f64) -> f64 {
    let mut sum = 0.0;
    let mut prev = f64::INFINITY;
    let mut t = 0;
    while t < max_lag {
        let mut p = rho(t) + rho(t + 1);
        if !(p > 0.0) {
            break;
        }
        if p > prev {
            p = prev;
        }
        prev = p;
        sum += p;
        t += 2;
    }
    -0.5 + sum
}

/// Integrated autocorrelation time of a scalar trace by Sokal's automatic window -- what
/// [`tau_int`] was until 2026-09-28, kept by name as a second opinion and as the object the
/// measurements of its failure are made on.
///
/// `tau_int = 1/2 + sum_k rho(k)`, truncated at the smallest window `W` satisfying `W >= 5 tau`.
/// Truncation is not optional for a window: the tail of an empirical autocorrelation is noise, and
/// summing all of it produces a number that grows with the length of the run rather than
/// describing it.
///
/// # This is a LOWER bound on a chain with a slow mode of small amplitude, and no trace length fixes it
///
/// Measured against [`crate::autocorr::tau_int_exact`] on a 12-spin frustrated grid at `beta = 1`
/// (2026-09-13, `examples/tau_exactness.rs`): the exact value is 33.2 sweeps and this returns
/// **0.04 to 0.06 of it at every trace length from 30 to 10,000 tau**, with a 2% spread at the
/// longest. The chain's autocorrelation there is a large fast mode plus a small slow one; `tau(W)`
/// is still about 1.8 when the window closes at lag 9, and the slow mode — most of the truth — is
/// never summed. On a hot, single-mode chain the same estimator lands within its noise of the
/// exact value. So an effective sample size from this is an UPPER bound wherever a slow mode
/// cannot be excluded, and a joules-per-independent-sample built on it a lower bound. This is why
/// [`tau_int`] is now Geyer's sequence.
///
/// # And it is biased either way on an autocorrelation that changes sign
///
/// The window stops wherever `k >= 5 tau` first holds, and an autocorrelation with a negative lobe
/// is cut mid-oscillation: 8.1% low and 13.3% high on the two exact cases in [`sokal_window`]'s
/// documentation, a fixed-lifetime point process read every `0.1 m`, where the effect is momentum.
/// Do not use it where momentum is the thing being measured.
#[must_use]
pub fn tau_int_sokal(trace: &[f64]) -> f64 {
    tau_int_by(trace, trace.len() >= FFT_FROM)
}

/// Above this many draws [`tau_int_sokal`] takes every lag at once from
/// [`crate::fft::autocovariance`], `O(L log L)`, instead of the direct sum, which is `O(L)` per lag
/// and on a long trace costs the window's width times the length (a `1e7`-draw trace with a window
/// of `1e4` lags is `1e11` operations). The two are the same estimator to `1e-12`; the threshold is
/// a clock, not a value. [`tau_int`] always takes the transform, because Geyer's sum is not
/// confined to a few `tau` of lags.
pub const FFT_FROM: usize = 4096;

/// Sokal's window over the autocorrelation, computed lag by lag or all at once.
fn tau_int_by(trace: &[f64], fft: bool) -> f64 {
    let n = trace.len();
    if n < 16 {
        return f64::NAN;
    }
    let mean = trace.iter().sum::<f64>() / n as f64;
    let var = trace.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
    if var <= 0.0 {
        return f64::INFINITY; // a constant trace never decorrelates
    }
    let max_lag = (n / 4).max(1);
    let cov = if fft { Some(crate::fft::autocovariance(trace, max_lag)) } else { None };
    sokal_window(max_lag, |k| {
        match &cov {
            Some(cov) => cov[k] / var,
            None => {
                let mut c = 0.0;
                for t in 0..(n - k) {
                    c += (trace[t] - mean) * (trace[t + k] - mean);
                }
                c / ((n - k) as f64 * var)
            }
        }
    })
}

/// Sokal's automatic window over an autocorrelation given lag by lag: `1/2 + sum_k rho(k)` for
/// `k = 1, 2, ...`, stopping at the first `k >= 5 tau` or at `max_lag`, and never below `1/2`. The
/// same window [`tau_int_sokal`] closes over a trace's empirical autocorrelation, taken out so that the
/// WINDOW's own error can be measured on an autocorrelation that is known exactly, with no sampling
/// noise in it.
///
/// # A negative lobe is summed wherever the window happens to close, and that can be either way
///
/// The window was designed for autocorrelations that decay without changing sign. One that dips
/// below zero and comes back is truncated mid-oscillation. [`crate::pointproc`] measures it on the
/// exact autocorrelation of one unit of a fixed-lifetime point process, `rho(t) = 2 e^{-t/m} - 1`
/// out to `t = m` at zero field: at a spacing of `0.1 m` the window closes at lag 12, inside the
/// negative lobe, and reads the integrated time **8.1% low**; in a field `beta h = 0.7` it closes
/// at lag 7, before the lobe, and reads it **13.3% high** (`sokal_window_misreads_a_fixed_lifetime`).
/// The unwindowed sum of the same values is exact there to `1e-9`.
#[must_use]
pub fn sokal_window(max_lag: usize, mut rho: impl FnMut(usize) -> f64) -> f64 {
    let mut tau = 0.5;
    for k in 1..=max_lag {
        tau += rho(k);
        if (k as f64) >= 5.0 * tau.max(0.5) {
            break;
        }
    }
    tau.max(0.5)
}

/// Integrated autocorrelation time by batch means: no window, no autocorrelation function.
///
/// Split the trace into `batches` consecutive batches of length `b`. For a stationary sequence
/// `Var(batch mean) ~ Var(x) * 2 tau / b`, so `tau = b * Var(batch means) / (2 Var(x))`. This is
/// unbiased only when `b` is much longer than the slowest mode, so on a short trace it reads LOW
/// -- but for a different reason and by a different amount from an autocorrelation sum, which is
/// what makes it a cross-check: its own noise is about `sqrt(2 / batches)` relative, `+-32%` at
/// twenty. [`certify`] now cross-checks with the OVERLAPPING form, [`tau_int_obm`], which at the
/// same batch length has two thirds of this variance; this one is kept because examples measure
/// with it.
///
/// `NaN` under sixteen draws per batch; `+inf` for a constant trace.
#[must_use]
pub fn tau_int_batch(trace: &[f64], batches: usize) -> f64 {
    let n = trace.len();
    let b = n / batches.max(2);
    if b < 16 {
        return f64::NAN;
    }
    let mean = trace.iter().sum::<f64>() / n as f64;
    let var = trace.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
    if var <= 0.0 {
        return f64::INFINITY;
    }
    let k = batches.max(2);
    let means: Vec<f64> =
        (0..k).map(|i| trace[i * b..(i + 1) * b].iter().sum::<f64>() / b as f64).collect();
    let vb = means.iter().map(|m| (m - mean).powi(2)).sum::<f64>() / (k as f64 - 1.0);
    (b as f64 * vb / (2.0 * var)).max(0.5)
}

/// Integrated autocorrelation time by OVERLAPPING batch means at batch length `b`: no window, no
/// autocorrelation function, and no reversibility assumed.
///
/// Every window of `b` consecutive draws, `N - b + 1` of them, contributes its mean; the long-run
/// variance is `sigma^2 = N b / ((N - b)(N - b + 1)) * sum_j (mean_j - mean)^2`, and
/// `tau = sigma^2 / (2 Var(x))` in this crate's convention. Overlapping batch means is one of the
/// estimators Flegal and Jones give conditions for strong consistency of ("Batch means and spectral
/// variance estimators in Markov chain Monte Carlo", Annals of Statistics 38(2), 2010,
/// arXiv:0811.1729), and like every lag-window estimator it reads LOW unless `b` is much longer than
/// the slowest mode -- "estimators of this matrix almost always exhibit significant negative bias"
/// under positive correlation (Vats and Flegal, arXiv:1809.04541). That is why it is a cross-check
/// and not the estimator: on 224 simulated traces at least 2,000 exact `tau` long, `b = N / 20`
/// read below 0.8 of the exact value on 59 and above 1.25 on 51, against Geyer's 23 and 13 -- and
/// `b = sqrt(N)`, the textbook choice, read below 0.8 on 86. What it does not need is a reversible
/// kernel, which is where [`tau_int`] can fail.
///
/// `NaN` for `b < 16` or `b >= N`; `+inf` for a constant trace; never below `1/2`.
#[must_use]
pub fn tau_int_obm(trace: &[f64], b: usize) -> f64 {
    let n = trace.len();
    if b < 16 || b >= n {
        return f64::NAN;
    }
    let mean = trace.iter().sum::<f64>() / n as f64;
    let var = trace.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
    if var <= 0.0 {
        return f64::INFINITY;
    }
    // Prefix sums of the CENTRED trace, so a batch mean is a difference of two sums whose size is
    // the walk's excursion and not N times the mean.
    let mut prefix = Vec::with_capacity(n + 1);
    let mut acc = 0.0f64;
    prefix.push(0.0);
    for x in trace {
        acc += x - mean;
        prefix.push(acc);
    }
    let bf = b as f64;
    let mut ss = 0.0f64;
    for j in 0..=(n - b) {
        let m = (prefix[j + b] - prefix[j]) / bf;
        ss += m * m;
    }
    let nf = n as f64;
    let sigma2 = nf * bf / ((nf - bf) * (nf - bf + 1.0)) * ss;
    (sigma2 / (2.0 * var)).max(0.5)
}

/// Batches per trace in the certificate's long-batch cross-check: [`tau_int_obm`] at `b = N / 20`.
/// Twenty batch lengths put its own noise near `+-26%`, so a disagreement of a factor of two is
/// about four of its standard deviations -- the threshold [`Finding::TauTruncated`] reports at.
pub const CROSS_CHECK_BATCHES: usize = 20;

/// Autocorrelation times of trace, below which a chain's `tau_int` is reported as a lower bound
/// ([`Finding::TauLowerBound`]). Chosen from a measurement, not for roundness: at 100 `tau` of
/// trace the median read on five slow-mode cells was 0.41 to 0.68 of the exact value, and at
/// 1,000 it was 0.74 to 0.99.
pub const RESOLVED_TAUS: f64 = 1000.0;

/// What the crate divides a chain's draws by, and how far to believe it. Built by [`tau_estimate`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TauEstimate {
    /// The integrated autocorrelation time carried: over every trace given, the larger of Geyer's
    /// sequence and long-batch means, never below `1/2`. `NaN` under sixteen draws; `+inf` when a
    /// trace is constant.
    pub tau: f64,
    /// Geyer's initial monotone sequence ([`tau_int`]), the largest over the traces where it has a
    /// value; `NaN` where it has none on any.
    pub geyer: f64,
    /// Overlapping batch means at `b = N / 20` ([`tau_int_obm`]), the largest over the traces;
    /// `NaN` under 320 draws, where the batches would be too short to mean anything.
    pub batch: f64,
    /// Draws per trace (the shortest, if they differ).
    pub draws: usize,
    /// Where Geyer's sum came to zero or less on a trace, that sum -- a failure, not a value, and
    /// the batch value is carried for that trace instead. `None` where it never failed.
    pub unresolved: Option<f64>,
}

impl TauEstimate {
    /// Effective sample size, `draws / (2 tau)`, which the `1/2` floor keeps at or below `draws`.
    /// `1` where `tau` is not finite: a frozen chain is worth one draw, and the infinite `tau` is
    /// left in [`Self::tau`] so a caller reading it still sees why.
    #[must_use]
    pub fn ess(&self) -> f64 {
        if self.tau.is_finite() && self.tau > 0.0 {
            self.draws as f64 / (2.0 * self.tau)
        } else {
            1.0
        }
    }

    /// Whether long-batch means came out more than twice Geyer's value: a slow mode Geyer's
    /// sequence stopped short of. See [`Finding::TauTruncated`].
    #[must_use]
    pub fn truncated(&self) -> bool {
        self.geyer.is_finite() && self.batch.is_finite() && self.batch > 2.0 * self.geyer
    }

    /// Whether the chain is under [`RESOLVED_TAUS`] autocorrelation times long, so that `tau` is a
    /// lower bound and [`Self::ess`] an upper one. See [`Finding::TauLowerBound`].
    #[must_use]
    pub fn lower_bound(&self) -> bool {
        self.tau.is_finite() && (self.draws as f64) < RESOLVED_TAUS * self.tau
    }
}

/// THE ONE ESTIMATOR ENTRY POINT: the autocorrelation time of a chain seen through one or more
/// scalar traces of it, each in chain order and of the same length.
///
/// Per trace, Geyer's initial monotone sequence ([`tau_int`]) and overlapping batch means at
/// `N / 20` ([`tau_int_obm`]); carried, the largest of all of them, never below `1/2`. Geyer's
/// sequence is exact on a reversible kernel's autocorrelation; batch means need no reversibility
/// and catch the non-reversible chain Geyer stops short on. Taking the larger rather than choosing
/// cost this much on 224 simulated traces at least 2,000 exact `tau` long: reads below 0.8 of the
/// exact value went from 23 to 19, the worst from 0.566 to 0.618, and reads above 1.25 from 13 to
/// 59, a geometric-mean ratio of 1.10 against 0.98 -- error bars about 5% wider on average, which is
/// the direction a certificate may err in. [`certify`], [`crate::samples::SampleSet`],
/// [`crate::free_energy`], [`crate::potts::estimate`], [`crate::sse`] and every other error bar in
/// the crate divide by `tau` from here.
///
/// Where Geyer's sum is zero or less on a trace, it has no value there: that trace contributes its
/// batch value and [`TauEstimate::unresolved`] says so. An empty slice gives `NaN`.
#[must_use]
pub fn tau_estimate(traces: &[&[f64]]) -> TauEstimate {
    let draws = traces.iter().map(|t| t.len()).min().unwrap_or(0);
    let (mut tau, mut geyer, mut batch) = (f64::NAN, f64::NAN, f64::NAN);
    let mut unresolved = None;
    // `f64::max` ignores a NaN operand, so a trace too short for one estimator leaves the others
    // standing -- and an infinite (constant) trace dominates, as a chain that never moved should.
    for t in traces {
        let raw = tau_int_geyer_raw(t);
        let obm = tau_int_obm(t, t.len() / CROSS_CHECK_BATCHES);
        if !raw.is_nan() && raw <= 0.0 {
            unresolved = Some(raw);
        } else {
            let g = if raw.is_finite() { raw.max(0.5) } else { raw };
            geyer = geyer.max(g);
            tau = tau.max(g);
        }
        batch = batch.max(obm);
        tau = tau.max(obm);
    }
    if unresolved.is_some() && tau.is_nan() && draws >= 16 {
        // Geyer failed and the trace is too short for batch means: carry the independent value,
        // which is the most a trace this short can claim, rather than no value at all.
        tau = 0.5;
    }
    TauEstimate { tau, geyer, batch, draws, unresolved }
}

/// Certify a set of samples against the model and temperature they claim to come from.
///
/// `trace` is a scalar observable, one value per sample, used for the autocorrelation estimate;
/// energy is the usual choice. Samples must be in chain order for `tau_int` to mean anything.
#[must_use]
pub fn certify(g: &Graph, beta_requested: f64, samples: &[Vec<i8>], trace: &[f64]) -> Certificate {
    let draws = samples.len();
    if draws < 16 {
        return Certificate {
            draws,
            beta_requested,
            beta_eff: f64::NAN,
            beta_ci: (f64::NAN, f64::NAN),
            tau_int: f64::NAN,
            ess: f64::NAN,
            tv_exact: None,
            noise_floor: None,
            findings: vec![Finding::TooFewSamples { draws }],
        };
    }

    let (beta_eff, beta_var) = fit_beta(g, samples);

    // Autocorrelation of ONE observable measures how fast that observable mixes, which is not the
    // same as how fast the configuration does. An ordered lattice is the case in point: it sits in
    // a single basin while its energy jitters quickly around a fixed value, so an energy trace
    // reports fast mixing for a chain that has not moved. Magnetization sees exactly what energy
    // misses, so both are measured and the worse one is reported.
    let mag: Vec<f64> = samples
        .iter()
        .map(|s| s.iter().map(|&x| x as f64).sum::<f64>() / g.n as f64)
        .collect();
    // THE AUTOCORRELATION TIME, from the one entry point every error bar in the crate uses: Geyer's
    // initial monotone sequence on each trace, overlapping batch means at N/20 beside it, the
    // larger carried and never below 1/2 -- see `tau_int` for why each, measured. Geyer's sequence
    // is exact for a reversible kernel and can stop short on a non-reversible one; this function
    // sees only samples and cannot know which drew them, so the cross-check is applied to every
    // chain. Where the two disagree by more than batch means' own noise, the certificate says so.
    let est = tau_estimate(&[trace, &mag]);
    let t = est.tau;
    let ess = est.ess();

    // Two dependences, two corrections. WITHIN a configuration the pseudolikelihood's conditionals
    // share spins, which `fit_beta`'s sandwich handles; BETWEEN configurations a chain is
    // autocorrelated, which is this inflation. Applying only one of them was worth a factor of two
    // in the interval's width -- see `fit_beta`.
    let se = if beta_var.is_finite() && beta_var > 0.0 { beta_var.sqrt() } else { f64::INFINITY };
    let inflate = if t.is_finite() { (2.0 * t).sqrt().max(1.0) } else { 1.0 };
    let half = 1.96 * se * inflate;
    let beta_ci = (beta_eff - half, beta_eff + half);

    let mut findings = Vec::new();
    if beta_eff.is_finite() && !(beta_ci.0 <= beta_requested && beta_requested <= beta_ci.1) {
        findings.push(Finding::BetaMismatch {
            requested: beta_requested,
            effective: beta_eff,
            ci: beta_ci,
        });
    }
    // Two ways a draw count can be misleading, and both are worth saying out loud. Fewer than 50
    // independent samples estimates nothing reliably whatever the raw count claims; and a tau
    // exceeding a fiftieth of the run means the estimate had too little to work with, so tau
    // itself is not to be trusted. The thresholds are round numbers, but they are round numbers
    // chosen against measured chains rather than picked to make the suite pass -- a run measuring
    // tau 43 with ess 35 out of 3,000 draws is undermixed by any reading, and an earlier ess < 30
    // line let exactly that through.
    if !t.is_finite() || ess < 50.0 || t > draws as f64 / 50.0 {
        findings.push(Finding::Undermixed { tau_int: t, ess, draws });
    }
    if est.truncated() {
        findings.push(Finding::TauTruncated { geyer: est.geyer, batch: est.batch });
    }
    if let Some(geyer) = est.unresolved {
        findings.push(Finding::TauUnresolved { geyer, batch: est.batch });
    }
    // Resolved or not is a statement about the MEASUREMENT, separate from Undermixed's statement
    // about the chain: at fewer than RESOLVED_TAUS autocorrelation times, the tau above is a lower
    // bound and the ess an upper one, whatever the draw count looks like.
    if est.lower_bound() {
        findings.push(Finding::TauLowerBound { tau_int: t, ess, draws });
    }

    // A Geweke-style check. beta_eff cannot do this job: pseudolikelihood is a LOCAL statistic, and
    // a chain trapped in a metastable configuration still has locally correct conditionals. Only
    // comparing the start of the run with the end sees a chain that is still travelling.
    if draws >= 60 {
        let cut = draws / 3;
        let early = &mag[..cut];
        let late = &mag[draws - cut..];
        let m = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
        let (me, ml) = (m(early), m(late));
        let var = |v: &[f64], mu: f64| {
            v.iter().map(|x| (x - mu).powi(2)).sum::<f64>() / (v.len() as f64 - 1.0).max(1.0)
        };
        // standard errors inflated by the autocorrelation, or the test fires on every chain
        let infl = if t.is_finite() { (2.0 * t).max(1.0) } else { 1.0 };
        let se = ((var(early, me) + var(late, ml)) * infl / cut as f64).sqrt();
        if se > 0.0 {
            let z = (me - ml).abs() / se;
            if z > 4.0 {
                findings.push(Finding::NotConverged { early: me, late: ml, sigma: z });
            }
        }
    }

    // Exact comparison where the model is small enough to enumerate.
    let (mut tv_exact, mut noise_floor) = (None, None);
    if g.n <= 20 {
        let exact = crate::ising::exact_boltzmann(g, beta_requested);
        let mut hist = vec![0.0f64; 1 << g.n];
        for s in samples {
            let mut k = 0usize;
            for (b, &v) in s.iter().enumerate() {
                if v > 0 {
                    k |= 1 << b;
                }
            }
            hist[k] += 1.0;
        }
        for h in &mut hist {
            *h /= draws as f64;
        }
        let tv = crate::ising::tv(&hist, &exact);
        // Expected TV from finite sampling of a distribution over this many states. Comparing a
        // measured TV against zero instead of this is how a correct sampler gets called broken.
        let floor = 0.5 * ((1usize << g.n) as f64 / ess.max(1.0)).sqrt();
        // A floor at or above 1 is not a floor, because TV between two distributions cannot exceed
        // 1 -- so `tv > floor` becomes unsatisfiable and this gate silently switches itself OFF on
        // exactly the models it matters for. Measured: iid uniform noise against a 16-spin lattice
        // at 4000 draws gives tv = 0.999 and floor = 2.04, so nothing was reported and
        // `Certificate::passed()` counted the absence as a pass -- while still printing
        // `noise_floor: Some(2.04)` as though it were a real threshold. The same noise at n = 9 is
        // caught. A gate that reports "fine" when it cannot see is worse than no gate.
        if floor >= 1.0 {
            findings.push(Finding::TooFewSamples { draws });
        } else if tv > floor {
            findings.push(Finding::AboveNoiseFloor { tv, floor });
        }
        tv_exact = Some(tv);
        noise_floor = Some(floor);
    }

    Certificate {
        draws,
        beta_requested,
        beta_eff,
        beta_ci,
        tau_int: t,
        ess,
        tv_exact,
        noise_floor,
        findings,
    }
}

/// Assert that a certificate says its chain is a Boltzmann sample of `g` at `requested`.
///
/// # Why not `assert!(cert.passed())`
///
/// Because `passed()` demands that a **95%** interval cover the truth, and a correct sampler fails
/// that one run in twenty by construction. A test asserting it on six certificates — three fixtures
/// times two moves, which is what `cluster` does — fails `1 - 0.95^6 = 26%` of the time. Such a test
/// is not flaky, which would at least be noticeable: seeds are fixed, so it is DETERMINISTIC and
/// ARBITRARY. It encodes one lucky draw from a coin that lands wrong a quarter of the time, and any
/// change that perturbs the random stream silently re-rolls it. Both new samplers shipped with that
/// shape before this existed.
///
/// So the two halves of a certificate are judged differently, because they are different kinds of
/// claim:
///
///   - `TooFewSamples`, `AboveNoiseFloor`, `NotConverged` and `Undermixed` are **strict**. They
///     carry real margin — a distribution that is wrong fails them by a mile — so demanding they be
///     absent costs nothing.
///   - `BetaMismatch` is a 95% coin flip, so it is re-judged at four sigma using the certificate's
///     OWN reported error rather than a tolerance written here. Self-calibrating: it scales with the
///     draws, the fixture and the temperature, and no number in this function has to be maintained
///     when any of those change.
///
/// Four sigma leaves about a `6e-5` false-failure rate per certificate and still catches everything
/// worth catching — the optional-stopping defect in `cluster::Sampler` ran 12% high in `beta`, about
/// thirteen sigma.
#[cfg(test)]
pub(crate) fn assert_boltzmann(cert: &Certificate, requested: f64, what: &str) {
    for f in &cert.findings {
        assert!(
            matches!(f, Finding::BetaMismatch { .. }),
            "{what}: this is not a sampling-noise finding, it is a defect:\n{cert}"
        );
    }
    let half = 0.5 * (cert.beta_ci.1 - cert.beta_ci.0);
    assert!(
        half.is_finite() && half > 0.0,
        "{what}: the certificate reported no usable interval:\n{cert}"
    );
    let sigma = (cert.beta_eff - requested).abs() / (half / 1.96);
    assert!(
        sigma < 4.0,
        "{what}: sampled at beta {:.4} against a requested {requested:.4}, {sigma:.1} standard \
         errors out. Four is the bar, because the certificate's own interval is 95% and this test \
         must not fail one run in twenty on correct code.\n{cert}",
        cert.beta_eff
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gibbs::Sampler;
    use crate::rng::Pcg;

    /// Draw from a graph with a sampler that may be deliberately wrong.
    ///
    /// `beta_actual` is what the sampler really runs at; `thin` is sweeps between recorded draws.
    fn run(g: &Graph, beta_actual: f64, thin: usize, burn: usize, draws: usize, seed: u64)
        -> (Vec<Vec<i8>>, Vec<f64>)
    {
        let mut smp = Sampler::new(g, beta_actual, seed);
        smp.sweeps(burn, None);
        let mut samples = Vec::with_capacity(draws);
        let mut trace = Vec::with_capacity(draws);
        for _ in 0..draws {
            smp.sweeps(thin.max(1), None);
            samples.push(smp.s.clone());
            trace.push(g.energy(&smp.s));
        }
        (samples, trace)
    }

    /// The 95% interval covers 95% of the time, measured on draws that cannot be wrong.
    ///
    /// The calibration test this module did not have, and the one that would have caught a defect
    /// it shipped with. `certify` fits `beta` by maximum *pseudo*likelihood and took its interval from
    /// `sqrt(1/H)`, the inverse Hessian. That is the variance of a real likelihood. Pseudolikelihood
    /// is a COMPOSITE likelihood — a product of conditionals that is not the likelihood of anything
    /// — and its variance is the Godambe sandwich `H^-1 J H^-1`. The two agree only when the
    /// multiplied terms are independent, and the `n` conditionals from one configuration all read
    /// the same spins.
    ///
    /// The consequence was not subtle. On identical exact draws, the naive interval missed the true
    /// `beta` on 13.8% of runs and the sandwich on 5.0%, with the naive standard error a stable
    /// 0.76 of the right one — an interval 32% too narrow, at every draw count tried. So the
    /// instrument reported `BetaMismatch` on correct samplers about one run in seven, and a
    /// verification tool with that false-alarm rate is one people learn to argue with.
    ///
    /// The draws here are independent and exactly Boltzmann, by inverse-CDF over the enumerated
    /// distribution. That is the point: there is no sampler in this test, so every failure is the
    /// instrument measuring itself, and the expected count is exactly what the confidence level
    /// says. The band is four standard deviations of `Binomial(trials, 0.05)` and is two-sided,
    /// because an interval that never fires is as broken as one that always does.
    #[test]
    fn the_interval_covers_at_the_rate_it_claims() {
        let g = crate::ising::ring(10, 1.0, 0.0);
        let beta = 0.4;
        let (trials, draws) = (250usize, 500usize);

        let p = crate::ising::exact_boltzmann(&g, beta);
        let mut cdf = Vec::with_capacity(p.len());
        let mut total = 0.0;
        for &x in &p {
            total += x;
            cdf.push(total);
        }

        let mut missed = 0usize;
        for seed in 0..trials {
            let mut rng = crate::rng::Pcg::new(seed as u64, 7);
            let mut samples = Vec::with_capacity(draws);
            let mut trace = Vec::with_capacity(draws);
            for _ in 0..draws {
                let u = rng.f64() * total;
                let m = cdf.partition_point(|&c| c < u).min(p.len() - 1);
                let st: Vec<i8> =
                    (0..g.n).map(|i| if (m >> i) & 1 == 1 { 1i8 } else { -1 }).collect();
                trace.push(g.energy(&st));
                samples.push(st);
            }
            let c = certify(&g, beta, &samples, &trace);
            if !(c.beta_ci.0 <= beta && beta <= c.beta_ci.1) {
                missed += 1;
            }
        }

        // Binomial(250, 0.05): mean 12.5, sd 3.45. Four sd either way.
        let (lo, hi) = (2usize, 27usize);
        assert!(
            (lo..=hi).contains(&missed),
            "the 95% interval missed on {missed} of {trials} runs of EXACT independent draws; \
             a calibrated interval misses about {:.0}. Below {lo} the interval is too wide to \
             detect anything; above {hi} it is too narrow and reports correct samplers as broken, \
             which is what the naive inverse-Hessian variance did at about {:.0}",
            0.05 * trials as f64,
            0.138 * trials as f64
        );
    }

    /// A more correlated chain gets a wider interval, which is what the inflation is for.
    ///
    /// The companion to `the_interval_covers_at_the_rate_it_claims`, and deliberately a test of the
    /// MECHANISM rather than of coverage. Coverage cannot pin this factor, and the reason is worth
    /// recording rather than discovering twice.
    ///
    /// On `ring(10, 1.0, 0.2)` at `beta = 0.6`, 500 draws after 400 sweeps of burn-in, seeds 0 to
    /// 299, re-measured 2026-09-28 with the certificate's `tau_int` as it now is (Geyer's sequence
    /// and long-batch means over energy AND magnetisation, the larger carried):
    ///
    /// ```text
    ///   thin   cert tau_int   exact E / m     with inflation   without
    ///      1        2.88      1.517 / 2.698      0.0% miss     8.3% miss      (target 5.0%)
    ///      2        1.47      0.872 / 1.377      0.0%          7.3%
    ///      5        0.74      0.563 / 0.666      1.7%          5.0%
    /// ```
    ///
    /// This table used to read `tau_int` 1.51 / 0.86 / 0.57 and 2.3 / 4.7 / 3.0% without the
    /// inflation. The `tau_int` column was the ENERGY's autocorrelation time, from before the
    /// certificate carried the larger of energy and magnetisation -- the magnetisation is the slower
    /// one here -- and the without-inflation column does not reproduce: the same protocol on the
    /// tree before 2026-09-28 gives 2.79 / 1.38 / 0.67 and 8.3 / 7.3 / 5.0% (and reproduces the
    /// with-inflation column, 0.0 / 0.7 / 2.3%, exactly). So the claim this paragraph made, that
    /// removing the inflation moves coverage toward the nominal rate, is not what it measures:
    /// with it the interval is too wide, without it too narrow, by comparable amounts. The
    /// mechanism is still the one stated here: the inflation scales by the autocorrelation of the
    /// traces as a proxy for the autocorrelation of the estimator's SCORE, and those are different
    /// quantities — a Newey-West estimate on the score itself asked for 1.05x where the energy's
    /// proxy asked for 1.74x.
    ///
    /// The inflation is kept anyway, because over-wide is the conservative direction for an
    /// instrument whose job is to accuse, and because one fixture is not enough evidence to swap a
    /// known-conservative heuristic for a differently-wrong one. What is NOT kept is the pretence
    /// that a coverage test covers it: this asserts the contract the factor actually has.
    #[test]
    fn a_more_correlated_chain_gets_a_wider_interval() {
        let g = crate::ising::ring(10, 1.0, 0.2);
        let beta = 0.6;
        let run = |thin: usize| {
            let mut smp = crate::gibbs::Sampler::new(&g, beta, 4);
            smp.sweeps(400, None);
            let (mut s, mut t) = (Vec::new(), Vec::new());
            for _ in 0..500 {
                smp.sweeps(thin, None);
                let st = smp.read_all(None);
                t.push(g.energy(&st));
                s.push(st);
            }
            let c = certify(&g, beta, &s, &t);
            (0.5 * (c.beta_ci.1 - c.beta_ci.0), c.tau_int)
        };
        let (wide, tau_hi) = run(1);
        let (narrow, tau_lo) = run(20);
        assert!(
            tau_hi > tau_lo,
            "the fixture must actually differ in correlation: tau {tau_hi:.2} vs {tau_lo:.2}"
        );
        assert!(
            wide > 1.2 * narrow,
            "an unthinned chain (tau {tau_hi:.2}) got a half-width of {wide:.5} against {narrow:.5} \
             for a thinned one (tau {tau_lo:.2}); correlated draws carry less information and the \
             interval has to say so"
        );
    }

    #[test]
    fn a_correct_run_passes() {
        let g = crate::ising::ring(10, 1.0, 0.2);
        let (s, t) = run(&g, 0.6, 10, 500, 6000, 1);
        let c = certify(&g, 0.6, &s, &t);
        assert!(c.passed(), "a correct sampler should certify clean:\n{c}");
        assert!((c.beta_eff - 0.6).abs() < 0.05, "beta_eff {} off", c.beta_eff);
    }

    #[test]
    fn a_sampler_at_the_wrong_temperature_is_caught() {
        // Break mode 1. The sampler is internally consistent and returns perfectly good samples --
        // of the wrong distribution. Nothing but beta_eff catches this.
        let g = crate::ising::ring(10, 1.0, 0.2);
        let (s, t) = run(&g, 1.4, 10, 500, 6000, 2);
        let c = certify(&g, 0.6, &s, &t); // claims 0.6, actually ran at 1.4
        assert!(!c.passed(), "a wrong temperature must be caught");
        assert!(
            c.findings.iter().any(|f| matches!(f, Finding::BetaMismatch { .. })),
            "expected a BetaMismatch, got {:?}",
            c.findings
        );
        assert!((c.beta_eff - 1.4).abs() < 0.1, "should recover the true beta, got {}", c.beta_eff);
    }

    #[test]
    fn correlated_draws_are_caught() {
        // Break mode 2, at the temperature where it actually happens. A 12x12 lattice at beta 0.44
        // sits essentially on the 2D Ising critical point (beta_c = ln(1+sqrt2)/2 ~ 0.4407), where
        // correlation length diverges and single-spin Glauber crawls. Measured tau ~ 62 even with
        // 20 sweeps between draws.
        //
        // Checked across several seeds on purpose. At criticality the autocorrelation is itself
        // strongly seed-dependent -- a marginal configuration measured tau 62 on one seed and 18 on
        // another -- so a single-seed assertion here would be a coin flip dressed as a test.
        // Recording every sweep puts the case far enough from the boundary to be unambiguous.
        let g = crate::ising::lattice2d(12, 1.0);
        for seed in 1..=3 {
            let (s, t) = run(&g, 0.44, 1, 500, 3000, seed);
            let c = certify(&g, 0.44, &s, &t);
            assert!(
                c.findings.iter().any(|f| matches!(f, Finding::Undermixed { .. })),
                "seed {seed}: critical slowing down must be flagged: {c}"
            );
            assert!(c.ess < c.draws as f64 / 10.0, "seed {seed}: ess too high: {c}");
        }
    }

    #[test]
    fn thinning_repairs_what_correlation_broke() {
        // The finding has to be actionable, so the prescribed fix must actually work.
        let g = crate::ising::lattice2d(24, 1.0);
        let tight = { let (s, t) = run(&g, 0.7, 1, 0, 600, 3); certify(&g, 0.7, &s, &t) };
        let fixed = { let (s, t) = run(&g, 0.7, 50, 500, 600, 3); certify(&g, 0.7, &s, &t) };
        assert!(!tight.passed(), "the unfixed run should be flagged");
        assert!(fixed.passed(), "burning in and thinning should clear it: {fixed}");
        assert!(
            fixed.ess > tight.ess * 10.0,
            "ess should improve by an order of magnitude: {:.0} -> {:.0}",
            tight.ess, fixed.ess
        );
    }

    #[test]
    fn an_unburned_chain_is_caught() {
        // Break mode 3. A 24x24 lattice below its critical temperature, sampled from the first
        // sweeps of a randomly initialised chain: it is still coarsening out domains and its
        // magnetization is travelling, so the draws describe where it started rather than the
        // model. Measured (2026-09-28, 2,000 draws, seed 4) tau 44.7 against 0.87 once burned in.
        //
        // Note the 1D ring will NOT do for this test, and an earlier version of it wrongly used
        // one: 1D Ising has no ordered phase, so a ring equilibrates fast at every temperature and
        // certifying it clean is correct behaviour, not a missed detection.
        //
        // 2,000 draws on both halves, so burn-in is the only difference. It was 600, and 600 draws
        // of this chain burned in are about 800 autocorrelation times -- too few for its `tau` to be
        // more than a lower bound, which the certificate now reports (`TauLowerBound`), so the
        // clean half could not pass on those draws. And the unburned half must be caught for being
        // UNDERMIXED, not merely for being short: a `!passed()` alone would now be satisfied by the
        // lower-bound finding whether or not the drift was seen.
        let g = crate::ising::lattice2d(24, 1.0);
        let (s, t) = run(&g, 0.7, 1, 0, 2_000, 4);
        let c = certify(&g, 0.7, &s, &t);
        assert!(
            c.findings.iter().any(|f| matches!(f, Finding::Undermixed { .. })),
            "an unburned coarsening chain must be reported undermixed, not just short:\n{c}"
        );

        let (s2, t2) = run(&g, 0.7, 1, 500, 2_000, 4);
        let c2 = certify(&g, 0.7, &s2, &t2);
        assert!(c2.passed(), "burning in should clear it:\n{c2}");
    }

    #[test]
    fn the_convergence_check_sees_a_drifting_trace() {
        // THIS TEST USED TO CALL NEITHER `certify` NOR ANYTHING IT TESTS.
        //
        // It built two synthetic traces and then asserted only that its own fixtures behaved --
        // that the drifting one drifted and the steady one did not. It never constructed a
        // `Certificate`, never mentioned `Finding::NotConverged`, and never touched the
        // standard-error inflation or the z > 4 threshold it is named for. It was green for every
        // possible implementation of the check, including no implementation at all.
        //
        // It drives `certify` now, and requires the finding to appear on a drifting chain and to
        // stay away from a steady one. The graph is small enough that everything else in the
        // certificate is computable, so a failure here is about convergence and not about setup.
        // THE FIXTURE TOOK THREE ATTEMPTS AND EACH FAILURE WAS INFORMATIVE, so they are recorded.
        //
        // (1) A hand-built ramp of independent draws: `NotConverged` did not fire, correctly -- a
        //     ramp is maximally autocorrelated, `tau_int` came out at 210, and the standard-error
        //     inflation the check applies swallowed the gap. The check was right, the fixture was a
        //     drift no honest statistic should call significant.
        // (2) A real chain on a 4x4 lattice: sixteen spins equilibrate in a few sweeps, so there is
        //     no drift to find.
        // (3) A real chain on 16x16: the transient finishes inside the first third of the window, so
        //     `early` is already at -0.96 and there is nothing left to compare.
        //
        // What works is a lattice big enough that coarsening is SLOW relative to the window: 32x32
        // just below the critical point, sampled from a random start with no burn-in, over a window
        // short enough that the transient spans it. early -0.27 -> late -0.87.
        let g = crate::ising::lattice2d(32, 1.0);
        let n = 200;

        // A REAL CHAIN, not a synthetic ramp. The first attempt at this test built the drift by
        // hand -- independent draws whose bias ramped upward -- and `NotConverged` did not fire,
        // correctly: a hand-built ramp is maximally autocorrelated, `tau_int` came out at 210, and
        // the standard-error inflation that the check applies swallowed the gap. The check was
        // right and the fixture was wrong.
        //
        // So this is what the module doc describes instead: a lattice below its critical
        // temperature, sampled from a random start with NO BURN-IN, coarsening out domains and
        // travelling from disorder toward saturation while it is being sampled.
        let mut smp = crate::gibbs::Sampler::new(&g, 0.5, 7);
        let drifting: Vec<Vec<i8>> = (0..n)
            .map(|_| {
                smp.sweep(None);
                smp.s.clone()
            })
            .collect();
        let trace_d: Vec<f64> = drifting
            .iter()
            .map(|s| s.iter().map(|&x| x as f64).sum::<f64>() / g.n as f64)
            .collect();
        let cert = certify(&g, 0.5, &drifting, &trace_d);
        let found = cert
            .findings
            .iter()
            .any(|f| matches!(f, Finding::NotConverged { .. }));
        assert!(
            found,
            "a chain travelling from disorder to saturation must be reported as not converged; \
             findings were {:?}",
            cert.findings
        );
        // And the finding must carry numbers a reader can act on, not just fire.
        if let Some(Finding::NotConverged { early, late, sigma }) = cert
            .findings
            .iter()
            .find(|f| matches!(f, Finding::NotConverged { .. }))
        {
            assert!(late.abs() > early.abs(), "it leaves disorder: early {early} late {late}");
            assert!(*sigma > 4.0, "and it must clear the threshold it uses: z = {sigma}");
        }

        // A steady chain must NOT trip it, or the check is an alarm that is always on.
        let mut warm = crate::gibbs::Sampler::new(&g, 0.5, 11);
        warm.sweeps(40_000, None); // burnt in, which is the whole difference
        let steady: Vec<Vec<i8>> = (0..n)
            .map(|_| {
                warm.sweep(None);
                warm.s.clone()
            })
            .collect();
        let trace_s: Vec<f64> = steady
            .iter()
            .map(|s| s.iter().map(|&x| x as f64).sum::<f64>() / g.n as f64)
            .collect();
        let clean = certify(&g, 0.5, &steady, &trace_s);
        assert!(
            !clean.findings.iter().any(|f| matches!(f, Finding::NotConverged { .. })),
            "a stationary chain must not be reported as drifting; findings were {:?}",
            clean.findings
        );
    }

    #[test]
    fn the_distributional_gate_fires_on_noise_rather_than_switching_itself_off() {
        // `AboveNoiseFloor` appeared in the enum, in Display, and at one push site -- and in NO
        // test, at any n. That mattered, because the floor `0.5*sqrt(2^n/ess)` passes 1 as n grows,
        // and TV can never exceed 1, so the comparison became unsatisfiable and the gate went
        // quiet on exactly the models it is for. `passed()` counted the silence as a pass.
        //
        // Two sizes on purpose: n=9 is where the floor is meaningful, n=16 at the same draw count
        // is where it used to go inert.
        let mut rng = 0x2545F4914F6CDD1Du64;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        for (side, draws) in [(3usize, 4000usize), (4, 4000)] {
            let g = crate::ising::lattice2d(side, 1.0);
            // Samples with no relation whatever to the model.
            let samples: Vec<Vec<i8>> = (0..draws)
                .map(|_| {
                    let r = next();
                    (0..g.n).map(|b| if r >> (b % 64) & 1 == 1 { 1 } else { -1 }).collect()
                })
                .collect();
            let trace: Vec<f64> = samples.iter().map(|s| g.energy(s)).collect();
            let c = certify(&g, 0.9, &samples, &trace);
            assert!(!c.passed(), "n={} pure noise must not pass", g.n);
            // Either it caught the discrepancy, or it said it could not look. Never silence.
            let spoke = c.findings.iter().any(|f| {
                matches!(f, Finding::AboveNoiseFloor { .. } | Finding::TooFewSamples { .. })
            });
            assert!(spoke, "n={} said nothing about the distribution: {:?}", g.n, c.findings);
            if let Some(floor) = c.noise_floor {
                assert!(
                    floor < 1.0 || c.findings.iter().any(|f| matches!(f, Finding::TooFewSamples { .. })),
                    "n={}: floor {floor} is vacuous and nothing said so",
                    g.n
                );
            }
        }
    }

    #[test]
    fn pure_noise_is_caught() {
        // The random-noise oracle: samples with no relation to the model at all. If a certificate
        // cannot reject this, it cannot reject anything.
        let g = crate::ising::ring(10, 1.0, 0.3);
        let mut rng = Pcg::new(9, 0);
        let samples: Vec<Vec<i8>> = (0..4000)
            .map(|_| (0..g.n).map(|_| if rng.f64() < 0.5 { 1 } else { -1 }).collect())
            .collect();
        let trace: Vec<f64> = samples.iter().map(|s| g.energy(s)).collect();
        let c = certify(&g, 1.0, &samples, &trace);
        assert!(!c.passed(), "uniform noise must never certify as Boltzmann:\n{c}");
        // and it is caught for the right reason: noise is infinite temperature
        assert!(c.beta_eff.abs() < 0.15, "noise should fit beta near 0, got {}", c.beta_eff);
    }

    #[test]
    fn the_noise_floor_is_reported_beside_the_distance() {
        // The rule the whole project runs on: never quote a distance without its floor.
        let g = crate::ising::ring(8, 1.0, 0.0);
        let (s, t) = run(&g, 0.5, 8, 400, 5000, 6);
        let c = certify(&g, 0.5, &s, &t);
        assert!(c.tv_exact.is_some() && c.noise_floor.is_some());
        assert!(c.noise_floor.unwrap() > 0.0);
    }

    #[test]
    fn too_few_samples_says_so_rather_than_guessing() {
        let g = crate::ising::ring(8, 1.0, 0.0);
        let (s, t) = run(&g, 1.0, 1, 10, 8, 7);
        let c = certify(&g, 1.0, &s, &t);
        assert_eq!(c.findings, vec![Finding::TooFewSamples { draws: 8 }]);
    }

    #[test]
    fn tau_int_recovers_a_known_correlation() {
        // An AR(1) process with coefficient p has tau_int = (1+p)/(2(1-p)); if the estimator cannot
        // recover that, it cannot be trusted on a real chain.
        let mut rng = Pcg::new(3, 0);
        for &p in &[0.0f64, 0.5, 0.8] {
            let mut x = 0.0;
            let trace: Vec<f64> = (0..200_000)
                .map(|_| {
                    let g = (-2.0 * rng.f64().max(1e-12).ln()).sqrt()
                        * (core::f64::consts::TAU * rng.f64()).cos();
                    x = p * x + (1.0 - p * p).sqrt() * g;
                    x
                })
                .collect();
            let want = (1.0 + p) / (2.0 * (1.0 - p));
            let got = tau_int(&trace);
            assert!((got - want).abs() / want < 0.25, "p={p}: got {got}, want {want}");
        }
    }

    /// CLOSED FORM, TWO MODES: the sum of two independent AR(1) processes with variance fractions
    /// `a_f + a_s = 1` has `rho(k) = a_f p_f^k + a_s p_s^k` exactly, so
    /// `tau = 1/2 + a_f p_f/(1-p_f) + a_s p_s/(1-p_s)` with no simulation in it. A 90% fast mode
    /// at `p_f = 0.3` plus a 10% slow mode at `p_s = 0.997` is the shape the exact operator found
    /// on a frustrated grid, and it is exactly the shape that closes Sokal's window early: `tau(W)`
    /// is about 1.8 when `W = 9 >= 5 tau(W)`, and the slow mode -- 33 of the 34 -- is never summed.
    ///
    /// Asserted three ways on the SAME two-million-draw trace. Sokal's window ([`tau_int_sokal`],
    /// the crate's `tau_int` until 2026-09-28) must land far below the truth -- the defect, pinned,
    /// so it stays measured. Geyer's sequence ([`tau_int`], the crate's estimator now) must land
    /// near it: two AR(1)s are a reversible spectrum, every pair sum is positive, and nothing stops
    /// the sum before the slow mode. Batch means must land near it too, and the entry point must
    /// NOT report a disagreement, because on a reversible spectrum there is none. And the
    /// single-mode control: on the fast process alone all three agree with `(1+p)/(2(1-p))`.
    #[test]
    fn a_small_slow_mode_closes_sokal_early_and_geyer_and_batch_means_see_it() {
        let (a_f, p_f, a_s, p_s) = (0.9f64, 0.3f64, 0.1f64, 0.997f64);
        let exact = 0.5 + a_f * p_f / (1.0 - p_f) + a_s * p_s / (1.0 - p_s);
        assert!((exact - 34.13).abs() < 0.05, "closed form {exact}");
        let n = 2_000_000usize;
        let mut rng = Pcg::new(77, 5);
        let mut gauss = || {
            (-2.0 * rng.f64().max(1e-12).ln()).sqrt() * (core::f64::consts::TAU * rng.f64()).cos()
        };
        let (mut xf, mut xs) = (0.0f64, 0.0f64);
        let mut trace = Vec::with_capacity(n);
        let mut fast_only = Vec::with_capacity(n);
        // The first 20,000 draws are discarded so the slow mode is stationary; n are kept.
        for _ in 0..(n + 20_000) {
            xf = p_f * xf + (1.0 - p_f * p_f).sqrt() * gauss();
            xs = p_s * xs + (1.0 - p_s * p_s).sqrt() * gauss();
            trace.push(a_f.sqrt() * xf + a_s.sqrt() * xs);
            fast_only.push(xf);
        }
        let trace = &trace[20_000..];
        let fast_only = &fast_only[20_000..];

        let sokal = tau_int_sokal(trace);
        let batch = tau_int_batch(trace, 20);
        assert!(
            sokal < 0.3 * exact,
            "Sokal must resolvably truncate the slow mode here: {sokal:.2} against exact {exact:.2}"
        );
        assert!(
            batch > 0.4 * exact && batch < 1.8 * exact,
            "batch means over 20 batches of {} draws should sit near {exact:.1}: {batch:.1}",
            trace.len() / 20
        );
        assert!(batch > 2.0 * sokal, "the window and batch means must resolvably disagree here");
        let geyer = tau_int(trace);
        assert!(
            geyer > 0.7 * exact && geyer < 1.4 * exact,
            "Geyer's sequence must sum the slow mode the window cut off: {geyer:.2} against {exact:.2}"
        );
        let est = tau_estimate(&[trace]);
        assert!(
            !est.truncated() && est.unresolved.is_none(),
            "on a reversible spectrum Geyer and batch means agree, and nothing may be reported: {est:?}"
        );
        assert!(est.tau >= geyer && est.tau > 0.7 * exact, "the carried value is the larger: {est:?}");

        // The control: one mode, every estimator agrees with the closed form.
        let want_f = (1.0 + p_f) / (2.0 * (1.0 - p_f));
        let (s1, b1, g1) = (tau_int_sokal(fast_only), tau_int_batch(fast_only, 20), tau_int(fast_only));
        assert!((s1 - want_f).abs() / want_f < 0.1, "single mode, Sokal {s1} vs {want_f}");
        assert!((b1 - want_f).abs() / want_f < 0.4, "single mode, batch {b1} vs {want_f}");
        assert!((g1 - want_f).abs() / want_f < 0.1, "single mode, Geyer {g1} vs {want_f}");
    }

    /// The 3x3 +-J glass with small random fields that the audit's exact sequences were taken on,
    /// built exactly as `autocorr`'s own `grid_glass(3, 3, 11)` builds it.
    fn glass3x3() -> Graph {
        let mut b = crate::graph::GraphBuilder::new(9);
        let mut rng = Pcg::new(11, 0x6A);
        for y in 0..3 {
            for x in 0..3 {
                let i = y * 3 + x;
                if x + 1 < 3 {
                    b.couple(i, i + 1, if rng.f64() < 0.5 { -1.0 } else { 1.0 });
                }
                if y + 1 < 3 {
                    b.couple(i, i + 3, if rng.f64() < 0.5 { -1.0 } else { 1.0 });
                }
            }
        }
        for i in 0..9 {
            b.bias(i, (rng.f64() - 0.5) * 0.4);
        }
        b.build()
    }

    /// The EXACT autocorrelation `rho(0..=lags)` of `obs` under `kernel` at stationarity, by
    /// repeated application of the kernel's operator to the centred observable: no sampling noise
    /// anywhere, so what an estimator reads from it is the estimator's own error.
    fn exact_rho(
        g: &Graph,
        beta: f64,
        kernel: crate::autocorr::Kernel,
        obs: impl Fn(&[i8]) -> f64,
        lags: usize,
    ) -> Vec<f64> {
        use crate::autocorr::{apply, boltzmann, spins};
        let pi = boltzmann(g, beta).expect("enumerable");
        let f: Vec<f64> = (0..1usize << g.n).map(|x| obs(&spins(x, g.n))).collect();
        let mean: f64 = pi.iter().zip(&f).map(|(p, v)| p * v).sum();
        let e: Vec<f64> = f.iter().map(|v| v - mean).collect();
        let c0: f64 = pi.iter().zip(&e).map(|(p, x)| p * x * x).sum();
        let mut rho = Vec::with_capacity(lags + 1);
        rho.push(1.0);
        let mut v = e.clone();
        for _ in 0..lags {
            v = apply(g, beta, kernel, &v);
            rho.push(pi.iter().zip(&e).zip(&v).map(|((p, x), y)| p * x * y).sum::<f64>() / c0);
        }
        rho
    }

    /// THE DEFECT, ON EXACT SEQUENCES, where no sampling noise can be blamed. Sokal's window reads
    /// the 3x3 glass's chromatic-sweep energy at `beta = 1` as 2.978 against an exact 5.4205 --
    /// 45% low, and the larger of energy and magnetisation, which is what a certificate carries,
    /// is the energy here, so the certificate was 45% low too -- because the window closes at lag
    /// 15 while a slow mode is still correlated. Geyer's sequence reads the same sequence exactly.
    /// And a case where the window errs on a NEGATIVE lobe: a sequential sweep of the
    /// antiferromagnetic 8-ring in a field, whose magnetisation is antithetic, exact 0.9199, which
    /// the window reads 19% low and Geyer within 1%.
    #[test]
    fn geyer_reads_on_exact_sequences_what_the_window_truncates() {
        use crate::autocorr::{tau_int_fundamental, Kernel};
        let g = glass3x3();
        let energy = |s: &[i8]| g.energy(s);
        let exact = tau_int_fundamental(&g, 1.0, Kernel::ChromaticGibbs, energy).unwrap().tau_int;
        assert!((exact - 5.4205).abs() < 1e-3, "the fixture's exact energy tau is 5.4205: {exact}");
        let rho = exact_rho(&g, 1.0, Kernel::ChromaticGibbs, energy, 3_000);
        assert!(rho[3_000].abs() < 1e-12, "3,000 lags must reach the tail: {}", rho[3_000]);
        let sokal = sokal_window(3_000, |k| rho[k]);
        assert!((sokal - 2.978).abs() < 2e-3, "the window reads 2.978 here, measured: {sokal}");
        let geyer = geyer_initial_monotone(3_000, |k| rho[k]);
        assert!(
            (geyer / exact - 1.0).abs() < 1e-4,
            "Geyer's sequence on the exact sequence must be the exact value: {geyer} vs {exact}"
        );

        let ring = crate::ising::ring(8, -1.0, 0.3);
        let mag = |s: &[i8]| s.iter().map(|&v| f64::from(v)).sum::<f64>();
        let exact = tau_int_fundamental(&ring, 1.5, Kernel::SequentialGibbs, mag).unwrap().tau_int;
        assert!((exact - 0.9199).abs() < 1e-3, "exact magnetisation tau 0.9199: {exact}");
        let rho = exact_rho(&ring, 1.5, Kernel::SequentialGibbs, mag, 2_000);
        assert!(rho.iter().any(|&r| r < -0.01), "the fixture must actually have a negative lobe");
        let sokal = sokal_window(2_000, |k| rho[k]);
        assert!(sokal < 0.85 * exact, "the window must read the lobe low (-19%): {sokal} vs {exact}");
        let geyer = geyer_initial_monotone(2_000, |k| rho[k]);
        assert!((geyer / exact - 1.0).abs() < 0.02, "Geyer within 1% here: {geyer} vs {exact}");
    }

    /// GEYER ALONE IS NOT CONSERVATIVE ON A NON-REVERSIBLE CHAIN, and the long-batch cross-check
    /// is what catches it -- the reason [`tau_estimate`] carries the larger of the two.
    ///
    /// In the tree: a sequential sweep of the antiferromagnetic 8-ring at `beta = 2`, observed
    /// through `m + 0.001 m_stag`, where the uniform magnetisation's negative pair comes before the
    /// staggered magnetisation's slow positive ones: Geyer's sequence reads the EXACT sequence 7.7%
    /// low. In closed form, a fast rotation beside a slow real mode,
    /// `rho(k) = 0.8 * 0.9^k cos(pi k / 2) + 0.2 * 0.99^k` with exact `tau` 19.94: the second pair
    /// sum is negative and Geyer stops at lag 2, reading 0.698 -- 96.5% low. A process with exactly
    /// that autocorrelation (the real part of a complex AR(1) plus an AR(1)) is simulated, and the
    /// entry point must carry batch means' value, not Geyer's, and must say they disagree.
    #[test]
    fn geyer_alone_under_reads_a_non_reversible_chain_and_the_batch_cross_check_is_carried() {
        use crate::autocorr::{tau_int_fundamental, Kernel};
        let ring = crate::ising::ring(8, -1.0, 0.0);
        let f = |s: &[i8]| {
            let m: f64 = s.iter().map(|&v| f64::from(v)).sum();
            let stag: f64 = s.iter().enumerate().map(|(i, &v)| if i % 2 == 0 { f64::from(v) } else { -f64::from(v) }).sum();
            m + 0.001 * stag
        };
        let exact = tau_int_fundamental(&ring, 2.0, Kernel::SequentialGibbs, f).unwrap().tau_int;
        assert!((exact - 2.46501).abs() < 1e-3, "exact tau 2.46501: {exact}");
        let rho = exact_rho(&ring, 2.0, Kernel::SequentialGibbs, f, 60_000);
        let geyer = geyer_initial_monotone(60_000, |k| rho[k]);
        assert!(
            geyer < 0.95 * exact && geyer > 0.85 * exact,
            "Geyer alone must read this in-tree non-reversible sequence about 7.7% low: {geyer} vs {exact}"
        );

        let (a, r, w, lam) = (0.8f64, 0.9f64, core::f64::consts::FRAC_PI_2, 0.99f64);
        // 1/2 + a Re(z / (1 - z)) + (1 - a) lam / (1 - lam), z = r e^{iw}.
        let (zr, zi) = (r * w.cos(), r * w.sin());
        let den = (1.0 - zr).powi(2) + zi * zi;
        let exact = 0.5 + a * (zr * (1.0 - zr) - zi * zi) / den + (1.0 - a) * lam / (1.0 - lam);
        assert!((exact - 19.942).abs() < 1e-3, "closed form {exact}");
        let closed = |k: usize| a * r.powi(k as i32) * (w * k as f64).cos() + (1.0 - a) * lam.powi(k as i32);
        let geyer = geyer_initial_monotone(20_000, closed);
        assert!((geyer - 0.698).abs() < 1e-3, "Geyer stops at lag 2 and reads 0.698: {geyer}");

        let n = 200_000usize;
        let mut rng = Pcg::new(5, 0x2A);
        let mut gauss = || {
            (-2.0 * rng.f64().max(1e-12).ln()).sqrt() * (core::f64::consts::TAU * rng.f64()).cos()
        };
        let (cr, ci) = (r * w.cos(), r * w.sin());
        let s = ((1.0 - r * r) / 2.0).sqrt();
        let (mut zre, mut zim, mut y) = (0.0f64, 0.0f64, 0.0f64);
        let mut x = Vec::with_capacity(n);
        for t in 0..(n + 5_000) {
            let (nr, ni) = (cr * zre - ci * zim + s * gauss(), cr * zim + ci * zre + s * gauss());
            zre = nr;
            zim = ni;
            y = lam * y + (1.0 - lam * lam).sqrt() * gauss();
            if t >= 5_000 {
                x.push((2.0 * a).sqrt() * zre + (1.0 - a).sqrt() * y);
            }
        }
        let alone = tau_int(&x);
        assert!(alone < 0.1 * exact, "Geyer alone on the simulated trace: {alone} against {exact}");
        let est = tau_estimate(&[&x]);
        assert!(est.truncated(), "the entry point must report the disagreement: {est:?}");
        assert!(
            est.tau >= est.batch && est.tau > 0.4 * exact,
            "and carry the batch value, not Geyer's: {est:?} against {exact}"
        );
    }

    /// THE CERTIFICATE CARRIES THE LARGER tau AND SAYS WHY. Real draws from a hot 3x3 glass supply
    /// the configurations (their magnetisation mixes in about a sweep), and the scalar observable
    /// the caller hands in is the non-reversible closed form above -- Geyer's sequence reads it at
    /// 3.5% of the truth and long-batch means near it. The certificate must report `TauTruncated`,
    /// carry at least the batch value, and give the smaller `ess`. With the chain's own energy as
    /// the observable, the same draws must NOT trigger it: one mode, both estimators agree. Both
    /// halves, so a cross-check that fired on everything would fail here.
    #[test]
    fn the_certificate_reports_a_truncated_sum_and_carries_the_larger_tau() {
        let g = glass3x3();
        let draws = 200_000usize;
        let mut s = Sampler::new(&g, 0.4, 21);
        s.sweeps(2_000, None);
        let mut samples = Vec::with_capacity(draws);
        let mut energy = Vec::with_capacity(draws);
        for _ in 0..draws {
            s.sweep(None);
            samples.push(s.s.clone());
            energy.push(g.energy(&s.s));
        }
        let hot = certify(&g, 0.4, &samples, &energy);
        assert!(
            !hot.findings.iter().any(|f| matches!(f, Finding::TauTruncated { .. })),
            "a single-mode chain must not trigger the cross-check: {hot}"
        );

        let (a, r, lam) = (0.8f64, 0.9f64, 0.99f64);
        let exact = 19.942;
        let mut rng = Pcg::new(6, 0x2B);
        let mut gauss = || {
            (-2.0 * rng.f64().max(1e-12).ln()).sqrt() * (core::f64::consts::TAU * rng.f64()).cos()
        };
        let sd = ((1.0 - r * r) / 2.0).sqrt();
        let (mut zre, mut zim, mut y) = (0.0f64, 0.0f64, 0.0f64);
        let mut obs = Vec::with_capacity(draws);
        for t in 0..(draws + 5_000) {
            // z <- r i z + noise: a quarter turn per draw.
            let (nr, ni) = (-r * zim + sd * gauss(), r * zre + sd * gauss());
            zre = nr;
            zim = ni;
            y = lam * y + (1.0 - lam * lam).sqrt() * gauss();
            if t >= 5_000 {
                obs.push((2.0 * a).sqrt() * zre + (1.0 - a).sqrt() * y);
            }
        }
        let c = certify(&g, 0.4, &samples, &obs);
        let fired = c.findings.iter().find_map(|f| match f {
            Finding::TauTruncated { geyer, batch } => Some((*geyer, *batch)),
            _ => None,
        });
        let (geyer, batch) = fired.expect("the rotating observable must trigger TauTruncated");
        assert!(batch > 2.0 * geyer);
        assert!(geyer < 0.1 * exact, "and Geyer must sit far below the exact {exact}: {geyer}");
        assert!(c.tau_int >= batch - 1e-12, "the certificate must carry the batch value: {c}");
        assert!(c.tau_int > 0.4 * exact, "which is near the truth: {} vs {exact}", c.tau_int);
        assert!(c.ess <= c.draws as f64 / (2.0 * batch) + 1e-9, "ess must be the conservative one");
    }

    /// ESS NEVER EXCEEDS THE DRAWS, and a failed sum is a finding, not a number.
    ///
    /// Geyer's sequence reads an antithetic trace below `1/2` -- an AR(1) at `-0.9` has an exact
    /// `tau` of `0.1 / 3.8 = 0.026`, "worth" 19 times its draws -- and a trace that alternates with
    /// a little noise to zero or less. Without the floor the first becomes an effective sample
    /// size larger than the chain and pulls the certificate's noise floor BELOW the independent
    /// one, which accuses correct samplers; `rhat::ess` turned the second into `N log10 N`. Checked
    /// short (no batch means) and long, through the entry point, the certificate and a sample set.
    #[test]
    fn ess_never_exceeds_the_draws_and_a_failed_sum_is_a_finding() {
        let sign = |t: usize| if t % 2 == 0 { 1.0 } else { -1.0 };
        let mut rng = Pcg::new(19, 3);
        let mut gauss = || {
            (-2.0 * rng.f64().max(1e-12).ln()).sqrt() * (core::f64::consts::TAU * rng.f64()).cos()
        };
        // Mildly antithetic and too short for batch means (b = 10 < 16): Geyer's sum is positive
        // and below 1/2 -- the exact value is (1 - 0.5) / (2 * 1.5) = 1/6 -- so the floor is the
        // only thing between it and an ess three times the draws, in both places it is applied.
        let mut x = 0.0f64;
        let mild: Vec<f64> = (0..200)
            .map(|_| {
                x = -0.5 * x + 0.75f64.sqrt() * gauss();
                x
            })
            .collect();
        let raw = tau_int_geyer_raw(&mild);
        assert!(raw > 0.0 && raw < 0.5, "the case must reach the floor: Geyer's raw sum {raw}");
        assert_eq!(tau_int(&mild), 0.5, "tau_int floors it");
        let est = tau_estimate(&[&mild]);
        assert!(est.batch.is_nan(), "200 draws are too few for batch means: {est:?}");
        assert!(est.tau == 0.5 && est.ess() == 200.0, "and so does the entry point: {est:?}");
        for n in [200usize, 20_000] {
            let mut x = 0.0f64;
            let anti: Vec<f64> = (0..n)
                .map(|_| {
                    x = -0.9 * x + (1.0 - 0.81f64).sqrt() * gauss();
                    x
                })
                .collect();
            // The exact value is 0.026, so Geyer's sum is either floored or -- on 200 draws, where
            // its noise is as large as the value -- zero or less, which is a failure and no value.
            let g = tau_int(&anti);
            assert!(g == 0.5 || g.is_nan(), "n {n}: never below the independent value: {g}");
            let est = tau_estimate(&[&anti]);
            assert!(est.tau >= 0.5 && est.ess() <= n as f64, "n {n}: {est:?}");

            let alternating: Vec<f64> = (0..n).map(|t| sign(t) + 0.5 * gauss()).collect();
            assert!(tau_int(&alternating).is_nan(), "n {n}: a non-positive sum is no value");
            let est = tau_estimate(&[&alternating]);
            assert!(est.unresolved.is_some_and(|v| v <= 0.0), "n {n}: the failure is recorded: {est:?}");
            assert!(est.tau >= 0.5 && est.ess() <= n as f64, "n {n}: and nothing optimistic carried: {est:?}");
        }

        // Through the certificate: exact independent draws for the configurations, the alternating
        // trace as the observable.
        let g = crate::ising::ring(8, 1.0, 0.2);
        let p = crate::ising::exact_boltzmann(&g, 0.5);
        let mut cdf = Vec::with_capacity(p.len());
        let mut total = 0.0;
        for &q in &p {
            total += q;
            cdf.push(total);
        }
        let draws = 20_000usize;
        let mut samples = Vec::with_capacity(draws);
        for _ in 0..draws {
            let u = rng.f64() * total;
            let k = cdf.partition_point(|&c| c < u).min(p.len() - 1);
            samples.push((0..g.n).map(|i| if (k >> i) & 1 == 1 { 1i8 } else { -1 }).collect::<Vec<i8>>());
        }
        let alternating: Vec<f64> = (0..draws).map(|t| sign(t) + 0.5 * rng.f64()).collect();
        let c = certify(&g, 0.5, &samples, &alternating);
        assert!(
            c.findings.iter().any(|f| matches!(f, Finding::TauUnresolved { .. })),
            "a failed sum must be reported: {c}"
        );
        assert!(c.tau_int >= 0.5 && c.ess <= draws as f64, "{c}");
        let iid_floor = 0.5 * ((1usize << g.n) as f64 / draws as f64).sqrt();
        assert!(c.noise_floor.unwrap() >= iid_floor - 1e-15, "the floor may not drop below the independent one: {c}");

        // And through a sample set: every estimate's ess is at most its draws.
        let energies: Vec<f64> = samples.iter().map(|s| g.energy(s)).collect();
        let set = crate::samples::SampleSet::from_chain(samples, energies, 0.5, 0, 1);
        assert!(set.chain_tau() >= 0.5);
        for i in 0..g.n {
            let e = set.mean_spin(i).unwrap();
            assert!(e.ess <= draws as f64 && e.tau_int >= 0.5, "site {i}: {e}");
        }
    }

    /// A CHAIN UNDER A THOUSAND AUTOCORRELATION TIMES IS REPORTED AS A LOWER BOUND. 300 draws of a
    /// chain thinned to near independence is 600 `tau` at most, so its `tau` is a lower bound and
    /// the certificate must say so, with the numbers it carries; 6,000 of the same are not.
    #[test]
    fn a_chain_under_a_thousand_taus_is_reported_as_a_lower_bound() {
        let g = crate::ising::ring(10, 1.0, 0.2);
        let (s, t) = run(&g, 0.6, 10, 500, 300, 1);
        let c = certify(&g, 0.6, &s, &t);
        let found = c.findings.iter().find_map(|f| match f {
            Finding::TauLowerBound { tau_int, ess, draws } => Some((*tau_int, *ess, *draws)),
            _ => None,
        });
        let (tau, ess, draws) = found.unwrap_or_else(|| panic!("300 draws must be reported short: {c}"));
        assert!((tau - c.tau_int).abs() < 1e-12 && (ess - c.ess).abs() < 1e-9 && draws == 300);
        assert!(format!("{c}").contains("lower bound"), "and say what it means: {c}");

        let (s, t) = run(&g, 0.6, 10, 500, 6_000, 1);
        let c = certify(&g, 0.6, &s, &t);
        assert!(
            !c.findings.iter().any(|f| matches!(f, Finding::TauLowerBound { .. })),
            "6,000 draws at tau near 1/2 are thousands of tau: {c}"
        );
    }

    /// The transform path and the direct sum are one estimator: on an AR(1) trace above
    /// `FFT_FROM` and on one below it they agree to `1e-10` relative, so the threshold changes
    /// the clock and nothing else.
    #[test]
    fn the_fft_path_and_the_direct_sum_agree() {
        let mut rng = crate::rng::Pcg::new(11, 5);
        for n in [600usize, 6000] {
            let mut x = Vec::with_capacity(n);
            let mut v = 0.0f64;
            for _ in 0..n {
                v = 0.95 * v + (rng.f64() - 0.5);
                x.push(v);
            }
            let (a, b) = (tau_int_by(&x, true), tau_int_by(&x, false));
            assert!((a - b).abs() < 1e-10 * b, "n {n}: fft {a} vs direct {b}");
            assert!(a > 5.0, "an AR(1) at 0.95 has tau near 20, not {a}");
        }
    }
}
