//! Discrete kernel Stein discrepancy: goodness of fit against an **unnormalised** model.
//!
//! Yang, Liu, Rao & Neville, *Goodness-of-fit Testing for Discrete Distributions via Stein
//! Discrepancy*, ICML 2018. Given draws `x_1..x_m` and a model `p`, the discrete Stein operator
//! built from single-site flips produces a kernel `kappa_p(x, y)` whose expectation under `p` is
//! exactly zero, and is positive otherwise. The whole construction touches `p` only through the
//! ratio
//!
//! ```text
//!     score_i(x) = 1 - p(flip_i x) / p(x) = 1 - exp(-beta * dE_i(x))
//! ```
//!
//! so the partition function cancels and never has to be computed.
//!
//! # Why this exists beside `certify`
//!
//! [`crate::certify::certify`] compares a sampler to the truth in total variation, which needs the
//! truth: [`crate::samples::enumerate`] materialises `2^n` states and stops near twenty spins.
//! This needs neither a normaliser nor an enumeration — one pass over the samples costs
//! `O(m^2 (n + edges))` and does not care how large `n` is. What it buys is weaker and different:
//! `certify` reports *how far* a sampler is from the model, this reports whether the gap is larger
//! than sampling noise under the null.
//!
//! # V-statistic, U-statistic, and which one the test uses
//!
//! [`Ksd::v`] is the V-statistic `(1/m^2) sum_{i,j} kappa_ij`. It is non-negative — the Stein
//! kernel is positive semi-definite whenever the base kernel is — but it is biased upward by its
//! own diagonal, `trace/m^2`, which is `O(1/m)` and does NOT vanish under the null. So "zero for a
//! correct model" is a statement about the population, not about `v` at finite `m`.
//!
//! [`Ksd::u`] drops the diagonal. It is unbiased, has mean exactly zero under the null, and is what
//! [`goodness_of_fit`] tests. Its null distribution is a weighted sum of centred chi-squares with
//! unknown weights, so the threshold is drawn by weighted bootstrap rather than tabulated.

use crate::graph::Graph;
use crate::rng::Pcg;

/// Why a discrepancy could not be computed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Invalid {
    /// There are no samples to take a discrepancy of.
    Empty,
    /// A V-statistic needs one sample; dropping the diagonal needs two.
    TooFew {
        /// Samples supplied.
        have: usize,
    },
    /// A sample has a different number of spins than the model.
    WrongWidth {
        /// Index of the offending sample.
        sample: usize,
        /// Spins that sample carries.
        found: usize,
        /// Spins the model has.
        spins: usize,
    },
    /// A sample carries something that is not `-1` or `+1`.
    NotASpin {
        /// Index of the offending sample.
        sample: usize,
        /// Site within that sample.
        site: usize,
        /// The value found there.
        value: i8,
    },
    /// `beta` is not a finite number, so no Boltzmann ratio exists.
    Beta(f64),
    /// The kernel bandwidth is not finite and positive.
    Bandwidth(f64),
    /// The score ratio `exp(-beta * dE)` overflowed: this `beta` is past what `f64` represents.
    ScoreOverflow {
        /// Index of the sample whose score blew up.
        sample: usize,
        /// Site within that sample.
        site: usize,
    },
    /// The test size must lie strictly between zero and one.
    Level(f64),
    /// A threshold cannot be read off zero bootstrap replicates.
    NoBootstraps,
}

impl core::fmt::Display for Invalid {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Invalid::Empty => write!(f, "no samples, so there is no discrepancy to take"),
            Invalid::TooFew { have } => write!(
                f,
                "{have} sample(s): the U-statistic averages over ORDERED PAIRS of distinct draws \
                 and there are none, so the unbiased statistic the test uses does not exist"
            ),
            Invalid::WrongWidth { sample, found, spins } => write!(
                f,
                "sample {sample} has {found} spins and the model has {spins}; a Stein score is one \
                 number per site of the model, and there is no site to attach the extras to"
            ),
            Invalid::NotASpin { sample, site, value } => write!(
                f,
                "sample {sample} site {site} holds {value}, and this operator is built from single \
                 flips of a two-state variable: only -1 and +1 have a flip"
            ),
            Invalid::Beta(b) => {
                write!(f, "beta is {b}; the score is exp(-beta * dE) and needs a finite exponent")
            }
            Invalid::Bandwidth(w) => write!(
                f,
                "kernel bandwidth {w} is not finite and positive; exp(-w * mismatch / n) is only \
                 positive definite for w > 0, and at w = 0 the kernel is constant and the \
                 discrepancy is identically zero for every model"
            ),
            Invalid::ScoreOverflow { sample, site } => write!(
                f,
                "the flip ratio at sample {sample} site {site} overflowed f64: beta times the flip \
                 gap exceeds ~709. Test at a lower beta, or rescale the couplings -- see \
                 `Graph::flip_gap_max` for the instance's own energy scale"
            ),
            Invalid::Level(l) => {
                write!(f, "test level {l} is not in (0, 1), so it names no rejection region")
            }
            Invalid::NoBootstraps => write!(
                f,
                "zero bootstrap replicates: the null distribution of this statistic is a weighted \
                 sum of centred chi-squares with unknown weights, so there is no tabulated \
                 threshold to fall back on"
            ),
        }
    }
}

/// Knobs for [`goodness_of_fit`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Options {
    /// Bandwidth of the exponentiated Hamming kernel `exp(-bandwidth * mismatches / n)`.
    pub bandwidth: f64,
    /// Weighted-bootstrap replicates drawn under the null to place the threshold.
    pub bootstraps: usize,
    /// Size of the test: reject when the p-value is at or below this.
    pub level: f64,
    /// Seed for the bootstrap weights, because a verdict must be reproducible.
    pub seed: u64,
}

impl Default for Options {
    /// The paper's kernel at unit bandwidth, 500 replicates, a 5% test, fixed seed.
    fn default() -> Self {
        Options { bandwidth: 1.0, bootstraps: 500, level: 0.05, seed: 0x5731 }
    }
}

/// The difference score `1 - p(flip_i s) / p(s)` at every site, which needs no normaliser.
///
/// For `p(s) ∝ exp(-beta E(s))` the ratio is `exp(-beta * dE_i)`, and `dE_i = 2 s_i f_i` is the
/// same flip energy [`crate::kernel::delta_e`] hands the samplers.
///
/// Entries can be large — the ratio is `exp(beta * gap)` when a flip lowers the energy — and
/// callers that need that checked should go through [`gram`], which refuses a non-finite score.
///
/// # Panics
///
/// If `s` is not one value per spin of `g`.
#[must_use]
pub fn difference_score(g: &Graph, beta: f64, s: &[i8]) -> Vec<f64> {
    assert_eq!(s.len(), g.n, "a state is one value per spin");
    (0..g.n)
        .map(|i| 1.0 - (-beta * crate::kernel::delta_e(g.field(i, s), s[i])).exp())
        .collect()
}

/// One entry of the Stein kernel matrix, with the model entering only through the two scores.
///
/// `u` is the per-mismatch kernel factor `exp(-bandwidth / n)`, so `k(x, y) = u^mismatches`. The
/// three flipped evaluations the operator needs are then one multiply each: flipping one
/// coordinate of `x` moves the mismatch count by exactly one, flipping the SAME coordinate of both
/// leaves the agreement pattern — and therefore the kernel — untouched.
fn stein_kernel(x: &[i8], y: &[i8], sx: &[f64], sy: &[f64], u: f64) -> f64 {
    let n = x.len();
    let mismatch = (0..n).filter(|&i| x[i] != y[i]).count();
    let k = u.powi(mismatch as i32);
    let u_inv = u.recip();
    let mut acc = 0.0;
    for i in 0..n {
        // k(flip_i x, y): one more mismatch where they agreed, one fewer where they did not.
        let k_x = if x[i] == y[i] { k * u } else { k * u_inv };
        // k(x, flip_i y) changes the same agreement, so it takes the same value.
        let k_y = k_x;
        // k(flip_i x, flip_i y): both moved, the agreement at i is what it was.
        let k_both = k;
        acc += sx[i] * sy[i] * k - sx[i] * (k - k_y) - sy[i] * (k - k_x)
            + (k - k_x - k_y + k_both);
    }
    acc
}

/// Validate the request and precompute one score vector per sample.
fn checked_scores(
    g: &Graph,
    beta: f64,
    samples: &[Vec<i8>],
    bandwidth: f64,
) -> Result<Vec<Vec<f64>>, Invalid> {
    if samples.is_empty() {
        return Err(Invalid::Empty);
    }
    if samples.len() < 2 {
        return Err(Invalid::TooFew { have: samples.len() });
    }
    if !beta.is_finite() {
        return Err(Invalid::Beta(beta));
    }
    if !(bandwidth.is_finite() && bandwidth > 0.0) {
        return Err(Invalid::Bandwidth(bandwidth));
    }
    let mut out = Vec::with_capacity(samples.len());
    for (a, s) in samples.iter().enumerate() {
        if s.len() != g.n {
            return Err(Invalid::WrongWidth { sample: a, found: s.len(), spins: g.n });
        }
        for (i, &v) in s.iter().enumerate() {
            if v != 1 && v != -1 {
                return Err(Invalid::NotASpin { sample: a, site: i, value: v });
            }
        }
        let sc = difference_score(g, beta, s);
        if let Some(i) = sc.iter().position(|v| !v.is_finite()) {
            return Err(Invalid::ScoreOverflow { sample: a, site: i });
        }
        out.push(sc);
    }
    Ok(out)
}

/// The `m x m` Stein kernel matrix of `samples` under `g` at `beta`, row-major.
///
/// # Errors
///
/// [`Invalid`] when the samples do not match the model, when `beta` or the bandwidth is not a
/// usable number, or when a flip ratio overflows `f64`.
pub fn gram(
    g: &Graph,
    beta: f64,
    samples: &[Vec<i8>],
    bandwidth: f64,
) -> Result<Vec<f64>, Invalid> {
    let sc = checked_scores(g, beta, samples, bandwidth)?;
    let m = samples.len();
    let u = (-bandwidth / g.n as f64).exp();
    let mut k = vec![0.0; m * m];
    for a in 0..m {
        for b in a..m {
            let v = stein_kernel(&samples[a], &samples[b], &sc[a], &sc[b], u);
            k[a * m + b] = v;
            k[b * m + a] = v;
        }
    }
    Ok(k)
}

/// A kernel Stein discrepancy and the two ways of averaging it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ksd {
    /// V-statistic `(1/m^2) sum_{i,j} kappa_ij`: non-negative, biased upward by `trace/m^2`.
    pub v: f64,
    /// U-statistic, the diagonal dropped: unbiased, mean exactly zero under the null, can be
    /// negative on a correct model and is the statistic [`goodness_of_fit`] tests.
    pub u: f64,
    /// Samples averaged over.
    pub samples: usize,
    /// Spins in the model, which is the length of every score vector.
    pub spins: usize,
    /// Kernel bandwidth used, since the statistic's scale depends on it.
    pub bandwidth: f64,
}

/// Kernel Stein discrepancy of `samples` against `g` at `beta`.
///
/// # Errors
///
/// [`Invalid`], as [`gram`].
pub fn ksd(g: &Graph, beta: f64, samples: &[Vec<i8>], bandwidth: f64) -> Result<Ksd, Invalid> {
    let k = gram(g, beta, samples, bandwidth)?;
    Ok(summarise(&k, samples.len(), g.n, bandwidth))
}

/// Both averages of an already-built Stein kernel matrix.
fn summarise(k: &[f64], m: usize, spins: usize, bandwidth: f64) -> Ksd {
    let total: f64 = k.iter().sum();
    let diag: f64 = (0..m).map(|i| k[i * m + i]).sum();
    Ksd {
        v: total / (m * m) as f64,
        u: (total - diag) / (m * (m - 1)) as f64,
        samples: m,
        spins,
        bandwidth,
    }
}

/// A verdict: the discrepancy, and where the null put its threshold.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Fit {
    /// The discrepancy itself.
    pub ksd: Ksd,
    /// Bootstrap p-value, `(1 + #{replicates >= u}) / (bootstraps + 1)` — never zero, because a
    /// finite bootstrap cannot certify a probability smaller than one over its own count.
    pub p_value: f64,
    /// The `1 - level` quantile of the bootstrap replicates: what [`Ksd::u`] had to beat.
    pub threshold: f64,
    /// Size the test was run at.
    pub level: f64,
    /// Replicates drawn.
    pub bootstraps: usize,
}

impl Fit {
    /// Whether the samples are rejected as draws from this model at this level.
    #[must_use]
    pub fn rejected(&self) -> bool {
        self.p_value <= self.level
    }
}

/// Test whether `samples` are draws from `g` at `beta`, thresholded by weighted bootstrap.
///
/// The replicates reweight the observed pairs by `w_i - 1/m` for multinomial `w`, which is the
/// standard bootstrap for a degenerate U-statistic (Huskova & Janssen 1993; Liu, Lee & Jordan
/// 2016, adapted to the discrete operator by Yang et al. 2018). Both `m * u` and `m` times a
/// replicate converge to the same weighted chi-square limit, so the replicates are compared with
/// [`Ksd::u`] directly.
///
/// # Errors
///
/// [`Invalid`], as [`gram`], plus [`Invalid::Level`] and [`Invalid::NoBootstraps`].
pub fn goodness_of_fit(
    g: &Graph,
    beta: f64,
    samples: &[Vec<i8>],
    opt: &Options,
) -> Result<Fit, Invalid> {
    if !(opt.level > 0.0 && opt.level < 1.0) {
        return Err(Invalid::Level(opt.level));
    }
    if opt.bootstraps == 0 {
        return Err(Invalid::NoBootstraps);
    }
    let k = gram(g, beta, samples, opt.bandwidth)?;
    let m = samples.len();
    let stat = summarise(&k, m, g.n, opt.bandwidth);

    let inv = 1.0 / m as f64;
    let mut rng = Pcg::new(opt.seed, 0x5f3e);
    let mut w = vec![0.0f64; m];
    let mut reps = Vec::with_capacity(opt.bootstraps);
    for _ in 0..opt.bootstraps {
        // Multinomial(m; uniform) counts, divided by m and centred at their mean.
        w.fill(-inv);
        for _ in 0..m {
            let idx = ((rng.f64() * m as f64) as usize).min(m - 1);
            w[idx] += inv;
        }
        // sum_{i != j} w_i w_j kappa_ij, the diagonal removed inside the row loop.
        let mut s = 0.0;
        for i in 0..m {
            let row = &k[i * m..(i + 1) * m];
            let mut t = 0.0;
            for j in 0..m {
                t += w[j] * row[j];
            }
            s += w[i] * (t - w[i] * row[i]);
        }
        reps.push(s);
    }

    let over = reps.iter().filter(|&&r| r >= stat.u).count();
    let p_value = (1 + over) as f64 / (opt.bootstraps + 1) as f64;
    reps.sort_by(f64::total_cmp);
    let rank = (((1.0 - opt.level) * opt.bootstraps as f64).ceil() as usize)
        .clamp(1, opt.bootstraps)
        - 1;
    Ok(Fit {
        ksd: stat,
        p_value,
        threshold: reps[rank],
        level: opt.level,
        bootstraps: opt.bootstraps,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::GraphBuilder;
    use crate::ising;

    /// A small dense spin glass: enumerable, frustrated, and its energy scale is O(1) so beta
    /// moves the distribution without pushing `exp(-beta dE)` anywhere near overflow.
    fn glass(n: usize, seed: u64) -> Graph {
        let mut r = Pcg::new(seed, 11);
        let mut b = GraphBuilder::new(n);
        for i in 0..n {
            b.bias(i, 0.6 * (r.f64() - 0.5));
            for j in (i + 1)..n {
                if r.f64() < 0.5 {
                    b.couple(i, j, 1.2 * (r.f64() - 0.5));
                }
            }
        }
        b.build()
    }

    /// Every state of an `n`-spin model, in the bit order `ising::exact_boltzmann` indexes by.
    fn all_states(n: usize) -> Vec<Vec<i8>> {
        (0..1usize << n)
            .map(|mask| (0..n).map(|b| if mask >> b & 1 == 1 { 1i8 } else { -1 }).collect())
            .collect()
    }

    /// Exact i.i.d. draws by inverse CDF over the enumerated Boltzmann distribution.
    ///
    /// This is the null the test must not reject: not a converged chain, not an approximation —
    /// independent draws from the exact distribution.
    fn exact_draws(g: &Graph, beta: f64, m: usize, seed: u64) -> Vec<Vec<i8>> {
        let p = ising::exact_boltzmann(g, beta);
        let mut cdf = Vec::with_capacity(p.len());
        let mut acc = 0.0;
        for &v in &p {
            acc += v;
            cdf.push(acc);
        }
        let mut r = Pcg::new(seed, 5);
        (0..m)
            .map(|_| {
                let x = r.f64() * acc;
                let (mut lo, mut hi) = (0usize, cdf.len() - 1);
                while lo < hi {
                    let mid = usize::midpoint(lo, hi);
                    if cdf[mid] < x { lo = mid + 1 } else { hi = mid }
                }
                (0..g.n).map(|b| if lo >> b & 1 == 1 { 1i8 } else { -1 }).collect()
            })
            .collect()
    }

    /// The exact population discrepancy: `sum_{x,y} q(x) q(y) kappa_p(x, y)` over EVERY state.
    ///
    /// Nothing is sampled, so this is the quantity the statistic estimates, computed exactly.
    fn population(g: &Graph, beta_model: f64, beta_q: f64, bandwidth: f64) -> f64 {
        let q = ising::exact_boltzmann(g, beta_q);
        let states = all_states(g.n);
        let sc: Vec<Vec<f64>> =
            states.iter().map(|s| difference_score(g, beta_model, s)).collect();
        let u = (-bandwidth / g.n as f64).exp();
        let mut acc = 0.0;
        for a in 0..states.len() {
            for b in 0..states.len() {
                acc += q[a] * q[b] * stein_kernel(&states[a], &states[b], &sc[a], &sc[b], u);
            }
        }
        acc
    }

    /// The score must equal the ratio of two NORMALISED probabilities, computed independently.
    ///
    /// This is the module's whole claim: `1 - exp(-beta dE)` and `1 - p(flip)/p(x)` are the same
    /// number, and the partition function that appears in the second one cancels. A sign error in
    /// `delta_e`, a missing factor of two, or beta applied once instead of twice all break here.
    #[test]
    fn the_score_is_the_exact_probability_ratio_and_needs_no_normaliser() {
        let g = glass(9, 7);
        for &beta in &[0.25, 0.8, 1.7] {
            let p = ising::exact_boltzmann(&g, beta);
            for (mask, s) in all_states(g.n).iter().enumerate() {
                let got = difference_score(&g, beta, s);
                for i in 0..g.n {
                    let want = 1.0 - p[mask ^ (1 << i)] / p[mask];
                    assert!(
                        (got[i] - want).abs() < 1e-9 * want.abs().max(1.0),
                        "beta={beta} state={mask} site={i}: {} vs {want}",
                        got[i]
                    );
                }
            }
        }
    }

    /// Stein's identity, exactly: `E_p[score_i(x) f(x) - (f(x) - f(flip_i x))] = 0` for any `f`.
    ///
    /// The operator is only a discrepancy because this holds for EVERY test function, so it is
    /// checked against a random one rather than a convenient one, on every coordinate.
    #[test]
    fn the_stein_identity_is_exact_under_the_model() {
        let g = glass(9, 3);
        let beta = 0.9;
        let p = ising::exact_boltzmann(&g, beta);
        let states = all_states(g.n);
        let mut r = Pcg::new(99, 1);
        let f: Vec<f64> = (0..states.len()).map(|_| 4.0 * r.f64() - 2.0).collect();
        let scale: f64 = p.iter().zip(&f).map(|(a, b)| a * b.abs()).sum();
        for i in 0..g.n {
            let mut acc = 0.0;
            for (mask, s) in states.iter().enumerate() {
                let sc = difference_score(&g, beta, s)[i];
                acc += p[mask] * (sc * f[mask] - (f[mask] - f[mask ^ (1 << i)]));
            }
            assert!(acc.abs() < 1e-12 * scale.max(1.0), "site {i} left a residual of {acc}");
        }
    }

    /// The population discrepancy is exactly zero when the samples' law IS the model.
    ///
    /// The double sum runs over all `2^n x 2^n` state pairs weighted by the exact Boltzmann
    /// distribution, so this is the defining property checked against itself with nothing sampled.
    #[test]
    fn the_population_discrepancy_is_zero_under_the_model() {
        let g = glass(8, 21);
        for &beta in &[0.3, 0.75, 1.4] {
            for &bw in &[0.5, 1.0, 3.0] {
                let d = population(&g, beta, beta, bw);
                assert!(d.abs() < 1e-9, "beta={beta} bandwidth={bw} gave {d}, not zero");
            }
        }
    }

    /// A wrong beta has a strictly positive population discrepancy, growing with the error.
    ///
    /// Zero-under-the-null is half a test: a statistic that is zero for everything would pass it.
    #[test]
    fn a_wrong_beta_has_a_positive_population_discrepancy() {
        let g = glass(8, 21);
        let beta = 0.75;
        // Nearest ratio first, walking outward in both directions: the discrepancy must be
        // positive everywhere and must GROW as the model's beta moves away from the samples'.
        for ratios in [[0.8, 0.6, 0.4], [1.2, 1.4, 1.6]] {
            let mut nearer = 0.0;
            for r in ratios {
                let d = population(&g, beta, beta * r, 1.0);
                assert!(d > 1e-4, "beta*{r} gave {d}, which is not distinguishable from zero");
                assert!(d > nearer, "further from the truth came out closer: {d} <= {nearer}");
                nearer = d;
            }
        }
    }

    /// The kernel is symmetric in its two arguments, as an inner product must be.
    #[test]
    fn the_stein_kernel_is_symmetric() {
        let g = glass(10, 5);
        let states = exact_draws(&g, 0.8, 12, 4);
        let sc: Vec<Vec<f64>> = states.iter().map(|s| difference_score(&g, 0.8, s)).collect();
        let u = (-1.0f64 / g.n as f64).exp();
        for a in 0..states.len() {
            for b in 0..states.len() {
                let ab = stein_kernel(&states[a], &states[b], &sc[a], &sc[b], u);
                let ba = stein_kernel(&states[b], &states[a], &sc[b], &sc[a], u);
                assert!((ab - ba).abs() < 1e-9 * ab.abs().max(1.0), "{a},{b}: {ab} vs {ba}");
            }
        }
    }

    /// The incremental Hamming update equals a kernel rebuilt from the flipped state.
    ///
    /// `stein_kernel` never materialises `flip_i x`; it multiplies by `u` or by `1/u`. That is an
    /// optimisation inside the only formula this module has, so it is checked against the
    /// definition, coordinate by coordinate, rather than assumed.
    #[test]
    fn the_flipped_kernel_values_match_a_fresh_evaluation() {
        let n = 11;
        let bw = 1.3;
        let u = (-bw / n as f64).exp();
        let fresh = |x: &[i8], y: &[i8]| {
            let d = (0..n).filter(|&i| x[i] != y[i]).count();
            (-bw * d as f64 / n as f64).exp()
        };
        let mut r = Pcg::new(17, 2);
        for _ in 0..64 {
            let x: Vec<i8> = (0..n).map(|_| r.spin(0.5)).collect();
            let y: Vec<i8> = (0..n).map(|_| r.spin(0.5)).collect();
            let k = u.powi((0..n).filter(|&i| x[i] != y[i]).count() as i32);
            assert!((k - fresh(&x, &y)).abs() < 1e-12);
            for i in 0..n {
                let mut fx = x.clone();
                fx[i] = -fx[i];
                let mut fy = y.clone();
                fy[i] = -fy[i];
                let incremental = if x[i] == y[i] { k * u } else { k / u };
                assert!((incremental - fresh(&fx, &y)).abs() < 1e-12, "flip x at {i}");
                assert!((incremental - fresh(&x, &fy)).abs() < 1e-12, "flip y at {i}");
                assert!((k - fresh(&fx, &fy)).abs() < 1e-12, "flipping both at {i}");
            }
        }
    }

    /// The V-statistic is never negative, on right models and wrong ones alike.
    ///
    /// It is a quadratic form in a positive semi-definite Gram matrix, which is what makes it a
    /// squared discrepancy rather than a signed one. A negative value would mean the kernel had
    /// lost positive definiteness — the property the whole construction rests on.
    #[test]
    fn the_v_statistic_is_never_negative() {
        let g = glass(10, 13);
        for &beta_q in &[0.2, 0.7, 1.5] {
            for &beta_p in &[0.2, 0.7, 1.5] {
                let s = exact_draws(&g, beta_q, 60, 900 + (beta_q * 10.0) as u64);
                let k = ksd(&g, beta_p, &s, 1.0).unwrap();
                assert!(k.v >= 0.0, "q={beta_q} p={beta_p} gave v = {}", k.v);
            }
        }
    }

    /// The V-statistic exceeds the U-statistic by exactly the diagonal it keeps.
    #[test]
    fn the_two_averages_differ_by_the_diagonal_and_nothing_else() {
        let g = glass(9, 31);
        let s = exact_draws(&g, 0.8, 40, 77);
        let k = gram(&g, 0.8, &s, 1.0).unwrap();
        let m = s.len();
        let stat = summarise(&k, m, g.n, 1.0);
        let diag: f64 = (0..m).map(|i| k[i * m + i]).sum();
        let want = (stat.u * (m * (m - 1)) as f64 + diag) / (m * m) as f64;
        assert!((stat.v - want).abs() < 1e-9 * stat.v.abs().max(1.0));
    }

    /// Exact draws are not rejected; draws at a wrong beta are, and the threshold is the
    /// bootstrap's.
    ///
    /// The null here is i.i.d. inverse-CDF sampling of the enumerated Boltzmann distribution — the
    /// distribution itself, not a sampler believed to have converged to it.
    #[test]
    fn exact_draws_pass_and_a_wrong_beta_is_rejected() {
        let g = glass(10, 41);
        let beta = 0.8;
        let opt = Options { bootstraps: 400, ..Options::default() };

        let good = exact_draws(&g, beta, 200, 2024);
        let fit = goodness_of_fit(&g, beta, &good, &opt).unwrap();
        assert!(!fit.rejected(), "exact draws rejected at p = {}", fit.p_value);
        assert!(fit.ksd.u <= fit.threshold, "u = {} above {}", fit.ksd.u, fit.threshold);

        for &wrong in &[0.4, 1.6] {
            let bad = exact_draws(&g, wrong, 200, 2024);
            let fit = goodness_of_fit(&g, beta, &bad, &opt).unwrap();
            assert!(fit.rejected(), "beta {wrong} against {beta} passed at p = {}", fit.p_value);
            assert!(fit.ksd.u > fit.threshold, "u = {} below {}", fit.ksd.u, fit.threshold);
            assert!(
                fit.p_value <= 2.0 / (opt.bootstraps + 1) as f64,
                "p = {} is not decisive",
                fit.p_value
            );
        }
    }

    /// The threshold moves with the data, which is what makes it a bootstrap and not a constant.
    ///
    /// Twenty independent null data sets, twenty different thresholds. A hard-coded cutoff would
    /// give one number twenty times, and would pass every other test in this file.
    #[test]
    fn the_threshold_is_read_off_the_data_not_off_a_constant() {
        let g = glass(10, 41);
        let beta = 0.8;
        let opt = Options { bootstraps: 200, ..Options::default() };
        let thresholds: Vec<f64> = (0..20)
            .map(|r| {
                let s = exact_draws(&g, beta, 120, 5000 + r);
                goodness_of_fit(&g, beta, &s, &opt).unwrap().threshold
            })
            .collect();
        let lo = thresholds.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = thresholds.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        assert!(hi > lo * 1.2, "thresholds {lo} .. {hi} barely moved; is this really resampled?");
        assert!(lo > 0.0, "a threshold at or below zero rejects half the correct models");
    }

    /// Under the null the test rejects at about the rate it was asked to.
    ///
    /// Forty independent exact data sets at a 20% level: the count is Binomial(40, 0.2), mean 8,
    /// sd 2.5, so a bound of 18 is four sd out and fails on a broken calibration rather than on
    /// bad luck. The lower bound matters too — a test that never rejects has no threshold at all,
    /// and would sail through `exact_draws_pass_and_a_wrong_beta_is_rejected`'s first half.
    #[test]
    fn the_null_rejection_rate_matches_the_level() {
        let g = glass(9, 61);
        let beta = 0.7;
        let opt = Options { bootstraps: 300, level: 0.2, ..Options::default() };
        let rejects = (0..40)
            .filter(|&r| {
                let s = exact_draws(&g, beta, 100, 31_000 + r);
                goodness_of_fit(&g, beta, &s, &opt).unwrap().rejected()
            })
            .count();
        assert!(rejects <= 18, "{rejects} of 40 exact data sets rejected at level 0.2");
        assert!(rejects >= 2, "{rejects} of 40: a test that never fires has no threshold");
    }

    /// Power grows as the model moves away from the law the samples came from.
    #[test]
    fn power_grows_with_the_error_in_beta() {
        let g = glass(10, 41);
        let beta = 0.8;
        let opt = Options { bootstraps: 300, ..Options::default() };
        let s = exact_draws(&g, beta, 150, 808);
        let mut last = f64::NEG_INFINITY;
        for &r in &[1.0, 1.15, 1.35, 1.6] {
            let u = goodness_of_fit(&g, beta * r, &s, &opt).unwrap().ksd.u;
            assert!(u > last, "the statistic fell going from the truth outward at ratio {r}");
            last = u;
        }
    }

    /// A discrepancy is refused rather than guessed when the request does not name a model.
    #[test]
    fn malformed_requests_are_refused_by_name() {
        let g = glass(6, 1);
        let s = exact_draws(&g, 0.5, 4, 1);
        assert_eq!(ksd(&g, 0.5, &[], 1.0), Err(Invalid::Empty));
        assert_eq!(ksd(&g, 0.5, &s[..1], 1.0), Err(Invalid::TooFew { have: 1 }));
        // `Beta(NaN)` never equals itself, so this one is matched rather than compared.
        assert!(matches!(ksd(&g, f64::NAN, &s, 1.0), Err(Invalid::Beta(b)) if b.is_nan()));
        assert_eq!(ksd(&g, f64::INFINITY, &s, 1.0), Err(Invalid::Beta(f64::INFINITY)));
        assert_eq!(ksd(&g, 0.5, &s, 0.0), Err(Invalid::Bandwidth(0.0)));
        assert_eq!(ksd(&g, 0.5, &s, -1.0), Err(Invalid::Bandwidth(-1.0)));

        let mut short = s.clone();
        short[2].pop();
        assert_eq!(
            ksd(&g, 0.5, &short, 1.0),
            Err(Invalid::WrongWidth { sample: 2, found: 5, spins: 6 })
        );

        let mut zero = s.clone();
        zero[1][3] = 0;
        assert_eq!(
            ksd(&g, 0.5, &zero, 1.0),
            Err(Invalid::NotASpin { sample: 1, site: 3, value: 0 })
        );

        // 1e9 * a flip gap of order one is far past exp()'s range, and an infinite score would
        // otherwise propagate a NaN statistic that compares false against every threshold.
        assert!(matches!(ksd(&g, 1e9, &s, 1.0), Err(Invalid::ScoreOverflow { .. })));

        let bad_level = Options { level: 0.0, ..Options::default() };
        assert_eq!(goodness_of_fit(&g, 0.5, &s, &bad_level), Err(Invalid::Level(0.0)));
        let none = Options { bootstraps: 0, ..Options::default() };
        assert_eq!(goodness_of_fit(&g, 0.5, &s, &none), Err(Invalid::NoBootstraps));

        // Every message says what was sent and what the model expects.
        assert!(Invalid::TooFew { have: 1 }.to_string().contains("ORDERED PAIRS"));
    }

    /// Same seed, same verdict.
    #[test]
    fn the_verdict_is_reproducible_from_its_seed() {
        let g = glass(9, 71);
        let s = exact_draws(&g, 0.6, 60, 12);
        let opt = Options { bootstraps: 120, ..Options::default() };
        let a = goodness_of_fit(&g, 0.6, &s, &opt).unwrap();
        let b = goodness_of_fit(&g, 0.6, &s, &opt).unwrap();
        assert_eq!(a, b);
        let c = goodness_of_fit(&g, 0.6, &s, &Options { seed: 99, ..opt }).unwrap();
        assert_eq!(a.ksd, c.ksd, "the statistic does not depend on the bootstrap seed");
    }

    // There is deliberately no test comparing this to a Gibbs chain, or to any other sampler.
    //
    // Two samplers agreeing proves that they share a bias as readily as that they share the truth,
    // and this module's job is precisely to detect the case where a sampler is confidently wrong.
    // Every oracle above is either a closed form (the probability ratio, Stein's identity), an
    // exhaustive sum over the state space (the population discrepancy), or the definition the code
    // optimises away (the flipped kernel values). The one sampled input, `exact_draws`, is an
    // inverse-CDF draw from the enumerated distribution: exact and independent by construction.
}
