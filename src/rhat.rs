//! Convergence diagnostics for a set of chains — split R-hat, rank-normalised and folded, bulk and
//! tail effective sample sizes, the Monte Carlo standard error of a mean, and nested R-hat for many
//! short chains — each with an oracle that does not come from a trace.
//!
//! Nested R-hat is the one a p-bit array needs, and its oracle is the strongest here:
//! [`crate::autocorr::nested_population`] computes the value it converges to from the kernel itself,
//! and with it the three blind spots `examples/nested_exact.rs` measures — a start every superchain
//! shares (it reads exactly its floor and passes at warmup 0), a bias every start shares, and a
//! kernel whose own law is not the Boltzmann law (a synchronous sweep passes, with its energy 1.19
//! standard deviations from the Boltzmann value).
//!
//! # Why this exists
//!
//! Everything in this crate that reports an effective sample size does it from ONE chain through
//! [`crate::certify::tau_int`], and `examples/tau_exactness.rs` measured what that is worth: on a
//! spectrum with a large fast mode beside a small slow one, Sokal's window closes early and no trace
//! length repairs it. The field's answer is to run several chains from dispersed starts and ask
//! whether they agree — Gelman and Rubin's R-hat — and its current form (Vehtari, Gelman, Simpson,
//! Carpenter and Bürkner, *Bayesian Analysis* 2021) splits every chain in half so a trend inside one
//! chain counts as disagreement, rank-normalises the draws so heavy tails and infinite variances
//! cannot hide, folds them about the median so a chain with the right mean and the wrong spread is
//! caught, and reports two effective sample sizes: bulk, for the centre, and tail, for the 5% and
//! 95% quantiles, which converge at different rates. This module is that toolkit, computed the way
//! Stan computes it, with the per-chain autocovariances from [`crate::fft::autocovariance`] and
//! Geyer's initial monotone sequence in place of a window.
//!
//! # Oracles
//!
//! Chains drawn i.i.d. have every R-hat at 1 and every effective sample size at the draw count, up
//! to the estimators' own noise. Chains with shifted means have a classical R-hat that is a closed
//! form of their means and variances, computed here by a second, independent six-line formula. An
//! AR(1) chain at `rho` has `ESS / N = (1 - rho) / (1 + rho)` exactly. A trend inside every chain is
//! invisible to the unsplit statistic and visible to the split one; a chain with the right mean and
//! three times the spread is invisible to the rank statistic and visible to the folded one. The
//! inverse normal CDF the rank-normalisation needs round-trips the normal CDF to `1e-12`.
//!
//! # Conventions
//!
//! `ess` is `M N / tau` with `tau = 1 + 2 sum rho`, the same quantity [`crate::certify`] reports as
//! `draws / (2 tau_int)`; the two agree on a single chain up to the window rule. `mcse_mean` is the
//! pooled standard deviation over the square root of the plain (not rank-normalised) ESS.

use crate::fft::autocovariance;

/// Every diagnostic of one scalar quantity across `chains` chains of `draws` draws each.
#[derive(Clone, Debug, PartialEq)]
pub struct Diagnostics {
    /// Classical split R-hat on the raw draws.
    pub rhat: f64,
    /// Split R-hat on the rank-normalised draws: robust to heavy tails and infinite variance.
    pub rhat_rank: f64,
    /// Split R-hat on the rank-normalised absolute deviations from the median: catches a chain
    /// with the right centre and the wrong spread.
    pub rhat_folded: f64,
    /// Effective sample size of the rank-normalised draws: the centre of the distribution.
    pub ess_bulk: f64,
    /// The smaller of the effective sample sizes of the 5% and 95% quantile indicators.
    pub ess_tail: f64,
    /// Monte Carlo standard error of the pooled mean.
    pub mcse_mean: f64,
    /// Chains diagnosed.
    pub chains: usize,
    /// Draws per chain, before splitting.
    pub draws: usize,
}

/// Every diagnostic at once. `NaN` throughout for fewer than two chains of four draws.
///
/// Chains of unequal length are truncated to the shortest.
#[must_use]
pub fn diagnose(chains: &[Vec<f64>]) -> Diagnostics {
    let nan = Diagnostics {
        rhat: f64::NAN,
        rhat_rank: f64::NAN,
        rhat_folded: f64::NAN,
        ess_bulk: f64::NAN,
        ess_tail: f64::NAN,
        mcse_mean: f64::NAN,
        chains: chains.len(),
        draws: chains.iter().map(Vec::len).min().unwrap_or(0),
    };
    let n = nan.draws;
    if chains.len() < 2 || n < 4 {
        return nan;
    }
    let even: Vec<Vec<f64>> = chains.iter().map(|c| c[..n].to_vec()).collect();
    let ranked = rank_normalise(&even);
    let folded = rank_normalise(&fold(&even));
    let pooled: Vec<f64> = even.iter().flatten().copied().collect();
    let sd = variance(&pooled).sqrt();
    let ess_plain = ess(&even);
    Diagnostics {
        rhat: split_rhat(&even),
        rhat_rank: split_rhat(&ranked),
        rhat_folded: split_rhat(&folded),
        ess_bulk: ess(&ranked),
        ess_tail: ess_tail(&even),
        mcse_mean: sd / ess_plain.sqrt(),
        chains: chains.len(),
        draws: n,
    }
}

/// Every chain cut in half, so a trend inside a chain becomes a disagreement between two.
#[must_use]
pub fn split(chains: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let mut out = Vec::with_capacity(2 * chains.len());
    for c in chains {
        let half = c.len() / 2;
        out.push(c[..half].to_vec());
        out.push(c[half..2 * half].to_vec());
    }
    out
}

fn mean(x: &[f64]) -> f64 {
    x.iter().sum::<f64>() / x.len() as f64
}

/// Sample variance, `n - 1` in the denominator.
fn variance(x: &[f64]) -> f64 {
    let m = mean(x);
    x.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (x.len() as f64 - 1.0)
}

/// Gelman and Rubin's potential scale reduction on the SPLIT chains: with `M` chains of `N`
/// draws, within-chain variance `W` (the mean of the sample variances) and between-chain variance
/// `B = N var(chain means)`, `R = sqrt(((N - 1) W / N + B / N) / W)`. Exactly 1 when every chain
/// mean agrees; `NaN` when the chains are constant.
#[must_use]
pub fn split_rhat(chains: &[Vec<f64>]) -> f64 {
    let s = split(chains);
    let n = s.first().map_or(0, Vec::len);
    if s.len() < 2 || n < 2 {
        return f64::NAN;
    }
    let means: Vec<f64> = s.iter().map(|c| mean(c)).collect();
    let w = mean(&s.iter().map(|c| variance(c)).collect::<Vec<_>>());
    let b = n as f64 * variance(&means);
    let n = n as f64;
    (((n - 1.0) * w / n + b / n) / w).sqrt()
}

/// NESTED R-hat (Margossian, Hoffman, Sountsov, Riou-Durand, Vehtari & Gelman, "Nested R-hat:
/// Assessing the convergence of Markov chain Monte Carlo when running many short chains",
/// Bayesian Analysis, 2024), computed as `posterior::rhat_nested` computes it (posterior 1.5.0,
/// `R/nested_rhat.R`).
///
/// The regime it exists for is the one a p-bit array lives in: thousands of chains, a handful of
/// draws each. Split R-hat asks whether each chain has forgotten its start, which a chain four draws
/// long cannot have done. Nested R-hat groups chains into SUPERCHAINS that share a start and asks
/// whether the superchains agree: the variance of the superchain means `B_nu`, over `W_nu` -- the
/// mean over superchains of the within-chain variance plus the variance of that superchain's chain
/// means -- and `R_nu = sqrt(1 + B_nu / W_nu)`. It is bounded below by 1, which classical R-hat is
/// not: with one chain per superchain, `R_nu^2` is the unsplit classical `R^2` plus `1 / N` exactly
/// (the paper's footnote 3), and the test holds it to that identity.
///
/// `superchain[c]` names chain `c`'s superchain. Every superchain must hold the same number of
/// chains and every chain the same number of draws. With one draw per chain the within-chain
/// variance is zero, and with one chain per superchain the between-chain variance is, as `posterior`
/// takes them. `NaN` for mismatched lengths, unequal superchains, fewer than two superchains, a
/// non-finite draw, or `W_nu = 0`. [`crate::autocorr::nested_population`] gives the value this tends
/// to as the superchains multiply, from the kernel itself.
#[must_use]
pub fn nested_rhat(chains: &[Vec<f64>], superchain: &[usize]) -> f64 {
    if chains.is_empty() || chains.len() != superchain.len() {
        return f64::NAN;
    }
    let n = chains[0].len();
    if n == 0 || chains.iter().any(|c| c.len() != n) || chains.iter().flatten().any(|v| !v.is_finite()) {
        return f64::NAN;
    }
    let mut ids: Vec<usize> = superchain.to_vec();
    ids.sort_unstable();
    ids.dedup();
    if ids.len() < 2 {
        return f64::NAN;
    }
    let members: Vec<Vec<usize>> =
        ids.iter().map(|&k| (0..chains.len()).filter(|&c| superchain[c] == k).collect()).collect();
    let per = members[0].len();
    if members.iter().any(|m| m.len() != per) {
        return f64::NAN;
    }
    let chain_mean: Vec<f64> = chains.iter().map(|c| mean(c)).collect();
    let chain_var: Vec<f64> = chains.iter().map(|c| if n > 1 { variance(c) } else { 0.0 }).collect();
    let mut super_mean = Vec::with_capacity(members.len());
    let mut within_super = Vec::with_capacity(members.len());
    for m in &members {
        // Every chain holds n draws, so the superchain's mean is the mean of its chain means.
        let means: Vec<f64> = m.iter().map(|&c| chain_mean[c]).collect();
        super_mean.push(mean(&means));
        let between_chain = if per > 1 { variance(&means) } else { 0.0 };
        let within_chain = mean(&m.iter().map(|&c| chain_var[c]).collect::<Vec<_>>());
        within_super.push(within_chain + between_chain);
    }
    let b = variance(&super_mean);
    let w = mean(&within_super);
    if !(w > 0.0) {
        return f64::NAN;
    }
    (1.0 + b / w).sqrt()
}

/// Fractional ranks of the pooled draws, ties averaged, sent through the normal quantile with
/// Blom's offset: `z = Phi^-1((r - 3/8) / (S + 1/4))`. What every rank-normalised statistic runs on.
#[must_use]
pub fn rank_normalise(chains: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let mut all: Vec<(f64, usize)> = chains.iter().flatten().copied().enumerate().map(|(i, v)| (v, i)).collect();
    let s = all.len();
    all.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(core::cmp::Ordering::Equal));
    let mut rank = vec![0.0f64; s];
    let mut i = 0;
    while i < s {
        let mut j = i;
        while j + 1 < s && all[j + 1].0 == all[i].0 {
            j += 1;
        }
        let avg = (i + j) as f64 / 2.0 + 1.0;
        for k in i..=j {
            rank[all[k].1] = avg;
        }
        i = j + 1;
    }
    let mut out = Vec::with_capacity(chains.len());
    let mut at = 0;
    for c in chains {
        out.push(rank[at..at + c.len()].iter().map(|r| inverse_normal_cdf((r - 0.375) / (s as f64 + 0.25))).collect());
        at += c.len();
    }
    out
}

/// Absolute deviation of every draw from the pooled median.
#[must_use]
pub fn fold(chains: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let mut pooled: Vec<f64> = chains.iter().flatten().copied().collect();
    pooled.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
    let med = if pooled.len() % 2 == 1 { pooled[pooled.len() / 2] } else { 0.5 * (pooled[pooled.len() / 2 - 1] + pooled[pooled.len() / 2]) };
    // Written as loops rather than closures so the mutation suite can name the line: its row
    // format splits on '|', which a closure's parameter list contains.
    let mut out = Vec::with_capacity(chains.len());
    for c in chains {
        let mut folded = Vec::with_capacity(c.len());
        for x in c {
            folded.push((x - med).abs());
        }
        out.push(folded);
    }
    out
}

/// Effective sample size of the pooled draws from `M` split chains of `N` draws: the multi-chain
/// autocorrelation `rho_t = 1 - (W - mean_m acov_m(t)) / var_plus`, summed under Geyer's initial
/// monotone positive sequence, `ESS = M N / (1 + 2 sum rho)`, capped at `M N log10(M N)` as Stan
/// caps it, since antithetic chains can estimate more independent draws than they hold.
#[must_use]
pub fn ess(chains: &[Vec<f64>]) -> f64 {
    let s = split(chains);
    let n = s.first().map_or(0, Vec::len);
    let m = s.len();
    if m < 2 || n < 4 {
        return f64::NAN;
    }
    // Biased per-chain autocovariances (divided by N), as Stan uses, from the unbiased ones.
    let acov: Vec<Vec<f64>> = s
        .iter()
        .map(|c| autocovariance(c, n - 1).into_iter().enumerate().map(|(k, v)| v * (n - k) as f64 / n as f64).collect())
        .collect();
    let chain_var: Vec<f64> = acov.iter().map(|a| a[0] * n as f64 / (n as f64 - 1.0)).collect();
    let means: Vec<f64> = s.iter().map(|c| mean(c)).collect();
    let mean_var = mean(&chain_var);
    let var_plus = mean_var * (n as f64 - 1.0) / n as f64 + variance(&means);
    if !(var_plus > 0.0) {
        return f64::NAN;
    }
    let rho = |t: usize| -> f64 { 1.0 - (mean_var - acov.iter().map(|a| a[t]).sum::<f64>() / m as f64) / var_plus };
    // Geyer: sum autocorrelations in pairs while the pair sum is positive, then force the pair
    // sums to be non-increasing.
    let max_t = n - 1;
    let mut sum = 0.0;
    let mut prev = f64::INFINITY;
    let mut t = 0;
    while t + 1 < max_t {
        let mut p = rho(t) + rho(t + 1);
        if p <= 0.0 {
            break;
        }
        if p > prev {
            p = prev;
        }
        prev = p;
        sum += p;
        t += 2;
    }
    let tau = -1.0 + 2.0 * sum;
    let total = (m * n) as f64;
    let ess = total / tau.max(1e-300);
    ess.min(total * total.log10())
}

/// The smaller of the effective sample sizes of the indicators `x <= q_05` and `x <= q_95`, the
/// quantiles taken over the pooled draws.
#[must_use]
pub fn ess_tail(chains: &[Vec<f64>]) -> f64 {
    let mut pooled: Vec<f64> = chains.iter().flatten().copied().collect();
    if pooled.len() < 20 {
        return f64::NAN;
    }
    pooled.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
    let q = |p: f64| pooled[((pooled.len() as f64 - 1.0) * p).round() as usize];
    let (lo, hi) = (q(0.05), q(0.95));
    let indicator = |cut: f64| -> Vec<Vec<f64>> { chains.iter().map(|c| c.iter().map(|x| if *x <= cut { 1.0 } else { 0.0 }).collect()).collect() };
    ess(&rank_normalise(&indicator(lo))).min(ess(&rank_normalise(&indicator(hi))))
}

/// The standard normal CDF, from [`crate::hopfield::erf`].
#[must_use]
pub fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + crate::hopfield::erf(x / core::f64::consts::SQRT_2))
}

/// The standard normal quantile: Acklam's rational approximation, then two Newton steps on the
/// CDF, which take it to the CDF's own precision. `-inf` at 0, `+inf` at 1, `NaN` outside.
#[must_use]
pub fn inverse_normal_cdf(p: f64) -> f64 {
    if !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if p == 0.0 {
        return f64::NEG_INFINITY;
    }
    if p == 1.0 {
        return f64::INFINITY;
    }
    const A: [f64; 6] = [-3.969_683_028_665_376e1, 2.209_460_984_245_205e2, -2.759_285_104_469_687e2, 1.383_577_518_672_69e2, -3.066_479_806_614_716e1, 2.506_628_277_459_239];
    const B: [f64; 5] = [-5.447_609_879_822_406e1, 1.615_858_368_580_409e2, -1.556_989_798_598_866e2, 6.680_131_188_771_972e1, -1.328_068_155_288_572e1];
    const C: [f64; 6] = [-7.784_894_002_430_293e-3, -3.223_964_580_411_365e-1, -2.400_758_277_161_838, -2.549_732_539_343_734, 4.374_664_141_464_968, 2.938_163_982_698_783];
    const D: [f64; 4] = [7.784_695_709_041_462e-3, 3.224_671_290_700_398e-1, 2.445_134_137_142_996, 3.754_408_661_907_416];
    let p_low = 0.024_25;
    let mut x = if p < p_low {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5]) / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= 1.0 - p_low {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5]) / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    };
    for _ in 0..2 {
        let e = normal_cdf(x) - p;
        let pdf = (-0.5 * x * x).exp() / (2.0 * core::f64::consts::PI).sqrt();
        if pdf > 0.0 {
            x -= e / pdf;
        }
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Pcg;

    fn normal(rng: &mut Pcg) -> f64 {
        let u1 = rng.f64().max(1e-300);
        let u2 = rng.f64();
        (-2.0 * u1.ln()).sqrt() * (core::f64::consts::TAU * u2).cos()
    }

    fn iid(m: usize, n: usize, seed: u64, mean: f64, sd: f64) -> Vec<Vec<f64>> {
        (0..m)
            .map(|c| {
                let mut rng = Pcg::new(seed + c as u64, 0x4A);
                (0..n).map(|_| mean + sd * normal(&mut rng)).collect()
            })
            .collect()
    }

    /// Independent draws: every R-hat at 1 and every effective sample size at the draw count, to
    /// the estimators' own noise; the mean's standard error at `sd / sqrt(N)`.
    #[test]
    fn independent_chains_read_one_and_their_full_size() {
        let chains = iid(4, 2000, 1, 0.0, 1.0);
        let d = diagnose(&chains);
        for (name, r) in [("rhat", d.rhat), ("rank", d.rhat_rank), ("folded", d.rhat_folded)] {
            assert!((r - 1.0).abs() < 0.01, "{name} = {r} on independent draws");
        }
        let total = 8000.0;
        assert!((d.ess_bulk / total - 1.0).abs() < 0.15, "bulk ESS {} of {total}", d.ess_bulk);
        assert!((d.ess_tail / total - 1.0).abs() < 0.3, "tail ESS {} of {total}", d.ess_tail);
        assert!((d.mcse_mean / (1.0 / total.sqrt()) - 1.0).abs() < 0.3, "mcse {}", d.mcse_mean);
    }

    /// Shifted chains: the classical statistic is a closed form of the split chains' means and
    /// variances, written here a second time in six lines, and it must be far from 1.
    #[test]
    fn shifted_chains_match_the_closed_form_and_disagree() {
        let mut chains = iid(4, 500, 7, 0.0, 1.0);
        for (k, c) in chains.iter_mut().enumerate() {
            for x in c.iter_mut() {
                *x += 2.0 * k as f64;
            }
        }
        let s = split(&chains);
        let n = s[0].len() as f64;
        let means: Vec<f64> = s.iter().map(|c| c.iter().sum::<f64>() / n).collect();
        let grand = means.iter().sum::<f64>() / means.len() as f64;
        let b = n * means.iter().map(|m| (m - grand).powi(2)).sum::<f64>() / (means.len() as f64 - 1.0);
        let w = s.iter().map(|c| { let m = c.iter().sum::<f64>() / n; c.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1.0) }).sum::<f64>() / s.len() as f64;
        let want = (((n - 1.0) / n * w + b / n) / w).sqrt();
        let got = split_rhat(&chains);
        assert!((got - want).abs() < 1e-12, "{got} vs closed form {want}");
        assert!(got > 2.0, "four chains two standard deviations apart: R-hat {got}");
    }

    /// An AR(1) chain at `rho` has `ESS / N = (1 - rho) / (1 + rho)` exactly.
    #[test]
    fn an_ar1_chain_has_its_closed_form_effective_size() {
        let rho = 0.9f64;
        let chains: Vec<Vec<f64>> = (0..4)
            .map(|c| {
                let mut rng = Pcg::new(20 + c, 0x4B);
                let mut x = 0.0f64;
                (0..20_000).map(|_| { x = rho * x + (1.0 - rho * rho).sqrt() * normal(&mut rng); x }).collect()
            })
            .collect();
        let e = ess(&chains);
        let want = 80_000.0 * (1.0 - rho) / (1.0 + rho);
        assert!((e / want - 1.0).abs() < 0.25, "ESS {e} vs closed form {want}");
    }

    /// A trend inside every chain -- the first half around 0, the second around 3 -- leaves the
    /// chains identical to each other, so the unsplit statistic reads 1; the split one must not.
    #[test]
    fn a_trend_inside_every_chain_is_caught_by_splitting() {
        let mut chains = iid(4, 1000, 3, 0.0, 1.0);
        for c in &mut chains {
            for x in c.iter_mut().skip(500) {
                *x += 3.0;
            }
        }
        let unsplit = {
            let n = 1000.0;
            let means: Vec<f64> = chains.iter().map(|c| c.iter().sum::<f64>() / n).collect();
            let grand = means.iter().sum::<f64>() / 4.0;
            let b = n * means.iter().map(|m| (m - grand).powi(2)).sum::<f64>() / 3.0;
            let w = chains.iter().map(|c| { let m = c.iter().sum::<f64>() / n; c.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1.0) }).sum::<f64>() / 4.0;
            (((n - 1.0) / n * w + b / n) / w).sqrt()
        };
        assert!((unsplit - 1.0).abs() < 0.02, "unsplit R-hat {unsplit} cannot see a trend");
        let r = split_rhat(&chains);
        assert!(r > 1.5, "split R-hat must see it: {r}");
    }

    /// A chain with the right mean and three times the spread: the rank statistic reads near 1,
    /// the folded one must not.
    #[test]
    fn a_wrong_spread_with_the_right_centre_is_caught_by_folding() {
        let mut chains = iid(4, 1000, 11, 0.0, 1.0);
        for x in &mut chains[3] {
            *x *= 3.0;
        }
        let d = diagnose(&chains);
        assert!(d.rhat_rank < 1.1, "rank R-hat {} on equal centres", d.rhat_rank);
        assert!(
            d.rhat_folded > 1.1 && d.rhat_folded > d.rhat_rank + 0.05,
            "folded R-hat must see the spread the rank one misses: folded {} vs rank {}",
            d.rhat_folded,
            d.rhat_rank
        );
    }

    /// The normal quantile round-trips the CDF to `1e-12` across the range, and rank
    /// normalisation preserves order and lands near a standard normal.
    #[test]
    fn the_normal_quantile_round_trips_and_ranks_are_standard_normal() {
        for &p in &[1e-9, 1e-4, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 0.9999, 1.0 - 1e-9] {
            let x = inverse_normal_cdf(p);
            assert!((normal_cdf(x) - p).abs() < 1e-12, "p {p}: x {x}, back {}", normal_cdf(x));
        }
        let chains = iid(2, 3000, 5, 10.0, 4.0);
        let z = rank_normalise(&chains);
        let pooled: Vec<f64> = z.iter().flatten().copied().collect();
        let raw: Vec<f64> = chains.iter().flatten().copied().collect();
        for i in 1..raw.len() {
            assert_eq!(raw[i] > raw[i - 1], pooled[i] > pooled[i - 1], "rank normalisation must preserve order");
        }
        assert!(mean(&pooled).abs() < 0.01 && (variance(&pooled) - 1.0).abs() < 0.02, "mean {} var {}", mean(&pooled), variance(&pooled));
    }

    /// NESTED R-hat, against two things it does not share arithmetic with. First a hand-worked
    /// case: superchains `{[0,2],[2,4]}` and `{[4,6],[6,8]}` have chain means 1, 3, 5, 7, every
    /// within-chain variance 2, both between-chain variances 2, superchain means 2 and 6, so
    /// `W = 4`, `B = 8` and `R = sqrt(3)`. Then the paper's footnote-3 identity: with ONE chain per
    /// superchain, `R_nu^2` is the UNSPLIT classical `R^2` plus exactly `1 / N`, the classical form
    /// written out again here in its own six lines -- on data where the two differ from 1 by a lot.
    #[test]
    fn nested_rhat_matches_a_hand_case_and_the_classical_identity() {
        let chains = vec![vec![0.0, 2.0], vec![2.0, 4.0], vec![4.0, 6.0], vec![6.0, 8.0]];
        let r = nested_rhat(&chains, &[0, 0, 1, 1]);
        assert!((r - 3f64.sqrt()).abs() < 1e-15, "hand case: {r}");
        // Relabelling superchains changes nothing; regrouping does.
        assert!((nested_rhat(&chains, &[7, 7, 2, 2]) - r).abs() < 1e-15);
        let regrouped = nested_rhat(&chains, &[0, 1, 0, 1]);
        assert!((regrouped - r).abs() > 0.1, "a different grouping is a different statistic: {regrouped}");

        // One chain per superchain: R_nu^2 - R_classical^2 = 1 / N, on chains that disagree.
        let mut shifted = iid(6, 40, 3, 0.0, 1.0);
        for (c, chain) in shifted.iter_mut().enumerate() {
            for v in chain.iter_mut() {
                *v += 0.4 * c as f64;
            }
        }
        let ids: Vec<usize> = (0..6).collect();
        let nu = nested_rhat(&shifted, &ids);
        let n = 40.0;
        let means: Vec<f64> = shifted.iter().map(|c| mean(c)).collect();
        let w = mean(&shifted.iter().map(|c| variance(c)).collect::<Vec<_>>());
        let b = n * variance(&means);
        let classical_sq = ((n - 1.0) * w / n + b / n) / w;
        assert!(classical_sq > 1.2, "the chains must actually disagree: {classical_sq}");
        assert!((nu * nu - classical_sq - 1.0 / n).abs() < 1e-12, "footnote 3: {} vs {}", nu * nu, classical_sq + 1.0 / n);

        // One draw per chain is the regime it exists for, and has no within-chain variance.
        let singles = vec![vec![1.0], vec![3.0], vec![5.0], vec![9.0]];
        // Superchain means 2 and 7, B = 12.5; between-chain variances 2 and 8, W = 5.
        let r1 = nested_rhat(&singles, &[0, 0, 1, 1]);
        assert!((r1 - 3.5f64.sqrt()).abs() < 1e-15, "one draw per chain: {r1}");

        // Refusals.
        assert!(nested_rhat(&chains, &[0, 0, 0, 1]).is_nan(), "unequal superchains");
        assert!(nested_rhat(&chains, &[0, 0, 0, 0]).is_nan(), "one superchain");
        assert!(nested_rhat(&chains, &[0, 1]).is_nan(), "ids and chains must pair up");
        assert!(nested_rhat(&[vec![1.0, f64::NAN], vec![2.0, 3.0]], &[0, 1]).is_nan(), "non-finite draw");
        assert!(nested_rhat(&[vec![1.0], vec![1.0]], &[0, 1]).is_nan(), "W = 0");
    }
}
