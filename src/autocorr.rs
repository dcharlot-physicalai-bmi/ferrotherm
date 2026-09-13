//! Exact integrated autocorrelation times on enumerable models — the oracle `tau_int` never had.
//!
//! # Why this exists
//!
//! Every effective sample size in this crate, every joules-per-independent-sample, every
//! [`crate::certify::Finding::Undermixed`], divides by [`crate::certify::tau_int`]: Sokal's
//! automatic windowing over an empirical autocorrelation, cut off at the first lag `W >= 5 tau(W)`.
//! Nothing had ever measured that estimator against a value that did not itself come from a trace.
//! When something did (`examples/tau_exactness.rs`, 2026-09-13), it found that on a 12-spin
//! frustrated grid at `beta = 1` the estimator reports **0.04 to 0.06 of the true autocorrelation
//! time at every trace length tried, up to 332,000 sweeps, with a 2% spread**. That is not a
//! short-trace bias. The chain's autocorrelation is a large fast mode plus a small slow one, so
//! `tau(W)` is still tiny when the window closes at lag 9, and the slow mode — most of the truth —
//! is never summed. No trace length repairs a window that has already closed.
//!
//! # What this computes
//!
//! On a model small enough to enumerate, a sampler's one-step kernel is an explicit linear
//! operator `P` on the `2^n` states, and the stationary autocovariance of any observable `f` at
//! lag `k` is exactly
//!
//! ```text
//!   C(k) = sum_x pi(x) e(x) (P^k e)(x),      e = f - <f>_pi,
//! ```
//!
//! so `tau_int = 1/2 + sum_{k >= 1} C(k) / C(0)`, summed until the tail is below floating point.
//! No sampling, no windowing, no trace. [`tau_int_exact`] does that for the kernels this crate
//! runs, applying `P` matrix-free so the `2^n x 2^n` operator is never stored.
//!
//! # What it is for
//!
//! Two things. It is the reference every autocorrelation ESTIMATOR in this crate is scored
//! against — `tau_int` included — on the models where a reference exists. And it makes small-model
//! comparisons between samplers exact rather than estimated: a `tau` that carries no sampling error
//! and cannot be truncated, so a ratio of two of them is a fact about the two kernels.
//!
//! # State encoding
//!
//! State `x` in `0..2^n` has spin `i` equal to `+1` when bit `i` of `x` is set. `pi` is computed
//! here from [`crate::graph::Graph::energy`] and normalised, because the point of an oracle is that
//! it shares nothing with the sampler it checks except the model.

use crate::graph::Graph;
use crate::informed::Balance;
use crate::kernel::p_up;
use std::fmt;

/// The most spins an exact operator will be built over. Above this a single application of the
/// chromatic sweep is `2^n * 2^(n/2)` work and the answer is a long time coming.
pub const MAX_SPINS: usize = 16;

/// Which sampler's one-step kernel to build.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kernel {
    /// One chromatic sweep of [`crate::gibbs::Sampler`]: each colour class in turn, every site of
    /// the class resampled from its heat-bath conditional given the others.
    ChromaticGibbs,
    /// One sequential sweep in site order `0..n`, each site resampled from its heat-bath
    /// conditional given the current state of all the others -- what [`crate::dtm::Ebm::gibbs`]
    /// runs, and what a plain single-site Gibbs sampler is.
    SequentialGibbs,
    /// One two-phase sweep of the shipped p-bit fabric, [`crate::hdl::FixedFabric`]: the same
    /// colour classes as [`Kernel::ChromaticGibbs`], but every site's flip probability is what the
    /// RTL computes -- couplings and fields rounded to Q.8, the field clamped to `[-8, 8)`, the
    /// sigmoid read from the 1,024-entry ROM at 16-bit resolution -- and the per-node random
    /// number taken as an IDEAL 16-bit uniform. That isolates the arithmetic: the stationary law
    /// of this kernel is what the fabric samples if its RNG were perfect, and its distance from
    /// the Boltzmann distribution is the price of the precision alone. Identical to
    /// `Quantised { frac_bits: 8, lut_bits: 10, prob_bits: 16 }`, and asserted to be.
    FixedFabric,
    /// The fabric's arithmetic at any precision: `frac_bits` fractional bits in the fixed-point
    /// couplings and fields (the field clamped to `[-8, 8)` as the RTL does), a sigmoid ROM of
    /// `2^lut_bits` entries over that same range, and the flip probability held to `prob_bits`
    /// bits. This is the knob the shipped RTL does not have, so a precision sweep can be run
    /// exactly rather than by rebuilding hardware: how far the sampled law sits from Boltzmann as
    /// a function of bits, at each temperature.
    Quantised {
        /// Fractional bits of the fixed-point weights and fields.
        frac_bits: u32,
        /// Address bits of the sigmoid ROM.
        lut_bits: u32,
        /// Bits of the flip probability, i.e. of the comparator and its uniform.
        prob_bits: u32,
    },
    /// One step of [`crate::informed::Informed`]: propose site `k` with probability
    /// `g(r_k) / Z(x)`, accept with `min(1, Z(x) / Z(y))`.
    Informed(Balance),
}

/// Why an exact autocorrelation could not be computed.
#[derive(Clone, Debug, PartialEq)]
pub enum AutocorrError {
    /// More spins than [`MAX_SPINS`].
    TooManySpins {
        /// Spins in the model.
        n: usize,
        /// The cap.
        max: usize,
    },
    /// The observable is constant under `pi`, so its autocorrelation is undefined.
    NoVariance,
}

impl fmt::Display for AutocorrError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AutocorrError::TooManySpins { n, max } => {
                write!(f, "{n} spins is more than the {max} an exact operator is built over")
            }
            AutocorrError::NoVariance => {
                write!(f, "the observable does not vary under the Boltzmann distribution")
            }
        }
    }
}

impl std::error::Error for AutocorrError {}

/// The result: the exact `tau_int`, and enough of how it was reached to read it.
#[derive(Clone, Debug, PartialEq)]
pub struct Autocorrelation {
    /// `1/2 + sum_{k >= 1} rho(k)`, in kernel steps (sweeps for Gibbs, single steps for informed).
    pub tau_int: f64,
    /// Lags summed before the tail fell below `tol`.
    pub lags: usize,
    /// The first autocorrelations `rho(1..)`, as many as were summed, capped at 64 entries.
    pub rho: Vec<f64>,
    /// Variance of the observable under `pi`.
    pub variance: f64,
}

/// State `x` as spins: bit `i` set means `s_i = +1`.
#[must_use]
pub fn spins(x: usize, n: usize) -> Vec<i8> {
    (0..n).map(|i| if (x >> i) & 1 == 1 { 1 } else { -1 }).collect()
}

/// The Boltzmann distribution over all `2^n` states, from the graph's own energy.
///
/// # Errors
///
/// [`AutocorrError::TooManySpins`] above [`MAX_SPINS`].
pub fn boltzmann(g: &Graph, beta: f64) -> Result<Vec<f64>, AutocorrError> {
    if g.n > MAX_SPINS {
        return Err(AutocorrError::TooManySpins { n: g.n, max: MAX_SPINS });
    }
    let m = 1usize << g.n;
    let energy: Vec<f64> = (0..m).map(|x| g.energy(&spins(x, g.n))).collect();
    let emin = energy.iter().copied().fold(f64::INFINITY, f64::min);
    let mut pi: Vec<f64> = energy.iter().map(|e| (-beta * (e - emin)).exp()).collect();
    let z: f64 = pi.iter().sum();
    for p in &mut pi {
        *p /= z;
    }
    Ok(pi)
}

/// The probability site `i` comes up `+1` when resampled, under the kernel's arithmetic.
///
/// Exact heat bath for the Gibbs kernels. For [`Kernel::FixedFabric`] it reproduces
/// [`crate::hdl::FixedFabric`] step by step: couplings and fields rounded to Q.8 (`FRAC` bits),
/// the integer field clamped to `[-2048, 2047]`, the ROM address `(field + 2048) >> 2`, the ROM
/// entry `p_up` at the address's centre in 16 bits, and the comparison against a 16-bit uniform,
/// so the probability is `entry / 65536` exactly. Kept beside the emulator's constants rather
/// than importing its private ones; `fabric_kernel_matches_the_emulator_in_distribution` is what
/// keeps the two from drifting apart.
fn p_site(g: &Graph, beta: f64, kernel: Kernel, i: usize, s: &[i8]) -> f64 {
    match kernel {
        Kernel::FixedFabric => p_site(
            g,
            beta,
            Kernel::Quantised { frac_bits: crate::hdl::FRAC, lut_bits: crate::hdl::LUT_BITS, prob_bits: 16 },
            i,
            s,
        ),
        Kernel::Quantised { frac_bits, lut_bits, prob_bits } => {
            // The field in fixed point, at `frac_bits` fractional bits.
            let scale = f64::from(1u32 << frac_bits);
            let mut field = (g.h[i] * scale).round() as i64;
            for k in g.offset[i]..g.offset[i + 1] {
                let w = (g.w[k] * scale).round() as i64;
                field += if s[g.nbr[k] as usize] > 0 { w } else { -w };
            }
            // Clamped to [-8, 8) in field units, as the RTL clamps: [-8 * 2^frac, 8 * 2^frac - 1].
            let half = 8i64 << frac_bits;
            let fc = field.clamp(-half, half - 1);
            // The ROM covers [-8, 8) with 2^lut entries. Its address is the clamped field's
            // position in that range at ROM resolution: the field grid has 16 * 2^frac cells over
            // the range and the ROM 2^lut, so the address is the field offset scaled by
            // 2^lut / (16 * 2^frac) -- a right shift when the ROM is coarser than the field grid
            // (Q.8 with 1,024 entries: shift 2, the RTL's `(fc + 2048) >> 2`) and a left shift
            // when it is finer.
            let offset = fc + half;
            let addr = if lut_bits <= frac_bits + 4 {
                offset >> (frac_bits + 4 - lut_bits)
            } else {
                offset << (lut_bits - frac_bits - 4)
            };
            let stride = 16.0 / f64::from(1u32 << lut_bits);
            let arg = (addr as f64 + 0.5) * stride - 8.0;
            // The entry held to `prob_bits`: the RTL's `round(p * 65535) / 65536` at 16 bits.
            let levels = f64::from(1u32 << prob_bits);
            let entry = (p_up(arg, beta) * (levels - 1.0)).round().min(levels - 1.0);
            entry / levels
        }
        _ => p_up(g.field(i, s), beta),
    }
}

fn log_g(balance: Balance, log_r: f64) -> f64 {
    let softplus = |x: f64| if x > 0.0 { x + (-x).exp().ln_1p() } else { x.exp().ln_1p() };
    match balance {
        Balance::Sqrt => 0.5 * log_r,
        Balance::Metropolis => log_r.min(0.0),
        Balance::Barker => -softplus(-log_r),
    }
}

/// Apply the kernel once: `v <- P v`, matrix-free.
///
/// For the chromatic sweep `P = P_{c_last} ... P_{c_1}`, so the classes are applied to `v` in
/// REVERSE order (operators compose right to left). Within a class the sites are pairwise
/// non-adjacent, so the class update factorises into independent heat-bath draws from the fields
/// at the pre-class state.
///
/// For the informed kernel `(P v)(x) = sum_k P(x -> y_k) v(y_k) + P(x -> x) v(x)` with
/// `P(x -> y_k) = w_k(x) / Z(x) * min(1, Z(x) / Z(y_k))`, exactly the acceptance
/// [`crate::informed`] derives; the shift that module carries cancels in every ratio and is
/// omitted.
///
/// # Panics
///
/// If `v` does not have `2^n` entries.
#[must_use]
pub fn apply(g: &Graph, beta: f64, kernel: Kernel, v: &[f64]) -> Vec<f64> {
    let n = g.n;
    let m = 1usize << n;
    assert_eq!(v.len(), m, "a function over states has 2^n entries");
    match kernel {
        Kernel::ChromaticGibbs | Kernel::FixedFabric | Kernel::Quantised { .. } => {
            let mut cur = v.to_vec();
            for class in g.classes.iter().rev() {
                let sites: Vec<usize> = class.iter().map(|&i| i as usize).collect();
                let k = sites.len();
                let mut next = vec![0.0f64; m];
                let mut p = vec![0.0f64; k];
                for x in 0..m {
                    let s = spins(x, n);
                    for (j, &i) in sites.iter().enumerate() {
                        p[j] = p_site(g, beta, kernel, i, &s);
                    }
                    let mut base = x;
                    for &i in &sites {
                        base &= !(1usize << i);
                    }
                    let mut acc = 0.0;
                    for a in 0..(1usize << k) {
                        let mut y = base;
                        let mut w = 1.0;
                        for (j, &i) in sites.iter().enumerate() {
                            if (a >> j) & 1 == 1 {
                                y |= 1usize << i;
                                w *= p[j];
                            } else {
                                w *= 1.0 - p[j];
                            }
                        }
                        acc += w * cur[y];
                    }
                    next[x] = acc;
                }
                cur = next;
            }
            cur
        }
        Kernel::SequentialGibbs => {
            // P = P_0 P_1 ... P_{n-1} on distributions, so on functions the LAST site's update is
            // applied first: (P v) = P_0 (P_1 (... (P_{n-1} v))). Each P_i is rank two per state --
            // the heat-bath conditional at site i given the current others.
            let mut cur = v.to_vec();
            for i in (0..n).rev() {
                let bit = 1usize << i;
                let mut next = vec![0.0f64; m];
                for x in 0..m {
                    let s = spins(x, n);
                    let p = p_up(g.field(i, &s), beta);
                    next[x] = p * cur[x | bit] + (1.0 - p) * cur[x & !bit];
                }
                cur = next;
            }
            cur
        }
        Kernel::Informed(balance) => {
            // Z(x) for every state first, since the acceptance needs Z at the neighbour too.
            let weights = |x: usize| -> Vec<f64> {
                let s = spins(x, n);
                (0..n)
                    .map(|k| {
                        let log_r = -2.0 * beta * f64::from(s[k]) * g.field(k, &s);
                        log_g(balance, log_r).exp()
                    })
                    .collect()
            };
            let z: Vec<f64> = (0..m).map(|x| weights(x).iter().sum()).collect();
            let mut next = vec![0.0f64; m];
            for x in 0..m {
                let w = weights(x);
                let mut stay = 1.0;
                let mut acc = 0.0;
                if z[x] > 0.0 {
                    for k in 0..n {
                        let y = x ^ (1usize << k);
                        let alpha = if z[y] > 0.0 { (z[x] / z[y]).min(1.0) } else { 1.0 };
                        let p = w[k] / z[x] * alpha;
                        stay -= p;
                        acc += p * v[y];
                    }
                }
                next[x] = acc + stay.max(0.0) * v[x];
            }
            next
        }
    }
}

/// Push a DISTRIBUTION one step: `mu <- mu P`, matrix-free -- the adjoint of [`apply`].
///
/// Where [`apply`] answers "what is the expected value of `v` one step from here", this answers
/// "where is the chain one step after being distributed as `mu`". It is what an exact
/// convergence curve needs: start at the distribution a sampler actually starts from (uniform, a
/// data point, a clamped context), push it `k` steps, and read the total variation to `pi`
/// exactly. For the chromatic and sequential sweeps the site updates are applied in FORWARD
/// order, since distributions multiply on the left.
///
/// # Panics
///
/// If `mu` does not have `2^n` entries.
#[must_use]
pub fn apply_distribution(g: &Graph, beta: f64, kernel: Kernel, mu: &[f64]) -> Vec<f64> {
    let n = g.n;
    let m = 1usize << n;
    assert_eq!(mu.len(), m, "a distribution over states has 2^n entries");
    match kernel {
        Kernel::SequentialGibbs => {
            let mut cur = mu.to_vec();
            for i in 0..n {
                let bit = 1usize << i;
                let mut next = vec![0.0f64; m];
                for x in 0..m {
                    let s = spins(x, n);
                    let p = p_up(g.field(i, &s), beta);
                    // Mass from both values of site i lands on x with x_i's own probability.
                    let pooled = cur[x | bit] + cur[x & !bit];
                    next[x] = if x & bit != 0 { p * pooled } else { (1.0 - p) * pooled };
                }
                cur = next;
            }
            cur
        }
        Kernel::ChromaticGibbs | Kernel::FixedFabric | Kernel::Quantised { .. } => {
            let mut cur = mu.to_vec();
            for class in &g.classes {
                let sites: Vec<usize> = class.iter().map(|&i| i as usize).collect();
                let k = sites.len();
                let mut next = vec![0.0f64; m];
                for x in 0..m {
                    let s = spins(x, n);
                    // The class sites' conditionals depend only on the OTHER sites, so they are
                    // the same for every state that differs from x only on the class.
                    let mut w = 1.0;
                    for &i in &sites {
                        let p = p_site(g, beta, kernel, i, &s);
                        w *= if x & (1usize << i) != 0 { p } else { 1.0 - p };
                    }
                    let mut base = x;
                    for &i in &sites {
                        base &= !(1usize << i);
                    }
                    let mut pooled = 0.0;
                    for a in 0..(1usize << k) {
                        let mut y = base;
                        for (j, &i) in sites.iter().enumerate() {
                            if (a >> j) & 1 == 1 {
                                y |= 1usize << i;
                            }
                        }
                        pooled += cur[y];
                    }
                    next[x] = w * pooled;
                }
                cur = next;
            }
            cur
        }
        Kernel::Informed(balance) => {
            let weights = |x: usize| -> Vec<f64> {
                let s = spins(x, n);
                (0..n)
                    .map(|k| {
                        let log_r = -2.0 * beta * f64::from(s[k]) * g.field(k, &s);
                        log_g(balance, log_r).exp()
                    })
                    .collect()
            };
            let w: Vec<Vec<f64>> = (0..m).map(weights).collect();
            let z: Vec<f64> = w.iter().map(|wx| wx.iter().sum()).collect();
            let mut next = vec![0.0f64; m];
            for x in 0..m {
                // Mass arriving from each neighbour y_k, plus what stays.
                let mut stay = 1.0;
                let mut acc = 0.0;
                for k in 0..n {
                    let y = x ^ (1usize << k);
                    if z[x] > 0.0 {
                        let alpha = if z[y] > 0.0 { (z[x] / z[y]).min(1.0) } else { 1.0 };
                        stay -= w[x][k] / z[x] * alpha;
                    }
                    if z[y] > 0.0 {
                        let alpha = if z[x] > 0.0 { (z[y] / z[x]).min(1.0) } else { 1.0 };
                        acc += mu[y] * w[y][k] / z[y] * alpha;
                    }
                }
                next[x] = acc + stay.max(0.0) * mu[x];
            }
            next
        }
    }
}

/// The kernel's stationary distribution, by pushing the uniform distribution forward until it
/// stops moving: `tol` in total variation between successive steps, or `max_steps`.
///
/// For the Gibbs kernels this is the Boltzmann distribution to floating point, and the call is a
/// long way round to [`boltzmann`]. For [`Kernel::FixedFabric`] it is NOT: the fabric's arithmetic
/// has its own stationary law, and this is the only way to get it. Returns the distribution and
/// the number of steps taken, so a caller can see whether it converged or was stopped.
///
/// # Errors
///
/// [`AutocorrError::TooManySpins`] above [`MAX_SPINS`].
pub fn stationary(
    g: &Graph,
    beta: f64,
    kernel: Kernel,
    tol: f64,
    max_steps: usize,
) -> Result<(Vec<f64>, usize), AutocorrError> {
    if g.n > MAX_SPINS {
        return Err(AutocorrError::TooManySpins { n: g.n, max: MAX_SPINS });
    }
    let m = 1usize << g.n;
    let mut mu = vec![1.0 / m as f64; m];
    let mut steps = 0;
    while steps < max_steps {
        let next = apply_distribution(g, beta, kernel, &mu);
        steps += 1;
        let moved = total_variation(&next, &mu);
        mu = next;
        if moved < tol {
            break;
        }
    }
    Ok((mu, steps))
}

/// Total variation distance between two distributions over the same states.
#[must_use]
pub fn total_variation(p: &[f64], q: &[f64]) -> f64 {
    0.5 * p.iter().zip(q).map(|(a, b)| (a - b).abs()).sum::<f64>()
}

/// The exact integrated autocorrelation time of `observable` under `kernel` at `beta`.
///
/// `tol` is the tail cut: lags are summed until `|rho(k)| < tol`, or until `max_lags`. Both are
/// reported back so a reader can see whether the sum converged or was stopped.
///
/// # Errors
///
/// [`AutocorrError::TooManySpins`] above [`MAX_SPINS`]; [`AutocorrError::NoVariance`] for a
/// constant observable.
pub fn tau_int_exact(
    g: &Graph,
    beta: f64,
    kernel: Kernel,
    observable: impl Fn(&[i8]) -> f64,
    tol: f64,
    max_lags: usize,
) -> Result<Autocorrelation, AutocorrError> {
    // The autocorrelation is STATIONARY: it must be taken under the kernel's own invariant law.
    // For the exact Gibbs and informed kernels that is the Boltzmann distribution; for the
    // fabric's arithmetic it is not, and using Boltzmann there would measure a transient.
    let pi = match kernel {
        Kernel::FixedFabric | Kernel::Quantised { .. } => stationary(g, beta, kernel, 1e-14, 500_000)?.0,
        _ => boltzmann(g, beta)?,
    };
    let n = g.n;
    let m = 1usize << n;
    let f: Vec<f64> = (0..m).map(|x| observable(&spins(x, n))).collect();
    let mean: f64 = pi.iter().zip(&f).map(|(p, v)| p * v).sum();
    let e: Vec<f64> = f.iter().map(|v| v - mean).collect();
    let c0: f64 = pi.iter().zip(&e).map(|(p, x)| p * x * x).sum();
    if !(c0 > 0.0) {
        return Err(AutocorrError::NoVariance);
    }
    let mut v = e.clone();
    let mut tau = 0.5;
    let mut rho = Vec::new();
    let mut lags = 0;
    while lags < max_lags {
        v = apply(g, beta, kernel, &v);
        lags += 1;
        let ck: f64 = pi.iter().zip(&e).zip(&v).map(|((p, x), y)| p * x * y).sum();
        let r = ck / c0;
        tau += r;
        if rho.len() < 64 {
            rho.push(r);
        }
        if r.abs() < tol {
            break;
        }
    }
    Ok(Autocorrelation { tau_int: tau, lags, rho, variance: c0 })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::GraphBuilder;
    use crate::rng::Pcg;

    fn grid_glass(w: usize, h: usize, seed: u64) -> Graph {
        let mut rng = Pcg::new(seed, 0x6A);
        let mut b = GraphBuilder::new(w * h);
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                if x + 1 < w {
                    b.couple(i, i + 1, if rng.f64() < 0.5 { -1.0 } else { 1.0 });
                }
                if y + 1 < h {
                    b.couple(i, i + w, if rng.f64() < 0.5 { -1.0 } else { 1.0 });
                }
            }
        }
        for i in 0..w * h {
            b.bias(i, (rng.f64() - 0.5) * 0.4);
        }
        b.build()
    }

    /// CLOSED FORM: a single site is resampled from its conditional every sweep, so successive
    /// draws are independent and `tau_int` is exactly one half -- for every field and every beta.
    #[test]
    fn a_single_site_decorrelates_in_one_sweep_exactly() {
        for (h, beta) in [(0.0, 1.0), (0.7, 0.3), (-2.0, 3.0)] {
            let mut b = GraphBuilder::new(1);
            b.bias(0, h);
            let g = b.build();
            let a = tau_int_exact(&g, beta, Kernel::ChromaticGibbs, |s| f64::from(s[0]), 1e-15, 100)
                .unwrap();
            assert_eq!(a.tau_int, 0.5, "h={h} beta={beta}: {:?}", a.rho);
            assert_eq!(a.lags, 1);
        }
    }

    /// THE OPERATOR IS A STOCHASTIC MATRIX WITH pi AS ITS STATIONARY DISTRIBUTION -- for both
    /// kernels. `pi P = pi` is the one property an exact kernel cannot fake, and it is checked to
    /// floating point against a `pi` computed from the energy alone. Row sums are checked through
    /// the constant function, which `P` must map to itself.
    #[test]
    fn both_kernels_leave_the_boltzmann_distribution_invariant_to_floating_point() {
        let g = grid_glass(3, 3, 5);
        let beta = 1.1;
        let m = 1usize << g.n;
        let pi = boltzmann(&g, beta).unwrap();
        for kernel in [Kernel::ChromaticGibbs, Kernel::Informed(Balance::Barker), Kernel::Informed(Balance::Sqrt)] {
            // pi P = pi, computed through the adjoint: (pi P)(y) = sum_x pi(x) P(x -> y). Apply P
            // to every basis vector is 2^n applications; instead check <pi, P f> = <pi, f> for a
            // family of f, which is the same statement tested on that family.
            let mut rng = Pcg::new(3, 9);
            for _ in 0..8 {
                let f: Vec<f64> = (0..m).map(|_| rng.f64()).collect();
                let pf = apply(&g, beta, kernel, &f);
                let lhs: f64 = pi.iter().zip(&pf).map(|(p, v)| p * v).sum();
                let rhs: f64 = pi.iter().zip(&f).map(|(p, v)| p * v).sum();
                assert!((lhs - rhs).abs() < 1e-12, "{kernel:?}: <pi, P f> = {lhs} vs <pi, f> = {rhs}");
            }
            let ones = vec![1.0f64; m];
            let p1 = apply(&g, beta, kernel, &ones);
            let worst = p1.iter().map(|v| (v - 1.0).abs()).fold(0.0f64, f64::max);
            assert!(worst < 1e-12, "{kernel:?}: rows do not sum to one, worst {worst:e}");
        }
    }

    /// THE MATRIX-FREE SWEEP MATCHES A DENSE TRANSITION MATRIX BUILT A DIFFERENT WAY: by
    /// simulating the class-by-class update as an explicit product of dense stochastic matrices,
    /// one per class, each assembled from the same heat-bath probabilities. Two constructions of
    /// the same operator that share only `p_up`.
    #[test]
    fn the_matrix_free_sweep_matches_a_dense_operator_built_class_by_class() {
        let g = grid_glass(3, 2, 2);
        let (n, beta) = (g.n, 0.8);
        let m = 1usize << n;
        // Dense P_c for each class, then P = P_last ... P_first as a matrix product on row vectors.
        let mut dense = vec![vec![0.0f64; m]; m];
        for x in 0..m {
            dense[x][x] = 1.0;
        }
        for class in &g.classes {
            let sites: Vec<usize> = class.iter().map(|&i| i as usize).collect();
            let mut pc = vec![vec![0.0f64; m]; m];
            for x in 0..m {
                let s = spins(x, n);
                let mut base = x;
                for &i in &sites {
                    base &= !(1usize << i);
                }
                for a in 0..(1usize << sites.len()) {
                    let mut y = base;
                    let mut w = 1.0;
                    for (j, &i) in sites.iter().enumerate() {
                        let p = p_up(g.field(i, &s), beta);
                        if (a >> j) & 1 == 1 {
                            y |= 1usize << i;
                            w *= p;
                        } else {
                            w *= 1.0 - p;
                        }
                    }
                    pc[x][y] += w;
                }
            }
            // dense <- dense * pc (row-vector convention: state distributions multiply on the left)
            let mut out = vec![vec![0.0f64; m]; m];
            for x in 0..m {
                for y in 0..m {
                    let d = dense[x][y];
                    if d != 0.0 {
                        for z in 0..m {
                            out[x][z] += d * pc[y][z];
                        }
                    }
                }
            }
            dense = out;
        }
        let mut rng = Pcg::new(7, 1);
        let v: Vec<f64> = (0..m).map(|_| rng.f64() - 0.5).collect();
        let free = apply(&g, beta, Kernel::ChromaticGibbs, &v);
        for x in 0..m {
            let pv: f64 = (0..m).map(|y| dense[x][y] * v[y]).sum();
            assert!((free[x] - pv).abs() < 1e-13, "state {x}: {} vs {pv}", free[x]);
        }
    }

    /// THE ESTIMATOR THE CRATE SHIPS IS SCORED AGAINST THE ORACLE, IN BOTH DIRECTIONS. On a hot
    /// chain (one mode, tau about two sweeps) Sokal's window lands on the exact value within its
    /// own noise. On the same model cold, the autocorrelation is a large fast mode plus a small
    /// slow one and the window closes before the slow mode is summed: the estimate must sit far
    /// BELOW the exact value on a trace that is hundreds of tau long -- which is the finding this
    /// module exists to record, asserted rather than described.
    #[test]
    fn sokal_agrees_on_a_hot_chain_and_truncates_a_cold_one() {
        use crate::certify::tau_int;
        use crate::gibbs::Sampler;
        let g = grid_glass(3, 3, 11);
        let run = |beta: f64, len: usize| -> f64 {
            let mut s = Sampler::new(&g, beta, 21);
            s.sweeps(2_000, None);
            let mut trace = Vec::with_capacity(len);
            for _ in 0..len {
                s.sweep(None);
                trace.push(g.energy(&s.s));
            }
            tau_int(&trace)
        };
        let hot = tau_int_exact(&g, 0.4, Kernel::ChromaticGibbs, |s| g.energy(s), 1e-12, 100_000).unwrap();
        let est = run(0.4, 200_000);
        assert!(
            (est / hot.tau_int - 1.0).abs() < 0.15,
            "hot: Sokal {est:.3} vs exact {:.3} over {} lags",
            hot.tau_int,
            hot.lags
        );
        let cold = tau_int_exact(&g, 1.6, Kernel::ChromaticGibbs, |s| g.energy(s), 1e-12, 200_000).unwrap();
        assert!(cold.tau_int > 5.0 * hot.tau_int, "the cold chain must actually be slow: {:?}", cold.tau_int);
        let len = (300.0 * cold.tau_int) as usize;
        let est_cold = run(1.6, len);
        assert!(
            est_cold < 0.5 * cold.tau_int,
            "the truncation must be RESOLVED for this test to record it: Sokal {est_cold:.2} vs exact {:.2} on {len} sweeps",
            cold.tau_int
        );
    }

    /// `apply_distribution` IS THE ADJOINT OF `apply`: `<mu P, v> = <mu, P v>` for every `mu`
    /// and `v`, for every kernel. Two independently written routines -- one pushes functions
    /// backward, the other pushes mass forward -- and this identity is the only thing that ties
    /// them together, so a sign, an index, or a class order wrong in either shows up here. Then
    /// the distribution form of stationarity, `pi P = pi`, which is the property an exact
    /// convergence curve needs to end at zero.
    #[test]
    fn pushing_mass_forward_is_the_adjoint_of_pulling_functions_back() {
        let g = grid_glass(3, 3, 8);
        let beta = 0.9;
        let m = 1usize << g.n;
        let pi = boltzmann(&g, beta).unwrap();
        let mut rng = Pcg::new(5, 3);
        for kernel in [
            Kernel::ChromaticGibbs,
            Kernel::SequentialGibbs,
            Kernel::FixedFabric,
            Kernel::Informed(Balance::Barker),
            Kernel::Informed(Balance::Sqrt),
        ] {
            for _ in 0..4 {
                let mut mu: Vec<f64> = (0..m).map(|_| rng.f64()).collect();
                let z: f64 = mu.iter().sum();
                for p in &mut mu {
                    *p /= z;
                }
                let v: Vec<f64> = (0..m).map(|_| rng.f64() - 0.5).collect();
                let lhs: f64 = apply_distribution(&g, beta, kernel, &mu).iter().zip(&v).map(|(a, b)| a * b).sum();
                let rhs: f64 = mu.iter().zip(apply(&g, beta, kernel, &v)).map(|(a, b)| a * b).sum();
                assert!((lhs - rhs).abs() < 1e-12, "{kernel:?}: <mu P, v> = {lhs} vs <mu, P v> = {rhs}");
                // Mass is conserved.
                let total: f64 = apply_distribution(&g, beta, kernel, &mu).iter().sum();
                assert!((total - 1.0).abs() < 1e-12, "{kernel:?}: mass {total}");
            }
            // The Boltzmann distribution is stationary for the exact kernels and NOT for the
            // fabric's arithmetic -- that gap is the fabric's own test below, not a failure here.
            if kernel != Kernel::FixedFabric {
                let pushed = apply_distribution(&g, beta, kernel, &pi);
                assert!(
                    total_variation(&pushed, &pi) < 1e-12,
                    "{kernel:?}: pi is not stationary in distribution form, TV {:e}",
                    total_variation(&pushed, &pi)
                );
            }
        }
    }

    /// THE FABRIC'S KERNEL IS THE FABRIC'S, NOT A MODEL OF IT: run the cycle-exact emulator on a
    /// six-spin frustrated ring for a long time and its state histogram must match the exact
    /// stationary distribution of `Kernel::FixedFabric` within sampling noise -- and that
    /// distribution must differ from the Boltzmann distribution by a resolvable amount, or the
    /// kernel is not modelling the quantisation at all. The emulator draws its uniforms from
    /// xorshift32 and the kernel assumes ideal uniforms, so the agreement also bounds what that
    /// RNG costs on this fixture.
    #[test]
    fn fabric_kernel_matches_the_emulator_in_distribution_and_is_not_boltzmann() {
        use crate::hdl::FixedFabric;
        let n = 6;
        let mut rng = Pcg::new(3, 0xFA);
        let mut b = GraphBuilder::new(n);
        for i in 0..n {
            b.couple(i, (i + 1) % n, if rng.f64() < 0.5 { -1.0 } else { 1.0 });
            b.bias(i, (rng.f64() - 0.5) * 0.6);
        }
        let g = b.build();
        assert_eq!(g.classes.len(), 2, "the fabric needs a bipartite graph");
        let beta = 1.3;
        let (fab, steps) = stationary(&g, beta, Kernel::FixedFabric, 1e-13, 100_000).unwrap();
        assert!(steps < 100_000, "the fabric law must converge");
        let pi = boltzmann(&g, beta).unwrap();
        let gap = total_variation(&fab, &pi);
        assert!(gap > 1e-6, "Q.8 and a 1,024-entry ROM must be visible: TV {gap:e}");
        assert!(gap < 0.05, "and small: TV {gap}");

        // The emulator, histogrammed over 300,000 sweeps after a burn-in.
        let mut f = FixedFabric::new(&g, beta, 77);
        for _ in 0..1_000 {
            f.sweep();
        }
        let sweeps = 300_000usize;
        let mut counts = vec![0u32; 1 << n];
        for _ in 0..sweeps {
            f.sweep();
            let x = f.s.iter().enumerate().fold(0usize, |a, (i, &b)| a | (usize::from(b) << i));
            counts[x] += 1;
        }
        let hist: Vec<f64> = counts.iter().map(|&c| f64::from(c) / sweeps as f64).collect();
        let to_fab = total_variation(&hist, &fab);
        // Noise on a 64-cell histogram from an autocorrelated chain: about sqrt(64 * 2 tau / N)
        // in TV, tau a few sweeps here; 0.02 is an order above that.
        assert!(to_fab < 0.02, "emulator histogram is {to_fab} from the kernel's stationary law");
        // And the kernel's law is the BETTER description of the emulator than Boltzmann is, when
        // the gap is above the histogram's own noise.
        let to_pi = total_variation(&hist, &pi);
        if gap > 3.0 * to_fab {
            assert!(to_fab < to_pi, "the emulator should sit nearer its own law ({to_fab}) than Boltzmann ({to_pi})");
        }
    }

    /// THE SHIPPED PRECISION IS ONE POINT OF THE KNOB, AND THE KNOB TURNS THE RIGHT WAY.
    /// `Quantised { 8, 10, 16 }` must be `FixedFabric` bit for bit -- the same operator, not a
    /// nearby one -- so the precision sweep is anchored to the hardware that was metered. Then the
    /// exact distance from Boltzmann must not grow as fractional bits are added, must be
    /// resolvable at four bits, and must be smaller at twelve than at eight: a kernel whose
    /// arithmetic error did not shrink with precision would be modelling something other than
    /// precision.
    #[test]
    fn the_quantised_kernel_reduces_to_the_fabric_and_improves_with_bits() {
        let n = 6;
        let mut rng = Pcg::new(3, 0xFA);
        let mut b = GraphBuilder::new(n);
        for i in 0..n {
            b.couple(i, (i + 1) % n, if rng.f64() < 0.5 { -1.0 } else { 1.0 });
            b.bias(i, (rng.f64() - 0.5) * 0.6);
        }
        let g = b.build();
        let beta = 1.3;
        let m = 1usize << n;
        let shipped = Kernel::Quantised { frac_bits: 8, lut_bits: 10, prob_bits: 16 };
        let mut mu: Vec<f64> = (0..m).map(|_| rng.f64()).collect();
        let z: f64 = mu.iter().sum();
        for p in &mut mu {
            *p /= z;
        }
        let a = apply_distribution(&g, beta, Kernel::FixedFabric, &mu);
        let q = apply_distribution(&g, beta, shipped, &mu);
        assert_eq!(a, q, "Quantised {{8, 10, 16}} must be the fabric's kernel exactly");

        let pi = boltzmann(&g, beta).unwrap();
        let gap = |frac: u32| {
            let k = Kernel::Quantised { frac_bits: frac, lut_bits: 10, prob_bits: 16 };
            total_variation(&stationary(&g, beta, k, 1e-13, 100_000).unwrap().0, &pi)
        };
        let gaps: Vec<f64> = [4u32, 5, 6, 7, 8, 10, 12].iter().map(|&f| gap(f)).collect();
        assert!(gaps[0] > 1e-3, "four bits must be resolvably off Boltzmann: {:e}", gaps[0]);
        // MEASURED, and not what the first version of this test assumed. Field bits are NOT a
        // monotone knob per instance: on this ring at beta 1.3 the gaps are 2.6e-3 at 4 bits,
        // 3.8e-2 at 5, 1.1e-2 at 6, then 9.8e-3 from 7 bits on. Four bits happen to round this
        // fixture's fields nearer their true values than five do -- the error at a given precision
        // depends on where the true values fall on the grid, not only on the grid's spacing -- and
        // from 7 bits the 1,024-entry ROM is the limit and the field grid stops mattering. So the
        // honest assertion is the envelope: the finest setting must sit at or under the worst of
        // the coarse ones, and the plateau must be flat once the ROM binds.
        let coarse_worst = gaps[..4].iter().copied().fold(0.0f64, f64::max);
        assert!(gaps[6] <= coarse_worst, "twelve bits must not be worse than the worst coarse setting: {gaps:?}");
        assert!((gaps[4] - gaps[6]).abs() < 1e-9, "past the ROM's resolution the field grid must not matter: {gaps:?}");
        // The ROM and the probability width are the other two knobs; each finer setting must
        // strictly beat a much coarser one, at a field precision fine enough not to be the limit.
        let gap_k = |k: Kernel| total_variation(&stationary(&g, beta, k, 1e-13, 100_000).unwrap().0, &pi);
        let coarse_rom = gap_k(Kernel::Quantised { frac_bits: 12, lut_bits: 6, prob_bits: 16 });
        let fine_rom = gap_k(Kernel::Quantised { frac_bits: 12, lut_bits: 12, prob_bits: 16 });
        assert!(fine_rom < coarse_rom, "a finer ROM must beat a coarser one: {fine_rom:e} vs {coarse_rom:e}");
        let coarse_p = gap_k(Kernel::Quantised { frac_bits: 12, lut_bits: 12, prob_bits: 4 });
        assert!(fine_rom < coarse_p, "a 4-bit probability must be worse than 16: {coarse_p:e} vs {fine_rom:e}");
    }

    /// The sequential sweep is a different kernel from the chromatic one -- same stationary law,
    /// different order, different tau -- and on a single site the two coincide exactly.
    #[test]
    fn the_sequential_and_chromatic_sweeps_are_different_kernels_with_the_same_law() {
        let g = grid_glass(3, 3, 8);
        let beta = 1.2;
        let a = tau_int_exact(&g, beta, Kernel::ChromaticGibbs, |s| g.energy(s), 1e-12, 100_000).unwrap();
        let b = tau_int_exact(&g, beta, Kernel::SequentialGibbs, |s| g.energy(s), 1e-12, 100_000).unwrap();
        assert!(
            (a.tau_int - b.tau_int).abs() > 1e-6,
            "two different orders should not give bit-identical taus: {} vs {}",
            a.tau_int,
            b.tau_int
        );
        // An exact convergence curve from the uniform start must end at zero for both.
        let m = 1usize << g.n;
        let pi = boltzmann(&g, beta).unwrap();
        for kernel in [Kernel::ChromaticGibbs, Kernel::SequentialGibbs] {
            let mut mu = vec![1.0 / m as f64; m];
            let mut tv = Vec::new();
            for _ in 0..400 {
                mu = apply_distribution(&g, beta, kernel, &mu);
                tv.push(total_variation(&mu, &pi));
            }
            assert!(tv[0] > tv[399], "{kernel:?}: the distance must fall");
            assert!(tv[399] < 1e-6, "{kernel:?}: TV after 400 sweeps {:e}", tv[399]);
            assert!(tv.windows(2).all(|w| w[1] <= w[0] + 1e-12), "{kernel:?}: TV must not rise");
        }
        let mut b1 = GraphBuilder::new(1);
        b1.bias(0, 0.4);
        let g1 = b1.build();
        let s = tau_int_exact(&g1, 2.0, Kernel::SequentialGibbs, |s| f64::from(s[0]), 1e-15, 10).unwrap();
        assert_eq!(s.tau_int, 0.5);
    }
}
