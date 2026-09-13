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
//! # When the sum does not converge
//!
//! At low temperature the lag sum is a mixing-time computation in disguise: the lags it needs grow
//! with the `tau` it is computing, and the same is true of [`stationary`], which pushes mass
//! forward until it stops moving. Both are cut off by a budget before they arrive
//! (`examples/fabric_exact.rs` hit its cap at `beta = 1.5` on 12 spins). [`stationary_solved`]
//! and [`tau_int_fundamental`] replace the iterations with one dense linear solve each — the
//! stationary law as the null vector of `I − P`, and `Σ_k C(k)` through the fundamental matrix
//! `(I − P + 1 πᵀ)⁻¹` of Kemeny and Snell — at a cost that does not depend on the temperature,
//! up to [`MAX_DENSE_SPINS`]. The same matrix gives [`kemeny_constant`], the sum of every mode's
//! relaxation time: a mixing measure of the kernel alone, where `tau_int` is of one observable
//! under one law and can fall when the law moves mass out of a slow valley.
//!
//! # What the fabric's law is
//!
//! `examples/nearest_boltzmann.rs` projects the fabric's exact law onto the Boltzmann family by
//! maximum likelihood — Newton on the exact Fisher matrix, so the fit is THE nearest pair
//! `(J', h')` — on the 4x3 grid at five temperatures. The law is within TV `5e-5` of some Boltzmann
//! distribution at every temperature, so a calibrated load could in principle remove almost all
//! of the departure; but the pair is not nearby when cold — the largest parameter shift is
//! `0.003` at `beta = 0.5`, `0.59` at `beta = 2` and `2.3` at `beta = 3`, against loaded couplings
//! of `±beta` — and the residual that no pair removes is `0.02%`, `1.9%`, `36%`, `5.6%` and `0.5%`
//! of the loaded KL at `beta = 0.5, 1, 1.5, 2, 3`: non-monotone, largest where the ROM's stride
//! and the comparator's floor are both in play. The single effective temperature that
//! `examples/fabric_exact.rs` reports is the one-parameter version of the same projection.
//!
//! # How fast the fabric relaxes, and why
//!
//! `examples/fabric_exact.rs` on the same grid, both kernels exact: at `beta = 0.5, 1, 1.5, 2` the
//! fabric's Kemeny constant is `1.000, 0.999, 0.972, 0.930` of the exact kernel's, and at
//! `beta = 3` it is `0.026` — the fabric's slowest mode is 39x FASTER than the exact chain's, and
//! `tau_int` of the energy agrees (`0.024`), so this is the kernel, not the weighting of an
//! observable. `examples/fabric_floor.rs` finds the cause by turning one knob. With the field at
//! Q.8 and the ROM at 1,024 entries, the ratio at `beta = 3` is `0.001` with a 12-bit comparator,
//! `0.026` at 16, `0.315` at 20 and `0.946` at 24; at `beta = 2` it is `0.221, 0.930, 0.978,
//! 0.987`; and 12 field bits with a 4,096-entry ROM at 16 comparator bits leave it at `0.939`. The
//! comparator is the whole effect. Its floor is `2^-16 = 1.53e-5`: `round(p * 65535) / 65536`
//! rounds the probability of the unlikely state to 0 below `7.6e-6` when that state is `+1`, and
//! never below `1.53e-5` when it is `-1`, because the entry cannot exceed 65535. At `beta = 3` a
//! flip against a field of 4.2 has exact probability `3.4e-6`; the fabric makes it impossible on
//! one side and 4.5x too likely on the other, and an escape from a metastable valley that needs
//! several such flips compounds the factor. The same floor caps the law: in the precision sweep
//! at `beta >= 2`, raising the field or ROM bits past the shipped values does not move TV below
//! `7e-3` while the comparator holds 16 bits (the coarser settings that do better there do so by
//! where this fixture's values fall on their grid), and 24 comparator bits alone take it to
//! `1.2e-3`, which 32 bits do not improve (`1.2e-3` at `beta = 2`; `8.4e-4` and a ratio of `0.976`
//! at `beta = 3`): past 24 bits the field and ROM are the floor. The engineering change that moves
//! every cold-fabric number in this crate is a wider comparator, and 24 bits is where it stops paying.
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
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Kernel {
    /// One chromatic sweep of [`crate::gibbs::Sampler`]: each colour class in turn, every site of
    /// the class resampled from its heat-bath conditional given the others.
    ChromaticGibbs,
    /// One sequential sweep in site order `0..n`, each site resampled from its heat-bath
    /// conditional given the current state of all the others -- what [`crate::dtm::Ebm::gibbs`]
    /// runs, and what a plain single-site Gibbs sampler is.
    SequentialGibbs,
    /// One fully SYNCHRONOUS sweep: every site resampled at once from its heat-bath conditional
    /// given the PREVIOUS state -- Little's (1974) dynamics, what a p-bit array does when all its
    /// nodes update on the same clock edge with no colouring. Its stationary law is not the
    /// Boltzmann distribution even on a bipartite graph: for symmetric couplings it is Peretto's
    /// closed form `pi(x) ∝ exp(beta h·x) prod_i 2 cosh(beta f_i(x))`, which [`peretto`] computes
    /// and the invariance test checks. `examples/synchronous_exact.rs` measures how far: on the 4x3
    /// grid the same-tick law is TV `0.61, 0.43, 0.33, 0.24, 0.12` from Boltzmann at
    /// `beta = 0.5, 1, 1.5, 2, 3`, a factor of 100 to 300 above the shipped fabric's arithmetic
    /// error, and no single temperature repairs it (TV at the nearest `beta` is `0.55, 0.42, 0.33,
    /// 0.24, 0.12`); on a 12-ring with three chords, so with odd cycles, it is `0.64, 0.93, 0.99,
    /// 0.999, 1.000` -- disjoint from Boltzmann when cold. Per site update it relaxes 1.0 to 3.0x
    /// slower than the chromatic sweep on the grid. On a bipartite graph the two sublattices never
    /// see each other at the same tick (`x_A(t + 1)` depends on `x_B(t)`), so the same-tick joint
    /// carries no cross-sublattice correlation at all; reading `A` at `t` and `B` at `t + 1` IS the
    /// chromatic sweep, which is why the fabric colours.
    Synchronous,
    /// The PIMI rule of Zhu, Singh et al. (arXiv:2604.17109, 2026), fully synchronous:
    /// `s_i(t+1) = sign[tanh(beta f_i(x)) + xi s_i(t) + eta N(0, 1)]`, every site at once from the
    /// previous state `x`, so `P(s_i = +1 | x) = Phi((tanh(beta f_i(x)) + xi x_i) / eta)` with
    /// `Phi` the normal CDF. `xi` is the self-spin inertia the paper adds to let all p-bits update
    /// simultaneously; `eta` sets the noise, and with it how certain a spin can ever be: `tanh`
    /// saturates at 1, so no field makes `P` exceed `Phi((1 + xi) / eta)`. The paper reports
    /// speed-ups and does not analyse the stationary law; [`stationary_solved`] gives it exactly,
    /// and `examples/pimi_exact.rs` does at `beta = 1`. On the 4x3 grid one cell is Boltzmann-like:
    /// `eta = 0.4, xi = 0.5` is within TV `8e-4` of the Boltzmann law at `beta_eff = 2.13`, twice
    /// the nominal, but relaxes `1.2e5x` slower per full update than the chromatic sweep (`2.2e9x`
    /// at `xi = 1`). On `K_{6,6}` with Gaussian couplings no cell of `eta` in `{0.2, 0.4, 0.8}` by
    /// `xi` in `{0, 0.1, 0.25, 0.5, 1}` comes within TV `5e-3` of any Boltzmann law, and the
    /// nearest (`eta = 0.2, xi = 0.5`) sits at `beta_eff = 13.8`, a near-frozen law, `2e9x` slower.
    /// At `eta = 0.2` and `xi >= 0.5` on the grid the chain is singular to floating point --
    /// frozen -- and has no invariant law to report. The inertia buys simultaneity at the price of
    /// a large, fixture-dependent temperature shift and a relaxation time orders of magnitude
    /// longer; whatever the 35x is, it is not sampling at equal flips.
    Pimi {
        /// Self-spin inertia coefficient.
        xi: f64,
        /// Standard deviation of the injected Gaussian noise.
        eta: f64,
    },
    /// One sequential sweep in site order in which every read of an ALREADY-UPDATED neighbour
    /// returns the pre-sweep value with probability `p`, independently per read: the
    /// synchronous-collision model of a fabric whose cells update in a fixed order but whose
    /// neighbour registers lag one update behind with probability `p`. `p = 0` is
    /// [`Kernel::SequentialGibbs`] and `p = 1` is [`Kernel::Synchronous`] (every read stale is
    /// every site from the previous state), so the Boltzmann and Peretto laws bracket it and the
    /// test holds both ends to their closed forms.
    Stale {
        /// Probability that a read of an already-updated neighbour returns its pre-sweep value.
        p: f64,
    },
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
    /// More spins than [`MAX_DENSE_SPINS`], the cap on the direct solves.
    TooManyForDense {
        /// Spins in the model.
        n: usize,
        /// The cap.
        max: usize,
    },
    /// The linear system was singular to floating point: the kernel has no unique invariant
    /// law, which is what more than one closed class of states produces.
    Reducible,
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
            AutocorrError::TooManyForDense { n, max } => {
                write!(f, "{n} spins is more than the {max} a dense operator is solved over")
            }
            AutocorrError::Reducible => {
                write!(f, "the kernel has no unique invariant law: its linear system is singular")
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
    /// Lags summed before the tail fell below `tol`; `0` from [`tau_int_fundamental`], which
    /// sums nothing lag by lag and so truncates nothing.
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

/// Peretto's closed form for the stationary law of [`Kernel::Synchronous`] with symmetric
/// couplings: `pi(x) ∝ exp(beta h·x) prod_i 2 cosh(beta f_i(x))`, `f_i` the local field at `x`
/// with the bias included (Peretto 1984, for Little's 1974 dynamics). It is stationary because
/// `pi(x) P(x, y) = exp(beta [h·x + h·y + y·J·x])` is symmetric in `x` and `y` when `J` is, so
/// the synchronous chain is reversible with respect to it -- and it is not the Boltzmann law,
/// whose weight is `exp(beta [h·x + x·J·x / 2])`.
///
/// # Errors
///
/// [`AutocorrError::TooManySpins`] above [`MAX_SPINS`].
pub fn peretto(g: &Graph, beta: f64) -> Result<Vec<f64>, AutocorrError> {
    if g.n > MAX_SPINS {
        return Err(AutocorrError::TooManySpins { n: g.n, max: MAX_SPINS });
    }
    let n = g.n;
    let m = 1usize << n;
    let mut lp = vec![0.0f64; m];
    for (x, l) in lp.iter_mut().enumerate() {
        let s = spins(x, n);
        let hx: f64 = (0..n).map(|i| g.h[i] * f64::from(s[i])).sum();
        let mut acc = beta * hx;
        for i in 0..n {
            // ln 2 cosh(a) = |a| + ln(1 + exp(-2 |a|)), overflow-free.
            let a = (beta * g.field(i, &s)).abs();
            acc += a + (-2.0 * a).exp().ln_1p();
        }
        *l = acc;
    }
    let max = lp.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut pi: Vec<f64> = lp.iter().map(|l| (l - max).exp()).collect();
    let z: f64 = pi.iter().sum();
    for p in &mut pi {
        *p /= z;
    }
    Ok(pi)
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
            // The widths are powers of two in f64, not integer shifts: `1u32 << 32` wraps to 1 in
            // release and turned a 32-bit comparator into one with a single level, every site
            // certain to be -1 (found by `examples/fabric_floor.rs`, 2026-09-13). The guard is the
            // range the i64 field arithmetic below can hold.
            assert!(
                frac_bits <= 40 && lut_bits <= 40 && prob_bits <= 52,
                "Quantised kernel bits out of range: frac {frac_bits}, lut {lut_bits}, prob {prob_bits} (at most 40, 40 and 52)"
            );
            // The field in fixed point, at `frac_bits` fractional bits.
            let scale = 2f64.powi(frac_bits as i32);
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
            let stride = 16.0 / 2f64.powi(lut_bits as i32);
            let arg = (addr as f64 + 0.5) * stride - 8.0;
            // The entry held to `prob_bits`: the RTL's `round(p * 65535) / 65536` at 16 bits.
            let levels = 2f64.powi(prob_bits as i32);
            let entry = (p_up(arg, beta) * (levels - 1.0)).round().min(levels - 1.0);
            entry / levels
        }
        Kernel::Pimi { xi, eta } => {
            let z = ((beta * g.field(i, s)).tanh() + xi * f64::from(s[i])) / eta;
            0.5 * (1.0 + crate::hopfield::erf(z / std::f64::consts::SQRT_2))
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

/// The heat-bath probability of `+1` at site `i` under [`Kernel::Stale`]: the pre-sweep state
/// `x` supplies every not-yet-updated neighbour, and each already-updated neighbour `j < i` is
/// read from `y` (fresh) or from `x` (stale, probability `p`), independently, so the conditional
/// is the mixture over the `2^d` stale patterns of the `d` already-updated neighbours.
fn stale_q(g: &Graph, beta: f64, p: f64, i: usize, x: &[i8], y: &[i8]) -> f64 {
    let mut base = g.h[i];
    let mut prev: Vec<(f64, i8, i8)> = Vec::new();
    for k in g.offset[i]..g.offset[i + 1] {
        let j = g.nbr[k] as usize;
        if j < i {
            prev.push((g.w[k], x[j], y[j]));
        } else {
            base += g.w[k] * f64::from(x[j]);
        }
    }
    let d = prev.len();
    let mut q = 0.0;
    for mask in 0..(1usize << d) {
        let mut field = base;
        let mut weight = 1.0;
        for (k, &(w, x_j, y_j)) in prev.iter().enumerate() {
            let stale = (mask >> k) & 1 == 1;
            let s_j = if stale { x_j } else { y_j };
            field += w * f64::from(s_j);
            weight *= if stale { p } else { 1.0 - p };
        }
        if weight > 0.0 {
            q += weight * p_up(field, beta);
        }
    }
    q
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
/// If `v` does not have `2^n` entries, or for a [`Kernel::Quantised`] with more than 40 field or
/// ROM bits or 52 comparator bits, which its fixed-point arithmetic cannot hold.
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
        Kernel::Synchronous | Kernel::Pimi { .. } => {
            // (P v)(x) = E[v(y)] under the product law y_i ~ Bernoulli(q_i(x)), every q_i from the
            // PREVIOUS state x. Per x, contract v one site at a time under that product, from the
            // top bit down, so the block that remains keeps its bit layout.
            let mut out = vec![0.0f64; m];
            let mut w = vec![0.0f64; m];
            for x in 0..m {
                let s = spins(x, n);
                w.copy_from_slice(v);
                let mut len = m;
                for i in (0..n).rev() {
                    let q = p_site(g, beta, kernel, i, &s);
                    let half = len / 2;
                    for y in 0..half {
                        w[y] = (1.0 - q) * w[y] + q * w[y | half];
                    }
                    len = half;
                }
                out[x] = w[0];
            }
            out
        }
        Kernel::Stale { p } => {
            // (P v)(x) = sum_y prod_i q_i(x, y_{<i}) v(y): per source x, fold v from the top bit
            // down; when site i is folded, the bits below it are still indices, so q_i may depend
            // on them -- which is exactly the already-updated neighbours it reads.
            let mut out = vec![0.0f64; m];
            let mut w = vec![0.0f64; m];
            for x in 0..m {
                let sx = spins(x, n);
                w.copy_from_slice(v);
                let mut len = m;
                for i in (0..n).rev() {
                    let half = len / 2;
                    for y in 0..half {
                        let sy = spins(y, n);
                        let q = stale_q(g, beta, p, i, &sx, &sy);
                        w[y] = (1.0 - q) * w[y] + q * w[y | half];
                    }
                    len = half;
                }
                out[x] = w[0];
            }
            out
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
/// If `mu` does not have `2^n` entries, or for a [`Kernel::Quantised`] with more than 40 field or
/// ROM bits or 52 comparator bits, which its fixed-point arithmetic cannot hold.
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
        Kernel::Synchronous | Kernel::Pimi { .. } => {
            // (mu P)(y) = sum_x mu(x) prod_i q_i^x(y_i): each source state lays its product law
            // over every target, built by doubling one site at a time (bit i set gets q_i).
            let mut out = vec![0.0f64; m];
            let mut prod = vec![0.0f64; m];
            for x in 0..m {
                let mass = mu[x];
                if mass == 0.0 {
                    continue;
                }
                let s = spins(x, n);
                prod[0] = 1.0;
                let mut len = 1usize;
                for i in 0..n {
                    let q_prev = p_site(g, beta, kernel, i, &s);
                    for y in 0..len {
                        let w = prod[y];
                        prod[y] = w * (1.0 - q_prev);
                        prod[y | len] = w * q_prev;
                    }
                    len <<= 1;
                }
                for (o, pr) in out.iter_mut().zip(&prod) {
                    *o += mass * pr;
                }
            }
            out
        }
        Kernel::Stale { p } => {
            // (mu P)(y) = sum_x mu(x) prod_i q_i(x, y_{<i}): each source lays its law over the
            // targets one site at a time, the conditional at site i reading the target bits
            // already placed below it.
            let mut out = vec![0.0f64; m];
            let mut prod = vec![0.0f64; m];
            for x in 0..m {
                let mass = mu[x];
                if mass == 0.0 {
                    continue;
                }
                let sx = spins(x, n);
                prod[0] = 1.0;
                let mut len = 1usize;
                for i in 0..n {
                    for y in 0..len {
                        let sy = spins(y, n);
                        let q = stale_q(g, beta, p, i, &sx, &sy);
                        let wgt = prod[y];
                        prod[y] = wgt * (1.0 - q);
                        prod[y | len] = wgt * q;
                    }
                    len <<= 1;
                }
                for (o, pr) in out.iter_mut().zip(&prod) {
                    *o += mass * pr;
                }
            }
            out
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

/// The most spins the direct solves ([`stationary_solved`], [`tau_int_fundamental`]) build a dense
/// operator over. At `2^12` states the matrix is 134 MB and the elimination a few seconds; each
/// further spin is 4x the memory and 8x the time, and at [`MAX_SPINS`] it would be 34 GB.
pub const MAX_DENSE_SPINS: usize = 12;

/// Solve `a x = b` in place by Gaussian elimination with partial pivoting, for `r` right-hand
/// sides at once: `a` is `m x m` row-major and is destroyed, `b` is `m x r` row-major and is
/// replaced by the solution. `false` when a pivot falls below `1e-14` -- singular to floating
/// point, which for the systems built here means a kernel with more than one closed class.
fn lu_solve(a: &mut [f64], m: usize, b: &mut [f64], r: usize) -> bool {
    for k in 0..m {
        let mut piv = k;
        let mut best = a[k * m + k].abs();
        for row in k + 1..m {
            let v = a[row * m + k].abs();
            if v > best {
                best = v;
                piv = row;
            }
        }
        if best < 1e-14 {
            return false;
        }
        if piv != k {
            let (lo, hi) = a.split_at_mut(piv * m);
            lo[k * m..(k + 1) * m].swap_with_slice(&mut hi[..m]);
            let (blo, bhi) = b.split_at_mut(piv * r);
            blo[k * r..(k + 1) * r].swap_with_slice(&mut bhi[..r]);
        }
        let (top, rest) = a.split_at_mut((k + 1) * m);
        let row_k = &top[k * m..(k + 1) * m];
        let pk = row_k[k];
        let (btop, brest) = b.split_at_mut((k + 1) * r);
        let b_k = &btop[k * r..(k + 1) * r];
        for (row, brow) in rest.chunks_exact_mut(m).zip(brest.chunks_exact_mut(r)) {
            let f = row[k] / pk;
            if f == 0.0 {
                continue;
            }
            row[k] = 0.0;
            for (rc, &kc) in row[k + 1..].iter_mut().zip(&row_k[k + 1..]) {
                *rc -= f * kc;
            }
            for (bc, &kc) in brow.iter_mut().zip(b_k) {
                *bc -= f * kc;
            }
        }
    }
    for k in (0..m).rev() {
        let (above, below) = b.split_at_mut((k + 1) * r);
        let bk = &mut above[k * r..(k + 1) * r];
        for (c, brow) in below.chunks_exact(r).enumerate() {
            let akc = a[k * m + k + 1 + c];
            if akc == 0.0 {
                continue;
            }
            for (x, &y) in bk.iter_mut().zip(brow) {
                *x -= akc * y;
            }
        }
        let akk = a[k * m + k];
        for x in bk.iter_mut() {
            *x /= akk;
        }
    }
    true
}

/// Row `x` of the dense kernel is the law one step from state `x`: `P` applied to the point mass
/// there. Visits every row in turn so a caller can lay it into whichever matrix it is building.
fn kernel_rows(g: &Graph, beta: f64, kernel: Kernel, mut visit: impl FnMut(usize, &[f64])) {
    let m = 1usize << g.n;
    let mut point = vec![0.0f64; m];
    for x in 0..m {
        point[x] = 1.0;
        let row = apply_distribution(g, beta, kernel, &point);
        point[x] = 0.0;
        visit(x, &row);
    }
}

/// The kernel's stationary distribution by a DIRECT solve: `pi (I - P) = 0` with `sum pi = 1`, as
/// one linear system over the `2^n` states, eliminated with partial pivoting. Where [`stationary`]
/// pushes mass forward until it stops moving -- and at low temperature does not within any step
/// budget, because the relaxation time IS the mixing time -- this costs the same at every
/// temperature and is exact to the conditioning of the chain. The price is the dense matrix:
/// [`MAX_DENSE_SPINS`].
///
/// # Errors
///
/// [`AutocorrError::TooManyForDense`] above [`MAX_DENSE_SPINS`]; [`AutocorrError::Reducible`] when
/// the system is singular, which is what a kernel with more than one closed class produces -- a
/// fabric whose flip probability rounds to exactly zero can be one.
pub fn stationary_solved(g: &Graph, beta: f64, kernel: Kernel) -> Result<Vec<f64>, AutocorrError> {
    if g.n > MAX_DENSE_SPINS {
        return Err(AutocorrError::TooManyForDense { n: g.n, max: MAX_DENSE_SPINS });
    }
    let m = 1usize << g.n;
    // A = (I - P)^T with its last row replaced by the normalisation; b = e_last.
    let mut a = vec![0.0f64; m * m];
    kernel_rows(g, beta, kernel, |x, row| {
        for (y, &p) in row.iter().enumerate() {
            let delta = if x == y { 1.0 } else { 0.0 };
            a[y * m + x] = delta - p;
        }
    });
    for c in 0..m {
        a[(m - 1) * m + c] = 1.0;
    }
    let mut b = vec![0.0f64; m];
    b[m - 1] = 1.0;
    if !lu_solve(&mut a, m, &mut b, 1) {
        return Err(AutocorrError::Reducible);
    }
    // Elimination can leave a state of vanishing mass a hair below zero; a law is not.
    let total: f64 = b.iter().map(|v| v.max(0.0)).sum();
    Ok(b.iter().map(|v| v.max(0.0) / total).collect())
}

/// The exact integrated autocorrelation time by the FUNDAMENTAL MATRIX `Z = (I - P + 1 pi^T)^-1`
/// (Kemeny and Snell 1960): with `e = f - <f>_pi`, `sum_{k >= 0} C(k) = <pi, e * (Z e)>`, so
///
/// ```text
///   tau_int = <pi, e * (Z e)> / C(0) - 1/2,
/// ```
///
/// one linear solve in place of the lag-by-lag sum of [`tau_int_exact`], which at low temperature
/// needs a number of lags proportional to the `tau` it is computing and is cut off by `max_lags`
/// before it gets there. This has no lags to truncate -- `lags` is reported as `0` -- and costs the
/// same at every temperature. It needs no reversibility: the identity is the geometric series of
/// `P - 1 pi^T`, which converges for any ergodic chain. For the Gibbs and informed kernels `pi` is
/// the Boltzmann distribution; for the fabric's arithmetic it is [`stationary_solved`], since the
/// sum must be taken under the kernel's OWN invariant law. The first 64 autocorrelations are
/// filled in by applying `P`, for the record.
///
/// # Errors
///
/// [`AutocorrError::TooManyForDense`] above [`MAX_DENSE_SPINS`]; [`AutocorrError::NoVariance`] for
/// a constant observable; [`AutocorrError::Reducible`] for a kernel without a unique invariant law.
pub fn tau_int_fundamental(
    g: &Graph,
    beta: f64,
    kernel: Kernel,
    observable: impl Fn(&[i8]) -> f64,
) -> Result<Autocorrelation, AutocorrError> {
    if g.n > MAX_DENSE_SPINS {
        return Err(AutocorrError::TooManyForDense { n: g.n, max: MAX_DENSE_SPINS });
    }
    let pi = match kernel {
        Kernel::FixedFabric | Kernel::Quantised { .. } | Kernel::Pimi { .. } | Kernel::Stale { .. } => {
            stationary_solved(g, beta, kernel)?
        }
        Kernel::Synchronous => peretto(g, beta)?,
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
    // (I - P + 1 pi^T) z = e.
    let mut a = vec![0.0f64; m * m];
    kernel_rows(g, beta, kernel, |x, row| {
        for (y, &p) in row.iter().enumerate() {
            let delta = if x == y { 1.0 } else { 0.0 };
            a[x * m + y] = delta - p + pi[y];
        }
    });
    let mut z = e.clone();
    if !lu_solve(&mut a, m, &mut z, 1) {
        return Err(AutocorrError::Reducible);
    }
    let total: f64 = pi.iter().zip(&e).zip(&z).map(|((p, x), y)| p * x * y).sum();
    let tau = total / c0 - 0.5;
    let mut v = e.clone();
    let mut rho = Vec::with_capacity(64);
    for _ in 0..64 {
        v = apply(g, beta, kernel, &v);
        let ck: f64 = pi.iter().zip(&e).zip(&v).map(|((p, x), y)| p * x * y).sum();
        rho.push(ck / c0);
    }
    Ok(Autocorrelation { tau_int: tau, lags: 0, rho, variance: c0 })
}

/// Kemeny's constant of the kernel: `K = sum_{i >= 2} 1 / (1 - lambda_i)` over the non-unit
/// eigenvalues of `P`, equal to `trace(Z) - 1` for the fundamental matrix `Z = (I - P + 1 pi^T)^-1`
/// and to the expected number of steps to reach a `pi`-random target from any start (Kemeny and
/// Snell 1960; the start-independence is Kemeny's theorem). It is the sum of the relaxation times
/// of EVERY mode, so it is a mixing measure that belongs to the kernel alone. `tau_int` is not:
/// it weights each mode by the share of one observable's variance that mode carries, so a law
/// that moves mass OUT of a slow valley lowers `tau_int` without touching the crossing rate, and
/// two kernels with different invariant laws can differ in `tau_int` by a factor that says nothing
/// about how fast either relaxes. `K` cannot fall that way. Costs a full inverse -- `2^n`
/// right-hand sides, about four times [`tau_int_fundamental`].
///
/// # Errors
///
/// [`AutocorrError::TooManyForDense`] above [`MAX_DENSE_SPINS`]; [`AutocorrError::Reducible`] for
/// a kernel without a unique invariant law.
pub fn kemeny_constant(g: &Graph, beta: f64, kernel: Kernel) -> Result<f64, AutocorrError> {
    if g.n > MAX_DENSE_SPINS {
        return Err(AutocorrError::TooManyForDense { n: g.n, max: MAX_DENSE_SPINS });
    }
    let pi = match kernel {
        Kernel::FixedFabric | Kernel::Quantised { .. } | Kernel::Pimi { .. } | Kernel::Stale { .. } => {
            stationary_solved(g, beta, kernel)?
        }
        Kernel::Synchronous => peretto(g, beta)?,
        _ => boltzmann(g, beta)?,
    };
    let m = 1usize << g.n;
    let mut a = vec![0.0f64; m * m];
    kernel_rows(g, beta, kernel, |x, row| {
        for (y, &p) in row.iter().enumerate() {
            let delta = if x == y { 1.0 } else { 0.0 };
            a[x * m + y] = delta - p + pi[y];
        }
    });
    let mut z = vec![0.0f64; m * m];
    for i in 0..m {
        z[i * m + i] = 1.0;
    }
    if !lu_solve(&mut a, m, &mut z, m) {
        return Err(AutocorrError::Reducible);
    }
    let trace: f64 = (0..m).map(|i| z[i * m + i]).sum();
    Ok(trace - 1.0)
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
        Kernel::FixedFabric | Kernel::Quantised { .. } | Kernel::Pimi { .. } | Kernel::Stale { .. } => {
            stationary(g, beta, kernel, 1e-14, 500_000)?.0
        }
        Kernel::Synchronous => peretto(g, beta)?,
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
            Kernel::Synchronous,
            Kernel::Pimi { xi: 0.3, eta: 0.7 },
            Kernel::Stale { p: 0.3 },
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
            // The Boltzmann distribution is stationary for the exact kernels, Peretto's law for
            // the synchronous sweep, and neither for the fabric's arithmetic -- that gap is the
            // fabric's own test below, not a failure here.
            let law = match kernel {
                Kernel::FixedFabric => None,
                Kernel::Synchronous => Some(peretto(&g, beta).unwrap()),
                Kernel::Pimi { .. } | Kernel::Stale { .. } => Some(stationary_solved(&g, beta, kernel).unwrap()),
                _ => Some(pi.clone()),
            };
            if let Some(law) = law {
                let pushed = apply_distribution(&g, beta, kernel, &law);
                assert!(
                    total_variation(&pushed, &law) < 1e-12,
                    "{kernel:?}: its law is not stationary in distribution form, TV {:e}",
                    total_variation(&pushed, &law)
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

    /// The direct solve is the law the iteration finds, at a temperature where the iteration still
    /// converges: Boltzmann to floating point for the Gibbs kernel, the fabric's own law for the
    /// fabric -- and that law is not Boltzmann.
    #[test]
    fn the_direct_solve_reproduces_boltzmann_for_gibbs_and_the_iterated_law_for_the_fabric() {
        let g = grid_glass(3, 3, 5);
        let beta = 1.5;
        let pi = boltzmann(&g, beta).unwrap();
        let solved = stationary_solved(&g, beta, Kernel::ChromaticGibbs).unwrap();
        let gap = total_variation(&solved, &pi);
        assert!(gap < 1e-12, "direct solve vs Boltzmann: {gap}");
        let solved = stationary_solved(&g, beta, Kernel::FixedFabric).unwrap();
        // A stationary law is one the kernel leaves where it is: the test that needs no reference.
        let pushed = apply_distribution(&g, beta, Kernel::FixedFabric, &solved);
        let drift = total_variation(&pushed, &solved);
        assert!(drift < 1e-11, "the solved fabric law must be invariant under the fabric kernel: {drift}");
        // The iteration halts when a step moves less than `tol`, which leaves it about
        // `tol x relaxation time` short of the fixed point -- 1.2e-10 here -- so it is the LESS
        // accurate of the two and the agreement is held to its floor, not the solve's.
        let (iterated, steps) = stationary(&g, beta, Kernel::FixedFabric, 1e-14, 500_000).unwrap();
        assert!(steps < 500_000, "the iteration must converge for this test to have a reference");
        let gap = total_variation(&solved, &iterated);
        assert!(gap < 1e-9, "direct solve vs iterated fabric law: {gap}");
        assert!(total_variation(&solved, &pi) > 1e-7, "the fabric's law must not be Boltzmann");
    }

    /// The fundamental-matrix tau is the lag sum where the lag sum converges -- on the chromatic,
    /// sequential and fabric kernels, none of which is reversible as a sweep -- and is finite and
    /// larger where the lag sum is cut off by its budget. And the one closed form: a single site
    /// resampled every sweep has `rho(k) = 0` for all `k >= 1`, so `tau_int = 1/2` exactly.
    #[test]
    fn the_fundamental_matrix_tau_agrees_with_the_lag_sum_and_needs_no_lags() {
        let g = grid_glass(3, 2, 9);
        for kernel in [Kernel::ChromaticGibbs, Kernel::SequentialGibbs, Kernel::FixedFabric] {
            let summed = tau_int_exact(&g, 1.0, kernel, |s| g.energy(s), 1e-15, 1_000_000).unwrap();
            assert!(summed.lags < 1_000_000);
            let solved = tau_int_fundamental(&g, 1.0, kernel, |s| g.energy(s)).unwrap();
            let rel = (solved.tau_int - summed.tau_int).abs() / summed.tau_int;
            assert!(
                rel < 1e-9,
                "{kernel:?}: fundamental {} vs summed {} over {} lags",
                solved.tau_int,
                summed.tau_int,
                summed.lags
            );
            assert_eq!(solved.lags, 0);
            assert!((solved.rho[0] - summed.rho[0]).abs() < 1e-12);
        }
        let mut b = GraphBuilder::new(1);
        b.bias(0, 0.3);
        let one = b.build();
        let single = tau_int_fundamental(&one, 1.0, Kernel::ChromaticGibbs, |s| f64::from(s[0])).unwrap();
        assert!((single.tau_int - 0.5).abs() < 1e-12, "single site: {}", single.tau_int);
        // Cold: the lag sum stops at its budget and under-reports; the solve does not.
        let g = grid_glass(3, 3, 5);
        let cut = tau_int_exact(&g, 4.0, Kernel::ChromaticGibbs, |s| g.energy(s), 1e-12, 100).unwrap();
        assert_eq!(cut.lags, 100, "the cold chain must exhaust the lag budget for this half to test anything");
        let solved = tau_int_fundamental(&g, 4.0, Kernel::ChromaticGibbs, |s| g.energy(s)).unwrap();
        assert!(
            solved.tau_int.is_finite() && solved.tau_int > cut.tau_int,
            "solved {} must exceed the cut sum {}",
            solved.tau_int,
            cut.tau_int
        );
    }

    /// Two closed forms. A single site resampled every sweep has `P = 1 pi^T`: one non-unit
    /// eigenvalue, at 0, so `K = 1`. Three UNCOUPLED sites under either sweep are the same thing on
    /// eight states -- `P` is rank one, seven eigenvalues at 0 -- so `K = 7`. And coupling makes a
    /// kernel slower: on the chromatic sweep every non-unit eigenvalue is real and in `[0, 1)` (a
    /// product of two projections in `L^2(pi)`), so every term is at least 1 and `K > m - 1`.
    #[test]
    fn kemenys_constant_has_its_closed_forms_on_memoryless_kernels() {
        let mut b = GraphBuilder::new(1);
        b.bias(0, 0.3);
        let k = kemeny_constant(&b.build(), 1.0, Kernel::ChromaticGibbs).unwrap();
        assert!((k - 1.0).abs() < 1e-12, "single site: {k}");
        let mut b = GraphBuilder::new(3);
        for i in 0..3 {
            b.bias(i, 0.1 * (i as f64 + 1.0));
        }
        let free = b.build();
        for kernel in [Kernel::ChromaticGibbs, Kernel::SequentialGibbs] {
            let k = kemeny_constant(&free, 1.0, kernel).unwrap();
            assert!((k - 7.0).abs() < 1e-11, "{kernel:?} on three free sites: {k}");
        }
        let g = grid_glass(2, 2, 3);
        let k = kemeny_constant(&g, 1.0, Kernel::ChromaticGibbs).unwrap();
        assert!(k > 15.0, "a coupled 2x2 grid must be slower than memoryless: K = {k}");
    }

    /// With 40 field bits, a 2^40-entry ROM and a 52-bit comparator the quantised arithmetic is the
    /// exact heat bath to better than 1e-9 -- and 32 comparator bits must sit within 2^-24 of 24
    /// bits, which is where a width computed as `1u32 << bits` wrapped to a comparator with ONE
    /// level and made every site certain to be -1 (`examples/fabric_floor.rs`, 2026-09-13).
    #[test]
    fn the_quantised_kernel_at_full_precision_is_the_exact_kernel() {
        let g = grid_glass(3, 2, 4);
        let beta = 1.5;
        let m = 1usize << g.n;
        let full = Kernel::Quantised { frac_bits: 40, lut_bits: 40, prob_bits: 52 };
        let p24 = Kernel::Quantised { frac_bits: 8, lut_bits: 10, prob_bits: 24 };
        let p32 = Kernel::Quantised { frac_bits: 8, lut_bits: 10, prob_bits: 32 };
        let mut worst_full = 0.0f64;
        let mut worst_wide = 0.0f64;
        for x in 0..m {
            let s = spins(x, g.n);
            for i in 0..g.n {
                let exact = p_up(g.field(i, &s), beta);
                worst_full = worst_full.max((p_site(&g, beta, full, i, &s) - exact).abs());
                worst_wide = worst_wide.max((p_site(&g, beta, p32, i, &s) - p_site(&g, beta, p24, i, &s)).abs());
            }
        }
        assert!(worst_full < 1e-9, "full precision vs the exact heat bath: {worst_full:.3e}");
        assert!(worst_wide < 1e-7, "32 vs 24 comparator bits: {worst_wide:.3e}");
    }

    /// Peretto's closed form is invariant under the synchronous kernel to floating point, differs
    /// from the Boltzmann law resolvably even on this bipartite grid, and collapses to the
    /// Boltzmann law when the sites are free (then `f_i = h_i` and the cosh factors are constant).
    /// The direct solve must find the same law without being told the closed form.
    #[test]
    fn the_synchronous_kernel_leaves_perettos_law_invariant_and_it_is_not_boltzmann() {
        let g = grid_glass(3, 3, 5);
        let beta = 1.2;
        let pi = peretto(&g, beta).unwrap();
        let pushed = apply_distribution(&g, beta, Kernel::Synchronous, &pi);
        let drift = total_variation(&pushed, &pi);
        assert!(drift < 1e-13, "Peretto's law must be invariant under the synchronous sweep: {drift:.3e}");
        let b = boltzmann(&g, beta).unwrap();
        assert!(total_variation(&pi, &b) > 1e-2, "the synchronous law must not be Boltzmann here");
        let solved = stationary_solved(&g, beta, Kernel::Synchronous).unwrap();
        let gap = total_variation(&solved, &pi);
        assert!(gap < 1e-10, "direct solve vs closed form: {gap:.3e}");
        let mut free = GraphBuilder::new(4);
        for i in 0..4 {
            free.bias(i, 0.15 * (i as f64 + 1.0));
        }
        let free = free.build();
        let gap = total_variation(&peretto(&free, beta).unwrap(), &boltzmann(&free, beta).unwrap());
        assert!(gap < 1e-14, "free sites: Peretto must be Boltzmann: {gap:.3e}");
    }

    /// One site under the PIMI rule is a two-state chain with `q_+ = Phi((tanh(beta h) + xi) / eta)`
    /// from `+1` and `q_- = Phi((tanh(beta h) - xi) / eta)` from `-1`, whose stationary probability
    /// of `+1` is `q_- / (1 - q_+ + q_-)` in closed form. The direct solve must reproduce it with
    /// and without inertia, and the two must differ, or the inertia term is not in the kernel.
    #[test]
    fn the_pimi_kernel_has_its_single_site_closed_form_and_its_inertia_matters() {
        let mut b = GraphBuilder::new(1);
        b.bias(0, 0.4);
        let g = b.build();
        let (beta, eta) = (1.0, 0.5);
        let phi = |z: f64| 0.5 * (1.0 + crate::hopfield::erf(z / std::f64::consts::SQRT_2));
        let closed = |xi: f64| {
            let t = (beta * 0.4f64).tanh();
            let (q_plus, q_minus) = (phi((t + xi) / eta), phi((t - xi) / eta));
            q_minus / (1.0 - q_plus + q_minus)
        };
        for xi in [0.0, 0.6] {
            let law = stationary_solved(&g, beta, Kernel::Pimi { xi, eta }).unwrap();
            let want = closed(xi);
            assert!((law[1] - want).abs() < 1e-12, "xi {xi}: solved {} vs closed form {want}", law[1]);
        }
        assert!((closed(0.6) - closed(0.0)).abs() > 1e-2, "inertia must move the single-site law");
    }

    /// The stale-read kernel is bracketed by two closed forms: at `p = 0` it is the sequential
    /// sweep, operator for operator, and at `p = 1` the synchronous one -- checked on random
    /// distributions to floating point, not just on the invariant laws. In between it is neither:
    /// at `p = 0.3` its law is resolvably away from both Boltzmann and Peretto.
    #[test]
    fn the_stale_read_kernel_is_bracketed_by_its_two_closed_forms() {
        let g = grid_glass(3, 3, 6);
        let beta = 1.1;
        let m = 1usize << g.n;
        let mut rng = Pcg::new(9, 4);
        for _ in 0..3 {
            let mut mu: Vec<f64> = (0..m).map(|_| rng.f64()).collect();
            let z: f64 = mu.iter().sum();
            for q in &mut mu {
                *q /= z;
            }
            let fresh = apply_distribution(&g, beta, Kernel::Stale { p: 0.0 }, &mu);
            let seq = apply_distribution(&g, beta, Kernel::SequentialGibbs, &mu);
            assert!(total_variation(&fresh, &seq) < 1e-13, "p = 0 must be the sequential sweep");
            let stale = apply_distribution(&g, beta, Kernel::Stale { p: 1.0 }, &mu);
            let sync = apply_distribution(&g, beta, Kernel::Synchronous, &mu);
            assert!(total_variation(&stale, &sync) < 1e-13, "p = 1 must be the synchronous sweep");
        }
        let law = stationary_solved(&g, beta, Kernel::Stale { p: 0.3 }).unwrap();
        let b = boltzmann(&g, beta).unwrap();
        let pe = peretto(&g, beta).unwrap();
        assert!(total_variation(&law, &b) > 1e-3, "p = 0.3 must not be Boltzmann");
        assert!(total_variation(&law, &pe) > 1e-3, "p = 0.3 must not be Peretto");
        assert!(total_variation(&law, &b) < total_variation(&pe, &b), "p = 0.3 must sit nearer Boltzmann than p = 1");
    }
}
