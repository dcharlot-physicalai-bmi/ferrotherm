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
//! is never summed. No trace length repairs a window that has already closed. That seventeen-fold
//! under-report is specific to the `beta = 1` spectrum: with the oracle moved to the fundamental
//! matrix and the estimators to the FFT (2026-09-13, `tau_exactness` v3), at `beta = 1.5`
//! (`tau = 13,270`) Sokal reads `0.67` to `0.77` of the truth and at `beta = 2` (`tau = 1.6e6`)
//! `0.70` at thirty `tau`, while batch means over twenty batches climb from `0.21` at thirty `tau`
//! to `0.91` at a thousand at `beta = 1.5` and reach `0.99` at ten thousand `tau` at `beta = 1`,
//! where Sokal still reads `0.06`. The batch-means cross-check in [`crate::certify`] exists for
//! exactly that spectrum.
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
//! rounds the probability of the unlikely state — `sigma(-2 beta f)`, with the heat bath's factor
//! of two — to 0 below `7.6e-6`, that is once `2 beta f > 11.8`, when that state is `+1`, and never
//! below `1.53e-5` when it is `-1`, because the entry cannot exceed 65535. At `beta = 2` every
//! field above `2.95` already forbids its flip: on the 4x3 grid 6.2 million of the 16.8 million
//! transitions are one-way and 848 states unreachable, so the fabric's entropy production there
//! is infinite; and at `beta = 3` a flip against a field of 4.2, exact probability `1e-11`, is
//! impossible on one side and `1.53e-5` — a million times too likely — on the other. An escape
//! from a metastable valley that needs several such flips compounds the factor. The same floor
//! caps the law: in the precision sweep
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
    /// One RANDOM-SCAN step: a site chosen uniformly at random, resampled from its heat-bath
    /// conditional -- the Glauber dynamics the mixing-time literature analyses, and what a fabric
    /// would run if it chose where to update with a random number instead of a fixed order. `n`
    /// steps do as many site updates as one [`Kernel::SequentialGibbs`] sweep, which is the
    /// comparison `examples/scan_order_exact.rs` makes. Reversible with respect to the Boltzmann
    /// law, where the fixed-order sweep is only invariant.
    RandomScan,
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
    /// test holds both ends to their closed forms. `examples/stale_exact.rs` measures the middle,
    /// exactly. On the 4x3 grid (degree 4) the equilibrium error is linear in `p` to `p = 0.3`:
    /// `TV = 0.39 p, 0.105 p, 0.048 p` at `beta = 0.5, 1, 2`. What it IS changes with temperature:
    /// at `beta = 0.5` and `1` it is a change of law -- the nearest single temperature removes
    /// only a fifth of it, and the shift is `-0.16 p` and `+0.11 p` of `beta` -- while at
    /// `beta = 2` it is almost purely a temperature shift, `+0.21 p` of `beta`, of which the
    /// nearest temperature removes 94%. On `K_{6,6}` (degree 6) the same pattern with a larger
    /// cold shift, `+2.4 p` of `beta` at `beta = 2`, and the sweep relaxes 3.3x slower at
    /// `p = 0.3`. So a fabric that reads 1% of its neighbours stale pays TV `3.9e-3` at
    /// `beta = 0.5` on the grid, twice its whole arithmetic error, and no effective-temperature
    /// certificate can see it there; at `beta = 2` the same 1% is a `0.2%` (grid) to `2.4%`
    /// (`K_{6,6}`) temperature error, which the certificate does see and a calibration removes.
    Stale {
        /// Probability that a read of an already-updated neighbour returns its pre-sweep value.
        p: f64,
    },
    /// TICK-RANDOM, the synchronous policy of Onizawa & Hanyu (arXiv:2604.01564, 2026): at each
    /// global tick an independent Bernoulli mask selects each spin with probability
    /// `p_flip = 1/c`, and every selected spin is resampled from its heat-bath conditional given
    /// the PREVIOUS state. `c` is their time-multiplexing reuse factor -- "the number of logical
    /// p-bits that are sequentially mapped onto a single physical p-bit" -- so `c >= 1` and
    /// `p` in `(0, 1]`. At `p = 1` every site moves every tick, which is exactly
    /// [`Kernel::Synchronous`]; as `p` falls, ticks in which two adjacent sites move together
    /// become rare and the law approaches the single-site Boltzmann limit. `p = 0` is the identity
    /// map, which has no unique invariant law, and [`stationary_solved`] says so rather than
    /// returning one.
    ///
    /// # The claim this kernel exists to settle
    ///
    /// The paper's Discussion argues reuse is free: *"time-multiplexed reuse corresponds to a
    /// temporal rescaling of the underlying Markov process rather than a change in its transition
    /// kernel"*, and *"Such time-thinning arguments are well established in stochastic simulation
    /// theory and imply that only the convergence speed, not the stationary distribution, is
    /// affected."* The Conclusion restates it: *"the effective update rate can be reduced without
    /// altering the target stationary distribution"*.
    ///
    /// **For this branch that is false, and `examples/tick_random_exact.rs` measures by how much.**
    /// The thinning theorem is sound for a continuous-time chain where at most one site moves at a
    /// time -- their Poisson/asynchronous branch. A Bernoulli mask is not a thinning of that chain:
    /// it puts probability `p^2` on two ADJACENT sites moving from the same stale state, and that
    /// term is what breaks detailed balance. It sits inside the per-site conditional, so changing
    /// `c` changes the kernel itself, not only the clock.
    TickRandom {
        /// Probability that a given spin is selected for update on a tick: the paper's
        /// `p_flip = 1/c`. Must lie in `(0, 1]`; `0` is the identity map and has no unique
        /// invariant law.
        p: f64,
    },
    /// STOCHASTIC CELLULAR AUTOMATA (SCA), the all-spins-at-once rule behind the STATICA (ISSCC
    /// 2020) and Amorphica (ISSCC 2023) annealing processors, as defined by Handa, Kamakura,
    /// Kamijima and Sakai (arXiv:1906.06645) and Fukushima-Kimura et al. (arXiv:2007.11287,
    /// J. Stat. Phys. 190:79, 2023): every site resampled at once from the PREVIOUS state with
    /// `P(s_x = +1 | x) = e^a / 2 cosh a`, `a = (beta / 2) f_x(x) + q x_x`. Two changes from
    /// [`Kernel::Synchronous`]: the field is HALVED, and each spin feels a pull `q` toward its own
    /// current value -- the "pinning" that makes a same-tick update rarely move two neighbours at
    /// once. Its law is closed-form ([`sca_law`]), reversible, and tends to the Boltzmann law at
    /// `beta` as `q -> inf`; `q = 0` is the synchronous sweep at `beta / 2`.
    ///
    /// It is also Momentum Annealing's construction (Hitachi, 2019): the pair weight
    /// `exp(beta/2 sum J x_i y_j + beta/2 h.(x + y) + q x.y)` is a Boltzmann law on the bipartite
    /// double of the graph, one tick of SCA is one half-sweep of block Gibbs on that double, and
    /// [`sca_law`] is its marginal. The test holds the law to that marginal, computed by
    /// enumeration over `2n` spins, and to the exact first-order rate at which it approaches
    /// Boltzmann: `TV ~ e^{-2q} * E_G|Phi - <Phi>| / 2` with `Phi(x) = sum_i exp(-beta f_i x_i)`.
    ///
    /// # The question this kernel exists to settle
    ///
    /// Pinning trades accuracy for motion: a large `q` brings the law to Boltzmann and freezes the
    /// chain, a small one moves many spins per tick and samples something else. The published
    /// guarantees are sufficient conditions on the two sides. K. Kamakura's Hokkaido note
    /// *確率的セルラオートマタによる最適解の探索* (Finding optimal solutions by stochastic cellular
    /// automata, 2020) proves more flips per step than Glauber when `2q <= ln|V| - beta K`
    /// (its Theorem 1, `K` the largest local field) and closeness to Gibbs, in its
    /// order-preservation sense, when `2q >= ln|V| + beta K - ln(eps sqrt(v) / 2K)` (its
    /// Theorem 3, `v` the sum of squared couplings and fields). Those two windows overlap only
    /// when `eps sqrt(v) >= 2K e^{2 beta K}`. `examples/sca_exact.rs` measures the frontier
    /// between them exactly, in ticks, against the coloured sweep a p-bit fabric already runs.
    Sca {
        /// The pinning (self-interaction) strength, in units of the exponent: dimensionless, and
        /// NOT multiplied by `beta`. `q >= 0`.
        q: f64,
    },
    /// The chromatic sweep with every p-bit at its OWN temperature: site `i` samples its heat-bath
    /// conditional at `beta * exp(spread * z_i)`, `z_i` a standard normal drawn from `seed` -- the
    /// gain (MTJ 'alpha') spread of a fabric whose sigmoid slopes differ cell to cell. The sweep
    /// has no common invariant law: each site's update is reversible only for its own
    /// temperature, and a Boltzmann law with couplings `beta_i J_ij` would have to be symmetric in
    /// `i` and `j`, so the stationary law is not the Boltzmann distribution of any pair.
    /// [`stationary_solved`] gives it; `examples/spread_exact.rs` measures how far, and from what.
    /// On the 4x3 grid at `beta = 1` the non-Boltzmann residual -- the KL no pair of couplings and
    /// fields reproduces -- is `0.09` to `0.10` times `spread^2` up to a spread of `0.1`, rising to
    /// `0.15 spread^2` at `0.4`; it is a fifth of the loss at the loaded couplings throughout, so a
    /// calibrated pair removes four fifths, moving couplings by about `1.7 spread` per unit `beta`,
    /// while the nearest single temperature moves under 1%. At `beta = 2` the residual is
    /// `3e-4 spread^2` and under half a percent of the loss: when cold the spread is almost purely a
    /// Boltzmann distortion, mostly a global temperature error (`beta_eff` from `1.95` to `1.42` for
    /// spreads `0.02` to `0.4`), and the sweep relaxes up to twice as slowly. The mean-temperature
    /// guess `J' = J (beta_i + beta_j) / 2`, `h' = beta_i h` removes only `36%` at `beta = 1` and
    /// `24%` at `beta = 2` of the loss at spread `0.2`, against the fitted pair's `78%` and `99.9%`.
    SiteSpread {
        /// Seed of the per-site factors.
        seed: u64,
        /// Standard deviation of `ln(beta_i / beta)`.
        spread: f64,
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

/// The stationary law of [`Kernel::Sca`] in closed form:
///
/// ```text
///   pi(x) ∝ exp(beta/2 h·x) prod_i 2 cosh(beta/2 f_i(x) + q x_i),
/// ```
///
/// `f_i` the local field at `x` with the bias included. It is the marginal over `y` of the pair
/// weight `exp(beta/2 [y·J·x + h·(x + y)] + q x·y)`, and `pi(x) P(x, y)` equals that pair weight,
/// which is symmetric in `x` and `y` -- so SCA is reversible with respect to it. At `q = 0` it is
/// [`peretto`] at `beta / 2`. Dividing by the Boltzmann weight gives the form that shows the limit
/// (Fukushima-Kimura et al., and the proof of Kamakura's Theorem 3):
/// `pi(x) ∝ e^{-beta E(x)} prod_i (1 + e^{-2q} e^{-beta f_i(x) x_i})`.
///
/// # Errors
///
/// [`AutocorrError::TooManySpins`] above [`MAX_SPINS`].
///
/// # Panics
///
/// If `q` is negative or not finite: a negative pinning pushes each spin AWAY from its own value
/// and is not the kernel the hardware or the theorems describe.
pub fn sca_law(g: &Graph, beta: f64, q: f64) -> Result<Vec<f64>, AutocorrError> {
    assert!(q.is_finite() && q >= 0.0, "SCA pinning q must be finite and non-negative, got {q}");
    if g.n > MAX_SPINS {
        return Err(AutocorrError::TooManySpins { n: g.n, max: MAX_SPINS });
    }
    let n = g.n;
    let m = 1usize << n;
    let mut lp = vec![0.0f64; m];
    for (x, l) in lp.iter_mut().enumerate() {
        let s = spins(x, n);
        let hx: f64 = (0..n).map(|i| g.h[i] * f64::from(s[i])).sum();
        let mut acc = 0.5 * beta * hx;
        for i in 0..n {
            let a = (0.5 * beta * g.field(i, &s) + q * f64::from(s[i])).abs();
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
/// The per-site temperature factor of [`Kernel::SiteSpread`]: `exp(spread * z_i)` with `z_i` a
/// standard normal from `seed` and the site index, by Box-Muller on a private stream, so the same
/// `(seed, spread)` names the same fabric in every call.
#[must_use]
pub fn site_factor(seed: u64, spread: f64, i: usize) -> f64 {
    let mut r = crate::rng::Pcg::new(seed.wrapping_add(1_000 * i as u64 + 1), 0x5B);
    let u1 = r.f64().max(1e-300);
    let u2 = r.f64();
    let z = (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
    (spread * z).exp()
}

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
        Kernel::SiteSpread { seed, spread } => p_up(g.field(i, s), beta * site_factor(seed, spread, i)),
        Kernel::Pimi { xi, eta } => {
            let z = ((beta * g.field(i, s)).tanh() + xi * f64::from(s[i])) / eta;
            0.5 * (1.0 + crate::hopfield::erf(z / std::f64::consts::SQRT_2))
        }
        // EXPLICIT, not folded into the catch-all: the arm below is the plain heat bath, which is
        // exactly `TickRandom { p: 1.0 }`, so a missing arm here would run every p as Synchronous
        // and the whole sweep would read as "the law does not move with p" -- the paper's claim,
        // produced by our own omission.
        Kernel::TickRandom { p } => {
            assert!((0.0..=1.0).contains(&p), "TickRandom p is a probability, got {p}");
            let stay = if s[i] > 0 { 1.0 } else { 0.0 };
            (1.0 - p) * stay + p * p_up(g.field(i, s), beta)
        }
        // EXPLICIT for the same reason: the catch-all would run SCA as the synchronous sweep at
        // the full field, with no pinning -- a different kernel with a different law.
        Kernel::Sca { q } => {
            assert!(q.is_finite() && q >= 0.0, "SCA pinning q must be finite and non-negative, got {q}");
            p_up(0.5 * beta * g.field(i, s) + q * f64::from(s[i]), 1.0)
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
        Kernel::ChromaticGibbs | Kernel::FixedFabric | Kernel::Quantised { .. } | Kernel::SiteSpread { .. } => {
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
            for i in (0..g.n).rev() {
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
        Kernel::RandomScan => {
            // (P v)(x) = (1/n) sum_i [ p_i v(x with i up) + (1 - p_i) v(x with i down) ].
            let mut out = vec![0.0f64; m];
            for x in 0..m {
                let s = spins(x, n);
                let mut acc = 0.0;
                for i in 0..n {
                    let bit = 1usize << i;
                    let p = p_up(g.field(i, &s), beta);
                    acc += p * v[x | bit] + (1.0 - p) * v[x & !bit];
                }
                out[x] = acc / n as f64;
            }
            out
        }
        Kernel::Synchronous | Kernel::Pimi { .. } | Kernel::TickRandom { .. } | Kernel::Sca { .. } => {
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
        Kernel::ChromaticGibbs | Kernel::FixedFabric | Kernel::Quantised { .. } | Kernel::SiteSpread { .. } => {
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
        Kernel::RandomScan => {
            // (mu P)(y) = (1/n) sum_i [mu(y with i up) + mu(y with i down)] q_i(y_i | the rest):
            // mass from both values of site i lands on y with site i's own conditional probability.
            let mut out = vec![0.0f64; m];
            for y in 0..m {
                let s = spins(y, n);
                let mut acc = 0.0;
                for i in 0..n {
                    let bit = 1usize << i;
                    let p = p_up(g.field(i, &s), beta);
                    let pooled = mu[y | bit] + mu[y & !bit];
                    acc += if y & bit != 0 { p * pooled } else { (1.0 - p) * pooled };
                }
                out[y] = acc / n as f64;
            }
            out
        }
        Kernel::Synchronous | Kernel::Pimi { .. } | Kernel::TickRandom { .. } | Kernel::Sca { .. } => {
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
pub(crate) fn lu_solve(a: &mut [f64], m: usize, b: &mut [f64], r: usize) -> bool {
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
    // Elimination is accurate to an ABSOLUTE 1e-16 or so, which on a cold chain is larger than the
    // mass of many states: those come back as noise of either sign, and clamping the negatives to
    // zero left states with positive inflow at exactly zero mass -- an entropy production of
    // +inf for stale reads at beta 2 that was the clamp, not the kernel (2026-09-13).
    // One push of the clamped law through the kernel rebuilds every small entry as a sum of
    // positive terms, accurate in RELATIVE terms, and leaves the large ones where the solve put
    // them.
    let clamped: Vec<f64> = b.iter().map(|v| v.max(0.0)).collect();
    let refined = apply_distribution(g, beta, kernel, &clamped);
    let total: f64 = refined.iter().sum();
    Ok(refined.iter().map(|v| v / total).collect())
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
        Kernel::FixedFabric
        | Kernel::Quantised { .. }
        | Kernel::Pimi { .. }
        | Kernel::Stale { .. }
        | Kernel::SiteSpread { .. }
        | Kernel::TickRandom { .. } => {
            stationary_solved(g, beta, kernel)?
        }
        Kernel::Synchronous => peretto(g, beta)?,
        Kernel::Sca { q } => sca_law(g, beta, q)?,
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
        Kernel::FixedFabric
        | Kernel::Quantised { .. }
        | Kernel::Pimi { .. }
        | Kernel::Stale { .. }
        | Kernel::SiteSpread { .. }
        | Kernel::TickRandom { .. } => {
            stationary_solved(g, beta, kernel)?
        }
        Kernel::Synchronous => peretto(g, beta)?,
        Kernel::Sca { q } => sca_law(g, beta, q)?,
        _ => boltzmann(g, beta)?,
    };
    let m = 1usize << g.n;
    let mut a = vec![0.0f64; m * m];
    kernel_rows(g, beta, kernel, |x, row| {
        for (y, &p) in row.iter().enumerate() {
            let delta = if x == y { 1.0 } else { 0.0 };
            let rank_one = pi[y];
            a[x * m + y] = delta - p + rank_one;
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

/// The kernel's own stationary law: Boltzmann for the exact Gibbs and informed kernels, Peretto's
/// closed form for the synchronous sweep, [`sca_law`] for SCA, and a direct solve for everything
/// whose law has no closed form (the fabric's arithmetic, PIMI, stale reads, a site temperature
/// spread, a tick-random mask).
///
/// # Errors
///
/// [`AutocorrError::TooManySpins`] above [`MAX_SPINS`]; [`AutocorrError::TooManyForDense`] above
/// [`MAX_DENSE_SPINS`] for a kernel that needs the solve; [`AutocorrError::Reducible`] for one
/// without a unique invariant law.
pub fn own_law(g: &Graph, beta: f64, kernel: Kernel) -> Result<Vec<f64>, AutocorrError> {
    match kernel {
        Kernel::FixedFabric
        | Kernel::Quantised { .. }
        | Kernel::Pimi { .. }
        | Kernel::Stale { .. }
        | Kernel::SiteSpread { .. }
        | Kernel::TickRandom { .. } => {
            stationary_solved(g, beta, kernel)
        }
        Kernel::Synchronous => peretto(g, beta),
        Kernel::Sca { q } => sca_law(g, beta, q),
        _ => boltzmann(g, beta),
    }
}

/// The steady-state entropy production rate of the kernel with respect to its OWN stationary law,
/// in nats per step:
///
/// ```text
///   Sigma = sum_{x, y} pi(x) P(x, y) ln [ pi(x) P(x, y) / (pi(y) P(y, x)) ],
/// ```
///
/// zero exactly when the chain is reversible with respect to `pi` and `+inf` when some transition
/// has no reverse. This is the population quantity every trajectory estimator of entropy
/// production converges to, and it measures NON-REVERSIBILITY, not correctness: a fixed-order
/// sweep of exact heat-bath updates is invariant but not reversible and produces entropy while
/// sampling the right law, and the synchronous sweep is reversible with respect to Peretto's law
/// and produces none while sampling the wrong one. `examples/entropy_production_exact.rs` on the
/// 4x3 grid, nats per full update beside each law's TV from the Boltzmann distribution it was
/// meant to sample:
///
/// ```text
///                          beta 0.5            beta 1              beta 2
///   chromatic (correct)    1.11     0          0.132    0          5.4e-4   0
///   sequential (correct)   0.89     0          0.110    0          5.0e-4   0
///   1.01 beta              1.10     1.0e-2     0.125    5.4e-3     4.9e-4   2.3e-3
///   shipped fabric         1.11     2.1e-3     0.133    1.2e-3     inf      7.7e-3
///   stale reads p 0.1      0.73     3.9e-2     0.077    1.0e-2     2.7e-4   4.7e-3
///   site spread 0.2        1.33     0.13       0.151    5.7e-2     5.1e-4   7.1e-2
///   synchronous            0        0.61       0        0.43       0        0.24
///   PIMI 0.5 / 0.4         2.9e-4   0.77       4.8e-4   0.24       9.0e-4   2.9e-2
/// ```
///
/// The two columns do not rank alike at any temperature. The fabric's `inf` at `beta = 2` is its
/// comparator forbidding flips once `2 beta f > 11.8`.
///
/// # Errors
///
/// As [`own_law`], and [`AutocorrError::TooManyForDense`] above [`MAX_DENSE_SPINS`] for every
/// kernel, since the dense operator is built.
pub fn entropy_production(g: &Graph, beta: f64, kernel: Kernel) -> Result<f64, AutocorrError> {
    if g.n > MAX_DENSE_SPINS {
        return Err(AutocorrError::TooManyForDense { n: g.n, max: MAX_DENSE_SPINS });
    }
    let pi = own_law(g, beta, kernel)?;
    let m = 1usize << g.n;
    let mut p = vec![0.0f64; m * m];
    kernel_rows(g, beta, kernel, |x, row| p[x * m..(x + 1) * m].copy_from_slice(row));
    let mut sigma = 0.0;
    for x in 0..m {
        for y in 0..m {
            let fwd = pi[x] * p[x * m + y];
            if fwd <= 0.0 {
                continue;
            }
            let bwd = pi[y] * p[y * m + x];
            if bwd <= 0.0 {
                return Ok(f64::INFINITY);
            }
            sigma += fwd * (fwd / bwd).ln();
        }
    }
    Ok(sigma.max(0.0))
}

/// What a superchain design's error is made of, and the nested R-hat it would read, EXACTLY.
///
/// Margossian et al. (Bayesian Analysis 2024, their Eq. 27) split the expected squared error of
/// one superchain's mean -- `M` chains from one shared start `θ0 ~ p0`, each run `warmup` steps and
/// then read `draws` times -- into three parts,
///
/// ```text
///   E (f̄_k − E_pi f)²  =  bias²  +  Var_p0 E(f̄ | θ0)  +  E_p0 Var(f̄_k | θ0)
///                                    nonstationary         persistent
/// ```
///
/// and set nested R-hat to watch the middle one, "and so, by proxy, the squared bias". Every field
/// here is computed from the kernel, not from chains: `E(f(X_t) | x0)` is `P^t f` and
/// `E(f(X_s) f(X_t) | x0)` is `P^s (f · P^(t−s) f)`, both applied to all `2^n` starts at once.
/// [`crate::rhat::nested_rhat`] on `K` superchains tends to `rhat` as `K` grows.
#[derive(Clone, Debug, PartialEq)]
pub struct NestedPopulation {
    /// The population nested R-hat, `sqrt(1 + B / W)`.
    pub rhat: f64,
    /// `(E f̄ − E_pi f)²`, against the kernel's OWN law ([`own_law`]).
    pub squared_bias: f64,
    /// `Var_p0 E(f̄ | θ0)`: how much the answer still depends on where a superchain started.
    pub nonstationary: f64,
    /// `E_p0 Var(f̄_k | θ0)` for one superchain of `M` chains: the variance that would remain at
    /// stationarity.
    pub persistent: f64,
    /// `W`, the within-superchain variance nested R-hat scales by.
    pub within: f64,
    /// `Var_pi f` under the kernel's own law, the scale a tolerance is usually stated in.
    pub stationary_variance: f64,
    /// `E f̄ − E_Boltzmann f`, signed: the distance nested R-hat cannot see when the kernel's own
    /// law is not the Boltzmann law.
    pub boltzmann_bias: f64,
    /// Total variation between the chains' law at their first draw and the kernel's own law.
    pub tv_first_draw: f64,
}

/// The exact [`NestedPopulation`] of observable `f` for superchains started from `start` (a law
/// over the `2^n` states, bit `i` set meaning spin `i` is `+1`), `chains` chains per superchain,
/// `warmup` kernel steps before the first of `draws` draws. The last point of
/// [`nested_population_curve`].
///
/// # Errors
///
/// As [`nested_population_curve`].
///
/// # Panics
///
/// As [`nested_population_curve`].
#[allow(clippy::too_many_arguments)]
pub fn nested_population(
    g: &Graph,
    beta: f64,
    kernel: Kernel,
    f: impl Fn(&[i8]) -> f64,
    start: &[f64],
    warmup: usize,
    draws: usize,
    chains: usize,
) -> Result<NestedPopulation, AutocorrError> {
    let mut curve = nested_population_curve(g, beta, kernel, f, start, warmup, draws, chains)?;
    Ok(curve.pop().expect("the curve holds warmup + 1 points"))
}

/// [`nested_population`] at every warmup from `0` to `max_warmup`, in one pass: the kernel's own
/// law is solved once and each extra step of warmup costs a fixed number of kernel applications,
/// so a whole convergence curve costs what one late point would.
///
/// # Errors
///
/// As [`own_law`], and [`AutocorrError::TooManySpins`] above [`MAX_SPINS`].
///
/// # Panics
///
/// If `start` does not have `2^n` entries, or `draws` or `chains` is zero.
#[allow(clippy::too_many_arguments)]
pub fn nested_population_curve(
    g: &Graph,
    beta: f64,
    kernel: Kernel,
    f: impl Fn(&[i8]) -> f64,
    start: &[f64],
    max_warmup: usize,
    draws: usize,
    chains: usize,
) -> Result<Vec<NestedPopulation>, AutocorrError> {
    use std::collections::VecDeque;
    if g.n > MAX_SPINS {
        return Err(AutocorrError::TooManySpins { n: g.n, max: MAX_SPINS });
    }
    let n = g.n;
    let m = 1usize << n;
    assert_eq!(start.len(), m, "a start law over states has 2^n entries");
    assert!(draws > 0 && chains > 0, "at least one draw of one chain");
    let fv: Vec<f64> = (0..m).map(|x| f(&spins(x, n))).collect();
    let step = |v: &[f64]| apply(g, beta, kernel, v);

    let pi = own_law(g, beta, kernel)?;
    let mu_pi: f64 = pi.iter().zip(&fv).map(|(p, v)| p * v).sum();
    let var_pi: f64 = pi.iter().zip(&fv).map(|(p, v)| p * (v - mu_pi).powi(2)).sum();
    let bolt = boltzmann(g, beta)?;
    let mu_b: f64 = bolt.iter().zip(&fv).map(|(p, v)| p * v).sum();

    // Tracked functions, each advanced one kernel step per tick t: P^t f, P^t f^2, and for every
    // lag d in 1..draws, P^t (f * P^d f) -- whose sum over s gives E(f(X_s) f(X_(s+d)) | x0).
    let mut tracked: Vec<Vec<f64>> = vec![fv.clone(), fv.iter().map(|v| v * v).collect()];
    let mut lag = fv.clone();
    for _ in 1..draws {
        lag = step(&lag);
        tracked.push(fv.iter().zip(&lag).map(|(a, b)| a * b).collect());
    }
    // The last `draws` values of each, at ticks t - draws + 1 ..= t.
    let mut windows: Vec<VecDeque<Vec<f64>>> = vec![VecDeque::with_capacity(draws); tracked.len()];
    let mut law = start.to_vec();
    let nd = draws as f64;
    let expect = |v: &[f64]| -> f64 { start.iter().zip(v).map(|(p, a)| p * a).sum() };
    let mut out = Vec::with_capacity(max_warmup + 1);
    for tick in 1..=max_warmup + draws {
        for (v, win) in tracked.iter_mut().zip(windows.iter_mut()) {
            *v = step(v);
            if win.len() == draws {
                win.pop_front();
            }
            win.push_back(v.clone());
        }
        if tick < draws {
            continue;
        }
        // One report per warmup w = tick - draws, whose first draw is at time w + 1: advancing the
        // law once per REPORT, not per tick, keeps it at start P^(w + 1).
        law = apply_distribution(g, beta, kernel, &law);
        // The window now covers draws at ticks w + 1 ..= w + draws, with w = tick - draws.
        let (mut sum_f, mut sum_f2, mut cross) = (vec![0.0f64; m], vec![0.0f64; m], vec![0.0f64; m]);
        for s in &windows[0] {
            sum_f.iter_mut().zip(s).for_each(|(a, b)| *a += b);
        }
        for s in &windows[1] {
            sum_f2.iter_mut().zip(s).for_each(|(a, b)| *a += b);
        }
        // Lag d pairs a draw at s with one at s + d, so s runs over the first draws - d ticks.
        for d in 1..draws {
            for s in windows[1 + d].iter().take(draws - d) {
                cross.iter_mut().zip(s).for_each(|(a, b)| *a += b);
            }
        }
        let mean_path: Vec<f64> = sum_f.iter().map(|s| s / nd).collect();
        let second: Vec<f64> = sum_f2.iter().zip(&cross).map(|(a, c)| (a + 2.0 * c) / (nd * nd)).collect();
        let chain_var: Vec<f64> = second.iter().zip(&mean_path).map(|(s, mu)| s - mu * mu).collect();
        let within_chain = if draws > 1 {
            expect(&sum_f2.iter().zip(&second).map(|(a, s)| (a - nd * s) / (nd - 1.0)).collect::<Vec<_>>())
        } else {
            0.0
        };
        let mean_all = expect(&mean_path);
        let nonstationary = expect(&mean_path.iter().map(|v| v * v).collect::<Vec<_>>()) - mean_all * mean_all;
        let e_chain_var = expect(&chain_var);
        let persistent = e_chain_var / chains as f64;
        let within = within_chain + if chains > 1 { e_chain_var } else { 0.0 };
        out.push(NestedPopulation {
            rhat: (1.0 + (nonstationary + persistent) / within).sqrt(),
            squared_bias: (mean_all - mu_pi).powi(2),
            nonstationary,
            persistent,
            within,
            stationary_variance: var_pi,
            boltzmann_bias: mean_all - mu_b,
            tv_first_draw: total_variation(&law, &pi),
        });
    }
    Ok(out)
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
        Kernel::FixedFabric
        | Kernel::Quantised { .. }
        | Kernel::Pimi { .. }
        | Kernel::Stale { .. }
        | Kernel::SiteSpread { .. }
        | Kernel::TickRandom { .. } => {
            stationary(g, beta, kernel, 1e-14, 500_000)?.0
        }
        Kernel::Synchronous => peretto(g, beta)?,
        Kernel::Sca { q } => sca_law(g, beta, q)?,
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
            Kernel::SiteSpread { seed: 2, spread: 0.3 },
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
                Kernel::Pimi { .. } | Kernel::Stale { .. } | Kernel::SiteSpread { .. } => Some(stationary_solved(&g, beta, kernel).unwrap()),
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

    /// Zero spread is the chromatic sweep operator for operator; free sites at their own
    /// temperatures have the product of their own marginals `sigma(2 beta_i h_i)` as law, exactly;
    /// and a coupled grid with spread is resolvably not Boltzmann at the nominal temperature.
    #[test]
    fn a_site_temperature_spread_reduces_to_the_sweep_at_zero_and_to_a_product_on_free_sites() {
        let g = grid_glass(3, 3, 6);
        let beta = 1.1;
        let m = 1usize << g.n;
        let mut rng = Pcg::new(4, 4);
        let mut mu: Vec<f64> = (0..m).map(|_| rng.f64()).collect();
        let z: f64 = mu.iter().sum();
        for q in &mut mu {
            *q /= z;
        }
        let none = apply_distribution(&g, beta, Kernel::SiteSpread { seed: 7, spread: 0.0 }, &mu);
        let sweep = apply_distribution(&g, beta, Kernel::ChromaticGibbs, &mu);
        assert!(total_variation(&none, &sweep) < 1e-13, "zero spread must be the chromatic sweep");
        let spread = Kernel::SiteSpread { seed: 7, spread: 0.3 };
        let law = stationary_solved(&g, beta, spread).unwrap();
        let b = boltzmann(&g, beta).unwrap();
        assert!(total_variation(&law, &b) > 1e-3, "a spread of 0.3 must move the law off Boltzmann");
        let mut fb = GraphBuilder::new(4);
        for i in 0..4 {
            fb.bias(i, 0.2 * (i as f64 + 1.0));
        }
        let free = fb.build();
        let law = stationary_solved(&free, beta, spread).unwrap();
        for (x, &lx) in law.iter().enumerate() {
            let s = spins(x, 4);
            let want: f64 = (0..4)
                .map(|i| {
                    let p = p_up(free.h[i], beta * site_factor(7, 0.3, i));
                    if s[i] > 0 { p } else { 1.0 - p }
                })
                .product();
            assert!((lx - want).abs() < 1e-12, "state {x}: solved {lx} vs product {want}");
        }
    }

    /// Three closed forms and one inequality. The synchronous sweep is reversible with respect to
    /// Peretto's law, so its entropy production is zero while its law is wrong; a single site and
    /// two free sites under the sequential sweep are rank-one kernels, reversible, zero; and the
    /// fixed-order chromatic sweep on a coupled grid is invariant but not reversible, so it
    /// produces entropy while sampling the right law.
    #[test]
    fn entropy_production_is_zero_for_reversible_kernels_and_positive_for_a_correct_fixed_order_sweep() {
        let g = grid_glass(3, 3, 6);
        let beta = 1.1;
        let sync = entropy_production(&g, beta, Kernel::Synchronous).unwrap();
        assert!(sync.abs() < 1e-12, "synchronous (reversible w.r.t. Peretto): {sync:.3e}");
        let mut b = GraphBuilder::new(2);
        b.bias(0, 0.3);
        b.bias(1, -0.5);
        let free = b.build();
        let seq = entropy_production(&free, beta, Kernel::SequentialGibbs).unwrap();
        assert!(seq.abs() < 1e-12, "two free sites, sequential: {seq:.3e}");
        let chrom = entropy_production(&g, beta, Kernel::ChromaticGibbs).unwrap();
        assert!(chrom.is_finite() && chrom > 1e-6, "a fixed-order sweep of exact updates must produce entropy: {chrom:.3e}");
        let law = own_law(&g, beta, Kernel::ChromaticGibbs).unwrap();
        let pushed = apply_distribution(&g, beta, Kernel::ChromaticGibbs, &law);
        assert!(total_variation(&pushed, &law) < 1e-13, "and still be invariant");
    }

    /// On a cold grid the smallest stationary masses are far below the solve's absolute accuracy,
    /// and a law with a zero where inflow is positive is not a law: every entry of the solved law
    /// must be positive, and the correct sweep's entropy production finite -- the fabric's +inf at
    /// `beta = 2` was this defect for stale reads; for the fabric it is real, its comparator forbidding
    /// flips once `2 beta f > 11.8`.
    #[test]
    fn the_solved_law_has_no_zero_entries_on_a_cold_grid_and_the_sweep_produces_finite_entropy() {
        let g = grid_glass(3, 3, 5);
        let beta = 3.0;
        let law = stationary_solved(&g, beta, Kernel::ChromaticGibbs).unwrap();
        let smallest = law.iter().copied().fold(f64::INFINITY, f64::min);
        assert!(smallest > 0.0, "a state with positive inflow has positive mass, not {smallest:e}");
        // The solve's absolute accuracy is the machine epsilon times the chain's conditioning,
        // and at beta 3 this grid's relaxation time is in the millions of sweeps, so the law is
        // Boltzmann to 1e-7 or so here, not to the 1e-12 of the warm tests.
        let b = boltzmann(&g, beta).unwrap();
        let gap = total_variation(&law, &b);
        assert!(gap < 1e-6, "cold solve vs Boltzmann: {gap:e}");
        let sigma = entropy_production(&g, beta, Kernel::ChromaticGibbs).unwrap();
        assert!(sigma.is_finite(), "the exact sweep has no forbidden flip at any beta; Sigma = {sigma}");
        // Stale reads mix heat-bath probabilities, all in (0, 1), so every transition has a
        // reverse and the +inf this reported at beta 2 was the clamp. (The fabric's +inf there is
        // real: its comparator forbids a flip once 2 beta f > 11.8, which a field of 2.95 reaches
        // at beta 2.)
        let sigma = entropy_production(&g, 2.0, Kernel::Stale { p: 0.1 }).unwrap();
        assert!(sigma.is_finite(), "stale reads at beta 2 have no forbidden flip; Sigma = {sigma}");
    }

    /// The ROM entry is evaluated at the CENTRE of its cell, not its edge: a field that lands
    /// exactly on a cell boundary (here `0.5 = 32/64`, the shipped stride being `1/64`) must return
    /// the heat bath at `0.5 + 1/128` with a comparator wide enough not to matter, and not the heat
    /// bath at `0.5`. Row 118 of the mutation suite reads the edge.
    #[test]
    fn the_rom_is_read_at_the_cell_centre() {
        let mut b = GraphBuilder::new(1);
        b.bias(0, 0.5);
        let g = b.build();
        let k = Kernel::Quantised { frac_bits: 8, lut_bits: 10, prob_bits: 52 };
        let got = p_site(&g, 1.0, k, 0, &[-1]);
        let centre = p_up(0.5 + 1.0 / 128.0, 1.0);
        let edge = p_up(0.5, 1.0);
        assert!((got - centre).abs() < 1e-12, "ROM read {got} vs centre {centre}");
        assert!((got - edge).abs() > 1e-3, "ROM read {got} must not be the cell edge {edge}");
    }

    /// Build the 10-spin fixtures the tick-random test uses: a 5x2 grid (bipartite) and a 10-ring
    /// with chords (0,4) and (2,7) (odd cycles, so frustrated). Twelve spins is what
    /// `examples/tick_random_exact.rs` runs; the dense solve is `O(8^n)` in the spin count, so the
    /// example takes seven minutes and this pair takes seconds.
    fn tick_fixtures() -> (Graph, Graph, Graph) {
        use crate::graph::GraphBuilder;
        use crate::rng::Pcg;
        let (w, h) = (5usize, 2usize);
        let mut rng = Pcg::new(7, 0x6A);
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
        let grid = b.build();

        let n = 10usize;
        let mut rng = Pcg::new(3, 0x6A);
        let mut b = GraphBuilder::new(n);
        for i in 0..n {
            b.couple(i, (i + 1) % n, if rng.f64() < 0.5 { -1.0 } else { 1.0 });
        }
        for &(i, j) in &[(0usize, 4usize), (2usize, 7usize)] {
            b.couple(i, j, if rng.f64() < 0.5 { -1.0 } else { 1.0 });
        }
        for i in 0..n {
            b.bias(i, (rng.f64() - 0.5) * 0.4);
        }
        let ring = b.build();

        // The SAME fields with every coupling removed. The control for the whole test: without an
        // interaction there is no adjacent pair to move together, so p must not matter at all.
        let mut rng = Pcg::new(3, 0x6A);
        let mut b = GraphBuilder::new(n);
        for i in 0..n {
            b.bias(i, (rng.f64() - 0.5) * 0.4);
        }
        (grid, ring, b.build())
    }

    /// **Time-multiplexed reuse changes the law it is claimed not to change** — Onizawa & Hanyu,
    /// arXiv:2604.01564, for their synchronous tick-random policy.
    ///
    /// The paper argues reuse is free: *"time-multiplexed reuse corresponds to a temporal rescaling
    /// of the underlying Markov process rather than a change in its transition kernel"*, and *"only
    /// the convergence speed, not the stationary distribution, is affected"*. The Conclusion
    /// restates it. So `TV(pi_{1/c}, pi_1)` should be zero for every `c`. It is not:
    ///
    /// | fixture | beta | c = 1.25 | c = 1.5 | c = 2 | c = 3 | c = 10 |
    /// |---|---|---|---|---|---|---|
    /// | 5x2 grid | 0.5 | 0.287 | 0.376 | 0.450 | 0.503 | 0.558 |
    /// | 5x2 grid | 2 | 0.620 | 0.650 | 0.671 | 0.684 | 0.697 |
    /// | 10-ring + chords | 1 | 0.371 | 0.431 | 0.476 | 0.507 | 0.538 |
    /// | 10-ring + chords | 3 | 0.136 | 0.176 | 0.211 | 0.235 | 0.260 |
    ///
    /// At **c = 3** — which is the paper's own headline reuse factor, and the top-scoring
    /// synchronous row in both its cost tables — the two laws disagree on between **23.5% and
    /// 68.4%** of the probability mass across these eight cells. That is not a temporal rescaling.
    ///
    /// # Why the thinning argument fails here and not for their other branch
    ///
    /// Thinning is sound for a continuous-time chain where at most one site moves at a time, which
    /// is their Poisson/asynchronous policy. A Bernoulli mask is not a thinning of that chain: it
    /// leaves probability `p^2` on two ADJACENT sites moving from the same stale state, and that is
    /// the term that breaks detailed balance. The **uncoupled control is the proof of mechanism** —
    /// with the same fields and no couplings there is no adjacent pair, and the law stops moving
    /// with `p` entirely (worst TV 3.9e-15 against Boltzmann, over every `p` and `beta`).
    ///
    /// # What is deliberately NOT asserted
    ///
    /// **Monotonicity.** On these 10-spin fixtures TV from Boltzmann does fall monotonically as `p`
    /// falls, in all 8 cells. On the 12-spin frustrated ring of `examples/tick_random_exact.rs` it
    /// does **not**: at `beta = 2` it runs 0.150, 0.0015, 0.0048, 0.0047, 0.0031, 0.0009 — down,
    /// up, and down again. The direction is fixture-dependent and an assertion on it would be a
    /// claim this crate cannot support. That the law MOVES is robust; which way it moves is not.
    #[test]
    fn tick_random_moves_the_stationary_law_that_reuse_is_claimed_not_to_move() {
        let (grid, ring, uncoupled) = tick_fixtures();
        let betas = [0.5f64, 1.0, 2.0, 3.0];
        // p = 1/c for the paper's own c-set {1, 1.25, 1.5, 2, 3} plus c = 10 from its Table 5.
        let reused = [0.8f64, 2.0 / 3.0, 0.5, 1.0 / 3.0, 0.1];

        let (mut cells, mut worst_identity, mut smallest_move, mut smallest_at_c3) =
            (0usize, 0.0f64, f64::INFINITY, f64::INFINITY);
        for g in [&grid, &ring] {
            for &beta in &betas {
                let one = stationary_solved(g, beta, Kernel::TickRandom { p: 1.0 }).expect("solvable");
                // p = 1 IS the synchronous kernel -- every site, every tick, from the previous
                // state. Held against that kernel and not against `peretto`, because the closed
                // form carries its own conditioning: the same comparison against `peretto` needs a
                // 1e-5 tolerance at beta = 3 on 12 spins, where this one is exact.
                let sync = stationary_solved(g, beta, Kernel::Synchronous).expect("solvable");
                let identity = total_variation(&one, &sync);
                assert_eq!(identity, 0.0, "TickRandom p=1 must BE Synchronous, got TV {identity:e} at beta {beta}");
                worst_identity = worst_identity.max(identity);

                for (k, &p) in reused.iter().enumerate() {
                    let law = stationary_solved(g, beta, Kernel::TickRandom { p }).expect("solvable");
                    let moved = total_variation(&law, &one);
                    smallest_move = smallest_move.min(moved);
                    if (p - 1.0 / 3.0).abs() < 1e-12 {
                        smallest_at_c3 = smallest_at_c3.min(moved);
                    }
                    assert!(moved > 0.10, "reuse must move the law: beta {beta}, p index {k}, TV {moved}");
                    cells += 1;
                }
            }
        }
        assert_eq!(cells, 40, "2 fixtures x 4 betas x 5 reuse factors");
        assert_eq!(worst_identity, 0.0, "the p=1 identity is exact in every cell, not merely close");
        // Measured 0.1359 (ring, beta = 3, c = 1.25) and 0.2354 (ring, beta = 3, c = 3).
        assert!(smallest_move > 0.13, "smallest movement over all 40 cells: {smallest_move}");
        assert!(smallest_at_c3 > 0.23, "smallest movement at the paper's own c = 3: {smallest_at_c3}");

        // THE CONTROL, and the mechanism. Same fields, no couplings, so no adjacent pair can move
        // together -- and the law stops depending on p. Without this the test could not tell "the
        // mask changes the law" from "the solver is sensitive to p".
        for &beta in &betas {
            let bolt = boltzmann(&uncoupled, beta).expect("small");
            for &p in [1.0f64].iter().chain(reused.iter()) {
                let law = stationary_solved(&uncoupled, beta, Kernel::TickRandom { p }).expect("solvable");
                let tv = total_variation(&law, &bolt);
                assert!(tv < 1e-13, "uncoupled sites cannot feel the mask: beta {beta}, p {p}, TV {tv}");
            }
        }

        // p = 0 is the identity map. It has no unique invariant law and must say so rather than
        // return whatever a singular solve leaves in the buffer.
        match stationary_solved(&ring, 1.0, Kernel::TickRandom { p: 0.0 }) {
            Err(AutocorrError::Reducible) => {}
            other => panic!("p = 0 is the identity map and must be refused as reducible, got {other:?}"),
        }

        // The movement is FIRST ORDER in p with no intercept: TV -> 0 as p -> 0, but TV/p tends to
        // a finite non-zero limit. Both halves are asserted -- a test that only checked TV -> 0
        // would also pass if the law never moved at all.
        let bolt = boltzmann(&ring, 1.0).expect("small");
        let mut ratios = Vec::new();
        for &p in &[0.01f64, 0.02, 0.05, 0.1] {
            let law = stationary_solved(&ring, 1.0, Kernel::TickRandom { p }).expect("solvable");
            let tv = total_variation(&law, &bolt);
            assert!(tv < 0.02, "TV must vanish with p: p {p}, TV {tv}");
            ratios.push(tv / p);
        }
        // Measured 0.1439, 0.1447, 0.1473, 0.1520 -- a finite non-zero limit, not a collapse.
        for (r, p) in ratios.iter().zip([0.01f64, 0.02, 0.05, 0.1]) {
            assert!((0.13..0.16).contains(r), "TV/p must tend to a finite non-zero limit: p {p}, TV/p {r}");
        }
        assert!(ratios[3] > ratios[0], "and it approaches that limit from below: {ratios:?}");
    }

    /// A Sherrington-Kirkpatrick instance: every pair coupled, `J ~ N(0, 1/n)`, fields `N(0, 0.01)`.
    /// The full connectivity SCA hardware is built for -- no two sites share a colour, so the
    /// coloured sweep costs `n` ticks.
    fn sk_fixture(n: usize, seed: u64) -> Graph {
        use crate::graph::GraphBuilder;
        use crate::rng::Pcg;
        let mut rng = Pcg::new(seed, 0x6A);
        let mut normal = || {
            let u1 = rng.f64().max(1e-300);
            let u2 = rng.f64();
            (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
        };
        let mut b = GraphBuilder::new(n);
        let scale = 1.0 / (n as f64).sqrt();
        for i in 0..n {
            for j in i + 1..n {
                b.couple(i, j, normal() * scale);
            }
        }
        for i in 0..n {
            b.bias(i, normal() * 0.1);
        }
        b.build()
    }

    /// The bipartite double of `g` whose Boltzmann law at `beta`, marginalised over the second copy,
    /// is SCA's: `x_i -- y_j` at `J_ij / 2` both ways round, `x_i -- y_i` at `q / beta`, half of
    /// every field on each copy. Momentum Annealing's construction, built from the graph's own
    /// couplings and scored by [`boltzmann`], so it shares no arithmetic with [`sca_law`].
    fn bipartite_double(g: &Graph, beta: f64, q: f64) -> Graph {
        use crate::graph::GraphBuilder;
        let n = g.n;
        let mut b = GraphBuilder::new(2 * n);
        for x in 0..n {
            // CSR holds every undirected edge once from each end; each appearance lays one of the
            // two cross couplings.
            for k in g.offset[x]..g.offset[x + 1] {
                b.couple(x, n + g.nbr[k] as usize, 0.5 * g.w[k]);
            }
            if q > 0.0 {
                b.couple(x, n + x, q / beta);
            }
            b.bias(x, 0.5 * g.h[x]);
            b.bias(n + x, 0.5 * g.h[x]);
        }
        b.build()
    }

    /// `c = E_G|Phi - <Phi>| / 2`, `Phi(x) = sum_i exp(-beta f_i(x) x_i)`, from the Boltzmann law.
    fn sca_first_order_constant(g: &Graph, beta: f64) -> f64 {
        let bolt = boltzmann(g, beta).expect("small");
        let phi: Vec<f64> = (0..bolt.len())
            .map(|x| {
                let s = spins(x, g.n);
                (0..g.n).map(|i| (-beta * g.field(i, &s) * f64::from(s[i])).exp()).sum()
            })
            .collect();
        let mean: f64 = bolt.iter().zip(&phi).map(|(p, f)| p * f).sum();
        0.5 * bolt.iter().zip(&phi).map(|(p, f)| p * (f - mean).abs()).sum::<f64>()
    }

    /// **SCA's law, three ways, and the kernel keeps it.** [`sca_law`] is held to (1) the marginal
    /// of the bipartite double, enumerated over `2n` spins -- Momentum Annealing's construction,
    /// sharing nothing with the closed form but the couplings; (2) the factorised form
    /// `e^{-beta E} prod_i (1 + e^{-2q} phi_i)` from the proof of Kamakura's Theorem 3; and (3)
    /// [`peretto`] at `beta / 2` when `q = 0`. Then the KERNEL is held to it: one tick of
    /// [`Kernel::Sca`] leaves it in place, and the kernel's own solved law is it -- the second
    /// check is what a kernel with the full field instead of the half, or no pinning, cannot pass,
    /// since each has its own fixed point.
    #[test]
    fn sca_law_is_the_enumerated_marginal_of_the_bipartite_double_and_the_kernel_keeps_it() {
        use crate::graph::GraphBuilder;
        use crate::rng::Pcg;
        // 8 spins, so the double fits under MAX_SPINS: a 4x2 glass on the tick fixtures' recipe.
        let mut rng = Pcg::new(7, 0x6A);
        let mut b = GraphBuilder::new(8);
        for y in 0..2usize {
            for x in 0..4usize {
                let i = y * 4 + x;
                if x + 1 < 4 {
                    b.couple(i, i + 1, if rng.f64() < 0.5 { -1.0 } else { 1.0 });
                }
                if y + 1 < 2 {
                    b.couple(i, i + 4, if rng.f64() < 0.5 { -1.0 } else { 1.0 });
                }
            }
        }
        for i in 0..8 {
            b.bias(i, (rng.f64() - 0.5) * 0.4);
        }
        let grid8 = b.build();
        let sk6 = sk_fixture(6, 11);

        let mut worst_double = 0.0f64;
        let mut cells = 0usize;
        for g in [&grid8, &sk6] {
            for &beta in &[0.5f64, 1.0, 2.0] {
                for &q in &[0.0f64, 0.7, 2.0] {
                    let law = sca_law(g, beta, q).expect("small");
                    let joint = boltzmann(&bipartite_double(g, beta, q), beta).expect("2n <= 16");
                    let mask = (1usize << g.n) - 1;
                    let mut marginal = vec![0.0f64; 1 << g.n];
                    for (z, p) in joint.iter().enumerate() {
                        marginal[z & mask] += p;
                    }
                    let tv = total_variation(&law, &marginal);
                    assert!(tv < 1e-13, "SCA law vs the double's marginal: n {}, beta {beta}, q {q}, TV {tv:e}", g.n);
                    worst_double = worst_double.max(tv);
                    cells += 1;
                }
            }
        }
        assert_eq!(cells, 18, "2 graphs x 3 betas x 3 pinnings");

        let (grid, ring, _) = tick_fixtures();
        for g in [&grid, &ring] {
            for &beta in &[0.5f64, 1.0, 2.0] {
                let tv0 = total_variation(&sca_law(g, beta, 0.0).expect("small"), &peretto(g, 0.5 * beta).expect("small"));
                assert!(tv0 < 1e-14, "q = 0 is the synchronous sweep at beta / 2: beta {beta}, TV {tv0:e}");
                let bolt = boltzmann(g, beta).expect("small");
                for &q in &[0.5f64, 1.5, 3.0] {
                    let law = sca_law(g, beta, q).expect("small");

                    let delta = (-2.0 * q).exp();
                    let mut factored: Vec<f64> = bolt
                        .iter()
                        .enumerate()
                        .map(|(x, p)| {
                            let s = spins(x, g.n);
                            p * (0..g.n)
                                .map(|i| 1.0 + delta * (-beta * g.field(i, &s) * f64::from(s[i])).exp())
                                .product::<f64>()
                        })
                        .collect();
                    let z: f64 = factored.iter().sum();
                    factored.iter_mut().for_each(|v| *v /= z);
                    let tvf = total_variation(&law, &factored);
                    assert!(tvf < 1e-13, "factorised form: beta {beta}, q {q}, TV {tvf:e}");

                    let pushed = apply_distribution(g, beta, Kernel::Sca { q }, &law);
                    let drift = total_variation(&pushed, &law);
                    assert!(drift < 1e-14, "one SCA tick must leave its law in place: beta {beta}, q {q}, TV {drift:e}");
                    let solved = stationary_solved(g, beta, Kernel::Sca { q }).expect("solvable");
                    let tvs = total_variation(&solved, &law);
                    assert!(tvs < 1e-10, "the kernel's solved law is the closed form: beta {beta}, q {q}, TV {tvs:e}");
                }
            }
        }

        // REVERSIBLE, not merely invariant: pi(x) P(x, y) is the symmetric pair weight, so the
        // entropy production against the kernel's own law is zero -- where the tick-random mask,
        // which also updates many sites per tick, produces entropy. Scored against `own_law`, so
        // this is also what fails if that dispatch hands SCA any law but its own.
        for &q in &[0.5f64, 2.0] {
            let sigma = entropy_production(&ring, 1.0, Kernel::Sca { q }).expect("dense");
            assert!(sigma < 1e-12, "SCA is reversible w.r.t. its law: q {q}, Sigma {sigma:e}");
        }
        let masked = entropy_production(&ring, 1.0, Kernel::TickRandom { p: 0.5 }).expect("dense");
        assert!(masked > 1e-4, "the control: a tick-random mask is not reversible: Sigma {masked:e}");
    }

    /// **SCA approaches the Boltzmann law at exactly its first-order rate.** From the factorised
    /// form, `TV(SCA, Boltzmann) e^{2q}` must tend to `c = E_G|Phi - <Phi>| / 2`, a number the
    /// Boltzmann law alone determines. So the pinning a target accuracy needs is
    /// `q*(eps) = ln(c / eps) / 2` to first order, on any graph, and `examples/sca_exact.rs` finds
    /// the exact `q*` within 0.35% of it at `eps = 1e-2` in all nine of its cells. Both the limit and
    /// the approach are asserted: a law that never moved would have `TV e^{2q}` blow up, and one
    /// with the wrong half-field would converge to a different Boltzmann law and leave TV finite.
    #[test]
    fn sca_approaches_boltzmann_at_exactly_its_first_order_rate() {
        let (grid, ring, _) = tick_fixtures();
        let sk = sk_fixture(10, 11);
        let mut cells = 0usize;
        for g in [&grid, &ring, &sk] {
            for &beta in &[1.0f64, 2.0] {
                let bolt = boltzmann(g, beta).expect("small");
                let c = sca_first_order_constant(g, beta);
                assert!(c > 0.1, "a coupled graph has a non-trivial first-order constant: {c}");
                let ratio = |q: f64| total_variation(&sca_law(g, beta, q).expect("small"), &bolt) * (2.0 * q).exp() / c;
                let (r4, r6) = (ratio(4.0), ratio(6.0));
                assert!((r4 - 1.0).abs() < 1e-3, "TV e^2q / c at q = 4: beta {beta}, {r4}");
                assert!((r6 - 1.0).abs() < 2e-5, "TV e^2q / c at q = 6: beta {beta}, {r6}");
                assert!((r6 - 1.0).abs() < (r4 - 1.0).abs(), "and it converges: {r4} then {r6}");
                cells += 1;
            }
        }
        assert_eq!(cells, 6, "3 graphs x 2 betas");
    }

    /// **All spins at once buys ticks only where the law is coarse.** SCA's selling point is that
    /// every spin updates on one tick; a coloured sweep updates one colour class per tick and is
    /// exact. At matched accuracy -- SCA's pinning set to the exact `q*` at which its law is within
    /// TV `eps` of Boltzmann -- the variance cost per tick is `2 tau_int` of the energy, in ticks:
    ///
    /// | fixture (colours) | beta | coloured | SCA at TV 1e-1 | at 1e-2 | at 1e-3 |
    /// |---|---|---|---|---|---|
    /// | 10-ring + chords (3) | 1 | 10.0 | 1.54x | 18.9x | 194x |
    /// | SK, n = 10 (10) | 1 | 9.54 | 0.59x | 6.17x | 62.1x |
    ///
    /// At this temperature the sparse ring loses even at TV 1e-1, and the fully-connected graph,
    /// where the coloured sweep has to spend a tick per spin, wins there and only there. Cold, the
    /// sparse fixtures win at 1e-1 too (0.12x on this ring at `beta = 2`), because the coloured
    /// sweep is itself slow; at 1e-2 SCA loses in every one of the example's nine cells. Each factor of ten in
    /// accuracy costs ten in ticks, because `q* = ln(c / eps) / 2` and every flip is suppressed by
    /// `e^{-2q} = eps / c`. The coloured sweep pays nothing for accuracy. The same measurement over
    /// three temperatures and three fixtures is `examples/sca_exact.rs`.
    ///
    /// What this does NOT say: STATICA and Amorphica are annealers for ground states, where the
    /// stationary law is a means and a coarse one may serve. This is SCA as a SAMPLER.
    #[test]
    fn sca_beats_the_coloured_sweep_only_where_its_law_is_coarse() {
        let (_, ring, _) = tick_fixtures();
        let sk = sk_fixture(10, 11);
        let beta = 1.0;
        let q_star = |g: &Graph, eps: f64| -> f64 {
            let bolt = boltzmann(g, beta).expect("small");
            let tv = |q: f64| total_variation(&sca_law(g, beta, q).expect("small"), &bolt);
            // TV falls through eps once on [0, 10] on both fixtures (the example scans a 0.01 grid
            // and checks it stays down); bisect that crossing.
            let (mut lo, mut hi) = (0.0f64, 10.0f64);
            assert!(tv(lo) > eps && tv(hi) < eps, "eps {eps} must be bracketed");
            for _ in 0..60 {
                let mid = 0.5 * (lo + hi);
                if tv(mid) > eps {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            hi
        };
        let tau_e = |g: &Graph, kernel: Kernel| tau_int_fundamental(g, beta, kernel, |s| g.energy(s)).expect("dense").tau_int;

        let mut ratios = Vec::new();
        for g in [&ring, &sk] {
            let chi = g.classes.len() as f64;
            let coloured = tau_e(g, Kernel::ChromaticGibbs) * chi;
            let row: Vec<f64> = [1e-1f64, 1e-2]
                .iter()
                .map(|&eps| tau_e(g, Kernel::Sca { q: q_star(g, eps) }) / coloured)
                .collect();
            ratios.push((chi, row));
        }
        let (ring_chi, ring) = &ratios[0];
        let (sk_chi, sk) = &ratios[1];
        assert_eq!((*ring_chi, *sk_chi), (3.0, 10.0), "the ring colours in three, SK in ten");
        // Measured 1.54 and 18.86 (ring); 0.59 and 6.17 (SK).
        assert!(ring[0] > 1.2, "sparse: SCA loses even at TV 1e-1: {ring:?}");
        assert!(sk[0] < 0.8, "dense: SCA wins at TV 1e-1: {sk:?}");
        assert!(ring[1] > 10.0 && sk[1] > 4.0, "and both lose by 4x or more at TV 1e-2: ring {ring:?}, SK {sk:?}");
        // Each factor of ten in accuracy costs close to ten in ticks, on both.
        for (name, r) in [("ring", ring), ("SK", sk)] {
            let step = r[1] / r[0];
            assert!((8.0..14.0).contains(&step), "{name}: tenfold accuracy costs {step}x");
        }
    }

    /// **The error decomposition is an identity, checked by a route that shares no code with it.**
    /// [`nested_population`] pulls functions BACKWARD through [`apply`]; here the chains' laws are
    /// pushed FORWARD through [`apply_distribution`] and the mean squared error of a one-chain
    /// superchain is summed directly -- `E(f(X) − mu)^2` for one draw, and the two-draw version from
    /// the marginal at the first draw and `E f(X1) f(X2) = sum_x q(x) f(x) (P f)(x)`. Margossian's
    /// Eq. 27 says those equal `bias^2 + nonstationary + persistent`, and they must, to rounding.
    #[test]
    fn nested_population_decomposes_the_error_exactly_by_an_independent_route() {
        let g = grid_glass(3, 3, 11);
        let m = 1usize << g.n;
        let mut rng = Pcg::new(9, 1);
        let mut start: Vec<f64> = (0..m).map(|_| rng.f64()).collect();
        let z: f64 = start.iter().sum();
        start.iter_mut().for_each(|v| *v /= z);
        let energy = |s: &[i8]| g.energy(s);
        let fv: Vec<f64> = (0..m).map(|x| g.energy(&spins(x, g.n))).collect();
        let mut cells = 0usize;
        for kernel in [Kernel::ChromaticGibbs, Kernel::FixedFabric, Kernel::Sca { q: 1.0 }] {
            for &(beta, warmup) in &[(0.7f64, 0usize), (1.5, 3)] {
                let pi = own_law(&g, beta, kernel).expect("small");
                let mu: f64 = pi.iter().zip(&fv).map(|(p, v)| p * v).sum();
                let mut q = start.clone();
                for _ in 0..=warmup {
                    q = apply_distribution(&g, beta, kernel, &q);
                }
                // One draw.
                let one = nested_population(&g, beta, kernel, energy, &start, warmup, 1, 1).expect("small");
                let mse1: f64 = q.iter().zip(&fv).map(|(p, v)| p * (v - mu).powi(2)).sum();
                let sum1 = one.squared_bias + one.nonstationary + one.persistent;
                assert!((sum1 - mse1).abs() < 1e-10 * mse1.max(1.0), "{kernel:?} beta {beta}: {sum1} vs {mse1}");
                // Two draws.
                let two = nested_population(&g, beta, kernel, energy, &start, warmup, 2, 1).expect("small");
                let q2 = apply_distribution(&g, beta, kernel, &q);
                let pf = apply(&g, beta, kernel, &fv);
                let e1: f64 = q.iter().zip(&fv).map(|(p, v)| p * v).sum();
                let e2: f64 = q2.iter().zip(&fv).map(|(p, v)| p * v).sum();
                let s1: f64 = q.iter().zip(&fv).map(|(p, v)| p * v * v).sum();
                let s2: f64 = q2.iter().zip(&fv).map(|(p, v)| p * v * v).sum();
                let c12: f64 = (0..m).map(|x| q[x] * fv[x] * pf[x]).sum();
                let mse2 = 0.25 * (s1 + s2 + 2.0 * c12) - mu * (e1 + e2) + mu * mu;
                let sum2 = two.squared_bias + two.nonstationary + two.persistent;
                assert!((sum2 - mse2).abs() < 1e-10 * mse2.max(1.0), "{kernel:?} beta {beta}: {sum2} vs {mse2}");
                // The law at the FIRST draw is start P^(warmup + 1) however many draws follow it.
                let tv = total_variation(&q, &pi);
                for p in [&one, &two] {
                    assert!((p.tv_first_draw - tv).abs() < 1e-14, "{kernel:?} beta {beta}: {} vs {tv}", p.tv_first_draw);
                }
                cells += 1;
            }
        }
        assert_eq!(cells, 6, "3 kernels x 2 settings");
    }

    /// **The floor is exact, as the paper's Corollary 3.5 says.** On uncoupled spins one chromatic
    /// sweep is an independent draw from the stationary law, whatever the start, so nothing is
    /// nonstationary and nested R-hat reads only its persistent floor: `R^2 = 1 + 1/M` for one draw
    /// per chain -- the reason their threshold is `sqrt(1 + 1/M + tau)` -- and `1 + 1/(M (N + 1))`
    /// for `N` draws. Both are asserted exactly, on a start law that is nowhere near stationary.
    #[test]
    fn nested_population_reads_exactly_its_persistent_floor_on_independent_draws() {
        let mut b = GraphBuilder::new(6);
        for i in 0..6 {
            b.bias(i, 0.3 * i as f64 - 0.7);
        }
        let g = b.build();
        let m = 1usize << g.n;
        let mut start = vec![0.0f64; m];
        start[0] = 1.0; // every superchain from all spins down
        let mag = |s: &[i8]| s.iter().map(|&v| f64::from(v)).sum::<f64>();
        for &chains in &[1usize, 4, 16] {
            for &draws in &[1usize, 3] {
                let p = nested_population(&g, 1.3, Kernel::ChromaticGibbs, mag, &start, 0, draws, chains).expect("small");
                assert!(p.nonstationary.abs() < 1e-13, "independent draws carry no memory of the start: {}", p.nonstationary);
                // Persistent variance over W: sigma^2/(M N) over sigma^2 (N > 1) plus sigma^2/N (M > 1).
                let (mm, nn) = (chains as f64, draws as f64);
                let floor = match (chains, draws) {
                    (1, _) => 1.0 / nn,
                    (_, 1) => 1.0 / mm,
                    _ => 1.0 / (mm * (nn + 1.0)),
                };
                let got = p.rhat * p.rhat - 1.0;
                if chains == 1 && draws == 1 {
                    // W has neither a within-chain nor a between-chain part: nothing to scale by.
                    assert!(got.is_infinite() || got.is_nan(), "one draw of one chain has W = 0: {got}");
                    continue;
                }
                assert!((got - floor).abs() < 1e-12, "M {chains}, N {draws}: R^2 - 1 = {got}, floor {floor}");
            }
        }
    }

    /// **The population value is what the statistic converges to.** 4,000 superchains of four
    /// chains, each started from a uniformly drawn state shared within its superchain, run by
    /// [`crate::gibbs::Sampler`] itself -- no warmup, two draws -- and scored by
    /// [`crate::rhat::nested_rhat`]. Its `R^2 − 1` must agree with [`nested_population`]'s to within
    /// the estimator's own noise, on a glass cold enough that the answer is far above the floor.
    #[test]
    fn nested_rhat_on_sampled_superchains_converges_to_the_population_value() {
        use crate::gibbs::Sampler;
        use crate::rhat::nested_rhat;
        let g = grid_glass(3, 3, 11);
        let beta = 1.2;
        let (superchains, per, warmup, draws) = (4000usize, 4usize, 0usize, 2usize);
        let m = 1usize << g.n;
        let uniform = vec![1.0 / m as f64; m];
        let pop = nested_population(&g, beta, Kernel::ChromaticGibbs, |s| g.energy(s), &uniform, warmup, draws, per)
            .expect("small");
        let floor = 1.0 / (per as f64 * (draws as f64 + 1.0));
        assert!(pop.rhat * pop.rhat - 1.0 > 3.0 * floor, "the fixture must be far from its floor: {pop:?}");

        let mut chains = Vec::with_capacity(superchains * per);
        let mut ids = Vec::with_capacity(superchains * per);
        for k in 0..superchains {
            let x0 = Sampler::new(&g, beta, 10_000 + k as u64).s;
            for c in 0..per {
                let mut s = Sampler::new(&g, beta, 1_000_000 + (k * per + c) as u64);
                s.s.clone_from(&x0);
                s.sweeps(warmup, None);
                let mut draw = Vec::with_capacity(draws);
                for _ in 0..draws {
                    s.sweep(None);
                    draw.push(g.energy(&s.s));
                }
                chains.push(draw);
                ids.push(k);
            }
        }
        let sampled = nested_rhat(&chains, &ids);
        let (got, want) = (sampled * sampled - 1.0, pop.rhat * pop.rhat - 1.0);
        // Measured 0.3196 sampled against 0.3205 in the population, over a floor of 0.0833.
        assert!((got / want - 1.0).abs() < 0.1, "sampled R^2 - 1 = {got}, population {want}");
    }

    /// **What nested R-hat's pass does and does not certify, on a p-bit array's regime** -- one
    /// draw per chain, sixteen chains per superchain, the paper's own threshold
    /// `sqrt(1 + 1/M + tau)` at `tau = 0.01` (Margossian et al. 2024, Eq. 29). Exact throughout;
    /// `examples/nested_exact.rs` has the full table. Three things it cannot see:
    ///
    /// 1. **A start every superchain shares.** From one reset state -- a cleared register array --
    ///    the nonstationary variance is zero by construction, so `R_nu^2 = 1 + 1/M` EXACTLY at every
    ///    warmup and the rule passes at warmup 0, here with the energy's squared bias 4.5 times its
    ///    variance.
    /// 2. **A bias every start shares.** From uniform starts on a cold ferromagnet the energy passes
    ///    at warmup 101 with squared bias 3.0 tau; the tolerance is not met until 117. The nine
    ///    glass cells of the example pass with the proxy holding (at most 0.53 tau).
    /// 3. **The wrong law, reached.** A synchronous sweep converges to Peretto's law and passes at
    ///    warmup 23; its energy then sits 1.19 standard deviations from the Boltzmann value.
    #[test]
    fn nested_rhat_passes_on_a_shared_start_a_shared_bias_and_the_wrong_law() {
        const M: usize = 16;
        let threshold = (1.0 + 1.0 / M as f64 + 0.01).sqrt();
        let glass = grid_glass(5, 2, 7);
        let states = 1usize << glass.n;
        let uniform = vec![1.0 / states as f64; states];
        let mut reset = vec![0.0f64; states];
        reset[0] = 1.0;

        // 1. A shared start: exactly the floor, at every warmup.
        let curve = nested_population_curve(&glass, 2.0, Kernel::ChromaticGibbs, |s| glass.energy(s), &reset, 30, 1, M)
            .expect("small");
        for (w, p) in curve.iter().enumerate() {
            assert!(p.nonstationary.abs() < 1e-12, "w {w}: nothing can differ between identical starts");
            assert!((p.rhat * p.rhat - 1.0 - 1.0 / M as f64).abs() < 1e-12, "w {w}: R^2 - 1 = {}", p.rhat * p.rhat - 1.0);
        }
        assert!(curve[0].rhat <= threshold, "and so it passes at warmup 0");
        let bias0 = curve[0].squared_bias / curve[0].stationary_variance;
        assert!(bias0 > 3.0, "while the energy is far from its law: bias^2/Var {bias0} (measured 4.455)");

        // 2. A shared bias: the ferromagnet's energy passes before its bias is within tolerance.
        let mut b = GraphBuilder::new(10);
        for y in 0..2usize {
            for x in 0..5usize {
                let i = y * 5 + x;
                if x + 1 < 5 {
                    b.couple(i, i + 1, 1.0);
                }
                if y + 1 < 2 {
                    b.couple(i, i + 5, 1.0);
                }
            }
        }
        let ferro = b.build();
        let curve = nested_population_curve(&ferro, 2.0, Kernel::ChromaticGibbs, |s| ferro.energy(s), &uniform, 150, 1, M)
            .expect("small");
        let pass = curve.iter().position(|p| p.rhat <= threshold).expect("it passes");
        let honest = curve.iter().position(|p| p.squared_bias / p.stationary_variance <= 0.01).expect("it gets there");
        let at_pass = curve[pass].squared_bias / curve[pass].stationary_variance;
        // Measured: pass at 101 with bias^2/Var 3.04e-2; within tolerance from 117.
        assert!((95..=110).contains(&pass), "pass at {pass}");
        assert!(at_pass > 0.02, "the bias at the pass is several tau: {at_pass}");
        assert!(honest > pass + 10, "the tolerance is met well after the pass: {honest} vs {pass}");

        // 3. The wrong law, reached: synchronous passes, and the energy is off Boltzmann by a lot.
        let curve = nested_population_curve(&glass, 1.0, Kernel::Synchronous, |s| glass.energy(s), &uniform, 60, 1, M)
            .expect("small");
        let pass = curve.iter().position(|p| p.rhat <= threshold).expect("it passes");
        assert!((18..=28).contains(&pass), "synchronous passes at {pass} (measured 23)");
        assert!(curve[pass].squared_bias / curve[pass].stationary_variance < 0.01, "against its OWN law it is converged");
        let off = curve[60].boltzmann_bias.powi(2) / curve[60].stationary_variance;
        assert!(off > 1.2, "against Boltzmann the energy is > 1 sd off: bias^2/Var {off} (measured 1.411)");
    }

    /// **A fixed visiting order costs about half the site updates of a random one.** First the
    /// random-scan kernel itself: its two directions are adjoint, its solved law is the Boltzmann
    /// law, and it is REVERSIBLE (zero entropy production) where the fixed-order sweep only keeps the
    /// law invariant. Then the comparison at equal work, `tau_int` per site update, with a closed
    /// form as the control: for uncoupled spins a random scan leaves each site untouched with
    /// probability `1 − 1/n` per step, so the magnetisation's `tau` is `n − 1/2` steps, while one
    /// fixed-order sweep refreshes every site and its `tau` is `1/2` sweep, `n/2` updates -- a ratio
    /// of exactly `2 − 1/n`. Coupled, `examples/scan_order_exact.rs` measures 1.85 to 1.97 for the
    /// magnetisation at every temperature on both 5x2 fixtures, and 1.01 to 1.89 for the energy.
    #[test]
    fn a_fixed_order_needs_about_half_the_site_updates_of_a_random_scan() {
        let (grid, _, uncoupled) = tick_fixtures();
        let beta = 1.0;
        let m = 1usize << grid.n;
        let mut rng = Pcg::new(17, 2);
        let mu: Vec<f64> = {
            let raw: Vec<f64> = (0..m).map(|_| rng.f64()).collect();
            let z: f64 = raw.iter().sum();
            raw.iter().map(|v| v / z).collect()
        };
        let v: Vec<f64> = (0..m).map(|_| rng.f64() - 0.5).collect();
        let left: f64 = apply_distribution(&grid, beta, Kernel::RandomScan, &mu).iter().zip(&v).map(|(a, b)| a * b).sum();
        let right: f64 = mu.iter().zip(&apply(&grid, beta, Kernel::RandomScan, &v)).map(|(a, b)| a * b).sum();
        assert!((left - right).abs() < 1e-14, "adjoint: {left} vs {right}");
        let solved = stationary_solved(&grid, beta, Kernel::RandomScan).expect("solvable");
        assert!(total_variation(&solved, &boltzmann(&grid, beta).expect("small")) < 1e-12);
        assert!(entropy_production(&grid, beta, Kernel::RandomScan).expect("dense") < 1e-12, "random scan is reversible");
        assert!(entropy_production(&grid, beta, Kernel::SequentialGibbs).expect("dense") > 1e-3, "a fixed order is not");

        let n = uncoupled.n as f64;
        let mag = |s: &[i8]| s.iter().map(|&x| f64::from(x)).sum::<f64>();
        let per_update = |g: &Graph, k: Kernel, sweeps_are_n: bool| {
            let t = tau_int_fundamental(g, beta, k, mag).expect("dense").tau_int;
            if sweeps_are_n { t * g.n as f64 } else { t }
        };
        let ratio = per_update(&uncoupled, Kernel::RandomScan, false) / per_update(&uncoupled, Kernel::SequentialGibbs, true);
        assert!((ratio - (2.0 - 1.0 / n)).abs() < 1e-9, "uncoupled: ratio {ratio}, closed form {}", 2.0 - 1.0 / n);

        // Coupled, the ferromagnet: every site pulls on its neighbours and the factor survives.
        let mut b = GraphBuilder::new(10);
        for y in 0..2usize {
            for x in 0..5usize {
                let i = y * 5 + x;
                if x + 1 < 5 {
                    b.couple(i, i + 1, 1.0);
                }
                if y + 1 < 2 {
                    b.couple(i, i + 5, 1.0);
                }
            }
        }
        let ferro = b.build();
        let coupled = per_update(&ferro, Kernel::RandomScan, false) / per_update(&ferro, Kernel::SequentialGibbs, true);
        // Measured 1.89 at beta = 1.
        assert!((1.8..2.0).contains(&coupled), "coupled ferromagnet, magnetisation: ratio {coupled}");
    }
}
