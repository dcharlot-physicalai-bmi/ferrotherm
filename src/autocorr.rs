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
//! # Above twelve spins, and past floating point
//!
//! The dense solve stops at [`MAX_DENSE_SPINS`]. [`tau_int_krylov`] solves the same system
//! matrix-free -- conjugate gradients where the kernel is [`reversible`], GMRES otherwise, both in the
//! `pi`-weighted inner product -- in a number of applications of the kernel that grows with
//! `ln tau`, to [`MAX_KRYLOV_SPINS`]: the informed-against-Gibbs table of
//! `examples/informed_scaling_exact.rs` runs to 20 spins in under seven minutes of one core, where
//! the lag sum had not finished 14 in three hours. A chain frozen past `||A^-1|| = 1/u` has no digit
//! for any normwise-stable f64 solve; [`tau_int_censored`] eliminates it exactly onto its metastable
//! states with GTH arithmetic and returns taus of `1e13` to `1e100` sweeps on the four-city TSP
//! chains. [`tau_int_solved`] runs the first and falls to the second, and each result names its
//! [`Route`]. [`time_to_mass`] answers the other question a frozen chain poses -- how long until a
//! pushed law holds its stationary mass -- exactly by pushing, and on the slow scale past that.
//! Accuracy, measured against a double-double reference on the 4x3 grid at `beta = 2`
//! (`tau = 1.6e6`, scout of 2026-09-28): censored `3.3e-12`, GMRES `1.2e-10`, dense LU `3.2e-9` --
//! so a test that compares routes uses a tolerance scaled by the attainable accuracy
//! `u ||A^-1||`, never a fixed one.
//!
//! # The complement
//!
//! Every exact operator here states a site's two outcomes separately ([`crate::kernel::p_pair`], and
//! per kernel `site_pair`), never one as `1 -` the other. Until 2026-09-28 the down-flip was
//! `1 - p_up`, which is exactly zero once `2 beta f > 36.74`: the TSP chains at penalty `A >= 26`
//! could not leave a tour, and their exact tau was 2.2 million to 15 million times the heat bath's.
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
use crate::kernel::{p_pair, p_up};
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
    /// law, where the fixed-order sweep on a coupled graph is only invariant (uncoupled, a sweep
    /// draws a fresh sample and is reversible too).
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
    /// A Krylov solve ran out of applications before its estimate met `rtol`. `tau` is the last
    /// iterate, NOT a value.
    NotConverged {
        /// The last iterate's tau.
        tau: f64,
        /// Its estimated absolute error.
        err_est: f64,
        /// Applications of the kernel made.
        matvecs: usize,
    },
    /// The TRUE residual of a Krylov solve stopped falling before its estimate met `rtol`: f64
    /// cannot certify `rtol` for this chain, whose attainable accuracy is about `u ||A^-1||`. `tau`
    /// is the iterate there, to be read as an estimate whose error may exceed `err_est` (by 14x on
    /// the fabric at beta 2), never as a bound. [`tau_int_censored`] is the route past it.
    AtFloor {
        /// The iterate's tau.
        tau: f64,
        /// Its estimated absolute error, a lower estimate.
        err_est: f64,
        /// Applications of the kernel made.
        matvecs: usize,
    },
    /// `||A^-1||` is past `1 / u`: the chain's slowest `1 - lambda` is below f64's resolution, and no
    /// normwise-stable f64 solve of the fundamental system has a digit to give. Plain GMRES on the
    /// TSP threshold chain returned a NEGATIVE tau here (scout, 2026-09-28). [`tau_int_censored`].
    BeyondF64 {
        /// The lower estimate of `||A^-1||` that crossed `1 / u`.
        inv_norm: f64,
        /// Applications of the kernel made.
        matvecs: usize,
    },
    /// The law handed to a solve is not the kernel's: one push moved it by `residual` in L1, past
    /// [`LAW_TOLERANCE`].
    LawNotInvariant {
        /// `||pi P - pi||_1`.
        residual: f64,
    },
    /// A metastable set that is empty or names a state outside `0..2^n`.
    BadMetastableSet,
    /// An inner solve of [`tau_int_censored`] missed its tolerance within its budget, or left a
    /// censored rate negative by more than rounding. `residual` is the relative residual (or the
    /// negative rate relative to its row).
    InnerSolve {
        /// What was missed.
        residual: f64,
        /// Pushes of the kernel made.
        pushes: usize,
    },
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
            AutocorrError::NotConverged { tau, err_est, matvecs } => write!(
                f,
                "the Krylov solve used {matvecs} applications without meeting its tolerance: last iterate {tau:e}, estimated error {err_est:e}"
            ),
            AutocorrError::AtFloor { tau, err_est, matvecs } => write!(
                f,
                "the Krylov solve reached the f64 floor after {matvecs} applications: {tau:e} with estimated error {err_est:e} (a lower estimate); use tau_int_censored"
            ),
            AutocorrError::BeyondF64 { inv_norm, matvecs } => write!(
                f,
                "||A^-1|| >= {inv_norm:e} is past 1/u after {matvecs} applications: the slowest mode is below f64's resolution; use tau_int_censored"
            ),
            AutocorrError::LawNotInvariant { residual } => {
                write!(f, "the law is not the kernel's: one push moves it by {residual:e} in L1")
            }
            AutocorrError::BadMetastableSet => {
                write!(f, "the metastable set is empty or names a state outside 0..2^n")
            }
            AutocorrError::InnerSolve { residual, pushes } => {
                write!(f, "an inner solve of the censored chain missed its tolerance: {residual:e} after {pushes} pushes")
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
    /// Which route produced `tau_int`, which fixes what its error is.
    pub route: Route,
    /// Applications of the kernel the route made (`apply` or `apply_distribution`), including the
    /// ones that filled `rho` and, for the Krylov and censored solves, the one that checked the law.
    pub matvecs: usize,
    /// The route's own estimate of `|error|` in `tau_int`, absolute. `None` where the route has
    /// none (the lag sum, the dense solve, the censored solve). Not a bound: see [`tau_int_krylov`].
    pub err_est: Option<f64>,
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
    Ok(boltzmann_law(g, beta))
}

/// [`boltzmann`] without its cap, for the solves that go past [`MAX_SPINS`]: the same arithmetic.
fn boltzmann_law(g: &Graph, beta: f64) -> Vec<f64> {
    let mut energy = vec![0.0f64; 1usize << g.n];
    for_each_state(g.n, |x, s| energy[x] = g.energy(s));
    let emin = energy.iter().copied().fold(f64::INFINITY, f64::min);
    let mut pi: Vec<f64> = energy.iter().map(|e| (-beta * (e - emin)).exp()).collect();
    let z: f64 = pi.iter().sum();
    for p in &mut pi {
        *p /= z;
    }
    pi
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

/// Exact heat bath for the Gibbs kernels: `(P(s_i = +1), P(s_i = -1))` at site `i` in state `s`,
/// BOTH computed directly, never one as `1 - ` the other. `1 - p` keeps only the absolute accuracy
/// of `p`, so it is exactly zero once `p` rounds to 1 -- from `2 beta f = 36.74` for the heat bath
/// -- and a chain built from it cannot make the down-flip whose reverse it still makes: see
/// [`crate::kernel::p_down`] for what that did to `examples/penalty_mixing.rs`. Every kernel below
/// therefore states its two tails separately, each from a form with no cancellation.
///
/// For [`Kernel::FixedFabric`] it reproduces [`crate::hdl::FixedFabric`] step by step: couplings
/// and fields rounded to Q.8 (`FRAC` bits), the integer field clamped to `[-2048, 2047]`, the ROM
/// address `(field + 2048) >> 2`, the ROM entry `p_up` at the address's centre in 16 bits, and the
/// comparison against a 16-bit uniform, so the probability is `entry / 65536` exactly -- and the
/// complement `(65536 - entry) / 65536`, which `1 - p` computes EXACTLY in `f64` because both are
/// multiples of `2^-prob_bits`. That is the one arm where the subtraction is the right form: it is
/// what the comparator does. Kept beside the emulator's constants rather than importing its
/// private ones; `fabric_kernel_matches_the_emulator_in_distribution` is what keeps the two from
/// drifting apart.
fn site_pair(g: &Graph, beta: f64, kernel: Kernel, i: usize, s: &[i8]) -> (f64, f64) {
    match kernel {
        Kernel::FixedFabric => site_pair(
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
            // Both exact: `entry` and `levels - entry` are integers below 2^53, and dividing by a
            // power of two is exact. The comparator's complement is `1 - entry / levels` exactly.
            (entry / levels, (levels - entry) / levels)
        }
        Kernel::SiteSpread { seed, spread } => p_pair(g.field(i, s), beta * site_factor(seed, spread, i)),
        Kernel::Pimi { xi, eta } => {
            let z = ((beta * g.field(i, s)).tanh() + xi * f64::from(s[i])) / eta;
            normal_tails(z)
        }
        // EXPLICIT, not folded into the catch-all: the arm below is the plain heat bath, which is
        // exactly `TickRandom { p: 1.0 }`, so a missing arm here would run every p as Synchronous
        // and the whole sweep would read as "the law does not move with p" -- the paper's claim,
        // produced by our own omission.
        Kernel::TickRandom { p } => {
            assert!((0.0..=1.0).contains(&p), "TickRandom p is a probability, got {p}");
            let stay = if s[i] > 0 { 1.0 } else { 0.0 };
            let (up, down) = p_pair(g.field(i, s), beta);
            ((1.0 - p) * stay + p * up, (1.0 - p) * (1.0 - stay) + p * down)
        }
        // EXPLICIT for the same reason: the catch-all would run SCA as the synchronous sweep at
        // the full field, with no pinning -- a different kernel with a different law.
        Kernel::Sca { q } => {
            assert!(q.is_finite() && q >= 0.0, "SCA pinning q must be finite and non-negative, got {q}");
            p_pair(0.5 * beta * g.field(i, s) + q * f64::from(s[i]), 1.0)
        }
        _ => p_pair(g.field(i, s), beta),
    }
}

/// `(Phi(z), Phi(-z))`, the standard normal distribution function at `z` and its complement, each
/// accurate in relative terms: the smaller tail directly, from `erfc`, and the larger as one minus
/// it, where the subtraction loses nothing that matters because the result is near 1. Where
/// `|z| / sqrt 2 < 1` both come from [`crate::hopfield::erf`] exactly as [`Kernel::Pimi`] always
/// computed them (`0.5 (1 + erf(z / sqrt 2))`, bit for bit), since the smaller tail is then above
/// `0.079` and `1 - erf` is accurate. Above that, `erfc` by its continued fraction: `erf` itself
/// returns exactly 1 past 5, so `1 - erf` was a tail of exactly zero from `|z| = 7.07`.
fn normal_tails(z: f64) -> (f64, f64) {
    let x = z.abs() / std::f64::consts::SQRT_2;
    let (small, large) = if x < 1.0 {
        let e = crate::hopfield::erf(x);
        (0.5 * (1.0 - e), 0.5 * (1.0 + e))
    } else {
        let small = 0.5 * erfc_tail(z.abs());
        (small, 1.0 - small)
    };
    if z >= 0.0 {
        (large, small)
    } else {
        (small, large)
    }
}

/// `erfc(z / sqrt 2)` for `z >= sqrt 2`, as `e^{-z^2/2} / sqrt(pi) * K(x)` with `x = z / sqrt 2`
/// and `K(x) = 1 / (x + (1/2) / (x + 1 / (x + (3/2) / (x + ...))))` the classical continued
/// fraction, evaluated by the modified Lentz method. `z^2 / 2` is formed with its rounding error
/// carried separately (a fused multiply-add gives it exactly), because `e^{-z^2/2}` amplifies a
/// relative error in its argument by `z^2 / 2`: 700 at `z = 37`.
fn erfc_tail(z: f64) -> f64 {
    let x = z / std::f64::consts::SQRT_2;
    let zz = z * z;
    let zz_err = z.mul_add(z, -zz);
    let gauss = (-0.5 * zz).exp() * (-0.5 * zz_err).exp();
    // K(x) = 1 / (x + a_1 / (x + a_2 / (x + ...))), a_k = k / 2, by modified Lentz.
    let tiny = 1e-300;
    let mut f = x;
    let mut c = x;
    let mut d = 0.0;
    for k in 1..=500 {
        let a = 0.5 * f64::from(k);
        d = x + a * d;
        if d.abs() < tiny {
            d = tiny;
        }
        c = x + a / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = 1.0 / d;
        let delta = c * d;
        f *= delta;
        if (delta - 1.0).abs() <= f64::EPSILON {
            break;
        }
    }
    gauss / (std::f64::consts::PI.sqrt() * f)
}

/// The balancing function `g(r)` from `ln r`, with one exponential for every balance: Barker's
/// `r / (1 + r)` is the logistic of `ln r`, which the first version reached as `exp(-softplus(-ln r))`
/// -- two exponentials and a logarithm per weight, three times the transcendental work of the whole
/// informed operator at 20 spins, for the same number to rounding.
fn balance_weight(balance: Balance, log_r: f64) -> f64 {
    match balance {
        Balance::Sqrt => (0.5 * log_r).exp(),
        Balance::Metropolis => log_r.min(0.0).exp(),
        Balance::Barker => 1.0 / (1.0 + (-log_r).exp()),
    }
}

/// Every state `0..2^n` in order, with its spins held in ONE buffer that is updated incrementally:
/// from `x - 1` to `x` only the bits that carried change, two writes on average. The matrix-free
/// operators used to call [`spins`] -- an allocation of `n` bytes -- once per state per site.
fn for_each_state(n: usize, mut visit: impl FnMut(usize, &[i8])) {
    let mut s = vec![-1i8; n];
    for x in 0..1usize << n {
        if x > 0 {
            let changed = x ^ (x - 1);
            let mut b = 0;
            while b < n && (changed >> b) != 0 {
                s[b] = if (x >> b) & 1 == 1 { 1 } else { -1 };
                b += 1;
            }
        }
        visit(x, &s);
    }
}

/// `s` set to the spins of state `x`, in place.
fn set_spins(s: &mut [i8], x: usize) {
    for (i, v) in s.iter_mut().enumerate() {
        *v = if (x >> i) & 1 == 1 { 1 } else { -1 };
    }
}

/// The heat-bath probabilities `(P(+1), P(-1))` at site `i` under [`Kernel::Stale`]: the pre-sweep
/// state `x` supplies every not-yet-updated neighbour, and each already-updated neighbour `j < i`
/// is read from `y` (fresh) or from `x` (stale, probability `p`), independently, so each tail is
/// the mixture over the `2^d` stale patterns of the `d` already-updated neighbours of that tail.
fn stale_pair(g: &Graph, beta: f64, p: f64, i: usize, x: &[i8], y: &[i8]) -> (f64, f64) {
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
    let (mut q, mut q_down) = (0.0, 0.0);
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
            let (up, down) = p_pair(field, beta);
            q += weight * up;
            q_down += weight * down;
        }
    }
    (q, q_down)
}

/// Apply the kernel once: `v <- P v`, matrix-free.
///
/// For the chromatic sweep `P = P_{c_last} ... P_{c_1}`, so the classes are applied to `v` in
/// REVERSE order (operators compose right to left). Within a class the sites are pairwise
/// non-adjacent, so no site's conditional reads another class site: the class update is a product
/// of commuting single-site heat-bath updates from the fields at the pre-class state, and it is
/// contracted one site at a time -- `2^n` work per site, where expanding each state over the
/// class's `2^|c|` joint outcomes cost `2^n 2^|c|` (17 s against 0.4 s per application at 20
/// spins). The two are the same operator; `the_matrix_free_sweep_matches_a_dense_operator_built_class_by_class`
/// holds this one to the class-by-class expansion.
///
/// For the informed kernel `(P v)(x) = sum_k P(x -> y_k) v(y_k) + P(x -> x) v(x)` with
/// `P(x -> y_k) = w_k(x) / Z(x) * min(1, Z(x) / Z(y_k))`, exactly the acceptance
/// [`crate::informed`] derives; the shift that module carries cancels in every ratio and is
/// omitted.
///
/// Every site's two outcomes are weighted by the two tails [`site_pair`] states separately, never by
/// `p` and `1 - p`.
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
            let mut next = vec![0.0f64; m];
            for class in g.classes.iter().rev() {
                for &i in class {
                    let i = i as usize;
                    let bit = 1usize << i;
                    for_each_state(n, |x, s| {
                        let (p, q) = site_pair(g, beta, kernel, i, s);
                        next[x] = p * cur[x | bit] + q * cur[x & !bit];
                    });
                    std::mem::swap(&mut cur, &mut next);
                }
            }
            cur
        }
        Kernel::SequentialGibbs => {
            // P = P_0 P_1 ... P_{n-1} on distributions, so on functions the LAST site's update is
            // applied first: (P v) = P_0 (P_1 (... (P_{n-1} v))). Each P_i is rank two per state --
            // the heat-bath conditional at site i given the current others.
            let mut cur = v.to_vec();
            let mut next = vec![0.0f64; m];
            for i in (0..g.n).rev() {
                let bit = 1usize << i;
                for_each_state(n, |x, s| {
                    let (p, q) = p_pair(g.field(i, s), beta);
                    next[x] = p * cur[x | bit] + q * cur[x & !bit];
                });
                std::mem::swap(&mut cur, &mut next);
            }
            cur
        }
        Kernel::RandomScan => {
            // (P v)(x) = (1/n) sum_i [ p_i v(x with i up) + q_i v(x with i down) ].
            let mut out = vec![0.0f64; m];
            for_each_state(n, |x, s| {
                let mut acc = 0.0;
                for i in 0..n {
                    let bit = 1usize << i;
                    let (p, q) = p_pair(g.field(i, s), beta);
                    acc += p * v[x | bit] + q * v[x & !bit];
                }
                out[x] = acc / n as f64;
            });
            out
        }
        Kernel::Synchronous | Kernel::Pimi { .. } | Kernel::TickRandom { .. } | Kernel::Sca { .. } => {
            // (P v)(x) = E[v(y)] under the product law y_i ~ Bernoulli(q_i(x)), every q_i from the
            // PREVIOUS state x. Per x, contract v one site at a time under that product, from the
            // top bit down, so the block that remains keeps its bit layout.
            let mut out = vec![0.0f64; m];
            let mut w = vec![0.0f64; m];
            for_each_state(n, |x, s| {
                w.copy_from_slice(v);
                let mut len = m;
                for i in (0..n).rev() {
                    let (q, q_down) = site_pair(g, beta, kernel, i, s);
                    let half = len / 2;
                    for y in 0..half {
                        w[y] = q_down * w[y] + q * w[y | half];
                    }
                    len = half;
                }
                out[x] = w[0];
            });
            out
        }
        Kernel::Stale { p } => {
            // (P v)(x) = sum_y prod_i q_i(x, y_{<i}) v(y): per source x, fold v from the top bit
            // down; when site i is folded, the bits below it are still indices, so q_i may depend
            // on them -- which is exactly the already-updated neighbours it reads.
            let mut out = vec![0.0f64; m];
            let mut w = vec![0.0f64; m];
            let mut sy = vec![-1i8; n];
            for_each_state(n, |x, sx| {
                w.copy_from_slice(v);
                let mut len = m;
                for i in (0..n).rev() {
                    let half = len / 2;
                    for y in 0..half {
                        set_spins(&mut sy, y);
                        let (q, q_down) = stale_pair(g, beta, p, i, sx, &sy);
                        w[y] = q_down * w[y] + q * w[y | half];
                    }
                    len = half;
                }
                out[x] = w[0];
            });
            out
        }
        Kernel::Informed(balance) => {
            // The weight of flipping k at a state with spin s_k and field f_k there. The field at k
            // does not read s_k, so the state across flip k has the same f_k and spin -s_k.
            let weight = |s_k: f64, f_k: f64| balance_weight(balance, -2.0 * beta * s_k * f_k);
            // Z(x) for every state first, since the acceptance needs Z at the neighbour too.
            let mut z = vec![0.0f64; m];
            for_each_state(n, |x, s| z[x] = (0..n).map(|k| weight(f64::from(s[k]), g.field(k, s))).sum());
            let mut next = vec![0.0f64; m];
            for_each_state(n, |x, s| {
                let mut stay = 1.0;
                let mut acc = 0.0;
                if z[x] > 0.0 {
                    for k in 0..n {
                        let y = x ^ (1usize << k);
                        let alpha = if z[y] > 0.0 { (z[x] / z[y]).min(1.0) } else { 1.0 };
                        let p = weight(f64::from(s[k]), g.field(k, s)) / z[x] * alpha;
                        stay -= p;
                        acc += p * v[y];
                    }
                }
                next[x] = acc + stay.max(0.0) * v[x];
            });
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
            let mut next = vec![0.0f64; m];
            for i in 0..n {
                let bit = 1usize << i;
                for_each_state(n, |x, s| {
                    let (p, q) = p_pair(g.field(i, s), beta);
                    // Mass from both values of site i lands on x with x_i's own probability.
                    let pooled = cur[x | bit] + cur[x & !bit];
                    next[x] = if x & bit != 0 { p * pooled } else { q * pooled };
                });
                std::mem::swap(&mut cur, &mut next);
            }
            cur
        }
        Kernel::ChromaticGibbs | Kernel::FixedFabric | Kernel::Quantised { .. } | Kernel::SiteSpread { .. } => {
            // One site at a time within a class, as in `apply`: the class sites' conditionals
            // depend only on the OTHER sites, so their updates commute.
            let mut cur = mu.to_vec();
            let mut next = vec![0.0f64; m];
            for class in &g.classes {
                for &i in class {
                    let i = i as usize;
                    let bit = 1usize << i;
                    for_each_state(n, |x, s| {
                        let (p, q) = site_pair(g, beta, kernel, i, s);
                        let pooled = cur[x | bit] + cur[x & !bit];
                        next[x] = if x & bit != 0 { p * pooled } else { q * pooled };
                    });
                    std::mem::swap(&mut cur, &mut next);
                }
            }
            cur
        }
        Kernel::RandomScan => {
            // (mu P)(y) = (1/n) sum_i [mu(y with i up) + mu(y with i down)] q_i(y_i | the rest):
            // mass from both values of site i lands on y with site i's own conditional probability.
            let mut out = vec![0.0f64; m];
            for_each_state(n, |y, s| {
                let mut acc = 0.0;
                for i in 0..n {
                    let bit = 1usize << i;
                    let (p, p_down) = p_pair(g.field(i, s), beta);
                    let pooled = mu[y | bit] + mu[y & !bit];
                    acc += if y & bit != 0 { p * pooled } else { p_down * pooled };
                }
                out[y] = acc / n as f64;
            });
            out
        }
        Kernel::Synchronous | Kernel::Pimi { .. } | Kernel::TickRandom { .. } | Kernel::Sca { .. } => {
            // (mu P)(y) = sum_x mu(x) prod_i q_i^x(y_i): each source state lays its product law
            // over every target, built by doubling one site at a time (bit i set gets q_i).
            let mut out = vec![0.0f64; m];
            let mut prod = vec![0.0f64; m];
            for_each_state(n, |x, s| {
                let mass = mu[x];
                if mass == 0.0 {
                    return;
                }
                prod[0] = 1.0;
                let mut len = 1usize;
                for i in 0..n {
                    let (q_prev, q_down) = site_pair(g, beta, kernel, i, s);
                    for y in 0..len {
                        let w = prod[y];
                        prod[y] = w * q_down;
                        prod[y | len] = w * q_prev;
                    }
                    len <<= 1;
                }
                for (o, pr) in out.iter_mut().zip(&prod) {
                    *o += mass * pr;
                }
            });
            out
        }
        Kernel::Stale { p } => {
            // (mu P)(y) = sum_x mu(x) prod_i q_i(x, y_{<i}): each source lays its law over the
            // targets one site at a time, the conditional at site i reading the target bits
            // already placed below it.
            let mut out = vec![0.0f64; m];
            let mut prod = vec![0.0f64; m];
            let mut sy = vec![-1i8; n];
            for_each_state(n, |x, sx| {
                let mass = mu[x];
                if mass == 0.0 {
                    return;
                }
                prod[0] = 1.0;
                let mut len = 1usize;
                for i in 0..n {
                    for y in 0..len {
                        set_spins(&mut sy, y);
                        let (q, q_down) = stale_pair(g, beta, p, i, sx, &sy);
                        let wgt = prod[y];
                        prod[y] = wgt * q_down;
                        prod[y | len] = wgt * q;
                    }
                    len <<= 1;
                }
                for (o, pr) in out.iter_mut().zip(&prod) {
                    *o += mass * pr;
                }
            });
            out
        }
        Kernel::Informed(balance) => {
            // As in `apply`: the weight of flipping k at x, and at the state across that flip (same
            // field at k, opposite spin), both from x's own spins.
            let weight = |s_k: f64, f_k: f64| balance_weight(balance, -2.0 * beta * s_k * f_k);
            let mut z = vec![0.0f64; m];
            for_each_state(n, |x, s| z[x] = (0..n).map(|k| weight(f64::from(s[k]), g.field(k, s))).sum());
            let mut next = vec![0.0f64; m];
            for_each_state(n, |x, s| {
                // Mass arriving from each neighbour y_k, plus what stays.
                let mut stay = 1.0;
                let mut acc = 0.0;
                for k in 0..n {
                    let y = x ^ (1usize << k);
                    let (s_k, f_k) = (f64::from(s[k]), g.field(k, s));
                    if z[x] > 0.0 {
                        let alpha = if z[y] > 0.0 { (z[x] / z[y]).min(1.0) } else { 1.0 };
                        stay -= weight(s_k, f_k) / z[x] * alpha;
                    }
                    if z[y] > 0.0 {
                        let alpha = if z[x] > 0.0 { (z[y] / z[x]).min(1.0) } else { 1.0 };
                        acc += mu[y] * weight(-s_k, f_k) / z[y] * alpha;
                    }
                }
                next[x] = acc + stay.max(0.0) * mu[x];
            });
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
    Ok(Autocorrelation { tau_int: tau, lags: 0, rho, variance: c0, route: Route::Dense, matvecs: m + 64, err_est: None })
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

/// Consecutive lags with `|rho(k)| < tol` that [`tau_int_exact`] needs before it stops. ONE was the
/// rule until 2026-09-28, and an autocorrelation that crosses zero stops that rule at the crossing:
/// the synchronous sweep of two coupled spins at zero field has `rho(k) = 0` at every odd lag and
/// `tanh(beta J)^k` at every even one, so the first quiet lag is lag 1 and the sum returned `1/2`
/// for a chain whose `tau` is `1.88` at `beta J = 1`
/// (`the_lag_sum_does_not_stop_where_the_autocorrelation_crosses_zero`). A non-reversible sweep's
/// complex eigenvalues can put a crossing at any lag.
pub const QUIET_LAGS: usize = 16;

/// The exact integrated autocorrelation time of `observable` under `kernel` at `beta`.
///
/// `tol` is the tail cut: lags are summed until [`QUIET_LAGS`] CONSECUTIVE lags have
/// `|rho(k)| < tol`, or until `max_lags`. Both are reported back so a reader can see whether the
/// sum converged or was stopped. It needs about `tau ln(1/tol)` applications of the kernel, so at
/// low temperature it is a mixing-time computation; [`tau_int_krylov`] needs a number that grows
/// with `ln tau` instead, and [`tau_int_censored`] reaches chains past floating point.
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
    let mut quiet = 0;
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
            quiet += 1;
            if quiet >= QUIET_LAGS {
                break;
            }
        } else {
            quiet = 0;
        }
    }
    Ok(Autocorrelation { tau_int: tau, lags, rho, variance: c0, route: Route::LagSum, matvecs: lags, err_est: None })
}

// ------------------------------------------------------------------------------------------------
// Exact tau above the dense cap, and past floating point.
//
// `tau_int_fundamental` solves `(I - P + 1 pi^T) z = e` densely and stops at 12 spins. The same
// system solves matrix-free: `A v = v - P v + <pi, v>` is one application of the kernel, and in the
// `pi`-weighted inner product any `pi`-invariant `P` is a contraction, so `A` has its field of
// values in the right half-plane -- and a REVERSIBLE `P` makes it self-adjoint positive definite.
// So: conjugate gradients where the kernel is reversible, restarted GMRES otherwise, both in
// `L^2(pi)`, `tau = <e, z>_pi / C0 - 1/2`. The applications needed grow with `ln tau`, not `tau`:
// 43 at `tau = 1.6e6` on the 4x3 grid at beta 2 (scout, 2026-09-28).
//
// A chain whose slowest `1 - lambda` is below f64's resolution has no digit to give any
// normwise-stable f64 solve of that system. `tau_int_censored` reaches it by an exact block
// elimination onto a small metastable set, with the elimination done so that no probability is ever
// formed as `1 -` another.
// ------------------------------------------------------------------------------------------------

/// The most spins [`tau_int_krylov`] and [`tau_int_censored`] build vectors over: a vector of `2^22`
/// `f64` is 32 MiB and GMRES keeps [`GMRES_RESTART`]` + 1` of them. [`max_krylov_spins`] is tighter
/// for kernels whose one application costs more than `n 2^n`.
pub const MAX_KRYLOV_SPINS: usize = 22;

/// The Krylov basis [`tau_int_krylov`]'s GMRES keeps before it restarts. Measured by the scout of
/// 2026-09-28 on the 4x3 grid at beta 2 and a 14-spin frustrated ring: GMRES(5) stagnated,
/// GMRES(10) needed 300 to 1,400 applications, 20 sufficed for the sweeps and 100 for the informed
/// chain (which takes CG here anyway).
pub const GMRES_RESTART: usize = 50;

/// The largest `||pi P - pi||_1` the Krylov and censored solves accept for the law they are handed.
/// The kernel's own law passes at `1e-14` on every fixture measured; the fabric's kernel handed the
/// Boltzmann law is `8.6e-4` from invariant on the 4x3 grid at beta 1.
pub const LAW_TOLERANCE: f64 = 1e-10;

/// How a tau was computed. The route fixes what the number's error is and whether there is an
/// estimate of it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Route {
    /// The lag-by-lag sum of [`tau_int_exact`], cut at a tail tolerance: truncated, by the tail
    /// below the cut.
    LagSum,
    /// The dense fundamental matrix of [`tau_int_fundamental`]: one LU with partial pivoting, whose
    /// error is about `u ||A^-1||` -- measured against a double-double reference it is the LEAST
    /// accurate route on a cold heat bath (`3.2e-9` at `tau = 1.6e6`, where GMRES was `1.2e-10` and
    /// the censored solve `3.3e-12`).
    Dense,
    /// Conjugate gradients in the `pi` inner product ([`tau_int_krylov`] on a reversible kernel).
    Cg,
    /// Restarted GMRES in the `pi` inner product ([`tau_int_krylov`] on any other kernel).
    Gmres,
    /// The censored (Schur-complement) solve of [`tau_int_censored`], eliminated GTH-style.
    Censored,
}

/// Which states [`tau_int_censored`] keeps: the metastable set `F` the chain is censored onto.
/// The answer does not depend on it -- the elimination is exact for any non-empty `F` -- but the
/// cost and the conditioning of the inner solves do: they are well conditioned when `F` holds
/// every trap, so that the chain leaves the rest of the space quickly.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Metastable<'a> {
    /// Every state no single flip lowers the energy of (`s_i f_i(x) >= 0` for every `i`): 36 states
    /// of the 65,536 on the four-city TSP chains.
    LocalMinima,
    /// [`Metastable::LocalMinima`] and every state one flip from one of them (612 on the TSP
    /// chains): a second set, with entirely different inner systems, for the check that two sets
    /// give the same tau.
    LocalMinimaAndNeighbours,
    /// Exactly these states (bit `i` set is spin `i` at `+1`).
    States(&'a [usize]),
}

/// The most spins the Krylov and censored solves take for `kernel`: [`MAX_KRYLOV_SPINS`] where one
/// application is `O(n 2^n)`, 14 for the synchronous family, whose application builds a product law
/// per source state (`O(4^n)`), and [`MAX_DENSE_SPINS`] for stale reads (`O(4^n 2^d)`). A kernel
/// whose law is only available by the dense solve stops at [`MAX_DENSE_SPINS`] whatever this says,
/// because its law does.
#[must_use]
pub fn max_krylov_spins(kernel: Kernel) -> usize {
    match kernel {
        Kernel::Synchronous | Kernel::Pimi { .. } | Kernel::TickRandom { .. } | Kernel::Sca { .. } => 14,
        Kernel::Stale { .. } => MAX_DENSE_SPINS,
        Kernel::ChromaticGibbs
        | Kernel::SequentialGibbs
        | Kernel::RandomScan
        | Kernel::Informed(_)
        | Kernel::FixedFabric
        | Kernel::Quantised { .. }
        | Kernel::SiteSpread { .. } => MAX_KRYLOV_SPINS,
    }
}

/// Whether `kernel` is REVERSIBLE with respect to its own law on `g`, which makes `I - P + 1 pi^T`
/// self-adjoint and positive definite in the `pi` inner product, so conjugate gradients apply. The
/// random scan and the informed chain are reversible with respect to Boltzmann, the synchronous
/// sweep with respect to Peretto's law, SCA with respect to [`sca_law`]. A fixed-order or chromatic
/// sweep is only INVARIANT unless the graph has no edges, when every sweep draws a fresh sample. CG
/// on a non-reversible sweep does not fail loudly: on the 4x3 grid it ran out of budget holding the
/// chromatic sweep's tau `9.7e-6` off and the sequential sweep's `66%` off (scout, 2026-09-28).
#[must_use]
pub fn reversible(kernel: Kernel, g: &Graph) -> bool {
    match kernel {
        Kernel::RandomScan | Kernel::Informed(_) | Kernel::Synchronous | Kernel::Sca { .. } => true,
        Kernel::ChromaticGibbs => g.n_edges == 0,
        Kernel::SequentialGibbs => g.n_edges == 0,
        Kernel::Pimi { .. }
        | Kernel::Stale { .. }
        | Kernel::TickRandom { .. }
        | Kernel::SiteSpread { .. }
        | Kernel::FixedFabric
        | Kernel::Quantised { .. } => false,
    }
}

/// The kernel's own invariant law for the Krylov and censored solves: a closed form where there is
/// one, at any size up to the cap; the direct solve otherwise, which stops at [`MAX_DENSE_SPINS`]
/// and REFUSES above it. Never the Boltzmann law by default: every arm is written out, so a kernel
/// added to [`Kernel`] does not compile here until someone states its law, and the one-push check
/// in [`checked_law`] catches an arm that states it wrong. The verifier of 2026-09-28 met exactly
/// that: GMRES handed the Boltzmann law converged on the fabric and on tick-random at `p = 0.5`, to
/// a tau `6.2e-4` and `13.7%` off.
fn solve_law(g: &Graph, beta: f64, kernel: Kernel) -> Result<Vec<f64>, AutocorrError> {
    match kernel {
        Kernel::ChromaticGibbs | Kernel::SequentialGibbs | Kernel::RandomScan | Kernel::Informed(_) => {
            Ok(boltzmann_law(g, beta))
        }
        Kernel::Synchronous | Kernel::Sca { .. } => own_law(g, beta, kernel),
        Kernel::FixedFabric
        | Kernel::Quantised { .. }
        | Kernel::Pimi { .. }
        | Kernel::Stale { .. }
        | Kernel::SiteSpread { .. }
        | Kernel::TickRandom { .. } => stationary_solved(g, beta, kernel),
    }
}

/// [`solve_law`], held to invariance by one push ([`check_law`]).
fn checked_law(g: &Graph, beta: f64, kernel: Kernel) -> Result<Vec<f64>, AutocorrError> {
    check_law(g, beta, kernel, solve_law(g, beta, kernel)?)
}

/// `pi` if one push moves it by at most [`LAW_TOLERANCE`] in L1, [`AutocorrError::LawNotInvariant`]
/// otherwise.
fn check_law(g: &Graph, beta: f64, kernel: Kernel, pi: Vec<f64>) -> Result<Vec<f64>, AutocorrError> {
    let pushed = apply_distribution(g, beta, kernel, &pi);
    let residual: f64 = pushed.iter().zip(&pi).map(|(a, b)| (a - b).abs()).sum();
    if residual <= LAW_TOLERANCE {
        Ok(pi)
    } else {
        Err(AutocorrError::LawNotInvariant { residual })
    }
}

/// `<u, v>_pi`, summed in four lanes to cut the rounding of a `2^n`-term sum.
fn dot_pi(pi: &[f64], u: &[f64], v: &[f64]) -> f64 {
    let mut acc = [0.0f64; 4];
    for (k, ((p, a), b)) in pi.iter().zip(u).zip(v).enumerate() {
        acc[k & 3] += p * a * b;
    }
    (acc[0] + acc[1]) + (acc[2] + acc[3])
}

/// `||u||_pi`.
fn norm_pi(pi: &[f64], u: &[f64]) -> f64 {
    dot_pi(pi, u, u).max(0.0).sqrt()
}

/// `<pi, v>`, the mean of `v` under `pi`, in four lanes.
fn mean_pi(pi: &[f64], v: &[f64]) -> f64 {
    let mut acc = [0.0f64; 4];
    for (k, (p, a)) in pi.iter().zip(v).enumerate() {
        acc[k & 3] += p * a;
    }
    (acc[0] + acc[1]) + (acc[2] + acc[3])
}

fn axpy(y: &mut [f64], a: f64, x: &[f64]) {
    for (u, v) in y.iter_mut().zip(x) {
        *u += a * v;
    }
}

/// The centred observable `e = f - <f>_pi` and `C0 = <e, e>_pi`, the mean removed twice: the first
/// pass leaves the rounding of a `2^n`-term mean in `e`, and a residual mean of `1e-6` in `e` sent
/// GMRES without the rank-one term to `1.3e21` (scout, `failures.txt` (c)).
fn centred(n: usize, pi: &[f64], observable: impl Fn(&[i8]) -> f64) -> Result<(Vec<f64>, f64), AutocorrError> {
    let mut e = vec![0.0f64; 1usize << n];
    for_each_state(n, |x, s| e[x] = observable(s));
    let mean = mean_pi(pi, &e);
    e.iter_mut().for_each(|v| *v -= mean);
    let again = mean_pi(pi, &e);
    e.iter_mut().for_each(|v| *v -= again);
    let c0 = dot_pi(pi, &e, &e);
    if c0 > 0.0 {
        Ok((e, c0))
    } else {
        Err(AutocorrError::NoVariance)
    }
}

/// Whether an estimate of `||A^-1||` is past `1 / u`, where no normwise-stable f64 solve of the
/// fundamental system has a digit to give ([`AutocorrError::BeyondF64`]).
fn past_f64(inv_norm: f64) -> bool {
    inv_norm * f64::EPSILON > 1.0
}

/// Smallest singular value of the `k x k` upper-triangular `R` (row-major, stride `ld`), by inverse
/// iteration on `R^T R`. Zero when a diagonal entry is: `R` is then singular.
fn sigma_min_upper(r: &[f64], ld: usize, k: usize) -> f64 {
    if k == 0 {
        return f64::INFINITY;
    }
    if (0..k).any(|i| r[i * ld + i] == 0.0) {
        return 0.0;
    }
    let mut x = vec![1.0 / (k as f64).sqrt(); k];
    let mut sigma = 0.0;
    for _ in 0..60 {
        // R^T y = x (forward), then R w = y (back).
        let mut y = x.clone();
        for i in 0..k {
            let mut s = y[i];
            for j in 0..i {
                s -= r[j * ld + i] * y[j];
            }
            y[i] = s / r[i * ld + i];
        }
        let mut w = y;
        for i in (0..k).rev() {
            let mut s = w[i];
            for j in i + 1..k {
                s -= r[i * ld + j] * w[j];
            }
            w[i] = s / r[i * ld + i];
        }
        let nw = w.iter().map(|v| v * v).sum::<f64>().sqrt();
        if !(nw > 0.0 && nw.is_finite()) {
            return 0.0;
        }
        let next = 1.0 / nw.sqrt();
        for (a, b) in x.iter_mut().zip(&w) {
            *a = b / nw;
        }
        let settled = (next - sigma).abs() <= 1e-10 * next;
        sigma = next;
        if settled {
            break;
        }
    }
    sigma
}

/// Smallest eigenvalue of the Lanczos tridiagonal that CG's coefficients assemble (diagonal
/// `1/alpha_j + beta_{j-1}/alpha_{j-1}`, off-diagonal `sqrt(beta_j)/alpha_j`), by bisection on the
/// Sturm count. It is a Ritz value, so it lies ABOVE the operator's smallest eigenvalue and its
/// reciprocal is a lower estimate of `||A^-1||`.
fn lanczos_min(alphas: &[f64], betas: &[f64]) -> f64 {
    let k = alphas.len();
    let mut d = vec![0.0f64; k];
    let mut o = vec![0.0f64; k.saturating_sub(1)];
    for j in 0..k {
        d[j] = 1.0 / alphas[j] + if j > 0 { betas[j - 1] / alphas[j - 1] } else { 0.0 };
        if j + 1 < k {
            o[j] = betas[j].sqrt() / alphas[j];
        }
    }
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    for j in 0..k {
        let rad = if j > 0 { o[j - 1].abs() } else { 0.0 } + if j + 1 < k { o[j].abs() } else { 0.0 };
        lo = lo.min(d[j] - rad);
        hi = hi.max(d[j] + rad);
    }
    lo = lo.min(0.0);
    let below = |x: f64| -> usize {
        let mut c = 0usize;
        let mut q = d[0] - x;
        if q < 0.0 {
            c += 1;
        }
        for j in 1..k {
            let qq = if q == 0.0 { 1e-300 } else { q };
            q = d[j] - x - o[j - 1] * o[j - 1] / qq;
            if q < 0.0 {
                c += 1;
            }
        }
        c
    };
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if below(mid) >= 1 {
            hi = mid;
        } else {
            lo = mid;
        }
        if hi - lo <= 1e-15 * hi.abs().max(1e-300) {
            break;
        }
    }
    hi.max(1e-300)
}

/// `y` with `R y = g` for the leading `k x k` of the rotated Hessenberg.
fn back_substitute(h: &[f64], ld: usize, g: &[f64], k: usize) -> Vec<f64> {
    let mut y = g[..k].to_vec();
    for i in (0..k).rev() {
        let mut s = y[i];
        for l in i + 1..k {
            s -= h[i * ld + l] * y[l];
        }
        y[i] = s / h[i * ld + i];
    }
    y
}

/// What a Krylov solve returns: tau, its absolute error estimate.
type Solved = Result<(f64, f64), AutocorrError>;

/// Conjugate gradients on `A z = e` in `L^2(pi)`, for a reversible kernel. The error of the tau of
/// an iterate is `<r, A^-1 r> / C0`, QUADRATIC in the residual, estimated as
/// `max(||r||^2 / C0, u ||z|| / ||e||) / theta_min` with `theta_min` the smallest Ritz value so far
/// -- the second term is the rounding floor, below which a computed residual is noise. Accepted
/// only on a TRUE residual, recomputed with one extra application.
fn solve_cg(pi: &[f64], e: &[f64], c0: f64, a_op: &dyn Fn(&[f64]) -> Vec<f64>, count: &std::cell::Cell<usize>, rtol: f64, max_matvecs: usize) -> Solved {
    let enorm = c0.sqrt();
    let mut z = vec![0.0f64; e.len()];
    let mut r = e.to_vec();
    let mut inv_norm = 1.0f64;
    let mut prev_true: Option<f64> = None;
    loop {
        let mut p = r.clone();
        let mut rr = dot_pi(pi, &r, &r);
        let (mut alphas, mut betas) = (Vec::new(), Vec::new());
        while rr > 0.0 && count.get() < max_matvecs {
            let ap = a_op(&p);
            let pap = dot_pi(pi, &p, &ap);
            if !(pap > 0.0) {
                break;
            }
            let alpha = rr / pap;
            axpy(&mut z, alpha, &p);
            axpy(&mut r, -alpha, &ap);
            let rr_next = dot_pi(pi, &r, &r);
            let beta_k = rr_next / rr;
            alphas.push(alpha);
            betas.push(beta_k);
            for (pv, rv) in p.iter_mut().zip(&r) {
                *pv = rv + beta_k * *pv;
            }
            rr = rr_next;
            inv_norm = inv_norm.max(1.0 / lanczos_min(&alphas, &betas));
            if past_f64(inv_norm) {
                return Err(AutocorrError::BeyondF64 { inv_norm, matvecs: count.get() });
            }
            let tau = dot_pi(pi, e, &z) / c0 - 0.5;
            let floor = f64::EPSILON * norm_pi(pi, &z) / enorm;
            // Below a tenth of the floor the recursive residual no longer describes the true one,
            // and iterating on regardless fed the Lanczos estimate rounding: on the 3x3 grid at
            // beta 2 the random scan ran 1,326 applications to a spurious ||A^-1|| of 5e15.
            let rel2 = rr / c0;
            let resolved = inv_norm * rel2.max(floor) <= rtol * (tau + 0.5);
            let below_floor = rel2 <= 0.01 * floor * floor;
            if resolved || below_floor {
                break;
            }
        }
        // The TRUE residual: the recursive one keeps falling after the true one has stopped.
        let az = a_op(&z);
        let rt: Vec<f64> = e.iter().zip(&az).map(|(a, b)| a - b).collect();
        let true_rel = norm_pi(pi, &rt) / enorm;
        let floor = f64::EPSILON * norm_pi(pi, &z) / enorm;
        let tau = dot_pi(pi, e, &z) / c0 - 0.5;
        let est = inv_norm * (true_rel * true_rel).max(floor);
        if est <= rtol * (tau + 0.5) {
            return Ok((tau, est));
        }
        if count.get() >= max_matvecs {
            return Err(AutocorrError::NotConverged { tau, err_est: est, matvecs: count.get() });
        }
        if true_rel <= floor || prev_true.is_some_and(|q| true_rel > 0.5 * q) {
            return Err(AutocorrError::AtFloor { tau, err_est: est, matvecs: count.get() });
        }
        prev_true = Some(true_rel);
        r = rt;
    }
}

/// Restarted GMRES([`GMRES_RESTART`]) on `A z = e` in `L^2(pi)`, classical Gram-Schmidt run twice,
/// Givens rotations. The error of the tau of an iterate is `<e, A^-1 r> / C0`, bounded by
/// `||A^-1|| ||r|| / ||e||`, and estimated as `max(||r||, u ||z||) / (sigma_min(R) ||e||)`:
/// `1 / sigma_min` of the rotated Hessenberg is a lower estimate of `||A^-1||` that only rises, and
/// `u ||z||` is what the f64 product `A z` can resolve -- without that floor term the first rule
/// accepted a residual of `4.1e-11` under a floor of `2.0e-10` and reported `5.3e-11` for a realised
/// `4.7e-10` (scout, `failures.txt` (g)). Accepted only on a TRUE residual at the end of a cycle.
fn solve_gmres(pi: &[f64], e: &[f64], c0: f64, a_op: &dyn Fn(&[f64]) -> Vec<f64>, count: &std::cell::Cell<usize>, rtol: f64, max_matvecs: usize) -> Solved {
    let enorm = c0.sqrt();
    let restart = GMRES_RESTART;
    let ld = restart + 1;
    let mut z = vec![0.0f64; e.len()];
    let mut r = e.to_vec();
    let mut inv_norm = 1.0f64;
    let mut prev_true: Option<f64> = None;
    loop {
        let beta0 = norm_pi(pi, &r);
        let (ez, zn0) = (dot_pi(pi, e, &z), norm_pi(pi, &z));
        let mut k = 0;
        if beta0 > 0.0 {
            let mut v: Vec<Vec<f64>> = vec![r.iter().map(|x| x / beta0).collect()];
            let mut ev = vec![dot_pi(pi, e, &v[0])];
            let mut h = vec![0.0f64; ld * ld];
            let (mut cs, mut sn) = (vec![0.0f64; restart], vec![0.0f64; restart]);
            let mut gv = vec![0.0f64; ld];
            gv[0] = beta0;
            for j in 0..restart {
                if count.get() >= max_matvecs {
                    break;
                }
                let mut w = a_op(&v[j]);
                for _ in 0..2 {
                    let c: Vec<f64> = v.iter().map(|vi| dot_pi(pi, vi, &w)).collect();
                    for (i, ci) in c.iter().enumerate() {
                        h[i * ld + j] += ci;
                        axpy(&mut w, -ci, &v[i]);
                    }
                }
                let hn = norm_pi(pi, &w);
                h[(j + 1) * ld + j] = hn;
                for i in 0..j {
                    let (a, b) = (h[i * ld + j], h[(i + 1) * ld + j]);
                    h[i * ld + j] = cs[i] * a + sn[i] * b;
                    h[(i + 1) * ld + j] = -sn[i] * a + cs[i] * b;
                }
                let (a, b) = (h[j * ld + j], h[(j + 1) * ld + j]);
                let rho = a.hypot(b);
                cs[j] = a / rho;
                sn[j] = b / rho;
                h[j * ld + j] = rho;
                h[(j + 1) * ld + j] = 0.0;
                gv[j + 1] = -sn[j] * gv[j];
                gv[j] *= cs[j];
                k = j + 1;
                let breakdown = !(hn > 1e-300);
                if !breakdown {
                    v.push(w.iter().map(|x| x / hn).collect());
                    ev.push(dot_pi(pi, e, &v[j + 1]));
                }
                if k % 5 == 0 || breakdown {
                    inv_norm = inv_norm.max(1.0 / sigma_min_upper(&h, ld, k));
                    if past_f64(inv_norm) {
                        return Err(AutocorrError::BeyondF64 { inv_norm, matvecs: count.get() });
                    }
                    let y = back_substitute(&h, ld, &gv, k);
                    let tau = (ez + y.iter().zip(&ev).map(|(a, b)| a * b).sum::<f64>()) / c0 - 0.5;
                    let rel = gv[k].abs() / enorm;
                    // ||z + V y|| <= ||z|| + ||y||, V orthonormal.
                    let floor = f64::EPSILON * (zn0 + y.iter().map(|a| a * a).sum::<f64>().sqrt()) / enorm;
                    // Below a tenth of the floor the recursive residual no longer describes the true
                    // one: end the cycle and look.
                    if inv_norm * rel.max(floor) <= rtol * (tau + 0.5) || rel <= 0.1 * floor || breakdown {
                        break;
                    }
                }
            }
            if k > 0 {
                inv_norm = inv_norm.max(1.0 / sigma_min_upper(&h, ld, k));
                if past_f64(inv_norm) {
                    return Err(AutocorrError::BeyondF64 { inv_norm, matvecs: count.get() });
                }
                let y = back_substitute(&h, ld, &gv, k);
                for (i, yi) in y.iter().enumerate() {
                    axpy(&mut z, *yi, &v[i]);
                }
            }
        }
        let az = a_op(&z);
        r = e.iter().zip(&az).map(|(a, b)| a - b).collect();
        let true_rel = norm_pi(pi, &r) / enorm;
        let floor = f64::EPSILON * norm_pi(pi, &z) / enorm;
        let tau = dot_pi(pi, e, &z) / c0 - 0.5;
        let est = inv_norm * true_rel.max(floor);
        if est <= rtol * (tau + 0.5) {
            return Ok((tau, est));
        }
        if count.get() >= max_matvecs {
            return Err(AutocorrError::NotConverged { tau, err_est: est, matvecs: count.get() });
        }
        // A cycle that does not halve the TRUE residual, or a true residual at `u ||z|| / ||e||`,
        // is the f64 floor: the tolerance asked for is not reachable for this chain.
        if true_rel <= floor || prev_true.is_some_and(|q| true_rel > 0.5 * q) || k == 0 {
            return Err(AutocorrError::AtFloor { tau, err_est: est, matvecs: count.get() });
        }
        prev_true = Some(true_rel);
    }
}

/// The exact integrated autocorrelation time by a MATRIX-FREE solve of the fundamental-matrix system
/// `(I - P + 1 pi^T) z = e` in the `pi`-weighted inner product, `tau = <e, z>_pi / C0 - 1/2`: the
/// number [`tau_int_fundamental`] computes, without its dense `2^n x 2^n` matrix, so up to
/// [`MAX_KRYLOV_SPINS`] where that stops at [`MAX_DENSE_SPINS`].
///
/// Conjugate gradients where the kernel is [`reversible`], GMRES otherwise. The law is the kernel's
/// OWN, never Boltzmann by default: closed forms at any size, the direct solve up to
/// [`MAX_DENSE_SPINS`] for the kernels whose law has no closed form (fabric, quantised, PIMI,
/// stale reads, site spread, tick-random), which are REFUSED above it; and every law is held to
/// invariance by one push first ([`LAW_TOLERANCE`]).
///
/// Stops when its error estimate is at most `rtol (tau + 1/2)`, confirmed on the true residual.
/// **The estimate is not a bound.** `||A^-1||` is estimated from below by the Krylov space, so it
/// can be optimistic: it covered the realised error in 8 of 8 heat-bath cases against a
/// double-double reference, and fell short by up to 14x on the fabric at `beta = 2` -- where the
/// solve had stopped at the floor and returned [`AutocorrError::AtFloor`], not a value. The
/// attainable accuracy is about `u ||A^-1||`: `1e-10` relative at `tau = 1.6e6`, `5e-7` at `3.6e8`.
/// `rho` is left empty (filling it costs 64 more applications); `lags` is 0.
///
/// Measured (scout and verifier, 2026-09-28): 12 to 133 applications at `n <= 12`, 32 to 102 at 14,
/// 37 to 314 at 18 and 20, growing with `ln tau`; against [`tau_int_fundamental`] to `1e-11`
/// relative wherever `tau < 3e4`.
///
/// # Errors
///
/// [`AutocorrError::TooManySpins`] above [`max_krylov_spins`]; [`AutocorrError::TooManyForDense`]
/// for a kernel whose law needs the dense solve, above [`MAX_DENSE_SPINS`];
/// [`AutocorrError::Reducible`] where that solve is singular; [`AutocorrError::LawNotInvariant`];
/// [`AutocorrError::NoVariance`]; and, of the solve, [`AutocorrError::NotConverged`] (budget),
/// [`AutocorrError::AtFloor`] (`rtol` is below what f64 can certify for this chain) and
/// [`AutocorrError::BeyondF64`] (its slowest mode is below f64's resolution): for those two,
/// [`tau_int_censored`].
pub fn tau_int_krylov(
    g: &Graph,
    beta: f64,
    kernel: Kernel,
    observable: impl Fn(&[i8]) -> f64,
    rtol: f64,
    max_matvecs: usize,
) -> Result<Autocorrelation, AutocorrError> {
    let cap = max_krylov_spins(kernel);
    if g.n > cap {
        return Err(AutocorrError::TooManySpins { n: g.n, max: cap });
    }
    let pi = checked_law(g, beta, kernel)?;
    let (e, c0) = centred(g.n, &pi, observable)?;
    // The invariance push is an application of the kernel too.
    let count = std::cell::Cell::new(1usize);
    let a_op = |v: &[f64]| -> Vec<f64> {
        count.set(count.get() + 1);
        let pv = apply(g, beta, kernel, v);
        // The rank-one term `1 <pi, v>`: without it `A` is singular, and on the 4x3 grid at beta 2
        // GMRES ran to its budget at a tau of 5.1e15 (scout, `failures.txt` (c)).
        let s = mean_pi(&pi, v);
        v.iter().zip(&pv).map(|(x, y)| x - y + s).collect()
    };
    let (route, solved) = if reversible(kernel, g) {
        (Route::Cg, solve_cg(&pi, &e, c0, &a_op, &count, rtol, max_matvecs))
    } else {
        (Route::Gmres, solve_gmres(&pi, &e, c0, &a_op, &count, rtol, max_matvecs))
    };
    let (tau, est) = solved?;
    Ok(Autocorrelation { tau_int: tau, lags: 0, rho: Vec::new(), variance: c0, route, matvecs: count.get(), err_est: Some(est) })
}

/// The states of `set`, sorted and distinct, checked to be states of `g`.
fn metastable_states(g: &Graph, set: Metastable<'_>) -> Result<Vec<usize>, AutocorrError> {
    let n = g.n;
    let m = 1usize << n;
    let minima = || {
        let mut out = Vec::new();
        for_each_state(n, |x, s| {
            if (0..n).all(|i| f64::from(s[i]) * g.field(i, s) >= 0.0) {
                out.push(x);
            }
        });
        out
    };
    let mut f = match set {
        Metastable::LocalMinima => minima(),
        Metastable::LocalMinimaAndNeighbours => {
            let lm = minima();
            let mut all = lm.clone();
            for &x in &lm {
                all.extend((0..n).map(|i| x ^ (1usize << i)));
            }
            all
        }
        Metastable::States(states) => states.to_vec(),
    };
    f.sort_unstable();
    f.dedup();
    match f.last() {
        Some(&last) if last < m => Ok(f),
        _ => Err(AutocorrError::BadMetastableSet),
    }
}

/// Restarted GMRES with classical Gram-Schmidt twice, EUCLIDEAN inner product, on `op x = b`, to a
/// relative TRUE residual `rtol` or `max_iter` applications. Returns `x` and its true relative
/// residual. The inner solve of [`tau_int_censored`].
fn gmres_euclid(op: &dyn Fn(&[f64]) -> Vec<f64>, b: &[f64], rtol: f64, max_iter: usize) -> (Vec<f64>, f64) {
    let dot = |u: &[f64], v: &[f64]| -> f64 {
        let mut acc = [0.0f64; 4];
        for (k, (a, c)) in u.iter().zip(v).enumerate() {
            acc[k & 3] += a * c;
        }
        (acc[0] + acc[1]) + (acc[2] + acc[3])
    };
    let restart = GMRES_RESTART;
    let ld = restart + 1;
    let bnorm = dot(b, b).sqrt();
    let mut x = vec![0.0f64; b.len()];
    if bnorm == 0.0 {
        return (x, 0.0);
    }
    let mut r = b.to_vec();
    let mut iters = 0;
    loop {
        let beta0 = dot(&r, &r).sqrt();
        let rel = beta0 / bnorm;
        if rel <= rtol || iters >= max_iter {
            return (x, rel);
        }
        let mut v: Vec<Vec<f64>> = vec![r.iter().map(|a| a / beta0).collect()];
        let mut h = vec![0.0f64; ld * ld];
        let (mut cs, mut sn) = (vec![0.0f64; restart], vec![0.0f64; restart]);
        let mut gv = vec![0.0f64; ld];
        gv[0] = beta0;
        let mut k = 0;
        for j in 0..restart {
            if iters >= max_iter {
                break;
            }
            let mut w = op(&v[j]);
            iters += 1;
            for _ in 0..2 {
                let c: Vec<f64> = v.iter().map(|vi| dot(vi, &w)).collect();
                for (i, ci) in c.iter().enumerate() {
                    h[i * ld + j] += ci;
                    axpy(&mut w, -ci, &v[i]);
                }
            }
            let hn = dot(&w, &w).sqrt();
            h[(j + 1) * ld + j] = hn;
            for i in 0..j {
                let (a, c) = (h[i * ld + j], h[(i + 1) * ld + j]);
                h[i * ld + j] = cs[i] * a + sn[i] * c;
                h[(i + 1) * ld + j] = -sn[i] * a + cs[i] * c;
            }
            let (a, c) = (h[j * ld + j], h[(j + 1) * ld + j]);
            let rho = a.hypot(c);
            cs[j] = a / rho;
            sn[j] = c / rho;
            h[j * ld + j] = rho;
            h[(j + 1) * ld + j] = 0.0;
            gv[j + 1] = -sn[j] * gv[j];
            gv[j] *= cs[j];
            k = j + 1;
            if gv[j + 1].abs() / bnorm <= 0.5 * rtol || !(hn > 1e-300) {
                break;
            }
            v.push(w.iter().map(|a| a / hn).collect());
        }
        if k == 0 {
            return (x, rel);
        }
        let y = back_substitute(&h, ld, &gv, k);
        for (i, yi) in y.iter().enumerate() {
            axpy(&mut x, *yi, &v[i]);
        }
        let ax = op(&x);
        iters += 1;
        r = b.iter().zip(&ax).map(|(p, q)| p - q).collect();
    }
}

/// Solve `S z = b`, `S = I - P_c` the generator of a censored chain whose OFF-DIAGONAL transition
/// probabilities are `rate[a * nf + c]`, with `z(keep) = 0` (the system is singular with null vector
/// `1`, and consistent). Gaussian elimination of every state but `keep`, in which each pivot -- the
/// total rate out of the state being eliminated, to the states still in play -- is recomputed as a
/// SUM of off-diagonal rates, never as `1 - P_c(f, f)`: the rule of Grassmann, Taksar and Heyman
/// (1985). Every quantity the matrix side touches is then a sum of products of nonnegative numbers,
/// accurate componentwise however small; the rates here reach `2.3e-245`. With `1 - P_c(f, f)` in
/// its place the four-city TSP's threshold tau moved from `1.64e24` to `4.5e34`, and to NaN at four
/// times the threshold (scout, `gthcheck.txt`). `None` when a state has no rate out: the censored
/// chain is reducible.
fn gth_poisson(rate: &[f64], b: &[f64], nf: usize, keep: usize) -> Option<Vec<f64>> {
    let mut rate = rate.to_vec();
    let mut b = b.to_vec();
    let mut alive = vec![true; nf];
    let order: Vec<usize> = (0..nf).filter(|&a| a != keep).collect();
    let mut out = vec![0.0f64; nf];
    for &k in &order {
        alive[k] = false;
        let mut total = 0.0f64;
        for j in 0..nf {
            if alive[j] {
                total += rate[k * nf + j];
            }
        }
        if !(total > 0.0) {
            return None;
        }
        out[k] = total;
        for i in 0..nf {
            let rik = rate[i * nf + k];
            if !alive[i] || rik == 0.0 {
                continue;
            }
            // Row i absorbs row k: every flow i -> k is continued along k's rates out.
            let factor = rik / total;
            for j in 0..nf {
                if alive[j] && j != i {
                    rate[i * nf + j] += factor * rate[k * nf + j];
                }
            }
            b[i] += factor * b[k];
            rate[i * nf + k] = 0.0;
        }
    }
    // Back substitution in reverse elimination order; row k still holds its rates at the time it was
    // eliminated, to exactly the states eliminated after it and `keep`.
    let mut z = vec![0.0f64; nf];
    for &k in order.iter().rev() {
        let mut acc = b[k];
        for j in 0..nf {
            if j != k {
                acc += rate[k * nf + j] * z[j];
            }
        }
        z[k] = acc / out[k];
    }
    Some(z)
}

/// The largest spin count [`SweepTable`] tables: two `f64` per site per state is 84 MB at 18 spins.
const MAX_TABLED_SPINS: usize = 18;

/// A sweep kernel's site probabilities, tabled once for a route that pushes the same kernel hundreds
/// or thousands of times -- the censored solves and [`time_to_mass`]. The sites in the order
/// [`apply_distribution`] updates them, and for each the two tails [`site_pair`] gives at every
/// state; a push is then the same products and sums [`apply_distribution`] forms, in the same
/// order, bit for bit (`a_tabled_push_is_apply_distribution_to_the_bit`), at a fifteenth of the cost
/// on the TSP chains.
struct SweepTable {
    order: Vec<usize>,
    up: Vec<Vec<f64>>,
    down: Vec<Vec<f64>>,
}

impl SweepTable {
    /// `None` for a kernel that is not a site-by-site sweep, or above [`MAX_TABLED_SPINS`].
    fn new(g: &Graph, beta: f64, kernel: Kernel) -> Option<SweepTable> {
        if g.n > MAX_TABLED_SPINS {
            return None;
        }
        let order: Vec<usize> = match kernel {
            Kernel::SequentialGibbs => (0..g.n).collect(),
            Kernel::ChromaticGibbs | Kernel::FixedFabric | Kernel::Quantised { .. } | Kernel::SiteSpread { .. } => {
                g.classes.iter().flat_map(|c| c.iter().map(|&i| i as usize)).collect()
            }
            _ => return None,
        };
        let m = 1usize << g.n;
        let (mut up, mut down) = (Vec::with_capacity(g.n), Vec::with_capacity(g.n));
        for &i in &order {
            let (mut u, mut d) = (vec![0.0f64; m], vec![0.0f64; m]);
            for_each_state(g.n, |x, s| {
                // The sequential arm of `apply_distribution` calls `p_pair` directly, which is what
                // `site_pair` does for it: the same two numbers.
                let (p, q) = site_pair(g, beta, kernel, i, s);
                u[x] = p;
                d[x] = q;
            });
            up.push(u);
            down.push(d);
        }
        Some(SweepTable { order, up, down })
    }

    /// `mu P`, as [`apply_distribution`] computes it.
    fn push(&self, mu: &[f64]) -> Vec<f64> {
        let mut cur = mu.to_vec();
        let mut next = vec![0.0f64; mu.len()];
        for (k, &i) in self.order.iter().enumerate() {
            let bit = 1usize << i;
            let (up, down) = (&self.up[k], &self.down[k]);
            for x in 0..mu.len() {
                let pooled = cur[x | bit] + cur[x & !bit];
                next[x] = if x & bit != 0 { up[x] * pooled } else { down[x] * pooled };
            }
            std::mem::swap(&mut cur, &mut next);
        }
        cur
    }
}

/// `mu P` through the table when there is one, [`apply_distribution`] otherwise.
fn push_with(table: Option<&SweepTable>, g: &Graph, beta: f64, kernel: Kernel, mu: &[f64]) -> Vec<f64> {
    match table {
        Some(t) => t.push(mu),
        None => apply_distribution(g, beta, kernel, mu),
    }
}

/// What both censored routes build: a metastable set `F`, the rest `T`, pushes of the kernel, and
/// the row-vector solve `v (I - P_TT) = r` on `T`.
struct Censor<'a> {
    g: &'a Graph,
    beta: f64,
    kernel: Kernel,
    fset: Vec<usize>,
    in_f: Vec<usize>,
    inner_rtol: f64,
    pushes: std::cell::Cell<usize>,
    table: Option<SweepTable>,
}

impl<'a> Censor<'a> {
    fn new(g: &'a Graph, beta: f64, kernel: Kernel, set: Metastable<'_>, inner_rtol: f64) -> Result<Self, AutocorrError> {
        let fset = metastable_states(g, set)?;
        let mut in_f = vec![usize::MAX; 1usize << g.n];
        for (k, &f) in fset.iter().enumerate() {
            in_f[f] = k;
        }
        let table = SweepTable::new(g, beta, kernel);
        Ok(Censor { g, beta, kernel, fset, in_f, inner_rtol, pushes: std::cell::Cell::new(0), table })
    }

    fn in_t(&self, x: usize) -> bool {
        self.in_f[x] == usize::MAX
    }

    fn push(&self, mu: &[f64]) -> Vec<f64> {
        self.pushes.set(self.pushes.get() + 1);
        push_with(self.table.as_ref(), self.g, self.beta, self.kernel, mu)
    }

    /// `v` with `v (I - P_TT) = r|_T`: restarted GMRES on pushes restricted to `T`, to a relative
    /// true residual `inner_rtol`.
    fn solve_t(&self, r: &[f64]) -> Result<Vec<f64>, AutocorrError> {
        let rhs: Vec<f64> = r.iter().enumerate().map(|(x, &v)| if self.in_t(x) { v } else { 0.0 }).collect();
        let op = |u: &[f64]| -> Vec<f64> {
            let up = self.push(u);
            u.iter().zip(&up).enumerate().map(|(x, (a, b))| if self.in_t(x) { a - b } else { 0.0 }).collect()
        };
        let (v, res) = gmres_euclid(&op, &rhs, self.inner_rtol, 4_000);
        if res <= self.inner_rtol {
            Ok(v)
        } else {
            Err(AutocorrError::InnerSolve { residual: res, pushes: self.pushes.get() })
        }
    }

    /// The censored chain's OFF-DIAGONAL transition probabilities (row-major over `F`, diagonal
    /// zero), and for each `f` the sums `<u_f, v>` over `T` for each `v` in `against`, where
    /// `u_f = P(f, T) (I - P_TT)^-1` is the expected number of visits to each state of `T` between
    /// leaving `f` and the next return to `F`.
    fn rows(&self, against: &[&[f64]]) -> Result<(Vec<f64>, Vec<Vec<f64>>), AutocorrError> {
        let m = 1usize << self.g.n;
        let nf = self.fset.len();
        let mut rate = vec![0.0f64; nf * nf];
        let mut proj = vec![vec![0.0f64; against.len()]; nf];
        for (k, &f) in self.fset.iter().enumerate() {
            let mut delta = vec![0.0f64; m];
            delta[f] = 1.0;
            let row = self.push(&delta);
            for (c, &g2) in self.fset.iter().enumerate() {
                rate[k * nf + c] += row[g2];
            }
            let u = self.solve_t(&row)?;
            let up = self.push(&u);
            for (c, &g2) in self.fset.iter().enumerate() {
                rate[k * nf + c] += up[g2];
            }
            for (j, v) in against.iter().enumerate() {
                proj[k][j] = (0..m).filter(|&x| self.in_t(x)).map(|x| u[x] * v[x]).sum();
            }
        }
        // The diagonal is never read: GTH recomputes it. A negative rate is an inner solve's rounding
        // around a true zero; one that is not negligible against its row is an error, not a zero.
        for a in 0..nf {
            rate[a * nf + a] = 0.0;
            let top = (0..nf).map(|c| rate[a * nf + c]).fold(0.0f64, f64::max);
            for c in 0..nf {
                let v = rate[a * nf + c];
                if v < 0.0 {
                    if -v > 1e-12 * top {
                        return Err(AutocorrError::InnerSolve { residual: -v / top, pushes: self.pushes.get() });
                    }
                    rate[a * nf + c] = 0.0;
                }
            }
        }
        Ok((rate, proj))
    }
}

/// The exact tau of a chain whose slowest modes sit below f64's resolution of `1 - lambda`, where
/// [`tau_int_krylov`] and [`tau_int_fundamental`] have no digit to give, by CENSORING it onto a
/// metastable set `F`. With `T` the rest, `(I - P) z = e` eliminates exactly to
///
/// ```text
///   S z_F = b_F,   S = I - P_FF - P_FT (I - P_TT)^-1 P_TF      (the censored chain's generator)
///   b_F = e_F + P_FT (I - P_TT)^-1 e_T
///   sum_T pi e z_T = <w, e_T> + sum_f z_F(f) (w P)(f),     w (I - P_TT) = (pi e)_T
/// ```
///
/// and `sum_x pi(x) e(x) z(x)` does not depend on which solution of the singular system is taken.
/// The `|F| + 1` inner solves with `I - P_TT` are GMRES on pushes of the kernel restricted to `T`
/// (to relative residual `inner_rtol`), and are well conditioned when `F` holds every trap: 5 to 24
/// iterations each on the TSP chains. `S` is `|F| x |F|` and is eliminated GTH-style (every
/// pivot a sum of rates, never `1 - P(f, f)`).
///
/// Exact for ANY non-empty `F`: the set changes the cost and the conditioning, not the answer. That
/// is also the check this route has in place of an error estimate: two sets with entirely different
/// inner systems ([`Metastable::LocalMinima`] and [`Metastable::LocalMinimaAndNeighbours`]) agreed to
/// `3e-14` on 23 of the 24 frozen TSP cells and `8.6e-11` on the other (scout, 2026-09-28), and on
/// the cold 12-spin grid this route was `3.3e-12` from a double-double reference where the dense
/// solve was `3.2e-9`. Where `F` misses the kernel's traps the inner solves are ill-conditioned and
/// the route REFUSES ([`AutocorrError::InnerSolve`]): the synchronous sweep's traps are not the
/// energy's local minima, and on the 3x3 grid at beta 2 it stops there. `rho` is left empty and
/// `err_est` is `None`; `matvecs` counts pushes.
///
/// # Errors
///
/// As [`tau_int_krylov`] for the size, the law and the observable; [`AutocorrError::BadMetastableSet`]
/// for an empty set or a state outside `0..2^n`; [`AutocorrError::InnerSolve`] when an inner solve
/// misses `inner_rtol` within its budget; [`AutocorrError::Reducible`] when the censored chain has a
/// state with no way out.
pub fn tau_int_censored(
    g: &Graph,
    beta: f64,
    kernel: Kernel,
    observable: impl Fn(&[i8]) -> f64,
    set: Metastable<'_>,
    inner_rtol: f64,
) -> Result<Autocorrelation, AutocorrError> {
    let cap = max_krylov_spins(kernel);
    if g.n > cap {
        return Err(AutocorrError::TooManySpins { n: g.n, max: cap });
    }
    let pi = checked_law(g, beta, kernel)?;
    let (e, c0) = centred(g.n, &pi, observable)?;
    let c = Censor::new(g, beta, kernel, set, inner_rtol)?;
    let (m, nf) = (1usize << g.n, c.fset.len());
    let (rate, proj) = c.rows(&[&e])?;
    let bf: Vec<f64> = c.fset.iter().enumerate().map(|(k, &f)| e[f] + proj[k][0]).collect();
    let pe: Vec<f64> = pi.iter().zip(&e).map(|(p, x)| p * x).collect();
    let w = c.solve_t(&pe)?;
    let wp = c.push(&w);
    let we: f64 = (0..m).filter(|&x| c.in_t(x)).map(|x| w[x] * e[x]).sum();
    // F is never empty (`metastable_states` refuses an empty set), so the 0 is never read.
    let keep = (0..nf).max_by(|&a, &b| pi[c.fset[a]].total_cmp(&pi[c.fset[b]])).unwrap_or(0);
    let zf = gth_poisson(&rate, &bf, nf, keep).ok_or(AutocorrError::Reducible)?;
    let mut total = we;
    for (k, &f) in c.fset.iter().enumerate() {
        total += pi[f] * e[f] * zf[k] + zf[k] * wp[f];
    }
    Ok(Autocorrelation {
        tau_int: total / c0 - 0.5,
        lags: 0,
        rho: Vec::new(),
        variance: c0,
        route: Route::Censored,
        // The law's invariance push and the censored solve's own.
        matvecs: 1 + c.pushes.get(),
        err_est: None,
    })
}

/// How [`time_to_mass`] found its answer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MassRoute {
    /// The law pushed through the kernel one step at a time until it held: exact, an integer.
    Pushed,
    /// Past the push budget, on the chain censored onto a metastable set and run in REAL time: a
    /// reduction, not an identity -- see [`time_to_mass`] for what it assumes and how far to trust it.
    SlowScale,
}

/// When a law pushed through a kernel first holds its stationary mass on a set of states.
#[derive(Clone, Debug, PartialEq)]
pub struct MassTime {
    /// Kernel steps (sweeps, for a sweep kernel) from the start law until
    /// `|m(t) - m_inf| <= rel m_inf` first holds, `m(t)` the law's mass on the set after `t` steps.
    pub steps: f64,
    /// `m_inf`, the stationary mass on the set under the kernel's own law.
    pub stationary: f64,
    /// How `steps` was found.
    pub route: MassRoute,
    /// For [`MassRoute::SlowScale`], `|m(K) - m_slow(K)|` at the hand-over step `K`: the pushed
    /// mass against the reduction's, which is how far the fast transient still was from dead when
    /// the reduction took over. Zero for [`MassRoute::Pushed`].
    pub handover: f64,
    /// Pushes of the kernel made, the law's invariance check included.
    pub pushes: usize,
}

/// `a b` for two `nf x nf` row-stochastic matrices, with every diagonal recomputed as one minus its
/// row's off-diagonals -- the GTH rule for a product. Squaring a matrix whose diagonal has rounded to
/// exactly 1 while a rate of `1e-18` sits beside it otherwise gains that rate as MASS every time, and
/// `2^60` squarings later the rows sum to `e`: the first version of [`time_to_mass`] ran its doubling
/// grid to infinity on a cold pair censored onto all four of its states that way.
fn stochastic_product(a: &[f64], b: &[f64], nf: usize) -> Vec<f64> {
    let mut c = vec![0.0f64; nf * nf];
    for i in 0..nf {
        for k in 0..nf {
            let aik = a[i * nf + k];
            if aik == 0.0 {
                continue;
            }
            for j in 0..nf {
                if j != i {
                    c[i * nf + j] += aik * b[k * nf + j];
                }
            }
        }
    }
    set_stochastic_diagonal(&mut c, nf);
    c
}

/// Every diagonal entry as one minus its row's off-diagonals, never below zero.
fn set_stochastic_diagonal(c: &mut [f64], nf: usize) {
    for i in 0..nf {
        let off: f64 = (0..nf).filter(|&j| j != i).map(|j| c[i * nf + j]).sum();
        c[i * nf + i] = (1.0 - off).max(0.0);
    }
}

/// `exp(h R)` for a generator `R` given by its off-diagonal RATES (`rate[f * nf + g]`, `f != g`) and
/// its uniformisation constant `lambda >= max_f sum_g rate`, by uniformisation: `e^{-h lambda}
/// sum_j (h lambda)^j / j! M^j`, `M = I + R / lambda` nonnegative, truncated at `j = 5` when
/// `h lambda <= 1e-4` (error below `1e-22`), then squared back up to `h`. Off the diagonal every
/// entry is a sum of products of nonnegative numbers, and every diagonal is recomputed from its row
/// ([`stochastic_product`]), so the rows sum to one however many squarings there are.
fn generator_exp(rate: &[f64], nf: usize, lambda: f64, h: f64) -> Vec<f64> {
    let mut squarings = 0;
    let mut step = h;
    while step * lambda > 1e-4 {
        step *= 0.5;
        squarings += 1;
    }
    let mut mm = vec![0.0f64; nf * nf];
    for f in 0..nf {
        for g in 0..nf {
            if g != f {
                mm[f * nf + g] = rate[f * nf + g] / lambda;
            }
        }
    }
    set_stochastic_diagonal(&mut mm, nf);
    let x = step * lambda;
    let mut e = vec![0.0f64; nf * nf];
    let mut term = vec![0.0f64; nf * nf];
    for f in 0..nf {
        term[f * nf + f] = 1.0;
    }
    let mut coef = 1.0;
    for j in 0..=5 {
        if j > 0 {
            term = stochastic_product(&term, &mm, nf);
            coef *= x / f64::from(j);
        }
        for (a, b) in e.iter_mut().zip(&term) {
            *a += coef * b;
        }
    }
    let scale = (-x).exp();
    e.iter_mut().for_each(|v| *v *= scale);
    set_stochastic_diagonal(&mut e, nf);
    for _ in 0..squarings {
        e = stochastic_product(&e, &e, nf);
    }
    e
}

/// The kernel steps until a law pushed from `start` first holds its stationary mass on the states
/// `target` picks, to a relative `rel`: the first `t` with `|m(t) - m_inf| <= rel m_inf`.
///
/// Up to `max_pushes` it pushes the law one step at a time with [`apply_distribution`] and the
/// answer is EXACT ([`MassRoute::Pushed`]). A chain frozen by barriers needs far more steps than
/// any budget -- the four-city TSP at its provable penalty relaxes over `1e17` sweeps and more --
/// and past the budget this answers on the SLOW SCALE ([`MassRoute::SlowScale`]): the chain is
/// censored onto the metastable set `set` exactly as [`tau_int_censored`] censors it, and each
/// state `f` of the set becomes a LABEL, "the last metastable state visited", which moves to `g`
/// at the real-time rate `P_c(f, g) / L_f` -- `P_c` the censored chain, `L_f = 1 + sum_T u_f` the
/// expected steps per visit to `f` -- and carries the mass `(1_S(f) + sum_T u_f 1_S) / L_f` on the
/// set `S`. The law pushed to the budget `K` is handed to the labels by where it will next enter
/// the set; `exp(t R)` is taken by uniformisation and squaring with nonnegative arithmetic
/// throughout; the crossing is bracketed on a doubling grid and bisected.
///
/// **What that assumes.** That by step `K` every mode faster than the barrier crossings has died,
/// and that a label's occupants are then distributed as its return cycle says. Both errors are of
/// order `t_fast / t_slow` -- the relaxation inside a basin over the time to leave it -- which is
/// below `1e-12` on a chain frozen at `tau = 1e17` and is `1e-3`-ish where the barrier is low
/// enough to push through. `handover` reports the first directly: the pushed mass at `K` against
/// the reduction's. The route is a reduction, validated where both routes run
/// (`the_slow_scale_mass_time_is_the_pushed_one_where_both_run`), not an identity.
///
/// # Errors
///
/// As [`tau_int_censored`]; [`AutocorrError::Reducible`] when the censored chain has no rate out
/// of anywhere; [`AutocorrError::NotConverged`] (with `tau` the last time tried) when the doubling
/// grid runs out before the mass holds.
///
/// # Panics
///
/// If `start` does not have `2^n` entries.
#[allow(clippy::too_many_arguments)]
pub fn time_to_mass(
    g: &Graph,
    beta: f64,
    kernel: Kernel,
    start: &[f64],
    target: impl Fn(usize) -> bool,
    rel: f64,
    max_pushes: usize,
    set: Metastable<'_>,
) -> Result<MassTime, AutocorrError> {
    let cap = max_krylov_spins(kernel);
    if g.n > cap {
        return Err(AutocorrError::TooManySpins { n: g.n, max: cap });
    }
    let m = 1usize << g.n;
    assert_eq!(start.len(), m, "a start law over states has 2^n entries");
    let pi = checked_law(g, beta, kernel)?;
    let tgt: Vec<f64> = (0..m).map(|x| if target(x) { 1.0 } else { 0.0 }).collect();
    let stationary: f64 = pi.iter().zip(&tgt).map(|(p, t)| p * t).sum();
    let holds = |mass: f64| (mass - stationary).abs() <= rel * stationary;
    let mass_of = |mu: &[f64]| mu.iter().zip(&tgt).map(|(p, t)| p * t).sum::<f64>();
    let table = SweepTable::new(g, beta, kernel);
    let mut mu = start.to_vec();
    for k in 1..=max_pushes {
        mu = push_with(table.as_ref(), g, beta, kernel, &mu);
        if holds(mass_of(&mu)) {
            return Ok(MassTime { steps: k as f64, stationary, route: MassRoute::Pushed, handover: 0.0, pushes: 1 + k });
        }
    }
    // The slow scale: labels on the metastable set, run in real time.
    let c = Censor::new(g, beta, kernel, set, 1e-13)?;
    let nf = c.fset.len();
    let ones = vec![1.0f64; m];
    let (rate, proj) = c.rows(&[&ones, &tgt])?;
    let cycle: Vec<f64> = (0..nf).map(|k| 1.0 + proj[k][0]).collect();
    let obs: Vec<f64> = c.fset.iter().enumerate().map(|(k, &f)| (tgt[f] + proj[k][1]) / cycle[k]).collect();
    let arrive = c.solve_t(&mu)?;
    let into = c.push(&arrive);
    let a0: Vec<f64> = c.fset.iter().map(|&f| mu[f] + into[f]).collect();
    let slow_mass = |a: &[f64]| a.iter().zip(&obs).map(|(x, o)| x * o).sum::<f64>();
    let handover = (mass_of(&mu) - slow_mass(&a0)).abs();
    let real: Vec<f64> = (0..nf * nf).map(|i| rate[i] / cycle[i / nf]).collect();
    let lambda = (0..nf).map(|f| (0..nf).map(|g2| real[f * nf + g2]).sum::<f64>()).fold(0.0f64, f64::max);
    if !(lambda > 0.0) {
        return Err(AutocorrError::Reducible);
    }
    let advance = |a: &[f64], e: &[f64]| -> Vec<f64> {
        let mut out = vec![0.0f64; nf];
        for (f, &af) in a.iter().enumerate() {
            for (g2, o) in out.iter_mut().enumerate() {
                *o += af * e[f * nf + g2];
            }
        }
        out
    };
    // Doubling grid from 1e-4 / lambda: a(2^j h0) = a0 E(h0)^(2^j).
    let h0 = 1e-4 / lambda;
    let mut e = generator_exp(&real, nf, lambda, h0);
    let (mut lo, mut a_lo, mut hi) = (0.0f64, a0.clone(), h0);
    let mut found = false;
    for _ in 0..4_000 {
        let a_hi = advance(&a0, &e);
        if holds(slow_mass(&a_hi)) {
            found = true;
            break;
        }
        lo = hi;
        a_lo = a_hi;
        hi *= 2.0;
        e = stochastic_product(&e, &e, nf);
    }
    let pushes = 1 + max_pushes + c.pushes.get();
    if !found {
        return Err(AutocorrError::NotConverged { tau: max_pushes as f64 + hi, err_est: f64::INFINITY, matvecs: pushes });
    }
    // Bisect (lo, hi]: the mass holds at hi and not at lo.
    for _ in 0..200 {
        if hi - lo <= 1e-13 * hi {
            break;
        }
        let mid = 0.5 * (lo + hi);
        let a_mid = advance(&a_lo, &generator_exp(&real, nf, lambda, mid - lo));
        if holds(slow_mass(&a_mid)) {
            hi = mid;
        } else {
            lo = mid;
            a_lo = a_mid;
        }
    }
    Ok(MassTime { steps: max_pushes as f64 + hi, stationary, route: MassRoute::SlowScale, handover, pushes })
}

/// The exact tau by the route that can deliver it: [`tau_int_krylov`] first, and
/// [`tau_int_censored`] on [`Metastable::LocalMinima`] (inner residual `1e-13`) where Krylov
/// returns [`AutocorrError::AtFloor`] or [`AutocorrError::BeyondF64`] -- a chain too cold for f64 to
/// certify `rtol` -- with [`Autocorrelation::route`] saying which answered.
///
/// It does not try [`tau_int_fundamental`] first below [`MAX_DENSE_SPINS`], though it could: at 12
/// spins the dense solve takes 8 to 250 s of one core against under a second for Krylov, and on a
/// cold heat bath it is the least accurate of the three routes (see [`Route::Dense`]). The dense
/// solve stays what it was, the independent cross-check.
///
/// # Errors
///
/// As [`tau_int_krylov`] and, on the fallback, [`tau_int_censored`].
pub fn tau_int_solved(
    g: &Graph,
    beta: f64,
    kernel: Kernel,
    observable: impl Fn(&[i8]) -> f64,
    rtol: f64,
    max_matvecs: usize,
) -> Result<Autocorrelation, AutocorrError> {
    match tau_int_krylov(g, beta, kernel, &observable, rtol, max_matvecs) {
        Err(AutocorrError::AtFloor { .. } | AutocorrError::BeyondF64 { .. }) => {
            tau_int_censored(g, beta, kernel, &observable, Metastable::LocalMinima, 1e-13)
        }
        other => other,
    }
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
            // Every lag is zero, and the sum stops after QUIET_LAGS of them in a row.
            assert_eq!(a.lags, QUIET_LAGS);
        }
    }

    /// THE COMPLEMENT, KERNEL BY KERNEL. One spin in a field so strong that `1 - p_up` is exactly
    /// zero (`2 beta h = 40`), or so strong that the tail is near the bottom of the normal range
    /// (`700`): the probability of leaving `+1` must be the exact tail, and nonzero, to 1e-14 --
    /// through `apply` and through `apply_distribution`, for every kernel whose update is a heat
    /// bath or is built on one. Until 2026-09-28 every one of them formed it as `1 - p` and returned
    /// exactly zero, which on the TSP chains of `examples/penalty_mixing.rs` made every tour a trap
    /// with no exit and the exact `tau_int` millions of times too long. The fabric's kernels are
    /// absent on purpose: their comparator's complement is exact by construction and is `0` below
    /// `2^-16`, which is the hardware, not a rounding.
    #[test]
    fn a_down_flip_against_a_strong_field_has_its_exact_probability_in_every_kernel() {
        // 1 / (1 + e^x) and e^-x by mpmath at 40 digits, rounded to the nearest f64 (at these x they
        // are the same f64): independent of this file. The informed chain's move probability is
        // `min(1, e^-x)` under every balance.
        let cases = [
            (40.0, 4.248354255291589e-18, 4.248354255291589e-18),
            (700.0, 9.85967654375977e-305, 9.85967654375977e-305),
        ];
        // PIMI at xi 0.5, eta 0.1: tanh(beta h) is 1 in f64 at both fields, so the tail is
        // Phi(-(1 + 0.5) / 0.1) = Phi(-15).
        let pimi_tail = 3.670966199312751e-51;
        for &(x, heat, informed) in &cases {
            let mut b = GraphBuilder::new(1);
            b.bias(0, x / 2.0);
            let g = b.build();
            let kernels = [
                (Kernel::ChromaticGibbs, 1.0, heat),
                (Kernel::SequentialGibbs, 1.0, heat),
                (Kernel::RandomScan, 1.0, heat),
                (Kernel::Synchronous, 1.0, heat),
                (Kernel::Stale { p: 0.3 }, 1.0, heat),
                (Kernel::SiteSpread { seed: 3, spread: 0.0 }, 1.0, heat),
                (Kernel::TickRandom { p: 0.5 }, 1.0, 0.5 * heat),
                // SCA reads half the field: at beta 2 its tail is the heat bath's at beta 1.
                (Kernel::Sca { q: 0.0 }, 2.0, heat),
                (Kernel::Pimi { xi: 0.5, eta: 0.1 }, 1.0, pimi_tail),
                (Kernel::Informed(Balance::Barker), 1.0, informed),
                (Kernel::Informed(Balance::Sqrt), 1.0, informed),
                (Kernel::Informed(Balance::Metropolis), 1.0, informed),
            ];
            for (kernel, beta, want) in kernels {
                // Mass from +1 (state 1) that lands on -1 (state 0), and the expectation from +1 of
                // the indicator of -1: the same transition probability by the two operators.
                let pushed = apply_distribution(&g, beta, kernel, &[0.0, 1.0])[0];
                let pulled = apply(&g, beta, kernel, &[1.0, 0.0])[1];
                for (route, got) in [("apply_distribution", pushed), ("apply", pulled)] {
                    assert!(got > 0.0, "{kernel:?} at 2 beta h = {x}: {route} gives {got}, a down-flip made impossible");
                    let rel = (got - want).abs() / want;
                    assert!(rel < 1e-14, "{kernel:?} at 2 beta h = {x}: {route} {got:e} against {want:e}, rel {rel:e}");
                }
            }
        }
    }

    /// The normal tails PIMI's kernel is built from, against mpmath at 40 digits, on both sides of
    /// the switch from `erf` to the continued fraction (`|z| = sqrt 2`) and past the point where
    /// `erf` returns exactly one (`|z| = 7.07`), and the two tails must sum to one.
    #[test]
    fn the_normal_tails_are_accurate_on_both_sides_and_far_out() {
        let exact = [
            (0.3, 0.3820885778110474),
            (1.0, 0.15865525393145705),
            (1.5, 0.06680720126885807),
            (2.0, 0.02275013194817921),
            (3.0, 0.0013498980316300946),
            (5.0, 2.866515718791939e-7),
            (7.5, 3.1908916729108963e-14),
            (10.0, 7.619853024160525e-24),
            (15.0, 3.670966199312751e-51),
            (20.0, 2.7536241186062337e-89),
            (37.0, 5.725571222524577e-300),
        ];
        for &(z, lower) in &exact {
            let (up, down) = normal_tails(z);
            let rel = (down - lower).abs() / lower;
            assert!(rel < 4e-15, "Phi(-{z}) = {down:e} against {lower:e}, rel {rel:e}");
            let (up_neg, down_neg) = normal_tails(-z);
            assert_eq!(up_neg.to_bits(), down.to_bits(), "Phi(-z) must be the same number from either side at {z}");
            assert_eq!(down_neg.to_bits(), up.to_bits());
            assert!((up + down - 1.0).abs() <= f64::EPSILON, "tails at {z} sum to {}", up + down);
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
                        let f = g.field(i, &s);
                        if (a >> j) & 1 == 1 {
                            y |= 1usize << i;
                            w *= p_up(f, beta);
                        } else {
                            w *= p_up(-f, beta);
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
        // The elimination is accurate to about u times the chain's conditioning, and Kemeny's
        // constant -- the sum of every mode's relaxation time -- measures that: u K is 1.6e-13,
        // 2.8e-12 and 1.4e-10 at beta 1, 1.5 and 2 on this grid, and the measured gaps were 6.6e-14,
        // 2.5e-12 and 2.0e-10. A fixed 1e-12 here held until the sweep's arithmetic changed in its
        // last bits (2026-09-28, the complement and the site-by-site class contraction), when the
        // same solve landed at 2.5e-12: the bound was below the solve's own floor, not the kernel's.
        let floor = f64::EPSILON * kemeny_constant(&g, beta, Kernel::ChromaticGibbs).unwrap();
        assert!(gap < 16.0 * floor, "direct solve vs Boltzmann: {gap}, attainable about {floor:e}");
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
                worst_full = worst_full.max((site_pair(&g, beta, full, i, &s).0 - exact).abs());
                worst_wide = worst_wide.max((site_pair(&g, beta, p32, i, &s).0 - site_pair(&g, beta, p24, i, &s).0).abs());
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
                    let b = beta * site_factor(7, 0.3, i);
                    if s[i] > 0 { p_up(free.h[i], b) } else { p_up(-free.h[i], b) }
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
        let got = site_pair(&g, 1.0, k, 0, &[-1]).0;
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
    /// law, and it is REVERSIBLE (zero entropy production) where the fixed-order sweep on a coupled
    /// grid only keeps the law invariant. Then the comparison at equal work, `tau_int` per site
    /// update, with a closed form as the control: for uncoupled spins a random scan leaves each site
    /// untouched with probability `1 − 1/n` per step, so the magnetisation's `tau` is `n − 1/2`
    /// steps, while one fixed-order sweep refreshes every site and its `tau` is `1/2` sweep, `n/2`
    /// updates -- a ratio of exactly `2 − 1/n`. Uncoupled, that sweep is REVERSIBLE (it draws a
    /// fresh sample, `P = 1 pi^T`), so the factor does not come from reversibility; it is `1 + c²`
    /// for the squared coefficient of variation `c² = 1 − 1/n` of the geometric wait between a
    /// random scan's visits to one site. Coupled, `examples/scan_order_exact.rs` measures 1.824 to
    /// 1.974 for the magnetisation on both 5x2 fixtures from `beta` 0.2 to 3, falling to 1.737 on a
    /// 4x2 ferromagnet at `beta = 3`, and 1.008 to 1.887 for the energy.
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
        let sweep_ep = entropy_production(&uncoupled, beta, Kernel::SequentialGibbs).expect("dense");
        assert!(sweep_ep < 1e-12, "uncoupled, a sweep is a fresh draw and reversible, yet the ratio is 2 - 1/n: {sweep_ep:.3e}");

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
        // Measured 1.888 at beta = 1.
        assert!((1.8..2.0).contains(&coupled), "coupled ferromagnet, magnetisation: ratio {coupled}");
        // Colder it keeps falling, which "at every temperature" (first measured over beta 0.2 to 1.5)
        // did not say: 1.824 at beta = 3.
        let cold = |k: Kernel| tau_int_fundamental(&ferro, 3.0, k, mag).expect("dense").tau_int;
        let cold_ratio = cold(Kernel::RandomScan) / (cold(Kernel::SequentialGibbs) * ferro.n as f64);
        assert!((1.80..1.85).contains(&cold_ratio) && cold_ratio < coupled, "ferromagnet at beta 3: ratio {cold_ratio}");
    }

    /// THE LAG SUM MUST NOT STOP AT A ZERO CROSSING. Two spins coupled at `beta J = 1`, zero field,
    /// the synchronous sweep: each new spin copies the other old one with `E[s_0' | x] = tanh(beta J)
    /// s_1`, so `s_0 + s_1` and `s_0 - s_1` are eigenfunctions with eigenvalues `t` and `-t`
    /// (`t = tanh(beta J)`), Peretto's law is uniform, and `s_0`, their average, has
    /// `rho(k) = 0` at every odd lag and `t^k` at every even one: `tau = 1/2 + t^2 / (1 - t^2)`.
    /// A sum that stopped at its first quiet lag stopped at lag 1 and returned `1/2`.
    #[test]
    fn the_lag_sum_does_not_stop_where_the_autocorrelation_crosses_zero() {
        let mut b = GraphBuilder::new(2);
        b.couple(0, 1, 1.0);
        let g = b.build();
        // 0.5 + t^2 / (1 - t^2) at t = tanh(1), mpmath at 40 digits.
        let want = 1.8810978455418157;
        let a = tau_int_exact(&g, 1.0, Kernel::Synchronous, |s| f64::from(s[0]), 1e-12, 10_000).unwrap();
        assert!(a.rho[0].abs() < 1e-15, "rho(1) is zero: {}", a.rho[0]);
        assert!((a.rho[1] - 1.0f64.tanh().powi(2)).abs() < 1e-14, "rho(2) = t^2: {}", a.rho[1]);
        assert!((a.tau_int - want).abs() < 1e-12, "tau {} against the closed form {want}", a.tau_int);
        assert!(a.lags > QUIET_LAGS && a.lags < 10_000, "lags {}", a.lags);
    }

    /// Two spins with `E = -J s_0 s_1` under the sequential sweep: `s_1` alone is a Markov chain
    /// that keeps its value with probability `a^2 + c^2` (`a = sigma(2 beta J)`, `c = sigma(-2 beta J)`:
    /// site 0 copies the old `s_1` with probability `a`, then site 1 copies the new `s_0`), so
    /// `rho(k) = (a - c)^{2k}` and `tau = (1 - 2ac) / (4ac)` in closed form. The escape is `c`: a
    /// flip AGAINST a field of `J`, exactly the transition `1 - p_up` rounded to zero.
    fn cold_pair(two_beta_j: f64) -> Graph {
        let mut b = GraphBuilder::new(2);
        b.couple(0, 1, 0.5 * two_beta_j);
        b.build()
    }

    /// THE COLD CHAIN, EXACTLY, by the censored solve on two metastable sets that share nothing but
    /// the answer -- the two energy minima with the other two states eliminated by inner solves, and
    /// all four states with no inner solve at all -- at `2 beta J = 16, 30, 40, 100`, against the
    /// closed form evaluated by mpmath. At 40 and 100 the Krylov solve must REFUSE
    /// ([`AutocorrError::BeyondF64`], `tau = 5.9e16` and `6.7e42`), and [`tau_int_solved`] must fall
    /// through to the censored route. Under the old `1 - p_up` kernel the escape at 40 and 100 was
    /// exactly zero and the chain reducible; at 30 it was `1e-3` off.
    #[test]
    fn a_cold_pair_has_its_closed_form_tau_by_the_route_that_can_reach_it() {
        let exact = [
            (16.0, 2221527.6301269964),
            (30.0, 2671618645381.1157),
            (40.0, 5.8846316709255e16),
            (100.0, 6.720292854540339e42),
        ];
        let s1 = |s: &[i8]| f64::from(s[1]);
        for &(x, want) in &exact {
            let g = cold_pair(x);
            for set in [Metastable::LocalMinima, Metastable::LocalMinimaAndNeighbours] {
                let a = tau_int_censored(&g, 1.0, Kernel::SequentialGibbs, s1, set, 1e-13).unwrap();
                assert_eq!(a.route, Route::Censored);
                let rel = (a.tau_int - want).abs() / want;
                assert!(rel < 1e-13, "2 beta J = {x}, {set:?}: {:e} against {want:e}, rel {rel:e}", a.tau_int);
            }
            let krylov = tau_int_krylov(&g, 1.0, Kernel::SequentialGibbs, s1, 1e-8, 1_000);
            let solved = tau_int_solved(&g, 1.0, Kernel::SequentialGibbs, s1, 1e-8, 1_000).unwrap();
            assert!((solved.tau_int - want).abs() <= 1e-8 * want, "2 beta J = {x}: tau_int_solved {:e}", solved.tau_int);
            if x >= 40.0 {
                assert!(
                    matches!(krylov, Err(AutocorrError::BeyondF64 { inv_norm, .. }) if inv_norm * f64::EPSILON > 1.0),
                    "2 beta J = {x}: Krylov must refuse a chain past f64, got {krylov:?}"
                );
                assert_eq!(solved.route, Route::Censored);
            }
            if x == 16.0 {
                let k = krylov.expect("tau 2.2e6 is inside what f64 can certify at 1e-8");
                assert!((k.tau_int - want).abs() <= 1e-8 * want, "Krylov at 16: {:e}", k.tau_int);
                // A tolerance below what f64 can certify here (u ||A^-1|| is about 5e-10) is met or
                // refused, never claimed: the estimate's `u ||z||` term is what stops a residual
                // that is only rounding from reading as convergence.
                match tau_int_krylov(&g, 1.0, Kernel::SequentialGibbs, s1, 1e-12, 1_000) {
                    Ok(k) => assert!((k.tau_int - want).abs() <= 2e-12 * want, "claimed 1e-12, realised {:e}", (k.tau_int - want).abs() / want),
                    Err(AutocorrError::AtFloor { .. }) => {}
                    other => panic!("2 beta J = 16 at rtol 1e-12: {other:?}"),
                }
            }
        }
    }

    /// THE KRYLOV TAU IS THE DENSE TAU FOR EVERY KERNEL, at `n = 9`, warm and cold, CG where the
    /// kernel is reversible and GMRES elsewhere -- and each under its OWN law: the verifier of
    /// 2026-09-28 found GMRES handed the Boltzmann law converging on the fabric and on tick-random to a
    /// tau `6.2e-4` and `13.7%` off. The tolerance is what each route can attain: the Krylov estimate
    /// plus `16 u K (tau + 1/2)`, `K` Kemeny's constant, for the dense LU, which at beta 2 is the
    /// LESS accurate of the two (the random scan: dense `3.65e-9` from Krylov, estimate `6.5e-9`).
    #[test]
    fn the_krylov_tau_is_the_dense_tau_for_every_kernel() {
        let g = grid_glass(3, 3, 5);
        let every = [
            Kernel::ChromaticGibbs,
            Kernel::SequentialGibbs,
            Kernel::RandomScan,
            Kernel::Synchronous,
            Kernel::Pimi { xi: 0.25, eta: 0.8 },
            Kernel::Stale { p: 0.1 },
            Kernel::TickRandom { p: 0.5 },
            Kernel::Sca { q: 1.0 },
            Kernel::SiteSpread { seed: 3, spread: 0.2 },
            Kernel::FixedFabric,
            Kernel::Quantised { frac_bits: 8, lut_bits: 10, prob_bits: 20 },
            Kernel::Informed(Balance::Barker),
            Kernel::Informed(Balance::Sqrt),
            Kernel::Informed(Balance::Metropolis),
        ];
        for (beta, rtol) in [(1.0, 1e-10), (2.0, 1e-8)] {
            for kernel in every {
                let dense = tau_int_fundamental(&g, beta, kernel, |s| g.energy(s)).unwrap();
                let k = tau_int_krylov(&g, beta, kernel, |s| g.energy(s), rtol, 5_000)
                    .unwrap_or_else(|e| panic!("{kernel:?} at beta {beta}: {e}"));
                let want = if reversible(kernel, &g) { Route::Cg } else { Route::Gmres };
                assert_eq!(k.route, want, "{kernel:?}");
                let kemeny = kemeny_constant(&g, beta, kernel).unwrap();
                let tol = k.err_est.unwrap() + 16.0 * f64::EPSILON * kemeny * (dense.tau_int + 0.5);
                let diff = (k.tau_int - dense.tau_int).abs();
                assert!(
                    diff <= tol,
                    "{kernel:?} at beta {beta}: Krylov {:.12e} dense {:.12e}, diff {diff:.2e} past {tol:.2e}",
                    k.tau_int,
                    dense.tau_int
                );
                assert!(k.matvecs < 200, "{kernel:?} at beta {beta}: {} applications", k.matvecs);
            }
        }
    }

    /// Above the dense cap the Krylov tau is the lag sum's, on a warm 14-spin grid where the lag sum
    /// converges -- two independent routes, one of them summing nothing but `apply`. And the kernels
    /// whose law needs the dense solve are REFUSED there rather than handed the Boltzmann law.
    #[test]
    fn above_the_dense_cap_the_krylov_tau_is_the_lag_sum_and_a_dense_law_is_refused() {
        let g = grid_glass(7, 2, 11);
        let beta = 0.5;
        assert!(tau_int_fundamental(&g, beta, Kernel::ChromaticGibbs, |s| g.energy(s)).is_err());
        for kernel in [Kernel::ChromaticGibbs, Kernel::SequentialGibbs, Kernel::RandomScan, Kernel::Informed(Balance::Barker)] {
            let k = tau_int_krylov(&g, beta, kernel, |s| g.energy(s), 1e-12, 2_000).unwrap();
            let lag = tau_int_exact(&g, beta, kernel, |s| g.energy(s), 1e-15, 100_000).unwrap();
            assert!(lag.lags < 100_000, "{kernel:?}: the lag sum must converge to be a reference");
            let rel = (k.tau_int - lag.tau_int).abs() / lag.tau_int;
            assert!(rel < 1e-11, "{kernel:?}: Krylov {} lag sum {} ({} lags), rel {rel:e}", k.tau_int, lag.tau_int, lag.lags);
        }
        for kernel in [Kernel::FixedFabric, Kernel::Quantised { frac_bits: 8, lut_bits: 10, prob_bits: 20 }, Kernel::SiteSpread { seed: 1, spread: 0.1 }] {
            assert_eq!(
                tau_int_krylov(&g, beta, kernel, |s| g.energy(s), 1e-10, 100),
                Err(AutocorrError::TooManyForDense { n: 14, max: MAX_DENSE_SPINS }),
                "{kernel:?}"
            );
        }
        let big = grid_glass(5, 3, 11);
        for kernel in [Kernel::Synchronous, Kernel::TickRandom { p: 0.5 }] {
            assert_eq!(
                tau_int_krylov(&big, beta, kernel, |s| big.energy(s), 1e-10, 100),
                Err(AutocorrError::TooManySpins { n: 15, max: 14 }),
                "{kernel:?}"
            );
        }
    }

    /// The law check: the fabric's own law is invariant under the fabric's kernel to rounding, and
    /// the Boltzmann law is not -- by more than [`LAW_TOLERANCE`] -- so a law arm that said
    /// "Boltzmann" for the fabric would be refused before any solve.
    #[test]
    fn a_law_that_is_not_the_kernels_is_refused() {
        let g = grid_glass(3, 3, 5);
        for kernel in [Kernel::FixedFabric, Kernel::TickRandom { p: 0.5 }] {
            let own = stationary_solved(&g, 1.0, kernel).unwrap();
            assert!(check_law(&g, 1.0, kernel, own).is_ok(), "{kernel:?}'s own law");
            match check_law(&g, 1.0, kernel, boltzmann(&g, 1.0).unwrap()) {
                Err(AutocorrError::LawNotInvariant { residual }) => assert!(residual > 1e-6, "{kernel:?}: {residual:e}"),
                other => panic!("{kernel:?} handed Boltzmann: {other:?}"),
            }
        }
    }

    /// The stopping rule refuses to return a number it cannot certify: a tolerance below f64's floor
    /// for a cold chain is [`AutocorrError::AtFloor`], not `Ok`, and a budget too small is
    /// [`AutocorrError::NotConverged`] -- each carrying its iterate for the record.
    #[test]
    fn a_tolerance_f64_cannot_certify_is_refused_not_returned() {
        let g = grid_glass(3, 3, 5);
        for kernel in [Kernel::ChromaticGibbs, Kernel::RandomScan] {
            let floor = tau_int_krylov(&g, 2.0, kernel, |s| g.energy(s), 1e-14, 5_000);
            assert!(matches!(floor, Err(AutocorrError::AtFloor { .. })), "{kernel:?} at rtol 1e-14: {floor:?}");
            let short = tau_int_krylov(&g, 2.0, kernel, |s| g.energy(s), 1e-8, 4);
            assert!(matches!(short, Err(AutocorrError::NotConverged { .. })), "{kernel:?} with 4 applications: {short:?}");
        }
    }
    /// THE SLOW-SCALE MASS TIME IS THE PUSHED ONE WHERE BOTH RUN, and closes on it as the barrier
    /// grows. Two spins, `J` coupling and a field of `0.3` on each, the sequential sweep from the
    /// uniform law: the sweeps until the mass on `++` holds its stationary value to 1%. Pushed step by
    /// step it is exact; censored onto the two minima (and onto all four states, a set with no inner
    /// solve) and run in real time it is a reduction whose error goes as the fast relaxation over the
    /// barrier crossing -- measured `1.7e-4, 1.8e-5, 2.4e-6` at `J = 5, 6, 7` -- and at `J = 20`, where
    /// no push could reach it (`1.8e17` sweeps), the two sets agree.
    #[test]
    fn the_slow_scale_mass_time_is_the_pushed_one_where_both_run() {
        let pair = |j: f64| {
            let mut b = GraphBuilder::new(2);
            b.couple(0, 1, j);
            b.bias(0, 0.3);
            b.bias(1, 0.3);
            b.build()
        };
        let start = [0.25; 4];
        let mut last = f64::INFINITY;
        for (j, tol) in [(5.0, 1e-3), (6.0, 1e-4), (7.0, 1e-5)] {
            let g = pair(j);
            let pushed = time_to_mass(&g, 1.0, Kernel::SequentialGibbs, &start, |x| x == 3, 0.01, 10_000_000, Metastable::LocalMinima).unwrap();
            assert_eq!(pushed.route, MassRoute::Pushed);
            for set in [Metastable::LocalMinima, Metastable::LocalMinimaAndNeighbours] {
                let slow = time_to_mass(&g, 1.0, Kernel::SequentialGibbs, &start, |x| x == 3, 0.01, 50, set).unwrap();
                assert_eq!(slow.route, MassRoute::SlowScale);
                let rel = (slow.steps - pushed.steps).abs() / pushed.steps;
                assert!(rel < tol, "J {j}, {set:?}: slow {} against pushed {}, rel {rel:e}", slow.steps, pushed.steps);
                if set == Metastable::LocalMinima {
                    assert!(rel < last, "J {j}: the reduction must close on the pushed time as the barrier grows");
                    last = rel;
                }
            }
        }
        let g = pair(20.0);
        let a = time_to_mass(&g, 1.0, Kernel::SequentialGibbs, &start, |x| x == 3, 0.01, 50, Metastable::LocalMinima).unwrap();
        let b = time_to_mass(&g, 1.0, Kernel::SequentialGibbs, &start, |x| x == 3, 0.01, 50, Metastable::LocalMinimaAndNeighbours).unwrap();
        assert!(a.steps > 1e17 && (a.steps - b.steps).abs() < 1e-9 * a.steps, "J 20: {} and {}", a.steps, b.steps);
    }
    /// The tabled push the censored routes use is `apply_distribution` to the bit, for every sweep
    /// kernel it tables, and there is no table for a kernel that is not a sweep.
    #[test]
    fn a_tabled_push_is_apply_distribution_to_the_bit() {
        let g = grid_glass(3, 3, 5);
        let mut rng = Pcg::new(9, 1);
        let mu: Vec<f64> = (0..512).map(|_| rng.f64()).collect();
        for kernel in [
            Kernel::SequentialGibbs,
            Kernel::ChromaticGibbs,
            Kernel::FixedFabric,
            Kernel::SiteSpread { seed: 3, spread: 0.2 },
        ] {
            let t = SweepTable::new(&g, 1.3, kernel).expect("a sweep kernel is tabled");
            let (a, b) = (t.push(&mu), apply_distribution(&g, 1.3, kernel, &mu));
            assert!(a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()), "{kernel:?}");
        }
        for kernel in [Kernel::RandomScan, Kernel::Synchronous, Kernel::Informed(Balance::Barker), Kernel::Stale { p: 0.1 }] {
            assert!(SweepTable::new(&g, 1.3, kernel).is_none(), "{kernel:?}");
        }
    }
}
