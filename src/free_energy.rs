//! Free energy — the number a sampler owes, with the bound it can prove.
//!
//! Every certificate in this crate so far says the chain *mixed*: β came out as requested, draws
//! decorrelated, marginals sat inside the noise floor. None of them says what the distribution
//! *is*. The quantity that does is the partition function `Z(β) = Σ_s exp(−β E(s))` — its logarithm
//! turns sampled statistics into normalised probabilities, gives an energy-based model a
//! likelihood instead of a marginal error, and is the one number every thermodynamic-computing
//! paper quotes and no sampling stack certifies. This module computes it three ways, each with the
//! guarantee it actually carries, and checks all three against exact oracles before any is trusted.
//!
//! # What is exact
//!
//! * [`exact_log_z`] enumerates `2^n` states (`n ≤ 24`).
//! * [`crate::exact::Elimination::log_partition`] is exact by variable elimination far past that,
//!   bounded by treewidth rather than by `n`.
//! * [`ring_log_z`] is the periodic chain in closed form, by transfer matrix.
//! * [`onsager_log_z_density`] is the infinite square lattice in closed form (Onsager 1944), the
//!   same oracle the sampler was first verified against.
//!
//! # What is bounded, and by which theorem
//!
//! **Annealed importance sampling** ([`ais`], Neal 2001) walks a ladder `0 = β_0 < … < β_K = β`
//! from the uniform distribution, whose `ln Z_0 = n ln 2` is known, accumulating importance
//! weights. Its estimator of `Z` is **unbiased for any transition kernels that leave each rung
//! invariant, however few sweeps they run** — mixing affects the variance, never the expectation.
//! So Markov's inequality gives, with no equilibrium assumption at all,
//!
//! ```text
//!   P( Ẑ ≥ Z e^t ) ≤ e^{−t}      ⇒      ln Z ≥ ln Ẑ − t   with probability ≥ 1 − e^{−t}.
//! ```
//!
//! That is [`Ais::lower_bound`]: an unconditional high-probability **lower** bound on `ln Z`.
//!
//! **Reverse AIS** ([`reverse_ais`], Burda–Grosse–Salakhutdinov 2015) runs the ladder downward
//! from a sample of the target and gives the mirror-image **upper** bound — *conditional* on that
//! starting sample really being from the target. In the tests it is (drawn from the enumerated
//! distribution), so the sandwich there is unconditional; in use it is as good as the chain that
//! produced the sample, and the result says so.
//!
//! **Bennett's acceptance ratio** ([`bar_pair`], [`bar_ladder`]) steps `ln Z` up a ladder from the
//! exact anchor `ln Z(0) = n ln 2` using samples at *both* ends of every step — the minimum-variance
//! two-sample estimator — and with `ln Z` at every rung the rest of thermodynamics follows:
//! entropy `S = ln Z + β⟨E⟩` and heat capacity `β² Var(E)`, per rung, from the same chains
//! ([`thermodynamics`]). Its error is a standard error, not a bound; it is the precise estimate to
//! sit beside the two bounds. [`ais_clamped`] is AIS with sites held fixed — the numerator of an
//! energy-based model's likelihood when the hidden part is too large to enumerate.
//!
//! **Thermodynamic integration** ([`thermodynamic_integration`]) uses
//! `ln Z(β) = n ln 2 − ∫_0^β ⟨E⟩_{β'} dβ'` and one fact: `d⟨E⟩/dβ = −Var(E) ≤ 0`, so `⟨E⟩` is
//! non-increasing in `β` and the left and right Riemann sums *bracket* the integral. With the means
//! widened by their own error bars, that is a two-sided bound whose only assumption is that each
//! rung's mean was measured at equilibrium — and each mean arrives as an [`Estimate`] carrying its
//! `tau_int`, so the reader can see how far to trust that.
//!
//! # Which bound to buy
//!
//! Measured in `examples/free_energy` on a 16-spin ring: the AIS sandwich at 99% is ±4.6 nats
//! wide, because `ln(1/0.01)` is Markov's slack and Markov assumes nothing; the TI bracket on the
//! same model is ±0.5, nine times tighter, and assumes every rung's chain equilibrated. Both are
//! reported with their assumption attached. Neither is the point estimate, which all four routes
//! (these three and `popanneal`) put within 0.02 of the truth there.
//!
//! # The kernel, and why it is palindromic
//!
//! Forward AIS needs each rung's kernel to leave that rung invariant; a chromatic sweep does.
//! Reverse AIS needs the kernel to be *reversible* (self-adjoint) as well, and a fixed-order sweep
//! is not. Sweeping the colour classes forward and then backward is: the composition `T T*` is
//! self-adjoint for any `T`. Both directions use it, so the two estimators are exact mirrors.
//!
//! # Rounding
//!
//! Bounds are published after one step of outward rounding ([`next_down`], [`next_up`]), so the
//! floating-point arithmetic that produced them cannot have narrowed the interval it reports. The
//! `proofs` module proves those steps do what they say for every finite `f64`.

use crate::certify::tau_int;
use crate::graph::Graph;
use crate::kernel::draw;
use crate::rng::Pcg;
use crate::samples::Estimate;

// ---- exact oracles ---------------------------------------------------------------------------

/// `ln Z(β)` by enumeration. Panics past 24 spins; use [`crate::exact::Elimination`] there.
pub fn exact_log_z(g: &Graph, beta: f64) -> f64 {
    assert!(g.n <= 24, "exact enumeration limited to 24 spins");
    let m = 1usize << g.n;
    let mut s = vec![-1i8; g.n];
    let mut logs = Vec::with_capacity(m);
    for mask in 0..m {
        for b in 0..g.n {
            s[b] = if mask >> b & 1 == 1 { 1 } else { -1 };
        }
        logs.push(-beta * g.energy(&s));
    }
    log_sum_exp(&logs)
}

/// `ln Z` of the periodic chain `E = −J Σ s_i s_{i+1} − h Σ s_i` on `n ≥ 3` sites, in closed form.
///
/// Transfer matrix `T = [[e^{β(J+h)}, e^{−βJ}], [e^{−βJ}, e^{β(J−h)}]]`, `Z = Tr Tⁿ = λ₊ⁿ + λ₋ⁿ`
/// with `λ± = e^{βJ} cosh βh ± √(e^{2βJ} sinh² βh + e^{−2βJ})`. This is what
/// [`crate::ising::ring`] builds, and the enumeration test pins the sign conventions.
pub fn ring_log_z(n: usize, j: f64, h: f64, beta: f64) -> f64 {
    assert!(n >= 3, "a ring needs three sites to have distinct edges");
    let (bj, bh) = (beta * j, beta * h);
    let root = (bj.exp() * bj.exp() * bh.sinh() * bh.sinh() + (-2.0 * bj).exp()).sqrt();
    let lp = bj.exp() * bh.cosh() + root;
    let lm = bj.exp() * bh.cosh() - root;
    // ln(λ₊ⁿ + λ₋ⁿ) = n ln λ₊ + ln(1 + (λ₋/λ₊)ⁿ); λ₋ may be negative, so raise the ratio as a
    // signed power rather than through logarithms.
    let ratio = (lm / lp).powi(n as i32);
    n as f64 * lp.ln() + (1.0 + ratio).ln()
}

/// `ln Z / N` of the infinite square lattice with coupling `J` and no field (Onsager 1944):
///
/// ```text
///   ln 2 + 1/(8π²) ∫₀^{2π}∫₀^{2π} ln[ cosh²(2K) − sinh(2K)(cos θ₁ + cos θ₂) ] dθ₁ dθ₂,   K = βJ.
/// ```
///
/// Evaluated by a periodic midpoint rule, which converges exponentially away from `K_c`; the
/// integrand has a logarithmic singularity at criticality, so a value near `K_c ≈ 0.4407` is
/// accurate only to the grid. `grid` points per axis; 512 is enough for `1e-9` at `K = 0.3`.
pub fn onsager_log_z_density(beta: f64, j: f64, grid: usize) -> f64 {
    let k = beta * j;
    let (c2, s2) = ((2.0 * k).cosh(), (2.0 * k).sinh());
    let step = core::f64::consts::TAU / grid as f64;
    let mut acc = 0.0;
    for a in 0..grid {
        let t1 = (a as f64 + 0.5) * step;
        for b in 0..grid {
            let t2 = (b as f64 + 0.5) * step;
            acc += (c2 * c2 - s2 * (t1.cos() + t2.cos())).ln();
        }
    }
    core::f64::consts::LN_2 + acc * step * step / (8.0 * core::f64::consts::PI * core::f64::consts::PI)
}

// ---- the kernel ------------------------------------------------------------------------------

/// One palindromic chromatic sweep at `beta`: every colour class forward, then every class back.
///
/// Self-adjoint with respect to the Boltzmann distribution at `beta`, which forward AIS does not
/// need and reverse AIS does. Costs two ordinary sweeps.
pub fn palindromic_sweep(g: &Graph, beta: f64, s: &mut [i8], rng: &mut Pcg) {
    for class in g.classes.iter().chain(g.classes.iter().rev()) {
        for &i in class {
            let i = i as usize;
            s[i] = draw(g.field(i, s), beta, rng);
        }
    }
}

/// [`palindromic_sweep`] with the sites where `free[i]` is false held fixed.
pub fn palindromic_sweep_masked(g: &Graph, beta: f64, s: &mut [i8], rng: &mut Pcg, free: &[bool]) {
    for class in g.classes.iter().chain(g.classes.iter().rev()) {
        for &i in class {
            let i = i as usize;
            if free[i] {
                s[i] = draw(g.field(i, s), beta, rng);
            }
        }
    }
}

/// Forward AIS with some sites clamped: `ln Z` of the conditional model, where the reference at
/// `β = 0` is uniform over the FREE sites only, `ln Z_0 = free · ln 2`.
///
/// This is the numerator of an energy-based model's likelihood, `ln Σ_h exp(−E(v, h))`, for a
/// hidden part too large to enumerate. The bound is the same Markov bound as [`ais`], with the
/// same unconditional standing. `n` in the result counts the free sites.
pub fn ais_clamped(g: &Graph, fixed: &[(usize, i8)], ladder: &[f64], sweeps: usize, runs: usize, seed: u64) -> Ais {
    check_ladder(ladder);
    assert!(runs >= 1);
    let mut free = vec![true; g.n];
    for &(i, v) in fixed {
        assert!(i < g.n && (v == 1 || v == -1), "clamp {i} to {v} is not a site and a spin");
        free[i] = false;
    }
    let n_free = free.iter().filter(|&&f| f).count();
    let mut log_weights = Vec::with_capacity(runs);
    for r in 0..runs {
        let mut rng = Pcg::new(seed, r as u64);
        let mut s = uniform_state(g.n, &mut rng);
        for &(i, v) in fixed {
            s[i] = v;
        }
        let mut lw = 0.0;
        for k in 1..ladder.len() {
            lw += -(ladder[k] - ladder[k - 1]) * g.energy(&s);
            for _ in 0..sweeps {
                palindromic_sweep_masked(g, ladder[k], &mut s, &mut rng, &free);
            }
        }
        log_weights.push(lw);
    }
    finish_ais(n_free, *ladder.last().unwrap(), log_weights)
}

fn uniform_state(n: usize, rng: &mut Pcg) -> Vec<i8> {
    (0..n).map(|_| if rng.f64() < 0.5 { -1 } else { 1 }).collect()
}

/// A linear ladder `0, β/K, 2β/K, …, β` with `rungs = K + 1` rungs.
///
/// Zero must be the first rung for the uniform reference to apply; the rest is the caller's to
/// shape, and linear is the default because the alternative was measured. A geometric ladder
/// (`β/2^{K−1}, …, β/2, β`) puts its largest step last, where it *doubles* `β`; on a 4×4 lattice at
/// `β = 0.6` that step alone drove the weights' effective sample size to about one, and the point
/// estimates landed a nat from the truth in both directions while the bounds, being bounds, still
/// held. The variance of AIS is governed by the largest `Δβ · spread(E)` on the ladder, so equal
/// steps are the right first guess and more of them is the right second one.
pub fn linear_ladder(beta: f64, rungs: usize) -> Vec<f64> {
    assert!(rungs >= 2 && beta > 0.0);
    let k = (rungs - 1) as f64;
    (0..rungs).map(|i| if i + 1 == rungs { beta } else { beta * i as f64 / k }).collect()
}

fn check_ladder(ladder: &[f64]) {
    assert!(ladder.len() >= 2, "a ladder needs at least the reference rung and the target");
    assert!(ladder[0] == 0.0, "the ladder must start at beta = 0, where ln Z = n ln 2 is known");
    for w in ladder.windows(2) {
        assert!(w[1] > w[0], "the ladder must increase strictly");
    }
}

fn log_sum_exp(v: &[f64]) -> f64 {
    let mx = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if mx == f64::NEG_INFINITY {
        return mx;
    }
    mx + v.iter().map(|x| (x - mx).exp()).sum::<f64>().ln()
}

// ---- annealed importance sampling ------------------------------------------------------------

/// What a forward AIS run produced.
#[derive(Clone, Debug)]
pub struct Ais {
    /// Number of spins, so `ln Z_0 = n ln 2` is on record.
    pub n: usize,
    /// The target `β`, the last rung.
    pub beta: f64,
    /// One log importance weight per run, *excluding* the `n ln 2` reference term.
    pub log_weights: Vec<f64>,
    /// The point estimate `ln Ẑ = n ln 2 + ln mean(w)`. Biased low for `ln Z` (Jensen), unbiased
    /// for `Z`.
    pub log_z: f64,
    /// Effective sample size of the weights, `(Σw)² / Σw²`. Near 1 means one run dominates and
    /// the lower bound, while still valid, is loose.
    pub ess: f64,
}

impl Ais {
    /// `ln Z ≥ lower_bound(delta)` with probability at least `1 − delta`, unconditionally.
    ///
    /// Markov's inequality on the unbiased estimator: `P(Ẑ ≥ Z e^t) ≤ e^{−t}`, so with `t = ln(1/δ)`
    /// the event `ln Z < ln Ẑ − t` has probability at most `δ`. Rounded outward.
    pub fn lower_bound(&self, delta: f64) -> f64 {
        assert!(delta > 0.0 && delta < 1.0);
        next_down(self.log_z - (1.0 / delta).ln())
    }
}

/// Forward annealed importance sampling: `runs` independent walks up `ladder`, `sweeps` palindromic
/// sweeps at every rung above zero. `ladder[0]` must be `0`.
pub fn ais(g: &Graph, ladder: &[f64], sweeps: usize, runs: usize, seed: u64) -> Ais {
    check_ladder(ladder);
    assert!(runs >= 1);
    let mut log_weights = Vec::with_capacity(runs);
    for r in 0..runs {
        let mut rng = Pcg::new(seed, r as u64);
        let mut s = uniform_state(g.n, &mut rng);
        let mut lw = 0.0;
        for k in 1..ladder.len() {
            // weight f_k(s)/f_{k-1}(s) at the state drawn from rung k-1, then move under rung k.
            lw += -(ladder[k] - ladder[k - 1]) * g.energy(&s);
            for _ in 0..sweeps {
                palindromic_sweep(g, ladder[k], &mut s, &mut rng);
            }
        }
        log_weights.push(lw);
    }
    finish_ais(g.n, *ladder.last().unwrap(), log_weights)
}

fn finish_ais(n: usize, beta: f64, log_weights: Vec<f64>) -> Ais {
    let lse = log_sum_exp(&log_weights);
    let log_z = n as f64 * core::f64::consts::LN_2 + lse - (log_weights.len() as f64).ln();
    let lse2 = log_sum_exp(&log_weights.iter().map(|w| 2.0 * w).collect::<Vec<_>>());
    let ess = (2.0 * lse - lse2).exp();
    Ais { n, beta, log_weights, log_z, ess }
}

/// What a reverse AIS run produced.
#[derive(Clone, Debug)]
pub struct ReverseAis {
    pub n: usize,
    pub beta: f64,
    /// One log weight per run for `Z_0 / Z`, excluding the reference term.
    pub log_weights: Vec<f64>,
    /// The point estimate `ln Ẑ = n ln 2 − ln mean(w')`. Biased *high* for `ln Z`.
    pub log_z: f64,
    pub ess: f64,
}

impl ReverseAis {
    /// `ln Z ≤ upper_bound(delta)` with probability at least `1 − delta`, **conditional on the
    /// starting states having been exact draws from the target**. Rounded outward.
    pub fn upper_bound(&self, delta: f64) -> f64 {
        assert!(delta > 0.0 && delta < 1.0);
        next_up(self.log_z + (1.0 / delta).ln())
    }
}

/// Reverse annealed importance sampling from `starts`, one walk down `ladder` per start.
///
/// Each start must be a draw from the Boltzmann distribution at `ladder.last()` for the bound to
/// hold; the caller vouches for that, and should say how (an enumerated draw, or a chain whose
/// `tau_int` it reports beside the result).
pub fn reverse_ais(g: &Graph, ladder: &[f64], sweeps: usize, starts: &[Vec<i8>], seed: u64) -> ReverseAis {
    check_ladder(ladder);
    assert!(!starts.is_empty());
    let mut log_weights = Vec::with_capacity(starts.len());
    for (r, start) in starts.iter().enumerate() {
        assert_eq!(start.len(), g.n);
        let mut rng = Pcg::new(seed, r as u64);
        let mut s = start.clone();
        let mut lw = 0.0;
        for k in (1..ladder.len()).rev() {
            // move under rung k (self-adjoint), then weigh f_{k-1}(s)/f_k(s) at the state reached.
            for _ in 0..sweeps {
                palindromic_sweep(g, ladder[k], &mut s, &mut rng);
            }
            lw += (ladder[k] - ladder[k - 1]) * g.energy(&s);
        }
        log_weights.push(lw);
    }
    let a = finish_ais(g.n, *ladder.last().unwrap(), log_weights);
    // mean(w') estimates Z_0 / Z, so ln Z = n ln 2 − ln mean(w') = 2·n ln 2 − (n ln 2 + ln mean).
    let log_z = 2.0 * g.n as f64 * core::f64::consts::LN_2 - a.log_z;
    ReverseAis { n: g.n, beta: a.beta, log_weights: a.log_weights, log_z, ess: a.ess }
}

/// A two-sided high-probability bound on `ln Z`, from a forward and a reverse run.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sandwich {
    pub lower: f64,
    pub upper: f64,
    /// Probability that BOTH bounds hold: `1 − 2δ` by a union bound.
    pub confidence: f64,
}

impl Sandwich {
    /// Combine at per-side risk `delta`. The upper side inherits reverse AIS's condition.
    pub fn new(fwd: &Ais, rev: &ReverseAis, delta: f64) -> Sandwich {
        assert_eq!(fwd.n, rev.n);
        Sandwich { lower: fwd.lower_bound(delta), upper: rev.upper_bound(delta), confidence: 1.0 - 2.0 * delta }
    }

    pub fn contains(&self, log_z: f64) -> bool {
        self.lower <= log_z && log_z <= self.upper
    }

    pub fn width(&self) -> f64 {
        self.upper - self.lower
    }
}

// ---- thermodynamic integration ---------------------------------------------------------------

/// What thermodynamic integration produced: the bracket and every rung's mean energy.
#[derive(Clone, Debug)]
pub struct Ti {
    pub n: usize,
    pub beta: f64,
    /// `(β_k, ⟨E⟩_{β_k})` for every rung; the zero rung's mean is exactly `0`.
    pub rungs: Vec<(f64, Estimate)>,
    /// The bracket on `ln Z` from monotonicity alone, means taken at face value.
    pub lower: f64,
    pub upper: f64,
    /// The bracket with every mean widened by `z` standard errors first.
    pub lower_widened: f64,
    pub upper_widened: f64,
    pub z: f64,
}

impl Ti {
    pub fn midpoint(&self) -> f64 {
        0.5 * (self.lower + self.upper)
    }
}

/// Thermodynamic integration up `ladder`: at every rung above zero, a chromatic chain of
/// `burn_in + draws` palindromic sweeps measures `⟨E⟩` with its own error bar; the bracket follows
/// from `⟨E⟩` being non-increasing in `β`. `z` is how many standard errors widen each mean.
pub fn thermodynamic_integration(g: &Graph, ladder: &[f64], burn_in: usize, draws: usize, z: f64, seed: u64) -> Ti {
    check_ladder(ladder);
    assert!(draws >= 4);
    let traces = sample_ladder_energies(g, ladder, burn_in, draws, seed);
    let rungs: Vec<(f64, Estimate)> = traces.iter().map(|(b, e)| (*b, estimate(e))).collect();
    ti_from_rungs(g.n, rungs, z)
}

/// The energies of `draws` states at every rung: exact uniform draws at `β = 0`, and a chromatic
/// chain of `burn_in + draws` palindromic sweeps at every rung above it. Shared by TI and BAR, so
/// the two estimators can be compared on the SAME samples.
pub fn sample_ladder_energies(g: &Graph, ladder: &[f64], burn_in: usize, draws: usize, seed: u64) -> Vec<(f64, Vec<f64>)> {
    check_ladder(ladder);
    assert!(draws >= 4);
    let mut out = Vec::with_capacity(ladder.len());
    let mut rng0 = Pcg::new(seed, 0);
    out.push((0.0, (0..draws).map(|_| g.energy(&uniform_state(g.n, &mut rng0))).collect()));
    for (k, &beta) in ladder.iter().enumerate().skip(1) {
        let mut rng = Pcg::new(seed, k as u64);
        let mut s = uniform_state(g.n, &mut rng);
        for _ in 0..burn_in {
            palindromic_sweep(g, beta, &mut s, &mut rng);
        }
        let mut trace = Vec::with_capacity(draws);
        for _ in 0..draws {
            palindromic_sweep(g, beta, &mut s, &mut rng);
            trace.push(g.energy(&s));
        }
        out.push((beta, trace));
    }
    out
}

/// The TI bracket from per-rung mean energies. The zero rung's mean is set to its exact value `0`
/// (uniform spins are uncorrelated), and the anchor `n ln 2` is exact.
pub fn ti_from_rungs(n: usize, mut rungs: Vec<(f64, Estimate)>, z: f64) -> Ti {
    rungs[0].1 = Estimate { value: 0.0, stderr: 0.0, ess: f64::INFINITY, tau_int: 0.0 };
    // Left sum uses the mean at the lower end of each interval (the larger value), right sum the
    // upper end; ⟨E⟩ non-increasing ⇒ left ≥ ∫ ≥ right ⇒ n ln 2 − left ≤ ln Z ≤ n ln 2 − right.
    let n_ln2 = n as f64 * core::f64::consts::LN_2;
    let (mut left, mut right, mut left_w, mut right_w) = (0.0, 0.0, 0.0, 0.0);
    for w in rungs.windows(2) {
        let d = w[1].0 - w[0].0;
        left += w[0].1.value * d;
        right += w[1].1.value * d;
        left_w += (w[0].1.value + z * w[0].1.stderr) * d;
        right_w += (w[1].1.value - z * w[1].1.stderr) * d;
    }
    Ti {
        n,
        beta: rungs.last().unwrap().0,
        rungs,
        lower: next_down(n_ln2 - left),
        upper: next_up(n_ln2 - right),
        lower_widened: next_down(n_ln2 - left_w),
        upper_widened: next_up(n_ln2 - right_w),
        z,
    }
}

/// Mean, `tau_int`-aware effective size and standard error of a trace.
pub(crate) fn estimate_trace(trace: &[f64]) -> Estimate {
    estimate(trace)
}

fn estimate(trace: &[f64]) -> Estimate {
    let n = trace.len() as f64;
    let mean = trace.iter().sum::<f64>() / n;
    let var = trace.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / (n - 1.0);
    let tau = tau_int(trace);
    let ess = n / (2.0 * tau);
    Estimate { value: mean, stderr: (var / ess).sqrt(), ess, tau_int: tau }
}

// ---- Bennett acceptance ratio ----------------------------------------------------------------

/// `ln(Z_b / Z_a)` between two rungs from samples of both, by the minimum-variance estimator.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BarPair {
    pub beta_a: f64,
    pub beta_b: f64,
    /// `ln(Z_b / Z_a)`.
    pub delta: f64,
    /// Bennett's asymptotic standard error, with each side's sample count replaced by its
    /// effective size under autocorrelation.
    pub stderr: f64,
    pub ess_a: f64,
    pub ess_b: f64,
}

fn fermi(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Bennett's acceptance-ratio estimate (1976) between rung `a` and rung `b`, from the ENERGIES of
/// samples drawn at each.
///
/// With `ℓ(E) = −(β_b − β_a) E` the log-ratio of unnormalised densities and `M = ln(N_a/N_b)`, the
/// estimate `Δ = ln(Z_b/Z_a)` is the unique root of
///
/// ```text
///   Σ_{i∈a} f(ℓ_i − Δ − M)  =  Σ_{j∈b} f(Δ + M − ℓ_j),      f(x) = 1 / (1 + e^{−x}),
/// ```
///
/// whose left side falls and right side rises in `Δ`, so bisection finds it to `1e-10`. Among all
/// estimators built from the two samples it has minimum asymptotic variance (Bennett; Shirts et
/// al. 2003), which is why it beats one-directional exponential averaging when the rungs overlap.
/// The standard error is Bennett's, with `N` replaced by `N / 2τ_int` of each side's `ℓ` trace.
pub fn bar_pair(beta_a: f64, energies_a: &[f64], beta_b: f64, energies_b: &[f64]) -> BarPair {
    assert!(beta_b > beta_a && energies_a.len() >= 2 && energies_b.len() >= 2);
    let d = beta_b - beta_a;
    let la: Vec<f64> = energies_a.iter().map(|e| -d * e).collect();
    let lb: Vec<f64> = energies_b.iter().map(|e| -d * e).collect();
    let (na, nb) = (la.len() as f64, lb.len() as f64);
    let m = (na / nb).ln();
    let resid = |delta: f64| -> f64 {
        la.iter().map(|&l| fermi(l - delta - m)).sum::<f64>() - lb.iter().map(|&l| fermi(delta + m - l)).sum::<f64>()
    };
    // Bracket the root: the residual is positive far left and negative far right.
    let spread = la.iter().chain(lb.iter()).fold(0.0f64, |acc, &l| acc.max(l.abs())) + m.abs() + 1.0;
    let (mut lo, mut hi) = (-spread, spread);
    while resid(lo) < 0.0 {
        lo *= 2.0;
    }
    while resid(hi) > 0.0 {
        hi *= 2.0;
    }
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if resid(mid) > 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
        if hi - lo < 1e-10 {
            break;
        }
    }
    let delta = 0.5 * (lo + hi);
    // Bennett's variance, per side: (⟨f²⟩/⟨f⟩² − 1) / N_eff.
    let c = delta + m;
    let side = |ls: &[f64], sign: f64| -> f64 {
        let fs: Vec<f64> = ls.iter().map(|&l| fermi(sign * (l - c))).collect();
        let mean = fs.iter().sum::<f64>() / fs.len() as f64;
        let mean2 = fs.iter().map(|f| f * f).sum::<f64>() / fs.len() as f64;
        mean2 / (mean * mean) - 1.0
    };
    let ess_a = na / (2.0 * tau_int(&la));
    let ess_b = nb / (2.0 * tau_int(&lb));
    let var = side(&la, 1.0) / ess_a + side(&lb, -1.0) / ess_b;
    BarPair { beta_a, beta_b, delta, stderr: var.max(0.0).sqrt(), ess_a, ess_b }
}

/// One rung of the free-energy curve.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ThermoRung {
    pub beta: f64,
    /// Absolute `ln Z(β)`: `n ln 2` plus the BAR steps up the ladder.
    pub log_z: f64,
    /// Standard error of `log_z`, the BAR steps added in quadrature.
    ///
    /// **This bar is optimistic, and by a measured amount.** Adjacent steps share the samples at
    /// their common rung, so they are correlated and the quadrature sum ignores that covariance;
    /// on a 12-spin ring over 40 seeds it understates the true spread by about 30%
    /// (`sd(z) = 1.28` where a calibrated bar gives 1). When the traces are in hand,
    /// [`crate::tempering::LadderTraces::log_z_total`] block-jackknifes the telescoped total
    /// instead and comes out conservative rather than optimistic. This field was documented as
    /// "an approximation" for one release before anyone measured which direction it erred.
    pub stderr: f64,
    /// `⟨E⟩` at this rung, with its error bar.
    pub mean_energy: Estimate,
    /// Entropy `S = ln Z + β⟨E⟩`, in nats. Exactly `n ln 2` at the zero rung.
    pub entropy: f64,
    /// Heat capacity `C = β² Var(E)` from the sample variance.
    pub heat_capacity: f64,
}

/// The free-energy curve and what follows from it.
#[derive(Clone, Debug)]
pub struct Thermo {
    pub n: usize,
    pub rungs: Vec<ThermoRung>,
    /// The TI bracket from the same samples, for comparison against the BAR curve's top rung.
    pub ti: Ti,
}

impl Thermo {
    pub fn top(&self) -> &ThermoRung {
        self.rungs.last().unwrap()
    }
}

/// `ln Z` at every rung by BAR steps from the exact anchor `ln Z(0) = n ln 2`, given the
/// energies of samples at each rung (the zero rung's must be exact uniform draws, which
/// [`sample_ladder_energies`] provides). Also the TI bracket from the same means.
pub fn bar_ladder(n: usize, traces: &[(f64, Vec<f64>)], z: f64) -> Thermo {
    assert!(traces.len() >= 2 && traces[0].0 == 0.0, "the curve is anchored at beta = 0");
    let mut log_z = n as f64 * core::f64::consts::LN_2;
    let mut var = 0.0;
    let mut rungs = Vec::with_capacity(traces.len());
    let mut means = Vec::with_capacity(traces.len());
    for (k, (beta, e)) in traces.iter().enumerate() {
        if k > 0 {
            let step = bar_pair(traces[k - 1].0, &traces[k - 1].1, *beta, e);
            log_z += step.delta;
            var += step.stderr * step.stderr;
        }
        let m = estimate(e);
        let nn = e.len() as f64;
        let v = e.iter().map(|x| (x - m.value) * (x - m.value)).sum::<f64>() / (nn - 1.0);
        rungs.push(ThermoRung {
            beta: *beta,
            log_z,
            stderr: var.sqrt(),
            mean_energy: m,
            entropy: log_z + beta * m.value,
            heat_capacity: beta * beta * v,
        });
        means.push((*beta, m));
    }
    Thermo { n, rungs, ti: ti_from_rungs(n, means, z) }
}

/// Draw the chains and build the curve: [`sample_ladder_energies`] then [`bar_ladder`].
pub fn thermodynamics(g: &Graph, ladder: &[f64], burn_in: usize, draws: usize, z: f64, seed: u64) -> Thermo {
    bar_ladder(g.n, &sample_ladder_energies(g, ladder, burn_in, draws, seed), z)
}

// ---- outward rounding --------------------------------------------------------------------------

/// The largest `f64` strictly below `x` (for finite `x`); `x` itself for NaN and `−∞`.
pub fn next_down(x: f64) -> f64 {
    if x.is_nan() || x == f64::NEG_INFINITY {
        return x;
    }
    if x == f64::INFINITY {
        return f64::MAX;
    }
    if x == 0.0 {
        return -f64::from_bits(1);
    }
    if x > 0.0 {
        f64::from_bits(x.to_bits() - 1)
    } else {
        f64::from_bits(x.to_bits() + 1)
    }
}

/// The smallest `f64` strictly above `x` (for finite `x`); `x` itself for NaN and `+∞`.
pub fn next_up(x: f64) -> f64 {
    if x.is_nan() || x == f64::INFINITY {
        return x;
    }
    -next_down(-x)
}

// ---- machine-checked ---------------------------------------------------------------------------
//
// The bounds above are theorems about probability; what the machine can check is the arithmetic
// they are published through. Compiled only under `cfg(kani)`; run by scripts/check-proofs.sh.
#[cfg(kani)]
mod proofs {
    use super::{next_down, next_up};

    /// Outward rounding never narrows: for every finite `x`, `next_down(x) < x < next_up(x)`.
    ///
    /// A bound reported through these cannot be tighter than the arithmetic that produced it,
    /// which is the whole reason they are applied. Exhaustive over the finite doubles.
    #[kani::proof]
    fn outward_rounding_strictly_widens() {
        let x: f64 = kani::any();
        kani::assume(x.is_finite());
        assert!(next_down(x) < x);
        assert!(next_up(x) > x);
        // and by exactly one ulp: nothing lies strictly between.
        assert!(next_up(next_down(x)) == x);
        assert!(next_down(next_up(x)) == x);
    }
}

// ---- tests -------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exact::Elimination;
    use crate::ising;

    /// Two exact routes and one closed form agree, which pins every sign convention at once.
    #[test]
    fn exact_routes_and_the_transfer_matrix_agree() {
        for &(n, j, h, beta) in &[(8usize, 1.0, 0.0, 0.7), (9, -0.8, 0.3, 1.3), (12, 0.5, -0.2, 2.0)] {
            let g = ising::ring(n, j, h);
            let enumerated = exact_log_z(&g, beta);
            let closed = ring_log_z(n, j, h, beta);
            let eliminated = Elimination::default().log_partition(&g, beta).unwrap().log_z.unwrap();
            assert!((enumerated - closed).abs() < 1e-9, "n={n}: enumeration {enumerated} vs transfer matrix {closed}");
            assert!((enumerated - eliminated).abs() < 1e-9, "n={n}: enumeration {enumerated} vs elimination {eliminated}");
        }
    }

    /// Onsager's density has the limits it must: `ln 2` at infinite temperature, `2K` at zero.
    #[test]
    fn onsager_density_has_the_right_limits() {
        assert!((onsager_log_z_density(0.0, 1.0, 64) - core::f64::consts::LN_2).abs() < 1e-12);
        let k = 3.0;
        let f = onsager_log_z_density(k, 1.0, 256);
        assert!((f - 2.0 * k).abs() < 1e-3, "at K=3 the density is 2K up to e^{{-4K}} terms: {f}");
        let (a, b) = (onsager_log_z_density(0.2, 1.0, 128), onsager_log_z_density(0.3, 1.0, 128));
        assert!(b > a, "ln Z / N increases with beta");
    }

    /// The forward bound holds at every seed, and the estimate lands near the truth.
    #[test]
    fn ais_lower_bound_holds_and_the_estimate_is_close() {
        let g = ising::lattice2d(4, 1.0);
        let beta = 0.6;
        let truth = exact_log_z(&g, beta);
        let ladder = linear_ladder(beta, 64);
        for seed in 0..20u64 {
            let a = ais(&g, &ladder, 2, 64, seed);
            // delta = 1e-6 makes a violation a one-in-a-million event; twenty seeds then cannot
            // produce one by chance, which is what lets a probabilistic bound sit in a test.
            assert!(a.lower_bound(1e-6) <= truth, "seed {seed}: lower {} > truth {truth}", a.lower_bound(1e-6));
            assert!((a.log_z - truth).abs() < 0.3, "seed {seed}: estimate {} vs truth {truth} (ess {})", a.log_z, a.ess);
            assert!(a.ess > 8.0, "seed {seed}: degenerate weights, ess {}", a.ess);
        }
        // And the bound is not vacuous: the Markov slack is the only slack.
        let a = ais(&g, &ladder, 2, 64, 7);
        assert!(truth - a.lower_bound(0.05) < 3.5);
    }

    /// From exact target draws, the reverse bound is unconditional, and the sandwich closes.
    #[test]
    fn reverse_ais_from_exact_draws_sandwiches_the_truth() {
        let g = ising::ring(10, 1.0, 0.2);
        let beta = 0.9;
        let truth = exact_log_z(&g, beta);
        let ladder = linear_ladder(beta, 64);
        // Exact draws from the enumerated Boltzmann distribution.
        let p = ising::exact_boltzmann(&g, beta);
        let mut rng = Pcg::new(99, 0);
        let starts: Vec<Vec<i8>> = (0..64)
            .map(|_| {
                let u = rng.f64();
                let mut acc = 0.0;
                let mut mask = 0usize;
                for (m, &pm) in p.iter().enumerate() {
                    acc += pm;
                    if acc >= u {
                        mask = m;
                        break;
                    }
                    mask = m;
                }
                (0..g.n).map(|b| if mask >> b & 1 == 1 { 1 } else { -1 }).collect()
            })
            .collect();
        for seed in 0..10u64 {
            let fwd = ais(&g, &ladder, 2, 64, seed);
            let rev = reverse_ais(&g, &ladder, 2, &starts, seed + 1000);
            let sw = Sandwich::new(&fwd, &rev, 1e-6);
            assert!(sw.contains(truth), "seed {seed}: [{}, {}] misses {truth}", sw.lower, sw.upper);
            assert!((rev.log_z - truth).abs() < 0.3, "seed {seed}: reverse estimate {} vs {truth} (ess {})", rev.log_z, rev.ess);
            assert!(rev.ess > 8.0, "seed {seed}: degenerate reverse weights, ess {}", rev.ess);
            // The two point estimates bracket the truth in expectation: forward low, reverse high.
            assert!(fwd.log_z <= rev.log_z + 0.2, "seed {seed}: forward {} above reverse {}", fwd.log_z, rev.log_z);
        }
    }

    /// The integration bracket contains the truth and tightens as the ladder is refined.
    #[test]
    fn thermodynamic_integration_brackets_and_tightens() {
        let g = ising::ring(16, 1.0, 0.1);
        let beta = 1.0;
        let truth = ring_log_z(16, 1.0, 0.1, beta);
        let coarse = thermodynamic_integration(&g, &linear_ladder(beta, 6), 200, 2000, 3.0, 1);
        let fine = thermodynamic_integration(&g, &linear_ladder(beta, 24), 200, 2000, 3.0, 1);
        for t in [&coarse, &fine] {
            assert!(t.lower_widened <= truth && truth <= t.upper_widened, "[{}, {}] misses {truth}", t.lower_widened, t.upper_widened);
            assert!(t.lower <= t.upper);
        }
        assert!(fine.upper - fine.lower < coarse.upper - coarse.lower, "refining the ladder must tighten the bracket");
        assert!((fine.midpoint() - truth).abs() < 0.2, "fine midpoint {} vs {truth}", fine.midpoint());
    }

    /// Past enumeration, AIS agrees with elimination on a lattice.
    #[test]
    fn ais_agrees_with_elimination_past_enumeration() {
        let g = ising::lattice2d(6, 1.0); // 36 spins: enumeration refuses, elimination does not
        let beta = 0.4;
        let truth = Elimination::default().log_partition(&g, beta).unwrap().log_z.unwrap();
        let a = ais(&g, &linear_ladder(beta, 64), 2, 128, 3);
        assert!(a.lower_bound(1e-6) <= truth);
        assert!((a.log_z - truth).abs() < 0.3, "estimate {} vs elimination {truth} (ess {})", a.log_z, a.ess);
    }

    /// Population annealing already reports an absolute `ln Z` by a different route (sequential
    /// Monte Carlo with resampling); the three must agree on the same model.
    #[test]
    fn three_estimators_agree_with_each_other_and_the_truth() {
        use crate::popanneal::{run, Params};
        let g = ising::ring(12, 1.0, 0.15);
        let beta = 0.8;
        let truth = ring_log_z(12, 1.0, 0.15, beta);
        let a = ais(&g, &linear_ladder(beta, 64), 2, 128, 5);
        let t = thermodynamic_integration(&g, &linear_ladder(beta, 32), 100, 1000, 3.0, 5);
        let p = run(&g, &Params::linear_from_zero(256, 2, beta, 64), 5);
        assert!(p.ln_z_is_absolute);
        for (name, est) in [("ais", a.log_z), ("ti midpoint", t.midpoint()), ("population annealing", p.ln_z)] {
            assert!((est - truth).abs() < 0.3, "{name}: {est} vs truth {truth}");
        }
        assert!(t.lower_widened <= truth && truth <= t.upper_widened);
    }

    /// BAR reproduces the transfer matrix at EVERY rung, and the entropy has its exact limits.
    #[test]
    fn bar_curve_matches_the_transfer_matrix_at_every_rung() {
        let (n, j, h) = (12usize, 1.0, 0.0);
        let g = ising::ring(n, j, h);
        let th = thermodynamics(&g, &linear_ladder(2.0, 40), 300, 3000, 3.0, 21);
        assert_eq!(th.rungs.len(), 40);
        assert!((th.rungs[0].log_z - n as f64 * core::f64::consts::LN_2).abs() < 1e-12, "the anchor is exact");
        assert!((th.rungs[0].entropy - n as f64 * core::f64::consts::LN_2).abs() < 1e-12);
        for r in &th.rungs[1..] {
            let truth = ring_log_z(n, j, h, r.beta);
            assert!((r.log_z - truth).abs() < 0.15 + 4.0 * r.stderr, "beta {}: BAR {} +- {} vs {truth}", r.beta, r.log_z, r.stderr);
            // entropy oracle: S = ln Z + beta <E>, <E> = -d ln Z / d beta by central difference.
            let hh = 1e-5;
            let mean_e = -(ring_log_z(n, j, h, r.beta + hh) - ring_log_z(n, j, h, r.beta - hh)) / (2.0 * hh);
            let s_truth = truth + r.beta * mean_e;
            // The claim is that the REPORTED error bars cover the truth: S inherits ln Z's
            // standard error and beta times the mean energy's.
            let tol = 0.05 + 4.0 * (r.stderr + r.beta * r.mean_energy.stderr);
            assert!((r.entropy - s_truth).abs() < tol, "beta {}: entropy {} vs {s_truth} (tol {tol})", r.beta, r.entropy);
        }
        // Entropy falls with beta and, with no field, heads for ln 2 (the two ground states).
        for w in th.rungs.windows(2) {
            // Non-increasing up to the two estimates' own noise, beta * se(<E>) each.
            let slack = 0.02 + 4.0 * (w[0].beta * w[0].mean_energy.stderr + w[1].beta * w[1].mean_energy.stderr);
            assert!(w[1].entropy <= w[0].entropy + slack, "entropy rose from {} to {} (slack {slack})", w[0].entropy, w[1].entropy);
        }
        assert!((th.top().entropy - core::f64::consts::LN_2).abs() < 0.3, "S(2.0) = {}", th.top().entropy);
        // Same samples, two estimators: the TI bracket contains the BAR top rung.
        assert!(th.ti.lower_widened <= th.top().log_z && th.top().log_z <= th.ti.upper_widened);
    }

    /// Clamped AIS estimates the conditional partition function enumeration can check.
    #[test]
    fn clamped_ais_matches_the_enumerated_conditional() {
        let g = ising::ring(10, 1.0, 0.3);
        let beta = 1.1;
        let fixed = [(0usize, 1i8), (3, -1), (7, 1)];
        let free: Vec<usize> = (0..g.n).filter(|i| !fixed.iter().any(|(f, _)| f == i)).collect();
        let mut s = vec![0i8; g.n];
        for &(i, v) in &fixed {
            s[i] = v;
        }
        let mut logs = Vec::new();
        for mask in 0..(1usize << free.len()) {
            for (b, &i) in free.iter().enumerate() {
                s[i] = if mask >> b & 1 == 1 { 1 } else { -1 };
            }
            logs.push(-beta * g.energy(&s));
        }
        let truth = log_sum_exp(&logs);
        for seed in 0..6u64 {
            let a = ais_clamped(&g, &fixed, &linear_ladder(beta, 64), 2, 64, seed);
            assert_eq!(a.n, free.len());
            assert!(a.lower_bound(1e-6) <= truth);
            assert!((a.log_z - truth).abs() < 0.3, "seed {seed}: {} vs {truth}", a.log_z);
        }
    }

    #[test]
    fn rounding_steps_one_ulp_in_both_directions() {
        for x in [0.0, -0.0, 1.0, -1.0, 123.456, -1e-300, f64::MAX, f64::MIN_POSITIVE] {
            assert!(next_down(x) < x);
            assert!(next_up(x) > x);
            assert_eq!(next_up(next_down(x)), x);
        }
        assert!(next_down(f64::NEG_INFINITY) == f64::NEG_INFINITY);
        assert!(next_up(f64::INFINITY) == f64::INFINITY);
        assert!(next_down(f64::INFINITY) == f64::MAX);
    }
}
