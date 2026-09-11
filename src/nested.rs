//! Nested sampling: the evidence by peeling the prior one shell at a time (Skilling 2006).
//!
//! Skilling, "Nested sampling for general Bayesian computation", *Bayesian Analysis* **1**(4):833,
//! 2006. The idea is a change of variable. The evidence is an integral over a space nobody can
//! picture,
//!
//! ```text
//!     Z = ∫ L(s) π(ds),
//! ```
//!
//! but if `X(λ) = π{ s : L(s) > λ }` is the prior mass above a likelihood level — a number in
//! `[0, 1]`, falling — then the same integral is one-dimensional:
//!
//! ```text
//!     Z = ∫₀¹ L(X) dX.
//! ```
//!
//! Nested sampling walks that line. Keep `N` live points from the prior, delete the worst, record
//! its likelihood, and replace it with a fresh prior draw *constrained* to beat the one deleted.
//! Each deletion shrinks the enclosed prior mass by a factor `t` whose law does not depend on the
//! model at all: `t` is the largest of `N` uniforms, so `ln t` is exponential with rate `N`,
//! `E[ln t] = −1/N` and `Var[ln t] = 1/N²`. That is why `X_i = exp(−i/N)` may be written down in
//! advance, and it is the one part of the method that can be checked without reference to any
//! answer — see `the_exact_replacement_obeys_the_beta_n_1_shrinkage_law_from_enumerated_volumes`.
//!
//! # Why here, and what it is the sibling of
//!
//! [`crate::free_energy`] already carries annealed importance sampling and its reverse
//! ([`crate::free_energy::ais`], [`crate::free_energy::reverse_ais`]), Bennett's acceptance ratio
//! and thermodynamic integration. Every one of those walks a *temperature* ladder: they ask what
//! happens as the model cools, and they inherit the ladder's problems — a step too large collapses
//! the effective sample size, and a first-order transition is a wall that no schedule crosses
//! cheaply. Nested sampling has no temperature and no ladder. It walks the *likelihood*, the
//! schedule is the geometric sequence above, and one run yields `Z` at the `β` it was given plus a
//! posterior-weighted sample of every shell it opened.
//!
//! So this is the missing sibling method, not a missing lane. Where the ladder methods are the
//! right tool, they are still the right tool; where the ladder is the problem, this is the
//! alternative the literature reaches for, and it was the one route out of that family this crate
//! did not have.
//!
//! # The map onto an Ising model
//!
//! The prior is uniform over the `2ⁿ` spin states, and the likelihood is the Boltzmann weight
//! `L(s) = exp(−β E(s))` — with **this crate's sign convention**, `E(s) = −Σ J sᵢsⱼ − Σ hᵢsᵢ`, so a
//! *lower* energy is a *higher* likelihood and the worst live point is the one with the largest
//! energy. The evidence is then the prior average of the Boltzmann weight, which is the partition
//! function divided by the state count:
//!
//! ```text
//!     Z_evidence = 2⁻ⁿ Σ_s exp(−β E(s))      ⇒      ln Z = ln Z_evidence + n ln 2.
//! ```
//!
//! [`Nested::log_partition`] applies that shift, so the result is directly comparable with
//! [`crate::free_energy::exact_log_z`], [`crate::free_energy::ring_log_z`] and
//! [`crate::exact::Elimination::log_partition`] — and it is compared with all three below.
//!
//! # A discrete spectrum breaks the method, and one auxiliary variable repairs it
//!
//! The shrinkage law assumes the likelihood is a continuous random variable under the prior. A spin
//! model's spectrum is discrete and massively degenerate: at `h = 0` every state is tied with its
//! own reflection, so several live points routinely share the worst likelihood. Delete one of them
//! and the survivors *do not satisfy* the constraint that defines the next region — the invariant
//! the whole derivation rests on is gone, silently, and `X_i = exp(−i/N)` becomes a schedule the run
//! is no longer following.
//!
//! The repair is standard and it is cheap: give every point an auxiliary `u ~ U(0,1)` and order by
//! `(L, u)` lexicographically. The prior over `(s, u)` is still uniform, `L` still does not depend
//! on `u`, so `Z` is unchanged; but the *ordering* is now total and its mass function is continuous,
//! and the Beta(N, 1) law holds exactly again. `better_than` is that order, [`Shell::tiebreak`]
//! records where each threshold fell inside its tier, and the shrinkage test recomputes the exact
//! prior volume from it by enumeration.
//!
//! # What the replacement sampler does, and what it costs
//!
//! A fresh *constrained prior* draw is the expensive part of any nested sampler, and the usual
//! answer is the one used here: clone a surviving live point — which is already distributed
//! correctly — and evolve it with Metropolis moves that reject anything outside the constraint.
//! Proposals flip one spin and redraw `u`; the proposal is symmetric, so acceptance is exactly the
//! indicator of the constraint.
//!
//! **That replacement is right in its marginal and wrong in its independence, and the shrinkage
//! test measures exactly how wrong.** The clone is uniform on the region and the kernel preserves
//! that, so each live point on its own is distributed correctly; but they are correlated, and a
//! correlated live set behaves like a smaller independent one. Against the exact prior volumes on
//! the twelve-spin fixture at `N = 40`, six seeds, about 1,860 shells — as the effective live count
//! `N_eff/N` each moment implies, and as that moment's distance from the law in its own standard
//! errors:
//!
//! | replacement | `N_eff/N` from `E[ln t]` | from `Var[ln t]` | mean dev | var dev |
//! |---|---|---|---|---|
//! | exact draw (`sweeps = 0`) | 1.01 | 1.06 | +0.6 se | −1.7 se |
//! | walk, 8 sweeps | 0.95 | 0.85 | −2.2 se | +5.9 se |
//! | walk, 20 sweeps | 0.94 | 0.92 | −2.7 se | +3.0 se |
//! | walk, 80 sweeps | 0.98 | 0.81 | −1.1 se | +8.0 se |
//!
//! **More sweeps do not fix it**, which is the part worth recording. The correlation is not slow
//! mixing inside the constrained region; it is that the region FRAGMENTS under single flips, and a
//! clone that starts in its parent's component stays there however long it walks. A 6% effective
//! loss is cheap and the test bounds it at 20%; what would not be honest is to call the replacement
//! independent.
//!
//! Once the region is down to a few states no single flip stays inside it at all. A walk that
//! accepts nothing falls back to `rejection` — an exact draw from the constrained prior, which on
//! a small model costs the `1/X` draws the method already told us to expect — and only if THAT
//! fails is the run [`NestedError::Stuck`]. `sweeps = 0` asks for that exact draw every time, which
//! is the reference the walk is measured against above.
//!
//! Each proposal recomputes the energy of the whole state with [`Graph::energy`] instead of
//! carrying an incremental `2 sᵢ fᵢ`. That is `O(edges)` per proposal rather than `O(degree)`, and
//! it is paid deliberately: the tie-break compares energies with `==`, and an incrementally carried
//! energy makes the *same state* compare equal to itself along one path and not along another after
//! a few hundred flips of rounding drift. The level identity has to be a function of the state.
//!
//! # What this carries, and what it does not
//!
//! `sqrt(H/N)` — `H` the information, [`Nested::information`] — is Skilling's standard error, and it
//! is a **standard error, not a bound**. It describes the dominant error, the random walk of `ln X`
//! down the sequence of shells; it says nothing about a mode the walk never found. Nothing in this
//! module is a bound, so nothing in it accumulates through [`crate::round`] — with one exception.
//! [`Nested::log_partition_floor`] is an honest unconditional lower bound on `ln Z`, because `Z` is
//! a sum of positive terms and the run found one of them, and it is computed from an energy summed
//! with [`crate::round::sum_up`] so that the floor cannot be lifted above the truth by the rounding
//! of its own arithmetic. When a *bound* on `ln Z` is what is wanted rather than an estimate,
//! [`crate::free_energy::Ais::lower_bound`] is the one with a theorem behind it.

use crate::graph::Graph;
use crate::rng::Pcg;

/// The generator stream, so a nested run's draws do not collide with another sampler's on the
/// same seed. "Nest" in hex.
const STREAM: u64 = 0x_4E65_7374;

/// A live point: the state, its energy, and the tie-break that totalises a discrete spectrum.
#[derive(Clone)]
struct Point {
    s: Vec<i8>,
    energy: f64,
    tiebreak: f64,
}

/// The sort key of a point: `(energy, tiebreak)`, the pair `better_than` orders.
fn key(p: &Point) -> (f64, f64) {
    (p.energy, p.tiebreak)
}

/// Is `a` strictly better than `b` — higher likelihood, ties broken by the auxiliary variable?
///
/// **This is the single line the sign convention lands on.** `E = −J s s − h s` and `L = exp(−βE)`,
/// so for any `β > 0` a LOWER energy is a HIGHER likelihood, and the point nested sampling deletes
/// is the one with the LARGEST energy. Reverse it and the run walks up the spectrum instead of down,
/// computing the evidence of a model nobody asked about.
///
/// Written with an early return rather than a short-circuiting `or` on one line. That is not a style
/// choice: the mutation harness that proves this crate's tests bite separates its fields with the
/// vertical bar, and a source line carrying one silently shifts every field after it, so the line
/// this module most wants mutated is written without one.
fn better_than(a: (f64, f64), b: (f64, f64)) -> bool {
    if a.0 != b.0 {
        return a.0 < b.0;
    }
    a.1 > b.1
}

/// One deleted point: the shell of prior mass it closed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Shell {
    /// `ln L` of the deleted point, `−β E`.
    pub log_l: f64,
    /// Energy of the deleted point — the likelihood threshold every later point had to beat.
    pub energy: f64,
    /// Its auxiliary variable, which completes the threshold: a point at exactly [`Shell::energy`]
    /// is inside the constraint only if its own tie-break is larger than this.
    pub tiebreak: f64,
    /// `ln X` of the prior mass remaining AFTER this deletion, `−(i+1)/N` for the `i`-th shell.
    /// Paired with [`Shell::energy`] and [`Shell::tiebreak`] this is the method's whole claim: that
    /// the region beating this threshold really does hold `exp(log_x)` of the prior.
    pub log_x: f64,
    /// `ln(X_i − X_{i+1})`, the shell's width. Computed through `exp_m1` rather than as a
    /// difference of exponentials, which cancels to nothing for large `N`.
    pub log_weight: f64,
}

/// Why a nested-sampling run produced no answer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum NestedError {
    /// The iteration cap was reached while the unexplored region could still hold real evidence.
    DidNotTerminate {
        /// Deletions performed before the cap.
        iterations: usize,
        /// `ln` of the largest share of the evidence the unexplored region could still hold,
        /// relative to what had been accumulated — the quantity `ln tol` is compared against.
        log_remaining_fraction: f64,
    },
    /// The walk accepted nothing AND the fallback draw could not find the constrained region
    /// either, so the live set would have gained an exact copy of a point it already held.
    ///
    /// This is the endgame of a finite model: the region has shrunk past what
    /// [`MAX_REJECTION_TRIES`] prior draws can hit. Stop earlier with a larger `tol`, or accept
    /// that the run has resolved the spectrum as far as prior draws reach.
    Stuck {
        /// The deletion this happened on.
        iteration: usize,
        /// Walk proposals made without one landing inside the constraint; `0` when `sweeps` is
        /// zero and there was no walk to begin with.
        proposals: usize,
        /// The energy the walk was pinned at.
        energy: f64,
    },
}

impl core::fmt::Display for NestedError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match *self {
            NestedError::DidNotTerminate { iterations, log_remaining_fraction } => write!(
                f,
                "nested sampling did not terminate in {iterations} deletions: the unexplored \
                 region could still hold exp({log_remaining_fraction:.3}) of the evidence found \
                 so far, which is above the tolerance"
            ),
            NestedError::Stuck { iteration, proposals, energy } => write!(
                f,
                "no replacement could be drawn at deletion {iteration}: the walk accepted none of \
                 {proposals} proposals from energy {energy}, and the fallback did not hit the \
                 constrained region in {MAX_REJECTION_TRIES} prior draws either. The region is \
                 smaller than prior draws can reach, so a replacement would have been a copy of a \
                 live point"
            ),
        }
    }
}

impl core::error::Error for NestedError {}

/// A finished nested-sampling run.
#[derive(Clone, Debug)]
pub struct Nested {
    /// Spins in the model.
    pub n: usize,
    /// Inverse temperature the likelihood was taken at.
    pub beta: f64,
    /// Live points the run was kept at, the `N` of `X_i = exp(−i/N)` and of `sqrt(H/N)`.
    pub live: usize,
    /// One entry per deleted point, in deletion order.
    pub shells: Vec<Shell>,
    /// `ln` of the prior-averaged evidence `2⁻ⁿ Σ exp(−β E)`. Add `n ln 2` for the partition
    /// function, which [`Nested::log_partition`] does.
    pub log_evidence: f64,
    /// Skilling's information `H = Σ pᵢ ln(Lᵢ/Z)` in nats: how many `e`-folds of prior mass the
    /// posterior occupies. It is what sets both the run length (`≈ N H` deletions) and the error.
    pub information: f64,
    /// `sqrt(H/N)`, the standard error of [`Nested::log_evidence`]. A standard error and not a
    /// bound — see the module docs.
    pub log_evidence_stderr: f64,
    /// `ln X` at termination: the prior mass the final live set still covered.
    pub log_x_final: f64,
    /// `ln` of the largest evidence the unexplored region could still have held when the run
    /// stopped, relative to the evidence accumulated by then. Below `ln tol` by construction.
    pub log_remaining_fraction: f64,
    /// The lowest-energy state the run ever held. Nested sampling is a minimiser as well as an
    /// integrator: the live set only ever improves.
    pub best: Vec<i8>,
    /// Its energy, as [`Graph::energy`] computes it.
    pub best_energy: f64,
    /// An unconditional lower bound on `ln Z` — the ONE thing here that is a bound rather than an
    /// estimate. `Z` is a sum of positive terms and [`Nested::best`] is one of them, so
    /// `ln Z ≥ −β E(best)`; the energy is summed with [`crate::round::sum_up`] and the product
    /// rounded down, so no rounding in this line can lift it above the truth.
    pub log_partition_floor: f64,
    /// Single-flip proposals made across every replacement walk.
    pub proposals: u64,
    /// How many were inside the constraint. A rate near zero means the walks are not moving and
    /// the live points are near-copies of each other, whatever the evidence says.
    pub accepts: u64,
    /// Replacements the walk could not make, which fell back to an exact draw by `rejection`.
    /// Zero on a model large enough that the constrained region never shrinks to a few states;
    /// a large share of the deletions on a small one, where it is the reason the run finished.
    pub fallbacks: usize,
    /// Prior draws those fallbacks spent. Divided by [`Nested::fallbacks`] this is the realised
    /// `1/X` of the regions the walk gave up on.
    pub fallback_draws: u64,
}

impl Nested {
    /// `ln Z(β) = ln Z_evidence + n ln 2`, on the same scale as
    /// [`crate::free_energy::exact_log_z`].
    #[must_use]
    pub fn log_partition(&self) -> f64 {
        self.log_evidence + self.n as f64 * core::f64::consts::LN_2
    }

    /// Deletions the run made.
    #[must_use]
    pub fn iterations(&self) -> usize {
        self.shells.len()
    }

    /// Fraction of proposals that landed inside the constraint; `0` if none were made.
    #[must_use]
    pub fn accept_rate(&self) -> f64 {
        if self.proposals == 0 {
            return 0.0;
        }
        self.accepts as f64 / self.proposals as f64
    }

    /// The realised shrinkage `ln(X_i / X_{i−1})` of every shell, which is `−1/N` by construction
    /// and therefore says nothing on its own.
    ///
    /// It is here for the test that checks the *truth* of the schedule: recompute each threshold's
    /// exact prior volume and compare the realised ratios with these. The distinction is the whole
    /// difference between a check that can fail and one that cannot.
    #[must_use]
    pub fn nominal_shrinkage(&self) -> f64 {
        -1.0 / self.live as f64
    }
}

/// The state's energy, rounded so it is certainly NOT BELOW the exact value.
///
/// [`Graph::energy`] accumulates in round-to-nearest, so the energy it returns may sit a rounding
/// BELOW the exact one — and `exp(−β E)` is then a rounding ABOVE the true Boltzmann weight, which
/// is the wrong side for a lower bound on `Z`. The error is `1e−16` and the claim it breaks is
/// absolute, which is exactly the failure [`crate::round`] exists for.
fn energy_up(g: &Graph, s: &[i8]) -> f64 {
    let mut terms = Vec::with_capacity(g.n + g.n_edges);
    for i in 0..g.n {
        let si = f64::from(s[i]);
        terms.push(-g.h[i] * si);
        for k in g.offset[i]..g.offset[i + 1] {
            let j = g.nbr[k] as usize;
            // Each undirected edge appears in both CSR rows; count it once.
            if j > i {
                terms.push(-g.w[k] * si * f64::from(s[j]));
            }
        }
    }
    crate::round::sum_up(&terms)
}

fn log_sum_exp(v: &[f64]) -> f64 {
    let m = v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !m.is_finite() {
        return m;
    }
    m + v.iter().map(|x| (x - m).exp()).sum::<f64>().ln()
}

fn log_add(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        return b;
    }
    if b == f64::NEG_INFINITY {
        return a;
    }
    let m = a.max(b);
    m + ((a - m).exp() + (b - m).exp()).ln()
}

/// Evolve `p` under the constrained prior: `sweeps · n` single-flip proposals, each accepted
/// exactly when the resulting `(E, u)` beats `threshold`.
///
/// The proposal flips one uniformly chosen spin and redraws `u`, which is symmetric — the density
/// of the proposed `u` does not depend on the current one — so Metropolis–Hastings reduces to the
/// indicator of the constraint and the chain is reversible with respect to the uniform distribution
/// on the constrained region.
///
/// Returns `(accepts, proposals)`.
fn evolve(
    g: &Graph,
    p: &mut Point,
    threshold: (f64, f64),
    sweeps: usize,
    rng: &mut Pcg,
    enforce: bool,
) -> (u64, u64) {
    let proposals = (sweeps * g.n) as u64;
    let mut accepts = 0u64;
    for _ in 0..proposals {
        let i = ((rng.f64() * g.n as f64) as usize).min(g.n - 1);
        p.s[i] = -p.s[i];
        // Recomputed, not carried: see the module docs. A state's energy must be a function of the
        // state, or the `==` in `better_than` compares an energy against a drifted copy of itself.
        let e_new = g.energy(&p.s);
        let u_new = rng.f64();
        let inside = better_than((e_new, u_new), threshold);
        if inside || !enforce {
            p.energy = e_new;
            p.tiebreak = u_new;
            accepts += 1;
        } else {
            p.s[i] = -p.s[i];
        }
    }
    (accepts, proposals)
}

/// Tries beyond the expected `1/X` that the fallback draw is given, so a miss is an `e⁻¹⁶` event
/// rather than bad luck.
pub const REJECTION_OVERSHOOT: f64 = 256.0;

/// The most prior draws the fallback will make. A region the run has peeled below `2⁻²⁰` of the
/// prior is past what any prior-draw scheme reaches, and the honest answer there is
/// [`NestedError::Stuck`] rather than a million energy evaluations per deletion.
pub const MAX_REJECTION_TRIES: u64 = 1 << 22;

/// An EXACT draw from the constrained prior, by rejection, or `None` within the budget.
///
/// The fallback for a walk that could not move, and on a small model it is not a poor relation: a
/// uniform state that passes the constraint is exactly what nested sampling asks for, with none of
/// the correlation the clone-and-evolve replacement carries. A region of prior mass `X` takes `1/X`
/// draws in expectation, and `X` is a number the method writes down in advance — so the budget is
/// set by the run's own claim about itself rather than by a knob.
///
/// This is why the module works at all on a twelve-spin model. The evidence there has
/// single STATES contributing percent-level shares, so the run must peel down to individual
/// configurations, and no single-flip walk navigates a region of three states. On a model large
/// enough for that never to happen, `1/X` is astronomical long before the walk stalls, and this
/// function is never reached.
fn rejection(g: &Graph, threshold: (f64, f64), log_x: f64, rng: &mut Pcg) -> Option<(Point, u64)> {
    let budget = ((REJECTION_OVERSHOOT * (-log_x).exp()).ceil() as u64).min(MAX_REJECTION_TRIES);
    for t in 0..budget {
        let s: Vec<i8> = (0..g.n).map(|_| rng.spin(0.5)).collect();
        let energy = g.energy(&s);
        let tiebreak = rng.f64();
        if better_than((energy, tiebreak), threshold) {
            return Some((Point { s, energy, tiebreak }, t + 1));
        }
    }
    None
}

/// Nested sampling of `exp(−β E)` under the uniform prior over the `2ⁿ` states.
///
/// `live` is Skilling's `N`: the shrinkage per deletion is `exp(−1/N)` and the error is
/// `sqrt(H/N)`, so the run is `≈ N·H` deletions long and costs `N sweeps n` flips per `e`-fold.
/// `sweeps` is how many full passes of single-flip proposals each replacement walk makes.
///
/// **`sweeps = 0` means no walk at all**: every replacement is an exact draw from the constrained
/// prior by `rejection`. That is the *reference* sampler — the live points are then genuinely
/// independent, and the shrinkage law holds to its own standard error rather than to the few
/// percent the clone-and-evolve walk costs. It is affordable only while `1/X` stays reachable, so
/// on a model of twenty spins or more it is not an option; on the small models this crate verifies
/// against it is both affordable and the thing the walk should be measured against.
///
/// The run stops when the unexplored region could hold no more than a fraction `tol` of the
/// evidence already found, judged by the best live likelihood. That criterion is conservative on
/// purpose and `0.1` is a sensible value for a small model, because the final live block is not
/// DROPPED at termination — it is estimated, and unbiasedly. Measured on the twelve-spin fixture
/// at `β = 0.9`: `ln Z` moved by 0.0008 nats between `tol = 0.1` and `tol = 1e−3`, while the run
/// got fifty times more expensive.
///
/// # Errors
///
/// [`NestedError::DidNotTerminate`] if `max_iterations` deletions are not enough to reach `tol`,
/// and [`NestedError::Stuck`] if a replacement walk accepted no proposal and the fallback draw
/// could not find the constrained region either.
///
/// # Panics
///
/// If `live` is below 2 (a single live point has nothing to clone and no shrinkage law), if
/// `max_iterations` is zero, if `beta` is negative or not finite, or if `tol` is not strictly
/// inside `(0, 1)`.
pub fn nested(
    g: &Graph,
    beta: f64,
    live: usize,
    sweeps: usize,
    tol: f64,
    max_iterations: usize,
    seed: u64,
) -> Result<Nested, NestedError> {
    run(g, beta, live, sweeps, tol, max_iterations, seed, true)
}

/// [`nested`] with the likelihood constraint DELIBERATELY NOT ENFORCED — the negative control.
///
/// Every replacement accepts every proposal, so the live set never marches inward and the prior
/// volumes the run assumes are a fiction. It exists so the tests can assert that the checks in this
/// module actually detect a broken constrained sampler, which is the one failure mode a nested
/// sampler cannot notice from the inside.
///
/// # Errors
///
/// As [`nested`]; [`NestedError::Stuck`] cannot occur, since every proposal is accepted.
#[cfg(test)]
pub(crate) fn nested_ignoring_the_constraint(
    g: &Graph,
    beta: f64,
    live: usize,
    sweeps: usize,
    tol: f64,
    max_iterations: usize,
    seed: u64,
) -> Result<Nested, NestedError> {
    run(g, beta, live, sweeps, tol, max_iterations, seed, false)
}

fn run(
    g: &Graph,
    beta: f64,
    live: usize,
    sweeps: usize,
    tol: f64,
    max_iterations: usize,
    seed: u64,
    enforce: bool,
) -> Result<Nested, NestedError> {
    assert!(live >= 2, "nested sampling needs at least two live points; got {live}");
    assert!(max_iterations >= 1, "a run needs at least one deletion");
    assert!(beta >= 0.0 && beta.is_finite(), "beta must be finite and non-negative; got {beta}");
    assert!(tol > 0.0 && tol < 1.0, "the tolerance is a fraction of the evidence; got {tol}");

    let mut rng = Pcg::new(seed, STREAM);
    let mut pts: Vec<Point> = (0..live)
        .map(|_| {
            let s: Vec<i8> = (0..g.n).map(|_| rng.spin(0.5)).collect();
            let energy = g.energy(&s);
            Point { s, energy, tiebreak: rng.f64() }
        })
        .collect();

    let nl = live as f64;
    let ln_shrink = -1.0 / nl;
    // ln(X_i − X_{i+1}) = ln X_i + ln(1 − e^{−1/N}). The second term is written with `exp_m1`
    // because `1.0 - (-1.0/nl).exp()` loses every significant digit for large N: at N = 1e8 it is
    // the difference of two numbers agreeing to eight places.
    let ln_shell = (-ln_shrink.exp_m1()).ln();
    let ln_tol = tol.ln();

    let mut shells: Vec<Shell> = Vec::new();
    let mut log_z_acc = f64::NEG_INFINITY;
    let mut log_remaining_fraction = f64::INFINITY;
    let mut stopped = false;
    let (mut proposals, mut accepts) = (0u64, 0u64);
    let (mut fallbacks, mut fallback_draws) = (0usize, 0u64);

    let mut best_at = 0usize;
    for i in 1..pts.len() {
        if pts[i].energy < pts[best_at].energy {
            best_at = i;
        }
    }
    let mut best = pts[best_at].s.clone();
    let mut best_energy = pts[best_at].energy;

    for k in 0..max_iterations {
        let log_x = k as f64 * ln_shrink;
        // The most evidence the unexplored region could still hold is its volume times the best
        // likelihood in it — and every live point is in it, so the best live point bounds that.
        let e_min = pts.iter().map(|p| p.energy).fold(f64::INFINITY, f64::min);
        log_remaining_fraction = log_x - beta * e_min - log_z_acc;
        if log_remaining_fraction < ln_tol {
            stopped = true;
            break;
        }

        let mut worst = 0usize;
        for i in 1..pts.len() {
            if better_than(key(&pts[worst]), key(&pts[i])) {
                worst = i;
            }
        }
        let threshold = key(&pts[worst]);
        let log_l = -beta * pts[worst].energy;
        let log_weight = log_x + ln_shell;
        shells.push(Shell {
            log_l,
            energy: pts[worst].energy,
            tiebreak: pts[worst].tiebreak,
            log_x: log_x + ln_shrink,
            log_weight,
        });
        log_z_acc = log_add(log_z_acc, log_weight + log_l);

        // Clone a SURVIVOR, never the point being deleted: the survivors are the ones the
        // derivation says are uniform on the region the new point must land in.
        let mut j = ((rng.f64() * (live - 1) as f64) as usize).min(live - 2);
        if j >= worst {
            j += 1;
        }
        let mut p = pts[j].clone();
        let (a, q) = evolve(g, &mut p, threshold, sweeps, &mut rng, enforce);
        accepts += a;
        proposals += q;
        if a == 0 {
            // The walk is pinned -- every single flip leaves the region -- or there was no walk,
            // `sweeps` being zero. Either way, draw from the constrained prior directly rather
            // than installing a copy of a live point.
            let Some((fresh, tries)) = rejection(g, threshold, log_x + ln_shrink, &mut rng) else {
                return Err(NestedError::Stuck {
                    iteration: k,
                    proposals: q as usize,
                    energy: p.energy,
                });
            };
            p = fresh;
            fallbacks += 1;
            fallback_draws += tries;
        }
        if p.energy < best_energy {
            best_energy = p.energy;
            best.copy_from_slice(&p.s);
        }
        pts[worst] = p;
    }
    if !stopped {
        return Err(NestedError::DidNotTerminate {
            iterations: shells.len(),
            log_remaining_fraction,
        });
    }

    // The final live set covers the remaining mass X_m, split equally: N points, X_m/N each.
    let log_x_final = shells.len() as f64 * ln_shrink;
    let log_w_live = log_x_final - nl.ln();
    let mut terms = Vec::with_capacity(shells.len() + live);
    let mut log_ls = Vec::with_capacity(shells.len() + live);
    for sh in &shells {
        terms.push(sh.log_weight + sh.log_l);
        log_ls.push(sh.log_l);
    }
    for p in &pts {
        let l = -beta * p.energy;
        terms.push(log_w_live + l);
        log_ls.push(l);
    }
    let log_evidence = log_sum_exp(&terms);
    // H = Σ pᵢ ln(Lᵢ/Z), pᵢ = wᵢLᵢ/Z. It is a Kullback–Leibler divergence and so non-negative;
    // rounding can put it a hair below zero when the posterior IS the prior (β = 0), and the
    // square root is taken from the clamped value rather than reporting a NaN error bar.
    let information: f64 =
        terms.iter().zip(&log_ls).map(|(t, l)| (t - log_evidence).exp() * (l - log_evidence)).sum();

    Ok(Nested {
        n: g.n,
        beta,
        live,
        shells,
        log_evidence,
        information,
        log_evidence_stderr: (information.max(0.0) / nl).sqrt(),
        log_x_final,
        log_remaining_fraction,
        log_partition_floor: crate::free_energy::next_down(-beta * energy_up(g, &best)),
        best,
        best_energy,
        proposals,
        accepts,
        fallbacks,
        fallback_draws,
    })
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::free_energy::{exact_log_z, ring_log_z};

    /// A twelve-spin model with fields, so the ± symmetry does not halve the spectrum, and chords,
    /// so the constrained region fragments and the walk has somewhere to get trapped.
    fn fixture() -> Graph {
        let mut b = crate::graph::GraphBuilder::new(12);
        let js = [1.0, -0.7, 1.0, 0.6, -1.0, 0.8, 1.0, -0.5, 0.9, 1.0, -0.8, 0.4];
        for i in 0..12 {
            b.couple(i, (i + 1) % 12, js[i]);
        }
        b.couple(0, 5, -0.9);
        b.couple(2, 8, 0.7);
        b.couple(4, 10, -0.6);
        for i in 0..12 {
            b.bias(i, 0.15 * (i as f64 - 5.5) / 6.0);
        }
        b.build()
    }

    /// Every energy in the model, ascending, computed by exactly the routine the run uses so the
    /// two agree bit for bit and `==` means what the tie-break needs it to mean.
    fn spectrum(g: &Graph) -> Vec<f64> {
        let mut s = vec![-1i8; g.n];
        let mut out = Vec::with_capacity(1usize << g.n);
        for mask in 0..(1usize << g.n) {
            for b in 0..g.n {
                s[b] = if mask >> b & 1 == 1 { 1 } else { -1 };
            }
            out.push(g.energy(&s));
        }
        out.sort_by(f64::total_cmp);
        out
    }

    /// The EXACT prior mass of the region a threshold `(energy, tiebreak)` encloses: every state
    /// strictly better than it, plus the fraction `1 − u` of the tier exactly at it.
    ///
    /// This is the oracle the shrinkage tests are measured against, and it never consults the run's
    /// own arithmetic — only the enumerated spectrum and the recorded threshold.
    fn volume(spec: &[f64], energy: f64, tiebreak: f64) -> f64 {
        let below = spec.partition_point(|&e| e < energy);
        let tier = spec.partition_point(|&e| e <= energy) - below;
        (below as f64 + tier as f64 * (1.0 - tiebreak)) / spec.len() as f64
    }

    /// `ln(X_i / X_{i−1})` of every shell, from the EXACT volumes rather than the nominal schedule.
    ///
    /// Every entry is negative for a correct run — the enclosed region can only shrink — and the
    /// negative control produces positive ones, which is a detection in itself.
    fn realised_shrinkage(spec: &[f64], r: &Nested) -> Vec<f64> {
        let mut prev = 1.0f64;
        let mut out = Vec::with_capacity(r.shells.len());
        for sh in &r.shells {
            let v = volume(spec, sh.energy, sh.tiebreak);
            assert!(v > 0.0, "a threshold enclosing nothing has no shrinkage");
            out.push((v / prev).ln());
            prev = v;
        }
        out
    }

    fn mean_var(v: &[f64]) -> (f64, f64) {
        let m = v.len() as f64;
        let mean = v.iter().sum::<f64>() / m;
        (mean, v.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / (m - 1.0))
    }

    /// ORACLE: exhaustive enumeration of all 4096 states, over twenty seeds.
    ///
    /// The answer must sit inside the method's OWN error bar, and the bar must be the right SIZE —
    /// a run reporting `±50` nats would pass a coverage test and mean nothing. So the calibration
    /// is asserted from both sides: the root-mean-square of the standardised error
    /// `z = (est − exact)/σ` has to be near one, which fails if the bar is inflated and fails if it
    /// is optimistic. Over twenty seeds `rms(z)` has a standard error of about `0.16` around one.
    ///
    /// Measured here: `rms(z) = 1.09`, mean `z = +0.08`, `σ = 0.168` nats, 13 of 20 seeds inside
    /// one σ (68% of 20 is 13.6) and 19 inside two (19.1). That is the distribution the theory
    /// predicts, not a band wide enough to contain anything.
    #[test]
    fn log_z_matches_exhaustive_enumeration_within_its_own_stated_uncertainty() {
        let g = fixture();
        let beta = 0.9;
        let exact = exact_log_z(&g, beta);
        let mut zs = Vec::new();
        let mut sigmas = Vec::new();
        for seed in 0..20u64 {
            let r = nested(&g, beta, 100, 12, 0.1, 500_000, seed).unwrap();
            assert!(r.information > 0.0, "seed {seed}: the posterior is narrower than the prior");
            zs.push((r.log_partition() - exact) / r.log_evidence_stderr);
            sigmas.push(r.log_evidence_stderr);
        }
        let within = |t: f64| zs.iter().filter(|z| z.abs() <= t).count();
        let rms = (zs.iter().map(|z| z * z).sum::<f64>() / zs.len() as f64).sqrt();
        let bias = zs.iter().sum::<f64>() / zs.len() as f64;
        let sigma = sigmas.iter().sum::<f64>() / sigmas.len() as f64;
        assert!(
            (0.5..=1.6).contains(&rms),
            "the stated uncertainty is not the realised one: rms(z) = {rms:.3} over {} seeds, \
             mean sigma {sigma:.4} nats, exact ln Z = {exact:.4}",
            zs.len()
        );
        assert!(bias.abs() < 0.8, "systematic offset of {bias:.3} sigma, sigma = {sigma:.4}");
        assert!(within(2.0) >= 16, "only {} of 20 seeds inside two sigma", within(2.0));
        assert_eq!(within(3.5), 20, "a seed landed past 3.5 sigma; z = {zs:?}");
        // And the bar is a real number of nats rather than a shrug: a hundred live points on this
        // model must buy better than a quarter of a nat, or the coverage was bought by inflating
        // the very quantity the coverage is measured in.
        assert!(sigma < 0.25, "sigma = {sigma:.4} nats is too loose to have tested anything");
    }

    /// ORACLE: the Beta(N, 1) shrinkage law, against prior volumes computed by enumeration — a test
    /// of the MECHANISM that never looks at the evidence.
    ///
    /// `t = X_i/X_{i−1}` is the largest of `N` uniforms whatever the model is, so `−ln t` is
    /// exponential with rate `N`: `E[ln t] = −1/N` and `Var[ln t] = 1/N²`. The `X_i` here is not the
    /// run's nominal `exp(−i/N)` — that would be a tautology — but the TRUE prior mass of the
    /// region each recorded threshold encloses, counted over all 4096 states by [`volume`].
    ///
    /// Run with `sweeps = 0`, where every replacement is an exact draw from the constrained prior,
    /// so the live points really are independent and the law must hold to its own standard error.
    /// The companion test measures what the walk costs against this.
    #[test]
    fn the_exact_replacement_obeys_the_beta_n_1_shrinkage_law_from_enumerated_volumes() {
        let g = fixture();
        let spec = spectrum(&g);
        for live in [40usize, 60] {
            let mut lnt = Vec::new();
            let mut want_mean = 0.0;
            for seed in 0..6u64 {
                let r = nested(&g, 0.9, live, 0, 0.1, 500_000, seed).unwrap();
                assert_eq!(r.fallbacks, r.iterations(), "sweeps = 0 is the exact sampler");
                assert_eq!(r.proposals, 0, "and it makes no walk proposals at all");
                // The law is asserted against what the RUN says its schedule is, not against a
                // number retyped here, so a schedule that drifted from `-1/N` would be caught.
                want_mean = r.nominal_shrinkage();
                assert_eq!(want_mean, -1.0 / live as f64);
                lnt.extend(realised_shrinkage(&spec, &r));
            }
            assert!(lnt.iter().all(|&x| x < 0.0), "the enclosed region can only shrink");
            let m = lnt.len() as f64;
            let (mean, var) = mean_var(&lnt);
            let want_var = want_mean * want_mean;
            // The mean of `m` exponentials has standard error `(1/N)/sqrt(m)`; the sample variance
            // of an exponential has `sqrt(8/m)/N²`. Four of each: this fails on a defect, not luck.
            let se_mean = want_var.sqrt() / m.sqrt();
            let se_var = want_var * (8.0f64 / m).sqrt();
            assert!(
                (mean - want_mean).abs() < 4.0 * se_mean,
                "N = {live}: E[ln t] = {mean:.6} over {m} shells, wanted {want_mean:.6} +- {:.6}",
                4.0 * se_mean
            );
            assert!(
                (var - want_var).abs() < 4.0 * se_var,
                "N = {live}: Var[ln t] = {var:.4e}, wanted {want_var:.4e} +- {:.4e}",
                4.0 * se_var
            );
        }
    }

    /// What the clone-and-evolve walk costs, in the only unit that matters: the effective live
    /// count the realised shrinkage implies.
    ///
    /// The walk's replacement is correct in its MARGINAL — it clones a survivor, which is already
    /// uniform on the region, and moves it with a kernel that preserves that — but the live points
    /// are then correlated, and a correlated live set behaves like a smaller independent one. Both
    /// moments say so and they agree: `N_eff/N = 0.94` from the mean and `0.92` from the variance
    /// at `N = 40, sweeps = 20`, against `1.01` and `1.06` for the exact replacement above.
    ///
    /// More sweeps do not fix it, which is the finding worth recording: measured at 8, 20, 40 and
    /// 80 sweeps the mean-implied `N_eff/N` is 0.95, 0.94, 0.94, 0.98 — flat. The correlation is
    /// not slow mixing inside the constrained region, it is that a fragmented region traps the
    /// clone in its parent's component however long it walks there.
    #[test]
    fn the_walk_replacement_costs_under_a_fifth_of_the_effective_live_count() {
        let g = fixture();
        let spec = spectrum(&g);
        let live = 40usize;
        let mut lnt = Vec::new();
        let mut stalls = 0usize;
        for seed in 0..6u64 {
            let r = nested(&g, 0.9, live, 20, 0.1, 500_000, seed).unwrap();
            assert!(r.accept_rate() > 0.1, "the walk must be moving: {}", r.accept_rate());
            stalls += r.fallbacks;
            lnt.extend(realised_shrinkage(&spec, &r));
        }
        assert!(lnt.iter().all(|&x| x < 0.0), "the enclosed region can only shrink");
        assert!(stalls > 0, "this fixture is meant to trap the walk sometimes");
        let (mean, var) = mean_var(&lnt);
        let n_eff_mean = -1.0 / mean;
        let n_eff_var = 1.0 / var.sqrt();
        for (what, n_eff) in [("mean", n_eff_mean), ("variance", n_eff_var)] {
            assert!(
                n_eff > 0.8 * live as f64,
                "the {what} puts the effective live count at {n_eff:.1} of {live}, so the \
                 replacement is far more correlated than the walk is supposed to be"
            );
            assert!(
                n_eff < 1.08 * live as f64,
                "the {what} puts the effective live count at {n_eff:.1}, ABOVE {live}, which no \
                 correlation can produce -- the shrinkage is being measured wrong"
            );
        }
    }

    /// NEGATIVE CONTROL: a constrained sampler that ignores the constraint must be CAUGHT.
    ///
    /// It is caught twice, and the two catches are worth different things. The evidence is wrong by
    /// twenty-four of its own standard errors, which the enumeration oracle sees — but a reader
    /// could object that some other defect might move the evidence the other way and cancel. The
    /// shrinkage is wrong at the mechanism: the live set never marches inward, so the true prior
    /// volumes barely fall (`E[ln t] = −0.005` against a required `−0.025`) while the run assumes
    /// they fall by `1/N` a step. That check cannot be cancelled by anything, because it never
    /// consults the answer.
    #[test]
    fn a_replacement_that_ignores_the_constraint_is_caught_by_the_evidence_and_the_shrinkage() {
        let g = fixture();
        let beta = 0.9;
        let exact = exact_log_z(&g, beta);
        let spec = spectrum(&g);
        let live = 40usize;
        let want = -1.0 / live as f64;
        for seed in [11u64, 12] {
            let broken =
                nested_ignoring_the_constraint(&g, beta, live, 20, 0.1, 500_000, seed).unwrap();
            let z = (broken.log_partition() - exact) / broken.log_evidence_stderr;
            assert!(
                z.abs() > 10.0,
                "seed {seed}: the broken sampler was not detected: ln Z = {:.4} against an exact \
                 {exact:.4}, {z:.2} of its own sigma",
                broken.log_partition()
            );
            let lnt = realised_shrinkage(&spec, &broken);
            let (mean, _) = mean_var(&lnt);
            assert!(
                mean > 0.5 * want,
                "seed {seed}: the shrinkage check did not see it: E[ln t] = {mean:.5} against a \
                 required {want:.5}"
            );
            // Sharper still, and it needs no statistics: the enclosed prior mass GREW at some
            // deletions. A correct run cannot do that once, and this one does it often.
            let grew = lnt.iter().filter(|&&x| x > 0.0).count();
            assert!(grew > lnt.len() / 20, "only {grew} of {} shells grew", lnt.len());
            // And the same fixture with the constraint ON passes both, or this test proves nothing
            // about the checks and only that the broken run is broken.
            let good = nested(&g, beta, live, 20, 0.1, 500_000, seed).unwrap();
            assert!(
                (good.log_partition() - exact).abs() < 3.0 * good.log_evidence_stderr,
                "seed {seed}: the control's twin must pass: {:.4} vs {exact:.4}",
                good.log_partition()
            );
            let (good_mean, _) = mean_var(&realised_shrinkage(&spec, &good));
            assert!(good_mean < 0.5 * want, "seed {seed}: and its shrinkage: {good_mean:.5}");
        }
    }

    /// ORACLE: the transfer matrix, on a model the exact-replacement sampler could not touch.
    ///
    /// Twenty spins is a million states, so `1/X` is out of reach and every replacement here is the
    /// WALK — which is the point of the test twice over: it is an independent closed form
    /// (`ring_log_z`, Kramers–Wannier, derived from nothing in this module), and it is the regime
    /// where the walk carries the whole run, with not one fallback draw.
    #[test]
    fn a_twenty_spin_ring_matches_the_transfer_matrix_closed_form() {
        let (n, j, h, beta) = (20usize, 1.0, 0.25, 0.6);
        let g = crate::ising::ring(n, j, h);
        let exact = ring_log_z(n, j, h, beta);
        let mut hits = 0;
        for seed in 0..6u64 {
            let r = nested(&g, beta, 120, 16, 0.1, 500_000, seed).unwrap();
            assert_eq!(r.fallbacks, 0, "on a model this size the walk never needs rescuing");
            assert!(r.log_evidence_stderr < 0.25, "sigma = {}", r.log_evidence_stderr);
            if (r.log_partition() - exact).abs() < 2.5 * r.log_evidence_stderr {
                hits += 1;
            }
        }
        assert!(hits >= 5, "only {hits} of 6 seeds inside 2.5 sigma of the transfer matrix");
    }

    /// At `β = 0` the likelihood is one everywhere and the evidence is EXACTLY one, so
    /// `ln Z = n ln 2` — every shell weight and the final live block must sum to the whole prior.
    ///
    /// The shells telescope to `1 − X_m` and the live block carries `X_m`, so the only error is the
    /// floating-point one and a tolerance looser than `1e−12` would let a real defect through. The
    /// realised value is `9e−16`, one ulp of the arithmetic that produced it.
    #[test]
    fn at_beta_zero_the_evidence_is_exactly_the_prior_and_ln_z_is_n_ln_2() {
        let g = fixture();
        let r = nested(&g, 0.0, 32, 4, 0.1, 500_000, 3).unwrap();
        assert!(r.log_evidence.abs() < 1e-12, "ln evidence = {:e}", r.log_evidence);
        let want = g.n as f64 * core::f64::consts::LN_2;
        assert!((r.log_partition() - want).abs() < 1e-12, "{} vs {want}", r.log_partition());
        // The posterior IS the prior, so there is no information and no error bar to report.
        assert!(r.information.abs() < 1e-12, "H = {:e}", r.information);
        assert!(r.log_evidence_stderr < 1e-6, "sigma = {:e}", r.log_evidence_stderr);
    }

    /// ORACLE: exhaustive enumeration again, but of a BOUND rather than an estimate — the floor is
    /// never above the exact value, at any temperature, and closes on it as the model freezes.
    ///
    /// `ln Z ≥ −β E(best)` holds because `Z` is a sum of positive terms and the run found one of
    /// them. What could break it is arithmetic rather than mathematics: an energy accumulated in
    /// round-to-nearest can land below the true one, and `exp(−βE)` is then above the true term.
    /// Hence [`energy_up`]. The gap at `β = 4` is 0.30 nats, which is the log of the effective
    /// degeneracy — a statement, not a shrug.
    #[test]
    fn the_log_partition_floor_never_exceeds_the_enumerated_log_z() {
        let g = fixture();
        for &beta in &[0.2f64, 0.9, 2.0, 4.0] {
            let exact = exact_log_z(&g, beta);
            let r = nested(&g, beta, 20, 8, 0.1, 500_000, 5).unwrap();
            assert!(
                r.log_partition_floor <= exact,
                "beta {beta}: floor {} is ABOVE the exact ln Z {exact}",
                r.log_partition_floor
            );
            if beta >= 4.0 {
                assert!(
                    exact - r.log_partition_floor < 0.5,
                    "beta {beta}: the floor is {} against {exact}, too loose to be a statement",
                    r.log_partition_floor
                );
            }
        }
    }

    /// The run finds the ground state, checked against exhaustive enumeration rather than against a
    /// remembered number. Nested sampling is a minimiser as well as an integrator.
    #[test]
    fn the_best_state_found_is_the_enumerated_ground_state() {
        let g = fixture();
        let gs = spectrum(&g)[0];
        let r = nested(&g, 2.0, 60, 12, 0.1, 500_000, 7).unwrap();
        assert_eq!(r.best_energy, gs, "nested sampling walks down to the ground state");
        assert_eq!(g.energy(&r.best), r.best_energy, "and the state it reports is that state");
    }

    /// Deterministic by seed, which is this crate's headline: the same seed reproduces the run bit
    /// for bit, and a different seed does not.
    #[test]
    fn a_run_is_reproducible_from_its_seed_and_a_different_seed_moves() {
        let g = fixture();
        let a = nested(&g, 0.9, 30, 6, 0.1, 500_000, 21).unwrap();
        let b = nested(&g, 0.9, 30, 6, 0.1, 500_000, 21).unwrap();
        assert_eq!(a.log_evidence.to_bits(), b.log_evidence.to_bits());
        assert_eq!(a.shells.len(), b.shells.len());
        assert_eq!(a.best, b.best);
        let c = nested(&g, 0.9, 30, 6, 0.1, 500_000, 22).unwrap();
        assert_ne!(a.log_evidence.to_bits(), c.log_evidence.to_bits(), "seeds must not collude");
    }

    /// The iteration cap is an error rather than a truncated answer, and it says how far short it
    /// fell. A run that stops early reports a `Z` missing everything below its last shell, which is
    /// exactly the failure that looks like a converged answer.
    #[test]
    fn a_cap_that_stops_the_run_early_is_an_error_and_not_a_short_answer() {
        let g = fixture();
        let e = nested(&g, 1.5, 20, 4, 1e-6, 30, 1).unwrap_err();
        match e {
            NestedError::DidNotTerminate { iterations, log_remaining_fraction } => {
                assert_eq!(iterations, 30, "every deletion up to the cap was made");
                assert!(
                    log_remaining_fraction > 1e-6f64.ln(),
                    "it stopped above the tolerance, which is why it is an error"
                );
            }
            NestedError::Stuck { .. } => panic!("wrong variant: {e}"),
        }
        assert!(e.to_string().contains("did not terminate"), "{e}");
        assert!(core::error::Error::source(&e).is_none());
        // The Stuck variant renders too, and says what to change.
        let stuck = NestedError::Stuck { iteration: 4, proposals: 96, energy: -12.0 };
        assert!(stuck.to_string().contains("96 proposals"), "{stuck}");
        assert_ne!(stuck, e);
    }

    /// The tie-break is what makes a degenerate spectrum tractable, and this is the ordering it
    /// defines: energy first, and only then the auxiliary variable, with a higher `u` better.
    ///
    /// Asserted directly because every other test depends on it and none of them would say which
    /// half was wrong.
    #[test]
    fn the_order_is_energy_first_then_the_tiebreak_with_no_point_better_than_itself() {
        assert!(better_than((-1.0, 0.1), (0.0, 0.9)), "lower energy wins whatever the tie-break");
        assert!(!better_than((0.0, 0.9), (-1.0, 0.1)));
        assert!(better_than((0.0, 0.9), (0.0, 0.1)), "at equal energy the larger tie-break wins");
        assert!(!better_than((0.0, 0.1), (0.0, 0.9)));
        assert!(!better_than((0.0, 0.5), (0.0, 0.5)), "the order is strict");
    }

    /// ORACLE: `i64` arithmetic. On a model whose couplings and fields are integers the energy is
    /// an integer and can be computed exactly without floating point at all, so `energy_up` can be
    /// checked against the truth rather than against another float sum.
    ///
    /// It must never be below that truth — that is what keeps the floor a floor — and must not
    /// wander above it either, or the floor would be sound and useless.
    #[test]
    fn energy_up_is_never_below_the_energy_that_integer_arithmetic_gives() {
        let mut b = crate::graph::GraphBuilder::new(10);
        let js: [i64; 10] = [3, -5, 7, 2, -11, 13, -1, 4, -6, 9];
        let hs: [i64; 10] = [1, -2, 3, 0, 5, -4, 2, -3, 1, -1];
        for i in 0..10 {
            b.couple(i, (i + 1) % 10, js[i] as f64);
            b.bias(i, hs[i] as f64);
        }
        let g = b.build();
        let mut worst_gap = 0.0f64;
        for mask in 0..(1usize << 10) {
            let s: Vec<i8> = (0..10).map(|i| if mask >> i & 1 == 1 { 1 } else { -1 }).collect();
            // E = -sum J s s - sum h s, in integers, with no rounding anywhere.
            let mut exact: i64 = 0;
            for i in 0..10 {
                exact -= js[i] * i64::from(s[i]) * i64::from(s[(i + 1) % 10]);
                exact -= hs[i] * i64::from(s[i]);
            }
            let up = energy_up(&g, &s);
            assert!(up >= exact as f64, "mask {mask}: {up} is below the exact energy {exact}");
            worst_gap = worst_gap.max(up - exact as f64);
        }
        assert!(worst_gap < 1e-12, "the guard has drifted to {worst_gap}, which is not a rounding");
        // And on the real fixture, where the couplings are not exact in binary, the direction still
        // holds against every one of the 4096 states.
        let f = fixture();
        let mut s = vec![-1i8; f.n];
        for mask in 0..(1usize << f.n) {
            for i in 0..f.n {
                s[i] = if mask >> i & 1 == 1 { 1 } else { -1 };
            }
            assert!(energy_up(&f, &s) - f.energy(&s) > -1e-9, "mask {mask}");
        }
    }

    /// The shell weights telescope: `Σ (X_{i−1} − X_i) + X_m` is the whole prior, exactly one.
    ///
    /// Written through `exp_m1` precisely so it stays one for large `N`; the naive
    /// `X_{i−1}(1 − exp(−1/N))` loses eight of sixteen digits at `N = 1e8`. This checks the
    /// telescoping at `N = 2048`, where the naive form has already lost three.
    #[test]
    fn the_shell_weights_and_the_live_block_telescope_to_the_whole_prior() {
        let g = fixture();
        let r = nested(&g, 0.6, 2048, 2, 0.1, 500_000, 4).unwrap();
        let mut terms: Vec<f64> = r.shells.iter().map(|s| s.log_weight.exp()).collect();
        terms.push(r.log_x_final.exp());
        let total: f64 = crate::round::sum_up(&terms);
        let floor: f64 = crate::round::sum_down(&terms);
        assert!(floor <= 1.0 && total >= 1.0, "the prior mass is not bracketed: [{floor}, {total}]");
        assert!(total - floor < 1e-12, "and the bracket must be tight: {}", total - floor);
        // The last shell's recorded X is the same number the live block is weighed by.
        let last = r.shells.last().unwrap();
        assert_eq!(last.log_x, r.log_x_final, "the schedule must be one sequence, not two");
    }
}
