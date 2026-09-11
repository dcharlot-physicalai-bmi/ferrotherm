//! Stochastic series expansion: quantum Monte Carlo for the transverse-field Ising model, with no
//! Trotter error to extrapolate away.
//!
//! # What this is, and what it is the exact version of
//!
//! [`crate::sqa`] samples the transverse-field Ising model
//!
//! ```text
//!   H = - sum_(ij) J_ij sz_i sz_j  -  sum_i h_i sz_i  -  Gamma sum_i sx_i
//! ```
//!
//! through the Suzuki-Trotter decomposition: `M` classical replicas coupled along an extra
//! dimension, exact only as `M -> infinity`. That module's own documentation records what a badly
//! chosen `M` costs -- a single-spin magnetisation of 0.987 where the truth is 0.316, an error
//! larger than the quantity.
//!
//! Stochastic series expansion is the method that approximation approximates. Sandvik and
//! Kurkijaervi (Phys. Rev. B 43, 5950, 1991) expand `exp(-beta H)` in its own power series and
//! sample the terms; Sandvik's operator-loop formulation (Phys. Rev. B 59, R14157, 1999) made the
//! updates efficient, and Phys. Rev. E 68, 056701 (2003) specialises it to quantum Ising models
//! with arbitrary interactions, which is exactly this Hamiltonian. **Imaginary time is continuous
//! here.** There is no `M`, no discretisation parameter, and nothing to extrapolate: the only
//! approximation left is statistical, and it arrives with an error bar.
//!
//! # The representation
//!
//! Split `-H` into terms with non-negative matrix elements in the `sz` basis, plus a constant:
//!
//! ```text
//!   -H + C = sum_t H_t,      C = sum_b |J_b| + sum_i |h_i| + Gamma N
//!
//!   H_b     = |J_b| + J_b sz_i sz_j     (diagonal, weight 0 or 2|J_b|)
//!   H_i     = |h_i| + h_i sz_i          (diagonal, weight 0 or 2|h_i|)
//!   H_i^1   = Gamma                     (diagonal, weight Gamma)
//!   H_i^x   = Gamma sx_i                (off-diagonal, weight Gamma)
//! ```
//!
//! Then `Z = Tr exp(-beta H)` becomes a sum over **operator strings** of fixed length `L`, padded
//! with identities, carrying `n` non-identity operators and weight `beta^n (L-n)! / L!` times the
//! product of matrix elements. `L` is a cutoff, not a physical parameter: any `L` above the largest
//! `n` the chain reaches gives the same answer, which is why [`Params::cutoff`] is only a starting
//! point and the string grows itself while equilibrating.
//!
//! # The energy estimator, and why it is a count
//!
//! Differentiating the series in `beta` gives `<sum_t H_t> = <n> / beta`, so
//!
//! ```text
//!   E = C - <n> / beta
//! ```
//!
//! The mean expansion order **is** the energy. Nothing is measured on a state; the estimator is a
//! count, and its error bar is that count's error bar divided by `beta`.
//!
//! The same derivative in a coupling gives the rest. `H_i^x = Gamma sx_i` appears `n_x` times, so
//! `<sx_i> = <n_x,i> / (beta Gamma)`; `H_i^1 = Gamma` appears `n_1` times and its operator is the
//! identity, so `<n_1> = beta Gamma N` **exactly** -- a prediction with no free parameter, which is
//! what `the_diagonal_field_operator_count_is_beta_gamma_n_exactly` checks.
//!
//! # The off-diagonal update is a cluster update, and it is free
//!
//! `H_i^1` and `H_i^x` have the *same* weight `Gamma`, so swapping one for the other costs nothing.
//! That is the whole content of Sandvik's cluster update for quantum Ising models: field operators
//! CUT a site's imaginary-time line, bond operators LINK the two lines they touch, and each
//! resulting cluster of time-line segments flips with probability one half. Every field operator on
//! a cluster boundary then switches between its diagonal and off-diagonal form by itself. There is
//! no acceptance test to fail.
//!
//! A longitudinal field would break that, because `|h_i| + h_i sz_i` is 0 or `2|h_i|` and a flip
//! would zero it -- a cluster carrying one could never move. So the field is removed the way
//! [`crate::cluster::with_ghost`] removes it: one extra spin `s_g`, couplings `J_ig = h_i`, and the
//! physical spin read back as `sz_i = s_i s_g`. That is a change of variables and not an
//! approximation (`s_g^2 = 1` cancels out of both energy terms), it doubles `Z` in a way that
//! cancels in every expectation, and it leaves every cluster flip free.
//!
//! At `Gamma = 0` there are no field operators, every site's line is a single segment, and what is
//! left is Swendsen-Wang on the classical model -- clusters built from the bond operators the
//! string happens to hold, flipped at probability one half, with the field carried by the ghost.
//! `at_zero_transverse_field_it_is_exactly_the_classical_model` asserts the consequence: zero
//! off-diagonal operators, and the exact Boltzmann distribution of [`crate::ising`].
//!
//! # What is deliberately not here
//!
//! The **general** operator-loop and directed-loop updates of the 1999 paper are not implemented,
//! and are not missing. They exist to solve a problem this Hamiltonian does not have: when an
//! off-diagonal operator's weight depends on the spins it acts on -- a Heisenberg exchange, say --
//! a loop must be built leg by leg with locally computed exit probabilities, and choosing those
//! well is the entire subject of the directed-loop literature. Here both field operators weigh
//! `Gamma` whatever the spins are, so the "loop" is determined by the operator string alone and
//! flips at one half with no weight to balance. Writing a directed-loop solver for this model would
//! add a solver whose every answer is one half.
//!
//! Also absent: susceptibilities and the specific heat (`Var(n) - <n>` gives `C_v`, and the
//! imaginary-time-integrated correlator gives the longitudinal susceptibility), and any
//! multi-canonical or replica-exchange layer over this chain.
//!
//! ```
//! use ferrotherm::{graph::GraphBuilder, sse};
//!
//! // Four decoupled spins in a transverse field: E = -Gamma N tanh(beta Gamma), exactly.
//! let g = GraphBuilder::new(4).build();
//! let p = sse::Params { beta: 1.3, gamma: 0.7, ..sse::Params::default() };
//! let out = sse::run(&g, &p, 11).unwrap();
//! let exact = -p.gamma * 4.0 * (p.beta * p.gamma).tanh();
//! assert!((out.energy.value - exact).abs() < 4.0 * out.energy.stderr);
//! assert!(out.energy.value >= out.energy_floor, "and never below the spectrum's floor");
//! ```

use crate::graph::Graph;
use crate::ledger::Ledger;
use crate::rng::Pcg;
use crate::samples::Estimate;

/// Measured sweeps below which no error bar can be computed, so no run is allowed.
///
/// [`crate::certify::tau_int`] returns `NaN` under sixteen points, and an estimate whose
/// autocorrelation is unknown falls back to the naive `sqrt(var/N)` interval -- the one that
/// understates the error by `sqrt(2 tau)` and is the number that gets published. Refusing is the
/// honest answer.
pub const MIN_MEASURE: usize = 16;

/// Why a run was refused.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Invalid {
    /// A model with no spins. There is nothing to expand.
    Empty,
    /// Inverse temperature must be finite and strictly positive; the expansion is a series in it.
    Beta {
        /// The value supplied.
        beta: f64,
    },
    /// Transverse field must be finite and non-negative. Zero is allowed and is the classical
    /// limit; a negative one is the same spectrum under `sx -> -sx`, and is refused rather than
    /// silently accepted because reading it in a log means a sign was lost upstream.
    Gamma {
        /// The value supplied.
        gamma: f64,
    },
    /// A coupling or bias on this site is not finite, so no weight can be formed from it.
    NonFinite {
        /// The offending site.
        site: usize,
    },
    /// Fewer measured sweeps than an autocorrelation time can be estimated from.
    TooFewSweeps {
        /// The value supplied.
        measure: usize,
        /// [`MIN_MEASURE`].
        min: usize,
    },
}

impl core::fmt::Display for Invalid {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Invalid::Empty => write!(f, "a model with no spins has no series to expand"),
            Invalid::Beta { beta } => write!(
                f,
                "beta must be finite and positive, got {beta}: the expansion is a power series in \
                 beta, and beta = 0 is the infinite-temperature limit, which needs no sampler"
            ),
            Invalid::Gamma { gamma } => write!(
                f,
                "the transverse field must be finite and non-negative, got {gamma}: a negative \
                 field is the same spectrum under sx -> -sx, so this is a lost sign upstream"
            ),
            Invalid::NonFinite { site } => {
                write!(f, "site {site} carries a non-finite coupling or bias, so it has no weight")
            }
            Invalid::TooFewSweeps { measure, min } => write!(
                f,
                "{measure} measured sweeps is below {min}, and an autocorrelation time cannot be \
                 estimated from fewer -- the error bar would be the naive one"
            ),
        }
    }
}

impl core::error::Error for Invalid {}

/// How the chain is run.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Params {
    /// Inverse temperature. The expansion order grows roughly linearly in it, and so does the cost.
    pub beta: f64,
    /// Transverse field `Gamma`. Zero is the classical model, exactly.
    pub gamma: f64,
    /// Sweeps discarded, during which the string length adapts. See [`Params::cutoff`].
    pub equilibrate: usize,
    /// Sweeps measured. At least [`MIN_MEASURE`].
    pub measure: usize,
    /// Starting string length `L`. Only a starting point: it grows to `n + n/3 + 8` whenever the
    /// expansion order demands it, and **only while equilibrating**, so the measured chain samples
    /// one fixed ensemble.
    pub cutoff: usize,
}

impl Default for Params {
    fn default() -> Self {
        Params { beta: 1.0, gamma: 1.0, equilibrate: 2_000, measure: 10_000, cutoff: 32 }
    }
}

/// What the chain measured, every number with the error bar its own autocorrelation earns.
#[derive(Clone, Debug)]
pub struct Outcome {
    /// Total energy `<H>`, in the units and sign convention of [`Graph::energy`]: this is
    /// `C - <n>/beta`, and it is extensive rather than per-site.
    pub energy: Estimate,
    /// Longitudinal magnetisation per site, `<(1/N) sum_i sz_i>`, averaged over every propagation
    /// slot of the string rather than over one state per sweep.
    pub mz: Estimate,
    /// `<|(1/N) sum_i sz_i|>`, the order parameter that survives the spin-flip symmetry when every
    /// `h_i` is zero and [`Outcome::mz`] must average to nothing.
    pub mz_abs: Estimate,
    /// Transverse magnetisation per site, `<(1/N) sum_i sx_i> = <n_x> / (beta Gamma N)`. Exactly
    /// zero, with a zero-width interval, when `Gamma = 0`: a diagonal Hamiltonian has no `sx`.
    pub mx: Estimate,
    /// Mean expansion order `<n>`. The energy is this number and nothing else.
    pub order: Estimate,
    /// Mean count of the diagonal field operators `Gamma`. Theory fixes it at `beta Gamma N` with
    /// no free parameter, so a disagreement is a defect in the sampler and not a physical result.
    pub identity_ops: Estimate,
    /// String length `L` the run ended with.
    pub cutoff: usize,
    /// Largest expansion order seen while measuring.
    pub max_order: usize,
    /// Measured sweeps in which the expansion order reached the string length.
    ///
    /// **Non-zero means the result is biased**, because the series was truncated where it still
    /// carried weight. It is reported rather than asserted away so a caller can see how far the
    /// truncation was from binding; the adaptive growth normally leaves it at zero.
    pub saturated: u64,
    /// The final physical state `sz_i`, length `n`.
    pub state: Vec<i8>,
    /// A rigorous lower bound on every eigenvalue of `H`, and so on the energy at any temperature.
    /// See [`Sse::energy_floor`].
    pub energy_floor: f64,
}

/// One term of the operator string.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Op {
    /// Identity padding: the slot the diagonal update inserts into.
    Id,
    /// Diagonal `|J_b| + J_b sz_i sz_j`, indexing the bond table.
    Bond(u32),
    /// Diagonal `Gamma` on a site. Pairs with [`Op::Flip`] at equal weight.
    Const(u32),
    /// Off-diagonal `Gamma sx_i`: the only operator that moves a spin.
    Flip(u32),
}

/// An undirected coupling as the expansion sees it. Ghost bonds carry `h_i`.
#[derive(Clone, Copy, Debug)]
struct Bond {
    i: u32,
    j: u32,
    w: f64,
}

/// A diagonal operator the insertion step may propose.
#[derive(Clone, Copy, Debug)]
enum Cand {
    Bond(u32),
    Const(u32),
}

/// Where a field operator cuts a site's imaginary-time line.
#[derive(Clone, Copy, Debug)]
struct Cut {
    p: usize,
    site: u32,
    below: usize,
    above: usize,
}

/// A stochastic-series-expansion chain on one model at one temperature and field.
pub struct Sse<'g> {
    g: &'g Graph,
    beta: f64,
    gamma: f64,
    /// Real couplings first, then the ghost bonds carrying `h_i`.
    bonds: Vec<Bond>,
    /// Index of the ghost spin, present only when some `h_i` is non-zero.
    ghost: Option<usize>,
    /// Spins carried: the model's, plus the ghost.
    nsite: usize,
    /// `C = sum_b |J_b| + sum_i |h_i| + Gamma N`, rounded UP. See [`Sse::energy_floor`].
    shift: f64,
    cand: Vec<Cand>,
    /// Prefix sums of the candidates' maximum weights, for the fixed proposal distribution.
    cum: Vec<f64>,
    /// `sum_t w_t^max`, the state-independent total the acceptances are written in.
    w_max: f64,
    s: Vec<i8>,
    op: Vec<Op>,
    n: usize,
    rng: Pcg,
    // Scratch, kept across sweeps so that a sweep allocates nothing.
    work: Vec<i8>,
    parent: Vec<usize>,
    seg_spin: Vec<i8>,
    cur: Vec<usize>,
    cuts: Vec<Cut>,
    decision: Vec<u8>,
}

/// Union-find root, with path halving.
fn find(parent: &mut [usize], mut x: usize) -> usize {
    while parent[x] != x {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    x
}

/// Merge two segments, the smaller index becoming the root so the forest is reproducible.
fn join(parent: &mut [usize], a: usize, b: usize) {
    let (ra, rb) = (find(parent, a), find(parent, b));
    if ra != rb {
        parent[ra.max(rb)] = ra.min(rb);
    }
}

/// A mean with the error bar its own trace earns, mirroring [`crate::samples::SampleSet`].
///
/// The standard error is `sqrt(var / ess)` with `ess = N / (2 tau)`, never `sqrt(var / N)`:
/// successive sweeps of one chain share a configuration, and the naive interval understates the
/// error by `sqrt(2 tau)`.
fn estimate(trace: &[f64]) -> Estimate {
    let n = trace.len();
    let value = trace.iter().sum::<f64>() / n as f64;
    let var = if n > 1 {
        trace.iter().map(|v| (v - value).powi(2)).sum::<f64>() / (n - 1) as f64
    } else {
        0.0
    };
    let tau = crate::certify::tau_int(trace);
    // An infinite tau is a frozen trace, and it is left in the estimate rather than sanitised so
    // that reading `tau_int.is_finite()` still catches one -- the choice `samples` documents.
    let ess = if tau.is_finite() && tau > 0.0 { n as f64 / (2.0 * tau) } else { 1.0 };
    let stderr = if ess.is_finite() && ess > 0.0 { (var / ess).sqrt() } else { 0.0 };
    Estimate { value, stderr, ess, tau_int: tau }
}

/// An affine image of an estimate, `add + mul * e`, carrying the interval exactly.
///
/// The energy is `C - <n>/beta` and the transverse magnetisation is `<n_x>/(beta Gamma N)`. Both
/// are linear in a count, so their error bars are that count's error bar scaled: nothing is
/// re-derived and nothing is lost.
fn affine(e: &Estimate, mul: f64, add: f64) -> Estimate {
    Estimate {
        value: add + mul * e.value,
        stderr: mul.abs() * e.stderr,
        ess: e.ess,
        tau_int: e.tau_int,
    }
}

impl<'g> Sse<'g> {
    /// Build a chain. Nothing is sampled until [`Sse::sweep`] is called.
    ///
    /// # Errors
    ///
    /// [`Invalid`] for an empty model, a non-positive or non-finite `beta`, a negative or
    /// non-finite `Gamma`, a non-finite coupling or bias, or fewer than [`MIN_MEASURE`] measured
    /// sweeps.
    pub fn new(g: &'g Graph, p: &Params, seed: u64) -> Result<Sse<'g>, Invalid> {
        if g.n == 0 {
            return Err(Invalid::Empty);
        }
        if !(p.beta > 0.0) || !p.beta.is_finite() {
            return Err(Invalid::Beta { beta: p.beta });
        }
        if !(p.gamma >= 0.0) || !p.gamma.is_finite() {
            return Err(Invalid::Gamma { gamma: p.gamma });
        }
        if p.measure < MIN_MEASURE {
            return Err(Invalid::TooFewSweeps { measure: p.measure, min: MIN_MEASURE });
        }
        for i in 0..g.n {
            if !g.h[i].is_finite() {
                return Err(Invalid::NonFinite { site: i });
            }
            for k in g.offset[i]..g.offset[i + 1] {
                if !g.w[k].is_finite() {
                    return Err(Invalid::NonFinite { site: i });
                }
            }
        }

        let has_field = g.h.iter().any(|&h| h != 0.0);
        let ghost = has_field.then_some(g.n);
        let nsite = g.n + usize::from(has_field);

        let mut bonds: Vec<Bond> = Vec::with_capacity(g.n_edges + g.n);
        for i in 0..g.n {
            for k in g.offset[i]..g.offset[i + 1] {
                let j = g.nbr[k] as usize;
                if j > i {
                    bonds.push(Bond { i: i as u32, j: j as u32, w: g.w[k] });
                }
            }
        }
        if let Some(gi) = ghost {
            for i in 0..g.n {
                if g.h[i] != 0.0 {
                    bonds.push(Bond { i: i as u32, j: gi as u32, w: g.h[i] });
                }
            }
        }

        // The constant offset, accumulated UPWARD on purpose: `energy_floor` is `-shift` and is
        // claimed as a bound on the spectrum, so `shift` must never come out below the exact sum.
        // Round-to-nearest addition lands either side, and a bound on the wrong side of what it
        // bounds is not a bound -- see `crate::round`, which exists because this crate shipped one.
        let mut terms: Vec<f64> = bonds.iter().map(|b| b.w.abs()).collect();
        terms.push(p.gamma * g.n as f64);
        let shift = crate::round::sum_up(&terms);

        let mut cand: Vec<Cand> = Vec::with_capacity(bonds.len() + g.n);
        let mut cum: Vec<f64> = Vec::with_capacity(bonds.len() + g.n);
        let mut acc = 0.0;
        for (b, bond) in bonds.iter().enumerate() {
            if bond.w != 0.0 {
                acc += 2.0 * bond.w.abs();
                cand.push(Cand::Bond(b as u32));
                cum.push(acc);
            }
        }
        if p.gamma > 0.0 {
            for i in 0..g.n {
                acc += p.gamma;
                cand.push(Cand::Const(i as u32));
                cum.push(acc);
            }
        }

        let mut rng = Pcg::new(seed, 0x0005_5E00);
        let s: Vec<i8> = (0..nsite).map(|_| rng.spin(0.5)).collect();
        let cutoff = p.cutoff.max(8);
        Ok(Sse {
            g,
            beta: p.beta,
            gamma: p.gamma,
            bonds,
            ghost,
            nsite,
            shift,
            cand,
            cum,
            w_max: acc,
            work: s.clone(),
            s,
            op: vec![Op::Id; cutoff],
            n: 0,
            rng,
            parent: Vec::new(),
            seg_spin: Vec::new(),
            cur: Vec::new(),
            cuts: Vec::new(),
            decision: Vec::new(),
        })
    }

    /// A rigorous lower bound on every eigenvalue of `H`, and so on the energy at any temperature.
    ///
    /// `H` is a sum of terms whose operator norms are `|J_b|`, `|h_i|` and `Gamma`, so the triangle
    /// inequality puts the whole spectrum inside `[-C, +C]` with
    /// `C = sum_b |J_b| + sum_i |h_i| + Gamma N`. That is the same `C` the energy estimator
    /// subtracts from, and it is why the sum is accumulated with [`crate::round::sum_up`]: an upper
    /// bound on `C` is a lower bound on `-C`, and the direction has to be certain for the claim to
    /// be one.
    ///
    /// It is coarse and is not trying to be tight. It is a floor a sampled energy may be checked
    /// against, and a check that can fail is worth more than a number that cannot.
    #[must_use]
    pub fn energy_floor(&self) -> f64 {
        -self.shift
    }

    /// The physical state `sz_i = s_i s_g`, length `n`.
    #[must_use]
    pub fn state(&self) -> Vec<i8> {
        let gs = self.ghost.map_or(1i8, |gi| self.s[gi]);
        self.s[..self.g.n].iter().map(|&v| v * gs).collect()
    }

    /// Non-identity operators currently in the string.
    #[must_use]
    pub fn order(&self) -> usize {
        self.n
    }

    /// The inverse temperature this chain runs at. `E = C - <n>/beta` needs it, and a caller
    /// driving [`Sse::sweep`] by hand should not have to carry the number separately.
    #[must_use]
    pub fn beta(&self) -> f64 {
        self.beta
    }

    /// The transverse field this chain runs at. Zero is the classical model.
    #[must_use]
    pub fn gamma(&self) -> f64 {
        self.gamma
    }

    /// Current string length `L`.
    #[must_use]
    pub fn cutoff(&self) -> usize {
        self.op.len()
    }

    /// One sweep: a diagonal pass over every slot of the string, then one cluster pass.
    pub fn sweep(&mut self) {
        self.diagonal_update();
        self.cluster_update();
    }

    /// Grow the string to `n + n/3 + 8` if the expansion order has outgrown it.
    ///
    /// Legitimate only while equilibrating. `L` is a cutoff and every `L` above the reachable order
    /// gives the same physics, but changing it *during* measurement would average two different
    /// truncations together, so [`run`] stops adapting the moment it starts measuring.
    pub fn adapt(&mut self) {
        let want = self.n + self.n / 3 + 8;
        if want > self.op.len() {
            self.op.resize(want, Op::Id);
        }
    }

    /// Insert and remove diagonal operators, propagating the spins through the off-diagonal ones.
    ///
    /// Metropolis-Hastings in the expansion order. Inserting into a free slot multiplies the weight
    /// by `beta w_t / (L - n)`; proposing `t` from the FIXED distribution `w_t^max / W_max` makes
    /// the acceptance `min(1, beta W_max / (L - n))` whenever the chosen operator has non-zero
    /// weight in the current state, and zero when it has not. The reverse move accepts at
    /// `min(1, (L - n + 1) / (beta W_max))`. Writing both in the state-INDEPENDENT `W_max` is what
    /// makes the pair balance exactly; the state enters only through which proposals are possible.
    fn diagonal_update(&mut self) {
        let l = self.op.len();
        if self.cand.is_empty() || self.w_max <= 0.0 {
            return;
        }
        self.work.copy_from_slice(&self.s);
        for p in 0..l {
            match self.op[p] {
                Op::Id => {
                    let u = self.rng.f64() * self.w_max;
                    let k = self.cum.partition_point(|&c| c <= u).min(self.cand.len() - 1);
                    let (op, live) = match self.cand[k] {
                        Cand::Bond(b) => {
                            let bond = self.bonds[b as usize];
                            let prod = f64::from(self.work[bond.i as usize])
                                * f64::from(self.work[bond.j as usize]);
                            (Op::Bond(b), bond.w * prod > 0.0)
                        }
                        Cand::Const(i) => (Op::Const(i), true),
                    };
                    if live {
                        let acc = self.beta * self.w_max / (l - self.n) as f64;
                        if acc >= 1.0 || self.rng.f64() < acc {
                            self.op[p] = op;
                            self.n += 1;
                        }
                    }
                }
                Op::Bond(_) | Op::Const(_) => {
                    let acc = (l - self.n + 1) as f64 / (self.beta * self.w_max);
                    if acc >= 1.0 || self.rng.f64() < acc {
                        self.op[p] = Op::Id;
                        self.n -= 1;
                    }
                }
                Op::Flip(i) => {
                    let i = i as usize;
                    self.work[i] = -self.work[i];
                }
            }
        }
    }

    /// Sandvik's cluster update: cut the time lines at field operators, link them at bond
    /// operators, flip each resulting cluster at probability one half.
    ///
    /// No acceptance test appears because none is needed. A bond operator's weight depends on
    /// `sz_i sz_j` and both its legs lie in one cluster, so a flip leaves it alone; a field
    /// operator weighs `Gamma` in whichever of its two forms it takes. The ghost spin is what keeps
    /// that true under a longitudinal field -- see the module note.
    fn cluster_update(&mut self) {
        let l = self.op.len();
        let ns = self.nsite;
        self.parent.clear();
        self.parent.extend(0..ns);
        self.seg_spin.clear();
        self.seg_spin.extend_from_slice(&self.s);
        self.cur.clear();
        self.cur.extend(0..ns);
        self.cuts.clear();

        for p in 0..l {
            match self.op[p] {
                Op::Id => {}
                Op::Bond(b) => {
                    let bond = self.bonds[b as usize];
                    let (a, c) = (self.cur[bond.i as usize], self.cur[bond.j as usize]);
                    join(&mut self.parent, a, c);
                }
                Op::Const(i) | Op::Flip(i) => {
                    let site = i as usize;
                    let below = self.cur[site];
                    let above = self.parent.len();
                    self.parent.push(above);
                    let sp = if matches!(self.op[p], Op::Flip(_)) {
                        -self.seg_spin[below]
                    } else {
                        self.seg_spin[below]
                    };
                    self.seg_spin.push(sp);
                    self.cuts.push(Cut { p, site: i, below, above });
                    self.cur[site] = above;
                }
            }
        }
        // The string is a trace: each line's last segment is its first one, come round again.
        for i in 0..ns {
            let last = self.cur[i];
            join(&mut self.parent, last, i);
        }

        let nseg = self.parent.len();
        self.decision.clear();
        self.decision.resize(nseg, 0);
        for seg in 0..nseg {
            let r = find(&mut self.parent, seg);
            if self.decision[r] == 0 {
                self.decision[r] = if self.rng.f64() < 0.5 { 2 } else { 1 };
            }
        }
        for seg in 0..nseg {
            let r = find(&mut self.parent, seg);
            if self.decision[r] == 2 {
                self.seg_spin[seg] = -self.seg_spin[seg];
            }
        }

        self.s[..ns].copy_from_slice(&self.seg_spin[..ns]);
        for k in 0..self.cuts.len() {
            let c = self.cuts[k];
            self.op[c.p] = if self.seg_spin[c.below] == self.seg_spin[c.above] {
                Op::Const(c.site)
            } else {
                Op::Flip(c.site)
            };
        }
    }

    /// Count the two field-operator species in the string: off-diagonal first.
    fn field_counts(&self) -> (usize, usize) {
        let mut flips = 0;
        let mut consts = 0;
        for op in &self.op {
            match op {
                Op::Flip(_) => flips += 1,
                Op::Const(_) => consts += 1,
                _ => {}
            }
        }
        (flips, consts)
    }

    /// Magnetisation averaged over every propagation slot of the string, and its absolute value.
    ///
    /// A diagonal observable may be read at any one slot -- the weight is invariant under a cyclic
    /// shift of the string, so every slot carries the same marginal -- and averaging over all `L`
    /// of them is the same estimator with less variance. It costs one walk, because only
    /// [`Op::Flip`] moves a spin.
    fn measure_trajectory(&mut self) -> (f64, f64) {
        let l = self.op.len();
        self.work.copy_from_slice(&self.s);
        let mut msum: i64 = self.work[..self.g.n].iter().map(|&v| i64::from(v)).sum();
        let gs = self.ghost.map_or(1i64, |gi| i64::from(self.work[gi]));
        let nn = self.g.n as f64;
        let (mut acc, mut acc_abs) = (0.0f64, 0.0f64);
        for p in 0..l {
            let m = (msum * gs) as f64 / nn;
            acc += m;
            acc_abs += m.abs();
            if let Op::Flip(i) = self.op[p] {
                let i = i as usize;
                msum -= 2 * i64::from(self.work[i]);
                self.work[i] = -self.work[i];
            }
        }
        (acc / l as f64, acc_abs / l as f64)
    }
}

/// Run a chain and report what it measured.
///
/// # Errors
///
/// [`Invalid`], as [`Sse::new`].
pub fn run(g: &Graph, p: &Params, seed: u64) -> Result<Outcome, Invalid> {
    run_metered(g, p, seed, None)
}

/// As [`run`], charging the string sweeps and the state read-outs to a [`Ledger`].
///
/// One slot of the operator string visited is one charged sample: the diagonal pass makes exactly
/// one accept-or-reject decision per slot, which is the unit a thermodynamic sampling unit bills.
/// Each measured sweep also charges one read per spin, because the magnetisation leaves the chip.
///
/// # Errors
///
/// [`Invalid`], as [`Sse::new`].
pub fn run_metered(
    g: &Graph,
    p: &Params,
    seed: u64,
    mut ledger: Option<&mut Ledger>,
) -> Result<Outcome, Invalid> {
    let mut sse = Sse::new(g, p, seed)?;
    for _ in 0..p.equilibrate {
        sse.sweep();
        sse.adapt();
        if let Some(l) = ledger.as_deref_mut() {
            l.samples += sse.op.len() as u64;
        }
    }

    let mut order = Vec::with_capacity(p.measure);
    let mut flips = Vec::with_capacity(p.measure);
    let mut consts = Vec::with_capacity(p.measure);
    let mut mz = Vec::with_capacity(p.measure);
    let mut mz_abs = Vec::with_capacity(p.measure);
    let (mut saturated, mut max_order) = (0u64, 0usize);
    for _ in 0..p.measure {
        sse.sweep();
        let (nf, nc) = sse.field_counts();
        let (m, ma) = sse.measure_trajectory();
        if sse.n >= sse.op.len() {
            saturated += 1;
        }
        max_order = max_order.max(sse.n);
        order.push(sse.n as f64);
        flips.push(nf as f64);
        consts.push(nc as f64);
        mz.push(m);
        mz_abs.push(ma);
        if let Some(l) = ledger.as_deref_mut() {
            l.samples += sse.op.len() as u64;
            l.reads += g.n as u64;
        }
    }

    let order = estimate(&order);
    let flips = estimate(&flips);
    let energy = affine(&order, -1.0 / p.beta, sse.shift);
    let mx = if p.gamma > 0.0 {
        affine(&flips, 1.0 / (p.beta * p.gamma * g.n as f64), 0.0)
    } else {
        // Not estimated, and deliberately given no interval: `sx` is purely off-diagonal, so its
        // expectation under a diagonal Hamiltonian is zero identically rather than on average.
        Estimate { value: 0.0, stderr: 0.0, ess: f64::INFINITY, tau_int: f64::NAN }
    };
    Ok(Outcome {
        energy,
        mz: estimate(&mz),
        mz_abs: estimate(&mz_abs),
        mx,
        order,
        identity_ops: estimate(&consts),
        cutoff: sse.op.len(),
        max_order,
        saturated,
        state: sse.state(),
        energy_floor: sse.energy_floor(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::GraphBuilder;
    use crate::ising;

    /// Exact diagonalisation of the transverse-field Ising model: energy, `mz`, `mx`, and the
    /// ground eigenvalue.
    ///
    /// **This is the oracle, and it shares no line with the sampler.** The matrix is built from
    /// [`Graph::energy`] — the crate's energy convention, not a restatement of it — plus `-Gamma`
    /// in every single-spin-flip off-diagonal entry, and diagonalised by
    /// [`crate::linalg::jacobi_eig`], which carries its own reconstruction and scale-invariance
    /// tests. Nothing in this module is consulted, so an error here and an error above cannot be
    /// the same error.
    fn ed(g: &Graph, gamma: f64, beta: f64) -> (f64, f64, f64, f64) {
        let n = g.n;
        let dim = 1usize << n;
        let mut hm = vec![0.0f64; dim * dim];
        let mut s = vec![0i8; n];
        for a in 0..dim {
            for b in 0..n {
                s[b] = if (a >> b) & 1 == 1 { 1 } else { -1 };
            }
            hm[a * dim + a] = g.energy(&s);
            for i in 0..n {
                hm[(a ^ (1 << i)) * dim + a] -= gamma;
            }
        }
        let v = crate::linalg::jacobi_eig(&mut hm, dim);
        let lam: Vec<f64> = (0..dim).map(|c| hm[c * dim + c]).collect();
        let lo = lam.iter().copied().fold(f64::INFINITY, f64::min);
        // Shifted so the largest exponent is zero: `exact_boltzmann` learned the same lesson.
        let w: Vec<f64> = lam.iter().map(|&l| (-beta * (l - lo)).exp()).collect();
        let z: f64 = w.iter().sum();
        let (mut e, mut mz, mut mx) = (0.0, 0.0, 0.0);
        for c in 0..dim {
            e += w[c] * lam[c];
            let (mut ez, mut ex) = (0.0, 0.0);
            for a in 0..dim {
                let va = v[a * dim + c];
                let mut sz = 0i32;
                for b in 0..n {
                    sz += if (a >> b) & 1 == 1 { 1 } else { -1 };
                }
                ez += va * va * f64::from(sz);
                for i in 0..n {
                    ex += va * v[(a ^ (1 << i)) * dim + c];
                }
            }
            mz += w[c] * ez;
            mx += w[c] * ex;
        }
        (e / z, mz / z / n as f64, mx / z / n as f64, lo)
    }

    /// Four standard errors, not two, and the reason is stated rather than tuned.
    ///
    /// A 95% interval fails one assertion in twenty by construction, and these tests make dozens of
    /// them; at four sigma the family-wise failure rate is under a part in a thousand while the
    /// band is still far narrower than any systematic error a wrong estimator would produce. Each
    /// call also asserts the error bar is SMALL, because "inside four sigma" says nothing when
    /// sigma is large — that is the shape of a check that cannot fail.
    fn agrees(got: &Estimate, want: f64, tight: f64, what: &str) {
        assert!(
            got.stderr < tight,
            "{what}: the error bar is {:.4} and this test is only a test while it is below {tight} \
             -- take more sweeps rather than widening it",
            got.stderr
        );
        assert!(
            (got.value - want).abs() <= 4.0 * got.stderr,
            "{what}: sampled {:.6} +- {:.6}, exact diagonalisation says {want:.6} ({:.1} sigma)",
            got.value,
            got.stderr,
            (got.value - want).abs() / got.stderr.max(f64::MIN_POSITIVE)
        );
    }

    /// THE ORACLE IS CHECKED AGAINST A CLOSED FORM BEFORE ANY SAMPLER IS COMPARED TO IT.
    ///
    /// With no couplings the sites decouple and every quantity is elementary:
    /// `E = -Gamma N tanh(beta Gamma)`, `<sx> = tanh(beta Gamma)`, `<sz> = 0`, and the ground
    /// energy is exactly `-Gamma N`. A dense eigensolve that got any of those wrong would make
    /// every comparison below agree with the wrong number.
    #[test]
    fn the_exact_diagonalisation_oracle_reproduces_the_decoupled_closed_form() {
        for n in [1usize, 2, 4] {
            let g = GraphBuilder::new(n).build();
            for &(beta, gamma) in &[(0.5f64, 1.0f64), (2.0, 0.3), (1.0, 2.0)] {
                let (e, mz, mx, lo) = ed(&g, gamma, beta);
                let want = -gamma * n as f64 * (beta * gamma).tanh();
                assert!((e - want).abs() < 1e-10, "n={n} beta={beta} gamma={gamma}: {e} vs {want}");
                assert!(mz.abs() < 1e-10, "no field, no longitudinal magnetisation: {mz}");
                assert!((mx - (beta * gamma).tanh()).abs() < 1e-10, "mx {mx}");
                assert!((lo + gamma * n as f64).abs() < 1e-10, "ground energy {lo}");
            }
        }
    }

    /// TWO SITES, AND THE ORACLE IS THE WHOLE 4x4 SPECTRUM.
    ///
    /// The smallest system where the transverse field and the coupling compete, run at five
    /// `(beta, Gamma)` points spanning the classical-looking and the field-dominated ends. Energy
    /// and transverse magnetisation both come from the dense eigensolve.
    #[test]
    fn energy_matches_exact_diagonalisation_of_a_two_site_chain() {
        let mut gb = GraphBuilder::new(2);
        gb.couple(0, 1, 1.0);
        let g = gb.build();
        for &(beta, gamma) in &[(0.5f64, 0.5f64), (1.0, 1.0), (2.0, 0.4), (1.5, 2.0), (3.0, 1.0)] {
            let p = Params { beta, gamma, equilibrate: 2_000, measure: 150_000, cutoff: 16 };
            let out = run(&g, &p, 7).unwrap();
            let (e, _, mx, lo) = ed(&g, gamma, beta);
            assert_eq!(out.saturated, 0, "beta={beta} gamma={gamma}: the string truncated");
            agrees(&out.energy, e, 0.014, &format!("E at beta={beta} gamma={gamma}"));
            agrees(&out.mx, mx, 0.007, &format!("mx at beta={beta} gamma={gamma}"));
            assert!(out.energy_floor <= lo, "floor {} above lambda_min {lo}", out.energy_floor);
        }
    }

    /// FOUR SITES IN A RING, WITH A LONGITUDINAL FIELD, AGAINST THE 16x16 SPECTRUM.
    ///
    /// The field is what makes this more than a bigger version of the two-site test: it is carried
    /// by the ghost spin, so `mz` is non-zero and a ghost that were wired up backwards would show
    /// as a sign here. It also puts the antiferromagnetic case on the same footing as the
    /// ferromagnetic one — the bond operator's weight is `|J| + J sz sz` either way.
    ///
    /// `(3.0, 0.2)` is in the list for a reason that is not physics. The removal acceptance
    /// `(L - n + 1) / (beta W_max)` is above one, and so clamps, at every warm point here: with the
    /// string adapted to `4n/3 + 8` the free slots outnumber `beta W_max` until the expansion order
    /// passes about fourteen. Cold and nearly classical is where that ratio drops below one and the
    /// formula starts deciding something, and a branch no test reaches is a branch no test checks.
    #[test]
    fn energy_and_magnetisations_match_exact_diagonalisation_of_a_four_site_ring() {
        for &(j, h) in &[(1.0f64, 0.0f64), (1.0, 0.45), (-1.0, 0.3)] {
            let g = ising::ring(4, j, h);
            for &(beta, gamma) in &[(0.8f64, 1.0f64), (2.0, 0.5), (1.2, 1.8), (3.0, 0.2)] {
                let p = Params { beta, gamma, equilibrate: 2_000, measure: 150_000, cutoff: 16 };
                let out = run(&g, &p, 19).unwrap();
                let (e, mz, mx, lo) = ed(&g, gamma, beta);
                let tag = format!("J={j} h={h} beta={beta} gamma={gamma}");
                assert_eq!(out.saturated, 0, "{tag}: the string truncated");
                agrees(&out.energy, e, 0.020, &format!("E at {tag}"));
                agrees(&out.mx, mx, 0.004, &format!("mx at {tag}"));
                agrees(&out.mz, mz, 0.004, &format!("mz at {tag}"));
                assert!(out.energy_floor <= lo, "{tag}: floor {} above {lo}", out.energy_floor);
            }
        }
    }

    /// DECOUPLED SITES HAVE AN EXACT ANSWER AND IT IS ASSERTED AS ONE.
    ///
    /// `E = -Gamma N tanh(beta Gamma)` and `<sx> = tanh(beta Gamma)`. No eigensolve, no
    /// enumeration, nothing but the closed form — the single-spin problem is two levels split by
    /// `2 Gamma`.
    #[test]
    fn decoupled_sites_match_minus_gamma_n_tanh_beta_gamma() {
        let g = GraphBuilder::new(6).build();
        for &(beta, gamma) in &[(0.4f64, 1.0f64), (1.0, 1.0), (2.5, 0.7), (1.0, 3.0)] {
            let p = Params { beta, gamma, equilibrate: 1_000, measure: 150_000, cutoff: 16 };
            let out = run(&g, &p, 23).unwrap();
            let want = -gamma * 6.0 * (beta * gamma).tanh();
            agrees(&out.energy, want, 0.035, &format!("E at beta={beta} gamma={gamma}"));
            agrees(&out.mx, (beta * gamma).tanh(), 0.004, &format!("mx at beta={beta}"));
            assert!(out.mz.value.abs() < 6.0 * out.mz.stderr, "mz {:?}", out.mz);
        }
    }

    /// AT `Gamma = 0` THIS IS THE CLASSICAL MODEL, AND THE ORACLE IS THE ENUMERATION.
    ///
    /// Two claims, and the first is exact rather than statistical: with no transverse field the
    /// string can hold no off-diagonal operator at all, at any order, in any sweep. The second is
    /// the distribution itself against [`ising::exact_boltzmann`], with a control — the distance
    /// from the uniform distribution to the truth — so the tolerance cannot be satisfied by a
    /// sampler that is merely producing states.
    #[test]
    fn at_zero_transverse_field_it_is_exactly_the_classical_model() {
        let g = ising::ring(4, 1.0, 0.35);
        let beta = 0.8;
        let draws = 200_000usize;
        let p = Params { beta, gamma: 0.0, equilibrate: 2_000, measure: draws, cutoff: 16 };
        let mut sse = Sse::new(&g, &p, 5).unwrap();
        for _ in 0..p.equilibrate {
            sse.sweep();
            sse.adapt();
        }
        let mut hist = vec![0u64; 1 << g.n];
        for _ in 0..draws {
            sse.sweep();
            assert_eq!(sse.field_counts(), (0, 0), "Gamma = 0 admits no field operator");
            let s = sse.state();
            let mask = (0..g.n).filter(|&i| s[i] == 1).fold(0usize, |m, i| m | (1 << i));
            hist[mask] += 1;
        }
        let got: Vec<f64> = hist.iter().map(|&c| c as f64 / draws as f64).collect();
        let want = ising::exact_boltzmann(&g, beta);
        let tv = ising::tv(&got, &want);
        let unif = vec![1.0 / (1 << g.n) as f64; 1 << g.n];
        let control = ising::tv(&unif, &want);
        assert!(
            control > 20.0 * tv.max(1e-12),
            "the tolerance is only meaningful while a wrong distribution is far outside it: \
             sampled {tv:.5}, uniform {control:.5}"
        );
        assert!(tv < 0.01, "total variation from the exact Boltzmann law is {tv:.5}");

        // And the energy estimator reduces with it: C - <n>/beta against the enumerated mean.
        let out = run(&g, &p, 5).unwrap();
        let mut s = vec![0i8; g.n];
        let mut exact = 0.0;
        for (mask, wk) in want.iter().enumerate() {
            for b in 0..g.n {
                s[b] = if (mask >> b) & 1 == 1 { 1 } else { -1 };
            }
            exact += wk * g.energy(&s);
        }
        agrees(&out.energy, exact, 0.015, "E at Gamma = 0");
        assert_eq!(out.mx.value, 0.0, "a diagonal Hamiltonian has <sx> = 0 identically");
        assert_eq!(out.mx.stderr, 0.0, "and that zero is not an estimate");
    }

    /// AN EXACT IDENTITY WITH NO FREE PARAMETER: `<n_1> = beta Gamma N`.
    ///
    /// The diagonal field operator IS the identity times `Gamma`, so differentiating the series in
    /// its coefficient gives `<n_1> / (beta Gamma) = <1> = 1` per site. It holds at every `beta`,
    /// every `Gamma`, and on every model — which makes it a check on the operator bookkeeping that
    /// no physics can excuse a failure of.
    #[test]
    fn the_diagonal_field_operator_count_is_beta_gamma_n_exactly() {
        for &(j, h) in &[(0.0f64, 0.0f64), (1.0, 0.4), (-0.7, 0.0)] {
            for &(beta, gamma) in &[(0.7f64, 1.3f64), (2.0, 0.6)] {
                let g = ising::ring(5, j, h);
                let p = Params { beta, gamma, equilibrate: 1_000, measure: 150_000, cutoff: 16 };
                let out = run(&g, &p, 31).unwrap();
                let want = beta * gamma * g.n as f64;
                agrees(&out.identity_ops, want, 0.003 * want, &format!("<n_1> at J={j} h={h}"));
            }
        }
    }

    /// THE FLOOR IS A BOUND, SO IT IS CHECKED AS ONE — AND IT IS ATTAINED.
    ///
    /// `-C` is below every eigenvalue by the triangle inequality, and for a field-free ferromagnet
    /// at `Gamma = 0` it is EXACTLY the ground energy, because the all-aligned state saturates
    /// every bond at once. So the test is not "the bound holds somewhere loose": at that point it
    /// must sit within rounding of the truth and never above it, which is precisely the direction
    /// [`crate::round::sum_up`] is there to guarantee.
    #[test]
    fn the_energy_floor_is_a_sound_lower_bound_and_is_attained_at_zero_field() {
        let p = Params { measure: MIN_MEASURE, ..Params::default() };
        for l in [2usize, 3, 4] {
            let g = ising::ring(l.max(3), 1.0, 0.0);
            let sse = Sse::new(&g, &Params { gamma: 0.0, ..p }, 1).unwrap();
            let (_, _, _, lo) = ed(&g, 0.0, 1.0);
            assert!(sse.energy_floor() <= lo, "floor {} above {lo}", sse.energy_floor());
            assert!(
                lo - sse.energy_floor() < 1e-9,
                "a ferromagnet at Gamma = 0 attains the floor: {} against {lo}",
                sse.energy_floor()
            );
        }
        // And it stays sound once a field and a transverse term are switched on.
        for &(j, h, gamma) in &[(1.0f64, 0.5f64, 1.0f64), (-1.0, 0.0, 2.0), (0.6, -0.9, 0.3)] {
            let g = ising::ring(4, j, h);
            let sse = Sse::new(&g, &Params { gamma, ..p }, 1).unwrap();
            let (_, _, _, lo) = ed(&g, gamma, 1.0);
            assert!(
                sse.energy_floor() <= lo,
                "J={j} h={h} gamma={gamma}: floor {} above lambda_min {lo}",
                sse.energy_floor()
            );
        }
    }

    /// THE STRING IS A TRACE, AND A CLUSTER UPDATE THAT BROKE THAT WOULD STILL RETURN NUMBERS.
    ///
    /// Propagating the whole operator string from the stored state must return to the stored state
    /// — every site flipped an even number of times — or the configuration is not a diagonal matrix
    /// element of anything and every estimator above is meaningless. The order counter is checked
    /// against a recount for the same reason.
    #[test]
    fn the_operator_string_stays_a_closed_trace() {
        let g = ising::ring(6, 1.0, 0.3);
        let p = Params { beta: 2.0, gamma: 1.1, ..Params::default() };
        let mut sse = Sse::new(&g, &p, 77).unwrap();
        for sweep in 0..500 {
            sse.sweep();
            sse.adapt();
            let mut s = sse.s.clone();
            let mut flips = vec![0usize; sse.nsite];
            let mut count = 0;
            for op in &sse.op {
                match *op {
                    Op::Id => {}
                    Op::Flip(i) => {
                        count += 1;
                        s[i as usize] = -s[i as usize];
                        flips[i as usize] += 1;
                    }
                    _ => count += 1,
                }
            }
            assert_eq!(count, sse.n, "sweep {sweep}: the order counter drifted from the string");
            assert_eq!(s, sse.s, "sweep {sweep}: the string does not close on itself");
            for (i, f) in flips.iter().enumerate() {
                assert_eq!(f % 2, 0, "sweep {sweep}: site {i} is flipped {f} times, an odd number");
            }
            if let Some(gi) = sse.ghost {
                assert_eq!(flips[gi], 0, "the ghost spin carries no transverse field");
            }
        }
        assert!(sse.n > 0, "500 sweeps at beta = 2 and Gamma = 1.1 produced an empty string");
    }

    /// A refusal is a named variant, and each of them is reachable.
    #[test]
    fn bad_parameters_are_refused_by_variant() {
        let g = ising::ring(4, 1.0, 0.0);
        let ok = Params::default();
        assert_eq!(Sse::new(&GraphBuilder::new(0).build(), &ok, 1).err(), Some(Invalid::Empty));
        assert_eq!(
            Sse::new(&g, &Params { beta: 0.0, ..ok }, 1).err(),
            Some(Invalid::Beta { beta: 0.0 })
        );
        assert!(matches!(
            Sse::new(&g, &Params { beta: f64::NAN, ..ok }, 1).err(),
            Some(Invalid::Beta { .. })
        ));
        assert_eq!(
            Sse::new(&g, &Params { gamma: -1.0, ..ok }, 1).err(),
            Some(Invalid::Gamma { gamma: -1.0 })
        );
        assert_eq!(
            Sse::new(&g, &Params { measure: 4, ..ok }, 1).err(),
            Some(Invalid::TooFewSweeps { measure: 4, min: MIN_MEASURE })
        );
        let mut gb = GraphBuilder::new(3);
        gb.couple(0, 1, f64::INFINITY);
        assert_eq!(
            Sse::new(&gb.build(), &ok, 1).err(),
            Some(Invalid::NonFinite { site: 0 }),
            "an infinite coupling has no Boltzmann weight and must be named, not sampled"
        );
        // And every variant prints something a reader can act on.
        for e in [
            Invalid::Empty,
            Invalid::Beta { beta: 0.0 },
            Invalid::Gamma { gamma: -1.0 },
            Invalid::NonFinite { site: 2 },
            Invalid::TooFewSweeps { measure: 4, min: MIN_MEASURE },
        ] {
            assert!(e.to_string().len() > 30, "{e:?} explains nothing");
        }
    }

    /// The ledger charges one sample per slot visited, which is what the diagonal pass costs.
    #[test]
    fn the_ledger_charges_every_slot_of_every_sweep() {
        let g = ising::ring(4, 1.0, 0.0);
        let p = Params { beta: 1.0, gamma: 1.0, equilibrate: 50, measure: 200, cutoff: 16 };
        let mut led = Ledger::default();
        let out = run_metered(&g, &p, 9, Some(&mut led)).unwrap();
        assert_eq!(led.reads, (p.measure * g.n) as u64, "one read per spin per measured sweep");
        let sweeps = (p.equilibrate + p.measure) as u64;
        assert!(led.samples >= sweeps * 16, "a sweep visits at least the initial cutoff of slots");
        assert!(
            led.samples <= sweeps * out.cutoff as u64,
            "and never more than the final string length"
        );
    }

    /// A MODEL WITH NO TERMS IS `H = 0`, AND THE SPINS MUST STILL MOVE.
    ///
    /// The degenerate case where the candidate table is empty: no couplings, no fields, no
    /// transverse field. The energy is zero exactly, the string stays empty, and the spins are
    /// independent fair coins — so `<|m|>` over four of them is `(6*0 + 8*(1/2) + 2*1)/16 = 0.375`,
    /// a closed form and not a tolerance. A sampler that returned the right energy by freezing
    /// would fail on that last line.
    #[test]
    fn a_model_with_no_terms_has_zero_energy_and_still_shuffles_the_spins() {
        let g = GraphBuilder::new(4).build();
        let p = Params { beta: 1.0, gamma: 0.0, equilibrate: 10, measure: 20_000, cutoff: 16 };
        let out = run(&g, &p, 4).unwrap();
        assert_eq!(out.energy.value, 0.0, "H = 0 has E = 0 exactly, at every temperature");
        assert_eq!(out.order.value, 0.0, "and an empty operator string");
        assert_eq!(out.energy_floor, 0.0, "with nothing for the spectrum to reach below");
        assert!(
            (out.mz_abs.value - 0.375).abs() < 0.012,
            "four fair coins average |m| = 0.375, and this chain gave {:.4}",
            out.mz_abs.value
        );
    }

    /// The same seed is the same chain, and a different seed is a different one.
    #[test]
    fn a_run_is_reproducible_from_its_seed() {
        let g = ising::ring(5, 1.0, 0.2);
        let p = Params { beta: 1.4, gamma: 0.8, equilibrate: 200, measure: 500, cutoff: 16 };
        let a = run(&g, &p, 2024).unwrap();
        let b = run(&g, &p, 2024).unwrap();
        let c = run(&g, &p, 2025).unwrap();
        assert_eq!(a.energy.value, b.energy.value, "same seed, same chain");
        assert_eq!(a.state, b.state);
        assert!(a.energy.value != c.energy.value, "a different seed must be a different chain");
    }
}
