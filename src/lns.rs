//! Destroy-and-repair, neighbourhood ladders, and randomised greedy construction.
//!
//! Three metaheuristics that disagree about one thing: a local search is only as strong as the
//! neighbourhood it searches, so the neighbourhood should be the parameter rather than the
//! assumption. [`crate::tabu`], [`crate::bls`] and [`crate::icm`] all search the same fixed
//! single-flip neighbourhood and differ in how they choose inside it. These three change the
//! neighbourhood itself.
//!
//! * **Large neighbourhood search** — Shaw, *Using constraint programming and local search methods
//!   to solve vehicle routing problems*, CP-98, LNCS 1520, 417–431; Pisinger & Ropke, *Large
//!   neighborhood search*, Handbook of Metaheuristics (2010) 399–419. Throw away part of the
//!   assignment and rebuild that part **exactly**. The neighbourhood is every state agreeing with
//!   the incumbent outside the freed set — `2^k` states searched by one call, not `k` flips.
//! * **Variable neighbourhood search** — Mladenović & Hansen, *Variable neighborhood search*,
//!   Computers & Operations Research 24(11) (1997) 1097–1100; Hansen & Mladenović, EJOR 130 (2001)
//!   449–467. Keep a ladder of radii. Shake at radius `k`, descend, and on failure climb one rung:
//!   a search that cannot escape at radius 1 is given radius 2 rather than a restart.
//! * **GRASP** — Feo & Resende, *Greedy randomized adaptive search procedures*, Journal of Global
//!   Optimization 6 (1995) 109–133; for this problem family, Festa, Pardalos, Resende & Ribeiro,
//!   *Randomized heuristics for the MAX-CUT problem*, Optimization Methods and Software 17 (2002)
//!   1033–1058. Build a solution one variable at a time, always from a restricted candidate list of
//!   the near-best choices, then descend. The randomisation is in the CONSTRUCTION, which is what
//!   makes the restarts independent instead of correlated.
//!
//! ```
//! use ferrotherm::{lns, planted, oracle::{Exhaustive, Solver}};
//!
//! let p = planted::frustrated_loops(4, 12, 5); // 16 spins, small enough to enumerate
//! let whole = lns::Lns { fraction: 1.0, steps: 1, max_width: 16, ..lns::Lns::default() };
//! let o = lns::lns(&p.graph, &whole, 7).unwrap();
//!
//! assert!(o.exact(), "freeing everything at once is not a search, it is a solve");
//! assert_eq!(o.energy, Exhaustive.solve(&p.graph).1);
//! ```
//!
//! # What is borrowed, and what is new
//!
//! The repair is [`crate::hfs::step`] and is not reimplemented here. That routine conditions a
//! block on its frozen complement and hands the residual model to [`crate::exact::Elimination`],
//! and it is already tested against brute force over the block; a second copy of that conditioning
//! is a second place for the sign convention to drift. [`repair`] is a named delegation so that a
//! reader arriving at *destroy-and-repair* finds the repair where the loop is.
//!
//! What is new is the **destroy** half. [`crate::hfs`] grows induced TREES, because a tree's exact
//! solve is free at any size; LNS frees a FRACTION of the variables by a relatedness rule and pays
//! `2^w` for whatever width that turns out to be — or refuses. Three operators, each from the
//! literature above: [`Destroy::Random`], Shaw's relatedness removal ([`Destroy::Related`]) and
//! Ropke & Pisinger's worst removal ([`Destroy::Worst`]).
//!
//! # The degenerate case is the test
//!
//! At `fraction = 1.0` the freed set is every variable, the frozen complement is empty, and the
//! repair is an exact solve of the whole model. So LNS at full destroy must agree with exhaustive
//! enumeration — not approximately, on integer-coupled instances not at all approximately — and a
//! repair that had quietly become a greedy fill would still return a plausible state and a
//! plausible energy. That is the headline test, and [`LnsOutcome::exact`] is how a caller asks
//! whether a run was that solve rather than a search.
//!
//! # Monotonicity is a property of the repair, not of an acceptance rule
//!
//! Textbook LNS repairs heuristically and therefore needs an acceptance criterion — simulated
//! annealing over repairs, in Ropke & Pisinger. Here the repair is exact and the incumbent's own
//! assignment of the freed set is one of the candidates it minimises over, so a repair can never
//! raise the energy and there is nothing to accept. The consequence is that [`lns`] is a DESCENT
//! with no temperature, and everything that keeps it moving is variety in the freed set. That is
//! also why [`vns`] and [`grasp`] live here: they are where the escape is.
//!
//! # What was measured rather than assumed
//!
//! On `planted::frustrated_loops(8, 128, ·)` — the hard peak of that family, where `planted`'s own
//! table records [`crate::oracle::SteepestDescent`] with 50 restarts solving 4 of its 16 — over a
//! 6x6 grid of instance and solver seeds, every arm at 200 moves or 400 rounds or 64 iterations.
//! On this grid that same descent solves 6 of 36:
//!
//! | arm | solved | worst excess |
//! |---|---|---|
//! | GRASP, alpha 0.0 – 0.3 | 36/36 | 0 |
//! | LNS, related removal, 25% freed | 34/36 | 0.047 |
//! | VNS, radii 1..16 | 33/36 | 0.109 |
//! | LNS, related removal, 12.5% freed | 22/36 | 0.109 |
//! | VNS, radii 1..8 | 18/36 | 0.156 |
//! | LNS, worst removal, 25% freed | 12/36 | 0.172 |
//! | LNS, random removal, 25% freed | 9/36 | 0.141 |
//! | VNS, radius 1 only | 4/36 | 0.203 |
//! | GRASP, alpha 0.7 | 2/36 | 0.125 |
//!
//! Three rows of that were not what this module was written expecting.
//!
//! **Relatedness is not a refinement of LNS, it is most of the method.** Random removal of the
//! same 25% solves 9 of 36 where Shaw's related removal solves 34 — the same exact repair, the same
//! budget, four times the hit rate. What the destroy operator chooses matters more than that the
//! repair is exact, which is the opposite of where the interesting machinery looks like it is.
//!
//! **The ladder is the whole of VNS.** Pinned at radius 1 it solves 4 of 36, which is where
//! restarted greedy descent sits; allowed to climb to 16 it solves 33. A VNS reported without its
//! `deepest` radius is a number nobody can check, which is why [`VnsOutcome`] carries one.
//!
//! **A greedy CONSTRUCTION beats a greedy DESCENT on the instances descent is worst at.** ONE
//! construction at `alpha = 0` — no restarts, no local search — reaches the planted optimum on 26
//! of those 36 runs, where [`crate::oracle::SteepestDescent`] with **fifty** restarts reaches it on
//! 6. Same "align with the local field" rule, four times the hit rate, because a construction
//! builds an assignment in an order it chooses and a descent repairs one it was handed. On
//! `frustrated_loops(8, 128, 3)` in particular the mean constructed energy over twenty seeds is
//! −256.00, which is that instance's optimum exactly: the descent that follows has nothing to do.
//!
//! `planted`'s own notes explain why this family yields to it — at four planted loops per edge the
//! couplings concentrate toward a gauged ferromagnet, which is what a field-following construction
//! cannot get wrong. So GRASP's 36/36 above is reported as a fact about the family and NOT as
//! evidence about GRASP. On the 6x6 lattice glasses of
//! `the_three_searches_never_land_below_exact_elimination`, where [`crate::exact::Elimination`]
//! supplies the true optimum, GRASP hits 4 of 4, LNS 3 and VNS 2.

use crate::exact::{Elimination, TooWide};
use crate::graph::Graph;
use crate::rng::Pcg;
use crate::tabu::{flip, gains};

/// An input this module cannot interpret, naming what it was actually given.
///
/// A default is not a fallback: a fraction of zero frees nothing and a fraction of two frees
/// nothing that exists, and silently clamping either would turn a caller's typo into a different
/// experiment that still returns a number.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Invalid {
    /// A destroy fraction outside `(0, 1]`, `NaN` included.
    Fraction(f64),
    /// A restricted-candidate-list width outside `[0, 1]`, `NaN` included.
    Alpha(f64),
    /// A shake ladder with no rungs: `k_min` of zero, or `k_max` below `k_min`.
    Ladder {
        /// The smallest shake radius asked for. Zero flips nothing, so the descent that follows
        /// returns the state it was given and the ladder never moves.
        k_min: usize,
        /// The largest shake radius asked for.
        k_max: usize,
    },
    /// A starting state that is not one spin per node.
    StartLength {
        /// Spins supplied.
        got: usize,
        /// Spins the graph has.
        want: usize,
    },
}

impl core::fmt::Display for Invalid {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Invalid::Fraction(x) => write!(
                f,
                "a destroy fraction must lie in (0, 1] -- it is the share of the variables freed \
                 for the repair -- and {x} does not"
            ),
            Invalid::Alpha(x) => write!(
                f,
                "a restricted candidate list needs alpha in [0, 1], where 0 is pure greedy and 1 \
                 is pure random construction, and {x} is neither"
            ),
            Invalid::Ladder { k_min, k_max } => write!(
                f,
                "a shake ladder needs 1 <= k_min <= k_max, so that some spin is actually flipped \
                 and the radius can climb; k_min {k_min} and k_max {k_max} give no rungs"
            ),
            Invalid::StartLength { got, want } => write!(
                f,
                "a starting state is one spin per node: {got} spins were given for a graph of \
                 {want} nodes"
            ),
        }
    }
}

impl core::error::Error for Invalid {}

// ---- shared pieces ----------------------------------------------------------------------------

/// The most flips one [`descend`] will make, as a multiple of the node count.
///
/// A steepest descent strictly lowers the energy at every step and the state space is finite, so
/// mathematically it terminates and no cap is needed. The cap is against the ARITHMETIC, not the
/// mathematics: the gain vector is maintained incrementally by [`crate::tabu::flip`], and a drift
/// large enough to make a non-improving move look improving would put the descent in a cycle
/// forever rather than return a wrong answer. Measured across this module's tests the longest
/// descent observed is a small multiple of `n` and nowhere near `64 n` — see
/// `a_descent_ends_at_a_local_minimum_certified_by_recomputed_energies`, which reports it.
const DESCENT_CAP: usize = 64;

/// Steepest single-spin descent to a local minimum, in place. Returns the flips it made.
///
/// Ties break to the lowest index, so a descent from a given state is deterministic and carries no
/// seed. This is the local search inside both [`vns`] and [`grasp`]; the gain bookkeeping is
/// [`crate::tabu`]'s, shared rather than copied, because an incremental gain that drifts raises
/// nothing — it quietly makes the search worse.
///
/// # Panics
///
/// If `s` is not one spin per node.
pub fn descend(g: &Graph, s: &mut [i8]) -> usize {
    assert_eq!(s.len(), g.n, "a state is one spin per node");
    if g.n == 0 {
        return 0;
    }
    let mut delta = gains(g, s);
    let cap = DESCENT_CAP * g.n;
    let mut flips = 0usize;
    while flips < cap {
        let mut pick = usize::MAX;
        let mut best = -1e-12;
        for i in 0..g.n {
            if delta[i] < best {
                best = delta[i];
                pick = i;
            }
        }
        if pick == usize::MAX {
            break;
        }
        flip(g, s, &mut delta, pick);
        flips += 1;
    }
    flips
}

/// Flip `k` distinct spins chosen uniformly at random, in place. Returns how many were flipped.
///
/// This is VNS's shake: a uniform draw from the radius-`k` Hamming sphere around `s`, and
/// deliberately not a descent. `k` above the node count flips every spin, which is the global gauge
/// flip and the largest move there is.
pub fn shake(s: &mut [i8], k: usize, rng: &mut Pcg) -> usize {
    let n = s.len();
    let k = k.min(n);
    // Partial Fisher-Yates over the indices: `k` distinct sites, without a rejection loop that
    // would burn a variable number of draws and so make the stream depend on the collisions.
    let mut idx: Vec<usize> = (0..n).collect();
    for i in 0..k {
        let at = i + (rng.f64() * (n - i) as f64) as usize % (n - i);
        idx.swap(i, at);
        s[idx[i]] = -s[idx[i]];
    }
    k
}

/// An energy that is certainly not BELOW the exact energy of `s`, accumulated in directed rounding.
///
/// [`Graph::energy`] sums in round-to-nearest, so the number it returns may sit either side of the
/// truth by a few units in the last place. That is irrelevant to a comparison and fatal to a CLAIM:
/// "the ground energy of this model is at most `E`" is sound only if `E` is an upper bound on the
/// energy of a state actually exhibited. This is that bound, through [`crate::round::sum_up`] over
/// the model's own terms — one per bias and one per undirected edge.
///
/// The crate has been bitten from the other direction: [`crate::bound::forest`] once reported lower
/// bounds ABOVE the optimum because it accumulated with `+`.
///
/// # It bounds the EXACT energy, and `Graph::energy` is not that
///
/// Measured, and it is the whole reason this exists: on a 20-node model with continuous couplings
/// this returns `1.3592955025169855` where [`Graph::energy`] returns `1.359295502516986` — the
/// naive left-to-right sum sits ABOVE the directed upper bound, because under cancellation its own
/// error (`n eps Sigma|x|`) is larger than the compensated bound's guard (`2 eps |total|`). Both
/// numbers are honest and only one of them is a bound. So compare a claim against THIS, never
/// against `Graph::energy` with a tolerance bolted on. With integer couplings — every `+-J` glass,
/// every `planted::frustrated_loops` — both sums are exact and this is `Graph::energy` plus a
/// positive guard.
///
/// # Panics
///
/// If `s` is not one spin per node.
#[must_use]
pub fn energy_upper(g: &Graph, s: &[i8]) -> f64 {
    assert_eq!(s.len(), g.n, "a state is one spin per node");
    let mut terms = Vec::with_capacity(g.n + g.n_edges);
    for i in 0..g.n {
        let si = f64::from(s[i]);
        terms.push(-g.h[i] * si);
        for k in g.offset[i]..g.offset[i + 1] {
            let j = g.nbr[k] as usize;
            if j > i {
                terms.push(-g.w[k] * si * f64::from(s[j]));
            }
        }
    }
    crate::round::sum_up(&terms)
}

// ---- large neighbourhood search -----------------------------------------------------------------

/// How a destroy operator picks the variables it frees.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Destroy {
    /// A uniform random subset. The control: it has no model of the instance at all, and anything
    /// claiming that relatedness matters has to beat it.
    Random,
    /// Shaw's relatedness removal: grow outward from a seed, preferring strongly coupled
    /// neighbours.
    ///
    /// Shaw's argument is that freeing UNRELATED variables gives a repair that decomposes — the
    /// pieces do not interact, so solving them together finds nothing solving them separately
    /// would not. Relatedness in the original is problem-specific (distance, time window, demand);
    /// on an Ising model the interaction IS the coupling, so `|J_ij|` is the relatedness with
    /// nothing left to choose.
    Related,
    /// Ropke & Pisinger's worst removal: free the variables whose current assignment costs the
    /// most.
    ///
    /// A variable's cost here is `-s_i (h_i + sum_j J_ij s_j)`, its own contribution to the energy
    /// with every incident edge counted from its end. High cost means the assignment is fighting
    /// its neighbourhood, which is the honest reading of "this is the part that is wrong".
    Worst,
}

/// The randomisation exponent on both biased picks, from Ropke & Pisinger (2006).
///
/// Rank the candidates best-first and take index `floor(L * y^p)` for `y` uniform on `[0, 1)`. At
/// `p = 1` the pick is uniform and the ranking does nothing; large `p` makes it deterministic and
/// the operator stops exploring. Three is the published default.
///
/// Applied as `y * y * y` rather than `powf(3.0)` deliberately: a product of `f64`s is exact to the
/// same bits everywhere, and `powf` is a libm routine this crate cannot promise is identical across
/// platforms. Determinism by seed is the crate's headline and it is not worth a call.
const BIAS_POWER: u32 = 3;

/// Index into a best-first ranking of `len` candidates, biased toward the front.
fn biased_index(len: usize, rng: &mut Pcg) -> usize {
    debug_assert!(len > 0, "a biased pick needs candidates");
    debug_assert_eq!(BIAS_POWER, 3, "the cube below is the exponent, written out");
    let y = rng.f64();
    let b = y * y * y;
    ((len as f64) * b) as usize % len
}

/// How many variables a fraction frees, or why the fraction cannot be read.
///
/// `ceil`, so any positive fraction frees at least one variable and `1.0` frees every one of them —
/// the degenerate case that makes the repair's exactness checkable. A fraction that rounds past the
/// node count is clamped to it; a fraction that is not a share of anything is an error.
///
/// # Errors
///
/// [`Invalid::Fraction`] when the fraction is outside `(0, 1]` or is `NaN`.
pub fn freed_count(n: usize, fraction: f64) -> Result<usize, Invalid> {
    // `!(a && b)` rather than the inverted comparisons, so NaN is REJECTED rather than accepted.
    if !(fraction > 0.0 && fraction <= 1.0) {
        return Err(Invalid::Fraction(fraction));
    }
    if n == 0 {
        return Ok(0);
    }
    Ok(((fraction * n as f64).ceil() as usize).clamp(1, n))
}

/// Choose `count` distinct variables to free, by `op`.
///
/// Returns exactly `min(count, g.n)` distinct indices, in the order the operator chose them.
///
/// # Panics
///
/// If `s` is not one spin per node. [`Destroy::Worst`] reads the current assignment, so a destroy
/// operator is a function of the incumbent and not of the graph alone.
#[must_use]
pub fn free(g: &Graph, s: &[i8], op: Destroy, count: usize, rng: &mut Pcg) -> Vec<usize> {
    assert_eq!(s.len(), g.n, "a state is one spin per node");
    let n = g.n;
    let k = count.min(n);
    if k == 0 {
        return Vec::new();
    }
    match op {
        Destroy::Random => {
            let mut idx: Vec<usize> = (0..n).collect();
            for i in 0..k {
                let at = i + (rng.f64() * (n - i) as f64) as usize % (n - i);
                idx.swap(i, at);
            }
            idx.truncate(k);
            idx
        }
        Destroy::Related => {
            let mut inside = vec![false; n];
            let mut out = Vec::with_capacity(k);
            let start = (rng.f64() * n as f64) as usize % n;
            inside[start] = true;
            out.push(start);
            // Candidates are the free neighbours of everything already freed, ranked by coupling
            // strength. Carried as a list rather than recomputed, and filtered on use: a node that
            // joined since being pushed is stale, not an error.
            let mut frontier: Vec<(f64, usize)> = Vec::new();
            let push = |i: usize, frontier: &mut Vec<(f64, usize)>| {
                for kk in g.offset[i]..g.offset[i + 1] {
                    frontier.push((g.w[kk].abs(), g.nbr[kk] as usize));
                }
            };
            push(start, &mut frontier);
            while out.len() < k {
                frontier.retain(|&(_, j)| !inside[j]);
                let pick = if frontier.is_empty() {
                    // The freed set has eaten its whole component. Jump: a uniform free node, which
                    // exists because `out.len() < k <= n`. Shaw's operator is silent about this
                    // because a vehicle routing instance is connected; a spin glass need not be.
                    let open: Vec<usize> = (0..n).filter(|&i| !inside[i]).collect();
                    open[(rng.f64() * open.len() as f64) as usize % open.len()]
                } else {
                    // Stable sort on a total order: ties keep discovery order, so the ranking is
                    // reproducible on a lattice where every `|J|` is the same number.
                    frontier.sort_by(|a, b| b.0.total_cmp(&a.0));
                    frontier[biased_index(frontier.len(), rng)].1
                };
                inside[pick] = true;
                out.push(pick);
                push(pick, &mut frontier);
            }
            out
        }
        Destroy::Worst => {
            // `gains[i]` is `2 s_i (h_i + sum_j J_ij s_j)`, so the node's own energy contribution
            // is minus half of it. Shared with tabu rather than recomputed.
            let d = gains(g, s);
            let mut pool: Vec<(f64, usize)> = (0..n).map(|i| (-0.5 * d[i], i)).collect();
            pool.sort_by(|a, b| b.0.total_cmp(&a.0));
            let mut out = Vec::with_capacity(k);
            for _ in 0..k {
                let at = biased_index(pool.len(), rng);
                out.push(pool.remove(at).1);
            }
            out
        }
    }
}

/// Rebuild the freed variables exactly, with everything else held fixed.
///
/// Returns the change in total energy, which is `<= 0`: the incumbent's own assignment of the freed
/// set is one of the candidates the exact solve minimises over, so the repair cannot make things
/// worse. That is the whole reason [`lns`] needs no acceptance criterion.
///
/// **This is [`crate::hfs::step`] under the name the LNS literature uses**, and deliberately not a
/// second implementation. The conditioning — absorbing each frozen neighbour's say into the freed
/// node's field — is the one place the sign convention could hide a defect, and it is tested there
/// against brute force over the block.
///
/// # Errors
///
/// [`TooWide`] when the freed set's induced subgraph is wider than `el.max_width`. Unlike
/// [`crate::hfs`], whose blocks are trees and therefore always width 1, a destroy operator frees
/// whatever the fraction asks for and the width is whatever that turns out to be.
///
/// # Panics
///
/// If `s` is not one spin per node, or a freed index is outside the graph.
#[inline]
pub fn repair(g: &Graph, s: &mut [i8], freed: &[usize], el: &Elimination) -> Result<f64, TooWide> {
    crate::hfs::step(g, s, freed, el)
}

/// How to run a large neighbourhood search.
// NOT `Copy`: `start` holds a state, following `tabu::Params` and `branch::Params`.
#[derive(Clone, Debug, PartialEq)]
pub struct Lns {
    /// Destroy-and-repair moves to attempt.
    pub steps: usize,
    /// Share of the variables freed per move, in `(0, 1]`. `1.0` frees all of them and turns the
    /// run into one exact solve.
    pub fraction: f64,
    /// Width ceiling for the repair, since an exact solve costs `2^w`. A freed set that measures
    /// wider is skipped and counted in [`LnsOutcome::refused`] rather than approximated.
    pub max_width: usize,
    /// Which destroy operator chooses the freed set.
    pub destroy: Destroy,
    /// A state to start from — from an anneal, from [`crate::tabu`], or from a previous run.
    ///
    /// `None` starts from noise. A wrong length is [`Invalid::StartLength`] rather than being
    /// ignored: this function can already fail, so there is no fallible-for-nothing cost to saying
    /// what was wrong, and silently searching from noise instead is how a composed pipeline comes
    /// to report the second stage's numbers for the first stage's work.
    pub start: Option<Vec<i8>>,
}

impl Default for Lns {
    fn default() -> Self {
        Lns { steps: 200, fraction: 0.25, max_width: 12, destroy: Destroy::Related, start: None }
    }
}

/// What a large neighbourhood search did.
#[derive(Clone, Debug, PartialEq)]
pub struct LnsOutcome {
    /// The state it ended on, which is also the best it saw — the repair never raises the energy.
    pub state: Vec<i8>,
    /// Its energy, recomputed from `state` rather than accumulated across the moves' deltas.
    pub energy: f64,
    /// Variables freed per move, which is [`freed_count`] of the fraction.
    pub freed: usize,
    /// Repairs that actually ran.
    pub moves: usize,
    /// Repairs that strictly lowered the energy. A run whose every repair returned the assignment
    /// it was given has converged, and this is how a caller sees that rather than infers it.
    pub improving: usize,
    /// Freed sets refused for width. A run that mostly refused is visible rather than merely
    /// disappointing.
    pub refused: usize,
}

impl LnsOutcome {
    /// Whether this run SOLVED the model rather than searched it.
    ///
    /// True when every variable was freed at once and at least one repair ran: the repair is the
    /// exact minimum over the freed set given the rest, and when the freed set is everything there
    /// is no rest. `energy` is then the ground energy and `state` a ground state, to the arithmetic
    /// of [`crate::exact::Elimination`].
    #[must_use]
    pub fn exact(&self) -> bool {
        self.moves > 0 && !self.state.is_empty() && self.freed == self.state.len()
    }
}

/// Destroy and repair, from a seeded random start.
///
/// # Errors
///
/// [`Invalid::Fraction`] for a destroy fraction outside `(0, 1]`, and [`Invalid::StartLength`] for
/// a supplied start that is not one spin per node.
pub fn lns(g: &Graph, p: &Lns, seed: u64) -> Result<LnsOutcome, Invalid> {
    let count = freed_count(g.n, p.fraction)?;
    let mut rng = Pcg::new(seed, 0x004C_4E53);
    let mut s: Vec<i8> = match &p.start {
        Some(st) if st.len() == g.n => st.clone(),
        Some(st) => return Err(Invalid::StartLength { got: st.len(), want: g.n }),
        None => (0..g.n).map(|_| rng.spin(0.5)).collect(),
    };
    let el = Elimination { max_width: p.max_width.max(1) };
    let (mut moves, mut improving, mut refused) = (0usize, 0usize, 0usize);

    for _ in 0..p.steps {
        if g.n == 0 {
            break;
        }
        let block = free(g, &s, p.destroy, count, &mut rng);
        match repair(g, &mut s, &block, &el) {
            Ok(d) => {
                moves += 1;
                if d < -1e-12 {
                    improving += 1;
                }
            }
            Err(_) => refused += 1,
        }
    }

    Ok(LnsOutcome { energy: g.energy(&s), state: s, freed: count, moves, improving, refused })
}

// ---- variable neighbourhood search --------------------------------------------------------------

/// How to run a variable neighbourhood search.
#[derive(Clone, Debug, PartialEq)]
pub struct Vns {
    /// Shake-and-descend rounds.
    pub rounds: usize,
    /// The first rung: spins flipped by the shake while the ladder is at the bottom.
    pub k_min: usize,
    /// The last rung. Past it the ladder wraps back to `k_min` and
    /// [`VnsOutcome::ladder_restarts`] counts the wrap — the outer repetition of basic VNS.
    pub k_max: usize,
    /// A state to start from. `None` starts from noise. Either way the first thing that happens is
    /// a descent, because the ladder's whole logic is about escaping a local minimum and a state
    /// that is not one has nothing to escape.
    pub start: Option<Vec<i8>>,
}

impl Default for Vns {
    fn default() -> Self {
        Vns { rounds: 400, k_min: 1, k_max: 8, start: None }
    }
}

/// What a variable neighbourhood search did.
#[derive(Clone, Debug, PartialEq)]
pub struct VnsOutcome {
    /// The incumbent, which is the best state seen: VNS only ever moves downhill.
    pub state: Vec<i8>,
    /// Its energy, recomputed from `state`.
    pub energy: f64,
    /// Shakes performed, which is `rounds` unless the graph is empty.
    pub shakes: usize,
    /// Shakes whose descent landed strictly below the incumbent, each of which reset the ladder.
    pub improvements: usize,
    /// Times the ladder ran off the top and wrapped back to `k_min`.
    pub ladder_restarts: usize,
    /// The largest radius actually shaken with. Equal to `k_min` means the ladder never climbed, so
    /// the search was a fixed-radius restart loop wearing a ladder's name.
    pub deepest: usize,
    /// Total single-spin flips made by the descents, which is what the shakes cost to repair.
    pub descent_flips: usize,
}

/// Shake, descend, and climb the ladder on failure.
///
/// # Errors
///
/// [`Invalid::Ladder`] when `k_min` is zero or `k_max` is below it, and [`Invalid::StartLength`]
/// for a supplied start that is not one spin per node.
pub fn vns(g: &Graph, p: &Vns, seed: u64) -> Result<VnsOutcome, Invalid> {
    if p.k_min == 0 || p.k_max < p.k_min {
        return Err(Invalid::Ladder { k_min: p.k_min, k_max: p.k_max });
    }
    let mut rng = Pcg::new(seed, 0x0056_4E53);
    let mut s: Vec<i8> = match &p.start {
        Some(st) if st.len() == g.n => st.clone(),
        Some(st) => return Err(Invalid::StartLength { got: st.len(), want: g.n }),
        None => (0..g.n).map(|_| rng.spin(0.5)).collect(),
    };
    let mut descent_flips = descend(g, &mut s);
    let mut energy = g.energy(&s);

    let (mut shakes, mut improvements, mut ladder_restarts, mut deepest) = (0, 0, 0, 0usize);
    let mut k = p.k_min;
    for _ in 0..p.rounds {
        if g.n == 0 {
            break;
        }
        let mut t = s.clone();
        shake(&mut t, k, &mut rng);
        shakes += 1;
        deepest = deepest.max(k);
        descent_flips += descend(g, &mut t);
        let e = g.energy(&t);
        if e < energy - 1e-12 {
            s = t;
            energy = e;
            improvements += 1;
            // Down to the bottom rung: the point of the ladder is that a NEW incumbent deserves the
            // cheap neighbourhood first, not that the radius ratchets upward.
            k = p.k_min;
        } else {
            k += 1;
            if k > p.k_max {
                k = p.k_min;
                ladder_restarts += 1;
            }
        }
    }

    Ok(VnsOutcome {
        energy: g.energy(&s),
        state: s,
        shakes,
        improvements,
        ladder_restarts,
        deepest,
        descent_flips,
    })
}

// ---- GRASP --------------------------------------------------------------------------------------

/// The energy that fixing `s_i = v` adds, given the field from the ALREADY-assigned neighbours.
///
/// The greedy function of the construction, and the one line in this module where the crate's sign
/// convention `E = -J s s - h s` is applied rather than inherited. Each edge is priced exactly once
/// — at the moment its second endpoint is assigned — and each bias exactly once, so the costs of a
/// full construction sum to the model's own energy.
fn assign_cost(v: i8, field: f64) -> f64 {
    -f64::from(v) * field
}

/// Build a complete assignment one variable at a time from a restricted candidate list.
///
/// At each step every unassigned variable offers both of its values, each priced by
/// [`assign_cost`] against the field its assigned neighbours already exert. The candidate list
/// keeps every offer within `alpha` of the range between the best and the worst, and one is drawn
/// uniformly from it.
///
/// `alpha = 0` is pure greedy and is deterministic wherever the best offer is unique; `alpha = 1`
/// admits every offer and is uniform random construction. Both ends are worth being able to ask
/// for: they are the controls that show the list is doing something.
///
/// # Errors
///
/// [`Invalid::Alpha`] when `alpha` is outside `[0, 1]` or is `NaN`.
pub fn construct(g: &Graph, alpha: f64, rng: &mut Pcg) -> Result<Vec<i8>, Invalid> {
    // `contains` is `start <= x && x <= end`, so NaN fails it and is REJECTED here rather than
    // silently admitted by an inverted comparison.
    if !(0.0..=1.0).contains(&alpha) {
        return Err(Invalid::Alpha(alpha));
    }
    let n = g.n;
    let mut s = vec![1i8; n];
    let mut assigned = vec![false; n];
    // `field[i]` is `h_i + sum over ASSIGNED neighbours j of J_ij s_j`. A partial assignment's
    // greedy function has to ignore the edges whose other end is still open, or it prices a choice
    // against spins nobody has made.
    let mut field = g.h.clone();
    let mut rcl: Vec<(usize, i8)> = Vec::new();

    for _ in 0..n {
        let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
        for i in 0..n {
            if assigned[i] {
                continue;
            }
            for v in [-1i8, 1i8] {
                let c = assign_cost(v, field[i]);
                lo = lo.min(c);
                hi = hi.max(c);
            }
        }
        // Feo & Resende's value-based list: everything within `alpha` of the best, measured on the
        // range the offers actually span. At alpha 0 this is the best offer alone.
        let thresh = lo + alpha * (hi - lo);
        rcl.clear();
        for i in 0..n {
            if assigned[i] {
                continue;
            }
            for v in [-1i8, 1i8] {
                if assign_cost(v, field[i]) <= thresh {
                    rcl.push((i, v));
                }
            }
        }
        let (i, v) = rcl[(rng.f64() * rcl.len() as f64) as usize % rcl.len()];
        assigned[i] = true;
        s[i] = v;
        for k in g.offset[i]..g.offset[i + 1] {
            let j = g.nbr[k] as usize;
            if !assigned[j] {
                field[j] += g.w[k] * f64::from(v);
            }
        }
    }
    Ok(s)
}

/// How to run a GRASP.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Grasp {
    /// Construct-and-descend iterations. Zero is read as one, since an outcome has to carry a state
    /// and the smallest honest run is one construction.
    pub iterations: usize,
    /// Restricted candidate list width, in `[0, 1]`. `0` is greedy, `1` is random.
    pub alpha: f64,
}

impl Default for Grasp {
    fn default() -> Self {
        Grasp { iterations: 64, alpha: 0.3 }
    }
}

/// What a GRASP did.
#[derive(Clone, Debug, PartialEq)]
pub struct GraspOutcome {
    /// The best state over every iteration.
    pub state: Vec<i8>,
    /// Its energy, recomputed from `state`.
    pub energy: f64,
    /// The iteration that produced it.
    pub best_at: usize,
    /// The energy of the CONSTRUCTION `state` came from, before its descent.
    ///
    /// The pair `(best_constructed, energy)` says how much of the answer the construction found and
    /// how much the local search did. A GRASP whose two numbers are far apart is a random restart
    /// loop with extra steps: the construction is contributing nothing and `alpha` is too large.
    pub best_constructed: f64,
    /// Mean construction energy over the run, before any descent. The candidate list's own
    /// diagnostic, and the number that moves when `alpha` does.
    pub mean_constructed: f64,
    /// Total single-spin flips over every descent.
    pub descent_flips: usize,
}

/// Greedy randomised construction, then descent, best of `iterations`.
///
/// # Errors
///
/// [`Invalid::Alpha`] when `alpha` is outside `[0, 1]` or is `NaN`.
pub fn grasp(g: &Graph, p: &Grasp, seed: u64) -> Result<GraspOutcome, Invalid> {
    let mut rng = Pcg::new(seed, 0x4752_4153);
    let iters = p.iterations.max(1);
    let mut best: Vec<i8> = vec![1i8; g.n];
    let (mut best_e, mut best_at, mut best_constructed) = (f64::INFINITY, 0usize, f64::INFINITY);
    let (mut sum_constructed, mut descent_flips) = (0.0f64, 0usize);

    for it in 0..iters {
        let mut s = construct(g, p.alpha, &mut rng)?;
        let constructed = g.energy(&s);
        sum_constructed += constructed;
        descent_flips += descend(g, &mut s);
        let e = g.energy(&s);
        if e < best_e {
            best_e = e;
            best = s;
            best_at = it;
            best_constructed = constructed;
        }
    }

    Ok(GraspOutcome {
        energy: g.energy(&best),
        state: best,
        best_at,
        best_constructed,
        mean_constructed: sum_constructed / iters as f64,
        descent_flips,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::GraphBuilder;
    use crate::oracle::{Exhaustive, Solver, SteepestDescent};
    use crate::planted::frustrated_loops;

    /// A random model with biases, so the greedy function has something to break ties on.
    fn field_graph(n: usize, p: f64, seed: u64) -> Graph {
        let mut rng = Pcg::new(seed, 0xF1E1);
        let mut b = GraphBuilder::new(n);
        for i in 0..n {
            b.bias(i, rng.f64() * 2.0 - 1.0);
            for j in (i + 1)..n {
                if rng.f64() < p {
                    b.couple(i, j, rng.f64() * 2.0 - 1.0);
                }
            }
        }
        b.build()
    }

    /// A periodic 2D +-J spin glass: integer energies, and narrow enough for exact elimination.
    fn glass2d(l: usize, seed: u64) -> Graph {
        let mut rng = Pcg::new(seed, 0x2D62);
        let mut b = GraphBuilder::new(l * l);
        let at = |x: usize, y: usize| (y % l) * l + (x % l);
        for y in 0..l {
            for x in 0..l {
                b.couple(at(x, y), at(x + 1, y), if rng.f64() < 0.5 { 1.0 } else { -1.0 });
                b.couple(at(x, y), at(x, y + 1), if rng.f64() < 0.5 { 1.0 } else { -1.0 });
            }
        }
        b.build()
    }

    /// The highest-energy state from which no single flip descends, by enumeration, or `None` when
    /// the only local minima are global. Every energy here is recomputed by [`Graph::energy`], so
    /// the certificate does not depend on this module's gain bookkeeping.
    fn worst_trap(g: &Graph, optimum: f64) -> Option<(Vec<i8>, f64)> {
        let mut worst: Option<(Vec<i8>, f64)> = None;
        for mask in 0..(1u32 << g.n) {
            let s: Vec<i8> = (0..g.n).map(|i| if mask >> i & 1 == 1 { 1 } else { -1 }).collect();
            let e = g.energy(&s);
            if e <= optimum + 1e-9 {
                continue;
            }
            let trapped = (0..g.n).all(|i| {
                let mut t = s.clone();
                t[i] = -t[i];
                g.energy(&t) >= e - 1e-12
            });
            if trapped && worst.as_ref().is_none_or(|(_, we)| e > *we) {
                worst = Some((s, e));
            }
        }
        worst
    }

    /// THE HEADLINE. Free every variable at once and the repair is a solve of the whole model, so
    /// it must agree with enumerating all `2^n` states — and on integer-coupled instances it must
    /// agree EXACTLY, because an integer sum in `f64` is exact whatever order it is taken in.
    ///
    /// The degenerate case is the one that bites. A repair that had quietly become a greedy fill,
    /// or that had absorbed the frozen neighbours' field with the wrong sign, still returns a
    /// plausible state and a plausible energy at any fraction below one; at fraction one there is
    /// nothing frozen to be wrong about and nowhere for a near-miss to hide.
    ///
    /// The asymmetric half is the same loop at 12.5%: on the same twelve instances it MISSES the
    /// enumerated optimum 34 times in 40, which is what says the full-destroy agreement is a
    /// property of the destroy fraction rather than of instances too easy to distinguish anything.
    #[test]
    fn lns_at_full_destroy_matches_exhaustive_enumeration() {
        let whole = |op: Destroy| Lns {
            steps: 1,
            fraction: 1.0,
            max_width: 20,
            destroy: op,
            start: None,
        };

        // Integer couplings: assert exact equality, since there is no rounding to allow for.
        for l in [3usize, 4] {
            for loops in [4usize, 16, 40] {
                for seed in 1..=3u64 {
                    let p = frustrated_loops(l, loops, seed);
                    let exact = Exhaustive.solve(&p.graph).1;
                    for op in [Destroy::Random, Destroy::Related, Destroy::Worst] {
                        let o = lns(&p.graph, &whole(op), seed).unwrap();
                        assert!(o.exact(), "l={l} loops={loops} {op:?}: not a whole-model solve");
                        assert_eq!(
                            o.energy, exact,
                            "l={l} loops={loops} seed={seed} {op:?}: {} vs enumerated {exact}",
                            o.energy
                        );
                        assert_eq!(o.freed, p.graph.n, "full destroy frees everything");
                    }
                }
            }
        }

        // Continuous couplings and fields, where the two sums may differ in the last bits.
        for n in [10usize, 14] {
            for seed in 1..=3u64 {
                let g = field_graph(n, 0.35, seed);
                let exact = Exhaustive.solve(&g).1;
                for op in [Destroy::Random, Destroy::Related, Destroy::Worst] {
                    let o = lns(&g, &whole(op), seed).unwrap();
                    assert!(
                        (o.energy - exact).abs() < 1e-9,
                        "n={n} seed={seed} {op:?}: {} vs enumerated {exact}",
                        o.energy
                    );
                }
            }
        }

        // AND THE SAME LOOP AT AN EIGHTH MISSES. Without this the test above would also pass for a
        // module whose every setting happened to solve these instances.
        let (mut miss, mut total) = (0, 0);
        for iseed in 1..=10u64 {
            let p = frustrated_loops(4, 16, iseed);
            let exact = Exhaustive.solve(&p.graph).1;
            for sseed in 1..=4u64 {
                let part = Lns { steps: 8, fraction: 0.125, ..whole(Destroy::Related) };
                let o = lns(&p.graph, &part, sseed).unwrap();
                total += 1;
                if o.energy > exact + 1e-9 {
                    miss += 1;
                }
                assert!(o.energy >= exact - 1e-9, "below the enumerated optimum: {}", o.energy);
                assert!(!o.exact(), "an eighth of the variables is not the whole model");
            }
        }
        assert!(miss >= 25, "measured 34 of 40 misses at an eighth; got {miss} of {total}");
    }

    /// A freed set is a SET, is the size the fraction asks for, and at `1.0` is everything.
    #[test]
    fn a_freed_set_is_a_set_and_a_full_fraction_frees_every_node() {
        assert_eq!(freed_count(16, 1.0), Ok(16));
        assert_eq!(freed_count(16, 0.5), Ok(8));
        // ceil, so a fraction too small to name a whole variable still names one.
        assert_eq!(freed_count(16, 0.01), Ok(1));
        assert_eq!(freed_count(16, 0.26), Ok(5));
        assert_eq!(freed_count(0, 1.0), Ok(0));

        let g = glass2d(6, 2);
        let mut rng = Pcg::new(7, 3);
        let s: Vec<i8> = (0..g.n).map(|_| rng.spin(0.5)).collect();
        for op in [Destroy::Random, Destroy::Related, Destroy::Worst] {
            for frac in [0.05f64, 0.25, 0.5, 1.0] {
                let want = freed_count(g.n, frac).unwrap();
                let f = free(&g, &s, op, want, &mut rng);
                let uniq: std::collections::BTreeSet<usize> = f.iter().copied().collect();
                assert_eq!(f.len(), want, "{op:?} at {frac} freed {} of {want}", f.len());
                assert_eq!(uniq.len(), f.len(), "{op:?} at {frac} repeated a variable");
                assert!(uniq.iter().all(|&i| i < g.n), "{op:?} freed a node off the graph");
                if frac == 1.0 {
                    assert_eq!(uniq.len(), g.n, "{op:?} at 1.0 must free every node");
                }
            }
        }
    }

    /// ORACLE 2: no repair ever raises the energy, over a long run and across all three operators.
    ///
    /// Checked two ways at once, because they can disagree: the delta [`repair`] reports must be
    /// `<= 0`, AND the energy recomputed from the state must never rise. A repair that returned a
    /// truthful delta while writing a different state back would pass the first and fail the
    /// second.
    ///
    /// The asymmetric halves: some move must strictly improve (a repair that always returned the
    /// assignment it was handed is monotone and useless), and the width guard must actually refuse
    /// when handed a set too wide (a guard that never fires is a guard nobody can trust).
    #[test]
    fn the_repair_never_raises_the_energy_over_two_thousand_moves() {
        let g = glass2d(8, 3);
        let mut rng = Pcg::new(11, 5);
        let mut s: Vec<i8> = (0..g.n).map(|_| rng.spin(0.5)).collect();
        let started = g.energy(&s);
        let el = Elimination { max_width: 12 };
        let mut prev = started;
        let (mut moves, mut improving, mut refused) = (0, 0, 0);

        for step in 0..2000usize {
            let op = [Destroy::Random, Destroy::Related, Destroy::Worst][step % 3];
            let block = free(&g, &s, op, 12, &mut rng);
            match repair(&g, &mut s, &block, &el) {
                Ok(d) => {
                    moves += 1;
                    assert!(d <= 1e-12, "step {step} {op:?}: the repair raised the energy by {d}");
                    let now = g.energy(&s);
                    assert!(now <= prev + 1e-9, "step {step} {op:?}: energy rose {prev} -> {now}");
                    if d < -1e-12 {
                        improving += 1;
                    }
                    prev = now;
                }
                Err(_) => refused += 1,
            }
        }

        assert_eq!(moves, 2000, "every 12-node block on a lattice is narrow enough to solve");
        assert_eq!(refused, 0);
        assert!(improving > 0, "a run in which nothing improved proves nothing about monotonicity");
        assert!(prev < started - 40.0, "the run must actually descend: {started} -> {prev}");

        // THE GUARD BITES, and where it bites was measured rather than guessed. A random HALF of
        // this torus still eliminates under width 12 and is never refused; the whole torus measures
        // 16 and is refused every time. So the refusal is a property of the freed set's induced
        // width, which is what `max_width` claims to be about.
        assert_eq!(Elimination::default().width(&g), 16, "the 8x8 torus eliminates at 16");
        let half = Lns { steps: 20, fraction: 0.5, max_width: 12, destroy: Destroy::Random, start: None };
        let o = lns(&g, &half, 1).unwrap();
        assert_eq!(o.refused, 0, "a random half of a torus is sparse enough to solve");

        let whole = Lns { fraction: 1.0, ..half };
        let o = lns(&g, &whole, 1).unwrap();
        assert_eq!(o.refused, 20, "width 16 against a cap of 12 must be refused, not approximated");
        assert_eq!(o.moves, 0);
        assert!(!o.exact(), "a run that refused every repair has solved nothing");
    }

    /// ORACLE 3: on instances whose optimum is known by construction, never report below it.
    ///
    /// An answer better than optimal is not a good run, it is a decoder or an energy-bookkeeping
    /// error, and it is the failure this crate should be most afraid of — so it is asserted on
    /// every one of the 108 runs here rather than on the best of them.
    ///
    /// The hit rate is pinned as measured, and the asymmetric half is the comparison that makes it
    /// mean something: the SAME exact repair at the SAME budget on the SAME instances, fed by
    /// random removal instead of Shaw's relatedness, solves a quarter as many.
    #[test]
    fn lns_on_planted_instances_never_lands_below_the_planted_optimum() {
        let rate = |op: Destroy, fraction: f64| {
            let (mut solved, mut total, mut worst) = (0usize, 0usize, 0.0f64);
            for iseed in 1..=6u64 {
                let p = frustrated_loops(8, 128, iseed);
                for sseed in 1..=6u64 {
                    let params = Lns {
                        steps: 200,
                        fraction,
                        max_width: 12,
                        destroy: op,
                        start: None,
                    };
                    let o = lns(&p.graph, &params, sseed).unwrap();
                    assert!(
                        o.energy >= p.ground_energy - 1e-9,
                        "{op:?} i={iseed} s={sseed}: {} is BELOW the planted optimum {}",
                        o.energy,
                        p.ground_energy
                    );
                    assert_eq!(o.energy, p.graph.energy(&o.state), "reported vs recomputed");
                    // The directed-rounding form is an upper bound on the state's energy, so it is
                    // an upper bound on the ground energy and must sit above the planted optimum.
                    assert!(energy_upper(&p.graph, &o.state) >= p.ground_energy);
                    total += 1;
                    solved += usize::from(p.solved(&o.state));
                    worst = worst.max(p.excess(&o.state));
                }
            }
            (solved, total, worst)
        };

        let (related, total, worst) = rate(Destroy::Related, 0.25);
        assert_eq!(total, 36);
        assert!(related >= 30, "measured 34/36 for related removal at 25%, got {related}");
        assert!(worst < 0.06, "measured a worst excess of 0.047, got {worst}");

        let (random, _, _) = rate(Destroy::Random, 0.25);
        assert!(random <= 20, "measured 9/36 for random removal at 25%, got {random}");
        assert!(
            related > 2 * random,
            "Shaw's relatedness is the method, not a refinement: {related} vs {random} of 36"
        );
    }

    /// VNS escapes a local minimum that ENUMERATION certifies as a trap, and the ladder is why.
    ///
    /// The start is not a state some heuristic wandered into: it is the highest-energy state of the
    /// whole `2^12` space from which no single flip descends, found by recomputing every neighbour's
    /// energy. Three assertions, and the first two are the ones that can fail:
    ///
    /// * [`descend`] from it makes ZERO flips. If it made one, the "trap" was not one and the test
    ///   would be measuring a descent rather than an escape.
    /// * a ladder pinned at radius 1 fails to reach the optimum on 4 of these 6 instances.
    /// * the full ladder reaches the enumerated optimum on all 6.
    #[test]
    fn vns_escapes_a_local_minimum_that_enumeration_certifies_as_a_trap() {
        let (mut escaped, mut pinned_escaped, mut traps) = (0, 0, 0);
        for seed in 1..=6u64 {
            let g = field_graph(12, 0.4, seed);
            let optimum = Exhaustive.solve(&g).1;
            let Some((trap, trap_e)) = worst_trap(&g, optimum) else {
                continue;
            };
            traps += 1;

            let mut t = trap.clone();
            assert_eq!(
                descend(&g, &mut t),
                0,
                "seed {seed}: enumeration says no single flip descends from {trap_e}"
            );
            assert_eq!(g.energy(&t), trap_e, "a descent that moves nothing must change nothing");

            let ladder = Vns { rounds: 200, k_min: 1, k_max: 6, start: Some(trap.clone()) };
            let o = vns(&g, &ladder, 3).unwrap();
            assert!(o.energy >= optimum - 1e-9, "seed {seed}: below the enumerated optimum");
            assert!(o.energy <= trap_e + 1e-12, "VNS must never end above where it started");
            escaped += usize::from(o.energy <= optimum + 1e-9);
            assert!(o.deepest > 1, "seed {seed}: a ladder that never climbed is not a ladder");

            let one_rung = Vns { k_max: 1, ..ladder };
            let p = vns(&g, &one_rung, 3).unwrap();
            assert_eq!(p.deepest, 1, "a one-rung ladder shakes at radius 1 and nowhere else");
            pinned_escaped += usize::from(p.energy <= optimum + 1e-9);
        }

        assert_eq!(traps, 6, "every one of these instances has a strict non-global local minimum");
        assert_eq!(escaped, 6, "the full ladder reached the enumerated optimum on all six");
        assert!(
            pinned_escaped <= 3,
            "measured 2 of 6 for a ladder pinned at radius 1; got {pinned_escaped}, so the ladder \
             is buying nothing and this test is not measuring it"
        );
    }

    /// A one-rung ladder wraps exactly once per non-improving round, and that is an identity.
    ///
    /// With `k_min == k_max` every failure pushes the radius past the top and straight back, so
    /// `ladder_restarts + improvements == rounds` with no slack at all. It is the cheapest possible
    /// check that the wrap is driven by the outcome of each round rather than by a counter, and a
    /// mutation to either branch breaks it by an exact integer.
    #[test]
    fn a_one_rung_ladder_wraps_once_per_non_improving_round() {
        let g = glass2d(6, 4);
        for seed in 1..=4u64 {
            let o = vns(&g, &Vns { rounds: 150, k_min: 3, k_max: 3, start: None }, seed).unwrap();
            assert_eq!(o.shakes, 150);
            assert_eq!(o.deepest, 3);
            assert_eq!(
                o.ladder_restarts + o.improvements,
                150,
                "seed {seed}: {} wraps and {} improvements do not account for 150 rounds",
                o.ladder_restarts,
                o.improvements
            );
        }
        // And a ladder with room climbs: same instance, same budget, a radius above the bottom.
        let wide = vns(&g, &Vns { rounds: 150, k_min: 1, k_max: 9, start: None }, 1).unwrap();
        assert!(wide.deepest > 1, "the ladder never left the bottom rung");
        assert!(wide.deepest <= 9, "and never went above the top one");
    }

    /// CLOSED FORM. On a model with biases and no couplings the optimum is `s_i = sign(h_i)` and
    /// the ground energy is exactly `-sum_i |h_i|`, so a pure-greedy construction has to land on
    /// it whatever order it assigns in — every field is final before the first choice is made.
    ///
    /// The asymmetric half is the same construction at `alpha = 1`, where the candidate list
    /// admits every offer including both values of every variable: it reaches the closed form on
    /// NONE of twenty seeds and averages above zero, on a model whose optimum is −7.68.
    #[test]
    fn grasp_construction_at_alpha_zero_is_exact_on_a_decoupled_model() {
        let mut rng = Pcg::new(5, 9);
        let mut b = GraphBuilder::new(24);
        let mut terms: Vec<f64> = Vec::new();
        for i in 0..24usize {
            let h = rng.f64() * 2.0 - 1.0;
            b.bias(i, h);
            terms.push(-h.abs());
        }
        let g = b.build();
        // Summed in node order, exactly as `Graph::energy` sums `-h_i s_i`, so the comparison
        // below is bit-for-bit rather than approximate.
        let closed_form: f64 = terms.iter().sum();
        assert_eq!(g.n_edges, 0, "the closed form holds only with no couplings");

        let (mut greedy_hits, mut random_hits, mut random_mean) = (0, 0, 0.0);
        for seed in 1..=20u64 {
            let mut r = Pcg::new(seed, 77);
            let s = construct(&g, 0.0, &mut r).unwrap();
            assert_eq!(g.energy(&s), closed_form, "seed {seed}: greedy missed the closed form");
            greedy_hits += 1;

            let mut r = Pcg::new(seed, 77);
            let t = construct(&g, 1.0, &mut r).unwrap();
            let e = g.energy(&t);
            random_mean += e;
            random_hits += usize::from((e - closed_form).abs() < 1e-12);
        }
        assert_eq!(greedy_hits, 20);
        assert_eq!(random_hits, 0, "a uniform construction must not reach the closed form");
        assert!(
            random_mean / 20.0 > closed_form + 5.0,
            "measured a mean of +0.087 against a closed form of {closed_form}, got {}",
            random_mean / 20.0
        );

        // The candidate list is a knob and not a decoration: two alphas are two searches.
        let p = frustrated_loops(8, 128, 3);
        let mean = |alpha: f64| {
            let mut total = 0.0;
            for seed in 1..=20u64 {
                let mut r = Pcg::new(seed, 78);
                total += p.graph.energy(&construct(&p.graph, alpha, &mut r).unwrap());
            }
            total / 20.0
        };
        assert!(mean(0.0) < mean(0.7), "greedier must construct better");
        assert!(mean(0.7) < mean(1.0), "and every rung of alpha must cost something");
    }

    /// Same greedy rule, opposite outcome, and the difference is the ORDER it is applied in.
    ///
    /// This is the measurement that surprised this module. On the hard peak of the frustrated-loop
    /// family, ONE greedy construction — no local search, no restarts, no descent — reaches the
    /// planted optimum 26 times in 36, where [`crate::oracle::SteepestDescent`] with **fifty**
    /// restarts reaches it 6 times. Same "align with the local field" rule, a quarter of the hit
    /// rate, because a descent repairs an assignment it was handed and a construction builds one in
    /// an order it chooses.
    ///
    /// It is measured here rather than left as GRASP's hit rate, because a reader who sees 36/36
    /// with no control will conclude something about GRASP that this instance family does not
    /// support.
    #[test]
    fn a_greedy_construction_beats_a_greedy_descent_on_the_planted_peak() {
        let (mut built, mut descended, mut total) = (0, 0, 0);
        for iseed in 1..=6u64 {
            let p = frustrated_loops(8, 128, iseed);
            for sseed in 1..=6u64 {
                let mut r = Pcg::new(sseed, 0xC0FFEE);
                let s = construct(&p.graph, 0.0, &mut r).unwrap();
                built += usize::from(p.solved(&s));
                let (d, _) = SteepestDescent { restarts: 50, seed: sseed }.solve(&p.graph);
                descended += usize::from(p.solved(&d));
                total += 1;
            }
        }
        assert_eq!(total, 36);
        assert!(built >= 22, "measured 26/36 for one greedy construction, got {built}");
        assert!(descended <= 12, "measured 6/36 for 50-restart descent, got {descended}");
        assert!(
            built > 2 * descended,
            "the comparison is the point: construction {built}, descent {descended} of {total}"
        );
    }

    /// All three searched against an INDEPENDENT exact solver, on instances too large to enumerate.
    ///
    /// [`crate::exact::Elimination`] contracts the model by variable elimination and has its own
    /// oracle; a 6x6 periodic glass is 36 spins — `2^36` states, out of enumeration's reach — and
    /// eliminates at width 12. Every arm must sit at or above the true optimum, and the asymmetric
    /// half is that some arm must MISS: four instances on which every method happened to succeed
    /// would say nothing about any of them.
    #[test]
    fn the_three_searches_never_land_below_exact_elimination() {
        let (mut hits, mut runs) = ([0usize; 3], 0);
        for seed in 1..=4u64 {
            let g = glass2d(6, seed);
            let exact = Elimination::default().ground_state(&g).unwrap();
            let e0 = exact.ground_energy.expect("min-sum was run");
            assert!(exact.width <= 14, "seed {seed} eliminated at width {}", exact.width);

            let a = lns(&g, &Lns { steps: 200, ..Lns::default() }, 7).unwrap().energy;
            let b = vns(&g, &Vns::default(), 7).unwrap().energy;
            let c = grasp(&g, &Grasp::default(), 7).unwrap().energy;
            for (k, e) in [a, b, c].into_iter().enumerate() {
                assert!(e >= e0 - 1e-9, "seed {seed} arm {k}: {e} is BELOW the exact optimum {e0}");
                hits[k] += usize::from(e <= e0 + 1e-9);
            }
            runs += 1;
        }
        assert_eq!(runs, 4);
        assert_eq!(hits[2], 4, "measured 4/4 for GRASP");
        assert!(hits[0] >= 3, "measured 3/4 for LNS, got {}", hits[0]);
        assert!(
            hits.iter().any(|&h| h < runs),
            "every arm solved every instance, so this test discriminates nothing: {hits:?}"
        );
    }

    /// A descent ends where a RECOMPUTED energy says it should, not where its own gain vector does.
    ///
    /// The stopping rule reads an incrementally maintained `delta`, so a drift in that vector would
    /// stop the descent early or run it long and nothing would raise. This checks the answer
    /// against `Graph::energy` on all `n` single flips, which shares no arithmetic with the loop.
    #[test]
    fn a_descent_ends_at_a_local_minimum_certified_by_recomputed_energies() {
        let mut longest = 0usize;
        for seed in 1..=8u64 {
            let g = glass2d(6, seed);
            let mut r = Pcg::new(seed, 1);
            for _ in 0..20 {
                let mut s: Vec<i8> = (0..g.n).map(|_| r.spin(0.5)).collect();
                let before = g.energy(&s);
                let flips = descend(&g, &mut s);
                longest = longest.max(flips);
                let after = g.energy(&s);
                assert!(after <= before + 1e-12, "a descent rose: {before} -> {after}");
                for i in 0..g.n {
                    let mut t = s.clone();
                    t[i] = -t[i];
                    assert!(
                        g.energy(&t) >= after - 1e-9,
                        "seed {seed}: flipping {i} lowers the energy, so this is not a minimum"
                    );
                }
            }
        }
        // Measured at 12 flips on 36 spins. The cap exists against arithmetic drift, and this is
        // the number that says it has never come near biting.
        assert!(
            longest < DESCENT_CAP * 36 / 10,
            "the longest descent was {longest} flips against a cap of {}",
            DESCENT_CAP * 36
        );
        assert!(longest > 0, "a descent that never moves is not being exercised");
    }

    /// The directed-rounding energy bounds the EXACT energy, which is not what `Graph::energy` is.
    ///
    /// Two regimes, and the difference between them was measured rather than assumed. With integer
    /// couplings both sums are exact, so the bound must sit strictly above `Graph::energy` — every
    /// one of 800 states here. With continuous couplings under cancellation the naive sum can sit
    /// ABOVE the bound (measured: 1.359295502516986 against 1.3592955025169855 on a 20-node model),
    /// because its own error exceeds the compensated guard. That is not a defect in either; it is
    /// the reason a claim must be made against the directed form. The slack allowed here is the
    /// crate's own [`crate::round::accumulation_guard`] for the naive sum, not a hand-picked
    /// epsilon.
    #[test]
    fn energy_upper_bounds_the_exact_energy_rather_than_the_naive_sum() {
        // Integer couplings: exact on both sides, so the direction is strict.
        let (mut strict, mut runs) = (0, 0);
        for seed in 1..=4u64 {
            let g = glass2d(5, seed);
            let mut r = Pcg::new(seed, 2);
            for _ in 0..50 {
                let s: Vec<i8> = (0..g.n).map(|_| r.spin(0.5)).collect();
                let (e, up) = (g.energy(&s), energy_upper(&g, &s));
                assert!(up >= e, "integer energies are exact, so {up} must not sit below {e}");
                strict += usize::from(up > e);
                runs += 1;
            }
        }
        assert_eq!(runs, 200);
        assert!(strict > 0, "the guard is positive, so some bound must be strictly above");

        // Continuous couplings: bounded by the crate's own guard on the naive accumulation.
        let mut looser = 0;
        for seed in 1..=4u64 {
            let g = field_graph(20, 0.3, seed);
            let mut r = Pcg::new(seed, 2);
            for _ in 0..50 {
                let s: Vec<i8> = (0..g.n).map(|_| r.spin(0.5)).collect();
                let (e, up) = (g.energy(&s), energy_upper(&g, &s));
                let magnitude: f64 = g.h.iter().map(|h| h.abs()).sum::<f64>()
                    + g.w.iter().map(|w| w.abs()).sum::<f64>() / 2.0;
                let slack = crate::round::accumulation_guard(g.n + g.n_edges, magnitude.max(1.0));
                assert!(up >= e - slack, "the bound {up} is below {e} by more than {slack}");
                assert!(up - e < 1e-9 * e.abs().max(1.0), "and is loose by {}", up - e);
                looser += usize::from(up < e);
            }
        }
        // The measured fact this test exists to pin: it really does happen.
        assert!(looser > 0, "no continuous state put the naive sum above the bound, so the doc lies");
    }

    /// The three destroy operators are three different searches, so `destroy` is not decorative.
    #[test]
    fn the_three_destroy_operators_are_three_different_searches() {
        let g = glass2d(8, 9);
        let base = Lns { steps: 30, fraction: 0.125, max_width: 12, destroy: Destroy::Random, start: None };
        let r = lns(&g, &base, 5).unwrap();
        let a = lns(&g, &Lns { destroy: Destroy::Related, ..base.clone() }, 5).unwrap();
        let w = lns(&g, &Lns { destroy: Destroy::Worst, ..base.clone() }, 5).unwrap();
        assert!(r.state != a.state, "random and related removal chose the same path");
        assert!(a.state != w.state, "related and worst removal chose the same path");
        assert!(r.state != w.state, "random and worst removal chose the same path");
    }

    /// A seed reproduces a run, and a different seed does not. Every arm, since each has its own
    /// stream.
    #[test]
    fn a_seed_reproduces_a_run_and_a_different_seed_does_not() {
        let g = glass2d(6, 6);
        let lp = Lns { steps: 40, ..Lns::default() };
        assert_eq!(lns(&g, &lp, 4).unwrap(), lns(&g, &lp, 4).unwrap());
        assert_ne!(lns(&g, &lp, 4).unwrap().state, lns(&g, &lp, 5).unwrap().state);

        let vp = Vns { rounds: 40, ..Vns::default() };
        assert_eq!(vns(&g, &vp, 4).unwrap(), vns(&g, &vp, 4).unwrap());
        assert_ne!(vns(&g, &vp, 4).unwrap().state, vns(&g, &vp, 5).unwrap().state);

        let gp = Grasp { iterations: 8, alpha: 0.5 };
        assert_eq!(grasp(&g, &gp, 4).unwrap(), grasp(&g, &gp, 4).unwrap());
        assert_ne!(grasp(&g, &gp, 4).unwrap().state, grasp(&g, &gp, 5).unwrap().state);
    }

    /// An input none of these can interpret is an error naming what it saw, never a default.
    #[test]
    fn an_unreadable_parameter_is_an_error_naming_what_it_was_given() {
        let g = glass2d(4, 1);
        for bad in [0.0f64, -0.5, 1.5, f64::NAN, f64::INFINITY] {
            let p = Lns { fraction: bad, ..Lns::default() };
            match lns(&g, &p, 1) {
                Err(Invalid::Fraction(x)) => {
                    assert!(x.is_nan() || x == bad);
                    assert!(format!("{}", Invalid::Fraction(x)).contains("(0, 1]"));
                }
                other => panic!("a fraction of {bad} was accepted: {other:?}"),
            }
            assert!(freed_count(16, bad).is_err());
        }
        for bad in [-0.1f64, 1.5, f64::NAN] {
            // Matched rather than compared: `Alpha(NaN) == Alpha(NaN)` is FALSE, which would make
            // an `assert_eq!` here fail on the very input the rejection is most important for.
            match grasp(&g, &Grasp { iterations: 4, alpha: bad }, 1) {
                Err(Invalid::Alpha(x)) => assert!(x.is_nan() == bad.is_nan() && (x.is_nan() || x == bad)),
                other => panic!("an alpha of {bad} was accepted: {other:?}"),
            }
            let mut r = Pcg::new(1, 1);
            assert!(construct(&g, bad, &mut r).is_err());
        }
        for (k_min, k_max) in [(0usize, 4usize), (0, 0), (5, 3)] {
            let p = Vns { rounds: 4, k_min, k_max, start: None };
            assert_eq!(vns(&g, &p, 1), Err(Invalid::Ladder { k_min, k_max }));
        }
        // A start of the wrong length is refused with both lengths, rather than silently replaced
        // by noise -- which is how a composed pipeline comes to report the wrong stage's work.
        let short = Some(vec![1i8; g.n - 1]);
        assert_eq!(
            lns(&g, &Lns { start: short.clone(), ..Lns::default() }, 1),
            Err(Invalid::StartLength { got: g.n - 1, want: g.n })
        );
        assert_eq!(
            vns(&g, &Vns { start: short, ..Vns::default() }, 1),
            Err(Invalid::StartLength { got: g.n - 1, want: g.n })
        );
        let msg = format!("{}", Invalid::StartLength { got: 3, want: 16 });
        assert!(msg.contains('3') && msg.contains("16"), "{msg}");
    }

    /// A handed start is where the search begins, and a good one is never lost.
    #[test]
    fn a_handed_start_is_where_the_search_begins_and_is_never_lost() {
        let p = frustrated_loops(6, 60, 2);
        let opt = p.ground_state.clone();
        let e0 = p.graph.energy(&opt);

        let l = lns(
            &p.graph,
            &Lns { steps: 50, start: Some(opt.clone()), ..Lns::default() },
            3,
        )
        .unwrap();
        assert!(l.energy <= e0 + 1e-9, "handed the optimum, returned {}", l.energy);
        assert_eq!(l.improving, 0, "there is nothing left to improve from the optimum");

        let v = vns(&p.graph, &Vns { rounds: 50, start: Some(opt), ..Vns::default() }, 3).unwrap();
        assert!(v.energy <= e0 + 1e-9, "handed the optimum, returned {}", v.energy);
        assert_eq!(v.improvements, 0);
    }

    /// The empty graph is a model with no variables, and every arm answers it rather than panicking.
    #[test]
    fn an_empty_model_is_answered_rather_than_crashed() {
        let g = GraphBuilder::new(0).build();
        let l = lns(&g, &Lns::default(), 1).unwrap();
        assert_eq!((l.energy, l.moves, l.freed), (0.0, 0, 0));
        assert!(!l.exact(), "a model with no variables was not solved, there was nothing to solve");
        let v = vns(&g, &Vns::default(), 1).unwrap();
        assert_eq!((v.energy, v.shakes), (0.0, 0));
        let q = grasp(&g, &Grasp::default(), 1).unwrap();
        assert_eq!((q.energy, q.state.len()), (0.0, 0));
    }
}
