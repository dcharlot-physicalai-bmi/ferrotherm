//! Generalized belief propagation on region graphs — the Kikuchi cluster variation method.
//!
//! Yedidia, Freeman & Weiss, "Constructing free-energy approximations and generalized belief
//! propagation algorithms", IEEE Trans. Inf. Theory 51:2282, 2005, building on Kikuchi's cluster
//! variation method (Phys. Rev. 81:988, 1951).
//!
//! # What a region graph is, and what it buys
//!
//! [`crate::meanfield::belief_propagation`] approximates `ln Z` by breaking the model into nodes
//! and edges, counting each edge once and each node `1 − d_i` times. That is the Bethe
//! approximation, and it is exact on a tree because a tree is exactly what that decomposition
//! describes. On a lattice it is not, and no amount of iterating fixes it: the error is in the
//! *decomposition*, not in the solver.
//!
//! The cluster variation method lets the decomposition be chosen. Pick a set of **outer regions** —
//! sets of variables, typically the faces of the lattice rather than its edges — close them under
//! intersection, and give every region `R` a Moebius counting number
//!
//! ```text
//!   c_R = 1 − Σ_{R' ⊋ R} c_{R'}
//! ```
//!
//! so that, summed over the regions containing it, every variable and every interaction is counted
//! **exactly once**. Those counting numbers are integers, this module stores them as integers, and
//! [`RegionGraph::variable_counting`] asserts the identity exactly rather than to a tolerance —
//! see [`RegionGraph::counts_each_variable_once`].
//!
//! The resulting Kikuchi free energy
//!
//! ```text
//!   F  =  Σ_R c_R [ Σ_{x_R} b_R(x_R) β E_R(x_R)  +  Σ_{x_R} b_R(x_R) ln b_R(x_R) ],   ln Z ≈ −F
//! ```
//!
//! contains Bethe as the special case "outer regions are the single edges", which
//! [`RegionGraph::bethe`] builds and the tests check against `meanfield`'s BP to `1e-9`.
//!
//! # Why the plaquette is the example everyone uses
//!
//! On a single four-cycle, Bethe is wrong — measurably, not marginally. Take the same four spins as
//! **one** outer region and the region graph has one node, `c = 1`, no messages, and a belief that
//! is the Boltzmann distribution itself: Kikuchi is `ln Z` exactly. That contrast is the headline
//! test here, and it is asserted in both directions on purpose. A test that only checked "GBP is
//! close to exact" would pass for an implementation that silently fell back to Bethe.
//!
//! # The messages
//!
//! The parent-to-child algorithm (YFW section VII-B). With `E(R)` the set of region-graph edges
//! entering `R`'s own subtree from outside it, the belief at a region is
//!
//! ```text
//!   b_R(x_R)  ∝  f_R(x_R) · Π_{(I→J) ∈ E(R)} m_{I→J}(x_J)
//! ```
//!
//! where `f_R` is the product of every interaction whose variables all lie in `R`. Imposing
//! `b_R = Σ_{x_P ∖ x_R} b_P` on a parent `P` and solving for the one message that appears on one
//! side only gives the update
//!
//! ```text
//!   m_{P→R}(x_R)  =  [ Σ_{x_P∖x_R} f_{P∖R}(x_P) · Π_{N(P,R)} m ]  /  Π_{D(P,R)} m
//!   N(P,R) = E(P) ∖ E(R),      D(P,R) = E(R) ∖ E(P) ∖ {(P→R)}
//! ```
//!
//! with `f_{P∖R}` the interactions inside `P` that are **not** inside `R`. Everything is carried in
//! logs and renormalised each pass, so `β` may be large without overflow; damping is geometric,
//! which is what damping in log space is.
//!
//! # This is not a bound, and does not pretend to be
//!
//! Nothing here goes through [`crate::round`]. The Kikuchi free energy sits on either side of the
//! truth — that is the whole difference between it and [`crate::trw`], which is an upper bound and
//! sums with [`crate::round::sum_up`] for exactly that reason. Directed rounding on a quantity with
//! no direction would dress an approximation as a guarantee, which is the failure this crate's
//! rounding module exists to prevent rather than to spread.

use crate::graph::Graph;

/// Largest region this module will build a table for.
///
/// A belief is `2^|R|` doubles, so sixteen variables is 512 KB per region and the message loop over
/// a parent is `2^|P|` — the cost of the cluster variation method is exponential in the cluster,
/// which is the trade it offers. Refused rather than attempted, because the failure mode of the
/// alternative is an allocation nobody asked for.
pub const MAX_REGION: usize = 16;

/// Why a set of outer regions is not a region graph.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RegionError {
    /// No outer regions were given for a model that has variables.
    NoOuterRegions {
        /// Variables the model has, all of them uncovered.
        n: usize,
    },
    /// Outer region `index` names no variables.
    EmptyRegion {
        /// Position in the list handed in.
        index: usize,
    },
    /// Outer region `index` names a variable the model does not have.
    OutOfRange {
        /// Position in the list handed in.
        index: usize,
        /// The offending variable.
        var: usize,
        /// Variables the model has.
        n: usize,
    },
    /// Outer region `index` is larger than [`MAX_REGION`], so its belief table is refused.
    TooLarge {
        /// Position in the list handed in.
        index: usize,
        /// Variables in that region.
        vars: usize,
        /// The cap, since a table costs `2^vars`.
        max: usize,
    },
    /// A variable lies in no outer region, so no counting number can sum to one over it.
    Uncovered {
        /// The variable nothing covers.
        var: usize,
    },
}

impl core::fmt::Display for RegionError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            RegionError::NoOuterRegions { n } => write!(
                f,
                "a region graph over {n} variables needs at least one outer region; the cluster \
                 variation method has nothing to close under intersection"
            ),
            RegionError::EmptyRegion { index } => {
                write!(f, "outer region {index} names no variables")
            }
            RegionError::OutOfRange { index, var, n } => write!(
                f,
                "outer region {index} names variable {var}, but the model has {n} of them (0..{n})"
            ),
            RegionError::TooLarge { index, vars, max } => write!(
                f,
                "outer region {index} has {vars} variables and its belief table would be 2^{vars}; \
                 the cap is {max}. Use smaller clusters -- the cluster variation method costs \
                 exponentially in the cluster, which is the trade it offers."
            ),
            RegionError::Uncovered { var } => write!(
                f,
                "variable {var} lies in no outer region, so the counting numbers cannot sum to one \
                 over it and the Kikuchi free energy would silently drop it"
            ),
        }
    }
}

impl core::error::Error for RegionError {}

/// A region graph: outer regions, everything their intersections generate, and the Moebius counting
/// numbers over the resulting inclusion lattice.
///
/// Regions are stored with their variables sorted ascending, and the regions themselves are ordered
/// by **decreasing size**, ties broken lexicographically. That ordering is load-bearing rather than
/// cosmetic: a proper superset is strictly larger, so every ancestor of a region precedes it, which
/// is what lets the Moebius recursion run in one forward pass and what makes the whole structure a
/// deterministic function of the outer regions.
#[derive(Clone, Debug)]
pub struct RegionGraph {
    /// Variables the model has; every one of them lies in at least one region.
    pub n: usize,
    /// Each region's variables, sorted ascending. Regions are ordered largest first.
    pub regions: Vec<Vec<usize>>,
    /// Moebius counting number `c_R = 1 − Σ_{R' ⊋ R} c_{R'}`, parallel to [`RegionGraph::regions`].
    ///
    /// Integer by construction — the recursion starts at one and only ever subtracts integers — and
    /// stored as one so the counting identity can be asserted exactly.
    pub counting: Vec<i64>,
    /// Hasse edges `(parent, child)`: `child ⊊ parent` with no region strictly between them.
    pub edges: Vec<(usize, usize)>,
    /// Every region strictly containing this one, as indices.
    pub ancestors: Vec<Vec<usize>>,
    /// Every region strictly contained in this one, as indices.
    pub descendants: Vec<Vec<usize>>,
}

impl RegionGraph {
    /// Close `outer` under intersection and compute the counting numbers.
    ///
    /// Outer regions are normalised (sorted, deduplicated) and reduced to an antichain: one that is
    /// a subset of another is dropped, since it is generated anyway and an outer region that is
    /// somebody's subset would be given `c = 1` twice.
    ///
    /// # Errors
    ///
    /// [`RegionError::NoOuterRegions`] when `n > 0` and nothing was given,
    /// [`RegionError::EmptyRegion`] for a region with no variables,
    /// [`RegionError::OutOfRange`] for a variable past `n`,
    /// [`RegionError::TooLarge`] past [`MAX_REGION`], and
    /// [`RegionError::Uncovered`] when some variable lies in no region at all.
    pub fn new(n: usize, outer: &[Vec<usize>]) -> Result<RegionGraph, RegionError> {
        if outer.is_empty() {
            if n == 0 {
                return Ok(RegionGraph {
                    n,
                    regions: Vec::new(),
                    counting: Vec::new(),
                    edges: Vec::new(),
                    ancestors: Vec::new(),
                    descendants: Vec::new(),
                });
            }
            return Err(RegionError::NoOuterRegions { n });
        }
        let mut norm: Vec<Vec<usize>> = Vec::with_capacity(outer.len());
        for (index, r) in outer.iter().enumerate() {
            if r.is_empty() {
                return Err(RegionError::EmptyRegion { index });
            }
            let mut v = r.clone();
            v.sort_unstable();
            v.dedup();
            if let Some(&var) = v.iter().find(|&&x| x >= n) {
                return Err(RegionError::OutOfRange { index, var, n });
            }
            if v.len() > MAX_REGION {
                return Err(RegionError::TooLarge { index, vars: v.len(), max: MAX_REGION });
            }
            norm.push(v);
        }
        let mut covered = vec![false; n];
        for v in &norm {
            for &x in v {
                covered[x] = true;
            }
        }
        if let Some(var) = (0..n).find(|&v| !covered[v]) {
            return Err(RegionError::Uncovered { var });
        }
        // Antichain: largest first, then keep only what nothing already kept contains.
        norm.sort_by(|a, b| b.len().cmp(&a.len()).then_with(|| a.cmp(b)));
        let mut anti: Vec<Vec<usize>> = Vec::new();
        for v in norm {
            if !anti.iter().any(|k| contains(k, &v)) {
                anti.push(v);
            }
        }
        // Close under intersection. The lattice is finite, so this terminates; each pass adds at
        // least one region or is the last.
        let mut set: std::collections::BTreeSet<Vec<usize>> = anti.into_iter().collect();
        loop {
            let cur: Vec<Vec<usize>> = set.iter().cloned().collect();
            let mut added = false;
            for a in 0..cur.len() {
                for b in (a + 1)..cur.len() {
                    let x = intersect(&cur[a], &cur[b]);
                    if !x.is_empty() && set.insert(x) {
                        added = true;
                    }
                }
            }
            if !added {
                break;
            }
        }
        let mut regions: Vec<Vec<usize>> = set.into_iter().collect();
        regions.sort_by(|a, b| b.len().cmp(&a.len()).then_with(|| a.cmp(b)));
        let k = regions.len();

        let mut ancestors: Vec<Vec<usize>> = vec![Vec::new(); k];
        let mut descendants: Vec<Vec<usize>> = vec![Vec::new(); k];
        for r in 0..k {
            for s in 0..k {
                if r != s && contains(&regions[s], &regions[r]) {
                    ancestors[r].push(s);
                    descendants[s].push(r);
                }
            }
        }
        // Moebius, in one forward pass: a proper superset is strictly larger, so it sorted earlier.
        let mut counting = vec![0i64; k];
        for r in 0..k {
            let mut c = 1i64;
            for &a in &ancestors[r] {
                c -= counting[a];
            }
            counting[r] = c;
        }
        // Hasse edges: a cover relation is a containment with nothing strictly in between.
        let mut edges = Vec::new();
        for p in 0..k {
            for &c in &descendants[p] {
                let covered = descendants[p].iter().any(|&m| {
                    m != c && contains(&regions[m], &regions[c]) && regions[m].len() > regions[c].len()
                });
                if !covered {
                    edges.push((p, c));
                }
            }
        }
        Ok(RegionGraph { n, regions, counting, edges, ancestors, descendants })
    }

    /// The Bethe region graph: every coupling is an outer region, plus a singleton for any node the
    /// couplings miss.
    ///
    /// Closing that under intersection produces the shared endpoints, whose counting number comes
    /// out at `1 − d_i` — the Bethe coefficient, derived rather than written down. This is the
    /// region graph the tests hold against [`crate::meanfield::belief_propagation`].
    ///
    /// # Panics
    ///
    /// Never, and the reason is worth stating: every region here has one or two variables, so
    /// [`MAX_REGION`] cannot bite; the singletons cover exactly the nodes no edge does, so nothing
    /// is uncovered; and [`crate::graph::GraphBuilder`] cannot emit a neighbour index past its own
    /// node count.
    #[must_use]
    pub fn bethe(g: &Graph) -> RegionGraph {
        let mut outer: Vec<Vec<usize>> = Vec::with_capacity(g.n_edges + 1);
        for i in 0..g.n {
            let mut isolated = true;
            for e in g.offset[i]..g.offset[i + 1] {
                let j = g.nbr[e] as usize;
                isolated = false;
                if j > i {
                    outer.push(vec![i, j]);
                }
            }
            if isolated {
                outer.push(vec![i]);
            }
        }
        RegionGraph::new(g.n, &outer).expect("edges are small and every node is covered")
    }

    /// Region count.
    #[must_use]
    pub fn len(&self) -> usize {
        self.regions.len()
    }

    /// Whether there are no regions at all, which happens only for a model with no variables.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.regions.is_empty()
    }

    /// How many times the counting numbers count variable `v`: `Σ_{R ∋ v} c_R`, which must be `1`.
    ///
    /// Integer arithmetic, so the identity is exact rather than nearly so.
    #[must_use]
    pub fn variable_counting(&self, v: usize) -> i64 {
        self.regions
            .iter()
            .zip(&self.counting)
            .filter(|(r, _)| r.binary_search(&v).is_ok())
            .map(|(_, &c)| c)
            .sum()
    }

    /// How many times the counting numbers count the pair `(u, v)`: `Σ_{R ⊇ {u,v}} c_R`.
    ///
    /// `1` for any pair that lies inside some region — which is every pair carrying an interaction,
    /// if the outer regions were chosen to cover the model's couplings — and `0` for a pair no
    /// region holds, since such a pair's coupling is dropped by the approximation rather than
    /// miscounted.
    #[must_use]
    pub fn pair_counting(&self, u: usize, v: usize) -> i64 {
        self.regions
            .iter()
            .zip(&self.counting)
            .filter(|(r, _)| r.binary_search(&u).is_ok() && r.binary_search(&v).is_ok())
            .map(|(_, &c)| c)
            .sum()
    }

    /// Whether every variable is counted exactly once — the invariant the whole construction exists
    /// to satisfy.
    #[must_use]
    pub fn counts_each_variable_once(&self) -> bool {
        (0..self.n).all(|v| self.variable_counting(v) == 1)
    }

    /// Whether every coupling of `g` lies inside some region and is counted exactly once.
    ///
    /// A coupling no region holds is silently absent from the Kikuchi free energy, which is a
    /// different model rather than an approximation of this one.
    #[must_use]
    pub fn counts_each_coupling_once(&self, g: &Graph) -> bool {
        (0..g.n).all(|i| {
            (g.offset[i]..g.offset[i + 1])
                .all(|e| g.nbr[e] as usize <= i || self.pair_counting(i, g.nbr[e] as usize) == 1)
        })
    }
}

/// The four-node faces of a `w` by `h` grid, indexed `y * w + x` to match [`crate::ising::grid2d`].
///
/// These are the outer regions of the square-lattice cluster variation method, the approximation
/// Kikuchi's paper is about: `(w−1)(h−1)` plaquettes, whose intersections generate the shared bonds
/// (`c = −1`) and the shared corners (`c = +1`).
///
/// Empty when either side is below two, since a grid with no face has no plaquette to take.
#[must_use]
pub fn grid_plaquettes(w: usize, h: usize) -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    for y in 0..h.saturating_sub(1) {
        for x in 0..w.saturating_sub(1) {
            out.push(vec![y * w + x, y * w + x + 1, (y + 1) * w + x, (y + 1) * w + x + 1]);
        }
    }
    out
}

/// What a generalized belief propagation run produced.
#[derive(Clone, Debug)]
pub struct Gbp {
    /// Inverse temperature the messages were passed at.
    pub beta: f64,
    /// Kikuchi approximation to `ln Z(β)`, equal to minus the Kikuchi free energy at the final
    /// beliefs. Neither an upper nor a lower bound.
    pub log_z: f64,
    /// The Kikuchi free energy itself, `βF ≈ −ln Z`.
    pub free_energy: f64,
    /// Belief `b_R(x_R)` per region, each of length `2^|R|`, indexed so bit `b` of the index is the
    /// spin of the region's `b`-th variable: set means `+1`.
    pub beliefs: Vec<Vec<f64>>,
    /// Single-site magnetisations `⟨s_i⟩`, read off the smallest region containing each site.
    pub m: Vec<f64>,
    /// Largest change in any log-message on the last pass.
    pub residual: f64,
    /// Passes actually run, which is the cap when it did not converge.
    pub iterations: usize,
    /// Largest disagreement between a parent belief marginalised onto a child and that child's own
    /// belief. Zero at a fixed point; the Kikuchi free energy is a functional of beliefs on the
    /// local polytope, and this is the distance from it.
    pub consistency: f64,
}

impl Gbp {
    /// Whether the last pass moved every log-message by less than `tol`.
    #[must_use]
    pub fn converged(&self, tol: f64) -> bool {
        self.residual < tol
    }
}

/// `ln f_R(x_R)` for every configuration of region `vars`: `β(Σ J s s + Σ h s)` over the
/// interactions that lie **entirely inside** the region.
///
/// This is `−β` times the induced sub-model's energy under the crate's `E = −J s s − h s`
/// convention, and the tests check it against [`Graph::energy`] on that sub-model rather than
/// against itself, because a sign here is a sign in every number this module reports.
fn region_log_factor(g: &Graph, beta: f64, vars: &[usize]) -> Vec<f64> {
    let k = vars.len();
    let mut out = vec![0.0f64; 1usize << k];
    for (t, slot) in out.iter_mut().enumerate() {
        let sp = |b: usize| if (t >> b) & 1 == 1 { 1.0f64 } else { -1.0 };
        let mut l = 0.0;
        for (b, &i) in vars.iter().enumerate() {
            l += g.h[i] * sp(b);
        }
        for a in 0..k {
            for b in (a + 1)..k {
                if let Some(j) = coupling(g, vars[a], vars[b]) {
                    l += j * sp(a) * sp(b);
                }
            }
        }
        *slot = beta * l;
    }
    out
}

/// `ln f_{P∖R}(x_P)`: the interactions inside parent `pv` that are **not** inside child `cv`.
///
/// Together with [`region_log_factor`] on the child this reconstructs the parent's factor exactly,
/// which is the bookkeeping the parent-to-child rule rests on and which the tests assert.
fn split_log_factor(g: &Graph, beta: f64, pv: &[usize], cv: &[usize]) -> Vec<f64> {
    let k = pv.len();
    let inr: Vec<bool> = pv.iter().map(|x| cv.binary_search(x).is_ok()).collect();
    let mut out = vec![0.0f64; 1usize << k];
    for (t, slot) in out.iter_mut().enumerate() {
        let sp = |b: usize| if (t >> b) & 1 == 1 { 1.0f64 } else { -1.0 };
        let mut l = 0.0;
        for (b, &i) in pv.iter().enumerate() {
            if !inr[b] {
                l += g.h[i] * sp(b);
            }
        }
        for a in 0..k {
            for b in (a + 1)..k {
                if inr[a] && inr[b] {
                    continue;
                }
                if let Some(j) = coupling(g, pv[a], pv[b]) {
                    l += j * sp(a) * sp(b);
                }
            }
        }
        *slot = beta * l;
    }
    out
}

fn coupling(g: &Graph, i: usize, j: usize) -> Option<f64> {
    (g.offset[i]..g.offset[i + 1]).find(|&e| g.nbr[e] as usize == j).map(|e| g.w[e])
}

fn contains(big: &[usize], small: &[usize]) -> bool {
    small.iter().all(|x| big.binary_search(x).is_ok())
}

fn intersect(a: &[usize], b: &[usize]) -> Vec<usize> {
    a.iter().copied().filter(|x| b.binary_search(x).is_ok()).collect()
}

/// Bit positions, within `outer`'s variable list, of each of `inner`'s variables.
fn positions(outer: &[usize], inner: &[usize]) -> Vec<usize> {
    inner
        .iter()
        .map(|x| outer.iter().position(|y| y == x).expect("inner is a subset of outer"))
        .collect()
}

/// Re-index a configuration of the outer region as one of the inner region.
#[inline]
fn project(cfg: usize, pos: &[usize]) -> usize {
    let mut t = 0usize;
    for (b, &p) in pos.iter().enumerate() {
        t |= ((cfg >> p) & 1) << b;
    }
    t
}

fn log_sum_exp(v: &[f64]) -> f64 {
    let m = v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !m.is_finite() {
        return m;
    }
    m + v.iter().map(|&x| (x - m).exp()).sum::<f64>().ln()
}

/// Everything about one region-graph edge that does not change between passes.
struct EdgePlan {
    parent: usize,
    child: usize,
    /// Bit positions of the child's variables inside the parent.
    child_pos: Vec<usize>,
    /// `N(P,R) = E(P) ∖ E(R)`: messages multiplied in, positioned inside the parent.
    numer: Vec<(usize, Vec<usize>)>,
    /// `D(P,R) = E(R) ∖ E(P) ∖ {(P→R)}`: messages divided out, positioned inside the child.
    denom: Vec<(usize, Vec<usize>)>,
    /// `ln f_{P∖R}` over the parent's configurations.
    extra: Vec<f64>,
}

/// Generalized belief propagation on `rg`, parent-to-child, damped in log space from uniform
/// messages; returns the Kikuchi free energy at the resulting beliefs.
///
/// `damping` is the weight kept on the previous log-message, so `0.0` is undamped Gauss-Seidel and
/// larger values trade speed for the convergence a loopy region graph often needs.
///
/// # Panics
///
/// If `rg` was not built over `g`'s variables, or `damping` is outside `[0, 1)`, or `beta` is not
/// finite — each of which produces numbers rather than an error, which is worse.
#[must_use]
pub fn generalized_belief_propagation(
    g: &Graph,
    rg: &RegionGraph,
    beta: f64,
    iters: usize,
    damping: f64,
) -> Gbp {
    assert_eq!(rg.n, g.n, "the region graph was built over a different model");
    assert!((0.0..1.0).contains(&damping), "damping must be in [0, 1)");
    assert!(beta.is_finite(), "beta must be finite; got {beta}");

    let k = rg.len();
    let local: Vec<Vec<f64>> =
        rg.regions.iter().map(|vars| region_log_factor(g, beta, vars)).collect();

    // inside[r] marks R and its descendants: the subtree E(R) is defined against.
    let mut inside: Vec<Vec<bool>> = vec![vec![false; k]; k];
    for r in 0..k {
        inside[r][r] = true;
        for &d in &rg.descendants[r] {
            inside[r][d] = true;
        }
    }
    // E(R): every region-graph edge from outside R's subtree into it.
    let entering: Vec<Vec<usize>> = (0..k)
        .map(|r| {
            (0..rg.edges.len())
                .filter(|&e| inside[r][rg.edges[e].1] && !inside[r][rg.edges[e].0])
                .collect()
        })
        .collect();

    let plans: Vec<EdgePlan> = rg
        .edges
        .iter()
        .enumerate()
        .map(|(e, &(p, c))| {
            let pv = &rg.regions[p];
            let cv = &rg.regions[c];
            let numer = entering[p]
                .iter()
                .filter(|x| !entering[c].contains(x))
                .map(|&f| (f, positions(pv, &rg.regions[rg.edges[f].1])))
                .collect();
            let denom = entering[c]
                .iter()
                .filter(|&&x| x != e && !entering[p].contains(&x))
                .map(|&f| (f, positions(cv, &rg.regions[rg.edges[f].1])))
                .collect();
            EdgePlan {
                parent: p,
                child: c,
                child_pos: positions(pv, cv),
                numer,
                denom,
                extra: split_log_factor(g, beta, pv, cv),
            }
        })
        .collect();

    // Uniform messages, normalised: a log-message always sums to one in probability.
    let mut log_m: Vec<Vec<f64>> = rg
        .edges
        .iter()
        .map(|&(_, c)| {
            let kc = rg.regions[c].len();
            vec![-(kc as f64) * std::f64::consts::LN_2; 1usize << kc]
        })
        .collect();

    let mut residual = f64::INFINITY;
    let mut it = 0;
    while it < iters && residual > 1e-14 {
        residual = 0.0;
        for (e, plan) in plans.iter().enumerate() {
            let kp = rg.regions[plan.parent].len();
            let kc = rg.regions[plan.child].len();
            let mut acc = vec![f64::NEG_INFINITY; 1usize << kc];
            for t in 0..(1usize << kp) {
                let mut val = plan.extra[t];
                for (mi, pos) in &plan.numer {
                    val += log_m[*mi][project(t, pos)];
                }
                let u = project(t, &plan.child_pos);
                // log-add in place, max-shifted so a large beta cannot overflow the exponential.
                let (lo, hi) = if acc[u] < val { (acc[u], val) } else { (val, acc[u]) };
                acc[u] = if lo.is_finite() { hi + (lo - hi).exp().ln_1p() } else { hi };
            }
            for u in 0..acc.len() {
                for (mi, pos) in &plan.denom {
                    acc[u] -= log_m[*mi][project(u, pos)];
                }
            }
            let lse = log_sum_exp(&acc);
            let mut next: Vec<f64> = acc
                .iter()
                .enumerate()
                .map(|(u, &a)| {
                    let fresh = a - lse;
                    if damping > 0.0 { damping * log_m[e][u] + (1.0 - damping) * fresh } else { fresh }
                })
                .collect();
            let renorm = log_sum_exp(&next);
            for x in &mut next {
                *x -= renorm;
            }
            for u in 0..next.len() {
                residual = residual.max((next[u] - log_m[e][u]).abs());
            }
            log_m[e] = next;
        }
        it += 1;
    }

    // Beliefs, then the free energy at them.
    let mut beliefs: Vec<Vec<f64>> = Vec::with_capacity(k);
    for r in 0..k {
        let kr = rg.regions[r].len();
        let mut lb = local[r].clone();
        for &f in &entering[r] {
            let pos = positions(&rg.regions[r], &rg.regions[rg.edges[f].1]);
            for (t, x) in lb.iter_mut().enumerate() {
                *x += log_m[f][project(t, &pos)];
            }
        }
        let lse = log_sum_exp(&lb);
        let b: Vec<f64> = lb.iter().map(|&x| (x - lse).exp()).collect();
        debug_assert_eq!(b.len(), 1usize << kr);
        beliefs.push(b);
    }

    let free_energy = kikuchi_free_energy(g, rg, beta, &beliefs);

    // Magnetisations from the smallest region holding each site: the most refined belief there is.
    let mut m = vec![0.0; g.n];
    for i in 0..g.n {
        if let Some(r) = (0..k)
            .filter(|&r| rg.regions[r].binary_search(&i).is_ok())
            .min_by_key(|&r| rg.regions[r].len())
        {
            let b = positions(&rg.regions[r], &[i])[0];
            m[i] = beliefs[r]
                .iter()
                .enumerate()
                .map(|(t, &p)| if (t >> b) & 1 == 1 { p } else { -p })
                .sum();
        }
    }

    // Local-polytope consistency: a parent marginalised onto a child must be that child.
    let mut consistency = 0.0f64;
    for plan in &plans {
        let kp = rg.regions[plan.parent].len();
        let kc = rg.regions[plan.child].len();
        let mut down = vec![0.0f64; 1usize << kc];
        for t in 0..(1usize << kp) {
            down[project(t, &plan.child_pos)] += beliefs[plan.parent][t];
        }
        for u in 0..down.len() {
            consistency = consistency.max((down[u] - beliefs[plan.child][u]).abs());
        }
    }

    Gbp {
        beta,
        log_z: -free_energy,
        free_energy,
        beliefs,
        m,
        residual,
        iterations: it,
        consistency,
    }
}

/// The Kikuchi free energy `βF = Σ_R c_R [⟨βE_R⟩ + Σ b_R ln b_R]` at the beliefs given, so that
/// `ln Z ≈ −βF`.
///
/// A functional of the beliefs alone: it may be evaluated at anything, not only at a fixed point,
/// which is what makes it a variational objective rather than a readout.
///
/// # Panics
///
/// If `rg` was not built over `g`, or `beliefs` is not one table of `2^|R|` entries per region.
#[must_use]
pub fn kikuchi_free_energy(g: &Graph, rg: &RegionGraph, beta: f64, beliefs: &[Vec<f64>]) -> f64 {
    assert_eq!(rg.n, g.n, "the region graph was built over a different model");
    assert_eq!(beliefs.len(), rg.len(), "one belief table per region");
    let mut f = 0.0;
    for r in 0..rg.len() {
        let vars = &rg.regions[r];
        assert_eq!(beliefs[r].len(), 1usize << vars.len(), "region {r} belief is the wrong size");
        let local = region_log_factor(g, beta, vars);
        let mut energy = 0.0;
        let mut neg_entropy = 0.0;
        for (t, &b) in beliefs[r].iter().enumerate() {
            // local is ln f_R = -beta E_R, so the energy term is minus it.
            energy -= b * local[t];
            if b > 0.0 {
                neg_entropy += b * b.ln();
            }
        }
        f += rg.counting[r] as f64 * (energy + neg_entropy);
    }
    f
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::free_energy::exact_log_z;
    use crate::graph::GraphBuilder;
    use crate::ising;
    use crate::meanfield::belief_propagation;
    use crate::rng::Pcg;

    fn random_tree(n: usize, seed: u64) -> Graph {
        let mut rng = Pcg::new(seed, 0);
        let mut gb = GraphBuilder::new(n);
        for i in 1..n {
            let parent = ((rng.f64() * i as f64) as usize).min(i - 1);
            gb.couple(parent, i, 2.0 * rng.f64() - 1.0);
        }
        for i in 0..n {
            gb.bias(i, rng.f64() - 0.5);
        }
        gb.build()
    }

    /// A four-cycle with fields, the smallest model where Bethe and Kikuchi part company.
    fn plaquette(j: f64) -> Graph {
        let mut gb = GraphBuilder::new(4);
        gb.couple(0, 1, j);
        gb.couple(1, 3, j);
        gb.couple(3, 2, j);
        gb.couple(2, 0, j);
        gb.bias(0, 0.2);
        gb.bias(1, -0.35);
        gb.bias(2, 0.1);
        gb.bias(3, 0.4);
        gb.build()
    }

    /// ORACLE: `crate::meanfield::belief_propagation`, which has its own oracle against
    /// `exact::Elimination` on trees.
    ///
    /// With the single couplings as outer regions the region graph IS the Bethe approximation --
    /// the derived counting number on a shared endpoint comes out at `1 - d_i` -- so the two must
    /// agree to solver precision, on a tree and on a loopy graph alike. This is the test that says
    /// the Moebius recursion, the `E(R)` bookkeeping and the parent-to-child rule all specialise
    /// correctly; an error in any of them moves this number.
    #[test]
    fn bethe_region_graph_matches_meanfield_belief_propagation() {
        let cases: [(&str, Graph, f64); 4] = [
            ("tree 20", random_tree(20, 3), 0.9),
            ("tree 12", random_tree(12, 11), 1.4),
            ("4-cycle", plaquette(1.0), 0.5),
            ("3x3 grid", ising::grid2d(3, 3, 0.8), 0.35),
        ];
        for (name, g, beta) in cases {
            let rg = RegionGraph::bethe(&g);
            // The derived counting numbers ARE the Bethe coefficients, not a coincidence to trust.
            for i in 0..g.n {
                let d = (g.offset[i + 1] - g.offset[i]) as i64;
                let singleton = rg.regions.iter().position(|r| r.as_slice() == [i]);
                let c = singleton.map_or(0, |r| rg.counting[r]);
                assert_eq!(c, if d >= 2 { 1 - d } else { 0 }, "{name}: singleton {i}");
            }
            let bp = belief_propagation(&g, beta, 4000, 0.4);
            let gbp = generalized_belief_propagation(&g, &rg, beta, 4000, 0.4);
            assert!(bp.converged(1e-12) && gbp.converged(1e-12), "{name}: not converged");
            assert!(
                (gbp.log_z - bp.log_z).abs() < 1e-9,
                "{name}: Kikuchi {} vs Bethe {}",
                gbp.log_z,
                bp.log_z
            );
            for i in 0..g.n {
                assert!(
                    (gbp.m[i] - bp.m[i]).abs() < 1e-9,
                    "{name} site {i}: GBP {} vs BP {}",
                    gbp.m[i],
                    bp.m[i]
                );
            }
        }
    }

    /// ORACLE: exhaustive enumeration of all `2^4` states via `free_energy::exact_log_z`, AND the
    /// demand that Bethe measurably fails the same check.
    ///
    /// THE SECOND HALF IS THE POINT. On a single four-cycle, taking the plaquette as one outer
    /// region makes Kikuchi exact -- the region graph is one node with `c = 1`, no messages, and a
    /// belief that is the Boltzmann distribution itself. Asserting only that GBP is close to exact
    /// would pass for an implementation that silently fell back to Bethe, so the gap Bethe leaves
    /// is asserted from below at every temperature tried.
    #[test]
    fn plaquette_kikuchi_is_exact_where_bethe_is_not_by_enumeration() {
        for &beta in &[0.25f64, 0.5, 0.9] {
            let g = plaquette(1.0);
            let truth = exact_log_z(&g, beta);
            let rg = RegionGraph::new(4, &[vec![0, 1, 2, 3]]).unwrap();
            assert_eq!(rg.len(), 1, "one outer region generates no intersections");
            assert_eq!(rg.counting, vec![1], "and its counting number is one");
            let gbp = generalized_belief_propagation(&g, &rg, beta, 200, 0.0);
            assert!(
                (gbp.log_z - truth).abs() < 1e-13,
                "beta {beta}: Kikuchi {} vs exact {truth}",
                gbp.log_z
            );
            // Bethe on the same model, at the same beta, is wrong -- and by a wide margin.
            let bethe = belief_propagation(&g, beta, 5000, 0.5);
            assert!(bethe.converged(1e-12), "beta {beta}: BP did not converge");
            let bethe_err = (bethe.log_z - truth).abs();
            assert!(
                bethe_err > 1e-3,
                "beta {beta}: Bethe error {bethe_err} is too small for this test to mean anything"
            );
        }
    }

    /// ORACLE: exhaustive enumeration on a 3x3 grid, where the plaquette CVM is a real
    /// approximation rather than a tautology.
    ///
    /// Four overlapping faces, nine regions after closure, counting numbers `+1, -1, +1`. The
    /// single-region plaquette test above cannot distinguish "built a region graph" from
    /// "enumerated the model", and this one can: here Kikuchi passes messages, is NOT exact, and
    /// must still beat Bethe by a wide margin.
    #[test]
    fn plaquette_cvm_beats_bethe_on_a_square_grid_by_enumeration() {
        // Measured, |ln Z error| against enumeration:
        //
        // | grid | beta | Kikuchi | Bethe   | ratio |
        // |------|------|---------|---------|-------|
        // | 3x3  | 0.35 | 2.15e-5 | 2.38e-2 | 1104  |
        // | 3x3  | 0.50 | 1.93e-4 | 9.39e-2 |  487  |
        // | 4x4  | 0.35 | 1.22e-4 | 5.51e-2 |  450  |
        // | 4x4  | 0.50 | 1.41e-3 | 2.25e-1 |  159  |
        //
        // The floor asserted below is 100x, an order under the worst of these, so this fails on a
        // broken region graph rather than on the third digit of a fixed point.
        for (side, beta, regions) in [(3usize, 0.35f64, 9usize), (3, 0.5, 9), (4, 0.35, 25)] {
            let g = ising::grid2d(side, side, 0.8);
            let truth = exact_log_z(&g, beta);
            let rg = RegionGraph::new(g.n, &grid_plaquettes(side, side)).unwrap();
            assert_eq!(rg.len(), regions, "{side}x{side}: faces + shared bonds + shared corners");
            assert!(rg.counts_each_variable_once() && rg.counts_each_coupling_once(&g));
            let gbp = generalized_belief_propagation(&g, &rg, beta, 20_000, 0.5);
            assert!(gbp.converged(1e-12), "{side} b{beta}: residual {}", gbp.residual);
            assert!(gbp.consistency < 1e-10, "{side} b{beta}: polytope {}", gbp.consistency);
            let bethe = belief_propagation(&g, beta, 20_000, 0.5);
            assert!(bethe.converged(1e-12), "{side} b{beta}: BP residual {}", bethe.residual);
            let (ke, be) = ((gbp.log_z - truth).abs(), (bethe.log_z - truth).abs());
            assert!(ke > 1e-9, "{side} b{beta}: Kikuchi must be an approximation here, not an \
                                enumeration, or this test compares nothing: error {ke}");
            assert!(ke < be / 100.0, "{side} b{beta}: Kikuchi error {ke} vs Bethe {be}");
        }
    }

    /// ORACLE: `Graph::energy` on the induced sub-model, which pins the crate's `E = -J s s - h s`
    /// convention rather than restating it.
    ///
    /// `region_log_factor` is `ln f_R`, and every belief, every message and the free energy itself
    /// are built on it. A sign here is a sign in every number this module reports, so it is checked
    /// against a graph built independently by `GraphBuilder` from the same couplings.
    #[test]
    fn region_log_factors_are_minus_beta_times_the_graphs_own_energy() {
        let g = ising::grid2d(3, 3, 0.8);
        let beta = 0.7;
        for vars in [vec![0usize, 1, 3, 4], vec![1, 4], vec![4], vec![0, 4, 8]] {
            // The same interactions, as a standalone model the graph module scores itself.
            let mut gb = GraphBuilder::new(vars.len());
            for (a, &i) in vars.iter().enumerate() {
                gb.bias(a, g.h[i]);
                for (b, &j) in vars.iter().enumerate() {
                    if b > a && let Some(w) = coupling(&g, i, j) {
                        gb.couple(a, b, w);
                    }
                }
            }
            let sub = gb.build();
            let got = region_log_factor(&g, beta, &vars);
            for t in 0..(1usize << vars.len()) {
                let s: Vec<i8> =
                    (0..vars.len()).map(|b| if (t >> b) & 1 == 1 { 1 } else { -1 }).collect();
                let want = -beta * sub.energy(&s);
                assert!((got[t] - want).abs() < 1e-12, "vars {vars:?} cfg {t}: {} vs {want}", got[t]);
            }
            // And the parent/child split reconstructs the parent exactly.
            if vars.len() == 4 {
                let child = vec![vars[0], vars[1]];
                let extra = split_log_factor(&g, beta, &vars, &child);
                let cl = region_log_factor(&g, beta, &child);
                let pos = positions(&vars, &child);
                for t in 0..16 {
                    let want = cl[project(t, &pos)] + extra[t];
                    assert!((got[t] - want).abs() < 1e-12, "split at cfg {t}");
                }
            }
        }
    }

    /// THE INVARIANT, in integers, on every region graph this module can build.
    ///
    /// `Σ_{R ∋ v} c_R = 1` for every variable and `Σ_{R ⊇ {i,j}} c_R = 1` for every coupling is
    /// what makes the Kikuchi free energy an approximation to THIS model rather than to a reweighted
    /// one. The counting numbers are integers by construction, so this is asserted exactly -- a
    /// tolerance here would be a tolerance on an identity that has none.
    #[test]
    fn counting_numbers_count_every_variable_and_coupling_exactly_once() {
        let cases: [(&str, Graph, Vec<Vec<usize>>); 5] = [
            ("bethe tree", random_tree(14, 5), Vec::new()),
            ("bethe 4x4 torus", ising::lattice2d(4, 1.0), Vec::new()),
            ("bethe ring", ising::ring(9, 1.0, 0.3), Vec::new()),
            ("plaquettes 4x4", ising::grid2d(4, 4, 1.0), grid_plaquettes(4, 4)),
            ("plaquettes 3x5", ising::grid2d(3, 5, 1.0), grid_plaquettes(3, 5)),
        ];
        for (name, g, outer) in cases {
            let rg = if outer.is_empty() {
                RegionGraph::bethe(&g)
            } else {
                RegionGraph::new(g.n, &outer).unwrap()
            };
            for v in 0..g.n {
                assert_eq!(rg.variable_counting(v), 1, "{name}: variable {v}");
            }
            for i in 0..g.n {
                for e in g.offset[i]..g.offset[i + 1] {
                    let j = g.nbr[e] as usize;
                    if j > i {
                        assert_eq!(rg.pair_counting(i, j), 1, "{name}: coupling ({i},{j})");
                    }
                }
            }
            // Moebius is a definition, so check it holds term by term as well as in aggregate.
            for r in 0..rg.len() {
                let want = 1 - rg.ancestors[r].iter().map(|&a| rg.counting[a]).sum::<i64>();
                assert_eq!(rg.counting[r], want, "{name}: region {r}");
            }
            // Every Hasse edge is a cover: containment with nothing strictly between.
            for &(p, c) in &rg.edges {
                assert!(contains(&rg.regions[p], &rg.regions[c]));
                assert!(rg.regions[p].len() > rg.regions[c].len());
                assert!(
                    !rg.regions.iter().any(|m| {
                        m.len() > rg.regions[c].len()
                            && m.len() < rg.regions[p].len()
                            && contains(&rg.regions[p], m)
                            && contains(m, &rg.regions[c])
                    }),
                    "{name}: ({p},{c}) is not a cover"
                );
            }
        }
    }

    /// The 3x3 plaquette region graph is the one Kikuchi's paper draws, region for region.
    ///
    /// Written out rather than derived, so a change to the closure or the ordering has to be
    /// defended: four faces at `+1`, four shared bonds at `-1`, the shared corner at `+1`.
    #[test]
    fn the_square_lattice_region_graph_is_the_published_one() {
        let rg = RegionGraph::new(9, &grid_plaquettes(3, 3)).unwrap();
        let faces: Vec<&Vec<usize>> = rg.regions.iter().filter(|r| r.len() == 4).collect();
        assert_eq!(faces.len(), 4);
        assert_eq!(
            rg.regions.iter().filter(|r| r.len() == 2).count(),
            4,
            "each neighbouring pair of faces shares a bond"
        );
        assert_eq!(rg.regions.iter().filter(|r| r.len() == 1).collect::<Vec<_>>(), vec![&vec![4]]);
        for (r, c) in rg.regions.iter().zip(&rg.counting) {
            let want = match r.len() {
                4 | 1 => 1,
                _ => -1,
            };
            assert_eq!(*c, want, "region {r:?}");
        }
        // Sixteen covers: each face covers its two bonds, each bond covers the centre.
        assert_eq!(rg.edges.len(), 12, "8 face-to-bond plus 4 bond-to-centre");
    }

    /// One region holding every variable makes the "approximation" the model itself.
    ///
    /// ORACLE: enumeration. The belief is the Boltzmann distribution and `⟨ln f⟩ + H` telescopes to
    /// `ln Z`, so this must be exact to the last few bits, not close.
    #[test]
    fn a_single_region_over_everything_is_exact_by_enumeration() {
        for n in 2..=8usize {
            let g = random_tree(n, n as u64 + 40);
            let beta = 1.1;
            let rg = RegionGraph::new(n, &[(0..n).collect()]).unwrap();
            let gbp = generalized_belief_propagation(&g, &rg, beta, 10, 0.0);
            let truth = exact_log_z(&g, beta);
            assert!((gbp.log_z - truth).abs() < 1e-13, "n {n}: {} vs {truth}", gbp.log_z);
            assert_eq!(gbp.consistency, 0.0, "no edges, so nothing to be inconsistent about");
        }
    }

    /// The free energy is a functional of the beliefs, so it must move when they do.
    ///
    /// A `kikuchi_free_energy` that ignored its argument and recomputed from the graph would pass
    /// every accuracy test above; this is what says it does not.
    #[test]
    fn the_free_energy_reads_the_beliefs_it_is_given() {
        let g = plaquette(1.0);
        let rg = RegionGraph::bethe(&g);
        let beta = 0.5;
        let gbp = generalized_belief_propagation(&g, &rg, beta, 5000, 0.4);
        let at_fixed = kikuchi_free_energy(&g, &rg, beta, &gbp.beliefs);
        assert!((at_fixed - gbp.free_energy).abs() < 1e-14);
        let uniform: Vec<Vec<f64>> =
            rg.regions.iter().map(|r| vec![1.0 / (1usize << r.len()) as f64; 1 << r.len()]).collect();
        let at_uniform = kikuchi_free_energy(&g, &rg, beta, &uniform);
        assert!(
            (at_uniform - at_fixed).abs() > 1e-3,
            "uniform beliefs gave {at_uniform}, fixed point {at_fixed}"
        );
        // At beta = 0 the uniform beliefs ARE the fixed point and -F is exactly n ln 2.
        let zero = kikuchi_free_energy(&g, &rg, 0.0, &uniform);
        assert!((-zero - 4.0 * std::f64::consts::LN_2).abs() < 1e-14, "beta 0: {}", -zero);
    }

    /// Every way of handing in a set of regions that is not one.
    #[test]
    fn the_constructor_refuses_what_it_cannot_count() {
        assert_eq!(RegionGraph::new(3, &[]).unwrap_err(), RegionError::NoOuterRegions { n: 3 });
        assert_eq!(
            RegionGraph::new(3, &[vec![0, 1], vec![]]).unwrap_err(),
            RegionError::EmptyRegion { index: 1 }
        );
        assert_eq!(
            RegionGraph::new(3, &[vec![0, 1], vec![1, 7]]).unwrap_err(),
            RegionError::OutOfRange { index: 1, var: 7, n: 3 }
        );
        assert_eq!(
            RegionGraph::new(40, &[(0..40).collect()]).unwrap_err(),
            RegionError::TooLarge { index: 0, vars: 40, max: MAX_REGION }
        );
        assert_eq!(
            RegionGraph::new(4, &[vec![0, 1], vec![1, 2]]).unwrap_err(),
            RegionError::Uncovered { var: 3 }
        );
        // A model with no variables is not an error; it is a model with ln Z = 0.
        let empty = RegionGraph::new(0, &[]).unwrap();
        assert!(empty.is_empty() && empty.counts_each_variable_once());
        // An outer region contained in another is dropped, not given a second `c = 1`. Duplicates
        // go the same way, which is what makes the construction a function of the SET of regions.
        let rg = RegionGraph::new(3, &[vec![0, 1, 2], vec![0, 1], vec![0, 1, 2]]).unwrap();
        assert_eq!(rg.regions, vec![vec![0, 1, 2]]);
        assert_eq!(rg.counting, vec![1]);
        assert!(rg.counts_each_variable_once());
        // Two genuine outer regions keep both, and their intersection is generated at `c = -1`.
        let rg = RegionGraph::new(4, &[vec![0, 1, 2], vec![1, 2, 3]]).unwrap();
        assert_eq!(rg.regions, vec![vec![0, 1, 2], vec![1, 2, 3], vec![1, 2]]);
        assert_eq!(rg.counting, vec![1, 1, -1]);
        assert!(rg.counts_each_variable_once());
        // Every error prints something a caller can act on.
        for e in [
            RegionError::NoOuterRegions { n: 3 },
            RegionError::EmptyRegion { index: 1 },
            RegionError::OutOfRange { index: 1, var: 7, n: 3 },
            RegionError::TooLarge { index: 0, vars: 40, max: MAX_REGION },
            RegionError::Uncovered { var: 3 },
        ] {
            assert!(e.to_string().len() > 20, "{e:?}");
        }
    }

    /// An isolated spin has no coupling to put it in a region, and must still be counted once.
    #[test]
    fn an_isolated_spin_still_gets_a_region() {
        let mut gb = GraphBuilder::new(4);
        gb.couple(0, 1, 0.7);
        gb.couple(1, 2, -0.4);
        gb.bias(3, 0.9);
        let g = gb.build();
        let rg = RegionGraph::bethe(&g);
        assert!(rg.counts_each_variable_once());
        let beta = 0.8;
        let gbp = generalized_belief_propagation(&g, &rg, beta, 2000, 0.0);
        // A tree plus a free spin: Bethe is exact, so this is exact.
        assert!((gbp.log_z - exact_log_z(&g, beta)).abs() < 1e-12, "{}", gbp.log_z);
        assert!((gbp.m[3] - (beta * 0.9).tanh()).abs() < 1e-12, "the free spin: {}", gbp.m[3]);
    }
}

