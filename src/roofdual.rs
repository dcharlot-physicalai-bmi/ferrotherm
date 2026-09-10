//! Roof duality (QPBO): a max-flow lower bound, and the spins it pins in **every** ground state.
//!
//! [`crate::bound`] names roof duality as the relaxation it does not implement. This is it. Where
//! [`crate::bound::forest`] splits the energy into tractable parts and tightens the split,
//! roof duality relaxes each spin into a PAIR of binary variables — one standing for `s_i = +1`,
//! one for `s_i = -1` — that are no longer required to disagree. The relaxed energy is submodular
//! whatever the couplings, so one min-cut minimises it exactly, and its minimum is a lower bound
//! because the states of the original problem are exactly the relaxation's consistent points.
//!
//! # The construction
//!
//! Write the energy in binary variables `x_i = (s_i + 1)/2`:
//!
//! ```text
//!   E(s) = c + sum_i a_i x_i + sum_{i<j} b_ij x_i x_j,
//!   a_i = -2 h_i + 2 sum_j w_ij,   b_ij = -4 w_ij,   c = sum_i h_i - sum_{i<j} w_ij
//! ```
//!
//! Now give spin `i` two independent binary variables: `y_i` for `x_i`, and `z_i` for `1 - x_i`.
//! Every term is written twice, once in each set, and halved:
//!
//! ```text
//!   a_i x_i          ->  (a_i/2) (y_i + 1 - z_i)
//!   b_ij x_i x_j     ->  (b_ij/2) [ y_i y_j + (1-z_i)(1-z_j) ]        for b_ij <= 0
//!                    ->  (b_ij/2) [ y_i (1-z_j) + (1-z_i) y_j ]       for b_ij >  0
//! ```
//!
//! Substituting `z = 1 - y` collapses both branches back to `b_ij y_i y_j`, so the relaxed energy
//! `G(y, z)` equals `E` at every consistent point — checked exhaustively by
//! `the_relaxation_is_the_energy_on_consistent_labellings`. And every quadratic coefficient above
//! is negative by construction, which is what makes `G` submodular and a min-cut its exact
//! minimiser. The supermodular case is the frustrated one: `b_ij > 0` is `w_ij < 0`.
//!
//! # Persistency, which is the part worth having
//!
//! A bound alone says how far a state might be from optimal. Roof duality also says, for some
//! spins, exactly what an optimal state does:
//!
//!  * [`RoofDual::fixed`] — spins whose relaxed pair is consistent in **every** minimiser of `G`.
//!    Those spins take that value in **every** ground state (strong persistency, Hammer, Hansen &
//!    Simeone 1984). Read off the residual graph, where a node is source-side in every min-cut iff
//!    it is reachable from the source and sink-side in every min-cut iff the sink is reachable from
//!    it.
//!  * [`RoofDual::labels`] — the consistent spins of ONE minimiser, walked towards consistency
//!    rather than read off the first min-cut found. SOME ground state agrees with all of them at
//!    once (weak persistency), which is what [`RoofDual::simplify`] needs and all it needs.
//!
//! The second is the larger set and the weaker claim, and the gap between them is real rather than
//! bookkeeping: on a zero-field ferromagnet `labels` names an entire ground state while `fixed`
//! names nothing, because the flipped state is a ground state too. On an unbiased frustrated
//! triangle both are empty, which is also correct — its two optima are a global flip apart.
//!
//! # What it costs and where it is weak
//!
//! One max-flow (Dinic) on `2n + 2` nodes and `2m + 2n` arcs. Exact where the model is submodular
//! — a ferromagnet with arbitrary fields is solved outright, every spin pinned — and no better than
//! the trivial floor on unbiased max-cut, which is the same wall [`crate::bound`] documents for
//! forests. Both bounds are sound, so take the larger.
//!
//! ```
//! use ferrotherm::{graph::GraphBuilder, roofdual};
//!
//! let mut b = GraphBuilder::new(4);
//! for i in 0..3 {
//!     b.couple(i, i + 1, 1.0); // ferromagnetic: submodular, so the relaxation is tight
//! }
//! b.bias(0, 0.5); // breaks the flip symmetry, so the optimum is unique
//! let g = b.build();
//!
//! let r = roofdual::roof_dual(&g);
//! assert_eq!(r.n_fixed(), 4, "a submodular instance is pinned everywhere");
//! let s = r.ground_state().expect("complete labelling");
//! assert!((g.energy(&s) - r.bound).abs() < 1e-9, "and the bound is attained");
//! ```

use crate::bound::Bound;
use crate::graph::{Graph, GraphBuilder};
use crate::round::{accumulation_guard, sum_down};
use std::collections::VecDeque;

/// Residual capacity treated as saturated, relative to the largest arc capacity.
///
/// Raising it can only REMOVE spins from [`RoofDual::fixed`]: reachability over fewer residual arcs
/// pins fewer nodes, so the slack is spent in the safe direction.
const TOL_REL: f64 = 1e-12;

/// The roof-dual bound and the spins it pins.
#[derive(Clone, Debug)]
pub struct RoofDual {
    /// Lower bound on `min_s E(s)`, rounded DOWN through [`crate::round`] so it is below the exact
    /// relaxation rather than near it. `f64::NEG_INFINITY` if the graph carries a non-finite weight.
    pub bound: f64,
    /// Per spin, the value it takes in **every** ground state, or `None` if roof duality cannot say.
    pub fixed: Vec<Option<i8>>,
    /// Per spin, the value from ONE minimiser of the relaxation. SOME ground state agrees with all
    /// of these at once — jointly, not one at a time — and a spin in [`RoofDual::fixed`] carries the
    /// same value here. Always at least as large a set as [`RoofDual::fixed`].
    pub labels: Vec<Option<i8>>,
    /// Max-flow value on the doubled network — the relaxation's cost above its constant part.
    pub flow: f64,
    /// Augmenting paths the flow took. Reported because a bound assembled from many augmentations
    /// carries more accumulated rounding than one from few, and the guard subtracted from
    /// [`RoofDual::bound`] is sized on that arithmetic rather than on this count alone.
    pub augmentations: usize,
}

impl RoofDual {
    /// How many spins are pinned in every ground state.
    #[must_use]
    pub fn n_fixed(&self) -> usize {
        self.fixed.iter().filter(|f| f.is_some()).count()
    }

    /// How many spins the single-minimiser labelling assigns.
    #[must_use]
    pub fn n_labelled(&self) -> usize {
        self.labels.iter().filter(|f| f.is_some()).count()
    }

    /// The ground state, when [`RoofDual::labels`] covers every spin — then the relaxation is tight
    /// and this state attains [`RoofDual::bound`]. `None` when any spin is unlabelled.
    #[must_use]
    pub fn ground_state(&self) -> Option<Vec<i8>> {
        self.labels.iter().copied().collect()
    }

    /// The same bound as a [`Bound`], for [`Bound::gap`] and [`Bound::proves_optimal`].
    ///
    /// `parts` is 1: this is one relaxation solved by one min-cut, not a decomposition into parts
    /// minimised separately, and reporting 2 for the doubled variable set would say otherwise.
    #[must_use]
    pub fn as_bound(&self) -> Bound {
        Bound {
            value: self.bound,
            parts: 1,
            method: "roof duality: min-cut on the doubled network",
            rounds: 0,
            best_round: 0,
        }
    }

    /// Eliminate the labelled spins, returning the model over what is left.
    ///
    /// Uses [`RoofDual::labels`] rather than [`RoofDual::fixed`], so it pins as much as weak
    /// persistency allows: `offset + min E_reduced == min E` exactly, though ground states that
    /// disagree with the labelling are dropped. Pin [`RoofDual::fixed`] yourself if every ground
    /// state must survive.
    ///
    /// # Panics
    ///
    /// If `g` has a different spin count from the run that produced this.
    #[must_use]
    pub fn simplify(&self, g: &Graph) -> Reduced {
        assert_eq!(g.n, self.labels.len(), "graph has {} spins, labelling has {}", g.n, self.labels.len());
        let pinned = self.labels.clone();
        // New index for each surviving spin; usize::MAX marks a pinned one, and is never used as an
        // index because every read is guarded by `pinned[i].is_none()`.
        let mut idx = vec![usize::MAX; g.n];
        let mut free = Vec::new();
        for i in 0..g.n {
            if pinned[i].is_none() {
                idx[i] = free.len();
                free.push(i);
            }
        }
        let mut gb = GraphBuilder::new(free.len());
        let mut offset = 0.0;
        for i in 0..g.n {
            match pinned[i] {
                // A pinned spin's field term is a constant; a free spin keeps its own.
                Some(v) => offset -= g.h[i] * f64::from(v),
                None => gb.bias(idx[i], g.h[i]),
            }
        }
        for i in 0..g.n {
            for k in g.offset[i]..g.offset[i + 1] {
                let j = g.nbr[k] as usize;
                if j <= i {
                    continue;
                }
                match (pinned[i], pinned[j]) {
                    (Some(a), Some(b)) => offset -= g.w[k] * f64::from(a) * f64::from(b),
                    // One end pinned: the coupling becomes a field on the free end.
                    (Some(a), None) => gb.bias(idx[j], g.w[k] * f64::from(a)),
                    (None, Some(b)) => gb.bias(idx[i], g.w[k] * f64::from(b)),
                    (None, None) => gb.couple(idx[i], idx[j], g.w[k]),
                }
            }
        }
        Reduced { graph: gb.build(), free, offset, pinned }
    }
}

/// The model left after the pinned spins are eliminated.
///
/// No `Debug`: [`Graph`] has none, and deriving one here would print a CSR array dump anyway.
pub struct Reduced {
    /// The model over the unpinned spins, renumbered `0..free.len()` in increasing original order.
    pub graph: Graph,
    /// Original index of each spin of [`Reduced::graph`].
    pub free: Vec<usize>,
    /// Energy of the pinned part, couplings between two pinned spins included. Add it to any energy
    /// of [`Reduced::graph`] to get the original energy.
    pub offset: f64,
    /// The labelling that was pinned, indexed by ORIGINAL spin.
    pub pinned: Vec<Option<i8>>,
}

impl Reduced {
    /// Put a state of [`Reduced::graph`] back on the original spins.
    ///
    /// # Panics
    ///
    /// If `sub` is not one spin per node of [`Reduced::graph`].
    #[must_use]
    pub fn lift(&self, sub: &[i8]) -> Vec<i8> {
        assert_eq!(sub.len(), self.free.len(), "expected {} free spins, got {}", self.free.len(), sub.len());
        let mut s = vec![0i8; self.pinned.len()];
        for i in 0..self.pinned.len() {
            if let Some(v) = self.pinned[i] {
                s[i] = v;
            }
        }
        for (k, &i) in self.free.iter().enumerate() {
            s[i] = sub[k];
        }
        s
    }
}

/// Roof-dual bound and persistency, at the default residual tolerance.
#[must_use]
pub fn roof_dual(g: &Graph) -> RoofDual {
    roof_dual_with(g, None)
}

/// Roof-dual bound and persistency, with the residual tolerance chosen.
///
/// A residual capacity at or below `tol` counts as saturated; `None` takes `TOL_REL` times the
/// largest arc capacity. Larger values pin FEWER spins, never more, so this is slack in the safe
/// direction. A negative or non-finite `tol` is refused and the default used, since a tolerance
/// that is not a number would make every residual arc a matter of chance.
#[must_use]
pub fn roof_dual_with(g: &Graph, tol: Option<f64>) -> RoofDual {
    let n = g.n;
    if !finite(g) {
        // A non-finite weight makes every capacity meaningless. `-inf` is the only bound that is
        // still true, and pinning nothing is the only persistency claim that is still true.
        return RoofDual {
            bound: f64::NEG_INFINITY,
            fixed: vec![None; n],
            labels: vec![None; n],
            flow: 0.0,
            augmentations: 0,
        };
    }
    let (src, snk) = (2 * n, 2 * n + 1);
    let mut net = Net::new(2 * n + 2);
    // Constants collected term by term rather than accumulated, so `sum_down` can bracket them.
    let mut consts: Vec<f64> = Vec::with_capacity(3 * n + 2 * g.n_edges);
    // Cost charged to a node when it takes label 1 (sink side). Label 0 is never charged directly:
    // every term below is written so its zero is at label 0, and the per-node normalisation moves
    // any negative charge into `consts` and an arc to the sink.
    let mut chg = vec![0.0f64; 2 * n];
    let (mut habs, mut wabs) = (0.0f64, 0.0f64);

    for i in 0..n {
        consts.push(g.h[i]); // c = sum_i h_i - sum_{i<j} w_ij
        habs += g.h[i].abs();
        // a_i = -2 h_i + 2 sum_j w_ij, halved on the spot: every use of a_i below is a_i/2.
        let mut half_a = -g.h[i];
        for k in g.offset[i]..g.offset[i + 1] {
            half_a += g.w[k];
        }
        consts.push(half_a);
        chg[i] += half_a; // (a_i/2) y_i
        chg[n + i] -= half_a; // -(a_i/2) z_i
    }

    for i in 0..n {
        for k in g.offset[i]..g.offset[i + 1] {
            let j = g.nbr[k] as usize;
            if j <= i {
                continue;
            }
            let w = g.w[k];
            wabs += w.abs();
            consts.push(-w); // the -sum_{i<j} w_ij half of c
            let half_b = -2.0 * w; // b_ij / 2, with b_ij = -4 w_ij
            if half_b <= 0.0 {
                // (b/2) y_i y_j  ->  unary half_b on y_j, arc y_i -> y_j of capacity -half_b
                chg[j] += half_b;
                net.arc(i, j, -half_b);
                // (b/2)(1-z_i)(1-z_j)  ->  constant half_b, unary -half_b on z_i, arc z_i -> z_j
                consts.push(half_b);
                chg[n + i] -= half_b;
                net.arc(n + i, n + j, -half_b);
            } else {
                // (b/2) y_i (1-z_j)  ->  unary half_b on y_i, -half_b on z_j, arc y_i -> z_j
                chg[i] += half_b;
                chg[n + j] -= half_b;
                net.arc(i, n + j, half_b);
                // (b/2) (1-z_i) y_j  ->  arc z_i -> y_j, no unary part
                net.arc(n + i, j, half_b);
            }
        }
    }

    for p in 0..2 * n {
        if chg[p] > 0.0 {
            net.arc(src, p, chg[p]); // cut exactly when p is on the sink side, i.e. label 1
        } else if chg[p] < 0.0 {
            // Charging label 1 by a negative amount is charging label 0 by its magnitude, once the
            // constant is banked. Capacities must be non-negative and this is the only place they
            // would not be.
            consts.push(chg[p]);
            net.arc(p, snk, -chg[p]);
        }
    }

    let tol = tol.filter(|t| t.is_finite() && *t >= 0.0).unwrap_or_else(|| TOL_REL * net.cmax());
    let (flow, augmentations) = net.maxflow(src, snk, tol);

    // Every capacity, constant and partial sum in the whole construction is under this. Overstating
    // it only widens the guard, which lowers the bound, which stays sound.
    let mag = 4.0 * (habs + wabs) + 1.0;
    let ops = net.ops + 4 * n + 4 * g.n_edges;
    consts.push(flow);
    let bound = sum_down(&consts) - accumulation_guard(ops, mag);

    let from_src = net.reach_forward(src, tol);
    let to_snk = net.reach_backward(snk, tol);
    let mut fixed = vec![None; n];
    for i in 0..n {
        // In EVERY min-cut: label(y_i) = 1 iff the sink is reachable from y_i, label(z_i) = 0 iff
        // y_i's partner is reachable from the source.
        //
        // THE CONJUNCTION IS REDUNDANT AND IS KEPT ANYWAY. `G` is invariant under the involution
        // (y, z) -> (1 - z, 1 - y), so its minimisers come in pairs and "y_i = 1 in all of them" is
        // the mirror image of "z_i = 0 in all of them" -- the same statement twice. Measured: the
        // two halves never once disagreed across forty thousand spins, and a mutation dropping the
        // second half survives every test in this file. It stays because the equality is a theorem
        // about EXACT arithmetic and these are floats, and an `&&` of two mirror tests is the cheap
        // side to be wrong on.
        if to_snk[i] && from_src[n + i] {
            fixed[i] = Some(1);
        } else if from_src[i] && to_snk[n + i] {
            fixed[i] = Some(-1);
        }
    }
    let labels = net.walk_to_a_consistent_cut(&from_src, n, snk, tol);
    RoofDual { bound, fixed, labels, flow, augmentations }
}

fn finite(g: &Graph) -> bool {
    g.h.iter().all(|x| x.is_finite()) && g.w.iter().all(|x| x.is_finite())
}

/// Residual network with paired arcs: `e` and `e ^ 1` are the two directions of one edge.
struct Net {
    to: Vec<u32>,
    cap: Vec<f64>,
    adj: Vec<Vec<u32>>,
    /// Capacity writes performed, for the bound's rounding guard.
    ops: usize,
}

impl Net {
    fn new(nodes: usize) -> Self {
        Net { to: Vec::new(), cap: Vec::new(), adj: vec![Vec::new(); nodes], ops: 0 }
    }

    fn arc(&mut self, u: usize, v: usize, c: f64) {
        // Every capacity here comes from a submodular pair term, so it is non-negative by
        // construction; a negative one would mean the decomposition above is wrong, not that the
        // instance is hard. Non-finite weights are refused before any of this runs.
        debug_assert!(c >= 0.0, "negative capacity {c} on arc {u} -> {v}");
        if c <= 0.0 {
            return; // a zero-capacity arc is not an arc
        }
        let e = self.to.len() as u32;
        self.to.push(v as u32);
        self.cap.push(c);
        self.to.push(u as u32);
        self.cap.push(0.0);
        self.adj[u].push(e);
        self.adj[v].push(e + 1);
    }

    fn cmax(&self) -> f64 {
        self.cap.iter().copied().fold(0.0, f64::max)
    }

    /// Dinic: level graph by breadth-first search, then a blocking flow on it, repeated.
    ///
    /// Phase count is bounded by the node count whatever the capacities are -- each phase strictly
    /// lengthens the shortest augmenting path -- which is why this terminates on `f64` capacities
    /// where plain augmenting-path search need not.
    fn maxflow(&mut self, s: usize, t: usize, tol: f64) -> (f64, usize) {
        let nodes = self.adj.len();
        let mut flow = 0.0;
        let mut augmentations = 0usize;
        let mut level = vec![-1i32; nodes];
        let mut it = vec![0usize; nodes];
        let mut queue = VecDeque::new();
        loop {
            level.iter_mut().for_each(|l| *l = -1);
            level[s] = 0;
            queue.clear();
            queue.push_back(s);
            while let Some(u) = queue.pop_front() {
                for k in 0..self.adj[u].len() {
                    let e = self.adj[u][k] as usize;
                    let v = self.to[e] as usize;
                    if level[v] < 0 && self.cap[e] > tol {
                        level[v] = level[u] + 1;
                        queue.push_back(v);
                    }
                }
            }
            if level[t] < 0 {
                break;
            }
            it.iter_mut().for_each(|x| *x = 0);
            let (f, a) = self.blocking(s, t, &level, &mut it, tol);
            flow += f;
            augmentations += a;
            if a == 0 {
                break; // no progress on a level graph that claimed to have a path: stop rather than spin
            }
        }
        (flow, augmentations)
    }

    /// One blocking flow, iteratively — the path is an explicit stack, so depth is not the call
    /// stack's problem on a graph with a hundred thousand nodes.
    fn blocking(&mut self, s: usize, t: usize, level: &[i32], it: &mut [usize], tol: f64) -> (f64, usize) {
        let mut total = 0.0;
        let mut augmentations = 0usize;
        let mut path: Vec<usize> = Vec::new();
        let mut u = s;
        loop {
            if u == t {
                // Bottleneck, and WHERE it is: the first arc that saturates is where the walk
                // resumes, since everything before it still has capacity.
                let mut f = f64::INFINITY;
                let mut cut = 0usize;
                for (k, &e) in path.iter().enumerate() {
                    if self.cap[e] < f {
                        f = self.cap[e];
                        cut = k;
                    }
                }
                for &e in &path {
                    // `cap - f >= 0` exactly: `f` is the minimum over the path and `f64` subtraction
                    // is monotone, so no capacity can go negative here.
                    self.cap[e] -= f;
                    self.cap[e ^ 1] += f;
                    self.ops += 2;
                }
                total += f;
                augmentations += 1;
                u = self.tail(path[cut]);
                path.truncate(cut);
                continue;
            }
            let mut advanced = false;
            while it[u] < self.adj[u].len() {
                let e = self.adj[u][it[u]] as usize;
                let v = self.to[e] as usize;
                if self.cap[e] > tol && level[v] == level[u] + 1 {
                    path.push(e);
                    u = v;
                    advanced = true;
                    break;
                }
                it[u] += 1;
            }
            if advanced {
                continue;
            }
            if u == s {
                return (total, augmentations);
            }
            // Dead end: retreat, and never look down that arc again this phase.
            let e = path.pop().expect("a non-source node was reached along a path");
            u = self.tail(e);
            it[u] += 1;
        }
    }

    fn tail(&self, e: usize) -> usize {
        self.to[e ^ 1] as usize
    }

    /// A min-cut chosen to make as many spins consistent as the greedy walk manages, and the
    /// labelling it gives.
    ///
    /// A set containing the source, not the sink, and closed under residual arcs IS a min-cut —
    /// that is the same fact reachability is read off, used constructively instead. So start from
    /// the smallest such set and, for each spin whose two nodes sit on the same side, try to pull
    /// the residual closure of one of them across. The closure keeps the set closed, the sink test
    /// keeps it a cut, and a spin whose partner would come with it is left alone. Every step lands
    /// on another minimiser of the relaxation, so the labelling stays a weak-persistency labelling
    /// however greedy the walk was.
    ///
    /// `O(n m)` worst case, and it is why this labels the zero-field ferromagnet — where the
    /// smallest min-cut alone labels nothing at all — rather than shrugging at it.
    fn walk_to_a_consistent_cut(&self, from_src: &[bool], n: usize, snk: usize, tol: f64) -> Vec<Option<i8>> {
        let mut inside = from_src.to_vec();
        for i in 0..n {
            let (u, v) = (i, n + i);
            if !inside[u] && !inside[v] {
                // Pull one of them in, whichever does not drag its partner along with it. Both
                // already inside is terminal: nothing ever leaves the set.
                if !self.close_over(&mut inside, v, &[snk, u], tol) {
                    self.close_over(&mut inside, u, &[snk, v], tol);
                }
            }
        }
        // READ THE LABELLING OFF THE FINAL SET, never off an intermediate one. The set only grows,
        // so a spin made consistent early can be un-made by a later step, and a labelling collected
        // as the walk went would be a mixture of several minimisers -- which extends to nothing in
        // particular, weak persistency being a statement about ONE of them. Collecting it early
        // has never been observed to differ here and survives as a mutation; this is written for
        // the argument, not for a failure that was seen.
        (0..n)
            .map(|i| match (inside[i], inside[n + i]) {
                (false, true) => Some(1),
                (true, false) => Some(-1),
                _ => None,
            })
            .collect()
    }

    /// Add `p` and everything residually reachable from it, unless that reaches a forbidden node —
    /// in which case nothing is added at all. Returns whether the set grew to include `p`.
    fn close_over(&self, inside: &mut [bool], p: usize, forbid: &[usize], tol: f64) -> bool {
        if inside[p] {
            return true;
        }
        let mut added = vec![p];
        inside[p] = true;
        let mut k = 0;
        let mut refused = false;
        while k < added.len() && !refused {
            let u = added[k];
            k += 1;
            for &e in &self.adj[u] {
                let e = e as usize;
                let v = self.to[e] as usize;
                if self.cap[e] > tol && !inside[v] {
                    // Reaching the sink cannot happen in exact arithmetic -- it would make `p`
                    // sink-side in every min-cut, hence its partner source-side in every one, hence
                    // already inside. Kept for the same reason as the conjunction above: a mutation
                    // deleting it survives, and floats are not exact arithmetic.
                    if forbid.contains(&v) {
                        refused = true;
                        break;
                    }
                    inside[v] = true;
                    added.push(v);
                }
            }
        }
        if refused {
            for &u in &added {
                inside[u] = false;
            }
            return false;
        }
        true
    }

    /// Nodes reachable from `from` over residual arcs: source-side in every min-cut.
    fn reach_forward(&self, from: usize, tol: f64) -> Vec<bool> {
        let mut seen = vec![false; self.adj.len()];
        let mut stack = vec![from];
        seen[from] = true;
        while let Some(u) = stack.pop() {
            for &e in &self.adj[u] {
                let e = e as usize;
                let v = self.to[e] as usize;
                if !seen[v] && self.cap[e] > tol {
                    seen[v] = true;
                    stack.push(v);
                }
            }
        }
        seen
    }

    /// Nodes from which `to` is reachable over residual arcs: sink-side in every min-cut.
    fn reach_backward(&self, to: usize, tol: f64) -> Vec<bool> {
        let mut seen = vec![false; self.adj.len()];
        let mut stack = vec![to];
        seen[to] = true;
        while let Some(v) = stack.pop() {
            // The arcs INTO v are the partners of the arcs out of it.
            for &e in &self.adj[v] {
                let e = e as usize;
                let back = e ^ 1;
                let u = self.to[e] as usize;
                if !seen[u] && self.cap[back] > tol {
                    seen[u] = true;
                    stack.push(u);
                }
            }
        }
        seen
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exact::Elimination;
    use crate::rng::Pcg;

    /// `E` in the binary form the construction is derived from: `(c, a_i, [(i, j, b_ij)])`.
    ///
    /// Written from the algebra in the module doc, NOT from the network, so a test using it and a
    /// test using `roof_dual` cannot agree by sharing a mistake in the transcription.
    fn posiform(g: &Graph) -> (f64, Vec<f64>, Vec<(usize, usize, f64)>) {
        let mut c = 0.0;
        let mut a = vec![0.0; g.n];
        let mut b = Vec::new();
        for i in 0..g.n {
            c += g.h[i];
            a[i] = -2.0 * g.h[i];
            for k in g.offset[i]..g.offset[i + 1] {
                let j = g.nbr[k] as usize;
                a[i] += 2.0 * g.w[k];
                if j > i {
                    c -= g.w[k];
                    b.push((i, j, -4.0 * g.w[k]));
                }
            }
        }
        (c, a, b)
    }

    /// The relaxed energy `G(y, z)`, straight from the formula.
    fn relaxed(g: &Graph, y: &[f64], z: &[f64]) -> f64 {
        let (c, a, b) = posiform(g);
        let mut e = c;
        for i in 0..g.n {
            e += a[i] / 2.0 * (y[i] + 1.0 - z[i]);
        }
        for &(i, j, bij) in &b {
            let half = bij / 2.0;
            e += if bij <= 0.0 {
                half * (y[i] * y[j] + (1.0 - z[i]) * (1.0 - z[j]))
            } else {
                half * (y[i] * (1.0 - z[j]) + (1.0 - z[i]) * y[j])
            };
        }
        e
    }

    /// Minimum of `G` over all `2^(2n)` relaxed points, and which spins every minimiser agrees on.
    fn relaxed_truth(g: &Graph) -> (f64, Vec<Option<i8>>) {
        let n = g.n;
        let mut best = f64::INFINITY;
        let mut agree: Vec<Option<i8>> = vec![None; n];
        let mut first = true;
        for mask in 0u64..(1u64 << (2 * n)) {
            let y: Vec<f64> = (0..n).map(|i| f64::from((mask >> i) as u32 & 1)).collect();
            let z: Vec<f64> = (0..n).map(|i| f64::from((mask >> (n + i)) as u32 & 1)).collect();
            let e = relaxed(g, &y, &z);
            if e < best - 1e-12 {
                best = e;
                first = true;
            }
            if e <= best + 1e-12 {
                let here: Vec<Option<i8>> = (0..n)
                    .map(|i| match (y[i] > 0.5, z[i] > 0.5) {
                        (true, false) => Some(1),
                        (false, true) => Some(-1),
                        _ => None,
                    })
                    .collect();
                if first {
                    agree = here;
                    first = false;
                } else {
                    for i in 0..n {
                        if agree[i] != here[i] {
                            agree[i] = None;
                        }
                    }
                }
            }
        }
        (best, agree)
    }

    /// Every minimum-energy state, by enumeration.
    fn optima(g: &Graph) -> (f64, Vec<Vec<i8>>) {
        let mut best = f64::INFINITY;
        let mut set = Vec::new();
        for mask in 0u32..(1u32 << g.n) {
            let s: Vec<i8> = (0..g.n).map(|i| if mask >> i & 1 == 1 { 1 } else { -1 }).collect();
            let e = g.energy(&s);
            if e < best - 1e-9 {
                best = e;
                set.clear();
            }
            if e <= best + 1e-9 {
                set.push(s);
            }
        }
        (best, set)
    }

    /// Random instance. `ferro` forces every coupling positive, which is the submodular corner.
    fn random_graph(n: usize, p: f64, field: f64, ferro: bool, seed: u64) -> Graph {
        let mut rng = Pcg::new(seed, 0xB0F);
        let mut gb = GraphBuilder::new(n);
        for i in 0..n {
            for j in (i + 1)..n {
                if rng.f64() < p {
                    let w = rng.f64() - 0.5;
                    gb.couple(i, j, if ferro { w.abs() + 0.05 } else { w });
                }
            }
            gb.bias(i, (rng.f64() - 0.5) * field);
        }
        gb.build()
    }

    /// Weights on a quarter grid, so every energy and every relaxed energy is exact in `f64` and
    /// ties are ties rather than near-ties. Degeneracy is the interesting case for persistency and
    /// random floats never produce it.
    fn dyadic_graph(n: usize, p: f64, seed: u64) -> Graph {
        let mut rng = Pcg::new(seed, 0x1D);
        let mut gb = GraphBuilder::new(n);
        for i in 0..n {
            for j in (i + 1)..n {
                if rng.f64() < p {
                    let w = f64::from(rng.next_u32() % 5) / 4.0 - 0.5;
                    if w != 0.0 {
                        gb.couple(i, j, w);
                    }
                }
            }
            gb.bias(i, f64::from(rng.next_u32() % 5) / 4.0 - 0.5);
        }
        gb.build()
    }

    /// THE IDENTITY THE WHOLE MODULE RESTS ON: the relaxation is the energy wherever the pair of
    /// variables for a spin disagrees, which is every state of the original problem.
    ///
    /// If this fails, the bound is a bound on a different function and nothing else here means
    /// anything. Exhaustive over states, over instances with both signs of coupling.
    #[test]
    fn the_relaxation_is_the_energy_on_consistent_labellings() {
        for seed in 0..40u64 {
            let g = random_graph(7, 0.6, 1.0, false, seed);
            for mask in 0u32..(1u32 << g.n) {
                let s: Vec<i8> = (0..g.n).map(|i| if mask >> i & 1 == 1 { 1 } else { -1 }).collect();
                let y: Vec<f64> = s.iter().map(|&v| if v > 0 { 1.0 } else { 0.0 }).collect();
                let z: Vec<f64> = y.iter().map(|v| 1.0 - v).collect();
                let (got, want) = (relaxed(&g, &y, &z), g.energy(&s));
                assert!((got - want).abs() < 1e-9, "seed {seed} mask {mask}: G={got} E={want}");
            }
        }
    }

    /// The max-flow answer IS the minimum of the relaxation, checked against `2^(2n)` enumeration
    /// of the relaxed energy. This is what verifies the network: capacities, constants, the
    /// submodular/supermodular split and Dinic itself, all against a formula that never sees a
    /// graph.
    #[test]
    fn the_flow_computes_the_minimum_of_the_relaxation() {
        for seed in 0..60u64 {
            let g = if seed % 2 == 0 { random_graph(6, 0.7, 1.0, false, seed) } else { dyadic_graph(6, 0.8, seed) };
            let r = roof_dual(&g);
            let (truth, _) = relaxed_truth(&g);
            assert!(r.bound <= truth + 1e-9, "seed {seed}: bound {} above the relaxation {truth}", r.bound);
            assert!(r.bound >= truth - 1e-6, "seed {seed}: bound {} below the relaxation {truth}", r.bound);
        }
    }

    /// SOUNDNESS, against `Elimination::ground_state`: no state is below the bound, on instances
    /// dense enough to be frustrated and small enough to solve exactly.
    #[test]
    fn the_bound_is_never_above_the_exact_ground_energy() {
        let el = Elimination::default();
        let mut worst = 0.0f64;
        for seed in 0..200u64 {
            for &(n, p, field) in &[(8usize, 0.4, 1.0), (10, 0.3, 0.0), (12, 0.25, 2.0)] {
                let g = random_graph(n, p, field, false, seed * 7 + n as u64);
                let truth = el.ground_state(&g).expect("small enough to eliminate").ground_energy.unwrap();
                let r = roof_dual(&g);
                assert!(r.bound <= truth + 1e-12, "n={n} seed={seed}: bound {} > ground {truth}", r.bound);
                worst = worst.max(truth - r.bound);
                // And the `Bound` view must agree with the module it borrows: a gap is never negative.
                let s = el.ground_state(&g).unwrap().ground_state.unwrap();
                assert!(r.as_bound().gap(&g, &s) >= 0.0, "negative gap at n={n} seed={seed}");
            }
        }
        assert!(worst > 0.0, "every instance was solved exactly -- this test is not exercising the gap");
    }

    /// PERSISTENCY, the claim worth having: a spin in `fixed` takes that value in EVERY ground
    /// state, checked against exhaustive enumeration of the optimum set — not one optimum, all of
    /// them, since the failure mode is a degenerate instance where the spin is free.
    #[test]
    fn every_fixed_spin_holds_in_every_optimum() {
        let mut pinned = 0usize;
        for seed in 0..300u64 {
            let g = if seed % 2 == 0 { dyadic_graph(9, 0.35, seed) } else { random_graph(9, 0.35, 1.0, false, seed) };
            let r = roof_dual(&g);
            let (_, opts) = optima(&g);
            for i in 0..g.n {
                if let Some(v) = r.fixed[i] {
                    pinned += 1;
                    for s in &opts {
                        assert_eq!(s[i], v, "seed {seed}: spin {i} pinned to {v}, optimum has {}", s[i]);
                    }
                }
            }
        }
        assert!(pinned > 500, "only {pinned} spins pinned across 300 instances -- the test is vacuous");
    }

    /// WEAK PERSISTENCY: the single-minimiser labelling is jointly extendable — SOME optimum agrees
    /// with all of it at once. Checking each label separately would pass on a labelling no single
    /// state realises, which is exactly the claim `simplify` needs.
    #[test]
    fn the_labelling_extends_to_one_whole_optimum() {
        for seed in 0..600u64 {
            // Dyadic weights at three densities: ties are exact, so the optimum set is genuinely
            // degenerate and a labelling assembled from more than one minimiser has somewhere to go
            // wrong. Random floats almost never produce a tie and would not exercise this.
            let g = match seed % 3 {
                0 => dyadic_graph(9, 0.7, seed),
                1 => dyadic_graph(9, 0.35, seed),
                _ => random_graph(9, 0.4, 1.0, false, seed),
            };
            let r = roof_dual(&g);
            let (_, opts) = optima(&g);
            let ok = opts.iter().any(|s| (0..g.n).all(|i| r.labels[i].is_none_or(|v| s[i] == v)));
            assert!(ok, "seed {seed}: no optimum agrees with the labelling {:?}", r.labels);
            // And a pinned spin must carry the same value here, since `fixed` is the subset of the
            // labelling that survives every min-cut.
            for i in 0..g.n {
                if let Some(v) = r.fixed[i] {
                    assert_eq!(r.labels[i], Some(v), "seed {seed}: spin {i} disagrees with its own label");
                }
            }
        }
    }

    /// `fixed` is exactly the set of spins every minimiser of the relaxation agrees on — the
    /// residual-reachability reading of the network against brute force over all `2^(2n)` relaxed
    /// points. Dyadic weights, so ties are exact and the two sides can be compared for EQUALITY
    /// rather than inclusion.
    #[test]
    fn the_pinned_set_is_what_every_minimiser_of_the_relaxation_agrees_on() {
        for seed in 0..120u64 {
            let g = dyadic_graph(6, 0.7, seed);
            let r = roof_dual_with(&g, Some(0.0));
            let (_, agree) = relaxed_truth(&g);
            assert_eq!(r.fixed, agree, "seed {seed}: reachability and enumeration disagree");
        }
    }

    /// A SUBMODULAR INSTANCE IS SOLVED OUTRIGHT. Every coupling ferromagnetic makes both halves of
    /// the relaxation separable copies of the original, so the bound is the ground energy, the
    /// labelling is complete, and the state it names is optimal. This is the exactness corner of
    /// roof duality and the strongest oracle available here.
    #[test]
    fn a_ferromagnet_with_fields_is_exact_and_pinned_everywhere() {
        let el = Elimination::default();
        for seed in 0..80u64 {
            let g = random_graph(10, 0.4, 2.0, true, seed);
            let r = roof_dual(&g);
            let truth = el.ground_state(&g).unwrap();
            let want = truth.ground_energy.unwrap();
            assert!((r.bound - want).abs() < 1e-9, "seed {seed}: bound {} vs ground {want}", r.bound);
            assert_eq!(r.n_fixed(), g.n, "seed {seed}: submodular instance left {} spins free", g.n - r.n_fixed());
            let s = r.ground_state().expect("complete labelling");
            assert!((g.energy(&s) - want).abs() < 1e-9, "seed {seed}: named state is not optimal");
        }
    }

    /// Fields alone: the bound must be exactly `-sum_i |h_i|` and every spin pinned to its field's
    /// sign. A closed form, and the one case where roof duality has nothing to relax.
    #[test]
    fn fields_alone_are_pinned_to_their_own_sign() {
        let mut rng = Pcg::new(11, 0x5E);
        let n = 12;
        let mut gb = GraphBuilder::new(n);
        let h: Vec<f64> = (0..n).map(|_| rng.f64() - 0.5).collect();
        for i in 0..n {
            gb.bias(i, h[i]);
        }
        let g = gb.build();
        let r = roof_dual(&g);
        let want: f64 = -h.iter().map(|x| x.abs()).sum::<f64>();
        assert!((r.bound - want).abs() < 1e-12, "bound {} vs -sum|h| {want}", r.bound);
        for i in 0..n {
            assert_eq!(r.fixed[i], Some(if h[i] > 0.0 { 1 } else { -1 }), "spin {i} with h={}", h[i]);
        }
    }

    /// THE HARD CORNER, stated as a number: the unbiased frustrated triangle. Its ground energy is
    /// −1 and its roof-dual bound is −3, the trivial floor — because the relaxed minimum sets both
    /// variables of every spin to 1, which costs nothing and means nothing. Nothing may be pinned:
    /// the two optima are related by a global flip, so no spin has a value to be pinned to.
    #[test]
    fn the_frustrated_triangle_is_bounded_at_the_trivial_floor_and_pins_nothing() {
        let mut gb = GraphBuilder::new(3);
        for (i, j) in [(0, 1), (1, 2), (0, 2)] {
            gb.couple(i, j, -1.0);
        }
        let g = gb.build();
        let r = roof_dual(&g);
        let (truth, _) = optima(&g);
        assert!((truth - (-1.0)).abs() < 1e-12, "fixture is wrong: ground {truth}");
        assert!((r.bound - (-3.0)).abs() < 1e-9, "bound {} is not the trivial floor", r.bound);
        assert_eq!(r.n_fixed(), 0, "a spin-flip symmetric instance cannot pin anything");
        assert_eq!(r.n_labelled(), 0, "and nothing consistent to label either");
    }

    /// ELIMINATION IS EXACT: pinning the labelled spins preserves the minimum, and a minimiser of
    /// what is left lifts to a minimiser of the original. Both by enumeration of both models.
    #[test]
    fn eliminating_the_pinned_spins_preserves_the_optimum() {
        let mut reduced_any = false;
        for seed in 0..200u64 {
            let g = if seed % 2 == 0 { dyadic_graph(10, 0.3, seed) } else { random_graph(10, 0.3, 1.0, false, seed) };
            let r = roof_dual(&g);
            let red = r.simplify(&g);
            let (truth, _) = optima(&g);
            if red.graph.n < g.n {
                reduced_any = true;
            }
            let (sub_min, sub_opts) = optima(&red.graph);
            assert!(
                (red.offset + sub_min - truth).abs() < 1e-9,
                "seed {seed}: offset {} + reduced {sub_min} != {truth}",
                red.offset
            );
            let lifted = red.lift(&sub_opts[0]);
            assert!((g.energy(&lifted) - truth).abs() < 1e-9, "seed {seed}: lifted state is not optimal");
        }
        assert!(reduced_any, "nothing was ever eliminated -- the test is vacuous");
    }

    /// NEGATING EVERY FIELD NEGATES EVERY ANSWER: `E(s; -h, w) = E(-s; h, w)`, so the two models
    /// have the same minimum and mirrored optima. `fixed` is defined by the optimum set, so it must
    /// mirror exactly — which is the check that the `+1` and `-1` paths through the construction,
    /// which are NOT symmetric in the code, are symmetric in what they compute.
    #[test]
    fn negating_every_field_mirrors_every_pinned_spin() {
        for seed in 0..300u64 {
            let g = if seed % 2 == 0 { dyadic_graph(9, 0.4, seed) } else { random_graph(9, 0.4, 1.5, false, seed) };
            let mut gb = GraphBuilder::new(g.n);
            for i in 0..g.n {
                gb.bias(i, -g.h[i]);
                for k in g.offset[i]..g.offset[i + 1] {
                    let j = g.nbr[k] as usize;
                    if j > i {
                        gb.couple(i, j, g.w[k]);
                    }
                }
            }
            let flipped = gb.build();
            let (a, b) = (roof_dual(&g), roof_dual(&flipped));
            assert!((a.bound - b.bound).abs() < 1e-9, "seed {seed}: {} vs {}", a.bound, b.bound);
            for i in 0..g.n {
                assert_eq!(b.fixed[i], a.fixed[i].map(|v| -v), "seed {seed}: spin {i} does not mirror");
            }
        }
    }

    /// A non-finite weight has no honest bound and no honest persistency, and the module must say
    /// so rather than run a max-flow on capacities that are not numbers.
    #[test]
    fn a_non_finite_weight_refuses_to_claim_anything() {
        let mut gb = GraphBuilder::new(3);
        gb.couple(0, 1, f64::NAN);
        gb.couple(1, 2, 1.0);
        let g = gb.build();
        let r = roof_dual(&g);
        assert_eq!(r.bound, f64::NEG_INFINITY);
        assert_eq!(r.n_fixed(), 0);
        assert_eq!(r.n_labelled(), 0);
    }

    /// The tolerance is slack in one direction only: raising it may unpin spins, never pin new
    /// ones, and it must never invent a value that disagrees with the tight run.
    #[test]
    fn a_looser_tolerance_only_ever_pins_fewer_spins() {
        for seed in 0..60u64 {
            let g = random_graph(9, 0.4, 1.0, false, seed);
            let tight = roof_dual_with(&g, Some(0.0));
            let loose = roof_dual_with(&g, Some(1e-3));
            for i in 0..g.n {
                if let Some(v) = loose.fixed[i] {
                    assert_eq!(tight.fixed[i], Some(v), "seed {seed}: spin {i} pinned only when loose");
                }
            }
            assert!(loose.n_fixed() <= tight.n_fixed(), "seed {seed}: loose pinned more");
        }
    }

    /// An empty model is not an error: no spins, no arcs, bound zero.
    #[test]
    fn an_empty_model_bounds_at_zero() {
        let g = GraphBuilder::new(0).build();
        let r = roof_dual(&g);
        assert!(r.bound.abs() < 1e-12, "bound {}", r.bound);
        assert_eq!(r.ground_state(), Some(Vec::new()));
    }
}

