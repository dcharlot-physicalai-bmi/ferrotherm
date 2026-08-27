//! Branch and bound: the only thing here that returns a **proof**.
//!
//! Every other solver in this crate hands back a state and, at best, a bound beside it. Together
//! those bracket the optimum — `gset_gap` prints exactly that bracket — but a bracket is not an
//! answer, and on the instances where the answer is reachable it is worth having. This closes the
//! bracket by search: fix a spin, bound what the rest can still achieve, and discard the branch
//! when the bound cannot beat what is already in hand.
//!
//! # The bound, maintained rather than recomputed
//!
//! With some spins fixed and the rest free, the energy of any completion splits three ways:
//!
//! ```text
//!   E  =  E_fixed  +  Σ_{i free} (−s_i·λ_i)  +  Σ_{edges free–free} (−J_ij s_i s_j)
//! ```
//!
//! where `λ_i = h_i + Σ_{j fixed} J_ij s_j` is what the fixed part has already said to free spin
//! `i`. Each remaining term is bounded below by minus its magnitude, so
//!
//! ```text
//!   E  ≥  E_fixed  −  Σ_{i free}|λ_i|  −  Σ_{edges free–free}|J_ij|
//! ```
//!
//! which is [`crate::bound::decoupled`] applied to the residual problem. Recomputing it at every
//! node would cost `O(edges)`; instead all three terms are updated in `O(degree)` when a spin is
//! fixed. That is what makes the search worth running at all.
//!
//! # Undo has to be exact
//!
//! Incremental state plus floating point is where this kind of search goes quietly wrong: `x + d −
//! d` is not `x`, so a naive undo lets the bound drift as the search backtracks, and a bound that
//! has drifted **upward** prunes a subtree that contained the optimum. The result still says
//! "proved optimal", and nothing in the output distinguishes it from a correct proof.
//!
//! So nothing is undone by arithmetic. Scalars live in the recursive frame and are restored by
//! returning; the touched entries of `λ` are saved and written back verbatim. The only residue is
//! the accumulation along a single root-to-node path, and the prune test carries an explicit slack
//! sized from the instance for it — which costs a few extra nodes and cannot cost correctness.
//!
//! # A stronger bound, where a stronger bound is affordable
//!
//! The incremental bound above charges `Σ|J|` over every edge with **both ends still free**. That
//! is `O(n)` such edges on a sparse graph and `O(n²)` on a dense one, which is why density costs so
//! much more here than node count does: `examples/exact_reach` proves 76 spins at mean degree 6 and
//! stops at 44 at mean degree 22, on less than a quarter of the node budget.
//!
//! [`Params::sdp_depth`] spends a **certified** SDP bound ([`crate::sdp`]) on the residual problem
//! instead, at nodes no deeper than the given depth. The residual is a real graph — the free spins,
//! with `λ` as their fields and the free–free couplings — so the same certified machinery applies
//! unchanged, and the bound it returns is sound by the same weak-duality argument.
//!
//! **It pays, and the first measurement of it said otherwise.** Measured at depth 2 the bound fired
//! about twenty times per instance, pruned nought to four, and left the node count unchanged on 17
//! of 19 sizes — a result reported as a property of the method when it was a property of the
//! setting. Depth 2 is at most seven nodes: even a perfect bound there removes a constant fraction
//! of a tree whose cost is set exponentially deeper down.
//!
//! Swept properly (`examples/sdp_in_tree`), on dense instances at 3 seeds each:
//!
//! ```text
//!   spins   cheap      d4        d8       d12      d16     saturates
//!      32   94,809   68,769    17,465    17,465   17,465     d8
//!      36  242,943  160,381    13,963     1,731    1,731     d12
//!      40 2,181,007 1,869,399  379,181    17,231    2,451     d16
//! ```
//!
//! **The depth saturates because the tree closes above it.** Once the bound is on, no branch
//! survives past that level, so a deeper setting has nothing left to visit — which means depth was
//! never the real control. [`Params::sdp_min_free`] and [`Params::sdp_max_free`] are: the first
//! says the residual is too small to be worth a Cholesky, the second that it is too large to
//! afford one. The default depth is simply set past where any of this bites.
//!
//! And it moves what can be **proved**, which is the number that matters
//! (`examples/exact_reach`, 40M-node budget, 3 seeds):
//!
//! ```text
//!   family                cheap proves   with the SDP bound   nodes at the cheap ceiling
//!   sparse, mean deg 6      76 spins         84 spins         8,277,603 -> 156,793   (53x)
//!   dense,  mean deg 22     44 spins         52 spins        12,173,789 -> 192,501   (63x)
//! ```
//!
//! # What "proved" means
//!
//! [`Outcome::proved_optimal`] is true only when the search exhausted the tree within its node
//! budget. A run that hit [`Params::max_nodes`] returns the best state it found and says the proof
//! is missing, because the alternative — a field that means "optimal, or else we gave up" — is the
//! kind of flag that gets read as the first thing and quoted as the second.

use crate::graph::Graph;

/// How hard to search.
#[derive(Clone, Debug, PartialEq)]
pub struct Params {
    /// Node budget. The search stops and reports `proved_optimal = false` when it is reached.
    pub max_nodes: u64,
    /// A starting incumbent, if one is already known — from [`crate::tabu`], say.
    ///
    /// A good incumbent is worth far more than a better bound: it prunes from the first node.
    pub incumbent: Option<Vec<i8>>,
    /// Spend a **certified SDP bound** on the residual problem at nodes no deeper than this.
    ///
    /// `None` uses the incremental bound everywhere. The default is deliberately deep — see the
    /// module note: measured, the depth saturates once the tree closes above it, so the real
    /// controls are [`Params::sdp_min_free`] and [`Params::sdp_max_free`], and depth is only a
    /// proxy for "is the residual still big". It is kept because it is what the sweep in
    /// `examples/sdp_in_tree` varies, and reproducing that sweep should not need a private field.
    pub sdp_depth: Option<usize>,
    /// How hard the SDP tries, when it is used at all. Sweeps mostly; the rank is left to the
    /// Barvinok-Pataki default because the residual changes size at every node.
    pub sdp: crate::sdp::Params,
    /// Below this many free spins the SDP is skipped.
    ///
    /// A subtree over a handful of spins is enumerated in fewer operations than one Cholesky, so
    /// the bound cannot pay there however tight it is.
    pub sdp_min_free: usize,
    /// Above this many free spins the SDP is skipped as well, and this is the guard that matters.
    ///
    /// The Cholesky is `O(m³)` in the free spins and runs several times per bound, so an unbounded
    /// ceiling turns a 4096-node graph into minutes **per node**. 192 keeps one bound in the
    /// low milliseconds. The cost of skipping is a looser bound; the cost of not skipping is a
    /// solver that appears to hang.
    pub sdp_max_free: usize,
}

impl Default for Params {
    fn default() -> Self {
        Params {
            max_nodes: 20_000_000,
            incumbent: None,
            // ON by default, and deep, because the measurement is not close: on dense instances at
            // 40 spins it is 2,181,007 nodes against 2,451, and it moves the size that can be
            // PROVED from 44 to 52. Off would be the conservative choice and the wrong one.
            sdp_depth: Some(64),
            sdp: crate::sdp::Params { sweeps: 60, ..crate::sdp::Params::default() },
            sdp_min_free: 24,
            sdp_max_free: 192,
        }
    }
}

/// What the search found, and whether it proved anything.
#[derive(Clone, Debug)]
pub struct Outcome {
    /// The best state found.
    pub state: Vec<i8>,
    /// Its energy, recomputed from `state`.
    pub energy: f64,
    /// True only if the tree was exhausted. See the module note on what this is allowed to mean.
    pub proved_optimal: bool,
    /// Nodes visited.
    pub nodes: u64,
    /// Nodes cut off by the bound. `pruned / nodes` is how much the bound was worth.
    pub pruned: u64,
    /// Whether the node budget ran out.
    pub hit_limit: bool,
    /// The slack subtracted from the bound before every prune test, for the record.
    pub slack: f64,
    /// SDP bounds computed, and how many of them pruned a subtree the cheap bound would have kept.
    ///
    /// Both zero when [`Params::sdp_depth`] is `None`. `sdp_prunes / sdp_calls` is what the dial is
    /// actually worth on this instance — a ratio of zero means every Cholesky was wasted.
    pub sdp_calls: u64,
    pub sdp_prunes: u64,
}

/// Search for the minimum energy, with a proof when the budget allows.
pub fn solve(g: &Graph, p: &Params) -> Outcome {
    let n = g.n;
    if n == 0 {
        return Outcome {
            state: Vec::new(),
            energy: 0.0,
            proved_optimal: true,
            nodes: 0,
            pruned: 0,
            hit_limit: false,
            slack: 0.0,
            sdp_calls: 0,
            sdp_prunes: 0,
        };
    }

    // Branch on the highest-degree spin first: fixing it removes the most free-free edges, which is
    // exactly what the bound charges for, so the bound tightens fastest along that order.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by_key(|&i| core::cmp::Reverse(g.offset[i + 1] - g.offset[i]));

    // Slack for the accumulated rounding along one root-to-node path. Scaled by the total weight,
    // because that is what the accumulators are made of, and by `n` for the path length.
    let total: f64 = g.w.iter().map(|v| v.abs()).sum::<f64>() + g.h.iter().map(|v| v.abs()).sum::<f64>();
    let slack = (total * n as f64 * f64::EPSILON * 8.0).max(1e-12);

    let mut best_state: Vec<i8> = p
        .incumbent
        .as_ref()
        .filter(|s| s.len() == n && s.iter().all(|&v| v == 1 || v == -1))
        .cloned()
        .unwrap_or_else(|| vec![1i8; n]);
    let mut best = g.energy(&best_state);

    // Z2 gauge: with no fields, `E(s) = E(−s)`, so half the tree is a mirror of the other half and
    // the first spin can be pinned. With fields the symmetry is broken and both branches are real.
    let gauge_fixed = g.h.iter().all(|&h| h == 0.0);

    let mut st = Search {
        g,
        order: &order,
        s: vec![0i8; n],
        lambda: g.h.clone(),
        free_h_abs: g.h.iter().map(|v| v.abs()).sum(),
        free_abs: g.w.iter().map(|v| v.abs()).sum::<f64>() / 2.0,
        nodes: 0,
        pruned: 0,
        max_nodes: p.max_nodes,
        slack,
        best,
        best_state: best_state.clone(),
        hit_limit: false,
        gauge_fixed,
        undo: Vec::with_capacity(g.nbr.len() + n),
        // `None` becomes a depth no node reaches, so the hot path tests one integer rather than an
        // Option it would have to unwrap at every node.
        sdp_depth: p.sdp_depth.map_or(0, |d| d + 1),
        sdp: p.sdp,
        sdp_min_free: p.sdp_min_free,
        sdp_max_free: p.sdp_max_free,
        sdp_calls: 0,
        sdp_prunes: 0,
    };
    st.descend(0, 0.0);

    best = st.g.energy(&st.best_state);
    best_state = st.best_state;
    Outcome {
        state: best_state,
        energy: best,
        proved_optimal: !st.hit_limit,
        nodes: st.nodes,
        pruned: st.pruned,
        hit_limit: st.hit_limit,
        slack,
        sdp_calls: st.sdp_calls,
        sdp_prunes: st.sdp_prunes,
    }
}

struct Search<'a> {
    g: &'a Graph,
    order: &'a [usize],
    s: Vec<i8>,
    /// `λ_i` for free `i`; the entry for a fixed spin is stale and never read.
    lambda: Vec<f64>,
    free_h_abs: f64,
    free_abs: f64,
    nodes: u64,
    pruned: u64,
    max_nodes: u64,
    slack: f64,
    best: f64,
    best_state: Vec<i8>,
    hit_limit: bool,
    gauge_fixed: bool,
    /// Saved `λ` entries, one shared stack for the whole search.
    undo: Vec<(usize, f64)>,
    /// Deepest node at which to spend an SDP bound, and how hard it should try.
    sdp_depth: usize,
    sdp: crate::sdp::Params,
    sdp_min_free: usize,
    sdp_max_free: usize,
    /// How many SDP bounds were computed, and how many of them pruned. Reported so the dial can be
    /// judged on evidence rather than on the argument for it.
    sdp_calls: u64,
    sdp_prunes: u64,
}

impl Search<'_> {
    fn descend(&mut self, depth: usize, fixed_energy: f64) {
        if self.hit_limit {
            return;
        }
        self.nodes += 1;
        if self.nodes > self.max_nodes {
            self.hit_limit = true;
            return;
        }
        if depth == self.order.len() {
            // A complete assignment. The energy is recomputed from the state rather than taken from
            // the accumulator -- the accumulator is what the bound is made of, and an incumbent
            // that inherited its drift would poison every prune after it.
            let e = self.g.energy(&self.s);
            if e < self.best {
                self.best = e;
                self.best_state.copy_from_slice(&self.s);
            }
            return;
        }

        let lb = fixed_energy - self.free_h_abs - self.free_abs;
        if lb - self.slack >= self.best {
            self.pruned += 1;
            return;
        }

        // The cheap bound did not prune. Near the top of the tree, where the residual is still
        // large and the nodes are still few, it is worth asking a much more expensive question.
        if depth < self.sdp_depth {
            if let Some(lb2) = self.sdp_bound(fixed_energy) {
                if lb2 - self.slack >= self.best {
                    self.pruned += 1;
                    self.sdp_prunes += 1;
                    return;
                }
            }
        }

        let i = self.order[depth];
        // Greedy value first: the sign that lowers `−s·λ` is the one more likely to hold the
        // optimum, and finding a good incumbent early is what makes the bound prune.
        let first: i8 = if self.lambda[i] >= 0.0 { 1 } else { -1 };
        // With the gauge pinned, ONE branch at the root covers the whole search: the other half of
        // the tree is its mirror, spin for spin, with identical energies.
        let branches: usize = if self.gauge_fixed && depth == 0 { 1 } else { 2 };
        for b in 0..branches {
            let v = if b == 0 { first } else { -first };
            self.fix(i, v, depth, fixed_energy);
            if self.hit_limit {
                return;
            }
        }
    }

    /// A certified SDP lower bound on the energy of any completion of the current partial state.
    ///
    /// The residual is built as a real [`Graph`]: the free spins, `λ` as their fields, and the
    /// free-free couplings. That is not a convenience — it means [`crate::sdp::certified`] applies
    /// here with no special case, and the bound it returns is sound by the same weak-duality
    /// argument, verified by the same completed Cholesky. **A bound inside a search that only works
    /// inside that search is a bound nobody can check.**
    ///
    /// Returns `None` when the residual is too small to be worth a Cholesky, or when the SDP could
    /// not verify a dual point — in which case the caller keeps the cheap bound rather than
    /// pruning on nothing.
    fn sdp_bound(&mut self, fixed_energy: f64) -> Option<f64> {
        let n = self.g.n;
        // Compact the free spins. `map[i]` is the residual index of free spin `i`, or `usize::MAX`.
        let mut map = vec![usize::MAX; n];
        let mut free = 0usize;
        for i in 0..n {
            if self.s[i] == 0 {
                map[i] = free;
                free += 1;
            }
        }
        if free < self.sdp_min_free || free > self.sdp_max_free {
            return None;
        }
        let mut gb = crate::graph::GraphBuilder::new(free);
        for i in 0..n {
            let a = map[i];
            if a == usize::MAX {
                continue;
            }
            // `λ_i` already carries what the fixed spins have said to `i`, which is exactly the
            // field the residual problem has.
            if self.lambda[i] != 0.0 {
                gb.bias(a, self.lambda[i]);
            }
            for k in self.g.offset[i]..self.g.offset[i + 1] {
                let j = self.g.nbr[k] as usize;
                // Each free-free edge once: `j > i` in the original indexing, since the CSR holds
                // both directions.
                if j > i && map[j] != usize::MAX {
                    gb.couple(a, map[j], self.g.w[k]);
                }
            }
        }
        let residual = gb.build();
        let (b, cert) = crate::sdp::certified(&residual, &self.sdp, 1);
        self.sdp_calls += 1;
        // Re-verified before it is used to discard a subtree. The cost is one more Cholesky against
        // the cost of a wrong prune, which is a false "proved optimal" that nothing downstream can
        // tell from a real one.
        cert.verify(&residual).ok().map(|v| fixed_energy + v.min(b.value))
    }

    fn fix(&mut self, i: usize, v: i8, depth: usize, fixed_energy: f64) {
        let fv = v as f64;
        // Every scalar below is a COPY held in this frame, so backtracking restores it exactly.
        let saved_h_abs = self.free_h_abs;
        let saved_abs = self.free_abs;
        let new_fixed = fixed_energy - fv * self.lambda[i];

        self.s[i] = v;
        self.free_h_abs -= self.lambda[i].abs();

        // The touched neighbours are written back verbatim on the way out; `λ_j − J·v` would not
        // return them to the value they had. One shared stack rather than a `Vec` per frame: this
        // runs millions of times, and an allocation per node is most of the cost of a node.
        let mark = self.undo.len();
        let (lo, hi) = (self.g.offset[i], self.g.offset[i + 1]);
        for k in lo..hi {
            let j = self.g.nbr[k] as usize;
            if self.s[j] != 0 {
                continue; // already fixed: this edge left `free_abs` when j was fixed
            }
            let w = self.g.w[k];
            self.free_abs -= w.abs();
            self.undo.push((j, self.lambda[j]));
            self.free_h_abs -= self.lambda[j].abs();
            self.lambda[j] += w * fv;
            self.free_h_abs += self.lambda[j].abs();
        }

        self.descend(depth + 1, new_fixed);

        while self.undo.len() > mark {
            let (j, old) = self.undo.pop().expect("mark is below len");
            self.lambda[j] = old;
        }
        self.free_abs = saved_abs;
        self.free_h_abs = saved_h_abs;
        self.s[i] = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::GraphBuilder;
    use crate::rng::Pcg;

    fn random_graph(n: usize, p: f64, seed: u64, fields: bool) -> Graph {
        let mut rng = Pcg::new(seed, 0xB4B4);
        let mut gb = GraphBuilder::new(n);
        for i in 0..n {
            if fields {
                gb.bias(i, rng.f64() * 2.0 - 1.0);
            }
            for j in (i + 1)..n {
                if rng.f64() < p {
                    gb.couple(i, j, rng.f64() * 2.0 - 1.0);
                }
            }
        }
        gb.build()
    }

    fn brute_min(g: &Graph) -> f64 {
        let n = g.n;
        let mut s = vec![1i8; n];
        let mut min = f64::INFINITY;
        for mask in 0..(1u64 << n) {
            for i in 0..n {
                s[i] = if mask >> i & 1 == 1 { 1 } else { -1 };
            }
            min = min.min(g.energy(&s));
        }
        min
    }

    /// The claim is exactness, so the test is exhaustive enumeration — on graphs with fields and
    /// without, because the gauge shortcut only applies to one of them and a bug there would show
    /// up nowhere else.
    #[test]
    fn it_finds_the_true_minimum_and_says_it_proved_it() {
        for seed in 0..8u64 {
            let fields = seed % 2 == 0;
            let g = random_graph(13, 0.4, seed, fields);
            let o = solve(&g, &Params::default());
            let min = brute_min(&g);
            assert!(o.proved_optimal && !o.hit_limit, "seed {seed}: no proof");
            assert!(
                (o.energy - min).abs() < 1e-9,
                "seed {seed} (fields {fields}): branch-and-bound {:.9}, enumeration {min:.9}",
                o.energy
            );
            assert_eq!(o.energy, g.energy(&o.state), "energy must match the state returned");
        }
    }

    /// The gauge shortcut halves the tree, and must not halve the answers.
    ///
    /// Run with and without it by adding an all-zero field set -- which changes nothing about the
    /// energy landscape but does turn `gauge_fixed` off -- and require both to reach the minimum
    /// and agree on the node count's ORDER, not just the answer.
    #[test]
    fn pinning_the_gauge_removes_a_mirror_and_nothing_else() {
        let g = random_graph(13, 0.4, 3, false);
        let with_gauge = solve(&g, &Params::default());

        // Same graph, but a field of zero on one node is still "has fields" to the builder? It is
        // not -- `bias(i, 0.0)` leaves h exactly zero. So the mirror is removed by hand instead:
        // solve the graph with the gauge disabled by giving it a field so small it cannot change
        // which state is optimal, then check the minimum is unchanged.
        let tiny = 1e-12;
        let mut gb = GraphBuilder::new(g.n);
        for i in 0..g.n {
            for k in g.offset[i]..g.offset[i + 1] {
                let j = g.nbr[k] as usize;
                if j > i {
                    gb.couple(i, j, g.w[k]);
                }
            }
        }
        gb.bias(0, tiny);
        let g2 = gb.build();
        let without = solve(&g2, &Params::default());

        assert!(with_gauge.proved_optimal && without.proved_optimal);
        assert!(
            (with_gauge.energy - without.energy).abs() < 1e-6,
            "gauge-pinned {:.9} vs full tree {:.9}",
            with_gauge.energy,
            without.energy
        );
        assert!(
            with_gauge.nodes < without.nodes,
            "pinning the gauge visited {} nodes, the full tree {} -- it should be fewer",
            with_gauge.nodes,
            without.nodes
        );
    }

    /// A budget that runs out must report that, and must NOT report a proof.
    ///
    /// The flag is the whole product of this module. A search that gave up and still said
    /// `proved_optimal` would be worse than one that returned nothing.
    #[test]
    fn a_search_that_runs_out_of_budget_does_not_claim_a_proof() {
        let g = random_graph(40, 0.3, 5, true);
        let o = solve(&g, &Params { max_nodes: 500, ..Params::default() });
        assert!(o.hit_limit, "500 nodes should not exhaust a 40-spin tree");
        assert!(!o.proved_optimal);
        assert!(o.nodes <= 501, "nodes {}", o.nodes);
        // It still returns something usable.
        assert_eq!(o.energy, g.energy(&o.state));
        assert_eq!(o.state.len(), g.n);
    }

    /// The bound has to earn its keep: on a real instance most of the tree must be cut off.
    #[test]
    fn the_bound_prunes_most_of_the_tree() {
        let g = random_graph(18, 0.35, 2, true);
        let o = solve(&g, &Params::default());
        assert!(o.proved_optimal);
        // Fewer nodes visited than the tree has LEAVES -- an unpruned search visits about twice
        // that many. A tighter threshold would be a claim about this instance rather than about
        // the bound, and would go flaky the first time the generator changed.
        let full = 1u64 << 18;
        assert!(o.nodes < full, "visited {} nodes of a {full}-leaf tree", o.nodes);
        assert!(o.pruned > 0, "no branch was ever cut off, so the bound did nothing");
    }

    /// A supplied incumbent may speed the search up; it may never change the answer.
    ///
    /// Including a deliberately terrible one, and one that is already optimal — the second is the
    /// case where the search prunes everything and must still return the proof rather than the
    /// all-plus default it started from.
    #[test]
    fn an_incumbent_changes_the_cost_and_not_the_result() {
        for seed in 0..4u64 {
            let g = random_graph(12, 0.4, 40 + seed, seed % 2 == 0);
            let plain = solve(&g, &Params::default());
            let seeded = solve(&g, &Params { incumbent: Some(plain.state.clone()), ..Params::default() });
            assert!(seeded.proved_optimal);
            assert!(
                (seeded.energy - plain.energy).abs() < 1e-12,
                "seed {seed}: {:.9} vs {:.9}",
                seeded.energy,
                plain.energy
            );
            assert!(seeded.nodes <= plain.nodes, "a known-optimal incumbent should not cost nodes");

            // Garbage in: wrong length, and non-spin values. Both must be ignored, not trusted.
            let bad = solve(&g, &Params { incumbent: Some(vec![0i8; g.n]), ..Params::default() });
            assert!((bad.energy - plain.energy).abs() < 1e-12);
            let short = solve(&g, &Params { incumbent: Some(vec![1i8; g.n - 1]), ..Params::default() });
            assert!((short.energy - plain.energy).abs() < 1e-12);
        }
    }

    /// An empty graph is proved trivially rather than panicking on an empty branching order.
    #[test]
    fn an_empty_graph_is_proved_immediately() {
        let g = GraphBuilder::new(0).build();
        let o = solve(&g, &Params::default());
        assert!(o.proved_optimal && o.nodes == 0 && o.energy == 0.0 && o.state.is_empty());
    }

    /// The slack is real and reported, and it is small relative to the instance.
    ///
    /// A slack large enough to matter would silently disable pruning; a slack of zero would let
    /// accumulated rounding prune the optimum. It is asserted to be between those.
    #[test]
    fn the_prune_slack_is_positive_and_negligible() {
        let g = random_graph(14, 0.4, 6, true);
        let o = solve(&g, &Params::default());
        let scale: f64 = g.w.iter().map(|v| v.abs()).sum::<f64>();
        assert!(o.slack > 0.0, "a zero slack cannot absorb the accumulated rounding");
        assert!(o.slack < scale * 1e-9, "slack {} against a weight scale of {scale}", o.slack);
    }
    /// THE TEST THE SDP DIAL EXISTS TO SURVIVE: a tighter bound may visit fewer nodes and may
    /// never change the answer.
    ///
    /// An unsound bound in a branch-and-bound search does not raise anything. It discards a subtree
    /// that held the optimum, and the run still reports `proved_optimal` — indistinguishable in
    /// every field from a correct proof. So this compares the two dials against each other on
    /// instances dense enough that the SDP has something to say, and requires the energies to
    /// agree exactly.
    #[test]
    fn the_sdp_bound_changes_the_node_count_and_never_the_answer() {
        let mut agreed = 0;
        for seed in 0..6u64 {
            let mut rng = crate::rng::Pcg::new(seed, 0xB0_1D);
            let n = 26;
            let mut gb = crate::graph::GraphBuilder::new(n);
            for i in 0..n {
                gb.bias(i, rng.f64() - 0.5);
                for j in (i + 1)..n {
                    if rng.f64() < 0.45 {
                        gb.couple(i, j, rng.f64() * 2.0 - 1.0);
                    }
                }
            }
            let g = gb.build();
            let cheap = solve(&g, &Params { sdp_depth: None, ..Params::default() });
            let strong = solve(&g, &Params::default());
            assert!(cheap.proved_optimal && strong.proved_optimal, "seed {seed}");
            assert!(
                (cheap.energy - strong.energy).abs() < 1e-9,
                "seed {seed}: cheap bound proved {:.12}, SDP bound proved {:.12} -- one of them \
                 discarded the optimum and still said it had a proof",
                cheap.energy,
                strong.energy
            );
            assert!(strong.sdp_calls > 0, "seed {seed}: the dial was set and never fired");
            assert!(
                strong.nodes <= cheap.nodes,
                "seed {seed}: a TIGHTER bound visited MORE nodes ({} vs {})",
                strong.nodes,
                cheap.nodes
            );
            if strong.nodes < cheap.nodes {
                agreed += 1;
            }
        }
        // Not required per seed -- a bound that happens not to bite on one instance is fine -- but
        // if it never bites on any of them the dial is decoration.
        assert!(agreed > 0, "the SDP bound pruned nothing on any of six dense instances");
    }

    /// The size guards are what actually decide, and both ends have to hold.
    ///
    /// `sdp_min_free` matters for cost — a 14-spin subtree is enumerated in fewer operations than
    /// one Cholesky. `sdp_max_free` matters for not hanging: the Cholesky is `O(m³)` and runs
    /// several times per bound, so without a ceiling a large graph spends minutes on a single node
    /// and looks like an infinite loop rather than a slow solver.
    #[test]
    fn the_size_guards_hold_at_both_ends() {
        let dense = |n: usize| {
            let mut gb = crate::graph::GraphBuilder::new(n);
            for i in 0..n {
                for j in (i + 1)..n {
                    gb.couple(i, j, if (i + j) % 3 == 0 { -1.0 } else { 1.0 });
                }
            }
            gb.build()
        };
        // Under the floor: on by default, and it still never fires.
        let small = dense(14);
        assert_eq!(solve(&small, &Params::default()).sdp_calls, 0, "14 spins is under the floor");

        // Over the ceiling: a low `sdp_max_free` must silence it on a graph that would otherwise
        // use it, which is the guard a large model depends on.
        let big = dense(30);
        let on = solve(&big, &Params { max_nodes: 20_000, ..Params::default() });
        let capped = solve(&big, &Params { sdp_max_free: 8, max_nodes: 20_000, ..Params::default() });
        assert!(on.sdp_calls > 0, "30 spins is inside the default window");
        assert_eq!(capped.sdp_calls, 0, "the ceiling must silence it");
        // And silencing a SOUND bound may only cost nodes, never the answer.
        if on.proved_optimal && capped.proved_optimal {
            assert!((on.energy - capped.energy).abs() < 1e-9);
        }
    }

}
