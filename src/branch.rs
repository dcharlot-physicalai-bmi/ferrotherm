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
}

impl Default for Params {
    fn default() -> Self {
        Params { max_nodes: 20_000_000, incumbent: None }
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
        let o = solve(&g, &Params { max_nodes: 500, incumbent: None });
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
}
