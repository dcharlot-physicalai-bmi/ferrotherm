//! Exact answers on sparse graphs, well past where enumeration stops.
//!
//! [`crate::oracle::Exhaustive`] is exact and dies at about twenty-six spins, because it visits
//! `2^n` states. Variable elimination visits `2^w` instead, where `w` is the **induced width** of
//! the elimination order — a property of the graph's shape rather than its size. A tree has width
//! 1 and a lattice strip has width equal to its short side, so a thousand-spin chain is exact and
//! instant while a thousand-spin dense graph is still hopeless. That is the honest trade, and
//! [`Elimination::width`] reports it up front so a caller can decide before waiting.
//!
//! Two questions, one algorithm:
//!
//! - **min-sum** eliminates by taking the minimum over each variable, giving the exact ground state.
//! - **sum-product** eliminates by log-sum-exp, giving the exact log partition function — and with
//!   it exact marginals, which is what lets a sampler be checked against truth on graphs far too
//!   large to enumerate.
//!
//! The elimination order comes from the min-fill heuristic. Finding the optimal order is NP-hard,
//! but the order only affects the width, and the width is measured rather than assumed: a bad order
//! makes this slow or refused, **never wrong**.
//!
//! How good is min-fill in practice? Measured against known treewidths
//! (`examples/width_probe.rs`):
//!
//! | graph | spins | min-fill width | true treewidth |
//! |---|---|---|---|
//! | chain, any length | 2000 | 1 | 1 |
//! | 3x20 strip | 60 | 3 | 3 |
//! | 4x20 strip | 80 | 4 | 4 |
//! | 5x30 strip | 150 | 5 | 5 |
//! | 6x40 strip | 240 | **8** | 6 |
//! | 8x50 strip | 400 | **11** | 8 |
//! | 10x10 grid | 100 | **13** | 10 |
//!
//! Optimal up to width 5, then drifting two or three above. Since cost is `2^width`, being three
//! over is an eightfold price — worth knowing before blaming the machine, and worth revisiting if
//! exact inference on wider graphs ever becomes load-bearing.

use crate::graph::Graph;

/// A function over a subset of spins, as a table indexed by a bitmask.
///
/// Bit `k` of the index is the value of `vars[k]`: 0 means −1, 1 means +1.
#[derive(Clone, Debug)]
struct Table {
    vars: Vec<usize>,
    vals: Vec<f64>,
}

impl Table {
    fn value_at(&self, assign: &[i8]) -> f64 {
        let mut idx = 0usize;
        for (k, &v) in self.vars.iter().enumerate() {
            if assign[v] > 0 {
                idx |= 1 << k;
            }
        }
        self.vals[idx]
    }
}

/// Exact inference by variable elimination.
pub struct Elimination {
    /// Refuse an order whose induced width exceeds this. `2^width` is the memory per table.
    pub max_width: usize,
}

impl Default for Elimination {
    fn default() -> Self {
        Elimination { max_width: 24 }
    }
}

/// What an elimination run produced.
#[derive(Clone, Debug)]
pub struct Exact {
    /// Induced width of the order actually used. Cost was `2^width` per step.
    pub width: usize,
    /// Ground energy, if min-sum was run.
    pub ground_energy: Option<f64>,
    /// A state attaining it.
    pub ground_state: Option<Vec<i8>>,
    /// `log Z` at the requested beta, if sum-product was run.
    pub log_z: Option<f64>,
}

/// Why elimination declined.
#[derive(Clone, Debug, PartialEq)]
pub enum TooWide {
    /// The best order this found still needs a table of `2^width`.
    Width { width: usize, max: usize },
}

impl core::fmt::Display for TooWide {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            TooWide::Width { width, max } => write!(
                f,
                "the elimination order has induced width {width}, needing tables of 2^{width}; the \
                 limit is {max}. This graph is too dense for exact inference -- use a planted \
                 instance for known ground truth instead."
            ),
        }
    }
}

/// Order variables by min-fill: repeatedly eliminate whichever variable adds fewest new edges.
///
/// Returns the order and the induced width it produces.
fn min_fill_order(n: usize, adj: &[Vec<usize>]) -> (Vec<usize>, usize) {
    let mut nbr: Vec<std::collections::BTreeSet<usize>> =
        adj.iter().map(|v| v.iter().copied().collect()).collect();
    let mut alive: Vec<bool> = vec![true; n];
    let mut order = Vec::with_capacity(n);
    let mut width = 0;

    for _ in 0..n {
        // pick the live variable whose elimination creates the fewest fill edges
        let mut best = usize::MAX;
        let mut best_fill = usize::MAX;
        let mut best_deg = usize::MAX;
        for v in 0..n {
            if !alive[v] {
                continue;
            }
            let ns: Vec<usize> = nbr[v].iter().copied().filter(|&u| alive[u]).collect();
            let mut fill = 0;
            for a in 0..ns.len() {
                for b in (a + 1)..ns.len() {
                    if !nbr[ns[a]].contains(&ns[b]) {
                        fill += 1;
                    }
                }
            }
            if fill < best_fill || (fill == best_fill && ns.len() < best_deg) {
                best = v;
                best_fill = fill;
                best_deg = ns.len();
            }
        }
        let v = best;
        let ns: Vec<usize> = nbr[v].iter().copied().filter(|&u| alive[u]).collect();
        width = width.max(ns.len());
        // connect the neighbourhood into a clique, which is what elimination does to the graph
        for a in 0..ns.len() {
            for b in (a + 1)..ns.len() {
                nbr[ns[a]].insert(ns[b]);
                nbr[ns[b]].insert(ns[a]);
            }
        }
        alive[v] = false;
        order.push(v);
    }
    (order, width)
}

fn initial_tables(g: &Graph, beta: f64) -> Vec<Table> {
    // Energies, scaled by beta once here so neither elimination pass has to think about it.
    let mut out = Vec::new();
    for i in 0..g.n {
        if g.h[i] != 0.0 {
            // -h s
            out.push(Table { vars: vec![i], vals: vec![beta * g.h[i], -beta * g.h[i]] });
        }
        for k in g.offset[i]..g.offset[i + 1] {
            let j = g.nbr[k] as usize;
            if j > i {
                let w = beta * g.w[k];
                // index bit0 = i, bit1 = j; value is -w * s_i * s_j
                out.push(Table { vars: vec![i, j], vals: vec![-w, w, w, -w] });
            }
        }
    }
    out
}

fn adjacency(g: &Graph) -> Vec<Vec<usize>> {
    (0..g.n)
        .map(|i| (g.offset[i]..g.offset[i + 1]).map(|k| g.nbr[k] as usize).collect())
        .collect()
}

impl Elimination {
    /// Exact ground state and its energy.
    pub fn ground_state(&self, g: &Graph) -> Result<Exact, TooWide> {
        self.run(g, 1.0, true)
    }

    /// Exact `log Z` at inverse temperature `beta`.
    pub fn log_partition(&self, g: &Graph, beta: f64) -> Result<Exact, TooWide> {
        self.run(g, beta, false)
    }

    /// Exact single-site marginals `P(s_i = +1)` at inverse temperature `beta`.
    ///
    /// The module says sum-product gives log Z "and with it exact marginals, which is what lets a
    /// sampler be checked against truth on graphs far too large to enumerate". It gave log Z. This
    /// is the rest of that sentence.
    ///
    /// # How, and what it costs
    ///
    /// Condition, do not differentiate. For each node, `log Z` is computed twice on the graph with
    /// that node pinned to `+1` and to `-1`, and
    ///
    /// ```text
    ///     P(s_i = +1) = sigma( log Z(s_i = +1) - log Z(s_i = -1) )
    /// ```
    ///
    /// which is a sigmoid of a difference: the total `log Z` cancels, so it never has to be
    /// accurate, and neither does the `ln 2` from the pinned node being left in the graph as an
    /// isolated free spin. Pinning `s_i = v` means dropping node `i`'s couplings and folding each
    /// into its neighbour's field as `h_j += J_ij * v`, plus the `beta * h_i * v` the node itself
    /// contributes — and those two constants differ between the `+1` and `-1` runs by exactly
    /// `2 * beta * h_i`, which is why the field appears in the difference below.
    ///
    /// **The cost is `2n` eliminations**, so `O(n * 2^w)` rather than the single `O(2^w)` of
    /// [`Self::log_partition`]. That is the price of an exact answer per node from a routine that
    /// returns one number; a message-passing formulation would get all of them from two passes and
    /// is a different algorithm. Refused, not approximated, when the width is too large: the same
    /// [`TooWide`] the other two return, from the same order.
    ///
    /// Conditioning changes the graph but never its width — pinning a node only REMOVES edges — so
    /// a model whose `log_partition` succeeds cannot have a marginal that is refused for width.
    pub fn marginals(&self, g: &Graph, beta: f64) -> Result<Vec<f64>, TooWide> {
        // Refuse up front on the unconditioned graph, so a caller learns the width before paying
        // for 2n eliminations rather than after the first one.
        let w = self.width(g);
        if w > self.max_width {
            return Err(TooWide::Width { width: w, max: self.max_width });
        }
        let mut out = Vec::with_capacity(g.n);
        for i in 0..g.n {
            let plus = self.log_partition(&pin(g, i, 1.0), beta)?.log_z.expect("sum-product was run");
            let minus = self.log_partition(&pin(g, i, -1.0), beta)?.log_z.expect("sum-product was run");
            // The pinned node is left in the graph as an isolated free spin in both runs, so its
            // factor of two cancels along with everything else that does not depend on v.
            let delta = 2.0 * beta * g.h[i] + plus - minus;
            out.push(1.0 / (1.0 + (-delta).exp()));
        }
        Ok(out)
    }

    /// Induced width of the order this would use, without running anything.
    pub fn width(&self, g: &Graph) -> usize {
        min_fill_order(g.n, &adjacency(g)).1
    }

    fn run(&self, g: &Graph, beta: f64, min_sum: bool) -> Result<Exact, TooWide> {
        let (order, width) = min_fill_order(g.n, &adjacency(g));
        if width > self.max_width {
            return Err(TooWide::Width { width, max: self.max_width });
        }

        let mut tables = initial_tables(g, beta);
        // For back-substitution: for each eliminated variable, the scope it depended on and the
        // choice that was optimal for every assignment of that scope.
        let mut decisions: Vec<(usize, Vec<usize>, Vec<bool>)> = Vec::new();
        let mut constant = 0.0f64;

        for &v in &order {
            let (mine, rest): (Vec<Table>, Vec<Table>) =
                tables.into_iter().partition(|t| t.vars.contains(&v));
            tables = rest;
            if mine.is_empty() {
                continue;
            }

            // scope of the new table: everything the gathered tables touch, minus v
            let mut scope: Vec<usize> = Vec::new();
            for t in &mine {
                for &u in &t.vars {
                    if u != v && !scope.contains(&u) {
                        scope.push(u);
                    }
                }
            }
            scope.sort_unstable();

            let m = scope.len();
            let mut vals = vec![0.0f64; 1 << m];
            let mut choice = vec![false; 1 << m];
            let mut assign = vec![0i8; g.n];

            for idx in 0..(1usize << m) {
                for (k, &u) in scope.iter().enumerate() {
                    assign[u] = if idx >> k & 1 == 1 { 1 } else { -1 };
                }
                // the two branches for v
                let mut branch = [0.0f64; 2];
                for (bi, sv) in [(-1i8, 0usize), (1i8, 1usize)].map(|(s, i)| (s, i)) {
                    assign[v] = bi;
                    branch[sv] = mine.iter().map(|t| t.value_at(&assign)).sum();
                }
                if min_sum {
                    let take_plus = branch[1] < branch[0];
                    vals[idx] = if take_plus { branch[1] } else { branch[0] };
                    choice[idx] = take_plus;
                } else {
                    // log-sum-exp of -energy, stably
                    let (a, b) = (-branch[0], -branch[1]);
                    let hi = a.max(b);
                    vals[idx] = -(hi + ((a - hi).exp() + (b - hi).exp()).ln());
                }
            }

            decisions.push((v, scope.clone(), choice));
            if m == 0 {
                constant += vals[0];
            } else {
                tables.push(Table { vars: scope, vals });
            }
        }

        for t in &tables {
            debug_assert!(t.vars.is_empty(), "a table survived elimination");
            constant += t.vals[0];
        }

        if min_sum {
            // Walk the decisions backwards, filling in each variable from the scope already fixed.
            let mut state = vec![-1i8; g.n];
            for (v, scope, choice) in decisions.iter().rev() {
                let mut idx = 0usize;
                for (k, &u) in scope.iter().enumerate() {
                    if state[u] > 0 {
                        idx |= 1 << k;
                    }
                }
                state[*v] = if choice[idx] { 1 } else { -1 };
            }
            Ok(Exact {
                width,
                ground_energy: Some(constant),
                ground_state: Some(state),
                log_z: None,
            })
        } else {
            Ok(Exact { width, ground_energy: None, ground_state: None, log_z: Some(-constant) })
        }
    }
}

/// The graph with node `i` pinned to `v`: its couplings removed and folded into its neighbours'
/// fields, and its own field zeroed so it contributes an identical constant factor whatever `v` is.
///
/// The node is kept rather than deleted so every other index is unchanged — renumbering would make
/// the returned marginals line up with a different graph than the one asked about, which is the
/// kind of error that produces a plausible answer.
fn pin(g: &Graph, i: usize, v: f64) -> Graph {
    let mut b = crate::graph::GraphBuilder::new(g.n);
    for a in 0..g.n {
        let mut h = if a == i { 0.0 } else { g.h[a] };
        for k in g.offset[a]..g.offset[a + 1] {
            let c = g.nbr[k] as usize;
            if a == i || c == i {
                // An edge touching the pinned node becomes a field on the other end.
                if a != i {
                    h += g.w[k] * v;
                }
            } else if c > a {
                b.couple(a, c, g.w[k]);
            }
        }
        if h != 0.0 {
            b.bias(a, h);
        }
    }
    b.build()
}

#[cfg(test)]
mod tests {

    /// Marginals against BRUTE FORCE, which is the only referee that leaves nothing to argue about.
    ///
    /// Enumerating 2^n states and summing the Boltzmann weights is a completely different
    /// computation from eliminating variables, so agreement to 1e-12 is not two implementations of
    /// one idea agreeing with themselves.
    #[test]
    fn marginals_match_exhaustive_enumeration() {
        for (seed, n) in [(1u64, 6usize), (7, 8), (99, 10)] {
            let mut rng = crate::rng::Pcg::new(seed, 0xE7AC);
            let mut b = GraphBuilder::new(n);
            for i in 0..n {
                b.bias(i, rng.f64() * 2.0 - 1.0);
                for j in (i + 1)..n {
                    if rng.f64() < 0.45 {
                        b.couple(i, j, rng.f64() * 2.0 - 1.0);
                    }
                }
            }
            let g = b.build();
            let beta = 0.8;

            let got = Elimination::default().marginals(&g, beta).expect("narrow enough");

            // Brute force: sum exp(-beta E) over every state, and over the states with s_i = +1.
            let mut z = 0.0f64;
            let mut zi = vec![0.0f64; n];
            for mask in 0..(1u32 << n) {
                let s: Vec<i8> =
                    (0..n).map(|i| if mask >> i & 1 == 1 { 1i8 } else { -1 }).collect();
                let wgt = (-beta * g.energy(&s)).exp();
                z += wgt;
                for i in 0..n {
                    if s[i] == 1 {
                        zi[i] += wgt;
                    }
                }
            }
            for i in 0..n {
                let want = zi[i] / z;
                assert!(
                    (got[i] - want).abs() < 1e-12,
                    "seed {seed}, n {n}, node {i}: elimination {} vs enumeration {want}",
                    got[i]
                );
            }
        }
    }

    /// The one case with a closed form, so a systematic error in BOTH of the above would show.
    #[test]
    fn a_single_spin_in_a_field_matches_the_sigmoid() {
        for h in [-1.5, -0.3, 0.0, 0.7, 2.0] {
            for beta in [0.1, 1.0, 3.0] {
                let mut b = GraphBuilder::new(1);
                b.bias(0, h);
                let g = b.build();
                let got = Elimination::default().marginals(&g, beta).unwrap()[0];
                // P(+1) = e^{beta h} / (e^{beta h} + e^{-beta h}) = sigma(2 beta h)
                let want = 1.0 / (1.0 + (-2.0 * beta * h).exp());
                assert!((got - want).abs() < 1e-13, "h {h}, beta {beta}: {got} vs {want}");
            }
        }
    }

    /// Marginals are the referee this module exists to be, so check a SAMPLER against them on a
    /// graph far past where enumeration stops -- which is the sentence in the module doc that had
    /// no code behind it.
    #[test]
    fn a_sampler_can_be_checked_against_truth_past_where_enumeration_stops() {
        // A 3x14 strip: 42 spins, so 2^42 states -- unenumerable -- and width 3.
        let (w, l) = (3usize, 14usize);
        let mut b = GraphBuilder::new(w * l);
        for y in 0..l {
            for x in 0..w {
                let i = y * w + x;
                if x + 1 < w {
                    b.couple(i, i + 1, 0.6);
                }
                if y + 1 < l {
                    b.couple(i, i + w, 0.6);
                }
            }
        }
        let g = b.build();
        let beta = 0.35;
        let e = Elimination::default();
        assert!(e.width(&g) <= 4, "a strip is narrow: width {}", e.width(&g));
        let truth = e.marginals(&g, beta).unwrap();

        let mut smp = crate::gibbs::Sampler::new(&g, beta, 0xC0FFEE);
        smp.sweeps(2000, None);
        let draws = 40_000;
        let mut up = vec![0u64; g.n];
        for _ in 0..draws {
            smp.sweep(None);
            for i in 0..g.n {
                if smp.s[i] == 1 {
                    up[i] += 1;
                }
            }
        }
        let worst = (0..g.n)
            .map(|i| (up[i] as f64 / draws as f64 - truth[i]).abs())
            .fold(0.0f64, f64::max);
        // Three sigma on 40k correlated draws is comfortably inside this; the point is that the
        // comparison is possible at all on 42 spins.
        assert!(worst < 0.02, "worst |sampled - exact| marginal = {worst:.4}");
    }

    #[test]
    fn a_graph_too_wide_for_marginals_is_refused_with_the_same_reason_as_log_z() {
        // Dense on 30 nodes: width far past the default ceiling.
        let mut b = GraphBuilder::new(30);
        for i in 0..30 {
            for j in (i + 1)..30 {
                b.couple(i, j, 0.4);
            }
        }
        let g = b.build();
        let e = Elimination::default();
        let m = e.marginals(&g, 1.0);
        assert!(matches!(m, Err(TooWide::Width { .. })), "{m:?}");
        // And it refuses BEFORE paying for 2n eliminations, which is why the width is checked up
        // front rather than being discovered inside the loop.
        // Refused for the SAME reason and from the same order: a caller that got a log Z cannot
        // then be told its marginals are too wide, because conditioning only removes edges.
        assert_eq!(m.unwrap_err(), e.log_partition(&g, 1.0).unwrap_err());
    }

    use super::*;
    use crate::graph::GraphBuilder;
    use crate::oracle::{Exhaustive, Solver};
    use crate::rng::Pcg;

    fn random_sparse(n: usize, p: f64, seed: u64) -> Graph {
        let mut rng = Pcg::new(seed, 0);
        let mut b = GraphBuilder::new(n);
        for i in 0..n {
            for j in (i + 1)..n {
                if rng.f64() < p {
                    b.couple(i, j, rng.f64() * 2.0 - 1.0);
                }
            }
            b.bias(i, rng.f64() - 0.5);
        }
        b.build()
    }

    #[test]
    fn the_ground_state_matches_enumeration() {
        // The only check that matters for an exact method.
        for (n, p, seed) in [(10, 0.3, 1), (14, 0.2, 2), (16, 0.15, 3), (12, 0.5, 4)] {
            let g = random_sparse(n, p, seed);
            let (bs, be) = Exhaustive.solve(&g);
            let e = Elimination::default().ground_state(&g).expect("small enough");
            let ge = e.ground_energy.unwrap();
            assert!(
                (ge - be).abs() < 1e-9,
                "n={n} p={p}: elimination {ge} vs enumeration {be}"
            );
            // and the recovered state really attains it, which back-substitution can get wrong
            // independently of the energy being right
            let st = e.ground_state.unwrap();
            assert!(
                (g.energy(&st) - be).abs() < 1e-9,
                "n={n}: recovered state has energy {} not {be} (enumeration found {bs:?})",
                g.energy(&st)
            );
        }
    }

    #[test]
    fn log_z_matches_enumeration() {
        for (n, p, beta, seed) in [(10, 0.3, 0.7, 1), (12, 0.25, 1.3, 2), (8, 0.6, 0.4, 3)] {
            let g = random_sparse(n, p, seed);
            let mut z = 0.0f64;
            let mut s = vec![-1i8; n];
            for mask in 0..(1usize << n) {
                for i in 0..n {
                    s[i] = if mask >> i & 1 == 1 { 1 } else { -1 };
                }
                z += (-beta * g.energy(&s)).exp();
            }
            let want = z.ln();
            let got = Elimination::default().log_partition(&g, beta).unwrap().log_z.unwrap();
            assert!((got - want).abs() < 1e-9, "n={n} beta={beta}: {got} vs {want}");
        }
    }

    #[test]
    fn a_chain_is_width_one_and_exact_at_any_length() {
        // The point of the method: sparse structure beats size. Enumeration cannot touch this.
        let n = 2000;
        let mut b = GraphBuilder::new(n);
        for i in 0..n - 1 {
            b.couple(i, i + 1, 1.0);
        }
        let g = b.build();
        let el = Elimination::default();
        assert_eq!(el.width(&g), 1, "a path has induced width 1");
        let e = el.ground_state(&g).unwrap();
        assert_eq!(e.ground_energy.unwrap(), -((n - 1) as f64), "every bond satisfiable");
        assert!(e.ground_state.unwrap().windows(2).all(|w| w[0] == w[1]));
    }

    #[test]
    fn a_lattice_strip_is_exact_far_past_enumeration() {
        // 6 x 40 = 240 spins, width 6. Enumeration would need 2^240 states.
        let (w, h) = (6usize, 40usize);
        let mut b = GraphBuilder::new(w * h);
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                if x + 1 < w {
                    b.couple(i, y * w + x + 1, 1.0);
                }
                if y + 1 < h {
                    b.couple(i, (y + 1) * w + x, 1.0);
                }
            }
        }
        let g = b.build();
        let el = Elimination { max_width: 12 };
        // The true treewidth of a 6-wide strip is 6; min-fill finds 8 here. Asserting 6 would be
        // asserting that a heuristic is optimal, which it is not past width 5 -- see the table in
        // the module docs. The bound is on measured behaviour, and a regression past it means the
        // ordering got worse, not that the answer got wrong.
        assert!(el.width(&g) <= 8, "min-fill measured 8 on this strip; got {}", el.width(&g));
        let e = el.ground_state(&g).unwrap();
        let bonds = (w - 1) * h + w * (h - 1);
        assert_eq!(e.ground_energy.unwrap(), -(bonds as f64));
    }

    #[test]
    fn a_dense_graph_is_refused_rather_than_attempted() {
        // Refusing loudly beats running for a week. The message must say what to do instead.
        let g = random_sparse(60, 0.9, 1);
        let err = Elimination { max_width: 20 }.ground_state(&g).unwrap_err();
        assert!(matches!(err, TooWide::Width { .. }));
        assert!(err.to_string().contains("planted instance"), "{err}");
    }

    #[test]
    fn it_agrees_with_a_planted_wishart_optimum_where_width_allows() {
        // Two independent notions of truth, cross-checked.
        let p = crate::planted::frustrated_loops(4, 12, 3);
        let e = Elimination { max_width: 20 }.ground_state(&p.graph).unwrap();
        assert!((e.ground_energy.unwrap() - p.ground_energy).abs() < 1e-9);
    }
}

#[cfg(test)]
mod closed_form {
    use super::*;
    use crate::graph::GraphBuilder;

    #[test]
    fn log_z_of_a_chain_matches_the_closed_form() {
        // A 1D open Ising chain has Z = 2 (2 cosh beta)^(n-1) exactly. Checking against theory
        // rather than against enumeration reaches sizes enumeration cannot, and catches an error
        // that would scale with n instead of showing up on small cases.
        for n in [8usize, 50, 400] {
            for beta in [0.25f64, 0.5, 1.5] {
                let mut b = GraphBuilder::new(n);
                for i in 0..n - 1 {
                    b.couple(i, i + 1, 1.0);
                }
                let got = Elimination::default().log_partition(&b.build(), beta).unwrap().log_z.unwrap();
                let want = 2f64.ln() + (n - 1) as f64 * (2.0 * beta.cosh()).ln();
                assert!(
                    (got - want).abs() < 1e-9 * want.abs().max(1.0),
                    "n={n} beta={beta}: {got} vs closed form {want}"
                );
            }
        }
    }
}
