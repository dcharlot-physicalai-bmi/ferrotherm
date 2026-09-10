//! Tree-reweighted belief propagation: a deterministic **upper** bound on `ln Z`.
//!
//! Wainwright, Jaakkola & Willsky, IEEE Trans. Inf. Theory 51:2313, 2005.
//!
//! `ln Z` is convex in the model parameters, so for any convex combination of spanning forests
//! `θ = Σ_t ρ_t θ^t`, Jensen's inequality gives
//!
//! ```text
//!   ln Z(θ)  <=  Σ_t ρ_t ln Z(θ^t)
//! ```
//!
//! and every `ln Z(θ^t)` is EXACT, because a forest has induced width 1. Handing each edge to the
//! forests that carry it, scaled by its appearance probability `ρ_e = Σ_{t ∋ e} ρ_t`, gives
//! [`tree_decomposition_bound`] — a valid upper bound with no iteration and no fixed point, only
//! [`crate::exact::Elimination`] run once per forest.
//!
//! [`trw`] passes messages to find the TIGHTEST split at the same `ρ`, and returns the
//! tree-reweighted free energy at its fixed point. Weak duality makes it no worse than the
//! iteration-free bound, which is an identity the tests check rather than a claim.
//!
//! This is the bound this crate did not have. [`crate::meanfield::gibbs_bogoliubov`] is a
//! deterministic LOWER bound, [`crate::free_energy`] has stochastic bounds and estimates, and the
//! Bethe free energy of [`crate::meanfield::belief_propagation`] is neither: it can sit on either
//! side of the truth. With mean field below, `ln Z` is now bracketed deterministically.
//!
//! # The messages
//!
//! Loopy belief propagation's cavity-field recursion with two changes — the coupling is divided by
//! the edge's appearance probability, and incoming messages enter the cavity field weighted by
//! their own:
//!
//! ```text
//!   V_{j∖i} = β h_j + Σ_{k∈∂j} ρ_kj v_{k→j} − v_{i→j}
//!   v_{j→i} = atanh( tanh(β w_ij / ρ_ij) · tanh(V_{j∖i}) )
//! ```
//!
//! (all fields scaled by β, so `β = 0` needs no special case). A forest admits exactly one convex
//! combination of spanning forests, `ρ ≡ 1`, at which these ARE the belief-propagation messages and
//! the bound is exact — which is the oracle the tests use.
//!
//! # What makes the bound sound
//!
//! `ρ` must lie in the spanning-tree polytope: it must be the edge-appearance vector of some real
//! convex combination of spanning forests. [`TreeCover`] carries that combination explicitly
//! instead of taking `ρ` on faith, so the hypothesis of the theorem is a checkable object.

use crate::exact::Elimination;
use crate::graph::{Graph, GraphBuilder};
use crate::rng::Pcg;
use crate::round::sum_up;

/// A convex combination of spanning forests, and the edge appearance probabilities it induces.
///
/// Built by [`TreeCover::random`]. The forests are the witness that `ρ` lies in the spanning-tree
/// polytope, which is the hypothesis every bound in this module rests on.
#[derive(Clone, Debug)]
pub struct TreeCover {
    /// Undirected edges as `(i, j)` with `i < j`, in CSR scan order; length is `g.n_edges`.
    pub edges: Vec<(usize, usize)>,
    /// The two directed CSR slots of each edge, `(slot i→j, slot j→i)`, parallel to `edges`.
    pub slots: Vec<(usize, usize)>,
    /// Convex weight `ρ_t` of each forest; strictly positive and summing to 1.
    pub weight: Vec<f64>,
    /// `member[t][e]` is whether forest `t` contains edge `e`.
    pub member: Vec<Vec<bool>>,
    /// Appearance probability `ρ_e = Σ_{t ∋ e} ρ_t` of each edge, in `(0, 1]`; never zero, since a
    /// zero would send the reweighted coupling `w_e / ρ_e` to infinity.
    pub rho: Vec<f64>,
}

impl TreeCover {
    /// `forests` random spanning forests of `g`, plus as many more as it takes for every edge to
    /// appear in at least one, all with equal weight.
    ///
    /// Each forest is a randomised Kruskal pass that visits least-used edges first, so the
    /// uncovered edges are offered before any other and the uncovered set strictly shrinks every
    /// pass: at most `g.n_edges` extra forests are ever added.
    ///
    /// # Panics
    ///
    /// If `g`'s adjacency is not symmetric, which would mean it was not built by [`GraphBuilder`].
    #[must_use]
    pub fn random(g: &Graph, forests: usize, seed: u64) -> TreeCover {
        let (edges, slots) = undirected(g);
        let ne = edges.len();
        let mut rng = Pcg::new(seed, 0);
        let mut uses = vec![0usize; ne];
        let mut member: Vec<Vec<bool>> = Vec::new();
        let target = forests.max(1);
        while member.len() < target || uses.contains(&0) {
            // Least-used first, ties broken at random: the key sits in [uses, uses + 1).
            let mut order: Vec<(f64, usize)> =
                (0..ne).map(|e| (uses[e] as f64 + rng.f64(), e)).collect();
            order.sort_by(|a, b| a.0.total_cmp(&b.0));
            let mut dsu = Dsu::new(g.n);
            let mut mem = vec![false; ne];
            for &(_, e) in &order {
                let (i, j) = edges[e];
                if dsu.union(i, j) {
                    mem[e] = true;
                    uses[e] += 1;
                }
            }
            member.push(mem);
        }
        let t = member.len() as f64;
        let weight = vec![1.0 / t; member.len()];
        let rho = (0..ne).map(|e| uses[e] as f64 / t).collect();
        TreeCover { edges, slots, weight, member, rho }
    }

    /// Coupling of edge `e`, read off the graph the cover was built from.
    ///
    /// # Panics
    ///
    /// If `g` is not that graph, so the slot index is out of range.
    #[must_use]
    pub fn coupling(&self, g: &Graph, e: usize) -> f64 {
        g.w[self.slots[e].0]
    }
}

/// What a tree-reweighted message-passing run produced.
#[derive(Clone, Debug)]
pub struct Trw {
    /// Inverse temperature the messages were passed at.
    pub beta: f64,
    /// Tree-reweighted pseudomarginals `⟨s_i⟩`.
    pub m: Vec<f64>,
    /// The tree-reweighted free energy at the final pseudomarginals: an **upper** bound on `ln Z`
    /// once the messages have converged and `consistency` is small.
    pub log_z: f64,
    /// Largest message change on the last iteration.
    pub residual: f64,
    /// Iterations actually run, which is the cap when it did not converge.
    pub iterations: usize,
    /// Largest disagreement between an edge pseudomarginal and the node pseudomarginal it should
    /// marginalise to. The bound is a theorem on the local polytope, and this is the distance from
    /// it: zero at a fixed point, and worth reading before trusting `log_z` at an iteration cap.
    pub consistency: f64,
}

impl Trw {
    /// Whether the last iteration moved every message by less than `tol`.
    #[must_use]
    pub fn converged(&self, tol: f64) -> bool {
        self.residual < tol
    }
}

/// Tree-reweighted belief propagation at edge appearance probabilities `cover.rho`, damped, from
/// zero messages; returns the tree-reweighted free energy as an upper bound on `ln Z(β)`.
///
/// # Panics
///
/// If `cover` was not built from `g`, or `damping` is not in `[0, 1)`.
#[must_use]
pub fn trw(g: &Graph, beta: f64, cover: &TreeCover, iters: usize, damping: f64) -> Trw {
    assert_eq!(cover.edges.len(), g.n_edges, "cover was built from another graph");
    assert!((0.0..1.0).contains(&damping), "damping must be in [0, 1)");
    let nd = g.nbr.len();
    let mut rev = vec![0usize; nd];
    let mut rho = vec![1.0f64; nd];
    for e in 0..cover.edges.len() {
        let (a, b) = cover.slots[e];
        rev[a] = b;
        rev[b] = a;
        rho[a] = cover.rho[e];
        rho[b] = cover.rho[e];
    }
    // Scaled parameters: every field below is already multiplied by beta.
    let hb: Vec<f64> = g.h.iter().map(|h| beta * h).collect();
    let ab: Vec<f64> = g.w.iter().map(|w| beta * w).collect();
    let mut v = vec![0.0f64; nd];

    // Full reweighted field at j: beta h_j + sum_k rho_kj v_{k->j}. The cavity field for the
    // message j -> i subtracts the FULL v_{i->j}, not its rho-weighted share: the (1 - rho) power
    // on the reverse message in Wainwright et al. eq. 39 and the rho inside the sum add to one.
    let full = |v: &[f64], j: usize| -> f64 {
        let mut f = hb[j];
        for s in g.offset[j]..g.offset[j + 1] {
            f += rho[s] * v[s];
        }
        f
    };

    let mut residual = f64::INFINITY;
    let mut it = 0;
    while it < iters && residual > 1e-14 {
        residual = 0.0;
        for e in 0..nd {
            let j = g.nbr[e] as usize; // the message j -> owner(e)
            let vj = full(&v, j) - v[rev[e]];
            let a = ab[e] / rho[e];
            let new = 0.5 * (ln_cosh(a + vj) - ln_cosh(a - vj));
            let next = damping * v[e] + (1.0 - damping) * new;
            residual = residual.max((next - v[e]).abs());
            v[e] = next;
        }
        it += 1;
    }

    // Node pseudomarginals, and the singleton half of the free energy.
    let mut m = vec![0.0; g.n];
    let mut energy = 0.0;
    let mut entropy = 0.0;
    for i in 0..g.n {
        m[i] = full(&v, i).tanh();
        energy += hb[i] * m[i];
        entropy += binary_entropy(m[i]);
    }
    // Edge pseudomarginals, their energy, and the mutual information the tree-reweighted entropy
    // subtracts with weight rho.
    let mut consistency = 0.0f64;
    for e in 0..cover.edges.len() {
        let (i, j) = cover.edges[e];
        let (sij, sji) = cover.slots[e];
        let hi = full(&v, i) - v[sij];
        let hj = full(&v, j) - v[sji];
        let a = ab[sij];
        let b = pair_belief(a / cover.rho[e], hi, hj);
        // index k = 2 * x + y with 0 meaning spin -1
        energy += a * (b[0] - b[1] - b[2] + b[3]);
        let bi = [b[0] + b[1], b[2] + b[3]];
        let bj = [b[0] + b[2], b[1] + b[3]];
        entropy -= cover.rho[e] * (entropy_of(&bi) + entropy_of(&bj) - entropy_of(&b));
        consistency = consistency.max((bi[1] - (1.0 + m[i]) / 2.0).abs());
        consistency = consistency.max((bj[1] - (1.0 + m[j]) / 2.0).abs());
    }
    Trw { beta, m, log_z: energy + entropy, residual, iterations: it, consistency }
}

/// The iteration-free bound: `Σ_t ρ_t ln Z(θ^t)` at the canonical split `w_e → w_e / ρ_e`, with
/// each forest's `ln Z` computed exactly by [`Elimination`].
///
/// Valid by convexity alone — no fixed point, no convergence, nothing to check — and summed with
/// [`sum_up`] so the returned float is not below the exact sum of its terms.
///
/// # Panics
///
/// If `cover` was not built from `g`, or if elimination refuses a forest, which it cannot: an
/// acyclic graph has induced width 1.
#[must_use]
pub fn tree_decomposition_bound(g: &Graph, beta: f64, cover: &TreeCover) -> f64 {
    assert_eq!(cover.edges.len(), g.n_edges, "cover was built from another graph");
    let mut terms = Vec::with_capacity(cover.weight.len());
    for (t, mem) in cover.member.iter().enumerate() {
        let mut gb = GraphBuilder::new(g.n);
        for i in 0..g.n {
            gb.bias(i, g.h[i]);
        }
        for e in 0..cover.edges.len() {
            if mem[e] {
                let (i, j) = cover.edges[e];
                gb.couple(i, j, cover.coupling(g, e) / cover.rho[e]);
            }
        }
        let forest = gb.build();
        let lz = Elimination::default()
            .log_partition(&forest, beta)
            .expect("a forest has induced width 1")
            .log_z
            .expect("sum-product was run");
        terms.push(cover.weight[t] * lz);
    }
    sum_up(&terms)
}

/// The bound anything must beat to be worth computing: `n ln 2 + β(Σ|w| + Σ|h|)`.
///
/// Every state's weight is at most `exp(β(Σ|w| + Σ|h|))` and there are `2^n` of them.
#[must_use]
pub fn trivial_upper_bound(g: &Graph, beta: f64) -> f64 {
    let w: f64 = g.w.iter().map(|x| x.abs()).sum::<f64>() / 2.0;
    let h: f64 = g.h.iter().map(|x| x.abs()).sum();
    g.n as f64 * std::f64::consts::LN_2 + beta.abs() * (w + h)
}

/// One pair per undirected edge: endpoints in one list, CSR slots in the other.
type Pairs = Vec<(usize, usize)>;

fn undirected(g: &Graph) -> (Pairs, Pairs) {
    let mut edges = Vec::with_capacity(g.n_edges);
    let mut slots = Vec::with_capacity(g.n_edges);
    for i in 0..g.n {
        for e in g.offset[i]..g.offset[i + 1] {
            let j = g.nbr[e] as usize;
            if j > i {
                let f = (g.offset[j]..g.offset[j + 1])
                    .find(|&f| g.nbr[f] as usize == i)
                    .expect("symmetric adjacency");
                edges.push((i, j));
                slots.push((e, f));
            }
        }
    }
    (edges, slots)
}

/// `ln cosh`, written so a large argument neither overflows nor loses the small term.
fn ln_cosh(x: f64) -> f64 {
    let a = x.abs();
    a + (-2.0 * a).exp().ln_1p() - std::f64::consts::LN_2
}

fn binary_entropy(m: f64) -> f64 {
    entropy_of(&[(1.0 - m) / 2.0, (1.0 + m) / 2.0])
}

fn entropy_of(p: &[f64]) -> f64 {
    -p.iter().map(|&x| if x > 0.0 { x * x.ln() } else { 0.0 }).sum::<f64>()
}

/// Normalised pair belief `∝ exp(a x y + hi x + hj y)`, indexed `2 * x + y` with 0 meaning `-1`.
fn pair_belief(a: f64, hi: f64, hj: f64) -> [f64; 4] {
    let sp = [-1.0f64, 1.0];
    let mut l = [0.0f64; 4];
    for xi in 0..2 {
        for yi in 0..2 {
            l[2 * xi + yi] = a * sp[xi] * sp[yi] + hi * sp[xi] + hj * sp[yi];
        }
    }
    let mx = l.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut b = [0.0f64; 4];
    let mut z = 0.0;
    for k in 0..4 {
        b[k] = (l[k] - mx).exp();
        z += b[k];
    }
    for x in &mut b {
        *x /= z;
    }
    b
}

struct Dsu {
    p: Vec<usize>,
}

impl Dsu {
    fn new(n: usize) -> Self {
        Dsu { p: (0..n).collect() }
    }
    fn find(&mut self, x: usize) -> usize {
        let mut r = x;
        while self.p[r] != r {
            r = self.p[r];
        }
        let mut c = x;
        while self.p[c] != c {
            let next = self.p[c];
            self.p[c] = r;
            c = next;
        }
        r
    }
    fn union(&mut self, a: usize, b: usize) -> bool {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra == rb {
            return false;
        }
        self.p[ra] = rb;
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ising;
    use crate::meanfield::naive_mean_field;

    fn random_tree(n: usize, seed: u64) -> Graph {
        let mut rng = Pcg::new(seed, 7);
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

    fn random_loopy(n: usize, extra: usize, seed: u64) -> Graph {
        let mut rng = Pcg::new(seed, 9);
        let mut gb = GraphBuilder::new(n);
        for i in 1..n {
            let parent = ((rng.f64() * i as f64) as usize).min(i - 1);
            gb.couple(parent, i, 1.6 * rng.f64() - 0.8);
        }
        for _ in 0..extra {
            let a = ((rng.f64() * n as f64) as usize).min(n - 1);
            let b = ((rng.f64() * n as f64) as usize).min(n - 1);
            if a != b {
                gb.couple(a, b, 1.6 * rng.f64() - 0.8);
            }
        }
        for i in 0..n {
            gb.bias(i, 0.6 * rng.f64() - 0.3);
        }
        gb.build()
    }

    fn exact(g: &Graph, beta: f64) -> f64 {
        Elimination::default().log_partition(g, beta).unwrap().log_z.unwrap()
    }

    /// The cover is a real convex combination of spanning forests: weights sum to one, every
    /// membership is acyclic and maximally so, and every edge appears.
    ///
    /// This is the hypothesis of the theorem, not a detail. A `rho` outside the spanning-tree
    /// polytope makes the bound false rather than loose, and "I chose rho sensibly" is exactly the
    /// kind of claim that decays; the forests are kept so it can be checked.
    #[test]
    fn cover_is_a_convex_combination_of_spanning_forests() {
        for (g, k) in [
            (ising::lattice2d(4, 1.0), 4),
            (random_loopy(14, 10, 3), 6),
            (random_tree(12, 4), 3),
            // one requested forest cannot cover a loopy graph, so the repair passes must run
            (random_loopy(16, 14, 8), 1),
        ] {
            let c = TreeCover::random(&g, k, 11);
            assert_eq!(c.edges.len(), g.n_edges);
            assert!(c.member.len() >= k, "asked for {k} forests, got {}", c.member.len());
            let total: f64 = c.weight.iter().sum();
            assert!((total - 1.0).abs() < 1e-12, "weights sum to {total}");
            for t in 0..c.member.len() {
                let mut dsu = Dsu::new(g.n);
                for e in 0..c.edges.len() {
                    if c.member[t][e] {
                        let (i, j) = c.edges[e];
                        assert!(dsu.union(i, j), "forest {t} has a cycle at edge {e}");
                    }
                }
                // maximal: no absent edge could have been added without closing a cycle
                for e in 0..c.edges.len() {
                    if !c.member[t][e] {
                        let (i, j) = c.edges[e];
                        assert_eq!(dsu.find(i), dsu.find(j), "forest {t} is not spanning at {e}");
                    }
                }
            }
            for e in 0..c.edges.len() {
                let want: f64 =
                    (0..c.member.len()).filter(|&t| c.member[t][e]).map(|t| c.weight[t]).sum();
                assert!((c.rho[e] - want).abs() < 1e-12);
                assert!(c.rho[e] > 0.0 && c.rho[e] <= 1.0 + 1e-15, "rho {} at {e}", c.rho[e]);
            }
        }
    }

    /// A forest admits one convex combination, `rho == 1`, and there the bound is EXACT.
    ///
    /// Checked against variable elimination on `ln Z` and on every marginal. This is the oracle
    /// that says the message algebra is right rather than merely large: a bound that is above
    /// `ln Z` because it is wrong passes an inequality test and fails this one.
    #[test]
    fn exact_on_a_forest() {
        for seed in 0..5u64 {
            let g = random_tree(18, seed);
            let beta = 0.9;
            let c = TreeCover::random(&g, 3, seed + 100);
            assert!(c.rho.iter().all(|&r| (r - 1.0).abs() < 1e-15), "a forest has rho = 1");
            let t = trw(&g, beta, &c, 500, 0.0);
            let truth = exact(&g, beta);
            assert!(t.converged(1e-12), "seed {seed}: residual {}", t.residual);
            assert!((t.log_z - truth).abs() < 1e-9, "seed {seed}: TRW {} vs {truth}", t.log_z);
            assert!(t.consistency < 1e-9, "seed {seed}: consistency {}", t.consistency);
            let marg = Elimination::default().marginals(&g, beta).unwrap();
            for i in 0..g.n {
                let want = 2.0 * marg[i] - 1.0;
                assert!((t.m[i] - want).abs() < 1e-9, "site {i}: {} vs {want}", t.m[i]);
            }
            // and the iteration-free route agrees, since the single forest IS the model
            let j = tree_decomposition_bound(&g, beta, &c);
            assert!((j - truth).abs() < 1e-9, "Jensen {j} vs {truth}");
        }
    }

    /// The whole claim: on loopy models the bound sits ABOVE exact `ln Z`, by both routes.
    ///
    /// The `consistency` assertion is not decoration. Mutating the cavity field to subtract the
    /// reweighted reverse message instead of the whole one leaves a fixed point whose value is
    /// still above `ln Z` on every instance here — the inequality passes and only the distance from
    /// the local polytope shows the message rule is wrong.
    #[test]
    fn bounds_exact_log_z_from_above() {
        let cases: Vec<(Graph, f64)> = vec![
            (ising::lattice2d(4, 1.0), 0.2),
            (ising::lattice2d(4, 1.0), 0.44),
            (ising::grid2d(4, 5, -1.0), 0.35),
            (ising::ring(9, -1.0, 0.15), 1.1),
            (random_loopy(16, 12, 1), 0.8),
            (random_loopy(16, 12, 2), 1.4),
            (random_loopy(20, 25, 3), 0.6),
            (random_loopy(20, 30, 4), 1.0),
            (random_loopy(12, 20, 5), 1.7),
        ];
        for (g, beta) in &cases {
            let truth = exact(g, *beta);
            for seed in 0..3u64 {
                let c = TreeCover::random(g, 8, seed);
                let t = trw(g, *beta, &c, 20_000, 0.5);
                assert!(t.converged(1e-11), "residual {} n={} beta={beta}", t.residual, g.n);
                assert!(t.consistency < 1e-8, "consistency {}", t.consistency);
                assert!(
                    t.log_z >= truth - 1e-9,
                    "TRW {} BELOW exact {truth} (n={}, beta={beta}, seed={seed})",
                    t.log_z,
                    g.n
                );
                let j = tree_decomposition_bound(g, *beta, &c);
                assert!(j >= truth - 1e-9, "Jensen {j} below exact {truth}");
            }
        }
    }

    /// Tighter than the bound anyone can write down without running anything.
    #[test]
    fn tighter_than_the_trivial_bound() {
        for (g, beta) in [
            (ising::lattice2d(4, 1.0), 0.44),
            (ising::grid2d(4, 5, -1.0), 0.35),
            (random_loopy(20, 25, 3), 0.6),
        ] {
            let truth = exact(&g, beta);
            let triv = trivial_upper_bound(&g, beta);
            let c = TreeCover::random(&g, 8, 2);
            let t = trw(&g, beta, &c, 20_000, 0.5);
            assert!(t.converged(1e-11));
            assert!(t.log_z < triv, "TRW {} not below trivial {triv}", t.log_z);
            // and by a margin worth having: at least half the trivial bound's slack is closed
            let closed = (triv - t.log_z) / (triv - truth);
            assert!(closed > 0.5, "closed only {closed} of the trivial slack");
        }
    }

    /// Weak duality: message passing is no worse than the canonical split it optimises over.
    ///
    /// Both are upper bounds at the same `rho`; the tree-reweighted free energy is the minimum over
    /// splits, so it cannot exceed the one split written down by hand. An identity between two
    /// quantities this module computes by completely different machinery — messages versus
    /// variable elimination on each forest.
    #[test]
    fn message_passing_is_no_worse_than_the_canonical_split() {
        for (g, beta) in [
            (ising::lattice2d(4, 1.0), 0.3),
            (random_loopy(16, 12, 1), 0.8),
            (random_loopy(20, 25, 3), 0.6),
        ] {
            let c = TreeCover::random(&g, 8, 5);
            let t = trw(&g, beta, &c, 20_000, 0.5);
            assert!(t.converged(1e-11));
            let j = tree_decomposition_bound(&g, beta, &c);
            assert!(t.log_z <= j + 1e-7, "TRW {} above canonical split {j}", t.log_z);
        }
    }

    /// With mean field below, `ln Z` is now bracketed deterministically from both sides.
    #[test]
    fn brackets_exact_log_z_with_mean_field() {
        for (g, beta) in [(ising::lattice2d(4, 1.0), 0.3), (random_loopy(18, 20, 6), 0.7)] {
            let truth = exact(&g, beta);
            let lo = naive_mean_field(&g, beta, 5000, 0.5).log_z;
            let c = TreeCover::random(&g, 8, 8);
            let hi = trw(&g, beta, &c, 20_000, 0.5).log_z;
            assert!(lo <= truth + 1e-12 && truth <= hi + 1e-9, "{lo} <= {truth} <= {hi}");
            assert!(hi - lo < 3.0, "bracket width {}", hi - lo);
        }
    }

    /// Closed forms: at `beta = 0` every model has `ln Z = n ln 2`, and a two-spin model is a tree
    /// whose `ln Z` is `ln(2(e^J cosh(h1+h2) + e^-J cosh(h1-h2)))` at `beta = 1`.
    #[test]
    fn closed_forms() {
        let g = random_loopy(14, 12, 12);
        let c = TreeCover::random(&g, 6, 1);
        let t = trw(&g, 0.0, &c, 200, 0.0);
        let want = g.n as f64 * std::f64::consts::LN_2;
        assert!((t.log_z - want).abs() < 1e-12, "beta = 0 gave {}", t.log_z);

        let (j, h1, h2) = (0.7, 0.3, -0.45);
        let mut gb = GraphBuilder::new(2);
        gb.couple(0, 1, j);
        gb.bias(0, h1);
        gb.bias(1, h2);
        let pair = gb.build();
        let want = (2.0 * (j.exp() * (h1 + h2).cosh() + (-j).exp() * (h1 - h2).cosh())).ln();
        let c = TreeCover::random(&pair, 2, 3);
        let t = trw(&pair, 1.0, &c, 500, 0.0);
        assert!((t.log_z - want).abs() < 1e-12, "pair gave {} want {want}", t.log_z);
        assert!((tree_decomposition_bound(&pair, 1.0, &c) - want).abs() < 1e-12);
    }

    /// A zero-field ring against its transfer-matrix closed form, and the bound above it.
    ///
    /// `Z = (2 cosh βJ)^n + (2 sinh βJ)^n`. An independent oracle: it agrees with elimination, and
    /// the single loop is the smallest model where `rho` is forced below 1 and the bound is loose.
    #[test]
    fn ring_transfer_matrix() {
        let (n, jj, beta) = (10usize, 1.0, 0.6);
        let g = ising::ring(n, jj, 0.0);
        let z =
            (2.0 * (beta * jj).cosh()).powi(n as i32) + (2.0 * (beta * jj).sinh()).powi(n as i32);
        let truth = z.ln();
        assert!(
            (exact(&g, beta) - truth).abs() < 1e-9,
            "elimination disagrees with the closed form"
        );
        let c = TreeCover::random(&g, n, 4);
        // every edge is in all but one of the n forests, so rho = (n-1)/n
        for &r in &c.rho {
            assert!(r < 1.0, "a loop cannot give rho = 1");
        }
        let t = trw(&g, beta, &c, 20_000, 0.5);
        assert!(t.converged(1e-11));
        assert!(t.log_z >= truth - 1e-9, "TRW {} below {truth}", t.log_z);
        assert!(t.log_z < trivial_upper_bound(&g, beta));
    }
}
