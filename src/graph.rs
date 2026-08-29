//! Sparse pairwise energy-based model over binary spins, with graph coloring for parallel Gibbs.
//!
//! Energy convention (statistical-mechanics standard):
//!     E(s) = - sum_{(i,j)} J_ij s_i s_j  -  sum_i h_i s_i,      s_i in {-1,+1}
//! so positive J is ferromagnetic (alignment lowers energy) and the Gibbs conditional is
//!     P(s_i = +1 | rest) = sigma(2 beta (sum_j J_ij s_j + h_i)).

// Builder-side edge list; finalized into CSR by Graph::build.
thread_local! {
    static BUILDS: core::cell::Cell<u64> = const { core::cell::Cell::new(0) };
}

/// Graphs built **on this thread**.
///
/// A program is a fixed thing and a schedule is a set of numbers; annealing must move the numbers,
/// never rebuild the program. This counter is how that claim is checked rather than asserted --
/// see the `anneal_never_rebuilds_the_program` test.
///
/// Deliberately per-thread rather than global: the question it answers is "did *this* run rebuild
/// anything", and a process-wide counter answers a different question the moment two runs share a
/// process. That is not hypothetical -- it is exactly what a parallel test runner does, and a
/// global counter here failed for that reason before this line existed.
pub fn graph_builds() -> u64 {
    BUILDS.with(|b| b.get())
}

pub struct GraphBuilder {
    n: usize,
    edges: Vec<(u32, u32, f64)>,
    bias: Vec<f64>,
}

impl GraphBuilder {
    pub fn new(n: usize) -> Self {
        GraphBuilder { n, edges: Vec::new(), bias: vec![0.0; n] }
    }

    /// Node count, so a caller across an FFI boundary can bounds-check before adding an edge.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Add an undirected coupling J_ij. Duplicate pairs are summed at build time.
    pub fn couple(&mut self, i: usize, j: usize, jij: f64) {
        assert!(i < self.n && j < self.n && i != j, "bad edge ({i},{j}) n={}", self.n);
        self.edges.push((i as u32, j as u32, jij));
    }

    /// Add bias h_i. Repeated calls on one node **accumulate**, matching `couple`.
    ///
    /// This replaced rather than accumulated until the domain-wall encoding caught it: with k = 2
    /// that encoding puts both of its boundary terms on the single spin, where they must cancel,
    /// and instead the second silently erased the first. Any two passes touching one node hit the
    /// same bug -- a user bias plus a penalty bias is the ordinary case -- so the asymmetry with
    /// `couple`, which has always summed duplicates, was the defect.
    pub fn bias(&mut self, i: usize, h: f64) {
        self.bias[i] += h;
    }

    /// Replace node `i`'s bias outright, discarding anything already accumulated.
    pub fn set_bias(&mut self, i: usize, h: f64) {
        self.bias[i] = h;
    }

    pub fn build(self) -> Graph {
        BUILDS.with(|b| b.set(b.get() + 1));
        let n = self.n;
        // Merge duplicates. BTreeMap, NOT HashMap.
        //
        // Rust randomises HashMap iteration per instance, and this map's iteration order decides
        // the CSR neighbour order, which decides the order every local field is SUMMED in. Float
        // addition is not associative, so with a HashMap here:
        //
        //   - eight builds of one graph gave eight different CSR orders,
        //   - the sampled state was identical every time (the RNG stream does not depend on it),
        //   - and the energy computed from that identical state took SIX distinct values, all of
        //     which print the same because they differ in the last bits.
        //
        // It also made `Program::to_ftp` non-reproducible: five runs of the same model emitted five
        // different programs, a pure permutation of one another. A program IR whose bytes depend on
        // which run produced it cannot be hashed, diffed, cached, or checked for reproducibility --
        // and "deterministic by seed" is this crate's headline.
        //
        // A BTreeMap iterates in key order. The merge goes from O(m) to O(m log m), which is
        // nothing beside the sampling it feeds, and the whole stack becomes byte-reproducible.
        let mut merged: std::collections::BTreeMap<(u32, u32), f64> = std::collections::BTreeMap::new();
        for (a, b, j) in self.edges {
            let key = if a < b { (a, b) } else { (b, a) };
            *merged.entry(key).or_insert(0.0) += j;
        }
        // CSR over both directions
        let mut deg = vec![0usize; n];
        for &(a, b) in merged.keys() {
            deg[a as usize] += 1;
            deg[b as usize] += 1;
        }
        let mut offset = vec![0usize; n + 1];
        for i in 0..n {
            offset[i + 1] = offset[i] + deg[i];
        }
        let m2 = offset[n];
        let mut nbr = vec![0u32; m2];
        let mut w = vec![0.0f64; m2];
        let mut cursor = offset.clone();
        for (&(a, b), &j) in merged.iter() {
            nbr[cursor[a as usize]] = b;
            w[cursor[a as usize]] = j;
            cursor[a as usize] += 1;
            nbr[cursor[b as usize]] = a;
            w[cursor[b as usize]] = j;
            cursor[b as usize] += 1;
        }
        let colors = color_for(n, &offset, &nbr);
        let n_colors = colors.iter().copied().max().map_or(1, |c| c as usize + 1);
        let mut classes: Vec<Vec<u32>> = vec![Vec::new(); n_colors];
        for i in 0..n {
            classes[colors[i] as usize].push(i as u32);
        }
        Graph { n, offset, nbr, w, h: self.bias, colors, classes, n_edges: merged.len() }
    }
}

/// Finalized CSR graph with a proper vertex coloring (no adjacent nodes share a color), so all
/// nodes of one color have conditionally independent Gibbs updates and sweep in parallel.
pub struct Graph {
    pub n: usize,
    pub offset: Vec<usize>,
    pub nbr: Vec<u32>,
    pub w: Vec<f64>,
    pub h: Vec<f64>,
    pub colors: Vec<u16>,
    pub classes: Vec<Vec<u32>>,
    pub n_edges: usize,
}

impl Graph {
    /// Local field at node i: sum_j J_ij s_j + h_i.
    #[inline]
    pub fn field(&self, i: usize, s: &[i8]) -> f64 {
        let mut f = self.h[i];
        for k in self.offset[i]..self.offset[i + 1] {
            f += self.w[k] * s[self.nbr[k] as usize] as f64;
        }
        f
    }

    /// Total energy E(s) = -sum_edges J s s - sum_i h s.
    pub fn energy(&self, s: &[i8]) -> f64 {
        let mut e = 0.0;
        for i in 0..self.n {
            let si = s[i] as f64;
            e -= self.h[i] * si;
            for k in self.offset[i]..self.offset[i + 1] {
                let j = self.nbr[k] as usize;
                if j > i {
                    e -= self.w[k] * si * s[j] as f64;
                }
            }
        }
        e
    }

    pub fn max_degree(&self) -> usize {
        (0..self.n).map(|i| self.offset[i + 1] - self.offset[i]).max().unwrap_or(0)
    }
}

/// Greedy coloring in vertex order. For bipartite graphs presented in any order this may exceed
/// two colors; callers with known structure (e.g. the Z1 grid) can verify with
/// [`Graph::colors`].len-of-classes or construct order so greedy finds the checkerboard.
/// A proper colouring, and as few colours as this crate's graphs actually need.
///
/// A chromatic sweep runs one pass per colour, so the colour count is the number of SEQUENTIAL
/// barriers in a sweep and, on the GPU path, the number of dispatches. Fewer colours is strictly
/// better: the same spins, updated in fewer synchronised waves.
///
/// Greedy in index order is already optimal on almost everything here, because almost everything
/// here is bipartite -- lattices, grids, rings of even length, every RBM and every deep machine.
/// CHIMERA IS THE EXCEPTION AND IT IS NOT A SMALL ONE: it is bipartite, and greedy spends THREE
/// colours on it, so every Chimera sweep paid an extra pass for nothing. It is also the topology
/// the hardware comparisons in this crate use.
///
/// So: greedy first, and only when greedy needed three or more is a two-colouring looked for.
/// That ordering is deliberate rather than tidy. Replacing an already-two-coloured graph's
/// assignment would change the order spins are visited in, which changes every seeded trajectory
/// in the repository for no gain at all -- the colour COUNT would be identical. This way the only
/// graphs whose results move are the ones that were being coloured badly.
///
/// A graph that is not bipartite keeps greedy's answer. Greedy is not optimal in general -- an odd
/// ring needs three and gets three, but a crafted graph can defeat it -- and DSATUR or a
/// largest-first order would do better on dense irregular graphs. This review did not locate a
/// non-bipartite graph in this crate that greedy colours suboptimally, so that work is not done
/// here rather than done speculatively.
fn color_for(n: usize, offset: &[usize], nbr: &[u32]) -> Vec<u16> {
    let greedy = color_greedy(n, offset, nbr);
    let used = greedy.iter().max().map_or(0, |&c| c as usize + 1);
    if used < 3 {
        return greedy;
    }
    match two_color(n, offset, nbr) {
        Some(c) => c,
        None => greedy,
    }
}

/// Breadth-first two-colouring, or `None` when an edge joins two nodes of the same part.
///
/// Every component is coloured independently, since a disconnected graph is bipartite exactly when
/// each of its components is.
fn two_color(n: usize, offset: &[usize], nbr: &[u32]) -> Option<Vec<u16>> {
    const UNSET: u16 = u16::MAX;
    let mut color = vec![UNSET; n];
    let mut stack = Vec::new();
    for s in 0..n {
        if color[s] != UNSET {
            continue;
        }
        color[s] = 0;
        stack.push(s);
        while let Some(v) = stack.pop() {
            for k in offset[v]..offset[v + 1] {
                let u = nbr[k] as usize;
                if color[u] == UNSET {
                    color[u] = 1 - color[v];
                    stack.push(u);
                } else if color[u] == color[v] {
                    return None;
                }
            }
        }
    }
    Some(color)
}

fn color_greedy(n: usize, offset: &[usize], nbr: &[u32]) -> Vec<u16> {
    let mut colors = vec![u16::MAX; n];
    let mut used: Vec<bool> = Vec::new();
    for i in 0..n {
        used.clear();
        used.resize(64, false);
        for k in offset[i]..offset[i + 1] {
            let c = colors[nbr[k] as usize];
            if c != u16::MAX {
                if (c as usize) >= used.len() {
                    used.resize(c as usize + 1, false);
                }
                used[c as usize] = true;
            }
        }
        let c = used.iter().position(|&u| !u).unwrap_or(used.len());
        colors[i] = c as u16;
    }
    colors
}

#[cfg(test)]
mod tests {
    use super::*;

    /// CHIMERA IS BIPARTITE AND GREEDY DID NOT NOTICE, which cost every Chimera sweep an extra
    /// sequential pass and left the classes lopsided.
    #[test]
    fn chimera_takes_two_colours_and_even_classes() {
        let g = crate::ising::chimera(8, 8, 4, 1.0);
        let nc = g.colors.iter().max().map_or(0, |&c| c as usize + 1);
        assert_eq!(nc, 2, "chimera is bipartite; three colours is a wasted pass");
        let mut sizes = vec![0usize; nc];
        for &c in &g.colors {
            sizes[c as usize] += 1;
        }
        // Even classes are the half of this that the parallel path feels: an undersized class
        // leaves threads idle behind a barrier they still have to wait at.
        assert_eq!(sizes, vec![256, 256], "and the two parts are the same size");
        // Still a proper colouring, which is the only thing that makes the sweep correct at all.
        for i in 0..g.n {
            for k in g.offset[i]..g.offset[i + 1] {
                assert_ne!(g.colors[i], g.colors[g.nbr[k] as usize]);
            }
        }
    }

    /// A graph that is NOT bipartite keeps greedy's answer rather than getting a wrong one.
    #[test]
    fn an_odd_ring_is_not_forced_into_two_colours() {
        let g = crate::ising::ring(7, 1.0, 0.0);
        let nc = g.colors.iter().max().map_or(0, |&c| c as usize + 1);
        assert_eq!(nc, 3, "an odd cycle needs three, and asking for two must not succeed");
        for i in 0..g.n {
            for k in g.offset[i]..g.offset[i + 1] {
                assert_ne!(g.colors[i], g.colors[g.nbr[k] as usize], "still proper");
            }
        }
    }

    /// THE BLAST RADIUS IS THE PROMISE. Replacing an already-two-coloured graph's assignment would
    /// change the order spins are visited in, and so every seeded trajectory in this repository,
    /// for a colour COUNT that was already identical. This asserts the rule directly rather than
    /// trusting the comment that states it: where greedy used fewer than three, its exact output
    /// survives byte for byte.
    #[test]
    fn a_graph_greedy_already_two_coloured_keeps_greedys_exact_assignment() {
        let cases = [
            crate::ising::lattice2d(16, 1.0),
            crate::ising::ring(8, 1.0, 0.0),
            crate::ising::grid2d(9, 7, 1.0),
        ];
        for g in &cases {
            let greedy = color_greedy(g.n, &g.offset, &g.nbr);
            let used = greedy.iter().max().map_or(0, |&c| c as usize + 1);
            assert!(used < 3, "this case is meant to test the untouched path, not the other one");
            assert_eq!(g.colors, greedy, "an already-good colouring must not be rewritten");
        }
    }

    #[test]
    fn coloring_is_proper() {
        // random-ish sparse graph
        let mut gb = GraphBuilder::new(100);
        let mut x = 1u64;
        for _ in 0..300 {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let a = (x >> 33) as usize % 100;
            let b = (x >> 13) as usize % 100;
            if a != b {
                gb.couple(a, b, 0.5);
            }
        }
        let g = gb.build();
        for i in 0..g.n {
            for k in g.offset[i]..g.offset[i + 1] {
                assert_ne!(g.colors[i], g.colors[g.nbr[k] as usize], "adjacent same color");
            }
        }
        // classes partition the vertex set
        let total: usize = g.classes.iter().map(|c| c.len()).sum();
        assert_eq!(total, g.n);
    }

    #[test]
    fn a_graph_builds_bit_identically_every_time() {
        // "Deterministic by seed" is this crate's headline, and it was only half true. The merge in
        // `build` used a HashMap, whose iteration order Rust randomises per instance, and that
        // order decides the CSR neighbour order -- which decides the order every local field is
        // SUMMED in. Float addition is not associative.
        //
        // Measured before the fix, over eight builds of one graph: eight distinct CSR orders, ONE
        // sampled state (the RNG stream does not depend on the order), and SIX distinct energies
        // computed from that identical state, all printing the same because they differed in the
        // last bits.
        use crate::gibbs::Sampler;
        use crate::planted::wishart;
        let mut orders = Vec::new();
        let mut states = Vec::new();
        let mut bits = Vec::new();
        for _ in 0..8 {
            let g = wishart(40, 1.0, 7).graph;
            orders.push(g.nbr.clone());
            let mut s = Sampler::new(&g, 1.2, 42);
            s.sweeps(200, None);
            bits.push(g.energy(&s.s).to_bits());
            states.push(s.s.clone());
        }
        assert!(orders.windows(2).all(|w| w[0] == w[1]), "CSR neighbour order must not vary");
        assert!(states.windows(2).all(|w| w[0] == w[1]), "the sampled state must not vary");
        assert!(
            bits.windows(2).all(|w| w[0] == w[1]),
            "energies must be BIT-identical WITHIN a platform, not merely equal to the digits that \
             get printed"
        );
        // Across platforms this is weaker, and the crate docs say so: the same state and the same
        // program come out identical on macOS/arm64, Linux/x86_64 and Linux/aarch64, while the
        // energy computed FROM that identical state differs by one ULP between macOS and Linux --
        // floating-point contraction, not libm, which was measured bit-identical on both. Within
        // one platform there is no excuse for variation, which is what this asserts.
    }

    #[test]
    fn the_compiled_program_is_byte_reproducible() {
        // A program IR whose bytes depend on which run produced it cannot be hashed, diffed, cached
        // or checked for reproducibility. Five runs of one model used to emit five different
        // programs -- a pure permutation of each other, identical in length, which is why nothing
        // noticed. It also meant two BINDINGS building the same model disagreed byte for byte,
        // which is how this was found: check-parity proves a symbol exists on nine surfaces and
        // says nothing about whether they compute the same thing.
        use crate::model::{Expr, Lit, Model, Sense};
        let build = || {
            let mut m = Model::new();
            let a = m.categorical("a", 3);
            let b = m.categorical("b", 3);
            m.not_equal(a, b);
            m.at_most(vec![Lit::Is(a, 0), Lit::Is(b, 0)], 1);
            m.objective(Sense::Maximize, Expr::product(3.0, &[Lit::Is(a, 1)]));
            m.compile().unwrap().program.to_ftp()
        };
        let first = build();
        for _ in 0..4 {
            assert_eq!(build(), first, "the same model must compile to the same bytes");
        }
    }
}
