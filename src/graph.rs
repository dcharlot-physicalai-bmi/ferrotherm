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
        let colors = color_greedy(n, &offset, &nbr);
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
