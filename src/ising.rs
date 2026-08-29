//! Ising-model constructions and the exact results the sampler must reproduce before it is
//! trusted with anything else: exact Boltzmann enumeration for small systems, and Onsager's
//! spontaneous magnetization for the 2D nearest-neighbor lattice (Onsager 1944 / Yang 1952):
//!     M(beta) = (1 - sinh(2 beta J)^-4)^(1/8)   for beta > beta_c = ln(1+sqrt(2))/2 ~ 0.4407,
//!     M = 0 above.

use crate::graph::{Graph, GraphBuilder};

/// Ring of n spins, uniform coupling j, per-site bias h.
pub fn ring(n: usize, j: f64, h: f64) -> Graph {
    let mut gb = GraphBuilder::new(n);
    for i in 0..n {
        gb.couple(i, (i + 1) % n, j);
        if h != 0.0 {
            gb.bias(i, h);
        }
    }
    gb.build()
}

/// 2D nearest-neighbor square lattice with periodic boundaries, uniform J, no field.
pub fn lattice2d(l: usize, j: f64) -> Graph {
    // A side of 1 wraps every neighbour onto the site itself, so the periodic boundary produces the
    // self-edge (0,0), which `GraphBuilder::couple` refuses with a panic -- reached through
    // `ft_ising2d_new(1, ..)` that is a NON-UNWINDING panic and aborts the caller's process, while
    // the header documents a NULL return. `l = 0` already returned a live empty handle, so the
    // guard band stopped one short of the fatal value. An uncoupled lattice is the honest answer
    // for a side with no distinct neighbours.
    if l < 2 {
        return GraphBuilder::new(l * l).build();
    }
    let mut gb = GraphBuilder::new(l * l);
    for y in 0..l {
        for x in 0..l {
            let i = y * l + x;
            gb.couple(i, y * l + (x + 1) % l, j);
            gb.couple(i, ((y + 1) % l) * l + x, j);
        }
    }
    gb.build()
}

/// A **Chimera** graph `C_{m,n,t}`: an `m × n` grid of unit cells, each a complete bipartite
/// `K_{t,t}`, with cells joined between their matching shores.
///
/// This is D-Wave's 2000Q topology (`C_{16,16,4}`, 2048 qubits) and the graph almost every published
/// annealer-versus-classical comparison was run on. This crate could not build one until now:
/// `ring`, `lattice2d` and `grid2d` are all lattices of degree 4, `embed`'s King's graph lives only
/// in its tests, and [`crate::hfs`] — the algorithm those comparisons turned on — had never been
/// measured on the structure it exploits.
///
/// # The indexing, which is D-Wave's
///
/// A qubit is `(i, j, u, k)`: cell row `i`, cell column `j`, shore `u ∈ {0, 1}`, index `k ∈ [0, t)`.
/// Linearised as
///
/// ```text
///     index = ((i * n) + j) * 2t  +  u * t  +  k
/// ```
///
/// so `dwave_networkx.chimera_graph(m, n, t)` with its default linear labelling gives the same
/// numbers. Getting this wrong would produce a graph with the right SHAPE and the wrong labels,
/// which no count would catch and every embedding would inherit.
///
/// Edges:
///
/// - **within a cell**, every shore-0 qubit to every shore-1 qubit: `t²` per cell;
/// - **vertically**, shore 0 to shore 0 between `(i, j)` and `(i+1, j)` at matching `k`;
/// - **horizontally**, shore 1 to shore 1 between `(i, j)` and `(i, j+1)` at matching `k`.
///
/// So `|V| = 2·t·m·n` and `|E| = m·n·t² + (m−1)·n·t + m·(n−1)·t`.
///
/// # Why the shores matter
///
/// Shore 0 touches shore 1 only *inside* a cell, and shore 0 touches shore 0 only *vertically*. So
/// **the subgraph induced on all of shore 0 is a disjoint union of paths** — one per `(j, k)`, of
/// length `m` — and therefore a forest of width 1. Shore 1 is the same, horizontally. Two blocks,
/// each exactly solvable, together covering every vertex.
///
/// That is the decomposition Selby's HFS implementation exploits, and a periodic square lattice has
/// no such split: its treewidth is its side.
///
/// **It does not follow that block methods win here, and measuring it said they do not.**
/// `examples/hfs_reach` runs [`crate::hfs`] against tabu on `C_{4,4,4}` through `C_{8,8,4}` at a
/// matched budget and HFS loses by about 4% at every size — after sweeping block size from 8 to
/// `n` and restart counts from 1 to 256, neither of which moved it. Recorded here because this
/// paragraph originally asserted the opposite from the literature rather than from a run.
///
/// `j` is the uniform coupling. For the spin-glass instances these comparisons actually use, build
/// with `j = 1.0` and rewrite the weights, or use [`chimera_glass`].
pub fn chimera(m: usize, n: usize, t: usize, j: f64) -> Graph {
    let cells = m * n;
    if cells == 0 || t == 0 {
        return GraphBuilder::new(0).build();
    }
    let idx = |i: usize, jj: usize, u: usize, k: usize| ((i * n) + jj) * 2 * t + u * t + k;
    let mut gb = GraphBuilder::new(2 * t * cells);
    for i in 0..m {
        for jj in 0..n {
            // The cell: K_{t,t} between the shores.
            for a in 0..t {
                for b in 0..t {
                    gb.couple(idx(i, jj, 0, a), idx(i, jj, 1, b), j);
                }
            }
            // Shore 0 runs vertically, shore 1 runs horizontally. Each edge once, forward only.
            if i + 1 < m {
                for k in 0..t {
                    gb.couple(idx(i, jj, 0, k), idx(i + 1, jj, 0, k), j);
                }
            }
            if jj + 1 < n {
                for k in 0..t {
                    gb.couple(idx(i, jj, 1, k), idx(i, jj + 1, 1, k), j);
                }
            }
        }
    }
    gb.build()
}

/// A Chimera **spin glass**: the same graph with couplings drawn uniformly from `{−1, +1}`.
///
/// The instance family the annealer-versus-classical literature is written about. Uniform ±1 rather
/// than Gaussian because that is what the D-Wave benchmark sets used, and the point of having this
/// is to be able to run the same comparisons rather than adjacent ones.
pub fn chimera_glass(m: usize, n: usize, t: usize, seed: u64) -> Graph {
    let g = chimera(m, n, t, 1.0);
    let mut rng = crate::rng::Pcg::new(seed, 0x00C1_1E5A);
    let mut gb = GraphBuilder::new(g.n);
    for i in 0..g.n {
        for k in g.offset[i]..g.offset[i + 1] {
            let jj = g.nbr[k] as usize;
            if jj > i {
                gb.couple(i, jj, if rng.f64() < 0.5 { 1.0 } else { -1.0 });
            }
        }
    }
    gb.build()
}

/// The qubits of one Chimera shore, which induce a **forest** and so are exactly solvable.
///
/// `u = 0` is the vertical shore, `u = 1` the horizontal one. Together the two cover every vertex
/// and neither induces a cycle, which is what makes them the natural blocks for [`crate::hfs`].
///
/// Returns an empty list when the arguments do not describe a graph.
pub fn chimera_shore(m: usize, n: usize, t: usize, u: usize) -> Vec<usize> {
    if m * n == 0 || t == 0 || u > 1 {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(t * m * n);
    for i in 0..m {
        for jj in 0..n {
            for k in 0..t {
                out.push(((i * n) + jj) * 2 * t + u * t + k);
            }
        }
    }
    out
}

/// Exact Boltzmann distribution over all 2^n states (n <= 24). Returns probabilities indexed by
/// bitmask (bit b set => spin b = +1).
/// A `w × h` grid with **open** boundaries: the planar one.
///
/// [`lattice2d`] wraps, which makes it a torus — genus 1, not planar, and the distinction is the
/// whole reason [`crate::planarcut`] cannot be pointed at it. This is the grid that embeds in the
/// plane, and it is also the classic planar max-cut instance: a spin glass on a square lattice with
/// free boundaries.
pub fn grid2d(w: usize, h: usize, j: f64) -> Graph {
    let mut gb = GraphBuilder::new(w * h);
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            if x + 1 < w {
                gb.couple(i, i + 1, j);
            }
            if y + 1 < h {
                gb.couple(i, i + w, j);
            }
        }
    }
    gb.build()
}

pub fn exact_boltzmann(g: &Graph, beta: f64) -> Vec<f64> {
    assert!(g.n <= 24, "exact enumeration limited to 24 spins");
    let m = 1usize << g.n;
    let mut p = vec![0.0f64; m];
    let mut s = vec![-1i8; g.n];

    // Two passes with a max shift, because `exp(-beta*E)` overflows long before beta gets large.
    //
    // This used to accumulate `(-beta * g.energy(&s)).exp()` directly. `f64` tops out near
    // exp(709), so on a 24-spin complete ferromagnet (E_min = -276) beta = 3 already overflows two
    // states to +inf, z becomes inf, and every p = w/inf comes back 0 or NaN. The crate's OWN
    // schedules run to beta 6 and 8, and this is the exact reference every sampler is verified
    // against -- so the oracle silently stopped answering inside the range it is used in, and
    // `certify` swallowed it because `NaN > floor` is false.
    //
    // Subtracting the maximum log-weight leaves the normalised distribution identical (the shift
    // cancels in w/z) and keeps every exponent at or below zero. `HetSampler::sweep` already did
    // exactly this; the enumerations simply never got it.
    let mut mx = f64::NEG_INFINITY;
    for mask in 0..m {
        for b in 0..g.n {
            s[b] = if mask >> b & 1 == 1 { 1 } else { -1 };
        }
        let l = -beta * g.energy(&s);
        p[mask] = l;
        if l > mx {
            mx = l;
        }
    }
    let mut z = 0.0;
    for v in p.iter_mut() {
        *v = (*v - mx).exp();
        z += *v;
    }
    for v in p.iter_mut() {
        *v /= z;
    }
    p
}

/// Onsager/Yang exact spontaneous magnetization for the infinite 2D lattice (J=1).
pub fn onsager_m(beta: f64) -> f64 {
    let s = (2.0 * beta).sinh();
    let x = 1.0 - s.powi(-4);
    if x <= 0.0 {
        0.0
    } else {
        x.powf(1.0 / 8.0)
    }
}

/// Total-variation distance between two distributions.
pub fn tv(p: &[f64], q: &[f64]) -> f64 {
    // `zip` stops at the shorter of the two, so mismatched lengths used to return a TRUNCATED
    // distance rather than an error: `tv(&[0.25; 4], &[0.5, 0.5])` gave 0.25 where the honest
    // answer over the shared 4-state space is 0.5. Truncation always under-estimates, and every
    // use of this function has the shape `assert!(tv < tolerance)` -- so the failure direction was
    // the one that turns a red test green.
    assert_eq!(
        p.len(),
        q.len(),
        "total variation needs two distributions over the SAME state space"
    );
    p.iter().zip(q).map(|(a, b)| (a - b).abs()).sum::<f64>() / 2.0
}

#[cfg(test)]
mod tests {

    /// Counts first, because they are cheap and they catch a whole class of wrong.
    #[test]
    fn chimera_has_the_vertices_and_edges_the_formula_says() {
        for (m, n, tt) in [(1usize, 1usize, 4usize), (2, 3, 4), (4, 4, 4), (16, 16, 4), (3, 3, 2)] {
            let g = chimera(m, n, tt, 1.0);
            assert_eq!(g.n, 2 * tt * m * n, "C_{{{m},{n},{tt}}} vertices");
            let want = m * n * tt * tt + (m - 1) * n * tt + m * (n - 1) * tt;
            assert_eq!(g.n_edges, want, "C_{{{m},{n},{tt}}} edges");
        }
        // The one everyone quotes: D-Wave 2000Q.
        let dw = chimera(16, 16, 4, 1.0);
        assert_eq!(dw.n, 2048, "C_16,16,4 is 2048 qubits");
        assert_eq!(dw.n_edges, 6016);
    }

    /// The DEGREE PROFILE, which is what a wrong wiring changes and a count does not.
    ///
    /// In `C_{m,n,t}` every qubit has `t` intra-cell neighbours plus one inter-cell neighbour per
    /// direction it is not on the boundary of. So the interior degree is `t + 2` and a corner cell's
    /// qubits sit at `t + 1`. Getting the shores backwards -- wiring shore 0 horizontally -- leaves
    /// every count above identical and this distribution unchanged too, which is why the shore test
    /// below exists as well.
    #[test]
    fn chimera_degrees_are_t_plus_the_directions_a_qubit_is_not_on_the_edge_of() {
        let (m, n, tt) = (4usize, 5usize, 4usize);
        let g = chimera(m, n, tt, 1.0);
        for i in 0..m {
            for jj in 0..n {
                for u in 0..2 {
                    for k in 0..tt {
                        let q = ((i * n) + jj) * 2 * tt + u * tt + k;
                        let deg = g.offset[q + 1] - g.offset[q];
                        // Shore 0 runs vertically (rows), shore 1 horizontally (columns).
                        let inter = if u == 0 {
                            usize::from(i > 0) + usize::from(i + 1 < m)
                        } else {
                            usize::from(jj > 0) + usize::from(jj + 1 < n)
                        };
                        assert_eq!(
                            deg,
                            tt + inter,
                            "({i},{jj},{u},{k}) index {q}: degree {deg}, expected {} + {inter}",
                            tt
                        );
                    }
                }
            }
        }
    }

    /// THE PROPERTY THE WHOLE THING IS FOR: each shore induces a FOREST.
    ///
    /// Shore 0 touches shore 1 only inside a cell and shore 0 only vertically, so the subgraph it
    /// induces is a disjoint union of `n·t` paths of length `m` -- acyclic, width 1, exactly
    /// solvable. Two such blocks cover every vertex. If the shores were wired the same way as each
    /// other, or a cell's `K_{t,t}` leaked a same-shore edge, this is what would catch it.
    #[test]
    fn each_chimera_shore_induces_a_forest_and_the_two_cover_everything() {
        let (m, n, tt) = (4usize, 5usize, 4usize);
        let g = chimera(m, n, tt, 1.0);
        let mut seen = vec![false; g.n];

        for u in 0..2 {
            let shore = chimera_shore(m, n, tt, u);
            assert_eq!(shore.len(), tt * m * n, "a shore is half the graph");
            let inside: std::collections::BTreeSet<usize> = shore.iter().copied().collect();
            assert_eq!(inside.len(), shore.len(), "a shore is a set");

            let mut edges = 0usize;
            for &q in &shore {
                seen[q] = true;
                for k in g.offset[q]..g.offset[q + 1] {
                    let r = g.nbr[k] as usize;
                    if r > q && inside.contains(&r) {
                        edges += 1;
                    }
                }
            }
            // A forest of `c` components on `v` vertices has exactly `v - c` edges. Shore 0 is
            // `n*t` vertical paths; shore 1 is `m*t` horizontal ones.
            let components = if u == 0 { n * tt } else { m * tt };
            assert_eq!(
                edges,
                shore.len() - components,
                "shore {u}: {} vertices, {edges} edges, {components} paths is not a forest",
                shore.len()
            );

            // And the exact solver agrees: width 1.
            let mut map = vec![usize::MAX; g.n];
            for (a, &q) in shore.iter().enumerate() {
                map[q] = a;
            }
            let mut gb = GraphBuilder::new(shore.len());
            for (a, &q) in shore.iter().enumerate() {
                for k in g.offset[q]..g.offset[q + 1] {
                    let r = g.nbr[k] as usize;
                    if map[r] != usize::MAX && r > q {
                        gb.couple(a, map[r], g.w[k]);
                    }
                }
            }
            assert_eq!(crate::exact::Elimination::default().width(&gb.build()), 1);
        }
        assert!(seen.iter().all(|&x| x), "the two shores must cover every vertex");
    }

    /// A glass has the same graph and different weights, and the weights are actually mixed.
    #[test]
    fn a_chimera_glass_is_the_same_graph_with_signs() {
        let plain = chimera(3, 3, 4, 1.0);
        let g = chimera_glass(3, 3, 4, 11);
        assert_eq!(g.n, plain.n);
        assert_eq!(g.n_edges, plain.n_edges);
        assert!(g.w.iter().all(|w| w.abs() == 1.0), "couplings are +/-1");
        let pos = g.w.iter().filter(|w| **w > 0.0).count();
        assert!(pos > 0 && pos < g.w.len(), "a glass is frustrated, not a ferromagnet: {pos}");
        // Same seed, same instance -- otherwise no comparison across arms means anything.
        assert_eq!(g.w, chimera_glass(3, 3, 4, 11).w);
        assert_ne!(g.w, chimera_glass(3, 3, 4, 12).w);
    }

    #[test]
    fn a_degenerate_chimera_is_empty_rather_than_a_panic() {
        assert_eq!(chimera(0, 4, 4, 1.0).n, 0);
        assert_eq!(chimera(4, 0, 4, 1.0).n, 0);
        assert_eq!(chimera(4, 4, 0, 1.0).n, 0);
        assert!(chimera_shore(0, 4, 4, 0).is_empty());
        assert!(chimera_shore(4, 4, 4, 2).is_empty(), "there are two shores");
        // A single cell is a legitimate K_{t,t} with no inter-cell edges at all.
        let one = chimera(1, 1, 4, 1.0);
        assert_eq!(one.n, 8);
        assert_eq!(one.n_edges, 16);
    }

    use super::*;

    #[test]
    fn the_exact_reference_survives_the_betas_its_own_schedules_use() {
        // This is the oracle every sampler in the crate is verified against, and it used to return
        // NaN. `exp(-beta*E)` overflows f64 near exp(709), so a large beta sent z to +inf and every
        // p to 0 or NaN -- and `certify` swallowed it, because `NaN > floor` is false. The crate's
        // own schedules run to beta 6 and 8, so this was inside the range it is actually used in.
        let g = ring(8, 1.0, 0.0);
        for beta in [1.0, 3.0, 6.0, 8.0, 24.0, 200.0] {
            let p = exact_boltzmann(&g, beta);
            assert!(p.iter().all(|v| v.is_finite()), "beta={beta} produced a non-finite entry");
            let sum: f64 = p.iter().sum();
            assert!((sum - 1.0).abs() < 1e-12, "beta={beta} sums to {sum}, not 1");
        }

        // Still RIGHT, not merely finite: at low temperature a ferromagnetic ring puts half its
        // mass on all-down and half on all-up, and nothing measurable anywhere else.
        let p = exact_boltzmann(&g, 20.0);
        assert!((p[0] - 0.5).abs() < 1e-9, "all-down should carry half the mass, got {}", p[0]);
        assert!((p[255] - 0.5).abs() < 1e-9, "all-up should carry half the mass, got {}", p[255]);
    }

    #[test]
    #[should_panic(expected = "SAME state space")]
    fn a_total_variation_between_different_sized_distributions_is_refused() {
        // `zip` stopped at the shorter one, so this returned 0.25 where the honest answer over the
        // shared 4-state space is 0.5. Truncation always UNDER-estimates, and every use of `tv` has
        // the shape `assert!(tv < tolerance)` -- the failure direction that turns a red test green.
        let _ = tv(&[0.25, 0.25, 0.25, 0.25], &[0.5, 0.5]);
    }
}
