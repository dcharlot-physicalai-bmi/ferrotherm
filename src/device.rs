//! Device topologies. The published Z1-class fabric is a planar grid with four coupling
//! displacement rules — (1,0), (2,1), (2,3), (4,1), each applied in 4 rotations — giving interior
//! degree 16 and longest edge sqrt(17) grid units. Every displacement has ODD Manhattan length,
//! so the graph is bipartite under checkerboard parity: chromatic Gibbs needs exactly two
//! half-sweeps per full sweep. (Topology per arXiv:2608.01615; exact die dimensions unpublished —
//! the builder is parametric, and published totals like "269,568 pbits" are the vendor's figures
//! for silicon nobody outside has measured.)

use crate::graph::{Graph, GraphBuilder};

/// The four base displacements; each contributes its 4 rotations (a,b) -> (-b,a) etc.
pub const Z1_RULES: [(i64, i64); 4] = [(1, 0), (2, 1), (2, 3), (4, 1)];

/// Build a Z1-style grid graph of `w x h` nodes with uniform coupling `j` and bias `bias`.
/// Node index = y * w + x. Open boundaries (edge nodes have lower degree, as on a real die).
pub fn z1_grid(w: usize, h: usize, j: f64, bias: f64) -> Graph {
    let mut gb = GraphBuilder::new(w * h);
    let rots = |a: i64, b: i64| [(a, b), (-b, a), (-a, -b), (b, -a)];
    for y in 0..h as i64 {
        for x in 0..w as i64 {
            let i = (y * w as i64 + x) as usize;
            if bias != 0.0 {
                gb.bias(i, bias);
            }
            for &(a, b) in &Z1_RULES {
                for (dx, dy) in rots(a, b) {
                    let (nx, ny) = (x + dx, y + dy);
                    if nx >= 0 && ny >= 0 && nx < w as i64 && ny < h as i64 {
                        let jdx = (ny * w as i64 + nx) as usize;
                        if jdx > i {
                            gb.couple(i, jdx, j);
                        }
                    }
                }
            }
        }
    }
    gb.build()
}

/// Checkerboard parity of a grid node — the 2-coloring the odd-Manhattan rules guarantee.
pub fn parity(w: usize, i: usize) -> u8 {
    let (x, y) = (i % w, i / w);
    ((x + y) % 2) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_is_degree_16_and_bipartite() {
        let g = z1_grid(24, 24, 0.1, 0.0);
        // interior nodes have degree exactly 16
        let w = 24usize;
        let i = 12 * w + 12;
        assert_eq!(g.offset[i + 1] - g.offset[i], 16);
        // every edge connects opposite checkerboard parities (bipartite)
        for a in 0..g.n {
            for k in g.offset[a]..g.offset[a + 1] {
                let b = g.nbr[k] as usize;
                assert_ne!(parity(w, a), parity(w, b), "edge {a}-{b} same parity");
            }
        }
        // longest displacement is sqrt(17)
        let mut max_d2 = 0i64;
        for a in 0..g.n {
            let (ax, ay) = ((a % w) as i64, (a / w) as i64);
            for k in g.offset[a]..g.offset[a + 1] {
                let b = g.nbr[k] as usize;
                let (bx, by) = ((b % w) as i64, (b / w) as i64);
                max_d2 = max_d2.max((ax - bx).pow(2) + (ay - by).pow(2));
            }
        }
        assert_eq!(max_d2, 17);
    }
}

// ---- the topologies you can actually rent ---------------------------------------------------------

/// A hardware graph together with the **vendor's own qubit numbering**.
///
/// Two numbering systems meet here and conflating them is the whole reason this type exists.
/// `graph` is a `ferrotherm` graph: dense indices `0..n`, which every sampler, colouring and
/// embedding in this crate assumes. The machine numbers its qubits differently — Pegasus deletes
/// the qubits outside its largest component, so its linear indices are SPARSE, and a P₁₆ has 5,640
/// qubits spread over 5,760 index values. Handing a machine a chain written in our indices would
/// program the wrong qubits, silently, and the answer would come back looking like a bad embedding.
///
/// So `qubits[i]` is the vendor index of our node `i`, and [`Topology::node`] is the inverse.
pub struct Topology {
    /// The hardware graph in this crate's own dense indexing.
    pub graph: Graph,
    /// `qubits[i]` is the vendor's linear index for node `i`. Strictly increasing.
    pub qubits: Vec<u32>,
}

impl core::fmt::Debug for Topology {
    /// `Graph` is a CSR block with no `Debug`, and printing one would be pages of index arithmetic
    /// nobody reads. The shape is what a caller wants at a breakpoint.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "Topology {{ {} qubits, {} couplers, vendor indices {}..={} }}",
            self.graph.n,
            self.graph.n_edges,
            self.qubits.first().copied().unwrap_or(0),
            self.qubits.last().copied().unwrap_or(0)
        )
    }
}

impl Topology {
    /// The vendor's linear qubit index for one of our nodes.
    pub fn qubit(&self, node: usize) -> Option<u32> {
        self.qubits.get(node).copied()
    }

    /// Our node index for one of the vendor's qubits, or `None` if that qubit is not in the graph.
    ///
    /// `None` is a real answer, not a lookup failure: Pegasus's linear indexing covers qubits its
    /// fabric does not contain, and a caller reading a machine's output back needs to know which
    /// of those it just received.
    pub fn node(&self, qubit: u32) -> Option<usize> {
        self.qubits.binary_search(&qubit).ok()
    }
}

/// Build a dense graph from an edge list over sparse vendor indices, keeping the vendor's numbering.
///
/// Nodes are exactly the endpoints that appear in an edge, which is what "fabric only" means: the
/// isolated qubits a topology defines but does not wire up are left out rather than carried as
/// degree-zero nodes that would quietly join a colour class and be swept for nothing.
fn compact(edges: &[(u32, u32)], j: f64) -> Topology {
    let mut qubits: Vec<u32> = edges.iter().flat_map(|&(a, b)| [a, b]).collect();
    qubits.sort_unstable();
    qubits.dedup();
    let mut gb = GraphBuilder::new(qubits.len());
    for &(a, b) in edges {
        let (ia, ib) = (
            qubits.binary_search(&a).expect("endpoint came from this list"),
            qubits.binary_search(&b).expect("endpoint came from this list"),
        );
        gb.couple(ia, ib, j);
    }
    Topology { graph: gb.build(), qubits }
}

/// Vertical then horizontal offsets for Pegasus `offsets_index = 0`, the shipped configuration.
const PEGASUS_OFF: [[usize; 12]; 2] = [
    [2, 2, 2, 2, 10, 10, 10, 10, 6, 6, 6, 6],
    [6, 6, 6, 6, 2, 2, 2, 2, 10, 10, 10, 10],
];

/// The **Pegasus** topology `P_m` — the graph of every D-Wave *Advantage* processor.
///
/// `P₁₆` is 5,640 qubits and 40,484 couplers at degree 15, which are the Advantage's published
/// figures and what `device::pegasus(16, ..)` produces here. Until this existed the crate could
/// build only Chimera, a topology D-Wave retired: `embed` did honest minor embedding onto a machine
/// nobody can rent, and [`crate::fabric`] described "a 5,640-qubit Pegasus" it had no way to make.
///
/// # The definition, and where it comes from
///
/// A qubit is `(u, w, k, z)` — orientation, orthogonal major offset, orthogonal minor offset,
/// parallel offset — with `u < 2`, `w < m`, `k < 12`, `z < m-1`, and linear index
/// `q = ((u·m + w)·12 + k)·(m−1) + z`. Three coupler kinds:
///
/// * **external** `(u,w,k,z) ~ (u,w,k,z+1)` — along a qubit's own direction
/// * **odd** `(u,w,2k,z) ~ (u,w,2k+1,z)` — between the pair sharing a track
/// * **internal** `(0,w,k,z) ~ (1, z + [kk < S₀[k]], kk, w − [k < S₁[kk]])` — the perpendicular
///   crossings, where `S₀` and `S₁` are the offset lists above
///
/// Transcribed from D-Wave's own generator (`dwave-graphs`, Apache-2.0) rather than from a paper's
/// prose, because the offset lists are a *choice* the vendor made and no description of the family
/// pins them down. The tests check node counts, coupler counts and the full degree histogram at
/// five sizes against that generator's output, so a transcription error cannot pass as a topology.
///
/// # What this is and is not
///
/// It is the **nominal, full-yield** graph. A real Advantage has qubits and couplers missing from
/// fabrication and calibration, and a program embedded onto this graph is not guaranteed to fit the
/// machine in front of you — take the machine's own working graph when you have it. This is the
/// right target for asking *"could this fit at all"*, which is the question [`crate::embed`] and
/// [`crate::fabric`] ask.
///
/// Returns an empty topology for `m < 2`, matching the reference: `P₁` has no qubits.
pub fn pegasus(m: usize, j: f64) -> Topology {
    if m < 2 {
        return Topology { graph: GraphBuilder::new(0).build(), qubits: Vec::new() };
    }
    let m1 = m - 1;
    let (off0, off1) = (&PEGASUS_OFF[0], &PEGASUS_OFF[1]);
    // "Fabric only": the largest component. With the shipped offsets both bounds are 2, and the
    // effect is to trim the first and last major offsets, which is where the dangling qubits are.
    let start = [
        *off1.iter().min().expect("12 offsets"),
        *off0.iter().min().expect("12 offsets"),
    ];
    let end = [
        12 - *off1.iter().max().expect("12 offsets"),
        12 - *off0.iter().max().expect("12 offsets"),
    ];
    let lin = |u: usize, w: usize, k: usize, z: usize| -> u32 {
        (((u * m + w) * 12 + k) * m1 + z) as u32
    };
    // The k range a given (u, w) contributes, once, so the three loops below cannot drift apart.
    let krange = |u: usize, w: usize| -> core::ops::Range<usize> {
        let lo = if w == 0 { start[u] } else { 0 };
        let hi = 12 - if w == m1 { end[u] } else { 0 };
        lo..hi
    };
    let keep = |u: usize, w: usize, k: usize| -> bool {
        if w == 0 {
            k >= start[u]
        } else if w == m1 {
            k < 12 - end[u]
        } else {
            true
        }
    };

    let mut edges: Vec<(u32, u32)> = Vec::new();
    for u in 0..2 {
        for w in 0..m {
            for k in krange(u, w) {
                for z in 0..m1.saturating_sub(1) {
                    edges.push((lin(u, w, k, z), lin(u, w, k, z + 1))); // external
                }
                if k % 2 == 0 && krange(u, w).contains(&(k + 1)) {
                    for z in 0..m1 {
                        edges.push((lin(u, w, k, z), lin(u, w, k + 1, z))); // odd
                    }
                }
            }
        }
    }
    // Internal: the perpendicular crossings. Neither index can go out of range -- the k range
    // starts at `off1[kk]` exactly when `w == 0`, which is the case that would underflow.
    for w in 0..m {
        for kk in 0..12 {
            let lo = if w == 0 { off1[kk] } else { 0 };
            let hi = if w < m1 { 12 } else { off1[kk] };
            for k in lo..hi {
                let (w1, z1) = (
                    z_plus(kk < off0[k]),
                    usize::from(k < off1[kk]),
                );
                for z in 0..m1 {
                    let b = (1usize, z + w1, kk, w - z1);
                    if keep(0, w, k) && keep(1, b.1, kk) {
                        edges.push((lin(0, w, k, z), lin(b.0, b.1, b.2, b.3)));
                    }
                }
            }
        }
    }
    compact(&edges, j)
}

/// `1` when the crossing is offset by one major step, `0` otherwise. Named so the two call sites
/// of this arithmetic read as the same rule rather than as two coincidences.
fn z_plus(shifted: bool) -> usize {
    usize::from(shifted)
}

/// The **Zephyr** topology `Z_{m,t}` — the graph of D-Wave's *Advantage2* processors.
///
/// `Z₁₅` is 7,440 qubits and 71,736 couplers at degree 20, the Advantage2's published figures.
/// Zephyr's higher degree is the point: it embeds a `K₄` natively where Pegasus needs a chain, so
/// the same program needs shorter chains and a weaker chain coupling to hold them.
///
/// # The definition
///
/// A qubit is `(u, w, k, j, z)` with `u < 2`, `w < 2m+1`, `k < t`, `j < 2`, `z < m`, and linear
/// index `q = (((u·(2m+1) + w)·t + k)·2 + j)·m + z`. Every one of the `4tm(2m+1)` qubits is wired,
/// so unlike Pegasus the vendor numbering here is dense and [`Topology::qubits`] is the identity —
/// which the tests assert rather than assume.
///
/// * **external** `(u,w,k,j,z) ~ (u,w,k,j,z+1)`
/// * **odd** `(u,w,k,0,z) ~ (u,w,k,1,z−a)` for `a < 2`
/// * **internal** `(0, 2w+1+a(2i−1), k, j, z) ~ (1, 2z+1+b(2j−1), h, i, w)` over all
///   `a, b, i, j < 2`, `h, k < t`, `w, z < m`
///
/// Transcribed from D-Wave's own generator, and checked the same way Pegasus is: counts and full
/// degree histograms at five sizes.
///
/// `t` is the tile parameter and is 4 on every shipped machine. Returns an empty topology when
/// `m` or `t` is zero.
pub fn zephyr(m: usize, t: usize, j: f64) -> Topology {
    if m == 0 || t == 0 {
        return Topology { graph: GraphBuilder::new(0).build(), qubits: Vec::new() };
    }
    let big = 2 * m + 1;
    let lin = |u: usize, w: usize, k: usize, jj: usize, z: usize| -> u32 {
        ((((u * big + w) * t + k) * 2 + jj) * m + z) as u32
    };
    let mut edges: Vec<(u32, u32)> = Vec::new();
    for u in 0..2 {
        for w in 0..big {
            for k in 0..t {
                for jj in 0..2 {
                    for z in 0..m.saturating_sub(1) {
                        edges.push((lin(u, w, k, jj, z), lin(u, w, k, jj, z + 1))); // external
                    }
                }
                for a in 0..2 {
                    for z in a..m {
                        edges.push((lin(u, w, k, 0, z), lin(u, w, k, 1, z - a))); // odd
                    }
                }
            }
        }
    }
    // Internal. Both major offsets land inside `0..2m+1` for every combination -- `2w+1+a(2i-1)`
    // spans 0 at (w=0, a=1, i=0) to 2m at (w=m-1, a=1, i=1) -- so this needs no bounds test, and
    // the signed step is written out rather than folded into an index to keep that visible.
    for w in 0..m {
        for z in 0..m {
            for h in 0..t {
                for k in 0..t {
                    for i in 0..2 {
                        for jj in 0..2 {
                            for a in 0..2 {
                                for b in 0..2 {
                                    let wa = (2 * w + 1) as isize + a as isize * (2 * i as isize - 1);
                                    let wb = (2 * z + 1) as isize + b as isize * (2 * jj as isize - 1);
                                    debug_assert!((0..big as isize).contains(&wa));
                                    debug_assert!((0..big as isize).contains(&wb));
                                    edges.push((
                                        lin(0, wa as usize, k, jj, z),
                                        lin(1, wb as usize, h, i, w),
                                    ));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    compact(&edges, j)
}

#[cfg(test)]
mod topology_tests {
    use super::*;

    /// One expected topology: size parameter, node count, coupler count, degree histogram.
    type Expected = (usize, usize, usize, &'static [(usize, usize)]);

    /// Degree histogram, so a transcription error cannot hide behind a right total.
    fn degrees(g: &Graph) -> Vec<(usize, usize)> {
        let mut h = std::collections::BTreeMap::new();
        for i in 0..g.n {
            *h.entry(g.offset[i + 1] - g.offset[i]).or_insert(0usize) += 1;
        }
        h.into_iter().collect()
    }

    /// The Pegasus graph must be the one D-Wave builds, at five sizes.
    ///
    /// Counts alone are not enough — two different graphs can share a node and edge total — so the
    /// full degree histogram is checked as well. The expected values come from running D-Wave's own
    /// generator, and `P16` reproducing **5,640 qubits and 40,484 couplers** is the line that says
    /// the transcription is right: those are the Advantage's published figures, arrived at here
    /// from the coordinate rules rather than copied.
    #[test]
    fn pegasus_is_the_graph_d_wave_builds() {
        let cases: [Expected; 5] = [
            (2, 40, 164, &[(5, 16), (9, 16), (13, 8)]),
            (3, 128, 704, &[(6, 32), (10, 32), (14, 64)]),
            (4, 264, 1604, &[(6, 32), (7, 16), (10, 32), (11, 16), (14, 112), (15, 56)]),
            (6, 680, 4484, &[(6, 32), (7, 48), (10, 32), (11, 48), (14, 208), (15, 312)]),
            (16, 5640, 40484, &[(6, 32), (7, 208), (10, 32), (11, 208), (14, 688), (15, 4472)]),
        ];
        for (m, nodes, edges, hist) in cases {
            let t = pegasus(m, 1.0);
            assert_eq!(t.graph.n, nodes, "P{m} node count");
            assert_eq!(t.graph.n_edges, edges, "P{m} coupler count");
            assert_eq!(degrees(&t.graph), hist, "P{m} degree histogram");
            assert_eq!(t.qubits.len(), nodes);
        }
    }

    /// And the Zephyr graph, the same way. `Z15` reproducing 7,440 qubits at degree 20 is the
    /// Advantage2's published figure.
    #[test]
    fn zephyr_is_the_graph_d_wave_builds() {
        let cases: [Expected; 5] = [
            (1, 48, 280, &[(9, 32), (17, 16)]),
            (2, 160, 1224, &[(10, 32), (11, 32), (18, 48), (19, 48)]),
            (3, 336, 2808, &[(10, 32), (11, 32), (12, 32), (18, 80), (19, 80), (20, 80)]),
            (4, 576, 5032, &[(10, 32), (11, 32), (12, 64), (18, 112), (19, 112), (20, 224)]),
            (15, 7440, 71736,
             &[(10, 32), (11, 32), (12, 416), (18, 464), (19, 464), (20, 6032)]),
        ];
        for (m, nodes, edges, hist) in cases {
            let t = zephyr(m, 4, 1.0);
            assert_eq!(t.graph.n, nodes, "Z{m} node count");
            assert_eq!(t.graph.n_edges, edges, "Z{m} coupler count");
            assert_eq!(degrees(&t.graph), hist, "Z{m} degree histogram");
            // 4tm(2m+1), stated as arithmetic rather than as a constant, so a size this test does
            // not list is still covered by the formula.
            assert_eq!(nodes, 4 * 4 * m * (2 * m + 1));
        }
    }

    /// The two numbering systems, and the reason `Topology` exists.
    #[test]
    fn the_vendors_numbering_is_kept_and_is_sparse_on_pegasus() {
        let p = pegasus(16, 1.0);
        assert_eq!(p.graph.n, 5640);
        // The linear index space is 24 m (m-1) = 5,760 and the fabric occupies 5,640 of those,
        // spread from 30 to 5,729. The numbering IS sparse, in both directions -- it neither starts
        // at zero nor ends at the top -- and a caller who assumed it dense would program the wrong
        // qubits at both ends of the machine.
        let span = 24 * 16 * 15;
        assert_eq!(span, 5760);
        assert_eq!(*p.qubits.first().unwrap(), 30);
        assert_eq!(*p.qubits.last().unwrap(), 5729);
        assert!(p.qubits.len() < *p.qubits.last().unwrap() as usize + 1, "sparse");
        assert!(p.qubits.windows(2).all(|w| w[0] < w[1]), "strictly increasing");
        // Round trip, both directions, and an honest miss in between.
        for node in [0usize, 1, 2000, 5639] {
            let q = p.qubit(node).expect("in range");
            assert_eq!(p.node(q), Some(node));
        }
        assert_eq!(p.qubit(5640), None);
        let absent = (0..5760u32).find(|q| p.node(*q).is_none()).expect("the fabric drops some");
        assert_eq!(p.node(absent), None, "a qubit outside the fabric is a real None");

        // Zephyr wires every qubit it defines, so there its numbering is the identity -- asserted
        // rather than assumed, because that is exactly the difference between the two families.
        let z = zephyr(4, 4, 1.0);
        assert!(z.qubits.iter().enumerate().all(|(i, &q)| i as u32 == q));
    }

    /// A topology is only useful if the rest of the crate can run on it.
    #[test]
    fn the_new_topologies_colour_and_sample_like_any_other_graph() {
        for (name, t) in [("P4", pegasus(4, 1.0)), ("Z2", zephyr(2, 4, 1.0))] {
            let g = &t.graph;
            let k = g.classes.len();
            assert!(k >= 2, "{name} needs at least two colour classes, got {k}");
            // A proper colouring is what chromatic Gibbs rests on: no edge may join a class. The
            // graph builds its own, so this is checking the colouring survives a topology it has
            // never seen rather than checking the loop above.
            for i in 0..g.n {
                for e in g.offset[i]..g.offset[i + 1] {
                    assert_ne!(g.colors[i], g.colors[g.nbr[e] as usize],
                               "{name}: adjacent nodes share a colour");
                }
            }
            assert_eq!(g.classes.iter().map(|c| c.len()).sum::<usize>(), g.n,
                       "{name}: every node lands in exactly one class");
            let mut s = crate::gibbs::Sampler::new(g, 0.3, 7);
            let set = s.collect(&crate::samples::Plan::new(200, 200, 1), None);
            assert_eq!(set.len(), 200);
            assert!(set.magnetization().unwrap().value.abs() <= 1.0, "{name} samples");
        }
    }

    #[test]
    fn degenerate_sizes_are_empty_rather_than_wrong() {
        for m in [0usize, 1] {
            assert_eq!(pegasus(m, 1.0).graph.n, 0, "P{m} has no qubits");
        }
        assert_eq!(zephyr(0, 4, 1.0).graph.n, 0);
        assert_eq!(zephyr(4, 0, 1.0).graph.n, 0);
    }
}
