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
