//! The `[n, k]` interaction layout the CPU, GPU and hardware emitters share.
//!
//! The CSR form in [`crate::graph::Graph`] is right for a CPU walking a variable-degree graph. A
//! GPU workgroup and an RTL datapath both want the opposite: a rectangle, one row per node, every
//! row the same width, so a lane's work is known before it starts. This is that rectangle.
//!
//! **On padding.** Rows are padded to the maximum degree, and the padding is described by an
//! explicit `active` mask. THRML pads with a sentinel value instead — `INVALID_BIAS = -1e10` — and
//! a sentinel is a number that is fine until someone does arithmetic on it, at which point it
//! silently dominates every real term in the sum. The mask here cannot do that, and the padding is
//! deliberately inert twice over: padded slots carry weight `0.0` *and* `active = 0`, so a kernel
//! that forgot the mask entirely still computes the right field. Defence in depth costs one `f64`
//! of zero per padded slot and removes a whole class of silent wrongness.

use crate::graph::Graph;

/// A graph as a dense `[n, k]` rectangle plus a per-slot active mask.
#[derive(Clone, Debug)]
pub struct Padded {
    /// Nodes.
    pub n: usize,
    /// Row width: the maximum degree in the graph, so every row is `k` wide.
    pub k: usize,
    /// Neighbour index per slot, `n * k`. Padded slots point at themselves, which is harmless
    /// because their weight is zero and their mask is clear.
    pub nbr: Vec<u32>,
    /// Coupling per slot, `n * k`. Padded slots are exactly `0.0`.
    pub w: Vec<f64>,
    /// 1 for a real coupling, 0 for padding, `n * k`.
    pub active: Vec<u8>,
    /// Per-node bias, `n`.
    pub h: Vec<f64>,
}

impl Padded {
    pub fn from_graph(g: &Graph) -> Padded {
        let n = g.n;
        let k = g.max_degree().max(1);
        let mut nbr = vec![0u32; n * k];
        let mut w = vec![0.0f64; n * k];
        let mut active = vec![0u8; n * k];
        for i in 0..n {
            // padded slots point at their own row so an ignored mask reads a defined index
            for slot in 0..k {
                nbr[i * k + slot] = i as u32;
            }
            for (slot, kk) in (g.offset[i]..g.offset[i + 1]).enumerate() {
                nbr[i * k + slot] = g.nbr[kk];
                w[i * k + slot] = g.w[kk];
                active[i * k + slot] = 1;
            }
        }
        Padded { n, k, nbr, w, active, h: g.h.clone() }
    }

    /// Local field at node `i`: `sum_j J_ij s_j + h_i`.
    pub fn field(&self, i: usize, s: &[i8]) -> f64 {
        let mut f = self.h[i];
        for slot in 0..self.k {
            let t = i * self.k + slot;
            f += self.w[t] * s[self.nbr[t] as usize] as f64;
        }
        f
    }

    /// Field computed while honouring the mask, for a kernel that branches on it.
    pub fn field_masked(&self, i: usize, s: &[i8]) -> f64 {
        let mut f = self.h[i];
        for slot in 0..self.k {
            let t = i * self.k + slot;
            if self.active[t] != 0 {
                f += self.w[t] * s[self.nbr[t] as usize] as f64;
            }
        }
        f
    }

    /// Fraction of slots that are real. Low occupancy means a rectangle is the wrong shape for this
    /// graph and the caller is paying for padding it will never use.
    pub fn occupancy(&self) -> f64 {
        if self.active.is_empty() {
            return 1.0;
        }
        self.active.iter().filter(|&&a| a != 0).count() as f64 / self.active.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::GraphBuilder;
    use crate::rng::Pcg;

    fn random_graph(n: usize, p: f64, seed: u64) -> Graph {
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
    fn the_field_matches_the_csr_form() {
        // If these two disagree, the GPU and the CPU are sampling different models.
        for (n, p, seed) in [(12, 0.3, 1), (40, 0.15, 2), (64, 0.5, 3), (7, 1.0, 4)] {
            let g = random_graph(n, p, seed);
            let d = Padded::from_graph(&g);
            let mut rng = Pcg::new(seed, 99);
            for _ in 0..20 {
                let s: Vec<i8> = (0..n).map(|_| if rng.f64() < 0.5 { 1 } else { -1 }).collect();
                for i in 0..n {
                    let a = g.field(i, &s);
                    assert!((a - d.field(i, &s)).abs() < 1e-12, "n={n} i={i}");
                    assert!((a - d.field_masked(i, &s)).abs() < 1e-12, "masked n={n} i={i}");
                }
            }
        }
    }

    #[test]
    fn padding_is_inert_even_if_the_mask_is_ignored() {
        // The contrast with a -1e10 sentinel: forgetting the mask must be survivable.
        let g = random_graph(30, 0.1, 7); // ragged degrees, so there is real padding
        let d = Padded::from_graph(&g);
        assert!(d.occupancy() < 1.0, "this test needs a graph with padding");

        for t in 0..d.n * d.k {
            if d.active[t] == 0 {
                assert_eq!(d.w[t], 0.0, "a padded slot must weigh exactly zero");
                assert!((d.nbr[t] as usize) < d.n, "a padded slot must still index in range");
            }
        }
        // field() ignores the mask entirely and still agrees with the CSR truth
        let s: Vec<i8> = (0..d.n).map(|i| if i % 3 == 0 { 1 } else { -1 }).collect();
        for i in 0..d.n {
            assert!((g.field(i, &s) - d.field(i, &s)).abs() < 1e-12);
        }
    }

    #[test]
    fn a_sentinel_would_have_destroyed_the_field() {
        // Demonstrating why this design choice is not fussiness. Substituting THRML's sentinel into
        // the padding and ignoring the mask makes the field meaningless.
        let g = random_graph(30, 0.1, 7);
        let mut d = Padded::from_graph(&g);
        for t in 0..d.n * d.k {
            if d.active[t] == 0 {
                d.w[t] = -1e10;
            }
        }
        let s: Vec<i8> = (0..d.n).map(|_| 1).collect();
        let ruined = (0..d.n).any(|i| (g.field(i, &s) - d.field(i, &s)).abs() > 1.0);
        assert!(ruined, "a sentinel in the padding should wreck an unmasked field, and it did");
    }

    #[test]
    fn a_regular_graph_pads_to_nothing() {
        // A lattice has uniform degree, so the rectangle is exactly the graph.
        let g = crate::ising::lattice2d(8, 1.0);
        let d = Padded::from_graph(&g);
        assert_eq!(d.k, 4);
        assert_eq!(d.occupancy(), 1.0);
    }

    #[test]
    fn occupancy_reports_what_the_rectangle_costs() {
        // One high-degree node forces every row wide; the caller deserves to know.
        let mut b = GraphBuilder::new(20);
        for j in 1..20 {
            b.couple(0, j, 1.0); // a star
        }
        let d = Padded::from_graph(&b.build());
        assert_eq!(d.k, 19);
        assert!(d.occupancy() < 0.11, "a star should pad badly, got {}", d.occupancy());
    }
}
