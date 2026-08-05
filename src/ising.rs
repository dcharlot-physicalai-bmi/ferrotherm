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

/// Exact Boltzmann distribution over all 2^n states (n <= 24). Returns probabilities indexed by
/// bitmask (bit b set => spin b = +1).
pub fn exact_boltzmann(g: &Graph, beta: f64) -> Vec<f64> {
    assert!(g.n <= 24, "exact enumeration limited to 24 spins");
    let m = 1usize << g.n;
    let mut p = vec![0.0f64; m];
    let mut z = 0.0;
    let mut s = vec![-1i8; g.n];
    for mask in 0..m {
        for b in 0..g.n {
            s[b] = if mask >> b & 1 == 1 { 1 } else { -1 };
        }
        let w = (-beta * g.energy(&s)).exp();
        p[mask] = w;
        z += w;
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
    p.iter().zip(q).map(|(a, b)| (a - b).abs()).sum::<f64>() / 2.0
}
