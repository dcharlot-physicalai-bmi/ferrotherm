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

/// Exact Boltzmann distribution over all 2^n states (n <= 24). Returns probabilities indexed by
/// bitmask (bit b set => spin b = +1).
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
