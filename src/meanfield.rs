//! Mean-field, TAP and belief propagation — the fast approximations, and the one that is a bound.
//!
//! Every sampler in this crate answers by drawing; these answer by iterating a few equations, and
//! two of them come with a theorem attached. They are the baseline the p-bit and neuromorphic
//! communities measure against, and the crate had none of them.
//!
//! # Naive mean field, and the Gibbs–Bogoliubov inequality
//!
//! For ANY product distribution `q(s) = Π_i (1 + m_i s_i)/2`,
//!
//! ```text
//!   ln Z  ≥  β Σ_i h_i m_i + β Σ_{i<j} J_ij m_i m_j + Σ_i H(m_i),
//!   H(m) = −[(1+m)/2 · ln((1+m)/2) + (1−m)/2 · ln((1−m)/2)]
//! ```
//!
//! — Jensen's inequality applied to `Z = Σ_s q(s) · e^{−βE(s)}/q(s)`. It holds at every `m`, not
//! only at the self-consistent fixed point `m_i = tanh(β(h_i + Σ_j J_ij m_j))`; the fixed point is
//! simply where the bound is tightest within the family. So [`gibbs_bogoliubov`] is a
//! **deterministic lower bound on `ln Z`** with no sampling and no probability of failure — the
//! fourth member of the family in [`crate::free_energy`], beside AIS's probabilistic bound, TI's
//! bracket and BAR's estimate — and the tests hold it against exact `ln Z` as a strict inequality.
//!
//! # TAP
//!
//! Thouless–Anderson–Palmer add the Onsager reaction term, `m_i = tanh(β(h_i + Σ_j J_ij m_j −
//! β m_i Σ_j J_ij² (1 − m_j²)))`, and the Plefka expansion's second-order free energy. Not a bound;
//! exact for the Sherrington–Kirkpatrick model in the thermodynamic limit above its transition,
//! and measured here to beat naive mean field on a small SK sample at high temperature.
//!
//! # Belief propagation, and the Bethe free energy
//!
//! Sum-product in cavity-field form: `u_{j→i} = (1/β) atanh(tanh(βJ_ij) tanh(β(h_j + Σ_{k∈∂j∖i}
//! u_{k→j})))`, marginals `m_i = tanh(β(h_i + Σ_j u_{j→i}))`. **Exact on trees**, where the
//! Bethe free energy equals `ln Z` and the marginals are the true ones — which is what the tests
//! check, against [`crate::exact::Elimination`], to `1e-9`. On graphs with loops it is an
//! approximation whose quality the crate can now measure rather than assume.

use crate::graph::Graph;

/// The Gibbs–Bogoliubov lower bound on `ln Z(β)` at magnetisations `m` (each in `(−1, 1)`).
///
/// A theorem for every `m`, so a caller may pass anything; [`naive_mean_field`] passes the fixed
/// point, where it is tightest. Entropy terms use `x ln x → 0` at the endpoints.
pub fn gibbs_bogoliubov(g: &Graph, beta: f64, m: &[f64]) -> f64 {
    assert_eq!(m.len(), g.n);
    let mut energy = 0.0; // −⟨E⟩_q = Σ h m + Σ_{i<j} J m m
    let mut entropy = 0.0;
    for i in 0..g.n {
        energy += g.h[i] * m[i];
        for e in g.offset[i]..g.offset[i + 1] {
            let j = g.nbr[e] as usize;
            if j > i {
                energy += g.w[e] * m[i] * m[j];
            }
        }
        entropy += binary_entropy(m[i]);
    }
    beta * energy + entropy
}

fn binary_entropy(m: f64) -> f64 {
    let xlnx = |x: f64| if x <= 0.0 { 0.0 } else { x * x.ln() };
    -(xlnx((1.0 + m) / 2.0) + xlnx((1.0 - m) / 2.0))
}

/// What a mean-field iteration produced.
#[derive(Clone, Debug)]
pub struct MeanField {
    pub beta: f64,
    /// Magnetisations at the last iterate.
    pub m: Vec<f64>,
    /// The Gibbs–Bogoliubov bound at `m` (naive), or the TAP/Plefka free energy at `m` (TAP) —
    /// see the constructor's doc for which.
    pub log_z: f64,
    /// Largest change in any `m_i` on the last iteration.
    pub residual: f64,
    pub iterations: usize,
}

impl MeanField {
    pub fn converged(&self, tol: f64) -> bool {
        self.residual < tol
    }
}

/// Naive mean field: damped iteration of `m_i = tanh(β(h_i + Σ_j J_ij m_j))` from `m = 0.01`,
/// returning the Gibbs–Bogoliubov **lower bound** on `ln Z` at the last iterate.
pub fn naive_mean_field(g: &Graph, beta: f64, iters: usize, damping: f64) -> MeanField {
    let mut m = vec![0.01; g.n];
    let mut residual = f64::INFINITY;
    let mut it = 0;
    while it < iters && residual > 1e-13 {
        residual = 0.0;
        for i in 0..g.n {
            let mut field = g.h[i];
            for e in g.offset[i]..g.offset[i + 1] {
                field += g.w[e] * m[g.nbr[e] as usize];
            }
            let new = (beta * field).tanh();
            let next = damping * m[i] + (1.0 - damping) * new;
            residual = residual.max((next - m[i]).abs());
            m[i] = next;
        }
        it += 1;
    }
    let log_z = gibbs_bogoliubov(g, beta, &m);
    MeanField { beta, m, log_z, residual, iterations: it }
}

/// TAP: naive mean field with the Onsager reaction term, returning the second-order Plefka free
/// energy `ln Z_TAP = ln Z_MF(m) + (β²/2) Σ_{i<j} J_ij² (1 − m_i²)(1 − m_j²)`. Not a bound.
pub fn tap(g: &Graph, beta: f64, iters: usize, damping: f64) -> MeanField {
    let mut m = vec![0.01; g.n];
    let mut residual = f64::INFINITY;
    let mut it = 0;
    while it < iters && residual > 1e-13 {
        residual = 0.0;
        for i in 0..g.n {
            let mut field = g.h[i];
            let mut reaction = 0.0;
            for e in g.offset[i]..g.offset[i + 1] {
                let j = g.nbr[e] as usize;
                field += g.w[e] * m[j];
                reaction += g.w[e] * g.w[e] * (1.0 - m[j] * m[j]);
            }
            let new = (beta * (field - beta * m[i] * reaction)).tanh();
            let next = damping * m[i] + (1.0 - damping) * new;
            residual = residual.max((next - m[i]).abs());
            m[i] = next;
        }
        it += 1;
    }
    let mut onsager = 0.0;
    for i in 0..g.n {
        for e in g.offset[i]..g.offset[i + 1] {
            let j = g.nbr[e] as usize;
            if j > i {
                onsager += g.w[e] * g.w[e] * (1.0 - m[i] * m[i]) * (1.0 - m[j] * m[j]);
            }
        }
    }
    let log_z = gibbs_bogoliubov(g, beta, &m) + 0.5 * beta * beta * onsager;
    MeanField { beta, m, log_z, residual, iterations: it }
}

/// What belief propagation produced.
#[derive(Clone, Debug)]
pub struct Bethe {
    pub beta: f64,
    /// BP marginals `⟨s_i⟩`.
    pub m: Vec<f64>,
    /// The Bethe free energy as `ln Z_Bethe`; equal to `ln Z` on a tree.
    pub log_z: f64,
    /// Largest message change on the last iteration.
    pub residual: f64,
    pub iterations: usize,
}

impl Bethe {
    pub fn converged(&self, tol: f64) -> bool {
        self.residual < tol
    }
}

/// Loopy belief propagation in cavity-field form, damped, from zero messages.
///
/// Messages live on directed edges, indexed by the adjacency slot `e` (which stores `i → nbr[e]`
/// as the message `nbr[e] → i`). Returns marginals and the Bethe free energy computed from the
/// converged beliefs:
///
/// ```text
///   ln Z_Bethe = Σ_i (1 − d_i) ln Z_i + Σ_{(ij)} ln Z_ij,
///   Z_i  = Σ_s exp(β H_i s),                       H_i = h_i + Σ_{k∈∂i} u_{k→i}
///   Z_ij = Σ_{s,t} exp(β(J_ij s t + H_i^{∖j} s + H_j^{∖i} t)),   H_i^{∖j} = H_i − u_{j→i}.
/// ```
pub fn belief_propagation(g: &Graph, beta: f64, iters: usize, damping: f64) -> Bethe {
    let ne = g.nbr.len();
    let mut u = vec![0.0f64; ne]; // u[e] = message from nbr[e] into the owner of slot e
    let owner: Vec<usize> = (0..g.n).flat_map(|i| core::iter::repeat_n(i, g.offset[i + 1] - g.offset[i])).collect();
    // reverse slot: the slot in nbr[e]'s list that points back at owner(e)
    let rev: Vec<usize> = (0..ne)
        .map(|e| {
            let (i, j) = (owner[e], g.nbr[e] as usize);
            (g.offset[j]..g.offset[j + 1]).find(|&f| g.nbr[f] as usize == i).expect("symmetric adjacency")
        })
        .collect();
    let cavity = |u: &[f64], i: usize, exclude: usize| -> f64 {
        let mut hh = g.h[i];
        for e in g.offset[i]..g.offset[i + 1] {
            if e != exclude {
                hh += u[e];
            }
        }
        hh
    };
    let mut residual = f64::INFINITY;
    let mut it = 0;
    while it < iters && residual > 1e-14 {
        residual = 0.0;
        for e in 0..ne {
            // message from j = nbr[e] into i = owner[e]: needs j's cavity field excluding i, which
            // is j's slot rev[e].
            let (i, j) = (owner[e], g.nbr[e] as usize);
            let _ = i;
            let hj = cavity(&u, j, rev[e]);
            let new = ((beta * g.w[e]).tanh() * (beta * hj).tanh()).atanh() / beta;
            let next = damping * u[e] + (1.0 - damping) * new;
            residual = residual.max((next - u[e]).abs());
            u[e] = next;
        }
        it += 1;
    }
    let mut m = vec![0.0; g.n];
    let mut log_z = 0.0;
    for i in 0..g.n {
        let hi = cavity(&u, i, usize::MAX);
        m[i] = (beta * hi).tanh();
        let d = (g.offset[i + 1] - g.offset[i]) as f64;
        log_z += (1.0 - d) * (2.0 * (beta * hi).cosh()).ln();
        for e in g.offset[i]..g.offset[i + 1] {
            let j = g.nbr[e] as usize;
            if j > i {
                let hij = hi - u[e];
                let hji = cavity(&u, j, rev[e]);
                let mut z = 0.0;
                for s in [-1.0, 1.0] {
                    for t in [-1.0, 1.0] {
                        z += (beta * (g.w[e] * s * t + hij * s + hji * t)).exp();
                    }
                }
                log_z += z.ln();
            }
        }
    }
    Bethe { beta, m, log_z, residual, iterations: it }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exact::Elimination;
    use crate::free_energy::exact_log_z;
    use crate::graph::GraphBuilder;
    use crate::ising;
    use crate::rng::Pcg;

    fn random_tree(n: usize, seed: u64) -> Graph {
        let mut rng = Pcg::new(seed, 0);
        let mut gb = GraphBuilder::new(n);
        for i in 1..n {
            let parent = (rng.f64() * i as f64) as usize;
            gb.couple(parent, i, 2.0 * rng.f64() - 1.0);
        }
        for i in 0..n {
            gb.bias(i, rng.f64() - 0.5);
        }
        gb.build()
    }

    /// The inequality holds strictly at random magnetisations and at the fixed point.
    #[test]
    fn gibbs_bogoliubov_is_a_lower_bound_everywhere() {
        let mut rng = Pcg::new(5, 0);
        for (g, beta) in [
            (ising::ring(12, 1.0, 0.2), 0.7),
            (ising::lattice2d(4, 1.0), 0.5),
            (random_tree(14, 3), 1.3),
        ] {
            let truth = exact_log_z(&g, beta);
            for _ in 0..20 {
                let m: Vec<f64> = (0..g.n).map(|_| 1.98 * rng.f64() - 0.99).collect();
                let b = gibbs_bogoliubov(&g, beta, &m);
                assert!(b <= truth + 1e-12, "bound {b} above ln Z {truth}");
            }
            let mf = naive_mean_field(&g, beta, 2000, 0.3);
            assert!(mf.log_z <= truth + 1e-12, "fixed-point bound {} above ln Z {truth}", mf.log_z);
            // and at m = 0 the bound is n ln 2 − 0, which is only ln Z at β = 0: below it here.
            assert!(gibbs_bogoliubov(&g, beta, &vec![0.0; g.n]) < truth);
        }
    }

    /// On a tree, belief propagation is exact: ln Z and every marginal, to 1e-9.
    #[test]
    fn belief_propagation_is_exact_on_trees() {
        for seed in 0..5u64 {
            let g = random_tree(20, seed);
            let beta = 0.9;
            let ex = Elimination::default();
            let truth = ex.log_partition(&g, beta).unwrap().log_z.unwrap();
            let marg = ex.marginals(&g, beta).unwrap(); // P(s_i = +1)
            let bp = belief_propagation(&g, beta, 500, 0.0);
            assert!(bp.converged(1e-12), "seed {seed}: residual {}", bp.residual);
            assert!((bp.log_z - truth).abs() < 1e-9, "seed {seed}: Bethe {} vs exact {truth}", bp.log_z);
            for i in 0..g.n {
                let m_true = 2.0 * marg[i] - 1.0;
                assert!((bp.m[i] - m_true).abs() < 1e-9, "seed {seed} site {i}: BP {} vs {m_true}", bp.m[i]);
            }
        }
    }

    /// With loops BP is approximate, and the crate can say by how much.
    #[test]
    fn belief_propagation_on_a_loop_is_close_but_not_exact() {
        let g = ising::lattice2d(4, 1.0);
        let beta = 0.3;
        let truth = exact_log_z(&g, beta);
        let bp = belief_propagation(&g, beta, 2000, 0.5);
        assert!(bp.converged(1e-10));
        let err = (bp.log_z - truth).abs();
        assert!(err < 0.5 && err > 1e-6, "Bethe error {err} on a 4x4 torus at beta 0.3");
    }

    /// TAP beats naive mean field on a small SK sample at high temperature, in ln Z and in m.
    #[test]
    fn tap_improves_on_naive_mean_field_for_sk() {
        let n = 16;
        let mut rng = Pcg::new(11, 0);
        let mut gb = GraphBuilder::new(n);
        for i in 0..n {
            for j in (i + 1)..n {
                gb.couple(i, j, (if rng.f64() < 0.5 { -1.0 } else { 1.0 }) / (n as f64).sqrt());
            }
            gb.bias(i, 0.3 * (rng.f64() - 0.5));
        }
        let g = gb.build();
        let beta = 0.5;
        let truth = exact_log_z(&g, beta);
        let marg = Elimination { max_width: 24 }.marginals(&g, beta).unwrap();
        let mf = naive_mean_field(&g, beta, 5000, 0.5);
        let tp = tap(&g, beta, 5000, 0.5);
        assert!(mf.converged(1e-10) && tp.converged(1e-10));
        assert!((tp.log_z - truth).abs() < (mf.log_z - truth).abs(), "TAP {} MF {} exact {truth}", tp.log_z, mf.log_z);
        let err = |m: &[f64]| (0..n).map(|i| (m[i] - (2.0 * marg[i] - 1.0)).abs()).fold(0.0, f64::max);
        assert!(err(&tp.m) < err(&mf.m), "TAP marginal error {} vs MF {}", err(&tp.m), err(&mf.m));
    }
}
