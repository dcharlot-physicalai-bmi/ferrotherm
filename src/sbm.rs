//! Simulated bifurcation — the Toshiba Ising-machine algorithm line (Goto et al.; ballistic bSB
//! and discrete dSB per the 2021 Science Advances formulation), as portable deterministic Rust.
//!
//! Classical mechanics rather than sampling: each spin is a particle x_i in [-1, 1] with momentum
//! y_i under a bifurcating potential ramped by a(t): 0 -> a0, coupled through the Ising J. The
//! symplectic (momentum-first) update with perfectly inelastic walls at |x| = 1:
//!     y <- y + { -(a0 - a(t)) x + c0 * force } dt ;   x <- x + a0 y dt ;
//!     wall: |x| > 1  =>  x = sgn(x), y = 0.
//! bSB uses force_i = sum_j J_ij x_j; dSB uses force_i = sum_j J_ij sgn(x_j) (the discretisation
//! is what suppresses analog error). Read out sgn(x) EVERY step and keep the best-so-far — the
//! trajectory is ergodic and the final state is not the best visited. c0 = 0.5 / (J_rms sqrt(N)).

use crate::graph::Graph;
use crate::rng::Pcg;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Variant {
    Ballistic,
    Discrete,
}

fn sgn(x: f64) -> f64 {
    if x < 0.0 {
        -1.0
    } else {
        1.0 // sgn(0) = +1, fixed for determinism
    }
}

fn spins_of(x: &[f64]) -> Vec<i8> {
    x.iter().map(|&v| if v < 0.0 { -1 } else { 1 }).collect()
}

/// One simulated-bifurcation run. Returns (best spins, best energy).
pub fn run(g: &Graph, variant: Variant, n_steps: usize, dt: f64, seed: u64) -> (Vec<i8>, f64) {
    let n = g.n;
    let a0 = 1.0f64;
    // c0 = 0.5 / (J_rms * sqrt(N)), J_rms over ordered node pairs (CSR stores both directions)
    let sum_j2: f64 = g.w.iter().map(|&w| w * w).sum();
    let j_rms = (sum_j2 / (n as f64 * (n as f64 - 1.0))).sqrt();
    let c0 = if j_rms > 0.0 { 0.5 / (j_rms * (n as f64).sqrt()) } else { 0.5 };

    let mut rng = Pcg::new(seed, 0x5B);
    // small random x-init in addition to y: with x = 0 exactly, symmetric graphs drive all
    // particles identically, every site hits the wall on the same step, and the wall's y = 0
    // reset erases the y-asymmetry — a synchronized trap (measured on K8 antiferromagnetic:
    // dSB converged to the WORST state). Breaking spatial symmetry at init removes it.
    let mut x: Vec<f64> = (0..n).map(|_| (rng.f64() - 0.5) * 0.02).collect();
    let mut y: Vec<f64> = (0..n).map(|_| (rng.f64() - 0.5) * 0.2).collect();
    let mut best = spins_of(&x);
    let mut best_e = g.energy(&best);
    let mut force = vec![0.0f64; n];
    for k in 0..n_steps {
        let a_t = a0 * k as f64 / n_steps as f64;
        // force from the PRE-step x (separate pass = full-copy semantics)
        for i in 0..n {
            let mut s = 0.0;
            for e in g.offset[i]..g.offset[i + 1] {
                let xj = x[g.nbr[e] as usize];
                s += g.w[e] * if variant == Variant::Discrete { sgn(xj) } else { xj };
            }
            force[i] = s + g.h[i];
        }
        for i in 0..n {
            y[i] += (-(a0 - a_t) * x[i] + c0 * force[i]) * dt;
            x[i] += a0 * y[i] * dt;
            if x[i].abs() > 1.0 {
                x[i] = sgn(x[i]);
                y[i] = 0.0;
            }
        }
        let s = spins_of(&x);
        let e = g.energy(&s);
        if e < best_e {
            best_e = e;
            best = s;
        }
    }
    (best, best_e)
}

/// Multi-restart wrapper: `restarts` seeded runs, best result kept.
pub fn run_restarts(
    g: &Graph,
    variant: Variant,
    n_steps: usize,
    dt: f64,
    seed: u64,
    restarts: usize,
) -> (Vec<i8>, f64) {
    let mut best: Option<(Vec<i8>, f64)> = None;
    for r in 0..restarts {
        let (s, e) = run(g, variant, n_steps, dt, seed ^ (r as u64).wrapping_mul(0x9E3779B97F4A7C15));
        if best.as_ref().is_none_or(|(_, be)| e < *be) {
            best = Some((s, e));
        }
    }
    best.unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::GraphBuilder;

    fn exact_ground(g: &Graph) -> f64 {
        let n = g.n;
        assert!(n <= 20);
        let mut e0 = f64::MAX;
        let mut s = vec![-1i8; n];
        for m in 0..(1u32 << n) {
            for b in 0..n {
                s[b] = if m >> b & 1 == 1 { 1 } else { -1 };
            }
            let e = g.energy(&s);
            if e < e0 {
                e0 = e;
            }
        }
        e0
    }

    /// Wall rule: overshoot lands exactly on the wall with zero momentum.
    #[test]
    fn wall_rule() {
        let mut x = 1.7f64;
        let mut y = 0.9f64;
        if x.abs() > 1.0 {
            x = sgn(x);
            y = 0.0;
        }
        assert_eq!(x, 1.0);
        assert_eq!(y, 0.0);
    }

    /// Closed-form graphs: K8 (all antiferromagnetic), C7 ring, Petersen — both variants must
    /// return the exhaustively-enumerated ground-state energy.
    #[test]
    fn closed_form_graphs_reach_ground_state() {
        let mut graphs: Vec<Graph> = Vec::new();
        let mut gb = GraphBuilder::new(8);
        for i in 0..8 {
            for j in (i + 1)..8 {
                gb.couple(i, j, -1.0);
            }
        }
        graphs.push(gb.build());
        let mut gb = GraphBuilder::new(7);
        for i in 0..7 {
            gb.couple(i, (i + 1) % 7, -1.0);
        }
        graphs.push(gb.build());
        let mut gb = GraphBuilder::new(10);
        let outer = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)];
        let inner = [(5, 7), (7, 9), (9, 6), (6, 8), (8, 5)];
        let spokes = [(0, 5), (1, 6), (2, 7), (3, 8), (4, 9)];
        for &(a, b) in outer.iter().chain(&inner).chain(&spokes) {
            gb.couple(a, b, -1.0);
        }
        graphs.push(gb.build());

        for (gi, g) in graphs.iter().enumerate() {
            let e0 = exact_ground(g);
            for (vi, v) in [Variant::Ballistic, Variant::Discrete].into_iter().enumerate() {
                let (_, e) = run_restarts(g, v, 2000, 1.0, 0x5B00 + gi as u64, 10);
                assert!(
                    (e - e0).abs() < 1e-9,
                    "graph {gi} variant {vi}: found {e} vs exact ground {e0}"
                );
            }
        }
    }

    /// Random Gaussian-J instances at N = 16: near-perfect ground-state hit rates with frozen
    /// seeds (bSB >= 18/20, dSB >= 19/20 per the published tuning).
    #[test]
    fn random_instances_hit_rate() {
        let mut hits_b = 0;
        let mut hits_d = 0;
        for inst in 0..20u64 {
            let mut rng = Pcg::new(0x6A55 ^ inst, 7);
            let mut gb = GraphBuilder::new(16);
            for i in 0..16 {
                for j in (i + 1)..16 {
                    let a = rng.f64().max(1e-12);
                    let b = rng.f64();
                    let gauss = (-2.0 * a.ln()).sqrt() * (std::f64::consts::TAU * b).cos();
                    gb.couple(i, j, gauss);
                }
            }
            let g = gb.build();
            let e0 = exact_ground(&g);
            let (_, eb) = run_restarts(&g, Variant::Ballistic, 2000, 1.0, 0xB0 ^ inst, 10);
            let (_, ed) = run_restarts(&g, Variant::Discrete, 2000, 1.0, 0xD0 ^ inst, 10);
            if (eb - e0).abs() < 1e-9 {
                hits_b += 1;
            }
            if (ed - e0).abs() < 1e-9 {
                hits_d += 1;
            }
        }
        assert!(hits_b >= 18, "bSB ground-state hits {hits_b}/20");
        assert!(hits_d >= 19, "dSB ground-state hits {hits_d}/20");
    }
}
