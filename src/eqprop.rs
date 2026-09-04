//! Equilibrium propagation for Boltzmann machines — learning from two nearby equilibria.
//!
//! Scellier & Bengio (2017) train an energy-based model with no backward pass: relax to the free
//! equilibrium with the inputs clamped, relax again with the outputs *nudged* toward the target by
//! a small `β`, and update every parameter by the difference of the two equilibria's statistics
//! divided by `β`. The theorem is that this converges to the gradient of the loss as `β → 0`.
//!
//! For a Boltzmann machine at unit temperature the statement is a linear-response identity, and
//! it is exact enough to be an oracle. With `E(s) = −Σ h_i s_i − Σ J_ij s_i s_j`, inputs clamped,
//! loss `ℓ(s)` on the output spins, and the nudged energy `E + β ℓ`,
//!
//! ```text
//!   d⟨ℓ⟩₀/dJ_ij = Cov₀(ℓ, s_i s_j) = lim_{β→0} ( ⟨s_i s_j⟩₀ − ⟨s_i s_j⟩_β ) / β,
//!   d⟨ℓ⟩₀/dh_i  = Cov₀(ℓ, s_i)     = lim_{β→0} ( ⟨s_i⟩₀     − ⟨s_i⟩_β     ) / β,
//! ```
//!
//! with the one-sided quotient in error by `O(β)` and the centered one,
//! `(⟨·⟩_{−β} − ⟨·⟩_{+β}) / 2β`, by `O(β²)` (Laborieux et al. 2021). The tests compute every term
//! by enumeration on small machines and check the two rates — halving `β` halves one error and
//! quarters the other — then check the sampled version, using the crate's Gibbs sampler with the
//! nudge applied as a field on the outputs, against the exact gradient within its error bars.
//!
//! Why it belongs here: it is a learning rule whose only primitive is *sample two nearby Boltzmann
//! distributions*, which is what a thermodynamic fabric does natively and what a GPU does not.

use crate::gibbs::Sampler;
use crate::graph::{Graph, GraphBuilder};

/// A supervised task on a Boltzmann machine: which spins are inputs, which are outputs.
#[derive(Clone, Debug)]
pub struct Task {
    pub inputs: Vec<usize>,
    pub outputs: Vec<usize>,
}

/// Hamming loss `ℓ(s) = Σ_o (1 − s_o t_o) / 2`, in units of wrong output spins.
pub fn hamming_loss(task: &Task, s: &[i8], target: &[i8]) -> f64 {
    task.outputs.iter().zip(target).map(|(&o, &t)| (1.0 - (s[o] as i32 * t as i32) as f64) / 2.0).sum()
}

/// Gradient of the expected loss with respect to every coupling (per edge slot, `i < j` once) and
/// bias, in the layout `(couplings, biases)` where couplings follow `pairs`.
#[derive(Clone, Debug)]
pub struct Gradient {
    pub pairs: Vec<(usize, usize)>,
    pub d_couplings: Vec<f64>,
    pub d_biases: Vec<f64>,
}

fn pairs_of(g: &Graph) -> Vec<(usize, usize)> {
    let mut v = Vec::with_capacity(g.n_edges);
    for i in 0..g.n {
        for e in g.offset[i]..g.offset[i + 1] {
            let j = g.nbr[e] as usize;
            if j > i {
                v.push((i, j));
            }
        }
    }
    v
}

/// The graph with `input` spins fixed to `x` (their biases become large) is the wrong tool; the
/// crate clamps in the sampler. For enumeration we instead sum only over states agreeing with `x`.
fn enumerate_moments(g: &Graph, task: &Task, x: &[i8], target: &[i8], nudge: f64) -> (f64, Vec<f64>, Vec<f64>) {
    // returns (⟨ℓ⟩, ⟨s_i s_j⟩ per pair, ⟨s_i⟩ per site) under exp(−(E + nudge·ℓ)), inputs clamped.
    assert!(g.n <= 20);
    let pairs = pairs_of(g);
    let mut s = vec![-1i8; g.n];
    let mut logs = Vec::new();
    let mut states = Vec::new();
    'outer: for mask in 0..(1usize << g.n) {
        for b in 0..g.n {
            s[b] = if mask >> b & 1 == 1 { 1 } else { -1 };
        }
        for (k, &i) in task.inputs.iter().enumerate() {
            if s[i] != x[k] {
                continue 'outer;
            }
        }
        let l = hamming_loss(task, &s, target);
        logs.push(-(g.energy(&s) + nudge * l));
        states.push((s.clone(), l));
    }
    let mx = logs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let w: Vec<f64> = logs.iter().map(|l| (l - mx).exp()).collect();
    let z: f64 = w.iter().sum();
    let mut mean_l = 0.0;
    let mut mm = vec![0.0; pairs.len()];
    let mut m1 = vec![0.0; g.n];
    for ((st, l), wi) in states.iter().zip(&w) {
        let p = wi / z;
        mean_l += p * l;
        for (k, &(i, j)) in pairs.iter().enumerate() {
            mm[k] += p * (st[i] as i32 * st[j] as i32) as f64;
        }
        for i in 0..g.n {
            m1[i] += p * st[i] as f64;
        }
    }
    (mean_l, mm, m1)
}

/// The exact gradient `d⟨ℓ⟩₀/dθ` by enumeration: the covariances of the loss with the sufficient
/// statistics under the free distribution.
pub fn exact_gradient(g: &Graph, task: &Task, x: &[i8], target: &[i8]) -> Gradient {
    assert!(g.n <= 20);
    let pairs = pairs_of(g);
    let mut s = vec![-1i8; g.n];
    let (mut logs, mut states) = (Vec::new(), Vec::new());
    'outer: for mask in 0..(1usize << g.n) {
        for b in 0..g.n {
            s[b] = if mask >> b & 1 == 1 { 1 } else { -1 };
        }
        for (k, &i) in task.inputs.iter().enumerate() {
            if s[i] != x[k] {
                continue 'outer;
            }
        }
        logs.push(-g.energy(&s));
        states.push((s.clone(), hamming_loss(task, &s, target)));
    }
    let mx = logs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let w: Vec<f64> = logs.iter().map(|l| (l - mx).exp()).collect();
    let z: f64 = w.iter().sum();
    let mean_l: f64 = states.iter().zip(&w).map(|((_, l), wi)| wi / z * l).sum();
    let mut dc = vec![0.0; pairs.len()];
    let mut db = vec![0.0; g.n];
    for ((st, l), wi) in states.iter().zip(&w) {
        let p = wi / z;
        for (k, &(i, j)) in pairs.iter().enumerate() {
            dc[k] += p * (l - mean_l) * (st[i] as i32 * st[j] as i32) as f64;
        }
        for i in 0..g.n {
            db[i] += p * (l - mean_l) * st[i] as f64;
        }
    }
    Gradient { pairs, d_couplings: dc, d_biases: db }
}

/// Equilibrium propagation by enumeration: the one-sided quotient `(⟨·⟩₀ − ⟨·⟩_β)/β`, or the
/// centered `(⟨·⟩_{−β} − ⟨·⟩_{+β})/2β` when `centered`.
pub fn eqprop_gradient_exact(g: &Graph, task: &Task, x: &[i8], target: &[i8], beta: f64, centered: bool) -> Gradient {
    let pairs = pairs_of(g);
    let (_, mp, m1p) = enumerate_moments(g, task, x, target, beta);
    let (_, mm, m1m) = if centered { enumerate_moments(g, task, x, target, -beta) } else { enumerate_moments(g, task, x, target, 0.0) };
    let denom = if centered { 2.0 * beta } else { beta };
    Gradient {
        pairs,
        d_couplings: mm.iter().zip(&mp).map(|(a, b)| (a - b) / denom).collect(),
        d_biases: m1m.iter().zip(&m1p).map(|(a, b)| (a - b) / denom).collect(),
    }
}

/// The nudged graph: `E + β ℓ` is `E` with `β t_o / 2` added to each output bias (the constant
/// drops out), so nudging is a field the sampler already understands.
pub fn nudged(g: &Graph, task: &Task, target: &[i8], beta: f64) -> Graph {
    let mut gb = GraphBuilder::new(g.n);
    for (i, j) in pairs_of(g) {
        let e = (g.offset[i]..g.offset[i + 1]).find(|&e| g.nbr[e] as usize == j).unwrap();
        gb.couple(i, j, g.w[e]);
    }
    for i in 0..g.n {
        gb.bias(i, g.h[i]);
    }
    for (&o, &t) in task.outputs.iter().zip(target) {
        gb.bias(o, beta * t as f64 / 2.0);
    }
    gb.build()
}

/// Sampled moments `(⟨s_i s_j⟩ per pair, ⟨s_i⟩ per site)` at unit temperature with inputs clamped.
pub fn sampled_moments(g: &Graph, task: &Task, x: &[i8], burn_in: usize, draws: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let pairs = pairs_of(g);
    let mut sm = Sampler::new(g, 1.0, seed);
    for (k, &i) in task.inputs.iter().enumerate() {
        sm.clamp(i, x[k]);
    }
    sm.sweeps(burn_in, None);
    let mut mm = vec![0.0; pairs.len()];
    let mut m1 = vec![0.0; g.n];
    for _ in 0..draws {
        sm.sweep(None);
        for (k, &(i, j)) in pairs.iter().enumerate() {
            mm[k] += (sm.s[i] as i32 * sm.s[j] as i32) as f64;
        }
        for i in 0..g.n {
            m1[i] += sm.s[i] as f64;
        }
    }
    let d = draws as f64;
    (mm.iter().map(|v| v / d).collect(), m1.iter().map(|v| v / d).collect())
}

/// Equilibrium propagation by sampling: two chains (free and nudged, or `∓β` when centered) and
/// their difference quotient.
pub fn eqprop_gradient(g: &Graph, task: &Task, x: &[i8], target: &[i8], beta: f64, centered: bool, burn_in: usize, draws: usize, seed: u64) -> Gradient {
    let pairs = pairs_of(g);
    let plus = nudged(g, task, target, beta);
    let (mp, m1p) = sampled_moments(&plus, task, x, burn_in, draws, seed);
    let (mm, m1m) = if centered {
        let minus = nudged(g, task, target, -beta);
        sampled_moments(&minus, task, x, burn_in, draws, seed + 1)
    } else {
        sampled_moments(g, task, x, burn_in, draws, seed + 1)
    };
    let denom = if centered { 2.0 * beta } else { beta };
    Gradient {
        pairs,
        d_couplings: mm.iter().zip(&mp).map(|(a, b)| (a - b) / denom).collect(),
        d_biases: m1m.iter().zip(&m1p).map(|(a, b)| (a - b) / denom).collect(),
    }
}

/// Apply a gradient step `θ ← θ − η ∇`, returning the new graph.
pub fn step(g: &Graph, grad: &Gradient, eta: f64) -> Graph {
    let mut gb = GraphBuilder::new(g.n);
    for (k, &(i, j)) in grad.pairs.iter().enumerate() {
        let e = (g.offset[i]..g.offset[i + 1]).find(|&e| g.nbr[e] as usize == j).unwrap();
        let w = g.w[e] - eta * grad.d_couplings[k];
        if w != 0.0 {
            gb.couple(i, j, w);
        }
    }
    for i in 0..g.n {
        gb.bias(i, g.h[i] - eta * grad.d_biases[i]);
    }
    gb.build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Pcg;

    fn small_machine(seed: u64) -> (Graph, Task) {
        // 2 inputs, 3 hidden, 1 output, all-to-all between layers plus hidden-hidden.
        let mut rng = Pcg::new(seed, 0);
        let mut gb = GraphBuilder::new(6);
        let r = |rng: &mut Pcg| 0.8 * (rng.f64() - 0.5);
        for i in 0..2 {
            for h in 2..5 {
                gb.couple(i, h, r(&mut rng));
            }
        }
        for h in 2..5 {
            gb.couple(h, 5, r(&mut rng));
        }
        gb.couple(2, 3, r(&mut rng));
        gb.couple(3, 4, r(&mut rng));
        for i in 0..6 {
            gb.bias(i, 0.3 * (rng.f64() - 0.5));
        }
        (gb.build(), Task { inputs: vec![0, 1], outputs: vec![5] })
    }

    fn max_err(a: &Gradient, b: &Gradient) -> f64 {
        a.d_couplings.iter().zip(&b.d_couplings).chain(a.d_biases.iter().zip(&b.d_biases)).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
    }

    /// The theorem, at its two rates: one-sided error halves with β, centered error quarters.
    #[test]
    fn the_difference_quotient_converges_at_the_stated_rates() {
        let (g, task) = small_machine(1);
        let (x, t) = ([1i8, -1], [1i8]);
        let truth = exact_gradient(&g, &task, &x, &t);
        let e1 = max_err(&eqprop_gradient_exact(&g, &task, &x, &t, 0.2, false), &truth);
        let e2 = max_err(&eqprop_gradient_exact(&g, &task, &x, &t, 0.1, false), &truth);
        let c1 = max_err(&eqprop_gradient_exact(&g, &task, &x, &t, 0.2, true), &truth);
        let c2 = max_err(&eqprop_gradient_exact(&g, &task, &x, &t, 0.1, true), &truth);
        assert!(e1 > 1e-4 && c1 > 1e-6, "the errors must be visible to have rates: {e1}, {c1}");
        let (r1, r2) = (e1 / e2, c1 / c2);
        assert!((1.7..2.3).contains(&r1), "one-sided error ratio {r1}, expected about 2");
        assert!((3.4..4.6).contains(&r2), "centered error ratio {r2}, expected about 4");
        assert!(c2 < e2, "centered is more accurate at the same beta");
        // and at small beta the quotient is the gradient to high precision
        let tiny = max_err(&eqprop_gradient_exact(&g, &task, &x, &t, 1e-3, true), &truth);
        assert!(tiny < 1e-6, "centered at beta 1e-3: error {tiny}");
    }

    /// The sampled rule lands within its statistical error of the exact gradient.
    #[test]
    fn the_sampled_rule_agrees_with_the_exact_gradient() {
        let (g, task) = small_machine(2);
        let (x, t) = ([-1i8, 1], [-1i8]);
        let truth = exact_gradient(&g, &task, &x, &t);
        let beta = 0.2;
        let bias = max_err(&eqprop_gradient_exact(&g, &task, &x, &t, beta, true), &truth);
        let sampled = eqprop_gradient(&g, &task, &x, &t, beta, true, 500, 40_000, 3);
        let err = max_err(&sampled, &truth);
        // each moment's standard error is ~1/sqrt(draws) ≈ 0.005, divided by 2β = 0.4 → ~0.0125;
        // allow four of those plus the finite-β bias
        assert!(err < 0.05 + bias, "sampled gradient error {err} (finite-beta bias {bias})");
    }

    /// Learning: a dozen exact-EqProp steps reduce the expected loss on a fixed pattern.
    #[test]
    fn a_few_steps_reduce_the_loss() {
        let (mut g, task) = small_machine(3);
        let (x, t) = ([1i8, 1], [-1i8]);
        let loss = |g: &Graph| enumerate_moments(g, &task, &x, &t, 0.0).0;
        let before = loss(&g);
        for _ in 0..12 {
            let grad = eqprop_gradient_exact(&g, &task, &x, &t, 0.05, true);
            g = step(&g, &grad, 0.5);
        }
        let after = loss(&g);
        assert!(after < before - 0.05, "loss {before} -> {after}");
    }
}
