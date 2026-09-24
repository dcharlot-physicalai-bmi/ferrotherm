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
    /// Nodes clamped to the example, held fixed in both phases.
    pub inputs: Vec<usize>,
    /// Nodes the loss is read from, and nudged in the second phase.
    pub outputs: Vec<usize>,
}

/// Hamming loss `ℓ(s) = Σ_o (1 − s_o t_o) / 2`, in units of wrong output spins.
#[must_use]
pub fn hamming_loss(task: &Task, s: &[i8], target: &[i8]) -> f64 {
    task.outputs.iter().zip(target).map(|(&o, &t)| (1.0 - (s[o] as i32 * t as i32) as f64) / 2.0).sum()
}

/// Gradient of the expected loss with respect to every coupling (per edge slot, `i < j` once) and
/// bias, in the layout `(couplings, biases)` where couplings follow `pairs`.
#[derive(Clone, Debug)]
pub struct Gradient {
    /// The edges the coupling gradients belong to, parallel to `d_couplings`.
    pub pairs: Vec<(usize, usize)>,
    /// Gradient with respect to each coupling.
    pub d_couplings: Vec<f64>,
    /// Gradient with respect to each node's bias.
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
    let mx = logs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
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
///
/// # Panics
///
/// If the graph has more than 20 spins, since the exact gradient enumerates them.
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
    let mx = logs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
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
#[must_use]
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
///
/// # Panics
///
/// If `target` does not cover the task's outputs.
#[must_use]
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
#[must_use]
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
#[must_use]
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

/// Where each phase of a two-phase equilibrium-propagation run starts relaxing from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Relaxation {
    /// Every phase relaxes for the same budget from the uniform law over clamp-consistent states —
    /// which is what [`eqprop_gradient`] samples, each phase from its own fresh chain. Both phases
    /// are equally short of equilibrium, so their residuals largely cancel in the difference.
    Cold,
    /// The nudged phase relaxes from **wherever the free phase actually got to** — the protocol
    /// arXiv:2604.23806 §5 states: *"The nudged-phase stationary state `x⋆^β(θ)` is reached by
    /// relaxation from `x⋆^0(θ)`"*. Under a finite budget `x⋆^0` is not the stationary state; it is
    /// the free law after `sweeps` sweeps, and the nudged phase then runs `sweeps` more from there.
    ///
    /// **That is the whole asymmetry.** At `β = 0` the nudged kernel IS the free kernel, so the
    /// nudged phase ends `2·sweeps` from the start while the free reference sits at `sweeps` — the
    /// two differ even with no nudge at all. That leftover is the paper's persistent free-phase
    /// residual `R`, and the one-sided quotient divides it by `β`.
    ///
    /// Writing this variant with the free phase at exact equilibrium instead — which is what
    /// `x⋆^0(θ)` means in the infinite-budget theory — removes `R` entirely and the divergence with
    /// it. That version was written first here and measured flat, which is how the distinction was
    /// found.
    Warm,
}

/// `Some(v)` for each site the task clamps, `None` otherwise.
fn clamp_mask(g: &Graph, task: &Task, x: &[i8]) -> Vec<Option<i8>> {
    let mut c = vec![None; g.n];
    for (k, &i) in task.inputs.iter().enumerate() {
        c[i] = Some(x[k]);
    }
    c
}

fn obeys(n: usize, state: usize, clamp: &[Option<i8>]) -> bool {
    (0..n).all(|b| match clamp[b] {
        Some(v) => (if state >> b & 1 == 1 { 1i8 } else { -1 }) == v,
        None => true,
    })
}

fn spins_of(n: usize, state: usize, s: &mut [i8]) {
    for b in 0..n {
        s[b] = if state >> b & 1 == 1 { 1 } else { -1 };
    }
}

/// The uniform law over the states the clamp permits, zero elsewhere.
fn uniform_law(g: &Graph, clamp: &[Option<i8>]) -> Vec<f64> {
    let m = 1usize << g.n;
    let ok: Vec<bool> = (0..m).map(|xi| obeys(g.n, xi, clamp)).collect();
    let count = ok.iter().filter(|b| **b).count() as f64;
    ok.iter().map(|b| if *b { 1.0 / count } else { 0.0 }).collect()
}

/// The Boltzmann law at unit temperature over the states a clamp permits — the equilibrium the
/// chromatic sweep is reversible with respect to, and therefore the fixed point of any amount of
/// relaxation.
///
/// Public because it is the oracle a sampler on this graph is checked against: advancing this law
/// through a sweep must return it exactly, and
/// `a_warm_start_turns_the_relaxation_residual_into_a_one_over_beta_divergence` opens by asserting
/// that at three budgets. `clamp[i]` pins site `i` when it is `Some`.
///
/// # Panics
///
/// If `g` has more than 20 spins, since the law is enumerated over `2^n` states.
#[must_use]
pub fn boltzmann_law(g: &Graph, clamp: &[Option<i8>]) -> Vec<f64> {
    assert!(g.n <= 20, "the law is enumerated over 2^n states");
    let (n, m) = (g.n, 1usize << g.n);
    let mut s = vec![-1i8; n];
    let mut logs = vec![f64::NEG_INFINITY; m];
    for (xi, l) in logs.iter_mut().enumerate() {
        if !obeys(n, xi, clamp) {
            continue;
        }
        spins_of(n, xi, &mut s);
        *l = -g.energy(&s);
    }
    let mx = logs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let w: Vec<f64> = logs.iter().map(|l| if l.is_finite() { (l - mx).exp() } else { 0.0 }).collect();
    let z: f64 = w.iter().sum();
    w.iter().map(|v| v / z).collect()
}

/// One chromatic sweep, applied to the LAW rather than to a state.
///
/// This mirrors [`crate::gibbs::Sampler::sweep`] exactly — the same `g.classes` order, the same
/// `kernel::p_up` conditional at unit temperature, the same skip on clamped sites — and there is no
/// Monte Carlo anywhere in it. Within a colour class the sites are pairwise non-adjacent, so every
/// field is read from the pre-class state and no member sees another member's new value; the class
/// update therefore factorises into independent Bernoulli draws and the whole `2^k` block of
/// outcomes can be written down.
///
/// Sampling cannot answer the question this exists for. The estimator divides a difference of
/// moments by `β`, so at small `β` the sampling noise diverges as `1/β` exactly as a relaxation
/// residual would, and the two are not separable from draws. Advancing the law separates them.
fn relax_once(g: &Graph, clamp: &[Option<i8>], mu: &[f64]) -> Vec<f64> {
    let (n, m) = (g.n, 1usize << g.n);
    let mut cur = mu.to_vec();
    let mut s = vec![-1i8; n];
    for class in &g.classes {
        let sites: Vec<usize> = class.iter().map(|&i| i as usize).filter(|&i| clamp[i].is_none()).collect();
        if sites.is_empty() {
            continue;
        }
        let mut next = vec![0.0; m];
        for xi in 0..m {
            let mass = cur[xi];
            if mass == 0.0 {
                continue;
            }
            spins_of(n, xi, &mut s);
            let qs: Vec<f64> = sites.iter().map(|&i| crate::kernel::p_up(g.field(i, &s), 1.0)).collect();
            for combo in 0..(1usize << sites.len()) {
                let (mut y, mut w) = (xi, mass);
                for (k, &i) in sites.iter().enumerate() {
                    if combo >> k & 1 == 1 {
                        y |= 1 << i;
                        w *= qs[k];
                    } else {
                        y &= !(1 << i);
                        w *= 1.0 - qs[k];
                    }
                }
                next[y] += w;
            }
        }
        cur = next;
    }
    cur
}

fn relax(g: &Graph, clamp: &[Option<i8>], mu: &[f64], sweeps: usize) -> Vec<f64> {
    let mut cur = mu.to_vec();
    for _ in 0..sweeps {
        cur = relax_once(g, clamp, &cur);
    }
    cur
}

fn law_moments(g: &Graph, pairs: &[(usize, usize)], mu: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = g.n;
    let (mut mm, mut m1) = (vec![0.0; pairs.len()], vec![0.0; n]);
    let mut s = vec![-1i8; n];
    for (xi, &p) in mu.iter().enumerate() {
        if p == 0.0 {
            continue;
        }
        spins_of(n, xi, &mut s);
        for (k, &(i, j)) in pairs.iter().enumerate() {
            mm[k] += p * f64::from(i32::from(s[i]) * i32::from(s[j]));
        }
        for (i, v) in m1.iter_mut().enumerate() {
            *v += p * f64::from(s[i]);
        }
    }
    (mm, m1)
}

/// **Equilibrium propagation at a FINITE relaxation budget, exactly** — the gradient estimator a
/// machine actually computes when it is given `sweeps` sweeps per phase instead of equilibrium.
///
/// [`eqprop_gradient_exact`] is the `sweeps → ∞` limit and [`eqprop_gradient`] is the sampled
/// version; this is the one in between, and it carries no Monte Carlo, so a relaxation residual can
/// be told apart from sampling noise. `start` selects whose protocol is being run — see
/// [`Relaxation`].
///
/// # Panics
///
/// If `g` has more than 20 spins, or `beta` is not positive and finite.
#[must_use]
pub fn eqprop_gradient_relaxed(
    g: &Graph,
    task: &Task,
    x: &[i8],
    target: &[i8],
    beta: f64,
    centered: bool,
    sweeps: usize,
    start: Relaxation,
) -> Gradient {
    assert!(g.n <= 20, "the law is enumerated over 2^n states");
    assert!(beta > 0.0 && beta.is_finite(), "beta must be positive and finite, got {beta}");
    let pairs = pairs_of(g);
    let clamp = clamp_mask(g, task, x);
    let plus_g = nudged(g, task, target, beta);
    // `nudged` rebuilds the graph from the same couplings and only moves biases, so the colour
    // classes are the same and the sweep visits sites in the same order in both phases. The scan
    // order matters here -- permuting it changes the one-sided numbers materially -- so this is
    // asserted rather than assumed.
    debug_assert_eq!(plus_g.classes, g.classes, "nudging must not reorder the sweep");
    let (free_law, nudged_start) = match start {
        Relaxation::Cold => {
            let u = uniform_law(g, &clamp);
            (relax(g, &clamp, &u, sweeps), u)
        }
        Relaxation::Warm => {
            // The free phase gets the same finite budget the hardware gives it, and the nudged
            // phase continues from there rather than from equilibrium.
            let free = relax(g, &clamp, &uniform_law(g, &clamp), sweeps);
            (free.clone(), free)
        }
    };
    let (mp, m1p) = law_moments(g, &pairs, &relax(&plus_g, &clamp, &nudged_start, sweeps));
    let (mm, m1m) = if centered {
        let minus_g = nudged(g, task, target, -beta);
        law_moments(g, &pairs, &relax(&minus_g, &clamp, &nudged_start, sweeps))
    } else {
        law_moments(g, &pairs, &free_law)
    };
    let denom = if centered { 2.0 * beta } else { beta };
    Gradient {
        pairs,
        d_couplings: mm.iter().zip(&mp).map(|(a, b)| (a - b) / denom).collect(),
        d_biases: m1m.iter().zip(&m1p).map(|(a, b)| (a - b) / denom).collect(),
    }
}

/// Apply a gradient step `θ ← θ − η ∇`, returning the new graph.
///
/// # Panics
///
/// If the gradient was computed for a different graph, so its pairs do not index this one.
#[must_use]
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

    /// **A warm start turns the relaxation residual into a 1/β divergence** — and symmetric
    /// nudging is what removes it. arXiv:2604.23806 §5 specifies the two-phase protocol as
    /// *"The nudged-phase stationary state x⋆^β(θ) is reached by relaxation from x⋆^0(θ)"*. Under a
    /// finite budget `x⋆^0` is not stationary: it is the free law after `b` sweeps, and the nudged
    /// phase then runs `b` more from there. At `β = 0` the nudged kernel IS the free kernel, so the
    /// nudged phase lands `2b` sweeps from the start while the free reference sits at `b` — they
    /// differ with no nudge at all, and the one-sided quotient divides that leftover by `β`.
    ///
    /// Maximum error against `exact_gradient`, `b = 1`:
    ///
    /// | β | 0.4 | 0.2 | 0.1 | 0.05 | 0.02 | 0.01 | 0.001 |
    /// |---|---|---|---|---|---|---|---|
    /// | warm, one-sided | 8.71e-2 | 7.46e-2 | 1.12e-1 | 2.23e-1 | 5.54e-1 | 1.11e0 | **1.10e1** |
    /// | warm, centered | 8.90e-2 | 8.66e-2 | 8.60e-2 | 8.58e-2 | 8.58e-2 | 8.58e-2 | 8.58e-2 |
    /// | cold, one-sided | 1.04e-1 | 9.91e-2 | 9.73e-2 | 9.66e-2 | 9.62e-2 | 9.60e-2 | 9.59e-2 |
    ///
    /// **Three things follow, and the third is the practical one.**
    ///
    /// The paper's headline is a rate upgrade, `O(β)` to `O(β²)`. At a finite budget on this
    /// substrate the symmetric estimator is doing something more important than converging faster:
    /// **it is the difference between converging and diverging.** At `β = 0` both `±β` phases
    /// relax to the same place, so the residual cancels in the symmetric difference and only the
    /// truncation floor is left.
    ///
    /// **It is protocol-dependent, and this crate's own protocol does not show it.**
    /// [`Relaxation::Cold`] — both phases from their own fresh chain, which is what
    /// [`eqprop_gradient`] samples — is flat at every `β` here. The divergence is a property of the
    /// warm start, not of one-sided equilibrium propagation.
    ///
    /// **At a finite budget with a warm start, `β → 0` is not the limit to chase.** The error has
    /// an interior minimum near `β = 0.2`; below it the residual term `R/β` wins. The infinite-
    /// budget theory says smaller `β` is always better, and here it is false by two orders of
    /// magnitude.
    ///
    /// # What is NOT claimed
    ///
    /// The paper's substrate is a `D = 64`, rank-16 bilinearly-coupled Langevin system with
    /// `K = 300` Euler–Maruyama steps; this is a six-spin Ising machine under chromatic Gibbs. Its
    /// reported constants — the `0.41` and `2.000` log-log slopes, the `−0.73`/`+0.77` cosine
    /// similarities — are not thresholds here and appear in no assertion. Only the **ordering** and
    /// the **sign of the degradation** are portable, and only those are asserted. (Its `E1` sign
    /// flip in particular does not transfer: with no sampling noise the one-sided cosine here
    /// degrades but stays positive, so a "cosine goes negative" gate would be a guaranteed red
    /// rather than a measurement.)
    #[test]
    fn a_warm_start_turns_the_relaxation_residual_into_a_one_over_beta_divergence() {
        let (g, task) = small_machine(1);
        let (x, t) = ([1i8, -1], [1i8]);
        let truth = exact_gradient(&g, &task, &x, &t);
        let err = |beta: f64, centered: bool, sweeps: usize, start: Relaxation| {
            max_err(&eqprop_gradient_relaxed(&g, &task, &x, &t, beta, centered, sweeps, start), &truth)
        };

        // CONTROL 0. The chromatic sweep is reversible with respect to the Boltzmann law, so that
        // law is the relaxation operator's FIXED POINT: advancing it must return it exactly, at any
        // budget. This is the invariant-measure check, and it is what says `relax` transports
        // probability correctly rather than merely converging to something.
        {
            let clamp = clamp_mask(&g, &task, &x);
            let eq = boltzmann_law(&g, &clamp);
            for sweeps in [1usize, 3, 17] {
                let moved = relax(&g, &clamp, &eq, sweeps);
                let drift = eq.iter().zip(&moved).map(|(a, b)| (a - b).abs()).fold(0.0f64, f64::max);
                assert!(drift < 1e-15, "the Boltzmann law is the sweep's fixed point: drift {drift:e} after {sweeps}");
            }
            // and it is a fixed point of THIS operator, not of any operator: the uniform law is not.
            let uni = uniform_law(&g, &clamp);
            let moved = relax(&g, &clamp, &uni, 1);
            let drift = uni.iter().zip(&moved).map(|(a, b)| (a - b).abs()).fold(0.0f64, f64::max);
            assert!(drift > 1e-6, "the uniform law must NOT be a fixed point, else the operator does nothing: {drift:e}");
        }

        // CONTROL. The finite-budget operator must become the crate's existing exact estimator when
        // the budget stops binding. These are independent implementations -- one advances the law
        // through `Sampler::sweep`'s own conditionals, the other enumerates a Boltzmann sum -- so
        // this anchors the new operator rather than restating it.
        for centered in [false, true] {
            for &beta in &[0.2f64, 0.05] {
                let lim = eqprop_gradient_relaxed(&g, &task, &x, &t, beta, centered, 400, Relaxation::Cold);
                let ex = eqprop_gradient_exact(&g, &task, &x, &t, beta, centered);
                let d = max_err(&lim, &ex);
                assert!(d < 1e-12, "the sweeps-to-infinity limit must be the exact estimator: {d:e}");
            }
        }

        // THE DIVERGENCE. Same estimator, same budget; only the protocol differs.
        let (hot, cold_beta) = (0.4f64, 0.001f64);
        let warm_one = (err(hot, false, 1, Relaxation::Warm), err(cold_beta, false, 1, Relaxation::Warm));
        let cold_one = (err(hot, false, 1, Relaxation::Cold), err(cold_beta, false, 1, Relaxation::Cold));
        let warm_ctr = (err(hot, true, 1, Relaxation::Warm), err(cold_beta, true, 1, Relaxation::Warm));

        // Measured 1.104e1 / 8.705e-2 = 127x.
        assert!(warm_one.1 / warm_one.0 > 100.0, "warm one-sided must diverge as beta falls: {warm_one:?}");
        // Measured 9.591e-2 / 1.036e-1 = 0.93 -- flat, and slightly improving.
        assert!((0.8..1.2).contains(&(cold_one.1 / cold_one.0)), "cold one-sided must be flat: {cold_one:?}");
        // Measured 8.578e-2 / 8.896e-2 = 0.96 -- the residual cancels in the symmetric difference.
        assert!((0.9..1.1).contains(&(warm_ctr.1 / warm_ctr.0)), "warm centered must be flat: {warm_ctr:?}");
        // and the partition is real: at the coldest beta the warm one-sided error is two orders of
        // magnitude above both of the others, which are within a factor of two of each other.
        assert!(warm_one.1 > 50.0 * cold_one.1, "warm vs cold at beta {cold_beta}: {} vs {}", warm_one.1, cold_one.1);
        assert!(warm_one.1 > 50.0 * warm_ctr.1, "one-sided vs centered at beta {cold_beta}: {} vs {}", warm_one.1, warm_ctr.1);
        assert!(cold_one.1 / warm_ctr.1 < 2.0, "the two flat rows sit together: {} vs {}", cold_one.1, warm_ctr.1);

        // THE INTERIOR MINIMUM: colder is not better once the budget is finite and the start warm.
        let mid = err(0.2, false, 1, Relaxation::Warm);
        assert!(mid < warm_one.0, "beta 0.2 must beat beta 0.4: {mid} vs {}", warm_one.0);
        assert!(mid < err(0.05, false, 1, Relaxation::Warm), "and beat beta 0.05: {mid}");

        // AND IT IS A FINITE-BUDGET EFFECT ONLY. Given enough sweeps both protocols reach the same
        // equilibrium and the two rows become identical, which is why this is about budgets and not
        // about one-sided EqProp being wrong.
        for &beta in &[0.4f64, 0.05, 0.001] {
            let w = err(beta, false, 200, Relaxation::Warm);
            let c = err(beta, false, 200, Relaxation::Cold);
            assert!((w - c).abs() < 1e-12, "at 200 sweeps the protocols must agree: {w} vs {c} at beta {beta}");
            // and there the classical picture is back: colder really is better.
            assert!(w < err(0.4, false, 200, Relaxation::Warm) || (beta - 0.4).abs() < 1e-12, "monotone at 200 sweeps");
        }
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
