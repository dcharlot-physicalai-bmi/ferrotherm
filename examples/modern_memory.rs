//! Modern memory and learning from two equilibria: dense associative memory, attention as its
//! update, and equilibrium propagation -- each held against something exact.
//!
//! usage: cargo run --release --example modern_memory

use ferrotherm::dense_memory::{corrupt, overlap, DenseMemory, Energy};
use ferrotherm::eqprop::{eqprop_gradient_exact, exact_gradient, step, Gradient, Task};
use ferrotherm::graph::{Graph, GraphBuilder};
use ferrotherm::hopfield::random_patterns;
use ferrotherm::rng::Pcg;

fn main() {
    // ---- 1. capacity by degree, as the theorems define it: stored patterns that are fixed points
    let n = 100;
    println!("--- dense memory, N = {n}: fraction of stored patterns that are fixed points of T = 0 dynamics");
    println!("{:<6} {:>10} {:>10} {:>10} {:>12}", "P", "alpha", "degree 2", "degree 3", "exponential");
    for p in [8usize, 14, 20, 40, 120, 400] {
        let pats = random_patterns(n, p, 5);
        let f = |e: Energy| DenseMemory::new(pats.clone(), e).stable_fraction();
        println!(
            "{:<6} {:>10.2} {:>10.2} {:>10.2} {:>12.2}",
            p, p as f64 / n as f64, f(Energy::Polynomial(2)), f(Energy::Polynomial(3)), f(Energy::Exponential { b: 1.0 })
        );
    }
    println!("  degree 2 is the classical memory (alpha_c ~ 0.138); degree 3 stores of order N^2 / ln N;\n  \
              the exponential memory, of order 2^(N/2). At degree 2 the energy equals the Hebbian one\n  \
              minus P/2 for every state -- an identity the tests hold to 1e-9.\n");

    // ---- 2. attention as the exponential memory's one-step update
    let n = 64;
    let pats = random_patterns(n, 200, 11);
    let m = DenseMemory::new(pats.clone(), Energy::Exponential { b: 1.0 });
    println!("--- attention update, {} patterns in {n} spins (alpha = {:.2}): one-step retrieval of 100 corrupted queries", pats.len(), 200.0 / n as f64);
    println!("{:<12} {:>14} {:>22}", "corruption", "original back", "nearest pattern back");
    for frac in [0.10, 0.15, 0.25, 0.35] {
        let (mut orig, mut near) = (0, 0);
        for mu in 0..100 {
            let qi = corrupt(&pats[mu], frac, 1000 + mu as u64);
            let q: Vec<f64> = qi.iter().map(|&v| v as f64).collect();
            let got: Vec<i8> = m.attention_update(&q, 2.0).iter().map(|&x| if x >= 0.0 { 1 } else { -1 }).collect();
            if overlap(&pats[mu], &got) > 0.99 {
                orig += 1;
            }
            let best = (0..pats.len()).map(|k| overlap(&pats[k], &qi)).fold(f64::NEG_INFINITY, f64::max);
            if (0..pats.len()).any(|k| (overlap(&pats[k], &qi) - best).abs() < 1e-12 && overlap(&pats[k], &got) > 0.99) {
                near += 1;
            }
        }
        println!("{:<12.2} {:>14} {:>22}", frac, format!("{orig}/100"), format!("{near}/100"));
    }
    println!("  Where the original is not returned, the query had drifted nearer another stored pattern\n  \
              (or tied, in which case the update is their mean); the softmax returns the nearest.\n");

    // ---- 3. equilibrium propagation: the theorem at its two rates, then learning
    let (g, task) = machine();
    let (x, t) = ([1i8, -1], [1i8]);
    let truth = exact_gradient(&g, &task, &x, &t);
    println!("--- equilibrium propagation on a 6-spin Boltzmann machine: max |EqProp - exact gradient|");
    println!("{:<8} {:>14} {:>14}", "beta", "one-sided", "centered");
    for beta in [0.4, 0.2, 0.1, 0.05, 0.025] {
        let e1 = max_err(&eqprop_gradient_exact(&g, &task, &x, &t, beta, false), &truth);
        let e2 = max_err(&eqprop_gradient_exact(&g, &task, &x, &t, beta, true), &truth);
        println!("{:<8.3} {:>14.3e} {:>14.3e}", beta, e1, e2);
    }
    println!("  one-sided error falls as beta, centered as beta^2 -- Scellier-Bengio 2017, Laborieux et al. 2021.\n");
    let mut g2 = g.clone_graph();
    print!("  learning a fixed (input, target) pair with centered EqProp, beta = 0.05, eta = 0.5: expected loss");
    for k in 0..=12 {
        if k % 3 == 0 {
            print!(" {:.3}", expected_loss(&g2, &task, &x, &t));
        }
        let grad = eqprop_gradient_exact(&g2, &task, &x, &t, 0.05, true);
        g2 = step(&g2, &grad, 0.5);
    }
    println!();
}

trait CloneGraph {
    fn clone_graph(&self) -> Graph;
}
impl CloneGraph for Graph {
    fn clone_graph(&self) -> Graph {
        let mut gb = GraphBuilder::new(self.n);
        for i in 0..self.n {
            for e in self.offset[i]..self.offset[i + 1] {
                let j = self.nbr[e] as usize;
                if j > i {
                    gb.couple(i, j, self.w[e]);
                }
            }
            gb.bias(i, self.h[i]);
        }
        gb.build()
    }
}

fn machine() -> (Graph, Task) {
    let mut rng = Pcg::new(1, 0);
    let mut gb = GraphBuilder::new(6);
    let mut r = || 0.8 * (rng.f64() - 0.5);
    for i in 0..2 {
        for h in 2..5 {
            gb.couple(i, h, r());
        }
    }
    for h in 2..5 {
        gb.couple(h, 5, r());
    }
    gb.couple(2, 3, r());
    gb.couple(3, 4, r());
    for i in 0..6 {
        gb.bias(i, 0.3 * r());
    }
    (gb.build(), Task { inputs: vec![0, 1], outputs: vec![5] })
}

fn max_err(a: &Gradient, b: &Gradient) -> f64 {
    a.d_couplings.iter().zip(&b.d_couplings).chain(a.d_biases.iter().zip(&b.d_biases)).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
}

fn expected_loss(g: &Graph, task: &Task, x: &[i8], t: &[i8]) -> f64 {
    // enumerate with inputs clamped
    let mut s = vec![-1i8; g.n];
    let (mut num, mut den) = (0.0, 0.0);
    let mut mx = f64::NEG_INFINITY;
    let mut rows = Vec::new();
    'o: for mask in 0..(1usize << g.n) {
        for b in 0..g.n {
            s[b] = if mask >> b & 1 == 1 { 1 } else { -1 };
        }
        for (k, &i) in task.inputs.iter().enumerate() {
            if s[i] != x[k] {
                continue 'o;
            }
        }
        let l = ferrotherm::eqprop::hamming_loss(task, &s, t);
        let e = -g.energy(&s);
        mx = mx.max(e);
        rows.push((e, l));
    }
    for (e, l) in rows {
        let w = (e - mx).exp();
        num += w * l;
        den += w;
    }
    num / den
}
