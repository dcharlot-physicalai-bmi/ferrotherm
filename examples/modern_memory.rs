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

    // ---- 3. the program path: the same memory from the closed form to a machine ----------------
    {
        use ferrotherm::embed;
        use ferrotherm::exact::Elimination;
        use ferrotherm::gibbs::Sampler;
        use ferrotherm::hubo;
        use ferrotherm::reduce::to_pairwise;
        use ferrotherm::schedule::Schedule;
        println!("--- the program path: a degree-4 memory of 10 spins as a higher-order program");
        let n = 10;
        let pats = random_patterns(n, 2, 31);
        let m = DenseMemory::new(pats.clone(), Energy::Polynomial(4));
        let (h, _) = m.to_hubo().unwrap();
        let out = hubo::anneal(&h, &hubo::Params { beta_min: 0.2, beta_max: 6.0, stages: 30, sweeps_per_stage: 10 }, 5);
        println!("  HUBO: {} terms, max arity {}; native annealing retrieves overlap {:.2}", h.terms(), h.max_arity(), overlap(&pats[0], &out.state).abs());
        let (prog, _) = m.to_program(&Schedule::constant(3.0, 100)).unwrap();
        let red = to_pairwise(&prog).unwrap();
        let g = red.program.to_graph().unwrap();
        println!("  reduced to pairwise: {} spins ({} ancillas), {} edges, max degree {}, penalty {:.1} against a memory signal of ~2.5",
            g.n, red.ancillas, g.n_edges, g.max_degree(), red.penalty);
        let mut worst = 0.0f64;
        let mut rng = Pcg::new(9, 0);
        for trial in 0..6 {
            let s: Vec<i8> = if trial == 0 { pats[0].clone() } else { (0..n).map(|_| if rng.f64() < 0.5 { -1 } else { 1 }).collect() };
            let mut sm = Sampler::new(&g, 8.0, trial);
            for i in 0..n { sm.clamp(i, s[i]); }
            sm.sweeps(400, None);
            worst = worst.max((g.energy(&sm.s) + red.offset - h.energy(&s)).abs());
        }
        println!("  the reduction is EXACT: min over ancillas + offset vs HUBO energy, worst |diff| {worst:.1e} over 6 states");
        let mut hits = 0;
        for seed in 0..5u64 {
            let mut sm = Sampler::new(&g, 0.02, seed);
            for k in 0..40 { sm.beta = 0.02 + 8.0 * k as f64 / 39.0; sm.sweeps(25, None); }
            if overlap(&pats[0], red.project(&sm.s)).abs() > 0.99 { hits += 1; }
        }
        let hw6 = ferrotherm::ising::chimera(6, 6, 4, 1.0);
        let placed = embed::embed_bounded(&g, &hw6, 3, 20, embed::DEFAULT_SEARCH_BUDGET).is_some();
        println!("  ...and dynamically FROZEN: annealing the reduced model retrieves in {hits} of 5 runs; a 288-site Chimera {} it",
            if placed { "places" } else { "does not place" });
        // degree 2 is pairwise: onto the machine by the structured clique
        let n2 = 12;
        let (pats2, m2) = (0..64u64).map(|seed| { let p = random_patterns(n2, 2, 500 + seed); let m = DenseMemory::new(p.clone(), Energy::Polynomial(2)); (p, m) })
            .find(|(p, m)| p.iter().all(|x| m.is_fixed_point(x))).unwrap();
        let (prog2, _) = m2.to_program(&Schedule::constant(3.0, 100)).unwrap();
        let g2 = to_pairwise(&prog2).unwrap().program.to_graph().unwrap();
        let hw = ferrotherm::ising::chimera(3, 3, 4, 1.0);
        let e = embed::chimera_clique(3, 4).unwrap();
        e.verify(&g2, &hw).unwrap();
        let max_w = g2.w.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
        let ground = Elimination::default().ground_state(&g2).unwrap().ground_energy.unwrap();
        for cs in [6.0 * max_w, 4.0] {
            let hwm = embed::apply_with(&g2, &hw, &e, cs);
            let mut solved = 0;
            for seed in 0..5u64 {
                let mut sh = Sampler::new(&hwm.graph, 0.05, 11 + seed);
                for k in 0..40 { sh.beta = 0.05 + 8.0 * k as f64 / 39.0; sh.sweeps(25, None); }
                let (logical, broken) = embed::unembed(&e, &sh.s);
                let ov = pats2.iter().map(|p| overlap(p, &logical).abs()).fold(0.0, f64::max);
                if broken.is_empty() && (g2.energy(&logical) - ground).abs() < 1e-9 && ov > 0.99 { solved += 1; }
            }
            println!("  degree 2 (pairwise, no ancillas) on chimera(3,3,4) by the K_12 clique, chain strength {cs:.2} ({:.0}x max |J|): exact ground state = a stored pattern in {solved} of 5 anneals",
                cs / max_w);
        }
        println!("  chain strength is relative to the couplings: the 4.0 that is right for unit couplings\n  \
                  elsewhere in this crate is 24x these, and freezes the chains before the problem orders.\n");
    }

    // ---- 4. equilibrium propagation: the theorem at its two rates, then learning
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
