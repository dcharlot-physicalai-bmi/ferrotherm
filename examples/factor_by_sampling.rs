#![allow(missing_docs)]
// A multiplier, run backwards, is a factorizer -- and this is it doing the job.
//
// THE CONSTRUCTION (Camsari, Faria, Sutton & Datta, Phys. Rev. X 7, 031014, 2017). A logic gate
// written as an Ising model whose ground states are its truth table has no preferred direction.
// Clamp the inputs and the ground states give the output; clamp the OUTPUT and they give every
// input consistent with it. An array multiplier built that way, with its product bits held at a
// semiprime, has exactly the factor pairs as ground states -- so factoring becomes sampling.
//
// WHY THIS IS THE INTERESTING SHAPE and not just a party trick. The naive way to make factoring an
// Ising problem is to write the multiplication as one dense polynomial and reduce it; that gives an
// all-to-all graph with wide coefficients. The circuit route gives a SPARSE graph whose couplings
// take a handful of small values -- which is what fabricated annealing silicon actually accepts.
// `silicon` here targets a machine with four-bit coefficients.
//
// WHAT IS MEASURED. Time-to-solution in spin flips, for the sampler this crate added most recently
// (locally-informed proposals) against plain Gibbs at the same flip budget, on the same clamped
// circuit. Both arms are given the identical graph and the identical number of flips, and success
// is checked by MULTIPLYING THE FACTORS BACK -- not by reading an energy, because an energy near
// the floor is not a factorisation.

use ferrotherm::gibbs::Sampler;
use ferrotherm::graph::{Graph, GraphBuilder};
use ferrotherm::informed::{Balance, Informed};
use ferrotherm::invertible::{Multiplier, multiplier, read};

/// Clamp the product bits by pinning them with a bias far larger than anything else in the model,
/// and return the clamped graph.
///
/// A pin rather than a substitution: substituting the clamped variables out would give a smaller
/// graph, and would also mean the thing being sampled is no longer the circuit this module
/// verified. The pin is `2 x` the total coefficient magnitude, so no assignment can profit by
/// breaking it.
fn clamp(g: &Graph, product: &[usize], value: u64) -> Graph {
    let scale: f64 = g.w.iter().map(|w| w.abs()).sum::<f64>()
        + g.h.iter().map(|h| h.abs()).sum::<f64>();
    let pin = 2.0 * scale + 1.0;
    let mut b = GraphBuilder::new(g.n);
    for i in 0..g.n {
        if g.h[i] != 0.0 {
            b.bias(i, g.h[i]);
        }
        for k in g.offset[i]..g.offset[i + 1] {
            let j = g.nbr[k] as usize;
            if j > i {
                b.couple(i, j, g.w[k]);
            }
        }
    }
    for (i, &v) in product.iter().enumerate() {
        b.bias(v, if value >> i & 1 == 1 { pin } else { -pin });
    }
    b.build()
}

/// Did this state factor `target`? Checked by multiplying back, never by energy.
fn factored(m: &Multiplier, s: &[i8], target: u64) -> Option<(u64, u64)> {
    let (x, y) = (read(&m.a, s), read(&m.b, s));
    if x > 1 && y > 1 && x * y == target { Some((x, y)) } else { None }
}

fn main() {
    println!("A multiplier run backwards: factoring by sampling\n");

    for bits in [3usize, 4] {
        let m = multiplier(bits).unwrap();
        let (g, red) = m.circuit.to_graph().unwrap();
        println!(
            "{bits}x{bits} multiplier: {} circuit vars, {} gates -> {} spins ({} ancillas), {} couplings",
            m.circuit.vars(),
            m.circuit.gates(),
            g.n,
            red.ancillas,
            g.n_edges
        );
        // The sparsity that makes this the interesting construction.
        let density = 2.0 * g.n_edges as f64 / (g.n as f64 * (g.n as f64 - 1.0));
        let mut mags: Vec<f64> = g.w.iter().map(|w| w.abs()).collect();
        mags.sort_by(|a, b| a.partial_cmp(b).unwrap());
        mags.dedup_by(|a, b| (*a - *b).abs() < 1e-9);
        println!(
            "  density {:.2}%, {} distinct coupling magnitudes, largest {:.3}",
            100.0 * density,
            mags.len(),
            mags.last().copied().unwrap_or(0.0)
        );

        // Semiprimes that need every bit of both operands, so a factorisation is not trivial.
        let targets: Vec<u64> = (2..(1u64 << bits))
            .flat_map(|x| (2..(1u64 << bits)).map(move |y| (x, y)))
            .filter(|(x, y)| x <= y)
            .map(|(x, y)| x * y)
            .filter(|t| *t >= (1 << (2 * bits - 2)))
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();

        let flips = 400_000usize;
        let (mut gibbs_hits, mut inf_hits) = (0usize, 0usize);
        for &t in &targets {
            let cg = clamp(&g, &m.product, t);

            let mut s = Sampler::new(&cg, 3.0, 7);
            let mut hit = false;
            for _ in 0..(flips / cg.n) {
                s.sweeps(1, None);
                if factored(&m, red.project(&s.s), t).is_some() {
                    hit = true;
                    break;
                }
            }
            gibbs_hits += usize::from(hit);

            let mut it = Informed::new(&cg, 3.0, 7).with_balance(Balance::Barker);
            let mut hit = false;
            for _ in 0..(flips / cg.n) {
                it.steps(cg.n);
                if factored(&m, red.project(&it.s), t).is_some() {
                    hit = true;
                    break;
                }
            }
            inf_hits += usize::from(hit);
        }
        println!(
            "  factored within {flips} flips: gibbs {}/{}, informed {}/{}\n",
            gibbs_hits,
            targets.len(),
            inf_hits,
            targets.len()
        );
    }
}
