#![allow(missing_docs)]
// DOES ENTROPY PRODUCTION DETECT A BROKEN SAMPLER? Exactly: the population quantity every
// trajectory estimator converges to, for correct, wrong and broken kernels side by side.
//
// Stochastic thermodynamics says a kernel obeying detailed balance with respect to its stationary
// law produces no entropy in steady state, and the trajectory estimator sum ln[w(x->y)/w(y->x)]
// is the standard way to ask a running sampler whether it is 'in equilibrium'. The open question
// is what that number can see. Two facts decide it, and on 12 spins both are exact:
//
//   a fixed-order sweep of EXACT heat-bath updates is invariant but not reversible (the operator
//   P_B P_A is not self-adjoint), so it produces entropy while sampling the right law;
//   the fully synchronous sweep is reversible with respect to Peretto's law, so it produces none
//   while sampling a law 40% away from Boltzmann.
//
// The table puts each kernel's exact Sigma (nats per full update, with respect to its OWN law)
// beside the total variation of that law from the Boltzmann distribution it was meant to sample.
// If Sigma detected brokenness the two columns would rank alike.
//
// run: cargo run --release --example entropy_production_exact

use ferrotherm::autocorr::{boltzmann, entropy_production, own_law, total_variation, Kernel};
use ferrotherm::graph::{Graph, GraphBuilder};
use ferrotherm::rng::Pcg;

fn grid_glass(w: usize, h: usize, seed: u64) -> Graph {
    let mut rng = Pcg::new(seed, 0x6A);
    let mut b = GraphBuilder::new(w * h);
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            if x + 1 < w {
                b.couple(i, i + 1, if rng.f64() < 0.5 { -1.0 } else { 1.0 });
            }
            if y + 1 < h {
                b.couple(i, i + w, if rng.f64() < 0.5 { -1.0 } else { 1.0 });
            }
        }
    }
    for i in 0..w * h {
        b.bias(i, (rng.f64() - 0.5) * 0.4);
    }
    b.build()
}

/// A row of the table: the kernel and the temperature it runs at, given the nominal one.
type Row = (&'static str, &'static str, Box<dyn Fn(f64) -> (Kernel, f64)>);

fn main() {
    let g = grid_glass(4, 3, 7);
    println!("ENTROPY PRODUCTION AGAINST CORRECTNESS, EXACTLY, ON ALL 2^{} STATES\n", g.n);
    println!("  model    4x3 open grid, random +-1 couplings, fields in [-0.2, 0.2]");
    println!("  Sigma    sum pi(x) P(x,y) ln[pi(x) P(x,y) / (pi(y) P(y,x))] with pi the KERNEL'S OWN law, nats per full update");
    println!("  TV       distance of that law from the Boltzmann distribution the kernel was meant to sample\n");
    let rows: Vec<Row> = vec![
        ("chromatic sweep", "correct, fixed order", Box::new(|b| (Kernel::ChromaticGibbs, b))),
        ("sequential sweep", "correct, fixed order", Box::new(|b| (Kernel::SequentialGibbs, b))),
        ("chromatic at 1.01 beta", "a 1% temperature error", Box::new(|b| (Kernel::ChromaticGibbs, 1.01 * b))),
        ("shipped fabric", "Q.8 / 1024-ROM / 16-bit", Box::new(|b| (Kernel::FixedFabric, b))),
        ("stale reads p = 0.1", "collisions", Box::new(|b| (Kernel::Stale { p: 0.1 }, b))),
        ("site spread 0.2", "per-site temperatures", Box::new(|b| (Kernel::SiteSpread { seed: 1, spread: 0.2 }, b))),
        ("synchronous", "wrong law, reversible", Box::new(|b| (Kernel::Synchronous, b))),
        ("PIMI xi 0.5 eta 0.4", "inertia + probit", Box::new(|b| (Kernel::Pimi { xi: 0.5, eta: 0.4 }, b))),
    ];
    for &beta in &[0.5f64, 1.0, 2.0] {
        let target = boltzmann(&g, beta).expect("small");
        println!("  beta {beta}\n");
        println!("  {:<24} {:<24}   {:>12} {:>12}", "kernel", "what it is", "Sigma", "TV from B");
        for (name, what, mk) in &rows {
            let (k, b) = mk(beta);
            let sigma = entropy_production(&g, b, k).expect("small");
            let law = own_law(&g, b, k).expect("small");
            let tv = total_variation(&law, &target);
            println!("  {name:<24} {what:<24}   {sigma:>12.4e} {tv:>12.3e}");
        }
        println!();
    }
    println!("  WHAT THE TABLE SAYS.\n");
    println!("  Sigma ranks the kernels by how far they are from reversible, and TV ranks them by how wrong their");
    println!("  law is; the two orderings do not agree. The correct fixed-order sweeps produce entropy at every");
    println!("  temperature, the synchronous sweep produces none while sampling a law 40% away from Boltzmann, and");
    println!("  a 1% temperature error is invisible to Sigma because the chain at 1.01 beta is exactly as reversible");
    println!("  as the chain at beta. A trajectory estimator of Sigma is an instrument for non-reversibility. It");
    println!("  cannot certify a sampler, and a reading of zero is consistent with sampling the wrong distribution.");
    println!("  An 'inf' is a kernel with a transition that has no reverse: the fabric's comparator forbids a flip");
    println!("  once 2 beta f > 11.8, which a field of 2.95 reaches at beta 2, and the PIMI probit saturates to");
    println!("  exactly 0 and 1 in floating point.");
}
