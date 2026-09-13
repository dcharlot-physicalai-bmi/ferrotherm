#![allow(missing_docs)]
// DOES THE PIMI RULE SAMPLE THE BOLTZMANN DISTRIBUTION AT ANY INERTIA? Exactly, on 12 spins.
//
// Zhu, Singh et al. (arXiv:2604.17109, 2026) let every p-bit update at once by adding a self-spin
// inertia term: s_i(t+1) = sign[tanh(beta f_i) + xi s_i + eta N(0,1)]. They report 35x speed-ups
// on Max-Cut, SK and MIMO and do not analyse the stationary distribution. Fully synchronous
// Glauber dynamics without the inertia has Peretto's law, not Boltzmann (`synchronous_exact`);
// the inertia and the probit-of-tanh change the law again. `autocorr::Kernel::Pimi` is that
// kernel, and on 12 spins its stationary law is one direct solve, so the sampling half of the
// claim has an exact answer: the total variation from Boltzmann of the PIMI equilibrium, as a
// function of the inertia xi and the noise eta, beside the fully synchronous logistic sweep and
// the shipped fabric's arithmetic at the same temperature.
//
// eta is not a free choice. At xi = 0 the PIMI marginal is Phi(tanh(beta f) / eta) against the
// heat bath's sigma(2 beta f); matching their slopes at f = 0 gives eta = 2 / sqrt(2 pi) = 0.80,
// and because tanh saturates at 1 no field can then make a spin more than Phi(1 / 0.80) = 89%
// certain. Smaller eta sharpens the rule and steepens it at the origin. The sweep runs eta in
// {0.2, 0.4, 0.8} to show both effects.
//
// run: cargo run --release --example pimi_exact

use ferrotherm::autocorr::{boltzmann, kemeny_constant, peretto, stationary_solved, total_variation, Kernel};
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

/// K_{6,6}: dense bipartite, the shape the paper's MIMO and Max-Cut instances take, on 12 spins.
fn dense_bipartite(seed: u64) -> Graph {
    let mut rng = Pcg::new(seed, 0x6C);
    let mut b = GraphBuilder::new(12);
    for i in 0..6 {
        for j in 6..12 {
            b.couple(i, j, (rng.f64() - 0.5) * 2.0 / 6f64.sqrt());
        }
    }
    for i in 0..12 {
        b.bias(i, (rng.f64() - 0.5) * 0.4);
    }
    b.build()
}

fn nearest_beta(g: &Graph, law: &[f64], lo: f64, hi: f64) -> (f64, f64) {
    let f = |b: f64| total_variation(law, &boltzmann(g, b).expect("small"));
    let phi = 0.5 * (3.0 - 5.0f64.sqrt());
    let (mut a, mut b) = (lo, hi);
    let mut c = a + phi * (b - a);
    let mut d = b - phi * (b - a);
    let (mut fc, mut fd) = (f(c), f(d));
    for _ in 0..60 {
        if fc < fd {
            b = d;
            d = c;
            fd = fc;
            c = a + phi * (b - a);
            fc = f(c);
        } else {
            a = c;
            c = d;
            fc = fd;
            d = b - phi * (b - a);
            fd = f(d);
        }
    }
    let best = 0.5 * (a + b);
    (best, f(best))
}

fn run(name: &str, g: &Graph, beta: f64) {
    let b = boltzmann(g, beta).expect("small");
    let sync = total_variation(&peretto(g, beta).expect("small"), &b);
    let fab = total_variation(&stationary_solved(g, beta, Kernel::FixedFabric).expect("small"), &b);
    let k_chrom = kemeny_constant(g, beta, Kernel::ChromaticGibbs).expect("small");
    println!("  {name}, beta {beta}: TV(sync logistic, B) = {sync:.3e}; TV(fabric, B) = {fab:.3e}; K chromatic = {k_chrom:.3e}\n");
    println!(
        "  {:>5} {:>5}   {:>10} {:>9} {:>10}   {:>12} {:>9}",
        "eta", "xi", "TV(pimi,B)", "beta_eff", "TV at eff", "K pimi", "ratio"
    );
    for &eta in &[0.2f64, 0.4, 0.8] {
        for &xi in &[0.0f64, 0.1, 0.25, 0.5, 1.0] {
            let k = Kernel::Pimi { xi, eta };
            // A rule whose spins can no longer flip to floating point has no unique invariant
            // law: the solve reports the singular system, and the row says so rather than
            // inventing a distribution for a frozen chain.
            let Ok(law) = stationary_solved(g, beta, k) else {
                println!("  {eta:>5.2} {xi:>5.2}   {:>10} {:>9} {:>10}   {:>12} {:>9}   frozen: no unique invariant law (singular to 1e-14)", "-", "-", "-", "-", "-");
                continue;
            };
            let tv = total_variation(&law, &b);
            let (b_eff, tv_eff) = nearest_beta(g, &law, 0.05 * beta, 16.0 * beta);
            let kk = if xi == 0.0 || xi == 0.5 || xi == 1.0 {
                kemeny_constant(g, beta, k).map_or_else(|_| format!("{:>12} {:>9}", "-", "frozen"), |kk| format!("{kk:>12.4e} {:>9.3e}", kk / k_chrom))
            } else {
                format!("{:>12} {:>9}", "-", "-")
            };
            println!("  {eta:>5.2} {xi:>5.2}   {tv:>10.3e} {b_eff:>9.4} {tv_eff:>10.3e}   {kk}");
        }
    }
    println!();
}

fn main() {
    println!("THE PIMI RULE AGAINST THE BOLTZMANN DISTRIBUTION, EXACTLY, ON ALL 2^12 STATES\n");
    println!("  kernel   Kernel::Pimi {{ xi, eta }}: every site at once, P(+1) = Phi((tanh(beta f) + xi s) / eta)");
    println!("  oracle   direct solve for the stationary law; Kemeny's constant for relaxation per full update\n");
    run("4x3 grid, +-1 couplings, fields in [-0.2, 0.2]", &grid_glass(4, 3, 7), 1.0);
    run("K_{6,6}, Gaussian couplings of scale 1/sqrt(6), fields in [-0.2, 0.2]", &dense_bipartite(5), 1.0);
    println!("  WHAT THE TABLES SAY.\n");
    println!("  TV(pimi,B) is the equilibrium error of the PIMI rule, the part no number of updates removes; the");
    println!("  'sampling floor' the claim would have to beat is the two-run noise of a finite sample, and these are");
    println!("  exact, so any value visibly above 1e-3 is a law the rule cannot sample. beta_eff and TV at eff say");
    println!("  whether the error is a temperature shift. The xi = 0 rows isolate the probit-of-tanh nonlinearity");
    println!("  from the inertia; the fully synchronous logistic sweep and the fabric's arithmetic are printed above");
    println!("  each table for scale. K is relaxation per full update against the chromatic sweep's.");
}
