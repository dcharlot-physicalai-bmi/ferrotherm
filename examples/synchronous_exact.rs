#![allow(missing_docs)]
// THE FULLY SYNCHRONOUS P-BIT UPDATE, EXACTLY: how far its equilibrium sits from Boltzmann, and
// how fast it gets there, against the chromatic sweep at the same number of site updates.
//
// A p-bit array with every node on one clock edge and no colouring runs Little's dynamics: each
// site resampled from the PREVIOUS state. The claim in the hardware literature is that this
// samples the Boltzmann distribution well enough at matched sweep count. Its stationary law is
// known in closed form for symmetric couplings (Peretto 1984), `autocorr::peretto`, and it is not
// the Boltzmann law even on a bipartite graph. On 12 spins everything below is exact:
//
//   TV(sync, B)   distance of the synchronous equilibrium from the Boltzmann law it was meant for
//   beta_eff      the single temperature whose Boltzmann law is nearest; TV at eff what remains
//   TV(fab, B)    the shipped fabric's arithmetic error at the same temperature, for scale
//   K             Kemeny's constant -- the sum of every mode's relaxation time -- for the synchronous
//                 step and for the chromatic sweep, both of which update every site once
//   drift         TV(pi P, pi) for Peretto's law: the closed form checked against the kernel
//
// Two fixtures: the bipartite 4x3 grid `fabric_exact` uses, and a 12-ring with three chords that
// make odd cycles, since the bipartite case is the one the synchronous update is usually excused on.
//
// run: cargo run --release --example synchronous_exact

use ferrotherm::autocorr::{apply_distribution, boltzmann, kemeny_constant, peretto, stationary_solved, total_variation, Kernel};
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

/// A 12-ring with chords (0,4), (2,7), (5,10): odd cycles, so not bipartite.
fn chorded_ring(seed: u64) -> Graph {
    let n = 12;
    let mut rng = Pcg::new(seed, 0x6B);
    let mut b = GraphBuilder::new(n);
    let mut edges: Vec<(usize, usize)> = (0..n).map(|i| (i, (i + 1) % n)).collect();
    edges.extend([(0, 4), (2, 7), (5, 10)]);
    for (i, j) in edges {
        b.couple(i, j, if rng.f64() < 0.5 { -1.0 } else { 1.0 });
    }
    for i in 0..n {
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

fn run(name: &str, g: &Graph) {
    println!("  {name}: {} spins, {} colour classes\n", g.n, g.classes.len());
    println!(
        "  {:>5}   {:>10} {:>9} {:>10}   {:>10}   {:>12} {:>12} {:>7}   {:>8}",
        "beta", "TV(sync,B)", "beta_eff", "TV at eff", "TV(fab,B)", "K sync", "K chromatic", "ratio", "drift"
    );
    for &beta in &[0.5f64, 1.0, 1.5, 2.0, 3.0] {
        let pi = peretto(g, beta).expect("small");
        let drift = total_variation(&apply_distribution(g, beta, Kernel::Synchronous, &pi), &pi);
        let b = boltzmann(g, beta).expect("small");
        let tv = total_variation(&pi, &b);
        let (b_eff, tv_eff) = nearest_beta(g, &pi, 0.25 * beta, 2.0 * beta);
        let fab = total_variation(&stationary_solved(g, beta, Kernel::FixedFabric).expect("small"), &b);
        let k_sync = kemeny_constant(g, beta, Kernel::Synchronous).expect("small");
        let k_chrom = kemeny_constant(g, beta, Kernel::ChromaticGibbs).expect("small");
        println!(
            "  {beta:>5.2}   {tv:>10.3e} {b_eff:>9.4} {tv_eff:>10.3e}   {fab:>10.3e}   {k_sync:>12.4e} {k_chrom:>12.4e} {:>7.3}   {drift:>8.1e}",
            k_sync / k_chrom
        );
    }
    println!();
}

fn main() {
    println!("THE SYNCHRONOUS UPDATE AGAINST THE CHROMATIC SWEEP, EXACTLY, ON ALL 2^12 STATES\n");
    println!("  kernels  Synchronous = Little's dynamics (every site from the previous state); ChromaticGibbs = the sweep the fabric runs");
    println!("  oracle   Peretto's closed form for the synchronous law (checked by 'drift'), direct solves, Kemeny's constant\n");
    run("4x3 open grid, +-1 couplings, fields in [-0.2, 0.2] (bipartite)", &grid_glass(4, 3, 7));
    run("12-ring with chords (0,4) (2,7) (5,10), +-1 couplings, fields in [-0.2, 0.2] (odd cycles)", &chorded_ring(3));
    println!("  WHAT THE TABLES SAY.\n");
    println!("  TV(sync,B) is the equilibrium error of updating every site at once: the part of the synchronous");
    println!("  sampler's output that no number of sweeps removes, beside the shipped fabric's arithmetic error");
    println!("  TV(fab,B) at the same temperature. beta_eff and TV at eff say how much of it is a temperature shift.");
    println!("  K is Kemeny's constant per step; a synchronous step and a chromatic sweep both update every site");
    println!("  once, so the ratio is relaxation per unit of work. 'drift' is the closed form's invariance under");
    println!("  the kernel and should sit at floating point.");
}
