//! **Does the ORDER a fabric visits its p-bits in matter, at equal work?**
//!
//! A p-bit fabric updates its sites in a fixed order -- a systematic scan. The mixing-time theory
//! was built for the random scan, which picks each site with a random number, and has only recently
//! reached the systematic one: Blanca and Rafid (arXiv:2609.05750) prove `O(log n)` mixing
//! and an `O(1)` relaxation time per scan under approximate tensorisation, and note that the
//! systematic scan *"is often favored in practice because it exhibits strong empirical
//! performance"*. This measures that performance exactly, at equal work: the integrated
//! autocorrelation time of the energy and of the magnetisation PER SITE UPDATE, for one sweep in a
//! fixed order (`n` updates) against `n` random-scan steps.
//!
//! ```text
//! cargo run --release --example scan_order_exact
//! ```
use ferrotherm::autocorr::{tau_int_fundamental, Kernel};
use ferrotherm::graph::{Graph, GraphBuilder};
use ferrotherm::rng::Pcg;

fn grid(w: usize, h: usize, glass: bool, seed: u64) -> Graph {
    let mut rng = Pcg::new(seed, 0x6A);
    let mut b = GraphBuilder::new(w * h);
    let mut j = || if glass && rng.f64() < 0.5 { -1.0 } else { 1.0 };
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            if x + 1 < w {
                b.couple(i, i + 1, j());
            }
            if y + 1 < h {
                b.couple(i, i + w, j());
            }
        }
    }
    b.build()
}

fn main() {
    let fixtures = [("5x2 ferromagnet", grid(5, 2, false, 1)), ("5x2 +-J glass", grid(5, 2, true, 7))];
    println!("tau_int per SITE UPDATE: a fixed-order sweep (n updates) against n random-scan steps");
    println!("  fixture            beta   energy: fixed  random  ratio   magnetisation: fixed  random  ratio");
    for (name, g) in &fixtures {
        let n = g.n as f64;
        for &beta in &[0.2f64, 0.4, 0.7, 1.0, 1.5] {
            let e = |k: Kernel| tau_int_fundamental(g, beta, k, |s| g.energy(s)).expect("dense").tau_int;
            let mg = |k: Kernel| tau_int_fundamental(g, beta, k, |s| s.iter().map(|&v| f64::from(v)).sum()).expect("dense").tau_int;
            // Per site update: a sweep's tau is in sweeps of n updates; a random-scan step is one.
            let (ef, er) = (e(Kernel::SequentialGibbs) * n, e(Kernel::RandomScan));
            let (mf, mr) = (mg(Kernel::SequentialGibbs) * n, mg(Kernel::RandomScan));
            println!(
                "  {name:17}  {beta:4}   {ef:13.2}  {er:6.2}  {:5.2}   {mf:20.2}  {mr:6.2}  {:5.2}",
                er / ef,
                mr / mf
            );
        }
    }
}
