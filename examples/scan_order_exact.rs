//! **Does the ORDER a fabric visits its p-bits in matter, at equal work?**
//!
//! A p-bit fabric updates its sites in a fixed order -- a systematic scan. Most of the mixing-time
//! theory was written for the random scan, which picks each site with a random number, but the
//! systematic scan has results of its own going back at least to Dyer, Goldberg and Jerrum
//! (arXiv:math/0603323, 2006), who bound its mixing time for colourings of a path and note that it
//! *"fails to be time reversible"*. Blanca, Caputo, Sinclair and Vigoda (arXiv:1612.01576, 2016)
//! proved that strong spatial mixing implies `O(log n)` mixing of systematic scan (under mild
//! conditions) on `n`-vertex cubes of `Z^d`. Blanca and Rafid extend it in 2026: mixing and cutoff
//! for the mean-field Potts model (arXiv:2607.09841), and `O(log n)` mixing for every scan order
//! from approximate tensorisation of entropy, under standard marginal, connectivity and
//! bounded-degree assumptions (arXiv:2609.05750). That last paper still calls the theory *"far less
//! developed than that of Glauber dynamics"*, and notes that the systematic scan *"is often favored
//! in practice because it exhibits strong empirical performance"*. This measures that performance
//! exactly, at equal work: the integrated autocorrelation time of the energy and of the
//! magnetisation PER SITE UPDATE, for one sweep in a fixed order (`n` updates) against `n`
//! random-scan steps, from `beta = 0.2` to `3` on a 5x2 ferromagnet, a 5x2 +-J glass and a 4x2
//! ferromagnet.
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
    let fixtures = [
        ("5x2 ferromagnet", grid(5, 2, false, 1)),
        ("5x2 +-J glass", grid(5, 2, true, 7)),
        ("4x2 ferromagnet", grid(4, 2, false, 1)),
    ];
    println!("tau_int per SITE UPDATE: a fixed-order sweep (n updates) against n random-scan steps");
    println!("  fixture            beta   energy: fixed    random  ratio   magnetisation: fixed        random  ratio");
    for (name, g) in &fixtures {
        let n = g.n as f64;
        for &beta in &[0.2f64, 0.4, 0.7, 1.0, 1.5, 2.0, 3.0] {
            let e = |k: Kernel| tau_int_fundamental(g, beta, k, |s| g.energy(s)).expect("dense").tau_int;
            let mg = |k: Kernel| tau_int_fundamental(g, beta, k, |s| s.iter().map(|&v| f64::from(v)).sum()).expect("dense").tau_int;
            // Per site update: a sweep's tau is in sweeps of n updates; a random-scan step is one.
            let (ef, er) = (e(Kernel::SequentialGibbs) * n, e(Kernel::RandomScan));
            let (mf, mr) = (mg(Kernel::SequentialGibbs) * n, mg(Kernel::RandomScan));
            println!(
                "  {name:17}  {beta:4}   {ef:13.2}  {er:8.2}  {:5.3}   {mf:20.2}  {mr:12.2}  {:5.3}",
                er / ef,
                mr / mf
            );
        }
    }
}
