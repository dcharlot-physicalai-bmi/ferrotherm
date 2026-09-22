//! **Does time-multiplexed reuse change the law, or only the clock?**
//!
//! Onizawa & Hanyu (arXiv:2604.01564, 2026) introduce a time-multiplexing reuse factor `c` — "the
//! number of logical p-bits that are sequentially mapped onto a single physical p-bit" — and argue
//! it is free:
//!
//! > time-multiplexed reuse corresponds to a temporal rescaling of the underlying Markov process
//! > rather than a change in its transition kernel.
//!
//! > Such time-thinning arguments are well established in stochastic simulation theory and imply
//! > that only the convergence speed, not the stationary distribution, is affected.
//!
//! Their Conclusion restates it: "the effective update rate can be reduced without altering the
//! target stationary distribution".
//!
//! For the ASYNCHRONOUS (Poisson) branch that is right — thinning a Poisson process really is a
//! time change. For the SYNCHRONOUS tick-random branch it is not, and this example is the exact
//! arithmetic that settles it: `p_flip = 1/c` sits inside the per-site conditional, so the
//! transition kernel itself moves with `c`. The two fixtures are the ones
//! `examples/synchronous_exact.rs` already uses, so the `p = 1` column is directly comparable.
//!
//! ```text
//! cargo run --release --example tick_random_exact
//! ```
use ferrotherm::autocorr::{boltzmann, peretto, stationary_solved, total_variation, AutocorrError, Kernel};
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
    let mut rng = Pcg::new(seed, 0x6A);
    let mut b = GraphBuilder::new(n);
    for i in 0..n {
        b.couple(i, (i + 1) % n, if rng.f64() < 0.5 { -1.0 } else { 1.0 });
    }
    for &(i, j) in &[(0usize, 4usize), (2, 7), (5, 10)] {
        b.couple(i, j, if rng.f64() < 0.5 { -1.0 } else { 1.0 });
    }
    for i in 0..n {
        b.bias(i, (rng.f64() - 0.5) * 0.4);
    }
    b.build()
}

/// The same ring with every coupling set to zero: the sites are independent, so the tick-random
/// law is the product of per-site Boltzmann marginals at EVERY p. The control that says the
/// movement below is the interaction and not the mask.
fn uncoupled(n: usize, seed: u64) -> Graph {
    let mut rng = Pcg::new(seed, 0x6A);
    let mut b = GraphBuilder::new(n);
    for i in 0..n {
        b.bias(i, (rng.f64() - 0.5) * 0.4);
    }
    b.build()
}

const PS: [(f64, &str); 6] =
    [(1.0, "1"), (0.8, "1/1.25"), (2.0 / 3.0, "1/1.5"), (0.5, "1/2"), (1.0 / 3.0, "1/3"), (0.1, "1/10")];

fn run(name: &str, g: &Graph) {
    println!("\n  {name}: {} spins\n", g.n);
    println!("        {:>7}   {:>10} {:>10}   {:>12}", "p = 1/c", "TV vs Bolt", "TV vs Per", "TV vs p=1");
    println!("                                              <- the paper says THIS column is zero");
    for &beta in &[0.5f64, 1.0, 2.0, 3.0] {
        let bolt = boltzmann(g, beta).expect("small");
        let per = peretto(g, beta).expect("small");
        let mut prev: Option<(f64, f64)> = None;
        let (mut mono_b, mut mono_p) = (true, true);
        let at_one = stationary_solved(g, beta, Kernel::TickRandom { p: 1.0 }).expect("solvable");
        println!("  beta {beta}");
        for &(p, label) in &PS {
            let law = stationary_solved(g, beta, Kernel::TickRandom { p }).expect("solvable");
            let tb = total_variation(&law, &bolt);
            let tp = total_variation(&law, &per);
            let t1 = total_variation(&law, &at_one);
            if let Some((pb, pp)) = prev {
                // p is descending through PS, so "as p falls" is this row against the last.
                if tb >= pb {
                    mono_b = false;
                }
                if tp <= pp {
                    mono_p = false;
                }
            }
            prev = Some((tb, tp));
            println!("        {label:>7}   {tb:>10.6} {tp:>10.6}   {t1:>12.6}");
        }
        println!(
            "                  TV-from-Boltzmann falls with p: {}     TV-from-Peretto rises with p: {}",
            if mono_b { "yes" } else { "NO" },
            if mono_p { "yes" } else { "NO" }
        );
    }
}

fn main() {
    println!("TICK-RANDOM (arXiv:2604.01564): does p = 1/c move the stationary law?");
    println!("  p = 1 is every site every tick (Synchronous, Peretto's law).");
    println!("  p -> 0 is at most one site at a time (the Boltzmann limit).");

    run("4x3 grid, +-1 couplings (bipartite)", &grid_glass(4, 3, 7));
    run("12-ring with chords (0,4) (2,7) (5,10) (odd cycles)", &chorded_ring(3));

    println!("\n  CONTROL -- 12 uncoupled sites: the law must not move with p at all.");
    let u = uncoupled(12, 3);
    for &beta in &[0.5f64, 1.0, 2.0, 3.0] {
        let bolt = boltzmann(&u, beta).expect("small");
        let worst = PS
            .iter()
            .map(|&(p, _)| total_variation(&stationary_solved(&u, beta, Kernel::TickRandom { p }).expect("solvable"), &bolt))
            .fold(0.0f64, f64::max);
        println!("        beta {beta}: worst TV from Boltzmann over all p = {worst:.3e}");
    }

    println!("\n  p = 1 must BE the synchronous kernel, and this is the residual of that identity:");
    for (nm, g) in [("grid", grid_glass(4, 3, 7)), ("ring", chorded_ring(3))] {
        for &beta in &[0.5f64, 1.0, 2.0, 3.0] {
            let a = stationary_solved(&g, beta, Kernel::TickRandom { p: 1.0 }).expect("solvable");
            let b = stationary_solved(&g, beta, Kernel::Synchronous).expect("solvable");
            let c = peretto(&g, beta).expect("small");
            println!("        {nm} beta {beta}: TV(tick p=1, synchronous) = {:.3e}   TV(tick p=1, peretto) = {:.3e}", total_variation(&a, &b), total_variation(&a, &c));
        }
    }

    println!("\n  p = 0 is the identity map: no unique invariant law, and it must SAY so.");
    let g = chorded_ring(3);
    match stationary_solved(&g, 1.0, Kernel::TickRandom { p: 0.0 }) {
        Ok(_) => println!("        ⛔ returned a law for the identity map"),
        Err(e) => println!("        refused, as it must: {e:?}"),
    }

    println!("\n  Small-p shape: TV/p against Boltzmann should tend to a finite non-zero limit");
    println!("  (the law moves LINEARLY in p, with no intercept).");
    for &beta in &[1.0f64] {
        let bolt = boltzmann(&g, beta).expect("small");
        for &p in &[0.01f64, 0.02, 0.05, 0.1] {
            let law = stationary_solved(&g, beta, Kernel::TickRandom { p }).expect("solvable");
            let tv = total_variation(&law, &bolt);
            println!("        beta {beta} p {p:>5}: TV = {tv:.6}   TV/p = {:.6}", tv / p);
        }
    }
    let _ = AutocorrError::TooManySpins { n: 0, max: 0 };
}
