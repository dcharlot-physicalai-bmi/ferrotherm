#![allow(missing_docs)]
// STALE READS, EXACTLY: how far a fabric drifts from Boltzmann when a fraction p of its neighbour
// reads lag one update behind, as a function of p, degree and temperature.
//
// Sequencerless and asynchronous p-bit fabrics are said to sample 'without compromising fidelity'
// (Sutton et al. 2020; Aadit et al. 2022); simultaneous updates of coupled cells are known to break
// invertible logic (Pervaiz et al. 2017) and to oscillate (Onizawa & Hanyu 2026). No published
// number gives the effective-temperature shift or the total-variation excess as a function of the
// collision fraction. `autocorr::Kernel::Stale { p }` is the collision model as an exact kernel: a
// sequential sweep in which each read of an already-updated neighbour returns the pre-sweep value
// with probability p. Its two ends are closed forms -- p = 0 is the sequential sweep (Boltzmann),
// p = 1 the synchronous one (Peretto) -- and everything between is one direct solve on 12 spins.
//
// Two fixtures: the 4x3 grid (degree <= 4) and K_{6,6} (degree 6, the denser end of what
// enumerates). Three temperatures. Six collision fractions. Reported per cell: TV of the stale
// equilibrium from Boltzmann, the nearest single temperature and what remains at it, and for
// three fractions Kemeny's constant against the p = 0 sweep.
//
// run: cargo run --release --example stale_exact

use ferrotherm::autocorr::{boltzmann, kemeny_constant, stationary_solved, total_variation, Kernel};
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

fn dense_bipartite(seed: u64) -> Graph {
    let mut rng = Pcg::new(seed, 0x6C);
    let mut b = GraphBuilder::new(12);
    for i in 0..6 {
        for j in 6..12 {
            b.couple(i, j, if rng.f64() < 0.5 { -1.0 } else { 1.0 });
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

fn run(name: &str, g: &Graph) {
    let degree = (0..g.n).map(|i| g.offset[i + 1] - g.offset[i]).max().unwrap_or(0);
    println!("  {name}: {} spins, max degree {degree}\n", g.n);
    println!(
        "  {:>5} {:>5}   {:>10} {:>9} {:>10} {:>12}   {:>12} {:>7}",
        "beta", "p", "TV(stale,B)", "beta_eff", "TV at eff", "dbeta/beta/p", "K stale", "K/K_seq"
    );
    for &beta in &[0.5f64, 1.0, 2.0] {
        let b = boltzmann(g, beta).expect("small");
        let k_seq = kemeny_constant(g, beta, Kernel::SequentialGibbs).expect("small");
        for &p in &[0.0f64, 0.01, 0.03, 0.1, 0.3, 1.0] {
            let k = Kernel::Stale { p };
            let law = stationary_solved(g, beta, k).expect("small");
            let tv = total_variation(&law, &b);
            let (b_eff, tv_eff) = nearest_beta(g, &law, 0.1 * beta, 2.0 * beta);
            let shift = if p > 0.0 { format!("{:>12.4}", (b_eff - beta) / beta / p) } else { format!("{:>12}", "-") };
            let kk = if p == 0.03 || p == 0.3 || p == 1.0 {
                let kk = kemeny_constant(g, beta, k).expect("small");
                format!("{kk:>12.4e} {:>7.3}", kk / k_seq)
            } else if p == 0.0 {
                format!("{k_seq:>12.4e} {:>7.3}", 1.0)
            } else {
                format!("{:>12} {:>7}", "-", "-")
            };
            println!("  {beta:>5.2} {p:>5.2}   {tv:>10.3e} {b_eff:>9.4} {tv_eff:>10.3e} {shift}   {kk}");
        }
    }
    println!();
}

fn main() {
    println!("STALE NEIGHBOUR READS AGAINST THE BOLTZMANN DISTRIBUTION, EXACTLY, ON ALL 2^12 STATES\n");
    println!("  kernel   Kernel::Stale {{ p }}: sequential sweep, each read of an already-updated neighbour stale with probability p");
    println!("  oracle   direct solve for the stationary law; Kemeny's constant against the p = 0 sweep\n");
    run("4x3 grid, +-1 couplings, fields in [-0.2, 0.2]", &grid_glass(4, 3, 7));
    run("K_{6,6}, +-1 couplings, fields in [-0.2, 0.2]", &dense_bipartite(5));
    println!("  WHAT THE TABLES SAY.\n");
    println!("  TV(stale,B) is the equilibrium error the collisions leave, the part no number of sweeps removes;");
    println!("  beta_eff is the temperature whose Boltzmann law is nearest and TV at eff what survives that shift,");
    println!("  so the two together say whether stale reads act as a temperature error (a certifiable, correctable");
    println!("  shift) or as a change of law. dbeta/beta/p is the shift per unit collision fraction: constant down");
    println!("  a column means the shift is linear in p there. K/K_seq is relaxation per sweep against the clean");
    println!("  sequential sweep; p = 1 is the fully synchronous sweep and its Peretto law.");
}
