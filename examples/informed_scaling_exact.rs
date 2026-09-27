#![allow(missing_docs)]
// THE INFORMED PROPOSAL AGAINST GIBBS WITH NO ESTIMATOR IN THE WAY: exact tau_int of both
// kernels, by linear algebra on the enumerated state space.
//
// `informed_scaling` measured the same comparison with `certify::tau_int` on sampled traces, and
// then `tau_exactness` found that estimator reporting a twentieth of the true autocorrelation time
// on a glassy chain, at every trace length, with no remedy inside the trace. So the sampled table
// is suspect on both arms, in unknown and possibly different proportions. This one has no arms to
// suspect: `autocorr::tau_int_exact` builds each kernel as a linear operator on all 2^n states and
// sums the exact autocorrelation of the energy until the tail is below floating point. The only
// limit is n.
//
// UNITS. Gibbs's tau is in sweeps and one sweep is n flips; the informed chain's tau is in steps
// and one step is one flip. Both are converted to FLIPS. Work per flip: deg+1 field terms for
// Gibbs; deg+1 reweighs plus the choice for the informed chain -- n weight reads under the linear
// scan that shipped until 2026-09-13, log2(n) under the Fenwick tree it has now.
//
// TIME. The full run is more than three hours of one core (killed at 10,800 s on 2026-09-27, inside
// n = 14, where the lag sum takes the place of the dense solve). It is listed in examples/LOCAL and
// run by hand; the weekly job runs it with FERROTHERM_EXAMPLE_SMOKE=1 -- one seed and lag caps a
// thousand times smaller, n = 14 included -- which proves every line still runs and measures nothing.
//
// run: cargo run --release --example informed_scaling_exact

use ferrotherm::autocorr::{tau_int_exact, tau_int_fundamental, Kernel, MAX_DENSE_SPINS};
use ferrotherm::graph::{Graph, GraphBuilder};
use ferrotherm::informed::Balance;
use ferrotherm::rng::Pcg;

/// The `informed_mixing` fixture at any size: a frustrated ring with n/4 random chords and fields.
fn frustrated(n: usize, seed: u64) -> Graph {
    let mut rng = Pcg::new(seed, 0xF5);
    let mut b = GraphBuilder::new(n);
    for i in 0..n {
        b.couple(i, (i + 1) % n, if rng.f64() < 0.5 { -1.0 } else { 1.0 });
    }
    for _ in 0..n / 4 {
        let (i, j) = ((rng.f64() * n as f64) as usize % n, (rng.f64() * n as f64) as usize % n);
        if i != j {
            b.couple(i, j, if rng.f64() < 0.5 { -1.0 } else { 1.0 });
        }
    }
    for i in 0..n {
        b.bias(i, (rng.f64() - 0.5) * 0.4);
    }
    b.build()
}

fn main() {
    let beta = 2.0;
    let smoke = std::env::var_os("FERROTHERM_EXAMPLE_SMOKE").is_some();
    let seeds = if smoke { 1u64 } else { 4 };
    let (cap_g, cap_b) = if smoke { (2_000, 20_000) } else { (2_000_000, 20_000_000) };
    if smoke {
        println!("SMOKE RUN (FERROTHERM_EXAMPLE_SMOKE): one seed and lag caps a thousand times smaller; these numbers are not the measurement\n");
    }
    println!("EXACT tau_int: BARKER-INFORMED AGAINST CHROMATIC GIBBS, IN FLIPS, ON ALL 2^n STATES\n");
    println!("  fixture   frustrated ring with n/4 chords and fields (the informed_mixing fixture), beta = {beta}");
    println!("  oracle    autocorr::tau_int_fundamental (one dense solve, no lags) up to {MAX_DENSE_SPINS} spins; the lag sum of");
    println!("            autocorr::tau_int_exact (tail cut 1e-12) above it; {seeds} seeds averaged; no sampling, no window\n");
    println!(
        "  {:>4} {:>5}   {:>12} {:>12} {:>8}   {:>9} {:>9}   {:>8} {:>8}",
        "n", "deg", "tau G flips", "tau B flips", "G/B", "lags G", "lags B", "adv:scan", "adv:tree"
    );
    for &n in &[6usize, 8, 10, 12, 14] {
        let (mut tg, mut tb, mut deg, mut lg, mut lb) = (0.0, 0.0, 0.0, 0usize, 0usize);
        // The direct solve where the dense operator fits; the lag sum above that, with its caps.
        let tau = |g: &ferrotherm::graph::Graph, kernel: Kernel, cap: usize| {
            if g.n <= MAX_DENSE_SPINS {
                tau_int_fundamental(g, beta, kernel, |s| g.energy(s)).expect("n is under the cap and the energy varies")
            } else {
                tau_int_exact(g, beta, kernel, |s| g.energy(s), 1e-12, cap).expect("n is under the cap and the energy varies")
            }
        };
        for seed in 0..seeds {
            let g = frustrated(n, seed);
            deg += 2.0 * g.n_edges as f64 / n as f64;
            let a = tau(&g, Kernel::ChromaticGibbs, cap_g);
            let b = tau(&g, Kernel::Informed(Balance::Barker), cap_b);
            tg += a.tau_int * n as f64;
            tb += b.tau_int;
            lg = lg.max(a.lags);
            lb = lb.max(b.lags);
        }
        let s = seeds as f64;
        let (tg, tb, deg) = (tg / s, tb / s, deg / s);
        let per_g = deg + 1.0;
        let per_b_scan = deg + 1.0 + n as f64;
        let per_b_tree = deg + 1.0 + (n as f64).log2();
        println!(
            "  {n:>4} {deg:>5.2}   {tg:>12.1} {tb:>12.1} {:>8.2}   {lg:>9} {lb:>9}   {:>8.2} {:>8.2}",
            tg / tb,
            (tg * per_g) / (tb * per_b_scan),
            (tg * per_g) / (tb * per_b_tree)
        );
    }
    println!("\n  Every number above is exact for its model: the ratio column is a fact about the two kernels");
    println!("  at that size, not an estimate of one. Where the sampled table (`informed_scaling`) and this");
    println!("  one disagree at a shared size, the sampled one is the one that was wrong.");
}
