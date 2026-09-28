#![allow(missing_docs)]
// THE INFORMED PROPOSAL AGAINST GIBBS WITH NO ESTIMATOR IN THE WAY: exact tau_int of both
// kernels, by linear algebra on the enumerated state space.
//
// `informed_scaling` measured the same comparison with `certify::tau_int` on sampled traces, and
// then `tau_exactness` found that estimator reporting a twentieth of the true autocorrelation time
// on a glassy chain, at every trace length, with no remedy inside the trace. So the sampled table
// is suspect on both arms, in unknown and possibly different proportions. This one has no arms to
// suspect: each kernel is a linear operator on all 2^n states, and tau_int of the energy is one
// solve of the fundamental-matrix system (I - P + 1 pi^T) z = e. The only limit is n.
//
// UNITS. Gibbs's tau is in sweeps and one sweep is n flips; the informed chain's tau is in steps
// and one step is one flip. Both are converted to FLIPS. Work per flip: deg+1 field terms for
// Gibbs; deg+1 reweighs plus the choice for the informed chain -- n weight reads under the linear
// scan that shipped until 2026-09-13, log2(n) under the Fenwick tree it has now.
//
// ROUTE. `autocorr::tau_int_solved`: the system solved MATRIX-FREE, conjugate gradients for the
// informed chain (reversible) and GMRES for the chromatic sweep (not), to an estimated relative
// error of 1e-8, and the censored solve wherever f64 cannot certify that. The dense solve,
// `tau_int_fundamental`, is the cross-check column up to n = 10; at 12 it takes minutes per call
// and on a cold chain it is the least accurate of the routes.
//
// HISTORY. Until 2026-09-28 this example stopped at n = 14, reached by the lag sum of
// `tau_int_exact`: one application of the kernel per lag, about tau ln(1/tol) of them, more than
// three hours of one core at n = 14 (it was killed at 10,800 s and listed in examples/LOCAL). The
// Krylov solve needs a number of applications that grows with ln tau instead -- tens to a few
// hundred -- which is what lets the table run to n = 20.
//
// run: cargo run --release --example informed_scaling_exact

use ferrotherm::autocorr::{tau_int_fundamental, tau_int_solved, Autocorrelation, Kernel, Route};
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

fn route_name(r: Route) -> &'static str {
    match r {
        Route::Cg => "cg",
        Route::Gmres => "gmres",
        Route::Censored => "censored",
        Route::Dense => "dense",
        Route::LagSum => "lag sum",
    }
}

fn main() {
    let beta = 2.0;
    let seeds = 4u64;
    let rtol = 1e-8;
    println!("EXACT tau_int: BARKER-INFORMED AGAINST CHROMATIC GIBBS, IN FLIPS, ON ALL 2^n STATES\n");
    println!("  fixture   frustrated ring with n/4 chords and fields (the informed_mixing fixture), beta = {beta}");
    println!("  solve     autocorr::tau_int_solved: (I - P + 1 pi^T) z = e matrix-free, CG for the informed chain and GMRES");
    println!("            for the sweep, to an estimated relative error of {rtol:.0e}, the censored solve where f64 cannot");
    println!("            certify that; {seeds} seeds averaged; no sampling, no window. dense = tau_int_fundamental, n <= 10\n");
    println!(
        "  {:>4} {:>5}   {:>12} {:>12} {:>6}   {:>8} {:>8}   {:>15} {:>6} {:>9} {:>9}",
        "n", "deg", "tau G flips", "tau B flips", "G/B", "adv:scan", "adv:tree", "routes G / B", "apps", "est rel", "vs dense"
    );
    let mut ratios = Vec::new();
    for n in (6usize..=20).step_by(2) {
        let (mut tg, mut tb, mut deg) = (0.0, 0.0, 0.0);
        let (mut apps, mut est, mut dense_gap) = (0usize, 0.0f64, 0.0f64);
        let mut routes: [Vec<&str>; 2] = [Vec::new(), Vec::new()];
        for seed in 0..seeds {
            let g = frustrated(n, seed);
            deg += 2.0 * g.n_edges as f64 / n as f64;
            for (arm, kernel) in [Kernel::ChromaticGibbs, Kernel::Informed(Balance::Barker)].into_iter().enumerate() {
                let a: Autocorrelation = tau_int_solved(&g, beta, kernel, |s| g.energy(s), rtol, 20_000)
                    .unwrap_or_else(|e| panic!("n {n} seed {seed} {kernel:?}: {e}"));
                if n <= 10 {
                    let d = tau_int_fundamental(&g, beta, kernel, |s| g.energy(s)).expect("n is under the dense cap");
                    dense_gap = dense_gap.max((a.tau_int - d.tau_int).abs() / (d.tau_int + 0.5));
                }
                apps = apps.max(a.matvecs);
                if let Some(e) = a.err_est {
                    est = est.max(e / (a.tau_int + 0.5));
                }
                let name = route_name(a.route);
                if !routes[arm].contains(&name) {
                    routes[arm].push(name);
                }
                if arm == 0 {
                    tg += a.tau_int * n as f64;
                } else {
                    tb += a.tau_int;
                }
            }
        }
        let s = seeds as f64;
        let (tg, tb, deg) = (tg / s, tb / s, deg / s);
        let per_g = deg + 1.0;
        let per_b_scan = deg + 1.0 + n as f64;
        let per_b_tree = deg + 1.0 + (n as f64).log2();
        let gap = if n <= 10 { format!("{dense_gap:.1e}") } else { "-".to_string() };
        println!(
            "  {n:>4} {deg:>5.2}   {tg:>12.1} {tb:>12.1} {:>6.2}   {:>8.2} {:>8.2}   {:>15} {apps:>6} {est:>9.1e} {gap:>9}",
            tg / tb,
            (tg * per_g) / (tb * per_b_scan),
            (tg * per_g) / (tb * per_b_tree),
            format!("{} / {}", routes[0].join("+"), routes[1].join("+"))
        );
        ratios.push(tg / tb);
    }
    println!("\n  Every number above is exact for its model to the estimate beside it: the ratio column is a fact about the two");
    println!("  kernels at that size, not an estimate of one. Where the sampled table (`informed_scaling`) and this one disagree");
    println!("  at a shared size, the sampled one is the one that was wrong. Four seeds a row, and one glassy seed can dominate");
    println!("  a row's mean, so the trend in n is not smooth: G/B from {:.2} to {:.2}.", ratios.iter().copied().fold(f64::INFINITY, f64::min), ratios.iter().copied().fold(0.0, f64::max));
}
