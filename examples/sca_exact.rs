//! **What does updating every spin at once buy, at a stated accuracy?**
//!
//! Stochastic cellular automata (SCA) is the all-spins-at-once rule of the STATICA and Amorphica
//! annealing processors: every site resampled on the same tick from the previous state, with its
//! field halved and a pull `q` toward its own current value (`Kernel::Sca`). Its law tends to the
//! Boltzmann law as `q -> inf` (Fukushima-Kimura et al., J. Stat. Phys. 190:79, 2023), and the
//! parallelism is the selling point: Amorphica's press release reports up to 58x the speed and
//! about 30,000x the power efficiency of a GPU.
//!
//! A p-bit fabric already updates in parallel without approximating anything: colour the graph,
//! and every site in a colour class is conditionally independent of the others, so one class per
//! tick is exact Gibbs. The question is which costs fewer ticks to reach the SAME accuracy:
//!
//! * the coloured sweep: exact law, `chi` ticks per sweep (`chi` colour classes);
//! * SCA: one tick per update, law within TV `eps` of Boltzmann only for `q >= q*(eps)`.
//!
//! Both are computed exactly here. The cost is the integrated autocorrelation time of the energy
//! and of the magnetisation, by the fundamental matrix, in TICKS: a coloured sweep is `chi` ticks,
//! an SCA step one. That is the variance cost of an estimate -- `2 tau` ticks per independent
//! draw. The accuracy is the total variation from the Boltzmann law at the same `beta`, from
//! `ferrotherm::autocorr::sca_law`.
//!
//! Kemeny's constant is deliberately NOT the cost here, though `examples/fabric_exact.rs` uses it.
//! It sums `1 / (1 - lambda)` over all `2^n - 1` modes, and every mode contributes at least a
//! half a STEP however fast it is, so on 1,024 states it sits near 1,024 steps for any fast chain.
//! Between two kernels with the same step that floor cancels; between a sweep and a tick it does
//! not, and multiplying a sweep's Kemeny by `chi` multiplies the floor. The first version of this
//! example did exactly that and reported SCA at `q = 0` as `1/chi` of the coloured sweep on every
//! fixture at every temperature -- the floor's ratio, not the kernels'.
//!
//! The pinning's first-order law is also checked here, not assumed: dividing the SCA weight by
//! the Boltzmann weight leaves `prod_i (1 + e^{-2q} phi_i(x))`, `phi_i = exp(-beta f_i x_i)`, so
//! `TV * e^{2q}` must tend to `c = E_G|Phi - <Phi>| / 2` with `Phi = sum_i phi_i`. The column
//! `TV e^2q / c` reads 1 once `q` is large, which ties the q* this example reports to a formula a
//! reader can evaluate on any graph without enumerating it.
//!
//! ```text
//! cargo run --release --example sca_exact
//! ```
use ferrotherm::autocorr::{boltzmann, sca_law, spins, tau_int_fundamental, total_variation, Kernel};
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

/// A 10-ring with chords (0,4) and (2,7): odd cycles, so not bipartite.
fn chorded_ring(seed: u64) -> Graph {
    let n = 10;
    let mut rng = Pcg::new(seed, 0x6A);
    let mut b = GraphBuilder::new(n);
    for i in 0..n {
        b.couple(i, (i + 1) % n, if rng.f64() < 0.5 { -1.0 } else { 1.0 });
    }
    for &(i, j) in &[(0usize, 4usize), (2, 7)] {
        b.couple(i, j, if rng.f64() < 0.5 { -1.0 } else { 1.0 });
    }
    for i in 0..n {
        b.bias(i, (rng.f64() - 0.5) * 0.4);
    }
    b.build()
}

/// Sherrington-Kirkpatrick on `n` spins: every pair coupled, `J ~ N(0, 1/n)`. The full
/// connectivity STATICA's title names ("complete spin-spin interactions") and the case SCA exists
/// for: no two sites share a colour, so the coloured sweep is `n` ticks.
fn sk(n: usize, seed: u64) -> Graph {
    let mut rng = Pcg::new(seed, 0x6A);
    let mut normal = || {
        let u1 = rng.f64().max(1e-300);
        let u2 = rng.f64();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    };
    let mut b = GraphBuilder::new(n);
    let scale = 1.0 / (n as f64).sqrt();
    for i in 0..n {
        for j in i + 1..n {
            b.couple(i, j, normal() * scale);
        }
    }
    for i in 0..n {
        b.bias(i, normal() * 0.1);
    }
    b.build()
}

/// `c = E_G|Phi - <Phi>_G| / 2`, `Phi(x) = sum_i exp(-beta f_i(x) x_i)`: the first-order constant
/// of `TV(SCA, Boltzmann) ~ c e^{-2q}`, computed from the Boltzmann law alone.
fn first_order_constant(g: &Graph, beta: f64) -> f64 {
    let bolt = boltzmann(g, beta).expect("small");
    let phi: Vec<f64> = (0..bolt.len())
        .map(|x| {
            let s = spins(x, g.n);
            (0..g.n).map(|i| (-beta * g.field(i, &s) * f64::from(s[i])).exp()).sum()
        })
        .collect();
    let mean: f64 = bolt.iter().zip(&phi).map(|(p, f)| p * f).sum();
    0.5 * bolt.iter().zip(&phi).map(|(p, f)| p * (f - mean).abs()).sum::<f64>()
}

fn tv_at(g: &Graph, beta: f64, q: f64, bolt: &[f64]) -> f64 {
    total_variation(&sca_law(g, beta, q).expect("small"), bolt)
}

/// The smallest `q` on a 0.01 grid, refined by bisection, at which TV first falls to `eps` and stays
/// there over the rest of the scan. `None` if it never does below `q = 20`.
fn q_star(g: &Graph, beta: f64, eps: f64, bolt: &[f64]) -> Option<f64> {
    let grid: Vec<f64> = (0..=2000).map(|k| f64::from(k) * 0.01).collect();
    let tvs: Vec<f64> = grid.iter().map(|&q| tv_at(g, beta, q, bolt)).collect();
    // The last grid point still above eps; everything after it is at or below.
    let last_above = tvs.iter().rposition(|&t| t > eps)?;
    if last_above + 1 >= grid.len() {
        return None;
    }
    let (mut lo, mut hi) = (grid[last_above], grid[last_above + 1]);
    for _ in 0..50 {
        let mid = 0.5 * (lo + hi);
        if tv_at(g, beta, mid, bolt) > eps {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Some(hi)
}

/// Kamakura's two windows: Theorem 1's largest `q` with more flips per step than Glauber, and
/// Theorem 3's smallest `q` with the law close in order-preservation at `eps`. `K` is the largest
/// local field over every state, `v` the sum of squared couplings and fields.
fn kamakura_windows(g: &Graph, beta: f64, eps: f64) -> (f64, f64) {
    let mut kbar = 0.0f64;
    for x in 0..(1usize << g.n) {
        let s = spins(x, g.n);
        for i in 0..g.n {
            kbar = kbar.max(g.field(i, &s).abs());
        }
    }
    let v: f64 = 0.5 * g.w.iter().map(|w| w * w).sum::<f64>() + g.h.iter().map(|h| h * h).sum::<f64>();
    let ln_n = (g.n as f64).ln();
    let more_flips = 0.5 * (ln_n - beta * kbar);
    let close = 0.5 * (ln_n + beta * kbar - (eps * v.sqrt() / (2.0 * kbar)).ln());
    (more_flips, close)
}

/// `tau_int` of the energy and of the magnetisation, in steps of the kernel.
fn taus(g: &Graph, beta: f64, kernel: Kernel) -> (f64, f64) {
    let e = tau_int_fundamental(g, beta, kernel, |s| g.energy(s)).expect("dense").tau_int;
    let m = tau_int_fundamental(g, beta, kernel, |s| s.iter().map(|&v| f64::from(v)).sum()).expect("dense").tau_int;
    (e, m)
}

fn main() {
    let fixtures: [(&str, Graph); 3] =
        [("5x2 +-J grid", grid_glass(5, 2, 7)), ("10-ring + chords", chorded_ring(3)), ("SK, n = 10", sk(10, 11))];
    let betas = [0.5f64, 1.0, 2.0];
    let qs = [0.0f64, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0];
    let targets = [1e-1f64, 1e-2, 1e-3];

    for (name, g) in &fixtures {
        let chi = g.classes.len() as f64;
        println!("\n=== {name}: n = {}, {chi} colour classes (the coloured sweep is {chi} ticks)", g.n);
        for &beta in &betas {
            let bolt = boltzmann(g, beta).expect("small");
            let c = first_order_constant(g, beta);
            let (ce, cm) = taus(g, beta, Kernel::ChromaticGibbs);
            let (ce, cm) = (ce * chi, cm * chi);
            println!(
                "\n  beta = {beta}: coloured sweep tau_E {ce:.2} ticks, tau_M {cm:.2} ticks, exact law; first-order c = {c:.4}"
            );
            println!("     q    TV(SCA, Boltzmann)   TV e^2q / c   tau_E (ticks)  tau_M (ticks)   E / coloured   M / coloured");
            for &q in &qs {
                let tv = tv_at(g, beta, q, &bolt);
                let (e, m) = taus(g, beta, Kernel::Sca { q });
                println!(
                    "   {q:4.1}    {tv:18.3e}   {:11.4}   {e:13.2}  {m:13.2}   {:12.3}   {:12.3}",
                    tv * (2.0 * q).exp() / c,
                    e / ce,
                    m / cm
                );
            }
            for &eps in &targets {
                match q_star(g, beta, eps, &bolt) {
                    Some(qs) => {
                        let (e, m) = taus(g, beta, Kernel::Sca { q: qs });
                        let predicted = 0.5 * (c / eps).ln();
                        let (flips, close) = kamakura_windows(g, beta, eps);
                        println!(
                            "   TV <= {eps:.0e}: q* = {qs:.3} (first-order ln(c/eps)/2 = {predicted:.3}); tau_E {e:.1} ticks = {:.2}x, tau_M {m:.1} = {:.2}x the coloured sweep; Kamakura windows: flips q <= {flips:.2}, close q >= {close:.2}",
                            e / ce,
                            m / cm
                        );
                    }
                    None => println!("   TV <= {eps:.0e}: not reached below q = 20"),
                }
            }
        }
    }

    // THE DENSE-GRAPH TREND. On SK the coloured sweep costs n ticks, which suggests SCA's relative
    // cost should fall as n grows. Measured, n = 6 to 12, at beta = 1, one instance per n: the ratio
    // at two accuracies, and the first-order break-even accuracy eps_BE = A c / tau_coloured with
    // A = tau_SCA(q) e^{-2q} read at q = 4, where the first-order regime has set in. On 2026-09-24
    // it read 0.058, 0.060, 0.062, 0.049 -- flat within one instance's spread, so the suggestion is
    // not supported at these sizes.
    println!("\n=== SK trend at beta = 1: SCA tau_E / coloured tau_E, both in ticks");
    println!("   n   colours   coloured tau_E   at TV 1e-1   at TV 1e-2   first-order break-even eps");
    for n in [6usize, 8, 10, 12] {
        let g = sk(n, 11);
        let beta = 1.0;
        let chi = g.classes.len() as f64;
        let bolt = boltzmann(&g, beta).expect("small");
        let tau = |k: Kernel| tau_int_fundamental(&g, beta, k, |s| g.energy(s)).expect("dense").tau_int;
        let col = tau(Kernel::ChromaticGibbs) * chi;
        let at = |eps: f64| tau(Kernel::Sca { q: q_star(&g, beta, eps, &bolt).expect("reached") }) / col;
        let (r1, r2) = (at(1e-1), at(1e-2));
        let a = tau(Kernel::Sca { q: 4.0 }) * (-8.0f64).exp();
        let c = first_order_constant(&g, beta);
        println!("  {n:2}   {chi:7}   {col:14.2}   {r1:10.3}   {r2:10.3}   {:.4}", a * c / col);
    }
}
