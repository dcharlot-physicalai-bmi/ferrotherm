//! **Is nested R-hat a stopping rule a p-bit array can trust?**
//!
//! A p-bit array runs thousands of chains for a handful of sweeps each. Split R-hat was built for a
//! few long chains and cannot certify that regime; nested R-hat (Margossian, Hoffman, Sountsov,
//! Riou-Durand, Vehtari & Gelman, Bayesian Analysis 2024) was built for it. It groups chains into
//! superchains that share a start and watches the variance of the superchain means -- the
//! "nonstationary variance" -- "and so, by proxy, the squared bias". With one draw per chain its
//! threshold is `R_nu <= sqrt(1 + 1/M + tau)` (their Eq. 29), `tau` the tolerated nonstationary
//! variance in units of the stationary one.
//!
//! Everything below is exact (`ferrotherm::autocorr::nested_population`): no chains are run, so a
//! warmup at which the rule passes is the warmup at which it WOULD pass given infinitely many
//! superchains. Three questions, one per block:
//!
//! 1. When the rule first passes, how large is the squared bias it was standing in for?
//! 2. What does it read when every superchain starts from the SAME state -- a hardware reset?
//! 3. On a kernel whose own law is not the Boltzmann law, what does passing certify?
//!
//! ```text
//! cargo run --release --example nested_exact
//! ```
use ferrotherm::autocorr::{nested_population, nested_population_curve, Kernel};
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

/// A 5x2 ferromagnet with no field: the slow mode is the global flip, and a uniform start splits
/// the superchains between the two signs.
fn ferromagnet() -> Graph {
    let (w, h) = (5usize, 2usize);
    let mut b = GraphBuilder::new(w * h);
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            if x + 1 < w {
                b.couple(i, i + 1, 1.0);
            }
            if y + 1 < h {
                b.couple(i, i + w, 1.0);
            }
        }
    }
    b.build()
}

const M: usize = 16;
const TAU: f64 = 0.01;

fn main() {
    let threshold = (1.0 + 1.0 / M as f64 + TAU).sqrt();
    let glass = grid_glass(5, 2, 7);
    let ferro = ferromagnet();
    let states = 1usize << glass.n;
    let uniform = vec![1.0 / states as f64; states];
    let mut reset = vec![0.0f64; states];
    reset[0] = 1.0; // every p-bit at -1: what a cleared register array holds

    type Obs = fn(&Graph, &[i8]) -> f64;
    let observables: [(&str, Obs); 3] = [
        ("energy", |g, s| g.energy(s)),
        ("magnetisation", |_, s| s.iter().map(|&v| f64::from(v)).sum()),
        ("spin 0", |_, s| f64::from(s[0])),
    ];

    println!("M = {M} chains per superchain, one draw each; pass at R_nu <= {threshold:.5} (tau = {TAU})");
    println!("\n=== 1. When the rule first passes, uniform starts, chromatic Gibbs");
    println!("  fixture       beta  observable      pass at w   bias^2/Var there   bias^2/Var <= tau at w");
    for (name, g) in [("5x2 glass", &glass), ("5x2 ferro", &ferro)] {
        for &beta in &[0.5f64, 1.0, 2.0] {
            for (oname, f) in &observables {
                let curve = nested_population_curve(g, beta, Kernel::ChromaticGibbs, |s| f(g, s), &uniform, 400, 1, M)
                    .expect("small");
                let pass = curve
                    .iter()
                    .position(|p| p.rhat <= threshold)
                    .map(|w| (w, curve[w].squared_bias / curve[w].stationary_variance));
                let honest = curve.iter().position(|p| p.squared_bias / p.stationary_variance <= TAU);
                let show = |o: Option<usize>| o.map_or("> 400".to_string(), |w| w.to_string());
                match pass {
                    Some((w, r)) => println!(
                        "  {name:12}  {beta:4}  {oname:14}  {w:9}   {r:16.3e}   {:>10}",
                        show(honest)
                    ),
                    None => println!("  {name:12}  {beta:4}  {oname:14}  {:>9}   {:>16}   {:>10}", "> 400", "-", show(honest)),
                }
            }
        }
    }

    println!("\n=== 2. Every superchain from the same reset state (all -1), chromatic Gibbs, w = 0");
    println!("  fixture       beta  observable      R_nu       passes?   bias^2/Var");
    for (name, g) in [("5x2 glass", &glass), ("5x2 ferro", &ferro)] {
        for &beta in &[1.0f64, 2.0] {
            for (oname, f) in &observables {
                let p = nested_population(g, beta, Kernel::ChromaticGibbs, |s| f(g, s), &reset, 0, 1, M).expect("small");
                println!(
                    "  {name:12}  {beta:4}  {oname:14}  {:.5}   {:7}   {:.3e}",
                    p.rhat,
                    p.rhat <= threshold,
                    p.squared_bias / p.stationary_variance
                );
            }
        }
    }

    println!("\n=== 3. What passing certifies when the kernel's own law is not Boltzmann, 5x2 glass");
    println!("  kernel            beta  observable      pass at w   own-law bias^2/Var   Boltzmann bias^2/Var at w = 400");
    let kernels: [(&str, Kernel); 3] =
        [("shipped fabric", Kernel::FixedFabric), ("synchronous", Kernel::Synchronous), ("SCA, q = 1", Kernel::Sca { q: 1.0 })];
    for (kname, kernel) in kernels {
        for &beta in &[1.0f64, 2.0] {
            for (oname, f) in &observables {
                let curve = nested_population_curve(&glass, beta, kernel, |s| f(&glass, s), &uniform, 400, 1, M)
                    .expect("small");
                let late = &curve[400];
                let late_b = late.boltzmann_bias.powi(2) / late.stationary_variance;
                match curve.iter().position(|p| p.rhat <= threshold) {
                    Some(w) => println!(
                        "  {kname:16}  {beta:4}  {oname:14}  {w:9}   {:18.3e}   {late_b:.3e}",
                        curve[w].squared_bias / curve[w].stationary_variance
                    ),
                    None => println!("  {kname:16}  {beta:4}  {oname:14}  {:>9}   {:>18}   {late_b:.3e}", "> 400", "-"),
                }
            }
        }
    }
}
