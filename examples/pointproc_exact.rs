//! **Does a spike that lives a fixed time sample faster than one that lives a random time?**
//!
//! Stewart & Sahani (arXiv:2603.09089) report that their point-process sampler *"always outperforms
//! these birth-death processes"* in multivariate effective sample size on 63 targets. The two differ in
//! one line: a spike's lifetime is fixed (`m`) or exponential with mean `m`. Same birth rates, same
//! limiting law. This measures, on Sherrington–Kirkpatrick instances, the integrated autocorrelation
//! time of the energy and the magnetisation IN UNITS OF THE LIFETIME `m`, from long runs, and the
//! births each spends per unit time -- the random draws a spiking substrate pays for.
//!
//! ```text
//! cargo run --release --example pointproc_exact
//! ```
use ferrotherm::certify::tau_int;
use ferrotherm::graph::{Graph, GraphBuilder};
use ferrotherm::pointproc::{Lifetime, PointProcess};
use ferrotherm::rng::Pcg;

fn sk(n: usize, seed: u64) -> Graph {
    let mut rng = Pcg::new(seed, 0x6A);
    let mut normal = || {
        let a = rng.f64().max(1e-300);
        let b = rng.f64();
        (-2.0 * a.ln()).sqrt() * (std::f64::consts::TAU * b).cos()
    };
    let mut b = GraphBuilder::new(n);
    for i in 0..n {
        for j in i + 1..n {
            b.couple(i, j, normal() / (n as f64).sqrt());
        }
    }
    for i in 0..n {
        b.bias(i, 0.1 * normal());
    }
    b.build()
}

fn main() {
    let (spacing, count, burn) = (0.1f64, 400_000usize, 200.0f64);
    println!("tau_int in units of the lifetime m, sampled every {spacing} m over {} m; births per unit m", spacing * count as f64);
    println!("   n  beta   lifetime      tau_E    tau_M   births/m   tau_E ratio (exp / fixed)");
    for &n in &[8usize, 16] {
        for &beta in &[0.5f64, 1.0, 1.5] {
            let g = sk(n, 11);
            let mut row = Vec::new();
            for lifetime in [Lifetime::Fixed, Lifetime::Exponential] {
                let mut pp = PointProcess::new(&g, beta, 1.0, lifetime, 5);
                let e = pp.trace(burn, spacing, count, |s| g.energy(s));
                let births_before = pp.births;
                let t_before = pp.t;
                let mut pp2 = PointProcess::new(&g, beta, 1.0, lifetime, 6);
                let m = pp2.trace(burn, spacing, count, |s| s.iter().map(|&v| f64::from(v)).sum());
                let tau_e = tau_int(&e) * spacing;
                let tau_m = tau_int(&m) * spacing;
                let rate = births_before as f64 / t_before;
                row.push(tau_e);
                println!("  {n:2}  {beta:4}   {:11}  {tau_e:8.3} {tau_m:8.3}   {rate:8.3}", format!("{lifetime:?}"));
            }
            println!("                                                        {:.2}", row[1] / row[0]);
        }
    }
}
