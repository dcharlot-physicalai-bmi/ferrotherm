//! **How far is one attention step from the energy it descends?**
//!
//! Reproduces the table in `WORKLOADS.md` section 7. `DenseMemory::attention_update` is exactly one
//! gradient step on `DenseMemory::lse_energy` (step size 1), which is held to central differences by
//! `attention_is_exactly_one_gradient_step_on_this_energy`. The step from there that the field takes
//! casually -- *a machine that relaxes to this energy computes attention* -- is what this measures,
//! by iterating the map to its fixed point and printing the distance one step fell short.
//!
//! The answer is that it holds in one corner only: a query already sitting on a stored pattern, at
//! high beta. On a diffuse query -- which is what a real attention head has -- one step is 41-90%
//! away at every beta here, and never gets small.
//!
//! ```text
//! cargo run --release --example attention_fixed_point
//! ```
//!
//! The `energy-increases` column is the concave-convex guarantee: the iteration is a descent
//! method, so it must read 0. `iters<=` is how many steps the furthest draw actually needed.
use ferrotherm::dense_memory::{DenseMemory, Energy};
use ferrotherm::hopfield::random_patterns;
use ferrotherm::rng::Pcg;

fn nrm(a: &[f64]) -> f64 { a.iter().map(|v| v * v).sum::<f64>().sqrt() }
fn dist(a: &[f64], b: &[f64]) -> f64 { a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f64>().sqrt() }

fn main() {
    let (k, d) = (8usize, 16usize);
    for &beta in &[0.25f64, 0.5, 1.0, 2.0, 4.0] {
        let mem = DenseMemory::new(random_patterns(d, k, 77), Energy::Exponential { b: 2.0 });
        let mut rng = Pcg::new(404, 0);
        let (mut mixed, mut retr) = (Vec::new(), Vec::new());
        let mut mono_fail = 0usize;
        let mut worst_iters = 0usize;
        for draw in 0..60 {
            let retrieval = draw % 3 == 2;
            let xi: Vec<f64> = if retrieval {
                mem.patterns[draw % k].iter().map(|&a| 0.5 * f64::from(a)).collect()
            } else {
                (0..d).map(|_| 0.2 * (2.0 * rng.f64() - 1.0)).collect()
            };
            let one = mem.attention_update(&xi, beta);
            // iterate to the fixed point
            let mut cur = one.clone();
            let mut e_prev = mem.lse_energy(&xi, beta);
            let mut iters = 1;
            loop {
                let e = mem.lse_energy(&cur, beta);
                if e > e_prev + 1e-12 { mono_fail += 1; }
                e_prev = e;
                let next = mem.attention_update(&cur, beta);
                if dist(&next, &cur) < 1e-12 || iters > 5000 { cur = next; break; }
                cur = next; iters += 1;
            }
            worst_iters = worst_iters.max(iters);
            let rel = dist(&one, &cur) / nrm(&cur).max(1e-30);
            if retrieval { retr.push(rel) } else { mixed.push(rel) }
        }
        let mean = |v: &Vec<f64>| v.iter().sum::<f64>() / v.len() as f64;
        let max = |v: &Vec<f64>| v.iter().copied().fold(0.0f64, f64::max);
        println!("beta {beta:>5}  mixed rel-gap mean {:>9.4} max {:>9.4} | retrieval mean {:>9.6} max {:>9.6} | iters<= {worst_iters:>4} | energy-increases {mono_fail}",
            mean(&mixed), max(&mixed), mean(&retr), max(&retr));
    }
}
