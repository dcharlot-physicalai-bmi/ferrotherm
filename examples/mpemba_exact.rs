//! **What does the "negligible" digital preprocessing cost, in the only unit it shares with the
//! answer?**
//!
//! Moroder, Binder & Goold (arXiv:2603.24183) initialise a thermodynamic matrix-inversion device
//! from the `K` smallest eigenpairs, found digitally by Lanczos, and write that *"the classical
//! preprocessing cost of the Lanczos algorithm is negligible compared to the thermalization time
//! across all matrix dimensions considered here"*. Flops and device seconds share no unit. Flops and
//! flops do: this example counts the Lanczos work (products, full reorthogonalisation, Ritz vectors)
//! and the work of computing the ENTIRE inverse digitally by Cholesky, both counted as performed, on
//! both of the paper's ensembles at `K = 10`.
//!
//! A ratio near or above one means the digital processor has spent what the whole answer costs
//! before the device starts. It also prints what the device buys in return: the thermalization
//! speedup at two tolerances, and the fraction of `A⁻¹` the initialisation already holds.
//!
//! ```text
//! cargo run --release --example mpemba_exact
//! ```
use ferrotherm::mpemba::{cholesky_inverse, covariance_error, ensemble, lanczos_smallest, thermalization_time};
use ferrotherm::rng::Pcg;

fn wishart(d: usize, m: usize, seed: u64) -> Vec<f64> {
    let mut rng = Pcg::new(seed, 0x3157);
    let mut gauss = || {
        let a = rng.f64().max(1e-300);
        let b = rng.f64();
        (-2.0 * a.ln()).sqrt() * (std::f64::consts::TAU * b).cos()
    };
    let x: Vec<f64> = (0..m * d).map(|_| gauss()).collect();
    let mut a = vec![0.0; d * d];
    for i in 0..d {
        for j in i..d {
            let s: f64 = (0..m).map(|r| x[r * d + i] * x[r * d + j]).sum::<f64>() / m as f64;
            a[i * d + j] = s;
            a[j * d + i] = s;
        }
    }
    a
}

fn main() {
    let k = 10;
    println!("K = {k}; Lanczos to a Ritz residual of tol * |A|, full reorthogonalisation; flops counted as performed.");
    println!("\n=== linearly spaced spectrum (lambda_min 0.5, delta 0.5) under a Haar basis: the paper's Eqs. 27-28");
    println!("     d     tol   Lanczos steps   Lanczos flops   Cholesky-inverse flops   ratio   init holds   speedup@1e-2  @1e-4  limit");
    for &d in &[100usize, 200, 500, 1000, 2000] {
        let (a, _, lam) = ensemble(d, 0.5, 0.5, 3);
        let (_, chol) = cholesky_inverse(&a, d).expect("SPD");
        let whole = covariance_error(&lam, 0, 0.0);
        let held = 1.0 - (covariance_error(&lam, k, 0.0) / whole).powi(2);
        let s2 = thermalization_time(&lam, 0, 1e-2) / thermalization_time(&lam, k, 1e-2);
        let s4 = thermalization_time(&lam, 0, 1e-4) / thermalization_time(&lam, k, 1e-4);
        for &tol in &[1e-4f64, 1e-8] {
            let r = lanczos_smallest(&a, d, k, tol, d, 1);
            println!(
                "  {d:5}  {tol:6.0e}   {:13}   {:13.3e}   {chol:22.3e}   {:5.2}   {:9.1}%   {s2:12.2}  {s4:5.2}  {:5.1}{}",
                r.steps,
                r.flops as f64,
                r.flops as f64 / chol as f64,
                100.0 * held,
                lam[k] / lam[0],
                if r.converged { "" } else { "  (NOT converged)" }
            );
        }
    }
    println!("\n=== positive Wishart, A = X^T X / m, m = 1.1 d: the paper's Eq. 29 (eigenvalues crowd at the lower edge)");
    println!("     d     tol   Lanczos steps   Lanczos flops   Cholesky-inverse flops   ratio");
    for &d in &[100usize, 200, 500, 1000] {
        let a = wishart(d, (1.1 * d as f64).round() as usize, 5);
        let (_, chol) = cholesky_inverse(&a, d).expect("SPD");
        for &tol in &[1e-4f64, 1e-8] {
            let r = lanczos_smallest(&a, d, k, tol, d, 1);
            println!(
                "  {d:5}  {tol:6.0e}   {:13}   {:13.3e}   {chol:22.3e}   {:5.2}{}",
                r.steps,
                r.flops as f64,
                r.flops as f64 / chol as f64,
                if r.converged { "" } else { "  (NOT converged within d steps)" }
            );
        }
    }
}
