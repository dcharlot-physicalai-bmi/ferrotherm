#![allow(missing_docs)]
// EVERY P-BIT AT ITS OWN TEMPERATURE, EXACTLY: what a gain spread does to the sampled law, and
// whether any pair of couplings and fields could have produced it.
//
// A p-bit's sigmoid slope is set by its device -- an MTJ's 'alpha', a comparator's reference, a
// ROM's scaling -- and no two cells have the same one. The sweep then samples site i at its own
// beta_i = beta exp(sigma z_i). It has no common invariant law: a Boltzmann law with couplings
// beta_i J_ij would have to be symmetric in i and j. Two questions with exact answers on 12 spins:
//
//   1. How far is the stationary law from the nominal Boltzmann law, and from the nearest single
//      temperature, as a function of sigma?
//   2. Is it the Boltzmann law of SOME nearby pair (J', h') -- a distortion a calibrated load could
//      cancel -- and how does the residual that no pair removes scale with sigma?
//
// The nearest pair is the exact maximum-likelihood projection (`nearest_boltzmann`): Newton on
// the exact Fisher matrix, strictly convex. Three seeds of the spread per sigma, averaged; the
// per-seed spread is printed as the reproducibility of every number.
//
// run: cargo run --release --example spread_exact

use ferrotherm::autocorr::{boltzmann, kemeny_constant, site_factor, spins, stationary_solved, total_variation, Kernel};
use ferrotherm::continuous::solve;
use ferrotherm::graph::{Graph, GraphBuilder};
use ferrotherm::rng::Pcg;

fn grid_glass(w: usize, h: usize, seed: u64) -> (Vec<(usize, usize)>, Vec<f64>) {
    let mut rng = Pcg::new(seed, 0x6A);
    let mut edges = Vec::new();
    let mut j = Vec::new();
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            if x + 1 < w {
                edges.push((i, i + 1));
                j.push(if rng.f64() < 0.5 { -1.0 } else { 1.0 });
            }
            if y + 1 < h {
                edges.push((i, i + w));
                j.push(if rng.f64() < 0.5 { -1.0 } else { 1.0 });
            }
        }
    }
    let mut theta = j;
    for _ in 0..w * h {
        theta.push((rng.f64() - 0.5) * 0.4);
    }
    (edges, theta)
}

fn build(n: usize, edges: &[(usize, usize)], theta: &[f64]) -> Graph {
    let mut b = GraphBuilder::new(n);
    for (k, &(i, j)) in edges.iter().enumerate() {
        b.couple(i, j, theta[k]);
    }
    for i in 0..n {
        b.bias(i, theta[edges.len() + i]);
    }
    b.build()
}

fn kl(target: &[f64], law: &[f64]) -> f64 {
    target.iter().zip(law).filter(|(t, _)| **t > 0.0).map(|(t, l)| t * (t / l.max(1e-300)).ln()).sum()
}

fn stats(x: usize, n: usize, edges: &[(usize, usize)]) -> Vec<f64> {
    let s = spins(x, n);
    let mut phi: Vec<f64> = edges.iter().map(|&(i, j)| f64::from(s[i] * s[j])).collect();
    phi.extend((0..n).map(|i| f64::from(s[i])));
    phi
}

fn moments_and_fisher(law: &[f64], n: usize, edges: &[(usize, usize)]) -> (Vec<f64>, Vec<f64>) {
    let d = edges.len() + n;
    let mut m = vec![0.0f64; d];
    let mut second = vec![0.0f64; d * d];
    for (x, &p) in law.iter().enumerate() {
        if p == 0.0 {
            continue;
        }
        let phi = stats(x, n, edges);
        for a in 0..d {
            m[a] += p * phi[a];
            for b in 0..d {
                second[a * d + b] += p * phi[a] * phi[b];
            }
        }
    }
    let mut f = second;
    for a in 0..d {
        for b in 0..d {
            f[a * d + b] -= m[a] * m[b];
        }
    }
    (m, f)
}

/// Newton's method on KL(target || Boltzmann(theta)) from `start`; stops when no step lowers the
/// KL at floating-point resolution. Returns theta* and the KL there.
fn nearest_pair(target: &[f64], n: usize, edges: &[(usize, usize)], start: &[f64]) -> (Vec<f64>, f64) {
    let d = edges.len() + n;
    let (m_target, _) = moments_and_fisher(target, n, edges);
    let mut theta = start.to_vec();
    let mut law = boltzmann(&build(n, edges, &theta), 1.0).expect("small");
    let mut k = kl(target, &law);
    for _ in 0..200 {
        let (m, mut fisher) = moments_and_fisher(&law, n, edges);
        let g: Vec<f64> = m_target.iter().zip(&m).map(|(a, b)| a - b).collect();
        if g.iter().fold(0.0f64, |acc, v| acc.max(v.abs())) < 1e-11 {
            break;
        }
        for a in 0..d {
            fisher[a * d + a] += 1e-12;
        }
        let dir = solve(&fisher, d, &g).expect("the Fisher matrix is positive definite");
        let slope: f64 = g.iter().zip(&dir).map(|(a, b)| a * b).sum();
        let mut t = 1.0;
        let mut stalled = false;
        loop {
            let trial: Vec<f64> = theta.iter().zip(&dir).map(|(th, dd)| th + t * dd).collect();
            let trial_law = boltzmann(&build(n, edges, &trial), 1.0).expect("small");
            let trial_k = kl(target, &trial_law);
            if trial_k <= k - 1e-4 * t * slope {
                theta = trial;
                law = trial_law;
                k = trial_k;
                break;
            }
            if t < 1e-8 {
                stalled = true;
                break;
            }
            t *= 0.5;
        }
        if stalled {
            break;
        }
    }
    (theta, k)
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

fn mean_sd(v: &[f64]) -> (f64, f64) {
    let m = v.iter().sum::<f64>() / v.len() as f64;
    let sd = (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64).sqrt();
    (m, sd)
}

fn main() {
    let (w, h) = (4usize, 3usize);
    let n = w * h;
    let (edges, theta0) = grid_glass(w, h, 7);
    let e = edges.len();
    let g = build(n, &edges, &theta0);
    let seeds = [1u64, 2, 3];
    println!("A PER-SITE TEMPERATURE SPREAD, EXACTLY, ON ALL 2^{n} STATES\n");
    println!("  model    {w}x{h} open grid, random +-1 couplings, fields in [-0.2, 0.2]; beta_i = beta exp(sigma z_i)");
    println!("  oracle   direct solve for the law; nearest single beta by TV; nearest pair by exact maximum likelihood");
    println!("  seeds    {} draws of the spread per sigma, mean (sd) over seeds\n", seeds.len());
    for &beta in &[1.0f64, 2.0] {
        let b = boltzmann(&g, beta).expect("small");
        let k_chrom = kemeny_constant(&g, beta, Kernel::ChromaticGibbs).expect("small");
        println!("  beta {beta}: K chromatic = {k_chrom:.3e}\n");
        println!(
            "  {:>6}   {:>10} {:>8} {:>10}   {:>10} {:>10} {:>9} {:>8}   {:>8}",
            "sigma", "TV(law,B)", "beta_eff", "TV at eff", "KL loaded", "KL pair", "resid/s2", "max|dJ|", "K/K_chr"
        );
        for &sigma in &[0.02f64, 0.05, 0.1, 0.2, 0.4] {
            let (mut tvs, mut beffs, mut tveffs, mut kls, mut klps, mut djs, mut kks) =
                (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
            for &seed in &seeds {
                let k = Kernel::SiteSpread { seed, spread: sigma };
                let law = stationary_solved(&g, beta, k).expect("small");
                tvs.push(total_variation(&law, &b));
                let (b_eff, tv_eff) = nearest_beta(&g, &law, 0.25 * beta, 4.0 * beta);
                beffs.push(b_eff);
                tveffs.push(tv_eff);
                let loaded: Vec<f64> = theta0.iter().map(|t| beta * t).collect();
                kls.push(kl(&law, &b));
                let (star, kl_pair) = nearest_pair(&law, n, &edges, &loaded);
                klps.push(kl_pair);
                djs.push(star[..e].iter().zip(&loaded[..e]).map(|(a, c)| (a - c).abs()).fold(0.0f64, f64::max) / beta);
                kks.push(kemeny_constant(&g, beta, k).expect("small") / k_chrom);
            }
            let (tv, tv_sd) = mean_sd(&tvs);
            let (be, _) = mean_sd(&beffs);
            let (te, _) = mean_sd(&tveffs);
            let (kl_l, _) = mean_sd(&kls);
            let (kl_p, kl_p_sd) = mean_sd(&klps);
            let (dj, _) = mean_sd(&djs);
            let (kk, _) = mean_sd(&kks);
            println!(
                "  {sigma:>6.2}   {tv:>10.3e} {be:>8.4} {te:>10.3e}   {kl_l:>10.3e} {kl_p:>10.3e} {:>9.3e} {dj:>8.4}   {kk:>8.3}   (sd: TV {:.0}%, KL pair {:.0}%)",
                kl_p / (sigma * sigma),
                100.0 * tv_sd / tv,
                100.0 * kl_p_sd / kl_p
            );
        }
        // The guess a calibration would make: J'_ij = J_ij (beta_i + beta_j) / 2, h'_i = beta_i h_i.
        // How much of the loaded loss does that pair remove, without any fit?
        let sigma = 0.2;
        let mut removed = Vec::new();
        for &seed in &seeds {
            let k = Kernel::SiteSpread { seed, spread: sigma };
            let law = stationary_solved(&g, beta, k).expect("small");
            let f: Vec<f64> = (0..n).map(|i| site_factor(seed, sigma, i)).collect();
            let mut guess: Vec<f64> = edges.iter().enumerate().map(|(k, &(i, j))| beta * theta0[k] * 0.5 * (f[i] + f[j])).collect();
            guess.extend((0..n).map(|i| beta * theta0[e + i] * f[i]));
            let kl_guess = kl(&law, &boltzmann(&build(n, &edges, &guess), 1.0).expect("small"));
            removed.push(1.0 - kl_guess / kl(&law, &b));
        }
        let (r, r_sd) = mean_sd(&removed);
        println!("\n  the mean-temperature pair J'_ij = J_ij (beta_i + beta_j)/2, h'_i = beta_i h_i at sigma {sigma} removes {:.1}% (sd {:.1}%) of the loaded KL without a fit\n", 100.0 * r, 100.0 * r_sd);
    }
    println!("  WHAT THE TABLE SAYS.\n");
    println!("  TV(law,B) is the equilibrium error the spread leaves at the nominal temperature; beta_eff and TV at");
    println!("  eff say how much of it is a global temperature error. KL pair is the part no pair of couplings and");
    println!("  fields reproduces -- the non-Boltzmann residual -- and resid/s2 is that residual over sigma^2: a");
    println!("  constant column means the residual is quadratic in the spread. max|dJ| is how far the nearest pair");
    println!("  sits from the loaded couplings, per unit beta. K/K_chr is relaxation per sweep against the sweep");
    println!("  with no spread.");
}
