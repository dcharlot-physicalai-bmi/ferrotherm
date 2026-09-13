#![allow(missing_docs)]
// IS THE FABRIC'S LAW THE BOLTZMANN DISTRIBUTION OF SOME NEARBY COUPLINGS? Exactly.
//
// `fabric_exact` shows the fabric's stationary law is not the Boltzmann distribution it was
// programmed for, and that a single effective temperature removes most but not all of the
// distance (TV at beta_eff is a third to a tenth of TV at the nominal beta). That is a
// one-parameter projection. The open question is the full one: is the fabric's law the Boltzmann
// distribution of SOME pair (J', h') near the loaded one -- a coupling distortion, which a
// calibrated load could pre-compensate -- or is it outside the Boltzmann family altogether, which
// no load can fix? On a 12-spin grid both halves are exact:
//
//   the fabric's law      `autocorr::stationary_solved` on all 2^12 states, one direct solve
//   the nearest pair      the maximum-likelihood Boltzmann fit to that law, which is the moment
//                         match `<s_i s_j>_theta = <s_i s_j>_fabric` on every edge and site; the
//                         objective is strictly convex in theta, so Newton's method with the exact
//                         Fisher matrix (the covariance of the sufficient statistics under the
//                         fit, 29 x 29) converges to THE nearest pair, not a nearby one.
//
// Everything is KL(fabric || Boltzmann): the projection minimises it, so the three rows are
// nested -- loaded >= beta_eff >= nearest pair -- and the last is the part of the fabric's law
// that is NOT a Boltzmann distribution of anything.
//
// run: cargo run --release --example nearest_boltzmann

use ferrotherm::autocorr::{boltzmann, spins, stationary_solved, total_variation, Kernel};
use ferrotherm::continuous::solve;
use ferrotherm::graph::{Graph, GraphBuilder};
use ferrotherm::rng::Pcg;

/// A w x h open grid with random +-1 couplings and fields in [-0.2, 0.2], as an edge list and a
/// parameter vector (couplings first, then fields), so the same model can be rebuilt at any theta.
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
    target
        .iter()
        .zip(law)
        .filter(|(t, _)| **t > 0.0)
        .map(|(t, l)| t * (t / l.max(1e-300)).ln())
        .sum()
}

/// The sufficient statistics of state `x`: `s_i s_j` on each edge, then `s_i` on each site.
fn stats(x: usize, n: usize, edges: &[(usize, usize)]) -> Vec<f64> {
    let s = spins(x, n);
    let mut phi: Vec<f64> = edges.iter().map(|&(i, j)| f64::from(s[i] * s[j])).collect();
    phi.extend((0..n).map(|i| f64::from(s[i])));
    phi
}

/// Moments and their covariance under `law`: the gradient's data term and the Fisher matrix.
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

/// Newton's method on the strictly convex objective KL(target || Boltzmann(theta)), from `start`.
/// Returns theta*, the KL there, the sup-norm of the last gradient, and the iterations taken.
fn nearest_pair(target: &[f64], n: usize, edges: &[(usize, usize)], start: &[f64]) -> (Vec<f64>, f64, f64, usize) {
    let d = edges.len() + n;
    let (m_target, _) = moments_and_fisher(target, n, edges);
    let mut theta = start.to_vec();
    let mut law = boltzmann(&build(n, edges, &theta), 1.0).expect("small");
    let mut k = kl(target, &law);
    let mut grad_inf = f64::NAN;
    let mut iters = 0;
    for it in 0..200 {
        iters = it;
        let (m, mut fisher) = moments_and_fisher(&law, n, edges);
        let g: Vec<f64> = m_target.iter().zip(&m).map(|(a, b)| a - b).collect();
        grad_inf = g.iter().fold(0.0f64, |acc, v| acc.max(v.abs()));
        if grad_inf < 1e-11 {
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
                // No step of any length lowers the KL at floating-point resolution: the optimum
                // to the precision the objective can be evaluated at, whatever the gap says.
                stalled = true;
                break;
            }
            t *= 0.5;
        }
        if stalled {
            break;
        }
    }
    (theta, k, grad_inf, iters)
}

/// The single temperature whose Boltzmann law is nearest in KL, by golden section on [lo, hi].
fn nearest_beta(target: &[f64], n: usize, edges: &[(usize, usize)], theta0: &[f64], lo: f64, hi: f64) -> (f64, f64) {
    let f = |b: f64| kl(target, &boltzmann(&build(n, edges, theta0), b).expect("small"));
    let phi = 0.5 * (3.0 - 5.0f64.sqrt());
    let (mut a, mut b) = (lo, hi);
    let mut c = a + phi * (b - a);
    let mut d = b - phi * (b - a);
    let (mut fc, mut fd) = (f(c), f(d));
    for _ in 0..80 {
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

fn main() {
    let (w, h) = (4usize, 3usize);
    let n = w * h;
    let (edges, theta0) = grid_glass(w, h, 7);
    let e = edges.len();
    println!("THE NEAREST BOLTZMANN PAIR TO THE FABRIC'S LAW, EXACTLY, ON ALL 2^{n} STATES\n");
    println!("  model    {w}x{h} open grid, random +-1 couplings, fields in [-0.2, 0.2]: {e} couplings + {n} fields = {} parameters", e + n);
    println!("  fabric   Q.8 weights, 1024-entry ROM, 16-bit comparator (autocorr::Kernel::FixedFabric), law by direct solve");
    println!("  fit      maximum likelihood to the fabric's law = exact moment matching, Newton with the exact Fisher matrix\n");
    println!(
        "  {:>5}   {:>10} {:>10}   {:>8} {:>10} {:>10}   {:>10} {:>10} {:>7} {:>7} {:>8}   {:>9}",
        "beta", "KL loaded", "TV loaded", "beta_eff", "KL b_eff", "TV b_eff", "KL pair", "TV pair", "max|dJ|", "max|dh|", "not-B %", "newton"
    );
    for &beta in &[0.5f64, 1.0, 1.5, 2.0, 3.0] {
        let g = build(n, &edges, &theta0);
        let fab = stationary_solved(&g, beta, Kernel::FixedFabric).expect("small");
        let loaded: Vec<f64> = theta0.iter().map(|t| beta * t).collect();
        let b_loaded = boltzmann(&g, beta).expect("small");
        let (kl_loaded, tv_loaded) = (kl(&fab, &b_loaded), total_variation(&fab, &b_loaded));
        let (b_eff, kl_eff) = nearest_beta(&fab, n, &edges, &theta0, 0.25 * beta, 2.0 * beta);
        let tv_eff = total_variation(&fab, &boltzmann(&g, b_eff).expect("small"));
        let (star, kl_pair, grad, iters) = nearest_pair(&fab, n, &edges, &loaded);
        let tv_pair = total_variation(&fab, &boltzmann(&build(n, &edges, &star), 1.0).expect("small"));
        let dj = star[..e].iter().zip(&loaded[..e]).map(|(a, b)| (a - b).abs()).fold(0.0f64, f64::max);
        let dh = star[e..].iter().zip(&loaded[e..]).map(|(a, b)| (a - b).abs()).fold(0.0f64, f64::max);
        println!(
            "  {beta:>5.2}   {kl_loaded:>10.3e} {tv_loaded:>10.3e}   {b_eff:>8.4} {kl_eff:>10.3e} {tv_eff:>10.3e}   {kl_pair:>10.3e} {tv_pair:>10.3e} {dj:>7.4} {dh:>7.4} {:>7.2}%   {iters:>3} {grad:>5.0e}",
            100.0 * kl_pair / kl_loaded
        );
    }
    println!("\n  WHAT THE TABLE SAYS.\n");
    println!("  'loaded' is the Boltzmann law the fabric was programmed for; 'b_eff' the nearest single temperature;");
    println!("  'pair' the nearest Boltzmann law of ANY couplings and fields, found exactly. max|dJ| and max|dh| are how");
    println!("  far that pair sits from the loaded parameters, in the units the fabric loads (beta already applied).");
    println!("  'not-B %' is the share of the loaded loss that survives the best possible pair: the part of the");
    println!("  fabric's law that is not the Boltzmann distribution of anything, which no calibrated load can remove.");
    println!("  'newton' is the iterations and the sup-norm of the moment gap at the stop; the objective is strictly");
    println!("  convex, so a gap at the noise floor is the global optimum.");
}
