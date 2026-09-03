//! The Hopfield model — statistical mechanics of learning, with its closed forms as oracles.
//!
//! An associative memory is an Ising model: store patterns `ξ^μ ∈ {±1}^N` in Hebbian couplings
//! `J_ij = (1/N) Σ_μ ξ_i^μ ξ_j^μ`, and retrieval is the sampler falling into the basin of a
//! pattern. This crate's samplers run it unchanged; what was missing is what the answer should
//! be, and for this model the theory says exactly.
//!
//! * **One pattern** is the Curie–Weiss ferromagnet in a gauge: the overlap solves `m = tanh(βm)`,
//!   [`curie_weiss_m`], exact in the thermodynamic limit.
//! * **Load `α = P/N > 0`** is Amit–Gutfreund–Sompolinsky (1985, 1987): the replica-symmetric
//!   equations [`ags_rs`]
//!
//!   ```text
//!     m = ∫Dz tanh(β(m + √(αr) z)),    q = ∫Dz tanh²(β(m + √(αr) z)),    r = q / (1 − β(1−q))²
//!   ```
//!
//!   have a retrieval solution `m > 0` below a load `α_c(T)` and none above it; at `T = 0`,
//!   `α_c ≈ 0.138`. That transition is the capacity of the memory, and it is observable: a chain
//!   started at a stored pattern holds its overlap below `α_c` and loses it above. The Gaussian
//!   integrals are Gauss–Hermite ([`gauss_hermite`]), checked on the moments of the normal.
//!
//! Replica symmetry is an approximation here (the retrieval phase has a small RSB correction near
//! `α_c`), and every finite-`N` measurement carries `O(1/√N)` corrections; the tests state their
//! tolerances from the samplers' own error bars, and the transition test asks only that retrieval
//! be clearly present at `α = 0.02` and clearly gone at `α = 0.30` — both far from the boundary.

use crate::gibbs::Sampler;
use crate::graph::{Graph, GraphBuilder};
use crate::rng::Pcg;
use crate::samples::Estimate;

/// `p` random patterns of `n` spins.
pub fn random_patterns(n: usize, p: usize, seed: u64) -> Vec<Vec<i8>> {
    let mut rng = Pcg::new(seed, 7);
    (0..p).map(|_| (0..n).map(|_| if rng.f64() < 0.5 { -1 } else { 1 }).collect()).collect()
}

/// The Hebbian couplings `J_ij = (1/N) Σ_μ ξ_i^μ ξ_j^μ` for `i ≠ j`, as a dense graph.
pub fn hebbian(patterns: &[Vec<i8>]) -> Graph {
    let n = patterns.first().map_or(0, |p| p.len());
    assert!(patterns.iter().all(|p| p.len() == n));
    let mut gb = GraphBuilder::new(n);
    for i in 0..n {
        for j in (i + 1)..n {
            let s: i32 = patterns.iter().map(|p| p[i] as i32 * p[j] as i32).sum();
            if s != 0 {
                gb.couple(i, j, s as f64 / n as f64);
            }
        }
    }
    gb.build()
}

/// The overlap `(1/N) Σ_i ξ_i s_i` of a state with a pattern.
pub fn overlap(pattern: &[i8], s: &[i8]) -> f64 {
    pattern.iter().zip(s).map(|(&a, &b)| (a as i32 * b as i32) as f64).sum::<f64>() / pattern.len() as f64
}

/// Curie–Weiss: the positive solution of `m = tanh(βm)`, or `0` for `β ≤ 1`.
pub fn curie_weiss_m(beta: f64) -> f64 {
    if beta <= 1.0 {
        return 0.0;
    }
    let (mut lo, mut hi) = (1e-12, 1.0);
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if (beta * mid).tanh() > mid {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Gauss–Hermite nodes and weights for `∫ e^{−t²} f(t) dt`, `n` points (physicists' convention).
pub fn gauss_hermite(n: usize) -> (Vec<f64>, Vec<f64>) {
    assert!(n >= 2);
    let mut x = vec![0.0; n];
    let mut w = vec![0.0; n];
    let pim4 = 0.751_125_544_464_942_5; // π^{-1/4}
    let mut z = 0.0;
    for i in 0..n.div_ceil(2) {
        // initial guesses (Numerical Recipes, gauher)
        z = match i {
            0 => (2.0 * n as f64 + 1.0).sqrt() - 1.85575 * (2.0 * n as f64 + 1.0).powf(-1.0 / 6.0),
            1 => z - 1.14 * (n as f64).powf(0.426) / z,
            2 => 1.86 * z - 0.86 * x[0],
            3 => 1.91 * z - 0.91 * x[1],
            _ => 2.0 * z - x[i - 2],
        };
        let mut pp = 0.0;
        for _ in 0..100 {
            let mut p1 = pim4;
            let mut p2 = 0.0;
            for j in 1..=n {
                let p3 = p2;
                p2 = p1;
                p1 = z * (2.0 / j as f64).sqrt() * p2 - ((j as f64 - 1.0) / j as f64).sqrt() * p3;
            }
            pp = (2.0 * n as f64).sqrt() * p2;
            let z1 = z;
            z = z1 - p1 / pp;
            if (z - z1).abs() <= 1e-14 {
                break;
            }
        }
        x[i] = z;
        x[n - 1 - i] = -z;
        w[i] = 2.0 / (pp * pp);
        w[n - 1 - i] = w[i];
    }
    (x, w)
}

/// `∫Dz f(z)` with `Dz` the standard normal, by `n`-point Gauss–Hermite.
pub fn gaussian_expectation(n: usize, f: impl Fn(f64) -> f64) -> f64 {
    let (x, w) = gauss_hermite(n);
    let s2 = core::f64::consts::SQRT_2;
    x.iter().zip(&w).map(|(&t, &wt)| wt * f(s2 * t)).sum::<f64>() / core::f64::consts::PI.sqrt()
}

/// A replica-symmetric solution of the AGS equations.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RsSolution {
    pub alpha: f64,
    pub beta: f64,
    /// Retrieval overlap; `0` (to tolerance) means no retrieval state at this load.
    pub m: f64,
    pub q: f64,
    pub r: f64,
    pub iterations: usize,
}

/// Solve the AGS replica-symmetric equations by damped fixed-point iteration from the retrieval
/// side (`m = 1`). If the iteration collapses to `m ≈ 0` there is no retrieval state at `(α, β)`.
pub fn ags_rs(alpha: f64, beta: f64) -> RsSolution {
    assert!(alpha >= 0.0 && beta > 0.0);
    let (mut m, mut q, mut r) = (1.0f64, 1.0f64, 1.0f64);
    let mut it = 0;
    for _ in 0..5000 {
        it += 1;
        let s = (alpha * r).sqrt();
        let m_new = gaussian_expectation(64, |z| (beta * (m + s * z)).tanh());
        let q_new = gaussian_expectation(64, |z| (beta * (m + s * z)).tanh().powi(2));
        let denom = 1.0 - beta * (1.0 - q_new);
        let r_new = if denom.abs() < 1e-9 { 1e18 } else { q_new / (denom * denom) };
        let (dm, dq, dr) = ((m_new - m).abs(), (q_new - q).abs(), (r_new - r).abs().min(1.0));
        m = 0.5 * m + 0.5 * m_new;
        q = 0.5 * q + 0.5 * q_new;
        r = 0.5 * r + 0.5 * r_new;
        if dm < 1e-12 && dq < 1e-12 && dr < 1e-9 {
            break;
        }
    }
    RsSolution { alpha, beta, m, q, r, iterations: it }
}

/// The error function, Abramowitz–Stegun 7.1.26: absolute error below `1.5e-7`, which is far
/// inside what the `T = 0` equations below are asked to resolve.
pub fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    sign * (1.0 - poly * (-x * x).exp())
}

/// The AGS equations at `T = 0`, in their closed form:
///
/// ```text
///   m = erf( m / √(2αr) ),    C = √(2/(παr)) · exp(−m²/(2αr)),    r = 1 / (1 − C)²,
/// ```
///
/// iterated from the retrieval side. `None` when no retrieval solution survives — `C` reaches 1
/// or `m` collapses — which happens above the capacity `α_c ≈ 0.138`.
pub fn ags_zero_t(alpha: f64) -> Option<RsSolution> {
    assert!(alpha > 0.0);
    let (mut m, mut r) = (1.0f64, 1.0f64);
    let mut it = 0;
    for _ in 0..20_000 {
        it += 1;
        let ar = alpha * r;
        let m_new = erf(m / (2.0 * ar).sqrt());
        let c = (2.0 / (core::f64::consts::PI * ar)).sqrt() * (-m * m / (2.0 * ar)).exp();
        if c >= 1.0 || m_new < 1e-6 {
            return None;
        }
        let r_new = 1.0 / ((1.0 - c) * (1.0 - c));
        let (dm, dr) = ((m_new - m).abs(), (r_new - r).abs());
        m = 0.5 * m + 0.5 * m_new;
        r = 0.5 * r + 0.5 * r_new;
        if dm < 1e-13 && dr < 1e-11 {
            break;
        }
    }
    // q = 1 at T = 0; the `C = β(1 − q)` limit is what `r` carries.
    Some(RsSolution { alpha, beta: f64::INFINITY, m, q: 1.0, r, iterations: it })
}

/// The `T = 0` capacity: the largest load with a retrieval solution, by bisection on
/// [`ags_zero_t`] to `1e-6`. The replica-symmetric value is `≈ 0.138`.
pub fn capacity_zero_t() -> f64 {
    let (mut lo, mut hi) = (0.05, 0.30);
    for _ in 0..60 {
        let mid = 0.5 * (lo + hi);
        if ags_zero_t(mid).is_some() {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Retrieval measured: a chain started at `pattern`, `burn_in` sweeps, then `draws` sweeps each
/// recording the overlap, with an error bar from the trace's autocorrelation.
pub fn retrieval_overlap(g: &Graph, pattern: &[i8], beta: f64, burn_in: usize, draws: usize, seed: u64) -> Estimate {
    let mut sm = Sampler::new(g, beta, seed);
    sm.s.copy_from_slice(pattern);
    sm.sweeps(burn_in, None);
    let mut trace = Vec::with_capacity(draws);
    for _ in 0..draws {
        sm.sweep(None);
        trace.push(overlap(pattern, &sm.s));
    }
    crate::free_energy::estimate_trace(&trace)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gauss_hermite_integrates_the_normal_moments() {
        for &(k, want) in &[(0, 1.0), (2, 1.0), (4, 3.0), (6, 15.0)] {
            let got = gaussian_expectation(64, |z| z.powi(k));
            assert!((got - want).abs() < 1e-9, "E[z^{k}] = {got}, want {want}");
        }
    }

    #[test]
    fn erf_is_accurate_where_it_is_used() {
        for &(x, want) in &[(0.0, 0.0), (0.5, 0.520_499_877_8), (1.0, 0.842_700_792_9), (2.0, 0.995_322_265_0)] {
            assert!((erf(x) - want).abs() < 2e-7 && (erf(-x) + want).abs() < 2e-7);
        }
    }

    #[test]
    fn curie_weiss_is_self_consistent_and_has_its_transition() {
        assert_eq!(curie_weiss_m(0.9), 0.0);
        for beta in [1.1, 1.5, 2.0, 3.0] {
            let m = curie_weiss_m(beta);
            assert!(m > 0.0 && ((beta * m).tanh() - m).abs() < 1e-10);
        }
        assert!((curie_weiss_m(1.5) - 0.8580).abs() < 1e-3);
    }

    /// One pattern IS Curie–Weiss: the sampled overlap matches m = tanh(βm) within its error bar.
    #[test]
    fn one_pattern_retrieves_at_the_curie_weiss_overlap() {
        let n = 256;
        let pats = random_patterns(n, 1, 1);
        let g = hebbian(&pats);
        for beta in [1.5, 2.5] {
            let want = curie_weiss_m(beta);
            let got = retrieval_overlap(&g, &pats[0], beta, 200, 2000, 9);
            // finite size: O(1/N) shift plus the chain's own error bar
            let tol = 4.0 * got.stderr + 3.0 / n as f64 + 0.01;
            assert!((got.value - want).abs() < tol, "beta {beta}: overlap {} vs Curie-Weiss {want} (tol {tol})", got.value);
        }
    }

    /// AGS says retrieval exists at α = 0.02 and not at α = 0.30 (β = 2); the sampler agrees.
    #[test]
    fn the_capacity_transition_is_where_the_replica_theory_puts_it() {
        let beta = 2.0;
        let low = ags_rs(0.02, beta);
        let high = ags_rs(0.30, beta);
        assert!(low.m > 0.9, "AGS retrieval at alpha 0.02: m = {}", low.m);
        assert!(high.m < 0.05, "AGS says no retrieval at alpha 0.30: m = {}", high.m);
        // and the famous number: at T = 0 the replica-symmetric capacity is 0.138, with the
        // retrieval overlap still about 0.97 just below it -- a first-order transition.
        let ac = capacity_zero_t();
        assert!((0.135..0.141).contains(&ac), "alpha_c(T=0) = {ac}, expected about 0.138");
        let just_below = ags_zero_t(ac - 0.002).expect("retrieval just below capacity");
        assert!(just_below.m > 0.95, "m just below alpha_c = {}", just_below.m);
        assert!(ags_zero_t(0.16).is_none());

        let n = 400;
        let pats_low = random_patterns(n, 8, 2); // α = 0.02
        let g_low = hebbian(&pats_low);
        let got_low = retrieval_overlap(&g_low, &pats_low[0], beta, 100, 400, 3);
        assert!((got_low.value - low.m).abs() < 4.0 * got_low.stderr + 0.06, "alpha 0.02: sampled {} vs AGS {}", got_low.value, low.m);

        let pats_high = random_patterns(n, 120, 4); // α = 0.30
        let g_high = hebbian(&pats_high);
        let got_high = retrieval_overlap(&g_high, &pats_high[0], beta, 100, 400, 5);
        assert!(got_high.value < 0.6, "alpha 0.30: the pattern is not retrieved, overlap {}", got_high.value);
        assert!(got_low.value > 0.85);
    }
}
