//! Lattice Random Walk SDE discretisation — the published algorithm behind Normal Computing's
//! CN101 "stochastic sampling with lattice random walk" (arXiv:2508.20883).
//!
//! Simulates dx = f(x,t) dt + sigma(x,t) dW (diagonal noise) using only ternary increments per
//! coordinate: Delta_i in {-dx_i, 0, +dx_i} with
//!     P[Delta = +-dx] = 0.5 * (dt/dx) * ( +-f + sigma^2/dx ),
//! which gives EXACT conditional moments `E[Delta] = dt f` and `E[Delta^2] = dt sigma^2`. With
//! dx = sqrt(dt) * sigma the zero branch vanishes (a pure coin flip). The stability mechanism,
//! worth stating: Euler-Maruyama's second moment is dt sigma^2 + dt^2 f^2 (unbounded for
//! super-linear drifts), the walk's is dt sigma^2 independent of f — so non-globally-Lipschitz
//! drifts that explode under EM stay bounded here. Clipping (sigma first so p_- + p_+ <= 1, then
//! f so probabilities stay nonnegative) vanishes as dt -> 0, preserving weak order 1.

use crate::rng::Pcg;

/// One LRW step for coordinate value `x` with drift `f`, noise `sigma`, step `dt`, lattice
/// spacing `dx`. Returns the increment in {-dx, 0, +dx}. Applies the validity clipping.
#[inline]
pub fn step_coord(f: f64, sigma: f64, dt: f64, dx: f64, rng: &mut Pcg) -> f64 {
    let (p_minus, p_plus) = probs(f, sigma, dt, dx);
    let u = rng.f64();
    if u < p_minus {
        -dx
    } else if u < p_minus + p_plus {
        dx
    } else {
        0.0
    }
}

/// The clipped branch probabilities (p_minus, p_plus).
#[inline]
pub fn probs(f: f64, sigma: f64, dt: f64, dx: f64) -> (f64, f64) {
    // clip sigma so dt*sigma^2/dx^2 <= 1 (total jump probability <= 1)
    let s2 = (sigma * sigma).min(dx * dx / dt);
    // clip f so |dt*f/dx| <= dt*s2/dx^2, i.e. |f| <= s2/dx (keeps both branches >= 0)
    let fmax = s2 / dx;
    let fc = f.clamp(-fmax, fmax);
    let base = 0.5 * (dt / dx) * (s2 / dx);
    let tilt = 0.5 * (dt / dx) * fc;
    (base - tilt, base + tilt)
}

/// Integrate a d-dimensional SDE for `n_steps` from `x`, with per-coordinate drift and noise
/// closures. `dx_lat` per coordinate (rule of thumb: sqrt(dt) * sigma_max).
pub fn integrate<D, S>(
    x: &mut [f64],
    t0: f64,
    dt: f64,
    dx_lat: &[f64],
    n_steps: usize,
    drift: D,
    sigma: S,
    rng: &mut Pcg,
) where
    D: Fn(&[f64], f64, usize) -> f64,
    S: Fn(&[f64], f64, usize) -> f64,
{
    let d = x.len();
    let mut incr = vec![0.0; d];
    let mut t = t0;
    for _ in 0..n_steps {
        for i in 0..d {
            incr[i] = step_coord(drift(x, t, i), sigma(x, t, i), dt, dx_lat[i], rng);
        }
        for i in 0..d {
            x[i] += incr[i];
        }
        t += dt;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// EXACT algebraic identities, no Monte Carlo: the branch probabilities must reproduce the
    /// first two conditional moments exactly, and the binary reduction must be a pure coin flip.
    #[test]
    fn exact_moment_identities() {
        let mut rng = Pcg::new(0x11A7, 9);
        for _ in 0..1000 {
            let sigma = 0.2 + rng.f64() * 2.0;
            let dt = 0.001 + rng.f64() * 0.05;
            // dx within the validity window sqrt(dt)*sigma <= dx <= sigma^2/|f|
            let dx = (dt.sqrt() * sigma) * (1.0 + rng.f64());
            let fmax = sigma * sigma / dx;
            let f = (rng.f64() * 2.0 - 1.0) * 0.9 * fmax;
            let (pm, pp) = probs(f, sigma, dt, dx);
            let mean = pp * dx - pm * dx;
            let second = (pp + pm) * dx * dx;
            assert!((mean - dt * f).abs() < 1e-14, "E[Delta] {} vs {}", mean, dt * f);
            assert!(
                (second - dt * sigma * sigma).abs() < 1e-13,
                "E[Delta^2] {} vs {}",
                second,
                dt * sigma * sigma
            );
            // binary reduction
            let dxb = dt.sqrt() * sigma;
            let fb = (rng.f64() * 2.0 - 1.0) * 0.9 * (sigma * sigma / dxb);
            let (bm, bp) = probs(fb, sigma, dt, dxb);
            assert!((bm + bp - 1.0).abs() < 1e-12, "binary: p-+p+ = {}", bm + bp);
        }
    }

    /// Constant drift: the N-step sum has exactly mean N dt c and variance N (dt sigma^2 - dt^2 c^2).
    #[test]
    fn constant_drift_law() {
        let (c, sigma, dt): (f64, f64, f64) = (0.7, 1.3, 0.01);
        let dx = dt.sqrt() * sigma;
        let n_steps = 2000usize;
        let n_paths = 4000usize;
        let mut rng = Pcg::new(0xC0157, 2);
        let (mut sum, mut sum2) = (0.0, 0.0);
        for _ in 0..n_paths {
            let mut x = [0.0f64];
            integrate(&mut x, 0.0, dt, &[dx], n_steps, |_, _, _| c, |_, _, _| sigma, &mut rng);
            sum += x[0];
            sum2 += x[0] * x[0];
        }
        let mean = sum / n_paths as f64;
        let var = sum2 / n_paths as f64 - mean * mean;
        let want_mean = n_steps as f64 * dt * c;
        let want_var = n_steps as f64 * (dt * sigma * sigma - dt * dt * c * c);
        let se_mean = (want_var / n_paths as f64).sqrt();
        assert!((mean - want_mean).abs() < 4.0 * se_mean, "mean {mean} vs {want_mean}");
        assert!((var - want_var).abs() / want_var < 0.1, "var {var} vs {want_var}");
    }

    /// The stability mechanism: dx = -x^3 dt + sqrt(2) dW at dt = 0.1. Euler-Maruyama is
    /// unstable once dt * x^2 > 2 (|x| > 4.47 here): each step OVERSHOOTS and grows, so from
    /// x0 = 5 it diverges deterministically. The walk's increment is bounded by dx regardless of
    /// the drift's magnitude (clipping), so from the same start it walks home and stays bounded.
    #[test]
    fn cubic_drift_stays_bounded() {
        let (sigma, dt): (f64, f64) = (2.0f64.sqrt(), 0.1);
        let dx = dt.sqrt() * sigma;
        let x0 = 5.0f64; // past the EM instability threshold sqrt(2/dt) = 4.47
        let mut rng = Pcg::new(0xC3B1C, 4);
        let mut x = [x0];
        let mut max_abs: f64 = 0.0;
        for _ in 0..100_000 {
            integrate(&mut x, 0.0, dt, &[dx], 1, |x, _, _| -x[0].powi(3), |_, _, _| sigma, &mut rng);
            max_abs = max_abs.max(x[0].abs());
        }
        assert!(max_abs <= x0 + dx + 1e-12, "walk escaped: max |x| = {max_abs}");
        // EM from the same start: |x_{k+1}| = |x_k| * |1 - dt x_k^2| grows without bound
        let mut rng2 = Pcg::new(0xC3B1C, 4);
        let mut xe = x0;
        let mut diverged = false;
        for _ in 0..100 {
            let a = rng2.f64().max(1e-15);
            let b = rng2.f64();
            let g = (-2.0 * a.ln()).sqrt() * (std::f64::consts::TAU * b).cos();
            xe += -xe.powi(3) * dt + sigma * dt.sqrt() * g;
            if !xe.is_finite() || xe.abs() > 1e6 {
                diverged = true;
                break;
            }
        }
        assert!(diverged, "EM did not diverge from x0 = {x0} (reached {xe}); contrast unearned");
    }

    /// OU stationary check: the walk's stationary mean must solve A x = b (weak order 1).
    #[test]
    fn ou_stationary_mean() {
        // dx = -(A x - b) dt + sqrt(2 T) dW, T = 0.5, A SPD 2x2
        let a = [[2.0, 1.0], [1.0, 2.0]];
        let b = [1.0, 1.0];
        let t_temp = 0.5f64;
        let sigma = (2.0 * t_temp).sqrt();
        let dt = 0.002f64;
        let dx = dt.sqrt() * sigma;
        let mut rng = Pcg::new(0x0057A7, 6);
        let mut x = [0.0f64, 0.0];
        integrate(&mut x, 0.0, dt, &[dx, dx], 200_000, // burn-in
            |x, _, i| b[i] - (a[i][0] * x[0] + a[i][1] * x[1]),
            |_, _, _| sigma, &mut rng);
        let (mut m0, mut m1) = (0.0, 0.0);
        let n = 2_000_000usize;
        for _ in 0..n {
            integrate(&mut x, 0.0, dt, &[dx, dx], 1,
                |x, _, i| b[i] - (a[i][0] * x[0] + a[i][1] * x[1]),
                |_, _, _| sigma, &mut rng);
            m0 += x[0];
            m1 += x[1];
        }
        // exact solution x* = A^-1 b = [1/3, 1/3]
        let (m0, m1) = (m0 / n as f64, m1 / n as f64);
        assert!((m0 - 1.0 / 3.0).abs() < 0.02, "x0 {m0}");
        assert!((m1 - 1.0 / 3.0).abs() < 0.02, "x1 {m1}");
    }
}
