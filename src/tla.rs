//! Thermodynamic linear algebra — the continuous (pmode) side of the field.
//!
//! An Ornstein-Uhlenbeck network with drift -(A x - b) and isotropic noise has stationary
//! distribution N(A^-1 b, A^-1 / beta) for symmetric positive-definite A. Time-averaging the
//! state solves A x = b; the sample covariance estimates A^-1. This is the algorithm class
//! behind thermodynamic linear-algebra hardware (Aifer et al., arXiv:2308.05660; the 8-cell SPU
//! of arXiv:2312.04836 ran the 2- and 3-dimensional cases physically). Here it is an explicit
//! Euler-Maruyama simulation with a ledger of steps, so the algorithm is usable, teachable, and
//! priceable on any machine — and swappable for physical dynamics when hardware exists to measure.
//!
//! Verification standard: exact solves on enumerably small SPD systems, tolerance stated.

use crate::rng::Pcg;

fn gauss(rng: &mut Pcg) -> f64 {
    let a = rng.f64().max(1e-15);
    let b = rng.f64();
    (-2.0 * a.ln()).sqrt() * (std::f64::consts::TAU * b).cos()
}

/// Dense symmetric positive-definite system, row-major.
pub struct Spd {
    pub n: usize,
    pub a: Vec<f64>,
    pub b: Vec<f64>,
}

impl Spd {
    pub fn new(n: usize, a: Vec<f64>, b: Vec<f64>) -> Spd {
        assert_eq!(a.len(), n * n);
        assert_eq!(b.len(), n);
        // symmetry check; positive-definiteness is the caller's contract (Gershgorin or Cholesky
        // can check it; the OU dynamics simply diverge if violated, which the estimator reports).
        for i in 0..n {
            for j in 0..n {
                assert!((a[i * n + j] - a[j * n + i]).abs() < 1e-9, "A must be symmetric");
            }
        }
        Spd { n, a, b }
    }

    fn drift(&self, x: &[f64], out: &mut [f64]) {
        for i in 0..self.n {
            let mut s = -self.b[i];
            for j in 0..self.n {
                s += self.a[i * self.n + j] * x[j];
            }
            out[i] = -s; // dx/dt = -(A x - b)
        }
    }
}

pub struct TlaResult {
    /// Time-averaged state — the estimate of A^-1 b.
    pub x: Vec<f64>,
    /// Sample covariance times beta — the estimate of A^-1.
    pub a_inv: Vec<f64>,
    /// Euler-Maruyama steps taken (burn-in + measurement): the cost the ledger prices.
    pub steps: u64,
}

/// Simulate the OU network and return the thermodynamic solve. `dt` must satisfy
/// dt < 2 / lambda_max(A) for stability; `burn` steps equilibrate, `measure` steps average.
pub fn solve_spd(sys: &Spd, beta: f64, dt: f64, burn: usize, measure: usize, seed: u64) -> TlaResult {
    let n = sys.n;
    let mut rng = Pcg::new(seed, 0x71A);
    let mut x = vec![0.0; n];
    let mut d = vec![0.0; n];
    let noise = (2.0 * dt / beta).sqrt();
    for _ in 0..burn {
        sys.drift(&x, &mut d);
        for i in 0..n {
            x[i] += dt * d[i] + noise * gauss(&mut rng);
        }
    }
    let mut mean = vec![0.0; n];
    let mut cov = vec![0.0; n * n];
    for _ in 0..measure {
        sys.drift(&x, &mut d);
        for i in 0..n {
            x[i] += dt * d[i] + noise * gauss(&mut rng);
        }
        for i in 0..n {
            mean[i] += x[i];
        }
        for i in 0..n {
            for j in 0..n {
                cov[i * n + j] += x[i] * x[j];
            }
        }
    }
    let m = measure as f64;
    for v in mean.iter_mut() {
        *v /= m;
    }
    let mut a_inv = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            a_inv[i * n + j] = beta * (cov[i * n + j] / m - mean[i] * mean[j]);
        }
    }
    TlaResult { x: mean, a_inv, steps: (burn + measure) as u64 }
}

/// Exact Ornstein-Uhlenbeck transition sampling — the bias-free integrator.
///
/// The Euler-Maruyama chain's stationary MEAN is exactly A^-1 b for any stable dt, but its
/// stationary covariance is biased per eigenmode by the factor 2 / (2 - dt * alpha) (the
/// discrete Lyapunov solution). The exact transition removes that bias: over a stride h,
///     x(t+h) = A^-1 b + exp(-A h) (x(t) - A^-1 b) + eta,
///     eta ~ N(0, beta^-1 A^-1 (I - exp(-2 A h))),
/// evaluated in the eigenbasis of A (Aifer et al., arXiv:2308.05660, Eq. 35). Strides of a few
/// relaxation times give near-independent samples.
pub fn solve_spd_exact_ou(
    sys: &Spd,
    beta: f64,
    stride_h: f64,
    burn_strides: usize,
    samples: usize,
    seed: u64,
) -> TlaResult {
    let n = sys.n;
    let mut d = sys.a.clone();
    let v = crate::linalg::jacobi_eig(&mut d, n);
    let lam: Vec<f64> = (0..n).map(|c| d[c * n + c]).collect();
    assert!(lam.iter().all(|&l| l > 0.0), "A must be positive definite");
    // x* = A^-1 b via the eigenbasis
    let mut xstar = vec![0.0; n];
    for i in 0..n {
        for c in 0..n {
            let mut btv = 0.0;
            for j in 0..n {
                btv += v[j * n + c] * sys.b[j];
            }
            xstar[i] += v[i * n + c] * btv / lam[c];
        }
    }
    let decay: Vec<f64> = lam.iter().map(|&l| (-l * stride_h).exp()).collect();
    let nstd: Vec<f64> = lam
        .iter()
        .zip(&decay)
        .map(|(&l, &e)| ((1.0 - e * e) / (beta * l)).max(0.0).sqrt())
        .collect();
    let mut rng = Pcg::new(seed, 0xE0);
    // state in the eigenbasis, centered on x*
    let mut y = vec![0.0; n];
    let step = |y: &mut Vec<f64>, rng: &mut Pcg| {
        for c in 0..n {
            y[c] = decay[c] * y[c] + nstd[c] * gauss(rng);
        }
    };
    for _ in 0..burn_strides {
        step(&mut y, &mut rng);
    }
    let mut mean = vec![0.0; n];
    let mut cov_y = vec![0.0; n * n];
    let mut ymean = vec![0.0; n];
    for _ in 0..samples {
        step(&mut y, &mut rng);
        for c in 0..n {
            ymean[c] += y[c];
        }
        for a in 0..n {
            for bq in 0..n {
                cov_y[a * n + bq] += y[a] * y[bq];
            }
        }
    }
    let m = samples as f64;
    for c in 0..n {
        ymean[c] /= m;
    }
    for i in 0..n {
        let mut xi = xstar[i];
        for c in 0..n {
            xi += v[i * n + c] * ymean[c];
        }
        mean[i] = xi;
    }
    // covariance back in the original basis, times beta -> A^-1 estimate
    let mut a_inv = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut s = 0.0;
            for a in 0..n {
                for bq in 0..n {
                    s += v[i * n + a] * (cov_y[a * n + bq] / m - ymean[a] * ymean[bq]) * v[j * n + bq];
                }
            }
            a_inv[i * n + j] = beta * s;
        }
    }
    TlaResult { x: mean, a_inv, steps: (burn_strides + samples) as u64 }
}

/// Exact reference solve by Gaussian elimination with partial pivoting (for verification).
pub fn solve_exact(sys: &Spd) -> Vec<f64> {
    let n = sys.n;
    let mut aug = vec![0.0; n * (n + 1)];
    for i in 0..n {
        for j in 0..n {
            aug[i * (n + 1) + j] = sys.a[i * n + j];
        }
        aug[i * (n + 1) + n] = sys.b[i];
    }
    for col in 0..n {
        let mut piv = col;
        for r in col + 1..n {
            if aug[r * (n + 1) + col].abs() > aug[piv * (n + 1) + col].abs() {
                piv = r;
            }
        }
        for k in 0..n + 1 {
            aug.swap(col * (n + 1) + k, piv * (n + 1) + k);
        }
        let p = aug[col * (n + 1) + col];
        for k in 0..n + 1 {
            aug[col * (n + 1) + k] /= p;
        }
        for r in 0..n {
            if r != col {
                let f = aug[r * (n + 1) + col];
                for k in 0..n + 1 {
                    aug[r * (n + 1) + k] -= f * aug[col * (n + 1) + k];
                }
            }
        }
    }
    (0..n).map(|i| aug[i * (n + 1) + n]).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_system() -> Spd {
        // 4x4 SPD: diagonally dominant symmetric
        Spd::new(
            4,
            vec![
                4.0, 1.0, 0.5, 0.0,
                1.0, 3.0, 0.7, 0.2,
                0.5, 0.7, 3.5, 1.0,
                0.0, 0.2, 1.0, 2.5,
            ],
            vec![1.0, -2.0, 0.5, 3.0],
        )
    }

    /// The thermodynamic solve must agree with exact Gaussian elimination.
    #[test]
    fn ou_solves_linear_system() {
        let sys = test_system();
        let exact = solve_exact(&sys);
        let r = solve_spd(&sys, 8.0, 0.02, 20_000, 400_000, 0x11A);
        for i in 0..sys.n {
            assert!(
                (r.x[i] - exact[i]).abs() < 0.02,
                "x[{i}]: thermo {} vs exact {}",
                r.x[i],
                exact[i]
            );
        }
    }

    /// THE BIAS LAW (the test that catches silently 'fixing' the discretization): the
    /// Euler-Maruyama chain's stationary variance per eigenmode is NOT beta^-1 alpha^-1 but
    /// beta^-1 alpha^-1 * 2/(2 - dt*alpha). On diagonal A the modes are the coordinates, so the
    /// empirical variances must match the BIASED closed form, not the ideal one.
    #[test]
    fn em_covariance_bias_law() {
        let sys = Spd::new(2, vec![1.0, 0.0, 0.0, 4.0], vec![0.0, 0.0]);
        let beta = 1.0;
        let dt = 0.2 / 4.0; // dt * alpha_max = 0.2 -> mode-2 bias factor 2/(2-0.2) = 1.111
        let r = solve_spd(&sys, beta, dt, 50_000, 2_000_000, 0x3B1A5);
        for (mode, alpha) in [(0usize, 1.0f64), (1, 4.0)] {
            let want = (1.0 / (beta * alpha)) * 2.0 / (2.0 - dt * alpha);
            let got = r.a_inv[mode * 2 + mode] / beta; // a_inv = beta * cov, so cov = a_inv/beta
            assert!(
                (got - want).abs() / want < 0.03,
                "mode {mode}: EM variance {got:.4} vs biased closed form {want:.4}"
            );
        }
    }

    /// The exact-transition integrator must be bias-free: variances land on beta^-1 alpha^-1.
    #[test]
    fn exact_ou_is_unbiased() {
        let sys = Spd::new(2, vec![1.0, 0.0, 0.0, 4.0], vec![0.5, -1.0]);
        let beta = 1.0;
        let r = solve_spd_exact_ou(&sys, beta, 2.0, 100, 400_000, 0xE0A7);
        let exact = solve_exact(&sys);
        for i in 0..2 {
            assert!((r.x[i] - exact[i]).abs() < 0.01, "mean[{i}] {} vs {}", r.x[i], exact[i]);
        }
        for (mode, alpha) in [(0usize, 1.0f64), (1, 4.0)] {
            let want = 1.0 / (beta * alpha);
            let got = r.a_inv[mode * 2 + mode] / beta;
            assert!(
                (got - want).abs() / want < 0.02,
                "mode {mode}: exact-OU variance {got:.4} vs unbiased {want:.4}"
            );
        }
    }

    /// The sample covariance must estimate A^-1: check A * a_inv ~ I.
    #[test]
    fn covariance_estimates_inverse() {
        let sys = test_system();
        let r = solve_spd(&sys, 8.0, 0.02, 20_000, 800_000, 0x22B);
        let n = sys.n;
        for i in 0..n {
            for j in 0..n {
                let mut s = 0.0;
                for k in 0..n {
                    s += sys.a[i * n + k] * r.a_inv[k * n + j];
                }
                let want = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (s - want).abs() < 0.12,
                    "(A a_inv)[{i}{j}] = {s}, want {want}"
                );
            }
        }
    }
}
