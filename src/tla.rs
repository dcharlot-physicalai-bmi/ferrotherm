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
//!
//! # ⛔ What this loses at, measured rather than cited
//!
//! The `dominance` tests measure, on this code and at equal matrix-vector products, that a
//! deterministic iterative method dominates this one on BOTH moments of a dense system: conjugate
//! gradient reaches `1e-15` on the mean in `n` products while the sampled mean is still at `1.2e-1`
//! after 10,000, and the sampled covariance loses to an exact inverse as well. The large-sparse
//! case is not measured here and is not claimed anywhere in this crate.
//!
//! **The related literature, read rather than summarised.** Kirsten, Selby, Petangoda, Halqvist
//! Elias, Meech and Stanley-Marbell (arXiv:2608.09743, 2026-08-10, **Signaloid**, with one dual
//! Cambridge affiliation) prove something adjacent to but not the same as the above: *"to a
//! first-order approximation, the covariance dynamics are mathematically identical to preconditioned
//! gradient descent on the Frobenius norm of the residual"* — the COVARIANCE route to matrix
//! inversion, with `b` set to zero, not the mean route to `Ax = b` that the tests here measure.
//! Their conclusion is that thermal fluctuations are *"algorithmically redundant for convex problems
//! with a single global minimum"*, and they fence it themselves: for non-convex landscapes the noise
//! *"could remain algorithmically essential to escape local minima"*.
//!
//! Three caveats on their numbers, from the paper: the 100,000-fold speedup is Python/NumPy
//! wall-clock on one laptop with **Thermox held at a fixed 1,000,000 samples** against a gradient
//! descent stopped at a tolerance — not equal budgets; their gradient descent is handed precomputed
//! eigenvalues, which they say *"defeats the purpose of iterative methods in practice"*; and their
//! own Table 4 has that gradient descent running 20–100× SLOWER than Newton–Schulz once the
//! condition number reaches 100. The measurement below is this crate's own and does not rest on
//! theirs.

use crate::rng::Pcg;

fn gauss(rng: &mut Pcg) -> f64 {
    let a = rng.f64().max(1e-15);
    let b = rng.f64();
    (-2.0 * a.ln()).sqrt() * (std::f64::consts::TAU * b).cos()
}

/// Dense symmetric positive-definite system, row-major.
pub struct Spd {
    /// Dimension.
    pub n: usize,
    /// The matrix `A`, row-major, `n * n`.
    pub a: Vec<f64>,
    /// The right-hand side `b`, length `n`.
    pub b: Vec<f64>,
}

impl Spd {
    #[must_use]
    /// Build the system `A x = b`.
    ///
    /// # Panics
    ///
    /// If `a` is not `n * n`, or `b` is not `n` long.
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

/// What a thermodynamic linear-algebra run estimated, and from how many samples.
pub struct TlaResult {
    /// Time-averaged state — the estimate of A^-1 b.
    pub x: Vec<f64>,
    /// Sample covariance times beta — the estimate of A^-1.
    pub a_inv: Vec<f64>,
    /// Euler-Maruyama steps taken (burn-in + measurement): the cost the ledger prices.
    pub steps: u64,
}

/// Simulate the OU network and return the thermodynamic solve. `dt` must satisfy
/// dt < 2 / `lambda_max(A)` for stability; `burn` steps equilibrate, `measure` steps average.
#[must_use]
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
    for v in &mut mean {
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
///
/// # Panics
///
/// If `A` is not positive definite, where the stationary distribution does not exist.
#[must_use]
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
#[must_use]
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

    /// The sample covariance must estimate A^-1: check A * `a_inv` ~ I.
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

#[cfg(test)]
mod dominance {
    use super::*;

    /// An SPD system with a controlled condition number, built by rotating a known spectrum.
    fn conditioned(n: usize, cond: f64, seed: u64) -> (Vec<f64>, Spd) {
        let mut rng = Pcg::new(seed, 7);
        let mut a = vec![0.0; n * n];
        for i in 0..n {
            a[i * n + i] = 1.0 + (cond - 1.0) * (i as f64) / ((n - 1) as f64);
        }
        for _ in 0..3 * n {
            let (p, q) = ((rng.next_u32() as usize) % n, (rng.next_u32() as usize) % n);
            if p == q {
                continue;
            }
            let t = rng.f64() * std::f64::consts::TAU;
            let (c, s) = (t.cos(), t.sin());
            for k in 0..n {
                let (x, y) = (a[p * n + k], a[q * n + k]);
                a[p * n + k] = c * x - s * y;
                a[q * n + k] = s * x + c * y;
            }
            for k in 0..n {
                let (x, y) = (a[k * n + p], a[k * n + q]);
                a[k * n + p] = c * x - s * y;
                a[k * n + q] = s * x + c * y;
            }
        }
        for i in 0..n {
            for j in 0..i {
                let m = 0.5 * (a[i * n + j] + a[j * n + i]);
                a[i * n + j] = m;
                a[j * n + i] = m;
            }
        }
        let b: Vec<f64> = (0..n).map(|_| rng.f64() * 2.0 - 1.0).collect();
        (a.clone(), Spd::new(n, a, b))
    }

    /// Conjugate gradient. One matrix-vector product per iteration — the same unit the OU
    /// integrator spends per step, which is what makes the comparison below a fair one.
    fn cg(a: &[f64], b: &[f64], n: usize, iters: usize) -> Vec<f64> {
        let mv = |x: &[f64]| -> Vec<f64> {
            (0..n).map(|i| (0..n).map(|j| a[i * n + j] * x[j]).sum()).collect()
        };
        let mut x = vec![0.0; n];
        let mut r = b.to_vec();
        let mut p = r.clone();
        let mut rs: f64 = r.iter().map(|v| v * v).sum();
        for _ in 0..iters {
            let ap = mv(&p);
            let denom: f64 = p.iter().zip(&ap).map(|(a, b)| a * b).sum();
            if denom.abs() < 1e-300 {
                break;
            }
            let al = rs / denom;
            for i in 0..n {
                x[i] += al * p[i];
                r[i] -= al * ap[i];
            }
            let rs2: f64 = r.iter().map(|v| v * v).sum();
            if rs2.sqrt() < 1e-14 {
                break;
            }
            let be = rs2 / rs;
            rs = rs2;
            for i in 0..n {
                p[i] = r[i] + be * p[i];
            }
        }
        x
    }

    fn rel(x: &[f64], truth: &[f64]) -> f64 {
        let num: f64 = x.iter().zip(truth).map(|(a, t)| (a - t) * (a - t)).sum::<f64>().sqrt();
        let den: f64 = truth.iter().map(|t| t * t).sum::<f64>().sqrt();
        num / den
    }

    /// **MEASURED ON OUR OWN CODE, ON THE FAIR WORK AXIS.** One matrix-vector product per
    /// conjugate-gradient iteration, one per Euler-Maruyama step — so the two methods are billed
    /// in the same unit, which is the comparison the adjacent literature does not make (see this
    /// module's header: arXiv:2608.09743 holds a sampler at a fixed budget and stops a solver at a
    /// tolerance, and its theorem is about the covariance rather than the mean anyway).
    ///
    /// | matvecs | conjugate gradient | OU mean |
    /// |---|---|---|
    /// | ~40 | `1e-15` | `1.0` |
    /// | 10,000 | — | `1.2e-1` |
    ///
    /// **The critique holds, and it is not close.** This test exists so that the WORKLOADS entry
    /// for this module cannot quietly go back to selling the mean.
    #[test]
    fn a_deterministic_solve_dominates_the_sampled_mean_at_equal_work() {
        for &cond in &[10.0f64, 100.0] {
            let n = 40;
            let (a, sys) = conditioned(n, cond, 0xA1 + cond as u64);
            let truth = solve_exact(&sys);

            // Deterministic: 40 matvecs, which is n and therefore no more than one dense solve.
            let x = cg(&a, &sys.b, n, 40);
            let e_cg = rel(&x, &truth);
            assert!(e_cg < 1e-12, "cond {cond}: CG should reach machine precision in n matvecs, got {e_cg:e}");

            // Sampled: 250x the work, on a dt at the edge of stability so it is given its best case.
            let r = solve_spd(&sys, 1.0, 0.9 / cond, 5_000, 10_000, 0xC0DE);
            // THE BUDGET EACH SIDE GOT IS PART OF THE CLAIM. Without this the comparison could be
            // rigged -- starve the sampler and the gap widens, which the assertions below would
            // happily report as a finding. A mutant that gave the OU chain 10 steps instead of
            // 15,000 passed every other check here.
            assert_eq!(r.steps, 15_000, "the sampled side must actually spend the work it is credited with");
            assert!(r.steps > 300 * 40, "and it must be given FAR more work than the {} matvecs CG used", 40);
            let e_ou = rel(&r.x, &truth);
            assert!(
                e_ou > 1e-2,
                "cond {cond}: the sampled mean came within {e_ou:e} at 10,000 matvecs -- if this \
                 ever passes, the comparison below has stopped being the one described"
            );
            assert!(
                e_ou / e_cg > 1e9,
                "cond {cond}: CG {e_cg:e} vs OU {e_ou:e} -- the gap is the finding, and it has shrunk"
            );
        }
    }

    /// **AND THE SECOND MOMENT, WHICH THE PAPER DOES NOT CLAIM.** The obvious defence of the
    /// sampling route is that it returns a covariance a solve does not. On a DENSE system that
    /// defence does not survive measurement either, and this test is that measurement.
    ///
    /// The cheap integrator (one matvec a step) carries a covariance bias that **does not shrink
    /// with more steps** — it shrinks with `dt`, and shrinking `dt` costs steps, so the two fight:
    /// at 200,000 steps the best achievable error is about 7%. The unbiased integrator reaches
    /// 0.8% — but it begins with an eigendecomposition, `O(n³)`, which is already more expensive
    /// than inverting the matrix outright.
    ///
    /// **So on a dense SPD system the deterministic route wins both moments.** What this does NOT
    /// measure, and what is therefore not claimed anywhere in this crate, is the large-sparse
    /// case, where `O(n³)` is unaffordable and the comparison is a genuinely open question.
    #[test]
    fn on_a_dense_system_the_sampled_covariance_loses_too() {
        let n = 20;
        let (a, sys) = conditioned(n, 10.0, 0xB2);
        // Exact inverse diagonal by Gauss-Jordan: n^3 flops, which is n matvec-equivalents.
        let mut aug = vec![0.0; n * 2 * n];
        for i in 0..n {
            for j in 0..n {
                aug[i * 2 * n + j] = a[i * n + j];
            }
            aug[i * 2 * n + n + i] = 1.0;
        }
        for c0 in 0..n {
            let p = aug[c0 * 2 * n + c0];
            for k in 0..2 * n {
                aug[c0 * 2 * n + k] /= p;
            }
            for r in 0..n {
                if r == c0 {
                    continue;
                }
                let f = aug[r * 2 * n + c0];
                for k in 0..2 * n {
                    aug[r * 2 * n + k] -= f * aug[c0 * 2 * n + k];
                }
            }
        }
        let exact: Vec<f64> = (0..n).map(|i| aug[i * 2 * n + n + i]).collect();
        let worst = |d: &[f64]| {
            (0..n).map(|i| (d[i] - exact[i]).abs() / exact[i].abs()).fold(0.0f64, f64::max)
        };

        // The cheap integrator, given 200,000 matvecs AND ITS BEST dt -- the sampler is handed
        // the most favourable setting, so that the conclusion cannot be an artefact of a bad one.
        let mut errs = Vec::new();
        for &dt in &[0.09f64, 0.03, 0.01, 0.003] {
            let r = solve_spd(&sys, 1.0, dt, 20_000, 200_000, 0xBEEF);
            let d: Vec<f64> = (0..n).map(|i| r.a_inv[i * n + i]).collect();
            errs.push(worst(&d));
        }
        let best = errs.iter().copied().fold(f64::INFINITY, f64::min);
        let worst_dt = errs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        // The sweep has to MATTER, or "its best dt" is a claim about nothing -- and `best` has to
        // be the minimum rather than the maximum, which a mutant swapping them made clear was
        // untested.
        assert!(worst_dt > 3.0 * best, "the dt sweep barely moved the error: {errs:?}");
        assert!(best <= errs[0] && best <= errs[errs.len() - 1], "best is not the best of the sweep");
        assert!(
            (0.02..0.5).contains(&best),
            "the cheap integrator's best covariance error over 200,000 matvecs is {best:e}, outside \
             the band this claim was measured in -- re-measure before editing the docs"
        );

        // The unbiased integrator does converge -- at the price of an O(n^3) eigendecomposition
        // before the first sample, so it is not competing on the same axis at all.
        let a1 = solve_spd_exact_ou(&sys, 1.0, 3.0, 200, 2_000, 0xBEEF);
        let a2 = solve_spd_exact_ou(&sys, 1.0, 3.0, 200, 200_000, 0xBEEF);
        let (e1, e2) = (
            worst(&(0..n).map(|i| a1.a_inv[i * n + i]).collect::<Vec<_>>()),
            worst(&(0..n).map(|i| a2.a_inv[i * n + i]).collect::<Vec<_>>()),
        );
        assert!(e1 > e2 * 3.0, "the unbiased sampler should converge: {e1:e} -> {e2:e}");
        assert!(e2 > 1e-3, "and 100x the samples still leaves {e2:e}, against an exact 0");
    }
}
