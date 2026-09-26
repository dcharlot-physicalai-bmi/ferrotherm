//! Mpemba-optimised initialisation for thermodynamic linear algebra, and what the digital half of
//! that protocol costs.
//!
//! # The claim
//!
//! Moroder, Binder and Goold (arXiv:2603.24183, 2026) accelerate the covariance route to a matrix
//! inverse on an Ornstein–Uhlenbeck device: *"a classical digital processor efficiently computes an
//! initialization that suppresses slow relaxation modes, after which the physical system performs
//! the remaining computation through its intrinsic relaxation dynamics"*. The initialisation is
//! their Eq. 21, `Σ₀ = Σ_{k≤K} (k_BT / λ_k) u_k u_kᵀ` over the `K` smallest eigenpairs, found by
//! Lanczos, and they write that *"the classical preprocessing cost of the Lanczos algorithm is
//! negligible compared to the thermalization time across all matrix dimensions considered here"*.
//!
//! That sentence compares digital operations with device TIME, which share no unit. This module
//! compares like with like: the floating-point work the preprocessing costs, counted, against the
//! floating-point work of computing the whole inverse digitally, also counted. If the first is as
//! large as the second, the digital processor has spent what the entire answer costs before the
//! device starts, and the device cannot make the protocol cheaper than not using it.
//!
//! # What is exact here, and against what
//!
//! * [`ensemble`] builds the paper's first matrix family, a linearly spaced spectrum under a
//!   Haar-random orthogonal basis (their Eqs. 27–28), so every eigenpair is known BY CONSTRUCTION.
//!   [`lanczos_smallest`] is held to those, which it never sees.
//! * [`covariance_error`] is the relaxation of the covariance in closed form, mode by mode (their
//!   Eq. 19), and is held to a fourth-order Runge–Kutta integration of the Lyapunov equation in the
//!   physical basis (their Eq. 18), which never diagonalises anything.
//! * [`cholesky_inverse`] counts its own multiply-adds as it performs them, and its output is held
//!   to `A A⁻¹ = I`.
//!
//! Convention: mobility and `k_BT` are 1, as in the paper's figures, so the equilibrium covariance
//! is `A⁻¹` and mode `k` relaxes at rate `2 λ_k`.

use crate::rng::Pcg;

fn gauss(rng: &mut Pcg) -> f64 {
    let a = rng.f64().max(1e-300);
    let b = rng.f64();
    (-2.0 * a.ln()).sqrt() * (std::f64::consts::TAU * b).cos()
}

/// A random orthogonal matrix, row-major `n x n`, from Gram–Schmidt (applied twice, for numerical
/// orthogonality) on a Gaussian matrix. Its columns are Haar-distributed.
#[must_use]
pub fn haar_orthogonal(n: usize, seed: u64) -> Vec<f64> {
    let mut rng = Pcg::new(seed, 0x4AA2);
    // Columns as rows of `q` while building, transposed at the end.
    let mut q: Vec<Vec<f64>> = Vec::with_capacity(n);
    for _ in 0..n {
        let mut v: Vec<f64> = (0..n).map(|_| gauss(&mut rng)).collect();
        for _pass in 0..2 {
            for u in &q {
                let d: f64 = u.iter().zip(&v).map(|(a, b)| a * b).sum();
                for (x, y) in v.iter_mut().zip(u) {
                    *x -= d * y;
                }
            }
        }
        let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        v.iter_mut().for_each(|x| *x /= norm);
        q.push(v);
    }
    let mut out = vec![0.0; n * n];
    for (c, col) in q.iter().enumerate() {
        for (r, &x) in col.iter().enumerate() {
            out[r * n + c] = x;
        }
    }
    out
}

/// The paper's linearly spaced spectrum: `λ_k = lambda_min + k δ`, ascending.
#[must_use]
pub fn linear_spectrum(n: usize, lambda_min: f64, delta: f64) -> Vec<f64> {
    (0..n).map(|k| lambda_min + delta * k as f64).collect()
}

/// `A = U diag(eigenvalues) Uᵀ` for an orthogonal `U` (row-major, columns the eigenvectors).
#[must_use]
pub fn from_spectrum(u: &[f64], eigenvalues: &[f64]) -> Vec<f64> {
    let n = eigenvalues.len();
    let mut a = vec![0.0; n * n];
    for i in 0..n {
        for j in i..n {
            let s: f64 = (0..n).map(|k| u[i * n + k] * eigenvalues[k] * u[j * n + k]).sum();
            a[i * n + j] = s;
            a[j * n + i] = s;
        }
    }
    a
}

/// The paper's first ensemble: the spectrum of [`linear_spectrum`] under [`haar_orthogonal`].
/// Returns `(A, U, eigenvalues)`.
#[must_use]
pub fn ensemble(n: usize, lambda_min: f64, delta: f64, seed: u64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let u = haar_orthogonal(n, seed);
    let lam = linear_spectrum(n, lambda_min, delta);
    (from_spectrum(&u, &lam), u, lam)
}

/// The Frobenius distance of the covariance from its equilibrium `A⁻¹` at time `t`, after starting
/// from the Mpemba initialisation with `k` prethermalised modes (`k = 0` is the standard start,
/// `Σ₀ = 0`). Mode `i` relaxes as `σ_i(t) = 1/λ_i + (σ_i(0) − 1/λ_i) e^{−2 λ_i t}`, and the
/// initialisation sets `σ_i(0) = 1/λ_i` for the `k` smallest eigenvalues and `0` for the rest, so
///
/// ```text
///   E(t)² = Σ_{i > k} λ_i⁻² e^{−4 λ_i t}.
/// ```
///
/// `eigenvalues` ascending. At `t = 0` this is the initialisation's own error, the paper's `E₀(K)`
/// before normalisation.
#[must_use]
pub fn covariance_error(eigenvalues: &[f64], k: usize, t: f64) -> f64 {
    eigenvalues[k.min(eigenvalues.len())..]
        .iter()
        .map(|&l| (-4.0 * l * t).exp() / (l * l))
        .sum::<f64>()
        .sqrt()
}

/// The first time [`covariance_error`] falls to `eps`: the paper's `t₀(ε, K)` (their Eq. 24), by
/// bisection on a function that only decreases. `0` when the initialisation is already within `eps`.
#[must_use]
pub fn thermalization_time(eigenvalues: &[f64], k: usize, eps: f64) -> f64 {
    if covariance_error(eigenvalues, k, 0.0) <= eps {
        return 0.0;
    }
    let (mut lo, mut hi) = (0.0f64, 1.0f64);
    while covariance_error(eigenvalues, k, hi) > eps {
        hi *= 2.0;
    }
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if covariance_error(eigenvalues, k, mid) > eps {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    hi
}

/// The `k` smallest eigenpairs found by Lanczos, and the work it took.
#[derive(Clone, Debug)]
pub struct Lanczos {
    /// Ritz values, ascending.
    pub values: Vec<f64>,
    /// Ritz vectors, one per value, each of length `n`.
    pub vectors: Vec<Vec<f64>>,
    /// Lanczos steps taken, each one matrix–vector product.
    pub steps: usize,
    /// Floating-point operations, a multiply-add counted as two: the products, the full
    /// reorthogonalisation, and forming the Ritz vectors. The tridiagonal work is `O(steps²)` and
    /// is left out, which only flatters Lanczos.
    pub flops: u64,
    /// Whether every Ritz residual `|β_m s_m|` reached `tol · ‖A‖` before `max_steps`.
    pub converged: bool,
}

/// Number of eigenvalues of the symmetric tridiagonal `(alpha, beta)` below `x`: the Sturm count,
/// by the signs of the `LDLᵀ` pivots of `T − x I`.
fn sturm_below(alpha: &[f64], beta: &[f64], x: f64) -> usize {
    let mut count = 0;
    let mut d = 1.0f64;
    for i in 0..alpha.len() {
        let off = if i == 0 { 0.0 } else { beta[i - 1] * beta[i - 1] / d };
        d = alpha[i] - x - off;
        if d == 0.0 {
            d = -f64::EPSILON * (alpha[i].abs() + x.abs()).max(1.0);
        }
        if d < 0.0 {
            count += 1;
        }
    }
    count
}

/// The `j`-th smallest eigenvalue (0-based) of the tridiagonal, by bisection on the Sturm count.
fn tridiagonal_eigenvalue(alpha: &[f64], beta: &[f64], j: usize) -> f64 {
    let m = alpha.len();
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    for i in 0..m {
        let r = if i > 0 { beta[i - 1].abs() } else { 0.0 } + if i + 1 < m { beta[i].abs() } else { 0.0 };
        lo = lo.min(alpha[i] - r);
        hi = hi.max(alpha[i] + r);
    }
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if sturm_below(alpha, beta, mid) > j {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Solve `(T − s I) x = rhs` for the symmetric tridiagonal `(alpha, beta)`, by Gaussian elimination
/// with partial pivoting (the fill is one extra superdiagonal), as LAPACK's `dgtsv` does.
fn tridiagonal_solve(alpha: &[f64], beta: &[f64], s: f64, rhs: &[f64]) -> Vec<f64> {
    let m = alpha.len();
    let mut dl: Vec<f64> = beta.to_vec();
    let mut d: Vec<f64> = alpha.iter().map(|a| a - s).collect();
    let mut du: Vec<f64> = beta.to_vec();
    let mut du2 = vec![0.0; m.saturating_sub(2)];
    let mut b = rhs.to_vec();
    for i in 0..m.saturating_sub(1) {
        if d[i].abs() >= dl[i].abs() {
            if d[i] == 0.0 {
                d[i] = f64::EPSILON;
            }
            let f = dl[i] / d[i];
            d[i + 1] -= f * du[i];
            b[i + 1] -= f * b[i];
            dl[i] = 0.0;
        } else {
            let f = d[i] / dl[i];
            d[i] = dl[i];
            let t = d[i + 1];
            d[i + 1] = du[i] - f * t;
            if i + 2 < m {
                du2[i] = du[i + 1];
                du[i + 1] = -f * du2[i];
            }
            du[i] = t;
            b.swap(i, i + 1);
            b[i + 1] -= f * b[i];
        }
    }
    if d[m - 1] == 0.0 {
        d[m - 1] = f64::EPSILON;
    }
    let mut x = vec![0.0; m];
    for i in (0..m).rev() {
        let mut v = b[i];
        if i + 1 < m {
            v -= du[i] * x[i + 1];
        }
        if i + 2 < m {
            v -= du2[i] * x[i + 2];
        }
        x[i] = v / d[i];
    }
    x
}

/// Eigenvector of the tridiagonal for eigenvalue `theta`, by two steps of inverse iteration.
fn tridiagonal_eigenvector(alpha: &[f64], beta: &[f64], theta: f64) -> Vec<f64> {
    let m = alpha.len();
    let shift = theta + 1e-12 * theta.abs().max(1.0);
    let mut y: Vec<f64> = (0..m).map(|i| 1.0 + 0.01 * i as f64).collect();
    for _ in 0..3 {
        y = tridiagonal_solve(alpha, beta, shift, &y);
        let n = y.iter().map(|v| v * v).sum::<f64>().sqrt();
        y.iter_mut().for_each(|v| *v /= n);
    }
    y
}

/// The `k` smallest eigenpairs of the symmetric `n x n` matrix `a` (row-major) by Lanczos with full
/// reorthogonalisation, stopping when every Ritz residual `|β_m s_{m,i}|` is at most `tol · ‖A‖`
/// (`‖A‖` estimated by the largest Ritz value's magnitude). The start vector is Gaussian from
/// `seed`. Every product with `a` and every reorthogonalisation is counted in [`Lanczos::flops`].
///
/// # Panics
///
/// If `a` is not `n x n` or `k` is zero or larger than `n`.
#[must_use]
pub fn lanczos_smallest(a: &[f64], n: usize, k: usize, tol: f64, max_steps: usize, seed: u64) -> Lanczos {
    assert_eq!(a.len(), n * n, "a is n x n");
    assert!(k >= 1 && k <= n, "k must lie in 1..=n");
    let mut rng = Pcg::new(seed, 0x1A2C);
    let mut basis: Vec<Vec<f64>> = Vec::new();
    let mut v: Vec<f64> = (0..n).map(|_| gauss(&mut rng)).collect();
    let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    v.iter_mut().for_each(|x| *x /= norm);
    let (mut alpha, mut beta) = (Vec::new(), Vec::new());
    let mut flops: u64 = 0;
    let steps_cap = max_steps.min(n);
    let mut converged = false;
    let mut theta = Vec::new();
    for _ in 0..steps_cap {
        let mut w = vec![0.0; n];
        for (r, wr) in w.iter_mut().enumerate() {
            *wr = a[r * n..(r + 1) * n].iter().zip(&v).map(|(x, y)| x * y).sum();
        }
        flops += 2 * (n * n) as u64;
        let aj: f64 = w.iter().zip(&v).map(|(x, y)| x * y).sum();
        alpha.push(aj);
        basis.push(v.clone());
        // Full reorthogonalisation against every basis vector, twice: 2 passes x (dot + axpy).
        for _pass in 0..2 {
            for q in &basis {
                let d: f64 = q.iter().zip(&w).map(|(x, y)| x * y).sum();
                for (x, y) in w.iter_mut().zip(q) {
                    *x -= d * y;
                }
            }
            flops += 4 * (n * basis.len()) as u64;
        }
        let bj = w.iter().map(|x| x * x).sum::<f64>().sqrt();
        if basis.len() >= k {
            theta = (0..k).map(|i| tridiagonal_eigenvalue(&alpha, &beta, i)).collect::<Vec<_>>();
            let scale = tridiagonal_eigenvalue(&alpha, &beta, alpha.len() - 1).abs().max(theta[0].abs());
            let ok = theta.iter().all(|&t| {
                let s = tridiagonal_eigenvector(&alpha, &beta, t);
                (bj * s[s.len() - 1]).abs() <= tol * scale
            });
            if ok || bj == 0.0 {
                converged = true;
                break;
            }
        }
        beta.push(bj);
        v = w.iter().map(|x| x / bj).collect();
    }
    // Ritz vectors: y_i = V s_i.
    let m = alpha.len();
    let vectors: Vec<Vec<f64>> = theta
        .iter()
        .map(|&t| {
            let s = tridiagonal_eigenvector(&alpha, &beta[..m - 1], t);
            let mut y = vec![0.0; n];
            for (q, &c) in basis.iter().zip(&s) {
                for (x, b) in y.iter_mut().zip(q) {
                    *x += c * b;
                }
            }
            y
        })
        .collect();
    flops += 2 * (m * n * k) as u64;
    Lanczos { values: theta, vectors, steps: m, flops, converged }
}

/// The inverse of a symmetric positive-definite matrix by Cholesky, `A = L Lᵀ`, then `L⁻¹`, then
/// `A⁻¹ = L⁻ᵀ L⁻¹`, counting every multiply-add as two flops as it is performed. Returns
/// `(A⁻¹, flops)`, or `None` if a pivot is not positive.
#[must_use]
pub fn cholesky_inverse(a: &[f64], n: usize) -> Option<(Vec<f64>, u64)> {
    let mut l = vec![0.0; n * n];
    let mut madds: u64 = 0;
    for j in 0..n {
        let mut s = a[j * n + j];
        for k in 0..j {
            s -= l[j * n + k] * l[j * n + k];
        }
        madds += j as u64;
        if s <= 0.0 {
            return None;
        }
        let d = s.sqrt();
        l[j * n + j] = d;
        for i in j + 1..n {
            let mut t = a[i * n + j];
            for k in 0..j {
                t -= l[i * n + k] * l[j * n + k];
            }
            madds += j as u64;
            l[i * n + j] = t / d;
        }
    }
    // L⁻¹, lower triangular, column by column.
    let mut li = vec![0.0; n * n];
    for c in 0..n {
        li[c * n + c] = 1.0 / l[c * n + c];
        for i in c + 1..n {
            let mut t = 0.0;
            for k in c..i {
                t += l[i * n + k] * li[k * n + c];
            }
            madds += (i - c) as u64;
            li[i * n + c] = -t / l[i * n + i];
        }
    }
    // A⁻¹ = L⁻ᵀ L⁻¹, symmetric: fill the upper triangle and mirror.
    let mut inv = vec![0.0; n * n];
    for i in 0..n {
        for j in i..n {
            let mut t = 0.0;
            for k in j..n {
                t += li[k * n + i] * li[k * n + j];
            }
            madds += (n - j) as u64;
            inv[i * n + j] = t;
            inv[j * n + i] = t;
        }
    }
    Some((inv, 2 * madds))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
    }

    /// The ensemble is what it says: `U` orthogonal to rounding, and `A U = U Λ` column by column,
    /// so the eigenpairs below are known by CONSTRUCTION, independently of any solver.
    #[test]
    fn the_ensemble_has_the_spectrum_it_was_built_with() {
        let n = 40;
        let (a, u, lam) = ensemble(n, 0.5, 0.5, 3);
        let mut worst_orth = 0.0f64;
        for i in 0..n {
            for j in 0..n {
                let d: f64 = (0..n).map(|k| u[k * n + i] * u[k * n + j]).sum();
                worst_orth = worst_orth.max((d - if i == j { 1.0 } else { 0.0 }).abs());
            }
        }
        assert!(worst_orth < 1e-12, "U must be orthogonal: {worst_orth:e}");
        for c in 0..n {
            for r in 0..n {
                let au: f64 = (0..n).map(|k| a[r * n + k] * u[k * n + c]).sum();
                assert!((au - lam[c] * u[r * n + c]).abs() < 1e-11, "column {c} is not an eigenvector");
            }
        }
    }

    /// Lanczos recovers the `k` smallest eigenvalues it was never shown, and eigenvectors parallel to
    /// the constructed ones; the residual stopping rule is honest (`converged` and the error agree).
    #[test]
    fn lanczos_finds_the_smallest_eigenpairs_the_ensemble_was_built_with() {
        let n = 120;
        let (a, u, lam) = ensemble(n, 0.5, 0.5, 9);
        let r = lanczos_smallest(&a, n, 10, 1e-10, n, 5);
        assert!(r.converged, "must converge within n steps");
        assert!(r.steps < n, "and before exhausting the space: {} steps", r.steps);
        for i in 0..10 {
            assert!((r.values[i] - lam[i]).abs() < 1e-8 * lam[n - 1], "eigenvalue {i}: {} vs {}", r.values[i], lam[i]);
            let overlap: f64 = (0..n).map(|k| r.vectors[i][k] * u[k * n + i]).sum::<f64>().abs();
            assert!((overlap - 1.0).abs() < 1e-6, "eigenvector {i}: |overlap| {overlap}");
        }
        // A cruder tolerance takes fewer steps: the rule is doing something.
        let coarse = lanczos_smallest(&a, n, 10, 1e-3, n, 5);
        assert!(coarse.steps < r.steps, "coarse {} vs fine {}", coarse.steps, r.steps);
    }

    /// The Cholesky inverse is an inverse, and the flops it counts are the textbook `n³` to within
    /// the lower-order terms (they are counted, not assumed).
    #[test]
    fn the_cholesky_inverse_inverts_and_counts_its_work() {
        let n = 60;
        let (a, _, _) = ensemble(n, 0.5, 0.5, 4);
        let (inv, flops) = cholesky_inverse(&a, n).expect("SPD");
        let mut worst = 0.0f64;
        for i in 0..n {
            for j in 0..n {
                let s: f64 = (0..n).map(|k| a[i * n + k] * inv[k * n + j]).sum();
                worst = worst.max((s - if i == j { 1.0 } else { 0.0 }).abs());
            }
        }
        assert!(worst < 1e-9, "A A^-1 = I: {worst:e}");
        let ratio = flops as f64 / (n as f64).powi(3);
        assert!((0.95..1.1).contains(&ratio), "flops / n^3 = {ratio}");
        assert!(cholesky_inverse(&[1.0, 2.0, 2.0, 1.0], 2).is_none(), "an indefinite matrix is refused");
    }

    /// The closed form is held to the paper's Lyapunov equation, `dΣ/dt = −(AΣ + ΣA) + 2I`,
    /// integrated by RK4 in the PHYSICAL basis from the Mpemba start built out of the constructed
    /// eigenvectors: no diagonalisation on this side. Also `E(0)` is the untouched modes' part of
    /// `A⁻¹`, the paper's `E₀(K)` before normalisation.
    #[test]
    fn the_closed_form_relaxation_is_the_lyapunov_equations() {
        let n = 12;
        let (a, u, lam) = ensemble(n, 0.5, 0.5, 21);
        for k in [0usize, 3] {
            let mut sigma = vec![0.0; n * n];
            for m in 0..k {
                for i in 0..n {
                    for j in 0..n {
                        sigma[i * n + j] += u[i * n + m] * u[j * n + m] / lam[m];
                    }
                }
            }
            let (inv, _) = cholesky_inverse(&a, n).expect("SPD");
            let e0: f64 = sigma.iter().zip(&inv).map(|(s, v)| (s - v).powi(2)).sum::<f64>().sqrt();
            assert!((e0 - covariance_error(&lam, k, 0.0)).abs() < 1e-12, "E(0) at k = {k}");
            let rhs = |s: &[f64]| -> Vec<f64> {
                let mut out = vec![0.0; n * n];
                for i in 0..n {
                    for j in 0..n {
                        let mut v = if i == j { 2.0 } else { 0.0 };
                        for m in 0..n {
                            v -= a[i * n + m] * s[m * n + j] + s[i * n + m] * a[m * n + j];
                        }
                        out[i * n + j] = v;
                    }
                }
                out
            };
            let (dt, steps) = (1e-3, 800);
            for _ in 0..steps {
                let k1 = rhs(&sigma);
                let s2: Vec<f64> = sigma.iter().zip(&k1).map(|(s, d)| s + 0.5 * dt * d).collect();
                let k2 = rhs(&s2);
                let s3: Vec<f64> = sigma.iter().zip(&k2).map(|(s, d)| s + 0.5 * dt * d).collect();
                let k3 = rhs(&s3);
                let s4: Vec<f64> = sigma.iter().zip(&k3).map(|(s, d)| s + dt * d).collect();
                let k4 = rhs(&s4);
                for idx in 0..n * n {
                    sigma[idx] += dt / 6.0 * (k1[idx] + 2.0 * k2[idx] + 2.0 * k3[idx] + k4[idx]);
                }
            }
            let t = dt * steps as f64;
            let integrated: f64 = sigma.iter().zip(&inv).map(|(s, v)| (s - v).powi(2)).sum::<f64>().sqrt();
            let closed = covariance_error(&lam, k, t);
            assert!(
                (integrated - closed).abs() < 1e-9 * closed.max(1e-12),
                "k = {k}, t = {t}: integrated {integrated} vs closed {closed}"
            );
            assert!(max_abs_diff(&sigma, &inv) < 1.0, "sanity: the covariance approaches A^-1");
        }
    }

    /// The paper's asymptotic speedup, their Eq. 26: `t₀(ε, 0) / t₀(ε, K) → λ_{K+1} / λ_1` as
    /// `ε → 0`. Measured here it approaches that limit from ABOVE, and slowly: on the paper's
    /// spectrum with `K = 10` (limit 11) it is `18.09` at `ε = 1e-2`, `14.29` at `1e-4`, and still
    /// 6.8% over at `1e-16`, because the Mpemba start is already within `0.60` of `A⁻¹`. At any
    /// tolerance a solver would use, the speedup is LARGER than the formula the paper plots it
    /// against.
    #[test]
    fn the_speedup_tends_to_the_spectral_ratio_from_above() {
        let lam = linear_spectrum(200, 0.5, 0.5);
        let k = 10;
        let limit = lam[k] / lam[0];
        let mut last = f64::INFINITY;
        for eps in [1e-2f64, 1e-4, 1e-8, 1e-16] {
            let s = thermalization_time(&lam, 0, eps) / thermalization_time(&lam, k, eps);
            assert!(s < last, "the speedup falls as the tolerance tightens: {s} after {last}");
            assert!(s > limit, "and stays above its limit {limit}: {s}");
            last = s;
        }
        assert!((last / limit - 1.0) < 0.07, "at eps 1e-16 it is within 7% of the limit: {last} vs {limit}");
        let at_1e2 = thermalization_time(&lam, 0, 1e-2) / thermalization_time(&lam, k, 1e-2);
        assert!((at_1e2 - 18.09).abs() < 0.01, "measured 18.09 at eps 1e-2: {at_1e2}");
        // An initialisation already inside the tolerance needs no device at all.
        assert_eq!(thermalization_time(&lam, k, covariance_error(&lam, k, 0.0) * 1.01), 0.0);
    }

    /// **The preprocessing the paper calls negligible costs about what the whole answer costs.**
    /// On the paper's first ensemble (`δ = 0.5`, `K = 10`), Lanczos for the `K` smallest eigenpairs
    /// to a residual of `1e-8 ‖A‖` is held against a counted Cholesky inverse of the same matrix.
    /// Measured by `examples/mpemba_exact.rs` at a residual of `1e-8`: `2.91x` the inverse's flops at
    /// `n = 200`, `1.48x` at the paper's own `n = 500`, and on its Wishart ensemble `3.7x` to `6.2x`
    /// up to `n = 1000`. Asserted here at `n = 150`, where it must exceed one. The count includes both
    /// reorthogonalisation passes; a single pass would take about a quarter off, and at `n = 500`
    /// that still leaves it near one.
    #[test]
    fn the_digital_preprocessing_costs_as_much_as_the_whole_digital_inverse() {
        let n = 150;
        let (a, _, lam) = ensemble(n, 0.5, 0.5, 3);
        let r = lanczos_smallest(&a, n, 10, 1e-8, n, 1);
        assert!(r.converged);
        for i in 0..10 {
            assert!((r.values[i] - lam[i]).abs() < 1e-6 * lam[n - 1]);
        }
        let (_, chol) = cholesky_inverse(&a, n).expect("SPD");
        let ratio = r.flops as f64 / chol as f64;
        assert!(ratio > 1.0, "Lanczos {} flops vs the whole inverse {} = {ratio}", r.flops, chol);
    }
}
