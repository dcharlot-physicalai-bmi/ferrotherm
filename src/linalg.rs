//! Minimal dense linear algebra used by the compiler and the thermodynamic linear-algebra
//! modules: a cyclic Jacobi eigensolver for symmetric matrices. std-only, deterministic.

/// Jacobi eigendecomposition of a symmetric n x n matrix (row-major, modified in place to its
/// diagonalized form). Returns the eigenvector matrix V flattened row-major: V[j * n + c] is
/// component j of eigenvector c; eigenvalue c ends up at position (c, c) of the input.
pub fn jacobi_eig(bm: &mut [f64], n: usize) -> Vec<f64> {
    let mut v = vec![0.0; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    // Both convergence thresholds are RELATIVE to the matrix, because absolute ones are not scale
    // invariant and this returns no convergence signal for a caller to check.
    //
    // They used to be the bare constants `1e-22` and `1e-16`. A well-conditioned SPD matrix scaled
    // down by 1e-13 has every off-diagonal below the second threshold, so no rotation is ever
    // applied and the function returns the UNTOUCHED diagonal as its eigenvalues, with the identity
    // as eigenvectors. Measured on [[3,1,0.5],[1,3,1],[0.5,1,3]]: correct at scale 1 and 1e-6,
    // wrong in the second digit at 1e-11, and exactly the input diagonal at 1e-13. Downstream,
    // `tla::solve_spd_exact_ou` checks only `lam.iter().all(|&l| l > 0.0)`, which those bogus
    // positive eigenvalues satisfy, so it returned a component with the wrong SIGN and no error.
    //
    // Normalising by the Frobenius norm makes the thresholds mean "small compared to this matrix",
    // which is what they were always meant to mean.
    let nrm2: f64 = bm.iter().map(|x| x * x).sum::<f64>().max(f64::MIN_POSITIVE);
    let off_eps = 1e-22 * nrm2;
    let piv_eps = 1e-16 * nrm2.sqrt();
    for _sweep in 0..100 {
        let mut off = 0.0;
        for p in 0..n {
            for q in (p + 1)..n {
                off += bm[p * n + q] * bm[p * n + q];
            }
        }
        if off < off_eps {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = bm[p * n + q];
                if apq.abs() < piv_eps {
                    continue;
                }
                let (app, aqq) = (bm[p * n + p], bm[q * n + q]);
                let theta = 0.5 * (aqq - app) / apq;
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..n {
                    let (akp, akq) = (bm[k * n + p], bm[k * n + q]);
                    bm[k * n + p] = c * akp - s * akq;
                    bm[k * n + q] = s * akp + c * akq;
                }
                for k in 0..n {
                    let (apk, aqk) = (bm[p * n + k], bm[q * n + k]);
                    bm[p * n + k] = c * apk - s * aqk;
                    bm[q * n + k] = s * apk + c * aqk;
                }
                for k in 0..n {
                    let (vkp, vkq) = (v[k * n + p], v[k * n + q]);
                    v[k * n + p] = c * vkp - s * vkq;
                    v[k * n + q] = s * vkp + c * vkq;
                }
            }
        }
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_eigensolver_is_scale_invariant() {
        // Both thresholds were absolute, so a well-conditioned SPD matrix scaled down far enough
        // had every off-diagonal below the pivot threshold, no rotation was ever applied, and this
        // returned the UNTOUCHED DIAGONAL as its eigenvalues with the identity as eigenvectors.
        // The only existing test used an O(1) matrix and an absolute 1e-9 tolerance, so it could
        // not see it. Downstream `tla::solve_spd_exact_ou` checks only that the eigenvalues are
        // positive, which the bogus ones are, and returned a component with the wrong sign.
        let base = [3.0f64, 1.0, 0.5, 1.0, 3.0, 1.0, 0.5, 1.0, 3.0];
        let want = [1.8138590, 2.5000000, 4.6861410];
        for scale in [1e13f64, 1e0, 1e-6, 1e-11, 1e-13] {
            let mut a: Vec<f64> = base.iter().map(|v| v * scale).collect();
            let _ = jacobi_eig(&mut a, 3);
            let mut got: Vec<f64> = (0..3).map(|i| a[i * 3 + i] / scale).collect();
            got.sort_by(|x, y| x.partial_cmp(y).unwrap());
            for (g, w) in got.iter().zip(want.iter()) {
                assert!(
                    (g - w).abs() < 1e-6,
                    "scale {scale:e}: eigenvalues {got:?} are not {want:?} -- \
                     a rescaled matrix has rescaled eigenvalues and nothing else changes"
                );
            }
        }
    }

    #[test]
    fn eigendecomposition_reconstructs() {
        let n = 4;
        let a = vec![
            4.0, 1.0, 0.5, 0.0,
            1.0, 3.0, 0.7, 0.2,
            0.5, 0.7, 3.5, 1.0,
            0.0, 0.2, 1.0, 2.5,
        ];
        let mut d = a.clone();
        let v = jacobi_eig(&mut d, n);
        // reconstruct A = V diag(lambda) V^T
        for i in 0..n {
            for j in 0..n {
                let mut s = 0.0;
                for c in 0..n {
                    s += v[i * n + c] * d[c * n + c] * v[j * n + c];
                }
                assert!((s - a[i * n + j]).abs() < 1e-9, "A[{i}{j}] {} vs {}", s, a[i * n + j]);
            }
        }
    }
}
