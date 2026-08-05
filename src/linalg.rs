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
    for _sweep in 0..100 {
        let mut off = 0.0;
        for p in 0..n {
            for q in (p + 1)..n {
                off += bm[p * n + q] * bm[p * n + q];
            }
        }
        if off < 1e-22 {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = bm[p * n + q];
                if apq.abs() < 1e-16 {
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
