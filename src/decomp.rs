//! Matrix and tensor decompositions as public, tested primitives: the singular value
//! decomposition with the Eckart--Young truncation bound, the higher-order SVD (Tucker) of a
//! dense tensor, and the randomized range finder that makes a large SVD affordable.
//!
//! # Why this exists
//!
//! [`crate::mps`] carries a private SVD for its bond truncations, built on
//! [`crate::linalg::jacobi_eig`] through the Gram matrix; it is adequate for that job and
//! reachable from nowhere else. A crate that certifies its samplers against exact oracles has three other
//! places a decomposition belongs -- compressing a coupling matrix to a low-rank model, reading
//! the factor structure of a sampled tensor of moments, and shrinking a dense linear-algebra
//! workload before a thermodynamic solver sees it -- and every one of them was reimplementing or
//! doing without. This module is that primitive, in the open, with the oracles that keep it
//! honest.
//!
//! # What is here, and what each is checked against
//!
//! * [`svd`]: `A = U diag(s) V^T` for a dense row-major `A`, `s` descending, `U` and `V^T` with
//!   orthonormal columns and rows, by one-sided Jacobi (Hestenes 1958) -- not the Gram route
//!   `mps` takes, which squares the condition number and loses every singular value below about
//!   `1e-8` of the largest. Here every singular value has high relative accuracy, and
//!   [`Svd::rank`] counts those above `1e-13` of the largest. Checked by reconstruction to
//!   `1e-10`, by orthonormality, and against a matrix whose singular values are known by
//!   construction.
//! * [`Svd::truncate`] and [`Svd::truncation_error`]: the best rank-`k` approximation and its
//!   Frobenius error `sqrt(sum_{i > k} s_i^2)` (Eckart and Young 1936; Mirsky 1960 for every
//!   unitarily invariant norm). The test reconstructs the truncation, measures its error against
//!   the closed form, and confirms that another rank-`k` matrix -- the truncation with its basis
//!   rotated -- does worse.
//! * [`hosvd`]: the Tucker decomposition of a dense tensor by the higher-order SVD (De Lathauwer,
//!   De Moor and Vandewalle 2000): factor `n` is the left singular basis of the mode-`n`
//!   unfolding, the core is the tensor contracted with every factor's transpose. Exact at full
//!   ranks, and the core is all-orthogonal (its mode-`n` slices are mutually orthogonal), both
//!   checked.
//! * [`randomized_svd`]: the range finder of Halko, Martinsson and Tropp (2011): sketch `Y = A
//!   Omega` with a Gaussian `Omega` of `k + p` columns, orthonormalise, project `B = Q^T A`,
//!   take the small SVD, lift `U = Q U_B`. With `q` power iterations the sketch is of
//!   `(A A^T)^q A`, which sharpens a slow spectrum. Checked against the full SVD on a matrix
//!   with a decaying spectrum: the top singular values to `1e-6` with one power iteration.


/// A singular value decomposition `A = U diag(s) V^T`.
#[derive(Clone, Debug)]
pub struct Svd {
    /// Rows of `A`.
    pub rows: usize,
    /// Columns of `A`.
    pub cols: usize,
    /// `U`, `rows x k` row-major, `k = min(rows, cols)`, orthonormal columns up to `rank`.
    pub u: Vec<f64>,
    /// Singular values, descending, length `k`.
    pub s: Vec<f64>,
    /// `V^T`, `k x cols` row-major, orthonormal rows up to `rank`.
    pub vt: Vec<f64>,
    /// Singular triplets the decomposition could resolve; the rest are zero to working precision.
    pub rank: usize,
}

/// Gram--Schmidt on the first `k` rows of a `k x cols` matrix, in place, twice for stability;
/// a row whose remainder is below `1e-10` is zeroed. Returns how many rows survived.
fn orthonormalise_rows(x: &mut [f64], k: usize, cols: usize) -> usize {
    let mut rank = 0;
    for r in 0..k {
        for _ in 0..2 {
            for p in 0..r {
                let dot: f64 = (0..cols).map(|c| x[r * cols + c] * x[p * cols + c]).sum();
                for c in 0..cols {
                    x[r * cols + c] -= dot * x[p * cols + c];
                }
            }
        }
        let norm: f64 = (0..cols)
            .map(|c| x[r * cols + c] * x[r * cols + c])
            .sum::<f64>()
            .sqrt();
        if norm < 1e-10 {
            for c in 0..cols {
                x[r * cols + c] = 0.0;
            }
        } else {
            for c in 0..cols {
                x[r * cols + c] /= norm;
            }
            rank += 1;
        }
    }
    rank
}

/// The singular value decomposition of a dense row-major `rows x cols` matrix by one-sided
/// Jacobi (Hestenes 1958): plane rotations orthogonalise the columns of the tall orientation
/// until every pair is orthogonal to working precision, the singular values are the column norms,
/// `V` is the product of the rotations and `U` the normalised columns. Every singular value comes
/// out with high relative accuracy, which the Gram route cannot give.
///
/// # Panics
///
/// If `a` does not hold `rows * cols` entries.
#[must_use]
pub fn svd(a: &[f64], rows: usize, cols: usize) -> Svd {
    assert_eq!(
        a.len(),
        rows * cols,
        "a {rows} x {cols} matrix holds {} entries",
        rows * cols
    );
    let k = rows.min(cols);
    if k == 0 {
        return Svd {
            rows,
            cols,
            u: Vec::new(),
            s: Vec::new(),
            vt: Vec::new(),
            rank: 0,
        };
    }
    // The tall orientation: m x n with m >= n, so the rotations act on at most min(rows, cols)
    // columns. A wide matrix is decomposed transposed and the factors swapped back.
    let transposed = rows < cols;
    let (m, n) = if transposed {
        (cols, rows)
    } else {
        (rows, cols)
    };
    let mut w = vec![0.0; m * n];
    for r in 0..rows {
        for c in 0..cols {
            if transposed {
                w[c * n + r] = a[r * cols + c];
            } else {
                w[r * n + c] = a[r * cols + c];
            }
        }
    }
    let mut v = vec![0.0; n * n];
    for j in 0..n {
        v[j * n + j] = 1.0;
    }
    for _sweep in 0..60 {
        let mut rotated = false;
        for p in 0..n {
            for q in p + 1..n {
                let mut alpha = 0.0;
                let mut beta = 0.0;
                let mut gamma = 0.0;
                for i in 0..m {
                    let wp = w[i * n + p];
                    let wq = w[i * n + q];
                    alpha += wp * wp;
                    beta += wq * wq;
                    gamma += wp * wq;
                }
                if gamma == 0.0 || gamma.abs() <= 1e-15 * (alpha * beta).sqrt() {
                    continue;
                }
                rotated = true;
                let zeta = (beta - alpha) / (2.0 * gamma);
                let t = zeta.signum() / (zeta.abs() + (1.0 + zeta * zeta).sqrt());
                let c = 1.0 / (1.0 + t * t).sqrt();
                let sn = c * t;
                for i in 0..m {
                    let wp = w[i * n + p];
                    let wq = w[i * n + q];
                    w[i * n + p] = c * wp - sn * wq;
                    w[i * n + q] = sn * wp + c * wq;
                }
                for i in 0..n {
                    let vp = v[i * n + p];
                    let vq = v[i * n + q];
                    v[i * n + p] = c * vp - sn * vq;
                    v[i * n + q] = sn * vp + c * vq;
                }
            }
        }
        if !rotated {
            break;
        }
    }
    let norms: Vec<f64> = (0..n)
        .map(|j| {
            (0..m)
                .map(|i| w[i * n + j] * w[i * n + j])
                .sum::<f64>()
                .sqrt()
        })
        .collect();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&x, &y| norms[y].total_cmp(&norms[x]));
    let top = norms[order[0]];
    let mut s = vec![0.0; n];
    // Left factor of the tall orientation, m x n; right factor transposed, n x n.
    let mut left = vec![0.0; m * n];
    let mut right_t = vec![0.0; n * n];
    let mut rank = 0;
    for (t, &j) in order.iter().enumerate() {
        let resolved = norms[j] > 1e-13 * top && norms[j] > 0.0;
        if resolved {
            rank += 1;
            s[t] = norms[j];
            for i in 0..m {
                left[i * n + t] = w[i * n + j] / norms[j];
            }
        }
        for i in 0..n {
            right_t[t * n + i] = v[i * n + j];
        }
    }
    if transposed {
        // A = W^T = (L S R^T)^T = R S L^T: U is R (rows x n), V^T is L^T (n x cols).
        let mut u = vec![0.0; rows * n];
        for r in 0..rows {
            for t in 0..n {
                u[r * n + t] = right_t[t * n + r];
            }
        }
        let mut vt = vec![0.0; n * cols];
        for t in 0..n {
            for c in 0..cols {
                vt[t * cols + c] = left[c * n + t];
            }
        }
        Svd {
            rows,
            cols,
            u,
            s,
            vt,
            rank,
        }
    } else {
        Svd {
            rows,
            cols,
            u: left,
            s,
            vt: right_t,
            rank,
        }
    }
}

impl Svd {
    /// `U_k diag(s_k) V_k^T`: the best rank-`k` approximation in every unitarily invariant norm,
    /// `rows x cols` row-major. `k` above the stored count is clamped.
    #[must_use]
    pub fn truncate(&self, k: usize) -> Vec<f64> {
        let kk = k.min(self.s.len());
        let width = self.s.len();
        let mut out = vec![0.0; self.rows * self.cols];
        for r in 0..self.rows {
            for c in 0..self.cols {
                out[r * self.cols + c] = (0..kk)
                    .map(|t| self.u[r * width + t] * self.s[t] * self.vt[t * self.cols + c])
                    .sum();
            }
        }
        out
    }

    /// The Frobenius error of [`Svd::truncate`] at rank `k`, `sqrt(sum_{i > k} s_i^2)`, without
    /// forming it: Eckart--Young.
    #[must_use]
    pub fn truncation_error(&self, k: usize) -> f64 {
        self.s.iter().skip(k).map(|v| v * v).sum::<f64>().sqrt()
    }

    /// The smallest rank whose truncation error is at most `tolerance` in Frobenius norm.
    #[must_use]
    pub fn rank_for(&self, tolerance: f64) -> usize {
        (0..=self.s.len())
            .find(|&k| self.truncation_error(k) <= tolerance)
            .unwrap_or(self.s.len())
    }
}

/// Frobenius norm of the difference of two equally long vectors.
#[must_use]
pub fn frobenius_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt()
}

/// A dense tensor in row-major (last index fastest) layout.
#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    /// Extent of each mode.
    pub dims: Vec<usize>,
    /// The entries, `dims.iter().product()` of them.
    pub data: Vec<f64>,
}

impl Tensor {
    /// A tensor of the given extents holding `data`.
    ///
    /// # Panics
    ///
    /// If `data` does not hold the product of the extents.
    #[must_use]
    pub fn new(dims: Vec<usize>, data: Vec<f64>) -> Tensor {
        let total: usize = dims.iter().product();
        assert_eq!(
            data.len(),
            total,
            "a tensor of extents {dims:?} holds {total} entries"
        );
        Tensor { dims, data }
    }

    /// The stride of each mode in row-major layout.
    fn strides(&self) -> Vec<usize> {
        let mut strides = vec![1usize; self.dims.len()];
        for n in (0..self.dims.len().saturating_sub(1)).rev() {
            strides[n] = strides[n + 1] * self.dims[n + 1];
        }
        strides
    }

    /// The mode-`n` unfolding: a `dims[n] x (product of the others)` matrix whose row `i` holds
    /// every entry with index `i` in mode `n`, the other modes in row-major order.
    #[must_use]
    pub fn unfold(&self, n: usize) -> Vec<f64> {
        let rows = self.dims[n];
        let cols = self.data.len() / rows;
        let strides = self.strides();
        let mut out = vec![0.0; rows * cols];
        for (flat, &v) in self.data.iter().enumerate() {
            let i = (flat / strides[n]) % rows;
            // Column: the flat index with mode n removed.
            let mut col = 0;
            let mut mult = 1;
            for m in (0..self.dims.len()).rev() {
                if m == n {
                    continue;
                }
                let idx = (flat / strides[m]) % self.dims[m];
                col += idx * mult;
                mult *= self.dims[m];
            }
            out[i * cols + col] = v;
        }
        out
    }

    /// The mode-`n` product with a `p x dims[n]` matrix: mode `n` becomes `p` long, each fibre
    /// along it multiplied by the matrix.
    ///
    /// # Panics
    ///
    /// If the matrix does not hold `p * dims[n]` entries.
    #[must_use]
    pub fn mode_product(&self, n: usize, matrix: &[f64], p: usize) -> Tensor {
        let dn = self.dims[n];
        assert_eq!(matrix.len(), p * dn, "a {p} x {dn} matrix for mode {n}");
        let mut dims = self.dims.clone();
        dims[n] = p;
        let strides_in = self.strides();
        let out_total: usize = dims.iter().product();
        let mut out = Tensor {
            dims,
            data: vec![0.0; out_total],
        };
        let strides_out = out.strides();
        for (flat, &v) in self.data.iter().enumerate() {
            let i = (flat / strides_in[n]) % dn;
            let base = flat - i * strides_in[n];
            // The same multi-index with mode n at zero, in the output's strides.
            let mut out_base = 0;
            for m in 0..self.dims.len() {
                if m == n {
                    continue;
                }
                let idx = (base / strides_in[m]) % self.dims[m];
                out_base += idx * strides_out[m];
            }
            for j in 0..p {
                out.data[out_base + j * strides_out[n]] += matrix[j * dn + i] * v;
            }
        }
        out
    }
}

/// A Tucker decomposition `X = G x_1 U_1 x_2 U_2 ... x_N U_N`.
#[derive(Clone, Debug)]
pub struct Tucker {
    /// The core tensor, extents `ranks`.
    pub core: Tensor,
    /// One factor per mode, `dims[n] x ranks[n]` row-major with orthonormal columns.
    pub factors: Vec<Vec<f64>>,
    /// The multilinear ranks kept.
    pub ranks: Vec<usize>,
}

/// The higher-order SVD of `x` truncated to `ranks` (each clamped to its mode's extent).
///
/// # Panics
///
/// If `ranks` has a different length from the tensor's modes.
#[must_use]
pub fn hosvd(x: &Tensor, ranks: &[usize]) -> Tucker {
    assert_eq!(ranks.len(), x.dims.len(), "one rank per mode");
    let mut factors = Vec::with_capacity(x.dims.len());
    let mut kept = Vec::with_capacity(x.dims.len());
    let mut core = x.clone();
    for n in 0..x.dims.len() {
        let unfolded = x.unfold(n);
        let rows = x.dims[n];
        let cols = unfolded.len() / rows;
        let dec = svd(&unfolded, rows, cols);
        let r = ranks[n].min(dec.s.len());
        let width = dec.s.len();
        let mut factor = vec![0.0; rows * r];
        for i in 0..rows {
            for t in 0..r {
                factor[i * r + t] = dec.u[i * width + t];
            }
        }
        // U^T is r x rows: row t holds column t of U.
        let mut ut = vec![0.0; r * rows];
        for t in 0..r {
            for i in 0..rows {
                ut[t * rows + i] = factor[i * r + t];
            }
        }
        core = core.mode_product(n, &ut, r);
        factors.push(factor);
        kept.push(r);
    }
    Tucker {
        core,
        factors,
        ranks: kept,
    }
}

impl Tucker {
    /// `G x_1 U_1 ... x_N U_N`: the tensor the decomposition represents.
    #[must_use]
    pub fn reconstruct(&self) -> Tensor {
        let mut t = self.core.clone();
        for (n, factor) in self.factors.iter().enumerate() {
            let rows = factor.len() / self.ranks[n];
            t = t.mode_product(n, factor, rows);
        }
        t
    }
}

fn splitmix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut x = z;
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// A standard normal by Box--Muller from two splitmix draws.
fn gaussian(seed: u64, i: u64) -> f64 {
    let a =
        (splitmix(seed ^ i.wrapping_mul(0x9E37_79B9_7F4A_7C15)) >> 11) as f64 / (1u64 << 53) as f64;
    let b = (splitmix(seed ^ i.wrapping_mul(0xD6E8_FEB8_6659_FD93).wrapping_add(1)) >> 11) as f64
        / (1u64 << 53) as f64;
    (-2.0 * (1.0 - a).ln()).sqrt() * (core::f64::consts::TAU * b).cos()
}

/// A rank-`k` SVD of a `rows x cols` matrix by the randomized range finder with `oversample`
/// extra sketch columns and `power` subspace iterations: `Y = (A A^T)^power A Omega`, `Q` its
/// orthonormal basis, `B = Q^T A`, and the SVD of `B` lifted through `Q`.
///
/// # Panics
///
/// If `a` does not hold `rows * cols` entries, or `k` is zero.
#[must_use]
pub fn randomized_svd(
    a: &[f64],
    rows: usize,
    cols: usize,
    k: usize,
    oversample: usize,
    power: usize,
    seed: u64,
) -> Svd {
    assert_eq!(
        a.len(),
        rows * cols,
        "a {rows} x {cols} matrix holds {} entries",
        rows * cols
    );
    assert!(k > 0, "a rank-zero sketch is empty");
    let l = (k + oversample).min(rows).min(cols);
    // Omega: cols x l Gaussian; Y = A Omega: rows x l, stored transposed (l x rows) so the
    // row-wise orthonormalisation applies.
    let mut yt = vec![0.0; l * rows];
    for r in 0..rows {
        for j in 0..l {
            let mut acc = 0.0;
            for c in 0..cols {
                acc += a[r * cols + c] * gaussian(seed, (c * l + j) as u64);
            }
            yt[j * rows + r] = acc;
        }
    }
    orthonormalise_rows(&mut yt, l, rows);
    for _ in 0..power {
        // Z = A^T Q (cols x l), then Y = A Z, orthonormalising after each product.
        let mut zt = vec![0.0; l * cols];
        for j in 0..l {
            for c in 0..cols {
                zt[j * cols + c] = (0..rows).map(|r| a[r * cols + c] * yt[j * rows + r]).sum();
            }
        }
        orthonormalise_rows(&mut zt, l, cols);
        for j in 0..l {
            for r in 0..rows {
                yt[j * rows + r] = (0..cols).map(|c| a[r * cols + c] * zt[j * cols + c]).sum();
            }
        }
        orthonormalise_rows(&mut yt, l, rows);
    }
    // B = Q^T A: l x cols.
    let mut b = vec![0.0; l * cols];
    for j in 0..l {
        for c in 0..cols {
            b[j * cols + c] = (0..rows).map(|r| yt[j * rows + r] * a[r * cols + c]).sum();
        }
    }
    let small = svd(&b, l, cols);
    let kk = k.min(small.s.len());
    let width = small.s.len();
    // U = Q U_B: rows x kk.
    let mut u = vec![0.0; rows * kk];
    for r in 0..rows {
        for t in 0..kk {
            u[r * kk + t] = (0..l)
                .map(|j| yt[j * rows + r] * small.u[j * width + t])
                .sum();
        }
    }
    let mut vt = vec![0.0; kk * cols];
    for t in 0..kk {
        vt[t * cols..(t + 1) * cols].copy_from_slice(&small.vt[t * cols..(t + 1) * cols]);
    }
    Svd {
        rows,
        cols,
        u,
        s: small.s[..kk].to_vec(),
        vt,
        rank: small.rank.min(kk),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A `rows x cols` matrix with prescribed singular values: `Q1 diag(s) Q2^T` from two
    /// orthonormalised Gaussian bases.
    fn with_spectrum(rows: usize, cols: usize, s: &[f64], seed: u64) -> Vec<f64> {
        let k = s.len();
        let mut q1t = (0..k * rows)
            .map(|i| gaussian(seed, i as u64))
            .collect::<Vec<_>>();
        let mut q2t = (0..k * cols)
            .map(|i| gaussian(seed ^ 0xABCD, i as u64))
            .collect::<Vec<_>>();
        assert_eq!(orthonormalise_rows(&mut q1t, k, rows), k);
        assert_eq!(orthonormalise_rows(&mut q2t, k, cols), k);
        let mut a = vec![0.0; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                a[r * cols + c] = (0..k)
                    .map(|t| q1t[t * rows + r] * s[t] * q2t[t * cols + c])
                    .sum();
            }
        }
        a
    }

    /// Whether the first `count` columns of a `rows x stride` matrix are orthonormal.
    fn orthonormal_columns(u: &[f64], rows: usize, stride: usize, count: usize, tol: f64) -> bool {
        (0..count).all(|s| {
            (0..count).all(|t| {
                let dot: f64 = (0..rows).map(|r| u[r * stride + s] * u[r * stride + t]).sum();
                (dot - if s == t { 1.0 } else { 0.0 }).abs() < tol
            })
        })
    }

    /// Reconstruction to `1e-10`, orthonormal factors, and the singular values a matrix was built
    /// with, on both orientations.
    #[test]
    fn the_svd_reconstructs_and_recovers_a_prescribed_spectrum() {
        let spectrum = [5.0, 3.0, 1.5, 0.25];
        for &(rows, cols) in &[(6usize, 9usize), (9, 6), (4, 4)] {
            let a = with_spectrum(rows, cols, &spectrum, 7);
            let dec = svd(&a, rows, cols);
            let k = rows.min(cols);
            assert_eq!(dec.rank, 4, "{rows}x{cols}: rank {}", dec.rank);
            for (got, want) in dec.s.iter().zip(&spectrum) {
                assert!((got - want).abs() < 1e-9, "{rows}x{cols}: {got} vs {want}");
            }
            let back = dec.truncate(k);
            assert!(frobenius_distance(&back, &a) < 1e-10);
            assert!(orthonormal_columns(&dec.u[..], rows, k, dec.rank.min(k), 1e-9));
            let mut vt = dec.vt.clone();
            assert_eq!(orthonormalise_rows(&mut vt, dec.rank, cols), dec.rank);
            assert!(frobenius_distance(&vt[..dec.rank * cols], &dec.vt[..dec.rank * cols]) < 1e-9);
        }
    }

    /// Eckart--Young: the rank-`k` truncation's error is `sqrt(sum_{i>k} s_i^2)` exactly, and a
    /// rank-`k` matrix built on a rotated basis does worse.
    #[test]
    fn the_truncation_is_the_best_rank_k_approximation() {
        let spectrum = [4.0, 2.0, 1.0, 0.5, 0.1];
        let (rows, cols) = (7usize, 8usize);
        let a = with_spectrum(rows, cols, &spectrum, 3);
        let dec = svd(&a, rows, cols);
        for k in 0..=5 {
            let want: f64 = spectrum.iter().skip(k).map(|v| v * v).sum::<f64>().sqrt();
            assert!((dec.truncation_error(k) - want).abs() < 1e-9);
            let formed = frobenius_distance(&dec.truncate(k), &a);
            assert!(
                (formed - want).abs() < 1e-9,
                "k {k}: formed {formed} vs closed form {want}"
            );
        }
        assert_eq!(dec.rank_for(0.6), 3);
        // A rival rank-2 matrix: the same top-2 singular values on a basis rotated by 30 degrees
        // inside the top-3 subspace of U.
        let (c, s) = (30f64.to_radians().cos(), 30f64.to_radians().sin());
        let width = dec.s.len();
        let mut rival = vec![0.0; rows * cols];
        for r in 0..rows {
            for col in 0..cols {
                let u0 = c * dec.u[r * width] + s * dec.u[r * width + 2];
                let u1 = dec.u[r * width + 1];
                rival[r * cols + col] =
                    u0 * dec.s[0] * dec.vt[col] + u1 * dec.s[1] * dec.vt[cols + col];
            }
        }
        let rival_error = frobenius_distance(&rival, &a);
        assert!(
            rival_error > dec.truncation_error(2) + 0.1,
            "rival {rival_error} vs optimum {}",
            dec.truncation_error(2)
        );
    }

    /// The HOSVD at full ranks reconstructs a `3 x 4 x 5` tensor to `1e-10`, its core is
    /// all-orthogonal, and a truncated Tucker approximation is never worse than the closed-form
    /// bound of the sum of the discarded mode singular values squared.
    #[test]
    fn the_hosvd_reconstructs_and_its_core_is_all_orthogonal() {
        let dims = vec![3usize, 4, 5];
        let data: Vec<f64> = (0..60)
            .map(|i| gaussian(11, i as u64) + 0.3 * (i as f64 * 0.1).sin())
            .collect();
        let x = Tensor::new(dims.clone(), data);
        let full = hosvd(&x, &[3, 4, 5]);
        let back = full.reconstruct();
        assert_eq!(back.dims, dims);
        assert!(frobenius_distance(&back.data, &x.data) < 1e-10);
        // All-orthogonality: for each mode, the slices of the core along it are orthogonal.
        for n in 0..3 {
            let unfolded = full.core.unfold(n);
            let rows = full.ranks[n];
            let cols = unfolded.len() / rows;
            for i in 0..rows {
                for j in 0..i {
                    let dot: f64 = (0..cols)
                        .map(|c| unfolded[i * cols + c] * unfolded[j * cols + c])
                        .sum();
                    assert!(dot.abs() < 1e-9, "mode {n} slices {i},{j}: {dot}");
                }
            }
        }
        // Truncation: error bounded by the root of the summed discarded mode-singular-values squared.
        let truncated = hosvd(&x, &[2, 3, 3]);
        let err = frobenius_distance(&truncated.reconstruct().data, &x.data);
        let mut bound = 0.0;
        for (n, &r) in [2usize, 3, 3].iter().enumerate() {
            let unfolded = x.unfold(n);
            let rows = dims[n];
            let dec = svd(&unfolded, rows, unfolded.len() / rows);
            bound += dec.s.iter().skip(r).map(|v| v * v).sum::<f64>();
        }
        assert!(
            err <= bound.sqrt() + 1e-9,
            "HOSVD error {err} above the bound {}",
            bound.sqrt()
        );
        assert!(err > 0.0);
    }

    /// The randomized range finder with one power iteration recovers the top three singular
    /// values of a `40 x 30` matrix with a decaying spectrum to `1e-6`, and its rank-`k`
    /// reconstruction is within a small factor of the optimum.
    #[test]
    fn the_randomized_svd_matches_the_full_one_on_a_decaying_spectrum() {
        let spectrum: Vec<f64> = (0..12).map(|i| 3.0 * 0.4f64.powi(i)).collect();
        let (rows, cols) = (40usize, 30usize);
        let a = with_spectrum(rows, cols, &spectrum, 5);
        let full = svd(&a, rows, cols);
        let sketch = randomized_svd(&a, rows, cols, 3, 5, 1, 99);
        for t in 0..3 {
            assert!(
                (sketch.s[t] - full.s[t]).abs() < 1e-6,
                "sigma_{t}: {} vs {}",
                sketch.s[t],
                full.s[t]
            );
        }
        let approx = sketch.truncate(3);
        let err = frobenius_distance(&approx, &a);
        let best = full.truncation_error(3);
        assert!(
            err < 1.05 * best + 1e-9,
            "sketch error {err} vs optimum {best}"
        );
        assert!(orthonormal_columns(&sketch.u, rows, 3, 3, 1e-8));
    }
}
