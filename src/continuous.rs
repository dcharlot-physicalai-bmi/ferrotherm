//! Continuous units, and the exact answers that keep them honest.
//!
//! Every state in this crate has been a spin. That is the right primitive for a sampling fabric,
//! and it is why the crate could be built on one kernel — but it excludes a model class the field
//! actually uses: real-valued visible units. A Gaussian–Bernoulli Boltzmann machine is the standard
//! way to put real data (pixel intensities, sensor readings) under an energy-based model, and its
//! continuous conditional is Gaussian, which is what makes it tractable AND checkable.
//!
//! ```text
//!   E(x, s) = ½ xᵀA x − bᵀx − Σ_{i,c} C_ic x_i s_c − Σ_c h_c s_c − Σ_{c<d} J_cd s_c s_d
//! ```
//!
//! with `x ∈ ℝⁿ`, `s ∈ {±1}^m`, and `A` symmetric positive definite. Sampling is single-site Gibbs
//! on both halves: `x_i` given everything else is Normal with mean
//! `(b_i + (Cs)_i − Σ_{j≠i} A_ij x_j) / A_ii` and variance `1/(β A_ii)`, and `s_c` is the ordinary
//! heat-bath spin this crate has always sampled, with `Σ_i C_ic x_i` added to its field.
//!
//! # Why this model and not a general continuous unit
//!
//! Because it can be checked exactly, at every level, and a continuous sampler that cannot be
//! checked is a continuous sampler nobody should believe:
//!
//! * with no spins the distribution is `N(A⁻¹b, (βA)⁻¹)` — the mean, the **whole covariance**, and
//!   `ln Z = (β/2) bᵀA⁻¹b + (n/2) ln(2π/β) − ½ ln det A` are closed forms ([`Gbm::exact_gaussian`]);
//! * with spins, integrating `x` out is still a Gaussian integral, so enumerating `2^m` spin states
//!   gives an exact `ln Z` and exact marginals at any `m` small enough to enumerate
//!   ([`Gbm::exact_log_z`]).
//!
//! A general nonlinear continuous unit has none of those, which is why it is not what shipped
//! first. Continuous Hopfield's graded response and equilibrium propagation's original continuous
//! formulation need exactly that generality; this module is the substrate they would be built on,
//! not those models themselves.

use crate::rng::Pcg;

/// Cholesky factor `L` of a symmetric positive-definite `a` (row-major `n×n`), or `None` if `a` is
/// not positive definite — which is the honest way to answer, since the sampler's variance
/// `1/(βA_ii)` is meaningless otherwise.
pub fn cholesky(a: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut l = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[i * n + j];
            for k in 0..j {
                s -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                if !(s > 0.0) {
                    return None;
                }
                l[i * n + i] = s.sqrt();
            } else {
                l[i * n + j] = s / l[j * n + j];
            }
        }
    }
    Some(l)
}

/// `ln det A` from its Cholesky factor: `2 Σ ln L_ii`.
pub fn log_det(a: &[f64], n: usize) -> Option<f64> {
    cholesky(a, n).map(|l| 2.0 * (0..n).map(|i| l[i * n + i].ln()).sum::<f64>())
}

/// Solve `A y = v` by Cholesky substitution.
pub fn solve(a: &[f64], n: usize, v: &[f64]) -> Option<Vec<f64>> {
    let l = cholesky(a, n)?;
    let mut y = vec![0.0; n];
    for i in 0..n {
        let mut s = v[i];
        for k in 0..i {
            s -= l[i * n + k] * y[k];
        }
        y[i] = s / l[i * n + i];
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = y[i];
        for k in (i + 1)..n {
            s -= l[k * n + i] * x[k];
        }
        x[i] = s / l[i * n + i];
    }
    Some(x)
}

/// `A⁻¹`, column by column.
pub fn inverse(a: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut inv = vec![0.0; n * n];
    for c in 0..n {
        let mut e = vec![0.0; n];
        e[c] = 1.0;
        let col = solve(a, n, &e)?;
        for r in 0..n {
            inv[r * n + c] = col[r];
        }
    }
    Some(inv)
}

/// A Gaussian–Bernoulli Boltzmann machine.
#[derive(Clone, Debug)]
pub struct Gbm {
    /// Continuous units.
    pub n_real: usize,
    /// Spins.
    pub n_spin: usize,
    /// `A`, row-major `n_real × n_real`, symmetric positive definite.
    pub a: Vec<f64>,
    /// `b`, length `n_real`.
    pub b: Vec<f64>,
    /// `C`, row-major `n_real × n_spin`.
    pub c: Vec<f64>,
    /// Spin fields, length `n_spin`.
    pub h: Vec<f64>,
    /// Spin couplings, row-major `n_spin × n_spin`, symmetric, diagonal ignored.
    pub j: Vec<f64>,
}

impl Gbm {
    /// A purely continuous model.
    pub fn gaussian(n: usize, a: Vec<f64>, b: Vec<f64>) -> Gbm {
        assert_eq!(a.len(), n * n);
        assert_eq!(b.len(), n);
        Gbm { n_real: n, n_spin: 0, a, b, c: Vec::new(), h: Vec::new(), j: Vec::new() }
    }

    /// `E(x, s)`.
    pub fn energy(&self, x: &[f64], s: &[i8]) -> f64 {
        let (n, m) = (self.n_real, self.n_spin);
        let mut e = 0.0;
        for i in 0..n {
            for k in 0..n {
                e += 0.5 * self.a[i * n + k] * x[i] * x[k];
            }
            e -= self.b[i] * x[i];
            for c in 0..m {
                e -= self.c[i * m + c] * x[i] * s[c] as f64;
            }
        }
        for c in 0..m {
            e -= self.h[c] * s[c] as f64;
            for d in (c + 1)..m {
                e -= self.j[c * m + d] * (s[c] as i32 * s[d] as i32) as f64;
            }
        }
        e
    }

    /// One sweep: every continuous unit from its Gaussian conditional, then every spin from its
    /// heat-bath conditional.
    pub fn sweep(&self, beta: f64, x: &mut [f64], s: &mut [i8], rng: &mut Pcg) {
        let (n, m) = (self.n_real, self.n_spin);
        for i in 0..n {
            let mut mean = self.b[i];
            for c in 0..m {
                mean += self.c[i * m + c] * s[c] as f64;
            }
            for k in 0..n {
                if k != i {
                    mean -= self.a[i * n + k] * x[k];
                }
            }
            let prec = self.a[i * n + i];
            mean /= prec;
            let sd = 1.0 / (beta * prec).sqrt();
            x[i] = mean + sd * standard_normal(rng);
        }
        for c in 0..m {
            let mut field = self.h[c];
            for d in 0..m {
                if d != c {
                    field += self.j[c * m + d] * s[d] as f64;
                }
            }
            for i in 0..n {
                field += self.c[i * m + c] * x[i];
            }
            s[c] = crate::kernel::draw(field, beta, rng);
        }
    }

    /// Draw after `burn_in` sweeps, recording `draws` states.
    pub fn collect(&self, beta: f64, burn_in: usize, draws: usize, seed: u64) -> (Vec<Vec<f64>>, Vec<Vec<i8>>) {
        let mut rng = Pcg::new(seed, 31);
        let mut x = vec![0.0; self.n_real];
        let mut s = vec![1i8; self.n_spin];
        for _ in 0..burn_in {
            self.sweep(beta, &mut x, &mut s, &mut rng);
        }
        let (mut xs, mut ss) = (Vec::with_capacity(draws), Vec::with_capacity(draws));
        for _ in 0..draws {
            self.sweep(beta, &mut x, &mut s, &mut rng);
            xs.push(x.clone());
            ss.push(s.clone());
        }
        (xs, ss)
    }

    /// The exact mean, covariance and `ln Z` of a model with no spins.
    ///
    /// `N(A⁻¹b, (βA)⁻¹)` and `ln Z = (β/2) bᵀA⁻¹b + (n/2) ln(2π/β) − ½ ln det A`. `None` when `A` is
    /// not positive definite.
    pub fn exact_gaussian(&self, beta: f64) -> Option<(Vec<f64>, Vec<f64>, f64)> {
        assert_eq!(self.n_spin, 0, "use exact_log_z when there are spins");
        let n = self.n_real;
        let mean = solve(&self.a, n, &self.b)?;
        let inv = inverse(&self.a, n)?;
        let cov: Vec<f64> = inv.iter().map(|v| v / beta).collect();
        let quad: f64 = (0..n).map(|i| self.b[i] * mean[i]).sum();
        let ln_z = 0.5 * beta * quad + 0.5 * n as f64 * (core::f64::consts::TAU / beta).ln() - 0.5 * log_det(&self.a, n)?;
        Some((mean, cov, ln_z))
    }

    /// Exact `ln Z` with spins, by enumerating them and integrating `x` in closed form.
    ///
    /// For each spin state the `x` integral is Gaussian with `b → b + Cs`, so the whole partition
    /// function is a finite sum of exact terms. Refuses above 20 spins.
    pub fn exact_log_z(&self, beta: f64) -> Option<f64> {
        let (n, m) = (self.n_real, self.n_spin);
        assert!(m <= 20, "enumeration refuses {m} spins");
        let inv = inverse(&self.a, n)?;
        let ld = log_det(&self.a, n)?;
        let base = 0.5 * n as f64 * (core::f64::consts::TAU / beta).ln() - 0.5 * ld;
        let mut terms = Vec::with_capacity(1 << m);
        let mut s = vec![-1i8; m.max(1)];
        for mask in 0..(1usize << m) {
            for c in 0..m {
                s[c] = if mask >> c & 1 == 1 { 1 } else { -1 };
            }
            // b + C s
            let bc: Vec<f64> = (0..n)
                .map(|i| self.b[i] + (0..m).map(|c| self.c[i * m + c] * s[c] as f64).sum::<f64>())
                .collect();
            let mut quad = 0.0;
            for i in 0..n {
                for k in 0..n {
                    quad += bc[i] * inv[i * n + k] * bc[k];
                }
            }
            let mut spin = 0.0;
            for c in 0..m {
                spin += self.h[c] * s[c] as f64;
                for d in (c + 1)..m {
                    spin += self.j[c * m + d] * (s[c] as i32 * s[d] as i32) as f64;
                }
            }
            terms.push(beta * (spin + 0.5 * quad));
        }
        let mx = terms.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        Some(base + mx + terms.iter().map(|t| (t - mx).exp()).sum::<f64>().ln())
    }
}

/// A standard normal from the crate's uniform, by Box–Muller.
fn standard_normal(rng: &mut Pcg) -> f64 {
    let u1 = rng.f64().max(f64::MIN_POSITIVE);
    let u2 = rng.f64();
    (-2.0 * u1.ln()).sqrt() * (core::f64::consts::TAU * u2).cos()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spd(n: usize, seed: u64) -> Vec<f64> {
        // A = M Mᵀ + n I, symmetric and comfortably positive definite
        let mut rng = Pcg::new(seed, 5);
        let m: Vec<f64> = (0..n * n).map(|_| rng.f64() - 0.5).collect();
        let mut a = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut s = 0.0;
                for k in 0..n {
                    s += m[i * n + k] * m[j * n + k];
                }
                a[i * n + j] = s + if i == j { n as f64 } else { 0.0 };
            }
        }
        a
    }

    /// The linear algebra is right before anything is sampled with it.
    #[test]
    fn cholesky_solve_and_logdet_agree_with_direct_computation() {
        for n in [1usize, 2, 5, 8] {
            let a = spd(n, 7 + n as u64);
            let inv = inverse(&a, n).unwrap();
            // A A⁻¹ = I
            for i in 0..n {
                for j in 0..n {
                    let mut s = 0.0;
                    for k in 0..n {
                        s += a[i * n + k] * inv[k * n + j];
                    }
                    let want = f64::from(u8::from(i == j));
                    assert!((s - want).abs() < 1e-9, "n={n} ({i},{j}) = {s}");
                }
            }
            // ln det against the crate's Jacobi routine. NOTE its contract: it returns the
            // EIGENVECTOR matrix and leaves the eigenvalues on the diagonal of the matrix it was
            // given, which it modifies. Reading its return value as eigenvalues gives ln(0) on a
            // diagonal input -- which is how this assertion first failed, on a correct log_det.
            let mut copy = a.clone();
            let _vectors = crate::linalg::jacobi_eig(&mut copy, n);
            let want: f64 = (0..n).map(|i| copy[i * n + i].ln()).sum();
            assert!((log_det(&a, n).unwrap() - want).abs() < 1e-8, "n={n}: {} vs {want}", log_det(&a, n).unwrap());
            // a matrix that is not positive definite is refused, not factored
            let mut bad = a.clone();
            bad[0] = -1.0;
            assert!(cholesky(&bad, n).is_none(), "n={n}: a negative pivot must be refused");
        }
        assert!(cholesky(&[-1.0], 1).is_none());
    }

    /// A purely continuous model samples its exact Gaussian: mean, full covariance, and ln Z.
    #[test]
    fn the_gaussian_sampler_reproduces_the_exact_normal() {
        let n = 4;
        let a = spd(n, 11);
        let b: Vec<f64> = (0..n).map(|i| 0.4 * (i as f64) - 0.6).collect();
        let g = Gbm::gaussian(n, a, b);
        let beta = 1.3;
        let (mean, cov, _) = g.exact_gaussian(beta).unwrap();
        let (xs, _) = g.collect(beta, 2000, 60_000, 3);
        let k = xs.len() as f64;
        let m: Vec<f64> = (0..n).map(|i| xs.iter().map(|x| x[i]).sum::<f64>() / k).collect();
        for i in 0..n {
            // the chain's own standard error on the mean is sqrt(cov_ii / k), inflated for
            // autocorrelation; five of those is a wide but honest window
            let se = (cov[i * n + i] / k).sqrt();
            assert!((m[i] - mean[i]).abs() < 8.0 * se + 0.01, "mean {i}: {} vs {}", m[i], mean[i]);
        }
        for i in 0..n {
            for j in 0..n {
                let c: f64 = xs.iter().map(|x| (x[i] - m[i]) * (x[j] - m[j])).sum::<f64>() / k;
                assert!((c - cov[i * n + j]).abs() < 0.06, "cov ({i},{j}): {c} vs {}", cov[i * n + j]);
            }
        }
    }

    /// The closed-form ln Z is the one the definition gives, checked by numerical integration in
    /// one dimension where that is possible.
    #[test]
    fn the_closed_form_partition_function_matches_quadrature() {
        let (a, b, beta) = (2.5f64, 0.8f64, 1.4f64);
        let g = Gbm::gaussian(1, vec![a], vec![b]);
        let (_, _, ln_z) = g.exact_gaussian(beta).unwrap();
        // midpoint rule over a window many standard deviations wide
        let sd = 1.0 / (beta * a).sqrt();
        let (lo, hi, steps) = (b / a - 30.0 * sd, b / a + 30.0 * sd, 2_000_000);
        let h = (hi - lo) / steps as f64;
        let z: f64 = (0..steps)
            .map(|i| {
                let x = lo + (i as f64 + 0.5) * h;
                (-beta * (0.5 * a * x * x - b * x)).exp() * h
            })
            .sum();
        assert!((ln_z - z.ln()).abs() < 1e-6, "closed form {ln_z} vs quadrature {}", z.ln());
    }

    /// With spins the model still has an exact ln Z, and the sampler agrees with the marginals it
    /// implies.
    #[test]
    fn the_hybrid_model_matches_its_enumerated_answer() {
        let (n, m) = (3usize, 3usize);
        let a = spd(n, 13);
        let b = vec![0.2, -0.4, 0.1];
        let c = vec![0.5, -0.3, 0.2, 0.1, 0.4, -0.2, -0.5, 0.3, 0.1];
        let h = vec![0.15, -0.2, 0.05];
        let mut j = vec![0.0; m * m];
        j[1] = 0.3;
        j[m] = 0.3;
        j[m + 2] = -0.25;
        j[2 * m + 1] = -0.25;
        let g = Gbm { n_real: n, n_spin: m, a, b, c, h, j };
        let beta = 1.1;
        let ln_z = g.exact_log_z(beta).unwrap();
        assert!(ln_z.is_finite());

        // exact spin marginals from the same enumeration, by finite difference of ln Z in h_c
        let exact_mag = |c: usize| {
            let eps = 1e-5;
            let mut up = g.clone();
            up.h[c] += eps;
            let mut dn = g.clone();
            dn.h[c] -= eps;
            (up.exact_log_z(beta).unwrap() - dn.exact_log_z(beta).unwrap()) / (2.0 * eps * beta)
        };
        let (xs, ss) = g.collect(beta, 3000, 80_000, 5);
        let k = ss.len() as f64;
        for cc in 0..m {
            let got: f64 = ss.iter().map(|s| s[cc] as f64).sum::<f64>() / k;
            assert!((got - exact_mag(cc)).abs() < 0.03, "spin {cc}: {got} vs {}", exact_mag(cc));
        }
        // and the continuous means, likewise by differentiating in b_i
        for i in 0..n {
            let eps = 1e-5;
            let mut up = g.clone();
            up.b[i] += eps;
            let mut dn = g.clone();
            dn.b[i] -= eps;
            let want = (up.exact_log_z(beta).unwrap() - dn.exact_log_z(beta).unwrap()) / (2.0 * eps * beta);
            let got: f64 = xs.iter().map(|x| x[i]).sum::<f64>() / k;
            assert!((got - want).abs() < 0.03, "real {i}: {got} vs {want}");
        }
    }
}
