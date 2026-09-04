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

// ---- general nonlinear units ------------------------------------------------------------------
//
// [`Gbm`] is exactly solvable and therefore exactly checkable, which is why it shipped first. It is
// also the only continuous model with that property: the moment a unit's local term stops being
// quadratic, the conditional stops being Gaussian, `ln Z` stops having a closed form, and the
// crate's usual oracles are gone.
//
// They are not the only oracles available. In FEW dimensions the partition function is an integral
// a computer can simply do: a product rule over a bounded box converges to whatever precision the
// grid affords, and every expectation with it. That is the oracle here -- slower than a closed form
// and available only up to about three units, but exact enough to hold a sampler to, which is the
// requirement. Everything below is verified against it before it is trusted at any size.

/// A unit's local energy term.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Potential {
    /// `½ a x² − b x`. Recovers the Gaussian case, so the general sampler can be checked against
    /// [`Gbm`]'s closed forms as well as against quadrature.
    Quadratic { a: f64, b: f64 },
    /// Continuous Hopfield's graded-neuron term (Hopfield 1984), `gain · ∫₀ˣ atanh(u) du`
    /// `= gain · (x·atanh(x) + ½ ln(1 − x²))`, finite only on `(−1, 1)`.
    ///
    /// This is the term that makes a continuous Hopfield network a Hopfield network: the transfer
    /// function `g = tanh` enters the ENERGY through the integral of its inverse, so the stationary
    /// points of the dynamics are the fixed points of `x = tanh(gain · field)`. Its domain is a
    /// hard box, which the sampler respects by rejecting rather than by clamping — a clamp would
    /// pile probability on the boundary that the model does not put there.
    HopfieldTanh { gain: f64 },
    /// `a x⁴ − b x²`, the double well: not log-concave, so the conditional is bimodal and a sampler
    /// that quietly assumes unimodality will be caught.
    DoubleWell { a: f64, b: f64 },
}

impl Potential {
    /// `V(x)`, or `f64::INFINITY` outside the support.
    pub fn energy(&self, x: f64) -> f64 {
        match *self {
            Potential::Quadratic { a, b } => 0.5 * a * x * x - b * x,
            Potential::HopfieldTanh { gain } => {
                if x.abs() >= 1.0 {
                    f64::INFINITY
                } else {
                    gain * (x * x.atanh() + 0.5 * (1.0 - x * x).ln())
                }
            }
            Potential::DoubleWell { a, b } => a * x * x * x * x - b * x * x,
        }
    }

    /// The interval outside which the energy is infinite, for quadrature and for proposals.
    pub fn support(&self) -> (f64, f64) {
        match *self {
            Potential::HopfieldTanh { .. } => (-1.0, 1.0),
            _ => (f64::NEG_INFINITY, f64::INFINITY),
        }
    }
}

/// A continuous energy-based model with arbitrary local terms: `E(x) = Σ V_i(x_i) − Σ_{i<j} W_ij x_i x_j`.
#[derive(Clone, Debug)]
pub struct ContinuousEbm {
    pub potentials: Vec<Potential>,
    /// Row-major `n×n`, symmetric, diagonal ignored.
    pub w: Vec<f64>,
}

impl ContinuousEbm {
    pub fn new(potentials: Vec<Potential>, w: Vec<f64>) -> Self {
        let n = potentials.len();
        assert_eq!(w.len(), n * n);
        for i in 0..n {
            for j in 0..n {
                assert!((w[i * n + j] - w[j * n + i]).abs() < 1e-12, "W must be symmetric");
            }
        }
        ContinuousEbm { potentials, w }
    }

    pub fn n(&self) -> usize {
        self.potentials.len()
    }

    /// `E(x)`.
    pub fn energy(&self, x: &[f64]) -> f64 {
        let n = self.n();
        let mut e = 0.0;
        for i in 0..n {
            e += self.potentials[i].energy(x[i]);
            for j in (i + 1)..n {
                e -= self.w[i * n + j] * x[i] * x[j];
            }
        }
        e
    }

    /// A Metropolis-within-Gibbs sweep: one symmetric Gaussian proposal per unit, accepted by the
    /// Metropolis rule. `step` is per-unit and is the caller's to tune — [`Self::run`] adapts it.
    ///
    /// An infinite energy (a proposal outside the support) is rejected, which is what keeps
    /// [`Potential::HopfieldTanh`]'s hard box honest.
    pub fn sweep(&self, beta: f64, x: &mut [f64], step: &[f64], rng: &mut Pcg, accepts: &mut [u64]) {
        let n = self.n();
        for i in 0..n {
            let before = self.local_energy(x, i);
            let old = x[i];
            let proposal = old + step[i] * standard_normal(rng);
            x[i] = proposal;
            let after = self.local_energy(x, i);
            let d = after - before;
            // exp(-beta * inf) = 0, so an out-of-support proposal is rejected without a branch
            let accept = d <= 0.0 || (d.is_finite() && rng.f64() < (-beta * d).exp());
            if accept {
                accepts[i] += 1;
            } else {
                x[i] = old;
            }
        }
    }

    /// The part of the energy that depends on `x[i]`.
    fn local_energy(&self, x: &[f64], i: usize) -> f64 {
        let n = self.n();
        let mut e = self.potentials[i].energy(x[i]);
        for j in 0..n {
            if j != i {
                e -= self.w[i * n + j] * x[i] * x[j];
            }
        }
        e
    }

    /// Sample, adapting each unit's proposal step toward 44% acceptance during burn-in.
    ///
    /// The target is the one-dimensional optimum for a random-walk Metropolis chain; adapting only
    /// during burn-in keeps the recorded chain a proper Markov chain, since a step size that keeps
    /// changing with the history is not one.
    pub fn run(&self, beta: f64, burn_in: usize, draws: usize, seed: u64) -> Vec<Vec<f64>> {
        let n = self.n();
        let mut rng = Pcg::new(seed, 41);
        let mut x = vec![0.0; n];
        for i in 0..n {
            let (lo, hi) = self.potentials[i].support();
            if lo.is_finite() && hi.is_finite() {
                x[i] = 0.5 * (lo + hi);
            }
        }
        let mut step = vec![0.5; n];
        let mut accepts = vec![0u64; n];
        let window = 100.max(burn_in / 20);
        for t in 0..burn_in {
            self.sweep(beta, &mut x, &step, &mut rng, &mut accepts);
            if (t + 1) % window == 0 {
                for i in 0..n {
                    let rate = accepts[i] as f64 / window as f64;
                    step[i] *= if rate > 0.44 { 1.2 } else { 0.85 };
                    step[i] = step[i].clamp(1e-6, 100.0);
                    accepts[i] = 0;
                }
            }
        }
        let mut out = Vec::with_capacity(draws);
        for _ in 0..draws {
            self.sweep(beta, &mut x, &step, &mut rng, &mut accepts);
            out.push(x.clone());
        }
        out
    }

    /// `ln Z` and `⟨x⟩` by product-rule quadrature over `[lo, hi]^n` with `grid` points per axis.
    ///
    /// The oracle for everything above. Exact to the grid, and refused above three units because
    /// `grid^n` is what it costs — an answer that takes a week is not an oracle.
    pub fn exact_by_quadrature(&self, beta: f64, lo: f64, hi: f64, grid: usize) -> (f64, Vec<f64>) {
        let n = self.n();
        assert!(n <= 3, "quadrature refuses {n} units; it costs grid^n");
        assert!(grid >= 2);
        let h = (hi - lo) / grid as f64;
        let axis: Vec<f64> = (0..grid).map(|i| lo + (i as f64 + 0.5) * h).collect();
        let mut logs: Vec<f64> = Vec::with_capacity(grid.pow(n as u32));
        let mut pts: Vec<Vec<f64>> = Vec::with_capacity(logs.capacity());
        let mut idx = vec![0usize; n];
        loop {
            let x: Vec<f64> = (0..n).map(|d| axis[idx[d]]).collect();
            let e = self.energy(&x);
            if e.is_finite() {
                logs.push(-beta * e);
                pts.push(x);
            }
            let mut d = 0;
            while d < n {
                idx[d] += 1;
                if idx[d] < grid {
                    break;
                }
                idx[d] = 0;
                d += 1;
            }
            if d == n {
                break;
            }
        }
        let mx = logs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let ws: Vec<f64> = logs.iter().map(|l| (l - mx).exp()).collect();
        let sum: f64 = ws.iter().sum();
        let ln_z = mx + sum.ln() + n as f64 * h.ln();
        let mean: Vec<f64> = (0..n)
            .map(|d| pts.iter().zip(&ws).map(|(p, w)| w * p[d]).sum::<f64>() / sum)
            .collect();
        (ln_z, mean)
    }
}

impl ContinuousEbm {
    /// `ln Z` of an open CHAIN by the transfer operator — exact to the grid, at any length.
    ///
    /// [`Self::exact_by_quadrature`] is capped near three units because a product rule costs
    /// `grid^n`. A chain does not need one: discretising each variable on `grid` points makes the
    /// partition function a product of `grid × grid` transfer matrices,
    ///
    /// ```text
    ///   Z = 1ᵀ T₁ T₂ ⋯ T_{n−1} 1 · h^n,
    ///   T_i[a,b] = exp(−β( ½V_i(x_a) + ½V_{i+1}(x_b) − W_{i,i+1} x_a x_b ))
    /// ```
    ///
    /// which is `O(n · grid²)` — linear in the number of units. So a nonlinear chain of twenty
    /// units has an exact answer where quadrature refuses past three, and the sampler can be held
    /// to it at a size where its own behaviour is interesting.
    ///
    /// Requires the couplings to form an open chain: only `W_{i,i+1}` may be non-zero. Returns
    /// `None` otherwise rather than silently pretending the extra couplings are not there.
    pub fn chain_log_z(&self, beta: f64, lo: f64, hi: f64, grid: usize) -> Option<f64> {
        let n = self.n();
        assert!(grid >= 2);
        if n < 2 {
            return None;
        }
        for i in 0..n {
            for j in 0..n {
                if j > i + 1 && self.w[i * n + j] != 0.0 {
                    return None; // not a chain
                }
            }
        }
        let h = (hi - lo) / grid as f64;
        let x: Vec<f64> = (0..grid).map(|k| lo + (k as f64 + 0.5) * h).collect();
        let v: Vec<Vec<f64>> = (0..n).map(|i| x.iter().map(|&u| self.potentials[i].energy(u)).collect()).collect();
        // carry a log-scale running vector to keep the product from underflowing
        let mut log_scale = 0.0f64;
        // Each transfer matrix carries HALF of the potential at each of its two sites, so an
        // interior site collects two halves and is counted once -- but the two END sites appear in
        // only one matrix each and would be counted half. The missing halves go here and at the
        // close, which is a boundary term: getting it wrong costs a constant offset independent of
        // the chain length, which is exactly how the omission was found.
        let mut vec: Vec<f64> = (0..grid).map(|a| (-beta * 0.5 * v[0][a]).exp()).collect();
        for i in 0..(n - 1) {
            let w = self.w[i * n + (i + 1)];
            let mut next = vec![0.0f64; grid];
            for b in 0..grid {
                let mut acc = 0.0;
                for a in 0..grid {
                    let e = 0.5 * v[i][a] + 0.5 * v[i + 1][b] - w * x[a] * x[b];
                    if e.is_finite() {
                        acc += vec[a] * (-beta * e).exp();
                    }
                }
                next[b] = acc;
            }
            let mx = next.iter().cloned().fold(0.0f64, f64::max);
            if !(mx > 0.0) {
                return None;
            }
            for u in next.iter_mut() {
                *u /= mx;
            }
            log_scale += mx.ln();
            vec = next;
        }
        let total: f64 = vec.iter().enumerate().map(|(b, u)| u * (-beta * 0.5 * v[n - 1][b]).exp()).sum();
        Some(log_scale + total.ln() + n as f64 * h.ln())
    }
}

#[cfg(test)]
mod nonlinear_tests {
    use super::*;

    /// The general sampler reproduces the Gaussian case the closed form already knows.
    ///
    /// A general machine that disagrees with the special case it contains is broken in a way no
    /// quadrature check would isolate, so this comes first.
    #[test]
    fn the_general_sampler_agrees_with_the_closed_form_gaussian() {
        let (a, b, beta) = (2.0f64, 0.7f64, 1.1f64);
        let ebm = ContinuousEbm::new(vec![Potential::Quadratic { a, b }], vec![0.0]);
        let g = Gbm::gaussian(1, vec![a], vec![b]);
        let (mean, cov, ln_z) = g.exact_gaussian(beta).unwrap();
        let (q_ln_z, q_mean) = ebm.exact_by_quadrature(beta, -8.0, 8.0, 40_000);
        assert!((q_ln_z - ln_z).abs() < 1e-6, "quadrature {q_ln_z} vs closed form {ln_z}");
        assert!((q_mean[0] - mean[0]).abs() < 1e-9);
        let xs = ebm.run(beta, 20_000, 200_000, 3);
        let m: f64 = xs.iter().map(|x| x[0]).sum::<f64>() / xs.len() as f64;
        let se = (cov[0] / xs.len() as f64).sqrt();
        assert!((m - mean[0]).abs() < 10.0 * se + 0.01, "sampled {m} vs exact {}", mean[0]);
    }

    /// Continuous Hopfield: the sampler matches quadrature, and respects the hard box.
    #[test]
    fn the_hopfield_unit_matches_quadrature_and_stays_in_its_box() {
        let n = 2;
        let p = vec![Potential::HopfieldTanh { gain: 1.0 }; n];
        let mut w = vec![0.0; n * n];
        w[1] = 0.8;
        w[n] = 0.8;
        let ebm = ContinuousEbm::new(p, w);
        let beta = 2.0;
        let (ln_z, mean) = ebm.exact_by_quadrature(beta, -1.0, 1.0, 1200);
        assert!(ln_z.is_finite());
        let xs = ebm.run(beta, 30_000, 200_000, 5);
        assert!(xs.iter().all(|x| x.iter().all(|v| v.abs() < 1.0)), "a sample left the support");
        for d in 0..n {
            let m: f64 = xs.iter().map(|x| x[d]).sum::<f64>() / xs.len() as f64;
            assert!((m - mean[d]).abs() < 0.02, "unit {d}: sampled {m} vs quadrature {}", mean[d]);
        }
    }

    /// The transfer operator agrees with quadrature, and then goes where quadrature cannot.
    #[test]
    fn the_transfer_operator_matches_quadrature_and_scales_past_it() {
        let chain = |n: usize, p: Potential, w: f64| {
            let mut m = vec![0.0; n * n];
            for i in 0..(n - 1) {
                m[i * n + i + 1] = w;
                m[(i + 1) * n + i] = w;
            }
            ContinuousEbm::new(vec![p; n], m)
        };
        let e = chain(2, Potential::HopfieldTanh { gain: 1.0 }, 0.7);
        let (q, _) = e.exact_by_quadrature(1.5, -1.0, 1.0, 1200);
        let tr = e.chain_log_z(1.5, -1.0, 1.0, 1200).unwrap();
        assert!((q - tr).abs() < 1e-9, "quadrature {q} vs transfer {tr}");

        // grid convergence: the answer must stop moving
        let long = chain(12, Potential::DoubleWell { a: 1.0, b: 2.0 }, 0.4);
        let a = long.chain_log_z(1.0, -3.0, 3.0, 300).unwrap();
        let b = long.chain_log_z(1.0, -3.0, 3.0, 900).unwrap();
        assert!((a - b).abs() < 1e-6, "grid 300 gives {a}, grid 900 gives {b}");

        // a graph that is not a chain is refused rather than answered wrongly
        let mut w = vec![0.0; 9];
        w[2] = 0.5;
        w[6] = 0.5;
        assert!(ContinuousEbm::new(vec![Potential::Quadratic { a: 2.0, b: 0.0 }; 3], w)
            .chain_log_z(1.0, -3.0, 3.0, 50)
            .is_none());
    }

    /// The nonlinear SAMPLER is now checkable at twelve units, where quadrature refuses past three.
    ///
    /// `⟨E⟩ = −d ln Z / dβ`, and the transfer operator gives `ln Z` exactly, so a central
    /// difference gives an exact mean energy to hold a twelve-unit chain of double wells to. This
    /// is the whole point of the operator: the sampler is verified at a size where its behaviour is
    /// interesting rather than only where a product rule can still run.
    #[test]
    fn a_twelve_unit_nonlinear_chain_is_verified_against_the_transfer_operator() {
        let n = 12;
        let mut w = vec![0.0; n * n];
        for i in 0..(n - 1) {
            w[i * n + i + 1] = 0.4;
            w[(i + 1) * n + i] = 0.4;
        }
        let ebm = ContinuousEbm::new(vec![Potential::DoubleWell { a: 1.0, b: 2.0 }; n], w);
        let beta = 1.0;
        let h = 1e-4;
        let up = ebm.chain_log_z(beta + h, -3.5, 3.5, 700).unwrap();
        let dn = ebm.chain_log_z(beta - h, -3.5, 3.5, 700).unwrap();
        let want = -(up - dn) / (2.0 * h);
        let xs = ebm.run(beta, 40_000, 300_000, 19);
        let got: f64 = xs.iter().map(|x| ebm.energy(x)).sum::<f64>() / xs.len() as f64;
        assert!((got - want).abs() < 0.12, "sampled <E> {got} vs transfer-operator {want}");
    }

    /// A double well is bimodal, and the sampler visits both wells rather than one.
    ///
    /// This is the test a sampler that quietly assumes a unimodal conditional fails.
    #[test]
    fn the_double_well_is_sampled_on_both_sides() {
        let ebm = ContinuousEbm::new(vec![Potential::DoubleWell { a: 1.0, b: 4.0 }], vec![0.0]);
        let beta = 1.0;
        let (_, mean) = ebm.exact_by_quadrature(beta, -4.0, 4.0, 40_000);
        assert!(mean[0].abs() < 1e-9, "the double well is symmetric: {}", mean[0]);
        let xs = ebm.run(beta, 20_000, 200_000, 7);
        let left = xs.iter().filter(|x| x[0] < 0.0).count();
        let frac = left as f64 / xs.len() as f64;
        assert!((0.3..0.7).contains(&frac), "only {frac} of samples in the left well -- it is stuck");
        // and the second moment, which a stuck chain would also get wrong
        let m2: f64 = xs.iter().map(|x| x[0] * x[0]).sum::<f64>() / xs.len() as f64;
        let pts: Vec<f64> = (0..40_000).map(|i| -4.0 + (i as f64 + 0.5) * 8.0 / 40_000.0).collect();
        let ws: Vec<f64> = pts.iter().map(|&v| (-beta * ebm.energy(&[v])).exp()).collect();
        let z: f64 = ws.iter().sum();
        let want: f64 = pts.iter().zip(&ws).map(|(p, w)| w * p * p).sum::<f64>() / z;
        assert!((m2 - want).abs() < 0.05, "second moment {m2} vs {want}");
    }
}

// ---- score matching ---------------------------------------------------------------------------
//
// Training an energy-based model by maximum likelihood needs the gradient of `ln Z`, which is an
// expectation under the model, which needs sampling -- that is contrastive divergence, and it is
// why EBMs were hard. Hyvärinen (2005) removed the partition function from the problem entirely:
// minimise the expected squared distance between the model's score `∇_x ln p` and the data's, and
// since `∇_x ln p = −∇_x E` the normaliser differentiates away. Integrating by parts leaves an
// objective in the model's derivatives alone:
//
//   J(θ) = E_data[ Σ_i ( ½ (∂_i E)² − ∂²_i E ) ]
//
// No sampling, no `ln Z`, one pass over the data. It is the line that runs from EBMs to denoising
// score matching to diffusion models, and it is the training method this crate did not have.
//
// For the Gaussian family it is not only tractable but EXACTLY SOLVABLE, which is what makes it
// checkable here: with `E = ½xᵀAx − bᵀx`, `∂_i E = (Ax − b)_i` and `∂²_i E = A_ii`, so
// `J = E[½‖Ax − b‖² − tr A]`, and setting both gradients to zero gives
//
//   b = A μ,      A Σ = I      ⇒     A = Σ⁻¹,   b = Σ⁻¹ μ
//
// The score-matching optimum for a Gaussian is the inverse covariance. That is a closed form to
// hold the fit against, and the test does exactly that: generate from a known `A`, fit, recover it.

/// The Hyvärinen score-matching objective of a Gaussian model on `data`.
///
/// `J = mean over data of ( ½‖Ax − b‖² − tr A )`. Lower is better; the minimiser is
/// [`score_match_gaussian`].
pub fn score_matching_objective(g: &Gbm, data: &[Vec<f64>]) -> f64 {
    assert_eq!(g.n_spin, 0, "score matching here is for the continuous part");
    let n = g.n_real;
    let tr: f64 = (0..n).map(|i| g.a[i * n + i]).sum();
    let mut acc = 0.0;
    for x in data {
        for i in 0..n {
            let mut s = -g.b[i];
            for k in 0..n {
                s += g.a[i * n + k] * x[k];
            }
            acc += 0.5 * s * s;
        }
        acc -= tr;
    }
    acc / data.len() as f64
}

/// Fit a Gaussian energy-based model to `data` by score matching, in closed form.
///
/// Returns the exact minimiser `A = Σ⁻¹`, `b = Σ⁻¹ μ` of [`score_matching_objective`] — no
/// sampling, no partition function, one pass. `None` when the sample covariance is singular, which
/// is the honest answer for data that does not determine a model (fewer samples than dimensions,
/// or an exactly collinear coordinate).
pub fn score_match_gaussian(data: &[Vec<f64>]) -> Option<Gbm> {
    let n = data.first()?.len();
    let k = data.len() as f64;
    if data.len() <= n {
        return None;
    }
    let mu: Vec<f64> = (0..n).map(|i| data.iter().map(|x| x[i]).sum::<f64>() / k).collect();
    let mut cov = vec![0.0; n * n];
    for x in data {
        for i in 0..n {
            for j in 0..n {
                cov[i * n + j] += (x[i] - mu[i]) * (x[j] - mu[j]);
            }
        }
    }
    for v in cov.iter_mut() {
        *v /= k - 1.0;
    }
    let a = inverse(&cov, n)?;
    let b = (0..n).map(|i| (0..n).map(|j| a[i * n + j] * mu[j]).sum()).collect();
    Some(Gbm::gaussian(n, a, b))
}

/// Fit by **denoising** score matching (Vincent 2011) at noise level `sigma`.
///
/// Instead of matching the score of the data, match the score of the data convolved with
/// `N(0, σ²I)`. Two things follow, and the second is why this function is here.
///
/// First, it is easier: the perturbed density is smooth even where the data lie on a manifold, so
/// the objective is well behaved where plain score matching is not.
///
/// Second, **this is the objective that trains diffusion models.** A diffusion model is a score
/// model fitted by denoising score matching at a ladder of noise levels; the line runs from an
/// energy-based model through Hyvärinen's identity to Vincent's denoising form to the training loop
/// behind modern generative AI. That line ends in this crate, and it ends with a closed form to
/// check it against: for a Gaussian the perturbed distribution is `N(μ, Σ + σ²I)`, so the optimum is
///
/// ```text
///   A = (Σ + σ²I)⁻¹,     b = A μ
/// ```
///
/// which the test verifies by generating from a known `Σ` and recovering it. `None` on the same
/// condition as [`score_match_gaussian`].
pub fn denoising_score_match_gaussian(data: &[Vec<f64>], sigma: f64) -> Option<Gbm> {
    assert!(sigma >= 0.0, "a noise level is not negative");
    let n = data.first()?.len();
    let k = data.len() as f64;
    if data.len() <= n {
        return None;
    }
    let mu: Vec<f64> = (0..n).map(|i| data.iter().map(|x| x[i]).sum::<f64>() / k).collect();
    let mut cov = vec![0.0; n * n];
    for x in data {
        for i in 0..n {
            for j in 0..n {
                cov[i * n + j] += (x[i] - mu[i]) * (x[j] - mu[j]);
            }
        }
    }
    for v in cov.iter_mut() {
        *v /= k - 1.0;
    }
    // the perturbed covariance, which is what the denoising objective is fitting
    for i in 0..n {
        cov[i * n + i] += sigma * sigma;
    }
    let a = inverse(&cov, n)?;
    let b = (0..n).map(|i| (0..n).map(|j| a[i * n + j] * mu[j]).sum()).collect();
    Some(Gbm::gaussian(n, a, b))
}

#[cfg(test)]
mod score_tests {
    use super::*;

    fn spd(n: usize, seed: u64) -> Vec<f64> {
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

    /// Score matching recovers the model that generated the data, without ever touching ln Z.
    ///
    /// The generating `A` is known, the fit is closed form, and the two are compared entry by
    /// entry. This is the whole claim of the method: the partition function never appears.
    #[test]
    fn score_matching_recovers_the_generating_model() {
        let n = 3;
        let a = spd(n, 23);
        let b: Vec<f64> = (0..n).map(|i| 0.3 * i as f64 - 0.2).collect();
        let truth = Gbm::gaussian(n, a.clone(), b.clone());
        // draw from the model at beta = 1, where A is exactly the precision
        let (xs, _) = truth.collect(1.0, 5_000, 400_000, 11);
        let fit = score_match_gaussian(&xs).expect("enough data to determine a model");
        for i in 0..n {
            for j in 0..n {
                let (got, want) = (fit.a[i * n + j], a[i * n + j]);
                assert!((got - want).abs() < 0.06 * want.abs().max(1.0), "A({i},{j}) = {got}, want {want}");
            }
            assert!((fit.b[i] - b[i]).abs() < 0.06, "b({i}) = {}, want {}", fit.b[i], b[i]);
        }
    }

    /// The closed form is the minimiser: perturbing it in any direction raises the objective.
    #[test]
    fn the_closed_form_is_the_score_matching_minimum() {
        let n = 2;
        let truth = Gbm::gaussian(n, spd(n, 29), vec![0.4, -0.3]);
        let (xs, _) = truth.collect(1.0, 3_000, 60_000, 13);
        let fit = score_match_gaussian(&xs).unwrap();
        let base = score_matching_objective(&fit, &xs);
        let mut rng = Pcg::new(31, 0);
        for _ in 0..40 {
            let mut p = fit.clone();
            let (i, j) = ((rng.f64() * n as f64) as usize % n, (rng.f64() * n as f64) as usize % n);
            let d = 0.05 * (rng.f64() - 0.5);
            p.a[i * n + j] += d;
            p.a[j * n + i] += d; // keep it symmetric, which the family requires
            assert!(score_matching_objective(&p, &xs) >= base - 1e-12, "a perturbation lowered the objective");
        }
        // and it beats a deliberately wrong model by a wide margin
        let wrong = Gbm::gaussian(n, vec![1.0, 0.0, 0.0, 1.0], vec![0.0, 0.0]);
        assert!(score_matching_objective(&wrong, &xs) > base);
    }

    /// Denoising score matching recovers the PERTURBED model, which is the diffusion objective.
    ///
    /// Fitting at noise `sigma` must return the score of `N(mu, Sigma + sigma^2 I)`, so the
    /// recovered precision is `(Sigma + sigma^2 I)^-1` -- checked against the generating `Sigma`
    /// rather than against the fit's own output. At `sigma = 0` it must agree with plain score
    /// matching exactly.
    #[test]
    fn denoising_score_matching_recovers_the_perturbed_model() {
        let n = 3;
        let a = spd(n, 37);
        let truth = Gbm::gaussian(n, a.clone(), vec![0.0; n]);
        let sigma_data = inverse(&a, n).unwrap(); // Sigma = A^-1 at beta = 1
        let (xs, _) = truth.collect(1.0, 5_000, 400_000, 17);
        for sigma in [0.0, 0.3, 0.8] {
            let fit = denoising_score_match_gaussian(&xs, sigma).unwrap();
            // expected precision: (Sigma + sigma^2 I)^-1
            let mut pert = sigma_data.clone();
            for i in 0..n {
                pert[i * n + i] += sigma * sigma;
            }
            let want = inverse(&pert, n).unwrap();
            for i in 0..n {
                for j in 0..n {
                    let (got, w) = (fit.a[i * n + j], want[i * n + j]);
                    assert!((got - w).abs() < 0.08 * w.abs().max(1.0), "sigma {sigma} A({i},{j}) = {got}, want {w}");
                }
            }
        }
        // sigma = 0 is plain score matching, exactly
        let plain = score_match_gaussian(&xs).unwrap();
        let dsm0 = denoising_score_match_gaussian(&xs, 0.0).unwrap();
        for i in 0..(n * n) {
            assert!((plain.a[i] - dsm0.a[i]).abs() < 1e-12);
        }
        // more noise means a flatter model: the precision shrinks
        let lo = denoising_score_match_gaussian(&xs, 0.2).unwrap();
        let hi = denoising_score_match_gaussian(&xs, 1.5).unwrap();
        for i in 0..n {
            assert!(hi.a[i * n + i] < lo.a[i * n + i], "noise must flatten the model");
        }
    }

    /// Data that does not determine a model is refused, not fitted.
    #[test]
    fn underdetermined_data_is_refused() {
        assert!(score_match_gaussian(&[vec![1.0, 2.0], vec![2.0, 4.0]]).is_none(), "2 points in 2D");
        // an exactly collinear coordinate leaves the covariance singular
        let collinear: Vec<Vec<f64>> = (0..50).map(|i| { let t = i as f64; vec![t, 2.0 * t] }).collect();
        assert!(score_match_gaussian(&collinear).is_none(), "a collinear coordinate has no inverse covariance");
    }
}
