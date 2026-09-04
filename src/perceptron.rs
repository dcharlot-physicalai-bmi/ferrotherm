//! The binary perceptron — Gardner's storage problem, where the couplings are spins.
//!
//! Gardner (1988) asked how many random patterns a perceptron can store, and answered it by
//! computing the volume of the space of couplings that classify them all. For CONTINUOUS couplings
//! on a sphere that volume shrinks to nothing at `α = P/N = 2`. For **binary** couplings
//! `J ∈ {±1}^N` — which is a spin configuration, so this crate's samplers search that space
//! natively — Krauth & Mézard (1989) put the capacity at `α_c ≈ 0.833`, proved rigorously by Ding
//! & Sun (2019).
//!
//! This module is that problem with its three layers separated, because they are three different
//! kinds of claim:
//!
//! | layer | what it is | in this module |
//! |---|---|---|
//! | first moment | a **theorem**: `E[Z] = 2^N p_sat^P`, so by Markov `P(Z ≥ 1) ≤ E[Z]` and no solutions survive `α > 1` | [`annealed_log_z`], [`annealed_capacity`] |
//! | replica | the *typical* count, `α_c ≈ 0.833` — cited, not derived here | [`KRAUTH_MEZARD_CAPACITY`] |
//! | enumeration | the exact count at this `N`, for these patterns | [`Perceptron::solution_count`] |
//!
//! The first moment is exact at every finite `N`, not only asymptotically, and the parity matters:
//! `p_sat = P(J·ξ > 0)` is `1/2` for odd `N`, but for even `N` a tie `J·ξ = 0` is a
//! misclassification, so `p_sat = (1 − C(N, N/2)/2^N)/2 < 1/2` and the annealed count is smaller.
//! [`p_sat`] is exact for both. Getting this wrong is how the textbook formula `2^N 2^{−P}` came
//! out 25% high against enumeration at `N = 10` while looking right.
//!
//! # The algorithmic gap, and where it is not
//!
//! The binary perceptron's solution space is *frozen*: at typical `α` the solutions are isolated
//! points separated by extensive Hamming distance, so a sampler descending the error count has
//! nothing to follow, and local algorithms are expected to fail well below `α_c`. Both sides are
//! measurable here — how many solutions exist ([`Perceptron::solution_count`]) and whether
//! annealing finds one ([`Perceptron::solve`]) — and the measurement says something the
//! expectation alone does not.
//!
//! **At every size enumeration can reach, there is no gap.** At `N = 15` and `N = 19`, across
//! loads from `0.2` to `1.1`, annealing found a solution in essentially every instance that had
//! one (313 of 314 solvable instances). The frozen structure is asymptotic; at `2^19` couplings
//! with thousands of solutions, a search does not need it to be otherwise.
//!
//! **The gap opens with `N`, and it opens fast.** The capacity is a constant, `0.833`, but the
//! load at which annealing still succeeds falls as the problem grows:
//!
//! ```text
//!   N     α=0.2   α=0.3   α=0.4   α=0.5   α=0.6   α=0.7      (20 instances, 3 restarts)
//!   21    20/20   20/20   20/20   20/20   20/20   14/20
//!   51    20/20   20/20   20/20   19/20   18/20    9/20
//!   101   20/20   20/20   19/20   17/20    4/20    0/20
//!   201   20/20   20/20   18/20    5/20    0/20    0/20
//!   401   20/20   19/20    3/20    0/20    0/20    0/20
//! ```
//!
//! At `α = 0.5` — where solutions are abundant, well below `α_c` — the success rate goes from 20
//! of 20 at `N = 21` to 0 of 20 at `N = 401`. That is the algorithmic gap: not a claim that the
//! instances became unsatisfiable, but a measurement that this algorithm's reach shrinks while
//! the satisfiable region does not.

use crate::rng::Pcg;

/// The largest `N` [`Perceptron::solution_count`] will enumerate: `2^N × P` field updates.
pub const MAX_ENUMERATED: usize = 24;

/// Krauth & Mézard's replica-symmetric capacity for binary couplings, `α_c ≈ 0.833`.
///
/// Cited (J. Phys. France 50, 3057, 1989; rigorous in Ding & Sun, Annals of Math 2019), not
/// derived here — the derivation is a one-step replica-symmetry-breaking calculation, and a
/// number this module cannot check is a number it should not claim to have computed.
pub const KRAUTH_MEZARD_CAPACITY: f64 = 0.833;

/// `P(J·ξ > 0)` for a fixed `J ∈ {±1}^N` and uniform `ξ`, exactly.
///
/// `1/2` for odd `N`. For even `N` a tie is a misclassification, so this is
/// `(1 − C(N, N/2)/2^N)/2`, strictly less.
pub fn p_sat(n: usize) -> f64 {
    if n == 0 {
        return 0.0;
    }
    if n % 2 == 1 {
        return 0.5;
    }
    // C(n, n/2) / 2^n, by a product that cannot overflow
    let mut tie = 1.0f64;
    for i in 0..(n / 2) {
        tie *= (n - i) as f64 / (n / 2 - i) as f64;
        tie *= 0.5;
    }
    tie *= 0.5f64.powi((n / 2) as i32);
    (1.0 - tie) / 2.0
}

/// `ln E[Z]` over random patterns: `N ln 2 + P ln p_sat(N)`. Exact at every finite `N`.
pub fn annealed_log_z(n: usize, p: usize) -> f64 {
    n as f64 * core::f64::consts::LN_2 + p as f64 * p_sat(n).ln()
}

/// The load where the annealed entropy vanishes, `ln 2 / −ln p_sat(N)` — exactly `1` for odd `N`.
///
/// **This is an upper bound on the true capacity, and a theorem**: `Z` is a non-negative integer,
/// so `P(Z ≥ 1) ≤ E[Z]`, and above this load `E[Z] → 0`. The true capacity is strictly smaller
/// ([`KRAUTH_MEZARD_CAPACITY`]) because the typical count is far below the mean — a few pattern
/// sets with very many solutions carry the average.
pub fn annealed_capacity(n: usize) -> f64 {
    core::f64::consts::LN_2 / -p_sat(n).ln()
}

/// A storage problem: `P` patterns to be classified `+1` by a binary coupling vector.
///
/// The desired outputs are absorbed by the gauge `ξ^μ ← σ^μ ξ^μ`, which is why there are no
/// targets here: classifying `ξ` as `σ` is classifying `σξ` as `+1`.
#[derive(Clone, Debug)]
pub struct Perceptron {
    pub patterns: Vec<Vec<i8>>,
    pub n: usize,
}

impl Perceptron {
    pub fn new(patterns: Vec<Vec<i8>>) -> Self {
        let n = patterns.first().map_or(0, |p| p.len());
        assert!(n > 0 && patterns.iter().all(|p| p.len() == n));
        Perceptron { patterns, n }
    }

    /// `P` random patterns of `n` spins.
    pub fn random(n: usize, p: usize, seed: u64) -> Self {
        Perceptron::new(crate::hopfield::random_patterns(n, p, seed))
    }

    pub fn load(&self) -> f64 {
        self.patterns.len() as f64 / self.n as f64
    }

    /// The stabilities `J·ξ^μ`, one per pattern.
    pub fn stabilities(&self, j: &[i8]) -> Vec<i32> {
        self.patterns.iter().map(|p| p.iter().zip(j).map(|(&a, &b)| a as i32 * b as i32).sum()).collect()
    }

    /// How many patterns `j` misclassifies. A tie counts as an error.
    pub fn errors(&self, j: &[i8]) -> usize {
        self.stabilities(j).iter().filter(|&&h| h <= 0).count()
    }

    /// Does `j` classify every pattern correctly?
    pub fn is_solution(&self, j: &[i8]) -> bool {
        self.errors(j) == 0
    }

    /// The exact number of solutions, by Gray-code enumeration over all `2^N` couplings.
    ///
    /// Each step flips one spin and updates the `P` stabilities, so the cost is `2^N × P` rather
    /// than `2^N × N × P`. Refuses above [`MAX_ENUMERATED`] rather than running for a day.
    pub fn solution_count(&self) -> Result<u64, String> {
        if self.n > MAX_ENUMERATED {
            return Err(format!("enumeration refuses {} spins, the limit is {MAX_ENUMERATED}", self.n));
        }
        let p = self.patterns.len();
        let mut j = vec![-1i8; self.n];
        let mut h: Vec<i32> = self.stabilities(&j);
        let mut bad = h.iter().filter(|&&v| v <= 0).count();
        let mut count = u64::from(bad == 0);
        for i in 1..(1u64 << self.n) {
            let b = i.trailing_zeros() as usize;
            let was = j[b];
            j[b] = -was;
            for mu in 0..p {
                let d = -2 * (was as i32) * (self.patterns[mu][b] as i32);
                let before = h[mu] <= 0;
                h[mu] += d;
                let after = h[mu] <= 0;
                match (before, after) {
                    (false, true) => bad += 1,
                    (true, false) => bad -= 1,
                    _ => {}
                }
            }
            count += u64::from(bad == 0);
        }
        Ok(count)
    }

    /// One heat-bath sweep on the error count at inverse temperature `beta`, stabilities cached.
    pub fn sweep(&self, beta: f64, j: &mut [i8], rng: &mut Pcg) {
        let p = self.patterns.len();
        let mut h = self.stabilities(j);
        for i in 0..self.n {
            // energy change of flipping spin i
            let mut delta = 0i32;
            for mu in 0..p {
                let d = -2 * (j[i] as i32) * (self.patterns[mu][i] as i32);
                let before = i32::from(h[mu] <= 0);
                let after = i32::from(h[mu] + d <= 0);
                delta += after - before;
            }
            let p_flip = 1.0 / (1.0 + (beta * delta as f64).exp());
            if rng.f64() < p_flip {
                for mu in 0..p {
                    h[mu] += -2 * (j[i] as i32) * (self.patterns[mu][i] as i32);
                }
                j[i] = -j[i];
            }
        }
    }

    /// Anneal from a random start and return the best coupling vector found and its error count.
    pub fn solve(&self, beta_min: f64, beta_max: f64, stages: usize, sweeps_per: usize, seed: u64) -> (Vec<i8>, usize) {
        let mut rng = Pcg::new(seed, 23);
        let mut j: Vec<i8> = (0..self.n).map(|_| if rng.f64() < 0.5 { -1 } else { 1 }).collect();
        let mut best = j.clone();
        let mut best_e = self.errors(&j);
        for k in 0..stages {
            let beta = beta_min + (beta_max - beta_min) * k as f64 / (stages.max(2) - 1) as f64;
            for _ in 0..sweeps_per {
                self.sweep(beta, &mut j, &mut rng);
                let e = self.errors(&j);
                if e < best_e {
                    best_e = e;
                    best.copy_from_slice(&j);
                    if best_e == 0 {
                        return (best, 0);
                    }
                }
            }
        }
        (best, best_e)
    }
}

// ---- the spherical case: continuous couplings, where Gardner's answer is exact ----------------
//
// Gardner's original question was about CONTINUOUS couplings on the sphere `|J|² = N`, and there
// the replica-symmetric calculation is exact -- no symmetry breaking, because the solution space
// is the intersection of half-spaces with a sphere, and that is connected. So the capacity has a
// closed form, and the algorithmic story is the opposite of the binary one above -- but not in the
// way a first guess suggests, and the measurement is worth stating precisely (N = 120, minover,
// 8 instances, `best margin` at the largest budget):
//
//   alpha   2k iters   20k iters   200k iters   best margin
//   1.5        7/8         8/8         8/8        +0.0253
//   1.8        0/8         6/8         7/8        +0.0087
//   1.9        0/8         3/8         5/8        +0.0085
//   2.0        0/8         0/8         4/8        +0.0037
//   2.2        0/8         0/8         0/8        -0.0505
//
// Below the capacity minover always gets there EVENTUALLY -- every failure is a budget, and more
// iterations always help, because the maximum margin is positive and the problem is convex. The
// budget diverges as `alpha` approaches 2 because that margin is going to zero. Above the capacity
// the converged margin is NEGATIVE and no budget changes it: the failure belongs to the model.
//
// That is the real contrast with the binary case, where the failure belongs to neither -- the
// margin is irrelevant because solutions are isolated points, and the failure is the ALGORITHM's,
// getting worse with N at a fixed load. Here the sign of minover's converged margin says which
// side of the transition an instance is on, which is a diagnosis the binary problem cannot offer.

/// The standard normal CDF, from [`crate::hopfield::erf`].
fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + crate::hopfield::erf(x / core::f64::consts::SQRT_2))
}

fn normal_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (core::f64::consts::TAU).sqrt()
}

/// Gardner's capacity for continuous couplings at stability margin `kappa` (1988):
///
/// ```text
///   1 / α_c(κ)  =  ∫_{−κ}^{∞} Dz (z + κ)²  =  (1 + κ²) Φ(κ) + κ φ(κ),
/// ```
///
/// the integral evaluated in closed form. At `κ = 0` this is exactly **2** — the classical result
/// that a perceptron stores two patterns per weight — and it decreases with the margin demanded.
/// Unlike [`KRAUTH_MEZARD_CAPACITY`] this is *computed*, not cited: the replica-symmetric solution
/// is exact for the spherical case, and the tests check the closed form against quadrature.
pub fn gardner_capacity(kappa: f64) -> f64 {
    1.0 / ((1.0 + kappa * kappa) * normal_cdf(kappa) + kappa * normal_pdf(kappa))
}

/// The same storage problem with couplings on the sphere `|J|² = N` instead of the hypercube.
#[derive(Clone, Debug)]
pub struct SphericalPerceptron {
    pub patterns: Vec<Vec<i8>>,
    pub n: usize,
}

impl SphericalPerceptron {
    pub fn new(patterns: Vec<Vec<i8>>) -> Self {
        let n = patterns.first().map_or(0, |p| p.len());
        assert!(n > 0 && patterns.iter().all(|p| p.len() == n));
        SphericalPerceptron { patterns, n }
    }

    pub fn random(n: usize, p: usize, seed: u64) -> Self {
        SphericalPerceptron::new(crate::hopfield::random_patterns(n, p, seed))
    }

    pub fn load(&self) -> f64 {
        self.patterns.len() as f64 / self.n as f64
    }

    /// The normalised stabilities `(J·ξ^μ) / (|J| √N)`, which is what `κ` is measured in.
    pub fn stabilities(&self, j: &[f64]) -> Vec<f64> {
        let norm = j.iter().map(|v| v * v).sum::<f64>().sqrt().max(f64::MIN_POSITIVE);
        let scale = 1.0 / (norm * (self.n as f64).sqrt());
        self.patterns.iter().map(|p| scale * p.iter().zip(j).map(|(&a, &b)| a as f64 * b).sum::<f64>()).collect()
    }

    /// The smallest normalised stability: positive iff `j` classifies every pattern.
    pub fn margin(&self, j: &[f64]) -> f64 {
        self.stabilities(j).into_iter().fold(f64::INFINITY, f64::min)
    }

    /// **Minover** (Krauth & Mézard 1987): repeatedly add the worst-classified pattern to the
    /// coupling vector. It converges to the maximum-stability solution, which is why it is the
    /// right algorithm to measure a capacity with — a failure is the problem's, not the search's.
    ///
    /// Returns the coupling vector and its margin.
    pub fn minover(&self, iters: usize) -> (Vec<f64>, f64) {
        let mut j = vec![0.0f64; self.n];
        for p in &self.patterns {
            for (v, &x) in j.iter_mut().zip(p) {
                *v += x as f64;
            }
        }
        for _ in 0..iters {
            let st = self.stabilities(&j);
            let (worst, _) = st.iter().enumerate().fold((0usize, f64::INFINITY), |(bi, bv), (i, &v)| if v < bv { (i, v) } else { (bi, bv) });
            for (v, &x) in j.iter_mut().zip(&self.patterns[worst]) {
                *v += x as f64;
            }
        }
        let m = self.margin(&j);
        (j, m)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The exact per-constraint probability, and the parity that the textbook formula hides.
    #[test]
    fn p_sat_is_exact_at_both_parities() {
        for n in [1usize, 3, 5, 9, 21, 101] {
            assert!((p_sat(n) - 0.5).abs() < 1e-15, "odd N has no ties: p_sat({n}) = {}", p_sat(n));
            assert!((annealed_capacity(n) - 1.0).abs() < 1e-12);
        }
        // N = 10: C(10,5)/2^10 = 252/1024 = 0.24609375, so p_sat = 0.376953125
        assert!((p_sat(10) - 0.376_953_125).abs() < 1e-12, "p_sat(10) = {}", p_sat(10));
        assert!((p_sat(4) - (1.0 - 6.0 / 16.0) / 2.0).abs() < 1e-15);
        for n in [2usize, 4, 10, 20] {
            assert!(p_sat(n) < 0.5 && annealed_capacity(n) < 1.0, "even N is strictly harder: {n}");
        }
    }

    /// Enumeration agrees with the annealed formula ON AVERAGE, which is what the formula claims.
    #[test]
    fn the_annealed_count_is_the_mean_of_the_enumerated_ones() {
        for (n, p) in [(9usize, 5usize), (11, 11), (10, 5)] {
            let sets = 400;
            let total: u64 = (0..sets).map(|s| Perceptron::random(n, p, 900 + s).solution_count().unwrap()).sum();
            let mean = total as f64 / sets as f64;
            let want = annealed_log_z(n, p).exp();
            // 400 pattern sets: the mean of a heavy-tailed count, so this is a loose but real check
            assert!(mean > 0.6 * want && mean < 1.6 * want, "N={n} P={p}: mean {mean} vs annealed {want}");
        }
    }

    /// Jensen, measured: the annealed entropy is above the typical (quenched) one, which is the
    /// whole reason the true capacity is 0.833 and not the first moment's 1.
    #[test]
    fn the_annealed_entropy_exceeds_the_typical_one() {
        let (n, p) = (11usize, 4usize); // low load, so nearly every set has solutions
        let sets = 300;
        let counts: Vec<u64> = (0..sets).map(|s| Perceptron::random(n, p, 1300 + s).solution_count().unwrap()).collect();
        assert!(counts.iter().all(|&c| c > 0), "at alpha = 0.36 every set should be solvable");
        let annealed = (counts.iter().sum::<u64>() as f64 / sets as f64).ln() / n as f64;
        let quenched = counts.iter().map(|&c| (c as f64).ln()).sum::<f64>() / sets as f64 / n as f64;
        assert!(annealed > quenched, "annealed {annealed} must exceed quenched {quenched}");
        assert!((annealed - annealed_log_z(n, p) / n as f64).abs() < 0.05);
    }

    /// The first-moment bound is a bound: above the annealed capacity, solutions are gone.
    #[test]
    fn no_solutions_survive_past_the_first_moment_bound() {
        let n = 13;
        let solvable = |p: usize, sets: u64| {
            (0..sets).filter(|&s| Perceptron::random(n, p, 2000 + s).solution_count().unwrap() > 0).count()
        };
        // alpha = 1.5 * annealed capacity: E[Z] = 2^13 * 2^-19 = 1/64, so by Markov at most 1.6%
        let above = solvable(19, 200);
        assert!(above <= 6, "{above} of 200 sets solvable at alpha 1.46, Markov allows about 3");
        // and well below it they nearly all are
        assert!(solvable(4, 100) >= 95, "at alpha 0.3 nearly every set is solvable");
    }

    /// At enumerable sizes annealing finds a solution whenever one exists: no gap here.
    #[test]
    fn at_enumerable_sizes_the_annealer_matches_enumeration() {
        let n = 15;
        let (mut solvable, mut agreed) = (0, 0);
        for p in [5usize, 9, 13] {
            for s in 0..12u64 {
                let per = Perceptron::random(n, p, 5000 + s + 97 * n as u64);
                let exists = per.solution_count().unwrap() > 0;
                let found = (0..3).any(|r| per.solve(0.05, 12.0, 60, 40, 700 + s * 10 + r).1 == 0);
                assert!(!found || exists, "the annealer returned a solution enumeration says is not there");
                if exists {
                    solvable += 1;
                    agreed += usize::from(found);
                }
            }
        }
        assert!(solvable >= 20, "the loads should leave plenty of solvable instances: {solvable}");
        assert!(agreed * 20 >= solvable * 19, "found {agreed} of {solvable} solvable instances");
    }

    /// ...and the gap opens with N: the same load goes from always-solved to never-solved.
    #[test]
    fn the_algorithmic_gap_opens_as_the_problem_grows() {
        let rate = |n: usize| {
            let p = (0.5 * n as f64).round() as usize;
            (0..12u64)
                .filter(|&s| {
                    let per = Perceptron::random(n, p, 8000 + s + 31 * n as u64);
                    (0..2).any(|r| per.solve(0.05, 12.0, 60, 25, 900 + s * 10 + r).1 == 0)
                })
                .count()
        };
        let (small, large) = (rate(21), rate(201));
        // The load is 0.5, well under the capacity, so a failure here is the ALGORITHM's limit and
        // not the model's -- which is the whole content of the test.
        let load = 0.5;
        assert!(load < KRAUTH_MEZARD_CAPACITY, "the test load must sit below the capacity");
        assert_eq!(small, 12, "at N = 21, alpha = {load} is easy: {small} of 12");
        assert!(large <= 3, "at N = 201 the same load should be mostly out of reach: {large} of 12");
    }

    /// Gardner's closed form is exactly 2 at zero margin, and matches quadrature elsewhere.
    #[test]
    fn the_gardner_capacity_is_two_and_matches_quadrature() {
        assert!((gardner_capacity(0.0) - 2.0).abs() < 1e-12, "alpha_c(0) = {}", gardner_capacity(0.0));
        // midpoint quadrature of 1/alpha_c = int_{-k}^inf Dz (z+k)^2
        let quad = |k: f64| {
            let (lo, hi, m) = (-k, -k + 40.0, 400_000);
            let h = (hi - lo) / m as f64;
            let s: f64 = (0..m).map(|i| {
                let z = lo + (i as f64 + 0.5) * h;
                normal_pdf(z) * (z + k) * (z + k) * h
            }).sum();
            1.0 / s
        };
        for k in [-0.5, 0.0, 0.25, 0.5, 1.0, 2.0] {
            let (c, q) = (gardner_capacity(k), quad(k));
            assert!((c - q).abs() < 1e-4 * q.max(1.0), "kappa {k}: closed {c} vs quadrature {q}");
        }
        // it falls with the margin demanded, and the binary capacity is far below the spherical one
        for w in [(-0.5, 0.0), (0.0, 0.5), (0.5, 1.0)] {
            assert!(gardner_capacity(w.0) > gardner_capacity(w.1));
        }
        assert!(KRAUTH_MEZARD_CAPACITY < gardner_capacity(0.0) / 2.0);
    }

    /// Below the capacity, minover's failures are a BUDGET; above it, they are the MODEL. The sign
    /// of the converged margin is the discriminator, and it is the diagnosis the binary problem
    /// cannot offer.
    #[test]
    fn minovers_failures_are_a_budget_below_the_capacity_and_the_model_above_it() {
        let n = 120;
        let margins = |alpha: f64, iters: usize| -> Vec<f64> {
            let p = (alpha * n as f64).round() as usize;
            (0..6u64).map(|s| SphericalPerceptron::random(n, p, 4000 + s).minover(iters).1).collect()
        };
        // below the capacity: more iterations only ever help, and every instance is solved
        let cheap = margins(1.5, 2_000);
        let rich = margins(1.5, 20_000);
        assert!(rich.iter().zip(&cheap).all(|(r, c)| r >= c), "more iterations never hurt below the capacity");
        assert!(rich.iter().all(|&m| m > 0.0), "alpha 1.5 < 2 is solvable, and minover gets there: {rich:?}");
        // above it: the converged margin is negative on every instance, and a 10x budget does not
        // move it across zero, because there is nothing to find
        let above = margins(2.2, 5_000);
        let above_rich = margins(2.2, 50_000);
        assert!(above.iter().all(|&m| m < 0.0) && above_rich.iter().all(|&m| m < 0.0), "alpha 2.2 > 2 has no solutions: {above_rich:?}");
        // the binary sampler at a load FAR below its own capacity fails for the other reason
        let bin = (0..8u64).filter(|&s| Perceptron::random(n, (0.5 * n as f64) as usize, 4100 + s).solve(0.05, 12.0, 60, 25, s).1 == 0).count();
        assert!(bin < 8, "binary annealing at alpha 0.5, N = {n} should already be missing some: {bin} of 8");
    }

    /// The flip cost the sampler uses is the true energy difference.
    #[test]
    fn the_cached_flip_cost_is_the_error_difference() {
        let per = Perceptron::random(15, 12, 77);
        let mut rng = Pcg::new(3, 0);
        let j: Vec<i8> = (0..15).map(|_| if rng.f64() < 0.5 { -1 } else { 1 }).collect();
        let h = per.stabilities(&j);
        for i in 0..15 {
            let mut t = j.clone();
            t[i] = -t[i];
            let brute = per.errors(&t) as i32 - per.errors(&j) as i32;
            let mut delta = 0i32;
            for mu in 0..per.patterns.len() {
                let d = -2 * (j[i] as i32) * (per.patterns[mu][i] as i32);
                delta += i32::from(h[mu] + d <= 0) - i32::from(h[mu] <= 0);
            }
            assert_eq!(delta, brute, "spin {i}");
        }
    }
}
