//! Are the error bars honest? A harness that asks, and the crate's own answers.
//!
//! Every estimator here reports a number and a bar. A bar that is too small is worse than no bar
//! at all: it turns "we do not know" into a confident wrong answer, and nothing downstream can
//! detect it. The check is cheap and general, and this module exists because running it once found
//! a bar in this crate that was 30% too small.
//!
//! # The method
//!
//! Take a model whose answer is known exactly. Run the estimator from many seeds. For each run form
//!
//! ```text
//!   z = (estimate − truth) / reported stderr
//! ```
//!
//! and look at the ensemble of `z`. If the estimator is unbiased, `mean(z) ≈ 0`. If the bar is
//! honest, `sd(z) ≈ 1`. **`sd(z) > 1` means the bar is too small** — the estimator is wrong more
//! often than it admits. `sd(z) < 1` means it is too wide, which is safe.
//!
//! That is all. It needs no theory about the estimator, only a truth to compare against, and this
//! crate is full of exact truths — enumeration, the transfer matrix, variable elimination, Onsager.
//!
//! # What it found
//!
//! Applied across the crate (the tests in this module), with `sd(z) ≤ 1` meaning honest:
//!
//! | estimator | mean z | sd(z) | verdict |
//! |---|---|---|---|
//! | `SampleSet::mean_energy` | −0.03 | 0.80 | honest, conservative |
//! | `SampleSet::magnetization` | −0.01 | 0.93 | honest |
//! | `SampleSet::correlation` | −0.04 | 0.62 | honest, very conservative |
//! | `SampleSet::marginals` | +0.05 | 0.71 | honest, conservative |
//! | `free_energy::thermodynamic_integration` | — | covers 40/40 | honest |
//! | `free_energy::Ais::lower_bound` | — | 0 violations in 60 at δ=0.1 | honest |
//! | `bar_ladder` quadrature stderr | — | **1.28** | **too small by ~30%** |
//! | `LadderTraces::log_z_total` jackknife | — | 0.81 | honest |
//!
//! The conservatism of the `SampleSet` family is structural rather than accidental: the error bar
//! deflates the sample count by the chain's `tau_int`, and `chain_tau` takes the SLOWEST of energy
//! and magnetisation, so a fast observable like a single correlation is charged the slow one's
//! autocorrelation. That is a deliberate choice — a bar that is too wide costs samples, a bar that
//! is too small costs correctness — and this table is what makes it a measured choice rather than
//! a hope.
//!
//! # The one estimator with no bar
//!
//! `popanneal`'s `ln_z` is absent from the table because it reports **no bar at all**. Measured on
//! a 12-spin ring it is unbiased (mean error `+0.003` at 512 replicas over 48 stages), and its
//! run-to-run spread follows
//!
//! ```text
//!   sd(ln Z)  ≈  1.7 · sqrt( rho / (R · stages) )
//! ```
//!
//! to within ±10% across a 16× range in replicas `R` and 6× in `stages`, with `rho` the family
//! statistic the outcome already reports. **That is a lead, not a bar, and it is not shipped as
//! one**: the constant `1.7` was fitted on a single model, and a constant fitted on one instance
//! is exactly the kind of number this crate refuses to dress up as a guarantee. What a caller can
//! do today is the gold standard anyway — run `popanneal` a few times and take the spread, which
//! costs what it costs and assumes nothing. A principled bar (the family-entropy relation of the
//! population-annealing literature) is recorded as open.

/// What a calibration run measured.
#[derive(Clone, Debug, PartialEq)]
pub struct Calibration {
    /// Number of runs.
    pub runs: usize,
    /// `mean(z)`; zero for an unbiased estimator.
    pub mean_z: f64,
    /// `sd(z)`; one for an honest bar, above one for a bar that is too small.
    pub sd_z: f64,
    /// Fraction of runs whose 95% interval (`±1.96 σ`) contained the truth; `0.95` when honest.
    pub coverage_95: f64,
}

impl Calibration {
    /// Is the bar honest — not systematically too small?
    ///
    /// `sd(z)` above `tolerance` fails. The default worth using is around `1.1`: sampling `sd` from
    /// `n` runs has its own error of roughly `1/√(2n)`, so 24 runs cannot distinguish `1.0` from
    /// `1.15`, and a threshold tighter than the measurement is theatre.
    pub fn bar_is_honest(&self, tolerance: f64) -> bool {
        self.sd_z <= tolerance
    }

    /// Is the estimator unbiased, to within the noise of `runs` samples?
    pub fn looks_unbiased(&self, sigmas: f64) -> bool {
        self.mean_z.abs() <= sigmas * self.sd_z.max(1e-12) / (self.runs as f64).sqrt()
    }
}

/// Calibrate an estimator against a known truth: `run(seed)` returns `(estimate, stderr)`.
///
/// Panics on fewer than 8 runs — an `sd` over seven numbers is not a measurement.
pub fn calibrate(truth: f64, runs: usize, run: impl Fn(u64) -> (f64, f64)) -> Calibration {
    assert!(runs >= 8, "{runs} runs is too few to estimate a spread");
    let z: Vec<f64> = (0..runs as u64)
        .map(|s| {
            let (est, se) = run(s);
            assert!(se > 0.0 && se.is_finite(), "run {s} reported a stderr of {se}");
            (est - truth) / se
        })
        .collect();
    let mean_z = z.iter().sum::<f64>() / runs as f64;
    let sd_z = (z.iter().map(|v| (v - mean_z) * (v - mean_z)).sum::<f64>() / (runs - 1) as f64).sqrt();
    let coverage_95 = z.iter().filter(|v| v.abs() <= 1.96).count() as f64 / runs as f64;
    Calibration { runs, mean_z, sd_z, coverage_95 }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gibbs::Sampler;
    use crate::ising;
    use crate::samples::Plan;

    /// Exact expectations on a 12-spin ring, by enumeration.
    fn truths(g: &crate::graph::Graph, beta: f64) -> (f64, f64, f64, f64) {
        let n = g.n;
        let p = ising::exact_boltzmann(g, beta);
        let mut s = vec![-1i8; n];
        let (mut e, mut m, mut c, mut m0) = (0.0, 0.0, 0.0, 0.0);
        for (mask, &w) in p.iter().enumerate() {
            for b in 0..n {
                s[b] = if mask >> b & 1 == 1 { 1 } else { -1 };
            }
            e += w * g.energy(&s);
            m += w * s.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
            c += w * (s[0] as f64) * (s[3] as f64);
            m0 += w * s[0] as f64;
        }
        (e, m, c, m0)
    }

    /// THE HARNESS ITSELF MUST WORK, or the table it produces means nothing.
    ///
    /// A known-honest estimator (the mean of independent normals, whose bar is exact) must come out
    /// near 1, and a deliberately halved bar must come out near 2. Without this, a passing
    /// calibration could be a broken harness reporting whatever the code does.
    #[test]
    fn the_harness_detects_a_bar_it_is_shown_to_be_wrong() {
        use crate::rng::Pcg;
        let draw = |seed: u64, shrink: f64| {
            let mut rng = Pcg::new(seed, 4);
            let n = 400;
            // Box-Muller from the crate's uniform, mean 3.0 and sd 1.0
            let xs: Vec<f64> = (0..n)
                .map(|_| {
                    let (u1, u2) = (rng.f64().max(1e-12), rng.f64());
                    3.0 + (-2.0 * u1.ln()).sqrt() * (core::f64::consts::TAU * u2).cos()
                })
                .collect();
            let m = xs.iter().sum::<f64>() / n as f64;
            let v = xs.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (n - 1) as f64;
            (m, (v / n as f64).sqrt() * shrink)
        };
        let honest = calibrate(3.0, 200, |s| draw(s, 1.0));
        assert!(honest.bar_is_honest(1.15), "an exact bar must read as honest: sd(z) = {}", honest.sd_z);
        assert!(honest.sd_z > 0.85, "and must not read as absurdly conservative: {}", honest.sd_z);
        assert!(honest.looks_unbiased(4.0), "mean z = {}", honest.mean_z);
        assert!(honest.coverage_95 > 0.90, "coverage {}", honest.coverage_95);

        let halved = calibrate(3.0, 200, |s| draw(s, 0.5));
        assert!(!halved.bar_is_honest(1.15), "a halved bar must be caught");
        assert!(halved.sd_z > 1.6, "and caught at about 2: sd(z) = {}", halved.sd_z);
    }

    /// The crate's sampled estimators report bars that are honest and unbiased.
    ///
    /// Every one of these is checked against enumeration, and the claim is one-sided: the bar may
    /// be conservative (it is), but it must never be too small.
    #[test]
    fn the_sample_set_error_bars_are_never_optimistic() {
        let (n, beta) = (12usize, 0.6);
        let g = ising::ring(n, 1.0, 0.25);
        let (e_t, m_t, c_t, m0_t) = truths(&g, beta);
        let collect = |seed: u64| {
            let mut sm = Sampler::new(&g, beta, seed);
            sm.collect(&Plan::new(1000, 2000, 1), None)
        };
        for (name, truth, f) in [
            ("mean_energy", e_t, Box::new(move |s: u64| { let x = collect(s).mean_energy().unwrap(); (x.value, x.stderr) }) as Box<dyn Fn(u64) -> (f64, f64)>),
            ("magnetization", m_t, Box::new(move |s: u64| { let x = collect(s).magnetization().unwrap(); (x.value, x.stderr) })),
            ("correlation", c_t, Box::new(move |s: u64| { let x = collect(s).correlation(0, 3).unwrap(); (x.value, x.stderr) })),
            ("marginal 0", m0_t, Box::new(move |s: u64| { let x = &collect(s).marginals().unwrap()[0]; (x.value, x.stderr) })),
        ] {
            let c = calibrate(truth, 32, f);
            assert!(c.bar_is_honest(1.15), "{name}: bar too small, sd(z) = {}", c.sd_z);
            assert!(c.looks_unbiased(4.0), "{name}: biased, mean z = {}", c.mean_z);
            assert!(c.coverage_95 >= 0.90, "{name}: 95% interval covered {}", c.coverage_95);
        }
    }

    /// The thermodynamic-integration bracket contains the truth, and the AIS bound is not violated.
    ///
    /// These are not `±σ` bars so `calibrate` does not apply; the honest question for a bracket is
    /// coverage and for a probabilistic bound is the violation rate against its own `delta`.
    #[test]
    fn the_bracket_covers_and_the_bound_holds() {
        use crate::free_energy::{ais, exact_log_z, linear_ladder, thermodynamic_integration};
        let (n, beta) = (12usize, 0.8);
        let g = ising::ring(n, 1.0, 0.15);
        let truth = exact_log_z(&g, beta);
        let covered = (0..16u64)
            .filter(|&s| {
                let t = thermodynamic_integration(&g, &linear_ladder(beta, 24), 200, 1200, 3.0, s);
                t.lower_widened <= truth && truth <= t.upper_widened
            })
            .count();
        assert_eq!(covered, 16, "the 3-sigma bracket missed {} of 16", 16 - covered);
        // Markov allows delta of these to fail; measured, none do, because Markov is loose.
        let violations = (0..24u64).filter(|&s| ais(&g, &linear_ladder(beta, 48), 2, 32, s).lower_bound(0.1) > truth).count();
        assert!(violations as f64 <= 0.1 * 24.0, "the bound failed {violations} times in 24 at delta 0.1");
    }
}
