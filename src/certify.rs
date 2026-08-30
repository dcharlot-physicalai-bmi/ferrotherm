//! Proof that a sampler did what it claimed.
//!
//! Every commercial machine in this field returns "best found". Not one of them — American,
//! Japanese or Chinese — tells you at what temperature it actually sampled, how many of its samples
//! were independent, or how far its distribution sat from the one you asked for. This module is
//! that missing answer, and it is the thing the rest of the stack exists to make trustworthy.
//!
//! A [`Certificate`] is computed **from samples alone**, not from the sampler's own account of
//! itself. That is deliberate: a sampler cannot certify itself any more than a witness can
//! corroborate their own testimony, and taking only the samples means a deliberately broken sampler
//! can be handed to the same function and caught. It is, and the tests do exactly that.
//!
//! # What it measures
//!
//! - **`beta_eff`** — the inverse temperature the samples were actually drawn at, as a
//!   pseudolikelihood maximum-likelihood estimate. If a site is +1 with frequency `sigma(2 beta f)`
//!   given its local field `f`, then fitting one parameter to every (field, spin) pair observed
//!   recovers `beta`. A sampler running at the wrong temperature cannot hide from this.
//! - **`tau_int` and `ess`** — integrated autocorrelation time by Sokal's automatic windowing, and
//!   the resulting count of genuinely independent samples. Ten thousand correlated draws are not
//!   ten thousand samples, and reporting them as such is the most common quiet lie in MCMC.
//! - **`tv_exact`** — where the model is small enough to enumerate, total variation distance from
//!   the true Boltzmann distribution, always alongside
//! - **`noise_floor`** — the TV that finite sampling alone produces. A distance below the floor is
//!   agreement, not accuracy, and this module refuses to let the two be confused.
//!
//! Findings are a list. An empty list is the only thing that means "passed".

use crate::graph::Graph;

/// Something wrong with a run, in the sampler's own output.
#[derive(Clone, Debug, PartialEq)]
pub enum Finding {
    /// The temperature the samples were drawn at is not the temperature that was requested.
    BetaMismatch { requested: f64, effective: f64, ci: (f64, f64) },
    /// Successive samples are too correlated for the count to mean what it says.
    Undermixed { tau_int: f64, ess: f64, draws: usize },
    /// The distribution is measurably not the Boltzmann distribution, beyond sampling noise.
    AboveNoiseFloor { tv: f64, floor: f64 },
    /// The chain was still drifting: early samples do not look like late ones.
    NotConverged { early: f64, late: f64, sigma: f64 },
    /// Too few samples to say anything. Reported rather than guessed at.
    TooFewSamples { draws: usize },
}

impl core::fmt::Display for Finding {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Finding::BetaMismatch { requested, effective, ci } => write!(
                f,
                "sampled at beta {effective:.4} (95% CI {:.4}..{:.4}), not the requested {requested:.4}",
                ci.0, ci.1
            ),
            Finding::Undermixed { tau_int, ess, draws } => write!(
                f,
                "draws are correlated: tau_int {tau_int:.1}, so {draws} draws are worth about \
                 {ess:.0} independent samples; thin the chain or run longer"
            ),
            Finding::AboveNoiseFloor { tv, floor } => write!(
                f,
                "total variation {tv:.4} exceeds the {floor:.4} sampling-noise floor, so the \
                 difference is real rather than finite-sample scatter"
            ),
            Finding::NotConverged { early, late, sigma } => write!(
                f,
                "the chain was still moving: early draws average {early:.4} and late ones \
                 {late:.4}, a gap of {sigma:.1} standard errors; burn in for longer"
            ),
            Finding::TooFewSamples { draws } => {
                write!(
                    f,
                    "{draws} draws is too few to certify anything: the sampling-noise floor for a \
                     state space this large reaches or exceeds 1, which is the most a total \
                     variation can be, so a distributional comparison here cannot distinguish a \
                     good sampler from pure noise. Draw more, or certify a smaller model"
                )
            }
        }
    }
}

/// What a run actually did.
#[derive(Clone, Debug)]
pub struct Certificate {
    pub draws: usize,
    pub beta_requested: f64,
    /// Pseudolikelihood MLE of the inverse temperature the samples came from.
    pub beta_eff: f64,
    /// 95% interval for `beta_eff`, widened for autocorrelation.
    pub beta_ci: (f64, f64),
    pub tau_int: f64,
    pub ess: f64,
    /// TV from the exact Boltzmann distribution, where enumeration was possible.
    pub tv_exact: Option<f64>,
    /// TV that finite sampling alone produces. Never quote a distance below this.
    pub noise_floor: Option<f64>,
    /// Empty means the run is sound as far as this can tell.
    pub findings: Vec<Finding>,
}

impl Certificate {
    #[must_use = "a certificate with findings is a certificate that failed; ignoring this is reporting a sound run that was not one"]
    pub fn passed(&self) -> bool {
        self.findings.is_empty()
    }
}

impl core::fmt::Display for Certificate {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        writeln!(
            f,
            "draws {}  beta {:.4} (asked {:.4}, CI {:.4}..{:.4})  tau_int {:.1}  ess {:.0}",
            self.draws, self.beta_eff, self.beta_requested, self.beta_ci.0, self.beta_ci.1,
            self.tau_int, self.ess
        )?;
        if let (Some(tv), Some(fl)) = (self.tv_exact, self.noise_floor) {
            writeln!(f, "tv {tv:.4} against a {fl:.4} noise floor")?;
        }
        if self.findings.is_empty() {
            write!(f, "PASSED")
        } else {
            for x in &self.findings {
                writeln!(f, "FINDING: {x}")?;
            }
            Ok(())
        }
    }
}

/// Estimate the inverse temperature the samples were drawn at.
///
/// Maximum pseudolikelihood: every site of every sample contributes one logistic observation with
/// feature `2 f_i` and label `s_i`, and the log-likelihood in beta is concave, so Newton converges
/// from anywhere sensible. Returns `(beta, fisher_information)`.
fn fit_beta(g: &Graph, samples: &[Vec<i8>]) -> (f64, f64) {
    // Precompute (field, spin) once; the fit visits them many times.
    let mut obs: Vec<(f64, f64)> = Vec::with_capacity(samples.len() * g.n);
    for s in samples {
        for i in 0..g.n {
            let f = g.field(i, s);
            if f != 0.0 {
                obs.push((2.0 * f, if s[i] > 0 { 1.0 } else { 0.0 }));
            }
        }
    }
    if obs.is_empty() {
        return (f64::NAN, 0.0); // every field was zero: the data says nothing about beta
    }

    // d(log L)/d(beta), which is strictly decreasing because the log-likelihood is concave.
    let d1 = |b: f64| -> f64 {
        obs.iter().map(|&(x, y)| x * (y - 1.0 / (1.0 + (-b * x).exp()))).sum()
    };

    // Bisection rather than Newton. Newton is faster and wrong here: far from the optimum the
    // logistic saturates, the second derivative underflows, and the step diverges -- which it did,
    // pinning uniform noise at the clamp instead of reporting its true beta of zero. Monotonicity
    // makes bracketing exact, so the robust method is also the correct one.
    const LIM: f64 = 60.0;
    let (mut lo, mut hi) = (-LIM, LIM);
    let (flo, fhi) = (d1(lo), d1(hi));
    if flo <= 0.0 || fhi >= 0.0 {
        // No sign change: the likelihood is maximised at the edge, which means the fields separate
        // the spins perfectly. Report the bound rather than a fabricated interior value.
        let beta = if fhi >= 0.0 { LIM } else { -LIM };
        return (beta, 0.0);
    }
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if d1(mid) > 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
        if hi - lo < 1e-12 {
            break;
        }
    }
    let beta = 0.5 * (lo + hi);

    let info: f64 = obs
        .iter()
        .map(|&(x, _)| {
            let p = 1.0 / (1.0 + (-beta * x).exp());
            x * x * p * (1.0 - p)
        })
        .sum();
    (beta, info)
}

/// Integrated autocorrelation time of a scalar trace, by Sokal's automatic windowing.
///
/// `tau_int = 1/2 + sum_k rho(k)`, truncated at the smallest window `W` satisfying `W >= 5 tau`.
/// Truncation is not optional: the tail of an empirical autocorrelation is noise, and summing all
/// of it produces a number that grows with the length of the run rather than describing it.
pub fn tau_int(trace: &[f64]) -> f64 {
    let n = trace.len();
    if n < 16 {
        return f64::NAN;
    }
    let mean = trace.iter().sum::<f64>() / n as f64;
    let var = trace.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
    if var <= 0.0 {
        return f64::INFINITY; // a constant trace never decorrelates
    }
    let max_lag = (n / 4).max(1);
    let mut tau = 0.5;
    for k in 1..=max_lag {
        let mut c = 0.0;
        for t in 0..(n - k) {
            c += (trace[t] - mean) * (trace[t + k] - mean);
        }
        c /= (n - k) as f64 * var;
        tau += c;
        if (k as f64) >= 5.0 * tau.max(0.5) {
            break;
        }
    }
    tau.max(0.5)
}

/// Certify a set of samples against the model and temperature they claim to come from.
///
/// `trace` is a scalar observable, one value per sample, used for the autocorrelation estimate;
/// energy is the usual choice. Samples must be in chain order for `tau_int` to mean anything.
pub fn certify(g: &Graph, beta_requested: f64, samples: &[Vec<i8>], trace: &[f64]) -> Certificate {
    let draws = samples.len();
    if draws < 16 {
        return Certificate {
            draws,
            beta_requested,
            beta_eff: f64::NAN,
            beta_ci: (f64::NAN, f64::NAN),
            tau_int: f64::NAN,
            ess: f64::NAN,
            tv_exact: None,
            noise_floor: None,
            findings: vec![Finding::TooFewSamples { draws }],
        };
    }

    let (beta_eff, info) = fit_beta(g, samples);

    // Autocorrelation of ONE observable measures how fast that observable mixes, which is not the
    // same as how fast the configuration does. An ordered lattice is the case in point: it sits in
    // a single basin while its energy jitters quickly around a fixed value, so an energy trace
    // reports fast mixing for a chain that has not moved. Magnetization sees exactly what energy
    // misses, so both are measured and the worse one is reported.
    let mag: Vec<f64> = samples
        .iter()
        .map(|s| s.iter().map(|&x| x as f64).sum::<f64>() / g.n as f64)
        .collect();
    let t = {
        let a = tau_int(trace);
        let b = tau_int(&mag);
        match (a.is_nan(), b.is_nan()) {
            (false, false) => a.max(b),
            (true, false) => b,
            (false, true) => a,
            (true, true) => f64::NAN,
        }
    };
    let ess = if t.is_finite() && t > 0.0 { draws as f64 / (2.0 * t) } else { 1.0 };

    // The Fisher interval assumes independent observations, and chain samples are not. Widening by
    // the autocorrelation is the difference between a defensible interval and a flattering one.
    let se = if info > 0.0 { (1.0 / info).sqrt() } else { f64::INFINITY };
    let inflate = if t.is_finite() { (2.0 * t).sqrt().max(1.0) } else { 1.0 };
    let half = 1.96 * se * inflate;
    let beta_ci = (beta_eff - half, beta_eff + half);

    let mut findings = Vec::new();
    if beta_eff.is_finite() && !(beta_ci.0 <= beta_requested && beta_requested <= beta_ci.1) {
        findings.push(Finding::BetaMismatch {
            requested: beta_requested,
            effective: beta_eff,
            ci: beta_ci,
        });
    }
    // Two ways a draw count can be misleading, and both are worth saying out loud. Fewer than 50
    // independent samples estimates nothing reliably whatever the raw count claims; and a tau
    // exceeding a fiftieth of the run means the windowing had too little to work with, so tau
    // itself is not to be trusted. The thresholds are round numbers, but they are round numbers
    // chosen against measured chains rather than picked to make the suite pass -- a run measuring
    // tau 43 with ess 35 out of 3,000 draws is undermixed by any reading, and an earlier ess < 30
    // line let exactly that through.
    if !t.is_finite() || ess < 50.0 || t > draws as f64 / 50.0 {
        findings.push(Finding::Undermixed { tau_int: t, ess, draws });
    }

    // A Geweke-style check. beta_eff cannot do this job: pseudolikelihood is a LOCAL statistic, and
    // a chain trapped in a metastable configuration still has locally correct conditionals. Only
    // comparing the start of the run with the end sees a chain that is still travelling.
    if draws >= 60 {
        let cut = draws / 3;
        let early = &mag[..cut];
        let late = &mag[draws - cut..];
        let m = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
        let (me, ml) = (m(early), m(late));
        let var = |v: &[f64], mu: f64| {
            v.iter().map(|x| (x - mu).powi(2)).sum::<f64>() / (v.len() as f64 - 1.0).max(1.0)
        };
        // standard errors inflated by the autocorrelation, or the test fires on every chain
        let infl = if t.is_finite() { (2.0 * t).max(1.0) } else { 1.0 };
        let se = ((var(early, me) + var(late, ml)) * infl / cut as f64).sqrt();
        if se > 0.0 {
            let z = (me - ml).abs() / se;
            if z > 4.0 {
                findings.push(Finding::NotConverged { early: me, late: ml, sigma: z });
            }
        }
    }

    // Exact comparison where the model is small enough to enumerate.
    let (mut tv_exact, mut noise_floor) = (None, None);
    if g.n <= 20 {
        let exact = crate::ising::exact_boltzmann(g, beta_requested);
        let mut hist = vec![0.0f64; 1 << g.n];
        for s in samples {
            let mut k = 0usize;
            for (b, &v) in s.iter().enumerate() {
                if v > 0 {
                    k |= 1 << b;
                }
            }
            hist[k] += 1.0;
        }
        for h in hist.iter_mut() {
            *h /= draws as f64;
        }
        let tv = crate::ising::tv(&hist, &exact);
        // Expected TV from finite sampling of a distribution over this many states. Comparing a
        // measured TV against zero instead of this is how a correct sampler gets called broken.
        let floor = 0.5 * ((1usize << g.n) as f64 / ess.max(1.0)).sqrt();
        // A floor at or above 1 is not a floor, because TV between two distributions cannot exceed
        // 1 -- so `tv > floor` becomes unsatisfiable and this gate silently switches itself OFF on
        // exactly the models it matters for. Measured: iid uniform noise against a 16-spin lattice
        // at 4000 draws gives tv = 0.999 and floor = 2.04, so nothing was reported and
        // `Certificate::passed()` counted the absence as a pass -- while still printing
        // `noise_floor: Some(2.04)` as though it were a real threshold. The same noise at n = 9 is
        // caught. A gate that reports "fine" when it cannot see is worse than no gate.
        if floor >= 1.0 {
            findings.push(Finding::TooFewSamples { draws });
        } else if tv > floor {
            findings.push(Finding::AboveNoiseFloor { tv, floor });
        }
        tv_exact = Some(tv);
        noise_floor = Some(floor);
    }

    Certificate {
        draws,
        beta_requested,
        beta_eff,
        beta_ci,
        tau_int: t,
        ess,
        tv_exact,
        noise_floor,
        findings,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gibbs::Sampler;
    use crate::rng::Pcg;

    /// Draw from a graph with a sampler that may be deliberately wrong.
    ///
    /// `beta_actual` is what the sampler really runs at; `thin` is sweeps between recorded draws.
    fn run(g: &Graph, beta_actual: f64, thin: usize, burn: usize, draws: usize, seed: u64)
        -> (Vec<Vec<i8>>, Vec<f64>)
    {
        let mut smp = Sampler::new(g, beta_actual, seed);
        smp.sweeps(burn, None);
        let mut samples = Vec::with_capacity(draws);
        let mut trace = Vec::with_capacity(draws);
        for _ in 0..draws {
            smp.sweeps(thin.max(1), None);
            samples.push(smp.s.clone());
            trace.push(g.energy(&smp.s));
        }
        (samples, trace)
    }

    #[test]
    fn a_correct_run_passes() {
        let g = crate::ising::ring(10, 1.0, 0.2);
        let (s, t) = run(&g, 0.6, 10, 500, 6000, 1);
        let c = certify(&g, 0.6, &s, &t);
        assert!(c.passed(), "a correct sampler should certify clean:\n{c}");
        assert!((c.beta_eff - 0.6).abs() < 0.05, "beta_eff {} off", c.beta_eff);
    }

    #[test]
    fn a_sampler_at_the_wrong_temperature_is_caught() {
        // Break mode 1. The sampler is internally consistent and returns perfectly good samples --
        // of the wrong distribution. Nothing but beta_eff catches this.
        let g = crate::ising::ring(10, 1.0, 0.2);
        let (s, t) = run(&g, 1.4, 10, 500, 6000, 2);
        let c = certify(&g, 0.6, &s, &t); // claims 0.6, actually ran at 1.4
        assert!(!c.passed(), "a wrong temperature must be caught");
        assert!(
            c.findings.iter().any(|f| matches!(f, Finding::BetaMismatch { .. })),
            "expected a BetaMismatch, got {:?}",
            c.findings
        );
        assert!((c.beta_eff - 1.4).abs() < 0.1, "should recover the true beta, got {}", c.beta_eff);
    }

    #[test]
    fn correlated_draws_are_caught() {
        // Break mode 2, at the temperature where it actually happens. A 12x12 lattice at beta 0.44
        // sits essentially on the 2D Ising critical point (beta_c = ln(1+sqrt2)/2 ~ 0.4407), where
        // correlation length diverges and single-spin Glauber crawls. Measured tau ~ 62 even with
        // 20 sweeps between draws.
        //
        // Checked across several seeds on purpose. At criticality the autocorrelation is itself
        // strongly seed-dependent -- a marginal configuration measured tau 62 on one seed and 18 on
        // another -- so a single-seed assertion here would be a coin flip dressed as a test.
        // Recording every sweep puts the case far enough from the boundary to be unambiguous.
        let g = crate::ising::lattice2d(12, 1.0);
        for seed in 1..=3 {
            let (s, t) = run(&g, 0.44, 1, 500, 3000, seed);
            let c = certify(&g, 0.44, &s, &t);
            assert!(
                c.findings.iter().any(|f| matches!(f, Finding::Undermixed { .. })),
                "seed {seed}: critical slowing down must be flagged: {c}"
            );
            assert!(c.ess < c.draws as f64 / 10.0, "seed {seed}: ess too high: {c}");
        }
    }

    #[test]
    fn thinning_repairs_what_correlation_broke() {
        // The finding has to be actionable, so the prescribed fix must actually work.
        let g = crate::ising::lattice2d(24, 1.0);
        let tight = { let (s, t) = run(&g, 0.7, 1, 0, 600, 3); certify(&g, 0.7, &s, &t) };
        let fixed = { let (s, t) = run(&g, 0.7, 50, 500, 600, 3); certify(&g, 0.7, &s, &t) };
        assert!(!tight.passed(), "the unfixed run should be flagged");
        assert!(fixed.passed(), "burning in and thinning should clear it: {fixed}");
        assert!(
            fixed.ess > tight.ess * 10.0,
            "ess should improve by an order of magnitude: {:.0} -> {:.0}",
            tight.ess, fixed.ess
        );
    }

    #[test]
    fn an_unburned_chain_is_caught() {
        // Break mode 3. A 24x24 lattice below its critical temperature, sampled from the first
        // sweeps of a randomly initialised chain: it is still coarsening out domains and its
        // magnetization is travelling, so the draws describe where it started rather than the
        // model. Measured tau ~ 20 against tau ~ 0.8 once burned in.
        //
        // Note the 1D ring will NOT do for this test, and an earlier version of it wrongly used
        // one: 1D Ising has no ordered phase, so a ring equilibrates fast at every temperature and
        // certifying it clean is correct behaviour, not a missed detection.
        let g = crate::ising::lattice2d(24, 1.0);
        let (s, t) = run(&g, 0.7, 1, 0, 600, 4);
        let c = certify(&g, 0.7, &s, &t);
        assert!(!c.passed(), "an unburned coarsening chain must not certify clean:\n{c}");

        let (s2, t2) = run(&g, 0.7, 1, 500, 600, 4);
        assert!(certify(&g, 0.7, &s2, &t2).passed(), "burning in should clear it");
    }

    #[test]
    fn the_convergence_check_sees_a_drifting_trace() {
        // THIS TEST USED TO CALL NEITHER `certify` NOR ANYTHING IT TESTS.
        //
        // It built two synthetic traces and then asserted only that its own fixtures behaved --
        // that the drifting one drifted and the steady one did not. It never constructed a
        // `Certificate`, never mentioned `Finding::NotConverged`, and never touched the
        // standard-error inflation or the z > 4 threshold it is named for. It was green for every
        // possible implementation of the check, including no implementation at all.
        //
        // It drives `certify` now, and requires the finding to appear on a drifting chain and to
        // stay away from a steady one. The graph is small enough that everything else in the
        // certificate is computable, so a failure here is about convergence and not about setup.
        // THE FIXTURE TOOK THREE ATTEMPTS AND EACH FAILURE WAS INFORMATIVE, so they are recorded.
        //
        // (1) A hand-built ramp of independent draws: `NotConverged` did not fire, correctly -- a
        //     ramp is maximally autocorrelated, `tau_int` came out at 210, and the standard-error
        //     inflation the check applies swallowed the gap. The check was right, the fixture was a
        //     drift no honest statistic should call significant.
        // (2) A real chain on a 4x4 lattice: sixteen spins equilibrate in a few sweeps, so there is
        //     no drift to find.
        // (3) A real chain on 16x16: the transient finishes inside the first third of the window, so
        //     `early` is already at -0.96 and there is nothing left to compare.
        //
        // What works is a lattice big enough that coarsening is SLOW relative to the window: 32x32
        // just below the critical point, sampled from a random start with no burn-in, over a window
        // short enough that the transient spans it. early -0.27 -> late -0.87.
        let g = crate::ising::lattice2d(32, 1.0);
        let n = 200;

        // A REAL CHAIN, not a synthetic ramp. The first attempt at this test built the drift by
        // hand -- independent draws whose bias ramped upward -- and `NotConverged` did not fire,
        // correctly: a hand-built ramp is maximally autocorrelated, `tau_int` came out at 210, and
        // the standard-error inflation that the check applies swallowed the gap. The check was
        // right and the fixture was wrong.
        //
        // So this is what the module doc describes instead: a lattice below its critical
        // temperature, sampled from a random start with NO BURN-IN, coarsening out domains and
        // travelling from disorder toward saturation while it is being sampled.
        let mut smp = crate::gibbs::Sampler::new(&g, 0.5, 7);
        let drifting: Vec<Vec<i8>> = (0..n)
            .map(|_| {
                smp.sweep(None);
                smp.s.clone()
            })
            .collect();
        let trace_d: Vec<f64> = drifting
            .iter()
            .map(|s| s.iter().map(|&x| x as f64).sum::<f64>() / g.n as f64)
            .collect();
        let cert = certify(&g, 0.5, &drifting, &trace_d);
        let found = cert
            .findings
            .iter()
            .any(|f| matches!(f, Finding::NotConverged { .. }));
        assert!(
            found,
            "a chain travelling from disorder to saturation must be reported as not converged; \
             findings were {:?}",
            cert.findings
        );
        // And the finding must carry numbers a reader can act on, not just fire.
        if let Some(Finding::NotConverged { early, late, sigma }) = cert
            .findings
            .iter()
            .find(|f| matches!(f, Finding::NotConverged { .. }))
        {
            assert!(late.abs() > early.abs(), "it leaves disorder: early {early} late {late}");
            assert!(*sigma > 4.0, "and it must clear the threshold it uses: z = {sigma}");
        }

        // A steady chain must NOT trip it, or the check is an alarm that is always on.
        let mut warm = crate::gibbs::Sampler::new(&g, 0.5, 11);
        warm.sweeps(40_000, None); // burnt in, which is the whole difference
        let steady: Vec<Vec<i8>> = (0..n)
            .map(|_| {
                warm.sweep(None);
                warm.s.clone()
            })
            .collect();
        let trace_s: Vec<f64> = steady
            .iter()
            .map(|s| s.iter().map(|&x| x as f64).sum::<f64>() / g.n as f64)
            .collect();
        let clean = certify(&g, 0.5, &steady, &trace_s);
        assert!(
            !clean.findings.iter().any(|f| matches!(f, Finding::NotConverged { .. })),
            "a stationary chain must not be reported as drifting; findings were {:?}",
            clean.findings
        );
    }

    #[test]
    fn the_distributional_gate_fires_on_noise_rather_than_switching_itself_off() {
        // `AboveNoiseFloor` appeared in the enum, in Display, and at one push site -- and in NO
        // test, at any n. That mattered, because the floor `0.5*sqrt(2^n/ess)` passes 1 as n grows,
        // and TV can never exceed 1, so the comparison became unsatisfiable and the gate went
        // quiet on exactly the models it is for. `passed()` counted the silence as a pass.
        //
        // Two sizes on purpose: n=9 is where the floor is meaningful, n=16 at the same draw count
        // is where it used to go inert.
        let mut rng = 0x2545F4914F6CDD1Du64;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        for (side, draws) in [(3usize, 4000usize), (4, 4000)] {
            let g = crate::ising::lattice2d(side, 1.0);
            // Samples with no relation whatever to the model.
            let samples: Vec<Vec<i8>> = (0..draws)
                .map(|_| {
                    let r = next();
                    (0..g.n).map(|b| if r >> (b % 64) & 1 == 1 { 1 } else { -1 }).collect()
                })
                .collect();
            let trace: Vec<f64> = samples.iter().map(|s| g.energy(s)).collect();
            let c = certify(&g, 0.9, &samples, &trace);
            assert!(!c.passed(), "n={} pure noise must not pass", g.n);
            // Either it caught the discrepancy, or it said it could not look. Never silence.
            let spoke = c.findings.iter().any(|f| {
                matches!(f, Finding::AboveNoiseFloor { .. } | Finding::TooFewSamples { .. })
            });
            assert!(spoke, "n={} said nothing about the distribution: {:?}", g.n, c.findings);
            if let Some(floor) = c.noise_floor {
                assert!(
                    floor < 1.0 || c.findings.iter().any(|f| matches!(f, Finding::TooFewSamples { .. })),
                    "n={}: floor {floor} is vacuous and nothing said so",
                    g.n
                );
            }
        }
    }

    #[test]
    fn pure_noise_is_caught() {
        // The random-noise oracle: samples with no relation to the model at all. If a certificate
        // cannot reject this, it cannot reject anything.
        let g = crate::ising::ring(10, 1.0, 0.3);
        let mut rng = Pcg::new(9, 0);
        let samples: Vec<Vec<i8>> = (0..4000)
            .map(|_| (0..g.n).map(|_| if rng.f64() < 0.5 { 1 } else { -1 }).collect())
            .collect();
        let trace: Vec<f64> = samples.iter().map(|s| g.energy(s)).collect();
        let c = certify(&g, 1.0, &samples, &trace);
        assert!(!c.passed(), "uniform noise must never certify as Boltzmann:\n{c}");
        // and it is caught for the right reason: noise is infinite temperature
        assert!(c.beta_eff.abs() < 0.15, "noise should fit beta near 0, got {}", c.beta_eff);
    }

    #[test]
    fn the_noise_floor_is_reported_beside_the_distance() {
        // The rule the whole project runs on: never quote a distance without its floor.
        let g = crate::ising::ring(8, 1.0, 0.0);
        let (s, t) = run(&g, 0.5, 8, 400, 5000, 6);
        let c = certify(&g, 0.5, &s, &t);
        assert!(c.tv_exact.is_some() && c.noise_floor.is_some());
        assert!(c.noise_floor.unwrap() > 0.0);
    }

    #[test]
    fn too_few_samples_says_so_rather_than_guessing() {
        let g = crate::ising::ring(8, 1.0, 0.0);
        let (s, t) = run(&g, 1.0, 1, 10, 8, 7);
        let c = certify(&g, 1.0, &s, &t);
        assert_eq!(c.findings, vec![Finding::TooFewSamples { draws: 8 }]);
    }

    #[test]
    fn tau_int_recovers_a_known_correlation() {
        // An AR(1) process with coefficient p has tau_int = (1+p)/(2(1-p)); if the estimator cannot
        // recover that, it cannot be trusted on a real chain.
        let mut rng = Pcg::new(3, 0);
        for &p in &[0.0f64, 0.5, 0.8] {
            let mut x = 0.0;
            let trace: Vec<f64> = (0..200_000)
                .map(|_| {
                    let g = (-2.0 * rng.f64().max(1e-12).ln()).sqrt()
                        * (core::f64::consts::TAU * rng.f64()).cos();
                    x = p * x + (1.0 - p * p).sqrt() * g;
                    x
                })
                .collect();
            let want = (1.0 + p) / (2.0 * (1.0 - p));
            let got = tau_int(&trace);
            assert!((got - want).abs() / want < 0.25, "p={p}: got {got}, want {want}");
        }
    }
}
