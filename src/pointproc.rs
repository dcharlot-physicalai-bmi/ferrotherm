//! Sampling a Boltzmann law with spikes that live a fixed time: the point-process sampler of
//! Stewart and Sahani, and the birth–death process it generalises.
//!
//! # The construction
//!
//! Stewart & Sahani (arXiv:2603.09089, 2026) build, *"for any target multivariate count distribution
//! with downward-closed support, a multivariate temporal point process whose event-count vector in a
//! fixed-length sliding window converges in distribution to the target"*. For a binary target — a
//! p-bit network, `s_i ∈ {−1, +1}` read as a count `c_i = (s_i + 1)/2` — it is a spiking network: a
//! unit with no live spike emits one at rate `η · f(c + e_i)/f(c)`, and a spike stays live for a
//! FIXED time `m`, then expires. With the base intensity constant, `η = 1/m`, and `f = e^{−βE}`, the
//! birth rate is `(1/m) e^{2β f_i}` for the local field `f_i`, and their Theorem 1 makes the Boltzmann
//! law the limit of the live-spike pattern. The deterministic lifetime is the whole idea: *"the
//! sampler exhibits a discrete form of momentum that suppresses random-walk behaviour"*.
//!
//! Replace the fixed lifetime with an exponential one of the same mean and the process is a
//! continuous-time birth–death chain, reversible with respect to the same law (births
//! `(1/m) e^{2β f_i}`, deaths `1/m`). The paper's abstract: *"The introduction of auxiliary
//! randomness reduces the sampler to a birth-death process, establishing the latter as a degenerate
//! case with the same limiting distribution."* Both live here as [`Lifetime`], simulated event by
//! event with no time step.
//!
//! # What is exact, and what is measured
//!
//! A single unit is an alternating renewal process — off for an exponential time, on for `m` — so its
//! on-fraction is `f(1) / (f(0) + f(1))` exactly, for either lifetime; the test holds both to it.
//!
//! **Its efficiency is exact too, and the answer is a factor of exactly two.** Off for a time of mean
//! `mu0` and standard deviation `sd0`, on for mean `mu1` and deviation `sd1`, the renewal-reward
//! central limit theorem gives the long-run variance of the time-averaged on-indicator,
//! `lim T Var(A_T)`, as
//!
//! ```text
//!   sigma^2 = (mu0^2 sd1^2 + mu1^2 sd0^2) / (mu0 + mu1)^3
//! ```
//!
//! ([`renewal_variance`]: a cycle off for `U` and on for `V` earns `V` against its share `p (U + V)` of
//! the mean, `p = mu1 / (mu0 + mu1)`; the difference `(1 - p) V - p U` has variance
//! `(1 - p)^2 sd1^2 + p^2 sd0^2`, and dividing by the mean cycle length gives the above). The wait is
//! exponential, `sd0 = mu0`, so a lifetime with coefficient of variation `c` has
//! `sigma^2 = mu0^2 mu1^2 (1 + c^2) / (mu0 + mu1)^3`: **an exponential lifetime (`c = 1`) has exactly
//! twice the asymptotic variance of a fixed one (`c = 0`), at every field**, and so twice the
//! integrated autocorrelation time — `tau = m (1 - p)` against `m (1 - p) / 2`. Any lifetime costs
//! `1 + c^2` relative to a fixed one; uniform on `[0, 2m]` costs `4/3` (the formula checked by a numpy
//! simulation outside the tree, 2026-09-27; the two lifetimes simulated here are held to it by test).
//! [`unit_variance`], [`unit_tau`] and [`unit_autocorrelation`] give one unit exactly.
//!
//! The fixed lifetime's autocorrelation has a NEGATIVE lobe — at zero field it is `2 e^{-t/m} - 1` out
//! to `t = m`, crossing zero at `m ln 2` — and that is exactly what Sokal's window, which
//! [`crate::certify::tau_int`] uses, is not built for: on the exact autocorrelation it misreads the
//! fixed lifetime's `tau` by `-8.1%` at zero field and `+13.3%` at `beta h = 0.7`. The comparison is
//! therefore measured WITHOUT a window: [`PointProcess::batch_means`] integrates observables exactly
//! along the path over consecutive batches, and [`long_run_variance`] turns the batch means into
//! `sigma^2` and its standard error, the estimator the one-unit tests hold to the exact variance. For
//! coupled units the law is held to enumeration statistically, from long runs, and the comparison the
//! paper makes (efficiency against birth–death) is measured by `examples/pointproc_exact.rs`.

use crate::graph::Graph;
use crate::rng::Pcg;

/// How long a spike stays live.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lifetime {
    /// Exactly `m`: the point-process sampler.
    Fixed,
    /// Exponential with mean `m`: the birth–death process.
    Exponential,
}

impl Lifetime {
    /// The lifetime's coefficient of variation, standard deviation over mean: `0` fixed, `1`
    /// exponential.
    #[must_use]
    pub fn cv(self) -> f64 {
        match self {
            Lifetime::Fixed => 0.0,
            Lifetime::Exponential => 1.0,
        }
    }
}

/// The long-run variance `lim T Var(A_T)` of the time-averaged on-indicator `A_T` of an alternating
/// renewal process: off for independent times of mean `mu_off` and standard deviation `sd_off`, on
/// for mean `mu_on` and deviation `sd_on`. By the renewal-reward central limit theorem,
/// `(mu_off^2 sd_on^2 + mu_on^2 sd_off^2) / (mu_off + mu_on)^3`; the module documentation derives it.
#[must_use]
pub fn renewal_variance(mu_off: f64, sd_off: f64, mu_on: f64, sd_on: f64) -> f64 {
    let cycle = mu_off + mu_on;
    (mu_off * mu_off * sd_on * sd_on + mu_on * mu_on * sd_off * sd_off) / (cycle * cycle * cycle)
}

/// The mean off time of one unit in field `h`: the wait for a birth at rate `(1/m) e^{2 beta h}`.
fn unit_wait(beta: f64, h: f64, m: f64) -> f64 {
    m * (-2.0 * beta * h).exp()
}

/// The exact long-run variance of ONE unit's time-averaged on-indicator, `lim T Var(A_T)`, in field
/// `h`: [`renewal_variance`] with an exponential wait of mean `m e^{-2 beta h}` and a lifetime of
/// mean `m`. An exponential lifetime gives exactly twice a fixed one, at every field.
#[must_use]
pub fn unit_variance(beta: f64, h: f64, m: f64, lifetime: Lifetime) -> f64 {
    let wait = unit_wait(beta, h, m);
    renewal_variance(wait, wait, m, lifetime.cv() * m)
}

/// The exact integrated autocorrelation time of one unit's on-indicator, `int_0^inf rho(t) dt`, in
/// the units of `m`: [`unit_variance`] over twice the indicator's variance `p (1 - p)`. It is
/// `m (1 - p)` for the exponential lifetime and exactly half that for the fixed one, where
/// `p = 1 / (1 + e^{-2 beta h})` is the on-fraction.
#[must_use]
pub fn unit_tau(beta: f64, h: f64, m: f64, lifetime: Lifetime) -> f64 {
    let wait = unit_wait(beta, h, m);
    let p = m / (wait + m);
    unit_variance(beta, h, m, lifetime) / (2.0 * p * (1.0 - p))
}

/// `P(N <= k - 1)` for `N` Poisson with mean `mu`, summed in logs so a large mean cannot underflow
/// the first term to zero.
fn poisson_below(k: usize, mu: f64) -> f64 {
    if k == 0 {
        return 0.0;
    }
    if mu <= 0.0 {
        return 1.0;
    }
    let (mut sum, mut log_fact) = (0.0, 0.0);
    for j in 0..k {
        if j > 0 {
            log_fact += (j as f64).ln();
        }
        sum += (-mu + j as f64 * mu.ln() - log_fact).exp();
    }
    sum.min(1.0)
}

/// `E[(x - S_k)^+]` for `S_k` the sum of `k >= 1` exponential waits of rate `rate`: the integral of
/// their Erlang distribution function up to `x`, `x F_k(x) - (k / rate) F_{k+1}(x)`.
fn erlang_excess(k: usize, rate: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    let cdf = |j: usize| 1.0 - poisson_below(j, rate * x);
    x * cdf(k) - k as f64 / rate * cdf(k + 1)
}

/// The exact autocorrelation of one unit's on-indicator at lag `t`, stationary, in field `h`.
///
/// Exponential lifetime: a two-state Markov chain switching on at rate `1/mu0` and off at `1/m`, so
/// `rho(t) = e^{-|t| (1/mu0 + 1/m)}` -- computed from the rates, not from [`unit_tau`], so that the
/// test integrating it to [`unit_tau`] checks two derivations against each other. Fixed lifetime:
/// given on at time 0 the spike's remaining life `R` is uniform on
/// `[0, m]`, and the `k`-th spike after it lives on `[R + (k-1) m + S_k, R + k m + S_k)` for `S_k` the
/// sum of `k` exponential waits, so
///
/// ```text
///   P(on at t | on at 0) = (1 - t/m)^+ + (1/m) sum_{k >= 1} [H_k(t - (k-1)m) - 2 H_k(t - km) + H_k(t - (k+1)m)],
/// ```
///
/// with `H_k(x) = E[(x - S_k)^+]`, a finite sum at any `t`; and `rho = (P - p) / (1 - p)`. At zero
/// field and `t <= m` this is `2 e^{-t/m} - 1`, which goes NEGATIVE at `t = m ln 2`.
#[must_use]
pub fn unit_autocorrelation(beta: f64, h: f64, m: f64, lifetime: Lifetime, t: f64) -> f64 {
    let t = t.abs();
    let wait = unit_wait(beta, h, m);
    match lifetime {
        Lifetime::Exponential => (-t * (1.0 / wait + 1.0 / m)).exp(),
        Lifetime::Fixed => {
            let (rate, p) = (1.0 / wait, m / (wait + m));
            let mut on = (1.0 - t / m).max(0.0);
            let mut k = 1usize;
            while (k - 1) as f64 * m <= t {
                let a = t - (k - 1) as f64 * m;
                on += (erlang_excess(k, rate, a) - 2.0 * erlang_excess(k, rate, a - m) + erlang_excess(k, rate, a - 2.0 * m)) / m;
                k += 1;
            }
            (on - p) / (1.0 - p)
        }
    }
}

/// The long-run variance `lim T Var(A_T)` of a time average, from the means of `batches` of length
/// `batch` time units: `batch` times their sample variance, with its standard error from the spread
/// of the squared deviations (so it does not assume the batch means are Gaussian).
///
/// No window and no autocorrelation function. It reads LOW by about `kappa / batch`, relative, where
/// `kappa = int t rho(t) dt / int rho(t) dt` is the correlation's mean time — `tau` itself for an
/// exponential autocorrelation — because a batch average forgives the correlation near its edges;
/// batches much longer than the slowest mode make that small, and comparing two batch lengths
/// measures it. Returns `(sigma^2, standard error)`.
///
/// # Panics
///
/// With fewer than two batches.
#[must_use]
pub fn long_run_variance(means: &[f64], batch: f64) -> (f64, f64) {
    let k = means.len();
    assert!(k >= 2, "a variance needs at least two batch means, got {k}");
    let kf = k as f64;
    let mean = means.iter().sum::<f64>() / kf;
    let dev2: Vec<f64> = means.iter().map(|x| (x - mean) * (x - mean)).collect();
    let s2 = dev2.iter().sum::<f64>() / (kf - 1.0);
    let m2 = dev2.iter().sum::<f64>() / kf;
    let spread = dev2.iter().map(|d| (d - m2) * (d - m2)).sum::<f64>() / (kf - 1.0);
    (batch * s2, batch * (spread / kf).sqrt() * kf / (kf - 1.0))
}

/// A function of the spins, integrated along the path by [`PointProcess::batch_means`].
pub type Observable<'a> = &'a dyn Fn(&[i8]) -> f64;

/// A spiking sampler over the spins of `g`, in continuous time.
pub struct PointProcess<'g> {
    g: &'g Graph,
    beta: f64,
    m: f64,
    lifetime: Lifetime,
    /// Current spins: `+1` while a spike is live.
    pub s: Vec<i8>,
    /// For each live spike, the time it expires.
    expiry: Vec<f64>,
    /// Current time.
    pub t: f64,
    /// Births so far: the events that cost a random draw of a unit.
    pub births: u64,
    rng: Pcg,
}

impl<'g> PointProcess<'g> {
    /// A sampler with every unit off at time zero.
    ///
    /// # Panics
    ///
    /// If `m` is not positive and finite.
    #[must_use]
    pub fn new(g: &'g Graph, beta: f64, m: f64, lifetime: Lifetime, seed: u64) -> Self {
        assert!(m.is_finite() && m > 0.0, "the spike lifetime must be positive, got {m}");
        PointProcess {
            g,
            beta,
            m,
            lifetime,
            s: vec![-1; g.n],
            expiry: vec![f64::INFINITY; g.n],
            t: 0.0,
            births: 0,
            rng: Pcg::new(seed, 0x5917),
        }
    }

    fn birth_rate(&self, i: usize) -> f64 {
        (2.0 * self.beta * self.g.field(i, &self.s)).exp() / self.m
    }

    /// Advance to the next event, or to `horizon` if nothing happens first. Returns the time the
    /// state held before this call's change, so a caller can weight observables by it.
    fn step_until(&mut self, horizon: f64) -> f64 {
        let n = self.g.n;
        let rates: Vec<f64> = (0..n).map(|i| if self.s[i] < 0 { self.birth_rate(i) } else { 0.0 }).collect();
        let total: f64 = rates.iter().sum();
        let (next_exp, who_exp) = self
            .expiry
            .iter()
            .enumerate()
            .fold((f64::INFINITY, usize::MAX), |acc, (i, &e)| if e < acc.0 { (e, i) } else { acc });
        let wait = if total > 0.0 { -(self.rng.f64().max(1e-300)).ln() / total } else { f64::INFINITY };
        let t_birth = self.t + wait;
        let start = self.t;
        if t_birth.min(next_exp) >= horizon {
            self.t = horizon;
            return horizon - start;
        }
        if t_birth < next_exp {
            let mut u = self.rng.f64() * total;
            let mut who = n - 1;
            for (i, &r) in rates.iter().enumerate() {
                if u < r {
                    who = i;
                    break;
                }
                u -= r;
            }
            self.t = t_birth;
            self.s[who] = 1;
            let life = match self.lifetime {
                Lifetime::Fixed => self.m,
                Lifetime::Exponential => -(self.rng.f64().max(1e-300)).ln() * self.m,
            };
            self.expiry[who] = self.t + life;
            self.births += 1;
        } else {
            self.t = next_exp;
            self.s[who_exp] = -1;
            self.expiry[who_exp] = f64::INFINITY;
        }
        self.t - start
    }

    /// Run until time `until`, adding `duration × weight(state)` into `acc` for every stretch of
    /// time the state is held: the exact time integral of an observable along the path.
    pub fn integrate(&mut self, until: f64, mut acc: impl FnMut(&[i8], f64)) {
        while self.t < until {
            let held = self.s.clone();
            let dt = self.step_until(until);
            if dt > 0.0 {
                acc(&held, dt);
            }
        }
    }

    /// Time averages of each observable over `batches` consecutive batches of length `batch`, after
    /// running `burn`: `out[o][b]` is observable `o` integrated EXACTLY along the path over batch `b`,
    /// divided by `batch`. No sampling grid, so nothing is lost between reads; hand each row to
    /// [`long_run_variance`].
    ///
    /// # Panics
    ///
    /// If `batch` is not positive and finite.
    pub fn batch_means(&mut self, burn: f64, batch: f64, batches: usize, observables: &[Observable<'_>]) -> Vec<Vec<f64>> {
        assert!(batch.is_finite() && batch > 0.0, "a batch must last a positive, finite time, got {batch}");
        self.integrate(self.t + burn, |_, _| {});
        let mut out = vec![Vec::with_capacity(batches); observables.len()];
        let mut sums = vec![0.0f64; observables.len()];
        for _ in 0..batches {
            sums.iter_mut().for_each(|v| *v = 0.0);
            self.integrate(self.t + batch, |s, dt| {
                for (acc, f) in sums.iter_mut().zip(observables) {
                    *acc += dt * f(s);
                }
            });
            for (row, v) in out.iter_mut().zip(&sums) {
                row.push(v / batch);
            }
        }
        out
    }

    /// The state read at `count` regularly spaced times, `spacing` apart, starting after `burn`.
    pub fn trace(&mut self, burn: f64, spacing: f64, count: usize, observable: impl Fn(&[i8]) -> f64) -> Vec<f64> {
        self.integrate(self.t + burn, |_, _| {});
        let mut out = Vec::with_capacity(count);
        for _ in 0..count {
            let target = self.t + spacing;
            self.integrate(target, |_, _| {});
            out.push(observable(&self.s));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::GraphBuilder;

    /// **The exact oracle: one unit is an alternating renewal process.** Off for an exponential time
    /// of rate `(1/m) e^{2βh}`, on for a time of mean `m`, so the long-run on-fraction is
    /// `e^{2βh} / (1 + e^{2βh})` exactly, for EITHER lifetime -- which is the Boltzmann probability of
    /// `+1` in a field `h`. Held within 1% of time integrated over `2e5 m`.
    #[test]
    fn a_single_unit_is_on_for_exactly_its_boltzmann_fraction() {
        for &h in &[-0.7f64, 0.0, 0.4] {
            let mut b = GraphBuilder::new(1);
            b.bias(0, h);
            let g = b.build();
            let want = 1.0 / (1.0 + (-2.0 * h).exp());
            for lifetime in [Lifetime::Fixed, Lifetime::Exponential] {
                let mut pp = PointProcess::new(&g, 1.0, 1.0, lifetime, 7);
                let (mut on, mut total) = (0.0, 0.0);
                pp.integrate(2e5, |s, dt| {
                    total += dt;
                    if s[0] > 0 {
                        on += dt;
                    }
                });
                let got = on / total;
                assert!((got - want).abs() < 0.01, "h {h}, {lifetime:?}: on {got} vs {want}");
            }
        }
    }

    /// **Coupled units sample the Boltzmann law**, both lifetimes: the time-weighted state law of a
    /// frustrated triangle with fields, against enumeration.
    #[test]
    fn coupled_units_sample_the_boltzmann_law() {
        let mut b = GraphBuilder::new(3);
        b.couple(0, 1, 0.8);
        b.couple(1, 2, -0.5);
        b.couple(0, 2, 0.6);
        b.bias(0, 0.3);
        b.bias(2, -0.2);
        let g = b.build();
        let beta = 1.0;
        let exact = crate::autocorr::boltzmann(&g, beta).expect("small");
        for lifetime in [Lifetime::Fixed, Lifetime::Exponential] {
            let mut pp = PointProcess::new(&g, beta, 1.0, lifetime, 11);
            let mut law = vec![0.0f64; 8];
            pp.integrate(1e5, |s, dt| {
                let x = s.iter().enumerate().fold(0usize, |a, (i, &v)| if v > 0 { a | (1 << i) } else { a });
                law[x] += dt;
            });
            let total: f64 = law.iter().sum();
            law.iter_mut().for_each(|v| *v /= total);
            let tv = crate::autocorr::total_variation(&law, &exact);
            assert!(tv < 0.01, "{lifetime:?}: TV {tv} from Boltzmann");
        }
    }

    fn one_unit(h: f64) -> crate::graph::Graph {
        let mut b = GraphBuilder::new(1);
        b.bias(0, h);
        b.build()
    }

    /// **One unit's efficiency, exactly: a random lifetime doubles the variance.** A single unit, both
    /// lifetimes, at zero field and at `beta h = 0.7`, run for `4e6 m` and cut into 20,000 batches of
    /// `200 m`. [`long_run_variance`] of the on-indicator's batch means must match [`unit_variance`]
    /// within four of its own standard errors plus the bias bound below, and the exponential/fixed
    /// ratio must be 2 within four of the ratio's plus twice that bound.
    ///
    /// Why the tolerance means something: the standard error is about `sqrt(2 / 20,000)` = 1% of the
    /// variance (asserted below 2%, so a broken estimator cannot widen its own band into a pass), and
    /// the estimator's bias is about `-kappa / batch` relative, for `kappa = int t rho / int rho`:
    /// `0.5 m` for the exponential lifetime at zero field (it is `tau` there), `0.2 m` at `0.7`, and
    /// `0.083 m` and `-0.068 m` for the fixed one at the two fields (from [`unit_autocorrelation`]),
    /// so at most 0.25%, a quarter of a standard error; it is allowed for, not ignored (at `100 m`
    /// batches it was half a standard error, and sixteen seeds put the exponential's mean `z` at
    /// `-0.80 +- 0.25`). Wrong answers the band excludes: the exponential's own variance for the fixed
    /// lifetime (+100%), a uniform lifetime's (+33%), or a variance per unit time missing the batch
    /// length.
    #[test]
    fn the_exact_one_unit_variance_holds_to_a_window_free_estimate() {
        let (beta, m, batch, batches) = (1.0, 1.0, 200.0, 20_000usize);
        let bias = 0.0025;
        for &h in &[0.0f64, 0.7] {
            let g = one_unit(h);
            let on = |s: &[i8]| if s[0] > 0 { 1.0 } else { 0.0 };
            let mut got = Vec::new();
            for (seed, lifetime) in [(21u64, Lifetime::Fixed), (22, Lifetime::Exponential)] {
                let mut pp = PointProcess::new(&g, beta, m, lifetime, seed);
                let means = pp.batch_means(50.0, batch, batches, &[&on]);
                let (v, se) = long_run_variance(&means[0], batch);
                let exact = unit_variance(beta, h, m, lifetime);
                assert!(se < 0.02 * exact, "h {h}, {lifetime:?}: standard error {se} is not resolving {exact}");
                assert!((v - exact).abs() < 4.0 * se + bias * exact, "h {h}, {lifetime:?}: batch means {v} +- {se}, exact {exact}");
                got.push((v, se));
            }
            let exact_ratio = unit_variance(beta, h, m, Lifetime::Exponential) / unit_variance(beta, h, m, Lifetime::Fixed);
            assert!((exact_ratio - 2.0).abs() < 1e-12, "h {h}: the exact ratio is {exact_ratio}");
            let ((vf, sf), (ve, s_e)) = (got[0], got[1]);
            let ratio = ve / vf;
            let se_ratio = ratio * ((sf / vf).powi(2) + (s_e / ve).powi(2)).sqrt();
            assert!((ratio - 2.0).abs() < 4.0 * se_ratio + 2.0 * 2.0 * bias, "h {h}: measured ratio {ratio} +- {se_ratio}, exact 2");
        }
        // The general law, as algebra: a lifetime with coefficient of variation c costs 1 + c^2.
        let (wait, m) = (0.8f64, 1.0f64);
        let fixed = renewal_variance(wait, wait, m, 0.0);
        let uniform = renewal_variance(wait, wait, m, m / 3f64.sqrt());
        assert!((uniform / fixed - 4.0 / 3.0).abs() < 1e-12, "uniform on [0, 2m]: {}", uniform / fixed);
    }

    /// **The exact autocorrelation, held three ways.** Its integral is [`unit_tau`], which comes
    /// from the renewal CLT and shares no algebra with the Erlang sum: to `1e-6` relative by
    /// Simpson's rule over `60 m` at a step of `m / 1000`, for both lifetimes at three fields. At
    /// zero field and `t <= m` it is the closed form `2 e^{-t/m} - 1` to `1e-12`, NEGATIVE past
    /// `m ln 2`. And a simulated fixed-lifetime unit shows the lobe: its empirical autocorrelation at
    /// `t = 0.5, 1, 1.5, 2 m`, from `1e6` reads, matches the formula within 0.01 (the noise is
    /// about `1/sqrt(1e6)` times a few correlation times, under 0.003).
    #[test]
    fn the_exact_autocorrelation_integrates_to_the_exact_tau() {
        let (beta, m) = (1.0, 1.0);
        for &h in &[0.0f64, 0.7, -0.7] {
            for lifetime in [Lifetime::Fixed, Lifetime::Exponential] {
                let (dt, steps) = (m / 1000.0, 60_000usize);
                let mut integral = 0.0;
                // Simpson's rule; the fixed lifetime's kinks at multiples of m fall on panel edges.
                for i in 0..=steps {
                    let w = if i == 0 || i == steps { 1.0 } else if i % 2 == 1 { 4.0 } else { 2.0 };
                    integral += w * dt / 3.0 * unit_autocorrelation(beta, h, m, lifetime, i as f64 * dt);
                }
                let tau = unit_tau(beta, h, m, lifetime);
                assert!((integral / tau - 1.0).abs() < 1e-6, "h {h}, {lifetime:?}: int rho {integral}, tau {tau}");
            }
        }
        for i in 0..=100 {
            let t = i as f64 * m / 100.0;
            let want = 2.0 * (-t / m).exp() - 1.0;
            let got = unit_autocorrelation(beta, 0.0, m, Lifetime::Fixed, t);
            assert!((got - want).abs() < 1e-12, "t {t}: {got} vs closed form {want}");
        }
        assert!(unit_autocorrelation(beta, 0.0, m, Lifetime::Fixed, m) < -0.26, "the lobe at t = m is 2/e - 1");

        let g = one_unit(0.0);
        let mut pp = PointProcess::new(&g, beta, m, Lifetime::Fixed, 31);
        let x = pp.trace(50.0, 0.1, 1_000_000, |s| if s[0] > 0 { 1.0 } else { 0.0 });
        let n = x.len();
        let mean = x.iter().sum::<f64>() / n as f64;
        let var = x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;
        for lag in [5usize, 10, 15, 20] {
            let c = (0..n - lag).map(|i| (x[i] - mean) * (x[i + lag] - mean)).sum::<f64>() / ((n - lag) as f64 * var);
            let want = unit_autocorrelation(beta, 0.0, m, Lifetime::Fixed, lag as f64 * 0.1);
            assert!((c - want).abs() < 0.01, "lag {lag}: simulated {c}, exact {want}");
        }
    }

    /// **Sokal's window misreads a fixed lifetime, and this is by how much.** The crate's window
    /// ([`crate::certify::sokal_window`], what [`crate::certify::tau_int`] closes over a trace) applied
    /// to the EXACT autocorrelation read every `0.1 m`, the spacing `examples/pointproc_exact.rs`
    /// used when it measured with the window. At zero field it closes at lag 12, inside the negative
    /// lobe, and reads `tau` **8.10% low**; at `beta h = 0.7` it closes at lag 7, before the lobe, and
    /// reads it **13.31% high**. The unwindowed sum of the same reads is exact to `1e-9` (a spike of
    /// length `m` covers exactly ten reads), so the error is the window's. The exponential lifetime's
    /// autocorrelation never changes sign and the window reads it within 2%, so the ratio the window
    /// reports is 2.17 and 1.80 where the truth is 2. And on a simulated trace of `1e6` reads,
    /// `tau_int` itself shows the same 8% deficit at zero field.
    #[test]
    fn sokal_window_misreads_a_fixed_lifetime() {
        let (beta, m, spacing) = (1.0, 1.0, 0.1);
        let window = |h: f64, lifetime: Lifetime| {
            spacing * crate::certify::sokal_window(100_000, |k| unit_autocorrelation(beta, h, m, lifetime, k as f64 * spacing))
        };
        for &(h, bias) in &[(0.0f64, -0.0810f64), (0.7, 0.1331)] {
            let exact = unit_tau(beta, h, m, Lifetime::Fixed);
            let read = window(h, Lifetime::Fixed) / exact - 1.0;
            assert!((read - bias).abs() < 5e-4, "h {h}: the window reads {read:+.4}, measured {bias:+.4}");
            let full = spacing * (0.5 + (1..=1200).map(|k| unit_autocorrelation(beta, h, m, Lifetime::Fixed, k as f64 * spacing)).sum::<f64>());
            assert!((full / exact - 1.0).abs() < 1e-9, "h {h}: the unwindowed sum {full} vs {exact}");
            let exp_read = window(h, Lifetime::Exponential) / unit_tau(beta, h, m, Lifetime::Exponential) - 1.0;
            assert!(exp_read.abs() < 0.02, "h {h}: exponential read {exp_read:+.4}");
        }
        let ratio0 = window(0.0, Lifetime::Exponential) / window(0.0, Lifetime::Fixed);
        let ratio7 = window(0.7, Lifetime::Exponential) / window(0.7, Lifetime::Fixed);
        assert!((ratio0 - 2.170).abs() < 0.005 && (ratio7 - 1.797).abs() < 0.005, "window ratios {ratio0}, {ratio7}");

        let g = one_unit(0.0);
        let mut pp = PointProcess::new(&g, beta, m, Lifetime::Fixed, 41);
        let x = pp.trace(50.0, spacing, 1_000_000, |s| if s[0] > 0 { 1.0 } else { 0.0 });
        let read = crate::certify::tau_int(&x) * spacing / unit_tau(beta, 0.0, m, Lifetime::Fixed) - 1.0;
        assert!((read + 0.081).abs() < 0.02, "tau_int on a simulated trace reads {read:+.4}");
    }
}
