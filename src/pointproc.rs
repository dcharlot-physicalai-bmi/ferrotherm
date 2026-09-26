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
//! For coupled units the law is held to enumeration statistically, from long runs, and the comparison
//! the paper makes (efficiency against birth–death) is measured by `examples/pointproc_exact.rs`.

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
}
