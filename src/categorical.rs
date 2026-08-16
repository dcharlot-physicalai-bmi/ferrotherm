//! Categorical variables as a workload: does the encoding choice actually pay?
//!
//! [`crate::encode`] offers three ways to spell a `k`-valued variable in spins, and the counting
//! argument for domain-wall encoding is easy: `k-1` spins and a *linear* chain of penalty couplings,
//! against one-hot's `k` spins and a *quadratic* all-to-all penalty. What the counting argument does
//! not tell you is whether a sampler actually finds valid states more often.
//!
//! That is a measurable question and this module measures it. The workload is deliberately the
//! simplest one where the encoding is the only thing under test: assign each of `n` independent
//! variables one of `k` values, with no objective at all. Every configuration is equally good, so
//! the only thing a sampler can get wrong is producing a state that does not decode — two hot bits,
//! or two domain walls. The score is the fraction of variables that come back valid.
//!
//! # What it shows, which is not what the counting argument suggests
//!
//! **At an adequate penalty both encodings are perfectly feasible** — 1.0 at every `k` up to 32.
//! There is no gap to find there, and a test looking for one passes vacuously.
//!
//! The difference is in **how weak a penalty each tolerates**. Smallest penalty reaching 0.99
//! feasible, measured over 8 seeds (`examples/encoding_probe.rs`):
//!
//! | k | domain wall | one-hot | ratio |
//! |---|---|---|---|
//! | 4 | 0.074 | 0.358 | 4.8× |
//! | 8 | 0.163 | 0.466 | 2.9× |
//! | 16 | 0.212 | 0.606 | 2.9× |
//! | 32 | 0.276 | 0.787 | 2.9× |
//!
//! Domain-wall needs roughly **three times weaker** a penalty, and that is the payoff worth having
//! rather than the spin count. A penalty does not sit alone: it is added to whatever objective the
//! model actually encodes, and a large one distorts that objective. Needing less of it means the
//! problem you solve is closer to the problem you posed — which is the same reason the penalty
//! strength is a ramped quantity in [`crate::schedule`] rather than a constant.

use crate::encode::{Encoding, Slot};
use crate::gibbs::Sampler;
use crate::graph::GraphBuilder;
use crate::schedule::Schedule;

/// A block of `n` independent `k`-valued variables in one encoding.
pub struct Categorical {
    pub slots: Vec<Slot>,
    pub graph: crate::graph::Graph,
    pub encoding: Encoding,
    pub k: usize,
    /// Whether the penalty pins these variables to their codewords EXACTLY.
    ///
    /// False for a binary encoding whose `k` is not a power of two: the spare codewords decode to
    /// nothing, and no pairwise penalty separates them from the valid ones — measured on k = 6, an
    /// invalid state costs exactly what a valid one costs. `Slot::decode` is then the only thing
    /// standing between the sampler and a wrong answer, so a caller that samples this block without
    /// checking every decode is trusting states the encoding never excluded.
    ///
    /// `Slot::add_penalty` has always returned this. It was discarded here and in `Model::compile`;
    /// the compile path grew `Compiled::caveats`, and the compiler found this one only once
    /// `add_penalty` was marked `#[must_use]` — a fix in one caller is a hypothesis about the rest.
    pub exact: bool,
}

impl Categorical {
    /// Lay out `n` variables side by side and add the encoding's penalty at strength `p`.
    pub fn new(n: usize, k: usize, encoding: Encoding, p: f64) -> Categorical {
        assert!(n >= 1 && k >= 2);
        let width = encoding.spins(k);
        let mut b = GraphBuilder::new(n * width);
        let mut slots = Vec::with_capacity(n);
        let mut exact = true;
        for v in 0..n {
            let s = Slot::new(v * width, k, encoding);
            exact &= s.add_penalty(&mut b, p);
            slots.push(s);
        }
        Categorical { slots, graph: b.build(), encoding, k, exact }
    }

    /// Spins this layout occupies.
    pub fn spins(&self) -> usize {
        self.graph.n
    }

    /// Fraction of variables that decode to a valid value in `state`.
    pub fn feasible_fraction(&self, state: &[i8]) -> f64 {
        let ok = self.slots.iter().filter(|s| s.decode(state).is_some()).count();
        ok as f64 / self.slots.len() as f64
    }

    /// Anneal, then report the fraction of variables that decoded.
    ///
    /// The penalty is what has to be satisfied, so this is a pure feasibility measurement: there is
    /// no objective competing with it and any failure is the encoding's.
    pub fn anneal_feasibility(&self, schedule: &Schedule, seed: u64) -> f64 {
        let (best, _) = crate::tempering::anneal_scheduled(&self.graph, schedule, seed, None);
        self.feasible_fraction(&best)
    }

    /// Sample at fixed temperature, then report the fraction that decoded.
    pub fn sample_feasibility(&self, beta: f64, sweeps: usize, seed: u64) -> f64 {
        let mut smp = Sampler::new(&self.graph, beta, seed);
        smp.sweeps(sweeps, None);
        self.feasible_fraction(&smp.s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ladder() -> Schedule {
        Schedule::geometric(0.05, 8.0, 80, 30)
    }

    /// Mean feasible fraction across seeds, which is what a rate needs to be measured over.
    fn rate(k: usize, enc: Encoding, p: f64) -> f64 {
        let n = 40;
        (1..=8u64)
            .map(|seed| Categorical::new(n, k, enc, p).anneal_feasibility(&ladder(), seed))
            .sum::<f64>()
            / 8.0
    }

    #[test]
    fn the_counting_argument_holds_in_the_layout() {
        // The claim that motivates the whole module, checked on the built graph rather than on
        // paper: fewer spins and fewer couplings, for every k.
        for k in 3..=12 {
            let dw = Categorical::new(10, k, Encoding::DomainWall, 1.0);
            let oh = Categorical::new(10, k, Encoding::OneHot, 1.0);
            assert!(dw.spins() < oh.spins(), "k={k}: {} vs {}", dw.spins(), oh.spins());
            let edges = |c: &Categorical| {
                (0..c.graph.n).map(|i| c.graph.offset[i + 1] - c.graph.offset[i]).sum::<usize>() / 2
            };
            assert!(edges(&dw) < edges(&oh), "k={k}: {} vs {} couplings", edges(&dw), edges(&oh));
        }
    }

    #[test]
    fn both_encodings_can_be_satisfied_at_all() {
        // A feasibility workload nobody can satisfy measures nothing.
        for enc in [Encoding::DomainWall, Encoding::OneHot] {
            let f = rate(4, enc, 2.0);
            assert!(f > 0.9, "{enc:?} only reached {f:.3} feasible");
        }
    }

    #[test]
    fn at_an_adequate_penalty_there_is_no_gap_at_all() {
        // Recorded because an earlier version of this file asserted a feasibility gap here and
        // passed vacuously: both encodings sit at exactly 1.0, so "domain wall >= one-hot" is true
        // and measures nothing. Pinning the equality makes the vacuity impossible to reintroduce.
        for k in [8usize, 16, 32] {
            assert_eq!(rate(k, Encoding::DomainWall, 2.0), 1.0, "k={k}");
            assert_eq!(rate(k, Encoding::OneHot, 2.0), 1.0, "k={k}");
        }
    }

    #[test]
    fn domain_wall_tolerates_a_much_weaker_penalty() {
        // The real payoff, and the one worth having. A penalty is added to whatever objective the
        // model encodes, so needing less of it means solving a problem closer to the one posed.
        // Measured: domain wall reaches 0.99 feasible at roughly a third of one-hot's penalty.
        let smallest = |k: usize, e: Encoding| {
            let mut p = 0.02f64;
            while p < 4.0 {
                if rate(k, e, p) >= 0.99 {
                    return p;
                }
                p *= 1.3;
            }
            f64::INFINITY
        };
        for k in [8usize, 32] {
            let (dw, oh) = (smallest(k, Encoding::DomainWall), smallest(k, Encoding::OneHot));
            assert!(dw.is_finite() && oh.is_finite(), "k={k}: neither encoding reached 0.99");
            assert!(dw < oh / 2.0, "k={k}: domain wall {dw:.3} vs one-hot {oh:.3}");
        }
    }

    #[test]
    fn a_weak_penalty_fails_and_says_so_by_failing() {
        // The penalty strength is a real knob, not decoration. If a near-zero penalty still gave
        // feasible states, the constraint would not be doing the work and the comparison above
        // would be measuring nothing.
        let weak = rate(8, Encoding::OneHot, 0.01);
        let strong = rate(8, Encoding::OneHot, 2.0);
        assert!(weak < strong, "weak penalty {weak:.3} should lose to strong {strong:.3}");
    }

    #[test]
    fn hot_sampling_is_infeasible_and_cold_sampling_is_not() {
        // Temperature does what temperature should: at high temperature the penalty is irrelevant
        // and states are mostly invalid; cold, they are not.
        let c = Categorical::new(40, 6, Encoding::DomainWall, 2.0);
        let hot = c.sample_feasibility(0.02, 400, 1);
        let cold = c.sample_feasibility(6.0, 400, 1);
        assert!(cold > hot, "cold {cold:.3} should beat hot {hot:.3}");
        assert!(hot < 0.9, "an essentially free penalty should let invalid states through");
    }

    #[test]
    fn binary_is_exact_only_on_a_power_of_two() {
        // The trap encode.rs documents, exercised as a workload: at k = 8 every code is valid and
        // feasibility is trivially 1; at k = 6 two codes are surplus and no pairwise penalty
        // removes them, so the sampler produces states that simply do not decode.
        let eight = Categorical::new(60, 8, Encoding::Binary, 2.0);
        assert_eq!(eight.feasible_fraction(&vec![1i8; eight.spins()]), 1.0);

        let six = Categorical::new(60, 6, Encoding::Binary, 2.0);
        let f = six.sample_feasibility(1.0, 300, 3);
        assert!(f < 1.0, "k=6 binary must let surplus codes through, got {f:.3}");
    }

    #[test]
    fn a_block_reports_whether_its_encoding_is_exact() {
        // The bool `add_penalty` returns, carried instead of discarded. One-hot and domain-wall
        // pin every codeword; binary does only when k is a power of two, and the difference is not
        // cosmetic -- for k = 6 an invalid state costs exactly what a valid one costs, so the
        // sampler has no reason to avoid it and `decode` is the only guard left.
        assert!(Categorical::new(4, 6, Encoding::OneHot, 1.0).exact);
        assert!(Categorical::new(4, 6, Encoding::DomainWall, 1.0).exact);
        assert!(Categorical::new(4, 8, Encoding::Binary, 1.0).exact, "8 is a power of two");
        assert!(!Categorical::new(4, 6, Encoding::Binary, 1.0).exact, "6 is not");
    }
}
