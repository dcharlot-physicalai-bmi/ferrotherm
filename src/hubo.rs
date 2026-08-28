//! Higher-order models, solved **without** reducing them to pairwise.
//!
//! [`crate::reduce`] makes a `k`-body model fit pairwise hardware by introducing ancilla spins and
//! a penalty that keeps them honest. That is the right pass when the target really is pairwise —
//! an annealer chip, a Chimera graph, a fabric that declares `max_arity: 2`. It is the wrong pass
//! when the target is a CPU, because the ancillas are pure overhead: more spins to search over,
//! a penalty weight to get right, and a solution that is only valid if every ancilla constraint
//! happens to hold at the end.
//!
//! On a CPU nothing forces arity two. A term of any width contributes `−w · Π s_i`, and the change
//! from flipping one spin is a sum over the terms containing it:
//!
//! ```text
//!   ΔE_i  =  2 · Σ_{T ∋ i} w_T · Π_{j ∈ T} s_j
//! ```
//!
//! which costs `O(terms containing i)` — the same shape as the pairwise incremental update, with
//! degree replaced by term-incidence. So the higher-order model is not harder to sample; it is
//! only harder to *put on pairwise hardware*, and those are different problems that the reduction
//! pass conflates whenever the hardware is not the constraint.
//!
//! # The measurable claim
//!
//! Ancillas. A model with `t` terms of arity `k` needs roughly `t·(k−2)` of them to become
//! pairwise, and every one is a spin the search has to get right for the answer to mean anything.
//! Solved natively there are none, and [`Hubo::from_graph`] exists so the two paths can be run
//! against each other on the same model rather than argued about.
//!
//! ```
//! use ferrotherm::hubo::{Hubo, Params, anneal};
//!
//! // A three-body parity term: −s0·s1·s2, minimised when the product is +1.
//! let mut h = Hubo::new(3);
//! h.add(&[0, 1, 2], 1.0).unwrap();
//! let out = anneal(&h, &Params::default(), 7);
//! assert_eq!(out.energy, -1.0);
//! assert_eq!(out.state[0] * out.state[1] * out.state[2], 1);
//! ```

use crate::factor::FactorError;
use crate::graph::Graph;
use crate::ledger::Ledger;
use crate::rng::Pcg;

/// A higher-order unconstrained binary optimisation model over `±1` spins.
///
/// `E(s) = −Σ_T w_T · Π_{i ∈ T} s_i`, which is the same sign convention as [`Graph::energy`] and
/// reduces to it exactly for terms of arity one and two.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Hubo {
    n: usize,
    terms: Vec<(Vec<u32>, f64)>,
    /// For each variable, the indices of the terms containing it. This is what makes the flip
    /// update `O(incidence)` rather than `O(terms)`, and it is the whole reason this is practical.
    incident: Vec<Vec<usize>>,
}

impl Hubo {
    pub fn new(n: usize) -> Hubo {
        Hubo { n, terms: Vec::new(), incident: vec![Vec::new(); n] }
    }

    pub fn len(&self) -> usize {
        self.n
    }

    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Terms in the model.
    pub fn terms(&self) -> usize {
        self.terms.len()
    }

    /// The widest term. This is the number [`crate::reduce`] exists to bring down to two.
    pub fn max_arity(&self) -> usize {
        self.terms.iter().map(|(v, _)| v.len()).max().unwrap_or(0)
    }

    /// Ancillas the pairwise reduction of this model would introduce.
    ///
    /// Each term of arity `k > 2` needs `k − 2` products built up. Reported so the cost the native
    /// path avoids is a number rather than a claim.
    pub fn ancillas_avoided(&self) -> usize {
        self.terms.iter().map(|(v, _)| v.len().saturating_sub(2)).sum()
    }

    /// Add a term. Rejects the same things [`crate::factor::Factor`] rejects, and for the same
    /// reason: a term that is silently not what was written is worse than one that is refused.
    pub fn add(&mut self, vars: &[usize], weight: f64) -> Result<(), FactorError> {
        let f = crate::factor::Factor::new(vars, weight, self.n)?;
        let mut vs: Vec<u32> = f.vars().map(|v| v as u32).collect();
        vs.sort_unstable();
        let idx = self.terms.len();
        for &v in &vs {
            self.incident[v as usize].push(idx);
        }
        self.terms.push((vs, weight));
        Ok(())
    }

    /// Lift a pairwise graph into a higher-order model, unchanged.
    ///
    /// Exists so the native and reduced paths can be compared on one model. The energies must agree
    /// exactly, which is the test that says this convention is the crate's convention.
    pub fn from_graph(g: &Graph) -> Hubo {
        let mut h = Hubo::new(g.n);
        for i in 0..g.n {
            if g.h[i] != 0.0 {
                h.add(&[i], g.h[i]).expect("a single in-range variable");
            }
            for k in g.offset[i]..g.offset[i + 1] {
                let j = g.nbr[k] as usize;
                if j > i {
                    h.add(&[i, j], g.w[k]).expect("two distinct in-range variables");
                }
            }
        }
        h
    }

    /// `E(s) = −Σ_T w_T Π s_i`.
    pub fn energy(&self, s: &[i8]) -> f64 {
        self.terms
            .iter()
            .map(|(vs, w)| {
                let prod: i32 = vs.iter().map(|&v| s[v as usize] as i32).product();
                -w * prod as f64
            })
            .sum()
    }

    /// The energy change from flipping spin `i`, in `O(terms containing i)`.
    pub fn delta(&self, s: &[i8], i: usize) -> f64 {
        2.0 * self.incident[i]
            .iter()
            .map(|&t| {
                let (vs, w) = &self.terms[t];
                let prod: i32 = vs.iter().map(|&v| s[v as usize] as i32).product();
                w * prod as f64
            })
            .sum::<f64>()
    }
}

/// How the anneal is run.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Params {
    pub beta_min: f64,
    pub beta_max: f64,
    pub stages: usize,
    pub sweeps_per_stage: usize,
}

impl Default for Params {
    fn default() -> Self {
        Params { beta_min: 0.05, beta_max: 8.0, stages: 200, sweeps_per_stage: 8 }
    }
}

/// What the anneal found.
#[derive(Clone, Debug)]
pub struct Outcome {
    pub state: Vec<i8>,
    /// Recomputed from the state by the model, not carried from the accumulator.
    pub energy: f64,
    pub proposals: u64,
    pub accepted: u64,
    /// Ancillas a pairwise reduction of this model would have needed, and this path did not.
    pub ancillas_avoided: usize,
}

/// Anneal a higher-order model directly, with no reduction and no ancillas.
pub fn anneal(h: &Hubo, p: &Params, seed: u64) -> Outcome {
    anneal_metered(h, p, seed, None)
}

/// As [`anneal`], charging every proposal to a [`Ledger`].
pub fn anneal_metered(h: &Hubo, p: &Params, seed: u64, mut ledger: Option<&mut Ledger>) -> Outcome {
    let n = h.n;
    if n == 0 {
        return Outcome {
            state: Vec::new(),
            energy: 0.0,
            proposals: 0,
            accepted: 0,
            ancillas_avoided: 0,
        };
    }
    let mut rng = Pcg::new(seed, 0x0000_40B0);
    let mut s: Vec<i8> = (0..n).map(|_| rng.spin(0.5)).collect();
    let mut best = s.clone();
    let mut best_e = h.energy(&s);
    let (mut proposals, mut accepted) = (0u64, 0u64);

    let stages = p.stages.max(1);
    let (b0, b1) = (p.beta_min.max(1e-12), p.beta_max.max(p.beta_min.max(1e-12)));
    for stage in 0..stages {
        let f = if stages == 1 { 1.0 } else { stage as f64 / (stages - 1) as f64 };
        let beta = b0 * (b1 / b0).powf(f);
        for _ in 0..p.sweeps_per_stage.max(1) {
            for i in 0..n {
                let d = h.delta(&s, i);
                proposals += 1;
                if d <= 0.0 || rng.f64() < (-beta * d).exp() {
                    s[i] = -s[i];
                    accepted += 1;
                }
            }
            if let Some(l) = ledger.as_deref_mut() {
                l.samples += n as u64;
            }
            // Recomputed each sweep rather than tracked incrementally: an accumulator over a
            // higher-order model drifts the same way it does over a pairwise one, and the incumbent
            // is the one number that must not inherit that drift.
            let e = h.energy(&s);
            if e < best_e {
                best_e = e;
                best.copy_from_slice(&s);
            }
        }
    }
    let energy = h.energy(&best);
    Outcome { state: best, energy, proposals, accepted, ancillas_avoided: h.ancillas_avoided() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::GraphBuilder;
    use crate::ising::lattice2d;

    /// The convention has to be the crate's convention, or every comparison downstream is between
    /// two different problems.
    #[test]
    fn a_lifted_graph_has_exactly_the_graph_energy() {
        for seed in 0..8u64 {
            let mut rng = Pcg::new(seed, 0x11);
            let n = 20;
            let mut gb = GraphBuilder::new(n);
            for i in 0..n {
                gb.bias(i, rng.f64() - 0.5);
                for j in (i + 1)..n {
                    if rng.f64() < 0.3 {
                        gb.couple(i, j, rng.f64() * 2.0 - 1.0);
                    }
                }
            }
            let g = gb.build();
            let h = Hubo::from_graph(&g);
            assert_eq!(h.max_arity(), 2);
            assert_eq!(h.ancillas_avoided(), 0, "a pairwise model needs no ancillas");
            for t in 0..40 {
                let s: Vec<i8> = (0..n).map(|_| rng.spin(0.5)).collect();
                assert!(
                    (h.energy(&s) - g.energy(&s)).abs() < 1e-9,
                    "seed {seed} trial {t}: hubo {} vs graph {}",
                    h.energy(&s),
                    g.energy(&s)
                );
            }
        }
    }

    /// The incremental update is the whole reason this is practical, so it is checked against the
    /// definition rather than trusted. A wrong `delta` does not raise anything: it makes a sampler
    /// quietly explore the wrong landscape.
    #[test]
    fn the_flip_update_agrees_with_recomputing_from_scratch() {
        let mut rng = Pcg::new(5, 0x22);
        let n = 12;
        let mut h = Hubo::new(n);
        for _ in 0..40 {
            let k = 1 + (rng.next_u32() as usize) % 4;
            let mut vars: Vec<usize> = Vec::new();
            while vars.len() < k {
                let v = (rng.next_u32() as usize) % n;
                if !vars.contains(&v) {
                    vars.push(v);
                }
            }
            h.add(&vars, rng.f64() * 2.0 - 1.0).unwrap();
        }
        assert!(h.max_arity() >= 3, "the test needs a genuinely higher-order model");
        for _ in 0..50 {
            let mut s: Vec<i8> = (0..n).map(|_| rng.spin(0.5)).collect();
            for i in 0..n {
                let before = h.energy(&s);
                let d = h.delta(&s, i);
                s[i] = -s[i];
                let after = h.energy(&s);
                assert!(
                    (after - before - d).abs() < 1e-9,
                    "site {i}: delta said {d}, the model moved by {}",
                    after - before
                );
                s[i] = -s[i];
            }
        }
    }

    /// The claim is that the native path solves the model the reduction would have needed ancillas
    /// for, and gets the same answer with none. Checked against exhaustive enumeration, which is
    /// the only ground truth that does not itself depend on a sampler.
    #[test]
    fn it_solves_a_higher_order_model_that_the_reduction_would_pay_ancillas_for() {
        let mut solved = 0;
        for seed in 0..12u64 {
            let mut rng = Pcg::new(seed, 0x33);
            let n = 14;
            let mut h = Hubo::new(n);
            for _ in 0..24 {
                let k = 3 + (rng.next_u32() as usize) % 2; // arity 3 or 4: strictly higher-order
                let mut vars: Vec<usize> = Vec::new();
                while vars.len() < k {
                    let v = (rng.next_u32() as usize) % n;
                    if !vars.contains(&v) {
                        vars.push(v);
                    }
                }
                h.add(&vars, rng.f64() * 2.0 - 1.0).unwrap();
            }
            assert!(h.ancillas_avoided() >= 24, "arity 3 and 4 terms cost at least one each");

            let out = anneal(&h, &Params::default(), seed);
            assert_eq!(out.ancillas_avoided, h.ancillas_avoided());
            assert!((out.energy - h.energy(&out.state)).abs() < 1e-9);

            // Exhaustive over 2^14, which is the ground truth.
            let mut truth = f64::INFINITY;
            let mut s = vec![1i8; n];
            for mask in 0..(1u32 << n) {
                for i in 0..n {
                    s[i] = if mask >> i & 1 == 1 { 1 } else { -1 };
                }
                truth = truth.min(h.energy(&s));
            }
            // SOUNDNESS is the invariant and holds on every seed: an energy below the exhaustive
            // minimum would mean the model and the enumeration disagree about what the energy IS.
            assert!(
                out.energy >= truth - 1e-9,
                "seed {seed}: reached {}, BELOW the exhaustive minimum {truth} -- the model's \
                 energy and the enumeration's disagree",
                out.energy
            );
            if (out.energy - truth).abs() < 1e-9 {
                solved += 1;
            }
        }
        // QUALITY is a majority, not a certainty. The first version of this required the optimum on
        // all twelve seeds, which passed on the RNG stream it was written against and broke the
        // moment an unrelated edit changed that stream -- a stochastic sampler asserted as if it
        // were deterministic. The soundness check above is the one that must never fail.
        assert!(
            solved >= 10,
            "the anneal reached the true minimum on only {solved} of 12 higher-order instances"
        );
    }

    /// A pairwise model lifted and annealed natively must not do worse than the same model annealed
    /// as a graph. If it did, the native path would be paying for generality it does not need.
    #[test]
    fn on_a_pairwise_model_it_matches_the_pairwise_path() {
        for l in [4usize, 5, 6] {
            let g = lattice2d(l, 1.0);
            let h = Hubo::from_graph(&g);
            let out = anneal(&h, &Params::default(), 3);
            let bonds = 2.0 * (l * l) as f64;
            assert!((out.energy + bonds).abs() < 1e-9, "{l}x{l}: got {}", out.energy);
        }
    }

    #[test]
    fn a_malformed_term_is_refused_and_an_empty_model_returns() {
        let mut h = Hubo::new(4);
        assert!(h.add(&[], 1.0).is_err(), "an empty term");
        assert!(h.add(&[0, 9], 1.0).is_err(), "a variable off the end");
        assert!(h.add(&[1, 1], 1.0).is_err(), "a repeated variable");
        assert!(h.add(&[0, 1], f64::NAN).is_err(), "a non-finite weight");
        assert_eq!(h.terms(), 0, "nothing malformed was recorded");

        let out = anneal(&Hubo::new(0), &Params::default(), 1);
        assert!(out.state.is_empty() && out.energy == 0.0);
    }
}
