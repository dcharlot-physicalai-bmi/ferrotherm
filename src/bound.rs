//! Lower bounds on the ground energy — how far from optimal a sampler's answer might be.
//!
//! Every sampler in this crate returns the best state it happened to find, and none of them can say
//! whether that is the optimum. [`planted`](crate::planted) answers it by constructing instances
//! whose optimum is known in advance, which only works on instances you built.
//!
//! A lower bound answers it for instances you did not build. If `L <= E(s*)` for every state `s*`,
//! then a sampler holding a state of energy `E` is within `E - L` of optimal, whatever it found and
//! however it found it. When the gap reaches zero the answer is **proven** optimal, and the proof is
//! checkable without trusting the sampler.
//!
//! # This is not an empty lane, and an earlier version of this doc said it was
//!
//! D-Wave's `dwave-preprocessing` has shipped `roof_duality()` for years: it returns a lower bound
//! on a binary quadratic model's energy together with variable assignments that hold at every
//! minimising point. 0.20.0 shipped claiming the field leaves this empty and reports only "best
//! known". That was wrong, and the error was in our reading rather than their library — the same
//! survey that missed it had `dwave-preprocessing` in its own component inventory.
//!
//! What this module is, stated narrowly: a lower bound in a std-only Rust stack, by a *different*
//! relaxation — Lagrangian decomposition rather than roof duality's max-flow construction — and an
//! **anytime** one, since every subgradient round yields a valid bound and the best so far is always
//! available. No comparison against roof duality has been run, so which is tighter on which
//! instances is **unmeasured**. Both are sound, so the maximum of the two is also sound, and that is
//! what a caller with access to both should use.
//!
//! # Where the bound comes from
//!
//! Split the energy into parts, `E(s) = Σ_k E_k(s)`, and minimise each part **independently**:
//!
//! ```text
//! min_s E(s)  =  min_s Σ_k E_k(s)  >=  Σ_k min_s E_k(s)
//! ```
//!
//! The inequality is the whole method: the parts are allowed to disagree about `s`, so their
//! separate minima can only be lower than any single state's total. Sound for **any** split, which
//! is what makes it safe to optimise the split without ever risking an invalid bound.
//!
//! Choose the parts so each one is exactly solvable. [`forest`] splits the couplings into forests,
//! where variable elimination has induced width 1, and shares each field across the parts. Then it
//! tightens the split by subgradient ascent — the parts that disagree about a node exchange field
//! mass until they stop, which is Lagrangian dual decomposition.
//!
//! ```
//! use ferrotherm::{bound, ising::lattice2d};
//!
//! let g = lattice2d(6, 1.0);
//! let b = bound::forest(&g, 60);
//! // Sound: no state can be below it.
//! let ferro = vec![1i8; g.n];
//! assert!(b.value <= g.energy(&ferro) + 1e-9);
//! // And on a ferromagnet the all-up state is optimal, which the gap reports.
//! println!("gap {:.3}", b.gap(&g, &ferro));
//! ```
//!
//! # What it does not do
//!
//! It does not certify a *sample*. [`certify`](crate::certify) asks whether draws came from the
//! right distribution; this asks whether one state is the lowest, and the two questions share
//! nothing but the word "certificate".

use crate::exact::Elimination;
use crate::graph::{Graph, GraphBuilder};

/// A lower bound on `min_s E(s)`, and how it was obtained.
#[derive(Clone, Debug)]
pub struct Bound {
    /// The bound itself. **No state has energy below this.**
    pub value: f64,
    /// How many independently-minimised parts the energy was split into.
    pub parts: usize,
    /// What produced it, for a reader deciding how much to trust the number.
    pub method: &'static str,
    /// Subgradient rounds actually run, for the methods that tighten.
    pub rounds: usize,
    /// Which round produced `value`.
    ///
    /// Reported because it is the only way to see, from outside, that taking the LAST round would
    /// have been worse: subgradient ascent is not monotone, and `best_round < rounds` is a run
    /// where the trajectory dipped after its peak. Without this the difference between "take the
    /// max" and "take the last" is invisible to any test, which is exactly what happened -- the
    /// claim sat in a doc comment with nothing able to check it.
    pub best_round: usize,
}

impl Bound {
    /// How far `s` might be above optimal: `E(s) - value`.
    ///
    /// Never negative for a sound bound, and a negative result means the bound is wrong rather than
    /// the state remarkable — which is why [`forest`] takes the maximum over rounds of quantities
    /// each individually valid, rather than trusting the last one.
    pub fn gap(&self, g: &Graph, s: &[i8]) -> f64 {
        g.energy(s) - self.value
    }

    /// Whether `s` is **proven** optimal: nothing can be lower, so nothing is.
    ///
    /// `tol` absorbs floating-point accumulation over the split; it is not slack in the argument.
    #[must_use = "false does not mean the state is suboptimal, only that this bound does not prove it optimal"]
    pub fn proves_optimal(&self, g: &Graph, s: &[i8], tol: f64) -> bool {
        self.gap(g, s) <= tol
    }
}

/// The bound you get by giving up on every interaction at once.
///
/// `E(s) = -Σ h_i s_i - Σ w_ij s_i s_j`, and each term is at least `-|h_i|` or `-|w_ij|` because a
/// spin product is `±1`. Summing those is a valid bound, achieved only when every term can be
/// satisfied simultaneously — true on an unfrustrated graph and false on anything interesting.
///
/// Loose by construction and worth having anyway: it costs one pass, it never fails, and it is the
/// floor every other method here must beat to have earned its cost.
pub fn decoupled(g: &Graph) -> Bound {
    let mut v = 0.0;
    for i in 0..g.n {
        v -= g.h[i].abs();
        for k in g.offset[i]..g.offset[i + 1] {
            if g.nbr[k] as usize > i {
                v -= g.w[k].abs();
            }
        }
    }
    Bound {
        value: v,
        parts: 1,
        method: "decoupled: every term at its own minimum",
        rounds: 0,
        best_round: 0,
    }
}

/// Split the couplings into forests, minimise each exactly, and tighten the split.
///
/// Each part is a forest, so [`Elimination`] runs at induced width 1 — exact, and linear in the
/// nodes. Fields are shared equally across parts to begin with, then moved by subgradient ascent:
/// where parts disagree about a node's spin, field mass flows toward the ones that are outvoted
/// until they agree or the step size runs out.
///
/// `rounds` bounds the ascent. Zero is legal and gives the untightened split, which is already a
/// valid bound — every round produces one, and this returns **the largest seen**. Taking the last
/// would be a bug: subgradient ascent is not monotone, and the final iterate is routinely worse
/// than one from the middle of the run.
pub fn forest(g: &Graph, rounds: usize) -> Bound {
    let parts = forest_partition(g);
    if parts.is_empty() {
        // No couplings at all: the fields alone are exactly minimisable, and `decoupled` is then
        // not a bound but the answer.
        let mut b = decoupled(g);
        b.method = "no couplings: the field-only optimum, which is exact";
        return b;
    }
    let k = parts.len();
    let elim = Elimination::default();

    // h[p][i]: the share of node i's field carried by part p. Starts equal, and every update below
    // preserves the column sum, so `Σ_p h[p][i] == g.h[i]` at every round -- which is what keeps
    // the decomposition a decomposition rather than a different problem.
    let mut share: Vec<Vec<f64>> = (0..k).map(|_| g.h.iter().map(|&h| h / k as f64).collect()).collect();

    let mut best = f64::NEG_INFINITY;
    let mut best_round = 0usize;
    let mut used = 0usize;
    for r in 0..=rounds {
        let mut total = 0.0;
        let mut states: Vec<Vec<i8>> = Vec::with_capacity(k);
        for (p, edges) in parts.iter().enumerate() {
            let mut gb = GraphBuilder::new(g.n);
            for &(i, j, w) in edges {
                gb.couple(i, j, w);
            }
            for i in 0..g.n {
                gb.bias(i, share[p][i]);
            }
            let part = gb.build();
            // A forest has induced width 1, so this cannot exceed the limit -- but if a future
            // partitioner emitted something denser, falling back to the decoupled bound keeps the
            // result SOUND rather than absent. Never unwrap a bound into a panic.
            match elim.ground_state(&part) {
                Ok(ex) => {
                    total += ex.ground_energy.unwrap_or(f64::NEG_INFINITY);
                    states.push(ex.ground_state.unwrap_or_else(|| vec![1; g.n]));
                }
                Err(_) => return decoupled(g),
            }
        }
        if total > best {
            best = total;
            best_round = r;
        }
        used = r;
        if r == rounds {
            break;
        }

        // Subgradient step. By Danskin, d(min E_p)/d h[p][i] = -s_p[i], and projecting onto the
        // constraint "the shares sum to h_i" subtracts the mean -- so the update is
        // (mean_p s_p[i]) - s_p[i], which is zero exactly where the parts already agree.
        let step = 1.0 / (r as f64 + 1.0);
        for i in 0..g.n {
            let mean: f64 = states.iter().map(|s| s[i] as f64).sum::<f64>() / k as f64;
            for p in 0..k {
                share[p][i] += step * (mean - states[p][i] as f64);
            }
        }
        // Agreement everywhere means the parts found one state, and the inequality that made this a
        // relaxation is tight: the bound IS the optimum and no further round can improve it.
        if (0..g.n).all(|i| states.iter().all(|s| s[i] == states[0][i])) {
            break;
        }
    }

    Bound {
        value: best,
        parts: k,
        method: "forest decomposition, tightened by subgradient ascent on the field split",
        rounds: used,
        best_round,
    }
}

/// Greedily peel spanning forests off the coupling list.
///
/// Each pass takes every edge whose endpoints are not yet joined in that pass, which is a forest by
/// construction. Repeat on what is left. A degree-`d` graph needs at most `d` passes, so a lattice
/// gives four parts and a chain gives one.
fn forest_partition(g: &Graph) -> Vec<Vec<(usize, usize, f64)>> {
    let mut remaining: Vec<(usize, usize, f64)> = Vec::new();
    for i in 0..g.n {
        for k in g.offset[i]..g.offset[i + 1] {
            let j = g.nbr[k] as usize;
            if j > i {
                remaining.push((i, j, g.w[k]));
            }
        }
    }
    let mut parts = Vec::new();
    while !remaining.is_empty() {
        let mut uf: Vec<usize> = (0..g.n).collect();
        let mut forest = Vec::new();
        let mut left = Vec::new();
        for &(i, j, w) in &remaining {
            let (ri, rj) = (find(&mut uf, i), find(&mut uf, j));
            if ri == rj {
                left.push((i, j, w));
            } else {
                uf[ri] = rj;
                forest.push((i, j, w));
            }
        }
        parts.push(forest);
        remaining = left;
    }
    parts
}

fn find(uf: &mut [usize], mut x: usize) -> usize {
    while uf[x] != x {
        uf[x] = uf[uf[x]];
        x = uf[x];
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::GraphBuilder;
    use crate::ising::lattice2d;
    use crate::rng::Pcg;

    /// Brute force, for instances small enough that the truth is available.
    fn true_min(g: &Graph) -> f64 {
        let mut best = f64::INFINITY;
        for mask in 0u32..(1u32 << g.n) {
            let s: Vec<i8> = (0..g.n).map(|i| if mask >> i & 1 == 1 { 1 } else { -1 }).collect();
            best = best.min(g.energy(&s));
        }
        best
    }

    fn random_graph(n: usize, p: f64, seed: u64) -> Graph {
        let mut rng = Pcg::new(seed, 0xB0);
        let mut gb = GraphBuilder::new(n);
        for i in 0..n {
            gb.bias(i, rng.f64() * 2.0 - 1.0);
            for j in (i + 1)..n {
                if rng.f64() < p {
                    gb.couple(i, j, rng.f64() * 2.0 - 1.0);
                }
            }
        }
        gb.build()
    }

    #[test]
    fn a_bound_is_never_above_the_true_minimum() {
        // SOUNDNESS, which is the only property that matters. A bound above the optimum does not
        // report a small gap -- it reports a NEGATIVE one, and every conclusion drawn from it is
        // backwards. Checked against brute force on 200 random instances rather than argued.
        for seed in 0..200u64 {
            let g = random_graph(10, 0.4, seed);
            let truth = true_min(&g);
            for b in [decoupled(&g), forest(&g, 0), forest(&g, 25)] {
                assert!(
                    b.value <= truth + 1e-9,
                    "seed {seed}: {} gave {} above the true minimum {truth}",
                    b.method,
                    b.value
                );
            }
        }
    }

    #[test]
    fn tightening_helps_and_never_hurts() {
        // Every round yields an individually valid bound, so the maximum over rounds is valid too
        // -- and taking the LAST would be a bug, because subgradient ascent is not monotone.
        let mut improved = 0;
        for seed in 0..40u64 {
            let g = random_graph(12, 0.35, seed);
            let cold = forest(&g, 0).value;
            let warm = forest(&g, 40).value;
            assert!(warm >= cold - 1e-9, "seed {seed}: tightening lost ground, {cold} -> {warm}");
            if warm > cold + 1e-6 {
                improved += 1;
            }
        }
        assert!(improved > 20, "tightening improved only {improved}/40; it is not earning its cost");
    }

    #[test]
    fn the_bound_is_the_best_round_not_the_last_one() {
        // The claim that sat in a doc comment with nothing able to check it. Subgradient ascent is
        // not monotone, so the final iterate is routinely worse than one from the middle -- and a
        // mutation replacing `if total > best` with `best = total` passed the entire suite, because
        // `forest(g, r)` already maximises over rounds 0..r and no test could see inside that.
        //
        // Measured, not assumed: across 200 random instances at 40 rounds, 145 peak BEFORE the last
        // round. Seed 1 is one of them, and `best_round < rounds` is the observable that says so.
        let g = random_graph(14, 0.35, 1);
        let b = forest(&g, 40);
        assert_eq!(b.rounds, 40, "this instance should run the full ladder, not stop early");
        assert!(
            b.best_round < b.rounds,
            "seed 1 was chosen because its trajectory dips; if it no longer does, this test is \
             blind and needs a new instance rather than deleting"
        );
        // Which means a take-the-last implementation returns something strictly worse here.
        let truncated = forest(&g, b.best_round);
        assert!(
            (truncated.value - b.value).abs() < 1e-9,
            "stopping at the peak must reproduce the bound: {} vs {}",
            truncated.value,
            b.value
        );
    }

    #[test]
    fn a_forest_is_solved_exactly_so_the_gap_closes() {
        // One part, no relaxation: the decomposition is the problem itself, and the bound has to be
        // the optimum rather than merely below it. A chain is a forest.
        let mut gb = GraphBuilder::new(9);
        for i in 0..8 {
            gb.couple(i, i + 1, if i % 2 == 0 { 1.0 } else { -0.7 });
        }
        gb.bias(3, 0.4);
        gb.bias(7, -0.9);
        let g = gb.build();
        let b = forest(&g, 5);
        assert_eq!(b.parts, 1, "a chain needs one forest");
        let truth = true_min(&g);
        assert!((b.value - truth).abs() < 1e-9, "chain bound {} vs exact {truth}", b.value);
    }

    #[test]
    fn the_ferromagnet_is_proven_optimal_rather_than_merely_unbeaten() {
        // The payoff. An unfrustrated lattice's all-up state is optimal, and the point is that this
        // says so from the bound alone -- no enumeration, no planted answer, no appeal to how long
        // somebody searched.
        let g = lattice2d(6, 1.0);
        let b = forest(&g, 80);
        let up = vec![1i8; g.n];
        assert!(
            b.proves_optimal(&g, &up, 1e-6),
            "gap {:.6} on an unfrustrated lattice; the bound should close",
            b.gap(&g, &up)
        );
        // And it is a real proof, not a tautology: a worse state has a positive gap.
        let mut mixed = up.clone();
        mixed[0] = -1;
        assert!(b.gap(&g, &mixed) > 1.0, "flipping a spin must open the gap");
    }

    #[test]
    fn the_forest_split_beats_the_decoupled_floor_where_there_is_room_to() {
        // FRUSTRATED, and the first version of this test was not -- it used a ferromagnetic
        // lattice, where every bond and field can be satisfied at once, so the decoupled floor of
        // -72 IS the optimum and nothing can beat it. The two bounds agreed exactly and the test
        // read that as the forest split failing to earn its cost. There is no room above a tight
        // bound; the question only means anything where the floor is loose.
        let g = random_graph(14, 0.35, 7);
        let (d, f) = (decoupled(&g), forest(&g, 60));
        assert!(f.parts >= 2, "a graph this dense does not fit in one forest");
        assert!(
            f.value > d.value + 1e-6,
            "forest {} did not beat decoupled {} on a frustrated instance",
            f.value,
            d.value
        );
        assert_eq!(d.parts, 1);

        // On an unfrustrated lattice they must instead AGREE, and both be exact.
        let ferro = lattice2d(6, 1.0);
        let (dd, ff) = (decoupled(&ferro), forest(&ferro, 20));
        assert!((dd.value - ff.value).abs() < 1e-9, "{} vs {}", dd.value, ff.value);
        assert!((ff.value - ferro.energy(&vec![1i8; ferro.n])).abs() < 1e-9);
    }

    #[test]
    fn a_graph_with_no_couplings_is_solved_rather_than_bounded() {
        let mut gb = GraphBuilder::new(5);
        for i in 0..5 {
            gb.bias(i, (i as f64) - 2.0);
        }
        let g = gb.build();
        let b = forest(&g, 3);
        assert!(b.method.contains("exact"), "{}", b.method);
        assert!((b.value - true_min(&g)).abs() < 1e-12);
    }

    #[test]
    fn the_partition_covers_every_edge_exactly_once() {
        // The decomposition identity `E = Σ_k E_k` requires it. A dropped edge makes the bound
        // apply to a DIFFERENT problem -- still a number, still plausible, and about nothing.
        let g = lattice2d(5, 1.0);
        let parts = forest_partition(&g);
        let mut seen: Vec<(usize, usize)> = parts
            .iter()
            .flat_map(|p| p.iter().map(|&(i, j, _)| (i, j)))
            .collect();
        let total: usize = parts.iter().map(|p| p.len()).sum();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen.len(), total, "an edge appears in two parts");
        let edges = (0..g.n).map(|i| g.offset[i + 1] - g.offset[i]).sum::<usize>() / 2;
        assert_eq!(total, edges, "the parts drop {} edge(s)", edges - total);
    }
}
