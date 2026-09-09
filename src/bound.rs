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
//! # Where `forest` is worth nothing: max-cut
//!
//! A forest is **never frustrated**. Any tree two-colours, so every edge in it can be satisfied at
//! once and the part's minimum is exactly `-Σ|J|` over its own edges. Sum that across parts and the
//! forest bound **is** [`decoupled`] — the trivial floor — whenever the graph carries no fields for
//! the subgradient to redistribute.
//!
//! That is exactly the G-set case, and it is not a corner: measured on real instances, where
//! `fields_all_zero` is true and the two bounds agree to the last digit.
//!
//! ```text
//!   G11:  decoupled -1600   forest -1600   -Σ|w| -1600
//!   G14:  decoupled -4694   forest -4694   -Σ|w| -4694
//! ```
//!
//! `forest` reports `best_round = 0` there — forty subgradient rounds improving nothing, because
//! with `h = 0` there is no field mass to move. **Trees cannot see the only thing that makes
//! max-cut hard**, which is odd cycles. Use [`odd_cycle`] on such instances, and prefer the maximum
//! of the two in general, since both are sound.
//!
//! # What it does not do
//!
//! It does not certify a *sample*. [`certify`](crate::certify) asks whether draws came from the
//! right distribution; this asks whether one state is the lowest, and the two questions share
//! nothing but the word "certificate".

use crate::round::{accumulation_guard, sum_down, sum_up};
use crate::exact::Elimination;
use crate::graph::{Graph, GraphBuilder};

/// A lower bound on `min_s E(s)`, and how it was obtained.
#[derive(Clone, Debug)]
pub struct Bound {
    /// The bound itself. **No state has energy below this** — including after rounding, which is
    /// a promise this field made for two releases without keeping it. Every method here now
    /// accumulates through [`crate::round`], so the value is below the exact relaxation rather
    /// than near it.
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
    ///
    /// That sentence was true and unchecked for two releases while `forest` returned negative gaps
    /// on a third of random trees. `no_bound_is_ever_above_a_state_it_bounds` checks it now.
    #[must_use]
    pub fn gap(&self, g: &Graph, s: &[i8]) -> f64 {
        // BOTH SIDES OR NEITHER. The gap must be an OVER-estimate of the true gap for
        // `proves_optimal` to be a proof, which needs the energy rounded UP and the bound rounded
        // DOWN. Fixing only the bound is worse than fixing neither, and that is measured rather
        // than argued: `decoupled` accumulates in exactly `Graph::energy`'s order, so its two
        // roundings cancelled and the gap came out exactly zero on every one of 4800 random trees.
        // Making the bound sound on its own broke that cancellation and produced 78 NEGATIVE gaps
        // where there had been none.
        sum_up(&[energy_up(g, s), -self.value])
    }

    /// Whether `s` is **proven** optimal: nothing can be lower, so nothing is.
    ///
    /// **`tol` is slack you choose, and `0.0` is now a legitimate choice.** It used to have a
    /// second job — absorbing the floating-point accumulation over the split — which made the
    /// honest value of `tol` unknowable: too small and a true optimum was rejected, too large and
    /// the proof was not one. Both sides of the comparison are directed now ([`gap`](Self::gap)),
    /// so the arithmetic is accounted for inside the numbers rather than inside the caller's
    /// tolerance.
    #[must_use = "false does not mean the state is suboptimal, only that this bound does not prove it optimal"]
    pub fn proves_optimal(&self, g: &Graph, s: &[i8], tol: f64) -> bool {
        self.gap(g, s) <= tol
    }
}

/// `E(s)`, never below the exact value.
///
/// Each term is exact — a spin is `±1`, so `h·s` and `w·s·s` are sign flips — and only the
/// summation rounds, which is why one directed sum is enough and no interval type is needed.
fn energy_up(g: &Graph, s: &[i8]) -> f64 {
    let mut terms = Vec::with_capacity(g.n + g.n_edges);
    for i in 0..g.n {
        let si = f64::from(s[i]);
        terms.push(-g.h[i] * si);
        for k in g.offset[i]..g.offset[i + 1] {
            let j = g.nbr[k] as usize;
            if j > i {
                terms.push(-g.w[k] * si * f64::from(s[j]));
            }
        }
    }
    sum_up(&terms)
}

/// The bound you get by giving up on every interaction at once.
///
/// `E(s) = -Σ h_i s_i - Σ w_ij s_i s_j`, and each term is at least `-|h_i|` or `-|w_ij|` because a
/// spin product is `±1`. Summing those is a valid bound, achieved only when every term can be
/// satisfied simultaneously — true on an unfrustrated graph and false on anything interesting.
///
/// Loose by construction and worth having anyway: it costs one pass, it never fails, and it is the
/// floor every other method here must beat to have earned its cost.
#[must_use]
pub fn decoupled(g: &Graph) -> Bound {
    // COLLECTED, NOT ACCUMULATED. `v -= x` rounds to nearest, and a lower bound that rounds up is
    // not a lower bound. See [`crate::round`] for the defect this class of `+=` produced in
    // [`forest`], which is the same arithmetic on a different sum.
    //
    // Nothing was measured wrong here, and the reason is worth writing down rather than relying
    // on: this loop visits the terms in exactly the order `Graph::energy` does, so for the state
    // that achieves the bound the two accumulate bit-identically and the gap is exactly zero. That
    // is an argument about one state. `value` promises something about EVERY state, and a
    // different state gives the same magnitudes different signs and so a different rounding.
    let mut terms = Vec::with_capacity(g.n + g.n_edges);
    for i in 0..g.n {
        terms.push(-g.h[i].abs());
        for k in g.offset[i]..g.offset[i + 1] {
            if g.nbr[k] as usize > i {
                terms.push(-g.w[k].abs());
            }
        }
    }
    Bound {
        value: sum_down(&terms),
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
#[must_use]
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
    // Reused across rounds so the soundness guard costs no allocation per round.
    let mut part_energies: Vec<f64> = Vec::with_capacity(k);
    let mut guards: Vec<f64> = Vec::with_capacity(k);
    let mut mags: Vec<f64> = Vec::with_capacity(g.n + g.n_edges);

    for r in 0..=rounds {
        part_energies.clear();
        guards.clear();
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
                    part_energies.push(ex.ground_energy.unwrap_or(f64::NEG_INFINITY));
                    // `ground_energy` is a float this function did not compute, and its rounding is
                    // the whole defect: summing these with `+=` reported a NEGATIVE gap on 1688 of
                    // 4800 random trees, where the bound is exact and the gap must be zero.
                    //
                    // The guard covers both halves of what elimination can get wrong -- the
                    // additions inside it, and an argmin chosen between two states whose computed
                    // energies differ by less than that error. One addition per field and two per
                    // coupling is a generous count, which is the direction to be generous in.
                    mags.clear();
                    mags.extend(share[p].iter().map(|x| x.abs()));
                    mags.extend(edges.iter().map(|&(_, _, w)| w.abs()));
                    guards.push(accumulation_guard(g.n + 2 * edges.len(), sum_up(&mags)));
                    states.push(ex.ground_state.unwrap_or_else(|| vec![1; g.n]));
                }
                Err(_) => return decoupled(g),
            }
        }
        // Downward for the parts, upward for the guard subtracted from them: both directions point
        // the same way, which is away from claiming more than was proved.
        let total = sum_down(&part_energies) - sum_up(&guards);
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

/// A bound that can see frustration: the decoupled floor, plus what odd cycles must cost.
///
/// A cycle is **frustrated** when the product of its coupling signs is negative — for max-cut,
/// where every `J` is negative, that is exactly an odd-length cycle. Around such a cycle no
/// assignment satisfies every edge, so at least one is violated and its term flips from `-|J|` to
/// `+|J|`: the cycle costs at least `2·min|J|` above the decoupled floor.
///
/// Those penalties **add** across EDGE-DISJOINT cycles, because no edge is asked to be violated
/// twice. Sharing an edge would double-count the one violation that pays for both, so this claims
/// each edge at most once and skips any cycle whose edges are already spoken for.
///
/// `max_len` caps the search. Short cycles are worth more per edge spent, and an uncapped search on
/// a degree-48 graph is a different program.
pub fn odd_cycle(g: &Graph, max_len: usize) -> Bound {
    let base = decoupled(g);
    if max_len < 3 {
        return base;
    }
    // Edges, indexed so "claimed" is a bitset over them rather than a set of pairs.
    let mut eid = std::collections::BTreeMap::new();
    let mut ew: Vec<f64> = Vec::new();
    for i in 0..g.n {
        for k in g.offset[i]..g.offset[i + 1] {
            let j = g.nbr[k] as usize;
            if j > i {
                eid.insert((i, j), ew.len());
                ew.push(g.w[k]);
            }
        }
    }
    let key = |a: usize, b: usize| if a < b { (a, b) } else { (b, a) };
    let mut claimed = vec![false; ew.len()];
    // `base.value` leads the slice: the sum of the floor and every penalty must round down
    // together, not floor-then-add.
    let mut penalties: Vec<f64> = vec![base.value];
    let mut cycles = 0usize;

    // For each still-free edge, look for the shortest cycle closing it through free edges only.
    for i in 0..g.n {
        for k in g.offset[i]..g.offset[i + 1] {
            let j = g.nbr[k] as usize;
            if j <= i {
                continue;
            }
            let e0 = eid[&key(i, j)];
            if claimed[e0] {
                continue;
            }
            // BFS from j back to i without reusing edge (i,j) or any claimed edge.
            let mut prev: Vec<Option<usize>> = vec![None; g.n];
            let mut seen = vec![false; g.n];
            seen[j] = true;
            let mut q = std::collections::VecDeque::from([(j, 0usize)]);
            let mut path: Option<Vec<usize>> = None;
            while let Some((u, d)) = q.pop_front() {
                if d + 1 >= max_len {
                    continue;
                }
                for kk in g.offset[u]..g.offset[u + 1] {
                    let v = g.nbr[kk] as usize;
                    let e = eid[&key(u, v)];
                    if e == e0 || claimed[e] {
                        continue;
                    }
                    if v == i {
                        // Walk back to build the cycle's vertex list.
                        let mut p = vec![i, u];
                        let mut cur = u;
                        while let Some(pp) = prev[cur] {
                            p.push(pp);
                            cur = pp;
                        }
                        path = Some(p);
                        break;
                    }
                    if !seen[v] {
                        seen[v] = true;
                        prev[v] = Some(u);
                        q.push_back((v, d + 1));
                    }
                }
                if path.is_some() {
                    break;
                }
            }
            let Some(p) = path else { continue };
            // Edges of the cycle: consecutive pairs, plus the closing edge (i,j).
            let mut edges = vec![e0];
            let mut ok = true;
            for w in p.windows(2) {
                let e = eid[&key(w[0], w[1])];
                if claimed[e] || edges.contains(&e) {
                    ok = false;
                    break;
                }
                edges.push(e);
            }
            if !ok || edges.len() < 3 {
                continue;
            }
            // FRUSTRATED? The product of signs decides it. An even number of negative couplings
            // means the cycle can be satisfied, and claiming its edges would spend them for nothing.
            let negatives = edges.iter().filter(|&&e| ew[e] < 0.0).count();
            if negatives % 2 == 0 {
                continue;
            }
            let min_abs = edges.iter().map(|&e| ew[e].abs()).fold(f64::INFINITY, f64::min);
            if !min_abs.is_finite() || min_abs <= 0.0 {
                continue;
            }
            for e in edges {
                claimed[e] = true;
            }
            // Collected, not accumulated: this RAISES the floor, so an overstated penalty is a
            // bound above what it bounds. `2.0 * min_abs` is exact; the sum of them is not.
            penalties.push(2.0 * min_abs);
            cycles += 1;
        }
    }
    Bound {
        value: sum_down(&penalties),
        parts: cycles,
        method: "decoupled floor plus 2*min|J| per edge-disjoint frustrated cycle",
        rounds: 0,
        best_round: 0,
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

    /// A BOUND THAT IS ABOVE WHAT IT BOUNDS IS NOT A BOUND, and the magnitude is not the point.
    ///
    /// `a_forest_is_solved_exactly_so_the_gap_closes` asserts the forest bound is within `1e-9` of
    /// the optimum, which is five orders of magnitude too loose to see the defect this test exists
    /// for: `forest` accumulated its parts with `+=`, and on random trees — where the bound is
    /// exact and the gap must be zero — **1688 of 4800 trials reported a NEGATIVE gap**, worst
    /// `−7.8e-14`. Every one of those passed the `1e-9` check.
    ///
    /// So this asserts the SIGN. A negative gap means the bound is wrong rather than the state
    /// remarkable, which [`Bound::gap`]'s own documentation had said since it was written with
    /// nothing able to check it.
    ///
    /// Both directions are exercised on purpose. Fixing only the bound made this WORSE for
    /// `decoupled` — 0 negative gaps became 78 — because its accumulation order matches
    /// `Graph::energy`'s exactly, so the two roundings had been cancelling. Soundness here is a
    /// property of the PAIR: bound down, energy up.
    #[test]
    fn no_bound_is_ever_above_a_state_it_bounds() {
        let mut rng = Pcg::new(7, 0x5EED);
        let mut probe = Vec::new();
        let (mut worst_f, mut worst_d, mut worst_c) = (f64::INFINITY, f64::INFINITY, f64::INFINITY);
        let mut trials = 0usize;

        for fields in [false, true] {
            for n in [8usize, 16, 32, 64, 128] {
                for seed in 0..40u64 {
                    // A random tree: unfrustrated in its couplings, so `forest` puts it in ONE part
                    // and the bound is EXACT. That is what makes the sign readable — on a loose
                    // bound a rounding error hides under the looseness.
                    let mut gb = GraphBuilder::new(n);
                    let mut r = Pcg::new(seed, 0xB0 + u64::from(fields));
                    for i in 1..n {
                        let par = (r.f64() * i as f64) as usize;
                        // Off the binary grid, so no sum here is exact by construction.
                        gb.couple(par.min(i - 1), i, (r.f64() - 0.5) * 0.2 + 0.1);
                    }
                    if fields {
                        for i in 0..n {
                            gb.bias(i, (r.f64() - 0.5) * 0.2 + 0.05);
                        }
                    }
                    let g = gb.build();
                    let Ok(ex) = Elimination::default().ground_state(&g) else { continue };
                    let Some(gs) = ex.ground_state else { continue };
                    trials += 1;

                    let (bf, bd, bc) = (forest(&g, 0), decoupled(&g), odd_cycle(&g, 5));
                    for b in [&bf, &bd, &bc] {
                        assert!(
                            b.gap(&g, &gs) >= 0.0,
                            "{}: gap {:e} at the ground state of a {n}-node tree (seed {seed}, \
                             fields {fields}) -- the bound is above the optimum",
                            b.method,
                            b.gap(&g, &gs)
                        );
                    }
                    worst_f = worst_f.min(bf.gap(&g, &gs));
                    worst_d = worst_d.min(bd.gap(&g, &gs));
                    worst_c = worst_c.min(bc.gap(&g, &gs));

                    // And against arbitrary states, because `value` promises something about every
                    // state and not only about the one that achieves it.
                    probe.clear();
                    probe.resize(n, 1i8);
                    for _ in 0..32 {
                        for x in &mut probe {
                            *x = if rng.f64() < 0.5 { -1 } else { 1 };
                        }
                        for b in [&bf, &bd, &bc] {
                            assert!(
                                b.gap(&g, &probe) >= 0.0,
                                "{}: a random state fell BELOW the bound by {:e}",
                                b.method,
                                b.gap(&g, &probe)
                            );
                        }
                    }
                }
            }
        }
        assert!(trials > 300, "the sweep must actually run: {trials} trials");
        // The bound stays tight while it is sound: the guard costs ulps, not accuracy. A tolerance
        // here would let a future guard grow without limit and still pass.
        for (what, w) in [("forest", worst_f), ("decoupled", worst_d), ("odd_cycle", worst_c)] {
            assert!(w < 1e-9, "{what}: soundness cost {w:e}, which is no longer a rounding guard");
        }
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
    fn the_odd_cycle_bound_is_sound_too() {
        // Same brute-force check the forest bound gets. A bound that sees frustration is worth
        // nothing if it ever climbs above the truth.
        for seed in 0..150u64 {
            let g = random_graph(9, 0.45, seed);
            let truth = true_min(&g);
            let b = odd_cycle(&g, 6);
            assert!(
                b.value <= truth + 1e-9,
                "seed {seed}: odd_cycle gave {} above the true minimum {truth}",
                b.value
            );
        }
    }

    #[test]
    fn a_triangle_is_frustrated_and_the_bound_says_so() {
        // The smallest frustrated object. Three antiferromagnetic bonds cannot all be satisfied,
        // so the optimum sits 2 above the decoupled floor of -3 -- exactly the 2*min|J| the cycle
        // term adds.
        let mut gb = GraphBuilder::new(3);
        for (i, j) in [(0, 1), (1, 2), (2, 0)] {
            gb.couple(i, j, -1.0);
        }
        let g = gb.build();
        assert!((true_min(&g) - (-1.0)).abs() < 1e-9, "a frustrated triangle bottoms out at -1");
        assert!((decoupled(&g).value - (-3.0)).abs() < 1e-9, "the trivial floor is -3");
        let b = odd_cycle(&g, 4);
        assert_eq!(b.parts, 1, "one cycle claimed");
        assert!((b.value - (-1.0)).abs() < 1e-9, "cycle bound {} should be exact here", b.value);
    }

    #[test]
    fn an_unfrustrated_cycle_is_not_charged_for() {
        // An even cycle of antiferromagnets two-colours perfectly, so there is nothing to claim --
        // and spending its edges would leave less for cycles that are frustrated.
        let mut gb = GraphBuilder::new(4);
        for (i, j) in [(0, 1), (1, 2), (2, 3), (3, 0)] {
            gb.couple(i, j, -1.0);
        }
        let g = gb.build();
        let b = odd_cycle(&g, 5);
        assert_eq!(b.parts, 0, "no frustrated cycle exists here");
        assert!((b.value - decoupled(&g).value).abs() < 1e-12);
    }

    #[test]
    fn the_cycle_bound_beats_the_forest_bound_where_the_forest_bound_is_blind() {
        // THE FINDING, as a test. With no fields `forest` degenerates to `decoupled`, because a
        // tree is never frustrated -- measured on G11 and G14, where the two agree to the last
        // digit. Two disjoint triangles are that situation in miniature.
        let mut gb = GraphBuilder::new(6);
        for (i, j) in [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)] {
            gb.couple(i, j, -1.0);
        }
        let g = gb.build();
        assert!(g.h.iter().all(|&h| h == 0.0), "no fields, which is the G-set case");
        let (f, c) = (forest(&g, 40), odd_cycle(&g, 4));
        assert!(
            (f.value - decoupled(&g).value).abs() < 1e-9,
            "forest {} should degenerate to decoupled {}",
            f.value,
            decoupled(&g).value
        );
        assert!(c.value > f.value + 1e-9, "cycle {} must beat forest {}", c.value, f.value);
        assert!(c.value <= true_min(&g) + 1e-9, "and still be sound");
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
        let total: usize = parts.iter().map(std::vec::Vec::len).sum();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen.len(), total, "an edge appears in two parts");
        let edges = (0..g.n).map(|i| g.offset[i + 1] - g.offset[i]).sum::<usize>() / 2;
        assert_eq!(total, edges, "the parts drop {} edge(s)", edges - total);
    }
}
