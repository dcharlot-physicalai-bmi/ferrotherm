//! Tabu search — the baseline every new max-cut heuristic is measured against.
//!
//! Worth stating why this is here rather than in a footnote. A 2024 benchmark study of learned
//! heuristics for max-cut concluded that **tabu search outperforms all of the learned methods
//! evaluated** on objective value, scalability and generalisation — with one exception — and
//! recommended it be included as a baseline for any future heuristic claiming to solve max-cut.
//! A sampling stack that cannot run it has no way to know whether its physics is earning anything.
//!
//! # The move evaluation is the whole algorithm
//!
//! Flipping spin `i` changes the energy by exactly
//!
//! ```text
//! Δ_i = 2 s_i ( h_i + Σ_j J_ij s_j )
//! ```
//!
//! Recomputing that for every node each iteration is `O(n·d)` per move and is what makes a naive
//! tabu search useless at G-set sizes. Keeping the whole `Δ` vector incrementally makes a move
//! `O(d)`: flipping `j` changes `Δ_j` in sign and changes `Δ_i` only for `i` adjacent to `j`.
//! That single decision is the difference between seconds and hours on an 800-node instance.
//!
//! # Tabu, and the exception to it
//!
//! A flipped spin is forbidden for exactly `tenure` iterations, which is what stops the search walking
//! straight back down the hill it just climbed. The **aspiration** rule overrides the ban when a
//! move would beat the best state ever seen: refusing a new global best because of bookkeeping is
//! the one case where the tabu list is certainly wrong.
//!
//! ```
//! use ferrotherm::{tabu, ising::lattice2d};
//!
//! let g = lattice2d(8, 1.0);           // ferromagnet: all-up is optimal
//! let r = tabu::search(&g, &tabu::Params::default(), 7);
//! assert!((r.energy - g.energy(&r.state)).abs() < 1e-9);
//! assert!(r.energy <= -2.0 * 64.0 + 1e-9, "should reach the ground state");
//! ```
//!
//! # What choosing a move costs, and why it is still a scan
//!
//! **Each iteration scans every gain.** The incremental update after a flip is already `O(degree)`
//! (`flip`), so the linear part is the argmin alone — the same gap [`crate::bls`] documents,
//! which uses Fiduccia–Mattheyses buckets in the literature to get it in `O(1)`. Until now that
//! cost was written down for `bls` and not for this module, which has it identically.
//!
//! **The obvious fix does not apply here, and that is worth writing down rather than leaving as an
//! open task somebody re-derives.** A max-gain heap answers "the best move"; this needs "the best
//! ADMISSIBLE move", and admissibility is not a property of the gain. A move is inadmissible while
//! it is tabu — which depends on `iter` and expires on its own, so a node becomes admissible again
//! with no change to its gain and nothing to trigger a heap update — and it is admissible anyway if
//! it beats the best state ever seen, which depends on the current energy and moves every
//! iteration. A heap keyed on gain would have to pop through inadmissible entries and push them
//! back, with a worst case of the scan it replaced.
//!
//! It would also change every seeded result. The scan takes the FIRST minimum, so ties break by
//! lowest index; a heap breaks them by heap order, and this crate's contract is that a seed
//! reproduces a run. Preserving the tie-break means keying on `(gain, index)` and handling the
//! time-varying admissibility on top, which is a different piece of work from "add a heap".
//!
//! So it stays a scan, and the cost is stated instead of hidden. What would justify the rewrite is
//! a workload that needs it — the paper's `200000·|V|` budget, which `bls` names as the thing this
//! shape of implementation cannot reach.

use crate::graph::Graph;
use crate::ledger::Ledger;
use crate::rng::Pcg;

/// How the search is run.
// NOT `Copy`: `start` holds a state, and a search that can be handed one is worth more
// than a Params a caller can pass twice without saying `.clone()`. `branch::Params` has
// carried a `Vec` incumbent since it existed, so this is the crate's own precedent.
#[derive(Clone, Debug)]
pub struct Params {
    /// Iterations to run. One iteration is one flip.
    pub iterations: usize,
    /// How many iterations a flipped spin stays forbidden.
    ///
    /// Zero disables the tabu list, which turns this into steepest descent and gets stuck in the
    /// first local minimum — worth being able to express, because it is the control that shows the
    /// tabu list is doing something.
    pub tenure: usize,
    /// Restart from a fresh random state after this many iterations with no improvement.
    ///
    /// `None` never restarts.
    pub restart_after: Option<usize>,
    /// A state to start from, if one is already known -- from an anneal, or from [`crate::hfs`].
    ///
    /// `None` starts from noise, which is what this did and only did. [`crate::branch::Params`] has
    /// carried an `incumbent` all along and these did not, so a caller holding a good state had no
    /// way to hand it over: composing meant running this FIRST and something else after, never the
    /// other way round.
    ///
    /// A wrong length is ignored and the search starts from noise, rather than returning a `Result`
    /// on a search that cannot otherwise fail.
    ///
    /// Restarts still go to noise. A restart exists to leave the basin the search is in, and
    /// restarting to the state it was handed would put it back where it could not escape from.
    pub start: Option<Vec<i8>>,
}

impl Default for Params {
    fn default() -> Self {
        // Tenure scaled to the instance is the usual advice; a fixed tenure is either too sticky on
        // small graphs or too loose on large ones. Set at call time by `search` when this is used.
        Params { iterations: 50_000, tenure: 0, restart_after: Some(5_000), start: None }
    }
}

/// What a search found.
#[derive(Clone, Debug)]
pub struct Outcome {
    /// The best state seen, not the last one.
    pub state: Vec<i8>,
    /// Its energy, recomputed from `state` rather than accumulated.
    pub energy: f64,
    /// Iteration at which the best state was found — the rest of the run improved nothing.
    pub found_at: usize,
    /// How many times the search restarted.
    pub restarts: usize,
    /// Iterations actually executed.
    ///
    /// Reported because truncation is otherwise INVISIBLE from outside. The first version of this
    /// module returned early when every move was tabu, spending 9 iterations of a 50,000-iteration
    /// budget and handing back a result that looked exactly like a completed run — worse on small
    /// graphs, for a reason no field of `Outcome` could express. `iterations_run < p.iterations`
    /// now says so.
    pub iterations_run: usize,
}

/// Run tabu search from a seeded random state.
///
/// `tenure` of zero in `p` is replaced by `max(10, n/10)`, the usual scaling. Pass a non-zero
/// tenure to override, or pass one deliberately with `restart_after: None` to get plain descent.
pub fn search(g: &Graph, p: &Params, seed: u64) -> Outcome {
    search_metered(g, p, seed, None)
}

/// The same, charging the ledger one sample per evaluated move.
///
/// A tabu iteration evaluates every node's move and commits one, so the honest count is `n` per
/// iteration — the same quantity a Gibbs sweep charges. Charging one per *flip* would make tabu
/// look `n` times cheaper than a sweep doing comparable work.
pub fn search_metered(g: &Graph, p: &Params, seed: u64, mut ledger: Option<&mut Ledger>) -> Outcome {
    let n = g.n;
    if n == 0 {
        return Outcome { state: Vec::new(), energy: 0.0, found_at: 0, restarts: 0, iterations_run: 0 };
    }
    // CAPPED AT n-1, which is what makes the deadlock below impossible rather than merely rare.
    //
    // `Params::default()` asks for `max(10, n/10)`, so every graph with n <= 10 got a tenure at
    // least as large as its spin count -- every spin tabu, nothing admissible, and the search
    // exited after n iterations of a 50,000-iteration budget. Measured before the cap: n=9 ran 9
    // iterations and missed the brute-force optimum on 7 of 30 seeds, while n=10 ran the full
    // budget and missed none. Leaving at least one spin admissible is a property of the tenure, not
    // something to detect afterwards.
    let tenure = if p.tenure == 0 { (n / 10).max(10) } else { p.tenure };
    let tenure = tenure.min(n.saturating_sub(1)).max(1);
    let mut rng = Pcg::new(seed, 0x7AB0);

    let mut s: Vec<i8> = match &p.start {
        Some(st) if st.len() == n => st.clone(),
        _ => (0..n).map(|_| rng.spin(0.5)).collect(),
    };
    let mut delta = gains(g, &s);
    let mut energy = g.energy(&s);
    let mut best = s.clone();
    let mut best_e = energy;
    let mut found_at = 0usize;
    let mut restarts = 0usize;
    // `last_used[i]` is the iteration `i` was last flipped; it is tabu while
    // `iter < last_used[i] + tenure`. Storing the iteration rather than a countdown means no
    // per-iteration sweep over the list.
    let mut last_used = vec![usize::MAX; n];
    let mut since_improve = 0usize;
    let mut ran = 0usize;

    for iter in 0..p.iterations {
        ran = iter + 1;
        if let Some(l) = ledger.as_deref_mut() {
            l.samples += n as u64;
        }
        // Best admissible move. A move is admissible when it is not tabu, or when taking it would
        // beat the best state ever seen -- the aspiration criterion.
        let mut pick = usize::MAX;
        let mut pick_d = f64::INFINITY;
        for i in 0..n {
            let d = delta[i];
            // `<=`, so a spin flipped at t is banned on t+1 ..= t+tenure -- exactly `tenure`
            // iterations. With `<` it was tenure-1, and `tenure: 1` banned nothing at all, which
            // this module's own descent-control test had quietly come to rely on.
            let is_tabu = last_used[i] != usize::MAX && iter <= last_used[i].saturating_add(tenure);
            let aspires = energy + d < best_e - 1e-12;
            if (!is_tabu || aspires) && d < pick_d {
                pick_d = d;
                pick = i;
            }
        }
        if pick == usize::MAX {
            // Unreachable while `tenure <= n-1`, because at least one spin is always admissible.
            // Kept as a restart rather than a `break`: the original returned here, silently
            // spending n iterations of a 50,000 budget and reporting the result as if the budget
            // had been used. A truncated search that reports success is worse than a slow one.
            s = (0..n).map(|_| rng.spin(0.5)).collect();
            delta = gains(g, &s);
            energy = g.energy(&s);
            last_used.iter_mut().for_each(|v| *v = usize::MAX);
            since_improve = 0;
            restarts += 1;
            continue;
        }

        flip(g, &mut s, &mut delta, pick);
        energy += pick_d;
        last_used[pick] = iter;

        if energy < best_e - 1e-12 {
            best_e = energy;
            best.copy_from_slice(&s);
            found_at = iter;
            since_improve = 0;
        } else {
            since_improve += 1;
        }

        if let Some(after) = p.restart_after {
            if since_improve >= after {
                s = (0..n).map(|_| rng.spin(0.5)).collect();
                delta = gains(g, &s);
                energy = g.energy(&s);
                last_used.iter_mut().for_each(|v| *v = usize::MAX);
                since_improve = 0;
                restarts += 1;
            }
        }
    }

    // Recomputed, not carried. `energy` is accumulated from deltas across thousands of flips and
    // drifts; the number reported has to be the one the returned STATE actually has.
    let energy = g.energy(&best);
    Outcome { state: best, energy, found_at, restarts, iterations_run: ran }
}

/// `Δ_i` for every node: the energy change from flipping `i`.
///
/// Shared with [`crate::bls`] rather than copied. This and [`flip`] are the incremental update,
/// which is the part most likely to drift between two implementations and the part where a drift
/// is invisible -- a wrong `Δ` does not raise anything, it quietly makes a search worse.
pub(crate) fn gains(g: &Graph, s: &[i8]) -> Vec<f64> {
    (0..g.n)
        .map(|i| {
            let mut field = g.h[i];
            for k in g.offset[i]..g.offset[i + 1] {
                field += g.w[k] * s[g.nbr[k] as usize] as f64;
            }
            2.0 * s[i] as f64 * field
        })
        .collect()
}

/// Flip `i` and repair the affected gains in `O(degree)`.
pub(crate) fn flip(g: &Graph, s: &mut [i8], delta: &mut [f64], i: usize) {
    s[i] = -s[i];
    delta[i] = -delta[i];
    let si = s[i] as f64;
    for k in g.offset[i]..g.offset[i + 1] {
        let j = g.nbr[k] as usize;
        // `i` just changed sign, so j's local field moved by 2*w*s_i(new); its Δ moves by twice
        // that, times its own spin.
        delta[j] += 4.0 * g.w[k] * si * s[j] as f64;
    }
}

#[cfg(test)]
mod tests {

    /// A search handed a state must actually begin there, and must never lose it.
    ///
    /// Both halves matter. Beginning there is what makes composition mean anything; never losing it
    /// is what makes a warm start safe, because the search tracks the best state ever seen and the
    /// handed state is the first one it sees.
    #[test]
    fn a_handed_state_is_where_the_search_starts_and_is_never_lost() {
        let g = crate::ising::lattice2d(8, 1.0);
        let optimum = vec![1i8; g.n];
        let e_opt = g.energy(&optimum);

        let p = Params {
            iterations: 500,
            tenure: 0,
            restart_after: None,
            start: Some(optimum.clone()),
        };
        let r = search(&g, &p, 5);
        assert!(r.energy <= e_opt + 1e-9, "handed the optimum, returned {} vs {e_opt}", r.energy);
        assert_eq!(r.found_at, 0, "the handed state IS the best, found before any flip");

        // And a wrong length is ignored rather than panicking or truncating: the search runs from
        // noise, which is what it did before `start` existed.
        let bad = Params { start: Some(vec![1i8; g.n + 3]), ..p.clone() };
        let r2 = search(&g, &bad, 5);
        assert!(r2.energy.is_finite());
        assert_eq!(r2.state.len(), g.n);
    }

    /// Warm and cold are different runs, so `start` is not decorative.
    #[test]
    fn a_warm_start_is_a_different_run_from_a_cold_one() {
        let mut rng = Pcg::new(3, 1);
        let mut b = crate::graph::GraphBuilder::new(64);
        for i in 0..64usize {
            b.couple(i, (i + 1) % 64, if rng.f64() < 0.5 { 1.0 } else { -1.0 });
            b.couple(i, (i + 7) % 64, if rng.f64() < 0.5 { 1.0 } else { -1.0 });
        }
        let g = b.build();
        let warm: Vec<i8> = (0..g.n).map(|i| if i % 3 == 0 { 1 } else { -1 }).collect();
        let p = Params { iterations: 300, tenure: 4, restart_after: None, start: None };
        let cold = search(&g, &p, 9);
        let hot = search(&g, &Params { start: Some(warm), ..p.clone() }, 9);
        assert!(
            cold.state != hot.state || (cold.energy - hot.energy).abs() > 1e-12,
            "same seed and a different start must be a different run, or `start` does nothing"
        );
    }

    use super::*;
    use crate::graph::GraphBuilder;
    use crate::ising::lattice2d;

    fn random_graph(n: usize, p: f64, seed: u64) -> Graph {
        let mut rng = Pcg::new(seed, 0xC0);
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

    fn brute_min(g: &Graph) -> f64 {
        (0..(1u32 << g.n))
            .map(|m| {
                let s: Vec<i8> = (0..g.n).map(|i| if m >> i & 1 == 1 { 1 } else { -1 }).collect();
                g.energy(&s)
            })
            .fold(f64::INFINITY, f64::min)
    }

    #[test]
    fn the_incremental_gain_matches_a_full_recomputation() {
        // The optimisation the whole algorithm rests on. If `flip` repairs the gain vector wrongly,
        // every subsequent move is chosen on stale numbers -- and the search still runs, still
        // returns a state, and still reports an energy, just a worse one for no visible reason.
        let g = random_graph(12, 0.4, 3);
        let mut rng = Pcg::new(9, 0);
        let mut s: Vec<i8> = (0..g.n).map(|_| rng.spin(0.5)).collect();
        let mut delta = gains(&g, &s);
        for _ in 0..200 {
            let i = (rng.f64() * g.n as f64) as usize % g.n;
            flip(&g, &mut s, &mut delta, i);
            let fresh = gains(&g, &s);
            for k in 0..g.n {
                assert!(
                    (delta[k] - fresh[k]).abs() < 1e-9,
                    "gain {k} drifted: incremental {} vs recomputed {}",
                    delta[k],
                    fresh[k]
                );
            }
        }
    }

    #[test]
    fn a_flip_changes_the_energy_by_exactly_its_gain() {
        let g = random_graph(10, 0.5, 11);
        let mut rng = Pcg::new(4, 0);
        let mut s: Vec<i8> = (0..g.n).map(|_| rng.spin(0.5)).collect();
        let mut delta = gains(&g, &s);
        for _ in 0..100 {
            let i = (rng.f64() * g.n as f64) as usize % g.n;
            let before = g.energy(&s);
            let predicted = delta[i];
            flip(&g, &mut s, &mut delta, i);
            let after = g.energy(&s);
            assert!((after - before - predicted).abs() < 1e-9, "predicted {predicted}, got {}", after - before);
        }
    }

    #[test]
    fn the_reported_energy_belongs_to_the_reported_state() {
        // Accumulating deltas across thousands of flips drifts. The returned number has to be the
        // energy the returned STATE actually has, or the whole result is unfalsifiable.
        for seed in 0..20u64 {
            let g = random_graph(14, 0.35, seed);
            let r = search(
                &g,
                &Params { iterations: 3_000, tenure: 0, restart_after: Some(400), start: None },
                seed,
            );
            assert!((r.energy - g.energy(&r.state)).abs() < 1e-9, "seed {seed}");
        }
    }

    /// True steepest descent: always take the best improving move, stop at a local minimum.
    ///
    /// Written out rather than expressed as `Params`, because the first version of the control
    /// below used `tenure: 1` and depended on an OFF-BY-ONE to mean "no memory at all". When the
    /// off-by-one was fixed, `tenure: 1` became a real one-step tabu search, the control quietly
    /// became a second treatment, and the test that was supposed to show the mechanism working
    /// started reporting that it barely did. A control has to be a different ALGORITHM, not the
    /// same algorithm with a parameter that happens to be inert.
    fn steepest_descent(g: &Graph, seed: u64) -> f64 {
        let mut rng = Pcg::new(seed, 0x7AB0);
        let mut s: Vec<i8> = (0..g.n).map(|_| rng.spin(0.5)).collect();
        let mut delta = gains(g, &s);
        loop {
            let mut pick = usize::MAX;
            let mut best = -1e-12;
            for i in 0..g.n {
                if delta[i] < best {
                    best = delta[i];
                    pick = i;
                }
            }
            if pick == usize::MAX {
                return g.energy(&s); // no improving move: a local minimum
            }
            flip(g, &mut s, &mut delta, pick);
        }
    }

    #[test]
    fn tabu_escapes_the_local_minima_that_steepest_descent_stops_in() {
        // The control is a DIFFERENT algorithm, started from the same seeded state, so the only
        // thing being compared is the ability to keep going after the first local minimum.
        let mut wins = 0;
        let mut losses = 0;
        for seed in 0..30u64 {
            let g = random_graph(16, 0.4, seed + 100);
            let descent = steepest_descent(&g, seed);
            let tabu = search(&g, &Params { iterations: 2_000, tenure: 0, restart_after: None, start: None }, seed);
            if tabu.energy < descent - 1e-9 {
                wins += 1;
            } else if tabu.energy > descent + 1e-9 {
                losses += 1;
            }
        }
        assert!(wins >= 20, "tabu improved on descent in only {wins}/30 instances");
        assert_eq!(losses, 0, "tabu was WORSE than plain descent on {losses}/30, which it cannot be \
                               if it is tracking the best state it ever saw");
    }

    #[test]
    fn a_tenure_larger_than_the_graph_spends_its_whole_budget() {
        // THE DEFECT, as a test. `Params::default()` asks for `max(10, n/10)`, so every graph with
        // n <= 10 had every spin banned at once, nothing was admissible, and the search RETURNED --
        // spending 9 iterations of a 50,000-iteration budget while reporting a result that looked
        // like a completed run. Measured before the fix: n=9 executed 9 iterations and missed the
        // brute-force optimum on 7 of 30 seeds; n=10 executed 49,999 and missed none.
        //
        // The assertion is about the BUDGET, not about solution quality: a tenure of n-1 leaves
        // exactly one admissible move per iteration, which is a legitimately hard search rather
        // than a broken one, and demanding the optimum there would test the wrong thing.
        for n in [3usize, 4, 6, 8, 9, 10, 11, 40] {
            let g = random_graph(n, 0.5, n as u64);
            for tenure in [50usize, 0, 1] {
                let p = Params { iterations: 3_000, tenure, restart_after: None, start: None };
                let r = search(&g, &p, 1);
                assert_eq!(
                    r.iterations_run, p.iterations,
                    "n={n} tenure={tenure}: ran {} of {} iterations",
                    r.iterations_run, p.iterations
                );
            }
        }
    }

    #[test]
    fn small_graphs_still_reach_their_optimum_with_the_default_shape() {
        // The consequence the truncation had, checked directly. These are exactly the sizes that
        // used to exit after n iterations.
        for n in [4usize, 6, 8, 9, 10] {
            let g = random_graph(n, 0.5, n as u64 + 7);
            let truth = brute_min(&g);
            let r = search(&g, &Params { iterations: 20_000, tenure: 0, restart_after: Some(200), start: None }, 3);
            assert!(
                r.energy <= truth + 1e-9,
                "n={n}: got {} against a true minimum of {truth} after {} iterations",
                r.energy, r.iterations_run
            );
        }
    }

    #[test]
    fn it_finds_the_true_optimum_on_instances_small_enough_to_enumerate() {
        for seed in 0..25u64 {
            let g = random_graph(12, 0.45, seed + 500);
            let truth = brute_min(&g);
            let r = search(&g, &Params { iterations: 20_000, tenure: 0, restart_after: Some(500), start: None }, seed);
            assert!(
                r.energy <= truth + 1e-9,
                "seed {seed}: found {} against a true minimum of {truth}",
                r.energy
            );
        }
    }

    #[test]
    fn the_ferromagnet_reaches_its_ground_state() {
        let g = lattice2d(8, 1.0);
        let r = search(&g, &Params { iterations: 20_000, tenure: 0, restart_after: Some(2_000), start: None }, 7);
        // 64 nodes, 2 bonds each on a periodic lattice: -128 with every bond satisfied.
        assert!((r.energy - (-128.0)).abs() < 1e-9, "got {}", r.energy);
    }

    #[test]
    fn the_ledger_charges_a_move_evaluation_per_node_per_iteration() {
        // A tabu iteration evaluates every node and commits one flip. Charging per FLIP would make
        // it look n times cheaper than a Gibbs sweep doing comparable work.
        let g = lattice2d(6, 1.0);
        let mut led = Ledger::default();
        let iters = 100;
        search_metered(&g, &Params { iterations: iters, tenure: 4, restart_after: None, start: None }, 1, Some(&mut led));
        assert_eq!(led.samples, (g.n * iters) as u64);
    }

    #[test]
    fn an_empty_graph_returns_rather_than_panicking() {
        let g = GraphBuilder::new(0).build();
        let r = search(&g, &Params::default(), 1);
        assert!(r.state.is_empty() && r.energy == 0.0);
    }
}
