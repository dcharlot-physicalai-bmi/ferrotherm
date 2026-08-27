//! Breakout local search — the algorithm that holds the max-cut record on most of G-set.
//!
//! [`crate::tabu`] is the baseline a new heuristic has to beat. This is the thing that beat it:
//! Benlic and Hao's BLS improved the best-known cut on **33** of 71 G-set instances and matched it
//! on 35 more, losing on one. A stack that reports a max-cut number without being able to run this
//! is reporting a number from the wrong decade.
//!
//! # It is descent plus a perturbation that thinks
//!
//! The local search is plain steepest descent with **no tabu list at all** — flip the best-improving
//! spin until none improves. The paper is explicit that this is the point rather than an omission:
//! diversification during descent is what tabu search and annealing both do, and BLS argues the
//! compromise between exploring and exploiting matters *only once a local optimum is reached*.
//!
//! What happens at that local optimum is the algorithm. Three move sets, chosen probabilistically:
//!
//! ```text
//!   M1  the single highest-gain spin                    directed, weakest
//!   M2  the highest-gain spin from each side, both      directed
//!   M3  a uniformly random spin                         random, strongest
//! ```
//!
//! The directed sets are filtered by a tabu list — a spin moved at iteration `Iter` is barred until
//! `Iter + γ` — with the usual aspiration: the ban is ignored by a move that would beat the best
//! energy ever seen. So a directed perturbation is *the least damaging move that is not a step
//! backwards*, and a random one is a jump with no regard for quality.
//!
//! # Two dials, both adaptive
//!
//! **How far to jump.** `L` starts at `L0`. If a descent lands on *the same local optimum as last
//! time*, the jump was too short, so `L` grows by one; if it landed anywhere else, `L` resets to
//! `L0`. That is the whole rule, and it is why the search neither cycles between two attractors nor
//! degenerates into random restart.
//!
//! **Which kind of jump.** With `ω` consecutive non-improving descents,
//!
//! ```text
//!   P = max(e^(−ω/T), P0)          directed with probability P, random with 1 − P
//!                                  and, within directed, M1 with Q and M2 with 1 − Q
//! ```
//!
//! so the search starts intensifying and diversifies as it stalls — the paper takes the shape of
//! that schedule from simulated annealing. After `T` non-improving descents `ω` resets and a strong
//! random perturbation fires.
//!
//! # Two things this implementation does not hide
//!
//! **The pseudo-code makes every post-improvement perturbation random.** Algorithm 1 sets `ω ← 0`
//! on an improvement, Algorithm 1 also sets `ω ← 0` on the stagnation reset, and Algorithm 2
//! branches on `ω = 0` to apply the *random* perturbation — whose comment says it is for the
//! stagnation case. The two paths are indistinguishable by the time Algorithm 2 sees them. This
//! follows the pseudo-code rather than the comment, and [`Params::random_after_improvement`] exists
//! so the other reading can be measured instead of argued about.
//!
//! **The best move is found by scanning.** The paper uses Fiduccia–Mattheyses bucket sorting to get
//! the highest-gain vertex in `O(1)`; this scans `Δ` in `O(n)`. At G-set sizes and the iteration
//! counts used here that is a constant factor, not a different algorithm — but it is the reason
//! this cannot be run at the paper's `200000·|V|` budget, and that is a gap rather than a detail.
//!
//! ```
//! use ferrotherm::{bls, ising::lattice2d};
//!
//! let g = lattice2d(8, 1.0);
//! let r = bls::search(&g, &bls::Params::default(), 7);
//! assert!((r.energy - g.energy(&r.state)).abs() < 1e-9);
//! assert!(r.energy <= -2.0 * 64.0 + 1e-9, "a ferromagnet has an all-aligned ground state");
//! ```
//!
//! [Benlic & Hao 2013]: "Breakout Local Search for the Max-Cut problem", Engineering Applications
//! of Artificial Intelligence 26(3), 1162–1173.

use crate::graph::Graph;
use crate::ledger::Ledger;
use crate::rng::Pcg;
use crate::tabu::{flip, gains};

/// How the search is run. Defaults are the paper's Table 1.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Params {
    /// Total moves. One move is one spin flip, in descent or in a perturbation.
    ///
    /// The paper uses `200000·|V|`, which this implementation cannot afford: see the module note on
    /// bucket sorting.
    pub iterations: usize,
    /// Initial jump magnitude `L0`. `None` is the paper's `0.01·|V|`, floored at 1.
    pub l0: Option<usize>,
    /// `T`: non-improving descents tolerated before a strong random perturbation.
    pub t: usize,
    /// `P0`: the floor under the probability of a directed perturbation.
    pub p0: f64,
    /// `Q`: within a directed perturbation, the split between the `M1` and `M2` move sets.
    pub q: f64,
    /// Tabu tenure `γ`, drawn uniformly from this range per move. `None` is the paper's
    /// `rand[3, |V|/10]`.
    pub tenure: Option<(usize, usize)>,
    /// Whether an improving descent is followed by a **random** perturbation.
    ///
    /// `true` follows Algorithm 1 + 2 as written; `false` follows the intent its comments describe,
    /// where the random branch is the stagnation case alone. See the module note — this is a real
    /// ambiguity in the source, and a parameter is the honest way to carry one.
    pub random_after_improvement: bool,
}

impl Default for Params {
    fn default() -> Self {
        Params {
            iterations: 50_000,
            l0: None,
            t: 1_000,
            p0: 0.8,
            q: 0.5,
            tenure: None,
            random_after_improvement: true,
        }
    }
}

/// How the perturbation budget was actually spent.
///
/// Reported because the adaptive schedule is the algorithm: a run that only ever fired random
/// perturbations is iterated local search wearing this module's name, and nothing else in the
/// output would say so.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Perturbations {
    /// `M1`: the single least-damaging non-tabu spin.
    pub directed_one: usize,
    /// `M2`: the least-damaging non-tabu spin from each side, both moved.
    pub directed_two: usize,
    /// `M3`: a uniformly random spin.
    pub random: usize,
}

/// What the search found.
#[derive(Clone, Debug)]
pub struct Outcome {
    /// The best state found.
    pub state: Vec<i8>,
    /// Its energy, recomputed from `state`.
    pub energy: f64,
    /// Local optima visited.
    pub descents: usize,
    /// Moves actually made. Below `iterations` means the search stopped early.
    pub iterations_run: usize,
    /// How the perturbations broke down. See [`Perturbations`].
    pub perturbations: Perturbations,
    /// The largest jump magnitude `L` reached — how hard the search had to work to escape.
    pub max_jump: usize,
    /// Times a descent returned to the immediately previous local optimum, which is what grows `L`.
    pub returns: usize,
}

/// Run breakout local search from a random state.
pub fn search(g: &Graph, p: &Params, seed: u64) -> Outcome {
    search_metered(g, p, seed, None)
}

/// As [`search`], charging every spin update to a [`Ledger`].
pub fn search_metered(g: &Graph, p: &Params, seed: u64, mut ledger: Option<&mut Ledger>) -> Outcome {
    let n = g.n;
    if n == 0 {
        return Outcome {
            state: Vec::new(),
            energy: 0.0,
            descents: 0,
            iterations_run: 0,
            perturbations: Perturbations::default(),
            max_jump: 0,
            returns: 0,
        };
    }
    let mut rng = Pcg::new(seed, 0x000B_155E);
    let mut s: Vec<i8> = (0..n).map(|_| rng.spin(0.5)).collect();
    let mut delta = gains(g, &s);
    let mut energy = g.energy(&s);

    let l0 = p.l0.unwrap_or_else(|| ((n as f64 * 0.01).round() as usize).max(1)).clamp(1, n);
    let (t_lo, t_hi) = p.tenure.unwrap_or((3, (n / 10).max(4)));
    let (t_lo, t_hi) = (t_lo.max(1), t_hi.max(t_lo + 1));
    let t = p.t.max(1);

    let mut best = s.clone();
    let mut best_e = energy;
    let mut prev_optimum = s.clone();

    // `H[i]` is the iteration up to which moving `i` is barred. `usize::MAX` is "never moved" and
    // is compared explicitly rather than by arithmetic, because `0 + γ` is a real deadline and
    // `MAX + γ` is a wrap.
    let mut tabu_until = vec![usize::MAX; n];
    let mut iter = 0usize;
    let (mut omega, mut jump) = (0usize, l0);
    let mut out = Outcome {
        state: Vec::new(),
        energy: 0.0,
        descents: 0,
        iterations_run: 0,
        perturbations: Perturbations::default(),
        max_jump: l0,
        returns: 0,
    };

    while iter < p.iterations {
        // ---- descent: steepest, and deliberately blind to the tabu list -----------------------
        loop {
            if iter >= p.iterations {
                break;
            }
            let mut pick = usize::MAX;
            let mut low = -1e-12; // strictly improving; a zero-gain move is not descent
            for i in 0..n {
                if delta[i] < low {
                    low = delta[i];
                    pick = i;
                }
            }
            if pick == usize::MAX {
                break;
            }
            energy += delta[pick];
            flip(g, &mut s, &mut delta, pick);
            tabu_until[pick] = iter + tenure(&mut rng, t_lo, t_hi);
            iter += 1;
            if let Some(l) = ledger.as_deref_mut() {
                // The same accounting `tabu` uses: one move evaluation per node, because the best
                // move is found by scanning every gain. Charging one sample per FLIP would price a
                // different algorithm -- the one with the bucket structure this does not have.
                l.samples += n as u64;
            }
        }
        out.descents += 1;

        if energy < best_e - 1e-12 {
            best_e = energy;
            best.copy_from_slice(&s);
            omega = 0;
        } else {
            omega += 1;
        }
        if omega > t {
            omega = 0; // stagnated: the next perturbation is the strong, random one
        }

        // The jump grows only when the descent came back to where it just was. That is the signal
        // that the last jump was too short -- not "no improvement", which happens constantly and
        // would make `L` run away.
        if s == prev_optimum {
            jump += 1;
            out.returns += 1;
        } else {
            jump = l0;
        }
        jump = jump.min(n);
        out.max_jump = out.max_jump.max(jump);
        prev_optimum.copy_from_slice(&s);

        if iter >= p.iterations {
            break;
        }

        // ---- perturbation ---------------------------------------------------------------------
        let kind = if omega == 0 && p.random_after_improvement {
            Kind::Random
        } else {
            // At `ω = 0` this is `max(e^0, P0) = 1`, so the other reading of the ambiguity needs no
            // special case: an improvement simply re-enters the schedule at full intensification.
            let prob = (-(omega as f64) / t as f64).exp().max(p.p0);
            Kind::choose(prob, p.q, &mut rng)
        };
        match kind {
            Kind::DirectedOne => out.perturbations.directed_one += 1,
            Kind::DirectedTwo => out.perturbations.directed_two += 1,
            Kind::Random => out.perturbations.random += 1,
        }

        for _ in 0..jump {
            if iter >= p.iterations {
                break;
            }
            let moves = match kind {
                Kind::Random => [Some((rng.next_u32() as usize) % n), None],
                Kind::DirectedOne => [pick_eligible(&delta, &tabu_until, iter, energy, best_e, &s, None), None],
                Kind::DirectedTwo => [
                    pick_eligible(&delta, &tabu_until, iter, energy, best_e, &s, Some(1)),
                    pick_eligible(&delta, &tabu_until, iter, energy, best_e, &s, Some(-1)),
                ],
            };
            for m in moves.into_iter().flatten() {
                // Checked INSIDE the loop, not only before it. `M2` applies two moves, so a budget
                // check that only guards the pair lets the second one through after the first has
                // already reached the ceiling -- an overshoot of exactly one flip, which is enough
                // to make `iterations_run` disagree with the budget it was given. Found by the HTTP
                // test rather than the Rust one, because the Rust test used an instance where the
                // pair never straddled the boundary.
                if iter >= p.iterations {
                    break;
                }
                energy += delta[m];
                flip(g, &mut s, &mut delta, m);
                tabu_until[m] = iter + tenure(&mut rng, t_lo, t_hi);
                iter += 1;
                if let Some(l) = ledger.as_deref_mut() {
                    l.samples += n as u64;
                }
                // A perturbation can stumble onto a new best. Recording it here rather than only
                // after the next descent costs one comparison and means a good state is never
                // walked out of and lost.
                if energy < best_e - 1e-12 {
                    best_e = energy;
                    best.copy_from_slice(&s);
                }
            }
        }
    }

    out.iterations_run = iter;
    // Recomputed from the state rather than carried: `energy` is an accumulator, and an accumulator
    // is exactly the thing that can drift away from the state it claims to describe.
    out.energy = g.energy(&best);
    out.state = best;
    out
}

/// A tabu tenure drawn uniformly from `[lo, hi]`.
///
/// The range is the paper's `rand[3, |V|/10]`, drawn per move rather than fixed: a constant tenure
/// lets a search settle into a cycle whose period matches it, which is the failure a randomised
/// tenure exists to break.
fn tenure(rng: &mut Pcg, lo: usize, hi: usize) -> usize {
    lo + (rng.next_u32() as usize) % (hi - lo + 1)
}

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    DirectedOne,
    DirectedTwo,
    Random,
}

impl Kind {
    fn choose(p: f64, q: f64, rng: &mut Pcg) -> Kind {
        if rng.f64() < p {
            if rng.f64() < q {
                Kind::DirectedOne
            } else {
                Kind::DirectedTwo
            }
        } else {
            Kind::Random
        }
    }
}

/// The least-damaging eligible move, or `None` if every spin is barred.
///
/// Eligible means not tabu **or** aspiring — a move whose result would beat the best energy ever
/// seen overrides the ban, because refusing a new global best on bookkeeping is the one case where
/// the tabu list is certainly wrong. `side` restricts the search to spins of one sign, which is what
/// makes `M2` a move that takes one from each partition rather than two from the same one.
fn pick_eligible(
    delta: &[f64],
    tabu_until: &[usize],
    iter: usize,
    energy: f64,
    best: f64,
    s: &[i8],
    side: Option<i8>,
) -> Option<usize> {
    let mut pick = usize::MAX;
    let mut low = f64::INFINITY;
    for i in 0..delta.len() {
        if let Some(v) = side {
            if s[i] != v {
                continue;
            }
        }
        let free = tabu_until[i] == usize::MAX || iter > tabu_until[i];
        let aspires = energy + delta[i] < best - 1e-12;
        if (free || aspires) && delta[i] < low {
            low = delta[i];
            pick = i;
        }
    }
    (pick != usize::MAX).then_some(pick)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::GraphBuilder;
    use crate::ising::lattice2d;

    fn frustrated(n: usize, p: f64, seed: u64) -> Graph {
        let mut rng = Pcg::new(seed, 0x000B_15C0);
        let mut gb = GraphBuilder::new(n);
        for i in 0..n {
            for j in (i + 1)..n {
                if rng.f64() < p {
                    gb.couple(i, j, rng.f64() * 2.0 - 1.0);
                }
            }
        }
        gb.build()
    }

    /// Steepest descent from the same seeded start. The control is a DIFFERENT algorithm, so the
    /// only thing being compared is the ability to keep going past the first local minimum — which
    /// is the entire claim BLS makes.
    fn steepest_descent(g: &Graph, seed: u64) -> f64 {
        let mut rng = Pcg::new(seed, 0x000B_155E);
        let mut s: Vec<i8> = (0..g.n).map(|_| rng.spin(0.5)).collect();
        let mut delta = gains(g, &s);
        loop {
            let mut pick = usize::MAX;
            let mut low = -1e-12;
            for i in 0..g.n {
                if delta[i] < low {
                    low = delta[i];
                    pick = i;
                }
            }
            if pick == usize::MAX {
                return g.energy(&s);
            }
            flip(g, &mut s, &mut delta, pick);
        }
    }

    #[test]
    fn the_energy_returned_is_the_energy_of_the_state_returned() {
        for seed in 0..8u64 {
            let g = frustrated(40, 0.3, seed);
            let r = search(&g, &Params { iterations: 4_000, ..Params::default() }, seed);
            assert_eq!(r.state.len(), g.n);
            assert!(r.state.iter().all(|&v| v == 1 || v == -1));
            assert!(
                (r.energy - g.energy(&r.state)).abs() < 1e-9,
                "seed {seed}: reported {}, the state gives {}",
                r.energy,
                g.energy(&r.state)
            );
        }
    }

    #[test]
    fn it_escapes_the_local_minima_that_steepest_descent_stops_in() {
        let (mut wins, mut losses) = (0, 0);
        for seed in 0..30u64 {
            let g = frustrated(30, 0.35, seed + 500);
            let d = steepest_descent(&g, seed);
            let b = search(&g, &Params { iterations: 3_000, ..Params::default() }, seed);
            if b.energy < d - 1e-9 {
                wins += 1;
            } else if b.energy > d + 1e-9 {
                losses += 1;
            }
        }
        assert_eq!(losses, 0, "BLS ended above a plain descent on {losses} of 30 instances");
        assert!(wins >= 20, "only beat descent on {wins} of 30; the perturbation is not working");
    }

    /// The adaptive schedule is the algorithm. A run that only ever fired one kind of perturbation
    /// is a different, simpler method wearing this module's name, and nothing else in the output
    /// would say so.
    #[test]
    fn the_perturbation_mix_is_genuinely_mixed() {
        let g = frustrated(60, 0.25, 3);
        let r = search(&g, &Params { iterations: 20_000, ..Params::default() }, 3);
        let p = r.perturbations;
        assert!(p.random > 0, "no random perturbation fired");
        assert!(p.directed_one > 0, "no M1 perturbation fired");
        assert!(p.directed_two > 0, "no M2 perturbation fired");
        assert!(r.descents > 10, "only {} descents in 20k moves", r.descents);
    }

    /// `L` grows only when a descent returns to the immediately previous local optimum. If that
    /// never happens the jump stays at `L0` — and if the counter never moved, the rule is
    /// decoration.
    #[test]
    fn the_jump_grows_exactly_when_the_search_comes_back() {
        let g = frustrated(50, 0.3, 11);
        let r = search(&g, &Params { iterations: 30_000, ..Params::default() }, 11);
        let l0 = ((50.0f64 * 0.01).round() as usize).max(1);
        assert!(r.max_jump >= l0);
        assert_eq!(
            r.max_jump > l0,
            r.returns > 0,
            "the jump grew {} times against {} returns to the previous optimum",
            r.max_jump - l0,
            r.returns
        );
    }

    /// The ambiguity dial is real: reading the pseudo-code and reading its comments give measurably
    /// different searches. Carried as a parameter precisely so this can be checked rather than
    /// argued about.
    #[test]
    fn the_two_readings_of_the_pseudo_code_are_different_searches() {
        let g = frustrated(60, 0.25, 21);
        let faithful = search(&g, &Params { iterations: 20_000, ..Params::default() }, 21);
        let intended = search(
            &g,
            &Params { iterations: 20_000, random_after_improvement: false, ..Params::default() },
            21,
        );
        assert_ne!(
            faithful.perturbations, intended.perturbations,
            "both readings produced the same perturbation mix, so the parameter does nothing"
        );
        // Both must still be sound searches, whatever the mix.
        for (name, r) in [("faithful", &faithful), ("intended", &intended)] {
            assert!((r.energy - g.energy(&r.state)).abs() < 1e-9, "{name}");
        }
    }

    /// Truncation has to be visible: a run that spent a fraction of its budget returns a result
    /// shaped exactly like a full one.
    /// The budget is a CEILING, and it is a ceiling on every instance rather than on the one that
    /// happened to be tested.
    ///
    /// The first version of this checked a single ferromagnet at a single budget and passed while
    /// the search could overshoot by one flip: the `M2` perturbation applies two moves, and the
    /// budget was only checked before the pair. It was the HTTP test that failed, on a different
    /// instance, at a different budget. So this sweeps both.
    #[test]
    fn the_budget_is_a_ceiling_on_every_instance_not_just_the_convenient_one() {
        for budget in [999, 5_000, 20_001] {
            for seed in 0..6u64 {
                let g = frustrated(24, 0.35, seed);
                let r = search(&g, &Params { iterations: budget, ..Params::default() }, seed);
                assert_eq!(
                    r.iterations_run, budget,
                    "budget {budget}, seed {seed}: ran {} flips",
                    r.iterations_run
                );
            }
        }
        // And a ferromagnet is solved on the way: all-aligned, one per bond, both directions.
        let g = lattice2d(6, 1.0);
        let r = search(&g, &Params { iterations: 5_000, ..Params::default() }, 2);
        assert!((r.energy + 2.0 * 36.0).abs() < 1e-9, "energy {}", r.energy);
    }

    #[test]
    fn an_empty_graph_returns_rather_than_panicking() {
        let g = GraphBuilder::new(0).build();
        let r = search(&g, &Params::default(), 1);
        assert!(r.state.is_empty() && r.energy == 0.0 && r.iterations_run == 0);
    }

    #[test]
    fn the_ledger_charges_a_move_evaluation_per_node_per_move() {
        let g = frustrated(20, 0.4, 5);
        let mut led = Ledger::default();
        let r = search_metered(&g, &Params { iterations: 1_000, ..Params::default() }, 5, Some(&mut led));
        assert_eq!(led.samples, (r.iterations_run * g.n) as u64);
    }
}
