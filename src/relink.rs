//! Path relinking and scatter search over an elite reference set.
//!
//! Every other search in this crate walks from one state. This one walks *between two*: keep a pool
//! of good, mutually distant solutions, take a pair, and move from one to the other by flipping only
//! the spins where they disagree, cheapest first. Both endpoints are local optima, so every state on
//! the way is one that neither hill-climb could reach, and the best of them is frequently below
//! both. That is the idea — Glover's scatter search, Ribeiro and Resende's path relinking — and it is
//! the recombination operator for a space where crossover has no meaning.
//!
//! # The path is exact
//!
//! A walk of `k` steps is the start with exactly `k` of the differing spins flipped and nothing
//! else, so [`replay`] rebuilds any state on it from [`Relink::flips`] alone. An implementation that
//! drifted off the path would still return plausible states and better energies; the invariant is
//! what makes "the best state between these two" mean anything, so it is tested rather than assumed.
//!
//! # The gauge
//!
//! A model with no biases is symmetric under flipping every spin at once, so `s` and `-s` are one
//! solution in two signs. Independent descents land on opposite gauges about half the time, and
//! relinking those spends most of the walk undoing a symmetry. [`path_gauged`] walks toward
//! whichever of the target and its global flip is nearer, and only where the two provably have the
//! same energy ([`field_free`]).
//!
//! ```
//! use ferrotherm::{planted, relink};
//!
//! let p = planted::frustrated_loops(8, 96, 3);
//! let o = relink::search(&p.graph, &relink::Params::default(), 7);
//! assert!((o.energy - p.graph.energy(&o.state)).abs() < 1e-9);
//! assert!(p.solved(&o.state), "reached {} of {}", o.energy, p.ground_energy);
//! ```

use crate::graph::Graph;
use crate::portfolio::{Budget, Found, Search};
use crate::rng::Pcg;
use crate::tabu::{flip, gains};

/// Spins at which two states disagree.
///
/// # Panics
///
/// If the two states have different lengths.
#[must_use]
pub fn hamming(a: &[i8], b: &[i8]) -> usize {
    assert_eq!(a.len(), b.len(), "states of different lengths have no distance");
    a.iter().zip(b).filter(|(x, y)| x != y).count()
}

/// Distance up to the global spin flip: `min(d, n - d)`.
///
/// # Panics
///
/// If the two states have different lengths.
#[must_use]
pub fn gauge_distance(a: &[i8], b: &[i8]) -> usize {
    let d = hamming(a, b);
    d.min(a.len() - d)
}

/// Whether every bias is zero, which is exactly when `s` and `-s` have the same energy.
#[must_use]
pub fn field_free(g: &Graph) -> bool {
    g.h.iter().all(|&h| h == 0.0)
}

/// The state `k` flips along a path, rebuilt from the flip order alone.
///
/// # Panics
///
/// If `k` exceeds the number of flips, or a flip index is outside `from`.
#[must_use]
pub fn replay(from: &[i8], flips: &[usize], k: usize) -> Vec<i8> {
    assert!(k <= flips.len(), "a path of {} flips has no step {k}", flips.len());
    let mut s = from.to_vec();
    for &i in &flips[..k] {
        s[i] = -s[i];
    }
    s
}

/// One state on a relinked path.
#[derive(Clone, Debug, PartialEq)]
pub struct Step {
    /// The state itself.
    pub state: Vec<i8>,
    /// Its energy, recomputed from `state` rather than accumulated along the walk.
    pub energy: f64,
    /// How many flips into the path it sits, so `replay(from, &flips, at)` rebuilds it.
    pub at: usize,
}

/// What one walk between two states produced.
#[derive(Clone, Debug, PartialEq)]
pub struct Relink {
    /// The differing spins in the order the walk took them; its length is the distance walked.
    pub flips: Vec<usize>,
    /// The best *strict* intermediate, or `None` when the endpoints are equal or adjacent.
    ///
    /// Strict on purpose: a path whose best state is an endpoint has recombined nothing, and
    /// returning the endpoint would make [`Relink::improved`] true of every pair.
    pub best: Option<Step>,
    /// Energy of the starting endpoint.
    pub from_energy: f64,
    /// Energy of the endpoint actually reached, which is the gauge-flipped target when `gauged`.
    pub to_energy: f64,
    /// Whether the target was replaced by its global spin flip before walking.
    pub gauged: bool,
    /// Single-spin proposals the walk evaluated: `d(d+1)/2` over a distance of `d`.
    pub proposals: u64,
}

impl Relink {
    /// Whether the best intermediate is strictly below both endpoints.
    #[must_use]
    pub fn improved(&self) -> bool {
        self.best.as_ref().is_some_and(|b| b.energy < self.from_energy.min(self.to_energy) - 1e-12)
    }

    /// How far the best intermediate falls below the better endpoint; `0.0` when there is none.
    ///
    /// Negative when the path improved on nothing, which is the ordinary case and worth seeing.
    #[must_use]
    pub fn gain(&self) -> f64 {
        match &self.best {
            Some(b) => self.from_energy.min(self.to_energy) - b.energy,
            None => 0.0,
        }
    }
}

/// Walk from `from` to `to`, flipping the differing spins cheapest-first.
///
/// # Panics
///
/// If either state's length differs from the graph's node count.
#[must_use]
pub fn path(g: &Graph, from: &[i8], to: &[i8]) -> Relink {
    walk(g, from, to, false)
}

/// The same, walking toward whichever of `to` or `-to` is nearer on a [`field_free`] model.
///
/// On a model with biases this is [`path`]: `-to` is then a different solution with a different
/// energy, and silently substituting it would relink to a state the caller never named.
///
/// # Panics
///
/// If either state's length differs from the graph's node count.
#[must_use]
pub fn path_gauged(g: &Graph, from: &[i8], to: &[i8]) -> Relink {
    if field_free(g) && 2 * hamming(from, to) > g.n {
        let neg: Vec<i8> = to.iter().map(|&s| -s).collect();
        return walk(g, from, &neg, true);
    }
    walk(g, from, to, false)
}

fn walk(g: &Graph, from: &[i8], to: &[i8], gauged: bool) -> Relink {
    assert!(
        from.len() == g.n && to.len() == g.n,
        "states of {} and {} spins on a {}-spin graph",
        from.len(),
        to.len(),
        g.n
    );
    let mut remaining: Vec<usize> = (0..g.n).filter(|&i| from[i] != to[i]).collect();
    let mut s = from.to_vec();
    let mut delta = gains(g, &s);
    let from_energy = g.energy(&s);
    // Accumulated from the incremental gains, exactly as `tabu` does, and used only to CHOOSE the
    // best intermediate; the energy reported for it is recomputed from the state below.
    let mut e = from_energy;
    let mut flips = Vec::with_capacity(remaining.len());
    let mut best: Option<Step> = None;
    let mut proposals = 0u64;

    while !remaining.is_empty() {
        proposals += remaining.len() as u64;
        // `remove` rather than `swap_remove`: `remaining` stays ascending, so a tie between two
        // equally cheap flips goes to the lower index and the walk is reproducible.
        let mut pick = 0usize;
        for k in 1..remaining.len() {
            if delta[remaining[k]] < delta[remaining[pick]] {
                pick = k;
            }
        }
        let i = remaining.remove(pick);
        e += delta[i];
        flip(g, &mut s, &mut delta, i);
        flips.push(i);
        // Strict intermediates only, and `<` keeps the earlier of two equal states.
        if !remaining.is_empty() && best.as_ref().is_none_or(|b| e < b.energy) {
            best = Some(Step { state: s.clone(), energy: e, at: flips.len() });
        }
    }

    let to_energy = g.energy(&s);
    if let Some(b) = best.as_mut() {
        b.energy = g.energy(&b.state);
    }
    Relink { flips, best, from_energy, to_energy, gauged, proposals }
}

/// A solution held in the reference set.
#[derive(Clone, Debug, PartialEq)]
pub struct Member {
    /// The state.
    pub state: Vec<i8>,
    /// Its energy, as the set was told; [`RefSet`] never recomputes it.
    pub energy: f64,
}

/// What the reference set did with an offered solution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Admit {
    /// It entered the set, growing it or displacing the worst member.
    Added,
    /// It displaced a member it was too close to, being strictly better than that member.
    Replaced,
    /// A member sat within the diversity floor and it was no better than that member.
    TooClose,
    /// The set was full and it was no better than the worst member.
    TooWeak,
}

/// An elite pool: the best solutions found, kept mutually distant.
///
/// Both halves are load-bearing. Quality alone collapses the pool into one basin, and relinking two
/// states that differ in three spins recombines nothing; distance alone fills it with noise.
#[derive(Clone, Debug)]
pub struct RefSet {
    max: usize,
    min_distance: usize,
    gauge: bool,
    members: Vec<Member>,
}

impl RefSet {
    /// A set of at most `max`, refusing anything within `min_distance` of a member it cannot beat.
    ///
    /// A floor of zero still refuses exact duplicates: a pool holding one state twice is a smaller
    /// pool wearing a larger number. `gauge` measures distance up to the global spin flip, which is
    /// what a model with no biases needs.
    ///
    /// # Panics
    ///
    /// If `max` is zero.
    #[must_use]
    pub fn new(max: usize, min_distance: usize, gauge: bool) -> RefSet {
        assert!(max > 0, "a reference set of zero members is not a pool");
        RefSet { max, min_distance, gauge, members: Vec::new() }
    }

    /// Members, best energy first.
    #[must_use]
    pub fn members(&self) -> &[Member] {
        &self.members
    }

    /// How many members it holds.
    #[must_use]
    pub fn len(&self) -> usize {
        self.members.len()
    }

    /// Whether it holds nothing yet.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.members.is_empty()
    }

    /// The best member, or `None` while empty.
    #[must_use]
    pub fn best(&self) -> Option<&Member> {
        self.members.first()
    }

    /// The distance this set measures with: gauge-aware when it was built that way.
    ///
    /// # Panics
    ///
    /// If the two states have different lengths.
    #[must_use]
    pub fn distance(&self, a: &[i8], b: &[i8]) -> usize {
        if self.gauge { gauge_distance(a, b) } else { hamming(a, b) }
    }

    /// Offer a solution, and say what became of it.
    ///
    /// # Panics
    ///
    /// If `s` has a different length from the members already held.
    pub fn insert(&mut self, s: &[i8], energy: f64) -> Admit {
        let near = self
            .members
            .iter()
            .enumerate()
            .map(|(k, m)| (k, self.distance(&m.state, s)))
            .min_by_key(|&(_, d)| d);
        if let Some((k, d)) = near
            && d < self.min_distance.max(1)
        {
            if energy < self.members[k].energy - 1e-12 {
                self.members[k] = Member { state: s.to_vec(), energy };
                self.sort();
                return Admit::Replaced;
            }
            return Admit::TooClose;
        }
        if self.members.len() < self.max {
            self.members.push(Member { state: s.to_vec(), energy });
            self.sort();
            return Admit::Added;
        }
        let worst = self.members.len() - 1;
        if energy < self.members[worst].energy - 1e-12 {
            self.members[worst] = Member { state: s.to_vec(), energy };
            self.sort();
            return Admit::Added;
        }
        Admit::TooWeak
    }

    fn sort(&mut self) {
        self.members.sort_by(|a, b| a.energy.total_cmp(&b.energy));
    }
}

/// How a scatter search is run.
#[derive(Clone, Debug, PartialEq)]
pub struct Params {
    /// Reference set size.
    pub pool: usize,
    /// Relinked pairs to attempt. One round is one walk.
    pub rounds: usize,
    /// Diversity floor in spins. `None` is `max(1, n/10)`.
    pub min_distance: Option<usize>,
    /// Treat a state and its global flip as one solution, where the model allows it.
    pub gauge: bool,
    /// Descend from the best intermediate before offering it to the pool.
    ///
    /// The difference between a recombination operator and a complete method: the best state on a
    /// path is rarely itself a local optimum.
    pub improve: bool,
    /// Stop once this many single-spin proposals have been spent. `None` runs `rounds` to the end.
    pub max_proposals: Option<u64>,
}

impl Default for Params {
    fn default() -> Self {
        Params {
            pool: 10,
            rounds: 200,
            min_distance: None,
            gauge: true,
            improve: true,
            max_proposals: None,
        }
    }
}

/// What a scatter search found.
#[derive(Clone, Debug, PartialEq)]
pub struct Outcome {
    /// The best state seen, whether it came from the pool or from a path.
    pub state: Vec<i8>,
    /// Its energy, recomputed from `state`.
    pub energy: f64,
    /// Walks performed.
    pub relinks: usize,
    /// Walks whose best intermediate beat both of their endpoints.
    ///
    /// The number that says whether recombination did anything: a run reporting zero is a
    /// multi-start descent that spent its budget walking between local optima for nothing.
    pub improving: usize,
    /// The reference set as it ended.
    pub pool: Vec<Member>,
    /// Single-spin proposals spent.
    pub proposals: u64,
}

/// Seed a pool of local optima, then relink pairs from it and feed the results back.
///
/// # Panics
///
/// If `p.pool` is zero.
#[must_use]
pub fn search(g: &Graph, p: &Params, seed: u64) -> Outcome {
    assert!(p.pool > 0, "a scatter search needs a reference set");
    let n = g.n;
    if n == 0 {
        return Outcome {
            state: Vec::new(),
            energy: 0.0,
            relinks: 0,
            improving: 0,
            pool: Vec::new(),
            proposals: 0,
        };
    }
    let cap = p.max_proposals.unwrap_or(u64::MAX);
    let floor = p.min_distance.unwrap_or_else(|| (n / 10).max(1));
    let gauge = p.gauge && field_free(g);
    let mut set = RefSet::new(p.pool, floor, gauge);
    let mut rng = Pcg::new(seed, 0x5CA7);
    let mut spent = 0u64;

    // Seed the pool with descents from noise. Attempts are bounded: with a diversity floor a run of
    // starts can land in one basin, and an unbounded loop would then never fill and never end.
    let mut attempts = 0usize;
    while set.len() < p.pool && attempts < 4 * p.pool + 8 && spent < cap {
        attempts += 1;
        let mut s: Vec<i8> = (0..n).map(|_| rng.spin(0.5)).collect();
        descend(g, &mut s, &mut spent, cap);
        let e = g.energy(&s);
        set.insert(&s, e);
    }
    let up = vec![1i8; n];
    let mut best =
        set.best().cloned().unwrap_or(Member { energy: g.energy(&up), state: up });

    let mut relinks = 0usize;
    let mut improving = 0usize;
    for round in 0..p.rounds {
        let m = set.len();
        if m < 2 || spent >= cap {
            break;
        }
        // Systematic over ORDERED pairs, so both directions of a pair get walked: the greedy order
        // from a to b is not the reverse of the order from b to a, and the two paths differ.
        let k = round % (m * (m - 1));
        let i = k / (m - 1);
        let mut j = k % (m - 1);
        if j >= i {
            j += 1;
        }
        let (a, b) = (set.members()[i].state.clone(), set.members()[j].state.clone());
        // A walk cut in half is not a path, so one is only started when its whole cost fits.
        let d = if gauge { gauge_distance(&a, &b) } else { hamming(&a, &b) } as u64;
        if spent + d * (d + 1) / 2 > cap {
            break;
        }
        let r = if p.gauge { path_gauged(g, &a, &b) } else { path(g, &a, &b) };
        spent += r.proposals;
        relinks += 1;
        if r.improved() {
            improving += 1;
        }
        if let Some(step) = r.best {
            let mut s = step.state;
            if p.improve {
                descend(g, &mut s, &mut spent, cap);
            }
            let e = g.energy(&s);
            if e < best.energy {
                best = Member { state: s.clone(), energy: e };
            }
            set.insert(&s, e);
        }
    }

    Outcome {
        energy: g.energy(&best.state),
        state: best.state,
        relinks,
        improving,
        pool: set.members().to_vec(),
        proposals: spent,
    }
}

/// Steepest descent to a local optimum, charging `n` proposals per scan and stopping at `cap`.
fn descend(g: &Graph, s: &mut [i8], spent: &mut u64, cap: u64) {
    let n = g.n as u64;
    *spent = spent.saturating_add(n);
    let mut delta = gains(g, s);
    while spent.saturating_add(n) <= cap {
        *spent += n;
        let mut pick = usize::MAX;
        let mut low = -1e-12;
        for i in 0..g.n {
            if delta[i] < low {
                low = delta[i];
                pick = i;
            }
        }
        if pick == usize::MAX {
            break;
        }
        flip(g, s, &mut delta, pick);
    }
}

/// Scatter search with path relinking, as a [`crate::portfolio`] arm.
///
/// The portfolio carries no population method at all; every other arm walks from a single state.
#[derive(Clone, Copy, Debug, Default)]
pub struct Scatter;

impl Search for Scatter {
    fn name(&self) -> &'static str {
        "scatter"
    }
    fn solve(&self, g: &Graph, budget: Budget, seed: u64) -> Found {
        // Rounds are capped by the budget rather than by a constant: a walk costs at least one
        // proposal, so the round count can never end a run before the budget does.
        let p = Params {
            rounds: usize::try_from(budget.proposals).unwrap_or(usize::MAX),
            max_proposals: Some(budget.proposals),
            ..Default::default()
        };
        let o = search(g, &p, seed);
        Found { arm: "scatter", energy: g.energy(&o.state), spent: o.proposals, state: o.state }
    }
}


#[cfg(test)]
mod tests {
    // There is deliberately no test here comparing this search against `tabu`, `bls` or the
    // portfolio. Two heuristics on one instance family differ by their seeds as much as by their
    // methods, and a "scatter beat tabu" assertion would be a measurement of the budget conversion
    // wearing the clothes of a verification. Everything below is checked against something exact:
    // an algebraic identity on the path, exhaustive enumeration of the interval a path walks, the
    // Z2 symmetry of a field-free model, deterministic bookkeeping, or a planted optimum.
    use super::*;
    use crate::graph::GraphBuilder;
    use crate::planted;

    /// The four planted instances every statistical claim in this module is made over.
    const CASES: [(usize, usize, u64); 4] = [(8, 96, 3), (10, 200, 5), (6, 48, 9), (12, 400, 1)];

    /// A local optimum of `g` from noise, by the same descent the search uses.
    fn optimum(g: &Graph, seed: u64) -> Vec<i8> {
        let mut rng = Pcg::new(seed, 0xD3);
        let mut s: Vec<i8> = (0..g.n).map(|_| rng.spin(0.5)).collect();
        let mut spent = 0u64;
        descend(g, &mut s, &mut spent, u64::MAX);
        s
    }

    /// Every state on a path is the start with exactly the flips taken so far, and nothing else.
    ///
    /// The invariant the whole module rests on, and the one a plausible-looking implementation
    /// breaks silently: a walk that wandered off the path would still return legal states with
    /// legal energies, and "the best state between these two" would quietly stop meaning that.
    /// Checked as an identity -- `replay` rebuilds each state from the flip order alone, and the
    /// distance to each endpoint is forced at every step.
    #[test]
    fn a_path_is_its_start_plus_exactly_the_flips_taken() {
        let p = planted::frustrated_loops(8, 96, 3);
        let g = &p.graph;
        for seed in 0..12u64 {
            let a = optimum(g, seed);
            let b = optimum(g, seed + 100);
            let d = hamming(&a, &b);
            let r = path(g, &a, &b);

            assert_eq!(r.flips.len(), d, "one step per differing spin");
            let mut once = r.flips.clone();
            once.sort_unstable();
            once.dedup();
            assert_eq!(once.len(), d, "a spin was flipped twice");
            for &i in &r.flips {
                assert_ne!(a[i], b[i], "spin {i} was flipped, but the endpoints agree on it");
            }
            for k in 0..=d {
                let s = replay(&a, &r.flips, k);
                assert_eq!(hamming(&a, &s), k, "step {k} is not {k} flips from the start");
                assert_eq!(hamming(&s, &b), d - k, "step {k} is not {} from the target", d - k);
            }
            assert_eq!(replay(&a, &r.flips, d), b, "the walk must end at the target");
            assert_eq!(r.proposals, d as u64 * (d as u64 + 1) / 2, "a step scans what is left");

            if let Some(step) = &r.best {
                assert!(step.at > 0 && step.at < d, "an endpoint is not an intermediate");
                assert_eq!(step.state, replay(&a, &r.flips, step.at), "best state is off the path");
                assert!(
                    (step.energy - g.energy(&step.state)).abs() < 1e-9,
                    "reported {} for a state worth {}",
                    step.energy,
                    g.energy(&step.state)
                );
            }
            assert!((r.from_energy - g.energy(&a)).abs() < 1e-9);
            assert!((r.to_energy - g.energy(&b)).abs() < 1e-9);
            assert!(!r.gauged, "`path` never substitutes the target");
        }
    }

    /// At every step the walk takes the cheapest remaining flip, ties to the lowest index.
    ///
    /// The oracle is the graph itself: each step's gains are recomputed from `field` rather than
    /// read out of the walk's own incremental update, so a drifting update is caught rather than
    /// confirmed. On these instances every coupling is +-1 and every gain is an exact integer, so
    /// the tie-break can be asserted rather than approximated.
    #[test]
    fn the_walk_takes_the_cheapest_remaining_flip_at_every_step() {
        let p = planted::frustrated_loops(6, 40, 11);
        let g = &p.graph;
        for seed in 0..8u64 {
            let a = optimum(g, seed);
            let b = optimum(g, seed + 50);
            let r = path(g, &a, &b);
            let mut left: Vec<usize> = (0..g.n).filter(|&i| a[i] != b[i]).collect();
            for (k, &taken) in r.flips.iter().enumerate() {
                let s = replay(&a, &r.flips, k);
                let d_of = |i: usize| crate::kernel::delta_e(g.field(i, &s), s[i]);
                let low = left.iter().map(|&i| d_of(i)).fold(f64::INFINITY, f64::min);
                assert!(
                    (d_of(taken) - low).abs() < 1e-9,
                    "step {k} took {taken} at {} when {low} was available",
                    d_of(taken)
                );
                let first = left
                    .iter()
                    .copied()
                    .find(|&i| (d_of(i) - low).abs() < 1e-9)
                    .expect("the minimum is attained");
                assert_eq!(taken, first, "ties must go to the lowest index at step {k}");
                left.retain(|&i| i != taken);
            }
            assert!(left.is_empty(), "the walk left differing spins unflipped");
        }
    }

    /// A path of length two has exactly one intermediate, and it is the better of the two available.
    ///
    /// An exact oracle with no statistics in it: both candidate states are written down and their
    /// energies compared directly.
    #[test]
    fn a_two_step_path_lands_on_the_better_of_its_two_intermediates() {
        let p = planted::frustrated_loops(5, 24, 5);
        let g = &p.graph;
        let base = optimum(g, 3);
        for i in 0..g.n {
            for j in (i + 1)..g.n {
                let mut b = base.clone();
                b[i] = -b[i];
                b[j] = -b[j];
                let r = path(g, &base, &b);
                assert_eq!(r.flips.len(), 2);
                let step = r.best.as_ref().expect("a two-step path has one intermediate");
                assert_eq!(step.at, 1);
                let want = g.energy(&replay(&base, &[i], 1)).min(g.energy(&replay(&base, &[j], 1)));
                assert!(
                    (step.energy - want).abs() < 1e-9,
                    "flipping {i} then {j}: landed on {} when {want} was there",
                    step.energy
                );
            }
        }
    }

    /// The greedy path never beats the best state in the interval it walks, and usually matches it.
    ///
    /// The interval -- every state agreeing with both endpoints wherever they agree -- is enumerated
    /// exhaustively, so this is the exact answer to the question the walk is a heuristic for. A
    /// state below that minimum would be a state off the path.
    #[test]
    fn the_best_intermediate_never_beats_the_interval_it_walks() {
        let p = planted::frustrated_loops(4, 12, 2);
        let g = &p.graph;
        let (mut cases, mut matched) = (0, 0);
        for seed in 0..40u64 {
            let a = optimum(g, seed);
            let b = optimum(g, seed + 77);
            let diff: Vec<usize> = (0..g.n).filter(|&i| a[i] != b[i]).collect();
            if !(3..=14).contains(&diff.len()) {
                continue;
            }
            let r = path(g, &a, &b);
            // Every strict intermediate, endpoints excluded -- the same set the walk may report.
            let mut lo = f64::INFINITY;
            for mask in 1..(1u32 << diff.len()) - 1 {
                let mut s = a.clone();
                for (k, &i) in diff.iter().enumerate() {
                    if mask >> k & 1 == 1 {
                        s[i] = -s[i];
                    }
                }
                lo = lo.min(g.energy(&s));
            }
            let step = r.best.as_ref().expect("three or more steps leaves an intermediate");
            assert!(
                step.energy >= lo - 1e-9,
                "the walk reported {}, below the interval minimum {lo}",
                step.energy
            );
            cases += 1;
            matched += usize::from((step.energy - lo).abs() < 1e-9);
        }
        assert!(cases >= 20, "only {cases} usable pairs, too few to conclude anything");
        assert!(2 * matched >= cases, "greedy matched the interval optimum on {matched} of {cases}");
    }

    /// Relinking two local optima beats both endpoints often enough to matter.
    ///
    /// The claim the module exists for. Measured over ordered pairs of independent descents on four
    /// planted instances, and stated three ways: the rate, the fact that some intermediate is
    /// strictly below EVERY endpoint, and -- on at least one instance -- that a path reaches the
    /// planted optimum which no descent reached. The first two thresholds sit well under the
    /// measured 47-91% and are per instance; the aggregate bar is the measured 70%.
    #[test]
    fn relinking_two_local_optima_beats_both_endpoints_often_enough_to_matter() {
        let (mut all_pairs, mut all_better, mut solved_by_a_path) = (0usize, 0usize, 0usize);
        for (l, loops, iseed) in CASES {
            let p = planted::frustrated_loops(l, loops, iseed);
            let g = &p.graph;
            let optima: Vec<Vec<i8>> = (0..12).map(|s| optimum(g, s)).collect();
            let floor = optima.iter().map(|s| g.energy(s)).fold(f64::INFINITY, f64::min);
            assert!(
                floor > p.ground_energy + 1e-9,
                "descent already solved l={l}: nothing is left for relinking to show"
            );

            let (mut pairs, mut better) = (0usize, 0usize);
            let mut lowest = f64::INFINITY;
            for i in 0..optima.len() {
                for j in 0..optima.len() {
                    if i == j {
                        continue;
                    }
                    let r = path_gauged(g, &optima[i], &optima[j]);
                    pairs += 1;
                    if r.improved() {
                        better += 1;
                        assert!(r.gain() > 0.0, "an improvement with no gain");
                    }
                    if let Some(b) = &r.best {
                        lowest = lowest.min(b.energy);
                    }
                }
            }
            assert!(
                3 * better >= pairs,
                "l={l}: only {better} of {pairs} paths beat both of their endpoints"
            );
            assert!(
                lowest < floor - 1e-9,
                "l={l}: the best relinked state was {lowest} against a best endpoint of {floor}"
            );
            solved_by_a_path += usize::from(lowest <= p.ground_energy + 1e-9);
            all_pairs += pairs;
            all_better += better;
        }
        assert!(2 * all_better >= all_pairs, "{all_better} of {all_pairs} paths improved");
        assert!(
            solved_by_a_path >= 1,
            "no path reached a planted optimum that its endpoints had missed"
        );
    }

    /// The search reaches the planted optimum, which is known rather than assumed.
    ///
    /// The only exact score available at this size, and the reason `planted` exists. Also checks the
    /// two internal claims a passing energy would hide: the returned state is never worse than the
    /// pool it kept, and at least one path per instance beat its own endpoints -- a run reporting
    /// zero there is multi-start descent wearing this module's name.
    #[test]
    fn scatter_search_reaches_the_planted_optimum() {
        for (l, loops, iseed) in CASES {
            let p = planted::frustrated_loops(l, loops, iseed);
            let g = &p.graph;
            let (mut solved, mut worst, mut improving) = (0usize, 0.0f64, 0usize);
            for seed in 0..12u64 {
                let o = search(g, &Params::default(), seed);
                assert!(
                    (o.energy - g.energy(&o.state)).abs() < 1e-9,
                    "reported {} for a state worth {}",
                    o.energy,
                    g.energy(&o.state)
                );
                let pool_best = o.pool.iter().map(|m| m.energy).fold(f64::INFINITY, f64::min);
                assert!(o.energy <= pool_best + 1e-9, "kept {pool_best} but returned {}", o.energy);
                assert_eq!(o.pool.len(), Params::default().pool, "the pool never filled");
                solved += usize::from(p.solved(&o.state));
                worst = worst.max(p.excess(&o.state));
                improving += o.improving;
            }
            assert!(2 * solved >= 12, "l={l}: solved {solved} of 12 planted instances");
            assert!(worst < 0.12, "l={l}: worst run landed {:.1}% above the optimum", worst * 100.0);
            assert!(improving > 0, "l={l}: no path anywhere beat its endpoints");
        }
    }

    /// The reference set keeps the best and refuses the near-duplicates, by exact bookkeeping.
    #[test]
    fn the_reference_set_keeps_the_best_and_stays_diverse() {
        let mut set = RefSet::new(3, 4, false);
        assert!(set.is_empty() && set.best().is_none());

        let a = vec![1i8; 10];
        assert_eq!(set.insert(&a, -5.0), Admit::Added);
        assert_eq!(set.insert(&a, -5.0), Admit::TooClose, "a pool holding one state twice is not a pool");

        // Inside the diversity floor and better: it takes that member's slot rather than a new one.
        let mut b = a.clone();
        b[0] = -1;
        b[1] = -1;
        assert_eq!(set.distance(&a, &b), 2);
        assert_eq!(set.insert(&b, -6.0), Admit::Replaced);
        assert_eq!(set.len(), 1);

        // Inside the floor and worse: refused, however good it looks against the rest.
        let mut c = b.clone();
        c[2] = -1;
        assert_eq!(set.insert(&c, -1.0), Admit::TooClose);
        assert_eq!(set.len(), 1);

        let far1 = vec![-1i8; 10];
        assert_eq!(set.insert(&far1, -4.0), Admit::Added);
        let mut far2 = vec![-1i8; 10];
        for v in far2.iter_mut().take(10).skip(6) {
            *v = 1;
        }
        assert_eq!(set.insert(&far2, -3.0), Admit::Added);
        assert_eq!(set.len(), 3, "the set is full");

        // Full, far from everything, and better than the worst: it displaces the worst.
        let mut strong = vec![1i8; 10];
        for v in strong.iter_mut().take(10).skip(6) {
            *v = -1;
        }
        assert_eq!(set.insert(&strong, -10.0), Admit::Added);
        assert_eq!(set.len(), 3);
        assert!(!set.members().iter().any(|m| m.state == far2), "the worst member should be gone");

        // Full, far from everything, and worse than the worst: refused.
        let mut weak = vec![1i8; 10];
        for v in weak.iter_mut().take(6).skip(2) {
            *v = -1;
        }
        assert_eq!(set.insert(&weak, -1.0), Admit::TooWeak);

        let energies: Vec<f64> = set.members().iter().map(|m| m.energy).collect();
        assert_eq!(energies, vec![-10.0, -6.0, -4.0], "members are ordered best first");
        assert_eq!(set.best().expect("not empty").energy, -10.0);

        // With the gauge on, a state and its global flip are one solution.
        let mut gauged = RefSet::new(2, 1, true);
        assert_eq!(gauged.insert(&a, -5.0), Admit::Added);
        let neg: Vec<i8> = a.iter().map(|&s| -s).collect();
        assert_eq!(gauged.distance(&a, &neg), 0);
        assert_eq!(gauged.insert(&neg, -5.0), Admit::TooClose);
        assert_eq!(gauged.len(), 1);

        // A floor of zero is still not a licence to hold one state twice -- a pool of duplicates
        // relinks a state with itself, walks nothing, and reports a full set.
        let mut loose = RefSet::new(3, 0, false);
        assert_eq!(loose.insert(&a, -5.0), Admit::Added);
        assert_eq!(loose.insert(&a, -5.0), Admit::TooClose);
        assert_eq!(loose.len(), 1);
        let mut nudged = a.clone();
        nudged[0] = -1;
        assert_eq!(loose.insert(&nudged, -5.0), Admit::Added, "one spin clears a floor of zero");
        assert_eq!(loose.len(), 2);
    }

    /// The gauge is free where it is taken, and it is not taken where it would not be.
    ///
    /// Free is exact: with no biases every term of the energy contains two spins, so flipping all of
    /// them changes nothing bit for bit. The refusal matters more -- on a model with fields `-to` is
    /// a different solution with a different energy, and substituting it would relink to a state the
    /// caller never named.
    #[test]
    fn the_gauge_is_free_where_it_is_taken_and_refused_where_it_is_not() {
        let p = planted::frustrated_loops(6, 48, 9);
        let g = &p.graph;
        assert!(field_free(g), "a planted instance carries no biases");
        let a = optimum(g, 1);
        let neg: Vec<i8> = a.iter().map(|&s| -s).collect();
        assert_eq!(g.energy(&a), g.energy(&neg), "Z2 symmetry, term by term");

        let r = path_gauged(g, &a, &neg);
        assert!(r.gauged && r.flips.is_empty(), "a state and its flip are one solution");
        assert!(r.best.is_none() && r.proposals == 0);
        assert_eq!(r.from_energy, r.to_energy);

        // Ungauged, the same pair walks the entire lattice to arrive back where it started.
        let q = path(g, &a, &neg);
        assert!(!q.gauged);
        assert_eq!(q.flips.len(), g.n);

        // With a bias anywhere, the substitution is refused.
        let mut bb = GraphBuilder::new(6);
        for i in 0..5 {
            bb.couple(i, i + 1, 1.0);
        }
        bb.bias(2, 0.75);
        let biased = bb.build();
        assert!(!field_free(&biased));
        let x = vec![1i8; 6];
        let y = vec![-1i8; 6];
        let br = path_gauged(&biased, &x, &y);
        assert!(!br.gauged, "a biased model has no gauge to take");
        assert_eq!(br.flips.len(), 6);
        assert!((br.to_energy - biased.energy(&y)).abs() < 1e-12, "it walked to the named target");
    }

    /// Endpoints that are equal or adjacent have nothing between them, and say so.
    #[test]
    fn degenerate_paths_have_no_intermediate() {
        let p = planted::frustrated_loops(5, 20, 4);
        let g = &p.graph;
        let a = optimum(g, 2);

        let same = path(g, &a, &a);
        assert!(same.flips.is_empty() && same.best.is_none() && same.proposals == 0);
        assert_eq!(same.from_energy, same.to_energy);
        assert!(!same.improved() && same.gain() == 0.0);

        let mut b = a.clone();
        b[3] = -b[3];
        let adjacent = path(g, &a, &b);
        assert_eq!(adjacent.flips, vec![3]);
        assert!(adjacent.best.is_none(), "adjacent endpoints have no state between them");
        assert!(!adjacent.improved() && adjacent.gain() == 0.0);
        assert_eq!(adjacent.proposals, 1);

        // A graph with no nodes is a search with no answer, not a panic.
        let empty = GraphBuilder::new(0).build();
        let o = search(&empty, &Params::default(), 1);
        assert!(o.state.is_empty() && o.relinks == 0 && o.proposals == 0);
    }

    /// Replaying past the end of a path is refused rather than clamped.
    #[test]
    #[should_panic(expected = "has no step")]
    fn replaying_past_the_end_of_a_path_is_refused() {
        let _ = replay(&[1i8, -1, 1], &[0, 2], 3);
    }

    /// The portfolio arm spends what it was given and reports the state it returns.
    ///
    /// The same two bounds `portfolio` holds its own arms to: an arm that overspends makes the
    /// comparison a measurement of who cheated, and one that spends a fraction is not answering the
    /// same question. The slack is one descent scan, `n`, which is charged before the cap is read.
    #[test]
    fn the_arm_stays_inside_its_budget_and_reports_the_state_it_returns() {
        let p = planted::frustrated_loops(8, 96, 3);
        let g = &p.graph;
        for b in [200_000u64, 800_000] {
            let f = Scatter.solve(g, Budget::new(b), 4);
            assert_eq!(f.arm, "scatter");
            assert!(f.spent <= b + g.n as u64, "spent {} against a budget of {b}", f.spent);
            assert!(f.spent * 4 >= b, "spent only {} of {b}", f.spent);
            assert_eq!(f.state.len(), g.n);
            assert!(f.state.iter().all(|&v| v == 1 || v == -1));
            assert!((f.energy - g.energy(&f.state)).abs() < 1e-9);
        }
        // The cap is read inside the descent rather than only between them, which is what holds the
        // overspend to a single scan. Checked against the same descent run without a cap.
        let mut rng = Pcg::new(19, 5);
        let start: Vec<i8> = (0..g.n).map(|_| rng.spin(0.5)).collect();
        let (mut full_state, mut full) = (start.clone(), 0u64);
        descend(g, &mut full_state, &mut full, u64::MAX);
        let half = full / 2;
        let (mut cut_state, mut cut) = (start.clone(), 0u64);
        descend(g, &mut cut_state, &mut cut, half);
        assert!(cut <= half + g.n as u64, "a capped descent spent {cut} of {half}");
        assert!(cut_state != full_state, "a capped descent must stop short of the optimum");

        // And it composes as an arm without the portfolio having to know about it.
        let arms: Vec<Box<dyn Search>> = vec![Box::new(Scatter), Box::new(crate::portfolio::Tabu)];
        let r = crate::portfolio::run(g, &arms, Budget::new(400_000), 5);
        assert_eq!(r.arms.len(), 2);
        for arm in &r.arms {
            assert!(r.best.energy <= arm.energy + 1e-12);
        }
    }
}
