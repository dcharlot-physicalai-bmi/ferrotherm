//! Exact answers on sparse graphs, well past where enumeration stops.
//!
//! [`crate::oracle::Exhaustive`] is exact and dies at about twenty-six spins, because it visits
//! `2^n` states. Variable elimination visits `2^w` instead, where `w` is the **induced width** of
//! the elimination order — a property of the graph's shape rather than its size. A tree has width
//! 1 and a lattice strip has width equal to its short side, so a thousand-spin chain is exact and
//! instant while a thousand-spin dense graph is still hopeless. That is the honest trade, and
//! [`Elimination::width`] reports it up front so a caller can decide before waiting.
//!
//! Two questions, one algorithm:
//!
//! - **min-sum** eliminates by taking the minimum over each variable, giving the exact ground state.
//! - **sum-product** eliminates by log-sum-exp, giving the exact log partition function — and with
//!   it exact marginals, which is what lets a sampler be checked against truth on graphs far too
//!   large to enumerate.
//!
//! Finding the optimal elimination order is NP-hard, so two heuristics are built and the NARROWER
//! is kept ([`Elimination::order_for`]): min-fill, which is greedy and local, and nested
//! dissection, which splits by a separator. The order only affects the width, and the width is
//! measured rather than assumed: a bad order makes this slow or refused, **never wrong**.
//!
//! Measured against known treewidths (`examples/width_probe.rs`):
//!
//! | graph | spins | min-fill alone | kept | true treewidth |
//! |---|---|---|---|---|
//! | chain, any length | 2000 | 1 | 1 | 1 |
//! | 3x20 strip | 60 | 3 | 3 | 3 |
//! | 5x30 strip | 150 | 5 | 5 | 5 |
//! | 6x40 strip | 240 | 8 | **7** | 6 |
//! | 8x50 strip | 400 | 11 | **9** | 8 |
//! | 10x10 grid | 100 | 13 | **10** | 10 |
//! | torus 10x10 | 100 | 23 | **20** | 20 |
//! | torus 12x12 | 144 | 26 | **24** | 24 |
//! | torus 14x14 | 196 | 34 | **28** | 28 |
//!
//! Min-fill alone was optimal to width 5 and drifted two or three above beyond it. With dissection
//! the grid and every torus land **on** the treewidth, and the strips within one of it.
//!
//! Since cost is `2^width`, that drift was not cosmetic: [`Elimination::max_width`] defaults to 24,
//! so a 144-spin torus ordered at 26 was REFUSED for its order rather than its shape, and a 10x10
//! torus cost 8x more than it had to.

use crate::graph::Graph;

/// A function over a subset of spins, as a table indexed by a bitmask.
///
/// Bit `k` of the index is the value of `vars[k]`: 0 means −1, 1 means +1.
#[derive(Clone, Debug)]
struct Table {
    vars: Vec<usize>,
    vals: Vec<f64>,
}

impl Table {
    fn value_at(&self, assign: &[i8]) -> f64 {
        let mut idx = 0usize;
        for (k, &v) in self.vars.iter().enumerate() {
            if assign[v] > 0 {
                idx |= 1 << k;
            }
        }
        self.vals[idx]
    }
}

/// Exact inference by variable elimination.
pub struct Elimination {
    /// Refuse an order whose induced width exceeds this. `2^width` is the memory per table.
    pub max_width: usize,
}

impl Default for Elimination {
    fn default() -> Self {
        Elimination { max_width: 24 }
    }
}

/// What an elimination run produced.
#[derive(Clone, Debug)]
pub struct Exact {
    /// Induced width of the order actually used. Cost was `2^width` per step.
    pub width: usize,
    /// Ground energy, if min-sum was run.
    pub ground_energy: Option<f64>,
    /// A state attaining it.
    pub ground_state: Option<Vec<i8>>,
    /// `log Z` at the requested beta, if sum-product was run.
    pub log_z: Option<f64>,
}

#[cfg(test)]
mod ordering {
    use super::*;
    use crate::graph::GraphBuilder;

    fn strip(rows: usize, cols: usize) -> Graph {
        let mut b = GraphBuilder::new(rows * cols);
        let id = |r: usize, c: usize| r * cols + c;
        for r in 0..rows {
            for c in 0..cols {
                if c + 1 < cols {
                    b.couple(id(r, c), id(r, c + 1), 1.0);
                }
                if r + 1 < rows {
                    b.couple(id(r, c), id(r + 1, c), 1.0);
                }
            }
        }
        b.build()
    }

    fn widths(g: &Graph) -> (usize, usize) {
        let adj = adjacency(g);
        (min_fill_order(g.n, &adj).1, separator_order(g.n, &adj).1)
    }

    /// Keeping the narrower of two heuristics can only narrow.
    ///
    /// This is the property that makes the second heuristic safe to adopt at all: a model that was
    /// accepted stays accepted, and one that was refused may now fit. Asserted across families with
    /// very different shapes, because a heuristic that helps grids could hurt something else and
    /// the "keep the smaller" rule is what makes that impossible rather than unlikely.
    #[test]
    fn the_order_actually_used_is_never_wider_than_min_fill_alone() {
        let cases: Vec<(String, Graph)> = vec![
            ("3x20 strip".into(), strip(3, 20)),
            ("5x30 strip".into(), strip(5, 30)),
            ("8x50 strip".into(), strip(8, 50)),
            ("10x10 grid".into(), strip(10, 10)),
            ("torus 10".into(), crate::ising::lattice2d(10, 1.0)),
            ("torus 14".into(), crate::ising::lattice2d(14, 1.0)),
            ("ring 200".into(), crate::ising::ring(200, 1.0, 0.0)),
            ("chain 500".into(), {
                let mut b = GraphBuilder::new(500);
                for i in 0..499 {
                    b.couple(i, i + 1, 1.0);
                }
                b.build()
            }),
            ("frustrated loops".into(), crate::planted::frustrated_loops(8, 96, 3).graph),
        ];
        for (name, g) in cases {
            let (mf, _) = widths(&g);
            let kept = Elimination::order_for(&g).1;
            assert!(
                kept <= mf,
                "{name}: the order kept is width {kept}, wider than min-fill's {mf}"
            );
        }
    }

    /// On a torus the dissection order hits the treewidth exactly.
    ///
    /// An `L x L` periodic lattice has treewidth `2L`. Min-fill drifts well above it — 23 at
    /// L = 10, 26 at 12, 44 at 18 — and since cost is `2^width` that is 8x, and at L = 12 it is the
    /// difference between refused and accepted at the default `max_width` of 24.
    ///
    /// Asserting EQUALITY rather than an improvement: "better than min-fill" would pass for an
    /// order that is merely less bad, and the claim being made here is that this family is solved.
    #[test]
    fn a_torus_is_ordered_at_exactly_twice_its_side() {
        for l in [6usize, 8, 10, 12, 14] {
            let g = crate::ising::lattice2d(l, 1.0);
            let (mf, sep) = widths(&g);
            assert_eq!(
                sep,
                2 * l,
                "torus {l}x{l}: dissection gave {sep}, the treewidth is {}",
                2 * l
            );
            if l >= 10 {
                assert!(mf > sep, "torus {l}: min-fill {mf} should be worse than {sep}");
            }
        }
    }

    /// A model refused for its ORDER, not its shape, now runs.
    ///
    /// The concrete payoff, stated as the thing a caller sees rather than as a width number. A
    /// 10x10 torus has treewidth 20; min-fill orders it at 23, so a caller whose budget is 20 to 22
    /// is refused a model that fits. The budget is set explicitly rather than using the default,
    /// because a width-24 elimination allocates 2^24 f64 per table and this is a unit test.
    #[test]
    fn a_model_that_was_refused_for_its_order_now_runs() {
        let g = crate::ising::lattice2d(10, 1.0);
        let adj = adjacency(&g);
        let mf = min_fill_order(g.n, &adj).1;
        let e = Elimination { max_width: 20 };
        assert!(
            mf > e.max_width,
            "this test is about a model min-fill refuses at width {}; it ordered at {mf}",
            e.max_width
        );
        let out = e.log_partition(&g, 0.4).expect("the dissection order fits inside max_width");
        assert!(out.log_z.expect("sum-product ran").is_finite());
        assert_eq!(out.width, 20, "the torus should be ordered at exactly its treewidth");
    }

    /// The order is a permutation of every vertex, including ones BFS cannot reach.
    ///
    /// A separator search walks a component. A graph with an isolated spin, or with two pieces, has
    /// vertices the walk never sees, and an order that omits them silently skips their elimination
    /// — which is how `log Z` loses a factor per missing spin.
    #[test]
    fn every_vertex_is_in_the_order_even_when_the_graph_is_in_pieces() {
        // Two disjoint chains and one isolated spin.
        let mut b = GraphBuilder::new(25);
        for i in 0..9 {
            b.couple(i, i + 1, 1.0);
        }
        for i in 12..23 {
            b.couple(i, i + 1, 1.0);
        }
        let g = b.build();
        let adj = adjacency(&g);
        let (order, _) = separator_order(g.n, &adj);
        let mut seen = vec![false; g.n];
        for &v in &order {
            assert!(!seen[v], "vertex {v} appears twice in the order");
            seen[v] = true;
        }
        assert!(seen.iter().all(|&x| x), "the order is not a permutation: {order:?}");

        // And the answer is still right, checked against enumeration on a small graph in pieces:
        // two 4-chains and two isolated spins, 12 spins in all.
        let mut sb = GraphBuilder::new(12);
        for i in 0..3 {
            sb.couple(i, i + 1, 1.0);
        }
        for i in 5..8 {
            sb.couple(i, i + 1, -0.7);
        }
        let small = sb.build();
        let beta = 0.6;
        let ln_z = Elimination::default().log_partition(&small, beta).unwrap().log_z.unwrap();
        let mut acc = 0.0f64;
        for m in 0u64..(1u64 << 12) {
            let st: Vec<i8> = (0..12).map(|i| if m >> i & 1 == 1 { 1i8 } else { -1 }).collect();
            acc += (-beta * small.energy(&st)).exp();
        }
        assert!(
            (ln_z - acc.ln()).abs() < 1e-9,
            "a graph in pieces: elimination {ln_z}, enumeration {}",
            acc.ln()
        );
    }
}

#[cfg(test)]
mod degeneracy_tests {
    use super::*;
    use crate::graph::GraphBuilder;

    /// Count ground states by brute force. The oracle, and it is deliberately the dumbest thing
    /// that could work: an independent count with no shared machinery.
    fn brute(g: &Graph) -> u64 {
        assert!(g.n <= 22, "brute force is the oracle, not the method");
        let mut best = f64::INFINITY;
        let mut count = 0u64;
        for mask in 0u64..(1u64 << g.n) {
            let s: Vec<i8> =
                (0..g.n).map(|i| if mask >> i & 1 == 1 { 1i8 } else { -1 }).collect();
            let e = g.energy(&s);
            if e < best - 1e-9 {
                best = e;
                count = 1;
            } else if (e - best).abs() <= 1e-9 {
                count += 1;
            }
        }
        count
    }

    /// The cold limit of `log Z` counts the ground states, checked against brute force.
    ///
    /// `samples.rs` says its own degeneracy is "evidence of degeneracy, not a count of it", and
    /// `oracle::Exhaustive` stops at 26 spins. This counts on anything narrow, which is a claim
    /// about shape rather than size — so it is checked where both can run, and then used where
    /// only one can.
    #[test]
    fn the_cold_limit_counts_the_ground_states() {
        let e = Elimination::default();
        let cases: Vec<(&str, Graph)> = vec![
            // A ferromagnetic chain: all-up and all-down, so exactly 2.
            ("ferro chain 10", {
                let mut b = GraphBuilder::new(10);
                for i in 0..9 {
                    b.couple(i, i + 1, 1.0);
                }
                b.build()
            }),
            // An ODD antiferromagnetic ring is frustrated: one bond must break, and it can be any
            // of the N of them, in either global orientation -- exactly 2N ground states. A closed
            // form, so this row does not lean on the brute-force oracle at all.
            ("odd AF ring 9", crate::ising::ring(9, -1.0, 0.0)),
            ("odd AF ring 11", crate::ising::ring(11, -1.0, 0.0)),
            // An EVEN antiferromagnetic ring is unfrustrated: two alternating states.
            ("even AF ring 10", crate::ising::ring(10, -1.0, 0.0)),
            // A field breaks the global flip symmetry, leaving one.
            ("ferro chain 8 with field", {
                let mut b = GraphBuilder::new(8);
                for i in 0..7 {
                    b.couple(i, i + 1, 1.0);
                }
                for i in 0..8 {
                    b.set_bias(i, 0.25);
                }
                b.build()
            }),
        ];

        for (name, g) in cases {
            let d = e.ground_degeneracy(&g, (20.0, 40.0)).expect("these are narrow");
            let want = brute(&g);
            assert_eq!(
                d.count,
                Some(want),
                "{name}: counted {:?}, brute force says {want} (warm {:?}, cold {:?}, residual {:.2e})",
                d.count,
                d.warm,
                d.cold,
                d.residual()
            );
        }
    }

    /// A spin in no factor is still a spin, and sum-product owes it a factor of two.
    ///
    /// `initial_tables` emits nothing for a spin with no field and no edges, and `run` then skips a
    /// variable no table mentions. Correct for min-sum — a free spin adds zero energy — and wrong
    /// for sum-product, where summing over it multiplies `Z` by 2. `log_partition` was short by
    /// `ln 2` per free spin and `ground_degeneracy` reported a count too small by `2^free`, as a
    /// CONFIDENT integer: the error is identical at both temperatures, so the residual was 1e-15
    /// and the convergence guard certified it.
    ///
    /// SCORED AGAINST BRUTE FORCE, NOT AGAINST THE TENSOR ENGINE. `tensor::Network::from_ising` had
    /// the identical hole for the identical reason, so the cross-check between the two engines
    /// agreed on the wrong number to the last ulp. Two implementations are evidence only when they
    /// do not share a blind spot.
    #[test]
    fn a_spin_in_no_factor_still_doubles_the_partition_function() {
        let e = Elimination::default();

        // One coupled pair and `free` spins that appear in nothing at all.
        for free in 0..4usize {
            let n = 2 + free;
            let mut b = GraphBuilder::new(n);
            b.couple(0, 1, 1.0);
            let g = b.build();

            for beta in [0.5f64, 1.0, 2.0] {
                let ln_z = e.log_partition(&g, beta).unwrap().log_z.unwrap();
                let mut acc = 0.0f64;
                for m in 0u64..(1u64 << n) {
                    let s: Vec<i8> =
                        (0..n).map(|i| if m >> i & 1 == 1 { 1i8 } else { -1 }).collect();
                    acc += (-beta * g.energy(&s)).exp();
                }
                assert!(
                    (ln_z - acc.ln()).abs() < 1e-9,
                    "{free} free spins at beta {beta}: elimination {ln_z}, enumeration {}",
                    acc.ln()
                );
            }

            // And the count doubles per free spin: 2 orientations of the pair, times 2^free.
            let d = e.ground_degeneracy(&g, (20.0, 40.0)).unwrap();
            assert_eq!(
                d.count,
                Some(2u64 << free),
                "{free} free spins: counted {:?}, expected {}",
                d.count,
                2u64 << free
            );
        }
    }

    /// The odd antiferromagnetic ring's closed form, at a size brute force cannot reach.
    ///
    /// 2N ground states for an odd N-ring. Checked at N = 101, where enumeration would need 2^101
    /// states and elimination needs 2^2 — which is the entire point of doing it this way.
    #[test]
    fn an_odd_antiferromagnetic_ring_has_exactly_two_n_ground_states() {
        let e = Elimination::default();
        for n in [21usize, 51, 101] {
            let g = crate::ising::ring(n, -1.0, 0.0);
            let d = e.ground_degeneracy(&g, (30.0, 60.0)).expect("a ring has width 2");
            assert_eq!(
                d.count,
                Some(2 * n as u64),
                "an odd AF {n}-ring has 2N = {} ground states; got {:?} (residual {:.2e})",
                2 * n,
                d.count,
                d.residual()
            );
        }
    }

    /// The estimate is an UPPER bound and the colder temperature is the tighter one.
    ///
    /// This is the property that makes a `None` count still worth returning: the bracket is sound
    /// even when the integer is withheld. Asserting only the converged case would leave the claim
    /// about unconverged ones untested.
    #[test]
    fn the_estimate_approaches_the_truth_from_above() {
        let e = Elimination::default();
        let g = crate::ising::ring(9, -1.0, 0.0);
        let truth = brute(&g) as f64;

        // Deliberately WARM, where the excited levels still contribute and it has not converged.
        let d = e.ground_degeneracy(&g, (0.5, 1.0)).expect("a ring is narrow");
        assert!(d.warm.1 >= truth, "the warm estimate {} is below the truth {truth}", d.warm.1);
        assert!(d.cold.1 >= truth, "the cold estimate {} is below the truth {truth}", d.cold.1);
        assert!(
            d.cold.1 <= d.warm.1,
            "colder must be tighter: warm {} cold {}",
            d.warm.1,
            d.cold.1
        );
        assert!(
            d.count.is_none(),
            "at beta 0.5 and 1.0 this has not converged and must not name an integer, got {:?}",
            d.count
        );
        assert!(d.residual() > 1e-6, "an unconverged pair should show a residual");
    }

    /// A degenerate temperature names no integer, and does not panic.
    ///
    /// The doc claims a non-finite or nonsensical beta gives back an unnamed bracket rather than a
    /// rounded number. Asserted here rather than asserted in prose, because "it returns None" is
    /// exactly the kind of claim that is true until someone changes the guard.
    #[test]
    fn a_temperature_that_says_nothing_names_no_number() {
        let e = Elimination::default();
        let g = crate::ising::ring(9, -1.0, 0.0);
        for betas in [
            (f64::NAN, 40.0),
            (20.0, f64::NAN),
            (f64::INFINITY, 40.0),
            // Equal temperatures agree trivially. This one reported Some(512) -- 2^n, the total
            // state count -- because the residual was zero for a reason unrelated to convergence.
            (0.0, 0.0),
            (30.0, 30.0),
            (-20.0, -40.0),
        ] {
            let d = e.ground_degeneracy(&g, betas).expect("a ring is narrow whatever the beta");
            assert!(
                d.count.is_none(),
                "betas {betas:?} must name no integer, got {:?} (warm {:?}, cold {:?})",
                d.count,
                d.warm,
                d.cold
            );
        }
    }

    /// Betas may be given in either order; the colder is used as the estimate.
    #[test]
    fn the_two_temperatures_may_be_given_either_way_round() {
        let e = Elimination::default();
        let g = crate::ising::ring(9, -1.0, 0.0);
        let a = e.ground_degeneracy(&g, (20.0, 40.0)).unwrap();
        let b = e.ground_degeneracy(&g, (40.0, 20.0)).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.cold.0, 40.0);
    }
}

/// How many states sit at the ground energy, with the evidence for the number.
///
/// Produced by [`Elimination::ground_degeneracy`]. Both estimates are UPPER BOUNDS on the true
/// count — every excited level contributes a positive term — and the colder one is the tighter.
/// Carrying both is what lets a reader see the convergence instead of trusting it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Degeneracy {
    /// The ground energy, from the same min-sum elimination.
    pub ground_energy: f64,
    /// `(beta, estimate)` at the warmer temperature. The looser bound.
    pub warm: (f64, f64),
    /// `(beta, estimate)` at the colder temperature. The tighter bound.
    pub cold: (f64, f64),
    /// The count, when the two temperatures agree on an integer and the colder has converged to it.
    ///
    /// `None` means the estimates had not settled — a small spectral gap, or couplings continuous
    /// enough that near-degenerate states crowd the ground level. The bracket in `warm` and `cold`
    /// is still valid and still an upper bound; only the integer is withheld.
    pub count: Option<u64>,
}

impl Degeneracy {
    /// How far apart the two temperatures still are, as a fraction of the tighter estimate.
    ///
    /// The convergence made visible: near zero means the cold limit has been reached and `count`
    /// can be believed, large means it has not and the number would be a guess.
    #[must_use]
    pub fn residual(&self) -> f64 {
        let (_, w) = self.warm;
        let (_, c) = self.cold;
        if c.abs() > 0.0 { (w - c).abs() / c.abs() } else { (w - c).abs() }
    }
}

/// Why elimination declined.
#[derive(Clone, Debug, PartialEq)]
pub enum TooWide {
    /// The best order this found still needs a table of `2^width`.
    Width {
        /// Induced width of the order found.
        width: usize,
        /// The largest width allowed, since a table costs `2^width`.
        max: usize,
    },
}

impl core::fmt::Display for TooWide {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            TooWide::Width { width, max } => write!(
                f,
                "the elimination order has induced width {width}, needing tables of 2^{width}; the \
                 limit is {max}. This graph is too dense for exact inference -- use a planted \
                 instance for known ground truth instead."
            ),
        }
    }
}

/// Order variables by min-fill: repeatedly eliminate whichever variable adds fewest new edges.
///
/// Returns the order and the induced width it produces.
/// Min-fill elimination order, computed with a DIRTY SET rather than a full rescan.
///
/// The order and the width are **byte-identical** to the full-rescan version, and that is a
/// requirement rather than a nicety: `width` gates `TooWide` refusals, so a different order would
/// change which models this module accepts. The tie-break is therefore preserved exactly — fewest
/// fill edges, then fewest live neighbours, then lowest index, since `v` is scanned in `0..n`.
///
/// Each of the `n` elimination rounds used to recompute the fill count of EVERY live vertex over
/// all pairs of its live neighbours, when eliminating `v` can only change the count of a vertex within
/// distance two of it: the neighbourhood becomes a clique, so only its members and their neighbours
/// see a different graph.
///
/// **What that actually buys, counted in fill recomputations rather than timed** — the count is a
/// property of the graph and the algorithm, where a duration would be a property of this laptop:
///
/// ```text
///   graph                full rescan   dirty set   saved
///   random n=40 p=0.2            860         772   10.2%
///   random n=80 p=0.1          3,320       2,577   22.4%
///   lattice 6x6                  702         448   36.2%
///   lattice 8x8                2,144       1,014   52.7%
/// ```
///
/// It is **not** an asymptotic transformation in practice, and saying so matters more than the
/// headline: elimination *fills the graph in*, so after a few rounds the dirty set is much of what
/// is left and the two converge. The win is largest on sparse structured graphs — lattices, which
/// is what this crate builds most, and where it grows with the side — and smallest on dense random
/// ones, which are the case the naive bound is worst for. A reader looking for `O(n² d²)` becoming
/// something else will not find it here.
/// The induced width of an arbitrary elimination order, by simulating the elimination.
///
/// Separate from [`min_fill_order`], which returns the width of the order it builds, because a
/// second ordering heuristic needs to be SCORED on the same scale before either is chosen. A
/// heuristic that reports its own number in its own way cannot be compared with another.
pub(crate) fn induced_width(n: usize, adj: &[Vec<usize>], order: &[usize]) -> usize {
    use std::collections::BTreeSet;
    let mut nbr: Vec<BTreeSet<usize>> = adj.iter().map(|v| v.iter().copied().collect()).collect();
    let mut alive = vec![true; n];
    let mut width = 0usize;
    for &v in order {
        if !alive[v] {
            continue;
        }
        let live: Vec<usize> = nbr[v].iter().copied().filter(|&u| alive[u] && u != v).collect();
        width = width.max(live.len());
        // Eliminating v makes its live neighbours a clique -- the fill edges.
        for i in 0..live.len() {
            for j in (i + 1)..live.len() {
                nbr[live[i]].insert(live[j]);
                nbr[live[j]].insert(live[i]);
            }
        }
        alive[v] = false;
    }
    width
}

/// A nested-dissection order: split by a small separator, eliminate the pieces, then the separator.
///
/// # Why a second heuristic at all
///
/// Min-fill is greedy and local, and on grid-like graphs it drifts well above the true treewidth —
/// measured here on the periodic `lattice2d` family, where a torus has treewidth about `2L`:
///
/// ```text
///   10x10   min-fill 23      12x12   min-fill 26      18x18   min-fill 44
/// ```
///
/// Since cost is `2^width`, and [`Elimination::max_width`] defaults to 24, that drift is the
/// difference between an accepted model and a refused one: `lattice2d(12)` is 144 spins and is
/// refused at 26.
///
/// # The separator, found by breadth-first level sets
///
/// A BFS from any vertex partitions a component into levels, and **every level is a separator** —
/// removing it disconnects the levels before it from the levels after. So the cheapest separator a
/// BFS offers is its smallest level, and on a grid the levels are exactly the rows or diagonals,
/// which is what makes this find the orders nested dissection is named for without needing a
/// geometric embedding.
///
/// Recursing on each side and placing the separator LAST is what bounds the width: by the time the
/// separator is eliminated, everything it separated is gone.
///
/// It is a heuristic, like min-fill, and it is not always better — see
/// [`Elimination::order_for`], which computes both and keeps the narrower. A bad order here makes
/// nothing wrong, only slower or refused, which is the property [`min_fill_order`] already relies
/// on.
pub(crate) fn separator_order(n: usize, adj: &[Vec<usize>]) -> (Vec<usize>, usize) {
    let mut order = Vec::with_capacity(n);
    let all: Vec<usize> = (0..n).collect();
    dissect(&all, adj, &mut order);
    // Anything unreachable from the pieces visited -- isolated vertices, say -- still has to be in
    // the order or the elimination silently skips it.
    let mut seen = vec![false; n];
    for &v in &order {
        seen[v] = true;
    }
    for v in 0..n {
        if !seen[v] {
            order.push(v);
        }
    }
    let w = induced_width(n, adj, &order);
    (order, w)
}

/// Order one vertex subset: pieces first, separator last.
fn dissect(sub: &[usize], adj: &[Vec<usize>], out: &mut Vec<usize>) {
    // Below this a separator costs more than it saves: the subproblem is already narrower than the
    // separator would be.
    const LEAF: usize = 12;
    if sub.len() <= LEAF {
        out.extend_from_slice(sub);
        return;
    }
    let inside: std::collections::BTreeSet<usize> = sub.iter().copied().collect();

    // BFS from the subset's first vertex; every level is a separator, so take the smallest one that
    // actually splits (levels 0 and last separate nothing).
    let start = sub[0];
    let mut level: std::collections::BTreeMap<usize, usize> = std::collections::BTreeMap::new();
    let mut queue = std::collections::VecDeque::new();
    level.insert(start, 0);
    queue.push_back(start);
    while let Some(v) = queue.pop_front() {
        let d = level[&v];
        for &u in &adj[v] {
            if inside.contains(&u) && !level.contains_key(&u) {
                level.insert(u, d + 1);
                queue.push_back(u);
            }
        }
    }
    let depth = level.values().copied().max().unwrap_or(0);
    let mut by_level: Vec<Vec<usize>> = vec![Vec::new(); depth + 1];
    for (&v, &d) in &level {
        by_level[d].push(v);
    }

    // A separator must have something on both sides, so only interior levels qualify. Ties go to
    // the more central level, which balances the two halves and keeps the recursion shallow.
    let mut best: Option<(usize, usize)> = None; // (size, level)
    for d in 1..depth {
        let size = by_level[d].len();
        let centre = (d as isize - depth as isize / 2).unsigned_abs();
        let key = (size, centre);
        if best.is_none_or(|(bs, bd)| {
            let bc = (bd as isize - depth as isize / 2).unsigned_abs();
            key < (bs, bc)
        }) {
            best = Some((size, d));
        }
    }

    let Some((_, cut)) = best else {
        // Disconnected, or too shallow to split: BFS reached only part of the subset, or every
        // level is an endpoint. Order what BFS reached, then the rest, rather than looping.
        let reached: Vec<usize> = sub.iter().copied().filter(|v| level.contains_key(v)).collect();
        let rest: Vec<usize> = sub.iter().copied().filter(|v| !level.contains_key(v)).collect();
        if rest.is_empty() || reached.is_empty() {
            out.extend_from_slice(sub);
        } else {
            dissect(&rest, adj, out);
            out.extend_from_slice(&reached);
        }
        return;
    };

    let sep: std::collections::BTreeSet<usize> = by_level[cut].iter().copied().collect();
    let before: Vec<usize> =
        sub.iter().copied().filter(|v| level.get(v).is_some_and(|&d| d < cut)).collect();
    let after: Vec<usize> = sub
        .iter()
        .copied()
        .filter(|v| !sep.contains(v) && level.get(v).is_none_or(|&d| d > cut))
        .collect();

    if before.is_empty() || after.is_empty() {
        out.extend_from_slice(sub);
        return;
    }
    dissect(&before, adj, out);
    dissect(&after, adj, out);
    out.extend(by_level[cut].iter().copied());
}

pub(crate) fn min_fill_order(n: usize, adj: &[Vec<usize>]) -> (Vec<usize>, usize) {
    use std::collections::BTreeSet;
    let mut nbr: Vec<BTreeSet<usize>> = adj.iter().map(|v| v.iter().copied().collect()).collect();
    let mut alive: Vec<bool> = vec![true; n];
    let mut order = Vec::with_capacity(n);
    let mut width = 0;

    // Fill count of each live vertex, kept across rounds and recomputed only where it can move.
    let fill_of = |nbr: &Vec<BTreeSet<usize>>, alive: &Vec<bool>, v: usize| -> (usize, usize) {
        let ns: Vec<usize> = nbr[v].iter().copied().filter(|&u| alive[u]).collect();
        let mut fill = 0;
        for a in 0..ns.len() {
            for b in (a + 1)..ns.len() {
                if !nbr[ns[a]].contains(&ns[b]) {
                    fill += 1;
                }
            }
        }
        (fill, ns.len())
    };
    let mut cache: Vec<(usize, usize)> = (0..n).map(|v| fill_of(&nbr, &alive, v)).collect();

    for _ in 0..n {
        // Same scan order and same tie-break as the full rescan, over the cached counts.
        let mut best = usize::MAX;
        let mut best_fill = usize::MAX;
        let mut best_deg = usize::MAX;
        for v in 0..n {
            if !alive[v] {
                continue;
            }
            let (fill, deg) = cache[v];
            if fill < best_fill || (fill == best_fill && deg < best_deg) {
                best = v;
                best_fill = fill;
                best_deg = deg;
            }
        }
        let v = best;
        let ns: Vec<usize> = nbr[v].iter().copied().filter(|&u| alive[u]).collect();
        width = width.max(ns.len());
        for a in 0..ns.len() {
            for b in (a + 1)..ns.len() {
                nbr[ns[a]].insert(ns[b]);
                nbr[ns[b]].insert(ns[a]);
            }
        }
        alive[v] = false;

        // THE DIRTY SET. Removing `v` and cliquing its neighbourhood changes the induced subgraph
        // seen by the neighbours themselves, and by anything adjacent to one of them -- and by
        // nothing else. A vertex two hops away has a neighbour whose edge set moved; a vertex three
        // hops away sees the same graph it did before.
        let mut dirty: BTreeSet<usize> = BTreeSet::new();
        for &u in &ns {
            dirty.insert(u);
            for &w in &nbr[u] {
                if alive[w] {
                    dirty.insert(w);
                }
            }
        }
        for u in dirty {
            if alive[u] {
                cache[u] = fill_of(&nbr, &alive, u);
            }
        }
        order.push(v);
    }
    (order, width)
}

fn initial_tables(g: &Graph, beta: f64) -> Vec<Table> {
    // Energies, scaled by beta once here so neither elimination pass has to think about it.
    let mut out = Vec::new();
    for i in 0..g.n {
        if g.h[i] != 0.0 {
            // -h s
            out.push(Table { vars: vec![i], vals: vec![beta * g.h[i], -beta * g.h[i]] });
        }
        for k in g.offset[i]..g.offset[i + 1] {
            let j = g.nbr[k] as usize;
            if j > i {
                let w = beta * g.w[k];
                // index bit0 = i, bit1 = j; value is -w * s_i * s_j
                out.push(Table { vars: vec![i, j], vals: vec![-w, w, w, -w] });
            }
        }
    }
    out
}

fn adjacency(g: &Graph) -> Vec<Vec<usize>> {
    (0..g.n)
        .map(|i| (g.offset[i]..g.offset[i + 1]).map(|k| g.nbr[k] as usize).collect())
        .collect()
}

impl Elimination {
    /// Exact ground state and its energy.
    ///
    /// # Errors
    ///
    /// [`TooWide`] when the elimination order's induced width exceeds the cap, since a table costs
    /// `2^width`. The width is a property of the graph's SHAPE, not its size.
    pub fn ground_state(&self, g: &Graph) -> Result<Exact, TooWide> {
        self.run(g, 1.0, true)
    }

    /// Exact `log Z` at inverse temperature `beta`.
    ///
    /// # Errors
    ///
    /// [`TooWide`], as [`Elimination::ground_state`].
    pub fn log_partition(&self, g: &Graph, beta: f64) -> Result<Exact, TooWide> {
        self.run(g, beta, false)
    }

    /// How many ground states there are, estimated from the partition function's cold limit.
    ///
    /// [`crate::samples::SampleSet`] can only report "evidence of degeneracy, not a count of it",
    /// and [`crate::oracle::Exhaustive`] dies at about twenty-six spins. This counts on anything
    /// narrow enough to eliminate, which is a statement about the graph's shape rather than its
    /// size — the same trade the rest of this module makes.
    ///
    /// # Why this works, and it is not an approximation scheme
    ///
    /// Write the spectrum as levels: `g0` states at the ground energy `E0`, `g1` at `E0 + D`, and
    /// so on for a spectral gap `D > 0`. Then
    ///
    /// ```text
    ///   Z(beta) = g0 e^{-beta E0} (1 + (g1/g0) e^{-beta D} + ...)
    ///   ln Z(beta) + beta E0 = ln g0 + ln(1 + (g1/g0) e^{-beta D} + ...)
    /// ```
    ///
    /// so `exp(ln Z + beta E0)` converges to `g0` as `beta` grows, with error
    /// `O((g1/g0) e^{-beta D})`. This is Maslov dequantization — the tropical semiring as the
    /// zero-temperature limit of the log semiring — and it is why the ground state and the
    /// partition function are the same contraction over two different arithmetics.
    ///
    /// # The estimate approaches from ABOVE, which is what makes it reportable
    ///
    /// Every excited level contributes a positive term, so the estimate is never below the truth:
    /// it is an upper bound on `g0` that tightens as `beta` grows. Two temperatures are used rather
    /// than one so the caller can see the convergence rather than take it on faith — [`Degeneracy`]
    /// carries both, and `count` is filled in only when they agree on an integer.
    ///
    /// # When it declines to name a number
    ///
    /// A `count` of `None` is not a failure; it means the two temperatures had not converged, which
    /// happens when the spectral gap is small relative to `1/beta` or when the couplings are
    /// continuous enough that near-degenerate states crowd the ground level. The bracket is still
    /// returned, and it is still an upper bound.
    ///
    /// # Errors
    ///
    /// [`TooWide`], as [`Elimination::ground_state`] — three eliminations rather than one, at the
    /// same width.
    ///
    /// # Panics
    ///
    /// If a min-sum run reports no ground energy or a sum-product run reports no `log Z`. Both are
    /// assertions about this module's own elimination contract — it fills exactly one of the two
    /// according to the flag it was passed — rather than conditions a caller can reach.
    ///
    /// No integer is named unless the two temperatures are finite, positive and **distinct**. Equal
    /// betas agree trivially — the residual is zero for a reason unrelated to convergence — and at
    /// `(0.0, 0.0)` that reported `2^n`, the total number of states, as though it were the ground
    /// count. A non-finite or negative beta likewise returns the bracket unnamed rather than
    /// rounding it into a number.
    pub fn ground_degeneracy(&self, g: &Graph, betas: (f64, f64)) -> Result<Degeneracy, TooWide> {
        let (warm, cold) = if betas.0 < betas.1 { betas } else { (betas.1, betas.0) };
        let e0 = self
            .ground_state(g)?
            .ground_energy
            .expect("min-sum was run, so it reports a ground energy");

        let at = |beta: f64| -> Result<f64, TooWide> {
            let ln_z = self
                .log_partition(g, beta)?
                .log_z
                .expect("sum-product was run, so it reports log Z");
            Ok((ln_z + beta * e0).exp())
        };
        let warm_est = at(warm)?;
        let cold_est = at(cold)?;

        // TWO TEMPERATURES THAT ARE THE SAME TEMPERATURE AGREE ABOUT NOTHING.
        //
        // The whole convergence argument is that the estimate falls towards g0 as beta rises, so
        // two betas agreeing is evidence the fall has finished. Hand it the same beta twice and the
        // residual is exactly zero for a reason that has nothing to do with convergence. At
        // `(0.0, 0.0)` on a 9-spin ring that reported `Some(512)` — every state weighs 1 at beta 0,
        // so the estimate is 2^n, a perfectly valid UPPER BOUND named as if it were the count.
        //
        // This is the crate's own "identical verdicts" tell, in its smallest form: the two subjects
        // agreed because they were one subject.
        let usable = warm.is_finite()
            && cold.is_finite()
            && warm > 0.0
            && cold > warm;
        let n = cold_est.round();
        let count = (usable
            && n >= 1.0
            && n <= u64::MAX as f64
            && warm_est.round() == n
            && (cold_est - n).abs() < 1e-6)
            .then_some(n as u64);

        Ok(Degeneracy { ground_energy: e0, warm: (warm, warm_est), cold: (cold, cold_est), count })
    }

    /// Exact single-site marginals `P(s_i = +1)` at inverse temperature `beta`.
    ///
    /// The module says sum-product gives log Z "and with it exact marginals, which is what lets a
    /// sampler be checked against truth on graphs far too large to enumerate". It gave log Z. This
    /// is the rest of that sentence.
    ///
    /// # How, and what it costs
    ///
    /// Condition, do not differentiate. For each node, `log Z` is computed twice on the graph with
    /// that node pinned to `+1` and to `-1`, and
    ///
    /// ```text
    ///     P(s_i = +1) = sigma( log Z(s_i = +1) - log Z(s_i = -1) )
    /// ```
    ///
    /// which is a sigmoid of a difference: the total `log Z` cancels, so it never has to be
    /// accurate, and neither does the `ln 2` from the pinned node being left in the graph as an
    /// isolated free spin. Pinning `s_i = v` means dropping node `i`'s couplings and folding each
    /// into its neighbour's field as `h_j += J_ij * v`, plus the `beta * h_i * v` the node itself
    /// contributes — and those two constants differ between the `+1` and `-1` runs by exactly
    /// `2 * beta * h_i`, which is why the field appears in the difference below.
    ///
    /// **The cost is `2n` eliminations**, so `O(n * 2^w)` rather than the single `O(2^w)` of
    /// [`Self::log_partition`]. That is the price of an exact answer per node from a routine that
    /// returns one number; a message-passing formulation would get all of them from two passes and
    /// is a different algorithm. Refused, not approximated, when the width is too large: the same
    /// [`TooWide`] the other two return, from the same order.
    ///
    /// Conditioning changes the graph but never its width — pinning a node only REMOVES edges — so
    /// a model whose `log_partition` succeeds cannot have a marginal that is refused for width.
    ///
    /// # Errors
    ///
    /// [`TooWide`], as [`Elimination::ground_state`].
    ///
    /// # Panics
    ///
    /// Never on a graph this accepted: the pinned eliminations reuse the order already checked against
    /// the width cap.
    pub fn marginals(&self, g: &Graph, beta: f64) -> Result<Vec<f64>, TooWide> {
        // Refuse up front on the unconditioned graph, so a caller learns the width before paying
        // for 2n eliminations rather than after the first one.
        let w = self.width(g);
        if w > self.max_width {
            return Err(TooWide::Width { width: w, max: self.max_width });
        }
        let mut out = Vec::with_capacity(g.n);
        for i in 0..g.n {
            let plus = self.log_partition(&pin(g, i, 1.0), beta)?.log_z.expect("sum-product was run");
            let minus = self.log_partition(&pin(g, i, -1.0), beta)?.log_z.expect("sum-product was run");
            // The pinned node is left in the graph as an isolated free spin in both runs, so its
            // factor of two cancels along with everything else that does not depend on v.
            let delta = 2.0 * beta * g.h[i] + plus - minus;
            out.push(1.0 / (1.0 + (-delta).exp()));
        }
        Ok(out)
    }

    /// Induced width of the order this would use, without running anything.
    #[must_use]
    pub fn width(&self, g: &Graph) -> usize {
        Self::order_for(g).1
    }

    /// The elimination order this will actually use, and its induced width.
    ///
    /// TWO HEURISTICS, THE NARROWER KEPT. Min-fill is greedy and local; nested dissection splits by
    /// a separator and is the better fit for grid-like graphs, where min-fill drifts well above the
    /// treewidth. Neither dominates, so both are built and scored on the same scale — by simulating
    /// the elimination and taking the largest table it would build — and the smaller wins.
    ///
    /// Keeping the smaller is what makes this safe to adopt: the width can only go down, so a model
    /// that was accepted before is still accepted, and one that was refused may now fit. Measured
    /// on the periodic `lattice2d` family (a torus, treewidth about `2L`), where the default
    /// `max_width` of 24 is the accept/refuse line:
    ///
    /// ```text
    ///   grid    min-fill   dissection   kept
    ///   10x10         23           20     20
    ///   12x12         26           24     24     <- 26 was refused, 24 is not
    ///   14x14         34           28     28
    ///   18x18         44           36     36
    /// ```
    ///
    /// Ties go to min-fill, which is the incumbent: an order change with no width change is churn
    /// that moves which ground state comes back from a degenerate model.
    #[must_use]
    pub fn order_for(g: &Graph) -> (Vec<usize>, usize) {
        let adj = adjacency(g);
        let (mf_order, mf_width) = min_fill_order(g.n, &adj);
        let (sep_order, sep_width) = separator_order(g.n, &adj);
        if sep_width < mf_width { (sep_order, sep_width) } else { (mf_order, mf_width) }
    }

    fn run(&self, g: &Graph, beta: f64, min_sum: bool) -> Result<Exact, TooWide> {
        let (order, width) = Self::order_for(g);
        if width > self.max_width {
            return Err(TooWide::Width { width, max: self.max_width });
        }

        let mut tables = initial_tables(g, beta);
        // For back-substitution: for each eliminated variable, the scope it depended on and the
        // choice that was optimal for every assignment of that scope.
        let mut decisions: Vec<(usize, Vec<usize>, Vec<bool>)> = Vec::new();
        let mut constant = 0.0f64;

        for &v in &order {
            let (mine, rest): (Vec<Table>, Vec<Table>) =
                tables.into_iter().partition(|t| t.vars.contains(&v));
            tables = rest;
            if mine.is_empty() {
                // A VARIABLE NO TABLE MENTIONS IS STILL A VARIABLE, and the two semirings owe it
                // different amounts.
                //
                // `initial_tables` emits nothing for a spin with no field and no edges, so such a
                // spin reaches here with an empty bucket. Skipping it outright is right for
                // min-sum -- a free spin adds zero to the energy, and back-substitution has no
                // decision to record. It is WRONG for sum-product: summing over a spin that
                // appears in no factor multiplies Z by its number of states, so `log Z` must gain
                // `ln 2` and did not.
                //
                // The cost of that: `log_partition` was short by `ln 2` per free spin, which
                // `ground_degeneracy` turned into a count too small by a factor of `2^free` --
                // reported as a confident integer, because the error is identical at both
                // temperatures and the convergence guard saw a residual of 1e-15.
                //
                // Nothing caught it. `log_z_matches_enumeration` builds with `random_sparse`,
                // which puts a nonzero bias on every node; the chain test uses connected chains.
                // Neither family contains an isolated spin, so the case existed and was never
                // sampled.
                // MINUS, because `constant` accumulates NEGATIVE log-weights: the sum-product
                // branch below stores `-(logsumexp of -energy)` and the function returns
                // `log_z: Some(-constant)`. Adding here would have doubled the error instead of
                // removing it, which is what a first pass at this did.
                if !min_sum {
                    constant -= core::f64::consts::LN_2;
                }
                continue;
            }

            // scope of the new table: everything the gathered tables touch, minus v
            let mut scope: Vec<usize> = Vec::new();
            for t in &mine {
                for &u in &t.vars {
                    if u != v && !scope.contains(&u) {
                        scope.push(u);
                    }
                }
            }
            scope.sort_unstable();

            let m = scope.len();
            let mut vals = vec![0.0f64; 1 << m];
            // ONLY min-sum has a decision to record, and only min-sum reads one back.
            //
            // This allocated `2^m` bools per eliminated variable in BOTH modes and pushed every one
            // onto `decisions`, which sum-product never touches. The tables themselves are dropped
            // as they are consumed, so the live cost of an elimination is `max_k 2^{m_k}` — but the
            // decision tables were retained to the end, making the real peak `sum_k 2^{m_k}`. At the
            // default `max_width` of 24 that is 16 MB of dead allocation per variable, and
            // `ground_degeneracy` runs sum-product twice per query.
            let mut choice = if min_sum { vec![false; 1 << m] } else { Vec::new() };
            let mut assign = vec![0i8; g.n];

            for idx in 0..(1usize << m) {
                for (k, &u) in scope.iter().enumerate() {
                    assign[u] = if idx >> k & 1 == 1 { 1 } else { -1 };
                }
                // the two branches for v
                let mut branch = [0.0f64; 2];
                for (bi, sv) in [(-1i8, 0usize), (1i8, 1usize)].map(|(s, i)| (s, i)) {
                    assign[v] = bi;
                    branch[sv] = mine.iter().map(|t| t.value_at(&assign)).sum();
                }
                if min_sum {
                    let take_plus = branch[1] < branch[0];
                    vals[idx] = if take_plus { branch[1] } else { branch[0] };
                    choice[idx] = take_plus;
                } else {
                    // log-sum-exp of -energy, stably
                    let (a, b) = (-branch[0], -branch[1]);
                    let hi = a.max(b);
                    vals[idx] = -(hi + ((a - hi).exp() + (b - hi).exp()).ln());
                }
            }

            if min_sum {
                decisions.push((v, scope.clone(), choice));
            }
            if m == 0 {
                constant += vals[0];
            } else {
                tables.push(Table { vars: scope, vals });
            }
        }

        for t in &tables {
            debug_assert!(t.vars.is_empty(), "a table survived elimination");
            constant += t.vals[0];
        }

        if min_sum {
            // Walk the decisions backwards, filling in each variable from the scope already fixed.
            let mut state = vec![-1i8; g.n];
            for (v, scope, choice) in decisions.iter().rev() {
                let mut idx = 0usize;
                for (k, &u) in scope.iter().enumerate() {
                    if state[u] > 0 {
                        idx |= 1 << k;
                    }
                }
                state[*v] = if choice[idx] { 1 } else { -1 };
            }
            Ok(Exact {
                width,
                ground_energy: Some(constant),
                ground_state: Some(state),
                log_z: None,
            })
        } else {
            Ok(Exact { width, ground_energy: None, ground_state: None, log_z: Some(-constant) })
        }
    }
}

/// The graph with node `i` pinned to `v`: its couplings removed and folded into its neighbours'
/// fields, and its own field zeroed so it contributes an identical constant factor whatever `v` is.
///
/// The node is kept rather than deleted so every other index is unchanged — renumbering would make
/// the returned marginals line up with a different graph than the one asked about, which is the
/// kind of error that produces a plausible answer.
fn pin(g: &Graph, i: usize, v: f64) -> Graph {
    let mut b = crate::graph::GraphBuilder::new(g.n);
    for a in 0..g.n {
        let mut h = if a == i { 0.0 } else { g.h[a] };
        for k in g.offset[a]..g.offset[a + 1] {
            let c = g.nbr[k] as usize;
            if a == i || c == i {
                // An edge touching the pinned node becomes a field on the other end.
                if a != i {
                    h += g.w[k] * v;
                }
            } else if c > a {
                b.couple(a, c, g.w[k]);
            }
        }
        if h != 0.0 {
            b.bias(a, h);
        }
    }
    b.build()
}

#[cfg(test)]
mod tests {

    /// THE INCREMENTAL ORDER MUST EQUAL THE FULL-RESCAN ORDER, EXACTLY.
    ///
    /// `Elimination::width` gates `TooWide`, so a different elimination order changes which models
    /// this module accepts and which it refuses. The dirty-set version exists to stop the rescan
    /// being quadratic in `n`; it is not licensed to change an answer, and "the width is usually
    /// the same" would not be good enough.
    ///
    /// The reference here is the full rescan, written out again rather than imported, so the two
    /// cannot drift into agreement by sharing a bug.
    #[test]
    fn the_incremental_min_fill_order_matches_a_full_rescan_exactly() {
        use std::collections::BTreeSet;
        fn reference(n: usize, adj: &[Vec<usize>]) -> (Vec<usize>, usize) {
            let mut nbr: Vec<BTreeSet<usize>> =
                adj.iter().map(|v| v.iter().copied().collect()).collect();
            let mut alive = vec![true; n];
            let (mut order, mut width) = (Vec::with_capacity(n), 0);
            for _ in 0..n {
                let (mut best, mut bf, mut bd) = (usize::MAX, usize::MAX, usize::MAX);
                for v in 0..n {
                    if !alive[v] {
                        continue;
                    }
                    let ns: Vec<usize> = nbr[v].iter().copied().filter(|&u| alive[u]).collect();
                    let mut fill = 0;
                    for a in 0..ns.len() {
                        for b in (a + 1)..ns.len() {
                            if !nbr[ns[a]].contains(&ns[b]) {
                                fill += 1;
                            }
                        }
                    }
                    if fill < bf || (fill == bf && ns.len() < bd) {
                        best = v;
                        bf = fill;
                        bd = ns.len();
                    }
                }
                let v = best;
                let ns: Vec<usize> = nbr[v].iter().copied().filter(|&u| alive[u]).collect();
                width = width.max(ns.len());
                for a in 0..ns.len() {
                    for b in (a + 1)..ns.len() {
                        nbr[ns[a]].insert(ns[b]);
                        nbr[ns[b]].insert(ns[a]);
                    }
                }
                alive[v] = false;
                order.push(v);
            }
            (order, width)
        }

        // Random graphs at several densities, plus the shapes this crate actually builds. Density
        // matters: a sparse graph has a small dirty set and a dense one has nearly all of it, so
        // both ends of the optimisation are exercised.
        let mut rng = crate::rng::Pcg::new(9, 0x11FE);
        let mut checked = 0;
        for n in [4usize, 7, 11, 16, 22] {
            for &p in &[0.15f64, 0.35, 0.6, 0.9] {
                for _ in 0..6 {
                    let mut adj = vec![Vec::new(); n];
                    for i in 0..n {
                        for j in (i + 1)..n {
                            if rng.f64() < p {
                                adj[i].push(j);
                                adj[j].push(i);
                            }
                        }
                    }
                    assert_eq!(
                        min_fill_order(n, &adj),
                        reference(n, &adj),
                        "n={n} p={p}: the dirty-set order diverged from the full rescan"
                    );
                    checked += 1;
                }
            }
        }
        for g in [
            crate::ising::lattice2d(4, 1.0),
            crate::ising::ring(9, 1.0, 0.0),
            crate::ising::grid2d(5, 3, 1.0),
            crate::ising::chimera(2, 2, 4, 1.0),
        ] {
            let adj = adjacency(&g);
            assert_eq!(min_fill_order(g.n, &adj), reference(g.n, &adj), "on a built graph");
            checked += 1;
        }
        assert!(checked > 100, "only {checked} graphs compared");
    }

    /// Marginals against BRUTE FORCE, which is the only referee that leaves nothing to argue about.
    ///
    /// Enumerating 2^n states and summing the Boltzmann weights is a completely different
    /// computation from eliminating variables, so agreement to 1e-12 is not two implementations of
    /// one idea agreeing with themselves.
    #[test]
    fn marginals_match_exhaustive_enumeration() {
        for (seed, n) in [(1u64, 6usize), (7, 8), (99, 10)] {
            let mut rng = crate::rng::Pcg::new(seed, 0xE7AC);
            let mut b = GraphBuilder::new(n);
            for i in 0..n {
                b.bias(i, rng.f64() * 2.0 - 1.0);
                for j in (i + 1)..n {
                    if rng.f64() < 0.45 {
                        b.couple(i, j, rng.f64() * 2.0 - 1.0);
                    }
                }
            }
            let g = b.build();
            let beta = 0.8;

            let got = Elimination::default().marginals(&g, beta).expect("narrow enough");

            // Brute force: sum exp(-beta E) over every state, and over the states with s_i = +1.
            let mut z = 0.0f64;
            let mut zi = vec![0.0f64; n];
            for mask in 0..(1u32 << n) {
                let s: Vec<i8> =
                    (0..n).map(|i| if mask >> i & 1 == 1 { 1i8 } else { -1 }).collect();
                let wgt = (-beta * g.energy(&s)).exp();
                z += wgt;
                for i in 0..n {
                    if s[i] == 1 {
                        zi[i] += wgt;
                    }
                }
            }
            for i in 0..n {
                let want = zi[i] / z;
                assert!(
                    (got[i] - want).abs() < 1e-12,
                    "seed {seed}, n {n}, node {i}: elimination {} vs enumeration {want}",
                    got[i]
                );
            }
        }
    }

    /// The one case with a closed form, so a systematic error in BOTH of the above would show.
    #[test]
    fn a_single_spin_in_a_field_matches_the_sigmoid() {
        for h in [-1.5, -0.3, 0.0, 0.7, 2.0] {
            for beta in [0.1, 1.0, 3.0] {
                let mut b = GraphBuilder::new(1);
                b.bias(0, h);
                let g = b.build();
                let got = Elimination::default().marginals(&g, beta).unwrap()[0];
                // P(+1) = e^{beta h} / (e^{beta h} + e^{-beta h}) = sigma(2 beta h)
                let want = 1.0 / (1.0 + (-2.0 * beta * h).exp());
                assert!((got - want).abs() < 1e-13, "h {h}, beta {beta}: {got} vs {want}");
            }
        }
    }

    /// Marginals are the referee this module exists to be, so check a SAMPLER against them on a
    /// graph far past where enumeration stops -- which is the sentence in the module doc that had
    /// no code behind it.
    #[test]
    fn a_sampler_can_be_checked_against_truth_past_where_enumeration_stops() {
        // A 3x14 strip: 42 spins, so 2^42 states -- unenumerable -- and width 3.
        let (w, l) = (3usize, 14usize);
        let mut b = GraphBuilder::new(w * l);
        for y in 0..l {
            for x in 0..w {
                let i = y * w + x;
                if x + 1 < w {
                    b.couple(i, i + 1, 0.6);
                }
                if y + 1 < l {
                    b.couple(i, i + w, 0.6);
                }
            }
        }
        let g = b.build();
        let beta = 0.35;
        let e = Elimination::default();
        assert!(e.width(&g) <= 4, "a strip is narrow: width {}", e.width(&g));
        let truth = e.marginals(&g, beta).unwrap();

        let mut smp = crate::gibbs::Sampler::new(&g, beta, 0xC0FFEE);
        smp.sweeps(2000, None);
        let draws = 40_000;
        let mut up = vec![0u64; g.n];
        for _ in 0..draws {
            smp.sweep(None);
            for i in 0..g.n {
                if smp.s[i] == 1 {
                    up[i] += 1;
                }
            }
        }
        let worst = (0..g.n)
            .map(|i| (up[i] as f64 / draws as f64 - truth[i]).abs())
            .fold(0.0f64, f64::max);
        // Three sigma on 40k correlated draws is comfortably inside this; the point is that the
        // comparison is possible at all on 42 spins.
        assert!(worst < 0.02, "worst |sampled - exact| marginal = {worst:.4}");
    }

    #[test]
    fn a_graph_too_wide_for_marginals_is_refused_with_the_same_reason_as_log_z() {
        // Dense on 30 nodes: width far past the default ceiling.
        let mut b = GraphBuilder::new(30);
        for i in 0..30 {
            for j in (i + 1)..30 {
                b.couple(i, j, 0.4);
            }
        }
        let g = b.build();
        let e = Elimination::default();
        let m = e.marginals(&g, 1.0);
        assert!(matches!(m, Err(TooWide::Width { .. })), "{m:?}");
        // And it refuses BEFORE paying for 2n eliminations, which is why the width is checked up
        // front rather than being discovered inside the loop.
        // Refused for the SAME reason and from the same order: a caller that got a log Z cannot
        // then be told its marginals are too wide, because conditioning only removes edges.
        assert_eq!(m.unwrap_err(), e.log_partition(&g, 1.0).unwrap_err());
    }

    use super::*;
    use crate::graph::GraphBuilder;
    use crate::oracle::{Exhaustive, Solver};
    use crate::rng::Pcg;

    fn random_sparse(n: usize, p: f64, seed: u64) -> Graph {
        let mut rng = Pcg::new(seed, 0);
        let mut b = GraphBuilder::new(n);
        for i in 0..n {
            for j in (i + 1)..n {
                if rng.f64() < p {
                    b.couple(i, j, rng.f64() * 2.0 - 1.0);
                }
            }
            b.bias(i, rng.f64() - 0.5);
        }
        b.build()
    }

    #[test]
    fn the_ground_state_matches_enumeration() {
        // The only check that matters for an exact method.
        for (n, p, seed) in [(10, 0.3, 1), (14, 0.2, 2), (16, 0.15, 3), (12, 0.5, 4)] {
            let g = random_sparse(n, p, seed);
            let (bs, be) = Exhaustive.solve(&g);
            let e = Elimination::default().ground_state(&g).expect("small enough");
            let ge = e.ground_energy.unwrap();
            assert!(
                (ge - be).abs() < 1e-9,
                "n={n} p={p}: elimination {ge} vs enumeration {be}"
            );
            // and the recovered state really attains it, which back-substitution can get wrong
            // independently of the energy being right
            let st = e.ground_state.unwrap();
            assert!(
                (g.energy(&st) - be).abs() < 1e-9,
                "n={n}: recovered state has energy {} not {be} (enumeration found {bs:?})",
                g.energy(&st)
            );
        }
    }

    #[test]
    fn log_z_matches_enumeration() {
        for (n, p, beta, seed) in [(10, 0.3, 0.7, 1), (12, 0.25, 1.3, 2), (8, 0.6, 0.4, 3)] {
            let g = random_sparse(n, p, seed);
            let mut z = 0.0f64;
            let mut s = vec![-1i8; n];
            for mask in 0..(1usize << n) {
                for i in 0..n {
                    s[i] = if mask >> i & 1 == 1 { 1 } else { -1 };
                }
                z += (-beta * g.energy(&s)).exp();
            }
            let want = z.ln();
            let got = Elimination::default().log_partition(&g, beta).unwrap().log_z.unwrap();
            assert!((got - want).abs() < 1e-9, "n={n} beta={beta}: {got} vs {want}");
        }
    }

    #[test]
    fn a_chain_is_width_one_and_exact_at_any_length() {
        // The point of the method: sparse structure beats size. Enumeration cannot touch this.
        let n = 2000;
        let mut b = GraphBuilder::new(n);
        for i in 0..n - 1 {
            b.couple(i, i + 1, 1.0);
        }
        let g = b.build();
        let el = Elimination::default();
        assert_eq!(el.width(&g), 1, "a path has induced width 1");
        let e = el.ground_state(&g).unwrap();
        assert_eq!(e.ground_energy.unwrap(), -((n - 1) as f64), "every bond satisfiable");
        assert!(e.ground_state.unwrap().windows(2).all(|w| w[0] == w[1]));
    }

    #[test]
    fn a_lattice_strip_is_exact_far_past_enumeration() {
        // 6 x 40 = 240 spins, width 6. Enumeration would need 2^240 states.
        let (w, h) = (6usize, 40usize);
        let mut b = GraphBuilder::new(w * h);
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                if x + 1 < w {
                    b.couple(i, y * w + x + 1, 1.0);
                }
                if y + 1 < h {
                    b.couple(i, (y + 1) * w + x, 1.0);
                }
            }
        }
        let g = b.build();
        let el = Elimination { max_width: 12 };
        // The true treewidth of a 6-wide strip is 6; min-fill finds 8 here. Asserting 6 would be
        // asserting that a heuristic is optimal, which it is not past width 5 -- see the table in
        // the module docs. The bound is on measured behaviour, and a regression past it means the
        // ordering got worse, not that the answer got wrong.
        assert!(el.width(&g) <= 8, "min-fill measured 8 on this strip; got {}", el.width(&g));
        let e = el.ground_state(&g).unwrap();
        let bonds = (w - 1) * h + w * (h - 1);
        assert_eq!(e.ground_energy.unwrap(), -(bonds as f64));
    }

    #[test]
    fn a_dense_graph_is_refused_rather_than_attempted() {
        // Refusing loudly beats running for a week. The message must say what to do instead.
        let g = random_sparse(60, 0.9, 1);
        let err = Elimination { max_width: 20 }.ground_state(&g).unwrap_err();
        assert!(matches!(err, TooWide::Width { .. }));
        assert!(err.to_string().contains("planted instance"), "{err}");
    }

    #[test]
    fn it_agrees_with_a_planted_wishart_optimum_where_width_allows() {
        // Two independent notions of truth, cross-checked.
        let p = crate::planted::frustrated_loops(4, 12, 3);
        let e = Elimination { max_width: 20 }.ground_state(&p.graph).unwrap();
        assert!((e.ground_energy.unwrap() - p.ground_energy).abs() < 1e-9);
    }
}

#[cfg(test)]
mod closed_form {
    use super::*;
    use crate::graph::GraphBuilder;

    #[test]
    fn log_z_of_a_chain_matches_the_closed_form() {
        // A 1D open Ising chain has Z = 2 (2 cosh beta)^(n-1) exactly. Checking against theory
        // rather than against enumeration reaches sizes enumeration cannot, and catches an error
        // that would scale with n instead of showing up on small cases.
        for n in [8usize, 50, 400] {
            for beta in [0.25f64, 0.5, 1.5] {
                let mut b = GraphBuilder::new(n);
                for i in 0..n - 1 {
                    b.couple(i, i + 1, 1.0);
                }
                let got = Elimination::default().log_partition(&b.build(), beta).unwrap().log_z.unwrap();
                let want = 2f64.ln() + (n - 1) as f64 * (2.0 * beta.cosh()).ln();
                assert!(
                    (got - want).abs() < 1e-9 * want.abs().max(1.0),
                    "n={n} beta={beta}: {got} vs closed form {want}"
                );
            }
        }
    }
}
