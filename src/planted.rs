//! Instances whose ground state is known because it was put there.
//!
//! Enumeration stops at about twenty-five spins. Past that, "did the solver find the optimum?"
//! becomes unanswerable — which is exactly the size range where every interesting claim in this
//! field is made, and exactly why so many of those claims compare against whatever the last paper
//! reported rather than against truth.
//!
//! A planted instance sidesteps this. The ground state is chosen first and the couplings are built
//! around it, so the optimum is known at any size, for free, with certainty.
//!
//! # Why frustrated loops work
//!
//! Pick a planted state `sigma`. Take any cycle in the graph and set `J_ij = sigma_i sigma_j` on
//! every edge but one, and `J_ij = -sigma_i sigma_j` on the last. Two things follow.
//!
//! The cycle is **frustrated**: the product of couplings around it is negative, because each
//! `sigma` appears exactly twice and the single sign flip survives. So no state satisfies every
//! bond, and for a cycle of length `L` the best any state can do is `-(L-2)`.
//!
//! The planted state **achieves** that: it satisfies the `L-1` ordinary bonds for `-(L-1)`, and
//! pays `+1` on the frustrated one.
//!
//! Since the total energy is the sum of the per-cycle energies, and `sigma` attains each cycle's
//! own minimum simultaneously, `sigma` is a global ground state. Overlapping cycles are fine: the
//! couplings add, and the argument is unaffected because it is a statement about sums.
//!
//! The tests do not take that argument on trust — they enumerate small instances and check.
//!
//! # Difficulty is not monotonic: there is a peak
//!
//! Measured on an 8x8 periodic lattice over a grid of instance and solver seeds, greedy descent
//! with 50 restarts solves (see `examples/planted_probe.rs`):
//!
//! | loops | hits per edge | greedy solved exactly | worst excess |
//! |---|---|---|---|
//! | 16 | 0.5 | 16/16 | 0 |
//! | 64 | 2 | 7/16 | 0.094 |
//! | **128** | **4** | **4/16** | **0.125** |
//! | 256 | 8 | 11/16 | 0.172 |
//! | 512 | 16 | 16/16 | 0 |
//!
//! An easy-hard-easy transition. Too few loops and there are barely any competing constraints; too
//! many and the accumulated couplings on each edge concentrate toward their mean, which is set by
//! `sigma`, so the instance relaxes back into a gauged ferromagnet. The interesting region is in
//! between, around four planted loops per edge.
//!
//! Both of the obvious guesses about this are wrong and were made here in turn: difficulty does not
//! rise with density, and the family is not uniformly easy. The second error came from a probe that
//! averaged over matched seeds, which hid the peak completely — the shape only appears when the
//! solve *rate* is measured across a seed grid rather than the mean excess.
//!
//! One caveat on what "hard" means. Ground states of a two dimensional spin glass in no external
//! field are computable in polynomial time by minimum-weight perfect matching, so nothing here is
//! hard in the complexity sense. It is a benchmark for *heuristics* — greedy and annealing really do
//! fail at the peak — and should be described that way rather than as evidence of intractability.

use crate::graph::{Graph, GraphBuilder};
use crate::rng::Pcg;

/// An instance with its optimum known by construction.
pub struct Planted {
    pub graph: Graph,
    pub ground_state: Vec<i8>,
    pub ground_energy: f64,
    /// How many frustrated cycles were planted. Higher is harder, up to a point.
    pub loops: usize,
}

impl Planted {
    /// How far a state is above the true optimum, as a fraction of it. Zero means solved.
    ///
    /// This is the number a benchmark should report, and it is only computable because the optimum
    /// is known rather than assumed to be the best anyone has found.
    pub fn excess(&self, s: &[i8]) -> f64 {
        let e = self.graph.energy(s);
        if self.ground_energy.abs() < 1e-12 {
            return (e - self.ground_energy).abs();
        }
        (e - self.ground_energy) / self.ground_energy.abs()
    }

    /// Whether a state reaches the planted optimum.
    pub fn solved(&self, s: &[i8]) -> bool {
        self.graph.energy(s) <= self.ground_energy + 1e-9
    }
}

/// Plant frustrated plaquettes on an `l` by `l` periodic lattice.
///
/// Each planted loop is an elementary 4-cycle with one bond reversed, contributing exactly `-2` to
/// the ground energy. `loops` controls frustration density: more loops on the same lattice means
/// more competing constraints and a harder instance.
pub fn frustrated_loops(l: usize, loops: usize, seed: u64) -> Planted {
    assert!(l >= 3, "a periodic lattice smaller than 3x3 has degenerate plaquettes");
    assert!(loops >= 1, "an instance with no frustration is a ferromagnet");
    let n = l * l;
    let mut rng = Pcg::new(seed, 0);

    // The planted state, chosen before any coupling exists.
    let sigma: Vec<i8> = (0..n).map(|_| if rng.f64() < 0.5 { 1 } else { -1 }).collect();

    let at = |x: usize, y: usize| (y % l) * l + (x % l);
    let mut b = GraphBuilder::new(n);

    for _ in 0..loops {
        let (x, y) = ((rng.f64() * l as f64) as usize % l, (rng.f64() * l as f64) as usize % l);
        // the four corners of one plaquette, in cycle order
        let c = [at(x, y), at(x + 1, y), at(x + 1, y + 1), at(x, y + 1)];
        let broken = (rng.f64() * 4.0) as usize % 4;
        for k in 0..4 {
            let (i, j) = (c[k], c[(k + 1) % 4]);
            let want = sigma[i] as f64 * sigma[j] as f64;
            b.couple(i, j, if k == broken { -want } else { want });
        }
    }

    Planted {
        graph: b.build(),
        ground_state: sigma,
        // every plaquette contributes -(L-2) = -2 at its own minimum, attained simultaneously
        ground_energy: -2.0 * loops as f64,
        loops,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::oracle::{Exhaustive, RandomGuess, Solver, SteepestDescent};

    #[test]
    fn the_planted_state_really_is_the_ground_state() {
        // The construction's whole claim, checked by enumeration rather than by the argument in the
        // module documentation. If these ever disagree, the documentation is wrong.
        for l in 3..=4 {
            for loops in [1, 3, 8, 20] {
                for seed in 1..=3 {
                    let p = frustrated_loops(l, loops, seed);
                    let (_s, exact) = Exhaustive.solve(&p.graph);
                    let planted = p.graph.energy(&p.ground_state);
                    assert!(
                        (planted - exact).abs() < 1e-9,
                        "l={l} loops={loops} seed={seed}: planted {planted} but true optimum {exact}"
                    );
                    assert!(
                        (planted - p.ground_energy).abs() < 1e-9,
                        "l={l} loops={loops}: predicted {} but measured {planted}",
                        p.ground_energy
                    );
                }
            }
        }
    }

    #[test]
    fn the_predicted_ground_energy_is_exactly_minus_two_per_loop() {
        for loops in [1, 5, 50, 500] {
            let p = frustrated_loops(8, loops, 11);
            assert_eq!(p.ground_energy, -2.0 * loops as f64);
            assert!((p.graph.energy(&p.ground_state) - p.ground_energy).abs() < 1e-9);
        }
    }

    #[test]
    fn it_is_frustrated_rather_than_merely_disguised() {
        // A gauge-transformed ferromagnet also has a known ground state and is trivial to solve.
        // These instances must actually leave bonds broken.
        let p = frustrated_loops(6, 60, 5);
        let g = &p.graph;
        let mut total = 0.0;
        for i in 0..g.n {
            for k in g.offset[i]..g.offset[i + 1] {
                if g.nbr[k] as usize > i {
                    total += g.w[k].abs();
                }
            }
        }
        // an unfrustrated model would reach -total; frustration means the optimum is strictly above
        assert!(
            p.ground_energy > -total + 1e-9,
            "ground energy {} vs unfrustrated bound {}",
            p.ground_energy,
            -total
        );
    }

    #[test]
    fn instances_are_reproducible_and_seeds_differ() {
        let a = frustrated_loops(6, 30, 42);
        let b = frustrated_loops(6, 30, 42);
        let c = frustrated_loops(6, 30, 43);
        assert_eq!(a.ground_state, b.ground_state, "same seed must reproduce");
        assert_ne!(a.ground_state, c.ground_state, "different seeds must differ");
    }

    #[test]
    fn the_noise_oracle_never_solves_one() {
        // The discriminating check. A planted instance that random guessing solves is not measuring
        // anything, and at these sizes it must not come close.
        for l in [6, 10] {
            let p = frustrated_loops(l, l * l * 2, 9);
            let (s, _) = RandomGuess { tries: 20_000, seed: 2 }.solve(&p.graph);
            assert!(!p.solved(&s), "random guessing solved a planted instance at l={l}");
            assert!(p.excess(&s) > 0.2, "noise should be far off, was {}", p.excess(&s));
        }
    }

    #[test]
    fn a_real_method_gets_much_closer_than_noise() {
        // The instance has to be solvable in principle, or it measures nothing either.
        let p = frustrated_loops(8, 128, 4);
        let noise = p.excess(&RandomGuess { tries: 20_000, seed: 2 }.solve(&p.graph).0);
        let greedy = p.excess(&SteepestDescent { restarts: 200, seed: 2 }.solve(&p.graph).0);
        let annealed = p.excess(
            &crate::oracle::Annealer {
                schedule: crate::schedule::Schedule::geometric(0.05, 6.0, 120, 40),
                seed: 2,
            }
            .solve(&p.graph)
            .0,
        );
        assert!(greedy < noise, "greedy {greedy} vs noise {noise}");
        assert!(annealed < noise);
        assert!(greedy < 0.20, "greedy should stay within 20% even near the hard peak, was {greedy}");
        assert!(annealed < 0.10, "annealing should land within 10% of the optimum, got {annealed}");
    }

    #[test]
    fn difficulty_peaks_in_the_middle_of_the_density_range() {
        // The measured shape, pinned. Two earlier versions of this test asserted guesses instead:
        // first that denser is harder, then that the whole family is easy. Both were wrong, and the
        // second came from a probe that averaged over matched seeds and hid the peak. The solve
        // RATE across a seed grid is what shows it.
        let rate = |loops: usize| {
            let (mut solved, mut total) = (0, 0);
            for iseed in 1..=4u64 {
                let p = frustrated_loops(8, loops, iseed);
                for sseed in 1..=4u64 {
                    let (s, _) = SteepestDescent { restarts: 50, seed: sseed }.solve(&p.graph);
                    total += 1;
                    if p.solved(&s) {
                        solved += 1;
                    }
                }
            }
            solved as f64 / total as f64
        };

        let sparse = rate(16);
        let peak = rate(128);
        let dense = rate(512);
        assert_eq!(sparse, 1.0, "a sparsely frustrated instance should always be solved");
        assert_eq!(dense, 1.0, "a saturated instance relaxes to a gauged ferromagnet");
        assert!(peak < 0.5, "the peak should defeat greedy more often than not, got {peak}");
        assert!(peak < sparse && peak < dense, "difficulty must peak in the middle");
    }
}

// ---- the Wishart planted ensemble ---------------------------------------------------------------

/// Standard normal draw, Box–Muller.
fn gauss(rng: &mut Pcg) -> f64 {
    let u = rng.f64().max(1e-15);
    let v = rng.f64();
    (-2.0 * u.ln()).sqrt() * (core::f64::consts::TAU * v).cos()
}

/// The Wishart planted ensemble: dense, tunably rugged, and genuinely hard.
///
/// Frustrated loops on a lattice are a correctness oracle but a mild benchmark — see the module
/// notes. This family is the answer to that, and it is hard for a structural reason rather than by
/// accident.
///
/// Draw `m = alpha * n` column vectors, each Gaussian but **projected orthogonal to the planted
/// state** `t`, and stack them as `W`. Set `J_ij = -(1/n) (W Wᵀ)_ij` off the diagonal. Then
///
/// ```text
///     E(s) = (1/2n) [ ‖Wᵀ s‖² − tr(W Wᵀ) ]
/// ```
///
/// The trace is a constant, so minimising the energy means minimising `‖Wᵀ s‖²`, a non-negative
/// quantity that `t` drives to exactly zero. The planted state is therefore a ground state by
/// construction, and the ground energy is `−tr(W Wᵀ) / 2n`, known in closed form.
///
/// `alpha` is the ruggedness knob, and unlike the lattice family it is **monotonic**. Measured at
/// `n = 24` with greedy descent over a 4x4 instance/solver seed grid:
///
/// | alpha | greedy solved | worst excess |
/// |---|---|---|
/// | 0.2 | 4/16 | 0.010 |
/// | 0.3 | 5/16 | 0.014 |
/// | 0.5 | 10/16 | 0.052 |
/// | 0.75 | 15/16 | 0.080 |
/// | 1.0 and above | 16/16 | 0 |
///
/// Hard below about `alpha = 1`, as the published ensemble reports.
///
/// Note the failure *signature*, which differs from the frustrated-loop family and is the more
/// useful half of this measurement. There, a miss could be 17% above the optimum. Here a miss is
/// under 2% at the hardest setting: the landscape is dense with near-degenerate minima, so a solver
/// gets very close and still misses. Any benchmark reporting mean excess would call this family
/// easy. Report the solve rate.
pub fn wishart(n: usize, alpha: f64, seed: u64) -> Planted {
    assert!(n >= 3, "a Wishart instance needs at least 3 spins");
    assert!(alpha > 0.0, "alpha must be positive");
    let m = ((alpha * n as f64).round() as usize).max(1);
    let mut rng = Pcg::new(seed, 0);

    let t: Vec<i8> = (0..n).map(|_| if rng.f64() < 0.5 { 1 } else { -1 }).collect();

    // Columns of W, each orthogonal to the planted state.
    let mut w = vec![0.0f64; n * m];
    for c in 0..m {
        let mut col: Vec<f64> = (0..n).map(|_| gauss(&mut rng)).collect();
        let dot: f64 = col.iter().zip(&t).map(|(x, &s)| x * s as f64).sum();
        for (i, x) in col.iter_mut().enumerate() {
            *x -= dot * t[i] as f64 / n as f64; // project out t
        }
        for i in 0..n {
            w[i * m + c] = col[i];
        }
    }

    // J = -(1/n) W Wᵀ off the diagonal; the diagonal is the constant trace term.
    let mut b = GraphBuilder::new(n);
    let mut trace = 0.0;
    for i in 0..n {
        for j in i..n {
            let mut dot = 0.0;
            for c in 0..m {
                dot += w[i * m + c] * w[j * m + c];
            }
            if i == j {
                trace += dot;
            } else {
                b.couple(i, j, -dot / n as f64);
            }
        }
    }

    Planted {
        graph: b.build(),
        ground_state: t,
        ground_energy: -trace / (2.0 * n as f64),
        loops: m,
    }
}

#[cfg(test)]
mod wishart_tests {
    use super::*;
    use crate::oracle::{Exhaustive, RandomGuess, Solver, SteepestDescent};

    #[test]
    fn the_planted_state_is_the_ground_state() {
        // Checked against enumeration, not against the derivation.
        for n in [8, 12, 16] {
            for alpha in [0.5, 1.0, 2.0] {
                for seed in 1..=3u64 {
                    let p = wishart(n, alpha, seed);
                    let (_s, exact) = Exhaustive.solve(&p.graph);
                    let planted = p.graph.energy(&p.ground_state);
                    assert!(
                        (planted - exact).abs() < 1e-7,
                        "n={n} alpha={alpha} seed={seed}: planted {planted} vs true {exact}"
                    );
                }
            }
        }
    }

    #[test]
    fn the_closed_form_ground_energy_is_right() {
        // -tr(W Wᵀ)/2n, predicted before the graph is built and measured after.
        for n in [10, 30, 80] {
            for alpha in [0.3, 1.5] {
                let p = wishart(n, alpha, 7);
                let measured = p.graph.energy(&p.ground_state);
                assert!(
                    (measured - p.ground_energy).abs() / p.ground_energy.abs() < 1e-9,
                    "n={n} alpha={alpha}: predicted {} measured {measured}",
                    p.ground_energy
                );
            }
        }
    }

    #[test]
    fn it_is_dense_where_the_lattice_family_is_sparse() {
        let p = wishart(40, 1.0, 1);
        assert_eq!(p.graph.max_degree(), 39, "every spin should couple to every other");
    }

    #[test]
    fn low_alpha_really_is_harder() {
        // Measured as a solve rate over a seed grid rather than a single run -- the lesson from the
        // frustrated-loop family, which hid its own structure behind an average.
        let rate = |alpha: f64| {
            let (mut solved, mut total) = (0, 0);
            for iseed in 1..=4u64 {
                let p = wishart(24, alpha, iseed);
                for sseed in 1..=4u64 {
                    let (s, _) = SteepestDescent { restarts: 100, seed: sseed }.solve(&p.graph);
                    total += 1;
                    if p.solved(&s) {
                        solved += 1;
                    }
                }
            }
            solved as f64 / total as f64
        };
        let hard = rate(0.3);
        let easy = rate(3.0);
        assert!(
            hard < easy,
            "alpha 0.3 should defeat greedy more often than alpha 3.0: {hard} vs {easy}"
        );
    }

    #[test]
    fn noise_never_solves_one() {
        let p = wishart(40, 0.5, 3);
        let (s, _) = RandomGuess { tries: 50_000, seed: 1 }.solve(&p.graph);
        assert!(!p.solved(&s));
        assert!(p.excess(&s) > 0.1, "noise was only {} off", p.excess(&s));
    }
}
