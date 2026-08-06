//! Things that find ground states, including one that must never work.
//!
//! Before you can say a sampler is good you need something to be good *against*. This module is
//! that set: an exact solver for models small enough to enumerate, a cheap greedy baseline that any
//! real method must beat, and — most importantly — a solver that does nothing at all.
//!
//! **The noise oracle earns its place by failing.** [`RandomGuess`] returns uniform random states.
//! Every quality test in this crate is run against it, and any test it passes is a test that
//! measures nothing. It is easy to build a benchmark that all methods pass; the way to find out
//! whether a benchmark discriminates is to hand it something that should be rejected and check that
//! it is.
//!
//! A baseline that has not been tuned is a fabricated win, so [`SteepestDescent`] takes restarts:
//! comparing against greedy-from-one-start flatters everything.

use crate::graph::Graph;
use crate::rng::Pcg;

/// Something that proposes a low-energy state.
pub trait Solver {
    fn name(&self) -> &str;
    /// Returns the best state found and its energy.
    fn solve(&self, g: &Graph) -> (Vec<i8>, f64);
}

/// Enumerate every configuration. Exact, and hopeless past about 24 spins.
pub struct Exhaustive;

impl Exhaustive {
    /// The largest model this will attempt. 2^26 states is already several seconds.
    pub const MAX_SPINS: usize = 26;
}

impl Solver for Exhaustive {
    fn name(&self) -> &str {
        "exhaustive"
    }
    fn solve(&self, g: &Graph) -> (Vec<i8>, f64) {
        assert!(
            g.n <= Self::MAX_SPINS,
            "exhaustive search over {} spins is 2^{} states; use a planted instance instead",
            g.n,
            g.n
        );
        let mut best = vec![-1i8; g.n];
        let mut best_e = f64::INFINITY;
        let mut s = vec![-1i8; g.n];
        for mask in 0..(1u64 << g.n) {
            for i in 0..g.n {
                s[i] = if mask >> i & 1 == 1 { 1 } else { -1 };
            }
            let e = g.energy(&s);
            if e < best_e {
                best_e = e;
                best.copy_from_slice(&s);
            }
        }
        (best, best_e)
    }
}

/// Greedy single-spin descent from random starts.
///
/// The honest cheap baseline. Flips whichever single spin lowers the energy most, until none does,
/// then restarts. Anything claiming to be a solver must beat a *tuned* version of this, which is
/// why the restart count is a parameter rather than a hidden 1.
pub struct SteepestDescent {
    pub restarts: usize,
    pub seed: u64,
}

impl Solver for SteepestDescent {
    fn name(&self) -> &str {
        "steepest-descent"
    }
    fn solve(&self, g: &Graph) -> (Vec<i8>, f64) {
        let mut rng = Pcg::new(self.seed, 0);
        let mut best = vec![1i8; g.n];
        let mut best_e = f64::INFINITY;
        for _ in 0..self.restarts.max(1) {
            let mut s: Vec<i8> = (0..g.n).map(|_| if rng.f64() < 0.5 { 1 } else { -1 }).collect();
            loop {
                // flipping i changes the energy by 2 * f_i * s_i
                let mut best_gain = 0.0;
                let mut pick = usize::MAX;
                for i in 0..g.n {
                    let d = crate::kernel::delta_e(g.field(i, &s), s[i]);
                    if d < best_gain - 1e-12 {
                        best_gain = d;
                        pick = i;
                    }
                }
                if pick == usize::MAX {
                    break;
                }
                s[pick] = -s[pick];
            }
            let e = g.energy(&s);
            if e < best_e {
                best_e = e;
                best = s;
            }
        }
        (best, best_e)
    }
}

/// Uniform random states. Exists to fail.
///
/// If a benchmark cannot tell this apart from a real solver, the benchmark is measuring nothing.
pub struct RandomGuess {
    pub tries: usize,
    pub seed: u64,
}

impl Solver for RandomGuess {
    fn name(&self) -> &str {
        "random-guess"
    }
    fn solve(&self, g: &Graph) -> (Vec<i8>, f64) {
        let mut rng = Pcg::new(self.seed, 1);
        let mut best = vec![1i8; g.n];
        let mut best_e = f64::INFINITY;
        for _ in 0..self.tries.max(1) {
            let s: Vec<i8> = (0..g.n).map(|_| if rng.f64() < 0.5 { 1 } else { -1 }).collect();
            let e = g.energy(&s);
            if e < best_e {
                best_e = e;
                best = s;
            }
        }
        (best, best_e)
    }
}

/// Simulated annealing behind the same trait, so it is compared on equal terms.
pub struct Annealer {
    pub schedule: crate::schedule::Schedule,
    pub seed: u64,
}

impl Solver for Annealer {
    fn name(&self) -> &str {
        "anneal"
    }
    fn solve(&self, g: &Graph) -> (Vec<i8>, f64) {
        crate::tempering::anneal_scheduled(g, &self.schedule, self.seed, None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schedule::Schedule;

    #[test]
    fn exhaustive_finds_the_ferromagnetic_ground_state() {
        let g = crate::ising::lattice2d(4, 1.0);
        let (_s, e) = Exhaustive.solve(&g);
        // 16 sites, 2 bonds each, all satisfiable
        assert_eq!(e, -32.0);
    }

    #[test]
    fn exhaustive_finds_the_frustrated_optimum() {
        let mut b = crate::graph::GraphBuilder::new(5);
        for i in 0..5 {
            b.couple(i, (i + 1) % 5, -1.0);
        }
        let (_s, e) = Exhaustive.solve(&b.build());
        assert_eq!(e, -3.0, "an odd antiferromagnetic ring cannot do better than -3");
    }

    #[test]
    fn the_noise_oracle_loses_to_everything() {
        // The point of it. If this ever ties a real solver, the instance is not discriminating.
        let g = crate::ising::lattice2d(8, 1.0);
        let noise = RandomGuess { tries: 2000, seed: 1 }.solve(&g).1;
        let greedy = SteepestDescent { restarts: 20, seed: 1 }.solve(&g).1;
        let annealed = Annealer { schedule: Schedule::geometric(0.05, 4.0, 60, 20), seed: 1 }
            .solve(&g)
            .1;
        assert!(greedy < noise, "greedy {greedy} should beat noise {noise}");
        assert!(annealed <= greedy, "annealing {annealed} should be at least greedy {greedy}");
        assert!(annealed < noise);
    }

    #[test]
    fn greedy_really_is_a_local_optimum() {
        // If a single flip still improves it, the descent stopped early and the baseline is weaker
        // than advertised -- which would flatter everything measured against it.
        let g = crate::ising::lattice2d(6, 1.0);
        let (s, _e) = SteepestDescent { restarts: 5, seed: 3 }.solve(&g);
        for i in 0..g.n {
            assert!(
                crate::kernel::delta_e(g.field(i, &s), s[i]) >= -1e-12,
                "spin {i} still improves the energy"
            );
        }
    }

    #[test]
    fn restarts_are_a_real_knob() {
        // Documenting that the baseline can be tuned, so "we beat greedy" has to say which greedy.
        let g = crate::planted::frustrated_loops(6, 40, 7);
        let one = SteepestDescent { restarts: 1, seed: 5 }.solve(&g.graph).1;
        let many = SteepestDescent { restarts: 200, seed: 5 }.solve(&g.graph).1;
        assert!(many <= one, "more restarts must not do worse: {many} vs {one}");
    }
}
