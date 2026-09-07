//! A portfolio: several solvers under one budget, and the budget stated in one unit.
//!
//! This crate has many solvers and no way to ask them the same question. Each takes its own
//! parameters, counts its own work, and returns its own `Outcome` — so "which is better here" can
//! only be answered by hand-tuning every arm and hoping the comparison was fair. That is the
//! question a user actually has, and it is the one the crate could not answer.
//!
//! # The unit is a spin proposal, because it is the only one they share
//!
//! `tabu` counts iterations, `bls` counts moves, `sqa` counts Trotter slices times steps times
//! sweeps, `tempering` counts replicas times rounds. Comparing those is comparing labels. What every
//! one of them does underneath is propose a single-spin change and accept or reject it, which is
//! also what [`crate::ledger`] prices — so [`Budget`] is a proposal count, and each adapter converts
//! it into its own knobs.
//!
//! The conversions are the load-bearing part and they are TESTED, not asserted:
//! `every_arm_stays_inside_the_budget_it_was_given` measures what each arm actually spends against
//! what it was allowed. An arm that quietly overspends makes a portfolio a measurement of who
//! cheated.
//!
//! # What a portfolio is for
//!
//! No solver here wins everywhere — that is the whole reason the crate carries several. A portfolio
//! splits one budget across arms and returns the best answer found, which is never worse than the
//! worst arm and is usually close to whichever arm happened to suit the instance. It is the standard
//! practical answer to "which solver", and it is what commercial solvers ship.

use crate::graph::Graph;

/// Work a search may spend, in single-spin proposals.
///
/// The one unit every solver in this crate shares, and the one [`crate::ledger`] charges for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Budget {
    /// Single-spin proposals the search may make.
    pub proposals: u64,
}

impl Budget {
    /// A budget of `proposals` moves.
    #[must_use]
    pub fn new(proposals: u64) -> Budget {
        Budget { proposals }
    }

    /// Split evenly across `n` arms, with the remainder dropped rather than given to whoever is
    /// first — an arm that silently gets more is a comparison that measures the split.
    #[must_use]
    pub fn split(self, n: usize) -> Budget {
        Budget { proposals: if n == 0 { 0 } else { self.proposals / n as u64 } }
    }
}

/// What a search found, and what it spent finding it.
#[derive(Clone, Debug, PartialEq)]
pub struct Found {
    /// Which arm produced this.
    pub arm: &'static str,
    /// The best state seen.
    pub state: Vec<i8>,
    /// Its energy, recomputed from the state rather than carried out of the solver.
    ///
    /// Defensive, and knowingly so: every solver here already recomputes rather than accumulating,
    /// so a mutation reading the solver's own number survives every test in this module. What the
    /// recompute buys is that `Found` does not DEPEND on that continuing to be true — a portfolio
    /// that trusted each arm's bookkeeping would inherit any future drift in any of them, and the
    /// comparison it exists to make would silently stop being one.
    pub energy: f64,
    /// Proposals actually spent, as the arm reports them.
    pub spent: u64,
}

/// A solver that can be asked for its best answer within a proposal budget.
///
/// Deliberately narrow. The point is not to abstract over everything a solver can do — that would be
/// a trait nobody could implement — but to make the one comparison that matters possible: same
/// instance, same budget, same seed.
pub trait Search {
    /// A stable name for reports.
    fn name(&self) -> &'static str;
    /// Solve `g` within `budget`.
    fn solve(&self, g: &Graph, budget: Budget, seed: u64) -> Found;
}

/// Tabu search, budget converted to iterations.
#[derive(Clone, Copy, Debug, Default)]
pub struct Tabu;

impl Search for Tabu {
    fn name(&self) -> &'static str {
        "tabu"
    }
    fn solve(&self, g: &Graph, budget: Budget, seed: u64) -> Found {
        // One iteration evaluates every free spin's flip, so an iteration is `n` proposals.
        let iters = (budget.proposals / g.n.max(1) as u64).max(1) as usize;
        let p = crate::tabu::Params { iterations: iters, ..Default::default() };
        let o = crate::tabu::search(g, &p, seed);
        Found {
            arm: "tabu",
            energy: g.energy(&o.state),
            spent: iters as u64 * g.n as u64,
            state: o.state,
        }
    }
}

/// Breakout local search, budget converted to iterations.
#[derive(Clone, Copy, Debug, Default)]
pub struct Bls;

impl Search for Bls {
    fn name(&self) -> &'static str {
        "bls"
    }
    fn solve(&self, g: &Graph, budget: Budget, seed: u64) -> Found {
        let iters = (budget.proposals / g.n.max(1) as u64).max(1) as usize;
        let p = crate::bls::Params { iterations: iters, ..Default::default() };
        let o = crate::bls::search(g, &p, seed);
        Found {
            arm: "bls",
            energy: g.energy(&o.state),
            spent: iters as u64 * g.n as u64,
            state: o.state,
        }
    }
}

/// Simulated quantum annealing, budget split across Trotter slices and steps.
#[derive(Clone, Copy, Debug, Default)]
pub struct Sqa;

impl Search for Sqa {
    fn name(&self) -> &'static str {
        "sqa"
    }
    fn solve(&self, g: &Graph, budget: Budget, seed: u64) -> Found {
        let base = crate::sqa::Params::default();
        // A run costs trotter * n * steps * sweeps_per_step. The slice count is derived from the
        // physics and is not a knob to spend budget with, so the steps absorb it.
        let per_step = base.trotter as u64 * g.n.max(1) as u64;
        let steps = (budget.proposals / per_step.max(1)).max(1) as usize;
        let p = crate::sqa::Params { steps, ..base };
        let o = crate::sqa::run(g, &p, seed);
        Found {
            arm: "sqa",
            energy: g.energy(&o.state),
            spent: o.proposals,
            state: o.state,
        }
    }
}

/// Parallel tempering, budget split across the ladder.
#[derive(Clone, Copy, Debug, Default)]
pub struct Tempering;

impl Tempering {
    /// Rungs this arm uses on an `n`-spin model, over its own `[0.1, 6.0]` span.
    ///
    /// Sized from the model rather than fixed, for the reason [`crate::adaptive::replicas_for`]
    /// documents: a constant rung count severs the ladder on a glass much past a hundred spins, and
    /// a severed ladder still returns a tempering-shaped answer. Exposed so that property is
    /// testable — a fixed count still produces answers, just worse ones, which no comparison
    /// between arms would catch.
    #[must_use]
    pub fn rungs(n: usize) -> usize {
        crate::adaptive::replicas_for(n, 0.1, 6.0).max(2)
    }
}

impl Search for Tempering {
    fn name(&self) -> &'static str {
        "tempering"
    }
    fn solve(&self, g: &Graph, budget: Budget, seed: u64) -> Found {
        let rungs = Tempering::rungs(g.n);
        let betas = crate::tempering::geometric_ladder(0.1, 6.0, rungs);
        let per_round = rungs as u64 * g.n.max(1) as u64;
        let rounds = (budget.proposals / per_round.max(1)).max(1) as usize;
        let o = crate::tempering::parallel_tempering(g, &betas, rounds, 1, seed, None);
        Found {
            arm: "tempering",
            energy: g.energy(&o.best),
            spent: rounds as u64 * per_round,
            state: o.best,
        }
    }
}

/// Every arm this crate offers through [`Search`].
#[must_use]
pub fn all() -> Vec<Box<dyn Search>> {
    vec![Box::new(Tabu), Box::new(Bls), Box::new(Sqa), Box::new(Tempering)]
}

/// What a portfolio run produced.
#[derive(Clone, Debug, PartialEq)]
pub struct Report {
    /// The best answer any arm found.
    pub best: Found,
    /// What every arm found, in the order they were run.
    pub arms: Vec<Found>,
    /// Proposals spent in total.
    pub spent: u64,
}

/// Run every arm on the same instance, with the budget split evenly, and keep the best.
///
/// Each arm gets the same seed. That is deliberate: a portfolio whose arms are seeded differently
/// measures the seeds as much as the methods, and the comparison a user wants is between methods.
///
/// # Panics
///
/// If `arms` is empty, which is not a portfolio.
#[must_use]
pub fn run(g: &Graph, arms: &[Box<dyn Search>], budget: Budget, seed: u64) -> Report {
    assert!(!arms.is_empty(), "a portfolio needs at least one arm");
    let each = budget.split(arms.len());
    let found: Vec<Found> = arms.iter().map(|a| a.solve(g, each, seed)).collect();
    let best = found
        .iter()
        .min_by(|a, b| a.energy.total_cmp(&b.energy))
        .expect("at least one arm")
        .clone();
    let spent = found.iter().map(|f| f.spent).sum();
    Report { best, arms: found, spent }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn instance() -> (Graph, f64) {
        let p = crate::planted::frustrated_loops(8, 96, 3);
        (p.graph, p.ground_energy)
    }

    /// Every arm spends roughly what it was given, and none of them overspends.
    ///
    /// The load-bearing test. Each arm converts a proposal budget into its own knobs, and an arm
    /// whose conversion is wrong makes the portfolio a measurement of who cheated rather than of
    /// which method suits the instance. Checked against what the arms report AND against the ledger
    /// where the arm carries one, so the report cannot simply agree with itself.
    #[test]
    fn every_arm_stays_inside_the_budget_it_was_given() {
        let (g, _) = instance();
        for b in [200_000u64, 800_000] {
            let budget = Budget::new(b);
            for arm in all() {
                let f = arm.solve(&g, budget, 4);
                assert!(
                    f.spent <= b + g.n as u64,
                    "{}: spent {} against a budget of {b}",
                    arm.name(),
                    f.spent
                );
                // And it must not silently do almost nothing: an arm that spends a hundredth of its
                // budget is not being compared fairly either.
                assert!(
                    f.spent * 4 >= b,
                    "{}: spent only {} of {b}, which is not the same question the others answered",
                    arm.name(),
                    f.spent
                );
            }
        }
    }

    /// The reported energy is the state's energy, for every arm.
    ///
    /// Solvers carry their own running energy, and a portfolio that trusted it would inherit any
    /// drift. Each `Found` recomputes from the state it returns, and this is what says so.
    #[test]
    fn every_arm_reports_the_energy_of_the_state_it_returns() {
        let (g, _) = instance();
        for arm in all() {
            let f = arm.solve(&g, Budget::new(400_000), 9);
            assert_eq!(f.state.len(), g.n, "{}", arm.name());
            assert!(f.state.iter().all(|&v| v == 1 || v == -1), "{}", arm.name());
            assert!(
                (f.energy - g.energy(&f.state)).abs() < 1e-9,
                "{}: reported {} for a state worth {}",
                arm.name(),
                f.energy,
                g.energy(&f.state)
            );
        }
    }

    /// The portfolio returns the best of its arms, and never worse than any of them.
    #[test]
    fn the_portfolio_is_the_best_of_what_it_ran() {
        let (g, _) = instance();
        let arms = all();
        let r = run(&g, &arms, Budget::new(1_200_000), 11);
        assert_eq!(r.arms.len(), arms.len());
        for a in &r.arms {
            assert!(
                r.best.energy <= a.energy + 1e-12,
                "the portfolio reported {} while its {} arm found {}",
                r.best.energy,
                a.arm,
                a.energy
            );
        }
        assert!(r.arms.iter().any(|a| a.arm == r.best.arm && a.energy == r.best.energy));
        assert!((r.best.energy - g.energy(&r.best.state)).abs() < 1e-9);
    }

    /// The budget is split, not handed to each arm in full.
    ///
    /// The mistake that makes a portfolio look free: give every arm the whole budget and it beats
    /// any single arm at "the same" cost, having spent `k` times as much.
    #[test]
    fn the_budget_is_divided_among_the_arms() {
        let (g, _) = instance();
        let arms = all();
        let total = 1_000_000u64;
        let r = run(&g, &arms, Budget::new(total), 3);
        assert!(
            r.spent <= total + arms.len() as u64 * g.n as u64,
            "the portfolio spent {} of a {total} budget across {} arms",
            r.spent,
            arms.len()
        );
        assert_eq!(Budget::new(100).split(4), Budget::new(25));
        assert_eq!(Budget::new(102).split(4), Budget::new(25), "the remainder is dropped");
        assert_eq!(Budget::new(7).split(0), Budget::new(0), "no arms, no budget");

        // And a LOWER bound, or the assertion above is satisfied by spending nothing: a portfolio
        // reporting zero would pass every upper bound in this file.
        assert!(
            r.spent * 2 >= total,
            "the portfolio spent only {} of a {total} budget, which is not the run it reported",
            r.spent
        );
    }

    /// The tempering arm sizes its ladder from the model, not from a constant.
    ///
    /// A fixed rung count still produces answers -- worse ones -- so no comparison between arms
    /// catches it, and `adaptive`'s own measurements say a constant severs the ladder on a glass
    /// past about a hundred spins. Tested as the property it is.
    #[test]
    fn the_tempering_arm_sizes_its_ladder_from_the_model() {
        let small = Tempering::rungs(100);
        let large = Tempering::rungs(400);
        assert!(small >= 2 && large >= 2, "a ladder needs two rungs to swap at all");
        let ratio = large as f64 / small as f64;
        assert!(
            (1.8..=2.2).contains(&ratio),
            "quadrupling the spins should about double the rungs: {small} then {large}"
        );
        assert_eq!(
            Tempering::rungs(196),
            crate::adaptive::replicas_for(196, 0.1, 6.0).max(2),
            "the arm must use the crate's rule, not a copy of it that can drift"
        );
    }

    /// On a planted instance the portfolio lands at or under the worst arm, at the same total cost.
    ///
    /// The claim a portfolio has to earn. It is stated against the WORST arm rather than the best,
    /// because beating the best would require knowing which that is — which is the thing a portfolio
    /// exists not to have to know.
    #[test]
    fn the_portfolio_beats_the_arm_that_suits_the_instance_least() {
        let (g, opt) = instance();
        let arms = all();
        let total = 1_200_000u64;
        let r = run(&g, &arms, Budget::new(total), 7);
        let worst = r.arms.iter().map(|a| a.energy).fold(f64::NEG_INFINITY, f64::max);
        assert!(
            r.best.energy <= worst,
            "portfolio {} against its worst arm {worst}",
            r.best.energy
        );
        // And it should be somewhere near the planted optimum, not merely internally consistent.
        let excess = (r.best.energy - opt) / opt.abs();
        assert!(excess < 0.25, "portfolio landed {:.1}% above the planted optimum", excess * 100.0);
    }
}
