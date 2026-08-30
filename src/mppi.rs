//! Sampling-based control: choosing an action by weighting sampled futures.
//!
//! This is the workload that connects a thermodynamic sampler to a robot, and it is the one no
//! thermodynamic vendor is pursuing. Model-predictive path integral control picks an action by
//! rolling out many perturbed control sequences, weighting each by `exp(-cost/λ)`, and taking the
//! weighted mean. That weighting **is** a Boltzmann distribution over trajectories with `λ` playing
//! the role of temperature — so a machine whose native operation is drawing Boltzmann samples is
//! doing the expensive part of this algorithm directly rather than emulating it.
//!
//! # It comes with an oracle
//!
//! Most control results are reported against whatever the last paper achieved. This one need not
//! be: on a linear system with quadratic cost, the optimal controller is known in closed form. The
//! discrete algebraic Riccati equation has an exact scalar solution, so [`Lqr`] gives the true
//! optimum and [`Mppi`] can be scored against it rather than against a rival heuristic.
//!
//! That is the whole point of putting this here. A sampling controller that lands within a few
//! percent of the provable optimum is a measurement; one that merely "performs well" is a press
//! release.
//!
//! # What it achieves, and where it stops
//!
//! Measured against the exact optimum by `examples/mppi_probe.rs`, which now PRINTS this table
//! (it did not: it swept horizons {5,15} and iters {1,10} only, so the two unstable rows came from
//! no command in the repository). All rows at `seed = 7`, `rollouts = 300`, `sigma = 0.2`,
//! `lambda = 0.3`, `x0 = 1`:
//!
//! | plant | horizon | iters | excess @200 | @100 | @800 |
//! |---|---|---|---|---|---|
//! | stable, `a = 0.9` | 5 | 10 | **7.1%** | 3.4% | 22.6% |
//! | stable, `a = 0.9` | 5 | 1 | 28.7% | 26.0% | 43.2% |
//! | stable, `a = 0.9` | 15 | 10 | 19.9% | 10.6% | 78.8% |
//! | unstable, `a = 1.1` | 10 | 30 | 15.1% | 7.2% | 61.2% |
//! | unstable, `a = 1.1` | 30 | 30 | **1446%** | 733.5% | 5400% |
//!
//! **The step count is part of every number in that table, and it was missing.** Excess over the
//! provable optimum is not a property of the method: MPPI injects `sigma` noise at every step forever,
//! while the LQR oracle's `cost_to_go` is a finite infinite-horizon cost from `x0 = 1`, so the ratio
//! grows without bound in the horizon it is measured over. The flagship 7.1% is **1.0% at 25 steps and
//! 22.6% at 800**. It is a coordinate — a number plus the run length it was taken over — and it was
//! published as though it were a property.
//!
//! **And the 729% that used to sit in the last row was wrong.** At 200 steps, where all three stable
//! rows reproduce to the printed digit, horizon 30 gives 1446%. 729% is what horizon 30 gives at *100*
//! steps — but at 100 steps the row above it reads 7.2%, not the 15.7% that was published beside it.
//! No single run produced both numbers.
//!
//! Three things that table says plainly:
//!
//! - **One refinement pass is not converged.** The textbook receding-horizon form applies a single
//!   weighted correction before committing an action; ten passes take 28.7% down to 7.1%.
//! - **A longer horizon makes it worse, not better**, which is the opposite of the usual intuition.
//!   The rollouts are open loop: nothing inside a rollout corrects for drift, so noise compounds
//!   over the horizon and the weighting ends up dominated by whichever sample was least unlucky.
//! - **An unstable plant is much harder.** With `a = 1.1` the state grows like `1.1^H` inside every
//!   rollout, and at `H = 30` the method fails outright. Practical MPPI on unstable systems rolls
//!   out around a stabilising base policy rather than around zero; that is not implemented here and
//!   the numbers above are what you get without it.
//!
//! None of that is hidden behind a favourable default. The point of having an oracle is to be able
//! to say where a method stops working.

use crate::rng::Pcg;

/// A scalar linear system `x' = a x + b u` with cost `Σ q x² + r u²`.
#[derive(Clone, Copy, Debug)]
pub struct System {
    pub a: f64,
    pub b: f64,
    pub q: f64,
    pub r: f64,
}

impl System {
    /// One step of the dynamics.
    pub fn step(&self, x: f64, u: f64) -> f64 {
        self.a * x + self.b * u
    }

    /// Stage cost.
    pub fn cost(&self, x: f64, u: f64) -> f64 {
        self.q * x * x + self.r * u * u
    }

    /// Total cost of driving `x0` for `steps` under a controller.
    pub fn rollout<F: FnMut(f64, usize) -> f64>(&self, x0: f64, steps: usize, mut policy: F) -> f64 {
        let mut x = x0;
        let mut total = 0.0;
        for k in 0..steps {
            let u = policy(x, k);
            total += self.cost(x, u);
            x = self.step(x, u);
        }
        total
    }
}

/// The exact optimal controller, from the closed-form solution of the Riccati equation.
///
/// This is the oracle. It is not an approximation and not a strong baseline — it is the best any
/// controller can do on this system, so a sampling controller's distance from it is a measurement
/// rather than a comparison.
#[derive(Clone, Copy, Debug)]
pub struct Lqr {
    /// Steady-state cost-to-go coefficient.
    pub p: f64,
    /// Optimal feedback gain: the optimal action is `-k x`.
    pub k: f64,
}

impl Lqr {
    /// Solve `p = q + a²p - (a b p)² / (r + b² p)` for its positive root.
    ///
    /// Iterating the Riccati recursion converges monotonically from `p = q` for a stabilisable
    /// system, which is both simpler and more obviously correct than the quadratic formula, and it
    /// is checked against the residual in the tests rather than assumed.
    pub fn solve(s: &System) -> Lqr {
        let mut p = s.q;
        for _ in 0..10_000 {
            let next = s.q + s.a * s.a * p - (s.a * s.b * p).powi(2) / (s.r + s.b * s.b * p);
            if (next - p).abs() < 1e-15 * next.abs().max(1.0) {
                p = next;
                break;
            }
            p = next;
        }
        let k = (s.a * s.b * p) / (s.r + s.b * s.b * p);
        Lqr { p, k }
    }

    /// The optimal action in state `x`.
    pub fn action(&self, x: f64) -> f64 {
        -self.k * x
    }

    /// Optimal infinite-horizon cost from `x0`, which is `p x0²`.
    pub fn cost_to_go(&self, x0: f64) -> f64 {
        self.p * x0 * x0
    }
}

/// Model-predictive path integral control.
///
/// Samples `rollouts` perturbed control sequences over a `horizon`, weights each by
/// `exp(-(cost - min_cost)/λ)`, and returns the weighted-mean first action.
#[derive(Clone, Copy, Debug)]
pub struct Mppi {
    pub horizon: usize,
    pub rollouts: usize,
    /// Exploration noise on the control.
    pub sigma: f64,
    /// Temperature. Small λ concentrates weight on the best rollout; large λ averages everything.
    pub lambda: f64,
    /// Refinement passes per control step.
    ///
    /// One pass is the textbook receding-horizon form and it is *not* converged: the nominal
    /// sequence only gets one weighted correction before an action is committed. More passes buy
    /// accuracy at linear cost, and the tests measure how much.
    pub iters: usize,
}

impl Mppi {
    /// Choose an action in state `x`, given a nominal control sequence to perturb around.
    ///
    /// `nominal` is warm-started from the previous solve in a real loop, which is what makes MPPI
    /// work at all: a cold start every step throws away the search that the last step paid for.
    pub fn action(&self, s: &System, x: f64, nominal: &mut [f64], rng: &mut Pcg) -> f64 {
        assert_eq!(nominal.len(), self.horizon, "nominal must cover the horizon");
        for _ in 0..self.iters.max(1) {
            self.refine(s, x, nominal, rng);
        }
        let u0 = nominal[0];
        // Shift forward and let the tail decay to zero.
        //
        // NOT `rotate_left`, which puts the action just executed at the FAR END of the horizon --
        // the largest stabilising action, reapplied where the state should already be near zero.
        // That error compounds with horizon length, which is how it was found: MPPI got worse as
        // the horizon grew, when a longer horizon should help.
        for k in 0..self.horizon - 1 {
            nominal[k] = nominal[k + 1];
        }
        nominal[self.horizon - 1] = 0.0;
        u0
    }

    /// One weighted-sampling pass over the nominal sequence.
    fn refine(&self, s: &System, x: f64, nominal: &mut [f64], rng: &mut Pcg) {
        let mut noise = vec![0.0f64; self.horizon * self.rollouts];
        let mut costs = vec![0.0f64; self.rollouts];

        for r in 0..self.rollouts {
            let mut xk = x;
            let mut c = 0.0;
            for k in 0..self.horizon {
                let e = gauss(rng) * self.sigma;
                noise[r * self.horizon + k] = e;
                let u = nominal[k] + e;
                c += s.cost(xk, u);
                xk = s.step(xk, u);
            }
            costs[r] = c;
        }

        // Subtract the minimum before exponentiating. Without it, a horizon of any length overflows
        // to zero weight everywhere and the controller silently returns the nominal sequence.
        let min = costs.iter().cloned().fold(f64::INFINITY, f64::min);
        let mut wsum = 0.0;
        let mut w = vec![0.0f64; self.rollouts];
        for r in 0..self.rollouts {
            w[r] = (-(costs[r] - min) / self.lambda).exp();
            wsum += w[r];
        }

        for k in 0..self.horizon {
            let mut d = 0.0;
            for r in 0..self.rollouts {
                d += w[r] * noise[r * self.horizon + k];
            }
            nominal[k] += d / wsum;
        }
    }

    /// Run a closed loop and return the total cost incurred.
    pub fn run(&self, s: &System, x0: f64, steps: usize, seed: u64) -> f64 {
        let mut rng = Pcg::new(seed, 0);
        let mut nominal = vec![0.0f64; self.horizon];
        let mut x = x0;
        let mut total = 0.0;
        for _ in 0..steps {
            let u = self.action(s, x, &mut nominal, &mut rng);
            total += s.cost(x, u);
            x = s.step(x, u);
        }
        total
    }
}

fn gauss(rng: &mut Pcg) -> f64 {
    let u = rng.f64().max(1e-15);
    let v = rng.f64();
    (-2.0 * u.ln()).sqrt() * (core::f64::consts::TAU * v).cos()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Stable plant. The unstable one is exercised separately and its limits are documented.
    const SYS: System = System { a: 0.9, b: 1.0, q: 1.0, r: 0.5 };
    const UNSTABLE: System = System { a: 1.1, b: 1.0, q: 1.0, r: 0.5 };
    const TUNED: Mppi = Mppi { horizon: 5, rollouts: 300, sigma: 0.2, lambda: 0.3, iters: 10 };

    #[test]
    fn the_riccati_solution_actually_solves_the_equation() {
        // The oracle must be checked before anything is scored against it.
        for s in [
            System { a: 1.1, b: 1.0, q: 1.0, r: 0.5 },
            System { a: 0.9, b: 0.5, q: 2.0, r: 1.0 },
            System { a: 1.5, b: 1.0, q: 1.0, r: 0.1 },
        ] {
            let l = Lqr::solve(&s);
            let residual =
                s.q + s.a * s.a * l.p - (s.a * s.b * l.p).powi(2) / (s.r + s.b * s.b * l.p) - l.p;
            assert!(residual.abs() < 1e-9, "Riccati residual {residual} for {s:?}");
            assert!(l.p > 0.0, "the cost-to-go coefficient must be positive");
        }
    }

    #[test]
    fn the_optimal_controller_is_optimal() {
        // Perturbing the optimal gain must make things worse in both directions, or it is not a
        // minimum and the oracle is wrong.
        let l = Lqr::solve(&SYS);
        let base = SYS.rollout(1.0, 400, |x, _| -l.k * x);
        for d in [-0.2, -0.05, 0.05, 0.2] {
            let worse = SYS.rollout(1.0, 400, |x, _| -(l.k + d) * x);
            assert!(worse > base, "gain {} beat the optimum {}", l.k + d, l.k);
        }
        // and the closed-form cost-to-go matches the realised cost of following it
        assert!((base - l.cost_to_go(1.0)).abs() / l.cost_to_go(1.0) < 1e-6);
    }

    #[test]
    fn sampling_control_lands_near_the_provable_optimum() {
        // The measurement this module exists for: 7.1% AT 200 STEPS.
        //
        // The step count is not incidental and this test used to hide that. `excess < 0.10` passes
        // at 200 steps and FAILS at 400, where the same code gives 11.7% -- not because anything
        // regressed, but because the metric grows without bound in run length. A guard that reads
        // as "sampling control is within 10% of optimal" while being true only at one hardcoded
        // horizon states a property the method does not have.
        let l = Lqr::solve(&SYS);
        let optimal = l.cost_to_go(1.0);
        let excess = |steps: usize| (TUNED.run(&SYS, 1.0, steps, 7) - optimal) / optimal;

        let at200 = excess(200);
        assert!(at200 > -1e-9, "nothing can beat the optimum: {at200}");
        // A BAND, not a ceiling. A ceiling is satisfied by a method that got better for the wrong
        // reason -- and by one that got worse in a way the ceiling happens to still admit.
        assert!(
            (0.065..0.078).contains(&at200),
            "the published 7.1% at 200 steps moved to {:.1}%",
            at200 * 100.0
        );

        // And the growth itself is the claim the docs now make, so it is the claim under test.
        // Monotone in run length, because the noise MPPI injects never stops and the oracle's
        // cost-to-go is finite.
        let series: Vec<f64> = [25usize, 50, 100, 200, 400, 800].iter().map(|&s| excess(s)).collect();
        assert!(
            series.windows(2).all(|w| w[1] > w[0]),
            "excess must grow with run length, and it went {series:?}"
        );
        assert!(series[0] < 0.02, "1.0% at 25 steps: {:.1}%", series[0] * 100.0);
        assert!(series[5] > 0.20, "22.6% at 800 steps: {:.1}%", series[5] * 100.0);
    }

    #[test]
    fn refinement_passes_are_what_buy_the_accuracy() {
        // One pass is the textbook form and it is not converged. If this stops holding, the
        // weighted update has stopped doing anything.
        let opt = Lqr::solve(&SYS).cost_to_go(1.0);
        let ex = |iters: usize| {
            let m = Mppi { iters, ..TUNED };
            (m.run(&SYS, 1.0, 200, 7) - opt) / opt
        };
        let (one, ten) = (ex(1), ex(10));
        assert!(ten < one / 2.0, "10 passes ({ten:.3}) should halve 1 pass ({one:.3})");
    }

    #[test]
    fn a_longer_horizon_makes_it_worse_and_that_is_expected() {
        // Documented rather than hidden. The rollouts are open loop, so noise compounds over the
        // horizon instead of being corrected inside it. Anyone reaching for a longer horizon to
        // improve this should find out here rather than in a robot.
        let opt = Lqr::solve(&SYS).cost_to_go(1.0);
        let ex = |h: usize| {
            let m = Mppi { horizon: h, ..TUNED };
            (m.run(&SYS, 1.0, 200, 7) - opt) / opt
        };
        assert!(ex(15) > ex(5), "the open-loop horizon penalty should be visible");
    }

    #[test]
    fn an_unstable_plant_is_much_harder() {
        // Also documented rather than hidden: the state grows like a^H inside every rollout.
        let opt = Lqr::solve(&UNSTABLE).cost_to_go(1.0);
        let ex = |h: usize| {
            let m = Mppi { horizon: h, iters: 30, ..TUNED };
            (m.run(&UNSTABLE, 1.0, 200, 7) - opt) / opt
        };
        assert!(ex(10) < ex(30), "a long horizon on an unstable plant should be far worse");
        // A BAND ROUND THE PUBLISHED NUMBERS, because `> 1.0` was how a wrong one survived.
        //
        // This used to assert only that horizon 30 exceeded 100%. The docs published 729% for that
        // cell; the real figure at these settings is 1446%, and any value from 101% to infinity
        // satisfied the old guard. A test whose bound is two orders of magnitude looser than the
        // number it is guarding does not guard it.
        assert!(
            (0.140..0.162).contains(&ex(10)),
            "the published 15.1% at horizon 10 moved to {:.1}%",
            ex(10) * 100.0
        );
        assert!(
            (13.5..15.5).contains(&ex(30)),
            "the published 1446% at horizon 30 moved to {:.1}%",
            ex(30) * 100.0
        );
    }

    #[test]
    fn the_weighting_overflow_is_handled() {
        // A long horizon makes raw costs large; exponentiating them without subtracting the minimum
        // gives zero weight everywhere and the controller silently returns its nominal sequence,
        // which looks like a working controller that ignores its samples.
        let l = Lqr::solve(&SYS);
        let m = Mppi { horizon: 200, rollouts: 100, sigma: 0.2, lambda: 0.3, iters: 1 };
        let cost = m.run(&SYS, 1.0, 100, 5);
        assert!(cost.is_finite(), "the weights must not all underflow to zero");
        assert!(cost > 0.0);
        let _ = l;
    }

    #[test]
    fn it_is_deterministic_by_seed() {
        assert_eq!(TUNED.run(&SYS, 1.0, 50, 11), TUNED.run(&SYS, 1.0, 50, 11));
        assert_ne!(TUNED.run(&SYS, 1.0, 50, 11), TUNED.run(&SYS, 1.0, 50, 12));
    }

    #[test]
    fn it_is_actually_controlling_something() {
        // An unstable plant diverges without control, so beating "do nothing" by orders of
        // magnitude is the floor, not the achievement.
        let uncontrolled = UNSTABLE.rollout(1.0, 40, |_, _| 0.0);
        let m = Mppi { horizon: 8, iters: 20, ..TUNED };
        let controlled = m.run(&UNSTABLE, 1.0, 200, 9);
        assert!(controlled < uncontrolled / 100.0,
                "controlled {controlled} vs uncontrolled {uncontrolled}");
    }
}
