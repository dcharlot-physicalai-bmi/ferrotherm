//! What a machine costs when it is mostly WAITING.
//!
//! Every energy comparison in this stack so far divides by work done: joules above idle, per node
//! update. That is the right question for a machine kept busy, and it is the wrong question for
//! most of the places a sampling substrate would actually go. A sensor that draws from a posterior
//! ten times a second computes for microseconds and waits for the rest of the second, and a
//! figure that subtracts idle prices the microseconds and throws away the wait.
//!
//! The wait is where the joules are. This module does the arithmetic that says so, and the result
//! is sharper than expected:
//!
//! > For an intermittent workload the standby power a competing device must beat is the
//! > incumbent's IDLE draw. Its sampling efficiency does not enter.
//!
//! That is the whole thermodynamic value proposition restated as one measurable hardware number,
//! and it is a number no thermodynamic vendor publishes. [`Prices`](crate::ledger::Prices) has
//! `e_sample`, `e_read`, `e_write` and a reflash cap, because those are what the Z1 tables state;
//! there is no standby term because there is no published standby figure to put in it. Reporting
//! that absence is more useful than a guess wearing a decimal point.
//!
//! ## "But our fabric switches off between tasks"
//!
//! The obvious objection, and the model already answers it: a challenger with zero standby wins
//! every comparison here trivially. It just does not get to keep its couplings. A device that
//! powers down has to restore them on every wake, which is [`Prices::e_write`](crate::ledger::Prices)
//! per node -- ~21,700 times the cost of a sample on the one device model that states both -- and
//! [`Prices::reflash_hz_cap`](crate::ledger::Prices) bounds how often it may. So a power-cycling
//! device pays the write path once per period, and that belongs in the `compute_joules` argument to
//! [`Machine::beaten_by`]. Price it with [`Ledger::joules`](crate::ledger::Ledger::joules) against
//! that device's own prices; the ledger has charged writes since 0.9.0 for exactly this reason.
//!
//! Standing off and staying resident are therefore the two ends of one trade, and this module and
//! the ledger price opposite ends of it. Neither end has a published number for the standby half.
//!
//! ```
//! use ferrotherm::duty::Machine;
//!
//! // A GPU that idles at 20 W and adds 100 W while it works, at 1e9 updates/s.
//! let gpu = Machine::new(20.0, 100.0, 1e9).unwrap();
//!
//! // Below this duty cycle, most of what the machine costs is being switched on.
//! assert!((gpu.idle_dominant_below() - 0.2).abs() < 1e-12);
//!
//! // One million updates, once a second: 1 ms of work in a 1 s period.
//! let b = gpu.standby_budget(1_000_000, 1.0).unwrap();
//! assert!((b - 20.1).abs() < 1e-9);   // ~= the idle draw, and almost nothing else
//! ```

/// A machine characterised by what it draws idle, what it adds while working, and how fast it works.
///
/// All three are meant to be MEASURED on one machine -- `ferrotherm_meter::Run` reports exactly
/// these -- rather than taken from a datasheet. The type carries no provenance string of its own
/// because it is arithmetic; the provenance belongs to whatever produced the watts.
#[derive(Clone, Copy, Debug)]
pub struct Machine {
    /// Whole-system power with the machine available but not working.
    pub idle_watts: f64,
    /// What the workload ADDS on top of idle while it runs: `mean_watts - idle_watts`.
    pub marginal_watts: f64,
    /// Work units per second while working.
    pub rate: f64,
}

/// Why a cadence could not be priced.
#[derive(Clone, Debug, PartialEq)]
pub enum DutyError {
    /// A watt or a rate that is negative, infinite or NaN.
    NotPhysical(&'static str),
    /// No work, or a period that is not a positive duration.
    Empty(&'static str),
    /// The machine cannot finish this work inside this period.
    ///
    /// Not a slow answer: an impossible one. Pricing it would describe a run that could not have
    /// happened, which is the same failure [`Ledger::reflash_seconds`](crate::ledger::Ledger::reflash_seconds)
    /// exists to catch on the device side.
    CannotSustain {
        /// Seconds of computation the work needs.
        needs_s: f64,
        /// Seconds the cadence allows.
        period_s: f64,
    },
}

impl core::fmt::Display for DutyError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            DutyError::NotPhysical(s) | DutyError::Empty(s) => write!(f, "{s}"),
            DutyError::CannotSustain { needs_s, period_s } => write!(
                f,
                "that work needs {needs_s:.6} s of computation and the cadence allows {period_s:.6} s, \
                 so this machine cannot sustain it -- a joules figure here would price a run that \
                 could not have happened"
            ),
        }
    }
}

/// The result of putting a challenger device against a measured machine on one task.
///
/// Both totals are carried, not just the verdict, because a boolean hides how close it was -- and
/// on this question the margin is usually the whole story.
#[derive(Clone, Copy, Debug)]
pub struct Verdict {
    /// What the measured machine spends over one period, idle included.
    pub incumbent_joules: f64,
    /// What the challenger spends over the same period.
    pub challenger_joules: f64,
    /// The standby power the challenger had to come in under, granting it free computation.
    pub standby_budget: f64,
    /// Whether the challenger is STRICTLY cheaper. A tie is not a reason to change fabric.
    pub challenger_wins: bool,
}

impl Machine {
    /// Refuses anything that is not a physical machine.
    ///
    /// `marginal_watts` of zero is allowed and means the workload did not rise above idle, which is
    /// a real measurement outcome; a NEGATIVE marginal is not, and the meter's own noise floor
    /// once turned one into zero silently.
    pub fn new(idle_watts: f64, marginal_watts: f64, rate: f64) -> Result<Machine, DutyError> {
        if !idle_watts.is_finite() || idle_watts < 0.0 {
            return Err(DutyError::NotPhysical("idle power must be finite and non-negative"));
        }
        if !marginal_watts.is_finite() || marginal_watts < 0.0 {
            return Err(DutyError::NotPhysical(
                "marginal power must be finite and non-negative; a negative delta is the meter's \
                 noise, not a machine that generates power while it works",
            ));
        }
        if !rate.is_finite() || rate <= 0.0 {
            return Err(DutyError::NotPhysical("rate must be finite and positive"));
        }
        Ok(Machine { idle_watts, marginal_watts, rate })
    }

    /// Seconds of computation `work` units need.
    pub fn run_seconds(&self, work: u64) -> f64 {
        work as f64 / self.rate
    }

    /// The duty cycle below which IDLE is more than half of what the machine costs.
    ///
    /// `idle / marginal`, and the derivation is two lines: over a period the machine pays
    /// `marginal * t_run + idle * period`, and with `t_run = duty * period` the first term beats
    /// the second exactly when `duty > idle / marginal`.
    ///
    /// Read it as a verdict on effort. Below this figure, making the computation cheaper cannot
    /// move the bill by more than a factor of two, no matter how much cheaper it gets -- which is
    /// the regime every peak-throughput benchmark in this field is measuring outside of.
    ///
    /// Returns `f64::INFINITY` when the workload never rose above idle, because then idle is
    /// everything at every cadence.
    #[must_use]
    pub fn idle_dominant_below(&self) -> f64 {
        if self.marginal_watts == 0.0 {
            return f64::INFINITY;
        }
        self.idle_watts / self.marginal_watts
    }

    /// Fraction of a period spent computing, or an error if the cadence is unsustainable.
    pub fn duty(&self, work: u64, period_s: f64) -> Result<f64, DutyError> {
        if work == 0 {
            return Err(DutyError::Empty("no work was done, so there is no duty cycle"));
        }
        if !period_s.is_finite() || period_s <= 0.0 {
            return Err(DutyError::Empty("the period must be a finite positive number of seconds"));
        }
        let needs_s = self.run_seconds(work);
        if needs_s > period_s {
            return Err(DutyError::CannotSustain { needs_s, period_s });
        }
        Ok(needs_s / period_s)
    }

    /// Total joules for one period: the work, plus the whole wait.
    ///
    /// `marginal * t_run + idle * period`. Idle is NOT subtracted, which is the entire point --
    /// a machine that must stay available pays for staying available, and the workload is the
    /// reason it is switched on.
    pub fn joules_per_period(&self, work: u64, period_s: f64) -> Result<f64, DutyError> {
        self.duty(work, period_s)?;
        Ok(self.marginal_watts * self.run_seconds(work) + self.idle_watts * period_s)
    }

    /// Effective joules per work unit at this cadence, idle included.
    ///
    /// Compare against the above-idle figure the meter reports: this one rises without bound as
    /// the cadence slackens, and that divergence is the honest shape of intermittent compute.
    pub fn joules_per_unit(&self, work: u64, period_s: f64) -> Result<f64, DutyError> {
        Ok(self.joules_per_period(work, period_s)? / work as f64)
    }

    /// The standby power a competing device must come in UNDER to be cheaper here, granting that
    /// device perfectly free computation.
    ///
    /// `idle + marginal * duty`. The generosity is deliberate: a bound that already assumes the
    /// challenger's sampling costs nothing cannot be argued down by a better sampler, so whatever
    /// it rules out stays ruled out.
    ///
    /// As the cadence slackens this collapses to the incumbent's idle draw, and the challenger's
    /// entire case reduces to one number about its own standby -- not to anything about physics,
    /// throughput or joules per flip.
    pub fn standby_budget(&self, work: u64, period_s: f64) -> Result<f64, DutyError> {
        let d = self.duty(work, period_s)?;
        Ok(self.idle_watts + self.marginal_watts * d)
    }

    /// Would a challenger with this standby draw and this compute cost actually be cheaper here?
    ///
    /// The two numbers a device has to state to answer the question, and the reason this method
    /// exists rather than a table: nobody in the thermodynamic field publishes the first one, so
    /// the only way to complete the comparison is for the party that knows it to supply it. This
    /// takes it as an argument and does the arithmetic in public.
    ///
    /// `compute_joules` is the challenger's ENERGY FOR THE WHOLE TASK, not per operation -- a
    /// device that samples differently does not do the same number of operations, so a per-op
    /// price would not be comparable. Price it with [`Ledger::joules`](crate::ledger::Ledger::joules)
    /// against that device's own [`Prices`](crate::ledger::Prices), which is what the ledger is for.
    pub fn beaten_by(
        &self,
        challenger_standby_watts: f64,
        challenger_compute_joules: f64,
        work: u64,
        period_s: f64,
    ) -> Result<Verdict, DutyError> {
        if !challenger_standby_watts.is_finite() || challenger_standby_watts < 0.0 {
            return Err(DutyError::NotPhysical(
                "the challenger's standby power must be finite and non-negative",
            ));
        }
        if !challenger_compute_joules.is_finite() || challenger_compute_joules < 0.0 {
            return Err(DutyError::NotPhysical(
                "the challenger's compute energy must be finite and non-negative",
            ));
        }
        let incumbent = self.joules_per_period(work, period_s)?;
        let challenger = challenger_compute_joules + challenger_standby_watts * period_s;
        Ok(Verdict {
            incumbent_joules: incumbent,
            challenger_joules: challenger,
            standby_budget: self.standby_budget(work, period_s)?,
            // Strictly cheaper. A tie is not a reason to change your compute fabric.
            challenger_wins: challenger < incumbent,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gpu() -> Machine {
        // Shaped like a measured discrete GPU: idles at 20 W, adds 100 W under load.
        Machine::new(20.0, 100.0, 1e9).unwrap()
    }

    #[test]
    fn idle_dominates_below_the_ratio_of_the_two_powers() {
        let m = gpu();
        let d_star = m.idle_dominant_below();
        assert!((d_star - 0.2).abs() < 1e-12, "{d_star}");

        // Straddle it and check the shares actually cross, rather than trusting the algebra.
        // At the threshold itself the two terms are equal by construction.
        let work = 1_000_000u64; // 1 ms of computation at 1e9/s
        let t_run = m.run_seconds(work);
        for (duty, idle_should_win) in [(0.02, true), (0.5, false)] {
            let period = t_run / duty;
            let total = m.joules_per_period(work, period).unwrap();
            let idle_part = m.idle_watts * period;
            assert_eq!(
                idle_part > total / 2.0,
                idle_should_win,
                "at duty {duty} idle share was {:.3}",
                idle_part / total
            );
        }
    }

    #[test]
    fn the_standby_budget_collapses_to_the_incumbents_idle_draw() {
        // THE FINDING. Rarefy the cadence and the budget a challenger must beat stops depending on
        // anything about computation at all -- it becomes the GPU's idle power and nothing else.
        let m = gpu();
        let work = 1_000_000u64;
        let busy = m.standby_budget(work, m.run_seconds(work)).unwrap(); // duty = 1
        let rare = m.standby_budget(work, 60.0).unwrap(); // once a minute

        assert!((busy - 120.0).abs() < 1e-9, "at full duty the whole draw is the budget: {busy}");
        // Relative, not absolute: the residual is `marginal * duty`, so what the claim really
        // says is that the compute term has become a rounding error on the idle term.
        let residual = (rare - m.idle_watts).abs() / m.idle_watts;
        assert!(
            residual < 1e-4,
            "once a minute the budget is the idle draw to within {:.4}%: {rare} vs {}",
            100.0 * residual,
            m.idle_watts
        );
        assert!(
            rare < busy / 5.0,
            "the two regimes have to be far apart or there is nothing to report"
        );
    }

    #[test]
    fn free_computation_does_not_win_on_its_own() {
        // A device model with zero compute energy still loses if it has to stay powered. This is
        // the trap the module exists to close: every published thermodynamic comparison prices the
        // sampling and leaves standby out, and at low duty cycle standby is the entire bill.
        let m = gpu();
        let work = 1_000_000u64;
        let period = 1.0;
        let budget = m.standby_budget(work, period).unwrap();

        let incumbent = m.joules_per_period(work, period).unwrap();
        // A challenger whose computation is free but which idles at twice the budget.
        let challenger = 0.0 + (2.0 * budget) * period;
        assert!(
            challenger > incumbent,
            "free sampling at {:.1} W standby still costs {challenger:.1} J against {incumbent:.1} J",
            2.0 * budget
        );
    }

    #[test]
    fn a_cadence_the_machine_cannot_meet_is_refused_rather_than_priced() {
        // Mirrors `Ledger::reflash_seconds`: an unphysical schedule gets an error, not a number.
        let m = gpu();
        let err = m.joules_per_period(10_000_000_000, 1.0).unwrap_err();
        match err {
            DutyError::CannotSustain { needs_s, period_s } => {
                assert!((needs_s - 10.0).abs() < 1e-9 && (period_s - 1.0).abs() < 1e-9);
            }
            other => panic!("expected a sustainability refusal, got {other:?}"),
        }
        assert!(m.joules_per_period(10_000_000_000, 11.0).is_ok(), "11 s is enough for 10 s of work");
    }

    #[test]
    fn effective_cost_per_unit_diverges_as_the_cadence_slackens() {
        // The above-idle figure is FLAT in the cadence by construction; the honest one is not.
        let m = gpu();
        let work = 1_000_000u64;
        let above_idle_per_unit = m.marginal_watts * m.run_seconds(work) / work as f64;

        let busy = m.joules_per_unit(work, m.run_seconds(work)).unwrap();
        let rare = m.joules_per_unit(work, 60.0).unwrap();
        assert!(rare > busy * 1000.0, "busy {busy:.3e} vs rare {rare:.3e}");
        assert!(
            above_idle_per_unit < busy,
            "subtracting idle always reports less than the machine actually spent"
        );
    }

    #[test]
    fn a_challenger_is_judged_on_standby_once_the_cadence_slackens() {
        // The whole argument in one test. Two challengers, identical except for standby: one at a
        // quarter of the incumbent's idle, one at twice it. Give BOTH of them perfectly free
        // computation, so nothing about sampling can be doing the work here.
        let m = gpu();
        let work = 1_000_000u64;
        let period = 60.0;

        let good = m.beaten_by(m.idle_watts / 4.0, 0.0, work, period).unwrap();
        let bad = m.beaten_by(m.idle_watts * 2.0, 0.0, work, period).unwrap();
        assert!(good.challenger_wins, "a quarter of the idle draw has to win: {good:?}");
        assert!(!bad.challenger_wins, "twice the idle draw cannot win on free compute: {bad:?}");

        // And the budget is what separated them, not the computation -- which was zero for both.
        assert!(m.idle_watts / 4.0 < good.standby_budget);
        assert!(m.idle_watts * 2.0 > bad.standby_budget);
    }

    #[test]
    fn free_sampling_cannot_rescue_a_device_that_must_stay_powered() {
        // Sharpen the previous test: hold standby just above the budget and drive compute to zero.
        // If sampling efficiency could win this, driving it to zero would.
        let m = gpu();
        let (work, period) = (1_000_000u64, 60.0);
        let budget = m.standby_budget(work, period).unwrap();
        let v = m.beaten_by(budget * 1.001, 0.0, work, period).unwrap();
        assert!(
            !v.challenger_wins,
            "0.1% over the budget with FREE computation still loses: {:.1} J vs {:.1} J",
            v.challenger_joules, v.incumbent_joules
        );
        // Just under, it wins -- so the budget really is the dividing line and not an artefact.
        let v2 = m.beaten_by(budget * 0.999, 0.0, work, period).unwrap();
        assert!(v2.challenger_wins, "0.1% under the budget wins: {v2:?}");
    }

    #[test]
    fn a_challenger_stated_in_nonsense_is_refused() {
        let m = gpu();
        assert!(matches!(m.beaten_by(f64::NAN, 0.0, 10, 1.0), Err(DutyError::NotPhysical(_))));
        assert!(matches!(m.beaten_by(-1.0, 0.0, 10, 1.0), Err(DutyError::NotPhysical(_))));
        assert!(matches!(m.beaten_by(1.0, f64::NAN, 10, 1.0), Err(DutyError::NotPhysical(_))));
        assert!(matches!(m.beaten_by(1.0, -1.0, 10, 1.0), Err(DutyError::NotPhysical(_))));
        // An unsustainable cadence is still refused through this path.
        assert!(matches!(
            m.beaten_by(1.0, 0.0, 10_000_000_000, 1.0),
            Err(DutyError::CannotSustain { .. })
        ));
    }

    #[test]
    fn unphysical_characterisations_are_refused() {
        assert!(matches!(Machine::new(f64::NAN, 1.0, 1.0), Err(DutyError::NotPhysical(_))));
        assert!(matches!(Machine::new(-1.0, 1.0, 1.0), Err(DutyError::NotPhysical(_))));
        // A negative delta is the meter's noise floor, not a generator.
        assert!(matches!(Machine::new(1.0, -0.5, 1.0), Err(DutyError::NotPhysical(_))));
        assert!(matches!(Machine::new(1.0, 1.0, 0.0), Err(DutyError::NotPhysical(_))));
        assert!(matches!(Machine::new(1.0, 1.0, f64::INFINITY), Err(DutyError::NotPhysical(_))));
        // Zero marginal is a real outcome: the workload did not clear the noise.
        let flat = Machine::new(20.0, 0.0, 1e9).unwrap();
        assert_eq!(flat.idle_dominant_below(), f64::INFINITY);
    }

    #[test]
    fn zero_work_and_impossible_periods_get_errors_not_numbers() {
        let m = gpu();
        assert!(matches!(m.duty(0, 1.0), Err(DutyError::Empty(_))));
        assert!(matches!(m.duty(10, 0.0), Err(DutyError::Empty(_))));
        assert!(matches!(m.duty(10, f64::NAN), Err(DutyError::Empty(_))));
    }
}
