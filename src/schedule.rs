//! What changes while a program runs.
//!
//! A program is a fixed thing: a graph, a coloring, a set of factors. Everything that *varies*
//! during a run — inverse temperature, and every penalty strength that gets ramped — lives here
//! instead, as numbers read at each stage.
//!
//! The rule this module exists to enforce: **annealing changes a number, never a program.** THRML
//! rebuilds its program at each of 4,000 annealing steps because beta is compiled into its weights;
//! our own DTM had the same defect until [`crate::kernel`] landed. A schedule makes the distinction
//! structural, and [`crate::graph::GRAPH_BUILDS`] makes it checkable.

/// Penalty strengths that a schedule may ramp.
///
/// These are the coefficients of constraint terms introduced by lowering passes, not by the user's
/// model. They start weak, so the sampler can move freely, and finish strong, so the constraint
/// actually binds. Ramping them is why they cannot live in compiled weights.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Penalties {
    /// Strength of the domain-wall constraint introduced when a categorical variable is lowered.
    pub domain_wall: f64,
    /// Strength of the agreement constraint between copies introduced by sparsification.
    pub copy: f64,
}

impl Default for Penalties {
    fn default() -> Self {
        Penalties { domain_wall: 1.0, copy: 1.0 }
    }
}

/// One rung: run `sweeps` sweeps at this temperature and these penalty strengths.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Stage {
    pub beta: f64,
    pub sweeps: usize,
    pub penalties: Penalties,
}

/// An ordered list of stages.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Schedule {
    stages: Vec<Stage>,
}

impl Schedule {
    pub fn new() -> Self {
        Schedule { stages: Vec::new() }
    }

    /// One temperature, held.
    pub fn constant(beta: f64, sweeps: usize) -> Self {
        Schedule { stages: vec![Stage { beta, sweeps, penalties: Penalties::default() }] }
    }

    /// A geometric ladder from `beta_min` to `beta_max` over `stages` rungs.
    ///
    /// Geometric rather than linear because the interesting physics is spread evenly in log beta,
    /// not in beta: a linear ladder spends most of its stages in the cold, already-frozen regime.
    pub fn geometric(beta_min: f64, beta_max: f64, stages: usize, sweeps_per: usize) -> Self {
        assert!(beta_min > 0.0, "beta_min must be positive; a geometric ladder cannot start at 0");
        assert!(beta_max > beta_min, "need beta_max > beta_min");
        assert!(stages >= 2, "a ladder needs at least 2 rungs");
        let r = (beta_max / beta_min).powf(1.0 / (stages - 1) as f64);
        Schedule {
            stages: (0..stages)
                .map(|i| Stage {
                    beta: beta_min * r.powi(i as i32),
                    sweeps: sweeps_per,
                    penalties: Penalties::default(),
                })
                .collect(),
        }
    }

    /// Ramp a penalty geometrically from `start` to `end` across the existing stages.
    ///
    /// Applied after the temperature ladder, so the two are specified independently.
    pub fn ramp_domain_wall(mut self, start: f64, end: f64) -> Self {
        for (i, s) in ramp(start, end, self.stages.len()).into_iter().enumerate() {
            self.stages[i].penalties.domain_wall = s;
        }
        self
    }

    /// Ramp the copy-agreement penalty geometrically across the existing stages.
    pub fn ramp_copy(mut self, start: f64, end: f64) -> Self {
        for (i, s) in ramp(start, end, self.stages.len()).into_iter().enumerate() {
            self.stages[i].penalties.copy = s;
        }
        self
    }

    pub fn push(&mut self, stage: Stage) {
        self.stages.push(stage);
    }

    pub fn stages(&self) -> &[Stage] {
        &self.stages
    }

    pub fn len(&self) -> usize {
        self.stages.len()
    }

    pub fn is_empty(&self) -> bool {
        self.stages.is_empty()
    }

    /// Total sweeps across every stage, for sizing a run before starting it.
    pub fn total_sweeps(&self) -> u64 {
        self.stages.iter().map(|s| s.sweeps as u64).sum()
    }

    /// Node updates this schedule will charge for a graph of `n` nodes.
    pub fn node_updates(&self, n: usize) -> u64 {
        self.total_sweeps() * n as u64
    }
}

fn ramp(start: f64, end: f64, n: usize) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 || start == end {
        return vec![end; n];
    }
    if start > 0.0 && end > 0.0 {
        let r = (end / start).powf(1.0 / (n - 1) as f64);
        (0..n).map(|i| start * r.powi(i as i32)).collect()
    } else {
        // a geometric ramp through zero is undefined; fall back to linear rather than emit NaN
        let step = (end - start) / (n - 1) as f64;
        (0..n).map(|i| start + step * i as f64).collect()
    }
}

impl From<&[(f64, usize)]> for Schedule {
    fn from(v: &[(f64, usize)]) -> Self {
        Schedule {
            stages: v
                .iter()
                .map(|&(beta, sweeps)| Stage { beta, sweeps, penalties: Penalties::default() })
                .collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geometric_ladder_hits_both_ends() {
        let s = Schedule::geometric(0.05, 4.0, 40, 10);
        assert_eq!(s.len(), 40);
        assert!((s.stages()[0].beta - 0.05).abs() < 1e-12);
        assert!((s.stages()[39].beta - 4.0).abs() < 1e-12);
        // monotone, and evenly spaced in log beta
        let st = s.stages();
        let r0 = st[1].beta / st[0].beta;
        for w in st.windows(2) {
            assert!(w[1].beta > w[0].beta);
            assert!((w[1].beta / w[0].beta - r0).abs() < 1e-12);
        }
    }

    #[test]
    fn penalties_ramp_independently_of_temperature() {
        let s = Schedule::geometric(0.1, 2.0, 10, 5).ramp_domain_wall(0.5, 8.0).ramp_copy(1.0, 4.0);
        let st = s.stages();
        assert!((st[0].penalties.domain_wall - 0.5).abs() < 1e-12);
        assert!((st[9].penalties.domain_wall - 8.0).abs() < 1e-12);
        assert!((st[0].penalties.copy - 1.0).abs() < 1e-12);
        assert!((st[9].penalties.copy - 4.0).abs() < 1e-12);
        // the temperature ladder is untouched by either ramp
        assert!((st[0].beta - 0.1).abs() < 1e-12);
        assert!((st[9].beta - 2.0).abs() < 1e-12);
    }

    #[test]
    fn a_ramp_through_zero_does_not_produce_nan() {
        let s = Schedule::geometric(0.1, 1.0, 5, 1).ramp_domain_wall(0.0, 4.0);
        for st in s.stages() {
            assert!(st.penalties.domain_wall.is_finite(), "{:?}", st.penalties);
        }
    }

    #[test]
    fn sizing_a_run_before_starting_it() {
        let s = Schedule::geometric(0.1, 2.0, 40, 25);
        assert_eq!(s.total_sweeps(), 1000);
        assert_eq!(s.node_updates(4900), 4_900_000);
    }

    #[test]
    fn degenerate_ladders_are_rejected_loudly() {
        assert!(std::panic::catch_unwind(|| Schedule::geometric(0.0, 1.0, 4, 1)).is_err());
        assert!(std::panic::catch_unwind(|| Schedule::geometric(1.0, 0.5, 4, 1)).is_err());
        assert!(std::panic::catch_unwind(|| Schedule::geometric(0.1, 1.0, 1, 1)).is_err());
    }
}
