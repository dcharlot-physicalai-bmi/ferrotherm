//! What changes while a program runs.
//!
//! A program is a fixed thing: a graph, a coloring, a set of factors. Everything that *varies*
//! during a run — inverse temperature, and every penalty strength that gets ramped — lives here
//! instead, as numbers read at each stage.
//!
//! The rule this module exists to enforce: **annealing changes a number, never a program.** THRML
//! rebuilds its program at each of 4,000 annealing steps because beta is compiled into its weights;
//! our own DTM had the same defect until [`crate::kernel`] landed. A schedule makes the distinction
//! structural, and [`crate::graph::graph_builds`] makes it checkable.

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
    /// Inverse temperature held for this stage.
    pub beta: f64,
    /// Sweeps run before the schedule advances.
    pub sweeps: usize,
    /// Constraint penalty weights in force during it.
    pub penalties: Penalties,
}

/// This graph has no energy scale, so no temperature can be measured against it.
///
/// Returned rather than panicking because an empty or uncoupled graph is a thing a caller can
/// legitimately hand over — a model that compiled to nothing, a subproblem after presolve — and
/// "there is nothing to anneal" is a better answer than a ladder of infinities.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NoScale;

impl core::fmt::Display for NoScale {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(
            "this graph has no couplings and no fields, so it has no energy scale for a \
             temperature to be measured against",
        )
    }
}

impl core::error::Error for NoScale {}

/// An ordered list of stages.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Schedule {
    stages: Vec<Stage>,
}

impl Schedule {
    #[must_use]
    /// An empty schedule with no stages.
    pub fn new() -> Self {
        Schedule { stages: Vec::new() }
    }

    /// One temperature, held.
    #[must_use]
    pub fn constant(beta: f64, sweeps: usize) -> Self {
        Schedule { stages: vec![Stage { beta, sweeps, penalties: Penalties::default() }] }
    }

    /// A geometric ladder from `beta_min` to `beta_max` over `stages` rungs.
    ///
    /// Geometric rather than linear because the interesting physics is spread evenly in log beta,
    /// not in beta: a linear ladder spends most of its stages in the cold, already-frozen regime.
    ///
    /// # Panics
    ///
    /// If `beta_min` is not positive -- a geometric ladder cannot start at zero -- or `beta_max` is not
    /// above it.
    #[must_use]
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

    /// The energy scale the shipped default ladder is written against.
    ///
    /// `lattice2d(L, 1.0)` has [`crate::graph::Graph::flip_gap_max`] of exactly `8.0` at every `L`
    /// — four unit couplings, no field — and it is the family this crate's flagship verification
    /// (`examples/onsager.rs`) is built on. Naming it here makes the anchor a citation rather than
    /// a magic number: [`Self::DEFAULT_LADDER_HAT`] divided by this reproduces
    /// `Compiled::DEFAULT_LADDER`'s `0.05 .. 8.0` exactly, so nothing moves for anyone whose
    /// instance is at that scale.
    ///
    /// It is a CONVENTION, not a derivation. No record survives of which family the `0.05 .. 8.0`
    /// ladder was tuned on; this is the anchor that reproduces it on the family the crate verifies
    /// against, chosen for that reason and written down so the next person does not have to guess.
    pub const REFERENCE_SCALE: f64 = 8.0;

    /// The default ladder in units of the instance's own energy scale: `(beta_lo_hat,
    /// beta_hi_hat, stages, sweeps_per)`.
    ///
    /// Same ladder, new units. `0.05 * 8.0 = 0.4` and `8.0 * 8.0 = 64.0`.
    pub const DEFAULT_LADDER_HAT: (f64, f64, usize, usize) = (0.4, 64.0, 120, 40);

    /// A geometric ladder in units of `g`'s energy scale, so the same problem in different units
    /// gets the same answer.
    ///
    /// `beta` and energy enter the Boltzmann weight only as `beta * E`, so a ladder in absolute
    /// `beta` is a claim about the units the modeller happened to write in. Measured on
    /// `planted::frustrated_loops(8, 96, 3)`, worst excess over the planted optimum across five
    /// seeds:
    ///
    /// ```text
    ///   scale     fixed ladder     this ladder
    ///    1e-3           52.08%           2.08%
    ///    1e-2           43.75%           2.08%
    ///     1e0            2.08%           2.08%
    ///     1e2           20.83%           2.08%
    ///     1e3           20.83%           2.08%
    /// ```
    ///
    /// # Errors
    ///
    /// [`NoScale`] when the graph has no couplings and no fields. Such a graph has no energy scale
    /// to measure a temperature against, every state has energy zero, and every derived `beta`
    /// would be infinite — so there is nothing to anneal and saying so beats dividing by zero.
    pub fn geometric_dimensionless(
        bhat_lo: f64,
        bhat_hi: f64,
        stages: usize,
        per: usize,
        g: &crate::graph::Graph,
    ) -> Result<Self, NoScale> {
        let scale = g.flip_gap_max().ok_or(NoScale)?;
        Ok(Self::geometric(bhat_lo / scale, bhat_hi / scale, stages, per))
    }

    /// [`Self::DEFAULT_LADDER_HAT`], measured against this instance's own energy scale.
    ///
    /// # Errors
    ///
    /// [`NoScale`], as [`Self::geometric_dimensionless`].
    pub fn for_instance(g: &crate::graph::Graph, stages: usize, per: usize) -> Result<Self, NoScale> {
        let (lo, hi, _, _) = Self::DEFAULT_LADDER_HAT;
        Self::geometric_dimensionless(lo, hi, stages, per, g)
    }

    /// The coldest and hottest inverse temperatures on this ladder, or `None` if it has no stages.
    ///
    /// The ends rather than the whole list, because that is what a caller comparing two ladders —
    /// or reporting which one a run used — actually needs.
    #[must_use]
    pub fn ends(&self) -> Option<(f64, f64)> {
        let first = self.stages.first()?.beta;
        let last = self.stages.last()?.beta;
        Some((first.min(last), first.max(last)))
    }

    /// Ramp a penalty geometrically from `start` to `end` across the existing stages.
    ///
    /// Applied after the temperature ladder, so the two are specified independently.
    #[must_use]
    pub fn ramp_domain_wall(mut self, start: f64, end: f64) -> Self {
        for (i, s) in ramp(start, end, self.stages.len()).into_iter().enumerate() {
            self.stages[i].penalties.domain_wall = s;
        }
        self
    }

    /// Ramp the copy-agreement penalty geometrically across the existing stages.
    #[must_use]
    pub fn ramp_copy(mut self, start: f64, end: f64) -> Self {
        for (i, s) in ramp(start, end, self.stages.len()).into_iter().enumerate() {
            self.stages[i].penalties.copy = s;
        }
        self
    }

    /// Append a stage, which runs after every stage already present.
    pub fn push(&mut self, stage: Stage) {
        self.stages.push(stage);
    }

    #[must_use]
    /// The stages, in the order they run.
    pub fn stages(&self) -> &[Stage] {
        &self.stages
    }

    #[must_use]
    /// How many stages the schedule has.
    pub fn len(&self) -> usize {
        self.stages.len()
    }

    #[must_use]
    /// Whether it has none.
    pub fn is_empty(&self) -> bool {
        self.stages.is_empty()
    }

    /// Total sweeps across every stage, for sizing a run before starting it.
    #[must_use]
    pub fn total_sweeps(&self) -> u64 {
        self.stages.iter().map(|s| s.sweeps as u64).sum()
    }

    /// Node updates this schedule will charge for a graph of `n` nodes.
    #[must_use]
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
mod scale_invariance {
    use super::*;
    use crate::graph::{rescaled, Graph};

    fn families() -> Vec<(&'static str, Graph)> {
        vec![
            ("lattice2d(6, 1.0)", crate::ising::lattice2d(6, 1.0)),
            // WITH FIELDS. `rescaled` scales couplings AND fields; `adaptive::scaled` scales only
            // couplings, so a family with h == 0 cannot tell the two transformations apart and a
            // test built only on lattices would pass for either.
            ("ring(12, 1.0, 0.3)", crate::ising::ring(12, 1.0, 0.3)),
            ("frustrated_loops(6, 40, 3)", crate::planted::frustrated_loops(6, 40, 3).graph),
        ]
    }

    /// Scaling the Hamiltonian by `k` and the ladder by `1/k` is the identity, BIT FOR BIT.
    ///
    /// Not an approximation and not a statistical claim. `beta` and energy meet only as `beta * E`;
    /// for `k` a power of two every multiplication involved is exact in IEEE-754, so the local
    /// field scales exactly, `beta * field` is unchanged exactly, the acceptance probability is the
    /// same bit pattern, the same random draws are compared against it, and the trajectory is
    /// identical. Anything less than `assert_eq!` here would be hiding a real difference behind a
    /// tolerance.
    #[test]
    fn a_power_of_two_change_of_units_changes_nothing() {
        for (name, g) in families() {
            let base = Schedule::for_instance(&g, 20, 5).expect("these families have a scale");
            for k in [0.25f64, 0.5, 2.0, 4.0, 1024.0] {
                let gk = rescaled(&g, k);
                let sk = Schedule::for_instance(&gk, 20, 5).expect("scaling preserves the scale");

                // The ladder itself is exactly 1/k times the original.
                for (a, b) in base.stages().iter().zip(sk.stages()) {
                    assert_eq!(a.beta / k, b.beta, "{name} at k={k}: ladder is not exactly scaled");
                }

                let (s0, e0) = crate::tempering::anneal_scheduled(&g, &base, 7, None);
                let (s1, e1) = crate::tempering::anneal_scheduled(&gk, &sk, 7, None);
                assert_eq!(s0, s1, "{name} at k={k}: a change of units moved the answer");
                assert_eq!(e0 * k, e1, "{name} at k={k}: energy did not scale exactly");
            }
        }
    }

    /// And the anchor reproduces the shipped ladder exactly on the family it is anchored to.
    ///
    /// If this fails, `REFERENCE_SCALE` has drifted from `Compiled::DEFAULT_LADDER` and every
    /// instance at the reference scale would silently get a different ladder than it does today.
    #[test]
    fn the_reference_scale_reproduces_the_shipped_ladder() {
        let g = crate::ising::lattice2d(8, 1.0);
        assert_eq!(g.flip_gap_max(), Some(Schedule::REFERENCE_SCALE));

        let (lo, hi, stages, per) = crate::model::Compiled::DEFAULT_LADDER;
        let derived = Schedule::for_instance(&g, stages, per).expect("a lattice has a scale");
        let shipped = Schedule::geometric(lo, hi, stages, per);
        assert_eq!(derived.stages().len(), shipped.stages().len());
        for (a, b) in derived.stages().iter().zip(shipped.stages()) {
            assert_eq!(a.beta, b.beta, "the derived ladder must BE the shipped one here");
        }
    }

    /// A graph with nothing in it has no scale, and says so rather than dividing by zero.
    #[test]
    fn a_graph_with_no_energy_scale_is_refused() {
        let empty = crate::graph::GraphBuilder::new(4).build();
        assert_eq!(empty.flip_gap_max(), None);
        assert_eq!(Schedule::for_instance(&empty, 10, 5), Err(NoScale));
    }

    /// The measured table in `geometric_dimensionless`'s doc, as a test.
    ///
    /// The fixed ladder must be BAD at the extremes and the derived ladder must be flat. Asserting
    /// only that the derived one is good would pass if scale-dependence had never existed, so the
    /// first half is what makes this about the defect.
    #[test]
    fn the_derived_ladder_is_flat_where_the_fixed_one_is_not() {
        let p = crate::planted::frustrated_loops(8, 96, 3);
        let (_, _, stages, per) = crate::model::Compiled::DEFAULT_LADDER;
        let fixed = crate::model::Compiled::default_schedule();

        let worst = |g: &Graph, sched: &Schedule| -> f64 {
            (0..5u64)
                .map(|seed| p.excess(&crate::tempering::anneal_scheduled(g, sched, seed, None).0))
                .fold(0.0f64, f64::max)
        };

        let at_unit = worst(&p.graph, &fixed);
        for k in [1e-3, 1e3] {
            let gk = rescaled(&p.graph, k);
            let d = Schedule::for_instance(&gk, stages, per).expect("a scale");
            assert!(
                worst(&gk, &fixed) > 5.0 * at_unit,
                "the fixed ladder is supposed to be much worse at k={k}; if this fails the \
                 defect is gone and the rest of this test proves nothing"
            );
            assert_eq!(
                worst(&gk, &d),
                at_unit,
                "the derived ladder must give the SAME answer at k={k} as at unit scale"
            );
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
