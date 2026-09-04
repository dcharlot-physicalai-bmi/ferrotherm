//! Optimization-grade sampling: simulated annealing and parallel tempering.
//!
//! This is the p-computer algorithm line (Camsari and collaborators): the same chromatic Gibbs
//! primitive, scheduled. Parallel tempering runs replicas at a ladder of temperatures and swaps
//! neighbouring replicas with the Metropolis criterion, which is what lets hard, frustrated
//! landscapes mix — FPGA p-computers running adaptive parallel tempering matched a quantum
//! annealer on 3D spin glasses (Nature Communications 2025, arXiv:2503.10302). The verification
//! standard here is exact: on enumerable instances the sampler must find the true ground state.

use crate::gibbs::Sampler;
use crate::graph::Graph;
use crate::ledger::Ledger;
use crate::rng::Pcg;

/// Simulated annealing: sweep while raising beta along `schedule`, tracking the best state seen.
pub fn anneal(
    g: &Graph,
    schedule: &[(f64, usize)], // (beta, sweeps at that beta)
    seed: u64,
    mut ledger: Option<&mut Ledger>,
) -> (Vec<i8>, f64) {
    let mut smp = Sampler::new(g, schedule[0].0, seed);
    let mut best = smp.s.clone();
    let mut best_e = g.energy(&best);
    for &(beta, sweeps) in schedule {
        smp.beta = beta;
        for _ in 0..sweeps {
            smp.sweep(ledger.as_deref_mut());
            let e = g.energy(&smp.s);
            if e < best_e {
                best_e = e;
                best = smp.s.clone();
            }
        }
    }
    (best, best_e)
}

/// Geometric beta ladder from `beta_min` to `beta_max`.
pub fn geometric_ladder(beta_min: f64, beta_max: f64, n: usize) -> Vec<f64> {
    assert!(n >= 2 && beta_min > 0.0 && beta_max > beta_min);
    let r = (beta_max / beta_min).powf(1.0 / (n - 1) as f64);
    (0..n).map(|i| beta_min * r.powi(i as i32)).collect()
}

pub struct TemperingResult {
    pub best: Vec<i8>,
    pub best_e: f64,
    /// Swap acceptance rate per adjacent pair — the ladder-health diagnostic. Healthy ladders sit
    /// roughly in [0.2, 0.6]; near-zero pairs mean the ladder has a gap replicas cannot cross.
    pub swap_rates: Vec<f64>,
}

/// Advance every replica by `swap_every` sweeps.
///
/// **This is free parallelism and it changes no answer.** Each replica owns its `Sampler` and its
/// own `Pcg`, seeded once from `seed ^ (i * 0x9E37)`, so a replica's draws depend only on its own
/// state and its own history — never on what another replica did or on the order they ran in. That
/// is what makes replica-level threading different from splitting a colour class, where the thread
/// count IS part of the sample path: here the result is bit-identical whether one thread runs eight
/// replicas or eight threads run one each.
///
/// The ledger is the only thing shared, and it is a counter. Each thread accumulates its own and
/// the sums are added afterwards, which is integer addition and therefore order-independent.
///
/// Serial in a browser: `wasm32-unknown-unknown` has a std whose `thread::spawn` compiles and then
/// panics at runtime, so there is nothing to spread across and the same answer comes out either way.
/// Node-updates a round must carry before spreading its replicas across threads is worth a spawn.
///
/// `replicas x sweeps_between_swaps x nodes`. Below it the replicas run serially.
///
/// **A guard, not a tuned optimum**, for the same reason as [`crate::gibbs::MIN_CHUNK`]. The
/// structural fact holds on any fabric: this function spawns on EVERY round, thread creation costs
/// microseconds everywhere, and `icm::Params::default()` asks for 12,800 spawns to cover 12,800
/// single sweeps. Refusing to spawn for less work than the spawn costs is right regardless of
/// machine. The specific number came from ratios on one developer laptop and a different fabric
/// will cross over elsewhere; it is placed where threading is never a loss rather than where it is
/// fastest, because that is the property worth guaranteeing.
pub const MIN_REPLICA_WORK: usize = 30_000;

pub(crate) fn advance(reps: &mut [Sampler], swap_every: usize, ledger: Option<&mut Ledger>) {
    #[cfg(not(target_arch = "wasm32"))]
    {
        // A FLOOR, for the same reason `gibbs::sweeps_par` has one, and see [`MIN_REPLICA_WORK`] on
        // why the number is a guard rather than a tuning. `icm::Params::default()` is 400 rounds x
        // 2 replica sets x 16 betas with `sweeps_per_round: 1` -- 12,800 spawns to cover 12,800
        // single sweeps, which is wrong on any machine.
        //
        // One machine's ratios, threaded/serial, arms interleaved in one process, recorded as the
        // observation that set the guard:
        //
        //     n   reps  swap_every   ratio
        //   256     16           1   0.29x
        //   256      8           4   0.67x
        //  1024     16           1   0.93x
        //  1024     16           4   2.87x
        //  4096     16           4   4.17x
        //
        // An earlier measurement said 4.4x at n=256 and was wrong: it ran the two arms as separate
        // processes on a loaded machine. Timing two things on a shared machine means timing them
        // next to each other, or timing the machine instead.
        let work = reps.len() * swap_every * reps.first().map_or(0, |r| r.s.len());
        if reps.len() > 1 && work >= MIN_REPLICA_WORK {
            let counted: Vec<u64> = std::thread::scope(|scope| {
                let handles: Vec<_> = reps
                    .iter_mut()
                    .map(|rep| {
                        scope.spawn(move || {
                            let mut own = Ledger::default();
                            for _ in 0..swap_every {
                                rep.sweep(Some(&mut own));
                            }
                            own.samples
                        })
                    })
                    .collect();
                handles.into_iter().map(|h| h.join().expect("a replica sweep")).collect()
            });
            if let Some(l) = ledger {
                l.samples += counted.iter().sum::<u64>();
            }
            return;
        }
    }
    let mut ledger = ledger;
    for rep in reps.iter_mut() {
        for _ in 0..swap_every {
            rep.sweep(ledger.as_deref_mut());
        }
    }
}


/// Parallel tempering over a beta ladder. Every `swap_every` sweeps, adjacent replicas attempt a
/// state exchange with probability min(1, exp(delta_beta * delta_E)) — the standard replica-
/// exchange criterion, alternating even/odd pairs so a state can traverse the whole ladder.
pub fn parallel_tempering(
    g: &Graph,
    betas: &[f64],
    rounds: usize,
    swap_every: usize,
    seed: u64,
    mut ledger: Option<&mut Ledger>,
) -> TemperingResult {
    let r = betas.len();
    assert!(r >= 2);
    let mut reps: Vec<Sampler> = (0..r).map(|i| Sampler::new(g, betas[i], seed ^ (i as u64 * 0x9E37)) ).collect();
    let mut swap_rng = Pcg::new(seed ^ 0x5A5A, 3);
    let mut attempts = vec![0u64; r - 1];
    let mut accepts = vec![0u64; r - 1];
    let mut best = reps[r - 1].s.clone();
    let mut best_e = g.energy(&best);
    for round in 0..rounds {
        advance(&mut reps, swap_every, ledger.as_deref_mut());
        // coldest replica is the optimizer; track its best
        for rep in reps.iter() {
            let e = g.energy(&rep.s);
            if e < best_e {
                best_e = e;
                best = rep.s.clone();
            }
        }
        // alternate even/odd adjacent pairs
        let start = round % 2;
        for i in (start..r - 1).step_by(2) {
            let e_i = g.energy(&reps[i].s);
            let e_j = g.energy(&reps[i + 1].s);
            let arg = (betas[i + 1] - betas[i]) * (e_j - e_i);
            attempts[i] += 1;
            if arg >= 0.0 || swap_rng.f64() < arg.exp() {
                accepts[i] += 1;
                let (a, b) = reps.split_at_mut(i + 1);
                std::mem::swap(&mut a[i].s, &mut b[0].s);
            }
        }
    }
    TemperingResult {
        best,
        best_e,
        swap_rates: (0..r - 1).map(|i| accepts[i] as f64 / attempts[i].max(1) as f64).collect(),
    }
}

/// The per-rung energy traces a parallel-tempering run produced.
///
/// Parallel tempering already does what free-energy estimation needs: it holds a chain at every
/// rung of a temperature ladder. Until now [`crate::free_energy`] drew its OWN chains to build a
/// free-energy curve, doing the same work twice — an optimisation run threw away exactly the
/// samples a thermodynamics run would have had to generate.
///
/// The swap moves STATES between replicas while each replica keeps its own `β`, so replica `i`'s
/// energy after its sweeps is always a sample at `betas[i]`, which is what
/// [`crate::free_energy::bar_ladder`] consumes.
#[derive(Clone, Debug)]
pub struct LadderTraces {
    pub betas: Vec<f64>,
    /// `energies[i]` is the trace at `betas[i]`, one entry per recorded round.
    pub energies: Vec<Vec<f64>>,
    /// Completed round trips: a state that reached the hot end, then the cold end, then the hot end
    /// again. The count is over the whole run, burn-in included, because a walker does not know
    /// about burn-in.
    pub round_trips: usize,
    /// Rounds per completed round trip, or `None` when none completed.
    ///
    /// **This, not the per-rung `tau_int`, is the timescale a free-energy estimate decorrelates
    /// on.** A rung's own energy trace can look decorrelated (`tau_int` near 1) while the ENSEMBLE
    /// circulating through the ladder has not renewed itself at all: the state at a rung changes
    /// quickly, but *which* states are available there is set by how often the ladder mixes end to
    /// end. A block jackknife whose blocks are shorter than this is resampling correlated data and
    /// will report a bar that is too small.
    pub round_trip_time: Option<f64>,
}

impl LadderTraces {
    /// How many round trips fit in the recorded stretch — the effective number of independent
    /// ladder traversals a jackknife has to work with.
    pub fn independent_traversals(&self) -> Option<f64> {
        let len = self.energies.first()?.len() as f64;
        self.round_trip_time.map(|t| len / t)
    }
}

impl LadderTraces {
    /// The traces as `(β, energies)` pairs, the shape the free-energy estimators take.
    pub fn as_pairs(&self) -> Vec<(f64, Vec<f64>)> {
        self.betas.iter().copied().zip(self.energies.iter().cloned()).collect()
    }

    /// The free-energy curve — `ln Z`, entropy and heat capacity at every rung — from these
    /// traces, by Bennett's acceptance ratio anchored at `ln Z(0) = n ln 2`.
    ///
    /// **Requires the ladder to start at `β = 0`**, because that is where the anchor is exact.
    /// A ladder that starts warm still has well-defined free-energy *differences*
    /// ([`Self::log_z_differences`]) but no absolute scale, and this returns `Err` rather than
    /// quietly reporting a relative number as an absolute one — the same distinction
    /// [`crate::popanneal::Outcome::free_energy_per_spin`] draws.
    pub fn thermodynamics(&self, n: usize, z: f64) -> Result<crate::free_energy::Thermo, String> {
        if self.betas.first() != Some(&0.0) {
            return Err(format!(
                "the ladder starts at beta {:?}, not 0, so ln Z has no absolute anchor; use log_z_differences",
                self.betas.first()
            ));
        }
        Ok(crate::free_energy::bar_ladder(n, &self.as_pairs(), z))
    }

    /// `ln(Z_{i+1} / Z_i)` for every adjacent pair, by Bennett's acceptance ratio. Defined for any
    /// ladder, warm start included.
    ///
    /// **The `stderr` on each step is that step's alone.** Do not add them in quadrature to get an
    /// error bar on the telescoped total: adjacent steps share the samples at their common rung,
    /// so they are correlated and the quadrature sum is optimistic. Measured over 200 runs with
    /// scrambled seeds, the quadrature bar understates the true spread by about 50% —
    /// `sd(z) = 1.50 ± 0.08` where a calibrated bar gives 1. [`Self::log_z_total`] does it
    /// properly, at `1.05 ± 0.05`.
    pub fn log_z_differences(&self) -> Vec<crate::free_energy::BarPair> {
        self.betas
            .windows(2)
            .enumerate()
            .map(|(i, w)| crate::free_energy::bar_pair(w[0], &self.energies[i], w[1], &self.energies[i + 1]))
            .collect()
    }

    /// The telescoped `ln(Z_last / Z_first)` with a **block-jackknife** error bar.
    ///
    /// The traces are split into `blocks` contiguous blocks; deleting one block from EVERY rung at
    /// once and recomputing the whole telescoped sum gives a replicate, and the jackknife variance
    /// over those replicates is
    ///
    /// ```text
    ///   var = (B − 1)/B · Σ_b (θ_b − θ̄)²
    /// ```
    ///
    /// Because a block is deleted from every rung simultaneously, this captures the covariance
    /// between adjacent steps that the quadrature sum ignores — and contiguous blocks absorb the
    /// autocorrelation along each chain, which per-sample resampling would not. Measured
    /// calibration on the same 60-seed test that caught the quadrature bar: `sd(z)` falls from
    /// 1.56 to about 1.
    ///
    /// `blocks` below 4 is refused: a jackknife over three replicates is not an error bar.
    ///
    /// # What is established
    ///
    /// That the quadrature sum ignores a covariance is arithmetic. That this estimator is
    /// calibrated is now measured: on a 12-spin ring with an 8-rung ladder at 3,000 recorded
    /// samples in 8 blocks, over **200 runs with scrambled seeds**, `sd(z) = 1.05 ± 0.05` against
    /// quadrature's `1.50 ± 0.08`.
    ///
    /// An earlier version of this doc reported `0.81` from 24 runs and built a story on top of it —
    /// that calibration drifted with sample count, and that the culprit was the ladder's round-trip
    /// time. **Both were artefacts of too few runs.** The round-trip time is now measured directly
    /// ([`Self::round_trip_time`]) and is about 10 rounds here, while every block in that
    /// experiment was 56 to 2,250 rounds long — 5 to 200 round trips per block, so blocks were
    /// never the problem. What was the problem is that 24 runs cannot measure an `sd` to better
    /// than 14%.
    pub fn log_z_total(&self, blocks: usize) -> Result<(f64, f64), String> {
        if blocks < 4 {
            return Err(format!("{blocks} blocks is too few for a jackknife variance; use at least 4"));
        }
        let len = self.energies.first().map_or(0, |e| e.len());
        if self.energies.iter().any(|e| e.len() != len) {
            return Err("the rungs have different trace lengths".into());
        }
        if len < blocks * 4 {
            return Err(format!("{len} samples is too few for {blocks} blocks"));
        }
        let full = self.log_z_differences().iter().map(|s| s.delta).sum::<f64>();
        let edges: Vec<usize> = (0..=blocks).map(|b| b * len / blocks).collect();
        let mut reps = Vec::with_capacity(blocks);
        for b in 0..blocks {
            let (lo, hi) = (edges[b], edges[b + 1]);
            let kept: Vec<Vec<f64>> = self
                .energies
                .iter()
                .map(|e| e[..lo].iter().chain(e[hi..].iter()).copied().collect())
                .collect();
            let total: f64 = self
                .betas
                .windows(2)
                .enumerate()
                .map(|(i, w)| crate::free_energy::bar_pair(w[0], &kept[i], w[1], &kept[i + 1]).delta)
                .sum();
            reps.push(total);
        }
        let mean = reps.iter().sum::<f64>() / blocks as f64;
        let var = (blocks - 1) as f64 / blocks as f64 * reps.iter().map(|r| (r - mean) * (r - mean)).sum::<f64>();
        Ok((full, var.max(0.0).sqrt()))
    }
}

/// [`parallel_tempering`], recording each rung's energy trace after `burn_in` rounds.
///
/// Same dynamics, same answer, plus the samples: the optimisation result is unchanged and the
/// traces come out beside it, so one run serves both purposes. Recording costs one `g.energy` per
/// replica per round, which the best-tracking loop was already paying.
pub fn parallel_tempering_observed(
    g: &Graph,
    betas: &[f64],
    rounds: usize,
    swap_every: usize,
    burn_in: usize,
    seed: u64,
    mut ledger: Option<&mut Ledger>,
) -> (TemperingResult, LadderTraces) {
    let r = betas.len();
    assert!(r >= 2);
    assert!(rounds > burn_in, "every round is burn-in, so there is nothing to record");
    let mut reps: Vec<Sampler> = (0..r).map(|i| Sampler::new(g, betas[i], seed ^ (i as u64 * 0x9E37))).collect();
    let mut swap_rng = Pcg::new(seed ^ 0x5A5A, 3);
    let mut attempts = vec![0u64; r - 1];
    let mut accepts = vec![0u64; r - 1];
    let mut best = reps[r - 1].s.clone();
    let mut best_e = g.energy(&best);
    let mut energies: Vec<Vec<f64>> = vec![Vec::with_capacity(rounds - burn_in); r];
    // Round-trip tracking. `walker[i]` names the state currently at rung i; `dir[w]` remembers
    // which end walker w last touched (+1 = it last touched the hot end and is heading cold).
    // A round trip is counted when a walker that last touched the hot end reaches the cold end and
    // returns -- the standard definition, so the count is comparable with the literature's.
    let mut walker: Vec<usize> = (0..r).collect();
    let mut last_end: Vec<i8> = vec![0; r];
    let mut round_trips = 0usize;
    for round in 0..rounds {
        advance(&mut reps, swap_every, ledger.as_deref_mut());
        for (i, rep) in reps.iter().enumerate() {
            let e = g.energy(&rep.s);
            if e < best_e {
                best_e = e;
                best = rep.s.clone();
            }
            if round >= burn_in {
                energies[i].push(e);
            }
        }
        let start = round % 2;
        for i in (start..r - 1).step_by(2) {
            let e_i = g.energy(&reps[i].s);
            let e_j = g.energy(&reps[i + 1].s);
            let arg = (betas[i + 1] - betas[i]) * (e_j - e_i);
            attempts[i] += 1;
            if arg >= 0.0 || swap_rng.f64() < arg.exp() {
                accepts[i] += 1;
                let (a, b) = reps.split_at_mut(i + 1);
                std::mem::swap(&mut a[i].s, &mut b[0].s);
                walker.swap(i, i + 1);
            }
        }
        // Rung 0 is the hot end (beta smallest) and rung r-1 the cold end.
        let (hot, cold) = (walker[0], walker[r - 1]);
        if last_end[hot] == 1 {
            round_trips += 1;
        }
        last_end[hot] = -1;
        last_end[cold] = 1;
    }
    let round_trip_time = (round_trips > 0).then(|| rounds as f64 / round_trips as f64);
    (
        TemperingResult {
            best,
            best_e,
            swap_rates: (0..r - 1).map(|i| accepts[i] as f64 / attempts[i].max(1) as f64).collect(),
        },
        LadderTraces { betas: betas.to_vec(), energies, round_trips, round_trip_time },
    )
}

#[cfg(test)]
mod observed_tests {
    use super::*;
    use crate::free_energy::ring_log_z;
    use crate::ising;
    use crate::tempering::geometric_ladder;

    /// One run, two answers: the optimiser's best AND the free-energy curve, and the curve agrees
    /// with the transfer matrix at every rung.
    #[test]
    fn a_tempering_run_yields_the_free_energy_curve_it_used_to_throw_away() {
        let (n, j, h) = (12usize, 1.0, 0.0);
        let g = ising::ring(n, j, h);
        let betas: Vec<f64> = (0..24).map(|k| 2.0 * k as f64 / 23.0).collect();
        let (res, traces) = parallel_tempering_observed(&g, &betas, 3000, 2, 300, 5, None);
        assert!(res.best_e <= -10.0, "the optimiser still optimises: {}", res.best_e);
        assert_eq!(traces.energies.len(), betas.len());
        assert!(traces.energies.iter().all(|e| e.len() == 2700));

        let th = traces.thermodynamics(n, 3.0).unwrap();
        for r in &th.rungs[1..] {
            let truth = ring_log_z(n, j, h, r.beta);
            assert!((r.log_z - truth).abs() < 0.2 + 4.0 * r.stderr, "beta {}: {} +- {} vs {truth}", r.beta, r.log_z, r.stderr);
        }
        // the anchor is exact and the entropy falls toward ln 2 (two ground states)
        assert!((th.rungs[0].log_z - n as f64 * core::f64::consts::LN_2).abs() < 1e-12);
        assert!((th.top().entropy - core::f64::consts::LN_2).abs() < 0.4, "S = {}", th.top().entropy);
    }

    /// OBSERVING IS NOT PARTICIPATING: the recorded run must give bit-identical answers.
    ///
    /// The observed variant duplicates the loop to add recording, and a duplicated loop is a loop
    /// that can drift from its original. Same seed, same ladder, same rounds -- the optimiser's
    /// state, energy and swap rates must match exactly, or the observer has changed the dynamics
    /// it was meant to watch.
    #[test]
    fn recording_changes_no_answer() {
        let g = ising::lattice2d(6, 1.0);
        let betas: Vec<f64> = (0..10).map(|k| 0.1 + 0.9 * k as f64 / 9.0).collect();
        for seed in 0..4u64 {
            let plain = parallel_tempering(&g, &betas, 400, 3, seed, None);
            let (obs, _) = parallel_tempering_observed(&g, &betas, 400, 3, 50, seed, None);
            assert_eq!(plain.best, obs.best, "seed {seed}: different state");
            assert_eq!(plain.best_e.to_bits(), obs.best_e.to_bits(), "seed {seed}: different energy");
            assert_eq!(plain.swap_rates, obs.swap_rates, "seed {seed}: different swap rates");
        }
    }

    /// The quadrature bar is optimistic; the jackknife errs the other way. Measured, not asserted.
    ///
    /// Adjacent BAR steps share the samples at their common rung, so adding their variances in
    /// quadrature ignores a covariance -- that much is arithmetic. What it costs is a measurement:
    /// run the same ladder from many seeds, form `z = (estimate - truth) / reported stderr`, and
    /// ask whether `sd(z)` is 1. At the scale below, quadrature gives about 1.3 (its bar is ~30%
    /// too small) and the jackknife about 0.8 (slightly too wide, which is the direction to err).
    #[test]
    fn the_quadrature_bar_is_optimistic_and_the_jackknife_is_not() {
        let (n, j, h) = (12usize, 1.0, 0.1);
        let g = ising::ring(n, j, h);
        let (bmin, bmax) = (0.2, 1.8);
        let truth = ring_log_z(n, j, h, bmax) - ring_log_z(n, j, h, bmin);
        let betas = geometric_ladder(bmin, bmax, 8);
        let (mut zq, mut zj) = (Vec::new(), Vec::new());
        for seed in 0..24u64 {
            let (_, tr) = parallel_tempering_observed(&g, &betas, 4000, 2, 1000, seed, None);
            let steps = tr.log_z_differences();
            let total: f64 = steps.iter().map(|s| s.delta).sum();
            let quad: f64 = steps.iter().map(|s| s.stderr * s.stderr).sum::<f64>().sqrt();
            let (tot_j, se_j) = tr.log_z_total(8).unwrap();
            assert!((tot_j - total).abs() < 1e-12, "the jackknife must not move the ESTIMATE");
            zq.push((total - truth) / quad);
            zj.push((total - truth) / se_j);
        }
        let sd = |v: &[f64]| {
            let m = v.iter().sum::<f64>() / v.len() as f64;
            (v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (v.len() - 1) as f64).sqrt()
        };
        let (sq, sj) = (sd(&zq), sd(&zj));
        // 24 runs measures an sd to about sd/sqrt(2*24) = 14%, so the assertions are two of those
        // wide. A tighter threshold than the measurement is theatre -- and is exactly how this
        // module once reported 0.81 for a quantity that is 1.05.
        let tol = |v: f64| 2.0 * v / (2.0 * 24.0f64).sqrt();
        assert!(sq > 1.15 - tol(sq), "the quadrature bar should be visibly optimistic: sd(z) = {sq}");
        assert!(sj < sq, "the jackknife must improve on it: {sj} vs {sq}");
        assert!(sj < 1.3 + tol(sj), "and must not be badly optimistic itself: sd(z) = {sj}");
    }

    /// The jackknife refuses what it cannot do rather than returning a number.
    #[test]
    fn the_jackknife_refuses_too_few_blocks_or_samples() {
        let g = ising::ring(8, 1.0, 0.0);
        let betas = geometric_ladder(0.2, 1.0, 4);
        let (_, tr) = parallel_tempering_observed(&g, &betas, 200, 2, 100, 1, None);
        assert!(tr.log_z_total(3).unwrap_err().contains("too few"));
        assert!(tr.log_z_total(64).unwrap_err().contains("too few"));
        assert!(tr.log_z_total(8).is_ok());
    }

    /// A warm ladder has differences but no absolute scale, and says so rather than guessing.
    #[test]
    fn a_ladder_that_skips_infinite_temperature_is_refused_an_absolute_answer() {
        let g = ising::ring(10, 1.0, 0.1);
        let betas: Vec<f64> = (0..8).map(|k| 0.5 + 1.0 * k as f64 / 7.0).collect();
        let (_, traces) = parallel_tempering_observed(&g, &betas, 800, 2, 100, 9, None);
        let err = traces.thermodynamics(10, 3.0).unwrap_err();
        assert!(err.contains("no absolute anchor"), "{err}");
        // but the differences are there, and they telescope to the exact ratio
        let steps = traces.log_z_differences();
        assert_eq!(steps.len(), betas.len() - 1);
        let total: f64 = steps.iter().map(|s| s.delta).sum();
        let truth = ring_log_z(10, 1.0, 0.1, *betas.last().unwrap()) - ring_log_z(10, 1.0, 0.1, betas[0]);
        assert!((total - truth).abs() < 0.15, "telescoped {total} vs exact {truth}");
    }
}

#[cfg(test)]
mod tests {

    /// THE WORK FLOOR IS A SCHEDULING DECISION AND MUST CHANGE NO ANSWER.
    ///
    /// Below `MIN_REPLICA_WORK` the replicas run serially and above it they run threaded, and each
    /// replica carries its own RNG either way -- so the two paths must produce bit-identical
    /// states. If they ever diverge, the floor would silently make a run's answer depend on how
    /// big its graph happened to be.
    #[test]
    fn the_work_floor_changes_scheduling_and_not_answers() {
        use crate::graph::GraphBuilder;
        // Two runs of the same model at the same seed, one comfortably under the floor and one
        // over it, differing ONLY in swap_every -- which changes the work per call and therefore
        // which side of the floor it lands on.
        let build = |n: usize| {
            let mut b = GraphBuilder::new(n);
            for i in 0..n {
                b.couple(i, (i + 1) % n, if i % 3 == 0 { 1.0 } else { -1.0 });
            }
            b.build()
        };
        let g = build(600);
        let betas = geometric_ladder(0.2, 3.0, 8);
        // 8 * 1 * 600 = 4,800, under the floor; 8 * 8 * 600 = 38,400, over it.
        assert!(8 * g.n < MIN_REPLICA_WORK && 64 * g.n > MIN_REPLICA_WORK);

        // Same schedule and seed, run twice: determinism must hold on each side independently.
        for swap_every in [1usize, 8] {
            let a = parallel_tempering(&g, &betas, 30, swap_every, 11, None);
            let b = parallel_tempering(&g, &betas, 30, swap_every, 11, None);
            assert_eq!(a.best, b.best, "same seed must reproduce at swap_every {swap_every}");
            assert!((a.best_e - b.best_e).abs() < 1e-12);
        }
    }

    /// Replica-level threading must change NOTHING, and this proves it against a hand-rolled
    /// serial reference rather than against the argument that it should.
    ///
    /// The reference does exactly what `advance` used to do -- one replica after another, each
    /// with its own sampler -- so if `advance` ever grows a shared RNG, a shared best-tracker or
    /// any cross-replica read, these two stop agreeing.
    #[test]
    fn replica_threading_is_bit_identical_to_running_them_one_at_a_time() {
        let g = crate::ising::lattice2d(16, 1.0);
        let betas = geometric_ladder(0.1, 3.0, 8);

        for seed in [1u64, 0xBEEF, 0x1234_5678] {
            let got = parallel_tempering(&g, &betas, 30, 4, seed, None);

            // The serial reference, written out.
            let r = betas.len();
            let mut reps: Vec<Sampler> =
                (0..r).map(|i| Sampler::new(&g, betas[i], seed ^ (i as u64 * 0x9E37))).collect();
            let mut swap_rng = Pcg::new(seed ^ 0x5A5A, 3);
            let mut best = reps[r - 1].s.clone();
            let mut best_e = g.energy(&best);
            for round in 0..30 {
                for rep in reps.iter_mut() {
                    for _ in 0..4 {
                        rep.sweep(None);
                    }
                }
                for rep in reps.iter() {
                    let e = g.energy(&rep.s);
                    if e < best_e {
                        best_e = e;
                        best = rep.s.clone();
                    }
                }
                for i in (round % 2..r - 1).step_by(2) {
                    let e_i = g.energy(&reps[i].s);
                    let e_j = g.energy(&reps[i + 1].s);
                    let arg = (betas[i + 1] - betas[i]) * (e_j - e_i);
                    if arg >= 0.0 || swap_rng.f64() < arg.exp() {
                        let (a, b) = reps.split_at_mut(i + 1);
                        std::mem::swap(&mut a[i].s, &mut b[0].s);
                    }
                }
            }

            assert_eq!(got.best_e, best_e, "seed {seed:#x}: energy must be bit-identical");
            assert_eq!(got.best, best, "seed {seed:#x}: state must be bit-identical");
        }
    }

    /// And the ledger counts the same total however the replicas were spread.
    #[test]
    fn the_ledger_totals_the_same_across_replicas() {
        let g = crate::ising::lattice2d(12, 1.0);
        let betas = geometric_ladder(0.2, 2.0, 6);
        let mut led = Ledger::default();
        parallel_tempering(&g, &betas, 10, 3, 7, Some(&mut led));
        // 6 replicas x 10 rounds x 3 sweeps x n nodes, and integer addition does not care in which
        // order the threads finished.
        assert_eq!(led.samples, 6 * 10 * 3 * g.n as u64);
    }

    use super::*;
    use crate::graph::GraphBuilder;

    /// Random frustrated glass on 16 spins: the exact ground state is enumerable, and parallel
    /// tempering must find it. Plain low-temperature Gibbs is NOT required to (it can trap),
    /// which is the point of the ladder.
    #[test]
    fn tempering_finds_exact_ground_state() {
        let n = 16usize;
        let mut rng = Pcg::new(0x61A55, 5);
        let mut gb = GraphBuilder::new(n);
        // dense-ish random +-J glass
        for i in 0..n {
            for j in (i + 1)..n {
                if rng.f64() < 0.5 {
                    gb.couple(i, j, if rng.f64() < 0.5 { 1.0 } else { -1.0 });
                }
            }
        }
        let g = gb.build();
        // exact ground state by enumeration
        let mut e0 = f64::MAX;
        let mut s = vec![-1i8; n];
        for m in 0..(1u32 << n) {
            for b in 0..n {
                s[b] = if m >> b & 1 == 1 { 1 } else { -1 };
            }
            let e = g.energy(&s);
            if e < e0 {
                e0 = e;
            }
        }
        let betas = geometric_ladder(0.1, 3.0, 8);
        let res = parallel_tempering(&g, &betas, 200, 5, 0xF00D, None);
        assert!((res.best_e - e0).abs() < 1e-9, "PT found {} but exact ground state is {}", res.best_e, e0);
        // ladder health: no dead pair
        assert!(res.swap_rates.iter().all(|&x| x > 0.05), "dead ladder pair: {:?}", res.swap_rates);
        // annealing sanity on the same instance
        let sched: Vec<(f64, usize)> = geometric_ladder(0.1, 3.0, 30).into_iter().map(|b| (b, 40)).collect();
        let (_, e_sa) = anneal(&g, &sched, 0xA11, None);
        assert!((e_sa - e0).abs() < 1e-9, "SA found {} vs exact {}", e_sa, e0);
    }
}

/// Anneal under a [`crate::schedule::Schedule`], leaving the best state found.
///
/// The graph is borrowed and never rebuilt: every quantity that varies during the run comes from
/// the schedule. That is the whole point of the type, and `anneal_never_rebuilds_the_program`
/// below is what keeps it true.
pub fn anneal_scheduled(
    g: &Graph,
    schedule: &crate::schedule::Schedule,
    seed: u64,
    mut ledger: Option<&mut Ledger>,
) -> (Vec<i8>, f64) {
    let mut smp = Sampler::new(g, schedule.stages().first().map_or(1.0, |s| s.beta), seed);
    let mut best = smp.s.clone();
    let mut best_e = g.energy(&best);
    for stage in schedule.stages() {
        smp.beta = stage.beta; // a number changes; nothing is rebuilt
        for _ in 0..stage.sweeps {
            smp.sweep(ledger.as_deref_mut());
            let e = g.energy(&smp.s);
            if e < best_e {
                best_e = e;
                best = smp.s.clone();
            }
        }
    }
    (best, best_e)
}

#[cfg(test)]
mod schedule_contract {
    use super::*;
    use crate::graph::graph_builds;
    use crate::schedule::Schedule;

    #[test]
    fn anneal_never_rebuilds_the_program() {
        // THRML rebuilds its program at each of 4,000 annealing steps because beta is compiled
        // into its weights. This is the test that stops us doing the same: the counter must not
        // move once the graph exists, no matter how long the ladder.
        let g = crate::ising::lattice2d(16, 1.0);
        let schedule = Schedule::geometric(0.05, 4.0, 4000, 1);
        assert_eq!(schedule.len(), 4000);

        let before = graph_builds();
        let (_s, e) = anneal_scheduled(&g, &schedule, 7, None);
        let after = graph_builds();

        assert_eq!(after, before, "a 4,000-stage anneal rebuilt the program {} time(s)", after - before);
        assert!(e.is_finite());
    }

    #[test]
    fn running_a_schedule_matches_building_fresh_for_it() {
        // The other half of the contract: a graph carries no schedule state, so reusing one is
        // indistinguishable from building it again for this particular ladder.
        let schedule = Schedule::geometric(0.1, 3.0, 50, 4);

        let g1 = crate::ising::lattice2d(12, 1.0);
        let reused = anneal_scheduled(&g1, &schedule, 11, None);
        let second = anneal_scheduled(&g1, &schedule, 11, None); // same graph, run again

        let g2 = crate::ising::lattice2d(12, 1.0); // built fresh
        let fresh = anneal_scheduled(&g2, &schedule, 11, None);

        assert_eq!(reused.1, second.1, "reusing a graph changed the result");
        assert_eq!(reused.0, fresh.0, "a reused graph disagreed with a freshly built one");
        assert_eq!(reused.1, fresh.1);
    }

    #[test]
    fn two_schedules_on_one_graph_are_independent() {
        // Running a cold ladder must not leave the graph in a state that changes a later hot one.
        let g = crate::ising::lattice2d(12, 1.0);
        let hot = Schedule::geometric(0.05, 0.3, 20, 4);
        let cold = Schedule::geometric(0.5, 6.0, 20, 4);

        let hot_first = anneal_scheduled(&g, &hot, 3, None).1;
        let _ = anneal_scheduled(&g, &cold, 3, None);
        let hot_again = anneal_scheduled(&g, &hot, 3, None).1;

        assert_eq!(hot_first, hot_again, "a cold run contaminated a later hot run");
    }

    #[test]
    fn the_ledger_matches_what_the_schedule_predicted() {
        // Sizing a run before starting it has to be right, or the energy budget is fiction.
        let g = crate::ising::lattice2d(10, 1.0);
        let schedule = Schedule::geometric(0.1, 2.0, 30, 7);
        let mut led = Ledger::default();
        anneal_scheduled(&g, &schedule, 1, Some(&mut led));
        assert_eq!(led.samples, schedule.node_updates(g.n));
    }
}
