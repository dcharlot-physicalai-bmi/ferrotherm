//! Run-length distributions: how often a solver reaches a KNOWN optimum, against how long it ran.
//!
//! A single number out of a solver — "it got to -412.0" — answers almost nothing. Heuristic search
//! is a random variable, and the object that describes it is the **run-length distribution**: the
//! probability of success as a function of the work allowed. Everything anyone wants to say about a
//! solver is a functional of that curve — the median run length, the tail, and the time to solution
//! at a confidence, which is the number this field publishes and which cannot be computed from a
//! single run at all.
//!
//! # Success has to be exact, and that is what pins this module to `crate::planted`
//!
//! An RLD is only a measurement if "solved" is decidable. Score against the best energy anyone has
//! seen and the curve moves whenever a better run happens — the y-axis is then a statement about the
//! ensemble of runs rather than about the instance. So success here means *reaching a known
//! optimum*: [`measure_planted`] takes a [`Planted`] instance, whose ground state was chosen before
//! its couplings existed, and every other entry point takes the target energy explicitly.
//!
//! If any attempt comes back BELOW that target, the target was not the optimum, and the whole curve
//! is measuring a different event than it claims. That is counted in [`Rld::impossible`] and
//! [`Rld::sound`] reports it, rather than being quietly absorbed as a success.
//!
//! # What comes out
//!
//! [`Rld`] carries the record and its summaries: every [`Trial`] as it came back, the solve rate per
//! rung with a Wilson interval, the per-seed first hit ([`Rld::first_solved_at`], the run-length
//! distribution proper, whose quantiles [`Rld::quantile`] reads off), and [`Rld::tts`].
//!
//! # Two ways to sweep a ladder, and they measure different solvers
//!
//! [`measure_anytime`] runs each seed ONCE to the longest rung and reads every shorter rung off that
//! one trajectory. It is the cheap path, and it is right exactly for a solver that MERELY RUNS
//! LONGER rather than re-planning: fixed-temperature Gibbs, plain local search.
//!
//! [`measure_ladder`] re-runs every seed at every rung through [`Search`]. It costs a ladder-sum
//! more work, and it is the only correct path for a solver whose SCHEDULE is scaled to its budget —
//! [`crate::portfolio::Sqa`] divides one transverse-field ramp over exactly the steps it was given,
//! so a 10-step run is not the prefix of a 10,000-step one and its curve cannot be read off a single
//! long run. It is also the path that sees ENERGIES rather than hits, which is what lets it check
//! the target ([`Rld::impossible`]).
//!
//! # The closed form this is verified against
//!
//! For a solver that succeeds independently on each step with probability `p`, the run-length
//! distribution is exactly
//!
//! ```text
//!     P(solved by t) = 1 - (1-p)^t
//! ```
//!
//! and its time to solution at confidence `s` is exactly `ln(1-s) / ln(1-p)` steps — the SAME value
//! at every rung, because `t` cancels. [`Bernoulli`] is that solver, it solves nothing, and it is
//! here so the harness has something whose answer is known in advance. A harness checked only
//! against another sampler would be checked against nothing.
//!
//! # Run-length, not run-time
//!
//! The x-axis is work — single-spin proposals, the unit [`crate::portfolio::Budget`] counts and
//! [`crate::ledger`] prices. Nothing in this crate reads a clock, because a seeded result must
//! reproduce. A run-TIME distribution is this curve with its x-axis multiplied by the measured cost
//! of one proposal, which is [`Rld::curve_scaled`] — pass seconds per proposal for an RTD, or joules
//! per proposal for an energy-to-solution curve.

use crate::graph::Graph;
use crate::planted::Planted;
use crate::portfolio::{Budget, Search};
use crate::rng::Pcg;

/// Energy slack that still counts as reaching the optimum, matching [`Planted::solved`].
pub const SOLVED_TOL: f64 = 1e-9;

/// How close the incrementally-carried energy must come before it is recomputed exactly.
///
/// A gate on when to spend an exact recompute, never a criterion for success: it is deliberately
/// looser than [`SOLVED_TOL`] so that drift cannot hide a hit, and answering from it instead would
/// accept any state within it.
const RESYNC_SLACK: f64 = 1e-6;

/// One attempt: a seed, the work it was allowed, and whether it reached the known optimum.
///
/// The unit an RLD is built from, and the unit a time-to-solution estimate consumes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Trial {
    /// Seed of the attempt. Same seed and same run length must reproduce it exactly.
    pub seed: u64,
    /// Work the solver was allowed, in single-spin proposals.
    pub run_length: u64,
    /// Whether it reached the target energy within that budget. Exact, not "best seen".
    pub solved: bool,
}

/// A solver that reports the step at which it first reached the target, not just its best answer.
///
/// Narrower than [`Search`] and stronger: a solver that can say WHEN it hit needs one run per seed
/// to fill an entire ladder. Implement it only where the trajectory is independent of the budget.
pub trait FirstHit {
    /// A stable name for reports.
    fn name(&self) -> &str;
    /// The 1-based step at which `seed` first reached the target, or `None` within `budget`.
    ///
    /// `Some(0)` means the initial state was already optimal. Must be deterministic in `seed`, and
    /// a shorter budget must return the same answer as the prefix of a longer one.
    fn first_hit(&self, seed: u64, budget: u64) -> Option<u64>;
}

/// The reference solver: succeeds on each step with probability `p`, independently.
///
/// It has no instance and finds nothing. Its RLD is `1 - (1-p)^t` in closed form, so it is what
/// says whether the harness around it counts correctly.
#[derive(Clone, Copy, Debug)]
pub struct Bernoulli {
    /// Per-step success probability, in `(0, 1]`.
    pub p: f64,
}

impl FirstHit for Bernoulli {
    fn name(&self) -> &str {
        "bernoulli"
    }
    fn first_hit(&self, seed: u64, budget: u64) -> Option<u64> {
        let mut rng = Pcg::new(seed, 0);
        // One draw per step, in order, so a shorter budget is the prefix of a longer run.
        (1..=budget).find(|_| rng.f64() < self.p)
    }
}

/// Single-site Gibbs at fixed temperature, stopped the moment it reaches the target energy.
///
/// One step is one proposal at a uniformly chosen site, through [`crate::kernel::draw`] — the
/// crate's one update, not a second copy of it. Fixed beta is what makes it an anytime solver: its
/// trajectory does not know the budget.
pub struct SpinFlip<'g> {
    /// The instance.
    pub g: &'g Graph,
    /// Inverse temperature, held fixed for the whole run.
    pub beta: f64,
    /// Energy that counts as solved, within [`SOLVED_TOL`].
    pub target: f64,
}

impl<'g> SpinFlip<'g> {
    /// A run against a planted instance's true optimum.
    #[must_use]
    pub fn on_planted(p: &'g Planted, beta: f64) -> SpinFlip<'g> {
        SpinFlip { g: &p.graph, beta, target: p.ground_energy }
    }
}

impl SpinFlip<'_> {
    /// The run, returning the state a hit is claimed FOR alongside the step it happened at.
    ///
    /// Exists so the claim is checkable: `first_hit` alone returns a number, and a number cannot be
    /// audited against the instance. On a miss the state is where the run ended.
    #[must_use]
    pub fn first_hit_state(&self, seed: u64, budget: u64) -> (Option<u64>, Vec<i8>) {
        let n = self.g.n;
        if n == 0 {
            return (None, Vec::new());
        }
        let mut rng = Pcg::new(seed, 1);
        let mut s: Vec<i8> = (0..n).map(|_| rng.spin(0.5)).collect();
        // Carried incrementally and RESYNCED at the decision point: a million additions of a delta
        // drift, and the one comparison that decides the answer must not be made against drift.
        let mut e = self.g.energy(&s);
        if e <= self.target + SOLVED_TOL {
            return (Some(0), s);
        }
        for t in 1..=budget {
            let i = (rng.next_u32() as usize) % n;
            let f = self.g.field(i, &s);
            let v = crate::kernel::draw(f, self.beta, &mut rng);
            if v != s[i] {
                e += crate::kernel::delta_e(f, s[i]);
                s[i] = v;
                // RESYNC_SLACK only decides when to recompute. What decides the ANSWER is the
                // recomputed energy against SOLVED_TOL -- a near-degenerate instance can put a
                // decoy state inside the slack, and a run that answered from the running number
                // would report the decoy as an optimum.
                if e <= self.target + RESYNC_SLACK {
                    e = self.g.energy(&s);
                    if e <= self.target + SOLVED_TOL {
                        return (Some(t), s);
                    }
                }
            }
        }
        (None, s)
    }
}

impl FirstHit for SpinFlip<'_> {
    fn name(&self) -> &str {
        "spin-flip"
    }
    fn first_hit(&self, seed: u64, budget: u64) -> Option<u64> {
        self.first_hit_state(seed, budget).0
    }
}

/// An empirical run-length distribution over a seed ensemble.
#[derive(Clone, Debug)]
pub struct Rld {
    /// Which solver produced it.
    pub solver: String,
    /// The seed ensemble, in the order [`Rld::first_solved_at`] is indexed.
    pub seeds: Vec<u64>,
    /// Run lengths measured, strictly increasing, in single-spin proposals.
    pub ladder: Vec<u64>,
    /// Every attempt as it came back. The record; every count below is a summary of it.
    ///
    /// Kept rather than rebuilt from [`Rld::first_solved_at`], because a budget-scaled solver can
    /// solve at one rung and miss a longer one — so the per-rung outcomes are not recoverable from
    /// a first-hit time, and a time-to-solution estimate fed rebuilt trials would be fed attempts
    /// nobody ran.
    pub trials: Vec<Trial>,
    /// Attempts that reached the target at each rung. Same length as [`Rld::ladder`].
    pub solved: Vec<usize>,
    /// Attempts made at each rung. Same length as [`Rld::ladder`].
    pub attempts: Vec<usize>,
    /// Per seed, the first run length at which it solved; `None` if it never did.
    ///
    /// Exact in [`measure_anytime`], where it is the true first-hit step. Quantized to the ladder in
    /// [`measure_ladder`], where the solver reports only its best answer — and there it is not
    /// necessarily monotone, since a budget-scaled schedule can solve at one rung and miss at a
    /// longer one. Rung counts are always taken from the attempts, never from this.
    pub first_solved_at: Vec<Option<u64>>,
    /// Proposals actually spent filling this curve, as the solvers report them.
    pub steps_used: u64,
    /// Attempts that came back BELOW the target, meaning the target was not the optimum.
    ///
    /// Always zero from [`measure_anytime`], which asks solvers for hits rather than energies.
    pub impossible: usize,
}

impl Rld {
    /// Rebuild a curve from trials measured elsewhere — another binding, a device, a GPU run.
    ///
    /// Rungs are the distinct run lengths present, in increasing order, and [`Rld::seeds`] comes
    /// back sorted. `steps_used` is the sum of the trials' run lengths, what they cost as runs.
    ///
    /// # Panics
    ///
    /// If `trials` is empty.
    #[must_use]
    pub fn from_trials(solver: &str, trials: &[Trial]) -> Rld {
        assert!(!trials.is_empty(), "an RLD over no trials is not a measurement");
        let mut ladder: Vec<u64> = trials.iter().map(|t| t.run_length).collect();
        ladder.sort_unstable();
        ladder.dedup();
        let mut seeds: Vec<u64> = trials.iter().map(|t| t.seed).collect();
        seeds.sort_unstable();
        seeds.dedup();

        let (solved, attempts) = tally(&ladder, trials);
        let mut first: Vec<Option<u64>> = vec![None; seeds.len()];
        let mut steps_used = 0u64;
        for t in trials {
            steps_used = steps_used.saturating_add(t.run_length);
            if t.solved {
                let si = seeds.binary_search(&t.seed).expect("seed was collected above");
                if first[si].is_none_or(|had| t.run_length < had) {
                    first[si] = Some(t.run_length);
                }
            }
        }
        Rld {
            solver: solver.to_string(),
            seeds,
            ladder,
            trials: trials.to_vec(),
            solved,
            attempts,
            first_solved_at: first,
            steps_used,
            impossible: 0,
        }
    }

    /// Solve rate at rung `k`.
    ///
    /// # Panics
    ///
    /// If `k` is past the end of the ladder, or the rung has no attempts.
    #[must_use]
    pub fn success(&self, k: usize) -> f64 {
        let rungs = self.ladder.len();
        assert!(k < rungs, "rung {k} is past the end of a {rungs}-rung ladder");
        assert!(self.attempts[k] > 0, "rung {k} has no attempts, so it has no solve rate");
        self.solved[k] as f64 / self.attempts[k] as f64
    }

    /// The curve itself: run length against solve rate.
    #[must_use]
    pub fn curve(&self) -> Vec<(u64, f64)> {
        (0..self.ladder.len()).map(|k| (self.ladder[k], self.success(k))).collect()
    }

    /// The same curve with its x-axis in another unit: seconds per proposal gives an RTD, joules
    /// per proposal an energy-to-solution curve.
    #[must_use]
    pub fn curve_scaled(&self, cost_per_step: f64) -> Vec<(f64, f64)> {
        (0..self.ladder.len())
            .map(|k| (self.ladder[k] as f64 * cost_per_step, self.success(k)))
            .collect()
    }

    /// Whether the target was consistent with every attempt — false means it was not the optimum.
    #[must_use]
    pub fn sound(&self) -> bool {
        self.impossible == 0
    }

    /// Wilson score interval for the solve rate at rung `k`, at `z` standard deviations.
    ///
    /// Wilson rather than `p +/- z*sqrt(p(1-p)/n)`, because the naive interval has width ZERO at
    /// zero and one successes — exactly the two rungs a ladder is built to contain, and exactly
    /// where "0/50 solved" would otherwise be reported as certainty.
    ///
    /// # Panics
    ///
    /// If `k` is past the end of the ladder, or the rung has no attempts.
    #[must_use]
    pub fn wilson(&self, k: usize, z: f64) -> (f64, f64) {
        let p = self.success(k);
        let n = self.attempts[k] as f64;
        let z2 = z * z;
        let denom = 1.0 + z2 / n;
        let centre = (p + z2 / (2.0 * n)) / denom;
        let half = z / denom * (p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt();
        ((centre - half).max(0.0), (centre + half).min(1.0))
    }

    /// The run length by which a fraction `q` of seeds had solved, from [`Rld::first_solved_at`].
    ///
    /// `None` if fewer than `q` of them ever did — which is the honest answer and the reason this
    /// returns an `Option` rather than the longest rung.
    ///
    /// # Panics
    ///
    /// If `q` is not in `(0, 1]`, or no seeds were measured.
    #[must_use]
    pub fn quantile(&self, q: f64) -> Option<u64> {
        assert!(q > 0.0 && q <= 1.0, "a quantile is in (0, 1], got {q}");
        assert!(!self.first_solved_at.is_empty(), "no seeds were measured");
        let mut hits: Vec<u64> = self.first_solved_at.iter().flatten().copied().collect();
        let need = (q * self.first_solved_at.len() as f64).ceil() as usize;
        if hits.len() < need.max(1) {
            return None;
        }
        hits.sort_unstable();
        Some(hits[need.max(1) - 1])
    }

    /// Time to solution at confidence `s`: the ladder rung minimising `t * ln(1-s) / ln(1-p(t))`.
    ///
    /// That expression is the expected work to see one success with probability `s`, spending it as
    /// independent restarts of length `t`. `None` if no rung ever solved, since restarting a solver
    /// that never succeeds costs unbounded work rather than a large number.
    ///
    /// # Panics
    ///
    /// If `s` is not strictly between 0 and 1.
    #[must_use]
    pub fn tts(&self, s: f64) -> Option<Tts> {
        assert!(s > 0.0 && s < 1.0, "a confidence is in (0, 1), got {s}");
        let mut best: Option<Tts> = None;
        for k in 0..self.ladder.len() {
            let p = self.success(k);
            if p <= 0.0 {
                continue;
            }
            let t = self.ladder[k] as f64;
            // At p = 1 the formula's ln(1-p) is -infinity and the ratio collapses to zero, which is
            // not the answer: one run of length t always succeeds, so the cost is t.
            let steps = if p >= 1.0 { t } else { t * (1.0 - s).ln() / (1.0 - p).ln() };
            if best.as_ref().is_none_or(|b| steps < b.steps) {
                best = Some(Tts { steps, run_length: self.ladder[k], success: p, confidence: s });
            }
        }
        best
    }
}

/// A time-to-solution estimate and the rung it came from.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Tts {
    /// Expected proposals to reach the optimum at least once with probability [`Tts::confidence`].
    pub steps: f64,
    /// The run length that minimised it. A ladder whose minimum sits at an endpoint is too short.
    pub run_length: u64,
    /// Solve rate measured at that rung.
    pub success: f64,
    /// The confidence asked for, conventionally 0.99.
    pub confidence: f64,
}

/// A log-spaced ladder of run lengths from `min` to `max`, strictly increasing.
///
/// Log-spaced because a solve rate rises over orders of magnitude of work, and a linear ladder
/// spends most of its rungs where the curve is already flat.
///
/// # Panics
///
/// If `min` is zero, `max` is below `min`, or `rungs` is zero.
#[must_use]
pub fn ladder(min: u64, max: u64, rungs: usize) -> Vec<u64> {
    assert!(min > 0, "a run length of zero measures nothing");
    assert!(max >= min, "ladder max {max} is below min {min}");
    assert!(rungs > 0, "a ladder needs at least one rung");
    if rungs == 1 || max == min {
        return vec![max];
    }
    let (lo, hi) = ((min as f64).ln(), (max as f64).ln());
    let mut out: Vec<u64> = (0..rungs)
        .map(|k| {
            let x = lo + (hi - lo) * k as f64 / (rungs - 1) as f64;
            x.exp().round() as u64
        })
        .collect();
    out[0] = min;
    out[rungs - 1] = max;
    out.dedup();
    out
}

/// Sweep a ladder by running each seed ONCE to the longest rung and reading the rest off it.
///
/// Correct only for a solver whose trajectory does not depend on its budget, which is what
/// [`FirstHit`] promises. For anything whose schedule is scaled to the budget, use
/// [`measure_ladder`].
///
/// # Panics
///
/// If `seeds` is empty, or `ladder` is empty or not strictly increasing.
#[must_use]
pub fn measure_anytime<S: FirstHit + ?Sized>(solver: &S, seeds: &[u64], ladder: &[u64]) -> Rld {
    check_ladder(seeds, ladder);
    let longest = ladder[ladder.len() - 1];
    let mut first = Vec::with_capacity(seeds.len());
    let mut steps_used = 0u64;
    for &seed in seeds {
        let h = solver.first_hit(seed, longest);
        steps_used = steps_used.saturating_add(h.unwrap_or(longest));
        first.push(h);
    }
    // The trial at a shorter rung is READ OFF the long run, which is sound exactly because
    // [`FirstHit`] promises a shorter budget returns the prefix's answer. That promise is what
    // `per_budget_reruns_and_one_long_run_agree_exactly` measures rather than assumes.
    let mut trials = Vec::with_capacity(ladder.len() * seeds.len());
    for &t in ladder {
        for (i, &seed) in seeds.iter().enumerate() {
            trials.push(Trial { seed, run_length: t, solved: first[i].is_some_and(|h| h <= t) });
        }
    }
    let (solved, attempts) = tally(ladder, &trials);
    Rld {
        solver: solver.name().to_string(),
        seeds: seeds.to_vec(),
        ladder: ladder.to_vec(),
        trials,
        solved,
        attempts,
        first_solved_at: first,
        steps_used,
        impossible: 0,
    }
}

/// Sweep a ladder by re-running every seed at every rung, scoring against `target`.
///
/// The general path: it asks only for [`Search`], so it works with every arm in
/// [`crate::portfolio`], including the ones whose schedule is scaled to the budget and whose curve
/// therefore cannot be read off one long run.
///
/// # Panics
///
/// If `seeds` is empty, or `ladder` is empty or not strictly increasing.
#[must_use]
pub fn measure_ladder(
    search: &dyn Search,
    g: &Graph,
    target: f64,
    seeds: &[u64],
    ladder: &[u64],
) -> Rld {
    check_ladder(seeds, ladder);
    let mut trials = Vec::with_capacity(ladder.len() * seeds.len());
    let mut first: Vec<Option<u64>> = vec![None; seeds.len()];
    let mut steps_used = 0u64;
    let mut impossible = 0usize;
    for &t in ladder {
        for (i, &seed) in seeds.iter().enumerate() {
            let found = search.solve(g, Budget::new(t), seed);
            steps_used = steps_used.saturating_add(found.spent);
            if found.energy < target - SOLVED_TOL {
                impossible += 1;
            }
            let solved = found.energy <= target + SOLVED_TOL;
            if solved && first[i].is_none_or(|had| t < had) {
                first[i] = Some(t);
            }
            trials.push(Trial { seed, run_length: t, solved });
        }
    }
    let (solved, attempts) = tally(ladder, &trials);
    Rld {
        solver: search.name().to_string(),
        seeds: seeds.to_vec(),
        ladder: ladder.to_vec(),
        trials,
        solved,
        attempts,
        first_solved_at: first,
        steps_used,
        impossible,
    }
}

/// [`measure_ladder`] against a planted instance's true optimum, which is the intended pairing.
///
/// # Panics
///
/// If `seeds` is empty, or `ladder` is empty or not strictly increasing.
#[must_use]
pub fn measure_planted(
    search: &dyn Search,
    p: &Planted,
    seeds: &[u64],
    ladder: &[u64],
) -> Rld {
    measure_ladder(search, &p.graph, p.ground_energy, seeds, ladder)
}

/// Rung counts, taken from the attempts themselves — the only place they may come from.
fn tally(ladder: &[u64], trials: &[Trial]) -> (Vec<usize>, Vec<usize>) {
    let mut solved = vec![0usize; ladder.len()];
    let mut attempts = vec![0usize; ladder.len()];
    for t in trials {
        let k = ladder.binary_search(&t.run_length).expect("every trial's run length is a rung");
        attempts[k] += 1;
        solved[k] += usize::from(t.solved);
    }
    (solved, attempts)
}

fn check_ladder(seeds: &[u64], ladder: &[u64]) {
    assert!(!seeds.is_empty(), "an RLD over no seeds is not a measurement");
    assert!(!ladder.is_empty(), "a ladder needs at least one rung");
    assert!(
        ladder.windows(2).all(|w| w[0] < w[1]),
        "the ladder must be strictly increasing, got {ladder:?}"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The exact geometric ensemble at `p = 1/2`: with `2^k` seeds, exactly `2^(k-j)` of them first
    /// hit at step `j`, so the empirical CDF is `1 - 2^-t` in exact binary floating point.
    fn exact_geometric_trials(k: u32) -> Vec<Trial> {
        let n = 1u64 << k;
        let mut hits: Vec<Option<u64>> = Vec::with_capacity(n as usize);
        for j in 1..=u64::from(k) {
            for _ in 0..(n >> j) {
                hits.push(Some(j));
            }
        }
        while (hits.len() as u64) < n {
            hits.push(None); // the 2^-k tail that has not hit by step k
        }
        let mut out = Vec::new();
        for t in 1..=u64::from(k) {
            for (seed, h) in hits.iter().enumerate() {
                out.push(Trial {
                    seed: seed as u64,
                    run_length: t,
                    solved: h.is_some_and(|x| x <= t),
                });
            }
        }
        out
    }

    /// EXACT. The counts are the empirical CDF of the first-hit times, computed independently here.
    /// A harness that counted `< t` instead of `<= t`, or reused a rung's count, fails this.
    #[test]
    fn rung_counts_are_the_empirical_cdf_of_the_first_hit_times() {
        let solver = Bernoulli { p: 0.1 };
        let seeds: Vec<u64> = (0..500).collect();
        let rungs = ladder(1, 64, 8);
        let r = measure_anytime(&solver, &seeds, &rungs);

        for (k, &t) in rungs.iter().enumerate() {
            let want = seeds
                .iter()
                .filter(|&&s| solver.first_hit(s, 64).is_some_and(|h| h <= t))
                .count();
            assert_eq!(r.solved[k], want, "rung {t}");
            assert_eq!(r.attempts[k], seeds.len());
        }
        // and the curve is a CDF: non-decreasing.
        assert!(r.curve().windows(2).all(|w| w[0].1 <= w[1].1), "{:?}", r.curve());
    }

    /// EXACT, no sampling: an ensemble whose first-hit histogram IS the geometric one at p = 1/2,
    /// so the solve rate must be `1 - (1-p)^t` to the last bit (these are binary fractions).
    #[test]
    fn the_exact_geometric_ensemble_reproduces_one_minus_one_minus_p_to_the_t() {
        let r = Rld::from_trials("exact-geometric", &exact_geometric_trials(10));
        assert_eq!(r.ladder, (1..=10).collect::<Vec<u64>>());
        for (k, &t) in r.ladder.iter().enumerate() {
            let closed = 1.0 - 0.5f64.powi(t as i32);
            assert_eq!(r.success(k), closed, "rung {t}");
        }
    }

    /// EXACT. For a per-step probability `p` the analytic TTS is `ln(1-s)/ln(1-p)`, the same at
    /// every rung because `t` cancels — so the harness must report that value, and report it as flat
    /// across the ladder. A TTS that forgot the leading `t` would still pass at t = 1 only.
    #[test]
    fn tts_from_the_exact_ensemble_equals_the_analytic_value() {
        let r = Rld::from_trials("exact-geometric", &exact_geometric_trials(10));
        let s: f64 = 0.99;
        let analytic = (1.0 - s).ln() / (1.0 - 0.5f64).ln(); // = ln(0.01)/ln(0.5) = 6.6438...

        for (k, &t) in r.ladder.iter().enumerate() {
            let p = r.success(k);
            let per_rung = t as f64 * (1.0 - s).ln() / (1.0 - p).ln();
            assert!(
                (per_rung - analytic).abs() < 1e-12,
                "rung {t}: {per_rung} vs analytic {analytic}"
            );
        }
        let got = r.tts(s).expect("the ensemble solves");
        assert!((got.steps - analytic).abs() < 1e-12, "{} vs {analytic}", got.steps);
        assert!((got.steps - 6.643_856_189_774_724).abs() < 1e-9, "ln(0.01)/ln(0.5)");
    }

    /// The median of the exact geometric ensemble at p = 1/2 is 1 step, and the 0.75 quantile is 2 —
    /// read off the histogram, not from a fit.
    #[test]
    fn quantiles_of_the_exact_ensemble_are_the_geometric_ones() {
        let r = Rld::from_trials("exact-geometric", &exact_geometric_trials(10));
        assert_eq!(r.quantile(0.5), Some(1));
        assert_eq!(r.quantile(0.75), Some(2));
        assert_eq!(r.quantile(0.875), Some(3));
        // 1023 of 1024 seeds ever hit, so anything past that has no answer rather than a big one.
        assert_eq!(r.quantile(1.0), None);
    }

    /// The sampled version of the same closed form: a real coin, flipped per step, through the whole
    /// harness. Statistical, so it is stated as such — the tolerance is four binomial standard
    /// errors, which a systematically miscounted rung (off by one step) blows through.
    #[test]
    fn a_bernoulli_solver_matches_the_closed_form_within_four_sigma() {
        let p = 0.05;
        let seeds: Vec<u64> = (0..20_000).collect();
        let rungs = ladder(1, 96, 10);
        let r = measure_anytime(&Bernoulli { p }, &seeds, &rungs);
        let n = seeds.len() as f64;

        for (k, &t) in rungs.iter().enumerate() {
            let closed = 1.0 - (1.0 - p).powi(t as i32);
            let sigma = (closed * (1.0 - closed) / n).sqrt();
            let got = r.success(k);
            assert!(
                (got - closed).abs() < 4.0 * sigma + 1e-12,
                "rung {t}: {got} vs closed form {closed}, 4 sigma = {}",
                4.0 * sigma
            );
            // and the interval quoted for that rung must cover the truth it is quoted about
            let (lo, hi) = r.wilson(k, 1.96);
            assert!(lo <= closed && closed <= hi, "rung {t}: [{lo}, {hi}] misses {closed}");
        }
        let analytic = (1.0 - 0.99f64).ln() / (1.0 - p).ln();
        let got = r.tts(0.99).expect("it solves").steps;
        assert!((got - analytic).abs() / analytic < 0.03, "TTS {got} vs analytic {analytic}");
    }

    /// The two sweep modes must AGREE exactly on a budget-independent solver, because the same seed
    /// at a shorter budget is the prefix of the longer run. This is the identity that says the cheap
    /// path is not cheating: it is why one run per seed may stand in for a ladder of them.
    #[test]
    fn per_budget_reruns_and_one_long_run_agree_exactly() {
        let solver = Bernoulli { p: 0.08 };
        let seeds: Vec<u64> = (0..300).collect();
        let rungs = ladder(1, 48, 6);
        let cheap = measure_anytime(&solver, &seeds, &rungs);

        // the same sweep done the expensive way, one fresh run per (seed, rung)
        let mut trials = Vec::new();
        let mut spent = 0u64;
        for &t in &rungs {
            for &seed in &seeds {
                let h = solver.first_hit(seed, t);
                spent += h.unwrap_or(t);
                trials.push(Trial { seed, run_length: t, solved: h.is_some() });
            }
        }
        let dear = Rld::from_trials("bernoulli", &trials);
        assert_eq!(cheap.trials, dear.trials, "every attempt, not merely every count");
        assert_eq!(cheap.solved, dear.solved);
        assert_eq!(cheap.first_solved_at.len(), dear.first_solved_at.len());
        assert!(spent > cheap.steps_used, "the ladder sweep must cost more, or it is not a saving");
    }

    /// EXACT, and it names what a trial list loses. A [`Trial`] carries `(run_length, solved)` and
    /// nothing finer, so rebuilding from one reproduces every rung count exactly and rounds each
    /// first hit UP to the next rung — a seed that hit at step 3 is indistinguishable from one that
    /// hit at step 4 once the ladder's shortest rung above them both is 8. Asserting equality of the
    /// hit times here would be asserting something trials cannot carry.
    #[test]
    fn trials_round_trip_the_counts_and_quantize_the_hit_times() {
        let seeds: Vec<u64> = (0..200).collect(); // sorted, so `from_trials` keeps this order
        let rungs = ladder(1, 32, 6);
        let r = measure_anytime(&Bernoulli { p: 0.12 }, &seeds, &rungs);
        let back = Rld::from_trials(&r.solver, &r.trials);
        assert_eq!(back.ladder, r.ladder);
        assert_eq!(back.solved, r.solved);
        assert_eq!(back.attempts, r.attempts);

        for (i, &h) in r.first_solved_at.iter().enumerate() {
            let up = h.and_then(|x| r.ladder.iter().copied().find(|&t| t >= x));
            assert_eq!(back.first_solved_at[i], up, "seed {i}, exact hit {h:?}");
        }
        // and the quantization is upward, never downward: a rebuilt curve can only look slower
        assert!(
            back.first_solved_at.iter().zip(&r.first_solved_at).all(|(b, e)| match (b, e) {
                (Some(b), Some(e)) => b >= e,
                (None, None) => true,
                _ => false,
            })
        );
    }

    /// Wilson at zero successes, against its own closed form: the upper bound is `z^2 / (n + z^2)`.
    /// The naive interval reports [0, 0] here, which is a claim of certainty from 0/100.
    #[test]
    fn the_wilson_interval_matches_its_closed_form_at_zero_successes() {
        let none: Vec<Trial> =
            (0..100).map(|seed| Trial { seed, run_length: 10, solved: false }).collect();
        let r = Rld::from_trials("none", &none);
        assert_eq!(r.attempts, vec![100]);
        let (lo, hi) = r.wilson(0, 1.96);
        let closed = 1.96 * 1.96 / (100.0 + 1.96 * 1.96);
        assert_eq!(lo, 0.0);
        assert!((hi - closed).abs() < 1e-12, "{hi} vs {closed}");
        assert!(r.tts(0.99).is_none(), "a solver that never solved has no time to solution");
        assert_eq!(r.quantile(0.5), None);
    }

    /// At p = 1 the restart formula divides by `ln(0)`; the answer is the run length itself.
    #[test]
    fn a_rung_that_always_solves_costs_exactly_one_run() {
        let all: Vec<Trial> =
            (0..50).map(|seed| Trial { seed, run_length: 7, solved: true }).collect();
        let r = Rld::from_trials("always", &all);
        assert_eq!(r.success(0), 1.0);
        let t = r.tts(0.99).expect("it solves");
        assert_eq!(t.steps, 7.0);
        assert_eq!(t.run_length, 7);
    }

    /// The success criterion is anchored to an EXACT optimum: the planted energy is confirmed by
    /// exhaustive enumeration first, and only then used as the target.
    #[test]
    fn success_on_a_planted_instance_is_scored_against_the_enumerated_optimum() {
        use crate::oracle::Solver;
        let p = crate::planted::frustrated_loops(4, 12, 7);
        let (_s, e) = crate::oracle::Exhaustive.solve(&p.graph);
        assert!(
            (e - p.ground_energy).abs() < 1e-9,
            "planted {} is not the true optimum {e}",
            p.ground_energy
        );

        let seeds: Vec<u64> = (0..64).collect();
        let rungs = ladder(1, 4096, 7);
        let r = measure_anytime(&SpinFlip::on_planted(&p, 2.0), &seeds, &rungs);
        assert!(r.curve().windows(2).all(|w| w[0].1 <= w[1].1), "a CDF cannot fall: {:?}", r.curve());
        assert_eq!(r.success(r.ladder.len() - 1), 1.0, "4096 proposals on 16 spins should solve");
        assert!(r.success(0) < 1.0, "one proposal should not");
        assert!(r.tts(0.99).is_some());
    }

    /// A reported hit must BE the optimum, not merely within the slack the running energy is
    /// resynced at. The instance is near-degenerate ON PURPOSE: its second-best state sits 5e-7
    /// above the optimum, INSIDE `RESYNC_SLACK` and far outside `SOLVED_TOL`, so a run that answered
    /// from the running energy would report that decoy. Deterministic by seed, and the optimum comes
    /// from enumeration rather than from the code under test.
    #[test]
    fn every_reported_hit_is_a_state_that_is_actually_at_the_optimum() {
        use crate::oracle::Solver;
        let mut b = crate::graph::GraphBuilder::new(2);
        b.bias(0, 1.0);
        b.bias(1, 2.5e-7);
        let g = b.build();
        let (_best, opt) = crate::oracle::Exhaustive.solve(&g);
        let decoy = g.energy(&[1, -1]);
        assert!(decoy - opt < RESYNC_SLACK, "the decoy must sit inside the resync slack");
        assert!(decoy - opt > SOLVED_TOL, "and outside the success tolerance, or it proves nothing");

        let solver = SpinFlip { g: &g, beta: 5.0, target: opt };
        let mut hits = 0;
        for seed in 0..64u64 {
            let (h, state) = solver.first_hit_state(seed, 200);
            if h.is_some() {
                hits += 1;
                assert!(
                    g.energy(&state) <= opt + SOLVED_TOL,
                    "seed {seed} claimed a hit at {:?} on a state of energy {}, optimum {opt}",
                    h,
                    g.energy(&state)
                );
            }
        }
        assert_eq!(hits, 64, "a two-spin chain at beta 5 reaches both states within 200 proposals");
    }

    /// The falsifier. Given a target BELOW the true optimum, nothing can reach it, so the curve must
    /// be flat zero — a harness scoring "best seen" instead of "reached the target" would report
    /// successes here, because every run does have a best.
    #[test]
    fn an_unreachable_target_is_never_reached() {
        let p = crate::planted::frustrated_loops(4, 12, 7);
        let below = SpinFlip { g: &p.graph, beta: 2.0, target: p.ground_energy - 1.0 };
        let seeds: Vec<u64> = (0..32).collect();
        let rungs = ladder(1, 2048, 5);
        let r = measure_anytime(&below, &seeds, &rungs);
        assert!(r.solved.iter().all(|&c| c == 0), "{:?}", r.solved);
        assert!(r.tts(0.99).is_none());
    }

    /// A target ABOVE the optimum is the "best seen" trap in the other direction: attempts come back
    /// below it, which means it was not the optimum, and the curve says so instead of hiding it.
    #[test]
    fn a_target_above_the_optimum_is_reported_as_unsound() {
        let p = crate::planted::frustrated_loops(4, 12, 7);
        let seeds: Vec<u64> = (0..8).collect();
        let rungs = ladder(64, 1024, 3);
        let honest = measure_planted(&crate::portfolio::Tabu, &p, &seeds, &rungs);
        assert!(honest.sound(), "the planted optimum is the optimum");

        // Deliberately far above every state's energy, so the count is 24 of 24 rather than a
        // number that depends on how well tabu happened to do.
        let wrong = measure_ladder(
            &crate::portfolio::Tabu,
            &p.graph,
            p.ground_energy + 1e6,
            &seeds,
            &rungs,
        );
        assert!(!wrong.sound(), "attempts beat the claimed optimum and that must be visible");
        assert!(wrong.impossible > 0);
    }

    /// A real arm through the general path, on an instance whose optimum is known by construction.
    #[test]
    fn a_portfolio_arm_sweeps_the_ladder() {
        let p = crate::planted::frustrated_loops(6, 40, 3);
        let seeds: Vec<u64> = (0..12).collect();
        let rungs = ladder(36, 36 * 64, 4);
        let r = measure_planted(&crate::portfolio::Tabu, &p, &seeds, &rungs);
        assert_eq!(r.solver, "tabu");
        assert_eq!(r.attempts, vec![seeds.len(); rungs.len()]);
        assert_eq!(r.trials.len(), rungs.len() * seeds.len(), "one trial per attempt, all kept");
        assert!(r.steps_used >= rungs.iter().sum::<u64>(), "every rung was actually run");
        assert!(r.sound());
        assert!(r.success(rungs.len() - 1) > 0.0, "tabu should solve a 36-spin planted instance");
        // the RTD is the RLD with a cost per proposal attached; the shape is unchanged.
        let rtd = r.curve_scaled(1e-9);
        assert_eq!(rtd.len(), r.ladder.len());
        assert_eq!(rtd[0].1, r.success(0));
    }

    /// EXACT, and it is why the trials are kept rather than rebuilt. This arm solves at ONE rung
    /// and misses the longer one — legal for a budget-scaled schedule, and the case where a curve
    /// rebuilt from first-hit times would claim a success at the longer rung that never happened.
    #[test]
    fn a_solver_that_works_at_only_one_rung_has_that_in_its_trials() {
        struct OneRung<'a> {
            planted: &'a Planted,
            rung: u64,
        }
        impl Search for OneRung<'_> {
            fn name(&self) -> &'static str {
                "one-rung"
            }
            fn solve(&self, g: &Graph, budget: Budget, _seed: u64) -> crate::portfolio::Found {
                let state = if budget.proposals == self.rung {
                    self.planted.ground_state.clone()
                } else {
                    vec![1i8; g.n]
                };
                crate::portfolio::Found {
                    arm: "one-rung",
                    energy: g.energy(&state),
                    spent: budget.proposals,
                    state,
                }
            }
        }

        let p = crate::planted::frustrated_loops(4, 12, 7);
        assert!(
            p.graph.energy(&vec![1i8; p.graph.n]) > p.ground_energy + 1.0,
            "the miss state must really be a miss"
        );
        let seeds: Vec<u64> = (0..5).collect();
        let rungs = vec![20u64, 40, 80];
        let r = measure_planted(&OneRung { planted: &p, rung: 40 }, &p, &seeds, &rungs);

        assert_eq!(r.solved, vec![0, 5, 0], "it solves at 40 and nowhere else");
        assert_eq!(r.first_solved_at, vec![Some(40); 5]);
        assert!(
            r.trials.iter().filter(|t| t.run_length == 80).all(|t| !t.solved),
            "the record must keep the miss at 80 that the summary alone would hide"
        );
        assert!(r.sound(), "it returns the true optimum, never anything below it");
        // TTS therefore comes from rung 40, the only one with a success.
        assert_eq!(r.tts(0.99).expect("one rung solves").run_length, 40);
    }

    /// The ladder is log-spaced, strictly increasing, and hits both endpoints exactly.
    #[test]
    fn the_ladder_is_strictly_increasing_between_its_endpoints() {
        let l = ladder(1, 10_000, 9);
        assert_eq!(l[0], 1);
        assert_eq!(l[l.len() - 1], 10_000);
        assert!(l.windows(2).all(|w| w[0] < w[1]), "{l:?}");
        assert_eq!(ladder(5, 5, 4), vec![5]);
        assert_eq!(ladder(3, 900, 1), vec![900]);
    }
}
