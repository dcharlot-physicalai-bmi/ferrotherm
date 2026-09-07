//! Wang–Landau: estimate the density of states, and get every temperature from one run.
//!
//! Every other sampler here runs at a temperature. [`crate::gibbs`] takes a `beta`,
//! [`crate::tempering`] takes a ladder of them, and a curve — energy against temperature, a heat
//! capacity, a free energy — costs one run per point. That is the wrong shape for the question
//! "what does this model do as it cools", which is the question a phase transition is an answer to.
//!
//! Wang–Landau estimates `g(E)`, the number of states at each energy. That is a property of the
//! MODEL and carries no temperature at all, and every thermodynamic quantity follows from it by a
//! sum:
//!
//! ```text
//!     Z(beta) = sum_E g(E) exp(-beta E)
//! ```
//!
//! So one run answers every temperature, including ones nobody thought to ask about, and including
//! the cold ones where an ordinary chain is stuck behind a barrier.
//!
//! # The algorithm, and the two places it is usually got wrong
//!
//! Walk in energy space, accepting a flip with probability `min(1, g(E_old) / g(E_new))` — biased
//! *against* energies already seen often, so the walk flattens its own histogram instead of settling
//! into the Boltzmann distribution. `g` is not known, so it is built while walking: every visit
//! multiplies `g(E)` by a modification factor `f`, and when the histogram is flat enough, `f` is
//! reduced toward 1 and the histogram reset. What converges is the RATIO of the `g`s, so the answer
//! is defined only up to a constant.
//!
//! The first mistake is updating `g` and the histogram only when a move is ACCEPTED. The update
//! belongs to the state the walk is in, and a rejected move leaves it in a state, so a rejection
//! must be counted too — `a_rejected_move_is_still_a_visit` is that claim. Getting it wrong biases
//! `g` toward energies that are easy to leave, and nothing else in the algorithm notices.
//!
//! The second is the constant. `g` known up to a factor cannot be compared with anything, so the
//! estimate is anchored by the one count known exactly: `sum_E g(E) = 2^n`, because every
//! configuration has an energy. That turns a shape into a number, and it is what makes [`Dos::log_z`]
//! directly comparable to [`crate::exact::Elimination::log_partition`] rather than comparable up to
//! an offset.
//!
//! # One bin per energy level, because a spin model's spectrum is discrete
//!
//! Textbook Wang–Landau bins energy into equal-width intervals. On a spin model that is wrong, and
//! it fails in a way that looks like slow convergence rather than a mistake.
//!
//! A 12-spin ring with `J = 1` and `h = 0.2` has 148 distinct energies over a range of 26.4. Cut
//! that into 24 equal bins and several energy LEVELS land exactly on a bin edge — one level per bin
//! for bins 3, 11, 15 and 19, whose mean recorded energy was exactly their upper boundary. Those
//! bins then collect about 72,000 visits where their neighbours collect 188,000, a ratio of 0.39
//! that is permanently under any sensible flatness threshold. The walk cannot converge, at any
//! budget: it reached `ln f = 1e-3` in 1.35 million steps and then made no further progress in four
//! hundred million.
//!
//! So this bins by LEVEL: one bin per distinct energy the model actually has, discovered by walking.
//! There are then no empty bins to wait for, no level split across two bins, and no width to choose.
//! The cost is that a model with a continuous spectrum has as many levels as states, so
//! [`TooManyLevels`] refuses one rather than exhausting memory.

use crate::graph::Graph;
use crate::rng::Pcg;
use std::collections::BTreeMap;

/// The model's spectrum is too finely divided for a level-per-energy density of states.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TooManyLevels {
    /// Distinct energies found before giving up.
    pub found: usize,
    /// The cap.
    pub cap: usize,
}

impl core::fmt::Display for TooManyLevels {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "this model has at least {} distinct energies, past a cap of {}. A density of states \
             indexed by level needs a spectrum with far fewer levels than states, which a model \
             with commensurate couplings has and a model with arbitrary real couplings does not",
            self.found, self.cap
        )
    }
}

impl core::error::Error for TooManyLevels {}

/// The walk did not converge within the budget it was given.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DidNotConverge {
    /// Steps taken before giving up.
    pub steps: u64,
    /// How many times the modification factor was halved.
    pub refinements: u32,
    /// Distinct energy levels found.
    pub levels: usize,
}

impl core::fmt::Display for DidNotConverge {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "Wang-Landau did not reach its final modification factor in {} steps: {} refinements \
             over {} energy levels",
            self.steps, self.refinements, self.levels
        )
    }
}

impl core::error::Error for DidNotConverge {}

/// Why a walk stopped.
#[derive(Clone, Debug, PartialEq)]
pub enum Failed {
    /// The spectrum is too fine to index by level.
    TooManyLevels(TooManyLevels),
    /// The budget ran out first.
    DidNotConverge(DidNotConverge),
}

impl core::fmt::Display for Failed {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Failed::TooManyLevels(e) => write!(f, "{e}"),
            Failed::DidNotConverge(e) => write!(f, "{e}"),
        }
    }
}

impl core::error::Error for Failed {}

/// An estimated density of states, one entry per energy level.
#[derive(Clone, Debug)]
pub struct Dos {
    /// The distinct energies, ascending.
    pub energy: Vec<f64>,
    /// `ln g(E)` for each, normalised so the densities sum to `2^n`.
    pub log_g: Vec<f64>,
    /// Steps the walk took.
    pub steps: u64,
}

impl Dos {
    /// `log Z(beta)`, for any beta, from the one run.
    ///
    /// This is what a density of states is for: the temperature enters HERE and not in the sampling,
    /// so a curve costs one sum per point rather than one chain per point.
    #[must_use]
    pub fn log_z(&self, beta: f64) -> f64 {
        log_sum_exp(&self.weights(beta))
    }

    /// `ln g(E) - beta E` per level, the log-weights every moment below is taken against.
    fn weights(&self, beta: f64) -> Vec<f64> {
        self.log_g.iter().zip(&self.energy).map(|(lg, e)| lg - beta * e).collect()
    }

    /// Normalised probability of each level at `beta`.
    fn probabilities(&self, beta: f64) -> Vec<f64> {
        let w = self.weights(beta);
        let z = log_sum_exp(&w);
        w.iter().map(|x| (x - z).exp()).collect()
    }

    /// Mean energy at `beta`.
    #[must_use]
    pub fn mean_energy(&self, beta: f64) -> f64 {
        self.probabilities(beta).iter().zip(&self.energy).map(|(p, e)| p * e).sum()
    }

    /// Heat capacity per spin at `beta`: `beta^2 (<E^2> - <E>^2) / n`.
    ///
    /// The quantity a fixed-temperature sampler is worst at. It is a VARIANCE, so it is set by the
    /// tails of the energy distribution — the states such a chain visits least — which is why it is
    /// the sharpest check on a density of states and the one that fails first when the wings are
    /// wrong.
    #[must_use]
    pub fn heat_capacity(&self, beta: f64, n: usize) -> f64 {
        let p = self.probabilities(beta);
        let m1: f64 = p.iter().zip(&self.energy).map(|(p, e)| p * e).sum();
        let m2: f64 = p.iter().zip(&self.energy).map(|(p, e)| p * e * e).sum();
        beta * beta * (m2 - m1 * m1) / n as f64
    }

    /// Entropy per spin at `beta`, `(ln Z + beta <E>) / n`.
    #[must_use]
    pub fn entropy(&self, beta: f64, n: usize) -> f64 {
        (self.log_z(beta) + beta * self.mean_energy(beta)) / n as f64
    }
}

/// Sum in the log domain without overflowing.
fn log_sum_exp(v: &[f64]) -> f64 {
    let m = v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !m.is_finite() {
        return m;
    }
    m + v.iter().map(|x| (x - m).exp()).sum::<f64>().ln()
}

/// How flat is flat enough: the least-visited level must reach this fraction of the mean.
///
/// The literature's usual value. It is a stopping rule and not a correctness condition — the
/// estimate is unbiased as `f` goes to one whatever this is — so it trades run time against how much
/// error is left when `f` stops shrinking.
pub const FLATNESS: f64 = 0.8;

/// The most distinct energies a walk will index before refusing.
pub const MAX_LEVELS: usize = 4096;

/// A Wang–Landau walk over a model's energy levels.
pub struct Wl<'g> {
    graph: &'g Graph,
    /// Level key (see `key`) to index into the vectors below.
    index: BTreeMap<i64, usize>,
    energy: Vec<f64>,
    log_g: Vec<f64>,
    hist: Vec<u64>,
    s: Vec<i8>,
    e: f64,
    /// Rounding step for level identity, from the model's own scale.
    quantum: f64,
    rng: Pcg,
}

impl<'g> Wl<'g> {
    /// A walk over `g`, seeded.
    #[must_use]
    pub fn new(g: &'g Graph, seed: u64) -> Wl<'g> {
        let scale: f64 = g.w.iter().map(|x| x.abs()).sum::<f64>() / 2.0
            + g.h.iter().map(|x| x.abs()).sum::<f64>();
        let mut rng = Pcg::new(seed, 0x_0D05);
        let s: Vec<i8> = (0..g.n).map(|_| rng.spin(0.5)).collect();
        let e = g.energy(&s);
        Wl {
            graph: g,
            index: BTreeMap::new(),
            energy: Vec::new(),
            log_g: Vec::new(),
            hist: Vec::new(),
            s,
            e,
            // A level is "the same energy" within a billionth of the model's total scale. Two
            // energies really this close are indistinguishable to the walk anyway, and without a
            // tolerance the same physical level reached by two different routes would key to two
            // levels on a last-bit difference.
            quantum: (scale.max(1.0)) * 1e-9,
            rng,
        }
    }

    /// The identity of an energy level: the energy rounded to the model's own resolution.
    #[inline]
    #[allow(clippy::cast_possible_truncation)]
    fn key(&self, e: f64) -> i64 {
        (e / self.quantum).round() as i64
    }

    /// The level an energy belongs to, adding it if it is new.
    ///
    /// A newly-found level starts at the smallest `ln g` seen so far rather than at zero. Zero would
    /// make a level discovered late look emptier than any other and pull the walk into it hard; the
    /// minimum makes it merely the most attractive, which is what a genuinely rare energy should be.
    fn level_of(&mut self, e: f64) -> Result<usize, TooManyLevels> {
        let k = self.key(e);
        if let Some(&i) = self.index.get(&k) {
            return Ok(i);
        }
        if self.energy.len() >= MAX_LEVELS {
            return Err(TooManyLevels { found: self.energy.len(), cap: MAX_LEVELS });
        }
        let floor = self.log_g.iter().copied().fold(f64::INFINITY, f64::min);
        let i = self.energy.len();
        self.index.insert(k, i);
        self.energy.push(e);
        self.log_g.push(if floor.is_finite() { floor } else { 0.0 });
        self.hist.push(0);
        Ok(i)
    }

    /// One proposed single-spin flip, with the Wang–Landau acceptance, then a recorded visit.
    fn step(&mut self, ln_f: f64) -> Result<(), TooManyLevels> {
        let i = ((self.rng.f64() * self.graph.n as f64) as usize).min(self.graph.n - 1);
        // E = -f_i s_i + (terms without i), so flipping s_i moves the energy by 2 f_i s_i.
        let de = 2.0 * self.graph.field(i, &self.s) * f64::from(self.s[i]);
        let e_new = self.e + de;
        let a = self.level_of(self.e)?;
        let b = self.level_of(e_new)?;
        // min(1, g(old)/g(new)) in the log domain. Biased AGAINST levels already built up, which is
        // what flattens the histogram.
        let ln_accept = self.log_g[a] - self.log_g[b];
        if ln_accept >= 0.0 || self.rng.f64() < ln_accept.exp() {
            self.s[i] = -self.s[i];
            // Re-derived from the level rather than carried, so the walk's energy stays exactly on
            // the level it is recorded against however long it runs.
            self.e = self.energy[b];
        }
        // Recorded for the state the walk is IN -- which a rejected proposal leaves it in.
        let here = self.level_of(self.e)?;
        self.log_g[here] += ln_f;
        self.hist[here] += 1;
        Ok(())
    }

    /// Is the histogram flat across every level found?
    ///
    /// This one test is also what protects a level set that is still growing. A level found late has
    /// few visits against the mean, so the ratio fails, so `ln f` does not halve until that level
    /// has been sampled like the rest. Three separate guards for that case were written here and all
    /// three measured inert, because this already covered it — see `run`.
    fn flat(&self) -> bool {
        if self.hist.is_empty() {
            return false;
        }
        let mean = self.hist.iter().sum::<u64>() as f64 / self.hist.len() as f64;
        self.hist.iter().all(|&h| h as f64 >= FLATNESS * mean)
    }

    /// Run until the modification factor falls below `ln_f_final`, or the budget runs out.
    ///
    /// # One mechanism protects an incomplete level set, and it is not the two that were tried
    ///
    /// Flatness is judged over the levels found so far, which at the first check may be one — and a
    /// histogram over one level is perfectly flat, so `ln f` could halve before the walk has been
    /// anywhere. Two guards against that were written here, and BOTH were removed after measuring
    /// them: a discovery phase that walked at a fixed `ln f = 1` until no new level appeared, and a
    /// histogram reset whenever a level was found late.
    ///
    /// Worst error in `ln g`, with each and without, over six to eight seeds per fixture:
    ///
    /// ```text
    ///                        discovery on / off        reset on / off
    ///   ring(12, 1, 0.2)      0.0536 / 0.0552          0.0590 / 0.0590
    ///   ring(16, 1, 0.2)      0.0978 / 0.0606          0.0746 / 0.0761
    ///   4x4 lattice           0.0292 / 0.0226          0.0190 / 0.0214
    ///   5x5 lattice           0.0193 / 0.0220          0.0209 / 0.0209
    /// ```
    ///
    /// No direction, no fixture outside noise, and every run found the complete level set either
    /// way. Both are gone.
    ///
    /// A third guard, a minimum of twenty visits per level before flatness could be declared, went
    /// the same way for the same reason.
    ///
    /// All three were inert because the flatness test already covers the case, and it is worth being
    /// precise about how, since three plausible mechanisms in a row turned out not to be the one. A
    /// level found late has few visits against the mean, so `all(h >= 0.8 * mean)` is false, so
    /// `ln f` does not halve — and it keeps not halving until that level has been sampled like the
    /// rest. The flatness ratio is the guard. Nothing else was needed, and a mutation dropping that
    /// ratio is caught.
    ///
    /// # Halving stops helping, so it stops: the 1/t schedule
    ///
    /// Halving `f` on a flat histogram is the original recipe and it does not converge. The error
    /// SATURATES: whatever statistical error a stage ends with is frozen into `ln g` when `f` drops,
    /// and later stages are too weak to remove it. Measured here on a 4x4 lattice over four seeds,
    /// the worst error in `ln g` went 0.2392, 0.2147, 0.2110, 0.2104 as the target went 1e-4 to
    /// 1e-7 — three more orders of magnitude of work for nothing.
    ///
    /// So once `ln f` falls to `1/t`, with `t` the elapsed sweeps, this stops halving and sets
    /// `ln f = 1/t` at every step thereafter (Belardinelli & Pereyra 2007). A factor that decreases
    /// continuously rather than in frozen stages has no error to freeze, and the remaining error
    /// falls as `1/sqrt(t)` instead of stalling. Flatness is not consulted after the switch — there
    /// is nothing left for it to decide.
    ///
    /// The switch is a handover, not a replacement, and the ORDER is what matters. Starting in the
    /// `1/t` regime instead of arriving at it is a disaster: worst error in `ln g` of 7.7, 7.0, 12.1
    /// and 34.7 on the four fixtures above, against 0.06 to 0.02 for the handover. `1/t` refines an
    /// estimate and cannot build one, because early on `1/t` is enormous and the density it writes is
    /// noise that later, tinier increments can never repair.
    ///
    /// # Errors
    ///
    /// [`Failed::TooManyLevels`] for a spectrum too fine to index, and
    /// [`Failed::DidNotConverge`] when `max_steps` runs out first.
    pub fn run(&mut self, ln_f_final: f64, max_steps: u64) -> Result<Dos, Failed> {
        let mut steps = 0u64;
        let go = |w: &mut Self, f: f64| w.step(f).map_err(Failed::TooManyLevels);
        let mut ln_f = 1.0f64;
        let mut refinements = 0u32;
        let mut one_over_t = false;
        while ln_f > ln_f_final {
            let check_every = (self.energy.len() as u64 * 400).max(4_000);
            for _ in 0..check_every {
                if one_over_t {
                    // Sweeps, not raw steps: `t` has to be an amount of simulated time, and a step
                    // means less on a bigger model.
                    ln_f = self.graph.n as f64 / steps.max(1) as f64;
                    if ln_f <= ln_f_final {
                        break;
                    }
                }
                go(self, ln_f)?;
                steps += 1;
            }
            if ln_f <= ln_f_final {
                break;
            }
            if steps >= max_steps {
                return Err(Failed::DidNotConverge(DidNotConverge {
                    steps,
                    refinements,
                    levels: self.energy.len(),
                }));
            }
            if one_over_t {
                continue;
            }
            if self.flat() {
                ln_f *= 0.5;
                refinements += 1;
                self.hist.iter_mut().for_each(|h| *h = 0);
                if ln_f <= self.graph.n as f64 / steps.max(1) as f64 {
                    one_over_t = true;
                }
            }
        }
        Ok(self.finish(steps))
    }

    /// Normalise, sort by energy, and package.
    fn finish(&self, steps: u64) -> Dos {
        let mut order: Vec<usize> = (0..self.energy.len()).collect();
        order.sort_by(|&a, &b| self.energy[a].total_cmp(&self.energy[b]));
        let energy: Vec<f64> = order.iter().map(|&i| self.energy[i]).collect();
        let mut log_g: Vec<f64> = order.iter().map(|&i| self.log_g[i]).collect();
        // The anchor. `g` is built up to a multiplicative constant, and the constant is fixed by the
        // one count known exactly: every configuration has an energy, so the densities sum to 2^n.
        let total = log_sum_exp(&log_g);
        let target = self.graph.n as f64 * core::f64::consts::LN_2;
        for x in &mut log_g {
            *x += target - total;
        }
        Dos { energy, log_g, steps }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exact::Elimination;

    /// The true density of states, by enumeration: energy to count.
    fn exact_dos(g: &Graph) -> BTreeMap<i64, (f64, u64)> {
        let mut out: BTreeMap<i64, (f64, u64)> = BTreeMap::new();
        for m in 0u64..(1u64 << g.n) {
            let s: Vec<i8> = (0..g.n).map(|i| if m >> i & 1 == 1 { 1i8 } else { -1 }).collect();
            let e = g.energy(&s);
            #[allow(clippy::cast_possible_truncation)]
            let k = (e / 1e-9).round() as i64;
            let slot = out.entry(k).or_insert((e, 0));
            slot.1 += 1;
        }
        out
    }

    /// The walk finds exactly the energies the model has — no more, no fewer.
    ///
    /// The failure this module was rebuilt around. With equal-width bins, a level landing on a bin
    /// edge is split, producing bins that hold a fraction of a level's states and can never reach
    /// the flatness threshold. Level indexing makes that unrepresentable, and this is the check that
    /// says so: the level set and the enumerated spectrum are the same set.
    #[test]
    fn the_walk_finds_exactly_the_energies_the_model_has() {
        for (name, g) in [
            ("ring 12, field 0.2", crate::ising::ring(12, 1.0, 0.2)),
            ("ring 12, no field", crate::ising::ring(12, 1.0, 0.0)),
            ("4x4 lattice", crate::ising::lattice2d(4, 1.0)),
        ] {
            let want = exact_dos(&g);
            let mut wl = Wl::new(&g, 4);
            let dos = wl.run(1e-4, 40_000_000).unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(
                dos.energy.len(),
                want.len(),
                "{name}: found {} levels, the model has {}",
                dos.energy.len(),
                want.len()
            );
            for (got, wanted) in dos.energy.iter().zip(want.values()) {
                assert!(
                    (got - wanted.0).abs() < 1e-9,
                    "{name}: level {got} against {}",
                    wanted.0
                );
            }
        }
    }

    /// The density of states matches enumeration, level for level.
    ///
    /// Checked as an absolute count, not a shape: normalising to `2^n` means `log_g` is a number of
    /// states rather than a number of states times an unknown constant, so nothing is fitted here.
    #[test]
    fn the_density_of_states_matches_enumeration() {
        for (name, g) in [
            ("ring 12, field 0.2", crate::ising::ring(12, 1.0, 0.2)),
            ("ring 12, no field", crate::ising::ring(12, 1.0, 0.0)),
            ("4x4 lattice", crate::ising::lattice2d(4, 1.0)),
        ] {
            let want = exact_dos(&g);
            let mut wl = Wl::new(&g, 7);
            let dos = wl.run(1e-5, 80_000_000).unwrap_or_else(|e| panic!("{name}: {e}"));

            let total = log_sum_exp(&dos.log_g);
            assert!(
                (total - g.n as f64 * core::f64::consts::LN_2).abs() < 1e-9,
                "{name}: the densities must sum to 2^n; got ln {total}"
            );
            for ((e, got), (_, count)) in dos.energy.iter().zip(&dos.log_g).zip(want.values()) {
                let wanted = (*count as f64).ln();
                // 0.15 is not a round number, it is a DISCRIMINATING one. On this fixture the
                // 1/t schedule reaches a worst error of 0.088 and the original halving schedule
                // saturates at 0.215, so a tolerance between them makes the schedule itself the
                // thing under test: revert `run` to halving and this fails.
                assert!(
                    (got - wanted).abs() < 0.15,
                    "{name}: E = {e}, ln g = {got:.4} against an enumerated {wanted:.4}"
                );
            }
        }
    }

    /// One run reproduces the whole temperature curve, against exact elimination at each point.
    ///
    /// The claim the module exists for. `log_z` here comes from a density of states that never saw a
    /// temperature; `log_partition` is an exact computation at each beta. Two different algorithms
    /// answering the same question, over a factor of forty in temperature including the ordered end
    /// where a single chain would be stuck.
    #[test]
    fn one_run_reproduces_the_whole_temperature_curve() {
        let g = crate::ising::ring(12, 1.0, 0.2);
        let mut wl = Wl::new(&g, 11);
        let dos = wl.run(1e-5, 80_000_000).expect("this budget is generous");
        let elim = Elimination::default();
        for beta in [0.05, 0.2, 0.5, 1.0, 2.0] {
            let want = elim.log_partition(&g, beta).unwrap().log_z.unwrap();
            let got = dos.log_z(beta);
            assert!(
                (got - want).abs() < 0.1,
                "beta {beta}: log Z {got:.5} from one Wang-Landau run against an exact {want:.5}"
            );
        }
    }

    /// Mean energy and heat capacity follow from the same estimate.
    ///
    /// The heat capacity is the sharper of the two: it is a variance, so it weighs the tails of the
    /// energy distribution, and a density that is slightly wrong in the wings shows up here before
    /// it shows up in a mean.
    #[test]
    fn the_derived_quantities_follow_from_the_same_estimate() {
        let g = crate::ising::ring(12, 1.0, 0.0);
        let mut wl = Wl::new(&g, 3);
        let dos = wl.run(1e-5, 80_000_000).expect("this budget is generous");

        for beta in [0.1, 0.4, 1.0] {
            let (mut z, mut m1, mut m2) = (0.0f64, 0.0f64, 0.0f64);
            for m in 0u64..(1u64 << g.n) {
                let s: Vec<i8> =
                    (0..g.n).map(|i| if m >> i & 1 == 1 { 1i8 } else { -1 }).collect();
                let e = g.energy(&s);
                let w = (-beta * e).exp();
                z += w;
                m1 += w * e;
                m2 += w * e * e;
            }
            let want_e = m1 / z;
            let want_c = beta * beta * (m2 / z - want_e * want_e) / g.n as f64;
            assert!(
                (dos.mean_energy(beta) - want_e).abs() < 0.05,
                "beta {beta}: <E> {:.4} against {want_e:.4}",
                dos.mean_energy(beta)
            );
            assert!(
                (dos.heat_capacity(beta, g.n) - want_c).abs() < 0.02,
                "beta {beta}: C {:.4} against {want_c:.4}",
                dos.heat_capacity(beta, g.n)
            );
        }
    }

    /// A rejected move is still a visit.
    ///
    /// The classic Wang-Landau mistake: the update belongs to the state the walk is IN, and a
    /// rejected proposal leaves it in one. Counting only accepted moves under-weights energies that
    /// are hard to leave. Checked directly, because the bias is not reliably large enough to fail
    /// the comparisons above — the histogram must grow once per PROPOSAL even when most are refused.
    #[test]
    fn a_rejected_move_is_still_a_visit() {
        let g = crate::ising::lattice2d(4, 1.0);
        let mut wl = Wl::new(&g, 9);
        for _ in 0..20_000 {
            wl.step(1.0).unwrap();
        }
        // Make every level but the current one look enormously populated, so leaving is refused.
        let here = wl.level_of(wl.e).unwrap();
        for (i, x) in wl.log_g.iter_mut().enumerate() {
            *x = if i == here { 0.0 } else { 1e6 };
        }
        wl.hist.iter_mut().for_each(|h| *h = 0);
        let before = wl.s.clone();
        for _ in 0..2_000 {
            wl.step(1.0).unwrap();
        }
        let visits: u64 = wl.hist.iter().sum();
        assert_eq!(visits, 2_000, "every proposal must be recorded, not every acceptance");
        let moved = before.iter().zip(&wl.s).filter(|(a, b)| a != b).count();
        assert!(
            moved < g.n,
            "this fixture is meant to refuse almost every move; {moved} of {} spins changed",
            g.n
        );
    }

    /// The walk's energy is always exactly a level's energy, not merely close to one.
    ///
    /// `step` sets `self.e` from the level table rather than from the arithmetic that proposed the
    /// move. Carrying the incremental value instead is not WRONG so much as un-guaranteed: measured
    /// drift is 1.8e-11 over twenty million flips against a level quantum of 3.2e-8, a margin of
    /// about 1700, so no accuracy check can tell the two apart and a mutation swapping them survives
    /// every other test here.
    ///
    /// What distinguishes them is the property itself, which is exact and therefore cheap to assert:
    /// re-deriving makes the energy bit-identical to a level, and carrying makes it a number that
    /// happens to round to one.
    #[test]
    fn the_walks_energy_is_exactly_a_level_and_not_merely_near_one() {
        let g = crate::ising::ring(12, 1.0, 0.2);
        let mut wl = Wl::new(&g, 6);
        for step in 0..200_000 {
            wl.step(0.5).unwrap();
            if step % 1_000 == 0 {
                assert!(
                    wl.energy.iter().any(|&e| e.to_bits() == wl.e.to_bits()),
                    "after {step} steps the walk sits at {} which is no level exactly",
                    wl.e
                );
            }
        }
    }

    /// The flatness predicate answers the question it is named for.
    ///
    /// Tested as a predicate rather than through its effect, because its effect is small: with the
    /// `1/t` schedule downstream, removing the gate entirely costs only 0.076 to 0.114 in worst
    /// `ln g` on the sharpest fixture — real, and under any tolerance the accuracy tests can carry
    /// without becoming seed lotteries. A mutation making `flat` always true therefore survives
    /// every one of them.
    ///
    /// It is still load-bearing enough to keep: the gate is what makes the halving phase build a
    /// usable estimate before the handover, and the handover cannot build one on its own — see
    /// `run`. So the contract is pinned directly, where it is exact.
    #[test]
    fn flatness_is_the_least_visited_level_against_the_mean() {
        let g = crate::ising::ring(6, 1.0, 0.0);
        let mut wl = Wl::new(&g, 1);
        for h in [100u64, 100, 100, 100] {
            wl.hist.push(h);
            wl.energy.push(0.0);
            wl.log_g.push(0.0);
        }
        assert!(wl.flat(), "four equal counts are as flat as a histogram gets");

        wl.hist[2] = 90; // mean 97.5, ratio 0.923 -- above the 0.8 bar
        assert!(wl.flat(), "a level at 0.92 of the mean is within the bar");

        wl.hist[2] = 40; // mean 85, ratio 0.47 -- below it
        assert!(!wl.flat(), "a level at less than half the mean is not flat");

        wl.hist[2] = 0; // a level found but never revisited
        assert!(!wl.flat(), "a level with no visits at all cannot be flat");
    }

    /// A walk that cannot converge says so, rather than running forever or returning a guess.
    #[test]
    fn an_exhausted_budget_is_an_error_and_not_an_answer() {
        let g = crate::ising::lattice2d(6, 1.0);
        let mut wl = Wl::new(&g, 2);
        match wl.run(1e-8, 20_000) {
            Err(Failed::DidNotConverge(e)) => assert!(e.steps > 0, "{e}"),
            other => panic!("20000 steps cannot refine to 1e-8: {other:?}"),
        }
    }

    /// A spectrum too fine to index is refused by name.
    ///
    /// Every coupling drawn at random makes almost every configuration its own energy level, so the
    /// level set grows with the state space. That is the case level indexing cannot serve, and it is
    /// refused rather than silently exhausting memory.
    #[test]
    fn a_continuous_spectrum_is_refused_rather_than_indexed() {
        let mut rng = Pcg::new(5, 2);
        let mut b = crate::graph::GraphBuilder::new(20);
        for i in 0..20 {
            for j in (i + 1)..20 {
                b.couple(i, j, rng.f64() * 2.0 - 1.0);
            }
        }
        let g = b.build();
        let mut wl = Wl::new(&g, 1);
        match wl.run(1e-4, 50_000_000) {
            Err(Failed::TooManyLevels(e)) => assert_eq!(e.cap, MAX_LEVELS, "{e}"),
            other => panic!("a random dense model has no usable level set: {other:?}"),
        }
    }
}
