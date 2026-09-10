//! Transition-matrix Monte Carlo: the density of states from every attempted move, refusals
//! included.
//!
//! [`crate::wanglandau`] estimates the same `g(E)` by pushing `ln g` up by a modification factor at
//! each visit and shrinking that factor on a schedule. The factor is the whole difficulty — it is a
//! knob, its schedule decides how much error is frozen in, and it throws away most of what a step
//! produces, because a proposal that was refused moves the walk nowhere and teaches Wang-Landau
//! nothing beyond one more visit to where it already was.
//!
//! Broad-histogram / transition-matrix Monte Carlo (Wang & Swendsen 2001; Fitzgerald, Picard &
//! Silver 1999) reads the refusal. Accumulate `C(E -> E')`, the number of ATTEMPTED single-spin
//! flips seen from a state of energy `E` to a state of energy `E'`, taken or not. Row-normalising
//! gives `T`, the infinite-temperature transition matrix over energy levels, and `T` obeys an exact
//! relation with the density of states:
//!
//! ```text
//!     g(E) T(E -> E') = g(E') T(E' -> E)
//! ```
//!
//! Both sides count the same objects — ordered pairs of configurations one flip apart, one end at
//! each energy — and a flip is its own inverse, so the two counts are equal configuration by
//! configuration. It is an identity about the model, not an approximation, and there is no
//! modification factor in it anywhere. On the EXACT matrix it returns `ln g` exactly, which is what
//! `the_solver_is_exact_on_the_enumerated_matrix` asserts at 1e-9.
//!
//! # What the walk is for, and the one assumption it carries
//!
//! `T(E -> E')` is an average over the states AT energy `E`, so a walk collecting it must visit
//! those states uniformly. Any sampler whose acceptance depends on energy alone does: Boltzmann,
//! multicanonical and flat-histogram acceptances are all constant within a level, so their
//! stationary distribution is too. This walk uses its own current estimate of `ln g` as a
//! multicanonical bias, which is what carries it into the tails. The bias steers WHERE the walk
//! goes and never enters the estimate — that is why there is no schedule to tune — and the
//! assumption it does carry is the one just named: while the bias is still moving the chain is not
//! stationary, so within-level uniformity holds only in the limit.
//!
//! # Solving the relation
//!
//! In logs it is linear: `ln g(E') - ln g(E) = ln T(E->E') - ln T(E'->E)`, one equation per level
//! pair seen in both directions, and overdetermined for a walk that saw many. It is solved as a
//! weighted least squares with weight `1 / (1/C_ab + 1/C_ba)`, the variance of that difference of
//! log counts. The normal equations are a weighted graph Laplacian, solved here by conjugate
//! gradients with one level pinned, and the free constant is fixed afterwards by the one count
//! known exactly, `sum_E g(E) = 2^n`.
//!
//! Levels are one per distinct energy rather than equal-width bins, as in [`crate::wanglandau`] and
//! for the reason given there.
//!
//! # What is deliberately not tested here
//!
//! Nothing compares this against the Wang-Landau walk. Two estimators of the same quantity agreeing
//! is not evidence that either is right — they can be wrong together, and when they disagree the
//! test cannot say which one to fix. The oracles used instead are exact: the enumerated transition
//! matrix, enumerated level counts, and [`crate::exact::Elimination::log_partition`].

use crate::graph::Graph;
use crate::rng::Pcg;
use crate::wanglandau::{Dos, MAX_LEVELS, TooManyLevels};
use std::collections::BTreeMap;

/// Why a density of states could not be solved for.
#[derive(Clone, Debug, PartialEq)]
pub enum Refused {
    /// The spectrum is too finely divided to index one level per energy.
    TooManyLevels(TooManyLevels),
    /// No move has been recorded, so there is nothing to solve.
    NothingRecorded,
    /// The levels seen do not form one component under two-way transitions, so their densities have
    /// no common constant and cannot be compared.
    Disconnected {
        /// Levels found.
        levels: usize,
        /// Levels reachable from the first.
        reached: usize,
    },
    /// The least-squares solve did not converge within its iteration cap.
    DidNotSolve {
        /// Gradient norm reached.
        residual: f64,
        /// Norm of the right-hand side the residual is judged against.
        scale: f64,
    },
}

impl core::fmt::Display for Refused {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Refused::TooManyLevels(e) => write!(f, "{e}"),
            Refused::NothingRecorded => write!(f, "no attempted move has been recorded"),
            Refused::Disconnected { levels, reached } => write!(
                f,
                "the transition matrix reaches only {reached} of {levels} energy levels: densities \
                 in different components share no constant, so run longer rather than reporting a \
                 number for each piece"
            ),
            Refused::DidNotSolve { residual, scale } => write!(
                f,
                "the detailed-balance least squares stalled at a gradient of {residual:.3e} \
                 against a right-hand side of {scale:.3e}"
            ),
        }
    }
}

impl core::error::Error for Refused {}

/// The collection matrix: counts of attempted moves between energy levels.
///
/// One row per level found, each holding the levels moved toward and how often. Row `a`'s total is
/// the number of attempts made from level `a`, which for a walk is the number of times it was
/// there.
#[derive(Clone, Debug)]
pub struct Collection {
    /// Rounding step for level identity.
    quantum: f64,
    /// Level key (energy over the quantum) to index.
    index: BTreeMap<i64, usize>,
    energy: Vec<f64>,
    row: Vec<BTreeMap<usize, u64>>,
}

impl Collection {
    /// An empty collection whose levels are identified to within `quantum` in energy.
    #[must_use]
    pub fn new(quantum: f64) -> Collection {
        Collection {
            quantum: if quantum > 0.0 { quantum } else { f64::MIN_POSITIVE },
            index: BTreeMap::new(),
            energy: Vec::new(),
            row: Vec::new(),
        }
    }

    /// An empty collection for `g`, with the level resolution taken from the model's own scale.
    ///
    /// A billionth of the total coupling and field magnitude, so two routes to the same physical
    /// energy key to one level rather than two on a last-bit difference.
    #[must_use]
    pub fn for_graph(g: &Graph) -> Collection {
        let scale = g.w.iter().map(|x| x.abs()).sum::<f64>() / 2.0
            + g.h.iter().map(|x| x.abs()).sum::<f64>();
        Collection::new(scale.max(1.0) * 1e-9)
    }

    /// Levels found so far.
    #[must_use]
    pub fn levels(&self) -> usize {
        self.energy.len()
    }

    /// The energy of each level, in discovery order.
    #[must_use]
    pub fn energies(&self) -> &[f64] {
        &self.energy
    }

    /// Attempted moves recorded from level `from` toward level `to`.
    #[must_use]
    pub fn count(&self, from: usize, to: usize) -> u64 {
        self.row.get(from).and_then(|r| r.get(&to)).copied().unwrap_or(0)
    }

    /// Attempts made from level `l` — for a walk, the number of times it was there.
    #[must_use]
    pub fn visits(&self, l: usize) -> u64 {
        self.row.get(l).map_or(0, |r| r.values().sum())
    }

    /// The level an energy belongs to, adding it if it is new.
    ///
    /// # Errors
    ///
    /// [`TooManyLevels`] past [`MAX_LEVELS`], since a spectrum with as many levels as states cannot
    /// be indexed one level per energy.
    pub fn level_of(&mut self, e: f64) -> Result<usize, TooManyLevels> {
        #[allow(clippy::cast_possible_truncation)]
        let k = (e / self.quantum).round() as i64;
        if let Some(&i) = self.index.get(&k) {
            return Ok(i);
        }
        if self.energy.len() >= MAX_LEVELS {
            return Err(TooManyLevels { found: self.energy.len(), cap: MAX_LEVELS });
        }
        let i = self.energy.len();
        self.index.insert(k, i);
        self.energy.push(e);
        self.row.push(BTreeMap::new());
        Ok(i)
    }

    /// Record one ATTEMPTED move, accepted or not. That distinction is the method's whole point.
    ///
    /// # Panics
    ///
    /// If either index is not a level of this collection — take them from [`Collection::level_of`].
    pub fn record(&mut self, from: usize, to: usize) {
        assert!(
            from < self.row.len() && to < self.row.len(),
            "record({from}, {to}) needs levels of this collection, which has {}",
            self.row.len()
        );
        *self.row[from].entry(to).or_insert(0) += 1;
    }

    /// `ln g` per level in DISCOVERY order, up to the additive constant the relation leaves free.
    fn log_g_by_level(&self) -> Result<Vec<f64>, Refused> {
        let m = self.energy.len();
        if m == 0 || self.row.iter().all(BTreeMap::is_empty) {
            return Err(Refused::NothingRecorded);
        }
        if m == 1 {
            return Ok(vec![0.0]);
        }
        let visits: Vec<f64> = (0..m).map(|a| self.visits(a) as f64).collect();

        // One equation per level pair seen in BOTH directions: a pair seen one way carries no
        // estimate of the reverse rate, and the relation needs both.
        let mut edges: Vec<(usize, usize, f64, f64)> = Vec::new();
        for a in 0..m {
            for (&b, &cab) in &self.row[a] {
                if b <= a {
                    continue;
                }
                let Some(&cba) = self.row[b].get(&a) else { continue };
                let (cab, cba) = (cab as f64, cba as f64);
                // ln T(a->b) - ln T(b->a), which the relation says is ln g(b) - ln g(a). The row
                // total divided by is every attempt made from the level, moves that changed no
                // level included: T is a stochastic matrix and its diagonal is part of the row.
                // Dropping the diagonal renormalises each row by a different factor and breaks the
                // relation, which `the_solver_is_exact_on_the_enumerated_matrix` catches.
                let d = (cab / visits[a]).ln() - (cba / visits[b]).ln();
                // Inverse variance of that difference: var(ln C) is about 1/C on either side. On an
                // exact matrix any positive weight gives the same answer — the system is consistent
                // — so this choice is about variance, not correctness.
                let w = cab * cba / (cab + cba);
                edges.push((a, b, w, d));
            }
        }

        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); m];
        for &(a, b, _, _) in &edges {
            adj[a].push(b);
            adj[b].push(a);
        }
        let mut seen = vec![false; m];
        let mut stack = vec![0usize];
        seen[0] = true;
        let mut reached = 1usize;
        while let Some(v) = stack.pop() {
            for &u in &adj[v] {
                if !seen[u] {
                    seen[u] = true;
                    reached += 1;
                    stack.push(u);
                }
            }
        }
        if reached < m {
            return Err(Refused::Disconnected { levels: m, reached });
        }

        // Normal equations of the weighted least squares: L x = rhs, with L the weighted Laplacian
        // of the level graph. L is singular on the constants, which is exactly the constant the
        // relation leaves free, so level 0 is pinned at zero and the anchor is applied afterwards.
        let mut rhs = vec![0.0f64; m];
        for &(a, b, w, d) in &edges {
            rhs[a] -= w * d;
            rhs[b] += w * d;
        }
        rhs[0] = 0.0;
        let lap = |x: &[f64]| -> Vec<f64> {
            let mut y = vec![0.0f64; m];
            for &(a, b, w, _) in &edges {
                let t = w * (x[a] - x[b]);
                y[a] += t;
                y[b] -= t;
            }
            y[0] = 0.0;
            y
        };
        let dot = |a: &[f64], b: &[f64]| -> f64 { a.iter().zip(b).map(|(x, y)| x * y).sum() };

        let mut x = vec![0.0f64; m];
        let mut r = rhs.clone();
        let mut p = r.clone();
        let mut rs = dot(&r, &r);
        let scale = rs.sqrt();
        let tol = 1e-15 * scale;
        // Conjugate gradients terminate in m steps exactly; the multiple is for rounding.
        for _ in 0..(20 * m + 500) {
            if rs.sqrt() <= tol {
                break;
            }
            let ap = lap(&p);
            let den = dot(&p, &ap);
            if !(den > 0.0) {
                break;
            }
            let alpha = rs / den;
            for i in 0..m {
                x[i] += alpha * p[i];
                r[i] -= alpha * ap[i];
            }
            let rs_new = dot(&r, &r);
            let bk = rs_new / rs;
            for i in 0..m {
                p[i] = r[i] + bk * p[i];
            }
            rs = rs_new;
        }
        // A guard, and it is recorded as one: no fixture in this module reaches it, because the
        // system is symmetric positive definite once a level is pinned and the level graph of a
        // spin model is narrow. Reporting a stalled solve as a density of states is the failure
        // worth refusing even when nothing has yet produced it.
        let residual = rs.sqrt();
        if residual > 1e-8 * scale {
            return Err(Refused::DidNotSolve { residual, scale });
        }
        Ok(x)
    }

    /// Solve `g(E) T(E->E') = g(E') T(E'->E)` for the density of states.
    ///
    /// Sorted by energy and anchored so the densities sum to `2^n_spins`, which makes `ln g` a
    /// number of states rather than a shape — and [`Dos::log_z`] directly comparable to
    /// [`crate::exact::Elimination::log_partition`].
    ///
    /// # Errors
    ///
    /// [`Refused::NothingRecorded`] for an empty matrix, [`Refused::Disconnected`] when the levels
    /// seen do not form one component, [`Refused::DidNotSolve`] if the least squares stalls.
    pub fn solve(&self, n_spins: usize) -> Result<Dos, Refused> {
        let x = self.log_g_by_level()?;
        let m = self.energy.len();
        let mut order: Vec<usize> = (0..m).collect();
        order.sort_by(|&a, &b| self.energy[a].total_cmp(&self.energy[b]));
        let energy: Vec<f64> = order.iter().map(|&i| self.energy[i]).collect();
        let mut log_g: Vec<f64> = order.iter().map(|&i| x[i]).collect();
        let total = log_sum_exp(&log_g);
        let target = n_spins as f64 * core::f64::consts::LN_2;
        for v in &mut log_g {
            *v += target - total;
        }
        Ok(Dos { energy, log_g, steps: (0..m).map(|a| self.visits(a)).sum() })
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

/// A walk that fills a [`Collection`], biased by what it has already solved.
pub struct Tmmc<'g> {
    graph: &'g Graph,
    coll: Collection,
    /// Multicanonical bias per level: the last solved `ln g`, or all zero before any solve, which
    /// is an infinite-temperature random walk.
    bias: Vec<f64>,
    s: Vec<i8>,
    e: f64,
    rng: Pcg,
    steps: u64,
}

impl<'g> Tmmc<'g> {
    /// A walk over `g`, seeded.
    #[must_use]
    pub fn new(g: &'g Graph, seed: u64) -> Tmmc<'g> {
        let mut rng = Pcg::new(seed, 0x_74D4D);
        let s: Vec<i8> = (0..g.n).map(|_| rng.spin(0.5)).collect();
        let e = g.energy(&s);
        Tmmc { graph: g, coll: Collection::for_graph(g), bias: Vec::new(), s, e, rng, steps: 0 }
    }

    /// The matrix collected so far.
    #[must_use]
    pub fn collection(&self) -> &Collection {
        &self.coll
    }

    /// Attempted moves made.
    #[must_use]
    pub fn steps(&self) -> u64 {
        self.steps
    }

    /// The level of an energy, keeping the bias vector as long as the level table.
    ///
    /// A level found late starts at the smallest bias known, so the walk is drawn toward it rather
    /// than repelled — the choice [`crate::wanglandau`] makes, for the same reason.
    fn level(&mut self, e: f64) -> Result<usize, TooManyLevels> {
        let i = self.coll.level_of(e)?;
        while self.bias.len() < self.coll.levels() {
            let floor = self.bias.iter().copied().fold(f64::INFINITY, f64::min);
            self.bias.push(if floor.is_finite() { floor } else { 0.0 });
        }
        Ok(i)
    }

    /// One proposed single-spin flip: recorded either way, then accepted under the current bias.
    fn step(&mut self) -> Result<(), TooManyLevels> {
        let i = ((self.rng.f64() * self.graph.n as f64) as usize).min(self.graph.n - 1);
        // E = -f_i s_i + (terms without i), so flipping s_i moves the energy by 2 f_i s_i.
        let de = 2.0 * self.graph.field(i, &self.s) * f64::from(self.s[i]);
        let a = self.level(self.e)?;
        let b = self.level(self.e + de)?;
        // The datum is the ATTEMPT. Recording only what was accepted would collect the biased
        // chain's own transition matrix instead of the infinite-temperature one.
        self.coll.record(a, b);
        let ln_accept = self.bias[a] - self.bias[b];
        if ln_accept >= 0.0 || self.rng.f64() < ln_accept.exp() {
            self.s[i] = -self.s[i];
            // Re-derived from the level, so the walk's energy stays exactly on the level it is
            // recorded against however long it runs.
            self.e = self.coll.energies()[b];
        }
        self.steps += 1;
        Ok(())
    }

    /// Re-solve the matrix and adopt the answer as the walk's bias.
    ///
    /// A solve that fails — too few levels connected yet — leaves the previous bias in place. The
    /// bias only steers the walk and never enters the estimate, so a stale one costs efficiency and
    /// cannot cost correctness.
    pub fn refresh(&mut self) {
        if let Ok(x) = self.coll.log_g_by_level() {
            self.bias = x;
        }
    }

    /// Walk for `steps` attempted moves, re-solving the bias every `refresh_every` of them, then
    /// solve the matrix.
    ///
    /// `refresh_every` of zero never re-solves, which leaves the walk at infinite temperature: an
    /// unbiased random walk on the hypercube, uniform over states and so the cleanest estimator per
    /// visit, but one that never leaves the middle of the spectrum.
    ///
    /// # Errors
    ///
    /// [`Refused::TooManyLevels`] for a spectrum too fine to index, and whatever
    /// [`Collection::solve`] refuses at the end.
    ///
    /// # Panics
    ///
    /// If the model has no spins.
    pub fn run(&mut self, steps: u64, refresh_every: u64) -> Result<Dos, Refused> {
        assert!(self.graph.n > 0, "a walk needs at least one spin");
        for k in 0..steps {
            self.step().map_err(Refused::TooManyLevels)?;
            if refresh_every > 0 && (k + 1) % refresh_every == 0 {
                self.refresh();
            }
        }
        self.coll.solve(self.graph.n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exact::Elimination;

    /// Every level of the model and how many states it holds, by enumeration.
    fn exact_levels(g: &Graph) -> Vec<(f64, u64)> {
        let set = crate::samples::enumerate(g, 0.0).expect("small enough to enumerate");
        let mut by_key: BTreeMap<i64, (f64, u64)> = BTreeMap::new();
        for &e in set.energies() {
            #[allow(clippy::cast_possible_truncation)]
            let k = (e / 1e-9).round() as i64;
            let slot = by_key.entry(k).or_insert((e, 0));
            slot.1 += 1;
        }
        let mut out: Vec<(f64, u64)> = by_key.into_values().collect();
        out.sort_by(|a, b| a.0.total_cmp(&b.0));
        out
    }

    /// The EXACT infinite-temperature transition matrix: every state, every single flip, no walk.
    fn enumerated_matrix(g: &Graph) -> Collection {
        let set = crate::samples::enumerate(g, 0.0).expect("small enough to enumerate");
        let mut c = Collection::for_graph(g);
        for s in set.states() {
            let e = g.energy(s);
            let a = c.level_of(e).expect("a lattice spectrum has few levels");
            for i in 0..g.n {
                let de = 2.0 * g.field(i, s) * f64::from(s[i]);
                let b = c.level_of(e + de).expect("a lattice spectrum has few levels");
                c.record(a, b);
            }
        }
        c
    }

    /// On the exact matrix the solver returns the exact `ln g` — this is not a fit.
    ///
    /// The claim the method rests on. Sampling is the only source of error in transition-matrix
    /// Monte Carlo: given the true transition counts the detailed-balance relation determines the
    /// density of states outright, so the oracle here is enumeration and the tolerance is a
    /// floating-point one, orders below any statistical test in this file.
    #[test]
    fn the_solver_is_exact_on_the_enumerated_matrix() {
        for (name, g) in [
            ("ring 12, field 0.2", crate::ising::ring(12, 1.0, 0.2)),
            ("ring 12, no field", crate::ising::ring(12, 1.0, 0.0)),
            ("4x4 lattice", crate::ising::lattice2d(4, 1.0)),
        ] {
            let want = exact_levels(&g);
            let dos = enumerated_matrix(&g).solve(g.n).unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(dos.energy.len(), want.len(), "{name}: level count");
            for ((e, got), (e_want, count)) in dos.energy.iter().zip(&dos.log_g).zip(&want) {
                assert!((e - e_want).abs() < 1e-9, "{name}: level {e} against {e_want}");
                let wanted = (*count as f64).ln();
                assert!(
                    (got - wanted).abs() < 1e-9,
                    "{name}: E = {e}, ln g = {got:.15} against an enumerated {wanted:.15}"
                );
            }
        }
    }

    /// The identity the method is built on, checked on the exact matrix as integers.
    ///
    /// `g(E) T(E->E') = g(E') T(E'->E)` because both sides count ordered pairs of configurations one
    /// flip apart — so the raw attempt counts are already symmetric, `C(a,b) == C(b,a)`, before any
    /// normalisation. Asserted with `==` on `u64`, since it is a counting statement and not a
    /// numerical one.
    #[test]
    fn attempted_move_counts_are_symmetric_and_carry_the_density_ratio() {
        let g = crate::ising::ring(10, 1.0, 0.3);
        let c = enumerated_matrix(&g);
        let want = exact_levels(&g);
        let mut pairs = 0;
        for a in 0..c.levels() {
            for b in 0..c.levels() {
                assert_eq!(
                    c.count(a, b),
                    c.count(b, a),
                    "a flip is its own inverse, so levels {a} and {b} must see equal traffic"
                );
                if a < b && c.count(a, b) > 0 {
                    pairs += 1;
                }
            }
        }
        assert!(pairs > 5, "only {pairs} level pairs communicate; this fixture is degenerate");

        // And the relation itself, in the form it is solved from.
        let by_energy: BTreeMap<i64, u64> = want
            .iter()
            .map(|&(e, n)| {
                #[allow(clippy::cast_possible_truncation)]
                let k = (e / 1e-9).round() as i64;
                (k, n)
            })
            .collect();
        let g_of = |l: usize| -> f64 {
            #[allow(clippy::cast_possible_truncation)]
            let k = (c.energies()[l] / 1e-9).round() as i64;
            by_energy[&k] as f64
        };
        for a in 0..c.levels() {
            for b in 0..c.levels() {
                if a == b || c.count(a, b) == 0 {
                    continue;
                }
                let t_ab = c.count(a, b) as f64 / c.visits(a) as f64;
                let t_ba = c.count(b, a) as f64 / c.visits(b) as f64;
                assert!(
                    (g_of(a) * t_ab - g_of(b) * t_ba).abs() < 1e-9,
                    "detailed balance failed on levels {a}, {b}"
                );
            }
        }
    }

    /// Every attempt is recorded, including the ones that were refused.
    ///
    /// The difference from [`crate::wanglandau`], and the reason this converges without a
    /// modification factor. Checked as a count, because it is exact: after `k` proposals the matrix
    /// holds exactly `k` entries however few of them were taken. The second half forces almost every
    /// move to be refused — a bias making every other level look enormously populated — and the
    /// count is unchanged, which a version recording only accepted moves fails outright.
    #[test]
    fn every_attempted_move_is_recorded_including_the_refused_ones() {
        let g = crate::ising::lattice2d(4, 1.0);
        let mut w = Tmmc::new(&g, 9);
        for _ in 0..20_000 {
            w.step().unwrap();
        }
        let total: u64 = (0..w.coll.levels()).map(|a| w.coll.visits(a)).sum();
        assert_eq!(total, 20_000, "one recorded attempt per proposal");

        let here = w.level(w.e).unwrap();
        for (i, x) in w.bias.iter_mut().enumerate() {
            *x = if i == here { 0.0 } else { 1e6 };
        }
        let before = w.s.clone();
        for _ in 0..2_000 {
            w.step().unwrap();
        }
        let after: u64 = (0..w.coll.levels()).map(|a| w.coll.visits(a)).sum();
        assert_eq!(after, 22_000, "a refusal is still an attempt and must still be recorded");
        let moved = before.iter().zip(&w.s).filter(|(a, b)| a != b).count();
        assert!(
            moved < g.n,
            "this fixture is meant to refuse almost every move; {moved} of {} spins changed",
            g.n
        );
    }

    /// The sampled density of states matches enumeration, level for level.
    ///
    /// An absolute count, not a shape: the anchor makes `log_g` a number of states, so nothing here
    /// is fitted. The tolerance comes from the spread and not from the one seed the test runs: over
    /// eight seeds per fixture at this budget the worst error in `ln g` was 0.0206, so 0.04 is a
    /// factor of two of margin — and a fifth of the budget takes that same worst case to 0.062.
    #[test]
    fn the_sampled_density_of_states_matches_enumeration() {
        for (name, g, steps) in [
            ("ring 12, field 0.2", crate::ising::ring(12, 1.0, 0.2), 20_000_000u64),
            ("ring 12, no field", crate::ising::ring(12, 1.0, 0.0), 20_000_000),
            ("4x4 lattice", crate::ising::lattice2d(4, 1.0), 20_000_000),
        ] {
            let want = exact_levels(&g);
            let mut w = Tmmc::new(&g, 7);
            let dos = w.run(steps, 100_000).unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(
                dos.energy.len(),
                want.len(),
                "{name}: found {} levels, the model has {}",
                dos.energy.len(),
                want.len()
            );
            let total = log_sum_exp(&dos.log_g);
            assert!(
                (total - g.n as f64 * core::f64::consts::LN_2).abs() < 1e-9,
                "{name}: the densities must sum to 2^n; got ln {total}"
            );
            for ((e, got), (_, count)) in dos.energy.iter().zip(&dos.log_g).zip(&want) {
                let wanted = (*count as f64).ln();
                assert!(
                    (got - wanted).abs() < 0.04,
                    "{name}: E = {e}, ln g = {got:.4} against an enumerated {wanted:.4}"
                );
            }
        }
    }

    /// One run answers every temperature, against exact elimination at each.
    ///
    /// `log_z` here comes from a density of states that never saw a temperature; `log_partition` is
    /// an exact contraction at each beta. The mean energy is checked against the same enumeration,
    /// over a factor of forty in temperature including the ordered end.
    #[test]
    fn log_z_matches_exact_elimination_across_temperatures() {
        let g = crate::ising::ring(12, 1.0, 0.2);
        let mut w = Tmmc::new(&g, 11);
        let dos = w.run(20_000_000, 100_000).expect("this budget is generous");
        let elim = Elimination::default();
        let levels = exact_levels(&g);
        for beta in [0.05, 0.2, 0.5, 1.0, 2.0] {
            let want = elim.log_partition(&g, beta).unwrap().log_z.unwrap();
            let got = dos.log_z(beta);
            assert!(
                (got - want).abs() < 0.03,
                "beta {beta}: log Z {got:.5} from one matrix against an exact {want:.5}"
            );
            let (mut z, mut m1) = (0.0f64, 0.0f64);
            for &(e, n) in &levels {
                let x = n as f64 * (-beta * e).exp();
                z += x;
                m1 += x * e;
            }
            assert!(
                (dos.mean_energy(beta) - m1 / z).abs() < 0.02,
                "beta {beta}: <E> {:.4} against an enumerated {:.4}",
                dos.mean_energy(beta),
                m1 / z
            );
        }
    }

    /// The bias is what reaches the cold end, and the oracle says so.
    ///
    /// On a model too big to enumerate — a 5x5 torus, 25 spins — the same walk runs with and
    /// without re-solving its bias, and both arms are judged against exact elimination. Without it
    /// the walk is an infinite-temperature random walk: it stops at the first excited level, never
    /// records the ground state at all, and its `log Z` is wrong by about 5 at beta 1 and 13 at
    /// beta 2 (over seeds 1, 3 and 7: 4.71 to 5.30, and 12.74 to 13.35). With it, the same budget
    /// reaches the ground level and lands within 0.024 everywhere.
    ///
    /// This is not one sampler against another. Both arms are measured against
    /// [`crate::exact::Elimination`], which is exact; setting them side by side is only what makes
    /// the mechanism's contribution legible.
    #[test]
    fn the_bias_is_what_reaches_the_cold_end() {
        let g = crate::ising::lattice2d(5, 1.0);
        let elim = Elimination::default();
        let e0 = elim.ground_state(&g).unwrap().ground_energy.unwrap();

        let mut biased = Tmmc::new(&g, 3);
        let with = biased.run(20_000_000, 200_000).expect("a torus has few levels");
        assert!(
            (with.energy[0] - e0).abs() < 1e-9,
            "the biased walk must reach the ground level {e0}, not stop at {}",
            with.energy[0]
        );

        let mut flat = Tmmc::new(&g, 3);
        let without = flat.run(20_000_000, 0).expect("a torus has few levels");
        assert!(
            without.energy[0] > e0 + 1.0,
            "an unbiased walk is not supposed to find the ground level; it reported {}",
            without.energy[0]
        );

        for beta in [0.44, 1.0, 2.0] {
            let want = elim.log_partition(&g, beta).unwrap().log_z.unwrap();
            assert!(
                (with.log_z(beta) - want).abs() < 0.05,
                "beta {beta}: biased log Z {:.4} against an exact {want:.4}",
                with.log_z(beta)
            );
        }
        let want = elim.log_partition(&g, 2.0).unwrap().log_z.unwrap();
        assert!(
            (without.log_z(2.0) - want).abs() > 1.0,
            "an unbiased walk that never saw the ground level cannot have log Z right at beta 2"
        );
    }

    /// Levels that never communicate both ways are refused, not reported side by side.
    ///
    /// Their densities have no constant in common, so a number for each is a number in different
    /// units. This is the state a walk is in before it has been anywhere, and it is why
    /// [`Tmmc::refresh`] keeps its old bias rather than treating an early failure as fatal.
    #[test]
    fn levels_that_never_communicate_are_refused() {
        let mut c = Collection::new(1e-9);
        let a = c.level_of(-4.0).unwrap();
        let b = c.level_of(4.0).unwrap();
        c.record(a, a);
        c.record(b, b);
        assert_eq!(c.solve(4).unwrap_err(), Refused::Disconnected { levels: 2, reached: 1 });

        // One-way traffic is not enough either: the reverse rate is what the relation divides by.
        c.record(a, b);
        assert_eq!(c.solve(4).unwrap_err(), Refused::Disconnected { levels: 2, reached: 1 });

        c.record(b, a);
        let dos = c.solve(4).expect("both directions seen, so the ratio is determined");
        assert_eq!(dos.energy.len(), 2);
    }

    /// An empty matrix is an error rather than a density of states over no levels.
    #[test]
    fn an_empty_matrix_is_refused() {
        let c = Collection::new(1e-9);
        assert_eq!(c.solve(4).unwrap_err(), Refused::NothingRecorded);
        let mut c = Collection::new(1e-9);
        c.level_of(0.0).unwrap();
        assert_eq!(
            c.solve(4).unwrap_err(),
            Refused::NothingRecorded,
            "a level with no attempt recorded is no data"
        );
    }

    /// A spectrum too fine to index by level is refused rather than exhausting memory.
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
        let mut w = Tmmc::new(&g, 1);
        match w.run(1_000_000, 0) {
            Err(Refused::TooManyLevels(e)) => assert_eq!(e.cap, MAX_LEVELS, "{e}"),
            other => panic!("a random dense model has no usable level set: {other:?}"),
        }
    }

    /// The walk's energy is always exactly a level's energy, not merely close to one.
    ///
    /// `step` sets `self.e` from the level table rather than from the arithmetic that proposed the
    /// move, so the energy a move is recorded against and the energy the walk holds are the same
    /// bits. Carrying the incremental value instead drifts, slowly enough that no accuracy test here
    /// can see it.
    #[test]
    fn the_walks_energy_is_exactly_a_level_and_not_merely_near_one() {
        let g = crate::ising::ring(12, 1.0, 0.2);
        let mut w = Tmmc::new(&g, 6);
        for step in 0..200_000 {
            w.step().unwrap();
            if step % 1_000 == 0 {
                assert!(
                    w.coll.energies().iter().any(|&e| e.to_bits() == w.e.to_bits()),
                    "after {step} steps the walk sits at {} which is no level exactly",
                    w.e
                );
            }
        }
    }
}
