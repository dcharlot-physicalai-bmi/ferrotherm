//! Rejection-free kinetic Monte Carlo: the n-fold way, with waiting times.
//!
//! Bortz, Kalos & Lebowitz, *A new algorithm for Monte Carlo simulation of Ising spin systems*,
//! J. Comput. Phys. **17**:10 (1975). The exponential waiting time is Gillespie (1976).
//!
//! Every other chain here proposes a flip and may decline it, and at low temperature it declines
//! nearly all of them: the draw is spent and the state does not move. This one never declines.
//! Each site `i` carries a flip RATE `r_i`; the algorithm draws the next site from the exact
//! rate-weighted distribution `r_i / R` with `R = sum_i r_i`, flips it **unconditionally**, and
//! advances a continuous clock by a waiting time drawn from `Exp(R)`.
//!
//! # The subtlety: a plain average over the visited states is wrong
//!
//! The visited sequence is the *embedded jump chain*, and its stationary distribution is not
//! Boltzmann — it is
//!
//! ```text
//!     nu(s)  proportional to  pi(s) R(s)
//! ```
//!
//! biased toward the states that are easy to leave. Nothing in the run reports this: the chain
//! looks healthy, the energies look plausible, and every expectation is off by the correlation
//! between the observable and the escape rate. The fix is the waiting time. Weighting each visited
//! state by the time spent in it turns the jump chain back into the continuous-time chain, whose
//! stationary distribution IS `pi`, because `E[dwell | s] = 1 / R(s)` cancels the `R(s)` above.
//!
//! Both laws are checked against exact enumeration in this module's tests — the time-weighted
//! histogram against [`crate::ising::exact_boltzmann`], and the unweighted one against
//! `pi(s) R(s)`, which it matches and which is measurably far from `pi`.
//!
//! # Rate classes
//!
//! Sites are grouped into classes of **bit-identical rate**, so a class weighs `rate x count` and a
//! site inside it is drawn uniformly. That is BKL's construction: a uniform square lattice at
//! `h = 0` has three distinct Metropolis rates however large it is, so selection costs O(1) in the
//! number of sites. A model with generic real couplings has no such structure and the table
//! degrades to one class per site — see [`Kmc::classes`], which reports what a given model got.
//!
//! ```
//! use ferrotherm::{ising, kmc::{Kmc, Rates}};
//!
//! let g = ising::lattice2d(4, 1.0);
//! let mut k = Kmc::new(&g, 0.8, Rates::Glauber, 7);
//! let (mut m, mut t) = (0.0, 0.0);
//! k.run(20_000, |s, dt| {
//!     m += dt * s.iter().map(|&v| f64::from(v)).sum::<f64>().abs();
//!     t += dt;
//! });
//! assert!(m / t / g.n as f64 > 0.5, "time-weighted |M|, not a plain average");
//! ```

use crate::graph::Graph;
use crate::rng::Pcg;
use crate::samples::{ENUMERATION_LIMIT, Refused};
use std::collections::BTreeMap;

/// The flip-rate law. Both satisfy detailed balance against the Boltzmann distribution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rates {
    /// `min(1, exp(-beta dE))` — Bortz, Kalos & Lebowitz's own choice.
    Metropolis,
    /// `sigma(-beta dE)` — the heat bath, so this chain is rejection-free random-scan Gibbs.
    Glauber,
}

impl Rates {
    /// Flip rate of a site holding `spin` in local field `field` at inverse temperature `beta`.
    ///
    /// The energy change of the flip is `dE = 2 * spin * field`, which is where the factor of two
    /// in [`crate::kernel::p_up`] comes from.
    #[must_use]
    pub fn rate(self, spin: i8, field: f64, beta: f64) -> f64 {
        let de = 2.0 * f64::from(spin) * field;
        match self {
            // `exp` overflows to +inf for a strongly downhill flip; `min` takes that to 1.
            Rates::Metropolis => (-beta * de).exp().min(1.0),
            // sigma(-2 beta spin field), written through the shared kernel, which avoids the
            // cancellation of `1 - p_up` when the flip is nearly certain.
            Rates::Glauber => crate::kernel::p_up(-f64::from(spin) * field, beta),
        }
    }
}

/// One jump: which site flipped, and how long the chain sat in the state it just left.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Jump {
    /// The site that flipped.
    pub site: usize,
    /// `Exp(R)` residence time of the state **before** the flip — the weight that state earns.
    pub dwell: f64,
    /// `1 / R`, the mean of `dwell`. Using it as the weight instead is exact too and has lower
    /// variance (it is `dwell` conditioned on the state), but it is not a clock.
    pub mean_dwell: f64,
}

/// One rate class: every member has this rate, bit for bit, so the class weighs `rate * len`.
struct Class {
    rate: f64,
    members: Vec<u32>,
}

/// `-0.0` and `0.0` are one rate, not two classes.
#[inline]
fn key(r: f64) -> u64 {
    if r == 0.0 { 0 } else { r.to_bits() }
}

/// Index of a state in a `2^n` enumeration: bit `b` set means spin `b` is `+1`.
///
/// The convention of [`crate::ising::exact_boltzmann`] and [`crate::samples::enumerate`], so the
/// histograms this module builds are directly comparable to both.
#[inline]
fn mask(s: &[i8]) -> usize {
    let mut m = 0usize;
    for (b, &v) in s.iter().enumerate() {
        if v == 1 {
            m |= 1 << b;
        }
    }
    m
}

/// A rejection-free chain: one flip per step, and a continuous clock.
pub struct Kmc<'g> {
    g: &'g Graph,
    beta: f64,
    rates: Rates,
    s: Vec<i8>,
    rng: Pcg,
    time: f64,
    steps: u64,
    /// Per-site flip rate, current for every site at all times.
    rate: Vec<f64>,
    /// Site -> its class, and site -> its slot in that class's member list (O(1) removal).
    class_of: Vec<u32>,
    slot: Vec<u32>,
    classes: Vec<Class>,
    /// Rate bits -> class, so a rate that already has a class finds it. `BTreeMap`, not `HashMap`:
    /// this crate's answers are reproducible byte for byte, and a randomised iteration order is how
    /// that stops being true.
    index: BTreeMap<u64, u32>,
    /// Emptied class slots, reused before growing. Without this a generic real-weighted model
    /// leaks one class per distinct rate it ever saw, which is one per step.
    free: Vec<u32>,
}

impl<'g> Kmc<'g> {
    /// A chain at `beta` under `rates`, started from a random state drawn from `seed`.
    #[must_use]
    pub fn new(g: &'g Graph, beta: f64, rates: Rates, seed: u64) -> Self {
        let mut rng = Pcg::new(seed, 0x4B4D_4321);
        let s = (0..g.n).map(|_| rng.spin(0.5)).collect();
        let mut k = Kmc {
            g,
            beta,
            rates,
            s,
            rng,
            time: 0.0,
            steps: 0,
            rate: vec![0.0; g.n],
            class_of: vec![0; g.n],
            slot: vec![0; g.n],
            classes: Vec::new(),
            index: BTreeMap::new(),
            free: Vec::new(),
        };
        k.rebuild();
        k
    }

    /// The current state.
    #[must_use]
    pub fn state(&self) -> &[i8] {
        &self.s
    }

    /// Elapsed continuous time. One unit is one attempted flip per site.
    #[must_use]
    pub fn time(&self) -> f64 {
        self.time
    }

    /// Flips performed. Every step is a flip, so this is also the number of proposals made.
    #[must_use]
    pub fn steps(&self) -> u64 {
        self.steps
    }

    /// Inverse temperature.
    #[must_use]
    pub fn beta(&self) -> f64 {
        self.beta
    }

    /// Energy of the current state.
    #[must_use]
    pub fn energy(&self) -> f64 {
        self.g.energy(&self.s)
    }

    /// `R = sum_i r_i`, the total escape rate of the current state.
    #[must_use]
    pub fn total_rate(&self) -> f64 {
        let mut t = 0.0;
        for c in &self.classes {
            t += c.rate * c.members.len() as f64;
        }
        t
    }

    /// Live rate classes: three for a uniform square lattice at `h = 0`, one per site for generic
    /// real couplings, where the class table buys nothing.
    #[must_use]
    pub fn classes(&self) -> usize {
        self.index.len()
    }

    /// Single-site attempts a rejection-based chain would have made in the same elapsed time.
    ///
    /// Each site attempts one flip per unit time and accepts with probability `r_i`, so the count
    /// is `n * t` in expectation. Against [`Kmc::steps`] this is the saving, and under
    /// [`Rates::Glauber`] the attempt being skipped is exactly one Gibbs draw.
    #[must_use]
    pub fn gibbs_draws_replaced(&self) -> f64 {
        self.g.n as f64 * self.time
    }

    /// Move to `s`, rebuilding the rate table. The clock and the step count are left alone.
    ///
    /// # Panics
    ///
    /// If `s` is the wrong length or holds anything but `+1` and `-1`.
    pub fn set_state(&mut self, s: &[i8]) {
        assert_eq!(s.len(), self.g.n, "state must have one spin per node");
        assert!(s.iter().all(|&v| v == 1 || v == -1), "spins are +1 or -1");
        self.s.copy_from_slice(s);
        self.rebuild();
    }

    /// Change the inverse temperature, rebuilding the rate table. Annealing moves a number.
    pub fn set_beta(&mut self, beta: f64) {
        self.beta = beta;
        self.rebuild();
    }

    /// One jump: flip the drawn site and advance the clock. `None` when every rate has underflowed
    /// to zero, which freezes the chain.
    ///
    /// **[`Jump::dwell`] is the weight of the state that has just been LEFT.** A caller
    /// accumulating an observable must have read it before this call; [`Kmc::run`] does that
    /// ordering for you and is the safer path.
    pub fn step(&mut self) -> Option<Jump> {
        let (site, dwell, mean_dwell) = self.draw()?;
        self.apply(site, dwell);
        Some(Jump { site, dwell, mean_dwell })
    }

    /// Take up to `steps` jumps, calling `f(state, dwell)` on the state being left and the time it
    /// held. Returns the jumps actually taken, short only if the chain froze.
    pub fn run<F: FnMut(&[i8], f64)>(&mut self, steps: usize, mut f: F) -> usize {
        for k in 0..steps {
            let Some((site, dwell, _)) = self.draw() else { return k };
            f(&self.s, dwell);
            self.apply(site, dwell);
        }
        steps
    }

    /// Time-weighted occupancy over all `2^n` states, indexed as [`crate::ising::exact_boltzmann`].
    ///
    /// This is the estimator the module exists for: the weights are dwell times, so the result
    /// converges to the Boltzmann distribution and not to the jump chain's.
    ///
    /// A chain that froze before earning any time reports all its mass on the state it froze in,
    /// which is where it would stay forever.
    ///
    /// # Errors
    ///
    /// [`Refused::TooLargeToEnumerate`] past [`ENUMERATION_LIMIT`] spins.
    pub fn occupancy(&mut self, steps: usize) -> Result<Vec<f64>, Refused> {
        if self.g.n > ENUMERATION_LIMIT {
            return Err(Refused::TooLargeToEnumerate { spins: self.g.n, limit: ENUMERATION_LIMIT });
        }
        let mut w = vec![0.0f64; 1usize << self.g.n];
        self.run(steps, |s, dwell| w[mask(s)] += dwell);
        let z: f64 = w.iter().sum();
        if z > 0.0 {
            for v in &mut w {
                *v /= z;
            }
        } else {
            w[mask(&self.s)] = 1.0;
        }
        Ok(w)
    }

    // ---- internals ----------------------------------------------------------------------------

    /// Draw the waiting time and the site, without moving. `None` if `R` is zero.
    fn draw(&mut self) -> Option<(usize, f64, f64)> {
        let total = self.total_rate();
        if !(total > 0.0) {
            return None;
        }
        // `f64()` is [0, 1); the waiting time needs (0, 1] or `ln` sees zero.
        let u = 1.0 - self.rng.f64();
        let dwell = -u.ln() / total;

        // Accumulated in the SAME order and with the same arithmetic as `total_rate`, so the final
        // partial sum equals `total` exactly and `x < total` must land somewhere; `last` carries
        // the answer out so a rounding tie on the very last class cannot fall off the end.
        let x = self.rng.f64() * total;
        let mut acc = 0.0;
        let mut last = usize::MAX;
        for (c, cl) in self.classes.iter().enumerate() {
            if cl.members.is_empty() {
                continue;
            }
            last = c;
            acc += cl.rate * cl.members.len() as f64;
            if x < acc {
                break;
            }
        }
        // `last` is set, because some class is non-empty whenever `total > 0`.
        let members = &self.classes[last].members;
        let len = members.len();
        let pick = ((self.rng.f64() * len as f64) as usize).min(len - 1);
        Some((members[pick] as usize, dwell, 1.0 / total))
    }

    /// Flip `site`, refresh the rates it changed, advance the clock.
    fn apply(&mut self, site: usize, dwell: f64) {
        let g = self.g;
        self.s[site] = -self.s[site];
        self.refresh(site);
        for k in g.offset[site]..g.offset[site + 1] {
            self.refresh(g.nbr[k] as usize);
        }
        self.time += dwell;
        self.steps += 1;
    }

    /// Recompute site `i`'s rate from the current state and re-file it if it changed class.
    ///
    /// Recomputed rather than updated incrementally: an incrementally maintained local field
    /// drifts, and a rate table that has drifted samples a distribution nobody wrote down.
    fn refresh(&mut self, i: usize) {
        let r = self.rates.rate(self.s[i], self.g.field(i, &self.s), self.beta);
        debug_assert!(r >= 0.0 && r.is_finite(), "rate {r} at site {i} is not a rate");
        if key(r) != key(self.rate[i]) {
            self.detach(i);
            self.attach(i, r);
        }
    }

    /// Rebuild every class from the current state.
    fn rebuild(&mut self) {
        self.classes.clear();
        self.index.clear();
        self.free.clear();
        for i in 0..self.g.n {
            let r = self.rates.rate(self.s[i], self.g.field(i, &self.s), self.beta);
            self.attach(i, r);
        }
    }

    /// Remove `i` from its class by swapping the last member into its slot.
    fn detach(&mut self, i: usize) {
        let c = self.class_of[i] as usize;
        let p = self.slot[i] as usize;
        let last = self.classes[c].members.pop().expect("a class holds its members");
        if p < self.classes[c].members.len() {
            self.classes[c].members[p] = last;
            self.slot[last as usize] = p as u32;
        }
        if self.classes[c].members.is_empty() {
            self.index.remove(&key(self.classes[c].rate));
            self.classes[c].rate = 0.0;
            self.free.push(c as u32);
        }
    }

    /// File `i` under rate `r`, creating or reusing a class slot as needed.
    fn attach(&mut self, i: usize, r: f64) {
        let c = if let Some(&c) = self.index.get(&key(r)) {
            c as usize
        } else {
            let c = if let Some(c) = self.free.pop() {
                c as usize
            } else {
                self.classes.push(Class { rate: r, members: Vec::new() });
                self.classes.len() - 1
            };
            self.classes[c].rate = r;
            self.index.insert(key(r), c as u32);
            c
        };
        self.slot[i] = self.classes[c].members.len() as u32;
        self.classes[c].members.push(i as u32);
        self.class_of[i] = c as u32;
        self.rate[i] = r;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::GraphBuilder;
    use crate::ising::{exact_boltzmann, lattice2d, tv};

    /// A small dense spin glass with fields, so nothing about the model is symmetric or special.
    fn glass(n: usize, seed: u64) -> Graph {
        let mut r = Pcg::new(seed, 0x0C0F_FEE0);
        let mut gb = GraphBuilder::new(n);
        for i in 0..n {
            for j in (i + 1)..n {
                if r.f64() < 0.6 {
                    gb.couple(i, j, if r.f64() < 0.5 { 0.8 } else { -0.8 });
                }
            }
        }
        for i in 0..n {
            gb.bias(i, 0.7 * (r.f64() - 0.5));
        }
        gb.build()
    }

    /// `R(s)` for an arbitrary state, straight from the definition.
    fn escape_rate(g: &Graph, s: &[i8], beta: f64, rates: Rates) -> f64 {
        (0..g.n).map(|i| rates.rate(s[i], g.field(i, s), beta)).sum()
    }

    /// The jump chain's exact stationary law, `nu(s) ∝ pi(s) R(s)`.
    fn jump_law(g: &Graph, beta: f64, rates: Rates) -> Vec<f64> {
        let pi = exact_boltzmann(g, beta);
        let mut nu = vec![0.0; pi.len()];
        let mut s = vec![-1i8; g.n];
        for (m, v) in nu.iter_mut().enumerate() {
            for b in 0..g.n {
                s[b] = if m >> b & 1 == 1 { 1 } else { -1 };
            }
            *v = pi[m] * escape_rate(g, &s, beta, rates);
        }
        let z: f64 = nu.iter().sum();
        for v in &mut nu {
            *v /= z;
        }
        nu
    }

    /// Normalise a histogram in place.
    fn norm(v: &mut [f64]) {
        let z: f64 = v.iter().sum();
        for x in v.iter_mut() {
            *x /= z;
        }
    }

    /// THE ORACLE: the time-weighted occupancy is the Boltzmann distribution, exactly enumerated.
    #[test]
    fn time_weighted_occupancy_is_boltzmann() {
        for (rates, seed) in [(Rates::Metropolis, 3u64), (Rates::Glauber, 11)] {
            let g = glass(6, seed);
            let beta = 0.65;
            let mut k = Kmc::new(&g, beta, rates, 0x00B0_1742);
            let got = k.occupancy(400_000).expect("6 spins enumerate");
            let want = exact_boltzmann(&g, beta);
            let d = tv(&got, &want);
            assert!(d < 0.01, "{rates:?}: TV to exact Boltzmann {d:.4}");
        }
    }

    /// AND THE TRAP: the unweighted average over visits converges to a DIFFERENT distribution,
    /// which is also known in closed form.
    ///
    /// Both halves are asserted. Matching `pi R` pins down what the jump chain does; being far from
    /// `pi` is what makes dropping the waiting times a bug rather than a rounding difference. A
    /// test that only checked the weighted estimator would also pass on an implementation that
    /// weighted by nothing, whenever the two laws happened to be close on that model.
    #[test]
    fn the_plain_average_over_visits_is_the_jump_chain_and_not_boltzmann() {
        let g = glass(6, 3);
        let beta = 0.65;
        let rates = Rates::Metropolis;
        let mut k = Kmc::new(&g, beta, rates, 0x5EED);

        let mut visits = vec![0.0f64; 1 << g.n];
        let mut dwelt = vec![0.0f64; 1 << g.n];
        k.run(400_000, |s, dt| {
            visits[mask(s)] += 1.0;
            dwelt[mask(s)] += dt;
        });
        norm(&mut visits);
        norm(&mut dwelt);

        let pi = exact_boltzmann(&g, beta);
        let nu = jump_law(&g, beta, rates);

        // The two laws really are different on this model, so the test below can tell them apart.
        let split = tv(&pi, &nu);
        assert!(split > 0.05, "pi and pi*R are only {split:.4} apart; nothing to detect");

        assert!(tv(&visits, &nu) < 0.01, "visits should be pi*R: {:.4}", tv(&visits, &nu));
        assert!(tv(&dwelt, &pi) < 0.01, "dwell-weighted should be pi: {:.4}", tv(&dwelt, &pi));
        assert!(
            tv(&visits, &pi) > 0.05,
            "the unweighted average is {:.4} from Boltzmann, and that is the bug",
            tv(&visits, &pi)
        );
    }

    /// Detailed balance, as an algebraic identity over every state and every site.
    ///
    /// `pi(s) r_i(s) = pi(s^i) r_i(s^i)` is what makes the continuous-time chain reversible with
    /// respect to Boltzmann; it holds exactly, so it is checked at 1e-12 rather than statistically.
    #[test]
    fn both_rate_laws_satisfy_detailed_balance_exactly() {
        let g = glass(8, 7);
        for rates in [Rates::Metropolis, Rates::Glauber] {
            for beta in [0.3, 1.0, 2.5] {
                let mut s = vec![-1i8; g.n];
                for m in 0..(1usize << g.n) {
                    for b in 0..g.n {
                        s[b] = if m >> b & 1 == 1 { 1 } else { -1 };
                    }
                    let e = g.energy(&s);
                    for i in 0..g.n {
                        let lhs = (-beta * e).exp() * rates.rate(s[i], g.field(i, &s), beta);
                        s[i] = -s[i];
                        let e2 = g.energy(&s);
                        let rhs = (-beta * e2).exp() * rates.rate(s[i], g.field(i, &s), beta);
                        s[i] = -s[i];
                        let scale = lhs.abs().max(rhs.abs());
                        assert!(
                            (lhs - rhs).abs() <= 1e-12 * scale,
                            "{rates:?} beta={beta} state {m} site {i}: {lhs:.17e} vs {rhs:.17e}"
                        );
                    }
                }
            }
        }
    }

    /// The waiting time is `Exp(R)`, checked against the closed-form CDF `1 - exp(-R t)`.
    ///
    /// A clock that were merely "random and positive" would pass a mean test after rescaling; the
    /// CDF at several points is what pins the distribution down.
    #[test]
    fn the_waiting_time_is_exponential_in_the_total_rate() {
        // The chain is re-seated in its starting state after every draw, so `R` is held fixed:
        // what is under test is the draw, not the walk.
        let g = lattice2d(4, 1.0);
        let mut k = Kmc::new(&g, 0.7, Rates::Metropolis, 99);
        let s0: Vec<i8> = k.state().to_vec();
        let r = k.total_rate();
        let n = 200_000;
        let mut d = Vec::with_capacity(n);
        for _ in 0..n {
            let j = k.step().expect("a live lattice always has an escape");
            d.push(j.dwell);
            assert!((j.mean_dwell - 1.0 / r).abs() < 1e-12, "mean dwell is 1/R");
            k.set_state(&s0);
            assert!((k.total_rate() - r).abs() < 1e-12, "the rate table restores exactly");
        }
        let mean: f64 = d.iter().sum::<f64>() / n as f64;
        assert!((mean * r - 1.0).abs() < 0.02, "mean dwell x R = {:.4}", mean * r);
        for q in [0.25f64, 0.5, 1.0, 2.0, 3.0] {
            let t = q / r;
            let got = d.iter().filter(|&&x| x <= t).count() as f64 / n as f64;
            let want = 1.0 - (-r * t).exp();
            assert!((got - want).abs() < 0.005, "CDF at {q}/R: {got:.4} vs {want:.4}");
        }
    }

    /// The class table must be the rate table: nothing cached, nothing stale, nothing leaked.
    ///
    /// Audited after **every** step, and over the whole table rather than site by site. Both of
    /// those are load-bearing. Deleting the swap-remove's `slot` fixup leaves one site pointing at
    /// a position it no longer occupies, and the next `attach` of that site repairs the pointer
    /// while leaving the damage done — so a check every few hundred steps sees a consistent table
    /// and only the sampled distribution goes wrong, five hundred lines away. A per-site check also
    /// misses it, because the corruption is a member removed twice and a member never removed: the
    /// invariant that catches it is that each site appears in exactly one class exactly once.
    ///
    /// And the model has to be one whose classes are shared. A Glauber glass gives every site its
    /// own rate, so every class holds one member, `detach` never takes its swap branch, and this
    /// test passes on an implementation with no swap branch at all — which is exactly what it did
    /// before the `crowded` assertion below was added. Metropolis's `min(1, .)` is what puts many
    /// sites in one class, so it is the law under test here.
    #[test]
    fn the_class_table_agrees_with_a_rebuild_bit_for_bit() {
        let g = glass(12, 21);
        let mut k = Kmc::new(&g, 1.1, Rates::Metropolis, 4242);
        let mut crowded = 0usize;
        for step in 0..4_000 {
            assert!(k.step().is_some(), "step {step}");
            let mut filed = vec![0u32; g.n];
            for (c, cl) in k.classes.iter().enumerate() {
                crowded = crowded.max(cl.members.len());
                for (p, &m) in cl.members.iter().enumerate() {
                    filed[m as usize] += 1;
                    assert_eq!(k.slot[m as usize] as usize, p, "step {step}: slot of site {m}");
                    assert_eq!(k.class_of[m as usize] as usize, c, "step {step}: class of {m}");
                    assert_eq!(key(cl.rate), key(k.rate[m as usize]), "step {step}: rate of {m}");
                }
            }
            let once = filed.iter().all(|&c| c == 1);
            assert!(once, "step {step}: each site filed exactly once, got {filed:?}");

            let mut sum = 0.0;
            for i in 0..g.n {
                let want = k.rates.rate(k.s[i], g.field(i, &k.s), k.beta);
                assert_eq!(k.rate[i].to_bits(), want.to_bits(), "step {step} site {i}");
                sum += want;
            }
            assert!((k.total_rate() - sum).abs() < 1e-12 * sum, "class totals are the rate sum");
            // The free list is what stops a generic model growing one class per step forever.
            assert!(k.classes.len() <= g.n, "{} class slots for {} sites", k.classes.len(), g.n);
            assert_eq!(k.classes(), k.index.len());
        }
        // Without this, everything above is checked on classes of one and the swap never runs.
        assert!(crowded >= 3, "no class held more than {crowded}: the swap branch never ran");
    }

    /// A uniform square lattice at `h = 0` has exactly three Metropolis rates, at any size.
    ///
    /// `dE = 2 s_i sum_j s_j` takes the values -8, -4, 0, 4, 8, and `min(1, exp(-beta dE))`
    /// collapses the three non-positive ones onto 1. This is the structure the n-fold way is named
    /// for, and it is why selection on a lattice costs O(1) rather than O(n).
    #[test]
    fn a_uniform_lattice_has_three_metropolis_rate_classes() {
        let beta = 0.44;
        for l in [4usize, 8, 16] {
            let g = lattice2d(l, 1.0);
            let mut k = Kmc::new(&g, beta, Rates::Metropolis, 5);
            k.run(5_000, |_, _| {});
            assert!(k.classes() <= 3, "{l}x{l}: {} classes", k.classes());
            let want: Vec<u64> = [1.0f64, (-4.0 * beta).exp(), (-8.0 * beta).exp()]
                .iter()
                .map(|&r| key(r))
                .collect();
            for b in k.index.keys() {
                assert!(want.contains(b), "unexpected rate {}", f64::from_bits(*b));
            }
            // Glauber has no `min`, so all five gaps stay distinct: the collapse is Metropolis's.
            let mut gl = Kmc::new(&g, beta, Rates::Glauber, 5);
            gl.run(5_000, |_, _| {});
            assert!(gl.classes() <= 5, "Glauber: {} classes", gl.classes());
        }
    }

    /// Rejection-free means one flip per proposal, and at low temperature that is a large factor.
    ///
    /// The comparison is exact rather than analogical: under [`Rates::Glauber`] the attempt this
    /// chain skips IS a Gibbs draw, so `n * time / steps` predicts the draws-per-flip a real Gibbs
    /// run must pay. Both numbers are measured here, and they have to agree.
    #[test]
    fn rejection_free_pays_one_proposal_per_flip_where_gibbs_pays_hundreds() {
        let g = lattice2d(8, 1.0);
        let beta = 1.2;

        let mut k = Kmc::new(&g, beta, Rates::Glauber, 77);
        k.run(20_000, |_, _| {}); // burn in
        let (t0, s0) = (k.time(), k.steps());
        k.run(60_000, |_, _| {});
        let flips = k.steps() - s0;
        assert_eq!(flips, 60_000, "every step is a flip, by construction");
        let predicted = g.n as f64 * (k.time() - t0) / flips as f64;

        let mut smp = crate::gibbs::Sampler::new(&g, beta, 77);
        smp.sweeps(2_000, None); // burn in
        let sweeps = 20_000u64;
        let mut changed = 0u64;
        let mut prev = smp.s.clone();
        for _ in 0..sweeps {
            smp.sweep(None);
            changed += prev.iter().zip(&smp.s).filter(|(a, b)| a != b).count() as u64;
            prev.copy_from_slice(&smp.s);
        }
        let measured = (sweeps * g.n as u64) as f64 / changed as f64;

        assert!(measured > 20.0, "Gibbs spent only {measured:.1} draws per flip; raise beta");
        assert!(
            (predicted - measured).abs() < 0.3 * measured,
            "predicted {predicted:.1} draws per flip, Gibbs measured {measured:.1}"
        );
        assert!(
            k.gibbs_draws_replaced() > 20.0 * k.steps() as f64,
            "{:.0} draws replaced by {} proposals",
            k.gibbs_draws_replaced(),
            k.steps()
        );
    }

    /// A state every one of whose flips is impossibly uphill has no escape, and says so.
    #[test]
    fn a_frozen_chain_reports_no_move_instead_of_dividing_by_zero() {
        let mut gb = GraphBuilder::new(2);
        gb.couple(0, 1, 1.0);
        let g = gb.build();
        let mut k = Kmc::new(&g, 1e4, Rates::Metropolis, 1);
        k.set_state(&[1, 1]); // the ground state: both rates underflow to exactly zero
        assert_eq!(k.total_rate(), 0.0);
        assert_eq!(k.step(), None);
        assert_eq!(k.run(100, |_, _| {}), 0, "a frozen chain takes no steps");
        let p = k.occupancy(10).expect("2 spins enumerate");
        assert_eq!(p[0b11], 1.0, "all the mass is where it froze");
    }

    /// Reproducible from the seed alone, and actually using it.
    #[test]
    fn deterministic_by_seed() {
        let g = glass(10, 2);
        let run = |seed| {
            let mut k = Kmc::new(&g, 0.9, Rates::Metropolis, seed);
            k.run(5_000, |_, _| {});
            (k.time(), k.state().to_vec())
        };
        let a = run(8);
        assert_eq!(a, run(8), "same seed, same clock and same state");
        assert!(run(9) != a, "a different seed is a different run");
    }
}
