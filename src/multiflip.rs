//! Multi-flip informed proposals — path auxiliary sampling and discrete Langevin.
//!
//! [`crate::informed`] moves one spin per proposal, and its own measured table says where that
//! runs out: `tau_int` 24,627 flips at `beta = 4`, because one flip cannot cross a barrier however
//! well it is weighted. Two published answers to that, both here and both keeping the exact
//! Boltzmann distribution:
//!
//! * [`Path`] — path auxiliary sampling (Sun, Dai, Xia & Ramamurthy, *Path Auxiliary Proposal for
//!   MCMC in Discrete Space*, ICLR 2022). Draw a path of `L` locally-balanced single flips and
//!   correct the WHOLE path with one Metropolis–Hastings ratio.
//! * [`Langevin`] — discrete Langevin, DULA and DMALA (Zhang, Liu & Zhang, *A Langevin-like
//!   Sampler for Discrete Distributions*, ICML 2022). Propose every coordinate at once from a
//!   factorised locally-balanced kernel; [`Langevin::adjust`] decides whether the correction is
//!   applied at all.
//!
//! # The path acceptance is a ratio of proposal normalisers, at every length
//!
//! Write `Z(y)` for the sum of proposal weights at `y` and `r_t` for the target ratio the path's
//! `t`-th flip takes. The forward path has probability `prod_t g(r_t)/Zf_t`, the reverse path —
//! the same sites in the opposite order — `prod_t g(1/r_t)/Zb_t`. A balancing function satisfies
//! `g(1/r) = g(r)/r`, and `prod_t r_t` is exactly `pi(y_L)/pi(y_0)`, so the target ratio cancels
//! against the `g`s and
//!
//! ```text
//!   alpha = min(1, [pi(y_L)/pi(y_0)] * prod_t [g(1/r_t) Zf_t] / [g(r_t) Zb_t])
//!         = min(1, prod_t Zf_t / Zb_t)
//! ```
//!
//! With backtracking allowed `Zf_t = Z(y_t)` and `Zb_t = Z(y_{t+1})`, the product telescopes to
//! `Z(y_0)/Z(y_L)` — [`crate::informed`]'s acceptance, unchanged by the path length. Suppressing
//! backtracking removes one term from each normaliser and the telescoping stops, so every factor is
//! carried. `the_path_acceptance_matches_the_metropolis_hastings_definition` checks the
//! implementation against the unsimplified ratio, energies and all, rather than against this
//! algebra.
//!
//! # A FIXED path length is reversible and not ergodic, and the difference is invisible
//!
//! A path of exactly `L` flips changes a number of spins with the same parity as `L`, so a fixed
//! EVEN `L` conserves the parity of the up-spin count and half the state space is unreachable. The
//! kernel is still exactly reversible — `pi P = pi` to 1e-12 on the enumerated transition matrix,
//! every acceptance right — it simply never leaves the block it started in, and total variation
//! against enumeration sat at **0.508** while every algebraic check passed.
//!
//! So [`Path::length`] is a MAXIMUM and each proposal draws `L` uniformly from `1..=length`. Drawn
//! independently of the state, `p(L)` cancels from the ratio, and the length-1 paths make the chain
//! irreducible. `the_path_kernel_reaches_every_state` is the test that catches this; the invariance
//! test beside it did not.
//!
//! # What it is worth, measured
//!
//! Flips against flips, one draw per `n` flips for both arms, `tau_int` over the worse of energy
//! and magnetization. On 32 disjoint 4-cliques — a barrier whose cheapest crossing is four
//! coordinated flips — a `length = 8` [`Path`] costs **381** flips per independent sample at
//! `beta = 1.5` against single-flip [`crate::informed::Informed`]'s **37,663**, and the two are
//! within 1.2x at `beta = 0.5` where there is no barrier.
//!
//! On the 256-spin spin glass of `examples/informed_mixing.rs` it buys **nothing**: 4,128 flips
//! against 3,527 at `beta = 2`. A glass's slow modes are extended, and a path of eight flips is not
//! their shape. Colder than that neither number is measurable — `tau_int` for adjacent path lengths
//! swung between 4.6e2 and 5.5e5 on one graph — and a frozen chain scores `tau_int` at its 0.5-draw
//! FLOOR, so the instrument reads a stuck chain as a perfect one.

use crate::graph::Graph;
use crate::informed::Balance;
use crate::rng::Pcg;

/// `ln(1 + e^x)`, without overflowing for large `x` or losing the small-`x` end.
fn softplus(x: f64) -> f64 {
    if x > 0.0 { x + (-x).exp().ln_1p() } else { x.exp().ln_1p() }
}

/// `ln g(r)` from `ln r`.
///
/// Restated rather than shared: `informed`'s copy is private, and this module is not permitted to
/// widen it. `the_balancing_functions_are_balanced` asserts the identity this copy has to satisfy,
/// so a drift between the two would be caught here rather than inferred.
fn log_g(balance: Balance, log_r: f64) -> f64 {
    match balance {
        Balance::Sqrt => 0.5 * log_r,
        Balance::Metropolis => log_r.min(0.0),
        Balance::Barker => -softplus(-log_r),
    }
}

/// Single flips between exact recomputations of the weight total, for [`Path`].
///
/// Counted in FLIPS rather than steps: a length-`L` path accumulates `L` times the rounding of a
/// single-flip step, so a bound stated in steps would loosen with the path length.
pub const REFRESH_FLIPS: u64 = 4096;

/// Path auxiliary sampling: a path of locally-balanced single flips, one acceptance for the lot.
///
/// Reversible with respect to the Boltzmann distribution at `beta` — checked against an exact
/// transition matrix in `the_path_kernel_leaves_the_boltzmann_distribution_invariant`.
pub struct Path<'g> {
    /// The model.
    pub g: &'g Graph,
    /// Inverse temperature.
    pub beta: f64,
    /// Current state.
    pub s: Vec<i8>,
    /// The sampler's stream.
    pub rng: Pcg,
    /// Which balancing function weights each step of the path.
    pub balance: Balance,
    /// LONGEST path a proposal may take; each one draws its length uniformly from `1..=length`.
    /// A fixed even length is reversible and not ergodic — see the module docs. `1` is the
    /// single-flip chain of [`crate::informed`].
    pub length: usize,
    /// Whether a step may immediately undo the one before it.
    pub backtrack: bool,
    /// `f_k` for every site, maintained incrementally.
    fields: Vec<f64>,
    /// `g(r_k)` for every site, at the current shift.
    w: Vec<f64>,
    /// `Σ_k w_k`.
    total: f64,
    /// A constant subtracted from every `ln g`, re-centred at each refresh on the largest observed
    /// log-weight. Cancels in every ratio the sampler forms.
    shift: f64,
    /// Sites of the most recent proposal, in order.
    last: Vec<usize>,
    steps: u64,
    accepted: u64,
    since: u64,
}

impl<'g> Path<'g> {
    /// A sampler at `beta`, started from a random state drawn from `seed`.
    #[must_use]
    pub fn new(g: &'g Graph, beta: f64, seed: u64) -> Path<'g> {
        let mut rng = Pcg::new(seed, 0x2C1B);
        let s: Vec<i8> = (0..g.n).map(|_| if rng.f64() < 0.5 { -1i8 } else { 1 }).collect();
        Self::from_state(g, beta, s, rng)
    }

    /// A sampler started from a given state.
    ///
    /// # Panics
    ///
    /// If `s` does not have one spin per node.
    #[must_use]
    pub fn from_state(g: &'g Graph, beta: f64, s: Vec<i8>, rng: Pcg) -> Path<'g> {
        assert_eq!(s.len(), g.n, "the state must have one spin per node");
        let mut it = Path {
            g,
            beta,
            s,
            rng,
            // `informed`'s measured table has Barker ahead of Zanella's recommended `Sqrt` at both
            // cold rungs, by 2.4x at beta = 2. The default follows the measurement.
            balance: Balance::Barker,
            length: 4,
            backtrack: false,
            fields: Vec::new(),
            w: Vec::new(),
            total: 0.0,
            shift: 0.0,
            last: Vec::new(),
            steps: 0,
            accepted: 0,
            since: 0,
        };
        it.refresh();
        it
    }

    /// Choose the balancing function. `Barker` is the default.
    #[must_use]
    pub fn with_balance(mut self, balance: Balance) -> Self {
        self.balance = balance;
        self.refresh();
        self
    }

    /// Set the longest path a proposal may take. Each proposal draws its length from `1..=length`.
    ///
    /// # Panics
    ///
    /// If `length` is zero: a proposal that moves nothing has no acceptance ratio to form.
    #[must_use]
    pub fn with_length(mut self, length: usize) -> Self {
        assert!(length > 0, "a path must take at least one flip");
        self.length = length;
        self
    }

    /// Allow or forbid a step immediately undoing the one before it.
    ///
    /// Forbidden by default. Allowing it makes the acceptance telescope to `Z(x)/Z(y)` and makes a
    /// long path likely to spend its flips oscillating on the one hottest site.
    #[must_use]
    pub fn with_backtrack(mut self, backtrack: bool) -> Self {
        self.backtrack = backtrack;
        self
    }

    /// Replace the state and recompute every derived quantity.
    ///
    /// # Panics
    ///
    /// If `s` does not have one spin per node.
    pub fn set_state(&mut self, s: Vec<i8>) {
        assert_eq!(s.len(), self.g.n, "the state must have one spin per node");
        self.s = s;
        self.refresh();
    }

    /// Recompute every field, weight and the total exactly, and re-centre the shift on the largest
    /// log-weight present.
    fn refresh(&mut self) {
        self.fields = (0..self.g.n).map(|i| self.g.field(i, &self.s)).collect();
        let logs: Vec<f64> = (0..self.g.n).map(|k| self.log_weight(k)).collect();
        self.shift = logs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if !self.shift.is_finite() {
            self.shift = 0.0;
        }
        self.w = logs.iter().map(|l| (l - self.shift).exp()).collect();
        self.total = crate::round::sum_up(&self.w).max(0.0);
        self.since = 0;
    }

    /// `ln g(r_k)` for flipping site `k`, before the shift.
    fn log_weight(&self, k: usize) -> f64 {
        let log_r = -2.0 * self.beta * f64::from(self.s[k]) * self.fields[k];
        log_g(self.balance, log_r)
    }

    /// `g(r_k)` at the current shift.
    fn weight(&self, k: usize) -> f64 {
        (self.log_weight(k) - self.shift).exp()
    }

    /// Flip site `k` and repair the fields, weights and total it touched.
    fn flip(&mut self, k: usize) {
        self.s[k] = -self.s[k];
        let sk = f64::from(self.s[k]);
        for e in self.g.offset[k]..self.g.offset[k + 1] {
            let j = self.g.nbr[e] as usize;
            self.fields[j] += self.g.w[e] * 2.0 * sk;
            let fresh = self.weight(j);
            self.total += fresh - self.w[j];
            self.w[j] = fresh;
        }
        let fresh = self.weight(k);
        self.total += fresh - self.w[k];
        self.w[k] = fresh;
    }

    /// `Σ_j w_j` with `excl` left out.
    ///
    /// The subtraction is `O(1)` and loses every significant digit when the excluded weight is
    /// nearly the whole total — which is exactly the cold, one-dominant-site regime this sampler
    /// exists for. So a collapsed difference is recomputed directly; the weights are non-negative,
    /// so that sum has no cancellation in it at all.
    fn sum_excluding(&self, excl: Option<usize>) -> f64 {
        let Some(k) = excl else { return self.total };
        let d = self.total - self.w[k];
        if d > 1e-9 * self.total {
            return d;
        }
        self.w.iter().enumerate().filter(|(i, _)| *i != k).map(|(_, v)| v).sum()
    }

    /// Draw a site with probability `w_k / norm`, skipping `excl`.
    fn draw(&mut self, norm: f64, excl: Option<usize>) -> Option<usize> {
        let target = self.rng.f64() * norm;
        let mut acc = 0.0;
        let mut fallback = None;
        for (i, &wi) in self.w.iter().enumerate() {
            if Some(i) == excl {
                continue;
            }
            acc += wi;
            if acc >= target {
                return Some(i);
            }
            if wi > 0.0 {
                fallback = Some(i);
            }
        }
        // Rounding can leave the running sum a hair under a target drawn against `norm`. The last
        // site with any weight is the one the scan was about to reach.
        fallback
    }

    /// Put the state back where the most recent walk started.
    fn rewind(&mut self) {
        for i in (0..self.last.len()).rev() {
            let k = self.last[i];
            self.flip(k);
        }
        // The total came back along a different arithmetic path than it left by, so it carries the
        // rounding of both. Recomputing it is the cheapest place to be exact rather than nearly so.
        self.total = crate::round::sum_up(&self.w).max(0.0);
    }

    /// Walk a path, leaving the state at its end.
    ///
    /// `forced` supplies the sites; `None` draws `len` of them. Returns `(ln q, ln alpha)` — the log
    /// probability the proposal assigns to this path GIVEN its length, and the log
    /// Metropolis–Hastings ratio for it. The length factor is left out of both because the reverse
    /// path has the same length and it cancels. A path that cannot be proposed rewinds and returns
    /// `None`.
    fn walk(&mut self, forced: Option<&[usize]>, len: usize) -> Option<(f64, f64)> {
        self.last.clear();
        if len == 0 {
            return None;
        }
        let (mut log_q, mut log_a) = (0.0f64, 0.0f64);
        let mut prev: Option<usize> = None;
        for t in 0..len {
            let back = if self.backtrack { None } else { prev };
            // Zf_t: the forward normaliser at y_t, minus the step that would undo y_{t-1}.
            let zf = self.sum_excluding(back);
            if !(zf > 0.0) || !zf.is_finite() {
                self.rewind();
                return None;
            }
            let k = match forced {
                Some(sites) => {
                    let k = sites[t];
                    if k >= self.g.n || (back == Some(k)) {
                        self.rewind();
                        return None;
                    }
                    k
                }
                None => match self.draw(zf, back) {
                    Some(k) => k,
                    None => {
                        self.rewind();
                        return None;
                    }
                },
            };
            let wk = self.w[k];
            if !(wk > 0.0) || !wk.is_finite() {
                self.rewind();
                return None;
            }
            // Zb_{t-1}: the reverse normaliser at y_t, minus the step that would undo y_{t+1} —
            // that is, minus the weight of the site the forward path is about to take.
            if t >= 1 {
                let zb = if self.backtrack { self.total } else { self.sum_excluding(Some(k)) };
                if !(zb > 0.0) || !zb.is_finite() {
                    self.rewind();
                    return None;
                }
                log_a -= zb.ln();
            }
            log_q += wk.ln() - zf.ln();
            log_a += zf.ln();
            self.flip(k);
            self.last.push(k);
            prev = Some(k);
        }
        // Zb_{L-1} is the whole normaliser at the path's end: the reverse path starts there and has
        // nothing behind it to exclude.
        if !(self.total > 0.0) || !self.total.is_finite() {
            self.rewind();
            return None;
        }
        log_a -= self.total.ln();
        Some((log_q, log_a))
    }

    /// The log proposal probability and log acceptance for flipping `sites` in order from the
    /// current state, leaving the state where it found it.
    ///
    /// `None` if the path is not proposable: a site out of range, a repeat where backtracking is
    /// forbidden, or a weight total that has collapsed. The `1/length` factor for drawing this
    /// path's length is not included — it cancels. Overwrites [`Path::last_path`].
    pub fn path_ratio(&mut self, sites: &[usize]) -> Option<(f64, f64)> {
        let r = self.walk(Some(sites), sites.len());
        if r.is_some() {
            self.rewind();
        }
        r
    }

    /// One path proposal, accepted or rejected. Returns whether the state moved.
    pub fn step(&mut self) -> bool {
        self.steps += 1;
        // Uniform on 1..=length, drawn independently of the state so it cancels from the ratio.
        // A FIXED length is the ergodicity trap the module docs describe.
        let len = 1 + (self.rng.next_u32() as usize) % self.length;
        self.since += len as u64;
        if self.since >= REFRESH_FLIPS {
            self.refresh();
        }
        if !(self.total > 0.0) || !self.total.is_finite() {
            self.refresh();
            if !(self.total > 0.0) || !self.total.is_finite() {
                return false;
            }
        }
        let Some((_, log_a)) = self.walk(None, len) else { return false };
        // ln u < ln alpha, so a path whose reverse is impossible (`log_a` = −inf) is never taken
        // and one with alpha >= 1 always is.
        let accept = self.rng.f64().ln() < log_a;
        if accept {
            self.accepted += 1;
            true
        } else {
            self.rewind();
            false
        }
    }

    /// Run `n` path proposals — `n * (length + 1) / 2` single flips of work on average.
    pub fn steps(&mut self, n: usize) {
        for _ in 0..n {
            self.step();
        }
    }

    /// Energy of the current state.
    #[must_use]
    pub fn energy(&self) -> f64 {
        self.g.energy(&self.s)
    }

    /// Sites of the most recent proposal, in order, accepted or not.
    #[must_use]
    pub fn last_path(&self) -> &[usize] {
        &self.last
    }

    /// Fraction of proposals accepted.
    ///
    /// Read it the way [`crate::informed::Informed::acceptance`] is read: the ratio is built from
    /// proposal normalisers, so a high number means the proposal landscape barely moved.
    #[must_use]
    pub fn acceptance(&self) -> f64 {
        if self.steps == 0 { 0.0 } else { self.accepted as f64 / self.steps as f64 }
    }

    /// Proposals taken.
    #[must_use]
    pub fn taken(&self) -> u64 {
        self.steps
    }
}

/// Discrete Langevin: every coordinate proposed at once, DULA or DMALA.
///
/// The proposal flips site `i` with probability `sigma(ln g(r_i) − 1/(2 alpha))`, independently
/// across sites. For a spin model `ln r_i = −2 beta s_i f_i` is exact — the gradient of the
/// multilinear extension at a vertex IS the local field — so nothing here is a Taylor expansion,
/// which is what the continuous-relaxation derivation in the paper is approximating.
pub struct Langevin<'g> {
    /// The model.
    pub g: &'g Graph,
    /// Inverse temperature.
    pub beta: f64,
    /// Current state.
    pub s: Vec<i8>,
    /// The sampler's stream.
    pub rng: Pcg,
    /// Which balancing function weights each coordinate. `Sqrt` is the paper's DMALA.
    pub balance: Balance,
    /// Step size, in the paper's `{0,1}` parametrisation: the per-site flip penalty is `1/(2 alpha)`
    /// nats. Larger flips more coordinates per step.
    pub alpha: f64,
    /// Apply the Metropolis correction. `false` is DULA, which is BIASED by construction.
    pub adjust: bool,
    steps: u64,
    accepted: u64,
    flips: u64,
}

impl<'g> Langevin<'g> {
    /// A sampler at `beta`, started from a random state drawn from `seed`.
    #[must_use]
    pub fn new(g: &'g Graph, beta: f64, seed: u64) -> Langevin<'g> {
        let mut rng = Pcg::new(seed, 0x5A17);
        let s: Vec<i8> = (0..g.n).map(|_| if rng.f64() < 0.5 { -1i8 } else { 1 }).collect();
        Self::from_state(g, beta, s, rng)
    }

    /// A sampler started from a given state.
    ///
    /// # Panics
    ///
    /// If `s` does not have one spin per node.
    #[must_use]
    pub fn from_state(g: &'g Graph, beta: f64, s: Vec<i8>, rng: Pcg) -> Langevin<'g> {
        assert_eq!(s.len(), g.n, "the state must have one spin per node");
        Langevin {
            g,
            beta,
            s,
            rng,
            balance: Balance::Sqrt,
            alpha: 0.2,
            adjust: true,
            steps: 0,
            accepted: 0,
            flips: 0,
        }
    }

    /// Choose the balancing function. `Sqrt` — the paper's DMALA — is the default.
    #[must_use]
    pub fn with_balance(mut self, balance: Balance) -> Self {
        self.balance = balance;
        self
    }

    /// Set the step size.
    ///
    /// # Panics
    ///
    /// If `alpha` is not positive and finite: the penalty is `1/(2 alpha)`.
    #[must_use]
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        assert!(alpha > 0.0 && alpha.is_finite(), "the step size must be positive and finite");
        self.alpha = alpha;
        self
    }

    /// Turn the Metropolis correction off (DULA) or on (DMALA).
    #[must_use]
    pub fn with_adjust(mut self, adjust: bool) -> Self {
        self.adjust = adjust;
        self
    }

    /// The per-site flip logit at `s`: `ln g(r_i) − 1/(2 alpha)`.
    ///
    /// # Panics
    ///
    /// If `s` does not have one spin per node.
    #[must_use]
    pub fn flip_logits(&self, s: &[i8]) -> Vec<f64> {
        assert_eq!(s.len(), self.g.n, "the state must have one spin per node");
        let penalty = 1.0 / (2.0 * self.alpha);
        (0..self.g.n)
            .map(|i| {
                let log_r = -2.0 * self.beta * f64::from(s[i]) * self.g.field(i, s);
                log_g(self.balance, log_r) - penalty
            })
            .collect()
    }

    /// `ln q(to | from)` for the factorised proposal.
    ///
    /// # Panics
    ///
    /// If the two states do not have one spin per node each.
    #[must_use]
    pub fn log_proposal(&self, from: &[i8], to: &[i8]) -> f64 {
        assert_eq!(from.len(), to.len(), "both states must be over the same model");
        let z = self.flip_logits(from);
        Self::log_q(&z, from, to)
    }

    /// `ln q(to | from)` from precomputed logits. `ln sigma(z) = −softplus(−z)`, so neither tail is
    /// ever `ln(0)`.
    fn log_q(z: &[f64], from: &[i8], to: &[i8]) -> f64 {
        let mut l = 0.0;
        for i in 0..z.len() {
            l -= if to[i] == from[i] { softplus(z[i]) } else { softplus(-z[i]) };
        }
        l
    }

    /// Log Metropolis–Hastings acceptance for moving from the current state to `to`.
    ///
    /// # Panics
    ///
    /// If `to` does not have one spin per node.
    #[must_use]
    pub fn log_alpha(&self, to: &[i8]) -> f64 {
        let z = self.flip_logits(&self.s);
        self.log_alpha_with(&z, to)
    }

    /// Log acceptance, reusing the forward logits the proposal was drawn from.
    fn log_alpha_with(&self, z_from: &[f64], to: &[i8]) -> f64 {
        let z_to = self.flip_logits(to);
        -self.beta * (self.g.energy(to) - self.g.energy(&self.s))
            + Self::log_q(&z_to, to, &self.s)
            - Self::log_q(z_from, &self.s, to)
    }

    /// One step: propose every coordinate, then correct unless [`Langevin::adjust`] is off.
    ///
    /// Returns whether the state moved.
    pub fn step(&mut self) -> bool {
        self.steps += 1;
        let z = self.flip_logits(&self.s);
        let mut to = self.s.clone();
        for i in 0..self.g.n {
            // sigma(z) written so a large negative z gives exactly 0 rather than an overflow.
            let p = 1.0 / (1.0 + (-z[i]).exp());
            if self.rng.f64() < p {
                to[i] = -to[i];
                self.flips += 1;
            }
        }
        if !self.adjust {
            self.s = to;
            self.accepted += 1;
            return true;
        }
        let log_a = self.log_alpha_with(&z, &to);
        if self.rng.f64().ln() < log_a {
            self.s = to;
            self.accepted += 1;
            true
        } else {
            false
        }
    }

    /// Run `n` steps — `n * g.n` single-site proposals of work.
    pub fn steps(&mut self, n: usize) {
        for _ in 0..n {
            self.step();
        }
    }

    /// Energy of the current state.
    #[must_use]
    pub fn energy(&self) -> f64 {
        self.g.energy(&self.s)
    }

    /// Fraction of steps accepted. Always `1` with the correction off.
    #[must_use]
    pub fn acceptance(&self) -> f64 {
        if self.steps == 0 { 0.0 } else { self.accepted as f64 / self.steps as f64 }
    }

    /// Mean number of sites the proposal flips per step, accepted or not.
    #[must_use]
    pub fn proposed_flips(&self) -> f64 {
        if self.steps == 0 { 0.0 } else { self.flips as f64 / self.steps as f64 }
    }

    /// Steps taken.
    #[must_use]
    pub fn taken(&self) -> u64 {
        self.steps
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::GraphBuilder;

    /// A frustrated ring with fields: small enough to enumerate, wrong for any sampler that takes a
    /// shortcut.
    fn frustrated(n: usize) -> Graph {
        let mut b = GraphBuilder::new(n);
        for i in 0..n {
            b.couple(i, (i + 1) % n, if i % 3 == 0 { -0.9 } else { 0.7 });
        }
        b.bias(0, 0.4);
        b.bias(n / 2, -0.55);
        b.build()
    }

    /// The mixing fixture of `examples/informed_mixing.rs`, transcribed so the two modules are
    /// measured on the same landscape: a random +/-1 ring with `n/4` chords and small fields.
    fn chorded(n: usize, seed: u64) -> Graph {
        let mut rng = Pcg::new(seed, 0xF5);
        let mut b = GraphBuilder::new(n);
        for i in 0..n {
            b.couple(i, (i + 1) % n, if rng.f64() < 0.5 { -1.0 } else { 1.0 });
        }
        for _ in 0..n / 4 {
            let (i, j) = ((rng.f64() * n as f64) as usize % n, (rng.f64() * n as f64) as usize % n);
            if i != j {
                b.couple(i, j, if rng.f64() < 0.5 { -1.0 } else { 1.0 });
            }
        }
        for i in 0..n {
            b.bias(i, (rng.f64() - 0.5) * 0.4);
        }
        b.build()
    }

    /// Exact Boltzmann weight of a state, unnormalised.
    fn pi(g: &Graph, beta: f64, s: &[i8]) -> f64 {
        (-beta * g.energy(s)).exp()
    }

    /// The proposal weight of every single flip at `s`, computed from scratch with no shift.
    fn weights(g: &Graph, beta: f64, balance: Balance, s: &[i8]) -> Vec<f64> {
        (0..g.n)
            .map(|k| {
                let log_r = -2.0 * beta * f64::from(s[k]) * g.field(k, s);
                log_g(balance, log_r).exp()
            })
            .collect()
    }

    /// Index of a state in the enumeration `exact_boltzmann` uses.
    fn index(s: &[i8]) -> usize {
        s.iter().enumerate().filter(|&(_, &v)| v > 0).map(|(b, _)| 1usize << b).sum()
    }

    /// The balancing condition `g(x) = x g(1/x)`, on THIS module's copy of `g`.
    ///
    /// The whole acceptance derivation rests on it. `informed` keeps its copy private, so a
    /// transcription slip here would be invisible to that module's identical test and would show up
    /// only as a small, plausible bias in the sampled distribution.
    #[test]
    fn the_balancing_functions_are_balanced() {
        for balance in [Balance::Sqrt, Balance::Barker, Balance::Metropolis] {
            let mut x = -8.0f64;
            while x <= 8.0 {
                let lhs = log_g(balance, x);
                let rhs = x + log_g(balance, -x);
                assert!(
                    (lhs - rhs).abs() < 1e-12,
                    "{balance:?} is not balanced at ln r = {x}: {lhs} against {rhs}"
                );
                x += 0.25;
            }
        }
    }

    /// THE ORACLE FOR THE ACCEPTANCE: the unsimplified Metropolis–Hastings ratio.
    ///
    /// The implementation computes `prod_t Zf_t / Zb_t`, which is the algebra in the module docs.
    /// This recomputes the thing that algebra is a simplification OF —
    /// `[pi(y)/pi(x)] * prod_t q(reverse step) / prod_t q(forward step)` — from energies and from
    /// weight sums built from scratch, and compares. A wrong exclusion, a dropped term or a
    /// telescoping that does not hold for the non-backtracking kernel all fail here, and none of
    /// them would fail a comparison against another sampler.
    #[test]
    fn the_path_acceptance_matches_the_metropolis_hastings_definition() {
        let g = frustrated(9);
        let beta = 0.6;
        let mut worst = 0.0f64;
        for balance in [Balance::Sqrt, Balance::Barker, Balance::Metropolis] {
            for backtrack in [false, true] {
                let mut p = Path::new(&g, beta, 4).with_balance(balance).with_backtrack(backtrack);
                let mut r = Pcg::new(17, 5);
                for _ in 0..200 {
                    let start: Vec<i8> =
                        (0..g.n).map(|_| if r.f64() < 0.5 { -1i8 } else { 1 }).collect();
                    p.set_state(start.clone());
                    let len = 1 + (r.next_u32() as usize) % 5;
                    let mut sites = Vec::new();
                    for _ in 0..len {
                        sites.push((r.next_u32() as usize) % g.n);
                    }
                    let Some((log_q, log_a)) = p.path_ratio(&sites) else { continue };
                    assert_eq!(p.s, start, "path_ratio must leave the state where it found it");

                    // The definition, rebuilt: states along the path, then every factor of the MH
                    // ratio written out.
                    let mut states = vec![start.clone()];
                    let mut cur = start.clone();
                    for &k in &sites {
                        cur[k] = -cur[k];
                        states.push(cur.clone());
                    }
                    let ws: Vec<Vec<f64>> =
                        states.iter().map(|y| weights(&g, beta, balance, y)).collect();
                    let (mut want_q, mut want_a) = (0.0f64, 0.0f64);
                    want_a += (pi(&g, beta, &states[len]) / pi(&g, beta, &start)).ln();
                    for t in 0..len {
                        let zf: f64 = ws[t].iter().sum::<f64>()
                            - if t >= 1 && !backtrack { ws[t][sites[t - 1]] } else { 0.0 };
                        let zb: f64 = ws[t + 1].iter().sum::<f64>()
                            - if t + 1 < len && !backtrack { ws[t + 1][sites[t + 1]] } else { 0.0 };
                        // forward: g(r_t)/Zf_t. reverse: g(1/r_t)/Zb_t, and g(1/r_t) is the weight
                        // of flipping the same site back, read off at the state after the flip.
                        want_q += ws[t][sites[t]].ln() - zf.ln();
                        want_a += ws[t + 1][sites[t]].ln() - zb.ln();
                        want_a -= ws[t][sites[t]].ln() - zf.ln();
                    }
                    worst = worst.max((log_a - want_a).abs()).max((log_q - want_q).abs());
                }
            }
        }
        assert!(worst < 1e-9, "the path ratio disagrees with its definition by {worst:e}");
    }

    /// The exact transition matrix of the path kernel: every length the sampler may draw, every
    /// path of that length, probability and acceptance taken from the sampler's own routines, and
    /// the rejected mass returned to the diagonal.
    fn path_kernel(g: &Graph, beta: f64, max_len: usize, backtrack: bool) -> Vec<Vec<f64>> {
        let m = 1usize << g.n;
        let mut p = Path::new(g, beta, 1).with_backtrack(backtrack);
        let mut mat = vec![vec![0.0f64; m]; m];
        for x in 0..m {
            let start: Vec<i8> = (0..g.n).map(|b| if x >> b & 1 == 1 { 1i8 } else { -1 }).collect();
            let mut stay = 0.0;
            for len in 1..=max_len {
                let share = 1.0 / max_len as f64; // the uniform length draw
                let mut sites = vec![0usize; len];
                for code in 0..g.n.pow(len as u32) {
                    let mut c = code;
                    for v in &mut sites {
                        *v = c % g.n;
                        c /= g.n;
                    }
                    p.set_state(start.clone());
                    let Some((log_q, log_a)) = p.path_ratio(&sites) else { continue };
                    let q = share * log_q.exp();
                    let a = log_a.exp().min(1.0);
                    let mut y = start.clone();
                    for &k in &sites {
                        y[k] = -y[k];
                    }
                    mat[x][index(&y)] += q * a;
                    stay += q * (1.0 - a);
                }
            }
            mat[x][x] += stay;
        }
        mat
    }

    /// EXACT INVARIANCE: build the whole transition matrix and check `pi P = pi` to 1e-12.
    ///
    /// This is the claim a sampling test can only ever support — it either holds to floating point
    /// or the kernel is not reversible — and it is checked at both backtracking settings because
    /// the two use different normalisers and only one of them telescopes. The row sums say the
    /// enumerated paths are the ones the sampler draws from and nothing else.
    ///
    /// **It is not enough on its own**, which is the point of the test after it.
    #[test]
    fn the_path_kernel_leaves_the_boltzmann_distribution_invariant() {
        let g = frustrated(6);
        let beta = 0.7;
        let exact = crate::ising::exact_boltzmann(&g, beta);
        let m = 1usize << g.n;
        for backtrack in [false, true] {
            for max_len in [1usize, 2, 3] {
                let mat = path_kernel(&g, beta, max_len, backtrack);
                for (x, row) in mat.iter().enumerate() {
                    let sum: f64 = row.iter().sum();
                    assert!((sum - 1.0).abs() < 1e-12, "row {x} sums to {sum}");
                }
                let mut worst = 0.0f64;
                for y in 0..m {
                    let got: f64 = (0..m).map(|x| exact[x] * mat[x][y]).sum();
                    worst = worst.max((got - exact[y]).abs());
                }
                assert!(
                    worst < 1e-12,
                    "max length {max_len}, backtrack {backtrack}: the kernel moves the Boltzmann \
                     distribution by {worst:e}"
                );
            }
        }
    }

    /// ERGODICITY, WHICH INVARIANCE DOES NOT IMPLY AND THIS MODULE LEARNED THE HARD WAY.
    ///
    /// A path of exactly `L` flips changes a number of spins with the same parity as `L`, so a
    /// FIXED even `L` conserves the parity of the up-spin count. That kernel is exactly reversible
    /// — `pi P = pi` to 1e-12, every acceptance right — and reaches half the state space. It
    /// shipped for the length of one test run and showed up as total variation 0.508 against
    /// enumeration while every algebraic check passed.
    ///
    /// So the reachability of the transition matrix is asserted directly, at an even maximum length
    /// where the defect lived.
    #[test]
    fn the_path_kernel_reaches_every_state() {
        let g = frustrated(6);
        let m = 1usize << g.n;
        for max_len in [2usize, 4] {
            let mat = path_kernel(&g, 0.7, max_len, false);
            let mut reach: Vec<Vec<bool>> =
                mat.iter().map(|r| r.iter().map(|&v| v > 1e-15).collect()).collect();
            // Transitive closure: m squarings cover any path length up to 2^m.
            for _ in 0..8 {
                let mut next = reach.clone();
                for x in 0..m {
                    for z in 0..m {
                        if reach[x][z] {
                            for y in 0..m {
                                next[x][y] |= reach[z][y];
                            }
                        }
                    }
                }
                reach = next;
            }
            let missed = reach.iter().flatten().filter(|&&v| !v).count();
            assert_eq!(
                missed, 0,
                "at max length {max_len} the kernel cannot reach {missed} of the {} state pairs, \
                 so it is reversible with respect to a distribution it never sees all of",
                m * m
            );
        }
    }

    /// End to end, against enumeration through the crate's own certificate.
    ///
    /// The invariance test above holds the proposal and acceptance fixed and checks the algebra;
    /// this one runs the sampler, site draws and refreshes included, at path lengths that exercise
    /// the incremental bookkeeping over many flips.
    #[test]
    fn the_path_chain_samples_the_boltzmann_distribution() {
        let g = frustrated(12);
        let beta = 0.8;
        for len in [1usize, 2, 5, 12] {
            let mut p = Path::new(&g, beta, 7).with_length(len);
            // One draw per n flips of opportunity, so the trace is comparable to a Gibbs one. The
            // mean drawn length is (len + 1) / 2.
            let per = (2 * g.n / (len + 1)).max(1);
            p.steps(20 * per);
            let (mut samples, mut trace) = (Vec::new(), Vec::new());
            for _ in 0..6_000 {
                p.steps(per);
                samples.push(p.s.clone());
                trace.push(p.energy());
            }
            let cert = crate::certify::certify(&g, beta, &samples, &trace);
            crate::certify::assert_boltzmann(&cert, beta, &format!("path/{len}"));
        }
    }

    /// The proposal must actually move more than one spin, or every test above is passing for a
    /// relabelled single-flip chain.
    #[test]
    fn an_accepted_path_moves_several_spins() {
        let g = frustrated(16);
        let mut p = Path::new(&g, 0.8, 3).with_length(6);
        let mut moved = std::collections::BTreeSet::new();
        for _ in 0..4_000 {
            let before = p.s.clone();
            if p.step() {
                let d = before.iter().zip(&p.s).filter(|(a, b)| a != b).count();
                moved.insert(d);
            }
        }
        let most = *moved.iter().next_back().expect("no proposal was ever accepted");
        assert!(most > 1, "every accepted move changed at most one spin: {moved:?}");
        // A length-6 path on 16 sites should reach six distinct spins sometimes, not just twice.
        assert!(most >= 4, "the path never went further than {most} spins from where it started");
    }

    /// The incremental fields, weights and total must agree with a recomputation, or the proposal
    /// is drawn from a normaliser that has drifted away from the weights it normalises.
    #[test]
    fn the_incremental_bookkeeping_agrees_with_a_recomputation() {
        let g = frustrated(24);
        let mut p = Path::new(&g, 0.7, 3).with_length(8);
        let mut worst = 0.0f64;
        for _ in 0..(REFRESH_FLIPS as usize / 8 - 1) {
            p.step();
            for i in 0..g.n {
                worst = worst.max((p.fields[i] - g.field(i, &p.s)).abs());
            }
            let fresh: f64 = crate::round::sum_up(&p.w);
            worst = worst.max((p.total - fresh).abs() / fresh.max(1.0));
        }
        assert!(worst < 1e-9, "incremental state drifted from a recomputation by {worst:e}");
        assert!(p.since < REFRESH_FLIPS, "the loop must stop short of a refresh to test the drift");
    }

    /// EXACT INVARIANCE for DMALA: the proposal factorises, so every one of the `2^n` moves out of
    /// every one of the `2^n` states can be enumerated and `pi P = pi` checked outright.
    ///
    /// The proposal and the acceptance come from the sampler's own public routines, which is what
    /// makes this a test of the correction rather than of a restatement of it.
    #[test]
    fn the_langevin_kernel_leaves_the_boltzmann_distribution_invariant() {
        let g = frustrated(7);
        let beta = 0.7;
        let exact = crate::ising::exact_boltzmann(&g, beta);
        let m = 1usize << g.n;
        let states: Vec<Vec<i8>> = (0..m)
            .map(|x| (0..g.n).map(|b| if x >> b & 1 == 1 { 1i8 } else { -1 }).collect())
            .collect();
        for (balance, alpha) in
            [(Balance::Sqrt, 0.2), (Balance::Sqrt, 2.0), (Balance::Barker, 0.5)]
        {
            let mut l = Langevin::new(&g, beta, 1).with_balance(balance).with_alpha(alpha);
            let mut mat = vec![vec![0.0f64; m]; m];
            for x in 0..m {
                l.s = states[x].clone();
                let z = l.flip_logits(&l.s);
                let mut stay = 0.0;
                for y in 0..m {
                    let q = Langevin::log_q(&z, &l.s, &states[y]).exp();
                    if x == y {
                        stay += q;
                        continue;
                    }
                    let a = l.log_alpha(&states[y]).exp().min(1.0);
                    mat[x][y] += q * a;
                    stay += q * (1.0 - a);
                }
                mat[x][x] += stay;
                let row: f64 = mat[x].iter().sum();
                assert!((row - 1.0).abs() < 1e-12, "row {x} sums to {row}");
            }
            let mut worst = 0.0f64;
            for y in 0..m {
                let got: f64 = (0..m).map(|x| exact[x] * mat[x][y]).sum();
                worst = worst.max((got - exact[y]).abs());
            }
            assert!(
                worst < 1e-12,
                "{balance:?} at alpha {alpha}: the kernel moves the Boltzmann distribution by \
                 {worst:e}"
            );
        }
    }

    /// End to end for DMALA, against enumeration.
    #[test]
    fn the_langevin_chain_samples_the_boltzmann_distribution() {
        // Ten spins, not twelve: certify's noise floor is 0.5 sqrt(2^n / ess), and at n = 12 with a
        // chain this correlated the floor exceeds 1, which is `TooFewSamples` rather than a test.
        let g = frustrated(10);
        let beta = 0.8;
        for alpha in [0.2, 1.0] {
            let mut l = Langevin::new(&g, beta, 11).with_alpha(alpha);
            l.steps(200);
            let (mut samples, mut trace) = (Vec::new(), Vec::new());
            for _ in 0..20_000 {
                l.steps(4);
                samples.push(l.s.clone());
                trace.push(l.energy());
            }
            let cert = crate::certify::certify(&g, beta, &samples, &trace);
            crate::certify::assert_boltzmann(&cert, beta, &format!("dmala/{alpha}"));
        }
    }

    /// THE CORRECTION IS NOT DECORATIVE: DULA misses the target and DMALA does not.
    ///
    /// Judged by the crate's own certificate, so the comparison is against the sampling-noise floor
    /// rather than against zero. An unadjusted chain that landed inside the floor would mean
    /// [`Langevin::adjust`] switched between two correct samplers, which is neither what the paper
    /// claims nor what the algebra says.
    #[test]
    fn the_unadjusted_chain_is_biased_and_the_adjusted_one_is_not() {
        let g = frustrated(10);
        let beta = 0.9;
        let alpha = 2.0;
        let run = |adjust: bool| {
            let mut l = Langevin::new(&g, beta, 5).with_alpha(alpha).with_adjust(adjust);
            l.steps(500);
            let (mut samples, mut trace) = (Vec::new(), Vec::new());
            for _ in 0..20_000 {
                l.step();
                samples.push(l.s.clone());
                trace.push(l.energy());
            }
            crate::certify::certify(&g, beta, &samples, &trace)
        };
        let biased = run(false);
        let corrected = run(true);
        assert!(
            biased
                .findings
                .iter()
                .any(|f| matches!(f, crate::certify::Finding::AboveNoiseFloor { .. })),
            "DULA at alpha {alpha} was not caught missing the target, so this test is not seeing \
             the bias it exists to see:\n{biased}"
        );
        crate::certify::assert_boltzmann(&corrected, beta, "dmala");
        let (b, c) = (biased.tv_exact.unwrap(), corrected.tv_exact.unwrap());
        assert!(c * 2.0 < b, "the correction bought little: tv {c:.5} adjusted against {b:.5} not");
    }

    /// AND WHERE IT BUYS NOTHING, on the fixture `informed` measured itself on.
    ///
    /// 256 spins, a random +/-1 ring with `n/4` chords and small fields — a spin glass, whose slow
    /// modes are extended rather than local. A path of eight or sixteen flips is not the shape of
    /// the barrier there, and at `beta = 2` the two chains are within a factor of two of each other.
    /// Asserted rather than left out, because a module that reported only the fixture it wins on
    /// would be reporting the fixture and not the algorithm.
    ///
    /// Colder than this the comparison stops being measurable at all: at `beta = 3` and `beta = 4`
    /// the `tau_int` estimates for adjacent path lengths swung between 4.6e2 and 5.5e5 on the same
    /// graph, which is the caveat `informed` states about its own `beta = 4` row and is why no
    /// number from that end is quoted here.
    #[test]
    fn a_spin_glass_is_not_the_shape_a_path_helps_with() {
        let n = 256;
        let g = chorded(n, 1);
        let beta = 2.0;
        let draws = 8_000usize;
        let len = 8;
        let per = (2 * n / (len + 1)).max(1);
        let mag = |s: &[i8]| s.iter().map(|&v| f64::from(v)).sum::<f64>();

        let mut it = crate::informed::Informed::new(&g, beta, 0);
        it.steps(draws * n / 10);
        let (mut e, mut m) = (Vec::with_capacity(draws), Vec::with_capacity(draws));
        for _ in 0..draws {
            it.steps(n);
            e.push(it.energy());
            m.push(mag(&it.s));
        }
        let a = tau_flips(n, &e, &m);

        let mut p = Path::new(&g, beta, 0).with_length(len);
        p.steps(draws * per / 10);
        let (mut e, mut m) = (Vec::with_capacity(draws), Vec::with_capacity(draws));
        for _ in 0..draws {
            p.steps(per);
            e.push(p.energy());
            m.push(mag(&p.s));
        }
        let b = tau_flips(n, &e, &m);

        println!("glass, beta {beta}: single-flip {a:.0} flips/sample, length-{len} path {b:.0}");
        assert!(
            a < b * 3.0 && b < a * 3.0,
            "the null result this test records no longer holds: single {a:.0} flips, path {b:.0}. \
             That is a finding either way and the module docs are now wrong"
        );
    }

    /// `m` disjoint `k`-cliques with small random fields.
    ///
    /// A designed barrier: a clique aligned by its `k-1` internal couplings costs `2(k-1)J` to
    /// break, and the cheapest way out is `k` coordinated flips. This is the shape a path proposal
    /// is FOR, and a fixture where the win can be attributed to a mechanism rather than to a seed.
    fn blocks(m: usize, k: usize, seed: u64) -> Graph {
        let mut rng = Pcg::new(seed, 0xB1);
        let n = m * k;
        let mut b = GraphBuilder::new(n);
        for blk in 0..m {
            for a in 0..k {
                for c in (a + 1)..k {
                    b.couple(blk * k + a, blk * k + c, 1.0);
                }
            }
        }
        for i in 0..n {
            b.bias(i, (rng.f64() - 0.5) * 0.6);
        }
        b.build()
    }

    /// `tau_int` in FLIPS, over the worse of energy and magnetization.
    ///
    /// Both, because [`crate::certify`] takes the worse of the two for a reason this fixture shows
    /// off: a chain frozen in one basin has a fast-jittering ENERGY, so an energy trace alone
    /// scores a stuck chain as perfectly mixed.
    fn tau_flips(n: usize, e: &[f64], m: &[f64]) -> f64 {
        crate::certify::tau_int(e).max(crate::certify::tau_int(m)) * n as f64
    }

    /// THE MEASUREMENT THE MODULE EXISTS FOR: a path crosses a barrier a single flip cannot.
    ///
    /// Both arms get the SAME number of single spin flips and one draw per `n` flips, so `tau_int`
    /// is in flips for both and the comparison is work against work — the accounting of
    /// `examples/informed_mixing.rs`. Median of three seeds.
    ///
    /// Two rows, and the null one is the point of the fixture as much as the win is: at `beta = 0.5`
    /// there is no barrier and the two are within a factor of two, which is what `informed` says
    /// about itself against Gibbs. At `beta = 1.5` the single-flip chain needs about a hundred times
    /// more flips per independent sample. THIS IS A MEASUREMENT, NOT A CORRECTNESS CLAIM;
    /// correctness is the invariance and reachability tests above.
    ///
    /// **Not reported at `beta >= 2`, and the reason is in the test.** There the single-flip chain
    /// is frozen, and `tau_int` of a frozen chain comes back at its 0.5 FLOOR — the best score
    /// available — because a configuration that never moves has no autocorrelation left to measure.
    /// The freeze is asserted directly instead, on the range of the magnetization.
    #[test]
    fn paths_cross_barriers_a_single_flip_cannot() {
        let g = blocks(32, 4, 2);
        let n = g.n;
        let draws = 8_000usize;
        let len = 8;
        let per = (2 * n / (len + 1)).max(1); // mean drawn length 4.5, so about n flips per draw
        let mag = |s: &[i8]| s.iter().map(|&v| f64::from(v)).sum::<f64>();
        let med = |mut v: Vec<f64>| {
            v.sort_by(f64::total_cmp);
            v[v.len() / 2]
        };
        let range = |v: &[f64]| {
            v.iter().copied().fold(f64::NEG_INFINITY, f64::max)
                - v.iter().copied().fold(f64::INFINITY, f64::min)
        };

        for (beta, bar) in [(0.5, 2.0), (1.5, 20.0)] {
            let (mut single, mut multi) = (Vec::new(), Vec::new());
            for seed in 0..3u64 {
                let mut it = crate::informed::Informed::new(&g, beta, seed);
                it.steps(draws * n / 10);
                let (mut e, mut m) = (Vec::with_capacity(draws), Vec::with_capacity(draws));
                for _ in 0..draws {
                    it.steps(n);
                    e.push(it.energy());
                    m.push(mag(&it.s));
                }
                single.push(tau_flips(n, &e, &m));

                let mut p = Path::new(&g, beta, seed).with_length(len);
                p.steps(draws * per / 10);
                let (mut e, mut m) = (Vec::with_capacity(draws), Vec::with_capacity(draws));
                for _ in 0..draws {
                    p.steps(per);
                    e.push(p.energy());
                    m.push(mag(&p.s));
                }
                multi.push(tau_flips(n, &e, &m));
            }
            let (a, b) = (med(single), med(multi));
            println!("beta {beta}: single-flip {a:.0} flips/sample, length-{len} path {b:.0}");
            if bar > 2.5 {
                assert!(
                    b * bar < a,
                    "at beta {beta} the path was not {bar}x cheaper per flip: single {a:.0} \
                     flips, path {b:.0}"
                );
            } else {
                assert!(
                    a < b * bar && b < a * bar,
                    "at beta {beta} there is no barrier to cross, so the two must be within {bar}x \
                     of each other: single {a:.0} flips, path {b:.0}"
                );
            }
        }

        // beta 2: the single-flip chain is stuck, which tau_int scores as perfect mixing.
        let beta = 2.0;
        let mut it = crate::informed::Informed::new(&g, beta, 0);
        it.steps(draws * n / 10);
        let (mut e, mut m) = (Vec::with_capacity(draws), Vec::with_capacity(draws));
        for _ in 0..draws {
            it.steps(n);
            e.push(it.energy());
            m.push(mag(&it.s));
        }
        let (stuck_tau, stuck_range) = (tau_flips(n, &e, &m), range(&m));
        let mut p = Path::new(&g, beta, 0).with_length(len);
        p.steps(draws * per / 10);
        let mut m2 = Vec::with_capacity(draws);
        for _ in 0..draws {
            p.steps(per);
            m2.push(mag(&p.s));
        }
        let moving_range = range(&m2);
        println!(
            "beta {beta}: single-flip tau {stuck_tau:.0} flips over a magnetization range of \
             {stuck_range}, against the path's {moving_range}"
        );
        assert!(
            stuck_tau < 200.0,
            "this assertion exists to record that tau_int reports {stuck_tau:.0} flips — near its \
             0.5-draw floor — for a chain that is not mixing at all"
        );
        assert!(
            moving_range > 4.0 * stuck_range,
            "and the magnetization is what says so: the single-flip chain covered {stuck_range} \
             against the path's {moving_range}"
        );
    }
}
