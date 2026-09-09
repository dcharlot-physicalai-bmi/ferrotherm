//! Locally-informed proposals — choosing WHICH spin to flip by how much it would help.
//!
//! # The gap this fills
//!
//! Every other single-flip sampler in this crate picks its next site the same way: in order, or
//! uniformly at random. [`crate::gibbs`], [`crate::icm`], [`crate::tabu`], [`crate::sqa`],
//! [`crate::popanneal`] — twelve samplers, and not one of them lets the ENERGY decide where to
//! look. That is the default the discrete-sampling literature moved away from: Zanella's
//! locally-balanced proposals (JASA 2020) and Grathwohl et al.'s *Oops I Took A Gradient*
//! (ICML 2021) both weight the proposal by how much each move would change the target, and both
//! report order-of-magnitude gains on structured models.
//!
//! # For an Ising model the informed proposal is exact and costs nothing extra
//!
//! Those papers approximate the energy change with a gradient, because their state spaces are too
//! large to score every neighbour. A single-flip neighbourhood on a spin model is not: flipping
//! site `k` changes the energy by exactly `ΔE_k = 2 s_k f_k`, and `f_k` is the local field every
//! sampler here already computes. So there is no approximation to make and no Taylor expansion to
//! justify — the weight is the exact ratio.
//!
//! # Locally balanced, and why the acceptance is so simple
//!
//! Propose site `k` with probability proportional to `g(r_k)`, where `r_k = π(flip k)/π(s)` and `g`
//! is any **balancing function**: one satisfying `g(x) = x · g(1/x)`. [`Balance`] carries the three
//! standard choices. That condition is exactly what makes the energy cancel out of the
//! Metropolis–Hastings ratio:
//!
//! ```text
//!   α = min(1, [π(j)/π(i)] · [Q(j→i)/Q(i→j)])
//!     = min(1, r · (g(1/r)/Z_j) · (Z_i/g(r)))
//!     = min(1, Z_i / Z_j)                        since  g(1/r) = g(r)/r
//! ```
//!
//! **The acceptance depends only on the two normalisers**, for every balancing function. A high
//! acceptance rate therefore means the proposal landscape barely moved, not that the move was
//! small — which is the opposite of what an acceptance rate usually tells you, and is why
//! [`Informed::acceptance`] is documented rather than left to be read the usual way.
//!
//! # Cost
//!
//! One step is `O(deg)`, the same per-flip cost as one site of a Gibbs sweep: flipping `k` moves
//! the field at its neighbours and nowhere else, so only `k` and its neighbours need reweighing.
//! The comparison against Gibbs is therefore flips against flips.
//!
//! # What it is worth here, measured
//!
//! `examples/informed_mixing.rs`, 256-spin frustrated ring with chords, four seeds, both arms given
//! the same number of spin flips. Integrated autocorrelation time of the energy, **in flips**, so
//! lower is better:
//!
//! ```text
//!    beta        gibbs         sqrt       barker   metropolis   sqrt acc
//!     0.2          134          131          137          144      0.998
//!     0.5          223          164          172          187      0.993
//!     1.0         1246          406          378          414      0.966
//!     2.0        27638         1598          671         5135      0.856
//!     4.0        95593        56463        24627        40391      0.927
//! ```
//!
//! **At `beta = 2` the informed chain is forty-one times faster per flip.** At `beta = 0.2` it is
//! not faster at all, which is the prediction stated above rather than a disappointment: a flat
//! landscape makes every weight equal, and this is then Gibbs with extra arithmetic. The
//! `sqrt acc` column is what says which regime you are in.
//!
//! **The balancing function matters more than the literature suggests, and not in the direction it
//! suggests.** Zanella recommends `√x` as the most concentrated choice for multimodal targets. Here
//! `Barker` wins at both cold rungs and by a wide margin — 671 against 1598 at `beta = 2`, a factor
//! of 2.4, and 24627 against 56463 at `beta = 4`. The spread across the three functions at
//! `beta = 2` is eightfold, which is larger than the gap between the worst informed chain and Gibbs
//! at `beta = 1`. Picking one without measuring is picking most of the available speedup.
//!
//! The `beta = 4` row is the least reliable of the five and is reported with that caveat: at
//! `tau_int` around 24,000 flips a 4,000-draw trace holds only about ten independent points, which
//! is thin for Sokal windowing. The ordering there is worth less than the ordering at `beta = 2`.

use crate::graph::Graph;
use crate::rng::Pcg;

/// A balancing function `g` with `g(x) = x · g(1/x)`, which is what makes the chain reversible.
///
/// All three are standard and none dominates: the choice changes how sharply the proposal
/// concentrates on the best move. `Sqrt` is the most concentrated and is Zanella's recommendation
/// for multimodal targets; `Barker` and `Metropolis` are bounded by 1, so they never chase a single
/// enormous weight.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Balance {
    /// `g(x) = √x`. Weight `exp(−β s_k f_k)`: the most concentrated of the three.
    Sqrt,
    /// `g(x) = x / (1 + x)`, Barker's rule. Bounded by 1.
    Barker,
    /// `g(x) = min(1, x)`. Bounded by 1, and zero-gradient once a move is downhill.
    Metropolis,
}

impl Balance {
    /// `ln g(r)` from `ln r`, computed in logs so a large field cannot overflow before the shift.
    fn log_g(self, log_r: f64) -> f64 {
        match self {
            Balance::Sqrt => 0.5 * log_r,
            Balance::Metropolis => log_r.min(0.0),
            // ln(r/(1+r)) = −ln(1+e^{−ln r}), written through the stable softplus so the two tails
            // are both exact rather than one of them being `ln(1 + inf)`.
            Balance::Barker => -softplus(-log_r),
        }
    }
}

/// `ln(1 + e^x)`, without overflowing for large `x` or losing the small-`x` end.
fn softplus(x: f64) -> f64 {
    if x > 0.0 { x + (-x).exp().ln_1p() } else { x.exp().ln_1p() }
}

/// A single-flip sampler whose proposal is weighted by the energy change.
///
/// Reversible with respect to the Boltzmann distribution at `beta` — checked against exact
/// enumeration in `the_informed_chain_samples_the_boltzmann_distribution`, not argued for here.
pub struct Informed<'g> {
    /// The model.
    pub g: &'g Graph,
    /// Inverse temperature.
    pub beta: f64,
    /// Current state.
    pub s: Vec<i8>,
    /// The sampler's stream.
    pub rng: Pcg,
    /// Which balancing function the proposal uses.
    pub balance: Balance,
    /// `f_k` for every site, maintained incrementally.
    fields: Vec<f64>,
    /// `g(r_k)` for every site, shifted so nothing overflows. Only ratios of sums are ever used.
    w: Vec<f64>,
    /// `Σ_k w_k`, maintained incrementally and refreshed exactly every [`REFRESH`] steps.
    total: f64,
    /// A constant subtracted from every `ln g`, so `Sqrt` cannot overflow. It cancels in `Z_i/Z_j`.
    shift: f64,
    steps: u64,
    accepted: u64,
}

/// Steps between exact recomputations of the weight total.
///
/// The total is maintained by adding and subtracting the entries that moved, which is `O(deg)`
/// rather than `O(n)` and accumulates rounding for as long as it runs. Refreshing bounds that drift
/// without giving up the incremental cost;
/// `the_incremental_total_agrees_with_a_recomputation` is what says the two agree in between.
pub const REFRESH: u64 = 4096;

impl<'g> Informed<'g> {
    /// A sampler at `beta`, started from a random state drawn from `seed`.
    #[must_use]
    pub fn new(g: &'g Graph, beta: f64, seed: u64) -> Informed<'g> {
        let mut rng = Pcg::new(seed, 0x1F0D);
        let s: Vec<i8> = (0..g.n).map(|_| if rng.f64() < 0.5 { -1i8 } else { 1 }).collect();
        Self::from_state(g, beta, s, rng)
    }

    /// A sampler started from a given state.
    ///
    /// # Panics
    ///
    /// If `s` does not have one spin per node — a state of the wrong length is a different model,
    /// and continuing would index past the end of the fields on the first step.
    #[must_use]
    pub fn from_state(g: &'g Graph, beta: f64, s: Vec<i8>, rng: Pcg) -> Informed<'g> {
        assert_eq!(s.len(), g.n, "the state must have one spin per node");
        // Every weight is at most `exp(shift)` before shifting, so subtracting it leaves them all
        // at or below 1 and the total at or below n. The shift is a CONSTANT, so it divides out of
        // `Z_i/Z_j` exactly and changes no probability.
        let shift = g.flip_gap_max().map_or(0.0, |gap| beta.abs() * gap * 0.5);
        let mut it = Informed {
            g,
            beta,
            s,
            rng,
            balance: Balance::Sqrt,
            fields: Vec::new(),
            w: Vec::new(),
            total: 0.0,
            shift,
            steps: 0,
            accepted: 0,
        };
        it.refresh();
        it
    }

    /// Choose the balancing function. `Sqrt` is the default.
    #[must_use]
    pub fn with_balance(mut self, balance: Balance) -> Self {
        self.balance = balance;
        self.refresh();
        self
    }

    /// Recompute every field, weight and the total from the state, exactly.
    fn refresh(&mut self) {
        self.fields = (0..self.g.n).map(|i| self.g.field(i, &self.s)).collect();
        self.w = (0..self.g.n).map(|k| self.weight(k)).collect();
        self.total = crate::round::sum_up(&self.w).max(0.0);
    }

    /// `g(r_k)` for flipping site `k`, shifted.
    fn weight(&self, k: usize) -> f64 {
        // ΔE_k = 2 s_k f_k, so ln r_k = −β ΔE_k = −2 β s_k f_k.
        let log_r = -2.0 * self.beta * f64::from(self.s[k]) * self.fields[k];
        (self.balance.log_g(log_r) - self.shift).exp()
    }

    /// One informed flip, accepted or rejected. Returns whether the state moved.
    ///
    /// # The rejected branch must restore EVERYTHING
    ///
    /// A rejected move has to put back the spin, the neighbours' fields, the weights and the total.
    /// Flipping twice restores the first three exactly — every update is `+= w·2s` and its own
    /// inverse — but the total is a running sum and would keep the rounding from both passes, so a
    /// rejection refreshes it. That is the cheapest place to be exact rather than nearly so.
    pub fn step(&mut self) -> bool {
        self.steps += 1;
        if self.steps.is_multiple_of(REFRESH) {
            self.refresh();
        }
        if !(self.total > 0.0) || !self.total.is_finite() {
            // Every neighbour is astronomically unfavourable, so the proposal is undefined rather
            // than uniform. Standing still is the honest move: reporting a flip here would be
            // sampling from a distribution nobody asked for.
            return false;
        }
        let target = self.rng.f64() * self.total;
        let mut acc = 0.0;
        let mut k = self.g.n - 1;
        for (i, &wi) in self.w.iter().enumerate() {
            acc += wi;
            if acc >= target {
                k = i;
                break;
            }
        }

        let before = self.total;
        self.flip(k);
        let after = self.total;
        // α = min(1, Z_i / Z_j), for every balancing function. See the module docs.
        if after <= before || self.rng.f64() < before / after {
            self.accepted += 1;
            true
        } else {
            self.flip(k);
            self.total = crate::round::sum_up(&self.w).max(0.0);
            false
        }
    }

    /// Flip site `k` and repair the fields, weights and total it touched.
    fn flip(&mut self, k: usize) {
        self.s[k] = -self.s[k];
        let sk = f64::from(self.s[k]);
        // `f_k` does not contain `s_k`, so only the neighbours' fields move.
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

    /// Run `n` steps.
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

    /// Fraction of proposals accepted.
    ///
    /// **Read this the other way round from a Metropolis acceptance rate.** The ratio is
    /// `Z_i / Z_j`, so a high number means the PROPOSAL LANDSCAPE barely changed, not that the move
    /// was timid. A locally-informed chain that accepts almost everything is one whose weights are
    /// nearly uniform — which is the regime where it has nothing to offer over plain Gibbs.
    #[must_use]
    pub fn acceptance(&self) -> f64 {
        if self.steps == 0 { 0.0 } else { self.accepted as f64 / self.steps as f64 }
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

    /// A frustrated ring with fields: no sampler can get this right by accident, and it is small
    /// enough for `certify` to compare against the exact Boltzmann distribution.
    fn frustrated(n: usize) -> Graph {
        let mut b = GraphBuilder::new(n);
        for i in 0..n {
            b.couple(i, (i + 1) % n, if i % 3 == 0 { -0.9 } else { 0.7 });
        }
        b.bias(0, 0.4);
        b.bias(n / 2, -0.55);
        b.build()
    }

    /// THE ONLY CLAIM THAT MATTERS: it samples the right distribution.
    ///
    /// A proposal weighted by the energy is exactly the kind of change that improves mixing and
    /// silently shifts the target — a chain that goes downhill more eagerly than it comes back up
    /// still converges, to the wrong thing. Checked against enumeration through the crate's own
    /// certificate rather than against a longer run of itself, and at all three balancing
    /// functions, because the acceptance cancellation is a property of the FAMILY and a rule that
    /// held for one `g` would prove nothing about the others.
    #[test]
    fn the_informed_chain_samples_the_boltzmann_distribution() {
        let g = frustrated(12);
        let beta = 0.8;
        for balance in [Balance::Sqrt, Balance::Barker, Balance::Metropolis] {
            let mut it = Informed::new(&g, beta, 7).with_balance(balance);
            // Burn in, then one draw per n accepted-or-not steps, so a draw is a sweep's worth of
            // opportunity and the trace is comparable to a Gibbs one.
            it.steps(20 * g.n);
            let (mut samples, mut trace) = (Vec::new(), Vec::new());
            for _ in 0..6_000 {
                it.steps(g.n);
                samples.push(it.s.clone());
                trace.push(it.energy());
            }
            let cert = crate::certify::certify(&g, beta, &samples, &trace);
            crate::certify::assert_boltzmann(&cert, beta, &format!("informed/{balance:?}"));
        }
    }

    /// THE BALANCING CONDITION `g(x) = x g(1/x)`, which is what the acceptance cancellation rests
    /// on. Asserted on the functions themselves: if a future `Balance` variant breaks it the chain
    /// is no longer reversible, and the distribution test above would have to notice a bias that
    /// might be small.
    #[test]
    fn every_balancing_function_is_balanced() {
        for balance in [Balance::Sqrt, Balance::Barker, Balance::Metropolis] {
            let mut x = -8.0f64;
            while x <= 8.0 {
                // ln g(x) against ln x + ln g(1/x), i.e. `log_g(v) == v + log_g(-v)`.
                let lhs = balance.log_g(x);
                let rhs = x + balance.log_g(-x);
                assert!(
                    (lhs - rhs).abs() < 1e-12,
                    "{balance:?} is not balanced at ln r = {x}: {lhs} against {rhs}"
                );
                x += 0.25;
            }
        }
    }

    /// The incremental bookkeeping must agree with a recomputation, or the proposal is drawn from
    /// a total that has drifted away from the weights it is meant to normalise.
    #[test]
    fn the_incremental_total_agrees_with_a_recomputation() {
        let g = frustrated(24);
        let mut it = Informed::new(&g, 0.7, 3);
        let mut worst = 0.0f64;
        for _ in 0..(REFRESH as usize - 1) {
            it.step();
            // Fields first: everything else is derived from them.
            for i in 0..g.n {
                let exact = g.field(i, &it.s);
                worst = worst.max((it.fields[i] - exact).abs());
            }
            let fresh: f64 = crate::round::sum_up(&it.w);
            worst = worst.max((it.total - fresh).abs() / fresh.max(1.0));
        }
        assert!(worst < 1e-9, "incremental state drifted from a recomputation by {worst:e}");
        // And the refresh must actually be reachable within a run, or the guard is decoration.
        assert!(it.taken() < REFRESH, "the loop must stop short of the refresh to test the drift");
    }

    /// A model with no couplings and no fields has no energy scale, and the shift divides by it.
    /// It must sample rather than divide by zero: every state is equally likely there.
    #[test]
    fn a_model_with_no_energy_scale_still_samples() {
        let g = GraphBuilder::new(6).build();
        let mut it = Informed::new(&g, 1.0, 5);
        it.steps(2_000);
        assert!(it.acceptance() > 0.9, "a flat landscape accepts nearly everything");
        assert!(it.energy().abs() < 1e-12, "and every state has zero energy");
    }

    /// The proposal must actually be INFORMED — i.e. it must not be uniform. On a model with one
    /// strongly-biased site, the weight for flipping that site into agreement has to dominate.
    ///
    /// Without this, every test above passes for a sampler that picks uniformly and accepts
    /// everything: uniform proposals are also reversible, also converge, and are exactly what this
    /// module exists not to be.
    #[test]
    fn the_proposal_is_not_uniform() {
        let mut b = GraphBuilder::new(8);
        for i in 0..7 {
            b.couple(i, i + 1, 0.05);
        }
        b.bias(3, 4.0); // site 3 wants to be +1, hard
        let g = b.build();
        let mut it = Informed::new(&g, 1.0, 11);
        // Force the state that makes site 3 the obviously-best flip.
        it.s = vec![1i8; 8];
        it.s[3] = -1;
        it.refresh();
        let best = it.w.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        assert!(
            (it.w[3] - best).abs() < 1e-12,
            "the site that most wants to flip must carry the largest weight: {:?}",
            it.w
        );
        let share = it.w[3] / it.total;
        assert!(share > 0.9, "and it must dominate the proposal, not merely lead it: {share:.4}");
    }
}
