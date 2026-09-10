//! Tempered negative phases for contrastive divergence.
//!
//! [`crate::ebm::train`] estimates `⟨s_i s_j⟩_model` from a chain run at the model's own
//! temperature. Once the fitted model is multimodal that chain stops crossing between modes and
//! the negative phase reports the statistics of ONE of them, so the gradient pushes the couplings
//! down until the chain mixes again: the fit is then limited by the sampler rather than by the
//! structure. Running the negative phase on a temperature ladder is the published fix, in two
//! forms, and this module is those two forms and nothing else — the positive phase, the parameter
//! update and the scoring are [`crate::ebm`]'s.
//!
//! ```text
//!   Negative::ParallelTempering    a persistent chain per rung, swapped   Desjardins et al. 2010
//!   Negative::TemperedTransitions  one chain, heated then cooled          Salakhutdinov 2009
//! ```
//!
//! The ladder runs hot end first and must END at `beta = 1`, the model's own temperature — the same
//! orientation [`crate::tempering::geometric_ladder`] produces, so a ladder built there is usable
//! here unchanged. `Params::k` is the Gibbs sweeps spent per rung.
//!
//! # What is verified
//!
//! A tempered transition's kernel is ENUMERATED on a four-spin model — every trajectory, not just
//! the endpoints — and the Boltzmann distribution is shown to be stationary under it to `1e-12`.
//! Then [`tempered_transition`] itself is sampled 200,000 times per row and matched against that
//! same matrix, because the enumeration is only mathematics until the implementation is joined to
//! it: a cooling leg that is not the heating leg's reversal passes the first check and fails the
//! second.
//!
//! The trainer is scored against [`crate::ebm::train_exact`], the exact maximum-likelihood ceiling,
//! on data drawn exactly from a deliberately bimodal model. The control is the same trainer with
//! the ladder's hot end raised until it reaches nothing, which fails the way PCD does — so what is
//! measured is the tempering and not the plumbing.

use crate::ebm::{Dataset, Params, Trained};
use crate::gibbs::Sampler;
use crate::graph::{Graph, GraphBuilder};
use crate::rng::Pcg;

/// Which tempered negative phase to run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Negative {
    /// Replica exchange: one persistent chain per rung, adjacent pairs swapped between advances,
    /// and the `beta = 1` chain supplies the negative sample (Desjardins, Courville, Bengio,
    /// Vincent & Delalleau, AISTATS 2010).
    ParallelTempering,
    /// A single chain taken up the ladder and back down, the whole excursion accepted or rejected
    /// at once (Neal 1996; Salakhutdinov, NIPS 2009).
    TemperedTransitions,
}

/// A negative-phase temperature ladder: inverse temperatures hot end first, ending at `1.0`.
#[derive(Clone, Debug, PartialEq)]
pub struct Ladder {
    /// Strictly increasing inverse temperatures whose last entry is the model's own `1.0`.
    pub betas: Vec<f64>,
    /// Which of the two tempered negative phases consumes it.
    pub negative: Negative,
}

impl Ladder {
    /// A geometric ladder from `beta_min` up to the model's own `beta = 1`.
    ///
    /// The top rung is pinned to exactly `1.0`: `beta_min · r^(n-1)` lands an ulp either side of it
    /// and a negative phase should be at the model's temperature, not next to it.
    ///
    /// # Panics
    ///
    /// As [`crate::tempering::geometric_ladder`]: fewer than two rungs, or `beta_min` outside
    /// `(0, 1)`.
    #[must_use]
    pub fn geometric(beta_min: f64, rungs: usize, negative: Negative) -> Ladder {
        let mut betas = crate::tempering::geometric_ladder(beta_min, 1.0, rungs);
        *betas.last_mut().expect("a ladder has rungs") = 1.0;
        Ladder { betas, negative }
    }

    /// Refuse a ladder that cannot carry a negative phase.
    ///
    /// The anchor is checked to `1e-12` rather than to the bit, so a ladder built by
    /// [`crate::tempering::geometric_ladder`] — whose top rung is a rounded power — is accepted.
    ///
    /// # Errors
    ///
    /// [`Error::TooShort`], [`Error::NotIncreasing`] or [`Error::NotAnchored`].
    pub fn validate(&self) -> Result<(), Error> {
        let r = self.betas.len();
        if r < 2 {
            return Err(Error::TooShort { rungs: r });
        }
        for i in 0..r - 1 {
            if !(self.betas[i + 1] > self.betas[i]) || !(self.betas[i] > 0.0) {
                return Err(Error::NotIncreasing {
                    at: i,
                    betas: (self.betas[i], self.betas[i + 1]),
                });
            }
        }
        if (self.betas[r - 1] - 1.0).abs() > 1e-12 {
            return Err(Error::NotAnchored { top: self.betas[r - 1] });
        }
        Ok(())
    }
}

/// Why a tempered fit was refused.
#[derive(Clone, Debug, PartialEq)]
pub enum Error {
    /// One rung is not a ladder: there is nothing to swap with and nothing to heat to.
    TooShort {
        /// Rungs the ladder has.
        rungs: usize,
    },
    /// The ladder is not strictly increasing and positive at this index.
    NotIncreasing {
        /// Index of the offending pair.
        at: usize,
        /// The pair, in ladder order.
        betas: (f64, f64),
    },
    /// The ladder does not end at `beta = 1`, so its cold chain samples a model nobody asked for.
    NotAnchored {
        /// The last rung, which should have been `1.0`.
        top: f64,
    },
    /// The structure or the data was refused, for the reasons [`crate::ebm::train`] refuses them.
    Fit(crate::ebm::Error),
}

impl From<crate::ebm::Error> for Error {
    fn from(e: crate::ebm::Error) -> Error {
        Error::Fit(e)
    }
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Error::TooShort { rungs } => {
                write!(f, "a tempered ladder needs at least two rungs and this one has {rungs}")
            }
            Error::NotIncreasing { at, betas } => write!(
                f,
                "rungs {at} and {} are {betas:?}, and a ladder runs hot end first through strictly \
                 increasing positive betas",
                at + 1
            ),
            Error::NotAnchored { top } => write!(
                f,
                "the ladder ends at beta {top}, not at the model's own 1.0, so its cold chain is \
                 not a sample from the model being fitted"
            ),
            Error::Fit(e) => write!(f, "{e}"),
        }
    }
}

/// One chromatic sweep with the colour classes visited in REVERSE order.
///
/// Neal's construction needs the cooling leg's operator to be the *reversal* of the heating leg's
/// with respect to that rung's distribution. A colour-class update is reversible on its own, so the
/// reversal of `C_0 ∘ C_1 ∘ …` is `… ∘ C_1 ∘ C_0` — the same sweep with the class order turned
/// around. [`Sampler`] has no such entry point, so the loop is restated here; it is
/// [`Sampler::sweep`] with one `rev()`, over the same field, kernel and stream primitives.
fn reverse_sweep(smp: &mut Sampler) {
    let g = smp.g;
    for class in g.classes.iter().rev() {
        for &iu in class {
            let i = iu as usize;
            if smp.clamped[i] {
                continue;
            }
            let f = g.field(i, &smp.s);
            let p = crate::kernel::p_up(f, smp.beta);
            smp.s[i] = smp.rng.spin(p);
        }
    }
}

/// The log acceptance ratio of a tempered transition.
///
/// `hot[t]` is the energy of the state entering the `t`-th heating step and `cool[t]` the energy of
/// the state leaving the matching cooling step, both indexed from the model end. Every partition
/// function on the way out cancels against the one on the way back, which is what leaves the whole
/// excursion computable from energies alone:
///
/// ```text
///   log a  =  Σ_t (beta[R-1-t] − beta[R-2-t]) · (hot[t] − cool[t])
/// ```
///
/// # Panics
///
/// Fewer than two rungs, or either leg not holding exactly `betas.len() - 1` energies.
#[must_use]
pub fn log_accept(betas: &[f64], hot: &[f64], cool: &[f64]) -> f64 {
    let r = betas.len();
    assert!(r >= 2, "a tempered transition needs at least two rungs");
    assert!(hot.len() == r - 1 && cool.len() == r - 1, "one energy per rung crossed, on each leg");
    (0..r - 1).map(|t| (betas[r - 1 - t] - betas[r - 2 - t]) * (hot[t] - cool[t])).sum()
}

/// Apply one tempered transition to `smp` in place; `true` if the excursion was accepted.
///
/// The chain is taken from `betas.last()` up to `betas[0]` with `sweeps` Gibbs sweeps per rung and
/// back down with the reversed sweep, and the whole trajectory is accepted or rejected by
/// [`log_accept`]. On return `smp.beta` is `betas.last()` again, whatever it was before. A
/// rejection restores the state and NOT the stream: the draws the excursion spent are spent.
///
/// # Panics
///
/// If the ladder has fewer than two rungs.
pub fn tempered_transition(smp: &mut Sampler, betas: &[f64], sweeps: usize) -> bool {
    let r = betas.len();
    assert!(r >= 2, "a tempered transition needs at least two rungs");
    let g = smp.g;
    let start = smp.s.clone();
    let steps = sweeps.max(1);

    let mut hot = Vec::with_capacity(r - 1);
    for t in 0..r - 1 {
        hot.push(g.energy(&smp.s));
        smp.beta = betas[r - 2 - t];
        smp.sweeps(steps, None);
    }
    let mut cool = vec![0.0; r - 1];
    for t in (0..r - 1).rev() {
        smp.beta = betas[r - 2 - t];
        for _ in 0..steps {
            reverse_sweep(smp);
        }
        cool[t] = g.energy(&smp.s);
    }

    smp.beta = betas[r - 1];
    let log_a = log_accept(betas, &hot, &cool);
    if log_a >= 0.0 || smp.rng.f64() < log_a.exp() {
        true
    } else {
        smp.s.copy_from_slice(&start);
        false
    }
}

/// A persistent ladder of fantasy chains, one per rung, swapped between advances.
///
/// This is the negative phase of Desjardins et al. 2010 as an object: it survives across parameter
/// updates the way [`crate::ebm::Params::persistent`]'s single chain does, and [`Replicas::cold`]
/// is the sample the gradient uses.
pub struct Replicas {
    /// The ladder, hot end first, ending at the model's `beta = 1`.
    pub betas: Vec<f64>,
    /// `(state, stream)` per rung. Swaps exchange states and leave each rung its own stream.
    chains: Vec<(Vec<i8>, Pcg)>,
    round: usize,
    attempts: Vec<u64>,
    accepts: Vec<u64>,
    swap_rng: Pcg,
}

impl Replicas {
    /// One chain of `n` spins per rung, started at random from `seed`.
    ///
    /// # Panics
    ///
    /// If the ladder has fewer than two rungs.
    #[must_use]
    pub fn new(n: usize, betas: &[f64], seed: u64) -> Replicas {
        assert!(betas.len() >= 2, "a swapped ladder needs at least two rungs");
        let mut rng = Pcg::new(seed, 0x00EB_7700);
        let chains = (0..betas.len())
            .map(|i| {
                let s: Vec<i8> = (0..n).map(|_| rng.spin(0.5)).collect();
                (s, Pcg::new(seed ^ (i as u64).wrapping_mul(0x9E37), 0x7E11))
            })
            .collect();
        Replicas {
            betas: betas.to_vec(),
            chains,
            round: 0,
            attempts: vec![0; betas.len() - 1],
            accepts: vec![0; betas.len() - 1],
            swap_rng: Pcg::new(seed ^ 0x5A5A, 3),
        }
    }

    /// Advance every rung by `sweeps` sweeps under `g`, then attempt swaps on this round's parity.
    ///
    /// The swap criterion is [`crate::tempering::parallel_tempering`]'s and the sweeps are its own
    /// replica advance, so the ladder here and the ladder there move the same way.
    ///
    /// # Panics
    ///
    /// If `g` has a different spin count than the chains were built with.
    pub fn advance(&mut self, g: &Graph, sweeps: usize) {
        let mut reps: Vec<Sampler> = self
            .chains
            .iter()
            .zip(&self.betas)
            .map(|((s, rng), &b)| {
                let mut smp = Sampler::new(g, b, 0);
                smp.s.copy_from_slice(s);
                smp.rng = rng.clone();
                smp
            })
            .collect();
        crate::tempering::advance(&mut reps, sweeps.max(1), None);
        for (chain, rep) in self.chains.iter_mut().zip(&reps) {
            chain.0.copy_from_slice(&rep.s);
            chain.1 = rep.rng.clone();
        }
        drop(reps);

        let r = self.betas.len();
        for i in (self.round % 2..r - 1).step_by(2) {
            let e_i = g.energy(&self.chains[i].0);
            let e_j = g.energy(&self.chains[i + 1].0);
            let arg = (self.betas[i + 1] - self.betas[i]) * (e_j - e_i);
            self.attempts[i] += 1;
            if arg >= 0.0 || self.swap_rng.f64() < arg.exp() {
                self.accepts[i] += 1;
                let (a, b) = self.chains.split_at_mut(i + 1);
                core::mem::swap(&mut a[i].0, &mut b[0].0);
            }
        }
        self.round += 1;
    }

    /// The `beta = 1` chain's state — the negative sample.
    #[must_use]
    pub fn cold(&self) -> &[i8] {
        &self.chains[self.betas.len() - 1].0
    }

    /// Swap acceptance rate per adjacent pair. A near-zero entry is a rung gap nothing crosses.
    #[must_use]
    pub fn swap_rates(&self) -> Vec<f64> {
        (0..self.attempts.len())
            .map(|i| self.accepts[i] as f64 / self.attempts[i].max(1) as f64)
            .collect()
    }
}

/// What a tempered fit produced.
pub struct TemperedFit {
    /// The fit, exactly as [`crate::ebm::train`] returns one.
    pub fit: Trained,
    /// Ladder health: swap acceptance per adjacent pair for [`Negative::ParallelTempering`],
    /// averaged over the fantasy slots; a single excursion-acceptance rate for
    /// [`Negative::TemperedTransitions`]. Near zero means the rungs are too far apart and the
    /// ladder has a gap nothing crosses.
    ///
    /// **A high rate is not health.** Adjacent rungs at nearly the same temperature always swap, so
    /// a ladder that never gets hot reports the best numbers here and fits the worst — measured, in
    /// `a_ladder_that_never_gets_hot_fails_the_way_pcd_does`, at 0.98 acceptance and 0.40 of a
    /// ceiling the same trainer reaches at 0.83 acceptance. Read this alongside `betas[0]`.
    pub acceptance: Vec<f64>,
}

impl core::fmt::Debug for TemperedFit {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("TemperedFit")
            .field("fit", &self.fit)
            .field("acceptance", &self.acceptance)
            .finish()
    }
}

/// The rows-and-shape guard [`crate::ebm`] applies before fitting.
///
/// Restated rather than shared because `ebm`'s copy is private to that module; the variants it
/// returns are `ebm`'s own, so a caller still sees one error type.
fn check_fit(g: &Graph, data: &Dataset) -> Result<(), crate::ebm::Error> {
    if data.rows.is_empty() {
        return Err(crate::ebm::Error::NoData);
    }
    if g.n < data.visible {
        return Err(crate::ebm::Error::TooSmall { spins: g.n, visible: data.visible });
    }
    for (r, row) in data.rows.iter().enumerate() {
        if row.len() != data.visible {
            return Err(crate::ebm::Error::RowWidth { row: r, len: row.len(), want: data.visible });
        }
        if let Some(at) = row.iter().position(|&v| v != 1 && v != -1) {
            return Err(crate::ebm::Error::NotASpin { row: r, at, value: row[at] });
        }
    }
    Ok(())
}

/// The fantasy pool, whichever kind the ladder asked for.
enum Pool {
    Swapped(Vec<Replicas>),
    Excursion(Vec<(Vec<i8>, Pcg)>),
}

/// Fit `structure`'s weights to `data` with a TEMPERED negative phase.
///
/// The positive phase, the gradient and the step are [`crate::ebm::train`]'s: visible units clamped
/// to the row, latent units settled for `p.positive_sweeps`, the same linear decay of
/// `p.learning_rate` to a tenth over the epochs. Only the model average differs, and it comes from
/// `ladder` instead of from a chain at `beta = 1`.
///
/// `p.k` is the sweeps spent PER RUNG, so an update costs `p.k × ladder.betas.len()` sweeps for
/// parallel tempering and about twice that for tempered transitions. `p.persistent` is not read: a
/// ladder is persistent by construction.
///
/// # Errors
///
/// [`Error::TooShort`], [`Error::NotIncreasing`] or [`Error::NotAnchored`] for a malformed ladder,
/// and [`Error::Fit`] for the data and structure faults [`crate::ebm::train`] refuses.
pub fn train_tempered(
    structure: &Graph,
    data: &Dataset,
    p: &Params,
    ladder: &Ladder,
    seed: u64,
) -> Result<TemperedFit, Error> {
    check_fit(structure, data)?;
    ladder.validate()?;
    let n = structure.n;
    let mut rng = Pcg::new(seed, 0x00EB_7000);

    let mut edges: Vec<(usize, usize, f64)> = Vec::with_capacity(structure.n_edges);
    for i in 0..n {
        for k in structure.offset[i]..structure.offset[i + 1] {
            let j = structure.nbr[k] as usize;
            if j > i {
                edges.push((i, j, structure.w[k]));
            }
        }
    }
    let mut bias: Vec<f64> = structure.h.clone();

    let build = |edges: &[(usize, usize, f64)], bias: &[f64]| {
        let mut gb = GraphBuilder::new(n);
        for &(i, j, w) in edges {
            gb.couple(i, j, w);
        }
        for (i, &b) in bias.iter().enumerate() {
            if b != 0.0 {
                gb.bias(i, b);
            }
        }
        gb.build()
    };

    let slots = p.batch.max(1);
    let mut pool = match ladder.negative {
        Negative::ParallelTempering => Pool::Swapped(
            (0..slots)
                .map(|s| Replicas::new(n, &ladder.betas, seed ^ 0xF0F0_0000 ^ s as u64))
                .collect(),
        ),
        Negative::TemperedTransitions => Pool::Excursion(
            (0..slots)
                .map(|s| {
                    (
                        (0..n).map(|_| rng.spin(0.5)).collect(),
                        Pcg::new(seed ^ 0xE0E0_0000 ^ s as u64, 0x9E37),
                    )
                })
                .collect(),
        ),
    };
    let (mut excursions, mut accepted) = (0u64, 0u64);

    let mut order: Vec<usize> = (0..data.rows.len()).collect();
    for epoch in 0..p.epochs {
        for i in (1..order.len()).rev() {
            let j = (rng.f64() * (i + 1) as f64) as usize % (i + 1);
            order.swap(i, j);
        }
        let g = build(&edges, &bias);
        let decay =
            if p.epochs > 1 { 1.0 - 0.9 * epoch as f64 / (p.epochs - 1) as f64 } else { 1.0 };

        for chunk in order.chunks(slots) {
            let mut d_edge = vec![0.0f64; edges.len()];
            let mut d_bias = vec![0.0f64; n];

            for (slot, &r) in chunk.iter().enumerate() {
                let row = &data.rows[r];

                // POSITIVE PHASE, unchanged: visible clamped to the row, latent settled around it.
                let rseed = (u64::from(rng.next_u32()) << 32) | u64::from(rng.next_u32());
                let mut smp = Sampler::new(&g, 1.0, rseed);
                for (i, &v) in row.iter().enumerate() {
                    smp.clamp(i, v);
                }
                smp.sweeps(p.positive_sweeps.max(1), None);
                let pos = smp.s;

                // NEGATIVE PHASE, on the ladder.
                let neg: Vec<i8> = match &mut pool {
                    Pool::Swapped(ladders) => {
                        let reps = &mut ladders[slot % slots];
                        reps.advance(&g, p.k.max(1));
                        reps.cold().to_vec()
                    }
                    Pool::Excursion(chains) => {
                        let (state, stream) = &mut chains[slot % slots];
                        let mut fs = Sampler::new(&g, 1.0, 0);
                        fs.s.copy_from_slice(state);
                        fs.rng = stream.clone();
                        excursions += 1;
                        accepted += u64::from(tempered_transition(&mut fs, &ladder.betas, p.k));
                        state.copy_from_slice(&fs.s);
                        *stream = fs.rng.clone();
                        fs.s
                    }
                };

                for (e, &(i, j, _)) in edges.iter().enumerate() {
                    d_edge[e] += f64::from(pos[i] * pos[j]) - f64::from(neg[i] * neg[j]);
                }
                for i in 0..n {
                    d_bias[i] += f64::from(pos[i]) - f64::from(neg[i]);
                }
            }

            let scale = p.learning_rate * decay / chunk.len() as f64;
            for (e, w) in edges.iter_mut().enumerate() {
                w.2 += scale * d_edge[e];
            }
            for i in 0..n {
                bias[i] += scale * d_bias[i];
            }
        }
    }

    let acceptance = match &pool {
        Pool::Swapped(ladders) => {
            let mut mean = vec![0.0; ladder.betas.len() - 1];
            for reps in ladders {
                for (m, r) in mean.iter_mut().zip(reps.swap_rates()) {
                    *m += r / ladders.len() as f64;
                }
            }
            mean
        }
        Pool::Excursion(_) => vec![accepted as f64 / excursions.max(1) as f64],
    };

    let graph = build(&edges, &bias);
    let log_likelihood = crate::ebm::exact_log_likelihood(&graph, data).ok();
    Ok(TemperedFit { fit: Trained { graph, log_likelihood, epochs_run: p.epochs }, acceptance })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ebm;
    use crate::ising::{exact_boltzmann, tv};

    /// A fully-connected ferromagnet with a uniform field: two modes with a barrier single-site
    /// Gibbs will not cross at `beta = 1`. On eight spins at `h = 0`, `E(all up) = -28 j` and
    /// `E(four and four) = +4 j`, so the free-energy barrier is `32 j − ln C(8,4)`.
    ///
    /// **The field is what makes the fitting tests tests rather than demonstrations.** On a
    /// SYMMETRIC two-mode model mode trapping costs almost nothing: `⟨s_i s_j⟩` is invariant under
    /// the global flip, so a chain stuck in one mode reports the same pair correlations as the whole
    /// distribution, and across several fantasy chains even `⟨s_i⟩` averages out. A field removes
    /// both escapes — the true `⟨s_i⟩` is a WEIGHTED average over two unequal modes, and a chain
    /// that cannot cross can never weigh them.
    fn two_modes(n: usize, j: f64, h: f64) -> Graph {
        let mut gb = GraphBuilder::new(n);
        for a in 0..n {
            for b in a + 1..n {
                gb.couple(a, b, j);
            }
            if h != 0.0 {
                gb.bias(a, h);
            }
        }
        gb.build()
    }

    fn key(s: &[i8]) -> usize {
        s.iter().enumerate().filter(|&(_, &v)| v == 1).map(|(i, _)| 1usize << i).sum()
    }

    /// A LADDER OF ONES IS PLAIN GIBBS. Every gap is zero, so `log a` is zero for any pair of
    /// trajectories and the excursion is always accepted — an algebraic identity, checked at the
    /// value level and then end to end.
    #[test]
    fn a_flat_ladder_always_accepts_and_moves_the_chain() {
        let betas = vec![1.0; 5];
        let hot = [-3.0, 7.5, 0.25, -11.0];
        let cool = [2.0, -6.0, 4.0, 9.0];
        assert_eq!(log_accept(&betas, &hot, &cool), 0.0);

        let g = two_modes(6, 0.3, 0.0);
        let mut smp = Sampler::new(&g, 1.0, 4);
        for _ in 0..200 {
            assert!(tempered_transition(&mut smp, &betas, 1), "a zero gap must never reject");
            assert_eq!(smp.beta, 1.0, "the transition must restore the model's own beta");
        }
    }

    /// The exact transition matrix of one chromatic sweep at `beta`, class order forward or
    /// reversed. Written independently of [`Sampler`]: a colour class is a set of non-adjacent
    /// sites, so its update draws each of them from `p_up` of a field the class cannot change.
    fn sweep_kernel(g: &Graph, beta: f64, reversed: bool) -> Vec<Vec<f64>> {
        let m = 1usize << g.n;
        let mut acc: Vec<Vec<f64>> =
            (0..m).map(|x| (0..m).map(|y| f64::from(u8::from(x == y))).collect()).collect();
        let classes: Vec<&Vec<u32>> =
            if reversed { g.classes.iter().rev().collect() } else { g.classes.iter().collect() };
        for class in classes {
            let mut k = vec![vec![0.0f64; m]; m];
            for x in 0..m {
                let s: Vec<i8> = (0..g.n).map(|i| if x >> i & 1 == 1 { 1i8 } else { -1 }).collect();
                let up: Vec<f64> = class
                    .iter()
                    .map(|&i| crate::kernel::p_up(g.field(i as usize, &s), beta))
                    .collect();
                for pick in 0..1usize << class.len() {
                    let mut y = x;
                    let mut w = 1.0;
                    for (b, &i) in class.iter().enumerate() {
                        if pick >> b & 1 == 1 {
                            w *= up[b];
                            y |= 1 << i;
                        } else {
                            w *= 1.0 - up[b];
                            y &= !(1usize << i);
                        }
                    }
                    k[x][y] += w;
                }
            }
            let mut next = vec![vec![0.0f64; m]; m];
            for x in 0..m {
                for z in 0..m {
                    if acc[x][z] != 0.0 {
                        for y in 0..m {
                            next[x][y] += acc[x][z] * k[z][y];
                        }
                    }
                }
            }
            acc = next;
        }
        acc
    }

    /// THE ACCEPTANCE RULE, CHECKED BY ENUMERATION AND NOT BY SAMPLING.
    ///
    /// A tempered transition's kernel is a sum over whole trajectories, because the acceptance
    /// depends on every energy visited and not only on the endpoints. On four spins that
    /// trajectory space is small enough to walk exactly, so `P(x -> z)` can be built term by term
    /// and the claim tested directly: the Boltzmann distribution at `beta = 1` is stationary under
    /// it, to floating point. A sign error in [`log_accept`], a rung read off by one, or a cooling
    /// leg that is not the reversal of the heating leg all break this, and none of them would show
    /// up in a total-variation test at any sample count anybody would run.
    #[test]
    fn the_transition_kernel_has_the_model_as_its_exact_stationary_distribution() {
        // Four spins in a ring with fields, so the two colour classes are genuine BLOCKS of two
        // and the model has no symmetry to hide an error behind.
        let mut gb = GraphBuilder::new(4);
        for i in 0..4 {
            gb.couple(i, (i + 1) % 4, if i % 2 == 0 { 0.7 } else { -0.4 });
        }
        gb.bias(0, 0.3);
        gb.bias(2, -0.55);
        let g = gb.build();
        assert_eq!(g.classes.len(), 2, "a 4-ring is bipartite");

        let m = 1usize << g.n;
        let pi = exact_boltzmann(&g, 1.0);
        let energy: Vec<f64> = (0..m)
            .map(|x| {
                let s: Vec<i8> = (0..g.n).map(|i| if x >> i & 1 == 1 { 1i8 } else { -1 }).collect();
                g.energy(&s)
            })
            .collect();

        for betas in [vec![0.35, 1.0], vec![0.2, 0.6, 1.0]] {
            let r = betas.len();
            // Heating step `t` and cooling step `t` both run at `betas[r - 2 - t]`.
            let fwd: Vec<Vec<Vec<f64>>> =
                (0..r - 1).map(|t| sweep_kernel(&g, betas[r - 2 - t], false)).collect();
            let rev: Vec<Vec<Vec<f64>>> =
                (0..r - 1).map(|t| sweep_kernel(&g, betas[r - 2 - t], true)).collect();

            let mut p = vec![vec![0.0f64; m]; m];
            // Walk every trajectory: the start, the r-1 heating results, the r-1 cooling results.
            let mut path = vec![0usize; 2 * (r - 1) + 1];
            let total = m.pow(path.len() as u32);
            for code in 0..total {
                let mut c = code;
                for slot in &mut path {
                    *slot = c % m;
                    c /= m;
                }
                let mut w = 1.0;
                for t in 0..r - 1 {
                    w *= fwd[t][path[t]][path[t + 1]];
                }
                for t in 0..r - 1 {
                    // The cooling leg runs the rungs in the opposite order to the heating leg.
                    w *= rev[r - 2 - t][path[r - 1 + t]][path[r + t]];
                }
                if w == 0.0 {
                    continue;
                }
                let hot: Vec<f64> = (0..r - 1).map(|t| energy[path[t]]).collect();
                // `cool[t]` leaves cooling step `t`, and step 0 is the one that leaves last.
                let cool: Vec<f64> = (0..r - 1).map(|t| energy[path[2 * (r - 1) - t]]).collect();
                let a = log_accept(&betas, &hot, &cool).exp().min(1.0);
                let (x, z) = (path[0], path[2 * (r - 1)]);
                p[x][z] += w * a;
                p[x][x] += w * (1.0 - a);
            }

            for x in 0..m {
                let row: f64 = p[x].iter().sum();
                assert!((row - 1.0).abs() < 1e-12, "row {x} of the kernel sums to {row}");
            }

            // AND THE IMPLEMENTATION MUST BE THAT KERNEL. Everything above is the mathematics; it
            // proves nothing about `tempered_transition` unless the two are joined here. A row of
            // `p` is a distribution over 16 states, so 200,000 draws pin it to about 0.007 in
            // total variation -- enough to see a cooling leg that is not the reversal, which is
            // otherwise a change no stationarity argument on paper can notice.
            for &x0 in &[0usize, 0b1010, 0b0111] {
                let mut smp = Sampler::new(&g, *betas.last().unwrap(), 77 + x0 as u64);
                let draws = 200_000usize;
                let mut got = vec![0.0f64; m];
                for _ in 0..draws {
                    for i in 0..g.n {
                        smp.s[i] = if x0 >> i & 1 == 1 { 1 } else { -1 };
                    }
                    tempered_transition(&mut smp, &betas, 1);
                    let mut z = 0usize;
                    for i in 0..g.n {
                        if smp.s[i] == 1 {
                            z |= 1 << i;
                        }
                    }
                    got[z] += 1.0 / draws as f64;
                }
                let d = tv(&got, &p[x0]);
                assert!(d < 0.02, "{r} rungs, row {x0}: sampled kernel is {d} from the exact one");
            }

            for z in 0..m {
                let got: f64 = (0..m).map(|x| pi[x] * p[x][z]).sum();
                assert!(
                    (got - pi[z]).abs() < 1e-12,
                    "{r} rungs, state {z}: pi P = {got}, pi = {}",
                    pi[z]
                );
            }
        }
    }

    /// The chain crosses a barrier, and the DISTRIBUTION it produces is the closed-form Boltzmann
    /// one — not another sampler's histogram.
    ///
    /// The control is not a different algorithm but the same one with the ladder removed: plain
    /// Gibbs at `beta = 1`, given the same number of sweeps. It stays in whichever mode it started
    /// in, which is the failure a tempered negative phase exists to remove.
    #[test]
    fn a_tempered_chain_reaches_both_modes_and_a_gibbs_chain_reaches_one() {
        let (n, j) = (8usize, 0.7);
        let g = two_modes(n, j, 0.0);
        let pi = exact_boltzmann(&g, 1.0);
        let betas = crate::tempering::geometric_ladder(0.05, 1.0, 12);
        let draws = 20_000usize;
        let per = 2 * (betas.len() - 1); // sweeps one excursion costs

        let mut smp = Sampler::new(&g, 1.0, 21);
        for _ in 0..500 {
            tempered_transition(&mut smp, &betas, 1);
        }
        let mut hist = vec![0.0f64; 1 << n];
        for _ in 0..draws {
            tempered_transition(&mut smp, &betas, 1);
            hist[key(&smp.s)] += 1.0 / draws as f64;
        }
        let tempered = tv(&hist, &pi);

        let mut plain = Sampler::new(&g, 1.0, 21);
        plain.sweeps(500 * per, None);
        let mut flat = vec![0.0f64; 1 << n];
        for _ in 0..draws {
            plain.sweeps(per, None);
            flat[key(&plain.s)] += 1.0 / draws as f64;
        }
        let gibbs = tv(&flat, &pi);

        assert!(tempered < 0.06, "tempered transitions should reproduce the model: tv = {tempered}");
        assert!(gibbs > 0.4, "plain Gibbs should be stuck in one mode: tv = {gibbs}");
    }

    /// The same statement for the swapped ladder: its cold chain's histogram must be the exact
    /// Boltzmann distribution, and the swap rates must show the ladder is actually connected.
    ///
    /// The model is TILTED so its two modes carry different mass. On a symmetric target a swap
    /// criterion with the wrong sign is nearly invisible here — it moves states between rungs in a
    /// way the `±` symmetry hides — and this test is meant to see one.
    #[test]
    fn the_cold_end_of_a_swapped_ladder_is_a_sample_from_the_model() {
        let n = 8usize;
        let g = two_modes(n, 0.7, 0.15);
        let pi = exact_boltzmann(&g, 1.0);
        let betas = crate::tempering::geometric_ladder(0.05, 1.0, 12);
        let mut reps = Replicas::new(n, &betas, 5);
        for _ in 0..1_000 {
            reps.advance(&g, 1);
        }
        let draws = 40_000usize;
        let mut hist = vec![0.0f64; 1 << n];
        for _ in 0..draws {
            reps.advance(&g, 1);
            hist[key(reps.cold())] += 1.0 / draws as f64;
        }
        let d = tv(&hist, &pi);
        assert!(d < 0.06, "the cold chain should reproduce the model: tv = {d}");
        assert!(
            reps.swap_rates().iter().all(|&r| r > 0.05),
            "a ladder nothing crosses is not a ladder: {:?}",
            reps.swap_rates()
        );
    }

    /// A small well-formed dataset for the tests that check plumbing rather than quality: `n` rows
    /// near all-up and `n` near all-down. The quality tests use [`draws_from`] instead, so that the
    /// data comes from a model whose modes are known.
    fn bimodal(n: usize) -> Dataset {
        let mut rows = Vec::new();
        for flip in 0..n {
            for sign in [1i8, -1] {
                let mut row = vec![sign; n];
                row[flip] = -sign;
                rows.push(row);
            }
        }
        Dataset { visible: n, rows }
    }

    /// A malformed ladder is refused before anything is fitted, and each fault is named.
    #[test]
    fn a_ladder_that_cannot_carry_a_negative_phase_is_refused() {
        let n = 4;
        let data = bimodal(n);
        let structure = two_modes(n, 0.0, 0.0);
        let p = Params { epochs: 1, ..Params::default() };
        let bad = |betas: Vec<f64>| {
            train_tempered(
                &structure,
                &data,
                &p,
                &Ladder { betas, negative: Negative::ParallelTempering },
                0,
            )
            .unwrap_err()
        };
        assert_eq!(bad(vec![1.0]), Error::TooShort { rungs: 1 });
        assert_eq!(bad(vec![0.5, 0.5, 1.0]), Error::NotIncreasing { at: 0, betas: (0.5, 0.5) });
        assert_eq!(bad(vec![0.5, 0.8]), Error::NotAnchored { top: 0.8 });
        assert!(matches!(
            train_tempered(
                &structure,
                &Dataset { visible: n, rows: Vec::new() },
                &p,
                &Ladder::geometric(0.2, 4, Negative::ParallelTempering),
                0,
            ),
            Err(Error::Fit(ebm::Error::NoData))
        ));
        assert!(bad(vec![0.5, 0.8]).to_string().contains("1.0"));
        assert!(bad(vec![1.0]).to_string().contains("two rungs"));
    }

    /// Exact draws from `g` by inverse CDF over its enumerated Boltzmann distribution — so the
    /// data is the model's own, with no sampler standing between them.
    fn draws_from(g: &Graph, rows: usize, seed: u64) -> Dataset {
        let mut cdf = exact_boltzmann(g, 1.0);
        for i in 1..cdf.len() {
            cdf[i] += cdf[i - 1];
        }
        let mut rng = Pcg::new(seed, 0xDA7A);
        let out = (0..rows)
            .map(|_| {
                let u = rng.f64();
                let mask = cdf.partition_point(|&c| c < u).min(cdf.len() - 1);
                (0..g.n).map(|i| if mask >> i & 1 == 1 { 1i8 } else { -1 }).collect()
            })
            .collect();
        Dataset { visible: g.n, rows: out }
    }

    /// The fixture the two training tests share: the tilted model, 200 exact draws from it, the
    /// untrained likelihood and the exact maximum-likelihood ceiling.
    fn ceiling_fixture() -> (Graph, Dataset, f64, f64) {
        let n = 10;
        let data = draws_from(&two_modes(n, 0.35, 0.05), 200, 9);
        let structure = two_modes(n, 0.0, 0.0);
        let untrained = ebm::exact_log_likelihood(&structure, &data).unwrap();
        // The ceiling has plateaued here: 1000, 2000, 4000 and 8000 epochs give -0.7395, -0.7361,
        // -0.7345 and -0.7336, so the last 6000 epochs are worth 0.003 nats and the fractions
        // below are not sensitive to where in that range it is cut off.
        let ceiling =
            ebm::train_exact(&structure, &data, &ebm::FitParams { epochs: 2_000, lr: 0.1, l2: 0.0 })
                .unwrap()
                .log_likelihood
                .unwrap();
        assert!(ceiling > untrained + 5.0, "the ceiling must be worth reaching: {ceiling}");
        (structure, data, untrained, ceiling)
    }

    /// THE HEADLINE, AND IT IS SCORED AGAINST THE EXACT CEILING.
    ///
    /// `train_exact` ascends the true likelihood with both averages enumerated, so it is what is
    /// reachable on this structure and this data. Every trainer is then scored by
    /// `ebm::exact_log_likelihood` as a fraction of the range between an untrained model and that
    /// ceiling — no sampler is compared with another sampler anywhere in this test.
    ///
    /// PCD is given the SAME sweep budget per update as the ladder (`k = rungs x k_tempered`), so
    /// the comparison is not about how much work each does.
    ///
    /// # What was measured, including where PCD does not lose
    ///
    /// Mean over three seeds, 200 epochs, twelve rungs from `beta = 0.05`:
    ///
    /// ```text
    ///   learning rate      0.1     0.05    0.02    0.01    0.005
    ///   parallel temp     0.943   0.998   0.998   0.996   0.996
    ///   tempered trans    0.998   0.991   0.998   0.997   0.996
    ///   PCD              -3.642  -0.111   0.938   0.971   0.992
    /// ```
    ///
    /// **PCD is not merely slower here, it is unusable above a step it cannot see.** A negative
    /// fraction is a fit WORSE than the untrained model: the fantasy chains freeze in one mode and
    /// the bias runs away from them. The tempered phases stay within half a percent of the ceiling
    /// across the whole decade. **But at PCD's own best step the gap is 0.4 points**, so what a
    /// ladder buys on a model this small is robustness to the step size rather than a better
    /// optimum, and this test asserts at `0.05`, where the difference is a collapse rather than a
    /// fraction of a percent.
    #[test]
    fn a_tempered_negative_phase_reaches_a_ceiling_pcd_cannot() {
        let (structure, data, untrained, ceiling) = ceiling_fixture();
        let frac = |ll: f64| (ll - untrained) / (ceiling - untrained);

        let rungs = 12;
        let base = Params {
            epochs: 200,
            k: 2,
            positive_sweeps: 1,
            learning_rate: 0.05,
            batch: 8,
            persistent: true,
        };
        let ladder = Ladder::geometric(0.05, rungs, Negative::ParallelTempering);
        let excursion =
            Ladder { betas: ladder.betas.clone(), negative: Negative::TemperedTransitions };
        let pcd = Params { k: base.k * rungs, ..base };

        for seed in 0..3u64 {
            let a = train_tempered(&structure, &data, &base, &ladder, seed).unwrap();
            let b = train_tempered(&structure, &data, &base, &excursion, seed).unwrap();
            let c = ebm::train(&structure, &data, &pcd, seed).unwrap();

            let (fa, fb, fc) = (
                frac(a.fit.log_likelihood.unwrap()),
                frac(b.fit.log_likelihood.unwrap()),
                frac(c.log_likelihood.unwrap()),
            );
            assert!(fa > 0.9, "seed {seed}: swapped ladder reached {fa} of the ceiling");
            assert!(fb > 0.9, "seed {seed}: tempered transitions reached {fb} of the ceiling");
            assert!(fc < 0.5, "seed {seed}: PCD reached {fc} -- it is meant to collapse here");
            assert!(
                a.acceptance.iter().all(|&r| r > 0.05),
                "seed {seed}: swap rates {:?}",
                a.acceptance
            );
            assert!(b.acceptance[0] > 0.05, "seed {seed}: excursion acceptance {:?}", b.acceptance);
        }
    }

    /// IT IS THE LADDER'S REACH THAT DOES THE WORK, and its acceptance rate does not say so.
    ///
    /// The same trainer, the same budget, the same seed — only the hot end moves. As `beta_min`
    /// rises the ladder stops reaching a temperature at which the barrier is crossable and the fit
    /// falls back toward PCD's, which is what rules out the alternative reading that this module's
    /// plumbing rather than its tempering is what helped:
    ///
    /// ```text
    ///   beta_min          0.05    0.3     0.6     0.8     0.95    (PCD, no ladder)
    ///   parallel temp    0.999   1.000   0.980   0.921   0.405      -0.111
    ///   tempered trans   0.991   0.997   0.576  -2.199   0.243
    ///   swap acceptance   0.83    0.93    0.98    0.98    0.98
    /// ```
    ///
    /// **The swap rate goes UP as the ladder gets worse**, which is the diagnostic trap worth
    /// naming: adjacent rungs at nearly the same temperature always swap, and a ladder whose rungs
    /// all agree with each other is a ladder that goes nowhere. A healthy rate is necessary and not
    /// sufficient; what it cannot tell you is whether the hot end is hot.
    #[test]
    fn a_ladder_that_never_gets_hot_fails_the_way_pcd_does() {
        let (structure, data, untrained, ceiling) = ceiling_fixture();
        let frac = |ll: f64| (ll - untrained) / (ceiling - untrained);
        let base = Params {
            epochs: 200,
            k: 2,
            positive_sweeps: 1,
            learning_rate: 0.05,
            batch: 8,
            persistent: true,
        };
        let run = |beta_min: f64, negative: Negative| {
            train_tempered(&structure, &data, &base, &Ladder::geometric(beta_min, 12, negative), 0)
                .unwrap()
        };
        for negative in [Negative::ParallelTempering, Negative::TemperedTransitions] {
            let deep = run(0.05, negative);
            let shallow = run(0.95, negative);
            let (fd, fs) =
                (frac(deep.fit.log_likelihood.unwrap()), frac(shallow.fit.log_likelihood.unwrap()));
            assert!(fd > 0.9, "{negative:?}: a ladder reaching beta 0.05 got {fd}");
            assert!(fs < 0.6, "{negative:?}: a ladder stopping at beta 0.95 got {fs}");
            assert!(
                shallow.acceptance.iter().all(|&r| r > 0.9),
                "{negative:?}: the failed ladder still accepted {:?}",
                shallow.acceptance
            );
        }
    }

    /// A tempered fit is reproducible from its seed, on both negative phases.
    #[test]
    fn the_same_seed_gives_the_same_model() {
        let n = 6;
        let data = bimodal(n);
        let structure = two_modes(n, 0.0, 0.0);
        let p = Params { epochs: 20, k: 1, batch: 4, ..Params::default() };
        for negative in [Negative::ParallelTempering, Negative::TemperedTransitions] {
            let ladder = Ladder::geometric(0.1, 5, negative);
            let a = train_tempered(&structure, &data, &p, &ladder, 3).unwrap();
            let b = train_tempered(&structure, &data, &p, &ladder, 3).unwrap();
            assert_eq!(a.fit.graph.w, b.fit.graph.w, "{negative:?}");
            assert_eq!(a.fit.graph.h, b.fit.graph.h, "{negative:?}");
            assert_eq!(a.acceptance, b.acceptance, "{negative:?}");
        }
    }
}
