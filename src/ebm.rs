//! Fitting an energy-based model to data — seven methods, and the exact likelihood and the exact
//! gradient to judge all seven by.
//!
//! Everything else in this crate takes a model as given and samples it, optimises it, or bounds it.
//! This is the one module that produces a model, and it exists because the field's central open
//! problem cannot be measured without one. The **mixing-expressivity tradeoff** is the claim that an
//! EBM's mixing time rises with its expressivity; expressivity is a property of a model that has
//! been FITTED TO DATA, so a stack that cannot fit one can only ever measure the structural half.
//!
//! # The gradient, and why it is two averages
//!
//! With `E(s) = −Σ h_i s_i − Σ J_ij s_i s_j` and `p(s) ∝ exp(−E(s))`, the gradient of the average
//! log-likelihood of a dataset is a difference of two correlations:
//!
//! ```text
//!   ∂ log L / ∂ J_ij  =  ⟨s_i s_j⟩_data  −  ⟨s_i s_j⟩_model
//!   ∂ log L / ∂ h_i   =  ⟨s_i⟩_data      −  ⟨s_i⟩_model
//! ```
//!
//! The first term is cheap: clamp the visible units to a data row and sample the rest. The second
//! is the whole difficulty of the field in one expression — it is an average over the model's own
//! distribution, which is exactly what is hard to sample. Contrastive divergence (Hinton 2002)
//! replaces it with `k` sweeps started from the data rather than from equilibrium, which is biased
//! and known to be biased, and is what everyone does.
//!
//! **At a fixed point the two averages are equal.** That is not an approximation and it is what
//! `a_fully_visible_fit_matches_the_data_correlations` checks: train a fully-visible model and its pairwise correlations must match the
//! data's, measured by exhaustive enumeration rather than by more sampling.
//!
//! # The seven methods, and the three ways they fail
//!
//! ```text
//!   train                  CD-k / PCD-k    the right objective, computed with a biased sampler
//!   train_free_energy      TAP / Bethe     the right objective, computed with a solved one
//!   train_pseudolikelihood                 a different objective, computed exactly
//!   train_mpf                              ... and another
//!   train_ratio_matching                   ... and another
//!   train_exact            enumeration     the right objective, computed exactly. Not runnable.
//! ```
//!
//! The split is the whole design. [`train`] targets the likelihood and cannot compute it, so its
//! error is a SAMPLING error and shrinks with `k` and with mixing. [`train_free_energy`] targets
//! the same likelihood and replaces the sampler with a fixed point, so its error is an
//! APPROXIMATION error: it does not shrink with anything, it is a fixed property of the free
//! energy chosen, and it is the same every run because nothing random happens. The three in the
//! middle compute their objective exactly and it is not the likelihood, so their error is an
//! OBJECTIVE error and shrinks with data instead — each is consistent, so each maximiser goes to
//! the truth as rows grow, which is measured in
//! `every_sampler_free_method_converges_on_the_model_that_generated_the_data`.
//!
//! Those three are also one method wearing three hats: all are a sum over (row, site) of a loss on
//! the **flip margin** `u_i = s_i f_i`, sharing a gradient, a loop and a cost. [`FlipLoss`] is
//! that shared core and its docs carry the table of what differs.
//!
//! # Judging it
//!
//! [`exact_log_likelihood`] enumerates. Every claim about expressivity in this crate is measured
//! against the true likelihood on models small enough to compute it, never against a bound, an ELBO
//! or a reconstruction error — because the tradeoff being measured is a claim about the true
//! distribution, and a proxy for it would put the proxy's own failure mode inside the result.
//!
//! A likelihood alone says what a fit ACHIEVED and not what there was to achieve. [`train_exact`]
//! ascends the gradient at the top of this page with both averages enumerated, so it is the
//! CEILING for a structure on a dataset, and every other method becomes a fraction of a reachable
//! target rather than a row in a league table against the others.
//! `examples/estimator_shootout.rs` runs all six against it.

use crate::gibbs::Sampler;
use crate::graph::{Graph, GraphBuilder};
use crate::rng::Pcg;

/// Rows of `±1`, the first `visible` entries of each being the observed part.
#[derive(Clone, Debug)]
pub struct Dataset {
    /// How many leading spins of a state are observed. The rest are latent.
    pub visible: usize,
    /// One row per example, each a full spin assignment over the visible units.
    pub rows: Vec<Vec<i8>>,
}

/// Why a fit was refused.
#[derive(Clone, Debug, PartialEq)]
pub enum Error {
    /// The dataset is empty, so there is nothing to fit.
    NoData,
    /// Pseudolikelihood was asked to fit a model with latent units.
    ///
    /// Its objective is a product of conditionals `P(s_i | rest)`, and "the rest" has to be
    /// observed. A hidden unit is not, so the conditional is not computable from the data and the
    /// method does not apply — it is not a matter of being slower or looser. Use [`train`], whose
    /// negative phase samples the latent units instead.
    HasLatent {
        /// Spins in the model.
        spins: usize,
        /// Spins the data observes.
        visible: usize,
    },
    /// A row is not `visible` long, so it cannot be clamped onto the model.
    RowWidth {
        /// Index of the offending row.
        row: usize,
        /// Its length.
        len: usize,
        /// The length every row must have.
        want: usize,
    },
    /// A row holds something other than `-1` or `+1`.
    NotASpin {
        /// Index of the offending row.
        row: usize,
        /// Position within it.
        at: usize,
        /// The value found, which is neither `+1` nor `-1`.
        value: i8,
    },
    /// The model has fewer spins than the data has visible units.
    TooSmall {
        /// Spins the model has.
        spins: usize,
        /// Visible units the data needs, which is more.
        visible: usize,
    },
    /// The model has more spins than [`MAX_ENUMERATED`], so its exact likelihood cannot be taken.
    ///
    /// **This used to be reported as [`Error::TooSmall`]**, whose message reads "the model has 24
    /// spins and the data needs 16 visible" -- true, irrelevant, and the exact opposite of what
    /// went wrong. It never named the limit and never said the model was too large. Fitting to 4x4
    /// data therefore lost its only quality metric somewhere past six hidden units, silently,
    /// because [`train`] takes the likelihood with `.ok()` and a mislabelled error looks the same
    /// as an absent one.
    TooLarge {
        /// Spins the model has.
        spins: usize,
        /// The largest that can be enumerated exactly.
        limit: usize,
    },
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Error::HasLatent { spins, visible } => write!(
                f,
                "pseudolikelihood needs every spin observed and this model has {spins} with only \
                 {visible} in the data; its objective conditions each spin on all the others, which \
                 a latent unit does not supply. `train` samples them instead"
            ),
            Error::NoData => write!(f, "no data rows; there is nothing to fit"),
            Error::RowWidth { row, len, want } => {
                write!(f, "row {row} has {len} visible entries, and the dataset declares {want}")
            }
            Error::NotASpin { row, at, value } => {
                write!(f, "row {row} position {at} is {value}, and a spin is -1 or +1")
            }
            Error::TooLarge { spins, limit } => write!(
                f,
                "this model has {spins} spins and the exact likelihood enumerates every state, \
                 which is refused above {limit}. It is refused rather than estimated because a \
                 likelihood is what expressivity is JUDGED by here, and an estimate is worst \
                 exactly where sampling is worst. Fit fewer hidden units, or score the model by \
                 something other than the exact likelihood."
            ),
            Error::TooSmall { spins, visible } => {
                write!(f, "the model has {spins} spins and the data needs {visible} visible")
            }
        }
    }
}

/// How the fit is run.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Params {
    /// Passes over the dataset.
    pub epochs: usize,
    /// Use **persistent** contrastive divergence (Tieleman 2008): keep a set of fantasy chains
    /// alive across the whole run and take the negative phase from them, instead of restarting it
    /// at the data every update.
    ///
    /// CD-k's model average is taken `k` sweeps from a data point, which is why it is biased — the
    /// chain never gets far enough to see the model's own modes. A persistent chain is never reset,
    /// so it keeps mixing while the parameters move slowly beneath it, and its average approaches
    /// the model's rather than the data's. It costs nothing extra per update; the same `k` sweeps
    /// are run, just from a different starting point.
    ///
    /// # What it is worth here, measured
    ///
    /// The textbook expectation is that PCD beats CD because it is less biased. On models small
    /// enough for [`exact_log_likelihood`] to judge the result, **it does not** — and the exact
    /// likelihood is what makes that sayable rather than guessable. On a 9-visible, 6-hidden RBM
    /// over bars-and-stripes, 600 epochs, mean exact log-likelihood over six seeds:
    ///
    /// ```text
    ///   k     lr      CD       PCD     PCD wins
    ///   1    0.050  −3.003   −4.975      0/6
    ///   1    0.020  −3.322   −3.290      2/6
    ///   5    0.050  −2.843   −3.192      0/6
    ///   5    0.020  −3.168   −3.129      3/6
    ///  20    0.050  −2.808   −2.863      1/6
    ///  20    0.020  −3.129   −3.084      3/6
    /// ```
    ///
    /// At a large step PCD is clearly WORSE, which is the documented failure mode made concrete:
    /// the fantasy chains cannot track a model that moves faster than they mix. At a small step the
    /// two are tied to within seed noise. There is no regime on this model where PCD wins, because
    /// CD's bias is not what limits a fifteen-spin machine — the same shape as this crate's
    /// perceptron finding, where the frozen-landscape gap is real but only asymptotically.
    ///
    /// It ships because it is the standard method and a caller training something larger will want
    /// it; it ships with this table because a method that is better in the literature and not here
    /// should say so.
    ///
    /// `false` keeps plain CD-k, which is what this module shipped first and what the existing
    /// tests pin.
    pub persistent: bool,
    /// `k` in CD-k: negative-phase sweeps started from the positive-phase state.
    ///
    /// One is Hinton's original and is the biased extreme; larger is closer to the true gradient
    /// and costs proportionally. The bias is the reason this module reports the EXACT likelihood
    /// rather than trusting the training loss.
    pub k: usize,
    /// Sweeps used to settle the latent units in the positive phase, with the visible clamped.
    pub positive_sweeps: usize,
    /// The starting step. It DECAYS LINEARLY to a tenth of this over the epochs, and that decay is
    /// not a refinement — without it the fit has a noise floor and never reaches its own fixed
    /// point. The gradient's model term is one sample per row, so the parameters random-walk around
    /// the optimum with an amplitude set by the step size; the fitted correlations then sit a
    /// constant distance from the data's however long it runs. Decaying the step is what makes
    /// `a_fully_visible_fit_matches_the_data_correlations` a test of moment matching rather
    /// than a test of the noise floor.
    pub learning_rate: f64,
    /// Rows per gradient step.
    pub batch: usize,
}

impl Default for Params {
    fn default() -> Self {
        Params { epochs: 300, k: 5, positive_sweeps: 5, learning_rate: 0.05, batch: 8, persistent: false }
    }
}

/// What the fit produced.
pub struct Trained {
    /// The fitted model. Its edge set is the structure it was given; only weights moved.
    pub graph: Graph,
    /// Mean log-likelihood per row, exact.
    ///
    /// `None` means the model has more than [`MAX_ENUMERATED`] spins, and nothing else: every other
    /// way of failing is caught before training starts. The fit still happened and the model is
    /// real; only its quality is unmeasured. Call [`exact_log_likelihood`] directly to get the
    /// reason as an [`Error::TooLarge`] naming the limit — this field swallows it, which is why the
    /// limit is written down here.
    ///
    /// Fitting to 4x4 data crosses that line at around seven hidden units, which is sooner than it
    /// looks: the ceiling counts VISIBLE PLUS HIDDEN spins, not hidden ones.
    pub log_likelihood: Option<f64>,
    /// Epochs completed, which is the cap unless training stopped early.
    pub epochs_run: usize,
}

impl core::fmt::Debug for Trained {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Trained")
            .field("spins", &self.graph.n)
            .field("edges", &self.graph.n_edges)
            .field("log_likelihood", &self.log_likelihood)
            .field("epochs_run", &self.epochs_run)
            .finish()
    }
}

/// Fit `structure`'s weights to `data` by contrastive divergence.
///
/// `structure` supplies the EDGE SET and nothing else: its weights are the starting point and are
/// overwritten. Biases on latent units are fitted too.
///
/// # Errors
///
/// [`Error::RowWidth`] or [`Error::NotASpin`] for malformed data, and [`Error::TooSmall`] when the
/// model has fewer spins than the data has visible units.
pub fn train(structure: &Graph, data: &Dataset, p: &Params, seed: u64) -> Result<Trained, Error> {
    check(structure, data)?;
    let n = structure.n;
    let mut rng = Pcg::new(seed, 0x00EB_3600);

    // Edges once, as (i, j, weight). Working from a list rather than from the CSR keeps the update
    // in one place; the graph is rebuilt from it at the end.
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

    // The fantasy chains for PCD. One per batch slot, started from random spins and never reset --
    // that is the whole idea. Unused when `persistent` is false.
    // Each chain carries its own RNG STATE, not just its spins. Rebuilding a sampler from a fixed
    // seed every update would replay the same random stream forever -- the chain would move, but
    // always the same way, which is not a persistent chain at all. It is the kind of bug that shows
    // up as a method quietly underperforming rather than as a failure.
    let mut fantasy: Vec<(Vec<i8>, Pcg)> = if p.persistent {
        (0..p.batch.max(1))
            .map(|s| {
                (
                    (0..n).map(|_| if rng.f64() < 0.5 { -1i8 } else { 1 }).collect(),
                    Pcg::new(seed ^ 0xF0F0_0000 ^ s as u64, 0x9E37),
                )
            })
            .collect()
    } else {
        Vec::new()
    };

    let mut order: Vec<usize> = (0..data.rows.len()).collect();
    for epoch in 0..p.epochs {
        // Shuffle, so a batch is not the same slice of the data every epoch.
        for i in (1..order.len()).rev() {
            let j = (rng.f64() * (i + 1) as f64) as usize % (i + 1);
            order.swap(i, j);
        }
        let g = build(&edges, &bias);
        let decay = if p.epochs > 1 {
            1.0 - 0.9 * epoch as f64 / (p.epochs - 1) as f64
        } else {
            1.0
        };

        for chunk in order.chunks(p.batch.max(1)) {
            let mut d_edge = vec![0.0f64; edges.len()];
            let mut d_bias = vec![0.0f64; n];


            for (slot_in_batch, &r) in chunk.iter().enumerate() {
                let row = &data.rows[r];

                // POSITIVE PHASE. Visible clamped to the data, latent settled around it.
                let seed = (rng.next_u32() as u64) << 32 | rng.next_u32() as u64;
                let mut smp = Sampler::new(&g, 1.0, seed);
                for (i, &v) in row.iter().enumerate() {
                    smp.clamp(i, v);
                }
                smp.sweeps(p.positive_sweeps.max(1), None);
                let pos = smp.s.clone();

                // NEGATIVE PHASE. Plain CD runs the same chain on, unclamped, from the data --
                // which is what makes it CONTRASTIVE DIVERGENCE and not maximum likelihood: the
                // model average is taken near the data rather than at equilibrium, and it is
                // biased for exactly that reason. PCD instead advances a chain that was never
                // reset, so the average it reports is the model's.
                let neg: Vec<i8> = if p.persistent {
                    let slot = slot_in_batch % fantasy.len();
                    let mut fs = Sampler::new(&g, 1.0, 0);
                    fs.s.copy_from_slice(&fantasy[slot].0);
                    fs.rng = fantasy[slot].1.clone();
                    fs.sweeps(p.k.max(1), None);
                    fantasy[slot].0.copy_from_slice(&fs.s);
                    fantasy[slot].1 = fs.rng.clone();
                    fs.s
                } else {
                    for i in 0..data.visible {
                        smp.unclamp(i);
                    }
                    smp.sweeps(p.k.max(1), None);
                    smp.s.clone()
                };
                for (e, &(i, j, _)) in edges.iter().enumerate() {
                    d_edge[e] += (pos[i] * pos[j]) as f64 - (neg[i] * neg[j]) as f64;
                }
                for i in 0..n {
                    d_bias[i] += pos[i] as f64 - neg[i] as f64;
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

    let graph = build(&edges, &bias);
    let log_likelihood = exact_log_likelihood(&graph, data).ok();
    Ok(Trained { graph, log_likelihood, epochs_run: p.epochs })
}

/// How many spins [`exact_log_likelihood`] will enumerate before refusing.
pub const MAX_ENUMERATED: usize = 22;

/// Mean log-likelihood per data row, by enumeration.
///
/// `log p(v) = log Σ_h exp(−E(v, h)) − log Z`, both sums taken over every state. Exhaustive, so
/// there is nothing to be wrong about beyond the model itself — which is the point. A tradeoff
/// between mixing and expressivity measured with an APPROXIMATE likelihood would carry the
/// approximation's failure mode inside the result, and that failure mode is worst exactly where
/// mixing is worst.
///
/// Refuses above [`MAX_ENUMERATED`] spins rather than returning something cheaper.
///
/// # Errors
///
/// As `train`, plus [`Error::TooLarge`] past the size that can be enumerated exactly -- this walks
/// all `2^n` states and refuses rather than taking a very long time.
pub fn exact_log_likelihood(g: &Graph, data: &Dataset) -> Result<f64, Error> {
    check(g, data)?;
    if g.n > MAX_ENUMERATED {
        return Err(Error::TooLarge { spins: g.n, limit: MAX_ENUMERATED });
    }
    // log-sum-exp over every state, and over the states agreeing with each row on the visible part.
    let mut max_neg_e = f64::NEG_INFINITY;
    let states = 1usize << g.n;
    let mut energies = Vec::with_capacity(states);
    let mut s = vec![-1i8; g.n];
    for mask in 0..states {
        for i in 0..g.n {
            s[i] = if mask >> i & 1 == 1 { 1 } else { -1 };
        }
        let e = -g.energy(&s);
        max_neg_e = max_neg_e.max(e);
        energies.push(e);
    }
    let z: f64 = energies.iter().map(|e| (e - max_neg_e).exp()).sum();
    let log_z = max_neg_e + z.ln();

    // The visible units are indices 0..visible, so the LOW BITS OF THE MASK ARE THE VISIBLE
    // PATTERN. One pass over the states therefore fills every row's numerator at once, instead of
    // re-scanning all 2^n states once per row.
    let vmask = (1usize << data.visible) - 1;
    let mut per_visible = vec![0.0f64; 1usize << data.visible];
    for (mask, &e) in energies.iter().enumerate() {
        per_visible[mask & vmask] += (e - max_neg_e).exp();
    }

    let mut total = 0.0;
    for row in &data.rows {
        let mut key = 0usize;
        for (i, &v) in row.iter().enumerate() {
            if v == 1 {
                key |= 1 << i;
            }
        }
        total += max_neg_e + per_visible[key].ln() - log_z;
    }
    Ok(total / data.rows.len() as f64)
}

/// A likelihood past enumeration: the numerator exact over the hidden units, `ln Z` by AIS.
#[derive(Clone, Debug)]
pub struct AisLikelihood {
    /// Mean log-likelihood per row, with `ln Z` at its AIS point estimate.
    pub estimate: f64,
    /// Mean over rows of the exact `ln Σ_h exp(−E(v, h))`.
    pub mean_log_numerator: f64,
    /// The `ln Z` run, with its own bound and effective sample size.
    pub log_z: crate::free_energy::Ais,
    /// Whether the numerator was enumerated (true) or estimated by clamped AIS (false).
    pub numerator_exact: bool,
}

impl AisLikelihood {
    /// The likelihood is at most this with probability at least `1 − delta`: the exact numerator
    /// minus the unconditional lower bound on `ln Z`. `None` when the numerator was itself
    /// estimated — a lower-bounded numerator over a lower-bounded `ln Z` bounds nothing, and an
    /// upper bound on the numerator is reverse AIS's conditional business.
    #[must_use]
    pub fn upper_bound(&self, delta: f64) -> Option<f64> {
        self.numerator_exact.then(|| self.mean_log_numerator - self.log_z.lower_bound(delta))
    }
}

/// Mean log-likelihood per row for a model too large to enumerate, when its HIDDEN part is not.
///
/// `log p(v) = ln Σ_h exp(−E(v, h)) − ln Z`. The first term enumerates the `2^hidden` completions
/// of each row exactly; the second is [`crate::free_energy::ais`] on the whole model, whose lower
/// bound is unconditional and therefore gives [`AisLikelihood::upper_bound`] the same standing.
/// Refuses when `hidden > MAX_ENUMERATED`; a clamped AIS for the numerator is the recorded next
/// step past that.
///
/// # Errors
///
/// As `train`. Unlike `exact_log_likelihood` there is no size limit, because AIS estimates rather
/// than enumerates.
pub fn log_likelihood_ais(
    g: &Graph,
    data: &Dataset,
    ladder: &[f64],
    sweeps: usize,
    runs: usize,
    seed: u64,
) -> Result<AisLikelihood, Error> {
    check(g, data)?;
    let hidden = g.n - data.visible;
    if hidden > MAX_ENUMERATED {
        return Err(Error::TooLarge { spins: hidden, limit: MAX_ENUMERATED });
    }
    let mut s = vec![-1i8; g.n];
    let mut total = 0.0;
    for row in &data.rows {
        s[..data.visible].copy_from_slice(row);
        let mut logs = Vec::with_capacity(1usize << hidden);
        for mask in 0..(1usize << hidden) {
            for b in 0..hidden {
                s[data.visible + b] = if mask >> b & 1 == 1 { 1 } else { -1 };
            }
            logs.push(-g.energy(&s));
        }
        let mx = logs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        total += mx + logs.iter().map(|x| (x - mx).exp()).sum::<f64>().ln();
    }
    let mean_log_numerator = total / data.rows.len() as f64;
    let log_z = crate::free_energy::ais(g, ladder, sweeps, runs, seed);
    Ok(AisLikelihood { estimate: mean_log_numerator - log_z.log_z, mean_log_numerator, log_z, numerator_exact: true })
}

/// [`log_likelihood_ais`] for a hidden part too large to enumerate: the numerator of every row by
/// [`crate::free_energy::ais_clamped`] with the visible units held at the row.
///
/// One AIS per row, so the cost is `rows × runs × rungs × sweeps` sweeps over the hidden units.
/// The result carries a point estimate and no bound — see [`AisLikelihood::upper_bound`].
///
/// # Errors
///
/// As `log_likelihood_ais`.
pub fn log_likelihood_ais_clamped(
    g: &Graph,
    data: &Dataset,
    ladder: &[f64],
    sweeps: usize,
    runs: usize,
    seed: u64,
) -> Result<AisLikelihood, Error> {
    check(g, data)?;
    let mut total = 0.0;
    for (r, row) in data.rows.iter().enumerate() {
        let fixed: Vec<(usize, i8)> = row.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        total += crate::free_energy::ais_clamped(g, &fixed, ladder, sweeps, runs, seed.wrapping_add(1 + r as u64)).log_z;
    }
    let mean_log_numerator = total / data.rows.len() as f64;
    let log_z = crate::free_energy::ais(g, ladder, sweeps, runs, seed);
    Ok(AisLikelihood { estimate: mean_log_numerator - log_z.log_z, mean_log_numerator, log_z, numerator_exact: false })
}

fn check(g: &Graph, data: &Dataset) -> Result<(), Error> {
    if data.rows.is_empty() {
        return Err(Error::NoData);
    }
    if g.n < data.visible {
        return Err(Error::TooSmall { spins: g.n, visible: data.visible });
    }
    for (r, row) in data.rows.iter().enumerate() {
        if row.len() != data.visible {
            return Err(Error::RowWidth { row: r, len: row.len(), want: data.visible });
        }
        if let Some(at) = row.iter().position(|&v| v != 1 && v != -1) {
            return Err(Error::NotASpin { row: r, at, value: row[at] });
        }
    }
    Ok(())
}

/// A restricted Boltzmann machine's edge set: `visible` × `hidden`, complete bipartite, no weights.
#[must_use]
pub fn rbm(visible: usize, hidden: usize) -> Graph {
    let mut gb = GraphBuilder::new(visible + hidden);
    for v in 0..visible {
        for h in 0..hidden {
            gb.couple(v, visible + h, 0.0);
        }
    }
    gb.build()
}

/// A deep Boltzmann machine's edge set: `visible` then each layer of `hidden`, chained.
///
/// Latent units here are added WITHOUT scaling each unit's connectivity, which is the arrangement
/// the field's tradeoff claim is about: "increasing latent variables increases the depth of the
/// Boltzmann machine, making sampling more difficult". [`rbm`] with the same latent count is the
/// control, since there every added unit also touches every visible one.
#[must_use]
pub fn dbm(visible: usize, hidden: &[usize]) -> Graph {
    let n = visible + hidden.iter().sum::<usize>();
    let mut gb = GraphBuilder::new(n);
    let mut below = (0..visible).collect::<Vec<_>>();
    let mut next = visible;
    for &w in hidden {
        let layer: Vec<usize> = (next..next + w).collect();
        for &a in &below {
            for &b in &layer {
                gb.couple(a, b, 0.0);
            }
        }
        next += w;
        below = layer;
    }
    gb.build()
}

/// The 3×3 bars-and-stripes dataset: every all-bars and all-stripes image, deduplicated.
///
/// The standard tiny benchmark for fitting an EBM, chosen here because at nine visible units the
/// exact likelihood and the exact partition function are both computable, so expressivity is
/// measured rather than estimated.
#[must_use]
pub fn bars_and_stripes(side: usize) -> Dataset {
    let n = side * side;
    let mut seen: Vec<Vec<i8>> = Vec::new();
    for mask in 0..(1usize << side) {
        for stripes in [false, true] {
            let mut row = vec![-1i8; n];
            for a in 0..side {
                if mask >> a & 1 == 1 {
                    for b in 0..side {
                        row[if stripes { a * side + b } else { b * side + a }] = 1;
                    }
                }
            }
            if !seen.contains(&row) {
                seen.push(row);
            }
        }
    }
    Dataset { visible: n, rows: seen }
}

// ---- the Jarzynski-reweighted negative phase --------------------------------------------------

/// How a Jarzynski-reweighted fit is run.
///
/// The negative phase is ONE ensemble carried through the whole run rather than `k` sweeps per
/// row, so `walkers` and `k` price it together: `walkers × k` sweeps per parameter update.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct JarzynskiParams {
    /// Passes over the dataset.
    pub epochs: usize,
    /// Walkers carried across the whole run, each with its own log-weight.
    pub walkers: usize,
    /// Palindromic sweeps applied to every walker after each update. Zero is legal and turns the
    /// run into annealed importance sampling along the training path.
    pub k: usize,
    /// Sweeps at the starting parameters before the first update. The weights survive them
    /// because a sweep at those parameters leaves their target invariant.
    pub burn_in: usize,
    /// Sweeps settling the latent units in the positive phase, visible clamped. Unused when the
    /// model is fully visible, where the positive phase is the data row itself.
    pub positive_sweeps: usize,
    /// The starting step, decaying linearly to a tenth over the epochs as [`Params::learning_rate`].
    pub learning_rate: f64,
    /// Rows per gradient step.
    pub batch: usize,
    /// Resample when the weights' effective sample size falls below this fraction of `walkers`.
    /// Zero never resamples.
    pub resample_at: f64,
}

impl Default for JarzynskiParams {
    fn default() -> Self {
        JarzynskiParams {
            epochs: 300,
            walkers: 64,
            k: 1,
            burn_in: 20,
            positive_sweeps: 5,
            learning_rate: 0.05,
            batch: 8,
            resample_at: 0.5,
        }
    }
}

/// What a Jarzynski-reweighted fit produced: the model, and what its walker weights were worth.
#[derive(Debug)]
pub struct JarzynskiFit {
    /// The fit, scored exactly as every other trainer in this module scores one.
    pub trained: Trained,
    /// Effective sample size `(Σw)²/Σw²` of the weights at the end, in walkers.
    pub ess: f64,
    /// The smallest effective sample size any negative-phase estimate rested on.
    pub min_ess: f64,
    /// Mean effective sample size across the updates.
    pub mean_ess: f64,
    /// How many times the ensemble was resampled.
    pub resamples: usize,
    /// `ln Z` of the fitted model by Jarzynski's equality: `n ln 2 + Σ ln mean w` over the run.
    /// Consistent, not unbiased — see [`train_jarzynski`] for the measured one-over-walkers term.
    pub log_z: f64,
    /// Parameter updates performed.
    pub updates: usize,
}

/// Effective sample size in walkers, and `ln mean w`, from a set of log-weights.
fn weight_summary(lw: &[f64]) -> (f64, f64) {
    let top = lw.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let (mut s, mut s2) = (0.0f64, 0.0f64);
    for &v in lw {
        let q = (v - top).exp();
        s += q;
        s2 += q * q;
    }
    (s * s / s2, top + (s / lw.len() as f64).ln())
}

/// Fit `structure` by **Jarzynski-reweighted maximum likelihood** (Carbone, Hamdani &
/// Vanden-Eijnden 2023) — an importance weight carried alongside every walker so the negative
/// phase stays right however fast the parameters move.
///
/// # What changes, and it is only the negative phase
///
/// [`train`]'s fantasy chains are asked to be at equilibrium under a model that is moving, and they
/// are not: that lag IS contrastive divergence's bias. Here each walker also carries a log-weight,
/// and every parameter step pays that step's WORK into it,
///
/// ```text
///   ln w_a  +=  E_θ(x_a) − E_θ'(x_a)
/// ```
///
/// after which the walker moves under a kernel invariant for the new `θ'`. Jarzynski's equality
/// says the reweighted ensemble then still targets `p_θ'` exactly — the lag is not removed, it is
/// PRICED — so the weighted averages `Σ w_a f(x_a) / Σ w_a` are consistent estimates of the model
/// term at every step, mixed or not.
///
/// The walkers start uniform, which is the all-zero model, and one importance step at
/// `ln w_a = −E_θ0(x_a)` carries them to the starting parameters. So `Σ w_a / walkers` tracks
/// `Z_θ / 2^n` all the way along, and [`JarzynskiFit::log_z`] reads it off — the free energy of the
/// fitted model falls out of the training run.
///
/// # The cost, which is the effective sample size
///
/// Nothing is free: fast training makes the weights degenerate, and a degenerate ensemble is a
/// one-walker estimate wearing `walkers` labels. That is why [`JarzynskiFit`] reports the weights'
/// effective sample size — end, minimum and mean — beside the model. Below
/// `resample_at × walkers` the ensemble is resampled with replacement in proportion to its
/// weights, which duplicates walkers instead of losing them.
///
/// # What `log_z` is worth, measured
///
/// The identity is exact for a protocol fixed in advance, and a training run's is not: `θ` is
/// chosen from the SAME walkers it reweights, so a heavy walker pulls the gradient toward
/// suppressing itself. The residual against an enumerated `ln Z` is therefore a bias and not
/// noise — on the six-spin fixture in this module's tests it is −0.41, −0.10 and −0.027 nats at
/// 256, 1024 and 4096 walkers, halving as the ensemble doubles while moving about 0.01 across
/// seeds. With `learning_rate` at zero the protocol is fixed again and the same code is unbiased.
/// The effective sample size does not see this: it stayed above 89% of the walkers throughout.
///
/// # What the reweighting buys, measured
///
/// On this crate's enumerable models, NOT accuracy. Deleting the weight update entirely — leaving
/// plain persistent chains — still passes both the moment-matching and the ceiling test in this
/// module, because a chain over six spins mixes whatever the parameters are doing, and CD-1 on the
/// same fixture reaches 99.95% of the ceiling against this method's 99.97%. The bias this corrects
/// is a claim about a chain that cannot reach a model's modes, and nothing [`train_exact`] can
/// price is that model. What it buys HERE is the free-energy readout and a diagnostic that says
/// when the negative phase has stopped being worth anything, which CD has no way to report.
///
/// # Errors
///
/// As [`train`].
pub fn train_jarzynski(
    structure: &Graph,
    data: &Dataset,
    p: &JarzynskiParams,
    seed: u64,
) -> Result<JarzynskiFit, Error> {
    check(structure, data)?;
    let n = structure.n;
    let walkers = p.walkers.max(1);
    let batch = p.batch.max(1);
    let mut rng = Pcg::new(seed, 0x00EB_1A20);
    let mut wt = Weights::of(structure);
    let mut g = wt.build();

    // The reference is the uniform distribution, whose ln Z is n ln 2, and the first weight is the
    // one importance step from there to the starting parameters.
    let mut x: Vec<Vec<i8>> = Vec::with_capacity(walkers);
    let mut wrng: Vec<Pcg> = Vec::with_capacity(walkers);
    let mut lw: Vec<f64> = Vec::with_capacity(walkers);
    for w in 0..walkers {
        let s: Vec<i8> = (0..n).map(|_| if rng.f64() < 0.5 { -1i8 } else { 1 }).collect();
        lw.push(-g.energy(&s));
        x.push(s);
        // Each walker carries its own stream, as `Params::persistent`'s chains do: one shared seed
        // rebuilt per update would move every walker the same way forever.
        wrng.push(Pcg::new(seed ^ 0x0A2E_0000 ^ w as u64, 0x9E37));
    }
    for w in 0..walkers {
        for _ in 0..p.burn_in {
            crate::free_energy::palindromic_sweep(&g, 1.0, &mut x[w], &mut wrng[w]);
        }
    }

    let mut log_z_accum = 0.0f64;
    let mut resamples = 0usize;
    let mut min_ess = f64::INFINITY;
    let mut ess_sum = 0.0f64;
    let mut updates = 0usize;

    let mut order: Vec<usize> = (0..data.rows.len()).collect();
    for epoch in 0..p.epochs {
        for i in (1..order.len()).rev() {
            let j = (rng.f64() * (i + 1) as f64) as usize % (i + 1);
            order.swap(i, j);
        }
        let decay = if p.epochs > 1 {
            1.0 - 0.9 * epoch as f64 / (p.epochs - 1) as f64
        } else {
            1.0
        };

        for chunk in order.chunks(batch) {
            // NEGATIVE PHASE: the weighted ensemble, self-normalised. The effective sample size is
            // taken in the same pass, so the number reported is the one this estimate rested on.
            let top = lw.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let (mut sw, mut sw2) = (0.0f64, 0.0f64);
            let mut neg_h = vec![0.0f64; n];
            let mut neg_w = vec![0.0f64; wt.edges.len()];
            for (w, s) in x.iter().enumerate() {
                let q = (lw[w] - top).exp();
                sw += q;
                sw2 += q * q;
                for (i, acc) in neg_h.iter_mut().enumerate() {
                    *acc += q * f64::from(s[i]);
                }
                for (e, &(i, j, _)) in wt.edges.iter().enumerate() {
                    neg_w[e] += q * f64::from(s[i] * s[j]);
                }
            }
            let ess = sw * sw / sw2;
            min_ess = min_ess.min(ess);
            ess_sum += ess;
            updates += 1;
            for v in &mut neg_h {
                *v /= sw;
            }
            for v in &mut neg_w {
                *v /= sw;
            }

            // POSITIVE PHASE, exactly `train`'s: the row itself when nothing is latent, and a
            // clamped chain around it when something is.
            let mut pos_h = vec![0.0f64; n];
            let mut pos_w = vec![0.0f64; wt.edges.len()];
            for &r in chunk {
                let row = &data.rows[r];
                let pos: Vec<i8> = if n == data.visible {
                    row.clone()
                } else {
                    let s2 = (u64::from(rng.next_u32()) << 32) | u64::from(rng.next_u32());
                    let mut smp = Sampler::new(&g, 1.0, s2);
                    for (i, &v) in row.iter().enumerate() {
                        smp.clamp(i, v);
                    }
                    smp.sweeps(p.positive_sweeps.max(1), None);
                    smp.s
                };
                for (i, acc) in pos_h.iter_mut().enumerate() {
                    *acc += f64::from(pos[i]);
                }
                for (e, &(i, j, _)) in wt.edges.iter().enumerate() {
                    pos_w[e] += f64::from(pos[i] * pos[j]);
                }
            }

            let m = chunk.len() as f64;
            let step = p.learning_rate * decay;
            for (e, w) in wt.edges.iter_mut().enumerate() {
                w.2 += step * (pos_w[e] / m - neg_w[e]);
            }
            for i in 0..n {
                wt.bias[i] += step * (pos_h[i] / m - neg_h[i]);
            }

            // THE JARZYNSKI STEP: pay the work of the move the parameters just made, then move the
            // walkers under the model they now belong to.
            let g_new = wt.build();
            for (w, s) in x.iter().enumerate() {
                lw[w] += g.energy(s) - g_new.energy(s);
            }
            g = g_new;

            if p.resample_at > 0.0 {
                let (after, log_mean_w) = weight_summary(&lw);
                if after < p.resample_at * walkers as f64 {
                    // Multinomial rather than systematic: drawing the survivors independently is
                    // what keeps `mean w` an unbiased estimator of the ratio accumulated so far,
                    // and systematic resampling is not. That ratio is banked here, and the weights
                    // restart at one -- the frozen-protocol test pins that the two accumulate the
                    // same product.
                    log_z_accum += log_mean_w;
                    let top = lw.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    let mut cdf = Vec::with_capacity(walkers);
                    let mut acc = 0.0f64;
                    for &v in &lw {
                        acc += (v - top).exp();
                        cdf.push(acc);
                    }
                    let picks: Vec<usize> = (0..walkers)
                        .map(|_| {
                            let u = rng.f64() * acc;
                            cdf.partition_point(|&c| c < u).min(walkers - 1)
                        })
                        .collect();
                    x = picks.iter().map(|&i| x[i].clone()).collect();
                    lw.fill(0.0);
                    resamples += 1;
                }
            }

            for w in 0..walkers {
                for _ in 0..p.k {
                    crate::free_energy::palindromic_sweep(&g, 1.0, &mut x[w], &mut wrng[w]);
                }
            }
        }
    }

    let (ess, log_mean_w) = weight_summary(&lw);
    let graph = wt.build();
    let log_likelihood = exact_log_likelihood(&graph, data).ok();
    Ok(JarzynskiFit {
        trained: Trained { graph, log_likelihood, epochs_run: p.epochs },
        ess,
        min_ess: if updates == 0 { ess } else { min_ess },
        mean_ess: if updates == 0 { ess } else { ess_sum / updates as f64 },
        resamples,
        log_z: n as f64 * core::f64::consts::LN_2 + log_z_accum + log_mean_w,
        updates,
    })
}


#[cfg(test)]
mod likelihood_tests {
    use super::*;
    use crate::free_energy::linear_ladder;

    /// Where enumeration can still judge it, the AIS likelihood agrees and its bound holds.
    #[test]
    fn the_ais_likelihood_agrees_with_enumeration_and_is_bounded() {
        let data = bars_and_stripes(3); // 9 visible
        let g = rbm(9, 6); // 15 spins: enumerable, so the exact likelihood exists
        let exact = exact_log_likelihood(&g, &data).unwrap();
        let a = log_likelihood_ais(&g, &data, &linear_ladder(1.0, 64), 2, 128, 4).unwrap();
        assert!((a.estimate - exact).abs() < 0.1, "ais {} vs exact {exact}", a.estimate);
        let ub = a.upper_bound(1e-6).expect("enumerated numerator carries a bound");
        assert!(exact <= ub, "exact {exact} above the bound {ub}");
        assert!(a.log_z.ess > 8.0);
        // Too many hidden units is refused by the enumerating route, not approximated...
        let wide = rbm(9, 30);
        assert!(matches!(log_likelihood_ais(&wide, &data, &linear_ladder(1.0, 8), 1, 4, 1), Err(Error::TooLarge { .. })));
        // ...and the clamped route takes it, with a point estimate and no bound.
        let c = log_likelihood_ais_clamped(&wide, &data, &linear_ladder(1.0, 32), 2, 16, 2).unwrap();
        assert!(c.estimate.is_finite() && c.upper_bound(0.05).is_none());
        // Where both routes apply they agree.
        let c_small = log_likelihood_ais_clamped(&g, &data, &linear_ladder(1.0, 64), 2, 64, 3).unwrap();
        assert!((c_small.estimate - exact).abs() < 0.15, "clamped {} vs exact {exact}", c_small.estimate);
    }
}

/// How to fit by any of the four methods that need no sampler: [`train_pseudolikelihood`],
/// [`train_mpf`], [`train_ratio_matching`] and [`train_exact`].
///
/// One struct for four methods because they differ only in the gradient, never in the loop. Each
/// is plain gradient ascent on a closed-form objective — no batch, no seed, no sampler, and so
/// nothing to set that is not written here.
///
/// The defaults were tuned for pseudolikelihood. The other three ascend differently shaped
/// objectives, and a step that suits one need not suit another; `examples/estimator_shootout.rs`
/// reports what each actually wants on a model small enough to score exactly.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FitParams {
    /// Gradient ascent steps over the whole dataset.
    pub epochs: usize,
    /// Step size.
    pub lr: f64,
    /// L2 penalty on weights and biases, per step. Zero for none.
    pub l2: f64,
}

/// The name [`FitParams`] shipped under while pseudolikelihood was the only method using it.
pub type PlParams = FitParams;

impl Default for FitParams {
    fn default() -> Self {
        FitParams { epochs: 400, lr: 0.1, l2: 0.0 }
    }
}

/// The parameters of a fit as a flat list, and the graph rebuilt from it.
///
/// Every sampler-free trainer in this module walks the same two vectors and rebuilds the same way,
/// so the walk lives here once rather than four times. [`train`] keeps its own copy: it also
/// carries a batch, a shuffle and a set of fantasy chains through the loop, and folding those in
/// would put contrastive divergence's machinery in the path of three methods that have none.
#[derive(Clone)]
struct Weights {
    n: usize,
    /// `(i, j, w)` with `i < j`, one entry per undirected edge, in CSR order.
    edges: Vec<(usize, usize, f64)>,
    bias: Vec<f64>,
}

impl Weights {
    /// Read a structure's edge set and starting weights out of its CSR.
    fn of(structure: &Graph) -> Weights {
        let n = structure.n;
        let mut edges = Vec::with_capacity(structure.n_edges);
        for i in 0..n {
            for k in structure.offset[i]..structure.offset[i + 1] {
                let j = structure.nbr[k] as usize;
                if j > i {
                    edges.push((i, j, structure.w[k]));
                }
            }
        }
        Weights { n, edges, bias: structure.h.clone() }
    }

    fn build(&self) -> Graph {
        let mut gb = GraphBuilder::new(self.n);
        for &(i, j, w) in &self.edges {
            gb.couple(i, j, w);
        }
        for (i, &b) in self.bias.iter().enumerate() {
            if b != 0.0 {
                gb.bias(i, b);
            }
        }
        gb.build()
    }

    /// One ascent step along `(dw, dh)`, with the L2 pull applied to the parameter itself.
    fn step(&mut self, dw: &[f64], dh: &[f64], p: &FitParams) {
        for (e, x) in self.edges.iter_mut().enumerate() {
            x.2 += p.lr * (dw[e] - p.l2 * x.2);
        }
        for (i, b) in self.bias.iter_mut().enumerate() {
            *b += p.lr * (dh[i] - p.l2 * *b);
        }
    }
}

/// The three sampler-free objectives, reduced to the one thing that separates them.
///
/// [`pseudo_log_likelihood`], [`minimum_probability_flow`] and [`ratio_matching`] are each a sum
/// over (row, site) of a function of ONE number — the **flip margin**
///
/// ```text
///   u_i  =  s_i f_i  =  (E(s with site i flipped) − E(s)) / 2
/// ```
///
/// half the energy it costs to flip site `i` out of the state the data put it in. A large positive
/// margin is a data point the model holds down against its own neighbours. Every one of the three
/// methods is "make the data's margins large"; they differ only in the price they put on a small
/// one, and in nothing else — not the pass over the data, not the per-site residual, not the two
/// terms an edge contributes.
///
/// ```text
///   method                score per (row, site)      d(score)/du
///   pseudolikelihood      log σ(2u)                  2 σ(−2u)
///   minimum prob. flow    −exp(−u)                   exp(−u)
///   ratio matching        −σ(−2u)²                   4 q² (1 − q),   q = σ(−2u)
/// ```
///
/// Written as scores to be MAXIMISED so all three share one ascent loop. The published forms of
/// the latter two are costs to be minimised, which is the same statement with the sign moved.
///
/// **Every one of these derivatives is strictly positive.** That is not an accident of the algebra
/// to be spot-checked once: each loss is decreasing in the margin, so each pushes the data further
/// below its own flips, and a sign error in any row of that table is a method that climbs away
/// from the data. `every_flip_loss_pushes_the_margin_up` pins it.
///
/// # Why three rather than one, and a prediction that did not survive
///
/// The three losses differ out in the negative-margin tail, where the model is getting a data point
/// badly wrong. `exp(−u)` is unbounded there, `σ(−2u)²` saturates at 1, and `log σ(2u)` is linear
/// in `−u` between them. The obvious reading is that minimum probability flow should be dragged
/// furthest by an outlier and ratio matching least.
///
/// **It is not what happens.** `examples/estimator_shootout.rs` replaces a fraction of the rows
/// with uniform noise, fits each method on the corrupted data and scores it on the clean
/// distribution, as percent of what exact maximum likelihood reaches — a 10-spin ring with chords
/// at coupling 1.5, every method run to convergence at its own best step:
///
/// ```text
///   rows replaced:        0%      5%     15%     30%
///   pseudolikelihood   100.0    97.9    90.9    78.9      (4000 rows)
///   min prob flow       99.7    97.7    90.9    79.1
///   ratio matching      99.8    97.8    90.5    78.3
///   CD-10              100.0    99.1    95.4    87.0
/// ```
///
/// The three are within ONE POINT of each other at every corruption level and at both data sizes
/// tested, and the ordering the tails predict does not appear — at 200 rows minimum probability
/// flow is the least damaged of the three, which is backwards. What the table does separate is the
/// two FAMILIES: contrastive divergence is about eight points ahead of all three at 30%, on a model
/// where its own bias costs nothing.
///
/// The reading is that at these data sizes the shape of the tail is not what limits these methods —
/// consistency is, and all three are consistent, so they converge to nearly the same place whatever
/// they charge for a bad row. The tail would have to matter somewhere thinner than 200 rows, and
/// this crate does not have a measurement there. Kept as a shape argument with the measurement that
/// contradicts it beside it, because a prediction a crate makes about itself and quietly drops is
/// the kind that stays true forever.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FlipLoss {
    /// Pseudolikelihood (Besag 1975).
    Log,
    /// Minimum probability flow (Sohl-Dickstein, Battaglino & DeWeese 2011).
    Flow,
    /// Ratio matching (Hyvärinen 2007).
    Ratio,
}

impl FlipLoss {
    /// The score at margin `u`, per (row, site). Maximised.
    #[must_use]
    pub fn score(self, u: f64) -> f64 {
        match self {
            // `p_up(u, 1)` IS σ(2u) — the factor of two is the crate's spin convention and lives
            // in the kernel, so this reads it from there rather than restating it.
            //
            // Floored before the log: a margin the model has saturated gives a very large finite
            // number rather than −inf. This objective is returned to callers and compared across
            // methods, and one −inf row makes the whole mean −inf.
            FlipLoss::Log => crate::kernel::p_up(u, 1.0).max(f64::MIN_POSITIVE).ln(),
            FlipLoss::Flow => -(-u).exp(),
            FlipLoss::Ratio => {
                let q = crate::kernel::p_up(-u, 1.0);
                -q * q
            }
        }
    }

    /// `d(score)/du`. Positive for every loss at every margin — see the type's own docs.
    #[must_use]
    pub fn slope(self, u: f64) -> f64 {
        match self {
            FlipLoss::Log => 2.0 * crate::kernel::p_up(-u, 1.0),
            FlipLoss::Flow => (-u).exp(),
            FlipLoss::Ratio => {
                let q = crate::kernel::p_up(-u, 1.0);
                4.0 * q * q * (1.0 - q)
            }
        }
    }
}

/// Mean score of a one-bit-flip objective over the data, per row.
fn flip_score(g: &Graph, data: &Dataset, loss: FlipLoss) -> f64 {
    let mut total = 0.0;
    for row in &data.rows {
        for i in 0..g.n {
            total += loss.score(f64::from(row[i]) * g.field(i, row));
        }
    }
    total / data.rows.len() as f64
}

/// The gradient of a one-bit-flip objective with respect to every edge weight and bias.
///
/// Returned as `(d/dw per edge, d/dh per spin)` matching `edges` order, as an ASCENT direction.
/// Exact and closed-form. With `u_i = s_i f_i` and the per-site residual `r_i = s_i · slope(u_i)`:
///
/// ```text
///   d/dh_i   =  r_i                       since  du_i/dh_i  = s_i
///   d/dw_ij  =  r_i s_j  +  r_j s_i       since an edge is in the field of BOTH endpoints
/// ```
///
/// The two terms per edge are the easiest thing here to get half right, and half of it is still a
/// direction that goes uphill — just at the wrong rate, which no convergence test would notice.
/// `every_flip_gradient_matches_a_finite_difference` differences the objective itself instead.
fn flip_gradient(
    g: &Graph,
    data: &Dataset,
    edges: &[(usize, usize, f64)],
    loss: FlipLoss,
) -> (Vec<f64>, Vec<f64>) {
    let mut dw = vec![0.0; edges.len()];
    let mut dh = vec![0.0; g.n];
    for row in &data.rows {
        // One pass for the per-site residuals, so each is computed once rather than per edge.
        let resid: Vec<f64> = (0..g.n)
            .map(|i| {
                let s = f64::from(row[i]);
                s * loss.slope(s * g.field(i, row))
            })
            .collect();
        for i in 0..g.n {
            dh[i] += resid[i];
        }
        for (e, &(i, j, _)) in edges.iter().enumerate() {
            dw[e] += resid[i] * f64::from(row[j]) + resid[j] * f64::from(row[i]);
        }
    }
    let d = data.rows.len() as f64;
    for x in &mut dw {
        *x /= d;
    }
    for x in &mut dh {
        *x /= d;
    }
    (dw, dh)
}

/// Fit `structure` to `data` by ascending a one-bit-flip objective.
///
/// The body of [`train_pseudolikelihood`], [`train_mpf`] and [`train_ratio_matching`], which are
/// this function with the loss fixed. Public so a caller can hold the choice as DATA rather than as
/// three call sites — sweeping the family, recording which member a run used, or adding a fourth
/// loss to the comparison without touching the loop it shares with the other three.
///
/// # Errors
///
/// [`Error::HasLatent`] when the model has spins the data does not observe: a flip margin is a
/// function of the whole state, so a hidden unit leaves it uncomputable for every loss. Otherwise
/// as [`train`].
pub fn train_flip(
    structure: &Graph,
    data: &Dataset,
    p: &FitParams,
    loss: FlipLoss,
) -> Result<Trained, Error> {
    check(structure, data)?;
    if structure.n != data.visible {
        return Err(Error::HasLatent { spins: structure.n, visible: data.visible });
    }
    let mut wt = Weights::of(structure);
    for _ in 0..p.epochs {
        let g = wt.build();
        let (dw, dh) = flip_gradient(&g, data, &wt.edges, loss);
        wt.step(&dw, &dh, p);
    }
    let graph = wt.build();
    let log_likelihood = exact_log_likelihood(&graph, data).ok();
    Ok(Trained { graph, log_likelihood, epochs_run: p.epochs })
}

/// Mean pseudo-log-likelihood per row: `(1/D) Σ_d Σ_i log P(s_i | rest)`.
///
/// The objective [`train_pseudolikelihood`] ascends, exposed because a training method whose
/// objective cannot be evaluated is a method whose progress cannot be checked.
///
/// # Errors
///
/// [`Error::NoData`], [`Error::HasLatent`], and the malformed-row errors `check` raises.
pub fn pseudo_log_likelihood(g: &Graph, data: &Dataset) -> Result<f64, Error> {
    check(g, data)?;
    if g.n != data.visible {
        return Err(Error::HasLatent { spins: g.n, visible: data.visible });
    }
    Ok(flip_score(g, data, FlipLoss::Log))
}

/// Mean **minimum probability flow** objective per row (Sohl-Dickstein, Battaglino & DeWeese 2011),
/// as a score to be MAXIMISED — the published `K(θ)` with its sign flipped, so it reads the same
/// way round as every other objective in this module.
///
/// ```text
///   −K(θ)  =  −(1/D) Σ_d Σ_i exp( (E(s_d) − E(s_d with site i flipped)) / 2 )
/// ```
///
/// # What it is
///
/// Set up a continuous-time dynamics whose stationary distribution is the model, start it at the
/// data, and minimise the probability that flows OUT of the data in the first instant. That is the
/// whole derivation, and the sum above is what it collapses to when the only allowed transitions
/// are single spin flips. Zero flow is a model already at equilibrium on the data — the maximum
/// likelihood condition — so the estimator is consistent, and unlike contrastive divergence it
/// gets there without ever running the dynamics it is derived from.
///
/// It is included because it is the one estimator in this family designed FOR an Ising model
/// rather than adapted to one, and because it is the only closed-form method here whose loss is
/// unbounded: a data point the model gets badly wrong contributes `exp(−u)`, not a bounded
/// penalty. See [`FlipLoss`] for what that buys and costs against the other two.
///
/// # Errors
///
/// [`Error::NoData`], [`Error::HasLatent`], and the malformed-row errors `check` raises.
pub fn minimum_probability_flow(g: &Graph, data: &Dataset) -> Result<f64, Error> {
    check(g, data)?;
    if g.n != data.visible {
        return Err(Error::HasLatent { spins: g.n, visible: data.visible });
    }
    Ok(flip_score(g, data, FlipLoss::Flow))
}

/// Mean **ratio matching** objective per row (Hyvärinen 2007), as a score to be MAXIMISED.
///
/// ```text
///   −J(θ)  =  −(1/D) Σ_d Σ_i ( 1 / (1 + P(s_d) / P(s_d with site i flipped)) )²
/// ```
///
/// # What it is
///
/// Score matching without derivatives. Score matching fits a model by matching the gradient of
/// `log p` and so needs a continuous state; a spin has none. Ratio matching replaces the gradient
/// with the RATIO between a state and its one-bit neighbours, which is the discrete object that
/// plays the same role — and, like the score, it is free of the normaliser, since `Z` cancels in
/// any ratio of two probabilities of the same model.
///
/// The bracket is `σ(−2u)`: near zero when the data point is far more probable than its flip, near
/// one when it is far less. Squaring it is what makes this a matching criterion rather than a
/// likelihood, and it is also why this is the most bounded loss of the three — its penalty for a
/// hopeless data point saturates at 1 per site, so a single mislabelled row cannot dominate the
/// fit. See [`FlipLoss`].
///
/// # Errors
///
/// [`Error::NoData`], [`Error::HasLatent`], and the malformed-row errors `check` raises.
pub fn ratio_matching(g: &Graph, data: &Dataset) -> Result<f64, Error> {
    check(g, data)?;
    if g.n != data.visible {
        return Err(Error::HasLatent { spins: g.n, visible: data.visible });
    }
    Ok(flip_score(g, data, FlipLoss::Ratio))
}

/// Fit `structure`'s weights to `data` by **pseudolikelihood** — no sampling anywhere.
///
/// # Why this is here beside [`train`]
///
/// Contrastive divergence needs a negative phase, and a negative phase needs a sampler: its cost,
/// its bias, and its seed all enter the fit. Pseudolikelihood replaces the intractable normaliser
/// with a product of conditionals `P(s_i | rest)`, each of which is a logistic function of the
/// local field and computable exactly from the data. So the objective and its gradient are both
/// closed-form, the fit is deterministic, and there is no sampler to tune.
///
/// The price is that the objective is not the likelihood. It is *consistent* — the maximiser
/// converges to the true parameters as data grows, which
/// `pseudolikelihood_converges_on_the_model_that_generated_the_data` measures — but at finite data
/// it is a different objective with a different optimum. This crate already relies on that
/// consistency elsewhere: [`crate::certify`] fits an inverse temperature the same way.
///
/// The objective is concave in the parameters, being a sum of logistic log-likelihoods, so plain
/// gradient ascent has nowhere else to go.
///
/// # Errors
///
/// [`Error::HasLatent`] when the model has spins the data does not observe — see that variant.
/// Otherwise as [`train`].
pub fn train_pseudolikelihood(
    structure: &Graph,
    data: &Dataset,
    p: &FitParams,
) -> Result<Trained, Error> {
    train_flip(structure, data, p, FlipLoss::Log)
}

/// Fit `structure`'s weights to `data` by **minimum probability flow** — no sampling anywhere.
///
/// See [`minimum_probability_flow`] for what the objective is and why it exists. The loop is
/// [`train_pseudolikelihood`]'s exactly; only the loss differs.
///
/// The objective is convex in the parameters — it is a positive sum of exponentials of affine
/// functions of them — so plain gradient ascent has nowhere else to go, exactly as with
/// pseudolikelihood. It is also the STEEPEST of the three sampler-free losses at a bad margin, so
/// a step size that suits pseudolikelihood can be too large here; the default is not tuned for it.
///
/// # What it is worth here, measured
///
/// `examples/estimator_shootout.rs` scores all six methods against [`train_exact`] on a 10-spin
/// ring with chords, 1000 exact draws, every method run to convergence at its own best step, as
/// percent of the reachable range between an untrained model and the ceiling:
///
/// ```text
///   coupling scale       0.5     1.0     1.5     2.0     2.5
///   CD-10              100.0   100.0   100.0   100.0   100.0
///   pseudolikelihood    99.9    99.9    99.9    99.7    98.9
///   min prob flow       99.9    99.8    99.7    97.9    88.0
///   ratio matching      99.9    99.8    99.7    99.8    99.0
///   epochs to converge   500     500    9500     750   40000+
/// ```
///
/// Two things are worth reading off it and one is not. **The gap is real and it is small**: through
/// coupling 1.5 every sampler-free method is within 0.3% of a ceiling contrastive divergence
/// reaches exactly, which is a statement nobody can make without computing the ceiling. **The cost
/// is in STEPS, not in step cost**: an epoch here is one closed-form pass and an epoch of CD is
/// thirty-one sampled updates, but the epoch count rises with coupling until it stops converging at
/// all — the last column hit the example's 40000-epoch cap for all three methods, so those numbers
/// are LOWER BOUNDS on what the method reaches and not measurements of it. The 88.0% in particular
/// is an unconverged fit and should not be read as this method being worse than the other two.
///
/// What is NOT readable from it is any claim about contrastive divergence's bias, which is a claim
/// about a chain that cannot reach a model's modes; ten sweeps of a ten-spin machine can reach
/// anywhere, and the regime where CD genuinely fails is larger than anything [`train_exact`] can
/// price. A small gap here is a lower bound on the gap at scale.
///
/// # Errors
///
/// [`Error::HasLatent`] when the model has spins the data does not observe: the flip margin is a
/// function of the whole state, so a hidden unit leaves it uncomputable. Otherwise as [`train`].
pub fn train_mpf(structure: &Graph, data: &Dataset, p: &FitParams) -> Result<Trained, Error> {
    train_flip(structure, data, p, FlipLoss::Flow)
}

/// Fit `structure`'s weights to `data` by **ratio matching** — no sampling anywhere.
///
/// See [`ratio_matching`] for what the objective is and why it exists. The loop is
/// [`train_pseudolikelihood`]'s exactly; only the loss differs.
///
/// Unlike the other two this objective is NOT convex — the square of a sigmoid is not — so this is
/// the one sampler-free method here that can land somewhere a different start would not. It is
/// still deterministic given its start, which is the property the sampler-free methods are chosen
/// for; determinism and convexity are different guarantees and only the first is claimed.
///
/// It is also the SLOWEST of the three to converge in the measurement at [`train_mpf`] — 6500,
/// 32250 and 4500 epochs where pseudolikelihood took 500, 7250 and 750 on the same three fixtures —
/// which is what a saturating gradient costs: `4q²(1−q)` vanishes at both ends of the margin, so a
/// parameter that is nearly right and a parameter that is hopelessly wrong both move slowly.
///
/// # Errors
///
/// As [`train_mpf`].
pub fn train_ratio_matching(
    structure: &Graph,
    data: &Dataset,
    p: &FitParams,
) -> Result<Trained, Error> {
    train_flip(structure, data, p, FlipLoss::Ratio)
}

/// Fit `structure` by **variational** contrastive divergence — the deep-Boltzmann-machine recipe
/// (Salakhutdinov & Hinton 2009).
///
/// # What changes, and it is only the positive phase
///
/// Fitting latent units needs `⟨s_i s_j⟩` under `p(h | v)`, the posterior over the hidden units
/// with the visible ones clamped to a data row. [`train`] draws ONE SAMPLE from that posterior per
/// row, which is unbiased and noisy. This solves a mean-field approximation to it instead and uses
/// the MEANS, which is noiseless and biased — the mean-field posterior is a product distribution
/// and the true one is not, so `⟨h_j h_k⟩` becomes `μ_j μ_k` and every correlation between hidden
/// units is thrown away.
///
/// That is the trade in one sentence, and neither half of it is a matter of opinion here:
/// [`train_exact`] computes the true gradient on models small enough to enumerate, so both can be
/// scored against the same reference rather than against each other.
///
/// The negative phase is unchanged and defaults to persistent chains, which is what the original
/// recipe uses — a variational positive phase does nothing about the model average, which is the
/// other half of the gradient and the harder one.
///
/// # What it is worth here, measured
///
/// On `dbm(4, [3, 2])` — 9 spins, so [`train_exact`] can compute the ceiling — fitted to
/// bars-and-stripes from a seeded start, as percent of the reachable range between the untrained
/// model and exact maximum likelihood:
///
/// ```text
///   initial weight scale     0.1     0.3     0.6
///   variational            84.9%   73.5%   46.0%      (mean field run to convergence)
///   sampled (`train`)      82.0%   86.3%   85.5%
/// ```
///
/// **It loses, and it loses for the reason the approximation predicts.** Mean field discards
/// `⟨h_j h_k⟩ − ⟨h_j⟩⟨h_k⟩`, and a deep machine has hidden-to-hidden edges where a restricted one
/// does not, so exactly the correlations it throws away are the ones those edges are there to
/// carry. They grow with coupling strength, which is the axis the table sweeps.
///
/// **This is not a convergence artifact, which was checked rather than assumed.** Damped at 0.5 the
/// clamped mean field reaches a residual below `1e-13` in at most 123 iterations at every scale
/// here, so the rows above are its fixed point and not its budget.
///
/// **A deliberately under-converged mean field beats the converged one at strong coupling**, which
/// is worth knowing before tuning `positive_sweeps` up:
///
/// ```text
///   scale 0.6:     5 iters 80.5%    20 iters 36.7%    100 iters 40.4%    500 iters 46.0%
/// ```
///
/// Five iterations is not a better approximation to the posterior; it is a different estimator that
/// happens to fit better here, in the same way early stopping regularises. Reported because the
/// obvious reflex on seeing 46% is to raise the cap, and raising it is not what helps.
///
/// It ships because it is the standard recipe for a deep Boltzmann machine and a caller whose
/// posterior is too expensive to sample will want it; it ships with this table because a method
/// that is standard in the literature and worse here should say so.
///
/// # Why this does not share [`train`]'s loop
///
/// It would have to. `train`'s non-persistent negative phase REUSES the positive-phase sampler,
/// carrying its advanced RNG state into the negative chain; there is no such sampler here, so
/// threading a mode through would change `train`'s random stream and with it every seeded test
/// that pins its behaviour. The parameter plumbing is shared through `Weights`; the loop is not.
///
/// # Errors
///
/// As [`train`].
pub fn train_variational(
    structure: &Graph,
    data: &Dataset,
    p: &Params,
    seed: u64,
) -> Result<Trained, Error> {
    check(structure, data)?;
    let n = structure.n;
    let mut rng = Pcg::new(seed, 0x00EB_5F00);
    let mut wt = Weights::of(structure);

    // One fantasy chain per batch slot, carrying its own RNG state -- rebuilding a sampler from a
    // fixed seed each update would replay one random stream forever, which is a chain that moves
    // and always the same way. See `Params::persistent`.
    let mut fantasy: Vec<(Vec<i8>, Pcg)> = (0..p.batch.max(1))
        .map(|s| {
            (
                (0..n).map(|_| if rng.f64() < 0.5 { -1i8 } else { 1 }).collect(),
                Pcg::new(seed ^ 0x5F5F_0000 ^ s as u64, 0x9E37),
            )
        })
        .collect();

    let mut order: Vec<usize> = (0..data.rows.len()).collect();
    for epoch in 0..p.epochs {
        for i in (1..order.len()).rev() {
            let j = (rng.f64() * (i + 1) as f64) as usize % (i + 1);
            order.swap(i, j);
        }
        let g = wt.build();
        let decay = if p.epochs > 1 {
            1.0 - 0.9 * epoch as f64 / (p.epochs - 1) as f64
        } else {
            1.0
        };

        for chunk in order.chunks(p.batch.max(1)) {
            let mut d_edge = vec![0.0f64; wt.edges.len()];
            let mut d_bias = vec![0.0f64; n];

            for (slot, &r) in chunk.iter().enumerate() {
                let row = &data.rows[r];

                // POSITIVE PHASE, solved rather than sampled. `positive_sweeps` is the iteration
                // cap here; damping is 0.5 because an undamped mean field on a frustrated model
                // oscillates rather than converging, and a positive phase that oscillates makes
                // the gradient a function of where the cap fell.
                let mf = crate::meanfield::naive_mean_field_clamped(
                    &g,
                    1.0,
                    row,
                    p.positive_sweeps.max(1),
                    0.5,
                );

                // NEGATIVE PHASE. A concrete state is needed to advance a chain from; it is drawn
                // from the variational posterior, while the STATISTICS above come from the means.
                let neg: Vec<i8> = if p.persistent {
                    let idx = slot % fantasy.len();
                    let mut fs = Sampler::new(&g, 1.0, 0);
                    fs.s.copy_from_slice(&fantasy[idx].0);
                    fs.rng = fantasy[idx].1.clone();
                    fs.sweeps(p.k.max(1), None);
                    fantasy[idx].0.copy_from_slice(&fs.s);
                    fantasy[idx].1 = fs.rng.clone();
                    fs.s
                } else {
                    let nseed = (u64::from(rng.next_u32()) << 32) | u64::from(rng.next_u32());
                    let mut fs = Sampler::new(&g, 1.0, nseed);
                    for i in 0..n {
                        fs.s[i] = if rng.f64() < (1.0 + mf.m[i]) / 2.0 { 1 } else { -1 };
                    }
                    for (i, &v) in row.iter().enumerate() {
                        fs.s[i] = v;
                    }
                    fs.sweeps(p.k.max(1), None);
                    fs.s
                };

                for (e, &(i, j, _)) in wt.edges.iter().enumerate() {
                    d_edge[e] += mf.m[i] * mf.m[j]
                        - f64::from(neg[i]) * f64::from(neg[j]);
                }
                for i in 0..n {
                    d_bias[i] += mf.m[i] - f64::from(neg[i]);
                }
            }

            let scale = p.learning_rate * decay / chunk.len() as f64;
            for (e, w) in wt.edges.iter_mut().enumerate() {
                w.2 += scale * d_edge[e];
            }
            for i in 0..n {
                wt.bias[i] += scale * d_bias[i];
            }
        }
    }

    let graph = wt.build();
    let log_likelihood = exact_log_likelihood(&graph, data).ok();
    Ok(Trained { graph, log_likelihood, epochs_run: p.epochs })
}

/// The TRUE maximum-likelihood gradient, by enumeration.
///
/// `⟨s_i s_j⟩_data − ⟨s_i s_j⟩_model`, with BOTH averages exact: the second over every one of the
/// `2^n` states, the first over the states agreeing with each row on its visible part. No sampler,
/// no bias, no `k`. This is the quantity contrastive divergence approximates and the quantity the
/// three flip losses do not target at all, and it is what makes the others scoreable rather than
/// merely comparable to each other.
///
/// Both phases read the same `2^n` table of `−E`. The visible units are indices `0..visible`, so
/// the LOW BITS OF A STATE INDEX ARE ITS VISIBLE PATTERN and a row's completions are the states
/// `vkey | (c << visible)` — already computed, never re-energised.
fn exact_gradient(
    g: &Graph,
    data: &Dataset,
    edges: &[(usize, usize, f64)],
) -> (Vec<f64>, Vec<f64>) {
    let n = g.n;
    let states = 1usize << n;

    let mut s = vec![-1i8; n];
    let mut neg_e = Vec::with_capacity(states);
    let mut top = f64::NEG_INFINITY;
    for mask in 0..states {
        for (i, x) in s.iter_mut().enumerate() {
            *x = if mask >> i & 1 == 1 { 1 } else { -1 };
        }
        let e = -g.energy(&s);
        top = top.max(e);
        neg_e.push(e);
    }

    // Negative phase: the model's own moments. Shifted by `top` before exponentiating, which
    // cancels in the ratio and keeps a strongly-coupled model off the overflow.
    let mut z = 0.0;
    let mut model_h = vec![0.0; n];
    let mut model_w = vec![0.0; edges.len()];
    for (mask, &e) in neg_e.iter().enumerate() {
        let p = (e - top).exp();
        z += p;
        for (i, acc) in model_h.iter_mut().enumerate() {
            *acc += if mask >> i & 1 == 1 { p } else { -p };
        }
        for (k, &(i, j, _)) in edges.iter().enumerate() {
            model_w[k] += if (mask >> i & 1) == (mask >> j & 1) { p } else { -p };
        }
    }

    // Positive phase: the data's moments, with the latent units integrated out under the model's
    // own conditional. When there are no latent units this is one state per row at weight one, so
    // it reduces to the plain data correlations without a branch to say so.
    let comps = 1usize << (n - data.visible);
    let mut data_h = vec![0.0; n];
    let mut data_w = vec![0.0; edges.len()];
    let mut row_h = vec![0.0; n];
    let mut row_w = vec![0.0; edges.len()];
    for row in &data.rows {
        let mut vkey = 0usize;
        for (i, &v) in row.iter().enumerate() {
            if v == 1 {
                vkey |= 1 << i;
            }
        }
        let mut best = f64::NEG_INFINITY;
        for c in 0..comps {
            best = best.max(neg_e[vkey | (c << data.visible)]);
        }
        row_h.fill(0.0);
        row_w.fill(0.0);
        let mut zr = 0.0;
        for c in 0..comps {
            let mask = vkey | (c << data.visible);
            let p = (neg_e[mask] - best).exp();
            zr += p;
            for (i, acc) in row_h.iter_mut().enumerate() {
                *acc += if mask >> i & 1 == 1 { p } else { -p };
            }
            for (k, &(i, j, _)) in edges.iter().enumerate() {
                row_w[k] += if (mask >> i & 1) == (mask >> j & 1) { p } else { -p };
            }
        }
        for (acc, x) in data_h.iter_mut().zip(&row_h) {
            *acc += x / zr;
        }
        for (acc, x) in data_w.iter_mut().zip(&row_w) {
            *acc += x / zr;
        }
    }

    let d = data.rows.len() as f64;
    let dh = (0..n).map(|i| data_h[i] / d - model_h[i] / z).collect();
    let dw = (0..edges.len()).map(|k| data_w[k] / d - model_w[k] / z).collect();
    (dw, dh)
}

/// Fit `structure`'s weights to `data` by **exact maximum likelihood**, by enumeration.
///
/// # Why a method nobody can run on a real model is worth shipping
///
/// It is the CEILING. Every other fit in this module is judged by [`exact_log_likelihood`], which
/// says what a fit achieved but not what was there to achieve: a run reporting `−3.0` is a good
/// fit if the structure's best is `−2.9` and a poor one if it is `−1.5`, and nothing in the number
/// itself distinguishes those. This computes the best, so the others become a fraction of a
/// reachable target rather than a league table against each other.
///
/// It also settles what contrastive divergence's bias costs, in the only way that is not an
/// argument: run both on the same structure and the same data and subtract.
///
/// Unlike the three flip losses, this one HANDLES LATENT UNITS — the positive phase integrates
/// them out exactly under the model's own conditional, which is precisely the sum a clamped
/// sampler is estimating. So it is also the reference for [`train`] on an RBM, not only for the
/// fully-visible case.
///
/// # Cost
///
/// `epochs × 2^n` energy evaluations, and that is the whole story: the enumeration is per STEP,
/// not once. [`MAX_ENUMERATED`] bounds `n` at the size a single pass is affordable at, which for a
/// few hundred epochs is far more than this is comfortable at — 15 spins is about a second, and
/// every spin after that doubles it. The limit that binds is patience, not the constant.
///
/// # Errors
///
/// [`Error::TooLarge`] past [`MAX_ENUMERATED`] spins, and otherwise as [`train`].
pub fn train_exact(structure: &Graph, data: &Dataset, p: &FitParams) -> Result<Trained, Error> {
    check(structure, data)?;
    if structure.n > MAX_ENUMERATED {
        return Err(Error::TooLarge { spins: structure.n, limit: MAX_ENUMERATED });
    }
    let mut wt = Weights::of(structure);
    for _ in 0..p.epochs {
        let g = wt.build();
        let (dw, dh) = exact_gradient(&g, data, &wt.edges);
        wt.step(&dw, &dh, p);
    }
    let graph = wt.build();
    let log_likelihood = exact_log_likelihood(&graph, data).ok();
    Ok(Trained { graph, log_likelihood, epochs_run: p.epochs })
}

/// Which deterministic free energy supplies the model's own moments in the negative phase.
///
/// [`train`] samples `⟨s_i s_j⟩_model` and [`train_exact`] enumerates it. [`crate::meanfield`]
/// already computes three approximate free energies, and each is **stationary in its
/// magnetisations** — mean field at `m = tanh(...)`, TAP at its own equations, Bethe at a belief
/// propagation fixed point. Stationarity is what turns a free energy into a moment estimator: the
/// fixed point's own motion contributes nothing to the total derivative, so
///
/// ```text
///   ∂ ln Z_approx / ∂h_i   =  m_i
///   ∂ ln Z_approx / ∂J_ij  =  ⟨s_i s_j⟩_approx
/// ```
///
/// and the negative phase can be SOLVED rather than drawn. `the_moments_are_the_derivative_of_the_free_energy`
/// checks that identity against central differences of `meanfield`'s own `log_z`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Moments {
    /// Naive mean field: `⟨s_i s_j⟩ ≈ m_i m_j`, which discards every connected correlation.
    MeanField,
    /// TAP, second-order Plefka: `m_i m_j + J_ij (1 − m_i²)(1 − m_j²)`.
    ///
    /// Gabrié, Tramel & Krzakala, "Training Restricted Boltzmann Machines via the
    /// Thouless-Anderson-Palmer free energy", NIPS 2015. The extra term is the derivative of the
    /// Onsager reaction, and it is the whole difference between this and mean field.
    Tap,
    /// Bethe: the pair belief that belief propagation converges to, **exact on a tree**.
    Bethe,
}

/// How hard to solve the fixed point that supplies the moments.
///
/// Separate from [`FitParams`] because these tune the inner solve, not the ascent, and because
/// adding fields to `FitParams` would break every existing struct literal.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Solve {
    /// Iteration cap for the mean-field, TAP or belief-propagation fixed point.
    pub iters: usize,
    /// Damping in `[0, 1)`. `0.5` keeps a frustrated model from oscillating instead of converging.
    pub damping: f64,
}

impl Default for Solve {
    fn default() -> Self {
        Solve { iters: 500, damping: 0.5 }
    }
}

/// A deterministic estimate of a model's own first and second moments.
#[derive(Clone, Debug)]
pub struct ModelMoments {
    /// `⟨s_i⟩` for every spin.
    pub m: Vec<f64>,
    /// `⟨s_i s_j⟩` indexed by CSR adjacency slot, exactly like [`crate::graph::Graph`]'s `w`: slot
    /// `e` belongs to the node whose `offset` range contains it and its partner is `nbr[e]`, so
    /// each undirected edge appears twice with the same value.
    pub pair: Vec<f64>,
    /// The free energy these are the parameter derivative of: `ln Z_MF`, `ln Z_TAP` or
    /// `ln Z_Bethe`.
    pub log_z: f64,
    /// Largest change on the last iteration of the underlying fixed point. Not zero means the cap
    /// bound before convergence did, and the moments are then a function of where the cap fell.
    pub residual: f64,
}

/// `⟨s_i s_j⟩` under the Bethe pair belief, from the two marginals and the coupling.
///
/// A 2×2 distribution is fixed by its two marginals plus one odds ratio, and at a belief
/// propagation fixed point the pair belief `b_ij(s,t) ∝ exp(J s t + H_i^{∖j} s + H_j^{∖i} t)`
/// supplies all three: its marginals are `m_i`, `m_j` and its odds ratio is `exp(4J)` whatever the
/// cavity fields are. Writing `b(s,t) = (1 + s m_i + t m_j + s t c)/4` turns that into
///
/// ```text
///   λ c² − 2(2+λ) c + λ(1 − m_i² − m_j²) + 2 m_i m_j (2+λ) = 0,     λ = exp(4J) − 1
/// ```
///
/// so the correlation is recoverable from what [`crate::meanfield::belief_propagation`] already
/// returns, without reaching into its messages.
fn bethe_pair(mi: f64, mj: f64, w: f64) -> f64 {
    // Clamped because exp(4J) overflows past J = 177 and loses the discriminant to cancellation
    // well before that. By |J| = 10 the correlation has saturated at 1 − |m_i − m_j| to far more
    // digits than a fit can distinguish, so the clamp is numerically invisible and the alternative
    // is a NaN.
    let lam = (4.0 * w.clamp(-10.0, 10.0)).exp() - 1.0;
    let b = 2.0 + lam;
    let c = lam * (1.0 - mi * mi - mj * mj) + 2.0 * mi * mj * b;
    // The root that is `m_i m_j` at J = 0, written as C/(B + √D) rather than (B − √D)/λ: the
    // second is 0/0 there and loses every digit near it, and J = 0 is where a fit STARTS.
    // `b` is `1 + exp(4J) > 1`, so the denominator never vanishes.
    let disc = (b * b - lam * c).max(0.0);
    c / (b + disc.sqrt())
}

/// The approximation's estimate of a model's own moments, at inverse temperature 1.
///
/// The negative phase of [`train_free_energy`], exposed because a deterministic negative phase is
/// worth having on its own — it is a sampler-free answer to `⟨s_i s_j⟩_model` for any model
/// [`crate::meanfield`] can solve, enumerable or not.
#[must_use]
pub fn model_moments(g: &Graph, which: Moments, s: &Solve) -> ModelMoments {
    let (iters, damping) = (s.iters.max(1), s.damping);
    let (m, log_z, residual) = match which {
        Moments::MeanField => {
            let r = crate::meanfield::naive_mean_field(g, 1.0, iters, damping);
            (r.m, r.log_z, r.residual)
        }
        Moments::Tap => {
            let r = crate::meanfield::tap(g, 1.0, iters, damping);
            (r.m, r.log_z, r.residual)
        }
        Moments::Bethe => {
            let r = crate::meanfield::belief_propagation(g, 1.0, iters, damping);
            (r.m, r.log_z, r.residual)
        }
    };
    let mut pair = vec![0.0f64; g.nbr.len()];
    for i in 0..g.n {
        for e in g.offset[i]..g.offset[i + 1] {
            let j = g.nbr[e] as usize;
            pair[e] = match which {
                Moments::MeanField => m[i] * m[j],
                Moments::Tap => {
                    m[i] * m[j] + g.w[e] * (1.0 - m[i] * m[i]) * (1.0 - m[j] * m[j])
                }
                Moments::Bethe => bethe_pair(m[i], m[j], g.w[e]),
            };
        }
    }
    ModelMoments { m, pair, log_z, residual }
}

/// The CSR slot holding edge `(i, j)`.
fn slot_of(g: &Graph, i: usize, j: usize) -> usize {
    (g.offset[i]..g.offset[i + 1]).find(|&e| g.nbr[e] as usize == j).expect("edge is in the CSR")
}

/// The model with spins `0..row.len()` pinned to `row`, over the same node set.
///
/// Pinning is graph surgery and not a large field: drop a pinned spin's edges and push `J_ij s_i`
/// into each free neighbour's bias, which leaves `p(rest | visible = row)` EXACTLY. The same
/// approximation then runs on it unchanged. Pinned nodes survive as isolated unbiased vertices
/// purely so that node indices do not move.
fn clamped_graph(g: &Graph, row: &[i8]) -> Graph {
    let fixed = row.len().min(g.n);
    let mut gb = GraphBuilder::new(g.n);
    for i in 0..g.n {
        for e in g.offset[i]..g.offset[i + 1] {
            let j = g.nbr[e] as usize;
            // Upper triangle only, and `j > i` means `j < fixed` implies `i < fixed`: an edge with
            // a pinned high endpoint and a free low one does not exist.
            if j <= i {
                continue;
            }
            if i < fixed {
                if j >= fixed {
                    gb.bias(j, g.w[e] * f64::from(row[i]));
                }
            } else {
                gb.couple(i, j, g.w[e]);
            }
        }
    }
    for i in fixed..g.n {
        gb.bias(i, g.h[i]);
    }
    gb.build()
}

/// The data's moments when every spin is observed: one pass, exact, parameter-free.
fn data_moments(n: usize, data: &Dataset, edges: &[(usize, usize, f64)]) -> (Vec<f64>, Vec<f64>) {
    let mut m = vec![0.0f64; n];
    let mut pair = vec![0.0f64; edges.len()];
    for row in &data.rows {
        for i in 0..n {
            m[i] += f64::from(row[i]);
        }
        for (k, &(i, j, _)) in edges.iter().enumerate() {
            pair[k] += f64::from(row[i]) * f64::from(row[j]);
        }
    }
    let d = data.rows.len() as f64;
    for v in &mut m {
        *v /= d;
    }
    for v in &mut pair {
        *v /= d;
    }
    (m, pair)
}

/// The data's moments with latent units: the same approximation, per row, with the visible units
/// pinned by [`clamped_graph`].
fn clamped_moments(
    g: &Graph,
    data: &Dataset,
    edges: &[(usize, usize, f64)],
    which: Moments,
    s: &Solve,
) -> (Vec<f64>, Vec<f64>) {
    let fixed = data.visible;
    let mut m = vec![0.0f64; g.n];
    let mut pair = vec![0.0f64; edges.len()];
    for row in &data.rows {
        let gc = clamped_graph(g, row);
        let mc = model_moments(&gc, which, s);
        let at = |i: usize| if i < fixed { f64::from(row[i]) } else { mc.m[i] };
        for i in 0..g.n {
            m[i] += at(i);
        }
        for (k, &(i, j, _)) in edges.iter().enumerate() {
            // A pinned endpoint carries no fluctuation, so the pair moment factorises exactly;
            // only a free–free edge still exists in `gc` and needs the approximation's answer.
            pair[k] += if i < fixed && j < fixed {
                f64::from(row[i]) * f64::from(row[j])
            } else if i < fixed {
                f64::from(row[i]) * mc.m[j]
            } else {
                mc.pair[slot_of(&gc, i, j)]
            };
        }
    }
    let d = data.rows.len() as f64;
    for v in &mut m {
        *v /= d;
    }
    for v in &mut pair {
        *v /= d;
    }
    (m, pair)
}

/// Fit `structure` to `data` with a **deterministic negative phase**: no sampler anywhere.
///
/// The gradient is the one at the top of this module, `⟨·⟩_data − ⟨·⟩_model`, with the model side
/// supplied by [`model_moments`] instead of by a chain. With every spin observed the data side is
/// the plain empirical average, so the approximation enters in exactly one place and the run is a
/// pure function of the structure, the data and `p` — same answer every time, no seed to pass.
/// With latent units the same approximation supplies the positive phase too, on the clamped model.
///
/// # What it is worth here, measured
///
/// Fully-visible 8-spin ring with chords, 512 rows drawn exactly from a planted model, 2000 epochs
/// at `lr = 0.05`, scored by [`exact_log_likelihood`] as percent of the range between the
/// untrained model and [`train_exact`]'s ceiling —
/// `the_deterministic_negative_phase_against_the_exact_ceiling` prints exactly this:
///
/// ```text
///   negative phase              J = 0.6   J = 1.4
///   naive mean field               23.7      84.8
///   TAP                            99.0      92.3
///   Bethe                          99.3      90.9
///   CD-10, 6 seeds (`train`)       99.7     100.0
///
///   ceiling,   nats per row      -4.959    -3.633
///   untrained, nats per row      -5.545    -5.545
/// ```
///
/// **TAP and Bethe reach 99% of the exact ceiling with no sampler anywhere**, which is the claim
/// worth making. Three things beside it are not flattering and are the reason the table is here.
///
/// **It loses to the sampled negative phase in BOTH regimes.** By 0.4 points at `J = 0.6` and by
/// 7.7 at `J = 1.4`. Contrastive divergence's bias is not what limits an eight-spin machine, and
/// a second-order Plefka expansion is a real approximation where a longer chain is not — so the
/// trade is determinism and cost, never accuracy. Same shape as [`train_variational`]'s table,
/// where the variational positive phase also loses at strong coupling.
///
/// **Mean field is far behind, and worst where the model is WEAKEST.** `m_i m_j` can only reach a
/// data correlation by driving `|m|` up, so the fit inflates its couplings; at `J = 0.6` there is
/// only 0.586 nats of range to win and it gives most of it back. It is the failure the Onsager
/// term exists to fix, and the 75-point gap to TAP is what fixing it is worth.
///
/// **Longer training makes mean field worse, not better** — 27.7 / 23.7 / 24.9 at 600 / 2000 /
/// 6000 epochs weak, and 88.4 / 84.8 / 79.2 strong. Its fixed point is the problem, not its
/// budget, and a shorter run only flatters it. TAP and Bethe are unmoved across the same sweep,
/// which is what says the other rows are converged rather than truncated.
///
/// **Bethe beats TAP at `J = 0.6` and loses to it at `J = 1.4`.** One instance, gaps of 0.3 and
/// 1.4 points, so it is reported and not asserted: this crate has no measurement wide enough to
/// call a crossover.
///
/// **On a tree the Bethe gradient is not an approximation at all.** Belief propagation is exact
/// there, so `train_bethe` reproduces `train_exact` step for step to `1e-7` in every weight —
/// which is the test that pins the pair-belief algebra rather than a tolerance chosen to pass.
///
/// # Errors
///
/// As [`train`].
pub fn train_free_energy(
    structure: &Graph,
    data: &Dataset,
    p: &FitParams,
    which: Moments,
    s: &Solve,
) -> Result<Trained, Error> {
    check(structure, data)?;
    let mut wt = Weights::of(structure);
    // Fully visible: the data moments do not depend on the parameters, so they are computed once.
    let fixed = if structure.n == data.visible {
        Some(data_moments(structure.n, data, &wt.edges))
    } else {
        None
    };
    for _ in 0..p.epochs {
        let g = wt.build();
        let (pos_m, pos_pair) = match &fixed {
            Some(x) => x.clone(),
            None => clamped_moments(&g, data, &wt.edges, which, s),
        };
        let neg = model_moments(&g, which, s);
        let dw: Vec<f64> = wt
            .edges
            .iter()
            .enumerate()
            .map(|(k, &(i, j, _))| pos_pair[k] - neg.pair[slot_of(&g, i, j)])
            .collect();
        let dh: Vec<f64> = (0..structure.n).map(|i| pos_m[i] - neg.m[i]).collect();
        wt.step(&dw, &dh, p);
    }
    let graph = wt.build();
    let log_likelihood = exact_log_likelihood(&graph, data).ok();
    Ok(Trained { graph, log_likelihood, epochs_run: p.epochs })
}

/// [`train_free_energy`] with the TAP negative phase (Gabrié, Tramel & Krzakala 2015).
///
/// # Errors
///
/// As [`train`].
pub fn train_tap(
    structure: &Graph,
    data: &Dataset,
    p: &FitParams,
    s: &Solve,
) -> Result<Trained, Error> {
    train_free_energy(structure, data, p, Moments::Tap, s)
}

/// [`train_free_energy`] with the Bethe negative phase, which is the EXACT gradient on a tree.
///
/// # Errors
///
/// As [`train`].
pub fn train_bethe(
    structure: &Graph,
    data: &Dataset,
    p: &FitParams,
    s: &Solve,
) -> Result<Trained, Error> {
    train_free_energy(structure, data, p, Moments::Bethe, s)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// THE FIXED POINT IS MOMENT MATCHING, and that is what makes this a fit rather than a loop.
    ///
    /// The gradient is `⟨s_i s_j⟩_data − ⟨s_i s_j⟩_model`, so at a fixed point the two are equal.
    /// The model side is computed by ENUMERATION here, not by more sampling: a check that compares
    /// a sampler's average against a sampler's average agrees with itself whatever it is doing.
    #[test]
    fn a_fully_visible_fit_matches_the_data_correlations() {
        // Three spins, fully connected, fitted to data with a definite correlation structure:
        // s0 and s1 agree, s2 is independent and biased up.
        let mut gb = GraphBuilder::new(3);
        gb.couple(0, 1, 0.0);
        gb.couple(0, 2, 0.0);
        gb.couple(1, 2, 0.0);
        let structure = gb.build();

        let rows: Vec<Vec<i8>> = vec![
            vec![1, 1, 1],
            vec![1, 1, 1],
            vec![1, 1, -1],
            vec![-1, -1, 1],
            vec![-1, -1, 1],
            vec![-1, -1, -1],
        ];
        let data = Dataset { visible: 3, rows: rows.clone() };
        let p = Params { epochs: 4_000, k: 20, learning_rate: 0.05, batch: 6, positive_sweeps: 1, persistent: false };
        let t = train(&structure, &data, &p, 7).unwrap();

        // Data moments.
        let m = rows.len() as f64;
        let dc = |i: usize, j: usize| {
            rows.iter().map(|r| (r[i] * r[j]) as f64).sum::<f64>() / m
        };
        let dm = |i: usize| rows.iter().map(|r| r[i] as f64).sum::<f64>() / m;

        // Model moments, by enumeration.
        let g = &t.graph;
        let mut z = 0.0;
        let mut corr = [[0.0f64; 3]; 3];
        let mut mag = [0.0f64; 3];
        for mask in 0..8usize {
            let s: Vec<i8> = (0..3).map(|i| if mask >> i & 1 == 1 { 1 } else { -1 }).collect();
            let w = (-g.energy(&s)).exp();
            z += w;
            for i in 0..3 {
                mag[i] += w * s[i] as f64;
                for j in 0..3 {
                    corr[i][j] += w * (s[i] * s[j]) as f64;
                }
            }
        }
        for i in 0..3 {
            assert!(
                (mag[i] / z - dm(i)).abs() < 0.05,
                "magnetisation {i}: model {:.4} vs data {:.4}",
                mag[i] / z,
                dm(i)
            );
            for j in (i + 1)..3 {
                assert!(
                    (corr[i][j] / z - dc(i, j)).abs() < 0.05,
                    "correlation ({i},{j}): model {:.4} vs data {:.4}",
                    corr[i][j] / z,
                    dc(i, j)
                );
            }
        }
    }

    /// PCD trains, and on a model small enough to judge exactly it does not beat CD.
    ///
    /// Two claims, both measured rather than assumed: the persistent path is a working trainer (it
    /// raises the exact likelihood over an untrained machine), and at a large learning rate it is
    /// WORSE than plain CD, because the fantasy chains cannot track a fast-moving model. The second
    /// is the textbook expectation failing on a verifiable model, which is worth a test rather than
    /// a footnote.
    #[test]
    fn persistent_divergence_trains_and_does_not_beat_cd_here() {
        let data = bars_and_stripes(3);
        let g = rbm(9, 6);
        let untrained = exact_log_likelihood(&g, &data).unwrap();
        let base = Params { epochs: 400, k: 5, positive_sweeps: 5, learning_rate: 0.05, batch: 8, persistent: false };
        let mut pp = base;
        pp.persistent = true;

        let (mut cd_sum, mut pcd_sum) = (0.0, 0.0);
        for seed in 0..4u64 {
            let lc = exact_log_likelihood(&train(&g, &data, &base, seed).unwrap().graph, &data).unwrap();
            let lp = exact_log_likelihood(&train(&g, &data, &pp, seed).unwrap().graph, &data).unwrap();
            assert!(lp > untrained, "seed {seed}: PCD must train at all, {lp} vs untrained {untrained}");
            cd_sum += lc;
            pcd_sum += lp;
        }
        assert!(cd_sum > pcd_sum, "at lr 0.05 CD should win here: CD {cd_sum}, PCD {pcd_sum}");
    }

    /// The likelihood must be a likelihood: negative, and improved by training.
    #[test]
    fn training_raises_the_exact_log_likelihood() {
        let data = bars_and_stripes(2);
        let structure = rbm(4, 4);
        let before = exact_log_likelihood(&structure, &data).unwrap();
        // An untrained model with all weights zero is uniform over 2^8 states, so every row has
        // probability 2^-4 given the visible marginal is uniform over 2^4. Its log-likelihood is
        // therefore exactly -4 ln 2, which is the only value it can be and is worth pinning.
        assert!((before - (-4.0 * 2f64.ln())).abs() < 1e-9, "{before}");

        let p = Params { epochs: 600, k: 10, ..Params::default() };
        let t = train(&structure, &data, &p, 3).unwrap();
        let after = t.log_likelihood.unwrap();
        assert!(after > before + 0.05, "training must help: {before:.4} -> {after:.4}");
        // And a likelihood is a log of something at most 1.
        assert!(after < 0.0, "a log-likelihood is negative: {after}");
    }

    /// Bars and stripes is the dataset it claims to be.
    #[test]
    fn bars_and_stripes_is_the_right_set() {
        // 2^side row patterns plus 2^side column patterns, minus the two counted twice: all-on and
        // all-off are both a bar pattern and a stripe pattern.
        for side in [2usize, 3, 4] {
            let d = bars_and_stripes(side);
            assert_eq!(d.visible, side * side);
            assert_eq!(d.rows.len(), 2 * (1 << side) - 2, "side {side}");
            assert!(d.rows.iter().all(|r| r.iter().all(|&v| v == 1 || v == -1)));
        }
        // Every row really is all-bars or all-stripes: constant along one axis.
        let d = bars_and_stripes(3);
        for r in &d.rows {
            let rows_const = (0..3).all(|a| (0..3).all(|b| r[a * 3 + b] == r[a * 3]));
            let cols_const = (0..3).all(|a| (0..3).all(|b| r[b * 3 + a] == r[a]));
            assert!(rows_const || cols_const, "{r:?}");
        }
    }

    /// A deep machine and a wide one with the same latent count are different graphs, and the deep
    /// one has fewer edges. That difference is the experiment, so it is worth pinning.
    #[test]
    fn a_deep_machine_has_fewer_edges_than_a_wide_one_with_the_same_latents() {
        let wide = rbm(9, 8);
        let deep = dbm(9, &[4, 4]);
        assert_eq!(wide.n, deep.n);
        assert_eq!(wide.n_edges, 9 * 8);
        assert_eq!(deep.n_edges, 9 * 4 + 4 * 4);
        assert!(deep.n_edges < wide.n_edges);
        // One layer of a dbm IS an rbm.
        assert_eq!(dbm(9, &[8]).n_edges, wide.n_edges);
    }

    #[test]
    fn a_malformed_dataset_is_refused_by_name() {
        let g = rbm(3, 2);
        let p = Params::default();
        let err = |d: Dataset| train(&g, &d, &p, 1).unwrap_err();
        assert_eq!(err(Dataset { visible: 3, rows: vec![] }), Error::NoData);
        assert_eq!(
            err(Dataset { visible: 3, rows: vec![vec![1, 1]] }),
            Error::RowWidth { row: 0, len: 2, want: 3 }
        );
        assert_eq!(
            err(Dataset { visible: 3, rows: vec![vec![1, 0, 1]] }),
            Error::NotASpin { row: 0, at: 1, value: 0 }
        );
        assert_eq!(
            err(Dataset { visible: 9, rows: vec![vec![1; 9]] }),
            Error::TooSmall { spins: 5, visible: 9 }
        );
    }

    #[test]
    fn an_enumeration_too_large_is_refused_rather_than_attempted() {
        let g = rbm(20, 10);
        let d = Dataset { visible: 20, rows: vec![vec![1i8; 20]] };
        match exact_log_likelihood(&g, &d) {
            Err(Error::TooLarge { spins, limit }) => {
                assert_eq!((spins, limit), (30, MAX_ENUMERATED));
                let msg = Error::TooLarge { spins, limit }.to_string();
                // The message must say the model is too LARGE and name the limit. It used to be
                // Error::TooSmall, whose text -- "the model has 28 spins and the data needs 20
                // visible" -- is true, irrelevant, and says the opposite of what went wrong.
                assert!(msg.contains("30 spins"), "names the size: {msg}");
                assert!(msg.contains(&limit.to_string()), "and the limit: {msg}");
                assert!(msg.contains("refused"), "and that it refused rather than estimated: {msg}");
            }
            other => panic!("an oversized model must be TooLarge, got {other:?}"),
        }
    }
    /// A dataset drawn exactly from a known model, by inverse CDF over the enumerated distribution.
    fn draw_from(g: &Graph, rows: usize, seed: u64) -> Dataset {
        let p = crate::ising::exact_boltzmann(g, 1.0);
        let mut cdf = Vec::with_capacity(p.len());
        let mut acc = 0.0;
        for &x in &p {
            acc += x;
            cdf.push(acc);
        }
        let mut rng = crate::rng::Pcg::new(seed, 11);
        let rows = (0..rows)
            .map(|_| {
                let u = rng.f64() * acc;
                let m = cdf.partition_point(|&c| c < u).min(p.len() - 1);
                (0..g.n).map(|i| if (m >> i) & 1 == 1 { 1i8 } else { -1 }).collect()
            })
            .collect();
        Dataset { visible: g.n, rows }
    }

    /// The closed-form gradient is the derivative of the objective it claims to be.
    ///
    /// The sharpest test available for a hand-derived gradient, and the one that catches the error
    /// this derivation invites: an edge weight enters the field of BOTH its endpoints, so its
    /// derivative has two terms. Drop either and the fit still converges to something, just not to
    /// the maximiser of the stated objective — which no accuracy check on the result would reveal.
    #[test]
    fn the_closed_form_gradient_matches_a_finite_difference() {
        let truth = {
            let mut b = GraphBuilder::new(5);
            b.couple(0, 1, 0.7);
            b.couple(1, 2, -0.4);
            b.couple(2, 3, 0.9);
            b.couple(3, 4, -0.6);
            b.couple(0, 4, 0.3);
            b.bias(0, 0.25);
            b.bias(2, -0.5);
            b.build()
        };
        let data = draw_from(&truth, 400, 3);

        let n = truth.n;
        let mut edges: Vec<(usize, usize, f64)> = Vec::new();
        for i in 0..n {
            for k in truth.offset[i]..truth.offset[i + 1] {
                let j = truth.nbr[k] as usize;
                if j > i {
                    edges.push((i, j, truth.w[k]));
                }
            }
        }
        let bias: Vec<f64> = truth.h.clone();
        let build = |e: &[(usize, usize, f64)], b: &[f64]| {
            let mut gb = GraphBuilder::new(n);
            for &(i, j, w) in e {
                gb.couple(i, j, w);
            }
            for (i, &x) in b.iter().enumerate() {
                if x != 0.0 {
                    gb.bias(i, x);
                }
            }
            gb.build()
        };

        let g0 = build(&edges, &bias);
        let (dw, dh) = flip_gradient(&g0, &data, &edges, FlipLoss::Log);
        let eps = 1e-6;

        for e in 0..edges.len() {
            let mut up = edges.clone();
            up[e].2 += eps;
            let mut dn = edges.clone();
            dn[e].2 -= eps;
            let num = (pseudo_log_likelihood(&build(&up, &bias), &data).unwrap()
                - pseudo_log_likelihood(&build(&dn, &bias), &data).unwrap())
                / (2.0 * eps);
            assert!(
                (dw[e] - num).abs() < 1e-5,
                "edge {e} ({}, {}): analytic {:.8} against a finite difference {num:.8}",
                edges[e].0,
                edges[e].1,
                dw[e]
            );
        }
        for i in 0..n {
            let mut up = bias.clone();
            up[i] += eps;
            let mut dn = bias.clone();
            dn[i] -= eps;
            let num = (pseudo_log_likelihood(&build(&edges, &up), &data).unwrap()
                - pseudo_log_likelihood(&build(&edges, &dn), &data).unwrap())
                / (2.0 * eps);
            assert!(
                (dh[i] - num).abs() < 1e-5,
                "bias {i}: analytic {:.8} against a finite difference {num:.8}",
                dh[i]
            );
        }
    }

    /// The objective rises every epoch, which a concave objective under ascent must.
    #[test]
    fn the_pseudolikelihood_never_falls_during_training() {
        let truth = crate::ising::ring(6, 0.8, 0.3);
        let data = draw_from(&truth, 300, 5);
        let blank = crate::ising::ring(6, 0.0, 0.0);
        let p = PlParams { epochs: 1, lr: 0.05, l2: 0.0 };

        let mut model = blank;
        let mut prev = f64::NEG_INFINITY;
        for step in 0..80 {
            let t = train_pseudolikelihood(&model, &data, &p).unwrap();
            let now = pseudo_log_likelihood(&t.graph, &data).unwrap();
            assert!(
                now >= prev - 1e-9,
                "step {step}: the objective fell from {prev:.8} to {now:.8}"
            );
            prev = now;
            model = t.graph;
        }
    }

    /// The fit converges on the model that generated the data, and more data gets closer.
    ///
    /// Consistency is the property pseudolikelihood is chosen FOR — it is not the likelihood, so
    /// the case for it rests entirely on the maximiser going to the right place as data grows.
    /// Asserting a single error threshold would test the fixture; asserting the error SHRINKS tests
    /// the claim.
    #[test]
    fn pseudolikelihood_converges_on_the_model_that_generated_the_data() {
        let truth = {
            let mut b = GraphBuilder::new(6);
            for i in 0..6 {
                b.couple(i, (i + 1) % 6, if i % 2 == 0 { 0.6 } else { -0.5 });
            }
            b.bias(0, 0.4);
            b.bias(3, -0.3);
            b.build()
        };
        let err = |rows: usize, seed: u64| -> f64 {
            let data = draw_from(&truth, rows, seed);
            let blank = {
                let mut b = GraphBuilder::new(6);
                for i in 0..6 {
                    b.couple(i, (i + 1) % 6, 0.0);
                }
                b.build()
            };
            let t = train_pseudolikelihood(
                &blank,
                &data,
                &PlParams { epochs: 400, lr: 0.1, l2: 0.0 },
            )
            .unwrap();
            // Mean absolute parameter error, over edges and biases alike.
            let mut sum = 0.0;
            let mut cnt = 0.0;
            for i in 0..truth.n {
                for k in truth.offset[i]..truth.offset[i + 1] {
                    let j = truth.nbr[k] as usize;
                    if j > i {
                        let got = (t.graph.offset[i]..t.graph.offset[i + 1])
                            .find(|&x| t.graph.nbr[x] as usize == j)
                            .map_or(0.0, |x| t.graph.w[x]);
                        sum += (got - truth.w[k]).abs();
                        cnt += 1.0;
                    }
                }
                sum += (t.graph.h[i] - truth.h[i]).abs();
                cnt += 1.0;
            }
            sum / cnt
        };

        let seeds = 4u64;
        let mean = |rows: usize| (0..seeds).map(|s| err(rows, s)).sum::<f64>() / seeds as f64;
        let (small, large) = (mean(150), mean(3_000));
        assert!(
            large < small * 0.5,
            "twenty times the data should at least halve the parameter error: {small:.4} at 150 \
             rows against {large:.4} at 3000"
        );
        assert!(large < 0.1, "and the fit should land near the truth: {large:.4}");
    }

    /// A model with latent units is refused by name, because the conditional does not exist.
    #[test]
    fn pseudolikelihood_refuses_a_model_with_hidden_units() {
        let structure = rbm(4, 3);
        let data = Dataset { visible: 4, rows: vec![vec![1, -1, 1, -1]; 8] };
        match train_pseudolikelihood(&structure, &data, &PlParams::default()) {
            Err(Error::HasLatent { spins, visible }) => {
                assert_eq!((spins, visible), (7, 4));
            }
            other => panic!("hidden units make the conditional uncomputable: {other:?}"),
        }
        assert!(matches!(
            pseudo_log_likelihood(&structure, &data),
            Err(Error::HasLatent { .. })
        ));
    }


    /// EVERY ONE OF THE THREE LOSSES IS DECREASING IN THE FLIP MARGIN. That is the property that
    /// makes all three of them fits rather than loops: the margin is how firmly the model holds a
    /// data point down against its own one-bit neighbours, and a loss that rose with it would
    /// climb away from the data while every convergence check said the objective was improving.
    ///
    /// The slope is also checked to BE the derivative of the score rather than merely to share its
    /// sign, because the two are stored as separate expressions in [`FlipLoss`] and a mismatch
    /// between them is a gradient that is uphill on a different function than the one reported.
    #[test]
    fn every_flip_loss_pushes_the_margin_up() {
        for loss in [FlipLoss::Log, FlipLoss::Flow, FlipLoss::Ratio] {
            let mut prev = f64::NEG_INFINITY;
            let mut u = -8.0;
            while u <= 8.0 {
                let s = loss.score(u);
                assert!(s > prev, "{loss:?}: score fell to {s} from {prev} at u={u}");
                assert!(loss.slope(u) > 0.0, "{loss:?}: slope {} at u={u}", loss.slope(u));
                let h = 1e-5;
                let fd = (loss.score(u + h) - loss.score(u - h)) / (2.0 * h);
                let tol = 1e-5 + 1e-4 * loss.slope(u).abs();
                assert!(
                    (fd - loss.slope(u)).abs() < tol,
                    "{loss:?} at u={u}: slope {} against a difference of {fd}",
                    loss.slope(u)
                );
                prev = s;
                u += 0.25;
            }
        }
    }

    /// The closed-form gradient of all three flip objectives, against a central difference of the
    /// objective itself.
    ///
    /// This is the check that catches the failure the shared code path makes easy: an edge appears
    /// in the field of BOTH its endpoints and so contributes two terms, and keeping only one still
    /// gives a direction that goes uphill — at half the rate on one end, which no convergence test
    /// and no likelihood curve would notice. It is differenced AWAY FROM THE TRUTH, because at the
    /// optimum the gradient is near zero and a broken gradient agrees with zero as well as a
    /// correct one does.
    #[test]
    fn every_flip_gradient_matches_a_finite_difference() {
        let truth = {
            let mut b = GraphBuilder::new(5);
            b.couple(0, 1, 0.7);
            b.couple(1, 2, -0.4);
            b.couple(2, 3, 0.9);
            b.couple(3, 4, -0.6);
            b.couple(0, 4, 0.3);
            b.bias(0, 0.25);
            b.bias(2, -0.5);
            b.build()
        };
        let data = draw_from(&truth, 400, 3);
        let mut wt = Weights::of(&truth);
        for x in &mut wt.edges {
            x.2 *= 0.4;
        }
        for b in &mut wt.bias {
            *b += 0.3;
        }

        let eps = 1e-6;
        for loss in [FlipLoss::Log, FlipLoss::Flow, FlipLoss::Ratio] {
            let (dw, dh) = flip_gradient(&wt.build(), &data, &wt.edges, loss);
            for e in 0..wt.edges.len() {
                let (mut a, mut b) = (wt.clone(), wt.clone());
                a.edges[e].2 += eps;
                b.edges[e].2 -= eps;
                let fd = (flip_score(&a.build(), &data, loss)
                    - flip_score(&b.build(), &data, loss))
                    / (2.0 * eps);
                assert!(
                    (fd - dw[e]).abs() < 1e-4 * fd.abs().max(1.0),
                    "{loss:?} edge {e}: closed form {} against a difference of {fd}",
                    dw[e]
                );
            }
            for i in 0..wt.n {
                let (mut a, mut b) = (wt.clone(), wt.clone());
                a.bias[i] += eps;
                b.bias[i] -= eps;
                let fd = (flip_score(&a.build(), &data, loss)
                    - flip_score(&b.build(), &data, loss))
                    / (2.0 * eps);
                assert!(
                    (fd - dh[i]).abs() < 1e-4 * fd.abs().max(1.0),
                    "{loss:?} bias {i}: closed form {} against a difference of {fd}",
                    dh[i]
                );
            }
        }
    }

    /// The exact maximum-likelihood gradient against a central difference of the exact likelihood.
    ///
    /// Run on a model WITH LATENT UNITS as well as without, because the two exercise different
    /// halves of the function and only the latent case exercises the harder one: the positive
    /// phase there is not the data's correlations but an average over the states agreeing with
    /// each row, weighted by the model's own conditional. Getting that wrong gives a gradient that
    /// is exactly right whenever `visible == n` — which is every other test in this module.
    #[test]
    fn the_exact_gradient_matches_a_finite_difference_of_the_true_likelihood() {
        // Fully visible: a five-spin ring with chords.
        let visible = {
            let mut b = GraphBuilder::new(5);
            b.couple(0, 1, 0.7);
            b.couple(1, 2, -0.4);
            b.couple(2, 3, 0.9);
            b.couple(3, 4, -0.6);
            b.couple(0, 4, 0.3);
            b.bias(0, 0.25);
            b.bias(2, -0.5);
            b.build()
        };
        let vdata = draw_from(&visible, 400, 3);

        // Latent: four visible units, three hidden, at weights that are not all equal — a model at
        // uniform weights has a symmetry that would hide an index error in the completion loop.
        let mut lat = Weights::of(&rbm(4, 3));
        for (k, x) in lat.edges.iter_mut().enumerate() {
            x.2 = 0.15 * (k % 5) as f64 - 0.3;
        }
        for (i, b) in lat.bias.iter_mut().enumerate() {
            *b = 0.1 * i as f64 - 0.3;
        }
        let ldata = bars_and_stripes(2);

        let eps = 1e-6;
        for (what, wt, data) in [
            ("fully visible", Weights::of(&visible), &vdata),
            ("with latent units", lat, &ldata),
        ] {
            let (dw, dh) = exact_gradient(&wt.build(), data, &wt.edges);
            for e in 0..wt.edges.len() {
                let (mut a, mut b) = (wt.clone(), wt.clone());
                a.edges[e].2 += eps;
                b.edges[e].2 -= eps;
                let fd = (exact_log_likelihood(&a.build(), data).unwrap()
                    - exact_log_likelihood(&b.build(), data).unwrap())
                    / (2.0 * eps);
                assert!(
                    (fd - dw[e]).abs() < 1e-4 * fd.abs().max(1.0),
                    "{what} edge {e}: closed form {} against a difference of {fd}",
                    dw[e]
                );
            }
            for i in 0..wt.n {
                let (mut a, mut b) = (wt.clone(), wt.clone());
                a.bias[i] += eps;
                b.bias[i] -= eps;
                let fd = (exact_log_likelihood(&a.build(), data).unwrap()
                    - exact_log_likelihood(&b.build(), data).unwrap())
                    / (2.0 * eps);
                assert!(
                    (fd - dh[i]).abs() < 1e-4 * fd.abs().max(1.0),
                    "{what} bias {i}: closed form {} against a difference of {fd}",
                    dh[i]
                );
            }
        }
    }

    /// THE FIXED POINT OF EXACT MAXIMUM LIKELIHOOD IS MOMENT MATCHING, with both sides exact.
    ///
    /// `a_fully_visible_fit_matches_the_data_correlations` makes the same claim for contrastive
    /// divergence and has to allow a wide tolerance, because CD's negative phase is one sample per
    /// row and the parameters random-walk around the optimum. Here neither average is sampled, so
    /// the residual is bounded by the step size and nothing else, and the assertion can be three
    /// orders of magnitude tighter — which is the point of having the exact method at all.
    #[test]
    fn the_exact_fit_matches_the_data_moments_exactly() {
        let truth = {
            let mut b = GraphBuilder::new(6);
            for i in 0..6 {
                b.couple(i, (i + 1) % 6, if i % 2 == 0 { 0.6 } else { -0.5 });
            }
            b.couple(0, 3, 0.4);
            b.bias(0, 0.4);
            b.bias(3, -0.3);
            b.build()
        };
        let data = draw_from(&truth, 2_000, 5);
        let start = {
            let mut b = GraphBuilder::new(6);
            for i in 0..6 {
                b.couple(i, (i + 1) % 6, 0.0);
            }
            b.couple(0, 3, 0.0);
            b.build()
        };
        let t = train_exact(&start, &data, &FitParams { epochs: 4_000, lr: 0.2, l2: 0.0 }).unwrap();

        // The gradient IS the moment difference, so its size is the moment mismatch in the units
        // the claim is made in.
        let wt = Weights::of(&t.graph);
        let (dw, dh) = exact_gradient(&t.graph, &data, &wt.edges);
        let worst = dw.iter().chain(&dh).fold(0.0f64, |m, x| m.max(x.abs()));
        assert!(worst < 1e-6, "the exact fit should match every moment: worst residual {worst:e}");
    }

    /// NOTHING BEATS MAXIMUM LIKELIHOOD, and that is what makes it a ceiling rather than a sixth
    /// row in the table.
    ///
    /// Both halves are asserted and they fail in opposite directions. That no method passes the
    /// ceiling is a check on `train_exact`: a broken exact gradient would ascend somewhere lower
    /// and one of the five would step over it. That every method gets reasonably CLOSE to it is a
    /// check on the ceiling being real rather than a runaway — a `train_exact` that diverged would
    /// report a huge likelihood no one came near, and the first assertion alone would pass.
    #[test]
    fn exact_maximum_likelihood_is_a_ceiling_the_other_methods_do_not_pass() {
        let truth = {
            let mut b = GraphBuilder::new(8);
            for i in 0..8 {
                b.couple(i, (i + 1) % 8, if i % 2 == 0 { 0.8 } else { -0.6 });
            }
            b.couple(0, 4, 0.5);
            b.bias(0, 0.4);
            b.bias(4, -0.35);
            b.build()
        };
        let data = draw_from(&truth, 2_000, 3);
        let start = {
            let mut b = GraphBuilder::new(8);
            for i in 0..8 {
                b.couple(i, (i + 1) % 8, 0.0);
            }
            b.couple(0, 4, 0.0);
            b.build()
        };

        let ep = 1_500;
        let ceiling = train_exact(&start, &data, &FitParams { epochs: ep, lr: 0.2, l2: 0.0 })
            .unwrap()
            .log_likelihood
            .unwrap();
        let floor = exact_log_likelihood(&start, &data).unwrap();

        let cd = Params {
            epochs: ep,
            k: 10,
            positive_sweeps: 1,
            learning_rate: 0.02,
            batch: 32,
            persistent: false,
        };
        let rivals = [
            ("CD-10", train(&start, &data, &cd, 7).unwrap()),
            (
                "PCD-10",
                train(&start, &data, &Params { persistent: true, ..cd }, 7).unwrap(),
            ),
            (
                "pseudolikelihood",
                train_pseudolikelihood(&start, &data, &FitParams { epochs: ep, lr: 0.1, l2: 0.0 })
                    .unwrap(),
            ),
            (
                "min prob flow",
                train_mpf(&start, &data, &FitParams { epochs: ep, lr: 0.02, l2: 0.0 }).unwrap(),
            ),
            (
                "ratio matching",
                train_ratio_matching(&start, &data, &FitParams { epochs: ep, lr: 0.2, l2: 0.0 })
                    .unwrap(),
            ),
        ];
        for (name, t) in &rivals {
            let v = t.log_likelihood.unwrap();
            assert!(
                v <= ceiling + 1e-9,
                "{name} reached {v} above the maximum-likelihood ceiling {ceiling}"
            );
            let reached = (v - floor) / (ceiling - floor);
            assert!(
                reached > 0.9,
                "{name} reached {:.1}% of a ceiling nothing gets near, so the ceiling is suspect: \
                 {v} against {ceiling} from {floor}",
                100.0 * reached
            );
        }
    }

    /// CONSISTENCY IS WHAT THE SAMPLER-FREE METHODS ARE CHOSEN FOR. None of the three is the
    /// likelihood, so the case for each rests entirely on its maximiser going to the right place
    /// as data grows. Asserting one error threshold would test the fixture; asserting the error
    /// SHRINKS with data tests the claim — the same shape as the pseudolikelihood test above,
    /// extended to the two methods that arrived with it.
    #[test]
    fn every_sampler_free_method_converges_on_the_model_that_generated_the_data() {
        let truth = {
            let mut b = GraphBuilder::new(6);
            for i in 0..6 {
                b.couple(i, (i + 1) % 6, if i % 2 == 0 { 0.6 } else { -0.5 });
            }
            b.bias(0, 0.4);
            b.bias(3, -0.3);
            b.build()
        };
        let blank = || {
            let mut b = GraphBuilder::new(6);
            for i in 0..6 {
                b.couple(i, (i + 1) % 6, 0.0);
            }
            b.build()
        };
        // Each at a step suited to its own loss: `exp(-u)` is the steepest of the three at a bad
        // margin and `sigma(-2u)^2` the flattest, so one shared step would be a test of which
        // loss that number happens to suit.
        /// A trainer, by the signature the three sampler-free ones share — which is the point
        /// being tested, so it is written down rather than inlined.
        type Fit = fn(&Graph, &Dataset, &FitParams) -> Result<Trained, Error>;
        let methods: [(&str, Fit, f64); 3] = [
            ("pseudolikelihood", train_pseudolikelihood, 0.1),
            ("min prob flow", train_mpf, 0.02),
            ("ratio matching", train_ratio_matching, 0.4),
        ];
        for (name, fit, lr) in methods {
            let err = |rows: usize, seed: u64| -> f64 {
                let data = draw_from(&truth, rows, seed);
                let t = fit(&blank(), &data, &FitParams { epochs: 800, lr, l2: 0.0 }).unwrap();
                let mut sum = 0.0;
                let mut cnt = 0.0;
                for i in 0..truth.n {
                    for k in truth.offset[i]..truth.offset[i + 1] {
                        let j = truth.nbr[k] as usize;
                        if j > i {
                            let got = (t.graph.offset[i]..t.graph.offset[i + 1])
                                .find(|&x| t.graph.nbr[x] as usize == j)
                                .map_or(0.0, |x| t.graph.w[x]);
                            sum += (got - truth.w[k]).abs();
                            cnt += 1.0;
                        }
                    }
                    sum += (t.graph.h[i] - truth.h[i]).abs();
                    cnt += 1.0;
                }
                sum / cnt
            };
            let seeds = 4u64;
            let mean = |rows: usize| (0..seeds).map(|s| err(rows, s)).sum::<f64>() / seeds as f64;
            let (small, large) = (mean(150), mean(3_000));
            assert!(
                large < small * 0.5,
                "{name}: twenty times the data should at least halve the parameter error, \
                 {small:.4} at 150 rows against {large:.4} at 3000"
            );
            assert!(large < 0.1, "{name}: and the fit should land near the truth, {large:.4}");
        }
    }

    /// The three flip objectives refuse a latent model BY NAME and the exact one accepts it.
    ///
    /// This is the line between the two families and it is not a limitation of the implementation:
    /// a flip margin is a function of the whole state, so a hidden unit leaves it uncomputable at
    /// any cost. The exact gradient integrates the hidden units out under the model's own
    /// conditional, which is the same sum a clamped sampler estimates, and so applies.
    #[test]
    fn the_flip_losses_refuse_hidden_units_and_the_exact_gradient_does_not() {
        let structure = rbm(4, 3);
        let data = Dataset { visible: 4, rows: bars_and_stripes(2).rows };
        for (name, r) in [
            ("mpf", train_mpf(&structure, &data, &FitParams::default())),
            ("ratio matching", train_ratio_matching(&structure, &data, &FitParams::default())),
        ] {
            match r {
                Err(Error::HasLatent { spins, visible }) => assert_eq!((spins, visible), (7, 4)),
                other => panic!("{name} should refuse a latent model by name: {other:?}"),
            }
        }
        assert!(matches!(
            minimum_probability_flow(&structure, &data),
            Err(Error::HasLatent { .. })
        ));
        assert!(matches!(ratio_matching(&structure, &data), Err(Error::HasLatent { .. })));

        // And the exact method fits the same model rather than refusing it, raising the true
        // likelihood as it goes. Started AWAY FROM ZERO WEIGHTS, which is not a convenience —
        // see `an_rbm_at_zero_weights_is_a_stationary_point_of_the_exact_likelihood`.
        let mut wt = Weights::of(&structure);
        for (k, x) in wt.edges.iter_mut().enumerate() {
            x.2 = 0.1 * (k % 7) as f64 - 0.3;
        }
        let seeded = wt.build();
        let before = exact_log_likelihood(&seeded, &data).unwrap();
        let t = train_exact(&seeded, &data, &FitParams { epochs: 400, lr: 0.1, l2: 0.0 })
            .expect("the exact gradient integrates latent units out");
        assert!(
            t.log_likelihood.unwrap() > before + 0.1,
            "the exact fit should learn something: {before} to {:?}",
            t.log_likelihood
        );
    }

    /// AN RBM AT ZERO WEIGHTS IS A STATIONARY POINT OF THE EXACT LIKELIHOOD, and exactly so.
    ///
    /// This is why every recipe for training a Boltzmann machine says to start from small random
    /// weights, and it is usually given as folklore about symmetry breaking. With the exact
    /// gradient in hand it is a two-line argument and an equality rather than a tolerance:
    ///
    /// At zero weights the model is uniform, so every model moment is zero. The conditional over
    /// the hidden units given a data row is ALSO uniform, so `⟨s_h⟩ = 0` and
    /// `⟨s_v s_h⟩ = s_v ⟨s_h⟩ = 0` on the data side too. Every edge of an RBM crosses the
    /// bipartition and every hidden bias multiplies a hidden unit, so every one of those gradients
    /// is zero — whatever the data is. The ONLY component that can move is a visible bias, which
    /// goes to the data's own mean.
    ///
    /// So a bipartite model started at zero learns nothing but the visible means, and on a
    /// complement-symmetric dataset like bars-and-stripes those are zero as well and it learns
    /// nothing at all. Asserted at `== 0.0` and not at a tolerance, because the argument gives
    /// exact zeros and a tolerance would hide an implementation that merely got close.
    #[test]
    fn an_rbm_at_zero_weights_is_a_stationary_point_of_the_exact_likelihood() {
        let structure = rbm(4, 3);
        // Deliberately NOT complement-symmetric, so the visible means are nonzero and the test
        // separates "this gradient is structurally zero" from "this dataset happens to be even".
        let data = Dataset {
            visible: 4,
            rows: vec![
                vec![1, 1, 1, 1],
                vec![1, 1, -1, -1],
                vec![1, -1, 1, -1],
                vec![1, 1, 1, -1],
            ],
        };
        let wt = Weights::of(&structure);
        let (dw, dh) = exact_gradient(&wt.build(), &data, &wt.edges);
        for (e, &(i, j, _)) in wt.edges.iter().enumerate() {
            assert_eq!(dw[e], 0.0, "edge {i}-{j} crosses the bipartition and must be exactly flat");
        }
        for h in data.visible..structure.n {
            assert_eq!(dh[h], 0.0, "hidden bias {h} must be exactly flat");
        }
        // The visible biases are the exception, and they are the data means exactly.
        for v in 0..data.visible {
            let mean = data.rows.iter().map(|r| f64::from(r[v])).sum::<f64>()
                / data.rows.len() as f64;
            assert!((dh[v] - mean).abs() < 1e-12, "visible bias {v}: {} against {mean}", dh[v]);
        }
        assert!(dh[..data.visible].iter().any(|x| x.abs() > 0.1), "the fixture must not be even");

        // And bars-and-stripes IS complement-symmetric, so there the whole gradient vanishes and
        // training is a no-op that no likelihood curve would flag as one.
        let bas = bars_and_stripes(2);
        let flat = train_exact(&structure, &bas, &FitParams { epochs: 200, lr: 0.1, l2: 0.0 })
            .unwrap();
        assert_eq!(
            flat.log_likelihood.unwrap(),
            exact_log_likelihood(&structure, &bas).unwrap(),
            "a bipartite model started at zero cannot move on complement-symmetric data"
        );
    }

    /// THE CLAMPED MEAN FIELD MUST ACTUALLY PIN, and the pinning must be exact rather than a
    /// strong field. A visible unit that drifts by even `1e-9` makes the positive phase an average
    /// over states that disagree with the data, which is a different objective wearing the same
    /// name.
    #[test]
    fn the_clamped_mean_field_pins_what_it_is_told() {
        let g = {
            let mut b = GraphBuilder::new(6);
            for i in 0..5 {
                b.couple(i, i + 1, if i % 2 == 0 { 0.9 } else { -0.7 });
            }
            b.couple(0, 5, 0.5);
            b.bias(0, -2.0); // pulling HARD against the clamp, which must lose
            b.bias(1, 1.5);
            b.build()
        };
        let row = [1i8, 1, -1];
        let mf = crate::meanfield::naive_mean_field_clamped(&g, 1.0, &row, 500, 0.5);
        for (i, &v) in row.iter().enumerate() {
            assert_eq!(mf.m[i], f64::from(v), "spin {i} was not pinned: {}", mf.m[i]);
        }
        // The free ones must have moved off the initial 0.01 and stayed inside [-1, 1].
        for i in row.len()..g.n {
            assert!(mf.m[i].abs() <= 1.0, "magnetisation {i} left the interval: {}", mf.m[i]);
            assert!((mf.m[i] - 0.01).abs() > 1e-6, "free spin {i} never moved");
        }
        assert!(mf.converged(1e-12), "residual {:e}", mf.residual);

        // Every spin pinned: nothing to iterate, and it must SAY it converged rather than spinning
        // to the cap with the initial infinity still in `residual`.
        let all = [1i8, -1, 1, 1, -1, -1];
        let full = crate::meanfield::naive_mean_field_clamped(&g, 1.0, &all, 500, 0.5);
        assert!(full.converged(1e-12), "a fully pinned field is converged by construction");
        assert!(full.iterations <= 1, "and costs no iteration: {}", full.iterations);
        for (i, &v) in all.iter().enumerate() {
            assert_eq!(full.m[i], f64::from(v));
        }
    }

    /// WITH NO HIDDEN UNITS THE VARIATIONAL POSITIVE PHASE IS EXACT, so this is the one case where
    /// the approximation costs nothing and the fixed point must be moment matching.
    ///
    /// Clamping every spin leaves the mean field with nothing to approximate: `μ` IS the data row.
    /// So the positive phase carries neither bias nor sampling noise, and the fit must land where
    /// `a_fully_visible_fit_matches_the_data_correlations` lands — it fails if the means are ever
    /// used where the data should be, or the pinning slips.
    ///
    /// **It is not a SHARPER check than the sampled version, and the first draft of this test said
    /// it was.** The residual here is the NEGATIVE phase's, which both methods share, so an exact
    /// positive phase buys nothing in tolerance: written at `0.05` against one seed it failed at
    /// `0.058`, which is that seed and not a defect. Averaged over seeds instead, because tuning
    /// the seed until it passes is how a fixture gets fitted to its answer.
    #[test]
    fn a_fully_visible_variational_fit_has_an_exact_positive_phase() {
        let rows: Vec<Vec<i8>> = vec![
            vec![1, 1, 1],
            vec![1, 1, 1],
            vec![1, 1, -1],
            vec![-1, -1, 1],
            vec![-1, -1, 1],
            vec![-1, -1, -1],
        ];
        let data = Dataset { visible: 3, rows: rows.clone() };
        let mut gb = GraphBuilder::new(3);
        gb.couple(0, 1, 0.0);
        gb.couple(0, 2, 0.0);
        gb.couple(1, 2, 0.0);
        let structure = gb.build();

        let p = Params {
            epochs: 4_000,
            k: 20,
            positive_sweeps: 20,
            learning_rate: 0.05,
            batch: 6,
            persistent: false,
        };
        let m = rows.len() as f64;
        let dc = |i: usize, j: usize| rows.iter().map(|r| f64::from(r[i] * r[j])).sum::<f64>() / m;
        let dm = |i: usize| rows.iter().map(|r| f64::from(r[i])).sum::<f64>() / m;
        // Model moments by ENUMERATION, not by more sampling: a check that compares a sampler's
        // average against a sampler's average agrees with itself whatever it is doing.
        let spin = |mask: usize, i: usize| if (mask >> i) & 1 == 1 { 1.0 } else { -1.0 };
        let seeds = 5u64;
        let fits: Vec<Vec<f64>> = (0..seeds)
            .map(|sd| crate::ising::exact_boltzmann(&train_variational(&structure, &data, &p, sd).unwrap().graph, 1.0))
            .collect();
        let mc = |i: usize, j: usize| {
            fits.iter()
                .map(|pr| pr.iter().enumerate().map(|(k, &q)| q * spin(k, i) * spin(k, j)).sum::<f64>())
                .sum::<f64>()
                / f64::from(u32::try_from(seeds).unwrap())
        };
        let mm = |i: usize| {
            fits.iter()
                .map(|pr| pr.iter().enumerate().map(|(k, &q)| q * spin(k, i)).sum::<f64>())
                .sum::<f64>()
                / f64::from(u32::try_from(seeds).unwrap())
        };
        for (i, j) in [(0usize, 1usize), (0, 2), (1, 2)] {
            assert!(
                (mc(i, j) - dc(i, j)).abs() < 0.05,
                "edge {i}-{j}: model {:.4} against data {:.4}",
                mc(i, j),
                dc(i, j)
            );
        }
        for i in 0..3 {
            assert!((mm(i) - dm(i)).abs() < 0.05, "bias {i}: model {:.4} against data {:.4}", mm(i), dm(i));
        }
    }

    /// The deep machine the method exists for: it must raise the TRUE likelihood, measured by
    /// enumeration rather than by the objective it ascends.
    ///
    /// Started away from zero weights, which is not a convenience — see
    /// `an_rbm_at_zero_weights_is_a_stationary_point_of_the_exact_likelihood`. The saddle there is
    /// a statement about the EXACT gradient; a sampled negative phase escapes it because the
    /// fantasy chains start at random states, and this test does not rely on either fact.
    #[test]
    fn the_variational_positive_phase_trains_a_deep_machine() {
        let structure = dbm(4, &[3, 2]);
        let data = bars_and_stripes(2);
        let mut wt = Weights::of(&structure);
        for (k, x) in wt.edges.iter_mut().enumerate() {
            x.2 = 0.1 * ((k % 7) as f64 - 3.0) / 3.0;
        }
        let start = wt.build();
        let before = exact_log_likelihood(&start, &data).unwrap();

        let p = Params {
            epochs: 400,
            k: 10,
            positive_sweeps: 100,
            learning_rate: 0.05,
            batch: 6,
            persistent: true,
        };
        let t = train_variational(&start, &data, &p, 7).unwrap();
        let after = t.log_likelihood.expect("nine spins is inside the enumeration limit");
        assert!(after > before + 0.3, "the variational fit should learn: {before:.4} to {after:.4}");
        assert_eq!(t.graph.n_edges, structure.n_edges, "only weights move, never the edge set");
    }

    /// Enumeration is refused past the size it is affordable at, rather than attempted.
    #[test]
    fn training_by_enumeration_is_refused_past_the_limit() {
        let structure = rbm(MAX_ENUMERATED, 1);
        let data = Dataset { visible: MAX_ENUMERATED, rows: vec![vec![1i8; MAX_ENUMERATED]; 4] };
        match train_exact(&structure, &data, &FitParams::default()) {
            Err(Error::TooLarge { spins, limit }) => {
                assert_eq!((spins, limit), (MAX_ENUMERATED + 1, MAX_ENUMERATED));
            }
            other => panic!("past the limit it should refuse rather than run: {other:?}"),
        }
    }

    // ---- a deterministic negative phase: TAP, Bethe and mean field as TRAINERS ----------------

    /// Rebuild a model from a flat parameter list, for finite differencing.
    fn from_params(n: usize, edges: &[(usize, usize, f64)], bias: &[f64]) -> Graph {
        let mut gb = GraphBuilder::new(n);
        for &(i, j, w) in edges {
            gb.couple(i, j, w);
        }
        for (i, &b) in bias.iter().enumerate() {
            gb.bias(i, b);
        }
        gb.build()
    }

    /// A ring with chords: loopy, so no approximation here is exact.
    fn chorded_ring(n: usize, j: f64, seed: u64) -> Graph {
        let mut rng = Pcg::new(seed, 0);
        let mut gb = GraphBuilder::new(n);
        for i in 0..n {
            gb.couple(i, (i + 1) % n, j * (2.0 * rng.f64() - 1.0));
        }
        for i in 0..n / 2 {
            gb.couple(i, i + n / 2, j * (2.0 * rng.f64() - 1.0));
        }
        for i in 0..n {
            gb.bias(i, 0.4 * (rng.f64() - 0.5));
        }
        gb.build()
    }

    fn random_tree_model(n: usize, j: f64, seed: u64) -> Graph {
        let mut rng = Pcg::new(seed, 0);
        let mut gb = GraphBuilder::new(n);
        for i in 1..n {
            let parent = (rng.f64() * i as f64) as usize;
            gb.couple(parent, i, j * (2.0 * rng.f64() - 1.0));
        }
        for i in 0..n {
            gb.bias(i, 0.5 * (rng.f64() - 0.5));
        }
        gb.build()
    }

    /// Rows drawn EXACTLY from a model, by inverting the enumerated Boltzmann CDF.
    fn draw_rows(g: &Graph, rows: usize, seed: u64) -> Vec<Vec<i8>> {
        let p = crate::ising::exact_boltzmann(g, 1.0);
        let mut rng = Pcg::new(seed, 0x0D_1234);
        (0..rows)
            .map(|_| {
                let u = rng.f64();
                let mut acc = 0.0;
                let mut pick = p.len() - 1;
                for (k, &q) in p.iter().enumerate() {
                    acc += q;
                    if u <= acc {
                        pick = k;
                        break;
                    }
                }
                (0..g.n).map(|i| if pick >> i & 1 == 1 { 1i8 } else { -1 }).collect()
            })
            .collect()
    }

    /// Exact `⟨s_i⟩` and `⟨s_i s_j⟩` by enumeration, the latter indexed by CSR slot.
    fn exact_moments(g: &Graph) -> (Vec<f64>, Vec<f64>) {
        let p = crate::ising::exact_boltzmann(g, 1.0);
        let mut m = vec![0.0; g.n];
        let mut pair = vec![0.0; g.nbr.len()];
        for (mask, &q) in p.iter().enumerate() {
            let s = |i: usize| if mask >> i & 1 == 1 { 1.0 } else { -1.0 };
            for i in 0..g.n {
                m[i] += q * s(i);
            }
            for i in 0..g.n {
                for e in g.offset[i]..g.offset[i + 1] {
                    pair[e] += q * s(i) * s(g.nbr[e] as usize);
                }
            }
        }
        (m, pair)
    }

    /// THE MOMENTS ARE THE DERIVATIVE OF THE FREE ENERGY, and that identity is what makes a
    /// deterministic negative phase an ascent direction rather than a plausible-looking number.
    ///
    /// Each of the three fixed points is a stationary point of its own free energy, so the fixed
    /// point's motion contributes nothing to the total derivative and `∂ ln Z/∂h_i = m_i`,
    /// `∂ ln Z/∂J_ij = ⟨s_i s_j⟩`. Checked by CENTRAL DIFFERENCES of `meanfield`'s own `log_z`,
    /// re-solving the fixed point on each side — which is an algebraic identity with an exact
    /// answer, not a comparison of one approximation against another. It pins the Onsager pair
    /// term of TAP and the reconstructed Bethe pair belief in one stroke: drop either and the
    /// edge derivative is wrong while the site derivative stays right.
    #[test]
    fn the_moments_are_the_derivative_of_the_free_energy() {
        let g0 = chorded_ring(8, 0.35, 4);
        let wt = Weights::of(&g0);
        let s = Solve { iters: 20_000, damping: 0.5 };
        let eps = 1e-5;
        for which in [Moments::MeanField, Moments::Tap, Moments::Bethe] {
            let base = model_moments(&g0, which, &s);
            assert!(base.residual < 1e-12, "{which:?} did not converge: {}", base.residual);

            for (k, &(i, j, w)) in wt.edges.iter().enumerate() {
                let mut e = wt.edges.clone();
                e[k].2 = w + eps;
                let up = model_moments(&from_params(g0.n, &e, &wt.bias), which, &s).log_z;
                e[k].2 = w - eps;
                let dn = model_moments(&from_params(g0.n, &e, &wt.bias), which, &s).log_z;
                let fd = (up - dn) / (2.0 * eps);
                let got = base.pair[slot_of(&g0, i, j)];
                assert!(
                    (fd - got).abs() < 2e-7,
                    "{which:?} d(lnZ)/dJ_{i}{j}: finite difference {fd:.12} against moment {got:.12}"
                );
            }
            for i in 0..g0.n {
                let mut b = wt.bias.clone();
                b[i] = wt.bias[i] + eps;
                let up = model_moments(&from_params(g0.n, &wt.edges, &b), which, &s).log_z;
                b[i] = wt.bias[i] - eps;
                let dn = model_moments(&from_params(g0.n, &wt.edges, &b), which, &s).log_z;
                let fd = (up - dn) / (2.0 * eps);
                assert!(
                    (fd - base.m[i]).abs() < 2e-7,
                    "{which:?} d(lnZ)/dh_{i}: finite difference {fd:.12} against moment {:.12}",
                    base.m[i]
                );
            }
        }
    }

    /// On a TREE the Bethe moments are exact — both of them — against enumeration.
    ///
    /// The marginals are `meanfield`'s and already pinned there; what is new is the pair belief
    /// reconstructed from two marginals and a coupling, which nothing else in the crate computes.
    /// Mean field and TAP are checked on the same trees to confirm the test can fail: if the tree
    /// were so weakly coupled that everything matched, this would prove nothing.
    #[test]
    fn bethe_pair_beliefs_are_exact_on_a_tree() {
        let s = Solve { iters: 4000, damping: 0.0 };
        for seed in 0..6u64 {
            let g = random_tree_model(10, 0.8, seed);
            let (m_true, pair_true) = exact_moments(&g);
            let bethe = model_moments(&g, Moments::Bethe, &s);
            for i in 0..g.n {
                assert!((bethe.m[i] - m_true[i]).abs() < 1e-9, "seed {seed} site {i}");
            }
            for e in 0..g.nbr.len() {
                assert!(
                    (bethe.pair[e] - pair_true[e]).abs() < 1e-9,
                    "seed {seed} slot {e}: Bethe {} against exact {}",
                    bethe.pair[e],
                    pair_true[e]
                );
            }
            let mf = model_moments(&g, Moments::MeanField, &Solve { iters: 4000, damping: 0.5 });
            let tap = model_moments(&g, Moments::Tap, &Solve { iters: 4000, damping: 0.5 });
            let worst = |a: &[f64]| (0..g.nbr.len()).map(|e| (a[e] - pair_true[e]).abs()).fold(0.0, f64::max);
            assert!(worst(&mf.pair) > 1e-3, "seed {seed}: a tree this weak proves nothing");
            assert!(worst(&tap.pair) > 1e-4, "seed {seed}: a tree this weak proves nothing");
        }
    }

    /// So the Bethe gradient IS the exact gradient on a tree, and `train_bethe` reproduces
    /// `train_exact` step for step.
    ///
    /// Not "close to": the two run the same ascent from the same start with gradients that agree
    /// analytically, so the fitted weights must agree to rounding. This is the strongest oracle
    /// available for a trainer whose negative phase is an approximation — on the family where the
    /// approximation is not one.
    #[test]
    fn bethe_learning_is_exact_maximum_likelihood_on_a_tree() {
        let truth = random_tree_model(9, 0.7, 21);
        let data = Dataset { visible: 9, rows: draw_rows(&truth, 400, 5) };
        let mut structure = Weights::of(&truth);
        for x in &mut structure.edges {
            x.2 = 0.0;
        }
        structure.bias.fill(0.0);
        let structure = structure.build();

        let p = FitParams { epochs: 300, lr: 0.1, l2: 0.0 };
        let exact = train_exact(&structure, &data, &p).unwrap();
        let bethe = train_bethe(&structure, &data, &p, &Solve { iters: 4000, damping: 0.0 }).unwrap();

        let we = Weights::of(&exact.graph);
        let wb = Weights::of(&bethe.graph);
        for (a, b) in we.edges.iter().zip(&wb.edges) {
            assert!((a.2 - b.2).abs() < 1e-7, "edge {a:?} against {b:?}");
        }
        for (a, b) in we.bias.iter().zip(&wb.bias) {
            assert!((a - b).abs() < 1e-7, "bias {a} against {b}");
        }
        assert!(
            (exact.log_likelihood.unwrap() - bethe.log_likelihood.unwrap()).abs() < 1e-9,
            "exact {:?} against Bethe {:?}",
            exact.log_likelihood,
            bethe.log_likelihood
        );
    }

    /// With no edges every approximation is exact, and so is the closed form: `h_i = atanh(⟨s_i⟩)`.
    ///
    /// The degenerate case is worth a test because it is the one place all three estimators have
    /// the same right answer, so it separates a broken pair term from a broken loop.
    #[test]
    fn an_edgeless_model_is_exact_for_every_approximation() {
        let n = 6;
        let structure = GraphBuilder::new(n).build();
        // Means strictly inside (-1, 1) and different at every site: 1/2, 2/3, ... 6/7 of the
        // rows are +1, so `atanh` has an answer and it is a different one per spin.
        let rows: Vec<Vec<i8>> = (0..420usize)
            .map(|r| (0..n).map(|i| if (r + i) % (i + 2) == 0 { -1i8 } else { 1 }).collect())
            .collect();
        let data = Dataset { visible: n, rows: rows.clone() };
        let means: Vec<f64> = (0..n)
            .map(|i| rows.iter().map(|r| f64::from(r[i])).sum::<f64>() / rows.len() as f64)
            .collect();

        let p = FitParams { epochs: 4000, lr: 0.2, l2: 0.0 };
        for which in [Moments::MeanField, Moments::Tap, Moments::Bethe] {
            let t = train_free_energy(&structure, &data, &p, which, &Solve::default()).unwrap();
            for i in 0..n {
                let want = means[i].atanh();
                assert!(
                    (t.graph.h[i] - want).abs() < 1e-6,
                    "{which:?} bias {i}: {} against atanh(m) {want}",
                    t.graph.h[i]
                );
            }
        }
    }

    /// Pinning by graph surgery is the EXACT conditional, which is what lets latent units use the
    /// same approximation unchanged.
    ///
    /// A clamped spin is deleted and its coupling pushed into each neighbour's bias. The claim is
    /// that the surgery leaves `p(free | visible = row)` untouched, and an energy is only defined
    /// up to a constant, so what must match is every energy DIFFERENCE over the free spins.
    #[test]
    fn clamping_by_graph_surgery_is_the_exact_conditional() {
        let g = chorded_ring(10, 0.9, 3);
        let fixed = 4;
        for r in 0..8usize {
            let row: Vec<i8> = (0..fixed).map(|i| if (r >> i) & 1 == 1 { 1i8 } else { -1 }).collect();
            let gc = clamped_graph(&g, &row);
            let free = g.n - fixed;
            let mut ref0 = f64::NAN;
            let mut cut0 = f64::NAN;
            for c in 0..(1usize << free) {
                let mut full = vec![0i8; g.n];
                full[..fixed].copy_from_slice(&row);
                let mut cut = vec![1i8; g.n];
                for k in 0..free {
                    let v = if (c >> k) & 1 == 1 { 1i8 } else { -1 };
                    full[fixed + k] = v;
                    cut[fixed + k] = v;
                }
                let (a, b) = (g.energy(&full), gc.energy(&cut));
                if c == 0 {
                    ref0 = a;
                    cut0 = b;
                }
                assert!(
                    ((a - ref0) - (b - cut0)).abs() < 1e-11,
                    "row {r} state {c}: full ΔE {} against clamped ΔE {}",
                    a - ref0,
                    b - cut0
                );
            }
        }
    }

    /// The fixed point of a deterministic fit is moment matching UNDER ITS OWN ESTIMATOR.
    ///
    /// The gradient is `⟨·⟩_data − ⟨·⟩_TAP`, so where it vanishes the TAP moments equal the data's
    /// exactly — including the Onsager term, which is the only reason the fitted weights differ
    /// from the mean-field fit's. Checked directly rather than by watching a loss go down.
    #[test]
    fn a_deterministic_fit_matches_the_data_moments_under_its_own_estimator() {
        let truth = chorded_ring(8, 0.5, 9);
        let data = Dataset { visible: 8, rows: draw_rows(&truth, 512, 11) };
        let mut structure = Weights::of(&truth);
        for x in &mut structure.edges {
            x.2 = 0.0;
        }
        structure.bias.fill(0.0);
        let structure = structure.build();

        let s = Solve { iters: 5000, damping: 0.5 };
        let t = train_tap(&structure, &data, &FitParams { epochs: 4000, lr: 0.1, l2: 0.0 }, &s).unwrap();
        let wt = Weights::of(&t.graph);
        let (dm, dp) = data_moments(t.graph.n, &data, &wt.edges);
        let got = model_moments(&t.graph, Moments::Tap, &s);
        for i in 0..t.graph.n {
            assert!((dm[i] - got.m[i]).abs() < 1e-4, "site {i}: data {} against TAP {}", dm[i], got.m[i]);
        }
        for (k, &(i, j, _)) in wt.edges.iter().enumerate() {
            let a = got.pair[slot_of(&t.graph, i, j)];
            assert!((dp[k] - a).abs() < 1e-4, "edge {i}-{j}: data {} against TAP {}", dp[k], a);
        }
    }

    /// The honest table: a deterministic negative phase against the exact ceiling, and against a
    /// sampled one.
    ///
    /// Percent of the range between the untrained model and [`train_exact`]'s maximum, on the same
    /// structure and the same data. The numbers this asserts are the ones in
    /// [`train_free_energy`]'s doc, and the ASSERTIONS are the ordering, not the digits.
    #[test]
    fn the_deterministic_negative_phase_against_the_exact_ceiling() {
        for (coupling, label) in [(0.6, "weak"), (1.4, "strong")] {
            let truth = chorded_ring(8, coupling, 9);
            let data = Dataset { visible: 8, rows: draw_rows(&truth, 512, 11) };
            let mut w0 = Weights::of(&truth);
            for x in &mut w0.edges {
                x.2 = 0.0;
            }
            w0.bias.fill(0.0);
            let structure = w0.build();

            let before = exact_log_likelihood(&structure, &data).unwrap();
            let p = FitParams { epochs: 2_000, lr: 0.05, l2: 0.0 };
            let ceiling = train_exact(&structure, &data, &p).unwrap().log_likelihood.unwrap();
            let pct = |ll: f64| 100.0 * (ll - before) / (ceiling - before);

            let s = Solve { iters: 2000, damping: 0.5 };
            let mf = pct(
                train_free_energy(&structure, &data, &p, Moments::MeanField, &s)
                    .unwrap()
                    .log_likelihood
                    .unwrap(),
            );
            let tap = pct(train_tap(&structure, &data, &p, &s).unwrap().log_likelihood.unwrap());
            let bethe = pct(train_bethe(&structure, &data, &p, &s).unwrap().log_likelihood.unwrap());

            let cdp = Params {
                epochs: 2_000,
                k: 10,
                positive_sweeps: 1,
                learning_rate: 0.05,
                batch: 16,
                persistent: false,
            };
            let cd = (0..6u64)
                .map(|seed| pct(train(&structure, &data, &cdp, seed).unwrap().log_likelihood.unwrap()))
                .sum::<f64>()
                / 6.0;

            println!(
                "{label} J={coupling}: untrained {before:.4} ceiling {ceiling:.4} | \
                 mean field {mf:.1}  TAP {tap:.1}  Bethe {bethe:.1}  CD-10 {cd:.1}"
            );

            // Bethe carries the loops mean field drops and TAP only corrects to second order, so
            // the ordering is not a tie to be broken by noise.
            assert!(mf > 0.0, "{label}: mean field should still climb: {mf:.1}");
            assert!(tap > mf + 5.0, "{label}: TAP {tap:.1} against mean field {mf:.1}");
            assert!(bethe > mf + 5.0, "{label}: Bethe {bethe:.1} against mean field {mf:.1}");
            // WHERE IT LOSES, and it loses in BOTH regimes here: a sampled negative phase is
            // ahead of every deterministic one on this model, by 0.4 points at J = 0.6 and by
            // 7.7 at J = 1.4. The deterministic methods buy their determinism, they do not get
            // it free.
            assert!(cd > tap, "{label}: CD {cd:.1} against TAP {tap:.1}");
            assert!(cd > bethe, "{label}: CD {cd:.1} against Bethe {bethe:.1}");
        }
    }

    /// Latent units: the same approximation on the clamped model, and the TRUE likelihood rises.
    ///
    /// Judged by enumeration, never by the free energy the fit ascends — an approximate objective
    /// scoring its own fit agrees with itself whatever it is doing.
    #[test]
    fn a_deterministic_fit_trains_a_machine_with_latent_units() {
        let structure = rbm(6, 4);
        let mut wt = Weights::of(&structure);
        for (k, x) in wt.edges.iter_mut().enumerate() {
            x.2 = 0.2 * ((k % 5) as f64 - 2.0) / 2.0;
        }
        let start = wt.build();
        let data = Dataset {
            visible: 6,
            rows: (0..32)
                .map(|r: usize| (0..6).map(|i| if (r >> (i / 2)) & 1 == 1 { 1i8 } else { -1 }).collect())
                .collect(),
        };
        let before = exact_log_likelihood(&start, &data).unwrap();
        let t = train_tap(
            &start,
            &data,
            &FitParams { epochs: 300, lr: 0.05, l2: 0.0 },
            &Solve { iters: 200, damping: 0.5 },
        )
        .unwrap();
        let after = t.log_likelihood.unwrap();
        assert!(after > before + 0.2, "TAP with latent units should learn: {before:.4} to {after:.4}");
        assert_eq!(t.graph.n_edges, structure.n_edges, "only weights move, never the edge set");
    }
}


#[cfg(test)]
mod jarzynski_tests {
    use super::*;
    use crate::free_energy::exact_log_z;

    /// A dataset drawn exactly from a known model, by inverse CDF over the enumerated distribution.
    ///
    /// `mod tests` holds the same three lines and they are private to it. Duplicated rather than
    /// widened: a test fixture reaching across module walls is how a fixture becomes an interface.
    fn draw_exact(g: &Graph, rows: usize, seed: u64) -> Dataset {
        let p = crate::ising::exact_boltzmann(g, 1.0);
        let mut cdf = Vec::with_capacity(p.len());
        let mut acc = 0.0;
        for &v in &p {
            acc += v;
            cdf.push(acc);
        }
        let mut rng = Pcg::new(seed, 11);
        let rows = (0..rows)
            .map(|_| {
                let u = rng.f64() * acc;
                let m = cdf.partition_point(|&c| c < u).min(p.len() - 1);
                (0..g.n).map(|i| if (m >> i) & 1 == 1 { 1i8 } else { -1 }).collect()
            })
            .collect();
        Dataset { visible: g.n, rows }
    }

    /// A frustrated six-spin ring with two chords, and the same edge set at zero weight. Six spins
    /// so that every oracle here — `ln Z`, the moments, the maximum-likelihood ceiling — is a sum
    /// over 64 states rather than an estimate.
    fn fixture() -> (Graph, Graph) {
        let mut b = GraphBuilder::new(6);
        for i in 0..6 {
            b.couple(i, (i + 1) % 6, if i % 2 == 0 { 0.7 } else { -0.5 });
        }
        b.couple(0, 3, 0.6);
        b.couple(1, 4, -0.4);
        b.bias(0, 0.3);
        b.bias(3, -0.25);
        let truth = b.build();

        let mut z = GraphBuilder::new(6);
        for i in 0..6 {
            z.couple(i, (i + 1) % 6, 0.0);
        }
        z.couple(0, 3, 0.0);
        z.couple(1, 4, 0.0);
        (truth, z.build())
    }

    /// Model magnetisations and correlations, by enumeration. The oracle the negative phase is
    /// trying to estimate.
    fn exact_moments(g: &Graph) -> (Vec<f64>, Vec<Vec<f64>>) {
        let mut z = 0.0;
        let mut mag = vec![0.0; g.n];
        let mut corr = vec![vec![0.0; g.n]; g.n];
        let mut s = vec![-1i8; g.n];
        for mask in 0..(1usize << g.n) {
            for (i, v) in s.iter_mut().enumerate() {
                *v = if mask >> i & 1 == 1 { 1 } else { -1 };
            }
            let w = (-g.energy(&s)).exp();
            z += w;
            for i in 0..g.n {
                mag[i] += w * f64::from(s[i]);
                for j in 0..g.n {
                    corr[i][j] += w * f64::from(s[i] * s[j]);
                }
            }
        }
        for i in 0..g.n {
            mag[i] /= z;
            for j in 0..g.n {
                corr[i][j] /= z;
            }
        }
        (mag, corr)
    }

    /// THE FIXED POINT IS MOMENT MATCHING, and it is what the reweighting is FOR.
    ///
    /// The gradient is `⟨·⟩_data − ⟨·⟩_model`, so if the weighted ensemble really estimates the
    /// model term then the fitted model's own moments must land on the data's. They are taken here
    /// by ENUMERATION, not by more sampling: a negative phase checked against a sampler agrees with
    /// itself whatever it is doing, and being biased is the failure this method exists to avoid.
    ///
    /// **This test also passes with the reweighting deleted**, which was measured rather than
    /// assumed. Six spins mix, so the weights are not what makes this fit good — they are pinned by
    /// the two free-energy tests below, and this one says the trainer is a working trainer.
    #[test]
    fn a_jarzynski_fit_matches_the_data_correlations() {
        let (truth, blank) = fixture();
        let data = draw_exact(&truth, 4_000, 3);
        let p = JarzynskiParams {
            epochs: 900,
            walkers: 256,
            k: 1,
            burn_in: 100,
            positive_sweeps: 1,
            learning_rate: 0.05,
            batch: 64,
            resample_at: 0.5,
        };
        let fit = train_jarzynski(&blank, &data, &p, 11).unwrap();

        let rows = data.rows.len() as f64;
        let dm = |i: usize| data.rows.iter().map(|r| f64::from(r[i])).sum::<f64>() / rows;
        let dc =
            |i: usize, j: usize| data.rows.iter().map(|r| f64::from(r[i] * r[j])).sum::<f64>() / rows;
        let (mag, corr) = exact_moments(&fit.trained.graph);
        for i in 0..6 {
            assert!(
                (mag[i] - dm(i)).abs() < 0.02,
                "magnetisation {i}: model {:.4} vs data {:.4} (min ess {:.1})",
                mag[i],
                dm(i),
                fit.min_ess
            );
        }
        // Only the FITTED edges are constrained; a pair the structure has no edge for is free.
        for &(i, j, _) in &Weights::of(&fit.trained.graph).edges {
            assert!(
                (corr[i][j] - dc(i, j)).abs() < 0.02,
                "correlation ({i},{j}): model {:.4} vs data {:.4}",
                corr[i][j],
                dc(i, j)
            );
        }
    }

    /// THE WORK IDENTITY, against the enumerated partition function, with the protocol frozen.
    ///
    /// At `learning_rate = 0` the parameters never move, so the ladder the walkers climb is fixed
    /// in advance and Jarzynski's equality applies with nothing conditional about it: `mean w` is
    /// an unbiased estimator of `Z / 2^n` and `log_z` must be `ln Z` of the model it was handed,
    /// which enumeration gives exactly. This is the reweighting arithmetic checked with the
    /// training dynamics switched off, so a failure here is arithmetic and not tuning.
    ///
    /// Both bookkeeping paths are exercised, and they must AGREE: banking `ln mean w` at a resample
    /// and reading it off at the end are two ways of accumulating the same product, so a frozen run
    /// cannot care which one it took.
    #[test]
    fn a_frozen_fit_leaves_the_model_alone_and_still_prices_it() {
        let (truth, _) = fixture();
        let data = draw_exact(&truth, 64, 9);
        let base = JarzynskiParams {
            epochs: 20,
            walkers: 4_096,
            k: 2,
            burn_in: 50,
            positive_sweeps: 1,
            learning_rate: 0.0,
            batch: 16,
            resample_at: 0.5,
        };
        let want = exact_log_z(&truth, 1.0);
        let fit = train_jarzynski(&truth, &data, &base, 13).unwrap();
        assert!(
            (fit.log_z - want).abs() < 0.08,
            "frozen run priced {} against the enumerated {want}",
            fit.log_z
        );
        assert!(fit.resamples > 0, "uniform weights on a coupled model must decay at least once");

        // Nothing moved, so the fitted likelihood is the starting model's, exactly.
        let before = exact_log_likelihood(&truth, &data).unwrap();
        assert!((fit.trained.log_likelihood.unwrap() - before).abs() < 1e-12);

        let never =
            train_jarzynski(&truth, &data, &JarzynskiParams { resample_at: 0.0, ..base }, 13)
                .unwrap();
        assert_eq!(never.resamples, 0, "resampling was switched off");
        assert!(
            (never.log_z - fit.log_z).abs() < 1e-9,
            "banked {} against unbanked {}: the same product accumulated two ways",
            fit.log_z,
            never.log_z
        );
    }

    /// AND WHAT THE WORK IDENTITY IS WORTH ONCE THE PROTOCOL IS READ OFF THE WALKERS.
    ///
    /// During training the parameters are chosen FROM the same ensemble that is being reweighted,
    /// so Jarzynski's equality holds only conditionally on the path and `log_z` is no longer
    /// unbiased. The residual is measured here against the enumerated `ln Z` of the model each run
    /// ended on, and it is not noise: at 200 epochs on this fixture it is −0.41, −0.10 and −0.027
    /// nats at 256, 1024 and 4096 walkers, halving as the ensemble doubles and moving by about 0.01
    /// across seeds. A one-over-walkers term with no seed spread is a BIAS, and the mechanism is
    /// the feedback — a heavy walker pulls the gradient toward suppressing itself.
    ///
    /// So the claim tested is consistency rather than unbiasedness: the error must shrink toward
    /// the enumerated value as walkers are added, which is what makes `log_z` usable and what
    /// separates it from a number that is merely stable.
    #[test]
    fn the_jarzynski_free_energy_converges_on_the_enumerated_one_as_walkers_grow() {
        let (truth, blank) = fixture();
        let data = draw_exact(&truth, 1_000, 5);
        let errs: Vec<f64> = [256usize, 1_024, 4_096]
            .iter()
            .map(|&walkers| {
                let p = JarzynskiParams {
                    epochs: 200,
                    walkers,
                    k: 1,
                    burn_in: 100,
                    positive_sweeps: 1,
                    learning_rate: 0.02,
                    batch: 250,
                    resample_at: 0.5,
                };
                let fit = train_jarzynski(&blank, &data, &p, 7).unwrap();
                fit.log_z - exact_log_z(&fit.trained.graph, 1.0)
            })
            .collect();
        assert!(
            errs[1].abs() < errs[0].abs() * 0.6 && errs[2].abs() < errs[1].abs() * 0.6,
            "four times the walkers must cut the error at least in half: {errs:?}"
        );
        assert!(errs[2].abs() < 0.06, "4096 walkers should be within 0.06 nats: {errs:?}");
    }

    /// Exact maximum likelihood is the ceiling: this does not pass it, and gets to within 0.5%.
    ///
    /// [`train_exact`] enumerates BOTH averages, so it is the best this structure can do on this
    /// data and a trainer reporting more is reporting a bug. On a fully visible model the positive
    /// phase is the data row itself, so the whole gap between this and the ceiling is the negative
    /// phase — which is the only thing the method changes.
    #[test]
    fn jarzynski_stays_under_the_exact_maximum_likelihood_ceiling() {
        let (truth, blank) = fixture();
        let data = draw_exact(&truth, 4_000, 3);
        let floor = exact_log_likelihood(&blank, &data).unwrap();
        let ceiling = train_exact(&blank, &data, &FitParams { epochs: 1_500, lr: 0.2, l2: 0.0 })
            .unwrap()
            .log_likelihood
            .unwrap();

        let p = JarzynskiParams {
            epochs: 900,
            walkers: 256,
            k: 1,
            burn_in: 100,
            positive_sweeps: 1,
            learning_rate: 0.05,
            batch: 64,
            resample_at: 0.5,
        };
        for seed in 0..3u64 {
            let fit = train_jarzynski(&blank, &data, &p, seed).unwrap();
            let got = fit.trained.log_likelihood.unwrap();
            assert!(got <= ceiling + 1e-9, "seed {seed} reached {got} above the ceiling {ceiling}");
            let reached = (got - floor) / (ceiling - floor);
            assert!(
                reached > 0.995,
                "seed {seed} reached {:.3}% of the ceiling: {got} against {ceiling} from {floor}",
                100.0 * reached
            );
        }
    }

    /// THE PRICE IS THE EFFECTIVE SAMPLE SIZE, and it is reported rather than hidden.
    ///
    /// Jarzynski's equality does not make a fast fit free; it moves the cost into the weights,
    /// where a fast-moving model concentrates them onto a few walkers. So the diagnostic has to be
    /// on the result, and it has to respond: with resampling switched off, a fifty-fold larger step
    /// must leave a visibly smaller effective sample size. It does — about 126 of 128 walkers at
    /// `lr = 0.01` against 19 to 31 at `lr = 0.5`.
    #[test]
    fn the_effective_sample_size_is_reported_and_falls_when_the_model_moves_fast() {
        let (truth, blank) = fixture();
        let data = draw_exact(&truth, 500, 17);
        let base = JarzynskiParams {
            epochs: 60,
            walkers: 128,
            k: 1,
            burn_in: 50,
            positive_sweeps: 1,
            learning_rate: 0.01,
            batch: 64,
            resample_at: 0.0,
        };
        let slow = train_jarzynski(&blank, &data, &base, 4).unwrap();
        let fast =
            train_jarzynski(&blank, &data, &JarzynskiParams { learning_rate: 0.5, ..base }, 4)
                .unwrap();

        for f in [&slow, &fast] {
            assert!(f.ess > 0.0 && f.ess <= 128.0 + 1e-9, "ess out of range: {}", f.ess);
            assert!(f.min_ess <= f.mean_ess + 1e-9, "the minimum is above the mean: {f:?}");
            assert!(f.mean_ess <= 128.0 + 1e-9, "mean ess above the walker count: {f:?}");
            assert_eq!(f.updates, 60 * data.rows.len().div_ceil(64));
            assert_eq!(f.resamples, 0, "resampling was switched off");
        }
        assert!(
            fast.min_ess < slow.min_ess * 0.5,
            "a fifty-times step must degenerate the weights: fast {:.2} against slow {:.2}",
            fast.min_ess,
            slow.min_ess
        );
    }

    /// It fits latent units too, and refuses the malformed data every trainer here refuses.
    #[test]
    fn the_jarzynski_trainer_takes_hidden_units_and_refuses_bad_data() {
        let data = bars_and_stripes(2);
        let structure = rbm(4, 3);
        let before = exact_log_likelihood(&structure, &data).unwrap();
        let p = JarzynskiParams {
            epochs: 400,
            walkers: 4_096,
            k: 1,
            burn_in: 50,
            positive_sweeps: 5,
            learning_rate: 0.05,
            batch: 6,
            resample_at: 0.5,
        };
        let fit = train_jarzynski(&structure, &data, &p, 2).unwrap();
        let after = fit.trained.log_likelihood.unwrap();
        assert!(after > before + 0.05, "training must help: {before:.4} -> {after:.4}");
        assert!(after < 0.0, "a log-likelihood is negative: {after}");
        // The work identity still prices a model with latent units, to the same accuracy.
        let want = exact_log_z(&fit.trained.graph, 1.0);
        assert!((fit.log_z - want).abs() < 0.25, "priced {} against {want}", fit.log_z);

        let err = |d: Dataset| train_jarzynski(&structure, &d, &p, 1).unwrap_err();
        assert_eq!(err(Dataset { visible: 4, rows: vec![] }), Error::NoData);
        assert_eq!(
            err(Dataset { visible: 4, rows: vec![vec![1, 1]] }),
            Error::RowWidth { row: 0, len: 2, want: 4 }
        );
        assert_eq!(
            err(Dataset { visible: 4, rows: vec![vec![1, 0, 1, 1]] }),
            Error::NotASpin { row: 0, at: 1, value: 0 }
        );
        assert_eq!(
            err(Dataset { visible: 9, rows: vec![vec![1i8; 9]] }),
            Error::TooSmall { spins: 7, visible: 9 }
        );
    }
}
