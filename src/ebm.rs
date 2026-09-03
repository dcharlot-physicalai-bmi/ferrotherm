//! Fitting an energy-based model to data — contrastive divergence, and the exact likelihood to
//! judge it by.
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
//! # Judging it
//!
//! [`exact_log_likelihood`] enumerates. Every claim about expressivity in this crate is measured
//! against the true likelihood on models small enough to compute it, never against a bound, an ELBO
//! or a reconstruction error — because the tradeoff being measured is a claim about the true
//! distribution, and a proxy for it would put the proxy's own failure mode inside the result.

use crate::gibbs::Sampler;
use crate::graph::{Graph, GraphBuilder};
use crate::rng::Pcg;

/// Rows of `±1`, the first `visible` entries of each being the observed part.
#[derive(Clone, Debug)]
pub struct Dataset {
    /// How many leading spins of a state are observed. The rest are latent.
    pub visible: usize,
    pub rows: Vec<Vec<i8>>,
}

/// Why a fit was refused.
#[derive(Clone, Debug, PartialEq)]
pub enum Error {
    /// The dataset is empty, so there is nothing to fit.
    NoData,
    /// A row is not `visible` long, so it cannot be clamped onto the model.
    RowWidth { row: usize, len: usize, want: usize },
    /// A row holds something other than `-1` or `+1`.
    NotASpin { row: usize, at: usize, value: i8 },
    /// The model has fewer spins than the data has visible units.
    TooSmall { spins: usize, visible: usize },
    /// The model has more spins than [`MAX_ENUMERATED`], so its exact likelihood cannot be taken.
    ///
    /// **This used to be reported as [`Error::TooSmall`]**, whose message reads "the model has 24
    /// spins and the data needs 16 visible" -- true, irrelevant, and the exact opposite of what
    /// went wrong. It never named the limit and never said the model was too large. Fitting to 4x4
    /// data therefore lost its only quality metric somewhere past six hidden units, silently,
    /// because [`train`] takes the likelihood with `.ok()` and a mislabelled error looks the same
    /// as an absent one.
    TooLarge { spins: usize, limit: usize },
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
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
        Params { epochs: 300, k: 5, positive_sweeps: 5, learning_rate: 0.05, batch: 8 }
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

            for &r in chunk {
                let row = &data.rows[r];

                // POSITIVE PHASE. Visible clamped to the data, latent settled around it.
                let seed = (rng.next_u32() as u64) << 32 | rng.next_u32() as u64;
                let mut smp = Sampler::new(&g, 1.0, seed);
                for (i, &v) in row.iter().enumerate() {
                    smp.clamp(i, v);
                }
                smp.sweeps(p.positive_sweeps.max(1), None);
                let pos = smp.s.clone();

                // NEGATIVE PHASE. The same chain, unclamped, k sweeps on -- which is what makes
                // this CONTRASTIVE DIVERGENCE and not maximum likelihood: the model average is
                // taken near the data rather than at equilibrium, and it is biased for exactly
                // that reason.
                for i in 0..data.visible {
                    smp.unclamp(i);
                }
                smp.sweeps(p.k.max(1), None);
                let neg = &smp.s;

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
}

impl AisLikelihood {
    /// The likelihood is at most this with probability at least `1 − delta`: the exact numerator
    /// minus the unconditional lower bound on `ln Z`. A lower bound on the likelihood would need
    /// an upper bound on `ln Z`, which is reverse AIS's conditional business.
    pub fn upper_bound(&self, delta: f64) -> f64 {
        self.mean_log_numerator - self.log_z.lower_bound(delta)
    }
}

/// Mean log-likelihood per row for a model too large to enumerate, when its HIDDEN part is not.
///
/// `log p(v) = ln Σ_h exp(−E(v, h)) − ln Z`. The first term enumerates the `2^hidden` completions
/// of each row exactly; the second is [`crate::free_energy::ais`] on the whole model, whose lower
/// bound is unconditional and therefore gives [`AisLikelihood::upper_bound`] the same standing.
/// Refuses when `hidden > MAX_ENUMERATED`; a clamped AIS for the numerator is the recorded next
/// step past that.
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
        let mx = logs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        total += mx + logs.iter().map(|x| (x - mx).exp()).sum::<f64>().ln();
    }
    let mean_log_numerator = total / data.rows.len() as f64;
    let log_z = crate::free_energy::ais(g, ladder, sweeps, runs, seed);
    Ok(AisLikelihood { estimate: mean_log_numerator - log_z.log_z, mean_log_numerator, log_z })
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
        assert!(exact <= a.upper_bound(1e-6), "exact {exact} above the bound {}", a.upper_bound(1e-6));
        assert!(a.log_z.ess > 8.0);
        // Too many hidden units is refused, not approximated.
        let wide = rbm(9, 30);
        assert!(matches!(log_likelihood_ais(&wide, &data, &linear_ladder(1.0, 8), 1, 4, 1), Err(Error::TooLarge { .. })));
    }
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
        let p = Params { epochs: 4_000, k: 20, learning_rate: 0.05, batch: 6, positive_sweeps: 1 };
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
}
