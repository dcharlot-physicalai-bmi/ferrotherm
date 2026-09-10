//! Noise-contrastive estimation (Gutmann & Hyvärinen, JMLR 13:307, 2012): fit an energy-based
//! model by logistic regression against a noise distribution, learning the normaliser.
//!
//! # What separates this from every other estimator in [`crate::ebm`]
//!
//! `E(s) = −Σ h_i s_i − Σ J_ij s_i s_j` gives an UNNORMALISED density, and `ln Z` is the whole
//! difficulty of the field. The three flip losses dodge it by targeting a different objective;
//! contrastive divergence dodges it by sampling. NCE does neither. It carries `c` — the log
//! normaliser — as **an ordinary free parameter fitted alongside the weights**, so the model is
//!
//! ```text
//!   ln p_m(s; θ, c) = −E(s; θ) − c
//! ```
//!
//! and nothing constrains `c` to equal `ln Z(θ)` except the objective itself. That it comes out
//! equal anyway is the theorem, and it is testable here against
//! [`crate::exact::Elimination::log_partition`] rather than taken on faith — see
//! `the_learned_normaliser_converges_to_the_true_log_partition`.
//!
//! # The objective
//!
//! Label the `D` data rows 1 and `N = ν·D` draws from a known noise density `p_n` as 0, and do
//! logistic regression. With `ν = N/D` and the log-odds
//!
//! ```text
//!   δ(s) = ln p_m(s) − ln p_n(s) − ln ν = −E(s) − c − ln p_n(s) − ln ν
//! ```
//!
//! the score to maximise, per data row, is
//!
//! ```text
//!   J = (1/D) Σ_data ln σ(δ(x))  +  (1/D) Σ_noise ln(1 − σ(δ(y)))
//! ```
//!
//! `δ` is AFFINE in `(θ, c)` and both `ln σ` and `ln(1 − σ)` are concave, so `J` is concave in
//! every parameter including the normaliser: plain gradient ascent has nowhere else to go, exactly
//! as with pseudolikelihood, and the L2 penalty keeps it concave.
//!
//! # What it costs and where it breaks
//!
//! No sampler of the model, no enumeration, no `2^n`: one pass over `D(1 + ν)` states per epoch.
//! The price is that the answer is only as good as the noise — a `p_n` with no mass where the data
//! lives makes the two classes trivially separable, every `σ` saturates, and the gradient dies.
//! That is why [`Noise::DataMarginals`] exists and is the default worth reaching for; the same
//! failure is the reason NCE is reported here with its noise distribution named rather than as a
//! single number.
//!
//! Latent units are refused ([`Error::HasLatent`]): `ln p_m(x)` at a data point would itself be a
//! sum over the hidden units, which is the partition function this method exists to avoid.

use crate::ebm::{Dataset, Error, FitParams, MAX_ENUMERATED, Trained, exact_log_likelihood};
use crate::graph::{Graph, GraphBuilder};
use crate::rng::Pcg;

/// `ln(1 + e^x)`, stable at both ends.
#[inline]
fn softplus(x: f64) -> f64 {
    if x > 0.0 { x + (-x).exp().ln_1p() } else { x.exp().ln_1p() }
}

/// `ln σ(x)`, never forming `σ(x)` and so never rounding a saturated class posterior to 0 or 1.
#[inline]
fn log_sigmoid(x: f64) -> f64 {
    -softplus(-x)
}

/// `σ(x)`. The crate's kernel writes the logistic as `σ(2βf)`, so `β = 1/2` is the plain one.
#[inline]
fn sigmoid(x: f64) -> f64 {
    crate::kernel::p_up(x, 0.5)
}

/// Which auxiliary distribution the data is contrasted against.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Noise {
    /// Uniform over all `2^n` states: `ln p_n(s) = −n ln 2`.
    Uniform,
    /// Independent spins, each site's `P(+1)` matched to the data's mean at that site.
    ///
    /// Laplace-smoothed as `(up + 1)/(D + 2)`, so a site that is constant in the data still gets a
    /// finite log-density instead of `−inf` on the one state the noise can never produce.
    DataMarginals,
}

/// A factorised noise distribution: independent spins, `P(s_i = +1) = up[i]`.
///
/// Factorised because NCE needs `ln p_n(s)` in closed form at every sample it scores, which rules
/// out anything that would itself need a partition function. The two logs per site are cached at
/// construction: a fit evaluates this once per row per epoch, and `ln` in that loop is the single
/// most expensive thing in the method.
#[derive(Clone, Debug, PartialEq)]
pub struct NoiseModel {
    up: Vec<f64>,
    ln_up: Vec<f64>,
    ln_down: Vec<f64>,
}

impl NoiseModel {
    /// A noise distribution from its per-site `P(s_i = +1)`.
    ///
    /// # Panics
    ///
    /// If `up` is empty or any entry is not strictly inside `(0, 1)` — a site the noise can never
    /// put in one of its states gives `ln p_n = −inf` on every state containing it, and the whole
    /// objective becomes `NaN` rather than merely a bad fit.
    #[must_use]
    pub fn new(up: Vec<f64>) -> NoiseModel {
        assert!(!up.is_empty(), "a noise model needs at least one site");
        assert!(
            up.iter().all(|q| *q > 0.0 && *q < 1.0),
            "every noise probability must be strictly inside (0, 1)"
        );
        let ln_up = up.iter().map(|q| q.ln()).collect();
        let ln_down = up.iter().map(|q| (1.0 - q).ln()).collect();
        NoiseModel { up, ln_up, ln_down }
    }

    /// Build the noise distribution of `kind` over `n` spins, reading `data` if it needs to.
    ///
    /// Sites beyond `data.visible` get `1/2`, having no data to match.
    ///
    /// # Panics
    ///
    /// If `n` is zero.
    #[must_use]
    pub fn fit(kind: Noise, n: usize, data: &Dataset) -> NoiseModel {
        let mut up = vec![0.5; n];
        if kind == Noise::DataMarginals {
            let d = data.rows.len() as f64;
            for i in 0..n.min(data.visible) {
                let ups = data.rows.iter().filter(|r| r[i] == 1).count() as f64;
                up[i] = (ups + 1.0) / (d + 2.0);
            }
        }
        NoiseModel::new(up)
    }

    /// `P(s_i = +1)` per site.
    #[must_use]
    pub fn up(&self) -> &[f64] {
        &self.up
    }

    /// How many spins this noise distribution covers.
    #[must_use]
    pub fn n(&self) -> usize {
        self.up.len()
    }

    /// `ln p_n(s)`, the sum of the per-site log-probabilities.
    ///
    /// # Panics
    ///
    /// If `s` is not one spin per site.
    #[must_use]
    pub fn log_density(&self, s: &[i8]) -> f64 {
        assert_eq!(s.len(), self.up.len(), "state width must match the noise model");
        let mut acc = 0.0;
        for (i, &v) in s.iter().enumerate() {
            acc += if v == 1 { self.ln_up[i] } else { self.ln_down[i] };
        }
        acc
    }

    /// Draw one state into `out`.
    ///
    /// # Panics
    ///
    /// If `out` is not one spin per site.
    pub fn draw(&self, rng: &mut Pcg, out: &mut [i8]) {
        assert_eq!(out.len(), self.up.len(), "state width must match the noise model");
        for (i, v) in out.iter_mut().enumerate() {
            *v = rng.spin(self.up[i]);
        }
    }
}

/// Draw `count` noise states, which is the sample the objective is defined on.
///
/// Exposed because the NCE objective is a function OF a noise sample, not of a noise distribution:
/// [`train_nce`] draws once and ascends a fixed objective, and a caller re-scoring the result has
/// to score it on the same draws.
#[must_use]
pub fn noise_rows(noise: &NoiseModel, count: usize, seed: u64) -> Vec<Vec<i8>> {
    let mut rng = Pcg::new(seed, 0x00_4E_43_45);
    let mut rows = Vec::with_capacity(count);
    for _ in 0..count {
        let mut s = vec![-1i8; noise.n()];
        noise.draw(&mut rng, &mut s);
        rows.push(s);
    }
    rows
}

/// A graph's undirected edges as `(i, j, w)` with `i < j`, in CSR order.
///
/// The order every gradient here is indexed by. Exposed because [`nce_gradient`] and
/// [`nce_population_gradient`] return one derivative per entry and a caller has no other way to
/// know which entry is which edge.
#[must_use]
pub fn edges(g: &Graph) -> Vec<(usize, usize, f64)> {
    let mut out = Vec::with_capacity(g.n_edges);
    for i in 0..g.n {
        for k in g.offset[i]..g.offset[i + 1] {
            let j = g.nbr[k] as usize;
            if j > i {
                out.push((i, j, g.w[k]));
            }
        }
    }
    out
}

/// What an NCE fit produced. Unlike [`Trained`] it carries a normaliser, because NCE fitted one.
pub struct Nce {
    /// The fitted model. Its edge set is the structure it was given; only weights moved.
    pub graph: Graph,
    /// The learned `c`, which the theory says approaches `ln Z` of [`Nce::graph`].
    ///
    /// It is a NUISANCE parameter for likelihood purposes — [`exact_log_likelihood`] normalises the
    /// graph itself and does not read this — and the point of the method for every other purpose.
    pub log_normaliser: f64,
    /// Mean exact log-likelihood per row, or `None` past [`MAX_ENUMERATED`] spins.
    pub log_likelihood: Option<f64>,
    /// The NCE objective reached, per data row. Never positive; `−(1 + ν) ln 2`-ish at chance.
    pub objective: f64,
    /// The noise distribution the fit was run against, without which the objective means nothing.
    pub noise: NoiseModel,
    /// Epochs completed, which is the cap since there is no early stop.
    pub epochs_run: usize,
}

impl core::fmt::Debug for Nce {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Nce")
            .field("spins", &self.graph.n)
            .field("edges", &self.graph.n_edges)
            .field("log_normaliser", &self.log_normaliser)
            .field("log_likelihood", &self.log_likelihood)
            .field("objective", &self.objective)
            .field("epochs_run", &self.epochs_run)
            .finish()
    }
}

impl Nce {
    /// `log_normaliser − ln Z` of the fitted graph, by [`crate::exact::Elimination`] at `beta = 1`.
    ///
    /// `None` when the graph's induced width exceeds `max_width`, since a table costs `2^width`.
    /// This is the residual NCE's own theory predicts goes to zero, so it is worth reading off a
    /// fit rather than reconstructing.
    #[must_use]
    pub fn normaliser_error(&self, max_width: usize) -> Option<f64> {
        let e = crate::exact::Elimination { max_width };
        e.log_partition(&self.graph, 1.0).ok()?.log_z.map(|z| self.log_normaliser - z)
    }

    /// Drop the normaliser and hand back the [`Trained`] the rest of [`crate::ebm`] compares.
    #[must_use]
    pub fn into_trained(self) -> Trained {
        Trained {
            graph: self.graph,
            log_likelihood: self.log_likelihood,
            epochs_run: self.epochs_run,
        }
    }
}

/// The parameters of a fit as a flat list: the same walk [`crate::ebm`] does, plus `c`.
struct Model {
    n: usize,
    /// `(i, j, w)` with `i < j`, one entry per undirected edge, in CSR order.
    edges: Vec<(usize, usize, f64)>,
    bias: Vec<f64>,
    c: f64,
}

impl Model {
    fn of(structure: &Graph) -> Model {
        Model {
            n: structure.n,
            edges: edges(structure),
            bias: structure.h.clone(),
            c: 0.0,
        }
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

    /// One ascent step. THE L2 PULL DOES NOT TOUCH `c`: shrinking a normaliser toward zero is not
    /// regularisation, it is a bias on the one quantity this method exists to estimate.
    fn step(&mut self, dw: &[f64], dh: &[f64], dc: f64, p: &FitParams) {
        for (e, x) in self.edges.iter_mut().enumerate() {
            x.2 += p.lr * (dw[e] - p.l2 * x.2);
        }
        for (i, b) in self.bias.iter_mut().enumerate() {
            *b += p.lr * (dh[i] - p.l2 * *b);
        }
        self.c += p.lr * dc;
    }
}

/// Reject data this method cannot take, with [`crate::ebm`]'s own errors.
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
    if g.n != data.visible {
        return Err(Error::HasLatent { spins: g.n, visible: data.visible });
    }
    Ok(())
}

/// Mean NCE score per data row, on a FIXED noise sample.
///
/// `ν` is read off the two sample sizes rather than passed, so an objective and the sample it was
/// computed on cannot disagree about how many noise draws there were.
///
/// # Errors
///
/// [`Error::NoData`] when either sample is empty, [`Error::HasLatent`] when the model has spins the
/// data does not observe, and the malformed-row errors otherwise.
pub fn nce_objective(
    g: &Graph,
    log_normaliser: f64,
    data: &Dataset,
    noise: &NoiseModel,
    drawn: &[Vec<i8>],
) -> Result<f64, Error> {
    check(g, data)?;
    if drawn.is_empty() {
        return Err(Error::NoData);
    }
    let d = data.rows.len() as f64;
    let ln_nu = ((drawn.len() as f64) / d).ln();
    let odds = |s: &[i8]| -g.energy(s) - log_normaliser - noise.log_density(s) - ln_nu;
    let mut acc = 0.0;
    for row in &data.rows {
        acc += log_sigmoid(odds(row));
    }
    for row in drawn {
        acc -= softplus(odds(row));
    }
    Ok(acc / d)
}

/// The gradient of [`nce_objective`] in every weight, bias and in `c`, as an ASCENT direction.
///
/// Returned as one derivative per entry of `edges`, one per spin, and one for `c`. Public so a
/// caller can drive it with an optimiser other than the plain ascent [`train_nce`] runs.
///
/// With `a(x) = σ(−δ(x))` on a data row and `b(y) = σ(δ(y))` on a noise row — each the class
/// posterior the logistic regression currently gets WRONG — and `∂δ/∂w_ij = s_i s_j`,
/// `∂δ/∂h_i = s_i`, `∂δ/∂c = −1`:
///
/// ```text
///   dJ/dw_ij = (1/D) [ Σ_data a x_i x_j  −  Σ_noise b y_i y_j ]
///   dJ/dh_i  = (1/D) [ Σ_data a x_i      −  Σ_noise b y_i     ]
///   dJ/dc    = (1/D) [ −Σ_data a         +  Σ_noise b         ]
/// ```
///
/// The `c` row is the one with the sign that cannot be checked by watching the fit converge: a
/// normaliser drifting the wrong way still leaves a well-fitted graph, because the likelihood does
/// not read it. It is differenced in `the_gradient_matches_a_central_difference` instead.
///
/// # Panics
///
/// If a row is not one spin per model site.
#[must_use]
pub fn nce_gradient(
    g: &Graph,
    log_normaliser: f64,
    data: &Dataset,
    noise: &NoiseModel,
    drawn: &[Vec<i8>],
    edges: &[(usize, usize, f64)],
) -> (Vec<f64>, Vec<f64>, f64) {
    let d = data.rows.len() as f64;
    let ln_nu = ((drawn.len() as f64) / d).ln();
    let mut dw = vec![0.0; edges.len()];
    let mut dh = vec![0.0; g.n];
    let mut dc = 0.0;

    let mut accumulate = |s: &[i8], weight: f64| {
        for (i, x) in dh.iter_mut().enumerate() {
            *x += weight * f64::from(s[i]);
        }
        for (k, &(i, j, _)) in edges.iter().enumerate() {
            dw[k] += weight * f64::from(s[i]) * f64::from(s[j]);
        }
        dc -= weight;
    };

    for row in &data.rows {
        let delta = -g.energy(row) - log_normaliser - noise.log_density(row) - ln_nu;
        accumulate(row, sigmoid(-delta));
    }
    for row in drawn {
        let delta = -g.energy(row) - log_normaliser - noise.log_density(row) - ln_nu;
        accumulate(row, -sigmoid(delta));
    }

    for x in &mut dw {
        *x /= d;
    }
    for x in &mut dh {
        *x /= d;
    }
    (dw, dh, dc / d)
}

/// Fit `structure`'s weights AND a log normaliser to `data` by noise-contrastive estimation.
///
/// `structure` supplies the edge set and the starting weights; `c` starts at zero. `ratio` is `ν`,
/// the noise draws per data row — the noise sample is drawn ONCE from `seed` and held fixed, so
/// what is ascended is a fixed concave function and not a stochastic approximation of one.
///
/// # Errors
///
/// [`Error::HasLatent`] when the model has spins the data does not observe — see that variant, and
/// this module's header for why NCE in particular cannot have them. [`Error::NoData`] and the
/// malformed-row errors otherwise.
///
/// # Panics
///
/// If `ratio` is zero: with no noise sample the objective is maximised by `c → −∞` and there is
/// nothing to contrast against.
pub fn train_nce(
    structure: &Graph,
    data: &Dataset,
    p: &FitParams,
    noise: Noise,
    ratio: usize,
    seed: u64,
) -> Result<Nce, Error> {
    check(structure, data)?;
    assert!(ratio > 0, "NCE needs at least one noise draw per data row");
    let nm = NoiseModel::fit(noise, structure.n, data);
    let drawn = noise_rows(&nm, ratio * data.rows.len(), seed);

    let mut m = Model::of(structure);
    for _ in 0..p.epochs {
        let g = m.build();
        let (dw, dh, dc) = nce_gradient(&g, m.c, data, &nm, &drawn, &m.edges);
        m.step(&dw, &dh, dc, p);
    }

    let graph = m.build();
    let objective = nce_objective(&graph, m.c, data, &nm, &drawn)?;
    let log_likelihood = exact_log_likelihood(&graph, data).ok();
    Ok(Nce {
        graph,
        log_normaliser: m.c,
        log_likelihood,
        objective,
        noise: nm,
        epochs_run: p.epochs,
    })
}

/// What both population routines require of their arguments, in one place.
fn guard(model: &Graph, data_probability: &[f64], noise: &NoiseModel, ratio: f64) {
    assert!(model.n <= MAX_ENUMERATED, "population NCE enumerates every state");
    assert_eq!(data_probability.len(), 1usize << model.n, "one probability per state");
    assert_eq!(noise.n(), model.n, "noise model must be as wide as the model");
    assert!(ratio > 0.0 && ratio.is_finite(), "the noise ratio must be positive and finite");
}

/// The NCE objective in the INFINITE-DATA limit: both expectations enumerated, no sampling at all.
///
/// `data_probability[mask]` is the true `p_d` of the state whose spin `i` is bit `i` of `mask`,
/// which is what [`crate::ising::exact_boltzmann`] returns. This is to [`nce_objective`] what
/// [`crate::ebm::train_exact`] is to a sampled fit: the quantity the finite-sample version
/// estimates, available exactly on models small enough to walk.
///
/// Its use is one identity — the population gradient VANISHES at `(θ_true, ln Z)` for every noise
/// distribution and every `ratio`, because `σ(−δ) p_d = ν σ(δ) p_n` there term by term — which is
/// the whole consistency argument and is checkable to machine precision.
///
/// # Panics
///
/// If `data_probability` is not `2^n` long, if `noise` is not `n` wide, if `ratio` is not positive
/// and finite, or if `n` exceeds [`MAX_ENUMERATED`], since this walks every state.
#[must_use]
pub fn nce_population(
    model: &Graph,
    log_normaliser: f64,
    data_probability: &[f64],
    noise: &NoiseModel,
    ratio: f64,
) -> f64 {
    guard(model, data_probability, noise, ratio);
    let ln_nu = ratio.ln();
    let mut s = vec![-1i8; model.n];
    let mut acc = 0.0;
    for mask in 0..(1usize << model.n) {
        for (i, x) in s.iter_mut().enumerate() {
            *x = if mask >> i & 1 == 1 { 1 } else { -1 };
        }
        let ln_pn = noise.log_density(&s);
        let delta = -model.energy(&s) - log_normaliser - ln_pn - ln_nu;
        acc += data_probability[mask] * log_sigmoid(delta) - ratio * ln_pn.exp() * softplus(delta);
    }
    acc
}

/// The gradient of [`nce_population`], as an ascent direction. Same algebra as [`nce_gradient`]
/// with the empirical averages replaced by the exact ones, so it is the quantity that is EXACTLY
/// zero at `(θ_true, ln Z)` — see [`nce_population`].
///
/// # Panics
///
/// As [`nce_population`].
#[must_use]
pub fn nce_population_gradient(
    model: &Graph,
    log_normaliser: f64,
    data_probability: &[f64],
    noise: &NoiseModel,
    ratio: f64,
    edges: &[(usize, usize, f64)],
) -> (Vec<f64>, Vec<f64>, f64) {
    guard(model, data_probability, noise, ratio);
    let ln_nu = ratio.ln();
    let mut s = vec![-1i8; model.n];
    let mut dw = vec![0.0; edges.len()];
    let mut dh = vec![0.0; model.n];
    let mut dc = 0.0;
    for mask in 0..(1usize << model.n) {
        for (i, x) in s.iter_mut().enumerate() {
            *x = if mask >> i & 1 == 1 { 1 } else { -1 };
        }
        let ln_pn = noise.log_density(&s);
        let delta = -model.energy(&s) - log_normaliser - ln_pn - ln_nu;
        let weight =
            data_probability[mask] * sigmoid(-delta) - ratio * ln_pn.exp() * sigmoid(delta);
        for (i, x) in dh.iter_mut().enumerate() {
            *x += weight * f64::from(s[i]);
        }
        for (k, &(i, j, _)) in edges.iter().enumerate() {
            dw[k] += weight * f64::from(s[i]) * f64::from(s[j]);
        }
        dc -= weight;
    }
    (dw, dh, dc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ebm::train_exact;
    use crate::exact::Elimination;
    use crate::ising::exact_boltzmann;

    /// A small fully-connected model with definite, asymmetric parameters.
    fn truth(n: usize) -> Graph {
        let mut gb = GraphBuilder::new(n);
        for i in 0..n {
            gb.bias(i, 0.35 - 0.2 * i as f64);
            for j in (i + 1)..n {
                gb.couple(i, j, 0.45 - 0.25 * ((i + 2 * j) % 3) as f64);
            }
        }
        gb.build()
    }

    /// The same edge set with every weight at zero: what a fit starts from.
    fn structure(n: usize) -> Graph {
        let mut gb = GraphBuilder::new(n);
        for i in 0..n {
            for j in (i + 1)..n {
                gb.couple(i, j, 0.0);
            }
        }
        gb.build()
    }

    /// Exact i.i.d. draws from `g`'s Boltzmann distribution by inverting its enumerated CDF.
    ///
    /// Not a Gibbs chain: there is no burn-in, no autocorrelation and no mixing assumption, so a
    /// consistency claim measured on these rows is a claim about the ESTIMATOR and not about a
    /// sampler that happened to keep up.
    fn draws(g: &Graph, count: usize, seed: u64) -> Vec<Vec<i8>> {
        let p = exact_boltzmann(g, 1.0);
        let mut cdf = Vec::with_capacity(p.len());
        let mut run = 0.0;
        for &x in &p {
            run += x;
            cdf.push(run);
        }
        let mut rng = Pcg::new(seed, 0xD_A7A);
        let mut rows = Vec::with_capacity(count);
        for _ in 0..count {
            let u = rng.f64() * run;
            let mask = cdf.partition_point(|&c| c < u).min(p.len() - 1);
            rows.push((0..g.n).map(|i| if mask >> i & 1 == 1 { 1i8 } else { -1 }).collect());
        }
        rows
    }

    fn log_z(g: &Graph) -> f64 {
        Elimination::default().log_partition(g, 1.0).unwrap().log_z.unwrap()
    }

    /// A model rebuilt with one scalar parameter nudged, for differencing.
    fn perturb(g: &Graph, which: usize, delta: f64) -> Graph {
        let edges = edges(g);
        let mut gb = GraphBuilder::new(g.n);
        for (k, &(i, j, w)) in edges.iter().enumerate() {
            gb.couple(i, j, if k == which { w + delta } else { w });
        }
        for i in 0..g.n {
            let b = g.h[i];
            gb.bias(i, if which == edges.len() + i { b + delta } else { b });
        }
        gb.build()
    }

    /// `Elimination` is the crate's exact `ln Z`, and this module's headline claim is measured
    /// against it — so its convention is pinned once, here, against a hand enumeration.
    ///
    /// If `log_partition` meant `Σ exp(+E)` or carried a beta the wrong way round, every
    /// normaliser test below would compare a learned constant against the wrong number and could
    /// still "pass" by drifting to it.
    #[test]
    fn the_reference_log_partition_is_the_enumerated_one() {
        let g = truth(5);
        let mut s = vec![-1i8; 5];
        let mut acc = 0.0;
        for mask in 0..32usize {
            for (i, x) in s.iter_mut().enumerate() {
                *x = if mask >> i & 1 == 1 { 1 } else { -1 };
            }
            acc += (-g.energy(&s)).exp();
        }
        assert!((log_z(&g) - acc.ln()).abs() < 1e-10, "{} vs {}", log_z(&g), acc.ln());
    }

    /// THE GRADIENT IS DIFFERENCED, not watched.
    ///
    /// Central differences of the objective itself, in every edge weight, every bias and in `c`.
    /// The `c` coordinate is the one no convergence test can reach: the likelihood of the fitted
    /// graph does not read the normaliser, so a sign error there leaves a fit that still looks
    /// good and a normaliser that goes to the wrong place.
    #[test]
    fn the_gradient_matches_a_central_difference() {
        let n = 5;
        let g = truth(n);
        let data = Dataset { visible: n, rows: draws(&g, 60, 11) };
        let nm = NoiseModel::fit(Noise::DataMarginals, n, &data);
        let drawn = noise_rows(&nm, 180, 5);
        let c = 0.7;

        let edges = edges(&g);
        let (dw, dh, dc) = nce_gradient(&g, c, &data, &nm, &drawn, &edges);

        let eps = 1e-6;
        for k in 0..edges.len() + n {
            let up = perturb(&g, k, eps);
            let dn = perturb(&g, k, -eps);
            let fd = (nce_objective(&up, c, &data, &nm, &drawn).unwrap()
                - nce_objective(&dn, c, &data, &nm, &drawn).unwrap())
                / (2.0 * eps);
            let analytic = if k < edges.len() { dw[k] } else { dh[k - edges.len()] };
            assert!(
                (fd - analytic).abs() < 1e-7,
                "coordinate {k}: finite difference {fd}, analytic {analytic}"
            );
        }

        let fd_c = (nce_objective(&g, c + eps, &data, &nm, &drawn).unwrap()
            - nce_objective(&g, c - eps, &data, &nm, &drawn).unwrap())
            / (2.0 * eps);
        assert!((fd_c - dc).abs() < 1e-7, "normaliser: finite difference {fd_c}, analytic {dc}");
    }

    /// The population gradient differenced the same way, since it is the one the identity below
    /// asserts is exactly zero and a wrong gradient would be zero in the wrong place.
    #[test]
    fn the_population_gradient_matches_a_central_difference() {
        let n = 5;
        let g = truth(n);
        let mut other = GraphBuilder::new(n);
        for i in 0..n {
            other.bias(i, 0.1 * i as f64);
            for j in (i + 1)..n {
                other.couple(i, j, 0.2);
            }
        }
        let pd = exact_boltzmann(&other.build(), 1.0);
        let nm = NoiseModel::new((0..n).map(|i| 0.4 + 0.05 * i as f64).collect());
        let c = 0.3;
        let ratio = 2.5;

        let edges = edges(&g);
        let (dw, dh, dc) = nce_population_gradient(&g, c, &pd, &nm, ratio, &edges);
        let eps = 1e-6;
        for k in 0..edges.len() + n {
            let fd = (nce_population(&perturb(&g, k, eps), c, &pd, &nm, ratio)
                - nce_population(&perturb(&g, k, -eps), c, &pd, &nm, ratio))
                / (2.0 * eps);
            let analytic = if k < edges.len() { dw[k] } else { dh[k - edges.len()] };
            assert!((fd - analytic).abs() < 1e-8, "coordinate {k}: {fd} vs {analytic}");
        }
        let fd_c = (nce_population(&g, c + eps, &pd, &nm, ratio)
            - nce_population(&g, c - eps, &pd, &nm, ratio))
            / (2.0 * eps);
        assert!((fd_c - dc).abs() < 1e-8, "normaliser: {fd_c} vs {dc}");
    }

    /// THE EXACT ORACLE: the population gradient is ZERO at the true parameters with `c = ln Z`.
    ///
    /// Not "small", not "smaller than at other points" — algebraically zero, because at that point
    /// `p_m = p_d` and every term of the data sum cancels its partner in the noise sum:
    /// `p_d · νp_n/(p_d + νp_n) = ν p_n · p_d/(p_d + νp_n)`. It holds for EVERY noise distribution
    /// and every ratio, which is why both are swept here, and it is the whole reason the learned
    /// normaliser lands on `ln Z` rather than somewhere convenient.
    ///
    /// Feeding a `c` that is not `ln Z` must break it, so that is checked too — otherwise a
    /// gradient that ignored `c` entirely would pass.
    #[test]
    fn the_population_gradient_vanishes_at_the_truth() {
        for n in [4usize, 5, 6] {
            let g = truth(n);
            let pd = exact_boltzmann(&g, 1.0);
            let z = log_z(&g);
            let edges = edges(&g);
            for (kind, ratio) in [
                (Noise::Uniform, 1.0),
                (Noise::Uniform, 7.0),
                (Noise::DataMarginals, 0.25),
                (Noise::DataMarginals, 3.0),
            ] {
                // The marginals of the truth itself, so the noise is a genuinely different
                // factorised distribution rather than the uniform one under another name.
                let m = Elimination::default().marginals(&g, 1.0).unwrap();
                let nm = match kind {
                    Noise::Uniform => NoiseModel::fit(kind, n, &Dataset { visible: n, rows: vec![vec![1i8; n]] }),
                    Noise::DataMarginals => NoiseModel::new(m),
                };
                let (dw, dh, dc) = nce_population_gradient(&g, z, &pd, &nm, ratio, &edges);
                let worst = dw
                    .iter()
                    .chain(&dh)
                    .chain(core::iter::once(&dc))
                    .fold(0.0f64, |a, x| a.max(x.abs()));
                assert!(worst < 1e-12, "n={n} ratio={ratio}: gradient {worst} is not zero");

                // Move the normaliser alone and the gradient must wake up.
                let (_, _, off) = nce_population_gradient(&g, z + 0.5, &pd, &nm, ratio, &edges);
                assert!(off.abs() > 1e-3, "n={n} ratio={ratio}: c is not being fitted");
            }
        }
    }

    /// And the vanishing point is a MAXIMUM, not a saddle: the population objective is strictly
    /// lower at every perturbation of every coordinate, the normaliser included.
    #[test]
    fn the_truth_maximises_the_population_objective() {
        let n = 5;
        let g = truth(n);
        let pd = exact_boltzmann(&g, 1.0);
        let z = log_z(&g);
        let nm = NoiseModel::new(Elimination::default().marginals(&g, 1.0).unwrap());
        let best = nce_population(&g, z, &pd, &nm, 2.0);
        let edges = edges(&g);
        for k in 0..edges.len() + n {
            for d in [-0.25, 0.25] {
                let j = nce_population(&perturb(&g, k, d), z, &pd, &nm, 2.0);
                assert!(j < best, "coordinate {k} at {d}: {j} >= {best}");
            }
        }
        for d in [-0.25, 0.25] {
            let j = nce_population(&g, z + d, &pd, &nm, 2.0);
            assert!(j < best, "normaliser at {d}: {j} >= {best}");
        }
    }

    /// THE POINT OF THE METHOD: `c` is a free parameter, and it converges on the true `ln Z`.
    ///
    /// Nothing in the objective knows what `ln Z` is. The fitted value is compared against
    /// [`Elimination::log_partition`] on the model that generated the data, and the gap must shrink
    /// with data — a fixed tolerance at one data size would pass for a `c` that had merely stopped
    /// somewhere.
    ///
    /// Measured, mean `|c − ln Z|` over sixteen seeds at 400 epochs — the fit is converged well
    /// before then, 400, 1000 and 2500 epochs agreeing to every digit printed:
    ///
    /// ```text
    ///   rows      125      500     2000
    ///   gap    0.1299   0.0584   0.0298
    /// ```
    #[test]
    fn the_learned_normaliser_converges_to_the_true_log_partition() {
        let n = 5;
        let g = truth(n);
        let z = log_z(&g);
        let p = FitParams { epochs: 400, lr: 0.2, l2: 0.0 };
        let mut gaps = Vec::new();
        for rows in [125usize, 500, 2000] {
            let mut acc = 0.0;
            for seed in 0..16u64 {
                let data = Dataset { visible: n, rows: draws(&g, rows, 900 + seed) };
                let fit =
                    train_nce(&structure(n), &data, &p, Noise::DataMarginals, 5, 40 + seed).unwrap();
                acc += (fit.log_normaliser - z).abs();
            }
            gaps.push(acc / 16.0);
        }
        assert!(gaps[2] < gaps[1] && gaps[1] < gaps[0], "normaliser gaps do not shrink: {gaps:?}");
        assert!(gaps[2] < 0.45 * gaps[0], "normaliser gap barely moved: {gaps:?}");
        assert!(gaps[2] < 0.05, "normaliser still {} from ln Z at 2000 rows", gaps[2]);
    }

    /// The learned normaliser also matches the FITTED model's own `ln Z`, which is the sharper
    /// statement: it says the fit is self-consistent rather than that two errors cancelled.
    #[test]
    fn the_learned_normaliser_matches_the_fitted_models_own_log_partition() {
        let n = 5;
        let g = truth(n);
        let data = Dataset { visible: n, rows: draws(&g, 6000, 77) };
        let p = FitParams { epochs: 600, lr: 0.2, l2: 0.0 };
        let fit = train_nce(&structure(n), &data, &p, Noise::DataMarginals, 5, 3).unwrap();
        let err = fit.normaliser_error(24).unwrap();
        assert!(err.abs() < 0.02, "learned c is {err} from the fitted graph's own ln Z");
    }

    /// CONSISTENCY: parameter error shrinks as data grows, on exact draws from a known model.
    ///
    /// Averaged over seeds because a single fit's error is itself a random variable, and a
    /// one-seed comparison of two data sizes measures the seeds as much as the sizes.
    #[test]
    fn the_parameter_error_shrinks_as_data_grows() {
        let n = 5;
        let g = truth(n);
        let want = edges(&g);
        let p = FitParams { epochs: 400, lr: 0.2, l2: 0.0 };
        let mut errs = Vec::new();
        for rows in [125usize, 500, 2000] {
            let mut acc = 0.0;
            for seed in 0..16u64 {
                let data = Dataset { visible: n, rows: draws(&g, rows, 300 + seed) };
                let fit =
                    train_nce(&structure(n), &data, &p, Noise::DataMarginals, 5, 7 + seed).unwrap();
                let got = edges(&fit.graph);
                let mut sq = 0.0;
                for (a, b) in got.iter().zip(&want) {
                    sq += (a.2 - b.2) * (a.2 - b.2);
                }
                for i in 0..n {
                    sq += (fit.graph.h[i] - g.h[i]) * (fit.graph.h[i] - g.h[i]);
                }
                acc += (sq / (want.len() + n) as f64).sqrt();
            }
            errs.push(acc / 16.0);
        }
        assert!(errs[2] < errs[1] && errs[1] < errs[0], "parameter error does not shrink: {errs:?}");
        assert!(errs[2] < 0.45 * errs[0], "parameter error barely moved: {errs:?}");
    }

    /// SCORED AGAINST THE CEILING, not against the other estimators.
    ///
    /// `train_exact` ascends the true likelihood gradient with both averages enumerated, so it is
    /// the best this structure can do on this data. NCE is reported as a fraction of the reachable
    /// range between the untrained model and that ceiling — a likelihood alone says what a fit
    /// achieved and not what there was to achieve.
    ///
    /// Measured, root-mean-square parameter error over sixteen seeds at 400 epochs:
    ///
    /// ```text
    ///   rows      125      500     2000
    ///   error  0.1344   0.0543   0.0307
    /// ```
    #[test]
    fn nce_reaches_most_of_the_exact_maximum_likelihood_ceiling() {
        let n = 5;
        let g = truth(n);
        let data = Dataset { visible: n, rows: draws(&g, 4000, 21) };
        let floor = exact_log_likelihood(&structure(n), &data).unwrap();

        let ceiling = train_exact(&structure(n), &data, &FitParams { epochs: 3000, lr: 0.2, l2: 0.0 })
            .unwrap()
            .log_likelihood
            .unwrap();
        let got = train_nce(
            &structure(n),
            &data,
            &FitParams { epochs: 800, lr: 0.2, l2: 0.0 },
            Noise::DataMarginals,
            5,
            9,
        )
        .unwrap()
        .log_likelihood
        .unwrap();

        assert!(ceiling > floor, "the ceiling must beat an untrained model");
        let frac = (got - floor) / (ceiling - floor);
        assert!(frac > 0.98, "NCE reached {frac} of the exact-ML range ({got} in [{floor}, {ceiling}])");
        assert!(got <= ceiling + 1e-9, "NCE beat the exact maximum likelihood, which is impossible");
    }

    /// Uniform noise is the harder contrast and it still works here, which is worth pinning: it is
    /// the one noise choice that reads nothing at all from the data.
    #[test]
    fn uniform_noise_also_finds_the_normaliser() {
        let n = 5;
        let g = truth(n);
        let z = log_z(&g);
        let data = Dataset { visible: n, rows: draws(&g, 5000, 55) };
        let p = FitParams { epochs: 1500, lr: 0.2, l2: 0.0 };
        let fit = train_nce(&structure(n), &data, &p, Noise::Uniform, 5, 12).unwrap();
        assert!(
            (fit.log_normaliser - z).abs() < 0.1,
            "uniform-noise c is {} against ln Z {z}",
            fit.log_normaliser
        );
    }

    /// Latent units are refused rather than silently mis-fitted, and malformed data is caught.
    #[test]
    fn the_refusals_name_what_is_wrong() {
        let p = FitParams::default();
        let s = structure(4);
        let data = Dataset { visible: 3, rows: vec![vec![1, -1, 1]] };
        assert_eq!(
            train_nce(&s, &data, &p, Noise::Uniform, 1, 0).unwrap_err(),
            Error::HasLatent { spins: 4, visible: 3 }
        );
        let empty = Dataset { visible: 4, rows: Vec::new() };
        assert_eq!(train_nce(&s, &empty, &p, Noise::Uniform, 1, 0).unwrap_err(), Error::NoData);
        let bad = Dataset { visible: 4, rows: vec![vec![1, -1, 0, 1]] };
        assert_eq!(
            train_nce(&s, &bad, &p, Noise::Uniform, 1, 0).unwrap_err(),
            Error::NotASpin { row: 0, at: 2, value: 0 }
        );
    }

    /// The noise density is a normalised distribution over the `2^n` states, which is the one
    /// property the whole objective rests on: `δ` is a ratio against it.
    #[test]
    fn the_noise_density_sums_to_one() {
        let n = 6;
        let nm = NoiseModel::new((0..n).map(|i| 0.15 + 0.12 * i as f64).collect());
        let mut s = vec![-1i8; n];
        let mut acc = 0.0;
        for mask in 0..(1usize << n) {
            for (i, x) in s.iter_mut().enumerate() {
                *x = if mask >> i & 1 == 1 { 1 } else { -1 };
            }
            acc += nm.log_density(&s).exp();
        }
        assert!((acc - 1.0).abs() < 1e-12, "noise density sums to {acc}");

        // Uniform is the flat member of the same family.
        let u = NoiseModel::fit(Noise::Uniform, n, &Dataset { visible: n, rows: vec![vec![1i8; n]] });
        assert!((u.log_density(&s) + (n as f64) * 2.0f64.ln()).abs() < 1e-12);
    }

    /// The finite-sample objective is the population one's estimator, so on a sample large enough
    /// it must land near it — with BOTH sides computed from the same fixed noise draws.
    ///
    /// This is an agreement between an average and the integral it estimates, not between two
    /// samplers: the population side enumerates every state and does no sampling at all.
    #[test]
    fn the_sample_objective_approaches_the_population_one() {
        let n = 5;
        let g = truth(n);
        let pd = exact_boltzmann(&g, 1.0);
        let data = Dataset { visible: n, rows: draws(&g, 40000, 4) };
        let nm = NoiseModel::fit(Noise::Uniform, n, &data);
        let drawn = noise_rows(&nm, 120_000, 6);
        let c = log_z(&g);
        let sample = nce_objective(&g, c, &data, &nm, &drawn).unwrap();
        let population = nce_population(&g, c, &pd, &nm, 3.0);
        assert!((sample - population).abs() < 0.01, "{sample} vs {population}");
    }
}
