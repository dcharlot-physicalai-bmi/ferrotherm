//! Bayesian logistic regression — where a classifier's **confidence** comes from, and the two
//! cheap ways of getting it wrong.
//!
//! The third application in this crate, and it was chosen by a rule the first two produced rather
//! than by taste. Restoring an image: a max-flow owns the most probable labelling, and sampling
//! earns its place on the per-pixel marginals ([`crate::apps::restore`]). A Gaussian factor graph:
//! message passing owns the mean exactly and is overconfident about every variance
//! ([`crate::gbp`]). **A deterministic method owns the first moment; the sampler earns its place on
//! the second.** So the next thing to build is a task whose value is a second moment, and in
//! machine learning that task is calibration: not what the classifier says, but how sure it is.
//!
//! # The model
//!
//! Data `(x_i, y_i)` with `y_i ∈ {0, 1}`, weights `w` with a Gaussian prior. Writing
//! `s_i = 2y_i − 1`,
//!
//! ```text
//!   U(w) = Σ_i softplus(−s_i · x_iᵀw)  +  ‖w‖² / (2σ²),      p(w | D) ∝ exp(−U(w))
//! ```
//!
//! and the thing actually wanted is never `w`. It is the **predictive probability** at a new point,
//!
//! ```text
//!   p(y = 1 | x, D) = ∫ sigmoid(xᵀw) p(w | D) dw
//! ```
//!
//! an integral over the whole posterior. There is no closed form for it in any dimension.
//!
//! # The three routes, and what each one is really doing
//!
//! * [`Route::PlugIn`] — find the most probable `w` and use it. This is what a trained classifier
//!   does: maximum a posteriori, one point, no integral. **The posterior's width is discarded.**
//! * [`Route::Laplace`] — keep the mode and a Gaussian fitted to the curvature there, then
//!   integrate that. The standard cheap uncertainty, and the one most "Bayesian deep learning"
//!   reduces to in practice.
//! * [`Route::Sampling`] — draw from `p(w | D)` and average `sigmoid(xᵀw)` over the draws. The
//!   integral, done by the method a Langevin machine performs natively.
//!
//! # The oracle
//!
//! In two dimensions the posterior can be integrated on a grid to whatever accuracy is asked for,
//! and [`Route::Exact`] does that — refusing outright above two dimensions rather than returning a
//! quadrature that has quietly stopped converging. Two dimensions is not a limitation of the
//! argument; it is what makes the argument checkable, and the failure modes below are not
//! dimension-dependent. They get worse with dimension.

use crate::hmc::Target;

/// A binary classification problem with a Gaussian prior on the weights.
#[derive(Clone, Debug)]
pub struct Logit {
    /// Weight count; also the width of each row of `x`.
    pub d: usize,
    /// Design matrix, row-major, `n · d`. No intercept is added: put a constant column in if one
    /// is wanted, and it then carries the prior like any other weight.
    pub x: Vec<f64>,
    /// Labels, one per row.
    pub y: Vec<bool>,
    /// Prior standard deviation on each weight, shared.
    pub prior_sd: f64,
}

/// Which route to the predictive probability.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Route {
    /// Integrate the posterior on a grid. Exact to the grid's own accuracy; two dimensions only.
    Exact {
        /// Points per axis.
        grid: usize,
        /// Half-width of the box around the mode, in prior standard deviations.
        half_width: f64,
    },
    /// The most probable weights, used as though they were the only ones.
    PlugIn,
    /// A Gaussian fitted at the mode, integrated exactly along the query direction.
    Laplace,
    /// Draw from the posterior and average the prediction over the draws.
    Sampling {
        /// Recorded draws.
        draws: usize,
        /// Warm-up draws, discarded.
        warmup: usize,
        /// Leapfrog step size.
        eps: f64,
        /// Leapfrog steps per draw.
        steps: usize,
    },
}

/// Why a route could not be taken.
#[derive(Clone, Debug, PartialEq)]
pub enum Ill {
    /// A vector's length disagrees with `d` or with the label count.
    Shape,
    /// The prior standard deviation is not positive and finite. An improper prior on separable
    /// data has no posterior mode at all, so this is refused rather than allowed to diverge.
    Prior(f64),
    /// Exact quadrature was asked for above two dimensions, where a grid stops being a reference.
    TooManyDimensions(usize),
    /// The mode search did not converge; carries the last gradient norm.
    NoMode(f64),
    /// The sampler refused its own arguments.
    Sampler(String),
}

impl core::fmt::Display for Ill {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Ill::Shape => write!(f, "a vector's length disagrees with the model's shape"),
            Ill::Prior(s) => write!(f, "a prior standard deviation of {s} is not positive and finite"),
            Ill::TooManyDimensions(d) => write!(
                f,
                "grid quadrature is a reference in two dimensions and not in {d}; it is refused \
                 rather than returned as one"
            ),
            Ill::NoMode(g) => write!(f, "the mode search did not converge; gradient norm {g:e}"),
            Ill::Sampler(m) => write!(f, "{m}"),
        }
    }
}

impl core::error::Error for Ill {}

/// Predictions and what they cost.
pub struct Predictions {
    /// `p(y = 1 | x, D)` per query point.
    pub probability: Vec<f64>,
    /// The route taken.
    pub route: Route,
    /// Gradient evaluations, posterior draws, or grid points — whichever the route spends.
    pub work: u64,
}

impl Predictions {
    /// How far from `0.5` the most confident prediction is. **The number the two cheap routes
    /// inflate**, and the one word "calibration" mostly means in practice.
    #[must_use]
    pub fn sharpest(&self) -> f64 {
        self.probability
            .iter()
            .map(|p| (p - 0.5).abs())
            .fold(0.0f64, f64::max)
    }
}

fn sigmoid(a: f64) -> f64 {
    if a >= 0.0 { 1.0 / (1.0 + (-a).exp()) } else { a.exp() / (1.0 + a.exp()) }
}

fn softplus(a: f64) -> f64 {
    if a > 0.0 { a + (-a).exp().ln_1p() } else { a.exp().ln_1p() }
}

impl Logit {
    /// Rows in the data set.
    #[must_use]
    pub fn n(&self) -> usize {
        self.y.len()
    }

    /// Check the shapes and the prior.
    ///
    /// # Errors
    ///
    /// [`Ill::Shape`] or [`Ill::Prior`].
    pub fn check(&self) -> Result<(), Ill> {
        if self.d == 0 || self.x.len() != self.y.len() * self.d {
            return Err(Ill::Shape);
        }
        if !(self.prior_sd > 0.0) || !self.prior_sd.is_finite() {
            return Err(Ill::Prior(self.prior_sd));
        }
        Ok(())
    }

    /// `xᵀw` for row `i`.
    fn row_dot(&self, i: usize, w: &[f64]) -> f64 {
        (0..self.d).map(|j| self.x[i * self.d + j] * w[j]).sum()
    }

    /// The negative log posterior, up to a constant.
    #[must_use]
    pub fn potential(&self, w: &[f64]) -> f64 {
        let mut u = 0.0;
        for i in 0..self.n() {
            let s = if self.y[i] { 1.0 } else { -1.0 };
            u += softplus(-s * self.row_dot(i, w));
        }
        let v = self.prior_sd * self.prior_sd;
        u + w.iter().map(|a| a * a).sum::<f64>() / (2.0 * v)
    }

    /// The posterior mode, by damped Newton. The starting point for both cheap routes.
    ///
    /// # Errors
    ///
    /// [`Ill::Shape`], [`Ill::Prior`], or [`Ill::NoMode`] if the gradient never got small.
    pub fn mode(&self) -> Result<Vec<f64>, Ill> {
        self.check()?;
        let d = self.d;
        let inv_v = 1.0 / (self.prior_sd * self.prior_sd);
        let mut w = vec![0.0; d];
        let mut last = f64::INFINITY;
        for _ in 0..200 {
            let mut g = vec![0.0; d];
            let mut h = vec![0.0; d * d];
            for j in 0..d {
                g[j] = w[j] * inv_v;
                h[j * d + j] = inv_v;
            }
            for i in 0..self.n() {
                let s = if self.y[i] { 1.0 } else { -1.0 };
                let a = self.row_dot(i, &w);
                // d/dw softplus(-s a) = -s x sigmoid(-s a)
                let c = -s * sigmoid(-s * a);
                let p = sigmoid(a);
                let wt = p * (1.0 - p);
                for j in 0..d {
                    let xj = self.x[i * d + j];
                    g[j] += c * xj;
                    for k in 0..d {
                        h[j * d + k] += wt * xj * self.x[i * d + k];
                    }
                }
            }
            last = g.iter().map(|v| v * v).sum::<f64>().sqrt();
            if last < 1e-12 {
                return Ok(w);
            }
            let step = solve_small(&h, &g, d).ok_or(Ill::NoMode(last))?;
            // Damped: the Hessian is exact and positive definite here, but a long first step on a
            // nearly separable data set overshoots into a flat region the next Newton step cannot
            // return from.
            let mut t = 1.0;
            let u0 = self.potential(&w);
            let mut next = w.clone();
            for _ in 0..40 {
                for j in 0..d {
                    next[j] = w[j] - t * step[j];
                }
                if self.potential(&next) <= u0 {
                    break;
                }
                t *= 0.5;
            }
            w = next;
        }
        Err(Ill::NoMode(last))
    }

    /// The posterior covariance of a Laplace approximation: the inverse Hessian at the mode.
    ///
    /// # Errors
    ///
    /// As [`Logit::mode`], or if the Hessian turned out singular.
    pub fn laplace_covariance(&self, w: &[f64]) -> Result<Vec<f64>, Ill> {
        self.check()?;
        let d = self.d;
        let inv_v = 1.0 / (self.prior_sd * self.prior_sd);
        let mut h = vec![0.0; d * d];
        for j in 0..d {
            h[j * d + j] = inv_v;
        }
        for i in 0..self.n() {
            let p = sigmoid(self.row_dot(i, w));
            let wt = p * (1.0 - p);
            for j in 0..d {
                for k in 0..d {
                    h[j * d + k] += wt * self.x[i * d + j] * self.x[i * d + k];
                }
            }
        }
        invert_small(&h, d).ok_or(Ill::NoMode(f64::NAN))
    }

    /// Predict `p(y = 1 | x, D)` at every row of `queries` (row-major, `d` wide).
    ///
    /// # Errors
    ///
    /// An [`Ill`] naming what was ill-formed, refused, or would not converge.
    pub fn predict(&self, queries: &[f64], route: Route, seed: u64) -> Result<Predictions, Ill> {
        self.check()?;
        let d = self.d;
        if !queries.len().is_multiple_of(d) {
            return Err(Ill::Shape);
        }
        let q: Vec<&[f64]> = queries.chunks(d).collect();
        match route {
            Route::Exact { grid, half_width } => {
                if d != 2 {
                    return Err(Ill::TooManyDimensions(d));
                }
                let centre = self.mode()?;
                let hw = half_width * self.prior_sd;
                let step = 2.0 * hw / (grid - 1).max(1) as f64;
                let mut num = vec![0.0; q.len()];
                let mut den = 0.0f64;
                // Weights are exp(-U) relative to the mode, so nothing overflows.
                let u0 = self.potential(&centre);
                let mut w = vec![0.0; 2];
                for a in 0..grid {
                    w[0] = centre[0] - hw + step * a as f64;
                    for b in 0..grid {
                        w[1] = centre[1] - hw + step * b as f64;
                        let p = (-(self.potential(&w) - u0)).exp();
                        den += p;
                        for (k, qq) in q.iter().enumerate() {
                            let a_q: f64 = (0..d).map(|j| qq[j] * w[j]).sum();
                            num[k] += p * sigmoid(a_q);
                        }
                    }
                }
                let probability = num.iter().map(|v| v / den).collect();
                Ok(Predictions { probability, route, work: (grid * grid) as u64 })
            }
            Route::PlugIn => {
                let w = self.mode()?;
                let probability = q
                    .iter()
                    .map(|qq| sigmoid((0..d).map(|j| qq[j] * w[j]).sum()))
                    .collect();
                Ok(Predictions { probability, route, work: 0 })
            }
            Route::Laplace => {
                let w = self.mode()?;
                let cov = self.laplace_covariance(&w)?;
                // Along one query direction the posterior is a 1-D Gaussian on the logit, and the
                // remaining integral is done by quadrature rather than by the usual probit
                // approximation -- so what this route reports is the Gaussian POSTERIOR's error
                // and nothing else's.
                let probability = q
                    .iter()
                    .map(|qq| {
                        let mu: f64 = (0..d).map(|j| qq[j] * w[j]).sum();
                        let mut var = 0.0;
                        for j in 0..d {
                            for k in 0..d {
                                var += qq[j] * cov[j * d + k] * qq[k];
                            }
                        }
                        gauss_logistic(mu, var.max(0.0).sqrt())
                    })
                    .collect();
                Ok(Predictions { probability, route, work: 0 })
            }
            Route::Sampling { draws, warmup, eps, steps } => {
                let start = self.mode()?;
                let mut h = crate::hmc::Hmc::new(self, &start, eps, steps, seed)
                    .map_err(|e| Ill::Sampler(e.to_string()))?;
                for _ in 0..warmup {
                    h.draw();
                }
                let mut acc = vec![0.0; q.len()];
                for _ in 0..draws {
                    let w = h.draw().to_vec();
                    for (k, qq) in q.iter().enumerate() {
                        let a: f64 = (0..d).map(|j| qq[j] * w[j]).sum();
                        acc[k] += sigmoid(a);
                    }
                }
                let probability = acc.iter().map(|v| v / draws as f64).collect();
                Ok(Predictions { probability, route, work: h.gradients() })
            }
        }
    }
}

/// `∫ sigmoid(a) N(a | mu, sd²) da`, by Gauss-Hermite on 64 nodes — enough that this is the
/// Gaussian's error being measured and not the quadrature's.
fn gauss_logistic(mu: f64, sd: f64) -> f64 {
    if sd <= 0.0 {
        return sigmoid(mu);
    }
    // Trapezoid on ±8 sd is simpler than Hermite nodes and, on a smooth integrand over a finite
    // effective support, converges geometrically. 2001 points is far past where it stops moving.
    let n = 2001;
    let lo = mu - 8.0 * sd;
    let hi = mu + 8.0 * sd;
    let h = (hi - lo) / (n - 1) as f64;
    let mut num = 0.0;
    let mut den = 0.0;
    for i in 0..n {
        let a = lo + h * i as f64;
        let z = (a - mu) / sd;
        let wgt = (-0.5 * z * z).exp();
        num += wgt * sigmoid(a);
        den += wgt;
    }
    num / den
}

impl Target for Logit {
    fn dim(&self) -> usize {
        self.d
    }

    fn potential(&self, q: &[f64]) -> f64 {
        Logit::potential(self, q)
    }

    fn grad(&self, q: &[f64], out: &mut [f64]) {
        let inv_v = 1.0 / (self.prior_sd * self.prior_sd);
        for j in 0..self.d {
            out[j] = q[j] * inv_v;
        }
        for i in 0..self.n() {
            let s = if self.y[i] { 1.0 } else { -1.0 };
            let c = -s * sigmoid(-s * self.row_dot(i, q));
            for j in 0..self.d {
                out[j] += c * self.x[i * self.d + j];
            }
        }
    }
}

/// Solve `H step = g` for small dense `H`, by Gauss-Jordan.
fn solve_small(h: &[f64], g: &[f64], d: usize) -> Option<Vec<f64>> {
    let inv = invert_small(h, d)?;
    Some((0..d).map(|i| (0..d).map(|j| inv[i * d + j] * g[j]).sum()).collect())
}

fn invert_small(h: &[f64], d: usize) -> Option<Vec<f64>> {
    let mut aug = vec![0.0; d * 2 * d];
    for i in 0..d {
        for j in 0..d {
            aug[i * 2 * d + j] = h[i * d + j];
        }
        aug[i * 2 * d + d + i] = 1.0;
    }
    for col in 0..d {
        let mut piv = col;
        for r in col + 1..d {
            if aug[r * 2 * d + col].abs() > aug[piv * 2 * d + col].abs() {
                piv = r;
            }
        }
        if aug[piv * 2 * d + col].abs() < 1e-300 {
            return None;
        }
        if piv != col {
            for k in 0..2 * d {
                aug.swap(col * 2 * d + k, piv * 2 * d + k);
            }
        }
        let p = aug[col * 2 * d + col];
        for k in 0..2 * d {
            aug[col * 2 * d + k] /= p;
        }
        for r in 0..d {
            if r == col {
                continue;
            }
            let f = aug[r * 2 * d + col];
            if f == 0.0 {
                continue;
            }
            for k in 0..2 * d {
                aug[r * 2 * d + k] -= f * aug[col * 2 * d + k];
            }
        }
    }
    Some((0..d).flat_map(|i| (d..2 * d).map(move |j| (i, j))).map(|(i, j)| aug[i * 2 * d + j]).collect())
}

/// Two well-separated clusters: the case every classifier meets and the case both cheap routes
/// handle worst, because a separating direction the data does not pin down is exactly where the
/// posterior is broad and a single point is silent about it.
#[must_use]
pub fn separated(n_per_class: usize, gap: f64, prior_sd: f64) -> Logit {
    let mut x = Vec::new();
    let mut y = Vec::new();
    for k in 0..n_per_class {
        let t = if n_per_class == 1 { 0.0 } else { k as f64 / (n_per_class - 1) as f64 - 0.5 };
        x.extend_from_slice(&[gap, t]);
        y.push(true);
        x.extend_from_slice(&[-gap, t]);
        y.push(false);
    }
    Logit { d: 2, x, y, prior_sd }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EXACT: Route = Route::Exact { grid: 801, half_width: 6.0 };

    /// The oracle must be an oracle. A grid quadrature is only a reference where refining it stops
    /// moving the answer, and where the box is wide enough to hold the mass — both checked here,
    /// because everything below is measured against it.
    #[test]
    fn the_quadrature_oracle_has_stopped_moving_before_anything_is_measured_against_it() {
        let m = separated(4, 1.0, 3.0);
        let queries = [0.0, 0.0, 0.5, 0.3, 2.0, -1.0, -0.2, 1.5];
        let mut prev: Option<Vec<f64>> = None;
        let mut deltas = Vec::new();
        for grid in [201usize, 401, 801] {
            let p = m
                .predict(&queries, Route::Exact { grid, half_width: 6.0 }, 1)
                .expect("two dimensions")
                .probability;
            if let Some(q) = &prev {
                deltas.push(
                    p.iter().zip(q).map(|(a, b)| (a - b).abs()).fold(0.0f64, f64::max),
                );
            }
            prev = Some(p);
        }
        assert!(
            deltas.iter().all(|&d| d < 1e-6),
            "refining the grid still moves the oracle: {deltas:?}"
        );
        // And widening the box does not move it either, so the tails are inside.
        let wide = m
            .predict(&queries, Route::Exact { grid: 801, half_width: 9.0 }, 1)
            .expect("two dimensions")
            .probability;
        let narrow = prev.as_ref().expect("ran");
        let d = wide.iter().zip(narrow).map(|(a, b)| (a - b).abs()).fold(0.0f64, f64::max);
        assert!(d < 1e-6, "widening the box moves the oracle by {d:e}: the tails are outside it");

        // A CONVERGENCE CHECK IS NOT A CORRECTNESS CHECK, and the two above are only convergence.
        // A mutant that averaged the LOGIT instead of the probability -- `p * a_q` for
        // `p * sigmoid(a_q)` -- passed both of them, because a wrong integrand converges just as
        // obediently as a right one. What follows is the part that has an independent answer.
        //
        // WITH NO DATA the posterior IS the prior, so the predictive at `x` is a one-dimensional
        // Gaussian-logistic integral: `∫ sigmoid(a) N(a | 0, σ²‖x‖²) da`. That comes out of a
        // different routine, on a different grid, in a different number of dimensions.
        let empty = Logit { d: 2, x: Vec::new(), y: Vec::new(), prior_sd: 1.7 };
        let probe = [1.0, 0.0, 0.0, 2.0, 1.5, -1.5, 0.3, 0.1];
        let got = empty.predict(&probe, EXACT, 1).expect("an empty data set is still a posterior");
        for (k, q) in probe.chunks(2).enumerate() {
            let sd = empty.prior_sd * (q[0] * q[0] + q[1] * q[1]).sqrt();
            let want = super::gauss_logistic(0.0, sd);
            assert!(
                (got.probability[k] - want).abs() < 1e-6,
                "no data, query {k}: quadrature {} vs the prior's own integral {want}",
                got.probability[k]
            );
        }
        // And a probability is a probability. The logit-averaging mutant reaches 7.75 here.
        for p in &got.probability {
            assert!((0.0..=1.0).contains(p), "a predictive probability of {p}");
        }
        for p in narrow {
            assert!((0.0..=1.0).contains(p), "a predictive probability of {p}");
        }

        // Above two dimensions it refuses rather than pretending.
        let mut three = m.clone();
        three.d = 3;
        three.x = vec![0.0; three.y.len() * 3];
        assert_eq!(
            three.predict(&[0.0, 0.0, 0.0], EXACT, 1).err(),
            Some(Ill::TooManyDimensions(3))
        );
    }

    /// **THE FINDING — AND IT IS NOT THE ONE I EXPECTED.** I wrote this test asserting that both
    /// cheap routes are overconfident, which is the received wisdom. The measurement says
    /// otherwise, and the truth is more useful: on this problem the plug-in is **over**confident
    /// and Laplace is **under**confident, at the same query, on the same data.
    ///
    /// So the two cheap routes bracket the answer without either one bounding it. Neither can be
    /// made safe by a margin, because a margin has to point somewhere. That is a stronger reason
    /// to do the integral than "the cheap one is too sure", which a conservative fudge would have
    /// answered.
    ///
    /// | query | exact | plug-in | Laplace |
    /// |---|---|---|---|
    /// | `(0.5, 2.5)`, away from the data | `0.614` | **`0.825`** | `0.593` |
    /// | `(-3.0, 1.0)` | `0.0072` | **`0.00009`** | `0.045` |
    #[test]
    fn the_cheap_routes_get_the_decision_right_and_bracket_the_confidence_without_bounding_it() {
        let m = separated(4, 1.0, 3.0);
        // On the boundary, near the data, and out where the data says little -- which is where a
        // discarded posterior width does the most damage.
        let queries = [0.0, 0.0, 0.8, 0.0, 2.5, 0.0, 0.5, 2.5, -3.0, 1.0];
        let exact = m.predict(&queries, EXACT, 1).expect("two dimensions");
        let plug = m.predict(&queries, Route::PlugIn, 1).expect("a mode exists");
        let lap = m.predict(&queries, Route::Laplace, 1).expect("a mode exists");

        // THE DECISION IS THE SAME. Every route puts every query on the same side of 0.5, so what
        // follows is about confidence and not about accuracy.
        for (k, e) in exact.probability.iter().enumerate() {
            for other in [&plug, &lap] {
                assert_eq!(
                    (other.probability[k] - 0.5).signum(),
                    (e - 0.5).signum(),
                    "route {:?} changed the DECISION at query {k}",
                    other.route
                );
            }
        }

        // THE CONFIDENCE IS NOT, AND THE TWO ROUTES MISS IN OPPOSITE DIRECTIONS.
        let conf = |p: &Predictions, k: usize| (p.probability[k] - 0.5).abs();
        let mut plug_over = 0usize;
        let mut lap_under = 0usize;
        let mut opposed = 0usize;
        let mut worst_plug = 0.0f64;
        for k in 0..exact.probability.len() {
            let t = conf(&exact, k);
            if t < 1e-6 {
                continue; // a query exactly on the boundary has no confidence to get wrong
            }
            let (dp, dl) = (conf(&plug, k) - t, conf(&lap, k) - t);
            if dp > 1e-9 {
                plug_over += 1;
                worst_plug = worst_plug.max(dp);
            }
            if dl < -1e-9 {
                lap_under += 1;
            }
            if dp * dl < 0.0 {
                opposed += 1;
            }
        }
        assert!(plug_over >= 3, "the plug-in must be overconfident where the data is thin: {plug_over}");
        assert_eq!(lap_under, 4, "and Laplace UNDERconfident at every query that has a confidence");
        assert!(
            opposed >= 3,
            "the point of this test: the two cheap routes must miss in OPPOSITE directions, so \
             neither bounds the other -- they agreed in direction at all but {opposed} queries"
        );
        assert!(
            worst_plug > 0.2,
            "and the plug-in's overconfidence must be worth reporting: {worst_plug:.3}"
        );

        let worst = |p: &Predictions| {
            p.probability
                .iter()
                .zip(&exact.probability)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f64, f64::max)
        };
        eprintln!(
            "worst predictive error: plug-in {:.4} (overconfident at {plug_over}/4), \
             Laplace {:.4} (underconfident at {lap_under}/4), opposed at {opposed}/4",
            worst(&plug),
            worst(&lap)
        );

        // AND THE SAMPLER CLOSES IT. Its error is a standard error, so it shrinks; the other two
        // approximate the posterior itself, so theirs do not shrink with anything.
        //
        // AVERAGED OVER SEEDS, because one chain's error at one budget is itself a random variable.
        // The single-seed version of this test read 0.0037 / 0.0088 / 0.0017 at 5e2 / 5e3 / 5e4
        // draws -- not monotone, and its "the error halved" assertion passed by 0.0002. That is a
        // coin toss wearing a convergence test's clothes.
        let rms = |p: &Predictions| -> f64 {
            let n = p.probability.len() as f64;
            (p.probability
                .iter()
                .zip(&exact.probability)
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f64>()
                / n)
                .sqrt()
        };
        let seeds = 8u64;
        let mut errs = Vec::new();
        for draws in [500usize, 5_000, 50_000] {
            let mut acc = 0.0;
            for seed in 0..seeds {
                let s = m
                    .predict(
                        &queries,
                        Route::Sampling { draws, warmup: 500, eps: 0.25, steps: 12 },
                        0xDA7A ^ seed.wrapping_mul(0x9E37_79B9),
                    )
                    .expect("the mode is a good start");
                acc += rms(&s);
            }
            errs.push(acc / seeds as f64);
        }
        // A hundred times the draws should buy about ten times the accuracy. Four is a wide margin
        // on that and still impossible for an estimator that had stopped converging.
        assert!(
            errs[0] > 4.0 * errs[2],
            "a 100x budget bought {:.1}x, so this is not a standard error: {errs:?}",
            errs[0] / errs[2]
        );
        assert!(errs[0] > errs[1] && errs[1] > errs[2], "and it must fall at every step: {errs:?}");
        assert!(
            errs[2] < rms(&lap).min(rms(&plug)) / 5.0,
            "at 5e4 draws the sampler should be well inside BOTH cheap routes' errors: {errs:?}"
        );
        eprintln!(
            "sampler RMS predictive error over {seeds} seeds at 5e2 / 5e3 / 5e4 draws: \
             {:.5} / {:.5} / {:.5}  (a 100x budget bought {:.1}x); plug-in {:.5}, Laplace {:.5}",
            errs[0], errs[1], errs[2], errs[0] / errs[2], rms(&plug), rms(&lap)
        );
    }

    /// The model itself, checked against its own definitions rather than against its outputs: the
    /// gradient against central differences, and the potential against the formula it is written
    /// from. Without this, every comparison above could be a comparison between two wrong things.
    #[test]
    fn the_potential_and_its_gradient_are_what_the_model_says_they_are() {
        let m = separated(3, 1.2, 2.0);
        let w = [0.37, -0.81];
        // The potential, spelled out term by term.
        let mut want = (w[0] * w[0] + w[1] * w[1]) / (2.0 * 4.0);
        for i in 0..m.n() {
            let a = m.x[i * 2] * w[0] + m.x[i * 2 + 1] * w[1];
            let s = if m.y[i] { 1.0 } else { -1.0 };
            want += (1.0 + (-s * a).exp()).ln();
        }
        assert!((m.potential(&w) - want).abs() < 1e-12, "{} vs {want}", m.potential(&w));

        let mut g = [0.0; 2];
        Target::grad(&m, &w, &mut g);
        for j in 0..2 {
            let h = 1e-6;
            let (mut a, mut b) = (w, w);
            a[j] += h;
            b[j] -= h;
            let fd = (m.potential(&a) - m.potential(&b)) / (2.0 * h);
            assert!((g[j] - fd).abs() < 1e-7, "grad[{j}] {} vs central difference {fd}", g[j]);
        }

        // The prior is the only thing holding the weights on separable data, so its strength must
        // reach the answer: a tighter prior must give a less confident prediction.
        let q = [2.0, 0.0];
        let mut sharp = Vec::new();
        for sd in [0.5, 1.0, 3.0] {
            let tight = separated(4, 1.0, sd);
            sharp.push(tight.predict(&q, EXACT, 1).expect("two dimensions").sharpest());
        }
        assert!(
            sharp[0] < sharp[1] && sharp[1] < sharp[2],
            "a tighter prior must predict less confidently: {sharp:?}"
        );
    }

    /// Every refusal.
    #[test]
    fn an_ill_formed_problem_is_refused_by_name() {
        let ok = separated(3, 1.0, 2.0);
        assert!(ok.check().is_ok());
        let mut bad = ok.clone();
        bad.y.pop();
        assert_eq!(bad.check(), Err(Ill::Shape));
        let mut bad = ok.clone();
        bad.d = 0;
        assert_eq!(bad.check(), Err(Ill::Shape));
        // An improper prior on separable data has no mode; refusing the prior is how that is
        // caught before a Newton iteration walks off to infinity looking for one.
        for sd in [0.0, -1.0, f64::INFINITY, f64::NAN] {
            let mut bad = ok.clone();
            bad.prior_sd = sd;
            assert!(matches!(bad.check(), Err(Ill::Prior(_))), "prior_sd {sd}");
        }
        // A query row that is not `d` wide.
        assert_eq!(ok.predict(&[1.0], Route::PlugIn, 1).err(), Some(Ill::Shape));
        // And a step size the sampler will not take.
        assert!(matches!(
            ok.predict(&[0.0, 0.0], Route::Sampling { draws: 10, warmup: 1, eps: 0.0, steps: 4 }, 1),
            Err(Ill::Sampler(_))
        ));
    }
}
