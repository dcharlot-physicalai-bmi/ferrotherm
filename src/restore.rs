//! Bayesian image restoration on an Ising Markov random field, and the Nishimori condition.
//!
//! Geman & Geman, IEEE PAMI 6:721 (1984) — the paper the Gibbs sampler was introduced in, for
//! exactly this problem. A binary image `x` in `{-1,+1}^n` carries an Ising prior on a `w x h`
//! grid; a binary symmetric channel flips each pixel independently with probability `p`; the
//! restored image is read off the posterior.
//!
//! The posterior is itself an Ising model, which is why it belongs in this crate:
//!
//! ```text
//!   P(x | y)  ~  exp( k sum_<ij> x_i x_j  +  hc sum_i y_i x_i ),   hc = (1/2) ln((1-p)/p)
//! ```
//!
//! so [`Model::posterior`] returns a [`Graph`] read at beta = 1 whose couplings are the prior's and
//! whose fields are the data's. `hc` is not a tuning knob: it is the channel's exact
//! log-likelihood ratio, and `exp(+-hc) / (2 cosh hc)` reproduces `(1-p, p)` to the last bit.
//!
//! # The Nishimori condition
//!
//! [`Model::posterior`] takes a scale `t` multiplying BOTH terms — an inverse model temperature
//! with the prior-to-data ratio pinned at its true value, which is the Nishimori line. `t = 1`
//! ([`NISHIMORI`]) is where the model equals the source, and there the MPM estimate (the sign of
//! each posterior marginal, [`mpm`]) is Bayes-optimal: no estimator, at any temperature, gets
//! fewer pixels wrong in expectation.
//!
//! That claim is checked rather than cited. [`Model::expected_error`] enumerates the whole joint
//! `P(x) P(y|x)` and returns the expected pixel error EXACTLY — no sampling anywhere in it — so
//! the temperature sweep is a deterministic curve with its minimum at `t = 1`, the MAP estimate
//! ([`Model::expected_error_map`], the `t -> infinity` limit) sits strictly above it, and
//! [`Model::bayes_risk`], which never mentions an estimator at all, equals the value at `t = 1`.

use crate::exact::{Elimination, TooWide};
use crate::gibbs::Sampler;
use crate::graph::Graph;
use crate::ising::{exact_boltzmann, grid2d};
use crate::rng::Pcg;

/// The scale at which the posterior is the true one: prior and channel both at their real strength.
///
/// Named because a bare `1.0` at a call site says nothing, and this one is the whole module.
pub const NISHIMORI: f64 = 1.0;

/// Largest image the joint-enumeration routes accept: they cost `4^pixels`.
pub const MAX_JOINT_PIXELS: usize = 12;

/// Why a restoration was refused.
#[derive(Clone, Debug, PartialEq)]
pub enum Error {
    /// Flip probability outside `(0, 0.5)`. At `0.5` the data carries no information and the
    /// channel field is zero; at `0` it is infinite; above `0.5` the honest model negates `y`.
    Noise(f64),
    /// A model scale that is not finite and positive.
    Scale(f64),
    /// An image of the wrong length.
    Size {
        /// What was passed.
        got: usize,
        /// `w * h`.
        want: usize,
    },
    /// A pixel that is not `+1` or `-1`.
    NotASpin {
        /// Its index.
        at: usize,
        /// The offending value.
        value: i8,
    },
    /// More pixels than [`MAX_JOINT_PIXELS`], so the joint has more than `4^12` terms.
    TooBig {
        /// The image's pixel count.
        pixels: usize,
        /// [`MAX_JOINT_PIXELS`].
        max: usize,
    },
    /// Exact inference declined this posterior.
    Width(TooWide),
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Error::Noise(p) => write!(
                f,
                "flip probability {p} is outside (0, 0.5): at 0.5 the observation is independent of \
                 the image and the channel field is exactly zero, at 0 the field is infinite, and \
                 above 0.5 the honest model is the same channel with the observation negated"
            ),
            Error::Scale(t) => {
                write!(f, "model scale {t} must be finite and positive; t = 1 is the Nishimori point")
            }
            Error::Size { got, want } => write!(f, "image has {got} pixels, the model has {want}"),
            Error::NotASpin { at, value } => {
                write!(f, "pixel {at} is {value}; a binary image is +1 or -1")
            }
            Error::TooBig { pixels, max } => write!(
                f,
                "enumerating the joint over {pixels} pixels is 4^{pixels} terms; the limit is \
                 4^{max}. Sample the posterior with restore_gibbs and measure the error on one \
                 image instead"
            ),
            Error::Width(w) => write!(f, "{w}"),
        }
    }
}

/// Which rule reads an image off a posterior. Private: each has its own public entry point.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Rule {
    /// Maximum posterior marginal — the sign of each marginal, pixel by pixel.
    Mpm,
    /// Maximum a posteriori — the single most probable image.
    Map,
}

/// A binary image on a `w x h` grid with an Ising prior, seen through a binary symmetric channel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Model {
    /// Grid width in pixels.
    pub w: usize,
    /// Grid height in pixels.
    pub h: usize,
    /// Prior nearest-neighbour coupling, at beta = 1. Positive is smoothing (ferromagnetic).
    pub k: f64,
    /// Channel flip probability, in `(0, 0.5)`.
    pub p: f64,
}

impl Model {
    /// A model on `w x h` pixels with prior coupling `k` and channel noise `p`.
    ///
    /// # Errors
    ///
    /// [`Error::Noise`] outside `(0, 0.5)`, [`Error::Size`] on an empty grid, [`Error::Scale`] on a
    /// non-finite coupling.
    pub fn new(w: usize, h: usize, k: f64, p: f64) -> Result<Model, Error> {
        if w == 0 || h == 0 {
            return Err(Error::Size { got: 0, want: w * h });
        }
        if !k.is_finite() {
            return Err(Error::Scale(k));
        }
        if !(p > 0.0 && p < 0.5) {
            return Err(Error::Noise(p));
        }
        Ok(Model { w, h, k, p })
    }

    /// Pixel count, `w * h`.
    #[must_use]
    pub fn pixels(&self) -> usize {
        self.w * self.h
    }

    /// The channel's field `hc = (1/2) ln((1-p)/p)`, the exact log-likelihood ratio per pixel.
    #[must_use]
    pub fn channel_field(&self) -> f64 {
        0.5 * ((1.0 - self.p) / self.p).ln()
    }

    /// The prior alone: the `w x h` open grid at coupling `k`, no fields.
    #[must_use]
    pub fn prior(&self) -> Graph {
        grid2d(self.w, self.h, self.k)
    }

    /// The posterior given observation `y`, scaled by `t`, as a graph to be read at beta = 1.
    ///
    /// `t = 1` ([`NISHIMORI`]) is the true posterior; other `t` are the same model at another
    /// temperature, with the prior-to-data ratio held at its true value.
    ///
    /// # Errors
    ///
    /// [`Error::Size`] or [`Error::NotASpin`] on a bad image, [`Error::Scale`] on a bad scale.
    pub fn posterior(&self, y: &[i8], t: f64) -> Result<Graph, Error> {
        self.check(y)?;
        check_scale(t)?;
        // The grid supplies the topology, the CSR and the colouring; the observation only ever adds
        // fields, which none of those depend on. So this reuses `grid2d` and writes `h` directly
        // rather than restating the lattice.
        let mut g = grid2d(self.w, self.h, t * self.k);
        let hc = t * self.channel_field();
        for (gi, &yi) in g.h.iter_mut().zip(y) {
            *gi = hc * f64::from(yi);
        }
        Ok(g)
    }

    /// Push `x` through the channel: each pixel independently flipped with probability `p`.
    ///
    /// # Errors
    ///
    /// [`Error::Size`] or [`Error::NotASpin`] on a bad image.
    pub fn corrupt(&self, x: &[i8], rng: &mut Pcg) -> Result<Vec<i8>, Error> {
        self.check(x)?;
        Ok(x.iter().map(|&v| if rng.f64() < self.p { -v } else { v }).collect())
    }

    /// Exact posterior marginals `P(x_i = +1 | y)` at scale `t`, by variable elimination.
    ///
    /// # Errors
    ///
    /// As [`Model::posterior`], plus [`Error::Width`] when the grid is too wide to eliminate.
    pub fn posterior_marginals(&self, y: &[i8], t: f64) -> Result<Vec<f64>, Error> {
        let g = self.posterior(y, t)?;
        Elimination::default().marginals(&g, 1.0).map_err(Error::Width)
    }

    /// The MPM restoration at scale `t`, from exact marginals.
    ///
    /// # Errors
    ///
    /// As [`Model::posterior_marginals`].
    pub fn restore_exact(&self, y: &[i8], t: f64) -> Result<Vec<i8>, Error> {
        let m = self.posterior_marginals(y, t)?;
        Ok(mpm(&m, y))
    }

    /// The MAP restoration: the single most probable image, by exact minimisation.
    ///
    /// Scale-free — `t` multiplies the whole energy, so it cannot move the argmax — which is why
    /// this takes no temperature and why MAP has no Nishimori point to sit at.
    ///
    /// # Errors
    ///
    /// As [`Model::posterior`], plus [`Error::Width`].
    ///
    /// # Panics
    ///
    /// Never: `ground_state` fills its `ground_state` field whenever it returns `Ok`.
    pub fn restore_map(&self, y: &[i8]) -> Result<Vec<i8>, Error> {
        let g = self.posterior(y, NISHIMORI)?;
        let e = Elimination::default().ground_state(&g).map_err(Error::Width)?;
        Ok(e.ground_state.expect("min-sum was run"))
    }

    /// Geman & Geman's own route: restore by sampling the posterior with Gibbs sweeps.
    ///
    /// `burn` sweeps are discarded, then `sweeps` sweeps are averaged into a per-pixel posterior
    /// mean whose sign is the MPM estimate. Beta is 1: the temperature lives in `t`.
    ///
    /// # Errors
    ///
    /// As [`Model::posterior`]; also [`Error::Scale`] when `sweeps` is zero, since averaging
    /// nothing has no sign.
    pub fn restore_gibbs(
        &self,
        y: &[i8],
        t: f64,
        burn: usize,
        sweeps: usize,
        seed: u64,
    ) -> Result<Restored, Error> {
        let g = self.posterior(y, t)?;
        if sweeps == 0 {
            return Err(Error::Scale(0.0));
        }
        let mut smp = Sampler::new(&g, 1.0, seed);
        smp.sweeps(burn, None);
        let mut acc = vec![0.0f64; g.n];
        for _ in 0..sweeps {
            smp.sweep(None);
            for (a, &s) in acc.iter_mut().zip(&smp.s) {
                *a += f64::from(s);
            }
        }
        let mean: Vec<f64> = acc.iter().map(|a| a / sweeps as f64).collect();
        let image = mean
            .iter()
            .zip(y)
            .map(|(&m, &yi)| {
                if m > 0.0 {
                    1
                } else if m < 0.0 {
                    -1
                } else {
                    yi
                }
            })
            .collect();
        Ok(Restored { mean, image, sweeps })
    }

    /// Exact expected pixel error of the MPM estimate at scale `t`, over the whole joint.
    ///
    /// Every image against every observation, weighted by `P(x) P(y|x)`. No sampling: the answer is
    /// a deterministic function of `(w, h, k, p, t)`, and it is minimised at `t = 1`.
    ///
    /// # Errors
    ///
    /// [`Error::TooBig`] past [`MAX_JOINT_PIXELS`], [`Error::Scale`] on a bad scale.
    pub fn expected_error(&self, t: f64) -> Result<f64, Error> {
        self.joint_error(t, Rule::Mpm)
    }

    /// Exact expected pixel error of the MAP estimate, over the same joint.
    ///
    /// # Errors
    ///
    /// As [`Model::expected_error`].
    pub fn expected_error_map(&self) -> Result<f64, Error> {
        self.joint_error(NISHIMORI, Rule::Map)
    }

    /// The Bayes risk: the smallest expected pixel error ANY estimator can reach.
    ///
    /// `sum_y P(y) sum_i min(mu_i, 1 - mu_i) / n` over the true posterior — no estimator appears in
    /// it. [`Model::expected_error`] at [`NISHIMORI`] must equal this, and can never go below it.
    ///
    /// # Errors
    ///
    /// As [`Model::expected_error`].
    pub fn bayes_risk(&self) -> Result<f64, Error> {
        let n = self.enumerable()?;
        let m = 1usize << n;
        let lik = self.likelihood_table();
        let pri = exact_boltzmann(&self.prior(), 1.0);
        let mut g = self.posterior(&vec![1i8; n], NISHIMORI)?;
        let hc = self.channel_field();
        let mut risk = 0.0;
        for ym in 0..m {
            set_fields(&mut g, ym, hc);
            let mu = marginals_by_enumeration(&g);
            // P(y), marginalised over every image that could have produced it.
            let py: f64 = (0..m).map(|xm| pri[xm] * lik[(xm ^ ym).count_ones() as usize]).sum();
            risk += py * mu.iter().map(|&u| u.min(1.0 - u)).sum::<f64>();
        }
        Ok(risk / n as f64)
    }

    /// Shared skeleton for the two exact expected-error routes.
    fn joint_error(&self, t: f64, rule: Rule) -> Result<f64, Error> {
        let n = self.enumerable()?;
        check_scale(t)?;
        let m = 1usize << n;
        let lik = self.likelihood_table();
        let pri = exact_boltzmann(&self.prior(), 1.0);
        let mut g = self.posterior(&vec![1i8; n], t)?;
        let hc = t * self.channel_field();
        let mut total = 0.0;
        for ym in 0..m {
            set_fields(&mut g, ym, hc);
            let est = match rule {
                Rule::Mpm => {
                    let mu = marginals_by_enumeration(&g);
                    let mut e = 0usize;
                    for (i, &u) in mu.iter().enumerate() {
                        // Ties go to the data, as in `mpm`.
                        if u > 0.5 || (u == 0.5 && (ym >> i) & 1 == 1) {
                            e |= 1 << i;
                        }
                    }
                    e
                }
                // Lowest mask among tied maxima; a tie needs two images of exactly equal posterior
                // weight, which a field generically forbids.
                Rule::Map => {
                    let post = exact_boltzmann(&g, 1.0);
                    let mut best = 0usize;
                    for xm in 1..m {
                        if post[xm] > post[best] {
                            best = xm;
                        }
                    }
                    best
                }
            };
            for xm in 0..m {
                let wrong = (est ^ xm).count_ones();
                if wrong > 0 {
                    total += pri[xm] * lik[(xm ^ ym).count_ones() as usize] * f64::from(wrong);
                }
            }
        }
        Ok(total / n as f64)
    }

    /// `P(y|x)` by Hamming distance: `p^d (1-p)^(n-d)`.
    fn likelihood_table(&self) -> Vec<f64> {
        let n = self.pixels();
        (0..=n).map(|d| self.p.powi(d as i32) * (1.0 - self.p).powi((n - d) as i32)).collect()
    }

    /// Pixel count, if the joint fits.
    fn enumerable(&self) -> Result<usize, Error> {
        let n = self.pixels();
        if n > MAX_JOINT_PIXELS {
            return Err(Error::TooBig { pixels: n, max: MAX_JOINT_PIXELS });
        }
        Ok(n)
    }

    /// Length and alphabet of an image.
    fn check(&self, x: &[i8]) -> Result<(), Error> {
        if x.len() != self.pixels() {
            return Err(Error::Size { got: x.len(), want: self.pixels() });
        }
        match x.iter().position(|&v| v != 1 && v != -1) {
            Some(at) => Err(Error::NotASpin { at, value: x[at] }),
            None => Ok(()),
        }
    }
}

/// What a posterior-sampling restoration produced.
#[derive(Clone, Debug)]
pub struct Restored {
    /// Per-pixel posterior mean `E[x_i | y]`, in `[-1, 1]`.
    pub mean: Vec<f64>,
    /// The MPM estimate: the sign of each mean, ties taken from the observation.
    pub image: Vec<i8>,
    /// Sweeps averaged, burn-in excluded.
    pub sweeps: usize,
}

/// The MPM estimate from posterior marginals `P(x_i = +1 | y)`: the sign of each marginal.
///
/// A marginal of exactly `0.5` decides nothing, so it keeps the observed pixel.
///
/// # Panics
///
/// If the slices differ in length — the marginals and the image must describe the same pixels.
#[must_use]
pub fn mpm(marginals: &[f64], y: &[i8]) -> Vec<i8> {
    assert_eq!(marginals.len(), y.len(), "one marginal per pixel");
    marginals
        .iter()
        .zip(y)
        .map(|(&u, &yi)| {
            if u > 0.5 {
                1
            } else if u < 0.5 {
                -1
            } else {
                yi
            }
        })
        .collect()
}

/// Fraction of pixels on which two images disagree.
///
/// # Panics
///
/// If the images differ in length: comparing a prefix would understate the error, which is the
/// direction that turns a red test green.
#[must_use]
pub fn pixel_error(a: &[i8], b: &[i8]) -> f64 {
    assert_eq!(a.len(), b.len(), "two images over the same pixels");
    a.iter().zip(b).filter(|(x, y)| x != y).count() as f64 / a.len() as f64
}

/// Rewrite a posterior's fields for observation `ym` (bit set = `+1`), reusing its CSR.
fn set_fields(g: &mut Graph, ym: usize, hc: f64) {
    for (i, gi) in g.h.iter_mut().enumerate() {
        *gi = if (ym >> i) & 1 == 1 { hc } else { -hc };
    }
}

/// Marginals `P(x_i = +1)` by enumerating all `2^n` states — the small-model twin of
/// [`Elimination::marginals`].
fn marginals_by_enumeration(g: &Graph) -> Vec<f64> {
    let p = exact_boltzmann(g, 1.0);
    let mut mu = vec![0.0f64; g.n];
    for (mask, &w) in p.iter().enumerate() {
        for (i, u) in mu.iter_mut().enumerate() {
            if (mask >> i) & 1 == 1 {
                *u += w;
            }
        }
    }
    mu
}

/// A model scale must be finite and positive.
fn check_scale(t: f64) -> Result<(), Error> {
    if t.is_finite() && t > 0.0 { Ok(()) } else { Err(Error::Scale(t)) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ising::tv;

    /// The two models every quantitative test below runs on, so one lucky instance cannot carry
    /// the claim: a weak prior with light noise, and a strong prior with heavy noise.
    const CASES: [(usize, usize, f64, f64); 2] = [(3, 3, 0.4, 0.15), (3, 3, 0.6, 0.2)];

    fn image(mask: usize, n: usize) -> Vec<i8> {
        (0..n).map(|i| if (mask >> i) & 1 == 1 { 1 } else { -1 }).collect()
    }

    /// `hc` is the channel, not a knob: `exp(+-hc) / (2 cosh hc)` must BE `(1-p, p)`.
    ///
    /// The closed form is exact — `exp(hc) = sqrt((1-p)/p)` and `2 cosh hc = 1/sqrt(p(1-p))` — so
    /// this is checked at 1e-15 rather than with a tolerance that would hide a factor of two.
    #[test]
    fn the_channel_field_is_the_exact_log_likelihood_ratio() {
        for p in [0.01, 0.05, 0.1, 0.15, 0.25, 0.4, 0.49] {
            let m = Model::new(2, 2, 0.3, p).unwrap();
            let hc = m.channel_field();
            let z = 2.0 * hc.cosh();
            assert!((hc.exp() / z - (1.0 - p)).abs() < 1e-15, "agreement weight at p={p}");
            assert!(((-hc).exp() / z - p).abs() < 1e-15, "disagreement weight at p={p}");
        }
    }

    /// The graph at `t = 1` IS Bayes' rule, term for term.
    ///
    /// The oracle is `P(x) P(y|x)` written out directly from the prior's Boltzmann distribution and
    /// the channel's binomial weights — no graph, no fields, no elimination. If the energy
    /// convention, the sign of the field or the factor of a half in `hc` were wrong, this is where
    /// it shows. The `t = 1.5` line is the negative control: the scale must MOVE the distribution,
    /// or the Nishimori sweep below would be measuring nothing.
    #[test]
    fn the_posterior_graph_is_bayes_rule() {
        for (w, h, k, p) in CASES {
            let m = Model::new(w, h, k, p).unwrap();
            let n = m.pixels();
            let pri = exact_boltzmann(&m.prior(), 1.0);
            for ym in [0usize, 0b101_010_011, 0b111_000_111, 511] {
                let y = image(ym, n);
                let mut want: Vec<f64> = (0..1 << n)
                    .map(|xm| {
                        let d = i32::try_from((xm ^ ym).count_ones()).unwrap();
                        pri[xm] * p.powi(d) * (1.0 - p).powi(i32::try_from(n).unwrap() - d)
                    })
                    .collect();
                let z: f64 = want.iter().sum();
                for v in &mut want {
                    *v /= z;
                }
                let got = exact_boltzmann(&m.posterior(&y, NISHIMORI).unwrap(), 1.0);
                assert!(tv(&got, &want) < 1e-12, "y={ym:b}: tv {}", tv(&got, &want));

                let off = exact_boltzmann(&m.posterior(&y, 1.5).unwrap(), 1.0);
                assert!(tv(&off, &want) > 1e-3, "t = 1.5 must not be the posterior");
            }
        }
    }

    /// Elimination and enumeration are different exact algorithms; on a model both accept they must
    /// return the same marginals.
    #[test]
    fn elimination_and_enumeration_agree_on_the_marginals() {
        let m = Model::new(3, 3, 0.4, 0.15).unwrap();
        for ym in [0usize, 0b010_101_010, 0b110_011_001] {
            let y = image(ym, m.pixels());
            for t in [0.5, NISHIMORI, 2.0] {
                let by_elim = m.posterior_marginals(&y, t).unwrap();
                let by_enum = marginals_by_enumeration(&m.posterior(&y, t).unwrap());
                let worst = by_elim
                    .iter()
                    .zip(&by_enum)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f64, f64::max);
                assert!(worst < 1e-12, "y={ym:b} t={t}: worst marginal gap {worst:e}");
            }
        }
    }

    /// **The Nishimori claim.** The exact expected pixel error is minimised at `t = 1`.
    ///
    /// Nothing here is sampled: [`Model::expected_error`] sums over every image and every
    /// observation, so each number is a deterministic function of `(w, h, k, p, t)` and the
    /// comparison is a numerical fact rather than a statistical one.
    ///
    /// Two assertions, and they are different strengths. The named temperatures either side must be
    /// STRICTLY worse, which is the falsifiable part. The 25-point sweep is only required not to go
    /// BELOW the Nishimori value — deliberately, because on a nine-pixel image the estimator is a
    /// map from 512 observations to 512 images, and a `t` close enough to 1 makes exactly the same
    /// 512 decisions and ties to the last bit. A tie is Bayes optimality holding, not failing.
    #[test]
    fn nishimori_minimises_the_exact_expected_error() {
        for (w, h, k, p) in CASES {
            let m = Model::new(w, h, k, p).unwrap();
            let best = m.expected_error(NISHIMORI).unwrap();

            for t in [0.25, 0.5, 0.75, 1.5, 2.5, 5.0] {
                let e = m.expected_error(t).unwrap();
                assert!(
                    e > best + 1e-5,
                    "{w}x{h} k={k} p={p}: t={t} gave {e:.9}, t=1 gave {best:.9}"
                );
            }

            for i in 0..25 {
                let t = 0.1 + 0.2 * f64::from(i);
                let e = m.expected_error(t).unwrap();
                assert!(e >= best - 1e-12, "t={t} beat the Nishimori point: {e:.12} < {best:.12}");
            }
        }
    }

    /// The Bayes risk is reached at `t = 1`, and it is a floor.
    ///
    /// `sum_y P(y) sum_i min(mu_i, 1-mu_i)` mentions no estimator at all, so this is an identity
    /// between two different computations rather than a comparison of two runs: the MPM estimate at
    /// the Nishimori point attains the smallest error ANY rule can, and nothing below it exists.
    #[test]
    fn the_bayes_risk_identity_holds_at_the_nishimori_point() {
        for (w, h, k, p) in CASES {
            let m = Model::new(w, h, k, p).unwrap();
            let risk = m.bayes_risk().unwrap();
            let at_one = m.expected_error(NISHIMORI).unwrap();
            assert!((risk - at_one).abs() < 1e-12, "risk {risk:.15} vs error {at_one:.15}");
            for i in 0..25 {
                let t = 0.1 + 0.2 * f64::from(i);
                assert!(m.expected_error(t).unwrap() >= risk - 1e-12, "t={t} beat the Bayes risk");
            }
            // And restoration is worth doing: fewer wrong pixels than the raw observation, which
            // is wrong on exactly a fraction p of them.
            assert!(risk < p, "{w}x{h} k={k}: risk {risk:.6} did not beat the channel {p}");
        }
    }

    /// MPM beats MAP: the most probable IMAGE is not the image of most probable pixels.
    ///
    /// This is the reason Geman & Geman's Gibbs sampler is the algorithm and energy minimisation is
    /// not. Exact on both sides — elimination for the marginals, enumeration for the mode.
    #[test]
    fn the_posterior_mode_loses_to_the_posterior_marginals() {
        for (w, h, k, p) in CASES {
            let m = Model::new(w, h, k, p).unwrap();
            let mpm_err = m.expected_error(NISHIMORI).unwrap();
            let map_err = m.expected_error_map().unwrap();
            assert!(map_err > mpm_err + 1e-5, "{w}x{h} k={k} p={p}: map {map_err} mpm {mpm_err}");
        }
    }

    /// At `t -> 0` the model forgets the prior and returns the data, so its error is exactly `p`.
    ///
    /// A closed form at the other end of the sweep, and a check that the scale reaches the limit it
    /// should: an infinite-temperature posterior has nothing to say, and every pixel keeps its
    /// observed value.
    #[test]
    fn an_infinitely_hot_model_returns_the_observation() {
        for (w, h, k, p) in CASES {
            let m = Model::new(w, h, k, p).unwrap();
            let e = m.expected_error(1e-9).unwrap();
            assert!((e - p).abs() < 1e-12, "{w}x{h} k={k}: hot error {e:.15}, channel {p}");
        }
    }

    /// The Gibbs restorer against exact elimination — a sampler is worth checking only against an
    /// oracle, and this one has one.
    ///
    /// Posterior means from 200k sweeps versus `2 mu - 1` from variable elimination on the same
    /// 4x4 posterior. A wrong field sign, a wrong beta or a missed burn-in moves these by far more
    /// than the 0.02 allowed here.
    #[test]
    fn gibbs_posterior_means_match_the_exact_marginals() {
        let m = Model::new(4, 4, 0.5, 0.2).unwrap();
        let mut rng = Pcg::new(7, 11);
        let truth: Vec<i8> = (0..m.pixels()).map(|_| rng.spin(0.5)).collect();
        let y = m.corrupt(&truth, &mut rng).unwrap();

        let exact = m.posterior_marginals(&y, NISHIMORI).unwrap();
        let run = m.restore_gibbs(&y, NISHIMORI, 2_000, 200_000, 20_260_910).unwrap();

        let mut worst = 0.0f64;
        for (i, (&mu, &mean)) in exact.iter().zip(&run.mean).enumerate() {
            let want = 2.0 * mu - 1.0;
            worst = worst.max((want - mean).abs());
            // Where the exact marginal is not near a coin flip, the sampled sign must agree: this
            // is the estimate itself, not just the mean it came from.
            if (mu - 0.5).abs() > 0.05 {
                assert_eq!(
                    run.image[i],
                    if mu > 0.5 { 1 } else { -1 },
                    "pixel {i}: exact marginal {mu:.4}, sampled mean {mean:.4}"
                );
            }
        }
        assert!(worst < 0.02, "worst posterior-mean gap {worst:.4}");
        assert_eq!(run.sweeps, 200_000);
    }

    /// The channel flips at its stated rate and emits nothing but spins.
    #[test]
    fn the_channel_flips_at_its_stated_rate() {
        let m = Model::new(20, 20, 0.4, 0.25).unwrap();
        let x = vec![1i8; m.pixels()];
        let mut rng = Pcg::new(3, 5);
        let mut flipped = 0usize;
        let trials = 200;
        for _ in 0..trials {
            let y = m.corrupt(&x, &mut rng).unwrap();
            assert!(y.iter().all(|&v| v == 1 || v == -1));
            flipped += y.iter().filter(|&&v| v == -1).count();
        }
        let n = (trials * m.pixels()) as f64;
        let rate = flipped as f64 / n;
        // Four standard deviations of a Binomial(n, p): this fails on a wrong rate, not on luck.
        let tol = 4.0 * (0.25 * 0.75 / n).sqrt();
        assert!((rate - 0.25).abs() < tol, "flip rate {rate:.4}, tolerance {tol:.4}");
    }

    /// A marginal of exactly one half decides nothing, so the observation stands.
    #[test]
    fn an_undecided_pixel_keeps_its_observation() {
        let y = [1i8, -1, 1];
        assert_eq!(mpm(&[0.5, 0.5, 0.5], &y), y);
        assert_eq!(mpm(&[0.5 + f64::EPSILON, 0.5 - f64::EPSILON, 0.5], &y), [1, -1, 1]);
        assert_eq!(pixel_error(&[1, 1, -1], &[1, -1, -1]), 1.0 / 3.0);
    }

    /// Every refusal, and each for its own reason.
    #[test]
    fn bad_models_and_bad_images_are_refused() {
        assert_eq!(Model::new(0, 3, 0.4, 0.1), Err(Error::Size { got: 0, want: 0 }));
        assert_eq!(Model::new(3, 3, 0.4, 0.5), Err(Error::Noise(0.5)));
        assert_eq!(Model::new(3, 3, 0.4, 0.0), Err(Error::Noise(0.0)));
        // NaN is not equal to itself, so this one is matched rather than compared.
        assert!(matches!(Model::new(3, 3, f64::NAN, 0.1), Err(Error::Scale(t)) if t.is_nan()));

        let m = Model::new(3, 3, 0.4, 0.15).unwrap();
        // `.err()` rather than the whole `Result`: a `Graph` is not `Debug`, and it is not this
        // module's business to make it one.
        assert_eq!(m.posterior(&[1i8; 8], NISHIMORI).err(), Some(Error::Size { got: 8, want: 9 }));
        let mut bad = [1i8; 9];
        bad[4] = 0;
        assert_eq!(m.posterior(&bad, NISHIMORI).err(), Some(Error::NotASpin { at: 4, value: 0 }));
        assert_eq!(m.posterior(&[1i8; 9], 0.0).err(), Some(Error::Scale(0.0)));
        assert_eq!(m.posterior(&[1i8; 9], -1.0).err(), Some(Error::Scale(-1.0)));
        assert_eq!(m.expected_error(f64::INFINITY), Err(Error::Scale(f64::INFINITY)));
        assert_eq!(m.restore_gibbs(&[1i8; 9], NISHIMORI, 10, 0, 1).err(), Some(Error::Scale(0.0)));

        let big = Model::new(4, 4, 0.4, 0.15).unwrap();
        assert_eq!(
            big.expected_error(NISHIMORI),
            Err(Error::TooBig { pixels: 16, max: MAX_JOINT_PIXELS })
        );
        assert!(!Error::Noise(0.5).to_string().is_empty());
    }
}
