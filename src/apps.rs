//! A task, its answer, its **certificate**, and its **bill** — the layer above the samplers.
//!
//! Everything else in this crate is addressed by its mathematics: build a [`Graph`], pick a
//! sampler, read a ledger. That is the right surface for the questions the crate is here to
//! settle, and the wrong one for anybody who has an image and a power budget. A 2026-09-20 survey
//! of what vendors ship (Ocean, Amplify, `JijModeling`, Kaiwu, THRML) found the streamlined ones
//! share one shape — the user states the TASK, the solver is one line, a classical fallback is
//! built in — and found no offering, open or closed, that returns **joules per answer alongside
//! the answer**. That is the gap this module is for.
//!
//! An entry point here returns four things together, and the last two are the point:
//!
//! 1. the answer, in the task's own units (pixels, not spins);
//! 2. how it was estimated, and whether the chain that estimated it had converged
//!    ([`crate::floors::Convergence`]);
//! 3. a bound or an exact optimum where one is available, so the answer carries its own
//!    optimality gap rather than a claim about it;
//! 4. what it cost, as a [`crate::ledger::Ledger`] priced by whatever machine the caller names —
//!    with the evidence grade of those prices attached, so a projection cannot be mistaken for a
//!    measurement.
//!
//! # The first task: restoring a binary image
//!
//! Chosen because it is the one vision problem where **both** estimators have exact oracles in
//! this crate, so the interesting claim can be checked rather than asserted.
//!
//! Write `s_i = ±1` for the true pixel and `t_i = ±1` for the observed one, with each pixel
//! independently flipped with probability `p`. Then
//!
//! ```text
//!   log P(t_i | s_i) = const + (s_i t_i / 2) · ln((1-p)/p)
//!   log P(s)         = const + J · sum_<ij> s_i s_j
//! ```
//!
//! so the negative log posterior is exactly this crate's energy with `w_ij = J` and
//! `h_i = t_i · ln((1-p)/p) / 2`, **at `β = 1`**. The temperature is not a knob here: the energy
//! already is the negative log posterior, so any other `β` samples a different problem. The one
//! free parameter is `J`, the prior's belief in smoothness.
//!
//! # Why a sampler, on a problem min-cut already solves
//!
//! The honest answer, and the reason this task earns its place. `J > 0` makes the energy
//! submodular, so [`crate::roofdual`] returns its exact minimiser by one max-flow: **the MAP is
//! not a problem a sampler should be sold for.**
//!
//! But the MAP is not the estimator this task wants. The labelling that minimises the expected
//! number of wrong PIXELS is the per-pixel posterior mode — marginal posterior mode, MPM
//! (Marroquin, Mitter and Poggio 1987) — because per-pixel error decomposes over pixels and each
//! term is minimised by its own marginal. A min-cut returns one labelling and no marginals, and
//! marginals are what a Boltzmann sampler produces natively. So the two estimators split cleanly:
//! **MAP is a flow problem, MPM is a sampling problem**, and only the second is a reason to own
//! sampling hardware.
//!
//! # A third application, and why it is not behind this door
//!
//! [`crate::logit`] is the same argument on a classifier's confidence, and it deliberately has no
//! entry point here. Its unit of work is a gradient evaluation of a non-Gaussian potential, and
//! **no price set in this crate states a cost for one** — a [`Ledger`] for it would be three
//! numbers in the wrong currency. The image task's bill could at least name the half nobody has
//! metered; that one could not name any of it, so it reports its work in gradient evaluations and
//! claims no joules. Both facts point the same way: the field prices a binary p-bit update and
//! nothing else.
//!
//! `an_mpm_estimate_beats_the_exact_map_on_pixel_error` puts that to the test with no sampler in
//! it at all: images drawn EXACTLY from the prior by [`crate::exact::Elimination::draws`],
//! corrupted, then restored by exact marginals against the exact minimiser. If MPM did not win
//! there, no sampler could rescue the argument, and this module would be pointing at the wrong
//! estimator.

use crate::exact::Elimination;
use crate::floors::Convergence;
use crate::graph::{Graph, GraphBuilder};
use crate::ledger::{Evidence, Ledger, Prices};
use crate::receipt::Receipt;

/// A binary image, row-major, `true` for a set pixel.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Image {
    /// Columns.
    pub width: usize,
    /// Rows.
    pub height: usize,
    /// `width * height` pixels, row-major.
    pub pixels: Vec<bool>,
}

impl Image {
    /// An image of `width × height`, or `None` if the pixel count disagrees with the shape.
    #[must_use]
    pub fn new(width: usize, height: usize, pixels: Vec<bool>) -> Option<Image> {
        if width == 0 || height == 0 || pixels.len() != width * height {
            return None;
        }
        Some(Image { width, height, pixels })
    }

    /// Pixels that differ between two images of the same shape, or `None` if the shapes differ.
    #[must_use]
    pub fn differences(&self, other: &Image) -> Option<usize> {
        if self.width != other.width || self.height != other.height {
            return None;
        }
        Some(
            self.pixels
                .iter()
                .zip(&other.pixels)
                .filter(|(a, b)| a != b)
                .count(),
        )
    }

    /// The image as spins: `+1` for a set pixel.
    #[must_use]
    pub fn spins(&self) -> Vec<i8> {
        self.pixels.iter().map(|&b| if b { 1i8 } else { -1 }).collect()
    }

    /// An image from spins, or `None` if the count disagrees with the shape.
    #[must_use]
    pub fn from_spins(width: usize, height: usize, s: &[i8]) -> Option<Image> {
        Image::new(width, height, s.iter().map(|&v| v > 0).collect())
    }
}

/// Which estimator to return, and what it is allowed to spend.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Estimator {
    /// The single most probable image. Exact here, by one max-flow, because `J > 0` makes the
    /// energy submodular — so this arm needs no sampler and gets no convergence question.
    Map,
    /// The same estimate, computed by the **p-bit fabric this crate emits for an FPGA** — Q.8
    /// weights, a sigmoid ROM, one `xorshift32` per node, two colour classes a sweep. The
    /// emulator is cycle-exact with that Verilog, and `the_restored_image_comes_out_of_emitted_
    /// hardware` runs the emitted RTL to prove it.
    ///
    /// This is the whole path in one call: an image becomes a posterior, the posterior becomes a
    /// netlist, the netlist restores the image. A 2026-09-20 survey of this field did not locate
    /// an open path from a problem-level model to emitted p-bit RTL anywhere.
    Fabric {
        /// Independent chains. **On this fabric a chain's randomness is a reset constant of the
        /// netlist**, so `R` chains are `R` netlists — a convergence diagnostic costs `R`
        /// implementations here, and the ledger charges it.
        chains: usize,
        /// Sweeps discarded per chain.
        burn_in: usize,
        /// Sweeps recorded per chain.
        sweeps: usize,
    },
    /// The per-pixel posterior mode, estimated from chains. The estimator that minimises expected
    /// pixel error, and the one a min-cut cannot produce.
    Mpm {
        /// Independent chains from dispersed starts. **Two or more**, or convergence cannot be
        /// diagnosed at all and the answer is refused rather than graded on a diagnostic that
        /// cannot fail (see [`crate::floors::Convergence`]).
        chains: usize,
        /// Sweeps discarded per chain before anything is recorded.
        burn_in: usize,
        /// Sweeps recorded per chain.
        sweeps: usize,
    },
}

/// Why a restoration could not be produced. Each variant is a refusal, not a degraded answer.
#[derive(Clone, Debug, PartialEq)]
pub enum Refused {
    /// The observation's pixel count disagrees with its shape.
    Shape,
    /// `p` is not a probability strictly inside `(0, 0.5)`. At `p = 0.5` the observation carries
    /// no information and at `p = 0` it carries all of it; both make the field infinite.
    FlipProbability(f64),
    /// The smoothness is not positive and finite. Negative `J` is not merely unsupported: it makes
    /// the energy supermodular, so the MAP arm's max-flow stops being exact.
    Smoothness(f64),
    /// Fewer than two chains were asked for, so no between-chain diagnostic exists.
    TooFewChains(usize),
    /// The chains ran but did not converge, with the diagnostic's own words.
    NotConverged(String),
    /// The posterior does not fit the fabric's registers, in the fabric's own words. A coupling
    /// outside signed Q.8 in twelve bits is refused rather than clamped: a clamped coupling
    /// samples a different problem and reports nothing about this one.
    DoesNotFit(String),
}

impl core::fmt::Display for Refused {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Refused::Shape => write!(f, "the pixel count disagrees with the width and height"),
            Refused::FlipProbability(p) => {
                write!(f, "a flip probability of {p} is not strictly inside (0, 0.5)")
            }
            Refused::Smoothness(j) => write!(f, "a smoothness of {j} is not positive and finite"),
            Refused::TooFewChains(n) => write!(
                f,
                "{n} chain(s) cannot be diagnosed for convergence; a single chain's own trace \
                 cannot report its own failure"
            ),
            Refused::NotConverged(why) => write!(f, "{why}"),
            Refused::DoesNotFit(why) => write!(f, "this posterior does not fit the fabric: {why}"),
        }
    }
}

impl core::error::Error for Refused {}

/// An answer, its certificate and its bill.
pub struct Restoration {
    /// The restored image.
    pub image: Image,
    /// Per-pixel posterior probability that the pixel is set. Exactly `0.0`/`1.0` on the
    /// [`Estimator::Map`] arm, which reports a labelling and has no marginals to report.
    pub posterior: Vec<f64>,
    /// Energy of the returned image under the posterior's own energy function.
    pub energy: f64,
    /// The exact minimum energy, when the max-flow pinned every pixel. `Some` means the gap below
    /// is exact rather than bounded.
    pub optimum: Option<f64>,
    /// How convergence was established. `None` on the MAP arm: a max-flow does not converge, it
    /// finishes.
    pub convergence: Option<Convergence>,
    /// The device operations this cost.
    pub cost: Ledger,
    /// The answer, its model digest and its bound, in a form that can be re-checked.
    pub receipt: Receipt,
}

impl Restoration {
    /// How far the returned image is from the most probable one, when that is known exactly.
    ///
    /// `Some(0.0)` is a proof of optimality; `None` means no oracle pinned the optimum, not that
    /// the answer is optimal.
    #[must_use]
    pub fn gap(&self) -> Option<f64> {
        self.optimum.map(|o| self.energy - o)
    }

    /// What this cost on `prices`, in joules, or `None` if those prices do not state a cost for
    /// every operation this run performed.
    ///
    /// The `None` is the point: a machine that metered sampling and never metered readback cannot
    /// price a run that read its answer out, and saying so is better than charging zero for the
    /// half nobody measured.
    ///
    /// **As of 2026-09-20 this returns `None` for every machine in [`crate::ledger::CATALOGUE`]**,
    /// and not by accident. Restoring an image writes one field per pixel — the observation IS the
    /// field — so every frame is a full reprogram, and no price set in this crate states
    /// `e_write`, because the write on our own board has not been metered. [`Restoration::bill`]
    /// says which half is priced instead of collapsing the question to a `None`.
    #[must_use]
    pub fn joules(&self, prices: &Prices) -> Option<f64> {
        self.cost.joules(prices)
    }

    /// The bill split into what `prices` can price and what it cannot.
    ///
    /// An application needs this where a bare refusal is useless: "the sampling and the readback
    /// cost X, and these writes are unpriced" is actionable, and "None" is not.
    #[must_use]
    pub fn bill(&self, prices: &Prices) -> Bill {
        // A price of NaN is "unstated", and an operation nobody performed needs no price -- the
        // same rule `Ledger::joules` applies, kept here so the two cannot drift apart.
        let split = |count: u64, price: f64| -> (f64, u64) {
            if count == 0 {
                (0.0, 0)
            } else if price.is_finite() {
                (count as f64 * price, 0)
            } else {
                (0.0, count)
            }
        };
        let (js, us) = split(self.cost.samples, prices.e_sample);
        let (jr, ur) = split(self.cost.reads, prices.e_read);
        let (jw, uw) = split(self.cost.writes, prices.e_write);
        Bill {
            priced: js + jr + jw,
            unpriced: Ledger { samples: us, reads: ur, writes: uw },
            evidence: self.evidence(prices),
        }
    }

    /// Joules per pixel of the restored image, on `prices`.
    #[must_use]
    pub fn joules_per_pixel(&self, prices: &Prices) -> Option<f64> {
        let n = self.image.pixels.len();
        if n == 0 {
            return None;
        }
        self.joules(prices).map(|j| j / n as f64)
    }

    /// The grade of the weakest link in a joules figure computed on `prices`: no better than the
    /// evidence behind the prices, and no better than a chain whose convergence was merely assumed.
    #[must_use]
    pub fn evidence(&self, prices: &Prices) -> Evidence {
        let by_prices = prices.evidence;
        match &self.convergence {
            // A run whose convergence rests on one chain's own trace is not a measurement of the
            // posterior, whatever the wattmeter said about the machine.
            Some(Convergence::SingleChain) => crate::ledger::weaker(by_prices, Evidence::Projected),
            _ => by_prices,
        }
    }
}

/// A bill: the part a price set could price, and the operations it could not.
#[derive(Clone, Copy, Debug)]
pub struct Bill {
    /// Joules for the operations `prices` states a cost for.
    pub priced: f64,
    /// The operations it does not state a cost for. **A non-zero field here means `priced` is an
    /// UNDERCOUNT, not a total**, and the whole point of carrying it beside the number.
    pub unpriced: Ledger,
    /// The grade of `priced`, no better than the prices behind it or the convergence in front.
    pub evidence: Evidence,
}

impl Bill {
    /// Whether every operation performed was priced, so `priced` is the whole bill.
    #[must_use]
    pub fn complete(&self) -> bool {
        self.unpriced.samples == 0 && self.unpriced.reads == 0 && self.unpriced.writes == 0
    }
}

/// The posterior of a binary image under a Bernoulli channel and an Ising prior, as a model this
/// crate's samplers take. Its `β` is `1`: the energy IS the negative log posterior.
///
/// # Errors
///
/// [`Refused::Shape`], [`Refused::FlipProbability`] or [`Refused::Smoothness`] — each a statement
/// about the arguments, checked before anything is built.
pub fn posterior(observed: &Image, flip_probability: f64, smoothness: f64) -> Result<Graph, Refused> {
    let (w, h) = (observed.width, observed.height);
    if w == 0 || h == 0 || observed.pixels.len() != w * h {
        return Err(Refused::Shape);
    }
    if !(flip_probability > 0.0 && flip_probability < 0.5) {
        return Err(Refused::FlipProbability(flip_probability));
    }
    if !(smoothness > 0.0) || !smoothness.is_finite() {
        return Err(Refused::Smoothness(smoothness));
    }
    // The channel's log-odds. This is the whole of the data term: no separate weight to tune,
    // because the observation's strength is a property of the noise, not a taste.
    let lambda = ((1.0 - flip_probability) / flip_probability).ln();
    let mut b = GraphBuilder::new(w * h);
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            if x + 1 < w {
                b.couple(i, i + 1, smoothness);
            }
            if y + 1 < h {
                b.couple(i, i + w, smoothness);
            }
            let t = if observed.pixels[i] { 1.0 } else { -1.0 };
            b.bias(i, t * lambda / 2.0);
        }
    }
    Ok(b.build())
}

/// The per-pixel posterior mode from accumulated up-counts, with the convergence the chains
/// earned — the tail both chain-based estimators share.
///
/// Extracted after `Estimator::Fabric` duplicated it and the mutation suite's precheck refused two
/// rows for naming a line that now existed twice. An ambiguous mutation target is not a stale one,
/// but it measures just as little: the harness cannot say which copy it broke.
fn mode_from_counts(
    sums: &[f64],
    traces: &[Vec<f64>],
    observed: &Image,
    g: &Graph,
    cost: Ledger,
    method: &str,
    seed: u64,
) -> Result<Restoration, Refused> {
    let convergence = Convergence::from_chains(traces);
    if let Some(why) = convergence.refusal() {
        return Err(Refused::NotConverged(why));
    }
    // DERIVED from the traces, not passed in: a caller and a callee that each believe a different
    // number of draws were averaged is a disagreement nothing would report, and the traces are the
    // record of what actually happened.
    let draws = (traces.len() * traces.first().map_or(0, Vec::len)) as f64;
    let posterior: Vec<f64> = sums.iter().map(|&v| v / draws).collect();
    // The per-pixel mode. A marginal of exactly 0.5 is a tie the data cannot break, and it is
    // resolved towards the observation rather than by the sign of a floating-point comparison.
    let state: Vec<i8> = posterior
        .iter()
        .zip(&observed.pixels)
        .map(|(&m, &t)| if m > 0.5 || (m == 0.5 && t) { 1i8 } else { -1i8 })
        .collect();
    let image = Image::from_spins(observed.width, observed.height, &state)
        .expect("the model has one node per pixel");
    let energy = g.energy(&state);
    let receipt = Receipt::of(g, state, method, seed, cost);
    Ok(Restoration {
        image,
        posterior,
        energy,
        optimum: None,
        convergence: Some(convergence),
        cost,
        receipt,
    })
}

/// Restore `observed`, and return the answer with its certificate and its bill.
///
/// ```
/// use ferrotherm::apps::{restore, Estimator, Image};
/// use ferrotherm::ledger::KV260_AXI_METERED;
///
/// // A 4x4 block of set pixels with one speck of noise in it.
/// let mut pixels = vec![true; 16];
/// pixels[5] = false;
/// let observed = Image::new(4, 4, pixels).expect("4x4");
///
/// let r = restore(&observed, 0.15, 0.8, Estimator::Map, 7).expect("a well-posed restoration");
/// assert!(r.image.pixels.iter().all(|&p| p), "the speck is smoothed away");
/// assert_eq!(r.gap(), Some(0.0), "the max-flow pinned every pixel, so this IS the optimum");
///
/// // And what it cost, on the machine whose read energy was measured rather than modelled --
/// // split, because restoring an image writes one field per pixel and NOBODY has metered a write.
/// let bill = r.bill(&KV260_AXI_METERED);
/// assert!(bill.priced > 0.0, "the sampling and the readback are metered");
/// assert!(!bill.complete(), "the per-frame reprogram is not");
/// assert_eq!(bill.unpriced.writes, 16);
/// ```
///
/// # Errors
///
/// A [`Refused`] naming the reason. The `Mpm` arm refuses a run whose chains did not converge
/// rather than returning the estimate it happens to have reached.
///
/// # Panics
///
/// Never, for a model this function built: [`posterior`] gives the graph one node per pixel, so
/// the shape checks inside cannot fail. The `expect`s are there because that invariant is worth
/// stating where it is relied on rather than silently returning a wrong-shaped image.
pub fn restore(
    observed: &Image,
    flip_probability: f64,
    smoothness: f64,
    estimator: Estimator,
    seed: u64,
) -> Result<Restoration, Refused> {
    let g = posterior(observed, flip_probability, smoothness)?;
    let n = g.n;
    // ONE GUARD FOR BOTH CHAIN-BASED ESTIMATORS, and before any chain is run. It sat inside each
    // arm until the two arms made it two identical lines, which the mutation suite's precheck
    // refused: an ambiguous target measures as little as a stale one.
    if let Estimator::Mpm { chains, .. } | Estimator::Fabric { chains, .. } = estimator
        && chains < 2
    {
        return Err(Refused::TooFewChains(chains));
    }
    // Loading the model is a write per node, charged here because it is charged everywhere else:
    // on this hardware class a write is the most expensive line in the ledger, and an application
    // layer that quietly dropped it would report the cheap half of its own bill.
    let mut cost = Ledger { samples: 0, reads: 0, writes: n as u64 };

    match estimator {
        Estimator::Map => {
            let rd = crate::roofdual::roof_dual(&g);
            // Submodular by construction (`smoothness > 0`), so the relaxation is tight and every
            // pixel is pinned. If a future change to the model broke that, this falls back to the
            // labelling roof duality could pin and reports NO optimum rather than a false one.
            let (state, optimum) = match rd.ground_state() {
                Some(s) => {
                    let e = g.energy(&s);
                    (s, Some(e))
                }
                None => {
                    // An unpinned pixel takes the observation, which is the honest fallback: no
                    // optimum is reported, so nothing downstream can mistake this for a minimiser.
                    let s: Vec<i8> = rd
                        .labels
                        .iter()
                        .zip(&observed.pixels)
                        .map(|(&l, &t)| l.unwrap_or(if t { 1 } else { -1 }))
                        .collect();
                    (s, None)
                }
            };
            // The answer leaves the device: one read per pixel, exactly as any other backend is
            // charged for carrying a state to the host.
            cost.reads += n as u64;
            let image = Image::from_spins(observed.width, observed.height, &state)
                .expect("the model has one node per pixel");
            let posterior = state.iter().map(|&s| f64::from(u8::from(s > 0))).collect();
            let energy = g.energy(&state);
            let receipt = Receipt::of(&g, state, "apps::restore/map(roof-dual)", seed, cost)
                .with_bound(rd.bound);
            Ok(Restoration {
                image,
                posterior,
                energy,
                optimum,
                convergence: None,
                cost,
                receipt,
            })
        }
        Estimator::Fabric { chains, burn_in, sweeps } => {
            let mut sums = vec![0.0f64; n];
            let mut traces: Vec<Vec<f64>> = Vec::with_capacity(chains);
            for c in 0..chains {
                // A NEW SEED IS A NEW NETLIST. The generators are reset constants of the emitted
                // Verilog, so independent chains are independent implementations -- charged here
                // as the load each one is, which is what makes the diagnostic's cost visible
                // rather than free. `cost.writes` already holds the first load.
                if c > 0 {
                    cost.writes += n as u64;
                }
                let mut fab = crate::writable::WritableFabric::new(
                    &g,
                    1.0, // beta is 1: the energy already IS the negative log posterior
                    seed ^ (c as u64).wrapping_mul(0x9E37_79B9),
                )
                .map_err(|e| Refused::DoesNotFit(e.to_string()))?;
                for _ in 0..burn_in {
                    fab.core.sweep();
                }
                cost.samples += (burn_in as u64) * (n as u64);
                let mut trace = Vec::with_capacity(sweeps);
                for _ in 0..sweeps {
                    fab.core.sweep();
                    cost.samples += n as u64;
                    // Every recorded draw leaves the fabric: n reads, at the site that performs them.
                    cost.reads += n as u64;
                    let mut up = 0.0;
                    for (acc, &b) in sums.iter_mut().zip(&fab.core.s) {
                        if b {
                            *acc += 1.0;
                            up += 1.0;
                        }
                    }
                    trace.push(2.0 * up / n as f64 - 1.0);
                }
                traces.push(trace);
            }
            let method = format!("apps::restore/fabric({chains} netlists, {burn_in}+{sweeps} sweeps)");
            mode_from_counts(&sums, &traces, observed, &g, cost, &method, seed)
        }
        Estimator::Mpm { chains, burn_in, sweeps } => {
            let mut sums = vec![0.0f64; n];
            let mut traces: Vec<Vec<f64>> = Vec::with_capacity(chains);
            for c in 0..chains {
                // beta = 1: the energy already IS the negative log posterior.
                let mut smp = crate::gibbs::Sampler::new(&g, 1.0, seed ^ (c as u64).wrapping_mul(0x9E37_79B9));
                // Dispersed starts, so the between-chain diagnostic has something to see. Half the
                // chains start at the observation and half at its negation: if the posterior has
                // two basins, chains started inside one of them will disagree and R-hat will say so.
                let start = if c % 2 == 0 { 1i8 } else { -1i8 };
                for (i, t) in observed.pixels.iter().enumerate() {
                    smp.s[i] = if *t { start } else { -start };
                }
                smp.sweeps(burn_in, Some(&mut cost));
                let mut trace = Vec::with_capacity(sweeps);
                for _ in 0..sweeps {
                    smp.sweep(Some(&mut cost));
                    // Every recorded draw is carried to the host: n reads, charged at the site
                    // that performs them.
                    let s = smp.read_all(Some(&mut cost));
                    let mut up = 0.0;
                    for (acc, &v) in sums.iter_mut().zip(&s) {
                        if v > 0 {
                            *acc += 1.0;
                            up += 1.0;
                        }
                    }
                    // Magnetisation is the scalar the chains are compared on: a global summary,
                    // so a pair of chains sitting in different basins cannot agree on it.
                    trace.push(2.0 * up / n as f64 - 1.0);
                }
                traces.push(trace);
            }
            let method = format!("apps::restore/mpm({chains} chains, {burn_in}+{sweeps} sweeps)");
            mode_from_counts(&sums, &traces, observed, &g, cost, &method, seed)
        }
    }
}

/// Which route to take to a Gaussian's marginals.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Route {
    /// Invert the precision matrix. `O(n³)` and exact: the oracle, not a method.
    Exact,
    /// Gaussian belief propagation. Local, sparse, and what the robotics literature runs — with
    /// **the wrong variances on any graph with a loop**, by a margin iteration cannot remove. See
    /// [`crate::gbp`].
    MessagePassing {
        /// Sweeps before giving up.
        max_iters: usize,
        /// Largest message change that counts as settled.
        tol: f64,
        /// Mixing with the previous message, in `[0, 1)`. Changes what converges, not where to.
        damping: f64,
    },
    /// Equilibrate an Ornstein-Uhlenbeck network whose stationary covariance IS `Λ⁻¹`. Unbiased in
    /// both moments, so its only error is a standard error.
    Sampling {
        /// Stride between recorded samples, in units of time. A few relaxation times gives
        /// near-independent draws.
        stride_h: f64,
        /// Strides discarded before recording.
        burn: usize,
        /// Samples recorded.
        samples: usize,
    },
}

/// A Gaussian's per-node marginals, by whichever route was asked for, and what it cost.
pub struct Marginals {
    /// Posterior mean per node.
    pub mean: Vec<f64>,
    /// Posterior marginal variance per node.
    pub variance: Vec<f64>,
    /// The route taken.
    pub route: Route,
    /// Message-passing sweeps, or recorded samples. `0` for the exact route.
    pub iterations: usize,
    /// The device operations this cost.
    ///
    /// **On the sampling route these are CONTINUOUS-node updates.** Every price in
    /// [`crate::ledger::CATALOGUE`] is for a binary p-bit update, and no machine this crate knows
    /// of has metered a continuous one — so pricing this ledger against them is a claim the caller
    /// is making and not one this crate makes. It is carried rather than withheld because the
    /// operation COUNT is a fact about the run either way.
    pub cost: Ledger,
}

/// A Gaussian's marginals: the mean every route gets right, and the variance only two of them do.
///
/// ```
/// use ferrotherm::apps::{marginals, Route};
/// use ferrotherm::gbp::grid;
///
/// let model = grid(5, 5, 1.0, -0.22);          // a 5x5 factor graph: loops everywhere
/// let exact = marginals(&model, Route::Exact).expect("positive definite");
/// let bp = marginals(&model, Route::MessagePassing { max_iters: 5_000, tol: 1e-13, damping: 0.0 })
///     .expect("this grid converges");
///
/// // Message passing has the mean exactly...
/// for i in 0..model.n {
///     assert!((bp.mean[i] - exact.mean[i]).abs() < 1e-9);
/// }
/// // ...and is overconfident about it on every single node.
/// assert!((0..model.n).all(|i| bp.variance[i] < exact.variance[i]));
/// ```
///
/// # Errors
///
/// A message naming what was ill-formed, singular, or did not converge.
pub fn marginals(model: &crate::gbp::Info, route: Route) -> Result<Marginals, String> {
    model.check().map_err(|e| e.to_string())?;
    let n = model.n;
    match route {
        Route::Exact => {
            let (mean, variance) = model.exact()?;
            Ok(Marginals { mean, variance, route, iterations: 0, cost: Ledger::default() })
        }
        Route::MessagePassing { max_iters, tol, damping } => {
            let b = model
                .belief_propagation(max_iters, tol, damping)
                .map_err(|e| e.to_string())?;
            // A sweep touches every node and every directed message; the readback is the beliefs.
            let cost = Ledger {
                samples: (b.iterations as u64) * (n as u64 + 2 * model.edges.len() as u64),
                reads: n as u64,
                writes: n as u64,
            };
            Ok(Marginals {
                mean: b.mean,
                variance: b.variance,
                route,
                iterations: b.iterations,
                cost,
            })
        }
        Route::Sampling { stride_h, burn, samples } => {
            let spd = model.to_spd().map_err(|e| e.to_string())?;
            let r = crate::tla::solve_spd_exact_ou(&spd, 1.0, stride_h, burn, samples, 0xC0FF_EE01);
            let variance = (0..n).map(|i| r.a_inv[i * n + i]).collect();
            let cost = Ledger {
                samples: r.steps * n as u64,
                reads: (samples as u64) * (n as u64),
                writes: n as u64,
            };
            Ok(Marginals { mean: r.x, variance, route, iterations: samples, cost })
        }
    }
}

/// The exact per-pixel posterior mode, by elimination rather than by sampling — the oracle the
/// [`Estimator::Mpm`] arm is held to.
///
/// # Errors
///
/// A [`Refused`] for an ill-posed model, or a message from [`Elimination`] when the grid is too
/// wide to eliminate. A grid of width `w` has treewidth `w`, so this is exact only for narrow
/// images, which is exactly why the sampler exists.
pub fn exact_mpm(
    observed: &Image,
    flip_probability: f64,
    smoothness: f64,
) -> Result<(Image, Vec<f64>), String> {
    let g = posterior(observed, flip_probability, smoothness).map_err(|e| e.to_string())?;
    let elim = Elimination::default();
    // `marginals` returns the probability of `+1` per node, at this beta.
    let m = elim.marginals(&g, 1.0).map_err(|e| e.to_string())?;
    let state: Vec<i8> = m
        .iter()
        .zip(&observed.pixels)
        .map(|(&p, &t)| if p > 0.5 || (p == 0.5 && t) { 1i8 } else { -1i8 })
        .collect();
    let image = Image::from_spins(observed.width, observed.height, &state)
        .ok_or("the model has one node per pixel")?;
    Ok((image, m))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ledger::{KV260_AXI_METERED, KV260_MEASURED, Z1_SPICE};

    /// A truth image drawn EXACTLY from the prior, and its corruption. Drawing the truth from the
    /// model rather than inventing it is what makes the MPM claim below a claim about estimators
    /// and not about my taste in test images.
    fn draw_and_corrupt(
        w: usize,
        h: usize,
        j: f64,
        p: f64,
        rng: &mut crate::rng::Pcg,
    ) -> (Image, Image) {
        let mut b = GraphBuilder::new(w * h);
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                if x + 1 < w {
                    b.couple(i, i + 1, j);
                }
                if y + 1 < h {
                    b.couple(i, i + w, j);
                }
            }
        }
        let prior = b.build();
        let draws = Elimination::default()
            .draws(&prior, 1.0)
            .expect("a narrow grid eliminates");
        let truth = draws.draw(rng);
        let mut noisy = truth.clone();
        for v in &mut noisy {
            // 53-bit uniform from the generator's own u64, so the noise is reproducible from
            // the seed and does not depend on a float helper this generator does not have.
            let u = (rng.next_u64() >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0); // 2^-53
            if u < p {
                *v = -*v;
            }
        }
        (
            Image::from_spins(w, h, &truth).expect("shape"),
            Image::from_spins(w, h, &noisy).expect("shape"),
        )
    }

    /// **THE MODEL IS THE DERIVATION, CHECKED AGAINST IT.** The field is not a tunable data weight;
    /// it is `ln((1-p)/p) / 2`, and everything downstream is a posterior only if that holds.
    ///
    /// This test exists because a mutant that replaced the whole log-odds with the constant `1.0`
    /// SURVIVED every other test in this module. At the noise level they run at, `p = 0.25`, the
    /// log-odds is `ln 3 = 1.0986` — within ten percent of the constant that replaced it. The
    /// estimator comparison was blind not because it was weak but because its parameter happened
    /// to sit where the right answer and the wrong one nearly coincide.
    #[test]
    fn the_field_is_the_channels_log_odds_and_not_a_tunable_weight() {
        let observed = Image::new(2, 2, vec![true, false, false, true]).expect("2x2");
        for p in [0.01, 0.1, 0.25, 0.4, 0.49] {
            let g = posterior(&observed, p, 0.7).expect("well posed");
            let want = ((1.0 - p) / p).ln() / 2.0;
            for (i, &t) in observed.pixels.iter().enumerate() {
                let sign = if t { 1.0 } else { -1.0 };
                assert!(
                    (g.h[i] - sign * want).abs() < 1e-12,
                    "p = {p}: field {} is not {} = sign * ln((1-p)/p) / 2",
                    g.h[i],
                    sign * want
                );
            }
            // The couplings are the prior and do not move with the channel.
            for &w in &g.w {
                assert!((w - 0.7).abs() < 1e-12);
            }
        }

        // AND THE CONSEQUENCE, which is what makes the formula load-bearing rather than decorative.
        //
        // A nearly-clean channel overrules the prior -- but only where it is STRONG ENOUGH to, and
        // the threshold is the field against the worst neighbourhood a 4-connected grid can put up:
        // flipping an isolated pixel costs `2 * (lambda/2)` in data and saves at most `2 * 4 * J`
        // in couplings, so the observation survives exactly when `lambda / 2 > 4 J`. My first cut
        // of this test asserted it at `J = 2` with `lambda/2 = 6.9`, and the prior correctly ate
        // the isolated pixels. The inequality is asserted here rather than assumed.
        let speckled = Image::new(5, 5, (0..25).map(|i| i % 7 == 0).collect()).expect("5x5");
        let (p_clean, j_weak) = (1e-6_f64, 1.0_f64);
        let half_lambda = ((1.0 - p_clean) / p_clean).ln() / 2.0;
        assert!(half_lambda > 4.0 * j_weak, "{half_lambda} must outweigh a full neighbourhood");
        let clean = restore(&speckled, p_clean, j_weak, Estimator::Map, 1).expect("well posed");
        assert_eq!(clean.image, speckled, "a channel that never lies must be believed exactly");
        // The same channel under a prior strong enough to outvote it does NOT give the observation
        // back, which is what says the assertion above is about the inequality and not about `p`.
        let j_strong = 3.0;
        assert!(half_lambda < 4.0 * j_strong);
        let eaten = restore(&speckled, p_clean, j_strong, Estimator::Map, 1).expect("well posed");
        assert_ne!(eaten.image, speckled, "a prior that outweighs the channel must overrule it");
        // And a channel that barely says anything must be overruled by the prior: one flat image.
        let vague = restore(&speckled, 0.499_999, 2.0, Estimator::Map, 1).expect("well posed");
        let first = vague.image.pixels[0];
        assert!(
            vague.image.pixels.iter().all(|&q| q == first),
            "a channel that says nothing must leave the prior in charge"
        );
        // Both are exact, so neither is an artefact of a sampler that had not mixed.
        assert_eq!(clean.gap(), Some(0.0));
        assert_eq!(vague.gap(), Some(0.0));
    }

    /// **THE CLAIM THIS MODULE RESTS ON, WITH NO SAMPLER IN IT.** The estimator a Boltzmann machine
    /// computes natively (per-pixel posterior mode) must beat the one a max-flow computes exactly
    /// (the single most probable image) on the quantity the task is scored by — wrong pixels.
    ///
    /// Both sides are exact: marginals by elimination, the minimiser by roof duality. If this
    /// failed, no sampler could rescue the argument and this module would be recommending the
    /// wrong estimator.
    #[test]
    fn an_mpm_estimate_beats_the_exact_map_on_pixel_error() {
        let (w, h, j, p) = (6usize, 10usize, 0.45, 0.25);
        let mut rng = crate::rng::Pcg::new(0x00B1_A5ED, 1);
        let (mut map_wrong, mut mpm_wrong, mut noisy_wrong) = (0usize, 0usize, 0usize);
        let trials = 200;
        for _ in 0..trials {
            let (truth, noisy) = draw_and_corrupt(w, h, j, p, &mut rng);
            noisy_wrong += truth.differences(&noisy).expect("same shape");
            let map = restore(&noisy, p, j, Estimator::Map, 1).expect("well posed");
            assert_eq!(map.gap(), Some(0.0), "the MAP arm must be exact on a submodular model");
            map_wrong += truth.differences(&map.image).expect("same shape");
            let (mpm, _) = exact_mpm(&noisy, p, j).expect("a 6-wide grid eliminates");
            mpm_wrong += truth.differences(&mpm).expect("same shape");
        }
        let pixels = (trials * w * h) as f64;
        let (rate_noisy, rate_map, rate_mpm) = (
            noisy_wrong as f64 / pixels,
            map_wrong as f64 / pixels,
            mpm_wrong as f64 / pixels,
        );
        // Both estimators must actually be doing something: a restoration worse than the input is
        // not a restoration, and this is the assertion that would catch a model built wrong.
        assert!(rate_map < rate_noisy * 0.9, "MAP {rate_map:.4} vs the noisy input {rate_noisy:.4}");
        assert!(rate_mpm < rate_noisy * 0.9, "MPM {rate_mpm:.4} vs the noisy input {rate_noisy:.4}");
        // And the finding: the sampler's estimator wins the metric the task is scored by.
        assert!(
            rate_mpm < rate_map,
            "MPM {rate_mpm:.4} must beat MAP {rate_map:.4} on pixel error; noisy input {rate_noisy:.4}"
        );
        eprintln!("pixel error over {trials} exact draws: noisy {rate_noisy:.4}, MAP {rate_map:.4}, MPM {rate_mpm:.4}");
    }

    /// The sampled MPM must converge to the exact MPM, marginal by marginal. Comparing the
    /// LABELLINGS alone would be far too forgiving: most marginals are near 0 or 1, so a badly
    /// wrong chain still lands on the same picture.
    #[test]
    fn the_sampled_posterior_converges_to_the_exact_one() {
        let (w, h, j, p) = (6usize, 8usize, 0.4, 0.2);
        let mut rng = crate::rng::Pcg::new(0x5EED_1234, 1);
        let (_, noisy) = draw_and_corrupt(w, h, j, p, &mut rng);
        let (exact_img, exact_m) = exact_mpm(&noisy, p, j).expect("eliminates");

        let mut worst = Vec::new();
        for sweeps in [200usize, 2_000, 20_000] {
            let r = restore(
                &noisy,
                p,
                j,
                Estimator::Mpm { chains: 4, burn_in: 200, sweeps },
                0xA5A5,
            )
            .expect("well posed and converged");
            let e = r
                .posterior
                .iter()
                .zip(&exact_m)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f64, f64::max);
            worst.push(e);
        }
        // Ten times the draws must buy roughly the sqrt(10) ~ 3.16x the accuracy. Asserting only
        // that the last one is small would pass for a chain that was never sampling the right
        // distribution but happened to be biased towards it.
        assert!(worst[2] < worst[0] / 3.0, "worst marginal error by budget: {worst:?}");
        assert!(worst[2] < 0.02, "the longest run must be close: {worst:?}");

        // AND THE PICTURE — but not by asserting the two labellings are identical, which is an
        // assertion about a TIE-BREAK. A marginal of 0.5063 is what pixel 8 of this instance
        // actually holds, about one standard error from a tie at this budget, so demanding the
        // same label there is demanding a coin land the same way twice.
        //
        // The honest statement is self-calibrating: every pixel whose exact marginal is further
        // from a tie than this run's own worst marginal error must agree.
        let r = restore(&noisy, p, j, Estimator::Mpm { chains: 4, burn_in: 500, sweeps: 20_000 }, 1)
            .expect("converged");
        let tol = r
            .posterior
            .iter()
            .zip(&exact_m)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        let mut decided = 0usize;
        for (i, &m) in exact_m.iter().enumerate() {
            if (m - 0.5).abs() > tol {
                decided += 1;
                assert_eq!(
                    r.image.pixels[i], exact_img.pixels[i],
                    "pixel {i}: exact marginal {m:.6} is decided, and the sampled estimate disagrees"
                );
            }
        }
        // ...and that must not be a handful of easy pixels. If the tie band swallowed the image,
        // the loop above would pass by testing almost nothing.
        assert!(
            decided * 10 >= exact_m.len() * 9,
            "only {decided} of {} pixels were decided at tolerance {tol:.4}",
            exact_m.len()
        );
    }

    /// The bill and its grade: what the run cost, priced on each machine, refused by the machine
    /// that never metered a read.
    #[test]
    fn an_answer_arrives_with_a_bill_whose_grade_is_its_weakest_link() {
        let mut pixels = vec![true; 36];
        pixels[8] = false;
        pixels[20] = false;
        let observed = Image::new(6, 6, pixels).expect("6x6");
        let r = restore(&observed, 0.2, 0.7, Estimator::Mpm { chains: 4, burn_in: 100, sweeps: 500 }, 3)
            .expect("converged");

        // Every operation is accounted: the model load, the sweeps, and every draw carried out.
        assert_eq!(r.cost.writes, 36, "loading the model is a write per pixel");
        assert_eq!(r.cost.samples, 36 * 4 * (100 + 500), "burn-in is sampling too, and is charged");
        assert_eq!(r.cost.reads, 36 * 4 * 500, "one read per pixel per recorded draw");

        // NO MACHINE IN THIS CRATE CAN PRICE THIS RUN, and the reason is the finding. Restoring an
        // image writes one field per pixel -- the observation IS the field -- so every frame is a
        // full reprogram, and `e_write` is unstated everywhere because the write on our own board
        // has not been metered. A bare `None` would hide which half is missing.
        assert_eq!(r.joules(&KV260_AXI_METERED), None, "nobody has metered a write");
        assert_eq!(r.joules(&KV260_MEASURED), None, "nor a read, on that older bitstream");
        assert_eq!(r.joules(&Z1_SPICE), Some(r.cost.joules(&Z1_SPICE).expect("a projection states all three")));

        let bill = r.bill(&KV260_AXI_METERED);
        assert!(!bill.complete(), "a bill with unpriced operations must say so");
        assert_eq!(bill.unpriced.writes, 36, "the unpriced half is exactly the per-frame reprogram");
        assert_eq!((bill.unpriced.samples, bill.unpriced.reads), (0, 0));
        assert!(bill.priced > 0.0, "and the metered half is still priced");
        assert_eq!(bill.evidence, Evidence::Metered);
        // A projection prices every operation, and the answer inherits the projection's grade.
        assert!(r.bill(&Z1_SPICE).complete());
        assert_eq!(r.bill(&Z1_SPICE).evidence, Evidence::Simulated);

        // The readback dominates the part that IS priced: 581 pJ a read against 9.13 pJ a flip is
        // the whole argument for reading less often, and here it is in an application's own bill.
        let sampling = (r.cost.samples as f64) * KV260_AXI_METERED.e_sample;
        assert!(
            sampling / bill.priced < 0.25,
            "the readback is the bill: sampling is only {:.1}% of it",
            100.0 * sampling / bill.priced
        );

        // And the receipt re-derives the energy from the model rather than trusting the run.
        let g = posterior(&observed, 0.2, 0.7).expect("well posed");
        let v = r.receipt.verify(&g).expect("the receipt must verify against its own model");
        assert!((v.energy - r.energy).abs() < 1e-12);
    }

    /// **THE SECOND TASK, AND THE SECOND TIME THE SAMPLER'S CASE IS THE SECOND MOMENT.** Three
    /// routes to one Gaussian's marginals, all three held to an exact inverse: every route has the
    /// mean, only two have the variance, and the one that is wrong is the one robotics runs.
    ///
    /// The shape of the argument is the same as the image task's — a deterministic method owns the
    /// first moment, and the sampler earns its place on the second — which is worth noticing,
    /// because it says where to look for the next application rather than where to hope.
    #[test]
    fn three_routes_to_one_gaussians_marginals_and_only_two_have_the_variance() {
        let model = crate::gbp::grid(5, 5, 1.0, -0.22);
        let exact = marginals(&model, Route::Exact).expect("positive definite");
        assert_eq!(exact.cost, Ledger::default(), "the oracle runs on no device and is charged as none");

        let bp = marginals(
            &model,
            Route::MessagePassing { max_iters: 20_000, tol: 1e-14, damping: 0.0 },
        )
        .expect("this grid converges");
        let sampled = marginals(&model, Route::Sampling { stride_h: 3.0, burn: 200, samples: 100_000 })
            .expect("positive definite");

        // EVERY route has the mean. This is the half a deterministic method owns.
        for r in [&bp, &sampled] {
            let e = (0..model.n)
                .map(|i| (r.mean[i] - exact.mean[i]).abs())
                .fold(0.0f64, f64::max);
            assert!(e < 1e-2, "{:?}: worst mean error {e:e}", r.route);
        }

        // THE VARIANCE SPLITS THEM. Message passing is overconfident on every node, by a margin
        // that is a property of the graph; the sampler is inside its own standard error.
        let bp_err = (0..model.n)
            .map(|i| (bp.variance[i] - exact.variance[i]).abs())
            .fold(0.0f64, f64::max);
        let s_err = (0..model.n)
            .map(|i| (sampled.variance[i] - exact.variance[i]).abs())
            .fold(0.0f64, f64::max);
        assert!(
            (0..model.n).all(|i| bp.variance[i] < exact.variance[i]),
            "message passing must be overconfident on every node"
        );
        assert!(bp_err > 0.05, "and by a visible margin: {bp_err:e}");
        assert!(s_err < bp_err / 2.0, "the sampler must be well under it: {s_err:.5} vs {bp_err:.5}");

        // AND THE BILL, in the operations each route actually performs -- counted EXACTLY, because
        // "more than zero" is what a mutant that billed the sampler per step instead of per node
        // update passed. The sampler's advantage on the second moment is bought, not free.
        let (n, e) = (model.n as u64, model.edges.len() as u64);
        assert_eq!(
            bp.cost.samples,
            bp.iterations as u64 * (n + 2 * e),
            "a message-passing sweep touches every node and every directed message"
        );
        assert_eq!(sampled.cost.samples, (200 + 100_000) * n, "burn-in is node updates too");
        assert_eq!(sampled.cost.reads, 100_000 * n, "one read per node per recorded draw");
        assert_eq!((bp.cost.reads, bp.cost.writes), (n, n));
        let ratio = sampled.cost.samples as f64 / bp.cost.samples as f64;
        assert!(ratio > 1.0, "the accurate variance costs more node updates, not fewer: {ratio:.1}x");
        eprintln!(
            "5x5 grid, worst variance error: message passing {bp_err:.5} in {} sweeps, \
             sampling {s_err:.5} in {} draws ({ratio:.0}x the node updates)",
            bp.iterations, sampled.iterations
        );

        // A model whose messages do not settle is refused by name, through this entry point too.
        let hot = crate::gbp::grid(6, 6, 1.0, -0.9);
        let out = marginals(&hot, Route::MessagePassing { max_iters: 500, tol: 1e-12, damping: 0.0 });
        assert!(out.is_err_and(|e| e.contains("did not settle")), "an unconverged run is not a belief");
    }

    /// **THE PATH, END TO END, WITH THE HARDWARE AT THE FAR END OF IT.** An image becomes a
    /// posterior, the posterior becomes a p-bit netlist, and the EMITTED VERILOG — simulated by
    /// icarus-verilog, not by this crate — reaches the state the restoration reports.
    ///
    /// A 2026-09-20 survey of this field did not locate an open path from a problem-level model to
    /// emitted p-bit RTL anywhere. This is that path, and the assertion is bit-for-bit rather than
    /// distributional: the Verilog's spins after `k` sweeps equal the emulator's, exactly.
    #[test]
    fn the_restored_image_comes_out_of_emitted_hardware() {
        if std::process::Command::new("iverilog").arg("-V").output().is_err() {
            eprintln!("SKIP: iverilog not installed; the emitted-hardware gate did not run");
            return;
        }
        // 5x5 keeps the emitted Verilog small enough to simulate in a unit test; the fabric this
        // crate metered on a KV260 is the same design at 1,024 p-bits.
        let speckle: Vec<bool> = (0..25).map(|i| i % 4 != 0).collect();
        let observed = Image::new(5, 5, speckle).expect("5x5");
        let (p, j, sweeps) = (0.2, 0.7, 17usize);
        let g = posterior(&observed, p, j).expect("well posed");
        let mut fab = crate::writable::WritableFabric::new(&g, 1.0, 0xF00D)
            .expect("a posterior at this noise and smoothness fits Q.8 in twelve bits");

        // What the emulator reaches -- and what `restore` would report from the same netlist.
        for _ in 0..sweeps {
            fab.core.sweep();
        }
        let want = fab
            .core
            .s
            .iter()
            .enumerate()
            .fold(0u32, |a, (i, &b)| a | (u32::from(b) << i));
        // A gate that could not fail is not a gate: one more sweep must move the state, or any
        // run length would reproduce `want`.
        let mut one_more = crate::writable::WritableFabric::new(&g, 1.0, 0xF00D).expect("fits");
        for _ in 0..=sweeps {
            one_more.core.sweep();
        }
        assert_ne!(one_more.core.s, fab.core.s, "vacuous: the fabric has frozen by sweep {sweeps}");

        let core = fab.emit_verilog("wfab");
        let shell = fab.emit_axi_shell("wf_axi", "wfab", "clk");
        let tb = format!(
            r#"`timescale 1ns/1ps
module tb;
  reg clk = 0, rst_n = 0;
  reg [31:0] awaddr = 0, wdata = 0, araddr = 0;
  reg awvalid = 0, wvalid = 0, bready = 1, arvalid = 0, rready = 1;
  wire awready, wready, bvalid, arready, rvalid;
  wire [31:0] rdata; wire [1:0] bresp, rresp;
  wf_axi dut(.clk(clk), .rst_n(rst_n),
    .s_axi_awaddr(awaddr), .s_axi_awvalid(awvalid), .s_axi_awready(awready),
    .s_axi_wdata(wdata), .s_axi_wstrb(4'hF), .s_axi_wvalid(wvalid), .s_axi_wready(wready),
    .s_axi_bresp(bresp), .s_axi_bvalid(bvalid), .s_axi_bready(bready),
    .s_axi_araddr(araddr), .s_axi_arvalid(arvalid), .s_axi_arready(arready),
    .s_axi_rdata(rdata), .s_axi_rresp(rresp), .s_axi_rvalid(rvalid), .s_axi_rready(rready));
  always #5 clk = ~clk;
  reg [31:0] got, got_done, got_status;
  integer guard;
  reg aw_done, w_done;
  task wr(input [31:0] a, input [31:0] d);
    begin
      aw_done = 0; w_done = 0;
      @(posedge clk); awaddr <= a; wdata <= d; awvalid <= 1; wvalid <= 1;
      while (!aw_done || !w_done) begin
        @(posedge clk);
        if (awready) begin awvalid <= 0; aw_done = 1; end
        if (wready)  begin wvalid  <= 0; w_done  = 1; end
      end
      while (!bvalid) @(posedge clk);
      @(posedge clk);
    end
  endtask
  task rd(input [31:0] a, output [31:0] d);
    begin
      @(posedge clk); araddr <= a; arvalid <= 1;
      @(posedge clk); while (!arready) @(posedge clk);
      arvalid <= 0;
      while (!rvalid) @(posedge clk);
      d = rdata;
      @(posedge clk);
    end
  endtask
  initial begin #20000000; $display("FERROTHERM_FAIL watchdog"); $finish; end
  initial begin
    repeat (4) @(posedge clk); rst_n = 1; repeat (2) @(posedge clk);
    wr(32'h08, 32'd{sweeps});
    wr(32'h00, 32'h1);
    guard = 0; got_status = 0;
    while ((got_status[1] !== 1'b1) && guard < 200000) begin rd(32'h04, got_status); guard = guard + 1; end
    if (got_status[1] !== 1'b1) begin $display("FERROTHERM_FAIL never reached target"); $finish; end
    rd(32'h0C, got_done);
    rd(32'h20, got);
    if (got_done !== 32'd{sweeps}) $display("FERROTHERM_FAIL sweeps %0d want {sweeps}", got_done);
    else if (got !== 32'h{want:08x}) $display("FERROTHERM_FAIL state %h want {want:08x}", got);
    else $display("FERROTHERM_PASS");
    $finish;
  end
endmodule
"#
        );
        let dir = std::env::temp_dir().join(format!("ferrotherm_restore_hw_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("fabric.v"), core).unwrap();
        std::fs::write(dir.join("shell.v"), shell).unwrap();
        std::fs::write(dir.join("tb.v"), tb).unwrap();
        let out = std::process::Command::new("iverilog")
            .current_dir(&dir)
            .args(["-g2012", "-o", "sim", "fabric.v", "shell.v", "tb.v"])
            .output()
            .unwrap();
        assert!(out.status.success(), "iverilog: {}", String::from_utf8_lossy(&out.stderr));
        let run = std::process::Command::new("vvp").current_dir(&dir).arg("sim").output().unwrap();
        let stdout = String::from_utf8_lossy(&run.stdout);
        assert!(stdout.contains("FERROTHERM_PASS"), "emitted-hardware gate:\n{stdout}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The fabric route as an application: it restores, it is graded, and its bill is in the unit
    /// the KV260 was metered in — a p-bit update and a spin read.
    #[test]
    fn the_fabric_route_restores_and_bills_in_the_unit_that_was_metered() {
        use crate::ledger::KV260_AXI_METERED;
        let speckle: Vec<bool> = (0..64).map(|i| i % 5 != 0).collect();
        let observed = Image::new(8, 8, speckle).expect("8x8");
        let (chains, burn_in, sweeps) = (4usize, 200usize, 2_000usize);
        let r = restore(&observed, 0.2, 0.7, Estimator::Fabric { chains, burn_in, sweeps }, 0xB0A7)
            .expect("a posterior that fits, and chains that mixed");

        // THE COUNTS, EXACTLY. A netlist per chain, because the generators are reset constants.
        let n = 64u64;
        assert_eq!(r.cost.writes, n * chains as u64, "a chain is a netlist on this fabric");
        assert_eq!(r.cost.samples, n * chains as u64 * (burn_in + sweeps) as u64);
        assert_eq!(r.cost.reads, n * chains as u64 * sweeps as u64);

        // It restores: the speckle is smoothed, so the answer is closer to plain white than the
        // observation was.
        let white = Image::new(8, 8, vec![true; 64]).expect("8x8");
        let before = observed.differences(&white).expect("same shape");
        let after = r.image.differences(&white).expect("same shape");
        assert!(after < before, "the fabric did not restore anything: {after} vs {before}");

        // AND IT RESTORES THE RIGHT POSTERIOR, which "it restored something" does not check: a
        // fabric quantised at the wrong temperature smooths an image too, just a different one.
        // Held against the exact marginals, and against the same estimator in floating point, so
        // what is measured is the QUANTISATION and nothing else.
        let long = Estimator::Fabric { chains: 4, burn_in: 500, sweeps: 20_000 };
        let fab = restore(&observed, 0.2, 0.7, long, 1).expect("converged");
        let cpu = restore(&observed, 0.2, 0.7, Estimator::Mpm { chains: 4, burn_in: 500, sweeps: 20_000 }, 1)
            .expect("converged");
        let (_, exact) = exact_mpm(&observed, 0.2, 0.7).expect("an 8-wide grid eliminates");
        let worst = |v: &[f64]| {
            v.iter().zip(&exact).map(|(a, b)| (a - b).abs()).fold(0.0f64, f64::max)
        };
        let (e_fab, e_cpu) = (worst(&fab.posterior), worst(&cpu.posterior));
        assert!(e_fab < 0.01, "the fabric is not sampling this posterior: worst marginal off by {e_fab}");
        // Q.8 weights and a 1,024-entry sigmoid ROM cost a small multiple of the sampling error --
        // not a different problem. Stated as a band so it catches drift in either direction.
        assert!(
            e_fab < 8.0 * e_cpu,
            "quantisation cost {e_fab} against floating point's {e_cpu}: that is a different model, \
             not a quantised one"
        );
        // And nothing the data decides comes out differently.
        let mut decided = 0usize;
        for (k, m) in exact.iter().enumerate() {
            if (m - 0.5).abs() > 0.05 {
                decided += 1;
                assert_eq!(
                    fab.posterior[k] > 0.5,
                    *m > 0.5,
                    "pixel {k}: exact marginal {m:.4} is decided and the fabric disagrees"
                );
            }
        }
        assert!(decided * 10 >= 9 * exact.len(), "only {decided} of {} pixels decided", exact.len());
        eprintln!(
            "emitted p-bit fabric vs the exact posterior: worst marginal off by {e_fab:.4} \
             (floating point {e_cpu:.4}), {decided}/{} pixels decided and none disagree",
            exact.len()
        );

        // And the bill is in metered units -- with the write still unpriced, as it is everywhere.
        let bill = r.bill(&KV260_AXI_METERED);
        assert!(bill.priced > 0.0);
        assert!(!bill.complete() && bill.unpriced.writes == n * chains as u64);
        assert_eq!(bill.evidence, crate::ledger::Evidence::Metered);
        eprintln!(
            "8x8 restored on the emitted fabric: {:.3} uJ of sampling and readback at metered \
             prices, {} node configurations unpriced",
            bill.priced * 1e6,
            bill.unpriced.writes
        );
    }

    /// Every refusal, and the reason each one is a refusal rather than a default.
    #[test]
    fn an_ill_posed_task_is_refused_by_name() {
        let ok = Image::new(3, 3, vec![true; 9]).expect("3x3");
        assert!(Image::new(3, 3, vec![true; 8]).is_none(), "a shape that does not match its pixels");
        // p = 0.5: the observation says nothing. p = 0: it says everything, and the field is
        // infinite. Both are refused rather than clamped into a model that is not the caller's.
        for p in [0.0, 0.5, 0.75, f64::NAN] {
            assert!(matches!(
                restore(&ok, p, 1.0, Estimator::Map, 1),
                Err(Refused::FlipProbability(_))
            ), "flip probability {p}");
        }
        // Negative smoothness is the one that would silently break the ORACLE: it makes the energy
        // supermodular, and the MAP arm's max-flow is exact only on submodular energies.
        for j in [0.0, -1.0, f64::INFINITY, f64::NAN] {
            assert!(matches!(
                restore(&ok, 0.1, j, Estimator::Map, 1),
                Err(Refused::Smoothness(_))
            ), "smoothness {j}");
        }
        assert!(matches!(
            restore(&ok, 0.1, 1.0, Estimator::Mpm { chains: 1, burn_in: 10, sweeps: 10 }, 1),
            Err(Refused::TooFewChains(1))
        ));
        // A run too short to have mixed is refused, and the message is the diagnostic's own.
        let stripes: Vec<bool> = (0..64).map(|i| (i / 8) % 2 == 0).collect();
        let hard = Image::new(8, 8, stripes).expect("8x8");
        let short = restore(&hard, 0.02, 3.0, Estimator::Mpm { chains: 4, burn_in: 0, sweeps: 6 }, 9);
        assert!(
            matches!(short, Err(Refused::NotConverged(_))),
            "a six-sweep run on a frozen model must not be graded as converged: {}",
            short.map_or_else(|e| e.to_string(), |r| format!("accepted, energy {}", r.energy))
        );
    }
}
