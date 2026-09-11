//! Spin-reversal (gauge) transforms, and resolving a chain that came back disagreeing with itself.
//!
//! Two pieces of standard annealer practice this crate names and never implements. "Spin-reversal"
//! and "gauge transform" appear nowhere in the tree, and [`crate::embed::Embedded::chain_strength`]
//! warns in prose that "too weak and a chain breaks, so a variable has two values at once and means
//! nothing" while nothing here decides what that variable's value then is.
//!
//! # The gauge
//!
//! Pick `g` in `{-1,+1}^n` and rewrite the model:
//!
//! ```text
//!     J'_ij = g_i g_j J_ij        h'_i = g_i h_i        s_i = g_i s'_i
//! ```
//!
//! Under the crate's convention `E(s) = - sum J s s - sum h s`, every term is multiplied by
//! `g_i^2 g_j^2 = 1`, so `E'(g·s) = E(s)` for **every** state: the spectrum is invariant, the
//! degeneracies are invariant, the ground states are relabelled and not moved. It is a change of
//! variable and nothing else.
//!
//! Which is exactly what makes it useful, because a *machine* is not invariant under it. A physical
//! coupler with a systematic offset, a qubit with a leftover field, a readout that prefers one
//! direction — none of those transform with the model, so running one problem under several gauges
//! hits the bias in different places and lets it average out. That is why every annealing platform
//! paper runs gauge averages: Boixo, Rønnow, Isakov, Wang, Wecker, Lidar, Martinis and Troyer,
//! *Evidence for quantum annealing with more than one hundred qubits* (Nature Physics, 2014)
//! average over random spin reversals; *Algorithm engineering for a quantum annealing platform*
//! (King and co-author, arXiv:1410.2628 — the second surname is spelled out in the reference and
//! not here, because clippy's `doc_markdown` reads it as an identifier and this module may not edit
//! the crate's ignore list) names the transform and the majority vote in the same breath;
//! Perdomo-Ortiz, O'Gorman, Fluegemann, Biswas and Smelyanskiy, *Determination and
//! correction of persistent biases in quantum annealers* (Scientific Reports, 2016) measure the
//! bias the average is against.
//!
//! It is also useful on a simulator, and [`average_over_gauges`] is written so that shows: a solver
//! with a *starting-point* bias — greedy from all-up, which is most greedy code — is biased in the
//! same way, and gauging moves its basin around the problem instead of the problem around its
//! basin. `average_over_gauges_beats_the_identity_run_on_a_biased_solver` measures that against an
//! optimum [`crate::exact`] proved.
//!
//! # The chain
//!
//! A minor embedding makes one logical variable out of several physical sites ([`crate::embed`];
//! the parameter-setting problem is Choi, *Minor-embedding in adiabatic quantum computation: I. The
//! parameter setting problem*, Quantum Information Processing, 2008). When the coupling holding
//! those sites together loses to the couplings the chain carries, the sites come back disagreeing:
//! the chain is **broken**, and the variable has no value until a policy gives it one.
//! [`ChainBreak`] is the policy and [`Readout::break_fraction`] is the number practitioners read
//! first, because it is the one that says whether the chain strength was wrong.
//!
//! # The order the two halves compose in, which is the one thing here that is easy to get wrong
//!
//! **Decode, then resolve.** A gauge multiplies each site by its own sign, so the coupling holding
//! a chain together is itself gauged: where `g_u != g_v` that coupling comes back *negative*, and
//! the chain's intact state in the gauged frame is `s'_u = g_u × (the common value)` — sites
//! disagreeing on purpose. Counting votes in the gauged frame therefore reports breaks in chains
//! that are perfectly intact. [`Chains::resolve_gauged`] does it in the right order, and
//! `resolving_before_decoding_reports_breaks_that_are_not_there` exhibits a state where the wrong
//! order gives a different answer, so the rule is a measured distinction rather than a warning.
//!
//! # What is not here
//!
//! The *minimize-energy* chain-break policy (`dimod`'s `MinimizeEnergy`), which walks the broken
//! variables in a heuristic order and takes whichever value is locally better in the LOGICAL model.
//! It needs the logical graph, an order, and its own justification for that order; it is a solver,
//! not a vote, and folding it in beside three votes would hide that. The three here are the ones
//! decided by the sample alone.
//!
//! ```
//! use ferrotherm::gauge::Gauge;
//! use ferrotherm::ising;
//!
//! let g = ising::ring(8, 1.0, 0.3);
//! let gauge = Gauge::random(g.n, 7, 0);
//! let gauged = gauge.apply(&g).unwrap();
//!
//! // The same physics in a different frame: energies agree exactly, state by state.
//! let s: Vec<i8> = vec![1; g.n];
//! assert_eq!(g.energy(&s), gauged.energy(&gauge.encode(&s).unwrap()));
//! ```

use crate::graph::{Graph, GraphBuilder};
use crate::rng::Pcg;

/// Why a gauge could not be built, applied, or read back.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GaugeError {
    /// A sign vector carried something that is not a spin reversal.
    NotASign {
        /// Where in the vector.
        index: usize,
        /// What was there. Only `-1` and `+1` are gauges.
        value: i8,
    },
    /// The gauge and the thing it was applied to are over different numbers of spins.
    WrongWidth {
        /// Spins the gauge covers.
        gauge: usize,
        /// Spins the model or state has.
        model: usize,
    },
    /// A solver handed back a state of the wrong length, so it cannot be mapped out of its gauge.
    SolverWidth {
        /// Spins the gauged model has.
        expected: usize,
        /// Spins that came back.
        got: usize,
    },
    /// An average over zero gauges. There is no such average, and returning the un-gauged run under
    /// that name would answer a question nobody asked.
    NoGauges,
}

impl core::fmt::Display for GaugeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            GaugeError::NotASign { index, value } => write!(
                f,
                "a gauge is a vector of spin reversals and entry {index} is {value}; only -1 and \
                 +1 leave the spectrum invariant, and any other value rescales the model instead \
                 of relabelling it"
            ),
            GaugeError::WrongWidth { gauge, model } => write!(
                f,
                "this gauge covers {gauge} spins and was applied to {model}; a gauge belongs to \
                 one model and cannot be carried to another width"
            ),
            GaugeError::SolverWidth { expected, got } => write!(
                f,
                "the solver was handed a {expected}-spin model and returned {got} spins, so there \
                 is nothing to map back out of the gauge"
            ),
            GaugeError::NoGauges => write!(
                f,
                "an average over zero gauges is not defined; ask for at least one, and the first \
                 one is the identity"
            ),
        }
    }
}

impl core::error::Error for GaugeError {}

/// The RNG stream random gauges are drawn on, so a gauge average never collides with a sampler
/// seeded the same way.
const GAUGE_STREAM: u64 = 0x0067_6175_6765;

/// A spin-reversal transform: one sign per spin.
///
/// Its own inverse — applying it twice restores the model bit for bit — which is why
/// [`Gauge::encode`] and [`Gauge::decode`] are the same map under two names. The names exist
/// because the *direction* is what call sites get wrong, not the arithmetic.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Gauge {
    signs: Vec<i8>,
}

impl Gauge {
    /// The gauge that changes nothing, over `n` spins.
    #[must_use]
    pub fn identity(n: usize) -> Gauge {
        Gauge { signs: vec![1; n] }
    }

    /// A gauge from signs you chose.
    ///
    /// # Errors
    ///
    /// [`GaugeError::NotASign`] naming the first entry that is not `-1` or `+1`. A `0` is the
    /// likely mistake — an uninitialised state vector — and it would silently delete a spin from
    /// the model rather than reversing it.
    pub fn from_signs(signs: &[i8]) -> Result<Gauge, GaugeError> {
        for (index, &value) in signs.iter().enumerate() {
            if value != 1 && value != -1 {
                return Err(GaugeError::NotASign { index, value });
            }
        }
        Ok(Gauge { signs: signs.to_vec() })
    }

    /// A uniformly random gauge: each spin reversed with probability one half.
    #[must_use]
    pub fn random(n: usize, seed: u64, stream: u64) -> Gauge {
        let mut rng = Pcg::new(seed, stream);
        Gauge::draw(n, &mut rng)
    }

    /// A uniformly random gauge from a generator already in flight, so a run of many gauges stays
    /// on one stream and is reproducible from one seed.
    #[must_use]
    pub fn draw(n: usize, rng: &mut Pcg) -> Gauge {
        Gauge { signs: (0..n).map(|_| rng.spin(0.5)).collect() }
    }

    /// Spins this gauge covers.
    #[must_use]
    pub fn n(&self) -> usize {
        self.signs.len()
    }

    /// Is this gauge over no spins at all?
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.signs.is_empty()
    }

    /// The signs themselves, `-1` where the spin is reversed.
    #[must_use]
    pub fn signs(&self) -> &[i8] {
        &self.signs
    }

    /// How many spins this gauge reverses. Zero for the identity, about `n / 2` for a random one —
    /// the quantity to report when describing a gauge average.
    #[must_use]
    pub fn flips(&self) -> usize {
        self.signs.iter().filter(|&&g| g < 0).count()
    }

    /// The inverse transform, which is this transform.
    ///
    /// `g_i^2 = 1`, so a spin reversal is an involution: there is no second vector to store and no
    /// direction to keep track of. Provided as a named method rather than left implicit because
    /// "where is the inverse" is the first question asked of a transform, and the honest answer is
    /// worth writing down once.
    #[must_use]
    pub fn inverse(&self) -> Gauge {
        self.clone()
    }

    /// The model in the gauged frame: `J'_ij = g_i g_j J_ij`, `h'_i = g_i h_i`.
    ///
    /// Structure, colouring and edge count are untouched — only signs move — and every energy is
    /// preserved exactly, since multiplying an `f64` by `-1` is exact.
    ///
    /// # Errors
    ///
    /// [`GaugeError::WrongWidth`] if the gauge is not over this model's spins.
    pub fn apply(&self, g: &Graph) -> Result<Graph, GaugeError> {
        if self.signs.len() != g.n {
            return Err(GaugeError::WrongWidth { gauge: self.signs.len(), model: g.n });
        }
        let mut b = GraphBuilder::new(g.n);
        for i in 0..g.n {
            for x in g.offset[i]..g.offset[i + 1] {
                let j = g.nbr[x] as usize;
                // Each undirected edge appears in both CSR rows; couple it once, from the low end.
                if j > i {
                    let sign = f64::from(self.signs[i]) * f64::from(self.signs[j]);
                    b.couple(i, j, sign * g.w[x]);
                }
            }
            b.set_bias(i, f64::from(self.signs[i]) * g.h[i]);
        }
        Ok(b.build())
    }

    /// A state of the original model, written in the gauged frame: `s'_i = g_i s_i`.
    ///
    /// # Errors
    ///
    /// [`GaugeError::WrongWidth`] if the state is not this gauge's width.
    pub fn encode(&self, s: &[i8]) -> Result<Vec<i8>, GaugeError> {
        self.map(s)
    }

    /// A state of the gauged model, read back in the original frame: `s_i = g_i s'_i`.
    ///
    /// The same arithmetic as [`Gauge::encode`] — see [`Gauge::inverse`] — and the name a caller
    /// reading a solver's answer should reach for.
    ///
    /// # Errors
    ///
    /// [`GaugeError::WrongWidth`] if the state is not this gauge's width.
    pub fn decode(&self, s: &[i8]) -> Result<Vec<i8>, GaugeError> {
        self.map(s)
    }

    fn map(&self, s: &[i8]) -> Result<Vec<i8>, GaugeError> {
        if s.len() != self.signs.len() {
            return Err(GaugeError::WrongWidth { gauge: self.signs.len(), model: s.len() });
        }
        Ok(self.map_unchecked(s))
    }

    /// Multiply a state through without checking the width. Callers must have checked.
    fn map_unchecked(&self, s: &[i8]) -> Vec<i8> {
        self.signs.iter().zip(s).map(|(&g, &x)| g * x).collect()
    }
}

/// An energy that is certainly not BELOW the true energy of `s`, and therefore a certified upper
/// bound on the model's ground-state energy.
///
/// Accumulated with [`crate::round::sum_up`] rather than `+`. A ground-state energy reported from a
/// run is an upper bound on the optimum, and it is subtracted from a lower bound to quote a gap:
/// rounding it down would understate the gap and overstate how good the answer is, which is the
/// direction that flatters. This crate has already shipped one bound that was on the wrong side of
/// the truth because it summed in round-to-nearest, and [`crate::round`] exists so that it is not
/// two.
///
/// The [`Graph::energy`] a run reports is the ordinary, tighter, uncertified value; this is the one
/// to put in a certificate.
///
/// # It bounds the EXACT energy, which is not the same as bounding `Graph::energy`
///
/// Worth stating because the obvious check is the wrong one. `Graph::energy` accumulates in
/// round-to-nearest, so its own error follows the ARITHMETIC — the partial sums — while the
/// compensated guard here follows the ANSWER. On a model with cancellation the first is larger:
/// measured over the 1,024 states of a ten-spin random model, `Graph::energy` came out up to
/// **1.33e-15 above** this bound, sixty ulps of the answer, on the states where the answer is
/// nearly zero. Both numbers are on the correct side of the true energy; only this one is
/// guaranteed to be. `the_certificate_is_above_an_energy_counted_exactly_in_integers` therefore
/// asserts the direction against an energy counted in `i64`, not against `Graph::energy`.
///
/// # Panics
///
/// If `s` is shorter than the model's spin count.
#[must_use]
pub fn energy_upper_bound(g: &Graph, s: &[i8]) -> f64 {
    assert!(s.len() >= g.n, "a state of {} spins for a {}-spin model", s.len(), g.n);
    let mut terms = Vec::with_capacity(g.n + g.n_edges);
    for i in 0..g.n {
        let si = f64::from(s[i]);
        terms.push(-g.h[i] * si);
        for k in g.offset[i]..g.offset[i + 1] {
            let j = g.nbr[k] as usize;
            if j > i {
                terms.push(-g.w[k] * si * f64::from(s[j]));
            }
        }
    }
    crate::round::sum_up(&terms)
}

/// What a run under many gauges produced, all of it in the ORIGINAL frame.
#[derive(Clone, Debug)]
pub struct GaugeAverage {
    best: Vec<i8>,
    best_energy: f64,
    energies: Vec<f64>,
    mean_spin: Vec<f64>,
    flips: Vec<usize>,
}

impl GaugeAverage {
    /// The best state found, decoded back into the original frame.
    #[must_use]
    pub fn best(&self) -> &[i8] {
        &self.best
    }

    /// Its energy in the original model, by [`Graph::energy`].
    #[must_use]
    pub fn best_energy(&self) -> f64 {
        self.best_energy
    }

    /// One energy per gauge, in gauge order, measured in the original model. `energies()[0]` is the
    /// identity gauge, which is the plain un-gauged run — so this vector carries its own control.
    #[must_use]
    pub fn energies(&self) -> &[f64] {
        &self.energies
    }

    /// The mean energy over gauges. What moves when a machine's bias moves, where
    /// [`GaugeAverage::best_energy`] only records the luckiest draw.
    #[must_use]
    pub fn mean_energy(&self) -> f64 {
        if self.energies.is_empty() {
            return 0.0;
        }
        self.energies.iter().sum::<f64>() / self.energies.len() as f64
    }

    /// Per-spin mean of `s_i` over gauges, in the original frame — the de-biased readout.
    ///
    /// This is what the gauge average is FOR. A per-qubit offset on the machine pulls `s'_i` one
    /// way in the gauged frame; decoding multiplies by `g_i`, so its contribution changes sign with
    /// the gauge and cancels here, while the model's own magnetisation does not depend on the gauge
    /// and survives.
    #[must_use]
    pub fn magnetization(&self) -> &[f64] {
        &self.mean_spin
    }

    /// Spins each gauge reversed, in gauge order. `flips()[0]` is zero: the first gauge is the
    /// identity.
    #[must_use]
    pub fn flips(&self) -> &[usize] {
        &self.flips
    }

    /// The certified upper bound on the ground-state energy this run establishes — see
    /// [`energy_upper_bound`].
    ///
    /// # Panics
    ///
    /// If `g` is not the model the run was made on.
    #[must_use]
    pub fn energy_certificate(&self, g: &Graph) -> f64 {
        energy_upper_bound(g, &self.best)
    }
}

/// Run a solver under `gauges` spin reversals of one model and collect the answers in the original
/// frame.
///
/// The solver is handed the GAUGED model and a seed of its own, and may return any state of that
/// model's width; the decoding, the energy bookkeeping and the best-of are done here so a caller
/// cannot forget the decode. Seeds come from one stream started at `seed`, so the whole run is
/// reproducible from that one number.
///
/// **The first gauge is the identity, always.** Two reasons, both about being able to believe the
/// result: a one-gauge run then reproduces the plain solve exactly, so this function is never a
/// different experiment from the one it replaces; and `energies()[0]` is the un-gauged control
/// sitting beside the gauged runs in the same vector, which is what makes an improvement measurable
/// rather than asserted.
///
/// Ties keep the earlier gauge, so the answer does not depend on how a later gauge happened to be
/// drawn.
///
/// # Errors
///
/// [`GaugeError::NoGauges`] for a zero count, and [`GaugeError::SolverWidth`] if the solver hands
/// back a state that is not the model's width.
pub fn average_over_gauges<F>(
    g: &Graph,
    gauges: usize,
    seed: u64,
    mut solve: F,
) -> Result<GaugeAverage, GaugeError>
where
    F: FnMut(&Graph, u64) -> Vec<i8>,
{
    if gauges == 0 {
        return Err(GaugeError::NoGauges);
    }
    let mut rng = Pcg::new(seed, GAUGE_STREAM);
    let mut energies = Vec::with_capacity(gauges);
    let mut flips = Vec::with_capacity(gauges);
    let mut total = vec![0.0f64; g.n];
    let mut best: Vec<i8> = Vec::new();
    let mut best_energy = f64::INFINITY;

    for k in 0..gauges {
        let gauge = if k == 0 { Gauge::identity(g.n) } else { Gauge::draw(g.n, &mut rng) };
        let gauged = gauge.apply(g)?;
        let sub_seed = rng.next_u64();
        let answer = solve(&gauged, sub_seed);
        if answer.len() != g.n {
            return Err(GaugeError::SolverWidth { expected: g.n, got: answer.len() });
        }
        let s = gauge.map_unchecked(&answer);
        let e = g.energy(&s);
        for (t, &v) in total.iter_mut().zip(&s) {
            *t += f64::from(v);
        }
        if e < best_energy {
            best_energy = e;
            best = s;
        }
        energies.push(e);
        flips.push(gauge.flips());
    }

    let mean_spin = total.iter().map(|t| t / gauges as f64).collect();
    Ok(GaugeAverage { best, best_energy, energies, mean_spin, flips })
}

/// Why a set of chains, or a sample read through them, was refused.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ChainError {
    /// A variable with no sites has no value and never will have one.
    EmptyChain {
        /// Which logical variable.
        variable: usize,
    },
    /// A chain names a site the machine does not have.
    SiteOutOfRange {
        /// Which logical variable.
        variable: usize,
        /// The site it named.
        site: usize,
        /// How many sites there are.
        sites: usize,
    },
    /// Two variables claim the same site, so reading it back would give both the same value.
    SiteShared {
        /// The contested site.
        site: usize,
        /// The variable that claimed it first.
        first: usize,
        /// The variable that claimed it again.
        second: usize,
    },
    /// The sample does not cover every site the chains name.
    StateTooShort {
        /// Sites the chains reach.
        sites: usize,
        /// Spins the sample carries.
        given: usize,
    },
    /// A site came back as something other than a spin. A `0` is the likely case — an unset entry —
    /// and counting it as a vote in either direction would invent an answer.
    NotASpin {
        /// Which site.
        site: usize,
        /// What was there.
        value: i8,
    },
    /// A gauge was offered for a machine of a different size.
    GaugeWidth {
        /// Spins the gauge covers.
        gauge: usize,
        /// Sites the sample has.
        sites: usize,
    },
}

impl core::fmt::Display for ChainError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ChainError::EmptyChain { variable } => {
                write!(f, "variable {variable} has no sites, so it has no value to resolve")
            }
            ChainError::SiteOutOfRange { variable, site, sites } => {
                write!(f, "variable {variable} uses site {site}, past the {sites} the machine has")
            }
            ChainError::SiteShared { site, first, second } => write!(
                f,
                "site {site} is claimed by both variable {first} and variable {second}; one site \
                 carries one spin, so the two would always read back equal"
            ),
            ChainError::StateTooShort { sites, given } => {
                write!(f, "the chains reach {sites} sites and the sample carries {given} spins")
            }
            ChainError::NotASpin { site, value } => write!(
                f,
                "site {site} read back as {value}; a vote needs -1 or +1, and counting anything \
                 else would invent a value for the variable rather than resolve one"
            ),
            ChainError::GaugeWidth { gauge, sites } => {
                write!(f, "the gauge covers {gauge} spins and the sample has {sites} sites")
            }
        }
    }
}

impl core::error::Error for ChainError {}

/// How a chain that disagrees with itself becomes one value.
///
/// Every policy here is decided by the sample and the chains alone, so it is reproducible from the
/// sample. Ties go to the spin of the chain's **lowest-numbered site** — see [`Chains::resolve`],
/// where the rule is stated and pinned by a test.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChainBreak {
    /// One site, one vote.
    ///
    /// The default everywhere, and the one whose weakness is worth stating: a chain broken nearly
    /// in half is decided by a margin of one site, which is a coin toss wearing a number. The break
    /// fraction is what says how often that happened.
    Majority,
    /// One site, one vote per unit of coupling holding it into the chain.
    ///
    /// A site welded to the rest of the chain by strong couplings carries more of the chain than
    /// one hanging off the end by a weak link. With the uniform chain strength
    /// [`crate::embed::apply`] writes, this is degree weighting: an interior site of a path chain
    /// outvotes an endpoint two to one, so a break near an end is decided by the long side rather
    /// than by a count. With unit welds — a [`Chains::new`] built without a machine — it is exactly
    /// [`ChainBreak::Majority`].
    Weighted,
    /// Throw the whole sample away if any chain broke.
    ///
    /// The honest policy and the expensive one: a broken chain means this sample did not answer the
    /// question, and the other variables in it were solved in the presence of that break. Watch the
    /// survival rate — [`Survey::kept`] — because discarding is how a badly tuned chain strength
    /// turns into a quiet loss of statistics rather than a wrong answer.
    Discard,
}

/// The chains of an embedding, with the weld strength holding each site into its chain.
///
/// Chains are stored **sorted by site index**, because a chain is a SET: a resolution that depended
/// on the order the placer happened to emit would make one sample resolve two ways, and both the
/// tie-break and the floating-point order of a weighted vote read that order.
#[derive(Clone, Debug)]
pub struct Chains {
    chains: Vec<Vec<usize>>,
    weld: Vec<f64>,
    sites: usize,
}

impl Chains {
    /// Chains with no machine behind them, so every site welds equally.
    ///
    /// [`ChainBreak::Weighted`] then coincides exactly with [`ChainBreak::Majority`], which is the
    /// right degenerate behaviour: with nothing known about the couplings there is nothing to
    /// weight by, and inventing a weight would be worse than counting.
    ///
    /// # Errors
    ///
    /// [`ChainError::EmptyChain`] or [`ChainError::SiteShared`].
    pub fn new(chains: &[Vec<usize>]) -> Result<Chains, ChainError> {
        let sites = chains.iter().flat_map(|c| c.iter().copied()).max().map_or(0, |m| m + 1);
        Chains::assemble(chains, sites, vec![1.0; sites])
    }

    /// Chains over a machine, welded by the couplings that actually hold them together.
    ///
    /// `hardware` is the graph the sample came from — [`crate::embed::Embedded::graph`], the one
    /// carrying the chain couplings, not the logical model. A site's weld is the total `|J|` on its
    /// edges to other sites of its OWN chain; edges leaving the chain carry the problem and are not
    /// part of what holds the chain together.
    ///
    /// A chain of one site has no internal edges and so weighs zero. That is correct and not a
    /// special case: a one-site chain cannot break, and its weighted vote falls through to the
    /// tie-break, which is that site's own spin.
    ///
    /// # Errors
    ///
    /// [`ChainError::EmptyChain`], [`ChainError::SiteOutOfRange`] or [`ChainError::SiteShared`].
    pub fn from_embedded(chains: &[Vec<usize>], hardware: &Graph) -> Result<Chains, ChainError> {
        let mut owner = vec![usize::MAX; hardware.n];
        for (v, chain) in chains.iter().enumerate() {
            if chain.is_empty() {
                return Err(ChainError::EmptyChain { variable: v });
            }
            for &site in chain {
                if site >= hardware.n {
                    return Err(ChainError::SiteOutOfRange {
                        variable: v,
                        site,
                        sites: hardware.n,
                    });
                }
                if owner[site] != usize::MAX && owner[site] != v {
                    return Err(ChainError::SiteShared { site, first: owner[site], second: v });
                }
                owner[site] = v;
            }
        }
        let mut weld = vec![0.0f64; hardware.n];
        for chain in chains {
            for &u in chain {
                let mut w = 0.0;
                for k in hardware.offset[u]..hardware.offset[u + 1] {
                    if owner[hardware.nbr[k] as usize] == owner[u] {
                        w += hardware.w[k].abs();
                    }
                }
                weld[u] = w;
            }
        }
        Chains::assemble(chains, hardware.n, weld)
    }

    /// The chains of an [`crate::embed::Embedded`] program, welded by its own chain couplings.
    ///
    /// # Errors
    ///
    /// As [`Chains::from_embedded`].
    pub fn of_embedded(e: &crate::embed::Embedded) -> Result<Chains, ChainError> {
        Chains::from_embedded(&e.embedding.chains, &e.graph)
    }

    fn assemble(chains: &[Vec<usize>], sites: usize, weld: Vec<f64>) -> Result<Chains, ChainError> {
        let mut owner = vec![usize::MAX; sites];
        let mut sorted: Vec<Vec<usize>> = Vec::with_capacity(chains.len());
        for (v, chain) in chains.iter().enumerate() {
            if chain.is_empty() {
                return Err(ChainError::EmptyChain { variable: v });
            }
            for &site in chain {
                if site >= sites {
                    return Err(ChainError::SiteOutOfRange { variable: v, site, sites });
                }
                if owner[site] != usize::MAX && owner[site] != v {
                    return Err(ChainError::SiteShared { site, first: owner[site], second: v });
                }
                owner[site] = v;
            }
            let mut c = chain.clone();
            c.sort_unstable();
            c.dedup();
            sorted.push(c);
        }
        Ok(Chains { chains: sorted, weld, sites })
    }

    /// Logical variables.
    #[must_use]
    pub fn len(&self) -> usize {
        self.chains.len()
    }

    /// Are there no chains at all?
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.chains.is_empty()
    }

    /// Sites the machine has.
    #[must_use]
    pub fn sites(&self) -> usize {
        self.sites
    }

    /// The chains, each sorted by site index.
    #[must_use]
    pub fn chains(&self) -> &[Vec<usize>] {
        &self.chains
    }

    /// Per-site weld strength — the total coupling holding that site into its own chain.
    #[must_use]
    pub fn weld(&self) -> &[f64] {
        &self.weld
    }

    /// Read one machine sample back as logical values.
    ///
    /// # The tie-break, which is the whole reason this is written down
    ///
    /// A vote is `sum over the chain of weight × spin`. Positive resolves to `+1`, negative to
    /// `-1`, and **exactly zero resolves to the spin of the chain's lowest-numbered site**.
    ///
    /// A tie is a real event, not a corner case: under [`ChainBreak::Majority`] every chain of even
    /// length that breaks down the middle is one. So the rule has to be fixed, and it has to be a
    /// function of the sample rather than of anything the placer or the allocator did, or two runs
    /// of one seed diverge and the divergence is invisible until someone diffs two result files.
    /// The lowest-numbered site is that: it is a property of the SET of sites, it returns a spin
    /// that was actually measured rather than a constant, and — unlike a constant `+1` — it
    /// commutes with a spin reversal, so [`Chains::resolve_gauged`] agrees with resolving in the
    /// original frame on ties as well as away from them.
    ///
    /// # Errors
    ///
    /// [`ChainError::StateTooShort`] or [`ChainError::NotASpin`].
    pub fn resolve(&self, state: &[i8], policy: ChainBreak) -> Result<Readout, ChainError> {
        if state.len() < self.sites {
            return Err(ChainError::StateTooShort { sites: self.sites, given: state.len() });
        }
        for chain in &self.chains {
            for &site in chain {
                if state[site] != 1 && state[site] != -1 {
                    return Err(ChainError::NotASpin { site, value: state[site] });
                }
            }
        }
        let weighted = matches!(policy, ChainBreak::Weighted);
        let mut values = vec![0i8; self.chains.len()];
        let mut broken = Vec::new();
        for (v, chain) in self.chains.iter().enumerate() {
            let up = chain.iter().filter(|&&site| state[site] > 0).count();
            if up != 0 && up != chain.len() {
                broken.push(v);
            }
            let mut score = 0.0f64;
            for &site in chain {
                let w = if weighted { self.weld[site] } else { 1.0 };
                score += w * f64::from(state[site]);
            }
            values[v] = if score > 0.0 {
                1
            } else if score < 0.0 {
                -1
            } else {
                state[chain[0]]
            };
        }
        let discarded = matches!(policy, ChainBreak::Discard) && !broken.is_empty();
        Ok(Readout {
            values: if discarded { None } else { Some(values) },
            broken,
            chains: self.chains.len(),
            policy,
        })
    }

    /// Read a sample that came off a GAUGED machine: decode first, then resolve.
    ///
    /// The order is the point. A gauge reverses each site independently, so the coupling holding a
    /// chain together is reversed wherever its two ends disagree, and an intact chain comes back
    /// looking split. Resolving before decoding therefore reports breaks in chains that never
    /// broke, and — worse — decides them by a vote over a quantity that is not the variable.
    ///
    /// # Errors
    ///
    /// [`ChainError::GaugeWidth`] if the gauge is not the sample's width, otherwise as
    /// [`Chains::resolve`].
    pub fn resolve_gauged(
        &self,
        gauge: &Gauge,
        state: &[i8],
        policy: ChainBreak,
    ) -> Result<Readout, ChainError> {
        if gauge.n() != state.len() {
            return Err(ChainError::GaugeWidth { gauge: gauge.n(), sites: state.len() });
        }
        let decoded = gauge.map_unchecked(state);
        self.resolve(&decoded, policy)
    }

    /// Read a whole sample set back under one policy.
    ///
    /// # Errors
    ///
    /// As [`Chains::resolve`], on the first sample that fails.
    pub fn resolve_many(
        &self,
        states: &[Vec<i8>],
        policy: ChainBreak,
    ) -> Result<Survey, ChainError> {
        let mut readouts = Vec::with_capacity(states.len());
        for s in states {
            readouts.push(self.resolve(s, policy)?);
        }
        Ok(Survey { readouts, chains: self.chains.len() })
    }
}

/// One sample, read back through the chains.
#[derive(Clone, Debug)]
pub struct Readout {
    values: Option<Vec<i8>>,
    broken: Vec<usize>,
    chains: usize,
    policy: ChainBreak,
}

impl Readout {
    /// The logical values, or `None` if [`ChainBreak::Discard`] threw the sample away.
    #[must_use]
    pub fn values(&self) -> Option<&[i8]> {
        self.values.as_deref()
    }

    /// The variables whose chains disagreed with themselves, ascending.
    #[must_use]
    pub fn broken(&self) -> &[usize] {
        &self.broken
    }

    /// How many chains broke.
    #[must_use]
    pub fn n_broken(&self) -> usize {
        self.broken.len()
    }

    /// **The chain-break fraction**: broken chains over all chains.
    ///
    /// The number practitioners read first, because it is the one that says whether the chain
    /// strength was right. It counts unbreakable one-site chains in the denominator, matching what
    /// `dimod` reports, so the figure is comparable with a tool chain that already exists rather
    /// than a better figure nobody can compare to. Zero chains reads as `0.0`.
    #[must_use]
    pub fn break_fraction(&self) -> f64 {
        if self.chains == 0 {
            return 0.0;
        }
        self.broken.len() as f64 / self.chains as f64
    }

    /// Was this sample thrown away?
    #[must_use]
    pub fn is_discarded(&self) -> bool {
        self.values.is_none()
    }

    /// The policy that produced it.
    #[must_use]
    pub fn policy(&self) -> ChainBreak {
        self.policy
    }
}

/// A sample set read back through the chains.
#[derive(Clone, Debug)]
pub struct Survey {
    readouts: Vec<Readout>,
    chains: usize,
}

impl Survey {
    /// The samples, in the order given.
    #[must_use]
    pub fn readouts(&self) -> &[Readout] {
        &self.readouts
    }

    /// How many samples were read.
    #[must_use]
    pub fn samples(&self) -> usize {
        self.readouts.len()
    }

    /// How many survived the policy. Equal to [`Survey::samples`] for every policy except
    /// [`ChainBreak::Discard`].
    #[must_use]
    pub fn kept(&self) -> usize {
        self.readouts.iter().filter(|r| !r.is_discarded()).count()
    }

    /// Broken (sample, chain) pairs over the whole set.
    #[must_use]
    pub fn broken_chains(&self) -> usize {
        self.readouts.iter().map(Readout::n_broken).sum()
    }

    /// **The chain-break fraction over the set**: broken pairs over `samples × chains`.
    ///
    /// Every sample here carries the same chains, so this is identical to the mean of the
    /// per-sample fractions and there is no pooling ambiguity to argue about. An empty set reads as
    /// `0.0`.
    #[must_use]
    pub fn break_fraction(&self) -> f64 {
        let pairs = self.readouts.len() * self.chains;
        if pairs == 0 {
            return 0.0;
        }
        self.broken_chains() as f64 / pairs as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exact::Elimination;

    /// A frustrated random model with fields, built deterministically from a seed.
    fn fixture(n: usize, seed: u64) -> Graph {
        let mut r = Pcg::new(seed, 11);
        let mut b = GraphBuilder::new(n);
        for i in 0..n {
            for j in (i + 1)..n {
                if r.f64() < 0.45 {
                    b.couple(i, j, (r.f64() - 0.5) * 2.0);
                }
            }
            b.bias(i, r.f64() - 0.5);
        }
        b.build()
    }

    /// Every state of an `n`-spin model, `n` small.
    fn all_states(n: usize) -> impl Iterator<Item = Vec<i8>> {
        (0u32..(1u32 << n))
            .map(move |m| (0..n).map(|i| if (m >> i) & 1 == 1 { 1i8 } else { -1 }).collect())
    }

    /// Greedy descent from all-up: the ordinary biased solver, and deterministic.
    fn greedy_from_all_up(g: &Graph) -> Vec<i8> {
        let mut s = vec![1i8; g.n];
        loop {
            let mut moved = false;
            for i in 0..g.n {
                // Flipping i changes the energy by 2 s_i (sum_j J_ij s_j + h_i); take it when that
                // is negative.
                if f64::from(s[i]) * g.field(i, &s) < 0.0 {
                    s[i] = -s[i];
                    moved = true;
                }
            }
            if !moved {
                return s;
            }
        }
    }

    /// A six-site machine: a three-site path chain, a lone site, and a two-site chain, with the
    /// middle of the path welded a hundred times harder than its end.
    fn welded_machine() -> Graph {
        let mut b = GraphBuilder::new(6);
        b.couple(0, 1, 0.1);
        b.couple(1, 2, 10.0);
        b.couple(4, 5, 2.0);
        // A coupling that LEAVES a chain: it is the problem, not the weld, and must not be counted.
        b.couple(2, 3, 7.0);
        b.build()
    }

    /// A four-site path chain whose middle coupling is a hundred times its ends.
    fn lopsided_machine() -> Graph {
        let mut b = GraphBuilder::new(4);
        b.couple(0, 1, 0.1);
        b.couple(1, 2, 10.0);
        b.couple(2, 3, 0.1);
        b.build()
    }

    /// ORACLE: exhaustive enumeration over all 16,384 states, and exact floating point.
    ///
    /// Gauge invariance is not an approximation — every term of the energy is multiplied by
    /// `g_i^2 g_j^2 = 1`, and multiplying an `f64` by `-1` is exact — so a tolerance here would
    /// hide the only defects that can occur: a dropped sign factor, a field left un-gauged, an edge
    /// gauged from one end. Two claims, and the second does not follow from the first: state by
    /// state the energies AGREE, and as multisets the two spectra ARE the same.
    #[test]
    fn the_spectrum_is_invariant_under_a_gauge_over_every_state_exactly() {
        let n = 14;
        let g = fixture(n, 20260911);
        let gauge = Gauge::random(n, 5, 3);

        // A gauge that reverses nothing, or a model whose couplings all sit on unreversed pairs,
        // would pass on an implementation that does nothing at all.
        assert!(gauge.flips() > 0 && gauge.flips() < n, "the gauge must be non-trivial");
        let gauged = gauge.apply(&g).unwrap();
        let flipped = (0..gauged.w.len()).filter(|&k| gauged.w[k] != g.w[k]).count();
        assert!(flipped > 0, "no coupling changed sign, so this asserts nothing");
        let fields = (0..n).filter(|&i| gauged.h[i] != g.h[i]).count();
        assert!(fields > 0, "no field changed sign, so this asserts nothing");

        let mut original = Vec::with_capacity(1 << n);
        let mut moved = Vec::with_capacity(1 << n);
        for s in all_states(n) {
            let e = g.energy(&s);
            let e_gauged = gauged.energy(&gauge.encode(&s).unwrap());
            assert_eq!(e, e_gauged, "state {s:?} moved in energy under a change of variable");
            original.push(e);
            moved.push(e_gauged);
        }
        assert_eq!(original.len(), 1 << n);
        original.sort_by(f64::total_cmp);
        moved.sort_by(f64::total_cmp);
        assert_eq!(original, moved, "the two spectra differ as multisets");
    }

    /// ORACLE: exhaustive enumeration. The map is an involution, so it is its own inverse on every
    /// one of the 16,384 states — and a solution read back through it is the solution that was sent.
    #[test]
    fn every_state_round_trips_through_the_gauge_and_back() {
        let n = 14;
        let gauge = Gauge::random(n, 77, 1);
        assert_eq!(gauge.inverse(), gauge, "a spin reversal is its own inverse");
        for s in all_states(n) {
            let there = gauge.encode(&s).unwrap();
            assert_eq!(gauge.decode(&there).unwrap(), s, "round trip lost {s:?}");
            // encode and decode are one map; if they ever diverge, this is what notices.
            assert_eq!(gauge.encode(&s).unwrap(), gauge.decode(&s).unwrap());
        }
    }

    /// Applying a gauge twice must give the model back BIT for bit, not merely close: the whole
    /// claim is that this is a change of variable, and a model that came back one ulp different
    /// would be a different model for every certificate in this crate.
    #[test]
    fn applying_a_gauge_twice_restores_the_model_bit_for_bit() {
        let g = fixture(12, 4242);
        let gauge = Gauge::random(12, 9, 2);
        let there = gauge.apply(&g).unwrap();
        let back = gauge.inverse().apply(&there).unwrap();
        assert_eq!(back.n, g.n);
        assert_eq!(back.n_edges, g.n_edges);
        assert_eq!(back.offset, g.offset);
        assert_eq!(back.nbr, g.nbr);
        assert_eq!(back.colors, g.colors, "structure must not move; only signs do");
        let w_bits: Vec<u64> = back.w.iter().map(|x| x.to_bits()).collect();
        let want_w: Vec<u64> = g.w.iter().map(|x| x.to_bits()).collect();
        assert_eq!(w_bits, want_w);
        let h_bits: Vec<u64> = back.h.iter().map(|x| x.to_bits()).collect();
        let want_h: Vec<u64> = g.h.iter().map(|x| x.to_bits()).collect();
        assert_eq!(h_bits, want_h);
    }

    /// ORACLE: `crate::exact` proves the ground state by variable elimination, and exhaustive
    /// enumeration proves the minimum independently of both.
    ///
    /// The gauged model's optimum must be the original's, and the state elimination finds in the
    /// gauged frame must decode to a state attaining it.
    #[test]
    fn the_gauged_optimum_is_the_originals_under_elimination_and_enumeration() {
        let n = 14;
        let g = fixture(n, 31337);
        let gauge = Gauge::random(n, 8, 6);
        let gauged = gauge.apply(&g).unwrap();

        let here = Elimination::default().ground_state(&g).unwrap();
        let there = Elimination::default().ground_state(&gauged).unwrap();
        let e_here = here.ground_energy.unwrap();
        let e_there = there.ground_energy.unwrap();

        let brute = all_states(n).map(|s| g.energy(&s)).fold(f64::INFINITY, f64::min);
        assert!((e_here - brute).abs() < 1e-12, "elimination {e_here} vs enumeration {brute}");
        assert!((e_there - brute).abs() < 1e-12, "a change of variable cannot move the optimum");

        let decoded = gauge.decode(&there.ground_state.unwrap()).unwrap();
        assert_eq!(g.energy(&decoded), brute, "the decoded state must attain it");
    }

    /// ORACLE: `crate::exact`. Handed a solver that is exactly optimal on whatever model it is
    /// given, the runner must report the proved optimum in the ORIGINAL frame under every gauge —
    /// which is what says the decode, the energy bookkeeping and the best-of are all in the right
    /// frame.
    #[test]
    fn average_over_gauges_with_an_exact_solver_returns_the_proved_optimum() {
        let g = fixture(14, 5150);
        let want = Elimination::default().ground_state(&g).unwrap().ground_energy.unwrap();
        let run = average_over_gauges(&g, 8, 99, |model, _seed| {
            Elimination::default().ground_state(model).unwrap().ground_state.unwrap()
        })
        .unwrap();
        assert_eq!(run.energies().len(), 8);
        for (k, &e) in run.energies().iter().enumerate() {
            assert!((e - want).abs() < 1e-12, "gauge {k} came back at {e}, not the optimum {want}");
        }
        assert_eq!(run.best_energy(), run.energies()[0]);
        assert_eq!(run.flips()[0], 0, "the first gauge is the identity");
        assert!(run.flips()[1..].iter().any(|&f| f > 0), "and the rest are not");
        // The certificate agrees with the energy it certifies to within its guard. The DIRECTION
        // claim is made against integers in
        // `the_certificate_is_above_an_energy_counted_exactly_in_integers`, because
        // `Graph::energy` is itself a rounded sum and is the wrong thing to compare a bound to.
        let cert = run.energy_certificate(&g);
        assert!((cert - run.best_energy()).abs() < 1e-12, "certificate {cert} vs energy {want}");
    }

    /// MEASURED, not asserted: what gauging is for. A greedy descent from all-up is biased by its
    /// starting point in exactly the way a machine is biased by its couplers, and a gauge moves
    /// that bias around the problem instead of moving the problem.
    ///
    /// `energies()[0]` is the identity gauge — the same run without any of this — so the comparison
    /// is against a control this function computed itself, and the optimum is proved by
    /// `crate::exact`.
    ///
    /// Measured on this fixture, 32 gauges, one seed:
    ///
    /// ```text
    ///   un-gauged greedy   -9.3056     <- energies()[0], the control
    ///   mean over gauges  -10.3074
    ///   best over gauges  -12.5288     <- the proved optimum, to 2e-15
    ///   proved optimum    -12.5288
    /// ```
    ///
    /// The last two agree to 1.8e-15 and not exactly, for the reason
    /// `the_certificate_is_above_an_energy_counted_exactly_in_integers` documents: one is
    /// `Graph::energy` of a state and the other is an elimination's accumulation, and both are
    /// rounded sums of the same terms in different orders.
    #[test]
    fn average_over_gauges_beats_the_identity_run_on_a_biased_solver() {
        let g = fixture(14, 909);
        let optimum = Elimination::default().ground_state(&g).unwrap().ground_energy.unwrap();
        let run =
            average_over_gauges(&g, 32, 2026, |model, _seed| greedy_from_all_up(model)).unwrap();
        let plain = run.energies()[0];
        assert_eq!(plain, g.energy(&greedy_from_all_up(&g)), "gauge 0 must be the plain run");
        assert!(plain > optimum + 1e-9, "the fixture must actually trap the biased solver");
        assert!(
            run.best_energy() < plain,
            "32 gauges ({}) did not beat the un-gauged run ({plain})",
            run.best_energy()
        );
        assert!(run.mean_energy() <= plain, "and the average is no worse than the control");
        assert!(
            (run.best_energy() - optimum).abs() < 1e-12,
            "and on this instance the gauges reach the proved optimum: {} vs {optimum}",
            run.best_energy()
        );
        // The magnetisation is an average over gauges and must stay inside the spin range.
        assert_eq!(run.magnetization().len(), g.n);
        assert!(run.magnetization().iter().all(|m| (-1.0..=1.0).contains(m)));
    }

    /// THE DEGENERATE CASE: a chain that agrees with itself has an answer, and no policy may
    /// disagree with it or report a break.
    #[test]
    fn an_unbroken_chain_resolves_to_its_own_value_under_every_policy() {
        let chains = vec![vec![0, 1, 2], vec![3], vec![4, 5]];
        let machine = welded_machine();
        let plain = Chains::new(&chains).unwrap();
        let welded = Chains::from_embedded(&chains, &machine).unwrap();
        for bits in 0u32..8 {
            let want: Vec<i8> =
                (0..3).map(|v| if (bits >> v) & 1 == 1 { 1i8 } else { -1 }).collect();
            let mut state = vec![0i8; 6];
            for (v, chain) in chains.iter().enumerate() {
                for &site in chain {
                    state[site] = want[v];
                }
            }
            for c in [&plain, &welded] {
                for policy in [ChainBreak::Majority, ChainBreak::Weighted, ChainBreak::Discard] {
                    let r = c.resolve(&state, policy).unwrap();
                    assert_eq!(r.values(), Some(&want[..]), "{policy:?} moved an intact chain");
                    assert!(r.broken().is_empty());
                    assert_eq!(r.break_fraction(), 0.0);
                    assert!(!r.is_discarded());
                    assert_eq!(r.policy(), policy);
                }
            }
        }
    }

    /// THE TIE-BREAK, PINNED. An undocumented tie-break is how two runs of one seed diverge, so the
    /// rule is asserted directly: a tie resolves to the spin of the chain's LOWEST-NUMBERED site,
    /// whatever order the chain was handed over in.
    #[test]
    fn a_tie_resolves_to_the_lowest_numbered_sites_spin() {
        // Sites 2 and 5 up, 7 and 9 down: an exact two-two tie. The chain is given out of order on
        // purpose -- a chain is a set, and the answer must not depend on the placer's bookkeeping.
        let orders = [vec![5, 2, 9, 7], vec![2, 5, 7, 9], vec![9, 7, 5, 2], vec![7, 9, 2, 5]];
        let mut state = vec![1i8; 10];
        state[7] = -1;
        state[9] = -1;
        for order in &orders {
            let c = Chains::new(std::slice::from_ref(order)).unwrap();
            assert_eq!(c.chains()[0], vec![2, 5, 7, 9], "chains are stored sorted");
            for policy in [ChainBreak::Majority, ChainBreak::Weighted] {
                let r = c.resolve(&state, policy).unwrap();
                assert_eq!(r.values(), Some(&[1i8][..]), "{policy:?}: site 2 is up, so the tie is");
                assert_eq!(r.broken(), &[0]);
                assert_eq!(r.break_fraction(), 1.0);
            }
        }
        // And the rule follows the data rather than a constant: flip the sample and the tie goes
        // the other way.
        let mut other = vec![-1i8; 10];
        other[7] = 1;
        other[9] = 1;
        let c = Chains::new(&[vec![2, 5, 7, 9]]).unwrap();
        assert_eq!(c.resolve(&other, ChainBreak::Majority).unwrap().values(), Some(&[-1i8][..]));
    }

    /// The weighted vote must actually be a different policy, or it is a slower majority vote.
    ///
    /// The chain is a path `0-1-2-3` whose middle coupling is a hundred times its ends: the two end
    /// sites agree with each other, the two hard-welded middle sites disagree with both, and the
    /// count says one thing while the coupling says the other.
    #[test]
    fn the_weighted_vote_follows_the_weld_where_the_majority_follows_the_count() {
        let machine = lopsided_machine();
        let chains = vec![vec![0, 1, 2, 3]];
        let welded = Chains::from_embedded(&chains, &machine).unwrap();
        let plain = Chains::new(&chains).unwrap();
        assert_eq!(welded.weld(), &[0.1, 10.1, 10.1, 0.1]);
        let state = vec![1i8, -1, -1, 1];
        assert_eq!(
            plain.resolve(&state, ChainBreak::Majority).unwrap().values(),
            Some(&[1i8][..]),
            "a two-two tie goes to site 0, which is up"
        );
        assert_eq!(
            welded.resolve(&state, ChainBreak::Weighted).unwrap().values(),
            Some(&[-1i8][..]),
            "the weld is a hundred to one the other way"
        );
        // With no machine behind them the two policies are the same policy, on every state.
        for s in all_states(4) {
            let a = plain.resolve(&s, ChainBreak::Majority).unwrap();
            let b = plain.resolve(&s, ChainBreak::Weighted).unwrap();
            assert_eq!(a.values(), b.values(), "unit welds must degenerate to a count: {s:?}");
        }
    }

    /// Discard drops exactly the samples that broke, and the fractions are exact rationals.
    #[test]
    fn discard_drops_exactly_the_samples_that_broke_and_the_fractions_are_exact() {
        let chains = vec![vec![0, 1], vec![2, 3], vec![4, 5], vec![6, 7]];
        let c = Chains::new(&chains).unwrap();
        let intact = vec![1i8, 1, -1, -1, 1, 1, -1, -1];
        let one_broken = vec![1i8, -1, -1, -1, 1, 1, -1, -1];
        let two_broken = vec![1i8, -1, -1, 1, 1, 1, -1, -1];
        let states = vec![intact.clone(), one_broken.clone(), two_broken];

        let r = c.resolve(&one_broken, ChainBreak::Majority).unwrap();
        assert_eq!(r.broken(), &[0]);
        assert_eq!(r.break_fraction(), 0.25, "one chain of four");
        assert!(!r.is_discarded(), "majority resolves, it does not discard");

        let d = c.resolve(&one_broken, ChainBreak::Discard).unwrap();
        assert!(d.is_discarded() && d.values().is_none());
        assert_eq!(d.break_fraction(), 0.25, "a discarded sample still reports its breaks");
        assert!(!c.resolve(&intact, ChainBreak::Discard).unwrap().is_discarded());

        let survey = c.resolve_many(&states, ChainBreak::Discard).unwrap();
        assert_eq!(survey.samples(), 3);
        assert_eq!(survey.kept(), 1, "only the intact sample survives");
        assert_eq!(survey.broken_chains(), 3, "0 + 1 + 2");
        assert_eq!(survey.break_fraction(), 0.25, "3 broken of 3 x 4 pairs");
        let kept = c.resolve_many(&states, ChainBreak::Majority).unwrap();
        assert_eq!(kept.kept(), 3, "every other policy keeps everything");
        assert_eq!(kept.break_fraction(), survey.break_fraction(), "and reports the same breaks");
        assert_eq!(kept.readouts().len(), 3);
    }

    /// DECODE, THEN RESOLVE -- and the wrong order is measurably wrong, not theoretically wrong.
    ///
    /// Exhaustive over all 64 states of a six-site machine. Resolving a gauged sample after
    /// decoding must reproduce, exactly, what resolving the original-frame sample gives -- values,
    /// broken set and fraction. Resolving the raw gauged sample instead disagrees on a counted
    /// number of those states, and the count is pinned: if it ever became zero this test would be
    /// asserting that the order does not matter, which is the opposite of the rule it exists for.
    #[test]
    fn resolving_before_decoding_reports_breaks_that_are_not_there() {
        let chains = vec![vec![0, 1, 2], vec![3, 4, 5]];
        let c = Chains::new(&chains).unwrap();
        // Reverses one site of each chain, which is what makes an intact chain look broken.
        let gauge = Gauge::from_signs(&[1, -1, 1, 1, 1, -1]).unwrap();
        let mut disagreements = 0;
        for s in all_states(6) {
            let gauged_sample = gauge.encode(&s).unwrap();
            let want = c.resolve(&s, ChainBreak::Majority).unwrap();
            let right = c.resolve_gauged(&gauge, &gauged_sample, ChainBreak::Majority).unwrap();
            assert_eq!(right.values(), want.values(), "decode-then-resolve moved a value");
            assert_eq!(right.broken(), want.broken(), "decode-then-resolve moved a break");
            assert_eq!(right.break_fraction(), want.break_fraction());
            let wrong = c.resolve(&gauged_sample, ChainBreak::Majority).unwrap();
            if wrong.values() != want.values() || wrong.broken() != want.broken() {
                disagreements += 1;
            }
        }
        assert_eq!(disagreements, PINNED_WRONG_ORDER, "the order must still matter");
    }

    /// States of the six-site fixture on which resolving before decoding gives a different answer.
    /// Measured, then pinned: **all 64 of them**, because this gauge reverses one site of each
    /// chain, so in the raw gauged frame every chain always reads as split — including the ones
    /// that are perfectly intact. The wrong order is not a corner case here, it is wrong on every
    /// sample, and a reader who took that as "resolve, then decode, it commutes" would report a
    /// hundred per cent break rate on a machine with none.
    const PINNED_WRONG_ORDER: usize = 64;

    /// A gauge is reproducible from its seed, independent across streams, and balanced.
    #[test]
    fn a_random_gauge_is_seeded_reproducible_and_balanced() {
        assert_eq!(Gauge::random(64, 7, 1), Gauge::random(64, 7, 1));
        assert_ne!(Gauge::random(64, 7, 1), Gauge::random(64, 7, 2));
        assert_ne!(Gauge::random(64, 8, 1), Gauge::random(64, 7, 1));
        assert_eq!(Gauge::identity(9).flips(), 0);
        assert!(!Gauge::identity(9).is_empty());
        assert!(Gauge::identity(0).is_empty());
        assert_eq!(Gauge::identity(4).signs(), &[1, 1, 1, 1]);
        // Four standard deviations of a Binomial(n, 1/2) is 2 sqrt(n): this fails on a stuck sign,
        // not on bad luck.
        let n = 4000;
        let flips = Gauge::random(n, 12345, 0).flips() as f64;
        assert!((flips - n as f64 / 2.0).abs() < 2.0 * (n as f64).sqrt(), "{flips} of {n}");
    }

    /// A model whose couplings and fields are EIGHTHS, so every energy is an exact multiple of
    /// 1/8 and `8E` can be counted in `i64` with no floating point anywhere. Returned with the
    /// numerators, which are the oracle.
    fn eighths(n: usize, seed: u64) -> (Graph, Vec<(usize, usize, i64)>, Vec<i64>) {
        let mut r = Pcg::new(seed, 3);
        let mut b = GraphBuilder::new(n);
        let mut edges = Vec::new();
        let mut fields = vec![0i64; n];
        for i in 0..n {
            for j in (i + 1)..n {
                if r.f64() < 0.5 {
                    let k = i64::from(r.next_u32() % 33) - 16;
                    if k != 0 {
                        edges.push((i, j, k));
                        b.couple(i, j, k as f64 / 8.0);
                    }
                }
            }
            let k = i64::from(r.next_u32() % 17) - 8;
            fields[i] = k;
            b.bias(i, k as f64 / 8.0);
        }
        (b.build(), edges, fields)
    }

    /// ORACLE: the energy counted in `i64`, with no floating point in the count at all.
    ///
    /// Two claims against it, both exact. The certificate from [`energy_upper_bound`] is never
    /// below the true energy — that is what makes it a certificate — and the gauged model's energy
    /// at the encoded state IS the original's, as an integer.
    ///
    /// # Why the certificate is compared against integers and not against `Graph::energy`
    ///
    /// [`energy_upper_bound`] bounds the EXACT energy, and `Graph::energy` is not exact: it is a
    /// round-to-nearest accumulation whose error follows the arithmetic, while the compensated
    /// guard follows the answer. Measured on the 1,024 states of `fixture(10, 6060)`,
    /// `Graph::energy` sits up to **1.33e-15 above** the certificate — sixty ulps of the answer,
    /// on the near-cancelling states where the answer is small. Asserting `cert >= g.energy(s)`
    /// therefore fails, and it fails because the assertion is wrong, not because the bound is: the
    /// certificate is on the right side of the true value and the comparison was against a rounded
    /// one. The integer fixture removes the rounding from the oracle entirely.
    #[test]
    fn the_certificate_is_above_an_energy_counted_exactly_in_integers() {
        let n = 10;
        let (g, edges, fields) = eighths(n, 6060);
        let gauge = Gauge::random(n, 4, 4);
        assert!(gauge.flips() > 0, "a trivial gauge would test nothing");
        let gauged = gauge.apply(&g).unwrap();
        for s in all_states(n) {
            // 8E = - sum k_ij s_i s_j - sum k_i s_i, counted in i64.
            let mut e8 = 0i64;
            for &(i, j, k) in &edges {
                e8 -= k * i64::from(s[i]) * i64::from(s[j]);
            }
            for i in 0..n {
                e8 -= fields[i] * i64::from(s[i]);
            }
            let exact = e8 as f64 / 8.0;
            assert_eq!(g.energy(&s), exact, "the fixture must be exactly representable");
            assert_eq!(
                gauged.energy(&gauge.encode(&s).unwrap()),
                exact,
                "the gauged model must agree with the integer count too"
            );
            let cert = energy_upper_bound(&g, &s);
            assert!(cert >= exact, "certificate {cert} below the true energy {exact}");
            assert!(cert - exact < 1e-12, "and must stay tight: {cert} vs {exact}");
        }
    }

    /// Every refusal names the first defect, and each is a case that would otherwise resolve to a
    /// silently wrong answer.
    #[test]
    fn every_refusal_names_the_first_defect() {
        assert_eq!(
            Gauge::from_signs(&[1, -1, 0]).unwrap_err(),
            GaugeError::NotASign { index: 2, value: 0 }
        );
        let g = fixture(6, 1);
        // `.err()` rather than `unwrap_err`: the Ok side is a Graph, which is not Debug.
        assert_eq!(
            Gauge::identity(5).apply(&g).err().unwrap(),
            GaugeError::WrongWidth { gauge: 5, model: 6 }
        );
        assert_eq!(
            Gauge::identity(6).encode(&[1, 1]).unwrap_err(),
            GaugeError::WrongWidth { gauge: 6, model: 2 }
        );
        assert_eq!(
            average_over_gauges(&g, 0, 1, |_, _| vec![1; 6]).unwrap_err(),
            GaugeError::NoGauges
        );
        assert_eq!(
            average_over_gauges(&g, 2, 1, |_, _| vec![1; 5]).unwrap_err(),
            GaugeError::SolverWidth { expected: 6, got: 5 }
        );

        assert_eq!(
            Chains::new(&[vec![0, 1], Vec::new()]).unwrap_err(),
            ChainError::EmptyChain { variable: 1 }
        );
        assert_eq!(
            Chains::new(&[vec![0, 1], vec![1, 2]]).unwrap_err(),
            ChainError::SiteShared { site: 1, first: 0, second: 1 }
        );
        assert_eq!(
            Chains::from_embedded(&[vec![0, 9]], &lopsided_machine()).unwrap_err(),
            ChainError::SiteOutOfRange { variable: 0, site: 9, sites: 4 }
        );
        let c = Chains::new(&[vec![0, 1], vec![2, 3]]).unwrap();
        assert_eq!(
            c.resolve(&[1, 1, 1], ChainBreak::Majority).unwrap_err(),
            ChainError::StateTooShort { sites: 4, given: 3 }
        );
        assert_eq!(
            c.resolve(&[1, 1, 0, 1], ChainBreak::Majority).unwrap_err(),
            ChainError::NotASpin { site: 2, value: 0 }
        );
        assert_eq!(
            c.resolve_gauged(&Gauge::identity(3), &[1, 1, 1, 1], ChainBreak::Majority).unwrap_err(),
            ChainError::GaugeWidth { gauge: 3, sites: 4 }
        );
        // Every message must say something a reader can act on.
        assert!(format!("{}", GaugeError::NoGauges).contains("identity"));
        assert!(format!("{}", ChainError::EmptyChain { variable: 3 }).contains("variable 3"));
    }

    /// The welds of a real embedded program are the chain couplings that program actually wrote --
    /// checked against [`crate::embed::Embedded::chain_strength`] and the hardware degree inside
    /// each chain, not against this module's own arithmetic.
    #[test]
    fn a_real_embeddings_welds_are_its_own_chain_couplings() {
        let logical = {
            let mut b = GraphBuilder::new(5);
            for i in 0..5 {
                for j in (i + 1)..5 {
                    b.couple(i, j, 1.0);
                }
            }
            b.build()
        };
        let hardware = crate::embed::topology::king(6);
        let e = crate::embed::embed(&logical, &hardware, 7).expect("K5 fits a 6x6 king's graph");
        let program = crate::embed::apply(&logical, &hardware, &e);
        let chains = Chains::of_embedded(&program).unwrap();
        assert_eq!(chains.len(), 5);
        assert!(!chains.is_empty());
        assert_eq!(chains.sites(), hardware.n);

        let mut owner = vec![usize::MAX; hardware.n];
        for (v, chain) in chains.chains().iter().enumerate() {
            for &s in chain {
                owner[s] = v;
            }
        }
        for (v, chain) in chains.chains().iter().enumerate() {
            for &u in chain {
                let inside = (hardware.offset[u]..hardware.offset[u + 1])
                    .filter(|&k| owner[hardware.nbr[k] as usize] == v)
                    .count();
                let want = inside as f64 * program.chain_strength;
                assert!(
                    (chains.weld()[u] - want).abs() < 1e-12,
                    "site {u} of variable {v}: weld {} for {inside} chain edges at {}",
                    chains.weld()[u],
                    program.chain_strength
                );
            }
        }
        // An intact readout of that program gives back a logical state of the right width.
        let state = vec![1i8; hardware.n];
        let r = chains.resolve(&state, ChainBreak::Majority).unwrap();
        assert_eq!(r.values(), Some(&[1i8, 1, 1, 1, 1][..]));
        assert_eq!(r.break_fraction(), 0.0);
    }
}
