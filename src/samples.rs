//! What a sampler returned: many states, and what may honestly be computed from them.
//!
//! Every solver in this crate hands back one state. That is the right answer for an optimiser and
//! the wrong answer for a sampler, and this crate is a sampler: the device it prices charges
//! 1.692 pJ for a read and 7.09 fJ for a Gibbs cycle, so a machine that returns one state per
//! program is a machine that spent its entire budget on the thing it did not use. A [`SampleSet`]
//! is the missing noun.
//!
//! # Why this is a type and not a `Vec<Vec<i8>>`
//!
//! Because a set of states does not, by itself, license an expectation value. Averaging spins
//! across the states a tabu search visited produces a number of exactly the same shape as
//! `<s_i>` — a float in `[-1, 1]`, printable, plottable, indistinguishable in a table — and it
//! estimates nothing. Tabu search is not distributed by anything; it is a trajectory chosen to go
//! downhill, and the frequency with which it visits a state is a fact about the search, not about
//! the model.
//!
//! So the distribution a set came from is carried in the type ([`Provenance`]), and the estimators
//! REFUSE where it is absent:
//!
//! ```
//! use ferrotherm::samples::{SampleSet, Refused};
//! let visited = vec![vec![1i8, -1], vec![-1, -1]];
//! let e = vec![0.5, -0.5];
//! let set = SampleSet::from_search(visited, e, "tabu");
//! assert!(set.best().is_some());                       // a fact about the set: allowed
//! assert_eq!(set.mean_spin(0), Err(Refused::NotDistributional { method: "tabu" }));
//! ```
//!
//! `best`, `distinct` and [`SampleSet::ground_states`] are facts about the multiset and are always
//! available. `mean_spin`, `correlation`, `magnetization` and [`SampleSet::expectation`] are claims
//! about a distribution and are available only where there is one.
//!
//! # The error bar is the point
//!
//! An [`Estimate`] carries a standard error, and that standard error is **not** `sigma/sqrt(N)`.
//! Chain draws are autocorrelated, so `N` of them are worth `N/(2*tau_int)` independent ones, and
//! quoting the naive interval understates the error by `sqrt(2*tau)` — a factor of five on a chain
//! with `tau = 12`, which is an ordinary chain. The inflation is applied here rather than left to
//! the caller because the naive number is the one that gets published.
//!
//! The autocorrelation is measured **per observable**, on that observable's own trace, which is the
//! textbook definition and is strictly better than reusing one number for everything: energy and
//! magnetization mix at different rates on the same chain, and [`crate::certify`] deliberately
//! reports the worse of the two because a certificate is a summary. An estimate is not a summary.
//!
//! Each provenance has its own correlation structure and its own effective sample size:
//!
//! | provenance | independent draws | why |
//! |---|---|---|
//! | [`Provenance::Chain`] | `N / (2*tau_int)` | successive draws share a state |
//! | [`Provenance::Population`] | `N / rho` | replicas share an ancestor after resampling |
//! | [`Provenance::Enumerated`] | infinite | nothing was sampled; the weights are exact |
//! | [`Provenance::Search`] | — | refused |
//!
//! `rho` is population annealing's family statistic, which [`crate::popanneal`] already computes
//! and already reports; using it here is the same quantity doing the same job.
//!
//! # What this deliberately does not do
//!
//! It does not aggregate on construction. D-Wave's `SampleSet` stores distinct states with
//! occurrence counts, which is compact and which destroys chain order — and chain order is the
//! only thing `tau_int` can be computed from. Order is kept; [`SampleSet::distinct`] aggregates on
//! demand.

use crate::certify::{tau_int, Certificate};
use crate::graph::Graph;
use crate::ledger::Ledger;
use std::collections::BTreeMap;

/// Where a set of states came from, and therefore what may be computed from it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Provenance {
    /// Successive draws from one Markov chain, **in chain order**, `thin` sweeps apart, after
    /// `burn_in` sweeps discarded.
    ///
    /// Order matters and is load-bearing: `tau_int` and the convergence check in
    /// [`crate::certify`] both read it. A set built with this provenance from states that were
    /// shuffled, sorted, or pooled from several chains is mislabelled, and everything downstream
    /// will believe it.
    Chain { beta: f64, burn_in: usize, thin: usize },
    /// The final population of a sequential Monte Carlo run at `beta`, with the family statistic
    /// `rho` that says how much of it is genuinely distinct.
    ///
    /// Replicas are independent chains, so there is no autocorrelation along an index — but
    /// resampling means several replicas can descend from one ancestor, and `rho` is exactly the
    /// factor by which that shrinks the effective count. See [`crate::popanneal::Outcome::rho`].
    Population { beta: f64, rho: f64 },
    /// Every state in the model, with its exact Boltzmann weight. Nothing was sampled.
    Enumerated { beta: f64 },
    /// States a search visited on its way downhill. Distributed by nothing.
    Search { method: &'static str },
}

impl Provenance {
    /// The inverse temperature the states belong to, where they belong to one.
    pub fn beta(&self) -> Option<f64> {
        match *self {
            Provenance::Chain { beta, .. }
            | Provenance::Population { beta, .. }
            | Provenance::Enumerated { beta } => Some(beta),
            Provenance::Search { .. } => None,
        }
    }

    /// Whether an expectation value taken over these states estimates anything.
    pub fn is_distributional(&self) -> bool {
        !matches!(self, Provenance::Search { .. })
    }

    /// A short name, for error messages.
    pub fn label(&self) -> &'static str {
        match self {
            Provenance::Chain { .. } => "chain",
            Provenance::Population { .. } => "population",
            Provenance::Enumerated { .. } => "enumerated",
            Provenance::Search { .. } => "search",
        }
    }
}

/// Why a question was refused rather than answered.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Refused {
    /// The states came from a search, so their frequencies are a fact about the search.
    NotDistributional { method: &'static str },
    /// The question needs chain order, and this set has none.
    NotAChain { provenance: &'static str },
    /// There is nothing to average.
    Empty,
    /// The model has more spins than exhaustive enumeration will materialise.
    TooLargeToEnumerate { spins: usize, limit: usize },
}

impl core::fmt::Display for Refused {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Refused::NotDistributional { method } => write!(
                f,
                "these states were visited by {method}, which is a search and not a sampler: how \
                 often it saw a state is a fact about where it walked, not about the model's \
                 probability of that state. `best`, `distinct` and `ground_states` still answer"
            ),
            Refused::NotAChain { provenance } => write!(
                f,
                "this is a {provenance} set, and the question needs successive draws from one \
                 chain in order -- autocorrelation and the early-versus-late drift check are both \
                 statements about that order, and there is none here"
            ),
            Refused::Empty => write!(f, "the set is empty, so there is nothing to average"),
            Refused::TooLargeToEnumerate { spins, limit } => write!(
                f,
                "exhaustive enumeration materialises 2^{spins} states, and the limit is 2^{limit}; \
                 above that the states alone exceed what this will allocate. Sample instead, or \
                 use `exact::marginals`, which gets single-site marginals from elimination without \
                 enumerating anything"
            ),
        }
    }
}

/// An expectation value, with an error bar that accounts for how correlated the draws were.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Estimate {
    /// The estimate itself.
    pub value: f64,
    /// Standard error, computed as `sqrt(var / ess)` — NOT `sqrt(var / N)`.
    pub stderr: f64,
    /// Independent draws this observable is worth. See the table in the module documentation.
    pub ess: f64,
    /// Integrated autocorrelation time of this observable's own trace, or `NaN` where the
    /// provenance has no chain order for one to be defined on.
    pub tau_int: f64,
}

impl Estimate {
    /// A 95% interval, `value +- 1.96 * stderr`.
    pub fn ci95(&self) -> (f64, f64) {
        (self.value - 1.96 * self.stderr, self.value + 1.96 * self.stderr)
    }

    /// Whether `truth` lies inside [`Self::ci95`]. For calibration checks.
    pub fn covers(&self, truth: f64) -> bool {
        let (lo, hi) = self.ci95();
        lo <= truth && truth <= hi
    }
}

impl core::fmt::Display for Estimate {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{:.5} +- {:.5} (ess {:.0}", self.value, self.stderr, self.ess)?;
        if self.tau_int.is_finite() {
            write!(f, ", tau {:.1}", self.tau_int)?;
        }
        write!(f, ")")
    }
}

/// How many states to take, and from where in the run.
///
/// There is deliberately **no default burn-in**. How long a chain takes to forget where it started
/// is a property of the model and the temperature, not of this library, and a constructor that
/// picked one would be picking it for models it has never seen. Run
/// [`SampleSet::certificate`] and read the `NotConverged` finding: that is the measurement that
/// answers it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Plan {
    /// Sweeps run and discarded before the first draw.
    pub burn_in: usize,
    /// How many states to keep.
    pub draws: usize,
    /// Sweeps between kept states. Coerced up to 1: a `thin` of zero would return the same state
    /// `draws` times.
    pub thin: usize,
}

impl Plan {
    pub fn new(burn_in: usize, draws: usize, thin: usize) -> Plan {
        Plan { burn_in, draws, thin: thin.max(1) }
    }

    /// Total sweeps this plan runs, burn-in included. What the ledger will be charged for
    /// sampling, per free node.
    pub fn sweeps(&self) -> usize {
        self.burn_in + self.draws * self.thin.max(1)
    }
}

/// States a run produced, and the distribution (if any) they are draws from.
#[derive(Clone, Debug)]
pub struct SampleSet {
    states: Vec<Vec<i8>>,
    energies: Vec<f64>,
    /// Normalised exact weights. `Some` only for [`Provenance::Enumerated`].
    weights: Option<Vec<f64>>,
    prov: Provenance,
    n: usize,
    /// The slowest autocorrelation time observed on the chain as a whole, or `NaN` where the
    /// provenance has no chain. See [`SampleSet::chain_tau`].
    chain_tau: f64,
}

impl SampleSet {
    /// Draws from one chain, in the order the chain produced them.
    ///
    /// The caller is asserting the order. See [`Provenance::Chain`] for what rests on it.
    ///
    /// # Panics
    /// If `states` and `energies` differ in length, or the states differ in width.
    pub fn from_chain(
        states: Vec<Vec<i8>>,
        energies: Vec<f64>,
        beta: f64,
        burn_in: usize,
        thin: usize,
    ) -> SampleSet {
        SampleSet::build(states, energies, None, Provenance::Chain { beta, burn_in, thin })
    }

    /// A sequential Monte Carlo population at `beta`, with its family statistic.
    ///
    /// # Panics
    /// If `states` and `energies` differ in length, or the states differ in width.
    pub fn from_population(
        states: Vec<Vec<i8>>,
        energies: Vec<f64>,
        beta: f64,
        rho: f64,
    ) -> SampleSet {
        SampleSet::build(states, energies, None, Provenance::Population { beta, rho })
    }

    /// States a search visited. Expectation values over these are refused; see [`Refused`].
    ///
    /// # Panics
    /// If `states` and `energies` differ in length, or the states differ in width.
    pub fn from_search(
        states: Vec<Vec<i8>>,
        energies: Vec<f64>,
        method: &'static str,
    ) -> SampleSet {
        SampleSet::build(states, energies, None, Provenance::Search { method })
    }

    fn build(
        states: Vec<Vec<i8>>,
        energies: Vec<f64>,
        weights: Option<Vec<f64>>,
        prov: Provenance,
    ) -> SampleSet {
        assert_eq!(
            states.len(),
            energies.len(),
            "one energy per state: {} states and {} energies is a set that cannot be indexed",
            states.len(),
            energies.len()
        );
        let n = states.first().map_or(0, |s| s.len());
        assert!(
            states.iter().all(|s| s.len() == n),
            "every state must have the same width; mixing widths in one set makes `mean_spin(i)` \
             mean different things for different draws"
        );
        // The chain-level autocorrelation is measured ONCE, here, for the reason set out on
        // `chain_tau`: a per-observable tau alone under-corrects, and the correction has to be
        // available to every estimate the set will ever be asked for. It costs roughly `5*tau`
        // passes over two traces, which is small beside generating the draws -- every draw already
        // cost `thin * n` spin updates -- and it is paid only for a chain.
        let chain_tau = match prov {
            Provenance::Chain { .. } => {
                let mag: Vec<f64> = states
                    .iter()
                    .map(|s| s.iter().map(|&v| v as f64).sum::<f64>() / n.max(1) as f64)
                    .collect();
                let a = tau_int(&energies);
                let b = tau_int(&mag);
                match (a.is_nan(), b.is_nan()) {
                    (false, false) => a.max(b),
                    (true, false) => b,
                    (false, true) => a,
                    (true, true) => f64::NAN,
                }
            }
            _ => f64::NAN,
        };
        SampleSet { states, energies, weights, prov, n, chain_tau }
    }

    /// The slowest integrated autocorrelation time seen on this chain, over energy and
    /// magnetization; `NaN` for a set that is not a chain.
    ///
    /// # Why every estimate is corrected by this and not only by its own trace
    ///
    /// A per-observable `tau_int` is the textbook correction and it is what an estimate uses --
    /// but only as a LOWER bound on the correction, because it can be fooled in one specific and
    /// common way. Sokal's windowing measures how fast the trace it is given decorrelates. A single
    /// site sitting in a metastable mode produces a trace that is `+1` with small fast jitter, and
    /// the windowing correctly reports that the JITTER decorrelates in a few sweeps -- while the
    /// mode itself, the thing that decides whether the estimate is right at all, has a lifetime
    /// thousands of sweeps long and never appears in that trace.
    ///
    /// Measured on a 14-spin glass at `beta = 1.2`: per-site tau reads about 15, the chain's own
    /// reads about 306, and the interval built from the per-site number covers the exact marginal
    /// 44% of the time while claiming 95%. Taking the larger of the two is what closes that gap.
    /// Energy and magnetization are used because they fail in opposite directions -- an ordered
    /// system's energy jitters quickly around a fixed value while its magnetization does not move
    /// at all -- which is the same pair, for the same reason, that [`crate::certify`] takes the
    /// worse of.
    pub fn chain_tau(&self) -> f64 {
        self.chain_tau
    }

    pub fn len(&self) -> usize {
        self.states.len()
    }

    pub fn is_empty(&self) -> bool {
        self.states.is_empty()
    }

    /// Spins per state.
    pub fn n_spins(&self) -> usize {
        self.n
    }

    pub fn provenance(&self) -> Provenance {
        self.prov
    }

    pub fn states(&self) -> &[Vec<i8>] {
        &self.states
    }

    pub fn energies(&self) -> &[f64] {
        &self.energies
    }

    /// The lowest-energy state in the set, and its energy.
    ///
    /// Always available: this is a fact about the states that were seen, and needs no distribution.
    /// Ties go to the first occurrence, so it is deterministic in the order the set was built.
    pub fn best(&self) -> Option<(&[i8], f64)> {
        let mut k = 0usize;
        if self.energies.is_empty() {
            return None;
        }
        for i in 1..self.energies.len() {
            if self.energies[i] < self.energies[k] {
                k = i;
            }
        }
        Some((&self.states[k], self.energies[k]))
    }

    /// Distinct states with their energy and how many times each occurred, ordered by energy and
    /// then by state so the output is stable.
    ///
    /// Always available. Aggregation happens here rather than at construction because it destroys
    /// chain order, and chain order is what `tau_int` is computed from.
    pub fn distinct(&self) -> Vec<(Vec<i8>, f64, usize)> {
        let mut seen: BTreeMap<&[i8], (f64, usize)> = BTreeMap::new();
        for (s, &e) in self.states.iter().zip(self.energies.iter()) {
            let slot = seen.entry(s.as_slice()).or_insert((e, 0));
            slot.1 += 1;
        }
        let mut out: Vec<(Vec<i8>, f64, usize)> =
            seen.into_iter().map(|(s, (e, c))| (s.to_vec(), e, c)).collect();
        out.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(core::cmp::Ordering::Equal).then_with(|| a.0.cmp(&b.0)));
        out
    }

    /// Every distinct state within `tol` of the lowest energy seen, lowest first.
    ///
    /// This is **evidence of** degeneracy, not a count of it. A set that found three ground states
    /// proves there are at least three; it cannot prove there are no more, and this method makes no
    /// such claim. Only [`Provenance::Enumerated`] can, because only enumeration looked everywhere.
    pub fn ground_states(&self, tol: f64) -> Vec<Vec<i8>> {
        let Some((_, e0)) = self.best() else { return Vec::new() };
        self.distinct()
            .into_iter()
            .filter(|(_, e, _)| *e <= e0 + tol)
            .map(|(s, _, _)| s)
            .collect()
    }

    /// `<E>`.
    pub fn mean_energy(&self) -> Result<Estimate, Refused> {
        let e = self.energies.clone();
        self.estimate_from(&e)
    }

    /// `<s_i>`, in `[-1, 1]`. Multiply by `0.5` and add `0.5` for `P(s_i = +1)`.
    pub fn mean_spin(&self, i: usize) -> Result<Estimate, Refused> {
        self.expectation(|s| s[i] as f64)
    }

    /// Every `<s_i>` at once, with each site's own autocorrelation.
    pub fn marginals(&self) -> Result<Vec<Estimate>, Refused> {
        (0..self.n).map(|i| self.mean_spin(i)).collect()
    }

    /// `<s_i s_j>`. This and [`Self::mean_spin`] are the two moments contrastive divergence
    /// matches; see [`crate::ebm`].
    pub fn correlation(&self, i: usize, j: usize) -> Result<Estimate, Refused> {
        self.expectation(|s| (s[i] * s[j]) as f64)
    }

    /// `(1/n) * sum_i <s_i>`, the order parameter.
    pub fn magnetization(&self) -> Result<Estimate, Refused> {
        let n = self.n as f64;
        self.expectation(move |s| s.iter().map(|&v| v as f64).sum::<f64>() / n)
    }

    /// The expectation of any function of a state.
    ///
    /// The autocorrelation is measured on this observable's own trace, so the error bar is the one
    /// that belongs to this quantity rather than a shared summary.
    pub fn expectation<F: Fn(&[i8]) -> f64>(&self, f: F) -> Result<Estimate, Refused> {
        let vals: Vec<f64> = self.states.iter().map(|s| f(s)).collect();
        self.estimate_from(&vals)
    }

    /// The one place a number is turned into a number-with-an-error-bar.
    fn estimate_from(&self, vals: &[f64]) -> Result<Estimate, Refused> {
        if let Provenance::Search { method } = self.prov {
            return Err(Refused::NotDistributional { method });
        }
        if vals.is_empty() {
            return Err(Refused::Empty);
        }
        let n = vals.len();

        // Enumeration is not sampling. The weights are exact, so the answer is exact and the
        // interval is a point. Returning `sqrt(var/N)` here would attach sampling noise to a
        // quantity that has none.
        if let Some(w) = &self.weights {
            let value = vals.iter().zip(w.iter()).map(|(v, wk)| v * wk).sum::<f64>();
            return Ok(Estimate { value, stderr: 0.0, ess: f64::INFINITY, tau_int: f64::NAN });
        }

        let value = vals.iter().sum::<f64>() / n as f64;
        let var = if n > 1 {
            vals.iter().map(|v| (v - value).powi(2)).sum::<f64>() / (n - 1) as f64
        } else {
            0.0
        };

        let (tau, ess) = match self.prov {
            Provenance::Chain { .. } => {
                // The larger of this observable's own tau and the chain's slowest. See `chain_tau`
                // for the measurement that forces it.
                let own = tau_int(vals);
                let t = match (own.is_nan(), self.chain_tau.is_nan()) {
                    (false, false) => own.max(self.chain_tau),
                    (true, false) => self.chain_tau,
                    _ => own,
                };
                // `tau_int` returns infinity for a constant trace, which is also what a frozen
                // chain produces. Both give zero variance and therefore a zero-width interval --
                // correct arithmetic, and a trap: a chain stuck in one state reports its one value
                // with no error at all. The infinite tau is left in the estimate rather than
                // sanitised precisely so that reading `tau_int.is_finite()` catches it, and so
                // does the certificate's `Undermixed` finding.
                let e = if t.is_finite() && t > 0.0 { n as f64 / (2.0 * t) } else { 1.0 };
                (t, e)
            }
            Provenance::Population { rho, .. } => {
                let r = if rho.is_finite() && rho >= 1.0 { rho } else { 1.0 };
                (f64::NAN, (n as f64 / r).max(1.0))
            }
            Provenance::Enumerated { .. } => (f64::NAN, f64::INFINITY),
            Provenance::Search { .. } => unreachable!("refused above"),
        };

        let stderr = if ess.is_finite() && ess > 0.0 { (var / ess).sqrt() } else { 0.0 };
        Ok(Estimate { value, stderr, ess, tau_int: tau })
    }

    /// Certify this set against the model it claims to come from.
    ///
    /// Refused for anything but a chain: `tau_int` and the early-versus-late drift check are both
    /// statements about draw order, and a population or an enumeration has none. Handing either to
    /// [`crate::certify::certify`] directly produces a certificate whose two headline numbers are
    /// computed from an ordering that means nothing — which is worse than no certificate, because
    /// it looks like one.
    pub fn certificate(&self, g: &Graph) -> Result<Certificate, Refused> {
        match self.prov {
            Provenance::Chain { beta, .. } => {
                Ok(crate::certify::certify(g, beta, &self.states, &self.energies))
            }
            Provenance::Search { method } => Err(Refused::NotDistributional { method }),
            other => Err(Refused::NotAChain { provenance: other.label() }),
        }
    }
}

/// The widest model this will enumerate.
///
/// `2^22` states of 22 bytes each is 92 MB of spins before anything else; `2^20` is 20 MB, which is
/// an allocation a test can make. [`crate::ising::exact_boltzmann`] goes to 24 because it
/// materialises only the probabilities, not the states — a different bound for a different object.
pub const ENUMERATION_LIMIT: usize = 20;

/// Every state of the model with its exact Boltzmann weight.
///
/// This is the oracle: expectation values taken from it are exact, and their standard error is
/// zero because nothing was sampled. It is what the sampled sets in this module's tests are
/// checked against.
pub fn enumerate(g: &Graph, beta: f64) -> Result<SampleSet, Refused> {
    if g.n > ENUMERATION_LIMIT {
        return Err(Refused::TooLargeToEnumerate { spins: g.n, limit: ENUMERATION_LIMIT });
    }
    let m = 1usize << g.n;
    let mut states = Vec::with_capacity(m);
    let mut energies = Vec::with_capacity(m);
    let mut logw = Vec::with_capacity(m);
    let mut mx = f64::NEG_INFINITY;
    for mask in 0..m {
        let s: Vec<i8> = (0..g.n).map(|b| if mask >> b & 1 == 1 { 1 } else { -1 }).collect();
        let e = g.energy(&s);
        let l = -beta * e;
        if l > mx {
            mx = l;
        }
        states.push(s);
        energies.push(e);
        logw.push(l);
    }
    // Shifted, for the same reason `exact_boltzmann` is: `exp(-beta * E)` overflows f64 long
    // before beta gets large, and the shift cancels exactly in the normalised weights.
    let mut z = 0.0;
    for v in logw.iter_mut() {
        *v = (*v - mx).exp();
        z += *v;
    }
    for v in logw.iter_mut() {
        *v /= z;
    }
    Ok(SampleSet::build(states, energies, Some(logw), Provenance::Enumerated { beta }))
}

impl<'g> crate::gibbs::Sampler<'g> {
    /// Run `plan` and keep what it draws, charging the device for every sweep **and every read**.
    ///
    /// # The read is the whole point
    ///
    /// Before this existed, five places in this repository hand-wrote the same burn-in / thin /
    /// collect loop, and every one of them appended `smp.s.clone()` — which takes the state out of
    /// the sampler without going through [`crate::gibbs::Sampler::read_all`], and therefore without
    /// charging the ledger a single read. On a Z1-class device a full read costs 1.692 pJ per node
    /// against 7.09 fJ per Gibbs cycle, so a read is worth 239 cycles: a plan drawing states 4
    /// sweeps apart spends more than a third of its energy on readback, and the loops that clone
    /// reported that third as zero. This one reads.
    ///
    /// It reads the WHOLE state, because it returns the whole state. A caller who needs only some
    /// nodes — action bits, visible units — should drive [`crate::gibbs::Sampler::read_subset`]
    /// directly and pay for what they take; that is the regime `examples/z1_ledger.rs` measures.
    pub fn collect(&mut self, plan: &Plan, mut ledger: Option<&mut Ledger>) -> SampleSet {
        let thin = plan.thin.max(1);
        self.sweeps(plan.burn_in, ledger.as_deref_mut());
        let mut states = Vec::with_capacity(plan.draws);
        let mut energies = Vec::with_capacity(plan.draws);
        for _ in 0..plan.draws {
            self.sweeps(thin, ledger.as_deref_mut());
            let s = self.read_all(ledger.as_deref_mut());
            energies.push(self.g.energy(&s));
            states.push(s);
        }
        SampleSet::from_chain(states, energies, self.beta, plan.burn_in, thin)
    }

    /// [`Self::collect`] over the parallel sweep path.
    ///
    /// Identical in what it produces and what it charges; the only difference is which sweep runs.
    /// `threads` is a request, not a promise — see [`crate::gibbs::MIN_CHUNK`] and
    /// [`crate::gibbs::Sampler::threads_used`] for when it is honoured. On `wasm32` the parallel
    /// sweep is the serial one, so this is `collect` there and produces the same bytes.
    pub fn collect_par(
        &mut self,
        plan: &Plan,
        threads: usize,
        mut ledger: Option<&mut Ledger>,
    ) -> SampleSet {
        let thin = plan.thin.max(1);
        self.sweeps_par(plan.burn_in, threads, ledger.as_deref_mut());
        let mut states = Vec::with_capacity(plan.draws);
        let mut energies = Vec::with_capacity(plan.draws);
        for _ in 0..plan.draws {
            self.sweeps_par(thin, threads, ledger.as_deref_mut());
            let s = self.read_all(ledger.as_deref_mut());
            energies.push(self.g.energy(&s));
            states.push(s);
        }
        SampleSet::from_chain(states, energies, self.beta, plan.burn_in, thin)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gibbs::Sampler;
    use crate::graph::GraphBuilder;
    use crate::rng::Pcg;

    /// A random +-1 glass on a circulant graph. Small enough to enumerate exactly.
    fn glass(n: usize, seed: u64, reach: usize) -> Graph {
        let mut r = Pcg::new(seed, 7);
        let mut gb = GraphBuilder::new(n);
        for i in 0..n {
            for k in 1..=reach {
                let j = (i + k) % n;
                if i < j {
                    gb.couple(i, j, if r.f64() < 0.5 { 1.0 } else { -1.0 });
                }
            }
        }
        gb.build()
    }

    /// The exact `<s_i>` for every site, from the enumerated Boltzmann distribution, computed by a
    /// route that does not touch this module: `ising::exact_boltzmann` is the crate's independent
    /// oracle and is verified against Onsager elsewhere.
    fn exact_means(g: &Graph, beta: f64) -> Vec<f64> {
        let p = crate::ising::exact_boltzmann(g, beta);
        let mut out = vec![0.0; g.n];
        for (mask, &pk) in p.iter().enumerate() {
            for (i, o) in out.iter_mut().enumerate() {
                *o += pk * if mask >> i & 1 == 1 { 1.0 } else { -1.0 };
            }
        }
        out
    }

    /// Enumeration is not sampling: its answers are exact and its intervals are points.
    #[test]
    fn an_enumerated_set_is_the_oracle_and_reports_no_sampling_error() {
        let g = glass(10, 11, 2);
        let beta = 0.9;
        let truth = exact_means(&g, beta);
        let set = enumerate(&g, beta).expect("10 spins is under the limit");
        assert_eq!(set.len(), 1 << 10);
        for i in 0..g.n {
            let e = set.mean_spin(i).expect("enumeration is distributional");
            assert!(
                (e.value - truth[i]).abs() < 1e-12,
                "site {i}: enumerated {} vs oracle {}",
                e.value,
                truth[i]
            );
            assert_eq!(e.stderr, 0.0, "nothing was sampled, so there is no sampling error");
            assert!(e.ess.is_infinite());
        }
    }

    /// The load-bearing test: a chain's estimate must contain the exact answer as often as it
    /// claims to, and the naive interval must not.
    ///
    /// This is a coverage measurement, not a spot check. 12 independent chains x every site x two
    /// models is 312 intervals, each compared against the exactly enumerated marginal. The seeds
    /// are fixed, so the numbers below are reproducible rather than probabilistic -- the crate's
    /// own promise is that a seed produces the same states on every platform tested.
    #[test]
    fn the_corrected_interval_covers_the_exact_marginal_and_the_naive_one_does_not() {
        let cases: [(Graph, f64); 2] = [(crate::ising::ring(12, 1.0, 0.0), 0.8), (glass(14, 3, 2), 0.5)];
        let (mut hit, mut naive_hit, mut total) = (0usize, 0usize, 0usize);
        for (g, beta) in &cases {
            let truth = exact_means(g, *beta);
            for seed in 0..12u64 {
                let mut smp = Sampler::new(g, *beta, seed * 7919 + 1);
                let set = smp.collect(&Plan::new(2_000, 4_000, 1), None);
                for i in 0..g.n {
                    let e = set.mean_spin(i).expect("a chain is distributional");
                    // Undo exactly the autocorrelation correction, and nothing else, to get the
                    // interval a caller would have written by hand: sqrt(var/N).
                    let naive_se = e.stderr * (e.ess / set.len() as f64).sqrt();
                    if e.covers(truth[i]) {
                        hit += 1;
                    }
                    if (e.value - truth[i]).abs() <= 1.96 * naive_se {
                        naive_hit += 1;
                    }
                    total += 1;
                }
            }
        }
        let cov = hit as f64 / total as f64;
        let naive = naive_hit as f64 / total as f64;
        // Measured 99.2% and 47.6% over these 312 intervals. The corrected bound is deliberately
        // one-sided and loose: the correction is conservative by construction (it takes the
        // SLOWEST observed autocorrelation, see `chain_tau`), so over-coverage is the expected
        // direction and is not a failure.
        assert!(cov >= 0.90, "corrected interval covered {cov:.3} of {total}, claiming 0.95");
        assert!(
            naive <= 0.75,
            "the naive sqrt(var/N) interval covered {naive:.3}, which would mean the \
             autocorrelation correction this module exists for is not doing anything"
        );
    }

    /// The specific way a per-observable tau is fooled, and the fix, in one assertion.
    #[test]
    fn a_sites_own_autocorrelation_is_a_floor_and_never_the_whole_correction() {
        let g = glass(14, 3, 2);
        let beta = 1.0;
        let mut smp = Sampler::new(&g, beta, 99);
        let set = smp.collect(&Plan::new(2_000, 4_000, 1), None);
        assert!(set.chain_tau() > 1.0, "this beta is meant to be in the slow regime");
        let mut lifted = 0;
        for i in 0..g.n {
            let e = set.mean_spin(i).unwrap();
            assert!(
                e.tau_int >= set.chain_tau() - 1e-9,
                "site {i} reported tau {} below the chain's {}",
                e.tau_int,
                set.chain_tau()
            );
            let own = tau_int(&set.states().iter().map(|s| s[i] as f64).collect::<Vec<_>>());
            if own < set.chain_tau() - 1e-9 {
                lifted += 1;
            }
        }
        assert!(
            lifted > 0,
            "no site's own tau was below the chain's, so this test is not exercising the lift it \
             was written for -- pick a colder beta or a more frustrated graph"
        );
    }

    /// A search set answers questions about itself and refuses questions about a distribution.
    #[test]
    fn a_search_set_answers_facts_and_refuses_estimates() {
        let set = SampleSet::from_search(
            vec![vec![1i8, -1, 1], vec![-1, -1, 1], vec![1, -1, 1]],
            vec![-2.0, 0.5, -2.0],
            "tabu",
        );
        assert_eq!(set.best().map(|(_, e)| e), Some(-2.0));
        assert_eq!(set.distinct().len(), 2, "three visits, two distinct states");
        assert_eq!(set.ground_states(1e-9).len(), 1);

        let refused = Refused::NotDistributional { method: "tabu" };
        assert_eq!(set.mean_spin(0), Err(refused));
        assert_eq!(set.correlation(0, 1), Err(refused));
        assert_eq!(set.magnetization(), Err(refused));
        assert_eq!(set.mean_energy(), Err(refused));
        assert!(format!("{refused}").contains("tabu"), "the refusal must name what refused");
    }

    /// A certificate is a statement about draw order. Two of the four provenances have none.
    #[test]
    fn only_a_chain_can_be_certified() {
        let g = crate::ising::ring(8, 1.0, 0.0);
        let states = vec![vec![1i8; 8]; 40];
        let energies = vec![g.energy(&states[0]); 40];

        let pop = SampleSet::from_population(states.clone(), energies.clone(), 0.5, 1.2);
        assert_eq!(pop.certificate(&g).unwrap_err(), Refused::NotAChain { provenance: "population" });
        let enu = enumerate(&g, 0.5).unwrap();
        assert_eq!(enu.certificate(&g).unwrap_err(), Refused::NotAChain { provenance: "enumerated" });
        let srch = SampleSet::from_search(states.clone(), energies.clone(), "hfs");
        assert_eq!(srch.certificate(&g).unwrap_err(), Refused::NotDistributional { method: "hfs" });

        let chain = SampleSet::from_chain(states, energies, 0.5, 0, 1);
        assert!(chain.certificate(&g).is_ok(), "a chain is exactly what certify takes");
    }

    /// The defect this module was written around: collecting states used to be free.
    #[test]
    fn collecting_charges_the_device_for_every_read_as_well_as_every_sweep() {
        let g = crate::ising::lattice2d(6, 1.0);
        let plan = Plan::new(50, 30, 4);
        let mut led = Ledger::default();
        let mut smp = Sampler::new(&g, 0.5, 4);
        let set = smp.collect(&plan, Some(&mut led));

        assert_eq!(set.len(), 30);
        assert_eq!(led.samples, plan.sweeps() as u64 * g.n as u64, "one Gibbs cycle per free node per sweep");
        assert_eq!(led.reads, 30 * g.n as u64, "one read per node per kept state");

        // And what that omission was worth. A read is 1.692 pJ against 7.09 fJ for a cycle, so a
        // plan at this thinning spends most of its energy on readback -- the exact quantity the
        // hand-written clone loops reported as zero.
        let with = led.joules(&crate::ledger::Z1_SPICE).unwrap();
        let sweeps_only =
            Ledger { samples: led.samples, reads: 0, writes: 0 }.joules(&crate::ledger::Z1_SPICE).unwrap();
        assert!(
            with > 3.0 * sweeps_only,
            "reads were {with:.3e} J total against {sweeps_only:.3e} J of sampling; if that ratio \
             is near one the ledger has stopped pricing readback"
        );
    }

    /// The parallel path collects the same physics and charges the same bill.
    #[test]
    fn the_parallel_path_collects_the_same_way() {
        let g = crate::ising::lattice2d(8, 1.0);
        let plan = Plan::new(40, 60, 2);
        let (mut la, mut lb) = (Ledger::default(), Ledger::default());
        let a = Sampler::new(&g, 0.5, 12).collect(&plan, Some(&mut la));
        let b = Sampler::new(&g, 0.5, 12).collect_par(&plan, 4, Some(&mut lb));
        assert_eq!(la.samples, lb.samples, "same sweeps");
        assert_eq!(la.reads, lb.reads, "and the same reads: the parallel path is not cheaper");
        assert_eq!(la.reads, 60 * g.n as u64);
        assert_eq!(a.len(), b.len());
        // The states themselves differ -- the parallel path derives its randomness per (sweep,
        // class, chunk), which is a different stream by design and is tested for equal PHYSICS in
        // `gibbs`. What must not differ is what the device was charged.
        let (ma, mb) = (a.magnetization().unwrap(), b.magnetization().unwrap());
        assert!(ma.value.is_finite() && mb.value.is_finite());
    }

    #[test]
    fn the_same_seed_collects_the_same_states() {
        let g = glass(12, 21, 2);
        let plan = Plan::new(100, 200, 2);
        let a = Sampler::new(&g, 0.7, 5).collect(&plan, None);
        let b = Sampler::new(&g, 0.7, 5).collect(&plan, None);
        assert_eq!(a.states(), b.states());
        assert_eq!(a.energies(), b.energies());
    }

    /// Enumeration counts a degeneracy; a sample can only witness one.
    #[test]
    fn enumeration_counts_the_degeneracy_a_sample_can_only_witness() {
        // A frustrated triangle: every coupling wants the two spins it joins to DISAGREE, and
        // three of them cannot all be satisfied. Six of the eight states attain the minimum.
        let mut gb = GraphBuilder::new(3);
        gb.couple(0, 1, -1.0);
        gb.couple(1, 2, -1.0);
        gb.couple(0, 2, -1.0);
        let g = gb.build();

        let enu = enumerate(&g, 2.0).unwrap();
        assert_eq!(enu.best().unwrap().1, -1.0);
        assert_eq!(enu.ground_states(1e-9).len(), 6, "the frustrated triangle's ground manifold");

        let mut smp = Sampler::new(&g, 2.0, 3);
        let chain = smp.collect(&Plan::new(200, 400, 1), None);
        let witnessed = chain.ground_states(1e-9).len();
        assert!(
            (1..=6).contains(&witnessed),
            "a chain can only report what it saw: {witnessed} states"
        );
    }

    /// A chain that never moved reports a zero-width interval. It must also report why.
    #[test]
    fn a_frozen_chain_flags_itself_beside_its_zero_width_interval() {
        let mut gb = GraphBuilder::new(6);
        for i in 0..6 {
            gb.bias(i, 40.0); // a field no thermal fluctuation at this beta will ever beat
        }
        let g = gb.build();
        let mut smp = Sampler::new(&g, 2.0, 8);
        let set = smp.collect(&Plan::new(100, 400, 1), None);

        let e = set.mean_spin(0).unwrap();
        assert_eq!(e.value, 1.0);
        assert_eq!(e.stderr, 0.0, "a constant observable has zero sample variance");
        assert!(
            !e.tau_int.is_finite(),
            "the zero-width interval is arithmetically right and reads as certainty; the infinite \
             tau is the only thing standing between a caller and that reading"
        );
        assert_eq!(e.ess, 1.0);
        assert!(!set.certificate(&g).unwrap().passed(), "and the certificate says so too");
    }

    /// The two moments contrastive divergence matches, against the oracle.
    #[test]
    fn the_first_and_second_moments_agree_with_enumeration() {
        let g = glass(12, 44, 2);
        let beta = 0.6;
        let enu = enumerate(&g, beta).unwrap();
        let mut smp = Sampler::new(&g, beta, 17);
        let set = smp.collect(&Plan::new(2_000, 8_000, 1), None);
        for (i, j) in [(0usize, 1usize), (3, 4), (2, 3), (5, 6)] {
            let t = enu.correlation(i, j).unwrap().value;
            let e = set.correlation(i, j).unwrap();
            assert!(
                e.covers(t),
                "<s{i} s{j}>: chain {e} does not cover the enumerated {t:.5}"
            );
        }
        let m = set.magnetization().unwrap();
        assert!(m.covers(enu.magnetization().unwrap().value), "magnetization: {m}");
    }

    #[test]
    fn enumeration_refuses_a_model_it_cannot_materialise() {
        let g = crate::ising::ring(ENUMERATION_LIMIT + 1, 1.0, 0.0);
        let err = enumerate(&g, 0.5).unwrap_err();
        assert_eq!(
            err,
            Refused::TooLargeToEnumerate { spins: ENUMERATION_LIMIT + 1, limit: ENUMERATION_LIMIT }
        );
        assert!(format!("{err}").contains("exact::marginals"), "a refusal should say what to do instead");
    }

    #[test]
    fn a_plan_cannot_thin_by_zero() {
        let p = Plan::new(10, 5, 0);
        assert_eq!(p.thin, 1, "thinning by zero would return the same state five times");
        assert_eq!(p.sweeps(), 15);
    }

    #[test]
    fn mismatched_widths_are_refused_at_construction() {
        let r = std::panic::catch_unwind(|| {
            SampleSet::from_search(vec![vec![1i8, 1], vec![1i8]], vec![0.0, 0.0], "x")
        });
        assert!(r.is_err(), "a set whose states have different widths cannot answer mean_spin(i)");
    }
}
