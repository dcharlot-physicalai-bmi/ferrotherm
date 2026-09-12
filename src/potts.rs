//! The `q`-state Potts model and the `q`-state clock model: categorical spins, natively.
//!
//! # What these models are
//!
//! The Ising model gives a site two states. The **Potts model** (Potts, *Some generalized
//! order-disorder transformations*, Proc. Camb. Phil. Soc. **48** (1952) 106; reviewed by Wu,
//! Rev. Mod. Phys. **54** (1982) 235) gives it `q`, and charges only for agreement:
//!
//! ```text
//!     H = - sum_<ij> J_ij delta(s_i, s_j)  -  sum_i h_i(s_i),      s_i in {0, .., q-1}
//! ```
//!
//! Every disagreement costs the same, whatever the two states are, so the symmetry group is the
//! whole permutation group on `q` letters. The **clock model** (the vector Potts model; José,
//! Kadanoff, Kirkpatrick & Nelson, Phys. Rev. B **16** (1977) 1217; Elitzur, Pearson & Shigemitsu,
//! Phys. Rev. D **19** (1979) 3698) puts the states on a circle instead and charges by angle:
//!
//! ```text
//!     H = - sum_<ij> J_ij cos(2 pi (s_i - s_j) / q)  -  sum_i h_i(s_i)
//! ```
//!
//! so its symmetry is only the `q`-fold rotation group, and `q -> infinity` is the `XY` model.
//! The two coincide for `q <= 3` and part company at `q = 4`, which this module asserts both halves
//! of rather than stating — see `the_clock_and_potts_models_coincide_at_q3_and_part_at_q4`.
//!
//! # Why this is here
//!
//! This crate is binary spins from end to end. [`crate::encode`] spells a `k`-valued variable in
//! spins and [`crate::categorical`] measures what that costs, but neither is a Potts model: there,
//! a categorical variable is a *penalty-enforced encoding* whose interaction is between SPINS, and
//! the penalty exists to forbid the states the encoding cannot read. Here the interaction is
//! between CATEGORIES, every configuration is legal, and there is no penalty to tune. The word
//! "Potts" appeared zero times in the 103 modules of `src/` before this one. It appears six times in
//! the repository's PROSE, and none of those is a model: `ROADMAP.md` lists a planned
//! "Potts to Ising (domain wall)" lowering pass — which is the [`crate::encode`] direction, a
//! categorical variable compiled DOWN into spins — and `docs/LANDSCAPE.md` counts what other
//! projects do with one.
//!
//! It matters because the Potts model is where this crate's verification vocabulary stops working.
//! `q = 2` has an up/down symmetry that makes a cluster flip a sign change, and two of the moves in
//! [`crate::cluster`] — the gauge, and the ghost spin that absorbs a field — are statements about
//! that sign. Above `q = 2` there is no sign, and what replaces it is a group action.
//!
//! # The three exact facts this module is checked against
//!
//! 1. **`q = 2` is the Ising model already in this crate, exactly.** `delta(a,b) = (1 + s_a s_b)/2`
//!    with `s_a = +1` for state 0 and `-1` for state 1, so `E_potts = E_ising(J_ising = J/2) + c`
//!    with `c = -(1/2) sum_edges J_ij`, and with fields `c` also takes
//!    `-sum_i (h_i(0) + h_i(1))/2` while `h_ising_i = (h_i(0) - h_i(1))/2`. [`Potts::to_ising`]
//!    performs that map and RETURNS the constant, because a partition function is not invariant
//!    under dropping it.
//! 2. **The square-lattice ferromagnetic Potts critical point is `beta_c = ln(1 + sqrt(q))`**
//!    (Potts 1952, from self-duality of the square lattice), and the internal energy AT that point
//!    is exactly `<delta> = (1 + 1/sqrt(q)) / 2` per bond (Baxter, *Exactly Solved Models in
//!    Statistical Mechanics*, ch. 12). At `q = 2` the second number is
//!    `(1 + 1/sqrt(2))/2 = 0.853553`, which is Onsager's `<s s> = 1/sqrt(2)` in Potts coordinates.
//!    [`critical_beta`] and [`critical_bond_energy`] are those two closed forms.
//! 3. **Enumeration over all `q^n` states** is exact for small `n`. [`enumerate`] is that oracle,
//!    and every sampler here is scored against it.
//!
//! ## How far a finite lattice may legitimately sit from `beta_c`
//!
//! `beta_c` is a **thermodynamic-limit** statement and an `L x L` torus does not have one. What a
//! finite lattice has is a pseudo-critical point offset from `beta_c` by `O(L^(-1/nu))`, with
//! `nu = 1` at `q = 2` and `nu = 5/6` at `q = 3`, smeared over a window of the same width; on top
//! of that the bond energy carries its own correction of order `1/L`. A test asserting that a
//! finite lattice sits ON `beta_c` would be asserting something false, and one that allowed any
//! window at all would assert nothing.
//!
//! The offset was MEASURED here rather than quoted, by bisecting for the `beta*` at which a finite
//! lattice's bond energy crosses the exact critical value (6000 Swendsen-Wang sweeps per point,
//! `beta* - beta_c`):
//!
//! | `L` | `q = 2` | `q = 3` | `q = 4` |
//! |---|---|---|---|
//! | 8 | −0.0248 | −0.0291 | −0.0234 |
//! | 16 | −0.0099 | −0.0124 | −0.0111 |
//! | 24 | −0.0065 | −0.0070 | −0.0074 |
//!
//! Every entry is NEGATIVE — a finite lattice orders LATE, and its apparent transition sits below
//! `beta_c` — and the magnitude falls roughly as `1/L`: about `0.011` at `L = 16` and half that
//! again by `L = 24`. That the offsets shrink toward zero is itself a check on BOTH closed forms at
//! once, since a wrong critical energy would make `beta*` converge somewhere other than `beta_c`.
//!
//! So what `the_square_lattice_transition_brackets_ln_one_plus_sqrt_q` asserts is a **bracket**
//! wider than that offset: the exact critical bond energy `(1 + 1/sqrt(q))/2` is straddled by the
//! measured bond energy at `beta_c - 0.04` and at `beta_c + 0.04` on a `16 x 16` lattice, at every
//! `q` in `{2, 3, 4}` (6000 sweeps per point; the test itself runs 2500 and lands in the same
//! place):
//!
//! | `q` | `beta_c` | `<delta>` at `beta_c - 0.04` | exact at `beta_c` | `<delta>` at `beta_c + 0.04` |
//! |---|---|---|---|---|
//! | 2 | 0.8814 | 0.8202 | 0.8536 | 0.8969 |
//! | 3 | 1.0051 | 0.7361 | 0.7887 | 0.8663 |
//! | 4 | 1.0986 | 0.6670 | 0.7500 | 0.8638 |
//!
//! `0.04` is not a tolerance chosen to pass: at `0.02` the bracket still holds at `L = 16` and at
//! `0.01` it FAILS for `q = 2` and `q = 3`, because the window has become narrower than the
//! finite-size shift in the table above. The test asserts that failure too, at `L = 8` where the
//! shift is largest — a bracket that held at every width would be measuring nothing.
//!
//! # The cluster update above `q = 2`
//!
//! [`crate::cluster`] builds Swendsen-Wang and Wolff for binary spins, and its two key moves —
//! gauging a balanced model to a ferromagnet, and absorbing a field into a ghost spin — are both
//! arguments about the SIGN of a spin. Neither survives `q > 2`: `Z_q` has no sign to flip. So the
//! generalisation here is not a port of that code, it is the group-theoretic statement that code is
//! a special case of (Kandel & Domany, Phys. Rev. B **43** (1991) 8539):
//!
//! > Pick an **involution** `R` of the single-site states that is a symmetry of the pair term,
//! > `f(Ra, Rb) = f(a, b)`. Open the bond on edge `(i,j)` with probability
//! > `1 - exp(-beta max(0, d_ij))`, where `d_ij = -J_ij [f(R s_i, s_j) - f(s_i, s_j)]` is the energy
//! > change if `i` ALONE were reflected. Apply `R` to the whole cluster.
//!
//! Detailed balance is then one line: `max(0,d) - max(0,-d) = d` for every real `d`, so the forward
//! and reverse proposals' boundary factors differ by exactly `exp(-beta dE)` and the move is
//! accepted with probability one. `d_ij` is symmetric in `i` and `j` for free, since
//! `f(Ra, b) = f(RRa, Rb) = f(a, Rb)`.
//!
//! For Potts the involution is the transposition of two colours, and it reproduces Wolff
//! (Phys. Rev. Lett. **62** (1989) 361) exactly: a transposition gives `d = J` on an aligned edge,
//! `d = -J` across an `a`-`b` edge and `d = 0` everywhere else, so the cluster is monochromatic and
//! the bond probability is the Fortuin-Kasteleyn `1 - exp(-beta J)`. For the clock model the
//! involution is the reflection `a -> (r - a) mod q`, which is why the clock model gets a cluster
//! update here at all.
//!
//! **Potts Swendsen-Wang is the one move here that is NOT an instance of the framework, and it is
//! stronger for it.** Swendsen & Wang's original (Phys. Rev. Lett. **58** (1987) 86) decomposes the
//! whole lattice on the same bond rule but is COLOUR-BLIND: each component then takes a uniformly
//! random state out of all `q`, rather than being mapped by one involution chosen in advance. That
//! is a larger move — a sweep can send two clusters to two different new colours — and it is the one
//! implemented for [`Interaction::Potts`]. The clock model has no colour-blind form, because a clock
//! cluster is only defined relative to a reflection, so [`Cluster::SwendsenWang`] there is the
//! Kandel-Domany version: one reflection for the sweep, each cluster reflected on its own coin.
//!
//! At `q = 2` the bond probability must come out as [`crate::cluster`]'s: `1 - exp(-beta J_potts)`
//! against its `1 - exp(-2 beta J_ising)`, with `J_ising = J_potts/2`. Those are the same number,
//! and `q2_cluster_sizes_match_the_binary_cluster_module` measures it rather than asserting it.
//!
//! **Fields are refused, and that is a fact about the model rather than an omission.** A cluster
//! move is free across a cluster only because `R` is a symmetry of the pair term. A field is not
//! symmetric under `R` — not being symmetric is what a field IS — so the move's acceptance would no
//! longer be one. [`crate::cluster::with_ghost`] escapes this at `q = 2` because a binary field is a
//! coupling to one extra spin; there is no `q`-state analogue that leaves the model unchanged.
//! [`Potts::to_ising`] is the honest route for a biased binary model.

use crate::graph::{Graph, GraphBuilder};
use crate::ledger::Ledger;
use crate::rng::Pcg;
use crate::samples::{Estimate, Plan};

/// Which pair interaction the categorical spins carry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Interaction {
    /// `delta(a, b)`: one when the two sites carry the same state, zero otherwise (Potts 1952).
    ///
    /// Every disagreement costs the same, so the symmetry group is the full permutation group on
    /// `q` letters.
    Potts,
    /// `cos(2 pi (a - b) / q)`: the states are directions on a circle, and disagreement costs by
    /// ANGLE.
    ///
    /// The symmetry group is only the `q`-fold rotation group. Equal to [`Interaction::Potts`] up
    /// to a rescaling of `J` and an additive constant for `q <= 3`, and genuinely different above.
    Clock,
}

impl Interaction {
    /// The pair term `f(a, b)`, which the Hamiltonian multiplies by `-J_ij`.
    ///
    /// Symmetric in its two arguments BIT FOR BIT, not merely to within rounding: the clock term
    /// folds `|a - b|` into `[0, q/2]` before taking a cosine, so `pair(a, b)` and `pair(b, a)`
    /// evaluate the identical expression. An energy that depended on which end of an edge was
    /// listed first would be an energy the CSR neighbour order could move.
    ///
    /// # Panics
    ///
    /// If `q` is below 2, or either state is not below `q`. A state the model has no room for is
    /// not reduced modulo `q`: that substitution is what this crate's tenth invariant forbids, and
    /// it would silently alias state `q` onto state 0.
    #[must_use]
    pub fn pair(self, q: usize, a: u8, b: u8) -> f64 {
        assert!(q >= 2, "a model with fewer than 2 states per site is a constant, got q = {q}");
        assert!(
            usize::from(a) < q && usize::from(b) < q,
            "states must be below q = {q}, got ({a}, {b})"
        );
        match self {
            Interaction::Potts => f64::from(u8::from(a == b)),
            Interaction::Clock => cos_step(q, usize::from(a.abs_diff(b))),
        }
    }

    /// `max f - min f` over every pair of states: the pair term's full swing.
    ///
    /// One for [`Interaction::Potts`] at every `q`. For [`Interaction::Clock`] it is `2` at even `q`
    /// (where `a` and `a + q/2` are opposite) and `1 + cos(pi/q)` at odd `q`, which is why it is
    /// measured over the table rather than written as a constant.
    ///
    /// # Panics
    ///
    /// If `q` is below 2.
    #[must_use]
    pub fn span(self, q: usize) -> f64 {
        assert!(q >= 2, "a model with fewer than 2 states per site is a constant, got q = {q}");
        match self {
            Interaction::Potts => 1.0,
            Interaction::Clock => 1.0 - (0..q).map(|k| cos_step(q, k)).fold(f64::INFINITY, f64::min),
        }
    }

    /// A short label, for messages and tables.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Interaction::Potts => "potts",
            Interaction::Clock => "clock",
        }
    }
}

/// `cos(2 pi k / q)`, exact wherever the value is a dyadic rational.
///
/// Folding `k` into `[0, q/2]` first is what makes the term symmetric bit for bit; writing the five
/// exact values down is what keeps a `q = 3` energy an exact multiple of one half instead of
/// `-0.4999999999999998`. `cos` is correct to an ulp, and an ulp is the difference between an energy
/// that is a dyadic rational and one that is not — which is the difference between the `q = 3`
/// clock-equals-Potts test asserting an equality and asserting a tolerance.
fn cos_step(q: usize, k: usize) -> f64 {
    let k = k % q;
    let k = k.min(q - k);
    if k == 0 {
        1.0
    } else if 2 * k == q {
        -1.0
    } else if 4 * k == q {
        0.0
    } else if 3 * k == q {
        -0.5
    } else if 6 * k == q {
        0.5
    } else {
        (core::f64::consts::TAU * k as f64 / q as f64).cos()
    }
}

/// A state vector this model cannot read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Invalid {
    /// The vector has the wrong number of sites.
    Width {
        /// Length handed in.
        got: usize,
        /// Sites the model has.
        want: usize,
    },
    /// A site carries a value the model has no state for.
    ///
    /// Not reduced modulo `q`, and not clamped. Either substitution maps a value the caller never
    /// meant onto a legal state, and answers a different question silently.
    State {
        /// Which site.
        site: usize,
        /// What it carried.
        value: u8,
        /// States the model has.
        q: usize,
    },
}

impl core::fmt::Display for Invalid {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Invalid::Width { got, want } => {
                write!(f, "state vector has {got} sites, model has {want}")
            }
            Invalid::State { site, value, q } => write!(
                f,
                "site {site} carries state {value}, but the model has {q} states (0..{})",
                q - 1
            ),
        }
    }
}

impl core::error::Error for Invalid {}

/// This model is not binary, so it is not an Ising model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NotBinary {
    /// States per site, which must be 2 for the map to exist.
    pub q: usize,
    /// Which interaction was asked to map.
    pub kind: Interaction,
}

impl core::fmt::Display for NotBinary {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "the {} model at q = {} has no Ising form: a spin has two states and a site here has {}",
            self.kind.label(),
            self.q,
            self.q
        )
    }
}

impl core::error::Error for NotBinary {}

/// No cluster update is valid on this model, and here is the term that stops it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum NoCluster {
    /// A negative coupling in a [`Interaction::Potts`] model.
    ///
    /// The Fortuin-Kasteleyn bond probability is `1 - exp(-beta J)`, which is negative for `J < 0`.
    /// At `q = 2` an antiferromagnet on a balanced graph can be gauged to a ferromagnet — see
    /// [`crate::cluster::gauge`], reachable through [`Potts::to_ising`]. Above `q = 2` there is no
    /// such gauge, because a gauge flips a SIGN and `Z_q` has none. This is a fact about the model.
    Antiferromagnetic {
        /// One end of the edge.
        i: usize,
        /// The other end.
        j: usize,
        /// Its coupling, which is negative.
        coupling: f64,
    },
    /// A site carries a field, which no cluster move can be a symmetry of.
    ///
    /// See the module documentation: the move is free across a cluster only because the pair term is
    /// invariant under the reflection, and a field's whole purpose is not to be.
    Fielded {
        /// Which site.
        site: usize,
        /// Which of its states carries the field.
        state: u8,
        /// The field, which is non-zero.
        field: f64,
    },
}

impl core::fmt::Display for NoCluster {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            NoCluster::Antiferromagnetic { i, j, coupling } => write!(
                f,
                "edge ({i},{j}) has J = {coupling}, and the Potts bond probability \
                 1 - exp(-beta J) is negative there; q > 2 has no gauge to a ferromagnet"
            ),
            NoCluster::Fielded { site, state, field } => write!(
                f,
                "site {site} has field {field} on state {state}, and a cluster move applies a \
                 symmetry of the PAIR term, which a field does not share"
            ),
        }
    }
}

impl core::error::Error for NoCluster {}

/// Too many states to enumerate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TooBig {
    /// States per site.
    pub q: usize,
    /// Sites.
    pub n: usize,
    /// `q^n`, or `None` when even that overflowed a `usize`.
    pub states: Option<usize>,
    /// The largest `q^n` [`enumerate`] will attempt.
    pub limit: usize,
}

impl core::fmt::Display for TooBig {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.states {
            Some(s) => write!(
                f,
                "q = {} on {} sites is {s} states, above the enumeration limit of {}",
                self.q, self.n, self.limit
            ),
            None => write!(
                f,
                "q = {} on {} sites overflows usize, far above the enumeration limit of {}",
                self.q, self.n, self.limit
            ),
        }
    }
}

impl core::error::Error for TooBig {}

/// Accumulates couplings and fields, then freezes them into a [`Potts`].
///
/// Duplicate pairs are SUMMED at build time and fields accumulate, exactly as
/// [`crate::graph::GraphBuilder`] does and for the same reason: two passes over one site — a model
/// term plus a penalty term — must compose rather than overwrite.
pub struct PottsBuilder {
    q: usize,
    n: usize,
    kind: Interaction,
    edges: Vec<(u32, u32, f64)>,
    h: Vec<f64>,
}

impl PottsBuilder {
    /// A builder over `n` sites of `q` states each, uncoupled and unbiased.
    ///
    /// # Panics
    ///
    /// If `q` is below 2, or above 255: a state is a `u8` here because a Potts state is a LABEL
    /// rather than a number, and a label that does not fit in a byte is a different data structure.
    #[must_use]
    pub fn new(q: usize, n: usize, kind: Interaction) -> PottsBuilder {
        assert!(q >= 2, "a model with fewer than 2 states per site is a constant, got q = {q}");
        assert!(q <= 255, "states are u8 labels, so q must be at most 255, got q = {q}");
        PottsBuilder { q, n, kind, edges: Vec::new(), h: vec![0.0; n * q] }
    }

    /// Add an undirected coupling `J_ij`. Duplicate pairs are summed at build time.
    ///
    /// # Panics
    ///
    /// If either index is past the site count, or they are equal — a self-coupling is a constant.
    pub fn couple(&mut self, i: usize, j: usize, jij: f64) {
        assert!(i < self.n && j < self.n && i != j, "bad edge ({i},{j}) n = {}", self.n);
        self.edges.push((i as u32, j as u32, jij));
    }

    /// Add field `h_i(a)`, which LOWERS the energy of state `a` at site `i`. Repeated calls
    /// **accumulate**.
    ///
    /// # Panics
    ///
    /// If the site is past the end, or `a` is not below `q` — naming both, because a state index
    /// silently reduced modulo `q` is the failure this crate's tenth invariant is about.
    pub fn field(&mut self, i: usize, a: u8, h: f64) {
        assert!(i < self.n, "site {i} is past the end of a {}-site model", self.n);
        assert!(usize::from(a) < self.q, "state {a} is not below q = {} at site {i}", self.q);
        self.h[i * self.q + usize::from(a)] += h;
    }

    /// Freeze into CSR, summing duplicate edges.
    ///
    /// Merged through a `BTreeMap` rather than a hash map for the reason spelled out in
    /// [`crate::graph::GraphBuilder::build`]: the merge order decides the CSR neighbour order, which
    /// decides the order every local field is summed in, and float addition is not associative. A
    /// randomised iteration order would make one model's energies take several values across runs
    /// that all print the same.
    #[must_use]
    pub fn build(self) -> Potts {
        let n = self.n;
        let mut merged: std::collections::BTreeMap<(u32, u32), f64> =
            std::collections::BTreeMap::new();
        for (a, b, j) in self.edges {
            let key = if a < b { (a, b) } else { (b, a) };
            *merged.entry(key).or_insert(0.0) += j;
        }
        let mut deg = vec![0usize; n];
        for &(a, b) in merged.keys() {
            deg[a as usize] += 1;
            deg[b as usize] += 1;
        }
        let mut offset = vec![0usize; n + 1];
        for i in 0..n {
            offset[i + 1] = offset[i] + deg[i];
        }
        let mut nbr = vec![0u32; offset[n]];
        let mut w = vec![0.0f64; offset[n]];
        let mut cursor = offset.clone();
        for (&(a, b), &j) in &merged {
            nbr[cursor[a as usize]] = b;
            w[cursor[a as usize]] = j;
            cursor[a as usize] += 1;
            nbr[cursor[b as usize]] = a;
            w[cursor[b as usize]] = j;
            cursor[b as usize] += 1;
        }
        Potts { q: self.q, n, kind: self.kind, offset, nbr, w, h: self.h, n_edges: merged.len() }
    }
}

/// A `q`-state model on a sparse graph, in CSR.
///
/// Energy convention, matching [`crate::graph::Graph`] so the two can be compared term by term:
/// `E(s) = - sum_edges J_ij f(s_i, s_j) - sum_i h_i(s_i)`, where `f` is the interaction's pair term.
/// Positive `J` is ferromagnetic under both interactions.
pub struct Potts {
    q: usize,
    n: usize,
    kind: Interaction,
    offset: Vec<usize>,
    nbr: Vec<u32>,
    w: Vec<f64>,
    h: Vec<f64>,
    n_edges: usize,
}

impl Potts {
    /// States per site.
    #[must_use]
    pub fn q(&self) -> usize {
        self.q
    }

    /// Sites.
    #[must_use]
    pub fn n(&self) -> usize {
        self.n
    }

    /// Which pair interaction this model carries.
    #[must_use]
    pub fn kind(&self) -> Interaction {
        self.kind
    }

    /// Undirected edges, which is half the CSR neighbour-list length.
    #[must_use]
    pub fn n_edges(&self) -> usize {
        self.n_edges
    }

    /// Every undirected edge once, as `(i, j, J_ij)` with `i < j`.
    pub fn edges(&self) -> impl Iterator<Item = (usize, usize, f64)> + '_ {
        (0..self.n).flat_map(move |i| {
            (self.offset[i]..self.offset[i + 1]).filter_map(move |k| {
                let j = self.nbr[k] as usize;
                (j > i).then_some((i, j, self.w[k]))
            })
        })
    }

    /// The field on state `a` at site `i`, zero where none was set.
    ///
    /// # Panics
    ///
    /// If the site is past the end or `a` is not below `q`.
    #[must_use]
    pub fn field(&self, i: usize, a: u8) -> f64 {
        assert!(i < self.n, "site {i} is past the end of a {}-site model", self.n);
        assert!(usize::from(a) < self.q, "state {a} is not below q = {}", self.q);
        self.h[i * self.q + usize::from(a)]
    }

    /// Whether any site carries a field.
    #[must_use]
    pub fn has_field(&self) -> bool {
        self.h.iter().any(|&x| x != 0.0)
    }

    /// Energy of a state, refusing anything this model cannot read.
    ///
    /// # Errors
    ///
    /// [`Invalid::Width`] for the wrong number of sites, [`Invalid::State`] for a value at or above
    /// `q`, naming the site and the value. Neither is reduced or clamped.
    pub fn energy(&self, s: &[u8]) -> Result<f64, Invalid> {
        self.check(s)?;
        Ok(self.energy_of(s))
    }

    /// The state check [`Potts::energy`] runs, on its own.
    ///
    /// # Errors
    ///
    /// As [`Potts::energy`].
    pub fn check(&self, s: &[u8]) -> Result<(), Invalid> {
        if s.len() != self.n {
            return Err(Invalid::Width { got: s.len(), want: self.n });
        }
        for (site, &v) in s.iter().enumerate() {
            if usize::from(v) >= self.q {
                return Err(Invalid::State { site, value: v, q: self.q });
            }
        }
        Ok(())
    }

    /// Energy of a state already known to fit. The hot path; samplers hold their own state.
    fn energy_of(&self, s: &[u8]) -> f64 {
        let mut e = 0.0;
        for i in 0..self.n {
            e -= self.h[i * self.q + usize::from(s[i])];
            for k in self.offset[i]..self.offset[i + 1] {
                let j = self.nbr[k] as usize;
                if j > i {
                    e -= self.w[k] * self.kind.pair(self.q, s[i], s[j]);
                }
            }
        }
        e
    }

    /// The part of the energy site `i` would carry in state `a`: `-sum_j J_ij f(a, s_j) - h_i(a)`.
    ///
    /// Every term not touching `i` is omitted, so the DIFFERENCE of two of these is the exact energy
    /// change of a single-site move — which is what both local updates use.
    fn site_energy(&self, i: usize, a: u8, s: &[u8]) -> f64 {
        let mut e = -self.h[i * self.q + usize::from(a)];
        for k in self.offset[i]..self.offset[i + 1] {
            e -= self.w[k] * self.kind.pair(self.q, a, s[self.nbr[k] as usize]);
        }
        e
    }

    /// The index of a state in `q`-ary counting order, site 0 least significant.
    ///
    /// The order [`enumerate`] uses, so a sampled histogram and an exact distribution line up.
    ///
    /// # Errors
    ///
    /// As [`Potts::energy`].
    pub fn index_of(&self, s: &[u8]) -> Result<usize, Invalid> {
        self.check(s)?;
        let mut idx = 0usize;
        for &v in s.iter().rev() {
            idx = idx * self.q + usize::from(v);
        }
        Ok(idx)
    }

    /// The state at an index in `q`-ary counting order. The inverse of [`Potts::index_of`].
    ///
    /// # Panics
    ///
    /// If the index is at or above `q^n`.
    #[must_use]
    pub fn state_at(&self, mut index: usize) -> Vec<u8> {
        let mut s = vec![0u8; self.n];
        for v in &mut s {
            *v = (index % self.q) as u8;
            index /= self.q;
        }
        assert_eq!(index, 0, "index is past q^n for q = {} on {} sites", self.q, self.n);
        s
    }

    /// The Potts order parameter of a state: `(q * max_a n_a / n - 1) / (q - 1)`.
    ///
    /// Zero when the states are spread evenly and one when every site agrees, at every `q`. The
    /// magnetisation of a `q`-state model is not a sum of spins — there is no direction to sum — so
    /// it is built from the occupation of the most popular state instead.
    ///
    /// # Errors
    ///
    /// As [`Potts::energy`].
    pub fn order_parameter(&self, s: &[u8]) -> Result<f64, Invalid> {
        self.check(s)?;
        let mut count = vec![0usize; self.q];
        for &v in s {
            count[usize::from(v)] += 1;
        }
        let top = count.iter().copied().max().unwrap_or(0) as f64;
        let q = self.q as f64;
        Ok((q * top / self.n as f64 - 1.0) / (q - 1.0))
    }

    /// An UPPER BOUND on the energy change any single-site move can produce.
    ///
    /// This instance's energy scale, and the number a `beta` ladder has to be measured against — the
    /// argument in [`crate::graph::Graph::flip_gap_max`] applies here unchanged: `beta` and energy
    /// enter the Boltzmann weight only as their product, so the same model written in different
    /// units is the same problem, and an absolute ladder answers it differently.
    ///
    /// A bound rather than a maximum, and the difference is deliberate. Moving site `i` from `a` to
    /// `a'` changes the energy by `sum_j J_ij [f(a',s_j) - f(a,s_j)] + h_i(a) - h_i(a')`, and each
    /// bracket is at most [`Interaction::span`]; above `q = 2` no single `a -> a'` need attain the
    /// span on every neighbour at once, so this can exceed the true maximum. It accumulates through
    /// [`crate::round::sum_up`] with each product pushed up one ulp first, so it is a bound and not
    /// merely an estimate — this crate shipped a "lower bound" that sat ABOVE the optimum because it
    /// summed in round-to-nearest.
    ///
    /// `None` where the scale is zero: a model with no couplings and no fields has no energy scale,
    /// every derived `beta` would be infinite, and returning `0.0` would invite that division.
    #[must_use]
    pub fn single_site_gap_max(&self) -> Option<f64> {
        let span = self.kind.span(self.q);
        let mut worst = 0.0f64;
        for i in 0..self.n {
            // `next_up` is guarded on the INPUT being zero, not on the product. Rounding a term up
            // one ulp is what makes the total a bound; doing it to a term that is EXACTLY zero
            // manufactures width out of nothing, and `f64::MIN_POSITIVE.next_down()` is 5e-324, not
            // zero. Measured: without these guards an empty model — no couplings, no fields —
            // reported a gap of `Some(5e-324)` instead of `None`, so a caller was invited to divide
            // by it. The bound direction was never at risk; the *no scale at all* answer was.
            let mut terms: Vec<f64> = (self.offset[i]..self.offset[i + 1])
                .map(|k| {
                    if self.w[k] == 0.0 { 0.0 } else { (span * self.w[k].abs()).next_up() }
                })
                .collect();
            let fields = &self.h[i * self.q..(i + 1) * self.q];
            let hi = fields.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let lo = fields.iter().copied().fold(f64::INFINITY, f64::min);
            terms.push(if hi == lo { 0.0 } else { (hi - lo).next_up() });
            worst = worst.max(crate::round::sum_up(&terms));
        }
        (worst > 0.0 && worst.is_finite()).then_some(worst)
    }

    /// The Ising model this IS, when `q = 2`, and the additive constant that completes it.
    ///
    /// Returns `(g, c)` with `E_potts(s) = g.energy(sigma(s)) + c`, where `sigma_i = +1` when
    /// `s_i == 0` and `-1` when `s_i == 1`.
    ///
    /// Writing `f(a,b) = A + B sigma_a sigma_b` with `A = (f_same + f_diff)/2` and
    /// `B = (f_same - f_diff)/2` gives `J_ising = B J` and an edge constant `-A sum_edges J`; the
    /// field splits the same way into `h_ising_i = (h_i(0) - h_i(1))/2` and a constant
    /// `-sum_i (h_i(0) + h_i(1))/2`. For [`Interaction::Potts`] that is `A = B = 1/2`, so
    /// `J_ising = J/2`; for [`Interaction::Clock`] at `q = 2` it is `A = 0`, `B = 1`, so
    /// `J_ising = J` and the edge constant vanishes.
    ///
    /// **The constant is returned, not dropped.** It is invisible in a ground state and in every
    /// energy DIFFERENCE, and it is exactly what a partition function is:
    /// `Z_potts = exp(-beta c) Z_ising`, so `log Z` moves by `-beta c` and a free energy taken from
    /// the Ising form without it is wrong by that amount.
    ///
    /// This is the bridge to the rest of the crate: through it, a binary Potts model reaches
    /// [`crate::exact`], [`crate::planarcut`], [`crate::sdp`], [`crate::cluster`] and everything
    /// else that speaks spins.
    ///
    /// # Errors
    ///
    /// [`NotBinary`] at any `q` other than 2. A spin has two states; there is no map to invent.
    pub fn to_ising(&self) -> Result<(Graph, f64), NotBinary> {
        if self.q != 2 {
            return Err(NotBinary { q: self.q, kind: self.kind });
        }
        let same = self.kind.pair(2, 0, 0);
        let diff = self.kind.pair(2, 0, 1);
        let a = (same + diff) / 2.0;
        let b = (same - diff) / 2.0;
        let mut gb = GraphBuilder::new(self.n);
        let mut c = 0.0;
        for (i, j, jij) in self.edges() {
            gb.couple(i, j, b * jij);
            c -= a * jij;
        }
        for i in 0..self.n {
            let (h0, h1) = (self.h[2 * i], self.h[2 * i + 1]);
            gb.bias(i, (h0 - h1) / 2.0);
            c -= (h0 + h1) / 2.0;
        }
        Ok((gb.build(), c))
    }
}

/// A ring of `n` sites with uniform coupling `j` and no field.
///
/// # Panics
///
/// If `n` is below 3: a two-site ring would couple the same pair twice.
#[must_use]
pub fn ring(n: usize, q: usize, j: f64, kind: Interaction) -> Potts {
    assert!(n >= 3, "a ring needs at least 3 sites, got {n}");
    let mut b = PottsBuilder::new(q, n, kind);
    for i in 0..n {
        b.couple(i, (i + 1) % n, j);
    }
    b.build()
}

/// An `l x l` square lattice with periodic boundaries, uniform `j`, no field.
///
/// The graph [`critical_beta`] is a statement about. A side below 2 wraps every neighbour onto the
/// site itself, which is a self-coupling and therefore a constant, so it comes back uncoupled — the
/// same choice [`crate::ising::lattice2d`] makes, for the same reason.
#[must_use]
pub fn lattice2d(l: usize, q: usize, j: f64, kind: Interaction) -> Potts {
    if l < 2 {
        return PottsBuilder::new(q, l * l, kind).build();
    }
    let mut b = PottsBuilder::new(q, l * l, kind);
    for y in 0..l {
        for x in 0..l {
            let i = y * l + x;
            b.couple(i, y * l + (x + 1) % l, j);
            b.couple(i, ((y + 1) % l) * l + x, j);
        }
    }
    b.build()
}

/// `ln(1 + sqrt(q))`: the exact critical inverse temperature of the ferromagnetic `q`-state Potts
/// model on the square lattice at `J = 1`.
///
/// From self-duality (Potts 1952): the square lattice is its own dual, and the Kramers-Wannier
/// duality maps `beta` to `beta*` with `(e^beta - 1)(e^beta* - 1) = q`. The fixed point of that map
/// is the transition — continuous for `q <= 4` and first order above, which changes what a finite
/// lattice shows but not where the point is.
///
/// At `q = 2` this is `ln(1 + sqrt(2)) = 0.8814`, which is Onsager's `0.4407` DOUBLED: the Potts
/// coupling is twice the Ising one, exactly as [`Potts::to_ising`] says.
///
/// # Panics
///
/// If `q` is below 2.
#[must_use]
pub fn critical_beta(q: usize) -> f64 {
    assert!(q >= 2, "a model with fewer than 2 states per site has no transition, got q = {q}");
    (1.0 + (q as f64).sqrt()).ln()
}

/// `(1 + 1/sqrt(q)) / 2`: the exact internal energy per bond, `<delta(s_i,s_j)>`, of the
/// square-lattice ferromagnetic Potts model AT [`critical_beta`], in the thermodynamic limit.
///
/// Baxter, *Exactly Solved Models in Statistical Mechanics*, ch. 12: the self-dual point's energy
/// follows from the duality relation without solving the model. At `q = 2` it is
/// `(1 + 1/sqrt(2))/2 = 0.853553`, which is Onsager's nearest-neighbour correlation
/// `<s_i s_j> = 1/sqrt(2)` rewritten as `(1 + <s s>)/2`.
///
/// A finite lattice does not sit on this value — see the module documentation for how far off it may
/// legitimately be, and for the measured bracket.
///
/// # Panics
///
/// If `q` is below 2.
#[must_use]
pub fn critical_bond_energy(q: usize) -> f64 {
    assert!(q >= 2, "a model with fewer than 2 states per site has no transition, got q = {q}");
    (1.0 + 1.0 / (q as f64).sqrt()) / 2.0
}

/// The largest `q^n` [`enumerate`] will attempt.
///
/// One million states, which is 16 MB of `f64` for the two tables [`Enumerated`] keeps. The limit
/// exists so that an enumeration that cannot run says so with the numbers in hand rather than
/// exhausting memory — [`TooBig`] names `q`, `n` and `q^n`.
pub const ENUMERATION_LIMIT: usize = 1 << 20;

/// This module's RNG stream. Streams do not correlate, so a Potts chain and an Ising chain on the
/// same seed are independent rather than accidentally identical.
const POTTS_STREAM: u64 = 0x_7075;

/// The exact Boltzmann distribution of a model, from every one of its `q^n` states.
///
/// The oracle every sampler in this module is scored against. Exact means exact: these are sums over
/// the whole state space, with no sampling error to put an interval on.
pub struct Enumerated {
    /// States per site.
    pub q: usize,
    /// Sites.
    pub n: usize,
    /// The inverse temperature these weights are at.
    pub beta: f64,
    /// `log Z` at that `beta`, computed with the maximum log-weight shifted out so `exp` never
    /// overflows — the failure [`crate::ising::exact_boltzmann`] carries a paragraph about, avoided
    /// here in the same way.
    pub log_z: f64,
    /// `<E>`, the exact internal energy.
    pub mean_energy: f64,
    /// Probability of every state, indexed as [`Potts::index_of`] does.
    pub p: Vec<f64>,
    /// Energy of every state, in the same order.
    pub energies: Vec<f64>,
}

impl Enumerated {
    /// The exact expectation of any function of the state.
    ///
    /// Exact, so there is no error bar to return: the weights are the whole distribution.
    #[must_use]
    pub fn mean<F: Fn(&[u8]) -> f64>(&self, f: F) -> f64 {
        let mut s = vec![0u8; self.n];
        let mut acc = 0.0;
        for (index, &pk) in self.p.iter().enumerate() {
            let mut rest = index;
            for v in &mut s {
                *v = (rest % self.q) as u8;
                rest /= self.q;
            }
            acc += pk * f(&s);
        }
        acc
    }
}

/// The exact Boltzmann distribution over all `q^n` states.
///
/// # Errors
///
/// [`TooBig`] when `q^n` is above [`ENUMERATION_LIMIT`], naming `q`, `n` and the count. A refusal
/// rather than a truncation: a partial enumeration is not an oracle, and one that silently stopped
/// early would be a wrong answer with the shape of a right one.
pub fn enumerate(m: &Potts, beta: f64) -> Result<Enumerated, TooBig> {
    let states = u32::try_from(m.n).ok().and_then(|e| m.q.checked_pow(e));
    let total = match states {
        Some(s) if s <= ENUMERATION_LIMIT => s,
        other => return Err(TooBig { q: m.q, n: m.n, states: other, limit: ENUMERATION_LIMIT }),
    };

    let mut energies = vec![0.0f64; total];
    let mut p = vec![0.0f64; total];
    let mut s = vec![0u8; m.n];
    let mut mx = f64::NEG_INFINITY;
    for index in 0..total {
        let mut rest = index;
        for v in &mut s {
            *v = (rest % m.q) as u8;
            rest /= m.q;
        }
        let e = m.energy_of(&s);
        energies[index] = e;
        let l = -beta * e;
        p[index] = l;
        mx = mx.max(l);
    }
    let mut z = 0.0;
    for v in &mut p {
        *v = (*v - mx).exp();
        z += *v;
    }
    let mut mean_energy = 0.0;
    for (v, &e) in p.iter_mut().zip(energies.iter()) {
        *v /= z;
        mean_energy += *v * e;
    }
    Ok(Enumerated { q: m.q, n: m.n, beta, log_z: mx + z.ln(), mean_energy, p, energies })
}

/// A scalar trace turned into a number with an error bar.
///
/// `sqrt(var / ess)`, **never** `sqrt(var / N)` — this crate's seventh invariant, measured in
/// `examples/interval_calibration.rs`, where the naive interval covers 24% while announcing 95%. The
/// effective sample size is `N / (2 tau)` with `tau` from [`crate::certify::tau_int`], the same
/// estimator [`crate::samples::SampleSet`] uses, so an interval from this module and one from that
/// module mean the same thing.
///
/// A constant trace gets an infinite `tau` and an `ess` of one rather than a zero-width interval: a
/// chain stuck in one state reports its one value, and pretending that has no error is exactly how a
/// frozen sampler passes a test.
#[must_use]
pub fn estimate(trace: &[f64]) -> Estimate {
    let n = trace.len();
    if n == 0 {
        return Estimate { value: f64::NAN, stderr: f64::NAN, ess: 0.0, tau_int: f64::NAN };
    }
    let value = trace.iter().sum::<f64>() / n as f64;
    let var = if n > 1 {
        trace.iter().map(|x| (x - value).powi(2)).sum::<f64>() / (n - 1) as f64
    } else {
        0.0
    };
    let tau = crate::certify::tau_int(trace);
    let ess = if tau.is_finite() && tau > 0.0 { n as f64 / (2.0 * tau) } else { 1.0 };
    Estimate { value, stderr: (var / ess).sqrt(), ess, tau_int: tau }
}

/// States a run produced, in the model's own values.
///
/// Deliberately not [`crate::samples::SampleSet`]: that type is `i8` spins throughout and a Potts
/// state is a `u8` label. Handing categorical states to a container that types them as spins is the
/// substitution this crate's sixth invariant is about — a spin variable once reported 0 and 1 to the
/// reader while the decoder handed back -1 and +1.
pub struct Run {
    q: usize,
    n: usize,
    states: Vec<Vec<u8>>,
    energies: Vec<f64>,
}

impl Run {
    /// The states, in chain order.
    #[must_use]
    pub fn states(&self) -> &[Vec<u8>] {
        &self.states
    }

    /// Their energies, in the same order.
    #[must_use]
    pub fn energies(&self) -> &[f64] {
        &self.energies
    }

    /// How many states were kept.
    #[must_use]
    pub fn len(&self) -> usize {
        self.states.len()
    }

    /// Whether nothing was kept.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.states.is_empty()
    }

    /// `<E>` with the error bar [`estimate`] gives it.
    #[must_use]
    pub fn mean_energy(&self) -> Estimate {
        estimate(&self.energies)
    }

    /// The expectation of any function of the state, with its own error bar.
    ///
    /// The autocorrelation is measured on THIS observable's trace, so the interval belongs to this
    /// quantity rather than being borrowed from the energy.
    #[must_use]
    pub fn expectation<F: Fn(&[u8]) -> f64>(&self, f: F) -> Estimate {
        let vals: Vec<f64> = self.states.iter().map(|s| f(s)).collect();
        estimate(&vals)
    }

    /// The empirical distribution over all `q^n` states, for a total-variation comparison against
    /// [`enumerate`].
    ///
    /// # Errors
    ///
    /// [`TooBig`], on the same limit [`enumerate`] uses — the two vectors have to be the same length
    /// to be compared at all, and [`crate::ising::tv`] refuses a truncated comparison.
    pub fn histogram(&self) -> Result<Vec<f64>, TooBig> {
        let states = u32::try_from(self.n).ok().and_then(|e| self.q.checked_pow(e));
        let total = match states {
            Some(s) if s <= ENUMERATION_LIMIT => s,
            other => {
                return Err(TooBig { q: self.q, n: self.n, states: other, limit: ENUMERATION_LIMIT });
            }
        };
        let mut hist = vec![0.0f64; total];
        for s in &self.states {
            let mut idx = 0usize;
            for &v in s.iter().rev() {
                idx = idx * self.q + usize::from(v);
            }
            hist[idx] += 1.0;
        }
        let inv = 1.0 / self.states.len() as f64;
        for v in &mut hist {
            *v *= inv;
        }
        Ok(hist)
    }
}

/// A uniformly random state in `0..q`.
fn uniform_state(rng: &mut Pcg, q: usize) -> u8 {
    (((rng.f64() * q as f64) as usize).min(q - 1)) as u8
}

/// A uniformly random state in `0..q` that is NOT `excluded`.
///
/// The `q - 1` alternatives, each with probability `1/(q-1)`. Both cluster moves and the Metropolis
/// proposal need exactly this, and all three rely on it being SYMMETRIC — the chance of proposing
/// `b` from `a` equals the chance of proposing `a` from `b`, which is what lets the acceptance be
/// the bare Boltzmann ratio with no proposal correction.
fn other_state(rng: &mut Pcg, q: usize, excluded: u8) -> u8 {
    let k = ((rng.f64() * (q - 1) as f64) as usize).min(q - 2);
    (if k >= usize::from(excluded) { k + 1 } else { k }) as u8
}

/// Which single-site update a sweep makes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Local {
    /// Metropolis: propose one of the `q - 1` OTHER states uniformly, accept with
    /// `min(1, exp(-beta dE))`.
    ///
    /// The proposal is symmetric, so the acceptance is the bare Metropolis ratio. Proposing
    /// uniformly over all `q` states — the current one included — is also correct and is strictly
    /// lazier: it spends a `1/q` share of its moves proposing no move at all.
    Metropolis,
    /// Heat bath: draw the site from its exact conditional `P(a) ~ exp(-beta E_i(a))` over all `q`
    /// states.
    ///
    /// The `q`-state Gibbs update, and the generalisation of [`crate::gibbs::Sampler`]'s sigmoid: at
    /// `q = 2` a softmax over two states IS a sigmoid of their difference. It costs `q` local
    /// energies against Metropolis's two, and never rejects.
    HeatBath,
}

/// What one local sweep did.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SweepStats {
    /// Single-site updates attempted, which is `n` per sweep under both updates.
    pub updates: u64,
    /// Updates that left the site in a DIFFERENT state than it started.
    ///
    /// An acceptance count for [`Local::Metropolis`]; for [`Local::HeatBath`], which never rejects,
    /// the number of draws that landed somewhere new. Defined the same way for both on purpose: a
    /// rate defined per update would mean different things for the two and could not be compared.
    pub changed: u64,
}

/// Single-site sampling of a [`Potts`] model.
///
/// # What the ledger is charged
///
/// `samples` counts single-site updates and `writes` counts sites whose value changed, as
/// [`crate::gibbs::Sampler`] does. `reads` is charged only by [`Sampler::collect`], one per site per
/// kept state, which is this crate's eighth invariant: reading a state costs the device, and on a
/// Z1-class device one read is worth 239 Gibbs cycles. [`crate::cluster`] charges `reads` per BOND
/// TESTED instead, so a joules figure compared across the two modules has to say which it means.
pub struct Sampler<'m> {
    m: &'m Potts,
    beta: f64,
    s: Vec<u8>,
    rng: Pcg,
    /// Conditional weights, reused so a heat-bath sweep does not allocate once per site.
    scratch: Vec<f64>,
}

impl<'m> Sampler<'m> {
    /// A sampler over `m` at inverse temperature `beta`, started from a uniformly random state.
    #[must_use]
    pub fn new(m: &'m Potts, beta: f64, seed: u64) -> Sampler<'m> {
        let mut rng = Pcg::new(seed, POTTS_STREAM);
        let s = (0..m.n).map(|_| uniform_state(&mut rng, m.q)).collect();
        Sampler { m, beta, s, rng, scratch: vec![0.0; m.q] }
    }

    /// Start from a given state instead of a random one.
    ///
    /// # Errors
    ///
    /// [`Invalid`], as [`Potts::energy`]: a state this model cannot read is refused rather than
    /// repaired.
    pub fn from_state(mut self, s: &[u8]) -> Result<Sampler<'m>, Invalid> {
        self.m.check(s)?;
        self.s.copy_from_slice(s);
        Ok(self)
    }

    /// The current state.
    #[must_use]
    pub fn state(&self) -> &[u8] {
        &self.s
    }

    /// Its energy.
    #[must_use]
    pub fn energy(&self) -> f64 {
        self.m.energy_of(&self.s)
    }

    /// The inverse temperature this chain is at.
    #[must_use]
    pub fn beta(&self) -> f64 {
        self.beta
    }

    /// One sweep: every site updated once, in index order.
    ///
    /// A deterministic scan rather than a random site order. Each single-site update leaves the
    /// Boltzmann distribution invariant on its own, so any FIXED composition of them does too. A
    /// stopping rule that depended on the state would not — see
    /// [`crate::cluster::Sampler::with_wolff_steps`] for what that costs when it happens.
    pub fn sweep(&mut self, update: Local, ledger: Option<&mut Ledger>) -> SweepStats {
        let stats = match update {
            Local::Metropolis => self.metropolis_sweep(),
            Local::HeatBath => self.heat_bath_sweep(),
        };
        if let Some(l) = ledger {
            l.samples += stats.updates;
            l.writes += stats.changed;
        }
        stats
    }

    /// `n` sweeps, accumulating the statistics.
    pub fn sweeps(
        &mut self,
        n: usize,
        update: Local,
        mut ledger: Option<&mut Ledger>,
    ) -> SweepStats {
        let mut acc = SweepStats::default();
        for _ in 0..n {
            let st = self.sweep(update, ledger.as_deref_mut());
            acc.updates += st.updates;
            acc.changed += st.changed;
        }
        acc
    }

    fn metropolis_sweep(&mut self) -> SweepStats {
        let m = self.m;
        let beta = self.beta;
        let mut changed = 0u64;
        for i in 0..m.n {
            let a = self.s[i];
            let b = other_state(&mut self.rng, m.q, a);
            let de = m.site_energy(i, b, &self.s) - m.site_energy(i, a, &self.s);
            // `f64()` is in [0,1) and `exp(-beta dE)` is at least 1 for a downhill move, so this one
            // comparison is `min(1, exp(-beta dE))` with no branch on the sign of dE.
            if self.rng.f64() < (-beta * de).exp() {
                self.s[i] = b;
                changed += 1;
            }
        }
        SweepStats { updates: m.n as u64, changed }
    }

    fn heat_bath_sweep(&mut self) -> SweepStats {
        let m = self.m;
        let beta = self.beta;
        let q = m.q;
        let mut changed = 0u64;
        for i in 0..m.n {
            // Log-weights first, shifted by their maximum before exponentiating. Without the shift a
            // cold chain overflows `exp` to infinity and every conditional comes back NaN, which is
            // the defect `ising::exact_boltzmann` carries a paragraph about.
            let mut mx = f64::NEG_INFINITY;
            for a in 0..q {
                let l = -beta * m.site_energy(i, a as u8, &self.s);
                self.scratch[a] = l;
                mx = mx.max(l);
            }
            let mut acc = 0.0;
            for a in 0..q {
                self.scratch[a] = (self.scratch[a] - mx).exp();
                acc += self.scratch[a];
            }
            let u = self.rng.f64() * acc;
            // The last state catches a `u` that rounding pushed past the final cumulative sum. That
            // is arithmetic, not a default standing in for an unreadable input: the shift leaves one
            // weight at exactly `exp(0)`, so `acc` is at least one and the sum is never empty.
            let mut pick = (q - 1) as u8;
            let mut c = 0.0;
            for a in 0..q {
                c += self.scratch[a];
                if u < c {
                    pick = a as u8;
                    break;
                }
            }
            if pick != self.s[i] {
                changed += 1;
            }
            self.s[i] = pick;
        }
        SweepStats { updates: m.n as u64, changed }
    }

    /// Draw a chain: burn in, then keep `draws` states one every `thin` sweeps.
    ///
    /// `reads` is charged here, `n` per kept state — see the type's documentation.
    pub fn collect(&mut self, plan: &Plan, update: Local, mut ledger: Option<&mut Ledger>) -> Run {
        self.sweeps(plan.burn_in, update, ledger.as_deref_mut());
        let thin = plan.thin.max(1);
        let mut states = Vec::with_capacity(plan.draws);
        let mut energies = Vec::with_capacity(plan.draws);
        for _ in 0..plan.draws {
            self.sweeps(thin, update, ledger.as_deref_mut());
            energies.push(self.energy());
            states.push(self.s.clone());
            if let Some(l) = ledger.as_deref_mut() {
                l.reads += self.m.n as u64;
            }
        }
        Run { q: self.m.q, n: self.m.n, states, energies }
    }
}

/// An involution of the single-site states that is a symmetry of the pair term.
///
/// The object the whole cluster construction is built on: `f(Ra, Rb) = f(a, b)`, so applying it to
/// every member of a cluster costs nothing across the cluster's INTERIOR, and the bond probabilities
/// only have to pay for its boundary. See the module documentation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Reflect {
    /// Swap two colours and leave the rest alone. The symmetry of [`Interaction::Potts`], whose
    /// group is all of `S_q`.
    Swap(u8, u8),
    /// `a -> (r - a) mod q`: reflect the circle about the axis at angle `pi r / q`. The symmetry of
    /// [`Interaction::Clock`], whose group is only `Z_q` plus these `q` reflections.
    Mirror(u8),
}

impl Reflect {
    fn apply(self, q: usize, x: u8) -> u8 {
        match self {
            Reflect::Swap(a, b) => {
                if x == a {
                    b
                } else if x == b {
                    a
                } else {
                    x
                }
            }
            Reflect::Mirror(r) => ((usize::from(r) + q - usize::from(x)) % q) as u8,
        }
    }
}

/// Which cluster move to make.
///
/// Both open bonds by the rule in the module documentation and both leave the Boltzmann
/// distribution invariant with acceptance one. They differ in what they build.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cluster {
    /// Swendsen-Wang: decompose the whole lattice into clusters and recolour each independently.
    ///
    /// For [`Interaction::Potts`] each cluster takes a uniformly random state out of all `q`, which
    /// is Swendsen & Wang's original move. For [`Interaction::Clock`] one reflection is drawn for
    /// the whole sweep and each cluster is reflected with probability one half, which is the
    /// Kandel-Domany form — a clock cluster is defined RELATIVE to a reflection, so there is no
    /// colour to draw.
    SwendsenWang,
    /// Wolff: grow one cluster from a random seed and apply the reflection to it with probability
    /// one.
    ///
    /// The seed is uniform over sites, so a cluster is reached in proportion to its size, and that
    /// bias is the point: Wolff spends its work on the large clusters and skips the singletons
    /// Swendsen-Wang pays to enumerate.
    Wolff,
}

/// What one cluster sweep did.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ClusterStats {
    /// Clusters the sweep formed. For [`Cluster::Wolff`], the number of single-cluster steps.
    pub clusters: usize,
    /// Size of the largest, which is the whole reason to run a cluster algorithm.
    pub largest: usize,
    /// Sites whose state changed.
    pub changed: u64,
    /// Bonds whose two endpoints the sweep compared, which is what a sweep actually costs.
    pub bonds_tested: u64,
    /// Sites visited.
    pub visited: u64,
}

/// Cluster sampling of a [`Potts`] model, valid at every `q`.
///
/// Read the module documentation for the construction and for why a field is refused rather than
/// approximated.
pub struct ClusterSampler<'m> {
    m: &'m Potts,
    beta: f64,
    s: Vec<u8>,
    rng: Pcg,
    stack: Vec<usize>,
    in_cluster: Vec<bool>,
    wolff_steps: usize,
}

impl<'m> ClusterSampler<'m> {
    /// A cluster sampler over `m` at inverse temperature `beta`.
    ///
    /// # Errors
    ///
    /// [`NoCluster::Fielded`] naming the first site and state carrying a field, and
    /// [`NoCluster::Antiferromagnetic`] naming the first negative edge of a [`Interaction::Potts`]
    /// model. A [`Interaction::Clock`] model with negative couplings is ACCEPTED: its bond rule is
    /// `1 - exp(-beta max(0, d))`, which is a probability for either sign of `J`, and the reflection
    /// it is built on exists whatever the couplings do. It will not be FAST on a frustrated clock
    /// model — a cluster that never grows is a single-site flip with extra bookkeeping — but it is
    /// correct, and refusing it would refuse a model this samples exactly.
    pub fn new(m: &'m Potts, beta: f64, seed: u64) -> Result<ClusterSampler<'m>, NoCluster> {
        for i in 0..m.n {
            for a in 0..m.q {
                let h = m.h[i * m.q + a];
                if h != 0.0 {
                    return Err(NoCluster::Fielded { site: i, state: a as u8, field: h });
                }
            }
        }
        if m.kind == Interaction::Potts {
            for (i, j, jij) in m.edges() {
                if jij < 0.0 {
                    return Err(NoCluster::Antiferromagnetic { i, j, coupling: jij });
                }
            }
        }
        let mut rng = Pcg::new(seed, POTTS_STREAM ^ 0xC1);
        let s = (0..m.n).map(|_| uniform_state(&mut rng, m.q)).collect();
        Ok(ClusterSampler {
            m,
            beta,
            s,
            rng,
            stack: Vec::new(),
            in_cluster: vec![false; m.n],
            wolff_steps: 1,
        })
    }

    /// How many Wolff steps make one sweep. Default 1.
    ///
    /// A CONSTANT, and it has to be. Stepping "until `n` sites have been visited" makes the number
    /// of steps a function of the cluster sizes, which are a function of the state, and sampling at
    /// a stopping time that depends on the trajectory does not preserve the stationary distribution
    /// however correct the kernel being stopped. [`crate::cluster::Sampler::with_wolff_steps`]
    /// records what that cost when this crate did it: an exact `<E>` of -3.8009 came back as
    /// -4.3262, and the total-variation check saw nothing.
    ///
    /// # Panics
    ///
    /// If `k` is zero: a sweep that takes no steps is not a sweep.
    #[must_use]
    pub fn with_wolff_steps(mut self, k: usize) -> ClusterSampler<'m> {
        assert!(k > 0, "a Wolff sweep of zero steps never moves");
        self.wolff_steps = k;
        self
    }

    /// Start from a given state instead of a random one.
    ///
    /// # Errors
    ///
    /// [`Invalid`], as [`Potts::energy`].
    pub fn from_state(mut self, s: &[u8]) -> Result<ClusterSampler<'m>, Invalid> {
        self.m.check(s)?;
        self.s.copy_from_slice(s);
        Ok(self)
    }

    /// The current state.
    #[must_use]
    pub fn state(&self) -> &[u8] {
        &self.s
    }

    /// Its energy.
    #[must_use]
    pub fn energy(&self) -> f64 {
        self.m.energy_of(&self.s)
    }

    /// The probability that a bond opens across an energy change of `d`, which is
    /// `1 - exp(-beta max(0, d))`.
    ///
    /// Zero when reflecting one endpoint alone would LOWER the energy: such an edge is already
    /// paying nothing to be broken, so joining it into a cluster would be paying twice.
    ///
    /// At `q = 2` on a Potts model this is `1 - exp(-beta J)`, which is
    /// [`crate::cluster`]'s `1 - exp(-2 beta J_ising)` with `J_ising = J/2` — the same number, and
    /// `q2_cluster_sizes_match_the_binary_cluster_module` measures that rather than asserting it.
    #[must_use]
    pub fn bond_probability(&self, d: f64) -> f64 {
        if d <= 0.0 { 0.0 } else { 1.0 - (-self.beta * d).exp() }
    }

    /// The energy change if site `i` ALONE were reflected, counting only the edge at CSR slot `k`.
    fn edge_delta(&self, refl: Reflect, i: usize, k: usize) -> f64 {
        let j = self.m.nbr[k] as usize;
        let (a, b) = (self.s[i], self.s[j]);
        let q = self.m.q;
        -self.m.w[k] * (self.m.kind.pair(q, refl.apply(q, a), b) - self.m.kind.pair(q, a, b))
    }

    /// One sweep with the given move, charged to `ledger`.
    ///
    /// `samples` is charged the sites visited, `writes` the sites changed, and `reads` nothing — a
    /// sweep reads no state out. See [`Sampler`] for the convention.
    pub fn sweep(&mut self, update: Cluster, ledger: Option<&mut Ledger>) -> ClusterStats {
        let stats = match update {
            Cluster::SwendsenWang => self.sw_sweep(),
            Cluster::Wolff => self.wolff_sweep(),
        };
        if let Some(l) = ledger {
            l.samples += stats.visited;
            l.writes += stats.changed;
        }
        stats
    }

    /// `n` sweeps, accumulating the statistics. `largest` is the largest over all of them.
    pub fn sweeps(
        &mut self,
        n: usize,
        update: Cluster,
        mut ledger: Option<&mut Ledger>,
    ) -> ClusterStats {
        let mut acc = ClusterStats::default();
        for _ in 0..n {
            let st = self.sweep(update, ledger.as_deref_mut());
            acc.clusters += st.clusters;
            acc.largest = acc.largest.max(st.largest);
            acc.changed += st.changed;
            acc.bonds_tested += st.bonds_tested;
            acc.visited += st.visited;
        }
        acc
    }

    /// Swendsen-Wang: decompose everything, then recolour each component.
    fn sw_sweep(&mut self) -> ClusterStats {
        let m = self.m;
        let n = m.n;
        let q = m.q;
        let mut uf = Uf::new(n);
        let mut bonds = 0u64;

        // A clock cluster is defined RELATIVE to a reflection, so one is drawn for the whole sweep;
        // a Potts cluster is monochromatic and needs none, because every transposition gives the
        // same bond rule on an aligned edge.
        let refl = match m.kind {
            Interaction::Potts => None,
            Interaction::Clock => Some(Reflect::Mirror(uniform_state(&mut self.rng, q))),
        };

        for i in 0..n {
            for k in m.offset[i]..m.offset[i + 1] {
                let j = m.nbr[k] as usize;
                if j <= i {
                    continue;
                }
                bonds += 1;
                let d = match refl {
                    // Aligned only: a misaligned edge is a domain wall and never joins a cluster.
                    None => {
                        if self.s[i] == self.s[j] {
                            m.w[k]
                        } else {
                            0.0
                        }
                    }
                    Some(r) => self.edge_delta(r, i, k),
                };
                let p = self.bond_probability(d);
                if p > 0.0 && self.rng.f64() < p {
                    uf.union(i, j);
                }
            }
        }

        // One decision per cluster, taken by its root, so every member reads the same one.
        let roots: Vec<usize> = (0..n).map(|v| uf.find(v)).collect();
        let mut decision = vec![u8::MAX; n];
        for v in 0..n {
            let r = roots[v];
            if decision[r] == u8::MAX {
                decision[r] = match refl {
                    None => uniform_state(&mut self.rng, q),
                    Some(_) => u8::from(self.rng.f64() < 0.5),
                };
            }
        }
        let mut changed = 0u64;
        let mut sizes = vec![0usize; n];
        for v in 0..n {
            let r = roots[v];
            sizes[r] += 1;
            let next = match refl {
                None => decision[r],
                Some(rf) => {
                    if decision[r] == 1 {
                        rf.apply(q, self.s[v])
                    } else {
                        self.s[v]
                    }
                }
            };
            if next != self.s[v] {
                changed += 1;
                self.s[v] = next;
            }
        }
        ClusterStats {
            clusters: sizes.iter().filter(|&&c| c > 0).count(),
            largest: sizes.iter().copied().max().unwrap_or(0),
            changed,
            bonds_tested: bonds,
            visited: n as u64,
        }
    }

    /// Wolff: a fixed number of single-cluster steps. See [`ClusterSampler::with_wolff_steps`].
    fn wolff_sweep(&mut self) -> ClusterStats {
        let mut acc = ClusterStats::default();
        for _ in 0..self.wolff_steps {
            let step = self.wolff_step();
            acc.clusters += 1;
            acc.largest = acc.largest.max(step.largest);
            acc.changed += step.changed;
            acc.bonds_tested += step.bonds_tested;
            acc.visited += step.visited;
        }
        acc
    }

    /// One Wolff step: grow a cluster from a random seed and reflect it.
    ///
    /// The reflection is chosen so that the SEED always moves — one of the `q - 1` other colours for
    /// Potts, and one of the `q - 1` reflections that do not fix the seed for the clock model.
    /// Excluding the identity-on-the-seed choice keeps the proposal symmetric (the reverse move
    /// excludes exactly the same one) and means a step never proposes nothing, which is what makes
    /// `a_wolff_step_always_moves_its_seed` a test rather than a coin flip.
    pub fn wolff_step(&mut self) -> ClusterStats {
        let m = self.m;
        let q = m.q;
        // A model with no sites has no seed to draw. An early return rather than a clamp: `min(n-1)`
        // on `n = 0` underflows a `usize`, and `PottsBuilder::new(q, 0, kind)` and
        // `lattice2d(0, ..)` both reach here.
        if m.n == 0 {
            return ClusterStats::default();
        }
        let seed = ((self.rng.f64() * m.n as f64) as usize).min(m.n - 1);
        let a = self.s[seed];
        let refl = match m.kind {
            Interaction::Potts => Reflect::Swap(a, other_state(&mut self.rng, q, a)),
            // `r = 2a mod q` is the one reflection fixing the seed, and it is the one excluded.
            Interaction::Clock => {
                Reflect::Mirror(other_state(&mut self.rng, q, ((2 * usize::from(a)) % q) as u8))
            }
        };

        self.stack.clear();
        self.stack.push(seed);
        self.in_cluster[seed] = true;
        let mut members = vec![seed];
        let mut bonds = 0u64;

        while let Some(i) = self.stack.pop() {
            for k in m.offset[i]..m.offset[i + 1] {
                let j = m.nbr[k] as usize;
                if self.in_cluster[j] {
                    continue;
                }
                bonds += 1;
                let p = self.bond_probability(self.edge_delta(refl, i, k));
                if p > 0.0 && self.rng.f64() < p {
                    self.in_cluster[j] = true;
                    members.push(j);
                    self.stack.push(j);
                }
            }
        }

        let mut changed = 0u64;
        for &v in &members {
            let next = refl.apply(q, self.s[v]);
            if next != self.s[v] {
                changed += 1;
                self.s[v] = next;
            }
            self.in_cluster[v] = false;
        }
        ClusterStats {
            clusters: 1,
            largest: members.len(),
            changed,
            bonds_tested: bonds,
            visited: members.len() as u64,
        }
    }

    /// Draw a chain: burn in, then keep `draws` states one every `thin` sweeps.
    ///
    /// `reads` is charged here, `n` per kept state.
    pub fn collect(&mut self, plan: &Plan, update: Cluster, mut ledger: Option<&mut Ledger>) -> Run {
        self.sweeps(plan.burn_in, update, ledger.as_deref_mut());
        let thin = plan.thin.max(1);
        let mut states = Vec::with_capacity(plan.draws);
        let mut energies = Vec::with_capacity(plan.draws);
        for _ in 0..plan.draws {
            self.sweeps(thin, update, ledger.as_deref_mut());
            energies.push(self.energy());
            states.push(self.s.clone());
            if let Some(l) = ledger.as_deref_mut() {
                l.reads += self.m.n as u64;
            }
        }
        Run { q: self.m.q, n: self.m.n, states, energies }
    }
}

/// Union-find with path halving and union by size. Enough for one sweep's clusters.
struct Uf {
    parent: Vec<usize>,
    size: Vec<usize>,
}

impl Uf {
    fn new(n: usize) -> Uf {
        Uf { parent: (0..n).collect(), size: vec![1; n] }
    }
    fn find(&mut self, mut v: usize) -> usize {
        while self.parent[v] != v {
            self.parent[v] = self.parent[self.parent[v]];
            v = self.parent[v];
        }
        v
    }
    fn union(&mut self, a: usize, b: usize) {
        let (mut ra, mut rb) = (self.find(a), self.find(b));
        if ra == rb {
            return;
        }
        if self.size[ra] < self.size[rb] {
            core::mem::swap(&mut ra, &mut rb);
        }
        self.parent[rb] = ra;
        self.size[ra] += self.size[rb];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The error a `Result` carries, for types whose `Ok` side is too large to want a `Debug` on.
    ///
    /// `expect_err` needs `T: Debug`, and deriving one on [`Enumerated`] would mean printing a
    /// million-entry table into a panic message.
    fn err_of<T, E>(r: Result<T, E>, what: &str) -> E {
        match r {
            Err(e) => e,
            Ok(_) => panic!("{what}"),
        }
    }

    /// A binary model with mixed-sign couplings and asymmetric fields, every parameter a dyadic
    /// rational.
    ///
    /// Dyadic on purpose. Every energy this fixture can produce is a multiple of `1/4` with
    /// magnitude far below `2^52`, so it is exactly representable and EVERY partial sum of such
    /// terms is exact whatever order they are added in. That is what makes the headline test an
    /// equality rather than a tolerance: if the two routes disagreed in the last bit, the map would
    /// be wrong and not merely differently rounded.
    fn binary_fixture(kind: Interaction) -> Potts {
        let n = 10;
        let mut b = PottsBuilder::new(2, n, kind);
        let js = [1.0, -2.0, 3.0, -1.0, 2.0, 4.0, -3.0, 1.0, 2.0, -1.0];
        for i in 0..n {
            b.couple(i, (i + 1) % n, js[i]);
        }
        b.couple(0, 5, 2.0);
        b.couple(2, 7, -1.0);
        let hs = [0.0, 0.5, -1.5, 2.0, 0.5, 0.0, -0.5, 1.0, 1.5, -2.0];
        for i in 0..n {
            b.field(i, 0, hs[i]);
            b.field(i, 1, hs[(i + 3) % n] * 0.5);
        }
        b.build()
    }

    /// THE HEADLINE. `q = 2` is the Ising model this crate already has, state for state, in exact
    /// `f64` — and the additive constant is part of the claim.
    ///
    /// The oracle is [`crate::graph::Graph::energy`], which is not this module's code and is
    /// verified by everything else in the crate. Exhaustive over all `2^10` states, for BOTH
    /// interactions, on a model with mixed-sign couplings and asymmetric per-state fields — the
    /// field is the half that is easy to get wrong and invisible at zero field, because a Potts
    /// field has one number per state and an Ising bias has one per site.
    ///
    /// Asserted with `==`. The fixture is dyadic (see `binary_fixture`), so both routes evaluate
    /// exactly and a tolerance would be hiding something rather than allowing for something.
    ///
    /// **And the constant is asserted to MATTER.** Dropping it is the plausible error — it is
    /// invisible in a ground state and in every energy difference — so the test also checks that
    /// the two energies DISAGREE without it. A test that only checked the map with the constant
    /// present would pass for an implementation that returned a constant of zero on a model whose
    /// true constant is `-4.625`.
    #[test]
    fn q2_potts_is_the_ising_model_state_for_state_oracle_graph_energy() {
        for kind in [Interaction::Potts, Interaction::Clock] {
            let m = binary_fixture(kind);
            let (g, c) = m.to_ising().expect("q = 2 has an Ising form");
            assert_ne!(c, 0.0, "{kind:?}: this fixture must have a non-zero constant to test one");

            let mut disagreed_without_c = 0usize;
            for index in 0..(1usize << m.n()) {
                let s = m.state_at(index);
                let sigma: Vec<i8> = s.iter().map(|&v| if v == 0 { 1i8 } else { -1 }).collect();
                let potts = m.energy(&s).expect("the fixture's own states");
                let ising = g.energy(&sigma);
                assert_eq!(
                    potts,
                    ising + c,
                    "{kind:?}: state {s:?} is {potts} in Potts and {} in Ising",
                    ising + c
                );
                if potts != ising {
                    disagreed_without_c += 1;
                }
            }
            assert_eq!(
                disagreed_without_c,
                1 << m.n(),
                "{kind:?}: the constant must move EVERY energy, or it is not being tested"
            );
        }
    }

    /// The `q = 2` Boltzmann distribution is the Ising one, state for state, against
    /// [`crate::ising::exact_boltzmann`].
    ///
    /// One level up from the energies: a map that was right on energies and wrong on the state
    /// INDEXING would still pass the test above, because that one converts the state itself. Here
    /// the two distributions are built independently — `enumerate` counts in `q`-ary with site 0
    /// least significant, `exact_boltzmann` counts in binary with bit `b` set meaning spin `+1` —
    /// and the permutation between them is the thing under test.
    #[test]
    fn q2_boltzmann_matches_ising_exact_boltzmann_oracle_ising_rs() {
        let m = binary_fixture(Interaction::Potts);
        let (g, _) = m.to_ising().unwrap();
        for beta in [0.15f64, 0.7, 2.0] {
            let mine = enumerate(&m, beta).expect("2^10 states");
            let theirs = crate::ising::exact_boltzmann(&g, beta);
            let mut worst = 0.0f64;
            for index in 0..(1usize << m.n()) {
                let s = m.state_at(index);
                let mut mask = 0usize;
                for (bit, &v) in s.iter().enumerate() {
                    if v == 0 {
                        mask |= 1 << bit;
                    }
                }
                worst = worst.max((mine.p[index] - theirs[mask]).abs());
            }
            // Not exact, and the reason is arithmetic rather than the map: both routes normalise by
            // summing 1024 exponentials, in different orders, so the two agree to about
            // `1024 * eps * max(p)` = 1.1e-13. Measured worst over the three temperatures: 1.0e-14
            // at beta = 2, where the distribution is most peaked and `max(p)` largest. The ENERGIES
            // are exact — that is the test above — and only the normalisation is not.
            assert!(worst < 1e-13, "beta {beta}: distributions differ by {worst:e}");
        }
    }

    /// `log Z` matches [`crate::exact`]'s variable elimination, shifted by the constant.
    ///
    /// `Z_potts = exp(-beta c) Z_ising`, so `log Z_potts = log Z_ising - beta c`. Elimination is a
    /// completely different algorithm from enumeration — it never visits a state — so this checks
    /// the partition function against something that shares no code with it.
    ///
    /// The second half is the asymmetric one: the UNSHIFTED comparison must FAIL, by exactly
    /// `beta |c|`. Free energy is where dropping the constant actually bites, and a test that only
    /// checked the shifted form would pass for an implementation that returned `c = 0`.
    #[test]
    fn q2_log_partition_matches_variable_elimination_oracle_exact_rs() {
        let m = binary_fixture(Interaction::Potts);
        let (g, c) = m.to_ising().unwrap();
        for beta in [0.2f64, 0.7, 1.5] {
            let mine = enumerate(&m, beta).unwrap();
            let theirs = crate::exact::Elimination::default()
                .log_partition(&g, beta)
                .expect("a 10-spin ring is narrow")
                .log_z
                .expect("log_partition returns one");
            assert!(
                (mine.log_z - (theirs - beta * c)).abs() < 1e-12,
                "beta {beta}: {} against {}",
                mine.log_z,
                theirs - beta * c
            );
            assert!(
                (mine.log_z - theirs).abs() > 0.5 * beta * c.abs(),
                "beta {beta}: dropping the constant must be visible in log Z"
            );
        }
    }

    /// The clock model IS the Potts model at `q <= 3`, exactly, and is NOT at `q = 4`.
    ///
    /// Both halves matter. `cos` takes only two distinct values when there are at most three states
    /// — `1` and `-1` at `q = 2`, `1` and `-1/2` at `q = 3` — so the pair term is an affine function
    /// of `delta` and the two models differ by a rescaling of `J` and a constant:
    /// `f_clock = 2 delta - 1` at `q = 2` and `(3 delta - 1)/2` at `q = 3`. Asserted with `==`, over
    /// every one of the `q^7` states, which the exact values in [`cos_step`] are what make possible.
    ///
    /// At `q = 4` the clock term takes THREE values, `1`, `0` and `-1`, where Potts takes two. No
    /// affine map sends `{1, 0, 0}` to `{1, 0, -1}`, so no rescaling of `J` and no constant can
    /// reconcile them — and the test exhibits the contradiction rather than asserting it. Without
    /// this half, an [`Interaction::Clock`] that had silently been implemented as Potts would pass
    /// every other test in this module at `q = 2` and `q = 3`.
    #[test]
    fn the_clock_and_potts_models_coincide_at_q3_and_part_at_q4() {
        for (q, scale, offset) in [(2usize, 2.0f64, 1.0f64), (3, 1.5, 0.5)] {
            let n = 7;
            let js = [2.0, -4.0, 2.0, 6.0, -2.0, 4.0, 2.0];
            let mut bc = PottsBuilder::new(q, n, Interaction::Clock);
            let mut bp = PottsBuilder::new(q, n, Interaction::Potts);
            for i in 0..n {
                bc.couple(i, (i + 1) % n, js[i]);
                bp.couple(i, (i + 1) % n, js[i] * scale);
            }
            bc.couple(0, 3, 2.0);
            bp.couple(0, 3, 2.0 * scale);
            let (clock, potts) = (bc.build(), bp.build());
            let c: f64 = clock.edges().map(|(_, _, j)| j).sum::<f64>() * offset;
            for index in 0..q.pow(n as u32) {
                let s = clock.state_at(index);
                assert_eq!(
                    clock.energy(&s).unwrap(),
                    potts.energy(&s).unwrap() + c,
                    "q = {q}: state {s:?}"
                );
            }
        }

        // q = 4: f_clock takes three values where f_potts takes two, so `f_clock = a f_potts + b`
        // would need b = 0 (from the distance-1 pair) and b = -1 (from the distance-2 pair).
        assert_eq!(Interaction::Potts.pair(4, 0, 1), 0.0);
        assert_eq!(Interaction::Potts.pair(4, 0, 2), 0.0);
        assert_eq!(Interaction::Clock.pair(4, 0, 1), 0.0);
        assert_eq!(Interaction::Clock.pair(4, 0, 2), -1.0);

        // And the same contradiction at the level of whole energies: three states whose
        // (E_potts, E_clock) pairs are not collinear, so no affine map relates the two models.
        let (clock, potts) = (ring(4, 4, 1.0, Interaction::Clock), ring(4, 4, 1.0, Interaction::Potts));
        let pts: Vec<(f64, f64)> = [[0u8, 0, 0, 0], [0, 1, 0, 1], [0, 2, 0, 2]]
            .iter()
            .map(|s| (potts.energy(s).unwrap(), clock.energy(s).unwrap()))
            .collect();
        let area = (pts[1].0 - pts[0].0) * (pts[2].1 - pts[0].1)
            - (pts[2].0 - pts[0].0) * (pts[1].1 - pts[0].1);
        assert!(area.abs() > 1e-9, "at q = 4 the two models were collinear: {pts:?}");
    }

    /// A model small enough to enumerate, its exact distribution, and the noise floor a
    /// total-variation comparison is allowed to sit under.
    ///
    /// The floor is `0.5 sqrt(K / ess)`, the expected total variation between a distribution over
    /// `K` states and `ess` independent draws of it. It is asserted to be BELOW ONE, which is the
    /// lesson `crate::cluster` records: total variation between two distributions can never exceed
    /// one, so a floor at or above one passes for every sampler including a constant.
    fn score_against_enumeration(run: &Run, ex: &Enumerated, label: &str) {
        let e = run.mean_energy();
        let z = (e.value - ex.mean_energy).abs() / e.stderr;
        assert!(
            z < 3.0,
            "{label}: <E> = {} +- {} against an exact {} — {z:.2} standard errors out",
            e.value,
            e.stderr,
            ex.mean_energy
        );
        let hist = run.histogram().expect("small enough to enumerate");
        let tv = crate::ising::tv(&hist, &ex.p);
        let floor = 0.5 * (ex.p.len() as f64 / e.ess).sqrt();
        assert!(floor < 1.0, "{label}: a noise floor of {floor:.2} passes every sampler alive");
        assert!(tv < floor, "{label}: total variation {tv:.4} above its noise floor {floor:.4}");
    }

    /// Both single-site updates reproduce the exact Boltzmann distribution, at several `q`, under
    /// both interactions. Oracle: [`enumerate`], over all `q^n` states.
    ///
    /// This is the claim a sampler has to earn before any of its numbers mean anything, and it is
    /// scored two ways: the mean energy against its own error bar, and the whole distribution
    /// against a noise floor that is itself asserted to be informative.
    #[test]
    fn local_updates_reproduce_exhaustive_enumeration_oracle_enumerate() {
        for kind in [Interaction::Potts, Interaction::Clock] {
            for (n, q) in [(6usize, 3usize), (5, 4)] {
                let m = ring(n, q, 1.0, kind);
                let beta = 0.5;
                let ex = enumerate(&m, beta).unwrap();
                for update in [Local::Metropolis, Local::HeatBath] {
                    let mut sm = Sampler::new(&m, beta, 7);
                    let run = sm.collect(&Plan::new(1000, 20000, 2), update, None);
                    score_against_enumeration(&run, &ex, &format!("{kind:?} q={q} {update:?}"));
                }
            }
        }
    }

    /// Both cluster updates reproduce the exact Boltzmann distribution. Oracle: [`enumerate`].
    ///
    /// The claim a CLUSTER algorithm has to earn: it moves a whole correlated region in one move and
    /// must still leave the same distribution invariant. The bond probability is the only number
    /// that makes that true, and it is the one this test is really about — see the module
    /// documentation for why the binary factor of two does not survive to `q > 2`.
    #[test]
    fn cluster_updates_reproduce_exhaustive_enumeration_oracle_enumerate() {
        for kind in [Interaction::Potts, Interaction::Clock] {
            for (n, q) in [(6usize, 3usize), (5, 4)] {
                let m = ring(n, q, 1.0, kind);
                let beta = 0.5;
                let ex = enumerate(&m, beta).unwrap();
                for update in [Cluster::SwendsenWang, Cluster::Wolff] {
                    let mut sm = ClusterSampler::new(&m, beta, 7).unwrap().with_wolff_steps(3);
                    let run = sm.collect(&Plan::new(1000, 20000, 2), update, None);
                    score_against_enumeration(&run, &ex, &format!("{kind:?} q={q} {update:?}"));
                }
            }
        }
    }

    /// A frustrated CLOCK model is sampled; a frustrated POTTS model is refused, naming the edge.
    ///
    /// The asymmetry is the physics. The clock cluster move is built on a reflection, and
    /// `1 - exp(-beta max(0, d))` is a probability whatever the sign of `J`, so mixed couplings are
    /// valid there — and the test proves it by scoring against enumeration rather than by asserting
    /// that nothing was returned. The Potts move is built on the Fortuin-Kasteleyn bond
    /// `1 - exp(-beta J)`, which is not a probability for `J < 0`, and above `q = 2` there is no
    /// gauge to repair it, so it is refused with the offending edge and its coupling.
    ///
    /// Without the refusal half, an implementation that clamped a negative probability to zero would
    /// pass every distribution test on ferromagnets and silently sample the wrong model here.
    #[test]
    fn a_frustrated_clock_model_is_sampled_and_a_frustrated_potts_model_is_refused() {
        let mut b = PottsBuilder::new(4, 6, Interaction::Clock);
        let js = [1.0, -1.0, 1.0, 1.0, -1.0, 1.0];
        for i in 0..6 {
            b.couple(i, (i + 1) % 6, js[i]);
        }
        b.couple(0, 3, -1.0);
        let clock = b.build();
        let beta = 0.6;
        let ex = enumerate(&clock, beta).unwrap();
        for update in [Cluster::SwendsenWang, Cluster::Wolff] {
            let mut sm = ClusterSampler::new(&clock, beta, 5)
                .expect("a clock model's reflection exists whatever the couplings do")
                .with_wolff_steps(3);
            let run = sm.collect(&Plan::new(1000, 20000, 2), update, None);
            score_against_enumeration(&run, &ex, &format!("frustrated clock {update:?}"));
        }

        let mut b = PottsBuilder::new(4, 6, Interaction::Potts);
        for i in 0..6 {
            b.couple(i, (i + 1) % 6, js[i]);
        }
        let potts = b.build();
        let err = err_of(
            ClusterSampler::new(&potts, beta, 5),
            "a q = 4 antiferromagnetic bond has no cluster move",
        );
        match err {
            NoCluster::Antiferromagnetic { i, j, coupling } => {
                assert_eq!((i, j, coupling), (1, 2, -1.0), "the refusal must name the real edge");
            }
            other => panic!("wrong refusal: {other}"),
        }
    }

    /// The square-lattice transition brackets `ln(1 + sqrt(q))`, scored against Baxter's exact
    /// critical bond energy `(1 + 1/sqrt(q))/2`.
    ///
    /// Two closed forms at once, and neither comes from this module: `beta_c` says WHERE the
    /// transition is and the critical energy says WHAT the model's energy is there. The measurement
    /// straddles the second by moving `beta` either side of the first.
    ///
    /// The window is `0.04` and the module documentation carries the measurement that sets it: a
    /// finite lattice's pseudo-critical point sits BELOW `beta_c` by about `0.16 / L`, so a window
    /// narrower than that offset cannot bracket. The second half of this test asserts that failure
    /// at `L = 8`, where the offset is `-0.025`: at `beta_c - 0.01` an eight-by-eight lattice is
    /// still on the ORDERED side of its own transition and its bond energy comes back ABOVE the
    /// exact critical value. A bracket that held at every width and every size would be measuring
    /// the tolerance rather than the physics.
    #[test]
    fn the_square_lattice_transition_brackets_ln_one_plus_sqrt_q_oracle_baxter_critical_energy() {
        let bond_energy = |l: usize, q: usize, beta: f64, seed: u64| -> Estimate {
            let m = lattice2d(l, q, 1.0, Interaction::Potts);
            let mut sm = ClusterSampler::new(&m, beta, seed).unwrap();
            let run = sm.collect(&Plan::new(800, 2500, 1), Cluster::SwendsenWang, None);
            let nb = m.n_edges() as f64;
            run.expectation(move |s| -m.energy(s).unwrap() / nb)
        };

        for q in [2usize, 3, 4] {
            let bc = critical_beta(q);
            let exact = critical_bond_energy(q);
            let lo = bond_energy(16, q, bc - 0.04, 20);
            let hi = bond_energy(16, q, bc + 0.04, 20);
            assert!(
                lo.value + 5.0 * lo.stderr < exact,
                "q = {q}: below beta_c the bond energy is {lo}, not under the exact {exact}"
            );
            assert!(
                hi.value - 5.0 * hi.stderr > exact,
                "q = {q}: above beta_c the bond energy is {hi}, not over the exact {exact}"
            );
        }

        // The finite-size shift, made visible: too narrow a window at too small a lattice does NOT
        // bracket, and the direction is the one the scaling table predicts.
        let exact = critical_bond_energy(2);
        let narrow = bond_energy(8, 2, critical_beta(2) - 0.01, 20);
        assert!(
            narrow.value - 3.0 * narrow.stderr > exact,
            "an 8x8 lattice 0.01 below beta_c should still read ABOVE the exact {exact}, \
             since its own transition sits 0.025 lower; got {narrow}"
        );
    }

    /// At `q = 2` the clusters this module builds are the ones [`crate::cluster`] builds.
    ///
    /// The bond probabilities are written differently — `1 - exp(-beta J_potts)` here against
    /// `1 - exp(-2 beta J_ising)` there — and `J_ising = J_potts / 2` makes them the same number.
    /// That identity is the single most plausible thing to get wrong when generalising the binary
    /// case, so it is measured against the binary module rather than asserted: the mean largest
    /// cluster per Swendsen-Wang sweep, at `beta_c`, on the same 12x12 lattice.
    ///
    /// The test also fixes what "the same" is worth by showing what a factor of two would look like:
    /// the same Ising sampler at HALF that `beta` has a largest cluster ten times smaller. So the 6%
    /// agreement demanded below is a real constraint and not a band anything would fall in.
    #[test]
    fn q2_cluster_sizes_match_the_binary_cluster_module() {
        let beta = critical_beta(2);
        let m = lattice2d(12, 2, 1.0, Interaction::Potts);
        let (g, _) = m.to_ising().unwrap();

        let mut mine = ClusterSampler::new(&m, beta, 3).unwrap();
        let mut mine_sum = 0.0;
        for _ in 0..600 {
            mine_sum += mine.sweep(Cluster::SwendsenWang, None).largest as f64;
        }
        let mine_avg = mine_sum / 600.0;

        let binary_avg = |b: f64| -> f64 {
            let mut s = crate::cluster::Sampler::new(&g, b, 3).unwrap();
            let mut acc = 0.0;
            for _ in 0..600 {
                acc += s.sweep_with(crate::cluster::Update::SwendsenWang, None).largest as f64;
            }
            acc / 600.0
        };
        let matched = binary_avg(beta);
        let halved = binary_avg(beta / 2.0);

        assert!(
            (mine_avg - matched).abs() / matched < 0.06,
            "q = 2 clusters average {mine_avg:.1} here and {matched:.1} in cluster.rs"
        );
        assert!(
            matched > 5.0 * halved,
            "the comparison is only worth making if beta matters: {matched:.1} against \
             {halved:.1} at half beta"
        );
    }

    /// A field is refused, naming the site, the state and the field.
    ///
    /// Not approximated and not ignored. The cluster move is free across a cluster only because the
    /// reflection is a symmetry of the pair term, and a field is by definition not symmetric under
    /// it; a sampler that ran anyway would sample a model nobody wrote.
    #[test]
    fn the_cluster_sampler_refuses_a_field_naming_the_site() {
        let mut b = PottsBuilder::new(3, 5, Interaction::Potts);
        for i in 0..5 {
            b.couple(i, (i + 1) % 5, 1.0);
        }
        b.field(3, 2, -0.75);
        let m = b.build();
        assert!(m.has_field());
        match err_of(ClusterSampler::new(&m, 0.5, 1), "a field has no cluster move") {
            NoCluster::Fielded { site, state, field } => {
                assert_eq!((site, state, field), (3, 2, -0.75));
            }
            other => panic!("wrong refusal: {other}"),
        }
        // The same model is perfectly samplable one site at a time, which is what makes the refusal
        // a statement about the MOVE rather than about the model.
        let ex = enumerate(&m, 0.5).unwrap();
        let mut sm = Sampler::new(&m, 0.5, 2);
        let run = sm.collect(&Plan::new(1000, 20000, 2), Local::HeatBath, None);
        score_against_enumeration(&run, &ex, "fielded model, heat bath");
    }

    /// A state this model cannot read is an error naming what was seen, not a value modulo `q`.
    #[test]
    fn energy_refuses_a_state_this_model_cannot_read() {
        let m = ring(4, 3, 1.0, Interaction::Potts);
        assert_eq!(m.energy(&[0, 1, 2]), Err(Invalid::Width { got: 3, want: 4 }));
        assert_eq!(
            m.energy(&[0, 1, 3, 2]),
            Err(Invalid::State { site: 2, value: 3, q: 3 })
        );
        // State 3 must not be read as state 0. If it were, this would be the energy of [0,1,0,2].
        let aliased = m.energy(&[0, 1, 0, 2]).unwrap();
        assert!(m.energy(&[0, 1, 3, 2]).is_err(), "state 3 was accepted as {aliased}");
        assert!(m.index_of(&[0, 1, 3, 2]).is_err());
        assert!(Sampler::new(&m, 0.5, 1).from_state(&[0, 1, 3, 2]).is_err());
        // And the message says what it saw.
        let msg = Invalid::State { site: 2, value: 3, q: 3 }.to_string();
        assert!(msg.contains("site 2") && msg.contains("state 3") && msg.contains("3 states"), "{msg}");
    }

    /// A model with more than two states has no Ising form, and says so.
    #[test]
    fn to_ising_refuses_a_model_that_is_not_binary() {
        for q in [3usize, 5] {
            for kind in [Interaction::Potts, Interaction::Clock] {
                let m = ring(4, q, 1.0, kind);
                assert_eq!(err_of(m.to_ising(), "q > 2 has no Ising form"), NotBinary { q, kind });
            }
        }
        assert!(ring(4, 2, 1.0, Interaction::Potts).to_ising().is_ok());
    }

    /// A Wolff step always moves its seed, at every `q`, under both interactions.
    ///
    /// The distribution tests cannot see this. A cluster move that applied its reflection with
    /// probability one half is still a valid kernel — a lazy one — leaving the same distribution
    /// invariant and simply mixing half as fast, so it survives every check on the distribution
    /// while quietly halving the algorithm's reason to exist. This is the claim stated exactly: the
    /// seed's state is different afterwards, every single step, and `changed` reports the truth.
    #[test]
    fn a_wolff_step_always_moves_its_seed() {
        for kind in [Interaction::Potts, Interaction::Clock] {
            for q in [2usize, 3, 4, 5] {
                let m = lattice2d(5, q, 1.0, kind);
                for beta in [0.05f64, 1.0, 3.0] {
                    let mut sm = ClusterSampler::new(&m, beta, 8).unwrap();
                    for step in 0..120 {
                        let before = sm.state().to_vec();
                        let st = sm.wolff_step();
                        let after = sm.state();
                        let moved = before.iter().zip(after).filter(|(a, b)| a != b).count();
                        assert_eq!(
                            moved as u64, st.changed,
                            "{kind:?} q={q} beta={beta} step {step}: reported {} and made {moved}",
                            st.changed
                        );
                        assert!(
                            st.changed > 0,
                            "{kind:?} q={q} beta={beta} step {step}: a Wolff step declined to move"
                        );
                    }
                }
            }
        }
    }

    /// A cold sweep moves a spanning region; a hot one does not.
    ///
    /// Both halves, because each alone is satisfied by something useless. Without the cold half the
    /// sampler could be single-site Metropolis in disguise — correct, and pointless. Without the hot
    /// half a sampler that flipped everything every sweep would pass, and that one is not even
    /// correct: at `beta` near zero there is no correlation to exploit and a cluster algorithm
    /// SHOULD degenerate to single-site flips.
    #[test]
    fn a_cold_cluster_sweep_spans_and_a_hot_one_does_not() {
        for kind in [Interaction::Potts, Interaction::Clock] {
            for q in [2usize, 3] {
                let m = lattice2d(8, q, 1.0, kind);
                for update in [Cluster::SwendsenWang, Cluster::Wolff] {
                    let mut cold = ClusterSampler::new(&m, 2.0, 3).unwrap();
                    let biggest = cold.sweeps(20, update, None).largest;
                    assert!(
                        biggest > m.n() / 2,
                        "{kind:?} q={q} {update:?}: a cold ferromagnet should move a spanning \
                         cluster; got {biggest} of {}",
                        m.n()
                    );
                    let mut hot = ClusterSampler::new(&m, 0.005, 5).unwrap();
                    let tiny = hot.sweeps(20, update, None).largest;
                    assert!(
                        tiny < m.n() / 4,
                        "{kind:?} q={q} {update:?}: at beta 0.005 clusters should be singletons; \
                         got {tiny}"
                    );
                }
            }
        }
    }

    /// The pair term is symmetric bit for bit, periodic, and exactly the dyadic rationals where the
    /// mathematics says so.
    ///
    /// Symmetry has to be EXACT rather than close: the CSR stores each edge from both ends, so an
    /// asymmetric pair term would make an energy depend on which end a loop happened to visit first,
    /// and the discrepancy would be one ulp — invisible everywhere except in the exact-equality
    /// tests this module's oracles are built on.
    #[test]
    fn the_pair_term_is_symmetric_bit_for_bit_and_exact_at_the_dyadic_angles() {
        for kind in [Interaction::Potts, Interaction::Clock] {
            for q in 2usize..=16 {
                for a in 0..q {
                    for b in 0..q {
                        let (x, y) = (a as u8, b as u8);
                        assert_eq!(
                            kind.pair(q, x, y),
                            kind.pair(q, y, x),
                            "{kind:?} q={q}: pair({a},{b}) is not symmetric"
                        );
                        assert!(kind.pair(q, x, y).abs() <= 1.0);
                    }
                }
            }
        }
        // The values the closed form makes exact: cos of 0, pi/3, pi/2, 2pi/3 and pi.
        assert_eq!(Interaction::Clock.pair(5, 2, 2), 1.0);
        assert_eq!(Interaction::Clock.pair(2, 0, 1), -1.0);
        assert_eq!(Interaction::Clock.pair(4, 0, 1), 0.0);
        assert_eq!(Interaction::Clock.pair(3, 0, 1), -0.5);
        assert_eq!(Interaction::Clock.pair(6, 0, 1), 0.5);
        assert_eq!(Interaction::Clock.pair(6, 0, 3), -1.0);
        // The span follows the table rather than a guess: 2 at even q, 1 + cos(pi/q) at odd.
        assert_eq!(Interaction::Potts.span(7), 1.0);
        assert_eq!(Interaction::Clock.span(4), 2.0);
        assert_eq!(Interaction::Clock.span(3), 1.5);
        assert!((Interaction::Clock.span(5) - (1.0 + (core::f64::consts::PI / 5.0).cos())).abs() < 1e-15);
    }

    /// The single-site gap is an UPPER BOUND on every single-site move, and on these instances it is
    /// attained.
    ///
    /// Checked exhaustively: every state, every site, every state that site could move to. Both
    /// halves are asserted, because each alone is weak. "Never below" alone is satisfied by
    /// returning infinity; "close to the truth" alone is satisfied by a value that is sometimes
    /// under it, which is precisely the defect [`crate::round`] exists for — this crate shipped a
    /// lower bound that sat above the optimum by 7.8e-14 and nothing noticed.
    #[test]
    fn the_single_site_gap_is_an_upper_bound_and_is_attained() {
        for kind in [Interaction::Potts, Interaction::Clock] {
            for q in [2usize, 3, 5] {
                let m = ring(6, q, 1.0, kind);
                let bound = m.single_site_gap_max().expect("a coupled ring has an energy scale");
                let mut worst = 0.0f64;
                for index in 0..q.pow(6) {
                    let s = m.state_at(index);
                    let e = m.energy(&s).unwrap();
                    for i in 0..6 {
                        let mut t = s.clone();
                        for a in 0..q {
                            t[i] = a as u8;
                            worst = worst.max((m.energy(&t).unwrap() - e).abs());
                        }
                    }
                }
                assert!(bound >= worst, "{kind:?} q={q}: bound {bound} is BELOW the true {worst}");
                assert!(
                    bound - worst < 1e-12,
                    "{kind:?} q={q}: bound {bound} is loose against the true {worst}"
                );
            }
        }
        // A model with nothing in it has no energy scale, and says so rather than returning zero.
        assert_eq!(PottsBuilder::new(3, 4, Interaction::Potts).build().single_site_gap_max(), None);
    }

    /// The same seed gives the same chain, and different seeds do not.
    ///
    /// Determinism is this crate's second invariant. The second half is what stops a sampler that
    /// ignores its seed from passing the first.
    #[test]
    fn a_seeded_run_is_reproducible_and_two_seeds_are_not() {
        let m = lattice2d(4, 3, 1.0, Interaction::Potts);
        for update in [Local::Metropolis, Local::HeatBath] {
            let trace = |seed: u64| {
                let mut s = Sampler::new(&m, 0.6, seed);
                (0..50).map(|_| { s.sweep(update, None); s.energy() }).collect::<Vec<_>>()
            };
            assert_eq!(trace(4), trace(4), "{update:?} is not reproducible");
            assert_ne!(trace(4), trace(5), "{update:?} ignores its seed");
        }
        for update in [Cluster::SwendsenWang, Cluster::Wolff] {
            let trace = |seed: u64| {
                let mut s = ClusterSampler::new(&m, 0.6, seed).unwrap();
                (0..50).map(|_| { s.sweep(update, None); s.energy() }).collect::<Vec<_>>()
            };
            assert_eq!(trace(4), trace(4), "{update:?} is not reproducible");
            assert_ne!(trace(4), trace(5), "{update:?} ignores its seed");
        }
    }

    /// The ledger is billed exactly what the sweeps report, and a readout costs what a readout
    /// costs.
    ///
    /// The bill is not derived from a sweep count, because a sweep is a convention and the local and
    /// cluster moves do not mean the same thing by it. `samples`, `writes` and `reads` are
    /// measurements of what happened.
    #[test]
    fn the_ledger_is_billed_exactly_what_the_sweeps_report() {
        let m = lattice2d(6, 3, 1.0, Interaction::Potts);
        for update in [Local::Metropolis, Local::HeatBath] {
            let mut s = Sampler::new(&m, 0.5, 2);
            let mut l = Ledger::default();
            let st = s.sweeps(20, update, Some(&mut l));
            assert_eq!((l.samples, l.writes, l.reads), (st.updates, st.changed, 0));
            assert_eq!(st.updates, 20 * m.n() as u64);
            assert!(st.changed > 0, "{update:?} never moved anything");
        }
        for update in [Cluster::SwendsenWang, Cluster::Wolff] {
            let mut s = ClusterSampler::new(&m, 0.5, 2).unwrap();
            let mut l = Ledger::default();
            let st = s.sweeps(20, update, Some(&mut l));
            assert_eq!((l.samples, l.writes, l.reads), (st.visited, st.changed, 0));
            assert!(st.bonds_tested > 0, "{update:?} tested bonds and did not count them");
        }
        // A readout is charged per site per kept state, which is this crate's eighth invariant.
        let mut s = Sampler::new(&m, 0.5, 2);
        let mut l = Ledger::default();
        let run = s.collect(&Plan::new(5, 30, 2), Local::HeatBath, Some(&mut l));
        assert_eq!(l.reads, 30 * m.n() as u64);
        assert_eq!(l.samples, (5 + 30 * 2) * m.n() as u64);
        assert_eq!(run.len(), 30);
    }

    /// A Wolff sweep takes the number of steps it was told to, and a Swendsen-Wang sweep visits
    /// every site.
    ///
    /// The step count is a constant and this is what pins it. If it ever became a function of the
    /// clusters the sweep happened to build — the optional-stopping defect
    /// [`crate::cluster::Sampler::with_wolff_steps`] records — `clusters` would stop equalling `k`
    /// and this would say so.
    #[test]
    fn a_sweep_takes_the_number_of_steps_it_was_told_to() {
        let m = lattice2d(6, 3, 1.0, Interaction::Potts);
        let mut sw = ClusterSampler::new(&m, 0.9, 2).unwrap();
        for _ in 0..5 {
            assert_eq!(sw.sweep(Cluster::SwendsenWang, None).visited, m.n() as u64);
        }
        for k in [1usize, 4, 9] {
            let mut c = ClusterSampler::new(&m, 0.9, 2).unwrap().with_wolff_steps(k);
            for _ in 0..5 {
                assert_eq!(c.sweep(Cluster::Wolff, None).clusters, k, "at k = {k}");
            }
        }
    }

    /// An enumeration that cannot run says so, with the numbers in hand.
    #[test]
    fn enumerate_refuses_a_model_it_cannot_hold() {
        let m = ring(40, 3, 1.0, Interaction::Potts);
        let err = err_of(enumerate(&m, 0.5), "3^40 is not enumerable");
        assert_eq!(err.q, 3);
        assert_eq!(err.n, 40);
        assert_eq!(err.limit, ENUMERATION_LIMIT);
        assert!(err.to_string().contains("above the enumeration limit"), "{err}");
        // The boundary itself: 2^20 is allowed and 2^21 is not.
        assert!(enumerate(&ring(20, 2, 1.0, Interaction::Potts), 0.5).is_ok());
        assert!(enumerate(&ring(21, 2, 1.0, Interaction::Potts), 0.5).is_err());
    }

    /// The order parameter is zero when the states are spread evenly and one when they agree.
    ///
    /// And it reads as a probability of agreement in between, which is what makes it comparable
    /// across `q` — `max_a n_a / n` alone is `1/q` for a random state and so would report a
    /// different "disordered" value at every `q`.
    #[test]
    fn the_order_parameter_is_zero_when_spread_and_one_when_aligned() {
        let m = ring(6, 3, 1.0, Interaction::Potts);
        assert_eq!(m.order_parameter(&[0, 0, 0, 0, 0, 0]).unwrap(), 1.0);
        assert_eq!(m.order_parameter(&[0, 1, 2, 0, 1, 2]).unwrap(), 0.0);
        let half = m.order_parameter(&[0, 0, 0, 0, 1, 2]).unwrap();
        assert!((half - 0.5).abs() < 1e-15, "{half}");
        assert!(m.order_parameter(&[0, 1, 2, 3, 0, 1]).is_err());
    }

    /// A model with no sites is handled by every entry point rather than crashing in one of them.
    ///
    /// Reachable: `lattice2d(0, ..)` and `lattice2d(1, ..)` both return uncoupled models, following
    /// [`crate::ising::lattice2d`], and `PottsBuilder::new(q, 0, kind)` is an empty model outright.
    /// The Wolff seed is the sharp edge — `min(n - 1)` on `n = 0` underflows a `usize` — so it is
    /// the one asserted here.
    #[test]
    fn an_empty_model_is_a_no_op_everywhere_rather_than_a_panic() {
        for n in [0usize, 1] {
            let m = PottsBuilder::new(3, n, Interaction::Potts).build();
            assert_eq!(m.energy(&vec![0u8; n]).unwrap(), 0.0);
            assert_eq!(m.single_site_gap_max(), None, "nothing to couple is no energy scale");
            for update in [Local::Metropolis, Local::HeatBath] {
                let mut s = Sampler::new(&m, 0.5, 1);
                let st = s.sweeps(3, update, None);
                assert_eq!(st.updates, 3 * n as u64);
                assert_eq!(s.energy(), 0.0);
            }
            let mut c = ClusterSampler::new(&m, 0.5, 1).unwrap();
            for update in [Cluster::SwendsenWang, Cluster::Wolff] {
                let st = c.sweeps(3, update, None);
                assert_eq!(st.bonds_tested, 0, "an uncoupled model has no bonds to test");
                assert_eq!(st.visited, 3 * n as u64);
                assert_eq!(c.energy(), 0.0);
            }
        }
        assert_eq!(lattice2d(1, 3, 1.0, Interaction::Potts).n_edges(), 0);
        assert_eq!(lattice2d(0, 3, 1.0, Interaction::Potts).n(), 0);
    }

    /// A state index round-trips, and `enumerate` counts in the order `index_of` says it does.
    #[test]
    fn a_state_index_round_trips_in_the_order_enumerate_counts_in() {
        let m = ring(5, 3, 1.0, Interaction::Potts);
        for index in 0..3usize.pow(5) {
            let s = m.state_at(index);
            assert_eq!(m.index_of(&s).unwrap(), index);
        }
        let ex = enumerate(&m, 0.4).unwrap();
        for index in [0usize, 1, 17, 100, 242] {
            let s = m.state_at(index);
            assert_eq!(ex.energies[index], m.energy(&s).unwrap());
        }
        // The exact mean and the table agree, which pins `Enumerated::mean`'s own decoding.
        let by_table: f64 = ex.p.iter().zip(&ex.energies).map(|(p, e)| p * e).sum();
        let by_mean = ex.mean(|s| m.energy(s).unwrap());
        assert!((by_table - ex.mean_energy).abs() < 1e-12);
        assert!((by_mean - ex.mean_energy).abs() < 1e-12, "{by_mean} against {}", ex.mean_energy);
    }
}
