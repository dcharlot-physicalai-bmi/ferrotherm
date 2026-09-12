//! The quantum approximate optimization algorithm, by exact state-vector simulation.
//!
//! Farhi, Goldstone & Gutmann, *A Quantum Approximate Optimization Algorithm*, arXiv:1411.4028
//! (2014). The `p`-level ansatz alternates two non-commuting evolutions on the uniform
//! superposition,
//!
//! ```text
//!   |gamma, beta> = e^{-i beta_p B} e^{-i gamma_p C} ... e^{-i beta_1 B} e^{-i gamma_1 C} |+>^n
//! ```
//!
//! with `C` the Ising cost Hamiltonian — diagonal in the computational basis, so it does nothing
//! but attach a phase `e^{-i gamma E(s)}` to each basis state — and `B = sum_i X_i`, the transverse
//! mixer, which factorises into one single-qubit rotation per site. A classical optimiser then
//! moves the `2p` angles to minimise `<C>`.
//!
//! # Why this module is here
//!
//! [`crate::sqa`] and the rest of the quantum-inspired lane simulate *thermodynamics*: a classical
//! `(d+1)`-dimensional system standing in for a quantum one. QAOA is the other lane, the gate-model
//! algorithm that Ising-machine papers benchmark against. It is the baseline this crate had no way
//! to compute, and a stack that prices annealing against nothing but annealing cannot say where the
//! interesting line is.
//!
//! **This is a simulator, not a quantum computer, and not a fast classical solver.** It holds all
//! `2^n` amplitudes, so it costs `2^n` memory and `p n 2^n` work; [`MAX_QUBITS`] is 20 and anything
//! larger is refused rather than attempted. Nothing here is charged to [`crate::ledger::Ledger`] —
//! that ledger prices Gibbs cycles, reads and writes on a thermodynamic sampling device, and a
//! gate-model circuit performs none of those. A joules figure for QAOA would need a gate-model
//! device model, and this crate does not have one; inventing one would produce a number
//! indistinguishable from a measured one, which is the thing [`crate::ledger::Prices::UNSTATED`]
//! exists to refuse.
//!
//! # Two sign conventions, both of which decide the answer
//!
//! `C` is this crate's energy, `E(s) = -sum_edges J s_i s_j - sum_i h_i s_i`, exactly
//! [`Graph::energy`], with bit `i` of a basis index set meaning `s_i = +1`. So this module
//! **minimises** `<C>`, matching every other optimiser here, where the max-cut literature maximises
//! its own cost operator. The translation for an antiferromagnetic unweighted graph is
//! `cut = (m - <C>) / 2` with `m` the edge count.
//!
//! The generator convention is `e^{-i theta H}` for both layers, with no factor of two hidden in
//! either. A `beta` here rotates each qubit by `2 beta` on the Bloch sphere, which is what
//! `e^{-i beta X}` does, and is why the closed form below carries `sin(4 beta)`.
//!
//! # The closed form this is checked against
//!
//! For a triangle-free unweighted graph at `p = 1` the per-edge expectation has a published closed
//! form (Wang, Hadfield, Jiang & Rieffel, *Quantum approximate optimization algorithm for
//! `MaxCut`: a fermionic view*, Phys. Rev. A 97, 022304 (2018), arXiv:1706.02998). Specialised to
//! the ring of disagrees — the unweighted cycle with `J = -1`, every vertex of degree two, no
//! triangles — it collapses to
//!
//! ```text
//!   <C>(gamma, beta) = (n / 2) sin(4 beta) sin(4 gamma)          n >= 4, p = 1
//! ```
//!
//! whose minimum is `-n/2`, a cut of `3n/4`: the level-one value Farhi, Goldstone & Gutmann report
//! for the ring of disagrees, and the first entry of their `(2p+1)/(2p+2)` sequence. Level two
//! reaches `5n/6`, which is `<C> = -2n/3`. Both are asserted here against the optimiser.
//!
//! Measured over 1734 angle pairs and six ring sizes, the simulation sits **5.3e-15** from that
//! formula at worst — which is as close as a `2 p n 2^n`-operation state vector gets, and is not
//! zero. Exact equality is not on offer and is not asserted.
//!
//! The formula holds for `n >= 4` and **fails on the triangle**, where the two endpoints of every
//! edge share a neighbour and a term the triangle-free derivation drops comes back — measured at
//! 3.0 absolute on a model whose entire spectrum is `[-1, 3]`. That case is asserted too: a test
//! that only showed the formula holding would also pass for an implementation that had quietly
//! evaluated the formula instead of the circuit.
//!
//! ```
//! use ferrotherm::{ising, qaoa};
//!
//! // the ring of disagrees on 8 vertices
//! let g = ising::ring(8, -1.0, 0.0);
//! let a = qaoa::Ansatz::new(&g).unwrap();
//! let best = qaoa::Optimiser::default().optimise(&a, 1).unwrap();
//!
//! // level one reaches three quarters of the maximum cut, and no more
//! assert!((best.expectation + 4.0).abs() < 1e-6, "{}", best.expectation);
//! let cut = (g.n_edges as f64 - best.expectation) / 2.0;
//! assert!((cut - 6.0).abs() < 1e-6, "cut {cut} of a maximum 8");
//! ```

use crate::graph::Graph;
use crate::rng::Pcg;

/// The largest model this will simulate, in qubits.
///
/// A state vector is `2^n` complex amplitudes: 20 qubits is 16 MiB of `f64` and one layer costs
/// `n 2^n` flops. 21 would be 32 MiB and twice the work per layer; 30 would be 16 GiB. The cap is a
/// refusal rather than a warning because the failure past it is an allocation, not a slow run.
pub const MAX_QUBITS: usize = 20;

/// Why a QAOA run was refused.
#[derive(Clone, Debug, PartialEq)]
pub enum Error {
    /// The model has more spins than a state vector can be held for.
    TooManyQubits {
        /// Spins in the graph handed over.
        n: usize,
        /// The cap, [`MAX_QUBITS`].
        max: usize,
    },
    /// A graph with no spins has no state vector to prepare `|+>` on.
    NoQubits,
    /// `gamma` and `beta` must come in pairs, one of each per layer.
    LayerMismatch {
        /// Cost angles supplied.
        gammas: usize,
        /// Mixer angles supplied.
        betas: usize,
    },
    /// A flat angle vector must hold `2p` entries: `p` cost angles, then `p` mixer angles.
    FlatLengthOdd {
        /// The length supplied.
        len: usize,
    },
    /// An angle that is not a finite number. `exp(-i NaN E)` is NaN, and a NaN amplitude poisons
    /// the whole state vector without failing anywhere visible.
    AngleNotFinite {
        /// `"gamma"` or `"beta"`.
        which: &'static str,
        /// Which layer it was in, counting from zero.
        layer: usize,
        /// The value seen.
        value: f64,
    },
    /// The model itself has a non-finite energy somewhere, so the cost layer has no phase to apply.
    EnergyNotFinite {
        /// The basis state, as a bitmask with bit `i` set meaning `s_i = +1`.
        state: usize,
        /// The energy computed for it.
        value: f64,
    },
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Error::TooManyQubits { n, max } => write!(
                f,
                "exact state-vector simulation of {n} qubits needs 2^{n} complex amplitudes, and \
                 the cap is {max} qubits. QAOA is being simulated here, not run -- for a larger \
                 model use `tempering` or `sqa`, which never hold the whole Hilbert space"
            ),
            Error::NoQubits => write!(
                f,
                "a graph with no spins has no state vector and no cost operator; there is nothing \
                 to prepare |+> on"
            ),
            Error::LayerMismatch { gammas, betas } => write!(
                f,
                "a p-level ansatz has p cost angles and p mixer angles, and this has {gammas} \
                 gamma(s) against {betas} beta(s); the layer missing its angle would silently \
                 become an identity"
            ),
            Error::FlatLengthOdd { len } => write!(
                f,
                "a flat angle vector is p gammas followed by p betas, so its length must be even, \
                 and {len} is not"
            ),
            Error::AngleNotFinite { which, layer, value } => write!(
                f,
                "{which} in layer {layer} is {value}, which is not finite: that layer's phase would \
                 be NaN and every amplitude with it, leaving an expectation of NaN and a norm no \
                 unitarity check can read"
            ),
            Error::EnergyNotFinite { state, value } => write!(
                f,
                "the cost operator has energy {value} on basis state {state:#x}, which is not \
                 finite: some coupling or bias in this graph is NaN or infinite, and no diagonal \
                 phase can be formed from it"
            ),
        }
    }
}

impl core::error::Error for Error {}

/// The `2p` variational angles of a `p`-level ansatz.
///
/// Held as two lists rather than one flat vector because they are not interchangeable — `gamma`
/// multiplies an energy and `beta` a Bloch angle — and a flat vector is where a transposed pair
/// goes unnoticed. [`Angles::to_flat`] produces the flat form an optimiser wants, `p` gammas then
/// `p` betas, and [`Angles::from_flat`] reads it back.
#[derive(Clone, Debug, PartialEq)]
pub struct Angles {
    gamma: Vec<f64>,
    beta: Vec<f64>,
}

impl Angles {
    /// The angles of a `p`-level ansatz, cost angles first.
    ///
    /// `p = 0` is allowed and is a reference point rather than a degenerate case: it is `|+>^n`
    /// with no layers at all, whose expectation is the mean energy over every basis state.
    ///
    /// # Errors
    ///
    /// [`Error::LayerMismatch`] when the two lists differ in length, and
    /// [`Error::AngleNotFinite`] naming the first angle that is not a finite number.
    pub fn new(gamma: Vec<f64>, beta: Vec<f64>) -> Result<Self, Error> {
        if gamma.len() != beta.len() {
            return Err(Error::LayerMismatch { gammas: gamma.len(), betas: beta.len() });
        }
        let angles = Angles { gamma, beta };
        check_finite(&angles)?;
        Ok(angles)
    }

    /// Read the flat form an optimiser works in: `p` cost angles followed by `p` mixer angles.
    ///
    /// # Errors
    ///
    /// [`Error::FlatLengthOdd`] when the length is not `2p`, and [`Error::AngleNotFinite`] for a
    /// non-finite entry.
    pub fn from_flat(v: &[f64]) -> Result<Self, Error> {
        if !v.len().is_multiple_of(2) {
            return Err(Error::FlatLengthOdd { len: v.len() });
        }
        let p = v.len() / 2;
        Angles::new(v[..p].to_vec(), v[p..].to_vec())
    }

    /// The flat form: `p` gammas, then `p` betas.
    #[must_use]
    pub fn to_flat(&self) -> Vec<f64> {
        let mut v = Vec::with_capacity(2 * self.gamma.len());
        v.extend_from_slice(&self.gamma);
        v.extend_from_slice(&self.beta);
        v
    }

    /// Levels `p`.
    #[must_use]
    pub fn p(&self) -> usize {
        self.gamma.len()
    }

    /// The cost angles, one per layer.
    #[must_use]
    pub fn gamma(&self) -> &[f64] {
        &self.gamma
    }

    /// The mixer angles, one per layer.
    #[must_use]
    pub fn beta(&self) -> &[f64] {
        &self.beta
    }

    /// The same state, written as a `p + 1`-level ansatz whose outermost layer is the identity.
    ///
    /// This is the entire content of "a `p`-level ansatz contains the `(p-1)`-level one": appending
    /// `gamma = beta = 0` makes the last pair of exponentials `e^0 = I`, so the state, and with it
    /// `<C>`, is unchanged. It is what [`Optimiser::ladder`] starts each new level from, and it is
    /// why that sequence is monotone rather than merely usually monotone.
    #[must_use]
    pub fn with_identity_layer(&self) -> Self {
        let mut next = self.clone();
        next.gamma.push(0.0);
        next.beta.push(0.0);
        next
    }
}

/// Scratch buffers for one state vector, reused across the thousands of evaluations an optimiser
/// makes. Allocating `2^n` amplitudes per objective call is most of the cost from 16 qubits up.
struct Work {
    re: Vec<f64>,
    im: Vec<f64>,
    terms: Vec<f64>,
}

/// What one evolution of the ansatz produced.
#[derive(Clone, Debug)]
pub struct Run {
    expectation: f64,
    bracket: (f64, f64),
    probabilities: Vec<f64>,
    layer_norms: Vec<f64>,
}

impl Run {
    /// `<C>`, the expected energy of the prepared state.
    ///
    /// Accumulated with [`crate::round::sum_up`], so it is never BELOW the exact sum of the terms
    /// it was built from. That direction is the load-bearing one: `<C>` is an average of energies,
    /// so it is an upper bound on the ground energy, and this crate has already shipped a "bound"
    /// that sat on the wrong side of what it bounded for accumulating in round-to-nearest.
    ///
    /// The guard covers the SUMMATION only. The amplitudes themselves come out of `2 p n 2^n`
    /// floating-point operations and carry their own error, which is why the closed-form test in
    /// this module asserts a measured tolerance rather than equality. See [`Run::bracket`].
    #[must_use]
    pub fn expectation(&self) -> f64 {
        self.expectation
    }

    /// `(sum_down, sum_up)` over the same terms: the interval the exact sum of the simulated
    /// probability-weighted energies certainly lies in.
    ///
    /// Its width is the summation's contribution and nothing else, so a wide bracket means
    /// cancellation in the sum and a narrow one means the remaining error is the simulation's.
    #[must_use]
    pub fn bracket(&self) -> (f64, f64) {
        self.bracket
    }

    /// `|amplitude|^2` for every basis state, indexed so bit `i` set means `s_i = +1`.
    #[must_use]
    pub fn probabilities(&self) -> &[f64] {
        &self.probabilities
    }

    /// The state-vector norm after each operator application: `|+>` first, then alternately after
    /// every cost layer and every mixer layer, so a `p`-level run reports `2p + 1` numbers.
    ///
    /// `C` and `B` are Hermitian, so `e^{-i gamma C}` and `e^{-i beta B}` are unitary and every one
    /// of these is 1. It is a cheap check that each layer is the exponential of something
    /// Hermitian — but note what it does NOT check. `e^{-i gamma C}`, `e^{+i gamma C}` and
    /// `e^{-i 2 gamma C}` are all equally unitary, so a flipped sign or a stray factor of two
    /// leaves every norm at exactly 1. Measured on the ring of disagrees on twelve vertices:
    /// conjugating the cost phase moves `<C>` by up to **12.0**, the full width of the reachable
    /// range, and moves every norm by **0**.
    ///
    /// Measured again by actually doing it: doubling the cost exponent to `e^{-i 2 gamma C}` in
    /// [`Ansatz::run`]'s cost layer leaves all seven layer norms within 8.9e-16 of 1, and leaves
    /// **eight of the nine tests in this module green** — including this one. The ninth, the
    /// closed-form check, fails by 1.41 on the first grid point it reaches. Unitarity catches a bad
    /// *shape*; only the closed form catches a bad *angle*.
    ///
    /// Each entry is the square root of a [`crate::round::sum_up`] total, so it is never below the
    /// true norm of the simulated amplitudes.
    #[must_use]
    pub fn layer_norms(&self) -> &[f64] {
        &self.layer_norms
    }

    /// The basis state carrying the largest probability, as spins, with that probability.
    ///
    /// Ties go to the lowest index, so this is deterministic. It is what a single noiseless
    /// measurement most likely returns, and unlike `<C>` its energy is an upper bound on the ground
    /// energy that is ATTAINED by a state a verifier can be handed.
    ///
    /// # Panics
    ///
    /// Never: a [`Run`] always holds at least two probabilities, so the maximum exists and the
    /// running best is always replaced on the first iteration.
    #[must_use]
    pub fn most_likely(&self) -> (Vec<i8>, f64) {
        let (mut best, mut best_p) = (0usize, f64::NEG_INFINITY);
        for (z, &p) in self.probabilities.iter().enumerate() {
            if p > best_p {
                best_p = p;
                best = z;
            }
        }
        let n = self.probabilities.len().trailing_zeros() as usize;
        (spins_of(best, n), best_p)
    }

    /// `shots` measurements in the computational basis, each a full spin configuration.
    ///
    /// Deterministic for a seed, like everything else here. This returns raw states rather than a
    /// [`crate::samples::SampleSet`] deliberately: that type carries a
    /// [`crate::samples::Provenance`], none of whose variants describes a projective measurement of
    /// a pure state, and labelling these `Chain` or `Enumerated` would hand `tau_int` and every
    /// error bar downstream an order and a temperature that are not there. The honest form is a new
    /// provenance variant, which is a change to that module and not to this one.
    #[must_use]
    pub fn measure(&self, shots: usize, seed: u64) -> Vec<Vec<i8>> {
        let n = self.probabilities.len().trailing_zeros() as usize;
        let mut cdf = Vec::with_capacity(self.probabilities.len());
        let mut acc = 0.0;
        for &p in &self.probabilities {
            acc += p;
            cdf.push(acc);
        }
        let total = cdf.last().copied().unwrap_or(1.0);
        let mut rng = Pcg::new(seed, 0x0051_A0A0);
        (0..shots)
            .map(|_| {
                let u = rng.f64() * total;
                // The first index whose cumulative mass exceeds u, clamped because a u drawn at the
                // very top of the range can exceed the last entry by an ulp.
                let z = cdf.partition_point(|&c| c <= u).min(cdf.len() - 1);
                spins_of(z, n)
            })
            .collect()
    }
}

/// Spins of a basis index: bit `i` set means `s_i = +1`.
fn spins_of(z: usize, n: usize) -> Vec<i8> {
    (0..n).map(|i| if (z >> i) & 1 == 1 { 1i8 } else { -1 }).collect()
}

/// The `p`-level QAOA ansatz over one Ising model, simulated exactly.
///
/// Construction precomputes the cost operator's diagonal — `E(s)` for all `2^n` basis states,
/// straight from [`Graph::energy`], so the numbers are the ones every other module in this crate
/// would compute — and everything after that is angles.
pub struct Ansatz<'a> {
    g: &'a Graph,
    diag: Vec<f64>,
}

/// Written by hand rather than derived: the derived form would print `2^n` energies, which at the
/// cap is a million numbers in a panic message.
impl core::fmt::Debug for Ansatz<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let lo = self.diag.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = self.diag.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        f.debug_struct("Ansatz")
            .field("n", &self.g.n)
            .field("states", &self.diag.len())
            .field("spectrum", &(lo, hi))
            .finish()
    }
}

impl<'a> Ansatz<'a> {
    /// Prepare the ansatz for `g`, precomputing the cost diagonal.
    ///
    /// # Errors
    ///
    /// [`Error::NoQubits`] for an empty graph, [`Error::TooManyQubits`] past [`MAX_QUBITS`], and
    /// [`Error::EnergyNotFinite`] naming the first basis state whose energy is not a number.
    pub fn new(g: &'a Graph) -> Result<Self, Error> {
        if g.n == 0 {
            return Err(Error::NoQubits);
        }
        if g.n > MAX_QUBITS {
            return Err(Error::TooManyQubits { n: g.n, max: MAX_QUBITS });
        }
        let dim = 1usize << g.n;
        let mut diag = Vec::with_capacity(dim);
        let mut s = vec![-1i8; g.n];
        for z in 0..dim {
            for i in 0..g.n {
                s[i] = if (z >> i) & 1 == 1 { 1 } else { -1 };
            }
            let e = g.energy(&s);
            if !e.is_finite() {
                return Err(Error::EnergyNotFinite { state: z, value: e });
            }
            diag.push(e);
        }
        Ok(Ansatz { g, diag })
    }

    /// The graph this ansatz encodes.
    #[must_use]
    pub fn graph(&self) -> &Graph {
        self.g
    }

    /// Qubits, which is the graph's spin count.
    #[must_use]
    pub fn n(&self) -> usize {
        self.g.n
    }

    /// The cost operator's diagonal: `E(s)` for every basis state, bit `i` set meaning `s_i = +1`.
    #[must_use]
    pub fn diagonal(&self) -> &[f64] {
        &self.diag
    }

    /// Evolve `|+>^n` through the ansatz and report everything the run knows.
    ///
    /// # Errors
    ///
    /// [`Error::AngleNotFinite`] — the angles are re-checked here because an [`Angles`] can be
    /// built and its flat form then edited by a caller's own search.
    pub fn run(&self, angles: &Angles) -> Result<Run, Error> {
        check_finite(angles)?;
        let mut w = self.work();
        let mut norms = Vec::with_capacity(2 * angles.p() + 1);
        self.evolve(&mut w, angles, Some(&mut norms));
        for (z, t) in w.terms.iter_mut().enumerate() {
            *t = w.re[z] * w.re[z] + w.im[z] * w.im[z];
        }
        let probabilities = w.terms.clone();
        for (z, t) in w.terms.iter_mut().enumerate() {
            *t *= self.diag[z];
        }
        let bracket = (crate::round::sum_down(&w.terms), crate::round::sum_up(&w.terms));
        Ok(Run { expectation: bracket.1, bracket, probabilities, layer_norms: norms })
    }

    /// `<C>` alone, without building the probability vector — the objective an optimiser calls.
    ///
    /// # Errors
    ///
    /// [`Error::AngleNotFinite`] for a non-finite angle.
    pub fn expectation(&self, angles: &Angles) -> Result<f64, Error> {
        check_finite(angles)?;
        let mut w = self.work();
        Ok(self.expectation_with(&mut w, angles))
    }

    fn work(&self) -> Work {
        let dim = 1usize << self.g.n;
        Work { re: vec![0.0; dim], im: vec![0.0; dim], terms: vec![0.0; dim] }
    }

    /// The objective on a reused workspace. Angles must already have been checked finite.
    fn expectation_with(&self, w: &mut Work, angles: &Angles) -> f64 {
        self.evolve(w, angles, None);
        for (z, t) in w.terms.iter_mut().enumerate() {
            *t = (w.re[z] * w.re[z] + w.im[z] * w.im[z]) * self.diag[z];
        }
        crate::round::sum_up(&w.terms)
    }

    /// Prepare `|+>^n` and apply every layer, optionally recording the norm after each.
    fn evolve(&self, w: &mut Work, angles: &Angles, mut norms: Option<&mut Vec<f64>>) {
        let dim = 1usize << self.g.n;
        // 2^{-n/2}: exact for even n, correctly rounded for odd n.
        let amp = (1.0 / dim as f64).sqrt();
        w.re.iter_mut().for_each(|v| *v = amp);
        w.im.iter_mut().for_each(|v| *v = 0.0);
        if let Some(ns) = norms.as_deref_mut() {
            let v = norm(w);
            ns.push(v);
        }
        for layer in 0..angles.p() {
            apply_cost(&mut w.re, &mut w.im, &self.diag, angles.gamma[layer]);
            if let Some(ns) = norms.as_deref_mut() {
                let v = norm(w);
                ns.push(v);
            }
            apply_mixer(&mut w.re, &mut w.im, self.g.n, angles.beta[layer]);
            if let Some(ns) = norms.as_deref_mut() {
                let v = norm(w);
                ns.push(v);
            }
        }
    }
}

/// Re-check that every angle is finite: the one thing standing between a caller's NaN and a state
/// vector of NaNs that reports an expectation of NaN and a norm nothing can read.
fn check_finite(angles: &Angles) -> Result<(), Error> {
    for (layer, &v) in angles.gamma.iter().enumerate() {
        if !v.is_finite() {
            return Err(Error::AngleNotFinite { which: "gamma", layer, value: v });
        }
    }
    for (layer, &v) in angles.beta.iter().enumerate() {
        if !v.is_finite() {
            return Err(Error::AngleNotFinite { which: "beta", layer, value: v });
        }
    }
    Ok(())
}

/// The state-vector norm, upper-directed: never below the true norm of these amplitudes.
fn norm(w: &mut Work) -> f64 {
    for (z, t) in w.terms.iter_mut().enumerate() {
        *t = w.re[z] * w.re[z] + w.im[z] * w.im[z];
    }
    crate::round::sum_up(&w.terms).max(0.0).sqrt()
}

/// `e^{-i gamma C}`: `C` is diagonal, so this is one phase per basis state and nothing else.
fn apply_cost(re: &mut [f64], im: &mut [f64], diag: &[f64], gamma: f64) {
    for z in 0..re.len() {
        let (s, c) = (gamma * diag[z]).sin_cos();
        // (re + i im)(cos gamma E - i sin gamma E)
        let (r, i) = (re[z], im[z]);
        re[z] = r * c + i * s;
        im[z] = i * c - r * s;
    }
}

/// `e^{-i beta B}` with `B = sum_i X_i`: a product of independent single-qubit rotations,
/// `cos(beta) I - i sin(beta) X` on each, applied in place one qubit at a time.
fn apply_mixer(re: &mut [f64], im: &mut [f64], n: usize, beta: f64) {
    let (s, c) = beta.sin_cos();
    for i in 0..n {
        let stride = 1usize << i;
        let mut base = 0usize;
        while base < re.len() {
            for a in base..base + stride {
                let b = a + stride;
                let (r0, i0, r1, i1) = (re[a], im[a], re[b], im[b]);
                // a0' = c a0 - i s a1 ,  a1' = c a1 - i s a0
                re[a] = c * r0 + s * i1;
                im[a] = c * i0 - s * r1;
                re[b] = c * r1 + s * i0;
                im[b] = c * i1 - s * r0;
            }
            base += stride << 1;
        }
    }
}

/// The best angles a search found, and what they were worth.
#[derive(Clone, Debug)]
pub struct Optimum {
    /// The angles themselves.
    pub angles: Angles,
    /// `<C>` at those angles — the quantity minimised, so lower is better.
    pub expectation: f64,
    /// Objective evaluations spent, summed over every restart. The honest cost of the answer.
    pub evaluations: usize,
}

/// A derivative-free classical optimiser over the `2p` angles.
///
/// Nelder–Mead with random restarts: no gradients, no dependencies, deterministic for a seed. The
/// QAOA landscape is periodic and dense with local optima, so the restart count is a PARAMETER and
/// not a hidden one — a single start from a fixed point flatters the method exactly the way a
/// greedy baseline with one restart flatters everything it is compared against.
#[derive(Clone, Debug, PartialEq)]
pub struct Optimiser {
    /// Random starts to try. One is not a search.
    pub restarts: usize,
    /// Objective evaluations allowed per restart.
    pub max_evals: usize,
    /// Stop a restart once the simplex has collapsed to this spread in the angles.
    pub tol: f64,
    /// Seed for the starts, so a run is reproducible.
    pub seed: u64,
}

impl Default for Optimiser {
    fn default() -> Self {
        Optimiser { restarts: 24, max_evals: 1200, tol: 1e-12, seed: 0x0051_A0A0 }
    }
}

impl Optimiser {
    /// Minimise `<C>` over a `p`-level ansatz from random starts.
    ///
    /// `p = 0` has no angles to move and returns the single evaluation of `|+>^n`.
    ///
    /// # Errors
    ///
    /// [`Error::AngleNotFinite`] can only come from a start this constructs, and it never
    /// constructs a non-finite one; the result is propagated rather than unwrapped so that a future
    /// change to the start distribution cannot turn into a panic.
    ///
    /// # Panics
    ///
    /// Never: the restart loop runs `restarts.max(1) >= 1` times and each pass sets the best
    /// candidate, so the `Option` is always filled by the time it is read.
    pub fn optimise(&self, a: &Ansatz<'_>, p: usize) -> Result<Optimum, Error> {
        let mut rng = Pcg::new(self.seed, 0x0051_0BEC);
        let mut w = a.work();
        let mut best: Option<Optimum> = None;
        let mut evals = 0usize;
        for _ in 0..self.restarts.max(1) {
            // gamma multiplies an energy and beta a Bloch angle; both are drawn over a full period
            // of their own layer. Halving either window on a symmetry argument would make the
            // answer depend on that argument being right, and `<C>` is only symmetric under
            // (gamma, beta) -> (-gamma, -beta) jointly.
            let start = Angles::new(
                (0..p).map(|_| (rng.f64() - 0.5) * 2.0 * core::f64::consts::PI).collect(),
                (0..p).map(|_| (rng.f64() - 0.5) * core::f64::consts::PI).collect(),
            )?;
            let (angles, value, used) = self.nelder_mead(a, &mut w, &start);
            evals += used;
            if best.as_ref().is_none_or(|b| value < b.expectation) {
                best = Some(Optimum { angles, expectation: value, evaluations: 0 });
            }
            if p == 0 {
                break;
            }
        }
        let mut out = best.expect("restarts.max(1) is at least one, so a candidate was set");
        out.evaluations = evals;
        Ok(out)
    }

    /// Minimise from angles a caller already has, with no restarts.
    ///
    /// The returned optimum is never worse than `start`: the starting point is a vertex of the
    /// initial simplex, and what comes back is the best vertex ever evaluated.
    ///
    /// # Errors
    ///
    /// [`Error::AngleNotFinite`] when `start` holds one.
    pub fn optimise_from(&self, a: &Ansatz<'_>, start: &Angles) -> Result<Optimum, Error> {
        check_finite(start)?;
        let mut w = a.work();
        let (angles, expectation, evaluations) = self.nelder_mead(a, &mut w, start);
        Ok(Optimum { angles, expectation, evaluations })
    }

    /// Optimise levels `1..=p_max`, each warm-started from the level below.
    ///
    /// Level `k` starts from level `k-1`'s answer with an identity layer appended
    /// ([`Angles::with_identity_layer`]), which evaluates to exactly level `k-1`'s expectation, and
    /// [`Optimiser::optimise_from`] never returns worse than its start. **So the sequence is
    /// monotone by construction, and that is the point.** The containment argument — a `p`-level
    /// ansatz can represent every `(p-1)`-level state — says the OPTIMUM improves with `p`. It says
    /// nothing whatever about what a local search finds, and "the optimised expectation improves
    /// monotonically with `p`" is simply false for a cold-started search.
    ///
    /// Measured on random Ising glasses with fields, cold-started [`Optimiser::optimise`] at each
    /// level against [`Optimiser::ladder`] on the same instances and seeds:
    ///
    /// ```text
    ///   instances          restarts   cold non-monotone   worst regression   ladder
    ///   10 spins, p 1..4       2          15 / 24               3.08         0 / 24
    ///   10 spins, p 1..4       6          12 / 24               4.05         0 / 24
    ///    9 spins, p 1..3      24           1 / 12               0.034        0 / 12
    /// ```
    ///
    /// A regression of 4.05 in energy is not a rounding artefact; it is the level-4 search landing
    /// in a worse basin than the level-3 search and having no memory that the better one exists.
    /// The ladder gives it that memory, and the last row says more restarts shrink the problem
    /// without removing it.
    ///
    /// Cold restarts are also run at each level and kept when they beat the warm start, so this is
    /// never worse than [`Optimiser::optimise`] either.
    ///
    /// # Errors
    ///
    /// [`Error::AngleNotFinite`], propagated from the starts.
    pub fn ladder(&self, a: &Ansatz<'_>, p_max: usize) -> Result<Vec<Optimum>, Error> {
        let mut out: Vec<Optimum> = Vec::with_capacity(p_max);
        for p in 1..=p_max {
            let cold = self.optimise(a, p)?;
            let best = match out.last() {
                None => cold,
                Some(prev) => {
                    let warm = self.optimise_from(a, &prev.angles.with_identity_layer())?;
                    let spent = warm.evaluations + cold.evaluations;
                    if warm.expectation <= cold.expectation {
                        Optimum { evaluations: spent, ..warm }
                    } else {
                        Optimum { evaluations: spent, ..cold }
                    }
                }
            };
            out.push(best);
        }
        Ok(out)
    }

    /// Nelder–Mead on the flat angle vector. Returns the best point ever evaluated, its value, and
    /// the evaluation count — the best point, not the final simplex's centroid, which is what makes
    /// the result impossible to be worse than the start.
    fn nelder_mead(&self, a: &Ansatz<'_>, w: &mut Work, start: &Angles) -> (Angles, f64, usize) {
        let x0 = start.to_flat();
        let d = x0.len();
        let mut evals = 0usize;
        let eval = |x: &[f64], w: &mut Work, evals: &mut usize| -> f64 {
            *evals += 1;
            match Angles::from_flat(x) {
                Ok(ang) => a.expectation_with(w, &ang),
                // A simplex step cannot change the length, so this is the non-finite case: rejected
                // as infinitely bad rather than evaluated to a NaN, which would win every
                // comparison it took part in.
                Err(_) => f64::INFINITY,
            }
        };
        let f0 = eval(&x0, w, &mut evals);
        if d == 0 {
            return (start.clone(), f0, evals);
        }
        // Initial simplex: the start plus one step along each axis. A tenth of a radian is small
        // against the pi-scale periodicity of the layers and large against `tol`.
        const STEP: f64 = 0.1;
        let mut pts: Vec<(Vec<f64>, f64)> = Vec::with_capacity(d + 1);
        pts.push((x0.clone(), f0));
        for k in 0..d {
            let mut x = x0.clone();
            x[k] += STEP;
            let f = eval(&x, w, &mut evals);
            pts.push((x, f));
        }
        let (mut best_x, mut best_f) = (x0, f0);
        for (x, f) in &pts {
            if *f < best_f {
                best_f = *f;
                best_x = x.clone();
            }
        }
        while evals < self.max_evals {
            pts.sort_by(|p, q| p.1.total_cmp(&q.1));
            // A collapsed simplex: every vertex within `tol` of the best in every coordinate.
            let spread = pts
                .iter()
                .map(|(x, _)| {
                    x.iter().zip(&pts[0].0).map(|(u, v)| (u - v).abs()).fold(0.0f64, f64::max)
                })
                .fold(0.0f64, f64::max);
            if spread < self.tol {
                break;
            }
            let mut centroid = vec![0.0; d];
            for (x, _) in &pts[..d] {
                for k in 0..d {
                    centroid[k] += x[k] / d as f64;
                }
            }
            let worst_x = pts[d].0.clone();
            let worst = pts[d].1;
            let step = |coef: f64, centroid: &[f64], worst_x: &[f64]| -> Vec<f64> {
                (0..d).map(|k| centroid[k] + coef * (centroid[k] - worst_x[k])).collect()
            };
            let xr = step(1.0, &centroid, &worst_x);
            let fr = eval(&xr, w, &mut evals);
            if fr < pts[0].1 {
                let xe = step(2.0, &centroid, &worst_x);
                let fe = eval(&xe, w, &mut evals);
                pts[d] = if fe < fr { (xe, fe) } else { (xr, fr) };
            } else if fr < pts[d - 1].1 {
                pts[d] = (xr, fr);
            } else {
                let (xc, fc) = if fr < worst {
                    let x = step(0.5, &centroid, &worst_x);
                    let f = eval(&x, w, &mut evals);
                    (x, f)
                } else {
                    let x = step(-0.5, &centroid, &worst_x);
                    let f = eval(&x, w, &mut evals);
                    (x, f)
                };
                if fc < worst {
                    pts[d] = (xc, fc);
                } else {
                    // Shrink toward the best vertex.
                    let anchor = pts[0].0.clone();
                    for (x, f) in pts.iter_mut().skip(1) {
                        for k in 0..d {
                            x[k] = anchor[k] + 0.5 * (x[k] - anchor[k]);
                        }
                        *f = eval(&x[..], w, &mut evals);
                    }
                }
            }
            for (x, f) in &pts {
                if *f < best_f {
                    best_f = *f;
                    best_x = x.clone();
                }
            }
        }
        let angles = Angles::from_flat(&best_x).unwrap_or_else(|_| start.clone());
        (angles, best_f, evals)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::GraphBuilder;
    use crate::ising::ring;
    use crate::oracle::{Exhaustive, Solver};

    /// A sparse random Ising model with fields, small enough to enumerate exhaustively.
    fn glass(n: usize, seed: u64) -> Graph {
        let mut rng = Pcg::new(seed, 0x0051_9111);
        let mut b = GraphBuilder::new(n);
        for i in 0..n {
            b.bias(i, rng.f64() * 2.0 - 1.0);
            for j in (i + 1)..n {
                if rng.f64() < 0.4 {
                    b.couple(i, j, rng.f64() * 2.0 - 1.0);
                }
            }
        }
        b.build()
    }

    /// One point of the angle grid the closed form is checked on.
    fn grid(k: usize) -> (f64, f64) {
        let gamma = -core::f64::consts::PI + 2.0 * core::f64::consts::PI * k as f64 / 16.0;
        let beta = -core::f64::consts::FRAC_PI_2 + core::f64::consts::PI * k as f64 / 16.0;
        (gamma, beta)
    }

    /// THE HEADLINE, and its oracle is a published formula rather than anything in this file.
    ///
    /// For a triangle-free unweighted graph the `p = 1` expectation of a cut edge term is closed
    /// (Wang, Hadfield, Jiang & Rieffel, Phys. Rev. A 97, 022304 (2018)):
    ///
    /// ```text
    ///   <C_uv> = 1/2 + (1/4) sin(4 beta) sin(gamma) (cos^{d_u - 1} gamma + cos^{d_v - 1} gamma)
    /// ```
    ///
    /// On the ring of disagrees every degree is two, so the bracket is `2 cos gamma`; translating
    /// from the cut operator to this crate's energy (`C = sum_edges Z Z` when `J = -1`, and
    /// `cut = (m - <C>) / 2`) gives `<C> = (n/2) sin(4 beta) sin(4 gamma)` over the whole ring. Its
    /// minimum is `-n/2`, a cut of `3n/4`, which is the level-one value Farhi, Goldstone & Gutmann
    /// publish for this instance (arXiv:1411.4028).
    ///
    /// It is checked over a grid of angles rather than at the optimum, so it constrains the whole
    /// landscape and not one point — 1734 pairs across six ring sizes, worst deviation 5.33e-15.
    ///
    /// **And it is asserted to FAIL on the triangle**, the smallest case the derivation excludes,
    /// where the endpoints of every edge share a neighbour. Without that half, an implementation
    /// that had quietly evaluated the formula instead of the circuit would pass.
    #[test]
    fn p1_ring_matches_the_wang_rieffel_closed_form_and_the_triangle_refutes_it() {
        let closed =
            |n: usize, gam: f64, bet: f64| 0.5 * n as f64 * (4.0 * bet).sin() * (4.0 * gam).sin();

        let mut worst = 0.0f64;
        for n in [4usize, 5, 6, 8, 10, 12] {
            let g = ring(n, -1.0, 0.0);
            let a = Ansatz::new(&g).unwrap();
            for gi in 0..=16 {
                for bi in 0..=16 {
                    let (gam, _) = grid(gi);
                    let (_, bet) = grid(bi);
                    let got = a.expectation(&Angles::new(vec![gam], vec![bet]).unwrap()).unwrap();
                    let want = closed(n, gam, bet);
                    let dev = (got - want).abs();
                    assert!(
                        dev < 1e-13,
                        "n={n} gamma={gam} beta={bet}: simulated {got}, closed form {want}, off \
                         by {dev:e}"
                    );
                    worst = worst.max(dev);
                }
            }
        }
        // Measured 5.33e-15, at n = 12 where the sum is longest. A deviation of exactly zero would
        // mean the formula was being evaluated rather than a state vector.
        assert!(worst > 0.0, "a deviation of exactly zero is a simulator that is not simulating");
        assert!(worst < 1e-13, "worst deviation over the grid was {worst:e}");

        let g3 = ring(3, -1.0, 0.0);
        let a3 = Ansatz::new(&g3).unwrap();
        let mut worst3 = 0.0f64;
        for gi in 0..=16 {
            for bi in 0..=16 {
                let (gam, _) = grid(gi);
                let (_, bet) = grid(bi);
                let got = a3.expectation(&Angles::new(vec![gam], vec![bet]).unwrap()).unwrap();
                worst3 = worst3.max((got - closed(3, gam, bet)).abs());
            }
        }
        // Measured 3.0, on a model whose entire spectrum is [-1, 3].
        assert!(
            worst3 > 1.0,
            "the triangle-free closed form must not survive a triangle, and it was off by only \
             {worst3:e}"
        );
    }

    /// ORACLE: the `(2p+1)/(2p+2)` sequence Farhi, Goldstone & Gutmann publish for the ring of
    /// disagrees (arXiv:1411.4028) — four independent numbers, one per level, plus exhaustive
    /// enumeration for the ground energy they are measured against.
    ///
    /// In this crate's energy that sequence is `<C>_p = -n p / (p + 1)`: `-5, -20/3, -15/2, -8` on
    /// ten vertices, against a ground energy of `-10`. So it says both what each level reaches and
    /// how far short it stops, and the second half is asserted too — a solver that simply found the
    /// ground state would sail through a monotonicity test and fail this one.
    #[test]
    fn the_ladder_reaches_the_published_ring_of_disagrees_sequence_and_never_goes_backwards() {
        let n = 10usize;
        let g = ring(n, -1.0, 0.0);
        let a = Ansatz::new(&g).unwrap();
        let lad = Optimiser::default().ladder(&a, 4).unwrap();
        assert_eq!(lad.len(), 4);

        for (k, o) in lad.iter().enumerate() {
            let p = k + 1;
            let want = -(n as f64) * p as f64 / (p as f64 + 1.0);
            assert!(
                (o.expectation - want).abs() < 1e-6,
                "level {p} reached {} against the published {want}",
                o.expectation
            );
            let cut = (g.n_edges as f64 - o.expectation) / 2.0;
            let want_cut = n as f64 * (2 * p + 1) as f64 / (2 * p + 2) as f64;
            assert!((cut - want_cut).abs() < 1e-6, "level {p} cut {cut} against {want_cut}");
            assert_eq!(o.angles.p(), p);
            assert!(o.evaluations > 0);
        }
        for k in 1..lad.len() {
            assert!(
                lad[k].expectation <= lad[k - 1].expectation + 1e-12,
                "level {} went backwards: {:?}",
                k + 1,
                lad.iter().map(|o| o.expectation).collect::<Vec<_>>()
            );
        }

        let ground = Exhaustive.solve(&g).1;
        assert!((ground + n as f64).abs() < 1e-12, "the ring of disagrees on {n} has ground -{n}");
        // Every level must stop SHORT: level one gives up half the answer and even level four is
        // two units away. This is the half that a ground-state solver fails.
        assert!(lad[0].expectation > ground + 4.9, "level one must not reach the ground state");
        assert!(lad[3].expectation > ground + 1.9, "nor may level four");
    }

    /// ORACLE: [`crate::oracle::Exhaustive`], which enumerates all `2^n` states.
    ///
    /// `<C>` is an average of the spectrum weighted by `|amplitude|^2`, so it can never fall below
    /// the smallest energy nor rise above the largest. Violating that is the signature of a
    /// normalisation error, a mis-indexed diagonal, or a probability that came out negative.
    ///
    /// The bound is asserted on the LOWER side of the directed sum as well as the value returned,
    /// and — the asymmetric half — the slack at random angles is asserted to be LARGE. A bound that
    /// held because every answer sat on the ground energy would be no evidence at all; measured,
    /// the tightest random-angle run sits 2.94 above it.
    #[test]
    fn the_expectation_stays_inside_the_exhaustively_enumerated_spectrum() {
        let mut rng = Pcg::new(11, 0x0051_7777);
        let mut min_slack = f64::INFINITY;
        let mut widest = 0.0f64;
        for inst in 0..5u64 {
            let g = glass(9, inst);
            let a = Ansatz::new(&g).unwrap();
            let e_min = Exhaustive.solve(&g).1;
            let e_max = a.diagonal().iter().copied().fold(f64::NEG_INFINITY, f64::max);
            assert!(e_max > e_min, "instance {inst} is flat and would test nothing");

            for _ in 0..40 {
                let p = 1 + (rng.next_u32() % 3) as usize;
                let ang = Angles::new(
                    (0..p).map(|_| (rng.f64() - 0.5) * 8.0).collect(),
                    (0..p).map(|_| (rng.f64() - 0.5) * 8.0).collect(),
                )
                .unwrap();
                let r = a.run(&ang).unwrap();
                let (lo, hi) = r.bracket();
                assert!(lo <= hi, "inverted bracket [{lo}, {hi}]");
                assert_eq!(r.expectation(), hi, "the reported value is the upper-directed sum");
                assert!(
                    lo >= e_min,
                    "instance {inst}: <C> = {lo} is below the enumerated ground energy {e_min}"
                );
                assert!(hi <= e_max, "instance {inst}: <C> = {hi} is above the top of the spectrum");
                min_slack = min_slack.min(lo - e_min);
                widest = widest.max(hi - lo);
            }

            // An optimised run is where the bound is tightest and a one-sided error would first
            // show. Measured across these five instances: -3.31 to -5.46, against ground energies
            // of -6.23 to -7.41.
            let best = Optimiser { restarts: 8, ..Optimiser::default() }.optimise(&a, 2).unwrap();
            assert!(
                best.expectation >= e_min - 1e-12,
                "instance {inst}: optimised <C> = {} is below the ground energy {e_min}",
                best.expectation
            );
            // And the ansatz has to DO something: `|+>` alone gives a mean energy of exactly zero.
            assert!(
                best.expectation < -1.0,
                "instance {inst}: the optimiser reached {} against a zero-level value of 0",
                best.expectation
            );
            // A state a verifier can be handed, which is in the spectrum for the same reason.
            let (s, pr) = a.run(&best.angles).unwrap().most_likely();
            assert_eq!(s.len(), g.n);
            assert!(pr > 0.0 && pr <= 1.0, "a probability of {pr}");
            assert!(g.energy(&s) >= e_min - 1e-12);
        }
        // Measured 2.94.
        assert!(min_slack > 1.0, "random angles sat only {min_slack:e} above the ground energy");
        // The summation's own contribution, for contrast: measured 3.55e-15.
        assert!(widest < 1e-13, "the directed-sum bracket widened to {widest:e}");
    }

    /// `B` and `C` are Hermitian, so every layer is unitary and every norm is 1.
    ///
    /// **And that is nearly all it proves, which is the point of the second half.** A norm check
    /// cannot see a wrong angle: `e^{-i gamma C}`, `e^{+i gamma C}` and `e^{-i 2 gamma C}` are all
    /// unitary. Conjugating the cost phase is the same computation as negating every `gamma`, so
    /// the state a sign-flipped implementation would produce is reachable here without touching the
    /// source — and it is asserted to have perfect norms and a `<C>` that is wrong by 9.
    #[test]
    fn every_layer_is_unitary_and_unitarity_alone_cannot_see_a_conjugated_cost_phase() {
        for n in [4usize, 7, 12] {
            let g = ring(n, -1.0, 0.31);
            let a = Ansatz::new(&g).unwrap();
            let ang = Angles::new(vec![0.7, -1.3, 2.2], vec![0.4, 1.1, -0.9]).unwrap();
            let r = a.run(&ang).unwrap();
            assert_eq!(r.layer_norms().len(), 2 * ang.p() + 1, "one norm per operator, plus |+>");
            for (k, &nm) in r.layer_norms().iter().enumerate() {
                // Measured worst 8.88e-16 over these cases.
                assert!((nm - 1.0).abs() < 1e-14, "n={n}, after operator {k}: norm {nm}");
            }
            let mass = crate::round::sum_up(r.probabilities());
            assert!((mass - 1.0).abs() < 1e-14, "n={n}: probabilities sum to {mass}");
            assert!(r.probabilities().iter().all(|&p| p >= 0.0));
        }

        let g = ring(12, -1.0, 0.0);
        let a = Ansatz::new(&g).unwrap();
        let right = a.run(&Angles::new(vec![0.55], vec![0.3]).unwrap()).unwrap();
        let flipped = a.run(&Angles::new(vec![-0.55], vec![0.3]).unwrap()).unwrap();
        for (k, &nm) in flipped.layer_norms().iter().enumerate() {
            assert!(
                (nm - 1.0).abs() < 1e-14,
                "the wrong-sign state is exactly as unitary as the right one, and this assertion \
                 is where that is written down: operator {k} norm {nm}"
            );
        }
        assert!(
            (right.expectation() - flipped.expectation()).abs() > 1.0,
            "conjugating the cost phase must move <C>, or this instance cannot tell them apart"
        );
    }

    /// ORACLE: the algebra of a uniform distribution over spins, which needs no simulation at all.
    ///
    /// With no layers the state is `|+>^n`, every basis state carries probability `2^-n`, and
    /// `<C> = -sum J <s_i s_j> - sum h <s_i> = 0` **exactly** for any Ising model without a
    /// constant term, because a uniform distribution over `{-1,+1}^n` has `<s_i> = <s_i s_j> = 0`.
    /// So it is a closed form that holds whatever the couplings are, checked on a random glass with
    /// fields where nothing else about the answer could be guessed.
    ///
    /// For even `n` the amplitude `2^{-n/2}` is exactly representable, so the probabilities are
    /// asserted with `==` rather than a tolerance.
    #[test]
    fn the_zero_level_ansatz_is_the_uniform_superposition_with_a_mean_energy_of_exactly_zero() {
        for n in [4usize, 8, 12] {
            let g = glass(n, 3);
            let a = Ansatz::new(&g).unwrap();
            assert_eq!(a.n(), n);
            assert_eq!(a.diagonal().len(), 1 << n);
            assert!(core::ptr::eq(a.graph(), &g));

            let none = Angles::new(Vec::new(), Vec::new()).unwrap();
            assert_eq!(none.p(), 0);
            let r = a.run(&none).unwrap();
            assert_eq!(r.layer_norms().len(), 1, "no layers, so only the prepared state");

            let want = 2f64.powi(-(n as i32));
            for (z, &p) in r.probabilities().iter().enumerate() {
                assert_eq!(p, want, "basis state {z} of the uniform superposition");
            }
            // Measured 4.16e-17 worst over these three sizes.
            assert!(
                r.expectation().abs() < 1e-15,
                "n={n}: the mean energy of a uniform distribution is zero, and this is {}",
                r.expectation()
            );
            // The same statement read off the enumerated diagonal, which is where it comes from.
            let mean = crate::round::sum_up(a.diagonal()) / a.diagonal().len() as f64;
            assert!(mean.abs() < 1e-12, "n={n}: enumerated mean {mean}");
        }
    }

    /// Appending `gamma = beta = 0` is bit-identical, and that is what makes the ladder monotone.
    ///
    /// `cos 0` is exactly 1 and `sin 0` is exactly 0, so the extra layer multiplies every amplitude
    /// by one and adds zero: not "close to the same state", the same state. If this ever stops
    /// holding, [`Optimiser::ladder`]'s monotonicity stops being a guarantee and becomes a hope.
    #[test]
    fn an_appended_identity_layer_is_bit_identical_and_no_search_from_it_comes_back_worse() {
        let g = glass(9, 2);
        let a = Ansatz::new(&g).unwrap();
        let ang = Angles::new(vec![0.9, -0.4], vec![1.7, 0.25]).unwrap();
        let base = a.expectation(&ang).unwrap();
        let grown = ang.with_identity_layer();
        assert_eq!(grown.p(), ang.p() + 1);
        assert_eq!(grown.gamma()[2], 0.0);
        assert_eq!(grown.beta()[2], 0.0);
        assert_eq!(a.expectation(&grown).unwrap(), base, "an identity layer is not approximate");

        let o = Optimiser { restarts: 1, max_evals: 300, ..Optimiser::default() };
        let from = o.optimise_from(&a, &grown).unwrap();
        assert!(
            from.expectation <= base,
            "a search started at {base} came back with {}",
            from.expectation
        );
        assert_eq!(from.angles.p(), 3);
    }

    /// Every refusal names what it saw, and a NaN angle is refused rather than defaulted.
    ///
    /// The last part is the one that matters: an [`Angles`] whose constructor was bypassed still
    /// cannot reach the state vector, because treating a NaN `gamma` as zero would return the
    /// `p-1` answer under a `p`-level name.
    #[test]
    fn every_refusal_names_what_it_saw() {
        let big = GraphBuilder::new(MAX_QUBITS + 1).build();
        let e = Ansatz::new(&big).unwrap_err();
        assert_eq!(e, Error::TooManyQubits { n: MAX_QUBITS + 1, max: MAX_QUBITS });
        assert!(e.to_string().contains("21"), "{e}");

        assert_eq!(Ansatz::new(&GraphBuilder::new(0).build()).unwrap_err(), Error::NoQubits);

        let mut b = GraphBuilder::new(3);
        b.couple(0, 1, f64::NAN);
        let nan_model = b.build();
        match Ansatz::new(&nan_model).unwrap_err() {
            Error::EnergyNotFinite { state, value } => {
                assert_eq!(state, 0);
                assert!(value.is_nan());
            }
            other => panic!("wrong refusal: {other}"),
        }

        assert_eq!(
            Angles::new(vec![0.1, 0.2], vec![0.3]).unwrap_err(),
            Error::LayerMismatch { gammas: 2, betas: 1 }
        );
        assert_eq!(
            Angles::from_flat(&[0.1, 0.2, 0.3]).unwrap_err(),
            Error::FlatLengthOdd { len: 3 }
        );
        assert_eq!(
            Angles::new(vec![0.1, f64::INFINITY], vec![0.0, 0.0]).unwrap_err(),
            Error::AngleNotFinite { which: "gamma", layer: 1, value: f64::INFINITY }
        );
        match Angles::new(vec![0.0], vec![f64::NAN]).unwrap_err() {
            Error::AngleNotFinite { which, layer, value } => {
                assert_eq!((which, layer), ("beta", 0));
                assert!(value.is_nan());
            }
            other => panic!("wrong refusal: {other}"),
        }
        for e in [
            Error::NoQubits,
            Error::LayerMismatch { gammas: 2, betas: 1 },
            Error::FlatLengthOdd { len: 3 },
            Error::AngleNotFinite { which: "beta", layer: 4, value: f64::NAN },
            Error::EnergyNotFinite { state: 7, value: f64::INFINITY },
        ] {
            assert!(e.to_string().len() > 40, "a refusal must say what happened: {e}");
        }

        // Past the constructor, which is exactly where a default would otherwise be substituted.
        let g = ring(6, -1.0, 0.0);
        let a = Ansatz::new(&g).unwrap();
        let smuggled = Angles { gamma: vec![f64::NAN], beta: vec![0.3] };
        assert!(a.run(&smuggled).is_err(), "run must re-check");
        assert!(a.expectation(&smuggled).is_err(), "so must the objective");
        assert!(Optimiser::default().optimise_from(&a, &smuggled).is_err());
    }

    /// Measurement reproduces the amplitudes it was drawn from, and repeats on its seed.
    #[test]
    fn measurement_reproduces_the_amplitudes_it_was_drawn_from_and_repeats_on_a_seed() {
        let g = ring(8, -1.0, 0.0);
        let a = Ansatz::new(&g).unwrap();
        let best = Optimiser { restarts: 6, ..Optimiser::default() }.optimise(&a, 2).unwrap();
        let r = a.run(&best.angles).unwrap();

        let shots = 40_000usize;
        let draws = r.measure(shots, 9);
        assert_eq!(draws.len(), shots);
        assert!(draws.iter().all(|s| s.len() == 8 && s.iter().all(|&v| v == 1 || v == -1)));

        let mut counts = vec![0usize; 1 << 8];
        for s in &draws {
            let mut z = 0usize;
            for (i, &v) in s.iter().enumerate() {
                if v > 0 {
                    z |= 1 << i;
                }
            }
            counts[z] += 1;
        }
        // Measured 1.72e-3 over all 256 outcomes.
        let worst = (0..1usize << 8)
            .map(|z| (counts[z] as f64 / shots as f64 - r.probabilities()[z]).abs())
            .fold(0.0f64, f64::max);
        assert!(worst < 0.01, "empirical frequencies are off the amplitudes by {worst}");

        // And the shot mean estimates <C>, which is the only reason a shot is worth taking.
        // Measured 3.6e-3 apart.
        let mean: f64 = draws.iter().map(|s| g.energy(s)).sum::<f64>() / shots as f64;
        assert!(
            (mean - r.expectation()).abs() < 0.1,
            "{shots} shots averaged {mean} against <C> = {}",
            r.expectation()
        );

        assert_eq!(r.measure(64, 9), r.measure(64, 9), "same seed, same draws");
        assert_ne!(r.measure(64, 9), r.measure(64, 10), "different seed, different draws");
    }

    /// The flat form round-trips, a transposed pair is a different ansatz, and the optimiser is
    /// deterministic by seed while actually reading it.
    #[test]
    fn angles_round_trip_and_the_optimiser_is_deterministic_by_seed() {
        let ang = Angles::new(vec![0.1, 0.2, 0.3], vec![-0.4, 0.5, -0.6]).unwrap();
        assert_eq!(ang.to_flat(), vec![0.1, 0.2, 0.3, -0.4, 0.5, -0.6]);
        assert_eq!(Angles::from_flat(&ang.to_flat()).unwrap(), ang);
        assert_eq!(ang.p(), 3);
        assert_eq!(ang.gamma(), &[0.1, 0.2, 0.3]);
        assert_eq!(ang.beta(), &[-0.4, 0.5, -0.6]);

        let g = ring(8, -1.0, 0.2);
        let a = Ansatz::new(&g).unwrap();
        // THE REASON THE TWO LISTS ARE NOT ONE LIST: swapping them is a different circuit.
        let swapped = Angles::new(ang.beta().to_vec(), ang.gamma().to_vec()).unwrap();
        assert!(
            (a.expectation(&ang).unwrap() - a.expectation(&swapped).unwrap()).abs() > 1e-3,
            "a transposed angle pair must not be the same ansatz"
        );

        let o = Optimiser { restarts: 4, ..Optimiser::default() };
        let x = o.optimise(&a, 2).unwrap();
        let y = o.optimise(&a, 2).unwrap();
        assert_eq!(x.expectation, y.expectation);
        assert_eq!(x.angles, y.angles);
        assert_eq!(x.evaluations, y.evaluations);
        let other = Optimiser { seed: o.seed ^ 0x5A5A, ..o.clone() }.optimise(&a, 2).unwrap();
        assert!(
            other.angles != x.angles || other.evaluations != x.evaluations,
            "two different seeds ran the identical search, so the seed is not being read"
        );
        assert_eq!(o, Optimiser { restarts: 4, ..Optimiser::default() });
    }
}
