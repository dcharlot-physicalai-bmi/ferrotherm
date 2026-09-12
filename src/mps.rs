//! Matrix product states and two-site DMRG: the variational tensor network for 1D ground states.
//!
//! [`crate::tensor`] contracts a network EXACTLY and says so — no bond-dimension truncation, nothing
//! approximated. That is the right tool when the treewidth is small and it is useless on a chain of
//! a hundred spins, where the exact contraction of an amplitude network is `2^50` at the middle cut.
//! What the field actually runs is the other thing: White's density-matrix renormalisation group
//! (Phys. Rev. Lett. 69, 2863, 1992), re-derived a decade later as variational optimisation over
//! matrix product states (Östlund & Rommer 1995; Schollwöck, Ann. Phys. 326, 96, 2011). A **bond
//! dimension** caps how much entanglement the ansatz may carry, and the cap is what makes the method
//! polynomial. This crate had neither the state nor the sweep.
//!
//! # What is here
//!
//! * [`Mps`] — a real, open-boundary matrix product state (a "tensor train") in **mixed canonical
//!   form**: every tensor left of the orthogonality centre is a left isometry, every tensor right of
//!   it a right isometry. That is not decoration. It makes the Schmidt values at a bond readable
//!   straight off the state, makes the two-site effective Hamiltonian a plain symmetric matrix, and
//!   makes `⟨ψ|ψ⟩` the norm of one tensor rather than a contraction of the whole chain.
//! * [`Tfim`] — the 1D transverse-field Ising Hamiltonian as a bond-dimension-3 matrix product
//!   operator, in **this crate's sign convention**:
//!
//!   ```text
//!     H = −J Σ_{i=1}^{N−1} Z_i Z_{i+1}  −  g Σ_{i=1}^{N} X_i  −  h Σ_{i=1}^{N} Z_i
//!   ```
//!
//!   so at `g = 0` it is exactly [`crate::graph::Graph::energy`] on a chain — positive `J`
//!   ferromagnetic, `h` the longitudinal field — and `g` is the one term that is not classical.
//! * [`ground_state`] — two-site DMRG with singular-value truncation, sweeping right then left,
//!   reporting what happened at every local update rather than only at the end.
//!
//! # Boundary conditions, and the closed form that does NOT apply to them
//!
//! The chain here is **open**. Jordan–Wigner still solves it exactly, but not with the momentum sum
//! that gets quoted. Writing the Majorana form of `H` (`b_i = a_{2i−1}`, `c_i = a_{2i}`) the
//! Hamiltonian is `H = i Σ_{ij} b_i K_{ij} c_j` with
//!
//! ```text
//!     K_ii = g,   K_{i,i−1} = J,   everything else zero,
//!     E_0  = − Σ_m σ_m(K)          (the singular values of that bidiagonal N×N matrix)
//! ```
//!
//! — [`Tfim::free_fermion_ground_energy`]. The familiar
//! `E_0 = −J Σ_k sqrt(1 + g² − 2g cos k)` with **equally spaced** `k` is the PERIODIC chain, with
//! antiperiodic momenta `k_m = (2m+1)π/N`; on an open chain the allowed momenta solve a
//! transcendental equation (Pfeuty, Ann. Phys. 57, 79, 1970) and are not equally spaced. The
//! difference is not small: at `N = 8`, `J = g = 1` the open chain's exact energy is `−9.837951`
//! and the equally-spaced sum is `−10.251662`, off by `0.4137`. Both forms are computed and both are
//! checked against brute-force diagonalisation in this module's tests, so the distinction is
//! measured here rather than asserted.
//!
//! # What the energy is, and what it is not
//!
//! DMRG is variational: `⟨ψ|H|ψ⟩/⟨ψ|ψ⟩ ≥ E_0` for every state the ansatz can write down, at every
//! sweep, at every bond dimension. [`Ground::certified`] is that upper bound accumulated through
//! [`crate::round`], so the claim survives floating point.
//!
//! **It is not monotone once truncation bites.** Each local update is an exact minimisation over a
//! space containing the current state, so the Lanczos step can only lower the energy — but the
//! singular-value truncation that follows throws part of that state away, and throwing part of a
//! state away can raise `⟨H⟩`. Measured here on a 12-site critical chain at bond dimension 2:
//! **70 of 168 local updates raised the energy**, the worst by `3.5e-5`. With the cap slack enough
//! that nothing is discarded the sequence is monotone to `1e-12`, which is what
//! [`Sweep::truncation`] exists to tell a caller apart.
//!
//! **And two-site DMRG does not reach the best state of its own bond dimension**, because the
//! rank-`D` truncation keeps the closest rank-`D` state rather than the cheapest one. At bond
//! dimension 1, where the answer is independently known, it stops as much as `1.2e-1` above the
//! product-state optimum and stays there however many sweeps are run. [`Options::polish`] runs
//! single-site sweeps, which change no bond dimension and therefore truncate nothing, and enough of
//! them close that gap to `2e-14`.
//!
//! ```
//! use ferrotherm::mps::{Options, Tfim, ground_state};
//!
//! let h = Tfim::new(16, 1.0, 1.0, 0.0).unwrap();
//! let opts = Options::new(32, 1e-12, 30, 20, 2).unwrap();
//! let out = ground_state(&h, &opts, 7).unwrap();
//!
//! let exact = h.free_fermion_ground_energy().unwrap();
//! assert!((out.energy - exact).abs() < 1e-10, "{} vs {exact}", out.energy);
//! assert!(out.certified >= exact, "a variational energy is an upper bound");
//! ```

use crate::linalg::jacobi_eig;
use crate::rng::Pcg;
use crate::round::{accumulation_guard, sum_down, sum_up};

/// Physical dimension of one site. A spin-1/2 has two states and this module is about spins.
const PHYS: usize = 2;

/// Bond dimension of the transverse-field Ising matrix product operator.
///
/// Three, and it is three for a structural reason rather than a tuning one: one channel to carry
/// "nothing has happened yet", one to carry "a `Z` is waiting for its partner", one to carry
/// "the Hamiltonian term is already placed". Any nearest-neighbour Hamiltonian with one two-site
/// term needs exactly these.
const MPO_BOND: usize = 3;

/// The `σ^z` eigenvalue of physical index `s`: index 0 is spin up, index 1 is spin down.
///
/// Stated as a function rather than inlined because the mapping is a CONVENTION, and this crate has
/// already been bitten once by a decoder that handed back `0`/`1` where the model said `−1`/`+1`.
#[inline]
fn z_of(s: usize) -> f64 {
    if s == 0 { 1.0 } else { -1.0 }
}

/// Which parameter a non-finite value arrived on.
///
/// Named rather than a string so the error can be matched on, and so the message can say what was
/// actually sent — a `NaN` coupling and a `NaN` field are different mistakes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Parameter {
    /// The Ising coupling `J`.
    Coupling,
    /// The transverse field `g`, the coefficient of `Σ X_i`.
    TransverseField,
    /// The longitudinal field `h`, the coefficient of `Σ Z_i`.
    LongitudinalField,
    /// The relative singular-value cutoff.
    Cutoff,
}

impl core::fmt::Display for Parameter {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let name = match self {
            Parameter::Coupling => "the coupling J",
            Parameter::TransverseField => "the transverse field g",
            Parameter::LongitudinalField => "the longitudinal field h",
            Parameter::Cutoff => "the singular-value cutoff",
        };
        f.write_str(name)
    }
}

/// Why a Hamiltonian or a sweep schedule could not be built.
///
/// Every variant carries the value it saw. A default bond dimension or a silently clamped cutoff
/// would let a caller believe it ran the calculation it asked for.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MpsError {
    /// Fewer than two sites. Two-site DMRG updates a PAIR, so a chain has to have one.
    TooFewSites {
        /// Sites requested.
        n: usize,
    },
    /// A Hamiltonian parameter is `NaN` or infinite.
    NotFinite {
        /// Which parameter.
        parameter: Parameter,
        /// The value that arrived.
        value: f64,
    },
    /// A bond dimension of zero, which describes no state at all.
    ZeroBond,
    /// The cutoff is not a fraction in `[0, 1)` of the largest singular value.
    BadCutoff {
        /// The cutoff that arrived.
        cutoff: f64,
    },
    /// Zero sweeps. The answer would be the random initial state's energy, which estimates nothing.
    NoSweeps,
    /// Fewer than two Krylov vectors, which cannot improve on the starting vector.
    BadKrylov {
        /// The Krylov dimension that arrived.
        krylov: usize,
    },
}

impl core::fmt::Display for MpsError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            MpsError::TooFewSites { n } => write!(
                f,
                "a chain of {n} sites has no nearest-neighbour pair; two-site DMRG needs at least 2"
            ),
            MpsError::NotFinite { parameter, value } => {
                write!(f, "{parameter} is {value}, which is not a finite number")
            }
            MpsError::ZeroBond => {
                write!(f, "a maximum bond dimension of 0 describes no state; the smallest is 1")
            }
            MpsError::BadCutoff { cutoff } => write!(
                f,
                "the cutoff {cutoff} is not in [0, 1); it is a fraction of the largest singular \
                 value, and 1 would discard the whole state"
            ),
            MpsError::NoSweeps => write!(
                f,
                "zero sweeps were requested, so the reported energy would be the random initial \
                 state's and would estimate nothing"
            ),
            MpsError::BadKrylov { krylov } => write!(
                f,
                "a Krylov dimension of {krylov} cannot improve on its own starting vector; the \
                 smallest useful value is 2"
            ),
        }
    }
}

impl core::error::Error for MpsError {}

/// The 1D transverse-field Ising Hamiltonian, open boundaries, as a matrix product operator.
///
/// ```text
///   H = −J Σ_{i<N−1} Z_i Z_{i+1} − g Σ_i X_i − h Σ_i Z_i
/// ```
///
/// The signs are [`crate::graph`]'s: positive `J` is ferromagnetic and `h` pulls spins up, so at
/// `g = 0` this is the classical chain the rest of the crate already solves, and `g` is the only
/// term that makes it quantum.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Tfim {
    n: usize,
    j: f64,
    g: f64,
    h: f64,
}

impl Tfim {
    /// A chain of `n` sites with coupling `j`, transverse field `g` and longitudinal field `h`.
    ///
    /// # Errors
    ///
    /// [`MpsError::TooFewSites`] below two sites, [`MpsError::NotFinite`] naming any parameter that
    /// is `NaN` or infinite.
    pub fn new(n: usize, j: f64, g: f64, h: f64) -> Result<Tfim, MpsError> {
        if n < 2 {
            return Err(MpsError::TooFewSites { n });
        }
        for (parameter, value) in [
            (Parameter::Coupling, j),
            (Parameter::TransverseField, g),
            (Parameter::LongitudinalField, h),
        ] {
            if !value.is_finite() {
                return Err(MpsError::NotFinite { parameter, value });
            }
        }
        Ok(Tfim { n, j, g, h })
    }

    /// Sites in the chain.
    #[must_use]
    pub fn n(&self) -> usize {
        self.n
    }

    /// The Ising coupling `J`.
    #[must_use]
    pub fn coupling(&self) -> f64 {
        self.j
    }

    /// The transverse field `g`.
    #[must_use]
    pub fn transverse_field(&self) -> f64 {
        self.g
    }

    /// The longitudinal field `h`.
    #[must_use]
    pub fn longitudinal_field(&self) -> f64 {
        self.h
    }

    /// The site tensor `W`, indexed `((w_left · 2 + s) · 2 + s') · 3 + w_right`, holding
    /// `⟨s| O_{w_left, w_right} |s'⟩`.
    ///
    /// One tensor for every site; the boundaries are handled by the environment vectors
    /// `(0, 0, 1)` on the left and `(1, 0, 0)` on the right, which is the standard
    ///
    /// ```text
    ///   W = [[ I,        0,     0 ],
    ///        [ Z,        0,     0 ],
    ///        [ −gX−hZ,  −JZ,    I ]]
    /// ```
    ///
    /// upper-triangular form. Multiplying `N` copies between those vectors reproduces `H` term for
    /// term, which is what the `mpo_reproduces_the_dense_hamiltonian` test checks against a Pauli
    /// matrix built by hand.
    #[must_use]
    pub fn mpo(&self) -> Vec<f64> {
        let mut w = vec![0.0; MPO_BOND * PHYS * PHYS * MPO_BOND];
        let mut put = |wl: usize, s: usize, sp: usize, wr: usize, v: f64| {
            w[((wl * PHYS + s) * PHYS + sp) * MPO_BOND + wr] += v;
        };
        for s in 0..PHYS {
            // row 0: the identity channel, before any term has been placed.
            put(0, s, s, 0, 1.0);
            // row 1: a dangling Z waiting for its right partner.
            put(1, s, s, 0, z_of(s));
            // row 2, column 2: the identity channel after the term is placed.
            put(2, s, s, 2, 1.0);
            // row 2, column 0: the on-site terms.
            put(2, s, s, 0, -self.h * z_of(s));
            // row 2, column 1: the left half of the ZZ bond term.
            put(2, s, s, 1, -self.j * z_of(s));
        }
        // −g X is the only off-diagonal entry in the whole operator.
        put(2, 0, 1, 0, -self.g);
        put(2, 1, 0, 0, -self.g);
        w
    }

    /// The **exact** ground energy of the open chain, by Jordan–Wigner.
    ///
    /// `E_0 = −Σ_m σ_m(K)` where `K` is the `N×N` bidiagonal matrix with `g` on the diagonal and
    /// `J` below it, and `σ_m` are its singular values. This is the Lieb–Schultz–Mattis /
    /// Pfeuty solution written in the form that needs no transcendental root-finding: the Majorana
    /// Hamiltonian `H = i Σ b_i K_ij c_j` is brought to `i Σ_m σ_m b'_m c'_m` by the singular value
    /// decomposition of `K`, and each of those two-Majorana modes contributes `−σ_m` in its ground
    /// state.
    ///
    /// Returns `None` when a longitudinal field is present: `h Σ Z_i` is a quartic term in the
    /// fermions and Jordan–Wigner does not solve it. A number computed as though `h` were zero would
    /// be indistinguishable from the exact one and would be wrong, so there is no number.
    #[must_use]
    pub fn free_fermion_ground_energy(&self) -> Option<f64> {
        if self.h != 0.0 {
            return None;
        }
        let n = self.n;
        // K^T K is tridiagonal: (g² + J²) on the diagonal except the last entry, which is g²
        // because the last column of K has no sub-diagonal partner, and gJ off it.
        let mut gram = vec![0.0; n * n];
        for i in 0..n {
            gram[i * n + i] = if i + 1 == n { self.g * self.g } else { self.g * self.g + self.j * self.j };
        }
        for i in 0..n - 1 {
            gram[i * n + i + 1] = self.g * self.j;
            gram[(i + 1) * n + i] = self.g * self.j;
        }
        let _ = jacobi_eig(&mut gram, n);
        let mut total = Vec::with_capacity(n);
        for i in 0..n {
            total.push(-gram[i * n + i].max(0.0).sqrt());
        }
        Some(sum_up(&total))
    }

    /// The **periodic** chain's ground energy from the momentum closed form, for comparison.
    ///
    /// `E_0 = −J Σ_m sqrt(1 + (g/J)² − 2(g/J) cos k_m)` over the antiperiodic (Neveu–Schwarz)
    /// momenta `k_m = (2m+1)π/N`, `m = 0 … N−1`. This is the formula usually quoted for "the" 1D
    /// transverse-field Ising ground energy, and it belongs to a RING, not to the open chain
    /// [`Tfim`] describes — see the module note. Provided so the two can be told apart by
    /// measurement instead of by memory.
    ///
    /// Returns `None` for `h ≠ 0` (Jordan–Wigner does not apply) or `J = 0` (there is no ring).
    #[must_use]
    pub fn periodic_ground_energy(&self) -> Option<f64> {
        if self.h != 0.0 || self.j == 0.0 {
            return None;
        }
        let ratio = self.g / self.j;
        let mut terms = Vec::with_capacity(self.n);
        for m in 0..self.n {
            let k = (2.0 * m as f64 + 1.0) * core::f64::consts::PI / self.n as f64;
            terms.push(-self.j.abs() * (1.0 + ratio * ratio - 2.0 * ratio * k.cos()).max(0.0).sqrt());
        }
        Some(sum_up(&terms))
    }
}

/// A real open-boundary matrix product state in mixed canonical form.
///
/// Site `i` is a tensor of shape `(D_i, 2, D_{i+1})` stored row-major, with `D_0 = D_n = 1`. Sites
/// left of [`Mps::centre`] satisfy the LEFT isometry condition `Σ_{a,s} A[a,s,b] A[a,s,b'] = δ_bb'`
/// and sites right of it the RIGHT one `Σ_{s,b} A[a,s,b] A[a',s,b] = δ_aa'`; the centre carries the
/// whole norm. [`Mps::isometry_defect`] measures how well that holds, and the answer in this module
/// is machine precision, not "close enough".
#[derive(Clone, Debug)]
pub struct Mps {
    n: usize,
    dims: Vec<usize>,
    a: Vec<Vec<f64>>,
    schmidt: Vec<Vec<f64>>,
    centre: usize,
}

impl Mps {
    /// A random state with bond dimensions `min(2^i, 2^(n−i), max_bond)`, right-canonicalised so the
    /// orthogonality centre sits on site 0.
    ///
    /// The bond profile matters: a chain of `n` sites cannot carry more than `2^i` Schmidt values at
    /// its `i`-th bond, so allocating `max_bond` everywhere would ship rows that are zero by
    /// construction and make the isometry condition unsatisfiable at the ends.
    ///
    /// # Panics
    ///
    /// If `n < 2` or `max_bond == 0`. Both are refused earlier by [`Tfim::new`] and
    /// [`Options::new`]; this constructor is public because a caller may want a state without a
    /// Hamiltonian, and it states its own preconditions rather than inventing a value for them.
    #[must_use]
    pub fn random(n: usize, max_bond: usize, seed: u64) -> Mps {
        assert!(n >= 2, "an MPS needs at least two sites, got {n}");
        assert!(max_bond >= 1, "a bond dimension of 0 describes no state");
        let mut rng = Pcg::new(seed, 0x4D_50_53);
        let cap = |k: usize| -> usize {
            if k >= 63 { max_bond } else { max_bond.min(1usize << k) }
        };
        let dims: Vec<usize> = (0..=n).map(|i| cap(i).min(cap(n - i))).collect();
        let mut a = Vec::with_capacity(n);
        for i in 0..n {
            let len = dims[i] * PHYS * dims[i + 1];
            let mut t: Vec<f64> = (0..len).map(|_| rng.f64() - 0.5).collect();
            let nrm = t.iter().map(|x| x * x).sum::<f64>().sqrt();
            for x in &mut t {
                *x /= nrm;
            }
            a.push(t);
        }
        let schmidt = (0..=n).map(|_| vec![1.0]).collect();
        let mut mps = Mps { n, dims, a, schmidt, centre: n - 1 };
        for i in (1..n).rev() {
            mps.shift_left(i, usize::MAX, 0.0);
        }
        mps.normalise_centre();
        mps
    }

    /// Sites in the chain.
    #[must_use]
    pub fn n(&self) -> usize {
        self.n
    }

    /// The orthogonality centre: the one site that is not an isometry.
    #[must_use]
    pub fn centre(&self) -> usize {
        self.centre
    }

    /// Dimension of bond `b`, where bond `0` is the left edge and bond `n` the right edge.
    #[must_use]
    pub fn bond_dim(&self, b: usize) -> Option<usize> {
        self.dims.get(b).copied()
    }

    /// The largest bond dimension anywhere in the chain.
    #[must_use]
    pub fn max_bond(&self) -> usize {
        self.dims.iter().copied().max().unwrap_or(0)
    }

    /// Site `i`'s tensor, shape `(D_i, 2, D_{i+1})` row-major.
    #[must_use]
    pub fn tensor(&self, i: usize) -> Option<&[f64]> {
        self.a.get(i).map(Vec::as_slice)
    }

    /// The Schmidt values across bond `b`, descending, normalised to `Σ s² = 1`.
    ///
    /// Edge bonds carry the single value `1`. An interior bond carries whatever the last split
    /// produced there, so on a freshly built state it is the random state's spectrum and after
    /// [`ground_state`] it is the ground state's.
    #[must_use]
    pub fn schmidt(&self, b: usize) -> Option<&[f64]> {
        self.schmidt.get(b).map(Vec::as_slice)
    }

    /// Von Neumann entanglement entropy `−Σ s² ln s²` across bond `b`, in nats.
    ///
    /// Bounded by `ln D_b` by construction, so a truncated state under-reports the true entropy —
    /// which is the honest direction and is why [`Sweep::truncation`] is reported beside it.
    #[must_use]
    pub fn entanglement_entropy(&self, b: usize) -> Option<f64> {
        let s = self.schmidt.get(b)?;
        let mut acc = 0.0;
        for &v in s {
            let p = v * v;
            if p > 0.0 {
                acc -= p * p.ln();
            }
        }
        Some(acc)
    }

    /// `⟨ψ|ψ⟩`, read off the orthogonality centre.
    ///
    /// One tensor's worth of arithmetic rather than a contraction of the chain — which is exactly
    /// what canonical form buys, and is only true while the isometries hold.
    #[must_use]
    pub fn norm_squared(&self) -> f64 {
        sum_up(&self.a[self.centre].iter().map(|x| x * x).collect::<Vec<f64>>())
    }

    /// The largest violation of the canonical form anywhere: isometry defects off the centre, and
    /// `|‖ψ‖² − 1|` at it.
    ///
    /// An exact quantity with an exact target. A number here above `1e-13` means the state is not
    /// the thing the rest of this module assumes it is, and every energy read off it is wrong in a
    /// way no energy comparison would reveal.
    #[must_use]
    pub fn isometry_defect(&self) -> f64 {
        let mut worst = (self.norm_squared() - 1.0).abs();
        for i in 0..self.n {
            if i == self.centre {
                continue;
            }
            let (dl, dr) = (self.dims[i], self.dims[i + 1]);
            let t = &self.a[i];
            if i < self.centre {
                for b in 0..dr {
                    for bp in 0..dr {
                        let mut acc = 0.0;
                        for aa in 0..dl {
                            for s in 0..PHYS {
                                acc += t[(aa * PHYS + s) * dr + b] * t[(aa * PHYS + s) * dr + bp];
                            }
                        }
                        let want = f64::from(u8::from(b == bp));
                        worst = worst.max((acc - want).abs());
                    }
                }
            } else {
                for aa in 0..dl {
                    for ap in 0..dl {
                        let mut acc = 0.0;
                        for s in 0..PHYS {
                            for b in 0..dr {
                                acc += t[(aa * PHYS + s) * dr + b] * t[(ap * PHYS + s) * dr + b];
                            }
                        }
                        let want = f64::from(u8::from(aa == ap));
                        worst = worst.max((acc - want).abs());
                    }
                }
            }
        }
        worst
    }

    /// The amplitude `⟨s_0 s_1 … | ψ⟩` for one spin configuration, `0` = up and `1` = down.
    ///
    /// Returns `None` if the configuration is not one index per site or names a state that is not
    /// `0` or `1` — a silently wrapped index would return an amplitude belonging to a different
    /// configuration.
    #[must_use]
    pub fn amplitude(&self, config: &[u8]) -> Option<f64> {
        if config.len() != self.n || config.iter().any(|&s| s > 1) {
            return None;
        }
        let mut row = vec![1.0];
        for i in 0..self.n {
            let (dl, dr) = (self.dims[i], self.dims[i + 1]);
            let s = config[i] as usize;
            let mut next = vec![0.0; dr];
            for aa in 0..dl {
                let v = row[aa];
                if v == 0.0 {
                    continue;
                }
                for b in 0..dr {
                    next[b] += v * self.a[i][(aa * PHYS + s) * dr + b];
                }
            }
            row = next;
        }
        Some(row[0])
    }

    /// The full `2^n` amplitude vector, index `Σ_i s_i · 2^(n−1−i)` — site 0 is the most
    /// significant bit.
    ///
    /// Returns `None` above 20 sites, where the vector is the thing an MPS exists to avoid
    /// building. For comparing against exact diagonalisation on a small chain, which is the only
    /// reason this exists.
    #[must_use]
    pub fn to_dense(&self) -> Option<Vec<f64>> {
        if self.n > 20 {
            return None;
        }
        let mut cur = vec![1.0];
        let mut cur_dim = 1usize;
        for i in 0..self.n {
            let (dl, dr) = (self.dims[i], self.dims[i + 1]);
            let mut next = vec![0.0; cur_dim * PHYS * dr];
            for c in 0..cur_dim {
                for aa in 0..dl {
                    let v = cur[c * dl + aa];
                    if v == 0.0 {
                        continue;
                    }
                    for s in 0..PHYS {
                        for b in 0..dr {
                            next[(c * PHYS + s) * dr + b] += v * self.a[i][(aa * PHYS + s) * dr + b];
                        }
                    }
                }
            }
            cur = next;
            cur_dim *= PHYS;
        }
        Some(cur)
    }

    /// Divide the centre tensor by its norm, making `‖ψ‖ = 1` to machine precision.
    fn normalise_centre(&mut self) {
        let t = &mut self.a[self.centre];
        let nrm = t.iter().map(|x| x * x).sum::<f64>().sqrt();
        if nrm > 0.0 {
            for x in t.iter_mut() {
                *x /= nrm;
            }
        }
    }

    /// Move the orthogonality centre from `i` to `i+1`, splitting site `i` and pushing the
    /// remainder right. Returns the discarded weight.
    fn shift_right(&mut self, i: usize, max_bond: usize, cutoff: f64) -> f64 {
        let (dl, dr) = (self.dims[i], self.dims[i + 1]);
        let (rows, cols) = (dl * PHYS, dr);
        let (u, s, vt, rank) = svd(&self.a[i], rows, cols);
        let (keep, discarded) = truncate(&s, rank, max_bond, cutoff);
        let scale = renorm(&s, keep);
        let mut left = vec![0.0; rows * keep];
        for r in 0..rows {
            for t in 0..keep {
                left[r * keep + t] = u[r * s.len() + t];
            }
        }
        // carry = diag(s) · Vᵀ, the part of site i that moves onto site i+1
        let mut carry = vec![0.0; keep * cols];
        for t in 0..keep {
            for c in 0..cols {
                carry[t * cols + c] = s[t] * scale * vt[t * cols + c];
            }
        }
        let dr2 = self.dims[i + 2];
        let merged = matmul(&carry, keep, cols, &self.a[i + 1], cols, PHYS * dr2);
        self.a[i] = left;
        self.a[i + 1] = merged;
        self.dims[i + 1] = keep;
        self.schmidt[i + 1] = (0..keep).map(|t| s[t] * scale).collect();
        self.centre = i + 1;
        discarded
    }

    /// Move the orthogonality centre from `i` to `i−1`. Returns the discarded weight.
    fn shift_left(&mut self, i: usize, max_bond: usize, cutoff: f64) -> f64 {
        let (dl, dr) = (self.dims[i], self.dims[i + 1]);
        let (rows, cols) = (dl, PHYS * dr);
        let (u, s, vt, rank) = svd(&self.a[i], rows, cols);
        let (keep, discarded) = truncate(&s, rank, max_bond, cutoff);
        let scale = renorm(&s, keep);
        let mut right = vec![0.0; keep * cols];
        right[..keep * cols].copy_from_slice(&vt[..keep * cols]);
        let mut carry = vec![0.0; rows * keep];
        for r in 0..rows {
            for t in 0..keep {
                carry[r * keep + t] = u[r * s.len() + t] * s[t] * scale;
            }
        }
        let dl2 = self.dims[i - 1];
        let merged = matmul(&self.a[i - 1], dl2 * PHYS, rows, &carry, rows, keep);
        self.a[i] = right;
        self.a[i - 1] = merged;
        self.dims[i] = keep;
        self.schmidt[i] = (0..keep).map(|t| s[t] * scale).collect();
        self.centre = i - 1;
        discarded
    }

    /// Put the centre on site `target` without discarding anything.
    fn move_centre(&mut self, target: usize) {
        while self.centre < target {
            self.shift_right(self.centre, usize::MAX, 0.0);
        }
        while self.centre > target {
            self.shift_left(self.centre, usize::MAX, 0.0);
        }
    }
}

/// How many singular values to keep, and the weight thrown away by keeping that many.
///
/// `rank` is how many rows the orthonormalisation actually produced: past it the factor is
/// numerical noise, and keeping noise would break the isometry the whole module rests on. The
/// discarded weight is `Σ_{t ≥ keep} s_t² / Σ_t s_t²`, which is what the truncation costs the state.
fn truncate(s: &[f64], rank: usize, max_bond: usize, cutoff: f64) -> (usize, f64) {
    let total: f64 = s.iter().map(|x| x * x).sum();
    if total <= 0.0 {
        return (1, 0.0);
    }
    let floor = s[0] * 1e-14;
    let mut keep = 0;
    while keep < rank && keep < max_bond && s[keep] > floor && (keep == 0 || s[keep] > cutoff * s[0])
    {
        keep += 1;
    }
    keep = keep.max(1);
    let kept: f64 = s[..keep].iter().map(|x| x * x).sum();
    (keep, ((total - kept) / total).max(0.0))
}

/// The factor that renormalises a truncated Schmidt spectrum back to `Σ s² = 1`.
fn renorm(s: &[f64], keep: usize) -> f64 {
    let kept: f64 = s[..keep].iter().map(|x| x * x).sum();
    if kept > 0.0 { 1.0 / kept.sqrt() } else { 1.0 }
}

/// Row-major `(ar × ac) · (br × bc)`.
fn matmul(a: &[f64], ar: usize, ac: usize, b: &[f64], br: usize, bc: usize) -> Vec<f64> {
    debug_assert_eq!(ac, br);
    let mut out = vec![0.0; ar * bc];
    for i in 0..ar {
        for k in 0..ac {
            let v = a[i * ac + k];
            if v == 0.0 {
                continue;
            }
            for jj in 0..bc {
                out[i * bc + jj] += v * b[k * bc + jj];
            }
        }
    }
    out
}

/// Modified Gram–Schmidt over the rows of a `k × cols` matrix, run twice.
///
/// Returns how many leading rows came out orthonormal. Twice is not superstition: one pass of
/// Gram–Schmidt loses orthogonality proportionally to the condition number, and the second pass
/// recovers it to machine precision (Giraud, Langou & Rozložník 2005). This module's canonical-form
/// test asserts `1e-13`, and a single pass does not reach it on the ill-conditioned factors a
/// near-converged DMRG produces.
fn orthonormalise_rows(x: &mut [f64], k: usize, cols: usize) -> usize {
    let mut rank = 0;
    for t in 0..k {
        let before: f64 = x[t * cols..(t + 1) * cols].iter().map(|v| v * v).sum::<f64>().sqrt();
        for _pass in 0..2 {
            for u in 0..rank {
                let mut dot = 0.0;
                for c in 0..cols {
                    dot += x[u * cols + c] * x[t * cols + c];
                }
                for c in 0..cols {
                    x[t * cols + c] -= dot * x[u * cols + c];
                }
            }
        }
        let nrm: f64 = x[t * cols..(t + 1) * cols].iter().map(|v| v * v).sum::<f64>().sqrt();
        if nrm <= 1e-13 * before || nrm == 0.0 {
            for c in 0..cols {
                x[t * cols + c] = 0.0;
            }
            break;
        }
        for c in 0..cols {
            x[t * cols + c] /= nrm;
        }
        rank += 1;
    }
    rank
}

/// Thin singular value decomposition of a row-major `rows × cols` matrix.
///
/// Returns `(U, s, Vᵀ, rank)` with `U` of shape `rows × k`, `s` of length `k = min(rows, cols)`
/// descending, `Vᵀ` of shape `k × cols`, and `rank` the number of leading columns/rows that are
/// genuinely orthonormal.
///
/// **Built on [`crate::linalg::jacobi_eig`] rather than on a second decomposition.** The smaller
/// Gram matrix is diagonalised and the other factor recovered as `Mᵀu/σ`, then re-orthonormalised.
/// The cost is stated plainly: a singular value recovered as `sqrt(λ)` carries relative error
/// `≈ ε(σ_max/σ)²/2`, so the SMALL singular values lose precision. That is tolerable here and only
/// here, for two reasons — the small ones are the ones about to be truncated, and the energy is at a
/// variational minimum, where a state perturbation of size `δ` costs `O(δ²)` in energy. The isometry
/// conditions do NOT tolerate it, and they are restored exactly by the Gram–Schmidt pass rather than
/// inherited from the division.
fn svd(m: &[f64], rows: usize, cols: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, usize) {
    let k = rows.min(cols);
    let mut s = vec![0.0; k];
    let mut u = vec![0.0; rows * k];
    let mut vt = vec![0.0; k * cols];
    if k == 0 {
        return (u, s, vt, 0);
    }
    let small = rows.min(cols);
    let mut gram = vec![0.0; small * small];
    if rows <= cols {
        for i in 0..rows {
            for j in i..rows {
                let mut acc = 0.0;
                for c in 0..cols {
                    acc += m[i * cols + c] * m[j * cols + c];
                }
                gram[i * rows + j] = acc;
                gram[j * rows + i] = acc;
            }
        }
    } else {
        for i in 0..cols {
            for j in i..cols {
                let mut acc = 0.0;
                for r in 0..rows {
                    acc += m[r * cols + i] * m[r * cols + j];
                }
                gram[i * cols + j] = acc;
                gram[j * cols + i] = acc;
            }
        }
    }
    let vecs = jacobi_eig(&mut gram, small);
    let mut order: Vec<usize> = (0..small).collect();
    order.sort_by(|&x, &y| gram[y * small + y].total_cmp(&gram[x * small + x]));
    for t in 0..k {
        s[t] = gram[order[t] * small + order[t]].max(0.0).sqrt();
    }
    let rank;
    if rows <= cols {
        for t in 0..k {
            for i in 0..rows {
                u[i * k + t] = vecs[i * small + order[t]];
            }
        }
        for t in 0..k {
            for c in 0..cols {
                let mut acc = 0.0;
                for i in 0..rows {
                    acc += m[i * cols + c] * u[i * k + t];
                }
                vt[t * cols + c] = acc;
            }
        }
        rank = orthonormalise_rows(&mut vt, k, cols);
    } else {
        for t in 0..k {
            for c in 0..cols {
                vt[t * cols + c] = vecs[c * small + order[t]];
            }
        }
        // Uᵀ first, so the same row-wise orthonormalisation applies, then transposed into place.
        let mut ut = vec![0.0; k * rows];
        for t in 0..k {
            for r in 0..rows {
                let mut acc = 0.0;
                for c in 0..cols {
                    acc += m[r * cols + c] * vt[t * cols + c];
                }
                ut[t * rows + r] = acc;
            }
        }
        rank = orthonormalise_rows(&mut ut, k, rows);
        for t in 0..k {
            for r in 0..rows {
                u[r * k + t] = ut[t * rows + r];
            }
        }
    }
    (u, s, vt, rank)
}

/// A symmetric linear operator known only by what it does to a vector.
///
/// Two implementations, one Lanczos. The one-site and two-site effective Hamiltonians differ only in
/// how many physical indices sit between the environments, and writing the Krylov loop twice is how
/// the two would come to disagree.
trait LinearOp {
    /// Length of the vectors this acts on.
    fn dim(&self) -> usize;
    /// `out <- H v`. `out` is `dim()` long and is overwritten, not accumulated into.
    fn apply(&self, v: &[f64], out: &mut [f64]);

    /// `⟨v|H|v⟩ / ⟨v|v⟩` — the Rayleigh quotient, which for exact environments is the FULL state's
    /// energy and not a local one.
    fn rayleigh(&self, v: &[f64]) -> f64 {
        let mut hv = vec![0.0; self.dim()];
        self.apply(v, &mut hv);
        let num: f64 = v.iter().zip(hv.iter()).map(|(a, b)| a * b).sum();
        let den: f64 = v.iter().map(|x| x * x).sum();
        num / den
    }
}

/// The ONE-site effective Hamiltonian `L · W_i · R`, as an operator on `A[a,s,b]`.
///
/// A single-site update changes no bond dimension, so it discards nothing and is exactly
/// variational. At bond dimension 1 it is precisely mean-field coordinate descent: the effective
/// matrix is `[[−A, −g], [−g, A]]` with `A = J(⟨Z⟩_{i−1} + ⟨Z⟩_{i+1}) + h`, whose ground state is
/// the self-consistent single-site state. That is why [`Options::polish`] exists — see its doc for
/// the measurement that motivates it.
struct Site<'a> {
    l: &'a [f64],
    r: &'a [f64],
    w: &'a [f64],
    dl: usize,
    dr: usize,
}

impl LinearOp for Site<'_> {
    fn dim(&self) -> usize {
        self.dl * PHYS * self.dr
    }

    fn apply(&self, v: &[f64], out: &mut [f64]) {
        let (dl, dr) = (self.dl, self.dr);
        let block = PHYS * dr;
        // t1[a, w, s', b'] = Σ_a' L[a,w,a'] v[a',s',b']
        let mut t1 = vec![0.0; dl * MPO_BOND * block];
        for aa in 0..dl {
            for wv in 0..MPO_BOND {
                let dst = (aa * MPO_BOND + wv) * block;
                for ap in 0..dl {
                    let lv = self.l[(aa * MPO_BOND + wv) * dl + ap];
                    if lv == 0.0 {
                        continue;
                    }
                    for idx in 0..block {
                        t1[dst + idx] += lv * v[ap * block + idx];
                    }
                }
            }
        }
        // t2[a, s, w', b'] = Σ_{w,s'} W[w,s,s',w'] t1[a,w,s',b']
        let mut t2 = vec![0.0; dl * PHYS * MPO_BOND * dr];
        for aa in 0..dl {
            for wv in 0..MPO_BOND {
                for s in 0..PHYS {
                    for sp in 0..PHYS {
                        for wn in 0..MPO_BOND {
                            let coeff = self.w[((wv * PHYS + s) * PHYS + sp) * MPO_BOND + wn];
                            if coeff == 0.0 {
                                continue;
                            }
                            let src = (aa * MPO_BOND + wv) * block + sp * dr;
                            let dst = ((aa * PHYS + s) * MPO_BOND + wn) * dr;
                            for bp in 0..dr {
                                t2[dst + bp] += coeff * t1[src + bp];
                            }
                        }
                    }
                }
            }
        }
        // out[a, s, b] = Σ_{w',b'} R[b,w',b'] t2[a,s,w',b']
        for aa in 0..dl {
            for s in 0..PHYS {
                let src = (aa * PHYS + s) * MPO_BOND * dr;
                let dst = (aa * PHYS + s) * dr;
                for b in 0..dr {
                    let mut acc = 0.0;
                    for wn in 0..MPO_BOND {
                        for bp in 0..dr {
                            acc += self.r[(b * MPO_BOND + wn) * dr + bp] * t2[src + wn * dr + bp];
                        }
                    }
                    out[dst + b] = acc;
                }
            }
        }
    }
}

/// The two-site effective Hamiltonian `L · W_i · W_{i+1} · R`, as an operator on `θ[a,s1,s2,b]`.
///
/// Never formed as a matrix. At bond dimension 32 the matrix is `4096 × 4096`; the contraction below
/// costs `O(D³)` per application, which is why DMRG is a Lanczos method and not an eigensolve.
struct Effective<'a> {
    l: &'a [f64],
    r: &'a [f64],
    w_left: &'a [f64],
    w_right: &'a [f64],
    dl: usize,
    dr: usize,
}

impl LinearOp for Effective<'_> {
    fn dim(&self) -> usize {
        self.dl * PHYS * PHYS * self.dr
    }

    fn apply(&self, theta: &[f64], out: &mut [f64]) {
        let (dl, dr) = (self.dl, self.dr);
        let block = PHYS * PHYS * dr;
        // t1[a, w, s1', s2', b'] = Σ_a' L[a,w,a'] θ[a',s1',s2',b']
        let mut t1 = vec![0.0; dl * MPO_BOND * block];
        for aa in 0..dl {
            for w in 0..MPO_BOND {
                let base = (aa * MPO_BOND + w) * block;
                for ap in 0..dl {
                    let lv = self.l[(aa * MPO_BOND + w) * dl + ap];
                    if lv == 0.0 {
                        continue;
                    }
                    for idx in 0..block {
                        t1[base + idx] += lv * theta[ap * block + idx];
                    }
                }
            }
        }
        // t2[a, s1, wm, s2', b'] = Σ_{w,s1'} W_i[w,s1,s1',wm] t1[a,w,s1',s2',b']
        let tail = PHYS * dr;
        let mut t2 = vec![0.0; dl * PHYS * MPO_BOND * tail];
        for aa in 0..dl {
            for w in 0..MPO_BOND {
                for s1 in 0..PHYS {
                    for s1p in 0..PHYS {
                        for wm in 0..MPO_BOND {
                            let coeff = self.w_left[((w * PHYS + s1) * PHYS + s1p) * MPO_BOND + wm];
                            if coeff == 0.0 {
                                continue;
                            }
                            let src = (aa * MPO_BOND + w) * block + s1p * tail;
                            let dst = ((aa * PHYS + s1) * MPO_BOND + wm) * tail;
                            for idx in 0..tail {
                                t2[dst + idx] += coeff * t1[src + idx];
                            }
                        }
                    }
                }
            }
        }
        // t3[a, s1, s2, wr, b'] = Σ_{wm,s2'} W_{i+1}[wm,s2,s2',wr] t2[a,s1,wm,s2',b']
        let mut t3 = vec![0.0; dl * PHYS * PHYS * MPO_BOND * dr];
        for aa in 0..dl {
            for s1 in 0..PHYS {
                for wm in 0..MPO_BOND {
                    for s2 in 0..PHYS {
                        for s2p in 0..PHYS {
                            for wb in 0..MPO_BOND {
                                let coeff =
                                    self.w_right[((wm * PHYS + s2) * PHYS + s2p) * MPO_BOND + wb];
                                if coeff == 0.0 {
                                    continue;
                                }
                                let src = ((aa * PHYS + s1) * MPO_BOND + wm) * tail + s2p * dr;
                                let dst = (((aa * PHYS + s1) * PHYS + s2) * MPO_BOND + wb) * dr;
                                for bp in 0..dr {
                                    t3[dst + bp] += coeff * t2[src + bp];
                                }
                            }
                        }
                    }
                }
            }
        }
        // out[a, s1, s2, b] = Σ_{wr,b'} R[b,wr,b'] t3[a,s1,s2,wr,b']
        for aa in 0..dl {
            for s1 in 0..PHYS {
                for s2 in 0..PHYS {
                    let src = ((aa * PHYS + s1) * PHYS + s2) * MPO_BOND * dr;
                    let dst = ((aa * PHYS + s1) * PHYS + s2) * dr;
                    for b in 0..dr {
                        let mut acc = 0.0;
                        for wb in 0..MPO_BOND {
                            for bp in 0..dr {
                                acc += self.r[(b * MPO_BOND + wb) * dr + bp] * t3[src + wb * dr + bp];
                            }
                        }
                        out[dst + b] = acc;
                    }
                }
            }
        }
    }
}

/// Lanczos with full reorthogonalisation, returning the lowest Ritz pair.
///
/// The start vector is the CURRENT two-site tensor, which is what makes each local update
/// non-increasing in energy: the Krylov space contains the starting vector, so the Ritz minimum is
/// at most the energy already there. Start it from a random vector and that guarantee is gone.
///
/// The `k × k` tridiagonal projection is diagonalised by [`crate::linalg::jacobi_eig`] — a third
/// eigensolver is not needed and a `k` of a dozen makes Jacobi free.
fn lanczos<O: LinearOp>(eff: &O, start: &[f64], k_max: usize) -> (f64, Vec<f64>) {
    let dim = eff.dim();
    let k_max = k_max.min(dim).max(1);
    let mut basis: Vec<Vec<f64>> = Vec::with_capacity(k_max);
    let mut v: Vec<f64> = start.to_vec();
    let nrm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if nrm == 0.0 {
        return (0.0, start.to_vec());
    }
    for x in &mut v {
        *x /= nrm;
    }
    basis.push(v);
    let mut alpha: Vec<f64> = Vec::with_capacity(k_max);
    let mut beta: Vec<f64> = Vec::with_capacity(k_max);
    let mut w = vec![0.0; dim];
    for j in 0..k_max {
        eff.apply(&basis[j], &mut w);
        let a: f64 = basis[j].iter().zip(w.iter()).map(|(x, y)| x * y).sum();
        alpha.push(a);
        if j + 1 == k_max {
            break;
        }
        for _pass in 0..2 {
            for b in &basis {
                let dot: f64 = b.iter().zip(w.iter()).map(|(x, y)| x * y).sum();
                for (wx, bx) in w.iter_mut().zip(b.iter()) {
                    *wx -= dot * bx;
                }
            }
        }
        let nb = w.iter().map(|x| x * x).sum::<f64>().sqrt();
        if nb < 1e-13 {
            break;
        }
        beta.push(nb);
        basis.push(w.iter().map(|x| x / nb).collect());
    }
    let m = alpha.len();
    let mut tri = vec![0.0; m * m];
    for i in 0..m {
        tri[i * m + i] = alpha[i];
        if i + 1 < m {
            tri[i * m + i + 1] = beta[i];
            tri[(i + 1) * m + i] = beta[i];
        }
    }
    let vecs = jacobi_eig(&mut tri, m);
    let mut best = 0;
    for c in 1..m {
        if tri[c * m + c] < tri[best * m + best] {
            best = c;
        }
    }
    let mut out = vec![0.0; dim];
    for (j, b) in basis.iter().enumerate() {
        let y = vecs[j * m + best];
        for (o, x) in out.iter_mut().zip(b.iter()) {
            *o += y * x;
        }
    }
    let on = out.iter().map(|x| x * x).sum::<f64>().sqrt();
    if on > 0.0 {
        for x in &mut out {
            *x /= on;
        }
    }
    (tri[best * m + best], out)
}

/// How a sweep schedule is run.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Options {
    max_bond: usize,
    cutoff: f64,
    sweeps: usize,
    krylov: usize,
    polish: usize,
}

impl Default for Options {
    /// Bond dimension 32, cutoff `1e-12`, 30 two-site sweeps, 20 Krylov vectors, 2 polish sweeps.
    ///
    /// Those last two are measured rather than chosen. On a 16-site chain at `g = 0.5` the error
    /// against the Jordan-Wigner answer is `3.8e-6` at 10 sweeps with 12 Krylov vectors, `2.7e-6`
    /// at 20, and `5.3e-13` at 30 with 20 — **the Krylov depth matters as much as the sweep
    /// count**, and neither is visible from the energy trace, which at 10 sweeps is still drifting
    /// by `3.5e-6` and at 20 by `2.3e-6`. A starting point, not a claim about any Hamiltonian.
    fn default() -> Options {
        Options { max_bond: 32, cutoff: 1e-12, sweeps: 30, krylov: 20, polish: 2 }
    }
}

impl Options {
    /// A schedule with the given bond cap, relative singular-value cutoff, sweep count and Krylov
    /// dimension.
    ///
    /// `cutoff` is a fraction of the LARGEST singular value at a bond, so it is scale-free.
    ///
    /// # Errors
    ///
    /// [`MpsError::ZeroBond`], [`MpsError::BadCutoff`], [`MpsError::NoSweeps`] and
    /// [`MpsError::BadKrylov`], each naming the value it received.
    pub fn new(
        max_bond: usize,
        cutoff: f64,
        sweeps: usize,
        krylov: usize,
        polish: usize,
    ) -> Result<Options, MpsError> {
        if max_bond == 0 {
            return Err(MpsError::ZeroBond);
        }
        if !cutoff.is_finite() {
            return Err(MpsError::NotFinite { parameter: Parameter::Cutoff, value: cutoff });
        }
        if !(0.0..1.0).contains(&cutoff) {
            return Err(MpsError::BadCutoff { cutoff });
        }
        if sweeps == 0 {
            return Err(MpsError::NoSweeps);
        }
        if krylov < 2 {
            return Err(MpsError::BadKrylov { krylov });
        }
        Ok(Options { max_bond, cutoff, sweeps, krylov, polish })
    }

    /// The bond-dimension cap.
    #[must_use]
    pub fn max_bond(&self) -> usize {
        self.max_bond
    }

    /// The relative singular-value cutoff.
    #[must_use]
    pub fn cutoff(&self) -> f64 {
        self.cutoff
    }

    /// Sweeps to run. One sweep is a pass right and a pass back left.
    #[must_use]
    pub fn sweeps(&self) -> usize {
        self.sweeps
    }

    /// Krylov vectors per local eigenproblem.
    #[must_use]
    pub fn krylov(&self) -> usize {
        self.krylov
    }

    /// SINGLE-site sweeps run after the two-site ones. Zero is legitimate and means pure two-site
    /// DMRG.
    ///
    /// **Two-site DMRG does not converge to the best state its own bond dimension can write down.**
    /// Each update solves the two-site problem exactly and then keeps the best rank-`D`
    /// APPROXIMATION of that solution, which is not the lowest-energy rank-`D` state. At bond
    /// dimension 1, where the gap is easiest to see because the answer is known independently,
    /// measured on a 10-site chain with `J = 1`: two-site DMRG stops `4.2e-4` above the
    /// product-state optimum at `g = 0.4`, `2.7e-2` at `g = 1.0` and **`1.2e-1`** at `g = 1.6`, and
    /// 300 sweeps do not move it — it is a fixed point, not slow convergence.
    ///
    /// A single-site update changes no bond dimension, so it truncates nothing and minimises
    /// exactly within the manifold the two-site sweeps found. **It is a convergence parameter like
    /// any other**: on that `g = 1.6` chain two polish sweeps leave `1.7e-2` and twenty close it to
    /// `2.8e-14`. They cannot GROW a bond, which is why they run after the two-site sweeps and not
    /// instead of them.
    #[must_use]
    pub fn polish(&self) -> usize {
        self.polish
    }
}

/// What one sweep did.
#[derive(Clone, Debug)]
pub struct Sweep {
    /// `⟨ψ|H|ψ⟩` after every local update in the sweep, in the order they happened.
    ///
    /// The FULL state's energy each time, not a local one: with exact environments the two-site
    /// Rayleigh quotient is the whole chain's expectation value. This is the sequence a
    /// monotonicity claim is about, and it is recorded because the claim is only true when
    /// [`Sweep::truncation`] is zero.
    pub local: Vec<f64>,
    /// The energy at the end of the sweep.
    pub energy: f64,
    /// The largest weight `Σ_{discarded} s² / Σ s²` thrown away at any split in the sweep.
    pub truncation: f64,
    /// The largest bond dimension in the state at the end of the sweep.
    pub max_bond: usize,
    /// [`Mps::isometry_defect`] at the end of the sweep.
    pub isometry_defect: f64,
    /// Whether this was a single-site polish sweep rather than a two-site one.
    ///
    /// Worth reporting because it decides what the energy trace promises: a single-site sweep
    /// discards nothing and is therefore monotone, and a two-site sweep is monotone only while
    /// [`Sweep::truncation`] is zero.
    pub single_site: bool,
}

/// The result of a DMRG run.
#[derive(Clone, Debug)]
pub struct Ground {
    /// The variational energy `⟨ψ|H|ψ⟩` of the final state.
    pub energy: f64,
    /// The same quantity as a **certified upper bound** on the exact ground energy: every sum
    /// accumulated through [`crate::round::sum_up`], with a [`crate::round::accumulation_guard`]
    /// for the contractions whose terms are not available as a slice, and the isometry defect
    /// charged as a first-order perturbation. Always `≥ energy`, and by the variational principle
    /// always `≥ E_0`.
    pub certified: f64,
    /// One entry per sweep, in order.
    pub sweeps: Vec<Sweep>,
    /// The optimised state, in mixed canonical form with the centre on site 0.
    pub mps: Mps,
}

/// Left environment of sites `0..i`, shape `(D_i, 3, D_i)`.
fn left_env(prev: &[f64], a: &[f64], w: &[f64], dl: usize, dn: usize) -> Vec<f64> {
    // p[w, a', s, b] = Σ_a L[a,w,a'] A[a,s,b]
    let mut p = vec![0.0; MPO_BOND * dl * PHYS * dn];
    for aa in 0..dl {
        for wv in 0..MPO_BOND {
            for ap in 0..dl {
                let lv = prev[(aa * MPO_BOND + wv) * dl + ap];
                if lv == 0.0 {
                    continue;
                }
                let dst = (wv * dl + ap) * PHYS * dn;
                let src = aa * PHYS * dn;
                for idx in 0..PHYS * dn {
                    p[dst + idx] += lv * a[src + idx];
                }
            }
        }
    }
    // q[w', a', s', b] = Σ_{w,s} W[w,s,s',w'] p[w,a',s,b]
    let mut q = vec![0.0; MPO_BOND * dl * PHYS * dn];
    for wv in 0..MPO_BOND {
        for s in 0..PHYS {
            for sp in 0..PHYS {
                for wn in 0..MPO_BOND {
                    let coeff = w[((wv * PHYS + s) * PHYS + sp) * MPO_BOND + wn];
                    if coeff == 0.0 {
                        continue;
                    }
                    for ap in 0..dl {
                        let src = (wv * dl + ap) * PHYS * dn + s * dn;
                        let dst = (wn * dl + ap) * PHYS * dn + sp * dn;
                        for b in 0..dn {
                            q[dst + b] += coeff * p[src + b];
                        }
                    }
                }
            }
        }
    }
    // L'[b, w', b'] = Σ_{a',s'} q[w',a',s',b] A[a',s',b']
    let mut out = vec![0.0; dn * MPO_BOND * dn];
    for wn in 0..MPO_BOND {
        for ap in 0..dl {
            for sp in 0..PHYS {
                let src = (wn * dl + ap) * PHYS * dn + sp * dn;
                let arow = (ap * PHYS + sp) * dn;
                for b in 0..dn {
                    let v = q[src + b];
                    if v == 0.0 {
                        continue;
                    }
                    for bp in 0..dn {
                        out[(b * MPO_BOND + wn) * dn + bp] += v * a[arow + bp];
                    }
                }
            }
        }
    }
    out
}

/// Right environment of sites `i..n`, shape `(D_i, 3, D_i)`.
fn right_env(next: &[f64], a: &[f64], w: &[f64], dl: usize, dn: usize) -> Vec<f64> {
    // p[a, s, w', b'] = Σ_b A[a,s,b] R[b,w',b']
    let mut p = vec![0.0; dl * PHYS * MPO_BOND * dn];
    for aa in 0..dl {
        for s in 0..PHYS {
            for b in 0..dn {
                let v = a[(aa * PHYS + s) * dn + b];
                if v == 0.0 {
                    continue;
                }
                let dst = (aa * PHYS + s) * MPO_BOND * dn;
                let src = b * MPO_BOND * dn;
                for idx in 0..MPO_BOND * dn {
                    p[dst + idx] += v * next[src + idx];
                }
            }
        }
    }
    // q[a, w, s', b'] = Σ_{s,w'} W[w,s,s',w'] p[a,s,w',b']
    let mut q = vec![0.0; dl * MPO_BOND * PHYS * dn];
    for wv in 0..MPO_BOND {
        for s in 0..PHYS {
            for sp in 0..PHYS {
                for wn in 0..MPO_BOND {
                    let coeff = w[((wv * PHYS + s) * PHYS + sp) * MPO_BOND + wn];
                    if coeff == 0.0 {
                        continue;
                    }
                    for aa in 0..dl {
                        let src = (aa * PHYS + s) * MPO_BOND * dn + wn * dn;
                        let dst = (aa * MPO_BOND + wv) * PHYS * dn + sp * dn;
                        for bp in 0..dn {
                            q[dst + bp] += coeff * p[src + bp];
                        }
                    }
                }
            }
        }
    }
    // R'[a, w, a'] = Σ_{s',b'} q[a,w,s',b'] A[a',s',b']
    let mut out = vec![0.0; dl * MPO_BOND * dl];
    for aa in 0..dl {
        for wv in 0..MPO_BOND {
            for sp in 0..PHYS {
                let src = (aa * MPO_BOND + wv) * PHYS * dn + sp * dn;
                for ap in 0..dl {
                    let arow = (ap * PHYS + sp) * dn;
                    let mut acc = 0.0;
                    for bp in 0..dn {
                        acc += q[src + bp] * a[arow + bp];
                    }
                    out[(aa * MPO_BOND + wv) * dl + ap] += acc;
                }
            }
        }
    }
    out
}

/// Two-site DMRG for the ground state of `h`.
///
/// One sweep is a pass from site 0 to the right end and back. Each local update forms the two-site
/// tensor, finds the lowest eigenvector of the effective Hamiltonian by Lanczos, splits it by
/// singular value decomposition and truncates. The state is left with its orthogonality centre on
/// site 0.
///
/// # Errors
///
/// Nothing here fails at run time; the result is a `Result` because a future variant of this
/// routine that refuses a Hamiltonian it cannot write as an MPO should not change its signature,
/// and because [`Options`] and [`Tfim`] are the things that can be wrong and are already checked.
#[allow(clippy::unnecessary_wraps)]
pub fn ground_state(h: &Tfim, opts: &Options, seed: u64) -> Result<Ground, MpsError> {
    let n = h.n;
    let w = h.mpo();
    let mut mps = Mps::random(n, opts.max_bond, seed);

    let mut l_env: Vec<Vec<f64>> = vec![Vec::new(); n + 1];
    let mut r_env: Vec<Vec<f64>> = vec![Vec::new(); n + 1];
    let mut boundary_l = vec![0.0; MPO_BOND];
    boundary_l[MPO_BOND - 1] = 1.0;
    let mut boundary_r = vec![0.0; MPO_BOND];
    boundary_r[0] = 1.0;
    l_env[0] = boundary_l;
    r_env[n] = boundary_r;
    for i in (1..n).rev() {
        r_env[i] = right_env(&r_env[i + 1], &mps.a[i], &w, mps.dims[i], mps.dims[i + 1]);
    }

    let mut sweeps = Vec::with_capacity(opts.sweeps + opts.polish);
    for _s in 0..opts.sweeps {
        let mut local = Vec::with_capacity(2 * (n - 1));
        let mut worst_trunc = 0.0f64;
        for i in 0..n - 1 {
            let e = update_pair(&mut mps, &l_env[i], &r_env[i + 2], &w, opts, i, true);
            worst_trunc = worst_trunc.max(e.1);
            local.push(e.0);
            l_env[i + 1] = left_env(&l_env[i], &mps.a[i], &w, mps.dims[i], mps.dims[i + 1]);
        }
        for i in (0..n - 1).rev() {
            let e = update_pair(&mut mps, &l_env[i], &r_env[i + 2], &w, opts, i, false);
            worst_trunc = worst_trunc.max(e.1);
            local.push(e.0);
            r_env[i + 1] = right_env(&r_env[i + 2], &mps.a[i + 1], &w, mps.dims[i + 1], mps.dims[i + 2]);
        }
        let energy = *local.last().unwrap_or(&f64::NAN);
        sweeps.push(Sweep {
            local,
            energy,
            truncation: worst_trunc,
            max_bond: mps.max_bond(),
            isometry_defect: mps.isometry_defect(),
            single_site: false,
        });
    }

    // Polish: single-site sweeps, which discard nothing. See `Options::polish` for what they are
    // for and the measurement that says the two-site sweeps above do not get there on their own.
    for _s in 0..opts.polish {
        let mut local = Vec::with_capacity(2 * n);
        for i in 0..n {
            local.push(optimise_site(&mut mps, &l_env[i], &r_env[i + 1], &w, opts, i));
            if i + 1 < n {
                mps.shift_right(i, usize::MAX, 0.0);
                l_env[i + 1] = left_env(&l_env[i], &mps.a[i], &w, mps.dims[i], mps.dims[i + 1]);
            }
        }
        for i in (0..n - 1).rev() {
            mps.shift_left(i + 1, usize::MAX, 0.0);
            r_env[i + 1] =
                right_env(&r_env[i + 2], &mps.a[i + 1], &w, mps.dims[i + 1], mps.dims[i + 2]);
            local.push(optimise_site(&mut mps, &l_env[i], &r_env[i + 1], &w, opts, i));
        }
        let energy = *local.last().unwrap_or(&f64::NAN);
        sweeps.push(Sweep {
            local,
            energy,
            truncation: 0.0,
            max_bond: mps.max_bond(),
            isometry_defect: mps.isometry_defect(),
            single_site: true,
        });
    }
    mps.move_centre(0);

    let energy = sweeps.last().map_or(f64::NAN, |s| s.energy);
    let certified = certified_energy(&mps, h);
    Ok(Ground { energy, certified, sweeps, mps })
}

/// One two-site update. Returns `(energy after truncation, discarded weight)`.
///
/// `rightward` decides which of the two tensors keeps the orthogonality centre, and therefore which
/// direction the sweep is moving.
fn update_pair(
    mps: &mut Mps,
    l: &[f64],
    r: &[f64],
    w: &[f64],
    opts: &Options,
    i: usize,
    rightward: bool,
) -> (f64, f64) {
    let (dl, dm, dr) = (mps.dims[i], mps.dims[i + 1], mps.dims[i + 2]);
    let theta = matmul(&mps.a[i], dl * PHYS, dm, &mps.a[i + 1], dm, PHYS * dr);
    let eff = Effective { l, r, w_left: w, w_right: w, dl, dr };
    let (_ritz, opt) = lanczos(&eff, &theta, opts.krylov);

    let (rows, cols) = (dl * PHYS, PHYS * dr);
    let (u, s, vt, rank) = svd(&opt, rows, cols);
    let (keep, discarded) = truncate(&s, rank, opts.max_bond, opts.cutoff);
    let scale = renorm(&s, keep);

    let mut left = vec![0.0; rows * keep];
    let mut right = vec![0.0; keep * cols];
    for t in 0..keep {
        for rr in 0..rows {
            left[rr * keep + t] = u[rr * s.len() + t];
        }
        for c in 0..cols {
            right[t * cols + c] = vt[t * cols + c];
        }
    }
    let sv: Vec<f64> = (0..keep).map(|t| s[t] * scale).collect();
    if rightward {
        for t in 0..keep {
            for c in 0..cols {
                right[t * cols + c] *= sv[t];
            }
        }
        mps.centre = i + 1;
    } else {
        for rr in 0..rows {
            for t in 0..keep {
                left[rr * keep + t] *= sv[t];
            }
        }
        mps.centre = i;
    }
    mps.a[i] = left;
    mps.a[i + 1] = right;
    mps.dims[i + 1] = keep;
    mps.schmidt[i + 1] = sv;

    // The energy of the state that actually remains, with the SAME environments -- which is exact,
    // because the environments describe sites this update did not touch.
    let kept = matmul(&mps.a[i], dl * PHYS, keep, &mps.a[i + 1], keep, PHYS * dr);
    (eff.rayleigh(&kept), discarded)
}

/// One single-site update, with the orthogonality centre already on site `i`. Returns the energy.
///
/// Nothing is discarded and no bond dimension moves, so the returned sequence is monotone by
/// construction: Lanczos starts from the tensor already there and minimises over a space containing
/// it.
fn optimise_site(
    mps: &mut Mps,
    l: &[f64],
    r: &[f64],
    w: &[f64],
    opts: &Options,
    i: usize,
) -> f64 {
    let (dl, dr) = (mps.dims[i], mps.dims[i + 1]);
    let op = Site { l, r, w, dl, dr };
    let (_ritz, opt) = lanczos(&op, &mps.a[i], opts.krylov);
    let energy = op.rayleigh(&opt);
    mps.a[i] = opt;
    energy
}

/// `⟨ψ|H|ψ⟩` as an upper bound that survives floating point.
///
/// The energy is a sum of `2N−1` local expectation values, each read off the canonical form with the
/// orthogonality centre on the site it belongs to. Those `2N−1` numbers go through
/// [`crate::round::sum_up`]; the contractions INSIDE each one do not have their terms available as a
/// slice, so they are charged through [`crate::round::accumulation_guard`] with the operation count
/// and the magnitude of the partial sums both taken from the loops that produced them. The isometry
/// defect is charged on top as a first-order perturbation, `2 δ Σ|coefficient|`, because a state
/// whose environments are not exactly isometric is not exactly the state the local expectation
/// values describe.
///
/// The width this adds is measured, not assumed: `the_certificate_is_sound_and_not_vacuous` asserts
/// it is both above the plain contraction and within `1e-9` of it.
fn certified_energy(mps: &Mps, h: &Tfim) -> f64 {
    let mut work = mps.clone();
    work.move_centre(0);
    work.normalise_centre();
    let n = work.n;
    let mut terms: Vec<f64> = Vec::with_capacity(2 * n);
    let mut guard = 0.0f64;
    let mut norm_lo = f64::INFINITY;
    let mut norm_hi = f64::NEG_INFINITY;

    for i in 0..n {
        debug_assert_eq!(work.centre, i, "every local expectation is read at the centre");
        let (dl, dr) = (work.dims[i], work.dims[i + 1]);
        let t = &work.a[i];
        let sq: Vec<f64> = t.iter().map(|x| x * x).collect();
        norm_lo = norm_lo.min(sum_down(&sq) - accumulation_guard(1, sq.iter().sum::<f64>()));
        norm_hi = norm_hi.max(sum_up(&sq) + accumulation_guard(1, sq.iter().sum::<f64>()));

        // ⟨Z_i⟩ and ⟨X_i⟩, both slice sums over the centre tensor.
        let mut zt = Vec::with_capacity(dl * PHYS * dr);
        let mut xt = Vec::with_capacity(dl * dr);
        for aa in 0..dl {
            for b in 0..dr {
                for s in 0..PHYS {
                    zt.push(-h.h * z_of(s) * t[(aa * PHYS + s) * dr + b]
                        * t[(aa * PHYS + s) * dr + b]);
                }
                xt.push(-2.0 * h.g * t[aa * PHYS * dr + b] * t[(aa * PHYS + 1) * dr + b]);
            }
        }
        guard += accumulation_guard(3, zt.iter().map(|x| x.abs()).sum::<f64>());
        guard += accumulation_guard(3, xt.iter().map(|x| x.abs()).sum::<f64>());
        terms.push(sum_up(&zt));
        terms.push(sum_up(&xt));

        if i + 1 < n {
            let dr2 = work.dims[i + 2];
            let b1 = &work.a[i + 1];
            // Rz[b,b'] = Σ_{s,c} z(s) B[b,s,c] B[b',s,c], with its absolute-value twin so the
            // rounding of THIS contraction can be charged rather than assumed away.
            let mut rz = vec![0.0; dr * dr];
            let mut rz_abs = vec![0.0; dr * dr];
            for b in 0..dr {
                for bp in 0..dr {
                    let mut acc = 0.0;
                    let mut mag = 0.0;
                    for s in 0..PHYS {
                        for c in 0..dr2 {
                            let p = b1[(b * PHYS + s) * dr2 + c] * b1[(bp * PHYS + s) * dr2 + c];
                            acc += z_of(s) * p;
                            mag += p.abs();
                        }
                    }
                    rz[b * dr + bp] = acc;
                    rz_abs[b * dr + bp] = mag;
                }
            }
            let mut bond = 0.0;
            let mut bond_mag = 0.0;
            for aa in 0..dl {
                for s in 0..PHYS {
                    for b in 0..dr {
                        let la = t[(aa * PHYS + s) * dr + b];
                        for bp in 0..dr {
                            let lb = t[(aa * PHYS + s) * dr + bp];
                            bond += -h.j * z_of(s) * la * rz[b * dr + bp] * lb;
                            bond_mag += h.j.abs() * la.abs() * rz_abs[b * dr + bp] * lb.abs();
                        }
                    }
                }
            }
            let ops = PHYS * dr2 + dl * PHYS * dr * dr + 4;
            guard += accumulation_guard(ops, bond_mag);
            terms.push(bond);
            // Every expectation value above is only the expectation value while the orthogonality
            // centre is on the site it belongs to. Walking it right is not bookkeeping -- read at
            // the wrong centre, the same loops return the trace of a right isometry instead.
            work.shift_right(i, usize::MAX, 0.0);
        }
    }

    let scale = (h.j.abs() + h.g.abs() + h.h.abs()) * (2 * n) as f64;
    guard += 2.0 * work.isometry_defect() * scale;
    let num_hi = sum_up(&terms) + guard;
    let num_lo = sum_down(&terms) - guard;
    if !(norm_lo > 0.0) {
        return f64::INFINITY;
    }
    let mut best = f64::NEG_INFINITY;
    for &num in &[num_lo, num_hi] {
        for &den in &[norm_lo, norm_hi] {
            best = best.max(num / den);
        }
    }
    best.next_up().next_up()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A dense `2^n` application of `H`, written directly from the Pauli algebra.
    ///
    /// Deliberately shares nothing with the module: no MPO, no environments, no canonical form.
    /// Index convention matches [`Mps::to_dense`] — site 0 is the most significant bit, `0` = up.
    fn apply_h(n: usize, j: f64, g: f64, hz: f64, periodic: bool, v: &[f64], out: &mut [f64]) {
        let dim = 1usize << n;
        for o in out.iter_mut() {
            *o = 0.0;
        }
        for idx in 0..dim {
            let bit = |i: usize| -> f64 {
                if (idx >> (n - 1 - i)) & 1 == 0 { 1.0 } else { -1.0 }
            };
            let mut diag = 0.0;
            let last = if periodic { n } else { n - 1 };
            for i in 0..last {
                diag -= j * bit(i) * bit((i + 1) % n);
            }
            for i in 0..n {
                diag -= hz * bit(i);
            }
            out[idx] += diag * v[idx];
            for i in 0..n {
                out[idx] -= g * v[idx ^ (1 << (n - 1 - i))];
            }
        }
    }

    /// Ground energy and ground vector by shifted power iteration on the dense Hamiltonian.
    ///
    /// The point of choosing power iteration is that it is not Lanczos: it shares no algorithm with
    /// [`lanczos`], so a bug in the Krylov code cannot hide in both. `shift − H` has its largest
    /// magnitude eigenvalue at the ground state as long as `shift` is above the spectral radius,
    /// and Gershgorin on the Hamiltonian gives one.
    fn dense_ground(n: usize, j: f64, g: f64, hz: f64, periodic: bool) -> (f64, Vec<f64>) {
        let dim = 1usize << n;
        let bonds = if periodic { n } else { n - 1 };
        let shift = j.abs() * bonds as f64 + (g.abs() + hz.abs()) * n as f64 + 1.0;
        let mut rng = Pcg::new(0xE7, 0x9);
        let mut v: Vec<f64> = (0..dim).map(|_| rng.f64() - 0.5).collect();
        let mut hv = vec![0.0; dim];
        let mut last = f64::NAN;
        // At h = 0 the spin flip P = prod X_i commutes with H, and in the ordered phase the two
        // lowest states -- one per parity -- are split by O(g^n). Measured at n = 8, J = 1,
        // g = 0.3 that is 3e-7, and power iteration on the full space stalls in a mixture and
        // reports an energy between the two. The ground state is P-even (Perron-Frobenius: the
        // off-diagonals are -g < 0, so the ground vector is positive), so projecting onto the even
        // sector every step removes the near-degeneracy exactly rather than out-iterating it.
        let project = hz == 0.0;
        let mask = dim - 1;
        for step in 0..200_000 {
            if project {
                for idx in 0..dim / 2 {
                    let m = (idx ^ mask) & mask;
                    let avg = 0.5 * (v[idx] + v[m]);
                    v[idx] = avg;
                    v[m] = avg;
                }
            }
            let nrm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            for x in &mut v {
                *x /= nrm;
            }
            apply_h(n, j, g, hz, periodic, &v, &mut hv);
            let ray: f64 = v.iter().zip(hv.iter()).map(|(a, b)| a * b).sum();
            let resid: f64 = v
                .iter()
                .zip(hv.iter())
                .map(|(a, b)| (b - ray * a) * (b - ray * a))
                .sum::<f64>()
                .sqrt();
            if step > 20 && resid < 1e-11 {
                return (ray, v);
            }
            last = ray;
            for (x, y) in v.iter_mut().zip(hv.iter()) {
                *x = shift * *x - y;
            }
        }
        (last, v)
    }

    /// The best PRODUCT state, by exact coordinate descent on the mean-field energy.
    ///
    /// Every site carries `⟨Z⟩ = cos θ_i`, `⟨X⟩ = sin θ_i`, and holding the neighbours fixed the
    /// energy in `θ_i` is `−A cos θ_i − g sin θ_i` with `A = J(cos θ_{i−1} + cos θ_{i+1}) + h`. That
    /// is minimised in CLOSED FORM at `−sqrt(A² + g²)`, so this is exact coordinate descent and not
    /// a numerical minimiser — independent of everything in this module.
    fn product_mean_field(n: usize, j: f64, g: f64, hz: f64) -> f64 {
        let mut c = vec![0.8f64; n];
        let mut s = vec![0.6f64; n];
        for _ in 0..20_000 {
            for i in 0..n {
                let mut field = hz;
                if i > 0 {
                    field += j * c[i - 1];
                }
                if i + 1 < n {
                    field += j * c[i + 1];
                }
                let nrm = (field * field + g * g).sqrt();
                if nrm == 0.0 {
                    c[i] = 1.0;
                    s[i] = 0.0;
                } else {
                    c[i] = field / nrm;
                    s[i] = g / nrm;
                }
            }
        }
        let mut e = 0.0;
        for i in 0..n - 1 {
            e -= j * c[i] * c[i + 1];
        }
        for i in 0..n {
            e -= g * s[i] + hz * c[i];
        }
        e
    }

    /// ORACLE: the Jordan–Wigner closed form against brute-force diagonalisation of `2^n`.
    ///
    /// The closed form is the one every other test in this module leans on, so it is checked first,
    /// against a routine that shares no line of code with it — a dense Hamiltonian assembled from
    /// Pauli matrices and a power iteration. If this fails, nothing below means anything.
    #[test]
    fn free_fermion_closed_form_matches_brute_force_diagonalisation() {
        for n in [2usize, 3, 5, 8] {
            for &g in &[0.0, 0.3, 1.0, 2.5] {
                for &j in &[1.0, 0.4, -1.0] {
                    let h = Tfim::new(n, j, g, 0.0).unwrap();
                    let closed = h.free_fermion_ground_energy().unwrap();
                    let (dense, _) = dense_ground(n, j, g, 0.0, false);
                    assert!(
                        (closed - dense).abs() < 1e-9,
                        "n={n} J={j} g={g}: closed form {closed} vs diagonalisation {dense}"
                    );
                }
            }
        }
    }

    /// ORACLE: the matrix product operator reproduces the dense Hamiltonian, entry for entry.
    ///
    /// Contracting the MPO between its boundary vectors must give the SAME matrix as the Pauli sum.
    /// A sign error in one of the five non-zero entries of `W` would change the ground energy by an
    /// amount a tolerance on the final answer could easily absorb; this cannot absorb it.
    #[test]
    fn the_mpo_reproduces_the_dense_hamiltonian_entry_for_entry() {
        let (n, j, g, hz) = (5usize, 0.7, 1.3, -0.4);
        let h = Tfim::new(n, j, g, hz).unwrap();
        let w = h.mpo();
        let dim = 1usize << n;
        for col in 0..dim {
            // The MPO's column: chain the 3-vectors through the operator.
            let mut row = vec![0.0; MPO_BOND];
            row[MPO_BOND - 1] = 1.0;
            let mut amp = vec![0.0; dim];
            // Expand over output configurations by carrying (config-prefix, mpo index) weights.
            let mut state: Vec<(usize, Vec<f64>)> = vec![(0, row)];
            for i in 0..n {
                let sp = (col >> (n - 1 - i)) & 1;
                let mut next: Vec<(usize, Vec<f64>)> = Vec::new();
                for (prefix, vec) in &state {
                    for s in 0..PHYS {
                        let mut nv = vec![0.0; MPO_BOND];
                        let mut any = false;
                        for wl in 0..MPO_BOND {
                            let cv = vec[wl];
                            if cv == 0.0 {
                                continue;
                            }
                            for wr in 0..MPO_BOND {
                                let e = w[((wl * PHYS + s) * PHYS + sp) * MPO_BOND + wr];
                                if e != 0.0 {
                                    nv[wr] += cv * e;
                                    any = true;
                                }
                            }
                        }
                        if any {
                            next.push(((prefix << 1) | s, nv));
                        }
                    }
                }
                state = next;
            }
            for (prefix, vec) in &state {
                amp[*prefix] += vec[0];
            }
            let mut unit = vec![0.0; dim];
            unit[col] = 1.0;
            let mut want = vec![0.0; dim];
            apply_h(n, j, g, hz, false, &unit, &mut want);
            for r in 0..dim {
                assert!(
                    (amp[r] - want[r]).abs() < 1e-12,
                    "H[{r},{col}] is {} from the MPO and {} from the Pauli sum",
                    amp[r],
                    want[r]
                );
            }
        }
    }

    /// THE SPEC SAID THE MOMENTUM SUM SOLVES THE OPEN CHAIN. IT SOLVES THE RING.
    ///
    /// `E_0 = −J Σ_k sqrt(1 + g² − 2g cos k)` over EQUALLY SPACED momenta is the periodic chain's
    /// answer with antiperiodic momenta. Both halves are asserted: it matches a periodic
    /// diagonalisation to `1e-9`, and it is wrong about the open chain by an amount far outside any
    /// tolerance — so an implementation that quietly used it as the open-chain oracle would be
    /// caught rather than flattered.
    #[test]
    fn the_momentum_closed_form_is_the_ring_and_the_open_chain_needs_singular_values() {
        for n in [4usize, 6, 8, 10] {
            for &g in &[0.4, 1.0, 1.7] {
                let h = Tfim::new(n, 1.0, g, 0.0).unwrap();
                let momentum = h.periodic_ground_energy().unwrap();
                let (ring, _) = dense_ground(n, 1.0, g, 0.0, true);
                assert!(
                    (momentum - ring).abs() < 1e-9,
                    "n={n} g={g}: the momentum sum {momentum} is not the RING's {ring}"
                );
                let open = h.free_fermion_ground_energy().unwrap();
                assert!(
                    momentum < open - 0.05,
                    "n={n} g={g}: the ring energy {momentum} and the open chain's {open} must not \
                     be confusable, and here they differ by only {}",
                    open - momentum
                );
            }
        }
    }

    /// HEADLINE ORACLE: DMRG reaches the Jordan–Wigner ground energy of the open chain.
    ///
    /// Sixteen sites, bond dimension 32, three points across the transition — `g = 0.5`
    /// (ordered), `g = 1` (critical, where the entanglement is largest and truncation hurts most)
    /// and `g = 1.5` (disordered). Measured errors are `5.3e-13`, `7.1e-15` and `2.1e-14`, so the
    /// tolerance is `1e-11` — what the method actually delivers, not a number chosen to pass.
    ///
    /// **The schedule is load-bearing and the trace does not reveal it.** At 10 sweeps with 14
    /// Krylov vectors the same run stops `3.8e-6` from the answer at `g = 0.5`, and at 20 sweeps
    /// with 12 it stops `2.7e-6` away — while the spread over its last four sweeps is `3.5e-6` and
    /// `2.3e-6` respectively, the same size as the error. A trace whose drift is the size of its
    /// own error tells a reader nothing about which one it is looking at, which is why this asserts
    /// against an external oracle instead.
    #[test]
    fn dmrg_reaches_the_free_fermion_ground_energy() {
        for &g in &[0.5, 1.0, 1.5] {
            let h = Tfim::new(16, 1.0, g, 0.0).unwrap();
            let opts = Options::new(32, 1e-12, 30, 20, 2).unwrap();
            let out = ground_state(&h, &opts, 11).unwrap();
            let exact = h.free_fermion_ground_energy().unwrap();
            assert!(
                (out.energy - exact).abs() < 1e-11,
                "g={g}: DMRG {} vs Jordan-Wigner {exact}, off by {:e}",
                out.energy,
                out.energy - exact
            );
            assert!(
                out.energy >= exact - 1e-11,
                "g={g}: a variational energy may not sit BELOW the exact one"
            );
        }
    }

    /// ORACLE: with a longitudinal field, where Jordan–Wigner does not apply, DMRG matches
    /// brute-force diagonalisation — and the closed form REFUSES to answer rather than guessing.
    ///
    /// The refusal is the asymmetric half. A module that silently dropped `h` from the free-fermion
    /// formula would return a plausible number here, and the only thing that catches it is asserting
    /// that no number comes back.
    #[test]
    fn with_a_longitudinal_field_dmrg_matches_diagonalisation_and_the_closed_form_refuses() {
        for &(g, hz) in &[(0.8, 0.35), (1.4, -0.6), (0.2, 1.0)] {
            let h = Tfim::new(10, 1.0, g, hz).unwrap();
            assert_eq!(
                h.free_fermion_ground_energy(),
                None,
                "Jordan-Wigner does not solve a longitudinal field and must not pretend to"
            );
            let opts = Options::new(24, 1e-12, 20, 16, 2).unwrap();
            let out = ground_state(&h, &opts, 5).unwrap();
            let (dense, _) = dense_ground(10, 1.0, g, hz, false);
            assert!(
                (out.energy - dense).abs() < 1e-9,
                "g={g} h={hz}: DMRG {} vs diagonalisation {dense}",
                out.energy
            );
        }
    }

    /// ORACLE: the energy is an upper bound on the exact ground energy at EVERY sweep, and at every
    /// local update inside every sweep.
    ///
    /// The asymmetric half: the first update of the first sweep must be strictly and visibly ABOVE
    /// the exact energy. A run that started from the answer would satisfy the bound trivially, and
    /// this refuses to call that a test.
    #[test]
    fn the_variational_energy_is_an_upper_bound_at_every_local_update() {
        for &(bond, g) in &[(2usize, 1.0), (8, 1.0), (32, 0.6)] {
            let h = Tfim::new(12, 1.0, g, 0.0).unwrap();
            let exact = h.free_fermion_ground_energy().unwrap();
            let opts = Options::new(bond, 1e-12, 6, 12, 0).unwrap();
            let out = ground_state(&h, &opts, 3).unwrap();
            for (si, sw) in out.sweeps.iter().enumerate() {
                for (li, &e) in sw.local.iter().enumerate() {
                    assert!(
                        e >= exact - 1e-11,
                        "bond={bond} g={g} sweep {si} update {li}: {e} is BELOW the exact ground \
                         energy {exact}, which no variational method may be"
                    );
                }
            }
            assert!(
                out.sweeps[0].local[0] > exact + 0.5,
                "bond={bond} g={g}: the first update already sat at {} against {exact}; a bound \
                 that starts at the answer tests nothing",
                out.sweeps[0].local[0]
            );
        }
    }

    /// MEASURED, AND THE SPEC WAS WRONG: the energy is monotone only while nothing is truncated.
    ///
    /// Each Lanczos step minimises over a space containing the current state, so it cannot raise the
    /// energy — but the singular-value truncation that follows can, and does. Both halves are
    /// asserted: with the bond cap slack enough that the discarded weight is zero the sequence is
    /// monotone to `1e-12`, and at bond dimension 2 on a critical chain it is NOT, which is the
    /// assertion that fails for an implementation that quietly stopped truncating.
    #[test]
    fn truncation_breaks_monotonicity_and_an_untruncated_sweep_keeps_it() {
        let h = Tfim::new(12, 1.0, 1.0, 0.0).unwrap();

        let slack = Options::new(64, 0.0, 5, 14, 0).unwrap();
        let clean = ground_state(&h, &slack, 21).unwrap();
        for (si, sw) in clean.sweeps.iter().enumerate() {
            assert_eq!(sw.truncation, 0.0, "sweep {si} discarded {}", sw.truncation);
            for k in 1..sw.local.len() {
                assert!(
                    sw.local[k] <= sw.local[k - 1] + 1e-12,
                    "sweep {si} update {k}: {} rose from {} with nothing discarded",
                    sw.local[k],
                    sw.local[k - 1]
                );
            }
        }

        let tight = Options::new(2, 1e-12, 8, 14, 0).unwrap();
        let squeezed = ground_state(&h, &tight, 21).unwrap();
        let mut rises = 0usize;
        let mut worst = 0.0f64;
        for sw in &squeezed.sweeps {
            for k in 1..sw.local.len() {
                let d = sw.local[k] - sw.local[k - 1];
                if d > 1e-12 {
                    rises += 1;
                    worst = worst.max(d);
                }
            }
        }
        assert!(
            rises > 0,
            "at bond dimension 2 on a critical chain the truncation must sometimes raise the \
             energy; none of {} updates did, so either nothing was discarded or the recorded \
             energy is not the post-truncation one",
            squeezed.sweeps.iter().map(|s| s.local.len()).sum::<usize>()
        );
        assert!(
            squeezed.sweeps.iter().any(|s| s.truncation > 1e-6),
            "the squeezed run must actually discard weight"
        );
        assert!(worst > 1e-6, "the worst rise was only {worst}, too small to be the truncation");
    }

    /// ORACLE: at bond dimension 1 the ansatz IS a product state, so DMRG must land on the
    /// mean-field energy computed by exact coordinate descent — an independent routine, using the
    /// closed-form single-site minimiser `−sqrt(A² + g²)` and nothing from this module.
    ///
    /// **The spec said two-site DMRG does this. It does not.** Measured on this chain, two-site
    /// updates stop `4.2e-4` above the product optimum at `g = 0.4`, `2.7e-2` at `g = 1.0` and
    /// `1.2e-1` at `g = 1.6`, and 300 sweeps leave it exactly there — the rank-1 truncation keeps
    /// the CLOSEST product state to the optimal two-site state, not the cheapest product state. The
    /// single-site sweeps of [`Options::polish`] are what make the claim true, to `2e-14`, and both
    /// halves are asserted below: the bound-1 run must match, and the two-site-only run must NOT.
    ///
    /// The other asymmetric half: the mean-field energy must sit clearly ABOVE the exact ground
    /// energy, or the test would also pass for an implementation that ignored the bond cap and
    /// converged to the true ground state.
    #[test]
    fn bond_dimension_one_is_the_product_state_mean_field() {
        for &(g, hz) in &[(0.4, 0.0), (1.0, 0.0), (1.6, 0.0), (0.7, 0.3)] {
            let n = 10;
            let h = Tfim::new(n, 1.0, g, hz).unwrap();
            let opts = Options::new(1, 0.0, 4, 8, 400).unwrap();
            let out = ground_state(&h, &opts, 2).unwrap();
            let mf = product_mean_field(n, 1.0, g, hz);
            assert!(
                (out.energy - mf).abs() < 1e-12,
                "g={g} h={hz}: bond-1 DMRG {} vs closed-form mean field {mf}, off by {:e}",
                out.energy,
                out.energy - mf
            );
            assert_eq!(out.mps.max_bond(), 1, "the bond cap was not respected");
            let (exact, _) = dense_ground(n, 1.0, g, hz, false);
            // Deep in the ordered phase a product state is already nearly exact -- the gap is
            // 9.3e-3 at g = 0.4 against 0.31 at g = 1.6 -- so the margin is set by the SMALLEST of
            // the four, and it is still ten orders above the 1e-12 the match is asserted to.
            assert!(
                mf > exact + 5e-3,
                "g={g} h={hz}: mean field {mf} and the exact energy {exact} differ by only {:e}, \
                 so this instance cannot tell a product state from a correlated one",
                mf - exact
            );
        }

        // The asymmetric half: WITHOUT the single-site polish the same run misses, and by a margin
        // far outside the tolerance above. A polish phase that had quietly become a no-op would
        // pass every assertion in the first loop and fail here.
        let h = Tfim::new(10, 1.0, 1.6, 0.0).unwrap();
        let two_site_only = Options::new(1, 0.0, 300, 12, 0).unwrap();
        let stuck = ground_state(&h, &two_site_only, 2).unwrap();
        let mf = product_mean_field(10, 1.0, 1.6, 0.0);
        assert!(
            stuck.energy - mf > 1e-2,
            "two-site DMRG at bond 1 reached {} against the product optimum {mf}; if it now gets \
             there on its own, the polish phase is no longer what closes the gap and this module's \
             documentation is wrong",
            stuck.energy
        );
    }

    /// A single-site sweep changes no bond dimension, so it discards nothing and CANNOT raise the
    /// energy — the property the two-site sweep does not have.
    ///
    /// Asserted as the pair: every polish sweep reports exactly zero truncation, and its energy
    /// trace is monotone across the whole sweep, on the same critical chain where the two-site
    /// sweep at bond dimension 2 is not.
    #[test]
    fn a_single_site_sweep_discards_nothing_and_is_monotone() {
        let h = Tfim::new(12, 1.0, 1.0, 0.2).unwrap();
        let opts = Options::new(8, 1e-12, 4, 12, 6).unwrap();
        let out = ground_state(&h, &opts, 41).unwrap();
        let polish: Vec<&Sweep> = out.sweeps.iter().filter(|s| s.single_site).collect();
        assert_eq!(polish.len(), 6, "six polish sweeps were asked for");
        let mut previous = f64::INFINITY;
        for (si, sw) in polish.iter().enumerate() {
            assert_eq!(sw.truncation, 0.0, "polish sweep {si} discarded weight");
            for k in 0..sw.local.len() {
                assert!(
                    sw.local[k] <= previous + 1e-12,
                    "polish sweep {si} update {k}: {} rose from {previous}",
                    sw.local[k]
                );
                previous = sw.local[k];
            }
            assert!(sw.isometry_defect < 1e-13, "polish sweep {si} broke the canonical form");
        }
    }

    /// ORACLE: at zero transverse field the model is CLASSICAL, and the answer must be the one the
    /// rest of the crate already gives — `crate::exact` by enumeration and `crate::meanfield` in its
    /// zero-temperature limit, on the same chain built through `crate::graph`.
    ///
    /// This is the seam between this module and the classical half of the crate. If the sign
    /// convention here disagreed with `E = −J s s − h s`, this is what would say so.
    #[test]
    fn at_zero_transverse_field_it_agrees_with_exact_and_meanfield() {
        let n = 8;
        let (j, hz) = (1.0, 0.25);
        let mut b = crate::graph::GraphBuilder::new(n);
        for i in 0..n - 1 {
            b.couple(i, i + 1, j);
        }
        for i in 0..n {
            b.bias(i, hz);
        }
        let gph = b.build();

        let classical = crate::exact::Elimination::default()
            .ground_state(&gph)
            .unwrap()
            .ground_energy
            .unwrap();
        let h = Tfim::new(n, j, 0.0, hz).unwrap();
        let opts = Options::new(4, 1e-12, 6, 8, 2).unwrap();
        let out = ground_state(&h, &opts, 4).unwrap();
        assert!(
            (out.energy - classical).abs() < 1e-10,
            "DMRG at g=0 gives {} and exact enumeration gives {classical}",
            out.energy
        );

        // Mean field at large beta: ln Z -> -beta E_0 on a ferromagnet with a field, so the
        // Gibbs-Bogoliubov bound divided by -beta is the classical ground energy.
        let beta = 60.0;
        let mf = crate::meanfield::naive_mean_field(&gph, beta, 4000, 0.3);
        let mf_energy = -mf.log_z / beta;
        assert!(
            (mf_energy - classical).abs() < 1e-8,
            "meanfield at beta={beta} gives {mf_energy} against {classical}"
        );
        assert!(
            (out.energy - mf_energy).abs() < 1e-8,
            "bond-4 DMRG {} and the meanfield module {mf_energy} must agree where the model is \
             classical and its ground state is a product state",
            out.energy
        );
    }

    /// ORACLE: canonical form is EXACT, so the isometry conditions hold to machine precision after
    /// every sweep — not to a tolerance chosen to accommodate the implementation.
    ///
    /// The asymmetric half: the same assertion is made on a state that has been deliberately
    /// damaged, and it must FAIL there. A defect measure that returns something small for every
    /// state measures nothing.
    #[test]
    fn canonical_form_is_exact_after_every_sweep() {
        let h = Tfim::new(14, 1.0, 1.0, 0.1).unwrap();
        let opts = Options::new(16, 1e-12, 6, 12, 2).unwrap();
        let out = ground_state(&h, &opts, 13).unwrap();
        for (si, sw) in out.sweeps.iter().enumerate() {
            assert!(
                sw.isometry_defect < 1e-13,
                "sweep {si} left an isometry defect of {:e}",
                sw.isometry_defect
            );
        }
        assert!((out.mps.norm_squared() - 1.0).abs() < 1e-13);

        let mut damaged = out.mps.clone();
        let far = if damaged.centre() == 0 { damaged.n() - 1 } else { 0 };
        damaged.a[far][0] += 1e-6;
        assert!(
            damaged.isometry_defect() > 1e-9,
            "the defect measure did not see a 1e-6 dent in an isometry, so it is not measuring one"
        );
    }

    /// ORACLE: the Schmidt spectrum across the middle bond matches the one read off the exactly
    /// diagonalised ground state.
    ///
    /// The strongest statement available about the STATE rather than its energy: two different
    /// states can sit within `1e-12` in energy near a minimum, and they cannot share a Schmidt
    /// spectrum. It also checks the canonical form is the thing it claims to be — Schmidt values are
    /// only readable off an MPS at all when the isometries hold.
    #[test]
    fn the_schmidt_spectrum_matches_exact_diagonalisation() {
        let (n, g) = (8usize, 1.2);
        let h = Tfim::new(n, 1.0, g, 0.0).unwrap();
        let opts = Options::new(16, 0.0, 14, 16, 2).unwrap();
        let out = ground_state(&h, &opts, 17).unwrap();
        let (_, vec) = dense_ground(n, 1.0, g, 0.0, false);

        let half = 1usize << (n / 2);
        let mut gram = vec![0.0; half * half];
        for i in 0..half {
            for jj in 0..half {
                let mut acc = 0.0;
                for c in 0..half {
                    acc += vec[i * half + c] * vec[jj * half + c];
                }
                gram[i * half + jj] = acc;
            }
        }
        let _ = jacobi_eig(&mut gram, half);
        let mut want: Vec<f64> = (0..half).map(|i| gram[i * half + i].max(0.0).sqrt()).collect();
        want.sort_by(|a, b| b.total_cmp(a));

        let got = out.mps.schmidt(n / 2).unwrap();
        for (k, &s) in got.iter().enumerate() {
            assert!(
                (s - want[k]).abs() < 1e-7,
                "Schmidt value {k} is {s} from the MPS and {} from diagonalisation",
                want[k]
            );
        }
        let entropy = out.mps.entanglement_entropy(n / 2).unwrap();
        let want_entropy: f64 =
            -want.iter().map(|&s| if s > 0.0 { s * s * (s * s).ln() } else { 0.0 }).sum::<f64>();
        assert!(
            (entropy - want_entropy).abs() < 1e-7,
            "entanglement entropy {entropy} vs {want_entropy}"
        );

        // And the amplitudes themselves, up to the global sign an eigenvector does not fix.
        let dense = out.mps.to_dense().unwrap();
        let overlap: f64 = dense.iter().zip(vec.iter()).map(|(a, b)| a * b).sum();
        assert!(
            (overlap.abs() - 1.0).abs() < 1e-9,
            "the DMRG state overlaps the exact ground state by {overlap}, not 1"
        );
    }

    /// The certified bound is SOUND — never below what it bounds — and NOT VACUOUS.
    ///
    /// Both halves matter. A bound of `+infinity` is sound and useless; a bound accumulated in
    /// round-to-nearest is tight and occasionally false, which is the defect
    /// [`crate::round`] exists because of.
    #[test]
    fn the_certificate_is_sound_and_not_vacuous() {
        for &g in &[0.5, 1.0, 1.9] {
            let h = Tfim::new(12, 1.0, g, 0.0).unwrap();
            let exact = h.free_fermion_ground_energy().unwrap();
            let opts = Options::new(16, 1e-12, 12, 14, 2).unwrap();
            let out = ground_state(&h, &opts, 29).unwrap();
            assert!(
                out.certified >= out.energy,
                "g={g}: the certificate {} is below the contraction it certifies {}",
                out.certified,
                out.energy
            );
            assert!(
                out.certified >= exact,
                "g={g}: the certificate {} is below the exact ground energy {exact}",
                out.certified
            );
            assert!(
                out.certified - out.energy < 1e-9,
                "g={g}: the certificate is {} wide, which is not a useful bound",
                out.certified - out.energy
            );
        }
    }

    /// Same seed, same numbers, bit for bit; different seed, same physics.
    #[test]
    fn the_run_is_deterministic_by_seed_and_seed_independent_in_its_answer() {
        let h = Tfim::new(10, 1.0, 0.9, 0.0).unwrap();
        let opts = Options::new(16, 1e-12, 12, 14, 2).unwrap();
        let a = ground_state(&h, &opts, 99).unwrap();
        let b = ground_state(&h, &opts, 99).unwrap();
        assert_eq!(a.energy.to_bits(), b.energy.to_bits(), "same seed must give the same bits");
        let c = ground_state(&h, &opts, 100).unwrap();
        assert!(a.energy.to_bits() != c.energy.to_bits(), "two seeds produced identical bits");
        assert!(
            (a.energy - c.energy).abs() < 1e-9,
            "two seeds must find the same ground energy: {} vs {}",
            a.energy,
            c.energy
        );
    }

    /// Bad input is a typed error naming what it saw, never a substituted default.
    #[test]
    fn bad_input_is_a_typed_error_naming_what_it_saw() {
        assert_eq!(Tfim::new(1, 1.0, 1.0, 0.0), Err(MpsError::TooFewSites { n: 1 }));
        // NaN != NaN, so the coupling case is matched rather than compared -- an `assert_eq!` on an
        // error carrying a NaN can never hold, and writing one is how a test comes to assert
        // nothing.
        match Tfim::new(4, f64::NAN, 1.0, 0.0) {
            Err(MpsError::NotFinite { parameter: Parameter::Coupling, value }) => {
                assert!(value.is_nan());
            }
            other => panic!("expected a named coupling error, got {other:?}"),
        }
        match Tfim::new(4, 1.0, f64::INFINITY, 0.0) {
            Err(MpsError::NotFinite { parameter: Parameter::TransverseField, value }) => {
                assert!(value.is_infinite());
            }
            other => panic!("expected a named transverse-field error, got {other:?}"),
        }
        assert_eq!(Options::new(0, 1e-12, 4, 8, 1), Err(MpsError::ZeroBond));
        assert_eq!(Options::new(4, 1.0, 4, 8, 1), Err(MpsError::BadCutoff { cutoff: 1.0 }));
        assert_eq!(Options::new(4, -0.1, 4, 8, 1), Err(MpsError::BadCutoff { cutoff: -0.1 }));
        assert_eq!(Options::new(4, 1e-12, 0, 8, 1), Err(MpsError::NoSweeps));
        assert_eq!(Options::new(4, 1e-12, 4, 1, 1), Err(MpsError::BadKrylov { krylov: 1 }));
        let msg = MpsError::TooFewSites { n: 1 }.to_string();
        assert!(msg.contains('1') && msg.contains("two-site"), "unhelpful message: {msg}");
        assert!(
            MpsError::NotFinite { parameter: Parameter::LongitudinalField, value: f64::NAN }
                .to_string()
                .contains("longitudinal"),
            "the message must name the parameter"
        );
    }

    /// Accessors report the state that is there, and refuse indices that are not.
    #[test]
    fn accessors_refuse_indices_that_do_not_exist() {
        let m = Mps::random(6, 4, 1);
        assert_eq!(m.n(), 6);
        assert_eq!(m.bond_dim(0), Some(1));
        assert_eq!(m.bond_dim(6), Some(1));
        assert_eq!(m.bond_dim(7), None);
        assert!(m.tensor(6).is_none());
        assert!(m.schmidt(7).is_none());
        assert!(m.entanglement_entropy(7).is_none());
        assert!(m.amplitude(&[0, 0, 0, 0, 0]).is_none(), "a short configuration is not a state");
        assert!(m.amplitude(&[0, 1, 0, 1, 0, 2]).is_none(), "a spin index of 2 is not a spin");
        assert!(m.amplitude(&[0, 1, 0, 1, 0, 1]).is_some());
        assert!(m.isometry_defect() < 1e-13, "a fresh random state must be canonical too");

        // The amplitudes and the dense vector are the same object seen two ways.
        let dense = m.to_dense().unwrap();
        assert_eq!(dense.len(), 64);
        for idx in 0..64usize {
            let cfg: Vec<u8> = (0..6).map(|i| ((idx >> (5 - i)) & 1) as u8).collect();
            assert!((m.amplitude(&cfg).unwrap() - dense[idx]).abs() < 1e-14);
        }
        let norm: f64 = dense.iter().map(|x| x * x).sum();
        assert!((norm - 1.0).abs() < 1e-12, "a canonical state is normalised, got {norm}");
    }
}
