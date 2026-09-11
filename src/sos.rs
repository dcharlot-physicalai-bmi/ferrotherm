//! **Lasserre level 2** — the degree-four moment relaxation of an Ising ground state, and a
//! certified lower bound read off its sum-of-squares dual.
//!
//! [`crate::sdp`] computes the standard max-cut relaxation and certifies it. That is **level 1** of
//! a hierarchy, and the hierarchy above it was absent from this crate: the words "Lasserre" and
//! "sum of squares" appeared zero times. This module is the next rung.
//!
//! # What the level is
//!
//! Level 1 asks for a positive semidefinite matrix indexed by the spins themselves, `{1, s_1, …,
//! s_n}`. Level 2 asks for one indexed by every monomial of degree at most two,
//! `{1} ∪ {s_i} ∪ {s_i s_j}` — `N = 1 + n + n(n−1)/2` rows. [`Level2`] is that index set.
//!
//! Two identifications make the bigger matrix a **relaxation of the same problem** rather than a
//! different problem:
//!
//!  * `s_i² = 1`, so a product of two square-free monomials is their **symmetric difference**:
//!    `(s_i s_j)(s_j s_k) = s_i s_k`, not `s_i s_j² s_k`.
//!  * every entry `M[A][B]` therefore *names* the monomial `A Δ B`, and **all entries naming the
//!    same monomial must be equal**. `M[{i}][{j}]`, `M[∅][{i,j}]` and `M[{i,k}][{j,k}]` are four
//!    of the ways to write `y_{ij}`, and a matrix that let them disagree would be relaxing
//!    something else entirely.
//!
//! So the free variables are one per monomial class, `y_C` for `|C| ≤ 4`, and the moment matrix is
//! `M(y)[A][B] = y_{A Δ B}` — the consistency constraints hold **by construction of the index**,
//! which is why [`Level2::class_of`] is the load-bearing object here. [`Level2::moments`] produces
//! the `y` of an actual state, and [`Level2::moment_matrix`] of that `y` is the rank-one matrix
//! `z zᵀ`; every `±1` state is feasible, which is what makes the minimum over `y` a lower bound.
//!
//! # The bound comes from the dual, because the primal bounds from the wrong side
//!
//! Exactly as in [`crate::sdp`]: a primal-feasible `y` gives a number **above** the relaxation's
//! optimum and so proves nothing. The dual of the moment problem is a **sum-of-squares
//! certificate**: a positive semidefinite Gram matrix `Q`, also indexed by the degree-≤2 monomials,
//! with
//!
//! ```text
//! E(s) − λ  =  z(s)ᵀ Q z(s)     for every s ∈ {−1,+1}ⁿ,     z(s)_A = Π_{i∈A} s_i
//! ```
//!
//! The right-hand side is a sum of squares, hence `≥ 0`, hence `E(s) ≥ λ` — for **every** state, with
//! no optimality, convergence or rank assumption anywhere. Matching monomials on both sides gives
//! `λ = −tr Q`, and the search for a good `Q` is a heuristic that can only move the bound down.
//!
//! # Certified means certified in floating point
//!
//! `E(s) − λ = zᵀQz` cannot be made to hold *exactly* in `f64`, and a certificate that is exact
//! "up to rounding" certifies nothing. So the identity is never assumed. Write
//! `q_C = Σ_{AΔB=C} Q[A][B]` for what `Q` actually says about monomial `C`, and `e_C` for what the
//! energy says. Then for every state,
//!
//! ```text
//! E(s)  =  z(s)ᵀQz(s)  +  Σ_C (e_C − q_C) s^C  ≥  −q_∅  −  Σ_{C≠∅} |e_C − q_C|
//! ```
//!
//! because `|s^C| = 1`. **That** is the number this module returns, and it is valid for any `Q`
//! whatsoever that is positive semidefinite — converged or not, feasible or not. An unconverged
//! search shows up as a loose bound, never as a wrong one.
//!
//! Each rounding therefore goes in the direction that can only *lower* the claim:
//!
//! | quantity | direction | why |
//! |---|---|---|
//! | `−tr Q` | DOWN, via [`crate::sdp::Certificate::verify`]'s own `sum_down` | it is added, so understating it weakens the bound |
//! | `\|e_C − q_C\|` | UP, `max(\|sum_down\|, \|sum_up\|)` | it is subtracted, so overstating it weakens the bound |
//! | the total | DOWN, [`crate::round::sum_down`] | the result is a lower bound |
//!
//! This crate shipped a "lower bound" that sat above the optimum because it summed in
//! round-to-nearest ([`crate::round`] records it), so none of the three is decoration.
//!
//! # The one claim that has to be true is `Q ⪰ 0`, and level 1 already verifies that
//!
//! Rather than write a second Cholesky and a second copy of [Rump 2006]'s criterion, the Gram
//! matrix is handed to the level-1 verifier. [`crate::sdp::Certificate::verify`] proves
//! `C − Diag(y) ≻ 0` for the cost matrix `C` of a graph, and `C[a][b] = −J_ab/2` — so the **lifted
//! graph** on `N` nodes with `J_ab = −2 Q[a][b]` and no fields has `C = ` the off-diagonal of `Q`,
//! and `y = −diag Q` makes `C − Diag(y)` equal `Q` **bit for bit** (multiplying and dividing by two
//! are exact). One verifier, one Rump constant, one place for that argument to live.
//!
//! A converged `Q` is singular — the optimum of a semidefinite program is on the boundary — and a
//! Cholesky proves *definiteness*, so the raw `Q` never verifies. The diagonal is shifted up by the
//! smallest `δ` that does verify, which costs exactly `δ·N` of bound and is reported as
//! [`Certificate::shift`].
//!
//! # The search
//!
//! Douglas–Rachford splitting between the two convex sets the certificate must lie in: the affine
//! set `{Q : q_C = e_C for C ≠ ∅, q_∅ = −λ}`, whose projection is closed-form because the monomial
//! classes **partition the entries**, and the positive semidefinite cone, via
//! [`crate::linalg::jacobi_eig`]. `λ` itself is bisected between the level-1 bound (feasible, and
//! used as the warm start) and a Goemans–Williamson rounding's energy (an upper limit, since
//! `λ ≤ min_s E(s)`), both taken from [`crate::sdp`]. Plain alternating projections were the first
//! version and are ten orders of magnitude worse at the same iteration count; `solve` carries the
//! measurement and the reason.
//!
//! Everything in that paragraph is a heuristic. The bound is whatever the best certificate found
//! re-verifies to.
//!
//! # What it is worth
//!
//! Measured at [`Params::default`], against level 1 from [`crate::sdp::certified`] and the exact
//! optimum by enumeration:
//!
//! | instance | level 1 | **level 2** | exact |
//! |---|---|---|---|
//! | `C₃` frustrated triangle | −1.500000 | **−1.000008** | −1 |
//! | `C₅` frustrated pentagon | −4.045085 | **−3.000017** | −3 |
//! | `C₇` | −6.306782 | **−5.000022** | −5 |
//! | `C₉` | −8.457234 | **−7.000216** | −7 |
//! | `K₅` | −2.500000 | −2.500000 | −2 |
//! | `K₇` | −3.500000 | −3.500000 | −3 |
//! | 10 random `n = 8` instances, mean gap | 0.0819 | **0.0000** | — |
//!
//! **The odd cycles are the point.** `C_n` antiferromagnetic is the textbook instance where level 1
//! is loose: its relaxation value is the closed form `−n cos(π/n)`, a 33% to 50% gap that no amount
//! of sweeping removes because it is the relaxation and not the solver. The degree-4 moment matrix
//! implies the odd-cycle inequalities, and level 2 is exact on every one of them — what is left is
//! the bisection's own resolution, not the relaxation's.
//!
//! **`K₅` is the honest other half.** Level 2 buys nothing there and this module says so: the
//! complete antiferromagnet's optimal pair moments are `−1/(n−1)`, every triangle inequality reads
//! `3·(−1/(n−1)) ≥ −1` and is slack for `n ≥ 4`, so there is nothing for degree 4 to tighten. A
//! level-2 number ABOVE `−n/2` on `K₅` would not be a better relaxation, it would be an unsound one.
//!
//! The cost is the reason the hierarchy is not simply run at level 3: `N = O(n²)` rows and an
//! eigendecomposition per iteration, so the work is `O(n⁶)`. [`MAX_SPINS`] refuses rather than
//! pretends.
//!
//! [Lasserre 2001]: J. B. Lasserre, "Global optimization with polynomials and the problem of
//! moments", SIAM Journal on Optimization 11(3):796–817.
//! [Parrilo 2003]: P. A. Parrilo, "Semidefinite programming relaxations for semialgebraic
//! problems", Mathematical Programming 96:293–320.
//! [Laurent 2003]: M. Laurent, "A comparison of the Sherali-Adams, Lovász-Schrijver and Lasserre
//! relaxations for 0-1 programming", Mathematics of Operations Research 28(3):470–496.
//! [Rump 2006]: S. M. Rump, "Verification of positive definiteness", BIT Numerical Mathematics.

use crate::bound::Bound;
use crate::graph::{Graph, GraphBuilder};
use crate::round::{sum_down, sum_up};
use crate::sdp;

/// Spin counts above this are refused rather than attempted.
///
/// The moment matrix has `1 + n + n(n−1)/2` rows and the search eigendecomposes it, so the work per
/// iteration grows as `n⁶`: 301 rows at `n = 24`, 5,051 at `n = 100`. A relaxation that takes longer
/// than the exhaustive enumeration it is meant to replace is not a relaxation, and silently
/// accepting `n = 200` would produce exactly that.
pub const MAX_SPINS: usize = 24;

/// Why a level-2 certificate could not be produced or could not be re-verified.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SosError {
    /// The graph has more spins than [`MAX_SPINS`].
    TooWide {
        /// Spins asked for.
        n: usize,
        /// The cap.
        max: usize,
    },
    /// The Gram matrix is not the size this graph's monomial basis implies.
    Shape {
        /// Entries supplied.
        got: usize,
        /// Entries required.
        want: usize,
    },
    /// `Q[a][b]` and `Q[b][a]` differ. A Gram matrix is symmetric by definition, and `zᵀQz` would
    /// otherwise be a claim about a matrix nobody wrote down.
    NotSymmetric {
        /// Row of the first of the two entries.
        a: usize,
        /// Column of the first of the two entries.
        b: usize,
    },
    /// The Cholesky did not complete: `Q` is not provably positive semidefinite, so `zᵀQz` is not
    /// provably a sum of squares and the certificate proves nothing.
    NotPsd,
    /// A non-finite entry.
    NotFinite,
}

impl core::fmt::Display for SosError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            SosError::TooWide { n, max } => write!(
                f,
                "level 2 needs a {}-row moment matrix for {n} spins, and this module refuses past \
                 {max}",
                Level2::dim_for(*n)
            ),
            SosError::Shape { got, want } => {
                write!(f, "the Gram matrix has {got} entries and this basis needs {want}")
            }
            SosError::NotSymmetric { a, b } => {
                write!(f, "the Gram matrix is not symmetric at ({a}, {b})")
            }
            SosError::NotPsd => write!(
                f,
                "the Gram matrix did not verify as positive definite, so z'Qz is not a sum of \
                 squares and this certificate proves nothing"
            ),
            SosError::NotFinite => write!(f, "the certificate contains a non-finite value"),
        }
    }
}

impl std::error::Error for SosError {}

/// The level-2 index: the degree-≤2 monomial basis, and the monomial each matrix entry names.
///
/// This is the relaxation. Everything else in the module is arithmetic on top of it.
///
/// Row `a` is the square-free monomial `basis[a]`, held as a bitmask over spins. Entry `(a, b)`
/// names the monomial `basis[a] Δ basis[b]` — the symmetric difference, because `s_i² = 1` — and
/// entries naming the same monomial form a **class**. Two facts follow and both are used:
///
///  * the classes **partition** the `N²` entries, so projecting a matrix onto "every class sums to
///    its target" is a closed-form, one-pass operation (`solve`'s affine step);
///  * a moment matrix is a *single* value per class ([`Level2::moment_matrix`]), so the consistency
///    constraints cannot be violated by construction rather than being imposed and checked.
#[derive(Clone, Debug)]
pub struct Level2 {
    n: usize,
    dim: usize,
    basis: Vec<u64>,
    /// `class_of[a * dim + b]`.
    class: Vec<u32>,
    /// Class id -> the monomial it names, as a spin bitmask.
    monomials: Vec<u64>,
    /// Class id -> how many matrix entries belong to it.
    sizes: Vec<u32>,
    /// The class of the empty monomial, i.e. the constant term. Held rather than assumed to be
    /// zero, so the ordering of `monomials` stays an implementation detail.
    empty: usize,
}

/// The monomial an entry names: `s^A · s^B = s^{A Δ B}`, since every `s_i² = 1`.
///
/// **The whole hierarchy is in this line.** Replace the symmetric difference with a union and the
/// matrix stops being a moment matrix: `(s_i s_j)(s_j s_k)` would name `s_i s_j s_k` instead of
/// `s_i s_k`, the `s_j² = 1` identification would be gone, and the "relaxation" would be of a
/// problem with no relation to the one asked about.
#[inline]
fn product(a: u64, b: u64) -> u64 {
    a ^ b
}

impl Level2 {
    /// How many rows the level-2 moment matrix has for `n` spins: `1 + n + n(n−1)/2`.
    #[must_use]
    pub fn dim_for(n: usize) -> usize {
        1 + n + n * n.saturating_sub(1) / 2
    }

    /// The monomial basis and class map for `n` spins.
    ///
    /// # Errors
    ///
    /// [`SosError::TooWide`] past [`MAX_SPINS`]; see that constant for why there is a cap at all.
    pub fn new(n: usize) -> Result<Level2, SosError> {
        if n > MAX_SPINS {
            return Err(SosError::TooWide { n, max: MAX_SPINS });
        }
        let dim = Level2::dim_for(n);
        let mut basis = Vec::with_capacity(dim);
        basis.push(0u64);
        for i in 0..n {
            basis.push(1u64 << i);
        }
        for i in 0..n {
            for j in (i + 1)..n {
                basis.push((1u64 << i) | (1u64 << j));
            }
        }
        // BTreeMap, not HashMap: this map's iteration order would decide the class numbering, and
        // a class numbering that depends on which run produced it makes every certificate in this
        // module unreproducible. `graph::GraphBuilder::build` records the same lesson.
        let mut ids: std::collections::BTreeMap<u64, u32> = std::collections::BTreeMap::new();
        let mut monomials: Vec<u64> = Vec::new();
        let mut sizes: Vec<u32> = Vec::new();
        let mut class = vec![0u32; dim * dim];
        for a in 0..dim {
            for b in 0..dim {
                let mask = product(basis[a], basis[b]);
                let id = *ids.entry(mask).or_insert_with(|| {
                    monomials.push(mask);
                    sizes.push(0);
                    (monomials.len() - 1) as u32
                });
                class[a * dim + b] = id;
                sizes[id as usize] += 1;
            }
        }
        let empty = ids[&0] as usize;
        Ok(Level2 { n, dim, basis, class, monomials, sizes, empty })
    }

    /// Spins.
    #[must_use]
    pub fn n(&self) -> usize {
        self.n
    }

    /// Rows of the moment matrix.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Monomial classes, one free variable each: the count of distinct `A Δ B`.
    #[must_use]
    pub fn classes(&self) -> usize {
        self.monomials.len()
    }

    /// Row `a`'s monomial, as a bitmask over spins.
    #[must_use]
    pub fn basis(&self) -> &[u64] {
        &self.basis
    }

    /// Which monomial class entry `(a, b)` belongs to.
    ///
    /// # Panics
    ///
    /// If either index is past [`Level2::dim`].
    #[must_use]
    pub fn class_of(&self, a: usize, b: usize) -> usize {
        assert!(a < self.dim && b < self.dim, "({a},{b}) outside a {}-row basis", self.dim);
        self.class[a * self.dim + b] as usize
    }

    /// The monomial class `c` names, as a bitmask over spins.
    ///
    /// # Panics
    ///
    /// If `c` is past [`Level2::classes`].
    #[must_use]
    pub fn monomial(&self, c: usize) -> u64 {
        self.monomials[c]
    }

    /// The class of the constant monomial `1`, whose moment is fixed to one.
    #[must_use]
    pub fn constant_class(&self) -> usize {
        self.empty
    }

    /// The moments of an actual state: `y_C = Π_{i∈C} s_i`, one per class.
    ///
    /// **This is what makes the relaxation a relaxation.** Every `±1` state maps to a feasible
    /// point of the moment problem — [`Level2::moment_matrix`] of this `y` is `z zᵀ`, positive
    /// semidefinite and rank one — so the minimum over the relaxed set is never above `min_s E(s)`.
    ///
    /// # Panics
    ///
    /// If `s` is shorter than the spin count.
    #[must_use]
    pub fn moments(&self, s: &[i8]) -> Vec<f64> {
        assert!(s.len() >= self.n, "a state of {} spins, for {} spins", s.len(), self.n);
        self.monomials
            .iter()
            .map(|&m| {
                let mut v = 1.0f64;
                for i in 0..self.n {
                    if m >> i & 1 == 1 {
                        v *= f64::from(s[i]);
                    }
                }
                v
            })
            .collect()
    }

    /// The moment matrix `M(y)[A][B] = y_{A Δ B}`, row-major.
    ///
    /// The consistency constraints are not imposed here; they are unrepresentable here, because
    /// there is one input value per class and every entry of a class reads it.
    ///
    /// # Panics
    ///
    /// If `y` is not one value per class.
    #[must_use]
    pub fn moment_matrix(&self, y: &[f64]) -> Vec<f64> {
        assert_eq!(y.len(), self.classes(), "one moment per monomial class");
        self.class.iter().map(|&c| y[c as usize]).collect()
    }

    /// The energy as a vector over monomial classes: `E(s) = Σ_C e_C s^C`.
    ///
    /// `e_{i} = −h_i` and `e_{ij} = −J_ij`, which is the crate's `E = −Σ J s s − Σ h s` written in
    /// the monomial basis; everything else is zero, including the constant, and including every
    /// monomial of degree three and four — those classes exist in the matrix and the energy says
    /// nothing about them, which is precisely the freedom level 2 buys over level 1.
    ///
    /// # Panics
    ///
    /// If the graph's spin count is not this basis's.
    #[must_use]
    pub fn objective(&self, g: &Graph) -> Vec<f64> {
        assert_eq!(g.n, self.n, "objective for a {}-spin graph on a {}-spin basis", g.n, self.n);
        let mut e = vec![0.0f64; self.classes()];
        for i in 0..self.n {
            // Row 0 is the constant monomial and row 1+i is s_i, so (0, 1+i) names s_i.
            e[self.class_of(0, 1 + i)] = -g.h[i];
            for k in g.offset[i]..g.offset[i + 1] {
                let j = g.nbr[k] as usize;
                if j > i {
                    e[self.class_of(1 + i, 1 + j)] = -g.w[k];
                }
            }
        }
        e
    }
}

/// How hard to try. None of it affects soundness.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Params {
    /// Bisection rounds on `λ`, between the level-1 bound and a rounded state's energy.
    pub rounds: usize,
    /// Alternating projections per round. More is tighter and slower, and never unsound.
    pub iters: usize,
    /// Hyperplanes for the Goemans–Williamson rounding that supplies the bisection's upper limit.
    pub hyperplanes: usize,
    /// Level-1 parameters, for the warm start and the upper limit.
    pub sdp: sdp::Params,
}

impl Default for Params {
    fn default() -> Self {
        Params { rounds: 16, iters: 400, hyperplanes: 32, sdp: sdp::Params::default() }
    }
}

/// A sum-of-squares certificate, and the bound it proves.
///
/// The artefact a sceptic runs. [`Certificate::verify`] rebuilds the monomial basis from the graph,
/// re-checks that the Gram matrix is positive definite, re-measures how far the sum-of-squares
/// identity misses by, and re-sums the bound — with no reference to the projections, the bisection,
/// the level-1 warm start or the seed that produced it.
#[derive(Clone, Debug)]
pub struct Certificate {
    /// The Gram matrix `Q`, row-major, `dim × dim`, **including** the diagonal shift.
    pub q: Vec<f64>,
    /// Rows of `Q`, which is [`Level2::dim`] for the graph's spin count.
    pub dim: usize,
    /// Spins the certificate is about.
    pub n: usize,
    /// The bound: `−tr Q` less the sum-of-squares residual, rounded down.
    pub value: f64,
    /// `Σ_{C≠∅} |e_C − q_C|` — how far the identity `E − λ = zᵀQz` misses, and therefore how much
    /// of the bound an unconverged search gave away. Zero would mean the identity holds exactly.
    pub residual: f64,
    /// The diagonal shift that made the Cholesky complete, costing `shift · dim` of bound.
    ///
    /// Reported because a converged semidefinite optimum is singular, so this number is not an
    /// accident of the implementation — it is the price of proving definiteness by factorisation,
    /// the same gap [`crate::sdp`]'s Gershgorin floor has to nudge across.
    pub shift: f64,
}

/// The level-2 Gram matrix as a graph, so [`crate::sdp::Certificate::verify`] can prove it definite.
///
/// `crate::sdp`'s cost matrix is `C[a][b] = −J_ab/2` with a zero diagonal, so couplings
/// `J_ab = −2 Q[a][b]` reproduce `Q`'s off-diagonal **exactly** — halving and doubling are exact in
/// binary floating point — and `y = −diag Q` makes `C − Diag(y)` the matrix `Q` itself.
fn lift(dim: usize, q: &[f64]) -> Graph {
    let mut gb = GraphBuilder::new(dim);
    for a in 0..dim {
        for b in (a + 1)..dim {
            let v = q[a * dim + b];
            if v != 0.0 {
                gb.couple(a, b, -2.0 * v);
            }
        }
    }
    gb.build()
}

/// The certified bound of a Gram matrix, or why it certifies nothing.
///
/// Shared by [`certified`] and [`Certificate::verify`] so that the number reported and the number
/// re-checked come from one piece of arithmetic. Everything upstream — the projections, the
/// bisection, the shift search — is outside this function, which is the sense in which the
/// certificate stands on its own.
fn extract(l2: &Level2, e: &[f64], q: &[f64]) -> Result<(f64, f64), SosError> {
    let dim = l2.dim;
    if q.len() != dim * dim {
        return Err(SosError::Shape { got: q.len(), want: dim * dim });
    }
    if q.iter().any(|v| !v.is_finite()) {
        return Err(SosError::NotFinite);
    }
    for a in 0..dim {
        for b in (a + 1)..dim {
            if q[a * dim + b] != q[b * dim + a] {
                return Err(SosError::NotSymmetric { a, b });
            }
        }
    }
    // `Q ⪰ 0`, discharged by level 1's verifier on the lifted graph. It returns `sum_down(y)`,
    // which is a number never ABOVE `−tr Q` — the direction this bound needs, since the trace
    // enters it positively.
    let lifted = lift(dim, q);
    let y: Vec<f64> = (0..dim).map(|a| -q[a * dim + a]).collect();
    let hint = sum_down(&y);
    let psd = sdp::Certificate {
        y,
        value: hint,
        homogenised: false,
        rump_c: 0.0,
        sweeps: 0,
        rank: 0,
    };
    let minus_trace = match psd.verify(&lifted) {
        Ok(v) => v,
        Err(sdp::CertError::NotPsd) => return Err(SosError::NotPsd),
        Err(sdp::CertError::NotFinite) => return Err(SosError::NotFinite),
        Err(sdp::CertError::Shape { got, want }) => return Err(SosError::Shape { got, want }),
    };

    // How far the sum-of-squares identity misses, class by class. `q_C = Σ_{AΔB=C} Q[A][B]` is an
    // exact real sum that `f64` cannot hold, so it is BRACKETED and the wider end taken: the
    // residual is subtracted, so overstating it can only weaken the bound.
    let mut terms: Vec<Vec<f64>> = vec![Vec::new(); l2.classes()];
    for c in 0..l2.classes() {
        terms[c].reserve(l2.sizes[c] as usize + 1);
    }
    for a in 0..dim {
        for b in 0..dim {
            terms[l2.class[a * dim + b] as usize].push(-q[a * dim + b]);
        }
    }
    let mut parts = Vec::with_capacity(l2.classes());
    parts.push(minus_trace);
    let mut residual_terms = Vec::with_capacity(l2.classes());
    for c in 0..l2.classes() {
        if c == l2.empty {
            continue;
        }
        terms[c].push(e[c]);
        let lo = sum_down(&terms[c]);
        let hi = sum_up(&terms[c]);
        let miss = lo.abs().max(hi.abs());
        parts.push(-miss);
        residual_terms.push(miss);
    }
    // DOWN, because the result is a lower bound. `parts.iter().sum()` here is exactly the defect
    // `crate::round` exists for.
    Ok((sum_down(&parts), sum_up(&residual_terms)))
}

impl Certificate {
    /// Re-check this certificate against the graph, from scratch.
    ///
    /// Rebuilds the monomial basis and the energy's class vector from `g`, re-verifies that `Q` is
    /// positive definite, re-bounds the sum-of-squares residual and re-sums the result. It looks at
    /// nothing from the search.
    ///
    /// # Errors
    ///
    /// [`SosError`] when the certificate does not check out against this graph: a wrong shape, an
    /// asymmetric Gram matrix, or one that is not provably positive definite. A certificate that
    /// cannot be re-verified is not a bound.
    pub fn verify(&self, g: &Graph) -> Result<f64, SosError> {
        let l2 = Level2::new(g.n)?;
        // Everything in ENTRIES, because that is the unit `Shape` is documented in, and the three
        // ways this can be wrong -- a certificate for a different spin count, a mis-stated `dim`,
        // and a Gram matrix that is not `dim x dim` -- all land in the same place.
        if self.n != l2.n || self.dim != l2.dim || self.q.len() != l2.dim * l2.dim {
            return Err(SosError::Shape { got: self.q.len(), want: l2.dim * l2.dim });
        }
        // An empty graph has one state, the empty one, of energy zero; see `certified`.
        if g.n == 0 {
            return Ok(0.0);
        }
        let e = l2.objective(g);
        extract(&l2, &e, &self.q).map(|(v, _)| v)
    }
}

/// Project `q` onto `{Q : q_C = target_C for every class C}`, in place.
///
/// Closed form, and that is the reason the monomial classes are the right object: they **partition**
/// the matrix entries, so the constraints are orthogonal and the Euclidean projection is one
/// uniform correction per class. Symmetry survives because every class is closed under transpose.
fn project_affine(l2: &Level2, q: &mut [f64], target: &[f64]) {
    let dim = l2.dim;
    let mut sums = vec![0.0f64; l2.classes()];
    for a in 0..dim {
        for b in 0..dim {
            sums[l2.class[a * dim + b] as usize] += q[a * dim + b];
        }
    }
    let delta: Vec<f64> = (0..l2.classes())
        .map(|c| (sums[c] - target[c]) / f64::from(l2.sizes[c]))
        .collect();
    for a in 0..dim {
        for b in 0..dim {
            q[a * dim + b] -= delta[l2.class[a * dim + b] as usize];
        }
    }
}

/// Project `q` onto the positive semidefinite cone, in place: clip the spectrum at zero.
///
/// The output is written from the upper triangle down, so it is EXACTLY symmetric rather than
/// symmetric to rounding — `extract` refuses an asymmetric Gram matrix, and a reassembly that
/// wrote both triangles independently would produce one on most instances.
fn project_psd(q: &mut [f64], dim: usize) {
    let mut m = q.to_vec();
    let v = crate::linalg::jacobi_eig(&mut m, dim);
    let pos: Vec<(usize, f64)> =
        (0..dim).map(|c| (c, m[c * dim + c])).filter(|&(_, l)| l > 0.0).collect();
    for a in 0..dim {
        for b in a..dim {
            let mut acc = 0.0;
            for &(c, lam) in &pos {
                acc += lam * v[a * dim + c] * v[b * dim + c];
            }
            q[a * dim + b] = acc;
            q[b * dim + a] = acc;
        }
    }
}

/// Douglas–Rachford splitting on the two sets the certificate must lie in, at this `λ`.
///
/// `x ← x + P_A(2 P_B(x) − x) − P_B(x)`, with `B` the positive semidefinite cone and `A` the affine
/// set `{Q : q_C = e_C for C ≠ ∅, q_∅ = −λ}`. The state `x` is carried in and out — it is NOT the
/// answer; the answer is `P_B(x)`, which is what this returns and what is therefore semidefinite up
/// to rounding.
///
/// **Von Neumann's alternating projections were the first version of this function and they are not
/// good enough.** Measured on the frustrated 5-cycle at its own level-2 optimum `λ = −3`, in the
/// residual `Σ_{C≠∅}|e_C − q_C|`, which is exactly what the bound gives up:
///
/// | iterations | alternating projections | Douglas–Rachford |
/// |---|---|---|
/// | 50 | 2.5e−1 | 8.1e−4 |
/// | 200 | 1.2e−1 | 3.0e−11 |
/// | 800 | 4.5e−2 | 3.4e−12 |
/// | 3,200 | 6.8e−3 | 5.6e−15 |
///
/// Alternating projections converge linearly at a rate set by the angle between the sets, and at a
/// semidefinite optimum that angle is zero — the sets are tangent, which is the same singularity
/// that forces the diagonal shift. Douglas–Rachford is not slowed by tangency, and the difference is
/// the whole reason the module reports `−3.000000` on that instance rather than `−3.01`.
///
/// Where the sets do NOT intersect — an over-ambitious `λ` — the iterates do not converge, and that
/// is harmless and even useful: the residual settles at a positive value, [`extract`] charges it to
/// the bound, and the answer is a weaker bound rather than a wrong one. At `λ = −2.95`, past the
/// 5-cycle's level-2 optimum, it settles at `−3.0333` for every iteration count above 50.
///
/// [Lions & Mercier 1979]: P.-L. Lions and B. Mercier, "Splitting algorithms for the sum of two
/// nonlinear operators", SIAM Journal on Numerical Analysis 16(6):964–979.
fn solve(l2: &Level2, e: &[f64], lambda: f64, x: &mut [f64], iters: usize) -> Vec<f64> {
    let dim = l2.dim;
    let mut target = e.to_vec();
    // `λ = −tr Q`: the constant monomial's class collects exactly the diagonal.
    target[l2.empty] = -lambda;
    let mut pb = vec![0.0f64; dim * dim];
    let mut refl = vec![0.0f64; dim * dim];
    for _ in 0..iters {
        pb.copy_from_slice(x);
        project_psd(&mut pb, dim);
        for i in 0..dim * dim {
            refl[i] = 2.0 * pb[i] - x[i];
        }
        project_affine(l2, &mut refl, &target);
        for i in 0..dim * dim {
            x[i] += refl[i] - pb[i];
        }
    }
    pb.copy_from_slice(x);
    project_psd(&mut pb, dim);
    pb
}

/// Shift the diagonal up until the Cholesky completes, and return the certified bound.
///
/// A converged semidefinite optimum is singular and a Cholesky proves DEFINITENESS, so some shift
/// is always needed; it is grown geometrically and then bisected back, exactly as
/// [`crate::sdp::certified`] recovers its own overshoot. The shift costs `δ · dim` of bound and
/// nothing else — it touches only the diagonal, which is the constant monomial's class, so no other
/// class's residual moves.
fn certify_shifted(l2: &Level2, e: &[f64], q: &[f64]) -> Option<(Vec<f64>, f64, f64, f64)> {
    let dim = l2.dim;
    let scale = q.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1.0);
    let shifted = |d: f64| {
        let mut w = q.to_vec();
        for a in 0..dim {
            w[a * dim + a] += d;
        }
        w
    };
    let mut delta = scale * (2.0f64).powi(-50);
    let mut last_fail = 0.0f64;
    let mut best: Option<(Vec<f64>, f64, f64, f64)> = None;
    for _ in 0..80 {
        let w = shifted(delta);
        if let Ok((v, r)) = extract(l2, e, &w) {
            best = Some((w, v, r, delta));
            break;
        }
        last_fail = delta;
        delta *= 4.0;
    }
    // Nothing verified at any shift, which cannot happen for a finite matrix; there is then no
    // overshoot to bisect back and no certificate to return.
    best.as_ref()?;
    let (mut lo, mut hi) = (last_fail, delta);
    for _ in 0..8 {
        let mid = 0.5 * (lo + hi);
        let w = shifted(mid);
        if let Ok((v, r)) = extract(l2, e, &w) {
            hi = mid;
            best = Some((w, v, r, mid));
        } else {
            lo = mid;
        }
    }
    best
}

/// The level-1 certificate, embedded as a level-2 Gram matrix.
///
/// Level 1's own sum-of-squares statement is `E(s) − eᵀy = zᵀ(C − Diag(y))z` over the basis
/// `{1, s_1, …, s_n}`, which is the first `1 + n` rows of the level-2 basis. So the level-1 optimum
/// is a **feasible level-2 point**, padded with zero rows — which is the theorem "level 2 is at
/// least as tight as level 1", written as a matrix rather than asserted.
///
/// It is used as the warm start, and it is why the bisection's lower end is known feasible.
fn embed_level_one(l2: &Level2, g: &Graph, cert: &sdp::Certificate) -> Option<Vec<f64>> {
    let dim = l2.dim;
    let want = if cert.homogenised { g.n + 1 } else { g.n };
    if cert.y.len() != want || cert.y.iter().any(|v| !v.is_finite()) {
        return None;
    }
    let mut q = vec![0.0f64; dim * dim];
    // Spin `i` is level-1 row `i + off` and level-2 row `1 + i`; the level-1 gauge row, when there
    // is one, is level-2 row 0 — the constant monomial, which is what a gauge spin means.
    let off = usize::from(cert.homogenised);
    for i in 0..g.n {
        if cert.homogenised && g.h[i] != 0.0 {
            let v = -g.h[i] / 2.0;
            q[1 + i] = v;
            q[(1 + i) * dim] = v;
        }
        for k in g.offset[i]..g.offset[i + 1] {
            let j = g.nbr[k] as usize;
            let v = -g.w[k] / 2.0;
            q[(1 + i) * dim + (1 + j)] = v;
            q[(1 + j) * dim + (1 + i)] = v;
        }
    }
    if cert.homogenised {
        q[0] = -cert.y[0];
    }
    for i in 0..g.n {
        q[(1 + i) * dim + (1 + i)] = -cert.y[i + off];
    }
    Some(q)
}

/// A certified level-2 lower bound on `min_s E(s)`, with the sum-of-squares certificate that proves
/// it.
///
/// The search is a heuristic and the bound is whatever the best certificate found re-verifies to;
/// see the module documentation for why that is the only arrangement worth having.
///
/// Unlike [`crate::sdp::certified`] there is **no floor**: the number reported is the number this
/// module's own certificate re-verifies to, and splicing in a better number from elsewhere would
/// mean [`Certificate::verify`] disagreed with [`Bound::value`]. The level-1 point is instead the
/// warm start, so level 2 can only fall below level 1 by the cost of re-proving a SINGULAR matrix
/// definite — that being what level 1's zero rows make it. Measured worst over this module's
/// sweeps: `1.3e-12`, which is the order of `shift · dim` plus the summation guards, and four
/// orders below the `1e-9` those tests assert at.
///
/// # Errors
///
/// [`SosError::TooWide`] past [`MAX_SPINS`], and [`SosError::NotPsd`] in the case that no shift at
/// all made the Cholesky complete — which cannot happen for a finite matrix, and is returned rather
/// than unwrapped because a panic inside a soundness check is the worst place for one.
pub fn certified(g: &Graph, p: &Params, seed: u64) -> Result<(Bound, Certificate), SosError> {
    let l2 = Level2::new(g.n)?;
    let dim = l2.dim;
    if g.n == 0 {
        // One state, the empty one, of energy zero. There is nothing to relax, and a 1x1 Gram
        // matrix of zero is singular, so the shift search would report `−δ` for a problem whose
        // answer is exactly zero.
        let b = Bound { value: 0.0, parts: 0, method: "sos level 2: empty graph", rounds: 0, best_round: 0 };
        let c = Certificate { q: vec![0.0], dim, n: 0, value: 0.0, residual: 0.0, shift: 0.0 };
        return Ok((b, c));
    }
    let e = l2.objective(g);

    // Both ends of the bisection come from level 1: its certified bound is feasible (and is the
    // warm start), and the energy of a state rounded out of the same relaxation is an upper limit,
    // because `λ ≤ min_s E(s) ≤ E(any s)`.
    let (_, c1) = sdp::certified(g, &p.sdp, seed);
    let hi_limit = sdp::goemans_williamson(g, &p.sdp, seed, p.hyperplanes).energy;
    let start = embed_level_one(&l2, g, &c1).unwrap_or_else(|| vec![0.0; dim * dim]);

    let Some((q0, v0, r0, s0)) = certify_shifted(&l2, &e, &start) else {
        return Err(SosError::NotPsd);
    };
    let mut best = (v0, q0, r0, s0);
    let mut best_round = 0usize;
    let mut warm = start;
    let mut lo = v0;
    let mut hi = if hi_limit.is_finite() && hi_limit > lo { hi_limit } else { 0.0f64.max(lo) };

    for round in 1..=p.rounds {
        let mid = 0.5 * (lo + hi);
        let mut x = warm.clone();
        let q = solve(&l2, &e, mid, &mut x, p.iters);
        let Some((qs, v, r, sh)) = certify_shifted(&l2, &e, &q) else {
            hi = mid;
            continue;
        };
        if v > best.0 {
            best = (v, qs, r, sh);
            best_round = round;
            warm = x;
            lo = mid;
        } else {
            hi = mid;
        }
    }

    let (value, q, residual, shift) = best;
    let cert = Certificate { q, dim, n: g.n, value, residual, shift };
    let b = Bound {
        value,
        parts: 1,
        method: "sos: Lasserre level 2, sum-of-squares Gram matrix verified positive definite",
        rounds: p.rounds,
        best_round,
    };
    Ok((b, cert))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::GraphBuilder;
    use crate::rng::Pcg;

    /// The frustrated odd cycle: every edge antiferromagnetic, an odd loop, so one edge must give.
    fn frustrated_cycle(n: usize) -> Graph {
        let mut gb = GraphBuilder::new(n);
        for i in 0..n {
            gb.couple(i, (i + 1) % n, -1.0);
        }
        gb.build()
    }

    /// `K_n` with every edge antiferromagnetic.
    fn complete(n: usize) -> Graph {
        let mut gb = GraphBuilder::new(n);
        for i in 0..n {
            for j in (i + 1)..n {
                gb.couple(i, j, -1.0);
            }
        }
        gb.build()
    }

    fn random_graph(n: usize, p: f64, seed: u64, fields: bool) -> Graph {
        let mut rng = Pcg::new(seed, 0x50_5E);
        let mut gb = GraphBuilder::new(n);
        for i in 0..n {
            if fields {
                gb.bias(i, rng.f64() * 2.0 - 1.0);
            }
            for j in (i + 1)..n {
                if rng.f64() < p {
                    gb.couple(i, j, rng.f64() * 2.0 - 1.0);
                }
            }
        }
        gb.build()
    }

    /// ORACLE: exhaustive enumeration of all `2^n` states, which knows nothing about relaxations.
    fn brute_min(g: &Graph) -> f64 {
        (0..(1u32 << g.n))
            .map(|m| {
                let s: Vec<i8> = (0..g.n).map(|i| if m >> i & 1 == 1 { 1 } else { -1 }).collect();
                g.energy(&s)
            })
            .fold(f64::INFINITY, f64::min)
    }

    /// Deliberately under-powered, for the sweeps: the point of those is soundness across many
    /// instances, and soundness is the one thing that does not depend on the search.
    fn quick() -> Params {
        Params {
            rounds: 8,
            iters: 120,
            hyperplanes: 16,
            sdp: sdp::Params { sweeps: 60, rank: None, lanczos: 24 },
        }
    }

    /// ORACLE: [`Graph::energy`], over every state of a small instance.
    ///
    /// The class map is the whole relaxation, and it is the kind of index arithmetic that goes wrong
    /// silently. This pins it to something that knows nothing about monomials: the energy written as
    /// a vector over classes, dotted with a state's moments, must be that state's energy — on all
    /// `2^7` states, with and without fields.
    #[test]
    fn the_class_vector_reproduces_graph_energy_on_every_state() {
        for seed in 0..8u64 {
            let g = random_graph(7, 0.55, seed, seed % 2 == 0);
            let l2 = Level2::new(g.n).unwrap();
            let e = l2.objective(&g);
            for m in 0..(1u32 << g.n) {
                let s: Vec<i8> = (0..g.n).map(|i| if m >> i & 1 == 1 { 1 } else { -1 }).collect();
                let y = l2.moments(&s);
                let got: f64 = (0..l2.classes()).map(|c| e[c] * y[c]).sum();
                assert!(
                    (got - g.energy(&s)).abs() < 1e-12,
                    "seed {seed} state {m}: the monomial basis gave {got}, the energy is {}",
                    g.energy(&s)
                );
            }
        }
    }

    /// ORACLE: the outer product `z zᵀ`, formed without any reference to the class map.
    ///
    /// A real state's moment matrix must be EXACTLY `z zᵀ` — rank one, entry for entry, with no
    /// tolerance, because both sides are products of `±1`. That is the statement "every `±1` state
    /// is feasible", which is what makes the relaxation's minimum a LOWER bound, and it is false the
    /// moment `A Δ B` is computed as anything other than a symmetric difference.
    #[test]
    fn a_states_moment_matrix_is_exactly_the_rank_one_outer_product() {
        let n = 6;
        let l2 = Level2::new(n).unwrap();
        let mut rng = Pcg::new(11, 0x5E_11);
        for _ in 0..40 {
            let s: Vec<i8> = (0..n).map(|_| rng.spin(0.5)).collect();
            let z: Vec<f64> = l2
                .basis()
                .iter()
                .map(|&m| {
                    let mut v = 1.0f64;
                    for i in 0..n {
                        if m >> i & 1 == 1 {
                            v *= f64::from(s[i]);
                        }
                    }
                    v
                })
                .collect();
            let mm = l2.moment_matrix(&l2.moments(&s));
            for a in 0..l2.dim() {
                for b in 0..l2.dim() {
                    assert_eq!(
                        mm[a * l2.dim() + b],
                        z[a] * z[b],
                        "({a},{b}) of a state's moment matrix is not z z'"
                    );
                }
            }
        }
    }

    /// ORACLE: exhaustive enumeration. SOUNDNESS, which is the only property that matters.
    ///
    /// With and without fields, so the level-1 homogenisation path — whose gauge spin lands on the
    /// CONSTANT monomial here — is exercised rather than assumed. The certificate is re-verified on
    /// every instance too, and must reproduce the reported number to the last bit but one: `verify`
    /// and `certified` share their arithmetic deliberately, so any difference at all would mean the
    /// certificate is not the thing that was measured.
    #[test]
    fn the_bound_never_exceeds_the_true_minimum_by_enumeration() {
        for seed in 0..10u64 {
            for fields in [false, true] {
                let g = random_graph(8, 0.5, seed + 300, fields);
                let truth = brute_min(&g);
                let (b, cert) = certified(&g, &quick(), seed).unwrap();
                assert!(
                    b.value <= truth + 1e-9,
                    "seed {seed} fields={fields}: level 2 gave {} above the true minimum {truth}",
                    b.value
                );
                let again = cert.verify(&g).expect("its own certificate must re-verify");
                assert_eq!(again, b.value, "verify {again} against a reported {}", b.value);
            }
        }
    }

    /// ORACLES: [`crate::sdp::certified`] and exhaustive enumeration, instance by instance.
    ///
    /// Level 2 CONTAINS level 1 — the level-1 Gram matrix is a level-2 one with zero rows, which is
    /// what `embed_level_one` writes down — so level 2 may never be looser. There is no floor in
    /// this module, so nothing but the search working makes that pass; the `1e-9` is the price of
    /// re-proving a SINGULAR matrix definite, which is what those zero rows make it, and the
    /// measured worst over this sweep is `1.3e-12`.
    ///
    /// **"Never looser" is satisfied by returning level 1 verbatim, so it is not the assertion that
    /// gives this test its teeth.** Enumeration supplies the true optimum, and every instance where
    /// level 1 has a real gap must have most of that gap CLOSED. Measured on this sweep, with the
    /// deliberately under-powered parameters: 8 of the 14 instances have a level-1 gap above 1e-3,
    /// and level 2 closes between 99.30% and 100.00% of each. The floor asserted is 90%, and an
    /// implementation that returned level 1 closes 0%.
    #[test]
    fn level_two_closes_the_level_one_gap_it_is_supposed_to_close() {
        let p = quick();
        let mut loose = 0usize;
        for seed in 0..14u64 {
            let g = random_graph(7, 0.55, seed + 700, seed % 3 == 0);
            let truth = brute_min(&g);
            let (b1, _) = sdp::certified(&g, &p.sdp, seed);
            let (b2, _) = certified(&g, &p, seed).unwrap();
            assert!(
                b2.value >= b1.value - 1e-9,
                "seed {seed}: level 2 gave {} below level 1's {}",
                b2.value,
                b1.value
            );
            assert!(
                b2.value <= truth + 1e-9,
                "seed {seed}: level 2 gave {} above the true minimum {truth}",
                b2.value
            );
            let gap = truth - b1.value;
            if gap > 1e-3 {
                loose += 1;
                let closed = (b2.value - b1.value) / gap;
                assert!(
                    closed > 0.90,
                    "seed {seed}: level 1 is loose by {gap:.4} and level 2 closed only \
                     {:.2}% of it",
                    closed * 100.0
                );
            }
        }
        assert!(
            loose >= 6,
            "only {loose} of 14 instances had a level-1 gap to close, so this sweep is no longer \
             testing what it says it tests"
        );
    }

    /// ORACLES: the closed form `−n cos(π/n)` for the level-1 relaxation of an odd cycle, and the
    /// exact ground energy `−(n−2)` by enumeration.
    ///
    /// **THE NAMED INSTANCES WHERE LEVEL 1 IS KNOWN TO BE LOOSE, AND THE POINT OF THE MODULE.** The
    /// max-cut relaxation of `C_n` puts the `n` unit vectors at `2π(n−1)/(2n)` and reports
    /// `n cos(π(n−1)/n) = −n cos(π/n)`: `−1.5` against `−1` on the triangle, `−4.04508…` against
    /// `−3` on the pentagon. No amount of sweeping closes that, because it is the relaxation and not
    /// the solver — [`crate::sdp`]'s own documentation records twenty times the work moving nothing.
    ///
    /// The degree-4 moment matrix implies the odd-cycle inequalities, so level 2 must be STRICTLY
    /// tighter, and in fact exact. A module that quietly returned level 1 fails on the very first
    /// assertion of each row.
    #[test]
    fn on_frustrated_odd_cycles_level_two_is_exact_where_level_one_is_loose() {
        for n in [3usize, 5, 7] {
            let g = frustrated_cycle(n);
            let truth = brute_min(&g);
            assert_eq!(truth, -((n - 2) as f64), "C{n}: one of n edges must be satisfied");

            let level_one_closed = -(n as f64) * (core::f64::consts::PI / n as f64).cos();
            let (b1, _) = sdp::certified(&g, &Params::default().sdp, 7);
            assert!(
                (b1.value - level_one_closed).abs() < 1e-6,
                "C{n}: level 1 gave {} against the closed form {level_one_closed}",
                b1.value
            );

            let (b2, cert) = certified(&g, &Params::default(), 7).unwrap();
            assert!(
                b2.value <= truth + 1e-9,
                "C{n}: level 2 gave {} above the optimum {truth}",
                b2.value
            );
            // STRICTLY tighter, by most of the gap rather than by an epsilon.
            assert!(
                b2.value > level_one_closed + 0.5 * (truth - level_one_closed),
                "C{n}: level 2 gave {}, no better than half way from the closed-form level-1 value \
                 {level_one_closed} to the optimum {truth}",
                b2.value
            );
            // And it is not merely better: on an odd cycle the degree-4 moment matrix is exact, and
            // what is left is the bisection's own resolution.
            assert!(
                (b2.value - truth).abs() < 1e-3,
                "C{n}: level 2 gave {} against the exact optimum {truth}",
                b2.value
            );
            assert_eq!(cert.verify(&g).unwrap(), b2.value);
        }
    }

    /// ORACLE: the closed form `−n/2`, and the exact optimum by enumeration.
    ///
    /// **THE INSTANCE WHERE LEVEL 2 BUYS NOTHING, asserted as such.** For `K_n` antiferromagnetic,
    /// `E(s) = ((Σs)² − n)/2 ≥ −n/2`, with equality whenever the vectors sum to zero — so the
    /// level-1 relaxation is exactly `−n/2` and its optimal pair moments are `−1/(n−1)`. Every
    /// triangle inequality reads `3·(−1/(n−1)) ≥ −1`, which is slack for `n ≥ 4`, so the degree-4
    /// moment matrix has nothing to add and the level-2 optimum is `−n/2` as well. Measured: it is,
    /// to 1.3e-12, while the true optima are `−2` and `−3`.
    ///
    /// A level-2 bound that came out ABOVE `−n/2` here would not be a tighter relaxation, it would
    /// be an unsound one; a test that only ever checked instances where level 2 wins could not tell
    /// those apart.
    #[test]
    fn on_complete_graphs_level_two_ties_level_one_at_the_closed_form_and_claims_no_more() {
        for n in [5usize, 7] {
            let g = complete(n);
            let closed = -(n as f64) / 2.0;
            let truth = brute_min(&g);
            assert_eq!(truth, -((n as f64) - 1.0) / 2.0, "K{n}: the best split is as even as it gets");
            let (b1, _) = sdp::certified(&g, &Params::default().sdp, 3);
            assert!((b1.value - closed).abs() < 1e-9, "K{n}: level 1 gave {}", b1.value);
            let (b2, _) = certified(&g, &Params::default(), 3).unwrap();
            assert!(
                (b2.value - closed).abs() < 1e-9,
                "K{n}: level 2 gave {}, and the level-2 optimum here is the closed form {closed}",
                b2.value
            );
            assert!(b2.value <= truth + 1e-9);
        }
    }

    /// A tampered certificate must be refused, or the artefact means nothing.
    ///
    /// Lowering the diagonal raises `−tr Q` and so makes the bound look better, and it is exactly
    /// what breaks positive definiteness. Breaking the symmetry makes `zᵀQz` a claim about a matrix
    /// nobody wrote down.
    #[test]
    fn a_tampered_certificate_is_refused() {
        let g = random_graph(6, 0.6, 4, true);
        let (_, cert) = certified(&g, &quick(), 4).unwrap();
        assert!(cert.verify(&g).unwrap().is_finite());

        let mut inflated = cert.clone();
        for a in 0..inflated.dim {
            inflated.q[a * inflated.dim + a] -= 1.0;
        }
        assert_eq!(inflated.verify(&g), Err(SosError::NotPsd), "a lowered diagonal must not verify");

        let mut lopsided = cert.clone();
        lopsided.q[1] += 1e-3;
        assert!(matches!(lopsided.verify(&g), Err(SosError::NotSymmetric { .. })));

        let mut wrong_shape = cert.clone();
        wrong_shape.q.push(0.0);
        assert!(matches!(wrong_shape.verify(&g), Err(SosError::Shape { .. })));

        // And a certificate for one graph must not certify another.
        let other = random_graph(6, 0.6, 5, true);
        let cross = cert.verify(&other).expect("the shape still fits, so it is a number");
        assert!(cross <= brute_min(&other) + 1e-9, "a re-aimed certificate is still only a bound");
    }

    /// The module's own licence, checked: everything upstream of the positive-definiteness proof is
    /// a heuristic, so a deliberately terrible search must still be SOUND.
    #[test]
    fn a_bad_search_loosens_the_bound_without_invalidating_it() {
        let p = Params {
            rounds: 1,
            iters: 1,
            hyperplanes: 1,
            sdp: sdp::Params { sweeps: 1, rank: Some(1), lanczos: 2 },
        };
        for seed in 0..8u64 {
            let g = random_graph(7, 0.5, seed + 900, seed % 2 == 0);
            let truth = brute_min(&g);
            let (b, cert) = certified(&g, &p, seed).unwrap();
            assert!(b.value <= truth + 1e-9, "seed {seed}: {} > {truth}", b.value);
            assert!(cert.verify(&g).is_ok(), "seed {seed}: the certificate must still verify");
        }
    }

    /// The index arithmetic, against counts computed from the combinatorics by hand.
    #[test]
    fn the_basis_and_its_classes_are_the_sizes_the_hierarchy_says() {
        for n in 0..8usize {
            let l2 = Level2::new(n).unwrap();
            assert_eq!(l2.dim(), 1 + n + n * n.saturating_sub(1) / 2);
            // One class per square-free monomial of degree at most four: the symmetric difference of
            // two degree-≤2 sets has size at most four, and every such set arises.
            let want: usize = (0..=4).map(|k| binomial(n, k)).sum();
            assert_eq!(l2.classes(), want, "n={n}");
            // The classes PARTITION the entries, which is what makes the affine projection exact.
            let total: u32 = (0..l2.classes()).map(|c| l2.sizes[c]).sum();
            assert_eq!(total as usize, l2.dim() * l2.dim(), "n={n}");
            assert_eq!(l2.monomial(l2.constant_class()), 0);
            // The constant class is the diagonal and nothing else: `λ = −tr Q` depends on it.
            assert_eq!(l2.sizes[l2.constant_class()] as usize, l2.dim(), "n={n}");
        }
        assert_eq!(
            Level2::new(MAX_SPINS + 1).err(),
            Some(SosError::TooWide { n: MAX_SPINS + 1, max: MAX_SPINS })
        );
        assert!(certified(&complete(MAX_SPINS + 1), &quick(), 0).is_err());
    }

    fn binomial(n: usize, k: usize) -> usize {
        if k > n {
            return 0;
        }
        let mut v = 1usize;
        for i in 0..k {
            v = v * (n - i) / (i + 1);
        }
        v
    }

    /// The degenerate sizes return rather than panicking, and are TIGHT rather than merely sound.
    ///
    /// A one-spin graph is the smallest thing the two paths differ on: unbiased, the level-1 point
    /// is not homogenised and the constant monomial's row starts empty; biased, level 1 prepends a
    /// gauge spin that lands on that same row. Both have a known answer — `0` and `−|h|` — so this
    /// checks the number, not just the absence of a panic.
    #[test]
    fn the_degenerate_sizes_return_rather_than_panicking() {
        let g = GraphBuilder::new(0).build();
        let (b, c) = certified(&g, &quick(), 1).unwrap();
        assert_eq!(b.value, 0.0);
        assert_eq!(c.verify(&g).unwrap(), 0.0);

        let lone = GraphBuilder::new(1).build();
        let (b, c) = certified(&lone, &quick(), 1).unwrap();
        assert!(b.value <= 0.0 && b.value > -1e-9, "one unbiased spin has energy 0: {}", b.value);
        assert!(c.verify(&lone).is_ok());

        let mut gb = GraphBuilder::new(1);
        gb.bias(0, 0.75);
        let biased = gb.build();
        let (b, c) = certified(&biased, &quick(), 1).unwrap();
        assert!(
            (b.value + 0.75).abs() < 1e-9,
            "one spin in a field of 0.75 has ground energy -0.75, not {}",
            b.value
        );
        assert!(c.verify(&biased).is_ok());

        // Two spins and one coupling, which has a closed form too.
        let mut gb = GraphBuilder::new(2);
        gb.couple(0, 1, -1.25);
        let pair = gb.build();
        let (b, _) = certified(&pair, &quick(), 1).unwrap();
        assert!((b.value + 1.25).abs() < 1e-9, "an antiferromagnetic pair is -1.25, not {}", b.value);
    }
}

