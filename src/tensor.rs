//! General tensor networks: the contraction engine the exact solver already was, said out loud.
//!
//! [`crate::exact::Elimination`] does bucket elimination over a pairwise spin graph, bounded by the
//! induced width of its order. Tensor-network contraction is the same computation in a different
//! vocabulary — Markov and Shi (2008) is the statement that contracting a network is polynomial in
//! its size and exponential in its treewidth, which is `exact.rs`'s `2^width` written in the other
//! field's notation. So this crate already had a tensor-network engine and did not call it one.
//!
//! What it did not have is the generality that makes the vocabulary worth adopting:
//!
//! | | `exact::Elimination` | here |
//! |---|---|---|
//! | factor arity | pairwise only (it takes a [`crate::graph::Graph`]) | any rank |
//! | index dimension | 2 (a spin) | any |
//! | index multiplicity | one variable per bucket | any number of tensors may carry an index |
//! | arithmetic | two, hardcoded (min-sum, log-sum-exp) | any commutative [`Semiring`] |
//! | output | `log Z`, ground state, marginals | any tensor, with indices left **open** |
//! | ordering | min-fill and nested dissection | greedy-cost, or the caller's own |
//! | prices the order first | `Elimination::width` | [`Network::plan`] |
//!
//! # One contraction, four questions
//!
//! The arithmetic row is the one that matters most. Aji and McEliece's Generalized Distributive Law
//! (2000) says elimination is correct over **any** commutative semiring, so the same network
//! contracted four ways answers four questions — not four algorithms, one algorithm and a type
//! parameter:
//!
//! ```text
//!   Network::<SumProduct>::from_ising(&g, beta)  ->  Z
//!   Network::<Tropical>::from_ising(&g)          ->  the ground energy
//!   Network::<MinCount>::from_ising(&g)          ->  the ground energy AND its degeneracy
//!   Network::<Counting>                          ->  how many configurations
//! ```
//!
//! [`MinCount`] is the one the crate could not do at all: `crate::samples::SampleSet` reports
//! "evidence of degeneracy, not a count of it", and enumeration stops at 26 spins. Counting falls
//! out of carrying a count beside the energy and adding counts on a tie — degeneracy is exactly what
//! the tropical semiring discards when it takes a minimum.
//!
//! This is also where the quantum-simulation literature lives: a quantum circuit is a tensor
//! network over `(ℂ, +, ×)`, and contracting one is how the classical rebuttals to the Sycamore
//! supremacy claim were computed. The schedule would be the same; only the scalar differs.
//!
//! # What this is not
//!
//! It is not a quantum simulator, and this module makes no quantum claim. The semirings shipped here
//! are real-valued and integer-valued, so it contracts probability, energy and counting networks and
//! **not amplitudes** — a complex semiring is a small addition and the claim that would come with it
//! is not, so it is absent rather than half-made. It is also **exact**: there is no bond-dimension
//! truncation, so nothing here approximates and nothing here needs an error bar.
//!
//! # Semantics, stated once
//!
//! A network is a set of tensors. Contracting it means: multiply them all together and **sum over
//! every index that is not declared open**. An index carried by three tensors is summed once, after
//! the last of them has been absorbed — which is exactly what bucket elimination does, and is why
//! no special hyperedge case is needed. Indices declared open with [`Network::open`] survive into
//! the result, which is how a marginal rather than a scalar comes out.
//!
//! # The oracle
//!
//! Checked against one the crate already owns. A pairwise Ising model at inverse temperature `beta`
//! is a tensor network whose contraction is `Z`, so [`Network::from_ising`] followed by
//! [`Network::contract`] must agree with [`crate::exact::Elimination::log_partition`] on the same
//! model — two implementations of one quantity sharing no code, one written as elimination over
//! spins and one as contraction over indices. Agreement is evidence about both; a mismatch names
//! one of them wrong without saying which, which is more than either reports alone.

use std::collections::{BTreeMap, BTreeSet};

use crate::graph::Graph;

/// An index's identity. Two tensors share an index exactly when they carry the same `Index`.
///
/// A plain integer rather than a string: an index is a wire between tensors, and giving it a name
/// invites two different wires to be spelled the same way by accident. Callers that need names keep
/// their own map.
pub type Index = u32;

/// The arithmetic a contraction is performed in.
///
/// # One contraction, four questions
///
/// Aji and McEliece's Generalized Distributive Law (2000) is the statement that variable
/// elimination — and therefore tensor contraction — is correct over **any commutative semiring**.
/// The elimination schedule is a combinatorial object over the graph and is blind to the scalar;
/// only the arithmetic changes. So the same network, contracted four ways, answers four questions:
///
/// | semiring | `⊕` | `⊗` | what the contraction is |
/// |---|---|---|---|
/// | [`SumProduct`] | `+` | `×` | the partition function `Z` |
/// | [`Tropical`] | `min` | `+` | the ground energy |
/// | [`Counting`] | `+` | `×` over integers | how many configurations |
/// | [`MinCount`] | min-and-tie-add | `+`, `×` | the ground energy **and its degeneracy** |
///
/// That is not four algorithms. It is one algorithm and a type parameter, and it is the single most
/// load-bearing fact in the correspondence between thermodynamic and quantum computing: the two
/// fields differ in the scalar their sum is taken over, not in the sum.
///
/// # What a semiring has to satisfy, and what breaks if it does not
///
/// `⊕` and `⊗` must both be associative and commutative, `⊗` must distribute over `⊕`, `zero` must
/// be the identity for `⊕` and annihilate under `⊗`, and `one` must be the identity for `⊗`.
/// Distributivity is the one that matters: it is exactly what licenses pulling a factor out of a
/// sum, which is the whole of what elimination does. A structure missing it will contract to a
/// number, and the number will be wrong.
pub trait Semiring {
    /// The scalar this arithmetic is over.
    type Elem: Copy + PartialEq + core::fmt::Debug;
    /// A name, for error messages and for a reader deciding what a result means.
    const NAME: &'static str;
    /// The identity for [`Semiring::add`], and the annihilator for [`Semiring::mul`].
    fn zero() -> Self::Elem;
    /// The identity for [`Semiring::mul`]. What an empty network contracts to.
    fn one() -> Self::Elem;
    /// `⊕` — how the results of summed-over values combine.
    fn add(a: Self::Elem, b: Self::Elem) -> Self::Elem;
    /// `⊗` — how factors combine.
    fn mul(a: Self::Elem, b: Self::Elem) -> Self::Elem;
}

/// `(ℝ, +, ×)` — the contraction is the partition function `Z`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SumProduct;

impl Semiring for SumProduct {
    type Elem = f64;
    const NAME: &'static str = "sum-product";
    fn zero() -> f64 {
        0.0
    }
    fn one() -> f64 {
        1.0
    }
    fn add(a: f64, b: f64) -> f64 {
        a + b
    }
    fn mul(a: f64, b: f64) -> f64 {
        a * b
    }
}

/// `(ℝ ∪ {∞}, min, +)` — the contraction is the ground energy.
///
/// The tropical semiring, and the zero-temperature limit of [`SumProduct`]: `−(1/β) ln Z → E₀` as
/// `β → ∞`, which is Maslov dequantization. Tensors carry **energies** rather than Boltzmann
/// weights, so `Network::<Tropical>::from_ising` takes no temperature.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tropical;

impl Semiring for Tropical {
    type Elem = f64;
    const NAME: &'static str = "tropical (min, +)";
    fn zero() -> f64 {
        f64::INFINITY
    }
    fn one() -> f64 {
        0.0
    }
    fn add(a: f64, b: f64) -> f64 {
        a.min(b)
    }
    fn mul(a: f64, b: f64) -> f64 {
        a + b
    }
}

/// `(ℕ, +, ×)` — the contraction counts configurations.
///
/// `u128` rather than a float, because a count is an integer and a count that has silently become
/// approximate is worse than one that overflows loudly. It saturates rather than wrapping: a
/// saturated count is visibly `u128::MAX` instead of a plausible small number.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Counting;

impl Semiring for Counting {
    type Elem = u128;
    const NAME: &'static str = "counting";
    fn zero() -> u128 {
        0
    }
    fn one() -> u128 {
        1
    }
    fn add(a: u128, b: u128) -> u128 {
        a.saturating_add(b)
    }
    fn mul(a: u128, b: u128) -> u128 {
        a.saturating_mul(b)
    }
}

/// The ground energy **and how many states attain it**, in one contraction.
///
/// An element is `(energy, count)`. `⊗` adds energies and multiplies counts — two independent
/// sub-configurations combine into one, at the sum of their energies. `⊕` takes the lower energy,
/// and on a **tie** keeps the energy and adds the counts, which is the whole trick: degeneracy is
/// what the tropical semiring throws away when it takes a minimum, and carrying the count alongside
/// is what recovers it.
///
/// This is the capability `crate::samples::SampleSet` says it does not have — "evidence of
/// degeneracy, not a count of it" — obtained here without a second pass and without the
/// two-temperature extrapolation [`crate::exact::Elimination::ground_degeneracy`] uses.
///
/// # The tie tolerance is the whole correctness question
///
/// Two energies that differ in the last bits are the same energy physically and different energies
/// to `==`, and a comparison that gets it wrong either misses degenerate states or merges distinct
/// ones. `TIE` is relative and is applied to the larger magnitude, so it means the same thing at
/// every scale. Integer or `±J` couplings — the family this is normally run on — have exactly
/// representable energies and are unaffected by it either way.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MinCount;

impl MinCount {
    /// Energies within this relative distance are one energy for the purpose of counting.
    pub const TIE: f64 = 1e-9;
}

impl Semiring for MinCount {
    type Elem = (f64, u128);
    const NAME: &'static str = "min-count";
    fn zero() -> (f64, u128) {
        (f64::INFINITY, 0)
    }
    fn one() -> (f64, u128) {
        (0.0, 1)
    }
    fn add(a: (f64, u128), b: (f64, u128)) -> (f64, u128) {
        let (ea, ca) = a;
        let (eb, cb) = b;
        // `zero` is (INF, 0) and must be a true identity, so a count of zero never contributes
        // however its energy compares.
        if ca == 0 {
            return b;
        }
        if cb == 0 {
            return a;
        }
        let tol = MinCount::TIE * ea.abs().max(eb.abs()).max(1.0);
        if (ea - eb).abs() <= tol {
            (ea.min(eb), ca.saturating_add(cb))
        } else if ea < eb {
            a
        } else {
            b
        }
    }
    fn mul(a: (f64, u128), b: (f64, u128)) -> (f64, u128) {
        (a.0 + b.0, a.1.saturating_mul(b.1))
    }
}

/// One tensor: a dense array over its indices, row-major with the FIRST index slowest.
///
/// The layout is stated because it is load-bearing — [`Tensor::at`] and every contraction below
/// depend on it, and a reader checking this module against another implementation needs to know
/// which convention is in force rather than inferring it from a loop.
pub struct Tensor<S: Semiring = SumProduct> {
    idx: Vec<Index>,
    dims: Vec<usize>,
    data: Vec<S::Elem>,
}

// Derived `Clone`/`Debug`/`PartialEq` would demand `S: Clone` and so on, but `S` is a marker with
// no data in it -- the bound belongs on `S::Elem`, which the trait already requires. Written out so
// a semiring can be a unit struct without deriving anything.
impl<S: Semiring> Clone for Tensor<S> {
    fn clone(&self) -> Self {
        Tensor { idx: self.idx.clone(), dims: self.dims.clone(), data: self.data.clone() }
    }
}

impl<S: Semiring> core::fmt::Debug for Tensor<S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Tensor")
            .field("semiring", &S::NAME)
            .field("idx", &self.idx)
            .field("dims", &self.dims)
            .field("data", &self.data)
            .finish()
    }
}

impl<S: Semiring> PartialEq for Tensor<S> {
    fn eq(&self, other: &Self) -> bool {
        self.idx == other.idx && self.dims == other.dims && self.data == other.data
    }
}

impl<S: Semiring> Clone for Network<S> {
    fn clone(&self) -> Self {
        Network { tensors: self.tensors.clone(), open: self.open.clone() }
    }
}

impl<S: Semiring> core::fmt::Debug for Network<S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Network")
            .field("semiring", &S::NAME)
            .field("tensors", &self.tensors.len())
            .field("open", &self.open)
            .finish()
    }
}

impl<S: Semiring> Default for Network<S> {
    fn default() -> Self {
        Network::new()
    }
}

/// Why a tensor could not be built, or why two disagree.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Malformed {
    /// The data length does not equal the product of the dimensions.
    Size {
        /// Entries supplied.
        got: usize,
        /// Entries the shape requires.
        want: usize,
    },
    /// The same index appears twice on one tensor.
    ///
    /// A repeated index means a trace, which is a different operation from a contraction and is not
    /// silently one: accepting it would make `T[i,i]` and `T[i,j]` behave differently for reasons
    /// the caller never wrote down.
    RepeatedIndex(Index),
    /// An index appears on two tensors with different dimensions.
    DimensionMismatch {
        /// The index in question.
        index: Index,
        /// One tensor's dimension for it.
        a: usize,
        /// The other's.
        b: usize,
    },
    /// A dimension of zero, which makes the tensor empty and every contraction through it zero.
    ZeroDimension(Index),
}

impl core::fmt::Display for Malformed {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Malformed::Size { got, want } => {
                write!(f, "this shape holds {want} entries and {got} were supplied")
            }
            Malformed::RepeatedIndex(i) => write!(
                f,
                "index {i} appears twice on one tensor, which is a trace rather than a \
                 contraction; sum it explicitly if that is what you meant"
            ),
            Malformed::DimensionMismatch { index, a, b } => write!(
                f,
                "index {index} is dimension {a} on one tensor and {b} on another; an index is a \
                 wire and both ends must be the same width"
            ),
            Malformed::ZeroDimension(i) => {
                write!(f, "index {i} has dimension zero, so every contraction through it is empty")
            }
        }
    }
}

impl core::error::Error for Malformed {}

impl<S: Semiring> Tensor<S> {
    /// A tensor over `idx` with dimensions `dims` and entries `data`, row-major, first index
    /// slowest.
    ///
    /// # Errors
    ///
    /// [`Malformed`] when the data length disagrees with the shape, an index repeats, or a
    /// dimension is zero.
    pub fn new(
        idx: Vec<Index>,
        dims: Vec<usize>,
        data: Vec<S::Elem>,
    ) -> Result<Tensor<S>, Malformed> {
        if idx.len() != dims.len() {
            return Err(Malformed::Size { got: idx.len(), want: dims.len() });
        }
        let mut seen = BTreeSet::new();
        for &i in &idx {
            if !seen.insert(i) {
                return Err(Malformed::RepeatedIndex(i));
            }
        }
        for (k, &d) in dims.iter().enumerate() {
            if d == 0 {
                return Err(Malformed::ZeroDimension(idx[k]));
            }
        }
        let want: usize = dims.iter().product();
        if data.len() != want {
            return Err(Malformed::Size { got: data.len(), want });
        }
        Ok(Tensor { idx, dims, data })
    }

    /// A rank-0 tensor holding one number. What a fully contracted network reduces to.
    #[must_use]
    pub fn scalar(v: S::Elem) -> Tensor<S> {
        Tensor { idx: Vec::new(), dims: Vec::new(), data: vec![v] }
    }

    /// Its indices, in layout order.
    #[must_use]
    pub fn indices(&self) -> &[Index] {
        &self.idx
    }

    /// Its dimensions, aligned with [`Tensor::indices`].
    #[must_use]
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    /// Entries held, which is the product of the dimensions.
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether it holds no entries. A well-formed tensor never does — rank 0 holds one.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// The single entry of a rank-0 tensor, or `None` if it has indices left.
    #[must_use]
    pub fn value(&self) -> Option<S::Elem> {
        (self.idx.is_empty()).then(|| self.data[0])
    }

    /// The entry at `pos`, one coordinate per index in layout order.
    #[must_use]
    pub fn at(&self, pos: &[usize]) -> Option<S::Elem> {
        if pos.len() != self.idx.len() {
            return None;
        }
        let mut flat = 0usize;
        for (k, &p) in pos.iter().enumerate() {
            if p >= self.dims[k] {
                return None;
            }
            flat = flat * self.dims[k] + p;
        }
        Some(self.data[flat])
    }

    /// Raw entries in layout order, for a caller that knows the convention.
    #[must_use]
    pub fn data(&self) -> &[S::Elem] {
        &self.data
    }

    /// Sum this tensor over every index not in `keep`.
    #[must_use]
    fn marginalise(&self, keep: &BTreeSet<Index>) -> Tensor<S> {
        let out_pos: Vec<usize> =
            (0..self.idx.len()).filter(|&k| keep.contains(&self.idx[k])).collect();
        if out_pos.len() == self.idx.len() {
            return self.clone();
        }
        let out_idx: Vec<Index> = out_pos.iter().map(|&k| self.idx[k]).collect();
        let out_dims: Vec<usize> = out_pos.iter().map(|&k| self.dims[k]).collect();
        let out_len: usize = out_dims.iter().product();
        let mut data = vec![S::zero(); out_len];

        let mut pos = vec![0usize; self.idx.len()];
        for flat in 0..self.data.len() {
            let mut rem = flat;
            for k in (0..self.idx.len()).rev() {
                pos[k] = rem % self.dims[k];
                rem /= self.dims[k];
            }
            let mut o = 0usize;
            for (n, &k) in out_pos.iter().enumerate() {
                o = o * out_dims[n] + pos[k];
            }
            data[o] = S::add(data[o], self.data[flat]);
        }
        Tensor { idx: out_idx, dims: out_dims, data }
    }
}

/// A set of tensors, contracted by multiplying them and summing every index not left open.
pub struct Network<S: Semiring = SumProduct> {
    tensors: Vec<Tensor<S>>,
    open: BTreeSet<Index>,
}

/// Why a network could not be contracted.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Uncontractable {
    /// Two tensors disagree about an index's width, or one is malformed.
    Malformed(Malformed),
    /// The plan's largest intermediate exceeds the budget.
    ///
    /// The analogue of [`crate::exact::TooWide`], refused for the same reason: cost is exponential
    /// in the width of the widest intermediate, so a caller is better told the number than left
    /// waiting for it.
    TooWide {
        /// Entries in the largest intermediate this order would build.
        entries: u128,
        /// The cap.
        max: u128,
    },
}

impl core::fmt::Display for Uncontractable {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Uncontractable::Malformed(m) => write!(f, "{m}"),
            Uncontractable::TooWide { entries, max } => write!(
                f,
                "the largest intermediate this order builds holds {entries} entries, over the cap \
                 of {max}; a different order may be cheaper -- see Network::plan"
            ),
        }
    }
}

impl core::error::Error for Uncontractable {}

impl From<Malformed> for Uncontractable {
    fn from(m: Malformed) -> Self {
        Uncontractable::Malformed(m)
    }
}

/// What a contraction order will cost, computed **before** running it.
///
/// The crate's idiom: [`crate::exact::Elimination::width`] reports the price up front so a caller
/// can decide before waiting, and a plan is that for a network. `peak_entries` is what this crate's
/// `2^width` was always really about — the widest intermediate — reported in entries rather than as
/// a width because a network's indices need not be binary.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Plan {
    order: Vec<(usize, usize)>,
    /// Entries in the largest intermediate the order builds.
    pub peak_entries: u128,
    /// Multiply-accumulates the order performs.
    pub flops: u128,
}

impl Plan {
    /// The contraction steps, as `(i, j)` positions into the working list at each step.
    #[must_use]
    pub fn steps(&self) -> &[(usize, usize)] {
        &self.order
    }
}

/// How to choose a contraction order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Order {
    /// At each step contract the pair whose result is smallest.
    ///
    /// The default because it generalises: it assumes nothing about binary indices or about the
    /// network having come from a graph. It is a heuristic — finding the optimal order is NP-hard —
    /// and like `exact.rs`'s min-fill, a bad order makes this slow or refused, **never wrong**.
    #[default]
    GreedySize,
    /// Contract in the order the tensors were added, left to right.
    ///
    /// Almost always worse. Provided because a caller comparing against another implementation
    /// needs to force the naive order, and because it is the control showing `GreedySize` does
    /// something.
    Sequential,
}

/// A tensor's shape during planning: indices with their widths, and no data.
type Shape = Vec<(Index, usize)>;

/// Entries a shape holds, saturating rather than overflowing.
///
/// `product()` on a `u128` iterator panics on overflow in debug and wraps in release. Both are
/// wrong here and the second is worse: `peak_entries` is the number [`Network::contract_with`]
/// refuses on, so a wrapped value makes an impossible network look affordable — a rank-130 network
/// of binary indices reported `2^66` instead of `2^130`, understating by a factor of `2^64`. That
/// defeats the module's premise that a bad order is slow or refused, never wrong.
///
/// Saturating is the safe direction: `u128::MAX` exceeds any budget a caller can state, so a
/// saturated peak refuses. It is reachable — 128 binary indices on one intermediate — through the
/// open-index mode, where nothing is summed and the final tensor carries every index at once.
fn extent(shape: &[(Index, usize)]) -> u128 {
    shape.iter().fold(1u128, |acc, &(_, d)| acc.saturating_mul(d as u128))
}

impl<S: Semiring> Network<S> {
    /// An empty network.
    #[must_use]
    pub fn new() -> Network<S> {
        Network { tensors: Vec::new(), open: BTreeSet::new() }
    }

    /// Add a tensor.
    pub fn push(&mut self, t: Tensor<S>) -> &mut Self {
        self.tensors.push(t);
        self
    }

    /// Leave `idx` open, so it survives contraction instead of being summed.
    ///
    /// This is how a marginal comes out rather than a scalar: leave one spin's index open on an
    /// Ising network and the contraction is the unnormalised marginal over that spin.
    pub fn open(&mut self, idx: Index) -> &mut Self {
        self.open.insert(idx);
        self
    }

    /// How many tensors it holds.
    #[must_use]
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Whether it holds none.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// Every index, with its dimension, checking that the tensors agree about widths.
    fn widths(&self) -> Result<BTreeMap<Index, usize>, Malformed> {
        let mut out: BTreeMap<Index, usize> = BTreeMap::new();
        for t in &self.tensors {
            for (k, &i) in t.idx.iter().enumerate() {
                let d = t.dims[k];
                match out.get(&i) {
                    None => {
                        out.insert(i, d);
                    }
                    Some(&prev) if prev != d => {
                        return Err(Malformed::DimensionMismatch { index: i, a: prev, b: d });
                    }
                    Some(_) => {}
                }
            }
        }
        Ok(out)
    }

    /// Choose a contraction order and price it, without contracting.
    ///
    /// # Errors
    ///
    /// [`Uncontractable::Malformed`] when two tensors disagree about an index's width. A plan comes
    /// back even when it is expensive — [`Plan::peak_entries`] is the number to judge it by, and
    /// [`Network::contract_with`] is where a budget is enforced.
    pub fn plan(&self, order: Order) -> Result<Plan, Uncontractable> {
        self.widths()?;
        let mut shapes: Vec<Shape> = self
            .tensors
            .iter()
            .map(|t| t.idx.iter().copied().zip(t.dims.iter().copied()).collect())
            .collect();

        let mut steps = Vec::new();
        let mut peak: u128 = shapes.iter().map(|s| extent(s)).max().unwrap_or(1);
        let mut flops: u128 = 0;

        while shapes.len() > 1 {
            let counts = live_counts(&shapes);
            let (i, j) = match order {
                Order::Sequential => (0, 1),
                Order::GreedySize => {
                    let mut best = (0usize, 1usize);
                    let mut best_size = u128::MAX;
                    for a in 0..shapes.len() {
                        for b in (a + 1)..shapes.len() {
                            let (res, _) =
                                merge(&shapes[a], &shapes[b], &counts, &self.open);
                            let size: u128 = extent(&res);
                            if size < best_size {
                                best_size = size;
                                best = (a, b);
                            }
                        }
                    }
                    best
                }
            };

            let (res, work) = merge(&shapes[i], &shapes[j], &counts, &self.open);
            peak = peak.max(extent(&res));
            flops = flops.saturating_add(work);

            let (lo, hi) = if i < j { (i, j) } else { (j, i) };
            shapes.remove(hi);
            shapes.remove(lo);
            shapes.push(res);
            steps.push((lo, hi));
        }

        Ok(Plan { order: steps, peak_entries: peak, flops })
    }

    /// Contract to a single tensor, choosing the order and refusing above `max_entries`.
    ///
    /// # Errors
    ///
    /// [`Uncontractable::TooWide`] when the order's largest intermediate exceeds `max_entries`, and
    /// the errors [`Network::plan`] returns.
    ///
    /// # Panics
    ///
    /// If the plan and the tensor list ever disagree about how many tensors there are. Every step
    /// removes two and pushes one, and [`Network::plan`] emits exactly `len - 1` steps for a
    /// non-empty network, so one tensor remains — but that is an invariant across two functions
    /// rather than something the types enforce, and a panic here would mean it had been broken.
    pub fn contract_with(
        &self,
        order: Order,
        max_entries: u128,
    ) -> Result<Tensor<S>, Uncontractable> {
        let plan = self.plan(order)?;
        if plan.peak_entries > max_entries {
            return Err(Uncontractable::TooWide { entries: plan.peak_entries, max: max_entries });
        }
        if self.tensors.is_empty() {
            // An empty network contracts to the multiplicative identity, which is what makes this
            // compose: adding one tensor to an empty network gives that tensor back.
            return Ok(Tensor::scalar(S::one()));
        }

        let mut live = self.tensors.clone();
        for &(i, j) in &plan.order {
            let counts = live_counts(
                &live
                    .iter()
                    .map(|t| t.idx.iter().copied().zip(t.dims.iter().copied()).collect::<Shape>())
                    .collect::<Vec<_>>(),
            );
            let b = live.remove(j);
            let a = live.remove(i);
            live.push(contract_pair(&a, &b, &counts, &self.open));
        }
        let last = live.pop().expect("a non-empty network leaves exactly one tensor");
        // A network of ONE tensor never enters the loop, and the final tensor may still carry
        // indices nothing else claimed. Summing them here is what makes `contract` mean "multiply
        // and sum everything not open" for every network rather than only for the ones with an
        // even shape.
        Ok(last.marginalise(&self.open))
    }

    /// Contract with the default order and a budget of `2^26` entries.
    ///
    /// The cap matches [`crate::exact::Elimination`]'s default `max_width` of 24 in spirit: chosen
    /// so a refusal arrives before the machine starts swapping, rather than derived from anything.
    ///
    /// # Errors
    ///
    /// As [`Network::contract_with`].
    pub fn contract(&self) -> Result<Tensor<S>, Uncontractable> {
        self.contract_with(Order::default(), 1 << 26)
    }

}

/// A graph's factors, listed once: the site term for each spin that needs one, then each edge.
///
/// Returned rather than visited by callback so every semiring's constructor shares the STRUCTURE of
/// the network and differs only in the entries. The two things a hand-rolled copy gets wrong are
/// here instead: an undirected edge appears in both rows of the CSR and must be emitted once, and a
/// spin with no field and no edges appears on no tensor at all unless one is made for it — which is
/// how a partition function loses a factor of two per isolated spin, silently.
enum IsingFactor {
    /// A single spin's field term. Emitted for a nonzero field, and for an isolated spin so that
    /// its index exists to be summed over.
    Site {
        /// The spin.
        i: usize,
        /// Its field, possibly zero when the spin is isolated.
        h: f64,
    },
    /// One undirected coupling.
    Edge {
        /// The lower-numbered spin.
        i: usize,
        /// The higher-numbered spin.
        j: usize,
        /// The coupling.
        w: f64,
    },
}

fn ising_factors(g: &Graph) -> Vec<IsingFactor> {
    let mut out = Vec::new();
    for i in 0..g.n {
        let isolated = g.offset[i] == g.offset[i + 1];
        if g.h[i] != 0.0 || isolated {
            out.push(IsingFactor::Site { i, h: g.h[i] });
        }
        for k in g.offset[i]..g.offset[i + 1] {
            let j = g.nbr[k] as usize;
            if j > i {
                out.push(IsingFactor::Edge { i, j, w: g.w[k] });
            }
        }
    }
    out
}

/// Build a network from `g`'s factors, given how to score one site value and one edge pair.
fn ising_network<S, F, E>(g: &Graph, site: F, edge: E) -> Network<S>
where
    S: Semiring,
    F: Fn(f64, f64) -> S::Elem,
    E: Fn(f64, f64, f64) -> S::Elem,
{
    let mut net = Network::new();
    for f in ising_factors(g) {
        match f {
            IsingFactor::Site { i, h } => {
                let d = vec![site(h, SPIN[0]), site(h, SPIN[1])];
                net.push(Tensor::new(vec![i as Index], vec![2], d).expect("rank-1, two entries"));
            }
            IsingFactor::Edge { i, j, w } => {
                let mut d = Vec::with_capacity(4);
                for &a in &SPIN {
                    for &b in &SPIN {
                        d.push(edge(w, a, b));
                    }
                }
                net.push(
                    Tensor::new(vec![i as Index, j as Index], vec![2, 2], d)
                        .expect("rank-2, four entries"),
                );
            }
        }
    }
    net
}

/// The two spin values, in index order: 0 is −1 and 1 is +1, as `crate::exact` encodes them.
const SPIN: [f64; 2] = [-1.0, 1.0];

impl Network<SumProduct> {
    /// The network whose contraction is the partition function of `g` at `beta`.
    ///
    /// One index per spin, dimension 2. Each edge contributes `exp(beta·w·sᵢ·sⱼ)` and each site with
    /// a field `exp(beta·hᵢ·sᵢ)`. The sign convention follows [`crate::graph::Graph::energy`],
    /// `E = −Σ w s s − Σ h s`, so the Boltzmann weight `exp(−beta·E)` puts a PLUS in both exponents.
    /// Getting that backwards produces a perfectly plausible number, so it is checked against
    /// [`crate::exact::Elimination::log_partition`] rather than asserted.
    #[must_use]
    pub fn from_ising(g: &Graph, beta: f64) -> Network<SumProduct> {
        ising_network(g, |h, s| (beta * h * s).exp(), |w, a, b| (beta * w * a * b).exp())
    }
}

impl Network<Tropical> {
    /// The network whose contraction is the **ground energy** of `g`.
    ///
    /// No temperature: the tropical semiring is the `beta → ∞` limit and its tensors carry energies
    /// directly. Each edge contributes `−w·sᵢ·sⱼ` and each site `−h·sᵢ`, which is
    /// [`crate::graph::Graph::energy`] term by term, and `⊗` being `+` is what adds them up.
    #[must_use]
    pub fn from_ising(g: &Graph) -> Network<Tropical> {
        ising_network(g, |h, s| -h * s, |w, a, b| -w * a * b)
    }
}

impl Network<MinCount> {
    /// The network whose contraction is the ground energy **and how many states attain it**.
    ///
    /// The same energies [`Network::<Tropical>::from_ising`] carries, each paired with a count of
    /// one. `⊗` adds energies and multiplies counts; `⊕` keeps the lower energy and, on a tie, adds
    /// the counts. Degeneracy is exactly what the tropical semiring discards when it takes a
    /// minimum, and the count carried alongside is what recovers it.
    ///
    /// An isolated spin contributes `(0, 1)` for each of its two states and so doubles the count: a
    /// spin in no factor is still a spin, and its two orientations are two ground states.
    #[must_use]
    pub fn from_ising(g: &Graph) -> Network<MinCount> {
        ising_network(g, |h, s| (-h * s, 1u128), |w, a, b| (-w * a * b, 1u128))
    }
}

/// How many live tensors carry each index.
fn live_counts(shapes: &[Shape]) -> BTreeMap<Index, usize> {
    let mut c: BTreeMap<Index, usize> = BTreeMap::new();
    for s in shapes {
        for &(i, _) in s {
            *c.entry(i).or_insert(0) += 1;
        }
    }
    c
}

/// The shape contracting `a` with `b` produces, and the multiply-accumulates it costs.
///
/// An index is summed away exactly when this pair holds every live copy of it and the caller did
/// not leave it open. An index also carried by a third live tensor SURVIVES on the result and is
/// summed later, when the last carrier is absorbed — which is bucket elimination, and is why a
/// hyperedge needs no special case.
fn merge(
    a: &Shape,
    b: &Shape,
    counts: &BTreeMap<Index, usize>,
    open: &BTreeSet<Index>,
) -> (Shape, u128) {
    let here: BTreeMap<Index, usize> = {
        let mut m: BTreeMap<Index, usize> = BTreeMap::new();
        for &(i, _) in a.iter().chain(b.iter()) {
            *m.entry(i).or_insert(0) += 1;
        }
        m
    };
    let summed = |i: Index| -> bool {
        !open.contains(&i) && here.get(&i).copied().unwrap_or(0) == counts.get(&i).copied().unwrap_or(0)
    };

    let mut out: Shape = Vec::new();
    let mut sum_extent: u128 = 1;
    let mut seen: BTreeSet<Index> = BTreeSet::new();
    for &(i, d) in a.iter().chain(b.iter()) {
        if !seen.insert(i) {
            continue;
        }
        if summed(i) {
            sum_extent = sum_extent.saturating_mul(d as u128);
        } else {
            out.push((i, d));
        }
    }
    let free: u128 = extent(&out);
    (out, free.saturating_mul(sum_extent))
}

/// Contract two tensors, summing every index this pair holds the last copies of.
fn contract_pair<S: Semiring>(
    a: &Tensor<S>,
    b: &Tensor<S>,
    counts: &BTreeMap<Index, usize>,
    open: &BTreeSet<Index>,
) -> Tensor<S> {
    let sa: Shape = a.idx.iter().copied().zip(a.dims.iter().copied()).collect();
    let sb: Shape = b.idx.iter().copied().zip(b.dims.iter().copied()).collect();
    let (out_shape, _) = merge(&sa, &sb, counts, open);
    let out_idx: Vec<Index> = out_shape.iter().map(|&(i, _)| i).collect();
    let out_dims: Vec<usize> = out_shape.iter().map(|&(_, d)| d).collect();
    let out_len: usize = out_dims.iter().product();

    // Everything this pair carries that is not in the output is summed over.
    let mut summed: Shape = Vec::new();
    let mut seen: BTreeSet<Index> = BTreeSet::new();
    for &(i, d) in sa.iter().chain(sb.iter()) {
        if seen.insert(i) && !out_idx.contains(&i) {
            summed.push((i, d));
        }
    }
    let sum_len: usize = summed.iter().map(|&(_, d)| d).product();

    let mut data = vec![S::zero(); out_len];
    let mut out_pos = vec![0usize; out_idx.len()];
    let mut sum_pos = vec![0usize; summed.len()];
    let mut pos_a = vec![0usize; a.idx.len()];
    let mut pos_b = vec![0usize; b.idx.len()];

    for flat in 0..out_len {
        let mut rem = flat;
        for k in (0..out_idx.len()).rev() {
            out_pos[k] = rem % out_dims[k];
            rem /= out_dims[k];
        }
        let mut acc = S::zero();
        for s in 0..sum_len {
            let mut r = s;
            for k in (0..summed.len()).rev() {
                sum_pos[k] = r % summed[k].1;
                r /= summed[k].1;
            }
            let coord = |i: Index| -> usize {
                if let Some(p) = out_idx.iter().position(|&o| o == i) {
                    out_pos[p]
                } else {
                    let p = summed.iter().position(|&(si, _)| si == i).expect(
                        "an index on this pair is either in the output or summed over; there is \
                         no third place for it",
                    );
                    sum_pos[p]
                }
            };
            for (k, &i) in a.idx.iter().enumerate() {
                pos_a[k] = coord(i);
            }
            for (k, &i) in b.idx.iter().enumerate() {
                pos_b[k] = coord(i);
            }
            let va = a.at(&pos_a).expect("a coordinate built from a's own dimensions is in range");
            let vb = b.at(&pos_b).expect("a coordinate built from b's own dimensions is in range");
            acc = S::add(acc, S::mul(va, vb));
        }
        data[flat] = acc;
    }

    Tensor { idx: out_idx, dims: out_dims, data }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exact::Elimination;
    use crate::graph::GraphBuilder;

    fn chain(n: usize, w: f64, h: f64) -> Graph {
        let mut b = GraphBuilder::new(n);
        for i in 0..n - 1 {
            b.couple(i, i + 1, w);
        }
        if h != 0.0 {
            for i in 0..n {
                b.set_bias(i, h);
            }
        }
        b.build()
    }

    /// Contracting the Ising network must give the partition function elimination gives.
    ///
    /// TWO IMPLEMENTATIONS OF ONE QUANTITY, SHARING NO CODE. `exact.rs` eliminates variables from a
    /// spin graph; this contracts indices in a network built from the same graph. Agreement is
    /// evidence about both. The lattice cases are the load-bearing ones: every site has degree
    /// four, so they exercise the rule that an index carried by several tensors is summed once,
    /// after the last is absorbed.
    #[test]
    fn contraction_agrees_with_variable_elimination_on_log_z() {
        let cases: Vec<(&str, Graph)> = vec![
            ("chain(8, 1.0, 0.0)", chain(8, 1.0, 0.0)),
            ("chain(9, -0.4, 0.35)", chain(9, -0.4, 0.35)),
            ("ring(10, 1.0, 0.0)", crate::ising::ring(10, 1.0, 0.0)),
            ("ring(12, -0.7, 0.3)", crate::ising::ring(12, -0.7, 0.3)),
            ("lattice2d(4, 1.0)", crate::ising::lattice2d(4, 1.0)),
            ("lattice2d(5, -0.6)", crate::ising::lattice2d(5, -0.6)),
        ];
        let e = Elimination::default();
        for (name, g) in cases {
            for beta in [0.1f64, 0.5, 1.0, 2.0] {
                let net = Network::<SumProduct>::from_ising(&g, beta);
                let z = net
                    .contract()
                    .unwrap_or_else(|err| panic!("{name} at beta {beta}: {err}"))
                    .value()
                    .expect("a fully contracted network is rank 0");
                let want = e
                    .log_partition(&g, beta)
                    .unwrap_or_else(|err| panic!("{name}: {err}"))
                    .log_z
                    .expect("sum-product was run");
                let got = z.ln();
                assert!(
                    (got - want).abs() < 1e-9 * want.abs().max(1.0),
                    "{name} at beta {beta}: contraction gives log Z {got}, elimination {want}"
                );
            }
        }
    }

    /// ONE NETWORK, FOUR ARITHMETICS, FOUR ANSWERS — each against its own oracle.
    ///
    /// The Generalized Distributive Law says the contraction schedule is blind to the scalar. This
    /// is that claim made checkable: the same graph, the same indices, the same elimination, and
    /// four different questions answered by changing a type parameter.
    ///
    /// Scored against brute-force enumeration rather than against `exact::Elimination`, because the
    /// two engines have already been caught sharing a blind spot — both omitted a tensor for an
    /// isolated spin, so they agreed on a wrong `log Z` to the last ulp. Enumeration shares nothing
    /// with either.
    #[test]
    fn one_network_over_four_semirings_answers_four_questions() {
        let cases: Vec<(&str, Graph)> = vec![
            ("ring(9, -1, 0)", crate::ising::ring(9, -1.0, 0.0)),
            ("ring(10, -1, 0)", crate::ising::ring(10, -1.0, 0.0)),
            ("chain(8) + field", chain(8, 1.0, 0.25)),
            ("torus 3x3", crate::ising::lattice2d(3, 1.0)),
            ("a pair and two free spins", {
                let mut b = GraphBuilder::new(4);
                b.couple(0, 1, 1.0);
                b.build()
            }),
        ];

        for (name, g) in cases {
            // Brute force: the ground energy, how many states attain it, and Z at beta.
            let beta = 0.6;
            let (mut e0, mut deg, mut z) = (f64::INFINITY, 0u128, 0.0f64);
            for m in 0u64..(1u64 << g.n) {
                let st: Vec<i8> =
                    (0..g.n).map(|i| if m >> i & 1 == 1 { 1i8 } else { -1 }).collect();
                let e = g.energy(&st);
                z += (-beta * e).exp();
                if e < e0 - 1e-9 {
                    e0 = e;
                    deg = 1;
                } else if (e - e0).abs() <= 1e-9 {
                    deg += 1;
                }
            }

            // sum-product -> Z
            let got_z = Network::<SumProduct>::from_ising(&g, beta).contract().unwrap().value().unwrap();
            assert!(
                (got_z.ln() - z.ln()).abs() < 1e-9,
                "{name}: sum-product log Z {} vs enumeration {}",
                got_z.ln(),
                z.ln()
            );

            // tropical -> the ground energy
            let got_e0 = Network::<Tropical>::from_ising(&g).contract().unwrap().value().unwrap();
            assert!(
                (got_e0 - e0).abs() < 1e-9,
                "{name}: tropical ground energy {got_e0} vs enumeration {e0}"
            );

            // min-count -> the ground energy AND its degeneracy, in one contraction
            let (mc_e, mc_n) =
                Network::<MinCount>::from_ising(&g).contract().unwrap().value().unwrap();
            assert!((mc_e - e0).abs() < 1e-9, "{name}: min-count energy {mc_e} vs {e0}");
            assert_eq!(mc_n, deg, "{name}: min-count counted {mc_n} ground states, there are {deg}");

            // counting -> how many configurations there are at all, which for a network of
            // all-ones tensors is 2^n. A control: it is the only one of the four whose answer does
            // not depend on the couplings, so a constructor that ignored them would pass here and
            // fail the other three.
            let mut ones = Network::<Counting>::new();
            for i in 0..g.n {
                ones.push(Tensor::new(vec![i as Index], vec![2], vec![1u128, 1]).unwrap());
            }
            assert_eq!(ones.contract().unwrap().value(), Some(1u128 << g.n), "{name}: counting");
        }
    }

    /// The degeneracy the min-count semiring reports agrees with the two-temperature estimate.
    ///
    /// `exact::ground_degeneracy` extrapolates from `log Z` at two betas; this counts directly in
    /// one contraction. Different methods, and neither is derived from the other — so where they
    /// both apply they must agree, and where they disagree one of them is wrong.
    #[test]
    fn counting_by_contraction_agrees_with_counting_by_cold_limit() {
        for n in [9usize, 11, 13] {
            let g = crate::ising::ring(n, -1.0, 0.0);
            let (_, by_contraction) =
                Network::<MinCount>::from_ising(&g).contract().unwrap().value().unwrap();
            let by_cold_limit = crate::exact::Elimination::default()
                .ground_degeneracy(&g, (20.0, 40.0))
                .unwrap()
                .count
                .expect("a ring converges");
            assert_eq!(
                by_contraction, u128::from(by_cold_limit),
                "odd AF ring {n}: contraction counted {by_contraction}, cold limit {by_cold_limit}"
            );
            // And the closed form both should be reproducing.
            assert_eq!(by_contraction, 2 * n as u128);
        }
    }

    /// The identities a semiring has to satisfy, checked rather than assumed.
    ///
    /// `zero` must be the identity for `⊕` and annihilate under `⊗`; `one` must be the identity for
    /// `⊗`. Contraction relies on all three — `zero` is what an empty accumulator starts at and
    /// `one` is what an empty network contracts to — and a semiring that gets one wrong produces a
    /// number rather than an error.
    #[test]
    fn every_semiring_has_the_identities_contraction_relies_on() {
        fn check<S: Semiring>(vals: &[S::Elem]) {
            for &v in vals {
                assert_eq!(S::add(S::zero(), v), v, "{}: zero is not the additive identity", S::NAME);
                assert_eq!(S::mul(S::one(), v), v, "{}: one is not the multiplicative identity", S::NAME);
                assert_eq!(S::mul(S::zero(), v), S::zero(), "{}: zero does not annihilate", S::NAME);
            }
        }
        check::<SumProduct>(&[0.0, 1.0, 2.5, -3.25]);
        check::<Counting>(&[0, 1, 7, 1_000_000]);
        // Tropical `zero` is +inf and `mul` is `+`, so inf + v = inf annihilates as required.
        check::<Tropical>(&[0.0, 1.0, -2.5, 17.0]);
        check::<MinCount>(&[(0.0, 1), (2.5, 3), (-1.0, 8)]);
    }

    /// An open index gives a marginal instead of a scalar, and it must match exact marginals.
    ///
    /// This is the capability `exact.rs` has as a dedicated method and a network has as a *mode*:
    /// leave a wire dangling and the contraction stops being a number. Checked against
    /// `Elimination::marginals`, which computes the same quantity by a different route.
    #[test]
    fn an_open_index_contracts_to_the_marginal_exact_inference_reports() {
        let g = crate::ising::ring(10, 0.8, 0.2);
        let beta = 0.7;
        let exact = Elimination::default().marginals(&g, beta).expect("a ring is narrow");

        for spin in [0usize, 3, 7] {
            let mut net = Network::<SumProduct>::from_ising(&g, beta);
            net.open(spin as Index);
            let t = net.contract().expect("a ring with one open leg is narrow");
            assert_eq!(t.indices(), &[spin as Index]);
            // p(+1) = weight(+1) / (weight(-1) + weight(+1)); index 1 is the +1 state.
            let down = t.at(&[0]).expect("state -1");
            let up = t.at(&[1]).expect("state +1");
            let p_up = up / (down + up);
            assert!(
                (p_up - exact[spin]).abs() < 1e-9,
                "spin {spin}: contraction says p(+1) = {p_up}, elimination says {}",
                exact[spin]
            );
        }
    }

    /// The order changes the price and must not change the answer.
    #[test]
    fn a_different_order_is_a_different_price_for_the_same_number() {
        let g = crate::ising::lattice2d(4, 1.0);
        let net = Network::<SumProduct>::from_ising(&g, 0.7);

        let greedy = net.plan(Order::GreedySize).expect("a small lattice is contractable");
        let naive = net.plan(Order::Sequential).expect("a small lattice is contractable");
        assert!(
            greedy.peak_entries <= naive.peak_entries,
            "the greedy order should not build a wider intermediate: {} vs {}",
            greedy.peak_entries,
            naive.peak_entries
        );

        let a = net.contract_with(Order::GreedySize, 1 << 26).unwrap().value().unwrap();
        let b = net.contract_with(Order::Sequential, 1 << 26).unwrap().value().unwrap();
        assert!(
            (a - b).abs() < 1e-9 * a.abs().max(1.0),
            "two orders gave two answers: {a} and {b}"
        );
    }

    /// The plan prices the order before it is paid, and the budget refuses above it.
    #[test]
    fn the_cost_is_reported_before_it_is_incurred_and_can_be_refused() {
        let g = crate::ising::lattice2d(5, 1.0);
        let net = Network::<SumProduct>::from_ising(&g, 0.5);
        let plan = net.plan(Order::default()).expect("a 5x5 lattice is contractable");
        assert!(plan.peak_entries >= 2);
        assert!(plan.flops > 0);

        let err = net
            .contract_with(Order::default(), plan.peak_entries - 1)
            .expect_err("a budget below the peak must refuse");
        match err {
            Uncontractable::TooWide { entries, max } => {
                assert_eq!(entries, plan.peak_entries);
                assert_eq!(max, plan.peak_entries - 1);
            }
            other => panic!("expected TooWide, got {other:?}"),
        }
    }

    /// Malformed tensors are refused when they are built, each with its own reason.
    #[test]
    fn a_tensor_that_cannot_exist_is_refused_when_it_is_built() {
        assert_eq!(
            Tensor::<SumProduct>::new(vec![0, 1], vec![2, 2], vec![1.0, 2.0]),
            Err(Malformed::Size { got: 2, want: 4 })
        );
        assert_eq!(
            Tensor::<SumProduct>::new(vec![0, 0], vec![2, 2], vec![1.0; 4]),
            Err(Malformed::RepeatedIndex(0))
        );
        assert_eq!(
            Tensor::<SumProduct>::new(vec![0], vec![0], vec![]),
            Err(Malformed::ZeroDimension(0))
        );
    }

    /// Two tensors disagreeing about a wire's width is caught before any arithmetic.
    #[test]
    fn an_index_that_is_two_widths_at_once_is_refused() {
        let mut net = Network::<SumProduct>::new();
        net.push(Tensor::new(vec![0], vec![2], vec![1.0, 1.0]).unwrap());
        net.push(Tensor::new(vec![0], vec![3], vec![1.0; 3]).unwrap());
        match net.plan(Order::default()) {
            Err(Uncontractable::Malformed(Malformed::DimensionMismatch { index, .. })) => {
                assert_eq!(index, 0);
            }
            other => panic!("expected a width mismatch, got {other:?}"),
        }
    }

    /// A network too wide to count is refused, not wrapped around.
    ///
    /// `peak_entries` was a non-saturating `u128` product while its neighbours saturated. At rank
    /// 128 over binary indices that panics in debug — from a function whose docs list only
    /// `Uncontractable::Malformed` and carry no `# Panics` — and WRAPS in release, reporting `2^66`
    /// for a `2^130` intermediate. `contract_with` refuses on exactly that number, so an impossible
    /// network looked affordable, which is the opposite of "slow or refused, never wrong".
    ///
    /// Reachable through the documented open-index mode, where nothing is summed and the final
    /// tensor carries every index at once.
    #[test]
    fn an_intermediate_too_wide_to_count_saturates_and_refuses() {
        let mut net = Network::<SumProduct>::new();
        for i in 0..130u32 {
            net.push(Tensor::new(vec![i], vec![2], vec![1.0, 1.0]).unwrap());
            net.open(i);
        }
        // Neither order may panic, and both must report something a budget can refuse.
        for order in [Order::Sequential, Order::GreedySize] {
            let plan = net.plan(order).expect("a shape-only plan does not allocate");
            assert_eq!(
                plan.peak_entries,
                u128::MAX,
                "a 2^130 intermediate must saturate rather than wrap: {order:?}"
            );
            match net.contract_with(order, 1 << 26) {
                Err(Uncontractable::TooWide { entries, max }) => {
                    assert_eq!(entries, u128::MAX);
                    assert_eq!(max, 1 << 26);
                }
                other => panic!("a saturated peak must refuse, got {other:?}"),
            }
        }
    }

    /// An empty network contracts to one, so adding a tensor to it gives that tensor back.
    #[test]
    fn the_empty_network_is_the_multiplicative_identity() {
        assert_eq!(Network::<SumProduct>::new().contract().unwrap().value(), Some(1.0));
    }

    /// A single tensor still has its indices summed, which is what "contract" means here.
    #[test]
    fn one_tensor_alone_is_summed_over_its_own_indices() {
        let mut net = Network::<SumProduct>::new();
        net.push(Tensor::new(vec![0, 1], vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap());
        assert_eq!(net.contract().unwrap().value(), Some(10.0));

        // Unless the caller leaves one open, in which case it is a partial sum.
        let mut kept = Network::<SumProduct>::new();
        kept.push(Tensor::new(vec![0, 1], vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap());
        kept.open(0);
        let t = kept.contract().unwrap();
        assert_eq!(t.indices(), &[0]);
        assert_eq!(t.at(&[0]), Some(3.0));
        assert_eq!(t.at(&[1]), Some(7.0));
    }
}
