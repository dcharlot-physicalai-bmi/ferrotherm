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
//! | output | `log Z`, ground state, marginals | any tensor, with indices left **open** |
//! | ordering | min-fill | greedy-cost, or the caller's own |
//! | prices the order first | `Elimination::width` | [`Network::plan`] |
//!
//! Those columns are what separate "exact inference on a spin glass" from "contract the network
//! somebody else's problem lowers to", and the last two rows are where the quantum-simulation
//! literature lives: a quantum circuit is a tensor network, and contracting one is how the
//! classical rebuttals to the Sycamore supremacy claim were computed.
//!
//! # What this is not
//!
//! It is not a quantum simulator, and this module makes no quantum claim. It is real-valued
//! (`f64`), so it contracts probability and partition-function networks and **not amplitudes**;
//! complex arithmetic is a separate question and pretending otherwise here would be exactly the
//! vocabulary-as-capability this crate refuses. It is also **exact**: there is no bond-dimension
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

/// One tensor: a dense array over its indices, row-major with the FIRST index slowest.
///
/// The layout is stated because it is load-bearing — [`Tensor::at`] and every contraction below
/// depend on it, and a reader checking this module against another implementation needs to know
/// which convention is in force rather than inferring it from a loop.
#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    idx: Vec<Index>,
    dims: Vec<usize>,
    data: Vec<f64>,
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

impl Tensor {
    /// A tensor over `idx` with dimensions `dims` and entries `data`, row-major, first index
    /// slowest.
    ///
    /// # Errors
    ///
    /// [`Malformed`] when the data length disagrees with the shape, an index repeats, or a
    /// dimension is zero.
    pub fn new(idx: Vec<Index>, dims: Vec<usize>, data: Vec<f64>) -> Result<Tensor, Malformed> {
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
    pub fn scalar(v: f64) -> Tensor {
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
    pub fn value(&self) -> Option<f64> {
        (self.idx.is_empty()).then(|| self.data[0])
    }

    /// The entry at `pos`, one coordinate per index in layout order.
    #[must_use]
    pub fn at(&self, pos: &[usize]) -> Option<f64> {
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
    pub fn data(&self) -> &[f64] {
        &self.data
    }

    /// Sum this tensor over every index not in `keep`.
    #[must_use]
    fn marginalise(&self, keep: &BTreeSet<Index>) -> Tensor {
        let out_pos: Vec<usize> =
            (0..self.idx.len()).filter(|&k| keep.contains(&self.idx[k])).collect();
        if out_pos.len() == self.idx.len() {
            return self.clone();
        }
        let out_idx: Vec<Index> = out_pos.iter().map(|&k| self.idx[k]).collect();
        let out_dims: Vec<usize> = out_pos.iter().map(|&k| self.dims[k]).collect();
        let out_len: usize = out_dims.iter().product();
        let mut data = vec![0.0f64; out_len];

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
            data[o] += self.data[flat];
        }
        Tensor { idx: out_idx, dims: out_dims, data }
    }
}

/// A set of tensors, contracted by multiplying them and summing every index not left open.
#[derive(Clone, Debug, Default)]
pub struct Network {
    tensors: Vec<Tensor>,
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

impl Network {
    /// An empty network.
    #[must_use]
    pub fn new() -> Network {
        Network::default()
    }

    /// Add a tensor.
    pub fn push(&mut self, t: Tensor) -> &mut Self {
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
        let mut peak: u128 =
            shapes.iter().map(|s| s.iter().map(|&(_, d)| d as u128).product::<u128>()).max().unwrap_or(1);
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
                            let size: u128 = res.iter().map(|&(_, d)| d as u128).product();
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
            peak = peak.max(res.iter().map(|&(_, d)| d as u128).product::<u128>());
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
    pub fn contract_with(&self, order: Order, max_entries: u128) -> Result<Tensor, Uncontractable> {
        let plan = self.plan(order)?;
        if plan.peak_entries > max_entries {
            return Err(Uncontractable::TooWide { entries: plan.peak_entries, max: max_entries });
        }
        if self.tensors.is_empty() {
            // An empty network contracts to the multiplicative identity, which is what makes this
            // compose: adding one tensor to an empty network gives that tensor back.
            return Ok(Tensor::scalar(1.0));
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
    pub fn contract(&self) -> Result<Tensor, Uncontractable> {
        self.contract_with(Order::default(), 1 << 26)
    }

    /// The tensor network whose contraction is the partition function of `g` at `beta`.
    ///
    /// One index per spin, dimension 2, where 0 means −1 and 1 means +1 — the same encoding
    /// [`crate::exact`] uses. Each edge contributes a rank-2 tensor `exp(beta * w * s_i * s_j)` and
    /// each site with a field a rank-1 tensor `exp(beta * h_i * s_i)`.
    ///
    /// The sign convention follows [`crate::graph::Graph::energy`], which is
    /// `E = -sum w s s - sum h s`, so the Boltzmann weight `exp(-beta E)` puts a PLUS sign in both
    /// exponents here. Getting that backwards is the classic error in this construction and it
    /// produces a perfectly plausible number, so it is checked rather than asserted — see the
    /// agreement test against `Elimination::log_partition`.
    ///
    /// Any degree is fine. A spin carried by five edge tensors is summed once, after the last of
    /// them is absorbed, which is what bucket elimination does and why no special case is needed.
    ///
    /// # Panics
    ///
    /// If a rank-1 tensor of two entries or a rank-2 tensor of four is rejected as malformed, which
    /// would mean [`Tensor::new`]'s shape check disagrees with arithmetic. The shapes here are
    /// literals, not caller input, so this is an assertion about this function rather than a
    /// condition a caller can reach.
    ///
    /// A non-finite `beta` or coupling produces `inf` or `NaN` entries rather than a panic — the
    /// contraction then carries them through to the result, where they are visible, instead of
    /// failing here where the cause would be clearer but the caller has already been told the graph
    /// is finite by whoever built it.
    #[must_use]
    pub fn from_ising(g: &Graph, beta: f64) -> Network {
        let mut net = Network::new();
        let s = [-1.0f64, 1.0];
        for i in 0..g.n {
            if g.h[i] != 0.0 {
                let data = vec![(beta * g.h[i] * s[0]).exp(), (beta * g.h[i] * s[1]).exp()];
                net.push(
                    Tensor::new(vec![i as Index], vec![2], data)
                        .expect("a rank-1 tensor of two entries is well formed"),
                );
            }
            for k in g.offset[i]..g.offset[i + 1] {
                let j = g.nbr[k] as usize;
                if j <= i {
                    continue; // each undirected edge once
                }
                let w = g.w[k];
                let mut data = Vec::with_capacity(4);
                for &a in &s {
                    for &b in &s {
                        data.push((beta * w * a * b).exp());
                    }
                }
                net.push(
                    Tensor::new(vec![i as Index, j as Index], vec![2, 2], data)
                        .expect("a rank-2 tensor of four entries is well formed"),
                );
            }
        }
        net
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
    let free: u128 = out.iter().map(|&(_, d)| d as u128).product();
    (out, free.saturating_mul(sum_extent))
}

/// Contract two tensors, summing every index this pair holds the last copies of.
fn contract_pair(
    a: &Tensor,
    b: &Tensor,
    counts: &BTreeMap<Index, usize>,
    open: &BTreeSet<Index>,
) -> Tensor {
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

    let mut data = vec![0.0f64; out_len];
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
        let mut acc = 0.0f64;
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
            acc += va * vb;
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
                let net = Network::from_ising(&g, beta);
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
            let mut net = Network::from_ising(&g, beta);
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
        let net = Network::from_ising(&g, 0.7);

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
        let net = Network::from_ising(&g, 0.5);
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
            Tensor::new(vec![0, 1], vec![2, 2], vec![1.0, 2.0]),
            Err(Malformed::Size { got: 2, want: 4 })
        );
        assert_eq!(
            Tensor::new(vec![0, 0], vec![2, 2], vec![1.0; 4]),
            Err(Malformed::RepeatedIndex(0))
        );
        assert_eq!(Tensor::new(vec![0], vec![0], vec![]), Err(Malformed::ZeroDimension(0)));
    }

    /// Two tensors disagreeing about a wire's width is caught before any arithmetic.
    #[test]
    fn an_index_that_is_two_widths_at_once_is_refused() {
        let mut net = Network::new();
        net.push(Tensor::new(vec![0], vec![2], vec![1.0, 1.0]).unwrap());
        net.push(Tensor::new(vec![0], vec![3], vec![1.0; 3]).unwrap());
        match net.plan(Order::default()) {
            Err(Uncontractable::Malformed(Malformed::DimensionMismatch { index, .. })) => {
                assert_eq!(index, 0);
            }
            other => panic!("expected a width mismatch, got {other:?}"),
        }
    }

    /// An empty network contracts to one, so adding a tensor to it gives that tensor back.
    #[test]
    fn the_empty_network_is_the_multiplicative_identity() {
        assert_eq!(Network::new().contract().unwrap().value(), Some(1.0));
    }

    /// A single tensor still has its indices summed, which is what "contract" means here.
    #[test]
    fn one_tensor_alone_is_summed_over_its_own_indices() {
        let mut net = Network::new();
        net.push(Tensor::new(vec![0, 1], vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap());
        assert_eq!(net.contract().unwrap().value(), Some(10.0));

        // Unless the caller leaves one open, in which case it is a partial sum.
        let mut kept = Network::new();
        kept.push(Tensor::new(vec![0, 1], vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap());
        kept.open(0);
        let t = kept.contract().unwrap();
        assert_eq!(t.indices(), &[0]);
        assert_eq!(t.at(&[0]), Some(3.0));
        assert_eq!(t.at(&[1]), Some(7.0));
    }
}
