//! Number partitioning — the NP-hard problem the Ising literature uses on itself.
//!
//! Split `n` positive integers into two sets so the two sums are as close as possible. Writing
//! `s_i = +1` for one set and `s_i = -1` for the other, the quantity to minimise is the
//! **discrepancy**
//!
//! ```text
//!     D(s) = | sum_i a_i s_i |
//! ```
//!
//! and the Hamiltonian of the literature is its square, `H(s) = (sum_i a_i s_i)^2`. Squared
//! because the square is a pairwise Ising energy and the absolute value is not: expanding it gives
//! a fully connected model with a coupling for every pair and no fields at all.
//!
//! Why it is here: this crate carries G-set, DIMACS, planted loops, Wishart and the Chook
//! ensembles, and every one of them is a graph whose hardness has to be argued for. Number
//! partitioning is the opposite — it is the one place in this field where the hard/easy boundary
//! is **known analytically**, sits at a named point, and can be crossed by turning one knob.
//! Mertens, *Phase transition in the number partitioning problem*, Phys. Rev. Lett. **81**, 4281
//! (1998), showed that for `n` numbers drawn uniformly from `{1, ..., 2^m}` the control parameter
//! is `kappa = m/n`: below one, almost every instance has a perfect partition and there are
//! exponentially many of them; above one, almost none does. The strong classical baseline is
//! Karmarkar and Karp's differencing heuristic, *The differencing method of set partitioning*,
//! report UCB/CSD 82/113, University of California, Berkeley (1982).
//!
//! # The coupling is `-2 a_i a_j`, not `-a_i a_j`
//!
//! Expanding the square over ORDERED pairs gives `H = sum_i a_i^2 + 2 sum_{i<j} a_i a_j s_i s_j`,
//! and this crate's energy counts each undirected edge ONCE,
//! `E(s) = -sum_{(i,j)} J_ij s_i s_j - sum_i h_i s_i`. Matching term for term therefore needs
//!
//! ```text
//!     J_ij = -2 a_i a_j,     h_i = 0,     E(s) = D(s)^2 - sum_i a_i^2
//! ```
//!
//! The widely quoted form `J_ij = -a_i a_j` belongs to the convention where the double sum runs
//! over ordered pairs and each pair is therefore counted twice. Used here it halves the energy:
//! `E(s) = (D(s)^2 - sum_i a_i^2) / 2`. That still RANKS states identically, so a ground state is
//! still a ground state and a solver would not notice — which is exactly why it is worth a test.
//! `the_halved_coupling_is_not_the_squared_discrepancy` measures the gap instead of arguing about
//! it: on the weights `{8, 7, 6, 5}` the all-up state has `D^2 = 676`, this module's graph reports
//! `676 - 174 = 502`, the halved form reports `251`, and both pick the same ground state.
//!
//! # Everything here is integer-exact, and that is enforced rather than hoped
//!
//! Weights are `u64`, discrepancies are `u64`, and the Ising energy of any state is an integer
//! held in an `f64`. [`Instance::new`] refuses a weight set whose total exceeds [`MAX_TOTAL`],
//! which is `floor(sqrt(2^53))`: every term and every partial sum inside
//! [`crate::graph::Graph::energy`] is then an integer no larger than `total^2 <= 2^53`, so the
//! energy is exact, not nearly exact.
//! `ising_energy_equals_the_squared_discrepancy_on_every_state` asserts the identity with `==`.
//!
//! This is also why nothing here accumulates through [`crate::round`]. That module exists because a
//! float sum can land on the wrong side of the truth and turn a bound into a falsehood; the sums in
//! this module are exact by the ceiling above, and [`Instance::lower_bound`] is pure integer
//! arithmetic. Directed rounding here would widen a bound that is already tight.
//!
//! The ceiling is what caps the family. A set of `n` weights of `m` bits is admitted FOR CERTAIN
//! while `n * 2^m <= MAX_TOTAL`, which at `n = 14` is 22 bits and `kappa = 1.57` — comfortably past
//! the transition, which is all the physics needs, but not arbitrarily far past it. A random draw
//! totals about half that worst case, so a bit or two more usually fits; [`uniform`] returns the
//! refusal when it does not, rather than rescaling the weights into something else's problem.
//!
//! # The transition, measured here
//!
//! Fraction of instances with a perfect partition at `n = 14`, 200 seeds per point, the optimum by
//! exhaustive enumeration over all `2^13` sign assignments
//! (`perfect_partitions_collapse_across_kappa_one_by_exact_enumeration`):
//!
//! | bits `m` | `kappa` | perfect |
//! |---|---|---|
//! | 7 | 0.50 | **1.000** |
//! | 10 | 0.71 | 0.960 |
//! | 12 | 0.86 | 0.660 |
//! | 13 | 0.93 | 0.380 |
//! | 14 | **1.00** | 0.235 |
//! | 16 | 1.14 | 0.040 |
//! | 18 | 1.29 | 0.010 |
//! | 21 | 1.50 | **0.000** |
//!
//! **At finite `n` the crossing is below one, and by a predictable amount.** `kappa = 1` itself is
//! already mostly past the transition at this size — 23.5% of instances still have a perfect
//! partition — and the half-way point sits at `kappa = 0.898`. That is not sampling noise, it is
//! the finite-size term: see [`critical_kappa`], whose closed form tracks the measured crossing to
//! within a quarter of a bit over `n = 8 ... 18`.
//!
//! # Karmarkar-Karp, measured against the optimum
//!
//! 200 instances per row at `n = 14`, differencing against exhaustive enumeration:
//!
//! | bits `m` | `kappa` | differencing optimal | mean `D / total` |
//! |---|---|---|---|
//! | 10 | 0.71 | 70/200 | 1.03e-3 |
//! | 16 | 1.14 | 14/200 | 8.87e-4 |
//! | 21 | 1.50 | 14/200 | 9.68e-4 |
//!
//! It is a strong heuristic and it is not a solver, which is the point of carrying it: below the
//! transition it lands on the optimum a third of the time because perfect partitions are
//! everywhere, and above it, hardly ever.

use crate::graph::{Graph, GraphBuilder};
use crate::rng::Pcg;

/// The largest weight total this module accepts: `floor(sqrt(2^53))`.
///
/// Above it, `total^2` no longer fits the `f64` mantissa, and the Ising energy of a state stops
/// being an exact integer — which would silently demote every identity in this module from an
/// equality to an approximation. The limit is on the TOTAL rather than on any single weight
/// because the energy's largest partial sum is `total^2 - sum_i a_i^2`.
pub const MAX_TOTAL: u64 = 94906265;

/// The widest weight drawn by [`uniform`], in bits.
///
/// A representational limit, not a useful one: `1u64 << 64` does not exist, so 63 is where the
/// draw itself stops working. Any instance with more than about 26 bits per weight is refused by
/// [`MAX_TOTAL`] long before this matters.
pub const MAX_BITS: u32 = 63;

/// The largest instance [`Instance::exact_optimum`] will enumerate.
///
/// The global spin flip leaves the discrepancy alone, so enumeration visits `2^(n-1)` states and
/// this limit is one spin above [`crate::oracle::Exhaustive::MAX_SPINS`] for the same work.
pub const MAX_ENUMERABLE: usize = 27;

/// What was rejected, and what was actually seen.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PartitionError {
    /// No weights at all. There is no partition of nothing, and `kappa = m/n` has no denominator.
    Empty,
    /// A weight of zero, at this index.
    ///
    /// Refused rather than dropped: a zero weight sits in either set for free, so it changes `n` —
    /// and therefore `kappa`, and therefore which side of the transition the instance is on —
    /// without changing the problem at all.
    ZeroWeight {
        /// Index of the offending weight.
        index: usize,
    },
    /// The weights sum past [`MAX_TOTAL`], where the Ising energy stops being an exact integer.
    TooHeavy {
        /// The total that was presented, in `u128` because it need not fit a `u64`.
        total: u128,
        /// [`MAX_TOTAL`].
        limit: u64,
    },
    /// More bits per weight than a `u64` draw carries.
    TooManyBits {
        /// The width requested.
        bits: u32,
        /// [`MAX_BITS`].
        limit: u32,
    },
    /// Too many weights to enumerate all `2^(n-1)` sign assignments.
    TooManyToEnumerate {
        /// The instance size.
        n: usize,
        /// [`MAX_ENUMERABLE`].
        limit: usize,
    },
}

impl core::fmt::Display for PartitionError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            PartitionError::Empty => write!(f, "a partitioning instance needs at least one weight"),
            PartitionError::ZeroWeight { index } => write!(
                f,
                "weight {index} is zero; a zero weight joins either set for free, so it moves n \
                 and therefore kappa without changing the problem"
            ),
            PartitionError::TooHeavy { total, limit } => write!(
                f,
                "the weights total {total}, above {limit} = floor(sqrt(2^53)); past that the \
                 squared discrepancy is no longer an exact f64 and every energy here would be \
                 approximate without saying so"
            ),
            PartitionError::TooManyBits { bits, limit } => {
                write!(f, "{bits} bits per weight, but a u64 draw carries at most {limit}")
            }
            PartitionError::TooManyToEnumerate { n, limit } => write!(
                f,
                "exhaustive enumeration over {n} weights is 2^{} states; the limit is {limit}",
                n - 1
            ),
        }
    }
}

impl core::error::Error for PartitionError {}

/// A set of positive integer weights, with its Ising image guaranteed exact.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Instance {
    a: Vec<u64>,
    total: u64,
}

/// A two-set split and the discrepancy it achieves.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Partition {
    signs: Vec<i8>,
    discrepancy: u64,
}

impl Partition {
    /// The assignment, `+1` for one set and `-1` for the other, in weight order.
    #[must_use]
    pub fn signs(&self) -> &[i8] {
        &self.signs
    }

    /// `|sum_i a_i s_i|` for these signs — the number being minimised.
    #[must_use]
    pub fn discrepancy(&self) -> u64 {
        self.discrepancy
    }

    /// The two sets as index lists, the `+1` side first.
    #[must_use]
    pub fn sets(&self) -> (Vec<usize>, Vec<usize>) {
        let mut up = Vec::new();
        let mut down = Vec::new();
        for (i, &s) in self.signs.iter().enumerate() {
            if s > 0 {
                up.push(i);
            } else {
                down.push(i);
            }
        }
        (up, down)
    }
}

impl Instance {
    /// Take a weight set, or say why it cannot be one.
    ///
    /// # Errors
    ///
    /// [`PartitionError::Empty`] for no weights, [`PartitionError::ZeroWeight`] for a zero, and
    /// [`PartitionError::TooHeavy`] when the total passes [`MAX_TOTAL`] and the energies would
    /// stop being exact.
    pub fn new(weights: &[u64]) -> Result<Self, PartitionError> {
        if weights.is_empty() {
            return Err(PartitionError::Empty);
        }
        let mut total: u128 = 0;
        for (index, &w) in weights.iter().enumerate() {
            if w == 0 {
                return Err(PartitionError::ZeroWeight { index });
            }
            total += u128::from(w);
        }
        if total > u128::from(MAX_TOTAL) {
            return Err(PartitionError::TooHeavy { total, limit: MAX_TOTAL });
        }
        Ok(Instance { a: weights.to_vec(), total: total as u64 })
    }

    /// The weights, in the order they were given.
    #[must_use]
    pub fn weights(&self) -> &[u64] {
        &self.a
    }

    /// How many weights, which is the spin count of [`Instance::graph`].
    #[must_use]
    pub fn n(&self) -> usize {
        self.a.len()
    }

    /// `sum_i a_i`, at most [`MAX_TOTAL`].
    #[must_use]
    pub fn total(&self) -> u64 {
        self.total
    }

    /// The smallest discrepancy arithmetic allows: `0` for an even total, `1` for an odd one.
    ///
    /// Every `D(s)` has the parity of the total, because flipping one spin moves the signed sum by
    /// an even number. This is the floor a "perfect partition" is measured against, and it is why
    /// the literature calls an odd-total split with `D = 1` perfect.
    #[must_use]
    pub fn parity(&self) -> u64 {
        self.total % 2
    }

    /// Whether a discrepancy is perfect, i.e. equal to [`Instance::parity`].
    #[must_use]
    pub fn is_perfect(&self, discrepancy: u64) -> bool {
        discrepancy == self.parity()
    }

    /// `sum_i a_i s_i`, the signed sum whose magnitude is the discrepancy.
    ///
    /// # Panics
    ///
    /// If `s` is not one spin per weight, or carries a value other than `+1` or `-1`. A spin of
    /// `0` is an unreadable input rather than a half-assignment, and silently treating it as either
    /// set would report a discrepancy for a partition nobody described.
    #[must_use]
    pub fn signed_sum(&self, s: &[i8]) -> i64 {
        assert_eq!(s.len(), self.a.len(), "{} spins for {} weights", s.len(), self.a.len());
        let mut d: i64 = 0;
        for (i, (&ai, &si)) in self.a.iter().zip(s.iter()).enumerate() {
            assert!(si == 1 || si == -1, "spin {i} is {si}; spins are +1 or -1");
            d += i64::from(si) * ai as i64;
        }
        d
    }

    /// `|sum_i a_i s_i|`.
    ///
    /// # Panics
    ///
    /// As [`Instance::signed_sum`].
    #[must_use]
    pub fn discrepancy(&self, s: &[i8]) -> u64 {
        self.signed_sum(s).unsigned_abs()
    }

    /// `sum_i a_i^2` — the constant the Ising energy sits below the squared discrepancy.
    ///
    /// Exact in `f64`: it is an integer no larger than `total^2 <= 2^53`.
    #[must_use]
    pub fn offset(&self) -> f64 {
        self.a.iter().map(|&x| (x * x) as f64).sum()
    }

    /// `E(s) = D(s)^2 - sum_i a_i^2`, without building the graph.
    ///
    /// # Panics
    ///
    /// As [`Instance::signed_sum`].
    #[must_use]
    pub fn energy(&self, s: &[i8]) -> f64 {
        let d = self.discrepancy(s);
        (d * d) as f64 - self.offset()
    }

    /// The fully connected Ising model with `J_ij = -2 a_i a_j` and no fields.
    ///
    /// `n(n-1)/2` edges, so this is quadratic in the instance size — which is the shape of the
    /// problem, not a choice: number partitioning has no sparse Ising form, every weight interacts
    /// with every other.
    #[must_use]
    pub fn graph(&self) -> Graph {
        let n = self.a.len();
        let mut b = GraphBuilder::new(n);
        for i in 0..n {
            for j in (i + 1)..n {
                b.couple(i, j, -2.0 * self.a[i] as f64 * self.a[j] as f64);
            }
        }
        b.build()
    }

    /// A discrepancy no partition can beat, from the two facts available without searching.
    ///
    /// The heaviest weight has to sit somewhere and everything else can at best cancel it, so
    /// `D >= 2 a_max - total`; and every `D` carries the parity of the total, so
    /// `D >= total mod 2`. The bound is the larger of the two.
    ///
    /// Integer arithmetic throughout, so it is exact rather than directed — see the module note on
    /// why nothing here goes through [`crate::round`]. A partition meeting this bound is optimal,
    /// and needs no search to prove it.
    #[must_use]
    pub fn lower_bound(&self) -> u64 {
        let max = self.a.iter().copied().max().unwrap_or(0);
        let dominance = (2 * max).saturating_sub(self.total);
        dominance.max(self.parity())
    }

    /// The Karmarkar-Karp differencing heuristic (1982).
    ///
    /// Repeatedly take the two largest remaining numbers and replace them with their difference.
    /// Differencing `a` and `b` is a COMMITMENT that those two land in opposite sets without saying
    /// which is which, so the run builds a spanning tree of such commitments; two-colouring it at
    /// the end turns the last remaining number into an actual partition achieving exactly that
    /// discrepancy.
    ///
    /// `O(n log n)`, and much stronger than greedy: on `n` weights uniform on the unit interval
    /// greedy's expected discrepancy decays like `n^-1` and differencing's like `n^(-c log n)`
    /// (Karmarkar, Karp, Lueker and Odlyzko 1986 — quoted, not measured here). It is not exact — it is
    /// optimal for `n <= 3` and loses on `{8, 7, 6, 5, 4}`, where the optimum is `0` and
    /// differencing returns `2`.
    ///
    /// Ties are broken by the lower index, so the result is deterministic.
    ///
    /// # Panics
    ///
    /// Never, and the invariant is the loop's own: the body runs only while two entries remain and
    /// each pass removes two and pushes one, so the two pops inside it always find something and
    /// exactly one entry survives for the pop after it. An [`Instance`] is never empty — the
    /// constructor refuses that — so there is always at least that one.
    #[must_use]
    pub fn karmarkar_karp(&self) -> Partition {
        use core::cmp::Reverse;
        use std::collections::BinaryHeap;
        let n = self.a.len();
        // (value, Reverse(representative)): the max-heap pops the largest value, and among equal
        // values the SMALLEST index, which is what makes this deterministic rather than a function
        // of heap internals.
        let mut heap: BinaryHeap<(u64, Reverse<usize>)> =
            self.a.iter().enumerate().map(|(i, &v)| (v, Reverse(i))).collect();
        // Each differencing step is an "opposite sets" edge between two representatives. n-1 steps
        // over n items, each joining two distinct components, so this is a spanning tree.
        let mut opposite: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut root = 0usize;
        while heap.len() >= 2 {
            let (va, Reverse(pa)) = heap.pop().expect("len >= 2");
            let (vb, Reverse(pb)) = heap.pop().expect("len >= 2");
            opposite[pa].push(pb);
            opposite[pb].push(pa);
            heap.push((va - vb, Reverse(pa)));
            root = pa;
        }
        let (discrepancy, _) = heap.pop().expect("one component remains");

        // Two-colour the commitment tree from the surviving representative. The invariant the
        // differencing maintains is that a representative's value is the signed sum of its own
        // component with the representative positive, so rooting at `root` with +1 reproduces the
        // reported discrepancy with the correct sign.
        let mut signs = vec![0i8; n];
        signs[root] = 1;
        let mut stack = vec![root];
        while let Some(v) = stack.pop() {
            for &u in &opposite[v] {
                if signs[u] == 0 {
                    signs[u] = -signs[v];
                    stack.push(u);
                }
            }
        }
        Partition { signs, discrepancy }
    }

    /// The true optimum, by enumerating every sign assignment.
    ///
    /// The global flip `s -> -s` leaves the discrepancy alone, so the first spin is pinned to `+1`
    /// and only `2^(n-1)` states are visited, in Gray-code order so each step moves one weight and
    /// updates the signed sum in constant time.
    ///
    /// # Errors
    ///
    /// [`PartitionError::TooManyToEnumerate`] past [`MAX_ENUMERABLE`] weights.
    pub fn exact_optimum(&self) -> Result<Partition, PartitionError> {
        let n = self.a.len();
        if n > MAX_ENUMERABLE {
            return Err(PartitionError::TooManyToEnumerate { n, limit: MAX_ENUMERABLE });
        }
        let mut signs = vec![1i8; n];
        let mut sum = self.total as i64;
        let mut best = sum.unsigned_abs();
        let mut best_signs = signs.clone();
        for step in 1u64..(1u64 << (n - 1)) {
            // Gray code: consecutive codes differ in exactly the bit `step` ends in.
            let k = step.trailing_zeros() as usize + 1;
            signs[k] = -signs[k];
            sum += 2 * i64::from(signs[k]) * self.a[k] as i64;
            let d = sum.unsigned_abs();
            if d < best {
                best = d;
                best_signs.copy_from_slice(&signs);
            }
        }
        Ok(Partition { signs: best_signs, discrepancy: best })
    }
}

/// Mertens' family: `n` weights drawn uniformly from `{1, ..., 2^bits}`.
///
/// `bits` is the knob the transition lives on — see [`kappa`]. The draw masks `bits` bits out of a
/// 64-bit word, which is exact rather than modulo-biased because `2^bits` divides `2^64`.
///
/// # Errors
///
/// [`PartitionError::Empty`] for `n = 0`, [`PartitionError::TooManyBits`] past [`MAX_BITS`], and
/// [`PartitionError::TooHeavy`] when the drawn weights total past [`MAX_TOTAL`] — which is the
/// binding constraint in practice, and caps `bits` near 22 for the sizes that can be enumerated.
pub fn uniform(n: usize, bits: u32, seed: u64) -> Result<Instance, PartitionError> {
    if n == 0 {
        return Err(PartitionError::Empty);
    }
    if bits > MAX_BITS {
        return Err(PartitionError::TooManyBits { bits, limit: MAX_BITS });
    }
    let mut rng = Pcg::new(seed, 0x9A_0000);
    let mask = (1u64 << bits) - 1;
    let w: Vec<u64> = (0..n).map(|_| 1 + (rng.next_u64() & mask)).collect();
    Instance::new(&w)
}

/// The control parameter `kappa = m/n`: bits per weight over weight count.
///
/// Mertens 1998. An instance is information-theoretically over-determined above `kappa = 1` —
/// there are `2^(n-1)` distinct partitions and the discrepancy is spread over about `2^m sqrt(n)`
/// values, so the expected number of perfect partitions is exponentially large below one and
/// exponentially small above it. Hardness follows the same curve: below the transition an answer
/// is easy to find because there are so many of them.
///
/// # Panics
///
/// If `n` is zero, where the parameter is undefined rather than infinite.
#[must_use]
pub fn kappa(n: usize, bits: u32) -> f64 {
    assert!(n > 0, "kappa = m/n is undefined with no weights");
    f64::from(bits) / n as f64
}

/// Where the transition sits for a FINITE `n`, in units of `kappa`.
///
/// `kappa_c = 1` holds only in the limit. Set the expected number of perfect partitions to one:
/// there are `2^(n-1)` distinct partitions, the discrepancy is Gaussian with variance
/// `n (2^m)^2 / 12` and lives on a lattice of spacing two, so the count is
/// `2^(n-m) sqrt(6 / (pi n))` and passes one at
///
/// ```text
///     m_c = n - log2(n)/2 + log2(6/pi)/2
/// ```
///
/// **THE CONSTANT IS CARRIED ON PURPOSE, AND IT IS THE MEASURED DIFFERENCE BETWEEN A GOOD
/// PREDICTION AND A BAD ONE.** The form usually quoted, `kappa_c = 1 - log2(n)/(2n)`, drops it.
/// Measured against where the perfect fraction actually crosses one half (200 instances per point,
/// optimum by enumeration), in bits of `m`:
///
/// | `n` | crossing | this formula | dropping `log2(6/pi)/2` |
/// |---|---|---|---|
/// | 8 | 6.87 | 6.97 | 6.50 |
/// | 10 | 8.78 | 8.81 | 8.34 |
/// | 12 | 10.51 | 10.67 | 10.21 |
/// | 14 | 12.57 | 12.56 | 12.10 |
/// | 16 | 14.71 | 14.47 | 14.00 |
/// | 18 | 16.60 | 16.38 | 15.92 |
///
/// Keeping the constant is within `+/-0.25` bits with no sign to the error; dropping it is low by
/// 0.3 to 0.7 bits at every single size, which is the signature of a missing term rather than of
/// noise. Two `O(1)` effects are NOT modelled here and partly cancel inside that quarter bit: an
/// odd total has two perfect discrepancies (`+1` and `-1`) where an even total has one, and
/// perfect partitions are correlated rather than Poisson, so the point where the expected count
/// reaches one is not exactly the point where half of instances have one.
///
/// # Panics
///
/// If `n` is zero.
#[must_use]
pub fn critical_kappa(n: usize) -> f64 {
    assert!(n > 0, "a transition point needs an instance size");
    let n = n as f64;
    1.0 - (n.log2() - (6.0 / core::f64::consts::PI).log2()) / (2.0 * n)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::oracle::{Exhaustive, Solver};

    /// Every one of the `2^n` sign assignments, in mask order.
    fn states(n: usize) -> impl Iterator<Item = Vec<i8>> {
        (0..(1u64 << n))
            .map(move |mask| (0..n).map(|i| if mask >> i & 1 == 1 { 1i8 } else { -1 }).collect())
    }

    /// Fraction of instances with a perfect partition, by exhaustive enumeration.
    fn perfect_fraction(n: usize, bits: u32, seeds: u64) -> f64 {
        let mut perfect = 0u64;
        for seed in 1..=seeds {
            let inst = uniform(n, bits, seed)
                .unwrap_or_else(|e| panic!("n={n} m={bits} seed={seed}: {e}"));
            let opt = inst.exact_optimum().expect("enumerable");
            if inst.is_perfect(opt.discrepancy()) {
                perfect += 1;
            }
        }
        perfect as f64 / seeds as f64
    }

    /// Where the perfect fraction crosses one half, interpolated in bits of `m`.
    fn crossing(n: usize, first_bit: u32, last_bit: u32, seeds: u64) -> f64 {
        let mut prev: Option<(u32, f64)> = None;
        for m in first_bit..=last_bit {
            let f = perfect_fraction(n, m, seeds);
            if let Some((pm, pf)) = prev
                && pf >= 0.5
                && f < 0.5
            {
                return f64::from(pm) + (pf - 0.5) / (pf - f);
            }
            prev = Some((m, f));
        }
        panic!("n={n}: the perfect fraction never crossed 1/2 over m in {first_bit}..={last_bit}");
    }

    /// THE REDUCTION, PROVED STATE BY STATE AGAINST INTEGER ARITHMETIC THAT NEVER SEES THE GRAPH.
    ///
    /// The oracle is exhaustive enumeration: for all `2^n` assignments, the squared discrepancy is
    /// recomputed from the weights in `i64`, and the Ising energy must equal it minus
    /// `sum_i a_i^2` — with `==`, because with integer weights under [`MAX_TOTAL`] both sides are
    /// exact integers in `f64` and a tolerance here would be an excuse rather than a margin.
    #[test]
    fn ising_energy_equals_the_squared_discrepancy_on_every_state() {
        let instances = [
            // one weight: no edges at all, and the identity still has to hold
            Instance::new(&[5]).unwrap(),
            Instance::new(&[8, 7, 6, 5]).unwrap(),
            Instance::new(&[1, 1, 1, 1, 1, 1, 1]).unwrap(),
            // deliberately lopsided: one weight no partition can balance
            Instance::new(&[100, 3, 2, 1]).unwrap(),
            // at the exactness ceiling, where the mantissa has nothing to spare
            Instance::new(&[23726566, 23726566, 23726566, 23726567]).unwrap(),
            uniform(10, 12, 7).unwrap(),
            uniform(12, 20, 99).unwrap(),
        ];
        for inst in &instances {
            let g = inst.graph();
            let n = inst.n();
            assert_eq!(g.n, n);
            assert_eq!(g.n_edges, n * (n - 1) / 2, "the partitioning model is fully connected");
            assert!(g.h.iter().all(|&h| h == 0.0), "number partitioning has no fields");
            let c = inst.offset();
            for s in states(n) {
                let d = inst.discrepancy(&s);
                let squared = (d * d) as f64;
                assert_eq!(
                    g.energy(&s) + c,
                    squared,
                    "weights {:?} state {s:?}: E {} + offset {c} is not D^2 = {d}^2",
                    inst.weights(),
                    g.energy(&s)
                );
                assert_eq!(inst.energy(&s), g.energy(&s), "the graph-free energy must agree");
            }
        }
    }

    /// THE ASYMMETRIC HALF OF THE REDUCTION: the halved coupling ranks identically and is WRONG.
    ///
    /// `J_ij = -a_i a_j` is the form quoted wherever the expansion is written over ordered pairs.
    /// Under this crate's each-edge-once energy it gives exactly half the squared discrepancy, so
    /// every ground state, every ordering and every solver comparison is unchanged — and the
    /// energy is not the quantity it is named after. A test that only checked "the optimum is the
    /// best partition" would pass on both.
    #[test]
    fn the_halved_coupling_is_not_the_squared_discrepancy() {
        let inst = Instance::new(&[8, 7, 6, 5]).unwrap();
        let n = inst.n();
        let mut b = GraphBuilder::new(n);
        for i in 0..n {
            for j in (i + 1)..n {
                b.couple(i, j, -(inst.weights()[i] as f64) * inst.weights()[j] as f64);
            }
        }
        let halved = b.build();
        let right = inst.graph();
        let c = inst.offset();

        let up = vec![1i8; n];
        assert_eq!(inst.discrepancy(&up), 26);
        assert_eq!(c, 174.0, "8^2 + 7^2 + 6^2 + 5^2");
        assert_eq!(right.energy(&up), 502.0, "26^2 - 174");
        assert_eq!(halved.energy(&up), 251.0, "half of it");

        let mut wrong = 0;
        let mut best_right = f64::INFINITY;
        let mut best_halved = f64::INFINITY;
        let (mut arg_right, mut arg_halved) = (Vec::new(), Vec::new());
        for s in states(n) {
            let d = inst.discrepancy(&s);
            let squared = (d * d) as f64;
            assert_eq!(right.energy(&s) + c, squared);
            assert_eq!(2.0 * halved.energy(&s), right.energy(&s), "it is exactly half, always");
            if halved.energy(&s) + c != squared {
                wrong += 1;
            }
            if right.energy(&s) < best_right {
                best_right = right.energy(&s);
                arg_right = s.clone();
            }
            if halved.energy(&s) < best_halved {
                best_halved = halved.energy(&s);
                arg_halved = s.clone();
            }
        }
        assert_eq!(wrong, 16, "the halved form misses the squared discrepancy on every state");
        assert_eq!(arg_right, arg_halved, "and yet it picks the same ground state, which is why");
    }

    /// The optimum found by enumeration is the optimum [`crate::oracle::Exhaustive`] finds on the
    /// graph — and the two arrive by different arithmetic.
    ///
    /// This module minimises `|sum a_i s_i|` over `2^(n-1)` states in `i64`, pinning the first
    /// spin; `Exhaustive` minimises the CSR float energy over all `2^n`. Agreement on the value
    /// AND on the state is what says the reduction and the shortcut are the same problem.
    #[test]
    fn exact_optimum_matches_oracle_exhaustive_through_the_reduction() {
        for (n, bits, seed) in [(8usize, 6u32, 1u64), (10, 10, 2), (12, 8, 3), (12, 18, 4), (11, 22, 5)] {
            let inst = uniform(n, bits, seed).unwrap();
            let opt = inst.exact_optimum().unwrap();
            let (state, e) = Exhaustive.solve(&inst.graph());
            let d = opt.discrepancy();
            assert_eq!(
                e + inst.offset(),
                (d * d) as f64,
                "n={n} m={bits} seed={seed}: exhaustive energy {e} is not D^2 - offset for D={d}"
            );
            assert_eq!(
                inst.discrepancy(&state),
                d,
                "n={n} m={bits} seed={seed}: the exhaustive ground state has a different discrepancy"
            );
            assert_eq!(
                inst.discrepancy(opt.signs()),
                d,
                "the returned signs must realise the discrepancy that was reported"
            );
            assert!(d >= inst.lower_bound());
        }
    }

    /// THE PUBLISHED PHYSICS, MEASURED: perfect partitions collapse across `kappa = 1`.
    ///
    /// Mertens 1998. The oracle is not this crate — it is the transition itself, and the quantity
    /// is measured by exhaustive enumeration of every sign assignment, so nothing in the
    /// measurement depends on the Ising reduction or on any solver being any good.
    ///
    /// Three separate claims, and the third is the one that could not pass by accident:
    /// the fraction is one far below, zero far above, never increases in between, and the half-way
    /// crossing follows [`critical_kappa`] across six instance sizes.
    #[test]
    fn perfect_partitions_collapse_across_kappa_one_by_exact_enumeration() {
        let n = 14;
        // far below and far above, at the ends the exactness ceiling allows
        let low = perfect_fraction(n, 7, 200);
        let high = perfect_fraction(n, 21, 200);
        assert!(low >= 0.98, "kappa = {}: perfect fraction {low}", kappa(n, 7));
        assert!(high <= 0.02, "kappa = {}: perfect fraction {high}", kappa(n, 21));

        // and the whole ladder in between never goes back up
        let ladder: Vec<(u32, f64)> =
            [7u32, 10, 12, 13, 14, 16, 18, 21].iter().map(|&m| (m, perfect_fraction(n, m, 200))).collect();
        for w in ladder.windows(2) {
            assert!(
                w[1].1 <= w[0].1,
                "m={} gave {} and m={} gave {}, which is not a transition",
                w[0].0,
                w[0].1,
                w[1].0,
                w[1].1
            );
        }
        // kappa = 1 exactly is already past the half-way point at this size
        let at_one = ladder.iter().find(|(m, _)| *m == 14).unwrap().1;
        assert!(at_one < 0.5, "kappa = 1 should be past the crossing at n = 14, got {at_one}");

        // the crossing follows the finite-size law over six sizes
        for n in [8usize, 10, 12, 14, 16, 18] {
            let predicted = critical_kappa(n) * n as f64;
            let first = (predicted.floor() as u32).saturating_sub(2);
            let measured = crossing(n, first, first + 5, 200);
            assert!(
                (measured - predicted).abs() < 0.4,
                "n={n}: the perfect fraction crosses 1/2 at m={measured:.3}, \
                 and critical_kappa says {predicted:.3}"
            );
        }
    }

    /// Differencing never beats the optimum, and never ties it either.
    ///
    /// The oracle is exhaustive enumeration. Both halves are asserted: an implementation that
    /// quietly returned the exact optimum would pass "never beats" and fail the loss count, and
    /// one whose signs did not match its own reported discrepancy would pass both and fail the
    /// realisation check.
    #[test]
    fn karmarkar_karp_never_beats_exact_enumeration() {
        let mut losses = 0;
        let mut total = 0;
        for (n, bits) in [(8usize, 6u32), (10, 10), (12, 12), (14, 16), (14, 21)] {
            for seed in 1..=40u64 {
                let inst = uniform(n, bits, seed).unwrap();
                let kk = inst.karmarkar_karp();
                let opt = inst.exact_optimum().unwrap();
                total += 1;
                assert!(
                    kk.discrepancy() >= opt.discrepancy(),
                    "n={n} m={bits} seed={seed}: differencing returned {} against an optimum of {}",
                    kk.discrepancy(),
                    opt.discrepancy()
                );
                assert!(kk.discrepancy() >= inst.lower_bound());
                // the signs it hands back must be the partition it is claiming
                assert_eq!(
                    inst.signed_sum(kk.signs()),
                    kk.discrepancy() as i64,
                    "n={n} m={bits} seed={seed}: the tree was two-coloured against its own root"
                );
                // and the two sets must be a partition of the weights
                let (up, down) = kk.sets();
                assert_eq!(up.len() + down.len(), n);
                let sum = |v: &[usize]| v.iter().map(|&i| inst.weights()[i]).sum::<u64>();
                assert_eq!(sum(&up) + sum(&down), inst.total());
                assert_eq!(sum(&up).abs_diff(sum(&down)), kk.discrepancy());
                if kk.discrepancy() > opt.discrepancy() {
                    losses += 1;
                }
            }
        }
        assert!(
            losses > total / 4,
            "only {losses} of {total} instances beat differencing; a heuristic that never loses \
             to enumeration is not a heuristic"
        );
    }

    /// Differencing is EXACT where it is known to be, and loses where it is known to lose.
    ///
    /// Exact for `n <= 3` is provable: with `a >= b >= c` the optimum is
    /// `min(|a-b-c|, a+c-b)` and differencing returns `|a-b-c|`, which is never the larger.
    /// Checked here over every triple with weights in `1..=12`, against enumeration.
    ///
    /// Measured beyond that: differencing is also optimal on all 20,736 four-tuples in the same
    /// range, and the smallest instance where it loses is `{3, 3, 2, 2, 2}` — 189 of the 2,002
    /// sorted five-tuples with weights in `1..=10`. `{8, 7, 6, 5, 4}` is the textbook example:
    /// `8 + 7 = 6 + 5 + 4`, so the optimum is 0, and differencing returns 2.
    #[test]
    fn karmarkar_karp_is_exact_on_every_triple_and_loses_on_a_known_five() {
        for a in 1..=12u64 {
            for b in 1..=12u64 {
                for c in 1..=12u64 {
                    let inst = Instance::new(&[a, b, c]).unwrap();
                    assert_eq!(
                        inst.karmarkar_karp().discrepancy(),
                        inst.exact_optimum().unwrap().discrepancy(),
                        "differencing is optimal for three numbers, and was not on {a} {b} {c}"
                    );
                }
            }
        }
        let mut four_losses = 0;
        for a in 1..=12u64 {
            for b in 1..=12u64 {
                for c in 1..=12u64 {
                    for d in 1..=12u64 {
                        let inst = Instance::new(&[a, b, c, d]).unwrap();
                        if inst.karmarkar_karp().discrepancy()
                            > inst.exact_optimum().unwrap().discrepancy()
                        {
                            four_losses += 1;
                        }
                    }
                }
            }
        }
        assert_eq!(four_losses, 0, "measured: no four-tuple in 1..=12 beats differencing");

        // and five is where it breaks, on the textbook instance and on the smallest one
        for (w, kk_want, opt_want) in
            [([8u64, 7, 6, 5, 4], 2u64, 0u64), ([3, 3, 2, 2, 2], 2, 0)]
        {
            let inst = Instance::new(&w).unwrap();
            assert_eq!(inst.karmarkar_karp().discrepancy(), kk_want, "differencing on {w:?}");
            assert_eq!(inst.exact_optimum().unwrap().discrepancy(), opt_want, "optimum of {w:?}");
        }
        let mut five_losses = 0;
        let mut five_total = 0;
        for a in 1..=10u64 {
            for b in 1..=a {
                for c in 1..=b {
                    for d in 1..=c {
                        for e in 1..=d {
                            let inst = Instance::new(&[a, b, c, d, e]).unwrap();
                            five_total += 1;
                            if inst.karmarkar_karp().discrepancy()
                                > inst.exact_optimum().unwrap().discrepancy()
                            {
                                five_losses += 1;
                            }
                        }
                    }
                }
            }
        }
        assert_eq!((five_losses, five_total), (189, 2002), "measured over sorted five-tuples");
    }

    /// The lower bound is a bound, and it is not vacuous.
    ///
    /// Checked against exhaustive enumeration: never above the optimum, tight where one weight
    /// dominates, and strictly slack where nothing does — a bound that were always tight would be
    /// an optimum, and one always slack would be the parity alone.
    #[test]
    fn the_lower_bound_never_exceeds_the_exact_optimum() {
        let mut tight = 0;
        let mut slack = 0;
        for (n, bits) in [(6usize, 3u32), (8, 6), (10, 10), (12, 16)] {
            for seed in 1..=30u64 {
                let inst = uniform(n, bits, seed).unwrap();
                let opt = inst.exact_optimum().unwrap().discrepancy();
                assert!(
                    inst.lower_bound() <= opt,
                    "n={n} m={bits} seed={seed}: bound {} above the optimum {opt}",
                    inst.lower_bound()
                );
                if inst.lower_bound() == opt {
                    tight += 1;
                } else {
                    slack += 1;
                }
            }
        }
        assert!(tight > 0 && slack > 0, "tight {tight}, slack {slack}");

        // dominance: nothing can balance the 100, so the bound is the answer
        let dom = Instance::new(&[100, 3, 2, 1]).unwrap();
        assert_eq!(dom.lower_bound(), 94);
        assert_eq!(dom.exact_optimum().unwrap().discrepancy(), 94);
        // parity: an odd total can never be split evenly
        let odd = Instance::new(&[4, 4, 4, 1]).unwrap();
        assert_eq!(odd.lower_bound(), 1);
        assert_eq!(odd.exact_optimum().unwrap().discrepancy(), 3, "and here the bound is slack");
        assert!(odd.is_perfect(1) && !odd.is_perfect(3));
    }

    /// The exactness ceiling is real arithmetic, not a bureaucratic limit.
    ///
    /// At the ceiling every energy is still an exact integer. One unit past it the instance is
    /// refused — and the same model built by hand from weights well past it reports an energy that
    /// is WRONG, which is what the refusal is protecting.
    #[test]
    fn the_exactness_ceiling_is_where_the_identity_actually_breaks() {
        let inst = Instance::new(&[23726566, 23726566, 23726566, 23726567]).unwrap();
        assert_eq!(inst.total(), MAX_TOTAL);
        let g = inst.graph();
        for s in states(4) {
            let d = inst.discrepancy(&s);
            assert_eq!(g.energy(&s) + inst.offset(), (d * d) as f64);
            assert_eq!(g.energy(&s).fract(), 0.0, "an exact integer, not nearly one");
        }
        assert_eq!(
            Instance::new(&[MAX_TOTAL, 1]),
            Err(PartitionError::TooHeavy { total: u128::from(MAX_TOTAL) + 1, limit: MAX_TOTAL })
        );

        // past the ceiling, by hand, with the same couplings this module would have used
        let big = [(1u64 << 28) + 1, (1 << 28) + 3, (1 << 28) + 5, (1 << 28) + 7];
        let mut b = GraphBuilder::new(4);
        for i in 0..4 {
            for j in (i + 1)..4 {
                b.couple(i, j, -2.0 * big[i] as f64 * big[j] as f64);
            }
        }
        let huge = b.build();
        let up = vec![1i8; 4];
        let d: i128 = big.iter().map(|&x| i128::from(x)).sum();
        let c: i128 = big.iter().map(|&x| i128::from(x) * i128::from(x)).sum();
        let exact = d * d - c;
        let got = huge.energy(&up) as i128;
        assert_ne!(
            got, exact,
            "if this ever agrees, the ceiling can be raised; it disagreed by {} when written",
            (got - exact).abs()
        );
    }

    /// [`MAX_TOTAL`] is the largest total whose square the mantissa holds, and one more is not.
    #[test]
    fn max_total_is_the_largest_total_whose_square_is_exact() {
        let m = u128::from(MAX_TOTAL);
        let limit = 1u128 << 53;
        assert!(m * m <= limit, "{} above 2^53", m * m);
        assert!((m + 1) * (m + 1) > limit, "MAX_TOTAL could be larger");
        // and an integer that size really does survive the round trip
        assert_eq!((m * m) as f64 as u128, m * m);
    }

    /// Every refusal names what it saw, and the family is reproducible from its seed.
    #[test]
    fn refusals_name_what_was_seen_and_draws_are_reproducible() {
        assert_eq!(Instance::new(&[]), Err(PartitionError::Empty));
        assert_eq!(Instance::new(&[3, 0, 1]), Err(PartitionError::ZeroWeight { index: 1 }));
        assert_eq!(uniform(0, 4, 1), Err(PartitionError::Empty));
        assert_eq!(
            uniform(4, 64, 1),
            Err(PartitionError::TooManyBits { bits: 64, limit: MAX_BITS })
        );
        assert!(matches!(uniform(64, 30, 1), Err(PartitionError::TooHeavy { .. })));
        assert_eq!(
            Instance::new(&[1; 30]).unwrap().exact_optimum(),
            Err(PartitionError::TooManyToEnumerate { n: 30, limit: MAX_ENUMERABLE })
        );
        let said = PartitionError::ZeroWeight { index: 7 }.to_string();
        assert!(said.contains('7'), "{said}");
        let said = PartitionError::TooHeavy { total: 99, limit: MAX_TOTAL }.to_string();
        assert!(said.contains("99") && said.contains("94906265"), "{said}");
        let said = PartitionError::TooManyToEnumerate { n: 40, limit: 27 }.to_string();
        assert!(said.contains("2^39") && said.contains("27"), "{said}");

        // deterministic by seed, different across seeds, and inside the declared range
        assert_eq!(uniform(20, 11, 5).unwrap(), uniform(20, 11, 5).unwrap());
        assert_ne!(uniform(20, 11, 5).unwrap(), uniform(20, 11, 6).unwrap());
        // `{1, ..., 2^bits}` means both ends are drawable, and neither 0 nor 2^bits + 1 is
        let inst = uniform(2000, 8, 3).unwrap();
        assert_eq!(inst.weights().iter().copied().min(), Some(1));
        assert_eq!(inst.weights().iter().copied().max(), Some(256));
        assert_eq!(uniform(50, 0, 3).unwrap().weights(), [1u64; 50], "zero bits is the range {{1}}");
        assert_eq!(kappa(20, 10), 0.5);
    }

    /// A spin that is neither `+1` nor `-1` is an unreadable input, not a half-assignment.
    #[test]
    #[should_panic(expected = "spin 1 is 0")]
    fn a_spin_of_zero_is_refused_rather_than_read_as_a_set() {
        let inst = Instance::new(&[3, 4, 5]).unwrap();
        let _ = inst.discrepancy(&[1, 0, -1]);
    }

    /// One spin per weight, or nothing.
    #[test]
    #[should_panic(expected = "2 spins for 3 weights")]
    fn a_state_of_the_wrong_length_is_refused() {
        let inst = Instance::new(&[3, 4, 5]).unwrap();
        let _ = inst.discrepancy(&[1, -1]);
    }
}
