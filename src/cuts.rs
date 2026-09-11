//! Triangle inequalities and cutting planes over the metric polytope.
//!
//! Every bound in this crate so far works in the SPINS. This one works in the PAIRS, which is the
//! space the exact max-cut literature has worked in since Barahona & Mahjoub: put
//!
//! ```text
//!     x_ij = (1 - s_i s_j) / 2,     so x_ij = 1 exactly when i and j land on opposite sides
//! ```
//!
//! and the energy becomes linear, `E(s) = const + Σ c_ij x_ij` with `c_ij = 2 J_ij`. The hard part
//! moves entirely into the feasible set: the `x` reachable from an actual state form the **cut
//! polytope**, which has exponentially many facets and no compact description. A relaxation is a
//! choice of how much of that polytope to describe.
//!
//! The cheapest choice is the box, `0 ≤ x ≤ 1`, and it is exactly [`crate::bound::decoupled`] —
//! every term at its own minimum, every disagreement between terms ignored. The next choice, and
//! the one the exact solvers are built on, is the **metric polytope**: add, for every triple,
//!
//! ```text
//!     x_ij + x_ik + x_jk ≤ 2        three nodes cannot be pairwise separated by two sides
//!     x_jk ≤ x_ij + x_ik            separation is a metric: it obeys the triangle inequality
//!     x_ik ≤ x_ij + x_jk
//!     x_ij ≤ x_ik + x_jk
//! ```
//!
//! There are `4·C(n,3)` of these and almost all of them are slack at any given point, so they are
//! not written down. They are **separated**: evaluate the current point, find the ones it breaks,
//! add those, re-solve, repeat. That loop is the cutting-plane method, and [`separate`] plus
//! [`cutting_plane`] are the two halves of it.
//!
//! # Why this module exists
//!
//! `branch` prunes with [`crate::bound::decoupled`], `bound::odd_cycle` adds what frustrated cycles
//! must cost, and `sdp` certifies a semidefinite dual. None of them is the polyhedral relaxation
//! that `BiqMac` — the reference exact max-cut code of Rendl, Rinaldi & Wiegele — actually runs, and
//! whose entire story is the SDP *intersected with* triangle inequalities. The words "triangle
//! inequality" and "cutting plane" appeared nowhere in this crate before this file.
//!
//! # Soundness: why any multiplier vector at all gives a valid bound
//!
//! The relaxation is not solved by an LP solver, because an LP solver's answer is a float and a
//! float is not a proof. It is solved in the **dual**, by Lagrangian relaxation: with multipliers
//! `λ_t ≥ 0` on the triangle rows `a_tᵀx ≤ b_t`,
//!
//! ```text
//!     L(λ)  =  const  +  min_{0 ≤ x ≤ 1} [ (c + Σ_t λ_t a_t)ᵀ x ]  −  Σ_t λ_t b_t
//! ```
//!
//! is a lower bound on `min_s E(s)` for **every** `λ ≥ 0` — no convergence, no feasibility, no
//! trust in the ascent required, because a cut vector is feasible for every row and so only ever
//! adds a non-positive quantity to its own energy. The inner minimisation separates: each variable
//! goes to `1` where its reduced cost is negative and `0` otherwise. Subgradient ascent on `λ` then
//! drives `L` up to the LP value, and every iterate along the way is a bound in its own right.
//!
//! By LP duality the ceiling of that ascent is exactly the LP over the box intersected with the
//! rows that have been added, so with all triples present it is the metric-polytope bound itself.
//!
//! **Every accumulation here goes through [`crate::round::sum_down`]**, including the reduced costs:
//! `min(0, ·)` is monotone, so a reduced cost rounded DOWN keeps the whole bound rounded down. This
//! crate has shipped a "lower bound" that sat above the optimum once (see [`crate::round`]), and
//! the sweep test here is written to see a violation of `1e-13`.
//!
//! # What triangle inequalities do and do not close, stated with the numbers
//!
//! `CUT(K_n) = MET(K_n)` for `n ≤ 4` and for no larger `n` (Barahona & Mahjoub 1986). So:
//!
//! | instance | box | + triangles | exact |
//! |---|---|---|---|
//! | `K_4`, unit max-cut weights | −6 | **−2** | −2 |
//! | `C_5`, unit max-cut weights | −5 | **−3** | −3 |
//! | `K_5`, unit max-cut weights | −10 | **−10/3** | −2 |
//! | `K_n`, unit max-cut weights | −n(n−1)/2 | **−n(n−1)/6** | −C(n,2) + 2⌊n²/4⌋ |
//!
//! The `K_n` row is a closed form, not a measurement: every pair lies in `n−2` triples, so summing
//! the perimeter row over all `C(n,3)` of them gives `(n−2) Σx ≤ 2 C(n,3)`, hence `Σx ≤ n(n−1)/3`,
//! and `x ≡ 2/3` is feasible and attains it. The bound is therefore exactly `−n(n−1)/6` for every
//! `n`, which the test checks from both sides at `n = 3…8` (it lands within `1e-14`).
//!
//! `K_5` is the standard counterexample and it is worth being exact about, because the folklore
//! that triangle inequalities "close" `K_5` is false and the error is easy to launder through a
//! test that only looks at the tightened side: the closed form gives `−10/3` against a true
//! optimum of `−2`. Five nodes is the smallest case with `CUT ⊊ MET`, and the gap grows from
//! there.
//!
//! `C_5` is closed because the triangle inequalities of the complete graph imply every cycle
//! inequality, and for a graph with no `K_5` minor the cycle inequalities describe the cut polytope
//! (Barahona 1983, Barahona & Mahjoub 1986).
//!
//! ```
//! use ferrotherm::{cuts, graph::GraphBuilder};
//!
//! // A 5-cycle of antiferromagnetic couplings: unit max-cut weights on an odd cycle.
//! let mut b = GraphBuilder::new(5);
//! for i in 0..5 {
//!     b.couple(i, (i + 1) % 5, -1.0);
//! }
//! let g = b.build();
//!
//! let t = cuts::cutting_plane(&g, &cuts::Params::default()).unwrap();
//! assert!(t.base <= -5.0, "the box relaxation gives up every edge: {}", t.base);
//! assert!(t.bound.value > -3.0001, "triangles reach the optimum: {}", t.bound.value);
//! assert!(t.bound.value <= -3.0, "and never pass it");
//! ```
//!
//! # References
//!
//! * F. Barahona and A. R. Mahjoub, "On the cut polytope", *Mathematical Programming* **36** (1986)
//!   157–173 — the triangle inequalities, their facet status, and `CUT = MET` exactly for `n ≤ 4`.
//! * F. Barahona, "The max-cut problem on graphs not contractible to K5", *Operations Research
//!   Letters* **2** (1983) 107–111.
//! * M. Deza and M. Laurent, *Geometry of Cuts and Metrics*, Springer 1997 — the metric polytope,
//!   its fractional vertices, and why `x ≡ 2/3` is the one that matters on five nodes.
//! * F. Rendl, G. Rinaldi and A. Wiegele, "Solving max-cut to optimality by intersecting
//!   semidefinite and polyhedral relaxations", *Mathematical Programming* **121** (2010) 307–335 —
//!   `BiqMac`, and the practice of separating triangles round by round.
//! * B. T. Polyak, "Minimization of unsmooth functionals", *USSR Computational Mathematics and
//!   Mathematical Physics* **9** (1969) 14–29 — the step rule the ascent uses, with the standard
//!   halving adjustment for an unknown optimum.

use crate::bound::Bound;
use crate::graph::Graph;
use crate::rng::Pcg;
use crate::round::sum_down;
use std::collections::BTreeSet;

/// The largest node count this module will build pair variables for.
///
/// The work is `O(n³)` per separation pass and `O(n²)` in memory, both in the **complete** graph
/// on the nodes rather than in the instance's own edges — a metric relaxation has a variable for
/// every pair whether or not that pair is coupled. At the limit that is 32,640 variables and 2.8
/// million triples per pass, which is seconds; twice the nodes is eight times the triples. A
/// caller with a bigger instance wants a different relaxation, not a longer wait, so this refuses
/// rather than crawls.
pub const MAX_NODES: usize = 256;

/// Why a cutting-plane run could not start.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CutsError {
    /// The graph, plus the gauge node its fields need, is past [`MAX_NODES`].
    TooLarge {
        /// Nodes the relaxation would have needed, gauge node included.
        nodes: usize,
        /// The limit, which is [`MAX_NODES`].
        limit: usize,
    },
    /// A coupling or field is not finite, so no bound derived from it would mean anything.
    NotFinite,
}

impl core::fmt::Display for CutsError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            CutsError::TooLarge { nodes, limit } => write!(
                f,
                "{nodes} nodes needs {} pair variables and {} triples; the limit is {limit}",
                nodes * (nodes - 1) / 2,
                nodes * (nodes - 1) * (nodes - 2) / 6
            ),
            CutsError::NotFinite => write!(f, "a coupling or field is not finite"),
        }
    }
}

impl core::error::Error for CutsError {}

/// The pair variables of a complete graph on `n` nodes, and the index each pair sits at.
///
/// A metric relaxation is indexed by unordered pairs including the ones the instance does not
/// couple — a triangle inequality on `{i, j, k}` constrains all three pairs whether or not the
/// graph has edges there — so this is the complete graph's pair set, laid out row by row:
/// `(0,1), (0,2), …, (0,n-1), (1,2), …`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pairs {
    n: usize,
}

impl Pairs {
    /// The pair set of a complete graph on `n` nodes.
    #[must_use]
    pub fn new(n: usize) -> Self {
        Pairs { n }
    }

    /// Node count.
    #[must_use]
    pub fn nodes(self) -> usize {
        self.n
    }

    /// Variable count, `n(n-1)/2`.
    #[must_use]
    pub fn len(self) -> usize {
        self.n * self.n.saturating_sub(1) / 2
    }

    /// Whether there are no pairs at all, which is every graph of fewer than two nodes.
    #[must_use]
    pub fn is_empty(self) -> bool {
        self.n < 2
    }

    /// The index of the variable for the unordered pair `{a, b}`.
    ///
    /// # Panics
    ///
    /// If `a == b`, or either is not a node. A self-pair is not a variable — `x_ii` is identically
    /// zero — and silently folding it onto some other index would corrupt a row of the relaxation.
    #[must_use]
    pub fn index(self, a: usize, b: usize) -> usize {
        assert!(a != b && a < self.n && b < self.n, "pair ({a},{b}) is not a pair of {} nodes", self.n);
        let (lo, hi) = if a < b { (a, b) } else { (b, a) };
        lo * self.n - lo * (lo + 1) / 2 + (hi - lo - 1)
    }

    /// The pair a variable index stands for, as `(lo, hi)` with `lo < hi`.
    ///
    /// # Panics
    ///
    /// If `k` is not a variable index of this pair set.
    #[must_use]
    pub fn ends(self, k: usize) -> (usize, usize) {
        let mut off = 0usize;
        for a in 0..self.n {
            let row = self.n - a - 1;
            if k < off + row {
                return (a, a + 1 + (k - off));
            }
            off += row;
        }
        panic!("pair index {k} is past the {} variables of {} nodes", self.len(), self.n);
    }
}

/// Which of the four triangle inequalities on a triple.
///
/// Four rather than one because the pair variables are a **semi-metric**: the perimeter row says
/// two sides cannot hold three mutually separated nodes, and the three apex rows are the triangle
/// inequality proper, once per choice of which node sits opposite the long side. Naming the four
/// as a closed enum rather than an index means there is no fifth value to validate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Facet {
    /// `x_ab + x_ac + x_bc ≤ 2`.
    Perimeter,
    /// `x_bc ≤ x_ab + x_ac` — the apex is the first node of the triple.
    ApexA,
    /// `x_ac ≤ x_ab + x_bc` — the apex is the second node.
    ApexB,
    /// `x_ab ≤ x_ac + x_bc` — the apex is the third node.
    ApexC,
}

impl Facet {
    /// All four, in a fixed order, so separation is one loop rather than four copies of one.
    pub const ALL: [Facet; 4] = [Facet::Perimeter, Facet::ApexA, Facet::ApexB, Facet::ApexC];

    /// Coefficients on `(x_ab, x_ac, x_bc)` and the right-hand side, as `a_t` and `b_t` of
    /// `a_tᵀx ≤ b_t`.
    ///
    /// Every entry is `0`, `±1` or `2`, so multiplying a multiplier by one of them is exact and the
    /// only rounding in the whole relaxation is in the sums — which is what makes
    /// [`crate::round::sum_down`] sufficient for soundness here.
    #[must_use]
    pub fn row(self) -> ([f64; 3], f64) {
        match self {
            Facet::Perimeter => ([1.0, 1.0, 1.0], 2.0),
            Facet::ApexA => ([-1.0, -1.0, 1.0], 0.0),
            Facet::ApexB => ([-1.0, 1.0, -1.0], 0.0),
            Facet::ApexC => ([1.0, -1.0, -1.0], 0.0),
        }
    }
}

/// One triangle inequality on one triple: a single cutting plane.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct TriangleCut {
    /// The triple, strictly increasing. [`Facet`] reads it positionally, so the order is part of
    /// the row's identity rather than a presentation choice.
    pub nodes: [u32; 3],
    /// Which of the four rows on that triple.
    pub facet: Facet,
}

impl TriangleCut {
    /// A cut on the triple `a < b < c`.
    ///
    /// # Panics
    ///
    /// If the nodes are not strictly increasing. [`Facet::ApexA`] means "the apex is the first
    /// node", so accepting an unsorted triple would silently name a different inequality.
    #[must_use]
    pub fn new(a: u32, b: u32, c: u32, facet: Facet) -> Self {
        assert!(a < b && b < c, "a triple must be strictly increasing, got ({a},{b},{c})");
        TriangleCut { nodes: [a, b, c], facet }
    }

    /// The variable indices of `(x_ab, x_ac, x_bc)`, in the order [`Facet::row`] expects.
    ///
    /// # Panics
    ///
    /// If a node of the triple is not a node of `p`.
    #[must_use]
    pub fn pair_indices(&self, p: Pairs) -> [usize; 3] {
        let [a, b, c] = self.nodes.map(|v| v as usize);
        [p.index(a, b), p.index(a, c), p.index(b, c)]
    }

    /// How far `x` breaks this inequality: `a_tᵀx − b_t`, positive when violated.
    ///
    /// Plain arithmetic, deliberately: a violation decides which row to ADD and how big a
    /// subgradient step to take, and neither of those can make a bound unsound. The bound itself
    /// is the only quantity that rounds in a direction.
    ///
    /// # Panics
    ///
    /// If `x` is not the length `p` implies.
    #[must_use]
    pub fn violation(&self, p: Pairs, x: &[f64]) -> f64 {
        assert_eq!(x.len(), p.len(), "x must carry one entry per pair");
        let idx = self.pair_indices(p);
        let (coef, rhs) = self.facet.row();
        coef[0] * x[idx[0]] + coef[1] * x[idx[1]] + coef[2] * x[idx[2]] - rhs
    }
}

/// The pair vector of a state: `x_ij = (1 − s_i s_j)/2`, which is `0` or `1` exactly.
///
/// The bridge between the two pictures, and the thing a separation routine must never flag: a
/// vector built this way is a vertex of the cut polytope and satisfies every triangle inequality.
///
/// # Panics
///
/// If `s` is not one spin per node of `p`.
#[must_use]
pub fn cut_vector(p: Pairs, s: &[i8]) -> Vec<f64> {
    assert_eq!(s.len(), p.nodes(), "one spin per node");
    let mut x = vec![0.0; p.len()];
    for a in 0..p.nodes() {
        for b in (a + 1)..p.nodes() {
            x[p.index(a, b)] = f64::from(u8::from(s[a] != s[b]));
        }
    }
    x
}

/// The pair variables [`cutting_plane`] uses for `g`.
///
/// One node per spin, plus a **gauge node** when any field is non-zero: `h_i s_i` is the coupling
/// `h_i s_i s_r` of a spin to a reference `s_r`, and fixing `s_r = +1` costs nothing because the
/// energy is invariant under flipping every spin at once. Without that node the fields would have
/// no home in a pair variable at all.
#[must_use]
pub fn variables(g: &Graph) -> Pairs {
    Pairs::new(g.n + usize::from(g.h.iter().any(|&h| h != 0.0)))
}

/// How hard to try.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Params {
    /// Cutting-plane rounds. Each one separates, adds rows, and re-ascends.
    pub rounds: usize,
    /// Most-violated rows added per round. Adding every violated row at once is possible and is
    /// usually worse: the dual has to move all of them, and most were slack a step later.
    pub per_round: usize,
    /// Subgradient steps per round.
    pub ascent: usize,
    /// A row is separated only if it is broken by more than this. Below it the row would be added,
    /// take a multiplier of essentially zero, and cost a pass forever.
    pub min_violation: f64,
    /// Restarts of the quench that supplies the incumbent state. The incumbent never enters the
    /// bound; it sets the step scale and reports the gap.
    pub restarts: usize,
    /// Seed for that quench. Everything else here is deterministic without one.
    pub seed: u64,
}

impl Default for Params {
    fn default() -> Self {
        Params {
            rounds: 20,
            per_round: 64,
            ascent: 400,
            min_violation: 1e-7,
            restarts: 16,
            seed: 0x00C0_FFEE,
        }
    }
}

/// What one cutting-plane round did.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Round {
    /// 1-based round number.
    pub round: usize,
    /// Rows added, after discarding ones already active.
    pub added: usize,
    /// The largest violation seen at separation. It falls towards zero as the relaxation tightens,
    /// and reaching zero means the separation point is IN the metric polytope.
    pub worst_violation: f64,
    /// The bound after this round. Never below the previous round's, because the ascent restarts
    /// from the best multipliers found so far and the new rows enter at multiplier zero, which
    /// reproduces that value exactly.
    pub value: f64,
}

/// A metric-polytope bound and the cutting planes that produced it.
#[derive(Clone, Debug)]
pub struct Tightened {
    /// The bound. **No state has energy below [`Bound::value`]**, whatever the ascent did.
    pub bound: Bound,
    /// The same relaxation before any triangle inequality: the box bound, which is
    /// [`crate::bound::decoupled`] by another route. Carried so a caller can see what the rows
    /// bought rather than take it on trust.
    pub base: f64,
    /// The pair variables the rows are indexed by.
    pub pairs: Pairs,
    /// The best state the quench found. Its energy is an upper bound on the optimum, so
    /// [`Bound::gap`] against it brackets the answer.
    pub incumbent: Vec<i8>,
    /// The active rows, in the order they were added.
    pub cuts: Vec<TriangleCut>,
    /// One entry per round actually run.
    pub rounds: Vec<Round>,
}

/// Find the most violated triangle inequalities at `x`.
///
/// Every triple is examined and its worst row kept, then the survivors are sorted by violation and
/// the top `limit` returned. **At most one of the four rows on a triple can be violated at a point
/// of the box**: two apex rows would need a negative variable, and a perimeter row alongside an
/// apex row would need one above `1`. So "the worst row of the triple" loses nothing, and a
/// separation routine that returned all four of them would be adding three rows it had just proved
/// slack.
///
/// The scan is `O(n³)` and that is the honest cost of exact separation over the metric polytope —
/// there is no cleverer exact method, which is why [`MAX_NODES`] exists.
///
/// # Panics
///
/// If `x` is not one entry per pair of `p`.
#[must_use]
pub fn separate(p: Pairs, x: &[f64], min_violation: f64, limit: usize) -> Vec<(TriangleCut, f64)> {
    assert_eq!(x.len(), p.len(), "x must carry one entry per pair");
    let n = p.nodes();
    let mut found: Vec<(TriangleCut, f64)> = Vec::new();
    if limit == 0 {
        return found;
    }
    for a in 0..n {
        for b in (a + 1)..n {
            let iab = p.index(a, b);
            for c in (b + 1)..n {
                let (xab, xac, xbc) = (x[iab], x[p.index(a, c)], x[p.index(b, c)]);
                let mut worst = f64::NEG_INFINITY;
                let mut which = Facet::Perimeter;
                for f in Facet::ALL {
                    let (coef, rhs) = f.row();
                    let v = coef[0] * xab + coef[1] * xac + coef[2] * xbc - rhs;
                    if v > worst {
                        worst = v;
                        which = f;
                    }
                }
                if worst > min_violation {
                    found.push((TriangleCut::new(a as u32, b as u32, c as u32, which), worst));
                }
            }
        }
    }
    // Stable, so triples tied at the same violation keep enumeration order and the run stays
    // reproducible without a tie-breaking key.
    found.sort_by(|l, r| r.1.total_cmp(&l.1));
    found.truncate(limit);
    found
}

/// The linearised instance: `E = Σ consts + cᵀx` over the pair variables.
struct Relax {
    pairs: Pairs,
    /// Cost per pair variable, `2 J_ab` on a coupled pair and `0` elsewhere.
    c: Vec<f64>,
    /// `−J_ab` per coupling, kept as terms rather than a total so the constant rounds down with
    /// everything else in one directed sum.
    consts: Vec<f64>,
    cuts: Vec<TriangleCut>,
    lambda: Vec<f64>,
    /// Reduced-cost terms per pair variable, reused across evaluations.
    terms: Vec<Vec<f64>>,
    outer: Vec<f64>,
}

impl Relax {
    fn build(g: &Graph) -> Result<Relax, CutsError> {
        if g.w.iter().chain(g.h.iter()).any(|v| !v.is_finite()) {
            return Err(CutsError::NotFinite);
        }
        let pairs = variables(g);
        if pairs.nodes() > MAX_NODES {
            return Err(CutsError::TooLarge { nodes: pairs.nodes(), limit: MAX_NODES });
        }
        let mut c = vec![0.0; pairs.len()];
        let mut consts = Vec::with_capacity(g.n_edges + g.n);
        for i in 0..g.n {
            for k in g.offset[i]..g.offset[i + 1] {
                let j = g.nbr[k] as usize;
                if j > i {
                    // Exact: doubling is a power of two, and this is the only place a coupling is
                    // scaled at all.
                    c[pairs.index(i, j)] = 2.0 * g.w[k];
                    consts.push(-g.w[k]);
                }
            }
        }
        if pairs.nodes() > g.n {
            let root = g.n;
            for i in 0..g.n {
                if g.h[i] != 0.0 {
                    c[pairs.index(i, root)] = 2.0 * g.h[i];
                    consts.push(-g.h[i]);
                }
            }
        }
        let terms = (0..pairs.len()).map(|_| Vec::with_capacity(4)).collect();
        Ok(Relax { pairs, c, consts, cuts: Vec::new(), lambda: Vec::new(), terms, outer: Vec::new() })
    }

    /// `L(λ)` at the current multipliers, with `x` left holding the box minimiser.
    ///
    /// Sound for any `λ ≥ 0`: the reduced cost of each variable is summed DOWN, and `min(0, ·)` is
    /// monotone, so a reduced cost that rounds low can only send the bound low. The outer sum is
    /// directed the same way.
    fn evaluate(&mut self, x: &mut [f64]) -> f64 {
        for (p, t) in self.terms.iter_mut().enumerate() {
            t.clear();
            t.push(self.c[p]);
        }
        for (k, cut) in self.cuts.iter().enumerate() {
            let l = self.lambda[k];
            if l == 0.0 {
                // Skipped rather than pushed as a zero: an added zero leaves the compensated total
                // alone but lengthens the term count the guard is computed from, and the
                // round-to-round monotonicity of `Round::value` rests on a row at multiplier zero
                // reproducing the previous value BIT FOR BIT.
                continue;
            }
            let (coef, _) = cut.facet.row();
            let idx = cut.pair_indices(self.pairs);
            for t in 0..3 {
                self.terms[idx[t]].push(coef[t] * l);
            }
        }
        self.outer.clear();
        self.outer.extend_from_slice(&self.consts);
        for p in 0..self.terms.len() {
            let d = sum_down(&self.terms[p]);
            if d < 0.0 {
                x[p] = 1.0;
                self.outer.push(d);
            } else {
                x[p] = 0.0;
            }
        }
        for (k, cut) in self.cuts.iter().enumerate() {
            let l = self.lambda[k];
            let (_, rhs) = cut.facet.row();
            if l != 0.0 && rhs != 0.0 {
                // Exact: rhs is 2.
                self.outer.push(-rhs * l);
            }
        }
        sum_down(&self.outer)
    }

    /// The projected subgradient of `L` at the current `λ`, and its squared norm.
    ///
    /// A row whose multiplier is already at zero and whose violation is negative cannot move: the
    /// step would be clipped straight back by the projection onto `λ ≥ 0`. Zeroing it here rather
    /// than letting the projection do it is what makes a zero norm mean "this `λ` maximises the
    /// dual" instead of "this `λ` is pinned at the boundary".
    fn subgradient(&self, x: &[f64], gr: &mut [f64]) -> f64 {
        let mut norm = 0.0;
        for (k, cut) in self.cuts.iter().enumerate() {
            let v = cut.violation(self.pairs, x);
            let d = if self.lambda[k] <= 0.0 && v < 0.0 { 0.0 } else { v };
            gr[k] = d;
            norm += d * d;
        }
        norm
    }

    /// Subgradient ascent with a Polyak step and the standard halving adjustment.
    ///
    /// Returns the best value seen and the step-weighted average of the box minimisers, which is
    /// the primal point the next round separates on. The average is the point that matters:
    /// the individual minimisers are vertices of the box and their violations are all exactly `1`,
    /// so separating on one of them cannot rank a triple against another, while the ergodic average
    /// of a subgradient run converges to a primal solution of the relaxation and is fractional
    /// where the relaxation is.
    ///
    /// The multipliers are left at the best point found, not the last one. Subgradient ascent is
    /// not monotone; leaving the last iterate would make the next round start below where this one
    /// finished.
    ///
    /// The third return is whether the ascent stopped on a **zero projected subgradient**, which is
    /// a certificate that these multipliers maximise the dual for the rows in hand — the only
    /// honest reason to stop adding rows.
    fn ascend(&mut self, p: &Params, target: f64, x: &mut [f64]) -> (f64, Vec<f64>, bool) {
        let mut best = self.evaluate(x);
        let mut best_lambda = self.lambda.clone();
        let mut gr = vec![0.0; self.cuts.len()];
        let mut xbar = vec![0.0; x.len()];
        let mut wsum = 0.0f64;
        // The initial guess at how far the dual can still climb: the whole known gap. It is halved
        // whenever the run stops improving, which is what lets the step size find a kink -- and the
        // dual optimum of a Lagrangian relaxation is essentially always at a kink.
        let mut delta = (target - best).max(f64::EPSILON * target.abs().max(1.0));
        let mut stall = 0usize;
        let mut certified = false;
        const PATIENCE: usize = 8;
        for it in 0..p.ascent {
            let value = if it == 0 { best } else { self.evaluate(x) };
            if value > best {
                best = value;
                best_lambda.copy_from_slice(&self.lambda);
                stall = 0;
            } else {
                stall += 1;
                if stall >= PATIENCE {
                    delta *= 0.5;
                    stall = 0;
                }
            }
            let norm = self.subgradient(x, &mut gr);
            if norm <= 0.0 {
                // A zero projected subgradient of a concave function is a certificate of dual
                // optimality, so more steps cannot help.
                certified = true;
                break;
            }
            let step = (best + delta - value) / norm;
            for k in 0..self.lambda.len() {
                self.lambda[k] = (self.lambda[k] + step * gr[k]).max(0.0);
            }
            for (b, &xv) in xbar.iter_mut().zip(x.iter()) {
                *b += step * xv;
            }
            wsum += step;
        }
        self.lambda.copy_from_slice(&best_lambda);
        if wsum > 0.0 {
            for b in &mut xbar {
                *b /= wsum;
            }
        } else {
            xbar.copy_from_slice(x);
        }
        (best, xbar, certified)
    }
}

/// A greedy quench: the incumbent whose energy scales the ascent and reports the gap.
///
/// It never enters the bound. Its only jobs are to give the Polyak step a target above the dual
/// optimum — any state's energy is one, since `E(s) ≥ min E ≥ L(λ)` — and to give the caller
/// something to measure the bound against.
fn quench(g: &Graph, restarts: usize, seed: u64) -> Vec<i8> {
    let mut best = vec![1i8; g.n];
    let mut best_e = g.energy(&best);
    let mut rng = Pcg::new(seed, 0x000C_0757);
    let mut s = vec![1i8; g.n];
    for _ in 0..restarts {
        for v in &mut s {
            *v = rng.spin(0.5);
        }
        // Flipping i changes the energy by 2 s_i (Σ_j w_ij s_j + h_i), so a spin opposed to its own
        // field is exactly a downhill move. Sweeps until a full pass moves nothing.
        for _ in 0..64 {
            let mut moved = false;
            for i in 0..g.n {
                if f64::from(s[i]) * g.field(i, &s) < 0.0 {
                    s[i] = -s[i];
                    moved = true;
                }
            }
            if !moved {
                break;
            }
        }
        let e = g.energy(&s);
        if e < best_e {
            best_e = e;
            best.copy_from_slice(&s);
        }
    }
    best
}

/// Tighten the box relaxation with triangle inequalities, round by round.
///
/// Separate the current point, add the most violated rows, run subgradient ascent on their
/// multipliers, repeat.
///
/// It stops early for exactly two reasons, and "a round separated nothing" is NOT one of them on
/// its own — separation reads an approximate primal point, so finding no violated row there proves
/// nothing about the polytope. The two are: nothing was ever violated (an unfrustrated instance,
/// where the box bound is already the answer), and the ascent certified its own optimality with a
/// zero projected subgradient while separation found nothing left. Otherwise the round budget is
/// spent, and every extra round is more ascent on the rows in hand.
///
/// The result is a lower bound on `min_s E(s)` at **every** stage, including a run cut off after
/// one step, because a Lagrangian bound is valid at every multiplier vector rather than only at the
/// optimal one.
///
/// # Errors
///
/// [`CutsError::TooLarge`] past [`MAX_NODES`], where the `O(n³)` separation stops being the right
/// tool, and [`CutsError::NotFinite`] for a graph carrying a non-finite coupling or field.
pub fn cutting_plane(g: &Graph, p: &Params) -> Result<Tightened, CutsError> {
    let mut relax = Relax::build(g)?;
    let incumbent = quench(g, p.restarts, p.seed);
    // An upper bound on the dual optimum, which is all the step rule needs. Its rounding cannot
    // reach the bound: it scales a step, and every step lands on a λ whose value is re-derived.
    let target = g.energy(&incumbent);

    let mut x = vec![0.0; relax.pairs.len()];
    let base = relax.evaluate(&mut x);
    let mut best = base;
    let mut best_round = 0usize;
    let mut xsep = x.clone();
    let mut rounds: Vec<Round> = Vec::new();
    let mut active: BTreeSet<TriangleCut> = BTreeSet::new();
    let mut certified = false;

    for r in 0..p.rounds {
        let found = separate(relax.pairs, &xsep, p.min_violation, p.per_round);
        let worst = found.first().map_or(0.0, |&(_, v)| v);
        let mut added = 0usize;
        for (cut, _) in found {
            if active.insert(cut) {
                relax.cuts.push(cut);
                relax.lambda.push(0.0);
                added += 1;
            }
        }
        // A round that separates nothing is NOT a reason to stop, and treating it as one was worth
        // measuring: separation reads the ergodic primal average, which is only approximately
        // optimal, so "no violated row" at an unconverged point says nothing about the polytope.
        // Stopping there made the bound WORSE the more rows were offered per round -- C_11 went
        // from 0.07% loose at 64 rows per round to 16% loose at 256, because the bigger row set
        // silenced separation before the dual had climbed. The two honest reasons to stop are no
        // rows at all (nothing here is frustrated) and a dual already certified optimal for the
        // rows in hand.
        if added == 0 && (relax.cuts.is_empty() || certified) {
            break;
        }
        let (value, xbar, done) = relax.ascend(p, target, &mut x);
        certified = done;
        if value > best {
            best = value;
            best_round = r + 1;
        }
        xsep = xbar;
        rounds.push(Round { round: r + 1, added, worst_violation: worst, value: best });
    }

    let bound = Bound {
        value: best,
        parts: relax.pairs.len(),
        method: "metric polytope: box relaxation plus Lagrangian triangle inequalities, separated round by round",
        rounds: rounds.len(),
        best_round,
    };
    Ok(Tightened { bound, base, pairs: relax.pairs, incumbent, cuts: relax.cuts, rounds })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::GraphBuilder;

    /// Exhaustive ground energy, computed from the graph's own `energy` and nothing of this module.
    fn enumerate(g: &Graph) -> f64 {
        assert!(g.n <= 20, "2^{} states is not an oracle, it is a wait", g.n);
        let mut best = f64::INFINITY;
        let mut s = vec![1i8; g.n];
        for mask in 0u32..(1u32 << g.n) {
            for i in 0..g.n {
                s[i] = if mask >> i & 1 == 1 { 1 } else { -1 };
            }
            best = best.min(g.energy(&s));
        }
        best
    }

    /// `K_m` with unit max-cut weights, which is `J = −1` under `E = −J s s`.
    fn complete(m: usize) -> Graph {
        let mut b = GraphBuilder::new(m);
        for i in 0..m {
            for j in (i + 1)..m {
                b.couple(i, j, -1.0);
            }
        }
        b.build()
    }

    fn cycle(m: usize) -> Graph {
        let mut b = GraphBuilder::new(m);
        for i in 0..m {
            b.couple(i, (i + 1) % m, -1.0);
        }
        b.build()
    }

    /// ORACLE: `CUT(K_4) = MET(K_4)` (Barahona & Mahjoub 1986), against exhaustive enumeration.
    ///
    /// BOTH SIDES. The box relaxation must be strictly loose — `−6` against an optimum of `−2`, and
    /// a test that only checked the tightened side would pass for an implementation that returned
    /// the optimum by accident — and the tightened bound must reach the optimum, because on four
    /// nodes the triangle inequalities are the ENTIRE cut polytope and there is nothing left for a
    /// deeper relaxation to find.
    #[test]
    fn k4_is_loose_in_the_box_and_exact_after_triangles_against_enumeration() {
        let g = complete(4);
        let exact = enumerate(&g);
        assert_eq!(exact, -2.0, "K_4's max cut is 4 of 6 edges: E = 6 - 2*4");

        let t = cutting_plane(&g, &Params::default()).unwrap();

        // Loose before: the box gives up every edge, -Σ|J| = -6.
        assert!(t.base <= -6.0, "the box bound must be -6 or below, got {}", t.base);
        assert!(t.base < exact - 3.9, "the box must be strictly loose: {} vs {exact}", t.base);

        // Exact after, and never past it.
        assert!(t.bound.value <= exact + 1e-14, "UNSOUND: {} > {exact}", t.bound.value);
        // 8.9e-16 measured, so 1e-12 is three orders of margin on an ascent that lands on the
        // vertex itself rather than near it.
        assert!(
            t.bound.value > exact - 1e-12,
            "triangles close K_4: got {} for an optimum of {exact}",
            t.bound.value
        );
    }

    /// ORACLE: a closed form for the whole family. `max Σx over MET(K_n) = n(n-1)/3`, so the
    /// metric-polytope bound on `K_n` with unit max-cut weights is exactly `−n(n−1)/6`.
    ///
    /// Proof, and it is four lines: every pair of `K_n` lies in `n−2` triples, so summing the
    /// perimeter row over all `C(n,3)` of them gives `(n−2) Σx ≤ 2 C(n,3)`, i.e. `Σx ≤ n(n−1)/3`.
    /// The point `x ≡ 2/3` is feasible — perimeter `2 ≤ 2`, apex `2/3 ≤ 4/3` — and attains it. So
    /// the LP optimum is that number for every `n`, and the bound is
    /// `C(n,2) − 2·n(n−1)/3 = −n(n−1)/6`.
    ///
    /// **`K_5` is where the folklore breaks and this test is written to say so.** The true optimum
    /// is `−2` and the metric bound is `−10/3`: triangle inequalities do NOT close `K_5`, because
    /// five nodes is the smallest case with `CUT ⊊ MET` (Barahona & Mahjoub 1986). Asserting that
    /// they close it would be asserting a number no correct implementation can produce — so what is
    /// asserted instead is the exact value of the relaxation, from both sides, plus equality with
    /// the enumerated optimum for `n ≤ 4` and strict looseness for `n ≥ 5`.
    #[test]
    fn complete_graphs_hit_the_closed_form_metric_bound_and_close_only_up_to_four_nodes() {
        for n in 3..=8usize {
            let g = complete(n);
            let exact = enumerate(&g);
            let box_bound = -((n * (n - 1) / 2) as f64);
            let met = -(n as f64) * (n as f64 - 1.0) / 6.0;
            let t = cutting_plane(&g, &Params::default()).unwrap();

            assert!(t.base <= box_bound, "K_{n} box must be {box_bound}, got {}", t.base);
            assert!(t.base > box_bound - 1e-12, "K_{n} box must be {box_bound}, got {}", t.base);
            // Both sides of the closed form. Measured error over n = 3..=10 is 4.4e-16 to 1.1e-14.
            assert!(t.bound.value <= met + 1e-12, "K_{n} UNSOUND vs MET: {} > {met}", t.bound.value);
            assert!(t.bound.value > met - 1e-12, "K_{n} short of MET: {} vs {met}", t.bound.value);
            // Never above the true optimum either, which for n >= 5 is a far weaker statement than
            // the line above and for n <= 4 is the same one.
            assert!(t.bound.value <= exact + 1e-14, "K_{n} UNSOUND: {} > {exact}", t.bound.value);

            if n <= 4 {
                assert!(
                    (t.bound.value - exact).abs() < 1e-12,
                    "CUT = MET at n = {n}: {} vs {exact}",
                    t.bound.value
                );
            } else {
                assert!(
                    t.bound.value < exact - 0.3,
                    "K_{n} must stay strictly loose: {} vs {exact}",
                    t.bound.value
                );
            }
            assert!(t.bound.value > t.base + 1.0, "K_{n}: rows bought nothing");
        }
    }

    /// ORACLE: an odd cycle, where the cycle inequalities describe the cut polytope (Barahona 1983,
    /// a graph with no `K_5` minor) and are implied by the triangle inequalities of the complete
    /// graph — so the bound must be exact, against enumeration.
    #[test]
    fn an_odd_cycle_is_closed_exactly_against_enumeration() {
        for m in [5usize, 7, 9] {
            let g = cycle(m);
            let exact = enumerate(&g);
            let t = cutting_plane(&g, &Params::default()).unwrap();
            assert!(t.base <= -(m as f64), "box on C_{m}: {}", t.base);
            assert!(t.bound.value <= exact + 1e-14, "UNSOUND on C_{m}: {} > {exact}", t.bound.value);
            // 1.3e-15, 1.1e-12 and 6.7e-9 measured at m = 5, 7, 9: the ascent has more rows to
            // move on the longer cycle and lands nearer rather than on the vertex.
            assert!(
                t.bound.value > exact - 1e-6,
                "C_{m} must close: got {} for {exact}",
                t.bound.value
            );
        }
    }

    /// REGRESSION, and the defect was mine: a round that separates nothing is not a reason to stop.
    ///
    /// Separation reads the ergodic primal average, which is only approximately optimal, so an
    /// unconverged point can have no violated row and prove nothing. Stopping there made the bound
    /// WORSE the more rows were offered per round — `C_11` came out at **−10.44** against an
    /// optimum of −9 with 256 rows per round, because the larger row set silenced separation after
    /// two rounds while the dual was still climbing. Offering more rows must not cost accuracy.
    #[test]
    fn offering_more_rows_per_round_does_not_make_the_bound_worse_on_c11() {
        let g = cycle(11);
        let exact = enumerate(&g);
        assert_eq!(exact, -9.0);
        for per_round in [32usize, 64, 128, 256] {
            let t = cutting_plane(&g, &Params { per_round, ..Params::default() }).unwrap();
            assert!(t.bound.value <= exact + 1e-14, "UNSOUND at {per_round}: {}", t.bound.value);
            assert!(
                t.bound.value > exact - 0.1,
                "{per_round} rows per round left the bound at {} for an optimum of {exact}",
                t.bound.value
            );
        }
    }

    /// THE SOUNDNESS SWEEP. 240 random instances, exact optimum by enumeration, and a tolerance
    /// that would see the `7.8e-14` this crate's last unsound bound was out by.
    ///
    /// Fields included: they are what the gauge node exists for, and a relaxation that mishandled
    /// them would be wrong in exactly the direction this test looks.
    #[test]
    fn no_bound_over_240_random_instances_exceeds_the_enumerated_optimum() {
        let mut rng = Pcg::new(0xB1A5, 0x2C);
        let p = Params { rounds: 6, per_round: 32, ascent: 120, restarts: 8, ..Params::default() };
        let mut worst = f64::NEG_INFINITY;
        let mut tightened = 0usize;
        for trial in 0..240 {
            let n = 4 + (rng.next_u32() as usize % 6); // 4..=9
            let density = 0.3 + 0.7 * rng.f64();
            let fields = trial % 3 == 0;
            let mut b = GraphBuilder::new(n);
            for i in 0..n {
                for j in (i + 1)..n {
                    if rng.f64() < density {
                        b.couple(i, j, 2.0 * rng.f64() - 1.0);
                    }
                }
                if fields {
                    b.bias(i, 2.0 * rng.f64() - 1.0);
                }
            }
            let g = b.build();
            let exact = enumerate(&g);
            let t = cutting_plane(&g, &p).unwrap();
            worst = worst.max(t.bound.value - exact);
            assert!(
                t.bound.value <= exact + 1e-14,
                "trial {trial}: bound {} is ABOVE the optimum {exact} by {:e}",
                t.bound.value,
                t.bound.value - exact
            );
            assert!(t.bound.value >= t.base, "trial {trial}: cuts loosened the box bound");
            if t.bound.value > t.base + 1e-9 {
                tightened += 1;
            }
        }
        // 205 of 240 measured. The floor is a coverage ratchet: below it something has stopped
        // separating, which is a failure a per-instance soundness check cannot see -- a relaxation
        // that added no row at all would pass every assertion above.
        assert!(tightened >= 180, "only {tightened} of 240 instances were tightened at all");
        assert!(worst <= 0.0, "worst margin above the optimum was {worst:e}");
    }

    /// MONOTONE: a round may add nothing useful, but it may never take something away.
    ///
    /// Exact `>=`, no tolerance: the guarantee is structural — a new row enters at multiplier zero
    /// and contributes no term at all, so the first evaluation of a round reproduces the previous
    /// round's best value bit for bit.
    #[test]
    fn no_cutting_plane_round_is_looser_than_the_round_before_it() {
        let mut rng = Pcg::new(0x51DE, 0x9);
        for _ in 0..40 {
            let n = 5 + (rng.next_u32() as usize % 4);
            let mut b = GraphBuilder::new(n);
            for i in 0..n {
                for j in (i + 1)..n {
                    if rng.f64() < 0.7 {
                        b.couple(i, j, 2.0 * rng.f64() - 1.0);
                    }
                }
            }
            let g = b.build();
            let t = cutting_plane(&g, &Params { rounds: 8, ascent: 60, ..Params::default() }).unwrap();
            let mut prev = t.base;
            for r in &t.rounds {
                assert!(r.value >= prev, "round {} loosened {} to {}", r.round, prev, r.value);
                prev = r.value;
            }
            assert_eq!(t.bound.value, prev, "the reported bound must be the last round's value");
        }
    }

    /// EXHAUSTIVE: every cut vector on five nodes satisfies all forty triangle inequalities, and
    /// every 0/1 point on a triple that is NOT a cut vector is separated.
    ///
    /// The first half is the property separation must never break; the second is the property that
    /// makes it worth running. Both are exact — a cut vector's entries are `0` and `1`, and the
    /// rows have integer coefficients, so the violations are integers and the assertions carry no
    /// tolerance at all.
    #[test]
    fn separation_accepts_every_cut_vector_and_rejects_every_non_cut_triple() {
        let p = Pairs::new(5);
        let mut s = vec![1i8; 5];
        for mask in 0u32..32 {
            for i in 0..5 {
                s[i] = if mask >> i & 1 == 1 { 1 } else { -1 };
            }
            let x = cut_vector(p, &s);
            assert!(separate(p, &x, 0.0, 64).is_empty(), "a cut vector was separated: {x:?}");
            for a in 0..5 {
                for b in (a + 1)..5 {
                    for c in (b + 1)..5 {
                        for f in Facet::ALL {
                            let cut = TriangleCut::new(a as u32, b as u32, c as u32, f);
                            assert!(cut.violation(p, &x) <= 0.0, "{cut:?} broken by a cut vector");
                        }
                    }
                }
            }
        }
        // The three pair variables of one triple. Of the eight 0/1 points, the four with an even
        // number of ones are cuts and the four with an odd number are not.
        let q = Pairs::new(3);
        for bits in 0u32..8 {
            let x: Vec<f64> = (0..3).map(|k| f64::from(bits >> k & 1)).collect();
            let odd = x.iter().sum::<f64>() as u32 % 2 == 1;
            let found = separate(q, &x, 0.0, 8);
            assert_eq!(
                found.is_empty(),
                !odd,
                "point {x:?} separated {:?}, odd={odd}",
                found.len()
            );
            if odd {
                assert_eq!(found[0].1, 1.0, "a 0/1 violation is exactly one: {x:?}");
            }
        }
    }

    /// At most one of the four rows on a triple is violated at any point of the box — the fact
    /// [`separate`] relies on when it keeps only the worst row per triple.
    #[test]
    fn at_most_one_row_per_triple_is_violated_inside_the_box() {
        let p = Pairs::new(3);
        let mut rng = Pcg::new(7, 0x11);
        for _ in 0..4000 {
            let x: Vec<f64> = (0..3).map(|_| rng.f64()).collect();
            let broken = Facet::ALL
                .iter()
                .filter(|&&f| TriangleCut::new(0, 1, 2, f).violation(p, &x) > 0.0)
                .count();
            assert!(broken <= 1, "{broken} rows broken at once by {x:?}");
        }
    }

    /// EXHAUSTIVE: pair indexing is a bijection onto `0..n(n-1)/2`, and `ends` inverts `index`.
    ///
    /// A relaxation indexed by pairs is only as correct as this map: a collision would silently
    /// merge two variables, and every bound above it would be about a different problem.
    #[test]
    fn pair_indexing_is_a_bijection_and_ends_inverts_it() {
        for n in 0..24usize {
            let p = Pairs::new(n);
            let mut seen = vec![false; p.len()];
            for a in 0..n {
                for b in (a + 1)..n {
                    let k = p.index(a, b);
                    assert!(!seen[k], "index collision at ({a},{b}) -> {k}");
                    seen[k] = true;
                    assert_eq!(p.index(b, a), k, "index must not depend on order");
                    assert_eq!(p.ends(k), (a, b), "ends must invert index");
                }
            }
            assert!(seen.iter().all(|&v| v), "n={n}: {} variables, some unused", p.len());
            assert_eq!(p.is_empty(), seen.is_empty(), "is_empty must agree with the variable count");
        }
    }

    /// CROSS-MODULE ORACLE: the pair encoding reproduces `Graph::energy` on every state, fields and
    /// the gauge node included.
    ///
    /// This is the identity the whole module rests on — `E(s) = Σ consts + cᵀx(s)` — and it is
    /// checked against the graph's own energy, which knows nothing about pair variables.
    #[test]
    fn the_pair_encoding_reproduces_the_graphs_own_energy() {
        let mut rng = Pcg::new(0xE0E0, 0x3);
        for _ in 0..60 {
            let n = 2 + (rng.next_u32() as usize % 7);
            let mut b = GraphBuilder::new(n);
            for i in 0..n {
                for j in (i + 1)..n {
                    if rng.f64() < 0.6 {
                        b.couple(i, j, 2.0 * rng.f64() - 1.0);
                    }
                }
                b.bias(i, 2.0 * rng.f64() - 1.0);
            }
            let g = b.build();
            let relax = Relax::build(&g).unwrap();
            let base: f64 = relax.consts.iter().sum();
            let mut s = vec![1i8; g.n];
            for mask in 0u32..(1u32 << g.n) {
                for i in 0..g.n {
                    s[i] = if mask >> i & 1 == 1 { 1 } else { -1 };
                }
                // The gauge node is the reference, so it is always +1 in the lifted state.
                let mut lifted = s.clone();
                lifted.resize(relax.pairs.nodes(), 1);
                let x = cut_vector(relax.pairs, &lifted);
                let lin: f64 = base + relax.c.iter().zip(x.iter()).map(|(c, x)| c * x).sum::<f64>();
                assert!(
                    (lin - g.energy(&s)).abs() < 1e-12,
                    "encoding {lin} vs energy {}",
                    g.energy(&s)
                );
            }
        }
    }

    /// CROSS-MODULE ORACLE: with no rows added, this IS [`crate::bound::decoupled`].
    ///
    /// The box relaxation's optimum is `−Σ|J|` term by term, which is the decoupled bound arrived
    /// at from the other side. They are summed in different orders, so they agree to the last
    /// digits rather than bit for bit — and a base that did NOT agree would mean the linearisation
    /// itself was wrong, before any triangle inequality entered.
    #[test]
    fn the_untightened_base_agrees_with_the_decoupled_bound() {
        let mut rng = Pcg::new(0xDEC0, 0x5);
        for _ in 0..40 {
            let n = 3 + (rng.next_u32() as usize % 8);
            let mut b = GraphBuilder::new(n);
            for i in 0..n {
                for j in (i + 1)..n {
                    if rng.f64() < 0.5 {
                        b.couple(i, j, 4.0 * rng.f64() - 2.0);
                    }
                }
                b.bias(i, 4.0 * rng.f64() - 2.0);
            }
            let g = b.build();
            let t = cutting_plane(&g, &Params { rounds: 0, ..Params::default() }).unwrap();
            let d = crate::bound::decoupled(&g).value;
            assert!((t.base - d).abs() < 1e-9, "base {} vs decoupled {d}", t.base);
            assert_eq!(t.bound.value, t.base, "zero rounds must report the box bound");
            assert!(t.rounds.is_empty());
        }
    }

    /// A ferromagnet is unfrustrated, the box bound is already exact, and separation finds nothing
    /// to do — so the loop must stop rather than spend its rounds.
    #[test]
    fn an_unfrustrated_instance_separates_nothing_and_stops() {
        let mut b = GraphBuilder::new(6);
        for i in 0..6 {
            for j in (i + 1)..6 {
                b.couple(i, j, 1.0);
            }
        }
        let g = b.build();
        let exact = enumerate(&g);
        let t = cutting_plane(&g, &Params::default()).unwrap();
        assert!(t.rounds.is_empty(), "nothing is violated on a ferromagnet: {:?}", t.rounds);
        assert!(t.bound.value > exact - 1e-9 && t.bound.value <= exact, "{} vs {exact}", t.bound.value);
        assert!(t.bound.proves_optimal(&g, &t.incumbent, 1e-9), "and the quench found it");
    }

    /// The refusals are refusals, not approximations.
    #[test]
    fn a_graph_too_large_or_not_finite_is_refused_by_name() {
        let mut b = GraphBuilder::new(MAX_NODES + 1);
        b.couple(0, 1, -1.0);
        let big = b.build();
        assert_eq!(
            cutting_plane(&big, &Params::default()).unwrap_err(),
            CutsError::TooLarge { nodes: MAX_NODES + 1, limit: MAX_NODES }
        );
        let mut b = GraphBuilder::new(3);
        b.couple(0, 1, f64::NAN);
        let nan = b.build();
        assert_eq!(cutting_plane(&nan, &Params::default()).unwrap_err(), CutsError::NotFinite);
        assert!(!CutsError::NotFinite.to_string().is_empty());
    }
}
