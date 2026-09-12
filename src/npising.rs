//! Ising formulations of the standard NP-hard problems, with decoders that read a spin state back
//! into a solution.
//!
//! Lucas, *Ising formulations of many NP problems*, Front. Phys. **2**:5 (2014), is the reference
//! every Ising-machine paper cites for how a combinatorial problem becomes a spin model. It is the
//! bridge between "this hardware minimises `E(s) = -sum J s s - sum h s`" and "this hardware solved
//! a scheduling instance". Until this module existed the crate could sample a Hamiltonian
//! beautifully and had no way to write one down from a problem anybody outside physics has:
//! maximum clique, vertex cover, set cover, colouring, Hamiltonian cycle, the travelling salesman.
//!
//! # The whole correctness argument is an inequality
//!
//! A reduction here is a penalty method. The objective is what you want minimised; the penalty is
//! what makes an infeasible state expensive. **The reduction is correct exactly when the penalty is
//! large enough that no infeasible state can undercut the best feasible one**, and that is a single
//! inequality per problem. Get it wrong and the model still builds, still samples, still returns a
//! state -- one that decodes to something that is not a solution, or to a feasible solution that is
//! not optimal. Nothing raises.
//!
//! So each constructor states its inequality, [`MaxClique::threshold`] and its siblings return the
//! number the penalty must exceed, and [`MaxClique::guarantees_feasible`] says whether it does.
//! The tests do not take the inequality on trust: they MEASURE the exact critical penalty by
//! enumerating both spaces, and assert the measured value is the documented one.
//!
//! | problem | penalty `A` charges | objective | the inequality | measured critical |
//! |---|---|---|---|---|
//! | [`MaxClique`] | each non-edge inside the chosen set | `-B` per chosen vertex | `A > B` | `A = B`, attained |
//! | [`VertexCover`] | each uncovered edge | `+B` per chosen vertex | `A > B` | `A = B`, attained |
//! | [`SetCover`] | each broken counting constraint | `+B` per chosen subset | `A > B` | `A = B`, attained |
//! | [`Colouring`] | each broken constraint | none | `A > 0` | -- |
//! | [`HamiltonianCycle`] | each broken constraint | none | `A > 0` | -- |
//! | [`Tsp`] | each broken constraint, each absent-edge step | `B W` per tour step | `A > B T*` | `A <= B max W`, attained |
//!
//! ## Why `A > B` is the threshold for the three covering-shaped problems, and why it is tight
//!
//! One argument covers clique, vertex cover and set cover. Take any infeasible state with `k`
//! violations. Each violation can be repaired by adding or removing **one** variable -- an
//! uncovered edge by taking one endpoint, an uncovered element by taking one subset that contains
//! it, a non-edge inside a clique candidate by dropping one endpoint. So there is a feasible state
//! at objective cost at most `k B` more, and penalty `k A` less. If `A > B` the repair strictly
//! wins, so no infeasible state is a ground state.
//!
//! It is tight because that repair can be exactly one-for-one. On the 4-cycle the whole vertex set
//! has two internal non-edges and two vertices more than the maximum clique, so at `A = B` it ties
//! the optimum and at `A < B` it beats it. The tests compute
//! `max over infeasible states of (optimal objective - objective) / violations` by enumeration,
//! which is the exact critical penalty, and find `1.0` on the nose.
//!
//! ## The travelling salesman's threshold is different, and Lucas's stated form is not tight
//!
//! Lucas gives `0 < B max(W) < A`. The condition this module can PROVE is weaker and larger: an
//! infeasible state pays at least one whole `A` and carries only non-negative weight, so
//! `A > B T*` -- the penalty above the cost of the optimal tour -- is sufficient, and `T*` is not
//! known in advance. [`Tsp::threshold`] therefore reports `B` times an upper bound on `T*` (the `n`
//! largest edge weights, summed through [`crate::round::sum_up`] so the bound cannot come out low),
//! and the default penalty sits one `B` above it.
//!
//! What the measurement says, over 27 four-city instances enumerated state by state:
//!
//! * `A > B max(W)` is **sufficient** on every one, so Lucas's condition is safe;
//! * it is **attained, not generally tight**. On the uniform complete graph the exact critical
//!   penalty is `B max(W)` on the nose, so the bound cannot be weakened as a general statement.
//!   On `W = [[0,2,5,1],[2,0,3,4],[5,3,0,6],[1,4,6,0]]` the exact critical penalty is **4.5** where
//!   `max(W)` is **6** -- Lucas is 33% above what that instance needs. The first version of this
//!   module asserted equality on every instance and was wrong for exactly that reason. The slack is
//!   observable, not bookkeeping: at `A = 5.9994`, just BELOW Lucas's stated threshold, the ground
//!   state of that instance is still the optimal tour, so a tightness check written against the
//!   stated threshold rather than the measured one would assert an infeasible ground state and be
//!   wrong;
//! * this module's own [`Tsp::threshold`] is larger again, by roughly `n`, and that is the price of
//!   promising something it can prove without solving the instance.
//!
//! A caller who wants the tight penalty passes it to [`Tsp::with_weights`];
//! `tsp_penalty_threshold_measured_by_enumeration_is_the_largest_edge_weight` shows the answer is
//! still the optimal tour just above the measured critical value and stops being a tour just below.
//!
//! # Encoding, and why there is an offset
//!
//! Every formulation here is naturally a quadratic model over `x` in `{0,1}`, and the spin model is
//! `x = (1 + s)/2`. That substitution leaves a constant behind -- the ones in `(1 - sum x)^2` have
//! to go somewhere -- so each model carries an [`MaxClique::offset`] and the identity is
//!
//! ```text
//!   H(x)  =  graph.energy(s)  +  offset
//! ```
//!
//! at **every** state, not only at the optimum. Dropping the offset leaves the minimiser unchanged
//! and makes every reported number wrong by a fixed amount, which is the error that survives a
//! solver test and fails a comparison against a published optimum. It is asserted exactly -- with
//! integer weights every coefficient is a multiple of a quarter, so there is no tolerance to hide
//! in. The one exception is [`Tsp::new`], whose penalty comes from [`crate::round::sum_up`] and
//! therefore carries that function's rounding guard; its tests use a tolerance and say why.
//!
//! # Decoding is reading AND checking
//!
//! `decode` returns a solution or a typed error. It never returns "the closest thing to a
//! solution". A state whose one-hot group has two spins up is not a tour with a small defect, it is
//! not a tour, and [`DecodeError`] names which group and how many spins were set. This is the same
//! rule as [`crate::encode::Slot::decode`] returning `None` for a surplus binary codeword: the
//! encoding is the only thing standing between an invalid state and a wrong answer.
//!
//! ```
//! use ferrotherm::npising::MaxClique;
//! use ferrotherm::oracle::{Exhaustive, Solver};
//!
//! // A 4-cycle: the largest clique is a single edge.
//! let c4 = MaxClique::new(4, &[(0, 1), (1, 2), (2, 3), (3, 0)]).unwrap();
//! assert!(c4.guarantees_feasible());
//! let (s, _) = Exhaustive.solve(c4.graph());
//! let clique = c4.decode(&s).unwrap();
//! assert_eq!(clique.len(), 2);
//! ```

use crate::graph::{Graph, GraphBuilder};
use std::collections::BTreeMap;

// ---------------------------------------------------------------------------------------------
// errors
// ---------------------------------------------------------------------------------------------

/// Why an instance could not be turned into a spin model.
///
/// Every variant names what was actually seen. A reduction that silently repaired its input would
/// produce a model for a problem the caller did not pose, and the caller would read its optimum as
/// the answer to the one they did.
#[derive(Clone, Debug, PartialEq)]
pub enum InstanceError {
    /// An edge names a vertex the graph does not have.
    VertexOutOfRange {
        /// The vertex named.
        vertex: usize,
        /// Vertices in the graph, so valid indices are `0..n`.
        n: usize,
    },
    /// An edge joins a vertex to itself. A self-loop is not coverable and not a clique edge.
    SelfLoop {
        /// The vertex named twice.
        vertex: usize,
    },
    /// Too few vertices for the formulation to mean anything.
    TooSmall {
        /// Vertices given.
        n: usize,
        /// The fewest this formulation accepts.
        min: usize,
    },
    /// A colouring with no colours, which no graph with a vertex admits.
    NoColours,
    /// A universe with no elements, or a collection with no subsets.
    Empty {
        /// Which of the two was empty.
        what: &'static str,
    },
    /// A subset with no elements. It can never help a cover and its variable is free.
    EmptySubset {
        /// Index of the offending subset.
        subset: usize,
    },
    /// A subset names an element outside the universe.
    ElementOutOfRange {
        /// Index of the offending subset.
        subset: usize,
        /// The element it named.
        element: usize,
        /// Elements in the universe, so valid indices are `0..universe`.
        universe: usize,
    },
    /// No subset contains this element, so the instance has no cover at all.
    ///
    /// Refused rather than solved: the model would be perfectly well formed and its ground state
    /// would decode to nothing, which reads exactly like a solver that failed.
    Uncoverable {
        /// The element nothing covers.
        element: usize,
    },
    /// A weight that is not a positive finite number.
    ///
    /// A zero or negative objective weight inverts the problem; a non-finite one poisons every
    /// energy it reaches.
    BadWeight {
        /// Which weight, by name.
        what: &'static str,
        /// The value supplied.
        got: f64,
    },
    /// A weight matrix of the wrong length for the city count.
    Matrix {
        /// Entries supplied.
        got: usize,
        /// Entries needed, which is `n * n`.
        want: usize,
    },
    /// A weight matrix that is not symmetric. This module formulates the SYMMETRIC problem.
    Asymmetric {
        /// Row index.
        u: usize,
        /// Column index.
        v: usize,
        /// The entry at `(u, v)`.
        uv: f64,
        /// The entry at `(v, u)`.
        vu: f64,
    },
    /// An edge weight that is negative or NaN.
    ///
    /// The sufficiency argument for the penalty rests on weights being non-negative: a negative
    /// edge would let an infeasible state buy back its penalty.
    EdgeWeight {
        /// Row index.
        u: usize,
        /// Column index.
        v: usize,
        /// The value supplied.
        got: f64,
    },
}

impl core::fmt::Display for InstanceError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            InstanceError::VertexOutOfRange { vertex, n } => {
                write!(f, "an edge names vertex {vertex} in a graph of {n} vertices")
            }
            InstanceError::SelfLoop { vertex } => write!(
                f,
                "vertex {vertex} carries a self-loop; no vertex set covers it and no clique \
                 contains it, so the problem it poses is not the one this formulation solves"
            ),
            InstanceError::TooSmall { n, min } => write!(
                f,
                "this formulation needs at least {min} vertices and was given {n}; at fewer the \
                 cyclic position index wraps onto itself and a step would be its own predecessor"
            ),
            InstanceError::NoColours => write!(
                f,
                "a colouring needs at least one colour, and a graph with a vertex needs at least one"
            ),
            InstanceError::Empty { what } => write!(f, "the {what} is empty"),
            InstanceError::EmptySubset { subset } => write!(
                f,
                "subset {subset} contains no elements, so it can never help a cover while still \
                 costing a variable the search has to get right"
            ),
            InstanceError::ElementOutOfRange { subset, element, universe } => write!(
                f,
                "subset {subset} names element {element} in a universe of {universe} elements"
            ),
            InstanceError::Uncoverable { element } => write!(
                f,
                "no subset contains element {element}, so this instance has no cover at all; the \
                 model would build, and its ground state would decode to nothing -- which reads \
                 exactly like a solver that failed"
            ),
            InstanceError::BadWeight { what, got } => {
                write!(f, "the {what} must be a positive finite number, and {got} is not")
            }
            InstanceError::Matrix { got, want } => {
                write!(f, "a weight matrix of {got} entries where {want} are needed")
            }
            InstanceError::Asymmetric { u, v, uv, vu } => write!(
                f,
                "the weight matrix is not symmetric: ({u},{v}) is {uv} and ({v},{u}) is {vu}. This \
                 module formulates the symmetric travelling salesman; an asymmetric instance needs \
                 a directed penalty, which is a different model"
            ),
            InstanceError::EdgeWeight { u, v, got } => write!(
                f,
                "edge ({u},{v}) weighs {got}; weights must be non-negative and finite, or infinite \
                 to mark an absent edge. The penalty argument rests on non-negativity -- a negative \
                 edge lets an infeasible state buy its own penalty back"
            ),
        }
    }
}

impl core::error::Error for InstanceError {}

/// Which one-hot group of spins a decoder found malformed.
///
/// Carried by [`DecodeError::NotOneHot`] so the failure names the constraint that broke rather than
/// a spin index whose meaning the caller would have to re-derive.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Group {
    /// The colours available to one vertex.
    VertexColour(usize),
    /// The positions one city may occupy in a tour.
    City(usize),
    /// The cities that may occupy one position in a tour.
    Position(usize),
    /// The coverage-count ancillas of one universe element.
    Coverage(usize),
}

impl core::fmt::Display for Group {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Group::VertexColour(v) => write!(f, "the colours of vertex {v}"),
            Group::City(v) => write!(f, "the positions of city {v}"),
            Group::Position(j) => write!(f, "the cities at position {j}"),
            Group::Coverage(a) => write!(f, "the coverage count of element {a}"),
        }
    }
}

/// Why a spin state does not encode a solution.
///
/// A decoder returns a solution or one of these. It never returns the nearest thing to a solution:
/// a state with two spins up in a one-hot group is not a tour with a defect, it is not a tour.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DecodeError {
    /// The state has the wrong number of spins for this model.
    WrongLength {
        /// Spins supplied.
        got: usize,
        /// Spins the model has.
        want: usize,
    },
    /// A spin outside `{-1, +1}`. Zero in particular is the shape of an uninitialised buffer.
    NotASpin {
        /// Index of the offending spin.
        index: usize,
        /// The value found.
        got: i8,
    },
    /// A group that must have exactly one spin up did not.
    NotOneHot {
        /// Which group.
        group: Group,
        /// How many of its spins were `+1`.
        set: usize,
    },
    /// Two chosen vertices with no edge between them, so the chosen set is not a clique.
    NotAClique {
        /// One endpoint.
        u: usize,
        /// The other.
        v: usize,
    },
    /// An edge with neither endpoint chosen, so the chosen set is not a vertex cover.
    EdgeUncovered {
        /// One endpoint.
        u: usize,
        /// The other.
        v: usize,
    },
    /// A universe element no chosen subset contains.
    ElementUncovered {
        /// The element.
        element: usize,
    },
    /// The coverage ancillas say one thing and the chosen subsets say another.
    ///
    /// A valid encoding is not merely a cover: the ancilla for each element must record how many
    /// chosen subsets contain it. A state that covers everything with inconsistent ancillas sits
    /// above the ground energy and is not what the formulation encodes.
    CountMismatch {
        /// The element.
        element: usize,
        /// What the ancillas recorded.
        encoded: usize,
        /// How many chosen subsets actually contain it.
        covering: usize,
    },
    /// Two adjacent vertices given the same colour.
    ColourConflict {
        /// One endpoint.
        u: usize,
        /// The other.
        v: usize,
        /// The colour they share.
        colour: usize,
    },
    /// The tour steps between two cities the graph has no edge between.
    StepNotAnEdge {
        /// City left.
        from: usize,
        /// City entered.
        to: usize,
        /// Which step of the tour, counting from zero.
        step: usize,
    },
}

impl core::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            DecodeError::WrongLength { got, want } => {
                write!(f, "a state of {got} spins for a model of {want}")
            }
            DecodeError::NotASpin { index, got } => write!(
                f,
                "spin {index} is {got}; spins are -1 or +1, and 0 is the shape of a buffer nothing \
                 has written to"
            ),
            DecodeError::NotOneHot { group, set } => write!(
                f,
                "{group} has {set} spins up where exactly one belongs; this state does not encode \
                 a solution, and it is not a solution with a defect"
            ),
            DecodeError::NotAClique { u, v } => {
                write!(f, "vertices {u} and {v} are both chosen and there is no edge between them")
            }
            DecodeError::EdgeUncovered { u, v } => {
                write!(f, "edge ({u},{v}) has neither endpoint in the chosen set")
            }
            DecodeError::ElementUncovered { element } => {
                write!(f, "no chosen subset contains element {element}")
            }
            DecodeError::CountMismatch { element, encoded, covering } => write!(
                f,
                "the ancillas record element {element} as covered {encoded} times and {covering} \
                 chosen subsets contain it; the counting constraint is what makes the penalty \
                 exact, so a state that breaks it is not the encoding"
            ),
            DecodeError::ColourConflict { u, v, colour } => {
                write!(f, "adjacent vertices {u} and {v} both take colour {colour}")
            }
            DecodeError::StepNotAnEdge { from, to, step } => {
                write!(f, "step {step} goes from {from} to {to} and the graph has no such edge")
            }
        }
    }
}

impl core::error::Error for DecodeError {}

// ---------------------------------------------------------------------------------------------
// the quadratic accumulator, and the one substitution every formulation goes through
// ---------------------------------------------------------------------------------------------

/// A quadratic model over `x` in `{0,1}`, assembled term by term and frozen into spins once.
///
/// Every formulation in Lucas is written over binary variables, and writing them that way here
/// keeps each construction readable against the paper. The substitution `x = (1 + s)/2` happens in
/// exactly one place, [`Qubo::into_graph`], so there is one place to get the sign convention right
/// rather than six.
struct Qubo {
    n: usize,
    lin: Vec<f64>,
    quad: BTreeMap<(u32, u32), f64>,
    offset: f64,
}

impl Qubo {
    fn new(n: usize) -> Qubo {
        Qubo { n, lin: vec![0.0; n], quad: BTreeMap::new(), offset: 0.0 }
    }

    fn constant(&mut self, c: f64) {
        self.offset += c;
    }

    fn linear(&mut self, v: usize, c: f64) {
        self.lin[v] += c;
    }

    /// `q x_u x_v`. A repeated variable folds into the linear part, since `x^2 = x` on `{0,1}`.
    fn quadratic(&mut self, u: usize, v: usize, q: f64) {
        if q == 0.0 {
            return;
        }
        if u == v {
            self.lin[u] += q;
            return;
        }
        let key = if u < v { (u as u32, v as u32) } else { (v as u32, u as u32) };
        *self.quad.entry(key).or_insert(0.0) += q;
    }

    /// Add `scale * (c0 + sum_i a_i x_i)^2`, expanded.
    ///
    /// `x_i^2 = x_i` puts each `a_i^2` on the linear part, which is where the "+ a^2" below comes
    /// from and is the step a QUBO expansion is usually wrong at. The variables in `terms` must be
    /// distinct; every caller here builds them from disjoint index ranges.
    fn square(&mut self, c0: f64, terms: &[(usize, f64)], scale: f64) {
        if scale == 0.0 {
            return;
        }
        self.constant(scale * c0 * c0);
        for &(i, a) in terms {
            self.linear(i, scale * (2.0 * c0 * a + a * a));
        }
        for p in 0..terms.len() {
            for r in (p + 1)..terms.len() {
                self.quadratic(terms[p].0, terms[r].0, scale * 2.0 * terms[p].1 * terms[r].1);
            }
        }
    }

    /// Freeze into spins: the graph, and the constant `H(x) - E(s)`.
    ///
    /// With `x = (1 + s)/2`,
    ///
    /// ```text
    ///   c x        = c/2 + (c/2) s
    ///   q x_u x_v  = q/4 + (q/4)(s_u + s_v) + (q/4) s_u s_v
    /// ```
    ///
    /// and this crate's energy is `E(s) = -sum J s s - sum h s`, so `J = -q/4` and the bias picks up
    /// a quarter of every incident quadratic coefficient. THAT incident sum is the term a
    /// hand-written substitution drops: without it the model still has the right couplings, the
    /// right shape and the wrong minimiser.
    fn into_graph(self) -> (Graph, f64) {
        let mut b = GraphBuilder::new(self.n);
        let mut incident = vec![0.0f64; self.n];
        for (&(u, v), &q) in &self.quad {
            if q != 0.0 {
                b.couple(u as usize, v as usize, -q / 4.0);
            }
            incident[u as usize] += q / 4.0;
            incident[v as usize] += q / 4.0;
        }
        let mut constant = self.offset;
        for v in 0..self.n {
            b.bias(v, -(self.lin[v] / 2.0 + incident[v]));
            constant += self.lin[v] / 2.0;
        }
        for &q in self.quad.values() {
            constant += q / 4.0;
        }
        (b.build(), constant)
    }
}

/// Read a state as binary selections, refusing anything that is not a spin.
fn binary(s: &[i8], want: usize) -> Result<Vec<u8>, DecodeError> {
    if s.len() != want {
        return Err(DecodeError::WrongLength { got: s.len(), want });
    }
    let mut x = vec![0u8; want];
    for (i, &v) in s.iter().enumerate() {
        x[i] = match v {
            1 => 1,
            -1 => 0,
            got => return Err(DecodeError::NotASpin { index: i, got }),
        };
    }
    Ok(x)
}

/// Adjacency matrix from an edge list, with every way an edge list can be wrong refused.
///
/// Duplicate edges collapse, because an edge list is a SET: listing `(0,1)` twice would otherwise
/// double that edge's penalty and move the threshold for that instance alone.
fn adjacency(n: usize, edges: &[(usize, usize)]) -> Result<Vec<bool>, InstanceError> {
    let mut adj = vec![false; n * n];
    for &(u, v) in edges {
        if u >= n {
            return Err(InstanceError::VertexOutOfRange { vertex: u, n });
        }
        if v >= n {
            return Err(InstanceError::VertexOutOfRange { vertex: v, n });
        }
        if u == v {
            return Err(InstanceError::SelfLoop { vertex: u });
        }
        adj[u * n + v] = true;
        adj[v * n + u] = true;
    }
    Ok(adj)
}

/// A positive finite weight, or the error naming what arrived instead.
fn positive(what: &'static str, got: f64) -> Result<f64, InstanceError> {
    // `!(got > 0.0)` rather than `got <= 0.0`: the difference is NaN, which this rejects.
    if !(got > 0.0) || !got.is_finite() {
        return Err(InstanceError::BadWeight { what, got });
    }
    Ok(got)
}

// ---------------------------------------------------------------------------------------------
// maximum clique
// ---------------------------------------------------------------------------------------------

/// Maximum clique, as the maximum independent set of the complement (Lucas section 4.2).
///
/// `x_v = 1` when vertex `v` is chosen, and
///
/// ```text
///   H  =  A * sum over non-edges (u,v) of x_u x_v   -   B * sum_v x_v
/// ```
///
/// A chosen set is a clique exactly when it contains no non-edge, so the first term is zero exactly
/// on cliques and the second rewards size.
///
/// **The inequality is `A > B`**, and it is tight. Given a chosen set with `k` internal non-edges,
/// dropping one endpoint of each removes every violation and costs at most `k B`, saving at least
/// `k A`; so when `A > B` every infeasible state is strictly beaten by a clique. At `A = B` the
/// 4-cycle's full vertex set ties a maximum clique, and below it wins --
/// `max_clique_penalty_threshold_measured_by_enumeration_is_exactly_the_reward` measures the
/// critical value and gets `B` exactly.
///
/// Lucas's own clique section states a DECISION Hamiltonian for "is there a clique of size K";
/// this is the optimisation form, which is his set-packing model on the complement graph and needs
/// no guess at `K`.
pub struct MaxClique {
    n: usize,
    adj: Vec<bool>,
    reward: f64,
    penalty: f64,
    graph: Graph,
    offset: f64,
}

impl MaxClique {
    /// A maximum-clique model over `n` vertices, with reward `1` and penalty `2`.
    ///
    /// # Errors
    ///
    /// [`InstanceError`] if an edge names a vertex outside `0..n` or joins a vertex to itself.
    pub fn new(n: usize, edges: &[(usize, usize)]) -> Result<MaxClique, InstanceError> {
        MaxClique::with_weights(n, edges, 1.0, 2.0)
    }

    /// The same model with the two weights chosen.
    ///
    /// **This does not enforce `penalty > reward`**, and that is deliberate: the inequality is the
    /// claim this module is built on, and a claim that cannot be violated cannot be measured. A
    /// caller who wants the soft model is also entitled to it. [`MaxClique::guarantees_feasible`]
    /// reports whether the ground state is certain to be a clique.
    ///
    /// # Errors
    ///
    /// [`InstanceError`] as [`MaxClique::new`], or [`InstanceError::BadWeight`] for a weight that
    /// is not positive and finite.
    pub fn with_weights(
        n: usize,
        edges: &[(usize, usize)],
        reward: f64,
        penalty: f64,
    ) -> Result<MaxClique, InstanceError> {
        let reward = positive("clique reward", reward)?;
        let penalty = positive("clique penalty", penalty)?;
        let adj = adjacency(n, edges)?;
        let mut q = Qubo::new(n);
        for v in 0..n {
            q.linear(v, -reward);
        }
        for u in 0..n {
            for v in (u + 1)..n {
                if !adj[u * n + v] {
                    q.quadratic(u, v, penalty);
                }
            }
        }
        let (graph, offset) = q.into_graph();
        Ok(MaxClique { n, adj, reward, penalty, graph, offset })
    }

    /// The spin model. Its ground state decodes to a maximum clique when
    /// [`MaxClique::guarantees_feasible`].
    #[must_use]
    pub fn graph(&self) -> &Graph {
        &self.graph
    }

    /// The constant with `H(x) = graph.energy(s) + offset` at every state.
    #[must_use]
    pub fn offset(&self) -> f64 {
        self.offset
    }

    /// Reward per chosen vertex, the `B` of the inequality.
    #[must_use]
    pub fn reward(&self) -> f64 {
        self.reward
    }

    /// Penalty per internal non-edge, the `A` of the inequality.
    #[must_use]
    pub fn penalty(&self) -> f64 {
        self.penalty
    }

    /// The value the penalty must strictly exceed, which for clique is the reward.
    #[must_use]
    pub fn threshold(&self) -> f64 {
        self.reward
    }

    /// Whether the penalty exceeds its threshold, so that every ground state is a maximum clique.
    #[must_use = "false means the ground state may decode to something that is not a clique"]
    pub fn guarantees_feasible(&self) -> bool {
        self.penalty > self.threshold()
    }

    /// The chosen vertices, in increasing order, or why the state is not a clique.
    ///
    /// # Errors
    ///
    /// [`DecodeError::WrongLength`] or [`DecodeError::NotASpin`] for a state that is not a state of
    /// this model, and [`DecodeError::NotAClique`] naming the first pair of chosen vertices with no
    /// edge between them.
    pub fn decode(&self, s: &[i8]) -> Result<Vec<usize>, DecodeError> {
        let x = binary(s, self.n)?;
        let chosen: Vec<usize> = (0..self.n).filter(|&v| x[v] == 1).collect();
        for (a, &u) in chosen.iter().enumerate() {
            for &v in &chosen[(a + 1)..] {
                if !self.adj[u * self.n + v] {
                    return Err(DecodeError::NotAClique { u, v });
                }
            }
        }
        Ok(chosen)
    }
}

// ---------------------------------------------------------------------------------------------
// minimum vertex cover
// ---------------------------------------------------------------------------------------------

/// Minimum vertex cover (Lucas section 4.3).
///
/// `x_v = 1` when vertex `v` is in the cover, and
///
/// ```text
///   H  =  A * sum over edges (u,v) of (1 - x_u)(1 - x_v)   +   B * sum_v x_v
/// ```
///
/// The first term charges `A` for every edge with neither endpoint chosen; the second charges `B`
/// per vertex used.
///
/// **The inequality is `A > B`**, for the same one-for-one repair as [`MaxClique`]: `k` uncovered
/// edges are covered by at most `k` vertices, costing `k B` and saving `k A`. It is tight on the
/// triangle, where a single vertex leaves one edge uncovered and a minimum cover needs two --
/// measured in `vertex_cover_penalty_threshold_measured_by_enumeration_is_exactly_the_vertex_weight`.
pub struct VertexCover {
    n: usize,
    edges: Vec<(usize, usize)>,
    weight: f64,
    penalty: f64,
    graph: Graph,
    offset: f64,
}

impl VertexCover {
    /// A minimum-vertex-cover model over `n` vertices, with vertex weight `1` and penalty `2`.
    ///
    /// # Errors
    ///
    /// [`InstanceError`] if an edge names a vertex outside `0..n` or joins a vertex to itself.
    pub fn new(n: usize, edges: &[(usize, usize)]) -> Result<VertexCover, InstanceError> {
        VertexCover::with_weights(n, edges, 1.0, 2.0)
    }

    /// The same model with the two weights chosen. As [`MaxClique::with_weights`], the inequality
    /// is reported by [`VertexCover::guarantees_feasible`] rather than enforced here.
    ///
    /// # Errors
    ///
    /// [`InstanceError`] as [`VertexCover::new`], or [`InstanceError::BadWeight`].
    pub fn with_weights(
        n: usize,
        edges: &[(usize, usize)],
        weight: f64,
        penalty: f64,
    ) -> Result<VertexCover, InstanceError> {
        let weight = positive("vertex weight", weight)?;
        let penalty = positive("uncovered-edge penalty", penalty)?;
        let adj = adjacency(n, edges)?;
        let mut list = Vec::new();
        for u in 0..n {
            for v in (u + 1)..n {
                if adj[u * n + v] {
                    list.push((u, v));
                }
            }
        }
        let mut q = Qubo::new(n);
        for v in 0..n {
            q.linear(v, weight);
        }
        for &(u, v) in &list {
            // (1 - x_u)(1 - x_v) = 1 - x_u - x_v + x_u x_v
            q.constant(penalty);
            q.linear(u, -penalty);
            q.linear(v, -penalty);
            q.quadratic(u, v, penalty);
        }
        let (graph, offset) = q.into_graph();
        Ok(VertexCover { n, edges: list, weight, penalty, graph, offset })
    }

    /// The spin model.
    #[must_use]
    pub fn graph(&self) -> &Graph {
        &self.graph
    }

    /// The constant with `H(x) = graph.energy(s) + offset` at every state.
    #[must_use]
    pub fn offset(&self) -> f64 {
        self.offset
    }

    /// Cost per chosen vertex, the `B` of the inequality.
    #[must_use]
    pub fn weight(&self) -> f64 {
        self.weight
    }

    /// Penalty per uncovered edge, the `A` of the inequality.
    #[must_use]
    pub fn penalty(&self) -> f64 {
        self.penalty
    }

    /// The value the penalty must strictly exceed, which is the vertex weight.
    #[must_use]
    pub fn threshold(&self) -> f64 {
        self.weight
    }

    /// Whether the penalty exceeds its threshold, so every ground state is a minimum cover.
    #[must_use = "false means the ground state may leave an edge uncovered"]
    pub fn guarantees_feasible(&self) -> bool {
        self.penalty > self.threshold()
    }

    /// The deduplicated edge list this model was built from.
    #[must_use]
    pub fn edges(&self) -> &[(usize, usize)] {
        &self.edges
    }

    /// The cover, in increasing order, or the first edge it fails to cover.
    ///
    /// # Errors
    ///
    /// [`DecodeError::WrongLength`], [`DecodeError::NotASpin`], or [`DecodeError::EdgeUncovered`].
    pub fn decode(&self, s: &[i8]) -> Result<Vec<usize>, DecodeError> {
        let x = binary(s, self.n)?;
        for &(u, v) in &self.edges {
            if x[u] == 0 && x[v] == 0 {
                return Err(DecodeError::EdgeUncovered { u, v });
            }
        }
        Ok((0..self.n).filter(|&v| x[v] == 1).collect())
    }
}

// ---------------------------------------------------------------------------------------------
// set cover
// ---------------------------------------------------------------------------------------------

/// Minimum set cover (Lucas section 5.1).
///
/// This is the first formulation here that needs ancillas, because "at least one chosen subset
/// contains element a" is an INEQUALITY and a quadratic penalty can only state an equality. Lucas's
/// device is a one-hot counter: `y_{a,m} = 1` when exactly `m` chosen subsets contain element `a`,
/// for `m` in `1..=M` where `M` is the largest number of subsets any single element appears in.
///
/// ```text
///   H_A  =  A sum_a (1 - sum_m y_{a,m})^2  +  A sum_a (sum_m m y_{a,m} - sum_{i : a in S_i} x_i)^2
///   H_B  =  B sum_i x_i
/// ```
///
/// The first square says element `a` has exactly one count; the second says that count is the
/// truth. Together they are zero exactly when every element is covered at least once, and cost `A`
/// per uncovered element once the free ancillas are at their best -- which is why the inequality
/// stays the covering one.
///
/// **The inequality is `A > B`.** Minimising over the ancillas first (they appear in no other term)
/// leaves `A * (uncovered elements) + B * (chosen subsets)`, and `k` uncovered elements are covered
/// by at most `k` subsets. Tight: two elements each covered by one subset of their own tie at
/// `A = B`, measured in
/// `set_cover_penalty_threshold_measured_by_enumeration_is_exactly_the_subset_weight`.
///
/// Spins are `subsets + universe * M`, and `M` is a property of the instance, not a tuning knob.
pub struct SetCover {
    universe: usize,
    subsets: Vec<Vec<usize>>,
    max_count: usize,
    weight: f64,
    penalty: f64,
    graph: Graph,
    offset: f64,
}

impl SetCover {
    /// A minimum-set-cover model, with subset weight `1` and penalty `2`.
    ///
    /// # Errors
    ///
    /// [`InstanceError::Empty`] for an empty universe or an empty collection,
    /// [`InstanceError::EmptySubset`], [`InstanceError::ElementOutOfRange`], or
    /// [`InstanceError::Uncoverable`] when some element appears in no subset at all.
    pub fn new(universe: usize, subsets: &[Vec<usize>]) -> Result<SetCover, InstanceError> {
        SetCover::with_weights(universe, subsets, 1.0, 2.0)
    }

    /// The same model with the two weights chosen. As [`MaxClique::with_weights`], the inequality
    /// is reported rather than enforced.
    ///
    /// # Errors
    ///
    /// [`InstanceError`] as [`SetCover::new`], or [`InstanceError::BadWeight`].
    pub fn with_weights(
        universe: usize,
        subsets: &[Vec<usize>],
        weight: f64,
        penalty: f64,
    ) -> Result<SetCover, InstanceError> {
        let weight = positive("subset weight", weight)?;
        let penalty = positive("coverage penalty", penalty)?;
        if universe == 0 {
            return Err(InstanceError::Empty { what: "universe" });
        }
        if subsets.is_empty() {
            return Err(InstanceError::Empty { what: "collection of subsets" });
        }
        let mut clean: Vec<Vec<usize>> = Vec::with_capacity(subsets.len());
        let mut covering: Vec<Vec<usize>> = vec![Vec::new(); universe];
        for (i, sub) in subsets.iter().enumerate() {
            if sub.is_empty() {
                return Err(InstanceError::EmptySubset { subset: i });
            }
            let mut members: Vec<usize> = Vec::with_capacity(sub.len());
            for &e in sub {
                if e >= universe {
                    return Err(InstanceError::ElementOutOfRange {
                        subset: i,
                        element: e,
                        universe,
                    });
                }
                if !members.contains(&e) {
                    members.push(e);
                    covering[e].push(i);
                }
            }
            members.sort_unstable();
            clean.push(members);
        }
        for (e, who) in covering.iter().enumerate() {
            if who.is_empty() {
                return Err(InstanceError::Uncoverable { element: e });
            }
        }
        let max_count = covering.iter().map(Vec::len).max().unwrap_or(0);

        let nsub = clean.len();
        let n = nsub + universe * max_count;
        // Ancilla for "element a is covered exactly m times", m in 1..=max_count.
        let anc = |a: usize, m: usize| nsub + a * max_count + (m - 1);
        let mut q = Qubo::new(n);
        for i in 0..nsub {
            q.linear(i, weight);
        }
        for a in 0..universe {
            let one_hot: Vec<(usize, f64)> =
                (1..=max_count).map(|m| (anc(a, m), -1.0)).collect();
            q.square(1.0, &one_hot, penalty);
            let mut count: Vec<(usize, f64)> =
                (1..=max_count).map(|m| (anc(a, m), m as f64)).collect();
            for &i in &covering[a] {
                count.push((i, -1.0));
            }
            q.square(0.0, &count, penalty);
        }
        let (graph, offset) = q.into_graph();
        Ok(SetCover { universe, subsets: clean, max_count, weight, penalty, graph, offset })
    }

    /// The spin model.
    #[must_use]
    pub fn graph(&self) -> &Graph {
        &self.graph
    }

    /// The constant with `H(x) = graph.energy(s) + offset` at every state.
    #[must_use]
    pub fn offset(&self) -> f64 {
        self.offset
    }

    /// Cost per chosen subset, the `B` of the inequality.
    #[must_use]
    pub fn weight(&self) -> f64 {
        self.weight
    }

    /// Penalty on each counting constraint, the `A` of the inequality.
    #[must_use]
    pub fn penalty(&self) -> f64 {
        self.penalty
    }

    /// The value the penalty must strictly exceed, which is the subset weight.
    #[must_use]
    pub fn threshold(&self) -> f64 {
        self.weight
    }

    /// Whether the penalty exceeds its threshold, so every ground state is a minimum cover.
    #[must_use = "false means the ground state may leave an element uncovered"]
    pub fn guarantees_feasible(&self) -> bool {
        self.penalty > self.threshold()
    }

    /// The largest number of subsets any one element belongs to.
    ///
    /// This is `M`, the width of each element's counting ancilla, and it is read off the instance:
    /// a smaller `M` would make some feasible covers unrepresentable and a larger one would add
    /// spins that can only ever be zero.
    #[must_use]
    pub fn max_count(&self) -> usize {
        self.max_count
    }

    /// The chosen subsets, in increasing order, or why the state is not a cover.
    ///
    /// Checks the ancillas too: a state that covers every element while its counters disagree is
    /// not the encoding, it merely contains a cover.
    ///
    /// # Errors
    ///
    /// [`DecodeError::WrongLength`], [`DecodeError::NotASpin`], [`DecodeError::NotOneHot`],
    /// [`DecodeError::ElementUncovered`], or [`DecodeError::CountMismatch`].
    pub fn decode(&self, s: &[i8]) -> Result<Vec<usize>, DecodeError> {
        let nsub = self.subsets.len();
        let x = binary(s, self.graph.n)?;
        let chosen: Vec<usize> = (0..nsub).filter(|&i| x[i] == 1).collect();
        for a in 0..self.universe {
            let base = nsub + a * self.max_count;
            let set: Vec<usize> =
                (0..self.max_count).filter(|&m| x[base + m] == 1).collect();
            if set.len() != 1 {
                return Err(DecodeError::NotOneHot {
                    group: Group::Coverage(a),
                    set: set.len(),
                });
            }
            let covering = chosen.iter().filter(|&&i| self.subsets[i].contains(&a)).count();
            if covering == 0 {
                return Err(DecodeError::ElementUncovered { element: a });
            }
            let encoded = set[0] + 1;
            if encoded != covering {
                return Err(DecodeError::CountMismatch { element: a, encoded, covering });
            }
        }
        Ok(chosen)
    }
}

// ---------------------------------------------------------------------------------------------
// graph colouring
// ---------------------------------------------------------------------------------------------

/// Graph `k`-colouring (Lucas section 6.1).
///
/// `x_{v,c} = 1` when vertex `v` takes colour `c`, and
///
/// ```text
///   H  =  A sum_v (1 - sum_c x_{v,c})^2  +  A sum over edges (u,v) sum_c x_{u,c} x_{v,c}
/// ```
///
/// **There is no objective, so the only inequality is `A > 0`** -- this is a decision problem, and
/// the whole claim is that the ground energy is zero exactly when the graph is `k`-colourable. That
/// makes the asymmetric test the interesting one: `K_4` at `k = 3` and the 5-cycle at `k = 2` must
/// have ground energy STRICTLY above zero, which a formulation that had quietly dropped a
/// constraint would not.
///
/// A word on what the ground energy is when the graph is not colourable: **it is NOT the minimum
/// number of monochromatic edges**. With both terms weighted `A`, leaving a vertex uncoloured costs
/// one `A` while colouring it into two conflicts costs two, so the minimiser abandons vertices
/// rather than over-colouring them. Measured: `K_4` with one colour has ground energy **3** -- one
/// coloured vertex and three abandoned -- while every full one-colouring has **6** monochromatic
/// edges. The zero/non-zero verdict is exact; the value above zero is not a conflict count, and
/// this module does not claim it is.
pub struct Colouring {
    n: usize,
    k: usize,
    edges: Vec<(usize, usize)>,
    penalty: f64,
    graph: Graph,
    offset: f64,
}

impl Colouring {
    /// A `k`-colouring model over `n` vertices, penalty `1`.
    ///
    /// # Errors
    ///
    /// [`InstanceError::NoColours`], or [`InstanceError`] for a malformed edge list.
    pub fn new(n: usize, edges: &[(usize, usize)], k: usize) -> Result<Colouring, InstanceError> {
        Colouring::with_penalty(n, edges, k, 1.0)
    }

    /// The same model with the penalty chosen. Any positive penalty is correct here; the scale
    /// matters to a sampler's temperature ladder and not to the ground state.
    ///
    /// # Errors
    ///
    /// [`InstanceError`] as [`Colouring::new`], or [`InstanceError::BadWeight`].
    pub fn with_penalty(
        n: usize,
        edges: &[(usize, usize)],
        k: usize,
        penalty: f64,
    ) -> Result<Colouring, InstanceError> {
        let penalty = positive("colouring penalty", penalty)?;
        if k == 0 {
            return Err(InstanceError::NoColours);
        }
        let adj = adjacency(n, edges)?;
        let mut list = Vec::new();
        for u in 0..n {
            for v in (u + 1)..n {
                if adj[u * n + v] {
                    list.push((u, v));
                }
            }
        }
        let mut q = Qubo::new(n * k);
        for v in 0..n {
            let one_hot: Vec<(usize, f64)> = (0..k).map(|c| (v * k + c, -1.0)).collect();
            q.square(1.0, &one_hot, penalty);
        }
        for &(u, v) in &list {
            for c in 0..k {
                q.quadratic(u * k + c, v * k + c, penalty);
            }
        }
        let (graph, offset) = q.into_graph();
        Ok(Colouring { n, k, edges: list, penalty, graph, offset })
    }

    /// The spin model. Ground energy zero exactly when the graph is `k`-colourable.
    #[must_use]
    pub fn graph(&self) -> &Graph {
        &self.graph
    }

    /// The constant with `H(x) = graph.energy(s) + offset` at every state.
    #[must_use]
    pub fn offset(&self) -> f64 {
        self.offset
    }

    /// Penalty on each constraint.
    #[must_use]
    pub fn penalty(&self) -> f64 {
        self.penalty
    }

    /// Zero: with no objective to outbid, any positive penalty suffices.
    #[must_use]
    pub fn threshold(&self) -> f64 {
        0.0
    }

    /// Whether the penalty exceeds its threshold, which for a pure feasibility model is always.
    #[must_use = "false means the model states no constraint at all"]
    pub fn guarantees_feasible(&self) -> bool {
        self.penalty > self.threshold()
    }

    /// Colours available per vertex.
    #[must_use]
    pub fn colours(&self) -> usize {
        self.k
    }

    /// The colour of each vertex, or why the state is not a proper colouring.
    ///
    /// # Errors
    ///
    /// [`DecodeError::WrongLength`], [`DecodeError::NotASpin`], [`DecodeError::NotOneHot`] for a
    /// vertex with anything but one colour, or [`DecodeError::ColourConflict`].
    pub fn decode(&self, s: &[i8]) -> Result<Vec<usize>, DecodeError> {
        let x = binary(s, self.graph.n)?;
        let mut colour = vec![0usize; self.n];
        for v in 0..self.n {
            let set: Vec<usize> = (0..self.k).filter(|&c| x[v * self.k + c] == 1).collect();
            if set.len() != 1 {
                return Err(DecodeError::NotOneHot {
                    group: Group::VertexColour(v),
                    set: set.len(),
                });
            }
            colour[v] = set[0];
        }
        for &(u, v) in &self.edges {
            if colour[u] == colour[v] {
                return Err(DecodeError::ColourConflict { u, v, colour: colour[u] });
            }
        }
        Ok(colour)
    }
}

// ---------------------------------------------------------------------------------------------
// Hamiltonian cycles, and the travelling salesman
// ---------------------------------------------------------------------------------------------

/// The three penalty groups a cyclic ordering needs, shared by [`HamiltonianCycle`] and [`Tsp`].
///
/// `x_{v,j} = 1` when vertex `v` sits at position `j` of the cycle, indexed `v * n + j`:
///
/// ```text
///   sum_v (1 - sum_j x_{v,j})^2     every vertex appears exactly once
///   sum_j (1 - sum_v x_{v,j})^2     every position holds exactly one vertex
///   sum_j sum over non-edges (u,v) of x_{u,j} x_{v,j+1}     no step crosses a missing edge
/// ```
///
/// with `j + 1` taken modulo `n`, which is what makes it a cycle rather than a path.
fn cycle_penalties(q: &mut Qubo, n: usize, adj: &[bool], penalty: f64) {
    for v in 0..n {
        let row: Vec<(usize, f64)> = (0..n).map(|j| (v * n + j, -1.0)).collect();
        q.square(1.0, &row, penalty);
    }
    for j in 0..n {
        let col: Vec<(usize, f64)> = (0..n).map(|v| (v * n + j, -1.0)).collect();
        q.square(1.0, &col, penalty);
    }
    for j in 0..n {
        let next = (j + 1) % n;
        for u in 0..n {
            for v in 0..n {
                if u != v && !adj[u * n + v] {
                    q.quadratic(u * n + j, v * n + next, penalty);
                }
            }
        }
    }
}

/// Read a cyclic ordering out of an `n * n` one-hot block, checking both directions.
fn decode_cycle(x: &[u8], n: usize) -> Result<Vec<usize>, DecodeError> {
    let mut at = vec![usize::MAX; n];
    for v in 0..n {
        let set: Vec<usize> = (0..n).filter(|&j| x[v * n + j] == 1).collect();
        if set.len() != 1 {
            return Err(DecodeError::NotOneHot { group: Group::City(v), set: set.len() });
        }
        at[v] = set[0];
    }
    let mut order = vec![usize::MAX; n];
    for j in 0..n {
        let set: Vec<usize> = (0..n).filter(|&v| x[v * n + j] == 1).collect();
        if set.len() != 1 {
            return Err(DecodeError::NotOneHot { group: Group::Position(j), set: set.len() });
        }
        order[j] = set[0];
    }
    // Rows one-hot and columns one-hot together already force a permutation; `at` is kept so a
    // future change that drops one of the two groups cannot pass this decoder silently.
    for v in 0..n {
        debug_assert_eq!(order[at[v]], v, "one-hot rows and columns must agree");
    }
    Ok(order)
}

/// Hamiltonian cycle (Lucas section 7.1).
///
/// A pure feasibility model: the ground energy is zero exactly when the graph has a Hamiltonian
/// cycle. See [`cycle_penalties`] for the three constraint groups.
///
/// **There is no objective, so the only inequality is `A > 0`.** The asymmetric test is that a
/// graph with no Hamiltonian cycle -- a path, a star -- has ground energy strictly above zero, and
/// the oracle is direct enumeration of the `(n-1)!/2` cyclic orders.
///
/// `n >= 3`: at two vertices the successor of position 1 is position 0 and the single edge would be
/// its own return leg, which is not what the formulation means.
pub struct HamiltonianCycle {
    n: usize,
    adj: Vec<bool>,
    penalty: f64,
    graph: Graph,
    offset: f64,
}

impl HamiltonianCycle {
    /// A Hamiltonian-cycle model over `n >= 3` vertices, penalty `1`.
    ///
    /// # Errors
    ///
    /// [`InstanceError::TooSmall`] below three vertices, or [`InstanceError`] for a malformed edge
    /// list.
    pub fn new(n: usize, edges: &[(usize, usize)]) -> Result<HamiltonianCycle, InstanceError> {
        HamiltonianCycle::with_penalty(n, edges, 1.0)
    }

    /// The same model with the penalty chosen.
    ///
    /// # Errors
    ///
    /// [`InstanceError`] as [`HamiltonianCycle::new`], or [`InstanceError::BadWeight`].
    pub fn with_penalty(
        n: usize,
        edges: &[(usize, usize)],
        penalty: f64,
    ) -> Result<HamiltonianCycle, InstanceError> {
        let penalty = positive("cycle penalty", penalty)?;
        if n < 3 {
            return Err(InstanceError::TooSmall { n, min: 3 });
        }
        let adj = adjacency(n, edges)?;
        let mut q = Qubo::new(n * n);
        cycle_penalties(&mut q, n, &adj, penalty);
        let (graph, offset) = q.into_graph();
        Ok(HamiltonianCycle { n, adj, penalty, graph, offset })
    }

    /// The spin model. Ground energy zero exactly when a Hamiltonian cycle exists.
    #[must_use]
    pub fn graph(&self) -> &Graph {
        &self.graph
    }

    /// The constant with `H(x) = graph.energy(s) + offset` at every state.
    #[must_use]
    pub fn offset(&self) -> f64 {
        self.offset
    }

    /// Penalty on each constraint.
    #[must_use]
    pub fn penalty(&self) -> f64 {
        self.penalty
    }

    /// Zero: with no objective to outbid, any positive penalty suffices.
    #[must_use]
    pub fn threshold(&self) -> f64 {
        0.0
    }

    /// Whether the penalty exceeds its threshold, which for a pure feasibility model is always.
    #[must_use = "false means the model states no constraint at all"]
    pub fn guarantees_feasible(&self) -> bool {
        self.penalty > self.threshold()
    }

    /// The cycle as a vertex sequence, position 0 first, or why the state is not a cycle.
    ///
    /// # Errors
    ///
    /// [`DecodeError::WrongLength`], [`DecodeError::NotASpin`], [`DecodeError::NotOneHot`], or
    /// [`DecodeError::StepNotAnEdge`] for a step the graph has no edge for.
    pub fn decode(&self, s: &[i8]) -> Result<Vec<usize>, DecodeError> {
        let x = binary(s, self.graph.n)?;
        let order = decode_cycle(&x, self.n)?;
        for j in 0..self.n {
            let (from, to) = (order[j], order[(j + 1) % self.n]);
            if !self.adj[from * self.n + to] {
                return Err(DecodeError::StepNotAnEdge { from, to, step: j });
            }
        }
        Ok(order)
    }
}

/// The symmetric travelling salesman (Lucas section 7.2).
///
/// The [`HamiltonianCycle`] constraints, plus the tour's own cost:
///
/// ```text
///   H  =  A * (cycle constraints)  +  B sum_j sum_{u != v} W_{uv} x_{u,j} x_{v,j+1}
/// ```
///
/// Weights arrive as a symmetric `n * n` row-major matrix. The diagonal is ignored. **An infinite
/// entry marks an absent edge**, exactly as [`crate::dimacs`] marks a hard clause, and such a step
/// is charged `A` by the cycle penalty instead of a weight.
///
/// **The inequality is `A > B T*`**, `T*` being the optimal tour cost: an infeasible state pays at
/// least one whole `A` and carries only non-negative weight, so nothing below the best feasible
/// energy `B T*` is reachable. `T*` is not known in advance, so [`Tsp::threshold`] reports `B`
/// times an upper bound -- the `n` largest edge weights, accumulated with [`crate::round::sum_up`]
/// so the bound is certainly not low -- and [`Tsp::new`] puts the penalty one `B` above that.
///
/// Lucas states the smaller `0 < B max(W) < A`. Measured over 27 four-city instances, `B max(W)` is
/// sufficient on every one and attained on the uniform complete graph, but it is not the exact
/// threshold in general: on `W = [[0,2,5,1],[2,0,3,4],[5,3,0,6],[1,4,6,0]]` the exact critical
/// penalty is 4.5 against a `max(W)` of 6. All three numbers are in
/// `tsp_penalty_threshold_measured_by_enumeration_is_the_largest_edge_weight`; a caller who wants a
/// smaller penalty passes it to [`Tsp::with_weights`] and reads
/// [`Tsp::guarantees_feasible`] for what is then still promised.
pub struct Tsp {
    n: usize,
    w: Vec<f64>,
    weight: f64,
    penalty: f64,
    tour_bound: f64,
    graph: Graph,
    offset: f64,
}

impl Tsp {
    /// A travelling-salesman model over `n >= 3` cities, unit weight scale and the provable
    /// penalty `B (1 + upper bound on T*)`.
    ///
    /// # Errors
    ///
    /// [`InstanceError::TooSmall`], [`InstanceError::Matrix`], [`InstanceError::Asymmetric`], or
    /// [`InstanceError::EdgeWeight`] for an entry that is negative or NaN.
    pub fn new(n: usize, weights: &[f64]) -> Result<Tsp, InstanceError> {
        let bound = Tsp::tour_upper_bound(n, weights)?;
        Tsp::with_weights(n, weights, 1.0, 1.0 + bound)
    }

    /// The same model with the weight scale and penalty chosen. As [`MaxClique::with_weights`], the
    /// inequality is reported by [`Tsp::guarantees_feasible`] rather than enforced.
    ///
    /// # Errors
    ///
    /// [`InstanceError`] as [`Tsp::new`], or [`InstanceError::BadWeight`].
    pub fn with_weights(
        n: usize,
        weights: &[f64],
        weight: f64,
        penalty: f64,
    ) -> Result<Tsp, InstanceError> {
        let weight = positive("tour weight scale", weight)?;
        let penalty = positive("tour penalty", penalty)?;
        let tour_bound = Tsp::tour_upper_bound(n, weights)?;
        let adj: Vec<bool> = (0..n * n).map(|k| weights[k].is_finite()).collect();
        let mut q = Qubo::new(n * n);
        cycle_penalties(&mut q, n, &adj, penalty);
        for j in 0..n {
            let next = (j + 1) % n;
            for u in 0..n {
                for v in 0..n {
                    if u != v && weights[u * n + v].is_finite() {
                        q.quadratic(u * n + j, v * n + next, weight * weights[u * n + v]);
                    }
                }
            }
        }
        let (graph, offset) = q.into_graph();
        Ok(Tsp { n, w: weights.to_vec(), weight, penalty, tour_bound, graph, offset })
    }

    /// Validate the matrix and return an upper bound on any tour's cost.
    ///
    /// The `n` largest finite edge weights, since a tour uses exactly `n` edges and no edge twice.
    /// Summed with [`crate::round::sum_up`]: this number is the floor a penalty has to clear, and a
    /// sum that rounded DOWN would hand back a threshold a sound penalty could sit below.
    fn tour_upper_bound(n: usize, weights: &[f64]) -> Result<f64, InstanceError> {
        if n < 3 {
            return Err(InstanceError::TooSmall { n, min: 3 });
        }
        if weights.len() != n * n {
            return Err(InstanceError::Matrix { got: weights.len(), want: n * n });
        }
        let mut finite: Vec<f64> = Vec::new();
        for u in 0..n {
            for v in (u + 1)..n {
                let (uv, vu) = (weights[u * n + v], weights[v * n + u]);
                // NaN is not equal to itself, so this also rejects a NaN entry as asymmetric only
                // if its mirror differs; the magnitude check below is what refuses NaN outright.
                if uv != vu && !(uv.is_nan() && vu.is_nan()) {
                    return Err(InstanceError::Asymmetric { u, v, uv, vu });
                }
                if uv.is_nan() || uv < 0.0 {
                    return Err(InstanceError::EdgeWeight { u, v, got: uv });
                }
                if uv.is_finite() {
                    finite.push(uv);
                }
            }
        }
        finite.sort_by(|a, b| b.partial_cmp(a).expect("no NaN survives the check above"));
        finite.truncate(n);
        Ok(crate::round::sum_up(&finite))
    }

    /// The spin model.
    #[must_use]
    pub fn graph(&self) -> &Graph {
        &self.graph
    }

    /// The constant with `H(x) = graph.energy(s) + offset` at every state.
    #[must_use]
    pub fn offset(&self) -> f64 {
        self.offset
    }

    /// The scale on the tour cost, the `B` of the inequality.
    #[must_use]
    pub fn weight(&self) -> f64 {
        self.weight
    }

    /// Penalty on each constraint, the `A` of the inequality.
    #[must_use]
    pub fn penalty(&self) -> f64 {
        self.penalty
    }

    /// The value the penalty must strictly exceed for the ground state to be an optimal tour.
    ///
    /// `B` times an upper bound on the optimal tour cost. Conservative on purpose: the tight value
    /// is instance-dependent and, on everything measured here, is `B max(W)`.
    #[must_use]
    pub fn threshold(&self) -> f64 {
        self.weight * self.tour_bound
    }

    /// Whether the penalty exceeds the provable threshold.
    ///
    /// `false` does not mean the model is wrong -- Lucas's much smaller `B max(W)` is enough on
    /// every instance measured here -- it means this module cannot promise it for this instance.
    #[must_use = "false means this module cannot promise the ground state is a tour"]
    pub fn guarantees_feasible(&self) -> bool {
        self.penalty > self.threshold()
    }

    /// Cost of a tour given as a city sequence, or `None` if it uses an absent edge.
    ///
    /// # Panics
    ///
    /// If `tour` is not `n` entries of distinct cities below `n`, which is not a tour of this
    /// instance and has no cost to report.
    #[must_use]
    pub fn tour_cost(&self, tour: &[usize]) -> Option<f64> {
        assert_eq!(tour.len(), self.n, "a tour of {} cities has {} entries", self.n, self.n);
        let mut seen = vec![false; self.n];
        for &c in tour {
            assert!(c < self.n && !seen[c], "city {c} repeats or is out of range");
            seen[c] = true;
        }
        let mut total = 0.0;
        for j in 0..self.n {
            let e = self.w[tour[j] * self.n + tour[(j + 1) % self.n]];
            if !e.is_finite() {
                return None;
            }
            total += e;
        }
        Some(total)
    }

    /// The tour as a city sequence, position 0 first, or why the state is not a tour.
    ///
    /// # Errors
    ///
    /// [`DecodeError::WrongLength`], [`DecodeError::NotASpin`], [`DecodeError::NotOneHot`], or
    /// [`DecodeError::StepNotAnEdge`] for a step across an absent edge.
    pub fn decode(&self, s: &[i8]) -> Result<Vec<usize>, DecodeError> {
        let x = binary(s, self.graph.n)?;
        let order = decode_cycle(&x, self.n)?;
        for j in 0..self.n {
            let (from, to) = (order[j], order[(j + 1) % self.n]);
            if !self.w[from * self.n + to].is_finite() {
                return Err(DecodeError::StepNotAnEdge { from, to, step: j });
            }
        }
        Ok(order)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Pcg;

    // -----------------------------------------------------------------------------------------
    // oracles: brute force over the ORIGINAL problem, and over the spin space
    // -----------------------------------------------------------------------------------------

    /// Every state at the ground energy, by enumerating all `2^n` spin states.
    ///
    /// Exact comparison rather than a tolerance: every coefficient in this module is a multiple of
    /// a quarter and every instance here is small, so each energy is exactly representable and two
    /// states at the same energy compare equal bit for bit. An epsilon would not make this test
    /// more robust, it would let a genuinely different energy join the ground set.
    fn all_ground_states(g: &Graph) -> (f64, Vec<Vec<i8>>) {
        assert!(g.n <= 20, "{} spins is too many to enumerate here", g.n);
        let mut best = f64::INFINITY;
        let mut states: Vec<Vec<i8>> = Vec::new();
        let mut s = vec![-1i8; g.n];
        for mask in 0u32..(1u32 << g.n) {
            for i in 0..g.n {
                s[i] = if (mask >> i) & 1 == 1 { 1 } else { -1 };
            }
            let e = g.energy(&s);
            if e < best {
                best = e;
                states.clear();
                states.push(s.clone());
            } else if e == best {
                states.push(s.clone());
            }
        }
        (best, states)
    }

    /// The exact penalty at which the ground state stops being feasible-and-optimal, in units of
    /// the objective weight.
    ///
    /// `H = A p(x) + obj(x)` with feasible meaning `p(x) == 0`. An infeasible `x` beats or ties the
    /// best feasible objective exactly when `A <= (opt - obj(x)) / p(x)`, so the largest such ratio
    /// IS the threshold: strictly above it the ground state is feasible and optimal, at it there is
    /// a tie, strictly below it an infeasible state wins. This is the measurement the module's
    /// inequalities are checked against, and it depends on nothing in `super` but the formula.
    fn critical_penalty(n: usize, p: impl Fn(&[u8]) -> f64, obj: impl Fn(&[u8]) -> f64) -> f64 {
        let mut x = vec![0u8; n];
        let mut opt = f64::INFINITY;
        for mask in 0u32..(1u32 << n) {
            for i in 0..n {
                x[i] = ((mask >> i) & 1) as u8;
            }
            if p(&x) == 0.0 {
                opt = opt.min(obj(&x));
            }
        }
        assert!(opt.is_finite(), "no feasible state, so there is nothing to be optimal against");
        let mut crit = f64::NEG_INFINITY;
        for mask in 0u32..(1u32 << n) {
            for i in 0..n {
                x[i] = ((mask >> i) & 1) as u8;
            }
            let viol = p(&x);
            if viol > 0.0 {
                crit = crit.max((opt - obj(&x)) / viol);
            }
        }
        crit
    }

    fn subsets_of(n: usize, mask: u32) -> Vec<usize> {
        (0..n).filter(|&v| (mask >> v) & 1 == 1).collect()
    }

    fn is_clique(vs: &[usize], n: usize, adj: &[bool]) -> bool {
        for (a, &u) in vs.iter().enumerate() {
            for &v in &vs[(a + 1)..] {
                if !adj[u * n + v] {
                    return false;
                }
            }
        }
        true
    }

    /// Largest clique, by looking at all `2^n` vertex subsets.
    fn brute_max_clique(n: usize, adj: &[bool]) -> usize {
        let mut best = 0;
        for mask in 0u32..(1u32 << n) {
            let vs = subsets_of(n, mask);
            if is_clique(&vs, n, adj) {
                best = best.max(vs.len());
            }
        }
        best
    }

    /// Smallest vertex cover, by looking at all `2^n` vertex subsets.
    fn brute_min_cover(n: usize, edges: &[(usize, usize)]) -> usize {
        let mut best = n;
        for mask in 0u32..(1u32 << n) {
            let vs = subsets_of(n, mask);
            if edges.iter().all(|&(u, v)| vs.contains(&u) || vs.contains(&v)) {
                best = best.min(vs.len());
            }
        }
        best
    }

    /// Smallest set cover, by looking at all `2^N` sub-collections.
    fn brute_min_set_cover(universe: usize, subsets: &[Vec<usize>]) -> usize {
        let nsub = subsets.len();
        let mut best = usize::MAX;
        for mask in 0u32..(1u32 << nsub) {
            let picked = subsets_of(nsub, mask);
            let covered = |a: usize| picked.iter().any(|&i| subsets[i].contains(&a));
            if (0..universe).all(covered) {
                best = best.min(picked.len());
            }
        }
        best
    }

    /// Whether a proper `k`-colouring exists, by looking at all `k^n` assignments.
    fn brute_colourable(n: usize, edges: &[(usize, usize)], k: usize) -> bool {
        let total = k.pow(n as u32);
        let mut c = vec![0usize; n];
        for code in 0..total {
            let mut rest = code;
            for v in 0..n {
                c[v] = rest % k;
                rest /= k;
            }
            if edges.iter().all(|&(u, v)| c[u] != c[v]) {
                return true;
            }
        }
        false
    }

    fn permute(v: &mut Vec<usize>, k: usize, f: &mut impl FnMut(&[usize])) {
        if k == v.len() {
            f(v);
            return;
        }
        for i in k..v.len() {
            v.swap(k, i);
            permute(v, k + 1, f);
            v.swap(k, i);
        }
    }

    /// Every cyclic order of `0..n` with city 0 pinned, which is every distinct tour twice (once
    /// per direction). `(n-1)!` sequences.
    fn cyclic_orders(n: usize) -> Vec<Vec<usize>> {
        let mut out = Vec::new();
        let mut rest: Vec<usize> = (1..n).collect();
        permute(&mut rest, 0, &mut |p: &[usize]| {
            let mut tour = vec![0usize];
            tour.extend_from_slice(p);
            out.push(tour);
        });
        out
    }

    /// Whether a Hamiltonian cycle exists, by looking at every cyclic order.
    fn brute_hamiltonian(n: usize, adj: &[bool]) -> bool {
        cyclic_orders(n).iter().any(|t| (0..n).all(|j| adj[t[j] * n + t[(j + 1) % n]]))
    }

    /// Cheapest tour, by looking at every cyclic order. `None` when every order uses an absent
    /// edge.
    fn brute_tsp(n: usize, w: &[f64]) -> Option<f64> {
        let mut best = f64::INFINITY;
        for t in cyclic_orders(n) {
            let mut c = 0.0;
            let mut ok = true;
            for j in 0..n {
                let e = w[t[j] * n + t[(j + 1) % n]];
                if e.is_finite() {
                    c += e;
                } else {
                    ok = false;
                    break;
                }
            }
            if ok {
                best = best.min(c);
            }
        }
        best.is_finite().then_some(best)
    }

    fn random_graph(n: usize, seed: u64, p: f64) -> (Vec<bool>, Vec<(usize, usize)>) {
        let mut rng = Pcg::new(seed, 0x4E50);
        let mut adj = vec![false; n * n];
        let mut edges = Vec::new();
        for u in 0..n {
            for v in (u + 1)..n {
                if rng.f64() < p {
                    adj[u * n + v] = true;
                    adj[v * n + u] = true;
                    edges.push((u, v));
                }
            }
        }
        (adj, edges)
    }

    // -----------------------------------------------------------------------------------------
    // the substitution itself
    // -----------------------------------------------------------------------------------------

    /// THE IDENTITY THE WHOLE MODULE RESTS ON, at every state of two models, against Lucas's
    /// formulas written out by hand.
    ///
    /// `H(x) = graph.energy(s) + offset`, exactly. The right-hand side is this module's
    /// `x = (1+s)/2` substitution; the left is the paper's Hamiltonian evaluated directly on binary
    /// variables, with no `Qubo` in the path. Every coefficient is a multiple of a quarter, so this
    /// is asserted with `==` -- a tolerance here would let a dropped incident-coupling term hide,
    /// and that is exactly the term a hand substitution loses.
    #[test]
    fn the_ising_energy_is_the_lucas_hamiltonian_at_every_state() {
        // Maximum clique on a 6-vertex graph.
        let n = 6;
        let (adj, edges) = random_graph(n, 11, 0.5);
        let (reward, penalty) = (1.0, 3.0);
        let mc = MaxClique::with_weights(n, &edges, reward, penalty).unwrap();
        let mut s = vec![-1i8; n];
        for mask in 0u32..(1u32 << n) {
            let mut x = vec![0u8; n];
            for i in 0..n {
                x[i] = ((mask >> i) & 1) as u8;
                s[i] = if x[i] == 1 { 1 } else { -1 };
            }
            let mut h = 0.0;
            for u in 0..n {
                for v in (u + 1)..n {
                    if !adj[u * n + v] && x[u] == 1 && x[v] == 1 {
                        h += penalty;
                    }
                }
            }
            for v in 0..n {
                if x[v] == 1 {
                    h -= reward;
                }
            }
            assert_eq!(
                h,
                mc.graph().energy(&s) + mc.offset(),
                "clique mask {mask:b}: Lucas says {h}, the spin model says {}",
                mc.graph().energy(&s) + mc.offset()
            );
        }

        // Travelling salesman on four cities, which exercises both squares and the step terms.
        let m = 4;
        let w = vec![
            0.0, 2.0, 5.0, 1.0, //
            2.0, 0.0, 3.0, 4.0, //
            5.0, 3.0, 0.0, 6.0, //
            1.0, 4.0, 6.0, 0.0,
        ];
        let (b, a) = (1.0, 9.0);
        let tsp = Tsp::with_weights(m, &w, b, a).unwrap();
        let mut s = vec![-1i8; m * m];
        for mask in 0u32..(1u32 << (m * m)) {
            let mut x = vec![0u8; m * m];
            for i in 0..(m * m) {
                x[i] = ((mask >> i) & 1) as u8;
                s[i] = if x[i] == 1 { 1 } else { -1 };
            }
            let mut h = 0.0;
            for v in 0..m {
                let r: i32 = (0..m).map(|j| i32::from(x[v * m + j])).sum();
                h += a * f64::from((1 - r) * (1 - r));
            }
            for j in 0..m {
                let c: i32 = (0..m).map(|v| i32::from(x[v * m + j])).sum();
                h += a * f64::from((1 - c) * (1 - c));
            }
            for j in 0..m {
                let next = (j + 1) % m;
                for u in 0..m {
                    for v in 0..m {
                        if u != v && x[u * m + j] == 1 && x[v * m + next] == 1 {
                            h += b * w[u * m + v];
                        }
                    }
                }
            }
            assert_eq!(
                h,
                tsp.graph().energy(&s) + tsp.offset(),
                "tsp mask {mask:b}: Lucas says {h}, the spin model says {}",
                tsp.graph().energy(&s) + tsp.offset()
            );
        }
    }

    // -----------------------------------------------------------------------------------------
    // maximum clique
    // -----------------------------------------------------------------------------------------

    /// EVERY ground state decodes to a MAXIMUM clique, against exhaustive subset enumeration.
    ///
    /// Optimality, not feasibility: a formulation that had lost the reward term would still produce
    /// cliques -- the empty set is a clique -- and would fail here on the size.
    #[test]
    fn max_clique_ground_state_is_a_maximum_clique_against_exhaustive_subset_enumeration() {
        let n = 8;
        for seed in 0..20u64 {
            let (adj, edges) = random_graph(n, seed, 0.5);
            let want = brute_max_clique(n, &adj);
            let mc = MaxClique::new(n, &edges).unwrap();
            assert!(mc.guarantees_feasible());
            let (e, states) = all_ground_states(mc.graph());
            assert_eq!(
                e + mc.offset(),
                -(want as f64) * mc.reward(),
                "seed {seed}: ground energy should be minus the reward times the clique number"
            );
            for s in &states {
                let clique = mc.decode(s).expect("a ground state must decode");
                assert!(is_clique(&clique, n, &adj), "seed {seed}: {clique:?} is not a clique");
                assert_eq!(clique.len(), want, "seed {seed}: {clique:?} is not MAXIMUM");
            }
        }
    }

    /// The penalty threshold `A > B`, MEASURED -- and the 4-cycle, where it is attained.
    ///
    /// A reduction tested only above its threshold is untested: the same assertions pass for a
    /// formulation whose penalty inequality is far stricter than claimed. So this pins all three
    /// sides. Below the threshold the ground state must be INFEASIBLE, at it there must be a TIE,
    /// above it the answer must be right.
    #[test]
    fn max_clique_penalty_threshold_measured_by_enumeration_is_exactly_the_reward() {
        let n = 4;
        let edges = [(0usize, 1usize), (1, 2), (2, 3), (3, 0)];
        let adj = adjacency(n, &edges).unwrap();
        let omega = brute_max_clique(n, &adj) as f64;
        assert_eq!(omega, 2.0, "the 4-cycle's largest clique is one edge");

        let crit = critical_penalty(
            n,
            |x| {
                let mut viol = 0.0;
                for u in 0..n {
                    for v in (u + 1)..n {
                        if !adj[u * n + v] && x[u] == 1 && x[v] == 1 {
                            viol += 1.0;
                        }
                    }
                }
                viol
            },
            |x| -f64::from(x.iter().map(|&b| u32::from(b)).sum::<u32>()),
        );
        assert_eq!(crit, 1.0, "the critical penalty is exactly the reward, and it is attained");

        // Below: the whole vertex set, two non-edges and two vertices to the good, wins.
        let soft = MaxClique::with_weights(n, &edges, 1.0, 0.99).unwrap();
        assert!(!soft.guarantees_feasible());
        let (_, states) = all_ground_states(soft.graph());
        for s in &states {
            assert_eq!(
                soft.decode(s),
                Err(DecodeError::NotAClique { u: 0, v: 2 }),
                "below the threshold the ground state must NOT be a clique"
            );
        }

        // At it: a tie, so an infeasible state sits at the ground energy.
        let tied = MaxClique::with_weights(n, &edges, 1.0, 1.0).unwrap();
        let (_, states) = all_ground_states(tied.graph());
        assert!(
            states.iter().any(|s| tied.decode(s).is_err()),
            "at the threshold an infeasible state must TIE the optimum"
        );
        assert!(
            states.iter().any(|s| tied.decode(s).is_ok_and(|c| c.len() == 2)),
            "and a maximum clique must still be among the ground states"
        );

        // Above: right again, and only just above.
        let hard = MaxClique::with_weights(n, &edges, 1.0, 1.01).unwrap();
        assert!(hard.guarantees_feasible());
        let (_, states) = all_ground_states(hard.graph());
        for s in &states {
            assert_eq!(hard.decode(s).unwrap().len(), 2);
        }
    }

    // -----------------------------------------------------------------------------------------
    // minimum vertex cover
    // -----------------------------------------------------------------------------------------

    /// EVERY ground state decodes to a MINIMUM vertex cover, against exhaustive subset enumeration.
    #[test]
    fn vertex_cover_ground_state_is_a_minimum_cover_against_exhaustive_subset_enumeration() {
        let n = 8;
        for seed in 0..20u64 {
            let (_, edges) = random_graph(n, 0x900 + seed, 0.4);
            let want = brute_min_cover(n, &edges);
            let vc = VertexCover::new(n, &edges).unwrap();
            assert!(vc.guarantees_feasible());
            let (e, states) = all_ground_states(vc.graph());
            assert_eq!(
                e + vc.offset(),
                want as f64 * vc.weight(),
                "seed {seed}: ground energy should be the weight times the cover number"
            );
            for s in &states {
                let cover = vc.decode(s).expect("a ground state must decode");
                assert!(
                    edges.iter().all(|&(u, v)| cover.contains(&u) || cover.contains(&v)),
                    "seed {seed}: {cover:?} is not a cover"
                );
                assert_eq!(cover.len(), want, "seed {seed}: {cover:?} is not MINIMUM");
            }
        }
    }

    /// The penalty threshold `A > B`, MEASURED on the triangle, where one vertex leaves exactly one
    /// edge uncovered and a minimum cover needs two.
    #[test]
    fn vertex_cover_penalty_threshold_measured_by_enumeration_is_exactly_the_vertex_weight() {
        let n = 3;
        let edges = [(0usize, 1usize), (1, 2), (2, 0)];
        assert_eq!(brute_min_cover(n, &edges), 2);

        let crit = critical_penalty(
            n,
            |x| edges.iter().filter(|&&(u, v)| x[u] == 0 && x[v] == 0).count() as f64,
            |x| f64::from(x.iter().map(|&b| u32::from(b)).sum::<u32>()),
        );
        assert_eq!(crit, 1.0, "the critical penalty is exactly the vertex weight");

        let soft = VertexCover::with_weights(n, &edges, 1.0, 0.99).unwrap();
        assert!(!soft.guarantees_feasible());
        let (_, states) = all_ground_states(soft.graph());
        for s in &states {
            assert!(
                matches!(soft.decode(s), Err(DecodeError::EdgeUncovered { .. })),
                "below the threshold the ground state must leave an edge uncovered"
            );
        }

        let tied = VertexCover::with_weights(n, &edges, 1.0, 1.0).unwrap();
        let (_, states) = all_ground_states(tied.graph());
        assert!(states.iter().any(|s| tied.decode(s).is_err()), "at the threshold, a tie");

        let hard = VertexCover::with_weights(n, &edges, 1.0, 1.01).unwrap();
        let (_, states) = all_ground_states(hard.graph());
        for s in &states {
            assert_eq!(hard.decode(s).unwrap().len(), 2);
        }
    }

    // -----------------------------------------------------------------------------------------
    // set cover
    // -----------------------------------------------------------------------------------------

    /// EVERY ground state decodes to a MINIMUM set cover, against exhaustive sub-collection
    /// enumeration -- ancillas included, which is the part the reduction can silently get wrong.
    #[test]
    fn set_cover_ground_state_is_a_minimum_cover_against_exhaustive_subset_enumeration() {
        let instances: [(usize, Vec<Vec<usize>>); 5] = [
            (3, vec![vec![0], vec![1], vec![2], vec![0, 1]]),
            (3, vec![vec![0, 1], vec![1, 2], vec![0, 2]]),
            (4, vec![vec![0, 1], vec![2, 3], vec![0, 2], vec![1, 3]]),
            (3, vec![vec![0, 1, 2], vec![0], vec![1], vec![2]]),
            (4, vec![vec![0, 1, 2], vec![2, 3], vec![3], vec![0]]),
        ];
        for (universe, subsets) in &instances {
            let want = brute_min_set_cover(*universe, subsets);
            let sc = SetCover::new(*universe, subsets).unwrap();
            assert!(sc.guarantees_feasible());
            assert!(sc.graph().n <= 18, "{} spins", sc.graph().n);
            let (e, states) = all_ground_states(sc.graph());
            assert_eq!(
                e + sc.offset(),
                want as f64 * sc.weight(),
                "{subsets:?}: the ground energy should be the weight times the cover number, with \
                 every penalty term at zero"
            );
            for s in &states {
                let picked = sc.decode(s).expect("a ground state must decode, ancillas included");
                for a in 0..*universe {
                    assert!(
                        picked.iter().any(|&i| subsets[i].contains(&a)),
                        "{subsets:?}: element {a} uncovered by {picked:?}"
                    );
                }
                assert_eq!(picked.len(), want, "{subsets:?}: {picked:?} is not MINIMUM");
            }
        }
    }

    /// The penalty threshold `A > B`, MEASURED -- on an instance where dropping one subset uncovers
    /// exactly one element, which is where the one-for-one repair is tight.
    ///
    /// The measurement runs over the FULL spin space, ancillas and all, so it also proves the claim
    /// that the free ancillas reduce the effective penalty to `A` per uncovered element. If they
    /// did not, the critical value would not be one.
    #[test]
    fn set_cover_penalty_threshold_measured_by_enumeration_is_exactly_the_subset_weight() {
        let universe = 3;
        let subsets = vec![vec![0usize], vec![1usize], vec![2usize], vec![0usize, 1usize]];
        assert_eq!(brute_min_set_cover(universe, &subsets), 2);
        let nsub = subsets.len();
        let max_count = (0..universe)
            .map(|a| subsets.iter().filter(|s| s.contains(&a)).count())
            .max()
            .unwrap();
        let n = nsub + universe * max_count;
        let anc = |a: usize, m: usize| nsub + a * max_count + (m - 1);

        let crit = critical_penalty(
            n,
            |x| {
                let mut viol = 0.0;
                for a in 0..universe {
                    let ones: i32 = (1..=max_count).map(|m| i32::from(x[anc(a, m)])).sum();
                    let count: i32 =
                        (1..=max_count).map(|m| m as i32 * i32::from(x[anc(a, m)])).sum();
                    let covering: i32 = (0..nsub)
                        .filter(|&i| subsets[i].contains(&a))
                        .map(|i| i32::from(x[i]))
                        .sum();
                    viol += f64::from((1 - ones) * (1 - ones));
                    viol += f64::from((count - covering) * (count - covering));
                }
                viol
            },
            |x| f64::from((0..nsub).map(|i| u32::from(x[i])).sum::<u32>()),
        );
        assert_eq!(crit, 1.0, "the critical penalty is exactly the subset weight");

        let soft = SetCover::with_weights(universe, &subsets, 1.0, 0.99).unwrap();
        assert!(!soft.guarantees_feasible());
        let (_, states) = all_ground_states(soft.graph());
        for s in &states {
            assert!(soft.decode(s).is_err(), "below the threshold the ground state is not a cover");
        }

        let hard = SetCover::with_weights(universe, &subsets, 1.0, 1.01).unwrap();
        let (_, states) = all_ground_states(hard.graph());
        for s in &states {
            assert_eq!(hard.decode(s).unwrap().len(), 2);
        }
    }

    // -----------------------------------------------------------------------------------------
    // colouring
    // -----------------------------------------------------------------------------------------

    /// Ground energy zero EXACTLY when a proper colouring exists, against `k^n` enumeration.
    ///
    /// Both directions, and the failing direction is the one that matters: `K_4` at three colours
    /// and the 5-cycle at two must come out STRICTLY positive. A formulation that had dropped the
    /// conflict term would pass the colourable half of this test and nothing else.
    #[test]
    fn colouring_ground_energy_is_zero_exactly_when_brute_force_finds_a_proper_colouring() {
        let k4: Vec<(usize, usize)> = vec![(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
        let c5: Vec<(usize, usize)> = vec![(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)];
        let cases = [(4, &k4, 3), (4, &k4, 4), (5, &c5, 3), (5, &c5, 2)];
        for (n, edges, k) in cases {
            let edges: &[(usize, usize)] = edges;
            let col = Colouring::new(n, edges, k).unwrap();
            assert!(col.guarantees_feasible());
            let (e, states) = all_ground_states(col.graph());
            let ground = e + col.offset();
            let want = brute_colourable(n, edges, k);
            if want {
                assert_eq!(ground, 0.0, "n={n} k={k}: colourable, so the ground energy is zero");
                for s in &states {
                    let colours = col.decode(s).expect("a zero-energy state is a proper colouring");
                    assert!(edges.iter().all(|&(u, v)| colours[u] != colours[v]));
                    assert!(colours.iter().all(|&c| c < k));
                }
            } else {
                assert!(
                    ground > 0.0,
                    "n={n} k={k}: NOT colourable, and the ground energy is {ground}"
                );
                for s in &states {
                    assert!(col.decode(s).is_err(), "no ground state can be a proper colouring");
                }
            }
        }

        // AND THE GROUND ENERGY ABOVE ZERO IS NOT A CONFLICT COUNT, which the module doc claims and
        // this measures. `K_4` with ONE colour: every full colouring has all six edges
        // monochromatic, but the minimiser leaves vertices uncoloured instead -- one colour and
        // three abandoned vertices cost 3, against 6 for colouring them all. A reader who took the
        // ground energy for a conflict count would be off by a factor of two here.
        let one = Colouring::new(4, &k4, 1).expect("one colour is a colouring");
        let (e, _) = all_ground_states(one.graph());
        assert_eq!(e + one.offset(), 3.0, "abandoning three vertices beats colouring all four");
        assert_eq!(k4.len(), 6, "and every full 1-colouring of K4 has six monochromatic edges");
    }

    // -----------------------------------------------------------------------------------------
    // Hamiltonian cycle
    // -----------------------------------------------------------------------------------------

    /// Ground energy zero EXACTLY when a Hamiltonian cycle exists, against cyclic-order
    /// enumeration.
    ///
    /// The star and the path are the asymmetric half: they have no cycle, and a formulation missing
    /// the non-edge term would happily report zero for them.
    #[test]
    fn hamiltonian_cycle_ground_energy_is_zero_exactly_when_permutation_enumeration_finds_one() {
        let cases: [(usize, Vec<(usize, usize)>); 5] = [
            (3, vec![(0, 1), (1, 2), (2, 0)]),                         // triangle: yes
            (4, vec![(0, 1), (1, 2), (2, 3), (3, 0)]),                 // 4-cycle: yes
            (4, vec![(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]), // K4: yes
            (4, vec![(0, 1), (1, 2), (2, 3)]),                         // path: no
            (4, vec![(0, 1), (0, 2), (0, 3)]),                         // star: no
        ];
        for (n, edges) in &cases {
            let adj = adjacency(*n, edges).unwrap();
            let want = brute_hamiltonian(*n, &adj);
            let hc = HamiltonianCycle::new(*n, edges).unwrap();
            let (e, states) = all_ground_states(hc.graph());
            let ground = e + hc.offset();
            if want {
                assert_eq!(ground, 0.0, "n={n} {edges:?}: a cycle exists");
                for s in &states {
                    let tour = hc.decode(s).expect("a zero-energy state is a cycle");
                    let mut seen = vec![false; *n];
                    for &c in &tour {
                        assert!(!seen[c], "the tour repeats city {c}");
                        seen[c] = true;
                    }
                    assert!((0..*n).all(|j| adj[tour[j] * n + tour[(j + 1) % n]]));
                }
            } else {
                assert!(ground > 0.0, "n={n} {edges:?}: no cycle, ground energy {ground}");
                for s in &states {
                    assert!(hc.decode(s).is_err());
                }
            }
        }
    }

    // -----------------------------------------------------------------------------------------
    // travelling salesman
    // -----------------------------------------------------------------------------------------

    /// EVERY ground state decodes to an OPTIMAL tour, against enumeration of every cyclic order.
    ///
    /// One of the instances has an absent edge, so a formulation that had let a missing edge cost
    /// nothing would return the cheap tour that crosses it.
    #[test]
    fn tsp_ground_state_is_an_optimal_tour_against_permutation_enumeration() {
        let inf = f64::INFINITY;
        let cases: [Vec<f64>; 4] = [
            vec![
                0.0, 1.0, 1.0, 1.0, //
                1.0, 0.0, 1.0, 1.0, //
                1.0, 1.0, 0.0, 1.0, //
                1.0, 1.0, 1.0, 0.0,
            ],
            vec![
                0.0, 2.0, 5.0, 1.0, //
                2.0, 0.0, 3.0, 4.0, //
                5.0, 3.0, 0.0, 6.0, //
                1.0, 4.0, 6.0, 0.0,
            ],
            vec![
                0.0, 1.0, 7.0, 3.0, //
                1.0, 0.0, 2.0, 9.0, //
                7.0, 2.0, 0.0, 1.0, //
                3.0, 9.0, 1.0, 0.0,
            ],
            // An absent edge (0,2): the cheap tour through it must not be reachable.
            vec![
                0.0, 1.0, inf, 1.0, //
                1.0, 0.0, 1.0, 8.0, //
                inf, 1.0, 0.0, 1.0, //
                1.0, 8.0, 1.0, 0.0,
            ],
        ];
        for w in &cases {
            let n = 4;
            let want = brute_tsp(n, w).expect("each instance has a tour");
            let tsp = Tsp::new(n, w).unwrap();
            assert!(tsp.guarantees_feasible());
            let (e, states) = all_ground_states(tsp.graph());
            // A TOLERANCE HERE AND NOWHERE ELSE IN THIS FILE. The default penalty comes from
            // `round::sum_up`, which adds a rounding guard on purpose, so it is not an integer and
            // neither is the offset it feeds. Every other model here has exact coefficients and is
            // asserted with `==`; this one is 1.4e-14 off a whole number because the guard is real.
            assert!(
                (e + tsp.offset() - want * tsp.weight()).abs() < 1e-9,
                "{w:?}: ground energy {} is not the tour cost {want}",
                e + tsp.offset()
            );
            for s in &states {
                let tour = tsp.decode(s).expect("a ground state must be a tour");
                assert_eq!(tsp.tour_cost(&tour), Some(want), "{tour:?} is not OPTIMAL");
            }
        }
    }

    /// The travelling salesman's penalty threshold, MEASURED, against both claims about it.
    ///
    /// Three numbers come out of this:
    ///
    ///   * the exact critical penalty, `max over infeasible states of (T* - W(x)) / P(x)`;
    ///   * `B max(W)`, Lucas's stated condition. It is SUFFICIENT on every instance measured and
    ///     ATTAINED on the uniform complete graph, so it cannot be weakened in general -- but it is
    ///     not the exact threshold: one instance here needs only 4.5 where `max(W)` is 6. The first
    ///     version of this test asserted equality on all three worked instances and FAILED on that
    ///     one, which is why the claim now reads as an inequality plus the case that attains it;
    ///   * [`Tsp::threshold`], which is `B` times an upper bound on the optimal tour and is what
    ///     this module can PROVE without solving the instance. It must never sit below the measured
    ///     value, or the default penalty would promise something it does not deliver.
    ///
    /// The last part is the asymmetry: at `1.0001 x` the critical penalty the ground state is the
    /// optimal tour, and at `0.9999 x` it is not a tour at all.
    /// The exact critical penalty of a four-city instance, by enumerating all `2^16` states.
    fn tsp_critical(n: usize, w: &[f64]) -> f64 {
        critical_penalty(
            n * n,
            |x| {
                let mut p = 0.0;
                for v in 0..n {
                    let r: i32 = (0..n).map(|j| i32::from(x[v * n + j])).sum();
                    p += f64::from((1 - r) * (1 - r));
                }
                for j in 0..n {
                    let c: i32 = (0..n).map(|v| i32::from(x[v * n + j])).sum();
                    p += f64::from((1 - c) * (1 - c));
                }
                p
            },
            |x| {
                let mut c = 0.0;
                for j in 0..n {
                    let next = (j + 1) % n;
                    for u in 0..n {
                        for v in 0..n {
                            if u != v && x[u * n + j] == 1 && x[v * n + next] == 1 {
                                c += w[u * n + v];
                            }
                        }
                    }
                }
                c
            },
        )
    }

    fn tsp_max_weight(n: usize, w: &[f64]) -> f64 {
        (0..n)
            .flat_map(|u| (0..n).map(move |v| (u, v)))
            .filter(|&(u, v)| u != v)
            .map(|(u, v)| w[u * n + v])
            .fold(0.0f64, f64::max)
    }

    #[test]
    fn tsp_penalty_threshold_measured_by_enumeration_is_the_largest_edge_weight() {
        let n = 4;
        let uniform: Vec<f64> = vec![
            0.0, 1.0, 1.0, 1.0, //
            1.0, 0.0, 1.0, 1.0, //
            1.0, 1.0, 0.0, 1.0, //
            1.0, 1.0, 1.0, 0.0,
        ];
        // Lucas's bound is ATTAINED: on the uniform complete graph the exact critical penalty is
        // max(W) on the nose, so `A > B max(W)` cannot be weakened as a general statement.
        assert_eq!(tsp_critical(n, &uniform), tsp_max_weight(n, &uniform));

        // And it is SUFFICIENT, measured over random integer instances rather than assumed. The
        // worst ratio seen is reported so a future change that pushes it toward one is visible.
        let mut rng = Pcg::new(0xB0A7, 0x5A1E);
        let mut worst = 0.0f64;
        for _ in 0..24 {
            let mut w = vec![0.0f64; n * n];
            for u in 0..n {
                for v in (u + 1)..n {
                    let e = f64::from(1 + (rng.next_u32() % 9));
                    w[u * n + v] = e;
                    w[v * n + u] = e;
                }
            }
            let (crit, w_max) = (tsp_critical(n, &w), tsp_max_weight(n, &w));
            assert!(
                crit <= w_max,
                "Lucas's A > B max(W) is NOT sufficient on {w:?}: critical {crit} exceeds {w_max}"
            );
            worst = worst.max(crit / w_max);

            // SOUNDNESS of what this module promises: the reported threshold must never sit below
            // the measured critical value. That is the direction that hurts a caller.
            let default = Tsp::new(n, &w).expect("a well-formed instance");
            assert!(
                default.threshold() >= crit,
                "the promised threshold {} is BELOW the measured critical penalty {crit}",
                default.threshold()
            );
            assert!(default.guarantees_feasible());
        }
        assert_eq!(worst, 1.0, "the uniform-shaped instances attain Lucas's bound");

        // Lucas's bound is SLACK on a graph with a spread of weights, and by how much is the number
        // the module doc quotes -- pinned here so the doc cannot drift away from the measurement.
        let spread: Vec<f64> = vec![
            0.0, 2.0, 5.0, 1.0, //
            2.0, 0.0, 3.0, 4.0, //
            5.0, 3.0, 0.0, 6.0, //
            1.0, 4.0, 6.0, 0.0,
        ];
        assert_eq!(tsp_critical(n, &spread), 4.5);
        assert_eq!(tsp_max_weight(n, &spread), 6.0, "Lucas asks for 6 where 4.5 would do");
        // AND THAT SLACK IS OBSERVABLE, which is what makes it a fact rather than an arithmetic
        // remark: dropping the penalty just below Lucas's STATED threshold does not break the
        // model at all. A tightness check written against the stated threshold instead of the
        // measured one would assert an infeasible ground state here and be wrong.
        let below_lucas = Tsp::with_weights(n, &spread, 1.0, 6.0 * 0.9999).expect("positive");
        let (e, states) = all_ground_states(below_lucas.graph());
        let best = brute_tsp(n, &spread).expect("a complete graph has tours");
        assert!((e + below_lucas.offset() - best).abs() < 1e-9);
        for s in &states {
            assert_eq!(below_lucas.tour_cost(&below_lucas.decode(s).expect("a tour")), Some(best));
        }

        // The measured critical penalty is a threshold, not a guideline: just above it the ground
        // state is the optimal tour, just below it is not a tour at all.
        for w in [
            uniform,
            spread,
            vec![
                0.0, 1.0, 7.0, 3.0, //
                1.0, 0.0, 2.0, 9.0, //
                7.0, 2.0, 0.0, 1.0, //
                3.0, 9.0, 1.0, 0.0,
            ],
        ] {
            let crit = tsp_critical(n, &w);
            assert!(crit <= tsp_max_weight(n, &w), "Lucas's bound must stay sufficient");
            let want = brute_tsp(n, &w).expect("a complete graph has tours");

            let tight = Tsp::with_weights(n, &w, 1.0, crit * 1.0001).expect("a positive penalty");
            let (e, states) = all_ground_states(tight.graph());
            assert!(
                (e + tight.offset() - want).abs() < 1e-9,
                "just above the critical penalty the answer is the optimal tour, not {}",
                e + tight.offset()
            );
            for s in &states {
                assert_eq!(tight.tour_cost(&tight.decode(s).expect("a tour")), Some(want));
            }

            let loose = Tsp::with_weights(n, &w, 1.0, crit * 0.9999).expect("a positive penalty");
            let (_, states) = all_ground_states(loose.graph());
            for s in &states {
                assert!(
                    loose.decode(s).is_err(),
                    "below the critical penalty the ground state must not decode to a tour"
                );
            }
        }
    }

    // -----------------------------------------------------------------------------------------
    // decoders and constructors refuse
    // -----------------------------------------------------------------------------------------

    /// Every decoder refuses a state that does not encode a solution, with the error that names it.
    #[test]
    fn decoders_refuse_states_that_are_not_solutions() {
        let path = [(0usize, 1usize), (1, 2), (2, 3)];

        let mc = MaxClique::new(4, &path).unwrap();
        assert_eq!(mc.decode(&[1, 1, 1]), Err(DecodeError::WrongLength { got: 3, want: 4 }));
        assert_eq!(mc.decode(&[1, 0, -1, -1]), Err(DecodeError::NotASpin { index: 1, got: 0 }));
        assert_eq!(mc.decode(&[1, -1, 1, -1]), Err(DecodeError::NotAClique { u: 0, v: 2 }));
        assert_eq!(mc.decode(&[1, 1, -1, -1]).unwrap(), vec![0, 1]);

        let vc = VertexCover::new(4, &path).unwrap();
        assert_eq!(vc.decode(&[-1, -1, -1, -1]), Err(DecodeError::EdgeUncovered { u: 0, v: 1 }));
        assert_eq!(vc.decode(&[-1, 1, 1, -1]).unwrap(), vec![1, 2]);

        let sc = SetCover::new(2, &[vec![0], vec![1]]).unwrap();
        assert_eq!(sc.max_count(), 1);
        // subsets 0 and 1, then one coverage ancilla per element.
        assert_eq!(
            sc.decode(&[1, 1, -1, 1]),
            Err(DecodeError::NotOneHot { group: Group::Coverage(0), set: 0 })
        );
        assert_eq!(sc.decode(&[-1, 1, 1, 1]), Err(DecodeError::ElementUncovered { element: 0 }));
        assert_eq!(sc.decode(&[1, 1, 1, 1]).unwrap(), vec![0, 1]);

        let two = SetCover::new(1, &[vec![0], vec![0]]).unwrap();
        assert_eq!(two.max_count(), 2);
        // Both subsets chosen, ancilla saying "covered once": a cover with a lying counter.
        assert_eq!(
            two.decode(&[1, 1, 1, -1]),
            Err(DecodeError::CountMismatch { element: 0, encoded: 1, covering: 2 })
        );

        let col = Colouring::new(2, &[(0, 1)], 2).unwrap();
        assert_eq!(
            col.decode(&[1, 1, 1, -1]),
            Err(DecodeError::NotOneHot { group: Group::VertexColour(0), set: 2 })
        );
        assert_eq!(
            col.decode(&[1, -1, 1, -1]),
            Err(DecodeError::ColourConflict { u: 0, v: 1, colour: 0 })
        );
        assert_eq!(col.decode(&[1, -1, -1, 1]).unwrap(), vec![0, 1]);

        // A 4-cycle has a Hamiltonian cycle; 0-2-1-3 is a permutation that is not one.
        let hc = HamiltonianCycle::new(4, &[(0, 1), (1, 2), (2, 3), (3, 0)]).unwrap();
        let mut s = vec![-1i8; 16];
        for (j, v) in [0usize, 2, 1, 3].iter().enumerate() {
            s[v * 4 + j] = 1;
        }
        assert_eq!(hc.decode(&s), Err(DecodeError::StepNotAnEdge { from: 0, to: 2, step: 0 }));
        let mut good = vec![-1i8; 16];
        for (j, v) in [0usize, 1, 2, 3].iter().enumerate() {
            good[v * 4 + j] = 1;
        }
        assert_eq!(hc.decode(&good).unwrap(), vec![0, 1, 2, 3]);
        // Two cities at position 0 and none at position 2.
        let mut clash = good.clone();
        clash[2 * 4 + 2] = -1;
        clash[2 * 4] = 1;
        assert_eq!(
            hc.decode(&clash),
            Err(DecodeError::NotOneHot { group: Group::Position(0), set: 2 }),
            "city 2 is still one-hot -- it moved -- so the group that broke is the POSITION, and \
             a decoder checking only the city rows would have accepted this state"
        );

        let inf = f64::INFINITY;
        let w = vec![
            0.0, 1.0, inf, 1.0, //
            1.0, 0.0, 1.0, 8.0, //
            inf, 1.0, 0.0, 1.0, //
            1.0, 8.0, 1.0, 0.0,
        ];
        let tsp = Tsp::new(4, &w).unwrap();
        assert_eq!(tsp.decode(&s), Err(DecodeError::StepNotAnEdge { from: 0, to: 2, step: 0 }));
        assert_eq!(tsp.decode(&good).unwrap(), vec![0, 1, 2, 3]);
        assert_eq!(tsp.tour_cost(&[0, 1, 2, 3]), Some(4.0));
        assert_eq!(tsp.tour_cost(&[0, 2, 1, 3]), None, "a tour across an absent edge has no cost");
    }

    /// Every way an instance can be malformed is an error naming what was seen, not a default.
    ///
    /// Read through `.err()` because a model owns a [`Graph`], which is deliberately neither
    /// `Debug` nor `PartialEq` in this crate -- comparing two CSR graphs for equality is a question
    /// with several different right answers and no caller has needed one.
    #[test]
    fn malformed_instances_are_refused() {
        assert_eq!(
            MaxClique::new(3, &[(0, 5)]).err(),
            Some(InstanceError::VertexOutOfRange { vertex: 5, n: 3 })
        );
        assert_eq!(
            MaxClique::new(3, &[(1, 1)]).err(),
            Some(InstanceError::SelfLoop { vertex: 1 })
        );
        assert_eq!(
            MaxClique::with_weights(3, &[], 0.0, 1.0).err(),
            Some(InstanceError::BadWeight { what: "clique reward", got: 0.0 })
        );
        assert!(MaxClique::with_weights(3, &[], 1.0, f64::NAN).is_err());

        assert_eq!(
            VertexCover::new(2, &[(0, 2)]).err(),
            Some(InstanceError::VertexOutOfRange { vertex: 2, n: 2 })
        );

        assert_eq!(
            SetCover::new(0, &[vec![0]]).err(),
            Some(InstanceError::Empty { what: "universe" })
        );
        assert_eq!(
            SetCover::new(2, &[]).err(),
            Some(InstanceError::Empty { what: "collection of subsets" })
        );
        assert_eq!(
            SetCover::new(2, &[vec![0], vec![]]).err(),
            Some(InstanceError::EmptySubset { subset: 1 })
        );
        assert_eq!(
            SetCover::new(2, &[vec![0, 4]]).err(),
            Some(InstanceError::ElementOutOfRange { subset: 0, element: 4, universe: 2 })
        );
        assert_eq!(
            SetCover::new(3, &[vec![0], vec![1]]).err(),
            Some(InstanceError::Uncoverable { element: 2 }),
            "an instance with no cover at all is refused, not solved to an empty answer"
        );
        // A repeated element inside one subset is a set, not a multiset: it must not double-count.
        let dup = SetCover::new(2, &[vec![0, 0, 1]]).expect("a set, written twice");
        assert_eq!(dup.max_count(), 1);

        assert_eq!(Colouring::new(2, &[(0, 1)], 0).err(), Some(InstanceError::NoColours));

        assert_eq!(
            HamiltonianCycle::new(2, &[(0, 1)]).err(),
            Some(InstanceError::TooSmall { n: 2, min: 3 })
        );

        assert_eq!(
            Tsp::new(2, &[0.0, 1.0, 1.0, 0.0]).err(),
            Some(InstanceError::TooSmall { n: 2, min: 3 })
        );
        assert_eq!(Tsp::new(3, &[0.0]).err(), Some(InstanceError::Matrix { got: 1, want: 9 }));
        assert_eq!(
            Tsp::new(3, &[0.0, 1.0, 1.0, 2.0, 0.0, 1.0, 1.0, 1.0, 0.0]).err(),
            Some(InstanceError::Asymmetric { u: 0, v: 1, uv: 1.0, vu: 2.0 })
        );
        assert_eq!(
            Tsp::new(3, &[0.0, -1.0, 1.0, -1.0, 0.0, 1.0, 1.0, 1.0, 0.0]).err(),
            Some(InstanceError::EdgeWeight { u: 0, v: 1, got: -1.0 }),
            "a negative edge lets an infeasible state buy its own penalty back"
        );
        assert!(
            matches!(
                Tsp::new(3, &[0.0, f64::NAN, 1.0, f64::NAN, 0.0, 1.0, 1.0, 1.0, 0.0]).err(),
                Some(InstanceError::EdgeWeight { .. })
            ),
            "NaN equals nothing, itself included, so it must be caught by magnitude"
        );

        // Display is not empty and names the thing.
        let e = SetCover::new(3, &[vec![0], vec![1]]).err().expect("no cover exists");
        assert!(e.to_string().contains("element 2"), "{e}");
        let d = DecodeError::NotOneHot { group: Group::City(2), set: 3 };
        assert!(d.to_string().contains("city 2"), "{d}");
    }
}
