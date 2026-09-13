//! QPLIB, Biq Mac, OR-Library and `MQLib`: four benchmark corpora, three grammars, and four
//! different things meant by the same nine characters.
//!
//! [`crate::gset`] reads G-set and [`crate::dimacs`] reads the MAX-SAT competition files. Those are
//! two of the six corpora a max-cut or QUBO result is normally reported against; the other four had
//! no reader in this crate at all, so an instance from any of them had to be converted by hand
//! before it could be sampled — and a hand conversion is exactly where a sign, a factor of two or a
//! dropped constant goes in unnoticed.
//!
//! * **QPLIB** — Furini, Traversi, Belotti, Frangioni, Gleixner, Gould, Liberti, Lodi, Misener,
//!   Mittelmann, Sahinidis, Vigerske and Wiegele, *QPLIB: a library of quadratic programming
//!   instances*, Mathematical Programming Computation 11(2):237–265, 2019.
//! * **Biq Mac** — Rendl, Rinaldi and Wiegele, *Solving max-cut to optimality by intersecting
//!   semidefinite and polyhedral relaxations*, Mathematical Programming 121(2):307–335, 2010. Its
//!   files are in the edge-list format Rinaldi's `rudy` generator emits, which is also what
//!   produced G-set.
//! * **OR-Library** — Beasley, *OR-Library: distributing test problems by electronic mail*, Journal
//!   of the Operational Research Society 41(11):1069–1072, 1990; the `bqp` set is from Beasley,
//!   *Heuristic algorithms for the unconstrained binary quadratic programming problem*, 1998.
//! * **`MQLib`** — Dunning, Gupta and Silberholz, *What works best when? A systematic evaluation of
//!   heuristics for max-cut and QUBO*, INFORMS Journal on Computing 30(3):608–624, 2018.
//!
//! # One affine map, stated once
//!
//! Every instance here lands as an Ising [`Graph`] built so that **minimising its energy optimises
//! the corpus's own objective**, plus the two numbers that turn an energy back into the number the
//! literature reports:
//!
//! ```text
//!     objective(s) = scale · E(s) + offset
//! ```
//!
//! `scale` is negative exactly when the objective is a MAXIMISATION, and it is always ±1 or ±½, so
//! multiplying by it is exact in binary. [`Instance::objective`] is the only place that arithmetic
//! lives, and [`Instance::objective_bound`] is its bound-direction twin: a lower bound on the
//! energy becomes an UPPER bound on a maximisation and a LOWER bound on a minimisation, which is
//! the whole reason to carry the sense rather than assume it.
//!
//! **The sense is not decoration.** `MQLib` and OR-Library maximise, QPLIB says which per file.
//! Reading a maximisation as a minimisation returns the WORST assignment while every intermediate
//! number stays plausible — the failure [`crate::gset`]'s header is written around, one layer up.
//!
//! **Nor is the offset.** It is the same for every state, so dropping it leaves every optimiser's
//! answer untouched and every reported value wrong by a fixed amount: [`crate::dimod`] records that
//! a solver test passes and a comparison against a published objective fails. Here the constant is
//! carried, and carried with a rigorous bracket ([`Instance::offset_err`]) because it is an
//! accumulated sum and a bound that rides on it must not be narrowed by the accumulation's own
//! rounding.
//!
//! # The QUBO these formats share, written once
//!
//! [`Qubo`] holds `f(x) = Σ_{i ≤ j} c_ij x_i x_j + constant` over `x ∈ {0,1}`, each stored pair
//! counted **once**, the diagonal `c_ii` being the linear coefficient because `x² = x`. Each reader
//! says how its file's entries map into `c`:
//!
//! | corpus | file says | `c_ii` | `c_ij`, `i < j` |
//! |---|---|---|---|
//! | OR-Library `bqp` | upper triangle of `Q`, `i ≤ j` | `q_ii` | `q_ij` |
//! | `MQLib` QUBO | upper triangle of `Q`, `i ≤ j` | `q_ii` | `q_ij` |
//! | QPLIB | `½ xᵀQ⁰x + b⁰ᵀx + q⁰`, lower triangle of `Q⁰` | `½ Q_ii + b_i` | `Q_ij` |
//!
//! The QPLIB row is where the factor of two lives: `½ xᵀQx` with a SYMMETRIC `Q` gives each
//! off-diagonal pair `½ · 2 · Q_ij = Q_ij`, counted once, while the diagonal keeps its half. Read
//! the lower triangle as if it were an upper-triangular QUBO and every off-diagonal is right and
//! every diagonal is doubled — which changes the optimum and not the shape, so nothing looks wrong.
//!
//! The substitution `x = (1+s)/2` then gives the Ising form, and the constant it produces is the
//! offset above:
//!
//! ```text
//!     h_i    = ½ c_ii + ¼ Σ_{j≠i} c_ij          w_ij = ¼ c_ij
//!     offset = Σ_i ½ c_ii + Σ_{i<j} ¼ c_ij + constant
//! ```
//!
//! with every `h` and `w` NEGATED when the sense is a minimisation, since this crate's energy is
//! `E = −Σ h s − Σ w s s` and a minimisation wants `E = f − offset` where a maximisation wants
//! `E = offset − f`.
//!
//! # Max-cut, and the negation [`crate::gset`] exists to get right
//!
//! `cut(s) = ½ (W − Σ w_ij s_i s_j)`, so loading `J_ij = −w_ij` makes `E = +Σ w s s` and
//! `cut = −½ E + W/2`: `scale = −½`, `offset = W/2`. Load `J = +w` instead and minimising the
//! energy MINIMISES the cut.
//!
//! # Three grammars, and what actually separates them
//!
//! Three of these four write `n m` and then `i j v`, one-based, and mean three different things:
//!
//! | | header | body | sense | per file |
//! |---|---|---|---|---|
//! | Biq Mac / `rudy` | `n m` | `i j w`, an EDGE, `i ≠ j` | maximise the cut | one instance |
//! | `MQLib` max-cut | `n m` | `i j w`, an EDGE, `i ≠ j` | maximise the cut | one instance |
//! | `MQLib` QUBO | `n nnz` | `i j q`, a COEFFICIENT, diagonal allowed | maximise `f` | one instance |
//! | OR-Library `bqp` | count, then `n nnz` per problem | `i j q`, a COEFFICIENT | maximise `f` | MANY |
//! | QPLIB | name, type, sense, `n`, sections | see above | the file says | one instance |
//!
//! So the grammar is not the thing to get right; the meaning is. `MQLib` additionally tolerates
//! `#` comment lines, which Biq Mac's own files do not carry.
//!
//! # What this refuses
//!
//! A self-loop in a max-cut file (an edge inside one vertex crosses no cut and is not a constant
//! anybody declared), a pair named twice in any file, a one-based index outside `1..=n`, a body
//! whose length disagrees with its declared count, a weight that is not a finite number, and — for
//! QPLIB — any type code outside the binary unconstrained subset, **by code**, before a single
//! coefficient is read.
//!
//! Writing refuses too: [`write_mqlib_qubo`] and [`write_orlib`] have nowhere to put a constant, so
//! a [`Qubo`] carrying one is refused by name rather than written out one term short.
//!
//! # Provenance of the conventions, since a convention is the whole content here
//!
//! Each reader is written against its format's published description, not against one example file,
//! and two of those descriptions carry a reading this review could not re-verify against the
//! distributed files. They are named here rather than buried, because in both cases a wrong reading
//! produces a file that parses, samples and reports a plausible number:
//!
//! * **The off-diagonal is counted ONCE** in OR-Library and `MQLib`. A producer who meant
//!   `xᵀQx` with a symmetric `Q` and wrote only its upper triangle means every off-diagonal
//!   DOUBLED. The two readings are different problems with the same shape.
//! * **`MQLib` normalises its QUBO to MAXIMISATION**, as it does its max-cut. Read as a
//!   minimisation, its answer is the worst assignment.
//!
//! Both are stated on [`read_orlib`] and [`read_mqlib_qubo`] and are one line each to change; what
//! this module guarantees is that the convention it applies is WRITTEN DOWN and checked by
//! `oracle_dimod_bqm_scores_every_qubo_format_the_same_at_every_state`, not that a corpus's
//! maintainers agree with it.
//!
//! # What is NOT implemented, deliberately
//!
//! QPLIB's full format covers continuous and general-integer variables, linear and quadratic
//! constraints, bounds, starting points and dual starting points. None of that is an Ising model,
//! so only the **binary unconstrained** subset — type code `?B N`, i.e. any objective over binary
//! variables with no constraints — is read, and every other code is refused naming itself. That
//! refusal is what makes it safe to stop reading after the objective constant: the sections QPLIB
//! writes after it for such an instance are a starting point and the variable names, neither of
//! which changes what is being solved.
//!
//! This review did not locate a separate max-cut set in OR-Library. The instances the max-cut
//! literature calls "Beasley" — `be100.1` and its family — are the OR-Library `bqp` instances
//! carried into max-cut by the root-node transformation (Hammer, *Some network flow problems solved
//! with pseudo-Boolean programming*, Operations Research 13:388–399, 1965), and distributed that
//! way by Biq Mac. [`Qubo::to_maxcut`] is that transformation, and
//! `oracle_the_root_node_transformation_makes_the_cut_the_qubo_objective` checks it state by state
//! rather than at the optimum.

use crate::graph::{Graph, GraphBuilder};

// ---- what a corpus's numbers mean ---------------------------------------------------------------

/// Which direction a corpus optimises in.
///
/// Carried rather than assumed: three of the four corpora here maximise, QPLIB says per file, and a
/// reader that guesses returns the worst assignment with every intermediate number looking right.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sense {
    /// Smaller is better.
    Minimise,
    /// Larger is better. [`Instance::scale`] is negative, so minimising the energy still optimises.
    Maximise,
}

impl Sense {
    /// The word QPLIB writes on its third line for this sense.
    #[must_use]
    pub const fn qplib_word(self) -> &'static str {
        match self {
            Sense::Minimise => "Minimize",
            Sense::Maximise => "Maximize",
        }
    }
}

impl core::fmt::Display for Sense {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(match self {
            Sense::Minimise => "minimise",
            Sense::Maximise => "maximise",
        })
    }
}

/// Which library a file came from, which is what decides how to read it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Corpus {
    /// QPLIB, the quadratic programming instance library.
    Qplib,
    /// The Biq Mac library, in the `rudy` edge-list format.
    BiqMac,
    /// Beasley's OR-Library, `bqp` set.
    OrLibrary,
    /// The `MQLib` instance library.
    MqLib,
}

impl Corpus {
    /// The library's own name, as its papers print it.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Corpus::Qplib => "QPLIB",
            Corpus::BiqMac => "Biq Mac",
            Corpus::OrLibrary => "OR-Library",
            Corpus::MqLib => "MQLib",
        }
    }
}

impl core::fmt::Display for Corpus {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.name())
    }
}

/// Which of the two problems a file states.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Problem {
    /// Maximise the weight of the edges crossing a bipartition.
    MaxCut,
    /// Optimise a quadratic form over `{0,1}` variables.
    Qubo,
}

// ---- errors -------------------------------------------------------------------------------------

/// Why a corpus file could not be read, or a model could not be written.
///
/// Every variant names the format it applies to, because three of these grammars are
/// character-for-character the same and a message that did not say which one was being read would
/// be no help at all.
#[derive(Clone, Debug, PartialEq)]
pub enum CorporaError {
    /// A header line missing, or not of the shape the format declares.
    Header {
        /// Which format was being read.
        format: &'static str,
        /// One-based line number, or `0` when the input ended before any line.
        line: usize,
        /// The line as read.
        text: String,
    },
    /// A body line that is not what the format requires there.
    Line {
        /// Which format was being read.
        format: &'static str,
        /// One-based line number.
        line: usize,
        /// The line as read.
        text: String,
    },
    /// A token where a finite number belongs: unparseable, or `nan`/`inf`, which parse as `f64`
    /// and poison every energy they reach.
    Number {
        /// Which format was being read.
        format: &'static str,
        /// One-based line number.
        line: usize,
        /// The token found.
        got: String,
    },
    /// An index outside `1..=n`.
    ///
    /// Named rather than clamped or shifted: every format here is ONE-based, and a `0` means the
    /// producer was zero-based, which parses perfectly and moves every coefficient by a variable.
    Index {
        /// Which format was being read.
        format: &'static str,
        /// One-based line number.
        line: usize,
        /// The index found.
        got: i64,
        /// Variables or vertices the header declared, so the valid range is `1..=n`.
        n: usize,
    },
    /// An edge from a vertex to itself, in a format whose entries are edges.
    ///
    /// A self-loop crosses no cut, so it is a constant — and it is a constant nobody declared,
    /// which is why it is refused rather than folded into the offset.
    SelfLoop {
        /// Which format was being read.
        format: &'static str,
        /// One-based line number.
        line: usize,
        /// The vertex, one-based as the file wrote it.
        at: usize,
    },
    /// The same unordered pair given twice.
    ///
    /// Summing them is what [`crate::graph::GraphBuilder`] would do, and it is the wrong answer
    /// here: a corpus file listing a pair twice is damaged, and the instance that results has a
    /// different optimum from the one everybody else solved.
    Duplicate {
        /// Which format was being read.
        format: &'static str,
        /// One-based line number of the repeat.
        line: usize,
        /// One-based line number of the first appearance.
        first: usize,
        /// The pair, one-based as the file wrote it.
        pair: (usize, usize),
    },
    /// A declared count and the body disagree.
    ///
    /// A truncated download parses into a valid SMALLER instance whose optimum is not comparable
    /// with anyone else's, which is why this is refused rather than solved.
    Count {
        /// Which format was being read.
        format: &'static str,
        /// What is being counted — `"edges"`, `"coefficients"`, `"problems"`.
        what: &'static str,
        /// What the header said.
        declared: usize,
        /// What the body held.
        found: usize,
    },
    /// The file ends where a section was expected.
    Truncated {
        /// Which format was being read.
        format: &'static str,
        /// The section that was missing.
        want: &'static str,
    },
    /// A QPLIB objective-sense line that is neither `Minimize` nor `Maximize`.
    Sense {
        /// One-based line number.
        line: usize,
        /// The line as read.
        got: String,
    },
    /// A QPLIB problem type this reader does not implement.
    Unsupported {
        /// The three-character type code, as written.
        code: String,
        /// Which character of it is the problem, and why.
        why: &'static str,
    },
    /// The model carries something the target format has nowhere to put.
    Unrepresentable {
        /// Which format was being written.
        format: &'static str,
        /// What could not be written.
        what: String,
    },
}

impl core::fmt::Display for CorporaError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            CorporaError::Header { format, line, text } => {
                write!(f, "{format}: line {line} is not the header this format declares: {text:?}")
            }
            CorporaError::Line { format, line, text } => {
                write!(f, "{format}: line {line} is not a body line of this format: {text:?}")
            }
            CorporaError::Number { format, line, got } => write!(
                f,
                "{format}: line {line} carries {got:?} where a finite number belongs (`nan` and \
                 `inf` parse as f64 and poison every energy they reach)"
            ),
            CorporaError::Index { format, line, got, n } => write!(
                f,
                "{format}: line {line} names index {got}, outside 1..={n}; this format is ONE-based \
                 and a 0 means the producer was zero-based"
            ),
            CorporaError::SelfLoop { format, line, at } => write!(
                f,
                "{format}: line {line} joins vertex {at} to itself. A self-loop crosses no cut, so \
                 it is a constant nobody declared"
            ),
            CorporaError::Duplicate { format, line, first, pair } => write!(
                f,
                "{format}: line {line} repeats the pair ({}, {}) first given on line {first}. \
                 Summing them would silently solve a different instance",
                pair.0, pair.1
            ),
            CorporaError::Count { format, what, declared, found } => write!(
                f,
                "{format}: the header declares {declared} {what} and the body has {found}. A \
                 truncated file parses into a valid SMALLER instance whose optimum is not \
                 comparable with anyone else's, so this is refused rather than solved"
            ),
            CorporaError::Truncated { format, want } => {
                write!(f, "{format}: the file ends where {want} was expected")
            }
            CorporaError::Sense { line, got } => write!(
                f,
                "QPLIB: line {line} is {got:?} where `Minimize` or `Maximize` belongs. Reading a \
                 maximisation as a minimisation returns the WORST assignment"
            ),
            CorporaError::Unsupported { code, why } => {
                write!(f, "QPLIB: the type code {code:?} is not read here: {why}")
            }
            CorporaError::Unrepresentable { format, what } => write!(
                f,
                "{format} has nowhere to put {what}, so this is refused rather than written out \
                 one term short"
            ),
        }
    }
}

impl core::error::Error for CorporaError {}

// ---- the two shapes a corpus file takes ---------------------------------------------------------

/// A quadratic form over `{0,1}` variables: `f(x) = Σ_{i ≤ j} c_ij x_i x_j + constant`.
///
/// Each stored pair is counted **once**; the diagonal `c_ii` is the linear coefficient, because
/// `x² = x` for a binary variable. Terms are canonical — `i ≤ j`, sorted by `(i, j)`, no pair
/// twice — which is what makes a write/read round trip bit-for-bit identical rather than merely
/// equivalent: the order the offset accumulates in is then a property of the model and not of the
/// file's line order.
#[derive(Clone, Debug, PartialEq)]
pub struct Qubo {
    /// The instance's name. For a corpus whose files carry none, the position in the file.
    pub name: String,
    /// Variable count. Variable `i` is written `i + 1` in every format here.
    pub n: usize,
    /// `(i, j, c_ij)` with `i ≤ j`, sorted, unique.
    pub terms: Vec<(usize, usize, f64)>,
    /// The additive constant. QPLIB's `q⁰`; zero for the formats with nowhere to write one.
    pub constant: f64,
    /// Which direction the corpus optimises `f` in.
    pub sense: Sense,
    /// Which library this came from.
    pub corpus: Corpus,
}

/// A weighted graph whose max-cut is the problem.
///
/// Edges are canonical — `i < j`, sorted, no pair twice, no self-loop — for the same reason
/// [`Qubo`]'s terms are.
#[derive(Clone, Debug, PartialEq)]
pub struct MaxCut {
    /// The instance's name.
    pub name: String,
    /// Vertex count. Vertex `i` is written `i + 1`.
    pub n: usize,
    /// `(i, j, w_ij)` with `i < j`, sorted, unique.
    pub edges: Vec<(usize, usize, f64)>,
    /// Which library this came from.
    pub corpus: Corpus,
}

/// A corpus instance as an Ising model, plus the affine map back to the corpus's own objective.
///
/// `objective(s) = scale · E(s) + offset`, and the graph is built so **minimising `E` optimises the
/// objective** whichever way the corpus points. See the module header.
pub struct Instance {
    /// The instance's name, as the file gave it.
    pub name: String,
    /// Which library it came from.
    pub corpus: Corpus,
    /// Which problem it states.
    pub problem: Problem,
    /// Which direction the objective is optimised in.
    pub sense: Sense,
    /// The Ising graph, signs already arranged so that minimising the energy optimises.
    pub graph: Graph,
    /// The energy-to-objective slope. Negative exactly for a [`Sense::Maximise`] instance, and
    /// always `±1` or `±½` — a power of two, so multiplying by it is exact.
    pub scale: f64,
    /// The energy-to-objective intercept: the constant the Ising form cannot hold.
    pub offset: f64,
    /// A rigorous half-width on [`Instance::offset`]'s own rounding error.
    ///
    /// The offset is an accumulated sum, so it is not exact. `offset ± offset_err` brackets the
    /// true constant, and [`Instance::objective_bound`] spends that width in the safe direction —
    /// a bound that took the offset as exact would be narrower than the arithmetic supports, which
    /// is the defect [`crate::round`] exists for.
    pub offset_err: f64,
    /// Variables or vertices.
    pub vars: usize,
    /// Terms or edges the file carried.
    pub terms: usize,
}

/// Summary only, deliberately: a corpus instance has tens of thousands of edges, and deriving
/// `Debug` would bury a failing `unwrap`'s message under the whole CSR.
impl core::fmt::Debug for Instance {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "Instance {{ {} {:?} {:?} {}, n: {}, terms: {}, objective = {} * E + {} }}",
            self.corpus,
            self.problem,
            self.sense,
            self.name,
            self.vars,
            self.terms,
            self.scale,
            self.offset
        )
    }
}

impl Instance {
    /// The number the corpus's own literature reports for this state.
    ///
    /// # Panics
    ///
    /// If `s` is shorter than [`Instance::vars`], which is [`Graph::energy`]'s requirement.
    #[must_use]
    pub fn objective(&self, s: &[i8]) -> f64 {
        self.scale * self.graph.energy(s) + self.offset
    }

    /// A bound on the optimal objective, from a lower bound on the energy — an UPPER bound for a
    /// maximisation and a LOWER bound for a minimisation.
    ///
    /// `L ≤ min E`, and the objective is affine in `E` with a slope whose sign is the sense, so the
    /// bound lands on the optimistic side either way. Every published max-cut number is a lower
    /// bound on the optimum (somebody found that cut); this is the other side, which is what turns
    /// "best known" into a gap.
    ///
    /// Accumulated through [`crate::round`] and charged [`Instance::offset_err`], so the answer is
    /// on the safe side of the truth rather than near it. `scale` is a power of two, so
    /// `scale · L` is exact and only the final addition needs guarding.
    #[must_use]
    pub fn objective_bound(&self, energy_lower_bound: f64) -> f64 {
        let l = self.scale * energy_lower_bound;
        match self.sense {
            Sense::Maximise => crate::round::sum_up(&[l, self.offset, self.offset_err]),
            Sense::Minimise => crate::round::sum_down(&[l, self.offset, -self.offset_err]),
        }
    }
}

/// A value for an accumulated constant, and a rigorous half-width around it.
///
/// [`crate::round::sum_down`] and [`crate::round::sum_up`] bracket the exact sum; the midpoint is
/// the value to report and half the width is what a bound must spend. `next_up` covers the
/// rounding of the halving itself.
///
/// **The width is not zero on an exact sum, and measuring that is why this note exists.**
/// `round`'s guard is `2 ε |total| + n² ε² Σ|x|`, a bound on what the arithmetic COULD have lost
/// rather than a detection of what it did: six integer half-weights summing to exactly `6.5` come
/// back with a half-width of `2.66e-15`. The bracket is sound either way — it contains the exact
/// value — and the only input it collapses on is the empty sum, which is the one case with no
/// arithmetic at all. A bound spending this width is therefore conservative by about two ULPs,
/// which is the direction a bound is allowed to be wrong in.
fn bracket(terms: &[f64]) -> (f64, f64) {
    let lo = crate::round::sum_down(terms);
    let hi = crate::round::sum_up(terms);
    if lo == hi {
        return (lo, 0.0);
    }
    let mid = lo + (hi - lo) * 0.5;
    (mid, (hi - mid).max(mid - lo).next_up())
}

impl Qubo {
    /// A checked [`Qubo`]: indices in range, pairs canonicalised to `i ≤ j`, sorted, and no pair
    /// given twice.
    ///
    /// # Errors
    ///
    /// [`CorporaError::Index`] for an index at or past `n`, [`CorporaError::Duplicate`] for a
    /// repeated pair, [`CorporaError::Number`] for a coefficient or constant that is not finite.
    /// Line numbers are zero, since this constructor has no file to point at.
    pub fn new(
        name: impl Into<String>,
        n: usize,
        terms: &[(usize, usize, f64)],
        constant: f64,
        sense: Sense,
        corpus: Corpus,
    ) -> Result<Qubo, CorporaError> {
        let format = "a QUBO";
        let mut t: Vec<(usize, usize, f64)> = Vec::with_capacity(terms.len());
        for &(i, j, c) in terms {
            for v in [i, j] {
                if v >= n {
                    return Err(CorporaError::Index { format, line: 0, got: v as i64 + 1, n });
                }
            }
            if !c.is_finite() {
                return Err(CorporaError::Number { format, line: 0, got: c.to_string() });
            }
            t.push((i.min(j), i.max(j), c));
        }
        t.sort_by_key(|a| (a.0, a.1));
        for w in t.windows(2) {
            if (w[0].0, w[0].1) == (w[1].0, w[1].1) {
                return Err(CorporaError::Duplicate {
                    format,
                    line: 0,
                    first: 0,
                    pair: (w[0].0 + 1, w[0].1 + 1),
                });
            }
        }
        if !constant.is_finite() {
            return Err(CorporaError::Number { format, line: 0, got: constant.to_string() });
        }
        Ok(Qubo { name: name.into(), n, terms: t, constant, sense, corpus })
    }

    /// `f(x)` at a `{0,1}` state, straight from the stored coefficients.
    ///
    /// # Panics
    ///
    /// If `x` is shorter than [`Qubo::n`].
    #[must_use]
    pub fn value(&self, x: &[i8]) -> f64 {
        assert!(x.len() >= self.n, "a state of {} cannot cover {} variables", x.len(), self.n);
        let mut v = self.constant;
        for &(i, j, c) in &self.terms {
            v += c * f64::from(x[i]) * f64::from(x[j]);
        }
        v
    }

    /// The Ising form, with the affine map back to `f`.
    ///
    /// `h_i = ½ c_ii + ¼ Σ_{j≠i} c_ij`, `w_ij = ¼ c_ij`, all negated for a minimisation; the
    /// constant the substitution leaves behind becomes [`Instance::offset`]. See the module header
    /// for the derivation.
    #[must_use]
    pub fn instance(&self) -> Instance {
        let mut a = vec![0.0f64; self.n];
        let mut consts: Vec<f64> = Vec::with_capacity(self.terms.len() + 1);
        let mut quad: Vec<(usize, usize, f64)> = Vec::with_capacity(self.terms.len());
        for &(i, j, c) in &self.terms {
            if i == j {
                a[i] += c * 0.5;
                consts.push(c * 0.5);
            } else {
                let b = c * 0.25;
                a[i] += b;
                a[j] += b;
                quad.push((i, j, b));
                consts.push(b);
            }
        }
        consts.push(self.constant);
        let (offset, offset_err) = bracket(&consts);
        // A maximisation wants E = offset − f, a minimisation E = f − offset, and E = −Σhs − Σwss.
        let sign = match self.sense {
            Sense::Maximise => 1.0,
            Sense::Minimise => -1.0,
        };
        let mut gb = GraphBuilder::new(self.n);
        for (i, &ai) in a.iter().enumerate() {
            gb.bias(i, sign * ai);
        }
        for (i, j, b) in quad {
            gb.couple(i, j, sign * b);
        }
        Instance {
            name: self.name.clone(),
            corpus: self.corpus,
            problem: Problem::Qubo,
            sense: self.sense,
            graph: gb.build(),
            scale: -sign,
            offset,
            offset_err,
            vars: self.n,
            terms: self.terms.len(),
        }
    }

    /// The same problem as a max-cut, by the root-node transformation (Hammer 1965).
    ///
    /// Vertex `0` is the root and vertex `i + 1` is variable `i`; a cut puts `x_i = 1` exactly when
    /// `i + 1` lands on the other side from the root. With
    ///
    /// ```text
    ///     w(0, i) = c_ii + ½ Σ_{j≠i} c_ij            w(i, j) = −½ c_ij
    /// ```
    ///
    /// the cut value **equals** `f(x)` at every state, not merely at the optimum — because
    /// `[y_i ≠ y_j] = x_i + x_j − 2 x_i x_j`, whose quadratic part carries the `−2` that the half
    /// cancels and whose linear part is exactly what the root edge is there to correct.
    ///
    /// This is how the max-cut literature's "Beasley" instances were made: they are OR-Library
    /// `bqp` instances put through this, and Biq Mac distributes the result.
    ///
    /// # Errors
    ///
    /// [`CorporaError::Unrepresentable`] for a minimisation (max-cut is a maximisation; negate the
    /// objective yourself if that is what you mean) or a nonzero constant (a max-cut file has
    /// nowhere to put one).
    pub fn to_maxcut(&self) -> Result<MaxCut, CorporaError> {
        let format = "a max-cut instance";
        if self.sense != Sense::Maximise {
            return Err(CorporaError::Unrepresentable {
                format,
                what: "a minimisation — max-cut maximises, so negate the objective first"
                    .to_string(),
            });
        }
        if self.constant != 0.0 {
            return Err(CorporaError::Unrepresentable {
                format,
                what: format!("a constant of {}", self.constant),
            });
        }
        let mut root = vec![0.0f64; self.n];
        let mut edges: Vec<(usize, usize, f64)> = Vec::with_capacity(self.terms.len() + self.n);
        for &(i, j, c) in &self.terms {
            if i == j {
                root[i] += c;
            } else {
                root[i] += c * 0.5;
                root[j] += c * 0.5;
                edges.push((i + 1, j + 1, -0.5 * c));
            }
        }
        for (i, &w) in root.iter().enumerate() {
            edges.push((0, i + 1, w));
        }
        edges.sort_by_key(|a| (a.0, a.1));
        Ok(MaxCut { name: self.name.clone(), n: self.n + 1, edges, corpus: self.corpus })
    }
}

impl MaxCut {
    /// A checked [`MaxCut`]: indices in range, no self-loop, pairs canonicalised to `i < j`,
    /// sorted, and no pair given twice.
    ///
    /// # Errors
    ///
    /// [`CorporaError::Index`], [`CorporaError::SelfLoop`], [`CorporaError::Duplicate`] and
    /// [`CorporaError::Number`], all with line number zero since this constructor has no file.
    pub fn new(
        name: impl Into<String>,
        n: usize,
        edges: &[(usize, usize, f64)],
        corpus: Corpus,
    ) -> Result<MaxCut, CorporaError> {
        let format = "a max-cut instance";
        let mut e: Vec<(usize, usize, f64)> = Vec::with_capacity(edges.len());
        for &(i, j, w) in edges {
            for v in [i, j] {
                if v >= n {
                    return Err(CorporaError::Index { format, line: 0, got: v as i64 + 1, n });
                }
            }
            if i == j {
                return Err(CorporaError::SelfLoop { format, line: 0, at: i + 1 });
            }
            if !w.is_finite() {
                return Err(CorporaError::Number { format, line: 0, got: w.to_string() });
            }
            e.push((i.min(j), i.max(j), w));
        }
        e.sort_by_key(|a| (a.0, a.1));
        for w in e.windows(2) {
            if (w[0].0, w[0].1) == (w[1].0, w[1].1) {
                return Err(CorporaError::Duplicate {
                    format,
                    line: 0,
                    first: 0,
                    pair: (w[0].0 + 1, w[0].1 + 1),
                });
            }
        }
        Ok(MaxCut { name: name.into(), n, edges: e, corpus })
    }

    /// The weight of the edges this bipartition cuts, straight from the stored edges.
    ///
    /// # Panics
    ///
    /// If `s` is shorter than [`MaxCut::n`].
    #[must_use]
    pub fn cut(&self, s: &[i8]) -> f64 {
        assert!(s.len() >= self.n, "a state of {} cannot cover {} vertices", s.len(), self.n);
        self.edges.iter().filter(|&&(i, j, _)| s[i] != s[j]).map(|&(_, _, w)| w).sum()
    }

    /// The Ising form, with the affine map back to the cut value.
    ///
    /// **The couplings are negated.** `J = +w` gives an energy whose minimum is the MINIMUM cut: a
    /// plausible number, a valid state, and the opposite problem.
    #[must_use]
    pub fn instance(&self) -> Instance {
        let mut halves: Vec<f64> = Vec::with_capacity(self.edges.len());
        let mut gb = GraphBuilder::new(self.n);
        for &(i, j, w) in &self.edges {
            gb.couple(i, j, -w);
            halves.push(w * 0.5);
        }
        let (offset, offset_err) = bracket(&halves);
        Instance {
            name: self.name.clone(),
            corpus: self.corpus,
            problem: Problem::MaxCut,
            sense: Sense::Maximise,
            graph: gb.build(),
            scale: -0.5,
            offset,
            offset_err,
            vars: self.n,
            terms: self.edges.len(),
        }
    }
}

// ---- reading ------------------------------------------------------------------------------------

/// Data lines, with each format's comment convention applied.
///
/// QPLIB comments run from a `!` to the end of the line, which covers a whole-line comment too
/// (the payload is then empty and skipped). `MQLib` comments are whole lines beginning with `#`.
/// Biq Mac and OR-Library files carry neither, and are read with both switched off rather than
/// with a permissive superset: a `#` in a Biq Mac file is not a comment, it is a damaged file.
struct Scanner<'a> {
    inner: core::iter::Enumerate<core::str::Lines<'a>>,
    comment: Option<char>,
    trailing: Option<char>,
}

impl<'a> Scanner<'a> {
    fn new(text: &'a str, comment: Option<char>, trailing: Option<char>) -> Scanner<'a> {
        Scanner { inner: text.lines().enumerate(), comment, trailing }
    }

    /// The next line carrying data, as `(one-based number, payload)`.
    fn next_line(&mut self) -> Option<(usize, &'a str)> {
        for (no, raw) in self.inner.by_ref() {
            let mut l = raw;
            if let Some(c) = self.trailing
                && let Some(p) = l.find(c)
            {
                l = &l[..p];
            }
            let l = l.trim();
            if l.is_empty() || self.comment.is_some_and(|c| l.starts_with(c)) {
                continue;
            }
            return Some((no + 1, l));
        }
        None
    }

    /// Data lines still unread. Consumes them, which is all a caller wants: this is only asked
    /// when the declared counts are already satisfied and anything left is a fault.
    fn remaining(&mut self) -> usize {
        let mut k = 0;
        while self.next_line().is_some() {
            k += 1;
        }
        k
    }
}

fn need<'a>(
    s: &mut Scanner<'a>,
    format: &'static str,
    want: &'static str,
) -> Result<(usize, &'a str), CorporaError> {
    s.next_line().ok_or(CorporaError::Truncated { format, want })
}

/// The line's single token, or [`CorporaError::Line`] when it carries none or more than one.
fn only<'a>(format: &'static str, line: usize, l: &'a str) -> Result<&'a str, CorporaError> {
    let mut it = l.split_whitespace();
    match (it.next(), it.next()) {
        (Some(t), None) => Ok(t),
        _ => Err(CorporaError::Line { format, line, text: l.to_string() }),
    }
}

/// A finite `f64`. `nan` and `inf` parse and are refused: both poison every energy they reach.
fn number(format: &'static str, line: usize, tok: &str) -> Result<f64, CorporaError> {
    match tok.parse::<f64>() {
        Ok(v) if v.is_finite() => Ok(v),
        _ => Err(CorporaError::Number { format, line, got: tok.to_string() }),
    }
}

/// A one-based index, returned zero-based. Out of range is named, never shifted or clamped.
fn index(format: &'static str, line: usize, tok: &str, n: usize) -> Result<usize, CorporaError> {
    let Ok(v) = tok.parse::<i64>() else {
        return Err(CorporaError::Number { format, line, got: tok.to_string() });
    };
    if v < 1 || v as u64 > n as u64 {
        return Err(CorporaError::Index { format, line, got: v, n });
    }
    Ok(v as usize - 1)
}

/// Exactly `count` entries of `i j v`, one-based, canonicalised to `i ≤ j` and sorted.
///
/// `diagonal` is whether `i == j` is meaningful: it is a linear coefficient in a QUBO and a
/// self-loop in a max-cut, and a self-loop crosses no cut.
fn read_entries(
    s: &mut Scanner<'_>,
    format: &'static str,
    what: &'static str,
    n: usize,
    count: usize,
    diagonal: bool,
) -> Result<Vec<(usize, usize, f64)>, CorporaError> {
    // Capacity from the header is capped. A damaged header is exactly the case this function
    // exists to catch, and `with_capacity` on a count of 10^11 aborts the process before any
    // error can be returned — a reader that dies on bad input has not refused it.
    let mut raw: Vec<(usize, usize, f64, usize)> = Vec::with_capacity(count.min(1 << 16));
    for _ in 0..count {
        let Some((no, l)) = s.next_line() else {
            return Err(CorporaError::Count { format, what, declared: count, found: raw.len() });
        };
        let mut it = l.split_whitespace();
        let (Some(a), Some(b), Some(c), None) = (it.next(), it.next(), it.next(), it.next()) else {
            return Err(CorporaError::Line { format, line: no, text: l.to_string() });
        };
        let i = index(format, no, a, n)?;
        let j = index(format, no, b, n)?;
        let v = number(format, no, c)?;
        if i == j && !diagonal {
            return Err(CorporaError::SelfLoop { format, line: no, at: i + 1 });
        }
        raw.push((i.min(j), i.max(j), v, no));
    }
    // Sorted by pair then by line, so a repeat is adjacent to its FIRST appearance and the error
    // can name both line numbers rather than whichever two the file happened to put together.
    raw.sort_by_key(|x| (x.0, x.1, x.3));
    for w in raw.windows(2) {
        if (w[0].0, w[0].1) == (w[1].0, w[1].1) {
            return Err(CorporaError::Duplicate {
                format,
                line: w[1].3,
                first: w[0].3,
                pair: (w[0].0 + 1, w[0].1 + 1),
            });
        }
    }
    Ok(raw.into_iter().map(|(i, j, v, _)| (i, j, v)).collect())
}

const RUDY: &str = "Biq Mac / rudy max-cut";
const MQ_CUT: &str = "MQLib max-cut";
const MQ_QUBO: &str = "MQLib QUBO";
const ORLIB: &str = "OR-Library bqp";
const QPLIB: &str = "QPLIB";

/// One `n m` header followed by `m` edges.
fn read_edge_list(
    text: &str,
    format: &'static str,
    corpus: Corpus,
    comment: Option<char>,
) -> Result<MaxCut, CorporaError> {
    let mut s = Scanner::new(text, comment, None);
    let (no, head) = need(&mut s, format, "the `n m` header")?;
    let bad = || CorporaError::Header { format, line: no, text: head.to_string() };
    let mut h = head.split_whitespace();
    let (Some(a), Some(b), None) = (h.next(), h.next(), h.next()) else {
        return Err(bad());
    };
    let (Ok(n), Ok(m)) = (a.parse::<usize>(), b.parse::<usize>()) else {
        return Err(bad());
    };
    let edges = read_entries(&mut s, format, "edges", n, m, false)?;
    let extra = s.remaining();
    if extra > 0 {
        return Err(CorporaError::Count { format, what: "edges", declared: m, found: m + extra });
    }
    Ok(MaxCut { name: String::new(), n, edges, corpus })
}

/// Read the Biq Mac library's max-cut format, which is Rinaldi's `rudy` edge list.
///
/// `n m`, then `m` lines of `i j w` with **one-based** vertices. The same grammar G-set is written
/// in ([`crate::gset`]), read here with the duplicate and self-loop checks that reader does not
/// have. MEASURED against `gset::Instance::parse`, on inputs this reader refuses by name:
///
/// * `"3 1\n2 2 1\n"` — a self-loop — **panics** there, `bad edge (1,1) n=3` from
///   `GraphBuilder::couple`'s assertion, rather than returning a `GsetError`.
/// * `"3 2\n1 2 1\n2 1 3\n"` — one pair given twice — is **accepted** there and merged, so the
///   returned instance reports `edges = 2` while its graph holds one edge of weight 4.
///
/// The instance carries no name in this format, so [`MaxCut::name`] comes back empty: the name a
/// Biq Mac instance is known by is its file's name, which only the caller has.
///
/// # Errors
///
/// [`CorporaError::Header`] for a header that is not `n m`, [`CorporaError::Line`] for a body line
/// that is not `i j w`, [`CorporaError::Index`] for a vertex outside `1..=n`,
/// [`CorporaError::Number`] for a weight that is not finite, [`CorporaError::SelfLoop`],
/// [`CorporaError::Duplicate`], and [`CorporaError::Count`] when the body length disagrees with the
/// header.
pub fn read_biqmac(text: &str) -> Result<MaxCut, CorporaError> {
    read_edge_list(text, RUDY, Corpus::BiqMac, None)
}

/// Read an `MQLib` max-cut instance.
///
/// The same `n m` / `i j w` grammar as [`read_biqmac`], and `MQLib` additionally tolerates `#`
/// comment lines. That tolerance is the entire difference between the two readers, and it is a
/// difference in what is ACCEPTED rather than in what anything means.
///
/// # Errors
///
/// As [`read_biqmac`].
pub fn read_mqlib_maxcut(text: &str) -> Result<MaxCut, CorporaError> {
    read_edge_list(text, MQ_CUT, Corpus::MqLib, Some('#'))
}

/// Read an `MQLib` QUBO instance: `n nnz`, then `nnz` lines of `i j q`, one-based, `i ≤ j`.
///
/// `MQLib` normalises both of its problems to **maximisation**, so the sense comes back
/// [`Sense::Maximise`]. The diagonal is the linear part, since `x² = x`, and an off-diagonal entry
/// is the coefficient of `x_i x_j` **once** — not half of a symmetric pair. The format has nowhere
/// to put a constant, so [`Qubo::constant`] is zero — which is what the format MEANS, not a default
/// standing in for something unreadable.
///
/// Those two conventions are the module header's "Provenance" section: a wrong reading of either
/// parses, samples, and reports a plausible number.
///
/// # Errors
///
/// As [`read_biqmac`], except that `i == j` is a coefficient here rather than a self-loop.
pub fn read_mqlib_qubo(text: &str) -> Result<Qubo, CorporaError> {
    let format = MQ_QUBO;
    let mut s = Scanner::new(text, Some('#'), None);
    let (no, head) = need(&mut s, format, "the `n nnz` header")?;
    let bad = || CorporaError::Header { format, line: no, text: head.to_string() };
    let mut h = head.split_whitespace();
    let (Some(a), Some(b), None) = (h.next(), h.next(), h.next()) else {
        return Err(bad());
    };
    let (Ok(n), Ok(nnz)) = (a.parse::<usize>(), b.parse::<usize>()) else {
        return Err(bad());
    };
    let terms = read_entries(&mut s, format, "coefficients", n, nnz, true)?;
    let extra = s.remaining();
    if extra > 0 {
        return Err(CorporaError::Count {
            format,
            what: "coefficients",
            declared: nnz,
            found: nnz + extra,
        });
    }
    Ok(Qubo {
        name: String::new(),
        n,
        terms,
        constant: 0.0,
        sense: Sense::Maximise,
        corpus: Corpus::MqLib,
    })
}

/// Read an OR-Library `bqp` file, which holds **many** instances.
///
/// A count of problems, then for each: `n nnz`, then `nnz` lines of `i j q`, one-based, `i ≤ j`.
/// The objective is a maximisation, the diagonal is the linear part, an off-diagonal entry is the
/// coefficient of `x_i x_j` **once** rather than half of a symmetric pair, and there is no
/// constant. See the module header's "Provenance" section for what a wrong reading of that last
/// convention would do, which is to parse and to report a plausible number.
///
/// The file carries no names, so a problem's **one-based position in it** is its name — `"1"` for
/// the first. A file of ten `bqp50` instances therefore comes back as ten distinguishable
/// [`Qubo`]s rather than ten anonymous ones.
///
/// # Errors
///
/// [`CorporaError::Header`] for the problem count or a per-problem `n nnz` line,
/// [`CorporaError::Count`] when the file holds fewer problems than it declares,
/// [`CorporaError::Line`] for content after the last declared problem, and the body errors of
/// [`read_biqmac`].
pub fn read_orlib(text: &str) -> Result<Vec<Qubo>, CorporaError> {
    let format = ORLIB;
    let mut s = Scanner::new(text, None, None);
    let (no, head) = need(&mut s, format, "the problem count")?;
    let Ok(p) = only(format, no, head)?.parse::<usize>() else {
        return Err(CorporaError::Header { format, line: no, text: head.to_string() });
    };
    let mut out = Vec::with_capacity(p.min(1 << 16));
    for k in 0..p {
        let Some((hno, hl)) = s.next_line() else {
            return Err(CorporaError::Count { format, what: "problems", declared: p, found: k });
        };
        let bad = || CorporaError::Header { format, line: hno, text: hl.to_string() };
        let mut h = hl.split_whitespace();
        let (Some(a), Some(b), None) = (h.next(), h.next(), h.next()) else {
            return Err(bad());
        };
        let (Ok(n), Ok(nnz)) = (a.parse::<usize>(), b.parse::<usize>()) else {
            return Err(bad());
        };
        let terms = read_entries(&mut s, format, "coefficients", n, nnz, true)?;
        out.push(Qubo {
            name: (k + 1).to_string(),
            n,
            terms,
            constant: 0.0,
            sense: Sense::Maximise,
            corpus: Corpus::OrLibrary,
        });
    }
    if let Some((no, l)) = s.next_line() {
        return Err(CorporaError::Line { format, line: no, text: l.to_string() });
    }
    Ok(out)
}

/// Read a QPLIB instance, in the **binary unconstrained** subset.
///
/// The sections, in the order the format writes them: the problem name, the three-character type
/// code, `Minimize` or `Maximize`, the variable count, the nonzero count of `Q⁰` and its entries
/// (absent when the objective type is `L`), the default value of `b⁰`, the count of its
/// non-default entries and those entries, and the objective constant `q⁰`. Comments run from a
/// `!` to the end of the line.
///
/// The type code is checked **first**, before a coefficient is read: the second character must be
/// `B` (binary variables) and the third `N` (no constraints), and everything else is refused naming
/// itself. That refusal is what makes it safe to stop at `q⁰` — the sections QPLIB writes after it
/// for an instance of this type describe a starting point and the variable names, neither of which
/// changes what is solved.
///
/// The objective is `½ xᵀQ⁰x + b⁰ᵀx + q⁰` with `Q⁰` symmetric and its **lower** triangle listed, so
/// an off-diagonal pair contributes `Q_ij` once and the diagonal keeps its half: `c_ii = ½ Q_ii +
/// b_i`. An entry above the diagonal is read as the symmetric pair it names, and naming the same
/// pair twice is refused rather than summed.
///
/// # Errors
///
/// [`CorporaError::Truncated`] for a file that ends inside a section,
/// [`CorporaError::Unsupported`] for a type code outside the subset, [`CorporaError::Sense`] for an
/// objective sense that is neither word, [`CorporaError::Header`] for an unreadable count, and the
/// body errors of [`read_biqmac`].
pub fn read_qplib(text: &str) -> Result<Qubo, CorporaError> {
    let format = QPLIB;
    let mut s = Scanner::new(text, None, Some('!'));

    let (nno, name) = need(&mut s, format, "the problem name")?;
    let name = only(format, nno, name)?.to_string();

    let (tno, code) = need(&mut s, format, "the three-character type code")?;
    let code = only(format, tno, code)?.to_string();
    let up: Vec<char> = code.to_ascii_uppercase().chars().collect();
    if up.len() != 3 {
        return Err(CorporaError::Header { format, line: tno, text: code });
    }
    if !matches!(up[0], 'L' | 'D' | 'C' | 'Q') {
        return Err(CorporaError::Unsupported {
            code,
            why: "the first character is the objective type and must be L, D, C or Q",
        });
    }
    if up[1] != 'B' {
        return Err(CorporaError::Unsupported {
            code,
            why: "the second character is the variable type, and only B (binary) is an Ising \
                  model — C, M, I and G are continuous, mixed or general-integer",
        });
    }
    if up[2] != 'N' {
        return Err(CorporaError::Unsupported {
            code,
            why: "the third character is the constraint type, and only N (unconstrained) is read \
                  — a constrained instance becomes an Ising model only through a penalty weight \
                  nobody declared",
        });
    }

    let (sno, sl) = need(&mut s, format, "`Minimize` or `Maximize`")?;
    let sense = if sl.eq_ignore_ascii_case("minimize") {
        Sense::Minimise
    } else if sl.eq_ignore_ascii_case("maximize") {
        Sense::Maximise
    } else {
        return Err(CorporaError::Sense { line: sno, got: sl.to_string() });
    };

    let (vno, vl) = need(&mut s, format, "the variable count")?;
    let Ok(n) = only(format, vno, vl)?.parse::<usize>() else {
        return Err(CorporaError::Header { format, line: vno, text: vl.to_string() });
    };

    // A linear objective has no Q0 section at all; the other three codes always write one, even
    // when it is empty.
    let quad = if up[0] == 'L' {
        Vec::new()
    } else {
        let (cno, cl) = need(&mut s, format, "the nonzero count of Q0")?;
        let Ok(nnz) = only(format, cno, cl)?.parse::<usize>() else {
            return Err(CorporaError::Header { format, line: cno, text: cl.to_string() });
        };
        read_entries(&mut s, format, "quadratic coefficients", n, nnz, true)?
    };

    let (dno, dl) = need(&mut s, format, "the default value of b0")?;
    let default = number(format, dno, only(format, dno, dl)?)?;
    let (bno, bl) = need(&mut s, format, "the count of non-default entries of b0")?;
    let Ok(nb) = only(format, bno, bl)?.parse::<usize>() else {
        return Err(CorporaError::Header { format, line: bno, text: bl.to_string() });
    };

    // A diagonal coefficient exists when EITHER Q0's diagonal or b0 named it; `default` being
    // nonzero names every variable at once. Tracked rather than inferred from the value, so a
    // coefficient a file wrote as zero survives a round trip instead of vanishing.
    let mut diag = vec![0.0f64; n];
    let mut named = vec![default != 0.0; n];
    let mut first_at = vec![0usize; n];
    let mut off: Vec<(usize, usize, f64)> = Vec::with_capacity(quad.len());
    for (i, j, v) in quad {
        if i == j {
            diag[i] += v * 0.5;
            named[i] = true;
        } else {
            off.push((i, j, v));
        }
    }
    for i in 0..n {
        diag[i] += default;
    }
    let mut seen = vec![false; n];
    for k in 0..nb {
        let Some((no, l)) = s.next_line() else {
            return Err(CorporaError::Count {
                format,
                what: "linear coefficients",
                declared: nb,
                found: k,
            });
        };
        let mut it = l.split_whitespace();
        let (Some(a), Some(b), None) = (it.next(), it.next(), it.next()) else {
            return Err(CorporaError::Line { format, line: no, text: l.to_string() });
        };
        let i = index(format, no, a, n)?;
        let v = number(format, no, b)?;
        if seen[i] {
            return Err(CorporaError::Duplicate {
                format,
                line: no,
                first: first_at[i],
                pair: (i + 1, i + 1),
            });
        }
        seen[i] = true;
        first_at[i] = no;
        // A NON-DEFAULT entry replaces the default rather than adding to it, which is what
        // "non-default" means; the Q0 diagonal's own half is a different coefficient and stays.
        diag[i] += v - default;
        named[i] = true;
    }

    let (qno, ql) = need(&mut s, format, "the objective constant q0")?;
    let constant = number(format, qno, only(format, qno, ql)?)?;

    let mut terms = off;
    for i in 0..n {
        if named[i] {
            terms.push((i, i, diag[i]));
        }
    }
    terms.sort_by_key(|a| (a.0, a.1));
    Ok(Qubo { name, n, terms, constant, sense, corpus: Corpus::Qplib })
}

// ---- writing ------------------------------------------------------------------------------------

/// Rust's shortest round-tripping decimal for an `f64`, which is what makes a write/read round trip
/// bit-exact rather than merely close. `Display` never emits exponent notation, which none of these
/// readers is specified to accept.
fn num(v: f64) -> String {
    format!("{v}")
}

/// Write the Biq Mac / `rudy` max-cut format: `n m`, then `i j w` one-based.
///
/// This is byte-for-byte what `MQLib` reads for a max-cut too. The two libraries differ in what
/// they TOLERATE — `MQLib` allows `#` comments — not in what they emit, so one writer serves both
/// and `a_round_trip_through_every_format_is_bitwise_identical` reads its output back through each.
#[must_use]
pub fn write_maxcut(mc: &MaxCut) -> String {
    let mut out = format!("{} {}\n", mc.n, mc.edges.len());
    for &(i, j, w) in &mc.edges {
        out.push_str(&format!("{} {} {}\n", i + 1, j + 1, num(w)));
    }
    out
}

/// Refuse a [`Qubo`] whose meaning a sense-less, constant-less format cannot carry.
fn qubo_fits(q: &Qubo, format: &'static str) -> Result<(), CorporaError> {
    if q.sense != Sense::Maximise {
        return Err(CorporaError::Unrepresentable {
            format,
            what: format!("a {} — this format is a maximisation by definition", q.sense),
        });
    }
    if q.constant != 0.0 {
        return Err(CorporaError::Unrepresentable {
            format,
            what: format!("a constant of {}", num(q.constant)),
        });
    }
    Ok(())
}

/// Write an `MQLib` QUBO: `n nnz`, then `i j q` one-based.
///
/// # Errors
///
/// [`CorporaError::Unrepresentable`] for a minimisation or a nonzero constant. The format has
/// nowhere to record either, and writing the model without them would produce a file that reads
/// back as a different problem — which is precisely the failure [`crate::dimod`]'s header
/// describes, committed by the writer instead of the reader.
pub fn write_mqlib_qubo(q: &Qubo) -> Result<String, CorporaError> {
    qubo_fits(q, MQ_QUBO)?;
    let mut out = format!("{} {}\n", q.n, q.terms.len());
    for &(i, j, c) in &q.terms {
        out.push_str(&format!("{} {} {}\n", i + 1, j + 1, num(c)));
    }
    Ok(out)
}

/// Write an OR-Library `bqp` file: the problem count, then each problem's `n nnz` and entries.
///
/// # Errors
///
/// [`CorporaError::Unrepresentable`], as [`write_mqlib_qubo`], for any problem in the list.
pub fn write_orlib(problems: &[Qubo]) -> Result<String, CorporaError> {
    let mut out = format!("{}\n", problems.len());
    for q in problems {
        qubo_fits(q, ORLIB)?;
        out.push_str(&format!("{} {}\n", q.n, q.terms.len()));
        for &(i, j, c) in &q.terms {
            out.push_str(&format!("{} {} {}\n", i + 1, j + 1, num(c)));
        }
    }
    Ok(out)
}

/// Write a QPLIB instance of type `QBN`: a quadratic objective over binary variables, no
/// constraints.
///
/// The whole linear part is written in `b⁰` and `Q⁰`'s diagonal is left empty, which is one of the
/// two ways the format can say the same thing and the one that survives a round trip without
/// needing to guess how a producer split `c_ii` between them. Off-diagonal entries go in the
/// **lower** triangle, as the format specifies.
///
/// # Errors
///
/// [`CorporaError::Unrepresentable`] for a name that is empty, or that carries whitespace or a `!`:
/// the name occupies one whole line and a comment marker or a line break inside it would shift
/// every section that follows.
pub fn write_qplib(q: &Qubo) -> Result<String, CorporaError> {
    if q.name.is_empty() || q.name.contains(char::is_whitespace) || q.name.contains('!') {
        return Err(CorporaError::Unrepresentable {
            format: QPLIB,
            what: format!(
                "the name {:?} — it occupies one whole line, so it cannot be empty or carry \
                 whitespace or a `!`",
                q.name
            ),
        });
    }
    let (diag, off): (Vec<_>, Vec<_>) = q.terms.iter().partition(|&&(i, j, _)| i == j);
    let mut out = String::new();
    out.push_str(&format!("{}\n", q.name));
    out.push_str("QBN\n");
    out.push_str(&format!("{}\n", q.sense.qplib_word()));
    out.push_str(&format!("{}\n", q.n));
    out.push_str(&format!("{}\n", off.len()));
    for &(i, j, c) in &off {
        // Lower triangle: the row index is the larger of the pair.
        out.push_str(&format!("{} {} {}\n", j + 1, i + 1, num(c)));
    }
    out.push_str("0\n");
    out.push_str(&format!("{}\n", diag.len()));
    for &(i, _, c) in &diag {
        out.push_str(&format!("{} {}\n", i + 1, num(c)));
    }
    out.push_str(&format!("{}\n", num(q.constant)));
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One 4-variable QUBO, written three ways. `f(x) = 5x₀ − 3x₁ + 6x₃ + 7x₀x₁ − x₀x₃ − 2x₁x₂ +
    /// 4x₂x₃`, maximised.
    ///
    /// The QPLIB copy deliberately splits `c₀₀ = 5` as `½·4 + 3` rather than putting it all in
    /// `b⁰`, so the `½` on `Q⁰`'s diagonal is exercised rather than assumed: read that half as a
    /// whole and `c₀₀` comes out 7.
    const ORLIB_TEXT: &str = "\
1
4 7
1 1 5
1 2 7
1 4 -1
2 2 -3
2 3 -2
3 4 4
4 4 6
";

    const MQLIB_TEXT: &str = "\
# the same instance, MQLib's envelope
4 7
1 1 5
1 2 7
1 4 -1
2 2 -3
2 3 -2
3 4 4
4 4 6
";

    const QPLIB_TEXT: &str = "\
! the same instance again
smallq          ! problem name
QBN             ! quadratic objective, binary variables, no constraints
Maximize        ! sense
4               ! variables
5               ! nonzeros in the lower triangle of Q0
1 1 4
2 1 7
4 1 -1
3 2 -2
4 3 4
0               ! default value in b0
3               ! non-default entries of b0
1 3
2 -3
4 6
0               ! q0
";

    /// A five-vertex weighted graph, mixed signs, in the `rudy` edge list.
    const RUDY_TEXT: &str = "\
5 6
1 2 3
1 3 2
2 3 -2
3 4 5
4 5 1
1 5 4
";

    fn states(n: usize) -> Vec<Vec<i8>> {
        (0u32..(1u32 << n))
            .map(|m| (0..n).map(|i| if m >> i & 1 == 1 { 1i8 } else { -1 }).collect())
            .collect()
    }

    fn three_qubos() -> Vec<Qubo> {
        vec![
            read_orlib(ORLIB_TEXT).expect("OR-Library").remove(0),
            read_mqlib_qubo(MQLIB_TEXT).expect("MQLib"),
            read_qplib(QPLIB_TEXT).expect("QPLIB"),
        ]
    }

    /// THE ORACLE: [`crate::dimod::Bqm`], which carries its own verification, scores every state
    /// of every QUBO format the same way this module does.
    ///
    /// The comparison runs twice over the same states and neither leg shares code with the reader
    /// under test. `Bqm::energy` is an independent evaluator of the QUBO itself, written against
    /// `dimod`'s own `E = offset + Σ a x + Σ b x x`; `Bqm::to_graph` is an independent
    /// implementation of the `x = (1+s)/2` substitution, which is the arithmetic
    /// [`Qubo::instance`] exists to perform. A factor of two on an off-diagonal, a half lost from
    /// a diagonal, or a sign on the substitution's constant all show up here at every one of the
    /// 16 states rather than at an optimum, where a wrong model can still name the right state.
    #[test]
    fn oracle_dimod_bqm_scores_every_qubo_format_the_same_at_every_state() {
        use crate::dimod::{Bqm, Vartype, binary_state};
        // The coefficients, transcribed from the fixture rather than taken from a reader.
        let raw: [(usize, usize, f64); 7] = [
            (0, 0, 5.0),
            (0, 1, 7.0),
            (0, 3, -1.0),
            (1, 1, -3.0),
            (1, 2, -2.0),
            (2, 3, 4.0),
            (3, 3, 6.0),
        ];
        let mut bqm = Bqm::new(Vartype::Binary, 4);
        for (i, j, c) in raw {
            if i == j {
                bqm.bias(i, c);
            } else {
                bqm.couple(i, j, c);
            }
        }
        let (dimod_graph, dimod_const) = bqm.to_graph();

        for q in three_qubos() {
            let inst = q.instance();
            assert_eq!(inst.sense, Sense::Maximise, "{}: all three corpora maximise", q.corpus);
            for s in states(4) {
                let x = binary_state(&s);
                let got = inst.objective(&s);
                let by_value = bqm.energy(&x);
                let by_graph = dimod_graph.energy(&s) + dimod_const;
                assert!(
                    (got - by_value).abs() < 1e-12,
                    "{} at {s:?}: objective {got}, dimod's QUBO evaluator {by_value}",
                    q.corpus
                );
                assert!(
                    (got - by_graph).abs() < 1e-12,
                    "{} at {s:?}: objective {got}, dimod's own Ising conversion {by_graph}",
                    q.corpus
                );
                // And the module's own QUBO evaluator, over the file's coefficients.
                assert!((q.value(&x) - by_value).abs() < 1e-12);
            }
        }
    }

    /// Three serialisations of one QUBO parse to the same coefficients and the same energies.
    ///
    /// This is the oracle no single parser can supply: QPLIB states the model as `½ xᵀQ⁰x + b⁰ᵀx`
    /// over a LOWER triangle, and the other two state it as an UPPER-triangular `Q` whose diagonal
    /// is already linear. The two conventions are converted by different code, so agreement over
    /// all 16 states is a statement about the conventions and not about a shared routine.
    #[test]
    fn oracle_cross_format_qplib_orlib_and_mqlib_agree_over_exhaustive_enumeration() {
        let qs = three_qubos();
        for q in &qs[1..] {
            assert_eq!(q.terms, qs[0].terms, "{} disagrees with OR-Library on coefficients", q.corpus);
            assert_eq!(q.constant, qs[0].constant);
        }
        assert_eq!(qs[2].name, "smallq", "QPLIB names its instances and the name is carried");
        assert_eq!(qs[0].name, "1", "OR-Library does not, so the position in the file is the name");

        let insts: Vec<Instance> = qs.iter().map(Qubo::instance).collect();
        for s in states(4) {
            let want = insts[0].objective(&s);
            for inst in &insts[1..] {
                assert!(
                    (inst.objective(&s) - want).abs() < 1e-12,
                    "{} gives {} where {} gives {want} at {s:?}",
                    inst.corpus,
                    inst.objective(&s),
                    insts[0].corpus
                );
            }
        }
    }

    /// THE ORACLE: the textbook definition of a cut — the weights of the edges whose ends land on
    /// opposite sides — computed over every one of the 32 states from the file's own numbers.
    ///
    /// This is what the negation in [`MaxCut::instance`] is for. Load `J = +w` and the energy
    /// minimum is the MINIMUM cut: a valid state, a plausible number, and the opposite problem.
    /// Both max-cut dialects are read, since one writer serves both.
    #[test]
    fn oracle_the_crossing_edge_sum_matches_every_max_cut_instance() {
        // Transcribed from RUDY_TEXT, zero-based, not taken from a reader.
        let edges: [(usize, usize, f64); 6] = [
            (0, 1, 3.0),
            (0, 2, 2.0),
            (1, 2, -2.0),
            (2, 3, 5.0),
            (3, 4, 1.0),
            (0, 4, 4.0),
        ];
        for mc in [read_biqmac(RUDY_TEXT).unwrap(), read_mqlib_maxcut(RUDY_TEXT).unwrap()] {
            let inst = mc.instance();
            assert_eq!(inst.problem, Problem::MaxCut);
            assert_eq!(inst.sense, Sense::Maximise);
            let mut best = f64::NEG_INFINITY;
            let mut best_at_min_energy = 0.0;
            let mut lowest = f64::INFINITY;
            for s in states(5) {
                let textbook: f64 =
                    edges.iter().filter(|&&(i, j, _)| s[i] != s[j]).map(|&(_, _, w)| w).sum();
                assert!(
                    (inst.objective(&s) - textbook).abs() < 1e-12,
                    "{}: objective {} where the crossing edges weigh {textbook} at {s:?}",
                    inst.corpus,
                    inst.objective(&s)
                );
                best = best.max(textbook);
                if inst.graph.energy(&s) < lowest {
                    lowest = inst.graph.energy(&s);
                    best_at_min_energy = textbook;
                }
            }
            // The energy MINIMUM must be the cut MAXIMUM, which is the whole point of the sign.
            assert!((best_at_min_energy - best).abs() < 1e-12, "{lowest} minimises energy at a cut of {best_at_min_energy}, max is {best}");
            // 15: every positive edge crosses and the single −2 does not, which is achievable
            // because s₁ = s₂ is consistent with s₀ ≠ s₁ and s₀ ≠ s₂.
            assert!((best - 15.0).abs() < 1e-12, "the max cut of this instance is 15, got {best}");
        }
    }

    /// THE ORACLE: Hammer's 1965 root-node transformation makes the cut EQUAL the QUBO objective —
    /// at every state, not at the optimum.
    ///
    /// The published relationship between two of these corpora: the max-cut instances the
    /// literature calls "Beasley" are OR-Library `bqp` instances put through exactly this. Checked
    /// across a format boundary as well as an encoding one — the transformed instance is written
    /// as a Biq Mac file and read back before it is scored — so a round trip that lost a weight
    /// would show here too.
    ///
    /// The state correspondence is `x_i = [y_{i+1} ≠ y_0]`, which is the ONLY place the root node
    /// means anything; checking at the optimum alone would pass a transformation that scrambled
    /// the states while preserving their multiset of values.
    #[test]
    fn oracle_the_root_node_transformation_makes_the_cut_the_qubo_objective() {
        let q = read_orlib(ORLIB_TEXT).unwrap().remove(0);
        let mc = q.to_maxcut().expect("a maximisation with no constant");
        assert_eq!(mc.n, q.n + 1, "one extra vertex, the root");
        let round = read_biqmac(&write_maxcut(&mc)).expect("its own rudy file");
        assert_eq!(round.edges, mc.edges);
        let inst = round.instance();

        let (mut best_cut, mut best_f) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
        for y in states(5) {
            let x: Vec<i8> = (0..4).map(|i| i8::from(y[i + 1] != y[0])).collect();
            let f = q.value(&x);
            let cut = inst.objective(&y);
            assert!(
                (cut - f).abs() < 1e-12,
                "cut {cut} but f(x) = {f} at y = {y:?}, x = {x:?}"
            );
            assert!((mc.cut(&y) - f).abs() < 1e-12, "the stored edges disagree with the graph");
            best_cut = best_cut.max(cut);
            best_f = best_f.max(f);
        }
        assert!((best_cut - best_f).abs() < 1e-12, "max cut {best_cut}, max f {best_f}");
    }

    /// Every writer's output reads back to an instance that scores a fixed random state to the
    /// BIT, and writing it again produces the same bytes.
    ///
    /// Bit-equality WITHIN one machine is what this asserts, which the crate does promise — see the
    /// crate header on the one ULP that floating-point contraction can move across operating
    /// systems. The canonical term order is what makes it hold: both the CSR and the offset's
    /// accumulation order are then properties of the model rather than of the file's line order, so
    /// a file whose lines were shuffled still round-trips to the same bits.
    #[test]
    fn a_round_trip_through_every_format_is_bitwise_identical() {
        let mut rng = crate::rng::Pcg::new(20_260_911, 7);
        let q = read_qplib(QPLIB_TEXT).unwrap();
        let s4: Vec<i8> = (0..4).map(|_| rng.spin(0.5)).collect();
        let s5: Vec<i8> = (0..5).map(|_| rng.spin(0.5)).collect();

        let cases: Vec<(&str, String, Qubo)> = vec![
            ("QPLIB", write_qplib(&q).unwrap(), read_qplib(&write_qplib(&q).unwrap()).unwrap()),
            (
                "MQLib QUBO",
                write_mqlib_qubo(&q).unwrap(),
                read_mqlib_qubo(&write_mqlib_qubo(&q).unwrap()).unwrap(),
            ),
            (
                "OR-Library",
                write_orlib(std::slice::from_ref(&q)).unwrap(),
                read_orlib(&write_orlib(std::slice::from_ref(&q)).unwrap()).unwrap().remove(0),
            ),
        ];
        let first = q.instance();
        for (name, text, back) in cases {
            assert_eq!(back.terms, q.terms, "{name}: coefficients moved");
            let again = back.instance();
            assert_eq!(
                first.graph.energy(&s4).to_bits(),
                again.graph.energy(&s4).to_bits(),
                "{name}: energy is not bit-identical after a round trip"
            );
            assert_eq!(
                first.objective(&s4).to_bits(),
                again.objective(&s4).to_bits(),
                "{name}: objective is not bit-identical after a round trip"
            );
            // Writing what was read back must reproduce the bytes, or the format is lossy in a way
            // the energy happens not to see.
            let rewritten = match name {
                "QPLIB" => write_qplib(&back).unwrap(),
                "MQLib QUBO" => write_mqlib_qubo(&back).unwrap(),
                _ => write_orlib(std::slice::from_ref(&back)).unwrap(),
            };
            assert_eq!(rewritten, text, "{name}: the second write differs from the first");
        }

        let mc = read_biqmac(RUDY_TEXT).unwrap();
        let text = write_maxcut(&mc);
        for back in [read_biqmac(&text).unwrap(), read_mqlib_maxcut(&text).unwrap()] {
            assert_eq!(back.edges, mc.edges);
            assert_eq!(
                mc.instance().graph.energy(&s5).to_bits(),
                back.instance().graph.energy(&s5).to_bits()
            );
            assert_eq!(
                mc.instance().objective(&s5).to_bits(),
                back.instance().objective(&s5).to_bits()
            );
            assert_eq!(write_maxcut(&back), text);
        }
    }

    fn refuse(which: &str, text: &str) -> Result<(), CorporaError> {
        match which {
            "qplib" => read_qplib(text).map(|_| ()),
            "biqmac" => read_biqmac(text).map(|_| ()),
            "orlib" => read_orlib(text).map(|_| ()),
            "mqcut" => read_mqlib_maxcut(text).map(|_| ()),
            "mqqubo" => read_mqlib_qubo(text).map(|_| ()),
            other => panic!("no reader named {other}"),
        }
    }

    /// Every malformed input, case by case, refused by a typed error that names the problem.
    ///
    /// A parser tested only on valid input is untested: invariant 10 of `AGENTS.md` is that a
    /// default is not a fallback, and the way that is broken is by a reader quietly repairing what
    /// it was handed. Each row asserts the SHAPE of the refusal, not merely that one happened, so
    /// a reader that collapsed every fault into one variant would fail here.
    #[test]
    fn malformed_input_is_refused_case_by_case_with_a_typed_error() {
        type Check = fn(&CorporaError) -> bool;
        let cases: &[(&str, &str, &str, Check)] = &[
            ("biqmac", "", "nothing at all", |e| {
                matches!(e, CorporaError::Truncated { .. })
            }),
            ("biqmac", "3\n1 2 1\n", "a header that is not `n m`", |e| {
                matches!(e, CorporaError::Header { .. })
            }),
            ("biqmac", "3 3\n1 2 1\n", "a truncated body", |e| {
                matches!(e, CorporaError::Count { declared: 3, found: 1, what: "edges", .. })
            }),
            ("biqmac", "3 1\n1 2 1\n2 3 1\n", "a body longer than the header", |e| {
                matches!(e, CorporaError::Count { declared: 1, found: 2, .. })
            }),
            ("biqmac", "3 1\n1 2 x\n", "a non-numeric weight", |e| {
                matches!(e, CorporaError::Number { line: 2, .. })
            }),
            ("biqmac", "3 1\n1 2 nan\n", "a nan weight, which parses as an f64", |e| {
                matches!(e, CorporaError::Number { .. })
            }),
            ("biqmac", "3 1\n1 2 inf\n", "an infinite weight", |e| {
                matches!(e, CorporaError::Number { .. })
            }),
            ("biqmac", "3 2\n1 2 1\n2 1 3\n", "a duplicate edge, given the other way round", |e| {
                matches!(e, CorporaError::Duplicate { line: 3, first: 2, pair: (1, 2), .. })
            }),
            ("biqmac", "3 1\n2 2 1\n", "a self-loop", |e| {
                matches!(e, CorporaError::SelfLoop { line: 2, at: 2, .. })
            }),
            ("biqmac", "3 1\n0 1 1\n", "a zero-based vertex", |e| {
                matches!(e, CorporaError::Index { got: 0, n: 3, .. })
            }),
            ("biqmac", "3 1\n1 4 1\n", "a vertex past the header's count", |e| {
                matches!(e, CorporaError::Index { got: 4, n: 3, .. })
            }),
            ("biqmac", "3 1\n1 2\n", "a body line with two fields", |e| {
                matches!(e, CorporaError::Line { line: 2, .. })
            }),
            ("biqmac", "3 1\n# a comment\n1 2 1\n", "a `#` line, which rudy does not carry", |e| {
                matches!(e, CorporaError::Number { line: 2, .. })
            }),
            ("mqcut", "3 1\n2 2 1\n", "a self-loop, in MQLib's dialect too", |e| {
                matches!(e, CorporaError::SelfLoop { at: 2, .. })
            }),
            ("mqqubo", "3 2\n1 1 5\n1 1 6\n", "a coefficient given twice", |e| {
                matches!(e, CorporaError::Duplicate { line: 3, first: 2, pair: (1, 1), .. })
            }),
            ("mqqubo", "3 1\n1 2 4\nextra\n", "content past the declared count", |e| {
                matches!(e, CorporaError::Count { declared: 1, found: 2, .. })
            }),
            ("orlib", "2\n2 1\n1 1 5\n", "fewer problems than declared", |e| {
                matches!(e, CorporaError::Count { what: "problems", declared: 2, found: 1, .. })
            }),
            ("orlib", "1\n2 1\n1 1 5\n2 1\n1 1 5\n", "content past the last problem", |e| {
                matches!(e, CorporaError::Line { line: 4, .. })
            }),
            ("orlib", "many\n", "a problem count that is not a number", |e| {
                matches!(e, CorporaError::Header { line: 1, .. })
            }),
            ("qplib", "p\nQCN\nMinimize\n4\n", "continuous variables", |e| {
                matches!(e, CorporaError::Unsupported { .. })
            }),
            ("qplib", "p\nQBL\nMinimize\n4\n", "linear constraints", |e| {
                matches!(e, CorporaError::Unsupported { .. })
            }),
            ("qplib", "p\nXBN\nMinimize\n4\n", "an objective type outside L, D, C, Q", |e| {
                matches!(e, CorporaError::Unsupported { .. })
            }),
            ("qplib", "p\nQB\nMinimize\n4\n", "a type code that is not three characters", |e| {
                matches!(e, CorporaError::Header { line: 2, .. })
            }),
            ("qplib", "p\nQBN\nSolve\n4\n", "an objective sense that is neither word", |e| {
                matches!(e, CorporaError::Sense { line: 3, .. })
            }),
            ("qplib", "p\nQBN\nMinimize\n4\n", "a file that stops before Q0's count", |e| {
                matches!(e, CorporaError::Truncated { want: "the nonzero count of Q0", .. })
            }),
            ("qplib", "p\nQBN\nMinimize\n4\n2\n1 1 3\n", "a Q0 section shorter than its count", |e| {
                matches!(
                    e,
                    CorporaError::Count { what: "quadratic coefficients", declared: 2, found: 1, .. }
                )
            }),
            (
                "qplib",
                "p\nQBN\nMinimize\n4\n0\n0\n2\n1 3\n1 4\n0\n",
                "the same linear coefficient named twice",
                |e| matches!(e, CorporaError::Duplicate { line: 9, first: 8, pair: (1, 1), .. }),
            ),
            (
                "qplib",
                "p\nQBN\nMinimize\n4\n0\n0\n0\nnot-a-number\n",
                "an objective constant that is not a number",
                |e| matches!(e, CorporaError::Number { line: 8, .. }),
            ),
            (
                "qplib",
                "p\nQBN\nMinimize\n4\n0\n0\n1\n5 3\n0\n",
                "a linear coefficient past the variable count",
                |e| matches!(e, CorporaError::Index { got: 5, n: 4, .. }),
            ),
        ];
        for &(which, text, why, ok) in cases {
            let err = refuse(which, text)
                .expect_err(&format!("{which} accepted {why}: {text:?}"));
            assert!(ok(&err), "{which} refused {why} with the wrong variant: {err:?} — {err}");
        }
    }

    /// And the decorations each format DOES carry are accepted, or the table above would pass for
    /// a reader that refused everything.
    #[test]
    fn the_decorations_each_format_carries_are_accepted() {
        let mq = read_mqlib_maxcut("# a comment\n\n3 1\n1 2 5\n").expect("MQLib allows `#`");
        assert_eq!(mq.edges, vec![(0, 1, 5.0)]);
        let q = read_mqlib_qubo("3 1\n2 2 5\n").expect("a diagonal is a coefficient, not a loop");
        assert_eq!(q.terms, vec![(1, 1, 5.0)]);
        let l = read_qplib("p\nLBN\nMinimize\n2\n0\n2\n1 4\n2 -1\n1.5\n")
            .expect("a linear objective has no Q0 section at all");
        assert_eq!(l.terms, vec![(0, 0, 4.0), (1, 1, -1.0)]);
        assert_eq!(l.constant, 1.5);
        assert_eq!(l.sense, Sense::Minimise);
        // A nonzero `b⁰` default names EVERY variable, and a listed entry REPLACES it rather than
        // adding to it — which is what "non-default entries" means. Reading it as an increment
        // gives variable 2 a coefficient of 1.5 instead of −1.
        let d = read_qplib("p\nQBN\nMinimize\n3\n0\n2.5\n1\n2 -1\n0\n").unwrap();
        assert_eq!(d.terms, vec![(0, 0, 2.5), (1, 1, -1.0), (2, 2, 2.5)]);
        let many = read_orlib("2\n2 1\n1 1 5\n3 2\n1 2 4\n3 3 1\n").expect("two problems");
        assert_eq!(many.len(), 2);
        assert_eq!((many[0].name.as_str(), many[1].name.as_str()), ("1", "2"));
        assert_eq!(many[1].n, 3);
    }

    /// THE ASYMMETRIC ONE. The sense is read, and reading it wrong is visible: the same
    /// coefficients under `Minimize` and `Maximize` put their energy minimum on DIFFERENT states.
    ///
    /// A reader that ignored the sense line — or an `instance()` that dropped the negation — would
    /// still produce a valid graph, a valid ground state and a plausible objective. What it could
    /// not do is give the two files different answers. Both directions are asserted against
    /// exhaustive enumeration, and the two optimal states are asserted to differ, so a test that
    /// merely "succeeded" would fail.
    #[test]
    fn the_sense_is_carried_and_a_maximisation_is_not_quietly_minimised() {
        let maxi = read_qplib(QPLIB_TEXT).unwrap();
        let mini = read_qplib(&QPLIB_TEXT.replace("Maximize", "Minimize")).unwrap();
        assert_eq!(maxi.terms, mini.terms, "only the sense line differs");
        assert_eq!((maxi.sense, mini.sense), (Sense::Maximise, Sense::Minimise));

        let (a, b) = (maxi.instance(), mini.instance());
        assert!(a.scale < 0.0 && b.scale > 0.0, "the slope's sign IS the sense");

        let mut hi = (f64::NEG_INFINITY, Vec::new());
        let mut lo = (f64::INFINITY, Vec::new());
        for s in states(4) {
            let f = maxi.value(&crate::dimod::binary_state(&s));
            if f > hi.0 {
                hi = (f, s.clone());
            }
            if f < lo.0 {
                lo = (f, s);
            }
        }
        assert!(hi.0 > lo.0, "the fixture must actually have two distinct optima");
        assert_ne!(hi.1, lo.1, "and they must be different states, or this proves nothing");

        let best = |inst: &Instance| {
            states(4)
                .into_iter()
                .min_by(|x, y| {
                    inst.graph.energy(x).partial_cmp(&inst.graph.energy(y)).expect("finite")
                })
                .unwrap()
        };
        assert_eq!(best(&a), hi.1, "the maximising file's energy minimum must be the MAXIMUM of f");
        assert_eq!(best(&b), lo.1, "and the minimising file's must be the MINIMUM");
        assert!((a.objective(&hi.1) - hi.0).abs() < 1e-12);
        assert!((b.objective(&lo.1) - lo.0).abs() < 1e-12);
    }

    /// The offset is invisible to the ranking and fatal to the value, and is carried for that
    /// reason.
    ///
    /// QPLIB's `q⁰` shifts every state by the same amount, so an optimiser cannot tell it was
    /// dropped: the graph is IDENTICAL and the argmin is identical. Only the reported number moves,
    /// which is what makes a comparison against a published objective fail while every solver test
    /// passes — the failure [`crate::dimod`] records.
    #[test]
    fn a_dropped_constant_would_leave_the_ranking_intact_and_the_value_wrong() {
        let plain = read_qplib(QPLIB_TEXT).unwrap();
        let shifted = read_qplib(&QPLIB_TEXT.replace("0               ! q0", "17.25           ! q0"))
            .unwrap();
        assert_eq!(plain.constant, 0.0);
        assert_eq!(shifted.constant, 17.25);
        assert_eq!(plain.terms, shifted.terms);

        let (a, b) = (plain.instance(), shifted.instance());
        for s in states(4) {
            assert!(
                (a.graph.energy(&s) - b.graph.energy(&s)).abs() < 1e-12,
                "the graphs must be identical: the constant is not in them"
            );
            assert!(
                (b.objective(&s) - a.objective(&s) - 17.25).abs() < 1e-12,
                "every reported value must move by exactly the constant"
            );
        }
    }

    /// THE ORACLE: [`crate::exact::Elimination`], which carries its own verification, finds the
    /// same optimum as enumeration — and [`Instance::objective`] reports it as the corpus's number.
    #[test]
    fn oracle_exact_elimination_finds_the_enumerated_optimum_of_a_parsed_instance() {
        let inst = read_biqmac(RUDY_TEXT).unwrap().instance();
        let ex = crate::exact::Elimination::default().ground_state(&inst.graph).unwrap();
        let state = ex.ground_state.expect("min-sum was run");
        let truth = states(5)
            .into_iter()
            .map(|s| inst.objective(&s))
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(
            (inst.objective(&state) - truth).abs() < 1e-9,
            "elimination's ground state cuts {} where the maximum is {truth}",
            inst.objective(&state)
        );
        assert!((ex.ground_energy.unwrap() - inst.graph.energy(&state)).abs() < 1e-9);
    }

    /// A bound on the energy becomes a bound on the objective IN THE SENSE'S OWN DIRECTION, and it
    /// must bracket the truth rather than merely be near it.
    ///
    /// Asymmetric on purpose: the maximisation's bound is asserted to sit ABOVE the enumerated
    /// optimum and the minimisation's BELOW it. An implementation that used one direction for both
    /// passes half of this and fails the other half.
    #[test]
    fn the_objective_bound_lands_on_the_senses_own_side_of_the_truth() {
        let cut = read_biqmac(RUDY_TEXT).unwrap().instance();
        let truth = states(5)
            .into_iter()
            .map(|s| cut.objective(&s))
            .fold(f64::NEG_INFINITY, f64::max);
        for b in [crate::bound::decoupled(&cut.graph), crate::bound::forest(&cut.graph, 40)] {
            let ub = cut.objective_bound(b.value);
            assert!(ub.is_finite(), "{}: an infinite bound says nothing", b.method);
            assert!(ub >= truth - 1e-12, "{}: upper bound {ub} sits BELOW the true max {truth}", b.method);
        }

        let mini = read_qplib(&QPLIB_TEXT.replace("Maximize", "Minimize")).unwrap().instance();
        let low = states(4)
            .into_iter()
            .map(|s| mini.objective(&s))
            .fold(f64::INFINITY, f64::min);
        for b in [crate::bound::decoupled(&mini.graph), crate::bound::forest(&mini.graph, 40)] {
            let lb = mini.objective_bound(b.value);
            assert!(lb.is_finite(), "{}: an infinite bound says nothing", b.method);
            assert!(lb <= low + 1e-12, "{}: lower bound {lb} sits ABOVE the true min {low}", b.method);
        }
    }

    /// The offset's bracket contains the exact constant even where plain addition loses it, and is
    /// exactly zero where the arithmetic is exact.
    ///
    /// `1e16 + 1 − 1e16` is one, and left-to-right `f64` addition gives zero; halved, the constant
    /// this max-cut instance carries is exactly `0.5`. A bound riding on an offset taken as exact
    /// would be narrower than the arithmetic supports, which is what [`crate::round`] exists for.
    #[test]
    fn the_offset_bracket_contains_the_exact_constant_under_cancellation() {
        let text = "3 3\n1 2 1e16\n2 3 1\n1 3 -1e16\n";
        let inst = read_biqmac(text).unwrap().instance();
        let naive: f64 = [5e15, 0.5, -5e15].iter().sum();
        assert_eq!(naive, 0.0, "the fixture must actually defeat plain addition, or it tests nothing");
        assert!(
            inst.offset - inst.offset_err <= 0.5 && 0.5 <= inst.offset + inst.offset_err,
            "the exact constant 0.5 is outside [{}, {}]",
            inst.offset - inst.offset_err,
            inst.offset + inst.offset_err
        );
        assert!(inst.offset_err < 1e-9, "and the bracket must stay useful: {}", inst.offset_err);

        // MEASURED TWICE, with opposite answers. When this test was written, `round`'s guard was a
        // bound on what the arithmetic COULD have lost rather than a detection of what it did
        // lose, so six integer half-weights summing to exactly 6.5 came back with `offset_err` =
        // 2.66e-15 -- sound, and looser than the arithmetic, and this test pinned that. Since
        // 2026-09-13 the guard is zero when no addition rounded (each step's error term is the
        // exact rounding error, and every one of them is exactly 0.0 here), so the same six
        // half-weights now come back with `offset_err` exactly 0.0. That is the tight answer, and
        // it is asserted with `==` so a guard that quietly becomes unconditional again fails here.
        let exact = read_biqmac(RUDY_TEXT).unwrap().instance();
        assert_eq!(exact.offset_err, 0.0, "nothing rounded, so the bracket must have no width");
        assert!(
            exact.offset - exact.offset_err <= 6.5 && 6.5 <= exact.offset + exact.offset_err,
            "W = 13, so the exact offset is 13/2 and must be inside [{}, {}]",
            exact.offset - exact.offset_err,
            exact.offset + exact.offset_err
        );
        // The empty sum IS exact, and that is the one case the bracket collapses in.
        let empty = MaxCut::new("e", 2, &[], Corpus::BiqMac).unwrap().instance();
        assert_eq!((empty.offset, empty.offset_err), (0.0, 0.0));
    }

    /// THE ASYMMETRIC ONE, on the writing side: a format with nowhere to put something refuses,
    /// and the same model without it is written happily.
    ///
    /// A writer that dropped the constant would produce a file that reads back as a different
    /// problem — the reader's failure committed one step earlier, where nothing is left to notice
    /// it. Both halves are asserted: the refusal AND the acceptance.
    #[test]
    fn a_format_that_cannot_hold_a_constant_or_a_sense_refuses_rather_than_dropping_it() {
        let plain = read_orlib(ORLIB_TEXT).unwrap().remove(0);
        assert!(write_mqlib_qubo(&plain).is_ok(), "nothing to lose here");
        assert!(write_orlib(std::slice::from_ref(&plain)).is_ok());

        let mut with_constant = plain.clone();
        with_constant.constant = 2.5;
        for e in [
            write_mqlib_qubo(&with_constant).unwrap_err(),
            write_orlib(std::slice::from_ref(&with_constant)).unwrap_err(),
        ] {
            assert!(matches!(e, CorporaError::Unrepresentable { .. }), "{e:?}");
            assert!(e.to_string().contains("2.5"), "the refusal must name the constant: {e}");
        }

        let mut minimising = plain.clone();
        minimising.sense = Sense::Minimise;
        assert!(matches!(
            write_mqlib_qubo(&minimising),
            Err(CorporaError::Unrepresentable { .. })
        ));
        // And QPLIB, which DOES carry both, writes them.
        let both = Qubo { name: "q".to_string(), constant: 2.5, ..minimising.clone() };
        let back = read_qplib(&write_qplib(&both).unwrap()).unwrap();
        assert_eq!((back.sense, back.constant), (Sense::Minimise, 2.5));

        // A name that would shift every section that follows it is refused, and a usable one is not.
        for bad in ["", "two words", "bang!"] {
            let q = Qubo { name: bad.to_string(), ..both.clone() };
            assert!(
                matches!(write_qplib(&q), Err(CorporaError::Unrepresentable { .. })),
                "QPLIB accepted the name {bad:?}"
            );
        }

        // The root-node transformation refuses the same two things, for the same reason.
        assert!(matches!(both.to_maxcut(), Err(CorporaError::Unrepresentable { .. })));
        assert!(matches!(minimising.to_maxcut(), Err(CorporaError::Unrepresentable { .. })));
    }

    /// The checked constructors refuse what the readers refuse, so a model built in Rust cannot
    /// reach an [`Instance`] by a route the parsers close off.
    #[test]
    fn the_checked_constructors_refuse_what_the_readers_refuse() {
        let c = Corpus::MqLib;
        assert!(matches!(
            MaxCut::new("x", 3, &[(1, 1, 2.0)], c),
            Err(CorporaError::SelfLoop { at: 2, .. })
        ));
        assert!(matches!(
            MaxCut::new("x", 3, &[(0, 1, 1.0), (1, 0, 2.0)], c),
            Err(CorporaError::Duplicate { pair: (1, 2), .. })
        ));
        assert!(matches!(
            MaxCut::new("x", 3, &[(0, 3, 1.0)], c),
            Err(CorporaError::Index { got: 4, n: 3, .. })
        ));
        assert!(matches!(
            MaxCut::new("x", 3, &[(0, 1, f64::NAN)], c),
            Err(CorporaError::Number { .. })
        ));
        assert!(matches!(
            Qubo::new("x", 2, &[(0, 0, 1.0), (0, 0, 2.0)], 0.0, Sense::Maximise, c),
            Err(CorporaError::Duplicate { pair: (1, 1), .. })
        ));
        assert!(matches!(
            Qubo::new("x", 2, &[], f64::INFINITY, Sense::Maximise, c),
            Err(CorporaError::Number { .. })
        ));
        // A diagonal is legal in a QUBO and not in a max-cut: the same entry, two meanings.
        assert!(Qubo::new("x", 2, &[(0, 0, 1.0)], 0.0, Sense::Maximise, c).is_ok());
    }

    /// Every error variant prints something that names the fault, and `Display` is not the derived
    /// `Debug`.
    #[test]
    fn every_refusal_prints_what_it_saw() {
        let cases: Vec<CorporaError> = vec![
            CorporaError::Header { format: "F", line: 1, text: "x".into() },
            CorporaError::Line { format: "F", line: 2, text: "y".into() },
            CorporaError::Number { format: "F", line: 3, got: "nan".into() },
            CorporaError::Index { format: "F", line: 4, got: 0, n: 9 },
            CorporaError::SelfLoop { format: "F", line: 5, at: 7 },
            CorporaError::Duplicate { format: "F", line: 6, first: 5, pair: (2, 3) },
            CorporaError::Count { format: "F", what: "edges", declared: 4, found: 3 },
            CorporaError::Truncated { format: "F", want: "a header" },
            CorporaError::Sense { line: 3, got: "Solve".into() },
            CorporaError::Unsupported { code: "QCN".into(), why: "continuous" },
            CorporaError::Unrepresentable { format: "F", what: "a constant".into() },
        ];
        for e in cases {
            let s = e.to_string();
            assert!(s.len() > 20, "{e:?} prints {s:?}");
            assert_ne!(s, format!("{e:?}"));
            let _: &dyn core::error::Error = &e;
        }
        assert!(
            CorporaError::Index { format: "F", line: 4, got: 0, n: 9 }
                .to_string()
                .contains("ONE-based"),
            "a zero index must say why zero is wrong"
        );
    }
}
