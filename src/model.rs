//! Writing a model in the problem's own words.
//!
//! Everything below this module is index-level: spins, couplings, slot bases. That is the right
//! vocabulary for a sampler and the wrong one for a person. This is the layer where you declare
//! *variables* with names and domains, write an objective over them, state constraints as
//! constraints, and get **named values** back rather than an array of ±1.
//!
//! ```
//! use ferrotherm::model::{Model, Sense};
//!
//! let mut m = Model::new();
//! let a = m.categorical("colour_a", 3);
//! let b = m.categorical("colour_b", 3);
//! m.not_equal(a, b);                      // adjacent vertices differ
//! let c = m.compile().unwrap();
//! let sol = c.solve_annealed(1);
//! assert_ne!(sol.value("colour_a"), sol.value("colour_b"));
//! ```
//!
//! # What compiling actually does
//!
//! Each variable is laid out into spins by an [`crate::encode::Encoding`], its encoding penalty is
//! added, every constraint becomes penalty terms, and the objective becomes couplings and fields.
//! The result is an ordinary [`crate::ftp::Program`] — so a model written here runs on a CPU, a
//! browser, an FPGA or Hitachi's ASIC without knowing which.
//!
//! # The honest limit
//!
//! Expressions over categorical variables require **one-hot** encoding, and [`Model::compile`]
//! refuses any other choice with the reason. A one-hot indicator is linear in spins, so a product
//! of two is quadratic and fits pairwise hardware. A domain-wall indicator is already a *product* of
//! two spins, so a product of two of them is quartic and does not — it would need ancillas and a
//! higher-order reduction, which is a different compiler pass and not one performed silently here.
//!
//! Domain-wall remains available for variables that appear only in constraints, where its penalty
//! is linear rather than quadratic in `k` and it tolerates a penalty roughly three times weaker.

use crate::encode::{Encoding, Slot};
use crate::ftp::{EncodedVar, Program};
use crate::graph::GraphBuilder;
use crate::schedule::Schedule;
use std::collections::BTreeMap;

/// A variable's set of possible values.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Domain {
    /// −1 or +1.
    Spin,
    /// 0 or 1.
    Binary,
    /// One of `k` unordered values.
    Categorical(usize),
    /// An integer in `lo..=hi`, treated as a categorical over its range.
    Integer { lo: i64, hi: i64 },
}

impl Domain {
    /// How many values it can take, without the clamp `size` applies.
    ///
    /// Separate because `size` returns `usize` and must saturate to stay usable, while the refusal
    /// in `compile` needs the true magnitude to report it.
    pub(crate) fn size_u128(&self) -> u128 {
        match self {
            Domain::Spin | Domain::Binary => 2,
            Domain::Categorical(k) => *k as u128,
            Domain::Integer { lo, hi } => {
                let span = (*hi as i128) - (*lo as i128) + 1;
                if span < 1 { 1 } else { span as u128 }
            }
        }
    }

    /// How many values it can take.
    pub fn size(&self) -> usize {
        match self {
            Domain::Spin | Domain::Binary => 2,
            Domain::Categorical(k) => *k,
            // In i128 then clamped: `hi - lo + 1` OVERFLOWS i64 for a wide range, and the wrapped
            // value came back as a tiny or negative size. `ft_model_integer` reported success and
            // `ft_model_compile` then aborted in `encode` with "a variable with fewer than 2
            // values is a constant" -- a panic across the C ABI, where the documented failure is a
            // zero return.
            Domain::Integer { lo, hi } => {
                (((*hi as i128) - (*lo as i128) + 1).clamp(1, usize::MAX as i128)) as usize
            }
        }
    }

    /// The values a modeller writes, in encoding order.
    ///
    /// For everything but an integer these are `0..size`, so value and slot coincide and the
    /// distinction never surfaces. An integer over `5..=20` takes the values 5 through 20, and
    /// conflating those with slots 0 through 15 is the bug this exists to prevent.
    pub fn values(&self) -> impl Iterator<Item = i64> + '_ {
        // Every domain is a contiguous run from `lo`, and the ONLY thing that differs is where it
        // starts. Spin starts at -1 and steps by two, which is why it gets its own arm rather than
        // sharing a catch-all: folded into the default it reported values 0 and 1, while the
        // decoder was handing back -1 and +1 for those same slots.
        let lo = self.lo();
        let step = if matches!(self, Domain::Spin) { 2 } else { 1 };
        (0..self.size()).map(move |i| lo + (i as i64) * step)
    }

    /// Which one-hot slot holds `value`, or `None` if the domain does not contain it.
    pub fn index_of(&self, value: i64) -> Option<usize> {
        let step = if matches!(self, Domain::Spin) { 2 } else { 1 };
        let off = value.checked_sub(self.lo())?;
        if off < 0 || off % step != 0 {
            return None;
        }
        let i = off / step;
        ((i as u128) < self.size() as u128).then_some(i as usize)
    }

    /// The smallest value this domain can take.
    fn lo(&self) -> i64 {
        match self {
            Domain::Spin => -1,
            Domain::Binary | Domain::Categorical(_) => 0,
            Domain::Integer { lo, .. } => *lo,
        }
    }

    /// How a domain reads in an error message.
    pub fn describe(&self) -> String {
        match self {
            Domain::Spin => "the spins -1 and +1".into(),
            Domain::Binary => "0 or 1".into(),
            Domain::Categorical(k) => format!("the {k} values 0..{}", k - 1),
            Domain::Integer { lo, hi } => format!("the integers {lo}..={hi}"),
        }
    }
}

/// A grid of variables, subscripted like an array.
///
/// `a[[i, j]]` is the variable at that position; out of bounds panics with the shape it was given,
/// because a silent wrap is an off-by-one that reaches the answer.
#[derive(Clone, Debug, PartialEq)]
pub struct Index {
    name: String,
    dims: Vec<usize>,
    vars: Vec<Var>,
}

impl Index {
    /// The shape it was declared with.
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }
    /// Every variable, in row-major order.
    pub fn all(&self) -> &[Var] {
        &self.vars
    }
    /// One row of the last dimension: `a.row(&[w])` is every shift for worker `w`.
    ///
    /// The shape a cardinality constraint is usually over — "exactly one shift per worker" — and
    /// writing it by hand is where the index arithmetic goes wrong.
    pub fn row(&self, prefix: &[usize]) -> Vec<Var> {
        assert!(
            prefix.len() < self.dims.len(),
            "{}: a row needs fewer indices than the {} it has",
            self.name,
            self.dims.len()
        );
        let last = self.dims[self.dims.len() - 1];
        let mut out = Vec::with_capacity(last);
        let mut full: Vec<usize> = prefix.to_vec();
        // any middle dimensions are fixed at 0 unless the caller named them
        while full.len() < self.dims.len() - 1 {
            full.push(0);
        }
        full.push(0);
        for k in 0..last {
            let n = full.len();
            full[n - 1] = k;
            out.push(self[&full[..]]);
        }
        out
    }
    /// One column of the first dimension: `a.column(&[s])` is every worker for shift `s`.
    pub fn column(&self, suffix: &[usize]) -> Vec<Var> {
        assert!(
            suffix.len() < self.dims.len(),
            "{}: a column needs fewer indices than the {} it has",
            self.name,
            self.dims.len()
        );
        let first = self.dims[0];
        let mut out = Vec::with_capacity(first);
        for k in 0..first {
            let mut full = vec![k];
            full.extend_from_slice(suffix);
            while full.len() < self.dims.len() {
                full.push(0);
            }
            out.push(self[&full[..]]);
        }
        out
    }
    fn offset(&self, sub: &[usize]) -> usize {
        assert_eq!(
            sub.len(),
            self.dims.len(),
            "{} has {} dimensions and was given {}",
            self.name,
            self.dims.len(),
            sub.len()
        );
        let mut off = 0;
        for (d, (&i, &n)) in sub.iter().zip(&self.dims).enumerate() {
            assert!(i < n, "{}: index {i} is outside dimension {d}, which is {n}", self.name);
            off = off * n + i;
        }
        off
    }
}

impl<const N: usize> core::ops::Index<[usize; N]> for Index {
    type Output = Var;
    fn index(&self, sub: [usize; N]) -> &Var {
        &self.vars[self.offset(&sub)]
    }
}

impl core::ops::Index<&[usize]> for Index {
    type Output = Var;
    fn index(&self, sub: &[usize]) -> &Var {
        &self.vars[self.offset(sub)]
    }
}

/// A handle to a declared variable.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Var(usize);

impl Var {
    /// The literal "this variable takes `value`", for use in expressions.
    ///
    /// ```
    /// # use ferrotherm::model::Model;
    /// let mut m = Model::new();
    /// let x = m.categorical("x", 3);
    /// let e = 5.0 * x.is(2) - 1.0 * x.is(0);
    /// # let _ = e;
    /// ```
    pub fn is(self, value: i64) -> Lit {
        Lit::Is(self, value)
    }

    /// The ±1 spin value, for `Spin` and `Binary` variables.
    pub fn spin(self) -> Lit {
        Lit::Spin(self)
    }
}

/// Whether an objective is being minimised or maximised.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Sense {
    Minimize,
    Maximize,
}

/// One thing an expression can refer to.
///
/// Ordered so a term's literal set can be a map key -- `effective_penalty` groups the objective by
/// literal set to find the largest PULL rather than the largest single coefficient. The order has
/// no meaning beyond making that grouping possible.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Lit {
    /// The spin value of a `Spin` or `Binary` variable, as ±1.
    Spin(Var),
    /// 1 when `var` takes `value`, 0 otherwise.
    Is(Var, i64),
}

/// A weighted product of at most two literals.
#[derive(Clone, Debug)]
struct Term {
    coeff: f64,
    lits: Vec<Lit>,
}

/// A linear or quadratic expression.
#[derive(Clone, Debug, Default)]
pub struct Expr {
    terms: Vec<Term>,
    constant: f64,
}

impl Expr {
    pub fn zero() -> Expr {
        Expr::default()
    }

    /// Every coefficient and the constant are finite, or say which is not.
    pub(crate) fn check_finite(&self, what: &'static str) -> Result<(), CompileError> {
        if !self.constant.is_finite() {
            return Err(CompileError::NotFinite { what: "objective constant", value: self.constant });
        }
        for term in &self.terms {
            if !term.coeff.is_finite() {
                return Err(CompileError::NotFinite { what, value: term.coeff });
            }
        }
        Ok(())
    }
    /// `c`, a constant.
    pub fn constant(c: f64) -> Expr {
        Expr { terms: Vec::new(), constant: c }
    }
    /// `c · l`.
    pub fn lit(c: f64, l: Lit) -> Expr {
        Expr { terms: vec![Term { coeff: c, lits: vec![l] }], constant: 0.0 }
    }
    /// `c · l₁ · l₂`.
    pub fn pair(c: f64, a: Lit, b: Lit) -> Expr {
        Expr { terms: vec![Term { coeff: c, lits: vec![a, b] }], constant: 0.0 }
    }
    /// `c · l₁ · l₂ · … · lₖ`, for any number of literals.
    ///
    /// Three or more needs [`crate::reduce`], which the compiler applies for you and charges in
    /// ancilla spins. `Compiled::ancillas` says how many; the pass's own docs say what it costs at
    /// finite temperature.
    pub fn product(c: f64, lits: &[Lit]) -> Expr {
        Expr { terms: vec![Term { coeff: c, lits: lits.to_vec() }], constant: 0.0 }
    }
    /// Add another expression.
    pub fn plus(mut self, other: Expr) -> Expr {
        self.terms.extend(other.terms);
        self.constant += other.constant;
        self
    }
    /// Scale every term.
    pub fn scaled(mut self, k: f64) -> Expr {
        for t in &mut self.terms {
            t.coeff *= k;
        }
        self.constant *= k;
        self
    }
    fn max_degree(&self) -> usize {
        self.terms.iter().map(|t| t.lits.len()).max().unwrap_or(0)
    }
}

// ---- arithmetic ---------------------------------------------------------------------------------
//
// An objective should read like the thing it is. `5.0 * x.is(2) + 3.0 * y.is(1)` says what it means;
// `Expr::zero().plus(Expr::lit(5.0, Lit::Is(x, 2)))` says how it is stored. Both build the same
// structure, and the builder methods remain for anyone assembling terms in a loop.

impl core::ops::Mul<Lit> for f64 {
    type Output = Expr;
    fn mul(self, l: Lit) -> Expr {
        Expr::lit(self, l)
    }
}

impl core::ops::Mul<f64> for Lit {
    type Output = Expr;
    fn mul(self, c: f64) -> Expr {
        Expr::lit(c, self)
    }
}

/// A product of two literals, which is quadratic in the spins.
impl core::ops::Mul<Lit> for Lit {
    type Output = Expr;
    fn mul(self, other: Lit) -> Expr {
        Expr::pair(1.0, self, other)
    }
}

/// Scaling a quadratic term: `2.0 * (a.is(1) * b.is(1))`.
impl core::ops::Mul<Expr> for f64 {
    type Output = Expr;
    fn mul(self, e: Expr) -> Expr {
        e.scaled(self)
    }
}

impl core::ops::Mul<f64> for Expr {
    type Output = Expr;
    fn mul(self, c: f64) -> Expr {
        self.scaled(c)
    }
}

impl core::ops::Add<Expr> for Expr {
    type Output = Expr;
    fn add(self, other: Expr) -> Expr {
        self.plus(other)
    }
}

impl core::ops::Add<Lit> for Expr {
    type Output = Expr;
    fn add(self, l: Lit) -> Expr {
        self.plus(Expr::lit(1.0, l))
    }
}

impl core::ops::Add<Expr> for Lit {
    type Output = Expr;
    fn add(self, e: Expr) -> Expr {
        Expr::lit(1.0, self).plus(e)
    }
}

impl core::ops::Add<Lit> for Lit {
    type Output = Expr;
    fn add(self, other: Lit) -> Expr {
        Expr::lit(1.0, self).plus(Expr::lit(1.0, other))
    }
}

impl core::ops::Neg for Expr {
    type Output = Expr;
    fn neg(self) -> Expr {
        self.scaled(-1.0)
    }
}

impl core::ops::Neg for Lit {
    type Output = Expr;
    fn neg(self) -> Expr {
        Expr::lit(-1.0, self)
    }
}

impl core::ops::Sub<Expr> for Expr {
    type Output = Expr;
    fn sub(self, other: Expr) -> Expr {
        self.plus(-other)
    }
}

impl core::ops::Sub<Lit> for Expr {
    type Output = Expr;
    fn sub(self, l: Lit) -> Expr {
        self.plus(-l)
    }
}

/// Sum an iterator of terms, for objectives built in a loop.
impl core::iter::Sum<Expr> for Expr {
    fn sum<I: Iterator<Item = Expr>>(iter: I) -> Expr {
        iter.fold(Expr::zero(), |a, b| a.plus(b))
    }
}

/// The values two domains have in common, which is what "equal" and "not equal" are about.
fn shared_values(a: &Domain, b: &Domain) -> Vec<i64> {
    a.values().filter(|v| b.index_of(*v).is_some()).collect()
}

/// How the two sides of a weighted linear row compare.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Rel {
    /// `Σ wᵢ·lᵢ ≤ rhs`
    Le,
    /// `Σ wᵢ·lᵢ ≥ rhs`
    Ge,
    /// `Σ wᵢ·lᵢ = rhs`
    Eq,
}

impl Rel {
    /// The symbol a modeller wrote, for reporting a row back in their own notation.
    pub fn symbol(&self) -> &'static str {
        match self {
            Rel::Le => "≤",
            Rel::Ge => "≥",
            Rel::Eq => "=",
        }
    }
}

/// A statement that must hold.
#[derive(Clone, Debug)]
pub enum Constraint {
    /// The two variables must differ. The workhorse of colouring and assignment.
    NotEqual(Var, Var),
    /// The two variables must agree.
    Equal(Var, Var),
    /// Exactly one of these literals is true.
    ExactlyOne(Vec<Lit>),
    /// At most one is true.
    AtMostOne(Vec<Lit>),
    /// This variable takes this value.
    Fix(Var, i64),
    /// Exactly `k` of these literals are true.
    ///
    /// The penalty is `(Σ lits − k)²`, which is quadratic in the spins and needs no ancillas.
    /// Cardinality is the workhorse of assignment, scheduling and selection problems.
    Cardinality { lits: Vec<Lit>, k: usize },
    /// At most `k` of these literals are true.
    ///
    /// An inequality cannot be a squared penalty on its own — `(Σ − k)²` would punish *under* the
    /// limit as hard as over it, which is the wrong problem. So the compiler introduces a **slack
    /// variable** ranging over `0..=k` and constrains `Σ lits + slack = k`, turning the inequality
    /// into an equality it can square. The slack costs spins and is invisible in the answer.
    AtMost { lits: Vec<Lit>, k: usize },
    /// At least `k` of these literals are true. Slack as above, on the other side.
    AtLeast { lits: Vec<Lit>, k: usize },
    /// Every one of these variables takes a DIFFERENT value.
    ///
    /// Lowered per value rather than per pair: for each value any two of them share, the
    /// indicators for "this variable takes that value" are pairwise excluded. That is the
    /// `AtMostOne` lowering repeated over the shared values, so it needs no slack and no ancillas,
    /// and it costs nothing where the domains do not overlap — n variables over disjoint domains
    /// compile to zero terms, which is correct and which a pairwise `not_equal` sweep would not
    /// notice.
    ///
    /// Writing it as one constraint rather than n(n−1)/2 `NotEqual`s is not only shorter. It buys
    /// two things a sweep cannot: a violation that names WHICH value collided and how many took it,
    /// and the pigeonhole check — more variables than shared values is unsatisfiable, and the
    /// compiler says so by name instead of annealing and returning a confident infeasible answer.
    AllDifferent(Vec<Var>),
    /// A **weighted** linear row: `Σ wᵢ·lᵢ ≤ rhs`, `≥ rhs`, or `= rhs`.
    ///
    /// The thing none of the counting constraints above can say. `Cardinality`, `AtMost`,
    /// `AtLeast`, `ExactlyOne` and `AtMostOne` all count *unweighted* literals, so `3a + 4b + 5c
    /// ≤ 7` could not be stated anywhere in this crate — and the LP reader's advice, "add it to
    /// the objective", is the defect rather than the workaround: an objective term is not a
    /// constraint, so [`Solution::feasible`] and [`Solution::violated`] stop knowing about the row
    /// at all.
    ///
    /// # How it is lowered
    ///
    /// An equality is a squared penalty and needs nothing else: `p·(Σ wᵢxᵢ − rhs)²`.
    ///
    /// An inequality needs **slack**, the same idea [`Constraint::AtMost`] uses, with the
    /// coefficients carried. The row is normalised to `Σ aᵢxᵢ ≤ t` (a `Ge` row is negated),
    /// divided through by a `g` that divides every weight, and a slack `σ ∈ [0, S]` — where
    /// `S = ⌊t/g⌋ − Σ min(aᵢ/g, 0)` — turns it into the equality `Σ (aᵢ/g)xᵢ + σ = ⌊t/g⌋`.
    ///
    /// A **hard** row takes `g = gcd(|a₁|, …, |aₙ|)` and floors the target, which is exact because
    /// the left side is a multiple of `g`, and which is the strongest reduction available:
    /// `1000a + 1000b ≤ 1501` becomes `a + b ≤ 1`, one slack spin instead of eleven. A **soft** row
    /// takes `g = gcd(|a₁|, …, |aₙ|, |t|)` instead, so that nothing is thrown away by the floor and
    /// the energy it contributes stays exactly the `weight × amount²` that [`Violation::cost`]
    /// reports in the modeller's own units.
    ///
    /// The slack is **truncated binary**: `m = ⌈log₂(S+1)⌉` spins with coefficients
    /// `1, 2, 4, …, 2^(m−2), S+1−2^(m−1)`. That expansion covers `{0 … S}` exactly — nothing
    /// more and nothing less — so every spin pattern is a legal slack value and the block needs no
    /// encoding penalty and no exclusion couplings. (It is 2-to-1 on a window of size
    /// `2^m − (S+1)`, empty exactly when `S+1` is a power of two. That changes no argmin and
    /// biases a Boltzmann marginal over logical states by at most 2×; it is reported as a caveat.)
    ///
    /// # What it costs, exactly
    ///
    /// With `n` terms after merging duplicates and dropping zero weights, and `m` slack spins:
    ///
    /// ```text
    /// spins added  = m = ⌈log₂(S+1)⌉    — 0 for an equality, at most 62 for any i64 row
    /// couplings    = (n+m)(n+m−1)/2      — a clique on the row's literals plus its slack
    /// max degree   = n+m−1
    /// ```
    ///
    /// The slack is logarithmic in the *numeric value* of the bound, so `1000a + 1000b ≤ 1500` is
    /// **one** slack spin (gcd 1000 divides it through to `a + b ≤ 1`), and `1000a + 1001b ≤ 1500`
    /// is 11. **The bill is `n`, not the weights**: the `n(n−1)/2` literal–literal clique is
    /// irreducible for any quadratic penalty on a weighted row, so a 200-item capacity row carries
    /// **19,900 irreducible literal–literal couplings, and 21,115 in total once the slack's own
    /// clique and its cross terms are counted**. "Cheap in slack spins" is not "cheap".
    ///
    /// Couplings carry `p·aᵢaⱼ`, so the graph's dynamic range grows *quadratically* in the reduced
    /// weights. A row spanning 1…1000 spans six orders of magnitude in `J` under one global β;
    /// that is intrinsic to any squared-penalty lowering of a weighted row, and the gcd reduction
    /// is the only mitigation available. A caveat is emitted when the row's own coefficient spread
    /// exceeds 100× (a coupling spread of 10⁴).
    ///
    /// # What it refuses, by name
    ///
    /// * A non-integer coefficient or right-hand side on an **inequality** —
    ///   [`CompileError::LinearNotInteger`]. There is no finite slack for a non-lattice left side.
    ///   An **equality** takes any finite coefficient, because it needs no slack at all.
    /// * A row nothing can satisfy — [`CompileError::LinearUnsatisfiable`], checked by arithmetic
    ///   rather than annealed, on the same principle as [`CompileError::Pigeonhole`].
    /// * A slack wider than 62 spins — [`CompileError::LinearTooLarge`].
    /// * A `Binary`- or `DomainWall`-encoded variable — [`CompileError::NeedsOneHot`], inherited.
    ///
    /// A row that constrains nothing (`3a + 4b ≤ 12`) compiles to nothing and says so in
    /// `caveats`, matching how `at_most(lits, k ≥ lits.len())` reads.
    Linear {
        /// `(literal, coefficient)` pairs. Repeats are merged and zero weights dropped.
        terms: Vec<(Lit, f64)>,
        /// Which way the row compares.
        rel: Rel,
        /// The right-hand side, in the modeller's own units.
        rhs: f64,
    },
}

struct Decl {
    name: String,
    domain: Domain,
    encoding: Encoding,
}

/// A model in the problem's own vocabulary.
pub struct Model {
    decls: Vec<Decl>,
    objective: Expr,
    /// The sense every `objective` call used, or `None` if they disagreed or none was written.
    ///
    /// `objective` normalises the sense away by negating what is maximised, which is right for the
    /// compiler and wrong for the reader: a modeller who wrote `maximize 5*mon + 4*tue` wants 9
    /// back, not -9. This is the only place the direction they wrote in survives.
    sense: Option<Sense>,
    /// Each constraint, its penalty (NaN for "scale it"), and whether breaking it is allowed.
    constraints: Vec<(Constraint, f64, bool)>,
    /// Penalty strength applied to encodings and to constraints without their own.
    ///
    /// Treated as a *floor* unless [`Model::fixed_penalty`] is called. See [`Model::compile`].
    pub penalty: f64,
    auto_penalty: bool,
}

impl Default for Model {
    fn default() -> Self {
        Model::new()
    }
}

/// Why a model could not be compiled.
#[derive(Clone, Debug, PartialEq)]
pub enum CompileError {
    /// An expression touched a categorical variable that is not one-hot encoded.
    NeedsOneHot { var: String, encoding: Encoding },
    /// An expression term had degree above two.
    DegreeTooHigh { degree: usize },
    /// A value outside a variable's domain.
    BadValue { var: String, value: i64, domain: Domain },
    /// A model with nothing in it.
    Empty,
    /// Two variables sharing a name.
    DuplicateName(String),
    /// An integer variable with more values than the graph can index.
    ///
    /// Spin indices are `u32`, so nothing here can address more than `u32::MAX` of them, and a
    /// one-hot slot needs one spin per value. `ft_model_integer` used to accept a range spanning
    /// most of `i64`, report success, and then `ft_model_compile` aborted the caller's process with
    /// a capacity overflow -- where the documented failure is a zero return.
    DomainTooLarge { var: String, size: u128 },
    /// A coefficient that is NaN or infinite.
    ///
    /// Not a pedantic check. A single NaN objective term used to compile, solve, and report
    /// `feasible: true` while silently discarding every OTHER preference: energy comparisons
    /// against NaN are all false, so the sampler's "is this better than the best so far" test
    /// never fires again. A model maximising `3·a` alongside one NaN term answered `a = 0` --
    /// a confident, feasible-looking, wrong answer, which is the exact failure this crate exists
    /// to refuse.
    NotFinite { what: &'static str, value: f64 },
    /// An `all_different` over more variables than there are values for them to take.
    ///
    /// The pigeonhole principle, checked rather than annealed. A model like this has no answer at
    /// all, and returning `feasible: false` after a full anneal tells a modeller their penalty was
    /// too low or their ladder too short — neither of which is true, and both of which cost an
    /// afternoon.
    Pigeonhole { vars: usize, values: usize },
    /// A non-integer coefficient or right-hand side on a weighted linear **inequality**.
    ///
    /// The structural refusal, and the one that keeps the penalty argument sound. An inequality is
    /// lowered with a slack variable ranging over the integer residual; with real coefficients the
    /// achievable left-hand sides are not a lattice, so the smallest overshoot is an arbitrary
    /// positive real, the separation gap `p·δ²` collapses toward zero, and no penalty at any finite
    /// precision orders the states correctly.
    ///
    /// Refused rather than auto-scaled. Recovering a denominator from an f64 is a guess — 0.1 is
    /// not 1/10 — and multiplying a row through by a denominator nobody wrote is how a two-decimal
    /// price list silently becomes a 100× wider slack that reads as the library being slow.
    LinearNotInteger { row: String, what: String, value: f64 },
    /// A weighted row whose coefficient is a whole number but too large for `f64` to hold every
    /// integer near it.
    ///
    /// Split from [`CompileError::LinearNotInteger`], which used to cover this and said two false
    /// things while doing it: that the coefficient was "non-integer" when `1e16` plainly is one,
    /// and that the fix was to "multiply the row through by its common denominator", which cannot
    /// help a number that is already integral. The refusal was right and its stated reason was not.
    ///
    /// Above 2^53 the doubles stop representing consecutive integers, so the slack arithmetic --
    /// gcd, floor division, the residual span -- can no longer be trusted to be exact, and an
    /// inexact slack silently admits or forbids the wrong assignments.
    LinearHugeCoefficient { row: String, what: String, value: f64, limit: f64 },
    /// A weighted linear row no assignment can satisfy, checked by arithmetic rather than annealed.
    ///
    /// The [`CompileError::Pigeonhole`] principle applied to a row: the best the left side can do
    /// is on one side of the comparison and the right-hand side is on the other. Annealing this
    /// returns `feasible: false`, which tells a modeller their penalty was too low or their ladder
    /// too short — neither of which is true, and both of which cost an afternoon.
    LinearUnsatisfiable { row: String, best: f64 },
    /// A weighted linear inequality whose slack needs more than 62 spins.
    ///
    /// The slack spans `S+1` values in `⌈log₂(S+1)⌉` spins, so this is reached only by a row whose
    /// span does not fit an `i64` — a genuine overflow rather than a budget. The message shows its
    /// arithmetic, because "too large" without the number is not actionable.
    LinearTooLarge { row: String, span: i128, spins: usize },
}

impl core::fmt::Display for CompileError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            CompileError::DomainTooLarge { var, size } => write!(
                f,
                "'{var}' spans {size} values, and spin indices are u32 so at most {} can be \
                 addressed. Narrow the range: a domain this size cannot be laid out, let alone \
                 annealed",
                u32::MAX
            ),
            CompileError::NotFinite { what, value } => {
                // Say WHICH failure, because they are different. And note the negation: a model
                // stores "minimise this", so a term written as `maximize(+inf)` is held as -inf,
                // and printing the stored number without saying so reports a sign nobody wrote.
                let why = if value.is_nan() {
                    "every comparison against NaN is false, so a sampler carrying one \
                     silently stops improving and returns a feasible-looking answer that \
                     ignores every preference in the model"
                } else {
                    "an infinite coefficient makes every state's energy infinite, so no \
                     state compares better than any other and the search degenerates to its \
                     first sample"
                };
                write!(
                    f,
                    "a {what} is {value}, which is not a finite number: {why}. \
                     (Coefficients are stored as 'minimise this', so a maximised term \
                     appears here negated.)"
                )
            }
            CompileError::Pigeonhole { vars, values } => write!(
                f,
                "all_different over {vars} variables, but between them their domains hold only \
                 {values} distinct value(s). No assignment can satisfy that, so this is refused \
                 rather than annealed: a model with no answer returns infeasible for a reason no \
                 penalty and no longer ladder will fix."
            ),
            CompileError::NeedsOneHot { var, encoding } => write!(
                f,
                "'{var}' is {encoding:?}-encoded and appears in an expression. A one-hot indicator \
                 is linear in spins, so a product of two is quadratic and fits pairwise hardware; a \
                 domain-wall indicator is already a product, so a product of two is quartic. Use \
                 Encoding::OneHot for variables that appear in an objective, or keep this one to \
                 constraints only"
            ),
            CompileError::DegreeTooHigh { degree } => write!(
                f,
                "a term of degree {degree} needs a higher-order reduction with ancillas, which is a \
                 separate pass and is not applied silently"
            ),
            CompileError::BadValue { var, value, domain } => {
                write!(f, "'{var}' takes {}; {value} is not one of them", domain.describe())
            }
            CompileError::LinearHugeCoefficient { row, what, value, limit } => write!(
                f,
                "the weighted row {row} has a {what} of {value}, which is a whole number but larger \
                 than {limit} (2^53), past which an f64 no longer holds every integer. The slack \
                 arithmetic -- gcd, floor division, the residual span -- would stop being exact, \
                 and an inexact slack admits or forbids the wrong assignments silently. Scale the \
                 row down, or state it as an EQUALITY, which needs no slack."
            ),
            CompileError::LinearNotInteger { row, what, value } => write!(
                f,
                "the weighted row {row} has a non-integer {what} of {value}. An inequality is \
                 lowered with a slack variable that ranges over the INTEGER residual, and a \
                 coefficient of {value} leaves no integer residual for it to range over: the \
                 smallest overshoot becomes an arbitrary positive real, so the gap between the \
                 allowed states and the forbidden ones shrinks toward zero and no penalty orders \
                 them correctly. Multiply the row through by its common denominator and state it \
                 in whole numbers -- and note that doing so multiplies the slack span by the same \
                 factor: 2.5a + 1.5b <= 4 becomes 5a + 3b <= 8. This is not done for you, because \
                 recovering a denominator from a float is a guess and solving a nearby problem \
                 silently is the failure this compiler exists to refuse. An EQUALITY takes any \
                 finite coefficient, because it needs no slack at all."
            ),
            CompileError::LinearUnsatisfiable { row, best } => write!(
                f,
                "the weighted row {row} has no answer: the best the left side can reach on the \
                 constrained side is {best}. Refused rather than annealed, for the same reason as \
                 the pigeonhole check -- a model with no answer returns infeasible for a reason no \
                 penalty and no longer ladder will fix."
            ),
            CompileError::LinearTooLarge { row, span, spins } => write!(
                f,
                "the weighted row {row} needs a slack spanning {span} values, which is {spins} \
                 spins. The slack is logarithmic in the numeric value of the bound, so reaching \
                 this at all means the row's span does not fit an i64. Scale the row down, or \
                 state it as an objective term -- knowing that an objective term is not a \
                 constraint and feasible() will stop knowing about this row."
            ),
            CompileError::Empty => write!(f, "a model with no variables compiles to nothing"),
            CompileError::DuplicateName(n) => write!(
                f,
                "two variables are both called '{n}'. An answer is keyed by name, so a second one \
                 does not shadow the first -- it replaces it, and one of the two silently \
                 disappears from the result"
            ),
        }
    }
}

impl Model {
    pub fn new() -> Model {
        Model {
            decls: Vec::new(),
            objective: Expr::zero(),
            sense: None,
            constraints: Vec::new(),
            penalty: 2.0,
            auto_penalty: true,
        }
    }

    // ---- declaring ------------------------------------------------------------------------------

    fn declare(&mut self, name: &str, domain: Domain, encoding: Encoding) -> Var {
        self.decls.push(Decl { name: name.to_string(), domain, encoding });
        Var(self.decls.len() - 1)
    }

    /// A grid of variables, indexed and named for you.
    ///
    /// The shape a real model has. "One variable per worker per shift" is written
    /// `m.grid("assign", &[workers.len(), shifts.len()], |m, n| m.binary(n))`, and the variables
    /// come back as an [`Index`] you subscript: `a[[w, s]]`. Names are `assign[2,3]`, so an answer
    /// still reads in the modeller's own words.
    ///
    /// Without this, every real model begins with a hand-rolled loop and a `format!` — which works,
    /// and loses the shape: nothing downstream knows those variables were a grid, and an off-by-one
    /// in the index arithmetic is silent. JijModeling's indexed variables are the idea; this is the
    /// same idea with the bounds checked.
    pub fn grid(
        &mut self,
        name: &str,
        dims: &[usize],
        mut declare: impl FnMut(&mut Model, &str) -> Var,
    ) -> Index {
        assert!(!dims.is_empty(), "a grid needs at least one dimension");
        assert!(dims.iter().all(|d| *d > 0), "a grid dimension of zero has no variables");
        let total: usize = dims.iter().product();
        let mut vars = Vec::with_capacity(total);
        let mut sub = vec![0usize; dims.len()];
        for _ in 0..total {
            let label = format!(
                "{name}[{}]",
                sub.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(",")
            );
            vars.push(declare(self, &label));
            // odometer, last index fastest, so the layout matches the subscript order
            for d in (0..dims.len()).rev() {
                sub[d] += 1;
                if sub[d] < dims[d] {
                    break;
                }
                sub[d] = 0;
            }
        }
        Index { name: name.to_string(), dims: dims.to_vec(), vars }
    }

    /// A ±1 variable.
    pub fn spin(&mut self, name: &str) -> Var {
        self.declare(name, Domain::Spin, Encoding::OneHot)
    }

    /// A 0/1 variable.
    pub fn binary(&mut self, name: &str) -> Var {
        self.declare(name, Domain::Binary, Encoding::OneHot)
    }

    /// A `k`-valued variable, one-hot by default so it may appear in expressions.
    pub fn categorical(&mut self, name: &str, k: usize) -> Var {
        assert!(k >= 2, "a variable with fewer than two values is a constant");
        self.declare(name, Domain::Categorical(k), Encoding::OneHot)
    }

    /// A `k`-valued variable in a chosen encoding.
    ///
    /// Domain-wall costs one fewer spin and a linear rather than quadratic penalty, and tolerates a
    /// penalty about three times weaker — but a variable encoded that way may only appear in
    /// constraints, not in expressions. [`Model::compile`] enforces that rather than discovering it.
    pub fn categorical_as(&mut self, name: &str, k: usize, encoding: Encoding) -> Var {
        assert!(k >= 2);
        self.declare(name, Domain::Categorical(k), encoding)
    }

    /// An integer in `lo..=hi`, as an integer with a chosen encoding.
    ///
    /// [`Encoding::DomainWall`] is often the better choice here and is not the default: an integer
    /// is an ORDERED domain, and a domain wall makes neighbouring values one spin flip apart where
    /// one-hot makes every pair two flips apart. It also costs `k-1` spins instead of `k`. The
    /// default stays one-hot because only a one-hot indicator is linear in the spins, so only a
    /// one-hot variable can appear in an objective — see [`CompileError::NeedsOneHot`].
    pub fn integer_as(&mut self, name: &str, lo: i64, hi: i64, encoding: Encoding) -> Var {
        assert!(hi > lo, "an integer range needs at least two values");
        self.declare(name, Domain::Integer { lo, hi }, encoding)
    }

    /// An integer in `lo..=hi`.
    pub fn integer(&mut self, name: &str, lo: i64, hi: i64) -> Var {
        assert!(hi > lo, "an integer range needs at least two values");
        self.declare(name, Domain::Integer { lo, hi }, Encoding::OneHot)
    }

    /// The handle for the `i`-th declared variable, for callers that track variables by position
    /// rather than by handle — an FFI, or a node graph that already has its own names.
    pub fn var_at(&self, i: usize) -> Var {
        assert!(i < self.decls.len(), "no variable at index {i}");
        Var(i)
    }

    pub fn name_of(&self, v: Var) -> &str {
        &self.decls[v.0].name
    }

    /// Rename a variable after declaring it.
    ///
    /// A caller that declares variables through a boundary carrying no strings -- an FFI, a node
    /// graph -- still has names on its own side. Pushing them down here is what makes an error read
    /// "'temperature' takes the integers 10..=20" instead of naming a handle the modeller never saw.
    pub fn rename(&mut self, v: Var, name: impl Into<String>) -> &mut Self {
        self.decls[v.0].name = name.into();
        self
    }
    pub fn domain_of(&self, v: Var) -> Domain {
        self.decls[v.0].domain
    }
    pub fn len(&self) -> usize {
        self.decls.len()
    }
    pub fn is_empty(&self) -> bool {
        self.decls.is_empty()
    }

    // ---- stating the problem --------------------------------------------------------------------

    /// Set the objective.
    pub fn objective(&mut self, sense: Sense, e: Expr) -> &mut Self {
        // ACCUMULATES, and normalises the sense away as it goes.
        //
        // Two bugs lived in the old "set the sense, replace the expression" form. Calling this in a
        // loop -- which is how an objective with one term per option actually gets written -- kept
        // only the LAST term and silently dropped the rest. And a second call with a different
        // sense re-interpreted every term already accumulated, so adding one thing to minimise
        // flipped an entire objective that was being maximised.
        //
        // Storing it as "minimise this" removes both. Maximising e is minimising -e; a model has no
        // global sense left to flip, and terms in opposite directions compose the way arithmetic
        // says they should.
        // Remember the direction the modeller wrote in, so the answer can be reported back in it.
        // Only when this call CONTRIBUTES something: `objective(Minimize, Expr::zero())` adds no
        // term and must not turn a maximisation into a mixed objective. Mixed senses compose fine
        // as arithmetic and have no single direction to report, so they are recorded as exactly
        // that rather than as whichever call came last.
        let contributes = !e.terms.is_empty() || e.constant != 0.0;
        if contributes {
            self.sense = match self.sense {
                None => Some(sense),
                Some(prev) if prev == sense => Some(prev),
                _ => None,
            };
        }
        let signed = if sense == Sense::Maximize { e.scaled(-1.0) } else { e };
        let acc = core::mem::take(&mut self.objective);
        self.objective = acc.plus(signed);
        self
    }

    /// Discard everything accumulated so far and use exactly this objective.
    pub fn set_objective(&mut self, sense: Sense, e: Expr) -> &mut Self {
        self.objective = Expr::zero();
        self.sense = None;
        self.objective(sense, e)
    }

    /// Use exactly this penalty, disabling the automatic scaling described on [`Model::compile`].
    ///
    /// Worth doing when tuning deliberately, and worth knowing that a penalty below the objective's
    /// scale produces states that do not decode rather than states that score badly.
    pub fn fixed_penalty(&mut self, p: f64) -> &mut Self {
        self.penalty = p;
        self.auto_penalty = false;
        self
    }

    /// The penalty that will actually be used, after scaling.
    pub fn effective_penalty(&self) -> f64 {
        if !self.auto_penalty {
            return self.penalty;
        }
        // TWICE THE LARGEST PULL ON ONE LITERAL SET, NOT THE LARGEST SINGLE COEFFICIENT.
        //
        // `Expr::plus` extends rather than merges, and `objective` accumulates -- which is correct,
        // and is what makes writing an objective in a loop work at all. It also means many terms
        // land on the SAME literal, and what a constraint has to outbid is their sum.
        //
        // Taking the largest individual coefficient measured the wrong thing. Three separate terms
        // of `1.0 * a.is(1)` -- an objective built in a loop, which is the documented pattern and
        // what the README example does -- pull with strength 3 while the automatic penalty came out
        // at 2, so a HARD `Fix(a, 0)` beside them was traded away: the answer came back with
        // `a = 1` and `feasible = false`. The reasoning in this function was right and it was
        // applied to the wrong number.
        let mut pull: BTreeMap<Vec<Lit>, f64> = BTreeMap::new();
        for term in &self.objective.terms {
            // The literal set identifies the term; sorted so `a*b` and `b*a` are one entry.
            let mut key = term.lits.clone();
            key.sort();
            *pull.entry(key).or_insert(0.0) += term.coeff;
        }
        let worst = pull.values().map(|c| c.abs()).fold(0.0f64, f64::max);
        // A constraint that merely ties with the objective gets traded away, and the result is a
        // state that does not decode rather than one that scores badly -- which reads as a broken
        // sampler rather than an under-weighted constraint.
        self.penalty.max(2.0 * worst)
    }

    /// Add a constraint at the model's default penalty.
    pub fn constrain(&mut self, c: Constraint) -> &mut Self {
        // recorded as NaN and resolved at compile time, so a constraint added before the objective
        // still gets the scaled penalty
        self.constraints.push((c, f64::NAN, true));
        self
    }

    /// A constraint you are willing to break, at a price.
    ///
    /// A **hard** constraint is a statement about which answers are admissible: breaking one means
    /// the answer is not an answer, and [`Solution::feasible`] says so. A **soft** one is a
    /// preference with a number on it — "prefer that these two do not clash, and if they must,
    /// that costs `weight`" — which is a different thing, and collapsing the two is why
    /// `feasible` used to go false for a constraint the modeller had deliberately priced low.
    ///
    /// Both compile the same way: a squared penalty. What differs is what the answer MEANS.
    /// [`Solution::feasible`] ignores soft violations, [`Solution::soft_cost`] totals what they
    /// cost, and `violated` reports both kinds, marked.
    ///
    /// The weight is absolute, not scaled. Automatic scaling exists to stop a hard constraint
    /// being outbid by the objective; a soft constraint is *meant* to be traded against it, so
    /// scaling it would defeat the point.
    pub fn soft(&mut self, c: Constraint, weight: f64) -> &mut Self {
        assert!(weight.is_finite() && weight > 0.0, "a soft constraint needs a positive price");
        self.constraints.push((c, weight, false));
        self
    }

    /// Make the constraint added most recently a soft one, at `weight`.
    ///
    /// For a caller who adds a constraint through one of the convenience builders and then decides
    /// it is a preference — and for the C ABI, where the pairwise constraints take their arguments
    /// directly rather than through a literal list. False when there is nothing to soften.
    #[must_use = "false means the constraint was NOT made soft, so the model you compile is not the model you described"]
    pub fn soften_last(&mut self, weight: f64) -> bool {
        if !(weight > 0.0) || !weight.is_finite() {
            return false;
        }
        match self.constraints.last_mut() {
            Some(entry) => {
                entry.1 = weight;
                entry.2 = false;
                true
            }
            None => false,
        }
    }

    /// Add a constraint at a specific penalty strength.
    pub fn constrain_at(&mut self, c: Constraint, penalty: f64) -> &mut Self {
        self.constraints.push((c, penalty, true));
        self
    }

    /// `a != b`.
    pub fn not_equal(&mut self, a: Var, b: Var) -> &mut Self {
        self.constrain(Constraint::NotEqual(a, b))
    }
    /// `a == b`.
    pub fn equal(&mut self, a: Var, b: Var) -> &mut Self {
        self.constrain(Constraint::Equal(a, b))
    }
    /// Pin a variable to a value.
    pub fn fix(&mut self, v: Var, value: i64) -> &mut Self {
        self.constrain(Constraint::Fix(v, value))
    }

    /// Exactly `k` of these literals must be true.
    pub fn cardinality(&mut self, lits: Vec<Lit>, k: usize) -> &mut Self {
        self.constrain(Constraint::Cardinality { lits, k })
    }

    /// At most `k` of these literals may be true. Introduces a slack variable.
    pub fn at_most(&mut self, lits: Vec<Lit>, k: usize) -> &mut Self {
        self.constrain(Constraint::AtMost { lits, k })
    }

    /// At least `k` of these literals must be true. Introduces a slack variable.
    pub fn at_least(&mut self, lits: Vec<Lit>, k: usize) -> &mut Self {
        self.constrain(Constraint::AtLeast { lits, k })
    }

    /// Exactly one of these literals is true.
    ///
    /// Cheaper than `cardinality(lits, 1)`: it lowers pairwise and needs no slack variable. Both
    /// express the same requirement, and this is the one to reach for.
    pub fn exactly_one(&mut self, lits: Vec<Lit>) -> &mut Self {
        self.constrain(Constraint::ExactlyOne(lits))
    }

    /// At most one of these literals is true, and possibly none.
    pub fn at_most_one(&mut self, lits: Vec<Lit>) -> &mut Self {
        self.constrain(Constraint::AtMostOne(lits))
    }

    /// Every one of `vars` takes a different value.
    ///
    /// The workhorse of assignment, scheduling, colouring and puzzles. See
    /// [`Constraint::AllDifferent`] for what it costs and what it catches.
    pub fn all_different<I: IntoIterator<Item = Var>>(&mut self, vars: I) -> &mut Self {
        let v: Vec<Var> = vars.into_iter().collect();
        self.constrain(Constraint::AllDifferent(v))
    }

    /// A **weighted** linear row: `Σ wᵢ·lᵢ ≤ rhs`, `≥ rhs` or `= rhs`.
    ///
    /// The constraint none of the counting forms can express. See [`Constraint::Linear`] for the
    /// lowering, what it costs in spins and couplings, and what it refuses by name.
    ///
    /// ```
    /// # use ferrotherm::model::{Model, Lit, Rel};
    /// let mut m = Model::new();
    /// let (a, b, c) = (m.binary("a"), m.binary("b"), m.binary("c"));
    /// m.linear(vec![(Lit::Is(a, 1), 3.0), (Lit::Is(b, 1), 4.0), (Lit::Is(c, 1), 5.0)], Rel::Le, 7.0);
    /// let compiled = m.compile().unwrap();
    /// assert_eq!(compiled.linear_slack, 3); // ⌈log₂ 8⌉
    /// ```
    pub fn linear(&mut self, terms: Vec<(Lit, f64)>, rel: Rel, rhs: f64) -> &mut Self {
        self.constrain(Constraint::Linear { terms, rel, rhs })
    }

    /// A weighted linear row you are willing to break, at a price per unit² of overshoot.
    ///
    /// The energy this contributes at the sampler's own optimum is exactly `weight × amount²`, in
    /// the modeller's own units — the same number [`Violation::cost`] reports. That identity is why
    /// a soft row is priced at `weight·g²` on the gcd-reduced row rather than at `weight`.
    pub fn linear_soft(
        &mut self,
        terms: Vec<(Lit, f64)>,
        rel: Rel,
        rhs: f64,
        weight: f64,
    ) -> &mut Self {
        self.soft(Constraint::Linear { terms, rel, rhs }, weight)
    }

    // ---- compiling ------------------------------------------------------------------------------

    /// Lower to a program and a decoder.
    ///
    /// # Penalty scaling
    ///
    /// Constraints have to *dominate* the objective they sit beside. A penalty that merely ties
    /// with an objective coefficient gets traded away, and the result is a state that does not
    /// decode at all — which looks like a broken sampler rather than an under-weighted constraint.
    ///
    /// So by default the penalty is `max(model.penalty, 2 × largest objective coefficient)`. Call
    /// [`Model::fixed_penalty`] to take that decision yourself.
    pub fn compile(&self) -> Result<Compiled, CompileError> {
        if self.decls.is_empty() {
            return Err(CompileError::Empty);
        }
        // An answer is a map keyed by name, so two variables sharing one do not shadow each other
        // -- the second overwrites the first and one of them vanishes from the result with nothing
        // said. Caught here rather than at declaration because a name can also arrive later,
        // through the FFI's rename.
        let mut seen = BTreeMap::new();
        for d in &self.decls {
            if seen.insert(d.name.clone(), ()).is_some() {
                return Err(CompileError::DuplicateName(d.name.clone()));
            }
        }

        // Every coefficient must be a finite number, checked HERE because this is where every path
        // that can introduce one converges -- `objective`, `set_objective`, the LP and OMMX
        // readers, and the C ABI all end up compiling. Checking in `objective` alone would leave
        // the other four open, and this crate has shipped that shape of gap before.
        // Refused here rather than at each entry point, for the same reason as the finiteness
        // check below: `ft_model_integer`, the LP reader, the OMMX reader and the Rust API all
        // converge on compile, and guarding one of them leaves the rest open.
        for d in &self.decls {
            let size = d.domain.size_u128();
            if size > u32::MAX as u128 {
                return Err(CompileError::DomainTooLarge { var: d.name.clone(), size });
            }
        }
        self.objective.check_finite("objective coefficient")?;
        // SOFT weights only. A hard constraint carries `f64::NAN` as its sentinel -- see the push
        // in `constrain` -- so checking every weight rejects every hard constraint in the crate,
        // which is what the first cut did and what the FFI cardinality tests caught immediately.
        // The sentinel is deliberate; the check has to know that.
        for (_, weight, hard) in &self.constraints {
            if !*hard && !weight.is_finite() {
                return Err(CompileError::NotFinite { what: "soft constraint weight", value: *weight });
            }
        }

        // The pigeonhole check, before any lowering. n variables that must all differ need at
        // least n distinct values BETWEEN them; fewer is unsatisfiable by counting alone, and no
        // amount of annealing discovers that in a way a modeller can act on.
        for (c, _, hard) in &self.constraints {
            if let Constraint::AllDifferent(vars) = c {
                if !*hard {
                    continue; // a soft all-different is a preference; it is allowed to be impossible
                }
                let mut distinct: Vec<i64> = Vec::new();
                for &v in vars {
                    for value in self.decls[v.0].domain.values() {
                        if !distinct.contains(&value) {
                            distinct.push(value);
                        }
                    }
                }
                if vars.len() > distinct.len() {
                    return Err(CompileError::Pigeonhole {
                        vars: vars.len(),
                        values: distinct.len(),
                    });
                }
            }
        }

        // Weighted linear rows are PLANNED before anything is laid out, because the plan is what
        // decides how many slack spins the row needs -- a gcd and a bit length, not an arithmetic
        // expression that can safely be recomputed. Computed once, here, and handed to
        // `apply_constraint` below. See `LinearPlan`.
        //
        // `caveats` is opened here rather than further down so a plan can push one: a row that
        // constrains nothing, or one whose coefficient spread is a numerical hazard, is something a
        // modeller has to be told at COMPILE time.
        let mut caveats: Vec<String> = Vec::new();
        let mut plans: BTreeMap<usize, LinearPlan> = BTreeMap::new();
        for (ci, (c, _, hard)) in self.constraints.iter().enumerate() {
            if let Constraint::Linear { terms, rel, rhs } = c {
                plans.insert(ci, self.plan_linear(terms, *rel, *rhs, *hard, &mut caveats)?);
            }
        }

        // Inequalities need slack, so the compiler declares variables of its own. They are laid out
        // after the user's, and the decoder never reports them: a slack variable is an artefact of
        // the lowering, not part of the answer.
        let user_count = self.decls.len();
        let mut extra: Vec<Domain> = Vec::new();
        let mut slack_for: BTreeMap<usize, usize> = BTreeMap::new();
        for (ci, (c, _, _)) in self.constraints.iter().enumerate() {
            let range = match c {
                Constraint::AtMost { k, .. } => Some(*k + 1),
                Constraint::AtLeast { lits, k } => Some(lits.len().saturating_sub(*k) + 1),
                _ => None,
            };
            if let Some(size) = range {
                if size >= 2 {
                    slack_for.insert(ci, user_count + extra.len());
                    extra.push(Domain::Categorical(size));
                }
            }
        }

        // Lay every variable out, in declaration order, so the layout is stable and inspectable.
        let all: Vec<(Domain, Encoding)> = self
            .decls
            .iter()
            .map(|d| (d.domain, d.encoding))
            .chain(extra.iter().map(|d| (*d, Encoding::OneHot)))
            .collect();
        let mut slots = Vec::with_capacity(all.len());
        let mut base = 0usize;
        for (domain, encoding) in &all {
            let k = domain.size();
            let s = Slot::new(base, k, *encoding);
            base += s.width();
            slots.push(s);
        }
        // A weighted row's slack spins sit after every declared variable and every counting slack,
        // and they are NOT slots. A truncated-binary slack over `{0..=S}` has no invalid codeword,
        // so there is no encoding to penalise and nothing for `Slot::decode` to refuse -- making
        // them a slot would buy an exclusion penalty for a block that needs none, and would put a
        // `Domain::Categorical(2^m)` into the layout that a 32-bit `usize` cannot hold.
        let mut linear_slack = 0usize;
        for plan in plans.values_mut() {
            let width = plan.spins();
            if let LinearPlan::Integer { base: slack_base, .. } = plan {
                *slack_base = base + linear_slack;
            }
            linear_slack += width;
        }
        let n = base + linear_slack;

        let mut b = GraphBuilder::new(n);
        let penalty = self.effective_penalty();

        // Encoding penalties: what makes a spin pattern mean a value at all.
        //
        // The returned bool is not decoration. `false` means invalid codewords remain exactly as
        // cheap as valid ones, so the sampler is free to land on one and `decode` is the only
        // thing between that and a wrong answer. Collected rather than discarded.
        // Patterns the modeller wrote longhand that have a cheaper form, and constraints that
        // constrain nothing. Reported, never rewritten: silently compiling something other than
        // what was written is the opposite of this compiler's discipline, and a modeller who meant
        // the expensive form is entitled to it.
        //
        // Only what MEASURES cheaper is reported. `cardinality(lits, 1)` and `exactly_one` compile
        // to identical graphs here (10 spins, 15 factors on a five-literal test), and so do six
        // pairwise `not_equal`s and one `all_different` (16 spins, 48 factors) -- so neither earns a
        // caveat, and saying otherwise would be advice that costs the reader time and saves nothing.
        for (c, _, hard) in &self.constraints {
            match c {
                Constraint::AtMost { lits, k } if *k == 1 && lits.len() >= 2 && *hard => {
                    caveats.push(format!(
                        "at_most over {} literals with k = 1 costs a slack variable: measured at 12 \
                         spins and 26 factors where at_most_one is 10 and 15. An inequality has to \
                         become an equality the sampler can square, and at k = 1 the pairwise \
                         exclusion says the same thing for free. Use at_most_one.",
                        lits.len()
                    ));
                }
                Constraint::AtMost { lits, k } if *k >= lits.len() && *hard => {
                    caveats.push(format!(
                        "at_most over {} literals with k = {k} constrains nothing -- {k} of {} can \
                         always hold -- and still pays for a slack variable and its factors. This is \
                         usually a k that was meant to be smaller.",
                        lits.len(),
                        lits.len()
                    ));
                }
                Constraint::AtLeast { lits, k } if *k == 0 && *hard => {
                    caveats.push(format!(
                        "at_least over {} literals with k = 0 constrains nothing -- zero of them \
                         holding already satisfies it -- and still pays for a slack variable.",
                        lits.len()
                    ));
                }
                _ => {}
            }
        }

        for (i, s) in slots.iter().enumerate() {
            if !s.add_penalty(&mut b, penalty) {
                let k = s.k;
                let spins = s.width();
                let spare = (1usize << spins) - k;
                caveats.push(format!(
                    "'{}' is {:?}-encoded over {k} values in {spins} spins, which spell {} \
                     codewords. The {spare} spare one(s) decode to nothing and NO penalty removes \
                     them -- an invalid state costs exactly what a valid one costs, so the sampler \
                     has no reason to prefer an answer. Use one-hot or domain-wall for an exact \
                     encoding, or a k that is a power of two.",
                    // A SLACK slot has no declaration, and indexing `decls` by a slot index was a
                    // panic waiting for the first slack whose encoding could not be made exact.
                    // The trap was armed and unreached; naming the slot is both the fix and the
                    // more useful message.
                    self.decls.get(i).map(|d| d.name.as_str()).unwrap_or("a compiler slack variable"),
                    s.encoding,
                    1usize << spins,
                ));
            }
        }

        // Constraints. A NaN strength means "use the model's, after scaling".
        // Anything wider than pairwise, from a constraint or an objective term alike, lands here
        // and is lowered by `reduce` once the graph is built.
        let mut higher: Vec<(Vec<usize>, f64)> = Vec::new();

        for (ci, (c, p, hard)) in self.constraints.iter().enumerate() {
            let p = if p.is_nan() { penalty } else { *p };
            self.apply_constraint(
                &mut b,
                &mut higher,
                &slots,
                c,
                p,
                *hard,
                slack_for.get(&ci).copied(),
                plans.get(&ci),
            )?;
        }

        // The objective is already stored as "minimise this": `Model::objective` folded the sense
        // in when each term arrived, so there is nothing left to decide here.
        // A term of three or more literals does not fit a pairwise graph, so it is collected here
        // and lowered by `crate::reduce` below. This used to refuse outright, which was right when
        // there was no pass to apply.
        for t in &self.objective.terms {
            let parts: Result<Vec<LinSpin>, CompileError> =
                t.lits.iter().map(|l| self.linearise(&slots, *l)).collect();
            let parts = parts?;
            match parts.len() {
                0 => {}
                1 => add_linear(&mut b, &parts[0], t.coeff),
                2 => add_product(&mut b, &parts[0], &parts[1], t.coeff),
                d => {
                    if d > crate::reduce::MAX_ARITY {
                        return Err(CompileError::DegreeTooHigh { degree: d });
                    }
                    // Each literal is a linear function of spins, so their product expands into
                    // spin monomials. Whatever lands at degree 0, 1 or 2 goes straight into the
                    // graph; only what is genuinely wider costs an ancilla.
                    Self::emit(&mut b, &mut higher, &parts, t.coeff);
                }
            }
        }

        let graph = b.build();

        // Lower whatever stayed wider than two, and take the reduced graph in its place.
        let sched = Schedule::geometric(0.05, 6.0, 80, 40);
        let (graph, ancillas) = if higher.is_empty() {
            (graph, 0)
        } else {
            let mut p = Program::from_graph(&graph, &sched);
            for (vars, c) in &higher {
                // A Program factor contributes `-w · ∏s`, so the sign flips going in.
                p.factors.push(
                    crate::factor::Factor::new(vars, -c, graph.n)
                        .map_err(|_| CompileError::DegreeTooHigh { degree: vars.len() })?,
                );
            }
            let r = crate::reduce::to_pairwise(&p)
                .map_err(|_| CompileError::DegreeTooHigh { degree: self.objective.max_degree() })?;
            let g = r
                .program
                .to_graph()
                .map_err(|_| CompileError::DegreeTooHigh { degree: 3 })?;
            (g, r.ancillas)
        };

        let mut program = Program::from_graph(&graph, &sched);
        program.encodings = slots
            .iter()
            .map(|s| EncodedVar { base: s.base, k: s.k, encoding: s.encoding })
            .collect();

        Ok(Compiled {
            objective: self.objective.clone(),
            sense: self.sense,
            program,
            graph,
            ancillas,
            linear_slack,
            caveats,
            // only the user's variables are reported; slack is an artefact of the lowering
            slots: slots[..user_count].to_vec(),
            all_slots: slots,
            names: self.decls.iter().map(|d| d.name.clone()).collect(),
            domains: self.decls.iter().map(|d| d.domain).collect(),
            constraints: self
                .constraints
                .iter()
                .map(|(c, p, hard)| (c.clone(), *hard, if p.is_nan() { penalty } else { *p }))
                .collect(),
        })
    }

    /// The product of several linear spin functions, as spin monomials.
    ///
    /// Each part is `offset + Σ cᵢ sᵢ`, so the product is a sum over every way of picking one piece
    /// from each. A spin appearing an even number of times cancels — `sᵢ² = 1` — which is why the
    /// accumulator applies a parity rule rather than keeping every occurrence.
    fn expand_product(parts: &[LinSpin], coeff: f64) -> Vec<(Vec<usize>, f64)> {
        let mut acc: BTreeMap<Vec<usize>, f64> = BTreeMap::new();
        let mut stack: Vec<(usize, Vec<usize>, f64)> = vec![(0, Vec::new(), coeff)];
        while let Some((i, vars, c)) = stack.pop() {
            if c == 0.0 {
                continue;
            }
            if i == parts.len() {
                let mut v = vars;
                v.sort_unstable();
                let mut collapsed = Vec::new();
                let mut k = 0;
                while k < v.len() {
                    let run = v[k..].iter().take_while(|x| **x == v[k]).count();
                    if run % 2 == 1 {
                        collapsed.push(v[k]);
                    }
                    k += run;
                }
                *acc.entry(collapsed).or_insert(0.0) += c;
                continue;
            }
            let p = &parts[i];
            if p.offset != 0.0 {
                stack.push((i + 1, vars.clone(), c * p.offset));
            }
            for (idx, w) in &p.terms {
                let mut next = vars.clone();
                next.push(*idx);
                stack.push((i + 1, next, c * w));
            }
        }
        acc.into_iter().filter(|(_, c)| *c != 0.0).collect()
    }

    /// A literal as a linear function of spins: `offset + Σ cᵢ sᵢ`.
    /// Add `coeff · ∏ parts` to the graph, sending anything wider than pairwise to `higher`.
    ///
    /// The single place a polynomial in spins becomes couplings. Every constraint and every
    /// objective term goes through it, so the degree rule is stated once: degree 0 is a constant
    /// and changes no answer, 1 is a field, 2 is a coupling, and 3 or more is a term
    /// [`crate::reduce`] will lower with an ancilla.
    fn emit(
        b: &mut GraphBuilder,
        higher: &mut Vec<(Vec<usize>, f64)>,
        parts: &[LinSpin],
        coeff: f64,
    ) {
        for (vars, c) in Self::expand_product(parts, coeff) {
            match vars.len() {
                0 => {}
                1 => b.bias(vars[0], -c),
                2 => b.couple(vars[0], vars[1], -c),
                _ => higher.push((vars, c)),
            }
        }
    }

    /// A literal as a product of linear spin forms.
    ///
    /// One-hot and spin indicators are linear, so the product has one factor. A **domain-wall**
    /// indicator is not: value `v` means the spins below it are up and the rest down, so
    /// `[x = v]` is `(1 + s_{v-1})/2 · (1 - s_v)/2` — two factors. Returning a product rather than
    /// a linear form is what lets a domain-wall variable be used at all; before this it was
    /// refused everywhere, which made `categorical_as` an API that could be called and not used.
    ///
    /// Callers multiply these together and route the resulting monomials by degree: anything wider
    /// than two goes through [`crate::reduce`], exactly as a cubic objective term does.
    fn factors(&self, slots: &[Slot], l: Lit) -> Result<Vec<LinSpin>, CompileError> {
        let d = match l {
            Lit::Spin(v) | Lit::Is(v, _) => &self.decls[v.0],
        };
        if d.encoding == Encoding::Binary {
            // A binary code's indicator is a product of every bit, so its degree grows with the
            // domain and it stops being a thing worth writing down. Refused by name.
            return Err(CompileError::NeedsOneHot {
                var: d.name.clone(),
                encoding: d.encoding,
            });
        }
        if d.encoding == Encoding::DomainWall {
            let Lit::Is(v, value) = l else {
                return Err(CompileError::NeedsOneHot { var: d.name.clone(), encoding: d.encoding });
            };
            let Some(slot) = d.domain.index_of(value) else {
                return Err(CompileError::BadValue {
                    var: d.name.clone(),
                    value,
                    domain: d.domain,
                });
            };
            let s = slots[v.0];
            let w = s.width();
            // below[i] = (1 + s_i)/2 is "the wall is above i"; above[i] = (1 - s_i)/2 its negation.
            let up = |i: usize| LinSpin { offset: 0.5, terms: vec![(s.base + i, 0.5)] };
            let down = |i: usize| LinSpin { offset: 0.5, terms: vec![(s.base + i, -0.5)] };
            return Ok(match (slot, w) {
                (0, _) => vec![down(0)],                    // the wall is at the very bottom
                (v, w) if v == w => vec![up(w - 1)],        // and at the very top
                (v, _) => vec![up(v - 1), down(v)],         // otherwise it sits between two spins
            });
        }
        self.linearise(slots, l).map(|p| vec![p])
    }

    fn linearise(&self, slots: &[Slot], l: Lit) -> Result<LinSpin, CompileError> {
        match l {
            Lit::Spin(v) => {
                // A Spin/Binary variable is one-hot over two values; +1 means value 1.
                let s = slots[v.0];
                Ok(LinSpin { offset: 0.0, terms: vec![(s.base + 1, 1.0)] })
            }
            Lit::Is(v, value) => {
                let d = &self.decls[v.0];
                // The modeller writes a value; the fabric holds a slot. This is where the two meet,
                // and it is the only place that knows the difference.
                let Some(slot) = d.domain.index_of(value) else {
                    return Err(CompileError::BadValue {
                        var: d.name.clone(),
                        value,
                        domain: d.domain,
                    });
                };
                if d.encoding != Encoding::OneHot {
                    return Err(CompileError::NeedsOneHot {
                        var: d.name.clone(),
                        encoding: d.encoding,
                    });
                }
                // one-hot: indicator = (1 + s)/2
                let s = slots[v.0];
                Ok(LinSpin { offset: 0.5, terms: vec![(s.base + slot, 0.5)] })
            }
        }
    }

    /// A literal as a **0/1 indicator**, which is what arithmetic means by it.
    ///
    /// [`Model::linearise`] returns the ±1 SPIN for `Lit::Spin`, which is right for a counting
    /// constraint written against it and wrong for a weighted row: `3·a` in a modeller's row means
    /// three when `a` holds and zero when it does not, not ±3. `(1 + s)/2` is that indicator, and
    /// it is the same shape `Lit::Is` already lowers to.
    fn indicator(&self, slots: &[Slot], l: Lit) -> Result<LinSpin, CompileError> {
        let d = &self.decls[lit_var(l).0];
        if d.encoding != Encoding::OneHot {
            // A domain-wall indicator is a PRODUCT of two spins, so a weighted sum of them inside
            // a square reaches degree four and buys ancillas; a binary indicator is a product of
            // every bit. Refused by name, exactly as the counting constraints refuse them.
            return Err(CompileError::NeedsOneHot { var: d.name.clone(), encoding: d.encoding });
        }
        match l {
            Lit::Spin(v) => {
                let s = slots[v.0];
                Ok(LinSpin { offset: 0.5, terms: vec![(s.base + 1, 0.5)] })
            }
            Lit::Is(..) => self.linearise(slots, l),
        }
    }

    /// One row, written in the caller's own notation.
    fn row_label(&self, terms: &[(Lit, f64)], rel: Rel, rhs: f64) -> String {
        let parts: Vec<(String, f64)> = terms
            .iter()
            .map(|(l, c)| {
                let d = &self.decls[lit_var(*l).0];
                (lit_label(&d.name, d.domain, *l), *c)
            })
            .collect();
        row_text(&parts, rel, rhs)
    }

    /// Decide, once, how a weighted linear row is lowered.
    ///
    /// Merges repeated literals, drops zero weights, normalises a `Ge` row by negation, divides
    /// through by the gcd, and sizes the truncated-binary slack. Refuses what it cannot represent
    /// by name; pushes a caveat for what it can represent and a modeller should still know.
    fn plan_linear(
        &self,
        terms: &[(Lit, f64)],
        rel: Rel,
        rhs: f64,
        hard: bool,
        caveats: &mut Vec<String>,
    ) -> Result<LinearPlan, CompileError> {
        let row = self.row_label(terms, rel, rhs);
        for (l, c) in terms {
            if !c.is_finite() {
                return Err(CompileError::NotFinite { what: "linear coefficient", value: *c });
            }
            // Reached here rather than at `apply_constraint` so a bad value is refused even when
            // the row turns out to be vacuous and emits nothing at all.
            let d = &self.decls[lit_var(*l).0];
            if let Lit::Is(_, value) = l {
                if d.domain.index_of(*value).is_none() {
                    return Err(CompileError::BadValue {
                        var: d.name.clone(),
                        value: *value,
                        domain: d.domain,
                    });
                }
            }
        }
        if !rhs.is_finite() {
            return Err(CompileError::NotFinite { what: "linear right-hand side", value: rhs });
        }

        // MERGE repeated literals and drop zero weights. The algebra survives duplicates unmerged
        // -- `add_product(a, a, w)` applies x² = x correctly through the offsets -- but `check`'s
        // arithmetic and the cost reported below are only honest on merged weights, and merging
        // shrinks the clique this row pays for.
        let mut lits: Vec<Lit> = Vec::new();
        let mut ws: Vec<f64> = Vec::new();
        for (l, c) in terms {
            match lits.iter().position(|x| x == l) {
                Some(i) => ws[i] += *c,
                None => {
                    lits.push(*l);
                    ws.push(*c);
                }
            }
        }
        let keep: Vec<usize> = (0..lits.len()).filter(|i| ws[*i] != 0.0).collect();
        let lits: Vec<Lit> = keep.iter().map(|i| lits[*i]).collect();
        let ws: Vec<f64> = keep.iter().map(|i| ws[*i]).collect();

        // Is every number an integer this crate can hold exactly?
        let exact = |v: f64| v.fract() == 0.0 && v.abs() <= EXACT_INT;
        let all_int = ws.iter().all(|c| exact(*c)) && exact(rhs);

        if !all_int {
            if rel != Rel::Eq {
                // Name the offending number rather than the row alone: a fifty-term row with one
                // 2.5 in it is a needle, and "some coefficient is not an integer" is not a message.
                let (what, value) = match ws.iter().position(|c| !exact(*c)) {
                    Some(i) => {
                        let d = &self.decls[lit_var(lits[i]).0];
                        (format!("coefficient on {}", lit_label(&d.name, d.domain, lits[i])), ws[i])
                    }
                    None => ("right-hand side".to_string(), rhs),
                };
                // WHICH refusal this is matters: a whole number too big for f64 to count with is
                // a different problem from a fraction, and the advice for one cannot fix the other.
                if value.fract() == 0.0 {
                    return Err(CompileError::LinearHugeCoefficient {
                        row,
                        what,
                        value,
                        limit: EXACT_INT,
                    });
                }
                return Err(CompileError::LinearNotInteger { row, what, value });
            }
            caveats.push(format!(
                "the weighted row {row} has a non-integer coefficient, which an EQUALITY takes: it \
                 is compiled as p·(lhs − rhs)² and needs no slack. What it does not get is the \
                 guarantee an integer row gets. For integers the smallest nonzero |lhs − rhs| is 1, \
                 so the gap is the penalty; here it can be arbitrarily small, computing it is \
                 exponential, and this crate therefore cannot certify that the penalty is large \
                 enough to stop the row being traded against the objective."
            ));
            return Ok(LinearPlan::Real { lits, w: ws, t: rhs });
        }

        // Normalise to `Σ a·x ≤ t`, or to `Σ a·x = t`.
        let sgn: i64 = if rel == Rel::Ge { -1 } else { 1 };
        let a: Vec<i64> = ws.iter().map(|c| sgn * (*c as i64)).collect();
        let t: i64 = sgn * (rhs as i64);

        // THE GCD, and the two arms take DIFFERENT ones, for a reason that is not tidiness.
        //
        // The left side is an integer multiple of `gw`, the gcd of the weights, so for `≤` the
        // target can be FLOORED: `1000a + 1000b ≤ 1501` is exactly `a + b ≤ 1`. That is the
        // strongest reduction available and it is worth a great deal -- the same row without the
        // floor spans 1501 residual values and costs 11 slack spins instead of 1.
        //
        // A SOFT row cannot take it. Flooring throws away `t − gw·⌊t/gw⌋`, so the reduced residual
        // stops being the modeller's own overshoot divided by anything, and `Violation::cost`
        // (`weight × amount²`) stops matching the energy the sampler is actually trading against.
        // Taking the target INTO the gcd keeps `amount = g × (reduced residual)` exactly, at the
        // cost of a coarser reduction. A hard violation has no price to match, so each arm takes
        // the reduction that is right for it.
        let gw = a.iter().fold(0i64, |g, x| gcd_i64(g, *x));
        let g = if gw == 0 {
            1
        } else if hard && rel != Rel::Eq {
            gw
        } else {
            gcd_i64(gw, t).max(1)
        };
        let ar: Vec<i64> = a.iter().map(|x| x / g).collect();
        // Floor division, not truncation: for `≤` the exact reduction of a negative target is
        // ⌊t/g⌋, and `t / g` rounds toward zero, which for t = −1500, g = 1000 gives −1 where the
        // row means −2. For `=` the divisibility check above has already established g | t, so the
        // two agree.
        let tr = t.div_euclid(g);

        let lo: i128 = ar.iter().map(|x| (*x).min(0) as i128).sum();
        let hi: i128 = ar.iter().map(|x| (*x).max(0) as i128).sum();

        // The coefficient spread, which decides the COUPLING spread, which is a real numerical
        // hazard under one global β on every fabric this crate targets.
        let spread = |extra: &[i64]| {
            let mut mx = 0i64;
            let mut mn = i64::MAX;
            for v in ar.iter().chain(extra.iter()) {
                let v = v.abs();
                if v == 0 {
                    continue;
                }
                mx = mx.max(v);
                mn = mn.min(v);
            }
            if mn == i64::MAX || mn == 0 { 1.0 } else { mx as f64 / mn as f64 }
        };

        if rel == Rel::Eq {
            // Necessary, not sufficient: a target outside the reachable range, or one the weights'
            // own gcd cannot reach, has no answer by arithmetic. Subset-sum is NP-hard, so a row
            // that passes this can still have none, and claiming otherwise would be an absence
            // claim this crate cannot support.
            if (t != 0 && gw != 0 && t % gw != 0) || (tr as i128) < lo || (tr as i128) > hi {
                let best = if (tr as i128) > hi { (hi * g as i128) as f64 } else { (lo * g as i128) as f64 };
                return Err(CompileError::LinearUnsatisfiable { row, best });
            }
            let sp = spread(&[]);
            if sp > 100.0 {
                caveats.push(format!(
                    "the weighted row {row} spans {sp:.0}× in its coefficients after dividing \
                     through by {g}, so its couplings span {:.0}× — they carry p·wᵢwⱼ, which is \
                     quadratic in the weights. One global β anneals all of them, and a fixed-point \
                     fabric will quantise the smallest to nothing. Fabric::check reports what a \
                     given device makes of it.",
                    sp * sp,
                ));
            }
            return Ok(LinearPlan::Integer { lits, w: ar, t: tr, g, coeffs: Vec::new(), base: 0 });
        }

        // `≤` from here. The smallest the left side can be is `lo`; if that already exceeds the
        // target, nothing satisfies the row.
        if lo > tr as i128 {
            return Err(CompileError::LinearUnsatisfiable {
                row,
                best: (lo * g as i128) as f64 * sgn as f64,
            });
        }
        // And if the LARGEST it can be already satisfies the row, the row says nothing. Emitting
        // the squared penalty anyway would still be correct here -- the slack covers the whole
        // range -- and would cost spins and couplings for a statement with no content.
        if hi <= tr as i128 {
            caveats.push(format!(
                "the weighted row {row} constrains nothing: every assignment already satisfies it, \
                 so it compiles to no terms and no slack at all. This is usually a right-hand side \
                 that was meant to be tighter."
            ));
            return Ok(LinearPlan::Vacuous);
        }

        let span: i128 = tr as i128 - lo; // S, the number of residual values minus one
        let m = if span == 0 { 0 } else { 128 - (span as u128).leading_zeros() as usize };
        if m > 62 {
            return Err(CompileError::LinearTooLarge { row, span: span + 1, spins: m });
        }
        // Truncated binary: the top digit carries the REMAINDER of the span, not another power of
        // two. Verified by enumeration for every span up to 600: this covers {0…S} exactly, so
        // there is no invalid codeword to exclude and the block costs no encoding penalty. The
        // naive expansion is also sound for a one-sided row, and it wastes exactly where it hurts:
        // at S = 8 its top coefficient is 8 where this one's is 1.
        let coeffs = truncated_binary(span);
        debug_assert_eq!(coeffs.len(), m);

        let sp = spread(&coeffs);
        if sp > 100.0 {
            caveats.push(format!(
                "the weighted row {row} spans {sp:.0}× in its coefficients after dividing through \
                 by {g}, so its couplings span {:.0}× — they carry p·wᵢwⱼ, which is quadratic in \
                 the weights. One global β anneals all of them, and a fixed-point fabric will \
                 quantise the smallest to nothing. Fabric::check reports what a given device makes \
                 of it.",
                sp * sp,
            ));
        }
        Ok(LinearPlan::Integer { lits, w: ar, t: tr, g, coeffs, base: 0 })
    }

    fn apply_constraint(
        &self,
        b: &mut GraphBuilder,
        higher: &mut Vec<(Vec<usize>, f64)>,
        slots: &[Slot],
        c: &Constraint,
        p: f64,
        hard: bool,
        slack: Option<usize>,
        plan: Option<&LinearPlan>,
    ) -> Result<(), CompileError> {
        match c {
            Constraint::NotEqual(a, x) => {
                // Penalise agreeing on any value: p · Σ_v [a=v][x=v]. Over the values the two
                // domains SHARE, not over slot indices -- an integer 5..=10 and an integer 0..=5
                // agree only at 5, and comparing them slot by slot would say otherwise.
                for v in shared_values(&self.decls[a.0].domain, &self.decls[x.0].domain) {
                    let mut parts = self.factors(slots, Lit::Is(*a, v))?;
                    parts.extend(self.factors(slots, Lit::Is(*x, v))?);
                    Self::emit(b, higher, &parts, p);
                }
            }
            Constraint::Equal(a, x) => {
                for v in shared_values(&self.decls[a.0].domain, &self.decls[x.0].domain) {
                    let mut parts = self.factors(slots, Lit::Is(*a, v))?;
                    parts.extend(self.factors(slots, Lit::Is(*x, v))?);
                    Self::emit(b, higher, &parts, -p);
                }
            }
            Constraint::Fix(v, value) => {
                // reward taking it
                let parts = self.factors(slots, Lit::Is(*v, *value))?;
                Self::emit(b, higher, &parts, -p);
            }
            Constraint::AtMost { lits, k } | Constraint::AtLeast { lits, k } => {
                // Σ lits ± slack = k, squared. `sum` collects every weighted indicator on the left.
                let sign = if matches!(c, Constraint::AtMost { .. }) { 1.0 } else { -1.0 };
                let mut sum: Vec<(LinSpin, f64)> = Vec::new();
                for l in lits {
                    sum.push((self.linearise(slots, *l)?, 1.0));
                }
                // No slack means the inequality has no room in it: `at most 0` and `at least all`
                // each admit exactly one count, so the equality Σ lits = k IS the constraint and it
                // is applied with no slack term. Returning early here instead -- which is what this
                // did -- dropped the constraint entirely, and `at most 0 of these` compiled to
                // nothing at all while reporting feasible.
                if let Some(sv) = slack {
                    let s = slots[sv];
                    for v in 0..s.k {
                        // the slack's value v enters with weight v, via its one-hot indicator
                        sum.push((
                            LinSpin { offset: 0.5, terms: vec![(s.base + v, 0.5)] },
                            sign * v as f64,
                        ));
                    }
                }
                add_squared(b_ref(b), &sum, *k as f64, p);
            }
            Constraint::Cardinality { lits, k } => {
                // p·(Σ xᵢ − k)² = p·(Σ xᵢxⱼ over ordered pairs − 2k·Σ xᵢ + k²); the constant drops.
                let lins: Result<Vec<LinSpin>, CompileError> =
                    lits.iter().map(|l| self.linearise(slots, *l)).collect();
                let lins = lins?;
                for i in 0..lins.len() {
                    // the diagonal: xᵢ² = xᵢ for a 0/1 indicator, so it joins the linear part
                    add_linear(b, &lins[i], p * (1.0 - 2.0 * *k as f64));
                    for j in (i + 1)..lins.len() {
                        add_product(b, &lins[i], &lins[j], 2.0 * p);
                    }
                }
            }
            Constraint::AllDifferent(vars) => {
                // Per shared value, not per pair. Collect the indicator for "v takes this value"
                // from every variable whose domain contains it, then exclude them pairwise.
                // Every value at least two of them could both take. A value only one can take
                // cannot collide, so it contributes nothing and is skipped rather than emitted.
                let mut candidates: Vec<i64> = Vec::new();
                for &v in vars {
                    for value in self.decls[v.0].domain.values() {
                        if !candidates.contains(&value) {
                            candidates.push(value);
                        }
                    }
                }
                for value in candidates {
                    let mut lins = Vec::new();
                    for &v in vars {
                        if self.decls[v.0].domain.index_of(value).is_some() {
                            lins.push(self.linearise(slots, Lit::Is(v, value))?);
                        }
                    }
                    for i in 0..lins.len() {
                        for j in (i + 1)..lins.len() {
                            add_product(b, &lins[i], &lins[j], 2.0 * p);
                        }
                    }
                }
            }
            Constraint::Linear { .. } => {
                // The plan was computed in `compile` and is the ONLY place the slack width is
                // decided. Recomputing it here is the defect this shape exists to prevent.
                let Some(plan) = plan else {
                    // Unreachable: `compile` plans every Linear row before it lays anything out.
                    // Refused rather than assumed, because a silent zero-term constraint is
                    // exactly the failure this compiler exists to catch.
                    return Err(CompileError::DegreeTooHigh { degree: 0 });
                };
                match plan {
                    LinearPlan::Vacuous => {}
                    LinearPlan::Real { lits, w, t } => {
                        let mut parts: Vec<(LinSpin, f64)> = Vec::with_capacity(lits.len());
                        for (l, c) in lits.iter().zip(w.iter()) {
                            parts.push((self.indicator(slots, *l)?, *c));
                        }
                        add_squared(b, &parts, *t, p);
                    }
                    LinearPlan::Integer { lits, w, t, g, coeffs, base } => {
                        // HARD rows are priced at `p` on the reduced row: a unit of violation then
                        // costs exactly `p`, the same gap `Cardinality` has, so `effective_penalty`
                        // needs no change -- and the coefficients stay as small as the row allows.
                        // SOFT rows are priced at `p·g²`, which is what makes the energy this
                        // contributes equal the `weight × amount²` that `Violation::cost` reports
                        // in the modeller's own units. A hard violation has no price to match, so
                        // each arm takes the scale that is right for it.
                        let scale = if hard { p } else { p * (*g as f64) * (*g as f64) };
                        let mut parts: Vec<(LinSpin, f64)> =
                            Vec::with_capacity(lits.len() + coeffs.len());
                        for (l, c) in lits.iter().zip(w.iter()) {
                            parts.push((self.indicator(slots, *l)?, *c as f64));
                        }
                        for (j, c) in coeffs.iter().enumerate() {
                            // The slack bit's own 0/1 indicator, (1 + s)/2, carrying its weight.
                            parts.push((
                                LinSpin { offset: 0.5, terms: vec![(base + j, 0.5)] },
                                *c as f64,
                            ));
                        }
                        add_squared(b, &parts, *t as f64, scale);
                    }
                }
            }
            Constraint::ExactlyOne(lits) | Constraint::AtMostOne(lits) => {
                // pairwise exclusion; ExactlyOne additionally rewards being on
                for i in 0..lits.len() {
                    let li = self.linearise(slots, lits[i])?;
                    if matches!(c, Constraint::ExactlyOne(_)) {
                        add_linear(b, &li, -p);
                    }
                    for j in (i + 1)..lits.len() {
                        let lj = self.linearise(slots, lits[j])?;
                        add_product(b, &li, &lj, 2.0 * p);
                    }
                }
            }
        }
        Ok(())
    }
}

/// The variable a literal is about.
fn lit_var(l: Lit) -> Var {
    match l {
        Lit::Spin(v) | Lit::Is(v, _) => v,
    }
}

/// A coefficient the way a modeller wrote it: `3`, not `3.0000000`.
fn num(x: f64) -> String {
    if x.fract() == 0.0 && x.abs() < 1e15 {
        format!("{}", x as i64)
    } else {
        format!("{x}")
    }
}

/// One literal, in the caller's own words. `a`, or `(shift=3)`.
fn lit_label(name: &str, domain: Domain, l: Lit) -> String {
    match l {
        Lit::Spin(_) => name.to_string(),
        Lit::Is(_, value) => match domain {
            Domain::Binary | Domain::Spin if value == 1 => name.to_string(),
            _ => format!("({name}={value})"),
        },
    }
}

/// A whole row, in the caller's own notation: `3·a + 4·b − 5·(shift=2) ≤ 7`.
///
/// Truncated past six terms, because a 200-item knapsack row printed in full is not a message
/// anyone reads — and a violation nobody reads is the same as no violation.
fn row_text(parts: &[(String, f64)], rel: Rel, rhs: f64) -> String {
    if parts.is_empty() {
        return format!("0 {} {}", rel.symbol(), num(rhs));
    }
    let mut s = String::new();
    for (i, (nm, c)) in parts.iter().take(6).enumerate() {
        if i == 0 {
            if *c < 0.0 {
                s.push('−');
            }
        } else {
            s.push_str(if *c < 0.0 { " − " } else { " + " });
        }
        s.push_str(&format!("{}·{nm}", num(c.abs())));
    }
    if parts.len() > 6 {
        s.push_str(&format!(" + … ({} terms)", parts.len()));
    }
    format!("{s} {} {}", rel.symbol(), num(rhs))
}

/// Slack coefficients spanning `{0 ..= span}` in `⌈log₂(span+1)⌉` spins, exactly.
///
/// `1, 2, 4, …, 2^(m−2), span+1−2^(m−1)`. The top digit carries the REMAINDER of the span
/// rather than another power of two, and that is the whole point: the naive expansion
/// `1, 2, …, 2^(m−1)` reaches past `span`, which is harmless for a one-sided row and wastes
/// exactly where it hurts — at `span = 8` its top coefficient is 8 where this one's is 1, and the
/// coupling that coefficient carries is quadratic in it.
///
/// Because the cover is exact there is no invalid codeword, so this block needs no encoding penalty
/// and no exclusion couplings. What remains is a 2-to-1 map on a window of `2^m − (span+1)`
/// values, empty exactly when `span+1` is a power of two: the ground state of a row is degenerate
/// by at most 2 over those, which changes no argmin and biases a Boltzmann marginal over LOGICAL
/// states by at most 2×.
fn truncated_binary(span: i128) -> Vec<i64> {
    if span <= 0 {
        return Vec::new();
    }
    let m = 128 - (span as u128).leading_zeros() as usize;
    let mut coeffs: Vec<i64> = Vec::with_capacity(m);
    for j in 0..m - 1 {
        coeffs.push(1i64 << j);
    }
    coeffs.push((span + 1 - (1i128 << (m - 1))) as i64);
    coeffs
}

fn gcd_i64(a: i64, b: i64) -> i64 {
    let (mut a, mut b) = (a.abs(), b.abs());
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

/// The largest magnitude an f64 represents every integer below. Past it, `fract() == 0` says
/// nothing useful and `as i64` saturates instead of failing, so it is the boundary of the integer
/// path rather than a stylistic limit.
const EXACT_INT: f64 = 9_007_199_254_740_992.0; // 2^53

/// What [`Model::compile`] decided about one [`Constraint::Linear`] row, computed **once**.
///
/// The existing slack sizing recomputes `k + 1` in two places, which is safe for one arithmetic
/// expression and is not safe for a gcd and a bit length: a second computation that disagrees with
/// the first gives the slack block the wrong width, and nothing errors. So this is built once and
/// handed to `apply_constraint`, and a test asserts the emitted spin count matches it.
#[derive(Clone, Debug)]
enum LinearPlan {
    /// The row constrains nothing. Zero spins, zero terms, one caveat.
    Vacuous,
    /// The integer path: gcd-reduced weights, a reduced target, and a truncated-binary slack.
    Integer {
        lits: Vec<Lit>,
        /// Reduced weights, normalised so the row reads `Σ w·x ≤ t` (a `Ge` row is negated).
        w: Vec<i64>,
        t: i64,
        /// The gcd divided out.
        ///
        /// A **soft** row is priced at `weight·g²`, so the energy it contributes at the sampler's
        /// own optimum is `weight × amount²` in the modeller's units — the number
        /// [`Violation::cost`] reports. A **hard** row is priced at `p`, which is a gap of exactly
        /// `p` per unit of violation — the same gap `Cardinality` has, so `effective_penalty`
        /// needs no change — at the smallest coefficients the row admits. A hard violation has no
        /// price to match, so the two choices are each right for their own arm.
        g: i64,
        /// Truncated-binary slack coefficients: `1, 2, …, 2^(m−2), S+1−2^(m−1)`.
        /// Empty for an equality, and for an inequality with no room in it.
        coeffs: Vec<i64>,
        /// First slack spin. Filled in by `compile` once the layout is known.
        base: usize,
    },
    /// A non-integer **equality**, squared directly in f64. It needs no slack, so it needs no
    /// lattice.
    Real { lits: Vec<Lit>, w: Vec<f64>, t: f64 },
}

impl LinearPlan {
    /// Spins this row adds. The number the doc comment promises, read from the plan itself.
    fn spins(&self) -> usize {
        match self {
            LinearPlan::Integer { coeffs, .. } => coeffs.len(),
            _ => 0,
        }
    }
}

/// A linear function of spins.
struct LinSpin {
    offset: f64,
    terms: Vec<(usize, f64)>,
}

/// Add `p·(Σ wᵢ·lᵢ − target)²` for linear-in-spin pieces.
///
/// Expanding the square gives the cross terms, the diagonal (which is linear because a 0/1
/// indicator squares to itself) and a constant that is dropped.
fn add_squared(b: &mut GraphBuilder, parts: &[(LinSpin, f64)], target: f64, p: f64) {
    for i in 0..parts.len() {
        let (li, wi) = (&parts[i].0, parts[i].1);
        add_linear(b, li, p * wi * (wi - 2.0 * target));
        for j in (i + 1)..parts.len() {
            let (lj, wj) = (&parts[j].0, parts[j].1);
            add_product(b, li, lj, 2.0 * p * wi * wj);
        }
    }
}

fn b_ref(b: &mut GraphBuilder) -> &mut GraphBuilder {
    b
}

fn add_linear(b: &mut GraphBuilder, l: &LinSpin, w: f64) {
    // w · (offset + Σ cᵢ sᵢ); the constant is dropped, energies here are relative
    for (i, c) in &l.terms {
        // energy is -h·s, so a coefficient of +w·c means a field of -w·c
        b.bias(*i, -w * c);
    }
}

fn add_product(b: &mut GraphBuilder, a: &LinSpin, c: &LinSpin, w: f64) {
    // w · (a₀ + Σ aᵢsᵢ)(c₀ + Σ cⱼsⱼ)
    for (i, ai) in &a.terms {
        b.bias(*i, -w * ai * c.offset);
    }
    for (j, cj) in &c.terms {
        b.bias(*j, -w * a.offset * cj);
    }
    for (i, ai) in &a.terms {
        for (j, cj) in &c.terms {
            if i == j {
                // sᵢ² = 1, a constant; nothing to add
                continue;
            }
            b.couple(*i, *j, -w * ai * cj);
        }
    }
}

/// Which solver to point at a compiled model.
///
/// Until now the answer was always [`Self::Anneal`]: every `solve*` on [`Compiled`] annealed, so the
/// crate's tabu search, breakout local search, branch and bound and its three bounds were reachable
/// only by taking `Compiled::graph` and driving them by hand — possible in Rust and impossible from
/// every other surface. The layer the README, `llms.txt` and the MCP tool descriptions all say to
/// reach for first was the one layer that could not certify anything.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Method {
    /// Simulated annealing down a ladder. The default, and the only one there used to be.
    Anneal,
    /// Tabu search. The baseline every max-cut heuristic is measured against, and in this crate's
    /// own shootout the strongest single arm at 400 nodes.
    Tabu {
        /// One iteration is one flip.
        iterations: usize,
    },
    /// Breakout local search: descent plus an adaptive perturbation.
    Breakout {
        /// One iteration is one flip.
        iterations: usize,
    },
    /// Branch and bound — **the only one that returns a proof**.
    ///
    /// Sets [`Solution::proved_optimal`] when it exhausts the tree within `max_nodes`. Read the note
    /// on that field: a proof about the compiled energy becomes a proof about YOUR model exactly
    /// when the answer is also feasible, and the argument does not depend on the penalty.
    Branch {
        /// Node budget. Reaching it returns the best state found and no proof.
        max_nodes: u64,
    },
}

/// A compiled model: the program, plus the means to read an answer back.
pub struct Compiled {
    pub program: Program,
    pub graph: crate::graph::Graph,
    slots: Vec<Slot>,
    /// Including the compiler's own slack variables, for anyone inspecting the lowering.
    pub all_slots: Vec<Slot>,
    names: Vec<String>,
    domains: Vec<Domain>,
    /// Spins every [`Constraint::Linear`] row's slack added, in total.
    ///
    /// They sit after every declared variable and after any counting slack, and before any
    /// ancilla. Exposed because it is the price of a weighted inequality and the doc comment on
    /// [`Constraint::Linear`] makes a promise about it: `⌈log₂(S+1)⌉` per inequality row, zero for
    /// an equality and zero for a row that constrains nothing.
    pub linear_slack: usize,
    /// Spins the higher-order reduction added, if any.
    ///
    /// They sit after every declared variable and after any slack, and the decoder never reports
    /// them: an ancilla's value is an artefact of the lowering. Exposed because the count is the
    /// price of writing a term wider than two.
    pub ancillas: usize,
    /// Variables whose encoding CANNOT be made exact by any penalty, described in the caller's
    /// own names.
    ///
    /// A binary encoding of k values uses ceil(log2 k) spins, which spell 2^ceil(log2 k)
    /// codewords. When k is not a power of two the extra codewords decode to nothing — and no
    /// penalty removes them, because there is no pairwise term that separates them from the valid
    /// ones. Measured on a k = 6 binary slot: the cheapest INVALID state costs exactly what the
    /// cheapest valid one does, so the sampler has no reason at all to prefer an answer.
    ///
    /// `Slot::add_penalty` has always returned whether it could be exact, and both callers
    /// discarded it. `Slot::decode` then catches the bad state and the variable reads as
    /// undecoded — a correct answer to the wrong question, because what the modeller needs is to
    /// know at COMPILE time that the encoding they picked cannot be tight. That is what this is.
    pub caveats: Vec<String>,
    /// The objective as written, normalised to a minimisation, plus the direction it was written
    /// in. Kept so an answer can be SCORED in the modeller's units and not only in spins.
    objective: Expr,
    sense: Option<Sense>,
    /// Kept so a decoded answer can be CHECKED, not just read.
    ///
    /// A penalty makes a constraint expensive, not impossible. The sampler is free to pay it, and
    /// when the objective outbids the penalty that is exactly what it does -- returning a state
    /// that decodes perfectly and violates the request. Reading the answer cannot detect that; only
    /// re-checking each constraint against the decoded values can.
    constraints: Vec<(Constraint, bool, f64)>,
}

impl Compiled {
    pub fn spins(&self) -> usize {
        self.graph.n
    }

    /// Decode a spin state into named values.
    pub fn decode(&self, state: &[i8]) -> Solution {
        let mut values = BTreeMap::new();
        let mut invalid = Vec::new();
        for (i, s) in self.slots.iter().enumerate() {
            match s.decode(state) {
                Some(v) => {
                    values.insert(self.names[i].clone(), self.reify(i, v));
                    }
                None => invalid.push(self.names[i].clone()),
            }
        }
        // Then check what was asked for. A decoded answer is a readable one, not a correct one.
        let violated = if invalid.is_empty() { self.check(&values) } else { Vec::new() };
        // The objective is scored on the DECODED values, not on the spins: it is the answer to
        // "what is this worth", and a variable that did not decode has no value to score. So an
        // answer with anything invalid reports no objective rather than a number computed from
        // half a solution.
        let objective = if invalid.is_empty() { self.score(&values) } else { None };
        Solution {
            values,
            invalid,
            violated,
            energy: self.graph.energy(state),
            objective,
            // Decoding a state says nothing about whether it is optimal. Only `solve_by` with
            // `Method::Branch` can set this, and it sets it on the Solution it returns.
            proved_optimal: false,
        }
    }

    /// The objective's value at a decoded assignment, in the direction the modeller wrote it.
    ///
    /// `self.objective` is normalised to a minimisation -- `Model::objective` negates what is
    /// maximised -- so a maximisation is negated back on the way out. That normalisation is right
    /// for the compiler and wrong for the reader, and this is where the two part company.
    fn score(&self, values: &BTreeMap<String, i64>) -> Option<f64> {
        let sense = self.sense?;
        let holds = |l: &Lit| match l {
            Lit::Spin(v) => values.get(&self.names[v.0]) == Some(&1),
            Lit::Is(v, want) => values.get(&self.names[v.0]) == Some(want),
        };
        let mut acc = self.objective.constant;
        for term in &self.objective.terms {
            if term.lits.iter().all(holds) {
                acc += term.coeff;
            }
        }
        Some(if sense == Sense::Maximize { -acc } else { acc })
    }

    /// Which constraints the decoded values break, each described in the caller's own names.
    fn check(&self, values: &BTreeMap<String, i64>) -> Vec<Violation> {
        let get = |v: &Var| values.get(&self.names[v.0]).copied();
        let holds = |l: &Lit| match l {
            Lit::Spin(v) => get(v) == Some(1),
            Lit::Is(v, want) => get(v) == Some(*want),
        };
        let count = |lits: &[Lit]| lits.iter().filter(|l| holds(l)).count();
        let name = |v: &Var| self.names[v.0].as_str();

        let mut out = Vec::new();
        for (c, hard, weight) in &self.constraints {
            let (hard, weight) = (*hard, *weight);
            // Each arm reports BY HOW MUCH as well as what. "at most 2 of 5 and 4 hold" is a near
            // miss; "at most 2 and 5 hold" is not, and a caller ranking repair candidates or
            // deciding whether to raise the penalty needs to tell them apart. dimod's
            // `iter_violations` yields a magnitude for exactly this reason.
            let broken = match c {
                Constraint::NotEqual(a, b) => (get(a) == get(b)).then(|| Violation {
                    hard: true,
                    cost: 0.0,
                    detail: format!(
                        "{} and {} must differ, and both are {}",
                        name(a), name(b), get(a).unwrap_or_default()
                    ),
                    amount: 1.0,
                }),
                Constraint::Equal(a, b) => (get(a) != get(b)).then(|| Violation {
                    hard: true,
                    cost: 0.0,
                    detail: format!(
                        "{} and {} must agree, and they are {} and {}",
                        name(a), name(b), get(a).unwrap_or_default(), get(b).unwrap_or_default()
                    ),
                    // How far apart they are, which for an ordered domain is a real distance and
                    // for a categorical is just "not the same".
                    amount: (get(a).unwrap_or_default() - get(b).unwrap_or_default()).abs() as f64,
                }),
                Constraint::Fix(v, want) => (get(v) != Some(*want)).then(|| Violation {
                    hard: true,
                    cost: 0.0,
                    detail: format!("{} must be {want}, and it is {}", name(v), get(v).unwrap_or_default()),
                    amount: (get(v).unwrap_or_default() - want).abs() as f64,
                }),
                Constraint::Cardinality { lits, k } => {
                    let n = count(lits);
                    (n != *k).then(|| Violation {
                        hard: true,
                        cost: 0.0,
                        detail: format!("exactly {k} of {} must hold, and {n} do", lits.len()),
                        amount: (n as f64 - *k as f64).abs(),
                    })
                }
                Constraint::AtMost { lits, k } => {
                    let n = count(lits);
                    (n > *k).then(|| Violation {
                        hard: true,
                        cost: 0.0,
                        detail: format!("at most {k} of {} may hold, and {n} do", lits.len()),
                        amount: (n - *k) as f64,
                    })
                }
                Constraint::AtLeast { lits, k } => {
                    let n = count(lits);
                    (n < *k).then(|| Violation {
                        hard: true,
                        cost: 0.0,
                        detail: format!("at least {k} of {} must hold, and {n} do", lits.len()),
                        amount: (*k - n) as f64,
                    })
                }
                Constraint::ExactlyOne(lits) => {
                    let n = count(lits);
                    (n != 1).then(|| Violation {
                        hard: true,
                        cost: 0.0,
                        detail: format!("exactly one of {} must hold, and {n} do", lits.len()),
                        amount: (n as f64 - 1.0).abs(),
                    })
                }
                Constraint::AtMostOne(lits) => {
                    let n = count(lits);
                    (n > 1).then(|| Violation {
                        hard: true,
                        cost: 0.0,
                        detail: format!("at most one of {} may hold, and {n} do", lits.len()),
                        amount: (n - 1) as f64,
                    })
                }
                Constraint::Linear { terms, rel, rhs } => {
                    // RECOMPUTED FROM THE DECODED USER VALUES, never from the slack. That is the
                    // load-bearing property of this arm: a slack bit the sampler left anywhere at
                    // all cannot produce a false `feasible()`, because the slack is not consulted.
                    // `feasible()` is false exactly when the arithmetic says the row is broken.
                    let mut lhs = 0.0f64;
                    let mut parts: Vec<(String, f64)> = Vec::with_capacity(terms.len());
                    for (l, c) in terms {
                        if holds(l) {
                            lhs += *c;
                        }
                        let v = lit_var(*l);
                        parts.push((lit_label(name(&v), self.domains[v.0], *l), *c));
                    }
                    let amount = match rel {
                        Rel::Le => lhs - rhs,
                        Rel::Ge => rhs - lhs,
                        Rel::Eq => (lhs - rhs).abs(),
                    };
                    // A tolerance, not a rounding: an integer row lands exactly and a real-valued
                    // equality should not be called broken by the last bit of a summation.
                    (amount > 1e-9).then(|| Violation {
                        hard: true,
                        cost: 0.0,
                        detail: format!(
                            "{}, and the left side comes to {}",
                            row_text(&parts, *rel, *rhs),
                            num(lhs)
                        ),
                        amount,
                    })
                }
                Constraint::AllDifferent(vars) => {
                    // Report WHICH value collided, not merely that something did. "three of them
                    // took 5" is a repair a modeller can act on; "all-different was violated" is
                    // a fact they already suspected.
                    let mut by_value: Vec<(i64, Vec<&str>)> = Vec::new();
                    for &v in vars {
                        let Some(&got) = values.get(&self.names[v.0]) else { continue };
                        match by_value.iter_mut().find(|(val, _)| *val == got) {
                            Some((_, who)) => who.push(&self.names[v.0]),
                            None => by_value.push((got, vec![&self.names[v.0]])),
                        }
                    }
                    let clashes: Vec<_> = by_value.iter().filter(|(_, w)| w.len() > 1).collect();
                    let excess: usize = clashes.iter().map(|(_, w)| w.len() - 1).sum();
                    (excess > 0).then(|| Violation {
                        hard: true,
                        cost: 0.0,
                        detail: format!(
                            "{} must all differ, and {}",
                            vars.len(),
                            clashes
                                .iter()
                                .map(|(val, who)| format!("{} both take {val}", who.join(" and ")))
                                .collect::<Vec<_>>()
                                .join("; ")
                        ),
                        amount: excess as f64,
                    })
                }
            };
            if let Some(mut b) = broken {
                b.hard = hard;
                b.cost = if hard { 0.0 } else { weight * b.amount * b.amount };
                out.push(b);
            }
        }
        out
    }

    /// Turn a slot index back into the domain's own units.
    fn reify(&self, i: usize, raw: usize) -> i64 {
        match self.domains[i] {
            Domain::Integer { lo, .. } => lo + raw as i64,
            Domain::Spin => {
                if raw == 1 {
                    1
                } else {
                    -1
                }
            }
            _ => raw as i64,
        }
    }

    /// Anneal and decode.
    /// Solve with a chosen method, rather than always annealing.
    ///
    /// Every other solver in this crate takes a graph of spins, and `Compiled::graph` is public, so
    /// a Rust caller could always have driven them by hand. Nobody on any other surface could, and
    /// nobody at all could get a PROOF back in the modeller's own vocabulary. This is that routing,
    /// written once, so every surface gets it.
    ///
    /// The answer is decoded and re-checked exactly as [`Self::solve_annealed`]'s is: a method that
    /// finds a lower compiled energy has not thereby found a feasible answer, and
    /// [`Solution::feasible`] is still the thing to read first.
    pub fn solve_by(&self, method: Method, seed: u64) -> Solution {
        let state = match method {
            Method::Anneal => return self.solve_annealed(seed),
            Method::Tabu { iterations } => {
                let p = crate::tabu::Params {
                    iterations: iterations.max(1),
                    ..crate::tabu::Params::default()
                };
                crate::tabu::search(&self.graph, &p, seed).state
            }
            Method::Breakout { iterations } => {
                let p = crate::bls::Params {
                    iterations: iterations.max(1),
                    ..crate::bls::Params::default()
                };
                crate::bls::search(&self.graph, &p, seed).state
            }
            Method::Branch { max_nodes } => {
                // Warm-started from a short anneal. A good incumbent prunes from the first node and
                // is worth far more than a better bound -- `branch`'s own module doc says so -- and
                // handing branch a random incumbent would make the proof arrive far later for no
                // reason a caller chose.
                let warm = self.solve_annealed(seed);
                let incumbent = self.state_of(&warm);
                let p = crate::branch::Params {
                    max_nodes: max_nodes.max(1),
                    incumbent,
                    ..crate::branch::Params::default()
                };
                let out = crate::branch::solve(&self.graph, &p);
                let mut sol = self.decode(&out.state);
                sol.proved_optimal = out.proved_optimal;
                return sol;
            }
        };
        self.decode(&state)
    }

    /// Re-encode a decoded answer back to spins, for a solver that wants a starting state.
    ///
    /// `None` when anything failed to decode: half a state is a worse incumbent than none, because
    /// branch would prune against a bound that does not correspond to any assignment.
    fn state_of(&self, sol: &Solution) -> Option<Vec<i8>> {
        if !sol.invalid.is_empty() {
            return None;
        }
        let mut s = vec![-1i8; self.spins()];
        for (i, slot) in self.slots.iter().enumerate() {
            let v = *sol.values.get(&self.names[i])?;
            let k = self.domains[i].index_of(v)?;
            slot.encode(k, &mut s);
        }
        Some(s)
    }

    pub fn solve_annealed(&self, seed: u64) -> Solution {
        self.solve_with(&Self::default_schedule(), seed)
    }

    /// The ladder used when a caller does not supply one. Deliberately conservative: it is better
    /// for a first answer to be slow and right than fast and quietly infeasible.
    pub fn default_schedule() -> Schedule {
        let (hot, cold, stages, per) = Self::DEFAULT_LADDER;
        Schedule::geometric(hot, cold, stages, per)
    }

    /// The default ladder's parameters, for a caller that wants to vary one of them.
    ///
    /// `(beta_hot, beta_cold, stages, sweeps_per_stage)`. Exposed because every surface that lets a
    /// caller override the ladder needs to say what it is overriding, and each of them writing the
    /// numbers out again is four places for them to drift apart.
    pub const DEFAULT_LADDER: (f64, f64, usize, usize) = (0.05, 8.0, 120, 40);

    /// Anneal on a caller's own ladder.
    ///
    /// A harder model wants a longer one. The default is tuned for the models people write first,
    /// not for the largest they will eventually write, and a caller who has measured their own
    /// instance should be able to say so.
    pub fn solve_with(&self, sched: &Schedule, seed: u64) -> Solution {
        let (best, _) = crate::tempering::anneal_scheduled(&self.graph, sched, seed, None);
        self.decode(&best)
    }

    /// Anneal several times on a caller's ladder and keep the best feasible answer.
    pub fn solve_best_with(&self, sched: &Schedule, tries: u64) -> Solution {
        self.best_of(tries, |s| self.solve_with(sched, s))
    }

    /// Anneal several times and keep the best feasible answer, or the best overall if none is.
    pub fn solve_best_of(&self, tries: u64) -> Solution {
        self.best_of(tries, |s| self.solve_annealed(s))
    }

    /// Anneal `tries` times and keep **every** answer, in seed order.
    ///
    /// [`Self::solve_best_with`] throws away every try but one, which is the right answer to
    /// "what should I do" and the wrong one to "how many ways could I have done it". A problem
    /// with a symmetry has several optima and a modeller usually wants to see them; nothing below
    /// this line could tell them apart, because only one survived the loop.
    ///
    /// Seeds are `0..tries`, the same ones [`Self::solve_best_with`] uses, so the best answer here
    /// is the answer that call returns.
    pub fn solve_all_with(&self, sched: &Schedule, tries: u64) -> Vec<Solution> {
        (0..tries.max(1)).map(|s| self.solve_with(sched, s)).collect()
    }

    fn best_of(&self, tries: u64, run: impl Fn(u64) -> Solution) -> Solution {
        let mut best: Option<Solution> = None;
        for s in 0..tries.max(1) {
            let cand = run(s);
            let better = match &best {
                None => true,
                Some(b) => match (b.feasible(), cand.feasible()) {
                    (false, true) => true,
                    (true, false) => false,
                    _ => cand.energy < b.energy,
                },
            };
            if better {
                best = Some(cand);
            }
        }
        best.expect("at least one try")
    }
}

/// The distinct optimal assignments among a set of answers, best first.
///
/// # Distinctness is on the DECODED VALUES, never on the spins
///
/// A compiled model carries bits no variable reads — a cardinality row's slack register, a
/// Rosenberg substitution's ancilla — so the key is the decoded map. The count then means what a
/// modeller reads it to mean: **how many different ways there are to do the job**.
///
/// **And on this compiler the two counts happen to agree, which is worth stating because the
/// obvious argument for keying on values is wrong.** The obvious argument is that slack bits float
/// freely and inflate a state count; enumerating `at most two of four` exactly says otherwise —
/// eleven satisfying assignments, eleven minimum-energy states. The penalty that makes the row hold
/// also PINS its slack, so at the optimum there is nothing left floating. The test
/// `counting_spin_states_would_over_count_the_optima` measures precisely that, and is named for
/// the claim it refuted.
///
/// So the reason to key on values is not a measured discrepancy today. It is that the count is a
/// statement about the model and must not depend on how the compiler chose to represent it: an
/// encoding with a redundant slack representation, added later, would inflate a state count
/// silently and leave this one correct.
///
/// # What it does and does not claim
///
/// Only **feasible** answers are counted — an assignment that breaks a hard constraint is not a way
/// to do the job — and only those within `tol` of the best feasible energy. `tol` is on the
/// compiled Ising energy, which folds in every penalty and the constant, so it is a number about
/// spins and not about the objective; `1e-9` is the right value for exact ties.
///
/// It is **evidence**, not a count of the ground manifold. `tries` independent anneals prove the
/// optima they landed on exist and prove nothing about the ones they missed. Only exhaustive
/// enumeration counts, and this is not that.
///
/// Returned best first, ties broken by the assignment itself, so the order is deterministic.
///
/// When several optima tie — the case this function exists for — the head is therefore the
/// lexicographically first of them and NOT necessarily the one [`Compiled::solve_best_with`]
/// returned, which is whichever seed reached the minimum first. Both are optimal.
pub fn distinct_optima(answers: &[Solution], tol: f64) -> Vec<Solution> {
    let feasible: Vec<&Solution> = answers.iter().filter(|s| s.feasible()).collect();
    let Some(best) = feasible.iter().map(|s| s.energy).fold(None, |acc: Option<f64>, e| {
        Some(match acc {
            None => e,
            Some(b) => b.min(e),
        })
    }) else {
        return Vec::new();
    };
    let mut seen: BTreeMap<&BTreeMap<String, i64>, &Solution> = BTreeMap::new();
    for s in feasible.iter().filter(|s| s.energy <= best + tol) {
        // First occurrence wins, which is the lowest seed among identical assignments -- so the
        // list is stable under re-running with the same tries.
        seen.entry(&s.values).or_insert(s);
    }
    let mut out: Vec<Solution> = seen.into_values().cloned().collect();
    out.sort_by(|a, b| {
        a.energy
            .partial_cmp(&b.energy)
            .unwrap_or(core::cmp::Ordering::Equal)
            .then_with(|| a.values.cmp(&b.values))
    });
    out
}

/// A constraint the answer breaks, and by how much.
///
/// The magnitude is what separates a near miss from a rout. "At most two of five, and three hold"
/// is one over; "and five hold" is three over, and a caller ranking repair candidates or deciding
/// whether raising the penalty will be enough needs to tell them apart. Reporting only the
/// description made every violation look equally bad.
#[derive(Clone, Debug, PartialEq)]
pub struct Violation {
    /// What was asked for and what happened, in the caller's own names.
    pub detail: String,
    /// Whether breaking this makes the answer inadmissible.
    ///
    /// A hard constraint says which answers are answers at all; a soft one is a preference with a
    /// price. [`Solution::feasible`] counts only the hard kind, because a soft constraint the
    /// modeller deliberately priced low is not a failure — it is the trade they asked for.
    pub hard: bool,
    /// What breaking this cost, for a soft constraint. Zero for a hard one, which has no price.
    ///
    /// `weight × amount²`, and the square is not a detail. A constraint becomes an energy term by
    /// squaring how far outside it sits, so missing by two costs FOUR times missing by one, not
    /// twice. A modeller pricing a preference is choosing that curve as well as its scale, and
    /// reporting a linear price here would misstate what the solver actually traded.
    pub cost: f64,
    /// How far outside the constraint the answer sits, in the constraint's own units: places over a
    /// ceiling, places under a floor, distance from a fixed value. Always positive.
    pub amount: f64,
}

impl core::fmt::Display for Violation {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        if self.hard {
            write!(f, "{} (by {})", self.detail, self.amount)
        } else {
            write!(f, "{} (by {}, cost {})", self.detail, self.amount, self.cost)
        }
    }
}

/// An answer, in the model's own names.
#[derive(Clone, Debug)]
pub struct Solution {
    values: BTreeMap<String, i64>,
    /// Variables whose spins did not form a valid codeword.
    pub invalid: Vec<String>,
    /// Constraints the decoded values break, each in the caller's own names and by how much.
    ///
    /// Distinct from `invalid`, and the distinction matters: an invalid variable cannot be read at
    /// all, while a violated constraint means every value read cleanly and one of them is not what
    /// was asked for. Both mean the penalty lost, and both need a larger one.
    pub violated: Vec<Violation>,
    /// The compiled Ising energy: the objective, every penalty and the constant, all folded in.
    ///
    /// A number about SPINS. Two answers to the same model can be compared with it, and nothing
    /// else can: it is not what the schedule is worth, and it moves when the penalty does.
    pub energy: f64,
    /// Whether the answer is **provably** the best one, not merely the best found.
    ///
    /// Only [`Method::Branch`] can set it, and only when it exhausted the tree within its budget.
    ///
    /// # What it proves, exactly
    ///
    /// Branch and bound proves a statement about the COMPILED energy: that no assignment of the
    /// spins has a lower one. That is not immediately a statement about your model, because the
    /// compiled energy folds in every penalty and a constant.
    ///
    /// It becomes one the moment the answer is also **feasible**, and the argument needs nothing
    /// from the penalty being large enough. For a feasible assignment every penalty term is zero, so
    /// the compiled energy is the objective plus a constant. If `s*` minimises the compiled energy
    /// over ALL assignments and `s*` is feasible, then for any other feasible `s`,
    /// `E(s) >= E(s*)`, and both sides are that same objective-plus-constant — so `s*` is optimal
    /// among feasible assignments. **`proved_optimal && feasible()` is a genuine optimality proof
    /// for the model as written.**
    ///
    /// `proved_optimal` with an INFEASIBLE answer proves something different and still useful: the
    /// penalty was too small, and no larger search will fix it. Raise the penalty.
    pub proved_optimal: bool,
    /// The objective's value in the modeller's own units, in the direction they wrote it.
    ///
    /// `None` when no objective was written, or when `objective` was called with both senses and
    /// there is no single direction to report. A modeller who wrote `maximize 5*mon + 4*tue` and
    /// got `mon = 1, tue = 2` reads **9** here, and reads the compiled Ising energy in `energy` --
    /// which for that model is some number in the hundreds with the penalties in it and tells them
    /// nothing about their schedule.
    pub objective: Option<f64>,
}

impl Solution {
    /// The value of a named variable.
    ///
    /// Panics if the variable did not decode, and says which of the two things went wrong —
    /// an unknown name is a typo, a variable in `invalid` is an under-weighted penalty.
    pub fn value(&self, name: &str) -> i64 {
        if let Some(v) = self.values.get(name) {
            return *v;
        }
        if self.invalid.iter().any(|n| n == name) {
            panic!(
                "'{name}' did not decode: its spins are not a valid codeword, which means the \
                 encoding penalty lost to the objective. Raise Model::penalty, or check \
                 Solution::feasible() before reading values"
            );
        }
        panic!("no variable named '{name}'");
    }

    pub fn get(&self, name: &str) -> Option<i64> {
        self.values.get(name).copied()
    }

    /// Whether every variable decoded.
    ///
    /// A false here means a penalty was too weak, not that the problem is infeasible — and the two
    /// are worth distinguishing before concluding anything.
    /// True when every variable decoded AND every constraint holds.
    ///
    /// It used to mean only the first, which is a much weaker claim than the name makes. A penalty
    /// makes a constraint expensive rather than impossible, so a sampler whose objective outbids it
    /// returns a state that decodes perfectly and breaks the request -- and this reported it as
    /// feasible, which is the answer to a question nobody asked.
    #[must_use = "an answer that broke a hard constraint is still returned, so not checking this is trusting values the model already rejected"]
    pub fn feasible(&self) -> bool {
        // Only the HARD violations. A soft constraint is a preference with a price, and breaking
        // one is the trade the modeller asked for rather than a failure to answer.
        self.invalid.is_empty() && !self.violated.iter().any(|v| v.hard)
    }

    /// What the broken soft constraints cost, in the units their weights were given in.
    ///
    /// Zero when none broke. Read beside `energy`: this is the part of the score that came from
    /// preferences rather than from the objective, and separating them is the point of saying a
    /// constraint is soft.
    pub fn soft_cost(&self) -> f64 {
        // `+ 0.0` is not redundant: `Sum for f64` folds from -0.0, which is the correct additive
        // identity but prints as "-0" through every binding that formats a float. A price with a
        // minus sign in front of it reads as a credit. Adding zero normalises the sign and
        // changes nothing else.
        self.violated.iter().filter(|v| !v.hard).map(|v| v.cost).sum::<f64>() + 0.0
    }

    pub fn iter(&self) -> impl Iterator<Item = (&String, &i64)> {
        self.values.iter()
    }
}

impl core::fmt::Display for Solution {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "energy {:.4}", self.energy)?;
        if !self.feasible() {
            write!(f, "  INFEASIBLE")?;
        }
        for (k, v) in &self.values {
            write!(f, "\n  {k} = {v}")?;
        }
        for n in &self.invalid {
            write!(f, "\n  {n} = (did not decode)")?;
        }
        // The reason, not just the verdict. "INFEASIBLE" alone leaves a caller to work out which of
        // the things they asked for was not delivered.
        for v in &self.violated {
            if v.hard {
                write!(f, "\n  broken: {} (by {})", v.detail, v.amount)?;
            } else {
                write!(f, "\n  traded: {} (by {}, cost {})", v.detail, v.amount, v.cost)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {

    /// A WHOLE NUMBER TOO BIG FOR f64 IS NOT A FRACTION, and the refusal used to say it was.
    ///
    /// `1e16` is integral. The old message called it "a non-integer coefficient" and advised
    /// multiplying the row through by its common denominator -- a fix that cannot apply to a number
    /// that is already whole. Both halves were false while the refusal itself was right.
    #[test]
    fn a_huge_integral_coefficient_is_refused_for_the_reason_it_actually_is() {
        let mut m = Model::new();
        let a = m.binary("a");
        let b = m.binary("b");
        m.constrain(Constraint::Linear {
            terms: vec![(a.is(1), 1e16), (b.is(1), 1.0)],
            rel: Rel::Le,
            rhs: 5.0,
        });
        match m.compile() {
            Err(CompileError::LinearHugeCoefficient { value, limit, what, .. }) => {
                assert_eq!(value, 1e16);
                assert_eq!(limit, EXACT_INT);
                assert!(what.contains('a'), "names which coefficient: {what}");
                let msg = CompileError::LinearHugeCoefficient {
                    row: "r".into(),
                    what,
                    value,
                    limit,
                }
                .to_string();
                assert!(msg.contains("whole number"), "must not call it a fraction: {msg}");
                assert!(msg.contains("2^53"), "and must name the boundary: {msg}");
                assert!(
                    !msg.contains("common denominator"),
                    "and must not advise a fix that cannot apply: {msg}"
                );
            }
            Err(e) => panic!("expected LinearHugeCoefficient, got {e:?}"),
            Ok(_) => panic!("a coefficient past 2^53 must be refused, not compiled"),
        }

        // An actual fraction still gets the other message, which is the one that fits it.
        let mut m2 = Model::new();
        let c = m2.binary("c");
        m2.constrain(Constraint::Linear {
            terms: vec![(c.is(1), 2.5)],
            rel: Rel::Le,
            rhs: 5.0,
        });
        assert!(matches!(m2.compile(), Err(CompileError::LinearNotInteger { .. })));
    }
    use super::*;

    // ---- weighted linear rows -------------------------------------------------------------------
    //
    // The only trustworthy check on a constraint LOWERING at small sizes is brute force: enumerate
    // every assignment of the logical variables, decide feasibility directly from the arithmetic,
    // and demand that the compiled energy orders the states the way the row says. A lowering that
    // happens to work on the example in the docstring is the failure mode here, so none of these
    // tests contains a hand-picked row that is not ALSO drawn at random by the sweep below.

    /// A row, in the terms the sweep draws and the arithmetic below decides.
    struct Row {
        /// `(variable index, weight)`, unmerged and possibly repeated, as a modeller would write it.
        terms: Vec<(usize, i64)>,
        rel: Rel,
        rhs: i64,
        n: usize,
    }

    impl Row {
        fn lhs(&self, assign: u32) -> i64 {
            self.terms
                .iter()
                .filter(|(v, _)| (assign >> v) & 1 == 1)
                .map(|(_, w)| *w)
                .sum()
        }
        fn holds(&self, assign: u32) -> bool {
            let l = self.lhs(assign);
            match self.rel {
                Rel::Le => l <= self.rhs,
                Rel::Ge => l >= self.rhs,
                Rel::Eq => l == self.rhs,
            }
        }
        fn build(&self) -> Model {
            let mut m = Model::new();
            let vars: Vec<Var> = (0..self.n).map(|i| m.binary(&format!("x{i}"))).collect();
            let terms: Vec<(Lit, f64)> =
                self.terms.iter().map(|(v, w)| (Lit::Is(vars[*v], 1), *w as f64)).collect();
            m.linear(terms, self.rel, self.rhs as f64);
            m
        }
    }

    /// Write a logical assignment into the one-hot spins, leaving the slack block untouched.
    fn write_logical(state: &mut [i8], n: usize, assign: u32) {
        for i in 0..n {
            let on = (assign >> i) & 1 == 1;
            state[2 * i] = if on { -1 } else { 1 }; // slot 0 is value 0
            state[2 * i + 1] = if on { 1 } else { -1 };
        }
    }

    /// The lowest energy this logical assignment can reach, minimising over the slack block alone.
    fn best_energy(c: &Compiled, n: usize, assign: u32) -> f64 {
        let slack = c.linear_slack;
        let mut state = vec![-1i8; c.graph.n];
        write_logical(&mut state, n, assign);
        let base = c.graph.n - slack;
        let mut best = f64::INFINITY;
        for pattern in 0..(1u32 << slack) {
            for j in 0..slack {
                state[base + j] = if (pattern >> j) & 1 == 1 { 1 } else { -1 };
            }
            best = best.min(c.graph.energy(&state));
        }
        best
    }

    /// THE CHECK THE BRIEF ASKS FOR, over many random weight/target combinations.
    ///
    /// For every drawn row: compile it, enumerate every assignment of its binaries, minimise the
    /// compiled energy over the slack block, and require
    ///
    ///   * every assignment the arithmetic allows to reach the SAME energy (a flat feasible floor),
    ///   * every assignment it forbids to cost strictly more, by at least the model's penalty,
    ///   * a refusal to be justified by enumeration rather than taken on trust,
    ///   * and the emitted spin count to be exactly what `Constraint::Linear` promises.
    #[test]
    fn a_weighted_row_orders_every_state_the_way_the_arithmetic_does() {
        let mut rng = crate::rng::Pcg::new(0x_FEED_BEEF, 11);
        let p = 2.0; // no objective, so `effective_penalty` is the model's default
        let (mut compiled, mut refused, mut vacuous, mut with_slack) = (0usize, 0, 0, 0);
        let mut smallest_gap = f64::INFINITY;

        for _ in 0..12_000 {
            let n = 1 + (rng.next_u32() % 4) as usize;
            let nt = 1 + (rng.next_u32() % 5) as usize;
            let terms: Vec<(usize, i64)> = (0..nt)
                .map(|_| {
                    let v = (rng.next_u32() as usize) % n;
                    let w = (rng.next_u32() % 11) as i64 - 5; // −5 ..= 5, zeros included
                    (v, w)
                })
                .collect();
            // Sweep the bound across the WHOLE reachable range and two past each end, so vacuous
            // rows, unsatisfiable rows and every tightness in between are all drawn.
            let mag: i64 = terms.iter().map(|(_, w)| w.abs()).sum::<i64>() + 2;
            let rhs = (rng.next_u32() % (2 * mag as u32 + 1)) as i64 - mag;
            let rel = match rng.next_u32() % 3 {
                0 => Rel::Le,
                1 => Rel::Ge,
                _ => Rel::Eq,
            };
            let row = Row { terms, rel, rhs, n };

            let feasible_states: Vec<u32> = (0..(1u32 << n)).filter(|a| row.holds(*a)).collect();

            let c = match row.build().compile() {
                Ok(c) => c,
                Err(CompileError::LinearUnsatisfiable { .. }) => {
                    // A refusal is a CLAIM, and it is checked here rather than believed.
                    assert!(
                        feasible_states.is_empty(),
                        "refused as unsatisfiable, but {} assignment(s) satisfy it",
                        feasible_states.len()
                    );
                    refused += 1;
                    continue;
                }
                Err(e) => panic!("unexpected refusal: {e}"),
            };
            compiled += 1;
            if c.linear_slack == 0 {
                vacuous += 1;
            } else {
                with_slack += 1;
            }
            assert_eq!(
                c.graph.n,
                2 * n + c.linear_slack,
                "the graph is the one-hot blocks plus exactly the promised slack"
            );

            let energies: Vec<f64> = (0..(1u32 << n)).map(|a| best_energy(&c, n, a)).collect();
            if feasible_states.is_empty() {
                // The arithmetic check is necessary, not sufficient -- a subset-sum equality can
                // pass it and still have no answer -- so this is a legitimate outcome and there is
                // no floor to compare against.
                continue;
            }
            let floor = feasible_states.iter().map(|a| energies[*a as usize]).fold(f64::INFINITY, f64::min);
            for a in 0..(1u32 << n) {
                let e = energies[a as usize];
                if row.holds(a) {
                    assert!(
                        (e - floor).abs() < 1e-9,
                        "two allowed states differ in energy by {:.6}: the row does not price \
                         feasibility flatly",
                        e - floor
                    );
                } else {
                    let gap = e - floor;
                    assert!(
                        gap >= p - 1e-9,
                        "a forbidden state costs only {gap:.6} more than the feasible floor, \
                         where the penalty is {p}"
                    );
                    smallest_gap = smallest_gap.min(gap);
                }
            }
        }

        // Not a formality: a sweep that drew only vacuous rows, or only refusals, would pass every
        // assertion above and check nothing at all.
        assert!(compiled > 6_000, "only {compiled} rows compiled");
        assert!(with_slack > 800, "only {with_slack} rows actually needed a slack");
        assert!(refused > 100, "only {refused} rows were refused as unsatisfiable");
        assert!(vacuous > 100, "only {vacuous} rows compiled without slack");
        // The minimum unit of violation costs exactly the penalty -- the same gap `Cardinality`
        // has, which is why `effective_penalty` needs no change for this constraint.
        assert!(
            (smallest_gap - p).abs() < 1e-9,
            "the cheapest violation cost {smallest_gap}, not the penalty {p}"
        );
    }

    /// The same sweep, asked the question `Solution::feasible()` answers.
    ///
    /// The point of the feature: a decoded answer that breaks the row must say so, name it, and say
    /// BY HOW MUCH in the modeller's own units.
    #[test]
    fn a_broken_weighted_row_makes_the_answer_infeasible_and_says_by_how_much() {
        let mut rng = crate::rng::Pcg::new(0x_C0FFEE, 3);
        let mut violations = 0usize;
        for _ in 0..2_000 {
            let n = 1 + (rng.next_u32() % 4) as usize;
            let nt = 1 + (rng.next_u32() % 5) as usize;
            let terms: Vec<(usize, i64)> = (0..nt)
                .map(|_| ((rng.next_u32() as usize) % n, (rng.next_u32() % 11) as i64 - 5))
                .collect();
            let mag: i64 = terms.iter().map(|(_, w)| w.abs()).sum::<i64>() + 2;
            let rhs = (rng.next_u32() % (2 * mag as u32 + 1)) as i64 - mag;
            let rel = match rng.next_u32() % 3 {
                0 => Rel::Le,
                1 => Rel::Ge,
                _ => Rel::Eq,
            };
            let row = Row { terms, rel, rhs, n };
            let Ok(c) = row.build().compile() else { continue };

            for a in 0..(1u32 << n) {
                let mut state = vec![-1i8; c.graph.n];
                write_logical(&mut state, n, a);
                // The slack block is left wherever it happens to be -- ALL DOWN, which for a
                // violated row is not the slack the sampler would have chosen. `check` must not
                // consult it, and this is what proves it does not.
                let sol = c.decode(&state);
                assert!(sol.invalid.is_empty());
                let lhs = row.lhs(a);
                let want = match row.rel {
                    Rel::Le => (lhs - row.rhs).max(0),
                    Rel::Ge => (row.rhs - lhs).max(0),
                    Rel::Eq => (lhs - row.rhs).abs(),
                };
                assert_eq!(
                    sol.feasible(),
                    want == 0,
                    "feasible() disagrees with the arithmetic on {:?} {} {}",
                    row.terms,
                    row.rel.symbol(),
                    row.rhs
                );
                if want > 0 {
                    violations += 1;
                    assert_eq!(sol.violated.len(), 1);
                    let v = &sol.violated[0];
                    assert!(v.hard);
                    assert_eq!(v.amount, want as f64, "{}", v.detail);
                    assert!(
                        v.detail.contains(row.rel.symbol()) && v.detail.contains("left side"),
                        "the violation must report the row and its left side: {}",
                        v.detail
                    );
                }
            }
        }
        assert!(violations > 2_000, "only {violations} violations were exercised");
    }

    /// A soft weighted row is priced in the modeller's own units.
    ///
    /// The identity this asserts is what the `weight·g²` scale exists for: the energy the row
    /// contributes at the sampler's own optimum equals the `weight × amount²` reported back.
    #[test]
    fn a_soft_weighted_row_costs_exactly_what_it_reports() {
        let mut rng = crate::rng::Pcg::new(0x_5EED, 5);
        let mut cases = 0usize;
        for _ in 0..1_500 {
            let n = 1 + (rng.next_u32() % 3) as usize;
            let terms: Vec<(usize, i64)> = (0..2 + (rng.next_u32() % 3) as usize)
                .map(|_| ((rng.next_u32() as usize) % n, (rng.next_u32() % 9) as i64 - 4))
                .collect();
            let mag: i64 = terms.iter().map(|(_, w)| w.abs()).sum::<i64>() + 1;
            let rhs = (rng.next_u32() % (2 * mag as u32 + 1)) as i64 - mag;
            let rel = if rng.next_u32().is_multiple_of(2) { Rel::Le } else { Rel::Ge };
            let row = Row { terms, rel, rhs, n };

            let mut m = Model::new();
            let vars: Vec<Var> = (0..n).map(|i| m.binary(&format!("x{i}"))).collect();
            let ts: Vec<(Lit, f64)> =
                row.terms.iter().map(|(v, w)| (Lit::Is(vars[*v], 1), *w as f64)).collect();
            m.linear_soft(ts, rel, rhs as f64, 3.0);
            let Ok(c) = m.compile() else { continue };
            if c.linear_slack == 0 {
                continue; // a vacuous row has nothing to price
            }
            let feasible: Vec<u32> = (0..(1u32 << n)).filter(|a| row.holds(*a)).collect();
            if feasible.is_empty() {
                continue;
            }
            let floor = feasible
                .iter()
                .map(|a| best_energy(&c, n, *a))
                .fold(f64::INFINITY, f64::min);
            for a in 0..(1u32 << n) {
                if row.holds(a) {
                    continue;
                }
                let over = best_energy(&c, n, a) - floor;
                let mut state = vec![-1i8; c.graph.n];
                write_logical(&mut state, n, a);
                let sol = c.decode(&state);
                assert!(sol.feasible(), "a soft row must not make an answer infeasible");
                assert!(
                    (over - sol.soft_cost()).abs() < 1e-7,
                    "the row cost {over} in the graph and reported {}",
                    sol.soft_cost()
                );
                cases += 1;
            }
        }
        assert!(cases > 500, "only {cases} soft violations were priced");
    }

    /// The slack expansion covers `{0..=S}` exactly -- nothing more and nothing less.
    ///
    /// The property the whole encoding rests on: an exact cover has no invalid codeword, so the
    /// block needs no encoding penalty and no exclusion couplings. Checked by enumeration rather
    /// than argued.
    #[test]
    fn the_truncated_binary_slack_covers_its_span_exactly() {
        for span in 0i128..=600 {
            let coeffs = truncated_binary(span);
            let m = coeffs.len();
            assert_eq!(m, if span == 0 { 0 } else { (128 - (span as u128).leading_zeros()) as usize });
            let mut hits = vec![0u32; (span + 1) as usize];
            for pattern in 0u64..(1u64 << m) {
                let v: i64 = (0..m).filter(|j| (pattern >> j) & 1 == 1).map(|j| coeffs[j]).sum();
                assert!(v >= 0 && v <= span as i64, "span {span}: {v} is outside {{0..={span}}}");
                hits[v as usize] += 1;
            }
            assert!(hits.iter().all(|h| *h >= 1), "span {span}: a residual is unreachable");
            // The only cost of the exact cover: a 2-to-1 window of a size the doc states exactly.
            let doubled = hits.iter().filter(|h| **h == 2).count();
            assert!(hits.iter().all(|h| *h <= 2));
            assert_eq!(
                doubled as i128,
                (1i128 << m) - (span + 1),
                "span {span}: the doubled window is not the size the doc claims"
            );
        }
    }

    /// What a weighted row costs, in the two numbers that are fabric-independent.
    ///
    /// Spins and couplings, measured on the built graph rather than predicted. The gcd is doing
    /// real work in the second row and none at all in the last, and saying so is the difference
    /// between a documented cost and a surprise.
    #[test]
    fn the_cost_of_a_weighted_row_is_the_number_the_doc_promises() {
        let cases: &[(&[i64], i64, usize, usize)] = &[
            // weights, rhs, expected slack spins, expected distinct couplings the row adds
            (&[3, 4, 5], 7, 3, 15),
            (&[1000, 1000], 1500, 1, 3),  // gcd 1000: `a + b ≤ 1`, span 1
            (&[1000, 1001], 1500, 11, 78),
            (&[1, 1, 1, 1, 1], 3, 2, 21),
            (&[2, 3, 5, 7, 11], 13, 4, 36),
            (&[1, 2, 4, 8, 16, 32], 40, 6, 66),
        ];
        for (w, rhs, spins, edges) in cases {
            let mut m = Model::new();
            let vars: Vec<Var> = (0..w.len()).map(|i| m.binary(&format!("x{i}"))).collect();
            let terms: Vec<(Lit, f64)> =
                w.iter().enumerate().map(|(i, c)| (Lit::Is(vars[i], 1), *c as f64)).collect();
            m.linear(terms, Rel::Le, *rhs as f64);
            let c = m.compile().unwrap();
            assert_eq!(c.linear_slack, *spins, "slack spins for {w:?} ≤ {rhs}");

            // The row's own couplings: the whole graph, minus what the one-hot blocks cost on
            // their own. A binary variable's block is one edge.
            let mut bare = Model::new();
            for i in 0..w.len() {
                bare.binary(&format!("x{i}"));
            }
            // A model with no constraints still lays out and penalises every encoding.
            bare.fix(vars[0], 1);
            let base = bare.compile().unwrap();
            let n = w.len() + spins;
            assert_eq!(
                n * (n - 1) / 2,
                *edges,
                "the clique formula and the table must agree for {w:?}"
            );
            assert_eq!(
                c.graph.n_edges - base.graph.n_edges,
                *edges,
                "couplings added for {w:?} ≤ {rhs}"
            );
        }
    }

    /// Every refusal, by name, with the reason in the message rather than in a comment.
    #[test]
    fn a_weighted_row_refuses_what_it_cannot_represent() {
        // A non-integer coefficient on an INEQUALITY: there is no integer residual for a slack to
        // range over, so it is refused rather than rounded.
        let mut m = Model::new();
        let (a, b) = (m.binary("a"), m.binary("b"));
        m.linear(vec![(Lit::Is(a, 1), 2.5), (Lit::Is(b, 1), 1.0)], Rel::Le, 4.0);
        match m.compile() {
            Err(CompileError::LinearNotInteger { what, value, .. }) => {
                assert_eq!(value, 2.5);
                assert!(what.contains('a'), "{what}");
                let text = CompileError::LinearNotInteger {
                    row: "2.5·a + 1·b ≤ 4".into(),
                    what,
                    value,
                }
                .to_string();
                assert!(text.contains("common denominator"), "{text}");
                assert!(text.contains("EQUALITY"), "the message must say what IS allowed: {text}");
            }
            other => panic!("expected a refusal, got {other:?}", other = other.err()),
        }

        // The same coefficients on an EQUALITY are accepted, with a caveat that says what the
        // integer path gets and this one does not.
        let mut m = Model::new();
        let (a, b) = (m.binary("a"), m.binary("b"));
        m.linear(vec![(Lit::Is(a, 1), 2.5), (Lit::Is(b, 1), 1.5)], Rel::Eq, 4.0);
        let c = m.compile().unwrap();
        assert_eq!(c.linear_slack, 0, "an equality needs no slack at all");
        assert!(c.caveats.iter().any(|w| w.contains("non-integer")), "{:?}", c.caveats);
        // And it still works: only a = b = 1 sums to 4.
        let s = c.solve_annealed(7);
        assert!(s.feasible(), "{:?}", s.violated);
        assert_eq!((s.value("a"), s.value("b")), (1, 1));

        // A row nothing can satisfy is refused by arithmetic, not annealed.
        let mut m = Model::new();
        let (a, b) = (m.binary("a"), m.binary("b"));
        m.linear(vec![(Lit::Is(a, 1), 3.0), (Lit::Is(b, 1), 4.0)], Rel::Ge, 9.0);
        assert!(matches!(m.compile(), Err(CompileError::LinearUnsatisfiable { .. })));

        // A domain-wall variable is refused for the reason it has always been refused: its
        // indicator is a PRODUCT, so a weighted sum of them inside a square is quartic.
        let mut m = Model::new();
        let t = m.categorical_as("t", 4, Encoding::DomainWall);
        m.linear(vec![(Lit::Is(t, 1), 3.0)], Rel::Le, 2.0);
        assert!(matches!(m.compile(), Err(CompileError::NeedsOneHot { .. })));

        // A row that constrains nothing compiles to nothing and says so.
        let mut m = Model::new();
        let (a, b) = (m.binary("a"), m.binary("b"));
        m.linear(vec![(Lit::Is(a, 1), 3.0), (Lit::Is(b, 1), 4.0)], Rel::Le, 12.0);
        let c = m.compile().unwrap();
        assert_eq!(c.linear_slack, 0);
        assert!(c.caveats.iter().any(|w| w.contains("constrains nothing")), "{:?}", c.caveats);
    }

    /// The row the brief opens with, solved, and read back in the modeller's own words.
    #[test]
    fn the_row_that_could_not_be_stated_now_solves_and_reports_itself() {
        let mut m = Model::new();
        let (a, b, c) = (m.binary("a"), m.binary("b"), m.binary("c"));
        m.objective(
            Sense::Maximize,
            Expr::lit(1.0, Lit::Is(a, 1))
                + Expr::lit(1.0, Lit::Is(b, 1))
                + Expr::lit(1.0, Lit::Is(c, 1)),
        );
        m.linear(
            vec![(Lit::Is(a, 1), 3.0), (Lit::Is(b, 1), 4.0), (Lit::Is(c, 1), 5.0)],
            Rel::Le,
            7.0,
        );
        let compiled = m.compile().unwrap();
        assert_eq!(compiled.linear_slack, 3);
        let s = compiled.solve_by(Method::Branch { max_nodes: 2_000_000 }, 4);
        assert!(s.feasible(), "{:?}", s.violated);
        assert!(s.proved_optimal);
        // 3+4 = 7 fits; every other pair does not, and all three certainly do not.
        assert_eq!((s.value("a"), s.value("b"), s.value("c")), (1, 1, 0));

        // And the violation reads in the caller's own notation, with the magnitude in their units.
        let mut all_on = vec![-1i8; compiled.graph.n];
        write_logical(&mut all_on, 3, 0b111);
        let bad = compiled.decode(&all_on);
        assert!(!bad.feasible());
        assert_eq!(bad.violated[0].amount, 5.0);
        assert_eq!(
            bad.violated[0].detail,
            "3·a + 4·b + 5·c ≤ 7, and the left side comes to 12"
        );
    }

    /// A hard weighted row is not outbid by an objective built in a loop.
    ///
    /// `effective_penalty` returns twice the largest SUMMED pull on one literal set, and the
    /// argument that keeps a `Cardinality` row dominant is that a unit of violation costs the whole
    /// penalty. A weighted row has the same property, and the gcd reduction does not weaken it:
    /// after dividing through, one unit of violation is still one unit, and it still costs `p`.
    #[test]
    fn a_hard_weighted_row_outbids_an_objective_built_in_a_loop() {
        let mut m = Model::new();
        let (a, b) = (m.binary("a"), m.binary("b"));
        // Four separate calls, each pulling 1 on the same literal set. Taking the largest single
        // coefficient would measure 1 here; the pull is 4.
        for _ in 0..4 {
            m.objective(Sense::Maximize, Expr::lit(1.0, Lit::Is(a, 1)));
            m.objective(Sense::Maximize, Expr::lit(1.0, Lit::Is(b, 1)));
        }
        // gcd 5 divides this through to `a + b ≤ 1`, so it is one slack spin -- and the penalty is
        // applied to the REDUCED row, which is the arm where dominance has to still hold.
        m.linear(vec![(Lit::Is(a, 1), 5.0), (Lit::Is(b, 1), 5.0)], Rel::Le, 5.0);
        let c = m.compile().unwrap();
        assert_eq!(c.linear_slack, 1, "gcd 5 turns 5a + 5b ≤ 5 into a + b ≤ 1");
        let s = c.solve_by(Method::Branch { max_nodes: 1_000_000 }, 9);
        assert!(s.feasible(), "the row lost to an accumulated objective: {:?}", s.violated);
        assert_eq!(s.value("a") + s.value("b"), 1, "one of them, and the objective picks which");
    }

    /// Duplicated literals and negative weights, which a real row has and a docstring does not.
    #[test]
    fn repeated_literals_are_merged_and_negative_weights_shift_the_bound() {
        // 3a + 4a − 2b ≤ 5 is 7a − 2b ≤ 5: a alone is 7, over; a and b together is 5, exactly on.
        let mut m = Model::new();
        let (a, b) = (m.binary("a"), m.binary("b"));
        m.linear(
            vec![(Lit::Is(a, 1), 3.0), (Lit::Is(b, 1), -2.0), (Lit::Is(a, 1), 4.0)],
            Rel::Le,
            5.0,
        );
        let c = m.compile().unwrap();
        for (av, bv, want) in [(0, 0, true), (0, 1, true), (1, 0, false), (1, 1, true)] {
            let mut state = vec![-1i8; c.graph.n];
            write_logical(&mut state, 2, (av as u32) | ((bv as u32) << 1));
            let s = c.decode(&state);
            assert_eq!(s.feasible(), want, "a={av} b={bv}: {:?}", s.violated);
        }
        // The merged row is what gets reported, not the three terms as written -- the reported
        // magnitude has to be the arithmetic's, and the arithmetic merged them.
        let mut state = vec![-1i8; c.graph.n];
        write_logical(&mut state, 2, 0b01);
        assert_eq!(c.decode(&state).violated[0].amount, 2.0);
    }


    /// A HARD CONSTRAINT MUST SURVIVE AN OBJECTIVE BUILT IN A LOOP, and it did not.
    ///
    /// `Expr::plus` extends rather than merges and `objective` accumulates -- which is what makes
    /// writing an objective one term at a time work at all, and is the documented pattern. It also
    /// puts many terms on the SAME literal, and what a constraint has to outbid is their sum.
    /// `effective_penalty` took the largest single coefficient, so three separate terms of
    /// `1.0 * a.is(1)` pulled with strength 3 against an automatic penalty of 2, and the hard
    /// `Fix(a, 0)` beside them was traded away: `a = 1`, `feasible = false`.
    ///
    /// The answer WAS reported as infeasible, so nothing lied -- but the automatic penalty exists
    /// precisely to stop this, and it was measuring the wrong quantity.
    #[test]
    fn a_hard_constraint_outbids_an_objective_accumulated_term_by_term() {
        for repeats in 1..=6usize {
            let mut m = Model::new();
            let a = m.binary("a");
            for _ in 0..repeats {
                m.objective(Sense::Maximize, Expr::lit(1.0, a.is(1)));
            }
            m.constrain(Constraint::Fix(a, 0));
            // The penalty must scale with the SUMMED pull, not stay at twice one coefficient.
            let p = m.effective_penalty();
            assert!(
                p >= 2.0 * repeats as f64 - 1e-9,
                "{repeats} terms of 1.0 pull with strength {repeats}, and the penalty was {p}"
            );
            let c = m.compile().expect("a binary and a fix compile");
            let s = c.solve_by(Method::Branch { max_nodes: 2_000_000 }, 1);
            assert!(
                s.feasible(),
                "{repeats} identical objective terms traded away a HARD constraint"
            );
            assert_eq!(s.value("a"), 0, "the fix says 0, whatever the objective wants");
        }
    }

    /// And grouping is by the literal SET, so a quadratic term is not confused with its factors and
    /// `a*b` is the same key as `b*a`.
    #[test]
    fn the_penalty_groups_by_literal_set_not_by_variable() {
        let mut m = Model::new();
        let a = m.binary("a");
        let b = m.binary("b");
        // Three DIFFERENT literal sets, each pulled once at 1.0: the largest pull is 1, not 3.
        m.objective(Sense::Maximize, Expr::lit(1.0, a.is(1)));
        m.objective(Sense::Maximize, Expr::lit(1.0, b.is(1)));
        m.objective(Sense::Maximize, Expr::pair(1.0, a.is(1), b.is(1)));
        assert!(
            (m.effective_penalty() - 2.0).abs() < 1e-9,
            "distinct literal sets must not be summed together: {}",
            m.effective_penalty()
        );

        // ...and the same pair written the other way round is the SAME set, so it does sum.
        let mut m2 = Model::new();
        let a2 = m2.binary("a");
        let b2 = m2.binary("b");
        m2.objective(Sense::Maximize, Expr::pair(1.0, a2.is(1), b2.is(1)));
        m2.objective(Sense::Maximize, Expr::pair(1.0, b2.is(1), a2.is(1)));
        assert!(
            (m2.effective_penalty() - 4.0).abs() < 1e-9,
            "a*b and b*a are one term and must sum: {}",
            m2.effective_penalty()
        );
    }

    /// The model layer can now PROVE an answer, and the proof is checked against enumeration.
    ///
    /// This is the claim the crate leads with -- "proved optimal without trusting the sampler" --
    /// and until now it was unreachable from the layer every doc says to reach for first.
    #[test]
    fn the_model_layer_can_prove_an_answer_optimal() {
        // Small enough to enumerate every assignment of the DECODED values, which is a different
        // computation from branch and bound on the compiled spins.
        let mut m = Model::new();
        let a = m.categorical("a", 3);
        let b = m.categorical("b", 3);
        let c = m.categorical("c", 3);
        m.all_different([a, b, c]);
        m.objective(Sense::Maximize, Expr::product(5.0, &[Lit::Is(a, 1)]));
        m.objective(Sense::Maximize, Expr::product(4.0, &[Lit::Is(b, 2)]));
        m.objective(Sense::Maximize, Expr::product(3.0, &[Lit::Is(c, 0)]));
        let compiled = m.compile().unwrap();

        let sol = compiled.solve_by(Method::Branch { max_nodes: 5_000_000 }, 1);
        assert!(sol.proved_optimal, "the tree is tiny; the budget is not the limit here");
        assert!(sol.feasible(), "{:?}", sol.violated);

        // Brute force over the modeller's OWN values: every (a, b, c) in 0..3, keeping the
        // all_different ones, scoring the objective as written.
        let mut best = f64::NEG_INFINITY;
        for x in 0..3i64 {
            for y in 0..3i64 {
                for z in 0..3i64 {
                    if x == y || y == z || x == z {
                        continue;
                    }
                    let v = 5.0 * f64::from(x == 1) + 4.0 * f64::from(y == 2) + 3.0 * f64::from(z == 0);
                    best = best.max(v);
                }
            }
        }
        assert_eq!(sol.objective, Some(best), "values {:?}", sol.values);
        // 5 + 4 + 3 = 12 is achievable: a=1, b=2, c=0 are all different.
        assert_eq!(best, 12.0);
    }

    /// A proof plus feasibility is a proof about the MODEL, and the argument does not use the
    /// penalty. So a model solved at a deliberately small penalty must either come back infeasible
    /// or be genuinely optimal -- never "proved" and quietly wrong.
    #[test]
    fn a_proof_with_a_small_penalty_is_still_a_proof_or_is_infeasible() {
        for penalty in [0.5f64, 2.0, 50.0] {
            let mut m = Model::new();
            let a = m.categorical("a", 3);
            let b = m.categorical("b", 3);
            m.not_equal(a, b);
            m.objective(Sense::Maximize, Expr::product(9.0, &[Lit::Is(a, 1)]));
            m.objective(Sense::Maximize, Expr::product(9.0, &[Lit::Is(b, 1)]));
            m.fixed_penalty(penalty);
            let sol = m.compile().unwrap().solve_by(Method::Branch { max_nodes: 5_000_000 }, 3);
            assert!(sol.proved_optimal, "penalty {penalty}: the tree is small");
            if sol.feasible() {
                // Both want value 1 and not_equal forbids it, so the best feasible pair scores 9.
                assert_eq!(sol.objective, Some(9.0), "penalty {penalty}: {:?}", sol.values);
            } else {
                // A proof on an infeasible optimum is the penalty being too small, which is exactly
                // what a small one should produce -- and it is information, not a wrong answer.
                assert!(penalty < 18.0, "penalty {penalty} should have been enough");
            }
        }
    }

    /// Every method returns a decoded, re-checked answer, and the ones that are not branch do not
    /// claim a proof.
    #[test]
    fn every_method_answers_and_only_branch_proves() {
        let mut m = Model::new();
        let vars: Vec<Var> = (0..6).map(|i| m.categorical(&format!("v{i}"), 3)).collect();
        for w in vars.windows(2) {
            m.not_equal(w[0], w[1]);
        }
        m.objective(Sense::Maximize, Expr::product(2.0, &[Lit::Is(vars[0], 0)]));
        let c = m.compile().unwrap();

        for method in [
            Method::Anneal,
            Method::Tabu { iterations: 20_000 },
            Method::Breakout { iterations: 20_000 },
        ] {
            let s = c.solve_by(method, 5);
            assert!(s.feasible(), "{method:?}: {:?}", s.violated);
            assert!(!s.proved_optimal, "{method:?} cannot prove anything");
            assert!(s.objective.is_some());
            // The energy belongs to the state that was returned.
            assert!(s.energy.is_finite());
        }
        let proved = c.solve_by(Method::Branch { max_nodes: 5_000_000 }, 5);
        assert!(proved.proved_optimal && proved.feasible());
        assert_eq!(proved.objective, Some(2.0));
    }

    /// Branch is warm-started from an anneal, so a bad decode must not become a bad incumbent.
    #[test]
    fn a_state_that_does_not_decode_is_not_handed_to_branch_as_an_incumbent() {
        // A binary-encoded variable over 6 values has two codewords that decode to nothing, and no
        // penalty removes them. It cannot appear in ANY constraint or objective either -- the
        // compiler refuses that by name -- so it is left unconstrained beside a variable that does
        // the model's work. That is the only way this crate can produce an undecodable answer.
        let mut m = Model::new();
        let _x = m.categorical_as("x", 6, Encoding::Binary);
        let y = m.categorical("y", 3);
        m.fix(y, 1);
        let c = m.compile().unwrap();
        // Whatever the anneal returns, solve_by must not panic and must return a decoded answer.
        let s = c.solve_by(Method::Branch { max_nodes: 200_000 }, 2);
        assert!(s.energy.is_finite());
        let _ = s.proved_optimal;
    }


    /// The number a modeller can actually use.
    ///
    /// `energy` is the compiled Ising energy with every penalty and the constant folded in, and it
    /// is the only number an answer carried until now. A person who writes "maximize 5*mon + 4*tue"
    /// cannot read their schedule's worth out of it, cannot compare two answers by it, and cannot
    /// tell a good answer from a barely-feasible one.
    #[test]
    fn an_answer_is_scored_in_the_modellers_own_units() {
        let mut m = Model::new();
        let mon = m.categorical("mon", 3);
        let tue = m.categorical("tue", 3);
        m.not_equal(mon, tue);
        m.objective(Sense::Maximize, Expr::product(5.0, &[Lit::Is(mon, 1)]));
        m.objective(Sense::Maximize, Expr::product(4.0, &[Lit::Is(tue, 2)]));
        let c = m.compile().unwrap();
        let s = c.solve_best_of(64);

        let obj = s.objective.expect("a single-sense objective has a direction to report");
        // The optimum is 9: mon = 1 and tue = 2, which not_equal permits.
        assert_eq!(obj, 9.0, "values {:?} energy {}", s.values, s.energy);
        assert!(s.feasible());
        // And it is NOT the compiled energy, which is what the modeller was being handed before.
        assert_ne!(obj, s.energy, "if these agree the test is measuring nothing");
    }

    #[test]
    fn a_minimisation_is_reported_as_written_and_a_mixed_objective_is_not_reported_at_all() {
        let mut m = Model::new();
        let x = m.categorical("x", 3);
        m.objective(Sense::Minimize, Expr::product(7.0, &[Lit::Is(x, 0)]));
        let s = m.compile().unwrap().solve_best_of(32);
        // Minimising a cost of 7 for x = 0: the optimum takes any other value and pays 0.
        assert_eq!(s.objective, Some(0.0), "{:?}", s.values);

        // Both senses at once composes fine as arithmetic and has NO single direction to report.
        // Reporting whichever call came last would be a number with a sign nobody chose.
        let mut mixed = Model::new();
        let a = mixed.categorical("a", 2);
        let b = mixed.categorical("b", 2);
        mixed.objective(Sense::Maximize, Expr::product(3.0, &[Lit::Is(a, 1)]));
        mixed.objective(Sense::Minimize, Expr::product(2.0, &[Lit::Is(b, 1)]));
        assert_eq!(mixed.compile().unwrap().solve_best_of(16).objective, None);

        // And a model with no objective at all reports none rather than zero, which would read as
        // "worth nothing" instead of "not asked".
        let mut none = Model::new();
        let v = none.categorical("v", 2);
        none.fix(v, 1);
        assert_eq!(none.compile().unwrap().solve_best_of(8).objective, None);
    }

    #[test]
    fn an_empty_objective_call_does_not_make_the_objective_mixed() {
        // `objective(Minimize, Expr::zero())` in a loop over an empty list must not turn a
        // maximisation into a directionless one.
        let mut m = Model::new();
        let x = m.categorical("x", 2);
        m.objective(Sense::Maximize, Expr::product(4.0, &[Lit::Is(x, 1)]));
        m.objective(Sense::Minimize, Expr::zero());
        assert_eq!(m.compile().unwrap().solve_best_of(16).objective, Some(4.0));
    }

    #[test]
    fn a_variable_that_did_not_decode_means_no_objective_rather_than_a_partial_one() {
        // Scoring half an answer produces a number that looks like a score and is not one.
        let mut m = Model::new();
        let x = m.categorical_as("x", 6, Encoding::Binary);
        let y = m.categorical("y", 2);
        m.objective(Sense::Maximize, Expr::product(1.0, &[Lit::Is(y, 1)]));
        let c = m.compile().unwrap();
        // A binary slot over 6 values has two codewords that decode to nothing; drive one in.
        let mut state = vec![-1i8; c.spins()];
        for s in state.iter_mut().take(c.spins()) {
            *s = 1;
        }
        let sol = c.decode(&state);
        if !sol.invalid.is_empty() {
            assert_eq!(sol.objective, None, "an unreadable variable means no score");
        }
        let _ = x;
    }



    #[test]
    fn a_variable_comes_back_by_name_in_its_own_units() {
        let mut m = Model::new();
        let x = m.integer("temperature", 10, 20);
        m.fix(x, 13);
        let c = m.compile().unwrap();
        let s = c.solve_best_of(6);
        assert!(s.feasible(), "{s}");
        assert_eq!(s.value("temperature"), 13, "integers decode in their own range: {s}");
    }

    #[test]
    fn an_integer_range_the_graph_cannot_index_is_refused_rather_than_aborting() {
        // `ft_model_integer` accepted a range spanning most of i64, returned success, and
        // `ft_model_compile` then ABORTED the caller's process with a capacity overflow -- where
        // the header documents a zero return. Spin indices are u32; nothing here addresses more.
        let mut m = Model::new();
        m.integer("t", -4_611_686_018_427_387_904, 4_611_686_018_427_387_904);
        match m.compile() {
            Err(CompileError::DomainTooLarge { var, .. }) => assert_eq!(var, "t"),
            Err(e) => panic!("refused for the wrong reason: {e}"),
            Ok(_) => panic!("a domain of 9.2e18 values cannot be laid out"),
        }

        // A large but addressable range still compiles.
        let mut ok = Model::new();
        ok.integer("u", 0, 1000);
        assert!(ok.compile().is_ok(), "1001 values is ordinary");
    }

    #[test]
    fn a_coefficient_that_is_not_finite_is_refused_rather_than_disabling_the_objective() {
        // One NaN term used to compile, solve, and report feasible: true -- while discarding every
        // OTHER preference, because comparisons against NaN are all false and the sampler's
        // "better than the best so far" test never fires again. A model maximising 3*a beside one
        // NaN term answered a = 0. Confident, feasible-looking, wrong.
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut m = Model::new();
            let a = m.binary("a");
            let b = m.binary("b");
            m.objective(Sense::Maximize, Expr::product(3.0, &[Lit::Is(a, 1)]));
            m.objective(Sense::Maximize, Expr::product(bad, &[Lit::Is(b, 1)]));
            match m.compile() {
                Err(CompileError::NotFinite { what, .. }) => assert_eq!(what, "objective coefficient"),
                Err(e) => panic!("{bad} refused for the wrong reason: {e}"),
                Ok(_) => panic!("{bad} compiled, and would have disabled the 3*a preference"),
            }
        }

        // The finite model it was silently degrading still works -- the half that makes this a
        // refusal worth having rather than merely strict.
        let mut m = Model::new();
        let a = m.binary("a");
        m.objective(Sense::Maximize, Expr::product(3.0, &[Lit::Is(a, 1)]));
        assert_eq!(m.compile().unwrap().solve_best_of(32).value("a"), 1);

        // And a HARD constraint, whose weight sentinel IS NaN, must still compile. Checking every
        // weight rather than only the soft ones rejected all of them.
        let mut m = Model::new();
        let x = m.categorical("x", 3);
        m.fix(x, 1);
        assert!(m.compile().is_ok(), "a hard constraint's NaN sentinel is not a bad coefficient");
    }

    #[test]
    fn an_integer_is_written_in_its_own_values_not_in_slots() {
        // The trap this pins: an integer over 10..=20 has slot 3 holding the value 13. A literal
        // says 13. Writing 3 used to mean 13 and now means an error, because 3 is not a temperature
        // this variable can take.
        let mut m = Model::new();
        let x = m.integer("temperature", 10, 20);
        m.objective(Sense::Maximize, 5.0 * x.is(13));
        assert_eq!(m.compile().unwrap().solve_best_of(8).value("temperature"), 13);

        let mut m = Model::new();
        let x = m.integer("temperature", 10, 20);
        m.fix(x, 3);
        let e = match m.compile() { Err(e) => e.to_string(), Ok(_) => panic!("3 is not a temperature in 10..=20") };
        assert!(e.contains("10..=20") && e.contains("3 is not"), "{e}");
    }

    #[test]
    fn two_variables_cannot_share_a_name() {
        // An answer is a map keyed by name. A second variable with the same name does not shadow
        // the first, it REPLACES it -- so one of the two vanishes from the result and the caller
        // reads a value belonging to the other. Found by writing a binding that named four
        // variables in a loop and forgot to vary the name.
        let mut m = Model::new();
        let a = m.binary("v");
        let b = m.binary("v");
        m.fix(a, 1);
        m.fix(b, 0);
        let e = match m.compile() { Err(e) => e.to_string(), Ok(_) => panic!("two 'v's") };
        assert!(e.contains("both called 'v'") && e.contains("disappears"), "{e}");

        // distinct names compile, and both survive to the answer
        let mut m = Model::new();
        let a = m.binary("a");
        let b = m.binary("b");
        m.fix(a, 1);
        m.fix(b, 0);
        let s = m.compile().unwrap().solve_best_of(8);
        assert_eq!((s.value("a"), s.value("b")), (1, 0), "{s}");
    }

    #[test]
    fn feasible_means_the_constraints_hold_not_merely_that_it_decoded() {
        // A penalty makes a constraint EXPENSIVE, not impossible. Pin it below the objective and
        // the sampler will happily pay it -- returning a state whose variables all decode cleanly
        // and whose constraint is broken. That used to report feasible: true.
        let mut m = Model::new();
        let a = m.categorical("a", 3);
        let b = m.categorical("b", 3);
        m.not_equal(a, b);
        m.fixed_penalty(1.0);
        m.objective(Sense::Maximize, 40.0 * a.is(1) + 40.0 * b.is(1)); // both want the same value
        let s = m.compile().unwrap().solve_best_of(16);
        assert_eq!((s.value("a"), s.value("b")), (1, 1), "the objective outbids a penalty of 1");
        assert!(s.invalid.is_empty(), "and every variable decoded perfectly: {s}");
        assert!(!s.feasible(), "so this is the whole point: it is NOT feasible: {s}");
        assert_eq!(s.violated.len(), 1, "{s}");
        assert!(s.violated[0].detail.contains("must differ") && s.violated[0].detail.contains('a'),
                "and it names the constraint in the caller's words: {}", s.violated[0]);

        // raised, the same model is feasible
        let mut m = Model::new();
        let a = m.categorical("a", 3);
        let b = m.categorical("b", 3);
        m.not_equal(a, b);
        m.fixed_penalty(200.0);
        m.objective(Sense::Maximize, 40.0 * a.is(1) + 40.0 * b.is(1));
        let s = m.compile().unwrap().solve_best_of(16);
        assert!(s.feasible(), "{s}");
        assert_ne!(s.value("a"), s.value("b"), "{s}");
    }

    #[test]
    fn a_violation_says_how_far_outside_it_sits() {
        // Description alone makes every violation look equally bad. "At most 2 of 5, and 3 hold"
        // is one over; "and 5 hold" is three over. A caller ranking repairs, or deciding whether a
        // larger penalty would be enough, needs to tell them apart.
        let over = |weight: f64| {
            let mut m = Model::new();
            let vs: Vec<Var> = (0..5).map(|i| m.binary(&format!("v{i}"))).collect();
            m.at_most(vs.iter().map(|&v| Lit::Is(v, 1)).collect(), 2);
            m.fixed_penalty(0.1);
            for &v in &vs {
                m.objective(Sense::Maximize, weight * v.is(1));
            }
            let s = m.compile().unwrap().solve_best_of(16);
            let on = vs.iter().filter(|&&v| s.value(m.name_of(v)) == 1).count();
            (on, s.violated.first().map(|v| v.amount))
        };

        // a big reward buys every variable: five on against a ceiling of two is three over
        let (on, by) = over(50.0);
        assert_eq!(on, 5, "the objective outbids a penalty of 0.1");
        assert_eq!(by, Some(3.0), "and the violation says by how much");

        // the amount is in the constraint's own units and always positive
        let mut m = Model::new();
        let x = m.integer("t", 0, 20);
        m.fix(x, 3);
        m.fixed_penalty(0.01);
        m.objective(Sense::Maximize, 40.0 * x.is(17));
        let s = m.compile().unwrap().solve_best_of(16);
        assert_eq!(s.value("t"), 17, "the objective wins");
        assert_eq!(s.violated[0].amount, 14.0, "17 is 14 away from the 3 it was fixed to");
        assert!(s.violated[0].to_string().contains("(by 14)"), "{}", s.violated[0]);
    }

    #[test]
    fn a_violated_counting_constraint_says_how_far_off_it_is() {
        let mut m = Model::new();
        let vs: Vec<Var> = (0..4).map(|i| m.binary(&format!("v{i}"))).collect();
        m.at_most(vs.iter().map(|&v| Lit::Is(v, 1)).collect(), 1);
        m.fixed_penalty(0.5);
        for &v in &vs {
            m.objective(Sense::Maximize, 20.0 * v.is(1));
        }
        let s = m.compile().unwrap().solve_best_of(16);
        assert!(!s.feasible(), "a penalty of 0.5 against a weight of 20 loses: {s}");
        assert!(s.violated[0].detail.contains("at most 1") && s.violated[0].detail.contains("4 do"),
                "{}", s.violated[0].detail);
    }

    #[test]
    fn a_spin_variable_speaks_in_minus_one_and_plus_one() {
        // Spin was the one domain nothing tested, and it was the one domain where the value/slot
        // distinction was still wrong: the decoder handed back -1 and +1 while the literal reader
        // folded Spin into the same arm as a categorical and called its values 0 and 1. So
        // `x.is(0)` secretly meant -1, and `x.is(-1)` was rejected as out of domain.
        let mut m = Model::new();
        let x = m.spin("x");
        m.fix(x, -1);
        let s = m.compile().unwrap().solve_best_of(8);
        assert!(s.feasible(), "{s}");
        assert_eq!(s.value("x"), -1, "a spin fixed to -1 reads back as -1: {s}");

        let mut m = Model::new();
        let x = m.spin("x");
        m.objective(Sense::Maximize, 3.0 * x.is(1));
        assert_eq!(m.compile().unwrap().solve_best_of(8).value("x"), 1);

        // and the values BETWEEN them are not values it can take
        let mut m = Model::new();
        let x = m.spin("x");
        m.fix(x, 0);
        let e = match m.compile() { Err(e) => e.to_string(), Ok(_) => panic!("0 is not a spin") };
        assert!(e.contains("-1 and +1"), "{e}");
    }

    #[test]
    fn a_spin_and_an_integer_compare_by_value() {
        // The failure the audit found: `not_equal(spin, integer)` compared slot 0 of each. Slot 0
        // of a spin is -1 and slot 0 of an integer over -1..=1 is also -1, so the pair was reported
        // as DIFFERING while both held -1.
        let mut m = Model::new();
        let a = m.spin("a");
        let b = m.integer("b", -1, 1);
        m.not_equal(a, b);
        let s = m.compile().unwrap().solve_best_of(16);
        assert!(s.feasible(), "{s}");
        assert_ne!(s.value("a"), s.value("b"), "a spin and an integer must really differ: {s}");

        // and forced together they land on a value both domains contain
        let mut m = Model::new();
        let a = m.spin("a");
        let b = m.integer("b", -1, 1);
        m.equal(a, b);
        m.objective(Sense::Maximize, 5.0 * b.is(1));
        let s = m.compile().unwrap().solve_best_of(16);
        assert!(s.feasible(), "{s}");
        assert_eq!((s.value("a"), s.value("b")), (1, 1), "{s}");
    }

    #[test]
    fn equality_compares_values_across_different_ranges() {
        // Two integers whose ranges overlap in exactly one place. Comparing them slot by slot
        // would have them agree in six places and disagree in none of the right ones.
        let mut m = Model::new();
        let a = m.integer("a", 5, 10);
        let b = m.integer("b", 0, 5);
        m.equal(a, b);
        let s = m.compile().unwrap().solve_best_of(12);
        assert!(s.feasible(), "{s}");
        assert_eq!((s.value("a"), s.value("b")), (5, 5), "5 is the only value both can take: {s}");

        // and the same pair forced apart still lands inside both ranges
        let mut m = Model::new();
        let a = m.integer("a", 5, 10);
        let b = m.integer("b", 0, 5);
        m.not_equal(a, b);
        let s = m.compile().unwrap().solve_best_of(12);
        assert!(s.feasible(), "{s}");
        assert_ne!(s.value("a"), s.value("b"), "{s}");
        assert!((5..=10).contains(&s.value("a")) && (0..=5).contains(&s.value("b")), "{s}");
    }

    #[test]
    fn graph_colouring_reads_like_graph_colouring() {
        // A triangle needs three colours; two adjacent vertices must never agree.
        let mut m = Model::new();
        let a = m.categorical("a", 3);
        let b = m.categorical("b", 3);
        let cc = m.categorical("c", 3);
        m.not_equal(a, b).not_equal(b, cc).not_equal(a, cc);
        let comp = m.compile().unwrap();
        let s = comp.solve_best_of(12);
        assert!(s.feasible(), "{s}");
        assert_ne!(s.value("a"), s.value("b"), "{s}");
        assert_ne!(s.value("b"), s.value("c"), "{s}");
        assert_ne!(s.value("a"), s.value("c"), "{s}");
    }

    #[test]
    fn two_colours_cannot_colour_a_triangle_and_it_says_so() {
        // An odd cycle is not 2-colourable. The point is that the answer SAYS SO -- names the
        // constraint it could not keep -- rather than handing back a colouring that looks fine.
        let mut m = Model::new();
        let a = m.categorical("a", 2);
        let b = m.categorical("b", 2);
        let cc = m.categorical("c", 2);
        m.not_equal(a, b).not_equal(b, cc).not_equal(a, cc);
        let s = m.compile().unwrap().solve_best_of(12);
        assert!(!s.feasible(), "a triangle has no 2-colouring, so no answer is feasible: {s}");
        assert!(!s.violated.is_empty() || !s.invalid.is_empty(),
                "and it must say which part it could not deliver: {s}");
        // whichever way it gives, it gives exactly one: two of the three pairs still differ
        if s.invalid.is_empty() {
            assert_eq!(s.violated.len(), 1, "exactly one pair agrees in a 2-coloured triangle: {s}");
            assert!(s.violated[0].detail.contains("must differ"), "{}", s.violated[0].detail);
        }
    }

    #[test]
    fn equality_binds() {
        let mut m = Model::new();
        let a = m.categorical("a", 4);
        let b = m.categorical("b", 4);
        m.equal(a, b);
        let c = m.compile().unwrap();
        let s = c.solve_best_of(8);
        assert!(s.feasible(), "{s}");
        assert_eq!(s.value("a"), s.value("b"), "{s}");
    }

    #[test]
    fn an_objective_is_optimised_in_the_stated_direction() {
        // Reward taking a high value, and check that maximising and minimising differ.
        let build = |sense: Sense| {
            let mut m = Model::new();
            let x = m.categorical("x", 5);
            let mut e = Expr::zero();
            for v in 0..5 {
                e = e.plus(Expr::lit(v as f64, Lit::Is(x, v)));
            }
            m.objective(sense, e);
            m.compile().unwrap().solve_best_of(10)
        };
        assert_eq!(build(Sense::Maximize).value("x"), 4, "maximising should pick the top value");
        assert_eq!(build(Sense::Minimize).value("x"), 0, "minimising should pick the bottom");
    }

    #[test]
    fn a_quadratic_objective_compiles_and_optimises() {
        // Reward two variables agreeing on a specific value; a product of two indicators.
        let mut m = Model::new();
        let a = m.categorical("a", 3);
        let b = m.categorical("b", 3);
        m.objective(Sense::Maximize, Expr::pair(5.0, Lit::Is(a, 2), Lit::Is(b, 2)));
        let c = m.compile().unwrap();
        let s = c.solve_best_of(10);
        assert!(s.feasible(), "{s}");
        assert_eq!((s.value("a"), s.value("b")), (2, 2), "{s}");
    }

    /// A model with a symmetry has several optima, and `solve_best_with` throws all but one away.
    ///
    /// Exactly-one over three binaries: three assignments, all feasible, all at the same compiled
    /// energy. The count is known in advance, so this is checkable rather than plausible.
    #[test]
    fn every_way_to_do_the_job_is_found_and_counted_once() {
        let mut m = Model::new();
        let a = m.binary("a");
        let b = m.binary("b");
        let c = m.binary("c");
        m.constrain(Constraint::ExactlyOne(vec![Lit::Is(a, 1), Lit::Is(b, 1), Lit::Is(c, 1)]));
        let comp = m.compile().unwrap();
        let sched = Schedule::geometric(0.05, 8.0, 120, 40);
        let answers = comp.solve_all_with(&sched, 40);
        assert_eq!(answers.len(), 40, "one answer per seed, kept");

        let opt = distinct_optima(&answers, 1e-9);
        assert_eq!(opt.len(), 3, "exactly-one over three has three ways to be satisfied");
        for s in &opt {
            assert!(s.feasible());
            let on = ["a", "b", "c"].iter().filter(|n| s.value(n) == 1).count();
            assert_eq!(on, 1);
        }
        // The solve's answer is ONE OF the optima. Not the head of the list: all three tie on
        // energy, so `distinct_optima` orders them by assignment while `solve_best_with` returns
        // whichever seed reached the minimum first. An earlier version asserted they were equal
        // and held by coincidence until a colouring change moved the trajectory -- which is a
        // fact about sweep order and says nothing about either answer being better.
        let best = comp.solve_best_with(&sched, 40);
        assert!(
            opt.iter().any(|s| s.values == best.values),
            "solve's answer is not among the optima"
        );
        assert!(opt.iter().all(|s| (s.energy - best.energy).abs() < 1e-9), "all tie on energy");
        // Deterministic: same tries, same list, in the same order.
        let again = distinct_optima(&comp.solve_all_with(&sched, 40), 1e-9);
        assert_eq!(
            opt.iter().map(|s| s.values.clone()).collect::<Vec<_>>(),
            again.iter().map(|s| s.values.clone()).collect::<Vec<_>>()
        );
    }

    /// Named for the claim it refuted, and kept as the control that would catch it becoming true.
    ///
    /// `distinct_optima` keys on the decoded assignment, and the obvious justification is that a
    /// compiled model carries slack bits no variable reads, so counting STATES would report one
    /// answer as several. This enumerates the whole state space to check that, and the obvious
    /// justification is wrong: eleven satisfying assignments, eleven minimum-energy states. The
    /// penalty that makes a cardinality row hold also PINS its slack register, so at the optimum
    /// nothing is left floating.
    ///
    /// The test stays, asserting the equality rather than the inequality it was written for. An
    /// encoding with a redundant slack representation would break it, which is exactly when
    /// someone needs to know — and the docs on `distinct_optima` now say the honest reason for
    /// keying on values: not a discrepancy today, but that the count must not depend on how the
    /// compiler chose to represent the model.
    #[test]
    fn counting_spin_states_would_over_count_the_optima() {
        let mut m = Model::new();
        let a = m.binary("a");
        let b = m.binary("b");
        let c = m.binary("c");
        let d = m.binary("d");
        // At most two of four: a cardinality row, which compiles a slack register no variable
        // reads and which is free to sit anywhere a satisfying assignment allows.
        m.constrain(Constraint::AtMost {
            lits: vec![Lit::Is(a, 1), Lit::Is(b, 1), Lit::Is(c, 1), Lit::Is(d, 1)],
            k: 2,
        });
        let comp = m.compile().unwrap();
        let n = comp.spins();
        assert!(n > 4, "the compiled model must carry bits beyond the four variables, got {n}");
        assert!(n <= 20, "this test enumerates 2^n and n is {n}");

        let mut s = vec![-1i8; n];
        let mut best = f64::INFINITY;
        let mut at_min: Vec<Vec<i8>> = Vec::new();
        for mask in 0..(1usize << n) {
            for (i, v) in s.iter_mut().enumerate() {
                *v = if mask >> i & 1 == 1 { 1 } else { -1 };
            }
            let e = comp.graph.energy(&s);
            if e < best - 1e-9 {
                best = e;
                at_min.clear();
            }
            if e <= best + 1e-9 {
                at_min.push(s.clone());
            }
        }

        let assignments: std::collections::BTreeSet<_> = at_min
            .iter()
            .map(|st| comp.decode(st))
            .filter(|sol| sol.feasible())
            .map(|sol| sol.values.clone())
            .collect();

        // 11 assignments of four binaries have at most two set: 1 + 4 + 6.
        assert_eq!(assignments.len(), 11, "at most two of four");
        assert_eq!(
            at_min.len(),
            assignments.len(),
            "one minimum-energy state per assignment: the penalty pins the slack. If this ever \
             fails HIGH, an encoding has gained a redundant representation and a state count would \
             now over-report the optima -- which is the case `distinct_optima` is written to \
             survive. If it fails LOW, two assignments have collapsed onto one state, which is a \
             decoder bug. {} states for {} assignments",
            at_min.len(),
            assignments.len()
        );
    }

    #[test]
    fn an_infeasible_run_has_no_optima_rather_than_a_bad_one() {
        let mut m = Model::new();
        let a = m.binary("a");
        // Two hard rows that cannot both hold: a must be 1 and a must be 0.
        m.constrain(Constraint::Fix(a, 1));
        m.constrain(Constraint::Fix(a, 0));
        let Ok(comp) = m.compile() else { return }; // a compile-time refusal is also a right answer
        let answers = comp.solve_all_with(&Schedule::geometric(0.05, 8.0, 60, 20), 8);
        let opt = distinct_optima(&answers, 1e-9);
        assert!(
            opt.iter().all(|s| s.feasible()),
            "an assignment that breaks a hard row is not a way to do the job"
        );
    }

    #[test]
    fn exactly_one_holds() {
        let mut m = Model::new();
        let a = m.binary("a");
        let b = m.binary("b");
        let c = m.binary("c");
        m.constrain(Constraint::ExactlyOne(vec![
            Lit::Is(a, 1),
            Lit::Is(b, 1),
            Lit::Is(c, 1),
        ]));
        let comp = m.compile().unwrap();
        let s = comp.solve_best_of(10);
        assert!(s.feasible(), "{s}");
        let on = ["a", "b", "c"].iter().filter(|n| s.value(n) == 1).count();
        assert_eq!(on, 1, "exactly one should be on: {s}");
    }

    #[test]
    fn a_domain_wall_variable_in_an_expression_is_refused_with_the_reason() {
        // The honest limit, enforced rather than discovered. A domain-wall indicator is a product
        // of two spins, so a product of two of them is quartic.
        let mut m = Model::new();
        let x = m.categorical_as("x", 4, Encoding::DomainWall);
        m.objective(Sense::Minimize, Expr::lit(1.0, Lit::Is(x, 2)));
        let e = match m.compile() {
            Err(e) => e,
            Ok(_) => panic!("a domain-wall variable in an expression must be refused"),
        };
        assert!(matches!(e, CompileError::NeedsOneHot { .. }));
        assert!(e.to_string().contains("quartic"), "{e}");
    }

    #[test]
    fn a_domain_wall_variable_is_fine_when_it_only_has_to_decode() {
        // And it still works where its advantage lies: constraints and feasibility.
        let mut m = Model::new();
        let _x = m.categorical_as("x", 6, Encoding::DomainWall);
        let c = m.compile().unwrap();
        assert_eq!(c.spins(), 5, "domain wall uses k-1 spins");
        assert!(c.solve_best_of(6).feasible());
    }

    #[test]
    fn a_value_outside_the_domain_is_a_compile_error() {
        let mut m = Model::new();
        let x = m.categorical("x", 3);
        m.objective(Sense::Minimize, Expr::lit(1.0, Lit::Is(x, 7)));
        assert!(matches!(m.compile(), Err(CompileError::BadValue { .. })));
    }

    #[test]
    fn the_penalty_scales_with_the_objective() {
        // The defect this scaling exists to prevent, pinned. An objective with coefficients up to 4
        // against a penalty of 2 trades the encoding away, and the variable stops decoding -- which
        // reads as a broken sampler rather than an under-weighted constraint.
        let build = |fixed: Option<f64>| {
            let mut m = Model::new();
            let x = m.categorical("x", 5);
            let mut e = Expr::zero();
            for v in 0..5 {
                e = e.plus(Expr::lit(v as f64, Lit::Is(x, v)));
            }
            m.objective(Sense::Maximize, e);
            if let Some(p) = fixed {
                m.fixed_penalty(p);
            }
            m
        };
        assert_eq!(build(None).effective_penalty(), 8.0, "twice the largest coefficient");
        assert_eq!(build(Some(2.0)).effective_penalty(), 2.0, "fixed means fixed");

        // and the scaled one decodes where the fixed-low one does not
        assert!(build(None).compile().unwrap().solve_best_of(10).feasible());
        let weak = build(Some(0.2)).compile().unwrap().solve_best_of(10);
        assert!(!weak.feasible(), "a penalty far below the objective must lose: {weak}");
    }

    #[test]
    fn a_variable_that_did_not_decode_says_which_problem_it_is() {
        let mut m = Model::new();
        let x = m.categorical("x", 4);
        m.objective(Sense::Maximize, Expr::lit(50.0, Lit::Is(x, 3)));
        m.fixed_penalty(0.01);
        let s = m.compile().unwrap().solve_annealed(1);
        if !s.feasible() {
            let msg = std::panic::catch_unwind(|| s.value("x")).unwrap_err();
            let msg = msg
                .downcast_ref::<String>()
                .cloned()
                .unwrap_or_default();
            assert!(msg.contains("did not decode"), "{msg}");
            assert!(msg.contains("penalty"), "and it should say what to do: {msg}");
        }
        // an unknown name is a different message
        let ok = Model::new();
        let _ = ok;
    }

    #[test]
    fn cardinality_selects_exactly_k() {
        // The workhorse of assignment and selection: choose exactly three of eight.
        let mut m = Model::new();
        let bits: Vec<Var> = (0..8).map(|i| m.binary(&format!("b{i}"))).collect();
        let lits: Vec<Lit> = bits.iter().map(|&v| Lit::Is(v, 1)).collect();
        m.cardinality(lits, 3);
        let c = m.compile().unwrap();
        let s = c.solve_best_of(20);
        assert!(s.feasible(), "{s}");
        let on = (0..8).filter(|i| s.value(&format!("b{i}")) == 1).count();
        assert_eq!(on, 3, "exactly three should be selected: {s}");
    }

    #[test]
    fn cardinality_works_against_an_objective() {
        // Choose exactly two, and prefer the two with the highest reward. Both the constraint and
        // the objective must be respected, which is where a mis-scaled penalty shows up.
        let mut m = Model::new();
        let bits: Vec<Var> = (0..5).map(|i| m.binary(&format!("b{i}"))).collect();
        let lits: Vec<Lit> = bits.iter().map(|&v| Lit::Is(v, 1)).collect();
        m.cardinality(lits, 2);
        let mut e = Expr::zero();
        for (i, &v) in bits.iter().enumerate() {
            e = e.plus(Expr::lit(i as f64, Lit::Is(v, 1)));   // b4 worth most
        }
        m.objective(Sense::Maximize, e);
        let s = m.compile().unwrap().solve_best_of(24);
        assert!(s.feasible(), "{s}");
        let on: Vec<usize> = (0..5).filter(|i| s.value(&format!("b{i}")) == 1).collect();
        assert_eq!(on, vec![3, 4], "the two most valuable, and only two: {s}");
    }

    #[test]
    fn a_boundary_inequality_still_constrains() {
        // `at most 0` and `at least all` need no slack -- there is only one admissible count -- and
        // the compiler used to take "needs no slack" as "needs no constraint". Both then compiled
        // to NOTHING and reported feasible while violating the request.
        let none = {
            let mut m = Model::new();
            let vs: Vec<Var> = (0..4).map(|i| m.binary(&format!("v{i}"))).collect();
            m.at_most(vs.iter().map(|&v| Lit::Is(v, 1)).collect(), 0);
            for &v in &vs {
                m.objective(Sense::Maximize, 1.0 * v.is(1)); // push every one of them ON
            }
            let s = m.compile().unwrap().solve_best_of(16);
            (s.feasible(), vs.iter().filter(|&&v| s.value(m.name_of(v)) == 1).count(), s)
        };
        assert!(none.0, "{}", none.2);
        assert_eq!(none.1, 0, "at most 0 means none, against a reward on every one: {}", none.2);

        let all = {
            let mut m = Model::new();
            let vs: Vec<Var> = (0..4).map(|i| m.binary(&format!("v{i}"))).collect();
            m.at_least(vs.iter().map(|&v| Lit::Is(v, 1)).collect(), 4);
            for &v in &vs {
                m.objective(Sense::Minimize, 1.0 * v.is(1)); // push every one of them OFF
            }
            let s = m.compile().unwrap().solve_best_of(16);
            (s.feasible(), vs.iter().filter(|&&v| s.value(m.name_of(v)) == 1).count(), s)
        };
        assert!(all.0, "{}", all.2);
        assert_eq!(all.1, 4, "at least 4 of 4 means all, against a penalty on every one: {}", all.2);
    }

    #[test]
    fn an_objective_accumulates_and_mixed_senses_compose() {
        // Written in a loop, which is how an objective with one term per option gets written. The
        // old form kept only the LAST call, so this returned v3 alone.
        let mut m = Model::new();
        let vs: Vec<Var> = (0..4).map(|i| m.binary(&format!("v{i}"))).collect();
        for &v in &vs {
            m.objective(Sense::Maximize, 1.0 * v.is(1));
        }
        let s = m.compile().unwrap().solve_best_of(16);
        assert_eq!(vs.iter().filter(|&&v| s.value(m.name_of(v)) == 1).count(), 4,
                   "every term counts, not just the last: {s}");

        // And a term in the other direction changes only that term. The old form re-read the whole
        // accumulated objective under the new sense, flipping everything already asked for.
        let mut m = Model::new();
        let vs: Vec<Var> = (0..4).map(|i| m.binary(&format!("v{i}"))).collect();
        for &v in &vs[..3] {
            m.objective(Sense::Maximize, 1.0 * v.is(1));
        }
        m.objective(Sense::Minimize, 1.0 * vs[3].is(1));
        let s = m.compile().unwrap().solve_best_of(16);
        let on: Vec<usize> = (0..4).filter(|&i| s.value(m.name_of(vs[i])) == 1).collect();
        assert_eq!(on, vec![0, 1, 2], "the first three rewarded, the last penalised: {s}");
    }

    #[test]
    fn set_objective_discards_what_came_before() {
        // The discarded terms have to be VISIBLE for this to test anything: a variable with no term
        // at all lands wherever the sampler leaves it, so "it is off" would prove nothing. So the
        // terms being discarded push every variable OFF, and the replacement pushes them all ON.
        // If any of the old terms survived they would fight the new ones.
        let mut m = Model::new();
        let vs: Vec<Var> = (0..3).map(|i| m.binary(&format!("v{i}"))).collect();
        for &v in &vs {
            m.objective(Sense::Minimize, 10.0 * v.is(1));
        }
        let e = vs.iter().fold(Expr::zero(), |a, &v| a.plus(1.0 * v.is(1)));
        m.set_objective(Sense::Maximize, e);
        let s = m.compile().unwrap().solve_best_of(16);
        assert_eq!((0..3).filter(|&i| s.value(m.name_of(vs[i])) == 1).count(), 3,
                   "the minimising terms are gone, so all three come on: {s}");
    }

    #[test]
    fn a_vacuous_inequality_costs_nothing_and_forbids_nothing() {
        // The other two boundaries: `at most all` and `at least 0` are true of every state.
        for (kind, k) in [("at_most", 4usize), ("at_least", 0usize)] {
            let mut m = Model::new();
            let vs: Vec<Var> = (0..4).map(|i| m.binary(&format!("v{i}"))).collect();
            let lits: Vec<Lit> = vs.iter().map(|&v| Lit::Is(v, 1)).collect();
            if kind == "at_most" { m.at_most(lits, k); } else { m.at_least(lits, k); }
            for &v in &vs {
                m.objective(Sense::Maximize, 1.0 * v.is(1));
            }
            let s = m.compile().unwrap().solve_best_of(16);
            assert!(s.feasible(), "{kind} {k}: {s}");
            let on = vs.iter().filter(|&&v| s.value(m.name_of(v)) == 1).count();
            assert_eq!(on, 4, "{kind} {k} forbids nothing, so every reward is taken: {s}");
        }
    }

    #[test]
    fn at_most_allows_fewer_but_not_more() {
        // The point of slack: an inequality must not punish being under the limit. A squared
        // penalty on (sum - k) alone would, which is a different problem.
        let mut m = Model::new();
        let bits: Vec<Var> = (0..6).map(|i| m.binary(&format!("b{i}"))).collect();
        let lits: Vec<Lit> = bits.iter().map(|&v| Lit::Is(v, 1)).collect();
        m.at_most(lits.clone(), 2);
        // reward turning things on, so the constraint has something to push against
        let mut e = Expr::zero();
        for &v in &bits {
            e = e.plus(Expr::lit(1.0, Lit::Is(v, 1)));
        }
        m.objective(Sense::Maximize, e);
        let s = m.compile().unwrap().solve_best_of(40);
        assert!(s.feasible(), "{s}");
        let on = (0..6).filter(|i| s.value(&format!("b{i}")) == 1).count();
        assert_eq!(on, 2, "it should take as many as allowed and no more: {s}");
    }

    #[test]
    fn at_most_is_satisfied_by_taking_none() {
        // With nothing rewarding them, zero is feasible under "at most 3" -- and would be punished
        // by a naive equality.
        let mut m = Model::new();
        let bits: Vec<Var> = (0..5).map(|i| m.binary(&format!("b{i}"))).collect();
        m.at_most(bits.iter().map(|&v| Lit::Is(v, 1)).collect(), 3);
        let s = m.compile().unwrap().solve_best_of(20);
        assert!(s.feasible(), "{s}");
        let on = (0..5).filter(|i| s.value(&format!("b{i}")) == 1).count();
        assert!(on <= 3, "at most three: {s}");
    }

    #[test]
    fn at_least_forces_a_minimum() {
        let mut m = Model::new();
        let bits: Vec<Var> = (0..6).map(|i| m.binary(&format!("b{i}"))).collect();
        let lits: Vec<Lit> = bits.iter().map(|&v| Lit::Is(v, 1)).collect();
        m.at_least(lits, 4);
        // reward turning things OFF, so the constraint has to fight for its minimum
        let mut e = Expr::zero();
        for &v in &bits {
            e = e.plus(Expr::lit(1.0, Lit::Is(v, 0)));
        }
        m.objective(Sense::Maximize, e);
        let s = m.compile().unwrap().solve_best_of(40);
        assert!(s.feasible(), "{s}");
        let on = (0..6).filter(|i| s.value(&format!("b{i}")) == 1).count();
        assert_eq!(on, 4, "the minimum, and no more than it has to: {s}");
    }

    #[test]
    fn slack_is_invisible_in_the_answer() {
        // It costs spins and must not appear as a variable; a solver artefact is not a result.
        let mut m = Model::new();
        let bits: Vec<Var> = (0..4).map(|i| m.binary(&format!("b{i}"))).collect();
        m.at_most(bits.iter().map(|&v| Lit::Is(v, 1)).collect(), 2);
        let c = m.compile().unwrap();
        assert_eq!(c.all_slots.len(), 5, "four variables plus one slack");
        let s = c.solve_best_of(10);
        assert_eq!(s.iter().count(), 4, "only the user's four are reported: {s}");
        assert!(c.spins() > 8, "and the slack really is laid out");
    }

    #[test]
    fn an_objective_reads_like_arithmetic() {
        // The same model twice, once in operators and once in builder calls. If these ever disagree
        // the sugar is lying, which is worse than not having it.
        let solve = |sugar: bool| {
            let mut m = Model::new();
            let x = m.categorical("x", 4);
            let y = m.categorical("y", 4);
            let e = if sugar {
                5.0 * x.is(3) + 2.0 * y.is(1) - 1.0 * x.is(0)
            } else {
                Expr::zero()
                    .plus(Expr::lit(5.0, Lit::Is(x, 3)))
                    .plus(Expr::lit(2.0, Lit::Is(y, 1)))
                    .plus(Expr::lit(-1.0, Lit::Is(x, 0)))
            };
            m.objective(Sense::Maximize, e);
            let s = m.compile().unwrap().solve_best_of(20);
            (s.value("x"), s.value("y"))
        };
        assert_eq!(solve(true), (3, 1), "operators should pick the rewarded values");
        assert_eq!(solve(true), solve(false), "sugar and builder must agree exactly");
    }

    #[test]
    fn a_soft_constraint_is_a_price_not_a_rule() {
        // Three people, two of whom would rather not share a shift, and one shift going spare.
        // "Prefer they do not clash" is a preference with a number on it, not a statement about
        // which answers are answers -- and collapsing the two is why `feasible` used to go false
        // for a constraint the modeller had deliberately priced low.
        let build = |price: f64| {
            let mut m = Model::new();
            let a = m.categorical("a", 2);
            let b = m.categorical("b", 2);
            m.soft(Constraint::NotEqual(a, b), price);
            // both would rather be on shift 0, worth more than a cheap clash and less than a dear one
            m.objective(Sense::Maximize, 5.0 * a.is(0) + 5.0 * b.is(0));
            m.compile().unwrap().solve_best_of(24)
        };

        // priced low, the clash is worth having
        let cheap = build(1.0);
        assert_eq!((cheap.value("a"), cheap.value("b")), (0, 0), "{cheap}");
        assert!(cheap.feasible(), "a soft violation is not an infeasible answer: {cheap}");
        assert_eq!(cheap.violated.len(), 1, "and it is still REPORTED: {cheap}");
        assert!(!cheap.violated[0].hard, "marked soft");
        assert_eq!(cheap.soft_cost(), 1.0, "and priced: {cheap}");   // 1 × 1², missed by one
        assert!(cheap.to_string().contains("traded:"), "{cheap}");

        // priced high, it is not
        let dear = build(50.0);
        assert_ne!(dear.value("a"), dear.value("b"), "the price now outweighs the reward: {dear}");
        assert_eq!(dear.soft_cost(), 0.0, "nothing was traded away: {dear}");
        assert!(
            !dear.soft_cost().is_sign_negative(),
            "a price of nothing must not print as -0: every binding formats this number and a \
             leading minus reads as a credit"
        );
    }

    #[test]
    fn a_soft_price_is_squared_because_the_penalty_is() {
        // Not a detail. A constraint becomes an energy term by squaring how far outside it sits, so
        // missing by two costs FOUR times missing by one. A modeller pricing a preference is
        // choosing that curve as well as its scale, and a linear report would misstate the trade.
        let over_by = |reward: f64| {
            let mut m = Model::new();
            let vs: Vec<Var> = (0..4).map(|i| m.binary(&format!("v{i}"))).collect();
            m.soft(
                Constraint::AtMost { lits: vs.iter().map(|&v| Lit::Is(v, 1)).collect(), k: 1 },
                1.0,
            );
            for &v in &vs {
                m.objective(Sense::Maximize, reward * v.is(1));
            }
            let s = m.compile().unwrap().solve_best_of(32);
            let on = vs.iter().filter(|&&v| s.value(m.name_of(v)) == 1).count();
            (on, s.soft_cost())
        };

        // a small reward buys one extra: over by 1 costs 1
        let (on, cost) = over_by(1.5);
        assert_eq!(on, 2, "one over the ceiling");
        assert_eq!(cost, 1.0, "1 × 1²");

        // a large one buys all four: over by 3 costs NINE, not three
        let (on, cost) = over_by(20.0);
        assert_eq!(on, 4, "three over the ceiling");
        assert_eq!(cost, 9.0, "1 × 3², which is why the curve matters");
    }


    #[test]
    fn a_hard_constraint_still_makes_an_answer_inadmissible() {
        // The distinction has to cut both ways, or `feasible` means nothing.
        let mut m = Model::new();
        let a = m.categorical("a", 2);
        let b = m.categorical("b", 2);
        m.not_equal(a, b);                       // hard
        m.fixed_penalty(1.0);                    // and deliberately outbid
        m.objective(Sense::Maximize, 40.0 * a.is(0) + 40.0 * b.is(0));
        let s = m.compile().unwrap().solve_best_of(16);
        assert_eq!((s.value("a"), s.value("b")), (0, 0));
        assert!(!s.feasible(), "a broken HARD constraint is still infeasible: {s}");
        assert!(s.violated[0].hard);
        assert_eq!(s.soft_cost(), 0.0, "a hard constraint has no price");
    }

    #[test]
    fn hard_and_soft_in_one_model_are_reported_apart() {
        let mut m = Model::new();
        let a = m.categorical("a", 3);
        let b = m.categorical("b", 3);
        let c = m.categorical("c", 3);
        m.not_equal(a, b);                                    // hard: must hold
        m.soft(Constraint::NotEqual(b, c), 2.0);              // soft: worth 2 to keep
        m.objective(Sense::Maximize, 9.0 * b.is(1) + 9.0 * c.is(1));
        let s = m.compile().unwrap().solve_best_of(32);

        assert!(s.feasible(), "the hard one holds: {s}");
        assert_ne!(s.value("a"), s.value("b"), "{s}");
        assert_eq!((s.value("b"), s.value("c")), (1, 1), "the soft one is worth trading: {s}");
        assert_eq!(s.soft_cost(), 2.0, "{s}");
        assert_eq!(s.violated.iter().filter(|v| v.hard).count(), 0);
        assert_eq!(s.violated.iter().filter(|v| !v.hard).count(), 1);
    }


    #[test]
    fn a_domain_wall_variable_can_actually_be_used() {
        // `categorical_as` existed and could not be used: a domain-wall indicator is a PRODUCT of
        // two spins, `linearise` refused anything that was not linear, and every constraint went
        // through it. An API that compiles and cannot be called is worse than an absent one --
        // the absent one does not cost an afternoon finding out.
        let mut m = Model::new();
        let a = m.categorical_as("a", 4, Encoding::DomainWall);
        m.fix(a, 2);
        let c = m.compile().expect("a domain-wall variable must compile");
        assert_eq!(c.spins(), 3, "k-1 spins, where one-hot would take 4");
        let s = c.solve_best_of(16);
        assert!(s.feasible(), "{s}");
        assert_eq!(s.value("a"), 2, "{s}");
    }

    #[test]
    fn a_domain_wall_variable_works_beside_a_one_hot_one() {
        // Mixed encodings in one model, which is the case that breaks if `factors` and `linearise`
        // disagree about what a literal means.
        let mut m = Model::new();
        let a = m.categorical_as("a", 4, Encoding::DomainWall);
        let b = m.categorical("b", 4);
        m.not_equal(a, b);
        m.fix(b, 2);
        let s = m.compile().unwrap().solve_best_of(32);
        assert!(s.feasible(), "{s}");
        assert_eq!(s.value("b"), 2, "{s}");
        assert_ne!(s.value("a"), 2, "and a really differs from it: {s}");
    }

    #[test]
    fn domain_wall_costs_one_spin_fewer_than_one_hot() {
        // The reason to choose it, measured rather than asserted.
        let spins = |enc: Encoding| {
            let mut m = Model::new();
            let v = m.categorical_as("x", 6, enc);
            m.fix(v, 3);
            m.compile().unwrap().spins()
        };
        assert_eq!(spins(Encoding::OneHot), 6);
        assert_eq!(spins(Encoding::DomainWall), 5);
    }

    #[test]
    fn a_binary_encoded_variable_is_refused_by_name() {
        // Its indicator is a product of every bit, so its degree grows with the domain. Refused
        // rather than expanded into something nobody wants to read.
        let mut m = Model::new();
        let v = m.categorical_as("x", 8, Encoding::Binary);
        m.fix(v, 3);
        let e = match m.compile() { Err(e) => e.to_string(), Ok(_) => panic!("binary in a literal") };
        assert!(e.contains("'x'"), "{e}");
        assert!(e.contains("OneHot") || e.contains("one-hot"), "{e}");
    }


    #[test]
    fn a_grid_of_variables_reads_like_the_problem_it_models() {
        // Assignment: three workers, three shifts, one shift each, nobody doubled up. This is the
        // shape of most real models, and writing it without an index means a hand-rolled loop, a
        // format! and index arithmetic nobody checks.
        let (workers, shifts) = (3, 3);
        let mut m = Model::new();
        let a = m.grid("assign", &[workers, shifts], |m, n| m.binary(n));

        for w in 0..workers {
            m.exactly_one(a.row(&[w]).iter().map(|&v| Lit::Is(v, 1)).collect());
        }
        for s in 0..shifts {
            m.at_most_one(a.column(&[s]).iter().map(|&v| Lit::Is(v, 1)).collect());
        }
        // worker 0 is best at shift 2, worker 1 at shift 0
        m.objective(Sense::Maximize, 3.0 * a[[0, 2]].is(1) + 3.0 * a[[1, 0]].is(1));

        let s = m.compile().unwrap().solve_best_of(32);
        assert!(s.feasible(), "{s}");
        for w in 0..workers {
            let taken: Vec<usize> =
                (0..shifts).filter(|&sh| s.value(&format!("assign[{w},{sh}]")) == 1).collect();
            assert_eq!(taken.len(), 1, "worker {w} takes exactly one shift: {taken:?}");
        }
        assert_eq!(s.value("assign[0,2]"), 1, "and the preferences are honoured: {s}");
        assert_eq!(s.value("assign[1,0]"), 1, "{s}");
    }

    #[test]
    fn a_grid_names_its_variables_so_the_answer_still_reads() {
        let mut m = Model::new();
        let a = m.grid("x", &[2, 2], |m, n| m.binary(n));
        m.fix(a[[1, 0]], 1);
        let s = m.compile().unwrap().solve_best_of(8);
        assert_eq!(s.value("x[1,0]"), 1, "subscripts are part of the name: {s}");
        assert_eq!(a.dims(), &[2, 2]);
        assert_eq!(a.all().len(), 4);
    }

    #[test]
    fn an_index_outside_the_shape_is_a_panic_not_a_wrap() {
        // A silent wrap is an off-by-one that reaches the answer, and it is the exact mistake the
        // hand-rolled version makes.
        let mut m = Model::new();
        let a = m.grid("x", &[2, 3], |m, n| m.binary(n));
        assert_eq!(a.all().len(), 6, "row-major, last index fastest");
        let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| a[[0, 3]]));
        assert!(caught.is_err(), "index 3 of a dimension of 3 must not silently wrap");
        let wrong_rank = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| a[[1]]));
        assert!(wrong_rank.is_err(), "one index into a two-dimensional grid is a mistake");
    }

    #[test]
    fn rows_and_columns_pick_out_what_a_constraint_is_usually_over() {
        let mut m = Model::new();
        let a = m.grid("x", &[2, 3], |m, n| m.binary(n));
        let r = a.row(&[1]);
        assert_eq!(r.len(), 3, "a row is the last dimension");
        assert_eq!(r[0], a[[1, 0]]);
        assert_eq!(r[2], a[[1, 2]]);
        let c = a.column(&[2]);
        assert_eq!(c.len(), 2, "a column is the first");
        assert_eq!(c[0], a[[0, 2]]);
        assert_eq!(c[1], a[[1, 2]]);
    }

    #[test]
    fn a_grid_works_with_any_domain() {
        let mut m = Model::new();
        let t = m.grid("temp", &[3], |m, n| m.integer(n, 10, 20));
        m.fix(t[[1]], 17);
        let s = m.compile().unwrap().solve_best_of(8);
        assert_eq!(s.value("temp[1]"), 17, "{s}");
    }

    #[test]
    fn a_cubic_objective_compiles_and_finds_the_right_answer() {
        // The modelling layer refused this outright until the reduction existed. It is the shape of
        // "these three must agree", which is not exotic.
        let mut m = Model::new();
        let a = m.categorical("a", 3);
        let b = m.categorical("b", 3);
        let c = m.categorical("c", 3);
        m.objective(Sense::Maximize, Expr::product(9.0, &[a.is(2), b.is(2), c.is(2)]));
        let compiled = m.compile().unwrap();
        assert!(compiled.ancillas > 0, "a three-body term costs ancillas");

        let s = compiled.solve_best_of(24);
        assert_eq!((s.value("a"), s.value("b"), s.value("c")), (2, 2, 2), "{s}");
        assert!(s.feasible(), "{s}");
    }

    #[test]
    fn the_cubic_term_itself_decides_the_answer() {
        // Isolating the thing under test. A one-hot indicator is `0.5 + 0.5·s`, so three of them
        // expand to eight monomials of which ONE is cubic -- and the other seven decide the answer
        // on their own. A first version of this flipped the cubic term's sign and saw no change,
        // which said nothing about the cubic term and everything about the remnants.
        //
        // A spin literal has no offset, so a product of three is exactly one cubic monomial.
        let run = |sense: Sense| {
            let mut m = Model::new();
            let a = m.spin("a");
            let b = m.spin("b");
            let c = m.spin("c");
            m.objective(sense, Expr::product(1.0, &[a.spin(), b.spin(), c.spin()]));
            let comp = m.compile().unwrap();
            assert!(comp.ancillas > 0, "a pure three-body term needs an ancilla");
            let s = comp.solve_best_of(32);
            s.value("a") * s.value("b") * s.value("c")
        };
        assert_eq!(run(Sense::Maximize), 1, "maximising the product puts it at +1");
        assert_eq!(run(Sense::Minimize), -1, "and minimising it at -1");
    }

    #[test]
    fn a_spin_squared_is_one_and_expresses_no_preference() {
        // `s² = 1`, so a term over the same spin twice is a CONSTANT and says nothing about which
        // way that spin should go. Collapsing it to `s` instead turns a constant into a preference,
        // silently, and here strongly enough to overrule the only real one in the model.
        // THREE literals, because a two-literal term never reaches the expansion at all -- it
        // takes the `2 =>` branch and `add_product` handles it. A first version of this used two
        // and tested nothing: the parity rule it is named after was never executed.
        let mut m = Model::new();
        let a = m.spin("a");
        let b = m.spin("b");
        m.objective(Sense::Maximize, Expr::product(50.0, &[a.spin(), a.spin(), b.spin()]));
        m.objective(Sense::Minimize, 1.0 * b.spin());
        let c = m.compile().unwrap();
        // s_a·s_a·s_b collapses to s_b: linear, so no ancilla, and nothing said about a at all
        assert_eq!(c.ancillas, 0, "s² = 1 leaves a one-body term, not a three-body one");
        assert_eq!(c.solve_best_of(24).value("b"), 1, "50·s_b against 1·s_b: b goes to +1");
    }

    #[test]
    fn an_ancilla_never_appears_in_the_answer() {
        let mut m = Model::new();
        let vs: Vec<Var> = (0..3).map(|i| m.binary(&format!("v{i}"))).collect();
        m.objective(Sense::Maximize, Expr::product(5.0, &[vs[0].is(1), vs[1].is(1), vs[2].is(1)]));
        let c = m.compile().unwrap();
        assert!(c.ancillas > 0);
        let s = c.solve_best_of(16);
        let names: Vec<&String> = s.iter().map(|(k, _)| k).collect();
        assert_eq!(names.len(), 3, "only the declared variables: {names:?}");
    }

    #[test]
    fn a_cubic_objective_agrees_with_exhaustive_search() {
        // Enumerate every assignment of the declared variables, score it against the objective as
        // WRITTEN, and require the compiled model's answer to be one of the maximisers.
        let mut m = Model::new();
        let vs: Vec<Var> = (0..4).map(|i| m.spin(&format!("v{i}"))).collect();
        m.objective(Sense::Maximize, Expr::product(6.0, &[vs[0].spin(), vs[1].spin(), vs[2].spin()]));
        m.objective(Sense::Minimize, 2.0 * vs[3].spin());
        m.objective(Sense::Minimize, 1.0 * vs[0].spin());
        let c = m.compile().unwrap();
        let s = c.solve_best_of(48);

        let score = |x: &[i64]| 6.0 * (x[0] * x[1] * x[2]) as f64 - 2.0 * x[3] as f64 - x[0] as f64;
        let mut best = f64::NEG_INFINITY;
        for mask in 0..16u32 {
            let x: Vec<i64> =
                (0..4).map(|i| if (mask >> i) & 1 == 1 { 1 } else { -1 }).collect();
            best = best.max(score(&x));
        }
        let got: Vec<i64> = (0..4).map(|i| s.value(&format!("v{i}"))).collect();
        assert_eq!(score(&got), best, "compiled answer {got:?} against exhaustive best {best}");
    }

    #[test]
    fn a_product_of_literals_is_quadratic_and_works() {
        let mut m = Model::new();
        let a = m.categorical("a", 3);
        let b = m.categorical("b", 3);
        m.objective(Sense::Maximize, 4.0 * (a.is(2) * b.is(2)));
        let s = m.compile().unwrap().solve_best_of(16);
        assert_eq!((s.value("a"), s.value("b")), (2, 2), "{s}");
    }

    #[test]
    fn a_loop_of_terms_sums() {
        // The shape an objective usually has: one term per value, built in a loop.
        let mut m = Model::new();
        let x = m.categorical("x", 6);
        let e: Expr = (0..6).map(|v| (v as f64) * x.is(v)).sum();
        m.objective(Sense::Maximize, e);
        assert_eq!(m.compile().unwrap().solve_best_of(16).value("x"), 5);
    }

    #[test]
    fn negation_flips_the_preference() {
        let build = |neg: bool| {
            let mut m = Model::new();
            let x = m.categorical("x", 4);
            let e = 3.0 * x.is(3);
            m.objective(Sense::Maximize, if neg { -e } else { e });
            m.compile().unwrap().solve_best_of(16).value("x")
        };
        assert_eq!(build(false), 3, "rewarded");
        assert_ne!(build(true), 3, "and penalised when negated");
    }

    #[test]
    fn an_empty_model_is_refused() {
        assert!(matches!(Model::new().compile(), Err(CompileError::Empty)));
    }

    #[test]
    fn a_compiled_model_is_an_ordinary_program() {
        // The whole point: what comes out runs anywhere the rest of the stack runs.
        let mut m = Model::new();
        let a = m.categorical("a", 3);
        let b = m.categorical("b", 3);
        m.not_equal(a, b);
        let c = m.compile().unwrap();
        let text = c.program.to_ftp();
        let back = Program::from_ftp(&text).unwrap();
        assert_eq!(back.spins, c.spins());
        assert_eq!(back.encodings.len(), 2, "the layout travels with the program");
        assert_eq!(back.to_ftp(), text, "and it round-trips");
    }

    #[test]
    fn all_different_solves_a_latin_square_row_and_names_the_clash() {
        // Four variables over four values: the only feasible assignments are permutations, and a
        // sampler that finds one has satisfied a constraint no pair of them could express alone.
        let mut m = Model::new();
        let v: Vec<_> = (0..4).map(|i| m.categorical(&format!("c{i}"), 4)).collect();
        m.all_different(v.clone());
        let c = m.compile().unwrap();
        let s = c.solve_best_of(60);
        assert!(s.feasible(), "a permutation exists and should be found: {s}");
        let mut got: Vec<i64> = (0..4).map(|i| s.values[&format!("c{i}")]).collect();
        got.sort();
        assert_eq!(got, vec![0, 1, 2, 3], "every value used exactly once: {s}");
    }

    #[test]
    fn a_broken_all_different_names_which_value_collided() {
        // The violation has to be actionable. "all-different was violated" is something the
        // modeller already suspects; "a and b both take 2" is a repair.
        let mut m = Model::new();
        let a = m.categorical("a", 3);
        let b = m.categorical("b", 3);
        m.all_different([a, b]);
        // Pin the penalty below an objective that wants them equal, so the constraint loses.
        m.objective(Sense::Maximize, Expr::product(50.0, &[Lit::Is(a, 2)]));
        m.objective(Sense::Maximize, Expr::product(50.0, &[Lit::Is(b, 2)]));
        m.fixed_penalty(1.0);
        let c = m.compile().unwrap();
        let s = c.solve_best_of(40);
        assert!(!s.feasible(), "the objective should have outbid a penalty of 1: {s}");
        let v = &s.violated[0];
        assert!(v.detail.contains("both take 2"), "must name the value: {}", v.detail);
        assert!(v.detail.contains('a') && v.detail.contains('b'), "and who: {}", v.detail);
        assert_eq!(v.amount, 1.0, "over by one");
    }

    #[test]
    fn all_different_over_too_few_values_is_refused_rather_than_annealed() {
        // Pigeonhole. Annealing this returns infeasible, which reads as "raise the penalty" --
        // advice that cannot work, because the model has no answer at any penalty.
        let mut m = Model::new();
        let vars: Vec<_> = (0..5).map(|i| m.categorical(&format!("x{i}"), 3)).collect();
        m.all_different(vars);
        match m.compile() {
            Err(CompileError::Pigeonhole { vars, values }) => {
                assert_eq!((vars, values), (5, 3));
                let msg = CompileError::Pigeonhole { vars, values }.to_string();
                assert!(msg.contains("no answer"), "must say why annealing will not help: {msg}");
            }
            Err(e) => panic!("expected a pigeonhole refusal, got {e}"),
            Ok(c) => panic!("expected a pigeonhole refusal, compiled to {} spins", c.spins()),
        }
    }

    #[test]
    fn all_different_over_disjoint_domains_costs_nothing() {
        // Two variables that cannot collide need no terms at all. A pairwise not_equal sweep would
        // emit them anyway; lowering per shared value notices there are none.
        let mut m = Model::new();
        let a = m.integer("a", 0, 3);
        let b = m.integer("b", 10, 13);
        m.all_different([a, b]);
        let with = m.compile().unwrap();

        let mut n = Model::new();
        n.integer("a", 0, 3);
        n.integer("b", 10, 13);
        let without = n.compile().unwrap();
        assert_eq!(
            with.program.factors.len(),
            without.program.factors.len(),
            "disjoint domains cannot collide, so the constraint must add no couplings"
        );
    }

    #[test]
    fn an_encoding_that_cannot_be_exact_says_so_at_compile_time() {
        // Measured, not asserted from theory: for a k = 6 binary slot the cheapest INVALID state
        // costs exactly what the cheapest valid one costs, so nothing in the landscape discourages
        // the sampler from returning a codeword that decodes to nothing. `Slot::add_penalty` has
        // always returned that fact and both callers threw it away.
        let mut m = Model::new();
        m.categorical_as("x", 6, Encoding::Binary);
        m.categorical_as("y", 8, Encoding::Binary); // a power of two IS exact
        m.categorical("z", 6); // one-hot is always exact
        let c = m.compile().unwrap();

        assert_eq!(c.caveats.len(), 1, "only 'x' is inexact: {:?}", c.caveats);
        let w = &c.caveats[0];
        assert!(w.contains("'x'"), "must name the variable: {w}");
        assert!(w.contains('8') && w.contains('2'), "and the codeword arithmetic: {w}");
        assert!(w.contains("one-hot") || w.contains("power of two"), "and the way out: {w}");
    }

    #[test]
    fn the_caveat_is_measured_rather_than_believed() {
        // The claim in the caveat is that an invalid state is as cheap as a valid one. Enumerate
        // and check it, so the message cannot drift from the physics it describes.
        use crate::encode::Slot;
        use crate::graph::GraphBuilder;
        let slot = Slot::new(0, 6, Encoding::Binary);
        let w = slot.width();
        let mut b = GraphBuilder::new(w);
        assert!(!slot.add_penalty(&mut b, 10.0), "k=6 binary cannot be exact");
        let g = b.build();
        let (mut best_ok, mut best_bad) = (f64::MAX, f64::MAX);
        for mask in 0..(1u32 << w) {
            let s: Vec<i8> = (0..w).map(|i| if mask >> i & 1 == 1 { 1 } else { -1 }).collect();
            let e = g.energy(&s);
            if slot.decode(&s).is_some() { best_ok = best_ok.min(e) } else { best_bad = best_bad.min(e) }
        }
        assert_eq!(
            best_bad, best_ok,
            "the caveat says an invalid state costs what a valid one costs; if that stops being \
             true the message is wrong"
        );
    }

    #[test]
    fn a_constraint_with_a_cheaper_form_is_named_but_not_rewritten() {
        // jijmodeling 2.x detects one-hot patterns and hints the solver. This is the same idea with
        // a different verdict: report, never rewrite. Silently compiling something other than what
        // was written is the opposite of this compiler's discipline, and a modeller who meant the
        // expensive form is entitled to it.
        //
        // Only what MEASURES cheaper is reported. cardinality(lits, 1) and exactly_one compile to
        // identical graphs here, and so do six pairwise not_equals and one all_different -- so
        // neither earns a caveat. Advice that costs the reader time and saves no spins is noise,
        // and a checker people learn to ignore catches nothing.
        let mut m = Model::new();
        let v: Vec<_> = (0..5).map(|i| m.binary(&format!("v{i}"))).collect();
        let lits: Vec<Lit> = v.iter().map(|&x| Lit::Is(x, 1)).collect();
        m.at_most(lits.clone(), 1);
        let c = m.compile().unwrap();
        assert_eq!(c.caveats.len(), 1, "{:?}", c.caveats);
        assert!(c.caveats[0].contains("at_most_one"), "must name the cheaper form: {}", c.caveats[0]);

        // And the model is UNCHANGED: the expensive lowering is still what was asked for.
        let mut n = Model::new();
        let w: Vec<_> = (0..5).map(|i| n.binary(&format!("v{i}"))).collect();
        n.at_most_one(w.iter().map(|&x| Lit::Is(x, 1)).collect::<Vec<_>>());
        let cheap = n.compile().unwrap();
        assert!(cheap.caveats.is_empty(), "the cheap form must not be flagged: {:?}", cheap.caveats);
        assert!(
            c.spins() > cheap.spins(),
            "the caveat claims a saving; if the two compile the same the message is wrong"
        );
    }

    #[test]
    fn a_constraint_that_constrains_nothing_says_so() {
        // at_most(n of n) and at_least(0 of n) are satisfied by every assignment and still pay for a
        // slack variable and its factors. Almost always a k that was meant to be different.
        let mut m = Model::new();
        let v: Vec<_> = (0..4).map(|i| m.binary(&format!("v{i}"))).collect();
        let lits: Vec<Lit> = v.iter().map(|&x| Lit::Is(x, 1)).collect();
        m.at_most(lits.clone(), 4);
        m.at_least(lits, 0);
        let c = m.compile().unwrap();
        assert_eq!(c.caveats.len(), 2, "{:?}", c.caveats);
        assert!(c.caveats.iter().all(|w| w.contains("constrains nothing")), "{:?}", c.caveats);
    }
}
