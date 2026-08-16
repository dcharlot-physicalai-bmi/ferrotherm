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
#[derive(Clone, Copy, Debug, PartialEq)]
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
        let signed = if sense == Sense::Maximize { e.scaled(-1.0) } else { e };
        let acc = core::mem::take(&mut self.objective);
        self.objective = acc.plus(signed);
        self
    }

    /// Discard everything accumulated so far and use exactly this objective.
    pub fn set_objective(&mut self, sense: Sense, e: Expr) -> &mut Self {
        self.objective = Expr::zero();
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
        let worst = self
            .objective
            .terms
            .iter()
            .map(|t| t.coeff.abs())
            .fold(0.0f64, f64::max);
        // Twice the largest objective coefficient. A constraint that merely ties with the objective
        // gets traded away, and the result is a state that does not decode rather than one that
        // scores badly -- which reads as a broken sampler rather than an under-weighted constraint.
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
        let n = base;

        let mut b = GraphBuilder::new(n);
        let penalty = self.effective_penalty();

        // Encoding penalties: what makes a spin pattern mean a value at all.
        //
        // The returned bool is not decoration. `false` means invalid codewords remain exactly as
        // cheap as valid ones, so the sampler is free to land on one and `decode` is the only
        // thing between that and a wrong answer. Collected rather than discarded.
        let mut caveats = Vec::new();
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
                    self.decls[i].name,
                    s.encoding,
                    1usize << spins,
                ));
            }
        }

        // Constraints. A NaN strength means "use the model's, after scaling".
        // Anything wider than pairwise, from a constraint or an objective term alike, lands here
        // and is lowered by `reduce` once the graph is built.
        let mut higher: Vec<(Vec<usize>, f64)> = Vec::new();

        for (ci, (c, p, _hard)) in self.constraints.iter().enumerate() {
            let p = if p.is_nan() { penalty } else { *p };
            self.apply_constraint(&mut b, &mut higher, &slots, c, p, slack_for.get(&ci).copied())?;
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
            program,
            graph,
            ancillas,
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

    fn apply_constraint(
        &self,
        b: &mut GraphBuilder,
        higher: &mut Vec<(Vec<usize>, f64)>,
        slots: &[Slot],
        c: &Constraint,
        p: f64,
        slack: Option<usize>,
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
                add_squared(&mut b_ref(b), &sum, *k as f64, p);
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

/// A compiled model: the program, plus the means to read an answer back.
pub struct Compiled {
    pub program: Program,
    pub graph: crate::graph::Graph,
    slots: Vec<Slot>,
    /// Including the compiler's own slack variables, for anyone inspecting the lowering.
    pub all_slots: Vec<Slot>,
    names: Vec<String>,
    domains: Vec<Domain>,
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
        Solution { values, invalid, violated, energy: self.graph.energy(state) }
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
    pub energy: f64,
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
    use super::*;

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
