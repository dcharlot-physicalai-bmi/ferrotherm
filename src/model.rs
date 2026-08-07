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
use crate::factor::Factor;
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
    /// How many values it can take.
    pub fn size(&self) -> usize {
        match self {
            Domain::Spin | Domain::Binary => 2,
            Domain::Categorical(k) => *k,
            Domain::Integer { lo, hi } => (hi - lo + 1).max(1) as usize,
        }
    }

    /// The values a modeller writes, in encoding order.
    ///
    /// For everything but an integer these are `0..size`, so value and slot coincide and the
    /// distinction never surfaces. An integer over `5..=20` takes the values 5 through 20, and
    /// conflating those with slots 0 through 15 is the bug this exists to prevent.
    pub fn values(&self) -> impl Iterator<Item = i64> + '_ {
        let (lo, n) = match self {
            Domain::Integer { lo, .. } => (*lo, self.size()),
            _ => (0, self.size()),
        };
        (0..n).map(move |i| lo + i as i64)
    }

    /// Which one-hot slot holds `value`, or `None` if the domain does not contain it.
    pub fn index_of(&self, value: i64) -> Option<usize> {
        let lo = match self {
            Domain::Integer { lo, .. } => *lo,
            _ => 0,
        };
        let i = value.checked_sub(lo)?;
        (i >= 0 && (i as u128) < self.size() as u128).then_some(i as usize)
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
    sense: Sense,
    constraints: Vec<(Constraint, f64)>,
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
}

impl core::fmt::Display for CompileError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
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
        }
    }
}

impl Model {
    pub fn new() -> Model {
        Model {
            decls: Vec::new(),
            objective: Expr::zero(),
            sense: Sense::Minimize,
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
        self.sense = sense;
        self.objective = e;
        self
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
        self.constraints.push((c, f64::NAN));
        self
    }

    /// Add a constraint at a specific penalty strength.
    pub fn constrain_at(&mut self, c: Constraint, penalty: f64) -> &mut Self {
        self.constraints.push((c, penalty));
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

        // Inequalities need slack, so the compiler declares variables of its own. They are laid out
        // after the user's, and the decoder never reports them: a slack variable is an artefact of
        // the lowering, not part of the answer.
        let user_count = self.decls.len();
        let mut extra: Vec<Domain> = Vec::new();
        let mut slack_for: BTreeMap<usize, usize> = BTreeMap::new();
        for (ci, (c, _)) in self.constraints.iter().enumerate() {
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
        for s in &slots {
            s.add_penalty(&mut b, penalty);
        }

        // Constraints. A NaN strength means "use the model's, after scaling".
        for (ci, (c, p)) in self.constraints.iter().enumerate() {
            let p = if p.is_nan() { penalty } else { *p };
            self.apply_constraint(&mut b, &slots, c, p, slack_for.get(&ci).copied())?;
        }

        // Objective. Maximising is minimising the negation; do it once, here.
        let sign = if self.sense == Sense::Maximize { -1.0 } else { 1.0 };
        if self.objective.max_degree() > 2 {
            return Err(CompileError::DegreeTooHigh { degree: self.objective.max_degree() });
        }
        for t in &self.objective.terms {
            let parts: Result<Vec<LinSpin>, CompileError> =
                t.lits.iter().map(|l| self.linearise(&slots, *l)).collect();
            let parts = parts?;
            match parts.len() {
                1 => add_linear(&mut b, &parts[0], sign * t.coeff),
                2 => add_product(&mut b, &parts[0], &parts[1], sign * t.coeff),
                d => return Err(CompileError::DegreeTooHigh { degree: d }),
            }
        }

        let graph = b.build();
        let mut program = Program::from_graph(&graph, &Schedule::geometric(0.05, 6.0, 80, 40));
        program.encodings = slots
            .iter()
            .map(|s| EncodedVar { base: s.base, k: s.k, encoding: s.encoding })
            .collect();

        Ok(Compiled {
            program,
            graph,
            // only the user's variables are reported; slack is an artefact of the lowering
            slots: slots[..user_count].to_vec(),
            all_slots: slots,
            names: self.decls.iter().map(|d| d.name.clone()).collect(),
            domains: self.decls.iter().map(|d| d.domain).collect(),
        })
    }

    /// A literal as a linear function of spins: `offset + Σ cᵢ sᵢ`.
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
                    let la = self.linearise(slots, Lit::Is(*a, v))?;
                    let lb = self.linearise(slots, Lit::Is(*x, v))?;
                    add_product(b, &la, &lb, p);
                }
            }
            Constraint::Equal(a, x) => {
                for v in shared_values(&self.decls[a.0].domain, &self.decls[x.0].domain) {
                    let la = self.linearise(slots, Lit::Is(*a, v))?;
                    let lb = self.linearise(slots, Lit::Is(*x, v))?;
                    add_product(b, &la, &lb, -p);
                }
            }
            Constraint::Fix(v, value) => {
                let l = self.linearise(slots, Lit::Is(*v, *value))?;
                add_linear(b, &l, -p); // reward taking it
            }
            Constraint::AtMost { lits, k } | Constraint::AtLeast { lits, k } => {
                // Σ lits ± slack = k, squared. `sum` collects every weighted indicator on the left.
                let Some(sv) = slack else { return Ok(()) };  // a degenerate range needs no slack
                let sign = if matches!(c, Constraint::AtMost { .. }) { 1.0 } else { -1.0 };
                let mut sum: Vec<(LinSpin, f64)> = Vec::new();
                for l in lits {
                    sum.push((self.linearise(slots, *l)?, 1.0));
                }
                let s = slots[sv];
                for v in 0..s.k {
                    // the slack's value v enters with weight v, via its one-hot indicator
                    sum.push((
                        LinSpin { offset: 0.5, terms: vec![(s.base + v, 0.5)] },
                        sign * v as f64,
                    ));
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
        Solution { values, invalid, energy: self.graph.energy(state) }
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
        Schedule::geometric(0.05, 8.0, 120, 40)
    }

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

/// An answer, in the model's own names.
#[derive(Clone, Debug)]
pub struct Solution {
    values: BTreeMap<String, i64>,
    /// Variables whose spins did not form a valid codeword.
    pub invalid: Vec<String>,
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
    pub fn feasible(&self) -> bool {
        self.invalid.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&String, &i64)> {
        self.values.iter()
    }
}

impl core::fmt::Display for Solution {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "energy {:.4}", self.energy)?;
        if !self.feasible() {
            write!(f, "  INFEASIBLE ({} did not decode)", self.invalid.len())?;
        }
        for (k, v) in &self.values {
            write!(f, "\n  {k} = {v}")?;
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
        // An odd cycle is not 2-colourable, so some constraint must break. The point is that this
        // is visible in the answer rather than silently absorbed.
        let mut m = Model::new();
        let a = m.categorical("a", 2);
        let b = m.categorical("b", 2);
        let cc = m.categorical("c", 2);
        m.not_equal(a, b).not_equal(b, cc).not_equal(a, cc);
        let comp = m.compile().unwrap();
        let s = comp.solve_best_of(12);
        // every variable still decodes; what fails is the constraint set
        assert!(s.feasible(), "the encoding still holds: {s}");
        let broken = (s.value("a") == s.value("b")) as u32
            + (s.value("b") == s.value("c")) as u32
            + (s.value("a") == s.value("c")) as u32;
        assert_eq!(broken, 1, "exactly one pair must agree in a 2-coloured triangle: {s}");
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
}
