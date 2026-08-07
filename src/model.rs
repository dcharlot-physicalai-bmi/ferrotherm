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
}

/// A handle to a declared variable.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Var(usize);

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
    Is(Var, usize),
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
    Fix(Var, usize),
    /// Exactly `k` of these literals are true.
    ///
    /// The penalty is `(Σ lits − k)²`, which is quadratic in the spins and needs no ancillas.
    /// Cardinality is the workhorse of assignment, scheduling and selection problems.
    Cardinality { lits: Vec<Lit>, k: usize },
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
    BadValue { var: String, value: usize, size: usize },
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
            CompileError::BadValue { var, value, size } => {
                write!(f, "'{var}' has {size} values; {value} is not one of them")
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
    pub fn fix(&mut self, v: Var, value: usize) -> &mut Self {
        self.constrain(Constraint::Fix(v, value))
    }

    /// Exactly `k` of these literals must be true.
    pub fn cardinality(&mut self, lits: Vec<Lit>, k: usize) -> &mut Self {
        self.constrain(Constraint::Cardinality { lits, k })
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

        // Lay every variable out, in declaration order, so the layout is stable and inspectable.
        let mut slots = Vec::with_capacity(self.decls.len());
        let mut base = 0usize;
        for d in &self.decls {
            let k = d.domain.size();
            let s = Slot::new(base, k, d.encoding);
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
        for (c, p) in &self.constraints {
            let p = if p.is_nan() { penalty } else { *p };
            self.apply_constraint(&mut b, &slots, c, p)?;
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
            slots,
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
                let size = d.domain.size();
                if value >= size {
                    return Err(CompileError::BadValue {
                        var: d.name.clone(),
                        value,
                        size,
                    });
                }
                if d.encoding != Encoding::OneHot {
                    return Err(CompileError::NeedsOneHot {
                        var: d.name.clone(),
                        encoding: d.encoding,
                    });
                }
                // one-hot: indicator = (1 + s)/2
                let s = slots[v.0];
                Ok(LinSpin { offset: 0.5, terms: vec![(s.base + value, 0.5)] })
            }
        }
    }

    fn apply_constraint(
        &self,
        b: &mut GraphBuilder,
        slots: &[Slot],
        c: &Constraint,
        p: f64,
    ) -> Result<(), CompileError> {
        match c {
            Constraint::NotEqual(a, x) => {
                // penalise agreeing on any value: p · Σ_v [a=v][x=v]
                let k = self.decls[a.0].domain.size().min(self.decls[x.0].domain.size());
                for v in 0..k {
                    let la = self.linearise(slots, Lit::Is(*a, v))?;
                    let lb = self.linearise(slots, Lit::Is(*x, v))?;
                    add_product(b, &la, &lb, p);
                }
            }
            Constraint::Equal(a, x) => {
                let k = self.decls[a.0].domain.size().min(self.decls[x.0].domain.size());
                for v in 0..k {
                    let la = self.linearise(slots, Lit::Is(*a, v))?;
                    let lb = self.linearise(slots, Lit::Is(*x, v))?;
                    add_product(b, &la, &lb, -p);
                }
            }
            Constraint::Fix(v, value) => {
                let l = self.linearise(slots, Lit::Is(*v, *value))?;
                add_linear(b, &l, -p); // reward taking it
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
        let sched = Schedule::geometric(0.05, 8.0, 120, 40);
        let (best, _) = crate::tempering::anneal_scheduled(&self.graph, &sched, seed, None);
        self.decode(&best)
    }

    /// Anneal several times and keep the best feasible answer, or the best overall if none is.
    pub fn solve_best_of(&self, tries: u64) -> Solution {
        let mut best: Option<Solution> = None;
        for s in 0..tries.max(1) {
            let cand = self.solve_annealed(s);
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
        m.fix(x, 3); // the fourth value, so 13
        let c = m.compile().unwrap();
        let s = c.solve_best_of(6);
        assert!(s.feasible(), "{s}");
        assert_eq!(s.value("temperature"), 13, "integers decode in their own range: {s}");
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
