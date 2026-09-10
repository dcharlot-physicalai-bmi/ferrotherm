//! Bayesian networks: a directed model, evidence, and exact answers.
//!
//! A Bayes net is a product of conditional probability tables, one per variable, over its family —
//! the variable and its parents. Turning that into inference is two steps, and only the first is
//! specific to directed models:
//!
//! 1. **Moralisation.** Each `CPT` becomes one factor over its whole family. A child with two
//!    parents is a factor of RANK THREE, which is the same thing as saying the two parents are
//!    adjacent in the undirected model even though the directed graph has no edge between them.
//!    Dropping that adjacency is the classic conversion bug: everything a chain can test still
//!    passes, and the network silently loses **explaining away** — the dependence a common
//!    observed child induces between independent causes.
//! 2. **Elimination.** Conditioning fixes the observed coordinates of each factor, and contracting
//!    what is left answers the query. [`crate::tensor::Network`] is the crate's elimination engine
//!    for factors of any rank, so this module builds tensors and contracts them rather than
//!    reimplementing elimination. [`crate::exact::Elimination`] cannot be the engine directly: it
//!    takes a [`crate::graph::Graph`], which is pairwise, and a two-parent family is not.
//!
//! # Log space, not probability space
//!
//! The tensors carry `ln P`, contracted over [`LogSumExp`] rather than
//! [`crate::tensor::SumProduct`]. A joint probability is a product of numbers below one, so a few
//! hundred factors is enough to flush `P(e)` to zero in `f64` and take every marginal with it —
//! `0/0`. In log space the same query is a sum, and `tests` checks a 64-variable chain whose
//! evidence probability is `1e-378`: exactly representable as a log, exactly zero as a float.
//!
//! The same tensors negated are energies, so [`crate::tensor::Tropical`] gives the most probable
//! explanation from the identical network — one construction, two arithmetics.
//!
//! # Scope
//!
//! Binary variables only, `false` and `true`, which map to spins `-1` and `+1`. Marginal MAP —
//! maximise over some variables while summing out others — is NOT offered: it is hard even at
//! bounded treewidth, and an approximation returned from a module called "exact" would be a lie.

use crate::graph::{Graph, GraphBuilder};
use crate::tensor::{Index, Network, Order, Semiring, Tensor, Tropical, Uncontractable};

/// `(R and -inf, logaddexp, +)` — sum-product carried out in LOG space.
///
/// The arithmetic [`crate::tensor`] was missing. `⊗` is `+` because a product of probabilities is
/// a sum of logs, and `⊕` is `logaddexp`, computed against the larger operand so it never
/// overflows. `zero` is `-inf` (an impossible configuration) and `one` is `0`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LogSumExp;

impl Semiring for LogSumExp {
    type Elem = f64;
    const NAME: &'static str = "log-sum-exp";
    fn zero() -> f64 {
        f64::NEG_INFINITY
    }
    fn one() -> f64 {
        0.0
    }
    fn add(a: f64, b: f64) -> f64 {
        // `-inf` must be a true identity: shifting by the larger operand when both are `-inf`
        // computes `-inf + ln(NaN + NaN)`, so the identity is taken before the shift.
        if a == f64::NEG_INFINITY {
            return b;
        }
        if b == f64::NEG_INFINITY {
            return a;
        }
        let m = if a > b { a } else { b };
        m + ((a - m).exp() + (b - m).exp()).ln()
    }
    fn mul(a: f64, b: f64) -> f64 {
        a + b
    }
}

/// The most parents one variable may have; its table has `2^parents` rows.
const MAX_PARENTS: usize = 20;

/// Why a network could not be built.
#[derive(Clone, Debug, PartialEq)]
pub enum BayesError {
    /// A variable index is not in the model.
    OutOfRange {
        /// The offending index.
        var: usize,
        /// Variables in the model, so valid indices are `0..n`.
        n: usize,
    },
    /// A variable was listed as its own parent.
    SelfParent {
        /// The variable.
        child: usize,
    },
    /// A parent appears twice, which would double-count one coordinate of the table.
    RepeatedParent {
        /// The child whose parent list repeats.
        child: usize,
        /// The parent listed more than once.
        parent: usize,
    },
    /// More parents than [`MAX_PARENTS`], so the table would not fit in memory.
    TooManyParents {
        /// The child.
        child: usize,
        /// Parents given.
        parents: usize,
        /// The cap.
        max: usize,
    },
    /// The table does not have one row per parent configuration.
    TableSize {
        /// The child.
        child: usize,
        /// Rows supplied.
        got: usize,
        /// Rows the parent list requires, which is `2^parents`.
        want: usize,
    },
    /// A table entry is not a probability in `[0, 1]`.
    NotAProbability {
        /// The child.
        child: usize,
        /// The row.
        row: usize,
        /// The value found.
        p: f64,
    },
    /// Two tables were given for one variable.
    AlreadySet {
        /// The variable.
        child: usize,
    },
    /// A variable has no table, so the product is not a distribution.
    Missing {
        /// The variable without a table.
        var: usize,
    },
    /// The parent edges contain a directed cycle, so the product is not a distribution.
    Cycle {
        /// A variable left unordered by the topological sort, hence on or below a cycle.
        var: usize,
    },
}

impl core::fmt::Display for BayesError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            BayesError::OutOfRange { var, n } => {
                write!(f, "variable {var} is out of range for a model of {n} variables")
            }
            BayesError::SelfParent { child } => {
                write!(f, "variable {child} is listed as its own parent")
            }
            BayesError::RepeatedParent { child, parent } => {
                write!(f, "variable {parent} appears twice among the parents of {child}")
            }
            BayesError::TooManyParents { child, parents, max } => write!(
                f,
                "variable {child} has {parents} parents, over the cap of {max}; its table would \
                 hold 2^{parents} rows"
            ),
            BayesError::TableSize { child, got, want } => write!(
                f,
                "variable {child} was given {got} rows and its parents require {want}, one per \
                 parent configuration"
            ),
            BayesError::NotAProbability { child, row, p } => {
                write!(f, "row {row} of variable {child}'s table is {p}, not a probability in [0, 1]")
            }
            BayesError::AlreadySet { child } => {
                write!(f, "variable {child} already has a table")
            }
            BayesError::Missing { var } => write!(
                f,
                "variable {var} has no table; every variable needs one or the product is not a \
                 joint distribution"
            ),
            BayesError::Cycle { var } => write!(
                f,
                "the parent edges cycle through variable {var}; a Bayesian network must be acyclic"
            ),
        }
    }
}

impl core::error::Error for BayesError {}

/// Why a query could not be answered.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum QueryError {
    /// The contraction's largest intermediate exceeds [`Bayes::max_entries`].
    TooWide {
        /// Entries in the largest intermediate the chosen order builds.
        entries: u128,
        /// The cap.
        max: u128,
    },
    /// The evidence has probability zero, so every conditional is `0/0`.
    ImpossibleEvidence,
    /// A queried variable is not in the model.
    OutOfRange {
        /// The offending index.
        var: usize,
        /// Variables in the model.
        n: usize,
    },
    /// The evidence vector is not one entry per variable.
    EvidenceSize {
        /// Entries supplied.
        got: usize,
        /// Variables in the model.
        want: usize,
    },
    /// A conditioned factor still has three or more free variables, so it has no Ising form.
    HigherOrder {
        /// The child whose family is too wide.
        child: usize,
        /// Free variables left in that family after evidence.
        free: usize,
    },
    /// A conditioned factor holds a zero probability, whose log is not a finite Ising weight.
    NotRepresentable {
        /// The child whose table has the zero.
        child: usize,
    },
}

impl core::fmt::Display for QueryError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            QueryError::TooWide { entries, max } => write!(
                f,
                "the contraction's largest intermediate holds {entries} entries, over the cap of \
                 {max}"
            ),
            QueryError::ImpossibleEvidence => {
                write!(f, "this evidence has probability zero, so no conditional is defined")
            }
            QueryError::OutOfRange { var, n } => {
                write!(f, "variable {var} is out of range for a model of {n} variables")
            }
            QueryError::EvidenceSize { got, want } => {
                write!(f, "the evidence holds {got} entries and the model has {want} variables")
            }
            QueryError::HigherOrder { child, free } => write!(
                f,
                "variable {child}'s family still has {free} free variables after evidence; an \
                 Ising graph is pairwise, so this model has no exact one"
            ),
            QueryError::NotRepresentable { child } => write!(
                f,
                "variable {child}'s conditioned table holds a zero, and ln 0 is not a finite Ising \
                 weight; a hard constraint belongs in the evidence"
            ),
        }
    }
}

impl core::error::Error for QueryError {}

/// What is observed. `None` is a free variable, `Some(v)` a variable clamped to `v`.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct Evidence {
    obs: Vec<Option<bool>>,
}

impl Evidence {
    /// Nothing observed, over `n` variables.
    #[must_use]
    pub fn none(n: usize) -> Evidence {
        Evidence { obs: vec![None; n] }
    }

    /// Observe `var = value`, replacing any earlier observation of it.
    ///
    /// # Panics
    ///
    /// If `var` is not below the variable count this was built with.
    pub fn observe(&mut self, var: usize, value: bool) -> &mut Evidence {
        assert!(var < self.obs.len(), "variable {var} is out of range for {}", self.obs.len());
        self.obs[var] = Some(value);
        self
    }

    /// Un-observe `var`.
    ///
    /// # Panics
    ///
    /// If `var` is not below the variable count this was built with.
    pub fn forget(&mut self, var: usize) -> &mut Evidence {
        assert!(var < self.obs.len(), "variable {var} is out of range for {}", self.obs.len());
        self.obs[var] = None;
        self
    }

    /// What `var` is observed to be, or `None` if it is free.
    #[must_use]
    pub fn get(&self, var: usize) -> Option<bool> {
        self.obs.get(var).copied().flatten()
    }

    /// Variables this evidence covers, observed or not.
    #[must_use]
    pub fn vars(&self) -> usize {
        self.obs.len()
    }

    /// How many variables are observed.
    #[must_use]
    pub fn observed(&self) -> usize {
        self.obs.iter().filter(|o| o.is_some()).count()
    }
}

/// One variable's conditional probability table, over its family.
#[derive(Clone, Debug, PartialEq)]
struct Family {
    child: usize,
    parents: Vec<usize>,
    /// `p_true[row]` is `P(child = true | parents)`, with bit `k` of `row` the value of
    /// `parents[k]`.
    p_true: Vec<f64>,
}

impl Family {
    /// The free variables of this family under `e`, and `ln P` over them in tensor layout.
    ///
    /// Tensor layout is row-major with the FIRST index slowest, matching [`crate::tensor::Tensor`].
    /// The child comes first, then the parents in the order they were declared.
    fn conditioned(&self, e: &Evidence) -> (Vec<usize>, Vec<f64>) {
        let mut vars: Vec<usize> = Vec::with_capacity(1 + self.parents.len());
        if e.get(self.child).is_none() {
            vars.push(self.child);
        }
        for &p in &self.parents {
            if e.get(p).is_none() {
                vars.push(p);
            }
        }
        let m = vars.len();
        let mut data = Vec::with_capacity(1 << m);
        for flat in 0..(1usize << m) {
            let value = |v: usize| -> bool {
                match e.get(v) {
                    Some(b) => b,
                    None => {
                        let k = vars
                            .iter()
                            .position(|&x| x == v)
                            .expect("a free family member is in the free list");
                        (flat >> (m - 1 - k)) & 1 == 1
                    }
                }
            };
            let mut row = 0usize;
            for (k, &p) in self.parents.iter().enumerate() {
                if value(p) {
                    row |= 1 << k;
                }
            }
            let pt = self.p_true[row];
            data.push(if value(self.child) { pt.ln() } else { (1.0 - pt).ln() });
        }
        (vars, data)
    }
}

/// Builds a [`Bayes`] one conditional probability table at a time.
#[derive(Clone, Debug, Default)]
pub struct BayesBuilder {
    n: usize,
    cpts: Vec<Option<Family>>,
}

impl BayesBuilder {
    /// A builder over `n` binary variables, none of which has a table yet.
    #[must_use]
    pub fn new(n: usize) -> BayesBuilder {
        BayesBuilder { n, cpts: vec![None; n] }
    }

    /// Give `child` a table: `p_true[row]` is `P(child = true | parents)`, bit `k` of `row` being
    /// the value of `parents[k]`.
    ///
    /// # Errors
    ///
    /// [`BayesError`] for an index out of range, a repeated or self parent, a table of the wrong
    /// length, an entry outside `[0, 1]`, or a second table for the same variable.
    pub fn cpt(
        &mut self,
        child: usize,
        parents: &[usize],
        p_true: &[f64],
    ) -> Result<(), BayesError> {
        if child >= self.n {
            return Err(BayesError::OutOfRange { var: child, n: self.n });
        }
        if self.cpts[child].is_some() {
            return Err(BayesError::AlreadySet { child });
        }
        if parents.len() > MAX_PARENTS {
            return Err(BayesError::TooManyParents {
                child,
                parents: parents.len(),
                max: MAX_PARENTS,
            });
        }
        for (k, &p) in parents.iter().enumerate() {
            if p >= self.n {
                return Err(BayesError::OutOfRange { var: p, n: self.n });
            }
            if p == child {
                return Err(BayesError::SelfParent { child });
            }
            if parents[..k].contains(&p) {
                return Err(BayesError::RepeatedParent { child, parent: p });
            }
        }
        let want = 1usize << parents.len();
        if p_true.len() != want {
            return Err(BayesError::TableSize { child, got: p_true.len(), want });
        }
        for (row, &p) in p_true.iter().enumerate() {
            // `contains` is false for NaN, so the negation rejects it -- which is the point.
            if !(0.0..=1.0).contains(&p) {
                return Err(BayesError::NotAProbability { child, row, p });
            }
        }
        self.cpts[child] =
            Some(Family { child, parents: parents.to_vec(), p_true: p_true.to_vec() });
        Ok(())
    }

    /// Give a parentless `var` the prior `P(var = true) = p_true`.
    ///
    /// # Errors
    ///
    /// As [`BayesBuilder::cpt`].
    pub fn root(&mut self, var: usize, p_true: f64) -> Result<(), BayesError> {
        self.cpt(var, &[], &[p_true])
    }

    /// Finish, checking that every variable has a table and the parent edges are acyclic.
    ///
    /// # Errors
    ///
    /// [`BayesError::Missing`] for a variable without a table, [`BayesError::Cycle`] when the
    /// parent edges contain a directed cycle.
    pub fn build(self) -> Result<Bayes, BayesError> {
        let mut fam = Vec::with_capacity(self.n);
        for (v, c) in self.cpts.into_iter().enumerate() {
            match c {
                Some(f) => fam.push(f),
                None => return Err(BayesError::Missing { var: v }),
            }
        }
        // Kahn's algorithm on parent -> child. A cycle leaves its own members, and everything
        // downstream of them, unordered; the lowest such variable is the one reported.
        let mut indeg: Vec<usize> = fam.iter().map(|f| f.parents.len()).collect();
        let mut children: Vec<Vec<usize>> = vec![Vec::new(); self.n];
        for f in &fam {
            for &p in &f.parents {
                children[p].push(f.child);
            }
        }
        let mut queue: Vec<usize> = (0..self.n).filter(|&v| indeg[v] == 0).collect();
        while let Some(v) = queue.pop() {
            for &c in &children[v] {
                indeg[c] -= 1;
                if indeg[c] == 0 {
                    queue.push(c);
                }
            }
        }
        // A variable the sort never reached still has an unmet parent, and everything that cycles
        // is such a variable. The lowest one is reported.
        if let Some(var) = (0..self.n).find(|&v| indeg[v] > 0) {
            return Err(BayesError::Cycle { var });
        }
        Ok(Bayes { n: self.n, fam, max_entries: 1 << 26 })
    }
}

/// A Bayesian network over binary variables, answered exactly.
#[derive(Clone, Debug, PartialEq)]
pub struct Bayes {
    n: usize,
    fam: Vec<Family>,
    /// Refuse a contraction whose largest intermediate exceeds this many entries. `2^26` by
    /// default, matching [`crate::tensor::Network::contract`].
    pub max_entries: u128,
}

/// The most probable explanation: the single likeliest completion of the evidence.
#[derive(Clone, Debug, PartialEq)]
pub struct Mpe {
    /// One value per variable, agreeing with the evidence on everything observed.
    pub state: Vec<bool>,
    /// `ln P(state)` under the joint, evidence included.
    pub log_prob: f64,
}

/// A conditioned network as an Ising graph: `P(x | e)` is `exp(-E(s))` normalised, at `beta = 1`.
pub struct IsingForm {
    /// One spin per free variable, `+1` for true.
    pub graph: Graph,
    /// Spin `i` is variable `vars[i]`.
    pub vars: Vec<usize>,
    /// The constant the multilinear expansion left behind: `ln P(e)` is this plus `ln Z(beta = 1)`.
    pub log_offset: f64,
}

impl Bayes {
    /// How many variables it has.
    #[must_use]
    pub fn n(&self) -> usize {
        self.n
    }

    /// The parents of `var`, in the order they were declared.
    ///
    /// # Panics
    ///
    /// If `var` is not below [`Bayes::n`].
    #[must_use]
    pub fn parents(&self, var: usize) -> &[usize] {
        &self.fam[var].parents
    }

    /// `ln P(x)` for a complete assignment: the sum of one table entry per variable.
    ///
    /// # Errors
    ///
    /// [`QueryError::EvidenceSize`] when `state` is not one value per variable.
    pub fn log_joint(&self, state: &[bool]) -> Result<f64, QueryError> {
        if state.len() != self.n {
            return Err(QueryError::EvidenceSize { got: state.len(), want: self.n });
        }
        let mut acc = 0.0;
        for f in &self.fam {
            let mut row = 0usize;
            for (k, &p) in f.parents.iter().enumerate() {
                if state[p] {
                    row |= 1 << k;
                }
            }
            let pt = f.p_true[row];
            acc += if state[f.child] { pt.ln() } else { (1.0 - pt).ln() };
        }
        Ok(acc)
    }

    /// `ln P(e)`: the log likelihood of the evidence, marginalising every free variable.
    ///
    /// # Errors
    ///
    /// [`QueryError::EvidenceSize`], [`QueryError::TooWide`] when the contraction does not fit
    /// [`Bayes::max_entries`], and [`QueryError::ImpossibleEvidence`] when `P(e)` is zero.
    ///
    /// # Panics
    ///
    /// Never: every tensor built here carries each free variable once at dimension two, which is
    /// the only thing the contraction rejects as malformed.
    pub fn log_evidence(&self, e: &Evidence) -> Result<f64, QueryError> {
        self.check_evidence(e)?;
        let net: Network<LogSumExp> = self.net(e, |l| l);
        let t = contract(&net, self.max_entries)?;
        let v = t.value().expect("a contraction with no open index is rank zero");
        if v == f64::NEG_INFINITY {
            return Err(QueryError::ImpossibleEvidence);
        }
        Ok(v)
    }

    /// `P(var = true | e)`.
    ///
    /// An observed variable answers `1.0` or `0.0`; the evidence is still checked, because a
    /// conditional on impossible evidence is not zero, it is undefined.
    ///
    /// # Errors
    ///
    /// [`QueryError::OutOfRange`], and the errors of [`Bayes::log_evidence`].
    ///
    /// # Panics
    ///
    /// Never, for the reason given on [`Bayes::log_evidence`].
    pub fn marginal(&self, var: usize, e: &Evidence) -> Result<f64, QueryError> {
        self.check_evidence(e)?;
        if var >= self.n {
            return Err(QueryError::OutOfRange { var, n: self.n });
        }
        if let Some(b) = e.get(var) {
            self.log_evidence(e)?;
            return Ok(if b { 1.0 } else { 0.0 });
        }
        Ok(self.free_marginal(var, e)?.0)
    }

    /// `P(var = true | e)` for every variable at once, in variable order.
    ///
    /// Costs one contraction per free variable, plus one for the evidence itself.
    ///
    /// # Errors
    ///
    /// As [`Bayes::marginal`].
    ///
    /// # Panics
    ///
    /// Never, for the reason given on [`Bayes::log_evidence`].
    pub fn marginals(&self, e: &Evidence) -> Result<Vec<f64>, QueryError> {
        self.check_evidence(e)?;
        self.log_evidence(e)?;
        let mut out = Vec::with_capacity(self.n);
        for v in 0..self.n {
            match e.get(v) {
                Some(b) => out.push(if b { 1.0 } else { 0.0 }),
                None => out.push(self.free_marginal(v, e)?.0),
            }
        }
        Ok(out)
    }

    /// The most probable explanation: the likeliest complete assignment agreeing with `e`.
    ///
    /// Max-product, over [`crate::tensor::Tropical`] on the negated log tables — so the energy it
    /// minimises is `-ln P`. One variable is decoded per contraction, each against the max-marginal
    /// of the variables not yet fixed, which is exact rather than greedy. Ties go to `false`.
    ///
    /// # Errors
    ///
    /// As [`Bayes::log_evidence`].
    ///
    /// # Panics
    ///
    /// Never, for the reason given on [`Bayes::log_evidence`].
    pub fn mpe(&self, e: &Evidence) -> Result<Mpe, QueryError> {
        self.check_evidence(e)?;
        let mut fixed = e.clone();
        for v in 0..self.n {
            if fixed.get(v).is_some() {
                continue;
            }
            let mut net: Network<Tropical> = self.net(&fixed, |l| -l);
            net.open(v as Index);
            let t = contract(&net, self.max_entries)?;
            let d = t.data();
            if d[0] == f64::INFINITY && d[1] == f64::INFINITY {
                return Err(QueryError::ImpossibleEvidence);
            }
            fixed.observe(v, d[1] < d[0]);
        }
        let state: Vec<bool> =
            (0..self.n).map(|v| fixed.get(v).expect("every variable is now fixed")).collect();
        let log_prob = self.log_joint(&state)?;
        if log_prob == f64::NEG_INFINITY {
            return Err(QueryError::ImpossibleEvidence);
        }
        Ok(Mpe { state, log_prob })
    }

    /// The conditioned network as an Ising graph, when every family fits in one.
    ///
    /// Evidence is what makes this possible: a child with two parents is a rank-three factor, but
    /// OBSERVING that child leaves a rank-two factor over the parents — the moralisation edge,
    /// as a coupling. `P(x | e)` is then the Boltzmann distribution of [`IsingForm::graph`] at
    /// `beta = 1`, so [`crate::exact::Elimination`] and every sampler in the crate apply.
    ///
    /// # Errors
    ///
    /// [`QueryError::HigherOrder`] when a conditioned family still has three or more free
    /// variables, [`QueryError::NotRepresentable`] when one holds a zero probability, plus
    /// [`QueryError::EvidenceSize`].
    pub fn to_ising(&self, e: &Evidence) -> Result<IsingForm, QueryError> {
        self.check_evidence(e)?;
        let mut vars = Vec::new();
        let mut slot = vec![usize::MAX; self.n];
        for v in 0..self.n {
            if e.get(v).is_none() {
                slot[v] = vars.len();
                vars.push(v);
            }
        }
        let mut b = GraphBuilder::new(vars.len());
        let mut log_offset = 0.0;
        for f in &self.fam {
            let (free, g) = f.conditioned(e);
            if g.iter().any(|v| !v.is_finite()) {
                return Err(QueryError::NotRepresentable { child: f.child });
            }
            // The multilinear expansion of a function of one or two spins. It is exact: a function
            // on `{-1,+1}^k` has 2^k coefficients and the table has 2^k entries.
            match free.len() {
                0 => log_offset += g[0],
                1 => {
                    log_offset += 0.5 * (g[0] + g[1]);
                    b.bias(slot[free[0]], 0.5 * (g[1] - g[0]));
                }
                2 => {
                    log_offset += 0.25 * (g[0] + g[1] + g[2] + g[3]);
                    b.bias(slot[free[0]], 0.25 * (-g[0] - g[1] + g[2] + g[3]));
                    b.bias(slot[free[1]], 0.25 * (-g[0] + g[1] - g[2] + g[3]));
                    b.couple(
                        slot[free[0]],
                        slot[free[1]],
                        0.25 * (g[0] - g[1] - g[2] + g[3]),
                    );
                }
                free_n => return Err(QueryError::HigherOrder { child: f.child, free: free_n }),
            }
        }
        Ok(IsingForm { graph: b.build(), vars, log_offset })
    }

    /// `(P(var = true | e), ln P(e))` for a variable known to be free.
    fn free_marginal(&self, var: usize, e: &Evidence) -> Result<(f64, f64), QueryError> {
        let mut net: Network<LogSumExp> = self.net(e, |l| l);
        net.open(var as Index);
        let t = contract(&net, self.max_entries)?;
        let d = t.data();
        let (lf, lt) = (d[0], d[1]);
        let m = if lf > lt { lf } else { lt };
        if m == f64::NEG_INFINITY {
            return Err(QueryError::ImpossibleEvidence);
        }
        let (pf, pt) = ((lf - m).exp(), (lt - m).exp());
        Ok((pt / (pf + pt), m + (pf + pt).ln()))
    }

    /// One tensor per family, its entries `map`ped from `ln P`.
    fn net<S, F>(&self, e: &Evidence, map: F) -> Network<S>
    where
        S: Semiring,
        F: Fn(f64) -> S::Elem,
    {
        let mut net = Network::new();
        for f in &self.fam {
            let (free, g) = f.conditioned(e);
            if free.is_empty() {
                net.push(Tensor::scalar(map(g[0])));
            } else {
                let idx: Vec<Index> = free.iter().map(|&v| v as Index).collect();
                let dims = vec![2usize; free.len()];
                let data: Vec<S::Elem> = g.into_iter().map(&map).collect();
                net.push(
                    Tensor::new(idx, dims, data)
                        .expect("one index per free variable, each of dimension two"),
                );
            }
        }
        net
    }

    fn check_evidence(&self, e: &Evidence) -> Result<(), QueryError> {
        if e.vars() == self.n {
            Ok(())
        } else {
            Err(QueryError::EvidenceSize { got: e.vars(), want: self.n })
        }
    }
}

/// Contract, translating the tensor engine's refusal into this module's.
fn contract<S: Semiring>(
    net: &Network<S>,
    max_entries: u128,
) -> Result<Tensor<S>, QueryError> {
    match net.contract_with(Order::GreedySize, max_entries) {
        Ok(t) => Ok(t),
        Err(Uncontractable::TooWide { entries, max }) => Err(QueryError::TooWide { entries, max }),
        Err(Uncontractable::Malformed(m)) => {
            unreachable!("every tensor here has one index per free variable at dimension two: {m}")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exact::Elimination;

    /// A network written down twice: once as tables this module is fed, once as a brute-force
    /// joint the tests score it against.
    ///
    /// The oracle multiplies out `2^n` assignments from the SAME tables and never calls anything in
    /// the module under test, so a defect in the conversion, the elimination or the log arithmetic
    /// has nowhere to hide. Only the documented row convention is shared.
    struct Ref {
        n: usize,
        fams: Vec<(usize, Vec<usize>, Vec<f64>)>,
    }

    impl Ref {
        fn bayes(&self) -> Bayes {
            let mut b = BayesBuilder::new(self.n);
            for (c, p, t) in &self.fams {
                b.cpt(*c, p, t).expect("a well-formed table");
            }
            b.build().expect("a well-formed network")
        }

        fn joint(&self, x: &[bool]) -> f64 {
            let mut p = 1.0;
            for (c, pa, t) in &self.fams {
                let mut row = 0usize;
                for (k, &v) in pa.iter().enumerate() {
                    if x[v] {
                        row |= 1 << k;
                    }
                }
                p *= if x[*c] { t[row] } else { 1.0 - t[row] };
            }
            p
        }

        fn states(&self) -> Vec<Vec<bool>> {
            (0..(1usize << self.n))
                .map(|m| (0..self.n).map(|b| m >> b & 1 == 1).collect())
                .collect()
        }

        /// `(P(e), P(v = true, e) for each v, the likeliest consistent state and its probability)`.
        fn brute(&self, e: &Evidence) -> (f64, Vec<f64>, f64, Vec<bool>) {
            let mut pe = 0.0;
            let mut num = vec![0.0; self.n];
            let mut best = -1.0;
            let mut arg = vec![false; self.n];
            for x in self.states() {
                if (0..self.n).any(|v| e.get(v).is_some_and(|b| b != x[v])) {
                    continue;
                }
                let p = self.joint(&x);
                pe += p;
                for v in 0..self.n {
                    if x[v] {
                        num[v] += p;
                    }
                }
                if p > best {
                    best = p;
                    arg = x;
                }
            }
            (pe, num, best, arg)
        }
    }

    /// The alarm network of Russell and Norvig, *Artificial Intelligence: A Modern Approach*,
    /// figure 14.2. Variables: 0 burglary, 1 earthquake, 2 alarm, 3 `JohnCalls`, 4 `MaryCalls`.
    fn alarm() -> Ref {
        Ref {
            n: 5,
            fams: vec![
                (0, vec![], vec![0.001]),
                (1, vec![], vec![0.002]),
                // parents [burglary, earthquake]: row bit 0 is burglary, bit 1 is earthquake
                (2, vec![0, 1], vec![0.001, 0.94, 0.29, 0.95]),
                (3, vec![2], vec![0.05, 0.90]),
                (4, vec![2], vec![0.01, 0.70]),
            ],
        }
    }

    /// A three-layer network with two multi-parent families, for the general checks.
    fn tangle() -> Ref {
        Ref {
            n: 6,
            fams: vec![
                (0, vec![], vec![0.3]),
                (1, vec![], vec![0.65]),
                (2, vec![0, 1], vec![0.1, 0.8, 0.45, 0.99]),
                (3, vec![0], vec![0.2, 0.7]),
                (4, vec![2, 3], vec![0.05, 0.5, 0.9, 0.15]),
                (5, vec![1, 4], vec![0.75, 0.25, 0.4, 0.6]),
            ],
        }
    }

    /// A network and the evidence patterns to score it under.
    type Cases = Vec<(Ref, Vec<Vec<(usize, bool)>>)>;

    fn evidence(n: usize, obs: &[(usize, bool)]) -> Evidence {
        let mut e = Evidence::none(n);
        for &(v, b) in obs {
            e.observe(v, b);
        }
        e
    }

    /// `logaddexp` and `+` really are a semiring, including at the identities.
    ///
    /// The oracle is the algebra itself: distributivity is what licenses elimination, and a
    /// structure missing it contracts to a number that is simply wrong. `-inf` is checked
    /// explicitly because the shift-by-the-larger trick produces `NaN` there if the identity is
    /// not taken first.
    #[test]
    fn log_sum_exp_obeys_the_semiring_laws() {
        let vals = [-30.0, -3.0, -0.5, 0.0, 2.0, f64::NEG_INFINITY];
        for &a in &vals {
            assert_eq!(LogSumExp::add(a, LogSumExp::zero()), a, "zero is the additive identity");
            assert_eq!(LogSumExp::mul(a, LogSumExp::one()), a, "one is the multiplicative identity");
            assert_eq!(
                LogSumExp::mul(a, LogSumExp::zero()),
                LogSumExp::zero(),
                "zero annihilates under multiplication"
            );
            for &b in &vals {
                assert_eq!(LogSumExp::add(a, b), LogSumExp::add(b, a));
                for &c in &vals {
                    let lhs = LogSumExp::mul(c, LogSumExp::add(a, b));
                    let rhs = LogSumExp::add(LogSumExp::mul(c, a), LogSumExp::mul(c, b));
                    assert!(
                        (lhs - rhs).abs() < 1e-12 || (lhs == rhs),
                        "distributivity at ({a}, {b}, {c}): {lhs} vs {rhs}"
                    );
                    let l2 = LogSumExp::add(LogSumExp::add(a, b), c);
                    let r2 = LogSumExp::add(a, LogSumExp::add(b, c));
                    assert!((l2 - r2).abs() < 1e-12 || (l2 == r2), "associativity at ({a},{b},{c})");
                }
            }
        }
        // and it is the arithmetic it claims to be
        assert_eq!(LogSumExp::add(0.0_f64.ln(), 0.0_f64.ln()), f64::NEG_INFINITY);
        assert!((LogSumExp::add(0.25_f64.ln(), 0.5_f64.ln()) - 0.75_f64.ln()).abs() < 1e-15);
    }

    /// The joint sums to one over every assignment, which is the definition of a Bayes net.
    #[test]
    fn the_joint_is_normalised() {
        for r in [alarm(), tangle()] {
            let b = r.bayes();
            let mut total = 0.0;
            for x in r.states() {
                let p = b.log_joint(&x).expect("a complete assignment").exp();
                assert!((p - r.joint(&x)).abs() < 1e-13, "log_joint disagrees with the product");
                total += p;
            }
            assert!((total - 1.0).abs() < 1e-12, "the joint sums to {total}");
        }
    }

    /// Evidence likelihood and every marginal, against the brute-force joint.
    ///
    /// Twenty-eight evidence patterns across two networks, including the empty one and patterns
    /// that observe a variable in the middle of a family. `P(e)` is compared in probability space
    /// and the marginals as conditionals, both against a sum over all `2^n` assignments.
    #[test]
    fn marginals_and_evidence_match_brute_force() {
        let cases: Cases = vec![
            (
                alarm(),
                vec![
                    vec![],
                    vec![(3, true), (4, true)],
                    vec![(2, true)],
                    vec![(2, true), (1, true)],
                    vec![(0, false), (4, true)],
                    vec![(0, true), (1, true), (2, true), (3, true), (4, true)],
                ],
            ),
            (
                tangle(),
                vec![
                    vec![],
                    vec![(5, true)],
                    vec![(4, false)],
                    vec![(2, true), (5, false)],
                    vec![(0, true), (3, false)],
                    vec![(1, true), (2, false), (4, true)],
                ],
            ),
        ];
        for (r, patterns) in cases {
            let b = r.bayes();
            for obs in patterns {
                let e = evidence(r.n, &obs);
                let (pe, num, _, _) = r.brute(&e);
                let le = b.log_evidence(&e).expect("this evidence is possible");
                assert!(
                    (le.exp() - pe).abs() < 1e-12 * pe.max(1e-12),
                    "P(e) for {obs:?}: got {}, brute force {pe}",
                    le.exp()
                );
                let got = b.marginals(&e).expect("this evidence is possible");
                for v in 0..r.n {
                    let want = num[v] / pe;
                    assert!(
                        (got[v] - want).abs() < 1e-12,
                        "P(x{v} | {obs:?}): got {}, brute force {want}",
                        got[v]
                    );
                    assert!(
                        (b.marginal(v, &e).expect("possible") - want).abs() < 1e-12,
                        "marginal() and marginals() must agree"
                    );
                }
            }
        }
    }

    /// The published posterior of the alarm network.
    ///
    /// `P(Burglary | JohnCalls = true, MaryCalls = true) = 0.284`, Russell and Norvig, section
    /// 14.4. An external number, so the whole pipeline — table convention, moralisation, evidence,
    /// elimination — is being checked against something this repository did not compute.
    #[test]
    fn the_alarm_network_reproduces_its_textbook_posterior() {
        let r = alarm();
        let b = r.bayes();
        let e = evidence(5, &[(3, true), (4, true)]);
        let p = b.marginal(0, &e).expect("possible evidence");
        assert!((p - 0.284_171_835_364_392_94).abs() < 1e-12, "got {p}, textbook 0.284");
        // and the evidence likelihood behind it
        let pe = b.log_evidence(&e).expect("possible evidence").exp();
        assert!((pe - 0.002_084_100_239).abs() < 1e-12, "P(j, m) = {pe}");
    }

    /// Explaining away: two independent causes become dependent once their common child is seen.
    ///
    /// The property a conversion that drops the moralisation edge loses, and the reason this test
    /// exists. Three claims, in order:
    ///
    /// 1. Without the alarm observed, earthquake tells you NOTHING about burglary — equal to
    ///    fourteen digits, not merely close.
    /// 2. With the alarm observed, burglary jumps by three orders of magnitude.
    /// 3. Learning the earthquake as well pushes it back down by an order of magnitude. That third
    ///    move is the induced dependence, and it is the one that vanishes without the moral edge.
    #[test]
    fn conditioning_on_a_common_child_makes_independent_parents_dependent() {
        let r = alarm();
        let b = r.bayes();
        let prior = b.marginal(0, &evidence(5, &[])).expect("possible");
        let given_quake = b.marginal(0, &evidence(5, &[(1, true)])).expect("possible");
        let given_alarm = b.marginal(0, &evidence(5, &[(2, true)])).expect("possible");
        let given_both = b.marginal(0, &evidence(5, &[(2, true), (1, true)])).expect("possible");

        assert!((prior - 0.001).abs() < 1e-15, "the prior is the root table");
        assert!(
            (given_quake - prior).abs() < 1e-14,
            "marginally independent: P(b|e) = {given_quake}, P(b) = {prior}"
        );
        assert!(given_alarm > 100.0 * prior, "the alarm implicates the burglar: {given_alarm}");
        assert!(
            given_both < given_alarm / 8.0,
            "the earthquake explains the alarm away: P(b|a,e) = {given_both} vs P(b|a) = \
             {given_alarm}"
        );
        // every one of them against the enumerated joint
        for obs in [vec![], vec![(1, true)], vec![(2, true)], vec![(2, true), (1, true)]] {
            let e = evidence(5, &obs);
            let (pe, num, _, _) = r.brute(&e);
            let got = b.marginal(0, &e).expect("possible");
            assert!((got - num[0] / pe).abs() < 1e-13, "{obs:?}");
        }
    }

    /// The moralisation edge is a coupling, and deleting it changes the answer.
    ///
    /// Stated as a mutation rather than as a claim about the code: take the Ising form of the
    /// alarm network with the alarm observed, delete the burglary-earthquake coupling — exactly
    /// what a conversion that forgot to moralise would have produced — and read the marginal off
    /// both graphs with the crate's own exact solver. If the edge did not matter, this test could
    /// not fail.
    #[test]
    fn deleting_the_moral_edge_changes_the_marginal() {
        let r = alarm();
        let b = r.bayes();
        let e = evidence(5, &[(2, true)]);
        let f = b.to_ising(&e).expect("observing the alarm leaves every family pairwise");
        let (bi, ei) = (
            f.vars.iter().position(|&v| v == 0).expect("burglary is free"),
            f.vars.iter().position(|&v| v == 1).expect("earthquake is free"),
        );
        let w = coupling(&f.graph, bi, ei).expect("the moral edge exists");
        assert!(w.abs() > 1.0, "the moral coupling is {w}, not a rounding artefact");

        let elim = Elimination::default();
        let with = elim.marginals(&f.graph, 1.0).expect("tiny graph")[bi];
        let moral = without_edge(&f.graph, bi, ei);
        let without = elim.marginals(&moral, 1.0).expect("tiny graph")[bi];
        let truth = b.marginal(0, &e).expect("possible");
        assert!((with - truth).abs() < 1e-12, "the Ising form must reproduce the answer: {with}");
        assert!(
            (without - truth).abs() > 0.05,
            "without the moral edge the marginal is {without}, and the truth is {truth}"
        );
    }

    fn coupling(g: &Graph, i: usize, j: usize) -> Option<f64> {
        (g.offset[i]..g.offset[i + 1]).find(|&k| g.nbr[k] as usize == j).map(|k| g.w[k])
    }

    /// `g` with the edge `(i, j)` deleted, everything else identical.
    fn without_edge(g: &Graph, i: usize, j: usize) -> Graph {
        let mut b = GraphBuilder::new(g.n);
        for v in 0..g.n {
            b.bias(v, g.h[v]);
            for k in g.offset[v]..g.offset[v + 1] {
                let u = g.nbr[k] as usize;
                if u > v && !((v == i && u == j) || (v == j && u == i)) {
                    b.couple(v, u, g.w[k]);
                }
            }
        }
        b.build()
    }

    /// The Ising form is the same distribution, checked by two exact routes and by enumeration.
    ///
    /// `ln P(e) = log_offset + ln Z(beta = 1)` is an algebraic identity between this module's
    /// log-space contraction and `exact::Elimination`'s partition function — two different
    /// engines, one number — and the marginals are checked against the enumerated joint as well,
    /// so agreement between the two engines cannot be agreement on a shared mistake.
    #[test]
    fn the_ising_form_carries_the_same_distribution() {
        let cases: Vec<(Ref, Vec<(usize, bool)>)> = vec![
            (alarm(), vec![(2, true)]),
            (alarm(), vec![(2, false), (3, true)]),
            (tangle(), vec![(2, true), (4, false), (5, true)]),
            (chain(6, 0.7, 0.8), vec![]),
            (chain(6, 0.7, 0.8), vec![(5, true)]),
        ];
        for (r, obs) in cases {
            let b = r.bayes();
            let e = evidence(r.n, &obs);
            let f = b.to_ising(&e).expect("every conditioned family is pairwise here");
            let elim = Elimination::default();
            let lz = elim.log_partition(&f.graph, 1.0).expect("tiny graph").log_z.expect("ran");
            let le = b.log_evidence(&e).expect("possible");
            assert!(
                (f.log_offset + lz - le).abs() < 1e-11,
                "{obs:?}: offset {} + lnZ {lz} = {}, and ln P(e) = {le}",
                f.log_offset,
                f.log_offset + lz
            );
            let (pe, num, _, _) = r.brute(&e);
            assert!((le.exp() - pe).abs() < 1e-14, "and both against the enumerated joint");
            let m = elim.marginals(&f.graph, 1.0).expect("tiny graph");
            for (i, &v) in f.vars.iter().enumerate() {
                assert!(
                    (m[i] - num[v] / pe).abs() < 1e-11,
                    "spin {i} is variable {v}: Ising says {}, enumeration {}",
                    m[i],
                    num[v] / pe
                );
            }
        }
    }

    /// A family that evidence did not narrow has no Ising form, and says so.
    #[test]
    fn a_three_way_family_is_refused_rather_than_approximated() {
        let b = alarm().bayes();
        assert!(matches!(
            b.to_ising(&evidence(5, &[])),
            Err(QueryError::HigherOrder { child: 2, free: 3 })
        ));
        // observing either parent narrows it, and then it fits
        assert!(b.to_ising(&evidence(5, &[(1, false)])).is_ok());
        // a zero in a conditioned table has no finite weight
        let mut bb = BayesBuilder::new(2);
        bb.root(0, 0.5).expect("valid");
        bb.cpt(1, &[0], &[0.0, 1.0]).expect("valid");
        let det = bb.build().expect("valid");
        assert!(matches!(
            det.to_ising(&Evidence::none(2)),
            Err(QueryError::NotRepresentable { child: 1 })
        ));
    }

    /// The most probable explanation, against the largest term of the enumerated joint.
    #[test]
    fn mpe_matches_the_enumerated_maximum() {
        let cases: Cases = vec![
            (alarm(), vec![vec![], vec![(3, true), (4, true)], vec![(0, true)], vec![(2, true)]]),
            (tangle(), vec![vec![], vec![(5, true)], vec![(0, true), (4, true)], vec![(2, false)]]),
        ];
        for (r, patterns) in cases {
            let b = r.bayes();
            for obs in patterns {
                let e = evidence(r.n, &obs);
                let (_, _, best, _) = r.brute(&e);
                let m = b.mpe(&e).expect("possible");
                for &(v, val) in &obs {
                    assert_eq!(m.state[v], val, "the explanation must agree with the evidence");
                }
                assert!(
                    (r.joint(&m.state) - best).abs() < 1e-15,
                    "{obs:?}: the state returned has probability {}, and the maximum is {best}",
                    r.joint(&m.state)
                );
                assert!((m.log_prob.exp() - best).abs() < 1e-15, "and log_prob agrees");
            }
        }
    }

    /// A hard constraint is respected: zero-probability rows are never explained.
    #[test]
    fn mpe_never_returns_an_impossible_state() {
        // x2 is the AND of x0 and x1, exactly. Observing it true forces both parents.
        let r = Ref {
            n: 3,
            fams: vec![
                (0, vec![], vec![0.4]),
                (1, vec![], vec![0.3]),
                (2, vec![0, 1], vec![0.0, 0.0, 0.0, 1.0]),
            ],
        };
        let b = r.bayes();
        let m = b.mpe(&evidence(3, &[(2, true)])).expect("possible");
        assert_eq!(m.state, vec![true, true, true]);
        assert!((m.log_prob.exp() - 0.4 * 0.3).abs() < 1e-15);
        // and the impossible evidence is refused rather than answered
        assert_eq!(b.mpe(&evidence(3, &[(0, false), (2, true)])), Err(QueryError::ImpossibleEvidence));
        assert_eq!(
            b.log_evidence(&evidence(3, &[(0, false), (2, true)])),
            Err(QueryError::ImpossibleEvidence)
        );
        assert_eq!(
            b.marginal(1, &evidence(3, &[(0, false), (2, true)])),
            Err(QueryError::ImpossibleEvidence)
        );
    }

    /// A chain of `n` variables, `p_flip` to start true and `p_stay` to copy the parent.
    fn chain(n: usize, p_flip: f64, p_stay: f64) -> Ref {
        let mut fams = vec![(0usize, Vec::new(), vec![p_flip])];
        for v in 1..n {
            fams.push((v, vec![v - 1], vec![1.0 - p_stay, p_stay]));
        }
        Ref { n, fams }
    }

    /// A likelihood of `1e-378`: exactly representable as a log, exactly zero as a float.
    ///
    /// The reason the tensors carry `ln P` over [`LogSumExp`] rather than probabilities over
    /// `SumProduct`. The oracle is closed form — every variable is observed, so `P(e)` is one
    /// product — and the test asserts BOTH that the log-space answer is right to eleven digits and
    /// that the same product computed in probability space is literally `0.0`, which is what every
    /// conditional would then have been divided by.
    #[test]
    fn a_likelihood_far_below_the_smallest_float_is_still_exact() {
        let n = 64;
        let (p_flip, p_stay) = (1e-6, 1e-6);
        let r = chain(n, p_flip, p_stay);
        let b = r.bayes();
        let e = {
            let mut e = Evidence::none(n);
            for v in 0..n {
                e.observe(v, true);
            }
            e
        };
        let want = p_flip.ln() + (n - 1) as f64 * p_stay.ln();
        let got = b.log_evidence(&e).expect("a log never underflows");
        assert!((got - want).abs() < 1e-9, "ln P(e) = {got}, closed form {want}");
        assert!(want < -300.0 * core::f64::consts::LN_10, "this is below 1e-300");

        let mut linear = p_flip;
        for _ in 1..n {
            linear *= p_stay;
        }
        assert_eq!(linear, 0.0, "in probability space the same product is zero");
        assert_eq!(want.exp(), 0.0, "and so is its exponential");

        // the marginals of a chain with a free tail, against the enumerated joint
        let short = chain(10, 0.7, 0.8);
        let sb = short.bayes();
        let ev = evidence(10, &[(0, true), (9, false)]);
        let (pe, num, _, _) = short.brute(&ev);
        let got = sb.marginals(&ev).expect("possible");
        for v in 0..10 {
            assert!((got[v] - num[v] / pe).abs() < 1e-12, "chain marginal {v}");
        }
    }

    /// Every way of writing down a table that would not mean what it says.
    #[test]
    fn malformed_networks_are_refused() {
        let mut b = BayesBuilder::new(3);
        assert_eq!(b.cpt(3, &[], &[0.5]), Err(BayesError::OutOfRange { var: 3, n: 3 }));
        assert_eq!(b.cpt(0, &[7], &[0.5, 0.5]), Err(BayesError::OutOfRange { var: 7, n: 3 }));
        assert_eq!(b.cpt(0, &[0], &[0.5, 0.5]), Err(BayesError::SelfParent { child: 0 }));
        assert_eq!(
            b.cpt(0, &[1, 1], &[0.5; 4]),
            Err(BayesError::RepeatedParent { child: 0, parent: 1 })
        );
        assert_eq!(
            b.cpt(0, &[1, 2], &[0.5; 3]),
            Err(BayesError::TableSize { child: 0, got: 3, want: 4 })
        );
        assert_eq!(
            b.cpt(0, &[], &[1.5]),
            Err(BayesError::NotAProbability { child: 0, row: 0, p: 1.5 })
        );
        assert!(matches!(b.cpt(0, &[], &[f64::NAN]), Err(BayesError::NotAProbability { .. })));
        b.root(0, 0.5).expect("valid");
        assert_eq!(b.root(0, 0.25), Err(BayesError::AlreadySet { child: 0 }));
        assert_eq!(b.clone().build(), Err(BayesError::Missing { var: 1 }));

        let mut wide = BayesBuilder::new(64);
        let parents: Vec<usize> = (1..=21).collect();
        assert_eq!(
            wide.cpt(0, &parents, &[0.5]),
            Err(BayesError::TooManyParents { child: 0, parents: 21, max: MAX_PARENTS })
        );
    }

    /// A cycle is not a Bayesian network, and the product of its tables is not a distribution.
    #[test]
    fn a_cyclic_model_is_refused() {
        let mut b = BayesBuilder::new(3);
        b.cpt(0, &[2], &[0.2, 0.8]).expect("valid");
        b.cpt(1, &[0], &[0.2, 0.8]).expect("valid");
        b.cpt(2, &[1], &[0.2, 0.8]).expect("valid");
        assert_eq!(b.build(), Err(BayesError::Cycle { var: 0 }));

        // a two-cycle, which no topological pass can start
        let mut c = BayesBuilder::new(2);
        c.cpt(0, &[1], &[0.2, 0.8]).expect("valid");
        c.cpt(1, &[0], &[0.2, 0.8]).expect("valid");
        assert!(matches!(c.build(), Err(BayesError::Cycle { .. })));
    }

    /// Queries check their arguments rather than indexing past the end.
    #[test]
    fn queries_check_their_arguments() {
        let b = alarm().bayes();
        assert_eq!(
            b.marginal(0, &Evidence::none(4)),
            Err(QueryError::EvidenceSize { got: 4, want: 5 })
        );
        assert_eq!(
            b.marginal(9, &Evidence::none(5)),
            Err(QueryError::OutOfRange { var: 9, n: 5 })
        );
        assert_eq!(b.log_joint(&[true; 3]), Err(QueryError::EvidenceSize { got: 3, want: 5 }));
        assert_eq!(b.parents(2), &[0, 1]);
        assert_eq!(b.n(), 5);
        let e = evidence(5, &[(1, true)]);
        assert_eq!(e.observed(), 1);
        assert_eq!(e.vars(), 5);
        assert_eq!(e.get(1), Some(true));
        assert_eq!(e.clone().forget(1).get(1), None);
    }

    /// A contraction that does not fit the budget is refused, not attempted.
    #[test]
    fn an_oversized_contraction_is_refused() {
        let mut b = tangle().bayes();
        b.max_entries = 2;
        assert!(matches!(
            b.log_evidence(&Evidence::none(6)),
            Err(QueryError::TooWide { max: 2, .. })
        ));
        b.max_entries = 1 << 26;
        assert!(b.log_evidence(&Evidence::none(6)).is_ok());
    }
}
