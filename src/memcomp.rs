//! Digital memcomputing — self-organizing logic as a continuous flow.
//!
//! A digital memcomputing machine writes a Boolean problem as a dynamical system whose equilibria
//! are its solutions, then integrates the system instead of searching the assignments. The
//! construction is Traversa & Di Ventra, *Polynomial-time solution of prime factorization and
//! NP-complete problems with digital memcomputing machines*, Chaos **27**:023107 (2017); the
//! self-organizing-logic-gate equations integrated here are that line's later SAT form (Bearden,
//! Sheldon & Di Ventra, *Efficient solution of Boolean satisfiability problems with digital
//! memcomputing*, Sci. Rep. **10**:19741, 2020).
//!
//! # What is claimed here, and what is not
//!
//! **This module does not claim a polynomial-time solution of anything.** That claim belongs to the
//! source literature and it is contested: the standing objection is that the scaling argument rests
//! on real variables carried at unbounded precision, and that a digital integration of the flow pays
//! for that precision somewhere — in the step count, in the stiffness, or in the growth of the
//! memory variables. Nothing here tests either side of that argument, and no test in this file
//! measures a scaling.
//!
//! What runs is a **classical ODE integration on a CPU**: an explicit forward-Euler flow with an
//! adaptive step chosen by step doubling. Its standing is that of every other heuristic in this
//! crate — [`crate::sbm`]'s simulated bifurcation is the closest neighbour, a second continuous
//! trajectory read out by sign — a run either reaches a solution that is **verified against the
//! formula before it is returned**, or says it did not. There is no memristor model here, no
//! hardware, and a converged run is evidence about one instance and one seed.
//!
//! # Using it
//!
//! ```
//! use ferrotherm::memcomp::{Formula, Params, solve};
//!
//! // (x1 or x2 or not x3) and (not x1 or x3) and (x2 or x3)
//! let mut f = Formula::new(3);
//! f.push(&[1, 2, -3])?;
//! f.push(&[-1, 3])?;
//! f.push(&[2, 3])?;
//!
//! let out = solve(&f, Params::default(), 20_000, 7)?;
//! // `solution` is the ONLY way an assignment comes out, so a trajectory that did not get there
//! // cannot be mistaken for one that did.
//! let s = out.solution().expect("this formula is satisfiable and the trajectory found it");
//! assert!(f.is_satisfied_by(s));
//! # Ok::<(), ferrotherm::memcomp::Error>(())
//! ```
//!
//! # The equations
//!
//! For a clause `m` over literals with polarities `q_mj = ±1`, the voltages `v` in `[−1, 1]^N` are
//! the fast variables carrying the assignment — the gates' own short-term memory — and each clause
//! carries two memory variables of its own: `x_s` in `[0, 1]`, short-term, and `x_l` in
//! `[1, x_max]`, long-term. With
//!
//! ```text
//!   t_mj = 1 − q_mj v_j  in [0, 2]         how far literal j is from satisfying
//!   C_m  = ½ min_j t_mj  in [0, 1]         the clause residual: 0 satisfied, 1 fully violated
//!   G_mn = ½ q_mn min_{j ≠ n} t_mj         the gradient term
//!   R_mn = ½ (q_mn − v_n)  if t_mn = min_j t_mj,  else 0      the rigidity term
//! ```
//!
//! the flow is
//!
//! ```text
//!   dv_n/dt  = Σ_{m ∋ n}  x_lm x_sm G_mn + (1 + ζ x_lm)(1 − x_sm) R_mn
//!   dx_sm/dt = β (x_sm + ε)(C_m − γ)
//!   dx_lm/dt = α (C_m − δ)
//! ```
//!
//! with `v`, `x_s`, `x_l` clamped to their ranges after every sub-step. A clause that stays violated
//! drives its `x_l` up and is heard louder; a satisfied one lets `x_l` decay back to 1. That is the
//! point of the long-term memory, and it is why the **total** residual is not monotone: the memory
//! exists to pay a temporary rise in order to leave a local minimum. [`Machine::run`] is therefore
//! not a descent, and that is measured rather than asserted — on a 40-variable, 170-clause planted
//! instance the total residual `Σ_m C_m` **rose on 918 of the 3505 accepted steps** before the
//! first solution, worst single-step rise 0.037, with 156,732 rises among the individual clause
//! residuals. Anyone reaching for this flow as a descent method should read that number first.
//!
//! A clause of one literal has no "other" literals for `G` to minimise over. The empty minimum is
//! taken as `2`, the largest value `t` can hold: with nothing else in the clause able to satisfy it,
//! the drive on the single literal is maximal. That is a stated extension of the published 3-SAT
//! equations, not a default standing in for something unreadable.
//!
//! # What is exact here
//!
//! Two statements about the flow are proved rather than observed, and both are asserted as exact
//! inequalities in the tests:
//!
//! * **Sign-consistent formulas descend, at every step.** If every occurrence of a variable carries
//!   the same polarity `q_n`, then `q_n G_mn ≥ 0` and `q_n R_mn = ½(1 − q_n v_n) ≥ 0`, and both
//!   coefficients `x_l x_s` and `(1 + ζ x_l)(1 − x_s)` are non-negative, so `q_n v_n` is
//!   non-decreasing, every `t_mj` is non-increasing, and so is every `C_m`. Every operation on the
//!   path from `v` to `C_m` is monotone in `f64` as well as in the reals, so the tests assert it
//!   with no tolerance at all. It is a property of one class of instance, not of the flow in
//!   general.
//! * **The analog residual bounds the discrete count.** If the sign readout fails clause `m`, then
//!   every literal has `q_j v_j ≤ 0`, so `t_mj ≥ 1` and `C_m ≥ ½`. Hence the number of clauses the
//!   readout fails is at most `2 Σ_m C_m` — [`Machine::violation_bound`], accumulated with
//!   [`crate::round::sum_up`] so the bound survives floating point. Below 1 it certifies a
//!   solution. Everything else this module reports about a trajectory is a diagnostic and is summed
//!   in round-to-nearest; only this one is a bound, and only this one is rounded.

use crate::dimacs::{Cnf, Format};
use crate::rng::Pcg;
use crate::round::sum_up;

/// Readout: `v >= 0` is true (`+1`), `v < 0` is false (`−1`).
///
/// `sign(0) = +1`, fixed rather than left to chance — the convention [`crate::sbm`] reads its own
/// positions out with.
#[inline]
fn sign(x: f64) -> i8 {
    if x < 0.0 { -1 } else { 1 }
}

/// Variable index of a DIMACS literal.
#[inline]
fn lit_var(l: i32) -> usize {
    l.unsigned_abs() as usize - 1
}

/// Polarity of a DIMACS literal, as the `q` of the equations.
#[inline]
fn lit_q(l: i32) -> f64 {
    if l > 0 { 1.0 } else { -1.0 }
}

/// Why a formula, a parameter set or a trajectory was refused.
#[derive(Clone, Debug, PartialEq)]
pub enum Error {
    /// A clause over no literals. It is violated by every assignment, and its residual `C_m` is a
    /// minimum over an empty set, so there is nothing to integrate.
    EmptyClause {
        /// Which clause, in the order they were added.
        clause: usize,
    },
    /// Literal `0`, which names no variable in DIMACS numbering.
    ZeroLiteral {
        /// Which clause it appeared in.
        clause: usize,
    },
    /// A literal naming a variable the formula does not have.
    UnknownVariable {
        /// The literal as written.
        lit: i32,
        /// How many variables the formula declares.
        vars: usize,
    },
    /// One clause naming the same variable twice. `x or x` is `x` and `x or not x` is a tautology;
    /// either way the clause written is not the clause meant, so it is named rather than repaired.
    RepeatedVariable {
        /// The variable, in DIMACS numbering.
        var: usize,
        /// Which clause.
        clause: usize,
    },
    /// A weighted clause. A digital memcomputing machine solves a **decision** problem: every clause
    /// must end satisfied, and there is no price at which one may be left broken. A MAX-SAT instance
    /// is a different question — [`crate::dimacs::Cnf::to_hubo`] is the route for that one.
    SoftClause {
        /// Which clause.
        clause: usize,
        /// The weight it carried.
        weight: f64,
    },
    /// A parameter outside the range its term is defined on.
    BadParam {
        /// Which one.
        name: &'static str,
        /// What arrived.
        got: f64,
    },
    /// The adaptive step reached its floor and still could not meet the tolerance, or the state left
    /// the reals. The trajectory is abandoned: integrating past this point would report numbers the
    /// integrator itself does not stand behind.
    StepFloor {
        /// The step size at the floor.
        h: f64,
        /// The scaled local-error estimate there, or `NaN` if the state went non-finite.
        err: f64,
    },
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Error::EmptyClause { clause } => {
                write!(f, "clause {clause} has no literals, so its residual minimises over nothing")
            }
            Error::ZeroLiteral { clause } => {
                write!(f, "clause {clause} contains literal 0, which names no variable")
            }
            Error::UnknownVariable { lit, vars } => {
                write!(f, "literal {lit} names a variable past the {vars} this formula declares")
            }
            Error::RepeatedVariable { var, clause } => write!(
                f,
                "variable {var} appears twice in clause {clause}; that is not the clause that was \
                 written"
            ),
            Error::SoftClause { clause, weight } => write!(
                f,
                "clause {clause} carries weight {weight}, and a memcomputing machine has no price \
                 for a broken clause; this solves satisfiability, not MAX-SAT"
            ),
            Error::BadParam { name, got } => {
                write!(f, "parameter {name} is {got}, outside the range its term is defined on")
            }
            Error::StepFloor { h, err } => write!(
                f,
                "the adaptive step hit its floor {h:e} with scaled error {err:e}; the trajectory is \
                 abandoned rather than integrated past the tolerance"
            ),
        }
    }
}

impl core::error::Error for Error {}

/// A CNF formula in DIMACS numbering: variable `i + 1` is spin `i`, `+1` true.
///
/// Built clause by clause with [`Formula::push`], which refuses anything the flow cannot represent
/// rather than repairing it.
#[derive(Clone, Debug, PartialEq)]
pub struct Formula {
    vars: usize,
    clauses: Vec<Vec<i32>>,
}

impl Formula {
    /// An empty formula over `vars` variables.
    #[must_use]
    pub fn new(vars: usize) -> Formula {
        Formula { vars, clauses: Vec::new() }
    }

    /// Add a clause, as DIMACS literals (`+v` positive, `−v` negated, one-based).
    ///
    /// # Errors
    ///
    /// [`Error::EmptyClause`], [`Error::ZeroLiteral`], [`Error::UnknownVariable`] or
    /// [`Error::RepeatedVariable`], each naming what arrived.
    pub fn push(&mut self, lits: &[i32]) -> Result<(), Error> {
        let clause = self.clauses.len();
        if lits.is_empty() {
            return Err(Error::EmptyClause { clause });
        }
        for (a, &l) in lits.iter().enumerate() {
            if l == 0 {
                return Err(Error::ZeroLiteral { clause });
            }
            let v = l.unsigned_abs() as usize;
            if v > self.vars {
                return Err(Error::UnknownVariable { lit: l, vars: self.vars });
            }
            if lits[..a].iter().any(|&p| p.unsigned_abs() == l.unsigned_abs()) {
                return Err(Error::RepeatedVariable { var: v, clause });
            }
        }
        self.clauses.push(lits.to_vec());
        Ok(())
    }

    /// Take a parsed DIMACS instance, provided it is a satisfiability problem.
    ///
    /// A plain `p cnf` file is one. A weighted file is one only if every clause is HARD; a finite
    /// weight is a MAX-SAT objective and is refused by name.
    ///
    /// # Errors
    ///
    /// [`Error::SoftClause`] for a weighted clause, and whatever [`Formula::push`] refuses —
    /// including [`Error::EmptyClause`], which DIMACS admits and this does not.
    pub fn from_cnf(cnf: &Cnf) -> Result<Formula, Error> {
        let mut f = Formula::new(cnf.vars);
        for (i, c) in cnf.clauses.iter().enumerate() {
            if cnf.format != Format::Cnf && !c.is_hard() {
                return Err(Error::SoftClause { clause: i, weight: c.weight });
            }
            f.push(&c.lits)?;
        }
        Ok(f)
    }

    /// How many variables the formula declares.
    #[must_use]
    pub fn vars(&self) -> usize {
        self.vars
    }

    /// The clauses, in the order they were added.
    #[must_use]
    pub fn clauses(&self) -> &[Vec<i32>] {
        &self.clauses
    }

    /// How many clauses `s` fails. Spin `i` is variable `i + 1`, `+1` true.
    ///
    /// # Panics
    ///
    /// If `s` is shorter than [`Formula::vars`] — a state that does not cover the formula cannot be
    /// scored against it, and scoring it anyway is how a partial assignment comes to look feasible.
    #[must_use]
    pub fn unsatisfied(&self, s: &[i8]) -> usize {
        assert!(s.len() >= self.vars, "a state of {} cannot cover {} variables", s.len(), self.vars);
        self.clauses
            .iter()
            .filter(|c| {
                c.iter().all(|&l| s[lit_var(l)] != if l > 0 { 1 } else { -1 })
            })
            .count()
    }

    /// Whether `s` satisfies every clause.
    ///
    /// # Panics
    ///
    /// If `s` is shorter than [`Formula::vars`].
    #[must_use]
    pub fn is_satisfied_by(&self, s: &[i8]) -> bool {
        self.unsatisfied(s) == 0
    }
}

/// The flow's constants, and the integrator's.
///
/// The dynamical constants are the published 3-SAT set (`α = 5`, `β = 20`, `γ = 0.25`, `δ = 0.05`,
/// `ε = 1e-3`, `ζ = 0.1`). They are a struct rather than literals because they are a modelling
/// choice and a caller may want another one; **no test in this module depends on their values**,
/// only on the mathematics of the flow they parameterise.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Params {
    /// Long-term memory rate: `dx_l/dt = α (C − δ)`. Above zero.
    pub alpha: f64,
    /// Short-term memory rate: `dx_s/dt = β (x_s + ε)(C − γ)`. Above zero.
    pub beta: f64,
    /// The residual a clause must beat for its short-term memory to relax. In `(0, 1)`.
    pub gamma: f64,
    /// The residual a clause must beat for its long-term memory to decay. In `(0, 1)`.
    pub delta: f64,
    /// Keeps `x_s = 0` from being a fixed point of its own equation. Above zero.
    pub epsilon: f64,
    /// How much the long-term memory amplifies the rigidity term. Non-negative.
    pub zeta: f64,
    /// Ceiling on `x_l`. A clause that is never satisfied would otherwise grow without bound and
    /// take the step size to the floor. At least 1; exactly 1 pins the long-term memory shut.
    pub xl_max: f64,
    /// First step size. Above zero.
    pub h0: f64,
    /// Smallest step the integrator will take before it gives up. Above zero.
    pub h_min: f64,
    /// Largest step the integrator will take. At least `h_min`.
    pub h_max: f64,
    /// Absolute component of the local-error scale. Above zero.
    pub atol: f64,
    /// Relative component of the local-error scale. Above zero.
    pub rtol: f64,
}

impl Default for Params {
    fn default() -> Params {
        Params {
            alpha: 5.0,
            beta: 20.0,
            gamma: 0.25,
            delta: 0.05,
            epsilon: 1e-3,
            zeta: 0.1,
            xl_max: 1e4,
            h0: 0.1,
            h_min: 1e-9,
            h_max: 1.0,
            atol: 1e-6,
            rtol: 1e-4,
        }
    }
}

impl Params {
    /// Refuse a parameter set the equations are not defined on.
    ///
    /// # Errors
    ///
    /// [`Error::BadParam`] naming the first parameter out of range and the value it held.
    pub fn check(&self) -> Result<(), Error> {
        for (name, got, ok) in [
            ("alpha", self.alpha, self.alpha > 0.0),
            ("beta", self.beta, self.beta > 0.0),
            ("gamma", self.gamma, self.gamma > 0.0 && self.gamma < 1.0),
            ("delta", self.delta, self.delta > 0.0 && self.delta < 1.0),
            ("epsilon", self.epsilon, self.epsilon > 0.0),
            ("zeta", self.zeta, self.zeta >= 0.0),
            ("xl_max", self.xl_max, self.xl_max >= 1.0),
            ("h0", self.h0, self.h0 > 0.0),
            ("h_min", self.h_min, self.h_min > 0.0),
            ("h_max", self.h_max, self.h_max >= self.h_min),
            ("atol", self.atol, self.atol > 0.0),
            ("rtol", self.rtol, self.rtol > 0.0),
        ] {
            if !(ok && got.is_finite()) {
                return Err(Error::BadParam { name, got });
            }
        }
        Ok(())
    }
}

/// How a trajectory ended. Non-convergence is one of these, never an assignment.
#[derive(Clone, Debug, PartialEq)]
pub enum Outcome {
    /// A satisfying assignment, checked against every clause before this was built.
    Solved {
        /// Spin per variable, `+1` true.
        assignment: Vec<i8>,
        /// Accepted integration steps taken.
        steps: usize,
        /// Integrated time reached.
        t: f64,
    },
    /// The step budget ran out. Carries the best readout the trajectory passed through and how many
    /// clauses that readout fails — at least one, by construction.
    Exhausted {
        /// The best readout seen, which is **not** a solution.
        best: Vec<i8>,
        /// How many clauses it fails.
        unsatisfied: usize,
        /// Accepted integration steps taken.
        steps: usize,
        /// Integrated time reached.
        t: f64,
    },
    /// The integrator refused to continue; see [`Error::StepFloor`]. Nothing is claimed about the
    /// formula: this is a statement about the trajectory, not about satisfiability.
    Stalled {
        /// The best readout seen.
        best: Vec<i8>,
        /// How many clauses it fails.
        unsatisfied: usize,
        /// Accepted integration steps taken.
        steps: usize,
        /// Integrated time reached.
        t: f64,
        /// Why the integrator stopped.
        why: Error,
    },
}

impl Outcome {
    /// The satisfying assignment, or `None`. The only way to read a solution out of an outcome, so
    /// that a caller cannot mistake a best-effort readout for one.
    #[must_use]
    pub fn solution(&self) -> Option<&[i8]> {
        match self {
            Outcome::Solved { assignment, .. } => Some(assignment),
            _ => None,
        }
    }

    /// Accepted integration steps taken, whatever the ending.
    #[must_use]
    pub fn steps(&self) -> usize {
        match self {
            Outcome::Solved { steps, .. }
            | Outcome::Exhausted { steps, .. }
            | Outcome::Stalled { steps, .. } => *steps,
        }
    }
}

/// The right-hand side of the flow at `y`, into `d`.
///
/// A free function rather than a method so that [`Machine::step`] can borrow the state and the
/// derivative buffer at once; it is the same arithmetic either way.
fn deriv(f: &Formula, p: &Params, y: &[f64], d: &mut [f64]) {
    let (n, m) = (f.vars, f.clauses.len());
    d[..n].fill(0.0);
    for (mi, cl) in f.clauses.iter().enumerate() {
        let (xs, xl) = (y[n + mi], y[n + m + mi]);
        // t_j = 1 - q_j v_j, and the two smallest of them: G_mn minimises over the OTHER literals,
        // which is the second smallest exactly when n is the one attaining the smallest.
        let (mut min1, mut min2, mut idx1) = (f64::INFINITY, f64::INFINITY, 0usize);
        for (j, &l) in cl.iter().enumerate() {
            let t = 1.0 - lit_q(l) * y[lit_var(l)];
            if t < min1 {
                min2 = min1;
                min1 = t;
                idx1 = j;
            } else if t < min2 {
                min2 = t;
            }
        }
        let cm = 0.5 * min1;
        for (j, &l) in cl.iter().enumerate() {
            let (var, q) = (lit_var(l), lit_q(l));
            let t = 1.0 - q * y[var];
            // A unit clause has no other literal; the empty minimum is the largest t can be.
            let other = if cl.len() == 1 {
                2.0
            } else if j == idx1 {
                min2
            } else {
                min1
            };
            let g = 0.5 * q * other;
            // The rigidity term is gated on the paper's condition literally: `C_m` IS this
            // literal's half-distance. Exact equality, because `t` is recomputed by the same
            // expression that produced `min1` and so is bit-identical — and because a tie is a
            // genuine tie, where every literal attaining the minimum is meant to be held.
            let r = if t == min1 { 0.5 * (q - y[var]) } else { 0.0 };
            d[var] += xl * xs * g + (1.0 + p.zeta * xl) * (1.0 - xs) * r;
        }
        d[n + mi] = p.beta * (xs + p.epsilon) * (cm - p.gamma);
        d[n + m + mi] = p.alpha * (cm - p.delta);
    }
}

/// One explicit Euler sub-step, clamped back into the domain.
fn euler(n: usize, m: usize, xl_max: f64, y: &[f64], d: &[f64], h: f64, out: &mut [f64]) {
    for i in 0..n {
        out[i] = (y[i] + h * d[i]).clamp(-1.0, 1.0);
    }
    for i in n..n + m {
        out[i] = (y[i] + h * d[i]).clamp(0.0, 1.0);
    }
    for i in n + m..n + 2 * m {
        out[i] = (y[i] + h * d[i]).clamp(1.0, xl_max);
    }
}

/// One trajectory of the flow over one formula.
///
/// The state is `[v | x_s | x_l]` in a single vector, which is what lets the step-doubling error
/// estimate treat every variable alike.
pub struct Machine<'a> {
    f: &'a Formula,
    p: Params,
    y: Vec<f64>,
    y1: Vec<f64>,
    y2: Vec<f64>,
    ymid: Vec<f64>,
    d: Vec<f64>,
    dmid: Vec<f64>,
    c: Vec<f64>,
    h: f64,
    t: f64,
    steps: usize,
}

impl<'a> Machine<'a> {
    /// Start a trajectory: `v` uniform on `[−1, 1]` from `seed`, `x_s = 0`, `x_l = 1`.
    ///
    /// # Errors
    ///
    /// [`Error::BadParam`] if the parameters are out of range, [`Error::EmptyClause`] if the formula
    /// has no clauses — a flow with nothing to satisfy has no equilibria to find.
    pub fn new(f: &'a Formula, p: Params, seed: u64) -> Result<Machine<'a>, Error> {
        p.check()?;
        if f.clauses.is_empty() {
            return Err(Error::EmptyClause { clause: 0 });
        }
        let (n, m) = (f.vars, f.clauses.len());
        let mut rng = Pcg::new(seed, 0x4D454D43); // "MEMC"
        let mut y = vec![0.0f64; n + 2 * m];
        for i in 0..n {
            y[i] = 2.0 * rng.f64() - 1.0;
        }
        for j in 0..m {
            y[n + j] = 0.0;
            y[n + m + j] = 1.0;
        }
        let z = vec![0.0f64; n + 2 * m];
        let mut mc = Machine {
            f,
            p,
            y,
            y1: z.clone(),
            y2: z.clone(),
            ymid: z.clone(),
            d: z.clone(),
            dmid: z,
            c: vec![0.0; m],
            h: p.h0.clamp(p.h_min, p.h_max),
            t: 0.0,
            steps: 0,
        };
        mc.refresh();
        Ok(mc)
    }

    /// The clause residuals `C_m` at the current state, in clause order.
    #[must_use]
    pub fn residuals(&self) -> &[f64] {
        &self.c
    }

    /// `Σ_m C_m`, summed left to right. A **diagnostic**, not a bound: see
    /// [`Machine::violation_bound`] for the one number here that is one.
    #[must_use]
    pub fn residual(&self) -> f64 {
        self.c.iter().sum()
    }

    /// An upper bound on how many clauses the current sign readout fails: `2 Σ_m C_m`, accumulated
    /// with [`crate::round::sum_up`] so it cannot round below the truth.
    ///
    /// A failed clause has every literal at `q_j v_j <= 0`, hence `t_mj >= 1` and `C_m >= ½`, so
    /// half the residual sum already counts the failures. Below `1` the bound certifies that the
    /// readout satisfies the formula.
    #[must_use]
    pub fn violation_bound(&self) -> f64 {
        2.0 * sum_up(&self.c)
    }

    /// The current assignment: `sign(v)` per variable, `+1` true.
    #[must_use]
    pub fn readout(&self) -> Vec<i8> {
        self.y[..self.f.vars].iter().map(|&v| sign(v)).collect()
    }

    /// How many clauses the current readout fails.
    #[must_use]
    pub fn unsatisfied(&self) -> usize {
        self.f.unsatisfied(&self.readout())
    }

    /// The voltages `v`, the assignment before it is rounded.
    #[must_use]
    pub fn voltages(&self) -> &[f64] {
        &self.y[..self.f.vars]
    }

    /// The short-term clause memories `x_s`.
    #[must_use]
    pub fn short_term(&self) -> &[f64] {
        let (n, m) = (self.f.vars, self.f.clauses.len());
        &self.y[n..n + m]
    }

    /// The long-term clause memories `x_l`.
    #[must_use]
    pub fn long_term(&self) -> &[f64] {
        let (n, m) = (self.f.vars, self.f.clauses.len());
        &self.y[n + m..n + 2 * m]
    }

    /// Integrated time so far.
    #[must_use]
    pub fn time(&self) -> f64 {
        self.t
    }

    /// The step size the integrator has settled on.
    #[must_use]
    pub fn step_size(&self) -> f64 {
        self.h
    }

    /// Accepted steps so far.
    #[must_use]
    pub fn steps(&self) -> usize {
        self.steps
    }

    /// Recompute the clause residuals from the current state.
    fn refresh(&mut self) {
        for (mi, cl) in self.f.clauses.iter().enumerate() {
            let mut min1 = f64::INFINITY;
            for &l in cl {
                let t = 1.0 - lit_q(l) * self.y[lit_var(l)];
                if t < min1 {
                    min1 = t;
                }
            }
            self.c[mi] = 0.5 * min1;
        }
    }

    /// Advance one accepted step, returning the step size taken.
    ///
    /// The step is chosen by doubling: one step of `h` against two of `h/2`, keep the pair, resize
    /// from the disagreement. A step that cannot meet the tolerance at [`Params::h_min`] is refused
    /// rather than taken — an integrator that quietly accepts whatever it can manage is a solver
    /// reporting a trajectory it did not compute.
    ///
    /// # Errors
    ///
    /// [`Error::StepFloor`] when the tolerance cannot be met at the smallest allowed step, or when
    /// the state leaves the reals.
    pub fn step(&mut self) -> Result<f64, Error> {
        // Enough shrinks to cross the whole [h_max, h_min] range at the 0.2 floor per try.
        const TRIES: usize = 64;
        let (n, m) = (self.f.vars, self.f.clauses.len());
        for _ in 0..TRIES {
            deriv(self.f, &self.p, &self.y, &mut self.d);
            if !self.d.iter().all(|x| x.is_finite()) {
                return Err(Error::StepFloor { h: self.h, err: f64::NAN });
            }
            let h = self.h;
            euler(n, m, self.p.xl_max, &self.y, &self.d, h, &mut self.y1);
            euler(n, m, self.p.xl_max, &self.y, &self.d, 0.5 * h, &mut self.ymid);
            deriv(self.f, &self.p, &self.ymid, &mut self.dmid);
            euler(n, m, self.p.xl_max, &self.ymid, &self.dmid, 0.5 * h, &mut self.y2);
            let mut err = 0.0f64;
            for i in 0..self.y.len() {
                let sc = self.p.atol + self.p.rtol * self.y2[i].abs();
                err = err.max((self.y2[i] - self.y1[i]).abs() / sc);
            }
            if !err.is_finite() {
                return Err(Error::StepFloor { h, err });
            }
            if err <= 1.0 {
                std::mem::swap(&mut self.y, &mut self.y2);
                self.t += h;
                self.steps += 1;
                self.refresh();
                // Forward Euler is first order, so the local error scales as h: grow by tol/err.
                let grow = if err > 0.0 { (0.9 / err).min(4.0) } else { 4.0 };
                self.h = (h * grow).clamp(self.p.h_min, self.p.h_max);
                return Ok(h);
            }
            if h <= self.p.h_min {
                return Err(Error::StepFloor { h, err });
            }
            self.h = (h * (0.9 / err).max(0.2)).clamp(self.p.h_min, self.p.h_max);
        }
        Err(Error::StepFloor { h: self.h, err: f64::INFINITY })
    }

    /// Integrate until the readout satisfies the formula, or `max_steps` accepted steps pass.
    ///
    /// The readout is checked after every accepted step, and the assignment in [`Outcome::Solved`]
    /// has been verified against every clause. A trajectory that does not get there returns
    /// [`Outcome::Exhausted`] or [`Outcome::Stalled`] with its best readout and the number of
    /// clauses that readout fails.
    pub fn run(&mut self, max_steps: usize) -> Outcome {
        let mut best = self.readout();
        let mut best_u = self.f.unsatisfied(&best);
        if best_u == 0 {
            return Outcome::Solved { assignment: best, steps: self.steps, t: self.t };
        }
        for _ in 0..max_steps {
            if let Err(why) = self.step() {
                return Outcome::Stalled { best, unsatisfied: best_u, steps: self.steps, t: self.t, why };
            }
            let s = self.readout();
            let u = self.f.unsatisfied(&s);
            if u == 0 {
                return Outcome::Solved { assignment: s, steps: self.steps, t: self.t };
            }
            if u < best_u {
                best_u = u;
                best = s;
            }
        }
        Outcome::Exhausted { best, unsatisfied: best_u, steps: self.steps, t: self.t }
    }
}

/// One trajectory from `seed`, integrated for at most `max_steps` accepted steps.
///
/// # Errors
///
/// [`Error::BadParam`] or [`Error::EmptyClause`] from [`Machine::new`]. A trajectory that fails to
/// solve the formula is an [`Outcome`], not an error.
pub fn solve(f: &Formula, p: Params, max_steps: usize, seed: u64) -> Result<Outcome, Error> {
    Ok(Machine::new(f, p, seed)?.run(max_steps))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dimacs::Cnf;

    /// A random 3-SAT instance. With `plant`, only clauses that assignment satisfies are kept, so
    /// the instance is satisfiable by construction; without it, satisfiability is whatever the
    /// draw gives and the enumeration in the test is what decides.
    fn random_3sat(vars: usize, clauses: usize, plant: Option<&[i8]>, seed: u64) -> Formula {
        let mut rng = Pcg::new(seed, 0xC0FFEE);
        let mut f = Formula::new(vars);
        while f.clauses().len() < clauses {
            let mut picked: Vec<usize> = Vec::new();
            while picked.len() < 3 {
                let v = (rng.next_u32() as usize) % vars;
                if !picked.contains(&v) {
                    picked.push(v);
                }
            }
            let lits: Vec<i32> = picked
                .iter()
                .map(|&v| if rng.f64() < 0.5 { -(v as i32 + 1) } else { v as i32 + 1 })
                .collect();
            if let Some(p) = plant
                && !lits.iter().any(|&l| p[lit_var(l)] == if l > 0 { 1 } else { -1 })
            {
                continue;
            }
            f.push(&lits).expect("generated clauses are non-empty and repeat no variable");
        }
        f
    }

    /// A formula where every occurrence of a variable carries the SAME polarity, half of them
    /// negative so a dropped `q` in the gradient term cannot hide. This is the class the descent
    /// proof in the module docs covers, and it is satisfied by `s_n = q_n`.
    fn sign_consistent(vars: usize, clauses: usize, seed: u64) -> Formula {
        let mut rng = Pcg::new(seed, 77);
        let pol: Vec<i32> = (0..vars).map(|_| if rng.f64() < 0.5 { -1 } else { 1 }).collect();
        let mut f = Formula::new(vars);
        while f.clauses().len() < clauses {
            let mut picked: Vec<usize> = Vec::new();
            while picked.len() < 3 {
                let v = (rng.next_u32() as usize) % vars;
                if !picked.contains(&v) {
                    picked.push(v);
                }
            }
            let lits: Vec<i32> = picked.iter().map(|&v| pol[v] * (v as i32 + 1)).collect();
            f.push(&lits).expect("three distinct variables, one literal each");
        }
        f
    }

    /// Every satisfying assignment, by enumerating all `2^n`. The independent oracle: it knows
    /// nothing of the flow, only of the clauses.
    fn all_solutions(f: &Formula) -> Vec<Vec<i8>> {
        let n = f.vars();
        assert!(n <= 20, "enumerating {n} variables is not cheap");
        (0..(1u32 << n))
            .map(|mask| (0..n).map(|i| if mask >> i & 1 == 1 { 1i8 } else { -1 }).collect::<Vec<i8>>())
            .filter(|s| f.is_satisfied_by(s))
            .collect()
    }

    /// The fewest clauses any assignment fails — the MAX-SAT optimum, by enumeration.
    fn min_unsatisfied(f: &Formula) -> usize {
        let n = f.vars();
        assert!(n <= 20, "enumerating {n} variables is not cheap");
        (0..(1u32 << n))
            .map(|mask| {
                let s: Vec<i8> = (0..n).map(|i| if mask >> i & 1 == 1 { 1 } else { -1 }).collect();
                f.unsatisfied(&s)
            })
            .min()
            .unwrap_or(0)
    }

    /// The eight clauses over three variables that rule out all eight assignments: UNSAT, and the
    /// test checks that by enumeration rather than asserting it.
    fn unsat_eight() -> Formula {
        let mut f = Formula::new(3);
        for m in 0..8i32 {
            let lits: Vec<i32> =
                (0..3).map(|b| if m >> b & 1 == 1 { b + 1 } else { -(b + 1) }).collect();
            f.push(&lits).unwrap();
        }
        f
    }

    /// THE LYAPUNOV-LIKE PROPERTY, against the closed-form proof in the module docs and with no
    /// tolerance at all.
    ///
    /// On a sign-consistent formula every drive on `v_n` carries the sign of `q_n`, so `q_n v_n`
    /// only rises, every `t_mj` only falls, and so does every `C_m`. Each step of that chain is
    /// monotone in `f64` as well as in the reals — a clamped `y + h d` with `h d` of a fixed sign,
    /// a subtraction, a minimum, a halving, and a left-to-right sum — so `>=` is asserted exactly.
    /// An epsilon here would be an admission that the proof is not the thing being checked.
    ///
    /// Half the variables are negative-polarity, so the `q` in `G_mn` is load-bearing: drop it and
    /// those variables are driven the wrong way and their residuals climb.
    #[test]
    fn a_sign_consistent_flow_lowers_every_residual_at_every_step_exactly() {
        for seed in 0..3u64 {
            let f = sign_consistent(16, 40, seed);
            let mut mc = Machine::new(&f, Params::default(), seed).unwrap();
            let start = mc.residual();
            assert!(start > 4.0, "a flat start would make this test vacuous: {start}");
            let mut prev: Vec<f64> = mc.residuals().to_vec();
            let mut prev_total = start;
            for k in 0..4000 {
                mc.step().unwrap();
                for (m, (&now, &was)) in mc.residuals().iter().zip(&prev).enumerate() {
                    assert!(now <= was, "seed {seed} step {k} clause {m}: {was} rose to {now}");
                }
                let total = mc.residual();
                assert!(total <= prev_total, "seed {seed} step {k}: {prev_total} rose to {total}");
                prev.copy_from_slice(mc.residuals());
                prev_total = total;
            }
            // And it descended to the solution rather than stalling: the monotonicity above is
            // only worth asserting on a trajectory that moved.
            assert!(prev_total < 1e-12, "seed {seed} ended at residual {prev_total}");
            assert_eq!(mc.unsatisfied(), 0, "seed {seed}");
        }
    }

    /// THE SAME QUANTITY RISES ON A FRUSTRATED INSTANCE, which is why the descent above is stated
    /// for one class of formula and not for the flow.
    ///
    /// This is the measurement behind the claim in the module docs that `run` is not a descent. It
    /// is also what stops the test above from passing an implementation whose residual never moves:
    /// the same accumulator, on a different instance, is asserted to go UP.
    ///
    /// Measured on the instance below (40 variables, 170 clauses, planted): the total residual rose
    /// on 918 of the 3505 accepted steps that precede the first solution, worst single-step rise
    /// 0.037, and individual clause residuals rose 156,732 times.
    #[test]
    fn the_total_residual_rises_on_a_frustrated_instance_so_it_is_no_lyapunov_function() {
        let mut rng = Pcg::new(0, 5);
        let plant: Vec<i8> = (0..40).map(|_| rng.spin(0.5)).collect();
        let f = random_3sat(40, 170, Some(&plant), 0);
        let mut mc = Machine::new(&f, Params::default(), 0).unwrap();
        let mut prev = mc.residual();
        let (mut rises, mut worst) = (0usize, 0.0f64);
        let mut steps_before = 0usize;
        for _ in 0..40_000 {
            mc.step().unwrap();
            if mc.unsatisfied() == 0 {
                break;
            }
            steps_before += 1;
            let now = mc.residual();
            if now > prev {
                rises += 1;
                worst = worst.max(now - prev);
            }
            prev = now;
        }
        assert_eq!(mc.unsatisfied(), 0, "the instance is planted and must be solved");
        assert!(
            rises > 100,
            "only {rises} rises in {steps_before} steps: the memory terms are not doing anything"
        );
        assert!(worst > 1e-3, "the rises are rounding noise, not escapes: worst {worst}");
    }

    /// THE READOUT ORACLE: exhaustive enumeration over all `2^14` assignments.
    ///
    /// For each instance the enumeration decides satisfiability and holds every solution. Then:
    /// a run that reports [`Outcome::Solved`] must return one of those assignments, and an instance
    /// the enumeration proves UNSAT must never be reported solved. The `unsatisfied` count an
    /// exhausted run reports is checked against the enumerated MAX-SAT optimum, which is the one
    /// number that can catch a best-so-far tracker that is counting something else.
    ///
    /// Both halves are asserted to be populated, so the test cannot pass by never solving anything
    /// or by drawing only satisfiable instances. Measured: 8 satisfiable, 16 unsatisfiable, 8 of 8
    /// solved, 0 false claims.
    #[test]
    fn converged_readouts_are_solutions_under_exhaustive_enumeration() {
        let (mut sat, mut unsat, mut solved) = (0usize, 0usize, 0usize);
        for seed in 0..24u64 {
            let f = random_3sat(14, 70, None, seed);
            let sols = all_solutions(&f);
            let floor = min_unsatisfied(&f);
            let out = solve(&f, Params::default(), 20_000, seed).unwrap();
            match &out {
                Outcome::Solved { assignment, .. } => {
                    assert!(!sols.is_empty(), "seed {seed}: claimed a solution to an UNSAT formula");
                    assert!(
                        sols.iter().any(|s| s == assignment),
                        "seed {seed}: {assignment:?} is not one of the {} enumerated solutions",
                        sols.len()
                    );
                    solved += 1;
                }
                Outcome::Exhausted { best, unsatisfied, .. } => {
                    assert_eq!(*unsatisfied, f.unsatisfied(best), "seed {seed}: the count and the state disagree");
                    assert!(
                        *unsatisfied >= floor,
                        "seed {seed}: reported {unsatisfied} failures below the enumerated optimum {floor}"
                    );
                    assert!(*unsatisfied > 0, "seed {seed}: an exhausted run holding a solution");
                }
                Outcome::Stalled { .. } => panic!("seed {seed}: unexpected stall on a well-posed run"),
            }
            if sols.is_empty() {
                unsat += 1;
            } else {
                sat += 1;
            }
        }
        assert!(sat >= 5 && unsat >= 5, "the draw must exercise both arms: {sat} SAT, {unsat} UNSAT");
        assert_eq!(solved, sat, "every satisfiable instance in this set is solvable in budget");
    }

    /// THE ASYMMETRIC TEST: an UNSAT instance must be REFUSED, and its satisfiable sibling SOLVED,
    /// under the same parameters, budget and seed.
    ///
    /// The eight clauses over three variables rule out all eight assignments; dropping one leaves
    /// exactly one solution. Enumeration establishes both facts here rather than the test asserting
    /// them. A solver that converged to "something" would pass the sibling half and fail this one,
    /// which is the point: the first half asserts the failure.
    #[test]
    fn an_unsat_instance_is_refused_while_its_satisfiable_sibling_is_solved() {
        let unsat = unsat_eight();
        assert!(all_solutions(&unsat).is_empty(), "the fixture must actually be UNSAT");
        let mut sibling = Formula::new(3);
        for c in unsat.clauses().iter().skip(1) {
            sibling.push(c).unwrap();
        }
        let sols = all_solutions(&sibling);
        assert_eq!(sols, vec![vec![1i8, 1, 1]], "dropping the first clause must leave one solution");

        for seed in 0..4u64 {
            let bad = solve(&unsat, Params::default(), 20_000, seed).unwrap();
            assert_eq!(bad.solution(), None, "seed {seed}: an assignment was returned for an UNSAT formula");
            match bad {
                Outcome::Exhausted { best, unsatisfied, .. } => {
                    assert!(unsatisfied >= 1);
                    assert_eq!(unsatisfied, unsat.unsatisfied(&best));
                }
                other => panic!("seed {seed}: expected the budget to run out, got {other:?}"),
            }
            let good = solve(&sibling, Params::default(), 20_000, seed).unwrap();
            assert_eq!(good.solution(), Some(&[1i8, 1, 1][..]), "seed {seed}: the sibling is solvable");
        }
    }

    /// THE BOUND: `2 Σ_m C_m` may never sit below the number of clauses the sign readout fails.
    ///
    /// Proved in the module docs — a failed clause has `C_m >= ½` — and rounded UP through
    /// [`crate::round::sum_up`], so the claim survives floating point. Asserted at every accepted
    /// step of a long frustrated run, in both regimes: steps where the bound is loose and the
    /// readout is failing clauses, and steps where it has dropped below 1, which certifies that the
    /// readout satisfies the formula.
    #[test]
    fn the_violation_bound_never_undercounts_the_readout_that_round_up_protects() {
        let mut rng = Pcg::new(2, 5);
        let plant: Vec<i8> = (0..30).map(|_| rng.spin(0.5)).collect();
        let f = random_3sat(30, 126, Some(&plant), 3);
        let mut mc = Machine::new(&f, Params::default(), 3).unwrap();
        let (mut certified, mut failing) = (0usize, 0usize);
        for k in 0..8000 {
            mc.step().unwrap();
            let u = mc.unsatisfied();
            let b = mc.violation_bound();
            assert!(u as f64 <= b, "step {k}: {u} failures under a bound of {b}");
            assert!(b >= 2.0 * mc.residual() - 1e-12, "the bound must not round below the sum: {b}");
            if b < 1.0 {
                assert_eq!(u, 0, "step {k}: a bound below 1 must certify the readout");
                certified += 1;
            }
            if u > 0 {
                failing += 1;
            }
        }
        assert!(certified > 0, "the certifying regime was never reached, so the bound tested nothing");
        assert!(failing > 0, "the failing regime was never reached, so the bound tested nothing");
    }

    /// THE LONG-TERM MEMORY IS LOAD-BEARING, not decoration on a gradient descent.
    ///
    /// Pinning `x_l` shut (`xl_max = 1`, `ζ = 0`) leaves a legal flow — the gradient and rigidity
    /// terms still run — and it loses instances the full flow solves from the same seed within the
    /// same budget. Without this, an implementation that dropped the `x_l` factor entirely would
    /// still pass every other test in this file, since roughly half of these instances fall to the
    /// short-term dynamics alone.
    ///
    /// Measured over seeds 0..6 at 50 variables and 213 clauses: full 6/6, pinned 3/6.
    #[test]
    fn pinning_the_long_term_memory_loses_instances_the_full_flow_solves() {
        let pinned = Params { xl_max: 1.0, zeta: 0.0, ..Params::default() };
        let mut lost = 0usize;
        for seed in 0..6u64 {
            let mut rng = Pcg::new(seed + 100, 5);
            let plant: Vec<i8> = (0..50).map(|_| rng.spin(0.5)).collect();
            let f = random_3sat(50, 213, Some(&plant), seed);
            let full = solve(&f, Params::default(), 30_000, seed).unwrap();
            let cut = solve(&f, pinned, 30_000, seed).unwrap();
            let a = full.solution().expect("the full flow solves all six of these");
            assert!(f.is_satisfied_by(a));
            if cut.solution().is_none() {
                lost += 1;
            } else {
                assert!(f.is_satisfied_by(cut.solution().unwrap()), "a pinned run must still be honest");
            }
        }
        assert!(lost >= 2, "pinning the long-term memory cost nothing ({lost} of 6 lost)");
        assert!(lost < 6, "the pinned flow is broken rather than weakened, which tests something else");
    }

    /// A DIMACS FILE IN, AND [`crate::dimacs::Cnf`] SCORES THE ANSWER — a module with its own
    /// oracle, not this one, deciding whether the assignment is a solution.
    ///
    /// The asymmetric half: a weighted instance is a MAX-SAT objective, which this flow has no
    /// representation for, and it is refused by name rather than solved as if the weights were not
    /// there.
    #[test]
    fn a_dimacs_instance_is_solved_and_dimacs_scores_it_zero() {
        let text = "p cnf 5 8\n1 -2 3 0\n-1 2 4 0\n2 3 -5 0\n-3 4 5 0\n1 -4 5 0\n-1 -2 -3 0\n\
                    3 -4 -5 0\n-2 4 -5 0\n";
        let cnf = Cnf::parse(text).unwrap();
        let f = Formula::from_cnf(&cnf).unwrap();
        assert_eq!(f.vars(), 5);
        assert_eq!(f.clauses().len(), 8);
        let out = solve(&f, Params::default(), 20_000, 11).unwrap();
        let a = out.solution().expect("a satisfiable five-variable instance");
        assert_eq!(cnf.unsatisfied_weight(a), 0.0, "dimacs disagrees: {a:?}");
        assert_eq!(cnf.cost(a), Some(0.0));

        let weighted = Cnf::parse("p wcnf 2 2 10\n10 1 2 0\n3 -1 -2 0\n").unwrap();
        assert_eq!(
            Formula::from_cnf(&weighted),
            Err(Error::SoftClause { clause: 1, weight: 3.0 }),
            "a soft clause must be named, not silently treated as hard"
        );
    }

    /// Every input the flow cannot represent is a typed error naming what arrived — no clause is
    /// repaired, deduplicated or dropped on the way in.
    #[test]
    fn the_formula_refuses_what_the_flow_cannot_represent() {
        let mut f = Formula::new(3);
        assert_eq!(f.push(&[]), Err(Error::EmptyClause { clause: 0 }));
        assert_eq!(f.push(&[1, 0]), Err(Error::ZeroLiteral { clause: 0 }));
        assert_eq!(f.push(&[1, 4]), Err(Error::UnknownVariable { lit: 4, vars: 3 }));
        assert_eq!(f.push(&[1, -1]), Err(Error::RepeatedVariable { var: 1, clause: 0 }));
        assert_eq!(f.push(&[2, 2]), Err(Error::RepeatedVariable { var: 2, clause: 0 }));
        assert!(f.clauses().is_empty(), "a refused clause must not be half-added");
        f.push(&[1, -2, 3]).unwrap();
        assert_eq!(f.clauses(), &[vec![1, -2, 3]]);
        // An error prints what was seen.
        assert!(format!("{}", Error::UnknownVariable { lit: 4, vars: 3 }).contains('4'));
        let empty = Formula::new(2);
        assert_eq!(
            Machine::new(&empty, Params::default(), 1).err(),
            Some(Error::EmptyClause { clause: 0 })
        );
    }

    /// A parameter outside its range is named, not clamped into one that happens to work.
    #[test]
    fn bad_parameters_are_named_rather_than_defaulted() {
        let f = sign_consistent(6, 10, 1);
        for (p, want) in [
            (Params { alpha: 0.0, ..Params::default() }, "alpha"),
            (Params { beta: -1.0, ..Params::default() }, "beta"),
            (Params { gamma: 1.0, ..Params::default() }, "gamma"),
            (Params { delta: 0.0, ..Params::default() }, "delta"),
            (Params { epsilon: f64::NAN, ..Params::default() }, "epsilon"),
            (Params { zeta: -0.5, ..Params::default() }, "zeta"),
            (Params { xl_max: 0.5, ..Params::default() }, "xl_max"),
            (Params { h0: 0.0, ..Params::default() }, "h0"),
            (Params { h_min: -1.0, ..Params::default() }, "h_min"),
            (Params { h_max: 1e-12, ..Params::default() }, "h_max"),
            (Params { atol: 0.0, ..Params::default() }, "atol"),
            (Params { rtol: f64::INFINITY, ..Params::default() }, "rtol"),
        ] {
            match solve(&f, p, 10, 1) {
                Err(Error::BadParam { name, .. }) => assert_eq!(name, want),
                other => panic!("{want} was accepted: {other:?}"),
            }
        }
        assert_eq!(Params::default().check(), Ok(()));
    }

    /// The documented extension for a clause of one literal: with no other literal to minimise
    /// over, the empty minimum is `2` and the drive is maximal, so the variable goes to its rail
    /// and the clause residual goes to zero. A silent `0` there — the other natural reading of an
    /// empty minimum — would leave the unit clause with no gradient term at all.
    #[test]
    fn a_unit_clause_drives_its_only_literal_to_the_rail() {
        let mut f = Formula::new(2);
        f.push(&[-1]).unwrap();
        f.push(&[1, 2]).unwrap();
        let mut mc = Machine::new(&f, Params::default(), 1).unwrap();
        for _ in 0..2000 {
            mc.step().unwrap();
        }
        assert!(mc.voltages()[0] < -0.999, "v1 = {}", mc.voltages()[0]);
        assert!(mc.residuals()[0] < 1e-15, "the unit clause is unsatisfied: {}", mc.residuals()[0]);
        assert_eq!(mc.readout(), vec![-1, 1]);
        assert_eq!(mc.unsatisfied(), 0);
    }

    /// An integrator that cannot meet its tolerance says so. Pin the step at a size the local error
    /// test cannot pass and the run ends [`Outcome::Stalled`] carrying [`Error::StepFloor`] — never
    /// an assignment, and never a quietly-accepted step. The same formula under the default
    /// tolerances is solved, so the refusal is the tolerance and not the flow.
    #[test]
    fn the_step_floor_is_a_typed_outcome_not_a_silent_acceptance() {
        let mut f = Formula::new(3);
        for c in unsat_eight().clauses().iter().skip(1) {
            f.push(c).unwrap();
        }
        let stiff =
            Params { h0: 10.0, h_min: 10.0, h_max: 10.0, atol: 1e-12, rtol: 1e-12, ..Params::default() };
        let out = solve(&f, stiff, 100, 1).unwrap();
        assert_eq!(out.solution(), None);
        match out {
            Outcome::Stalled { steps, why: Error::StepFloor { h, err }, .. } => {
                assert_eq!(steps, 0, "no step should have been accepted");
                assert_eq!(h, 10.0);
                assert!(err > 1.0, "a stall must report an error above tolerance, got {err}");
            }
            other => panic!("the integrator accepted a step it could not resolve: {other:?}"),
        }
        assert!(solve(&f, Params::default(), 20_000, 1).unwrap().solution().is_some());
    }

    /// Determinism by seed, which is the crate's headline: the same seed replays the trajectory
    /// bit for bit, and a different seed does not.
    #[test]
    fn the_same_seed_replays_the_trajectory_and_another_seed_does_not() {
        let f = random_3sat(20, 85, None, 9);
        let state = |seed: u64| {
            let mut mc = Machine::new(&f, Params::default(), seed).unwrap();
            for _ in 0..200 {
                mc.step().unwrap();
            }
            (mc.voltages().to_vec(), mc.long_term().to_vec(), mc.time())
        };
        assert_eq!(state(5), state(5));
        assert_ne!(state(5), state(6));
        assert_eq!(solve(&f, Params::default(), 5_000, 5), solve(&f, Params::default(), 5_000, 5));
    }
}
