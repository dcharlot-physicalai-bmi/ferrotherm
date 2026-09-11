//! Survey propagation and the 1RSB cavity method, over CNF factor graphs.
//!
//! Mézard, Parisi & Zecchina, *Analytic and algorithmic solution of random satisfiability
//! problems*, Science **297**:812 (2002). Braunstein, Mézard & Zecchina, *Survey propagation: an
//! algorithm for satisfiability*, Random Structures & Algorithms **27**:201 (2005). The warning
//! propagation limit and the cover interpretation follow Mézard & Montanari, *Information, Physics
//! and Computation* (2009), ch. 19 and 22.
//!
//! # Why a crate about this physics did not have this
//!
//! Every other message-passing routine here — [`crate::meanfield::belief_propagation`],
//! [`crate::trw`], [`crate::structured`], [`crate::exact`] — passes messages about a VARIABLE:
//! how likely is this spin to be up. That question has one answer only while the measure it
//! describes is a single lump. Above a clustering transition — near 3.86 clauses per variable for
//! random 3-SAT — the solutions of a random formula shatter into exponentially many clusters, each
//! internally connected and mutually distant. A message about a variable then averages over
//! clusters that disagree about it, and that average describes nothing that exists.
//!
//! Survey propagation passes messages about a CLUSTER. The one-step replica-symmetry-breaking
//! (1RSB) cavity method writes belief propagation over an auxiliary model whose variables are the
//! *fixed points of warning propagation* — the clusters themselves — so its messages are
//! probabilities over warnings rather than over spins. That is the algorithm this field is named
//! for, and before this module the string "survey propagation" appeared nowhere in the crate.
//!
//! # Warning propagation, which comes first
//!
//! A warning `u` from clause `a` to variable `i` is one bit: *every other variable of `a` is
//! already forced to a value that does not satisfy `a`, so `i` has to*. Writing `sat(a,j)` for the
//! value of `j` that satisfies `a`, and summing warnings into a cavity field,
//!
//! ```text
//!   H(j -> a)  =  sum over b != a containing j  of  sat(b,j) * u(b -> j)
//!   u(a -> i)  =  product over j in a, j != i  of  [ sat(a,j) * H(j -> a) < 0 ]
//! ```
//!
//! This is the zero-temperature, zero-energy limit of the whole construction, and it is honest
//! about its own weakness: all-zero warnings are a fixed point of *any* formula with no unit
//! clause, so warning propagation alone says nothing whatever about a random 3-SAT instance at
//! four clauses per variable. That is precisely why survey propagation exists.
//!
//! # The surveys
//!
//! `eta(a -> i)` in `[0, 1]` is the probability, over clusters, that clause `a` sends `i` a
//! warning. For a variable `j` in clause `a`, split the other clauses containing `j` into those
//! that AGREE with `a` about `j` (same satisfying value) and those that DISAGREE, and let `A` and
//! `D` be the products of `1 - eta` over those two sets — the probabilities that nothing in each
//! set warns. Then the three states of `j` in the cavity of `a` are
//!
//! ```text
//!   Pi(u) = (1 - D) * A     j is forced AWAY from satisfying a
//!   Pi(s) = (1 - A) * D     j is forced TOWARD satisfying a
//!   Pi(0) = A * D           nothing forces j at all
//!   Pi(u) + Pi(s) + Pi(0)  =  A + D - A*D
//! ```
//!
//! and the survey is the probability that every other variable of the clause is in state `u`:
//!
//! ```text
//!   eta(a -> i)  =  product over j in a, j != i  of  Pi(u) / (Pi(u) + Pi(s) + Pi(0))
//! ```
//!
//! The fourth state — warned in both directions at once — is a contradiction, carries no weight at
//! zero energy, and is why the three probabilities above sum to less than one before they are
//! normalised. When it is the ONLY state left the normalisation is `0/0`: that is
//! [`SurveyError::Contradiction`], reported rather than divided through.
//!
//! Restricting every `eta` to 0 or 1 recovers warning propagation, wherever no variable is warned
//! both ways — which is the case the zero-energy limit is derived under, and is not a coincidence
//! but the definition of the limit the surveys are taken in. Where a variable IS warned both ways
//! the two differ on purpose: warning propagation's field is a SUM and resolves by majority, while
//! the surveys have no state left to normalise and report the contradiction.
//!
//! # Conventions
//!
//! A state is `i8` spins, `+1` meaning TRUE, and spin `i` is DIMACS variable `i + 1` — the
//! convention [`crate::dimacs`] already uses, so [`crate::dimacs::Cnf::cost`] evaluates an
//! assignment this module produced with no translation. Clause weights are IGNORED: survey
//! propagation solves the decision problem, and a weighted instance is treated as the SAT instance
//! of all its clauses. For the MAX-SAT objective, compile with [`crate::dimacs::Cnf::to_hubo`] and
//! minimise.
//!
//! # What is deliberately not here
//!
//! No complexity (the cluster entropy) and no finite-`y` surveys. Both are a page of algebra away,
//! and the only test this crate could give the complexity — that it is exactly zero on a tree,
//! which has exactly one cover — passes just as well for several WRONG formulas. A number whose
//! only check cannot fail is the defect this crate spends its tests hunting, so the honest move is
//! to leave it out and say why.
//!
//! Nothing here claims a bound, so nothing here accumulates through [`crate::round`]: the surveys
//! are a fixed point of an approximation, and rounding one downward would not make it true. The
//! single rigorous claim the module makes — *this assignment satisfies this formula* — is settled
//! by evaluating the formula, not by summing anything.

use crate::dimacs::Cnf;
use crate::rng::Pcg;

/// One literal occurrence: which variable, and the value that satisfies the clause through it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Lit {
    /// Zero-based variable index; DIMACS calls this variable `var + 1`.
    pub var: u32,
    /// The spin of `var` that SATISFIES the clause: `+1` for a positive literal, `-1` for a
    /// negated one.
    ///
    /// Named for what it does rather than for what the file printed, because every equation in
    /// this module asks "is this variable pushed away from `sat`", and none of them asks which
    /// sign the literal was written with.
    pub sat: i8,
}

impl Lit {
    /// A literal on `var`, satisfied when that variable takes spin `sat`.
    ///
    /// # Panics
    ///
    /// If `sat` is neither `+1` nor `-1`, or `var` does not fit a `u32`.
    #[must_use]
    pub fn new(var: usize, sat: i8) -> Lit {
        assert!(sat == 1 || sat == -1, "a literal is satisfied by +1 or -1, not {sat}");
        assert!(u32::try_from(var).is_ok(), "variable {var} does not fit a u32");
        Lit { var: var as u32, sat }
    }

    /// The DIMACS form: `var + 1` for a positive literal, `-(var + 1)` for a negated one.
    #[must_use]
    pub fn dimacs(self) -> i32 {
        (self.var as i32 + 1) * i32::from(self.sat)
    }

    /// Whether `s` satisfies the owning clause through this literal.
    ///
    /// # Panics
    ///
    /// If `s` does not cover [`Lit::var`].
    #[must_use]
    pub fn satisfied(self, s: &[i8]) -> bool {
        s[self.var as usize] == self.sat
    }
}

/// Why the cavity equations could not be run, or could not be read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SurveyError {
    /// A clause with no literals. Nothing satisfies it, so the instance is unsatisfiable before a
    /// single message is passed — and there is no variable to pass one to.
    EmptyClause {
        /// Zero-based clause index.
        clause: usize,
    },
    /// A variable warned in BOTH directions: `A` and `D` are both zero, so the three cavity states
    /// have total weight zero and the normalisation is `0/0`.
    ///
    /// At zero energy this says no cover covers this variable's neighbourhood. On a tree the
    /// warnings are exact and this is a refutation; on a loopy factor graph they are an
    /// approximation and this is evidence, not a proof.
    Contradiction {
        /// Zero-based variable index; DIMACS variable `var + 1`.
        var: usize,
    },
}

impl core::fmt::Display for SurveyError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            SurveyError::EmptyClause { clause } => write!(
                f,
                "clause {clause} has no literals: nothing satisfies it, and the cavity equations \
                 have no variable to send a message to"
            ),
            SurveyError::Contradiction { var } => write!(
                f,
                "variable {} is warned in both directions, so the zero-energy cavity equations \
                 have no consistent state for it (a refutation on a tree, evidence on a loop)",
                var + 1
            ),
        }
    }
}

impl core::error::Error for SurveyError {}

/// A CNF instance in the two-sided flat form the cavity equations need.
///
/// Messages live on literal OCCURRENCES — one per clause-variable incidence, which is one edge of
/// the factor graph — so both directions index one array: `eta[k]` is the survey from the clause
/// owning occurrence `k` to the variable that occurrence names.
#[derive(Clone, Debug)]
pub struct FactorGraph {
    n: usize,
    lits: Vec<Lit>,
    cstart: Vec<usize>,
    owner: Vec<usize>,
    occ: Vec<usize>,
    vstart: Vec<usize>,
}

impl FactorGraph {
    /// Build from explicit clauses over `n` variables.
    ///
    /// # Errors
    ///
    /// [`SurveyError::EmptyClause`] if any clause has no literals.
    ///
    /// # Panics
    ///
    /// If a literal names a variable at or past `n`, or one clause names a variable twice. The
    /// second breaks the derivation and not merely the indexing: the cavity equations treat the
    /// other variables of a clause as independent, and a variable is not independent of itself.
    /// [`crate::dimacs::Cnf`] deduplicates literals and drops tautologies at parse time, so this
    /// cannot fire on a parsed file.
    pub fn from_clauses(n: usize, clauses: &[Vec<Lit>]) -> Result<FactorGraph, SurveyError> {
        let mut lits = Vec::new();
        let mut cstart = Vec::with_capacity(clauses.len() + 1);
        let mut owner = Vec::new();
        cstart.push(0);
        for (a, c) in clauses.iter().enumerate() {
            if c.is_empty() {
                return Err(SurveyError::EmptyClause { clause: a });
            }
            for (t, l) in c.iter().enumerate() {
                assert!((l.var as usize) < n, "clause {a} names variable {} of {n}", l.var);
                assert!(
                    !c[..t].iter().any(|o| o.var == l.var),
                    "clause {a} names variable {} twice",
                    l.var
                );
                lits.push(*l);
                owner.push(a);
            }
            cstart.push(lits.len());
        }
        let mut deg = vec![0usize; n];
        for l in &lits {
            deg[l.var as usize] += 1;
        }
        let mut vstart = vec![0usize; n + 1];
        for i in 0..n {
            vstart[i + 1] = vstart[i] + deg[i];
        }
        let mut cursor = vstart.clone();
        let mut occ = vec![0usize; lits.len()];
        for (k, l) in lits.iter().enumerate() {
            occ[cursor[l.var as usize]] = k;
            cursor[l.var as usize] += 1;
        }
        Ok(FactorGraph { n, lits, cstart, owner, occ, vstart })
    }

    /// Build from a parsed DIMACS instance. Weights and hardness are ignored — see the module
    /// note: this solves the decision problem.
    ///
    /// # Errors
    ///
    /// [`SurveyError::EmptyClause`] if the file carried a clause with no literals, which is
    /// unsatisfiable on its own.
    pub fn from_cnf(cnf: &Cnf) -> Result<FactorGraph, SurveyError> {
        let clauses: Vec<Vec<Lit>> = cnf
            .clauses
            .iter()
            .map(|c| {
                c.lits
                    .iter()
                    .map(|&l| Lit::new(l.unsigned_abs() as usize - 1, if l > 0 { 1 } else { -1 }))
                    .collect()
            })
            .collect();
        FactorGraph::from_clauses(cnf.vars, &clauses)
    }

    /// Variables.
    #[must_use]
    pub fn n(&self) -> usize {
        self.n
    }

    /// Clauses.
    #[must_use]
    pub fn clauses(&self) -> usize {
        self.cstart.len() - 1
    }

    /// Literal occurrences, which is the length every message array here must have.
    #[must_use]
    pub fn messages(&self) -> usize {
        self.lits.len()
    }

    /// The literals of clause `a`.
    #[must_use]
    pub fn clause(&self, a: usize) -> &[Lit] {
        &self.lits[self.cstart[a]..self.cstart[a + 1]]
    }

    /// The occurrence indices of clause `a` — the message slots it owns.
    #[must_use]
    pub fn clause_slots(&self, a: usize) -> core::ops::Range<usize> {
        self.cstart[a]..self.cstart[a + 1]
    }

    /// The literal at occurrence `k`.
    #[must_use]
    pub fn lit(&self, k: usize) -> Lit {
        self.lits[k]
    }

    /// The clause occurrence `k` belongs to.
    #[must_use]
    pub fn owner(&self, k: usize) -> usize {
        self.owner[k]
    }

    /// Every occurrence of variable `i`, in clause order.
    #[must_use]
    pub fn occurrences(&self, i: usize) -> &[usize] {
        &self.occ[self.vstart[i]..self.vstart[i + 1]]
    }

    /// Clauses `s` fails.
    ///
    /// # Panics
    ///
    /// If `s` is shorter than the variable count.
    #[must_use]
    pub fn unsatisfied(&self, s: &[i8]) -> usize {
        assert!(s.len() >= self.n, "a state of {} cannot cover {} variables", s.len(), self.n);
        (0..self.clauses()).filter(|&a| !self.clause(a).iter().any(|l| l.satisfied(s))).count()
    }

    /// Whether `s` satisfies every clause.
    ///
    /// # Panics
    ///
    /// If `s` is shorter than the variable count.
    #[must_use]
    pub fn satisfies(&self, s: &[i8]) -> bool {
        self.unsatisfied(s) == 0
    }

    /// The probability that the variable at occurrence `k` is forced AWAY from satisfying the
    /// clause that owns `k` — the `Pi(u) / (Pi(u) + Pi(s) + Pi(0))` of the module note.
    ///
    /// # Errors
    ///
    /// [`SurveyError::Contradiction`] when the three cavity states have total weight zero, i.e.
    /// the variable is certainly warned in both directions at once.
    ///
    /// # Panics
    ///
    /// If `eta` is not one survey per occurrence.
    pub fn forcing_probability(&self, eta: &[f64], k: usize) -> Result<f64, SurveyError> {
        assert_eq!(eta.len(), self.lits.len(), "one survey per literal occurrence");
        let lit = self.lits[k];
        // The probability that NOTHING in each set warns this variable: `agree` over the other
        // clauses satisfied by the same value of it, `disagree` over those satisfied by the other.
        let (mut agree, mut disagree) = (1.0f64, 1.0f64);
        for &kk in self.occurrences(lit.var as usize) {
            if kk == k {
                continue;
            }
            if self.lits[kk].sat == lit.sat {
                agree *= 1.0 - eta[kk];
            } else {
                disagree *= 1.0 - eta[kk];
            }
        }
        // Pi(u) + Pi(s) + Pi(0) = (1-D)A + (1-A)D + AD, which collapses to this.
        let total = agree + disagree - agree * disagree;
        // `!(total > 0.0)` rather than `total <= 0.0`: the difference is NaN, which this refuses
        // and the other would divide by.
        if !(total > 0.0) {
            return Err(SurveyError::Contradiction { var: lit.var as usize });
        }
        let forced_away = (1.0 - disagree) * agree;
        Ok(forced_away / total)
    }

    /// One survey update: `eta(a -> i)` for occurrence `k`, from the surveys `eta`.
    ///
    /// The whole algorithm in one product — over the OTHER variables of the clause, of the
    /// probability each is forced away from satisfying it. A clause of one literal has an empty
    /// product and therefore always warns, which is unit propagation falling out of the cavity
    /// equations on its own.
    ///
    /// Public because it is also the only honest way to check the update against a closed form:
    /// the caller supplies the incoming surveys, so a test does not have to reach a fixed point
    /// before it knows what the answer must be.
    ///
    /// # Errors
    ///
    /// [`SurveyError::Contradiction`], from [`FactorGraph::forcing_probability`].
    ///
    /// # Panics
    ///
    /// If `eta` is not one survey per occurrence.
    pub fn survey_update(&self, eta: &[f64], k: usize) -> Result<f64, SurveyError> {
        let a = self.owner[k];
        let mut p = 1.0f64;
        for kk in self.clause_slots(a) {
            if kk != k {
                p *= self.forcing_probability(eta, kk)?;
            }
        }
        Ok(p)
    }

    /// The three-state survey for variable `i`: frozen true, frozen false, or free across clusters.
    ///
    /// # Errors
    ///
    /// [`SurveyError::Contradiction`] if `i` is certainly warned both ways.
    ///
    /// # Panics
    ///
    /// If `eta` is not one survey per occurrence.
    pub fn bias(&self, eta: &[f64], i: usize) -> Result<Bias, SurveyError> {
        assert_eq!(eta.len(), self.lits.len(), "one survey per literal occurrence");
        let (mut pos, mut neg) = (1.0f64, 1.0f64);
        for &k in self.occurrences(i) {
            if self.lits[k].sat > 0 {
                pos *= 1.0 - eta[k];
            } else {
                neg *= 1.0 - eta[k];
            }
        }
        let total = pos + neg - pos * neg;
        if !(total > 0.0) {
            return Err(SurveyError::Contradiction { var: i });
        }
        Ok(Bias {
            positive: (1.0 - pos) * neg / total,
            negative: (1.0 - neg) * pos / total,
            free: pos * neg / total,
        })
    }
}

/// What the surveys say about one variable: the probabilities that a cluster freezes it true,
/// freezes it false, or leaves it free. The three sum to one.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bias {
    /// Probability a cluster freezes the variable TRUE.
    pub positive: f64,
    /// Probability a cluster freezes the variable FALSE.
    pub negative: f64,
    /// Probability no cluster freezes it — the joker state, free to take either value.
    pub free: f64,
}

impl Bias {
    /// `positive - negative`: which way to fix the variable, and how strongly.
    ///
    /// Decimation freezes the largest magnitude first, and that is the entire heuristic: the
    /// variable the clusters agree about most is the one whose value costs the fewest clusters.
    #[must_use]
    pub fn polarization(&self) -> f64 {
        self.positive - self.negative
    }
}

/// A warning-propagation fixed point, or the state the iteration stopped at.
#[derive(Clone, Debug)]
pub struct Warnings {
    /// One warning bit per literal occurrence: `true` where that clause forces that variable.
    pub u: Vec<bool>,
    /// Sweeps actually run.
    pub iterations: usize,
    /// Whether a whole sweep changed nothing, which for binary messages is convergence exactly.
    pub converged: bool,
}

impl Warnings {
    /// The field on variable `i`: warnings toward `+1` minus warnings toward `-1`.
    #[must_use]
    pub fn field(&self, fg: &FactorGraph, i: usize) -> i32 {
        fg.occurrences(i)
            .iter()
            .filter(|&&k| self.u[k])
            .map(|&k| i32::from(fg.lit(k).sat))
            .sum()
    }

    /// The forced value of every variable: `+1`, `-1`, or `0` where nothing forces it.
    ///
    /// On a tree factor graph this is exactly the BACKBONE — the variables taking the same value
    /// in every satisfying assignment — which is the oracle this module's headline test checks
    /// against by enumerating all `2^n` states.
    #[must_use]
    pub fn forced(&self, fg: &FactorGraph) -> Vec<i8> {
        (0..fg.n()).map(|i| self.field(fg, i).signum() as i8).collect()
    }
}

/// Warning propagation: the zero-energy limit, swept in place until nothing changes.
///
/// Started from all-zero warnings, which is a fixed point of any formula with no unit clause. That
/// is not a weakness of this implementation but the content of the method — see the module note.
///
/// # Errors
///
/// [`SurveyError::Contradiction`] if some variable ends up warned in both directions. Checked on
/// the final state rather than during the sweep, so a transient conflict on the way to a fixed
/// point is not reported as one.
pub fn warning_propagation(fg: &FactorGraph, iters: usize) -> Result<Warnings, SurveyError> {
    let mut u = vec![false; fg.messages()];
    let mut it = 0;
    let mut converged = false;
    while it < iters {
        let mut changed = false;
        for k in 0..u.len() {
            let a = fg.owner(k);
            let mut warn = true;
            for kk in fg.clause_slots(a) {
                if kk == k {
                    continue;
                }
                let lit = fg.lit(kk);
                // The cavity field on that variable, excluding the clause being updated.
                let mut h = 0i32;
                for &k3 in fg.occurrences(lit.var as usize) {
                    if k3 != kk && u[k3] {
                        h += i32::from(fg.lit(k3).sat);
                    }
                }
                // Forced away from satisfying the clause: the field points against `sat`.
                if i32::from(lit.sat) * h >= 0 {
                    warn = false;
                    break;
                }
            }
            if u[k] != warn {
                u[k] = warn;
                changed = true;
            }
        }
        it += 1;
        if !changed {
            converged = true;
            break;
        }
    }
    let w = Warnings { u, iterations: it, converged };
    for i in 0..fg.n() {
        let occ = fg.occurrences(i);
        let up = occ.iter().any(|&k| w.u[k] && fg.lit(k).sat > 0);
        let down = occ.iter().any(|&k| w.u[k] && fg.lit(k).sat < 0);
        if up && down {
            return Err(SurveyError::Contradiction { var: i });
        }
    }
    Ok(w)
}

/// How hard to push the cavity equations, and what to do when they stop saying anything.
#[derive(Clone, Debug, PartialEq)]
pub struct Params {
    /// Sweeps before [`survey_propagation`] gives up on a fixed point.
    pub iters: usize,
    /// Convergence tolerance on the largest single survey change in a sweep.
    pub tol: f64,
    /// Damping in `[0, 1)`, the weight kept from the previous survey.
    ///
    /// Zero by default, and that default is load-bearing: on a tree the fixed point is exactly
    /// binary and undamped sweeps REACH it exactly, in as many sweeps as the tree is deep. Damping
    /// makes the approach geometric instead, leaving the surveys merely near 0 and 1 — and the
    /// test comparing them to an enumerated backbone would then need a tolerance where the
    /// mathematics needs none.
    pub damping: f64,
    /// Fraction of the still-unfixed variables to freeze per decimation round, at least one.
    ///
    /// The published algorithm freezes a fixed fraction because one variable per survey run is
    /// `O(n)` survey runs; on small instances the two coincide, the ceiling being one either way.
    pub fix_fraction: f64,
    /// Largest polarization magnitude below which the surveys are treated as saying nothing, and
    /// the residual formula is handed to [`walksat`].
    ///
    /// All-zero surveys are a fixed point of every formula, and in the replica-symmetric phase it
    /// is the one the iteration finds. Below the clustering transition that is the CORRECT answer,
    /// and local search then solves the instance in linear time.
    pub trivial: f64,
    /// Flips the fallback [`walksat`] is allowed on the residual formula.
    pub flips: usize,
    /// Walksat noise: the probability of flipping a random variable of the chosen clause rather
    /// than the least damaging one. One half is Selman, Kautz & Cohen's original setting.
    pub noise: f64,
}

impl Default for Params {
    fn default() -> Self {
        Params {
            iters: 1000,
            tol: 1e-9,
            damping: 0.0,
            fix_fraction: 0.01,
            trivial: 0.01,
            flips: 200_000,
            noise: 0.5,
        }
    }
}

/// A survey-propagation fixed point, or the state the iteration stopped at.
#[derive(Clone, Debug)]
pub struct Surveys {
    /// One survey per literal occurrence: the probability that clause warns that variable.
    pub eta: Vec<f64>,
    /// Sweeps actually run.
    pub iterations: usize,
    /// Largest single survey change in the last sweep.
    pub residual: f64,
}

impl Surveys {
    /// Whether the last sweep moved every survey by less than `tol`.
    #[must_use]
    pub fn converged(&self, tol: f64) -> bool {
        self.residual < tol
    }

    /// The largest survey. Zero means the surveys are trivial: no cluster forces anything, and
    /// there is nothing for decimation to act on.
    #[must_use]
    pub fn max_survey(&self) -> f64 {
        self.eta.iter().copied().fold(0.0, f64::max)
    }

    /// The three-state survey of every variable.
    ///
    /// # Errors
    ///
    /// [`SurveyError::Contradiction`], from [`FactorGraph::bias`].
    pub fn biases(&self, fg: &FactorGraph) -> Result<Vec<Bias>, SurveyError> {
        (0..fg.n()).map(|i| fg.bias(&self.eta, i)).collect()
    }
}

/// Survey propagation from random surveys, swept in place until the largest change is under
/// `p.tol`.
///
/// The initial condition matters, and is what the seed is for: all-zero surveys are a fixed point
/// of every formula, so an iteration started there reports the replica-symmetric answer on every
/// instance — correct below the clustering transition, vacuous above it.
///
/// # Errors
///
/// [`SurveyError::Contradiction`] if the cavity states of some variable reach total weight zero.
///
/// # Panics
///
/// If `p.damping` is not in `[0, 1)`.
pub fn survey_propagation(fg: &FactorGraph, p: &Params, seed: u64) -> Result<Surveys, SurveyError> {
    assert!((0.0..1.0).contains(&p.damping), "damping must be in [0, 1)");
    let mut rng = Pcg::new(seed, 0x5B);
    let mut eta: Vec<f64> = (0..fg.messages()).map(|_| rng.f64()).collect();
    let mut residual = f64::INFINITY;
    let mut it = 0;
    while it < p.iters && residual > p.tol {
        residual = 0.0;
        for k in 0..eta.len() {
            let fresh = fg.survey_update(&eta, k)?;
            let next = p.damping * eta[k] + (1.0 - p.damping) * fresh;
            residual = residual.max((next - eta[k]).abs());
            eta[k] = next;
        }
        it += 1;
    }
    Ok(Surveys { eta, iterations: it, residual })
}

/// How a decimation run ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    /// Every clause satisfied, with the surveys and unit propagation doing all the fixing.
    Surveys,
    /// The surveys went trivial, or failed to converge, and local search finished the residual
    /// formula with every fix so far still in place.
    LocalSearch,
    /// The fixes led nowhere, were DROPPED, and local search solved the original formula from
    /// scratch. The surveys cost time on this instance and bought nothing.
    ///
    /// This exists so the report cannot flatter the method: an answer found after the guesses were
    /// thrown away is not an answer the guesses produced.
    Restart,
    /// UNIT PROPAGATION ALONE produced an empty clause, with nothing guessed before it. Unit
    /// resolution is sound, so this is a refutation of the formula itself. No assignment.
    Refuted,
    /// The cavity equations contradicted and neither local search pass recovered. No assignment,
    /// and — on a loopy factor graph — no refutation either: see [`SurveyError::Contradiction`].
    Contradiction,
    /// None of the above, and local search ran out of flips. No assignment, and nothing is claimed
    /// about the formula.
    Unsolved,
}

/// What a decimation run produced.
#[derive(Clone, Debug)]
pub struct Decimation {
    /// A satisfying assignment, VERIFIED against the original formula before being returned, or
    /// `None`. `+1` is true, and spin `i` is DIMACS variable `i + 1`.
    pub assignment: Option<Vec<i8>>,
    /// How the run ended.
    pub outcome: Outcome,
    /// Survey-propagation runs, one per decimation round.
    pub sp_calls: usize,
    /// Of those, the ones that hit the iteration cap without converging.
    pub unconverged: usize,
    /// Variables frozen because the surveys were polarised. On an [`Outcome::Restart`] these were
    /// dropped again, so the count is what the surveys ATTEMPTED, not what the answer used.
    pub survey_fixes: usize,
    /// Variables frozen by unit propagation on the simplified formula.
    pub unit_fixes: usize,
    /// Flips the local search spent, over every pass it made.
    pub flips: usize,
    /// The largest survey seen at any fixed point in the run.
    ///
    /// Zero means survey propagation never said anything on this instance, and every fix came from
    /// unit propagation or from local search — which is the truth about a formula below the
    /// clustering transition, and is worth reporting rather than hiding behind a solved flag.
    pub max_survey: f64,
}

/// Survey-propagation-guided decimation: run the surveys, freeze the most polarised variables,
/// simplify, repeat — and hand the residual to [`walksat`] when the surveys go trivial.
///
/// Unit propagation runs before every survey run. It is exact, since a clause of one literal
/// forces that literal, and it is also what the surveys themselves produce for a unit clause, so
/// doing it directly costs nothing and saves a fixed point per forced variable.
///
/// # A survey fix is a guess, so it has to be droppable
///
/// Freezing the most polarised variable is a heuristic and it is wrong sometimes — measurably so
/// at small `n`, where the loops the cavity method assumes away are short. On a 14-variable
/// instance of this crate's own test set the surveys fix one variable at polarization 0.94, the
/// residual is then unsatisfiable, and no seed recovers; plain [`walksat`] solves the same formula
/// from scratch in about a hundred flips. So when the fixes lead nowhere they are DROPPED and the
/// original formula is handed to local search on a full flip budget, reported as
/// [`Outcome::Restart`]. What that buys is worth stating plainly: **a failed decimation ends with
/// exactly the run local search alone would have made**, so the surveys can cost time and cannot
/// cost answers.
///
/// Only one verdict survives that, and it is the one unit propagation reaches with nothing guessed
/// before it: [`Outcome::Refuted`] is a sound refutation, and every other failure claims nothing
/// about the formula.
///
/// The returned assignment is checked against the ORIGINAL formula before it is handed back, so a
/// bug in the simplification cannot be reported as a solution.
///
/// # Panics
///
/// If an assignment that satisfied the residual formula fails the original one, which would mean
/// the simplification dropped a constraint.
#[must_use]
pub fn decimate(fg: &FactorGraph, p: &Params, seed: u64) -> Decimation {
    let mut out = Decimation {
        assignment: None,
        outcome: Outcome::Unsolved,
        sp_calls: 0,
        unconverged: 0,
        survey_fixes: 0,
        unit_fixes: 0,
        flips: 0,
        max_survey: 0.0,
    };
    let mut master = Pcg::new(seed, 0x5D);
    let mut fixed: Vec<Option<i8>> = vec![None; fg.n()];
    let mut clauses: Vec<Vec<Lit>> = (0..fg.clauses()).map(|a| fg.clause(a).to_vec()).collect();

    // Why the run stopped, if it stopped without an assignment.
    let failure = loop {
        match simplify(&mut clauses, &mut fixed) {
            None => {
                // An empty clause. That refutes the FORMULA only if nothing was guessed on the way
                // to it; otherwise it refutes this line of fixes and says nothing else.
                if out.survey_fixes == 0 {
                    out.outcome = Outcome::Refuted;
                    return out;
                }
                break Outcome::Unsolved;
            }
            Some(units) => out.unit_fixes += units,
        }
        if clauses.is_empty() {
            finish(fg, &fixed, None, Outcome::Surveys, &mut out);
            return out;
        }
        let Ok(residual) = FactorGraph::from_clauses(fg.n(), &clauses) else {
            // `simplify` already returns None on an empty clause, so this is unreachable today;
            // refusing is still the right answer if it ever stops being.
            break Outcome::Unsolved;
        };

        out.sp_calls += 1;
        let sp = survey_propagation(&residual, p, master.next_u64());
        let biases = match &sp {
            Ok(s) => {
                out.max_survey = out.max_survey.max(s.max_survey());
                if !s.converged(p.tol) {
                    out.unconverged += 1;
                }
                s.biases(&residual)
            }
            Err(e) => Err(*e),
        };
        let converged = sp.as_ref().is_ok_and(|s| s.converged(p.tol));

        let Ok(biases) = biases else {
            // A contradiction is not a proof on a loopy factor graph, so the residual still gets
            // its chance at local search before the run is called a failure.
            let (ws, flips) = walksat(&residual, p.flips, p.noise, master.next_u64());
            out.flips += flips;
            if let Some(ws) = ws {
                finish(fg, &fixed, Some(&ws), Outcome::LocalSearch, &mut out);
                return out;
            }
            break Outcome::Contradiction;
        };

        // Rank the unfixed variables by how strongly the clusters agree about them.
        let mut order: Vec<(usize, f64)> = (0..fg.n())
            .filter(|&i| fixed[i].is_none())
            .map(|i| (i, biases[i].polarization()))
            .collect();
        order.sort_by(|a, b| b.1.abs().total_cmp(&a.1.abs()));
        let strongest = order.first().map_or(0.0, |&(_, pol)| pol.abs());

        if !converged || strongest < p.trivial {
            let (ws, flips) = walksat(&residual, p.flips, p.noise, master.next_u64());
            out.flips += flips;
            if let Some(ws) = ws {
                finish(fg, &fixed, Some(&ws), Outcome::LocalSearch, &mut out);
                return out;
            }
            break Outcome::Unsolved;
        }

        let take = ((p.fix_fraction * order.len() as f64).ceil() as usize).clamp(1, order.len());
        for &(i, pol) in order.iter().take(take) {
            if pol.abs() >= p.trivial {
                fixed[i] = Some(if pol > 0.0 { 1 } else { -1 });
                out.survey_fixes += 1;
            }
        }
    };

    // The fixes were guesses and they led nowhere. Drop every one of them and give the whole
    // formula to local search, so that this can never do worse than the fallback alone.
    let (ws, flips) = walksat(fg, p.flips, p.noise, master.next_u64());
    out.flips += flips;
    match ws {
        Some(ws) => finish(fg, &vec![None; fg.n()], Some(&ws), Outcome::Restart, &mut out),
        None => out.outcome = failure,
    }
    out
}

/// Fill in the unfixed variables, verify against the original formula, and record the outcome.
fn finish(
    fg: &FactorGraph,
    fixed: &[Option<i8>],
    rest: Option<&[i8]>,
    outcome: Outcome,
    out: &mut Decimation,
) {
    // A variable still unfixed once the residual is empty appears in no surviving clause, so its
    // value cannot matter; `+1` is chosen so the answer is a function of the input alone.
    let s: Vec<i8> =
        (0..fg.n()).map(|i| fixed[i].unwrap_or_else(|| rest.map_or(1, |r| r[i]))).collect();
    assert!(
        fg.satisfies(&s),
        "decimation returned an assignment failing {} clauses of the ORIGINAL formula: the \
         simplification dropped a constraint",
        fg.unsatisfied(&s)
    );
    out.outcome = outcome;
    out.assignment = Some(s);
}

/// Remove satisfied clauses, drop falsified literals, and propagate units to a fixed point.
///
/// Returns how many variables unit propagation fixed, or `None` if a clause became empty.
fn simplify(clauses: &mut Vec<Vec<Lit>>, fixed: &mut [Option<i8>]) -> Option<usize> {
    let mut units = 0;
    loop {
        let mut progress = false;
        let mut kept: Vec<Vec<Lit>> = Vec::with_capacity(clauses.len());
        for c in clauses.iter() {
            if c.iter().any(|l| fixed[l.var as usize] == Some(l.sat)) {
                progress = true;
                continue;
            }
            let rest: Vec<Lit> =
                c.iter().copied().filter(|l| fixed[l.var as usize].is_none()).collect();
            if rest.len() != c.len() {
                progress = true;
            }
            match rest.len() {
                0 => return None,
                1 => {
                    fixed[rest[0].var as usize] = Some(rest[0].sat);
                    units += 1;
                    progress = true;
                }
                _ => kept.push(rest),
            }
        }
        *clauses = kept;
        if !progress {
            return Some(units);
        }
    }
}

/// Walksat: pick an unsatisfied clause, flip one of its variables, repeat.
///
/// Selman, Kautz & Cohen, *Noise strategies for improving local search*, AAAI 1994. With
/// probability `noise` the variable is drawn uniformly from the clause, otherwise it is the one
/// whose flip breaks the fewest currently-satisfied clauses. This is what the surveys hand the
/// residual formula to, and on its own it is also the reference that says how much the surveys
/// were worth.
///
/// Returns the assignment and the flips it took; `None`, with the flips spent, if it did not
/// finish. A variable occurring in no clause keeps whatever the initial coin gave it.
///
/// # Panics
///
/// If `noise` is not in `[0, 1]`.
#[must_use]
pub fn walksat(fg: &FactorGraph, flips: usize, noise: f64, seed: u64) -> (Option<Vec<i8>>, usize) {
    assert!((0.0..=1.0).contains(&noise), "noise must be in [0, 1]");
    let mut rng = Pcg::new(seed, 0x5A7);
    let mut s: Vec<i8> = (0..fg.n()).map(|_| rng.spin(0.5)).collect();
    let mut nsat: Vec<u32> = (0..fg.clauses())
        .map(|a| fg.clause(a).iter().filter(|l| l.satisfied(&s)).count() as u32)
        .collect();
    let mut unsat: Vec<usize> = (0..fg.clauses()).filter(|&a| nsat[a] == 0).collect();
    let mut at = vec![usize::MAX; fg.clauses()];
    for (pos, &a) in unsat.iter().enumerate() {
        at[a] = pos;
    }

    for step in 0..flips {
        if unsat.is_empty() {
            return (Some(s), step);
        }
        let pick = unsat[(rng.f64() * unsat.len() as f64) as usize % unsat.len()];
        let lits = fg.clause(pick);
        let v = if rng.f64() < noise {
            lits[(rng.f64() * lits.len() as f64) as usize % lits.len()].var as usize
        } else {
            let mut best = (u32::MAX, 0usize);
            for l in lits {
                let j = l.var as usize;
                let broken = fg
                    .occurrences(j)
                    .iter()
                    .filter(|&&k| nsat[fg.owner(k)] == 1 && fg.lit(k).satisfied(&s))
                    .count() as u32;
                if broken < best.0 {
                    best = (broken, j);
                }
            }
            best.1
        };
        s[v] = -s[v];
        for &k in fg.occurrences(v) {
            let a = fg.owner(k);
            if fg.lit(k).satisfied(&s) {
                nsat[a] += 1;
                if nsat[a] == 1 {
                    let pos = at[a];
                    let last = unsat.pop().expect("the clause is in the unsatisfied list");
                    if pos < unsat.len() {
                        unsat[pos] = last;
                        at[last] = pos;
                    }
                    at[a] = usize::MAX;
                }
            } else {
                nsat[a] -= 1;
                if nsat[a] == 0 {
                    at[a] = unsat.len();
                    unsat.push(a);
                }
            }
        }
    }
    if unsat.is_empty() { (Some(s), flips) } else { (None, flips) }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn coin(r: &mut Pcg) -> i8 {
        if r.f64() < 0.5 { 1 } else { -1 }
    }

    /// A random TREE factor graph, grown so that it cannot be anything else: every new clause
    /// attaches to exactly ONE variable already in the tree, and all its other variables are brand
    /// new. Unit clauses are frequent on purpose — without them nothing is ever forced, the
    /// backbone is empty, and a test comparing two empty answers proves nothing.
    fn random_tree_formula(target: usize, seed: u64) -> (usize, Vec<Vec<Lit>>) {
        let mut rng = Pcg::new(seed, 0x77);
        let mut used = 1usize;
        let mut clauses: Vec<Vec<Lit>> = Vec::new();
        while used < target && clauses.len() < 4 * target {
            let anchor = (rng.f64() * used as f64) as usize % used;
            if rng.f64() < 0.35 {
                clauses.push(vec![Lit::new(anchor, coin(&mut rng))]);
            } else {
                let k = 2 + (rng.next_u32() % 2) as usize;
                let mut c = vec![Lit::new(anchor, coin(&mut rng))];
                for _ in 1..k {
                    if used < target {
                        c.push(Lit::new(used, coin(&mut rng)));
                        used += 1;
                    }
                }
                clauses.push(c);
            }
        }
        (used, clauses)
    }

    /// Whether the factor graph is acyclic, by union-find over its `n + m` nodes.
    ///
    /// The premise of every tree claim below, checked rather than assumed: on a loopy factor graph
    /// the cavity equations are an approximation and none of the exactness holds.
    fn is_tree(n: usize, clauses: &[Vec<Lit>]) -> bool {
        fn find(p: &mut [usize], x: usize) -> usize {
            let mut r = x;
            while p[r] != r {
                r = p[r];
            }
            let mut c = x;
            while p[c] != c {
                let next = p[c];
                p[c] = r;
                c = next;
            }
            r
        }
        let mut p: Vec<usize> = (0..n + clauses.len()).collect();
        for (a, c) in clauses.iter().enumerate() {
            for l in c {
                let (x, y) = (find(&mut p, l.var as usize), find(&mut p, n + a));
                if x == y {
                    return false;
                }
                p[x] = y;
            }
        }
        true
    }

    /// The backbone by brute force over all `2^n` states: the value every satisfying assignment
    /// agrees on, `0` where they do not. `None` when the formula is unsatisfiable.
    fn enumerate_backbone(n: usize, clauses: &[Vec<Lit>]) -> Option<Vec<i8>> {
        let mut backbone: Option<Vec<i8>> = None;
        for m in 0u32..(1u32 << n) {
            let s: Vec<i8> = (0..n).map(|i| if m >> i & 1 == 1 { 1i8 } else { -1 }).collect();
            if !clauses.iter().all(|c| c.iter().any(|l| l.satisfied(&s))) {
                continue;
            }
            match &mut backbone {
                None => backbone = Some(s),
                Some(b) => {
                    for i in 0..n {
                        if b[i] != s[i] {
                            b[i] = 0;
                        }
                    }
                }
            }
        }
        backbone
    }

    /// Uniform random 3-SAT as DIMACS text, so the instance reaches the solver through
    /// [`Cnf::parse`] — the path a downloaded competition file takes.
    fn random_3sat(n: usize, m: usize, seed: u64) -> String {
        let mut rng = Pcg::new(seed, 0x3A);
        let mut out = format!("p cnf {n} {m}\n");
        for _ in 0..m {
            let mut vars: Vec<usize> = Vec::new();
            while vars.len() < 3 {
                let v = (rng.f64() * n as f64) as usize % n;
                if !vars.contains(&v) {
                    vars.push(v);
                }
            }
            for v in vars {
                out.push_str(&format!("{} ", (v as i32 + 1) * i32::from(coin(&mut rng))));
            }
            out.push_str("0\n");
        }
        out
    }

    fn clauses_of(fg: &FactorGraph) -> Vec<Vec<Lit>> {
        (0..fg.clauses()).map(|a| fg.clause(a).to_vec()).collect()
    }

    /// A complete DPLL: unit propagation, then branch on a variable of a shortest open clause.
    ///
    /// The ORACLE for instances too large to enumerate, and itself checked against enumeration by
    /// `the_dpll_oracle_agrees_with_exhaustive_enumeration`. `None` means the node budget ran out,
    /// and every caller asserts that it did not: an oracle allowed to say "I gave up" while being
    /// read as "unsatisfiable" is worse than no oracle.
    fn dpll(n: usize, clauses: &[Vec<Lit>], budget: u64) -> Option<Option<Vec<i8>>> {
        fn go(a: &mut [i8], clauses: &[Vec<Lit>], nodes: &mut u64, budget: u64) -> Option<bool> {
            if *nodes > budget {
                return None;
            }
            *nodes += 1;
            let mut trail: Vec<usize> = Vec::new();
            loop {
                let mut progress = false;
                for c in clauses {
                    let mut open: Option<Lit> = None;
                    let mut count = 0;
                    let mut sat = false;
                    for &l in c {
                        let v = a[l.var as usize];
                        if v == l.sat {
                            sat = true;
                            break;
                        } else if v == 0 {
                            open = Some(l);
                            count += 1;
                        }
                    }
                    if sat {
                        continue;
                    }
                    if count == 0 {
                        for &v in &trail {
                            a[v] = 0;
                        }
                        return Some(false);
                    }
                    if count == 1 {
                        let l = open.expect("counted one open literal");
                        a[l.var as usize] = l.sat;
                        trail.push(l.var as usize);
                        progress = true;
                    }
                }
                if !progress {
                    break;
                }
            }
            let mut pick = None;
            let mut best = usize::MAX;
            for c in clauses {
                let mut open: Vec<usize> = Vec::new();
                let mut sat = false;
                for &l in c {
                    let v = a[l.var as usize];
                    if v == l.sat {
                        sat = true;
                        break;
                    } else if v == 0 {
                        open.push(l.var as usize);
                    }
                }
                if !sat && !open.is_empty() && open.len() < best {
                    best = open.len();
                    pick = Some(open[0]);
                }
            }
            // Every clause satisfied: the trail STAYS, it is part of the answer.
            let Some(pick) = pick else {
                return Some(true);
            };
            for v in [1i8, -1] {
                a[pick] = v;
                match go(a, clauses, nodes, budget) {
                    None => return None,
                    Some(true) => return Some(true),
                    Some(false) => {}
                }
            }
            a[pick] = 0;
            for &v in &trail {
                a[v] = 0;
            }
            Some(false)
        }
        let mut a = vec![0i8; n];
        let mut nodes = 0u64;
        match go(&mut a, clauses, &mut nodes, budget) {
            None => None,
            Some(false) => Some(None),
            Some(true) => {
                for x in &mut a {
                    if *x == 0 {
                        *x = 1;
                    }
                }
                Some(Some(a))
            }
        }
    }

    /// THE FIXTURE MUST BITE. The generator has to produce actual trees, satisfiable AND
    /// unsatisfiable ones, and — the part that is easy to lose — formulas with a NON-EMPTY
    /// backbone. Every exactness claim below is about which variables are forced, so a fixture
    /// where nothing is ever forced would let a `forced()` returning all zeros pass.
    #[test]
    fn the_tree_fixtures_are_trees_with_backbones_to_find() {
        let (mut sat, mut unsat, mut forcing) = (0, 0, 0);
        for seed in 0..300u64 {
            let (n, clauses) = random_tree_formula(12, seed);
            assert!(is_tree(n, &clauses), "seed {seed} is not a tree factor graph");
            match enumerate_backbone(n, &clauses) {
                None => unsat += 1,
                Some(b) => {
                    sat += 1;
                    if b.iter().any(|&v| v != 0) {
                        forcing += 1;
                    }
                }
            }
        }
        assert!(sat >= 100 && unsat >= 30, "sat {sat}, unsat {unsat} of 300");
        assert!(forcing >= 100, "only {forcing} of the fixtures force anything");
    }

    /// ORACLE: exhaustive enumeration of all `2^n` assignments.
    ///
    /// On a tree factor graph warning propagation is exact, so the variables it calls forced must
    /// be precisely the BACKBONE — the variables every satisfying assignment agrees on — and an
    /// unsatisfiable tree must come back as a contradiction. Asserted as equality of the whole
    /// integer vector: on a tree there is nothing approximate to make room for.
    #[test]
    fn warning_propagation_is_the_backbone_by_exhaustive_enumeration_on_trees() {
        let (mut sat, mut unsat) = (0, 0);
        for seed in 0..300u64 {
            let (n, clauses) = random_tree_formula(12, seed);
            let fg = FactorGraph::from_clauses(n, &clauses).unwrap();
            let truth = enumerate_backbone(n, &clauses);
            match (truth, warning_propagation(&fg, 500)) {
                (Some(b), Ok(w)) => {
                    sat += 1;
                    assert!(w.converged, "seed {seed}: {} sweeps and no fixed point", w.iterations);
                    assert_eq!(w.forced(&fg), b, "seed {seed}: the forced set is not the backbone");
                }
                (None, Err(SurveyError::Contradiction { .. })) => unsat += 1,
                (None, Err(e)) => panic!("seed {seed}: unsatisfiable, but WP said {e}"),
                (Some(_), Err(e)) => panic!("seed {seed}: satisfiable, but WP contradicted: {e}"),
                (None, Ok(w)) => {
                    panic!("seed {seed}: unsatisfiable, but WP was content: {:?}", w.forced(&fg))
                }
            }
        }
        assert!(sat > 0 && unsat > 0, "both verdicts must occur: sat {sat}, unsat {unsat}");
    }

    /// ORACLE: exhaustive enumeration of all `2^n` assignments.
    ///
    /// A tree has exactly one cover, so at the fixed point the surveys are not merely near 0 and 1
    /// — they ARE 0 and 1, and each variable's three-state bias is a point mass on its backbone
    /// value. Both are asserted with `==`: the iteration reaches that fixed point by exact
    /// arithmetic on exact zeros and ones, and a tolerance here would hide the error it covers for.
    ///
    /// This is also where the `u` and `s` halves of the cavity state meet something outside the
    /// module. Swapping them leaves a perfectly plausible algorithm — it still converges, its
    /// surveys still lie in `[0, 1]`, and decimation still returns satisfying assignments through
    /// the local-search fallback — and it forces the wrong variables here.
    #[test]
    fn survey_propagation_is_the_backbone_by_exhaustive_enumeration_on_trees() {
        let p = Params::default();
        assert_eq!(p.damping, 0.0, "damping would make the tree fixed point only approximate");
        let (mut sat, mut unsat) = (0, 0);
        for seed in 0..300u64 {
            let (n, clauses) = random_tree_formula(12, seed);
            let fg = FactorGraph::from_clauses(n, &clauses).unwrap();
            let truth = enumerate_backbone(n, &clauses);
            // The contradiction can surface either during the sweeps or when the biases are read,
            // depending on where in the tree the two warnings meet; both are the same verdict.
            let sp = survey_propagation(&fg, &p, seed);
            let read = match &sp {
                Ok(s) => s.biases(&fg),
                Err(e) => Err(*e),
            };
            match (truth, read) {
                (Some(b), Ok(bias)) => {
                    sat += 1;
                    let s = sp.expect("a satisfiable tree has a fixed point");
                    assert_eq!(s.residual, 0.0, "seed {seed}: a tree fixed point is exact");
                    for (k, &e) in s.eta.iter().enumerate() {
                        assert!(e == 0.0 || e == 1.0, "seed {seed}: survey {k} is {e}, not a bit");
                    }
                    for i in 0..n {
                        let got = bias[i];
                        let want = match b[i] {
                            1 => (1.0, 0.0, 0.0),
                            -1 => (0.0, 1.0, 0.0),
                            _ => (0.0, 0.0, 1.0),
                        };
                        assert_eq!(
                            (got.positive, got.negative, got.free),
                            want,
                            "seed {seed} variable {i}: the backbone says {}",
                            b[i]
                        );
                    }
                }
                (None, Err(SurveyError::Contradiction { .. })) => unsat += 1,
                (None, Err(e)) => panic!("seed {seed}: unsatisfiable, but the surveys said {e}"),
                (Some(_), Err(e)) => panic!("seed {seed}: satisfiable, but SP contradicted: {e}"),
                (None, Ok(_)) => panic!("seed {seed}: unsatisfiable, but the surveys were content"),
            }
        }
        assert!(sat > 0 && unsat > 0, "both verdicts must occur: sat {sat}, unsat {unsat}");
    }

    /// ORACLE: the closed form, derived on paper from the update in the module note.
    ///
    /// Three cases where the surveys can be written down:
    ///
    ///  * a UNIT clause always warns — the product over the other variables is empty — so its
    ///    survey is 1 and its variable is frozen;
    ///  * an ISOLATED clause of `k >= 2` variables never warns: every other variable has
    ///    `A = D = 1`, so `Pi(u) = (1 - D) A = 0`, the survey is 0, and all `k` variables are free;
    ///  * a clause whose other `k - 1` variables each carry ONE disagreeing survey `x` has `A = 1`
    ///    and `D = 1 - x`, hence a forcing probability of exactly `x` per variable and a survey of
    ///    `x^(k-1)`.
    ///
    /// The third is asserted as an EQUALITY at dyadic `x`, where `1 - (1 - x)` round-trips exactly,
    /// and to four ulp at `x = 0.3`, where it does not — which is as tight as the arithmetic
    /// allows, not as tight as the mathematics does.
    #[test]
    fn an_isolated_clause_matches_the_closed_form_surveys() {
        let p = Params::default();

        let unit = FactorGraph::from_clauses(1, &[vec![Lit::new(0, 1)]]).unwrap();
        assert_eq!(unit.survey_update(&[0.0], 0).unwrap(), 1.0, "a unit clause always warns");
        let sp = survey_propagation(&unit, &p, 1).unwrap();
        assert_eq!(sp.eta, vec![1.0]);
        let b = unit.bias(&sp.eta, 0).unwrap();
        assert_eq!((b.positive, b.negative, b.free), (1.0, 0.0, 0.0));

        for k in 2..=5usize {
            let c: Vec<Lit> =
                (0..k).map(|i| Lit::new(i, if i % 2 == 0 { 1 } else { -1 })).collect();
            let fg = FactorGraph::from_clauses(k, &[c]).unwrap();
            let sp = survey_propagation(&fg, &p, k as u64).unwrap();
            assert!(sp.eta.iter().all(|&e| e == 0.0), "k={k}: an isolated clause warns {:?}", sp.eta);
            for i in 0..k {
                assert_eq!(fg.bias(&sp.eta, i).unwrap().free, 1.0, "k={k}, variable {i}");
            }
        }

        for k in 2..=5usize {
            // One wide clause over x0..x(k-1), and a unit clause DISAGREEING with it on each
            // variable but x0. The surveys of those unit clauses are then injected by hand, which
            // is what makes the answer a closed form rather than another fixed point.
            let mut clauses: Vec<Vec<Lit>> = vec![(0..k).map(|i| Lit::new(i, 1)).collect()];
            for j in 1..k {
                clauses.push(vec![Lit::new(j, -1)]);
            }
            let fg = FactorGraph::from_clauses(k, &clauses).unwrap();
            let target = fg.clause_slots(0).start; // the message from the wide clause to x0
            for x in [0.5f64, 0.25, 0.75, 0.125] {
                let mut eta = vec![0.0; fg.messages()];
                for j in 1..k {
                    eta[fg.clause_slots(j).start] = x;
                }
                assert_eq!(
                    fg.survey_update(&eta, target).unwrap(),
                    x.powi(k as i32 - 1),
                    "k={k}, x={x}"
                );
            }
            let x = 0.3f64;
            let mut eta = vec![0.0; fg.messages()];
            for j in 1..k {
                eta[fg.clause_slots(j).start] = x;
            }
            let want = x.powi(k as i32 - 1);
            let got = fg.survey_update(&eta, target).unwrap();
            assert!(
                (got - want).abs() <= 4.0 * f64::EPSILON * want,
                "k={k}: {got} is not {want} to four ulp"
            );
        }
    }

    /// The oracle for instances enumeration cannot reach, checked where enumeration can: the DPLL
    /// must agree with brute force verdict for verdict, and every assignment it claims must
    /// satisfy the formula.
    #[test]
    fn the_dpll_oracle_agrees_with_exhaustive_enumeration() {
        let (mut sat, mut unsat) = (0, 0);
        for seed in 0..80u64 {
            let n = 14;
            let cnf = Cnf::parse(&random_3sat(n, 4 * n, seed + 1000)).unwrap();
            let fg = FactorGraph::from_cnf(&cnf).unwrap();
            let clauses = clauses_of(&fg);
            let brute = enumerate_backbone(n, &clauses).is_some();
            let d = dpll(n, &clauses, 1 << 22).expect("the node budget must not run out");
            assert_eq!(brute, d.is_some(), "seed {seed}");
            match d {
                Some(s) => {
                    sat += 1;
                    assert!(fg.satisfies(&s), "seed {seed}: DPLL returned a non-solution");
                }
                None => unsat += 1,
            }
        }
        assert!(sat > 0 && unsat > 0, "both verdicts must occur: sat {sat}, unsat {unsat}");
    }

    /// ORACLE: a complete DPLL, on random 3-SAT at four clauses per variable and 60 variables —
    /// the size and density this module is posed at.
    ///
    /// Every instance the complete solver calls satisfiable must be SOLVED, with the assignment
    /// verified by [`Cnf::cost`] — the parser's own evaluator, which knows nothing about surveys.
    /// Every instance it refutes must come back with no assignment at all. At `n = 60` the
    /// satisfiability threshold is smeared wide enough that some of these instances really are
    /// unsatisfiable, and a test assuming otherwise would be testing the generator.
    ///
    /// The last two assertions keep the SURVEYS load-bearing here: on this set they must converge
    /// to something non-trivial and freeze variables on their own, or this test would pass on a
    /// module that had deleted survey propagation and kept only its fallback.
    #[test]
    fn decimation_agrees_with_a_complete_dpll_on_random_3sat() {
        let p = Params::default();
        let n = 60;
        let (mut sat, mut unsat, mut fixes) = (0, 0, 0);
        let mut strongest = 0.0f64;
        for seed in 0..12u64 {
            let cnf = Cnf::parse(&random_3sat(n, 4 * n, seed)).unwrap();
            let fg = FactorGraph::from_cnf(&cnf).unwrap();
            let truth = dpll(n, &clauses_of(&fg), 1 << 22).expect("the node budget must not run out");
            let d = decimate(&fg, &p, seed);
            fixes += d.survey_fixes;
            strongest = strongest.max(d.max_survey);
            if truth.is_some() {
                sat += 1;
                let s = d.assignment.unwrap_or_else(|| {
                    panic!("seed {seed}: satisfiable, decimation said {:?}", d.outcome)
                });
                assert_eq!(cnf.cost(&s), Some(0.0), "seed {seed}: the formula says otherwise");
            } else {
                unsat += 1;
                assert!(d.assignment.is_none(), "seed {seed}: solved an unsatisfiable formula");
            }
        }
        assert!(sat > 0 && unsat > 0, "both verdicts must occur: sat {sat}, unsat {unsat}");
        assert!(fixes > 0, "the surveys froze nothing on any of these instances");
        assert!(strongest > 0.5, "the surveys stayed trivial throughout: {strongest}");
    }

    /// ORACLE: exhaustive enumeration, on formulas small enough for it.
    ///
    /// The negative control of the pair, at scale: 60 random instances, of which a third are
    /// unsatisfiable, and not one of them may produce an assignment.
    #[test]
    fn decimation_agrees_with_exhaustive_enumeration_on_small_formulas() {
        let p = Params::default();
        let (mut sat, mut unsat) = (0, 0);
        for seed in 0..60u64 {
            let n = 14;
            let cnf = Cnf::parse(&random_3sat(n, 4 * n, seed + 1000)).unwrap();
            let fg = FactorGraph::from_cnf(&cnf).unwrap();
            let truth = enumerate_backbone(n, &clauses_of(&fg)).is_some();
            let d = decimate(&fg, &p, seed);
            if truth {
                sat += 1;
                let s = d.assignment.unwrap_or_else(|| {
                    panic!("seed {seed}: satisfiable, decimation said {:?}", d.outcome)
                });
                assert_eq!(cnf.cost(&s), Some(0.0), "seed {seed}");
            } else {
                unsat += 1;
                assert!(d.assignment.is_none(), "seed {seed}: solved an unsatisfiable formula");
            }
        }
        assert!(sat > 0 && unsat > 0, "both verdicts must occur: sat {sat}, unsat {unsat}");
    }

    /// THE NEGATIVE CONTROL, on cores whose unsatisfiability is not a matter of opinion: all eight
    /// clauses over three variables, a variable with its negation, and three pigeons in two holes.
    /// Enumeration confirms each is unsatisfiable, and decimation must report failure on every
    /// seed — never an assignment, and never [`Outcome::Surveys`] or [`Outcome::LocalSearch`],
    /// which would each mean it believed it had solved one.
    #[test]
    fn an_unsatisfiable_core_is_refuted_and_never_solved() {
        let mut all_eight = Vec::new();
        for m in 0..8u32 {
            all_eight
                .push((0..3).map(|i| Lit::new(i, if m >> i & 1 == 1 { 1 } else { -1 })).collect());
        }
        // Three pigeons, two holes: variable `2p + h` is pigeon p in hole h.
        let mut php: Vec<Vec<Lit>> = (0..3).map(|p| vec![Lit::new(2 * p, 1), Lit::new(2 * p + 1, 1)]).collect();
        for h in 0..2 {
            for a in 0..3 {
                for b in (a + 1)..3 {
                    php.push(vec![Lit::new(2 * a + h, -1), Lit::new(2 * b + h, -1)]);
                }
            }
        }
        let cases: Vec<(&str, usize, Vec<Vec<Lit>>)> = vec![
            ("every clause over three variables", 3, all_eight),
            ("a variable and its negation", 1, vec![vec![Lit::new(0, 1)], vec![Lit::new(0, -1)]]),
            ("three pigeons in two holes", 6, php),
        ];

        let p = Params::default();
        for (name, n, clauses) in cases {
            assert!(enumerate_backbone(n, &clauses).is_none(), "{name} is satisfiable after all");
            let fg = FactorGraph::from_clauses(n, &clauses).unwrap();
            for seed in 0..4u64 {
                let d = decimate(&fg, &p, seed);
                assert!(d.assignment.is_none(), "{name}, seed {seed}: decimation solved it");
                assert!(
                    matches!(
                        d.outcome,
                        Outcome::Refuted | Outcome::Contradiction | Outcome::Unsolved
                    ),
                    "{name}, seed {seed}: reported {:?}",
                    d.outcome
                );
            }
        }
        // Unit resolution alone closes the second core, and that verdict is a REFUTATION rather
        // than a shrug: nothing was guessed before the empty clause appeared.
        let unit = FactorGraph::from_clauses(1, &[vec![Lit::new(0, 1)], vec![Lit::new(0, -1)]])
            .unwrap();
        assert_eq!(decimate(&unit, &p, 3).outcome, Outcome::Refuted);
    }

    /// ORACLE: the published clustering threshold for random 3-SAT, 3.86 clauses per variable
    /// (Mézard, Mora & Zecchina, Phys. Rev. Lett. 94:197205, 2005).
    ///
    /// Below it the solutions form a single lump, the only fixed point the iteration finds is the
    /// trivial one, and every survey decays to zero — survey propagation correctly reports that
    /// there is nothing to report, and the fallback does the work. Above it the surveys freeze
    /// variables hard. A module that returned the same surveys on both sides of that line would be
    /// returning its initial condition.
    #[test]
    fn the_surveys_are_trivial_below_the_clustering_threshold_and_not_above() {
        let p = Params::default();
        let n = 150;
        for seed in 0..4u64 {
            for m in [2 * n, 7 * n / 2] {
                let cnf = Cnf::parse(&random_3sat(n, m, seed)).unwrap();
                let fg = FactorGraph::from_cnf(&cnf).unwrap();
                let sp = survey_propagation(&fg, &p, seed).unwrap();
                assert!(sp.converged(p.tol), "seed {seed}, {m} clauses: no fixed point");
                assert!(
                    sp.max_survey() < 1e-9,
                    "seed {seed}, {m} clauses: below the transition, yet warns {:e}",
                    sp.max_survey()
                );
            }
            let cnf = Cnf::parse(&random_3sat(n, 21 * n / 5, seed)).unwrap();
            let fg = FactorGraph::from_cnf(&cnf).unwrap();
            let sp = survey_propagation(&fg, &p, seed).unwrap();
            assert!(
                sp.max_survey() > 0.5,
                "seed {seed}: above the transition, yet warns only {}",
                sp.max_survey()
            );
        }
    }

    /// An empty clause is unsatisfiable on its own and has no variable to message, so it is
    /// refused rather than quietly ignored — and the refusal says which clause.
    #[test]
    fn an_empty_clause_is_refused_with_a_readable_error() {
        // A weighted clause that terminates immediately: the DIMACS dialects can express an empty
        // clause, and a bare `0` line cannot -- the parser reads that as a competition trailer.
        let cnf = Cnf::parse("p wcnf 2 2\n3 1 -2 0\n5 0\n").unwrap();
        assert_eq!(cnf.clauses.len(), 2, "the fixture must carry the empty clause into the model");
        assert_eq!(cnf.clauses[1].lits.len(), 0);
        let e = FactorGraph::from_cnf(&cnf).unwrap_err();
        assert_eq!(e, SurveyError::EmptyClause { clause: 1 });
        assert_eq!(
            FactorGraph::from_clauses(2, &[vec![Lit::new(0, 1)], Vec::new()]).unwrap_err(),
            SurveyError::EmptyClause { clause: 1 }
        );
        assert!(e.to_string().contains("no literals"), "{e}");
        let c = SurveyError::Contradiction { var: 4 };
        assert!(c.to_string().contains("variable 5"), "the message is one-based: {c}");
    }

    /// Local search reports success only for an assignment that really satisfies the formula, and
    /// a formula with no clauses is solved before the first flip.
    #[test]
    fn walksat_reports_only_verified_solutions() {
        let empty = FactorGraph::from_clauses(3, &[]).unwrap();
        let (a, flips) = walksat(&empty, 100, 0.5, 7);
        assert!(a.is_some() && flips == 0, "no clauses is solved before the first flip");

        let mut solved = 0;
        for seed in 0..20u64 {
            let n = 20;
            let cnf = Cnf::parse(&random_3sat(n, 4 * n, seed + 500)).unwrap();
            let fg = FactorGraph::from_cnf(&cnf).unwrap();
            let (a, _) = walksat(&fg, 20_000, 0.5, seed);
            if let Some(s) = a {
                assert_eq!(cnf.cost(&s), Some(0.0), "seed {seed}: reported a non-solution");
                solved += 1;
            }
        }
        assert!(solved > 10, "the fallback solved only {solved} of 20 easy instances");
    }
}
