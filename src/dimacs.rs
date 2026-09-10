//! DIMACS CNF and WCNF: the MAX-SAT benchmark corpus, as an Ising model.
//!
//! [`crate::gset`] brings in the max-cut benchmarks. This brings in the other standard family, and
//! it is a much larger one: the SAT and MAX-SAT competitions have published DIMACS instances for
//! thirty years, and every one of them is an energy model this crate can already solve.
//!
//! The translation is exact and needs no penalty. A clause is violated exactly when all of its
//! literals are false, so its cost is the indicator of that event:
//!
//! ```text
//!   violated(C) = Π_{l ∈ C} [literal l is false] = Π_{l ∈ C} (1 − σ_l s_i) / 2
//! ```
//!
//! with `σ_l = +1` for a positive literal and `−1` for a negated one, since literal `l` is false
//! exactly when `s_i = −σ_l`. Expanding that product over subsets gives a `k`-body spin polynomial,
//! which is what [`crate::hubo::Hubo`] holds — and [`crate::reduce`] lowers to pairwise hardware
//! from there. So a MAX-SAT instance becomes an Ising minimisation with no modelling choices left
//! to the caller and nothing to tune.
//!
//! # The constant is returned, not dropped
//!
//! The subset expansion has an empty-subset term, a constant `w / 2^k` per clause. `Factor::new`
//! refuses a term with no variables — correctly, since a constant is not a factor — so
//! [`Cnf::to_hubo`] returns it separately. Add it to the energy and you have the **unsatisfied
//! weight** exactly, at every assignment, which is what makes
//! `the_energy_is_the_unsatisfied_weight_at_every_assignment` a proof rather than a spot check.
//! Drop it and the optimum is unchanged but every reported number is off by a fixed amount, which
//! is the kind of error that survives a solver test and fails a comparison against a published
//! result.
//!
//! # Three dialects, one model
//!
//! `p cnf` gives every clause weight one. `p wcnf` puts a weight in front of each clause, and its
//! optional fourth header field is the `top` weight marking hard clauses. The MAX-SAT Evaluation
//! **2022 and later** format dropped the header line entirely: a soft clause carries its weight,
//! and a hard clause is marked with a leading `h` instead of a weight nobody can read off. All
//! three parse into the same [`Cnf`], and [`Cnf::format`] says which was read.
//!
//! Without a header the 2022 format has no declared variable count — [`Cnf::vars`] is the largest
//! index a literal names — and no declared clause count, so a truncated 2022 file cannot be
//! detected the way a truncated headered one is.
//!
//! # A hard clause is hard
//!
//! A hard clause is held with weight [`f64::INFINITY`], because that is what "hard" means: no
//! finite weight buys it. Compiling cannot put infinity in a coefficient, so [`Cnf::to_hubo`]
//! charges every hard clause [`Cnf::hard_penalty`] — one more than every soft clause in the
//! instance put together. Satisfying all the soft clauses therefore never pays for breaking one
//! hard clause, so the compiled model's minimiser satisfies every hard clause whenever anything
//! does. In the headered dialect this replaces the file's own `top`, which was only ever a large
//! number and is kept as [`Cnf::top`] for provenance.
//!
//! # Weighted and unweighted, and the clauses that are not really clauses
//!
//! Real files contain clauses that are not clauses. One holding both `v` and `−v` is a tautology and
//! is dropped, since a term that is never violated is a term with no energy. One repeating a literal
//! is deduplicated, because `Factor` refuses a repeated variable and `x ∨ x` is `x`. An empty clause
//! is violated by every assignment and becomes part of the constant.

use crate::hubo::Hubo;

/// The widest clause this will expand.
///
/// A clause of `k` literals becomes `2^k` spin monomials, so this is a real wall rather than a
/// tidiness rule. Twenty is already 1,048,576 terms from a single clause.
pub const MAX_CLAUSE: usize = 20;

/// Which DIMACS dialect a file was written in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    /// `p cnf <vars> <clauses>`: every clause weight one, none hard.
    Cnf,
    /// `p wcnf <vars> <clauses> [top]`: a weight per clause, `top` and above meaning hard.
    Wcnf,
    /// MAX-SAT Evaluation 2022 and later: no header at all, `h` marking a hard clause.
    Wcnf2022,
}

/// Why a DIMACS file could not be read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DimacsError {
    /// A `p` line that could not be read, a second one, one after the body has begun, or an
    /// input with no content at all.
    Header(String),
    /// A body line that is not a clause.
    Line {
        /// One-based line number.
        line: usize,
        /// The line itself.
        text: String,
    },
    /// A clause weight that is not a positive finite number.
    ///
    /// `nan` parses as an `f64` and would poison every energy it reached, and a non-positive
    /// weight is not a cost.
    Weight {
        /// One-based line number.
        line: usize,
        /// The token found where a weight belongs.
        got: String,
    },
    /// A literal naming a variable outside `1..=vars`.
    Variable {
        /// One-based line number.
        line: usize,
        /// The literal found.
        got: i64,
        /// Variables the header declared.
        vars: usize,
    },
    /// A literal too large for the `i32` a clause holds, which would otherwise truncate silently.
    Literal {
        /// One-based line number.
        line: usize,
        /// The literal found.
        got: i64,
    },
    /// The body has a different number of clauses than the header declared.
    ///
    /// A truncated download parses into a valid SMALLER instance whose optimum is not comparable
    /// with anyone else's, which is why this is refused rather than solved.
    Count {
        /// What the header said.
        declared: usize,
        /// What the body held.
        found: usize,
    },
    /// A clause too wide to expand into spin monomials.
    TooWide {
        /// One-based line number.
        line: usize,
        /// Literals in the clause.
        literals: usize,
        /// The limit.
        limit: usize,
    },
}

impl core::fmt::Display for DimacsError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            DimacsError::Header(h) => write!(
                f,
                "expected a `p cnf <vars> <clauses>` or `p wcnf <vars> <clauses> [top]` header, \
                 or a headerless MAX-SAT Evaluation 2022 body, found {h:?}"
            ),
            DimacsError::Line { line, text } => {
                write!(f, "line {line} is not a clause: {text:?}")
            }
            DimacsError::Weight { line, got } => write!(
                f,
                "line {line} carries {got:?} where a positive finite weight belongs (a hard clause \
                 is written `h` in the 2022 format, not as a large number)"
            ),
            DimacsError::Variable { line, got, vars } => write!(
                f,
                "line {line} names variable {} but the header declares {vars}",
                got.abs()
            ),
            DimacsError::Literal { line, got } => write!(
                f,
                "line {line} names literal {got}, which does not fit the i32 a clause holds"
            ),
            DimacsError::Count { declared, found } => write!(
                f,
                "the header declares {declared} clauses and the body has {found}. A truncated file \
                 parses into a valid SMALLER instance whose optimum is not comparable with \
                 anyone else's, so this is refused rather than solved"
            ),
            DimacsError::TooWide { line, literals, limit } => write!(
                f,
                "the clause on line {line} has {literals} literals and expands to 2^{literals} \
                 spin monomials, over the {limit} this will attempt"
            ),
        }
    }
}

impl core::error::Error for DimacsError {}

/// One clause: a weight and its literals, in DIMACS numbering (`+v` / `−v`, one-based).
#[derive(Clone, Debug, PartialEq)]
pub struct Clause {
    /// What violating it costs. One for every clause of a plain `cnf`, [`f64::INFINITY`] for a
    /// hard one — see [`Cnf::hard_penalty`] for the finite number that stands in when compiling.
    pub weight: f64,
    /// Literals, already deduplicated. Empty means a clause nothing satisfies.
    pub lits: Vec<i32>,
}

impl Clause {
    /// Whether this clause is hard, i.e. carries an infinite weight.
    #[must_use]
    pub fn is_hard(&self) -> bool {
        !self.weight.is_finite()
    }

    /// Whether `s` fails this clause: every literal false, and literal `l` is false at `s = −σ_l`.
    fn violated(&self, s: &[i8]) -> bool {
        self.lits.iter().all(|&l| {
            let want = if l > 0 { 1i8 } else { -1 };
            s[l.unsigned_abs() as usize - 1] != want
        })
    }
}

/// A parsed CNF or WCNF instance.
#[derive(Clone, Debug, PartialEq)]
pub struct Cnf {
    /// Variables the header declared, or the largest index a literal names in the 2022 format.
    /// Spin `i` is variable `i + 1`.
    pub vars: usize,
    /// The clauses, tautologies already dropped.
    pub clauses: Vec<Clause>,
    /// The `top` weight from a `wcnf` header, when it carried one. Provenance only: hard clauses
    /// are held at infinity whichever dialect named them.
    pub top: Option<f64>,
    /// The dialect this was read from.
    pub format: Format,
}

impl Cnf {
    /// Parse DIMACS `cnf`, `wcnf`, or the headerless MAX-SAT Evaluation 2022 format.
    ///
    /// The dialect is decided by the first line that is not a comment: a `p` line picks the
    /// headered forms, anything else picks the 2022 form, where `h` marks a hard clause.
    ///
    /// # Errors
    ///
    /// See [`DimacsError`]. A clause count disagreeing with the header is refused, because a
    /// truncated file is otherwise a perfectly valid smaller instance.
    pub fn parse(text: &str) -> Result<Cnf, DimacsError> {
        let mut vars = 0usize;
        let mut declared = 0usize;
        let mut top = None;
        let mut format: Option<Format> = None;
        let mut clauses = Vec::new();
        let mut found = 0usize;

        for (no, raw) in text.lines().enumerate() {
            let line = raw.trim();
            // `c` comments, `%` and `0` trailers that some competition files carry, and blanks.
            if line.is_empty() || line.starts_with('c') || line.starts_with('%') {
                continue;
            }
            if let Some(rest) = line.strip_prefix('p') {
                // A header once the dialect is settled is a second header, or a header after the
                // body began — either way the file is not one instance.
                if format.is_some() {
                    return Err(DimacsError::Header(line.to_string()));
                }
                let mut it = rest.split_whitespace();
                let kind = it.next().unwrap_or("");
                let f = match kind {
                    "cnf" => Format::Cnf,
                    "wcnf" => Format::Wcnf,
                    _ => return Err(DimacsError::Header(line.to_string())),
                };
                let (Some(v), Some(m)) = (it.next(), it.next()) else {
                    return Err(DimacsError::Header(line.to_string()));
                };
                let (Ok(v), Ok(m)) = (v.parse::<usize>(), m.parse::<usize>()) else {
                    return Err(DimacsError::Header(line.to_string()));
                };
                vars = v;
                declared = m;
                // An unreadable `top` would silently demote every hard clause to soft, so it is
                // refused rather than dropped.
                if let Some(t) = it.next() {
                    let Some(t) = t.parse::<f64>().ok().filter(|t| *t > 0.0 && t.is_finite()) else {
                        return Err(DimacsError::Header(line.to_string()));
                    };
                    top = Some(t);
                }
                format = Some(f);
                continue;
            }
            if line == "0" {
                continue;
            }
            let fmt = *format.get_or_insert(Format::Wcnf2022);

            let mut it = line.split_whitespace();
            let weight = match fmt {
                Format::Cnf => 1.0,
                Format::Wcnf | Format::Wcnf2022 => {
                    let Some(tok) = it.next() else {
                        return Err(DimacsError::Line { line: no + 1, text: line.to_string() });
                    };
                    if fmt == Format::Wcnf2022 && tok == "h" {
                        f64::INFINITY
                    } else {
                        let Ok(w) = tok.parse::<f64>() else {
                            return Err(DimacsError::Line { line: no + 1, text: line.to_string() });
                        };
                        // `!(w > 0.0)` rather than `w <= 0.0`: the difference is NaN, which this
                        // rejects and the other would accept.
                        if !(w > 0.0) || !w.is_finite() {
                            return Err(DimacsError::Weight {
                                line: no + 1,
                                got: tok.to_string(),
                            });
                        }
                        // The headered dialect marks hard clauses by weight; hold them the way the
                        // 2022 dialect states them.
                        if top.is_some_and(|t| w >= t) { f64::INFINITY } else { w }
                    }
                }
            };
            let mut lits: Vec<i32> = Vec::new();
            let mut terminated = false;
            for tok in it {
                let Ok(l) = tok.parse::<i64>() else {
                    return Err(DimacsError::Line { line: no + 1, text: line.to_string() });
                };
                if l == 0 {
                    terminated = true;
                    break;
                }
                // `l as i32` below truncates silently otherwise, and a header may declare more
                // variables than an i32 can index.
                if l > i64::from(i32::MAX) || l < i64::from(i32::MIN) + 1 {
                    return Err(DimacsError::Literal { line: no + 1, got: l });
                }
                let v = l.unsigned_abs() as usize;
                if fmt == Format::Wcnf2022 {
                    // No header, so the variable count is whatever the literals reach.
                    vars = vars.max(v);
                } else if v > vars {
                    return Err(DimacsError::Variable { line: no + 1, got: l, vars });
                }
                #[allow(clippy::cast_possible_truncation)]
                lits.push(l as i32);
            }
            if !terminated {
                return Err(DimacsError::Line { line: no + 1, text: line.to_string() });
            }
            if lits.len() > MAX_CLAUSE {
                return Err(DimacsError::TooWide {
                    line: no + 1,
                    literals: lits.len(),
                    limit: MAX_CLAUSE,
                });
            }
            found += 1;

            // `x ∨ x` is `x`, and `Factor` refuses a repeated variable; `x ∨ ¬x` is satisfied by
            // everything, so it has no energy and is not carried.
            lits.sort_unstable_by_key(|l| (l.unsigned_abs(), *l));
            lits.dedup();
            if lits.windows(2).any(|w| w[0] == -w[1]) {
                continue;
            }
            clauses.push(Clause { weight, lits });
        }

        let Some(format) = format else {
            return Err(DimacsError::Header(String::new()));
        };
        // The 2022 format declares no count, so there is nothing to disagree with.
        if format != Format::Wcnf2022 && found != declared {
            return Err(DimacsError::Count { declared, found });
        }
        Ok(Cnf { vars, clauses, top, format })
    }

    /// Total weight of the soft clauses, rounded UP, so no sum of them exceeds it.
    #[must_use]
    pub fn soft_weight(&self) -> f64 {
        let w: Vec<f64> =
            self.clauses.iter().filter(|c| !c.is_hard()).map(|c| c.weight).collect();
        crate::round::sum_up(&w)
    }

    /// The finite weight a hard clause is compiled at: strictly above [`Cnf::soft_weight`].
    ///
    /// Satisfying every soft clause in the instance therefore cannot pay for breaking one hard
    /// clause, which is what makes the compiled minimiser hard-feasible whenever anything is.
    #[must_use]
    pub fn hard_penalty(&self) -> f64 {
        let s = self.soft_weight();
        let p = s + 1.0;
        // Past 2^53 a unit is below the spacing, so step to the next representable instead.
        if p > s { p } else { s.next_up() }
    }

    /// The MAX-SAT objective at `s`: soft weight unsatisfied, or `None` if a hard clause is broken.
    ///
    /// This is the number an evaluation reports, and it is deliberately not a large weight: an
    /// infeasible assignment has no cost, it has no answer.
    ///
    /// # Panics
    ///
    /// If `s` is shorter than the variable count.
    #[must_use]
    pub fn cost(&self, s: &[i8]) -> Option<f64> {
        assert!(s.len() >= self.vars, "a state of {} cannot cover {} variables", s.len(), self.vars);
        let mut sum = 0.0;
        for c in &self.clauses {
            if c.violated(s) {
                if c.is_hard() {
                    return None;
                }
                sum += c.weight;
            }
        }
        Some(sum)
    }

    /// Total charge of the clauses `s` fails: its own weight for a soft clause,
    /// [`Cnf::hard_penalty`] for a hard one. Spin `i` is variable `i + 1`, `+1` true.
    ///
    /// # Panics
    ///
    /// If `s` is shorter than the declared variable count.
    #[must_use]
    pub fn unsatisfied_weight(&self, s: &[i8]) -> f64 {
        assert!(s.len() >= self.vars, "a state of {} cannot cover {} variables", s.len(), self.vars);
        let hard = self.hard_penalty();
        self.clauses
            .iter()
            .filter(|c| c.violated(s))
            .map(|c| if c.is_hard() { hard } else { c.weight })
            .sum()
    }

    /// The instance as a higher-order Ising model, with the constant it dropped.
    ///
    /// `hubo.energy(s) + offset` is [`Cnf::unsatisfied_weight`] at every assignment, so minimising
    /// the model maximises satisfied weight. Feed the result to [`crate::hubo::anneal`] directly, or
    /// through [`crate::reduce`] to reach pairwise hardware.
    ///
    /// # Panics
    ///
    /// Never on a `Cnf` from [`Cnf::parse`]: its clauses are deduplicated and within `MAX_CLAUSE`,
    /// which is what `Hubo::add` requires.
    #[must_use]
    pub fn to_hubo(&self) -> (Hubo, f64) {
        let mut h = Hubo::new(self.vars);
        let mut offset = 0.0;
        let hard = self.hard_penalty();
        for c in &self.clauses {
            let k = c.lits.len();
            let weight = if c.is_hard() { hard } else { c.weight };
            let scale = weight / f64::from(1u32 << k.min(31)) * if k > 31 { 0.0 } else { 1.0 };
            // Π (1 − σ_l s_i) / 2^k, expanded over subsets of the clause.
            for mask in 0u32..(1u32 << k) {
                let mut vars = Vec::with_capacity(k);
                let mut sign = 1.0f64;
                for (b, &l) in c.lits.iter().enumerate() {
                    if mask & (1 << b) != 0 {
                        vars.push(l.unsigned_abs() as usize - 1);
                        sign *= if l > 0 { 1.0 } else { -1.0 };
                    }
                }
                // (−1)^{|S|} from the product, times the literals' own signs.
                let term = scale * sign * if vars.len() % 2 == 0 { 1.0 } else { -1.0 };
                if vars.is_empty() {
                    offset += term;
                } else {
                    // Hubo's convention is E = −Σ w_T Π s, so the coefficient flips.
                    h.add(&vars, -term).expect("deduplicated, in range, and within MAX_CLAUSE");
                }
            }
        }
        (h, offset)
    }

    /// The instance written in the MAX-SAT Evaluation 2022 format.
    ///
    /// Hard clauses come back as `h`, soft ones as their weight. Variables no literal mentions are
    /// not representable in a headerless format and are lost, so this round-trips exactly when
    /// `vars` is a literal somewhere.
    #[must_use]
    pub fn to_wcnf2022(&self) -> String {
        let mut out = String::new();
        for c in &self.clauses {
            if c.is_hard() {
                out.push('h');
            } else {
                out.push_str(&c.weight.to_string());
            }
            for l in &c.lits {
                out.push(' ');
                out.push_str(&l.to_string());
            }
            out.push_str(" 0\n");
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every assignment, every clause: the model's energy IS the unsatisfied weight.
    ///
    /// The whole translation in one statement, checked exhaustively rather than at an optimum. An
    /// error that only moved the constant would leave every optimum right and every reported number
    /// wrong, which is what makes comparing against a published MAX-SAT result fail while a solver
    /// test passes.
    #[test]
    fn the_energy_is_the_unsatisfied_weight_at_every_assignment() {
        for (name, src) in [
            (
                "plain cnf, mixed signs",
                "c a comment\np cnf 5 4\n1 -2 3 0\n-1 4 0\n2 -3 -5 0\n5 0\n",
            ),
            (
                "weighted, with a unit and a wide clause",
                "p wcnf 5 4 100\n7 1 -2 0\n3 -1 -3 4 -5 0\n100 2 0\n0.5 -4 5 0\n",
            ),
            ("a single clause", "p cnf 3 1\n-1 -2 -3 0\n"),
            (
                "the 2022 format, hard and soft together",
                "c a comment\nh 1 -2 0\n7 -1 -3 4 -5 0\n3 2 0\nh -4 5 0\n",
            ),
        ] {
            let cnf = Cnf::parse(src).unwrap_or_else(|e| panic!("{name}: {e}"));
            let (h, off) = cnf.to_hubo();
            for m in 0u32..(1u32 << cnf.vars) {
                let s: Vec<i8> =
                    (0..cnf.vars).map(|i| if m >> i & 1 == 1 { 1i8 } else { -1 }).collect();
                let want = cnf.unsatisfied_weight(&s);
                let got = h.energy(&s) + off;
                assert!(
                    (got - want).abs() < 1e-9,
                    "{name}: state {s:?} fails {want} of weight, model says {got}"
                );
            }
        }
    }

    /// Minimising the model maximises satisfied weight, checked against brute force.
    #[test]
    fn the_models_optimum_is_the_max_sat_optimum() {
        let cnf = Cnf::parse(
            "p wcnf 6 6 50\n5 1 -2 0\n2 -1 3 0\n9 2 -3 4 0\n1 -4 5 0\n4 -5 6 0\n7 -6 -1 0\n",
        )
        .unwrap();
        let (h, off) = cnf.to_hubo();
        let (mut best_e, mut best_w) = (f64::INFINITY, f64::INFINITY);
        for m in 0u32..(1u32 << cnf.vars) {
            let s: Vec<i8> = (0..cnf.vars).map(|i| if m >> i & 1 == 1 { 1i8 } else { -1 }).collect();
            best_e = best_e.min(h.energy(&s) + off);
            best_w = best_w.min(cnf.unsatisfied_weight(&s));
        }
        assert!((best_e - best_w).abs() < 1e-9, "model optimum {best_e}, true optimum {best_w}");

        // And the crate's own higher-order solver finds it.
        let p = crate::hubo::Params { stages: 200, sweeps_per_stage: 20, ..Default::default() };
        let best = (0..8u64)
            .map(|seed| {
                let o = crate::hubo::anneal(&h, &p, seed);
                cnf.unsatisfied_weight(&o.state)
            })
            .fold(f64::INFINITY, f64::min);
        assert!((best - best_w).abs() < 1e-9, "anneal found {best}, optimum is {best_w}");
    }

    /// The two dialects are one model: a `p wcnf` file and its headerless 2022 twin agree
    /// clause for clause, penalty for penalty, and energy for energy at every assignment.
    ///
    /// `top` and `format` are provenance and necessarily differ — the whole point of the 2022
    /// format is that it has neither — so the comparison is of the model, not of the struct.
    #[test]
    fn both_wcnf_formats_parse_to_the_same_model() {
        let old = Cnf::parse("c old\np wcnf 4 4 20\n20 1 -2 0\n3 -1 3 0\n5 2 -3 4 0\n20 -4 0\n")
            .unwrap();
        let new = Cnf::parse("c new\nh 1 -2 0\n3 -1 3 0\n5 2 -3 4 0\nh -4 0\n").unwrap();

        assert_eq!(old.format, Format::Wcnf);
        assert_eq!(new.format, Format::Wcnf2022);
        assert_eq!(old.top, Some(20.0), "the header's own top is kept as provenance");
        assert_eq!(new.top, None, "and the 2022 format has none to keep");

        assert_eq!(new.vars, old.vars, "the largest literal is the variable count");
        assert_eq!(new.clauses, old.clauses, "hard is hard in both, at infinity");
        assert!(new.clauses[0].is_hard() && !new.clauses[1].is_hard());
        assert!((new.hard_penalty() - old.hard_penalty()).abs() < 1e-12);
        assert!((new.hard_penalty() - 9.0).abs() < 1e-12, "1 + (3 + 5), not the file's 20");

        let (ha, oa) = old.to_hubo();
        let (hb, ob) = new.to_hubo();
        for m in 0u32..(1u32 << old.vars) {
            let s: Vec<i8> = (0..old.vars).map(|i| if m >> i & 1 == 1 { 1i8 } else { -1 }).collect();
            assert!((ha.energy(&s) + oa - (hb.energy(&s) + ob)).abs() < 1e-9, "state {s:?}");
            assert_eq!(old.cost(&s), new.cost(&s), "state {s:?}");
        }
    }

    /// A hard clause cannot be bought: exhaustively, no assignment that breaks one is ever as
    /// cheap as one that breaks none.
    ///
    /// The soft clauses that a hard clause forces you to break are worth 1600 here, so any
    /// penalty read off a single clause — the largest weight, the mean, a fixed constant — sells
    /// both hard clauses. The oracle is enumeration, not a solver: every hard-feasible state is
    /// cheaper than every infeasible one, `max over feasible < min over infeasible`.
    #[test]
    fn a_hard_clause_is_unviolable_in_the_compiled_energy() {
        // Hard: x1, x2. Soft, and expensive: ¬x1, ¬x2, and a clause needing x3.
        let src = "h 1 0\nh 2 0\n900 -1 0\n700 -2 0\n11 3 0\n";
        let cnf = Cnf::parse(src).unwrap();
        assert_eq!(cnf.clauses.iter().filter(|c| c.is_hard()).count(), 2);
        assert!((cnf.soft_weight() - 1611.0).abs() < 1e-9);
        assert!((cnf.hard_penalty() - 1612.0).abs() < 1e-9);

        let (h, off) = cnf.to_hubo();
        let states: Vec<Vec<i8>> = (0u32..(1u32 << cnf.vars))
            .map(|m| (0..cnf.vars).map(|i| if m >> i & 1 == 1 { 1i8 } else { -1 }).collect())
            .collect();
        let (mut worst_ok, mut best_bad) = (f64::NEG_INFINITY, f64::INFINITY);
        for s in &states {
            let e = h.energy(s) + off;
            if cnf.cost(s).is_some() {
                worst_ok = worst_ok.max(e);
            } else {
                best_bad = best_bad.min(e);
            }
        }
        assert!(worst_ok.is_finite() && best_bad.is_finite(), "both classes must be non-empty");
        assert!(
            worst_ok < best_bad,
            "the worst hard-feasible state costs {worst_ok} and the best infeasible one \
             {best_bad}: a solver could pay for the hard clause"
        );

        // So the minimiser is hard-feasible, and it is the true MAX-SAT optimum.
        let best = states
            .iter()
            .min_by(|a, b| {
                (h.energy(a) + off).partial_cmp(&(h.energy(b) + off)).expect("finite energies")
            })
            .unwrap();
        assert_eq!(best, &vec![1i8, 1, 1], "x1 and x2 forced true, x3 free and worth 11");
        assert_eq!(cnf.cost(best), Some(1600.0), "¬x1 and ¬x2 are paid for, and that is the bill");
    }

    /// A `p wcnf` with no `top` field has no hard clauses, however large its weights are.
    #[test]
    fn without_a_top_field_nothing_is_hard() {
        let cnf = Cnf::parse("p wcnf 2 2\n1000000 1 0\n2 -2 0\n").unwrap();
        assert!(cnf.clauses.iter().all(|c| !c.is_hard()));
        assert_eq!(cnf.top, None);
        assert!((cnf.unsatisfied_weight(&[-1, -1]) - 1_000_000.0).abs() < 1e-9);
    }

    /// The 2022 writer round-trips through the 2022 parser, clause for clause.
    #[test]
    fn the_2022_format_round_trips_through_its_writer() {
        let src = "c source\nh 1 -2 0\n3.5 -1 3 0\n7 2 -3 4 0\nh -4 0\n5 0\n";
        let a = Cnf::parse(src).unwrap();
        let b = Cnf::parse(&a.to_wcnf2022()).unwrap();
        assert_eq!(a, b, "the written form parses back to the same instance");

        // And a headered instance survives the trip into the modern format.
        let old = Cnf::parse("p wcnf 3 3 40\n40 1 2 0\n6 -1 0\n2 -2 3 0\n").unwrap();
        let round = Cnf::parse(&old.to_wcnf2022()).unwrap();
        assert_eq!(round.clauses, old.clauses);
        assert_eq!(round.vars, old.vars);
        assert!((round.hard_penalty() - old.hard_penalty()).abs() < 1e-12);
    }

    /// Clauses that are not clauses: tautologies vanish, repeats collapse, empties are constant.
    #[test]
    fn degenerate_clauses_are_handled_rather_than_rejected() {
        // `1 -1` is a tautology; `2 2` is `2`; the header counts all three.
        let cnf = Cnf::parse("p cnf 3 3\n1 -1 0\n2 2 0\n-3 0\n").unwrap();
        assert_eq!(cnf.clauses.len(), 2, "the tautology should be dropped, not stored");
        assert_eq!(cnf.clauses[0].lits, vec![2], "a repeated literal collapses");

        let (h, off) = cnf.to_hubo();
        for m in 0u32..8u32 {
            let s: Vec<i8> = (0..3).map(|i| if m >> i & 1 == 1 { 1i8 } else { -1 }).collect();
            assert!((h.energy(&s) + off - cnf.unsatisfied_weight(&s)).abs() < 1e-9);
        }

        // An empty clause is violated by everything, so it is pure constant.
        let e = Cnf::parse("p wcnf 2 1 10\n3 0\n").unwrap();
        assert_eq!(e.clauses.len(), 1);
        assert!(e.clauses[0].lits.is_empty());
        let (h, off) = e.to_hubo();
        assert!((off - 3.0).abs() < 1e-9, "the empty clause's weight is the constant, got {off}");
        assert_eq!(h.terms(), 0, "and it introduces no term");

        // An empty HARD clause is an unsatisfiable instance: every assignment has no cost at all.
        let u = Cnf::parse("h 0\n4 1 0\n").unwrap();
        assert_eq!(u.cost(&[1]), None);
        assert_eq!(u.cost(&[-1]), None);
    }

    /// A truncated file is refused rather than solved as a smaller instance.
    #[test]
    fn a_clause_count_that_disagrees_with_the_header_is_refused() {
        match Cnf::parse("p cnf 3 4\n1 0\n-2 0\n3 0\n") {
            Err(DimacsError::Count { declared, found }) => assert_eq!((declared, found), (4, 3)),
            other => panic!("a truncated file must not parse: {other:?}"),
        }
    }

    /// The other refusals, each by name.
    #[test]
    fn malformed_input_is_refused_by_name() {
        assert!(matches!(Cnf::parse(""), Err(DimacsError::Header(_))), "nothing at all");
        assert!(matches!(Cnf::parse("c only a comment\n"), Err(DimacsError::Header(_))));
        assert!(
            matches!(Cnf::parse("p qbf 2 1\n1 0\n"), Err(DimacsError::Header(_))),
            "a format this does not read"
        );
        assert!(
            matches!(Cnf::parse("p cnf 2 1\n1 5 0\n"), Err(DimacsError::Variable { got: 5, .. })),
            "a literal past the declared variable count"
        );
        assert!(
            matches!(Cnf::parse("p cnf 2 1\n1 2\n"), Err(DimacsError::Line { line: 2, .. })),
            "a clause with no terminating zero"
        );
        let wide: String = (1..=MAX_CLAUSE + 1).map(|v| format!("{v} ")).collect();
        let src = format!("p cnf {} 1\n{wide}0\n", MAX_CLAUSE + 1);
        assert!(matches!(Cnf::parse(&src), Err(DimacsError::TooWide { .. })));
    }

    /// The 2022 format's own refusals, each by name.
    #[test]
    fn malformed_2022_input_is_refused_by_name() {
        assert!(
            matches!(Cnf::parse("h 1 0\n3 -2 0\np wcnf 2 2 9\n"), Err(DimacsError::Header(_))),
            "a header after the body has begun"
        );
        assert!(
            matches!(Cnf::parse("p wcnf 2 1 9\np wcnf 2 1 9\n"), Err(DimacsError::Header(_))),
            "a second header"
        );
        assert!(
            matches!(Cnf::parse("p wcnf 2 1 hard\n1 1 0\n"), Err(DimacsError::Header(_))),
            "an unreadable top would silently demote every hard clause"
        );
        assert!(
            matches!(Cnf::parse("p wcnf 2 1 9\nh 1 0\n"), Err(DimacsError::Line { line: 2, .. })),
            "`h` is the 2022 marker and means nothing in a headered file"
        );
        for (why, src) in [
            ("zero", "0.0 1 0\n"),
            ("negative", "-3 1 0\n"),
            ("nan, which parses as an f64 and poisons every energy", "nan 1 0\n"),
            ("infinite, which only `h` may be", "inf 1 0\n"),
        ] {
            assert!(
                matches!(Cnf::parse(src), Err(DimacsError::Weight { line: 1, .. })),
                "a weight that is {why} must be refused: {src:?} gave {:?}",
                Cnf::parse(src)
            );
        }
        assert!(
            matches!(Cnf::parse("h 1 -2\n"), Err(DimacsError::Line { line: 1, .. })),
            "a hard clause with no terminating zero"
        );
        assert!(
            matches!(Cnf::parse("hard 1 0\n"), Err(DimacsError::Line { line: 1, .. })),
            "`h` is a whole token, not a prefix"
        );
        assert!(
            matches!(Cnf::parse("h 4294967297 0\n"), Err(DimacsError::Literal { .. })),
            "a literal that would truncate into an i32 as variable 1"
        );
        // The same hole in the headered dialect, where a huge declared count used to admit it.
        assert!(
            matches!(
                Cnf::parse("p cnf 99999999999 1\n4294967297 0\n"),
                Err(DimacsError::Literal { .. })
            ),
            "a header cannot license a literal the clause type cannot hold"
        );
    }

    /// Comments, blank lines and the trailing `%` some competition files carry.
    #[test]
    fn the_decorations_real_files_carry_are_skipped() {
        let cnf = Cnf::parse("c header\nc more\n\np cnf 2 2\n1 0\n\n-2 0\n%\n0\n").unwrap();
        assert_eq!(cnf.vars, 2);
        assert_eq!(cnf.clauses.len(), 2);

        let modern = Cnf::parse("c header\n\nh 1 0\n\n3 -2 0\n%\n0\n").unwrap();
        assert_eq!(modern.vars, 2);
        assert_eq!(modern.clauses.len(), 2);
        assert_eq!(modern.format, Format::Wcnf2022);
    }
}
