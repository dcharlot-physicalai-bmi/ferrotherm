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
//! # Weighted and unweighted, and the clauses that are not really clauses
//!
//! `p cnf` gives every clause weight one. `p wcnf` puts a weight in front of each clause, and its
//! optional fourth header field is the `top` weight marking hard clauses; hard clauses are ordinary
//! clauses with a large weight here, because that is what they are — this does not pretend to
//! guarantee them.
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

/// Why a DIMACS file could not be read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DimacsError {
    /// No `p cnf` or `p wcnf` line, or one that could not be read.
    Header(String),
    /// A body line that is not a clause.
    Line {
        /// One-based line number.
        line: usize,
        /// The line itself.
        text: String,
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
                 found {h:?}"
            ),
            DimacsError::Line { line, text } => {
                write!(f, "line {line} is not a clause: {text:?}")
            }
            DimacsError::Variable { line, got, vars } => write!(
                f,
                "line {line} names variable {} but the header declares {vars}",
                got.abs()
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
    /// What violating it costs. One for every clause of a plain `cnf`.
    pub weight: f64,
    /// Literals, already deduplicated. Empty means a clause nothing satisfies.
    pub lits: Vec<i32>,
}

/// A parsed CNF or WCNF instance.
#[derive(Clone, Debug, PartialEq)]
pub struct Cnf {
    /// Variables the header declared. Spin `i` is variable `i + 1`.
    pub vars: usize,
    /// The clauses, tautologies already dropped.
    pub clauses: Vec<Clause>,
    /// The `top` weight from a `wcnf` header, when it carried one.
    pub top: Option<f64>,
}

impl Cnf {
    /// Parse DIMACS `cnf` or `wcnf`.
    ///
    /// # Errors
    ///
    /// See [`DimacsError`]. A clause count disagreeing with the header is refused, because a
    /// truncated file is otherwise a perfectly valid smaller instance.
    pub fn parse(text: &str) -> Result<Cnf, DimacsError> {
        let mut vars = 0usize;
        let mut declared = 0usize;
        let mut top = None;
        let mut weighted = false;
        let mut seen_header = false;
        let mut clauses = Vec::new();
        let mut found = 0usize;

        for (no, raw) in text.lines().enumerate() {
            let line = raw.trim();
            // `c` comments, `%` and `0` trailers that some competition files carry, and blanks.
            if line.is_empty() || line.starts_with('c') || line.starts_with('%') {
                continue;
            }
            if let Some(rest) = line.strip_prefix('p') {
                let mut it = rest.split_whitespace();
                let kind = it.next().unwrap_or("");
                weighted = match kind {
                    "cnf" => false,
                    "wcnf" => true,
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
                top = it.next().and_then(|t| t.parse::<f64>().ok());
                seen_header = true;
                continue;
            }
            if !seen_header {
                return Err(DimacsError::Header(line.to_string()));
            }
            if line == "0" {
                continue;
            }

            let mut it = line.split_whitespace();
            let weight = if weighted {
                let Some(Ok(w)) = it.next().map(str::parse::<f64>) else {
                    return Err(DimacsError::Line { line: no + 1, text: line.to_string() });
                };
                w
            } else {
                1.0
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
                if l.unsigned_abs() as usize > vars {
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

        if !seen_header {
            return Err(DimacsError::Header(String::new()));
        }
        if found != declared {
            return Err(DimacsError::Count { declared, found });
        }
        Ok(Cnf { vars, clauses, top })
    }

    /// Total weight of the clauses `s` fails to satisfy. Spin `i` is variable `i + 1`, `+1` true.
    ///
    /// # Panics
    ///
    /// If `s` is shorter than the declared variable count.
    #[must_use]
    pub fn unsatisfied_weight(&self, s: &[i8]) -> f64 {
        assert!(s.len() >= self.vars, "a state of {} cannot cover {} variables", s.len(), self.vars);
        self.clauses
            .iter()
            .filter(|c| {
                // Violated when EVERY literal is false, and literal `l` is false at `s = -sign(l)`.
                c.lits.iter().all(|&l| {
                    let want = if l > 0 { 1i8 } else { -1 };
                    s[l.unsigned_abs() as usize - 1] != want
                })
            })
            .map(|c| c.weight)
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
        for c in &self.clauses {
            let k = c.lits.len();
            let scale = c.weight / f64::from(1u32 << k.min(31)) * if k > 31 { 0.0 } else { 1.0 };
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
        assert!(matches!(Cnf::parse("1 2 0\n"), Err(DimacsError::Header(_))), "no header at all");
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

    /// Comments, blank lines and the trailing `%` some competition files carry.
    #[test]
    fn the_decorations_real_files_carry_are_skipped() {
        let cnf = Cnf::parse("c header\nc more\n\np cnf 2 2\n1 0\n\n-2 0\n%\n0\n").unwrap();
        assert_eq!(cnf.vars, 2);
        assert_eq!(cnf.clauses.len(), 2);
    }
}
