//! Read a problem written in LP format.
//!
//! LP is how optimisation problems are exchanged. Every solver worth the name reads it — CPLEX,
//! Gurobi, HiGHS, SCIP, and `dimod.lp.load` on the annealing side — so a stack that cannot is a
//! stack you have to hand-translate into. That was this one.
//!
//! ```text
//! Maximize
//!   obj: 5 mon + 4 tue + 3 wed
//! Subject To
//!   c1: mon + tue + wed <= 2
//!   c2: mon - tue = 0
//! Binary
//!   mon tue wed
//! End
//! ```
//!
//! # What is supported, and what is refused
//!
//! **Binary and bounded-integer variables**, linear objectives, and linear constraints with `<=`,
//! `>=` or `=`. That is the subset an Ising machine can actually run, and it is roughly what
//! `dimod` accepts too.
//!
//! Everything else is **refused by name with its line number** rather than approximated: a
//! continuous variable has no encoding here, a quadratic term in an LP file means something this
//! parser does not read, a range constraint (`1 <= x + y <= 3`) is two constraints written as one,
//! and a one-sided `x >= 3` has no upper end to compile into a finite spin domain. Silently
//! dropping any of them would return a confident answer to a different problem, and the whole point
//! of reading someone else's file is that you did not write it and cannot check it by eye.

use crate::model::{Expr, Lit, Model, Sense, Var};
use std::collections::BTreeMap;

/// Why a file could not be read.
#[derive(Clone, Debug, PartialEq)]
pub struct LpError {
    pub line: usize,
    pub message: String,
}

impl core::fmt::Display for LpError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "line {}: {}", self.line, self.message)
    }
}

fn err<T>(line: usize, message: impl Into<String>) -> Result<T, LpError> {
    Err(LpError { line, message: message.into() })
}

#[derive(Clone, Copy, PartialEq)]
enum Section {
    None,
    Objective(Sense),
    Constraints,
    Bounds,
    Binary,
    General,
}

/// A term `coefficient · name`, or a bare constant.
struct Atom {
    coeff: f64,
    name: Option<String>,
}

/// Parse an LP file into a [`Model`].
///
/// Declared variables become binaries, or integers over their declared bounds. Inequalities become
/// `at_most`/`at_least` over the terms; equalities become a cardinality when the coefficients allow
/// it and are refused otherwise, because a weighted equality is not a counting constraint and
/// pretending it is would change the problem.
pub fn parse(src: &str) -> Result<Model, LpError> {
    let mut section = Section::None;
    let mut objective: Vec<Atom> = Vec::new();
    let mut sense = Sense::Minimize;
    let mut constraints: Vec<(usize, String, Vec<Atom>, String, f64)> = Vec::new();
    let mut binaries: Vec<String> = Vec::new();
    let mut generals: Vec<String> = Vec::new();
    let mut bounds: BTreeMap<String, (i64, i64)> = BTreeMap::new();

    for (n, raw) in src.lines().enumerate() {
        let line = n + 1;
        let text = raw.split('\\').next().unwrap_or("").trim();
        let text = text.split('#').next().unwrap_or("").trim();
        if text.is_empty() {
            continue;
        }
        let lower = text.to_ascii_lowercase();

        // Section headers. LP is generous about spelling; these are the forms in the wild.
        let header = match lower.as_str() {
            "maximize" | "maximise" | "max" => Some(Section::Objective(Sense::Maximize)),
            "minimize" | "minimise" | "min" => Some(Section::Objective(Sense::Minimize)),
            "subject to" | "such that" | "st" | "s.t." | "subjectto" => Some(Section::Constraints),
            "bounds" | "bound" => Some(Section::Bounds),
            "binary" | "binaries" | "bin" => Some(Section::Binary),
            "general" | "generals" | "gen" | "integer" | "integers" => Some(Section::General),
            "end" => break,
            _ => None,
        };
        if let Some(h) = header {
            if let Section::Objective(s) = h {
                sense = s;
            }
            section = h;
            continue;
        }

        match section {
            Section::None => {
                return err(line, format!("{text:?} appears before any section header"))
            }
            Section::Objective(_) => {
                let body = strip_label(text);
                objective.extend(atoms(body, line)?);
            }
            Section::Constraints => {
                let (label, body) = split_label(text);
                let (lhs, rel, rhs) = relation(body, line)?;
                constraints.push((line, label, atoms(lhs, line)?, rel, rhs));
            }
            Section::Binary => binaries.extend(text.split_whitespace().map(str::to_string)),
            Section::General => generals.extend(text.split_whitespace().map(str::to_string)),
            Section::Bounds => {
                let (lo, name, hi) = bound(text, line)?;
                bounds.insert(name, (lo, hi));
            }
        }
    }

    if binaries.is_empty() && generals.is_empty() {
        return err(
            0,
            "no Binary or General section: every variable would be continuous, and a continuous \
             variable has no encoding on a spin machine. Declare the variables, or use a solver \
             that speaks linear programming",
        );
    }

    let mut m = Model::new();
    let mut vars: BTreeMap<String, Var> = BTreeMap::new();
    for name in &binaries {
        // A `Bounds` line on a Binary variable was parsed into `bounds` and then never read, because
        // this path does not consult the map. `x <= 0` means x is fixed OFF, and the model solved as
        // though x were free -- a confident answer to a different problem, from a file the caller
        // did not write and cannot check by eye. Anything other than the implied 0..1 is refused by
        // name rather than silently dropped.
        if let Some((lo, hi)) = bounds.get(name) {
            if (*lo, *hi) != (0, 1) {
                return err(
                    0,
                    format!(
                        "'{name}' is declared Binary and also has Bounds {lo}..{hi}. A \
                         bound on a binary either says nothing or fixes it, and this \
                         reader will not guess which you meant: drop the bound, or move \
                         '{name}' to General"
                    ),
                );
            }
        }
        vars.insert(name.clone(), m.binary(name));
    }
    for name in &generals {
        let (lo, hi) = bounds.get(name).copied().unwrap_or((0, 1));
        if hi <= lo {
            return err(0, format!("'{name}' is a general variable with bounds {lo}..{hi}"));
        }
        // Refused HERE, before anything walks the domain.
        //
        // The objective loop below emits one term per value in `lo..=hi`, so a range spanning most
        // of `i64` is not merely a large model -- it is an unbounded loop allocating on every
        // iteration, and it runs during PARSING, before `compile()` and therefore before the
        // `DomainTooLarge` refusal there can see it. Found by the parser fuzzer as a 1 GB
        // allocation from a six-line LP file.
        //
        // The ceiling is the graph's own spin index type, matching `CompileError::DomainTooLarge`,
        // so the two refusals agree on what is representable.
        let span = (hi as i128) - (lo as i128) + 1;
        if span > u32::MAX as i128 {
            return err(
                0,
                format!(
                    "'{name}' is a general variable over {lo}..{hi}, which is {span} values. \
                     Spin indices are u32, so at most {} can be addressed -- and reading this file \
                     would emit one objective term per value before anything checked. Narrow the \
                     bounds",
                    u32::MAX
                ),
            );
        }
        vars.insert(name.clone(), m.integer(name, lo, hi));
    }

    let find = |name: &str, line: usize| -> Result<Var, LpError> {
        vars.get(name).copied().ok_or_else(|| LpError {
            line,
            message: format!(
                "'{name}' is used but never declared Binary or General. An undeclared LP \
                 variable is continuous, which has no encoding here"
            ),
        })
    };
    let is_general = |name: &str| generals.iter().any(|g| g == name);

    // Objective. `max t` over an integer means maximise its VALUE, which in an indicator model is
    // the sum over its domain weighted by each value. Treating it as `[t == 1]` -- which this first
    // did -- asks for something else entirely, and for a variable over 10..=20 asks for a value it
    // cannot take.
    let mut e = Expr::zero();
    for a in &objective {
        let Some(nm) = &a.name else { continue };  // a constant shifts every state equally
        let v = find(nm, 0)?;
        if is_general(nm) {
            let (lo, hi) = bounds.get(nm).copied().unwrap_or((0, 1));
            for value in lo..=hi {
                e = e.plus(Expr::lit(a.coeff * value as f64, Lit::Is(v, value)));
            }
        } else {
            e = e.plus(Expr::lit(a.coeff, Lit::Is(v, 1)));
        }
    }
    m.objective(sense, e);

    // Constraints
    for (line, label, lhs, rel, rhs) in constraints {
        let mut lits = Vec::new();
        let mut constant = 0.0;
        let mut unit = true;
        for a in &lhs {
            match &a.name {
                None => constant += a.coeff,
                Some(nm) => {
                    if a.coeff != 1.0 {
                        unit = false;
                    }
                    // A linear constraint over an integer is a real thing and it is not a counting
                    // constraint: `t <= 5` bounds a value, where `at_most` counts how many
                    // literals hold. Mapping one onto the other would silently solve a different
                    // problem, so it is refused with the distinction spelled out.
                    if is_general(nm) {
                        return err(
                            line,
                            format!(
                                "constraint '{label}' bounds the general variable '{nm}'. That is \
                                 an arithmetic constraint on a value, not a count of how many \
                                 things hold, and this reads only the second. Narrow the \
                                 variable's Bounds instead, which says the same thing where it \
                                 belongs"
                            ),
                        );
                    }
                    lits.push(Lit::Is(find(nm, line)?, 1));
                }
            }
        }
        let target = rhs - constant;
        if !unit {
            return err(
                line,
                format!(
                    "constraint '{label}' has coefficients other than 1. A weighted linear \
                     constraint is not a counting constraint, and treating it as one would change \
                     the problem. Rewrite it as a counting constraint, or add it to the objective"
                ),
            );
        }
        if target < 0.0 || target.fract() != 0.0 {
            return err(
                line,
                format!("constraint '{label}' compares against {target}, which is not a count"),
            );
        }
        let k = target as usize;
        if k > lits.len() {
            return err(
                line,
                format!(
                    "constraint '{label}' asks for {k} of {} terms, which nothing can satisfy",
                    lits.len()
                ),
            );
        }
        match rel.as_str() {
            "<=" => { m.at_most(lits, k); }
            ">=" => { m.at_least(lits, k); }
            "=" => { m.cardinality(lits, k); }
            other => return err(line, format!("unknown relation {other:?}")),
        }
    }

    Ok(m)
}

fn strip_label(s: &str) -> &str {
    split_label(s).1
}

/// `c1: x + y` becomes `("c1", "x + y")`. A colon inside a number is not a label.
fn split_label(s: &str) -> (String, &str) {
    match s.find(':') {
        Some(i) if s[..i].split_whitespace().count() == 1 && !s[..i].trim().is_empty() => {
            (s[..i].trim().to_string(), s[i + 1..].trim())
        }
        _ => (String::new(), s),
    }
}

fn relation(s: &str, line: usize) -> Result<(&str, String, f64), LpError> {
    // A range constraint has TWO relations: `1 <= a + b <= 2`. Reading the first and failing on the
    // rest gives a message about the remainder not being a number, which is true and useless. Say
    // what it actually is, because splitting it in two is the caller's decision.
    let relations = s.matches("<=").count()
        + s.matches(">=").count()
        + s.matches("=<").count()
        + s.matches("=>").count();
    if relations >= 2 {
        return err(
            line,
            "a range constraint must be written as two constraints; this reads one relation each",
        );
    }

    for rel in ["<=", ">=", "=<", "=>", "<", ">", "="] {
        if let Some(i) = s.find(rel) {
            let lhs = &s[..i];
            let rhs_text = s[i + rel.len()..].trim();
            let rhs: f64 = rhs_text
                .parse()
                .map_err(|_| LpError { line, message: format!("{rhs_text:?} is not a number") })?;
            let norm = match rel {
                "=<" | "<" => "<=",
                "=>" | ">" => ">=",
                r => r,
            };
            return Ok((lhs, norm.to_string(), rhs));
        }
    }
    err(line, format!("{s:?} has no relation (<=, >= or =)"))
}

/// `0 <= x <= 5`, or `x <= 5`, or `x >= 0`.
fn bound(s: &str, line: usize) -> Result<(i64, String, i64), LpError> {
    let parts: Vec<&str> = s.split_whitespace().collect();
    let num = |t: &str| -> Result<i64, LpError> {
        t.parse::<f64>()
            .ok()
            .filter(|v| v.fract() == 0.0)
            .map(|v| v as i64)
            .ok_or_else(|| LpError { line, message: format!("{t:?} is not a whole bound") })
    };
    match parts.as_slice() {
        [lo, "<=", name, "<=", hi] => Ok((num(lo)?, name.to_string(), num(hi)?)),
        // LP format's default LOWER bound is 0, so `x <= 5` is genuinely `0 <= x <= 5`.
        [name, "<=", hi] => Ok((0, name.to_string(), num(hi)?)),
        // Its default UPPER bound is +infinity, and there is no such spin domain.
        //
        // This used to return `lo + 1` as the upper bound, wrapped in an `i64::MAX.min(..)` that
        // reads like an overflow guard and is not one -- the addition happens first, so it guards
        // nothing, and clippy calling it dead code is what exposed the line. The effect was that
        // `t >= 10` became the domain 10..=11: `Maximize t` subject to `t >= 10` returned 11, a
        // confident optimum to a problem that is unbounded above.
        //
        // Refusing is the only honest answer. A finite default would be a different fabrication,
        // just one that takes longer to notice.
        [name, ">=", _] => err(
            line,
            format!(
                "{name:?} is bounded below but not above, and a spin domain must be finite. \
                 Write both ends, as in `{} <= {name} <= <upper>`.",
                parts[2]
            ),
        ),
        _ => err(line, format!("{s:?} is not a bound this reads")),
    }
}

/// Split `5 mon - 3 tue + 2` into atoms.
fn atoms(s: &str, line: usize) -> Result<Vec<Atom>, LpError> {
    if s.contains('^') || s.contains('*') || s.contains('[') {
        return err(
            line,
            "quadratic terms in an LP file are not read here; lower them to a linear model first",
        );
    }
    let mut out = Vec::new();
    let mut sign = 1.0;
    let mut pending: Option<f64> = None;
    // Split on signs while keeping them, then read each piece as [coefficient] [name].
    let spaced = s.replace('+', " + ").replace('-', " - ");
    for tok in spaced.split_whitespace() {
        match tok {
            "+" => {
                flush(&mut out, &mut pending, sign);
                sign = 1.0;
            }
            "-" => {
                flush(&mut out, &mut pending, sign);
                sign = -1.0;
            }
            _ => {
                if let Ok(v) = tok.parse::<f64>() {
                    if let Some(p) = pending.replace(v) {
                        out.push(Atom { coeff: sign * p, name: None });
                    }
                } else {
                    out.push(Atom { coeff: sign * pending.take().unwrap_or(1.0), name: Some(tok.to_string()) });
                    sign = 1.0;
                }
            }
        }
    }
    flush(&mut out, &mut pending, sign);
    Ok(out)
}

fn flush(out: &mut Vec<Atom>, pending: &mut Option<f64>, sign: f64) {
    if let Some(v) = pending.take() {
        out.push(Atom { coeff: sign * v, name: None });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SHIFTS: &str = "\
Maximize
  obj: 5 mon + 4 tue + 3 wed + 2 thu + 1 fri
Subject To
  cap: mon + tue + wed + thu + fri <= 3
Binary
  mon tue wed thu fri
End
";

    #[test]
    fn a_file_someone_else_wrote_solves_to_the_right_answer() {
        let m = parse(SHIFTS).expect("a plain LP file");
        let s = m.compile().unwrap().solve_best_of(24);
        assert!(s.feasible(), "{s}");
        let picked: Vec<&str> = ["mon", "tue", "wed", "thu", "fri"]
            .into_iter()
            .filter(|d| s.value(d) == 1)
            .collect();
        assert_eq!(picked, vec!["mon", "tue", "wed"], "the three most valuable, and no more: {s}");
    }

    #[test]
    fn minimise_and_maximise_are_opposite() {
        let up = parse(SHIFTS).unwrap().compile().unwrap().solve_best_of(24);
        let down = parse(&SHIFTS.replace("Maximize", "Minimize")).unwrap()
            .compile().unwrap().solve_best_of(24);
        assert_eq!(up.value("mon"), 1, "the most valuable day is taken when maximising");
        assert_eq!(down.value("mon"), 0, "and dropped when minimising: {down}");
    }

    #[test]
    fn every_relation_is_read() {
        let src = "\
Minimize
  obj: a + b + c
Subject To
  atmost:  a + b + c <= 2
  atleast: a + b >= 1
  exact:   c = 0
Binary
  a b c
End
";
        let m = parse(src).unwrap();
        let s = m.compile().unwrap().solve_best_of(24);
        assert!(s.feasible(), "{s}");
        let on = ["a", "b", "c"].iter().filter(|v| s.value(v) == 1).count();
        assert!(on <= 2, "at most two: {s}");
        assert!(s.value("a") + s.value("b") >= 1, "at least one of a, b: {s}");
        assert_eq!(s.value("c"), 0, "c is fixed off: {s}");
    }

    #[test]
    fn a_bounds_line_on_a_binary_is_refused_rather_than_dropped() {
        // The Bounds line was parsed into the map and then never consulted, because the binary path
        // does not read it. `x <= 0` fixes x OFF, and the model solved as though x were free -- a
        // confident answer to a different problem, from a file the caller did not write.
        let src = "Maximize\n  obj: x + y\nSubject To\n  c: x + y <= 2\nBounds\n  x <= 0\nBinary\n  x y\nEnd\n";
        let Err(e) = parse(src) else { panic!("a bound that fixes a binary must not be dropped") };
        assert!(e.message.contains("declared Binary"), "says what is wrong: {e}");
        assert!(e.message.contains("General"), "and offers the fix: {e}");

        // A redundant 0..1 bound says nothing new and stays legal.
        let ok = "Maximize\n  obj: x\nSubject To\n  c: x <= 1\nBounds\n  0 <= x <= 1\nBinary\n  x\nEnd\n";
        assert!(parse(ok).is_ok(), "0..1 on a binary is redundant, not contradictory");
    }

    #[test]
    fn a_bound_with_no_upper_end_is_refused_rather_than_invented() {
        // `t >= 10` compiled to the domain 10..=11, so `Maximize t` answered 11 -- an optimum
        // reported with full confidence for a problem that has none. Every bound test used the
        // two-sided form, so nothing looked here.
        let src = "Maximize\n  obj: t\nBounds\n  t >= 10\nGeneral\n  t\nEnd\n";
        // `expect_err` needs `Model: Debug`, which it deliberately is not.
        let Err(e) = parse(src) else { panic!("an unbounded-above variable has no spin domain") };
        assert!(e.message.contains("not above"), "says what is wrong: {e}");
        assert!(e.message.contains("10 <= t <= "), "and shows the fix: {e}");

        // The one-sided `<=` form stays legal: LP format's default LOWER bound really is 0.
        let ok = parse("Maximize\n  obj: t\nBounds\n  t <= 20\nGeneral\n  t\nEnd\n")
            .expect("`t <= 20` means 0 <= t <= 20");
        assert_eq!(ok.compile().unwrap().solve_best_of(16).value("t"), 20);
    }

    #[test]
    fn an_integer_variable_takes_its_declared_bounds() {
        // `max t` over an integer means maximise its VALUE. In an indicator model that is the sum
        // over the domain weighted by each value, not `[t == 1]` -- which is a value this variable
        // cannot even take.
        let src = "\
Maximize
  obj: t
Bounds
  10 <= t <= 20
General
  t
End
";
        let s = parse(src).unwrap().compile().unwrap().solve_best_of(16);
        assert_eq!(s.value("t"), 20, "maximising a value takes the top of its range: {s}");

        let down = parse(&src.replace("Maximize", "Minimize")).unwrap()
            .compile().unwrap().solve_best_of(16);
        assert_eq!(down.value("t"), 10, "and minimising takes the bottom: {down}");

        // an arithmetic constraint on that value is refused, with the distinction spelled out
        let bounded = "\
Maximize
  obj: t
Subject To
  c: t <= 15
Bounds
  10 <= t <= 20
General
  t
End
";
        let e = refusal(bounded);
        assert!(e.message.contains("not a count"), "{e}");
        assert!(e.message.contains("Bounds"), "and says where it belongs: {e}");
    }

    /// `Model` has no `Debug`, so `unwrap_err` will not do.
    fn refusal(src: &str) -> LpError {
        match parse(src) {
            Err(e) => e,
            Ok(_) => panic!("this should not have parsed"),
        }
    }

    #[test]
    fn what_it_cannot_read_it_refuses_by_name_and_line() {
        // The point of reading someone else's file is that you did not write it and cannot check
        // it by eye. Every one of these would otherwise be a confident answer to a different
        // problem.
        let undeclared = "\
Maximize
  obj: x + y
Subject To
  c: x + y <= 1
Binary
  x
End
";
        let e = refusal(undeclared);
        assert!(e.message.contains("'y'") && e.message.contains("never declared"), "{e}");

        let weighted = "\
Maximize
  obj: a + b
Subject To
  c: 3 a + 2 b <= 4
Binary
  a b
End
";
        let e = refusal(weighted);
        assert!(e.message.contains("other than 1"), "{e}");
        assert_eq!(e.line, 4, "and says which line: {e}");

        let quadratic = "\
Maximize
  obj: [ x * y ] / 2
Subject To
  c: x + y <= 1
Binary
  x y
End
";
        assert!(refusal(quadratic).message.contains("quadratic"));

        let continuous = "\
Maximize
  obj: x
Subject To
  c: x <= 1
End
";
        let e = refusal(continuous);
        assert!(e.message.contains("continuous"), "{e}");

        let range = "\
Minimize
  obj: a
Subject To
  c: 1 <= a + b <= 2
Binary
  a b
End
";
        assert!(refusal(range).message.contains("two constraints"));
    }

    #[test]
    fn comments_labels_and_spelling_variants_are_tolerated() {
        let src = "\
\\ a backslash comment, which is LP's own
MAXIMISE
  the_objective: a + b
subject to
  first: a + b <= 1
BIN
  a b
end
";
        let m = parse(src).expect("case and spelling vary in the wild");
        assert!(m.compile().unwrap().solve_best_of(8).feasible());
    }

    #[test]
    fn a_constraint_nothing_can_satisfy_is_refused_rather_than_compiled() {
        let src = "\
Minimize
  obj: a
Subject To
  c: a + b >= 5
Binary
  a b
End
";
        let e = refusal(src);
        assert!(e.message.contains("nothing can satisfy"), "{e}");
    }
}
