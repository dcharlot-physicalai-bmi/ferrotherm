//! G-set: the standard max-cut benchmark, and the sign convention that decides everything.
//!
//! G-set (Helmberg and Rendl) is what max-cut solvers have been compared on for twenty-five years.
//! Every paper reports a **best cut found**, and the field's league table is a list of those — which
//! says who has looked hardest, not how good the answer is. With [`bound`](crate::bound) the same
//! run can report an **upper bound on the max cut** and therefore a gap, which is a statement about
//! the instance rather than about the community.
//!
//! # Max-cut is an Ising minimisation with the couplings NEGATED
//!
//! For a cut `S`, with `s_i = +1` exactly when `i ∈ S`:
//!
//! ```text
//! cut(s) = ½ Σ_ij w_ij (1 − s_i s_j) = ½ [ W − Σ_ij w_ij s_i s_j ],    W = Σ_ij w_ij
//! ```
//!
//! This crate's energy is `E(s) = −Σ h_i s_i − Σ_ij J_ij s_i s_j`. Loading `J_ij = w_ij` gives
//! `E = −Σ w s s`, so `cut = (W + E)/2` — and **minimising that energy minimises the cut**, which
//! is the opposite of the intended problem and produces a number that looks entirely plausible.
//!
//! Load `J_ij = −w_ij` instead. Then `E = +Σ w s s` and:
//!
//! ```text
//! cut = (W − E) / 2
//! ```
//!
//! so minimising energy maximises the cut, and a **lower** bound `L ≤ min E` becomes an **upper**
//! bound on the cut:
//!
//! ```text
//! max cut ≤ (W − L) / 2
//! ```
//!
//! [`Instance::cut`] and [`Instance::cut_upper_bound`] are the only places that arithmetic lives,
//! so a sign error is one edit rather than one per caller.
//!
//! # The file format
//!
//! One header line `n m`, then `m` lines of `i j w` with **1-based** vertex numbers. Weights are
//! `±1` in the G-set proper; this parser takes any integer so the same loader reads the weighted
//! variants.

use crate::graph::{Graph, GraphBuilder};

/// A parsed max-cut instance, ready to minimise.
pub struct Instance {
    /// The Ising graph, couplings already **negated** so that minimising energy maximises the cut.
    pub graph: Graph,
    /// Σ of the edge weights, needed to convert energy to a cut value.
    pub total_weight: f64,
    pub nodes: usize,
    pub edges: usize,
}

/// Why a G-set file could not be read.
#[derive(Clone, Debug, PartialEq)]
pub enum GsetError {
    /// The header is missing or is not `n m`.
    Header(String),
    /// A body line is not `i j w`.
    Line { line: usize, text: String },
    /// A vertex number outside `1..=n`.
    ///
    /// Named rather than clamped: the format is 1-based and an off-by-one silently shifts every
    /// edge onto the wrong vertices, which still parses and still samples.
    Vertex { line: usize, got: i64, n: usize },
    /// The header promised `m` edges and the body had a different number.
    ///
    /// A truncated download is the common cause, and it produces a *valid* smaller instance whose
    /// cut values are quietly incomparable with everyone else's.
    Count { declared: usize, found: usize },
}

impl core::fmt::Display for GsetError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            GsetError::Header(t) => write!(f, "expected a header line `n m`, got {t:?}"),
            GsetError::Line { line, text } => write!(f, "line {line}: expected `i j w`, got {text:?}"),
            GsetError::Vertex { line, got, n } => {
                write!(f, "line {line}: vertex {got} is outside 1..={n}; this format is 1-based")
            }
            GsetError::Count { declared, found } => write!(
                f,
                "the header declares {declared} edges and the body has {found}. A truncated file \
                 parses into a valid SMALLER instance whose cut values are not comparable with \
                 anyone else's, so this is refused rather than solved"
            ),
        }
    }
}

/// Summary only, deliberately.
///
/// Deriving this would print the whole CSR, and a G-set instance has tens of thousands of edges --
/// so a failing `unwrap` would bury its own message under the graph it was complaining about.
impl core::fmt::Debug for Instance {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Instance {{ nodes: {}, edges: {}, W: {} }}", self.nodes, self.edges, self.total_weight)
    }
}

impl Instance {
    /// Parse the G-set edge-list format.
    pub fn parse(text: &str) -> Result<Instance, GsetError> {
        let mut lines = text.lines().enumerate().filter(|(_, l)| !l.trim().is_empty());
        let (_, head) = lines.next().ok_or_else(|| GsetError::Header(String::new()))?;
        let mut h = head.split_whitespace();
        let n: usize = h
            .next()
            .and_then(|v| v.parse().ok())
            .ok_or_else(|| GsetError::Header(head.to_string()))?;
        let m: usize = h
            .next()
            .and_then(|v| v.parse().ok())
            .ok_or_else(|| GsetError::Header(head.to_string()))?;

        let mut gb = GraphBuilder::new(n);
        let mut total = 0.0;
        let mut found = 0usize;
        for (no, l) in lines {
            let mut it = l.split_whitespace();
            let (Some(a), Some(b), Some(w)) = (it.next(), it.next(), it.next()) else {
                return Err(GsetError::Line { line: no + 1, text: l.to_string() });
            };
            let (Ok(a), Ok(b), Ok(w)) = (a.parse::<i64>(), b.parse::<i64>(), w.parse::<f64>()) else {
                return Err(GsetError::Line { line: no + 1, text: l.to_string() });
            };
            for v in [a, b] {
                if v < 1 || v as usize > n {
                    return Err(GsetError::Vertex { line: no + 1, got: v, n });
                }
            }
            // THE NEGATION. Everything in this module exists to get this sign right once.
            gb.couple(a as usize - 1, b as usize - 1, -w);
            total += w;
            found += 1;
        }
        if found != m {
            return Err(GsetError::Count { declared: m, found });
        }
        Ok(Instance { graph: gb.build(), total_weight: total, nodes: n, edges: m })
    }

    /// The cut a state achieves.
    pub fn cut(&self, s: &[i8]) -> f64 {
        (self.total_weight - self.graph.energy(s)) / 2.0
    }

    /// An upper bound on the max cut, from a lower bound on the energy.
    ///
    /// `max cut = (W − min E)/2`, and `L ≤ min E`, so `(W − L)/2 ≥ max cut`. Every published G-set
    /// result is a *lower* bound on the max cut (somebody found that cut); this is the other side,
    /// which is what turns "best known" into a gap.
    pub fn cut_upper_bound(&self, energy_lower_bound: f64) -> f64 {
        (self.total_weight - energy_lower_bound) / 2.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 5-cycle with unit weights. An odd cycle cannot be 2-coloured, so one edge always stays
    /// inside a side and the max cut is 4 of 5 — small enough to check by hand and by enumeration.
    const C5: &str = "5 5\n1 2 1\n2 3 1\n3 4 1\n4 5 1\n5 1 1\n";

    fn brute_max_cut(inst: &Instance) -> f64 {
        let n = inst.nodes;
        let mut best = f64::NEG_INFINITY;
        for mask in 0u32..(1u32 << n) {
            let s: Vec<i8> = (0..n).map(|i| if mask >> i & 1 == 1 { 1 } else { -1 }).collect();
            best = best.max(inst.cut(&s));
        }
        best
    }

    #[test]
    fn the_sign_convention_maximises_the_cut_rather_than_minimising_it() {
        // THE ERROR THIS MODULE EXISTS TO PREVENT. Load `J = +w` and minimising energy minimises
        // the cut: a plausible number, a valid state, and the wrong problem. With `J = -w` the
        // energy minimum must coincide with the cut maximum, checked by enumeration.
        let inst = Instance::parse(C5).unwrap();
        let n = inst.nodes;
        let (mut lo_e, mut best_cut_at_lo_e) = (f64::INFINITY, 0.0);
        for mask in 0u32..(1u32 << n) {
            let s: Vec<i8> = (0..n).map(|i| if mask >> i & 1 == 1 { 1 } else { -1 }).collect();
            let e = inst.graph.energy(&s);
            if e < lo_e {
                lo_e = e;
                best_cut_at_lo_e = inst.cut(&s);
            }
        }
        assert!((best_cut_at_lo_e - brute_max_cut(&inst)).abs() < 1e-9,
                "the energy minimum must BE the cut maximum: {best_cut_at_lo_e} vs {}",
                brute_max_cut(&inst));
        assert!((brute_max_cut(&inst) - 4.0).abs() < 1e-9, "C5's max cut is 4 of 5 edges");
    }

    #[test]
    fn the_bound_is_an_upper_bound_on_the_cut() {
        // Every published G-set number is a LOWER bound on the max cut -- somebody found that cut.
        // This is the other side, and it has to bracket the truth rather than merely be near it.
        let inst = Instance::parse(C5).unwrap();
        let b = crate::bound::forest(&inst.graph, 60);
        let ub = inst.cut_upper_bound(b.value);
        let truth = brute_max_cut(&inst);
        assert!(ub >= truth - 1e-9, "upper bound {ub} sits BELOW the true max cut {truth}");
    }

    #[test]
    fn a_truncated_file_is_refused_rather_than_solved() {
        // The common failure of a benchmark harness: a short download parses into a valid smaller
        // instance, samples happily, and reports a cut nobody else's number can be compared to.
        let short = "5 5\n1 2 1\n2 3 1\n";
        assert_eq!(
            Instance::parse(short).unwrap_err(),
            GsetError::Count { declared: 5, found: 2 }
        );
    }

    #[test]
    fn one_based_vertices_are_enforced_not_assumed() {
        // A 0 in the file means the producer was 0-based. Silently accepting it shifts every edge
        // by one vertex, which still parses and still samples.
        let zero = "3 1\n0 1 1\n";
        assert!(matches!(Instance::parse(zero), Err(GsetError::Vertex { got: 0, .. })));
        let over = "3 1\n1 4 1\n";
        assert!(matches!(Instance::parse(over), Err(GsetError::Vertex { got: 4, n: 3, .. })));
    }

    #[test]
    fn a_malformed_line_names_its_line_number() {
        let bad = "3 2\n1 2 1\nnot an edge\n";
        match Instance::parse(bad) {
            Err(GsetError::Line { line, .. }) => assert_eq!(line, 3),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn weights_are_summed_for_the_cut_conversion() {
        let w = Instance::parse("3 3\n1 2 2\n2 3 3\n1 3 5\n").unwrap();
        assert!((w.total_weight - 10.0).abs() < 1e-12);
        // All three on one side cuts nothing.
        assert!((w.cut(&[1, 1, 1]) - 0.0).abs() < 1e-12);
        // Splitting off vertex 1 cuts the two edges touching it: 2 + 5.
        assert!((w.cut(&[1, -1, -1]) - 7.0).abs() < 1e-12);
    }
}
