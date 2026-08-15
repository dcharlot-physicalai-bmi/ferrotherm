//! Higher-order reduction: run a k-body model on pairwise hardware.
//!
//! A factor of arity three or more has nowhere to go. Every fabric in [`crate::fabric`] declares
//! `max_arity: 2`, `Program::to_graph` refuses anything wider, and `Model::compile` returns
//! [`crate::model::CompileError::DegreeTooHigh`] — all three saying, correctly, that the program as
//! written does not fit. This is the pass that makes it fit.
//!
//! # What it does
//!
//! Introduce an **ancilla** spin constrained to equal the product of two existing ones, substitute
//! it for that pair wherever the pair appears, and repeat until nothing is wider than two. The
//! constraint is paid for with a penalty that costs nothing when it holds and more than the model
//! is worth when it does not.
//!
//! # Why it goes through binary
//!
//! In spin space, "`t` equals `s_a · s_b`" is itself a three-body statement, so enforcing it with
//! two-body terms is the problem we are trying to solve. In binary it is not: for `x ∈ {0, 1}`,
//!
//! ```text
//!     P(x_a, x_b, y) = 3y + x_a·x_b − 2·x_a·y − 2·x_b·y
//! ```
//!
//! is zero when `y = x_a·x_b` and at least one otherwise, and every term in it is quadratic. That is
//! Rosenberg's reduction, and it is why this converts `s = 2x − 1` on the way in and back on the
//! way out rather than trying to be clever in spin space.
//!
//! # What is guaranteed
//!
//! For every assignment of the ORIGINAL spins, the reduced energy minimised over the ancillas
//! equals the original energy plus a constant. That is the property that makes the reduction sound:
//! the ground states correspond exactly, and no assignment is reordered. `tests` checks it by
//! enumerating every state of both models, which is the only check that leaves nothing to argue
//! about.
//!
//! It is a statement about **optimisation**. The ancillas add states, so the Boltzmann distribution
//! over the original variables is not preserved at finite temperature; the penalty makes violating
//! assignments expensive rather than impossible. Use this to find ground states, and read
//! [`Reduction::penalty`] before sampling from one.


use crate::ftp::Program;
use std::collections::BTreeMap;
use std::collections::BTreeSet;

/// A multilinear polynomial over binary variables: monomial → coefficient.
///
/// The empty monomial is the constant. Multilinear because `x² = x` for binary `x`, so no variable
/// ever appears twice — the same reason [`Factor`] refuses a repeated variable.
type Poly = BTreeMap<BTreeSet<usize>, f64>;

fn add(p: &mut Poly, mono: BTreeSet<usize>, c: f64) {
    if c == 0.0 {
        return;
    }
    let e = p.entry(mono).or_insert(0.0);
    *e += c;
    if *e == 0.0 {
        // Drop a term that cancelled. Leaving a zero behind would make `degree` report a width the
        // polynomial does not have, and the loop below would reduce a monomial that is not there.
        let key = p.iter().find(|(_, v)| **v == 0.0).map(|(k, _)| k.clone());
        if let Some(k) = key {
            p.remove(&k);
        }
    }
}

fn degree(p: &Poly) -> usize {
    p.keys().map(|m| m.len()).max().unwrap_or(0)
}

/// What a reduction cost, and what it assumed.
#[derive(Clone, Debug, PartialEq)]
pub struct Reduction {
    /// The pairwise program.
    pub program: Program,
    /// Spins the pass added. They occupy indices `original..program.spins` and are not part of the
    /// answer — an ancilla's value is an artefact of the lowering.
    pub ancillas: usize,
    /// How many spins the original had, so a caller can slice the answer back down.
    pub original_spins: usize,
    /// The penalty weight enforcing each ancilla.
    ///
    /// Chosen as more than the whole model can pay: the sum of every coefficient's magnitude, so no
    /// assignment ever profits by breaking an ancilla's definition. Larger than strictly needed,
    /// and deliberately — a penalty tuned to the edge is a penalty that fails on the next model.
    pub penalty: f64,
    /// Constant energy offset dropped during the conversion. Add it to compare energies with the
    /// original; ignore it to compare states.
    pub offset: f64,
}

impl Reduction {
    /// Keep only the original spins from a solved state.
    pub fn project<'a>(&self, state: &'a [i8]) -> &'a [i8] {
        &state[..self.original_spins.min(state.len())]
    }
}

/// Why a program could not be reduced.
#[derive(Clone, Debug, PartialEq)]
pub enum ReduceError {
    /// A factor wide enough that expanding it would not finish.
    ///
    /// Converting a k-body spin product to binary produces 2^k monomials, so this is a real wall
    /// rather than a tidiness rule. Refused loudly instead of allocating until something dies.
    TooWide { arity: usize, limit: usize },
    /// The program has no factors and nothing to reduce.
    Empty,
}

impl core::fmt::Display for ReduceError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ReduceError::TooWide { arity, limit } => write!(
                f,
                "a factor of arity {arity} expands to 2^{arity} binary monomials, over the {limit} \
                 this will attempt; split the term before reducing it"
            ),
            ReduceError::Empty => write!(f, "nothing to reduce"),
        }
    }
}

/// The widest factor this will expand. 2^20 monomials from one factor is already unreasonable.
pub const MAX_ARITY: usize = 20;

/// Lower every factor to arity two, adding ancillas as needed.
///
/// A program already pairwise comes back unchanged with no ancillas, so this is safe to apply
/// unconditionally.
pub fn to_pairwise(p: &Program) -> Result<Reduction, ReduceError> {
    if let Some(f) = p.factors.iter().find(|f| f.arity() > MAX_ARITY) {
        return Err(ReduceError::TooWide { arity: f.arity(), limit: MAX_ARITY });
    }

    // 1. The energy as a binary polynomial. E = -Σ w·∏s - Σ h·s, and s = 2x - 1.
    let mut poly: Poly = BTreeMap::new();
    for f in &p.factors {
        let vars: Vec<usize> = f.vars().collect();
        expand_spin_product(&mut poly, &vars, -f.weight());
    }
    for &(i, h) in &p.bias {
        expand_spin_product(&mut poly, &[i], -h);
    }

    // 2. Rosenberg, until nothing is wider than two.
    let mut next = p.spins;
    let mut ancillas = 0usize;
    let scale: f64 = poly.values().map(|v| v.abs()).sum();
    // Nothing to outbid means nothing to enforce, but an ancilla still needs a positive weight.
    let penalty = if scale > 0.0 { scale * 2.0 } else { 1.0 };

    while degree(&poly) > 2 {
        // The pair appearing in the most wide monomials. Reducing the commonest pair first removes
        // the most degree per ancilla, which is the whole reason to choose rather than take any.
        let mut count: BTreeMap<(usize, usize), usize> = BTreeMap::new();
        for m in poly.keys().filter(|m| m.len() > 2) {
            let v: Vec<usize> = m.iter().copied().collect();
            for i in 0..v.len() {
                for j in (i + 1)..v.len() {
                    *count.entry((v[i], v[j])).or_insert(0) += 1;
                }
            }
        }
        let (a, b) = *count
            .iter()
            .max_by_key(|(_, n)| **n)
            .map(|(k, _)| k)
            .expect("degree > 2 means some monomial has a pair");

        let y = next;
        next += 1;
        ancillas += 1;

        // Substitute y for {a, b} in every monomial containing both.
        let mut rewritten: Poly = BTreeMap::new();
        for (m, c) in poly.iter() {
            if m.len() > 2 && m.contains(&a) && m.contains(&b) {
                let mut n: BTreeSet<usize> = m.iter().copied().filter(|v| *v != a && *v != b).collect();
                n.insert(y);
                add(&mut rewritten, n, *c);
            } else {
                add(&mut rewritten, m.clone(), *c);
            }
        }
        // P = 3y + x_a·x_b − 2·x_a·y − 2·x_b·y, zero exactly when y = x_a·x_b.
        add(&mut rewritten, BTreeSet::from([y]), 3.0 * penalty);
        add(&mut rewritten, BTreeSet::from([a, b]), penalty);
        add(&mut rewritten, BTreeSet::from([a, y]), -2.0 * penalty);
        add(&mut rewritten, BTreeSet::from([b, y]), -2.0 * penalty);
        poly = rewritten;
    }

    // 3. Back to spins. x = (1 + s)/2.
    let (program, offset) = to_program(&poly, next)?;
    Ok(Reduction { program, ancillas, original_spins: p.spins, penalty, offset })
}

/// `c · ∏(2x_i − 1)` expanded into binary monomials.
fn expand_spin_product(poly: &mut Poly, vars: &[usize], c: f64) {
    let k = vars.len();
    for mask in 0u32..(1u32 << k) {
        let mut mono = BTreeSet::new();
        let mut taken = 0;
        for (bit, &v) in vars.iter().enumerate() {
            if mask & (1 << bit) != 0 {
                mono.insert(v);
                taken += 1;
            }
        }
        // 2^taken from the chosen x's, (−1) for each factor that contributed its −1
        let sign = if (k - taken) % 2 == 0 { 1.0 } else { -1.0 };
        add(poly, mono, c * sign * (1u64 << taken) as f64);
    }
}

/// A degree-≤2 binary polynomial as a spin program, returning the dropped constant.
fn to_program(poly: &Poly, spins: usize) -> Result<(Program, f64), ReduceError> {
    // E = c0 + Σ a_i x_i + Σ b_ij x_i x_j, with x = (1+s)/2:
    //   x_i        = 1/2 + s_i/2
    //   x_i x_j    = 1/4 + s_i/4 + s_j/4 + s_i s_j /4
    let mut offset = 0.0;
    let mut lin: BTreeMap<usize, f64> = BTreeMap::new();
    let mut quad: BTreeMap<(usize, usize), f64> = BTreeMap::new();

    for (m, c) in poly {
        match m.len() {
            0 => offset += c,
            1 => {
                let i = *m.iter().next().unwrap();
                offset += c / 2.0;
                *lin.entry(i).or_insert(0.0) += c / 2.0;
            }
            2 => {
                let v: Vec<usize> = m.iter().copied().collect();
                offset += c / 4.0;
                *lin.entry(v[0]).or_insert(0.0) += c / 4.0;
                *lin.entry(v[1]).or_insert(0.0) += c / 4.0;
                *quad.entry((v[0], v[1])).or_insert(0.0) += c / 4.0;
            }
            n => return Err(ReduceError::TooWide { arity: n, limit: 2 }),
        }
    }

    // The program's convention is E = -Σ w·∏s - Σ h·s, so every coefficient flips sign.
    let mut src = format!("ftp 1\nname reduced\nspins {spins}\n");
    for ((i, j), w) in &quad {
        if *w != 0.0 {
            src.push_str(&format!("factor {} {i} {j}\n", -w));
        }
    }
    for (i, h) in &lin {
        if *h != 0.0 {
            src.push_str(&format!("bias {i} {}\n", -h));
        }
    }
    let program = Program::from_ftp(&src).map_err(|_| ReduceError::Empty)?;
    Ok((program, offset))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Energy of a spin state under a program, by its own definition.
    fn energy(p: &Program, s: &[i8]) -> f64 {
        let mut e = 0.0;
        for f in &p.factors {
            let prod: f64 = f.vars().map(|v| s[v] as f64).product();
            e -= f.weight() * prod;
        }
        for &(i, h) in &p.bias {
            e -= h * s[i] as f64;
        }
        e
    }

    /// For every assignment of the original spins, the reduced energy minimised over the ancillas.
    fn minimised_over_ancillas(r: &Reduction, orig: &[i8]) -> f64 {
        let k = r.ancillas;
        let mut best = f64::INFINITY;
        for mask in 0u32..(1u32 << k) {
            let mut s = orig.to_vec();
            for a in 0..k {
                s.push(if mask & (1 << a) != 0 { 1 } else { -1 });
            }
            best = best.min(energy(&r.program, &s));
        }
        best
    }

    /// The whole guarantee, checked by enumeration: reduced-minimised-over-ancillas equals
    /// original plus a constant, for EVERY state.
    fn agrees_everywhere(src: &str) -> Reduction {
        let p = Program::from_ftp(src).unwrap();
        let r = to_pairwise(&p).unwrap();
        assert!(
            r.program.factors.iter().all(|f| f.arity() <= 2),
            "the point of the pass is that nothing is wider than two"
        );

        let n = p.spins;
        let mut delta: Option<f64> = None;
        for mask in 0u32..(1u32 << n) {
            let s: Vec<i8> = (0..n).map(|i| if mask & (1 << i) != 0 { 1 } else { -1 }).collect();
            let want = energy(&p, &s);
            let got = minimised_over_ancillas(&r, &s);
            let d = got - want;
            match delta {
                None => delta = Some(d),
                Some(d0) => assert!(
                    (d - d0).abs() < 1e-9,
                    "state {s:?}: original {want}, reduced {got}, offset {d} but {d0} elsewhere — \
                     the reduction reordered states rather than shifting them"
                ),
            }
        }
        r
    }

    #[test]
    fn a_three_body_term_becomes_pairwise_without_moving_any_state() {
        let r = agrees_everywhere("ftp 1\nspins 3\nfactor 1.0 0 1 2\n");
        assert_eq!(r.ancillas, 1, "one pair replaced, one ancilla");
        assert_eq!(r.original_spins, 3);
    }

    #[test]
    fn a_four_body_term_and_a_negative_weight() {
        agrees_everywhere("ftp 1\nspins 4\nfactor -2.5 0 1 2 3\n");
    }

    #[test]
    fn several_higher_order_terms_sharing_variables() {
        // Sharing is what makes the pair choice matter: reducing the commonest pair first should
        // serve more than one monomial.
        let r = agrees_everywhere(
            "ftp 1\nspins 5\nfactor 1.0 0 1 2\nfactor 1.0 0 1 3\nfactor -1.0 0 1 4\nbias 2 0.5\n",
        );
        assert_eq!(r.ancillas, 1, "one ancilla for the pair (0,1) serves all three terms");
    }

    #[test]
    fn mixed_orders_including_terms_already_pairwise() {
        agrees_everywhere(
            "ftp 1\nspins 4\nfactor 1.0 0 1\nfactor 0.5 0 1 2\nfactor -1.0 1 2 3\nbias 0 0.25\n",
        );
    }

    #[test]
    fn a_pairwise_program_is_returned_untouched() {
        let p = Program::from_ftp("ftp 1\nspins 3\nfactor 1.0 0 1\nfactor 1.0 1 2\nbias 0 0.5\n")
            .unwrap();
        let r = to_pairwise(&p).unwrap();
        assert_eq!(r.ancillas, 0, "nothing to reduce");
        assert_eq!(r.program.spins, 3, "and no spins added");
        // still the same energies
        for mask in 0u32..8 {
            let s: Vec<i8> = (0..3).map(|i| if mask & (1 << i) != 0 { 1 } else { -1 }).collect();
            assert!((energy(&p, &s) - (energy(&r.program, &s) + r.offset)).abs() < 1e-9);
        }
    }

    #[test]
    fn the_ground_state_of_the_reduction_projects_to_the_original_ground_state() {
        // The property a caller actually uses: solve the pairwise version, throw the ancillas away,
        // and be holding an optimum of the model you wrote.
        let src = "ftp 1\nspins 4\nfactor 1.0 0 1 2\nfactor -1.5 1 2 3\nbias 0 0.3\nbias 3 -0.2\n";
        let p = Program::from_ftp(src).unwrap();
        let r = to_pairwise(&p).unwrap();

        let best_original = (0u32..(1 << p.spins))
            .map(|m| {
                let s: Vec<i8> =
                    (0..p.spins).map(|i| if m & (1 << i) != 0 { 1 } else { -1 }).collect();
                (energy(&p, &s) * 1e9) as i64
            })
            .min()
            .unwrap();

        let mut best = f64::INFINITY;
        let mut best_state = Vec::new();
        for m in 0u32..(1 << r.program.spins) {
            let s: Vec<i8> = (0..r.program.spins)
                .map(|i| if m & (1 << i) != 0 { 1 } else { -1 })
                .collect();
            let e = energy(&r.program, &s);
            if e < best {
                best = e;
                best_state = s;
            }
        }
        let projected = r.project(&best_state);
        assert_eq!(projected.len(), p.spins, "the ancillas are not part of the answer");
        assert_eq!(
            (energy(&p, projected) * 1e9) as i64,
            best_original,
            "the projected state must be an optimum of the original: {projected:?}"
        );
    }

    #[test]
    fn a_higher_order_program_runs_end_to_end_on_a_pairwise_fabric() {
        // The whole point, exercised the way a caller would: a model no fabric here accepts,
        // reduced, checked against a fabric that declares max_arity 2, sampled by the CPU backend,
        // and projected back to an optimum of the model that was written.
        use crate::fabric::{Cpu, Device, Fabric, Unsupported};
        use crate::ledger::Z1_SPICE;

        let src = "ftp 1\nspins 5\nfactor 1.0 0 1 2\nfactor -1.5 2 3 4\nbias 0 0.4\n";
        let p = Program::from_ftp(src).unwrap();

        // as written, every fabric refuses it -- and now says what to do about it
        let dw = Fabric::dwave_advantage2(Z1_SPICE);
        let refusal = dw.check(&p);
        assert!(
            refusal.iter().any(|u| matches!(u, Unsupported::ArityTooHigh { .. })),
            "{refusal:?}"
        );
        assert!(
            refusal.iter().any(|u| u.to_string().contains("to_pairwise")),
            "the refusal names the remedy: {refusal:?}"
        );

        // reduced, it passes the arity check
        let r = to_pairwise(&p).unwrap();
        assert!(r.ancillas > 0);
        assert!(
            !Fabric::unconstrained("sim", Z1_SPICE).check(&r.program).iter()
                .any(|u| matches!(u, Unsupported::ArityTooHigh { .. })),
            "nothing is wider than two now"
        );

        // and the CPU backend, which lowers through to_graph, accepts it where it refused before
        let mut cpu = Cpu::default();
        assert!(!cpu.program(&p).is_empty(), "the original does not run");
        assert!(cpu.program(&r.program).is_empty(), "{:?}", cpu.program(&r.program));

        // solved and projected, it is an optimum of the ORIGINAL model
        let best = crate::exact::Elimination::default()
            .ground_state(&r.program.to_graph().unwrap())
            .unwrap()
            .ground_state
            .expect("a ground state");
        let projected = r.project(&best);
        let want = (0u32..(1 << p.spins))
            .map(|m| {
                let s: Vec<i8> =
                    (0..p.spins).map(|i| if m & (1 << i) != 0 { 1 } else { -1 }).collect();
                (energy(&p, &s) * 1e9) as i64
            })
            .min()
            .unwrap();
        assert_eq!(
            (energy(&p, projected) * 1e9) as i64,
            want,
            "projected {projected:?} must minimise the model that was written"
        );
    }

    #[test]
    fn a_factor_too_wide_to_expand_is_refused_rather_than_attempted() {
        let vars: Vec<String> = (0..25).map(|i| i.to_string()).collect();
        let src = format!("ftp 1\nspins 30\nfactor 1.0 {}\n", vars.join(" "));
        let p = Program::from_ftp(&src).unwrap();
        match to_pairwise(&p) {
            Err(ReduceError::TooWide { arity: 25, limit: 20 }) => {}
            other => panic!("2^25 monomials should be refused, got {other:?}"),
        }
    }

    #[test]
    fn the_penalty_outweighs_the_model_it_guards() {
        // An ancilla whose definition is cheap to break is not a definition. The penalty must
        // exceed anything the rest of the model could pay for breaking it.
        let p = Program::from_ftp("ftp 1\nspins 3\nfactor 100.0 0 1 2\n").unwrap();
        let r = to_pairwise(&p).unwrap();
        assert!(r.penalty > 100.0, "penalty {} against a weight of 100", r.penalty);
    }
}
