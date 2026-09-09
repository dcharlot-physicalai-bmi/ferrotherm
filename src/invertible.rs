//! Invertible logic — circuits that run backwards, so a multiplier is a factorizer.
//!
//! # What this is, and why a sampling crate is where it belongs
//!
//! A deterministic gate maps inputs to outputs and nothing else. The same gate written as an ISING
//! MODEL — an energy whose ground states are exactly the gate's truth table — has no direction at
//! all: clamp the inputs and the ground states are the output, clamp the OUTPUT and the ground
//! states are every input consistent with it. Camsari, Faria, Sutton & Datta (Phys. Rev. X 7,
//! 031014, 2017) called this invertible logic, and it is the canonical application of a
//! probabilistic-bit machine: a multiplier built this way, run with its product clamped, factors.
//!
//! It is the one thing every p-bit review names and this crate did not have. What it produces is
//! also exactly what the hardware here wants: an array multiplier is SPARSE and its couplings take
//! a handful of small values, where a naive all-to-all factorisation Hamiltonian is dense and needs
//! wide coefficients. The `ferrotherm-silicon` crate ships against a fabric whose coefficients
//! are four bits wide.
//!
//! # Gates are truth tables, not transcribed matrices
//!
//! The literature gives `J` and `h` per gate as tables. Those are easy to copy wrongly and
//! impossible to check by reading, so nothing here is transcribed. A gate is declared by its
//! TRUTH TABLE and the penalty is derived:
//!
//! ```text
//!   P(s) = Σ over unsatisfying assignments a of  Π_i (1 + a_i s_i)/2
//! ```
//!
//! Each term is 1 exactly on `a` and 0 on every other assignment, so `P` is zero on the truth table
//! and at least one off it. Expanding the product over subsets gives a polynomial of arity up to
//! the gate's width, which [`crate::reduce::to_pairwise_exact`] lowers to couplings with no penalty
//! coefficient to tune. Every gate's ground-state set is checked against its truth table by
//! enumeration in this module's tests — the construction is verified, not trusted.
//!
//! # The order matters and is checked
//!
//! An `n`-bit array multiplier is a few hundred gates by `n = 8`, and an off-by-one in the carry
//! chain produces a circuit that is still a valid Ising model with a perfectly good ground state —
//! for the wrong function. `a_multiplier_computes_products_exhaustively` enumerates every operand
//! pair at small widths and checks the product, which is the only way to know.

use crate::factor::Factor;
use crate::ftp::Program;
use crate::graph::Graph;
use crate::reduce::{Reduction, to_pairwise_exact};
use crate::schedule::Schedule;
use std::collections::{BTreeMap, BTreeSet};

/// Why a circuit could not be built or lowered.
#[derive(Clone, Debug, PartialEq)]
pub enum Error {
    /// A gate named a variable the circuit does not have.
    UnknownVar {
        /// The offending index.
        var: usize,
        /// How many the circuit has.
        vars: usize,
    },
    /// A gate wider than [`MAX_GATE`], whose truth table would not fit.
    TooWide {
        /// The gate's width.
        width: usize,
        /// The limit.
        limit: usize,
    },
    /// A gate no assignment satisfies, which no penalty can express as a ground state.
    Unsatisfiable,
    /// The lowering to pairwise failed; see [`crate::reduce::ReduceError`].
    Reduce(crate::reduce::ReduceError),
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Error::UnknownVar { var, vars } => {
                write!(f, "gate names variable {var}, but the circuit has {vars}")
            }
            Error::TooWide { width, limit } => {
                write!(f, "a {width}-variable gate is wider than the {limit} this builds tables for")
            }
            Error::Unsatisfiable => write!(f, "a gate no assignment satisfies has no ground state"),
            Error::Reduce(e) => write!(f, "lowering to pairwise failed: {e}"),
        }
    }
}

/// The widest gate a truth table is enumerated for. A gate of width `k` costs `2^k` rows twice
/// over, and every gate this crate builds circuits from is five variables or fewer.
pub const MAX_GATE: usize = 16;

/// One gate: the variables it constrains, and the assignments it permits.
///
/// `sat` holds bitmasks over `vars`, bit `i` being variable `vars[i]` at `+1`.
#[derive(Clone, Debug, PartialEq)]
struct GateSpec {
    vars: Vec<usize>,
    sat: BTreeSet<u32>,
}

/// A circuit of gates over a shared set of spins.
///
/// The energy is a sum of per-gate penalties, so the ground states are exactly the assignments that
/// satisfy EVERY gate — which is the circuit's relation, in whichever direction you clamp it.
#[derive(Clone, Debug, Default)]
pub struct Circuit {
    vars: usize,
    gates: Vec<GateSpec>,
}

impl Circuit {
    /// An empty circuit with `vars` variables.
    #[must_use]
    pub fn new(vars: usize) -> Circuit {
        Circuit { vars, gates: Vec::new() }
    }

    /// How many variables the circuit has.
    #[must_use]
    pub fn vars(&self) -> usize {
        self.vars
    }

    /// How many gates it holds.
    #[must_use]
    pub fn gates(&self) -> usize {
        self.gates.len()
    }

    /// Add a variable and return its index.
    pub fn alloc(&mut self) -> usize {
        self.vars += 1;
        self.vars - 1
    }

    /// Constrain `vars` to the assignments `f` accepts, where `f` receives one bool per variable in
    /// the order given, `true` meaning spin `+1`.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownVar`] for an index the circuit does not have, [`Error::TooWide`] past
    /// [`MAX_GATE`], and [`Error::Unsatisfiable`] for a relation nothing satisfies.
    pub fn gate(&mut self, vars: &[usize], f: impl Fn(&[bool]) -> bool) -> Result<(), Error> {
        if vars.len() > MAX_GATE {
            return Err(Error::TooWide { width: vars.len(), limit: MAX_GATE });
        }
        for &v in vars {
            if v >= self.vars {
                return Err(Error::UnknownVar { var: v, vars: self.vars });
            }
        }
        let k = vars.len();
        let mut sat = BTreeSet::new();
        let mut bits = vec![false; k];
        for a in 0u32..(1u32 << k) {
            for (i, b) in bits.iter_mut().enumerate() {
                *b = a >> i & 1 == 1;
            }
            if f(&bits) {
                sat.insert(a);
            }
        }
        if sat.is_empty() {
            return Err(Error::Unsatisfiable);
        }
        self.gates.push(GateSpec { vars: vars.to_vec(), sat });
        Ok(())
    }

    /// `y = a AND b`.
    ///
    /// # Errors
    /// As [`Self::gate`].
    pub fn and(&mut self, a: usize, b: usize, y: usize) -> Result<(), Error> {
        self.gate(&[a, b, y], |v| (v[0] && v[1]) == v[2])
    }

    /// `y = a OR b`.
    ///
    /// # Errors
    /// As [`Self::gate`].
    pub fn or(&mut self, a: usize, b: usize, y: usize) -> Result<(), Error> {
        self.gate(&[a, b, y], |v| (v[0] || v[1]) == v[2])
    }

    /// `y = a XOR b`.
    ///
    /// # Errors
    /// As [`Self::gate`].
    pub fn xor(&mut self, a: usize, b: usize, y: usize) -> Result<(), Error> {
        self.gate(&[a, b, y], |v| (v[0] ^ v[1]) == v[2])
    }

    /// `y = NOT a`.
    ///
    /// # Errors
    /// As [`Self::gate`].
    pub fn not(&mut self, a: usize, y: usize) -> Result<(), Error> {
        self.gate(&[a, y], |v| v[0] != v[1])
    }

    /// Half adder: `sum = a XOR b`, `carry = a AND b`. Declared as ONE four-variable gate rather
    /// than two three-variable ones, so the penalty has a single ground-state set to be right about.
    ///
    /// # Errors
    /// As [`Self::gate`].
    pub fn half_adder(&mut self, a: usize, b: usize, sum: usize, carry: usize) -> Result<(), Error> {
        self.gate(&[a, b, sum, carry], |v| {
            let t = usize::from(v[0]) + usize::from(v[1]);
            (t % 2 == 1) == v[2] && (t >= 2) == v[3]
        })
    }

    /// Full adder: `sum` and `cout` of `a + b + cin`.
    ///
    /// # Errors
    /// As [`Self::gate`].
    pub fn full_adder(
        &mut self,
        a: usize,
        b: usize,
        cin: usize,
        sum: usize,
        cout: usize,
    ) -> Result<(), Error> {
        self.gate(&[a, b, cin, sum, cout], |v| {
            let t = usize::from(v[0]) + usize::from(v[1]) + usize::from(v[2]);
            (t % 2 == 1) == v[3] && (t >= 2) == v[4]
        })
    }

    /// Whether `s` satisfies every gate. The relation the circuit encodes, evaluated directly.
    #[must_use]
    pub fn satisfied(&self, s: &[i8]) -> bool {
        self.gates.iter().all(|g| {
            let mut a = 0u32;
            for (i, &v) in g.vars.iter().enumerate() {
                if s[v] > 0 {
                    a |= 1 << i;
                }
            }
            g.sat.contains(&a)
        })
    }

    /// How many gates `s` violates — the penalty, evaluated from the truth tables directly.
    ///
    /// The independent check on [`Self::to_program`]: the program's energy, minus its minimum, must
    /// equal this for every state.
    #[must_use]
    pub fn violations(&self, s: &[i8]) -> usize {
        self.gates
            .iter()
            .filter(|g| {
                let mut a = 0u32;
                for (i, &v) in g.vars.iter().enumerate() {
                    if s[v] > 0 {
                        a |= 1 << i;
                    }
                }
                !g.sat.contains(&a)
            })
            .count()
    }

    /// Run the circuit FORWARD from `fixed`, completing each gate in declaration order.
    ///
    /// Gates are added with their outputs allocated after their inputs, so walking them in order
    /// determines the whole state. Returns `None` if some gate has no completion consistent with
    /// what is already known, or more than one — either means the circuit is not a function of
    /// `fixed`, which for a clamped input is a construction error worth surfacing rather than
    /// guessing past.
    ///
    /// This is the ordinary direction. The interesting one is to clamp the OUTPUTS instead and let
    /// a sampler find the inputs, which is what [`multiplier`] is for.
    #[must_use]
    pub fn forward(&self, fixed: &[(usize, bool)]) -> Option<Vec<i8>> {
        let mut known: Vec<Option<bool>> = vec![None; self.vars];
        for &(v, b) in fixed {
            if v >= self.vars {
                return None;
            }
            known[v] = Some(b);
        }
        for g in &self.gates {
            let k = g.vars.len();
            let free: Vec<usize> = (0..k).filter(|&i| known[g.vars[i]].is_none()).collect();
            let mut fits: Vec<u32> = Vec::new();
            for &a in &g.sat {
                if (0..k).all(|i| known[g.vars[i]].is_none_or(|b| b == (a >> i & 1 == 1))) {
                    fits.push(a);
                }
            }
            // Distinct only in the FREE positions: two satisfying rows that agree there are the
            // same completion seen twice, not an ambiguity.
            let mut seen = BTreeSet::new();
            for a in &fits {
                seen.insert(free.iter().map(|&i| u32::from(a >> i & 1 == 1) << i).sum::<u32>());
            }
            if seen.len() != 1 {
                return None;
            }
            let a = fits[0];
            for &i in &free {
                known[g.vars[i]] = Some(a >> i & 1 == 1);
            }
        }
        Some(known.iter().map(|b| if b.unwrap_or(false) { 1i8 } else { -1 }).collect())
    }

    /// The penalty as a program of arbitrary-arity factors, plus the constant that makes a
    /// satisfying assignment score exactly zero.
    ///
    /// # Errors
    ///
    /// Cannot currently fail; the signature is a `Result` so a future width or coefficient limit
    /// does not become a panic in code already written against it.
    ///
    /// # Panics
    ///
    /// Never in practice: the `expect` covers a `Factor` rejecting an index, and every index was
    /// checked against `vars` when its gate was added. It is an assertion that this invariant still
    /// holds, not a failure a caller can provoke.
    pub fn to_program(&self) -> Result<(Program, f64), Error> {
        let mut poly: BTreeMap<Vec<usize>, f64> = BTreeMap::new();
        for g in &self.gates {
            let k = g.vars.len();
            let norm = 1.0 / (1u64 << k) as f64;
            for a in 0u32..(1u32 << k) {
                if g.sat.contains(&a) {
                    continue;
                }
                // Π_i (1 + a_i s_i)/2, expanded over subsets: the indicator of exactly `a`.
                for mask in 0u32..(1u32 << k) {
                    let mut coeff = norm;
                    let mut t: Vec<usize> = Vec::with_capacity(k);
                    for i in 0..k {
                        if mask >> i & 1 == 1 {
                            if a >> i & 1 == 0 {
                                coeff = -coeff;
                            }
                            t.push(g.vars[i]);
                        }
                    }
                    t.sort_unstable();
                    *poly.entry(t).or_insert(0.0) += coeff;
                }
            }
        }

        let offset = poly.remove(&Vec::new()).unwrap_or(0.0);
        let mut p = Program {
            name: Some("invertible".into()),
            spins: self.vars,
            bias: Vec::new(),
            factors: Vec::new(),
            colors: Vec::new(),
            encodings: Vec::new(),
            schedule: Schedule::default(),
            observe: Vec::new(),
            target: None,
            price: None,
        };
        for (t, c) in poly {
            // A coefficient that cancelled to zero is not a term. Kept at an absolute threshold
            // rather than `== 0.0`: the expansion sums 2^k signed halves and lands a few ulps off.
            if c.abs() < 1e-12 {
                continue;
            }
            // Energy here is `-w · Π s`, so a polynomial term `c · Π s` is a factor of weight `-c`.
            match t.len() {
                1 => p.bias.push((t[0], -c)),
                _ => p.factors.push(
                    Factor::new(&t, -c, self.vars).expect("indices were checked when the gate was added"),
                ),
            }
        }
        Ok((p, offset))
    }

    /// The circuit as a pairwise graph, through the crate's penalty-free reduction.
    ///
    /// Returns the graph and the [`Reduction`] that maps a solved state back — the ancillas the
    /// lowering added are an artefact and [`Reduction::project`] drops them.
    ///
    /// # Errors
    ///
    /// [`Error::Reduce`] when the reduction refuses, which for these circuits means a gate wider
    /// than it lowers.
    ///
    /// # Panics
    ///
    /// Never in practice: the `expect` covers `to_graph` on the reduction's own output, which is
    /// pairwise by construction. It asserts that `to_pairwise_exact` kept its contract.
    pub fn to_graph(&self) -> Result<(Graph, Reduction), Error> {
        let (p, _) = self.to_program()?;
        let red = to_pairwise_exact(&p).map_err(Error::Reduce)?;
        let g = red.program.to_graph().expect("the reduction returns a pairwise program");
        Ok((g, red))
    }
}

/// An array multiplier, and the variables holding its operands and product.
///
/// Clamp `a` and `b` and the ground states give the product; clamp `product` and they give every
/// pair of operands that multiplies to it. That second direction is factorisation, and it is the
/// same circuit.
#[derive(Clone, Debug)]
pub struct Multiplier {
    /// The circuit.
    pub circuit: Circuit,
    /// Operand A, least significant bit first.
    pub a: Vec<usize>,
    /// Operand B, least significant bit first.
    pub b: Vec<usize>,
    /// The `2 * bits`-wide product, least significant bit first.
    pub product: Vec<usize>,
}

/// A `bits` x `bits` array multiplier.
///
/// Shift-and-add: one AND per operand-bit pair, then a ripple of half and full adders. The carry
/// structure is the part worth doubting, so it is checked by enumerating every operand pair rather
/// than by reading — see `a_multiplier_computes_products_exhaustively`.
///
/// # Errors
///
/// Propagates [`Circuit::gate`]'s errors, none of which this construction can currently trigger.
///
/// # Panics
///
/// If `bits` is zero, which is not a multiplier.
pub fn multiplier(bits: usize) -> Result<Multiplier, Error> {
    assert!(bits > 0, "a multiplier needs at least one bit");
    let mut c = Circuit::new(0);
    let a: Vec<usize> = (0..bits).map(|_| c.alloc()).collect();
    let b: Vec<usize> = (0..bits).map(|_| c.alloc()).collect();
    let mut product = Vec::with_capacity(2 * bits);

    // Row j of partial products: a_i AND b_j.
    let row = |c: &mut Circuit, a: &[usize], bj: usize| -> Result<Vec<usize>, Error> {
        let mut out = Vec::with_capacity(a.len());
        for &ai in a {
            let y = c.alloc();
            c.and(ai, bj, y)?;
            out.push(y);
        }
        Ok(out)
    };

    let first = row(&mut c, &a, b[0])?;
    product.push(first[0]);
    // Everything above bit 0 of the first row is what the next row adds to.
    let mut acc: Vec<usize> = first[1..].to_vec();

    for &bj in &b[1..] {
        let pp = row(&mut c, &a, bj)?;
        let mut carry: Option<usize> = None;
        let mut next: Vec<usize> = Vec::with_capacity(bits);
        for (i, &x) in pp.iter().enumerate() {
            let y = acc.get(i).copied();
            let (s, co) = match (y, carry) {
                (Some(y), Some(ci)) => {
                    let (s, co) = (c.alloc(), c.alloc());
                    c.full_adder(x, y, ci, s, co)?;
                    (s, Some(co))
                }
                (Some(y), None) => {
                    let (s, co) = (c.alloc(), c.alloc());
                    c.half_adder(x, y, s, co)?;
                    (s, Some(co))
                }
                (None, Some(ci)) => {
                    let (s, co) = (c.alloc(), c.alloc());
                    c.half_adder(x, ci, s, co)?;
                    (s, Some(co))
                }
                (None, None) => (x, None),
            };
            if i == 0 {
                product.push(s);
            } else {
                next.push(s);
            }
            carry = co;
        }
        if let Some(ci) = carry {
            next.push(ci);
        }
        acc = next;
    }
    product.extend_from_slice(&acc);
    // A `bits` x `bits` product is exactly `2 * bits` wide; the ripple can leave it one short when
    // the top carry is structurally zero, so pad with a variable pinned to zero by its own gate.
    while product.len() < 2 * bits {
        let z = c.alloc();
        c.gate(&[z], |v| !v[0])?;
        product.push(z);
    }
    Ok(Multiplier { circuit: c, a, b, product })
}

/// Read a little-endian bit field out of a spin state.
#[must_use]
pub fn read(vars: &[usize], s: &[i8]) -> u64 {
    let mut v = 0u64;
    for (i, &x) in vars.iter().enumerate() {
        if s[x] > 0 {
            v |= 1 << i;
        }
    }
    v
}

/// Write a little-endian bit field into a spin state.
pub fn write(vars: &[usize], value: u64, s: &mut [i8]) {
    for (i, &x) in vars.iter().enumerate() {
        s[x] = if value >> i & 1 == 1 { 1 } else { -1 };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The program's energy, in the crate's convention.
    fn energy(p: &Program, s: &[i8]) -> f64 {
        let mut e = 0.0;
        for &(i, h) in &p.bias {
            e -= h * f64::from(s[i]);
        }
        for f in &p.factors {
            e += f.energy(s);
        }
        e
    }

    fn states(n: usize) -> impl Iterator<Item = Vec<i8>> {
        (0u32..(1u32 << n)).map(move |m| {
            (0..n).map(|i| if m >> i & 1 == 1 { 1i8 } else { -1 }).collect()
        })
    }

    /// THE CONSTRUCTION IS AN IDENTITY, AND THIS IS IT: the program's energy plus its offset equals
    /// the number of gates the state violates, EXACTLY, at every state.
    ///
    /// The penalty is built by expanding `Π (1 + a_i s_i)/2` over subsets, which is a signed sum of
    /// `2^k` halves per unsatisfying row. Getting a sign or a normalisation wrong there still gives
    /// a polynomial with a sensible-looking ground state — for a different relation. Checking the
    /// identity rather than the ground state alone catches that on every state at once.
    #[test]
    fn the_penalty_counts_violated_gates_exactly() {
        let mut c = Circuit::new(6);
        c.and(0, 1, 2).unwrap();
        c.xor(2, 3, 4).unwrap();
        c.not(4, 5).unwrap();
        let (p, offset) = c.to_program().unwrap();
        for s in states(6) {
            let got = energy(&p, &s) + offset;
            let want = c.violations(&s) as f64;
            assert!(
                (got - want).abs() < 1e-9,
                "state {s:?}: penalty {got} against {want} violated gates"
            );
        }
    }

    /// EVERY GATE'S GROUND STATES ARE EXACTLY ITS TRUTH TABLE. Enumerated, per gate, rather than
    /// transcribed from the published J and h tables — which is the point of deriving them.
    #[test]
    fn every_gate_has_its_truth_table_as_its_ground_states() {
        type Build = fn(&mut Circuit) -> Result<(), Error>;
        let cases: [(&str, usize, Build); 6] = [
            ("and", 3, |c| c.and(0, 1, 2)),
            ("or", 3, |c| c.or(0, 1, 2)),
            ("xor", 3, |c| c.xor(0, 1, 2)),
            ("not", 2, |c| c.not(0, 1)),
            ("half_adder", 4, |c| c.half_adder(0, 1, 2, 3)),
            ("full_adder", 5, |c| c.full_adder(0, 1, 2, 3, 4)),
        ];
        for (name, n, build) in cases {
            let mut c = Circuit::new(n);
            build(&mut c).unwrap();
            let (p, offset) = c.to_program().unwrap();
            let mut ground = Vec::new();
            let mut sat = Vec::new();
            let mut worst_above = f64::INFINITY;
            for s in states(n) {
                let e = energy(&p, &s) + offset;
                if e < 1e-9 {
                    ground.push(s.clone());
                } else {
                    worst_above = worst_above.min(e);
                }
                if c.satisfied(&s) {
                    sat.push(s);
                }
            }
            assert_eq!(ground, sat, "{name}: ground states are not the truth table");
            assert!(!sat.is_empty(), "{name}: the truth table cannot be empty");
            // A gap of exactly one: every unsatisfying assignment violates the single gate once.
            assert!(
                (worst_above - 1.0).abs() < 1e-9,
                "{name}: the gap above the ground state is {worst_above}, not 1"
            );
        }
    }

    /// The carry chain is the part worth doubting, so every operand pair is checked.
    ///
    /// An off-by-one in the ripple produces a circuit that is still a valid Ising model with a
    /// perfectly good ground state, for the wrong function. Reading it cannot tell you; enumerating
    /// it can.
    #[test]
    fn a_multiplier_computes_products_exhaustively() {
        for bits in 1..=4usize {
            let m = multiplier(bits).unwrap();
            assert_eq!(m.product.len(), 2 * bits, "{bits}-bit product must be {} wide", 2 * bits);
            for x in 0..(1u64 << bits) {
                for y in 0..(1u64 << bits) {
                    let mut fixed: Vec<(usize, bool)> = Vec::new();
                    for (i, &v) in m.a.iter().enumerate() {
                        fixed.push((v, x >> i & 1 == 1));
                    }
                    for (i, &v) in m.b.iter().enumerate() {
                        fixed.push((v, y >> i & 1 == 1));
                    }
                    let s = m.circuit.forward(&fixed).expect("a clamped multiplier is determined");
                    assert!(m.circuit.satisfied(&s), "{bits}-bit {x}*{y}: forward state violates a gate");
                    assert_eq!(read(&m.product, &s), x * y, "{bits}-bit multiplier: {x} * {y}");
                }
            }
        }
    }

    /// RUN BACKWARDS IT IS A FACTORIZER, and that is checked by enumeration rather than by sampling
    /// — a sampler that failed to find the factors would be indistinguishable from a circuit that
    /// does not encode them.
    ///
    /// Two bits is small enough to enumerate the WHOLE circuit, ancillas and all, so this is the
    /// complete statement: with the product clamped, the states of zero penalty are exactly the
    /// operand pairs that multiply to it, and nothing else.
    #[test]
    fn a_clamped_product_has_exactly_its_factor_pairs_as_ground_states() {
        let bits = 2usize;
        let m = multiplier(bits).unwrap();
        let n = m.circuit.vars();
        assert!(n <= 20, "the circuit must stay enumerable for this to be exhaustive: {n} spins");
        let (p, offset) = m.circuit.to_program().unwrap();

        for target in 0..(1u64 << (2 * bits)) {
            let mut want: Vec<(u64, u64)> = Vec::new();
            for x in 0..(1u64 << bits) {
                for y in 0..(1u64 << bits) {
                    if x * y == target {
                        want.push((x, y));
                    }
                }
            }
            let mut got: Vec<(u64, u64)> = Vec::new();
            for s in states(n) {
                if read(&m.product, &s) != target {
                    continue;
                }
                if energy(&p, &s) + offset < 1e-9 {
                    got.push((read(&m.a, &s), read(&m.b, &s)));
                }
            }
            got.sort_unstable();
            got.dedup();
            assert_eq!(got, want, "clamping the product to {target} must give exactly its factors");
        }
    }

    /// The pairwise lowering must preserve which states are ground states. It adds ancillas with
    /// their own penalties, and a reduction that let an ancilla pay its way out would answer a
    /// different question.
    #[test]
    fn the_pairwise_reduction_keeps_the_same_satisfying_states() {
        let mut c = Circuit::new(5);
        c.full_adder(0, 1, 2, 3, 4).unwrap();
        let (g, red) = c.to_graph().unwrap();
        assert!(red.ancillas > 0, "a five-variable gate must need ancillas, or this tests nothing");
        assert_eq!(red.original_spins, 5);

        // Minimise over the FULL lowered space, then project: for each original assignment, the
        // best completion over the ancillas is what the reduction claims to preserve.
        let mut best: Vec<f64> = vec![f64::INFINITY; 32];
        for s in states(g.n) {
            let key = (0..5).fold(0usize, |k, i| k | (usize::from(s[i] > 0) << i));
            best[key] = best[key].min(g.energy(&s));
        }
        let floor = best.iter().copied().fold(f64::INFINITY, f64::min);
        let ground: Vec<usize> = (0..32).filter(|&k| best[k] - floor < 1e-9).collect();
        let sat: Vec<usize> = (0..32)
            .filter(|&k| {
                let s: Vec<i8> = (0..5).map(|i| if k >> i & 1 == 1 { 1i8 } else { -1 }).collect();
                c.satisfied(&s)
            })
            .collect();
        assert_eq!(ground, sat, "the lowered graph's ground states are not the gate's truth table");
    }

    /// Refusals are by name, and an unsatisfiable gate is refused rather than encoded as a penalty
    /// no assignment can pay.
    #[test]
    fn malformed_gates_are_refused_by_name() {
        let mut c = Circuit::new(2);
        assert_eq!(c.and(0, 1, 7), Err(Error::UnknownVar { var: 7, vars: 2 }));
        assert_eq!(c.gate(&[0, 1], |_| false), Err(Error::Unsatisfiable));
        let wide: Vec<usize> = (0..MAX_GATE + 1).map(|_| 0).collect();
        assert_eq!(
            c.gate(&wide, |_| true),
            Err(Error::TooWide { width: MAX_GATE + 1, limit: MAX_GATE })
        );
        assert_eq!(c.gates(), 0, "no refused gate may have been recorded");
    }
}
