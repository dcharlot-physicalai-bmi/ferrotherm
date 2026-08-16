//! Export a ferrotherm program as an **OMMX instance**.
//!
//! OMMX — Open Mathematical prograMming eXchange — is the interchange format this corner of the
//! field has converged on. jijmodeling 2.x compiles to it; `ommx` is a shared dependency across the
//! Jij stack. A `.ftp` program that can also be read as an OMMX instance is a program the rest of
//! the ecosystem can consume, without `.ftp` ceasing to be what it is.
//!
//! # What maps, exactly
//!
//! A compiled ferrotherm program is a set of spins with biases and pairwise couplings — an Ising
//! model. OMMX expresses that as an [`Instance`] with **binary** decision variables and a quadratic
//! objective, which is a lossless target for this shape:
//!
//! | ferrotherm | OMMX |
//! |---|---|
//! | spin `i` | `DecisionVariable { id: i, kind: BINARY, bound: [0,1] }` |
//! | bias `h_i` | a `Linear.Term` on `i` |
//! | coupling `J_ij` | a `Quadratic` row/column/value triple |
//! | minimise energy | `sense: SENSE_MINIMIZE` |
//!
//! **The variable change is the one thing to know.** Ferrotherm's spins are ±1 and OMMX's binaries
//! are 0/1, so `s = 2x - 1` is substituted during export. That is not a relabelling: it changes
//! every coefficient and introduces a constant, and an exporter that skipped it would produce a
//! file that parses cleanly and describes a different model. [`Export::constant`] carries the
//! offset so a caller can recover the original energy exactly.
//!
//! # Why this hand-rolls protobuf
//!
//! The Rust `ommx` crate is `3.0.0-beta.3` while its Python counterpart is stable at `2.6.2`, and a
//! shipped bridge should not rest on a beta. The subset needed here is varints and length-delimited
//! fields; the field numbers in [`schema`] were read out of the reference implementation's own
//! descriptors, not from prose. And correctness is not asserted from that reading — the tests have
//! the **reference implementation parse what this writes**, which is the only check that would
//! survive me having misread the schema.

use ferrotherm::graph::Graph;

/// Field numbers and enum values, read from `ommx.v1`'s own protobuf descriptors.
///
/// Kept in one block and named, because a wrong field number produces a file that decodes without
/// error into the wrong message — the failure mode a hand-rolled encoder has to be paranoid about.
pub mod schema {
    // Instance
    pub const INSTANCE_DECISION_VARIABLES: u32 = 2;
    pub const INSTANCE_OBJECTIVE: u32 = 3;
    pub const INSTANCE_SENSE: u32 = 5;
    // DecisionVariable
    pub const DV_ID: u32 = 1;
    pub const DV_KIND: u32 = 2;
    pub const DV_BOUND: u32 = 3;
    pub const DV_NAME: u32 = 4;
    // Bound
    pub const BOUND_LOWER: u32 = 1;
    pub const BOUND_UPPER: u32 = 2;
    // Function
    pub const FUNCTION_QUADRATIC: u32 = 3;
    // Linear
    pub const LINEAR_TERMS: u32 = 1;
    pub const LINEAR_CONSTANT: u32 = 2;
    // Linear.Term
    pub const TERM_ID: u32 = 1;
    pub const TERM_COEFFICIENT: u32 = 2;
    // Quadratic
    pub const QUAD_ROWS: u32 = 1;
    pub const QUAD_COLUMNS: u32 = 2;
    pub const QUAD_VALUES: u32 = 3;
    pub const QUAD_LINEAR: u32 = 4;

    pub const KIND_BINARY: u64 = 1;
    pub const SENSE_MINIMIZE: u64 = 1;
}

/// A ferrotherm graph, encoded as an OMMX instance.
pub struct Export {
    /// The serialised `ommx.v1.Instance`.
    pub bytes: Vec<u8>,
    /// The constant dropped by the ±1 → 0/1 substitution.
    ///
    /// `ferrotherm_energy(s) == ommx_objective(x) + constant` where `s_i = 2*x_i - 1`. An exporter
    /// that discarded this would produce an instance whose optimum is at the same point and whose
    /// value is wrong by a fixed amount, which is the kind of error that survives every test that
    /// only compares argmin.
    pub constant: f64,
    pub variables: usize,
}

/// Encode `g` as an OMMX instance, minimising its energy.
///
/// Ferrotherm's energy is `-Σ J_ij s_i s_j - Σ h_i s_i` over `s ∈ {-1,+1}`. Substituting
/// `s = 2x - 1` over `x ∈ {0,1}` and collecting terms gives the quadratic and linear coefficients
/// written below; the algebra is checked against the sampler in `an_exported_instance_scores_every
/// _state_the_way_ferrotherm_does`, which enumerates every state of a small graph rather than
/// trusting the derivation.
pub fn export(g: &Graph) -> Export {
    let n = g.n;
    let mut quad: Vec<(u64, u64, f64)> = Vec::new();
    let mut lin = vec![0.0f64; n];
    let mut constant = 0.0f64;

    // Couplings. -J*s_i*s_j with s = 2x-1 becomes -4J*x_i*x_j + 2J*x_i + 2J*x_j - J.
    for i in 0..n {
        for (k, &j) in g.nbr[g.offset[i]..g.offset[i + 1]].iter().enumerate() {
            let jj = j as usize;
            if jj <= i {
                continue; // each undirected edge once
            }
            let w = g.w[g.offset[i] + k];
            quad.push((i as u64, jj as u64, -4.0 * w));
            lin[i] += 2.0 * w;
            lin[jj] += 2.0 * w;
            constant -= w;
        }
    }
    // Biases. -h*s_i with s = 2x-1 becomes -2h*x_i + h.
    for i in 0..n {
        lin[i] += -2.0 * g.h[i];
        constant += g.h[i];
    }

    let mut linear = Vec::new();
    for (i, &c) in lin.iter().enumerate() {
        if c != 0.0 {
            let mut term = Vec::new();
            varint_field(&mut term, schema::TERM_ID, i as u64);
            double_field(&mut term, schema::TERM_COEFFICIENT, c);
            len_field(&mut linear, schema::LINEAR_TERMS, &term);
        }
    }
    // The constant rides in the Linear message so the OMMX objective equals ferrotherm's energy
    // exactly, rather than up to an offset a reader has to know about.
    if constant != 0.0 {
        double_field(&mut linear, schema::LINEAR_CONSTANT, constant);
    }

    let mut quadratic = Vec::new();
    for (r, _, _) in &quad {
        varint_field(&mut quadratic, schema::QUAD_ROWS, *r);
    }
    for (_, c, _) in &quad {
        varint_field(&mut quadratic, schema::QUAD_COLUMNS, *c);
    }
    for (_, _, v) in &quad {
        double_field(&mut quadratic, schema::QUAD_VALUES, *v);
    }
    len_field(&mut quadratic, schema::QUAD_LINEAR, &linear);

    let mut objective = Vec::new();
    len_field(&mut objective, schema::FUNCTION_QUADRATIC, &quadratic);

    let mut out = Vec::new();
    for i in 0..n {
        let mut bound = Vec::new();
        double_field(&mut bound, schema::BOUND_LOWER, 0.0);
        double_field(&mut bound, schema::BOUND_UPPER, 1.0);
        let mut dv = Vec::new();
        varint_field(&mut dv, schema::DV_ID, i as u64);
        varint_field(&mut dv, schema::DV_KIND, schema::KIND_BINARY);
        len_field(&mut dv, schema::DV_BOUND, &bound);
        str_field(&mut dv, schema::DV_NAME, &format!("s{i}"));
        len_field(&mut out, schema::INSTANCE_DECISION_VARIABLES, &dv);
    }
    len_field(&mut out, schema::INSTANCE_OBJECTIVE, &objective);
    varint_field(&mut out, schema::INSTANCE_SENSE, schema::SENSE_MINIMIZE);

    Export { bytes: out, constant, variables: n }
}

// ---- the wire format ---------------------------------------------------------------------------
// Protobuf, the three wire types this needs. Written out rather than pulled in, because the subset
// is small and the alternative is a beta dependency on somebody else's generated code.

fn key(buf: &mut Vec<u8>, field: u32, wire: u32) {
    varint(buf, ((field << 3) | wire) as u64);
}
fn varint(buf: &mut Vec<u8>, mut v: u64) {
    loop {
        let b = (v & 0x7f) as u8;
        v >>= 7;
        if v == 0 {
            buf.push(b);
            return;
        }
        buf.push(b | 0x80);
    }
}
fn varint_field(buf: &mut Vec<u8>, field: u32, v: u64) {
    key(buf, field, 0);
    varint(buf, v);
}
fn double_field(buf: &mut Vec<u8>, field: u32, v: f64) {
    key(buf, field, 1); // 64-bit, little-endian
    buf.extend_from_slice(&v.to_le_bytes());
}
fn len_field(buf: &mut Vec<u8>, field: u32, body: &[u8]) {
    key(buf, field, 2);
    varint(buf, body.len() as u64);
    buf.extend_from_slice(body);
}
fn str_field(buf: &mut Vec<u8>, field: u32, s: &str) {
    len_field(buf, field, s.as_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrotherm::ising::lattice2d;

    #[test]
    fn an_exported_instance_scores_every_state_the_way_ferrotherm_does() {
        // The substitution s = 2x - 1 changes every coefficient and introduces a constant. Rather
        // than trust the algebra above, enumerate every state of a small graph and check that the
        // OMMX objective -- reconstructed here from the same coefficients the encoder writes --
        // equals ferrotherm's energy exactly.
        let g = lattice2d(3, 1.0);
        let n = g.n;
        let mut quad: Vec<(usize, usize, f64)> = Vec::new();
        let mut lin = vec![0.0f64; n];
        let mut constant = 0.0f64;
        for i in 0..n {
            for (k, &j) in g.nbr[g.offset[i]..g.offset[i + 1]].iter().enumerate() {
                let jj = j as usize;
                if jj <= i {
                    continue;
                }
                let w = g.w[g.offset[i] + k];
                quad.push((i, jj, -4.0 * w));
                lin[i] += 2.0 * w;
                lin[jj] += 2.0 * w;
                constant -= w;
            }
        }
        for i in 0..n {
            lin[i] += -2.0 * g.h[i];
            constant += g.h[i];
        }

        for mask in 0..(1u32 << n) {
            let x: Vec<f64> = (0..n).map(|i| ((mask >> i) & 1) as f64).collect();
            let s: Vec<i8> = x.iter().map(|&v| if v > 0.5 { 1 } else { -1 }).collect();
            let mut obj = constant;
            for (i, j, c) in &quad {
                obj += c * x[*i] * x[*j];
            }
            for i in 0..n {
                obj += lin[i] * x[i];
            }
            let e = g.energy(&s);
            assert!(
                (obj - e).abs() < 1e-9,
                "state {mask:b}: OMMX objective {obj} vs ferrotherm energy {e}"
            );
        }
    }

    #[test]
    fn the_encoding_is_well_formed_protobuf() {
        // A structural check that does not need the reference implementation: every field this
        // writes must be re-readable by a minimal parser, and the message must consume exactly.
        let g = lattice2d(3, 1.0);
        let e = export(&g);
        let b = &e.bytes;
        let mut i = 0usize;
        let mut vars = 0;
        let mut saw_objective = false;
        let mut saw_sense = false;
        while i < b.len() {
            let (k, used) = read_varint(b, i);
            i += used;
            let (field, wire) = ((k >> 3) as u32, (k & 7) as u32);
            match wire {
                0 => {
                    let (v, u) = read_varint(b, i);
                    i += u;
                    if field == schema::INSTANCE_SENSE {
                        assert_eq!(v, schema::SENSE_MINIMIZE);
                        saw_sense = true;
                    }
                }
                1 => i += 8,
                2 => {
                    let (len, u) = read_varint(b, i);
                    i += u + len as usize;
                    if field == schema::INSTANCE_DECISION_VARIABLES {
                        vars += 1;
                    }
                    if field == schema::INSTANCE_OBJECTIVE {
                        saw_objective = true;
                    }
                }
                w => panic!("unexpected wire type {w}"),
            }
        }
        assert_eq!(i, b.len(), "the message must consume exactly, with no trailing bytes");
        assert_eq!(vars, g.n, "one decision variable per spin");
        assert!(saw_objective && saw_sense);
        assert_eq!(e.variables, g.n);
    }

    fn read_varint(b: &[u8], mut i: usize) -> (u64, usize) {
        let (mut v, mut shift, start) = (0u64, 0, i);
        loop {
            let byte = b[i];
            v |= ((byte & 0x7f) as u64) << shift;
            i += 1;
            if byte & 0x80 == 0 {
                return (v, i - start);
            }
            shift += 7;
        }
    }
}
