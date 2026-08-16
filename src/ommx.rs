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
//! file that parses cleanly and describes a different model. The exporter applies it: the constant
//! is written into the instance, so `ommx_objective(x) == ferrotherm_energy(s)` with nothing left
//! for a caller to do. [`Export::constant`] reports the value for inspection and must NOT be added
//! again.
//!
//! It sits beside [`crate::ftp`] and [`crate::lp`] because that is what it is: a format bridge, and
//! this crate already keeps those in the core rather than out at the edge. Putting it in a sibling
//! crate, as the first version did, meant the C ABI could not reach it and eight of the nine
//! surfaces could not export at all.
//!
//! # Why this hand-rolls protobuf
//!
//! The Rust `ommx` crate is `3.0.0-beta.3` while its Python counterpart is stable at `2.6.2`, and a
//! shipped bridge should not rest on a beta. The subset needed here is varints and length-delimited
//! fields; the field numbers in [`schema`] were read out of the reference implementation's own
//! descriptors, not from prose. And correctness is not asserted from that reading — the tests have
//! the **reference implementation parse what this writes**, which is the only check that would
//! survive me having misread the schema.

use crate::graph::Graph;

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
    /// The offset the ±1 → 0/1 substitution produced, **already folded into the instance**.
    ///
    /// Read it, do not add it. `ommx_objective(x) == ferrotherm_energy(s)` exactly, because the
    /// constant is written into the `Linear` message during export rather than left for a caller to
    /// apply. Adding it again double-counts.
    ///
    /// It is reported because the substitution is a fact about the model worth being able to see —
    /// the value that makes the two objectives line up — not because anything downstream needs to
    /// apply it. `an_exported_objective_needs_no_correction` pins that, and it is pinned because the
    /// first version of this field said the opposite: the docs told callers to add a number the
    /// exporter had already applied, and the reference test agreed with the code rather than the
    /// prose for a whole release.
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

    // PACKED, which is proto3's default for repeated scalars and what the reference emits. The
    // first version wrote them unpacked, one key per element -- also valid, and the reference read
    // it, but emitting the canonical form keeps the bytes comparable with everyone else's.
    let mut quadratic = Vec::new();
    if !quad.is_empty() {
        let mut rows = Vec::new();
        for (r, _, _) in &quad { varint(&mut rows, *r); }
        len_field(&mut quadratic, schema::QUAD_ROWS, &rows);
        let mut cols = Vec::new();
        for (_, c, _) in &quad { varint(&mut cols, *c); }
        len_field(&mut quadratic, schema::QUAD_COLUMNS, &cols);
        let mut vals = Vec::new();
        for (_, _, v) in &quad { vals.extend_from_slice(&v.to_le_bytes()); }
        len_field(&mut quadratic, schema::QUAD_VALUES, &vals);
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

/// Why an OMMX instance could not be read as a ferrotherm model.
///
/// Refused by name, the way `src/lp.rs` refuses an LP file it cannot express. A bridge that
/// silently dropped what it could not represent would hand back a model that solves a different
/// problem, which is worse than not reading the file at all.
#[derive(Debug, Clone, PartialEq)]
pub enum ImportError {
    /// A variable this sampler cannot represent.
    ///
    /// Ferrotherm samples spins. A continuous variable has no spin encoding at any width, and a
    /// bounded integer needs one the caller has to choose, so neither is silently guessed at.
    UnsupportedKind { id: u64, name: String, kind: u64 },
    /// A variable whose bounds are not 0/1.
    NotBinary { id: u64, name: String, lower: f64, upper: f64 },
    /// An objective of degree three or higher.
    ///
    /// `crate::reduce` lowers those onto pairwise hardware with ancillas, so this is a
    /// deliberate boundary rather than a limit: read the model, then reduce it, so the caller sees
    /// what the reduction cost.
    TooHighDegree,
    /// The bytes are not a well-formed instance.
    Malformed(String),
    /// An empty instance.
    NoVariables,
}

impl core::fmt::Display for ImportError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ImportError::UnsupportedKind { id, name, kind } => write!(
                f,
                "decision variable {id} ('{name}') has kind {kind}; ferrotherm samples spins, so \
                 only KIND_BINARY (1) can be read directly. A continuous variable has no spin \
                 encoding at any width; a general integer needs one the caller must choose, and \
                 guessing it would silently change the problem."
            ),
            ImportError::NotBinary { id, name, lower, upper } => write!(
                f,
                "decision variable {id} ('{name}') is bounded [{lower}, {upper}] rather than \
                 [0, 1]. A binary variable with other bounds is a different variable."
            ),
            ImportError::TooHighDegree => write!(
                f,
                "this objective has degree three or higher. crate::reduce lowers such a model \
                 onto pairwise hardware with one ancilla per substituted pair -- run it explicitly, \
                 so the ancilla count is visible rather than paid silently here."
            ),
            ImportError::Malformed(why) => write!(f, "not a well-formed ommx.v1.Instance: {why}"),
            ImportError::NoVariables => write!(f, "this instance declares no decision variables"),
        }
    }
}

/// Read an `ommx.v1.Instance` as a ferrotherm graph.
///
/// The inverse of [`export`], and the direction that lets ferrotherm consume what the rest of the
/// ecosystem emits -- a jijmodeling problem compiled to OMMX, for instance.
///
/// **The substitution runs the other way.** OMMX binaries are 0/1 and ferrotherm spins are ±1, so
/// `x = (s + 1) / 2` is applied on the way in. As on the way out it changes every coefficient, and
/// the constant it produces is returned rather than dropped, so a caller can reconstruct the OMMX
/// objective from a ferrotherm energy exactly.
///
/// Repeated scalar fields are accepted **packed or unpacked**. The reference implementation packs;
/// this crate's own encoder did not until it was checked. A decoder that handled only one of them
/// would read half the files it is given.
pub fn import(bytes: &[u8]) -> Result<(Graph, f64), ImportError> {
    let mut vars: Vec<(u64, String, u64, f64, f64)> = Vec::new();
    let mut objective: Option<&[u8]> = None;
    let mut sense = schema::SENSE_MINIMIZE;

    for (field, body) in Fields::new(bytes) {
        match (field, body) {
            (schema::INSTANCE_DECISION_VARIABLES, Body::Bytes(b)) => {
                // Same proto3 rule throughout: absent means the default. A bound is [0, 1] here
                // because that is what a binary variable's is, and an omitted `lower` is 0.0
                // rather than missing.
                let (mut id, mut name, mut kind, mut lo, mut hi) = (0u64, String::new(), 0u64, 0.0, 1.0);
                for (f2, b2) in Fields::new(b) {
                    match (f2, b2) {
                        (schema::DV_ID, Body::Varint(v)) => id = v,
                        (schema::DV_KIND, Body::Varint(v)) => kind = v,
                        (schema::DV_NAME, Body::Bytes(s)) => name = String::from_utf8_lossy(s).into(),
                        (schema::DV_BOUND, Body::Bytes(bb)) => {
                            for (f3, b3) in Fields::new(bb) {
                                match (f3, b3) {
                                    (schema::BOUND_LOWER, Body::Fixed64(v)) => lo = f64::from_bits(v),
                                    (schema::BOUND_UPPER, Body::Fixed64(v)) => hi = f64::from_bits(v),
                                    _ => {}
                                }
                            }
                        }
                        _ => {}
                    }
                }
                vars.push((id, name, kind, lo, hi));
            }
            (schema::INSTANCE_OBJECTIVE, Body::Bytes(b)) => objective = Some(b),
            (schema::INSTANCE_SENSE, Body::Varint(v)) => sense = v,
            _ => {}
        }
    }

    if vars.is_empty() {
        return Err(ImportError::NoVariables);
    }
    for (id, name, kind, lo, hi) in &vars {
        if *kind != schema::KIND_BINARY {
            return Err(ImportError::UnsupportedKind { id: *id, name: name.clone(), kind: *kind });
        }
        if *lo != 0.0 || *hi != 1.0 {
            return Err(ImportError::NotBinary { id: *id, name: name.clone(), lower: *lo, upper: *hi });
        }
    }
    // Ids are the caller's; index by position after sorting so a sparse or shuffled id space maps
    // onto a dense spin space without silently renumbering anything the caller can see.
    let mut ids: Vec<u64> = vars.iter().map(|v| v.0).collect();
    ids.sort_unstable();
    let index = |id: u64| ids.binary_search(&id).ok();
    let n = ids.len();

    let mut lin = vec![0.0f64; n];
    let mut quad: Vec<(usize, usize, f64)> = Vec::new();
    let mut constant = 0.0f64;

    if let Some(obj) = objective {
        let mut linear_body: Option<&[u8]> = None;
        for (f, b) in Fields::new(obj) {
            match (f, b) {
                (1, Body::Fixed64(v)) => constant += f64::from_bits(v),
                (2, Body::Bytes(bb)) => linear_body = Some(bb),
                (schema::FUNCTION_QUADRATIC, Body::Bytes(q)) => {
                    let (mut rows, mut cols, mut vals) = (Vec::new(), Vec::new(), Vec::new());
                    for (fq, bq) in Fields::new(q) {
                        match (fq, bq) {
                            (schema::QUAD_ROWS, Body::Varint(v)) => rows.push(v),
                            (schema::QUAD_ROWS, Body::Bytes(p)) => packed_varints(p, &mut rows),
                            (schema::QUAD_COLUMNS, Body::Varint(v)) => cols.push(v),
                            (schema::QUAD_COLUMNS, Body::Bytes(p)) => packed_varints(p, &mut cols),
                            (schema::QUAD_VALUES, Body::Fixed64(v)) => vals.push(f64::from_bits(v)),
                            (schema::QUAD_VALUES, Body::Bytes(p)) => packed_doubles(p, &mut vals),
                            (schema::QUAD_LINEAR, Body::Bytes(l)) => linear_body = Some(l),
                            _ => {}
                        }
                    }
                    if rows.len() != cols.len() || rows.len() != vals.len() {
                        return Err(ImportError::Malformed(format!(
                            "quadratic has {} rows, {} columns and {} values",
                            rows.len(), cols.len(), vals.len()
                        )));
                    }
                    for k in 0..rows.len() {
                        let (Some(i), Some(j)) = (index(rows[k]), index(cols[k])) else {
                            return Err(ImportError::Malformed(format!(
                                "quadratic term names variable {} or {}, which is not declared",
                                rows[k], cols[k]
                            )));
                        };
                        quad.push((i, j, vals[k]));
                    }
                }
                (4, Body::Bytes(_)) => return Err(ImportError::TooHighDegree),
                _ => {}
            }
        }
        if let Some(l) = linear_body {
            for (f, b) in Fields::new(l) {
                match (f, b) {
                    (schema::LINEAR_TERMS, Body::Bytes(tb)) => {
                        // Default 0, NOT a sentinel. proto3 omits a field at its default value,
                        // so `Term { id: 0, coefficient: c }` serialises the coefficient alone --
                        // and a sentinel turns the commonest variable in the file into "not
                        // declared". This crate's own encoder writes id 0 explicitly, so its reader
                        // and writer agreed with each other and were both wrong about the format;
                        // it took reading a file the reference produced to see it.
                        let (mut id, mut c) = (0u64, 0.0f64);
                        for (ft, bt) in Fields::new(tb) {
                            match (ft, bt) {
                                (schema::TERM_ID, Body::Varint(v)) => id = v,
                                (schema::TERM_COEFFICIENT, Body::Fixed64(v)) => c = f64::from_bits(v),
                                _ => {}
                            }
                        }
                        let Some(i) = index(id) else {
                            return Err(ImportError::Malformed(format!(
                                "linear term names variable {id}, which is not declared"
                            )));
                        };
                        lin[i] += c;
                    }
                    (schema::LINEAR_CONSTANT, Body::Fixed64(v)) => constant += f64::from_bits(v),
                    _ => {}
                }
            }
        }
    }

    // MAXIMIZE is MINIMIZE of the negation. Ferrotherm minimises energy, so flip once here rather
    // than carrying a sense through everything downstream.
    let flip = if sense == 2 { -1.0 } else { 1.0 };

    // x = (s+1)/2. A quadratic c*x_i*x_j becomes c/4 * (s_i s_j + s_i + s_j + 1); a linear a*x_i
    // becomes a/2 * s_i + a/2. Ferrotherm's energy is -J s_i s_j - h_i s_i, so the signs invert.
    let mut b = crate::graph::GraphBuilder::new(n);
    let mut h = vec![0.0f64; n];
    let mut out_const = constant * flip;
    for (i, j, c) in &quad {
        let c = c * flip;
        b.couple(*i, *j, -c / 4.0);
        h[*i] -= c / 4.0;
        h[*j] -= c / 4.0;
        out_const += c / 4.0;
    }
    for i in 0..n {
        let a = lin[i] * flip;
        h[i] -= a / 2.0;
        out_const += a / 2.0;
    }
    for (i, v) in h.iter().enumerate() {
        b.bias(i, *v);
    }
    Ok((b.build(), out_const))
}

/// One protobuf field: its number and its payload.
enum Body<'a> {
    Varint(u64),
    Fixed64(u64),
    Bytes(&'a [u8]),
}

/// Walk the fields of a protobuf message, skipping anything unrecognised.
///
/// Forward compatibility is the point: OMMX has grown fields between 2.0 and 2.6 and will grow more,
/// and a reader that choked on one it did not know would break on the next release.
struct Fields<'a> {
    b: &'a [u8],
    i: usize,
}

impl<'a> Fields<'a> {
    fn new(b: &'a [u8]) -> Self {
        Fields { b, i: 0 }
    }
}

impl<'a> Iterator for Fields<'a> {
    type Item = (u32, Body<'a>);
    fn next(&mut self) -> Option<Self::Item> {
        if self.i >= self.b.len() {
            return None;
        }
        let (k, used) = read_varint(self.b, self.i)?;
        self.i += used;
        let (field, wire) = ((k >> 3) as u32, (k & 7) as u32);
        match wire {
            0 => {
                let (v, u) = read_varint(self.b, self.i)?;
                self.i += u;
                Some((field, Body::Varint(v)))
            }
            1 => {
                if self.i + 8 > self.b.len() {
                    return None;
                }
                let v = u64::from_le_bytes(self.b[self.i..self.i + 8].try_into().ok()?);
                self.i += 8;
                Some((field, Body::Fixed64(v)))
            }
            2 => {
                let (len, u) = read_varint(self.b, self.i)?;
                self.i += u;
                let end = self.i + len as usize;
                if end > self.b.len() {
                    return None;
                }
                let s = &self.b[self.i..end];
                self.i = end;
                Some((field, Body::Bytes(s)))
            }
            5 => {
                self.i += 4;
                Some((field, Body::Varint(0)))
            }
            _ => None,
        }
    }
}

fn read_varint(b: &[u8], mut i: usize) -> Option<(u64, usize)> {
    let (mut v, mut shift, start) = (0u64, 0u32, i);
    loop {
        let byte = *b.get(i)?;
        v |= ((byte & 0x7f) as u64) << shift;
        i += 1;
        if byte & 0x80 == 0 {
            return Some((v, i - start));
        }
        shift += 7;
        if shift > 63 {
            return None;
        }
    }
}

fn packed_varints(p: &[u8], out: &mut Vec<u64>) {
    let mut i = 0;
    while let Some((v, u)) = read_varint(p, i) {
        out.push(v);
        i += u;
        if i >= p.len() {
            break;
        }
    }
}

fn packed_doubles(p: &[u8], out: &mut Vec<f64>) {
    for c in p.chunks_exact(8) {
        out.push(f64::from_bits(u64::from_le_bytes(c.try_into().unwrap())));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ising::lattice2d;

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

    #[test]
    fn a_graph_survives_a_round_trip_through_ommx() {
        // Export then import, and check the two graphs score every state identically. Not "the
        // same coefficients" -- the substitution runs both ways and the coefficients are supposed
        // to change -- but the same ENERGY, which is the only thing that has to survive.
        for g in [lattice2d(3, 1.0), lattice2d(2, -0.7)] {
            let e = export(&g);
            let (back, constant) = import(&e.bytes).expect("our own export must import");
            assert_eq!(back.n, g.n);
            for mask in 0..(1u32 << g.n) {
                let s: Vec<i8> = (0..g.n).map(|i| if (mask >> i) & 1 == 1 { 1 } else { -1 }).collect();
                let want = g.energy(&s);
                let got = back.energy(&s) + constant;
                assert!(
                    (want - got).abs() < 1e-9,
                    "state {mask:b}: {want} before the round trip, {got} after"
                );
            }
        }
    }

    #[test]
    fn what_this_sampler_cannot_represent_is_refused_by_name() {
        // The lp.rs discipline: a bridge that silently dropped what it could not represent would
        // hand back a model solving a different problem, which is worse than not reading the file.
        let mut inst = Vec::new();
        let mut dv = Vec::new();
        varint_field(&mut dv, schema::DV_ID, 0);
        varint_field(&mut dv, schema::DV_KIND, 3); // KIND_CONTINUOUS
        str_field(&mut dv, schema::DV_NAME, "temperature");
        len_field(&mut inst, schema::INSTANCE_DECISION_VARIABLES, &dv);
        match import(&inst) {
            Err(ImportError::UnsupportedKind { id, name, kind }) => {
                assert_eq!((id, kind), (0, 3));
                assert_eq!(name, "temperature");
                let msg = ImportError::UnsupportedKind { id, name, kind }.to_string();
                assert!(msg.contains("no spin encoding"), "must say WHY: {msg}");
            }
            Err(e) => panic!("expected UnsupportedKind, got {e}"),
            Ok(_) => panic!("a continuous variable must be refused, not read"),
        }

        // Degree three or higher points at `reduce` rather than being silently lowered here.
        let mut poly = Vec::new();
        let mut dv2 = Vec::new();
        varint_field(&mut dv2, schema::DV_ID, 0);
        varint_field(&mut dv2, schema::DV_KIND, schema::KIND_BINARY);
        let mut bound = Vec::new();
        double_field(&mut bound, schema::BOUND_LOWER, 0.0);
        double_field(&mut bound, schema::BOUND_UPPER, 1.0);
        len_field(&mut dv2, schema::DV_BOUND, &bound);
        len_field(&mut poly, schema::INSTANCE_DECISION_VARIABLES, &dv2);
        let mut obj = Vec::new();
        len_field(&mut obj, 4, &[1, 2, 3]); // Function.polynomial
        len_field(&mut poly, schema::INSTANCE_OBJECTIVE, &obj);
        assert!(matches!(import(&poly), Err(ImportError::TooHighDegree)));
        assert!(matches!(import(&[]), Err(ImportError::NoVariables)));
    }

    #[test]
    fn packed_and_unpacked_repeated_fields_both_read() {
        // The reference PACKS repeated scalars; this crate's own encoder did not until it was
        // checked against real output. A decoder that handled only one of the two would read half
        // the files it is given, and would look correct against its own writer forever.
        let g = lattice2d(3, 1.0);
        let packed = export(&g);
        let (from_packed, c1) = import(&packed.bytes).unwrap();

        // The same instance with rows/columns/values written one key per element.
        let mut unpacked = Vec::new();
        for (f, b) in Fields::new(&packed.bytes) {
            match (f, b) {
                (schema::INSTANCE_OBJECTIVE, Body::Bytes(obj)) => {
                    let mut newobj = Vec::new();
                    for (f2, b2) in Fields::new(obj) {
                        match (f2, b2) {
                            (schema::FUNCTION_QUADRATIC, Body::Bytes(q)) => {
                                let mut nq = Vec::new();
                                for (f3, b3) in Fields::new(q) {
                                    match (f3, b3) {
                                        (schema::QUAD_ROWS, Body::Bytes(p)) => {
                                            let mut v = Vec::new();
                                            packed_varints(p, &mut v);
                                            for x in v { varint_field(&mut nq, schema::QUAD_ROWS, x); }
                                        }
                                        (schema::QUAD_COLUMNS, Body::Bytes(p)) => {
                                            let mut v = Vec::new();
                                            packed_varints(p, &mut v);
                                            for x in v { varint_field(&mut nq, schema::QUAD_COLUMNS, x); }
                                        }
                                        (schema::QUAD_VALUES, Body::Bytes(p)) => {
                                            let mut v = Vec::new();
                                            packed_doubles(p, &mut v);
                                            for x in v { double_field(&mut nq, schema::QUAD_VALUES, x); }
                                        }
                                        (fx, Body::Bytes(bx)) => len_field(&mut nq, fx, bx),
                                        _ => {}
                                    }
                                }
                                len_field(&mut newobj, schema::FUNCTION_QUADRATIC, &nq);
                            }
                            (fx, Body::Bytes(bx)) => len_field(&mut newobj, fx, bx),
                            _ => {}
                        }
                    }
                    len_field(&mut unpacked, schema::INSTANCE_OBJECTIVE, &newobj);
                }
                (fx, Body::Bytes(bx)) => len_field(&mut unpacked, fx, bx),
                (fx, Body::Varint(v)) => varint_field(&mut unpacked, fx, v),
                _ => {}
            }
        }
        let (from_unpacked, c2) = import(&unpacked).unwrap();
        assert_eq!(c1, c2);
        for mask in 0..(1u32 << g.n) {
            let s: Vec<i8> = (0..g.n).map(|i| if (mask >> i) & 1 == 1 { 1 } else { -1 }).collect();
            assert!((from_packed.energy(&s) - from_unpacked.energy(&s)).abs() < 1e-12);
        }
    }

    #[test]
    fn an_exported_objective_needs_no_correction() {
        // The contract, pinned. The exporter folds the substitution's constant into the instance,
        // so the two objectives are equal and a caller who adds `Export::constant` gets a number
        // that is wrong by exactly that much.
        //
        // This is here because the docs said the opposite for a release: `Export::constant` was
        // described as an offset to apply, the C header and all four bindings repeated it, and the
        // reference test agreed with the CODE rather than the prose -- so nothing failed.
        let g = lattice2d(3, 1.0);
        let e = export(&g);
        let (back, leftover) = import(&e.bytes).unwrap();
        for mask in 0..(1u32 << g.n) {
            let s: Vec<i8> = (0..g.n).map(|i| if (mask >> i) & 1 == 1 { 1 } else { -1 }).collect();
            let direct = g.energy(&s);
            let reimported = back.energy(&s) + leftover;
            assert!(
                (direct - reimported).abs() < 1e-9,
                "a round trip must need only the IMPORT leftover: {direct} vs {reimported}"
            );
            assert!(
                (direct - (reimported + e.constant)).abs() > 1e-12 || e.constant == 0.0,
                "adding the EXPORT constant on top must be wrong -- if it is not, the exporter \
                 stopped folding it in and this doc is stale again"
            );
        }
    }
}
