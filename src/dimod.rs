//! `dimod` / Ocean interoperability: D-Wave's binary quadratic model file, both vartypes, and the
//! QUBO coefficient-triplet text format.
//!
//! [`crate::ommx`] speaks the Jij stack's interchange format and [`crate::dimacs`] speaks MAX-SAT's.
//! This speaks the one with the largest installed base: every Ocean sampler, every `dwave-system`
//! embedding, every `dwave-neal` run and every `qbsolv` benchmark is a `dimod.BinaryQuadraticModel`,
//! and `bqm.to_file()` is how one is moved between processes.
//!
//! # The energy conventions differ, and that is the first thing to get right
//!
//! `dimod` writes `E(s) = offset + Σ h_i s_i + Σ_{i<j} J_ij s_i s_j` — **plus** signs.
//! This crate's [`crate::graph::Graph`] uses the statistical-mechanics sign,
//! `E(s) = −Σ h_i s_i − Σ_{i<j} w_ij s_i s_j`. So [`Bqm::from_graph`] negates every coefficient and
//! [`Bqm::to_graph`] negates them back. A bridge that skipped the negation would read a file whose
//! ferromagnet is an antiferromagnet, agree on the energy scale, and rank every state backwards.
//!
//! # The vartype change carries a constant
//!
//! QUBO variables are `x ∈ {0,1}` and spins are `s ∈ {−1,+1}`, related by `x = (1+s)/2`. Substituting
//! into a spin model and collecting terms gives
//!
//! ```text
//!   a_i    = 2 h_i − 2 Σ_{j~i} J_ij          (linear)
//!   b_ij   = 4 J_ij                          (quadratic)
//!   offset = offset − Σ_i h_i + Σ_{i<j} J_ij (constant)
//! ```
//!
//! and the inverse runs `h_i = a_i/2 + Σ_{j~i} b_ij/4`, `J_ij = b_ij/4`,
//! `offset = offset + Σ_i a_i/2 + Σ_{i<j} b_ij/4`.
//!
//! **That constant is the whole hazard.** It is the same for every state, so dropping it leaves the
//! ordering of states — and therefore every optimiser's answer — untouched, while every reported
//! energy is wrong by a fixed amount. A solver test passes; a comparison against a published
//! objective value fails. [`Bqm::to_vartype`] carries it, and
//! `the_offset_is_invisible_to_the_ranking_and_fatal_to_the_value` is the test that pins it by
//! enumeration rather than by argument.
//!
//! # The byte layout, read out of `dimod` 0.12.21 rather than from prose
//!
//! `dimod`'s own docstring says the per-variable index field holds "neighborhood starts" while the
//! method that writes it is called `_ilinear_and_degree`. Those disagree, and a triangle — where
//! every degree is 2 and every start is even — cannot tell them apart. A graph with degrees 3,1,2,0
//! can, and it says **starts**:
//!
//! ```text
//!   "DIMODBQM"          8 bytes
//!   major, minor        1 byte each; 2.0 is written here, 1.0 and 2.0 are read
//!   HEADER_LEN          u32, little-endian
//!   header              JSON, sorted keys, then '\n', then spaces so 14+HEADER_LEN % 64 == 0
//!   offset              1 x dtype
//!   linear              num_variables x (ntype start, dtype bias)
//!   quadratic           2 x num_interactions x (itype neighbour, dtype bias), each edge twice,
//!                       variable v's slice being starts[v]..starts[v+1]
//!   "VARS" len json     only in 2.0, only when the labels are not 0..n-1
//! ```
//!
//! Version 1.0 puts the label array in the header instead. Both are read; 2.0 is written, which is
//! what `dimod` itself defaults to.
//!
//! # What this refuses
//!
//! Labels are integers or strings. `dimod` also permits tuples — `dwave_networkx`'s Chimera
//! coordinates are the common case — and those are refused by name rather than flattened into
//! something that would re-serialise as a different model.

use crate::graph::{Graph, GraphBuilder};

/// The eight bytes every `dimod` BQM file starts with.
pub const MAGIC: [u8; 8] = *b"DIMODBQM";

/// The serialization version [`write_bqm`] emits, which is `dimod`'s own default.
pub const VERSION: (u8, u8) = (2, 0);

// ---- vartypes and labels ------------------------------------------------------------------------

/// Which domain a model's variables live in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Vartype {
    /// `s ∈ {−1, +1}`, an Ising model.
    Spin,
    /// `x ∈ {0, 1}`, a QUBO.
    Binary,
}

impl Vartype {
    /// `dimod`'s own spelling: `SPIN` or `BINARY`.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Vartype::Spin => "SPIN",
            Vartype::Binary => "BINARY",
        }
    }

    /// Read `dimod`'s spelling, case-insensitively. `None` for anything else.
    #[must_use]
    pub fn from_name(s: &str) -> Option<Vartype> {
        if s.eq_ignore_ascii_case("SPIN") {
            Some(Vartype::Spin)
        } else if s.eq_ignore_ascii_case("BINARY") {
            Some(Vartype::Binary)
        } else {
            None
        }
    }
}

/// A variable's label, which `dimod` keeps and this crate's spin indices do not have.
///
/// Carried so a file written by a modeller who named their variables re-serialises as the model
/// they wrote, rather than as an anonymous one that merely scores the same.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Label {
    /// An integer label. `dimod` writes these for embedded problems, where the label is a qubit.
    Index(i64),
    /// A string label.
    Text(String),
}

/// Map a spin state to the 0/1 state of the same model: `x = (1 + s) / 2`.
#[must_use]
pub fn binary_state(spins: &[i8]) -> Vec<i8> {
    spins.iter().map(|&s| i8::from(s > 0)).collect()
}

/// Map a 0/1 state to the spin state of the same model: `s = 2x − 1`.
#[must_use]
pub fn spin_state(bits: &[i8]) -> Vec<i8> {
    bits.iter().map(|&x| if x != 0 { 1i8 } else { -1 }).collect()
}

// ---- the model ----------------------------------------------------------------------------------

/// A binary quadratic model: linear biases, quadratic biases, a constant, and a vartype.
///
/// The energy is `dimod`'s, **not** [`crate::graph::Graph`]'s:
/// `E = offset + Σ linear[i] · v_i + Σ (u,v,b) b · v_u · v_v`.
#[derive(Clone, Debug, PartialEq)]
pub struct Bqm {
    /// Whether the variables are `{−1,+1}` or `{0,1}`. Every coefficient is read against this.
    pub vartype: Vartype,
    /// Linear bias per variable; its length is the variable count.
    pub linear: Vec<f64>,
    /// One entry per interaction as `(u, v, bias)`. Canonically `u < v`, sorted, and unique — see
    /// [`Bqm::canonicalize`], which the writers apply for themselves.
    pub quadratic: Vec<(usize, usize, f64)>,
    /// The constant term. **Not** decorative: see this module's header.
    pub offset: f64,
    /// One label per variable, parallel to [`Bqm::linear`].
    pub labels: Vec<Label>,
}

impl Bqm {
    /// An all-zero model over `n` variables, labelled `0..n`.
    #[must_use]
    pub fn new(vartype: Vartype, n: usize) -> Bqm {
        Bqm {
            vartype,
            linear: vec![0.0; n],
            quadratic: Vec::new(),
            offset: 0.0,
            labels: (0..n as i64).map(Label::Index).collect(),
        }
    }

    /// Variable count.
    #[must_use]
    pub fn num_variables(&self) -> usize {
        self.linear.len()
    }

    /// Interactions, after [`Bqm::canonicalize`] would merge duplicates.
    #[must_use]
    pub fn num_interactions(&self) -> usize {
        self.canonical().quadratic.len()
    }

    /// Add to variable `i`'s linear bias.
    ///
    /// # Panics
    ///
    /// If `i` is past the variable count.
    pub fn bias(&mut self, i: usize, b: f64) {
        assert!(i < self.linear.len(), "variable {i} of {}", self.linear.len());
        self.linear[i] += b;
    }

    /// Add a quadratic bias between `i` and `j`.
    ///
    /// A self term `i == j` is folded rather than refused, because it is exactly representable:
    /// `x·x = x` for a binary variable, so it becomes linear, and `s·s = 1` for a spin, so it
    /// becomes a constant. Note that a diagonal entry in a QUBO text file is a LINEAR bias by that
    /// format's definition, not a self-interaction — [`read_triplets`] sends those to [`Bqm::bias`].
    ///
    /// # Panics
    ///
    /// If either index is past the variable count.
    pub fn couple(&mut self, i: usize, j: usize, b: f64) {
        let n = self.linear.len();
        assert!(i < n && j < n, "variables ({i},{j}) of {n}");
        if i == j {
            match self.vartype {
                Vartype::Binary => self.linear[i] += b,
                Vartype::Spin => self.offset += b,
            }
            return;
        }
        self.quadratic.push((i.min(j), i.max(j), b));
    }

    /// Energy of `state`, whose values are in this model's own vartype.
    ///
    /// # Panics
    ///
    /// If `state` is shorter than the variable count.
    #[must_use]
    pub fn energy(&self, state: &[i8]) -> f64 {
        assert!(state.len() >= self.linear.len(), "state of {} for {} variables", state.len(), self.linear.len());
        let mut e = self.offset;
        for (i, &h) in self.linear.iter().enumerate() {
            e += h * f64::from(state[i]);
        }
        for &(u, v, b) in &self.quadratic {
            e += b * f64::from(state[u]) * f64::from(state[v]);
        }
        e
    }

    /// Sort the interactions by `(u, v)` with `u < v`, summing duplicates and folding self terms.
    ///
    /// Interactions whose bias is zero are **kept**: `dimod` treats an explicit zero interaction as
    /// part of the model's shape, it is counted in the file's `shape` field, and dropping it here
    /// would make a round trip return a model that scores the same and is not equal.
    pub fn canonicalize(&mut self) {
        let mut terms: Vec<(usize, usize, f64)> = Vec::with_capacity(self.quadratic.len());
        for &(i, j, b) in &self.quadratic {
            if i == j {
                match self.vartype {
                    Vartype::Binary => self.linear[i] += b,
                    Vartype::Spin => self.offset += b,
                }
            } else {
                terms.push((i.min(j), i.max(j), b));
            }
        }
        terms.sort_by_key(|t| (t.0, t.1));
        let mut out: Vec<(usize, usize, f64)> = Vec::with_capacity(terms.len());
        for (u, v, b) in terms {
            match out.last_mut() {
                Some(last) if last.0 == u && last.1 == v => last.2 += b,
                _ => out.push((u, v, b)),
            }
        }
        self.quadratic = out;
    }

    /// A canonical copy: see [`Bqm::canonicalize`].
    #[must_use]
    pub fn canonical(&self) -> Bqm {
        let mut c = self.clone();
        c.canonicalize();
        c
    }

    /// The same model over the other vartype, with the constant the substitution produces folded
    /// into [`Bqm::offset`].
    ///
    /// `self.energy(s)` and `converted.energy(mapped)` agree exactly, where `mapped` is
    /// [`binary_state`] or [`spin_state`] of `s`.
    #[must_use]
    pub fn to_vartype(&self, want: Vartype) -> Bqm {
        if want == self.vartype {
            return self.clone();
        }
        let mut out = self.canonical();
        out.vartype = want;
        match want {
            // s = 2x − 1
            Vartype::Binary => {
                let mut offset = out.offset;
                for h in &out.linear {
                    offset -= *h;
                }
                for h in &mut out.linear {
                    *h *= 2.0;
                }
                for &mut (u, v, ref mut b) in &mut out.quadratic {
                    offset += *b;
                    out.linear[u] -= 2.0 * *b;
                    out.linear[v] -= 2.0 * *b;
                    *b *= 4.0;
                }
                out.offset = offset;
            }
            // x = (1 + s) / 2
            Vartype::Spin => {
                let mut offset = out.offset;
                for h in &mut out.linear {
                    *h *= 0.5;
                    offset += *h;
                }
                for &mut (u, v, ref mut b) in &mut out.quadratic {
                    *b *= 0.25;
                    offset += *b;
                    out.linear[u] += *b;
                    out.linear[v] += *b;
                }
                out.offset = offset;
            }
        }
        out
    }

    /// A spin model with this crate's sign convention, read as a `dimod` `SPIN` model.
    ///
    /// Every coefficient is negated: `h_dimod = −h`, `J_dimod = −w`. The resulting model scores
    /// every state exactly as [`Graph::energy`] does.
    #[must_use]
    pub fn from_graph(g: &Graph) -> Bqm {
        let mut out = Bqm::new(Vartype::Spin, g.n);
        for i in 0..g.n {
            out.linear[i] = -g.h[i];
            for k in g.offset[i]..g.offset[i + 1] {
                let j = g.nbr[k] as usize;
                if j > i {
                    out.quadratic.push((i, j, -g.w[k]));
                }
            }
        }
        out
    }

    /// This model as a ferrotherm graph, plus the constant the graph cannot hold.
    ///
    /// `bqm.energy(state) == graph.energy(spins) + constant`, where `spins` is `state` for a `SPIN`
    /// model and [`spin_state`] of it for a `BINARY` one. The constant is returned rather than
    /// dropped for the reason the module header gives.
    #[must_use]
    pub fn to_graph(&self) -> (Graph, f64) {
        let spin = self.to_vartype(Vartype::Spin);
        let mut b = GraphBuilder::new(spin.linear.len());
        for (i, &h) in spin.linear.iter().enumerate() {
            b.bias(i, -h);
        }
        for &(u, v, j) in &spin.quadratic {
            b.couple(u, v, -j);
        }
        (b.build(), spin.offset)
    }

    /// Whether the labels are exactly `0..n`, which is what `dimod` records as `"variables": false`.
    #[must_use]
    pub fn is_index_labeled(&self) -> bool {
        self.labels.iter().enumerate().all(|(i, l)| *l == Label::Index(i as i64))
    }

    /// Neighbourhoods in the file's own order: per variable, ascending by neighbour.
    fn neighborhoods(&self) -> Vec<Vec<(usize, f64)>> {
        let mut nb = vec![Vec::new(); self.linear.len()];
        for &(u, v, b) in &self.quadratic {
            nb[u].push((v, b));
            nb[v].push((u, b));
        }
        for list in &mut nb {
            list.sort_by_key(|e| e.0);
        }
        nb
    }
}

// ---- errors -------------------------------------------------------------------------------------

/// Why a `dimod` model could not be read.
#[derive(Clone, Debug, PartialEq)]
pub enum BqmError {
    /// The first eight bytes are not [`MAGIC`].
    Magic {
        /// What was there instead, or as much of it as the input held.
        got: Vec<u8>,
    },
    /// A major version this reader does not know. 1 and 2 are read.
    Version {
        /// Major version byte.
        major: u8,
        /// Minor version byte.
        minor: u8,
    },
    /// The file ends inside a section.
    Truncated {
        /// Which section ran out.
        section: &'static str,
        /// Bytes the header said were there.
        want: usize,
        /// Bytes that remained.
        got: usize,
    },
    /// The JSON header is missing, malformed, or missing a field.
    Header(String),
    /// A `dtype`, `itype` or `ntype` this reader does not implement.
    Dtype(String),
    /// A `vartype` that is neither `SPIN` nor `BINARY`.
    UnknownVartype(String),
    /// A label that is neither an integer nor a string — a `dimod` tuple label, typically.
    Label(String),
    /// The body contradicts itself: non-monotone neighbourhood starts, an out-of-range neighbour,
    /// a label count that is not the variable count, or two copies of one edge that disagree.
    Corrupt(String),
    /// A text line that is not a comment, a header, or an `i j bias` triplet.
    Line {
        /// One-based line number.
        line: usize,
        /// The line itself.
        text: String,
    },
    /// A `p qubo` header whose declared count is not what the body held.
    ///
    /// Refused rather than solved, for [`crate::dimacs`]'s reason: a truncated file is otherwise a
    /// perfectly valid smaller instance whose optimum is not comparable with anyone else's.
    Count {
        /// `diagonal` or `coupler`.
        kind: &'static str,
        /// What the header said.
        declared: usize,
        /// What the body held.
        found: usize,
    },
}

impl core::fmt::Display for BqmError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            BqmError::Magic { got } => {
                write!(f, "expected the magic string {MAGIC:?}, found {got:?}")
            }
            BqmError::Version { major, minor } => write!(
                f,
                "this is a dimod BQM file of version {major}.{minor}; versions 1.x and 2.x are read"
            ),
            BqmError::Truncated { section, want, got } => {
                write!(f, "the {section} section wants {want} bytes and {got} remain")
            }
            BqmError::Header(why) => write!(f, "the JSON header is not usable: {why}"),
            BqmError::Dtype(name) => write!(
                f,
                "unsupported numeric type {name:?}; float32 and float64 biases and 8- to 64-bit \
                 integer indices are read"
            ),
            BqmError::UnknownVartype(name) => {
                write!(f, "vartype {name:?} is neither SPIN nor BINARY")
            }
            BqmError::Label(what) => write!(
                f,
                "variable label {what} is neither an integer nor a string. dimod also permits \
                 tuple labels (dwave_networkx coordinates, typically); flattening one would \
                 re-serialise as a different model, so it is refused instead"
            ),
            BqmError::Corrupt(why) => write!(f, "the model body contradicts itself: {why}"),
            BqmError::Line { line, text } => {
                write!(f, "line {line} is not a comment, a header, or an `i j bias` triplet: {text:?}")
            }
            BqmError::Count { kind, declared, found } => write!(
                f,
                "the `p qubo` header declares {declared} {kind} entries and the body holds {found}. \
                 A truncated file parses into a valid SMALLER instance whose optimum is not \
                 comparable with anyone else's, so this is refused rather than solved"
            ),
        }
    }
}

impl core::error::Error for BqmError {}

// ---- the binary format --------------------------------------------------------------------------

/// Numeric width and signedness of one of the header's three type names.
fn width_of(name: &str) -> Result<usize, BqmError> {
    let w = match name {
        "int8" | "uint8" => 1,
        "int16" | "uint16" => 2,
        "int32" | "uint32" => 4,
        "int64" | "uint64" => 8,
        _ => return Err(BqmError::Dtype(name.to_string())),
    };
    Ok(w)
}

fn read_int(b: &[u8], at: usize, width: usize) -> i64 {
    let mut v = 0u64;
    for k in 0..width {
        v |= u64::from(b[at + k]) << (8 * k);
    }
    // Sign-extend from `width` bytes. dimod writes int32 here, so a value that arrives negative is
    // corruption rather than a large index, and the caller rejects it as such.
    let shift = 64 - 8 * width;
    ((v << shift) as i64) >> shift
}

fn read_float(b: &[u8], at: usize, width: usize) -> f64 {
    if width == 4 {
        f64::from(f32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]]))
    } else {
        let mut a = [0u8; 8];
        a.copy_from_slice(&b[at..at + 8]);
        f64::from_le_bytes(a)
    }
}

/// Serialize as a `dimod` BQM file, version 2.0, `float64` biases.
///
/// Byte-for-byte what `dimod.BinaryQuadraticModel.to_file()` produces for the same model — pinned
/// by `the_bytes_we_write_are_the_bytes_dimod_wrote`, which compares against files `dimod` 0.12.21
/// actually emitted rather than against this encoder's own idea of the format.
///
/// Index widths are `int32` unless the model needs more, in which case `int64` is written and the
/// header says so.
#[must_use]
pub fn write_bqm(bqm: &Bqm) -> Vec<u8> {
    let m = bqm.canonical();
    let n = m.linear.len();
    let ni = m.quadratic.len();
    let nb = m.neighborhoods();
    let wide = n > i32::MAX as usize || 2 * ni > i32::MAX as usize;
    let (iname, iw) = if wide { ("int64", 8usize) } else { ("int32", 4usize) };
    let labeled = !m.is_index_labeled();

    let mut header = String::new();
    header.push_str("{\"dtype\": \"float64\", \"itype\": \"");
    header.push_str(iname);
    header.push_str("\", \"ntype\": \"");
    header.push_str(iname);
    header.push_str("\", \"shape\": [");
    header.push_str(&n.to_string());
    header.push_str(", ");
    header.push_str(&ni.to_string());
    header.push_str("], \"type\": \"BinaryQuadraticModel\", \"variables\": ");
    header.push_str(if labeled { "true" } else { "false" });
    header.push_str(", \"vartype\": \"");
    header.push_str(m.vartype.name());
    header.push_str("\"}");

    let mut out = Vec::new();
    out.extend_from_slice(&MAGIC);
    out.push(VERSION.0);
    out.push(VERSION.1);
    let len_at = out.len();
    out.extend_from_slice(&[0u8; 4]);
    let data_at = out.len();
    out.extend_from_slice(header.as_bytes());
    out.push(b'\n');
    if out.len() % 64 != 0 {
        out.resize(out.len() + (64 - out.len() % 64), b' ');
    }
    let header_len = (out.len() - data_at) as u32;
    out[len_at..len_at + 4].copy_from_slice(&header_len.to_le_bytes());

    out.extend_from_slice(&m.offset.to_le_bytes());
    let mut start = 0i64;
    for (i, &h) in m.linear.iter().enumerate() {
        out.extend_from_slice(&start.to_le_bytes()[..iw]);
        out.extend_from_slice(&h.to_le_bytes());
        start += nb[i].len() as i64;
    }
    for list in &nb {
        for &(j, b) in list {
            out.extend_from_slice(&(j as i64).to_le_bytes()[..iw]);
            out.extend_from_slice(&b.to_le_bytes());
        }
    }

    if labeled {
        let mut json = String::from("[");
        for (k, l) in m.labels.iter().enumerate() {
            if k > 0 {
                json.push_str(", ");
            }
            match l {
                Label::Index(v) => json.push_str(&v.to_string()),
                Label::Text(s) => json_string(s, &mut json),
            }
        }
        json.push(']');
        let mut data = json.len();
        if (data + 8) % 64 != 0 {
            data += 64 - (data + 8) % 64;
        }
        out.extend_from_slice(b"VARS");
        out.extend_from_slice(&(data as u32).to_le_bytes());
        out.extend_from_slice(json.as_bytes());
        out.resize(out.len() + (data - json.len()), b' ');
    }
    out
}

/// Read a `dimod` BQM file, version 1.x or 2.x, `float32` or `float64`.
///
/// # Errors
///
/// [`BqmError`]. A body that contradicts its own header — non-monotone neighbourhood starts, a
/// neighbour past the variable count, two copies of one edge that disagree — is [`BqmError::Corrupt`]
/// rather than a silently smaller model, and no input of any shape may panic.
pub fn read_bqm(bytes: &[u8]) -> Result<Bqm, BqmError> {
    if bytes.len() < 14 {
        return Err(if bytes.len() < 8 || bytes[..8] != MAGIC {
            BqmError::Magic { got: bytes[..bytes.len().min(8)].to_vec() }
        } else {
            BqmError::Truncated { section: "prefix", want: 14, got: bytes.len() }
        });
    }
    if bytes[..8] != MAGIC {
        return Err(BqmError::Magic { got: bytes[..8].to_vec() });
    }
    let (major, minor) = (bytes[8], bytes[9]);
    if major == 0 || major > 2 {
        return Err(BqmError::Version { major, minor });
    }
    let header_len = u32::from_le_bytes([bytes[10], bytes[11], bytes[12], bytes[13]]) as usize;
    // Checked additively: `14 + header_len` on a hostile length would wrap on a 32-bit target and
    // the slice below would then be in bounds by accident.
    if header_len > bytes.len() - 14 {
        return Err(BqmError::Truncated {
            section: "header",
            want: header_len,
            got: bytes.len() - 14,
        });
    }
    let head = json_parse(&bytes[14..14 + header_len])
        .ok_or_else(|| BqmError::Header("not well-formed JSON".to_string()))?;

    let field = |k: &str| head.get(k).ok_or_else(|| BqmError::Header(format!("no {k:?} field")));
    let dtype = field("dtype")?
        .as_str()
        .ok_or_else(|| BqmError::Header("\"dtype\" is not a string".to_string()))?;
    let dw = match dtype {
        "float32" => 4usize,
        "float64" => 8,
        other => return Err(BqmError::Dtype(other.to_string())),
    };
    let iw = width_of(
        field("itype")?
            .as_str()
            .ok_or_else(|| BqmError::Header("\"itype\" is not a string".to_string()))?,
    )?;
    let nw = width_of(
        field("ntype")?
            .as_str()
            .ok_or_else(|| BqmError::Header("\"ntype\" is not a string".to_string()))?,
    )?;
    let vname = field("vartype")?
        .as_str()
        .ok_or_else(|| BqmError::Header("\"vartype\" is not a string".to_string()))?;
    let vartype =
        Vartype::from_name(vname).ok_or_else(|| BqmError::UnknownVartype(vname.to_string()))?;
    let shape = field("shape")?
        .as_arr()
        .filter(|a| a.len() == 2)
        .ok_or_else(|| BqmError::Header("\"shape\" is not a pair".to_string()))?;
    let dim = |k: usize| -> Result<usize, BqmError> {
        let v = shape[k]
            .as_f64()
            .ok_or_else(|| BqmError::Header("\"shape\" holds a non-number".to_string()))?;
        if v < 0.0 || v > 2f64.powi(53) || v.fract() != 0.0 {
            return Err(BqmError::Header(format!("\"shape\" entry {v} is not a count")));
        }
        Ok(v as usize)
    };
    let (n, ni) = (dim(0)?, dim(1)?);

    let body = &bytes[14 + header_len..];
    // Computed in the widths the header declares, so a hostile `shape` cannot overflow into a
    // small number and pass the bounds check.
    let need = (dw as u128) + (n as u128) * (nw + dw) as u128 + 2 * (ni as u128) * (iw + dw) as u128;
    if need > body.len() as u128 {
        return Err(BqmError::Truncated {
            section: "biases",
            want: need.min(usize::MAX as u128) as usize,
            got: body.len(),
        });
    }

    let offset = read_float(body, 0, dw);
    let mut at = dw;
    let mut starts = Vec::with_capacity(n + 1);
    let mut linear = Vec::with_capacity(n);
    for _ in 0..n {
        starts.push(read_int(body, at, nw));
        at += nw;
        linear.push(read_float(body, at, dw));
        at += dw;
    }
    starts.push(2 * ni as i64);
    for k in 0..n {
        if starts[k] < 0 || starts[k] > starts[k + 1] {
            return Err(BqmError::Corrupt(format!(
                "variable {k}'s neighbourhood starts at {} and variable {}'s at {}",
                starts[k],
                k + 1,
                starts[k + 1]
            )));
        }
    }

    let mut quad: Vec<(usize, usize, f64)> = Vec::new();
    for v in 0..n {
        for k in starts[v]..starts[v + 1] {
            let e = at + (k as usize) * (iw + dw);
            let u = read_int(body, e, iw);
            let b = read_float(body, e + iw, dw);
            if u < 0 || u as usize >= n {
                return Err(BqmError::Corrupt(format!(
                    "variable {v} has a neighbour {u}, outside 0..{n}"
                )));
            }
            let u = u as usize;
            if u > v {
                quad.push((v, u, b));
            } else if u < v {
                // The mirror copy. dimod ignores it; checking it instead turns a corrupted file
                // into an error rather than into a model that is quietly half right.
                let want = quad
                    .binary_search_by(|p| (p.0, p.1).cmp(&(u, v)))
                    .map(|k| quad[k].2)
                    .map_err(|_| {
                        BqmError::Corrupt(format!("edge ({u},{v}) is stored once, not twice"))
                    })?;
                if want != b && !(want.is_nan() && b.is_nan()) {
                    return Err(BqmError::Corrupt(format!(
                        "edge ({u},{v}) is stored as {want} from one end and {b} from the other"
                    )));
                }
            } else {
                return Err(BqmError::Corrupt(format!("variable {v} is its own neighbour")));
            }
        }
    }
    if quad.len() != ni {
        return Err(BqmError::Corrupt(format!(
            "the header declares {ni} interactions and the neighbourhoods hold {}",
            quad.len()
        )));
    }

    let mut labels: Vec<Label> = (0..n as i64).map(Label::Index).collect();
    if major >= 2 {
        let want_labels = head.get("variables").and_then(Json::as_bool).unwrap_or(false);
        if want_labels {
            let tail = &body[at + 2 * ni * (iw + dw)..];
            if tail.len() < 8 || &tail[..4] != b"VARS" {
                return Err(BqmError::Truncated {
                    section: "VARS",
                    want: 8,
                    got: tail.len(),
                });
            }
            let len = u32::from_le_bytes([tail[4], tail[5], tail[6], tail[7]]) as usize;
            if len > tail.len() - 8 {
                return Err(BqmError::Truncated { section: "VARS", want: len, got: tail.len() - 8 });
            }
            labels = json_labels(
                &json_parse(&tail[8..8 + len])
                    .ok_or_else(|| BqmError::Header("the VARS array is not JSON".to_string()))?,
            )?;
        }
    } else if let Some(v) = head.get("variables") {
        // Version 1.0 keeps the labels in the header. An index-labelled 1.0 file still lists them.
        if v.as_arr().is_some() {
            labels = json_labels(v)?;
        }
    }
    if labels.len() != n {
        return Err(BqmError::Corrupt(format!(
            "the file declares {n} variables and lists {} labels",
            labels.len()
        )));
    }

    Ok(Bqm { vartype, linear, quadratic: quad, offset, labels })
}

fn json_labels(v: &Json) -> Result<Vec<Label>, BqmError> {
    let arr = v.as_arr().ok_or_else(|| BqmError::Label("the list is not an array".to_string()))?;
    let mut out = Vec::with_capacity(arr.len());
    for e in arr {
        match e {
            Json::Str(s) => out.push(Label::Text(s.clone())),
            Json::Num(x) if x.fract() == 0.0 && x.abs() <= 2f64.powi(53) => {
                out.push(Label::Index(*x as i64));
            }
            other => return Err(BqmError::Label(format!("{other:?}"))),
        }
    }
    Ok(out)
}

// ---- the QUBO coefficient-triplet text format ---------------------------------------------------

/// Write the QUBO coefficient-triplet text format: `qbsolv`'s `p qubo` header and `i j bias` lines.
///
/// A `SPIN` model is converted first, because this format has no vartype — it is a QUBO by
/// construction. **The constant rides in a `c offset` comment**, which [`read_triplets`] reads back
/// and every other reader ignores; a QUBO file has nowhere else to put it, and dropping it would
/// leave every energy wrong by a fixed amount while every optimum stayed right.
///
/// All `n` diagonal entries are written, zeros included, so the variable count survives a round
/// trip through a format whose only other record of it is the header count.
///
/// `dimod.serialization.coo.load` reads this if given `vartype=dimod.BINARY`: its own vartype
/// header must begin with `#`, and `qbsolv` comments begin with `c`.
#[must_use]
pub fn write_qubo(bqm: &Bqm) -> String {
    let m = bqm.to_vartype(Vartype::Binary).canonical();
    let n = m.linear.len();
    let mut s = String::new();
    s.push_str("c ferrotherm QUBO triplets, vartype=BINARY\n");
    s.push_str("c offset ");
    s.push_str(&fmt_f64(m.offset));
    s.push('\n');
    s.push_str(&format!("p qubo 0 {n} {n} {}\n", m.quadratic.len()));
    s.push_str("c nodes\n");
    for (i, &h) in m.linear.iter().enumerate() {
        s.push_str(&format!("{i} {i} {}\n", fmt_f64(h)));
    }
    s.push_str("c couplers\n");
    for &(u, v, b) in &m.quadratic {
        s.push_str(&format!("{u} {v} {}\n", fmt_f64(b)));
    }
    s
}

/// Rust's shortest round-tripping decimal, in positional notation.
///
/// `dimod`'s own COO reader matches `[+-]?\d*(\.\d+)?` and would reject `1e-3`, so exponent
/// notation is never emitted. `f64`'s `Display` never emits it either, which is why this is a
/// one-liner and a note rather than a formatter.
fn fmt_f64(v: f64) -> String {
    format!("{v}")
}

/// Read QUBO coefficient triplets: `qbsolv`'s `p qubo` files and `dimod`'s COO files alike.
///
/// Both are `i j bias` lines with a diagonal `i i bias` for a linear term. A `p qubo` header fixes
/// the variable count and its declared counts are checked; without one the count is one past the
/// largest index seen. `# vartype=SPIN` (`dimod`'s marker) and `c offset <v>` (this crate's) are
/// honoured from comments of either flavour; a file that says neither is a QUBO with no constant,
/// which is what the format means.
///
/// # Errors
///
/// [`BqmError::Line`] for a line that is not a comment, a header, or a triplet; [`BqmError::Count`]
/// when a `p qubo` header's declared counts disagree with the body.
pub fn read_triplets(text: &str) -> Result<Bqm, BqmError> {
    let mut vartype = Vartype::Binary;
    let mut offset = 0.0f64;
    let mut declared: Option<(usize, usize, usize)> = None;
    let mut diag: Vec<(usize, f64)> = Vec::new();
    let mut coup: Vec<(usize, usize, f64)> = Vec::new();
    let mut seen = 0usize;

    for (no, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }
        let bad = || BqmError::Line { line: no + 1, text: line.to_string() };
        if line.starts_with('c') || line.starts_with('#') {
            if let Some(v) = tagged(line, "vartype").and_then(Vartype::from_name) {
                vartype = v;
            }
            if let Some(v) = tagged(line, "offset").and_then(|t| t.parse::<f64>().ok()) {
                offset = v;
            }
            continue;
        }
        if line.starts_with('p') {
            let mut it = line.split_whitespace();
            it.next();
            if it.next() != Some("qubo") {
                return Err(bad());
            }
            let nums: Vec<&str> = it.collect();
            if nums.len() != 4 {
                return Err(bad());
            }
            let mut got = [0usize; 4];
            for (k, t) in nums.iter().enumerate() {
                got[k] = t.parse::<usize>().map_err(|_| bad())?;
            }
            declared = Some((got[1], got[2], got[3]));
            continue;
        }
        let mut it = line.split_whitespace();
        let (Some(a), Some(b), Some(c), None) = (it.next(), it.next(), it.next(), it.next()) else {
            return Err(bad());
        };
        let (Ok(i), Ok(j)) = (a.parse::<usize>(), b.parse::<usize>()) else {
            return Err(bad());
        };
        let Ok(w) = c.parse::<f64>() else {
            return Err(bad());
        };
        if !w.is_finite() {
            return Err(bad());
        }
        seen = seen.max(i.max(j) + 1);
        if i == j {
            diag.push((i, w));
        } else {
            coup.push((i.min(j), i.max(j), w));
        }
    }

    let n = match declared {
        Some((max_nodes, nd, ne)) => {
            if nd != diag.len() {
                return Err(BqmError::Count {
                    kind: "diagonal",
                    declared: nd,
                    found: diag.len(),
                });
            }
            if ne != coup.len() {
                return Err(BqmError::Count {
                    kind: "coupler",
                    declared: ne,
                    found: coup.len(),
                });
            }
            if seen > max_nodes {
                return Err(BqmError::Corrupt(format!(
                    "the header declares {max_nodes} nodes and an entry names variable {}",
                    seen - 1
                )));
            }
            max_nodes
        }
        None => seen,
    };

    let mut out = Bqm::new(vartype, n);
    out.offset = offset;
    // A diagonal `i i b` is the LINEAR bias of variable `i`, which is what `dimod`'s COO reader
    // does with it and what `qbsolv` calls a "node". It is NOT a self-interaction: routing it
    // through `Bqm::couple` would be right for a binary model and would fold it into the constant
    // for a spin one, leaving every linear bias zero and every non-uniform state mis-scored.
    // `dimods_own_coo_output_reads_as_the_model_dimod_printed` is the test that caught exactly that.
    for (i, w) in diag {
        out.bias(i, w);
    }
    for (u, v, w) in coup {
        out.couple(u, v, w);
    }
    out.canonicalize();
    Ok(out)
}

/// The token after `tag`, allowing `tag=value`, `tag: value` and `tag value`.
fn tagged<'a>(line: &'a str, tag: &str) -> Option<&'a str> {
    let p = line.find(tag)?;
    let mut rest = line[p + tag.len()..].trim_start();
    if let Some(r) = rest.strip_prefix('=') {
        rest = r.trim_start();
    } else if let Some(r) = rest.strip_prefix(':') {
        rest = r.trim_start();
    }
    let end = rest.find(char::is_whitespace).unwrap_or(rest.len());
    (end > 0).then(|| &rest[..end])
}

// ---- just enough JSON ---------------------------------------------------------------------------
//
// The header and the VARS array are JSON, and the subset they use is small: a flat object of
// strings, booleans and two-element number arrays, plus an array of integers or strings. Written
// out rather than pulled in, for the reason `crate::ommx` hand-rolls its protobuf -- this crate has
// no dependencies, and a header parser is a hundred lines.

/// A parsed JSON value.
#[derive(Clone, Debug, PartialEq)]
enum Json {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

impl Json {
    fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Obj(kv) => kv.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }
    fn as_str(&self) -> Option<&str> {
        match self {
            Json::Str(s) => Some(s),
            _ => None,
        }
    }
    fn as_f64(&self) -> Option<f64> {
        match self {
            Json::Num(v) => Some(*v),
            _ => None,
        }
    }
    fn as_bool(&self) -> Option<bool> {
        match self {
            Json::Bool(b) => Some(*b),
            _ => None,
        }
    }
    fn as_arr(&self) -> Option<&[Json]> {
        match self {
            Json::Arr(a) => Some(a),
            _ => None,
        }
    }
}

struct JsonP<'a> {
    b: &'a [u8],
    i: usize,
}

impl JsonP<'_> {
    fn ws(&mut self) {
        while self.i < self.b.len() && matches!(self.b[self.i], b' ' | b'\t' | b'\n' | b'\r') {
            self.i += 1;
        }
    }
    fn lit(&mut self, want: &[u8]) -> Option<()> {
        if self.b.len() - self.i >= want.len() && &self.b[self.i..self.i + want.len()] == want {
            self.i += want.len();
            Some(())
        } else {
            None
        }
    }
    fn hex4(&mut self) -> Option<u32> {
        let mut v = 0u32;
        for _ in 0..4 {
            let c = *self.b.get(self.i)?;
            let d = char::from(c).to_digit(16)?;
            v = v * 16 + d;
            self.i += 1;
        }
        Some(v)
    }
    fn string(&mut self) -> Option<String> {
        if *self.b.get(self.i)? != b'"' {
            return None;
        }
        self.i += 1;
        let mut units: Vec<u16> = Vec::new();
        loop {
            let c = *self.b.get(self.i)?;
            self.i += 1;
            match c {
                b'"' => break,
                b'\\' => {
                    let e = *self.b.get(self.i)?;
                    self.i += 1;
                    match e {
                        b'"' => units.push(u16::from(b'"')),
                        b'\\' => units.push(u16::from(b'\\')),
                        b'/' => units.push(u16::from(b'/')),
                        b'b' => units.push(8),
                        b'f' => units.push(12),
                        b'n' => units.push(10),
                        b'r' => units.push(13),
                        b't' => units.push(9),
                        b'u' => units.push(self.hex4()? as u16),
                        _ => return None,
                    }
                }
                _ => {
                    // Raw bytes are UTF-8; widen them here and let `from_utf16` put them back.
                    // `json.dumps` escapes everything non-ASCII, so this path is for other writers.
                    let mut buf = [0u8; 4];
                    let len = utf8_len(c)?;
                    buf[0] = c;
                    for k in 1..len {
                        buf[k] = *self.b.get(self.i)?;
                        self.i += 1;
                    }
                    let s = core::str::from_utf8(&buf[..len]).ok()?;
                    units.extend(s.encode_utf16());
                }
            }
        }
        String::from_utf16(&units).ok()
    }
    fn number(&mut self) -> Option<Json> {
        let start = self.i;
        while self.i < self.b.len()
            && matches!(self.b[self.i], b'-' | b'+' | b'.' | b'e' | b'E' | b'0'..=b'9')
        {
            self.i += 1;
        }
        core::str::from_utf8(&self.b[start..self.i]).ok()?.parse::<f64>().ok().map(Json::Num)
    }
    fn value(&mut self) -> Option<Json> {
        self.ws();
        match *self.b.get(self.i)? {
            b'{' => {
                self.i += 1;
                let mut kv = Vec::new();
                self.ws();
                if self.b.get(self.i) == Some(&b'}') {
                    self.i += 1;
                    return Some(Json::Obj(kv));
                }
                loop {
                    self.ws();
                    let k = self.string()?;
                    self.ws();
                    if *self.b.get(self.i)? != b':' {
                        return None;
                    }
                    self.i += 1;
                    let v = self.value()?;
                    kv.push((k, v));
                    self.ws();
                    match *self.b.get(self.i)? {
                        b',' => self.i += 1,
                        b'}' => {
                            self.i += 1;
                            return Some(Json::Obj(kv));
                        }
                        _ => return None,
                    }
                }
            }
            b'[' => {
                self.i += 1;
                let mut a = Vec::new();
                self.ws();
                if self.b.get(self.i) == Some(&b']') {
                    self.i += 1;
                    return Some(Json::Arr(a));
                }
                loop {
                    a.push(self.value()?);
                    self.ws();
                    match *self.b.get(self.i)? {
                        b',' => self.i += 1,
                        b']' => {
                            self.i += 1;
                            return Some(Json::Arr(a));
                        }
                        _ => return None,
                    }
                }
            }
            b'"' => self.string().map(Json::Str),
            b't' => {
                self.lit(b"true")?;
                Some(Json::Bool(true))
            }
            b'f' => {
                self.lit(b"false")?;
                Some(Json::Bool(false))
            }
            b'n' => {
                self.lit(b"null")?;
                Some(Json::Null)
            }
            _ => self.number(),
        }
    }
}

fn utf8_len(lead: u8) -> Option<usize> {
    match lead {
        0x00..=0x7F => Some(1),
        0xC2..=0xDF => Some(2),
        0xE0..=0xEF => Some(3),
        0xF0..=0xF4 => Some(4),
        _ => None,
    }
}

fn json_parse(b: &[u8]) -> Option<Json> {
    let mut p = JsonP { b, i: 0 };
    let v = p.value()?;
    p.ws();
    (p.i == p.b.len()).then_some(v)
}

/// Write `s` as a JSON string, escaping everything outside printable ASCII.
///
/// `dimod` decodes its own headers with `.decode('ascii')`, so a raw UTF-8 label would make the file
/// unreadable by the tool it is written for.
fn json_string(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (' '..='~').contains(&c) => out.push(c),
            c => {
                let mut buf = [0u16; 2];
                for u in c.encode_utf16(&mut buf) {
                    out.push_str(&format!("\\u{u:04x}"));
                }
            }
        }
    }
    out.push('"');
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exact::Elimination;

    /// The exact bytes `dimod` 0.12.21 wrote, for five models chosen to separate the readings of
    /// the format that a symmetric example cannot.
    ///
    /// `A` is a triangle, where every degree is 2 and every neighbourhood start is even, so it
    /// cannot distinguish "start" from "degree". `C` has degrees 3,1,2,0 -- an isolated variable, an
    /// explicit zero interaction -- and it can. `B` and `E` carry string and integer labels; `D` is
    /// empty.
    const A_SPIN_TRIANGLE: &str = concat!(
        "44494d4f4442514d0200b20000007b226474797065223a2022666c6f61743634222c20226974797065223a2022696e74",
        "3332222c20226e74797065223a2022696e743332222c20227368617065223a205b332c20335d2c202274797065223a20",
        "2242696e6172795175616472617469634d6f64656c222c20227661726961626c6573223a2066616c73652c2022766172",
        "74797065223a20225350494e227d0a202020202020202020202020202020202020202020202020202020202020202020",
        "0000000000000a4000000000000000000000f8bf02000000000000000000d03f04000000000000000000000001000000",
        "000000000000004002000000000000000000e83f00000000000000000000004002000000000000000000e0bf00000000",
        "000000000000e83f01000000000000000000e0bf",
    );
    const B_LABELED_BINARY: &str = concat!(
        "44494d4f4442514d0200b20000007b226474797065223a2022666c6f61743634222c20226974797065223a2022696e74",
        "3332222c20226e74797065223a2022696e743332222c20227368617065223a205b322c20315d2c202274797065223a20",
        "2242696e6172795175616472617469634d6f64656c222c20227661726961626c6573223a20747275652c202276617274",
        "797065223a202242494e415259227d0a2020202020202020202020202020202020202020202020202020202020202020",
        "000000000000000000000000000000000000f03f0100000000000000000000c001000000000000000000e03f00000000",
        "000000000000e03f56415253380000005b2261222c202262225d20202020202020202020202020202020202020202020",
        "202020202020202020202020202020202020202020202020",
    );
    const C_RAGGED_BINARY: &str = concat!(
        "44494d4f4442514d0200b20000007b226474797065223a2022666c6f61743634222c20226974797065223a2022696e74",
        "3332222c20226e74797065223a2022696e743332222c20227368617065223a205b342c20325d2c202274797065223a20",
        "2242696e6172795175616472617469634d6f64656c222c20227661726961626c6573223a2066616c73652c2022766172",
        "74797065223a202242494e415259227d0a20202020202020202020202020202020202020202020202020202020202020",
        "000000000000c0bf00000000000000000000f43f0100000000000000000000000200000000000000000008c004000000",
        "0000000000000000020000000000000000000000020000000000000000001ec000000000000000000000000001000000",
        "0000000000001ec0",
    );
    const D_EMPTY_SPIN: &str = concat!(
        "44494d4f4442514d0200b20000007b226474797065223a2022666c6f61743634222c20226974797065223a2022696e74",
        "3332222c20226e74797065223a2022696e743332222c20227368617065223a205b302c20305d2c202274797065223a20",
        "2242696e6172795175616472617469634d6f64656c222c20227661726961626c6573223a2066616c73652c2022766172",
        "74797065223a20225350494e227d0a202020202020202020202020202020202020202020202020202020202020202020",
        "0000000000000000",
    );
    const E_INT_LABELED_SPIN: &str = concat!(
        "44494d4f4442514d0200b20000007b226474797065223a2022666c6f61743634222c20226974797065223a2022696e74",
        "3332222c20226e74797065223a2022696e743332222c20227368617065223a205b322c20315d2c202274797065223a20",
        "2242696e6172795175616472617469634d6f64656c222c20227661726961626c6573223a20747275652c202276617274",
        "797065223a20225350494e227d0a20202020202020202020202020202020202020202020202020202020202020202020",
        "000000000000e03f00000000000000000000f03f01000000000000000000f0bf01000000000000000000084000000000",
        "000000000000084056415253380000005b352c20325d2020202020202020202020202020202020202020202020202020",
        "202020202020202020202020202020202020202020202020",
    );

    fn unhex(s: &str) -> Vec<u8> {
        let b = s.as_bytes();
        (0..b.len() / 2)
            .map(|k| u8::from_str_radix(core::str::from_utf8(&b[2 * k..2 * k + 2]).unwrap(), 16).unwrap())
            .collect()
    }

    /// The five models the fixtures encode, written out by hand from what `dimod` printed.
    fn fixtures() -> Vec<(&'static str, &'static str, Bqm)> {
        let mut a = Bqm::new(Vartype::Spin, 3);
        a.linear = vec![-1.5, 0.25, 0.0];
        a.quadratic = vec![(0, 1, 2.0), (0, 2, 0.75), (1, 2, -0.5)];
        a.offset = 3.25;

        let mut b = Bqm::new(Vartype::Binary, 2);
        b.linear = vec![1.0, -2.0];
        b.quadratic = vec![(0, 1, 0.5)];
        b.labels = vec![Label::Text("a".into()), Label::Text("b".into())];

        let mut c = Bqm::new(Vartype::Binary, 4);
        c.linear = vec![1.25, 0.0, -3.0, 0.0];
        c.quadratic = vec![(0, 2, 0.0), (1, 2, -7.5)];
        c.offset = -0.125;

        let d = Bqm::new(Vartype::Spin, 0);

        let mut e = Bqm::new(Vartype::Spin, 2);
        e.linear = vec![1.0, -1.0];
        e.quadratic = vec![(0, 1, 3.0)];
        e.offset = 0.5;
        e.labels = vec![Label::Index(5), Label::Index(2)];

        vec![
            ("A spin triangle", A_SPIN_TRIANGLE, a),
            ("B string labels", B_LABELED_BINARY, b),
            ("C ragged degrees", C_RAGGED_BINARY, c),
            ("D empty", D_EMPTY_SPIN, d),
            ("E integer labels", E_INT_LABELED_SPIN, e),
        ]
    }

    /// The oracle is `dimod` itself: files it actually wrote, parsed back to the coefficients it
    /// printed.
    #[test]
    fn a_file_dimod_wrote_reads_back_as_the_model_dimod_printed() {
        for (name, hex, want) in fixtures() {
            let got = read_bqm(&unhex(hex)).unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(got, want, "{name}");
        }
    }

    /// And the other direction, byte for byte.
    ///
    /// This is the check that would survive me having misread the layout: an encoder and a decoder
    /// that agree with each other prove nothing, and the ragged fixture `C` is here because a
    /// triangle cannot tell a neighbourhood START from a DEGREE -- `dimod`'s own docstring says the
    /// first and its writer is named for the second.
    #[test]
    fn the_bytes_we_write_are_the_bytes_dimod_wrote() {
        for (name, hex, model) in fixtures() {
            let want = unhex(hex);
            let got = write_bqm(&model);
            assert_eq!(got.len(), want.len(), "{name}: length");
            if got != want {
                let at = got.iter().zip(&want).position(|(a, b)| a != b).unwrap();
                panic!("{name}: first difference at byte {at}: {:#04x} vs {:#04x}", got[at], want[at]);
            }
        }
    }

    /// Every state, both vartypes: the substitution moves every coefficient and the energies agree.
    ///
    /// Enumerated rather than argued, because the algebra `a_i = 2h_i − 2Σ J_ij` is exactly the
    /// place a sign or a factor of two hides and still produces a plausible model.
    #[test]
    fn vartype_conversion_preserves_the_energy_of_every_state() {
        for (name, _, m) in fixtures() {
            let n = m.num_variables();
            let spin = m.to_vartype(Vartype::Spin);
            let binary = m.to_vartype(Vartype::Binary);
            for mask in 0u32..(1u32 << n) {
                let s: Vec<i8> =
                    (0..n).map(|i| if mask >> i & 1 == 1 { 1i8 } else { -1 }).collect();
                let x = binary_state(&s);
                assert_eq!(spin_state(&x), s, "{name}: the state maps are not inverse");
                let (a, b) = (spin.energy(&s), binary.energy(&x));
                assert!((a - b).abs() < 1e-12, "{name}: state {mask}: spin {a} vs binary {b}");
                // and the original scores it too, in whichever vartype it already was
                let own = if m.vartype == Vartype::Spin { m.energy(&s) } else { m.energy(&x) };
                assert!((a - own).abs() < 1e-12, "{name}: state {mask}: {a} vs {own}");
            }
        }
    }

    /// The failure a solver test cannot see: dropping the offset leaves every ranking intact.
    ///
    /// This is the reason the constant is carried rather than returned as advice. The assertion is
    /// two-sided on purpose -- the ranking is IDENTICAL, so no optimiser notices, and every absolute
    /// value is wrong by the same nonzero amount, so every comparison against a published objective
    /// is wrong too.
    #[test]
    fn the_offset_is_invisible_to_the_ranking_and_fatal_to_the_value() {
        let mut m = Bqm::new(Vartype::Spin, 4);
        m.linear = vec![0.3, -1.1, 0.7, 0.0];
        m.quadratic = vec![(0, 1, 1.0), (1, 2, -2.0), (2, 3, 0.5), (0, 3, 0.25)];
        let binary = m.to_vartype(Vartype::Binary);
        let mut dropped = binary.clone();
        dropped.offset = 0.0;
        assert!(binary.offset.abs() > 1e-9, "the substitution must produce a constant here");

        let mut with: Vec<(f64, u32)> = Vec::new();
        let mut without: Vec<(f64, u32)> = Vec::new();
        for mask in 0u32..16 {
            let s: Vec<i8> = (0..4).map(|i| if mask >> i & 1 == 1 { 1i8 } else { -1 }).collect();
            let x = binary_state(&s);
            let (a, b) = (binary.energy(&x), dropped.energy(&x));
            assert!(
                (a - b - binary.offset).abs() < 1e-12,
                "the difference must be exactly the constant"
            );
            assert!((a - m.energy(&s)).abs() < 1e-12);
            with.push((a, mask));
            without.push((b, mask));
        }
        with.sort_by(|p, q| p.0.total_cmp(&q.0));
        without.sort_by(|p, q| p.0.total_cmp(&q.0));
        let order: Vec<u32> = with.iter().map(|p| p.1).collect();
        let order_without: Vec<u32> = without.iter().map(|p| p.1).collect();
        assert_eq!(order, order_without, "a dropped constant is invisible to every ranking");
        assert!(
            (with[0].0 - without[0].0).abs() > 1e-9,
            "and it is exactly what a published objective value would disagree about"
        );
    }

    /// A model written and re-read is the same model, for both vartypes and both label kinds.
    #[test]
    fn a_written_model_re_reads_identical() {
        for (name, _, m) in fixtures() {
            let round = read_bqm(&write_bqm(&m)).unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(round, m.canonical(), "{name}: binary round trip");
        }
        // and one the fixtures do not cover: a label that needs escaping, and a duplicate term
        let mut m = Bqm::new(Vartype::Binary, 3);
        m.labels = vec![
            Label::Text("q\"0\\".into()),
            Label::Text("β".into()),
            Label::Index(-7),
        ];
        m.couple(0, 1, 1.5);
        m.couple(1, 0, 0.5); // the same edge again, which canonicalising sums
        m.couple(2, 2, 4.0); // a self term, which for a binary variable is linear
        m.bias(0, -1.0);
        let canon = m.canonical();
        assert_eq!(canon.quadratic, vec![(0, 1, 2.0)]);
        assert_eq!(canon.linear, vec![-1.0, 0.0, 4.0]);
        let round = read_bqm(&write_bqm(&m)).unwrap();
        assert_eq!(round, canon);
    }

    /// The text format, both directions, against the same energies.
    #[test]
    fn the_qubo_text_round_trips_and_scores_identically() {
        for (name, _, m) in fixtures() {
            let text = write_qubo(&m);
            let back = read_triplets(&text).unwrap_or_else(|e| panic!("{name}: {e}\n{text}"));
            let want = m.to_vartype(Vartype::Binary).canonical();
            assert_eq!(back.vartype, Vartype::Binary, "{name}");
            assert_eq!(back.linear, want.linear, "{name}: linear");
            assert_eq!(back.quadratic, want.quadratic, "{name}: quadratic");
            assert!((back.offset - want.offset).abs() < 1e-12, "{name}: offset");
            let n = m.num_variables();
            for mask in 0u32..(1u32 << n) {
                let x: Vec<i8> = (0..n).map(|i| i8::from(mask >> i & 1 == 1)).collect();
                let s = spin_state(&x);
                let a = back.energy(&x);
                let b = if m.vartype == Vartype::Spin { m.energy(&s) } else { m.energy(&x) };
                assert!((a - b).abs() < 1e-12, "{name}: state {mask}: {a} vs {b}");
            }
        }
    }

    /// `dimod`'s own COO output, which has no `p qubo` header and states its vartype in a `#`
    /// comment, read as the model `dimod` printed for it.
    ///
    /// The string is verbatim `dimod.serialization.coo.dumps(bqm, vartype_header=True)` for the
    /// triangle of fixture `A` with its constant removed -- COO has nowhere to put one.
    #[test]
    fn dimods_own_coo_output_reads_as_the_model_dimod_printed() {
        let text = "# vartype=SPIN\n0 0 -1.500000\n0 1 2.000000\n0 2 0.750000\n1 1 0.250000\n1 2 -0.500000";
        let got = read_triplets(text).unwrap();
        assert_eq!(got.vartype, Vartype::Spin);
        assert_eq!(got.linear, vec![-1.5, 0.25, 0.0]);
        assert_eq!(got.quadratic, vec![(0, 1, 2.0), (0, 2, 0.75), (1, 2, -0.5)]);
        assert_eq!(got.offset, 0.0);
    }

    /// The graph bridge scores every state exactly as `dimod` would, sign convention included.
    ///
    /// `Graph`'s energy has minus signs where `dimod`'s has plus signs. An unnegated bridge agrees
    /// on the magnitude of every coefficient and ranks every state backwards, which is why this
    /// checks values over the whole state space rather than checking an optimum.
    #[test]
    fn the_graph_bridge_negates_and_scores_every_state() {
        let g = crate::ising::lattice2d(3, 1.0);
        let bqm = Bqm::from_graph(&g);
        assert_eq!(bqm.vartype, Vartype::Spin);
        for mask in 0u32..(1u32 << g.n) {
            let s: Vec<i8> = (0..g.n).map(|i| if mask >> i & 1 == 1 { 1i8 } else { -1 }).collect();
            let (a, b) = (bqm.energy(&s), g.energy(&s));
            assert!((a - b).abs() < 1e-12, "state {mask}: bqm {a} vs graph {b}");
        }
        let (back, off) = bqm.to_graph();
        assert_eq!(off, 0.0);
        for mask in 0u32..(1u32 << g.n) {
            let s: Vec<i8> = (0..g.n).map(|i| if mask >> i & 1 == 1 { 1i8 } else { -1 }).collect();
            assert!((back.energy(&s) - g.energy(&s)).abs() < 1e-12);
        }
    }

    /// A QUBO read from text, minimised exactly by variable elimination, against brute force.
    ///
    /// The graph bridge is where the offset and the sign both have to be right at once: the ground
    /// STATE would survive either error alone, so this compares the ground ENERGY in the QUBO's own
    /// units against every one of its 2^n values.
    #[test]
    fn the_ground_energy_of_an_imported_qubo_matches_brute_force() {
        let text = "c a hand-written qubo\np qubo 0 6 6 7\n\
                    0 0 -3.5\n1 1 2.0\n2 2 0.25\n3 3 -1.0\n4 4 0.0\n5 5 4.5\n\
                    0 1 -2.0\n0 2 1.5\n1 3 -4.0\n2 4 3.0\n3 5 -0.5\n4 5 2.25\n1 4 -1.25\n";
        let q = read_triplets(text).unwrap();
        assert_eq!(q.num_variables(), 6);
        assert_eq!(q.vartype, Vartype::Binary);

        let (g, off) = q.to_graph();
        let exact = Elimination::default().ground_state(&g).unwrap();
        let got = exact.ground_energy.unwrap() + off;

        let mut best = f64::INFINITY;
        for mask in 0u32..(1u32 << 6) {
            let x: Vec<i8> = (0..6).map(|i| i8::from(mask >> i & 1 == 1)).collect();
            best = best.min(q.energy(&x));
        }
        assert!((got - best).abs() < 1e-9, "elimination {got} vs brute force {best}");

        // and the state elimination returned is one that attains it
        let x = binary_state(&exact.ground_state.unwrap());
        assert!((q.energy(&x) - best).abs() < 1e-9);
    }

    /// No input of any shape may panic, and a truncated file must not read as a smaller model.
    ///
    /// Every prefix of a valid file, plus a hostile header length -- the shape that made
    /// `crate::ommx` abort its caller's process before its own reader was hardened.
    #[test]
    fn malformed_input_is_refused_rather_than_panicking() {
        let full = unhex(A_SPIN_TRIANGLE);
        for cut in 0..full.len() {
            assert!(
                read_bqm(&full[..cut]).is_err(),
                "a {cut}-byte prefix of a {}-byte file must not parse",
                full.len()
            );
        }
        assert!(read_bqm(&full).is_ok());

        let mut hostile = full.clone();
        hostile[10..14].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(matches!(read_bqm(&hostile), Err(BqmError::Truncated { .. })));

        let mut wrong_magic = full.clone();
        wrong_magic[0] = b'X';
        assert!(matches!(read_bqm(&wrong_magic), Err(BqmError::Magic { .. })));

        let mut future = full.clone();
        future[8] = 9;
        assert!(matches!(read_bqm(&future), Err(BqmError::Version { major: 9, .. })));

        // a shape that claims more interactions than the body holds
        let mut lying = full.clone();
        let at = lying.windows(9).position(|w| w == b"[3, 3], \"").unwrap();
        lying[at + 4] = b'9';
        assert!(read_bqm(&lying).is_err(), "a shape the body cannot fill must be refused");

        // text
        assert!(matches!(read_triplets("0 1"), Err(BqmError::Line { line: 1, .. })));
        assert!(matches!(read_triplets("0 1 nope"), Err(BqmError::Line { .. })));
        assert!(matches!(read_triplets("0 1 inf"), Err(BqmError::Line { .. })));
        assert!(matches!(
            read_triplets("p qubo 0 2 1 1\n0 0 1\n"),
            Err(BqmError::Count { kind: "coupler", declared: 1, found: 0 })
        ));
        assert!(matches!(
            read_triplets("p qubo 0 2 0 1\n0 5 1\n"),
            Err(BqmError::Corrupt(_))
        ));
    }

    /// Version 1.0 keeps the labels in the header; it is still read.
    ///
    /// The bytes are `bqm.to_file(version=1)` for the triangle, whose body is byte-identical to the
    /// 2.0 file -- only the header differs, which is the whole content of the version bump.
    #[test]
    fn a_version_one_file_is_read_too() {
        let v1 = concat!(
            "44494d4f4442514d0100b20000007b226474797065223a2022666c6f61743634222c20226974797065223a2022696e74",
            "3332222c20226e74797065223a2022696e743332222c20227368617065223a205b332c20335d2c202274797065223a20",
            "2242696e6172795175616472617469634d6f64656c222c20227661726961626c6573223a205b302c20312c20325d2c20",
            "2276617274797065223a20225350494e227d0a2020202020202020202020202020202020202020202020202020202020",
            "0000000000000a4000000000000000000000f8bf02000000000000000000d03f04000000000000000000000001000000",
            "000000000000004002000000000000000000e83f00000000000000000000004002000000000000000000e0bf00000000",
            "000000000000e83f01000000000000000000e0bf",
        );
        let got = read_bqm(&unhex(v1)).unwrap();
        let want = fixtures().into_iter().next().unwrap().2;
        assert_eq!(got, want);
    }

    /// A `float32` file reads as the `f64` values `float32` can hold.
    #[test]
    fn a_float32_file_is_read_at_its_own_precision() {
        // dimod's `BinaryQuadraticModel({0: 1.0}, {}, 0.0, SPIN, dtype='float32')`
        let f32_file = concat!(
            "44494d4f4442514d0200b20000007b226474797065223a2022666c6f61743332222c20226974797065223a2022696e74",
            "3332222c20226e74797065223a2022696e743332222c20227368617065223a205b312c20305d2c202274797065223a20",
            "2242696e6172795175616472617469634d6f64656c222c20227661726961626c6573223a2066616c73652c2022766172",
            "74797065223a20225350494e227d0a202020202020202020202020202020202020202020202020202020202020202020",
            "00000000000000000000803f",
        );
        let got = read_bqm(&unhex(f32_file)).unwrap();
        assert_eq!(got.linear, vec![1.0]);
        assert_eq!(got.offset, 0.0);
        assert_eq!(got.vartype, Vartype::Spin);
    }

    /// A tuple label is refused by name rather than flattened.
    #[test]
    fn a_tuple_label_is_refused() {
        // the labelled fixture with `["a", "b"]` replaced by `[[0, 1], 2]`, same length
        let mut bytes = unhex(B_LABELED_BINARY);
        let at = bytes.windows(10).position(|w| w == b"[\"a\", \"b\"]").unwrap();
        bytes[at..at + 10].copy_from_slice(b"[[0, 1],2]");
        assert!(matches!(read_bqm(&bytes), Err(BqmError::Label(_))));
    }
}
