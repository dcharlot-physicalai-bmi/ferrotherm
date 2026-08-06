//! The `.ftp` program format.
//!
//! `(J, h, coloring, schedule)` is the de facto interchange of the whole probabilistic-computing
//! field, and nobody has specified it: it lives as scattered `.mat` files beside a `colorMap.csv`,
//! and every vendor ships a different upload shape for the same matrix. This is that format,
//! written down.
//!
//! It describes a **program**, not a problem. A problem is variables, an objective and constraints;
//! a program additionally says how it is to be run — which spins update together, at what
//! temperatures, with which penalties ramping, what to observe, and what an operation costs. That
//! difference is why this exists rather than an adoption of somebody's instance format.
//!
//! # Format
//!
//! Line-oriented UTF-8 text. One directive per line, fields separated by whitespace. Blank lines
//! are ignored; `#` begins a comment that runs to end of line. Order within a file is free except
//! that `ftp` must come first. Text, not binary, because the audience currently exchanges MATLAB
//! files nobody outside the sending lab can read; a format you can `grep`, `diff` and read in a
//! terminal is worth more here than a few saved bytes.
//!
//! ```text
//! ftp 1                     format version. must be the first directive.
//! name <string>             optional label, no whitespace.
//! spins <n>                 number of spins. required.
//! bias <i> <h>              external field on spin i. omit for zero.
//! factor <w> <v>...         energy term -w * prod(s_v). arity >= 1.
//! color <c> <i>...          spins in colour class c; these update together.
//! encode <base> <k> <kind>  provenance: spins [base, base+width) hold one k-valued variable,
//!                           spelled onehot | binary | domainwall.
//! stage <beta> <sweeps> <domain_wall> <copy>    one rung of the schedule, in order.
//! observe <name>            a quantity to reduce inside the sampling loop.
//! target <name>             intended backend or device topology.
//! price <name>              price table for the energy ledger.
//! ```
//!
//! Numbers are written with Rust's shortest round-tripping representation, so a parse of a write is
//! bit-identical to the original and a write of a parse is byte-identical to the input.
//!
//! # Example
//!
//! ```text
//! ftp 1
//! name frustrated-ring
//! spins 5
//! factor -1 0 1
//! factor -1 1 2
//! factor -1 2 3
//! factor -1 3 4
//! factor -1 4 0
//! stage 0.05 40 1 1
//! stage 4 40 1 1
//! observe energy
//! target cpu
//! price z1_spice
//! ```

use crate::encode::Encoding;
use crate::factor::Factor;
use crate::graph::{Graph, GraphBuilder};
use crate::schedule::{Penalties, Schedule, Stage};

/// Current format version.
pub const VERSION: u32 = 1;

/// Where a group of spins came from.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EncodedVar {
    pub base: usize,
    pub k: usize,
    pub encoding: Encoding,
}

/// A complete program.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Program {
    pub name: Option<String>,
    pub spins: usize,
    pub bias: Vec<(usize, f64)>,
    pub factors: Vec<Factor>,
    pub colors: Vec<Vec<usize>>,
    pub encodings: Vec<EncodedVar>,
    pub schedule: Schedule,
    pub observe: Vec<String>,
    pub target: Option<String>,
    pub price: Option<String>,
}

/// Why a program could not be read.
#[derive(Clone, Debug, PartialEq)]
pub struct FtpError {
    pub line: usize,
    pub message: String,
}

impl core::fmt::Display for FtpError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "line {}: {}", self.line, self.message)
    }
}

fn err<T>(line: usize, message: impl Into<String>) -> Result<T, FtpError> {
    Err(FtpError { line, message: message.into() })
}

impl Program {
    /// Build a program from a graph and a schedule.
    pub fn from_graph(g: &Graph, schedule: &Schedule) -> Program {
        let mut factors = Vec::new();
        for i in 0..g.n {
            for k in g.offset[i]..g.offset[i + 1] {
                let j = g.nbr[k] as usize;
                if j > i {
                    // couplings are pairwise factors; the arity-2 case is the Ising term
                    factors.push(Factor::new(&[i, j], g.w[k], g.n).expect("graph edges are valid"));
                }
            }
        }
        Program {
            name: None,
            spins: g.n,
            bias: (0..g.n).filter(|&i| g.h[i] != 0.0).map(|i| (i, g.h[i])).collect(),
            factors,
            colors: g.classes.iter().map(|c| c.iter().map(|&x| x as usize).collect()).collect(),
            encodings: Vec::new(),
            schedule: schedule.clone(),
            observe: Vec::new(),
            target: None,
            price: None,
        }
    }

    /// Rebuild a graph from this program. Factors of arity above two are refused here rather than
    /// silently dropped; lowering them to pairwise is a separate pass with its own ancillas.
    pub fn to_graph(&self) -> Result<Graph, FtpError> {
        let mut b = GraphBuilder::new(self.spins);
        for &(i, h) in &self.bias {
            b.bias(i, h);
        }
        for f in &self.factors {
            let vars: Vec<usize> = f.vars().collect();
            match vars.len() {
                1 => b.bias(vars[0], f.weight()),
                2 => b.couple(vars[0], vars[1], f.weight()),
                n => {
                    return err(
                        0,
                        format!(
                            "factor of arity {n} cannot become a graph directly; lower it to \
                             pairwise first"
                        ),
                    )
                }
            }
        }
        Ok(b.build())
    }

    /// Serialise. The output re-parses to an equal program and re-writes byte-identically.
    pub fn to_ftp(&self) -> String {
        let mut o = String::new();
        o.push_str(&format!("ftp {VERSION}\n"));
        if let Some(n) = &self.name {
            o.push_str(&format!("name {n}\n"));
        }
        o.push_str(&format!("spins {}\n", self.spins));
        for (i, h) in &self.bias {
            o.push_str(&format!("bias {i} {h}\n"));
        }
        for f in &self.factors {
            o.push_str(&format!("factor {}", f.weight()));
            for v in f.vars() {
                o.push_str(&format!(" {v}"));
            }
            o.push('\n');
        }
        for (c, class) in self.colors.iter().enumerate() {
            o.push_str(&format!("color {c}"));
            for i in class {
                o.push_str(&format!(" {i}"));
            }
            o.push('\n');
        }
        for e in &self.encodings {
            let kind = match e.encoding {
                Encoding::OneHot => "onehot",
                Encoding::Binary => "binary",
                Encoding::DomainWall => "domainwall",
            };
            o.push_str(&format!("encode {} {} {kind}\n", e.base, e.k));
        }
        for s in self.schedule.stages() {
            o.push_str(&format!(
                "stage {} {} {} {}\n",
                s.beta, s.sweeps, s.penalties.domain_wall, s.penalties.copy
            ));
        }
        for ob in &self.observe {
            o.push_str(&format!("observe {ob}\n"));
        }
        if let Some(t) = &self.target {
            o.push_str(&format!("target {t}\n"));
        }
        if let Some(p) = &self.price {
            o.push_str(&format!("price {p}\n"));
        }
        o
    }

    /// Parse. Errors carry the line number, because a format nobody can debug is a format nobody
    /// adopts.
    pub fn from_ftp(text: &str) -> Result<Program, FtpError> {
        let mut p = Program::default();
        let mut saw_header = false;
        let mut saw_spins = false;

        for (ln, raw) in text.lines().enumerate() {
            let line = ln + 1;
            let body = raw.split('#').next().unwrap_or("").trim();
            if body.is_empty() {
                continue;
            }
            let mut it = body.split_whitespace();
            let directive = it.next().unwrap();

            if !saw_header && directive != "ftp" {
                return err(line, format!("expected `ftp <version>` first, found `{directive}`"));
            }

            match directive {
                "ftp" => {
                    let v: u32 = num(it.next(), line, "ftp version")?;
                    if v != VERSION {
                        return err(line, format!("unsupported format version {v}, expected {VERSION}"));
                    }
                    saw_header = true;
                }
                "name" => p.name = Some(it.next().unwrap_or("").to_string()),
                "spins" => {
                    p.spins = num(it.next(), line, "spins")?;
                    saw_spins = true;
                }
                "bias" => {
                    let i: usize = num(it.next(), line, "bias index")?;
                    let h: f64 = num(it.next(), line, "bias value")?;
                    check_spin(i, p.spins, saw_spins, line)?;
                    p.bias.push((i, h));
                }
                "factor" => {
                    let w: f64 = num(it.next(), line, "factor weight")?;
                    let mut vars = Vec::new();
                    for tok in it {
                        let v: usize = tok
                            .parse()
                            .map_err(|_| FtpError { line, message: format!("bad variable `{tok}`") })?;
                        check_spin(v, p.spins, saw_spins, line)?;
                        vars.push(v);
                    }
                    let f = Factor::new(&vars, w, p.spins)
                        .map_err(|e| FtpError { line, message: e.to_string() })?;
                    p.factors.push(f);
                }
                "color" => {
                    let c: usize = num(it.next(), line, "colour index")?;
                    let mut class = Vec::new();
                    for tok in it {
                        let v: usize = tok
                            .parse()
                            .map_err(|_| FtpError { line, message: format!("bad spin `{tok}`") })?;
                        check_spin(v, p.spins, saw_spins, line)?;
                        class.push(v);
                    }
                    if p.colors.len() <= c {
                        p.colors.resize(c + 1, Vec::new());
                    }
                    p.colors[c] = class;
                }
                "encode" => {
                    let base: usize = num(it.next(), line, "encode base")?;
                    let k: usize = num(it.next(), line, "encode k")?;
                    let kind = it.next().unwrap_or("");
                    let encoding = match kind {
                        "onehot" => Encoding::OneHot,
                        "binary" => Encoding::Binary,
                        "domainwall" => Encoding::DomainWall,
                        other => {
                            return err(
                                line,
                                format!(
                                    "unknown encoding `{other}`; expected onehot, binary or \
                                     domainwall"
                                ),
                            )
                        }
                    };
                    if k < 2 {
                        return err(line, format!("a {k}-valued variable is a constant"));
                    }
                    p.encodings.push(EncodedVar { base, k, encoding });
                }
                "stage" => {
                    let beta: f64 = num(it.next(), line, "stage beta")?;
                    let sweeps: usize = num(it.next(), line, "stage sweeps")?;
                    let dw: f64 = num(it.next(), line, "domain-wall penalty")?;
                    let cp: f64 = num(it.next(), line, "copy penalty")?;
                    p.schedule.push(Stage {
                        beta,
                        sweeps,
                        penalties: Penalties { domain_wall: dw, copy: cp },
                    });
                }
                "observe" => p.observe.push(it.next().unwrap_or("").to_string()),
                "target" => p.target = Some(it.next().unwrap_or("").to_string()),
                "price" => p.price = Some(it.next().unwrap_or("").to_string()),
                other => return err(line, format!("unknown directive `{other}`")),
            }
        }

        if !saw_header {
            return err(0, "empty or missing `ftp <version>` header");
        }
        if !saw_spins {
            return err(0, "missing `spins <n>`");
        }
        Ok(p)
    }

    /// A stable digest of the canonical text, for asserting that two runs ran the same program.
    ///
    /// FNV-1a over the serialisation. Not a security hash and not claimed to be one; it answers
    /// "is this the same program" and nothing else.
    pub fn digest(&self) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for b in self.to_ftp().as_bytes() {
            h ^= *b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        h
    }
}

fn num<T: core::str::FromStr>(tok: Option<&str>, line: usize, what: &str) -> Result<T, FtpError> {
    let t = tok.ok_or(FtpError { line, message: format!("missing {what}") })?;
    t.parse().map_err(|_| FtpError { line, message: format!("bad {what}: `{t}`") })
}

fn check_spin(i: usize, n: usize, saw_spins: bool, line: usize) -> Result<(), FtpError> {
    if !saw_spins {
        return err(line, "`spins <n>` must appear before anything that refers to a spin");
    }
    if i >= n {
        return err(line, format!("spin {i} is out of range for a program of {n} spins"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn models() -> Vec<(&'static str, Graph)> {
        vec![
            ("ring", crate::ising::ring(12, 1.0, 0.25)),
            ("lattice", crate::ising::lattice2d(6, 1.0)),
            ("z1", crate::device::z1_grid(5, 5, 1.0, 0.1)),
            ("single-edge", {
                let mut b = GraphBuilder::new(2);
                b.couple(0, 1, -0.75);
                b.build()
            }),
        ]
    }

    #[test]
    fn every_model_round_trips_byte_exactly() {
        // The acceptance criterion for this format.
        for (name, g) in models() {
            let sched = Schedule::geometric(0.05, 4.0, 12, 25).ramp_domain_wall(0.5, 6.0);
            let mut p = Program::from_graph(&g, &sched);
            p.name = Some(name.to_string());
            p.observe.push("energy".into());
            p.target = Some("cpu".into());
            p.price = Some("z1_spice".into());

            let text = p.to_ftp();
            let back = Program::from_ftp(&text).unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(back, p, "{name}: parse of a write differs");
            assert_eq!(back.to_ftp(), text, "{name}: write of a parse is not byte-identical");
            assert_eq!(back.digest(), p.digest(), "{name}: digest differs");
        }
    }

    #[test]
    fn floats_survive_the_round_trip_bit_for_bit() {
        // A format that loses the last bit of a coupling is a format that changes the model.
        let awkward = [0.1, 1.0 / 3.0, 1e-300, 1e300, -2.220446049250313e-16, core::f64::consts::PI];
        let mut b = GraphBuilder::new(awkward.len() + 1);
        for (i, &w) in awkward.iter().enumerate() {
            b.couple(i, i + 1, w);
            b.bias(i, w);
        }
        let p = Program::from_graph(&b.build(), &Schedule::constant(core::f64::consts::E, 3));
        let back = Program::from_ftp(&p.to_ftp()).unwrap();
        for (a, c) in p.factors.iter().zip(back.factors.iter()) {
            assert_eq!(a.weight().to_bits(), c.weight().to_bits(), "coupling lost bits");
        }
        for ((_, a), (_, c)) in p.bias.iter().zip(back.bias.iter()) {
            assert_eq!(a.to_bits(), c.to_bits(), "bias lost bits");
        }
        assert_eq!(
            p.schedule.stages()[0].beta.to_bits(),
            back.schedule.stages()[0].beta.to_bits()
        );
    }

    #[test]
    fn a_program_rebuilds_the_graph_it_came_from() {
        // The other half of "runs unchanged": the reconstructed model must be the same model.
        for (name, g) in models() {
            let p = Program::from_graph(&g, &Schedule::constant(1.0, 10));
            let g2 = Program::from_ftp(&p.to_ftp()).unwrap().to_graph().unwrap();
            assert_eq!(g2.n, g.n, "{name}");
            let mut rng = crate::rng::Pcg::new(5, 0);
            for _ in 0..50 {
                let s: Vec<i8> = (0..g.n).map(|_| if rng.f64() < 0.5 { 1 } else { -1 }).collect();
                assert!((g.energy(&s) - g2.energy(&s)).abs() < 1e-12, "{name}: energies differ");
            }
        }
    }

    #[test]
    fn comments_and_blank_lines_are_ignored() {
        let text = "# a program\n\nftp 1\n\n  spins 2  # two spins\nfactor 1 0 1\n\n";
        let p = Program::from_ftp(text).unwrap();
        assert_eq!(p.spins, 2);
        assert_eq!(p.factors.len(), 1);
    }

    #[test]
    fn the_worked_example_in_the_module_docs_parses() {
        // Documentation that does not parse is a bug report waiting to happen.
        let text = "ftp 1\nname frustrated-ring\nspins 5\n\
                    factor -1 0 1\nfactor -1 1 2\nfactor -1 2 3\nfactor -1 3 4\nfactor -1 4 0\n\
                    stage 0.05 40 1 1\nstage 4 40 1 1\nobserve energy\ntarget cpu\nprice z1_spice\n";
        let p = Program::from_ftp(text).unwrap();
        assert_eq!(p.spins, 5);
        assert_eq!(p.factors.len(), 5);
        assert_eq!(p.schedule.len(), 2);
        // and it is the frustrated ring it claims to be
        let g = p.to_graph().unwrap();
        let (_, e) = crate::tempering::anneal_scheduled(&g, &Schedule::geometric(0.05, 6.0, 60, 40), 1, None);
        assert_eq!(e, -3.0, "the documented example should be the frustrated 5-ring");
    }

    #[test]
    fn errors_name_the_line_and_the_fix() {
        let cases: Vec<(&str, usize, &str)> = vec![
            ("spins 4\n", 1, "expected `ftp <version>` first"),
            ("ftp 9\n", 1, "unsupported format version"),
            ("ftp 1\nspins 4\nfactor 1 0 9\n", 3, "out of range"),
            ("ftp 1\nspins 4\nfactor 1 0 0\n", 3, "appears 2 times"),
            ("ftp 1\nspins 4\nwobble 3\n", 3, "unknown directive"),
            ("ftp 1\nspins 4\nencode 0 3 trinary\n", 3, "unknown encoding"),
            ("ftp 1\nbias 0 1\n", 2, "must appear before"),
            ("ftp 1\nspins 4\nstage 0.5\n", 3, "missing stage sweeps"),
            ("ftp 1\n", 0, "missing `spins"),
        ];
        for (text, line, needle) in cases {
            let e = Program::from_ftp(text).unwrap_err();
            assert_eq!(e.line, line, "{text:?} -> {e}");
            assert!(e.message.contains(needle), "{text:?} said {:?}", e.message);
        }
    }

    #[test]
    fn the_digest_tracks_the_program_not_the_formatting() {
        let g = crate::ising::ring(8, 1.0, 0.0);
        let p = Program::from_graph(&g, &Schedule::constant(1.0, 5));
        let spaced = p.to_ftp().replace('\n', "  # noise\n");
        let back = Program::from_ftp(&spaced).unwrap();
        assert_eq!(back.digest(), p.digest(), "comments must not change the digest");

        let mut different = p.clone();
        different.schedule = Schedule::constant(2.0, 5);
        assert_ne!(different.digest(), p.digest(), "a changed schedule must change the digest");
    }

    #[test]
    fn encodings_survive_as_provenance() {
        let mut p = Program::from_graph(&crate::ising::ring(7, 1.0, 0.0), &Schedule::constant(1.0, 1));
        p.encodings.push(EncodedVar { base: 0, k: 4, encoding: Encoding::DomainWall });
        p.encodings.push(EncodedVar { base: 3, k: 5, encoding: Encoding::OneHot });
        let back = Program::from_ftp(&p.to_ftp()).unwrap();
        assert_eq!(back.encodings, p.encodings);
    }
}
