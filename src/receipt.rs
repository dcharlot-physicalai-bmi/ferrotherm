//! A run's answer, its cost, and its claims — as a file someone else can check.
//!
//! Every verification this crate does dies with the process that did it. `certify` returns a
//! certificate, `sdp` returns a dual point, `exact` returns a width, `portfolio` returns what each
//! arm spent — and all of it is a value in memory. Hand someone the answer and they have a list of
//! spins and your word for everything else.
//!
//! A receipt is that record written down. It carries the answer, what the run cost, and whatever the
//! run claimed about optimality, together with a digest of the model it was computed against. Given
//! the model, [`Receipt::verify`] re-checks every claim from scratch: it recomputes the energy, it
//! checks the state is a state, and it checks a claimed lower bound is actually below the answer.
//!
//! # What it is and is not
//!
//! It is a **re-checkable record**, not a proof of provenance. Nothing here stops someone writing a
//! receipt by hand; what it stops is a receipt whose parts disagree with each other or with the
//! model. That is the useful property, because it is the one that catches an honest mistake — a
//! state copied from the wrong run, an energy carried from a solver that drifted, a bound that was
//! never below the answer it was reported beside.
//!
//! The digest answers "is this the same model" and nothing else. It is FNV-1a, matching
//! [`crate::ftp::Program::digest`], and is not a security hash and not claimed to be one.
//!
//! # The energy is a claim, not the answer
//!
//! `verify` recomputes the energy from the state and compares. The stored number is what the run
//! SAID, and the recomputed one is what the state is worth; a receipt whose two disagree is exactly
//! the case worth catching, and trusting the stored number would make the check vacuous.

use crate::graph::Graph;
use crate::ledger::Ledger;

/// A stable digest of a graph's structure and weights.
///
/// FNV-1a over the CSR in its canonical order — which is canonical because `GraphBuilder::build`
/// merges through a `BTreeMap` rather than a `HashMap`, a fix made so that exactly this kind of
/// digest means something. Answers "is this the same model" and nothing else.
///
/// Weights go in as their bit patterns. Two graphs whose couplings differ in the last ulp are not
/// the same model, and a digest over decimal renderings would say they were.
#[must_use]
pub fn digest(g: &Graph) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut eat = |x: u64| {
        for b in x.to_le_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    };
    eat(g.n as u64);
    for i in 0..g.n {
        eat(g.h[i].to_bits());
        for k in g.offset[i]..g.offset[i + 1] {
            let j = g.nbr[k] as usize;
            if j > i {
                eat(i as u64);
                eat(j as u64);
                eat(g.w[k].to_bits());
            }
        }
    }
    h
}

/// Why a receipt did not check out.
#[derive(Clone, Debug, PartialEq)]
pub enum Mismatch {
    /// The receipt was written against a different model.
    DifferentModel {
        /// What the receipt says.
        receipt: u64,
        /// What the graph digests to.
        graph: u64,
    },
    /// The state is not a state of this model.
    NotAState {
        /// Spins in the receipt.
        got: usize,
        /// Spins the model has.
        want: usize,
    },
    /// A spin that is neither `+1` nor `-1`.
    NotASpin {
        /// Where.
        at: usize,
        /// What was there.
        got: i8,
    },
    /// The stored energy is not the energy of the stored state.
    Energy {
        /// What the receipt claims.
        claimed: f64,
        /// What the state is actually worth.
        actual: f64,
    },
    /// A claimed lower bound sits above the answer it was reported beside.
    BoundAboveAnswer {
        /// The bound.
        bound: f64,
        /// The energy it was supposed to be below.
        energy: f64,
    },
}

impl core::fmt::Display for Mismatch {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Mismatch::DifferentModel { receipt, graph } => write!(
                f,
                "this receipt was written against model {receipt:016x} and the graph given digests \
                 to {graph:016x}; nothing else about it can be checked"
            ),
            Mismatch::NotAState { got, want } => {
                write!(f, "the receipt carries {got} spins and the model has {want}")
            }
            Mismatch::NotASpin { at, got } => {
                write!(f, "position {at} holds {got}, which is not a spin")
            }
            Mismatch::Energy { claimed, actual } => write!(
                f,
                "the receipt claims an energy of {claimed} and its own state is worth {actual}"
            ),
            Mismatch::BoundAboveAnswer { bound, energy } => write!(
                f,
                "the receipt claims a lower bound of {bound} on a model whose answer it gives as \
                 {energy}; a bound above the answer is not a bound"
            ),
        }
    }
}

impl core::error::Error for Mismatch {}

/// A receipt that checked out, and what it establishes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Verified {
    /// The energy, recomputed rather than read.
    pub energy: f64,
    /// How far the answer may be from optimal, when the receipt carried a bound.
    ///
    /// `None` means the receipt made no optimality claim — which is honest, and different from a
    /// gap of zero.
    pub gap: Option<f64>,
}

/// A run's answer and claims, in a form that outlives the process.
#[derive(Clone, Debug, PartialEq)]
pub struct Receipt {
    /// Digest of the model this was computed against.
    pub model: u64,
    /// The answer.
    pub state: Vec<i8>,
    /// The energy the run reported. A claim, re-checked by [`Receipt::verify`].
    pub energy: f64,
    /// A lower bound on the model's ground energy, if the run had one.
    pub bound: Option<f64>,
    /// What the run cost, in device operations.
    pub cost: Ledger,
    /// What produced it.
    pub method: String,
    /// The seed it ran at, so it can be reproduced rather than merely believed.
    pub seed: u64,
}

/// Why a receipt could not be read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParseError {
    /// One-based line number.
    pub line: usize,
    /// What was wrong.
    pub message: String,
}

impl core::fmt::Display for ParseError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "line {}: {}", self.line, self.message)
    }
}

impl core::error::Error for ParseError {}

impl Receipt {
    /// A receipt for `state` as an answer to `g`.
    ///
    /// The energy is recomputed here rather than taken from the caller: a receipt whose energy came
    /// from the same place as the state's own bookkeeping would carry that bookkeeping's errors, and
    /// this is the one place it can be independently established.
    #[must_use]
    pub fn of(g: &Graph, state: Vec<i8>, method: &str, seed: u64, cost: Ledger) -> Receipt {
        let energy = g.energy(&state);
        Receipt {
            model: digest(g),
            state,
            energy,
            bound: None,
            cost,
            method: method.to_string(),
            seed,
        }
    }

    /// Attach a lower bound on the ground energy.
    #[must_use]
    pub fn with_bound(mut self, bound: f64) -> Receipt {
        self.bound = Some(bound);
        self
    }

    /// The receipt as text.
    ///
    /// One field per line, and the state as `+`/`-` characters — readable, diffable, and small.
    /// Floats are written with Rust's shortest round-tripping form, so parsing returns the same
    /// bits; `a_receipt_survives_a_round_trip_through_text` is what says so rather than the claim.
    #[must_use]
    pub fn to_text(&self) -> String {
        let spins: String =
            self.state.iter().map(|&v| if v > 0 { '+' } else { '-' }).collect();
        let mut out = String::new();
        out.push_str("receipt 1\n");
        out.push_str(&format!("model {:016x}\n", self.model));
        out.push_str(&format!("method {}\n", self.method));
        out.push_str(&format!("seed {}\n", self.seed));
        out.push_str(&format!("energy {:?}\n", self.energy));
        if let Some(b) = self.bound {
            out.push_str(&format!("bound {b:?}\n"));
        }
        out.push_str(&format!(
            "cost {} {} {}\n",
            self.cost.samples, self.cost.reads, self.cost.writes
        ));
        out.push_str(&format!("state {spins}\n"));
        out
    }

    /// Read a receipt back.
    ///
    /// # Errors
    ///
    /// [`ParseError`] naming the line. A field that is present but malformed is an error rather than
    /// a default: a receipt that silently reads as a different receipt is worse than one that does
    /// not read at all.
    pub fn parse(text: &str) -> Result<Receipt, ParseError> {
        let mut model = None;
        let mut state = None;
        let mut energy = None;
        let mut bound = None;
        let mut cost = Ledger::default();
        let mut method = None;
        let mut seed = None;
        let mut saw_header = false;

        for (no, raw) in text.lines().enumerate() {
            let line = raw.trim();
            let at = no + 1;
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let mut it = line.split_whitespace();
            let key = it.next().unwrap_or("");
            let bad = |what: &str| ParseError { line: at, message: format!("bad {what}") };
            match key {
                "receipt" => {
                    if it.next() != Some("1") {
                        return Err(ParseError {
                            line: at,
                            message: "only version 1 is understood".into(),
                        });
                    }
                    saw_header = true;
                }
                "model" => {
                    let t = it.next().ok_or_else(|| bad("model"))?;
                    model =
                        Some(u64::from_str_radix(t, 16).map_err(|_| bad("model digest"))?);
                }
                "method" => method = Some(it.collect::<Vec<_>>().join(" ")),
                "seed" => {
                    seed = Some(it.next().and_then(|t| t.parse().ok()).ok_or_else(|| bad("seed"))?);
                }
                "energy" => {
                    energy =
                        Some(it.next().and_then(|t| t.parse().ok()).ok_or_else(|| bad("energy"))?);
                }
                "bound" => {
                    bound =
                        Some(it.next().and_then(|t| t.parse().ok()).ok_or_else(|| bad("bound"))?);
                }
                "cost" => {
                    let mut next = || it.next().and_then(|t| t.parse::<u64>().ok());
                    let (Some(s), Some(r), Some(w)) = (next(), next(), next()) else {
                        return Err(bad("cost"));
                    };
                    cost = Ledger { samples: s, reads: r, writes: w };
                }
                "state" => {
                    let t = it.next().ok_or_else(|| bad("state"))?;
                    let mut v = Vec::with_capacity(t.len());
                    for c in t.chars() {
                        match c {
                            '+' => v.push(1i8),
                            '-' => v.push(-1i8),
                            _ => {
                                return Err(ParseError {
                                    line: at,
                                    message: format!("`{c}` is not a spin; use + or -"),
                                });
                            }
                        }
                    }
                    state = Some(v);
                }
                other => {
                    return Err(ParseError {
                        line: at,
                        message: format!("unknown field `{other}`"),
                    });
                }
            }
        }
        if !saw_header {
            return Err(ParseError { line: 1, message: "no `receipt 1` header".into() });
        }
        let missing = |what: &str| ParseError { line: 0, message: format!("no {what}") };
        Ok(Receipt {
            model: model.ok_or_else(|| missing("model"))?,
            state: state.ok_or_else(|| missing("state"))?,
            energy: energy.ok_or_else(|| missing("energy"))?,
            bound,
            cost,
            method: method.ok_or_else(|| missing("method"))?,
            seed: seed.ok_or_else(|| missing("seed"))?,
        })
    }

    /// Re-check every claim against the model, from scratch.
    ///
    /// # Errors
    ///
    /// [`Mismatch`], naming which claim failed. The model digest is checked first, because nothing
    /// else means anything if the receipt describes a different model.
    pub fn verify(&self, g: &Graph) -> Result<Verified, Mismatch> {
        let d = digest(g);
        if d != self.model {
            return Err(Mismatch::DifferentModel { receipt: self.model, graph: d });
        }
        if self.state.len() != g.n {
            return Err(Mismatch::NotAState { got: self.state.len(), want: g.n });
        }
        if let Some((at, &got)) =
            self.state.iter().enumerate().find(|&(_, &v)| v != 1 && v != -1)
        {
            return Err(Mismatch::NotASpin { at, got });
        }
        let actual = g.energy(&self.state);
        // Recomputed, not read. Scaled to the magnitude, so a large model is not held to absolute
        // bits it cannot carry.
        let tol = 1e-9 * actual.abs().max(1.0);
        if (actual - self.energy).abs() > tol {
            return Err(Mismatch::Energy { claimed: self.energy, actual });
        }
        if let Some(b) = self.bound
            && b > actual + tol
        {
            return Err(Mismatch::BoundAboveAnswer { bound: b, energy: actual });
        }
        Ok(Verified { energy: actual, gap: self.bound.map(|b| actual - b) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> (Graph, Vec<i8>) {
        let g = crate::ising::ring(10, 1.0, 0.35);
        let e = crate::exact::Elimination::default().ground_state(&g).unwrap();
        (g, e.ground_state.unwrap())
    }

    /// A receipt round-trips through text with every field intact, bits included.
    ///
    /// Floats go out in Rust's shortest round-tripping form. If that were not exact the energy would
    /// come back a rounding different, `verify` would still pass on its tolerance, and the receipt
    /// would quietly stop being the record of what ran.
    #[test]
    fn a_receipt_survives_a_round_trip_through_text() {
        let (g, st) = fixture();
        let cost = Ledger { samples: 12_345, reads: 67, writes: 8 };
        let r = Receipt::of(&g, st, "exact::ground_state", 7, cost).with_bound(-14.0);
        let back = Receipt::parse(&r.to_text()).expect("its own output must parse");
        assert_eq!(back, r);
        assert_eq!(back.energy.to_bits(), r.energy.to_bits(), "the energy lost bits in transit");
        assert_eq!(
            back.bound.unwrap().to_bits(),
            r.bound.unwrap().to_bits(),
            "the bound lost bits in transit"
        );
    }

    /// A receipt verifies against the model it was made from, and reports the gap it claimed.
    #[test]
    fn a_receipt_verifies_against_its_own_model() {
        let (g, st) = fixture();
        let r = Receipt::of(&g, st, "exact", 1, Ledger::default());
        let v = r.verify(&g).expect("its own model must check out");
        assert!((v.energy - r.energy).abs() < 1e-12);
        assert_eq!(v.gap, None, "no bound claimed means no gap, not a gap of zero");

        let with = r.clone().with_bound(r.energy - 0.5);
        let v2 = with.verify(&g).unwrap();
        assert!((v2.gap.unwrap() - 0.5).abs() < 1e-12);
    }

    /// Every way of tampering with a receipt is caught.
    ///
    /// The property that makes it re-checkable rather than decorative: no single field can be edited
    /// and still pass. Each case below is a plausible honest mistake, not a forgery — a state copied
    /// from the wrong run, an energy carried from a solver that drifted, a bound pasted beside the
    /// wrong answer.
    #[test]
    fn no_single_field_can_be_edited_and_still_pass() {
        let (g, st) = fixture();
        let good = Receipt::of(&g, st, "exact", 1, Ledger::default());
        assert!(good.verify(&g).is_ok());

        // The energy no longer matches the state.
        let mut bad = good.clone();
        bad.energy += 0.5;
        assert!(matches!(bad.verify(&g), Err(Mismatch::Energy { .. })));

        // The state was replaced by another valid state, so the energy no longer describes it.
        let mut bad = good.clone();
        bad.state[0] = -bad.state[0];
        bad.state[1] = -bad.state[1];
        assert!(matches!(bad.verify(&g), Err(Mismatch::Energy { .. })));

        // A "lower bound" that sits above the answer.
        let bad = good.clone().with_bound(good.energy + 1.0);
        assert!(matches!(bad.verify(&g), Err(Mismatch::BoundAboveAnswer { .. })));

        // A state of the wrong width.
        let mut bad = good.clone();
        bad.state.push(1);
        assert!(matches!(bad.verify(&g), Err(Mismatch::NotAState { .. })));

        // Something that is not a spin.
        let mut bad = good.clone();
        bad.state[3] = 0;
        assert!(matches!(bad.verify(&g), Err(Mismatch::NotASpin { at: 3, got: 0 })));
    }

    /// A receipt for one model does not verify against another, even a similar one.
    ///
    /// The digest is checked first and on its own, because every other check is meaningless if the
    /// model is not the one the run used. The two graphs here have the same size, the same edge
    /// count and the same shape — only the field differs, which is exactly the near-miss a
    /// size-and-shape comparison would wave through.
    #[test]
    fn a_receipt_does_not_verify_against_a_different_model() {
        let (g, st) = fixture();
        let r = Receipt::of(&g, st, "exact", 1, Ledger::default());
        let other = crate::ising::ring(10, 1.0, 0.36);
        assert_eq!(other.n, g.n);
        match r.verify(&other) {
            Err(Mismatch::DifferentModel { receipt, graph }) => {
                assert_eq!(receipt, r.model);
                assert_ne!(receipt, graph);
            }
            v => panic!("a receipt for another model must not verify: {v:?}"),
        }
    }

    /// The digest separates models that differ in the last ulp of one coupling.
    ///
    /// Weights go in as bit patterns. Two models a decimal rendering could not tell apart are still
    /// two models, and a digest that merged them would let a receipt verify against a graph it was
    /// never computed on.
    #[test]
    fn the_digest_separates_models_that_differ_by_one_ulp() {
        let a = crate::ising::ring(8, 1.0, 0.2);
        let mut b = crate::graph::GraphBuilder::new(8);
        for i in 0..8 {
            let w = if i == 0 { 1.0f64.next_up() } else { 1.0 };
            b.couple(i, (i + 1) % 8, w);
            b.bias(i, 0.2);
        }
        let b = b.build();
        assert_ne!(digest(&a), digest(&b), "one ulp is a different model");
        // And the digest is stable: the same graph built twice digests the same.
        assert_eq!(digest(&a), digest(&crate::ising::ring(8, 1.0, 0.2)));
    }

    /// Malformed text is refused by line, not defaulted into a different receipt.
    #[test]
    fn malformed_text_is_refused_rather_than_guessed() {
        let (g, st) = fixture();
        let good = Receipt::of(&g, st, "exact", 1, Ledger::default()).to_text();

        assert!(Receipt::parse("model deadbeef\n").is_err(), "no header");
        assert!(Receipt::parse("receipt 2\nmodel 0\n").is_err(), "a version this cannot read");
        assert!(
            Receipt::parse(&good.replace("state +", "state ?")).is_err(),
            "a spin that is not a spin"
        );
        // An unknown field ADDED, not one substituted for a required one. Renaming `energy` to
        // `enrgy` errors either way -- on the missing `energy` -- so it passes whether or not
        // unknown fields are rejected, and a mutation that skipped them survived it.
        assert!(
            Receipt::parse(&format!("{good}provenance whoever\n")).is_err(),
            "an unknown field is an error, not something to skip: a receipt that reads as a \
             DIFFERENT receipt is worse than one that does not read"
        );
        assert!(
            Receipt::parse(&good.replace("energy", "enrgy")).is_err(),
            "and a required field renamed away is still missing"
        );
        // A missing required field is an error rather than a default.
        let without_state: String =
            good.lines().filter(|l| !l.starts_with("state")).collect::<Vec<_>>().join("\n");
        assert!(Receipt::parse(&without_state).is_err(), "no state");
        // Comments and blank lines are fine.
        let decorated = format!("# written by a test\n\n{good}\n");
        assert!(Receipt::parse(&decorated).is_ok());
    }

    /// The cost survives, because a receipt without it is an answer without a price.
    #[test]
    fn the_ledger_round_trips_with_the_answer() {
        let (g, st) = fixture();
        let cost = Ledger { samples: 9_000_000_000, reads: 12, writes: 3 };
        let r = Receipt::of(&g, st, "tempering", 42, cost);
        let back = Receipt::parse(&r.to_text()).unwrap();
        assert_eq!(back.cost.samples, 9_000_000_000);
        assert_eq!(back.cost.reads, 12);
        assert_eq!(back.cost.writes, 3);
        assert_eq!(back.seed, 42);
        assert_eq!(back.method, "tempering");
    }
}
