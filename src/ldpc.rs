//! Error-correcting codes as k-body Ising models: decoding is ground-state search.
//!
//! # The encoding
//!
//! Take `bit 0 -> spin +1`, `bit 1 -> spin -1`. A parity check over a set `C` is satisfied when an
//! even number of its bits are one, which under that map is exactly `prod_{i in C} s_i = +1`, so
//!
//! ```text
//!     v_C(s) = (1 - prod_{i in C} s_i) / 2
//! ```
//!
//! is zero on a satisfied check and one on a violated one. It is a `|C|`-body term, which is why a
//! code does not fit pairwise hardware as written; [`crate::reduce::to_pairwise_exact`] lowers it
//! with no penalty coefficient to tune, the same route [`crate::invertible`] takes for logic gates.
//!
//! The channel enters as a field. With log-likelihood ratios `L_i = ln P(y_i | 0) / P(y_i | 1)`,
//! the negative log-likelihood of a word is `-sum_i L_i s_i / 2` up to a constant, so `h_i = L_i/2`
//! and the decoder's objective is
//!
//! ```text
//!     E(s) = lambda * (violated checks) - sum_i L_i s_i / 2
//! ```
//!
//! # Why the ground state is the maximum-likelihood codeword
//!
//! The channel term spans `sum_i |L_i|` between its best and worst state, so any `lambda` strictly
//! above that makes one violated check cost more than the entire channel can repay. Then no state
//! off the code can undercut the best state on it, and the ground state is the maximum-likelihood
//! codeword — exactly, not approximately. [`Code::penalty`] returns that bound with the sum rounded
//! UP by [`crate::round::sum_up`], so the strictness survives floating point.
//!
//! # What this is and is not
//!
//! This is EXACT maximum-likelihood decoding, which is NP-hard in general and is not what a
//! production `LDPC` receiver runs — belief propagation is, and it is approximate. What an exact
//! decoder is for is being the oracle that an approximate one is scored against, on codes small
//! enough that [`crate::exact::Elimination`] can contract them.

use crate::exact::{Elimination, Exact, TooWide};
use crate::factor::Factor;
use crate::ftp::Program;
use crate::graph::Graph;
use crate::reduce::{Reduction, to_pairwise_exact};
use crate::round::sum_up;
use crate::schedule::Schedule;

/// The widest parity check this builds a model for.
///
/// A check of arity `k` expands to `2^k` binary monomials inside the penalty-free reduction, each
/// taking its own auxiliaries, so the cost is exponential in the check's weight and not in the
/// code's length. Real `LDPC` checks are single digits wide.
pub const MAX_CHECK: usize = 12;

/// The longest code [`Code::codewords`] will enumerate: `2^24` words is already a minute of work.
pub const MAX_ENUM: usize = 24;

/// Why a code could not be built, modelled or decoded.
#[derive(Clone, Debug, PartialEq)]
pub enum Error {
    /// A check named a bit the code does not have.
    UnknownBit {
        /// The offending index.
        bit: usize,
        /// How many bits the code has.
        bits: usize,
    },
    /// A check named the same bit twice, which over `GF(2)` is not the check that was written.
    RepeatedBit {
        /// The bit listed more than once.
        bit: usize,
    },
    /// A check over no bits at all, which constrains nothing.
    EmptyCheck,
    /// A check wider than [`MAX_CHECK`].
    CheckTooWide {
        /// The check's weight.
        arity: usize,
        /// The limit.
        limit: usize,
    },
    /// One log-likelihood ratio per bit is required, and that is not what arrived.
    LlrLen {
        /// How many were given.
        got: usize,
        /// How many the code has bits.
        want: usize,
    },
    /// A log-likelihood ratio that is `NaN` or infinite. Infinity is a channel claiming certainty,
    /// which no penalty can outbid, so it is refused rather than silently dominating the model.
    NonFinite {
        /// Which bit carried it.
        bit: usize,
    },
    /// A channel parameter outside the range its formula is defined on.
    BadChannel,
    /// The code is longer than [`MAX_ENUM`], so its codewords will not be enumerated.
    TooLongToEnumerate {
        /// The code's length.
        bits: usize,
        /// The limit.
        limit: usize,
    },
    /// The lowering to pairwise failed; see [`crate::reduce::ReduceError`].
    Reduce(crate::reduce::ReduceError),
    /// Exact elimination declined the reduced graph; see [`crate::exact::TooWide`].
    Width(TooWide),
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Error::UnknownBit { bit, bits } => {
                write!(f, "a check names bit {bit}, but the code has {bits}")
            }
            Error::RepeatedBit { bit } => write!(
                f,
                "bit {bit} appears twice in one check; over GF(2) it cancels, so this is not the \
                 check that was written"
            ),
            Error::EmptyCheck => write!(f, "a parity check over no bits constrains nothing"),
            Error::CheckTooWide { arity, limit } => write!(
                f,
                "a weight-{arity} check expands to 2^{arity} monomials in the reduction, over the \
                 {limit} this builds"
            ),
            Error::LlrLen { got, want } => {
                write!(f, "{got} log-likelihood ratios for a code of {want} bits")
            }
            Error::NonFinite { bit } => write!(
                f,
                "the log-likelihood ratio for bit {bit} is not finite; a channel claiming certainty \
                 has no penalty that outbids it"
            ),
            Error::BadChannel => {
                write!(f, "the channel parameter is outside the range its formula is defined on")
            }
            Error::TooLongToEnumerate { bits, limit } => {
                write!(f, "enumerating a {bits}-bit code means 2^{bits} words, over the 2^{limit} limit")
            }
            Error::Reduce(e) => write!(f, "lowering to pairwise failed: {e}"),
            Error::Width(e) => write!(f, "exact decoding declined this code: {e}"),
        }
    }
}

/// A binary linear code, given by its parity checks.
///
/// The rows are the parity-check matrix `H` written as index sets rather than as a dense matrix of
/// zeros: `LDPC` means the matrix is sparse, and a sparse row is what the k-body term needs anyway.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Code {
    bits: usize,
    checks: Vec<Vec<usize>>,
}

impl Code {
    /// A code of `bits` bits with no checks yet — every word is a codeword until one is added.
    #[must_use]
    pub fn new(bits: usize) -> Code {
        Code { bits, checks: Vec::new() }
    }

    /// The code's length in bits.
    #[must_use]
    pub fn bits(&self) -> usize {
        self.bits
    }

    /// How many parity checks it has. The rank may be lower, so this is not the number of parity
    /// bits.
    #[must_use]
    pub fn checks(&self) -> usize {
        self.checks.len()
    }

    /// The checks, each as the sorted set of bits it covers.
    #[must_use]
    pub fn rows(&self) -> &[Vec<usize>] {
        &self.checks
    }

    /// Add a parity check: the listed bits must sum to zero over `GF(2)`.
    ///
    /// # Errors
    ///
    /// [`Error::EmptyCheck`], [`Error::CheckTooWide`] past [`MAX_CHECK`], [`Error::UnknownBit`] for
    /// an index the code does not have, and [`Error::RepeatedBit`] for one listed twice.
    pub fn check(&mut self, bits: &[usize]) -> Result<(), Error> {
        if bits.is_empty() {
            return Err(Error::EmptyCheck);
        }
        if bits.len() > MAX_CHECK {
            return Err(Error::CheckTooWide { arity: bits.len(), limit: MAX_CHECK });
        }
        let mut sorted = bits.to_vec();
        sorted.sort_unstable();
        for w in sorted.windows(2) {
            if w[0] == w[1] {
                return Err(Error::RepeatedBit { bit: w[0] });
            }
        }
        if let Some(&b) = sorted.last()
            && b >= self.bits
        {
            return Err(Error::UnknownBit { bit: b, bits: self.bits });
        }
        self.checks.push(sorted);
        Ok(())
    }

    /// The syndrome of a word: one bit per check, set when that check is violated.
    ///
    /// # Panics
    ///
    /// If `word` is shorter than the code.
    #[must_use]
    pub fn syndrome(&self, word: &[u8]) -> Vec<u8> {
        assert!(word.len() >= self.bits, "a word must cover the code's bits");
        self.checks
            .iter()
            .map(|c| c.iter().fold(0u8, |a, &i| a ^ (word[i] & 1)))
            .collect()
    }

    /// Whether every check is satisfied.
    ///
    /// # Panics
    ///
    /// If `word` is shorter than the code.
    #[must_use]
    pub fn is_codeword(&self, word: &[u8]) -> bool {
        self.syndrome(word).iter().all(|&b| b == 0)
    }

    /// How many checks a SPIN state violates, with `+1` meaning bit zero.
    ///
    /// Ancillas past the code's bits are ignored, so this reads a reduced state directly.
    ///
    /// # Panics
    ///
    /// If `s` is shorter than the code.
    #[must_use]
    pub fn violations(&self, s: &[i8]) -> usize {
        assert!(s.len() >= self.bits, "a state must cover the code's bits");
        self.checks
            .iter()
            .filter(|c| c.iter().map(|&i| i32::from(s[i])).product::<i32>() < 0)
            .count()
    }

    /// The word a spin state stands for: `+1` is bit zero.
    ///
    /// # Panics
    ///
    /// If `s` is shorter than the code.
    #[must_use]
    pub fn word(&self, s: &[i8]) -> Vec<u8> {
        assert!(s.len() >= self.bits, "a state must cover the code's bits");
        (0..self.bits).map(|i| u8::from(s[i] < 0)).collect()
    }

    /// The spin state a word stands for.
    ///
    /// # Panics
    ///
    /// If `word` is shorter than the code.
    #[must_use]
    pub fn state(&self, word: &[u8]) -> Vec<i8> {
        assert!(word.len() >= self.bits, "a word must cover the code's bits");
        (0..self.bits).map(|i| if word[i] & 1 == 1 { -1i8 } else { 1 }).collect()
    }

    /// Every codeword, by enumeration.
    ///
    /// # Errors
    ///
    /// [`Error::TooLongToEnumerate`] past [`MAX_ENUM`] bits.
    pub fn codewords(&self) -> Result<Vec<Vec<u8>>, Error> {
        if self.bits > MAX_ENUM {
            return Err(Error::TooLongToEnumerate { bits: self.bits, limit: MAX_ENUM });
        }
        let mut out = Vec::new();
        let mut w = vec![0u8; self.bits];
        for m in 0u64..(1u64 << self.bits) {
            for (i, b) in w.iter_mut().enumerate() {
                *b = u8::from(m >> i & 1 == 1);
            }
            if self.is_codeword(&w) {
                out.push(w.clone());
            }
        }
        Ok(out)
    }

    /// The minimum Hamming distance, which for a linear code is the least weight of a non-zero
    /// codeword. `None` when the code holds only the zero word and no distance is defined.
    ///
    /// # Errors
    ///
    /// As [`Code::codewords`].
    pub fn min_distance(&self) -> Result<Option<usize>, Error> {
        Ok(self
            .codewords()?
            .iter()
            .map(|c| c.iter().filter(|&&b| b == 1).count())
            .filter(|&w| w > 0)
            .min())
    }

    /// The smallest penalty that makes the ground state a maximum-likelihood codeword.
    ///
    /// Strictly above `sum_i |L_i|`, the span of the channel term, with the sum rounded UP so the
    /// inequality is not lost to rounding. Understating this is the one way to get a decoder that
    /// silently returns non-codewords.
    ///
    /// # Errors
    ///
    /// [`Error::LlrLen`] for the wrong number of ratios and [`Error::NonFinite`] for one that is
    /// not a number.
    pub fn penalty(&self, llr: &[f64]) -> Result<f64, Error> {
        if llr.len() != self.bits {
            return Err(Error::LlrLen { got: llr.len(), want: self.bits });
        }
        for (i, &l) in llr.iter().enumerate() {
            if !l.is_finite() {
                return Err(Error::NonFinite { bit: i });
            }
        }
        let mags: Vec<f64> = llr.iter().map(|l| l.abs()).collect();
        Ok(sum_up(&mags) + 1.0)
    }

    /// The channel's contribution to the energy at a state: `-sum_i L_i s_i / 2`.
    ///
    /// # Errors
    ///
    /// As [`Code::penalty`].
    ///
    /// # Panics
    ///
    /// If `s` is shorter than the code.
    pub fn channel_energy(&self, llr: &[f64], s: &[i8]) -> Result<f64, Error> {
        self.penalty(llr)?;
        assert!(s.len() >= self.bits, "a state must cover the code's bits");
        Ok(-llr.iter().enumerate().map(|(i, l)| l * f64::from(s[i]) / 2.0).sum::<f64>())
    }

    /// The decoding objective as a program of k-body factors, with the penalty [`Code::penalty`]
    /// chooses.
    ///
    /// Returns the program and the constant that makes the identity exact:
    /// `energy(s) + offset = lambda * violations(s) + channel_energy(s)` at every state.
    ///
    /// # Errors
    ///
    /// As [`Code::penalty`].
    pub fn to_program(&self, llr: &[f64]) -> Result<(Program, f64), Error> {
        let lambda = self.penalty(llr)?;
        self.to_program_with(llr, lambda)
    }

    /// The same, with the penalty supplied.
    ///
    /// For a caller with a tighter bound of their own. Below [`Code::penalty`] the ground state
    /// stops being a codeword, which `a_penalty_below_the_bound_decodes_off_the_code` demonstrates
    /// rather than assumes.
    ///
    /// # Errors
    ///
    /// As [`Code::penalty`], plus [`Error::BadChannel`] for a penalty that is not finite and
    /// positive.
    ///
    /// # Panics
    ///
    /// Never: every index was checked against `bits` when its check was added, so the `Factor`
    /// cannot be rejected.
    pub fn to_program_with(&self, llr: &[f64], lambda: f64) -> Result<(Program, f64), Error> {
        self.penalty(llr)?;
        if !(lambda.is_finite() && lambda > 0.0) {
            return Err(Error::BadChannel);
        }
        let mut bias = vec![0.0f64; self.bits];
        for (i, &l) in llr.iter().enumerate() {
            bias[i] += l / 2.0;
        }
        let mut factors = Vec::new();
        for c in &self.checks {
            // -(lambda/2) * prod s, whose constant partner lambda/2 is carried in `offset`.
            if c.len() == 1 {
                bias[c[0]] += lambda / 2.0;
            } else {
                factors.push(
                    Factor::new(c, lambda / 2.0, self.bits)
                        .expect("indices were checked when the check was added"),
                );
            }
        }
        let p = Program {
            name: Some("ldpc".into()),
            spins: self.bits,
            bias: bias
                .iter()
                .enumerate()
                .filter(|(_, h)| **h != 0.0)
                .map(|(i, &h)| (i, h))
                .collect(),
            factors,
            colors: Vec::new(),
            encodings: Vec::new(),
            schedule: Schedule::default(),
            observe: Vec::new(),
            target: None,
            price: None,
        };
        Ok((p, lambda * self.checks.len() as f64 / 2.0))
    }

    /// The decoding objective as a pairwise graph, through the penalty-free reduction.
    ///
    /// The [`Reduction`] maps a solved state back: its ancillas are an artefact of the lowering and
    /// [`Reduction::project`] drops them.
    ///
    /// # Errors
    ///
    /// As [`Code::to_program`], plus [`Error::Reduce`] when a check is too wide to lower.
    ///
    /// # Panics
    ///
    /// Never: `to_graph` runs on the reduction's own output, which is pairwise by construction.
    pub fn to_graph(&self, llr: &[f64]) -> Result<(Graph, Reduction), Error> {
        let (p, _) = self.to_program(llr)?;
        let red = to_pairwise_exact(&p).map_err(Error::Reduce)?;
        let g = red.program.to_graph().expect("the reduction returns a pairwise program");
        Ok((g, red))
    }

    /// Decode exactly, by eliminating the reduced graph.
    ///
    /// # Errors
    ///
    /// As [`Code::to_graph`], plus [`Error::Width`] when the reduced graph is too dense to
    /// eliminate.
    pub fn decode(&self, llr: &[f64]) -> Result<Decoded, Error> {
        self.decode_with(llr, &Elimination::default())
    }

    /// Decode exactly with a given elimination budget.
    ///
    /// # Errors
    ///
    /// As [`Code::decode`].
    ///
    /// # Panics
    ///
    /// Never: min-sum was requested, so the run reports a ground state.
    pub fn decode_with(&self, llr: &[f64], e: &Elimination) -> Result<Decoded, Error> {
        let lambda = self.penalty(llr)?;
        let (g, red) = self.to_graph(llr)?;
        let ex: Exact = e.ground_state(&g).map_err(Error::Width)?;
        let s = ex.ground_state.expect("min-sum was run, so it reports a ground state");
        let word = self.word(red.project(&s));
        let violations = self.violations(&s);
        let energy = lambda * violations as f64 + self.channel_energy(llr, &s)?;
        Ok(Decoded { word, violations, energy, penalty: lambda, ancillas: red.ancillas, width: ex.width })
    }
}

/// What an exact decode produced.
#[derive(Clone, Debug, PartialEq)]
pub struct Decoded {
    /// The decoded word, one byte per bit.
    pub word: Vec<u8>,
    /// Checks the answer violates. Zero whenever `penalty` came from [`Code::penalty`].
    pub violations: usize,
    /// The objective at the answer: `penalty * violations + channel_energy`.
    pub energy: f64,
    /// The penalty used.
    pub penalty: f64,
    /// Ancillas the reduction added, which are not part of the answer.
    pub ancillas: usize,
    /// Induced width of the elimination that solved it.
    pub width: usize,
}

/// The `[7, 4, 3]` Hamming code, in systematic form: bits 0..4 carry data, 4..7 parity.
///
/// A published object with published parameters — 16 codewords, minimum distance 3, and perfect, so
/// the eight cosets are the zero syndrome and the seven single-bit errors. The tests check all of
/// that rather than trusting the transcription.
///
/// # Panics
///
/// Never: the checks are three fixed weight-four rows over seven bits.
#[must_use]
pub fn hamming74() -> Code {
    let mut c = Code::new(7);
    for row in [[0usize, 1, 2, 4], [0, 1, 3, 5], [0, 2, 3, 6]] {
        c.check(&row).expect("a fixed weight-four row over seven bits");
    }
    c
}

/// The length-`n` repetition code: every bit tied to bit zero.
///
/// # Panics
///
/// If `n` is zero, which is not a code.
#[must_use]
pub fn repetition(n: usize) -> Code {
    assert!(n > 0, "a code needs at least one bit");
    let mut c = Code::new(n);
    for i in 1..n {
        c.check(&[0, i]).expect("a weight-two row inside the code");
    }
    c
}

/// Log-likelihood ratios for a binary symmetric channel with crossover `p`.
///
/// `L_i = (1 - 2 y_i) ln((1 - p) / p)`, positive where the channel says bit zero.
///
/// # Errors
///
/// [`Error::BadChannel`] unless `0 < p < 1`, since `p = 0` is a channel claiming certainty and its
/// ratio is infinite.
pub fn bsc_llr(received: &[u8], p: f64) -> Result<Vec<f64>, Error> {
    if !(p.is_finite() && p > 0.0 && p < 1.0) {
        return Err(Error::BadChannel);
    }
    let l = ((1.0 - p) / p).ln();
    Ok(received.iter().map(|&y| if y & 1 == 1 { -l } else { l }).collect())
}

/// Log-likelihood ratios for `BPSK` over an additive white Gaussian noise channel of variance
/// `sigma2`, with bit zero sent as `+1`.
///
/// `L_i = 2 y_i / sigma2`.
///
/// # Errors
///
/// [`Error::BadChannel`] unless `sigma2` is finite and positive.
pub fn awgn_llr(received: &[f64], sigma2: f64) -> Result<Vec<f64>, Error> {
    if !(sigma2.is_finite() && sigma2 > 0.0) {
        return Err(Error::BadChannel);
    }
    if let Some(i) = received.iter().position(|y| !y.is_finite()) {
        return Err(Error::NonFinite { bit: i });
    }
    Ok(received.iter().map(|y| 2.0 * y / sigma2).collect())
}

/// Tests.
///
/// # There is no sampler-against-sampler test here, deliberately
///
/// Running Gibbs on the reduced graph and comparing it with another sampler would prove almost
/// nothing: two approximations agreeing is evidence about neither. Every check below is against
/// something that cannot be wrong — enumerated codewords, the published parameters of the Hamming
/// code, minimum Hamming distance computed directly, and an algebraic identity that must hold at
/// EVERY state rather than at the optimum.
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

    fn words(n: usize) -> impl Iterator<Item = Vec<u8>> {
        (0u64..(1u64 << n)).map(move |m| (0..n).map(|i| u8::from(m >> i & 1 == 1)).collect())
    }

    fn distance(a: &[u8], b: &[u8]) -> usize {
        a.iter().zip(b).filter(|(x, y)| x != y).count()
    }

    /// PUBLISHED PARAMETERS, checked rather than transcribed.
    ///
    /// A wrong row in `H` still gives a perfectly good linear code, with a ground state, for the
    /// wrong code. `[7, 4, 3]` and perfectness are the facts that pin it down: 2^4 codewords,
    /// minimum weight three, and eight distinct syndromes covering the zero error and all seven
    /// single-bit ones.
    #[test]
    fn the_hamming_code_has_its_published_parameters() {
        let c = hamming74();
        assert_eq!(c.bits(), 7);
        assert_eq!(c.checks(), 3);
        assert_eq!(c.codewords().unwrap().len(), 16, "a [7,4] code has 2^4 codewords");
        assert_eq!(c.min_distance().unwrap(), Some(3), "the Hamming code's distance is 3");

        // Perfect: the seven single-bit error syndromes are distinct and non-zero, so with the zero
        // syndrome they are all eight of 2^3.
        let mut seen = vec![c.syndrome(&[0; 7])];
        for i in 0..7 {
            let mut w = [0u8; 7];
            w[i] = 1;
            let s = c.syndrome(&w);
            assert!(s.contains(&1), "bit {i} has a zero syndrome");
            assert!(!seen.contains(&s), "bit {i} shares a syndrome with an earlier error");
            seen.push(s);
        }
        assert_eq!(seen.len(), 8);
    }

    /// THE PARITY PENALTY IS ZERO EXACTLY ON CODEWORDS — both directions, every word.
    #[test]
    fn the_parity_penalty_is_zero_exactly_on_codewords() {
        let c = hamming74();
        let code: Vec<Vec<u8>> = c.codewords().unwrap();
        for w in words(7) {
            let v = c.violations(&c.state(&w));
            assert_eq!(
                v == 0,
                code.contains(&w),
                "word {w:?} violates {v} checks but codeword membership says otherwise"
            );
        }
    }

    /// THE CONSTRUCTION IS AN IDENTITY, AND THIS IS IT: at every state,
    /// `energy + offset = penalty * violations + channel_energy`.
    ///
    /// Checking the identity rather than the optimum is what catches a sign or a factor of two in
    /// the spin convention: `bit 0 -> spin -1` instead of `+1` flips the parity of every odd-weight
    /// check, and a model built that way still has a tidy ground state, for a different code.
    #[test]
    fn the_energy_is_the_penalty_plus_the_channel_at_every_state() {
        let c = hamming74();
        // Deliberately lopsided, so a swapped or halved field cannot hide behind symmetry.
        let llr = [2.0, -1.0, 0.5, -3.25, 0.0, 4.0, -0.125];
        let lambda = c.penalty(&llr).unwrap();
        let (p, offset) = c.to_program(&llr).unwrap();
        for w in words(7) {
            let s = c.state(&w);
            let got = energy(&p, &s) + offset;
            let want =
                lambda * c.violations(&s) as f64 + c.channel_energy(&llr, &s).unwrap();
            assert!((got - want).abs() < 1e-9, "word {w:?}: {got} against {want}");
        }
    }

    /// The penalty bound is strictly above the channel span, which is what makes it work.
    #[test]
    fn the_penalty_strictly_exceeds_the_channel_span() {
        let c = hamming74();
        for llr in [
            vec![0.0f64; 7],
            vec![1.0; 7],
            vec![-7.5, 0.25, 3.0, -0.5, 9.0, 0.0, 2.0],
            vec![1e-16; 7],
        ] {
            let span: f64 = llr.iter().map(|l| l.abs()).sum();
            let lambda = c.penalty(&llr).unwrap();
            assert!(lambda > span, "penalty {lambda} does not clear the span {span}");
        }
    }

    /// THE REDUCTION MOVES NO STATE, checked by enumerating the ancillas as well as the bits.
    ///
    /// Small enough that this is a proof rather than evidence: five bits, two overlapping weight-4
    /// checks, and every ancilla assignment. The overlap matters — the checks share the monomial
    /// `{1,2,3}`, whose coefficients ADD before reduction, and a per-check reduction that missed
    /// the merge would still produce a valid model.
    #[test]
    fn the_reduction_preserves_every_state_on_a_small_code() {
        let mut c = Code::new(5);
        c.check(&[0, 1, 2, 3]).unwrap();
        c.check(&[1, 2, 3, 4]).unwrap();
        let llr = [1.5, -0.5, 0.25, 2.0, -1.0];
        let (p, _) = c.to_program(&llr).unwrap();
        let (g, red) = c.to_graph(&llr).unwrap();
        assert_eq!(red.penalty, 0.0, "the lowering must carry no penalty of its own");
        assert!(red.ancillas > 0, "two weight-4 checks cannot lower for free");

        let k = red.ancillas;
        for w in words(5) {
            let orig = c.state(&w);
            let want = energy(&p, &orig);
            let mut best = f64::INFINITY;
            for mask in 0u32..(1u32 << k) {
                let mut s = orig.clone();
                for a in 0..k {
                    s.push(if mask >> a & 1 == 1 { 1 } else { -1 });
                }
                best = best.min(g.energy(&s));
            }
            assert!(
                (best + red.offset - want).abs() < 1e-9,
                "word {w:?}: reduced minimum {best} plus offset {} against {want}",
                red.offset
            );
        }
    }

    /// THE HEADLINE: every correctable error pattern is corrected, by enumeration.
    ///
    /// The Hamming code is perfect at distance three, so the sixteen codewords times the eight
    /// patterns of weight at most one are all 128 received words, each reached exactly once — the
    /// sweep is therefore complete rather than sampled. The exact solve runs on the REDUCED
    /// pairwise graph, so this checks the k-body model, the penalty-free lowering and the
    /// elimination together, against the codeword that was actually sent.
    #[test]
    fn every_correctable_error_is_corrected() {
        let c = hamming74();
        let code = c.codewords().unwrap();
        assert_eq!(code.len(), 16);
        let mut seen: Vec<Vec<u8>> = Vec::new();
        for cw in &code {
            for e in 0..=7usize {
                let mut y = cw.clone();
                if e < 7 {
                    y[e] ^= 1;
                }
                assert!(!seen.contains(&y), "received word {y:?} reached twice");
                seen.push(y.clone());

                let llr = bsc_llr(&y, 0.1).unwrap();
                let d = c.decode(&llr).unwrap();
                assert!(d.ancillas > 0, "weight-four checks cannot have lowered for free");
                assert_eq!(d.violations, 0, "decoded {:?} is not a codeword", d.word);
                assert_eq!(&d.word, cw, "received {y:?} decoded away from the sent word");
            }
        }
        assert_eq!(seen.len(), 128, "the sweep must cover every received word once");
    }

    /// The decoder attains the LIKELIHOOD OPTIMUM even where the error is not correctable.
    ///
    /// Brute force over the codewords is the oracle: for a binary symmetric channel with
    /// `p < 1/2`, maximum likelihood is minimum Hamming distance. This code has distance two and
    /// genuine ties — received `0100` is one flip from both `0000` and `0110` — so the assertion is
    /// that the answer is a codeword ATTAINING the minimum, not that it is any particular one.
    #[test]
    fn decoding_attains_the_likelihood_optimum_when_correction_is_impossible() {
        let mut c = Code::new(4);
        c.check(&[0, 1, 2]).unwrap();
        c.check(&[1, 2, 3]).unwrap();
        let code = c.codewords().unwrap();
        assert_eq!(c.min_distance().unwrap(), Some(2), "this code must have ties to be worth using");

        for y in words(4) {
            let llr = bsc_llr(&y, 0.2).unwrap();
            let d = c.decode(&llr).unwrap();
            assert_eq!(d.violations, 0, "decoded {:?} is not a codeword", d.word);
            let best = code.iter().map(|cw| distance(cw, &y)).min().unwrap();
            assert_eq!(
                distance(&d.word, &y),
                best,
                "received {y:?} decoded to {:?} at distance {} where {best} was available",
                d.word,
                distance(&d.word, &y)
            );
        }
    }

    /// THE PENALTY BOUND IS LOAD-BEARING, demonstrated rather than asserted.
    ///
    /// Under the bound the ground state simply follows the channel off the code. Without this the
    /// bound is a number nobody has seen fail, and a decoder returning non-codewords is exactly the
    /// failure it prevents.
    #[test]
    fn a_penalty_below_the_bound_decodes_off_the_code() {
        let c = hamming74();
        // Received word at distance 2 from the nearest codeword, so the channel and the code
        // genuinely disagree.
        let y = [1u8, 1, 0, 0, 0, 0, 0];
        let llr = bsc_llr(&y, 0.05).unwrap();
        assert!(!c.is_codeword(&y));

        let sound = c.decode(&llr).unwrap();
        assert_eq!(sound.violations, 0, "the derived penalty must land on the code");

        let (p, _) = c.to_program_with(&llr, 1e-6).unwrap();
        let red = to_pairwise_exact(&p).unwrap();
        let g = red.program.to_graph().unwrap();
        let ex = Elimination::default().ground_state(&g).unwrap();
        let s = ex.ground_state.unwrap();
        assert_eq!(
            c.word(red.project(&s)),
            y.to_vec(),
            "a negligible penalty should leave the channel's own word as the minimum"
        );
        assert!(c.violations(&s) > 0, "and that word is off the code");
    }

    /// The elimination's own energy agrees with the objective read off the decoded word.
    ///
    /// Two offsets sit between them — the model's and the reduction's — and getting either sign
    /// backwards moves the reported energy without moving the answer, so nothing else here would
    /// notice.
    #[test]
    fn the_reported_energy_matches_the_eliminations_own() {
        let c = hamming74();
        let llr = bsc_llr(&[0, 1, 0, 0, 1, 0, 0], 0.15).unwrap();
        let (_, offset) = c.to_program(&llr).unwrap();
        let (g, red) = c.to_graph(&llr).unwrap();
        let ex = Elimination::default().ground_state(&g).unwrap();
        let want = ex.ground_energy.unwrap() + red.offset + offset;
        let d = c.decode(&llr).unwrap();
        assert!(
            (d.energy - want).abs() < 1e-9,
            "decoded energy {} against the elimination's {want}",
            d.energy
        );
    }

    /// A repetition code decodes by majority, which is a closed form to check against.
    #[test]
    fn the_repetition_code_decodes_by_majority() {
        for n in [3usize, 5] {
            let c = repetition(n);
            assert_eq!(c.codewords().unwrap().len(), 2, "a repetition code has two codewords");
            assert_eq!(c.min_distance().unwrap(), Some(n));
            for y in words(n) {
                let ones = y.iter().filter(|&&b| b == 1).count();
                let llr = bsc_llr(&y, 0.25).unwrap();
                let d = c.decode(&llr).unwrap();
                let want = u8::from(ones * 2 > n);
                assert!(d.word.iter().all(|&b| b == want), "{y:?} decoded to {:?}", d.word);
            }
        }
    }

    /// Malformed checks and channels are refused rather than modelled.
    #[test]
    fn bad_input_is_refused() {
        let mut c = Code::new(4);
        assert_eq!(c.check(&[]), Err(Error::EmptyCheck));
        assert_eq!(c.check(&[0, 4]), Err(Error::UnknownBit { bit: 4, bits: 4 }));
        assert_eq!(c.check(&[1, 2, 1]), Err(Error::RepeatedBit { bit: 1 }));
        let wide: Vec<usize> = (0..MAX_CHECK + 1).collect();
        assert_eq!(
            c.check(&wide),
            Err(Error::CheckTooWide { arity: MAX_CHECK + 1, limit: MAX_CHECK })
        );
        c.check(&[0, 1, 2]).unwrap();
        assert_eq!(c.penalty(&[1.0]), Err(Error::LlrLen { got: 1, want: 4 }));
        assert_eq!(
            c.penalty(&[1.0, f64::INFINITY, 0.0, 0.0]),
            Err(Error::NonFinite { bit: 1 })
        );
        assert_eq!(bsc_llr(&[0, 1], 0.0), Err(Error::BadChannel));
        assert_eq!(bsc_llr(&[0, 1], 1.0), Err(Error::BadChannel));
        assert_eq!(awgn_llr(&[0.5], 0.0), Err(Error::BadChannel));
        assert_eq!(awgn_llr(&[f64::NAN], 1.0), Err(Error::NonFinite { bit: 0 }));
        assert_eq!(c.to_program_with(&[0.0; 4], 0.0), Err(Error::BadChannel));
        assert!(bsc_llr(&[0, 1], 0.5).unwrap().iter().all(|&l| l == 0.0));
        assert_eq!(awgn_llr(&[1.5, -0.5], 0.5).unwrap(), vec![6.0, -2.0]);
    }

    /// A code with no checks is the identity decoder: the channel's own word, every time.
    #[test]
    fn a_code_with_no_checks_returns_the_channel_word() {
        let c = Code::new(5);
        for y in words(5) {
            let llr = bsc_llr(&y, 0.3).unwrap();
            let d = c.decode(&llr).unwrap();
            assert_eq!(d.word, y);
            assert_eq!(d.violations, 0);
            assert_eq!(d.ancillas, 0, "nothing to lower means no ancillas");
        }
    }
}
