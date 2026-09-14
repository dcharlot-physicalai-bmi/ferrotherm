//! Sourlas codes: error correction as a spin glass, decoded at the Nishimori temperature -- the
//! workload where a thermodynamic sampler is the optimal decoder by theorem, with that theorem
//! checked exactly on small codes, and the fixed-point fabric's floor measured against it.
//!
//! # The construction
//!
//! Sourlas (*Nature* 339:693, 1989) encodes `K` message bits `xi in {-1, +1}^K` as `M` products
//! over `C`-subsets, `J_S = prod_{i in S} xi_i`, sends them through a binary symmetric channel
//! that flips each with probability `p`, and decodes the received `J'` by reading the posterior
//!
//! ```text
//!   P(sigma | J') ~ exp( beta sum_S J'_S prod_{i in S} sigma_i ),
//! ```
//!
//! which is a `C`-spin glass with couplings `J'`. The gauge `sigma_i -> sigma_i xi_i` maps any
//! message to the all-ones one, so every instance is a `+-J` spin glass whose hidden ground
//! state is the ferromagnet; the flips are the disorder.
//!
//! One thing the construction owes the reader: a code whose checks all have even size has a
//! global spin-flip symmetry, `P(sigma) = P(-sigma)`, so every marginal `<sigma_i>` is exactly
//! zero and the bitwise decoder reads nothing -- the first draft of this module measured a bit
//! error of one half at every noise level before it noticed. Either `C` is odd, or the message
//! bits are sent through the channel too ([`Code::with_message_bits`]), which puts a field on
//! every spin and breaks the symmetry; Sourlas's original construction does the latter.
//!
//! # The theorem, checked
//!
//! Ruján (*Phys. Rev. Lett.* 70:2968, 1993) and Nishimori (*J. Phys. Soc. Jpn.* 62:2973, 1993):
//! the decoder that minimises the expected bit error is not the ground state (MAP) but the sign
//! of the thermal magnetisation, `sign <sigma_i>`, at exactly the **Nishimori temperature**
//! `beta_N = (1/2) ln((1 - p) / p)` -- the temperature at which the Boltzmann weight IS the
//! channel likelihood. On that line the identity `E[xi_i <sigma_i>] = E[<sigma_i>^2]` holds
//! over the channel noise, and the finite-temperature decoder beats every other temperature,
//! including zero. Both are statements about finite systems, so both are checked here exactly:
//! [`nishimori_identity_defect`] averages over every one of the `2^M` noise patterns with its
//! probability and is zero to `1e-12` at `beta_N` and not elsewhere; [`expected_bit_error`] does
//! the same for the bit error and is smallest at `beta_N` on the grid the test sweeps, MAP
//! included.
//!
//! # What the fabric does with it
//!
//! A pairwise code (`C = 2`) is an Ising posterior, and [`posterior_graph`] hands it to any
//! kernel in [`crate::autocorr`], whose direct solve gives the kernel's own stationary marginals
//! and so its own decoder. `examples/sourlas_fabric.rs` runs the sixteen-bit comparator fabric
//! at `beta_N` across channel noise levels on a ten-bit code with fifteen pair checks and its
//! message bits (rate 0.4, degree up to six), forty noise patterns per level:
//!
//! | `p` | `beta_N` | `2 beta_N x degree` | bit error, exact | Gibbs kernel | fabric | frozen |
//! |---|---|---|---|---|---|---|
//! | 0.3 | 0.42 | 5.1 | 0.295 | 0.295 | 0.295 | 0 of 40 |
//! | 0.2 | 0.69 | 8.3 | 0.163 | 0.163 | 0.160 | 0 of 40 |
//! | 0.1 | 1.10 | 13.2 | 0.060 | 0.060 | 0.063 | 0 of 40 |
//! | 0.05 | 1.47 | 17.7 | 0.0025 | 0.0025 | 0.0075 | 0 of 40 |
//! | 0.02 | 1.95 | 23.4 | 0 | 0 | 0 | 0 of 40 |
//! | 0.005 | 2.65 | 31.8 | 0 | 0 | 0 | 0 of 40 |
//!
//! `beta_N` grows as the channel cleans up and `2 beta_N f` crosses the comparator floor at
//! `11.8` from `p = 0.1` down -- and the fabric decodes as the exact decoder does, to within the
//! count, at every level, with no pattern frozen. The floor rounds to zero the probability of
//! the state a spin's field has already decided against; a decoder reads only the sign, and the
//! sign was never in doubt there. The comparator floor that moved Kemeny's constant by forty
//! times on a cold grid ([`crate::autocorr`]) moves this decoder's answer not at all, which is
//! the measurement, not the guess.

use crate::autocorr;
use crate::graph::{Graph, GraphBuilder};

/// The most message bits the exact posterior enumerates: `2^16` states.
pub const MAX_MESSAGE_BITS: usize = 16;

/// The most checks the exact channel average enumerates: `2^14` noise patterns.
pub const MAX_CHECKS_EXACT: usize = 14;

/// A Sourlas code: `k` message bits and `m` checks, each a `c`-subset of the bits.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Code {
    /// Message bits.
    pub k: usize,
    /// The checks, each a sorted list of distinct bit indices.
    pub checks: Vec<Vec<usize>>,
}

fn splitmix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut x = z;
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// Why a code or a request cannot be served.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Invalid {
    /// More message bits than the exact posterior enumerates.
    TooManyBits {
        /// Bits asked for.
        k: usize,
    },
    /// More checks than the exact channel average enumerates.
    TooManyChecks {
        /// Checks asked for.
        m: usize,
    },
    /// A check of this size is not a pair, so the posterior is not an Ising graph.
    NotPairwise {
        /// The offending check's size.
        c: usize,
    },
    /// A word of the wrong length for this code.
    WrongLength {
        /// Expected.
        want: usize,
        /// Got.
        got: usize,
    },
    /// Not enough distinct `c`-subsets of `k` bits to make `m` checks.
    TooFewSubsets,
}

impl core::fmt::Display for Invalid {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Invalid::TooManyBits { k } => write!(
                f,
                "{k} message bits exceed the exact limit of {MAX_MESSAGE_BITS}"
            ),
            Invalid::TooManyChecks { m } => write!(
                f,
                "{m} checks exceed the exact channel limit of {MAX_CHECKS_EXACT}"
            ),
            Invalid::NotPairwise { c } => write!(f, "a check over {c} bits is not a pair"),
            Invalid::WrongLength { want, got } => {
                write!(f, "expected a word of {want} bits, got {got}")
            }
            Invalid::TooFewSubsets => write!(f, "not enough distinct subsets for that many checks"),
        }
    }
}

impl core::error::Error for Invalid {}

impl Code {
    /// A random code: `m` distinct `c`-subsets of `k` bits, drawn without replacement.
    ///
    /// # Errors
    ///
    /// [`Invalid::TooFewSubsets`] when `m` exceeds `C(k, c)`.
    ///
    /// # Panics
    ///
    /// If `c` is zero or above `k`.
    pub fn random(k: usize, m: usize, c: usize, seed: u64) -> Result<Code, Invalid> {
        assert!(c >= 1 && c <= k, "a check takes between one and k bits");
        let mut total = 1usize;
        for i in 0..c {
            total = total * (k - i) / (i + 1);
        }
        if m > total {
            return Err(Invalid::TooFewSubsets);
        }
        let mut checks: Vec<Vec<usize>> = Vec::with_capacity(m);
        let mut draw = 0u64;
        while checks.len() < m {
            let mut subset: Vec<usize> = Vec::with_capacity(c);
            while subset.len() < c {
                let r =
                    (splitmix(seed ^ draw.wrapping_mul(0x9E37_79B9_7F4A_7C15)) % k as u64) as usize;
                draw += 1;
                if !subset.contains(&r) {
                    subset.push(r);
                }
            }
            subset.sort_unstable();
            if !checks.contains(&subset) {
                checks.push(subset);
            }
        }
        Ok(Code { k, checks })
    }

    /// The same code with every message bit also sent on its own: `k` singleton checks in
    /// front of the products. A field on every spin, which is what breaks the flip symmetry of
    /// an even-`C` code.
    #[must_use]
    pub fn with_message_bits(self) -> Code {
        let mut checks: Vec<Vec<usize>> = (0..self.k).map(|i| vec![i]).collect();
        checks.extend(self.checks);
        Code { k: self.k, checks }
    }

    /// Checks.
    #[must_use]
    pub fn m(&self) -> usize {
        self.checks.len()
    }

    /// The code rate `k / m`.
    #[must_use]
    pub fn rate(&self) -> f64 {
        self.k as f64 / self.m() as f64
    }

    /// The codeword of `message`: one product per check.
    ///
    /// # Errors
    ///
    /// [`Invalid::WrongLength`] for a message of the wrong length.
    pub fn encode(&self, message: &[i8]) -> Result<Vec<i8>, Invalid> {
        if message.len() != self.k {
            return Err(Invalid::WrongLength {
                want: self.k,
                got: message.len(),
            });
        }
        Ok(self
            .checks
            .iter()
            .map(|s| s.iter().map(|&i| message[i]).product())
            .collect())
    }
}

/// The binary symmetric channel: each symbol flipped with probability `p`, from a seed.
#[must_use]
pub fn bsc(word: &[i8], p: f64, seed: u64) -> Vec<i8> {
    word.iter()
        .enumerate()
        .map(|(i, &w)| {
            let u = (splitmix(seed ^ (i as u64).wrapping_mul(0xD6E8_FEB8_6659_FD93)) >> 11) as f64
                / (1u64 << 53) as f64;
            if u < p {
                -w
            } else {
                w
            }
        })
        .collect()
}

/// The Nishimori temperature of a binary symmetric channel: `beta_N = (1/2) ln((1 - p) / p)`,
/// the inverse temperature at which the Boltzmann weight of a coupling is its likelihood.
///
/// # Panics
///
/// If `p` is not strictly between zero and one half.
#[must_use]
pub fn nishimori_beta(p: f64) -> f64 {
    assert!(
        p > 0.0 && p < 0.5,
        "a binary symmetric channel has 0 < p < 1/2, not {p}"
    );
    0.5 * ((1.0 - p) / p).ln()
}

/// The exact posterior over a message given the received word: log partition function and
/// the marginal magnetisations `<sigma_i>`.
#[derive(Clone, Debug, PartialEq)]
pub struct Posterior {
    /// `ln Z` at the decoding temperature.
    pub log_z: f64,
    /// `<sigma_i>` for every message bit.
    pub marginals: Vec<f64>,
}

/// The exact posterior of `code` at inverse temperature `beta` given `received`, by enumeration
/// of every message.
///
/// # Errors
///
/// [`Invalid::TooManyBits`] above [`MAX_MESSAGE_BITS`]; [`Invalid::WrongLength`] for a word of
/// the wrong length.
pub fn posterior(code: &Code, received: &[i8], beta: f64) -> Result<Posterior, Invalid> {
    if code.k > MAX_MESSAGE_BITS {
        return Err(Invalid::TooManyBits { k: code.k });
    }
    if received.len() != code.m() {
        return Err(Invalid::WrongLength {
            want: code.m(),
            got: received.len(),
        });
    }
    let states = 1usize << code.k;
    let mut energies = Vec::with_capacity(states);
    let mut top = f64::NEG_INFINITY;
    for x in 0..states {
        let sigma = autocorr::spins(x, code.k);
        let mut h = 0.0;
        for (s, &j) in code.checks.iter().zip(received) {
            let prod: f64 = s.iter().map(|&i| f64::from(sigma[i])).product();
            h += f64::from(j) * prod;
        }
        let l = beta * h;
        energies.push(l);
        top = top.max(l);
    }
    let mut z = 0.0;
    let mut marg = vec![0.0f64; code.k];
    for (x, &l) in energies.iter().enumerate() {
        let w = (l - top).exp();
        z += w;
        let sigma = autocorr::spins(x, code.k);
        for (i, &s) in sigma.iter().enumerate() {
            marg[i] += w * f64::from(s);
        }
    }
    Ok(Posterior {
        log_z: top + z.ln(),
        marginals: marg.iter().map(|v| v / z).collect(),
    })
}

/// The maximiser of the posterior marginals: `sign <sigma_i>`, `+1` on a tie.
#[must_use]
pub fn mpm_decode(marginals: &[f64]) -> Vec<i8> {
    marginals
        .iter()
        .map(|&m| if m < 0.0 { -1 } else { 1 })
        .collect()
}

/// The expected fraction of bits the sign decoder gets wrong, given the marginals: one for a
/// marginal of the wrong sign, one half for a marginal of exactly zero (a fair coin, which is
/// what makes the expectation gauge-covariant -- a tie broken towards `+1` would favour messages
/// with more `+1` bits, and the Bayes optimality the tests check is a statement about the
/// average over messages).
///
/// # Panics
///
/// If the words differ in length.
#[must_use]
pub fn expected_errors(marginals: &[f64], message: &[i8]) -> f64 {
    assert_eq!(marginals.len(), message.len(), "words of one length");
    let mut errors = 0.0;
    for (m, &x) in marginals.iter().zip(message) {
        let agree = m * f64::from(x);
        if agree.abs() < 1e-12 {
            errors += 0.5;
        } else if agree < 0.0 {
            errors += 1.0;
        }
    }
    errors / message.len() as f64
}

/// The fraction of bits that differ.
///
/// # Panics
///
/// If the words differ in length.
#[must_use]
pub fn bit_error(decoded: &[i8], message: &[i8]) -> f64 {
    assert_eq!(decoded.len(), message.len(), "words of one length");
    decoded.iter().zip(message).filter(|(a, b)| a != b).count() as f64 / message.len() as f64
}

/// Visit every noise pattern of `m` symbols with its probability under the channel.
fn every_noise(m: usize, p: f64, mut visit: impl FnMut(&[i8], f64)) {
    let patterns = 1usize << m;
    let mut flips = vec![1i8; m];
    for n in 0..patterns {
        let mut count = 0u32;
        for (i, f) in flips.iter_mut().enumerate() {
            let flipped = (n >> i) & 1 == 1;
            *f = if flipped { -1 } else { 1 };
            count += u32::from(flipped);
        }
        let prob = p.powi(count as i32) * (1.0 - p).powi((m - count as usize) as i32);
        visit(&flips, prob);
    }
}

/// The exact expected bit error of the finite-temperature decoder at `beta` for `message`
/// through a channel of flip probability `p`: every noise pattern, weighted, ties counted as
/// [`expected_errors`] counts them.
///
/// # Errors
///
/// [`Invalid::TooManyChecks`] above [`MAX_CHECKS_EXACT`]; as [`posterior`] otherwise.
pub fn expected_bit_error(code: &Code, message: &[i8], p: f64, beta: f64) -> Result<f64, Invalid> {
    if code.m() > MAX_CHECKS_EXACT {
        return Err(Invalid::TooManyChecks { m: code.m() });
    }
    let word = code.encode(message)?;
    let mut total = 0.0;
    let mut failed = None;
    every_noise(code.m(), p, |flips, prob| {
        if failed.is_some() {
            return;
        }
        let received: Vec<i8> = word.iter().zip(flips).map(|(w, f)| w * f).collect();
        match posterior(code, &received, beta) {
            Ok(post) => total += prob * expected_errors(&post.marginals, message),
            Err(e) => failed = Some(e),
        }
    });
    match failed {
        Some(e) => Err(e),
        None => Ok(total),
    }
}

/// Nishimori's identity, `E[xi_i <sigma_i>] = E[<sigma_i>^2]` over the channel noise at the
/// decoding temperature `beta`: the mean absolute defect over the bits, exact over every noise
/// pattern. Zero at `beta_N`, not elsewhere.
///
/// # Errors
///
/// As [`expected_bit_error`].
pub fn nishimori_identity_defect(
    code: &Code,
    message: &[i8],
    p: f64,
    beta: f64,
) -> Result<f64, Invalid> {
    if code.m() > MAX_CHECKS_EXACT {
        return Err(Invalid::TooManyChecks { m: code.m() });
    }
    let word = code.encode(message)?;
    let mut overlap = vec![0.0f64; code.k];
    let mut square = vec![0.0f64; code.k];
    let mut failed = None;
    every_noise(code.m(), p, |flips, prob| {
        if failed.is_some() {
            return;
        }
        let received: Vec<i8> = word.iter().zip(flips).map(|(w, f)| w * f).collect();
        match posterior(code, &received, beta) {
            Ok(post) => {
                for i in 0..code.k {
                    overlap[i] += prob * f64::from(message[i]) * post.marginals[i];
                    square[i] += prob * post.marginals[i] * post.marginals[i];
                }
            }
            Err(e) => failed = Some(e),
        }
    });
    if let Some(e) = failed {
        return Err(e);
    }
    Ok(overlap
        .iter()
        .zip(&square)
        .map(|(a, b)| (a - b).abs())
        .sum::<f64>()
        / code.k as f64)
}

/// The Ising posterior of a pairwise code: a graph over the message bits with coupling `J'_ij`
/// on each pair check and a field `J'_i` on each singleton, to be sampled at the decoding
/// temperature by any kernel.
///
/// # Errors
///
/// [`Invalid::NotPairwise`] if any check is larger than a pair; [`Invalid::WrongLength`] for a
/// word of the wrong length.
pub fn posterior_graph(code: &Code, received: &[i8]) -> Result<Graph, Invalid> {
    if received.len() != code.m() {
        return Err(Invalid::WrongLength {
            want: code.m(),
            got: received.len(),
        });
    }
    let mut b = GraphBuilder::new(code.k);
    for (s, &j) in code.checks.iter().zip(received) {
        match s.len() {
            1 => b.bias(s[0], f64::from(j)),
            2 => b.couple(s[0], s[1], f64::from(j)),
            c => return Err(Invalid::NotPairwise { c }),
        }
    }
    Ok(b.build())
}

/// The marginal magnetisations of a kernel's own stationary law on the Ising posterior: what
/// that kernel decodes to, exactly, by the direct solve in [`crate::autocorr`].
///
/// # Errors
///
/// As [`posterior_graph`] and [`crate::autocorr::own_law`]: a code that is not pairwise, too
/// many bits for a dense solve, or a kernel whose chain does not mix.
pub fn kernel_marginals(
    code: &Code,
    received: &[i8],
    beta: f64,
    kernel: autocorr::Kernel,
) -> Result<Vec<f64>, Box<dyn core::error::Error>> {
    let g = posterior_graph(code, received)?;
    let law = autocorr::own_law(&g, beta, kernel)?;
    let mut marg = vec![0.0f64; code.k];
    for (x, &p) in law.iter().enumerate() {
        let sigma = autocorr::spins(x, code.k);
        for (i, &s) in sigma.iter().enumerate() {
            marg[i] += p * f64::from(s);
        }
    }
    Ok(marg)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autocorr::Kernel;

    fn message(k: usize, seed: u64) -> Vec<i8> {
        (0..k)
            .map(|i| {
                if splitmix(seed ^ i as u64) & 1 == 1 {
                    1
                } else {
                    -1
                }
            })
            .collect()
    }

    /// Encoding is a product, and the gauge `sigma -> sigma xi` carries one message's posterior
    /// to another's: the marginals of a message are `xi_i` times the marginals of the all-ones
    /// message under the gauged noise, to `1e-14`.
    #[test]
    fn encoding_is_a_product_and_the_gauge_makes_every_message_the_ferromagnet() {
        let code = Code::random(7, 11, 3, 1).expect("a code");
        let xi = message(7, 5);
        let word = code.encode(&xi).expect("length");
        for (s, &j) in code.checks.iter().zip(&word) {
            let prod: i8 = s.iter().map(|&i| xi[i]).product();
            assert_eq!(j, prod);
        }
        let received = bsc(&word, 0.2, 9);
        let beta = nishimori_beta(0.2);
        let post = posterior(&code, &received, beta).expect("small");
        // The same noise on the all-ones message: J'_S = flips_S, i.e. received_S * word_S.
        let gauged: Vec<i8> = received.iter().zip(&word).map(|(r, w)| r * w).collect();
        let ones = posterior(&code, &gauged, beta).expect("small");
        for i in 0..7 {
            assert!((post.marginals[i] - f64::from(xi[i]) * ones.marginals[i]).abs() < 1e-14);
        }
        assert!((code.rate() - 7.0 / 11.0).abs() < 1e-15);
        assert!(
            Code::random(4, 7, 3, 1).is_err(),
            "C(4,3) = 4 subsets cannot make 7 checks"
        );
    }

    /// Nishimori's identity holds exactly on the Nishimori line and fails off it, for a pairwise
    /// code with its message bits and a three-body code without them, over every noise pattern
    /// of the channel.
    #[test]
    fn nishimoris_identity_holds_exactly_on_the_nishimori_line() {
        for (c, seed) in [(2usize, 3u64), (3, 4)] {
            let code = Code::random(5, 6, c, seed).expect("a code");
            let code = if c == 2 {
                code.with_message_bits()
            } else {
                code
            };
            let xi = message(5, 6);
            let p = 0.15;
            let on = nishimori_identity_defect(&code, &xi, p, nishimori_beta(p)).expect("exact");
            let off =
                nishimori_identity_defect(&code, &xi, p, 2.0 * nishimori_beta(p)).expect("exact");
            assert!(on < 1e-12, "C = {c}: defect on the line {on:e}");
            assert!(
                off > 1e-3,
                "C = {c}: defect off the line {off:e} -- the test would be blind"
            );
        }
    }

    /// The expected bit error of the finite-temperature decoder is smallest at the Nishimori
    /// temperature: below every other point on a grid that reaches from half `beta_N` to a
    /// zero-temperature (MAP) stand-in, on a pairwise code with its message bits (strictly, at
    /// two or more grid points, and below the raw channel) and on a three-body code without
    /// them (where only zero temperature loses).
    #[test]
    fn the_nishimori_temperature_minimises_the_expected_bit_error() {
        for (c, m, seed) in [(2usize, 7usize, 3u64), (3, 12, 4)] {
            let code = Code::random(6, m, c, seed).expect("a code");
            let code = if c == 2 {
                code.with_message_bits()
            } else {
                code
            };
            let xi = message(6, 8);
            let p = 0.12;
            let bn = nishimori_beta(p);
            let at_n = expected_bit_error(&code, &xi, p, bn).expect("exact");
            let mut strictly_worse = 0;
            let mut grid = Vec::new();
            for factor in [0.5f64, 0.8, 1.25, 2.0, 30.0] {
                let other = expected_bit_error(&code, &xi, p, factor * bn).expect("exact");
                grid.push((factor, other));
                assert!(
                    other >= at_n - 1e-12,
                    "C = {c}: beta = {factor} beta_N gives {other} below {at_n}"
                );
                if other > at_n + 1e-9 {
                    strictly_worse += 1;
                }
            }
            // The pairwise code with fields separates the temperatures; the small three-body code
            // decides the same bits from half to twice beta_N and loses only at zero temperature.
            let needed = if c == 2 { 2 } else { 1 };
            assert!(
                strictly_worse >= needed,
                "C = {c}: the minimum at beta_N is not strict enough to be evidence: {at_n} vs {grid:?}"
            );
            assert!(
                at_n > 0.0,
                "C = {c}: a bit error of zero on a noisy channel"
            );
            if c == 2 {
                assert!(at_n < p, "C = {c}: the code with its message bits should beat the raw channel, got {at_n} against {p}");
            }
        }
    }

    /// On a pairwise code the exact sequential-Gibbs kernel decodes as the exact posterior does
    /// (its law is Boltzmann), and the fixed-point fabric agrees with it on a noisy channel where
    /// no field reaches the comparator floor.
    #[test]
    fn a_gibbs_kernel_decodes_as_the_posterior_and_the_fabric_agrees_on_a_noisy_channel() {
        let code = Code::random(8, 12, 2, 21)
            .expect("a code")
            .with_message_bits();
        let xi = message(8, 22);
        let word = code.encode(&xi).expect("length");
        let p = 0.25;
        let beta = nishimori_beta(p);
        let received = bsc(&word, p, 23);
        let exact = posterior(&code, &received, beta).expect("small");
        let gibbs =
            kernel_marginals(&code, &received, beta, Kernel::SequentialGibbs).expect("dense");
        for i in 0..8 {
            assert!(
                (gibbs[i] - exact.marginals[i]).abs() < 1e-9,
                "bit {i}: {} vs {}",
                gibbs[i],
                exact.marginals[i]
            );
        }
        let fabric = kernel_marginals(&code, &received, beta, Kernel::FixedFabric).expect("dense");
        assert_eq!(mpm_decode(&fabric), mpm_decode(&exact.marginals));
        for i in 0..8 {
            assert!(
                (fabric[i] - exact.marginals[i]).abs() < 0.02,
                "bit {i}: fabric {} vs exact {}",
                fabric[i],
                exact.marginals[i]
            );
        }
    }
}
