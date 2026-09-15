//! Syndrome decoding of a parity-check code as a spin glass: the posterior over noise spins, its
//! exact marginals, belief propagation, and the relaxed C-body model a thermodynamic sampler can
//! hold -- with the check strength that makes the relaxation honest measured against the mixing
//! it costs. [`crate::ldpc`] decodes the same codes as a ground-state search with an exact
//! penalty; this module is the finite-temperature half, where the decoder is a marginal and the
//! penalty is a strength to measure.
//!
//! # The construction
//!
//! A code is the nullspace of a sparse check matrix `H` ([`crate::gf2`]); a received word
//! `y = x + n` has syndrome `z = H y = H n`, which depends on the noise alone. Decoding is
//! finding the most probable noise pattern with that syndrome (Gallager 1962). In spin language
//! (Sourlas 1994; Kabashima and Saad 1999) the noise bits are spins `tau_i = (-1)^(n_i)`, each
//! check `mu` demands `prod_{i in mu} tau_i = J_mu = (-1)^(z_mu)`, and the channel puts a field
//! `F = (1/2) ln((1 - p)/p)` -- the Nishimori temperature of [`crate::sourlas`] -- on every
//! spin, favouring "no flip":
//!
//! ```text
//!   P(tau | z) ~ prod_mu delta( prod_{i in mu} tau_i, J_mu ) exp( F sum_i tau_i ).
//! ```
//!
//! The delta is a hard constraint, and a single-spin sampler cannot move under it (one flip
//! breaks every check the spin sits on). The sampler's model is the **relaxed** one: the delta
//! becomes a `C`-body coupling `gamma J_mu prod tau` of strength `gamma`, exact as
//! `gamma -> infinity` and mobile at small `gamma`. That is a penalty, and the whole crate's
//! lesson about penalties ([`crate::pdit`], [`crate::npising`]) applies: the strength that
//! makes the relaxation honest is the strength that freezes the chain.
//!
//! # What is checked
//!
//! * A regular code has the column and row weights it was asked for, and its codewords have
//!   zero syndrome ([`Code::regular`], [`Code::syndrome`]).
//! * On a Tanner graph without cycles, belief propagation ([`bp_marginals`]) is the exact
//!   posterior ([`hard_marginals`]) to `1e-9`: the theorem (Pearl 1988) on a code built to be a
//!   tree. On a code with cycles it is not, which the same test records.
//! * The relaxed marginals ([`relaxed_marginals`]) converge to the hard ones as `gamma` grows,
//!   and the sequential Gibbs chain over the relaxed model has the model's Boltzmann law as its
//!   stationary law while its autocorrelation time grows with `gamma`
//!   ([`relaxed_sweep_kernel`], with [`crate::pdit`]'s dense-chain measures).
//!
//! `examples/syndrome_gibbs.rs` puts the numbers side by side on a (3,5)-regular code of ten
//! bits (six checks, rate one half, cyclic), thirty noise patterns per channel, the relaxed
//! chain's autocorrelation time of one noise spin in sweeps:
//!
//! | `p` | hard posterior | belief propagation | `gamma` | relaxed | `tau_int`, sweeps |
//! |---|---|---|---|---|---|
//! | 0.05 | 0.0233 | 0.0400 | 0.5 | 0.0333 | 1.25 |
//! | | | | 1 | 0.0233 | 5.5 |
//! | | | | 2 | 0.0233 | 4,111 |
//! | | | | 4 | 0.0233 | 8.8e8 |
//! | | | | 8 | 0.0233 | frozen: no unique law |
//! | 0.1 | 0.0500 | 0.0467 | 0.5 | 0.0800 | 2.0 |
//! | | | | 1 | 0.0467 | 21.6 |
//! | | | | 2 | 0.0467 | 11,812 |
//! | | | | 4 | 0.0467 | 2.1e9 |
//! | | | | 8 | 0.0467 | frozen: no unique law |
//!
//! A check strength of one already decodes as the hard posterior does on every pattern: the
//! relaxed marginals differ from the hard ones, their signs do not. Every doubling of the
//! strength past that buys nothing in bit error and multiplies the chain's autocorrelation time
//! by about a thousand, until at eight the direct solve finds no unique stationary law at all.
//! The strength that makes a relaxation exact is the strength that freezes it, and the decoder
//! never needed it; a Gibbs decoder should run the checks as soft as the bit error allows. On
//! this code belief propagation, run forty rounds, sits within a bit of the exact posterior.

use crate::gf2::Matrix;
use crate::het::{exact_boltzmann, HetBuilder, HetGraph, Kind};
use crate::sourlas::nishimori_beta;

/// The most noise bits the exact posterior enumerates: `2^16` patterns.
pub const MAX_BITS_EXACT: usize = 16;

/// A parity-check code: `m` checks over `n` bits.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Code {
    /// The check matrix, `m x n`.
    pub h: Matrix,
}

/// Why a request cannot be served.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Invalid {
    /// `n dv` is not `m dc` for any integer `m`: no regular code has those weights.
    NoRegularCode,
    /// A random socket assignment kept producing a repeated edge; the parameters are too tight.
    CouldNotPlace,
    /// More bits than the exact enumeration allows.
    TooManyBits {
        /// Bits asked for.
        n: usize,
    },
    /// A word of the wrong length.
    WrongLength {
        /// Expected.
        want: usize,
        /// Got.
        got: usize,
    },
    /// No noise pattern has this syndrome, which cannot happen for a syndrome read from a word.
    NoPattern,
}

impl core::fmt::Display for Invalid {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Invalid::NoRegularCode => write!(f, "n dv must equal m dc for a regular code"),
            Invalid::CouldNotPlace => write!(f, "could not place the edges without a repeat"),
            Invalid::TooManyBits { n } => {
                write!(f, "{n} bits exceed the exact limit of {MAX_BITS_EXACT}")
            }
            Invalid::WrongLength { want, got } => write!(f, "expected {want} bits, got {got}"),
            Invalid::NoPattern => write!(f, "no noise pattern has that syndrome"),
        }
    }
}

impl core::error::Error for Invalid {}

fn splitmix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut x = z;
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

impl Code {
    /// A code from its check matrix.
    #[must_use]
    pub fn from_matrix(h: Matrix) -> Code {
        Code { h }
    }

    /// A regular Gallager code: every bit in `dv` checks, every check over `dc` bits. Each check
    /// draws `dc` distinct bits from those with degree still to fill, in a random order; a draw
    /// that runs out of distinct bits restarts, up to a thousand times.
    ///
    /// # Errors
    ///
    /// [`Invalid::NoRegularCode`] unless `n dv` is a multiple of `dc`; [`Invalid::CouldNotPlace`]
    /// if a thousand attempts each ran out of distinct bits.
    pub fn regular(n: usize, dv: usize, dc: usize, seed: u64) -> Result<Code, Invalid> {
        if dv == 0 || dc == 0 || !(n * dv).is_multiple_of(dc) || dc > n {
            return Err(Invalid::NoRegularCode);
        }
        let m = n * dv / dc;
        for attempt in 0..1000u64 {
            let mut state = seed ^ attempt.wrapping_mul(0x9E37_79B9_7F4A_7C15);
            let mut remaining = vec![dv; n];
            let mut h = Matrix::zeros(m, n);
            let mut ok = true;
            for check in 0..m {
                // Bits with capacity left, shuffled; the check takes the first dc of them.
                let mut pool: Vec<usize> = (0..n).filter(|&b| remaining[b] > 0).collect();
                for i in (1..pool.len()).rev() {
                    state = splitmix(state);
                    let j = (state % (i as u64 + 1)) as usize;
                    pool.swap(i, j);
                }
                if pool.len() < dc {
                    ok = false;
                    break;
                }
                for &bit in &pool[..dc] {
                    h.set(check, bit, true);
                    remaining[bit] -= 1;
                }
            }
            if ok {
                return Ok(Code { h });
            }
        }
        Err(Invalid::CouldNotPlace)
    }

    /// Bits.
    #[must_use]
    pub fn n(&self) -> usize {
        self.h.cols
    }

    /// Checks.
    #[must_use]
    pub fn m(&self) -> usize {
        self.h.rows
    }

    /// The rate, `(n - rank H) / n`.
    #[must_use]
    pub fn rate(&self) -> f64 {
        (self.n() - self.h.rank()) as f64 / self.n() as f64
    }

    /// The bits of each check.
    #[must_use]
    pub fn checks(&self) -> Vec<Vec<usize>> {
        (0..self.m())
            .map(|r| (0..self.n()).filter(|&c| self.h.get(r, c)).collect())
            .collect()
    }

    /// `H y`.
    ///
    /// # Errors
    ///
    /// [`Invalid::WrongLength`] for a word of the wrong length.
    pub fn syndrome(&self, y: &[u8]) -> Result<Vec<u8>, Invalid> {
        if y.len() != self.n() {
            return Err(Invalid::WrongLength {
                want: self.n(),
                got: y.len(),
            });
        }
        Ok(self.h.mul_vec(y))
    }

    /// A random codeword: a random combination of the nullspace basis.
    #[must_use]
    pub fn random_codeword(&self, seed: u64) -> Vec<u8> {
        let basis = self.h.nullspace();
        let mut x = vec![0u8; self.n()];
        for (k, v) in basis.iter().enumerate() {
            if splitmix(seed ^ k as u64) & 1 == 1 {
                for (xi, vi) in x.iter_mut().zip(v) {
                    *xi ^= vi;
                }
            }
        }
        x
    }

    /// Whether the Tanner graph (bits and checks as nodes, ones of `H` as edges) has no cycle:
    /// a forest has exactly `nodes - components` edges.
    #[must_use]
    pub fn tanner_is_forest(&self) -> bool {
        let nodes = self.n() + self.m();
        let mut parent: Vec<usize> = (0..nodes).collect();
        fn find(parent: &mut [usize], mut x: usize) -> usize {
            while parent[x] != x {
                parent[x] = parent[parent[x]];
                x = parent[x];
            }
            x
        }
        let mut edges = 0;
        for (r, check) in self.checks().iter().enumerate() {
            for &c in check {
                edges += 1;
                let a = find(&mut parent, c);
                let b = find(&mut parent, self.n() + r);
                if a == b {
                    return false;
                }
                parent[a] = b;
            }
        }
        let components = (0..nodes).filter(|&x| find(&mut parent, x) == x).count();
        edges == nodes - components
    }
}

/// The binary symmetric channel on a word: each bit flipped with probability `p`.
#[must_use]
pub fn bsc(word: &[u8], p: f64, seed: u64) -> Vec<u8> {
    word.iter()
        .enumerate()
        .map(|(i, &b)| {
            let u = (splitmix(seed ^ (i as u64).wrapping_mul(0xD6E8_FEB8_6659_FD93)) >> 11) as f64
                / (1u64 << 53) as f64;
            if u < p {
                b ^ 1
            } else {
                b
            }
        })
        .collect()
}

/// The exact marginals `<tau_i>` of the noise spins under the hard posterior: every noise
/// pattern with the syndrome, weighted by the channel.
///
/// # Errors
///
/// [`Invalid::TooManyBits`] above [`MAX_BITS_EXACT`]; [`Invalid::WrongLength`] for a syndrome of
/// the wrong length; [`Invalid::NoPattern`] if nothing has that syndrome.
pub fn hard_marginals(code: &Code, syndrome: &[u8], p: f64) -> Result<Vec<f64>, Invalid> {
    let n = code.n();
    if n > MAX_BITS_EXACT {
        return Err(Invalid::TooManyBits { n });
    }
    if syndrome.len() != code.m() {
        return Err(Invalid::WrongLength {
            want: code.m(),
            got: syndrome.len(),
        });
    }
    let mut z = 0.0;
    let mut marg = vec![0.0f64; n];
    let mut pattern = vec![0u8; n];
    let ratio = p / (1.0 - p);
    for x in 0..1usize << n {
        let mut flips = 0i32;
        for (i, b) in pattern.iter_mut().enumerate() {
            *b = ((x >> i) & 1) as u8;
            flips += i32::from(*b);
        }
        if code.h.mul_vec(&pattern) != syndrome {
            continue;
        }
        let w = ratio.powi(flips);
        z += w;
        for (i, &b) in pattern.iter().enumerate() {
            marg[i] += w * if b == 1 { -1.0 } else { 1.0 };
        }
    }
    if z == 0.0 {
        return Err(Invalid::NoPattern);
    }
    Ok(marg.iter().map(|v| v / z).collect())
}

/// Belief propagation (sum-product in the log-likelihood domain) on the syndrome posterior for
/// `iterations` rounds of flooding: the marginals `<tau_i> = tanh(L_i / 2)` it settles on.
///
/// # Panics
///
/// If the syndrome has the wrong length.
#[must_use]
pub fn bp_marginals(code: &Code, syndrome: &[u8], p: f64, iterations: usize) -> Vec<f64> {
    assert_eq!(syndrome.len(), code.m(), "a syndrome of {} bits", code.m());
    let checks = code.checks();
    let prior = 2.0 * nishimori_beta(p);
    // Messages indexed by (check, position in check).
    let mut to_var: Vec<Vec<f64>> = checks.iter().map(|c| vec![0.0; c.len()]).collect();
    let mut to_check: Vec<Vec<f64>> = checks.iter().map(|c| vec![prior; c.len()]).collect();
    let n = code.n();
    // Where each variable appears: (check, position).
    let mut appearances: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n];
    for (mu, c) in checks.iter().enumerate() {
        for (pos, &i) in c.iter().enumerate() {
            appearances[i].push((mu, pos));
        }
    }
    for _ in 0..iterations {
        for (mu, c) in checks.iter().enumerate() {
            let sign = if syndrome[mu] == 1 { -1.0 } else { 1.0 };
            for pos in 0..c.len() {
                let mut prod = 1.0;
                for (other, &m) in to_check[mu].iter().enumerate() {
                    if other != pos {
                        prod *= (0.5 * m).tanh();
                    }
                }
                let clipped = prod.clamp(-1.0 + 1e-15, 1.0 - 1e-15);
                to_var[mu][pos] = sign * 2.0 * clipped.atanh();
            }
        }
        for i in 0..n {
            let total: f64 = appearances[i]
                .iter()
                .map(|&(mu, pos)| to_var[mu][pos])
                .sum();
            for &(mu, pos) in &appearances[i] {
                to_check[mu][pos] = prior + total - to_var[mu][pos];
            }
        }
    }
    (0..n)
        .map(|i| {
            let l = prior
                + appearances[i]
                    .iter()
                    .map(|&(mu, pos)| to_var[mu][pos])
                    .sum::<f64>();
            (0.5 * l).tanh()
        })
        .collect()
}

/// The relaxed model at unit inverse temperature: a spin per noise bit with field
/// `F = beta_N(p)`, and a `C`-body factor of energy `-gamma J_mu prod tau` per check.
///
/// # Panics
///
/// If the syndrome has the wrong length.
#[must_use]
pub fn relaxed_graph(code: &Code, syndrome: &[u8], p: f64, gamma: f64) -> HetGraph {
    assert_eq!(syndrome.len(), code.m(), "a syndrome of {} bits", code.m());
    let mut b = HetBuilder::new();
    let nodes: Vec<u32> = (0..code.n()).map(|_| b.node(Kind::Spin)).collect();
    let field = nishimori_beta(p);
    for &node in &nodes {
        b.bias_spin(node, field);
    }
    for (mu, c) in code.checks().iter().enumerate() {
        let j = if syndrome[mu] == 1 { -1.0 } else { 1.0 };
        let arity = c.len();
        let mut table = vec![0.0; 1 << arity];
        for (idx, entry) in table.iter_mut().enumerate() {
            // Row-major over the check's spins, index 0 = -1: the product's sign is the parity of
            // the zero bits, the last spin varying fastest.
            let minus_ones = (0..arity)
                .filter(|&k| (idx >> (arity - 1 - k)) & 1 == 0)
                .count();
            let prod = if minus_ones % 2 == 1 { -1.0 } else { 1.0 };
            *entry = -gamma * j * prod;
        }
        b.factor(c.iter().map(|&i| nodes[i]).collect(), table);
    }
    b.build()
}

/// The state index convention of [`crate::het::exact_boltzmann`]: node `i` is bit `n - 1 - i`.
fn spin_of(index: usize, n: usize, i: usize) -> f64 {
    if (index >> (n - 1 - i)) & 1 == 1 {
        1.0
    } else {
        -1.0
    }
}

/// The relaxed model's exact marginals `<tau_i>` at unit inverse temperature, by enumeration.
///
/// # Errors
///
/// [`Invalid::TooManyBits`] above [`MAX_BITS_EXACT`].
pub fn relaxed_marginals(
    code: &Code,
    syndrome: &[u8],
    p: f64,
    gamma: f64,
) -> Result<Vec<f64>, Invalid> {
    let n = code.n();
    if n > MAX_BITS_EXACT {
        return Err(Invalid::TooManyBits { n });
    }
    let g = relaxed_graph(code, syndrome, p, gamma);
    let law = exact_boltzmann(&g, 1.0);
    let mut marg = vec![0.0f64; n];
    for (x, &pr) in law.iter().enumerate() {
        for (i, m) in marg.iter_mut().enumerate() {
            *m += pr * spin_of(x, n, i);
        }
    }
    Ok(marg)
}

/// The decoded noise from marginals: a flip wherever `<tau_i> < 0`.
#[must_use]
pub fn decisions(marginals: &[f64]) -> Vec<u8> {
    marginals.iter().map(|&m| u8::from(m < 0.0)).collect()
}

/// The dense kernel of one sequential heat-bath sweep over the relaxed model's spins, in the
/// index convention of [`crate::het::exact_boltzmann`], for [`crate::pdit::stationary_of`],
/// [`crate::pdit::tau_int_of`] and [`crate::pdit::mixing_time`].
///
/// # Errors
///
/// [`Invalid::TooManyBits`] above twelve bits: the kernel is `4^n` entries.
pub fn relaxed_sweep_kernel(g: &HetGraph) -> Result<Vec<f64>, Invalid> {
    let n = g.n();
    if n > 12 {
        return Err(Invalid::TooManyBits { n });
    }
    let states = 1usize << n;
    let mut kernel = vec![0.0f64; states * states];
    let mut dist = vec![0.0f64; states];
    let mut next = vec![0.0f64; states];
    let mut s = vec![0u8; n];
    for x in 0..states {
        dist.iter_mut().for_each(|v| *v = 0.0);
        dist[x] = 1.0;
        for i in 0..n {
            next.iter_mut().for_each(|v| *v = 0.0);
            let bit = 1usize << (n - 1 - i);
            for y in 0..states {
                let p = dist[y];
                if p == 0.0 {
                    continue;
                }
                for (k, v) in s.iter_mut().enumerate() {
                    *v = ((y >> (n - 1 - k)) & 1) as u8;
                }
                s[i] = 1;
                let e_up = g.energy(&s);
                s[i] = 0;
                let e_down = g.energy(&s);
                let p_up = 1.0 / (1.0 + (e_up - e_down).exp());
                next[y | bit] += p * p_up;
                next[y & !bit] += p * (1.0 - p_up);
            }
            core::mem::swap(&mut dist, &mut next);
        }
        kernel[x * states..(x + 1) * states].copy_from_slice(&dist);
    }
    Ok(kernel)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pdit::{stationary_of, tau_int_of};

    /// Column and row weights as asked, codewords with zero syndrome, and a rate no less than
    /// `1 - m/n`.
    #[test]
    fn a_regular_code_has_its_weights_and_its_codewords_have_zero_syndrome() {
        let code = Code::regular(12, 3, 4, 5).expect("a (3,4) code on 12 bits");
        assert_eq!(code.m(), 9);
        for c in 0..12 {
            assert_eq!(
                (0..9).filter(|&r| code.h.get(r, c)).count(),
                3,
                "column {c}"
            );
        }
        for check in code.checks() {
            assert_eq!(check.len(), 4);
        }
        assert!(code.rate() >= 1.0 - 9.0 / 12.0 - 1e-12);
        for seed in 0..5u64 {
            let x = code.random_codeword(seed);
            assert!(code.syndrome(&x).expect("length").iter().all(|&b| b == 0));
        }
        assert_eq!(Code::regular(10, 3, 4, 1), Err(Invalid::NoRegularCode));
    }

    /// On a Tanner graph without cycles belief propagation is the exact hard posterior; on one
    /// with cycles it is not; and the relaxed marginals converge to the hard ones as the check
    /// strength grows.
    #[test]
    fn belief_propagation_is_exact_on_a_tree_and_the_hard_posterior_is_the_relaxed_limit() {
        // A chain of checks sharing single bits: {0,1,2}, {2,3,4}, {4,5,6} -- a tree.
        let rows = vec![
            vec![1u8, 1, 1, 0, 0, 0, 0],
            vec![0, 0, 1, 1, 1, 0, 0],
            vec![0, 0, 0, 0, 1, 1, 1],
        ];
        let tree = Code::from_matrix(Matrix::from_rows(&rows));
        assert!(tree.tanner_is_forest());
        let p = 0.1;
        let noise = bsc(&[0u8; 7], 0.3, 4);
        let syndrome = tree.syndrome(&noise).expect("length");
        let hard = hard_marginals(&tree, &syndrome, p).expect("small");
        let bp = bp_marginals(&tree, &syndrome, p, 20);
        for i in 0..7 {
            assert!(
                (hard[i] - bp[i]).abs() < 1e-9,
                "bit {i}: hard {} vs BP {}",
                hard[i],
                bp[i]
            );
        }
        let far = relaxed_marginals(&tree, &syndrome, p, 12.0).expect("small");
        let near = relaxed_marginals(&tree, &syndrome, p, 0.5).expect("small");
        let gap_far: f64 = hard
            .iter()
            .zip(&far)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        let gap_near: f64 = hard
            .iter()
            .zip(&near)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(gap_far < 1e-3, "gamma 12: {gap_far}");
        assert!(
            gap_near > 1e-2,
            "gamma 0.5: {gap_near} -- the relaxation would be vacuous"
        );
        // With cycles: a regular code on 9 bits.
        let cyclic = Code::regular(9, 2, 3, 2).expect("a (2,3) code on 9 bits");
        assert!(!cyclic.tanner_is_forest());
        let noise = bsc(&[0u8; 9], 0.2, 8);
        let syndrome = cyclic.syndrome(&noise).expect("length");
        let hard = hard_marginals(&cyclic, &syndrome, p).expect("small");
        let bp = bp_marginals(&cyclic, &syndrome, p, 30);
        let gap: f64 = hard
            .iter()
            .zip(&bp)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(
            gap > 1e-6,
            "belief propagation on a cyclic graph should not be exact: {gap:e}"
        );
        assert!(hard.iter().all(|m| m.abs() <= 1.0 + 1e-12));
    }

    /// The sequential chain over the relaxed model has its Boltzmann law as stationary law, and
    /// its autocorrelation time grows with the check strength: the relaxation's price.
    #[test]
    fn the_relaxed_chain_is_boltzmann_and_stiffens_with_the_check_strength() {
        let code = Code::regular(8, 2, 4, 3).expect("a (2,4) code on 8 bits");
        let noise = bsc(&[0u8; 8], 0.15, 6);
        let syndrome = code.syndrome(&noise).expect("length");
        let mut taus = Vec::new();
        for gamma in [1.0f64, 6.0] {
            let g = relaxed_graph(&code, &syndrome, 0.1, gamma);
            let kernel = relaxed_sweep_kernel(&g).expect("small");
            let pi = stationary_of(&kernel, 256).expect("irreducible");
            let law = exact_boltzmann(&g, 1.0);
            // A stiff chain (large gamma) conditions the direct solve; a few parts in ten
            // million is the solve's accuracy, not a different law.
            let tv = crate::autocorr::total_variation(&pi, &law);
            assert!(
                tv < 1e-6,
                "gamma {gamma}: stationary law vs Boltzmann, TV {tv}"
            );
            let first: Vec<f64> = (0..256).map(|x| spin_of(x, 8, 0)).collect();
            taus.push(tau_int_of(&kernel, 256, &pi, &first).expect("variance"));
        }
        assert!(
            taus[1] > taus[0],
            "tau_int at gamma 6 ({}) should exceed gamma 1 ({})",
            taus[1],
            taus[0]
        );
    }
}
