//! p-bit LUT truth-table generators — the compact hardware form of a probabilistic bit.
//!
//! The 1-LUT p-bit (ported with permission from Open Interface Engineering's openie-fpga, and
//! re-verified here by exhaustive enumeration): a LUT6 whose inputs I0..I4 are five neighbour
//! states and I5 is one random bit from an LFSR; the output is a stochastic threshold,
//!     out = popcount(I0..I4) + I5 >= threshold,
//! the hardware-cheap analogue of the sigmoid update for UNIFORM couplings. One p-bit per LUT6
//! is the maximum-density fabric grade (grade A); `ferrotherm::hdl`'s Q.8 weighted fabric is the
//! exact grade (grade B). Both hold the same law: software == emulator == silicon configuration.

/// LUT6 init word for the stochastic-threshold p-bit update.
pub fn bsn_threshold_init(threshold: u8) -> u64 {
    let mut init = 0u64;
    for idx in 0u64..64 {
        let neighbours = (idx & 0b1_1111).count_ones() as u8; // I0..I4
        let rand = ((idx >> 5) & 1) as u8; // I5
        if neighbours + rand >= threshold {
            init |= 1u64 << idx;
        }
    }
    init
}

/// LUT6 init word for an LFSR feedback bit: XOR of the tapped inputs (tap indices 0..6).
pub fn lfsr_feedback_init(taps: &[u8]) -> u64 {
    let mut init = 0u64;
    for idx in 0u64..64 {
        let mut x = 0u8;
        for &t in taps {
            x ^= ((idx >> t) & 1) as u8;
        }
        if x == 1 {
            init |= 1u64 << idx;
        }
    }
    init
}

/// The probability that a threshold p-bit fires given `k` high neighbours, under a fair random
/// bit: exact closed form for verifying fabric statistics against the ferrotherm model layer.
pub fn bsn_fire_prob(threshold: u8, k: u8) -> f64 {
    // out = 1 if k + r >= threshold, r ~ Bernoulli(1/2)
    if k >= threshold {
        1.0
    } else if k + 1 == threshold {
        0.5
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Exhaustive: every LUT entry must equal the defining predicate.
    #[test]
    fn threshold_table_exhaustive() {
        for threshold in 0..=6u8 {
            let init = bsn_threshold_init(threshold);
            for idx in 0u64..64 {
                let n = (idx & 0b1_1111).count_ones() as u8;
                let r = ((idx >> 5) & 1) as u8;
                let want = n + r >= threshold;
                assert_eq!(init >> idx & 1 == 1, want, "t={threshold} idx={idx}");
            }
        }
    }

    /// The XOR table for taps [0,1] is the classic 0x6666...: verify structurally.
    #[test]
    fn lfsr_table_exhaustive() {
        let init = lfsr_feedback_init(&[0, 1]);
        for idx in 0u64..64 {
            let want = ((idx & 1) ^ ((idx >> 1) & 1)) == 1;
            assert_eq!(init >> idx & 1 == 1, want);
        }
    }

    /// The closed-form fire probability matches the table marginalized over the random bit.
    #[test]
    fn fire_prob_matches_table() {
        for threshold in 0..=6u8 {
            let init = bsn_threshold_init(threshold);
            for k in 0..=5u8 {
                // pick any index with popcount k in the low 5 bits
                let base: u64 = (0..k as u64).map(|b| 1u64 << b).sum();
                let p_table = ((init >> base & 1) as f64 + (init >> (base | 32) & 1) as f64) / 2.0;
                assert_eq!(p_table, bsn_fire_prob(threshold, k), "t={threshold} k={k}");
            }
        }
    }
}
