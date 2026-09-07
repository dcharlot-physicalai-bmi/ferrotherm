//! Deterministic, seedable RNG (PCG-XSH-RR 64/32). std-only, wasm-clean, no external deps.
//! Determinism is a feature: every published number must be reproducible from its seed.

#[derive(Clone)]
/// PCG-XSH-RR, a small deterministic generator with independent streams.
///
/// Deterministic by seed is load-bearing: every result in this crate must be reproducible from its
/// seed alone, and a generator with hidden global state cannot promise that.
pub struct Pcg {
    state: u64,
    inc: u64,
}

impl Pcg {
    #[must_use]
    /// A generator on `stream`, started at `seed`. Different streams do not correlate.
    pub fn new(seed: u64, stream: u64) -> Self {
        let mut r = Pcg { state: 0, inc: (stream << 1) | 1 };
        r.next_u32();
        r.state = r.state.wrapping_add(seed);
        r.next_u32();
        r
    }

    #[inline]
    /// The next 32 bits, advancing the state once.
    pub fn next_u32(&mut self) -> u32 {
        let old = self.state;
        self.state = old.wrapping_mul(6364136223846793005).wrapping_add(self.inc);
        let xorshifted = (((old >> 18) ^ old) >> 27) as u32;
        let rot = (old >> 59) as u32;
        xorshifted.rotate_right(rot)
    }

    /// 64 uniform bits, as two draws.
    ///
    /// The generator's state step produces 32 bits, so a full word is two of them. Used where the
    /// bits are wanted AS bits rather than as a number -- [`crate::multispin`] treats one word as
    /// 64 independent coin flips, one per replica.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        (u64::from(self.next_u32()) << 32) | u64::from(self.next_u32())
    }

    /// Uniform in [0, 1).
    #[inline]
    pub fn f64(&mut self) -> f64 {
        // 53 bits of mantissa from two draws
        let hi = (self.next_u32() >> 6) as u64; // 26 bits
        let lo = (self.next_u32() >> 5) as u64; // 27 bits
        ((hi << 27) | lo) as f64 / (1u64 << 53) as f64
    }

    /// Bernoulli(p) as a spin: +1 with probability p, else -1.
    #[inline]
    pub fn spin(&mut self, p: f64) -> i8 {
        if self.f64() < p { 1 } else { -1 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A full word is the two 32-bit draws, in that order, and nothing else.
    ///
    /// `multispin` reads one word as 64 independent coin flips, so a `next_u64` that reused a draw
    /// or dropped one would correlate the replicas that share it -- and correlated replicas each
    /// remain individually correct, which is exactly the failure a per-replica check cannot see.
    #[test]
    fn a_full_word_is_two_draws_and_the_bits_are_balanced() {
        let (mut a, mut b) = (Pcg::new(9, 3), Pcg::new(9, 3));
        let want = (u64::from(b.next_u32()) << 32) | u64::from(b.next_u32());
        assert_eq!(a.next_u64(), want);

        let mut ones = [0u32; 64];
        let n = 20_000;
        for _ in 0..n {
            let w = a.next_u64();
            for (bit, c) in ones.iter_mut().enumerate() {
                *c += ((w >> bit) & 1) as u32;
            }
        }
        // Every bit position must be a fair coin. Four standard deviations of a Binomial(n, 1/2) is
        // 2*sqrt(n), so this fails on a stuck or duplicated half-word rather than on bad luck.
        let tol = 2.0 * f64::from(n).sqrt();
        for (bit, &c) in ones.iter().enumerate() {
            let dev = (f64::from(c) - f64::from(n) / 2.0).abs();
            assert!(dev < tol, "bit {bit} was set {c} times of {n}, off by {dev:.0} (tol {tol:.0})");
        }
    }

    #[test]
    fn deterministic_and_uniform() {
        let mut a = Pcg::new(42, 7);
        let mut b = Pcg::new(42, 7);
        for _ in 0..1000 {
            assert_eq!(a.next_u32(), b.next_u32());
        }
        let mut r = Pcg::new(1, 1);
        let n = 100_000;
        let mean: f64 = (0..n).map(|_| r.f64()).sum::<f64>() / n as f64;
        assert!((mean - 0.5).abs() < 0.005, "mean {mean}");
    }
}
