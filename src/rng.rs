//! Deterministic, seedable RNG (PCG-XSH-RR 64/32). std-only, wasm-clean, no external deps.
//! Determinism is a feature: every published number must be reproducible from its seed.

#[derive(Clone)]
pub struct Pcg {
    state: u64,
    inc: u64,
}

impl Pcg {
    pub fn new(seed: u64, stream: u64) -> Self {
        let mut r = Pcg { state: 0, inc: (stream << 1) | 1 };
        r.next_u32();
        r.state = r.state.wrapping_add(seed);
        r.next_u32();
        r
    }

    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        let old = self.state;
        self.state = old.wrapping_mul(6364136223846793005).wrapping_add(self.inc);
        let xorshifted = (((old >> 18) ^ old) >> 27) as u32;
        let rot = (old >> 59) as u32;
        xorshifted.rotate_right(rot)
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
