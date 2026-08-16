//! How a discrete variable is spelled in spins.
//!
//! The fabric has one variable: the spin, ±1. A problem may have a variable that takes one of `k`
//! values, and someone has to choose how to write it down. That choice is a compiler decision with
//! measurable consequences — spin count, how many penalty couplings it drags in, and whether the
//! penalty strength then needs tuning — which is exactly why it is a pass here rather than a type
//! in the IR.
//!
//! Three encodings, and the trade is real:
//!
//! | Encoding | Spins | Penalty couplings | Note |
//! |---|---|---|---|
//! | [`Encoding::OneHot`] | k | k(k−1)/2, **quadratic** | the obvious one, and the expensive one |
//! | [`Encoding::Binary`] | ⌈log₂k⌉ | none *if k is a power of two* | fewest spins, densest factors |
//! | [`Encoding::DomainWall`] | k−1 | k−2, **linear**, a chain | Chancellor 2019 |
//!
//! The domain-wall advantage is usually stated as "no penalty", which is not quite right and worth
//! being precise about: it still needs terms to suppress states with more than one wall. What it
//! does is replace one-hot's all-to-all penalty with a *chain*, so the penalty cost falls from
//! quadratic to linear in k, and it saves a spin. That is the honest claim.
//!
//! Binary is the trap. It uses the fewest spins, but only excludes surplus codes for free when `k`
//! is a power of two; otherwise the leftover codes are invalid states that cannot in general be
//! excluded by pairwise couplings at all. [`Slot::decode`] returns `None` for them and
//! [`Slot::add_penalty`] says so rather than pretending.

use crate::graph::GraphBuilder;

/// How to spell a `k`-valued variable in spins.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Encoding {
    /// One spin per value; exactly one is +1. Penalty is all-to-all.
    OneHot,
    /// Binary expansion. Fewest spins, but surplus codes when `k` is not a power of two.
    Binary,
    /// A chain of `k-1` spins holding a single domain wall (Chancellor 2019).
    DomainWall,
}

impl Encoding {
    /// Spins needed for a `k`-valued variable.
    pub fn spins(&self, k: usize) -> usize {
        assert!(k >= 2, "a variable with fewer than 2 values is a constant");
        match self {
            Encoding::OneHot => k,
            Encoding::Binary => (usize::BITS - (k - 1).leading_zeros()) as usize,
            Encoding::DomainWall => k - 1,
        }
    }

    /// Penalty couplings this encoding drags in for a `k`-valued variable.
    pub fn penalty_couplings(&self, k: usize) -> usize {
        match self {
            Encoding::OneHot => k * (k - 1) / 2,
            Encoding::Binary => 0,
            Encoding::DomainWall => k.saturating_sub(2),
        }
    }

    /// Whether this encoding can exclude every invalid state with pairwise terms.
    ///
    /// Binary cannot when `k` is not a power of two: the surplus codes are an arbitrary subset of
    /// the hypercube, and no pairwise penalty carves that out in general.
    #[must_use = "false means invalid codewords remain reachable and `decode` is the only thing between them and a wrong answer"]
    pub fn is_exact(&self, k: usize) -> bool {
        match self {
            Encoding::Binary => k.is_power_of_two(),
            _ => true,
        }
    }
}

/// Where one categorical variable's spins live in a graph.
#[derive(Clone, Copy, Debug)]
pub struct Slot {
    pub base: usize,
    pub k: usize,
    pub encoding: Encoding,
}

impl Slot {
    pub fn new(base: usize, k: usize, encoding: Encoding) -> Self {
        assert!(k >= 2, "a variable with fewer than 2 values is a constant");
        Slot { base, k, encoding }
    }

    /// Spins this slot occupies.
    pub fn width(&self) -> usize {
        self.encoding.spins(self.k)
    }

    /// Range of spin indices this slot occupies.
    pub fn range(&self) -> std::ops::Range<usize> {
        self.base..self.base + self.width()
    }

    /// Write `value` into `s` in this encoding.
    pub fn encode(&self, value: usize, s: &mut [i8]) {
        assert!(value < self.k, "value {value} out of range for a {}-valued variable", self.k);
        let w = self.width();
        match self.encoding {
            Encoding::OneHot => {
                for i in 0..w {
                    s[self.base + i] = if i == value { 1 } else { -1 };
                }
            }
            Encoding::Binary => {
                for i in 0..w {
                    s[self.base + i] = if (value >> i) & 1 == 1 { 1 } else { -1 };
                }
            }
            Encoding::DomainWall => {
                // value v: the first v spins are +1, the rest -1. v = 0 is all -1, v = k-1 all +1.
                for i in 0..w {
                    s[self.base + i] = if i < value { 1 } else { -1 };
                }
            }
        }
    }

    /// Read this slot's value, or `None` if the spins do not form a valid codeword.
    ///
    /// Returning `None` rather than a nearest-valid guess is deliberate: a sampler that has landed
    /// on an invalid state is telling you the penalty was too weak, and silently rounding that away
    /// is how a constraint violation becomes a wrong answer nobody notices.
    pub fn decode(&self, s: &[i8]) -> Option<usize> {
        let w = self.width();
        let bits = &s[self.base..self.base + w];
        match self.encoding {
            Encoding::OneHot => {
                let mut found = None;
                for (i, &b) in bits.iter().enumerate() {
                    if b > 0 {
                        if found.is_some() {
                            return None; // more than one hot
                        }
                        found = Some(i);
                    }
                }
                found
            }
            Encoding::Binary => {
                let mut v = 0usize;
                for (i, &b) in bits.iter().enumerate() {
                    if b > 0 {
                        v |= 1 << i;
                    }
                }
                if v < self.k {
                    Some(v)
                } else {
                    None // a surplus code
                }
            }
            Encoding::DomainWall => {
                // valid iff non-increasing: a block of +1 then a block of -1, one wall
                let v = bits.iter().take_while(|&&b| b > 0).count();
                if bits[v..].iter().all(|&b| b < 0) {
                    Some(v)
                } else {
                    None // more than one wall
                }
            }
        }
    }

    /// Add this encoding's penalty terms at strength `p`.
    ///
    /// Returns whether the penalty is exact. A `false` means invalid states remain reachable and
    /// [`Slot::decode`] is the only thing standing between them and a wrong answer.
    #[must_use = "this says whether the encoding can be made EXACT. Discarding it is how a k=6 binary variable shipped with invalid codewords costing exactly what valid ones cost, for three releases"]
    pub fn add_penalty(&self, b: &mut GraphBuilder, p: f64) -> bool {
        let w = self.width();
        match self.encoding {
            Encoding::OneHot => {
                // P * (sum_i x_i - 1)^2 with x_i = (1 + s_i)/2, dropped to pairwise:
                //   J_ij = -P/2 for every pair, h_i = -P(k-2)/2
                for i in 0..w {
                    for j in (i + 1)..w {
                        b.couple(self.base + i, self.base + j, -p / 2.0);
                    }
                    b.bias(self.base + i, -p * (self.k as f64 - 2.0) / 2.0);
                }
                true
            }
            Encoding::Binary => self.k.is_power_of_two(),
            Encoding::DomainWall => {
                // A ferromagnetic chain: every wall costs 2p, and the fixed boundaries force
                // exactly one, so the k valid codewords are the degenerate ground states.
                for i in 0..w.saturating_sub(1) {
                    b.couple(self.base + i, self.base + i + 1, p);
                }
                // boundaries s_0 = +1 and s_k = -1, folded into biases on the end spins
                b.bias(self.base, p);
                b.bias(self.base + w - 1, -p);
                true
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::GraphBuilder;

    const ALL: [Encoding; 3] = [Encoding::OneHot, Encoding::Binary, Encoding::DomainWall];

    #[test]
    fn encode_decode_round_trips() {
        for enc in ALL {
            for k in 2..=9 {
                let slot = Slot::new(0, k, enc);
                let mut s = vec![0i8; slot.width()];
                for v in 0..k {
                    slot.encode(v, &mut s);
                    assert_eq!(slot.decode(&s), Some(v), "{enc:?} k={k} v={v}");
                }
            }
        }
    }

    #[test]
    fn domain_wall_uses_fewer_spins_than_one_hot() {
        // The acceptance criterion for this pass.
        for k in 2..=32 {
            assert!(
                Encoding::DomainWall.spins(k) < Encoding::OneHot.spins(k),
                "k={k}"
            );
        }
    }

    #[test]
    fn domain_wall_penalty_is_linear_where_one_hot_is_quadratic() {
        // The claim that actually matters, and the one usually mis-stated as "no penalty".
        for k in [4, 8, 16, 64] {
            let dw = Encoding::DomainWall.penalty_couplings(k);
            let oh = Encoding::OneHot.penalty_couplings(k);
            assert_eq!(dw, k - 2);
            assert_eq!(oh, k * (k - 1) / 2);
            assert!(dw < oh, "k={k}: dw {dw} vs oh {oh}");
        }
        // and it really is a chain, not merely smaller
        assert_eq!(Encoding::DomainWall.penalty_couplings(1000), 998);
    }

    /// Enumerate every spin configuration and check the penalty's ground states are exactly the
    /// valid codewords. This is the test that proves the construction rather than the algebra.
    fn ground_states(enc: Encoding, k: usize, p: f64) -> Vec<usize> {
        let slot = Slot::new(0, k, enc);
        let w = slot.width();
        let mut b = GraphBuilder::new(w);
        slot.add_penalty(&mut b, p);
        let g = b.build();

        let mut best = f64::INFINITY;
        let mut at_best = Vec::new();
        for mask in 0..(1usize << w) {
            let s: Vec<i8> = (0..w).map(|i| if mask >> i & 1 == 1 { 1 } else { -1 }).collect();
            let e = g.energy(&s);
            if e < best - 1e-9 {
                best = e;
                at_best.clear();
            }
            if e < best + 1e-9 {
                at_best.push(mask);
            }
        }
        at_best
    }

    #[test]
    fn domain_wall_ground_states_are_exactly_the_codewords() {
        for k in 2..=8 {
            let slot = Slot::new(0, k, Encoding::DomainWall);
            let g = ground_states(Encoding::DomainWall, k, 2.0);
            assert_eq!(g.len(), k, "k={k}: expected {k} degenerate ground states, got {}", g.len());
            // and each one decodes to a distinct value
            let w = slot.width();
            let mut seen: Vec<usize> = g
                .iter()
                .map(|&mask| {
                    let s: Vec<i8> =
                        (0..w).map(|i| if mask >> i & 1 == 1 { 1 } else { -1 }).collect();
                    slot.decode(&s).expect("a ground state must be a valid codeword")
                })
                .collect();
            seen.sort_unstable();
            assert_eq!(seen, (0..k).collect::<Vec<_>>(), "k={k}");
        }
    }

    #[test]
    fn one_hot_ground_states_are_exactly_the_codewords() {
        for k in 2..=7 {
            let g = ground_states(Encoding::OneHot, k, 2.0);
            assert_eq!(g.len(), k, "k={k}");
        }
    }

    #[test]
    fn binary_is_honest_about_surplus_codes() {
        // k = 6 needs 3 spins, so codes 6 and 7 are invalid and no pairwise penalty removes them.
        let slot = Slot::new(0, 6, Encoding::Binary);
        assert_eq!(slot.width(), 3);
        assert!(!Encoding::Binary.is_exact(6));
        let mut b = GraphBuilder::new(3);
        assert!(!slot.add_penalty(&mut b, 1.0), "must report that it cannot be exact");

        let mut s = vec![-1i8; 3];
        s[1] = 1;
        s[2] = 1; // code 6
        assert_eq!(slot.decode(&s), None, "a surplus code must decode to None, not a guess");

        // powers of two have no surplus and are exact
        assert!(Encoding::Binary.is_exact(8));
        let mut b = GraphBuilder::new(3);
        assert!(Slot::new(0, 8, Encoding::Binary).add_penalty(&mut b, 1.0));
    }

    #[test]
    fn invalid_states_decode_to_none_rather_than_a_guess() {
        // two hot
        let oh = Slot::new(0, 4, Encoding::OneHot);
        assert_eq!(oh.decode(&[1, 1, -1, -1]), None);
        assert_eq!(oh.decode(&[-1, -1, -1, -1]), None, "none hot is also invalid");
        // two walls
        let dw = Slot::new(0, 5, Encoding::DomainWall);
        assert_eq!(dw.decode(&[1, -1, 1, -1]), None);
        assert_eq!(dw.decode(&[1, 1, -1, -1]), Some(2), "one wall is fine");
    }

    #[test]
    fn slots_can_be_packed_side_by_side() {
        // A model has many variables; their spins must not collide.
        let a = Slot::new(0, 4, Encoding::DomainWall); // 3 spins: 0..3
        let b = Slot::new(a.range().end, 5, Encoding::OneHot); // 5 spins: 3..8
        assert_eq!(a.range(), 0..3);
        assert_eq!(b.range(), 3..8);

        let mut s = vec![0i8; 8];
        a.encode(2, &mut s);
        b.encode(3, &mut s);
        assert_eq!(a.decode(&s), Some(2));
        assert_eq!(b.decode(&s), Some(3));
    }
}
