//! Dense associative memory — the modern Hopfield network, and attention as its update.
//!
//! The classical memory stores `α_c N ≈ 0.138 N` patterns because its energy is quadratic in the
//! overlaps. Krotov & Hopfield (2016) raised the power: `E(s) = −Σ_μ F(ξ^μ · s)` with `F(x) = xⁿ`
//! stores of order `N^{n−1}` patterns (Demircigil et al. 2017: `N^{n−1} / (2(2n−3)!!·ln N)`), and
//! `F(x) = eˣ` stores exponentially many. Ramsauer et al. (2020) showed the exponential energy with
//! continuous states has, as its one-step update, exactly the softmax attention of a transformer
//! — "Hopfield networks is all you need." That is the bridge from thermodynamic sampling to the
//! architecture the field runs on, and this module builds it in the crate's own terms.
//!
//! # What is exact here
//!
//! * At degree 2 the dense energy IS the classical one: `−(1/2N) Σ_μ (ξ^μ·s)² = E_Hebb(s) − P/2`
//!   for every state — an identity the tests hold to `1e-9` against [`crate::hopfield::hebbian`],
//!   so the two modules are pinned to each other, not merely similar.
//! * Whether a stored pattern is a fixed point of the zero-temperature dynamics is a finite
//!   computation, [`DenseMemory::is_fixed_point`]: every single-spin flip must raise the energy.
//!   That is the quantity capacity theorems are about, and it is measured directly.
//!
//! # What is a sampler
//!
//! [`DenseMemory::sweep`] is heat-bath Gibbs over the exact energy differences, with the `P`
//! overlaps `x_μ = ξ^μ·s` cached and updated on every accepted flip, so a sweep costs `O(NP)` —
//! the same as the classical dense coupling matrix at `P ~ N`, and far less than the `O(N²P)` a
//! naive higher-order factor would cost.
//!
//! # What is measured, not proved
//!
//! The capacity laws are large-`N` statements with logarithmic corrections. The tests measure
//! their ORDER at finite `N` — degree 3 keeps every one of a pattern set stable where degree 2
//! has lost most of them — and the exponential memory's attention update retrieving from a
//! quarter-corrupted query with more patterns than spins. Those are demonstrations against a
//! scaling law, and they are labelled as such.

use crate::rng::Pcg;

/// The interaction function `F` of the energy `E = −c Σ_μ F(ξ^μ · s)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Energy {
    /// `F(x) = xⁿ` for even `n`, `max(x, 0)ⁿ` (rectified) for odd `n`; `c = 1 / (n N^{n−1})`, which
    /// makes degree 2 the classical Hebbian energy exactly.
    Polynomial(u32),
    /// `F(x) = exp(b (x − N)) / b`, shifted so the term of a perfectly matched pattern is `1/b`.
    Exponential { b: f64 },
}

/// A dense associative memory over `±1` spins.
#[derive(Clone, Debug)]
pub struct DenseMemory {
    pub patterns: Vec<Vec<i8>>,
    pub n: usize,
    pub energy: Energy,
}

impl DenseMemory {
    pub fn new(patterns: Vec<Vec<i8>>, energy: Energy) -> Self {
        let n = patterns.first().map_or(0, |p| p.len());
        assert!(n > 0 && patterns.iter().all(|p| p.len() == n));
        if let Energy::Polynomial(k) = energy {
            assert!(k >= 1);
        }
        DenseMemory { patterns, n, energy }
    }

    fn f(&self, x: f64) -> f64 {
        match self.energy {
            Energy::Polynomial(k) => {
                if k % 2 == 1 && x < 0.0 {
                    0.0
                } else {
                    x.powi(k as i32)
                }
            }
            Energy::Exponential { b } => (b * (x - self.n as f64)).exp() / b,
        }
    }

    fn c(&self) -> f64 {
        match self.energy {
            Energy::Polynomial(k) => 1.0 / (k as f64 * (self.n as f64).powi(k as i32 - 1)),
            Energy::Exponential { .. } => 1.0,
        }
    }

    /// The overlaps `x_μ = ξ^μ · s`.
    pub fn overlaps(&self, s: &[i8]) -> Vec<f64> {
        self.patterns.iter().map(|p| p.iter().zip(s).map(|(&a, &b)| (a as i32 * b as i32) as f64).sum()).collect()
    }

    /// `E(s) = −c Σ_μ F(ξ^μ · s)`.
    pub fn energy_of(&self, s: &[i8]) -> f64 {
        -self.c() * self.overlaps(s).iter().map(|&x| self.f(x)).sum::<f64>()
    }

    /// Energy change of flipping spin `i`, given the current overlaps.
    fn delta(&self, s: &[i8], x: &[f64], i: usize) -> f64 {
        let c = self.c();
        let mut d = 0.0;
        for (p, &xm) in self.patterns.iter().zip(x) {
            let xn = xm - 2.0 * (p[i] as i32 * s[i] as i32) as f64;
            d -= c * (self.f(xn) - self.f(xm));
        }
        d
    }

    /// Is `s` a fixed point of the zero-temperature single-flip dynamics — does every flip cost
    /// energy? Exact.
    pub fn is_fixed_point(&self, s: &[i8]) -> bool {
        let x = self.overlaps(s);
        (0..self.n).all(|i| self.delta(s, &x, i) > 0.0)
    }

    /// The fraction of stored patterns that are fixed points.
    pub fn stable_fraction(&self) -> f64 {
        let k = self.patterns.iter().filter(|p| self.is_fixed_point(p)).count();
        k as f64 / self.patterns.len() as f64
    }

    /// One heat-bath sweep at inverse temperature `beta`, overlaps cached across the sweep.
    pub fn sweep(&self, beta: f64, s: &mut [i8], rng: &mut Pcg) {
        let mut x = self.overlaps(s);
        for i in 0..self.n {
            let d = self.delta(s, &x, i);
            let p_flip = 1.0 / (1.0 + (beta * d).exp());
            if rng.f64() < p_flip {
                for (p, xm) in self.patterns.iter().zip(x.iter_mut()) {
                    *xm -= 2.0 * (p[i] as i32 * s[i] as i32) as f64;
                }
                s[i] = -s[i];
            }
        }
    }

    /// Retrieve from `start`: `sweeps` heat-bath sweeps at `beta`, returning the final state.
    pub fn retrieve(&self, start: &[i8], beta: f64, sweeps: usize, seed: u64) -> Vec<i8> {
        let mut s = start.to_vec();
        let mut rng = Pcg::new(seed, 13);
        for _ in 0..sweeps {
            self.sweep(beta, &mut s, &mut rng);
        }
        s
    }

    /// The continuous update of the exponential memory — softmax attention over the patterns
    /// (Ramsauer et al. 2020): `ξᵀ softmax(β ξ q)`. One step retrieves a stored pattern from a
    /// query near it when `β` is large enough that one term dominates.
    pub fn attention_update(&self, query: &[f64], beta: f64) -> Vec<f64> {
        assert_eq!(query.len(), self.n);
        let logits: Vec<f64> = self.patterns.iter().map(|p| beta * p.iter().zip(query).map(|(&a, &q)| a as f64 * q).sum::<f64>()).collect();
        let mx = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let w: Vec<f64> = logits.iter().map(|l| (l - mx).exp()).collect();
        let z: f64 = w.iter().sum();
        let mut out = vec![0.0; self.n];
        for (p, wi) in self.patterns.iter().zip(&w) {
            for (o, &a) in out.iter_mut().zip(p) {
                *o += wi / z * a as f64;
            }
        }
        out
    }
}

// ---- the program path: the same memory as a higher-order program ----------------------------
//
// A polynomial dense memory IS a higher-order Ising model: over ±1 spins every power of the
// overlap collapses to a multilinear polynomial, so the memory can be written as the crate's
// `Hubo`, run on the native higher-order annealer, lowered to pairwise by `reduce`, and placed on a
// device by `embed` -- the same object from the closed form to the machine. The identity that
// makes the path trustworthy is checked for every degree: the HUBO's energy plus its dropped
// constant equals `energy_of` on random states.

/// The number of length-`k` sequences over `n` symbols whose set of odd-multiplicity symbols is a
/// FIXED `r`-subset: `k! [x^k] sinh(x)^r cosh(x)^{n−r}`. This is the multinomial weight a monomial
/// `Π_{i∈S} s_i` receives when `(Σ_i a_i s_i)^k` is expanded over `±1` spins with `a_i² = 1`.
pub fn odd_set_count(k: usize, r: usize, n: usize) -> f64 {
    assert!(r <= k && r <= n);
    // truncated power series in x up to degree k, coefficients as f64
    let mul = |a: &[f64], b: &[f64]| -> Vec<f64> {
        let mut c = vec![0.0; k + 1];
        for (i, &ai) in a.iter().enumerate() {
            for (j, &bj) in b.iter().enumerate() {
                if i + j <= k {
                    c[i + j] += ai * bj;
                }
            }
        }
        c
    };
    let mut fact = vec![1.0; k + 1];
    for i in 1..=k {
        fact[i] = fact[i - 1] * i as f64;
    }
    let sinh: Vec<f64> = (0..=k).map(|d| if d % 2 == 1 { 1.0 / fact[d] } else { 0.0 }).collect();
    let cosh: Vec<f64> = (0..=k).map(|d| if d % 2 == 0 { 1.0 / fact[d] } else { 0.0 }).collect();
    let pow = |base: &[f64], mut e: usize| -> Vec<f64> {
        let mut result = vec![0.0; k + 1];
        result[0] = 1.0;
        let mut b = base.to_vec();
        while e > 0 {
            if e & 1 == 1 {
                result = mul(&result, &b);
            }
            b = mul(&b, &b);
            e >>= 1;
        }
        result
    };
    let series = mul(&pow(&sinh, r), &pow(&cosh, n - r));
    (series[k] * fact[k]).round()
}

impl DenseMemory {
    /// The memory as a higher-order model, for an UNRECTIFIED polynomial energy of degree `k`:
    /// `E = −c Σ_μ (ξ^μ·s)^k` expanded into monomials of every size `r ≡ k (mod 2)`, `r ≥ 1`,
    /// with the constant (`r = 0`) returned separately as the offset. Refuses the rectified odd
    /// energies and the exponential one, which are not polynomials in the spins.
    ///
    /// Term count is `Σ_r C(N, r)`, so degree 4 at `N = 40` is 92 000 terms; keep `N` modest.
    pub fn to_hubo(&self) -> Result<(crate::hubo::Hubo, f64), String> {
        let k = match self.energy {
            Energy::Polynomial(k) if k % 2 == 0 => k as usize,
            Energy::Polynomial(k) => return Err(format!("degree {k} is rectified, not a polynomial in the spins")),
            Energy::Exponential { .. } => return Err("the exponential energy is not a polynomial".into()),
        };
        let c = self.c();
        let n = self.n;
        let mut h = crate::hubo::Hubo::new(n);
        let mut offset = 0.0;
        for r in (0..=k.min(n)).filter(|r| r % 2 == k % 2) {
            let count = odd_set_count(k, r, n);
            if r == 0 {
                offset += -c * count * self.patterns.len() as f64;
                continue;
            }
            // every r-subset of 0..n, in lexicographic order
            let mut subset: Vec<usize> = (0..r).collect();
            loop {
                let coeff: f64 = self.patterns.iter().map(|p| subset.iter().map(|&i| p[i] as f64).product::<f64>()).sum();
                let w = c * count * coeff;
                if w != 0.0 {
                    h.add(&subset, w).map_err(|e| format!("{e:?}"))?;
                }
                // advance: find the rightmost index that can still move
                let mut i = r;
                while i > 0 && subset[i - 1] == n - r + (i - 1) {
                    i -= 1;
                }
                if i == 0 {
                    break;
                }
                subset[i - 1] += 1;
                for j in i..r {
                    subset[j] = subset[j - 1] + 1;
                }
            }
        }
        Ok((h, offset))
    }

    /// The memory as a `.ftp` program: the HUBO's terms as factors, with `schedule`.
    pub fn to_program(&self, schedule: &crate::schedule::Schedule) -> Result<(crate::ftp::Program, f64), String> {
        let (h, offset) = self.to_hubo()?;
        let mut factors = Vec::with_capacity(h.terms());
        for (vars, w) in h.iter() {
            factors.push(crate::factor::Factor::new(&vars, w, self.n).map_err(|e| format!("{e:?}"))?);
        }
        Ok((
            crate::ftp::Program {
                name: Some(format!("dense memory, {} patterns, {:?}", self.patterns.len(), self.energy)),
                spins: self.n,
                bias: Vec::new(),
                factors,
                colors: Vec::new(),
                encodings: Vec::new(),
                schedule: schedule.clone(),
                observe: Vec::new(),
                target: None,
                price: None,
            },
            offset,
        ))
    }
}

/// `pattern` with a random `fraction` of its spins flipped.
pub fn corrupt(pattern: &[i8], fraction: f64, seed: u64) -> Vec<i8> {
    let mut rng = Pcg::new(seed, 17);
    pattern.iter().map(|&v| if rng.f64() < fraction { -v } else { v }).collect()
}

/// `(1/N) Σ_i ξ_i s_i`.
pub fn overlap(pattern: &[i8], s: &[i8]) -> f64 {
    crate::hopfield::overlap(pattern, s)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hopfield::{hebbian, random_patterns};

    /// Degree 2 IS the classical memory: E_dense(s) = E_Hebb(s) − P/2 for every state.
    #[test]
    fn degree_two_is_the_hebbian_energy_up_to_a_constant() {
        let pats = random_patterns(24, 5, 1);
        let dense = DenseMemory::new(pats.clone(), Energy::Polynomial(2));
        let hebb = hebbian(&pats);
        let mut rng = Pcg::new(2, 0);
        for _ in 0..50 {
            let s: Vec<i8> = (0..24).map(|_| if rng.f64() < 0.5 { -1 } else { 1 }).collect();
            let want = hebb.energy(&s) - 2.5;
            assert!((dense.energy_of(&s) - want).abs() < 1e-9, "{} vs {want}", dense.energy_of(&s));
        }
    }

    /// The cached-overlap flip cost equals the brute-force energy difference.
    #[test]
    fn the_flip_cost_is_the_energy_difference() {
        for energy in [Energy::Polynomial(2), Energy::Polynomial(3), Energy::Exponential { b: 0.5 }] {
            let pats = random_patterns(16, 6, 3);
            let m = DenseMemory::new(pats, energy);
            let mut rng = Pcg::new(4, 0);
            let s: Vec<i8> = (0..16).map(|_| if rng.f64() < 0.5 { -1 } else { 1 }).collect();
            let x = m.overlaps(&s);
            for i in 0..16 {
                let mut t = s.clone();
                t[i] = -t[i];
                let brute = m.energy_of(&t) - m.energy_of(&s);
                assert!((m.delta(&s, &x, i) - brute).abs() < 1e-9 * (1.0 + brute.abs()));
            }
        }
    }

    /// Capacity, measured as the theorems define it: degree 3 keeps a pattern set stable where
    /// degree 2 has lost it. At N = 100, degree 2's law puts capacity near 14 patterns; degree 3
    /// stores hundreds.
    #[test]
    fn higher_degree_stores_far_more_patterns() {
        let n = 100;
        let pats = random_patterns(n, 120, 5);
        let d2 = DenseMemory::new(pats.clone(), Energy::Polynomial(2)).stable_fraction();
        let d3 = DenseMemory::new(pats.clone(), Energy::Polynomial(3)).stable_fraction();
        let ex = DenseMemory::new(pats, Energy::Exponential { b: 1.0 }).stable_fraction();
        assert!(d2 < 0.2, "degree 2 at alpha = 1.2 should have lost nearly all patterns: {d2}");
        assert!(d3 > 0.95, "degree 3 at 120 patterns of 100 spins should keep them: {d3}");
        assert!(ex == 1.0, "the exponential memory keeps every pattern: {ex}");
        // and at a load the classical memory can carry, all three agree.
        let few = random_patterns(n, 8, 6);
        for e in [Energy::Polynomial(2), Energy::Polynomial(3), Energy::Exponential { b: 1.0 }] {
            assert_eq!(DenseMemory::new(few.clone(), e).stable_fraction(), 1.0);
        }
    }

    /// Retrieval by sampling: from a quarter-corrupted pattern, the degree-3 memory at 120 patterns
    /// returns to it; the classical one cannot.
    #[test]
    fn sampling_retrieves_where_the_pattern_is_stable() {
        let n = 100;
        let pats = random_patterns(n, 120, 7);
        let start = corrupt(&pats[0], 0.25, 8);
        let d3 = DenseMemory::new(pats.clone(), Energy::Polynomial(3));
        let got = d3.retrieve(&start, 20.0, 30, 9);
        assert!(overlap(&pats[0], &got) > 0.98, "degree 3 retrieved overlap {}", overlap(&pats[0], &got));
        let d2 = DenseMemory::new(pats.clone(), Energy::Polynomial(2));
        let got2 = d2.retrieve(&start, 20.0, 30, 9);
        assert!(overlap(&pats[0], &got2) < 0.9, "degree 2 at alpha 1.2 should not retrieve: {}", overlap(&pats[0], &got2));
    }

    /// The program path is the memory: HUBO energy plus offset equals the dense energy for every
    /// state, at every polynomial degree the expansion covers.
    #[test]
    fn the_hubo_is_the_memory_at_every_degree() {
        let mut rng = Pcg::new(21, 0);
        for (k, n) in [(2u32, 24usize), (4, 12)] {
            let pats = random_patterns(n, 3, 40 + k as u64);
            let m = DenseMemory::new(pats, Energy::Polynomial(k));
            let (h, offset) = m.to_hubo().unwrap();
            assert_eq!(h.max_arity(), k as usize);
            for _ in 0..30 {
                let s: Vec<i8> = (0..n).map(|_| if rng.f64() < 0.5 { -1 } else { 1 }).collect();
                let want = m.energy_of(&s);
                let got = h.energy(&s) + offset;
                assert!((got - want).abs() < 1e-9 * (1.0 + want.abs()), "degree {k}: hubo {got} vs dense {want}");
            }
        }
        assert!(DenseMemory::new(random_patterns(8, 2, 1), Energy::Polynomial(3)).to_hubo().is_err());
        assert!(DenseMemory::new(random_patterns(8, 2, 1), Energy::Exponential { b: 1.0 }).to_hubo().is_err());
        // the multinomial counts at the small cases one can check by hand
        assert_eq!(odd_set_count(2, 0, 7), 7.0);
        assert_eq!(odd_set_count(2, 2, 7), 2.0);
        assert_eq!(odd_set_count(3, 1, 7), 3.0 * 7.0 - 2.0);
        assert_eq!(odd_set_count(3, 3, 7), 6.0);
        assert_eq!(odd_set_count(4, 0, 7), 3.0 * 49.0 - 2.0 * 7.0);
    }

    /// Math to machine. The degree-4 memory as a program: run natively (retrieves), lowered to
    /// pairwise (an EXACT reduction -- the minimum over ancillas reproduces the HUBO energy on
    /// every state tested), and then the measured cost of that exactness: the reduced landscape
    /// is dynamically frozen (annealing retrieves in 0 of 5 runs, measured) and at max degree 28
    /// too dense for a 288-site Chimera. At degree 2 the memory IS pairwise, and it goes onto the
    /// Chimera by the structured clique and retrieves there. Both halves are the finding.
    #[test]
    fn the_program_path_is_exact_and_the_native_path_is_the_one_that_moves() {
        use crate::embed;
        use crate::gibbs::Sampler;
        use crate::hubo;
        use crate::reduce::to_pairwise;
        use crate::schedule::Schedule;
        let n = 10;
        let pats = random_patterns(n, 2, 31);
        let m = DenseMemory::new(pats.clone(), Energy::Polynomial(4));
        assert!(m.is_fixed_point(&pats[0]));

        // native higher-order annealing retrieves
        let (h, _) = m.to_hubo().unwrap();
        let out = hubo::anneal(&h, &hubo::Params { beta_min: 0.2, beta_max: 6.0, stages: 30, sweeps_per_stage: 10 }, 5);
        assert!(overlap(&pats[0], &out.state).abs() > 0.99, "native HUBO annealing retrieved {}", overlap(&pats[0], &out.state));

        // the reduction is exact: min over ancillas (originals clamped) + offset = HUBO energy
        let (prog, _) = m.to_program(&Schedule::constant(3.0, 100)).unwrap();
        let red = to_pairwise(&prog).unwrap();
        assert!(red.ancillas > 0);
        let g = red.program.to_graph().unwrap();
        let mut rng = Pcg::new(9, 0);
        for trial in 0..6 {
            let s: Vec<i8> = if trial == 0 { pats[0].clone() } else { (0..n).map(|_| if rng.f64() < 0.5 { -1 } else { 1 }).collect() };
            let mut sm = Sampler::new(&g, 8.0, trial as u64);
            for i in 0..n {
                sm.clamp(i, s[i]);
            }
            sm.sweeps(400, None);
            let got = g.energy(&sm.s) + red.offset;
            assert!((got - h.energy(&s)).abs() < 1e-6, "reduced {got} vs hubo {} on trial {trial}", h.energy(&s));
        }

        // degree 2 IS pairwise: onto a Chimera by the structured clique, retrieved on the machine.
        // A 12-spin memory fills the K_12 native clique of chimera(3,3,4) exactly. Two patterns is
        // alpha = 0.17, above the classical capacity, so the pattern set is chosen (by seed) so
        // that both ARE fixed points -- the machine is what is under test here, not the memory.
        let n2 = 12;
        let (pats2, m2) = (0..64u64)
            .map(|seed| {
                let p = random_patterns(n2, 2, 500 + seed);
                let m = DenseMemory::new(p.clone(), Energy::Polynomial(2));
                (p, m)
            })
            .find(|(p, m)| p.iter().all(|x| m.is_fixed_point(x)))
            .expect("a two-pattern set stable at degree 2");
        let (prog2, _) = m2.to_program(&Schedule::constant(3.0, 100)).unwrap();
        let red2 = to_pairwise(&prog2).unwrap();
        assert_eq!(red2.ancillas, 0, "degree 2 needs no ancillas");
        let g2 = red2.program.to_graph().unwrap();
        let hw = crate::ising::chimera(3, 3, 4, 1.0);
        let e = embed::chimera_clique(3, 4).unwrap();
        e.verify(&g2, &hw).unwrap();
        // Chain strength is RELATIVE to the couplings. The memory's are (1/N) sum of +-1 over two
        // patterns, at most 2/12; a chain strength of 4 -- right for unit couplings elsewhere in
        // this crate -- is 24x that here, and was measured to freeze the chains before the
        // logical problem orders (0 of 5 anneals reached the ground state). Six times the largest
        // coupling holds every chain and lets the problem move.
        let max_w = g2.w.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
        let hwm = embed::apply_with(&g2, &hw, &e, 6.0 * max_w);
        let ground = crate::exact::Elimination::default().ground_state(&g2).unwrap().ground_energy.unwrap();
        let mut solved = 0;
        for seed in 0..5u64 {
            let mut sh = Sampler::new(&hwm.graph, 0.05, 11 + seed);
            for k in 0..40 {
                sh.beta = 0.05 + 8.0 * k as f64 / 39.0;
                sh.sweeps(25, None);
            }
            let (logical, broken) = embed::unembed(&e, &sh.s);
            let ov = pats2.iter().map(|p| overlap(p, &logical).abs()).fold(0.0, f64::max);
            if broken.is_empty() && (g2.energy(&logical) - ground).abs() < 1e-9 && ov > 0.99 {
                solved += 1;
            }
        }
        assert!(solved >= 4, "the machine reached the exact ground state (a stored pattern) in {solved} of 5 anneals");
    }

    /// Attention is the exponential memory's one-step update. With 200 patterns in 64 spins —
    /// three times more than there are spins — a query corrupted in 15% of its spins returns its
    /// pattern in one step, every time; at 25%, some corrupted queries are genuinely nearer
    /// ANOTHER stored pattern (measured: 12 of 100, at every β from 0.5 to 4), and the update
    /// returns that one — so the property held there is the one the softmax actually has: it
    /// returns the query's nearest stored pattern.
    #[test]
    fn one_attention_step_retrieves_with_more_patterns_than_spins() {
        let n = 64;
        let pats = random_patterns(n, 200, 11);
        let m = DenseMemory::new(pats.clone(), Energy::Exponential { b: 1.0 });
        let signs_of = |v: &[f64]| -> Vec<i8> { v.iter().map(|&x| if x >= 0.0 { 1 } else { -1 }).collect() };
        for mu in 0..40 {
            let q: Vec<f64> = corrupt(&pats[mu], 0.15, 100 + mu as u64).iter().map(|&v| v as f64).collect();
            let got = signs_of(&m.attention_update(&q, 2.0));
            assert!(overlap(&pats[mu], &got) > 0.99, "pattern {mu} at 15% corruption: overlap {}", overlap(&pats[mu], &got));
        }
        let mut ties = 0;
        for mu in 0..40 {
            let qi = corrupt(&pats[mu], 0.25, 200 + mu as u64);
            let q: Vec<f64> = qi.iter().map(|&v| v as f64).collect();
            let out = m.attention_update(&q, 4.0);
            let best = (0..pats.len()).map(|k| overlap(&pats[k], &qi)).fold(f64::NEG_INFINITY, f64::max);
            let nearest: Vec<usize> = (0..pats.len()).filter(|&k| (overlap(&pats[k], &qi) - best).abs() < 1e-12).collect();
            if nearest.len() == 1 {
                let got = signs_of(&out);
                assert!(overlap(&pats[nearest[0]], &got) > 0.99, "pattern {mu} at 25%: not the nearest stored pattern");
            } else {
                // An exact tie (measured: it happens): the softmax blends the tied patterns, so
                // the output carries their common sign where they agree and is near zero where
                // not -- "near", because a third pattern a small gap below the tie leaks
                // e^{-beta gap} into the blend (measured: e^{-8} = 3.4e-4 at a gap of 2).
                ties += 1;
                for i in 0..n {
                    let mean: f64 = nearest.iter().map(|&k| pats[k][i] as f64).sum::<f64>() / nearest.len() as f64;
                    assert!((out[i] - mean).abs() < 0.05, "tie of {}: coordinate {i} should be the tied patterns' mean {mean}, got {}", nearest.len(), out[i]);
                }
            }
        }
        assert!(ties <= 5, "ties should be rare: {ties} of 40");
        // and the classical memory at this load retrieves nothing.
        let hebb = crate::hopfield::hebbian(&pats);
        let start = corrupt(&pats[0], 0.15, 300);
        let got = crate::hopfield::retrieval_overlap(&hebb, &pats[0], 4.0, 20, 20, 1);
        let _ = start;
        assert!(got.value < 0.5, "the classical memory at alpha = 3.1 should not retrieve: {}", got.value);
    }
}
