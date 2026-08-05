//! Chromatic block-Gibbs sampling.
//!
//! One sweep updates every node exactly once, color class by color class. Within a class, node
//! updates are conditionally independent (no two adjacent) — the parallelism a TSU exploits in
//! physics and a GPU exploits in threads; here the classes are simple loops, kept in the same
//! order so CPU, WebGPU, and device runs are cross-checkable draw for draw.

use crate::graph::Graph;
use crate::ledger::Ledger;
use crate::rng::Pcg;

#[inline]
fn sigma(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

pub struct Sampler<'g> {
    pub g: &'g Graph,
    pub beta: f64,
    pub s: Vec<i8>,
    pub rng: Pcg,
    /// Nodes whose value is held fixed (conditioning / "clamping"); sweeps skip them.
    pub clamped: Vec<bool>,
}

impl<'g> Sampler<'g> {
    pub fn new(g: &'g Graph, beta: f64, seed: u64) -> Self {
        let mut rng = Pcg::new(seed, 0x5EED);
        let s = (0..g.n).map(|_| rng.spin(0.5)).collect();
        Sampler { g, beta, s, rng, clamped: vec![false; g.n] }
    }

    /// Clamp node i to value v (observation / conditioning input).
    pub fn clamp(&mut self, i: usize, v: i8) {
        debug_assert!(v == 1 || v == -1);
        self.s[i] = v;
        self.clamped[i] = true;
    }

    pub fn unclamp(&mut self, i: usize) {
        self.clamped[i] = false;
    }

    /// One full chromatic sweep (every free node updated once). If a ledger is given, it is
    /// charged one Gibbs cycle per free node — the device-side price of this sweep.
    pub fn sweep(&mut self, ledger: Option<&mut Ledger>) {
        let mut updated = 0u64;
        for class in &self.g.classes {
            for &iu in class {
                let i = iu as usize;
                if self.clamped[i] {
                    continue;
                }
                let f = self.g.field(i, &self.s);
                let p_up = sigma(2.0 * self.beta * f);
                self.s[i] = self.rng.spin(p_up);
                updated += 1;
            }
        }
        if let Some(l) = ledger {
            l.samples += updated;
        }
    }

    /// Run `n` sweeps.
    pub fn sweeps(&mut self, n: usize, mut ledger: Option<&mut Ledger>) {
        for _ in 0..n {
            self.sweep(ledger.as_deref_mut());
        }
    }

    /// Read the full state (device price: one read per node). Prefer [`Self::read_subset`]:
    /// full-state readback is the crossings-tax regime.
    pub fn read_all(&self, ledger: Option<&mut Ledger>) -> Vec<i8> {
        if let Some(l) = ledger {
            l.reads += self.g.n as u64;
        }
        self.s.clone()
    }

    /// Read only the named nodes (e.g. action bits).
    pub fn read_subset(&self, idx: &[usize], ledger: Option<&mut Ledger>) -> Vec<i8> {
        if let Some(l) = ledger {
            l.reads += idx.len() as u64;
        }
        idx.iter().map(|&i| self.s[i]).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::GraphBuilder;

    /// The sampler's stationary distribution must match the exact Boltzmann distribution on an
    /// enumerable system. 4-node cycle, mixed couplings and biases, TV < 0.02.
    #[test]
    fn matches_exact_boltzmann() {
        let mut gb = GraphBuilder::new(4);
        gb.couple(0, 1, 0.7);
        gb.couple(1, 2, -0.4);
        gb.couple(2, 3, 0.55);
        gb.couple(3, 0, 0.3);
        gb.bias(0, 0.2);
        gb.bias(2, -0.35);
        let g = gb.build();
        let beta = 0.9;

        // exact
        let mut z = 0.0;
        let mut p_exact = [0.0f64; 16];
        for m in 0..16u32 {
            let s: Vec<i8> = (0..4).map(|b| if m >> b & 1 == 1 { 1 } else { -1 }).collect();
            let w = (-beta * g.energy(&s)).exp();
            p_exact[m as usize] = w;
            z += w;
        }
        for p in p_exact.iter_mut() {
            *p /= z;
        }

        // sampled
        let mut smp = Sampler::new(&g, beta, 0xC0FFEE);
        smp.sweeps(200, None); // burn-in
        let mut counts = [0u64; 16];
        let n_samples = 200_000;
        for _ in 0..n_samples {
            smp.sweep(None);
            let mut m = 0usize;
            for b in 0..4 {
                if smp.s[b] == 1 {
                    m |= 1 << b;
                }
            }
            counts[m] += 1;
        }
        let tv: f64 = (0..16)
            .map(|m| (counts[m] as f64 / n_samples as f64 - p_exact[m]).abs())
            .sum::<f64>()
            / 2.0;
        assert!(tv < 0.02, "TV distance to exact Boltzmann = {tv}");
    }

    /// Clamped nodes must never change and must steer the conditional distribution.
    #[test]
    fn clamping_conditions() {
        let mut gb = GraphBuilder::new(2);
        gb.couple(0, 1, 1.5);
        let g = gb.build();
        let mut smp = Sampler::new(&g, 1.0, 7);
        smp.clamp(0, 1);
        let mut up = 0u64;
        let n = 20_000;
        for _ in 0..n {
            smp.sweep(None);
            assert_eq!(smp.s[0], 1);
            if smp.s[1] == 1 {
                up += 1;
            }
        }
        // exact: P(s1=+1 | s0=+1) = sigma(2*beta*J) = sigma(3.0)
        let want = 1.0 / (1.0 + (-3.0f64).exp());
        let got = up as f64 / n as f64;
        assert!((got - want).abs() < 0.01, "got {got}, want {want}");
    }
}
