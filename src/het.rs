//! Heterogeneous factor-graph Gibbs — the general engine the spin sampler is a special case of.
//!
//! Nodes carry different state kinds (spin, categorical with K states); factors are weighted
//! energy tables over the product space of any number of nodes (arbitrary arity subsumes pairwise
//! couplings and biases). Sampling is block Gibbs over a proper coloring of the factor-sharing
//! graph: nodes in one color class share no factor, so their conditionals are independent and the
//! class updates as one parallel block — the same chromatic structure as [`crate::gibbs`], which
//! remains the fast path for the pure-spin pairwise case.
//!
//! Energy convention: E(s) = sum over factors f of table_f[index(states of f's nodes)], with
//! row-major indexing in node order (spin contributes dimension 2: index 0 = -1, index 1 = +1).
//! The conditional of node i enumerates its K_i states against the current neighbours:
//! p(k) proportional to exp(-beta * sum of touching-factor entries).

use crate::rng::Pcg;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    Spin,
    /// Categorical with the given number of states (2..=255).
    Cat(u8),
}

impl Kind {
    pub fn states(&self) -> usize {
        match self {
            Kind::Spin => 2,
            Kind::Cat(k) => *k as usize,
        }
    }
}

pub struct Factor {
    pub nodes: Vec<u32>,
    /// Energy contributions, length = product of the nodes' state counts, row-major in node order.
    pub table: Vec<f64>,
}

pub struct HetGraph {
    pub kinds: Vec<Kind>,
    pub factors: Vec<Factor>,
    /// node -> factor indices that touch it
    touching: Vec<Vec<u32>>,
    pub colors: Vec<u16>,
    pub classes: Vec<Vec<u32>>,
}

pub struct HetBuilder {
    kinds: Vec<Kind>,
    factors: Vec<Factor>,
}

impl HetBuilder {
    pub fn new() -> Self {
        HetBuilder { kinds: Vec::new(), factors: Vec::new() }
    }
    pub fn node(&mut self, kind: Kind) -> u32 {
        self.kinds.push(kind);
        (self.kinds.len() - 1) as u32
    }
    /// Add a factor over `nodes` with the given energy table (row-major over their state spaces).
    pub fn factor(&mut self, nodes: Vec<u32>, table: Vec<f64>) {
        let want: usize = nodes.iter().map(|&n| self.kinds[n as usize].states()).product();
        assert_eq!(table.len(), want, "table length {} != product of state spaces {}", table.len(), want);
        self.factors.push(Factor { nodes, table });
    }
    /// Convenience: pairwise spin coupling J (energy -J s_i s_j).
    pub fn couple_spins(&mut self, i: u32, j: u32, jij: f64) {
        self.factor(vec![i, j], vec![-jij, jij, jij, -jij]);
    }
    /// Convenience: spin bias h (energy -h s).
    pub fn bias_spin(&mut self, i: u32, h: f64) {
        self.factor(vec![i], vec![h, -h]);
    }
    pub fn build(self) -> HetGraph {
        let n = self.kinds.len();
        let mut touching: Vec<Vec<u32>> = vec![Vec::new(); n];
        for (fi, f) in self.factors.iter().enumerate() {
            for &node in &f.nodes {
                touching[node as usize].push(fi as u32);
            }
        }
        // adjacency = sharing a factor; greedy proper coloring
        let mut colors = vec![u16::MAX; n];
        for i in 0..n {
            let mut used = vec![false; 8];
            for &fi in &touching[i] {
                for &nb in &self.factors[fi as usize].nodes {
                    let c = colors[nb as usize];
                    if c != u16::MAX {
                        if c as usize >= used.len() {
                            used.resize(c as usize + 1, false);
                        }
                        used[c as usize] = true;
                    }
                }
            }
            colors[i] = used.iter().position(|&u| !u).unwrap_or(used.len()) as u16;
        }
        let n_colors = colors.iter().copied().max().map_or(1, |c| c as usize + 1);
        let mut classes: Vec<Vec<u32>> = vec![Vec::new(); n_colors];
        for i in 0..n {
            classes[colors[i] as usize].push(i as u32);
        }
        HetGraph { kinds: self.kinds, factors: self.factors, touching, colors, classes }
    }
}

impl Default for HetBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl HetGraph {
    pub fn n(&self) -> usize {
        self.kinds.len()
    }

    /// Row-major index of a factor's table for the given full state.
    fn f_index(&self, f: &Factor, state: &[u8]) -> usize {
        let mut idx = 0usize;
        for &node in &f.nodes {
            idx = idx * self.kinds[node as usize].states() + state[node as usize] as usize;
        }
        idx
    }

    /// Total energy of a full state (each entry the node's state index).
    pub fn energy(&self, state: &[u8]) -> f64 {
        self.factors.iter().map(|f| f.table[self.f_index(f, state)]).sum()
    }

    /// Sum of touching-factor energies with node i set to state k, others fixed.
    fn local_energy(&self, i: usize, k: u8, state: &mut [u8]) -> f64 {
        let old = state[i];
        state[i] = k;
        let mut e = 0.0;
        for &fi in &self.touching[i] {
            let f = &self.factors[fi as usize];
            e += f.table[self.f_index(f, state)];
        }
        state[i] = old;
        e
    }
}

pub struct HetSampler<'g> {
    pub g: &'g HetGraph,
    pub beta: f64,
    pub state: Vec<u8>,
    pub clamped: Vec<bool>,
    pub rng: Pcg,
}

impl<'g> HetSampler<'g> {
    pub fn new(g: &'g HetGraph, beta: f64, seed: u64) -> Self {
        let mut rng = Pcg::new(seed, 0x4E7);
        let state = (0..g.n())
            .map(|i| (rng.f64() * g.kinds[i].states() as f64) as u8)
            .collect();
        HetSampler { g, beta, state, clamped: vec![false; g.n()], rng }
    }

    pub fn clamp(&mut self, i: usize, k: u8) {
        assert!((k as usize) < self.g.kinds[i].states());
        self.state[i] = k;
        self.clamped[i] = true;
    }

    /// One full chromatic sweep: every free node resampled from its exact conditional once.
    pub fn sweep(&mut self, ledger: Option<&mut crate::ledger::Ledger>) {
        let mut probs: Vec<f64> = Vec::with_capacity(8);
        let mut updated = 0u64;
        for class in &self.g.classes {
            for &iu in class {
                let i = iu as usize;
                if self.clamped[i] {
                    continue;
                }
                let ks = self.g.kinds[i].states();
                probs.clear();
                let mut mx = f64::NEG_INFINITY;
                for k in 0..ks {
                    let l = -self.beta * self.g.local_energy(i, k as u8, &mut self.state);
                    probs.push(l);
                    if l > mx {
                        mx = l;
                    }
                }
                let mut z = 0.0;
                for p in probs.iter_mut() {
                    *p = (*p - mx).exp();
                    z += *p;
                }
                let mut u = self.rng.f64() * z;
                let mut pick = ks - 1;
                for (k, &p) in probs.iter().enumerate() {
                    if u < p {
                        pick = k;
                        break;
                    }
                    u -= p;
                }
                self.state[i] = pick as u8;
                updated += 1;
            }
        }
        if let Some(l) = ledger {
            l.samples += updated;
        }
    }

    pub fn sweeps(&mut self, n: usize, mut ledger: Option<&mut crate::ledger::Ledger>) {
        for _ in 0..n {
            self.sweep(ledger.as_deref_mut());
        }
    }
}

/// Exact Boltzmann distribution over all joint states (small systems), indexed mixed-radix
/// big-endian in node order (node 0 is the highest digit, matching factor-table convention).
pub fn exact_boltzmann(g: &HetGraph, beta: f64) -> Vec<f64> {
    let dims: Vec<usize> = (0..g.n()).map(|i| g.kinds[i].states()).collect();
    let total: usize = dims.iter().product();
    assert!(total <= 1 << 22, "exact enumeration too large");
    let mut p = vec![0.0; total];
    let mut state = vec![0u8; g.n()];

    // Max-shifted, for the reason spelled out in `ising::exact_boltzmann`: `exp(-beta*E)` overflows
    // f64 near exp(709), so a large beta turned this reference distribution into NaN. `HetSampler`
    // itself already shifts; the enumeration did not.
    let mut mx = f64::NEG_INFINITY;
    for m in 0..total {
        let mut rem = m;
        for i in (0..g.n()).rev() {
            state[i] = (rem % dims[i]) as u8;
            rem /= dims[i];
        }
        let l = -beta * g.energy(&state);
        p[m] = l;
        if l > mx {
            mx = l;
        }
    }
    let mut z = 0.0;
    for v in p.iter_mut() {
        *v = (*v - mx).exp();
        z += *v;
    }
    for v in p.iter_mut() {
        *v /= z;
    }
    p
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ising::tv;

    /// Mixed spin + categorical model with a genuine 3-ary factor: the sampled distribution must
    /// match exact Boltzmann enumeration — the same standard the spin sampler passed.
    #[test]
    fn mixed_model_matches_exact() {
        let mut hb = HetBuilder::new();
        let s0 = hb.node(Kind::Spin);
        let s1 = hb.node(Kind::Spin);
        let c0 = hb.node(Kind::Cat(3));
        let c1 = hb.node(Kind::Cat(3));
        hb.couple_spins(s0, s1, 0.6);
        hb.bias_spin(s0, 0.25);
        hb.factor(vec![s1, c0], vec![0.4, 0.0, -0.5, -0.5, 0.0, 0.4]);
        let mut agree = vec![0.3; 9];
        for k in 0..3 {
            agree[k * 3 + k] = -0.4;
        }
        hb.factor(vec![c0, c1], agree);
        let mut tri = vec![0.0; 2 * 3 * 3];
        for si in 0..2 {
            for a in 0..3 {
                for b in 0..3 {
                    tri[si * 9 + a * 3 + b] =
                        0.2 * (if si == 1 { 1.0 } else { -1.0 }) * (a as f64 - b as f64);
                }
            }
        }
        hb.factor(vec![s0, c0, c1], tri);
        let g = hb.build();
        let beta = 0.9;
        let exact = exact_boltzmann(&g, beta);

        let dims = [2usize, 2, 3, 3];
        let total: usize = dims.iter().product();
        let mut counts = vec![0u64; total];
        let mut smp = HetSampler::new(&g, beta, 0x4E7C0);
        smp.sweeps(300, None);
        let n_samples = 300_000;
        for _ in 0..n_samples {
            smp.sweep(None);
            let mut m = 0usize;
            for i in 0..4 {
                m = m * dims[i] + smp.state[i] as usize;
            }
            counts[m] += 1;
        }
        let emp: Vec<f64> = counts.iter().map(|&c| c as f64 / n_samples as f64).collect();
        let d = tv(&emp, &exact);
        assert!(d < 0.02, "TV(sampled, exact) = {d}");
    }

    /// The het engine on a pure-spin pairwise model must produce the same exact distribution as
    /// the spin engine — two independent code paths against one enumeration.
    #[test]
    fn reduces_to_spin_engine() {
        let mut hb = HetBuilder::new();
        let n: Vec<u32> = (0..4).map(|_| hb.node(Kind::Spin)).collect();
        hb.couple_spins(n[0], n[1], 0.7);
        hb.couple_spins(n[1], n[2], -0.4);
        hb.couple_spins(n[2], n[3], 0.55);
        hb.couple_spins(n[3], n[0], 0.3);
        hb.bias_spin(n[0], 0.2);
        hb.bias_spin(n[2], -0.35);
        let hg = hb.build();

        let mut gb = crate::graph::GraphBuilder::new(4);
        gb.couple(0, 1, 0.7);
        gb.couple(1, 2, -0.4);
        gb.couple(2, 3, 0.55);
        gb.couple(3, 0, 0.3);
        gb.bias(0, 0.2);
        gb.bias(2, -0.35);
        let sg = gb.build();

        let beta = 0.9;
        let p_het = exact_boltzmann(&hg, beta);
        let p_spin = crate::ising::exact_boltzmann(&sg, beta);
        // het index: node 0 = highest mixed-radix digit; spin mask: bit b set => spin b = +1
        for m in 0..16usize {
            let mut state = [0usize; 4];
            let mut rem = m;
            for i in (0..4).rev() {
                state[i] = rem % 2;
                rem /= 2;
            }
            let mut spin_mask = 0usize;
            for (b, &s) in state.iter().enumerate() {
                if s == 1 {
                    spin_mask |= 1 << b;
                }
            }
            assert!(
                (p_het[m] - p_spin[spin_mask]).abs() < 1e-12,
                "state {m}: het {} vs spin {}",
                p_het[m],
                p_spin[spin_mask]
            );
        }
    }

    /// Clamped categorical conditioning: with c0 clamped, the sampled conditional of c1 must
    /// match the exact conditional row.
    #[test]
    fn categorical_clamping() {
        let mut hb = HetBuilder::new();
        let c0 = hb.node(Kind::Cat(3));
        let _c1 = hb.node(Kind::Cat(3));
        let mut agree = vec![0.5; 9];
        for k in 0..3 {
            agree[k * 3 + k] = -0.6;
        }
        hb.factor(vec![c0, 1], agree);
        let g = hb.build();
        let beta = 1.0;
        let mut smp = HetSampler::new(&g, beta, 0xC1A);
        smp.clamp(0, 2);
        let mut counts = [0u64; 3];
        let n = 60_000;
        for _ in 0..n {
            smp.sweep(None);
            assert_eq!(smp.state[0], 2);
            counts[smp.state[1] as usize] += 1;
        }
        let ws: Vec<f64> =
            (0..3).map(|k| (-beta * (if k == 2 { -0.6 } else { 0.5 })).exp()).collect();
        let z: f64 = ws.iter().sum();
        for k in 0..3 {
            let got = counts[k] as f64 / n as f64;
            let want = ws[k] / z;
            assert!((got - want).abs() < 0.01, "p(c1={k}) {got:.4} vs {want:.4}");
        }
    }
}
