//! Denoising Thermodynamic Models — the flagship architecture of the thermodynamic-computing
//! program (Jelincic et al., arXiv:2510.23972 / npj Unconventional Computing 2026): a chain of
//! shallow Boltzmann machines, each denoising one step of a closed-form forward noising process.
//! Capacity comes from the CHAIN, not any single EBM — each conditional stays easy to sample,
//! which is the same capacity-through-factorization law our chain benches measured.
//!
//! Everything here is verified against exact enumeration before the sampled training path is
//! trusted, including the one thing the paper's own text gets wrong: the printed sign of the
//! forward-coupling energy (their Eq. D1). The keep-probability test below pins the correct
//! NEGATIVE sign under the P proportional to e^{-E} convention; the printed sign fails it.

use crate::rng::Pcg;

// ---------- forward process (per-site uniform-jump kernels, closed form) ----------

/// Gamma(t) = ln( (1 + (M-1) e^{-gamma t}) / (1 - e^{-gamma t}) ) — the coupling strength that
/// writes the closed-form jump kernel as (1/Z) exp(Gamma * delta_{x', x}).
pub fn gamma_coupling(gamma: f64, t: f64, m: usize) -> f64 {
    let e = (-gamma * t).exp();
    ((1.0 + (m as f64 - 1.0) * e) / (1.0 - e)).ln()
}

/// Keep-probability of the M-state uniform-jump kernel over time t.
pub fn keep_prob(gamma: f64, t: f64, m: usize) -> f64 {
    let e = (-gamma * t).exp();
    (1.0 + (m as f64 - 1.0) * e) / m as f64
}

/// One forward step for binary spins: each site keeps its value w.p. keep_prob, else flips.
pub fn forward_step(x: &mut [i8], gamma: f64, dt: f64, rng: &mut Pcg) {
    let keep = keep_prob(gamma, dt, 2);
    for s in x.iter_mut() {
        if rng.f64() >= keep {
            *s = -*s;
        }
    }
}

// ---------- pattern grids (Table II of the DTM paper) ----------

/// Connection-rule orbit: (a,b) adds offsets (a,b), (-b,a), (-a,-b), (b,-a). All published rules
/// have a+b odd, so the graphs are bipartite under checkerboard parity.
pub const G8: [(i64, i64); 2] = [(0, 1), (4, 1)];
pub const G12: [(i64, i64); 3] = [(0, 1), (4, 1), (9, 10)];
pub const G16: [(i64, i64); 4] = [(0, 1), (4, 1), (8, 7), (14, 9)];

/// Edge list of an L x L pattern grid (open boundaries, deduplicated undirected edges).
pub fn pattern_grid(l: usize, rules: &[(i64, i64)]) -> Vec<(u32, u32)> {
    let mut edges = Vec::new();
    for y in 0..l as i64 {
        for x in 0..l as i64 {
            let i = (y * l as i64 + x) as u32;
            for &(a, b) in rules {
                for (dx, dy) in [(a, b), (-b, a), (-a, -b), (b, -a)] {
                    let (nx, ny) = (x + dx, y + dy);
                    if nx >= 0 && ny >= 0 && nx < l as i64 && ny < l as i64 {
                        let j = (ny * l as i64 + nx) as u32;
                        if j > i {
                            edges.push((i, j));
                        }
                    }
                }
            }
        }
    }
    edges.sort_unstable();
    edges.dedup();
    edges
}

// ---------- a small self-contained trainable EBM (explicit edge list) ----------

pub struct Ebm {
    pub n: usize,
    pub edges: Vec<(u16, u16)>,
    pub j: Vec<f64>,
    pub h: Vec<f64>,
    /// CSR adjacency: for node i, neighbours in `nbr[offset[i]..offset[i+1]]` with the index of
    /// the edge that connects them in `eidx`. Without this a sweep is O(N * E) and the published
    /// scale (4,900 nodes, ~29k edges) is unreachable; with it a sweep is O(E).
    offset: Vec<u32>,
    nbr: Vec<u32>,
    eidx: Vec<u32>,
    /// A proper 2-colouring. Every published pattern grid is bipartite (all connection rules have
    /// odd Manhattan length), so one sweep is two independent half sweeps.
    pub classes: [Vec<u32>; 2],
    /// What the BFS actually concluded, rather than a re-inference from `classes`.
    bipartite: bool,
}

impl Ebm {
    pub fn new(n: usize, edges: Vec<(u16, u16)>) -> Ebm {
        // Endpoints are u16 while `n` is usize, so an n past the u16 space means a caller narrowing
        // its own indices aliases them and builds a DIFFERENT GRAPH without error: every truncated
        // index is still < n, so nothing indexes out of bounds and nothing panics. Measured on
        // `pattern_grid(300, ..)`: 90,000 nodes collapse to 65,536 (27% aliased), 525,188 edges to
        // 389,603, and the aliasing introduces an odd cycle so the chromatic sweep silently
        // degrades to sequential. The threshold is n > 65,536, i.e. a grid side of 257.
        //
        // The type-level fix is to widen `edges` to `(u32, u32)`, which is what `pattern_grid`
        // already emits and would delete the narrowing at every call site. That is a breaking
        // change to a published signature; this refusal is not, and it converts silent corruption
        // into a message.
        assert!(
            n <= u16::MAX as usize + 1,
            "n={n} exceeds the u16 edge-index space (max {}), so edge endpoints would alias \
             silently and build a different graph",
            u16::MAX as usize + 1
        );
        let ne = edges.len();
        let mut deg = vec![0u32; n];
        for &(a, b) in &edges {
            deg[a as usize] += 1;
            deg[b as usize] += 1;
        }
        let mut offset = vec![0u32; n + 1];
        for i in 0..n {
            offset[i + 1] = offset[i] + deg[i];
        }
        let mut nbr = vec![0u32; offset[n] as usize];
        let mut eidx = vec![0u32; offset[n] as usize];
        let mut cur = offset.clone();
        for (k, &(a, b)) in edges.iter().enumerate() {
            let (a, b) = (a as usize, b as usize);
            nbr[cur[a] as usize] = b as u32;
            eidx[cur[a] as usize] = k as u32;
            cur[a] += 1;
            nbr[cur[b] as usize] = a as u32;
            eidx[cur[b] as usize] = k as u32;
            cur[b] += 1;
        }
        // BFS 2-colouring; falls back to "everything in one class" if the graph is not bipartite,
        // which would make sweeps sequential rather than wrong.
        let mut colour = vec![u8::MAX; n];
        let mut bipartite = true;
        let mut stack = Vec::new();
        for s in 0..n {
            if colour[s] != u8::MAX {
                continue;
            }
            colour[s] = 0;
            stack.push(s);
            while let Some(u) = stack.pop() {
                for k in offset[u]..offset[u + 1] {
                    let v = nbr[k as usize] as usize;
                    if colour[v] == u8::MAX {
                        colour[v] = 1 - colour[u];
                        stack.push(v);
                    } else if colour[v] == colour[u] {
                        bipartite = false;
                    }
                }
            }
        }
        let mut classes = [Vec::new(), Vec::new()];
        for i in 0..n {
            let c = if bipartite { colour[i] as usize } else { 0 };
            classes[c].push(i as u32);
        }
        Ebm { n, edges, j: vec![0.0; ne], h: vec![0.0; n], offset, nbr, eidx, classes, bipartite }
    }

    pub fn is_bipartite(&self) -> bool {
        // The BFS above already knows. This used to re-infer it as `!classes[1].is_empty()`, which
        // is a different question: a graph with NO EDGES is bipartite, every node lands in class 0,
        // and the inference reported false. Measured: edgeless -> false, single edge -> true,
        // triangle -> false. Two of those three were right for the wrong reason.
        self.bipartite
    }

    pub fn energy(&self, s: &[i8]) -> f64 {
        let mut e = 0.0;
        for (k, &(a, b)) in self.edges.iter().enumerate() {
            e -= self.j[k] * (s[a as usize] * s[b as usize]) as f64;
        }
        for i in 0..self.n {
            e -= self.h[i] * s[i] as f64;
        }
        e
    }

    #[inline]
    fn field(&self, i: usize, s: &[i8], extra: &[f64]) -> f64 {
        let mut f = self.h[i] + extra[i];
        for k in self.offset[i]..self.offset[i + 1] {
            let k = k as usize;
            f += self.j[self.eidx[k] as usize] * s[self.nbr[k] as usize] as f64;
        }
        f
    }

    /// Gibbs sweeps over the nodes in `free`, with per-node extra external fields.
    pub fn gibbs(&self, s: &mut [i8], free: &[usize], extra: &[f64], sweeps: usize, rng: &mut Pcg) {
        self.gibbs_at(s, free, extra, sweeps, 1.0, rng)
    }

    /// As [`Self::gibbs`], with beta as a runtime parameter.
    pub fn gibbs_at(
        &self,
        s: &mut [i8],
        free: &[usize],
        extra: &[f64],
        sweeps: usize,
        beta: f64,
        rng: &mut Pcg,
    ) {
        for _ in 0..sweeps {
            for &i in free {
                let f = self.field(i, s, extra);
                s[i] = crate::kernel::draw(f, beta, rng);
            }
        }
    }

    /// Chromatic sweeps over ALL nodes: two half sweeps per sweep, the schedule hardware uses.
    pub fn gibbs_chromatic(&self, s: &mut [i8], extra: &[f64], sweeps: usize, rng: &mut Pcg) {
        self.gibbs_chromatic_at(s, extra, sweeps, 1.0, rng)
    }

    /// As [`Self::gibbs_chromatic`], with beta as a runtime parameter.
    ///
    /// Beta used to live inside the weights here, which meant annealing this model required
    /// rewriting every coupling. It is a number now.
    pub fn gibbs_chromatic_at(
        &self,
        s: &mut [i8],
        extra: &[f64],
        sweeps: usize,
        beta: f64,
        rng: &mut Pcg,
    ) {
        for _ in 0..sweeps {
            for c in 0..2 {
                for idx in 0..self.classes[c].len() {
                    let i = self.classes[c][idx] as usize;
                    let f = self.field(i, s, extra);
                    s[i] = crate::kernel::draw(f, beta, rng);
                }
            }
        }
    }

    /// Accumulate sufficient statistics from the current state.
    #[inline]
    pub fn accumulate(&self, s: &[i8], ss: &mut [f64], si: &mut [f64], w: f64) {
        for (k, &(a, b)) in self.edges.iter().enumerate() {
            ss[k] += w * (s[a as usize] * s[b as usize]) as f64;
        }
        for i in 0..self.n {
            si[i] += w * s[i] as f64;
        }
    }
}

// ---------- the DTM ----------

/// A T-step DTM over `n` sites per step (first `nv` visible, rest latent), binary spins,
/// uniform forward jump rate `gamma`, step times t_0 < .. < t_T.
pub struct Dtm {
    pub steps: Vec<Ebm>,
    pub nv: usize,
    pub gamma: f64,
    pub times: Vec<f64>,
}

impl Dtm {
    pub fn new(t_steps: usize, n: usize, nv: usize, edges: Vec<(u16, u16)>, gamma: f64, times: Vec<f64>) -> Dtm {
        assert_eq!(times.len(), t_steps + 1);
        Dtm {
            steps: (0..t_steps).map(|_| Ebm::new(n, edges.clone())).collect(),
            nv,
            gamma,
            times,
        }
    }

    /// The forward-coupling field on visible site i from the clamped x^t value: Gamma(dt)/2 * x^t_i.
    /// (The NEGATIVE-sign energy E^f = -(1/2) sum Gamma x^t x^{t-1}; the paper's printed Eq. D1
    /// sign fails the keep-probability test below.)
    fn clamp_field(&self, t: usize, xt: &[i8]) -> Vec<f64> {
        let dt = self.times[t + 1] - self.times[t];
        let g = gamma_coupling(self.gamma, dt, 2);
        let n = self.steps[0].n;
        let mut extra = vec![0.0; n];
        for i in 0..self.nv {
            extra[i] = 0.5 * g * xt[i] as f64;
        }
        extra
    }

    /// One contrastive gradient step at chain position t (0-based: models P(x^t | x^{t+1})),
    /// from a batch of (x_prev = x^t, x_next = x^{t+1}) pairs, with K Gibbs sweeps per phase.
    /// Returns the parameter update applied (for inspection). lambda_tc weights the TC penalty.
    pub fn train_step(
        &mut self,
        t: usize,
        batch: &[(Vec<i8>, Vec<i8>)],
        k_sweeps: usize,
        lr: f64,
        lambda_tc: f64,
        rng: &mut Pcg,
    ) {
        let n = self.steps[t].n;
        let ne = self.steps[t].edges.len();
        let nv = self.nv;
        let latents: Vec<usize> = (nv..n).collect();
        let all: Vec<usize> = (0..n).collect();
        let mut pos_ss = vec![0.0; ne];
        let mut pos_s = vec![0.0; n];
        let mut neg_ss = vec![0.0; ne];
        let mut neg_s = vec![0.0; n];
        for (x_prev, x_next) in batch {
            let extra = self.clamp_field(t, x_next);
            let ebm = &self.steps[t];
            // positive phase: visibles clamped to x_prev, Gibbs over latents
            let mut s = vec![1i8; n];
            s[..nv].copy_from_slice(x_prev);
            for i in nv..n {
                s[i] = if rng.f64() < 0.5 { 1 } else { -1 };
            }
            ebm.gibbs(&mut s, &latents, &extra, k_sweeps, rng);
            for (k, &(a, b)) in ebm.edges.iter().enumerate() {
                pos_ss[k] += (s[a as usize] * s[b as usize]) as f64;
            }
            for i in 0..n {
                pos_s[i] += s[i] as f64;
            }
            // negative phase: everything free under the same clamp field
            let mut sneg: Vec<i8> = (0..n).map(|_| if rng.f64() < 0.5 { 1 } else { -1 }).collect();
            ebm.gibbs(&mut sneg, &all, &extra, k_sweeps, rng);
            for (k, &(a, b)) in ebm.edges.iter().enumerate() {
                neg_ss[k] += (sneg[a as usize] * sneg[b as usize]) as f64;
            }
            for i in 0..n {
                neg_s[i] += sneg[i] as f64;
            }
        }
        let m = batch.len() as f64;
        let ebm = &mut self.steps[t];
        // grad L_DN: E_pos[dE] - E_neg[dE], dE/dJ = -ss, dE/dh = -s  =>  descent step:
        for k in 0..ne {
            let mut g = -(pos_ss[k] - neg_ss[k]) / m;
            if lambda_tc > 0.0 {
                // TC penalty (closed form H3/H4, reusing negative-phase statistics):
                // grad_J = -(m_a m_b - <s_a s_b>); grad_h cancels exactly.
                let (a, b) = ebm.edges[k];
                let (ma, mb) = (neg_s[a as usize] / m, neg_s[b as usize] / m);
                g += lambda_tc * -(ma * mb - neg_ss[k] / m);
            }
            ebm.j[k] -= lr * g;
        }
        for i in 0..n {
            ebm.h[i] -= lr * -(pos_s[i] - neg_s[i]) / m;
        }
    }

    /// Reverse-chain sampling: x^T uniform, then for t = T-1 down to 0 clamp x^{t+1} and Gibbs
    /// (x^t, z) jointly for k_mix sweeps. Returns x^0 (visible sites).
    pub fn sample(&self, k_mix: usize, rng: &mut Pcg) -> Vec<i8> {
        let n = self.steps[0].n;
        let nv = self.nv;
        let all: Vec<usize> = (0..n).collect();
        let mut x: Vec<i8> = (0..nv).map(|_| if rng.f64() < 0.5 { 1 } else { -1 }).collect();
        for t in (0..self.steps.len()).rev() {
            let extra = self.clamp_field(t, &x);
            let mut s: Vec<i8> = (0..n).map(|_| if rng.f64() < 0.5 { 1 } else { -1 }).collect();
            self.steps[t].gibbs(&mut s, &all, &extra, k_mix, rng);
            x.copy_from_slice(&s[..nv]);
        }
        x
    }

    // ---------- exact-enumeration reference machinery (small systems only) ----------

    /// Exact log P_theta(x_prev | x_next) at step t (latents summed out).
    pub fn exact_log_cond(&self, t: usize, x_prev: &[i8], x_next: &[i8]) -> f64 {
        let n = self.steps[t].n;
        let nv = self.nv;
        let nl = n - nv;
        let extra = self.clamp_field(t, x_next);
        let ebm = &self.steps[t];
        let full_e = |xv: &[i8], zm: usize| -> f64 {
            let mut s = vec![0i8; n];
            s[..nv].copy_from_slice(xv);
            for l in 0..nl {
                s[nv + l] = if zm >> l & 1 == 1 { 1 } else { -1 };
            }
            let mut e = ebm.energy(&s);
            for i in 0..n {
                e -= extra[i] * s[i] as f64;
            }
            e
        };
        // Log-sum-exp, both accumulators. The result is `num.ln() - den.ln()`, so this wanted log
        // space all along -- and exponentiating first overflows f64 near exp(709), which turned a
        // conditional log-probability into NaN at exactly the low temperatures a denoiser runs at.
        // Shifting by the max is exact: it cancels in the ratio.
        let lse = |mut acc: Vec<f64>| -> f64 {
            let mx = acc.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            if !mx.is_finite() {
                return mx; // all -inf, or an already-broken energy; do not manufacture a number
            }
            let mut s = 0.0;
            for v in acc.iter_mut() {
                s += (*v - mx).exp();
            }
            mx + s.ln()
        };

        let mut num_l = Vec::with_capacity(1usize << nl);
        for zm in 0..(1usize << nl) {
            num_l.push(-full_e(x_prev, zm));
        }
        let mut den_l = Vec::with_capacity(1usize << (nv + nl));
        let mut xv = vec![0i8; nv];
        for xm in 0..(1usize << nv) {
            for b in 0..nv {
                xv[b] = if xm >> b & 1 == 1 { 1 } else { -1 };
            }
            for zm in 0..(1usize << nl) {
                den_l.push(-full_e(&xv, zm));
            }
        }
        lse(num_l) - lse(den_l)
    }

    /// Exact theta-dependent loss: NLL(theta) = -E_Q[ sum_t ln P_theta(x^t | x^{t+1}) ] where Q
    /// is the exact forward chain from a uniform mixture over `data` patterns.
    pub fn exact_nll(&self, data: &[Vec<i8>]) -> f64 {
        let nv = self.nv;
        let t_steps = self.steps.len();
        let mut nll = 0.0;
        // enumerate trajectories (x^0..x^T), each visible mask
        let masks = 1usize << nv;
        let to_x = |m: usize| -> Vec<i8> {
            (0..nv).map(|b| if m >> b & 1 == 1 { 1 } else { -1 }).collect()
        };
        let step_q = |xa: &[i8], xb: &[i8], dt: f64| -> f64 {
            let keep = keep_prob(self.gamma, dt, 2);
            let mut q = 1.0;
            for i in 0..nv {
                q *= if xa[i] == xb[i] { keep } else { 1.0 - keep };
            }
            q
        };
        // recursive product over chain: Q(x0) from data mixture, then per-step kernels
        let mut traj = vec![0usize; t_steps + 1];
        loop {
            // compute Q(traj) and add contribution
            let x0 = to_x(traj[0]);
            let mut q = data
                .iter()
                .map(|d| if d[..] == x0[..] { 1.0 / data.len() as f64 } else { 0.0 })
                .sum::<f64>();
            if q > 0.0 {
                for t in 0..t_steps {
                    let xa = to_x(traj[t]);
                    let xb = to_x(traj[t + 1]);
                    q *= step_q(&xa, &xb, self.times[t + 1] - self.times[t]);
                }
                if q > 0.0 {
                    let mut lp = 0.0;
                    for t in 0..t_steps {
                        lp += self.exact_log_cond(t, &to_x(traj[t]), &to_x(traj[t + 1]));
                    }
                    nll -= q * lp;
                }
            }
            // odometer over trajectories
            let mut c = 0;
            loop {
                traj[c] += 1;
                if traj[c] < masks {
                    break;
                }
                traj[c] = 0;
                c += 1;
                if c > t_steps {
                    return nll;
                }
            }
        }
    }
}

/// The ACP (autocorrelation-penalty) controller update law (DTM paper, Appendix H):
/// eps = 0.03, delta = 0.2, lambda_min = 1e-4 are the published defaults.
pub fn acp_update(
    lambda: f64,
    a_m: f64,
    a_prev: Option<f64>,
    eps: f64,
    delta: f64,
    lambda_min: f64,
) -> f64 {
    let lp = lambda.max(lambda_min);
    let out = if a_m < eps {
        (1.0 - delta) * lp
    } else if a_prev.is_none() || a_m <= a_prev.unwrap() {
        lp
    } else {
        (1.0 + delta) * lp
    };
    if out < lambda_min {
        0.0
    } else {
        out
    }
}

/// Normalized autocorrelation of a series at lag k (time-average estimator).
pub fn autocorr(series: &[f64], k: usize) -> f64 {
    let n = series.len();
    assert!(k < n);
    let mu = series.iter().sum::<f64>() / n as f64;
    let var = series.iter().map(|v| (v - mu) * (v - mu)).sum::<f64>() / n as f64;
    if var == 0.0 {
        return 0.0;
    }
    let mut c = 0.0;
    for i in 0..n - k {
        c += (series[i] - mu) * (series[i + k] - mu);
    }
    c / ((n - k) as f64 * var)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_edgeless_graph_is_bipartite_and_a_triangle_is_not() {
        // `is_bipartite` re-inferred the answer as `!classes[1].is_empty()`, which is a different
        // question from the one the BFS had already answered. A graph with no edges IS bipartite --
        // every node lands in class 0 and the inference said false.
        assert!(Ebm::new(4, vec![]).is_bipartite(), "no edges means trivially bipartite");
        assert!(Ebm::new(2, vec![(0, 1)]).is_bipartite(), "a single edge is bipartite");
        assert!(
            !Ebm::new(3, vec![(0, 1), (1, 2), (2, 0)]).is_bipartite(),
            "a triangle has an odd cycle"
        );
        // and a 4-cycle is, which the old inference also got right -- for the right reason now
        assert!(Ebm::new(4, vec![(0, 1), (1, 2), (2, 3), (3, 0)]).is_bipartite());
    }

    #[test]
    #[should_panic(expected = "u16 edge-index space")]
    fn a_node_count_the_edge_indices_cannot_address_is_refused() {
        // Endpoints are u16 while n is usize. Past 65,536 a caller narrowing its own indices
        // aliases them and builds a different graph with no error at all: every truncated index is
        // still < n, so nothing is out of bounds and nothing panics. `pattern_grid(300, ..)`
        // collapsed 90,000 nodes to 65,536 and introduced an odd cycle.
        let _ = Ebm::new(u16::MAX as usize + 2, vec![]);
    }

    /// THE SIGN TRAP (the paper's printed Eq. D1): the energy-form forward coupling with the
    /// NEGATIVE sign must reproduce the closed-form keep probability sigma(Gamma) =
    /// (1 + e^{-gamma t})/2 exactly; the printed positive sign gives 1 - keep and fails.
    #[test]
    fn forward_sign_trap() {
        for gamma in [0.3, 1.0, 2.5] {
            for t in [0.1, 0.5, 1.0, 3.0] {
                let g = gamma_coupling(gamma, t, 2);
                let keep_closed = keep_prob(gamma, t, 2);
                // our convention: clamp field = +Gamma/2 * x^t; isolated site keep-prob =
                // sigma(2 * field) with field = Gamma/2 => sigma(Gamma)
                let keep_energy = 1.0 / (1.0 + (-g).exp());
                assert!(
                    (keep_energy - keep_closed).abs() < 1e-14,
                    "gamma {gamma} t {t}: energy-form keep {keep_energy} vs closed {keep_closed}"
                );
                // the printed sign gives sigma(-Gamma), which is EXACTLY the complement of the
                // keep probability — right only in the Gamma -> 0 noise-saturation limit, and
                // glaringly wrong at short times (keep > 1/2 always).
                let wrong = 1.0 / (1.0 + g.exp());
                assert!((wrong - (1.0 - keep_closed)).abs() < 1e-14);
                if keep_closed - 0.5 > 1e-3 {
                    assert!((wrong - keep_closed).abs() > 1e-3, "trap not discriminable at gamma {gamma}, t {t}");
                }
            }
        }
    }

    /// Semigroup property of the closed-form kernel: keep(t1) composed with keep(t2) equals
    /// keep(t1 + t2) for binary spins (flip parity algebra).
    #[test]
    fn kernel_semigroup() {
        for gamma in [0.4, 1.3] {
            for (t1, t2) in [(0.2, 0.7), (0.5, 0.5), (1.0, 2.0)] {
                let k1 = keep_prob(gamma, t1, 2);
                let k2 = keep_prob(gamma, t2, 2);
                let k12 = keep_prob(gamma, t1 + t2, 2);
                // P(keep over both) = k1 k2 + (1-k1)(1-k2)
                let composed = k1 * k2 + (1.0 - k1) * (1.0 - k2);
                assert!((composed - k12).abs() < 1e-14);
            }
        }
    }

    /// Pattern grids: interior degree = 4 * #rules; every edge flips checkerboard parity.
    #[test]
    fn pattern_grid_structure() {
        for (rules, deg) in [(&G8[..], 8usize), (&G12[..], 12), (&G16[..], 16)] {
            let l = 40usize;
            let edges = pattern_grid(l, rules);
            let mut count = vec![0usize; l * l];
            for &(a, b) in &edges {
                count[a as usize] += 1;
                count[b as usize] += 1;
                let (ax, ay) = (a as usize % l, a as usize / l);
                let (bx, by) = (b as usize % l, b as usize / l);
                assert_eq!((ax + ay + bx + by) % 2, 1, "edge does not flip parity");
            }
            // an interior node: safely away from boundaries
            let i = (l / 2) * l + l / 2;
            assert_eq!(count[i], deg, "interior degree for rule set of {} rules", rules.len());
        }
    }

    /// ACP controller law on the scripted sequence from the spec.
    #[test]
    fn acp_scripted_sequence() {
        let (eps, delta, lmin) = (0.03, 0.2, 1e-4);
        let a = [0.5, 0.6, 0.4, 0.01];
        let mut lambda = 0.01;
        let mut prev: Option<f64> = None;
        let mut traj = Vec::new();
        for (m, &am) in a.iter().enumerate() {
            let ap = if m == 0 { None } else { prev };
            lambda = acp_update(lambda, am, ap, eps, delta, lmin);
            traj.push(lambda);
            prev = Some(am);
        }
        let want = [0.01, 0.012, 0.012, 0.0096];
        for (got, want) in traj.iter().zip(&want) {
            assert!((got - want).abs() < 1e-12, "traj {:?} vs {:?}", traj, want);
        }
    }

    /// Autocorrelation estimator: the alternating sequence gives exactly (-1)^k.
    #[test]
    fn autocorr_alternating_exact() {
        let series: Vec<f64> = (0..1000).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
        for k in 0..5 {
            let want = if k % 2 == 0 { 1.0 } else { -1.0 };
            assert!((autocorr(&series, k) - want).abs() < 1e-12);
        }
    }

    /// THE GOLD TEST: on a fully-enumerable DTM (T = 2, 3 visible + 2 latent), the Eq.-14
    /// gradient with EXACT conditional expectations must match central finite differences of the
    /// exact NLL for every parameter.
    #[test]
    fn exact_enumeration_gradient_check() {
        let edges: Vec<(u16, u16)> = vec![(0, 1), (1, 2), (0, 3), (1, 3), (2, 4), (1, 4), (3, 4)];
        let mut dtm = Dtm::new(2, 5, 3, edges.clone(), 1.0, vec![0.0, 1.0, 2.0]);
        // arbitrary small parameters
        let mut rng = Pcg::new(0x601D, 1);
        for t in 0..2 {
            for j in dtm.steps[t].j.iter_mut() {
                *j = (rng.f64() - 0.5) * 0.6;
            }
            for h in dtm.steps[t].h.iter_mut() {
                *h = (rng.f64() - 0.5) * 0.4;
            }
        }
        let data = vec![vec![1i8, 1, -1], vec![-1, 1, 1]];

        // Eq. 14 with exact expectations, step t, parameter = (t, edge k) and (t, bias i)
        let nv = 3usize;
        let nl = 2usize;
        let to_x = |m: usize| -> Vec<i8> {
            (0..nv).map(|b| if m >> b & 1 == 1 { 1 } else { -1 }).collect()
        };
        let keep = |dt: f64| keep_prob(1.0, dt, 2);
        let q_step = |xa: &[i8], xb: &[i8], dt: f64| -> f64 {
            let k = keep(dt);
            (0..nv).map(|i| if xa[i] == xb[i] { k } else { 1.0 - k }).product()
        };
        // Q(x^t = a, x^{t+1} = b) for t in {0, 1}
        let q_pair = |t: usize, a: usize, b: usize| -> f64 {
            let xa = to_x(a);
            let xb = to_x(b);
            match t {
                0 => {
                    let q0: f64 = data
                        .iter()
                        .map(|d| if d[..] == xa[..] { 0.5 } else { 0.0 })
                        .sum();
                    q0 * q_step(&xa, &xb, 1.0)
                }
                _ => {
                    // Q(x1 = a) = sum_x0 Q0(x0) Q(a | x0); then times Q(b | a)
                    let mut q1 = 0.0;
                    for m0 in 0..8usize {
                        let x0 = to_x(m0);
                        let q0: f64 = data
                            .iter()
                            .map(|d| if d[..] == x0[..] { 0.5 } else { 0.0 })
                            .sum();
                        q1 += q0 * q_step(&x0, &xa, 1.0);
                    }
                    q1 * q_step(&xa, &xb, 1.0)
                }
            }
        };
        // exact conditional expectations of sufficient statistics
        let stat = |dtm: &Dtm, t: usize, clamp_prev: Option<usize>, b_mask: usize| -> (Vec<f64>, Vec<f64>) {
            // returns (E[s_a s_b] per edge, E[s_i] per node) under P_theta(. | constraints)
            let ebm = &dtm.steps[t];
            let extra = dtm.clamp_field(t, &to_x(b_mask));
            let mut zsum = 0.0;
            let mut ess = vec![0.0; ebm.edges.len()];
            let mut es = vec![0.0; ebm.n];
            let xs: Vec<usize> = match clamp_prev {
                Some(a) => vec![a],
                None => (0..8).collect(),
            };
            for &xm in &xs {
                let xv = to_x(xm);
                for zm in 0..(1usize << nl) {
                    let mut s = vec![0i8; 5];
                    s[..nv].copy_from_slice(&xv);
                    for l in 0..nl {
                        s[nv + l] = if zm >> l & 1 == 1 { 1 } else { -1 };
                    }
                    let mut e = ebm.energy(&s);
                    for i in 0..5 {
                        e -= extra[i] * s[i] as f64;
                    }
                    let w = (-e).exp();
                    zsum += w;
                    for (k, &(a, b)) in ebm.edges.iter().enumerate() {
                        ess[k] += w * (s[a as usize] * s[b as usize]) as f64;
                    }
                    for i in 0..5 {
                        es[i] += w * s[i] as f64;
                    }
                }
            }
            for v in ess.iter_mut() {
                *v /= zsum;
            }
            for v in es.iter_mut() {
                *v /= zsum;
            }
            (ess, es)
        };
        // gradient per Eq. 14: sum over (a, b) pairs weighted by Q, positive phase minus over
        // b weighted by Q(x^{t+1} = b), negative phase; dE/dJ = -ss, dE/dh = -s.
        let ne = edges.len();
        for t in 0..2usize {
            let mut g_j = vec![0.0; ne];
            let mut g_h = vec![0.0; 5];
            for b_mask in 0..8usize {
                let mut qb = 0.0;
                for a_mask in 0..8usize {
                    let q = q_pair(t, a_mask, b_mask);
                    if q == 0.0 {
                        continue;
                    }
                    qb += q;
                    let (ess, es) = stat(&dtm, t, Some(a_mask), b_mask);
                    for k in 0..ne {
                        g_j[k] += q * -ess[k];
                    }
                    for i in 0..5 {
                        g_h[i] += q * -es[i];
                    }
                }
                if qb == 0.0 {
                    continue;
                }
                let (ess, es) = stat(&dtm, t, None, b_mask);
                for k in 0..ne {
                    g_j[k] -= qb * -ess[k];
                }
                for i in 0..5 {
                    g_h[i] -= qb * -es[i];
                }
            }
            // finite differences of the exact NLL
            let fd = 1e-6;
            for k in 0..ne {
                let orig = dtm.steps[t].j[k];
                dtm.steps[t].j[k] = orig + fd;
                let up = dtm.exact_nll(&data);
                dtm.steps[t].j[k] = orig - fd;
                let dn = dtm.exact_nll(&data);
                dtm.steps[t].j[k] = orig;
                let want = (up - dn) / (2.0 * fd);
                assert!(
                    (g_j[k] - want).abs() < 1e-7,
                    "t {t} J[{k}]: eq14 {} vs FD {}",
                    g_j[k],
                    want
                );
            }
            for i in 0..5 {
                let orig = dtm.steps[t].h[i];
                dtm.steps[t].h[i] = orig + fd;
                let up = dtm.exact_nll(&data);
                dtm.steps[t].h[i] = orig - fd;
                let dn = dtm.exact_nll(&data);
                dtm.steps[t].h[i] = orig;
                let want = (up - dn) / (2.0 * fd);
                assert!(
                    (g_h[i] - want).abs() < 1e-7,
                    "t {t} h[{i}]: eq14 {} vs FD {}",
                    g_h[i],
                    want
                );
            }
        }
    }

    /// End-to-end training smoke, scored EXACTLY: sampled-gradient training on a toy dataset
    /// must reduce the exact NLL by a clear margin.
    #[test]
    fn training_reduces_exact_nll() {
        let edges: Vec<(u16, u16)> = vec![(0, 1), (1, 2), (2, 3), (0, 4), (1, 4), (2, 5), (3, 5)];
        let mut dtm = Dtm::new(2, 6, 4, edges, 1.2, vec![0.0, 0.8, 1.6]);
        let data = vec![vec![1i8, 1, 1, 1], vec![-1, -1, -1, -1]];
        let nll0 = dtm.exact_nll(&data);
        let mut rng = Pcg::new(0x7247, 3);
        for _iter in 0..150 {
            // draw forward pairs from the data
            for t in 0..2 {
                let mut batch = Vec::new();
                for _ in 0..24 {
                    let d = &data[(rng.f64() * 2.0) as usize % 2];
                    let mut xa = d.clone();
                    // evolve to time t
                    if t > 0 {
                        forward_step(&mut xa, dtm.gamma, dtm.times[t] - dtm.times[0], &mut rng);
                    }
                    let mut xb = xa.clone();
                    forward_step(&mut xb, dtm.gamma, dtm.times[t + 1] - dtm.times[t], &mut rng);
                    batch.push((xa, xb));
                }
                dtm.train_step(t, &batch, 25, 0.05, 0.0, &mut rng);
            }
        }
        let nll1 = dtm.exact_nll(&data);
        assert!(
            nll1 < nll0 - 0.5,
            "training did not reduce exact NLL: {nll0:.3} -> {nll1:.3}"
        );
    }
}
