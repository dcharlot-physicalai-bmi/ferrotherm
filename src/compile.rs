//! Variational compilation — fit a target conditional kernel P(y|x) with the conditional of a
//! DEVICE-NATIVE Boltzmann machine: couplings restricted to the hardware graph's actual edges,
//! inputs clamped, hidden spins marginalized.
//!
//! ```text
//! P~(y|x; phi) = sum_h exp(-beta E(x,y,h; phi)) / Z(x)
//! objective per factor:  eps(x) = D_KL( P(.|x) || P~(.|x; phi) ),  averaged over inputs mu.
//! ```
//!
//! For a chain of compiled factors the chain rule of KL gives (the compilation error bound,
//! arXiv:2608.01615 Eq. 17 — the same shape as our chain law, in KL form):
//!
//! ```text
//! KL(readout) <= KL(trajectory) = sum_l E_{x ~ mu_l}[ eps_l(x) ]
//! ```
//!
//! Everything here is EXACT on enumerable kernels (free bits <= ~13): exact conditionals, exact
//! KL, exact positive/negative-phase gradients. Verify the math small, then scale with samplers.

use crate::graph::Graph;
use crate::ledger::Ledger;
use crate::rng::Pcg;

/// A device-native conditional kernel on a hardware graph patch.
/// Node roles: `n_in` clamped inputs, `n_out` read-out nodes, `n_hid` marginalized hidden nodes.
/// Only edges present in the device graph exist; edges between two inputs are dropped (both ends
/// clamped, constant energy). Free = out ++ hid, indexed out-first.
pub struct Kernel {
    pub n_in: usize,
    pub n_out: usize,
    pub n_hid: usize,
    /// free-free native edges (indices into free vector)
    pub e_ff: Vec<(u16, u16)>,
    /// input-free native edges (input index, free index)
    pub e_if: Vec<(u16, u16)>,
    pub j_ff: Vec<f64>,
    pub j_if: Vec<f64>,
    /// biases of free nodes
    pub b: Vec<f64>,
    pub beta: f64,
}

impl Kernel {
    pub fn n_free(&self) -> usize {
        self.n_out + self.n_hid
    }
    pub fn n_params(&self) -> usize {
        self.e_ff.len() + self.e_if.len() + self.n_free()
    }

    /// -beta * E restricted to terms involving free nodes, for free-state mask `m` (bit f set =>
    /// spin +1) given input spins `x`.
    #[inline]
    fn neg_beta_energy(&self, x: &[i8], m: usize) -> f64 {
        let sf = |f: usize| if m >> f & 1 == 1 { 1.0 } else { -1.0 };
        let mut acc = 0.0;
        for (k, &(a, b)) in self.e_ff.iter().enumerate() {
            acc += self.j_ff[k] * sf(a as usize) * sf(b as usize);
        }
        for (k, &(i, f)) in self.e_if.iter().enumerate() {
            acc += self.j_if[k] * x[i as usize] as f64 * sf(f as usize);
        }
        for f in 0..self.n_free() {
            acc += self.b[f] * sf(f);
        }
        self.beta * acc
    }

    /// Exact conditional over OUTPUT states (hidden marginalized): `p[o]` for `o` in `0..2^n_out`.
    pub fn exact_conditional(&self, x: &[i8]) -> Vec<f64> {
        let (no, nh) = (self.n_out, self.n_hid);
        let mut p = vec![0.0f64; 1 << no];
        // stabilize with max-shift
        let mut mx = f64::NEG_INFINITY;
        let n_free = 1usize << (no + nh);
        let mut nbe = vec![0.0f64; n_free];
        for m in 0..n_free {
            let v = self.neg_beta_energy(x, m);
            nbe[m] = v;
            if v > mx {
                mx = v;
            }
        }
        let mut z = 0.0;
        for m in 0..n_free {
            let w = (nbe[m] - mx).exp();
            p[m & ((1 << no) - 1)] += w;
            z += w;
        }
        for v in p.iter_mut() {
            *v /= z;
        }
        p
    }

    /// Exact KL( target(.|x) || P~(.|x) ) for a full target conditional (length 2^n_out).
    pub fn kl_from_target(&self, x: &[i8], target: &[f64]) -> f64 {
        let q = self.exact_conditional(x);
        let mut kl = 0.0;
        for o in 0..target.len() {
            if target[o] > 0.0 {
                kl += target[o] * (target[o].ln() - q[o].max(1e-300).ln());
            }
        }
        kl
    }

    /// Exact gradient of E_{y~target}[-log P~(y|x)] (cross-entropy; equals KL up to the target
    /// entropy constant). Positive phase: for each target-supported y, expectation over hidden
    /// given (x, y). Negative phase: expectation over all free. Accumulates into `grad`
    /// (layout: `[e_ff..][e_if..][b..]`), scaled by `weight`. Returns weighted cross-entropy.
    pub fn ce_grad(&self, x: &[i8], target: &[f64], weight: f64, grad: &mut [f64]) -> f64 {
        let (no, nh) = (self.n_out, self.n_hid);
        let n_free_states = 1usize << (no + nh);
        let out_mask = (1usize << no) - 1;

        // one pass: per-output-state log-sum over hidden, plus global expectations
        let mut nbe = vec![0.0f64; n_free_states];
        let mut mx = f64::NEG_INFINITY;
        for m in 0..n_free_states {
            let v = self.neg_beta_energy(x, m);
            nbe[m] = v;
            if v > mx {
                mx = v;
            }
        }
        let mut z = 0.0f64;
        let mut zy = vec![0.0f64; 1 << no];
        for m in 0..n_free_states {
            let w = (nbe[m] - mx).exp();
            z += w;
            zy[m & out_mask] += w;
        }

        let np = self.n_params();
        let mut pos = vec![0.0f64; np]; // sum over y of target[y] * E_{h|x,y}[suff stats]
        let mut neg = vec![0.0f64; np]; // E_{y,h|x}[suff stats]
        let sf = |m: usize, f: usize| if m >> f & 1 == 1 { 1.0 } else { -1.0 };
        for m in 0..n_free_states {
            let w = (nbe[m] - mx).exp();
            let o = m & out_mask;
            let w_pos = if zy[o] > 0.0 { target[o] * w / zy[o] } else { 0.0 };
            let w_neg = w / z;
            let mut idx = 0;
            for &(a, b) in &self.e_ff {
                let s = sf(m, a as usize) * sf(m, b as usize);
                pos[idx] += w_pos * s;
                neg[idx] += w_neg * s;
                idx += 1;
            }
            for &(i, f) in &self.e_if {
                let s = x[i as usize] as f64 * sf(m, f as usize);
                pos[idx] += w_pos * s;
                neg[idx] += w_neg * s;
                idx += 1;
            }
            for f in 0..no + nh {
                let s = sf(m, f);
                pos[idx] += w_pos * s;
                neg[idx] += w_neg * s;
                idx += 1;
            }
        }
        // d CE / d theta = -beta * (pos - neg); caller MINIMIZES CE, so grad += weight * that.
        for k in 0..np {
            grad[k] += weight * (-self.beta) * (pos[k] - neg[k]);
        }
        // weighted CE = -sum_y target[y] log p(y|x)
        let mut ce = 0.0;
        for o in 0..(1 << no) {
            if target[o] > 0.0 {
                ce -= target[o] * ((zy[o] / z).max(1e-300)).ln();
            }
        }
        weight * ce
    }

    /// Fold clamped-input contributions into effective free-node biases (once per input pattern).
    pub fn fold_bias(&self, x: &[i8]) -> Vec<f64> {
        let mut beff = self.b.clone();
        for (k, &(i, f)) in self.e_if.iter().enumerate() {
            beff[f as usize] += self.j_if[k] * x[i as usize] as f64;
        }
        beff
    }

    /// -beta*E over free terms only, with inputs pre-folded into `beff`.
    #[inline]
    fn nbe_folded(&self, beff: &[f64], m: usize) -> f64 {
        let sf = |f: usize| if m >> f & 1 == 1 { 1.0 } else { -1.0 };
        let mut acc = 0.0;
        for (k, &(a, b)) in self.e_ff.iter().enumerate() {
            acc += self.j_ff[k] * sf(a as usize) * sf(b as usize);
        }
        for f in 0..self.n_free() {
            acc += beff[f] * sf(f);
        }
        self.beta * acc
    }

    /// Exact NLL -log p(y*|x) for a one-hot target, via the folded fast path.
    pub fn nll_onehot(&self, x: &[i8], y_star: usize) -> f64 {
        let beff = self.fold_bias(x);
        let (no, nh) = (self.n_out, self.n_hid);
        let out_mask = (1usize << no) - 1;
        let mut mx = f64::NEG_INFINITY;
        let n_free = 1usize << (no + nh);
        let mut nbe = vec![0.0f64; n_free];
        for m in 0..n_free {
            let v = self.nbe_folded(&beff, m);
            nbe[m] = v;
            if v > mx {
                mx = v;
            }
        }
        let (mut z, mut zy) = (0.0f64, 0.0f64);
        for m in 0..n_free {
            let w = (nbe[m] - mx).exp();
            z += w;
            if m & out_mask == y_star {
                zy += w;
            }
        }
        -(zy / z).max(1e-300).ln()
    }

    /// Exact argmax_y p(y|x) — the idealized (noise-free) readout.
    pub fn argmax_out(&self, x: &[i8]) -> usize {
        let beff = self.fold_bias(x);
        let (no, nh) = (self.n_out, self.n_hid);
        let out_mask = (1usize << no) - 1;
        let n_free = 1usize << (no + nh);
        let mut mx = f64::NEG_INFINITY;
        let mut nbe = vec![0.0f64; n_free];
        for m in 0..n_free {
            let v = self.nbe_folded(&beff, m);
            nbe[m] = v;
            if v > mx {
                mx = v;
            }
        }
        let mut zy = vec![0.0f64; 1 << no];
        for m in 0..n_free {
            zy[m & out_mask] += (nbe[m] - mx).exp();
        }
        let mut best = 0;
        for o in 1..zy.len() {
            if zy[o] > zy[best] {
                best = o;
            }
        }
        best
    }

    /// Exact gradient of weight * (-log p(y*|x)) for a ONE-HOT target, optimized:
    /// positive phase enumerates hidden only (out clamped to y*); negative phase enumerates all
    /// free states; input-edge and bias gradients use per-node means, so only free-free edges pay
    /// a per-state pair cost. Returns the weighted NLL.
    pub fn ce_grad_onehot(&self, x: &[i8], y_star: usize, weight: f64, grad: &mut [f64]) -> f64 {
        let beff = self.fold_bias(x);
        let (no, nh) = (self.n_out, self.n_hid);
        let nf = no + nh;
        let out_mask = (1usize << no) - 1;

        // ---- negative phase: all free states ----
        let n_free = 1usize << nf;
        let mut nbe = vec![0.0f64; n_free];
        let mut mx = f64::NEG_INFINITY;
        for m in 0..n_free {
            let v = self.nbe_folded(&beff, m);
            nbe[m] = v;
            if v > mx {
                mx = v;
            }
        }
        let mut z = 0.0f64;
        let mut zy = 0.0f64;
        let mut mean_neg = vec![0.0f64; nf];
        let mut pair_neg = vec![0.0f64; self.e_ff.len()];
        let sf = |m: usize, f: usize| if m >> f & 1 == 1 { 1.0 } else { -1.0 };
        for m in 0..n_free {
            let w = (nbe[m] - mx).exp();
            z += w;
            if m & out_mask == y_star {
                zy += w;
            }
            for f in 0..nf {
                mean_neg[f] += w * sf(m, f);
            }
            for (k, &(a, b)) in self.e_ff.iter().enumerate() {
                pair_neg[k] += w * sf(m, a as usize) * sf(m, b as usize);
            }
        }
        for v in mean_neg.iter_mut() {
            *v /= z;
        }
        for v in pair_neg.iter_mut() {
            *v /= z;
        }

        // ---- positive phase: out clamped to y*, hidden enumerated ----
        let n_hid_states = 1usize << nh;
        let mut zp = 0.0f64;
        let mut mean_pos = vec![0.0f64; nf];
        let mut pair_pos = vec![0.0f64; self.e_ff.len()];
        let mut mxp = f64::NEG_INFINITY;
        let mut nbe_p = vec![0.0f64; n_hid_states];
        for hm in 0..n_hid_states {
            let m = y_star | (hm << no);
            let v = self.nbe_folded(&beff, m);
            nbe_p[hm] = v;
            if v > mxp {
                mxp = v;
            }
        }
        for hm in 0..n_hid_states {
            let m = y_star | (hm << no);
            let w = (nbe_p[hm] - mxp).exp();
            zp += w;
            for f in 0..nf {
                mean_pos[f] += w * sf(m, f);
            }
            for (k, &(a, b)) in self.e_ff.iter().enumerate() {
                pair_pos[k] += w * sf(m, a as usize) * sf(m, b as usize);
            }
        }
        for v in mean_pos.iter_mut() {
            *v /= zp;
        }
        for v in pair_pos.iter_mut() {
            *v /= zp;
        }

        // ---- accumulate: d NLL / d theta = -beta (pos - neg) on each sufficient statistic ----
        let mut idx = 0;
        for k in 0..self.e_ff.len() {
            grad[idx] += weight * (-self.beta) * (pair_pos[k] - pair_neg[k]);
            idx += 1;
        }
        for &(i, f) in &self.e_if {
            let xi = x[i as usize] as f64;
            grad[idx] += weight * (-self.beta) * xi * (mean_pos[f as usize] - mean_neg[f as usize]);
            idx += 1;
        }
        for f in 0..nf {
            grad[idx] += weight * (-self.beta) * (mean_pos[f] - mean_neg[f]);
            idx += 1;
        }
        weight * (-(zy / z).max(1e-300).ln())
    }

    pub fn apply_grad(&mut self, grad: &[f64], lr: f64) {
        let mut idx = 0;
        for k in 0..self.e_ff.len() {
            self.j_ff[k] -= lr * grad[idx];
            idx += 1;
        }
        for k in 0..self.e_if.len() {
            self.j_if[k] -= lr * grad[idx];
            idx += 1;
        }
        for f in 0..self.n_free() {
            self.b[f] -= lr * grad[idx];
            idx += 1;
        }
    }

    /// Draw output bits the way the DEVICE would: clamp inputs, run `sweeps` chromatic Gibbs
    /// sweeps over the free nodes, read the outputs. Charges the ledger (samples + reads; the
    /// caller decides how clamping is billed, since that price is unpublished).
    pub fn sample(
        &self,
        x: &[i8],
        sweeps: usize,
        rng: &mut Pcg,
        ledger: Option<&mut Ledger>,
    ) -> Vec<i8> {
        // free-node Gibbs with input contributions folded into effective biases
        let nf = self.n_free();
        let mut beff = self.b.clone();
        for (k, &(i, f)) in self.e_if.iter().enumerate() {
            beff[f as usize] += self.j_if[k] * x[i as usize] as f64;
        }
        // adjacency among free nodes
        let mut adj: Vec<Vec<(usize, f64)>> = vec![Vec::new(); nf];
        for (k, &(a, b)) in self.e_ff.iter().enumerate() {
            adj[a as usize].push((b as usize, self.j_ff[k]));
            adj[b as usize].push((a as usize, self.j_ff[k]));
        }
        let mut s: Vec<i8> = (0..nf).map(|_| rng.spin(0.5)).collect();
        let mut count = 0u64;
        for _ in 0..sweeps {
            for i in 0..nf {
                let mut f = beff[i];
                for &(j, w) in &adj[i] {
                    f += w * s[j] as f64;
                }
                let p_up = crate::kernel::p_up(f, self.beta);
                s[i] = rng.spin(p_up);
                count += 1;
            }
        }
        if let Some(l) = ledger {
            l.samples += count;
            l.reads += self.n_out as u64;
        }
        s[..self.n_out].to_vec()
    }
}

/// Build a kernel on a `w x h` patch of the Z1-class device graph. Roles are assigned by site
/// index from the caller-provided role map; only native device edges survive.
/// role codes: 0 = off (site unused), 1 = input, 2 = output, 3 = hidden.
pub fn patch_kernel(w: usize, h: usize, roles: &[u8], beta: f64, seed: u64) -> Kernel {
    assert_eq!(roles.len(), w * h);
    let g: Graph = crate::device::z1_grid(w, h, 1.0, 0.0); // topology only; weights re-learned
    let mut in_map = vec![u16::MAX; w * h];
    let mut free_map = vec![u16::MAX; w * h];
    let (mut n_in, mut n_out, mut n_hid) = (0usize, 0usize, 0usize);
    // outputs first in the free vector, then hidden
    for (i, &r) in roles.iter().enumerate() {
        if r == 2 {
            free_map[i] = n_out as u16;
            n_out += 1;
        }
    }
    for (i, &r) in roles.iter().enumerate() {
        if r == 3 {
            free_map[i] = (n_out + n_hid) as u16;
            n_hid += 1;
        }
    }
    for (i, &r) in roles.iter().enumerate() {
        if r == 1 {
            in_map[i] = n_in as u16;
            n_in += 1;
        }
    }
    let mut e_ff = Vec::new();
    let mut e_if = Vec::new();
    for a in 0..g.n {
        for k in g.offset[a]..g.offset[a + 1] {
            let b = g.nbr[k] as usize;
            if b <= a {
                continue;
            }
            let (ra, rb) = (roles[a], roles[b]);
            if ra == 0 || rb == 0 {
                continue;
            }
            match (ra == 1, rb == 1) {
                (true, true) => {} // input-input: constant under clamping, dropped
                (true, false) => e_if.push((in_map[a], free_map[b])),
                (false, true) => e_if.push((in_map[b], free_map[a])),
                (false, false) => e_ff.push((free_map[a].min(free_map[b]), free_map[a].max(free_map[b]))),
            }
        }
    }
    let mut rng = Pcg::new(seed, 0xC0);
    let s = 0.1;
    let nf = n_out + n_hid;
    Kernel {
        n_in,
        n_out,
        n_hid,
        j_ff: (0..e_ff.len()).map(|_| (rng.f64() - 0.5) * s).collect(),
        j_if: (0..e_if.len()).map(|_| (rng.f64() - 0.5) * s).collect(),
        e_ff,
        e_if,
        b: vec![0.0; nf],
        beta,
    }
}

/// Fully-enumerable target conditional: a table `target[x_mask][y_mask]`.
pub type Cpt = Vec<Vec<f64>>;

/// Fit a kernel to a target CPT under input distribution `mu` (weights over x masks).
/// Full-batch exact gradient descent; returns the CE trajectory (first, last).
pub fn fit(
    kernel: &mut Kernel,
    target: &Cpt,
    mu: &[f64],
    iters: usize,
    lr: f64,
) -> (f64, f64) {
    let n_in = kernel.n_in;
    let mut first = 0.0;
    let mut last = 0.0;
    let xs: Vec<(usize, f64)> = mu
        .iter()
        .enumerate()
        .filter(|(_, &w)| w > 0.0)
        .map(|(m, &w)| (m, w))
        .collect();
    for it in 0..iters {
        let mut grad = vec![0.0; kernel.n_params()];
        let mut ce = 0.0;
        for &(xm, wx) in &xs {
            let x: Vec<i8> = (0..n_in).map(|b| if xm >> b & 1 == 1 { 1 } else { -1 }).collect();
            ce += kernel.ce_grad(&x, &target[xm], wx, &mut grad);
        }
        kernel.apply_grad(&grad, lr);
        if it == 0 {
            first = ce;
        }
        last = ce;
    }
    (first, last)
}

/// Mean KL over mu — the per-factor epsilon of the compilation bound.
pub fn factor_eps(kernel: &Kernel, target: &Cpt, mu: &[f64]) -> f64 {
    let n_in = kernel.n_in;
    let mut eps = 0.0;
    for (xm, &wx) in mu.iter().enumerate() {
        if wx <= 0.0 {
            continue;
        }
        let x: Vec<i8> = (0..n_in).map(|b| if xm >> b & 1 == 1 { 1 } else { -1 }).collect();
        eps += wx * kernel.kl_from_target(&x, &target[xm]);
    }
    eps
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_kernel() -> Kernel {
        // 2 in, 2 out, 2 hid on a 3x2 patch, all sites used
        patch_kernel(3, 2, &[1, 3, 2, 1, 3, 2], 1.0, 5)
    }

    #[test]
    fn conditional_normalizes() {
        let k = tiny_kernel();
        let p = k.exact_conditional(&[1, -1]);
        let s: f64 = p.iter().sum();
        assert!((s - 1.0).abs() < 1e-12);
    }

    /// Exact gradient must match finite differences of the exact CE.
    #[test]
    fn grad_matches_fd() {
        let mut k = tiny_kernel();
        let x = [1i8, -1];
        let target = vec![0.6, 0.1, 0.1, 0.2]; // over 2 out bits
        let mut grad = vec![0.0; k.n_params()];
        let _ = k.ce_grad(&x, &target, 1.0, &mut grad);
        // check a coupling and a bias by FD
        for &pi in &[0usize, k.e_ff.len() + 1, k.n_params() - 1] {
            let d = 1e-5;
            let bump = |kk: &mut Kernel, idx: usize, by: f64| {
                if idx < kk.e_ff.len() {
                    kk.j_ff[idx] += by;
                } else if idx < kk.e_ff.len() + kk.e_if.len() {
                    kk.j_if[idx - kk.e_ff.len()] += by;
                } else {
                    kk.b[idx - kk.e_ff.len() - kk.e_if.len()] += by;
                }
            };
            let ce_at = |kk: &Kernel| {
                let q = kk.exact_conditional(&x);
                -target.iter().zip(&q).map(|(t, p)| if *t > 0.0 { t * p.max(1e-300).ln() } else { 0.0 }).sum::<f64>()
            };
            bump(&mut k, pi, d);
            let up = ce_at(&k);
            bump(&mut k, pi, -2.0 * d);
            let dn = ce_at(&k);
            bump(&mut k, pi, d);
            let fd = (up - dn) / (2.0 * d);
            assert!((grad[pi] - fd).abs() < 1e-6, "param {pi}: exact {} fd {}", grad[pi], fd);
        }
    }

    /// The optimized one-hot path must agree with the general path exactly.
    #[test]
    fn onehot_matches_general() {
        let k = tiny_kernel();
        let x = [1i8, -1];
        let y_star = 2usize;
        let mut onehot = vec![0.0; 4];
        onehot[y_star] = 1.0;
        let mut g1 = vec![0.0; k.n_params()];
        let ce1 = k.ce_grad(&x, &onehot, 1.0, &mut g1);
        let mut g2 = vec![0.0; k.n_params()];
        let ce2 = k.ce_grad_onehot(&x, y_star, 1.0, &mut g2);
        assert!((ce1 - ce2).abs() < 1e-10, "CE {ce1} vs {ce2}");
        for j in 0..g1.len() {
            assert!((g1[j] - g2[j]).abs() < 1e-10, "grad[{j}] {} vs {}", g1[j], g2[j]);
        }
        assert!((k.nll_onehot(&x, y_star) - ce2).abs() < 1e-10);
        let _ = k.argmax_out(&x);
    }

    /// Fitting must drive KL down on a representable target.
    #[test]
    fn fit_reduces_kl() {
        let mut k = tiny_kernel();
        // target: y = copy of x (2 bits), softened
        let mut cpt: Cpt = vec![vec![0.0; 4]; 4];
        for xm in 0..4 {
            for ym in 0..4 {
                cpt[xm][ym] = if ym == xm { 0.85 } else { 0.05 };
            }
        }
        let mu = vec![0.25; 4];
        let e0 = factor_eps(&k, &cpt, &mu);
        fit(&mut k, &cpt, &mu, 400, 0.15);
        let e1 = factor_eps(&k, &cpt, &mu);
        assert!(e1 < 0.2 * e0, "eps {e0} -> {e1}");
    }
}
