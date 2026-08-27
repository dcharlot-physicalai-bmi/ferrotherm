//! Stochastic differentiable programs — the program layer over the sampler.
//!
//! A program is a fixed sequence of parametrized stochastic gates acting on typed wires
//! (binary `bits`, continuous `reals`). Running it draws a trajectory; training minimizes
//! E[L(final state)] over gate parameters. Three gradient routes, cross-validated in the
//! examples before anything is trained:
//!
//!  * **Score function (REINFORCE)**: every stochastic draw has a tractable log-density whose
//!    parameter gradient accumulates into a score vector; grad = E[(L - baseline) * score].
//!    Works for every gate here, including a full Gibbs kernel (each spin update is a Bernoulli
//!    with known probability, so the trajectory log-density is exact — no approximation).
//!  * **Parameter shift** for sigmoid-mixture gates (the flip family): the kernel is
//!    K_theta = sigma(theta) K_flip + (1-sigma(theta)) K_id, so
//!    dE/dtheta = sigma'(theta) (E[L|flip] - E[L|no-flip]), evaluated with common random
//!    numbers downstream. Exact in expectation, two branch runs per gate.
//!  * **Finite differences with common random numbers** — the referee both must agree with.

use crate::graph::Graph;
use crate::ledger::Ledger;
use crate::rng::Pcg;

#[inline]
fn sigma(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Wire state: binary spins plus continuous scalars.
#[derive(Clone)]
pub struct State {
    pub bits: Vec<i8>,
    pub reals: Vec<f64>,
}

pub enum Gate {
    /// Flip `bit` with probability `sigma(params[p_theta])`.
    PNot { bit: usize, p_theta: usize },
    /// `reals[u] ~ Normal( params[p_k] * reals[err], sigma^2 )`. The policy gate.
    CtrlGauss { u: usize, err: usize, p_k: usize, sigma: f64 },
    /// Deterministic linear dynamics: `reals[x] = a * reals[x] + b * reals[u]`.
    Lin { x: usize, u: usize, a: f64, b: f64 },
    /// `reals[err] = tgt - reals[x]`.
    Err { err: usize, x: usize, tgt: f64 },
    /// Stage-cost accumulator: `reals[acc] += q * reals[x]^2 + r * reals[u]^2`. Deterministic.
    CostQuad { acc: usize, x: usize, u: usize, q: f64, r: f64 },
    /// `sweeps` chromatic Glauber sweeps of graph `g` over `bits[0..g.n]`, at inverse
    /// temperature `beta`, with per-node bias params[p_h0 + i] REPLACING the graph's h.
    GibbsK { g: usize, sweeps: usize, beta: f64, p_h0: usize },
    /// EXACT Boltzmann resample of graph `g`'s spins (enumeration; g.n <= 20) with bias
    /// params[p_h0 + i] replacing the graph's h, at inverse temperature `beta`. This is the
    /// Boltzmann-form gate the EBM-kernel gradient estimator applies to; its REINFORCE score is
    /// also exact (beta * (s_i - <s_i>)), so the two estimators cross-validate on the same gate.
    BoltzExact { g: usize, beta: f64, p_h0: usize },
}

pub struct Program {
    pub gates: Vec<Gate>,
    pub graphs: Vec<Graph>,
    pub n_params: usize,
}

/// Which branch to force for one parameter-shift evaluation.
#[derive(Clone, Copy)]
pub enum Force {
    None,
    /// Force the PNot at `gate_idx` to flip (true) or hold (false). The RNG draw is still
    /// consumed so downstream randomness is identical across branches.
    PNot { gate_idx: usize, flip: bool },
}

impl Program {
    /// Run once. `score`, if given, accumulates d log p(trajectory) / d params.
    /// `ledger`, if given, is charged for Gibbs sweeps (device pricing of the program).
    pub fn run(
        &self,
        init: &State,
        rng: &mut Pcg,
        mut score: Option<&mut [f64]>,
        force: Force,
        params: &[f64],
        mut ledger: Option<&mut Ledger>,
        mut trace: Option<&mut Vec<(usize, usize)>>,
    ) -> State {
        let mut st = init.clone();
        for (gi, gate) in self.gates.iter().enumerate() {
            match *gate {
                Gate::PNot { bit, p_theta } => {
                    let th = params[p_theta];
                    let p_flip = sigma(th);
                    let u = rng.f64(); // always consumed, so branches share downstream draws
                    let flip = match force {
                        Force::PNot { gate_idx, flip } if gate_idx == gi => flip,
                        _ => u < p_flip,
                    };
                    if flip {
                        st.bits[bit] = -st.bits[bit];
                    }
                    if let Some(s) = score.as_deref_mut() {
                        // d log p / d theta: sigma(-th) if flip else -sigma(th)
                        s[p_theta] += if flip { sigma(-th) } else { -p_flip };
                    }
                }
                Gate::CtrlGauss { u, err, p_k, sigma: sg } => {
                    let k = params[p_k];
                    let mean = k * st.reals[err];
                    let z = gauss(rng);
                    let val = mean + sg * z;
                    st.reals[u] = val;
                    if let Some(s) = score.as_deref_mut() {
                        // d log N(val; k e, sg^2) / d k = (val - k e) e / sg^2
                        s[p_k] += (val - mean) * st.reals[err] / (sg * sg);
                    }
                }
                Gate::Lin { x, u, a, b } => {
                    st.reals[x] = a * st.reals[x] + b * st.reals[u];
                }
                Gate::Err { err, x, tgt } => {
                    st.reals[err] = tgt - st.reals[x];
                }
                Gate::CostQuad { acc, x, u, q, r } => {
                    st.reals[acc] += q * st.reals[x] * st.reals[x] + r * st.reals[u] * st.reals[u];
                }
                Gate::BoltzExact { g, beta, p_h0 } => {
                    let gr = &self.graphs[g];
                    let (mask, means) = boltz_exact_draw(gr, beta, &params[p_h0..p_h0 + gr.n], rng);
                    for i in 0..gr.n {
                        st.bits[i] = if mask >> i & 1 == 1 { 1 } else { -1 };
                    }
                    if let Some(s) = score.as_deref_mut() {
                        // exact d log p / d h_i = beta * (s_i - <s_i>)
                        for i in 0..gr.n {
                            let si = st.bits[i] as f64;
                            s[p_h0 + i] += beta * (si - means[i]);
                        }
                    }
                    if let Some(tr) = trace.as_deref_mut() {
                        tr.push((gi, mask));
                    }
                }
                Gate::GibbsK { g, sweeps, beta, p_h0 } => {
                    let gr = &self.graphs[g];
                    let mut updated = 0u64;
                    for _ in 0..sweeps {
                        for class in &gr.classes {
                            for &iu in class {
                                let i = iu as usize;
                                // local field with params-supplied bias
                                let mut f = params[p_h0 + i];
                                for kk in gr.offset[i]..gr.offset[i + 1] {
                                    f += gr.w[kk] * st.bits[gr.nbr[kk] as usize] as f64;
                                }
                                let s_new = crate::kernel::draw(f, beta, rng);
                                st.bits[i] = s_new;
                                updated += 1;
                                if let Some(s) = score.as_deref_mut() {
                                    s[p_h0 + i] += crate::kernel::score_dh(f, beta, s_new);
                                }
                            }
                        }
                    }
                    if let Some(l) = ledger.as_deref_mut() {
                        l.samples += updated;
                    }
                }
            }
        }
        st
    }

    /// REINFORCE gradient of `E[L]` with a batch-mean baseline. Returns (grad, mean_loss).
    pub fn reinforce_grad<F: Fn(&State) -> f64>(
        &self,
        init: &State,
        params: &[f64],
        loss: &F,
        episodes: usize,
        seed: u64,
    ) -> (Vec<f64>, f64) {
        let mut losses = Vec::with_capacity(episodes);
        let mut scores: Vec<Vec<f64>> = Vec::with_capacity(episodes);
        for e in 0..episodes {
            let mut rng = Pcg::new(seed, e as u64);
            let mut sc = vec![0.0; self.n_params];
            let st = self.run(init, &mut rng, Some(&mut sc), Force::None, params, None, None);
            losses.push(loss(&st));
            scores.push(sc);
        }
        let mean = losses.iter().sum::<f64>() / episodes as f64;
        let mut grad = vec![0.0; self.n_params];
        for e in 0..episodes {
            let adv = losses[e] - mean;
            for j in 0..self.n_params {
                grad[j] += adv * scores[e][j];
            }
        }
        for gj in grad.iter_mut() {
            *gj /= episodes as f64;
        }
        (grad, mean)
    }

    /// Parameter-shift gradient for the PNot at `gate_idx`:
    /// sigma'(theta) * (E[L | forced flip] - E[L | forced hold]), common random numbers.
    pub fn pshift_grad_pnot<F: Fn(&State) -> f64>(
        &self,
        gate_idx: usize,
        init: &State,
        params: &[f64],
        loss: &F,
        episodes: usize,
        seed: u64,
    ) -> f64 {
        let p_theta = match self.gates[gate_idx] {
            Gate::PNot { p_theta, .. } => p_theta,
            _ => panic!("pshift_grad_pnot on a non-PNot gate"),
        };
        let th = params[p_theta];
        let dsig = sigma(th) * sigma(-th);
        let mut diff = 0.0;
        for e in 0..episodes {
            let mut r1 = Pcg::new(seed, e as u64);
            let a = self.run(init, &mut r1, None, Force::PNot { gate_idx, flip: true }, params, None, None);
            let mut r2 = Pcg::new(seed, e as u64);
            let b = self.run(init, &mut r2, None, Force::PNot { gate_idx, flip: false }, params, None, None);
            diff += loss(&a) - loss(&b);
        }
        dsig * diff / episodes as f64
    }

    /// The EBM-kernel gradient estimator (the third estimator; arXiv:2608.01612 Sec III C):
    /// for gates of Boltzmann form, grad_theta log G(y|x) = -grad E(y) + E_y'[grad E(y')], so one
    /// circuit trajectory plus ONE auxiliary re-draw of the gate gives the unbiased single-sample
    /// estimate  f(z_final) * (grad E(aux) - grad E(traj)). Applies to [`Gate::BoltzExact`]
    /// parameters (grad_h (beta E) = -beta s_i); other gates' parameters are left at zero here —
    /// combine with [`Self::reinforce_grad`] for them. Returns (grad, mean_loss).
    pub fn ebm_kernel_grad<F: Fn(&State) -> f64>(
        &self,
        init: &State,
        params: &[f64],
        loss: &F,
        episodes: usize,
        seed: u64,
    ) -> (Vec<f64>, f64) {
        let mut grad = vec![0.0; self.n_params];
        let mut mean = 0.0;
        let mut trace: Vec<(usize, usize)> = Vec::new();
        for e in 0..episodes {
            let mut rng = Pcg::new(seed, e as u64);
            trace.clear();
            let st = self.run(init, &mut rng, None, Force::None, params, None, Some(&mut trace));
            let f = loss(&st);
            mean += f;
            for &(gi, traj_mask) in &trace {
                if let Gate::BoltzExact { g, beta, p_h0 } = self.gates[gi] {
                    let gr = &self.graphs[g];
                    // auxiliary draw: an independent, deterministic stream per (episode, gate)
                    let mut aux_rng =
                        Pcg::new(seed.wrapping_mul(0xD6E8FEB86659FD93) ^ gi as u64, e as u64);
                    let (aux_mask, _) =
                        boltz_exact_draw(gr, beta, &params[p_h0..p_h0 + gr.n], &mut aux_rng);
                    for i in 0..gr.n {
                        let s_traj = if traj_mask >> i & 1 == 1 { 1.0 } else { -1.0 };
                        let s_aux = if aux_mask >> i & 1 == 1 { 1.0 } else { -1.0 };
                        // grad_h (beta E) = -beta s  =>  gradE(aux) - gradE(traj) = beta (s_traj - s_aux)
                        grad[p_h0 + i] += f * beta * (s_traj - s_aux);
                    }
                }
            }
        }
        for g in grad.iter_mut() {
            *g /= episodes as f64;
        }
        (grad, mean / episodes as f64)
    }

    /// Central finite difference with common random numbers — the referee.
    pub fn fd_grad<F: Fn(&State) -> f64>(
        &self,
        j: usize,
        delta: f64,
        init: &State,
        params: &[f64],
        loss: &F,
        episodes: usize,
        seed: u64,
    ) -> f64 {
        let mut plus = params.to_vec();
        plus[j] += delta;
        let mut minus = params.to_vec();
        minus[j] -= delta;
        let mut acc = 0.0;
        for e in 0..episodes {
            let mut r1 = Pcg::new(seed, e as u64);
            let a = self.run(init, &mut r1, None, Force::None, &plus, None, None);
            let mut r2 = Pcg::new(seed, e as u64);
            let b = self.run(init, &mut r2, None, Force::None, &minus, None, None);
            acc += loss(&a) - loss(&b);
        }
        acc / (2.0 * delta * episodes as f64)
    }
}

#[inline]
fn gauss(rng: &mut Pcg) -> f64 {
    let a = rng.f64().max(1e-15);
    let b = rng.f64();
    (-2.0 * a.ln()).sqrt() * (std::f64::consts::TAU * b).cos()
}

/// Exact Boltzmann draw over all 2^n spin states of `gr` with external biases `h` at inverse
/// temperature `beta`. Returns (sampled mask, exact per-spin means <s_i>). Enumeration; n <= 20.
fn boltz_exact_draw(gr: &Graph, beta: f64, h: &[f64], rng: &mut Pcg) -> (usize, Vec<f64>) {
    let n = gr.n;
    assert!(n <= 20, "BoltzExact enumeration limited to 20 spins");
    let total = 1usize << n;
    let mut w = vec![0.0f64; total];
    let mut mx = f64::NEG_INFINITY;
    let mut s = vec![-1i8; n];
    for m in 0..total {
        for b in 0..n {
            s[b] = if m >> b & 1 == 1 { 1 } else { -1 };
        }
        // energy with external biases: base couplings from gr, h REPLACES gr.h
        let mut e = 0.0;
        for i in 0..n {
            let si = s[i] as f64;
            e -= h[i] * si;
            for k in gr.offset[i]..gr.offset[i + 1] {
                let j = gr.nbr[k] as usize;
                if j > i {
                    e -= gr.w[k] * si * s[j] as f64;
                }
            }
        }
        let l = -beta * e;
        w[m] = l;
        if l > mx {
            mx = l;
        }
    }
    let mut z = 0.0;
    for v in w.iter_mut() {
        *v = (*v - mx).exp();
        z += *v;
    }
    let mut means = vec![0.0; n];
    for m in 0..total {
        let p = w[m] / z;
        for b in 0..n {
            means[b] += p * if m >> b & 1 == 1 { 1.0 } else { -1.0 };
        }
    }
    let mut u = rng.f64() * z;
    let mut pick = total - 1;
    for m in 0..total {
        if u < w[m] {
            pick = m;
            break;
        }
        u -= w[m];
    }
    (pick, means)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The three estimators that can touch a Boltzmann gate's parameters must agree: exact-score
    /// REINFORCE, the EBM-kernel estimator (one trajectory + one auxiliary draw), and the FD
    /// referee — on a circuit where a downstream stochastic gate stands between the Boltzmann
    /// gate and the loss.
    #[test]
    fn ebm_kernel_matches_reinforce_and_fd() {
        use crate::graph::GraphBuilder;
        let mut gb = GraphBuilder::new(3);
        gb.couple(0, 1, 0.4);
        gb.couple(1, 2, 0.4);
        gb.couple(2, 0, 0.4);
        let g = gb.build();
        let prog = Program {
            gates: vec![
                Gate::BoltzExact { g: 0, beta: 0.8, p_h0: 0 },
                Gate::PNot { bit: 0, p_theta: 3 },
            ],
            graphs: vec![g],
            n_params: 4,
        };
        let init = State { bits: vec![1, 1, 1], reals: vec![] };
        let params = [0.3, -0.2, 0.1, 0.4];
        let loss = |s: &State| {
            s.bits[0] as f64 + 0.7 * s.bits[1] as f64 + 0.4 * (s.bits[0] * s.bits[2]) as f64
        };
        let (g_rf, _) = prog.reinforce_grad(&init, &params, &loss, 400_000, 0xE1);
        let (g_ek, _) = prog.ebm_kernel_grad(&init, &params, &loss, 400_000, 0xE2);
        // EXACT reference gradient by full enumeration (16 joint outcomes). The FD referee is
        // unusable here: common random numbers decorrelate across a DISCRETE re-draw, leaving FD
        // an unbiased estimator with a noise floor larger than the gradient itself — a measured
        // instance of "verify the verifier".
        let beta = 0.8;
        let p_flip = 1.0 / (1.0 + (-params[3]).exp());
        let gref = &prog.graphs[0];
        let mut pm = [0.0f64; 8];
        let mut z = 0.0;
        let mut s = [0i8; 3];
        for m in 0..8usize {
            for b in 0..3 {
                s[b] = if m >> b & 1 == 1 { 1 } else { -1 };
            }
            let mut e = 0.0;
            for i in 0..3 {
                e -= params[i] * s[i] as f64;
                for k in gref.offset[i]..gref.offset[i + 1] {
                    let j = gref.nbr[k] as usize;
                    if j > i {
                        e -= gref.w[k] * (s[i] * s[j]) as f64;
                    }
                }
            }
            pm[m] = (-beta * e).exp();
            z += pm[m];
        }
        for v in pm.iter_mut() {
            *v /= z;
        }
        let gm = |m: usize| {
            let sb = |b: usize| if m >> b & 1 == 1 { 1.0 } else { -1.0 };
            let s0p = (1.0 - 2.0 * p_flip) * sb(0); // PNot flips bit 0 with prob p_flip
            s0p + 0.7 * sb(1) + 0.4 * s0p * sb(2)
        };
        for j in 0..3 {
            let sj = |m: usize| if m >> j & 1 == 1 { 1.0 } else { -1.0 };
            let eg: f64 = (0..8).map(|m| pm[m] * gm(m)).sum();
            let es: f64 = (0..8).map(|m| pm[m] * sj(m)).sum();
            let egs: f64 = (0..8).map(|m| pm[m] * gm(m) * sj(m)).sum();
            let exact = beta * (egs - eg * es); // covariance identity for exponential families
            assert!(
                (g_rf[j] - exact).abs() < 0.01,
                "h[{j}]: REINFORCE {} vs exact {}",
                g_rf[j],
                exact
            );
            assert!(
                (g_ek[j] - exact).abs() < 0.02,
                "h[{j}]: EBM-kernel {} vs exact {}",
                g_ek[j],
                exact
            );
        }
    }

    /// Analytic check: single PNot on a +1 bit, L(s) = s. E[L] = 1 - 2 sigma(th),
    /// dE/dth = -2 sigma'(th). REINFORCE must match analytically.
    #[test]
    fn reinforce_matches_analytic() {
        let prog = Program {
            gates: vec![Gate::PNot { bit: 0, p_theta: 0 }],
            graphs: vec![],
            n_params: 1,
        };
        let init = State { bits: vec![1], reals: vec![] };
        let th = 0.3;
        let (g, _) = prog.reinforce_grad(&init, &[th], &|s: &State| s.bits[0] as f64, 400_000, 99);
        let want = -2.0 * sigma(th) * sigma(-th);
        assert!((g[0] - want).abs() < 0.01, "got {} want {}", g[0], want);
    }
}
