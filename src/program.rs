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
    /// Flip `bit` with probability sigma(params[p_theta]).
    PNot { bit: usize, p_theta: usize },
    /// reals[u] ~ Normal( params[p_k] * reals[err], sigma^2 ). The policy gate.
    CtrlGauss { u: usize, err: usize, p_k: usize, sigma: f64 },
    /// Deterministic linear dynamics: reals[x] = a * reals[x] + b * reals[u].
    Lin { x: usize, u: usize, a: f64, b: f64 },
    /// reals[err] = tgt - reals[x].
    Err { err: usize, x: usize, tgt: f64 },
    /// Stage-cost accumulator: reals[acc] += q * reals[x]^2 + r * reals[u]^2. Deterministic.
    CostQuad { acc: usize, x: usize, u: usize, q: f64, r: f64 },
    /// `sweeps` chromatic Glauber sweeps of graph `g` over `bits[0..g.n]`, at inverse
    /// temperature `beta`, with per-node bias params[p_h0 + i] REPLACING the graph's h.
    GibbsK { g: usize, sweeps: usize, beta: f64, p_h0: usize },
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
                                let arg = 2.0 * beta * f;
                                let p_up = sigma(arg);
                                let s_new: i8 = if rng.f64() < p_up { 1 } else { -1 };
                                st.bits[i] = s_new;
                                updated += 1;
                                if let Some(s) = score.as_deref_mut() {
                                    // log p(s') = log sigma(2 beta f s') ;
                                    // d/dh_i = 2 beta s' sigma(-2 beta f s')
                                    let sp = s_new as f64;
                                    s[p_h0 + i] += 2.0 * beta * sp * sigma(-arg * sp);
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

    /// REINFORCE gradient of E[L] with a batch-mean baseline. Returns (grad, mean_loss).
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
            let st = self.run(init, &mut rng, Some(&mut sc), Force::None, params, None);
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
            let a = self.run(init, &mut r1, None, Force::PNot { gate_idx, flip: true }, params, None);
            let mut r2 = Pcg::new(seed, e as u64);
            let b = self.run(init, &mut r2, None, Force::PNot { gate_idx, flip: false }, params, None);
            diff += loss(&a) - loss(&b);
        }
        dsig * diff / episodes as f64
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
            let a = self.run(init, &mut r1, None, Force::None, &plus, None);
            let mut r2 = Pcg::new(seed, e as u64);
            let b = self.run(init, &mut r2, None, Force::None, &minus, None);
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

#[cfg(test)]
mod tests {
    use super::*;

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
