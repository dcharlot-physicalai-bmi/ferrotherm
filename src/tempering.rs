//! Optimization-grade sampling: simulated annealing and parallel tempering.
//!
//! This is the p-computer algorithm line (Camsari and collaborators): the same chromatic Gibbs
//! primitive, scheduled. Parallel tempering runs replicas at a ladder of temperatures and swaps
//! neighbouring replicas with the Metropolis criterion, which is what lets hard, frustrated
//! landscapes mix — FPGA p-computers running adaptive parallel tempering matched a quantum
//! annealer on 3D spin glasses (Nature Communications 2025, arXiv:2503.10302). The verification
//! standard here is exact: on enumerable instances the sampler must find the true ground state.

use crate::gibbs::Sampler;
use crate::graph::Graph;
use crate::ledger::Ledger;
use crate::rng::Pcg;

/// Simulated annealing: sweep while raising beta along `schedule`, tracking the best state seen.
pub fn anneal(
    g: &Graph,
    schedule: &[(f64, usize)], // (beta, sweeps at that beta)
    seed: u64,
    mut ledger: Option<&mut Ledger>,
) -> (Vec<i8>, f64) {
    let mut smp = Sampler::new(g, schedule[0].0, seed);
    let mut best = smp.s.clone();
    let mut best_e = g.energy(&best);
    for &(beta, sweeps) in schedule {
        smp.beta = beta;
        for _ in 0..sweeps {
            smp.sweep(ledger.as_deref_mut());
            let e = g.energy(&smp.s);
            if e < best_e {
                best_e = e;
                best = smp.s.clone();
            }
        }
    }
    (best, best_e)
}

/// Geometric beta ladder from `beta_min` to `beta_max`.
pub fn geometric_ladder(beta_min: f64, beta_max: f64, n: usize) -> Vec<f64> {
    assert!(n >= 2 && beta_min > 0.0 && beta_max > beta_min);
    let r = (beta_max / beta_min).powf(1.0 / (n - 1) as f64);
    (0..n).map(|i| beta_min * r.powi(i as i32)).collect()
}

pub struct TemperingResult {
    pub best: Vec<i8>,
    pub best_e: f64,
    /// Swap acceptance rate per adjacent pair — the ladder-health diagnostic. Healthy ladders sit
    /// roughly in [0.2, 0.6]; near-zero pairs mean the ladder has a gap replicas cannot cross.
    pub swap_rates: Vec<f64>,
}

/// Parallel tempering over a beta ladder. Every `swap_every` sweeps, adjacent replicas attempt a
/// state exchange with probability min(1, exp(delta_beta * delta_E)) — the standard replica-
/// exchange criterion, alternating even/odd pairs so a state can traverse the whole ladder.
pub fn parallel_tempering(
    g: &Graph,
    betas: &[f64],
    rounds: usize,
    swap_every: usize,
    seed: u64,
    mut ledger: Option<&mut Ledger>,
) -> TemperingResult {
    let r = betas.len();
    assert!(r >= 2);
    let mut reps: Vec<Sampler> = (0..r).map(|i| Sampler::new(g, betas[i], seed ^ (i as u64 * 0x9E37)) ).collect();
    let mut swap_rng = Pcg::new(seed ^ 0x5A5A, 3);
    let mut attempts = vec![0u64; r - 1];
    let mut accepts = vec![0u64; r - 1];
    let mut best = reps[r - 1].s.clone();
    let mut best_e = g.energy(&best);
    for round in 0..rounds {
        for rep in reps.iter_mut() {
            for _ in 0..swap_every {
                rep.sweep(ledger.as_deref_mut());
            }
        }
        // coldest replica is the optimizer; track its best
        for rep in reps.iter() {
            let e = g.energy(&rep.s);
            if e < best_e {
                best_e = e;
                best = rep.s.clone();
            }
        }
        // alternate even/odd adjacent pairs
        let start = round % 2;
        for i in (start..r - 1).step_by(2) {
            let e_i = g.energy(&reps[i].s);
            let e_j = g.energy(&reps[i + 1].s);
            let arg = (betas[i + 1] - betas[i]) * (e_j - e_i);
            attempts[i] += 1;
            if arg >= 0.0 || swap_rng.f64() < arg.exp() {
                accepts[i] += 1;
                let (a, b) = reps.split_at_mut(i + 1);
                std::mem::swap(&mut a[i].s, &mut b[0].s);
            }
        }
    }
    TemperingResult {
        best,
        best_e,
        swap_rates: (0..r - 1).map(|i| accepts[i] as f64 / attempts[i].max(1) as f64).collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::GraphBuilder;

    /// Random frustrated glass on 16 spins: the exact ground state is enumerable, and parallel
    /// tempering must find it. Plain low-temperature Gibbs is NOT required to (it can trap),
    /// which is the point of the ladder.
    #[test]
    fn tempering_finds_exact_ground_state() {
        let n = 16usize;
        let mut rng = Pcg::new(0x61A55, 5);
        let mut gb = GraphBuilder::new(n);
        // dense-ish random +-J glass
        for i in 0..n {
            for j in (i + 1)..n {
                if rng.f64() < 0.5 {
                    gb.couple(i, j, if rng.f64() < 0.5 { 1.0 } else { -1.0 });
                }
            }
        }
        let g = gb.build();
        // exact ground state by enumeration
        let mut e0 = f64::MAX;
        let mut s = vec![-1i8; n];
        for m in 0..(1u32 << n) {
            for b in 0..n {
                s[b] = if m >> b & 1 == 1 { 1 } else { -1 };
            }
            let e = g.energy(&s);
            if e < e0 {
                e0 = e;
            }
        }
        let betas = geometric_ladder(0.1, 3.0, 8);
        let res = parallel_tempering(&g, &betas, 200, 5, 0xF00D, None);
        assert!((res.best_e - e0).abs() < 1e-9, "PT found {} but exact ground state is {}", res.best_e, e0);
        // ladder health: no dead pair
        assert!(res.swap_rates.iter().all(|&x| x > 0.05), "dead ladder pair: {:?}", res.swap_rates);
        // annealing sanity on the same instance
        let sched: Vec<(f64, usize)> = geometric_ladder(0.1, 3.0, 30).into_iter().map(|b| (b, 40)).collect();
        let (_, e_sa) = anneal(&g, &sched, 0xA11, None);
        assert!((e_sa - e0).abs() < 1e-9, "SA found {} vs exact {}", e_sa, e0);
    }
}
