//! ferrotherm side of the comparison.
//!
//! Ising chains go through `gibbs::Sampler::collect`, which charges a read for every kept
//! state. Potts chains go through `potts::Sampler::collect` the same way. The three-body
//! chain uses `hubo::Hubo::energy` and `Hubo::delta` — the library's exact energy and flip
//! gap — inside a chromatic heat bath, because `hubo::anneal` searches for a minimum and
//! does not target the Boltzmann law. The ledger is charged one update per spin per sweep
//! and one read per site per kept state, which is the accounting `collect` uses. Those
//! counts are counts. Nothing here multiplies them by a price.

use crate::score::{self, Score};
use crate::spec::{Kind, Workload};
use ferrotherm::graph::GraphBuilder;
use ferrotherm::hubo::Hubo;
use ferrotherm::ising::exact_boltzmann;
use ferrotherm::kernel::p_up;
use ferrotherm::ledger::Ledger;
use ferrotherm::potts::{self, Interaction, Local, Potts, PottsBuilder};
use ferrotherm::rng::Pcg;
use ferrotherm::samples::Plan;
use std::time::Instant;

pub struct Chain {
    pub score: Score,
    pub updates: u64,
    pub reads: u64,
    pub energies: Vec<f64>,
    pub states_spin: Vec<Vec<i8>>,
    pub states_potts: Vec<Vec<u8>>,
    /// Compile time of an external runner. Absent for ferrotherm, which is not a JIT.
    pub jit_seconds: Option<f64>,
}

pub fn run(w: &Workload, seed: u64) -> Result<Chain, String> {
    match w.kind {
        Kind::Ising => run_ising(w, seed),
        Kind::Potts => run_potts(w, seed),
        Kind::Hubo => run_hubo(w, seed),
    }
}

fn run_ising(w: &Workload, seed: u64) -> Result<Chain, String> {
    let g = ising_graph(w);
    if g.classes.iter().map(Vec::len).sum::<usize>() != w.n {
        return Err(format!("{} colouring does not cover the graph", w.id));
    }
    // The spec's blocks are this graph's classes. A mismatch means the Python side
    // would be sweeping a different partition than this sampler.
    for (got, want) in g.classes.iter().zip(&w.blocks) {
        let got: Vec<usize> = got.iter().map(|&i| i as usize).collect();
        if &got != want {
            return Err(format!("{} block {want:?} is not the graph colour class {got:?}", w.id));
        }
    }
    let mut smp = ferrotherm::gibbs::Sampler::new(&g, w.beta, seed);
    for s in &mut smp.s {
        *s = -1;
    }
    let mut led = Ledger::default();
    let plan = Plan::new(w.warmup, w.draws, w.thin);
    let t0 = Instant::now();
    let set = smp.collect(&plan, Some(&mut led));
    let seconds = t0.elapsed().as_secs_f64();
    let exact = exact_boltzmann(&g, w.beta);
    let hist = score::histogram_spins(set.states());
    let (e_mean, mag) = exact_spin_observables(&g, &exact);
    let mae = score::mae(&score::empirical_magnetization(set.states()), &mag);
    let sc = score::score_chain(&hist, &exact, set.energies(), e_mean, Some(mae), seconds);
    Ok(Chain {
        score: sc,
        updates: led.samples,
        reads: led.reads,
        energies: set.energies().to_vec(),
        states_spin: set.states().to_vec(),
        states_potts: Vec::new(),
        jit_seconds: None,
    })
}

fn run_potts(w: &Workload, seed: u64) -> Result<Chain, String> {
    let m = potts_model(w);
    let zeros = vec![0u8; w.n];
    let mut smp = potts::Sampler::new(&m, w.beta, seed).from_state(&zeros).map_err(|e| format!("{e:?}"))?;
    let mut led = Ledger::default();
    let plan = Plan::new(w.warmup, w.draws, w.thin);
    let t0 = Instant::now();
    let run = smp.collect(&plan, Local::HeatBath, Some(&mut led));
    let seconds = t0.elapsed().as_secs_f64();
    let exact = potts::enumerate(&m, w.beta).map_err(|e| format!("{e:?}"))?;
    let hist = score::histogram_potts(run.states(), w.q);
    let sc = score::score_chain(&hist, &exact.p, run.energies(), exact.mean_energy, None, seconds);
    Ok(Chain {
        score: sc,
        updates: led.samples,
        reads: led.reads,
        energies: run.energies().to_vec(),
        states_spin: Vec::new(),
        states_potts: run.states().to_vec(),
        jit_seconds: None,
    })
}

fn run_hubo(w: &Workload, seed: u64) -> Result<Chain, String> {
    let h = hubo_model(w)?;
    let thin = w.thin.max(1);
    let mut states = Vec::with_capacity(w.draws);
    let mut energies = Vec::with_capacity(w.draws);
    // Same window `Sampler::collect` has: burn-in is inside the timed call, then `thin`
    // sweeps between kept states, and the state at the end of burn-in is not itself kept.
    let mut s = vec![-1i8; w.n];
    let mut rng = Pcg::new(seed, 0x4855_424F);
    let mut updates = 0u64;
    let t0 = Instant::now();
    for _ in 0..w.warmup {
        hubo_sweep(&h, &mut s, w.beta, &mut rng, &w.blocks, &mut updates);
    }
    for _ in 0..w.draws {
        for _ in 0..thin {
            hubo_sweep(&h, &mut s, w.beta, &mut rng, &w.blocks, &mut updates);
        }
        energies.push(h.energy(&s));
        states.push(s.clone());
    }
    let seconds = t0.elapsed().as_secs_f64();
    let reads = (w.draws * w.n) as u64;
    let exact_p = exact_hubo(&h, w.beta);
    let hist = score::histogram_spins(&states);
    let e_mean = exact_mean_hubo(&h, &exact_p);
    let mag = score::exact_magnetization(&exact_p, w.n);
    let mae = score::mae(&score::empirical_magnetization(&states), &mag);
    let sc = score::score_chain(&hist, &exact_p, &energies, e_mean, Some(mae), seconds);
    Ok(Chain {
        score: sc,
        updates,
        reads,
        energies,
        states_spin: states,
        states_potts: Vec::new(),
        jit_seconds: None,
    })
}

/// Score states some other runner produced, with this crate's energy and this crate's oracle.
///
/// `updates` and `reads` are the schedule's counts (one update per site per sweep, one read per
/// site per kept state), not a ledger the other runner charged. ferrotherm's own chains report
/// the ledger instead, which for these schedules is the same arithmetic.
pub fn score_observed(
    w: &Workload,
    seconds: f64,
    jit_seconds: Option<f64>,
    spins: &[Vec<i8>],
    potts_states: &[Vec<u8>],
) -> Result<Chain, String> {
    let sweeps = (w.warmup + w.draws * w.thin.max(1)) as u64;
    let updates = sweeps * w.n as u64;
    let reads = (w.draws * w.n) as u64;
    match w.kind {
        Kind::Ising => {
            let g = ising_graph(w);
            let exact = exact_boltzmann(&g, w.beta);
            let hist = score::histogram_spins(spins);
            let energies: Vec<f64> = spins.iter().map(|s| g.energy(s)).collect();
            let (e_mean, mag) = exact_spin_observables(&g, &exact);
            let mae = score::mae(&score::empirical_magnetization(spins), &mag);
            let sc = score::score_chain(&hist, &exact, &energies, e_mean, Some(mae), seconds);
            Ok(Chain {
                score: sc,
                updates,
                reads,
                energies,
                states_spin: spins.to_vec(),
                states_potts: Vec::new(),
                jit_seconds,
            })
        }
        Kind::Potts => {
            let m = potts_model(w);
            let exact = potts::enumerate(&m, w.beta).map_err(|e| format!("{e:?}"))?;
            let hist = score::histogram_potts(potts_states, w.q);
            let energies: Vec<f64> = potts_states
                .iter()
                .map(|s| m.energy(s).map_err(|e| format!("{e}")))
                .collect::<Result<Vec<_>, _>>()?;
            let sc = score::score_chain(&hist, &exact.p, &energies, exact.mean_energy, None, seconds);
            Ok(Chain {
                score: sc,
                updates,
                reads,
                energies,
                states_spin: Vec::new(),
                states_potts: potts_states.to_vec(),
                jit_seconds,
            })
        }
        Kind::Hubo => {
            let h = hubo_model(w)?;
            let exact_p = exact_hubo(&h, w.beta);
            let hist = score::histogram_spins(spins);
            let energies: Vec<f64> = spins.iter().map(|s| h.energy(s)).collect();
            let e_mean = exact_mean_hubo(&h, &exact_p);
            let mag = score::exact_magnetization(&exact_p, w.n);
            let mae = score::mae(&score::empirical_magnetization(spins), &mag);
            let sc = score::score_chain(&hist, &exact_p, &energies, e_mean, Some(mae), seconds);
            Ok(Chain {
                score: sc,
                updates,
                reads,
                energies,
                states_spin: spins.to_vec(),
                states_potts: Vec::new(),
                jit_seconds,
            })
        }
    }
}

fn hubo_sweep(h: &Hubo, s: &mut [i8], beta: f64, rng: &mut Pcg, blocks: &[Vec<usize>], updates: &mut u64) {
    for block in blocks {
        let mut next = Vec::with_capacity(block.len());
        for &i in block {
            // `delta` is E(flipped) - E(current) = 2 * field * s_i, so field = s_i * delta / 2.
            let field = f64::from(s[i]) * h.delta(s, i) / 2.0;
            next.push(rng.spin(p_up(field, beta)));
            *updates += 1;
        }
        for (k, &i) in block.iter().enumerate() {
            s[i] = next[k];
        }
    }
}

fn ising_graph(w: &Workload) -> ferrotherm::graph::Graph {
    let mut gb = GraphBuilder::new(w.n);
    for (i, &b) in w.biases.iter().enumerate() {
        if b != 0.0 {
            gb.bias(i, b);
        }
    }
    for &(i, j, jij) in &w.edges {
        gb.couple(i, j, jij);
    }
    gb.build()
}

fn potts_model(w: &Workload) -> Potts {
    let mut b = PottsBuilder::new(w.q, w.n, Interaction::Potts);
    for &(i, j, jij) in &w.edges {
        b.couple(i, j, jij);
    }
    b.build()
}

fn hubo_model(w: &Workload) -> Result<Hubo, String> {
    let mut h = Hubo::new(w.n);
    for (i, &b) in w.biases.iter().enumerate() {
        if b != 0.0 {
            h.add(&[i], b).map_err(|e| format!("{e:?}"))?;
        }
    }
    for &(i, j, jij) in &w.edges {
        h.add(&[i, j], jij).map_err(|e| format!("{e:?}"))?;
    }
    for &(i, j, k, wj) in &w.triples {
        h.add(&[i, j, k], wj).map_err(|e| format!("{e:?}"))?;
    }
    Ok(h)
}

fn exact_spin_observables(g: &ferrotherm::graph::Graph, p: &[f64]) -> (f64, Vec<f64>) {
    let n = g.n;
    let mut mean_e = 0.0;
    let mut s = vec![-1i8; n];
    for (mask, &pk) in p.iter().enumerate() {
        for b in 0..n {
            s[b] = if mask >> b & 1 == 1 { 1 } else { -1 };
        }
        mean_e += pk * g.energy(&s);
    }
    (mean_e, score::exact_magnetization(p, n))
}

fn exact_hubo(h: &Hubo, beta: f64) -> Vec<f64> {
    let n = h.len();
    let m = 1usize << n;
    let mut logw = vec![0.0f64; m];
    let mut s = vec![-1i8; n];
    let mut mx = f64::NEG_INFINITY;
    for mask in 0..m {
        for b in 0..n {
            s[b] = if mask >> b & 1 == 1 { 1 } else { -1 };
        }
        let l = -beta * h.energy(&s);
        logw[mask] = l;
        mx = mx.max(l);
    }
    let mut z = 0.0;
    for v in &mut logw {
        *v = (*v - mx).exp();
        z += *v;
    }
    for v in &mut logw {
        *v /= z;
    }
    logw
}

fn exact_mean_hubo(h: &Hubo, p: &[f64]) -> f64 {
    let n = h.len();
    let mut s = vec![-1i8; n];
    let mut mean = 0.0;
    for (mask, &pk) in p.iter().enumerate() {
        for b in 0..n {
            s[b] = if mask >> b & 1 == 1 { 1 } else { -1 };
        }
        mean += pk * h.energy(&s);
    }
    mean
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::canonical;

    #[test]
    fn a_hot_chain_is_close_to_its_boltzmann_law() {
        // beta = 0 is uniform and does not exercise the factor of two in the heat bath.
        // This is the README chain at beta 1, one seed, long enough that a stuck sampler
        // cannot land inside the bound.
        let mut w = canonical(1, 200, 4000).into_iter().next().unwrap();
        w.trials = 1;
        let chain = run(&w, 7).unwrap();
        assert!(chain.score.tv < 0.06, "tv {}", chain.score.tv);
        assert!(chain.score.energy_abs_err < 0.08, "energy err {}", chain.score.energy_abs_err);
        assert!(chain.score.kl_empirical_exact < 0.02, "kl {}", chain.score.kl_empirical_exact);
        assert!(chain.reads > 0, "keeping a state has to charge a read");
        assert!(chain.updates > 0);
    }

    #[test]
    fn three_body_heat_bath_hits_the_enumerated_law() {
        // One triple on three spins. Three colours, so the sweep is single-site heat bath
        // in index order, and the oracle is the 8-state Boltzmann law of that term.
        let w = Workload {
            id: "triple",
            kind: Kind::Hubo,
            beta: 1.0,
            warmup: 100,
            draws: 4000,
            thin: 1,
            trials: 1,
            seed0: 1,
            source: "test",
            n: 3,
            q: 2,
            biases: vec![0.0; 3],
            edges: Vec::new(),
            triples: vec![(0, 1, 2, 0.7)],
            blocks: vec![vec![0], vec![1], vec![2]],
        };
        let chain = run(&w, 3).unwrap();
        assert!(chain.score.tv < 0.05, "tv {}", chain.score.tv);
        assert!(chain.score.energy_abs_err < 0.05, "dE {}", chain.score.energy_abs_err);
    }
}
