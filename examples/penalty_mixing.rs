#![allow(missing_docs)]
// WHAT A LARGE PENALTY COSTS IN MIXING -- the other half of the penalty question, exactly.
//
// `penalty_exact` settled the equilibrium half: at fixed temperature the Boltzmann mass on the
// optimal tours can only RISE with the penalty A, so "the smallest feasible penalty is best" is
// false about mass. If it is true about anything it is true about sweeps: a large A makes every
// infeasible state a high wall, and a sampler that starts anywhere must climb over walls to reach
// the tours. That is a statement about the kernel's dynamics, and on a four-city model it is
// exactly computable:
//
//   * the exact integrated autocorrelation time of the energy under sequential Gibbs at each A;
//   * the number of sweeps from the UNIFORM start until the mass on the optimal tours is within 1%
//     (relative) of its equilibrium value.
//
// Together with `penalty_exact`'s P_GS(A) they give the number a finite-budget sampler actually
// pays: the sweeps to reach a given fraction of the optimal-tour mass at each penalty. Whether a
// small A wins on that -- and by how much against what it loses in equilibrium mass -- is the
// measured answer.
//
// ROUTES, named per cell. tau: `autocorr::tau_int_solved` -- GMRES on the fundamental-matrix system
// (G), and where the chain is too cold for f64 to certify that, the censored solve onto the local
// minima with GTH elimination (C). Sweeps to 1%: `autocorr::time_to_mass` -- the uniform law pushed
// sweep by sweep (p, exact), and past the push budget the same chain censored onto its local minima
// and run in real time (s, a slow-scale reduction whose error is the fast relaxation over the
// barrier crossing; the validation rows below measure it where both run).
//
// WHAT CHANGED ON 2026-09-28, AND WHY EVERY NUMBER AT THE TOP TWO PENALTIES MOVED. The first version
// summed the lag series to 40,000 lags and printed "a row at the cap is a bound"; the capped rows
// read 40000.5. The truth is 1e13 to 1e113 sweeps. And the kernel it summed was not the heat bath:
// the crate formed each down-flip's probability as 1 - p_up, which is EXACTLY ZERO once
// 2 beta f > 36.7, so at A >= 26 no occupied slot of any tour could ever be vacated while the
// reverse flips kept their e^{-2 beta f}. That kernel's tau was 2.2 million to 15 million times the
// heat bath's. Both are fixed: the complement is p_up(-f), and the cells are values.
//
// Count-based throughout; valid on a busy machine.
//
// run: cargo run --release --example penalty_mixing

use ferrotherm::autocorr::{
    spins, tau_int_censored, tau_int_solved, time_to_mass, Kernel, MassRoute, MassTime, Metastable, Route,
};
use ferrotherm::npising::Tsp;
use ferrotherm::rng::Pcg;

struct Instance {
    a_crit: f64,
    max_w: f64,
    threshold: f64,
    t_star: f64,
    /// Per state: feasible and optimal.
    optimal: Vec<bool>,
}

fn analyse(n: usize, weights: &[f64]) -> Instance {
    let one = Tsp::with_weights(n, weights, 1.0, 1.0).expect("valid");
    let two = Tsp::with_weights(n, weights, 1.0, 2.0).expect("valid");
    let ns = one.graph().n;
    let m = 1usize << ns;
    let mut pw = Vec::with_capacity(m);
    let mut feasible = Vec::with_capacity(m);
    let mut t_star = f64::INFINITY;
    for x in 0..m {
        let s = spins(x, ns);
        let h1 = one.graph().energy(&s) + one.offset();
        let h2 = two.graph().energy(&s) + two.offset();
        let p = h2 - h1;
        let w = h1 - p;
        let f = one.decode(&s).is_ok();
        if f {
            t_star = t_star.min(w);
        }
        pw.push((p, w));
        feasible.push(f);
    }
    let mut a_crit = 0.0f64;
    for (k, &(p, w)) in pw.iter().enumerate() {
        if !feasible[k] && p > 0.0 {
            a_crit = a_crit.max((t_star - w) / p);
        }
    }
    let optimal: Vec<bool> = pw.iter().zip(&feasible).map(|(&(_, w), &f)| f && (w - t_star).abs() < 1e-9).collect();
    let max_w = weights.iter().copied().filter(|w| w.is_finite()).fold(0.0f64, f64::max);
    let threshold = Tsp::new(n, weights).expect("valid").threshold();
    Instance { a_crit, max_w, threshold, t_star, optimal }
}

fn weights(n: usize, seed: u64) -> Vec<f64> {
    let mut rng = Pcg::new(seed, 0x7E5);
    let mut w = vec![0.0f64; n * n];
    for u in 0..n {
        for v in (u + 1)..n {
            let e = 1.0 + (rng.f64() * 9.0).floor().min(8.0);
            w[u * n + v] = e;
            w[v * n + u] = e;
        }
    }
    w
}

/// A time in sweeps, to the digits it carries.
fn sweeps(x: f64) -> String {
    if x < 1e6 {
        format!("{x:.1}")
    } else {
        format!("{x:.3e}")
    }
}

fn tau_code(r: Route) -> char {
    match r {
        Route::Gmres => 'G',
        Route::Cg => 'g',
        Route::Censored => 'C',
        Route::Dense => 'D',
        Route::LagSum => 'L',
    }
}

fn mass_code(r: MassRoute) -> char {
    match r {
        MassRoute::Pushed => 'p',
        MassRoute::SlowScale => 's',
    }
}

fn main() {
    let n = 4usize;
    let instances = 12u64;
    let beta = 1.0;
    let rel = 0.01;
    // Pushes before the slow scale takes over. Where tau is under a million sweeps the law is pushed
    // all the way (a budget of 20 tau + 10,000 sweeps), so the answer is exact; above that the chain
    // is frozen, its fast relaxation is long over after 200 sweeps (the hand-over line below says by
    // how much), and the slow scale answers. The ROUTE of tau is no guide: instance 2's max(W) chain
    // (tau 354) is too ill-conditioned for GMRES to certify 1e-10 and goes to the censored solve.
    let push_frozen = 200usize;
    let budget_for = |tau: f64| if tau < 1e6 { (20.0 * tau) as usize + 10_000 } else { push_frozen };
    println!("WHAT A PENALTY COSTS IN SWEEPS -- exact, sequential Gibbs on all 2^16 states of four-city TSPs, beta = {beta}\n");
    println!("  penalties  just above A_crit; Lucas's max(W); the provable threshold; four times it");
    println!("  columns    P_GS at equilibrium; tau_int of the energy (sweeps); sweeps from uniform to within {}% of P_GS", rel * 100.0);
    println!("  routes     tau: G = GMRES, C = censored (GTH); sweeps: p = pushed (exact), s = slow scale (reduction)\n");
    println!(
        "  {:>4}   {:>27}   {:>27}   {:>27}   {:>27}",
        "inst", "A_crit+: P_GS tau K1%", "max(W): P_GS tau K1%", "thresh: P_GS tau K1%", "4x thr: P_GS tau K1%"
    );
    let mut log_tau = [0.0f64; 4];
    let mut log_k = [0.0f64; 4];
    let mut mass = [0.0f64; 4];
    let mut wins = [0usize; 4];
    let mut worst_handover = 0.0f64;
    let mut frozen_route = [0usize; 4];
    let mut slow_route = [0usize; 4];
    for seed in 0..instances {
        let w = weights(n, seed);
        let inst = analyse(n, &w);
        let penalties = [inst.a_crit * 1.05 + 1e-9, inst.max_w, inst.threshold, 4.0 * inst.threshold];
        let mut cells = Vec::new();
        let (mut best_k, mut best_idx) = (f64::INFINITY, 0);
        for (i, &a) in penalties.iter().enumerate() {
            let model = Tsp::with_weights(n, &w, 1.0, a).expect("valid");
            let g = model.graph();
            let tau = tau_int_solved(g, beta, Kernel::SequentialGibbs, |s| g.energy(s), 1e-10, 3_000)
                .unwrap_or_else(|e| panic!("instance {seed}, A = {a}: {e}"));
            let budget = budget_for(tau.tau_int);
            let start = vec![1.0 / (1usize << g.n) as f64; 1usize << g.n];
            let k: MassTime = time_to_mass(g, beta, Kernel::SequentialGibbs, &start, |x| inst.optimal[x], rel, budget, Metastable::LocalMinima)
                .unwrap_or_else(|e| panic!("instance {seed}, A = {a}: {e}"));
            if k.route == MassRoute::SlowScale {
                worst_handover = worst_handover.max(k.handover / (rel * k.stationary));
            }
            if tau.route == Route::Censored {
                frozen_route[i] += 1;
            }
            if k.route == MassRoute::SlowScale {
                slow_route[i] += 1;
            }
            cells.push(format!(
                "{:>5.3} {:>9}{} {:>9}{}",
                k.stationary,
                sweeps(tau.tau_int),
                tau_code(tau.route),
                sweeps(k.steps),
                mass_code(k.route)
            ));
            log_tau[i] += tau.tau_int.ln();
            log_k[i] += k.steps.ln();
            mass[i] += k.stationary;
            if k.steps < best_k {
                best_k = k.steps;
                best_idx = i;
            }
        }
        wins[best_idx] += 1;
        println!(
            "  {seed:>4}   {}   (T* {}, A_crit {:.2}, max W {}, thr {:.0})",
            cells.join("   "),
            inst.t_star,
            inst.a_crit,
            inst.max_w,
            inst.threshold
        );
    }
    let f = instances as f64;
    println!("\n  over {instances} instances (tau and sweeps as GEOMETRIC means: at the top two penalties they span decades):");
    let names = ["just above A_crit", "max(W)", "provable threshold", "4x threshold"];
    for (i, name) in names.iter().enumerate() {
        println!(
            "    {name:<20} P_GS {:.3}   tau_int {:>10} sweeps   sweeps to 1% of P_GS {:>10}   fastest on {:>2}   routes: censored {:>2}, slow scale {:>2}",
            mass[i] / f,
            sweeps((log_tau[i] / f).exp()),
            sweeps((log_k[i] / f).exp()),
            wins[i],
            frozen_route[i],
            slow_route[i]
        );
    }
    println!("    slow-scale hand-over: the pushed mass and the reduction's differed by at most {worst_handover:.1e} of the 1% band");

    // THE CHECKS THE TWO NEW ROUTES CARRY IN PLACE OF AN ERROR BAR.
    println!("\n  CHECKS");
    // (1) The censored tau does not depend on the metastable set: two sets whose inner systems share
    // nothing, the 36 local minima and those with every single-flip neighbour.
    let w0 = weights(n, 0);
    let inst0 = analyse(n, &w0);
    for (name, a) in [("threshold", inst0.threshold), ("4x threshold", 4.0 * inst0.threshold)] {
        let model = Tsp::with_weights(n, &w0, 1.0, a).expect("valid");
        let g = model.graph();
        let one = tau_int_censored(g, beta, Kernel::SequentialGibbs, |s| g.energy(s), Metastable::LocalMinima, 1e-13).expect("censored");
        let two = tau_int_censored(g, beta, Kernel::SequentialGibbs, |s| g.energy(s), Metastable::LocalMinimaAndNeighbours, 1e-13)
            .expect("censored");
        println!(
            "    instance 0, {name:<12} censored tau on the minima {:.10e}, on minima and neighbours {:.10e}: rel {:.1e}",
            one.tau_int,
            two.tau_int,
            (one.tau_int - two.tau_int).abs() / one.tau_int
        );
    }
    // (2) The slow-scale reduction where the push also runs: every max(W) cell, pushed and reduced
    // from a 200-sweep hand-over. This is where the barrier is LOWEST and the reduction weakest.
    let mut worst = 0.0f64;
    for seed in 0..instances {
        let w = weights(n, seed);
        let inst = analyse(n, &w);
        let model = Tsp::with_weights(n, &w, 1.0, inst.max_w).expect("valid");
        let g = model.graph();
        let start = vec![1.0 / (1usize << g.n) as f64; 1usize << g.n];
        let pushed = time_to_mass(g, beta, Kernel::SequentialGibbs, &start, |x| inst.optimal[x], rel, 200_000, Metastable::LocalMinima)
            .expect("pushed");
        let reduced = time_to_mass(g, beta, Kernel::SequentialGibbs, &start, |x| inst.optimal[x], rel, push_frozen, Metastable::LocalMinima)
            .expect("reduced");
        if pushed.route == MassRoute::Pushed && reduced.route == MassRoute::SlowScale {
            worst = worst.max((reduced.steps - pushed.steps).abs() / pushed.steps);
        }
    }
    println!("    slow scale against pushing on the twelve max(W) cells, the lowest barriers of the frozen-capable: worst rel {worst:.1e}");

    println!("\n  WHAT THE TABLE SAYS.\n");
    let decades = |i: usize| (log_tau[i] / f) / std::f64::consts::LN_10;
    println!("  P_GS rises with A -- the equilibrium half, a theorem: {:.3} just above A_crit to {:.3} at four times the threshold.", mass[0] / f, mass[3] / f);
    println!(
        "  tau_int rises from about 10^{:.1} sweeps to 10^{:.1}, 10^{:.1} and 10^{:.1}: the dynamics half, measured exactly on every cell.",
        decades(0),
        decades(1),
        decades(2),
        decades(3)
    );
    println!("  A small penalty wins for any sampler with a sweep budget; the large ones buy mass no sampler reaches.");
}
