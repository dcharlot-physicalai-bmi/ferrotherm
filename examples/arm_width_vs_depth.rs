//! The width-versus-depth question on a body, priced in joules.
//!
//! `sequential_depth.rs` and `depth_vs_dimension.rs` establish, against an exact Riccati optimum,
//! that sampling width has an accuracy floor which only sequential refinement clears, and that the
//! floor worsens as the action space grows. Both are linear plants. This probe asks the same
//! question on a three-joint arm with gravity, actuator lag, torque limits and an identified
//! electrical power model, and adds the quantity a linear plant cannot supply: **what each
//! strategy costs in joules on the body**.
//!
//! # What anchors it, and what does not
//!
//! There is no exact optimum here, and this probe does not pretend otherwise. The anchor is a
//! **reference controller**: the analytic Jacobian-transpose expert used throughout the Institute's
//! arm work, velocity-capped, which reaches every target in this set. A reference is weaker than an
//! oracle and the numbers below are stated against it as such: "the expert reaches in 43 J" is a
//! measurement of that controller, not a statement about the best achievable.
//!
//! What the linear probes had and this one does not is truth. What this one has and they do not is
//! a body: real gravity torque, a first-order actuator, a torque limit, and a power model whose
//! copper, viscous and Coulomb coefficients are calibrated to a measured 71.5 J reach on a Unitree
//! G1 arm (arXiv:2606.15918). Both are needed, and neither substitutes for the other.
//!
//! # The objective does not contain the energy
//!
//! The rollout cost is distance to target plus control effort. `joint_power` is evaluated only
//! in `tick`, on the true body, as a measurement. So any energy result here is what better
//! planning is worth in the actuation term, not what an energy-aware objective could reach.
//! Gravity is likewise absent from the rollout's state update, and that is correct rather than
//! an omission: in this model gravity torque enters the power, not the state, so a predictor
//! that ignores it still predicts the state exactly.
//!
//! Arm, expert and power model are the ones in `reach_on_z1.rs` and `research/efa/total_task_energy.rs`.
//!
//! ```text
//! cargo run --release --example arm_width_vs_depth
//! ```

use ferrotherm::rng::Pcg;

const NJ: usize = 3;
const DT: f64 = 0.01;
const LINK: [f64; NJ] = [0.30, 0.26, 0.16];
const MASS: [f64; NJ] = [1.6, 1.1, 0.5];
const G: f64 = 9.81;
const TOL: f64 = 0.04;
const MAX_T: f64 = 6.0;
const VCAP: f64 = 2.4;
const TAU_ACT: f64 = 0.06;
const TAU_LIM: f64 = 40.0;
const START: [f64; NJ] = [0.35, 0.55, -0.25];
const K_CU: [f64; NJ] = [0.055, 0.150, 0.120];
const K_VISC: [f64; NJ] = [0.900, 0.220, 0.260];
const K_COUL: [f64; NJ] = [0.220, 0.180, 0.520];
const P_IDLE: f64 = 9.0;

fn fk(q: &[f64; NJ]) -> (f64, f64) {
    let (mut x, mut y, mut a) = (0.0, 0.0, 0.0);
    for j in 0..NJ { a += q[j]; x += LINK[j] * a.cos(); y += LINK[j] * a.sin(); }
    (x, y)
}
fn gravity_tau(q: &[f64; NJ]) -> [f64; NJ] {
    let mut t = [0.0; NJ];
    for j in 0..NJ {
        let mut acc = 0.0;
        for k in j..NJ {
            let mut a = 0.0;
            for i in 0..=k { a += q[i]; }
            let r = if k == j { 0.5 * LINK[k] } else { LINK[k] };
            acc += MASS[k] * G * r * a.cos();
        }
        t[j] = acc;
    }
    t
}
/// Electrical power: copper (torque squared), viscous (speed squared), Coulomb (speed), plus idle.
fn joint_power(qd: &[f64; NJ], tau: &[f64; NJ]) -> f64 {
    let mut p = P_IDLE;
    for j in 0..NJ {
        p += K_CU[j] * tau[j] * tau[j] + K_VISC[j] * qd[j] * qd[j] + K_COUL[j] * qd[j].abs();
    }
    p
}
/// The reference controller: Jacobian-transpose descent on Cartesian error, velocity-capped.
fn expert(q: &[f64; NJ], tgt: (f64, f64)) -> [f64; NJ] {
    let (x, y) = fk(q);
    let (ex, ey) = (tgt.0 - x, tgt.1 - y);
    let mut out = [0.0; NJ];
    for j in 0..NJ {
        let (mut a, mut dx, mut dy) = (0.0, 0.0, 0.0);
        for k in 0..NJ {
            a += q[k];
            if k >= j { dx += -LINK[k] * a.sin(); dy += LINK[k] * a.cos(); }
        }
        out[j] = 35.0 * (dx * ex + dy * ey) / 1.08;
    }
    cap(&mut out);
    out
}
fn cap(u: &mut [f64; NJ]) {
    let n: f64 = u.iter().map(|v| v * v).sum::<f64>().sqrt();
    if n > VCAP { for v in u.iter_mut() { *v *= VCAP / n; } }
}
/// One control tick of the true dynamics. Returns electrical joules spent in the tick.
fn tick(q: &mut [f64; NJ], qd: &mut [f64; NJ], cmd: &[f64; NJ]) -> f64 {
    let gt = gravity_tau(q);
    let mut tau = [0.0; NJ];
    for j in 0..NJ {
        let qd_new = qd[j] + (cmd[j] - qd[j]) * (DT / TAU_ACT).min(1.0);
        let a = (qd_new - qd[j]) / DT;
        tau[j] = (0.22 * a + gt[j]).clamp(-TAU_LIM, TAU_LIM);
        qd[j] = qd_new;
        q[j] += qd[j] * DT;
    }
    joint_power(qd, &tau) * DT
}
fn rand_target(r: &mut Pcg) -> (f64, f64) {
    loop {
        let a = r.f64() * 1.6 - 0.2;
        let rad = 0.30 + r.f64() * 0.38;
        let (x, y) = (rad * a.cos(), rad * a.sin());
        if x > 0.10 && (x * x + y * y).sqrt() < 0.70 { return (x, y); }
    }
}

struct Run { ok: bool, e_act: f64, evals: u64 }

/// The reference controller, closed loop.
fn run_expert(tgt: (f64, f64)) -> Run {
    let (mut q, mut qd, mut e, mut t) = (START, [0.0; NJ], 0.0, 0.0);
    while t < MAX_T {
        let cmd = expert(&q, tgt);
        e += tick(&mut q, &mut qd, &cmd);
        t += DT;
        let (cx, cy) = fk(&q);
        if ((cx - tgt.0).powi(2) + (cy - tgt.1).powi(2)).sqrt() < TOL {
            return Run { ok: true, e_act: e, evals: (t / DT) as u64 };
        }
    }
    Run { ok: false, e_act: e, evals: (MAX_T / DT) as u64 }
}

/// MPPI on the same arm. `rollouts` divide across lanes; `passes` are a chain.
fn run_mppi(tgt: (f64, f64), rollouts: usize, passes: usize, horizon: usize,
            sigma: f64, lambda: f64, seed: u64) -> Run {
    let (mut q, mut qd, mut e, mut t) = (START, [0.0; NJ], 0.0, 0.0);
    let mut rng = Pcg::new(seed, 0);
    let mut nominal = vec![[0.0f64; NJ]; horizon];
    let mut evals: u64 = 0;
    while t < MAX_T {
        for _ in 0..passes {
            let mut costs = vec![0.0f64; rollouts];
            let mut noise = vec![[0.0f64; NJ]; rollouts * horizon];
            for r in 0..rollouts {
                let (mut sq, mut sqd) = (q, qd);
                for h in 0..horizon {
                    let mut u = [0.0f64; NJ];
                    for d in 0..NJ {
                        let z = gauss(&mut rng) * sigma;
                        noise[r * horizon + h][d] = z;
                        u[d] = nominal[h][d] + z;
                    }
                    cap(&mut u);
                    // the rollout uses the same dynamics the body will run
                    let gt = gravity_tau(&sq);
                    for j in 0..NJ {
                        let qn = sqd[j] + (u[j] - sqd[j]) * (DT / TAU_ACT).min(1.0);
                        let _ = gt[j];
                        sqd[j] = qn;
                        sq[j] += sqd[j] * DT;
                    }
                    let (cx, cy) = fk(&sq);
                    let d2 = (cx - tgt.0).powi(2) + (cy - tgt.1).powi(2);
                    costs[r] += d2 * 60.0 + u.iter().map(|v| v * v).sum::<f64>() * 0.02;
                    evals += 1;
                }
            }
            let best = costs.iter().cloned().fold(f64::INFINITY, f64::min);
            let w: Vec<f64> = costs.iter().map(|c| (-(c - best) / lambda).exp()).collect();
            let sw: f64 = w.iter().sum();
            for h in 0..horizon { for d in 0..NJ {
                let mut acc = 0.0;
                for r in 0..rollouts { acc += w[r] * noise[r * horizon + h][d]; }
                nominal[h][d] += acc / sw;
            } }
        }
        let mut cmd = nominal[0];
        cap(&mut cmd);
        e += tick(&mut q, &mut qd, &cmd);
        t += DT;
        let (cx, cy) = fk(&q);
        if ((cx - tgt.0).powi(2) + (cy - tgt.1).powi(2)).sqrt() < TOL {
            return Run { ok: true, e_act: e, evals };
        }
        for h in 0..horizon - 1 { nominal[h] = nominal[h + 1]; }
        nominal[horizon - 1] = [0.0; NJ];
    }
    Run { ok: false, e_act: e, evals }
}


/// MPPI with the identified power model **inside** the objective.
///
/// The variant above scores a rollout on distance and control effort, and measures joules
/// afterwards. This one prices the joules while choosing, using the same coefficients the body is
/// billed with. Sampling-based control can do this without a differentiable power model, which is
/// what makes it available here: the cost only has to be evaluable, not differentiable.
///
/// `w_e` is joules-weight. At zero this is the controller above. As it rises the controller should
/// spend less and, past some point, decline to reach at all — so the metric that decides is joules
/// per COMPLETED task, which charges every failed attempt to the task rather than discarding it.
fn run_mppi_energy(tgt: (f64, f64), rollouts: usize, passes: usize, horizon: usize,
                   sigma: f64, lambda: f64, w_e: f64, seed: u64) -> Run {
    let (mut q, mut qd, mut e, mut t) = (START, [0.0; NJ], 0.0, 0.0);
    let mut rng = Pcg::new(seed, 0);
    let mut nominal = vec![[0.0f64; NJ]; horizon];
    let mut evals: u64 = 0;
    while t < MAX_T {
        for _ in 0..passes {
            let mut costs = vec![0.0f64; rollouts];
            let mut noise = vec![[0.0f64; NJ]; rollouts * horizon];
            for r in 0..rollouts {
                let (mut sq, mut sqd) = (q, qd);
                for h in 0..horizon {
                    let mut u = [0.0f64; NJ];
                    for d in 0..NJ {
                        let z = gauss(&mut rng) * sigma;
                        noise[r * horizon + h][d] = z;
                        u[d] = nominal[h][d] + z;
                    }
                    cap(&mut u);
                    // gravity is load-bearing HERE, unlike the distance-only rollout: torque is
                    // what the power model is computed from, so predicting joules needs it.
                    let gt = gravity_tau(&sq);
                    let mut tau = [0.0f64; NJ];
                    for j in 0..NJ {
                        let qn = sqd[j] + (u[j] - sqd[j]) * (DT / TAU_ACT).min(1.0);
                        let a = (qn - sqd[j]) / DT;
                        tau[j] = (0.22 * a + gt[j]).clamp(-TAU_LIM, TAU_LIM);
                        sqd[j] = qn;
                        sq[j] += sqd[j] * DT;
                    }
                    let (cx, cy) = fk(&sq);
                    let d2 = (cx - tgt.0).powi(2) + (cy - tgt.1).powi(2);
                    costs[r] += d2 * 60.0
                              + u.iter().map(|v| v * v).sum::<f64>() * 0.02
                              + w_e * joint_power(&sqd, &tau) * DT;
                    evals += 1;
                }
            }
            let best = costs.iter().cloned().fold(f64::INFINITY, f64::min);
            let w: Vec<f64> = costs.iter().map(|c| (-(c - best) / lambda).exp()).collect();
            let sw: f64 = w.iter().sum();
            for h in 0..horizon { for d in 0..NJ {
                let mut acc = 0.0;
                for r in 0..rollouts { acc += w[r] * noise[r * horizon + h][d]; }
                nominal[h][d] += acc / sw;
            } }
        }
        let mut cmd = nominal[0];
        cap(&mut cmd);
        e += tick(&mut q, &mut qd, &cmd);
        t += DT;
        let (cx, cy) = fk(&q);
        if ((cx - tgt.0).powi(2) + (cy - tgt.1).powi(2)).sqrt() < TOL {
            return Run { ok: true, e_act: e, evals };
        }
        for h in 0..horizon - 1 { nominal[h] = nominal[h + 1]; }
        nominal[horizon - 1] = [0.0; NJ];
    }
    Run { ok: false, e_act: e, evals }
}

fn energy_sweep() {
    println!("PREDICTED before measuring: raising the joules-weight lowers actuation, and past");
    println!("some weight it costs reaches. Joules per COMPLETED task should therefore have an");
    println!("interior optimum rather than falling monotonically.\n");
    let mut tr = Pcg::new(20260815, 11);
    let targets: Vec<(f64, f64)> = (0..24).map(|_| rand_target(&mut tr)).collect();
    let ex: Vec<Run> = targets.iter().map(|&t| run_expert(t)).collect();
    let ok_ex = ex.iter().filter(|r| r.ok).count();
    let tot_ex: f64 = ex.iter().map(|r| r.e_act).sum();
    println!("reference controller: {ok_ex}/{} reached, {:.1} J per completed task\n",
             targets.len(), tot_ex / ok_ex as f64);
    println!("{:>10} {:>9} {:>12} {:>22} {:>11} {:>13} {:>10}", "w_energy", "reached", "median J", "J per completed task", "median t", "worst J", "worst t");
    for &w_e in &[0.0f64, 0.05, 0.2, 0.8, 3.0] {
        let runs: Vec<Run> = targets.iter().enumerate()
            .map(|(i, &t)| run_mppi_energy(t, 200, 8, 12, 0.9, 4.0, w_e, 100 + i as u64))
            .collect();
        let ok = runs.iter().filter(|r| r.ok).count();
        let total: f64 = runs.iter().map(|r| r.e_act).sum();   // failures charged to the task
        if ok == 0 { println!("{w_e:>10} {:>8}/{:<2}      none reached", 0, targets.len()); continue; }
        let mut es: Vec<f64> = runs.iter().filter(|r| r.ok).map(|r| r.e_act).collect();
        es.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let per_action = (200 * 8 * 12) as f64;
        let mut secs: Vec<f64> = runs.iter().filter(|r| r.ok)
            .map(|r| (r.evals as f64 / per_action) * DT).collect();
        secs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let t_med = secs[secs.len()/2];
        // If duration is what the energy term buys, the idle draw pays for it: P_IDLE * time.
        // The mean rose while the median held, which is a TAIL, and a median is the one statistic
        // guaranteed to hide one. Report the worst case alongside it.
        let worst_e = *es.last().unwrap();
        let t_worst = *secs.last().unwrap();
        println!("{w_e:>10} {ok:>7}/{:<2} {:>11.1} J {:>21.1} J {:>9.2} s {:>11.1} J {:>9.2} s",
                 targets.len(), es[es.len()/2], total / ok as f64, t_med, worst_e, t_worst);
    }
    println!("\nThe median column rewards a controller for giving up on hard targets.");
    println!("The last column does not: every failed attempt is charged to the task.");
}

fn gauss(r: &mut Pcg) -> f64 {
    let u = r.f64().max(1e-15);
    let v = r.f64();
    (-2.0 * u.ln()).sqrt() * (core::f64::consts::TAU * v).cos()
}

fn main() {
    if std::env::var("ONLY").as_deref() == Ok("energy") { energy_sweep(); return; }
    let mut tr = Pcg::new(20260815, 11);
    let targets: Vec<(f64, f64)> = (0..24).map(|_| rand_target(&mut tr)).collect();

    // the reference, on the same targets
    let ex: Vec<Run> = targets.iter().map(|&t| run_expert(t)).collect();
    let ok_ex = ex.iter().filter(|r| r.ok).count();
    let mut e_ex: Vec<f64> = ex.iter().filter(|r| r.ok).map(|r| r.e_act).collect();
    e_ex.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med_ex = e_ex[e_ex.len() / 2];
    println!("reference controller (analytic Jacobian-transpose, velocity-capped)");
    println!("  reaches {ok_ex}/{} targets, median actuation {med_ex:.1} J, 1 dynamics eval per tick\n",
             targets.len());

    println!("{:>9} {:>7} {:>8} {:>12} {:>14} {:>13}",
             "rollouts", "passes", "reached", "median J", "evals/action", "J vs expert");
    for &rollouts in &[50usize, 200, 800] {
        for &passes in &[1usize, 2, 4, 8] {
            let runs: Vec<Run> = targets.iter().enumerate()
                .map(|(i, &t)| run_mppi(t, rollouts, passes, 12, 0.9, 4.0, 100 + i as u64))
                .collect();
            let ok = runs.iter().filter(|r| r.ok).count();
            if ok == 0 { println!("{rollouts:>9} {passes:>7} {:>7}/{:<2}          none reached", 0, targets.len()); continue; }
            let mut es: Vec<f64> = runs.iter().filter(|r| r.ok).map(|r| r.e_act).collect();
            es.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let med = es[es.len() / 2];
            let per_action = (rollouts * passes * 12) as f64;
            // The published figure is this formula; the runs also count evaluations for real.
            // Ticks vary (an early success stops sooner), so check per-tick rather than per-run.
            for r in runs.iter() {
                let ticks = (r.evals as f64 / per_action).round();
                let implied = ticks * per_action;
                debug_assert!((implied - r.evals as f64).abs() < 1.0,
                    "evals/action formula disagrees with the measured count");
            }
            println!("{rollouts:>9} {passes:>7} {ok:>6}/{:<2} {med:>11.1} J {per_action:>13.0} {:>12.2}x",
                     targets.len(), med / med_ex);
        }
    }
    println!("\nRollouts divide across lanes; passes do not. The joules column is the body,");
    println!("the evals column is the compute, and they are different denominators.");
}
