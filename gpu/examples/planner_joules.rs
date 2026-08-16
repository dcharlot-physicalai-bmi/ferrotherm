//! What does the THINKING cost, next to what the MOVING costs?
//!
//! TR-2026-40 measures a sampling controller on a three-joint arm and reports the actuation: at a
//! matched budget of 19,200 dynamics evaluations per action, refinement reaches all 24 targets on
//! 17.8 joules of actuation against a reference controller's 26.4. That is the body's half. The
//! report leaves one quantity open, and it is the one a deployment decision turns on: joules per
//! COMPLETED TASK, with the compute counted in the same sum as the motors.
//!
//! This measures the compute half at the wall. It runs the same rollout inner loop the arm
//! controller runs, under the SoC's own power counters, and divides by the evaluations performed.
//! Multiply that by the evaluations a completed task costs and the two halves finally add.
//!
//! Deliberately NOT a port of the arm bench. The rollout kernel below is the loop the `evals`
//! counter counts and nothing else: no target set, no closed loop, and above all no copy of the
//! identified power model, because a calibrated model duplicated across two files becomes two
//! different published numbers within a month.
//!
//! PROVENANCE, which decides how the sum may be read. The compute figure is MEASURED on this
//! machine. The actuation figure is MODELLED, from coefficients calibrated to a measured 71.5 J
//! reach on a Unitree G1 arm. A metered number and a modelled one are different claims and this
//! prints them as two columns rather than one total, so a reader can see which is which.
//!
//! PREDICTED before measuring, recorded so it cannot be adjusted afterwards: a general-purpose CPU
//! planning at this budget costs joules of the same ORDER as the arm's actuation, not orders below
//! it. Falsified if the compute share lands under one percent, which would mean the whole
//! compute-versus-actuation question is settled and nobody needed to ask it.

use ferrotherm::rng::Pcg;
use std::time::{Duration, Instant};

const NJ: usize = 3;
const DT: f64 = 0.01;
const TAU_ACT: f64 = 0.06;
const VCAP: f64 = 2.4;

/// Published in TR-2026-40 section 7.4: the winning configuration's compute budget per action.
const EVALS_PER_ACTION: u64 = 19_200;
/// Section 7.5, median reach time, at the same DT the controller ticks on.
const MEDIAN_REACH_S: f64 = 1.35;
/// Section 7.4: actuation joules per completed task for that configuration, and for the reference.
const E_ACT_MPPI: f64 = 17.8;
const E_ACT_REFERENCE: f64 = 26.4;

fn gauss(r: &mut Pcg) -> f64 {
    // Box-Muller. The controller's noise draw is part of the work being priced, so it is inside
    // the measured loop rather than precomputed.
    let (u1, u2) = (r.f64().max(1e-12), r.f64());
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// One rollout of `horizon` dynamics evaluations, returning the accumulated cost.
///
/// This is the arm bench's inner loop: sample a perturbation, cap it, apply first-order actuator
/// lag, integrate, and score against a target. Returned rather than discarded so the optimiser
/// cannot delete the work being measured.
#[inline(never)]
fn rollout(rng: &mut Pcg, horizon: usize, sigma: f64, tgt: (f64, f64)) -> f64 {
    let (mut q, mut qd) = ([0.35f64, 0.55, -0.25], [0.0f64; NJ]);
    let mut cost = 0.0;
    for _ in 0..horizon {
        let mut u = [0.0f64; NJ];
        for d in 0..NJ {
            u[d] = gauss(rng) * sigma;
            u[d] = u[d].clamp(-VCAP, VCAP);
        }
        for j in 0..NJ {
            qd[j] += (u[j] - qd[j]) * (DT / TAU_ACT).min(1.0);
            q[j] += qd[j] * DT;
        }
        // Forward kinematics of the three-link planar arm, then the quadratic tracking cost.
        let (a, b, c) = (q[0], q[0] + q[1], q[0] + q[1] + q[2]);
        let cx = 0.30 * a.cos() + 0.26 * b.cos() + 0.16 * c.cos();
        let cy = 0.30 * a.sin() + 0.26 * b.sin() + 0.16 * c.sin();
        cost += ((cx - tgt.0).powi(2) + (cy - tgt.1).powi(2)) * 60.0
            + u.iter().map(|v| v * v).sum::<f64>() * 0.02;
    }
    cost
}

fn main() {
    let Some(mut meter) = ferrotherm_meter::Meter::detect() else {
        eprintln!("no power backend on this machine; nothing to measure");
        return;
    };
    println!("machine : {}", meter.machine());
    println!();
    println!("PREDICTED before measuring: planning at this budget costs joules of the same order as");
    println!("the arm's actuation, not orders below it. Falsified if the compute share is under 1%.");
    println!();

    let horizon = 16usize;
    let sigma = 0.6f64;
    let window = Duration::from_secs(6);

    // Two independent passes, because a figure that has not been repeated is not yet a figure.
    // The clamp_rate bench on this machine moved 36 percent between consecutive invocations while
    // its ratio held, and that was on a quieter afternoon than most.
    let mut rows: Vec<(f64, f64, f64, f64)> = Vec::new();
    for pass in 0..5 {
        std::thread::sleep(Duration::from_secs(8));
        // A refused pass is a skipped pass, never a fatal one. This machine shares itself with
        // another session's build, so quiet windows arrive irregularly; aborting the whole run on
        // the first noisy one throws away the good windows that come after it.
        let idle = match meter.idle(Duration::from_secs(4)) {
            Ok(b) => b,
            Err(e) => { println!("  pass {}: baseline refused, {e}", pass + 1); continue; }
        };
        // ALL CORES, for two reasons that happen to agree. One busy core is invisible on this SoC:
        // the first version of this bench drew -1.60 W against a baseline wandering by 1.45 W and
        // the meter refused it, which is the same wall `joules.rs` hit and solved the same way. The
        // other reason is fidelity. Rollouts are the parallel dimension of a sampling controller,
        // so a planner that used one core would be a planner nobody would ship.
        let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8);
        let evals = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
        let sink = std::sync::Arc::new(std::sync::Mutex::new(0.0f64));
        let run = match meter.measure(idle, || {
            std::thread::scope(|sc| {
                for tid in 0..threads {
                    let evals = evals.clone();
                    let sink = sink.clone();
                    sc.spawn(move || {
                        let mut rng = Pcg::new(0xC0FFEE ^ pass ^ (tid as u64) << 32, 7);
                        let mut local = 0.0f64;
                        let mut n = 0u64;
                        let t0 = Instant::now();
                        while t0.elapsed() < window {
                            // A block of rollouts between clock reads: checking the clock every
                            // evaluation would put the clock inside the thing being priced.
                            for _ in 0..64 {
                                local += rollout(&mut rng, horizon, sigma, (0.42, 0.18));
                                n += horizon as u64;
                            }
                        }
                        evals.fetch_add(n, std::sync::atomic::Ordering::Relaxed);
                        *sink.lock().unwrap() += local;
                    });
                }
            });
        }) {
            Ok(r) => r,
            Err(e) => { println!("  pass {}: refused, {e}", pass + 1); continue; }
        };
        let evals = evals.load(std::sync::atomic::Ordering::Relaxed);
        let sink = *sink.lock().unwrap();
        let j_per_eval = run.joules_above_idle / evals as f64;
        println!(
            "pass {}  {:>6.2} s  {:>6.2} W above {:>5.2} W idle (sigma {:.2})  {:>12} evals  {:.3e} J/eval",
            pass + 1, run.seconds, run.mean_watts - run.idle_watts, run.idle_watts,
            run.idle_sigma, evals, j_per_eval);
        let _ = threads;
        rows.push((j_per_eval, run.mean_watts - run.idle_watts, run.idle_sigma, sink.abs().min(1.0)));
    }

    // A pass whose BASELINE wandered is not a pass. The first run of this bench averaged a pass
    // with a 9.24 W idle sigma together with one at 0.59 W and reported the mean, which is a
    // contaminated number wearing a clean one's clothes. A baseline that moves by a quarter of the
    // signal cannot locate the signal, so those passes are printed and then set aside.
    let kept: Vec<&(f64, f64, f64, f64)> = rows.iter().filter(|r| r.2 < 0.25 * r.1).collect();
    println!();
    for (i, r) in rows.iter().enumerate() {
        let ok = r.2 < 0.25 * r.1;
        println!("  pass {}  idle sigma {:>5.2} W against a {:>5.2} W signal  {}",
                 i + 1, r.2, r.1, if ok { "kept" } else { "set aside, baseline too noisy" });
    }
    if kept.is_empty() {
        println!();
        println!("No pass had a baseline quiet enough to divide by. This machine is running other");
        println!("work; the figure below would be noise wearing a decimal point. Re-run when idle.");
        return;
    }
    let j_eval = kept.iter().map(|r| r.0).sum::<f64>() / kept.len() as f64;
    let (lo, hi) = kept.iter().fold((f64::MAX, 0.0f64), |(l, h), r| (l.min(r.0), h.max(r.0)));
    println!();
    if kept.len() > 1 {
        println!("joules per dynamics evaluation: {j_eval:.3e}   ({} passes kept, agreeing to {:.2}x)",
                 kept.len(), hi / lo);
    } else {
        println!("joules per dynamics evaluation: {j_eval:.3e}   (ONE pass kept; unrepeated)");
    }

    // Evaluations a completed task costs: the published per-action budget times the actions in a
    // median reach. Both numbers come from TR-2026-40 rather than from this run.
    let actions = (MEDIAN_REACH_S / DT).round() as u64;
    let evals_task = EVALS_PER_ACTION * actions;
    let e_compute = j_eval * evals_task as f64;

    println!();
    println!("=== the two halves of one completed task ===");
    println!("{:<34}{:>14}{:>12}", "quantity", "value", "provenance");
    println!("{:<34}{:>14}{:>12}", "actions in a median reach", actions, "TR-2026-40");
    println!("{:<34}{:>14}{:>12}", "dynamics evaluations per task", evals_task, "TR-2026-40");
    println!("{:<34}{:>12.2} J{:>12}", "compute, this machine", e_compute, "measured");
    println!("{:<34}{:>12.2} J{:>12}", "actuation, sampling controller", E_ACT_MPPI, "modelled");
    println!("{:<34}{:>12.2} J{:>12}", "actuation, reference controller", E_ACT_REFERENCE, "modelled");
    println!();
    let share = 100.0 * e_compute / (e_compute + E_ACT_MPPI);
    println!("compute share of E_task: {share:.1} %   (E_task = {:.2} J, mixed provenance)",
             e_compute + E_ACT_MPPI);
    println!();
    if share < 1.0 {
        println!("PREDICTION FALSIFIED: the compute term is negligible at this budget.");
    } else {
        println!("PREDICTION HELD: the compute term is not negligible beside the motors.");
    }
    println!();
    println!("Read this as an UPPER BOUND on the compute half. It prices a general-purpose CPU");
    println!("doing the planning, which is the machine the argument is about escaping. An embedded");
    println!("part, a fixed-function sampler or a settling fabric would each land lower, and the");
    println!("distance between this number and theirs is exactly what a substrate has to buy.");
}
