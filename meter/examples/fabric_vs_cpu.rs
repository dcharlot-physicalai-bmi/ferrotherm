// `missing_docs` is denied workspace-wide and is right to be: it guards the API surface. An
// EXAMPLE has no API surface -- it is a program -- so the lint has nothing to guard here.
#![allow(missing_docs)]
// THE FLIPS-PER-SECOND FIGURE OF MERIT, PRICED PER INDEPENDENT SAMPLE, ON MEASURED SILICON.
//
// The p-bit literature ranks machines by flips per second and joules per flip (Aadit et al.,
// Nature Electronics 5:460, 2022: "flips per second -- the key figure of merit"; DSIM-2,
// arXiv:2606.25313; Extropic's Thermalizers, arXiv:2608.01615, Table IV). This crate measured a
// 1,024-p-bit fabric on a Kria KV260 at 10.85 pJ per flip (`ledger::KV260_MEASURED`), about
// 1.5e4 times cheaper per flip than the CPU rows in `joules_per_sample`. Per flip, the fabric wins
// by four orders of magnitude at every temperature, and that is the number the field publishes.
//
// A flip is not a sample. What a sampler is bought for is INDEPENDENT draws, and the fabric runs
// a single-temperature chromatic Gibbs kernel whose autocorrelation time on a spin glass diverges
// as the glass freezes, while a CPU can run a tuned parallel-tempering ladder that keeps producing
// independent cold draws after single-temperature Gibbs has stopped moving. So the question with
// an answer is:
//
//   at what inverse temperature beta* does the fabric, priced at its MEASURED per-flip cost, stop
//   being cheaper PER INDEPENDENT SAMPLE than a CPU parallel-tempering ladder priced at its
//   METERED cost on this machine?
//
// If beta* is below the temperatures sampling hardware is pitched for, the figure of merit ranks
// the two machines backwards exactly where it matters. If the fabric wins everywhere, the four
// orders of magnitude survive the honest denominator on real silicon.
//
// THE THREE ARMS, and what each number is:
//
//   F  the fabric.  `hdl::FixedFabric` is the cycle-exact emulator of the RTL that was metered --
//      Q.8 weights, a 1,024-entry sigmoid ROM, one xorshift32 per node, two colour phases per sweep.
//      Its autocorrelation time is measured by running it; its joules are PRICED, not metered:
//      flips x `KV260_MEASURED.e_sample`. The emulator's own CPU joules are irrelevant and are not
//      reported. The KV260 measurement exercised no reads, so the readout term is UNSTATED, and a
//      sensitivity row is printed with one Z1-class SPICE read charged per recorded draw, labelled
//      as the projection it is.
//   P  the CPU ladder.  `adaptive::adapt` tunes a parallel-tempering ladder down to `beta`, the
//      replica count escalates until the worst pair is alive, and `parallel_tempering_observed`
//      runs under a real wattmeter (`ferrotherm_meter::Meter`) after `host::require_quiet`. Its
//      joules are joules above the machine's measured idle.
//   G  the control.  f64 chromatic Gibbs on the CPU at the same beta, metered. It is the same
//      algorithm as the fabric in a different arithmetic and a different RNG: its tau must agree
//      with the emulator's within the tau error, or the emulator's tau is not to be trusted and
//      the row says so. It also supplies the CPU J/flip that makes "1.5e4x per flip" a measured
//      ratio on this machine rather than a quoted one.
//
// WHAT IS REFUSED. An ESS below `MIN_ESS` on an arm whose figure is being compared is refused,
// because tau's relative error is about 1/sqrt(ESS). A sampler that produced LESS THAN ONE
// independent draw is NOT refused: its J/independent-sample is at least the whole run's joules,
// which is a bound and an answer. The fabric arm escalates its own sweep budget, since it is cheap
// to run long; the CPU arm escalates its rounds as `joules_per_sample` does.
//
// run: cargo run --release -p ferrotherm-meter --example fabric_vs_cpu
//      (build first, let the machine settle, then run the binary directly -- see the refusal text)

use ferrotherm::certify::{tau_int, tau_int_batch};
use ferrotherm::graph::{Graph, GraphBuilder};
use ferrotherm::gibbs::Sampler;
use ferrotherm::hdl::FixedFabric;
use ferrotherm::host;
use ferrotherm::ledger::{KV260_MEASURED, Z1_SPICE};
use ferrotherm::rng::Pcg;
use ferrotherm::tempering::parallel_tempering_observed;

/// The metered configuration: a 32x32 periodic lattice, 1,024 nodes, random +-1 couplings, no
/// fields. Rebuilt edge by edge from `ising::lattice2d` in that function's own insertion order,
/// so the greedy colouring lands on the same two classes the fabric requires.
fn lattice_glass(l: usize, seed: u64) -> Graph {
    let base = ferrotherm::ising::lattice2d(l, 1.0);
    let mut rng = Pcg::new(seed, 0x9E37);
    let mut b = GraphBuilder::new(base.n);
    // lattice2d couples (i, right) then (i, down) for each i in order; walking the CSR of a graph
    // built that way in (i, j > i) order is NOT the same order, and a different order can cost the
    // two-colouring. So reproduce the constructor's order directly.
    for y in 0..l {
        for x in 0..l {
            let i = y * l + x;
            let right = y * l + (x + 1) % l;
            let down = ((y + 1) % l) * l + x;
            let jr = if rng.f64() < 0.5 { -1.0 } else { 1.0 };
            let jd = if rng.f64() < 0.5 { -1.0 } else { 1.0 };
            b.couple(i, right, jr);
            b.couple(i, down, jd);
        }
    }
    b.build()
}

/// Below this an effective sample size cannot support a comparison; tau's relative error is about
/// `1/sqrt(ESS)`, so 25 is +-20%.
const MIN_ESS: f64 = 25.0;

struct Arm {
    label: &'static str,
    flips: f64,
    /// Joules charged to this arm: metered above idle for the CPU arms, priced for the fabric.
    joules: f64,
    /// Integrated autocorrelation time in COLD-REPLICA SWEEPS, the one unit every arm shares:
    /// the LARGER of Sokal's window and batch means, as `certify()` carries it.
    tau: f64,
    /// Sokal's automatic window, same unit. Where `tau_batch` is more than twice this, the window
    /// closed on a fast mode and the row's tau is a lower bound.
    tau_sokal: f64,
    /// Batch means over twenty batches, same unit.
    tau_batch: f64,
    draws: usize,
    /// How the joules were obtained, printed beside every figure derived from them.
    basis: &'static str,
}

impl Arm {
    fn ess(&self) -> f64 {
        self.draws as f64 / (2.0 * self.tau)
    }
    fn j_per_flip(&self) -> f64 {
        self.joules / self.flips
    }
    fn j_per_independent(&self) -> f64 {
        self.joules / self.ess()
    }
    fn tau_rel_err(&self) -> f64 {
        1.0 / self.ess().sqrt()
    }
    fn cell(&self) -> String {
        if self.ess() < 1.0 {
            format!(">={:.3e}", self.joules)
        } else if self.ess() < MIN_ESS {
            format!("unres@{:.0}", self.ess())
        } else {
            format!("{:.3e}", self.j_per_independent())
        }
    }
}

fn spins(f: &FixedFabric) -> Vec<i8> {
    f.s.iter().map(|&b| if b { 1 } else { -1 }).collect()
}

/// The fabric arm: run the emulator, escalate the budget until its own tau resolves or the cap
/// says how far it got. Priced, not metered.
fn fabric_arm(g: &Graph, beta: f64, seed: u64) -> Arm {
    let n = g.n;
    let mut sweeps = 20_000usize;
    let cap = 2_560_000usize;
    loop {
        let mut f = FixedFabric::new(g, beta, seed);
        let burn = sweeps / 10;
        for _ in 0..burn {
            f.sweep();
        }
        let mut trace = Vec::with_capacity(sweeps);
        for _ in 0..sweeps {
            f.sweep();
            trace.push(g.energy(&spins(&f)));
        }
        let (ts, tb) = (tau_int(&trace), tau_int_batch(&trace, 20));
        let arm = Arm {
            label: "fabric (emulated, priced)",
            flips: (n * (burn + sweeps)) as f64,
            joules: (n * (burn + sweeps)) as f64 * KV260_MEASURED.e_sample,
            tau: if tb.is_finite() { ts.max(tb) } else { ts },
            tau_sokal: ts,
            tau_batch: tb,
            draws: trace.len(),
            basis: "KV260_MEASURED.e_sample x flips",
        };
        if arm.ess() >= MIN_ESS || sweeps >= cap || arm.ess() < 1.0 && sweeps >= cap / 4 {
            return arm;
        }
        let want = ((MIN_ESS / arm.ess().max(0.25)) * 1.3).ceil() as usize;
        sweeps = (sweeps * want.clamp(2, 8)).min(cap);
    }
}

fn refuse(e: &str) -> ! {
    eprintln!("\n  this run could not be priced: {e}");
    eprintln!("  Build first, let the machine settle, and run the compiled binary directly:");
    eprintln!("    cargo build --release -p ferrotherm-meter --example fabric_vs_cpu");
    eprintln!("    ./target/release/examples/fabric_vs_cpu");
    std::process::exit(2);
}

/// The two CPU arms at one temperature, iso-flip with the ladder, both metered.
fn cpu_arms(
    meter: &mut ferrotherm_meter::Meter,
    idle: ferrotherm_meter::Baseline,
    g: &Graph,
    beta: f64,
    betas: &[f64],
    rounds: usize,
) -> (Arm, Arm, usize) {
    let n = g.n;
    let rungs = betas.len();
    let burn_in = rounds / 10;

    // G. f64 chromatic Gibbs with the ladder's whole flip budget: one draw per `rungs` sweeps,
    // tau converted back to sweeps so every arm's tau is the same quantity.
    let sweeps_g = rounds * rungs;
    let mut trace_g: Vec<f64> = Vec::with_capacity(rounds);
    let run_g = meter
        .measure(idle, || {
            let mut smp = Sampler::new(g, beta, 0xA11CE);
            for r in 0..sweeps_g {
                smp.sweep(None);
                if r % rungs == 0 && r / rungs >= burn_in {
                    trace_g.push(g.energy(&smp.s));
                }
            }
        })
        .unwrap_or_else(|e| refuse(&e));
    let (ts, tb) = (tau_int(&trace_g) * rungs as f64, tau_int_batch(&trace_g, 20) * rungs as f64);
    let gibbs = Arm {
        label: "CPU Gibbs f64 (metered)",
        flips: (n * sweeps_g) as f64,
        joules: run_g.joules_above_idle,
        tau: if tb.is_finite() { ts.max(tb) } else { ts },
        tau_sokal: ts,
        tau_batch: tb,
        draws: trace_g.len(),
        basis: "wattmeter, above idle",
    };

    // P. parallel tempering on the tuned ladder; the cold trace is one draw per cold sweep.
    let mut out = None;
    let run_p = meter
        .measure(idle, || {
            out = Some(parallel_tempering_observed(g, betas, rounds, 1, burn_in, 0xB0B, None));
        })
        .unwrap_or_else(|e| refuse(&e));
    let (_res, tr) = out.expect("the closure ran");
    let cold = tr.energies.last().expect("a ladder has rungs");
    let (ts, tb) = (tau_int(cold), tau_int_batch(cold, 20));
    let pt = Arm {
        label: "CPU parallel tempering (metered)",
        flips: (n * rounds * rungs) as f64,
        joules: run_p.joules_above_idle,
        tau: if tb.is_finite() { ts.max(tb) } else { ts },
        tau_sokal: ts,
        tau_batch: tb,
        draws: cold.len(),
        basis: "wattmeter, above idle",
    };
    (gibbs, pt, tr.round_trips)
}

fn main() {
    let quiet = match host::require_quiet("a power and timing measurement") {
        Ok(q) => q,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(3);
        }
    };
    let Some(mut meter) = ferrotherm_meter::Meter::detect() else {
        eprintln!("no power backend on this machine; nothing to report and no modelled number in its place.");
        std::process::exit(2);
    };

    let l = 32usize;
    let g = lattice_glass(l, 0xC0FFEE);
    let n = g.n;
    if g.classes.len() != 2 {
        eprintln!("the lattice glass coloured to {} classes; the v1 fabric needs exactly 2", g.classes.len());
        std::process::exit(2);
    }
    let seed_f = 0xFAB;
    let rounds_start = 8_000usize;
    let rounds_cap = 256_000usize;

    println!("THE FABRIC'S FLIPS, PRICED PER INDEPENDENT SAMPLE, AGAINST A METERED CPU LADDER");
    println!("  graph      {l}x{l} periodic lattice, {n} nodes, random +-1 couplings, no fields (the KV260 configuration)");
    println!("  fabric     hdl::FixedFabric, Q.{} weights, {}-entry ROM, xorshift32 per node -- priced at {:.3e} J/flip",
             ferrotherm::hdl::FRAC, 1usize << ferrotherm::hdl::LUT_BITS, KV260_MEASURED.e_sample);
    println!("  machine    {}  (CPU arms metered above idle)", meter.machine());
    println!("  budget     fabric escalates its own sweeps until tau resolves; CPU ladder from {rounds_start} rounds to at most {rounds_cap}");

    let idle = match meter.idle(std::time::Duration::from_secs(3)) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("no usable idle baseline: {e}");
            std::process::exit(2);
        }
    };
    println!(
        "  idle       {:.2} W +- {:.2} over {} readings, load1 {}\n",
        idle.watts,
        idle.sigma,
        idle.samples,
        idle.load1.map_or("unknown".to_string(), |l| format!("{l:.2}"))
    );

    println!(
        "  {:>5} {:>6}   {:>10} {:>10}   {:>11} {:>11} {:>11}   {:>9} {:>9}  cheaper/indep",
        "beta", "rungs", "J/flip F", "J/flip CPU", "J/ind F", "J/ind PT", "J/ind Gibbs", "F/PT flip", "F/PT ind"
    );
    let mut evidence: Vec<String> = Vec::new();
    let mut crossings: Vec<(f64, f64)> = Vec::new();

    for &beta in &[0.5f64, 0.75, 1.0, 1.5, 2.0, 3.0] {
        // ---- arm F, no meter needed ------------------------------------------------------------
        let f = fabric_arm(&g, beta, seed_f);

        // ---- the ladder is tuned and escalated until its worst pair is alive -----------------
        let mut betas = Vec::new();
        let mut worst = 0.0f64;
        for replicas in [8usize, 16, 24, 48] {
            let p = ferrotherm::adaptive::Params {
                replicas,
                epochs: 5,
                rounds: 300,
                swap_every: 1,
                beta_min: 0.2_f64.min(beta / 2.0),
                beta_max: beta,
            };
            let out = ferrotherm::adaptive::adapt(&g, &p, 0x1AD);
            worst = out.swap_rates.iter().copied().fold(f64::INFINITY, f64::min);
            betas = out.betas;
            if worst >= 0.10 {
                break;
            }
        }
        let rungs = betas.len();
        if worst < 0.05 {
            println!("  {beta:>5.2} {rungs:>6}   ladder never came alive (worst pair {worst:.3}); CPU arms not priced");
            continue;
        }

        // ---- arms G and P, metered, escalated until PT resolves ------------------------------
        let mut rounds = rounds_start;
        let (gibbs, pt, round_trips) = loop {
            let (gi, p, rt) = cpu_arms(&mut meter, idle, &g, beta, &betas, rounds);
            if p.ess() >= MIN_ESS || rounds >= rounds_cap {
                break (gi, p, rt);
            }
            let want = ((MIN_ESS / p.ess().max(0.25)) * 1.3).ceil() as usize;
            rounds = (rounds * want.clamp(2, 8)).min(rounds_cap);
        };

        // ---- the control: does the emulator's tau agree with f64 Gibbs's? ---------------------
        let control = if f.ess() >= 1.0 && gibbs.ess() >= 1.0 {
            let rel = (f.tau - gibbs.tau).abs() / f.tau.max(gibbs.tau);
            let allow = 2.0 * (f.tau_rel_err().min(1.0) + gibbs.tau_rel_err().min(1.0));
            if rel <= allow { "agree" } else { "DISAGREE" }
        } else if f.ess() < 1.0 && gibbs.ess() < 1.0 {
            "both frozen"
        } else {
            "one frozen"
        };

        // ---- verdict ---------------------------------------------------------------------------
        let verdict = if pt.ess() < MIN_ESS {
            "PT unresolved".to_string()
        } else if f.ess() < 1.0 {
            format!("PT (fabric produced <1 indep. draw; F >= {:.2e} J)", f.joules)
        } else if f.ess() < MIN_ESS {
            format!("PT? (fabric ESS {:.0}, unresolved)", f.ess())
        } else if f.j_per_independent() < pt.j_per_independent() {
            "fabric".to_string()
        } else {
            "PT".to_string()
        };
        let flip_ratio = f.j_per_flip() / gibbs.j_per_flip();
        let ind_ratio = if f.ess() >= MIN_ESS && pt.ess() >= MIN_ESS {
            format!("{:.3e}", f.j_per_independent() / pt.j_per_independent())
        } else if f.ess() < 1.0 && pt.ess() >= MIN_ESS {
            format!(">={:.2e}", f.joules / pt.j_per_independent())
        } else {
            "-".to_string()
        };
        // The per-flip price at which the fabric would TIE the ladder per independent sample.
        let tie = if pt.ess() >= MIN_ESS && f.ess() >= 1.0 {
            Some(pt.j_per_independent() * f.ess() / f.flips)
        } else if pt.ess() >= MIN_ESS {
            Some(pt.j_per_independent() / f.flips) // ESS < 1: the price at which one whole run ties
        } else {
            None
        };
        println!(
            "  {beta:>5.2} {rungs:>6}   {:>10.3e} {:>10.3e}   {:>11} {:>11} {:>11}   {:>9.2e} {:>9}  {verdict}",
            f.j_per_flip(),
            gibbs.j_per_flip(),
            f.cell(),
            pt.cell(),
            gibbs.cell(),
            flip_ratio,
            ind_ratio
        );
        if let Some(t) = tie {
            crossings.push((beta, t));
        }
        // Sensitivity: charge one Z1-class read per recorded fabric draw. Unstated on the KV260,
        // so this is a projection, and it is printed as one.
        let f_with_reads = f.joules + f.draws as f64 * Z1_SPICE.e_read;
        evidence.push(format!(
            "beta {beta}: {} tau {:.1} sweeps, ESS {:.1} (+-{:.0}% on tau) over {} draws, {:.3e} J priced \
             [{}]; +1 Z1_SPICE read/draw would make it {:.3e} J. {} tau {:.1}, ESS {:.1} (+-{:.0}%), \
             {:.3} J above idle [{}]; control {control}. {} {rungs} rungs, {rounds} rounds, {round_trips} round \
             trips, tau {:.1}, ESS {:.1} (+-{:.0}%), {:.3} J above idle. Idle drift +-{:.3} J over the longer arm.{}",
            f.label, f.tau, f.ess(), 100.0 * f.tau_rel_err(), f.draws, f.joules, f.basis, f_with_reads,
            gibbs.label, gibbs.tau, gibbs.ess(), 100.0 * gibbs.tau_rel_err(), gibbs.joules, gibbs.basis,
            pt.label, pt.tau, pt.ess(), 100.0 * pt.tau_rel_err(), pt.joules,
            idle.sigma * (pt.flips / n as f64 / 1e6).max(1.0),
            tie.map_or(String::new(), |t| format!(" Per-flip price at which the fabric would TIE PT per independent sample: {t:.3e} J/flip ({:.1e}x the measured price).", t / KV260_MEASURED.e_sample))
        ));
        // The window cross-check, per arm: Sokal against batch means, and whether the row's tau
        // came from the batch side because the window closed early. A truncated arm's tau -- and
        // every J/ind built on it -- is a lower bound on the cost, not a value.
        let window = |a: &Arm| -> String {
            let flag = if a.tau_batch.is_finite() && a.tau_batch > 2.0 * a.tau_sokal { " TRUNCATED" } else { "" };
            format!("{} Sokal {:.1} / batch {:.1}{flag}", a.label, a.tau_sokal, a.tau_batch)
        };
        evidence.push(format!("    windows: {}; {}; {}", window(&f), window(&gibbs), window(&pt)));
    }

    println!("\n  WHAT THE TABLE SAYS.\n");
    println!("  J/flip: the fabric's measured price against this CPU's measured price, the ratio the");
    println!("  field publishes. J/ind: the same joules divided by effective samples. Where the");
    println!("  verdict column changes from `fabric` to `PT`, the figure of merit has ranked the two");
    println!("  machines backwards, and the row's `F/PT ind` is by how much.");
    if let Some((b, t)) = crossings.iter().find(|(_, t)| *t < KV260_MEASURED.e_sample) {
        println!("\n  The first temperature at which the fabric's own flips cost more per independent sample");
        println!("  than the metered ladder is beta = {b}; there, a fabric would have to flip at");
        println!("  {t:.3e} J/flip -- {:.1e}x below the measured KV260 price -- to tie.", KV260_MEASURED.e_sample / t);
    } else {
        println!("\n  No row crossed: the fabric stayed cheaper per independent sample at every resolved beta.");
    }
    println!("\n  PER-POINT EVIDENCE:");
    for e in &evidence {
        println!("    {e}");
    }
    println!("    tau_int's relative error is about 1/sqrt(ESS); every J/ind inherits it. `>=X` means the");
    println!("    arm produced LESS THAN ONE independent draw in its budget, so its whole joules is a lower");
    println!("    bound on the cost of its first one. The fabric's joules are PRICED at a measured per-flip");
    println!("    cost with reads UNSTATED; the CPU arms are METERED above idle on this machine.");
    if let Some(c) = quiet.caveat() {
        println!("\n  {c}");
    }
}
