//! The bill for being switched on -- and the hardware spec it implies for anyone who wants to win.
//!
//! `joules.rs` measures both paths ABOVE IDLE, which is the right question for a machine kept busy.
//! Most places a sampling substrate would actually go do not keep a machine busy. A sensor drawing
//! from a posterior a few times a second computes for microseconds and waits out the rest, and
//! subtracting idle prices the microseconds and discards the wait -- which is where the joules are.
//!
//! So this asks the other question, on the same two paths, with the idle term left in. It then
//! inverts it into the number a challenger has to hit:
//!
//!   standby budget = idle + marginal * duty
//!
//! That bound already grants the challenger perfectly FREE computation, so no better sampler can
//! argue it down. As the cadence slackens it collapses to the incumbent's idle draw, and the entire
//! case for a thermodynamic fabric reduces to one number about its own standby power -- not to
//! joules per flip, not to throughput, not to physics.
//!
//! Which is where the field stops being able to answer. `ledger::Z1_SPICE` carries e_sample, e_read,
//! e_write and a reflash cap, because those are what Table IV states. There is no standby term
//! because no thermodynamic vendor publishes one. This example therefore prints the budget and
//! stops, rather than filling the gap with a number nobody measured.
//!
//! PREDICTED before measuring, recorded here so it cannot be adjusted afterwards: the GPU's
//! idle-dominance threshold lands above one percent, meaning any workload running less than a
//! percent of the time spends most of its energy idle and cannot be helped by a faster sampler.
//! Falsified if the threshold comes in under 0.1 percent, which would mean idle is negligible and
//! this whole line of argument is void.

use ferrotherm::{duty::Machine, gibbs::Sampler, ising::lattice2d, ledger::Ledger, wgsl::GpuModel};
use std::time::Duration;

/// Application cadences worth pricing, as (label, seconds between task repetitions).
const CADENCES: &[(&str, f64)] = &[
    ("continuous", 0.0), // replaced by the measured run time: duty = 1
    ("100 Hz control", 0.01),
    ("10 Hz sensing", 0.1),
    ("1 Hz telemetry", 1.0),
    ("once a minute", 60.0),
    ("once an hour", 3600.0),
];

fn main() {
    let Some(gpu) = ferrotherm_gpu::Gpu::new() else {
        eprintln!("no GPU adapter; nothing to compare");
        return;
    };
    let Some(mut meter) = ferrotherm_meter::Meter::detect() else {
        eprintln!("no power backend; this example is a measurement and cannot be simulated");
        return;
    };
    println!("adapter : {} ({:?})", gpu.adapter().name, gpu.adapter().device_type);
    println!("machine : {}", meter.machine());
    if !gpu.is_hardware() {
        println!("\nSoftware rasteriser. Numbers below describe a CPU twice over; not reported as a ratio.");
    }
    println!("\nsettling...");
    std::thread::sleep(Duration::from_secs(5));

    // 1024, not 512, and the reason is the CPU path rather than the GPU's.
    //
    // `sweeps_par` spawns its threads inside each sweep, so at 512x512 the 200-sweep pass spends
    // most of its time in 3,600 spawn/join pairs and delivers 2.1x on 18 cores -- a machine that
    // is not working does not draw power, and the first run of this example was refused by the
    // meter at 0.13 W above a baseline wandering by 5.86 W. At 1024x1024 there is enough work per
    // sweep to amortise the spawn and the same call delivers 4.9x, which the instrument can see.
    // Measured, not guessed: 7.09e7 -> 1.48e8 at 512, and 1.16e8 -> 5.65e8 at 1024.
    let l = 1024usize;
    let g = lattice2d(l, 1.0);
    let n = (l * l) as u64;
    let sweeps = 200u32;
    let window = Duration::from_secs(4);
    println!("model   : {l}x{l} = {n} nodes; each path repeats {sweeps}-sweep passes for {:.0} s\n",
             window.as_secs_f64());

    // Same discipline as joules.rs: cool down BETWEEN paths, take a long baseline, and let the
    // meter refuse a run it cannot see rather than dividing by a window that was mostly idle.
    let mut row = |label: &str, pass: &mut dyn FnMut()| -> Option<Machine> {
        std::thread::sleep(Duration::from_secs(10));
        let idle = match meter.idle(Duration::from_secs(3)) {
            Ok(w) => w,
            Err(e) => { eprintln!("{label}: {e}"); return None; }
        };
        let mut passes = 0u64;
        let m = match meter.measure(idle, || {
            let t0 = std::time::Instant::now();
            while t0.elapsed() < window {
                pass();
                passes += 1;
            }
        }) {
            Ok(m) => m,
            Err(e) => { eprintln!("{label}: {e}"); return None; }
        };
        let updates = passes * n * sweeps as u64;
        let rate = updates as f64 / m.seconds;
        let marginal = m.mean_watts - m.idle_watts;
        println!(
            "  {label:<4}: {updates} updates in {:.2} s -> {rate:.3e}/s; idle {:.1} W (sigma {:.2}), \
             marginal {marginal:.1} W",
            m.seconds, m.idle_watts, m.idle_sigma
        );
        match Machine::new(m.idle_watts, marginal, rate) {
            Ok(machine) => Some(machine),
            Err(e) => { eprintln!("  {label}: {e}"); None }
        }
    };

    let gm = GpuModel::from_graph(&g);
    let mut spins = vec![1i8; n as usize];
    gpu.sweep(&gm, &mut spins, 0.7, 1).unwrap(); // warm the pipeline, off the clock
    let g_m = row("gpu", &mut || { gpu.sweep(&gm, &mut spins, 0.7, sweeps).unwrap(); });

    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let mut led = Ledger::default();
    let c_m = row("cpu", &mut || {
        let mut s = Sampler::new(&g, 0.7, 1);
        s.sweeps_par(sweeps as usize, threads, Some(&mut led));
    });

    // Report every path the instrument COULD see. An earlier version returned early unless both
    // measured, which threw away a perfectly good GPU measurement because the CPU's delta sat in
    // the noise -- and the finding here is a property of one machine at a time, not of the ratio.
    let paths: Vec<(&str, Machine)> =
        [("gpu", g_m), ("cpu", c_m)].into_iter().filter_map(|(k, v)| v.map(|m| (k, m))).collect();
    if paths.is_empty() {
        eprintln!("\nno path could be measured; each one's reason is printed above");
        return;
    }
    if paths.len() == 1 {
        println!("\n  Only the {} path was measurable; what follows describes it alone.", paths[0].0);
    }

    // A task worth pricing: one 512x512 lattice sampled for 200 sweeps, which is what a small
    // posterior draw looks like. Fixed across cadences so only the WAIT changes.
    let task_work = n * sweeps as u64;

    println!("\nIDLE-DOMINANCE THRESHOLD -- below this duty cycle, most of the bill is being switched on:");
    for (label, m) in &paths {
        let d = m.idle_dominant_below();
        println!("  {label}: {:.4}%  (idle {:.1} W vs marginal {:.1} W)",
                 100.0 * d, m.idle_watts, m.marginal_watts);
        if d < 0.001 {
            println!("       ** under 0.1% -- the prediction is FALSIFIED on this path **");
        }
    }

    println!("\nONE TASK ({task_work} node updates), priced at each cadence.");
    println!("  'above idle' is what joules.rs reports; 'total' leaves the wait in.\n");
    println!("  {:<16} {:>10} {:>13} {:>13} {:>10} {:>14}",
             "cadence", "duty", "above idle", "total", "understated", "standby budget");
    for &(label, period) in CADENCES {
        for (who, m) in &paths {
            let period = if period == 0.0 { m.run_seconds(task_work) } else { period };
            let (Ok(d), Ok(total), Ok(budget)) = (
                m.duty(task_work, period),
                m.joules_per_period(task_work, period),
                m.standby_budget(task_work, period),
            ) else {
                println!("  {:<16} {who}: cannot sustain this cadence", label);
                continue;
            };
            let above = m.marginal_watts * m.run_seconds(task_work);
            // The ratio is the point of the table: how far the above-idle figure -- the one this
            // whole field reports -- sits from what the machine actually spent.
            println!("  {:<12} {who}  {:>9.4}% {:>11.4} J {:>11.4} J {:>9.0}x {:>12.2} W",
                     label, 100.0 * d, above, total, total / above, budget);
        }
    }

    println!("\nHOW TO READ THE LAST COLUMN.");
    println!("  It is the standby power a competing device must come in UNDER to be cheaper on this");
    println!("  task at that cadence, GRANTING IT FREE COMPUTATION. A better sampler cannot argue it");
    println!("  down, because the bound already assumes a perfect one.");
    println!();
    println!("  Notice what happens down the column: as the cadence slackens the budget stops moving");
    println!("  and settles on the idle draw. At that point nothing about sampling matters, and the");
    println!("  whole thermodynamic value proposition is one number -- standby power.");
    println!();
    println!("  WHICH COMPARISON THIS IS, because the budget means different things in two cases.");
    println!("  The meter reads WHOLE-SYSTEM power, so the idle term above is the entire machine. That");
    println!("  prices the case where the fabric REPLACES the host -- a standalone device that answers");
    println!("  the query itself. It is the generous case, and the bar looks easy because of it.");
    println!();
    println!("  The other case is a fabric sitting BESIDE a host that has to stay awake regardless. In");
    println!("  that arrangement the host's idle is common to both sides and cancels, and what remains");
    println!("  is the accelerator's own standby against the GPU's own contribution to idle. This");
    println!("  machine cannot report that: an integrated SoC shares one power rail, so per-component");
    println!("  idle is not separable here. Saying so beats splitting 67 W by assumption.");
    println!();
    println!("  That number is not published. ledger::Z1_SPICE states e_sample, e_read, e_write and a");
    println!("  reflash cap because Table IV states them; it carries no standby term because no");
    println!("  thermodynamic vendor has published one. So the device column of this table cannot be");
    println!("  filled in by anybody today, and the field's strongest argument -- the intermittent,");
    println!("  low-duty edge workload -- is the one it has never priced.");
}
