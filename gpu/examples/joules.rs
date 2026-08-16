//! Is the GPU also CHEAPER? Not faster -- cheaper.
//!
//! `bench.rs` measures wall time and reports 54x at 262k nodes. That is a throughput claim and it
//! answers a different question from the one this stack is built around: joules per unit of work.
//! A part that runs 54x faster while drawing 54x the power costs exactly the same, and a stack that
//! prices computation in joules should be able to say which happened.
//!
//! Both sides measured the same way -- whole-system wall power above idle, divided by the node
//! updates the ledger counted -- so the comparison is between two ways of doing one thing on one
//! machine, not between a measurement and a datasheet.

use ferrotherm::{gibbs::Sampler, ising::lattice2d, ledger::Ledger, wgsl::GpuModel};
use std::time::Duration;

fn main() {
    let Some(gpu) = ferrotherm_gpu::Gpu::new() else {
        eprintln!("no GPU adapter; nothing to compare");
        return;
    };
    let Some(mut meter) = ferrotherm_meter::Meter::detect() else {
        eprintln!("no power backend; the timing comparison lives in bench.rs");
        return;
    };
    println!("adapter : {} ({:?})", gpu.adapter().name, gpu.adapter().device_type);
    println!("machine : {}", meter.machine());
    if !gpu.is_hardware() {
        println!("\nSoftware rasteriser. Numbers below describe a CPU twice over; not reported as a ratio.");
    }
    println!("\nsettling...");
    std::thread::sleep(Duration::from_secs(5));

    // REPEAT until the window is long enough, rather than sizing the model to fit the instrument.
    //
    // The first version ran each path once and both were refused. The GPU's was the instructive
    // one: 157M updates finish in about 30 ms, so the measurement window was almost entirely idle
    // and the run showed 1.26 W above a baseline that wanders by 1.10 W. Nothing was wrong with the
    // GPU or the meter -- you cannot measure something faster than your sampling interval by
    // running it once. Run it until the instrument can see it, and divide by the work done.
    let l = 512usize;
    let g = lattice2d(l, 1.0);
    let n = (l * l) as u64;
    let sweeps = 200u32;
    let window = Duration::from_secs(4);
    println!("model   : {l}x{l} = {n} nodes; each path repeats {sweeps}-sweep passes for {:.0} s\n",
             window.as_secs_f64());

    // A longer baseline: sigma is what the noise guard compares against, and a baseline measured
    // over 1.5 s on a machine with fans reports a spread that is mostly its own shortness.
    let mut row = |label: &str, pass: &mut dyn FnMut()| -> Option<(f64, f64, f64, u64)> {
        // Cool down BETWEEN paths, not just before the first. The CPU's baseline was first taken
        // straight after a four-second GPU burn and came back wandering by 7.2 W, which swallowed
        // the CPU's own 1.4 W signal. Two workloads measured back to back are not independent
        // measurements unless the machine is allowed to forget the first one.
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
        let per = m.joules_above_idle / updates as f64;
        println!(
            "  {label:<4}: {passes} passes ({updates} updates) in {:.2} s at {:.1} W over {:.1} W \
             idle = {:.1} J  ->  {per:.3e} J/update",
            m.seconds, m.mean_watts, m.idle_watts, m.joules_above_idle
        );
        Some((m.seconds, m.joules_above_idle, per, updates))
    };

    let gm = GpuModel::from_graph(&g);
    let mut spins = vec![1i8; n as usize];
    gpu.sweep(&gm, &mut spins, 0.7, 1).unwrap(); // warm the pipeline, off the clock
    let g_res = row("gpu", &mut || { gpu.sweep(&gm, &mut spins, 0.7, sweeps).unwrap(); });

    // ALL cores, not one.
    //
    // The first version called the single-threaded `sweeps`, which is unfair twice over. A GPU is
    // the whole device; comparing it against one core of eighteen overstates it by roughly the core
    // count, and that is the oldest way to flatter a GPU benchmark. It also made the measurement
    // impossible: one busy core added 0.13 W to a baseline wandering by 1.43 W, so the CPU's own
    // signal was inside the noise. Using the machine fairly and being able to measure it turn out
    // to be the same fix.
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    println!("  (cpu uses all {threads} cores; comparing a whole GPU to one core is not a comparison)\n");
    let mut led = Ledger::default();
    let c_res = row("cpu", &mut || {
        let mut s = Sampler::new(&g, 0.7, 1);
        s.sweeps_par(sweeps as usize, threads, Some(&mut led));
    });

    let (Some((_, gj, gp, gu)), Some((_, cj, cp, cu))) = (g_res, c_res) else {
        eprintln!("\nboth sides have to be measurable for the comparison to mean anything");
        return;
    };
    // The ledger counted the CPU's updates independently of the pass arithmetic above. If those
    // disagree, one of the two numbers in the ratio is not what it says it is.
    assert_eq!(led.samples, cu, "the ledger and the pass count must agree on the CPU's work");

    println!("\n  throughput : gpu {:.3e} vs cpu {:.3e} updates/s  ->  {:.1}x faster",
             gu as f64 / window.as_secs_f64(), cu as f64 / window.as_secs_f64(),
             gu as f64 / cu as f64);
    println!("  energy     : gpu {gj:.1} J for {gu} updates, cpu {cj:.1} J for {cu}  ->  {:.1}x cheaper per update",
             cp / gp);
    println!("  per op : gpu {gp:.3e} vs cpu {cp:.3e} J/update");
    println!();
    if cp / gp < gu as f64 / cu as f64 {
        println!("  The speedup is larger than the saving, so the GPU is drawing more power while it");
        println!("  works -- which is the expected shape. Time and joules are different questions and");
        println!("  a stack that prices in joules has to answer the second one on its own terms.");
    }
    println!("  Both sides: whole-system wall power above idle, divided by counted node updates, on");
    println!("  one machine. Neither is a datasheet.");
}
