// The CPU anchor for the impedance-tax measurement: chromatic Gibbs throughput at DIE SCALE on
// the published Z1 topology (269,568 nodes, degree 16), single thread, plain safe Rust.
//
// The energy comparison follows our E_compute convention: a platform pays watts x time, so
// joules-per-flip = platform watts / (flips per second). The Z1-class SPICE projection is
// 7.09 fJ per flip (arXiv:2608.01615 Table IV, pre-silicon). The honest academic anchor for
// physics-native hardware over OPTIMIZED digital samplers is 5-18x throughput (Aadit et al.,
// Nature Electronics 2022) — marketed multipliers beyond that come from naive baselines.
//
// run: cargo run --release --example flips_bench

use ferrotherm::device::z1_grid;
use ferrotherm::gibbs::Sampler;
use ferrotherm::host::{self, Quiet};
use ferrotherm::ledger::Z1_SPICE;
use std::time::Instant;

fn main() {
    // EVERY number below is a wall-clock time divided into something -- flips/s, ns/flip, and the
    // whole joules-per-flip table, which is watts over a rate. On a contended machine each of them
    // measures the run queue and looks exactly like a measurement of the code. So this stops rather
    // than annotates; `gset_gap` annotates instead, because a CUT is the same number either way.
    let quiet = match host::require_quiet("chromatic Gibbs throughput and the J/flip table") {
        Ok(q) => q,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(3);
        }
    };
    if let Some(c) = quiet.caveat() {
        println!("{c}\n");
    }
    // full die: 576 x 468 = 269,568 nodes (die dimensions unpublished; area matches the vendor's
    // stated pbit count)
    let (w, h) = (576usize, 468usize);
    println!("building Z1-topology grid {w} x {h} = {} nodes, degree 16...", w * h);
    let t0 = Instant::now();
    let g = z1_grid(w, h, 0.08, 0.0);
    println!("  built in {:.1} s; {} edges; {} color classes\n", t0.elapsed().as_secs_f64(), g.n_edges, g.classes.len());

    let mut smp = Sampler::new(&g, 0.9, 0xBE7C);
    // warmup
    smp.sweeps(3, None);

    let n_sweeps = 30usize;
    let t1 = Instant::now();
    smp.sweeps(n_sweeps, None);
    let dt = t1.elapsed().as_secs_f64();
    let flips = (n_sweeps * g.n) as f64;
    let fps = flips / dt;
    let ns_per_flip = 1e9 / fps;
    println!("single-thread CPU chromatic Gibbs, {} sweeps of {} nodes in {:.2} s:", n_sweeps, g.n, dt);
    println!("  {:.2e} flips/s   ({:.1} ns/flip)", fps, ns_per_flip);
    if let Quiet::Yes { load1: Some(l) } = quiet {
        println!("  (1-minute load average {l:.2} -- the machine was this code's)");
    }
    println!("  independent samples/s at K_mix = 250 sweeps: {:.2}\n", fps / (250.0 * g.n as f64));

    println!("joules per flip = platform watts / flips per second (our E_compute convention):");
    println!("  {:>22} {:>14} {:>22}", "platform assumption", "J/flip", "vs Z1 SPICE 7.09 fJ");
    for (label, watts) in [("5 W (embedded)", 5.0), ("15 W (Orin NX class)", 15.0), ("30 W (laptop pkg)", 30.0), ("60 W (desktop pkg)", 60.0)] {
        let jpf = watts / fps;
        println!("  {:>22} {:>11.1} pJ {:>18.0}x", label, jpf * 1e12, jpf / Z1_SPICE.e_sample);
    }
    println!("\nREADING: this is ONE CPU thread of portable safe Rust — the floor, not the ceiling. The browser");
    println!("WebGPU bench (gibbs_bench.html) measures the same sweep on whatever GPU the visitor owns; the");
    println!("Aadit 2022 measured anchor says optimized digital samplers sit within 5-18x of physics-native");
    println!("hardware on throughput, so the decisive comparison is measured-GPU vs the 7.09 fJ projection.");
}
