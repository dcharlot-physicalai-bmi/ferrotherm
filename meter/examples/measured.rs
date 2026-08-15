//! What a node update costs on THIS machine, measured, beside what it was being priced at.
//!
//! The ledger has always counted operations exactly. Until now the only price table in the tree was
//! Z1_SPICE -- pre-silicon estimates for an accelerator nobody has characterised -- and every
//! fabric declared it, so a laptop reported another company's unfabricated chip's energy. This
//! replaces the borrowed number with a measured one and prints both, because the ratio is the
//! interesting part and neither number means much alone.

use ferrotherm::{gibbs::Sampler, ising::lattice2d, ledger::{Ledger, Z1_SPICE}};
use std::time::Duration;

fn main() {
    let Some(mut meter) = ferrotherm_meter::Meter::detect() else {
        eprintln!("no power backend on this machine (macmon not on PATH); nothing to measure");
        return;
    };
    println!("machine: {}", meter.machine());

    // The build that produced this binary leaves the machine warm, so the first baseline taken
    // after it reads high -- measured at 66 W against a 56 W workload, which the library refuses
    // rather than reporting as negative energy. Let it settle.
    println!("settling...");
    std::thread::sleep(Duration::from_secs(5));

    // Repeats, because one measurement of a fluctuating quantity is not an estimate of it.
    let g = lattice2d(256, 1.0);
    let mut per_sample = Vec::new();
    for r in 1..=5 {
        // Re-baseline before every run. Idle drifts with temperature and with whatever else the
        // machine is doing, and a baseline measured once at the top is stale by the third run --
        // which is exactly what `Meter::idle`'s own documentation says to do about it.
        let idle = match meter.idle(Duration::from_millis(1500)) {
            Ok(w) => w,
            Err(e) => { eprintln!("{e}"); return; }
        };
        let mut led = Ledger::default();
        let run = match meter.measure(idle, || {
            let mut s = Sampler::new(&g, 0.7, r);
            s.sweeps(3_000, Some(&mut led));
        }) {
            Ok(r) => r,
            Err(e) => { eprintln!("run {r}: {e}"); return; }
        };
        let p = run.prices_from(&led).expect("a pure sampling run is sample-dominated");
        println!(
            "  run {r}: {:.1} W idle -> {:.1} W for {:.2} s ({:.1} J above idle, {} readings) \
             over {} updates = {:.3e} J/update",
            run.idle_watts, run.mean_watts, run.seconds, run.joules_above_idle, run.samples,
            led.samples, p.e_sample
        );
        per_sample.push(p.e_sample);
    }

    per_sample.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = per_sample[per_sample.len() / 2];
    println!("\nmedian measured : {median:.3e} J per node update, on this machine, at the wall");
    println!("Z1_SPICE claims : {:.3e} J per node update", Z1_SPICE.e_sample);
    println!("ratio           : {:.0}x", median / Z1_SPICE.e_sample);
    println!();
    println!("Both numbers are real and they measure different things. The measured one is whole-");
    println!("system wall power above idle divided by node updates, on a general-purpose CPU doing");
    println!("everything a CPU does to run a sweep. The Z1 figure is a SPICE estimate for a");
    println!("purpose-built device that has not been fabricated. The ratio is the size of the prize");
    println!("being claimed, not a measurement of it -- and now at least one side of it is measured.");
}
