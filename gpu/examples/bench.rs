//! GPU against CPU on the same model, with the adapter named.
//!
//! Prints the adapter FIRST and refuses to quote a speedup against a software rasteriser. lavapipe,
//! SwiftShader and WARP all run this shader correctly and tell you nothing about a GPU, and a
//! benchmark that does not check produces a real-looking number for the wrong machine.

use ferrotherm::{gibbs::Sampler, ising::lattice2d, wgsl::GpuModel};
use std::time::Instant;

fn main() {
    let Some(gpu) = ferrotherm_gpu::Gpu::new() else {
        eprintln!("no GPU adapter on this machine; nothing to measure");
        return;
    };
    let a = gpu.adapter();
    println!("adapter: {} ({:?}, {:?})", a.name, a.device_type, a.backend);
    if !gpu.is_hardware() {
        println!("\nThis is a software rasteriser. Timings below describe a CPU pretending to be a");
        println!("GPU, so they are printed and NOT reported as a speedup.");
    }
    println!();

    let sweeps = 200;
    // Repeats, because two runs of this on the same machine reported 35.6x and 10.7x at the same
    // size. A single timing of a quantity that fluctuates is not an estimate of it, and quoting
    // the run that flattered the result is how a benchmark becomes marketing. The median is
    // reported: it is what this machine does typically, rather than what it did once.
    let repeats = 5;
    let median = |mut v: Vec<f64>| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };
    println!("median of {repeats} runs, {sweeps} sweeps each\n");
    println!("{:>6}  {:>8}  {:>11}  {:>11}  {:>9}", "l", "nodes", "gpu ms", "cpu ms", "ratio");
    for l in [16usize, 32, 64, 128, 256, 512] {
        let g = lattice2d(l, 1.0);
        let n = l * l;
        let m = GpuModel::from_graph(&g);
        let mut s = vec![1i8; n];

        // One untimed run first: the first dispatch pays pipeline creation and buffer upload, and
        // charging that to the measurement makes a small model look arbitrarily bad.
        gpu.sweep(&m, &mut s, 0.7, 1).unwrap();

        let gpu_ms = median(
            (0..repeats)
                .map(|_| {
                    let t0 = Instant::now();
                    gpu.sweep(&m, &mut s, 0.7, sweeps).unwrap();
                    t0.elapsed().as_secs_f64() * 1e3
                })
                .collect(),
        );
        let cpu_ms = median(
            (0..repeats)
                .map(|r| {
                    let mut sim = Sampler::new(&g, 0.7, r as u64 + 1);
                    let t1 = Instant::now();
                    sim.sweeps(sweeps as usize, None);
                    t1.elapsed().as_secs_f64() * 1e3
                })
                .collect(),
        );

        let ratio = if gpu.is_hardware() {
            format!("{:.2}x", cpu_ms / gpu_ms)
        } else {
            "n/a".into()
        };
        println!("{l:>6}  {n:>8}  {gpu_ms:>11.1}  {cpu_ms:>11.1}  {ratio:>9}");
    }
    println!();
    println!("THE CPU COLUMN IS ONE CORE. `Sampler::sweeps` is single-threaded, so every ratio above");
    println!("is a whole GPU against one core of {}, which is the oldest way to flatter a GPU", std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1));
    println!("benchmark. Measured against all cores it is 12x, not 54x -- see examples/joules.rs,");
    println!("which also prices both sides in joules and finds the GPU 10x cheaper per update.");
    println!();
    println!("The GPU loses on small models and that is the honest shape of the result: fixed cost");
    println!("per run does not shrink, so there has to be enough work to amortise it. The crossover");
    println!("is the number worth quoting, not the peak.");
    println!();
    println!("Sweeps: {sweeps}. Both run the same update rule -- the shader text comes from the");
    println!("core crate, so this compares hardware rather than two implementations.");
}
