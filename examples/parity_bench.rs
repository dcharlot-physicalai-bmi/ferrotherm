// Same-machine performance parity: ferrotherm CPU (1..threads) at full die scale, the numbers the
// THRML comparison script (scripts/thrml_bench.py) is measured against on identical hardware.
//
// run: cargo run --release --example parity_bench

use ferrotherm::device::z1_grid;
use ferrotherm::gibbs::Sampler;
use std::time::Instant;

fn main() {
    let side: Option<usize> = std::env::args().nth(1).and_then(|s| s.parse().ok());
    let (w, h) = side.map_or((576usize, 468usize), |s| (s, s));
    let g = z1_grid(w, h, 0.08, 0.0);
    println!("Z1-topology grid {} nodes, degree 16; chromatic Gibbs throughput on this machine:\n", g.n);
    println!("  {:>10} {:>14} {:>12} {:>10}", "threads", "flips/s", "ns/flip", "speedup");
    let mut base = 0.0;
    let cores = std::thread::available_parallelism().map(|c| c.get()).unwrap_or(8);
    for &t in &[1usize, 2, 4, 8, cores] {
        let mut smp = Sampler::new(&g, 0.9, 0xBE7C);
        // warmup
        if t == 1 { smp.sweeps(3, None); } else { smp.sweeps_par(3, t, None); }
        let n_sweeps = if t == 1 { 30 } else { 120 };
        let t0 = Instant::now();
        if t == 1 { smp.sweeps(n_sweeps, None); } else { smp.sweeps_par(n_sweeps, t, None); }
        let dt = t0.elapsed().as_secs_f64();
        let fps = (n_sweeps * g.n) as f64 / dt;
        if t == 1 { base = fps; }
        println!("  {:>10} {:>14.3e} {:>12.1} {:>9.1}x", t, fps, 1e9 / fps, fps / base);
    }
    println!("\n  browser WebGPU on this machine (measured separately, web/gibbs_bench.html): 9.35e9 flips/s");
    println!("  published FPGA anchor (Aadit et al., Nat. Electronics 2022, VCU118): 1.44e11 flips/s");
    println!("  Z1 SPICE projection (pre-silicon): 269,568 nodes x 2-color at 50 MHz ~ 6.7e12 flips/s");
}
