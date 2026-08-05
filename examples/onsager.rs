// 2D Ising vs Onsager's exact solution — the sampler against 1944.
//
// Spontaneous magnetization on the nearest-neighbor square lattice has a closed form
// (Onsager 1944 / Yang 1952): M = (1 - sinh(2*beta)^-4)^(1/8) below beta_c ~ 0.4407, 0 above.
// A sampler that cannot reproduce this has no business simulating anything else.
//
// run: cargo run --release --example onsager

use ferrotherm::gibbs::Sampler;
use ferrotherm::ising::{lattice2d, onsager_m};

fn main() {
    let l = 64usize;
    let g = lattice2d(l, 1.0);
    println!("2D Ising {l}x{l}, periodic, J=1 — |M| sampled vs Onsager exact (infinite lattice)");
    println!("  {:>7} {:>12} {:>12} {:>9}", "beta", "|M| sampled", "M exact", "|delta|");

    let mut worst: f64 = 0.0;
    for &(beta, burn, meas, tol) in &[
        // above Tc (beta < beta_c): M_exact = 0; finite-size |M| ~ O(1/L) fluctuation
        (0.35, 2_000usize, 4_000usize, 0.06f64),
        // below Tc: exact curve, tight tolerance
        (0.50, 4_000, 8_000, 0.01),
        (0.60, 4_000, 8_000, 0.01),
        (0.70, 4_000, 8_000, 0.01),
    ] {
        let mut smp = Sampler::new(&g, beta, 0x150D ^ (beta * 1000.0) as u64);
        // ordered start below Tc avoids domain-wall trapping; disordered above
        if beta > 0.45 {
            for s in smp.s.iter_mut() {
                *s = 1;
            }
        }
        smp.sweeps(burn, None);
        let mut acc = 0.0;
        for _ in 0..meas {
            smp.sweep(None);
            let m: i64 = smp.s.iter().map(|&v| v as i64).sum();
            acc += (m as f64 / g.n as f64).abs();
        }
        let m_sim = acc / meas as f64;
        let m_ex = onsager_m(beta);
        let d = (m_sim - m_ex).abs();
        worst = worst.max(d - tol);
        println!(
            "  {beta:>7.2} {m_sim:>12.4} {m_ex:>12.4} {d:>9.4}  {}",
            if d <= tol { "ok" } else { "OUT OF TOLERANCE" }
        );
    }
    let ok = worst <= 0.0;
    println!("  verdict: {}", if ok { "PASS — sampler reproduces the exact 2D Ising solution" } else { "FAIL" });
    std::process::exit(if ok { 0 } else { 1 });
}
