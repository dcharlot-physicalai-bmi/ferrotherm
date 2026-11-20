#![allow(missing_docs)]
//! Work, traffic, wall time and ESS of the open-grid sweep.
//!
//! Not a joule measurement. `bytes/sweep` counts spin-plane loads and stores the kernel
//! issues, plus the energy pass on the fraction of sweeps that keep a sample. `work/sweep`
//! counts site updates, 64-bit RNG words, and bond terms on that same fraction.
//!
//! ```text
//! cargo run --release --example grid_cost -- --sides 64,128,256 --sweeps 200 --warmup 200 --draws 1024
//! ```

use ferrotherm::grid::{Levers, OpenIsing, TILE_ROWS};
use ferrotherm::potts::estimate;
use std::time::Instant;

fn run_parity(threads: usize) {
    let onsager = ferrotherm::free_energy::onsager_energy_density(0.3, 1.0);
    println!("onsager_density {onsager:.6}  two seeds, combined, beta 0.3 J 1");
    println!(
        "{:>6} {:>8} {:>8} {:>12} {:>12} {:>10}  gate",
        "side", "warmup", "draws", "mean_a", "mean_b", "per_site"
    );
    for side in [8usize, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096] {
        let (warmup, draws) = if side == 8 { (400, 4096) } else { (200, 512) };
        let mut a = OpenIsing::try_new(side, 1.0, 0.3, 1, Levers::COMBINED, threads).expect("a");
        let mut b = OpenIsing::try_new(side, 1.0, 0.3, 2, Levers::COMBINED, threads).expect("b");
        let ea = a.energies(warmup, draws, 1);
        let eb = b.energies(warmup, draws, 1);
        let ma = ea.iter().sum::<f64>() / ea.len() as f64;
        let mb = eb.iter().sum::<f64>() / eb.len() as f64;
        let per_site = (ma - mb).abs() / (side * side) as f64;
        let density = ma / (side * side) as f64;
        let gap = (density - onsager).abs();
        let ok = per_site < 0.05;
        println!(
            "{side:>6} {warmup:>8} {draws:>8} {ma:>12.3} {mb:>12.3} {per_site:>10.5}  {ok} onsager_gap {gap:.5}"
        );
        assert!(ok, "side {side} two seeds {per_site}");
    }
}

fn time_sweeps(side: usize, levers: Levers, threads: usize, sweeps: usize) -> (f64, u64) {
    let mut g = OpenIsing::try_new(side, 1.0, 0.3, 0xC0FFEE, levers, threads).expect("grid");
    g.sweeps(4);
    let t0 = Instant::now();
    let prof = g.sweeps(sweeps);
    let ns = t0.elapsed().as_nanos() as f64 / sweeps as f64;
    (ns, prof.rng_words / sweeps as u64)
}

fn time_window(
    side: usize,
    levers: Levers,
    threads: usize,
    warmup: usize,
    draws: usize,
) -> (f64, f64, f64) {
    let mut a = OpenIsing::try_new(side, 1.0, 0.3, 1, levers, threads).expect("grid");
    let mut b = OpenIsing::try_new(side, 1.0, 0.3, 2, levers, threads).expect("grid");
    let t0 = Instant::now();
    let ea = a.energies(warmup, draws, 1);
    let sec = t0.elapsed().as_secs_f64();
    let eb = b.energies(warmup, draws, 1);
    let est = estimate(&ea);
    let sweeps = (warmup + draws) as f64;
    let ns = sec * 1e9 / sweeps;
    let agree = (est.value - estimate(&eb).value).abs() / (side * side) as f64;
    (ns, est.ess / draws as f64, agree)
}

fn main() {
    let mut sides = vec![64usize, 128, 256];
    let mut sweeps = 100usize;
    let mut warmup = 100usize;
    let mut draws = 512usize;
    let mut threads = std::thread::available_parallelism().map_or(4, std::num::NonZero::get);
    let mut parity = false;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--parity" => parity = true,
            "--sides" => {
                sides = args
                    .next()
                    .unwrap()
                    .split(',')
                    .map(|s| s.parse().unwrap())
                    .collect();
            }
            "--sweeps" => sweeps = args.next().unwrap().parse().unwrap(),
            "--warmup" => warmup = args.next().unwrap().parse().unwrap(),
            "--draws" => draws = args.next().unwrap().parse().unwrap(),
            "--threads" => threads = args.next().unwrap().parse().unwrap(),
            other => panic!("unknown {other}"),
        }
    }
    if parity {
        run_parity(threads);
        return;
    }
    println!(
        "threads_offered {threads} tile_rows {TILE_ROWS} warmup {warmup} draws {draws} sweep_reps {sweeps}"
    );
    println!(
        "{:>6} {:>10} {:>12} {:>12} {:>12} {:>10} {:>10} {:>10} {:>10}",
        "side", "lever", "ns/sweep", "ns/window", "ESS/sweep", "rng_w", "B/sweep", "work", "agree"
    );
    for side in sides {
        for (name, levers) in [
            ("scalar", Levers::SCALAR),
            ("combined", Levers::COMBINED),
            (
                "sparse",
                Levers {
                    sparse_rng: true,
                    ..Levers::COMBINED
                },
            ),
        ] {
            if side % 2 == 1 && levers.simd {
                continue;
            }
            let probe = OpenIsing::try_new(side, 1.0, 0.3, 1, levers, threads).expect("grid");
            let tr = probe.sweep_load();
            let (ns, rng) = time_sweeps(side, levers, threads, sweeps);
            let (win, ess, agree) = time_window(side, levers, threads, warmup, draws);
            let keep = draws as f64 / (warmup + draws) as f64;
            let bytes = tr.bytes_read + tr.bytes_written + (tr.energy_bytes as f64 * keep) as u64;
            let rng_full = if name == "sparse" {
                rng
            } else {
                tr.rng_words_full
            };
            let work = tr.updates + rng_full + (tr.bonds as f64 * keep) as u64;
            let rng_w = if rng > 0 { rng } else { tr.rng_words_full };
            println!(
                "{side:>6} {name:>10} {ns:>12.1} {win:>12.1} {ess:>12.4} {rng_w:>10} {bytes:>10} {work:>10} {agree:>10.5}"
            );
        }
    }
}
