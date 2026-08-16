//! Restream against clamp: what a sampling fabric costs to LOAD versus what it costs to RE-CONDITION.
//!
//! TR-2026-40 names conditioning bandwidth as the bound between a sampling substrate and a control
//! loop, and separates two operations that are easy to read as one. RESTREAMING loads a model onto
//! the fabric. CLAMPING imposes new boundary conditions on a model already resident. A controller
//! does the second on every tick, so the clamp rate is the quantity that governs closed-loop use,
//! and the two can differ by orders of magnitude.
//!
//! That distinction had never been measured on this stack. `Gpu::sweep` creates every buffer, the
//! shader module and the compute pipeline inside the call, so as the API stands there is no
//! resident path at all: every call is a restream followed by a clamp. Timing one call therefore
//! measures the sum and tells you nothing about either part.
//!
//! Separating them needs no new API. Wall time for k sweeps is fixed + marginal*k, so sweeping k
//! and fitting a line splits the two: the intercept is what a restream costs and the slope is what
//! a resident sweep costs. The ratio is how much a resident-model API would be worth.
//!
//! PREDICTED before measuring, recorded here so the prediction cannot be adjusted afterwards:
//! the intercept dominates at small k, it is milliseconds rather than microseconds because it
//! carries pipeline creation and a mapped readback, and the implied resident clamp rate is at
//! least an order of magnitude above the rate the current call achieves. Falsified if the
//! intercept is comparable to a single sweep, which would mean the restream is already free and
//! the bound lies somewhere else.

use ferrotherm::{gibbs::Sampler, ising::lattice2d, wgsl::GpuModel};
use std::time::Instant;

/// Least squares through (k, ms) for ms = a + b*k. Returns (a, b).
///
/// Deliberately not a library call: the whole measurement is this fit, and a reader checking the
/// number should be able to see the arithmetic that produced it without leaving the file.
fn fit(points: &[(f64, f64)]) -> (f64, f64) {
    let n = points.len() as f64;
    let sx: f64 = points.iter().map(|p| p.0).sum();
    let sy: f64 = points.iter().map(|p| p.1).sum();
    let sxx: f64 = points.iter().map(|p| p.0 * p.0).sum();
    let sxy: f64 = points.iter().map(|p| p.0 * p.1).sum();
    let d = n * sxx - sx * sx;
    if d.abs() < 1e-12 { return (sy / n, 0.0); }
    let b = (n * sxy - sx * sy) / d;
    ((sy - b * sx) / n, b)
}

/// Worst relative residual of the fit, as a fraction. A line that does not describe the data makes
/// the intercept meaningless, so this is printed beside every intercept rather than kept back.
fn worst_residual(points: &[(f64, f64)], a: f64, b: f64) -> f64 {
    points.iter()
        .map(|&(k, ms)| ((a + b * k) - ms).abs() / ms.max(1e-9))
        .fold(0.0f64, f64::max)
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn minimum(v: &[f64]) -> f64 { v.iter().cloned().fold(f64::INFINITY, f64::min) }

/// This machine runs other work. A first pass at this bench took five samples per point and
/// produced a fit whose worst residual was 47 percent, with k=256 timing FASTER than the line
/// through k=128 predicted, which is not a thing a workload does. The cause was contention, not
/// the GPU.
///
/// Two changes follow from that. Samples per point go up, and the capability question is read off
/// the MINIMUM rather than the median: the fastest observed run is the least contaminated estimate
/// of what the machine can do, while the median describes what it did while sharing itself. Both
/// are printed, because the gap between them is the contention and hiding it would make a
/// contended run look like a clean one.
fn load_average() -> Option<f64> {
    let out = std::process::Command::new("uptime").output().ok()?;
    let s = String::from_utf8_lossy(&out.stdout);
    let tail = s.rsplit("load averages:").next().or_else(|| s.rsplit("load average:").next())?;
    tail.split_whitespace().next()?.trim_end_matches(',').parse().ok()
}

fn main() {
    let Some(gpu) = ferrotherm_gpu::Gpu::new() else {
        eprintln!("no GPU adapter on this machine; nothing to measure");
        return;
    };
    let a = gpu.adapter();
    println!("adapter: {} ({:?}, {:?})", a.name, a.device_type, a.backend);
    if !gpu.is_hardware() {
        println!();
        println!("This is a software rasteriser. Every figure below describes a CPU pretending to");
        println!("be a GPU, so they are printed and NOT reported as a fabric measurement.");
    }
    println!();
    println!("PREDICTED before measuring: the intercept is milliseconds, dominates at small k, and");
    println!("implies a resident clamp rate at least 10x the rate a single call achieves.");
    println!();

    let ks = [1usize, 2, 4, 8, 16, 32, 64, 128, 256];
    let repeats = 21;
    let beta = 0.7;
    match load_average() {
        Some(la) if la > 1.5 => {
            println!("load average {la:.2}: this machine is running other work. The minimum column");
            println!("is the figure to read; the median column is what it managed while sharing.");
            println!();
        }
        Some(la) => { println!("load average {la:.2}"); println!(); }
        None => {}
    }

    // The cold call is reported separately. It carries shader compilation, which the driver caches,
    // so folding it into the median would describe the first tick of a process and no other.
    println!("=== cold start: the first call on this process ===");
    println!("{:>6}  {:>8}  {:>12}", "l", "nodes", "cold ms");
    let mut cold = Vec::new();
    for l in [16usize, 64, 256] {
        let g = lattice2d(l, 1.0);
        let m = GpuModel::from_graph(&g);
        let mut s = vec![1i8; l * l];
        let t0 = Instant::now();
        gpu.sweep(&m, &mut s, beta, 1).unwrap();
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        cold.push((l, ms));
        println!("{:>6}  {:>8}  {:>12.3}", l, l * l, ms);
    }
    println!();

    println!("=== restream against clamp, warm ===");
    println!("Fitting ms = restream + clamp*k over k = 1..256, {repeats} runs per point.");
    println!("The fit is on the minimum series. The median series is printed beside it.");
    println!();
    println!(
        "{:>6}  {:>8}  {:>12}  {:>12}  {:>9}  {:>11}  {:>13}  {:>7}  {:>9}  {:>9}",
        "l", "nodes", "restream ms", "clamp ms", "ratio", "call Hz", "resident Hz", "fit err", "med/min", "pass/pass"
    );

    let mut rows = Vec::new();
    for l in [16usize, 32, 64, 128, 256, 512] {
        let g = lattice2d(l, 1.0);
        let n = l * l;
        let m = GpuModel::from_graph(&g);
        let mut s = vec![1i8; n];

        // Warm the pipeline cache for THIS model before any timed point.
        gpu.sweep(&m, &mut s, beta, 1).unwrap();

        // TWO independent passes, always. Running this once gave restream figures that moved by
        // 36 percent and clamp figures by 43 percent between consecutive invocations on this
        // machine, while the RATIO between them stayed inside one order of magnitude. A bench that
        // reports a single pass cannot tell a reader which of its digits survive a repeat, so both
        // passes are printed and the agreement between them is a column.
        let mut passes: Vec<(f64, f64, f64, f64)> = Vec::new();
        for pass in 0..2 {
            let mut lo = Vec::new();
            let mut mid = Vec::new();
            for &k in &ks {
                let runs: Vec<f64> = (0..repeats)
                    .map(|_| {
                        let t0 = Instant::now();
                        gpu.sweep(&m, &mut s, beta, k as u32).unwrap();
                        t0.elapsed().as_secs_f64() * 1e3
                    })
                    .collect();
                lo.push((k as f64, minimum(&runs)));
                mid.push((k as f64, median(runs)));
            }
            // The raw points, always. A bench that prints a fit without the data behind it is
            // asking to be believed rather than checked, and the first version of this one had a
            // 47 percent residual that only the raw column explained.
            print!("  min l={:<4} p{}", l, pass + 1);
            for &(k, ms) in &lo { print!("  {}:{:.3}", k as u32, ms); }
            println!();
            let (restream, clamp) = fit(&lo);
            let err = worst_residual(&lo, restream, clamp);
            let spread = mid.last().unwrap().1 / lo.last().unwrap().1;
            passes.push((restream, clamp, err, spread));
        }

        let (r1, c1, e1, sp) = passes[0];
        let (r2, c2, e2, _) = passes[1];
        let restream = 0.5 * (r1 + r2);
        let clamp = 0.5 * (c1 + c2);
        let err = e1.max(e2);
        // How far the two passes disagree, as the larger over the smaller. This is the digit count
        // the figure actually supports.
        let agree = (r1.max(r2) / r1.min(r2)).max(c1.max(c2) / c1.min(c2));
        // What one conditioning step costs today, and what it would cost if the model stayed put.
        let call_hz = 1e3 / (restream + clamp).max(1e-9);
        let resident_hz = 1e3 / clamp.max(1e-9);
        let ratio = restream / clamp.max(1e-9);
        rows.push((l, n, restream, clamp, call_hz, resident_hz));
        println!(
            "{:>6}  {:>8}  {:>12.3}  {:>12.4}  {:>8.1}x  {:>11.1}  {:>13.1}  {:>6.1}%  {:>8.2}x  {:>8.2}x",
            l, n, restream, clamp, ratio, call_hz, resident_hz, err * 100.0, sp, agree
        );
    }

    println!();
    println!("=== against the CPU path on the same models ===");
    println!("A controller with a small model may not need the fabric at all, and a fabric figure");
    println!("that never asks this question is comparing against nothing.");
    println!();
    println!("{:>6}  {:>8}  {:>14}  {:>14}", "l", "nodes", "cpu ms/sweep", "gpu ms/sweep");
    for &(l, n, _, clamp, _, _) in &rows {
        let g = lattice2d(l, 1.0);
        let cpu_ms = median(
            (0..repeats)
                .map(|r| {
                    let mut sim = Sampler::new(&g, beta, r as u64 + 1);
                    let t0 = Instant::now();
                    sim.sweeps(32, None);
                    t0.elapsed().as_secs_f64() * 1e3 / 32.0
                })
                .collect(),
        );
        println!("{:>6}  {:>8}  {:>14.4}  {:>14.4}", l, n, cpu_ms, clamp);
    }

    println!();
    println!("=== what fits under a control deadline ===");
    println!("Sweeps available per tick at each rate, under the current API and with the model");
    println!("resident. A count of zero means the tick is spent before a single sweep runs.");
    println!();
    print!("{:>6}  {:>8}", "l", "nodes");
    for hz in [10, 30, 50, 100] { print!("  {:>8}", format!("{hz} Hz now")); }
    for hz in [10, 30, 50, 100] { print!("  {:>10}", format!("{hz} Hz res")); }
    println!();
    for &(l, n, restream, clamp, _, _) in &rows {
        print!("{:>6}  {:>8}", l, n);
        for hz in [10.0, 30.0, 50.0, 100.0] {
            let budget = 1e3 / hz;
            let k = ((budget - restream) / clamp.max(1e-9)).floor().max(0.0);
            print!("  {:>8}", k as i64);
        }
        for hz in [10.0, 30.0, 50.0, 100.0] {
            let budget = 1e3 / hz;
            let k = (budget / clamp.max(1e-9)).floor().max(0.0);
            print!("  {:>10}", k as i64);
        }
        println!();
    }

    println!();
    println!("The restream column is a property of this API, not of this hardware: `Gpu::sweep`");
    println!("builds every buffer, the shader module and the pipeline inside the call. The clamp");
    println!("column is what the fabric does. Whichever of the two is larger is the one that sets");
    println!("the conditioning bandwidth, and only one of them needs new physics to move.");
}
