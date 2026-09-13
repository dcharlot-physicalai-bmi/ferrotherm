#![allow(missing_docs)]
// WHICH PROBLEM DOES A LOW-PRECISION FABRIC SOLVE? Exact: the Boltzmann law of the REQUANTISED
// Hamiltonian against the law of the one that was asked for, under the two precision models the
// crate's fabrics declare, across dynamic range and absolute scale.
//
// Two premises compete. Ohno & Togawa (IEEE TQE 2026) hold that sample quality under limited
// precision is governed by the Hamiltonian's dynamic range max|J| / min|J|; the crate's own
// defect record (an RTL fabric mis-declared as normalising, which then quantised every coupling of
// a small-weight program to zero) says that on an ABSOLUTE grid the loss is set by min|J| against
// the step alone, and a program of dynamic range 1 can lose everything. Both are right about a
// different model:
//
//   Fixed { bits }   signed fixed point with the step sized by the LARGEST coefficient present --
//                    a normalising fabric; the step is max|J| / (2^(bits-1) - 1)
//   Grid { step }    an absolute step -- the Q.8 RTL, Hitachi's 4-bit ASIC
//
// This example applies exactly the rounding fabric::requantize applies, round(w / step) * step,
// to matched instances -- the same dynamic range at two absolute scales, and the same scale at
// four dynamic ranges -- and measures on 12 spins, exactly: the total variation between the two
// Boltzmann laws, the KL per spin, and the mass the requantised law puts on the true ground state.
// Fields are requantised too, as a fabric would; the dynamic range is over couplings.
//
// run: cargo run --release --example requant_exact

use ferrotherm::autocorr::{boltzmann, spins, total_variation, Kernel};
use ferrotherm::graph::{Graph, GraphBuilder};
use ferrotherm::rng::Pcg;

const N: usize = 12;

fn grid_edges() -> Vec<(usize, usize)> {
    let (w, h) = (4usize, 3usize);
    let mut e = Vec::new();
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            if x + 1 < w {
                e.push((i, i + 1));
            }
            if y + 1 < h {
                e.push((i, i + w));
            }
        }
    }
    e
}

/// Couplings with |J| log-uniform in [1/range, 1] and random signs, fields in [-0.2, 0.2].
fn instance(range: f64, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut rng = Pcg::new(seed, 0x7E);
    let e = grid_edges().len();
    let j: Vec<f64> = (0..e)
        .map(|_| {
            let mag = (-rng.f64() * range.ln()).exp();
            if rng.f64() < 0.5 { -mag } else { mag }
        })
        .collect();
    let h: Vec<f64> = (0..N).map(|_| (rng.f64() - 0.5) * 0.4).collect();
    (j, h)
}

fn build(j: &[f64], h: &[f64]) -> Graph {
    let mut b = GraphBuilder::new(N);
    for (k, &(a, c)) in grid_edges().iter().enumerate() {
        b.couple(a, c, j[k]);
    }
    for i in 0..N {
        b.bias(i, h[i]);
    }
    b.build()
}

/// `round(w / step) * step` on every coefficient: what `fabric::requantize` does.
fn quantise(v: &[f64], step: f64) -> Vec<f64> {
    v.iter().map(|w| (w / step).round() * step).collect()
}

/// The step of `Precision::Fixed { bits }` for these coefficients.
fn fixed_step(j: &[f64], h: &[f64], bits: u32) -> f64 {
    let max = j.iter().chain(h).map(|w| w.abs()).fold(0.0f64, f64::max);
    max / ((1u64 << (bits - 1)) - 1) as f64
}

fn kl(p: &[f64], q: &[f64]) -> f64 {
    p.iter().zip(q).filter(|(a, _)| **a > 0.0).map(|(a, b)| a * (a / b.max(1e-300)).ln()).sum()
}

/// Ground states of `g`: every state within 1e-9 of the minimum energy.
fn ground_states(g: &Graph) -> Vec<usize> {
    let m = 1usize << N;
    let e: Vec<f64> = (0..m).map(|x| g.energy(&spins(x, N))).collect();
    let min = e.iter().copied().fold(f64::INFINITY, f64::min);
    (0..m).filter(|&x| e[x] < min + 1e-9).collect()
}

struct Cell {
    tv: f64,
    kl_per_spin: f64,
    gs_mass: f64,
}

fn measure(j: &[f64], h: &[f64], jq: &[f64], hq: &[f64], beta: f64) -> Cell {
    let g = build(j, h);
    let gq = build(jq, hq);
    let p = boltzmann(&g, beta).expect("small");
    let q = boltzmann(&gq, beta).expect("small");
    let gs = ground_states(&g);
    Cell { tv: total_variation(&p, &q), kl_per_spin: kl(&p, &q) / N as f64, gs_mass: gs.iter().map(|&x| q[x]).sum() }
}

fn slope(xs: &[f64], ys: &[f64]) -> f64 {
    let n = xs.len() as f64;
    let (mx, my) = (xs.iter().sum::<f64>() / n, ys.iter().sum::<f64>() / n);
    let sxy: f64 = xs.iter().zip(ys).map(|(x, y)| (x - mx) * (y - my)).sum();
    let sxx: f64 = xs.iter().map(|x| (x - mx).powi(2)).sum();
    sxy / sxx
}

fn main() {
    let _ = Kernel::ChromaticGibbs;
    let ranges = [1.0f64, 4.0, 16.0, 64.0];
    let scales = [1.0f64, 0.125];
    let seeds = [1u64, 2, 3];
    println!("REQUANTISATION, EXACTLY: TV BETWEEN THE ASKED-FOR AND THE LOADED BOLTZMANN LAWS, ON ALL 2^{N} STATES\n");
    println!("  model    4x3 grid; |J| log-uniform in [1/R, 1] times a scale s, random signs; fields in s * [-0.2, 0.2]; {} seeds", seeds.len());
    println!("  beta     applied to the UNscaled instance, so (s, beta / s) is the same physics as (1, beta): a matched pair");
    println!("  Fixed    step = max|coef| / (2^(bits-1) - 1); Grid: absolute step. Each cell: TV, KL/spin (nats), P(true ground states)\n");
    for &beta in &[0.5f64, 1.0, 2.0] {
        println!("  beta {beta}\n");
        println!("  {:>4} {:>6}   {:>8}   {:>10} {:>10} {:>10} {:>10}   {:>10} {:>10} {:>10} {:>10}", "R", "scale", "", "Fixed 3b", "Fixed 4b", "Fixed 6b", "Fixed 8b", "Grid 1/4", "Grid 1/16", "Grid 1/64", "Grid 1/256");
        let mut fixed_rows: Vec<(f64, [f64; 4])> = Vec::new();
        let mut grid_rows: Vec<(f64, f64, [f64; 4])> = Vec::new();
        for &range in &ranges {
            for &scale in &scales {
                let mut fixed = [0.0f64; 4];
                let mut grid = [0.0f64; 4];
                let mut fixed_gs = [0.0f64; 4];
                let mut grid_gs = [0.0f64; 4];
                let mut fixed_kl = [0.0f64; 4];
                let mut grid_kl = [0.0f64; 4];
                let mut gs_ref = 0.0f64;
                for &seed in &seeds {
                    let (j0, h0) = instance(range, seed);
                    let j: Vec<f64> = j0.iter().map(|v| v * scale).collect();
                    let h: Vec<f64> = h0.iter().map(|v| v * scale).collect();
                    let b = beta / scale;
                    gs_ref += measure(&j, &h, &j, &h, b).gs_mass / seeds.len() as f64;
                    for (k, &bits) in [3u32, 4, 6, 8].iter().enumerate() {
                        let step = fixed_step(&j, &h, bits);
                        let c = measure(&j, &h, &quantise(&j, step), &quantise(&h, step), b);
                        fixed[k] += c.tv / seeds.len() as f64;
                        fixed_gs[k] += c.gs_mass / seeds.len() as f64;
                        fixed_kl[k] += c.kl_per_spin / seeds.len() as f64;
                    }
                    for (k, &step) in [0.25f64, 1.0 / 16.0, 1.0 / 64.0, 1.0 / 256.0].iter().enumerate() {
                        let c = measure(&j, &h, &quantise(&j, step), &quantise(&h, step), b);
                        grid[k] += c.tv / seeds.len() as f64;
                        grid_gs[k] += c.gs_mass / seeds.len() as f64;
                        grid_kl[k] += c.kl_per_spin / seeds.len() as f64;
                    }
                }
                let f = |v: &[f64; 4]| v.iter().map(|x| format!("{x:>10.3e}")).collect::<Vec<_>>().join(" ");
                println!("  {range:>4.0} {scale:>6.3}   {:>8}   {}   {}", "TV", f(&fixed), f(&grid));
                println!("  {:>4} {:>6}   {:>8}   {}   {}", "", "", "KL/spin", f(&fixed_kl), f(&grid_kl));
                println!("  {:>4} {:>6}   {:>8}   {}   {}   (unquantised {gs_ref:.3e})", "", "", "P(GS)", f(&fixed_gs), f(&grid_gs));
                if scale == 1.0 {
                    fixed_rows.push((range, fixed));
                }
                grid_rows.push((range, scale, grid));
            }
        }
        // Slopes: log TV against log R at fixed bits (the dynamic-range premise) for the
        // normalising model, and log TV against log(min|J| / step) for the absolute grid.
        let lr: Vec<f64> = fixed_rows.iter().map(|(r, _)| r.max(1.0001).ln()).collect();
        let fixed_slopes: Vec<String> = (0..4)
            .map(|k| {
                let ys: Vec<f64> = fixed_rows.iter().map(|(_, v)| v[k].max(1e-300).ln()).collect();
                format!("{:.2}", slope(&lr, &ys))
            })
            .collect();
        let grid_slopes: Vec<String> = (0..4)
            .map(|k| {
                let step = [0.25f64, 1.0 / 16.0, 1.0 / 64.0, 1.0 / 256.0][k];
                let xs: Vec<f64> = grid_rows.iter().map(|(r, s, _)| (s / r / step).ln()).collect();
                let ys: Vec<f64> = grid_rows.iter().map(|(_, _, v)| v[k].max(1e-300).ln()).collect();
                format!("{:.2}", slope(&xs, &ys))
            })
            .collect();
        println!("\n  slope of log TV vs log R (Fixed, scale 1, per bits 3/4/6/8):        {}", fixed_slopes.join("  "));
        println!("  slope of log TV vs log(min|J|/step) (Grid, both scales, per step):   {}\n", grid_slopes.join("  "));
    }
    println!("  WHAT THE TABLE SAYS.\n");
    println!("  Under Fixed the two scales of a matched pair give the SAME cells: the step follows the largest");
    println!("  coefficient, so the loss is a function of dynamic range and bits alone. Under Grid the same pair");
    println!("  differs by the scale: the loss is a function of min|J| against the step, and a range-1 instance at");
    println!("  a small scale can lose everything. The slopes are the two premises measured: a positive slope");
    println!("  against R for Fixed is Ohno & Togawa's, a negative slope against min|J|/step for Grid is the");
    println!("  crate's. P(GS) is the optimisation face of the same substitution: how much of the loaded law sits");
    println!("  on the ground states of the problem that was asked.");
}
