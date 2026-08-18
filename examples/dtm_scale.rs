// DTM at the published flagship scale: L=70 grid, G12 connectivity, T=8 chained EBMs, trained on
// binarized Fashion-MNIST. This is the configuration the architecture's own paper reports
// (arXiv:2510.23972): 4,900 nodes per layer, visible sites drawn uniformly at random.
//
// Reports the measured cost of a training step at that scale, then trains within a wall-clock
// budget and reports how the model's visible statistics approach the data's.
//
// usage: cargo run --release --example dtm_scale -- <fmnist-images> [seconds] [L] [T]
use ferrotherm::dtm::{forward_step, gamma_coupling, pattern_grid, Ebm, G12};
use ferrotherm::rng::Pcg;
use std::time::{Duration, Instant};

fn load_images(path: &str, limit: usize) -> (Vec<Vec<i8>>, usize) {
    let raw = std::fs::read(path).expect("read images");
    let n = u32::from_be_bytes([raw[4], raw[5], raw[6], raw[7]]) as usize;
    let rows = u32::from_be_bytes([raw[8], raw[9], raw[10], raw[11]]) as usize;
    let cols = u32::from_be_bytes([raw[12], raw[13], raw[14], raw[15]]) as usize;
    let px = rows * cols;
    let take = n.min(limit);
    let mut out = Vec::with_capacity(take);
    for i in 0..take {
        let off = 16 + i * px;
        out.push(raw[off..off + px].iter().map(|&v| if v > 127 { 1i8 } else { -1 }).collect());
    }
    (out, px)
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    // A missing argument is a usage error, not a crash.
    //
    // This was `.expect(..)`, so running the example with no arguments printed "thread 'main'
    // panicked" and a backtrace note -- which reads as the example being broken rather than as the
    // caller having forgotten a path. Every other example in this directory runs with no arguments;
    // this one needs a dataset, and the difference should look deliberate.
    let Some(path) = a.first() else {
        eprintln!("usage: dtm_scale <fmnist-images> [seconds] [L] [T]");
        eprintln!();
        eprintln!("  <fmnist-images>  an idx3-ubyte file, e.g. train-images-idx3-ubyte");
        eprintln!("  [seconds]        wall-clock budget for training      (default 120)");
        eprintln!("  [L]              pattern-grid side                   (default 70)");
        eprintln!("  [T]              denoising steps                     (default 8)");
        std::process::exit(2);
    };
    let budget = Duration::from_secs(a.get(1).and_then(|s| s.parse().ok()).unwrap_or(120));
    let l: usize = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(70);
    let t_steps: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(8);

    let (data, px) = load_images(path, 60_000);
    println!("data: {} binarized images, {px} pixels each", data.len());

    // ---- the fabric, at the published scale ----
    let edges32 = pattern_grid(l, &G12);
    let n = l * l;
    let edges: Vec<(u16, u16)> = edges32.iter().map(|&(a, b)| (a as u16, b as u16)).collect();
    println!("grid: {l}x{l} = {n} nodes, G12, {} edges (mean degree {:.1})",
             edges.len(), 2.0 * edges.len() as f64 / n as f64);

    // visible sites drawn uniformly at random, as the paper specifies
    let mut rng = Pcg::new(0xD7A1, 1);
    let mut perm: Vec<u32> = (0..n as u32).collect();
    for i in (1..perm.len()).rev() {
        let j = (rng.f64() * (i + 1) as f64) as usize;
        perm.swap(i, j);
    }
    let visible: Vec<usize> = perm[..px].iter().map(|&v| v as usize).collect();
    println!("visible sites: {} of {n} (uniformly random), latent: {}", visible.len(), n - px);

    let mut layers: Vec<Ebm> = (0..t_steps).map(|_| Ebm::new(n, edges.clone())).collect();
    println!("layers: {t_steps}; bipartite: {}; parameters: {}",
             layers[0].is_bipartite(), t_steps * (edges.len() + n));

    // forward-process schedule
    let gamma_x = 1.0f64;
    let times: Vec<f64> = (0..=t_steps).map(|i| i as f64 * 0.35).collect();

    // ---- measured cost of one training step at this scale ----
    let k_sweeps = 25usize;
    let batch = 8usize;
    let t0 = Instant::now();
    {
        let mut s = vec![1i8; n];
        let extra = vec![0.0f64; n];
        layers[0].gibbs_chromatic(&mut s, &extra, 10, &mut rng);
    }
    let per_sweep = t0.elapsed().as_secs_f64() / 10.0;
    println!("\nmeasured: {:.2} ms per chromatic sweep of {n} nodes ({:.2e} node updates/s)",
             per_sweep * 1e3, n as f64 / per_sweep);
    let step_cost = per_sweep * k_sweeps as f64 * 2.0 * batch as f64 * t_steps as f64;
    println!("one training step (batch {batch}, K={k_sweeps}, both phases, {t_steps} layers): {:.2} s",
             step_cost);

    // ---- train within the budget ----
    println!("\ntraining for {} s...", budget.as_secs());
    let lr = 0.02;
    let start = Instant::now();
    let mut steps = 0usize;
    let mut pos_ss = vec![0.0f64; edges.len()];
    let mut pos_si = vec![0.0f64; n];
    let mut neg_ss = vec![0.0f64; edges.len()];
    let mut neg_si = vec![0.0f64; n];
    let all: Vec<usize> = (0..n).collect();
    let latents: Vec<usize> = (0..n).filter(|i| !visible.contains(i)).collect();

    while start.elapsed() < budget {
        let t = steps % t_steps;
        let dt = times[t + 1] - times[t];
        let gam = gamma_coupling(gamma_x, dt, 2);
        pos_ss.iter_mut().for_each(|v| *v = 0.0);
        pos_si.iter_mut().for_each(|v| *v = 0.0);
        neg_ss.iter_mut().for_each(|v| *v = 0.0);
        neg_si.iter_mut().for_each(|v| *v = 0.0);

        for _ in 0..batch {
            // a data image, noised to time t and t+1
            let img = &data[(rng.f64() * data.len() as f64) as usize % data.len()];
            let mut x_prev: Vec<i8> = img.clone();
            if t > 0 {
                forward_step(&mut x_prev, gamma_x, times[t], &mut rng);
            }
            let mut x_next = x_prev.clone();
            forward_step(&mut x_next, gamma_x, dt, &mut rng);

            // the clamp field from x^{t+1} on the visible sites
            let mut extra = vec![0.0f64; n];
            for (k, &v) in visible.iter().enumerate() {
                extra[v] = 0.5 * gam * x_next[k] as f64;
            }
            // positive phase: visibles clamped to x^t, Gibbs over latents
            let mut s = vec![0i8; n];
            for i in 0..n {
                s[i] = if rng.f64() < 0.5 { 1 } else { -1 };
            }
            for (k, &v) in visible.iter().enumerate() {
                s[v] = x_prev[k];
            }
            layers[t].gibbs(&mut s, &latents, &extra, k_sweeps, &mut rng);
            layers[t].accumulate(&s, &mut pos_ss, &mut pos_si, 1.0);

            // negative phase: everything free under the same clamp field
            let mut sn = vec![0i8; n];
            for i in 0..n {
                sn[i] = if rng.f64() < 0.5 { 1 } else { -1 };
            }
            layers[t].gibbs(&mut sn, &all, &extra, k_sweeps, &mut rng);
            layers[t].accumulate(&sn, &mut neg_ss, &mut neg_si, 1.0);
        }

        let m = batch as f64;
        // The total-correlation penalty from the architecture's own paper. Without it the
        // negative phase must mix fully or the model's correlations are underestimated and the
        // couplings grow without bound: |J| climbing monotonically with no settling is that
        // failure, not learning. The gradient reuses the negative-phase statistics, so it is free.
        let lambda = 0.35;
        let lay = &mut layers[t];
        for k in 0..lay.edges.len() {
            let (a, b) = lay.edges[k];
            let (ma, mb) = (neg_si[a as usize] / m, neg_si[b as usize] / m);
            let tc = ma * mb - neg_ss[k] / m;
            lay.j[k] += lr * ((pos_ss[k] - neg_ss[k]) / m - lambda * tc);
        }
        for i in 0..n {
            lay.h[i] += lr * (pos_si[i] - neg_si[i]) / m;
        }
        steps += 1;
        if steps.is_multiple_of(40) {
            println!("  step {steps:5}  layer {t}  |J| mean {:.4}  elapsed {:.0} s",
                     lay.j.iter().map(|v| v.abs()).sum::<f64>() / lay.j.len() as f64,
                     start.elapsed().as_secs_f64());
        }
    }
    println!("completed {steps} steps in {:.0} s", start.elapsed().as_secs_f64());

    // ---- does the model reproduce the data's visible statistics? ----
    let mut data_m = vec![0.0f64; px];
    for img in data.iter().take(4000) {
        for k in 0..px {
            data_m[k] += img[k] as f64;
        }
    }
    data_m.iter_mut().for_each(|v| *v /= 4000.0);

    // reverse chain: start from noise, denoise through the layers
    let n_samples = 32usize;
    let mut model_m = vec![0.0f64; px];
    for _ in 0..n_samples {
        let mut x: Vec<i8> = (0..px).map(|_| if rng.f64() < 0.5 { 1 } else { -1 }).collect();
        for t in (0..t_steps).rev() {
            let dt = times[t + 1] - times[t];
            let gam = gamma_coupling(gamma_x, dt, 2);
            let mut extra = vec![0.0f64; n];
            for (k, &v) in visible.iter().enumerate() {
                extra[v] = 0.5 * gam * x[k] as f64;
            }
            let mut s = vec![0i8; n];
            for i in 0..n {
                s[i] = if rng.f64() < 0.5 { 1 } else { -1 };
            }
            layers[t].gibbs(&mut s, &all, &extra, k_sweeps, &mut rng);
            for (k, &v) in visible.iter().enumerate() {
                x[k] = s[v];
            }
        }
        for k in 0..px {
            model_m[k] += x[k] as f64;
        }
    }
    model_m.iter_mut().for_each(|v| *v /= n_samples as f64);

    let mae: f64 = (0..px).map(|k| (model_m[k] - data_m[k]).abs()).sum::<f64>() / px as f64;
    let data_mean: f64 = data_m.iter().sum::<f64>() / px as f64;
    let model_mean: f64 = model_m.iter().sum::<f64>() / px as f64;
    // an untrained chain outputs unbiased noise, so its per-pixel mean is ~0 and the MAE is
    // just the data's own |mean| profile: that is the number to beat.
    let baseline: f64 = data_m.iter().map(|v| v.abs()).sum::<f64>() / px as f64;
    println!("\nper-pixel mean activation:  data {data_mean:+.3}   model {model_mean:+.3}");
    println!("per-pixel MAE vs data:      model {mae:.3}   untrained-noise baseline {baseline:.3}");
    println!("verdict: {}", if mae < baseline {
        format!("the trained chain is closer to the data than noise by {:.1}%", 100.0 * (1.0 - mae / baseline))
    } else {
        "no better than noise within this budget".to_string()
    });
}
