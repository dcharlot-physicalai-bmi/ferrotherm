#![allow(missing_docs)]
// IS K_mix = 250 SWEEPS PER DENOISING STEP ENOUGH -- measured EXACTLY, on models small enough to
// enumerate, with the trend in size.
//
// The DTM paper (Jelincic et al., arXiv:2510.23972) samples each denoising step by running Gibbs
// for K_mix = 250 sweeps on the step's conditional EBM, from a random start, and the 10,000x
// energy figure multiplies that constant straight through: E = T * K_mix * (per-sweep cost). A
// K_mix that is too small biases every sample; one that is too large inflates the joules. Nothing
// published measures how far 250 sweeps land from the conditional's equilibrium, because on a
// 4,900-site model nothing can.
//
// On a small model everything can. This trains a DTM on a 3x3 and a 4x3 grid, and for each
// denoising step and a handful of clamp contexts x^{t+1} it builds the step's conditional as an
// explicit kernel on all 2^n states -- SEQUENTIAL single-site Gibbs, the sweep `Dtm::sample`
// actually runs -- and reads off, with no sampling anywhere:
//
//   * tau_exact      the conditional chain's integrated autocorrelation time, in sweeps;
//   * K(1%), K(0.1%) the sweeps from the UNIFORM start `Dtm::sample` uses until the exact total
//                    variation to the conditional's equilibrium is below 1% and 0.1%.
//
// The second is the one K_mix has to clear, and it is a fact about the trained model, not an
// estimate. Whether 250 is generous or short at n = 9 and n = 12, and which way the numbers move
// between them, is what the paper's constant rests on.
//
// Count-based throughout; valid on a busy machine.
//
// run: cargo run --release --example kmix_exact

use ferrotherm::autocorr::{apply_distribution, boltzmann, tau_int_exact, total_variation, Kernel};
use ferrotherm::dtm::{forward_step, gamma_coupling, Dtm};
use ferrotherm::graph::{Graph, GraphBuilder};
use ferrotherm::rng::Pcg;

/// Nearest-neighbour edges of a w x h open grid, as the DTM's edge list.
fn grid_edges(w: usize, h: usize) -> Vec<(u16, u16)> {
    let mut e = Vec::new();
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) as u16;
            if x + 1 < w {
                e.push((i, i + 1));
            }
            if y + 1 < h {
                e.push((i, i + w as u16));
            }
        }
    }
    e
}

/// The data distribution: a frustrated +-J grid at `beta`, sampled EXACTLY by enumeration.
struct Data {
    cdf: Vec<f64>,
    n: usize,
}

impl Data {
    fn new(w: usize, h: usize, beta: f64, seed: u64) -> Data {
        let mut rng = Pcg::new(seed, 0xDA7A);
        let n = w * h;
        let mut b = GraphBuilder::new(n);
        for &(a, c) in &grid_edges(w, h) {
            b.couple(a as usize, c as usize, if rng.f64() < 0.5 { -1.0 } else { 1.0 });
        }
        let g = b.build();
        let pi = boltzmann(&g, beta).expect("small");
        let mut cdf = Vec::with_capacity(pi.len());
        let mut acc = 0.0;
        for p in pi {
            acc += p;
            cdf.push(acc);
        }
        Data { cdf, n }
    }
    fn draw(&self, rng: &mut Pcg) -> Vec<i8> {
        let u = rng.f64();
        let x = self.cdf.partition_point(|&c| c < u).min(self.cdf.len() - 1);
        (0..self.n).map(|i| if (x >> i) & 1 == 1 { 1 } else { -1 }).collect()
    }
}

/// The step-t conditional `P(x^t, z | x^{t+1})` as a Graph: the step's couplings and fields plus
/// the forward-coupling field the clamp induces on the visible sites.
fn conditional(dtm: &Dtm, t: usize, x_next: &[i8]) -> Graph {
    let ebm = &dtm.steps[t];
    let dt = dtm.times[t + 1] - dtm.times[t];
    let g = gamma_coupling(dtm.gamma, dt, 2);
    let mut b = GraphBuilder::new(ebm.n);
    for (k, &(a, c)) in ebm.edges.iter().enumerate() {
        b.couple(a as usize, c as usize, ebm.j[k]);
    }
    for i in 0..ebm.n {
        let extra = if i < dtm.nv { 0.5 * g * f64::from(x_next[i]) } else { 0.0 };
        b.bias(i, ebm.h[i] + extra);
    }
    b.build()
}

/// Sweeps from the uniform start until the exact TV to equilibrium is under each threshold.
fn sweeps_to(g: &Graph, thresholds: &[f64], cap: usize) -> Vec<usize> {
    let m = 1usize << g.n;
    let pi = boltzmann(g, 1.0).expect("small");
    let mut mu = vec![1.0 / m as f64; m];
    let mut out = vec![cap; thresholds.len()];
    for k in 1..=cap {
        mu = apply_distribution(g, 1.0, Kernel::SequentialGibbs, &mu);
        let tv = total_variation(&mu, &pi);
        for (j, &th) in thresholds.iter().enumerate() {
            if out[j] == cap && tv < th {
                out[j] = k;
            }
        }
        if out.iter().all(|&k| k < cap) {
            break;
        }
    }
    out
}

fn main() {
    let t_steps = 4usize;
    let gamma = 1.0;
    let times = vec![0.0, 0.15, 0.35, 0.7, 1.4];
    let (train_iters, batch, k_sweeps, lr) = (400usize, 64usize, 10usize, 0.05);
    let contexts = 8u64;
    println!("K_mix, EXACTLY: sweeps a denoising step needs from a uniform start, on enumerable DTMs\n");
    println!("  chain     T = {t_steps} steps, gamma = {gamma}, times {times:?}; visible = all sites, no latents");
    println!("  training  {train_iters} contrastive steps per stage, batch {batch}, {k_sweeps} sweeps per phase, lr {lr}");
    println!("  data      a frustrated +-J grid at beta 0.8, drawn exactly by enumeration");
    println!("  kernel    sequential single-site Gibbs, the sweep Dtm::sample runs; start uniform, as it does\n");
    println!(
        "  {:>5} {:>4}   {:>9} {:>9}   {:>7} {:>7}   {:>8} {:>8}   {:>12}",
        "grid", "step", "tau mean", "tau max", "K1% avg", "K1% max", "K.1% avg", "K.1% max", "250 covers?"
    );
    for &(w, h) in &[(3usize, 3usize), (4, 3)] {
        let n = w * h;
        let data = Data::new(w, h, 0.8, 1);
        let mut rng = Pcg::new(9, 0xD7);
        let mut dtm = Dtm::new(t_steps, n, n, grid_edges(w, h), gamma, times.clone());
        // Train every stage on (x^t, x^{t+1}) pairs made by forward-noising exact data.
        for t in 0..t_steps {
            for _ in 0..train_iters {
                let pairs: Vec<(Vec<i8>, Vec<i8>)> = (0..batch)
                    .map(|_| {
                        let mut x = data.draw(&mut rng);
                        for u in 0..t {
                            forward_step(&mut x, gamma, times[u + 1] - times[u], &mut rng);
                        }
                        let x_t = x.clone();
                        forward_step(&mut x, gamma, times[t + 1] - times[t], &mut rng);
                        (x_t, x)
                    })
                    .collect();
                dtm.train_step(t, &pairs, k_sweeps, lr, 0.0, &mut rng);
            }
        }
        // IS THE MODEL TRAINED? A chain whose couplings barely moved has conditionals that are
        // nearly products and mix in a sweep, and a K_mix verdict on it says nothing about the
        // paper's model. So: the exact per-step conditional negative log-likelihood of held-out
        // forward pairs, -ln P_theta(x^t | x^{t+1}) through `Dtm::exact_log_cond`, summed over the
        // steps, for the trained chain against the same chain untrained (all couplings zero);
        // and the coupling magnitudes. (`Dtm::exact_nll` would be the one number, but it
        // enumerates every (T+1)-tuple of visible states -- (2^9)^5 here -- and does not say so.)
        let mut hrng = Pcg::new(4242, 1);
        let untrained = Dtm::new(t_steps, n, n, grid_edges(w, h), gamma, times.clone());
        let (mut nll_t, mut nll_u) = (0.0f64, 0.0f64);
        let held = 256usize;
        for _ in 0..held {
            let mut x = data.draw(&mut hrng);
            for t in 0..t_steps {
                let x_t = x.clone();
                forward_step(&mut x, gamma, times[t + 1] - times[t], &mut hrng);
                nll_t -= dtm.exact_log_cond(t, &x_t, &x);
                nll_u -= untrained.exact_log_cond(t, &x_t, &x);
            }
        }
        let (nll_t, nll_u) = (nll_t / held as f64, nll_u / held as f64);
        // The data's own entropy per sample, the floor for x^0 alone; printed for scale.
        let entropy: f64 = {
            let mut prev = 0.0;
            let mut hsum = 0.0;
            for &c in &data.cdf {
                let p = c - prev;
                prev = c;
                if p > 0.0 {
                    hsum -= p * p.ln();
                }
            }
            hsum
        };
        let mut jmax = 0.0f64;
        let mut jmean = 0.0f64;
        let mut jcount = 0usize;
        for ebm in &dtm.steps {
            for &j in &ebm.j {
                jmax = jmax.max(j.abs());
                jmean += j.abs();
                jcount += 1;
            }
        }
        println!(
            "  {w}x{h} trained: conditional NLL summed over {t_steps} steps, 256 held-out pairs: {nll_t:.3} nats \
             (untrained {nll_u:.3}; data entropy {entropy:.3} nats/sample for scale); |J| mean {:.3} max {jmax:.3} \
             over {jcount} couplings",
            jmean / jcount as f64
        );
        // Score every step over several clamp contexts drawn from the forward process.
        let mut worst_k1 = 0usize;
        for t in 0..t_steps {
            let (mut taus, mut k1s, mut k01s) = (Vec::new(), Vec::new(), Vec::new());
            for c in 0..contexts {
                let mut crng = Pcg::new(100 + c, t as u64);
                let mut x = data.draw(&mut crng);
                for u in 0..=t {
                    forward_step(&mut x, gamma, times[u + 1] - times[u], &mut crng);
                }
                let g = conditional(&dtm, t, &x);
                let a = tau_int_exact(&g, 1.0, Kernel::SequentialGibbs, |s| g.energy(s), 1e-12, 500_000)
                    .expect("small model, varying energy");
                let ks = sweeps_to(&g, &[0.01, 0.001], 5_000);
                taus.push(a.tau_int);
                k1s.push(ks[0]);
                k01s.push(ks[1]);
            }
            let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
            let meanu = |v: &[usize]| v.iter().sum::<usize>() as f64 / v.len() as f64;
            let (k1max, k01max) = (*k1s.iter().max().unwrap(), *k01s.iter().max().unwrap());
            worst_k1 = worst_k1.max(k1max);
            println!(
                "  {:>5} {t:>4}   {:>9.2} {:>9.2}   {:>7.1} {k1max:>7}   {:>8.1} {k01max:>8}   {:>12}",
                format!("{w}x{h}"),
                mean(&taus),
                taus.iter().copied().fold(0.0f64, f64::max),
                meanu(&k1s),
                meanu(&k01s),
                if k1max <= 250 { "yes" } else { "NO" }
            );
        }
        println!("  {w}x{h}: the largest K(1%) over every step and context is {worst_k1} sweeps against the paper's 250.\n");
    }
    println!("  WHAT THE TABLE SAYS.\n");
    println!("  K(1%) is the number of sequential Gibbs sweeps, from the uniform start Dtm::sample uses, after");
    println!("  which the step's sampled distribution is within 1% total variation of its exact conditional.");
    println!("  It is computed by pushing the uniform distribution through the exact kernel; there is no");
    println!("  sampling in it and no window. If 250 covers every row with room, the paper's constant is");
    println!("  generous at these sizes and the trend between 3x3 and 4x3 says which way it moves; if any");
    println!("  row exceeds it, samples from that step are biased by at least the TV at 250 sweeps.");
}
