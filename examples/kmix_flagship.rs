// `missing_docs` is denied workspace-wide and is right to be: it guards the API surface, and
// every public item in every library here carries a doc. An EXAMPLE has no API surface -- it is
// a program, and its helpers are private to it -- so the lint has nothing to guard and asks for
// doc comments on `fn main`'s scaffolding instead. Scoped off here rather than weakened there.
#![allow(missing_docs)]
// K_mix on the FLAGSHIP SHAPE: twelve neighbours, and most of the sites latent.
//
// `kmix_certified` certified the DTM paper's K_mix = 250 on an all-visible nearest-neighbour grid
// up to 4,900 sites (worst coalescence 64 sweeps) and said what it had not measured: the paper's
// own configuration (Jelincic et al., arXiv:2510.23972) wires each site to TWELVE others (pattern
// G12), makes only 784 of 4,900 sites visible -- 16%, placed uniformly at random -- and chains
// eight steps. That is a different mixing problem. In the all-visible model every site carries a
// clamp field from x^{t+1}; here five sites in six are latent and carry none, and an unpinned site
// with twelve neighbours is exactly where a bounding chain's unknowns breed.
//
// So this carries TWO instruments, because one of them may decline to answer:
//   certificate   `cftp::Bounding`: the look-back at which every start is forgotten, exactly, or
//                 a refusal at the cap. A refusal is "not certified", never "too slow".
//   R-hat         four chains from dispersed starts (all up, all down, two random), run the
//                 paper's 250 sweeps, then traced; `floors::Convergence::from_chains` takes the
//                 strictest of the split, rank-normalised and folded R-hats and refuses above
//                 1.01. This CAN say 250 is not enough.
//
// AND IT IS A LADDER IN TRAINING LENGTH, WITH TWO ARMS, because a mixing verdict on a model that
// has barely moved says nothing: `kmix_exact` makes the point that a chain whose couplings are
// near zero has conditionals that are nearly products. So each size is scored at 2,000, 8,000 and
// 32,000 steps, with the total-correlation penalty at `dtm_scale`'s 0.35 and with it off, and
// beside every mixing verdict is how much the model LEARNED: the share of the data's
// nearest-neighbour correlations that generated samples reproduce.
//
// The shape is the paper's and the settings are `dtm_scale`'s (k = 25, batch 8, lr 0.02, times
// i * 0.35); the trainer is `Dtm::train_step`, whose penalty sign is pinned by a test. A FIXED
// penalty is this crate's simplification -- the paper adapts it with the ACP controller. The DATA is not the paper's: Fashion-MNIST is not in this
// repository, so the visible sites are trained on a frustrated +-J grid at beta 0.8 drawn by a
// persistent Gibbs chain. What is measured is the mixing of conditionals trained on that.
//
// Count-based throughout; valid on a busy machine. NOT run in CI: the 70x70 stage takes minutes.
// run: cargo run --release --example kmix_flagship [max_L]

use ferrotherm::cftp::Bounding;
use ferrotherm::dtm::{forward_step, gamma_coupling, pattern_grid, Dtm, G12};
use ferrotherm::floors::Convergence;
use ferrotherm::gibbs;
use ferrotherm::graph::{Graph, GraphBuilder};
use ferrotherm::rng::Pcg;

fn data_model(side: usize, seed: u64) -> Graph {
    let mut rng = Pcg::new(seed, 0xDA7A);
    let mut b = GraphBuilder::new(side * side);
    for y in 0..side {
        for x in 0..side {
            let i = y * side + x;
            if x + 1 < side {
                b.couple(i, i + 1, if rng.f64() < 0.5 { -1.0 } else { 1.0 });
            }
            if y + 1 < side {
                b.couple(i, i + side, if rng.f64() < 0.5 { -1.0 } else { 1.0 });
            }
        }
    }
    b.build()
}

fn conditional(dtm: &Dtm, t: usize, x_next: &[i8]) -> Graph {
    let ebm = &dtm.steps[t];
    let g = gamma_coupling(dtm.gamma, dtm.times[t + 1] - dtm.times[t], 2);
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

/// Energy traces of four chains from dispersed starts, recorded AFTER `k_mix` sweeps.
fn rhat_after(g: &Graph, k_mix: usize, trace: usize) -> Convergence {
    let mut chains = Vec::with_capacity(4);
    for (c, start) in [Some(1i8), Some(-1i8), None, None].into_iter().enumerate() {
        let mut s = gibbs::Sampler::new(g, 1.0, 0xC4A1 + c as u64);
        if let Some(v) = start {
            s.s = vec![v; g.n];
        }
        s.sweeps(k_mix, None);
        let mut e = Vec::with_capacity(trace);
        for _ in 0..trace {
            s.sweep(None);
            e.push(g.energy(&s.read_all(None)));
        }
        chains.push(e);
    }
    Convergence::from_chains(&chains)
}

/// Mean nearest-neighbour correlation over the data grid's edges, from a set of visible vectors.
fn edge_correlations(samples: &[Vec<i8>], side: usize) -> Vec<f64> {
    let mut out = Vec::new();
    for y in 0..side {
        for x in 0..side {
            let i = y * side + x;
            for j in [if x + 1 < side { Some(i + 1) } else { None }, if y + 1 < side { Some(i + side) } else { None }] {
                let Some(j) = j else { continue };
                let mut acc = 0.0;
                for v in samples {
                    acc += f64::from(v[i]) * f64::from(v[j]);
                }
                out.push(acc / samples.len() as f64);
            }
        }
    }
    out
}

fn main() {
    let max_l: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(28);
    let t_steps = 8usize;
    let gamma = 1.0;
    let times: Vec<f64> = (0..=t_steps).map(|i| i as f64 * 0.35).collect();
    let (batch, k_sweeps, lr) = (8usize, 25usize, 0.02);
    let (k_mix, cap) = (250usize, 1usize << 13);

    println!("K_mix ON THE FLAGSHIP SHAPE, AS THE MODEL LEARNS: G12 wiring, 16% visible at random, T = {t_steps}\n");
    println!("  training  round-robin over the layers, batch {batch}, k = {k_sweeps}, lr {lr}; two arms, TC penalty 0.35 and 0");
    println!("  data      +-J grid on the visible sites at beta 0.8, persistent Gibbs chain (NOT Fashion-MNIST)");
    println!("  learned   share of the data's nearest-neighbour correlations that GENERATED samples reproduce:");
    println!("            1 - mean|c_model - c_data| / mean|c_data|; zero for a model that learned nothing");
    println!("  score     certificate over every layer (cap {cap}); R-hat: worst layer, 4 dispersed chains after {k_mix}\n");
    println!(
        "  {:>3} {:>5} {:>4} {:>5}  {:>7} {:>6}   {:>7} {:>6} {:>6}   {:>9} {:>9} {:>8}   {:>9}  verdict",
        "L", "n", "nv", "deg", "arm", "steps", "learned", "|J|avg", "|J|max", "cert. med", "cert. max", "refused", "R-hat@250"
    );

    for &l in &[10usize, 20, 28, 40, 70] {
        if l > max_l {
            break;
        }
        let n = l * l;
        let side = (0.4 * l as f64).round() as usize;
        let nv = side * side;
        let mut perm: Vec<usize> = (0..n).collect();
        let mut prng = Pcg::new(0x5EED, l as u64);
        for i in (1..n).rev() {
            let j = (prng.f64() * (i + 1) as f64) as usize;
            perm.swap(i, j.min(i));
        }
        let mut name = vec![0usize; n];
        for (new, &old) in perm.iter().enumerate() {
            name[old] = new;
        }
        let raw = pattern_grid(l, &G12);
        let edges: Vec<(u16, u16)> = raw.iter().map(|&(a, b)| (name[a as usize] as u16, name[b as usize] as u16)).collect();
        let degree = 2.0 * edges.len() as f64 / n as f64;
        let model = data_model(side, 1);
        let checkpoints: &[usize] = if n >= 1_600 { &[2_000, 8_000] } else { &[2_000, 8_000, 32_000] };
        let gen_samples = if n >= 1_600 { 120usize } else { 300usize };

        for (arm, lambda_tc) in [("TC .35", 0.35f64), ("no TC", 0.0)] {
            let mut chain = gibbs::Sampler::new(&model, 0.8, 0xDA7A + l as u64);
            chain.sweeps(2_000, None);
            let draw = |c: &mut gibbs::Sampler| -> Vec<i8> {
                c.sweeps(50, None);
                c.read_all(None)
            };
            let mut reference = Vec::with_capacity(600);
            for _ in 0..600 {
                reference.push(draw(&mut chain));
            }
            let c_data = edge_correlations(&reference, side);
            let scale = c_data.iter().map(|c| c.abs()).sum::<f64>() / c_data.len() as f64;

            let mut rng = Pcg::new(9, 0xD7);
            let mut dtm = Dtm::new(t_steps, n, nv, edges.clone(), gamma, times.clone());
            let mut done = 0usize;
            for &target in checkpoints {
                while done < target {
                    let t = done % t_steps;
                    let mut pairs = Vec::with_capacity(batch);
                    for _ in 0..batch {
                        let mut x = draw(&mut chain);
                        for u in 0..t {
                            forward_step(&mut x, gamma, times[u + 1] - times[u], &mut rng);
                        }
                        let x_t = x.clone();
                        forward_step(&mut x, gamma, times[t + 1] - times[t], &mut rng);
                        pairs.push((x_t, x));
                    }
                    dtm.train_step(t, &pairs, k_sweeps, lr, lambda_tc, &mut rng);
                    done += 1;
                }

                let mut srng = Pcg::new(77, done as u64);
                let mut generated = Vec::with_capacity(gen_samples);
                for _ in 0..gen_samples {
                    generated.push(dtm.sample(k_mix, &mut srng));
                }
                let c_model = edge_correlations(&generated, side);
                let err = c_model.iter().zip(&c_data).map(|(a, b)| (a - b).abs()).sum::<f64>() / c_data.len() as f64;
                let learned = 1.0 - err / scale;

                let (mut jsum, mut jcount, mut jmax) = (0.0f64, 0usize, 0.0f64);
                let (mut certs, mut refused) = (Vec::new(), 0usize);
                let mut worst = 1.0f64;
                let mut rhat_refused = false;
                let (contexts, per) = if n >= 1_600 { (2u64, 3usize) } else { (3u64, 6usize) };
                for t in 0..t_steps {
                    for &j in &dtm.steps[t].j {
                        jsum += j.abs();
                        jcount += 1;
                        jmax = jmax.max(j.abs());
                    }
                    for c in 0..contexts {
                        let mut crng = Pcg::new(100 + c, t as u64);
                        let mut x = draw(&mut chain);
                        for u in 0..=t {
                            forward_step(&mut x, gamma, times[u + 1] - times[u], &mut crng);
                        }
                        let g = conditional(&dtm, t, &x);
                        let b = Bounding::new(&g, 1.0).expect("finite").with_max_steps(cap);
                        for d in 0..per {
                            match b.draw(1_000 * (c + 1) + d as u64, None) {
                                Ok(x) => certs.push(x.coalesced_at),
                                Err(_) => refused += 1,
                            }
                        }
                        if c == 0 {
                            let r = rhat_after(&g, k_mix, 400);
                            rhat_refused |= r.refusal().is_some();
                            if let Convergence::MultiChain { rhat } = r
                                && (rhat.is_nan() || rhat > worst)
                            {
                                worst = rhat;
                            }
                        }
                    }
                }
                certs.sort_unstable();
                let med = certs.get(certs.len() / 2).copied();
                let max = certs.last().copied();
                let verdict = if refused == 0 && max.is_some_and(|m| m <= k_mix) {
                    "CERTIFIED: 250 covers"
                } else if rhat_refused {
                    "250 NOT ENOUGH (R-hat)"
                } else if refused == 0 {
                    "exact, but needs > 250"
                } else {
                    "not certified; R-hat passes"
                };
                let show = |v: Option<usize>| v.map_or("-".to_string(), |x| x.to_string());
                println!(
                    "  {l:>3} {n:>5} {nv:>4} {degree:>5.1}  {arm:>7} {done:>6}   {learned:>7.3} {:>6.3} {jmax:>6.3}   {:>9} {:>9} {refused:>8}   {worst:>9.4}  {verdict}",
                    jsum / jcount as f64,
                    show(med),
                    show(max)
                );
            }
        }
    }
}
