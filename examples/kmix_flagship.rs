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
// A SECOND MODE, `frontier`, asks the question the first one raises. If a fixed penalty at 0.35
// certifies by not learning and no penalty learns by not mixing, where between them is the point
// that does both -- and does a CONTROLLER find it? It sweeps fixed strengths beside two
// controllers: `ACP`, `dtm::acp_update` fed each layer's per-site spin autocorrelation at the
// training lag (the paper's definition of the input; WHICH observable is our choice, the paper
// names none, and the constants are this crate's); and `R-ACP`, the same multiplicative law fed
// by `Convergence::from_chains` over dispersed starts. Each row also prints `a(K)`, what the
// paper-style controller would read on that very model, beside the R-hat it is being held against.
//
// Count-based throughout; valid on a busy machine. NOT run in CI: the larger stages take minutes.
// run: cargo run --release --example kmix_flagship [max_L] [base|frontier] [min_L] [arm-filter]

use ferrotherm::cftp::Bounding;
use ferrotherm::dtm::{acp_update, forward_step, gamma_coupling, pattern_grid, Dtm, G12};
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

/// How the total-correlation penalty is set.
#[derive(Clone, Copy)]
enum Arm {
    /// One strength for every layer, for the whole run.
    Fixed(f64),
    /// The paper's adaptive correlation penalty, per layer, from this starting strength.
    Acp(f64),
    /// The same multiplicative controller driven by a CONVERGENCE diagnostic instead: raise a
    /// layer's penalty when four chains from dispersed starts still disagree after `K_mix` sweeps
    /// (`Convergence::from_chains` refuses), lower it when they agree. A normalised single-chain
    /// autocorrelation subtracts each site's own mean, so a chain stuck in one mode reads as
    /// decorrelated; chains started apart cannot make that mistake.
    Rhat(f64),
    /// The CERTIFICATE AT THE DEPLOYMENT BUDGET as the sensor. Each update asks a layer's
    /// conditional the exact question that matters: does a bounding chain coalesce within `K_mix`
    /// sweeps? Refused on any of four draws: DOUBLE the penalty. All four inside half the budget:
    /// ease it by a tenth. Otherwise hold. Two things the failed `Rhat` arm taught: the input must
    /// be able to see a stuck chain, and the law must rise much faster than it falls, or it spends
    /// the run climbing back from a floor it reached while the young model still mixed easily.
    Cert(f64),
    /// THE MATCHED ACTUATOR. `Cert` failed at size because its actuator, a penalty on correlations,
    /// does not act on what its sensor measures: a bounding chain coalesces according to coupling
    /// MAGNITUDES. So this one trains with no penalty at all and, whenever a layer fails the
    /// certificate, scales that layer's couplings down by a tenth until it passes -- a projection
    /// onto the certifiable set. It is held to HALF the deployment budget, because a four-draw
    /// sensor ran about one doubling optimistic against the scoring's 144 draws, and it projects
    /// once more before scoring, which is what one would do before deploying any model.
    Project,
}

/// The paper's controller input: "the autocorrelations of each learned conditional at a delay
/// equal to the number of sampling iterations used during gradient estimation". The paper does not
/// say WHICH observable. This takes each site's own spin autocorrelation at that lag and averages
/// over the sites that move at all, so one slow site anywhere registers.
fn layer_autocorrelation(g: &Graph, lag: usize, trace: usize, seed: u64) -> f64 {
    let mut s = gibbs::Sampler::new(g, 1.0, seed);
    s.sweeps(4 * lag, None);
    let mut states: Vec<Vec<i8>> = Vec::with_capacity(trace);
    for _ in 0..trace {
        s.sweep(None);
        states.push(s.read_all(None));
    }
    let (mut total, mut counted) = (0.0f64, 0usize);
    for i in 0..g.n {
        let mut mean = 0.0;
        for st in &states {
            mean += f64::from(st[i]);
        }
        mean /= trace as f64;
        let var = 1.0 - mean * mean;
        if var < 1e-6 {
            continue;
        }
        let mut c = 0.0;
        for k in 0..trace - lag {
            c += f64::from(states[k][i]) * f64::from(states[k + lag][i]);
        }
        total += (c / (trace - lag) as f64 - mean * mean) / var;
        counted += 1;
    }
    if counted == 0 { 0.0 } else { total / counted as f64 }
}

/// Scale `layer`'s couplings down by a tenth until four bounding-chain draws on its conditional
/// under `x_next` all coalesce within `cap` sweeps. Returns the factor applied.
fn project_layer(dtm: &mut Dtm, layer: usize, x_next: &[i8], cap: usize, seed: u64) -> f64 {
    let mut factor = 1.0;
    for _ in 0..80 {
        let g = conditional(dtm, layer, x_next);
        let b = Bounding::new(&g, 1.0).expect("finite").with_max_steps(cap);
        let mut ok = true;
        for d in 0..4u64 {
            ok &= b.draw(seed + d, None).is_ok();
        }
        if ok {
            break;
        }
        for j in &mut dtm.steps[layer].j {
            *j *= 0.9;
        }
        factor *= 0.9;
    }
    factor
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
    // `base` is the two arms the first measurement ran; `frontier` is the adaptive penalty beside a
    // sweep of fixed strengths, which is what the controller is supposed to find its way along.
    let frontier = std::env::args().nth(2).is_some_and(|a| a == "frontier");
    let min_l: usize = std::env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(0);
    let only: Option<String> = std::env::args().nth(4);
    let arms: Vec<(&str, Arm)> = if frontier {
        vec![
            ("ACP", Arm::Acp(0.01)),
            ("R-ACP", Arm::Rhat(0.01)),
            ("C-ACP", Arm::Cert(0.05)),
            ("PROJECT", Arm::Project),
            ("TC .02", Arm::Fixed(0.02)),
            ("TC .05", Arm::Fixed(0.05)),
            ("TC .10", Arm::Fixed(0.10)),
            ("TC .20", Arm::Fixed(0.20)),
        ]
    } else {
        vec![("TC .35", Arm::Fixed(0.35)), ("no TC", Arm::Fixed(0.0))]
    };
    let (acp_every, acp_eps, acp_delta, acp_min) = (100usize, 0.03, 0.2, 1e-4);
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
        "  {:>3} {:>5} {:>4} {:>5}  {:>7} {:>6}   {:>7} {:>6} {:>6}   {:>9} {:>9} {:>8}   {:>9} {:>7}  {:>17}  verdict",
        "L", "n", "nv", "deg", "arm", "steps", "learned", "|J|avg", "|J|max", "cert. med", "cert. max", "refused", "R-hat@250", "a(K)", "lambda min/avg/max"
    );

    for &l in &[10usize, 20, 28, 40, 70] {
        if l > max_l {
            break;
        }
        if l < min_l {
            continue;
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

        for &(arm, how) in &arms {
            if only.as_ref().is_some_and(|f| !arm.contains(f.as_str())) {
                continue;
            }
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
            let mut lambda = vec![
                match how {
                    Arm::Fixed(v) | Arm::Acp(v) | Arm::Rhat(v) | Arm::Cert(v) => v,
                    Arm::Project => 0.0,
                };
                t_steps
            ];
            // For `Project`: the factor each layer's couplings have been scaled by so far.
            let mut shrunk = vec![1.0f64; t_steps];
            let mut a_prev: Vec<Option<f64>> = vec![None; t_steps];
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
                    dtm.train_step(t, &pairs, k_sweeps, lr, lambda[t], &mut rng);
                    done += 1;
                    // THE CONTROLLER. Every `acp_every` updates of a layer, measure that layer's
                    // conditional at the training lag under a fresh clamp context and move its
                    // penalty. The lag is `k_sweeps`, because that is the paper's definition: the
                    // question is whether the samples the GRADIENT used had forgotten their start.
                    if !matches!(how, Arm::Fixed(_)) && done.is_multiple_of(t_steps * acp_every) {
                        for layer in 0..t_steps {
                            let mut x = draw(&mut chain);
                            for u in 0..=layer {
                                forward_step(&mut x, gamma, times[u + 1] - times[u], &mut rng);
                            }
                            let g = conditional(&dtm, layer, &x);
                            match how {
                                Arm::Fixed(_) => {}
                                Arm::Acp(_) => {
                                    let a = layer_autocorrelation(&g, k_sweeps, 40 * k_sweeps, 0xACB0 + done as u64 + layer as u64);
                                    lambda[layer] = acp_update(lambda[layer], a, a_prev[layer], acp_eps, acp_delta, acp_min);
                                    a_prev[layer] = Some(a);
                                }
                                Arm::Rhat(_) => {
                                    // Held to the constraint that matters at deployment: K_mix sweeps.
                                    let disagree = rhat_after(&g, k_mix, 200).refusal().is_some();
                                    let lp = lambda[layer].max(acp_min);
                                    lambda[layer] = if disagree { (1.0 + acp_delta) * lp } else { (1.0 - acp_delta) * lp };
                                }
                                Arm::Project => {
                                    shrunk[layer] *= project_layer(&mut dtm, layer, &x, k_mix / 2, 0x9807 + done as u64 * 8 + layer as u64 * 4);
                                }
                                Arm::Cert(_) => {
                                    // A refusal at this cap costs about 2 * K_mix sweeps, so asking
                                    // the deployment question directly is cheap.
                                    let b = Bounding::new(&g, 1.0).expect("finite").with_max_steps(k_mix);
                                    let (mut slowest, mut refused_here) = (0usize, false);
                                    for d in 0..4u64 {
                                        match b.draw(0xCE27 + done as u64 * 8 + layer as u64 * 4 + d, None) {
                                            Ok(x) => slowest = slowest.max(x.coalesced_at),
                                            Err(_) => refused_here = true,
                                        }
                                    }
                                    let lp = lambda[layer].max(1e-3);
                                    lambda[layer] = if refused_here {
                                        2.0 * lp
                                    } else if slowest <= k_mix.div_ceil(2) + 3 {
                                        0.9 * lp
                                    } else {
                                        lp
                                    };
                                }
                            }
                        }
                    }
                }

                if let Arm::Project = how {
                    // Project before deploying: two fresh clamp contexts per layer.
                    for layer in 0..t_steps {
                        for ctx in 0..2u64 {
                            let mut x = draw(&mut chain);
                            for u in 0..=layer {
                                forward_step(&mut x, gamma, times[u + 1] - times[u], &mut rng);
                            }
                            shrunk[layer] *= project_layer(&mut dtm, layer, &x, k_mix / 2, 0xDE91 + done as u64 * 16 + layer as u64 * 2 + ctx);
                        }
                    }
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
                let mut worst_a = f64::NEG_INFINITY;
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
                            // What the paper-style controller would read on this very layer, so
                            // the two diagnostics can be held against each other on one model.
                            worst_a = worst_a.max(layer_autocorrelation(&g, k_sweeps, 40 * k_sweeps, 0xA11C + t as u64));
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
                // The last column is the penalty per layer, or for `Project` the scale per layer.
                let shown = if let Arm::Project = how { &shrunk } else { &lambda };
                let lmin = shown.iter().copied().fold(f64::INFINITY, f64::min);
                let lmax = shown.iter().copied().fold(0.0f64, f64::max);
                let lavg = shown.iter().sum::<f64>() / t_steps as f64;
                println!(
                    "  {l:>3} {n:>5} {nv:>4} {degree:>5.1}  {arm:>7} {done:>6}   {learned:>7.3} {:>6.3} {jmax:>6.3}   {:>9} {:>9} {refused:>8}   {worst:>9.4} {worst_a:>7.3}  {lmin:>5.3}/{lavg:>5.3}/{lmax:>5.3}  {verdict}",
                    jsum / jcount as f64,
                    show(med),
                    show(max)
                );
            }
        }
    }
}
