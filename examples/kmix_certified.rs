// `missing_docs` is denied workspace-wide and is right to be: it guards the API surface, and
// every public item in every library here carries a doc. An EXAMPLE has no API surface -- it is
// a program, and its helpers are private to it -- so the lint has nothing to guard and asks for
// doc comments on `fn main`'s scaffolding instead. Scoped off here rather than weakened there.
#![allow(missing_docs)]
// K_mix, CERTIFIED -- past the sizes enumeration can reach.
//
// `kmix_exact` trains a denoising chain on 3x3 and 4x3 grids, builds every step's conditional as an
// exact kernel on all 2^n states, and finds the sweeps needed from a uniform start are at most 6,
// against the DTM paper's K_mix = 250 (Jelincic et al., arXiv:2510.23972 / npj Unconv. Comput.
// 2026), whose 10,000x energy figure multiplies that constant straight through. It ends by naming
// what it cannot do: "what it needs at 4,900 sites with latents is not a question enumeration can
// answer."
//
// A trained conditional has couplings of both signs, so monotone coupling from the past refuses
// it. `cftp::Bounding` does not: every site is +1, -1 or unknown, an unknown neighbour widens the
// field bracket by its full coupling, and if nothing is unknown at time 0 the draw is EXACT. The
// look-back at which that first happens is a CERTIFIED number of sweeps after which the chain has
// forgotten EVERY start -- not an estimate from a trace, and not a quantity with a window. Unknowns
// die under strong fields and breed through strong couplings, so it coalesces quickly exactly when
// a conditional is pinned and weakly coupled, which is the DTM's own premise. If the premise fails
// on a trained step the sampler hits its cap and says so.
//
// CALIBRATED BEFORE IT IS TRUSTED. The chain, schedule, training recipe, seeds and clamp contexts
// are `kmix_exact`'s, so at 3x3 and 4x3 the trained models are the same models and the exact
// K(1%) is printed beside the certificate. The certificate is an UPPER bound and a power of two:
// it answers "when has every start been forgotten", which is more than "when is the uniform start
// within 1%".
//
// WHAT CHANGES WITH SIZE, STATED. Past 20 spins the data cannot be drawn by enumeration. It is
// drawn exactly by the bounding chain where that coalesces on the data model and by a persistent
// Gibbs chain where it does not, and the table says which. "Is it trained" is `exact_log_cond` at
// the small sizes and coupling magnitudes at the large ones, because the former enumerates.
//
// Count-based throughout; valid on a busy machine. NOT run in CI: the 70x70 stage trains a
// 4,900-site chain and the whole ladder takes minutes.
// run: cargo run --release --example kmix_certified

use ferrotherm::autocorr::{apply_distribution, boltzmann, total_variation, Kernel};
use ferrotherm::cftp::Bounding;
use ferrotherm::dtm::{forward_step, gamma_coupling, Dtm};
use ferrotherm::gibbs;
use ferrotherm::graph::{Graph, GraphBuilder};
use ferrotherm::rng::Pcg;

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

/// The data model `kmix_exact` uses: a frustrated +-J grid, the same couplings from the same seed.
fn data_model(w: usize, h: usize, seed: u64) -> Graph {
    let mut rng = Pcg::new(seed, 0xDA7A);
    let mut b = GraphBuilder::new(w * h);
    for &(a, c) in &grid_edges(w, h) {
        b.couple(a as usize, c as usize, if rng.f64() < 0.5 { -1.0 } else { 1.0 });
    }
    b.build()
}

enum Data {
    Enumerated { cdf: Vec<f64>, n: usize },
    Perfect { g: Graph, beta: f64, next: u64 },
    Chain { s: Box<gibbs::Sampler<'static>>, thin: usize },
}

impl Data {
    fn label(&self) -> &'static str {
        match self {
            Data::Enumerated { .. } => "enumeration (exact)",
            Data::Perfect { .. } => "bounding chain (exact)",
            Data::Chain { .. } => "persistent Gibbs chain",
        }
    }
    fn draw(&mut self, rng: &mut Pcg) -> Vec<i8> {
        match self {
            Data::Enumerated { cdf, n } => {
                let u = rng.f64();
                let x = cdf.partition_point(|&c| c < u).min(cdf.len() - 1);
                (0..*n).map(|i| if (x >> i) & 1 == 1 { 1 } else { -1 }).collect()
            }
            Data::Perfect { g, beta, next } => {
                *next += 1;
                Bounding::new(g, *beta).expect("finite beta").draw(*next, None).expect("coalesced when probed").state
            }
            Data::Chain { s, thin } => {
                s.sweeps(*thin, None);
                s.read_all(None)
            }
        }
    }
}

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

/// `kmix_exact`'s exact K(1%): sweeps from the uniform start until TV to equilibrium is under 1%.
fn exact_k1(g: &Graph, cap: usize) -> usize {
    let m = 1usize << g.n;
    let pi = boltzmann(g, 1.0).expect("small");
    let mut mu = vec![1.0 / m as f64; m];
    for k in 1..=cap {
        mu = apply_distribution(g, 1.0, Kernel::SequentialGibbs, &mu);
        if total_variation(&mu, &pi) < 0.01 {
            return k;
        }
    }
    cap
}

fn main() {
    let t_steps = 4usize;
    let gamma = 1.0;
    let times = vec![0.0, 0.15, 0.35, 0.7, 1.4];
    let (train_iters, batch, k_sweeps, lr) = (400usize, 64usize, 10usize, 0.05);
    let (contexts, draws_per_context, cap) = (8u64, 32usize, 1usize << 16);
    let data_beta = 0.8;

    println!("K_mix, CERTIFIED: the look-back at which a bounding chain has forgotten EVERY start\n");
    println!("  chain     T = {t_steps} steps, gamma = {gamma}, times {times:?}; visible = all sites, no latents");
    println!("  training  {train_iters} contrastive steps per stage, batch {batch}, {k_sweeps} sweeps per phase, lr {lr}");
    println!("  data      a frustrated +-J grid at beta {data_beta}; how it is drawn is stated per size");
    println!("  score     {contexts} clamp contexts per step x {draws_per_context} exact draws each; cap {cap} sweeps\n");
    println!(
        "  {:>7} {:>5} {:>4}   {:>6} {:>6}   {:>9}   {:>10} {:>10} {:>8}   {:>11}",
        "grid", "n", "step", "|J|avg", "|J|max", "exact K1%", "cert. med", "cert. max", "refused", "250 covers?"
    );

    for &(w, h) in &[(3usize, 3usize), (4, 3), (6, 6), (8, 8), (12, 12), (16, 16), (24, 24), (32, 32), (48, 48), (70, 70)] {
        let n = w * h;
        let model = data_model(w, h, 1);
        let mut data = if n <= 16 {
            let pi = boltzmann(&model, data_beta).expect("small");
            let mut acc = 0.0;
            let cdf = pi.iter().map(|p| { acc += p; acc }).collect();
            Data::Enumerated { cdf, n }
        } else if Bounding::new(&model, data_beta).expect("finite").with_max_steps(1 << 12).draw(7, None).is_ok() {
            Data::Perfect { g: model, beta: data_beta, next: 0 }
        } else {
            // The sampler borrows its graph; the graph lives for the program, so leak one per size.
            let g: &'static Graph = Box::leak(Box::new(model));
            let mut s = gibbs::Sampler::new(g, data_beta, 0xDA7A + n as u64);
            s.sweeps(2_000, None);
            Data::Chain { s: Box::new(s), thin: 50 }
        };

        let mut rng = Pcg::new(9, 0xD7);
        let mut dtm = Dtm::new(t_steps, n, n, grid_edges(w, h), gamma, times.clone());
        for t in 0..t_steps {
            for _ in 0..train_iters {
                let mut pairs = Vec::with_capacity(batch);
                for _ in 0..batch {
                    let mut x = data.draw(&mut rng);
                    for u in 0..t {
                        forward_step(&mut x, gamma, times[u + 1] - times[u], &mut rng);
                    }
                    let x_t = x.clone();
                    forward_step(&mut x, gamma, times[t + 1] - times[t], &mut rng);
                    pairs.push((x_t, x));
                }
                dtm.train_step(t, &pairs, k_sweeps, lr, 0.0, &mut rng);
            }
        }

        println!("  {w}x{h}: data by {}", data.label());
        for t in 0..t_steps {
            let (mut jsum, mut jmax) = (0.0f64, 0.0f64);
            for &j in &dtm.steps[t].j {
                jsum += j.abs();
                jmax = jmax.max(j.abs());
            }
            let javg = jsum / dtm.steps[t].j.len() as f64;
            let (mut certs, mut refused, mut k1max) = (Vec::new(), 0usize, 0usize);
            for c in 0..contexts {
                let mut crng = Pcg::new(100 + c, t as u64);
                let mut x = data.draw(&mut crng);
                for u in 0..=t {
                    forward_step(&mut x, gamma, times[u + 1] - times[u], &mut crng);
                }
                let g = conditional(&dtm, t, &x);
                if n <= 12 {
                    k1max = k1max.max(exact_k1(&g, 5_000));
                }
                let b = Bounding::new(&g, 1.0).expect("finite").with_max_steps(cap);
                for d in 0..draws_per_context {
                    match b.draw(1_000 * (c + 1) + d as u64, None) {
                        Ok(x) => certs.push(x.coalesced_at),
                        Err(_) => refused += 1,
                    }
                }
            }
            certs.sort_unstable();
            let med = certs.get(certs.len() / 2).copied().unwrap_or(0);
            let max = certs.last().copied().unwrap_or(0);
            let covered = refused == 0 && max <= 250;
            println!(
                "  {:>7} {n:>5} {t:>4}   {javg:>6.3} {jmax:>6.3}   {:>9}   {med:>10} {max:>10} {refused:>8}   {:>11}",
                format!("{w}x{h}"),
                if n <= 12 { k1max.to_string() } else { "-".to_string() },
                if covered { "yes" } else { "NO" }
            );
        }
    }
}
