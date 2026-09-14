#![allow(missing_docs)]
// F_dyn / F*: does a fixed-budget sampler want its chains weaker than the exact threshold?
//
// `chain_threshold_exact` computes F*(I, E), the smallest chain strength at which every ground
// state of the embedded model is chain-aligned. Fang & Warburton (2020) argue the ground-state
// probability is maximised by the smallest strength with no domain walls, r = F_dyn / F* ~ 1;
// Gilbert & Rodriguez (2024) find that TARGETING a small chain-break rate beats the torque rule,
// i.e. r < 1 and chain breaks are load-bearing after resolution. With F* exact per instance the
// ratio is measurable: sweep F / F*, anneal the embedded model on a fixed budget, resolve broken
// chains by majority, by coupling-weighted vote, or by discarding the sample, and score the
// resolved logical state against the exact logical optimum.
//
// Same fixtures as the threshold: K_n on a 2x2x4 Chimera and 12-vertex d-regular graphs on a
// 3x3x4 Chimera, +-J, no fields; instances with F* under 1e-3 (no chain needed) are skipped.
//
// run: cargo run --release --example chain_dynamic_exact

use ferrotherm::embed::{apply_with, chimera_clique, embed_bounded, Embedding, DEFAULT_SEARCH_BUDGET};
use ferrotherm::exact::Elimination;
use ferrotherm::gauge::{ChainBreak, Chains};
use ferrotherm::graph::{Graph, GraphBuilder};
use ferrotherm::ising::chimera;
use ferrotherm::rng::Pcg;
use ferrotherm::tempering::{anneal, geometric_ladder};

fn complete(n: usize, seed: u64) -> Graph {
    let mut rng = Pcg::new(seed, 0xC1);
    let mut b = GraphBuilder::new(n);
    for i in 0..n {
        for j in (i + 1)..n {
            b.couple(i, j, if rng.f64() < 0.5 { 1.0 } else { -1.0 });
        }
    }
    b.build()
}

fn regular(n: usize, d: usize, seed: u64) -> Graph {
    let mut rng = Pcg::new(seed, 0xD3);
    'attempt: for _ in 0..2_000_000 {
        let mut stubs: Vec<usize> = (0..n).flat_map(|v| std::iter::repeat_n(v, d)).collect();
        for i in (1..stubs.len()).rev() {
            let j = (rng.f64() * (i + 1) as f64) as usize;
            stubs.swap(i, j.min(i));
        }
        let mut edges: Vec<(usize, usize)> = Vec::new();
        for pair in stubs.chunks(2) {
            let (a, c) = (pair[0], pair[1]);
            if a == c || edges.iter().any(|&(x, y)| (x == a && y == c) || (x == c && y == a)) {
                continue 'attempt;
            }
            edges.push((a, c));
        }
        let mut b = GraphBuilder::new(n);
        for (a, c) in edges {
            b.couple(a, c, if rng.f64() < 0.5 { 1.0 } else { -1.0 });
        }
        return b.build();
    }
    panic!("no simple {d}-regular graph on {n} vertices in 2,000,000 pairings");
}

fn chain_edges(e: &Embedding, hw: &Graph) -> usize {
    let mut count = 0;
    for chain in &e.chains {
        for a in 0..chain.len() {
            for c in (a + 1)..chain.len() {
                let (u, v) = (chain[a], chain[c]);
                if (hw.offset[u]..hw.offset[u + 1]).any(|k| hw.nbr[k] as usize == v) {
                    count += 1;
                }
            }
        }
    }
    count
}

fn choi(logical: &Graph) -> f64 {
    (0..logical.n)
        .map(|i| logical.h[i].abs() + (logical.offset[i]..logical.offset[i + 1]).map(|k| logical.w[k].abs()).sum::<f64>())
        .fold(0.0f64, f64::max)
}

/// `(F*, E_I*)` by bisection on an exact elimination, as in `chain_threshold_exact`.
fn threshold(logical: &Graph, hw: &Graph, emb: &Embedding) -> Option<(f64, f64)> {
    let elim = Elimination::default();
    let e_star = elim.ground_state(logical).ok()?.ground_energy?;
    let n_ce = chain_edges(emb, hw) as f64;
    let ground = |f: f64| -> Option<f64> { elim.ground_state(&apply_with(logical, hw, emb, f).graph).ok()?.ground_energy };
    let aligned_ok = |f: f64| -> Option<bool> { Some(ground(f)? >= e_star - f * n_ce - 1e-9) };
    let mut hi = 2.0 * choi(logical).max(1.0);
    let mut lo = 0.0;
    if aligned_ok(0.0)? {
        return Some((0.0, e_star));
    }
    for _ in 0..40 {
        let mid = 0.5 * (lo + hi);
        if aligned_ok(mid)? {
            hi = mid;
        } else {
            lo = mid;
        }
        if (hi - lo) < 1e-4 * hi {
            break;
        }
    }
    Some((hi, e_star))
}

fn embed_or_clique(logical: &Graph, hw: &Graph, seed: u64, clique_m: Option<usize>) -> Option<Embedding> {
    if let Some(e) = embed_bounded(logical, hw, seed, 10, DEFAULT_SEARCH_BUDGET) {
        return Some(e);
    }
    let m = clique_m?;
    let e = chimera_clique(m, 4)?;
    (e.chains.len() >= logical.n).then(|| Embedding { chains: e.chains[..logical.n].to_vec(), ..e })
}

const MULTS: [f64; 8] = [0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0, 3.0];
const POLICIES: [(&str, ChainBreak); 3] = [("majority", ChainBreak::Majority), ("weighted", ChainBreak::Weighted), ("discard", ChainBreak::Discard)];

/// Per instance: P(ground) at each multiplier for each policy, and the break fraction at each.
struct Curve {
    p_gs: [[f64; 8]; 3],
    breaks: [f64; 8],
}

fn curve(logical: &Graph, hw: &Graph, emb: &Embedding, f_star: f64, e_star: f64, seeds: u64, schedule: &[(f64, usize)]) -> Option<Curve> {
    let mut p_gs = [[0.0f64; 8]; 3];
    let mut breaks = [0.0f64; 8];
    for (mi, &m) in MULTS.iter().enumerate() {
        let embedded = apply_with(logical, hw, emb, m * f_star);
        let chains = Chains::of_embedded(&embedded).ok()?;
        for seed in 0..seeds {
            let (state, _) = anneal(&embedded.graph, schedule, 9_000 + seed, None);
            for (pi, (_, policy)) in POLICIES.iter().enumerate() {
                let readout = chains.resolve(&state, *policy).ok()?;
                if pi == 0 {
                    breaks[mi] += readout.break_fraction() / seeds as f64;
                }
                if let Some(vals) = readout.values()
                    && logical.energy(vals) <= e_star + 1e-9
                {
                    p_gs[pi][mi] += 1.0 / seeds as f64;
                }
            }
        }
    }
    Some(Curve { p_gs, breaks })
}

fn report(name: &str, curves: &[Curve]) {
    let n = curves.len();
    println!("  {name}: {n} instances\n");
    println!("  {:<10}   {}", "F / F*", MULTS.iter().map(|m| format!("{m:>7.2}")).collect::<Vec<_>>().join(" "));
    let mean_breaks: Vec<f64> = (0..8).map(|mi| curves.iter().map(|c| c.breaks[mi]).sum::<f64>() / n as f64).collect();
    println!("  {:<10}   {}", "breaks", mean_breaks.iter().map(|b| format!("{b:>7.3}")).collect::<Vec<_>>().join(" "));
    for (pi, (pname, _)) in POLICIES.iter().enumerate() {
        let mean: Vec<f64> = (0..8).map(|mi| curves.iter().map(|c| c.p_gs[pi][mi]).sum::<f64>() / n as f64).collect();
        println!("  {:<10}   {}", format!("P(GS) {pname}"), mean.iter().map(|p| format!("{p:>7.3}")).collect::<Vec<_>>().join(" "));
    }
    println!();
    for (pi, (pname, _)) in POLICIES.iter().enumerate() {
        // F_dyn per instance: the smallest multiplier attaining that instance's best P(GS).
        let rs: Vec<f64> = curves
            .iter()
            .map(|c| {
                let best = c.p_gs[pi].iter().copied().fold(0.0f64, f64::max);
                MULTS[c.p_gs[pi].iter().position(|&p| p >= best - 1e-12).unwrap_or(3)]
            })
            .collect();
        let below = rs.iter().filter(|&&r| r < 1.0).count();
        let at = rs.iter().filter(|&&r| (r - 1.0).abs() < 1e-12).count();
        let above = rs.iter().filter(|&&r| r > 1.0).count();
        let mean = rs.iter().sum::<f64>() / n as f64;
        println!("  {pname:<10}   r = F_dyn / F*: mean {mean:.2}; r < 1 on {below}, r = 1 on {at}, r > 1 on {above} of {n}");
    }
    println!();
}

fn main() {
    let instances = 20u64;
    let seeds = 100u64;
    let ladder = geometric_ladder(0.05, 6.0, 40);
    let schedule: Vec<(f64, usize)> = ladder.iter().map(|&b| (b, 20)).collect();
    println!("F_dyn AGAINST THE EXACT THRESHOLD F*, UNDER A FIXED ANNEALING BUDGET\n");
    println!("  anneal   tempering::anneal on a geometric ladder 0.05..6.0 of 40 rungs x 20 sweeps = 800 sweeps, {seeds} seeds per cell");
    println!("  score    resolved logical state attains the exact optimum E_I*; 'breaks' = mean fraction of chains broken (majority readout)");
    println!("  F_dyn    per instance, the smallest multiplier of F* attaining that instance's best P(GS)\n");
    let hw2 = chimera(2, 2, 4, 1.0);
    let mut curves = Vec::new();
    for n in [5usize, 6, 7, 8] {
        for seed in 0..instances {
            let logical = complete(n, 100 + seed);
            let Some(emb) = embed_or_clique(&logical, &hw2, seed, Some(2)) else { continue };
            let Some((f_star, e_star)) = threshold(&logical, &hw2, &emb) else { continue };
            if f_star < 1e-3 {
                continue;
            }
            if let Some(c) = curve(&logical, &hw2, &emb, f_star, e_star, seeds, &schedule) {
                curves.push(c);
            }
        }
    }
    report("K_5..K_8, +-J, chimera(2,2,4)", &curves);
    let hw3 = chimera(3, 3, 4, 1.0);
    let mut curves = Vec::new();
    for d in [3usize, 4, 6] {
        for seed in 0..instances {
            let logical = regular(12, d, 200 + seed);
            let Some(emb) = embed_or_clique(&logical, &hw3, seed, None) else { continue };
            let Some((f_star, e_star)) = threshold(&logical, &hw3, &emb) else { continue };
            if f_star < 1e-3 {
                continue;
            }
            if let Some(c) = curve(&logical, &hw3, &emb, f_star, e_star, seeds, &schedule) {
                curves.push(c);
            }
        }
    }
    report("12-vertex 3/4/6-regular, +-J, chimera(3,3,4)", &curves);
    println!("  WHAT THE TABLE SAYS.\n");
    println!("  The P(GS) rows are the ground-state probability of the resolved answer as the chain strength");
    println!("  crosses its exact threshold: below F / F* = 1 the embedded ground state is a broken chain and");
    println!("  the answer depends on the resolution policy; above it the chains hold and the search has to");
    println!("  work against them. r < 1 on an instance means the fixed budget did best with chains the exact");
    println!("  threshold says are too weak -- the resolver was doing part of the optimisation.");
}
