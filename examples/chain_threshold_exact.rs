#![allow(missing_docs)]
// THE EXACT MINIMAL CHAIN STRENGTH F*(I, E), and how it scales with logical degree.
//
// Choi (2008) proved that a chain coupling above |h_i| + sum_j |J_ij| preserves the ground state
// and asked how tight that is -- 'can one get a better bound without solving the original
// problem?' -- and the question has stood since. D-Wave's uniform torque compensation guesses
// 1.414 rms(J) sqrt(mean degree) instead, which grows as the root of the degree where Choi's bound
// grows linearly. On instances small enough that the embedded model's ground energy is exact
// (`exact::Elimination`, width under 24), the threshold itself is computable:
//
//   F*(I, E) = the smallest F at which the embedded model's ground energy equals the energy of the
//              best CHAIN-ALIGNED state, E_I* - F x (chain edges) -- below it some broken-chain
//              state is strictly lower, at it the two tie, above it every ground state unembeds to
//              a ground state of I.
//
// Found by bisection on F, each step one exact elimination, to a relative 1e-4. Two ensembles at
// rms |J| = 1 and no fields, so degree is the only thing that moves: complete graphs K_n (degree
// n - 1) on a 2x2x4 Chimera, and random d-regular graphs on 12 variables on a 3x3x4 Chimera. For
// each instance: F* against Choi's bound (which is the degree here), against UTC, and whether UTC
// would have broken a chain. Then the slope of log F* against log degree, which is the answer to
// 'd or sqrt(d)'.
//
// run: cargo run --release --example chain_threshold_exact

use ferrotherm::embed::{apply_with, chimera_clique, embed_bounded, Embedding, DEFAULT_SEARCH_BUDGET};
use ferrotherm::exact::Elimination;
use ferrotherm::graph::{Graph, GraphBuilder};
use ferrotherm::ising::chimera;
use ferrotherm::rng::Pcg;
use ferrotherm::torque;

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

/// A random d-regular simple graph on n vertices by the pairing model, retried until simple.
fn regular(n: usize, d: usize, seed: u64) -> Graph {
    let mut rng = Pcg::new(seed, 0xD3);
    // A random pairing is simple with probability about exp(-(d^2 - 1) / 4): one in 6,000 at d = 6.
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

/// Hardware edges inside chains: what every aligned state satisfies.
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

struct Threshold {
    f_star: f64,
    choi: f64,
    utc: f64,
    longest_chain: usize,
}

/// F* by bisection; `None` when the instance could not be embedded or eliminated.
fn threshold(logical: &Graph, hw: &Graph, emb: &Embedding) -> Option<Threshold> {
    let elim = Elimination::default();
    let e_star = elim.ground_state(logical).ok()?.ground_energy?;
    let n_ce = chain_edges(emb, hw) as f64;
    let ground = |f: f64| -> Option<f64> { elim.ground_state(&apply_with(logical, hw, emb, f).graph).ok()?.ground_energy };
    let aligned_ok = |f: f64| -> Option<bool> { Some(ground(f)? >= e_star - f * n_ce - 1e-9) };
    let c = choi(logical);
    let mut hi = 2.0 * c.max(1.0);
    // The self-check that the rewrite preserved the logical energies: at a strength no broken
    // chain can beat, the embedded ground energy is exactly E_I* - F x chain edges.
    let g_hi = ground(hi)?;
    assert!((g_hi - (e_star - hi * n_ce)).abs() < 1e-9, "the embedded model at F = {hi} is not the logical one: {g_hi} vs {}", e_star - hi * n_ce);
    let mut lo = 0.0;
    if aligned_ok(0.0)? {
        return Some(Threshold { f_star: 0.0, choi: c, utc: torque::strength(logical), longest_chain: emb.longest_chain() });
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
    Some(Threshold { f_star: hi, choi: c, utc: torque::strength(logical), longest_chain: emb.longest_chain() })
}

fn embed_or_clique(logical: &Graph, hw: &Graph, seed: u64, clique_m: Option<usize>) -> Option<Embedding> {
    if let Some(e) = embed_bounded(logical, hw, seed, 10, DEFAULT_SEARCH_BUDGET) {
        return Some(e);
    }
    let m = clique_m?;
    let e = chimera_clique(m, 4)?;
    (e.chains.len() >= logical.n).then(|| Embedding { chains: e.chains[..logical.n].to_vec(), ..e })
}

fn slope(xs: &[f64], ys: &[f64]) -> f64 {
    let n = xs.len() as f64;
    let (mx, my) = (xs.iter().sum::<f64>() / n, ys.iter().sum::<f64>() / n);
    let sxy: f64 = xs.iter().zip(ys).map(|(x, y)| (x - mx) * (y - my)).sum();
    let sxx: f64 = xs.iter().map(|x| (x - mx).powi(2)).sum();
    sxy / sxx
}

fn report(name: &str, rows: &[(f64, Vec<Threshold>)]) {
    println!("  {name}\n");
    println!(
        "  {:>6} {:>5}   {:>8} {:>8} {:>8} {:>5}   {:>8} {:>10}   {:>9} {:>6}",
        "degree", "inst", "F* mean", "F* max", "F* min", "F*=0", "F*/Choi", "UTC/F* med", "UTC<F*", "chain"
    );
    let (mut lx, mut ly) = (Vec::new(), Vec::new());
    for (d, ts) in rows {
        if ts.is_empty() {
            println!("  {d:>6.0} {:>5}   (no instance embedded)", 0);
            continue;
        }
        let n = ts.len() as f64;
        let mean = ts.iter().map(|t| t.f_star).sum::<f64>() / n;
        let max = ts.iter().map(|t| t.f_star).fold(0.0f64, f64::max);
        let min = ts.iter().map(|t| t.f_star).fold(f64::INFINITY, f64::min);
        let ratio = ts.iter().map(|t| t.f_star / t.choi).sum::<f64>() / n;
        let zero = ts.iter().filter(|t| t.f_star == 0.0).count();
        // The ratio to UTC over the instances that need a chain at all, as a median: an instance
        // with F* = 0 has an infinite ratio and says nothing about the rule.
        let mut ratios: Vec<f64> = ts.iter().filter(|t| t.f_star > 0.0).map(|t| t.utc / t.f_star).collect();
        ratios.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let utc = if ratios.is_empty() { f64::NAN } else { ratios[ratios.len() / 2] };
        let broke = ts.iter().filter(|t| t.utc < t.f_star).count();
        let chain = ts.iter().map(|t| t.longest_chain).max().unwrap_or(0);
        println!(
            "  {d:>6.0} {:>5}   {mean:>8.3} {max:>8.3} {min:>8.3} {zero:>5}   {ratio:>8.3} {utc:>10.3}   {:>4} of {:>2}   {chain:>6}",
            ts.len(),
            broke,
            ts.len()
        );
        lx.push(d.ln());
        ly.push(mean.ln());
    }
    if lx.len() >= 2 {
        println!("\n  slope of log F*(mean) against log degree: {:.2}   (1 = Choi's linear rule, 0.5 = the torque rule)\n", slope(&lx, &ly));
    }
}

fn main() {
    let instances = 20u64;
    println!("THE EXACT MINIMAL CHAIN STRENGTH, BY BISECTION ON AN EXACT ELIMINATION\n");
    println!("  F*       smallest F at which the embedded ground energy equals E_I* - F x (chain edges); relative 1e-4");
    println!("  Choi     max_i (|h_i| + sum_j |J_ij|), the sufficient strength of Choi's Theorem 4.1 (the degree, at +-J and no field)");
    println!("  UTC      torque::strength = 1.414 rms(J) sqrt(mean degree); 'UTC<F*' counts instances whose chains break at UTC");
    println!("  hardware K_n on chimera(2,2,4) (32 sites); 12-vertex d-regular on chimera(3,3,4) (72 sites); {instances} instances each\n");

    let hw2 = chimera(2, 2, 4, 1.0);
    let mut rows = Vec::new();
    for n in [4usize, 5, 6, 7, 8] {
        let mut ts = Vec::new();
        for seed in 0..instances {
            let logical = complete(n, 100 + seed);
            let Some(emb) = embed_or_clique(&logical, &hw2, seed, Some(2)) else { continue };
            if let Some(t) = threshold(&logical, &hw2, &emb) {
                ts.push(t);
            }
        }
        rows.push(((n - 1) as f64, ts));
    }
    report("K_n, +-J, no fields", &rows);

    let hw3 = chimera(3, 3, 4, 1.0);
    let mut rows = Vec::new();
    for d in [3usize, 4, 6] {
        let mut ts = Vec::new();
        for seed in 0..instances {
            let logical = regular(12, d, 200 + seed);
            let Some(emb) = embed_or_clique(&logical, &hw3, seed, None) else { continue };
            if let Some(t) = threshold(&logical, &hw3, &emb) {
                ts.push(t);
            }
        }
        rows.push((d as f64, ts));
    }
    report("random d-regular on 12 vertices, +-J, no fields", &rows);

    println!("  WHAT THE TABLE SAYS.\n");
    println!("  F*/Choi is how much of Choi's sufficient strength the instance actually needs; the slope is how F*");
    println!("  grows with degree when nothing else changes. UTC/F* above 1 is a chain that holds at the torque");
    println!("  rule's strength, below 1 a chain that breaks in the ground state at that strength -- and the");
    println!("  count says on how many instances the torque rule under-provisions. Every F* is exact for its");
    println!("  instance and embedding: F* depends on the embedding (chain lengths, which edges carry which");
    println!("  coupling), so the same logical graph has a different F* on a different placement.");
}
