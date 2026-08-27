// Exact max-cut on a planar spin glass, at sizes no search in this crate can reach.
//
// `exact_reach` measured where branch and bound stops: 76 spins sparse, 52 dense, and that is with
// a certified SDP bound inside the tree. Everything else here SEARCHES, and a search on a hard
// instance returns the best it found.
//
// This does not search. Max-cut is NP-hard in general and POLYNOMIAL on a planar graph, and the
// difference is a theorem rather than an engineering margin: a cut in the graph is a cycle in the
// dual, so the problem becomes a minimum-weight T-join and then a minimum-weight perfect matching.
// The answer is the maximum, not the best found, and there is no budget to run out of.
//
// The comparison printed is against breakout local search on the same instance -- the algorithm
// that holds the max-cut record on most of G-set. It is the right control precisely because it is
// good: a heuristic that cannot reach the optimum on a 10,000-spin planar glass is not a bad
// heuristic, it is a demonstration that structure beats search when the structure is there.
//
// NOT run in CI: the largest sizes take minutes.
//
// run: cargo run --release --example planar_exact

use ferrotherm::graph::{Graph, GraphBuilder};
use ferrotherm::host::Timing;
use ferrotherm::rng::Pcg;
use ferrotherm::{bls, planarcut};

/// A planar spin glass: a `w x h` grid with couplings drawn uniformly from {-1, +1}.
fn glass(w: usize, h: usize, seed: u64) -> Graph {
    let mut rng = Pcg::new(seed, 0x091A_5533);
    let mut gb = GraphBuilder::new(w * h);
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            if x + 1 < w {
                gb.couple(i, i + 1, if rng.f64() < 0.5 { 1.0 } else { -1.0 });
            }
            if y + 1 < h {
                gb.couple(i, i + w, if rng.f64() < 0.5 { 1.0 } else { -1.0 });
            }
        }
    }
    gb.build()
}

fn cut_of(g: &Graph, s: &[i8]) -> f64 {
    let mut c = 0.0;
    for u in 0..g.n {
        for k in g.offset[u]..g.offset[u + 1] {
            if (g.nbr[k] as usize) > u && s[u] != s[g.nbr[k] as usize] {
                c -= g.w[k]; // max-cut weights are the NEGATED couplings
            }
        }
    }
    c
}

fn main() {
    println!("exact max-cut on planar spin glasses, couplings uniform in {{-1,+1}}\n");
    println!("  {:>9} {:>7} {:>6} {:>10} {:>12} {:>12} {:>8}",
             "grid", "spins", "faces", "odd faces", "EXACT cut", "BLS cut", "BLS gap");

    let mut dirty: Option<String> = None;
    for (w, h) in [(10usize, 10usize), (20, 20), (40, 40), (60, 60), (100, 100)] {
        let g = glass(w, h, 7);
        let (r, t) = Timing::around(|| planarcut::solve(&g, &planarcut::Params::default()));
        dirty = dirty.or_else(|| t.caveat());
        let out = match r {
            Ok(o) => o,
            Err(e) => {
                println!("  {:>9} {:>7} -- refused: {e}", format!("{w}x{h}"), w * h);
                continue;
            }
        };
        // The control gets a budget scaled to the instance so it is not being starved.
        let iters = (w * h * 200).max(100_000);
        let heur = bls::search(&g, &bls::Params { iterations: iters, ..Default::default() }, 5);
        let hcut = cut_of(&g, &heur.state);
        // A heuristic can never beat an exact answer. If it does, the "exact" one is wrong -- so
        // this is a running check rather than a decoration.
        assert!(
            hcut <= out.cut + 1e-9,
            "{w}x{h}: breakout local search found {hcut}, above a claimed EXACT maximum of {}",
            out.cut
        );
        println!("  {:>9} {:>7} {:>6} {:>10} {:>12.0} {:>12.0} {:>7.2}%",
                 format!("{w}x{h}"), w * h, out.faces, out.odd_faces, out.cut, hcut,
                 (out.cut - hcut) / out.cut * 100.0);
    }

    if let Some(c) = &dirty {
        println!("\nNOTE: {c}");
        println!("No timing is reported above for exactly that reason. The CUTS are unaffected: an");
        println!("exact optimum is the same number whoever else is on the CPU.");
    }
    println!("\nREADING: the `EXACT cut` column is the maximum, not the best found. `odd faces` is");
    println!("the size of the matching problem and the real cost driver -- it is what makes this");
    println!("O(n^3) rather than O(2^n), and it is why a 10,000-spin instance is a few minutes");
    println!("rather than the age of the universe.");
}
