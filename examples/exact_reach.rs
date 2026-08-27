// How far does the exact solver actually reach? The number `branch` cannot report about itself.
//
// `exact_bracket` proves optima at 22 spins because that size is chosen to ALWAYS prove -- a tree
// over 22 spins with fields has at most 2^23 - 1 nodes, under the budget, so the gate cannot
// silently stop applying. That makes it useless for the question a user actually has, which is how
// big an instance they can hand this thing.
//
// So this walks the size up until the budget runs out, on two families, and reports where it
// stopped. Density is the variable that matters: the bound charges for edges with both ends still
// free, and fixing a high-degree spin removes many of them at once, so a sparse graph of the same
// node count is a completely different problem from a dense one.
//
// NOT run in CI. It is a measurement, not a gate -- it deliberately runs until it fails, which is
// the opposite of what a per-push check should do.
//
// run: cargo run --release --example exact_reach

use ferrotherm::graph::{Graph, GraphBuilder};
use ferrotherm::host::Timing;
use ferrotherm::rng::Pcg;
use ferrotherm::{branch, tabu};

fn instance(n: usize, p: f64, seed: u64) -> Graph {
    let mut rng = Pcg::new(seed, 0x2EAC);
    let mut gb = GraphBuilder::new(n);
    for i in 0..n {
        gb.bias(i, rng.f64() - 0.5);
        for j in (i + 1)..n {
            if rng.f64() < p {
                gb.couple(i, j, rng.f64() * 2.0 - 1.0);
            }
        }
    }
    gb.build()
}

fn main() {
    // ANNOTATES rather than refuses, and the distinction is the one `host` is built around. The
    // answer this example exists to give -- the largest instance that still proves, and how many
    // nodes it took -- is a SEARCH OUTCOME: the same number whoever else is using the CPU. Only the
    // seconds column is a rate. An earlier draft of this file called `require_quiet` and exited on
    // a busy machine, which would have withheld the reach because the clock beside it was spoiled.
    const BUDGET: u64 = 40_000_000;
    println!("branch-and-bound reach, node budget {BUDGET}, tabu incumbent, 3 seeds per size\n");
    println!("  {:>7} {:>7} {:>6} {:>14} {:>9} {:>10}", "family", "spins", "deg", "median nodes", "proved", "median s");
    let mut dirty: Option<String> = None;

    for (family, degree) in [("sparse", 6.0f64), ("dense", 0.0)] {
        let mut n = 16usize;
        loop {
            let p = if degree > 0.0 { (degree / (n - 1) as f64).min(1.0) } else { 0.5 };
            let mut nodes = Vec::new();
            let mut secs = Vec::new();
            let mut proved = 0usize;
            for seed in 0..3u64 {
                let g = instance(n, p, seed);
                let inc = tabu::search(
                    &g,
                    &tabu::Params { iterations: 20_000, tenure: 0, restart_after: Some(2_000) },
                    seed,
                );
                let (out, t) = Timing::around(|| {
                    branch::solve(&g, &branch::Params { max_nodes: BUDGET, incumbent: Some(inc.state.clone()) })
                });
                nodes.push(out.nodes);
                secs.push(t.seconds);
                if dirty.is_none() {
                    dirty = t.caveat();
                }
                proved += usize::from(out.proved_optimal);
            }
            nodes.sort_unstable();
            secs.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
            let mean_deg = 2.0 * instance(n, p, 0).n_edges as f64 / n as f64;
            let clock = if dirty.is_some() {
                "  spoiled".to_string()
            } else {
                format!("{:>10.2}", secs[1])
            };
            println!("  {family:>7} {n:>7} {mean_deg:>6.1} {:>14} {proved:>6}/3 {clock}", nodes[1]);
            // Stop at the first size where a majority of seeds could not be proved: past that the
            // numbers are about the budget rather than about the solver.
            if proved < 2 {
                println!("  {family:>7} stops here -- {} of 3 proved at {n} spins\n", proved);
                break;
            }
            n += 4;
            if n > 96 {
                println!("  {family:>7} still proving at 96 spins; raise the ceiling to see more\n");
                break;
            }
        }
    }

    if let Some(c) = &dirty {
        println!("NOTE ON THE `median s` COLUMN: {c}\n");
    }
    println!("READING: the bound is `decoupled` on the residual problem, and it charges for every");
    println!("edge with both ends still FREE. A sparse graph has O(n) of those, so a few fixings");
    println!("retire most of them and the bound becomes informative early; a dense one has O(n^2),");
    println!("so the root bound sits far below the optimum and stays loose for many levels. That is");
    println!("why density costs so much more here than node count does -- and it is exactly where");
    println!("an SDP bound inside the tree, rather than only at the root, would start to pay.");
}
