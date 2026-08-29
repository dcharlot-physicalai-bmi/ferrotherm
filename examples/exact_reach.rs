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
    // The dial under test. `None` is the incremental bound everywhere -- what every measurement
    // before 0.26.0 was taken with; `Some(d)` spends a certified SDP bound on the residual at every
    // node down to depth `d`. Both are run on the SAME instances so the comparison is paired.
    // Depth 12, not 2. The first paired run of this table used depth 2 and reported that the SDP
    // bound did nothing -- which was true of depth 2 and false of the method: `sdp_in_tree` sweeps
    // the dial and finds 0.01x the nodes on dense instances once it is deep enough to reach where
    // the nodes actually are.
    let dials: [(&str, Option<usize>); 2] = [("cheap", None), ("sdp d12", Some(12))];
    println!("branch-and-bound reach, node budget {BUDGET}, tabu incumbent, 3 seeds per size\n");
    println!("  {:>7} {:>7} {:>6} {:>7} {:>14} {:>9} {:>12} {:>10}",
             "family", "spins", "deg", "bound", "median nodes", "proved", "sdp fired", "median s");
    let mut dirty: Option<String> = None;

    for (family, degree) in [("sparse", 6.0f64), ("dense", 0.0)] {
        let mut n = 16usize;
        loop {
            let p = if degree > 0.0 { (degree / (n - 1) as f64).min(1.0) } else { 0.5 };
            let mean_deg = 2.0 * instance(n, p, 0).n_edges as f64 / n as f64;
            let mut best_proved = 0usize;
            for (dial, sdp_depth) in dials {
                let mut nodes = Vec::new();
                let mut secs = Vec::new();
                let mut proved = 0usize;
                let mut fired = 0u64;
                let mut bit = 0u64;
                for seed in 0..3u64 {
                    let g = instance(n, p, seed);
                    let inc = tabu::search(
                        &g,
                        &tabu::Params { iterations: 20_000, tenure: 0, restart_after: Some(2_000) , start: None },
                        seed,
                    );
                    let (out, t) = Timing::around(|| {
                        branch::solve(&g, &branch::Params {
                            max_nodes: BUDGET,
                            incumbent: Some(inc.state.clone()),
                            sdp_depth,
                            ..branch::Params::default()
                        })
                    });
                    nodes.push(out.nodes);
                    secs.push(t.seconds);
                    if dirty.is_none() {
                        dirty = t.caveat();
                    }
                    proved += usize::from(out.proved_optimal);
                    fired += out.sdp_calls;
                    bit += out.sdp_prunes;
                }
                nodes.sort_unstable();
                secs.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
                let clock = if dirty.is_some() {
                    "  spoiled".to_string()
                } else {
                    format!("{:>10.2}", secs[1])
                };
                // Calls AND prunes: a bound that fires a hundred times and cuts nothing is a
                // hundred wasted Choleskys, and "it ran" would read as "it helped".
                let sdp = if sdp_depth.is_none() {
                    "--".to_string()
                } else {
                    format!("{bit}/{fired}")
                };
                println!("  {family:>7} {n:>7} {mean_deg:>6.1} {dial:>7} {:>14} {proved:>6}/3 {sdp:>12} {clock}",
                         nodes[1]);
                best_proved = best_proved.max(proved);
            }
            // Stop at the first size NEITHER dial can prove for a majority of seeds: past that the
            // numbers are about the budget rather than about the solver.
            if best_proved < 2 {
                println!("  {family:>7} stops here -- neither bound proved a majority at {n} spins\n");
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
    println!("READING: the cheap bound is `decoupled` on the residual, and it charges for every edge");
    println!("with both ends still FREE. A sparse graph has O(n) of those, so a few fixings retire");
    println!("most of them and the bound becomes informative early; a dense one has O(n^2), so the");
    println!("root bound sits far below the optimum and stays loose for many levels. That is why");
    println!("density costs so much more here than node count does.");
    println!();
    println!("`sdp d2` replaces it with a certified semidefinite bound down to depth 2. Read the");
    println!("`sdp fired` column as prunes/calls: calls alone would say it RAN, which is not the");
    println!("same as saying it HELPED. Judge the dial on median nodes against the paired `cheap`");
    println!("row -- same instances, same incumbents, same seeds.");
}
