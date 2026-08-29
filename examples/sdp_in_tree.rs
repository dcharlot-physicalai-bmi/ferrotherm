// Does a certified SDP bound INSIDE the branch-and-bound tree pay, and at what depth?
//
// `exact_reach` measured where the cheap bound runs out: 76 spins at mean degree 6, 44 at mean
// degree 22. Its READING said the dense column is where an SDP bound inside the tree would start to
// pay, and at depth 2 that turned out to be wrong -- 21 Choleskys per instance, 0 to 4 prunes, node
// counts unchanged on 17 of 19 sizes, and the reach not extended by a single spin.
//
// One setting is not a sweep. Depth 2 is at most seven nodes: even a perfect bound there removes a
// constant fraction of the tree, and the reach is set by what happens where the nodes actually are.
// This sweeps the depth on the instances where the cheap bound is genuinely struggling, and reports
// nodes against calls so a bound that RAN is never mistaken for a bound that HELPED.
//
// NOT run in CI: it is a measurement whose whole point is to run the expensive dial to the point
// where it stops being worth it.
//
// run: cargo run --release --example sdp_in_tree

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
    const BUDGET: u64 = 40_000_000;
    let sizes: [(usize, f64); 3] = [(32, 0.5), (36, 0.5), (40, 0.5)];
    // Measured once at depth 2 and the answer was "does nothing" -- 21 calls, 0-4 prunes, node
    // counts unchanged. That was one arbitrary setting reported as a property of the method: depth
    // 2 is at most seven nodes, so even a perfect bound there removes a constant fraction of a tree
    // whose cost is set exponentially deeper. The sweep runs until the ratio stops falling.
    let depths: [Option<usize>; 6] = [None, Some(4), Some(8), Some(12), Some(16), Some(20)];

    println!("SDP-in-tree depth sweep, dense instances, node budget {BUDGET}, tabu incumbent, 3 seeds\n");
    println!("  {:>6} {:>8} {:>14} {:>8} {:>12} {:>10}",
             "spins", "depth", "median nodes", "vs cheap", "prunes/calls", "median s");

    let mut dirty: Option<String> = None;
    for (n, p) in sizes {
        let mut baseline = 0f64;
        for depth in depths {
            let (mut nodes, mut secs) = (Vec::new(), Vec::new());
            let (mut fired, mut bit, mut proved) = (0u64, 0u64, 0usize);
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
                        sdp_depth: depth,
                        ..branch::Params::default()
                    })
                });
                nodes.push(out.nodes);
                secs.push(t.seconds);
                fired += out.sdp_calls;
                bit += out.sdp_prunes;
                proved += usize::from(out.proved_optimal);
                if dirty.is_none() {
                    dirty = t.caveat();
                }
            }
            assert_eq!(proved, 3, "n={n} depth={depth:?} failed to prove within the budget");
            nodes.sort_unstable();
            secs.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
            let med = nodes[1] as f64;
            if depth.is_none() {
                baseline = med;
            }
            // Against the PAIRED baseline -- same instances, same incumbents, same seeds -- because
            // a ratio against some other run's numbers would measure the instances, not the dial.
            let vs = if baseline > 0.0 { format!("{:.2}x", med / baseline) } else { "--".into() };
            let label = depth.map_or("cheap".to_string(), |d| format!("d{d}"));
            let sdp = if depth.is_none() { "--".to_string() } else { format!("{bit}/{fired}") };
            let clock = if dirty.is_some() { " spoiled".to_string() } else { format!("{:>10.2}", secs[1]) };
            println!("  {n:>6} {label:>8} {:>14} {vs:>8} {sdp:>12} {clock}", nodes[1]);
        }
        println!();
    }

    if let Some(c) = &dirty {
        println!("NOTE ON THE `median s` COLUMN: {c}\n");
    }
    println!("READING: `vs cheap` below 1.00 is the dial working. It is the only column that can");
    println!("say so -- a bound that fires at every node and prunes at none leaves this at 1.00");
    println!("while looking busy in `prunes/calls`, and the wall clock cannot separate the two on a");
    println!("machine this contended. Where the ratio stops falling is where the extra Choleskys");
    println!("stop buying anything, and that depth is the answer this example exists to give.");
}
