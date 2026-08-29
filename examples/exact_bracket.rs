// The bracket, closed. A GATE example: a non-zero exit means one of this crate's bounds is unsound.
//
// Everywhere else in this repository a bound is checked against enumeration on graphs small enough
// to enumerate (<= 20 spins) or against a published best-known cut, which is itself only a lower
// bound. Neither reaches the interesting size. `branch` does: it returns the true minimum with a
// proof, so at 22 spins -- a 4-million-state space, 256x what the unit tests' `brute_min` walks --
// every bound in the crate can be held against ground truth on every push.
//
// The check is one-sided and that is the point. A lower bound on the energy may be loose by any
// amount; it may never exceed the true minimum. If one does, the number it produced was never a
// bound, and nothing downstream that quoted it was true either.
//
// run: cargo run --release --example exact_bracket

use ferrotherm::graph::{Graph, GraphBuilder};
use ferrotherm::host::Timing;
use ferrotherm::rng::Pcg;
use ferrotherm::{bound, branch, popanneal, sdp, tabu};

fn instance(n: usize, p: f64, seed: u64) -> Graph {
    let mut rng = Pcg::new(seed, 0x8AC_E70);
    let mut gb = GraphBuilder::new(n);
    for i in 0..n {
        // Fields on, deliberately: they turn off the Z2 gauge shortcut in `branch` and they are the
        // only thing `forest`'s subgradient can move, so a fieldless instance would exercise
        // neither.
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
    // 22 and not more, for a reason about the GATE rather than about patience: a tree over 22
    // spins with fields has at most 2^23 - 1 nodes, which is under the budget below, so the search
    // cannot end without a proof. An example whose check silently stops applying when the instance
    // gets a little harder is not a gate.
    let n = 22;
    let seeds = 6u64;
    println!("bounds against a PROVED optimum, {n} spins x {seeds} instances\n");
    println!(
        "  {:>4} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>9}",
        "seed", "optimum", "decoupled", "odd-cycle", "sdp", "tabu", "pop-ann", "nodes"
    );

    let mut bad = 0usize;
    let mut total_nodes = 0u64;
    let (_, t) = Timing::around(|| {
        for seed in 0..seeds {
            let g = instance(n, 0.18, seed);

            // A heuristic incumbent first. Branch and bound with a good incumbent prunes from the
            // root; without one it spends its first descent finding what tabu found in milliseconds.
            let t = tabu::search(
                &g,
                &tabu::Params { iterations: 20_000, tenure: n / 4, restart_after: Some(2_000) , start: None },
                seed,
            );
            let pa = popanneal::run(
                &g,
                &popanneal::Params::linear_from_zero(200, 3, 8.0, 40),
                seed,
            );

            let ex = branch::solve(
                &g,
                &branch::Params {
                    max_nodes: 20_000_000,
                    incumbent: Some(t.state.clone()),
                    ..branch::Params::default()
                },
            );
            total_nodes += ex.nodes;
            if !ex.proved_optimal {
                eprintln!(
                    "\n  ** seed {seed}: the search hit its node budget after {} nodes, so there \
                     is no proved optimum to check the bounds against. This example is sized \
                     wrong, not merely slow: without the proof it checks nothing. **",
                    ex.nodes
                );
                std::process::exit(2);
            }

            let d = bound::decoupled(&g);
            let c = bound::odd_cycle(&g, 6);
            let (sd, cert) = sdp::certified(&g, &sdp::Params::default(), seed);
            // The certificate is re-verified from the graph, not trusted, exactly as `gset_gap`
            // does it -- a bound whose proof is only checked by its own author is not a bound.
            let sdp_value = match cert.verify(&g) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("\n  ** seed {seed}: the SDP certificate did not re-verify: {e} **");
                    std::process::exit(1);
                }
            };

            println!(
                "  {seed:>4} {:>10.3} {:>10.3} {:>10.3} {:>10.3} {:>10.3} {:>10.3} {:>9}",
                ex.energy, d.value, c.value, sd.value, t.energy, pa.energy, ex.nodes
            );

            // --- the gate ---------------------------------------------------------------------
            //
            // Tolerance, not exact comparison: `sdp` snaps its dual point onto a power-of-two grid
            // and `branch` carries a rounding slack, so both carry a few ulps of the instance
            // scale. It is set well below any real violation -- a bound that is genuinely unsound
            // is wrong by a term of the energy, not by 1e-9.
            let tol = 1e-9;
            for (name, v) in [
                ("decoupled", d.value),
                ("odd_cycle", c.value),
                ("sdp", sd.value),
                ("sdp (re-verified)", sdp_value),
            ] {
                if v > ex.energy + tol {
                    eprintln!(
                        "\n  ** UNSOUND: seed {seed}, `{name}` returned {v:.9} as a LOWER bound on \
                         the ground energy, and the proved ground energy is {:.9}. A state with \
                         that energy exists, so the bound is not one. **",
                        ex.energy
                    );
                    bad += 1;
                }
            }
            // The heuristics are the other side of the bracket and cannot beat the optimum either.
            for (name, v) in [("tabu", t.energy), ("popanneal", pa.energy)] {
                if v < ex.energy - tol {
                    eprintln!(
                        "\n  ** IMPOSSIBLE: seed {seed}, `{name}` reports energy {v:.9}, below the \
                         PROVED minimum {:.9}. Either the search is scoring states with a \
                         different energy function or the proof is wrong. **",
                        ex.energy
                    );
                    bad += 1;
                }
            }
        }
    });

    println!("\n  {total_nodes} nodes across {seeds} proofs ({t})");
    if bad > 0 {
        eprintln!("\n{bad} bound violation(s). This is a soundness failure, not a regression.");
        std::process::exit(1);
    }
    println!("  every bound sits at or below the proved optimum, and every heuristic at or above.");
    println!("\nREADING: the gap between `decoupled` and `optimum` is what the harder bounds are");
    println!("competing to close, and the gap between `optimum` and `tabu` is what the heuristics");
    println!("are competing to close. At this size both are measurable against truth; at G-set size");
    println!("neither is, which is why `gset_gap` reports a bracket instead of a number.");
}
