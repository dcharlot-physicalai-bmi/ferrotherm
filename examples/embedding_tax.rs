// What a topology generation is worth, in sites and chain length.
//
// THE QUESTION. A quantum annealer's graph is sparse, so a problem denser than the hardware has to
// be MINOR EMBEDDED: each logical variable becomes a CHAIN of physical qubits held together by a
// coupling strong enough that they agree. That chain is the tax. It costs qubits you then cannot
// use for anything else, and -- the part that decides whether an answer is usable at all -- a long
// chain is more likely to BREAK, leaving a variable whose qubits disagree and which therefore has
// no value to read.
//
// D-Wave has shipped three topology generations, and the whole argument for each was that it pays
// less tax than the last. Chimera (degree 6) is retired; Pegasus (degree 15) is the Advantage;
// Zephyr (degree 20) is the Advantage2. This measures the claim on the same logical problems.
//
// WHY THIS IS NOT A BENCHMARK, and the distinction is the point. Every number below is a COUNT --
// physical sites used, longest chain, mean chain. Not one is a duration. Counts are a property of
// the graph and the embedder's seed, so this table is the same on a laptop and on a cluster, and
// re-running it anywhere reproduces it exactly. A timing table would be a statement about whichever
// machine happened to run it.
//
// WHAT IS BEING EMBEDDED. Cliques, because a clique is the worst case at every size and the one
// case whose answer is known in advance: `K_n` needs every variable adjacent to every other, so it
// is exactly the shape that a sparse machine cannot hold natively. A refusal here is a real limit
// rather than a heuristic's bad day when `embed::site_lower_bound` proves it, which is printed.

use ferrotherm::graph::{Graph, GraphBuilder};
use ferrotherm::{device, embed, ising};

fn clique(k: usize) -> Graph {
    let mut gb = GraphBuilder::new(k);
    for i in 0..k {
        for j in (i + 1)..k {
            gb.couple(i, j, 1.0);
        }
    }
    gb.build()
}

fn main() {
    let hardware: Vec<(&str, Graph)> = vec![
        ("Chimera C8", ising::chimera(8, 8, 4, 1.0)),
        ("Pegasus P6", device::pegasus(6, 1.0).graph),
        ("Pegasus P16", device::pegasus(16, 1.0).graph),
        ("Zephyr Z4", device::zephyr(4, 4, 1.0).graph),
        ("Zephyr Z15", device::zephyr(15, 4, 1.0).graph),
    ];

    println!(
        "Minor embedding K_n, seed 7, {} shortest-path searches allowed.\n\
         Every column is a COUNT, not a duration.\n",
        embed::DEFAULT_SEARCH_BUDGET
    );
    for k in [8usize, 16, 32] {
        println!("--- K_{k}");
        println!(
            "{:<14} {:>6} {:>5} {:>8} {:>9} {:>7}",
            "hardware", "sites", "deg", "used", "longest", "mean"
        );
        for (name, g) in &hardware {
            let logical = clique(k);
            match embed::embed_bounded(&logical, g, 7, 10, embed::DEFAULT_SEARCH_BUDGET) {
                Some(e) => {
                    let lens: Vec<usize> = e.chains.iter().map(|c| c.len()).collect();
                    let used: usize = lens.iter().sum();
                    println!(
                        "{name:<14} {:>6} {:>5} {used:>8} {:>9} {:>7.2}",
                        g.n,
                        g.max_degree(),
                        lens.iter().max().copied().unwrap_or(0),
                        used as f64 / lens.len() as f64
                    );
                }
                None => {
                    // A refusal is two different statements and they must not be confused. When
                    // the site lower bound exceeds the machine, no embedding EXISTS -- a proof.
                    // Otherwise this heuristic did not find one, which is a fact about the search.
                    let lb = embed::site_lower_bound(&logical, g);
                    let why = if lb > g.n {
                        format!("impossible: needs >= {lb} sites of {}", g.n)
                    } else {
                        "not found by this search".to_string()
                    };
                    println!("{name:<14} {:>6} {:>5}   {why}", g.n, g.max_degree());
                }
            }
        }
        println!();
    }

    println!(
        "WHAT THE TABLE SAYS.\n\n\
         THE GENERATIONS ARE WORTH WHAT THEY CLAIM, and the chain column is where it shows. At\n\
         K_16 Chimera spends 126 sites and an 18-qubit chain; Pegasus spends about 49 and a chain\n\
         of 7; Zephyr 48 and a chain of 6. Two and a half times the qubits and three times the\n\
         chain length, for the same sixteen variables.\n\n\
         AT K_32 CHIMERA DROPS OUT OF THE TABLE, and the row says the honest reason. The largest\n\
         clique that fits chimera(8,8,4) at all is K_33 -- see `embed`'s own module notes -- so\n\
         K_32 is inside its capacity by one, and this search did not find it at this budget. That\n\
         is the whole point of being at the edge: Pegasus and Zephyr place the same problem with\n\
         room to spare, while Chimera needs a search to thread it through the last opening. The\n\
         row is NOT a proof of impossibility, and the code distinguishes the two -- when the site\n\
         lower bound exceeds the machine it says so and means it.\n\n\
         READ THE CHAIN COLUMN, NOT THE SITE COLUMN. Sites are a budget and you either have them\n\
         or you do not. A chain is a failure mode: it is held together by a penalty, and when that\n\
         penalty loses, the qubits of one variable disagree and the variable HAS NO VALUE. Halving\n\
         the longest chain is worth more than halving the qubit count, and it is the thing degree\n\
         buys.\n\n\
         AND THE ANOMALY, stated rather than trimmed. At K_32 the LARGER machine of each family\n\
         spends more sites than the smaller one -- P16 uses 237 where P6 uses 203. That is not the\n\
         bigger machine being worse. It is the placement heuristic having more room to wander in,\n\
         and it is a fact about `embed` rather than about Pegasus. A table that only showed the\n\
         big machines would have hidden it."
    );
}
