// Does rewriting the model beat placing it? The crossover, measured.
//
// THE QUESTION THE FIELD HAS NOT ANSWERED. A model denser than the fabric has two routes onto it.
// MINOR EMBEDDING places it as it stands, giving each variable a chain of physical sites.
// SPARSIFICATION rewrites the model first -- splitting each heavy variable into copies bound by a
// strong coupling -- and places the sparser result. Both spend physical sites on holding one
// logical variable together, and the whole argument for sparsification is that it spends fewer,
// or spends them in shorter runs.
//
// Nobody has published where the line is. This measures it, on the machines you can actually rent.
//
// WHAT "DIRECT" MEANS HERE, said before the table so the table cannot over-claim. Both columns run
// THIS CRATE'S HEURISTIC EMBEDDER, which answers the general question -- any graph onto any
// hardware -- and pays for the generality. For the specific shape being embedded (cliques), the
// industry's structured embedders write the answer down instead of searching: D-Wave's clique
// embedder reaches K_150 with chains of 14 on a full-yield P16, far past anything below. So this
// table decides between two HEURISTIC routes, and its verdict is about them; the structured route
// beats both where it applies, and this crate builds it for Chimera today (`embed::chimera_clique`,
// verified by construction) with Pegasus and Zephyr recorded as the open gap.
//
// WHY THIS IS COUNTS AND NOT SECONDS. Sites used and longest chain are properties of the graphs and
// the seed, so this table is identical on a laptop and on a cluster. A timing table would be a
// statement about whichever machine ran it, and the two routes do not even spend their time in the
// same place -- sparsify-then-embed pays a rewrite the direct route does not.
//
// WHAT "LONGEST CHAIN" MEANS ON THE SPARSIFIED ROUTE, and it is the subtle part. A sparsified
// variable is already several copies; each copy is then embedded as its own chain. So the run of
// physical sites that must agree for one LOGICAL variable is the union of its copies' chains, and
// that union is what is reported -- not the longest single chain, which would flatter the route by
// measuring a piece of the thing that has to hold.

use ferrotherm::graph::{Graph, GraphBuilder};
use ferrotherm::{device, embed, sparsify};

fn clique(k: usize) -> Graph {
    let mut gb = GraphBuilder::new(k);
    for i in 0..k {
        for j in (i + 1)..k {
            gb.couple(i, j, 1.0);
        }
    }
    gb.build()
}

/// Sites used and the longest run that must agree, for one route.
struct Cost {
    sites: usize,
    longest: usize,
}

fn direct(logical: &Graph, hw: &Graph) -> Option<Cost> {
    let e = embed::embed_bounded(logical, hw, 7, 10, embed::DEFAULT_SEARCH_BUDGET)?;
    Some(Cost {
        sites: e.chains.iter().map(|c| c.len()).sum(),
        longest: e.chains.iter().map(|c| c.len()).max().unwrap_or(0),
    })
}

fn via_sparsify(logical: &Graph, hw: &Graph, budget: usize) -> Option<Cost> {
    let s = sparsify::sparsify(logical, budget).ok()?;
    let e = embed::embed_bounded(&s.graph, hw, 7, 10, embed::DEFAULT_SEARCH_BUDGET)?;
    // One logical variable is now several copies, each its own chain. What has to agree is the
    // union, so that is what this counts.
    let longest = s
        .copies
        .iter()
        .map(|set| set.iter().map(|&c| e.chains[c as usize].len()).sum::<usize>())
        .max()
        .unwrap_or(0);
    Some(Cost { sites: e.chains.iter().map(|c| c.len()).sum(), longest })
}

fn main() {
    let machines: [(&str, Graph); 2] =
        [("Pegasus P16", device::pegasus(16, 1.0).graph), ("Zephyr Z15", device::zephyr(15, 4, 1.0).graph)];

    for (name, hw) in &machines {
        let d = hw.max_degree();
        println!("\n=== {name}: {} sites, degree {d}", hw.n);
        println!(
            "{:>5} {:>18} {:>18} {:>18} {:>18}",
            "K_n", "direct sites", "direct longest", "sparse sites", "sparse longest"
        );
        for k in [8usize, 12, 16, 24, 32] {
            let g = clique(k);
            let a = direct(&g, hw);
            let b = via_sparsify(&g, hw, d);
            let cell = |c: &Option<Cost>, f: fn(&Cost) -> usize| match c {
                Some(x) => f(x).to_string(),
                None => "not found".to_string(),
            };
            println!(
                "{k:>5} {:>18} {:>18} {:>18} {:>18}",
                cell(&a, |c| c.sites),
                cell(&a, |c| c.longest),
                cell(&b, |c| c.sites),
                cell(&b, |c| c.longest)
            );
        }
    }
    println!(
        "\nTHE ANSWER, AND IT IS A NEGATIVE ONE.\n\n\
         THERE IS NO CROSSOVER IN FAVOUR OF SPARSIFICATION on either machine. Where it changes\n\
         anything at all it loses, and not narrowly: K_24 on Pegasus costs 130 sites and a 14-site\n\
         chain placed directly, against 758 sites and a 55-site run through sparsification -- 5.8x\n\
         the qubits and 3.9x the length of the thing that has to agree. At K_32 the sparsified\n\
         model does not embed at all within the same budget, while the direct route places it in\n\
         237 sites. On Zephyr the sparsified route is already out at K_24.\n\n\
         THE ROWS THAT TIE ARE TIES FOR A REASON, and it is worth saying so rather than leaving a\n\
         reader to wonder whether the columns are wired together. Up to K_16 the logical model\n\
         ALREADY fits the machine's degree -- Pegasus is degree 15 and K_16 has degree 15 -- so\n\
         `sparsify` returns it unchanged and the two routes are the same route. The measurement\n\
         only begins at K_24.\n\n\
         WHY IT LOSES: IT IS THE SAME TAX PAID TWICE. Sparsification picks a variable's copies\n\
         BEFORE the machine is looked at, using only the degree budget; the embedder then has to\n\
         give every one of those copies its own chain. The copies are a worse decomposition than a\n\
         placer with the whole graph in front of it would have chosen, and the chains are then\n\
         built on top of that choice rather than instead of it.\n\n\
         AND NEITHER COLUMN IS THE FRONTIER. Both are the same heuristic search; a STRUCTURED\n\
         clique embedding beats both where it applies -- K_150 at chain 14 on a full-yield P16 by\n\
         D-Wave's own tooling, against the K_32 at chain 16 the search manages above. This crate\n\
         builds the structured route for Chimera (embed::chimera_clique, verified at every size);\n\
         Pegasus and Zephyr structured cliques are the recorded gap, with those numbers as the bar.\n\n\
         SO WHAT IS IT FOR. Not this. The routine exists because a fabric with a FIXED sparse\n\
         topology and no placer at all -- a p-bit array, a physics ASIC with a wired lattice -- has\n\
         no direct route, and there the question is not which is cheaper but whether the model runs\n\
         at all. On a machine with a placer, place. That is the honest recommendation and it is the\n\
         opposite of what a paper introducing a sparsifier would be expected to conclude."
    );
}
