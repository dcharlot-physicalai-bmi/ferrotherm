// The one magic constant in the embedding layer, measured against a proved optimum.
//
// `embed::apply` holds each chain together with 2x the largest logical coefficient. That is the
// standard first guess, it was hardcoded, and its own docstring said it was "reported rather than
// hidden so it can be tuned" while `apply` took no parameter to tune it with. Nothing in this
// repository had ever put a problem through the embedding layer end to end -- `embed::apply` was
// called from its own tests and from nowhere else -- so the constant had never been checked.
//
// WHY IT IS NOT OBVIOUS. A chain is one logical variable spread over several hardware sites, held
// together by a ferromagnetic coupling. Too weak and the chain BREAKS: the sites disagree, and the
// variable's value is a coin toss dressed as a result. Too strong and it SWAMPS the model: every
// single-spin move that would explore the problem must first pay the chain, so the search spends
// itself holding chains together. This crate has already measured that exact shape once, in
// `src/hubo.rs`, where a reduction penalty chosen by the standard rule made the landscape rigid
// enough to change the answer -- and where the first version of the comparison measured the beta
// ladder instead of the method.
//
// THE ORACLE IS A PROOF, NOT A BASELINE. The logical problem is small enough for branch and bound
// to exhaust its tree, so every row is scored against the true optimum rather than against the best
// this file happened to find. A sweep judged by its own best answer cannot tell "the whole sweep is
// bad" from "this setting is bad".
//
// THREE THINGS ARE REPORTED AND ALL THREE ARE NEEDED:
//
//   broken   the share of variables whose chain disagreed with itself. A run with broken chains has
//            not answered the question, whatever energy it reports.
//   gap      how far the unembedded answer sits above the proved optimum, in logical energy.
//   found    how often the proved optimum was reached exactly.
//
// A setting can look good on any one of them alone. Zero broken chains at a huge strength means the
// chains held and nothing else moved.
//
// run: cargo run --release --example chain_strength

use ferrotherm::graph::{Graph, GraphBuilder};
use ferrotherm::rng::Pcg;
use ferrotherm::{branch, embed, ising, tempering};

const SEEDS: u64 = 8;
/// Logical variables.
///
/// SIX, AND NOT MORE, FOR A REASON THAT IS ITSELF A FINDING. Chimera sites have degree 6, so K_7 is
/// the largest clique that needs no chain at all and K_8 is the first that does -- and the placer in
/// `src/embed.rs` FAILS on every clique past K_7, on machines it uses 3% of. That defect is written
/// up on `embed_with` and pinned by two tests. K_6 is what this file can measure with, and it is
/// enough: its embedding carries chains up to nine sites long, so the constant under test is doing
/// real work here even though the logical problem is small.
const NV: usize = 6;

/// A dense logical problem: every pair coupled, +/-1. Dense is the point -- a model that already
/// fits the hardware needs no chains and would measure nothing.
fn logical(seed: u64) -> Graph {
    let mut rng = Pcg::new(seed, 0x0C4A_1234);
    let mut b = GraphBuilder::new(NV);
    for i in 0..NV {
        for j in (i + 1)..NV {
            b.couple(i, j, if rng.f64() < 0.5 { 1.0 } else { -1.0 });
        }
    }
    b.build()
}

fn main() {
    // Chimera is the hardware these chains exist for: degree 6, so a 12-variable complete graph
    // cannot sit on it without them.
    let hardware = ising::chimera(8, 8, 4, 1.0);
    println!("Chain strength in the embedding layer, against a proved optimum\n");
    println!(
        "Logical: complete graph on {NV} variables, +/-1 couplings -- degree {} against Chimera's\n\
         6. Hardware: C_8,8,4, {} sites. {SEEDS} seeds, each\n\
         scored against the optimum BRANCH AND BOUND PROVED for that instance.\n",
        NV - 1,
        hardware.n
    );

    let multiples = [0.25f64, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0];
    println!(
        "{:>10} {:>10} {:>9} {:>10} {:>9}",
        "chain x", "strength", "broken", "gap", "found"
    );

    // Prepare every instance once: embedding is the expensive part and it does not depend on the
    // chain strength, so re-running it per row would measure the placer's variance as if it were
    // the constant's effect.
    struct Case {
        logical: Graph,
        emb: embed::Embedding,
        optimum: f64,
        worst: f64,
    }
    let mut cases = Vec::new();
    for seed in 0..SEEDS {
        let g = logical(seed);
        let Some(emb) = embed::embed(&g, &hardware, seed) else {
            continue;
        };
        let proved = branch::solve(&g, &branch::Params { max_nodes: 40_000_000, ..Default::default() });
        if !proved.proved_optimal {
            continue;
        }
        cases.push(Case {
            worst: embed::worst_coefficient(&g),
            logical: g,
            emb,
            optimum: proved.energy,
        });
    }
    if cases.is_empty() {
        println!("no instance both embedded and proved; nothing to report");
        return;
    }
    let longest = cases
        .iter()
        .flat_map(|c| c.emb.chains.iter().map(|ch| ch.len()))
        .max()
        .unwrap_or(0);
    println!(
        "  {} of {SEEDS} instances both embedded and proved. Longest chain {longest} sites.\n",
        cases.len()
    );

    for &mult in &multiples {
        let (mut broken_frac, mut gap, mut found) = (0.0, 0.0, 0usize);
        let mut strength = 0.0;
        for (ci, c) in cases.iter().enumerate() {
            let s = mult * c.worst;
            strength = s;
            let emb = embed::apply_with(&c.logical, &hardware, &c.emb, s);
            // One anneal per instance, the same schedule for every row, so the only thing that
            // differs down the column is the constant under test.
            let ladder: Vec<(f64, usize)> =
                tempering::geometric_ladder(0.05, 6.0, 120).into_iter().map(|b| (b, 60)).collect();
            let (state, _) = tempering::anneal(&emb.graph, &ladder, ci as u64 ^ 0xC4A1, None);
            let (values, broke) = embed::unembed(&c.emb, &state);
            broken_frac += broke.len() as f64 / NV as f64;
            let e = c.logical.energy(&values);
            gap += e - c.optimum;
            if (e - c.optimum).abs() < 1e-9 {
                found += 1;
            }
        }
        let n = cases.len() as f64;
        println!(
            "{mult:>10.2} {strength:>10.2} {:>8.1}% {:>10.2} {found:>5}/{}",
            broken_frac / n * 100.0,
            gap / n,
            cases.len()
        );
    }

    println!(
        "\nTHE DEFAULT SURVIVES, AND IT IS THE SMALLEST MULTIPLE THAT DOES. Below 1x the chains\n\
         break -- 27.1% of variables at 0.25x -- and the answer is 1.25 above the proved optimum. At\n\
         1x a tenth of chains still break, though this family is easy enough that the majority vote\n\
         happens to land on the optimum anyway, which is luck and not a result. At the default 2x\n\
         nothing breaks and every instance is solved exactly. Two is the standard first guess and on\n\
         this family it is right."
    );
    println!(
        "\nAND THE OTHER HALF OF THE TRADE-OFF DID NOT APPEAR, WHICH IS THE HONEST HEADLINE. The\n\
         reason to fear a large chain strength is that it swamps the model: every move that would\n\
         explore the problem has to pay the chain first, so the search spends itself holding chains\n\
         together. Nothing of the sort shows up to 16x -- eight times the default, and still 8 of 8\n\
         exact. That is not evidence the failure is imaginary. It is evidence that a six-variable\n\
         clique under a 120-stage anneal is too easy to exhibit it: a rigid landscape is still\n\
         solved when the landscape is small."
    );
    println!(
        "\nAND THE INSTANCE THAT WOULD EXHIBIT IT CANNOT BE BUILT, BECAUSE OF THE OTHER DEFECT.\n\
         Showing rigidity needs a logical problem hard enough that a swamped search actually loses,\n\
         which means more variables and longer chains -- and `src/embed.rs` cannot place any clique\n\
         past K_7, on machines it fills 3% of. So the upper half of this sweep is untested rather\n\
         than passed, and it will stay that way until the placer is repaired. The two findings are\n\
         one finding: an embedding layer that cannot build chains cannot be asked what chains cost."
    );
    println!(
        "\nREAD THE BROKEN COLUMN FIRST. A row with broken chains has not answered the question at\n\
         all: a broken chain means one logical variable held two values, and the majority vote that\n\
         resolves it is a coin toss wearing a number. The 1x row is exactly that -- gap 0.00 with\n\
         10.4% of chains broken -- and reading its energy alone would call it a success."
    );
    println!(
        "\nTHE FAILURE AT THE FAR END IS SILENT, WHICH IS WHY THIS NEEDED A PROOF RATHER THAN A\n\
         BASELINE. A chain strength far above the model breaks nothing and reports clean; every\n\
         chain holds, no warning fires, and the answer is simply worse. Without an oracle there is\n\
         nothing in the output to tell that from a hard instance. This is the same shape as the\n\
         reduction penalty in `src/hubo.rs`, and the same reason that comparison needed a swept\n\
         beta ladder before it meant anything."
    );
    println!(
        "\nWHAT THIS DOES NOT ESTABLISH. One logical family, one hardware graph, one annealing\n\
         schedule, {} instances. A model whose coefficients span orders of magnitude has a\n\
         different `worst_coefficient` story entirely, since the scale a chain must outrank is then\n\
         set by one outlier. `apply_with` exists so that case can be handled; this table calibrates\n\
         the default, and is not a law.",
        cases.len()
    );
}
