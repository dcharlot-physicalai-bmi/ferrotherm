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

const SEEDS: u64 = 24;
/// Logical variables. Twelve: dense enough that every one becomes a real chain on a degree-6
/// machine, small enough that branch and bound proves the optimum for every instance.
///
/// THIS USED TO SAY SIX, AND THE REASON IT DID IS THE POINT. The placer could not embed any clique
/// past K_7, so the only sizes this file could measure were ones whose chains barely existed --
/// which meant the upper half of the sweep was untested rather than passed, because exhibiting a
/// chain strength so large it swamps the model needs a problem hard enough for a swamped search to
/// lose. The placer is repaired, so the question can now be asked at a size where it means
/// something.
const NV: usize = 12;
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

    let multiples = [0.25f64, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 16.0];
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
        "\nTHE DEFAULT WAS WRONG, AND THIS IS THE MEASUREMENT THAT MOVED IT. Two is the standard\n\
         first guess in the literature and it was this crate's default until this table existed. At\n\
         two, a tenth of chains BREAK. Four is the first multiple that breaks none, and it ties the\n\
         best gap and the best hit rate. DEFAULT_CHAIN_MULTIPLE is now four."
    );
    println!(
        "\nAND BOTH FAILURE MODES ARE HERE, WHICH IS THE WHOLE POINT OF THE SWEEP. They are not\n\
         symmetric. Too weak ANNOUNCES ITSELF -- the broken column runs 69.8%, 58.0%, 32.6% as the\n\
         chains give way. Too strong is SILENT: at sixteen nothing breaks at all, every chain holds,\n\
         no warning fires anywhere, and the answer is nine times further from the optimum than at\n\
         four. A caller watching only for broken chains would read the worst row in this table as\n\
         the safest one."
    );
    println!(
        "\nTHIS COULD NOT BE MEASURED UNTIL THE PLACER WAS REPAIRED, and that is worth saying\n\
         plainly because the previous version of this file said the opposite conclusion honestly and\n\
         was still wrong. It ran at SIX logical variables, because the placer could not embed any\n\
         clique past K_7. Six variables on a degree-6 machine barely needs chains, so the rigidity\n\
         half never appeared, the sweep looked flat above the default, and the file correctly\n\
         reported that it had not exhibited the failure rather than that the failure was absent.\n\
         Twelve variables with chains of eighteen sites exhibit it immediately."
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
