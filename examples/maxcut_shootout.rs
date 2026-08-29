// Three solvers on the same instance, at the SAME move budget. The head-to-head this crate did not
// have.
//
// `gset_gap` reports one number per instance and brackets it with a bound. That says how good the
// answer is and nothing about which method got there, and "our sampler reached the best-known cut"
// is not a comparison -- it is a claim with the control missing. This runs parallel tempering, tabu
// search and breakout local search on one instance, gives each of them the same number of SPIN
// UPDATES, and prints what each reached.
//
// MATCHED BUDGET, NOT MATCHED TIME. A wall-clock comparison needs a quiet machine and this project
// has an instrument that refuses to pretend otherwise; a flip budget is the same number on any
// machine, and it isolates the algorithm from the implementation. Both are worth having and only
// one of them can be taken here honestly.
//
// THE UNIT IS ONE SPIN FLIP, and getting that wrong is easy enough that the first version of this
// file did. It gave tempering `budget` flips and gave tabu and BLS `budget / n` -- because their
// LEDGER charge is n move evaluations per flip, and a budget expressed in ledger samples is not a
// budget expressed in flips. At n = 800 that handed the deterministic searches 500 flips against
// tempering's 320,000, and the table read as a clean win for tempering on every instance.
//
// The remaining asymmetry is stated rather than fixed: tempering pays `O(degree)` to make a flip,
// tabu and BLS pay `O(n)` to CHOOSE one, because neither has the bucket structure that would make
// the choice constant-time. So a matched-flip table flatters the deterministic searches on quality
// and says nothing about what their flips cost. There is no single budget that is fair to both, and
// pretending otherwise is worse than naming the asymmetry.
//
//   cargo run --release --example maxcut_shootout -- <file> [best-known-cut] [budget]
//
// NOT run in CI: it needs a G-set file the runner does not have.

use ferrotherm::gset::Instance;
use ferrotherm::host::Timing;
use ferrotherm::schedule::Schedule;
use ferrotherm::{bls, tabu, tempering};

fn main() {
    let mut args = std::env::args().skip(1);
    let Some(path) = args.next() else {
        eprintln!("usage: maxcut_shootout <file> [best-known-cut] [budget]");
        std::process::exit(2);
    };
    let best_known: Option<f64> = args.next().and_then(|v| v.parse().ok());
    let budget: usize = args.next().and_then(|v| v.parse().ok()).unwrap_or(400_000);

    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        eprintln!("cannot read {path}: {e}");
        std::process::exit(2);
    });
    let inst = Instance::parse(&text).unwrap_or_else(|e| {
        eprintln!("{path}: {e}");
        std::process::exit(2);
    });
    let name = path.rsplit('/').next().unwrap_or(&path);
    let n = inst.nodes;
    println!("{name}: {n} nodes, {} edges, mean degree {:.1}", inst.edges, 2.0 * inst.edges as f64 / n as f64);
    println!("budget {budget} SPIN FLIPS per solver, 8 seeds each\n");
    println!("  {:>22} {:>10} {:>10} {:>12}", "solver", "best cut", "mean cut", "note");

    let mut rows: Vec<(String, f64, f64, String)> = Vec::new();
    let mut dirty: Option<String> = None;

    // Parallel tempering. Its budget is stages x sweeps x n updates, so the ladder is sized to land
    // on the same total rather than to a round number of stages.
    {
        let stages = 200usize;
        let per = (budget / (stages * n)).max(1);
        let ladder = Schedule::geometric(0.05, 6.0, stages, per);
        let (cuts, t) = Timing::around(|| {
            (0..8u64)
                .map(|seed| {
                    let (s, _) = tempering::anneal_scheduled(&inst.graph, &ladder, seed, None);
                    inst.cut(&s)
                })
                .collect::<Vec<f64>>()
        });
        dirty = dirty.or_else(|| t.caveat());
        rows.push((
            "parallel tempering".into(),
            cuts.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            cuts.iter().sum::<f64>() / cuts.len() as f64,
            format!("{stages} stages x {per} sweeps"),
        ));
    }

    // One tabu or BLS iteration IS one flip, so the iteration count is the budget. It is not the
    // budget divided by n: that is the ledger's move-evaluation count, which is a different number
    // measuring a different thing. See the note at the top.
    let iters = budget.max(1);
    {
        let p = tabu::Params { iterations: iters, tenure: 0, restart_after: Some(iters / 10 + 1) , start: None };
        let (cuts, t) = Timing::around(|| {
            (0..8u64).map(|seed| inst.cut(&tabu::search(&inst.graph, &p, seed).state)).collect::<Vec<f64>>()
        });
        dirty = dirty.or_else(|| t.caveat());
        rows.push((
            "tabu search".into(),
            cuts.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            cuts.iter().sum::<f64>() / cuts.len() as f64,
            format!("{iters} iterations"),
        ));
    }
    {
        let p = bls::Params { iterations: iters, ..bls::Params::default() };
        let (out, t) = Timing::around(|| {
            (0..8u64)
                .map(|seed| {
                    let r = bls::search(&inst.graph, &p, seed);
                    (inst.cut(&r.state), r.descents, r.max_jump)
                })
                .collect::<Vec<(f64, usize, usize)>>()
        });
        dirty = dirty.or_else(|| t.caveat());
        let cuts: Vec<f64> = out.iter().map(|o| o.0).collect();
        let descents: usize = out.iter().map(|o| o.1).sum::<usize>() / out.len();
        let jump = out.iter().map(|o| o.2).max().unwrap_or(0);
        rows.push((
            "breakout local search".into(),
            cuts.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            cuts.iter().sum::<f64>() / cuts.len() as f64,
            format!("{descents} descents, L<={jump}"),
        ));
    }

    let winner = rows.iter().cloned().fold(("".to_string(), f64::NEG_INFINITY, 0.0, String::new()), |a, b| if b.1 > a.1 { b } else { a });
    for (solver, best, mean, note) in &rows {
        let mark = if *best >= winner.1 - 1e-9 { " *" } else { "  " };
        println!("  {solver:>22} {best:>10.0} {mean:>10.1} {note:>12}{mark}");
    }

    if let Some(bk) = best_known {
        println!("\n  best known             {bk:>10.0}");
        for (solver, best, _, _) in &rows {
            println!("  {solver:>22} reached {:.3}% of it", best / bk * 100.0);
        }
        // The published figure is a LOWER bound -- somebody achieved it -- so beating it is
        // possible and worth saying loudly rather than quietly printing over 100%.
        if winner.1 > bk + 1e-9 {
            println!("\n  ** {} EXCEEDS the best-known cut by {:.0}. Check the instance file before \
                      believing it. **", winner.0, winner.1 - bk);
        }
    }
    if let Some(c) = &dirty {
        println!("\nNOTE: {c}");
        println!("The CUTS above are unaffected -- a search outcome is the same number whoever else \
                  is on the CPU. Only a timing would have been spoiled, and none is reported.");
    }
}
