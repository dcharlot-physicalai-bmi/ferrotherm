// EVERY shipped optimiser on the same instance, at the SAME move budget. The head-to-head this
// crate did not have -- and then had for three of nine, which is a different way of not having it.
//
// `gset_gap` reports one number per instance and brackets it with a bound. That says how good the
// answer is and nothing about which method got there, and "our sampler reached the best-known cut"
// is not a comparison -- it is a claim with the control missing. This runs parallel tempering,
// tabu, breakout local search, isoenergetic cluster moves, simulated quantum annealing, both
// simulated-bifurcation variants, population annealing and HFS block descent on one instance, gives
// each of them the same number of SPIN UPDATES, and prints what each reached.
//
// WHAT IS DELIBERATELY NOT HERE. Goemans-Williamson and branch and bound are not budgeted arms and
// putting them in this table would be a category error: GW returns a rounding with a 0.87856
// worst-case GUARANTEE and an SDP bound, and branch returns a PROOF or nothing. Neither is "a
// heuristic given some flips", and `examples/exact_bracket` is where they belong. The bound is
// printed below the table instead, because what a table of heuristics most needs is a ceiling.
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
use ferrotherm::{bls, hfs, icm, popanneal, sbm, sqa, tabu, tempering};

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

    // Isoenergetic cluster moves: two replica SETS over one ladder, so the budget divides by twice
    // the ladder length. Charging it the same total as a single-ladder method would hand it 2r
    // times the work and the table would read as a win for the cluster move.
    {
        let betas = tempering::geometric_ladder(0.05, 6.0, 8);
        let replicas = 2 * betas.len();
        let rounds = 200usize;
        let per = (budget / (rounds * replicas * n)).max(1);
        let p = icm::Params {
            betas: betas.clone(),
            rounds,
            sweeps_per_round: per,
            swap_every: 4,
            icm_every: 4,
        };
        let (cuts, t2) = Timing::around(|| {
            (0..8u64)
                .filter_map(|seed| icm::run(&inst.graph, &p, seed).ok().map(|o| inst.cut(&o.state)))
                .collect::<Vec<f64>>()
        });
        dirty = dirty.or_else(|| t2.caveat());
        if cuts.is_empty() {
            // `icm` refuses a graph with fields, and a G-set instance has none -- but an empty row
            // reads as a zero cut, which is a much worse lie than a stated refusal.
            rows.push(("isoenergetic cluster".into(), f64::NAN, f64::NAN, "refused: fields".into()));
        } else {
            rows.push((
                "isoenergetic cluster".into(),
                cuts.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
                cuts.iter().sum::<f64>() / cuts.len() as f64,
                format!("{} replicas x {rounds} x {per}", replicas),
            ));
        }
    }

    // Simulated quantum annealing: the system is M x n spins, so a sweep costs M times a classical
    // one and the budget divides by the Trotter count as well.
    {
        let trotter = 20usize;
        let steps = 200usize;
        let per = (budget / (steps * trotter * n)).max(1);
        let p = sqa::Params {
            trotter,
            beta: 4.0,
            gamma_max: 3.0,
            gamma_min: 0.05,
            steps,
            sweeps_per_step: per,
        };
        let (cuts, t2) = Timing::around(|| {
            (0..8u64).map(|seed| inst.cut(&sqa::run(&inst.graph, &p, seed).state)).collect::<Vec<f64>>()
        });
        dirty = dirty.or_else(|| t2.caveat());
        rows.push((
            "simulated quantum".into(),
            cuts.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            cuts.iter().sum::<f64>() / cuts.len() as f64,
            format!("M={trotter}, {steps} steps x {per}"),
        ));
    }

    // Simulated bifurcation, both variants. A step updates every oscillator, so a step is a sweep.
    for (variant, label) in [(sbm::Variant::Ballistic, "sim. bifurcation bSB"),
                             (sbm::Variant::Discrete, "sim. bifurcation dSB")] {
        let steps = (budget / n).max(1);
        let (cuts, t2) = Timing::around(|| {
            (0..8u64)
                .map(|seed| inst.cut(&sbm::run(&inst.graph, variant, steps, 0.5, seed).0))
                .collect::<Vec<f64>>()
        });
        dirty = dirty.or_else(|| t2.caveat());
        rows.push((
            label.into(),
            cuts.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            cuts.iter().sum::<f64>() / cuts.len() as f64,
            format!("{steps} steps"),
        ));
    }

    // Population annealing: R replicas swept at every rung, so the budget divides by both.
    {
        let stages = 40usize;
        let population = 32usize;
        let per = (budget / (stages * population * n)).max(1);
        let p = popanneal::Params::linear_from_zero(population, per, 6.0, stages);
        let (cuts, t2) = Timing::around(|| {
            (0..8u64).map(|seed| inst.cut(&popanneal::run(&inst.graph, &p, seed).state)).collect::<Vec<f64>>()
        });
        dirty = dirty.or_else(|| t2.caveat());
        rows.push((
            "population annealing".into(),
            cuts.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            cuts.iter().sum::<f64>() / cuts.len() as f64,
            format!("R={population}, {stages} rungs x {per}"),
        ));
    }

    // HFS block descent, charged one flip per spin in every block -- which understates a block
    // move's arithmetic and is therefore generous to it here, the opposite of the direction an
    // author's own algorithm usually gets flattered in.
    {
        let block = 64usize.min(n);
        let steps = (budget / block).max(1);
        let p = hfs::Params { steps, block, ..hfs::Params::default() };
        let (out, t2) = Timing::around(|| {
            (0..8u64)
                .map(|seed| {
                    let o = hfs::run(&inst.graph, &p, seed);
                    (inst.cut(&o.state), o.improving)
                })
                .collect::<Vec<(f64, usize)>>()
        });
        dirty = dirty.or_else(|| t2.caveat());
        let cuts: Vec<f64> = out.iter().map(|o| o.0).collect();
        let improving: usize = out.iter().map(|o| o.1).sum::<usize>() / out.len();
        rows.push((
            "HFS block descent".into(),
            cuts.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            cuts.iter().sum::<f64>() / cuts.len() as f64,
            format!("{steps} blocks of {block}, {improving} improving"),
        ));
    }

    // And the composition, because the warm start that makes it possible is new and because a
    // stack's best answer is usually not one method. Tabu with a tenth of its budget handed to a
    // block polish.
    {
        let short = iters - iters / 10;
        let tp = tabu::Params { iterations: short, tenure: 0, restart_after: Some(short / 10 + 1), start: None };
        let block = 64usize.min(n);
        let (cuts, t2) = Timing::around(|| {
            (0..8u64)
                .map(|seed| {
                    let warm = tabu::search(&inst.graph, &tp, seed);
                    let hp = hfs::Params {
                        steps: ((iters / 10) / block).max(1),
                        block,
                        ..hfs::Params::default()
                    };
                    inst.cut(&hfs::run_from(&inst.graph, warm.state, &hp, seed).state)
                })
                .collect::<Vec<f64>>()
        });
        dirty = dirty.or_else(|| t2.caveat());
        rows.push((
            "tabu then HFS polish".into(),
            cuts.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            cuts.iter().sum::<f64>() / cuts.len() as f64,
            "90% tabu, 10% blocks".into(),
        ));
    }
    }

    let winner = rows.iter().cloned().fold(("".to_string(), f64::NEG_INFINITY, 0.0, String::new()), |a, b| if b.1 > a.1 { b } else { a });
    for (solver, best, mean, note) in &rows {
        let mark = if *best >= winner.1 - 1e-9 { " *" } else { "  " };
        println!("  {solver:>22} {best:>10.0} {mean:>10.1} {note:>12}{mark}");
    }

    // A best-known of zero -- or a negative one -- is not a scale to measure against, and dividing
    // by it printed "reached inf% of it" for every solver. An instance whose optimum nobody has
    // published is the normal case for a graph you generated yourself, and the honest output there
    // is the table on its own.
    if let Some(bk) = best_known.filter(|b| *b > 0.0) {
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
    if best_known.is_some_and(|b| b <= 0.0) {
        println!(
            "\n  no best-known cut given ({}), so the percentages are omitted rather than\n  \
             divided by zero. The table above stands on its own: it is a comparison BETWEEN\n  \
             solvers, and it needs no external number to be read.",
            best_known.unwrap_or(0.0)
        );
    }
    if let Some(c) = &dirty {
        println!("\nNOTE: {c}");
        println!("The CUTS above are unaffected -- a search outcome is the same number whoever else \
                  is on the CPU. Only a timing would have been spoiled, and none is reported.");
    }
}
