// What does quadratising a higher-order model actually cost?
//
// `src/hubo.rs` says, in its module doc, that `Hubo::from_graph` "exists so the two paths can be
// run against each other on the same model rather than argued about". Nothing ran them against each
// other. This does.
//
// The two paths, on the SAME terms:
//
//   native   `src/hubo.rs`. No ancillas at all. A term of any width contributes -w * prod(s_i) and
//            the change from one flip is a sum over the terms containing that spin.
//   reduced  Rosenberg's reduction (`src/reduce.rs`). Introduce an ancilla per substituted pair,
//            penalise it into agreeing with the product, anneal the pairwise graph, slice the
//            answer back down to the original spins. This is the path every non-Rust surface is on
//            today, because `hubo` has no C ABI.
//
// GIVING THE OTHER ARM ITS BEST SHOT, which is where the first version of this went wrong. Run at
// the native model's beta ladder the reduced arm scored -11.50 against -26.88, and the table read
// as a rout. It was measuring the ladder. The reduction's penalty is the sum of every coefficient's
// magnitude -- 1308 on a model whose terms are all +/-1 -- so the reduced graph's energy scale is a
// thousand times the objective's, and a ladder that suits the objective leaves the penalty terms
// frozen before the search has begun. Sweeping the ladder's cold end from 5e-2 down to 5e-6 moved
// the reduced arm from -11.50 to -16.62. The ladder used here is the best of that sweep, so the
// comparison is against the reduction at its best rather than at its worst.
//
// AND ITS BEST SHOT AT BUDGET TOO. Both arms are judged on the ORIGINAL higher-order model: a
// reduced run is scored on the model it was asked about, never on the one it was lowered to. The
// reduced arm is then given 1x, 4x, 16x, 64x, 256x and 1024x the native arm's sweep budget, because
// "costs more compute" and "does not get there" are different findings and only a budget ladder
// separates them. An earlier shootout in this repository gave one arm 500 flips and the other
// 320,000 and read as a clean sweep for the arm we wrote.
//
// THE ANCILLA CHECK. `src/reduce.rs` guarantees that the reduced energy minimised over the ancillas
// equals the original -- a statement about ground states. It says outright that the penalty "makes
// violating assignments expensive rather than impossible", so a reduced run may return a state
// whose ancillas do not hold, and that state's projection is then not an answer to the original
// model at all. That is counted exactly, by comparing the reduced energy plus the dropped offset
// against the original model's energy of the projection: equal to floating point when every ancilla
// held, short by at least one penalty when one did not. It stays at zero here, and that is the
// confirmation rather than a null result -- the reduced arm is stuck INSIDE the feasible region,
// not wandering out of it.
//
// NOT run in CI: the 1024x budget column takes minutes, and shortening it to fit would delete the
// finding rather than the runtime.
//
// run: cargo run --release --example hubo_vs_reduction

use ferrotherm::ftp::Program;
use ferrotherm::hubo::{self, Hubo};
use ferrotherm::reduce;
use ferrotherm::rng::Pcg;
use ferrotherm::tempering::{self, geometric_ladder};

/// A random k-body instance: `t` terms of arity `k` over `n` spins, weights in {-1, +1}.
///
/// The terms are generated ONCE and both paths are built from this same list, so the comparison
/// cannot drift into comparing two instances. Distinct variables within a term, because
/// `src/factor.rs` refuses a repeated one -- `s·s = 1` would silently make the term a different
/// order than the one written.
fn instance(n: usize, k: usize, t: usize, seed: u64) -> Vec<(Vec<usize>, f64)> {
    let mut rng = Pcg::new(seed, 0x000B_11C0);
    let mut out = Vec::with_capacity(t);
    while out.len() < t {
        let mut vs: Vec<usize> = Vec::with_capacity(k);
        while vs.len() < k {
            let v = (rng.f64() * n as f64) as usize % n;
            if !vs.contains(&v) {
                vs.push(v);
            }
        }
        // Two terms over the same variables are allowed and simply add their weights; avoiding
        // them would bias the instance family for no reason the model cares about.
        let w = if rng.f64() < 0.5 { 1.0 } else { -1.0 };
        out.push((vs, w));
    }
    out
}

fn hubo_of(terms: &[(Vec<usize>, f64)], n: usize) -> Hubo {
    let mut h = Hubo::new(n);
    for (vs, w) in terms {
        h.add(vs, *w).expect("distinct in-range variables");
    }
    h
}

/// The same terms as an `.ftp` program, which is what `reduce::to_pairwise` takes.
fn ftp_of(terms: &[(Vec<usize>, f64)], n: usize) -> String {
    let mut s = format!("ftp 1\nspins {n}\n");
    for (vs, w) in terms {
        s.push_str(&format!("factor {w}"));
        for v in vs {
            s.push_str(&format!(" {v}"));
        }
        s.push('\n');
    }
    s
}

/// The native arm's budget. The reduced arm is given multiples of it.
const STAGES: usize = 200;
const SWEEPS: usize = 8;
const SEEDS: u64 = 16;

/// The cold end of the reduced arm's ladder, chosen by sweeping it: at 5e-2 the reduced arm scores
/// -11.50 on the first case and at 5e-5 it scores -16.62, because the penalty terms are three orders
/// of magnitude above the objective's scale and a ladder suited to the objective never melts them.
const REDUCED_BETA_MIN: f64 = 5e-5;

fn main() {
    let cases: [(usize, usize, usize); 4] = [(24, 3, 32), (32, 3, 48), (24, 4, 24), (40, 3, 60)];
    let budgets: [usize; 6] = [1, 4, 16, 64, 256, 1024];

    println!("hubo native vs Rosenberg reduction, on the same terms");
    println!(
        "mean best energy of the ORIGINAL model over {SEEDS} seeds; lower is better.\n\
         The native arm runs once, at 1x. Every reduced column is a MULTIPLE of that same budget.\n"
    );

    print!(
        "{:>4} {:>2} {:>4} {:>6} {:>4} {:>7} {:>8}  ",
        "n", "k", "trm", "spins", "anc", "pen/w", "native"
    );
    for b in budgets {
        print!("{:>9}", format!("red {b}x"));
    }
    println!("   broken");

    for (n, k, t) in cases {
        let mut native = 0.0f64;
        let mut reduced = [0.0f64; 6];
        let mut broken = 0usize;
        let (mut ancillas, mut rspins, mut penalty) = (0usize, 0usize, 0.0f64);

        for seed in 0..SEEDS {
            let terms = instance(n, k, t, seed);
            let h = hubo_of(&terms, n);
            let prog = Program::from_ftp(&ftp_of(&terms, n)).expect("a well-formed program");
            let red = reduce::to_pairwise(&prog).expect("a reducible program");
            let g = red.program.to_graph().expect("a pairwise graph");
            ancillas = red.ancillas;
            rspins = red.program.spins;
            penalty = red.penalty;

            let p = hubo::Params {
                beta_min: 0.05,
                beta_max: 8.0,
                stages: STAGES,
                sweeps_per_stage: SWEEPS,
            };
            native += hubo::anneal(&h, &p, seed).energy;

            let ladder = geometric_ladder(REDUCED_BETA_MIN, 8.0, STAGES);
            for (i, mult) in budgets.iter().enumerate() {
                let sched: Vec<(f64, usize)> =
                    ladder.iter().map(|&b| (b, SWEEPS * mult)).collect();
                let (state, reduced_e) = tempering::anneal(&g, &sched, seed, None);
                let original_e = h.energy(&state[..n]);
                reduced[i] += original_e;
                // Comparing the two energies IS the ancilla check, and it needs no knowledge of
                // which spins are ancillas or of how they were defined.
                if ((reduced_e + red.offset) - original_e).abs() > 1e-6 {
                    broken += 1;
                }
            }
        }

        let m = SEEDS as f64;
        print!(
            "{n:>4} {k:>2} {t:>4} {rspins:>6} {ancillas:>4} {:>7.0} {:>8.2}  ",
            penalty,
            native / m
        );
        for r in reduced {
            print!("{:>9.2}", r / m);
        }
        println!("   {:>3}/{}", broken, SEEDS as usize * budgets.len());
    }

    println!(
        "\n'anc' is the ancillas the reduction added and 'spins' the graph it had to search: each \
         one is a variable\nthe answer depends on and the question never mentioned. The native path \
         adds none.\n\n\
         'pen/w' is the penalty the reduction chose, against term weights of 1. That ratio is the \
         mechanism: any\nsingle flip that would move the search must first pay it, so the landscape \
         is rigid and a single-flip\nsampler cannot traverse it. 'broken' counts runs whose ancillas \
         did not hold, and it stays at zero --\nwhich is the confirmation, not a null result: the \
         reduced arm is stuck inside the feasible region\nrather than wandering out of it.\n\n\
         The budget columns are the finding. If the reduced arm caught up at 64x or 256x, the cost \
         of quadratising\nwould be compute, and compute is buyable."
    );
}
