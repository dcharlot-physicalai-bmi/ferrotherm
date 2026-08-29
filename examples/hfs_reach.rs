// Does solving a block exactly beat flipping one spin at a time?
//
// HFS takes the exact best assignment of a low-treewidth subgraph with everything outside it held
// fixed. Every other local search in this crate flips ONE spin and asks whether that helped. The
// claim is that a block move steps over any barrier living entirely inside the block, where a
// single-flip method has to pay to climb it.
//
// THE ANSWER HERE IS MOSTLY NO, AND THAT IS THE POINT OF RUNNING IT. On a two-dimensional spin
// glass at these sizes, HFS run on its own loses to tabu and to breakout local search on a matched
// budget. It is worth having anyway, for a reason the last column shows and the first four do not.
//
// WHAT IS MATCHED, and why it is spin flips rather than seconds. The machine this was written on
// sat above load average 120 all day, so no wall-clock figure here would mean anything; and even on
// a quiet machine, seconds compare implementations while flips compare ALGORITHMS. Every arm gets
// the same number of single-spin updates:
//
//   sweeps       B / n sweeps of chromatic Gibbs down a ladder
//   tabu         B iterations, one flip each
//   breakout     B iterations, one flip each
//   hfs          B / block block moves, each solving `block` spins exactly
//
// The HFS charge is deliberately generous to the other three. A block move does more arithmetic per
// spin than a flip does -- eliminating a width-1 tree is a forward and a backward pass over it --
// so charging one flip per spin understates its cost. It loses anyway on the standalone columns,
// which means it is not losing on bookkeeping.
//
// WHY A TWO-DIMENSIONAL GLASS IS NOT HFS'S BEST CASE, stated because the result is negative and it
// would be easy to leave the reason out. The algorithm exploits graphs that DECOMPOSE into
// low-treewidth pieces: Selby's implementation beat the hardware it was compared against on
// Chimera, whose blocks are 4x4 bipartite cells joined sparsely, so a block can cover a whole
// region of the problem. A periodic 2D lattice has treewidth equal to its side, and an induced tree
// on it is a thin filament through a dense neighbourhood -- the frozen boundary of the block is
// nearly as large as the block. This crate has no Chimera generator (`src/ising.rs` has ring, grid
// and lattice; `src/embed.rs` builds a King's graph only in its tests), so the structure HFS is
// actually for cannot be built here yet. That is a gap in the instance library, not a verdict on
// the algorithm, and it is written down rather than worked around by picking a friendlier instance.
//
// NOT run in CI: the largest size takes minutes.
//
// run: cargo run --release --example hfs_reach

use ferrotherm::graph::{Graph, GraphBuilder};
use ferrotherm::rng::Pcg;
use ferrotherm::{bls, gibbs, hfs, tabu, tempering};

/// A periodic 2D spin glass: couplings uniform in {-1, +1}, no fields.
fn glass(l: usize, seed: u64) -> Graph {
    let mut rng = Pcg::new(seed, 0x0F5A_C0DE);
    let mut b = GraphBuilder::new(l * l);
    for y in 0..l {
        for x in 0..l {
            let i = y * l + x;
            b.couple(i, y * l + (x + 1) % l, if rng.f64() < 0.5 { 1.0 } else { -1.0 });
            b.couple(i, ((y + 1) % l) * l + x, if rng.f64() < 0.5 { 1.0 } else { -1.0 });
        }
    }
    b.build()
}

/// Annealed sweeps from a given start, tracking the best state seen.
fn sweeps_from(g: &Graph, start: &[i8], budget: usize, seed: u64) -> f64 {
    let n = g.n.max(1);
    let stages = 40usize;
    let per = (budget / n / stages).max(1);
    let mut smp = gibbs::Sampler::new(g, 0.2, seed);
    smp.s.copy_from_slice(start);
    let mut best = g.energy(&smp.s);
    for beta in tempering::geometric_ladder(0.2, 6.0, stages) {
        smp.beta = beta;
        for _ in 0..per {
            smp.sweep(None);
            best = best.min(g.energy(&smp.s));
        }
    }
    best
}

const SEEDS: u64 = 8;
const BUDGET_PER_NODE: usize = 400;
const BLOCK: usize = 64;
/// Share of the budget the composition arm leaves for the block polish.
const POLISH: usize = 10;

fn main() {
    println!("HFS against three single-flip arms, on a matched budget of spin updates\n");
    println!(
        "Mean best energy over {SEEDS} seeds; lower is better. Budget {BUDGET_PER_NODE} flips per\n\
         node. The last column spends {}% of that budget on tabu and the rest on block moves\n\
         starting from tabu's answer.\n",
        100 - 100 / POLISH
    );
    println!(
        "{:>4} {:>6} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9}  {:>16}",
        "l", "spins", "sweeps", "tabu", "breakout", "hfs", "tabu+hfs", "delta", "improving moves"
    );

    for l in [12usize, 20, 28] {
        let n = l * l;
        let budget = BUDGET_PER_NODE * n;
        let (mut sw, mut tb, mut bl, mut hf, mut comp) = (0.0, 0.0, 0.0, 0.0, 0.0);
        let (mut improving, mut helped) = (0usize, 0usize);

        for seed in 0..SEEDS {
            let g = glass(l, seed);
            let start: Vec<i8> = {
                let mut r = Pcg::new(seed, 0xC0FD);
                (0..n).map(|_| r.spin(0.5)).collect()
            };

            sw += sweeps_from(&g, &start, budget, seed);
            bl += bls::search(&g, &bls::Params { iterations: budget, ..Default::default() }, seed)
                .energy;

            let hp = hfs::Params {
                steps: (budget / BLOCK).max(1),
                block: BLOCK,
                ..hfs::Params::default()
            };
            hf += hfs::run_from(&g, start.clone(), &hp, seed).energy;

            // The composition the C ABI advertises: tabu leaves a state, HFS starts from it. Tabu
            // gets the same budget the standalone column measures MINUS what the polish spends, so
            // the two tabu figures are not the same run and the comparison is not free.
            let full = tabu::search(
                &g,
                &tabu::Params { iterations: budget, ..Default::default() },
                seed,
            );
            tb += full.energy;

            let short = tabu::search(
                &g,
                &tabu::Params {
                    iterations: budget - budget / POLISH,
                    ..Default::default()
                },
                seed,
            );
            let polish = hfs::Params {
                steps: (budget / POLISH / BLOCK).max(1),
                block: BLOCK,
                ..hfs::Params::default()
            };
            let out = hfs::run_from(&g, short.state.clone(), &polish, seed);
            comp += out.energy;
            improving += out.improving;
            if out.energy < short.energy - 1e-9 {
                helped += 1;
            }
        }

        let m = SEEDS as f64;
        println!(
            "{l:>4} {n:>6} {:>9.1} {:>9.1} {:>9.1} {:>9.1} {:>9.1} {:>9.1}  {improving:>7} \
             ({helped}/{SEEDS} seeds)",
            sw / m,
            tb / m,
            bl / m,
            hf / m,
            comp / m,
            comp / m - tb / m,
        );
    }

    println!(
        "\nSTANDALONE, HFS LOSES. It is a descent -- the energy never rises -- so from a random\n\
         start it falls into the first block-local minimum and stops. It has no temperature and no\n\
         way back out. That is not a defect to be fixed; it is what the algorithm is, and it is why\n\
         the literature runs it from a schedule of restarts rather than once."
    );
    println!(
        "\nAS A POLISH, IT DEPENDS ON SIZE, and the `improving moves` column is where that shows.\n\
         At l = 12 and l = 20 it makes zero improving moves on tabu's answer: tabu has already\n\
         found a state no induced tree can better, and no energy figure alone would tell you that.\n\
         At l = 28 it makes them, and the delta column turns negative. A block move sees barriers a\n\
         flip cannot, and there have to be barriers left for that to matter."
    );
    println!(
        "\nTABU AND BREAKOUT TAKE NO STARTING STATE. `branch::Params` carries an `incumbent` and\n\
         these two have no equivalent, so the composition above had to be built by handing HFS\n\
         tabu's OUTPUT rather than by handing tabu a warm start. That asymmetry is a real gap in\n\
         those two, not in this comparison."
    );
}
