// Does solving a block exactly beat flipping one spin at a time?
//
// HFS takes the exact best assignment of a low-treewidth subgraph with everything outside it held
// fixed. Every other local search in this crate flips ONE spin and asks whether that helped. The
// claim is that a block move steps over any barrier living entirely inside the block, where a
// single-flip method has to pay to climb it.
//
// THE ANSWER IS NO, ON BOTH FAMILIES, AND THAT IS THE POINT OF RUNNING IT. HFS on its own loses to
// tabu at every size -- including on Chimera, which is the structure it exploits and where the
// literature's result was obtained. Swept before that was believed: block size from 8 to n moves
// the answer by under 1%, and restart counts from 1 to 256 never close the gap. The hubo comparison
// in this repository once published a wrong negative because its beta ladder had not been swept.
//
// What this module has is the block MOVE plus random block selection. Selby's algorithm is that
// plus a specific schedule over subgraphs, at C_16,16,4 and budgets far past these. The gap between
// "has the move" and "is the algorithm" is where the difference plausibly lives.
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
// TWO INSTANCE FAMILIES, because the first one is not the structure HFS is for. A periodic 2D
// lattice has treewidth equal to its side and does not decompose; an induced tree on it is a thin
// filament through a dense neighbourhood, so the block's frozen boundary is nearly as large as the
// block. CHIMERA is what Selby's implementation beat the hardware on: an m x n grid of K_{t,t}
// cells joined only between matching shores, so each shore induces a forest and half the graph is
// exactly solvable in one move. Running only the lattice would have measured HFS away from home
// and reported the number as though it were the algorithm.
//
// NOT run in CI: the largest size takes minutes.
//
// run: cargo run --release --example hfs_reach

use ferrotherm::graph::{Graph, GraphBuilder};
use ferrotherm::rng::Pcg;
use ferrotherm::{bls, gibbs, hfs, ising, tabu, tempering};

/// A periodic 2D spin glass: couplings uniform in {-1, +1}, no fields. Treewidth = the side, so it
/// does not decompose and a block is always mostly boundary.
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

/// The instance families, each a label and a builder. Both are +/-1 glasses on the same number of
/// spins where possible, so the two rows of a size are comparable to each other and not only down
/// their own column.
fn instance(family: usize, size: usize, seed: u64) -> (String, Graph) {
    if family == 0 {
        let l = [12usize, 20, 28][size];
        (format!("lattice {l}x{l}"), glass(l, seed))
    } else {
        let (m, t) = [(4usize, 4usize), (6, 4), (8, 4)][size];
        (format!("chimera C_{m},{m},{t}"), ising::chimera_glass(m, m, t, seed))
    }
}

fn main() {
    println!("HFS against three single-flip arms, on a matched budget of spin updates\n");
    println!(
        "Mean best energy over {SEEDS} seeds; lower is better. Budget {BUDGET_PER_NODE} flips per\n\
         node. `tabu+hfs` spends {}% of that on tabu and the rest on block moves from tabu's answer.\n",
        100 - 100 / POLISH
    );
    println!(
        "{:>16} {:>6} {:>9} {:>9} {:>9} {:>9} {:>9} {:>8}  {:>7}",
        "instance", "spins", "sweeps", "tabu", "breakout", "hfs", "tabu+hfs", "delta", "improving"
    );

    for family in 0..2 {
        for size in 0..3 {
            let (label, probe) = instance(family, size, 0);
            let n = probe.n;
            let budget = BUDGET_PER_NODE * n;
            let (mut sw, mut tb, mut bl, mut hf, mut comp) = (0.0, 0.0, 0.0, 0.0, 0.0);
            let (mut improving, mut helped) = (0usize, 0usize);

            for seed in 0..SEEDS {
                let (_, g) = instance(family, size, seed);
                let start: Vec<i8> = {
                    let mut r = Pcg::new(seed, 0xC0FD);
                    (0..n).map(|_| r.spin(0.5)).collect()
                };

                sw += sweeps_from(&g, &start, budget, seed);
                bl += bls::search(
                    &g,
                    &bls::Params { iterations: budget, ..Default::default() },
                    seed,
                )
                .energy;

                let hp = hfs::Params {
                    steps: (budget / BLOCK).max(1),
                    block: BLOCK,
                    ..hfs::Params::default()
                };
                hf += hfs::run_from(&g, start.clone(), &hp, seed).energy;

                // The composition the C ABI advertises: tabu leaves a state, HFS starts from it.
                // Tabu gets the standalone column's budget MINUS what the polish spends, so the two
                // tabu figures are not the same run and the comparison is not free.
                tb += tabu::search(
                    &g,
                    &tabu::Params { iterations: budget, ..Default::default() },
                    seed,
                )
                .energy;

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
                "{label:>16} {n:>6} {:>9.1} {:>9.1} {:>9.1} {:>9.1} {:>9.1} {:>8.1}  {improving:>4} \
                 ({helped}/{SEEDS})",
                sw / m,
                tb / m,
                bl / m,
                hf / m,
                comp / m,
                comp / m - tb / m,
            );
        }
        if family == 0 {
            println!();
        }
    }

    println!(
        "\nSTANDALONE, HFS LOSES ON THE LATTICE. It is a descent -- the energy never rises -- so\n\
         from a random start it falls into the first block-local minimum and stops. It has no\n\
         temperature and no way back out. That is what the algorithm is, and it is why the\n\
         literature runs it from a schedule of restarts rather than once."
    );
    println!(
        "\nCHIMERA IS THE STRUCTURE IT IS FOR, AND IT DOES NOT RESCUE IT. Chimera's shores each\n\
         induce a forest, so half the graph is exactly solvable in one move, where a periodic\n\
         lattice of treewidth l has no such split. HFS is still about 4% behind tabu on every\n\
         Chimera row -- and the polish gains 1.2 there against 21.5 on the 28x28 lattice, which is\n\
         the opposite of the ordering the structural argument predicts."
    );
    println!(
        "\nTHE `improving` COLUMN IS THE DIAGNOSTIC, not the energy. A polish that makes zero\n\
         improving moves has been handed a state no block of its shape can better, and no energy\n\
         figure says that. Zero with a good energy means tabu already won; zero with a bad one\n\
         means the blocks are wrong for the graph."
    );
    println!(
        "\nTABU AND BREAKOUT NOW TAKE A STARTING STATE, so the composition above could equally be\n\
         built the other way round -- anneal, block-descend, then tabu from there. It is written\n\
         as tabu-then-polish because that is the order the C ABI's `ft_hfs` doc recommends."
    );
}
