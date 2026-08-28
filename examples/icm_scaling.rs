// Do isoenergetic cluster moves earn their place, and where?
//
// The literature's claim about ICM is not that it is faster in general -- it is that the advantage
// GROWS WITH SYSTEM SIZE. That is a claim about a trend, and a trend cannot be checked at one size.
//
// The control is the identical code path with `icm_every = 0`: same ladder, same seeds, same sweep
// budget, two replica sets either way. Anything else compares two programs instead of one feature.
//
// Measured here rather than asserted, because the first version of the unit test for this ran at
// 8x8 and found 0 wins, 0 losses, 20 ties -- a 64-spin glass is solved by either arm, so the
// comparison discriminated nothing while reading as a passing test.
//
// NOT run in CI: the large sizes take minutes.
//
// run: cargo run --release --example icm_scaling

use ferrotherm::graph::{Graph, GraphBuilder};
use ferrotherm::host::Timing;
use ferrotherm::rng::Pcg;
use ferrotherm::tempering::geometric_ladder;
use ferrotherm::icm;

/// A periodic 2D spin glass: couplings uniform in {-1, +1}, no fields.
fn glass(l: usize, seed: u64) -> Graph {
    let mut rng = Pcg::new(seed, 0x001C_3A55);
    let mut gb = GraphBuilder::new(l * l);
    for y in 0..l {
        for x in 0..l {
            let i = y * l + x;
            gb.couple(i, y * l + (x + 1) % l, if rng.f64() < 0.5 { 1.0 } else { -1.0 });
            gb.couple(i, ((y + 1) % l) * l + x, if rng.f64() < 0.5 { 1.0 } else { -1.0 });
        }
    }
    gb.build()
}

fn main() {
    const SEEDS: u64 = 20;
    println!("parallel tempering, with and without isoenergetic cluster moves");
    println!("same ladder, same seeds, same sweep budget; 12 rungs from beta 0.1 to 4.0\n");
    println!("  {:>7} {:>6} {:>7} {:>5} {:>5} {:>5} {:>12} {:>14}",
             "lattice", "spins", "rounds", "win", "lose", "tie", "mean dE", "mean cluster");

    let mut dirty: Option<String> = None;
    for (l, rounds) in [(8usize, 300usize), (12, 200), (16, 200), (20, 150), (24, 150)] {
        let (mut w, mut lo, mut tie) = (0, 0, 0);
        let (mut de, mut spins, mut moves) = (0.0f64, 0u64, 0usize);
        for seed in 0..SEEDS {
            let g = glass(l, seed);
            let base = icm::Params {
                betas: geometric_ladder(0.1, 4.0, 12),
                rounds,
                sweeps_per_round: 1,
                swap_every: 1,
                icm_every: 0,
            };
            let (pair, t) = Timing::around(|| {
                let plain = icm::run(&g, &base, seed).expect("no fields");
                let with = icm::run(&g, &icm::Params { icm_every: 1, ..base.clone() }, seed)
                    .expect("no fields");
                (plain, with)
            });
            dirty = dirty.or_else(|| t.caveat());
            let (plain, with) = pair;
            de += with.energy - plain.energy;
            spins += with.icm_spins;
            moves += with.icm_moves;
            if with.energy < plain.energy - 1e-9 {
                w += 1;
            } else if with.energy > plain.energy + 1e-9 {
                lo += 1;
            } else {
                tie += 1;
            }
        }
        // The mean cluster size is the diagnostic that says the move is a CLUSTER move: a mean of
        // one would be single-spin flips under a grander name.
        let mean_cluster = if moves == 0 { 0.0 } else { spins as f64 / moves as f64 };
        println!("  {:>7} {:>6} {rounds:>7} {w:>5} {lo:>5} {tie:>5} {:>12.2} {:>14.1}",
                 format!("{l}x{l}"), l * l, de / SEEDS as f64, mean_cluster);
    }

    if let Some(c) = &dirty {
        println!("\nNOTE: {c}");
        println!("The win/loss counts and energies are unaffected -- a search outcome is the same");
        println!("number whoever else is on the CPU. No timing is reported.");
    }
    println!("\nREADING: `mean dE` below zero is the cluster move winning, and the column that");
    println!("matters is how it MOVES with size. A single size cannot distinguish 'ICM helps' from");
    println!("'this instance was hard'; a trend can. Negative and growing is the literature's");
    println!("claim, and it is the reason the unit test for this runs at 16 and not at 8.");
}
