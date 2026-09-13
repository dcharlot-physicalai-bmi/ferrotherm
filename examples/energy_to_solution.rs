#![allow(missing_docs)]
// DOES CHARGING FOR READOUT CHANGE WHICH SOLVER IS CHEAPEST? Energy-to-solution at R99, with the
// reads priced at what the device model says a read costs.
//
// Every published Ising-machine ranking prices the UPDATES: flips per second, nanojoules per
// edge-bit, solutions per second per watt. DSIM-2's tables exclude readout explicitly. On a
// Z1-class device model (`ledger::Z1_SPICE`) one node read costs 1.692 pJ against 7.09 fJ per
// Gibbs update -- 239 updates per read -- so a solver that reads its state often is paying for
// something the ranking cannot see. The question with an answer:
//
//   at R99 on planted instances, does the cheapest of tabu, breakout local search, simulated
//   quantum annealing and parallel tempering CHANGE when reads are charged, under each of three
//   read policies a real run would follow?
//
//   read-free      what the tables assume: updates only
//   read at end    each run reads its n nodes once, when it stops
//   read per sweep each run reads its state every sweep, as a monitor or an anytime solver does
//
// The third policy cannot reorder anything, and the example says so rather than counting: a read
// per sweep is n reads per n proposals, so every arm's energy is its work times the same
// (e_sample + e_read), a uniform factor of about 240 on this device model. It is printed for
// scale. The second policy CAN reorder: an arm that needs many short runs to reach R99 pays n
// reads per run, and one that needs few long runs pays almost nothing for readout.
//
// TTS is `tts::tts` minimised over run length from a Wilson-bounded success ladder (Ronnow et
// al. 2014), in single-spin proposals -- the one unit every arm here shares -- and the joules are
// that work priced per operation, so nothing here is a wall clock and the run is valid on a busy
// machine. Where the KV260's MEASURED price is used, only the read-free column can be priced,
// because that measurement exercised no reads; the column says so rather than inventing a price.
//
// run: cargo run --release --example energy_to_solution

use ferrotherm::ledger::{KV260_MEASURED, Z1_SPICE};
use ferrotherm::planted::{frustrated_loops, Planted};
use ferrotherm::portfolio::{Bls, Budget, Search, Sqa, Tabu, Tempering};
use ferrotherm::tts::{tts, Trial};

/// Success ladder for one arm on one instance: at each run length, how many of `seeds` runs
/// reached the planted optimum.
fn ladder(arm: &dyn Search, p: &Planted, lengths: &[u64], seeds: u64) -> Vec<Trial> {
    lengths
        .iter()
        .map(|&len| {
            let successes = (0..seeds)
                .filter(|&s| {
                    let f = arm.solve(&p.graph, Budget::new(len), 7_000 + s);
                    p.solved(&f.state)
                })
                .count();
            Trial { length: len, successes, trials: seeds as usize }
        })
        .collect()
}

struct Priced {
    arm: &'static str,
    work: f64,
    runs: f64,
    read_free_z1: f64,
    read_end_z1: f64,
    read_sweep_z1: f64,
    read_free_kv: f64,
}

fn main() {
    let (l, loops) = (8usize, 96usize);
    let instances = 6u64;
    let seeds = 16u64;
    let arms: Vec<Box<dyn Search>> = vec![Box::new(Tabu), Box::new(Bls), Box::new(Sqa), Box::new(Tempering)];
    println!("ENERGY TO SOLUTION AT R99, WITH AND WITHOUT THE READS\n");
    println!("  instances  {instances} planted {l}x{l} frustrated-loop glasses ({loops} loops), known optimum, {seeds} seeds per run length");
    println!("  prices     Z1_SPICE: {:.3e} J/update, {:.3e} J/read ({:.0} updates per read); KV260_MEASURED: {:.3e} J/update, reads UNSTATED",
             Z1_SPICE.e_sample, Z1_SPICE.e_read, Z1_SPICE.e_read / Z1_SPICE.e_sample, KV260_MEASURED.e_sample);
    println!("  TTS        tts::tts at 99%, minimised over run length in proposals; energy = R99 runs x (work + reads) per run\n");
    let mut rank_changes_end = 0usize;
    let mut priced_any = 0usize;
    // The largest factor by which charging the end-of-run read multiplied any arm's energy, and
    // the arm it hit hardest: the margins move even where the winner does not.
    let mut worst_factor = 1.0f64;
    let mut worst_arm = "-";
    for seed in 0..instances {
        let p = frustrated_loops(l, loops, 100 + seed);
        let n = p.graph.n as u64;
        let lengths: Vec<u64> = (3..=11).map(|k| n << k).collect();
        let mut rows: Vec<Priced> = Vec::new();
        for arm in &arms {
            let trials = ladder(arm.as_ref(), &p, &lengths, seeds);
            let Some(t) = tts(&trials, 0.99) else { continue };
            let Some(work) = t.tts else { continue };
            let runs = work / t.best_length as f64;
            // Reads per policy, per run: none; n once; n per sweep of n proposals (= 1 per proposal).
            let reads_end = runs * n as f64;
            let reads_sweep = work;
            rows.push(Priced {
                arm: arm.name(),
                work,
                runs,
                read_free_z1: work * Z1_SPICE.e_sample,
                read_end_z1: work * Z1_SPICE.e_sample + reads_end * Z1_SPICE.e_read,
                read_sweep_z1: work * Z1_SPICE.e_sample + reads_sweep * Z1_SPICE.e_read,
                read_free_kv: work * KV260_MEASURED.e_sample,
            });
        }
        if rows.is_empty() {
            println!("  instance {seed}: no arm reached the optimum on any run length; nothing to rank");
            continue;
        }
        priced_any += 1;
        let best = |key: &dyn Fn(&Priced) -> f64| -> &'static str {
            rows.iter().min_by(|a, b| key(a).partial_cmp(&key(b)).unwrap()).map_or("-", |r| r.arm)
        };
        let (b_free, b_end, b_sweep) = (best(&|r| r.read_free_z1), best(&|r| r.read_end_z1), best(&|r| r.read_sweep_z1));
        if b_end != b_free {
            rank_changes_end += 1;
        }
        for r in &rows {
            let factor = r.read_end_z1 / r.read_free_z1;
            if factor > worst_factor {
                worst_factor = factor;
                worst_arm = r.arm;
            }
        }
        println!("  instance {seed} ({n} spins): cheapest read-free {b_free}, read-at-end {b_end}, read-per-sweep {b_sweep}");
        println!(
            "    {:<10} {:>12} {:>7}   {:>11} {:>11} {:>11}   {:>11}",
            "arm", "work (prop.)", "R99", "Z1 free", "Z1 end", "Z1 sweep", "KV260 free"
        );
        for r in &rows {
            println!(
                "    {:<10} {:>12.3e} {:>7.1}   {:>11.3e} {:>11.3e} {:>11.3e}   {:>11.3e}",
                r.arm, r.work, r.runs, r.read_free_z1, r.read_end_z1, r.read_sweep_z1, r.read_free_kv
            );
        }
        let arms_unsolved: Vec<&str> = arms.iter().map(|a| a.name()).filter(|nm| !rows.iter().any(|r| r.arm == *nm)).collect();
        if !arms_unsolved.is_empty() {
            println!("    unsolved at every run length up to {}: {}", lengths.last().unwrap(), arms_unsolved.join(", "));
        }
    }
    println!("\n  WHAT THE TABLE SAYS.\n");
    println!("  Over {priced_any} instances with a ranked winner: charging one read of every node at the end of each");
    println!("  run changed the cheapest arm on {rank_changes_end}, and multiplied an arm's energy by as much as");
    println!("  {worst_factor:.1}x ({worst_arm}) -- the margins move on every instance whether the winner does or not.");
    println!("  A read every sweep is a uniform factor on every arm and cannot reorder them; it is printed for scale.");
    println!("  The read-free column is what published rankings compute. The end-of-run column is what a device");
    println!("  that charges 239 updates per read would bill for the same runs -- an arm that needs many short");
    println!("  runs to reach R99 pays n reads per run, and the tables that exclude readout cannot see it.");
}
