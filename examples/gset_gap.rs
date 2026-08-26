//! G-set with a gap: what a max-cut result looks like when it is a claim about the instance.
//!
//! The G-set league table is twenty-five years of **best cut found**. Every entry is a lower bound
//! on the max cut — somebody achieved that cut — so the table ranks how hard people have looked. It
//! cannot say whether the best entry is optimal, or how far from optimal any entry is.
//!
//! This reports both sides. The sampler produces a cut (a lower bound); [`bound`] produces a lower
//! bound on the energy, which is an **upper** bound on the cut; and the two bracket the truth:
//!
//! ```text
//!   cut found   <=   max cut   <=   (W - L) / 2
//! ```
//!
//! ```sh
//! cargo run --release --example gset_gap -- <file> [best-known-cut]
//! ```
//!
//! PREDICTED before running: the bound would be **loose on dense instances and tight on sparse
//! ones**, because a relaxation that lets parts disagree suffers most where they share the most.
//!
//! Half right, and the wrong half is the interesting one. The density trend held exactly — 2.6% at
//! mean degree 4, 15.1% at 11.7, 22.3% at 47.9. But the prediction named the FOREST bound, and the
//! forest bound turned out to contribute nothing at all here: a tree is never frustrated, G-set has
//! no fields, so it degenerates to the trivial `-Σ|w|` on every instance. The trend appears in
//! [`bound::odd_cycle`], which did not exist when the prediction was written. Right about the
//! phenomenon, wrong about which mechanism would show it.
//!
//! Measured, 800-node instances, 8 restarts:
//!
//! ```text
//!   instance  degree   cut found   best known          UB     gap
//!   G11          4.0         564          564  (100.00%)     579    2.6%
//!   G14         11.7        3058         3064   (99.80%)    3602   15.1%
//!   G1          47.9       11624        11624  (100.00%)   14958   22.3%
//! ```

use ferrotherm::{bound, gset::Instance, schedule::Schedule, tempering};
use std::time::Instant;

fn main() {
    let mut args = std::env::args().skip(1);
    let Some(path) = args.next() else {
        eprintln!("usage: gset_gap <file> [best-known-cut]");
        std::process::exit(2);
    };
    let best_known: Option<f64> = args.next().and_then(|v| v.parse().ok());

    let text = match std::fs::read_to_string(&path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("cannot read {path}: {e}");
            std::process::exit(2);
        }
    };
    let inst = match Instance::parse(&text) {
        Ok(i) => i,
        Err(e) => {
            eprintln!("{path}: {e}");
            std::process::exit(2);
        }
    };
    let name = path.rsplit('/').next().unwrap_or(&path);
    let degree = 2.0 * inst.edges as f64 / inst.nodes as f64;
    println!("{name}: {} nodes, {} edges, mean degree {degree:.1}, W = {}",
             inst.nodes, inst.edges, inst.total_weight);

    // ---- the cut, from the sampler --------------------------------------------------------------
    let t0 = Instant::now();
    let ladder = Schedule::geometric(0.05, 6.0, 200, 120);
    let mut best = f64::NEG_INFINITY;
    let mut best_state = Vec::new();
    for seed in 0..8u64 {
        let (s, _) = tempering::anneal_scheduled(&inst.graph, &ladder, seed, None);
        let c = inst.cut(&s);
        if c > best {
            best = c;
            best_state = s;
        }
    }
    let t_sample = t0.elapsed().as_secs_f64();
    // The cut has to be recomputed from the STATE, not carried from the loop: a cut reported by the
    // thing that searched for it is the one number in this file nothing else would catch.
    let verified = inst.cut(&best_state);
    assert!((verified - best).abs() < 1e-9, "reported {best}, state gives {verified}");
    println!("  cut found      {best:>12.0}   ({t_sample:.1} s, 8 restarts x 200-stage ladder)");

    // ---- the upper bound ------------------------------------------------------------------------
    //
    // BOTH, and the better of the two. Each is sound on its own, so their maximum is sound -- and
    // on max-cut they are not close: `forest` degenerates to the trivial floor here because a tree
    // is never frustrated and G-set carries no fields for the subgradient to move, while
    // `odd_cycle` charges for exactly the frustration that makes the problem hard.
    let t1 = Instant::now();
    let f = bound::forest(&inst.graph, 40);
    let c = bound::odd_cycle(&inst.graph, 6);
    let t_bound = t1.elapsed().as_secs_f64();
    let (best_bound, which) =
        if c.value >= f.value { (&c, "odd-cycle") } else { (&f, "forest") };
    let ub = inst.cut_upper_bound(best_bound.value);
    println!("  forest ub      {:>12.0}   ({} forests, peak at round {} of {})",
             inst.cut_upper_bound(f.value), f.parts, f.best_round, f.rounds);
    println!("  odd-cycle ub   {:>12.0}   ({} edge-disjoint frustrated cycles)",
             inst.cut_upper_bound(c.value), c.parts);
    println!("  upper bound    {ub:>12.0}   ({which}, {t_bound:.1} s)");

    // ---- the gap --------------------------------------------------------------------------------
    //
    // Relative to the UPPER bound, because that is the quantity that is certain: the true max cut
    // is somewhere in the bracket, and dividing by a number we are sure of keeps the percentage
    // honest when the bound is loose.
    let gap = (ub - best) / ub * 100.0;
    println!("  gap            {gap:>11.1}%   of the upper bound");

    if let Some(bk) = best_known {
        let pct = best / bk * 100.0;
        println!("  best known     {bk:>12.0}   -- this run reached {pct:.2}% of it");
        // The published figure is itself only a lower bound, so it must sit inside our bracket. If
        // it does not, our bound is wrong -- and that is worth failing over rather than printing.
        if bk > ub + 1e-6 {
            eprintln!(
                "\n  ** the published best-known cut {bk} EXCEEDS this upper bound {ub:.0}. A cut \
                 that has actually been achieved cannot be above a valid upper bound, so the bound \
                 is unsound. **"
            );
            std::process::exit(1);
        }
    }
}
