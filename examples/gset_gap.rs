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
//! Then [`sdp`] arrived and the trend inverted. The relaxation that lets parts disagree suffers
//! most where they share the most, so the CYCLE bound degrades with density — but a semidefinite
//! relaxation does not decompose the graph at all, and it wins by more the denser the instance is.
//! At degree 47.9 it beats the cycle bound by 2,875; at degree 4 it loses to it by 50.
//!
//! Measured, 800-node instances, 8 restarts, three bounds and the better of them:
//!
//! ```text
//!   instance  degree   cut found   best known     forest   odd-cycle    sdp      UB    gap
//!   G11          4.0         564          564        817        *579    629    *579   2.6%
//!   G14         11.7        3058         3064       4694        3602  *3192   *3192   4.2%
//!   G1          47.9       11624        11624      19176       14958 *12083  *12083   3.8%
//! ```
//!
//! The starred column is the one that set the bound. G11's optimum is provably in **[564, 579]**.

use ferrotherm::host::Timing;
use ferrotherm::rng::Pcg;
use ferrotherm::{bound, gset::Instance, schedule::Schedule, sdp, tempering};

/// A G-set-shaped instance, in G-set's own text format, for the no-argument case.
///
/// CI runs every unattended example and requires exit 0. This example took a file path and exited
/// 2 without one, so from the moment it landed the example gate failed on every push -- and the
/// release it landed in was reported as green. Adding it to the skip list would have silenced the
/// symptom and left the example unexercised, which is the same outcome with better manners.
///
/// So it generates one instead: an 8x8 torus with +-1 couplings, the shape of G11..G13, emitted as
/// G-set text and parsed back through [`Instance::parse`]. That exercises the parser, both bounds,
/// the SDP certificate and its independent re-verification on every push, and it ships no 200 KB
/// data file inside a source crate.
fn builtin() -> String {
    let (w, h) = (8usize, 8);
    let mut rng = Pcg::new(0x6_5E7, 0xC0DE);
    let mut edges = Vec::new();
    for y in 0..h {
        for x in 0..w {
            let a = y * w + x + 1; // G-set vertices are 1-based
            let right = y * w + (x + 1) % w + 1;
            let down = ((y + 1) % h) * w + x + 1;
            for b in [right, down] {
                if a != b {
                    edges.push((a, b, if rng.f64() < 0.5 { -1i32 } else { 1 }));
                }
            }
        }
    }
    let mut s = format!("{} {}\n", w * h, edges.len());
    for (a, b, j) in edges {
        s.push_str(&format!("{a} {b} {j}\n"));
    }
    s
}

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next();
    let best_known: Option<f64> = args.next().and_then(|v| v.parse().ok());

    let (path, text) = match &path {
        Some(p) => match std::fs::read_to_string(p) {
            Ok(t) => (p.clone(), t),
            Err(e) => {
                eprintln!("cannot read {p}: {e}");
                std::process::exit(2);
            }
        },
        None => {
            println!(
                "no file given, so this is the BUILT-IN 8x8 torus. It has no published best-known \n\
                 cut, and the numbers below are not comparable to anything. For the real thing:\n\
                 \n    cargo run --release --example gset_gap -- <G-set file> [best-known-cut]\n\
                 \nG-set lives at https://web.stanford.edu/~yyye/yyye/Gset/\n"
            );
            ("builtin-8x8-torus".to_string(), builtin())
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
    //
    // Timed through `Timing`, which samples the load average on both sides and refuses to call the
    // result a measurement if the machine was busy. This example is why that type exists: it once
    // reported 85.7 s for a G1 search that takes about 14 s on a quiet machine, in the same format
    // as every honest timing beside it. The CUTS were unaffected -- a search outcome is the same
    // number whoever else is using the CPU -- so the two are now printed differently on purpose.
    let ladder = Schedule::geometric(0.05, 6.0, 200, 120);
    let mut best = f64::NEG_INFINITY;
    let mut best_state = Vec::new();
    let (_, t_sample) = Timing::around(|| {
        for seed in 0..8u64 {
            let (s, _) = tempering::anneal_scheduled(&inst.graph, &ladder, seed, None);
            let c = inst.cut(&s);
            if c > best {
                best = c;
                best_state = s;
            }
        }
    });
    // The cut has to be recomputed from the STATE, not carried from the loop: a cut reported by the
    // thing that searched for it is the one number in this file nothing else would catch.
    let verified = inst.cut(&best_state);
    assert!((verified - best).abs() < 1e-9, "reported {best}, state gives {verified}");
    println!("  cut found      {best:>12.0}   (8 restarts x 200-stage ladder, {t_sample})");

    // ---- the upper bound ------------------------------------------------------------------------
    //
    // BOTH, and the better of the two. Each is sound on its own, so their maximum is sound -- and
    // on max-cut they are not close: `forest` degenerates to the trivial floor here because a tree
    // is never frustrated and G-set carries no fields for the subgradient to move, while
    // `odd_cycle` charges for exactly the frustration that makes the problem hard.
    // The sweep count is the one dial that trades time for a tighter bound without touching
    // soundness -- the mixing method only chooses WHICH dual point to verify, and a worse choice
    // moves the bound down rather than making it wrong. Exposed so the saturation point can be
    // measured rather than assumed.
    let sweeps = std::env::var("FERROTHERM_SDP_SWEEPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(sdp::Params::default().sweeps);
    let params = sdp::Params { sweeps, ..sdp::Params::default() };
    let ((f, c, sd, cert), t_bound) = Timing::around(|| {
        let f = bound::forest(&inst.graph, 40);
        let c = bound::odd_cycle(&inst.graph, 6);
        let (sd, cert) = sdp::certified(&inst.graph, &params, 1);
        (f, c, sd, cert)
    });

    // THE CERTIFICATE IS RE-CHECKED HERE, not trusted. `verify` rebuilds the cost matrix from the
    // graph and re-runs the positive-definiteness proof, touching nothing from the search that
    // produced it. A bound that only its own author can reproduce is not a bound.
    let sdp_ok = cert.verify(&inst.graph);

    println!("  forest ub      {:>12.0}   ({} forests, peak at round {} of {})",
             inst.cut_upper_bound(f.value), f.parts, f.best_round, f.rounds);
    println!("  odd-cycle ub   {:>12.0}   ({} edge-disjoint frustrated cycles)",
             inst.cut_upper_bound(c.value), c.parts);
    match &sdp_ok {
        Ok(v) => println!("  sdp ub         {:>12.0}   (rank {}, {sweeps} sweeps, re-verified independently)",
                          inst.cut_upper_bound(*v), cert.rank),
        Err(e) => println!("  sdp ub                 --   REFUSED: {e}"),
    }

    // The maximum of sound bounds is sound. They disagree by a lot and in both directions --
    // odd_cycle wins on the degree-4 torus, sdp wins on everything dense -- so taking the better
    // is not a tie-break, it is the result.
    let mut best_bound = &f;
    let mut which = "forest";
    if c.value >= best_bound.value { best_bound = &c; which = "odd-cycle"; }
    if sdp_ok.is_ok() && sd.value >= best_bound.value { best_bound = &sd; which = "sdp"; }
    let ub = inst.cut_upper_bound(best_bound.value);
    println!("  upper bound    {ub:>12.0}   ({which}, {t_bound})");

    // ---- the gap --------------------------------------------------------------------------------
    //
    // Relative to the UPPER bound, because that is the quantity that is certain: the true max cut
    // is somewhere in the bracket, and dividing by a number we are sure of keeps the percentage
    // honest when the bound is loose.
    let gap = (ub - best) / ub * 100.0;
    println!("  gap            {gap:>11.1}%   of the upper bound");

    // The explanation once, at the bottom, rather than four lines of prose per row. The BOUNDS and
    // the CUT above are unaffected by contention -- they are the same numbers on any machine -- so
    // the caveat is scoped to the seconds and says so.
    for c in [t_sample.caveat(), t_bound.caveat()].into_iter().flatten().take(1) {
        println!("\n  NOTE ON THE TIMINGS: {c}");
    }

    // The check that needs no published number, and therefore runs on EVERY instance including the
    // built-in one: a cut this run actually achieved cannot exceed a valid upper bound on the max
    // cut. `best` came from a state whose energy was re-verified above, so if this fires, one of
    // the bounds is unsound on an instance nobody had to supply.
    if best > ub + 1e-6 {
        eprintln!(
            "\n  ** UNSOUND: this run FOUND a cut of {best:.0}, and the upper bound is {ub:.0}. A \
             cut that has been achieved cannot exceed a bound on the maximum. **"
        );
        std::process::exit(1);
    }

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
