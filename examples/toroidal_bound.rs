// The G-set toroidal instances, bracketed. How close is the published best-known cut to optimal?
//
// G11, G12 and G13 are toroidal grids, and a torus is not a plane -- `planarcut::solve` refuses
// them, correctly. But the dual argument needs only FACES, and an embedding on any surface has
// them, so the same reduction runs on a toroidal embedding. What changes is what the answer means:
// on a torus the cycle space of the dual is four times the cut space, so the relaxation ranges over
// sets that are not cuts and its optimum is an UPPER BOUND.
//
// Which is exactly what G-set has been missing. Every published figure is a best cut FOUND -- a
// lower bound. This is the other side, and together they bracket the optimum.
//
// And the bound can turn into a proof for free: an even subgraph of the dual is a cut exactly when
// it two-colours the graph, so when it does, the bound is ATTAINED and the best-known cut is
// proved optimal. When it does not, the bracket stands and says so.
//
// The grid dimensions are recovered from the edge list rather than assumed. A match on all 1,600
// edges is a proof of structure, not a guess.
//
// NOT run in CI: it needs the G-set files.
//
//   cargo run --release --example toroidal_bound -- <file> [best-known-cut]

use ferrotherm::gset::Instance;
use ferrotherm::host::Timing;
use ferrotherm::{bls, planar, planarcut};

fn main() {
    let mut args = std::env::args().skip(1);
    let Some(path) = args.next() else {
        eprintln!("usage: toroidal_bound <file> [best-known-cut]");
        std::process::exit(2);
    };
    let best_known: Option<f64> = args.next().and_then(|v| v.parse().ok());
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        eprintln!("cannot read {path}: {e}");
        std::process::exit(2);
    });
    let inst = Instance::parse(&text).unwrap_or_else(|e| {
        eprintln!("{path}: {e}");
        std::process::exit(2);
    });
    let name = path.rsplit('/').next().unwrap_or(&path);
    println!("{name}: {} nodes, {} edges", inst.nodes, inst.edges);

    let Some(emb) = planar::torus_grid_of(&inst.graph) else {
        println!("  not a toroidal grid: this example has nothing to say about it");
        std::process::exit(0);
    };
    let w = emb.rotation(0).len();
    println!("  recovered a toroidal embedding: {} faces, chi = {}, genus {:?}",
             emb.faces().len(), emb.euler(), emb.genus());
    let _ = w;

    let (r, t) = Timing::around(|| {
        planarcut::bound_on_surface(&inst.graph, &emb, &planarcut::Params::default())
    });
    let b = match r {
        Ok(b) => b,
        Err(e) => {
            println!("  refused: {e}");
            std::process::exit(1);
        }
    };
    println!("  matching over {} odd dual faces", b.odd_faces);

    // OUR OWN lower bound, so the bracket does not depend on a number from a table. A cut this
    // program produced and can hand back is a lower bound in the only sense that matters.
    let ours = bls::search(
        &inst.graph,
        &bls::Params { iterations: 2_000_000, ..bls::Params::default() },
        11,
    );
    let ours_cut = inst.cut(&ours.state);

    println!();
    println!("  upper bound   {:>12.0}   (this run, from the toroidal dual)", b.cut);
    println!("  our cut       {:>12.0}   (this run, breakout local search)", ours_cut);
    if let Some(bk) = best_known {
        println!("  best known    {:>12.0}   (published; a lower bound, somebody achieved it)", bk);
        // A cut ABOVE a claimed upper bound would mean the bound is unsound. Worth failing over.
        if bk > b.cut + 1e-6 {
            eprintln!("\n  ** the published cut {bk} EXCEEDS this upper bound {:.0}, so the bound \
                       is UNSOUND **", b.cut);
            std::process::exit(1);
        }
    }
    if ours_cut > b.cut + 1e-6 {
        eprintln!("\n  ** our own cut {ours_cut} EXCEEDS our own upper bound {:.0}. One of the two \
                   is wrong and neither is reported. **", b.cut);
        std::process::exit(1);
    }

    let lower = best_known.map_or(ours_cut, |bk| bk.max(ours_cut));
    let gap = b.cut - lower;
    println!();
    if (b.cut - ours_cut).abs() < 1e-9 {
        println!("  ** PROVED OPTIMAL, END TO END. This run found a cut of {ours_cut:.0} and this");
        println!("  run proved nothing above {:.0} exists. No published number was needed.", b.cut);
    } else if gap.abs() < 1e-9 {
        println!("  ** THE BRACKET IS CLOSED: the maximum cut is exactly {lower:.0}. **");
        println!("  The upper bound is this run's; the matching lower bound is the published one,");
        println!("  which is a cut somebody achieved. Our own search reached {ours_cut:.0}.");
    } else {
        println!("  the maximum cut lies in [{lower:.0}, {:.0}] -- a gap of {gap:.0}, {:.2}% of the",
                 b.cut, gap / b.cut * 100.0);
        println!("  bound. Every published G-set figure is the LEFT end of a bracket like this one.");
    }

    println!();
    if b.attained {
        println!("  ATTAINED: the relaxation's optimum is itself a genuine cut, so the bound is the");
        println!("  maximum by construction. Energy {:.0}.", b.energy.unwrap_or(f64::NAN));
    } else {
        println!("  NOT ATTAINED: the relaxation's optimum is an even subgraph of the dual that is");
        println!("  not null-homologous, so it is not a cut and no state is reported. The bound");
        println!("  stands regardless -- every cut IS such a subgraph, so the maximum over the");
        println!("  larger set can only be larger.");
    }
    if let Some(c) = t.caveat() {
        println!("\nNOTE: {c}");
    }
}
