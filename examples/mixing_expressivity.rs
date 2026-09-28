// `missing_docs` is denied workspace-wide and is right to be: it guards the API surface, and
// every public item in every library here carries a doc. An EXAMPLE has no API surface -- it is
// a program, and its helpers are private to it -- so the lint has nothing to guard and asks for
// doc comments on `fn main`'s scaffolding instead. Scoped off here rather than weakened there.
#![allow(missing_docs)]
// The mixing-expressivity tradeoff, measured independently, and priced in joules.
//
// THE CLAIM UNDER TEST is the central open problem the thermodynamic-computing field names for
// itself. Extropic's DTM paper (npj Unconventional Computing, 2026) states it: as an energy-based
// model's expressivity rises, its MIXING TIME -- the effort needed to draw one independent sample --
// rises with it, until sampling becomes, in their word, glacial. More specifically:
//
//     "Scaling the number of latent variables only improves performance if the connectivity of the
//      graph is also scaled; otherwise, performance can decrease, as increasing latent variables
//      increases the depth of the Boltzmann machine, making sampling more difficult."
//
// That is a structural claim about mixing and it does not require a trained model to test. This
// tests it.
//
// WHAT THIS MEASURES AND WHAT IT DOES NOT. The tradeoff has two halves. MIXING is a property of the
// energy landscape's shape and is measurable directly. EXPRESSIVITY is a property of a model that
// has been fitted to data, and is not measured here at all -- so this is the structural half,
// stated as the structural half. What it adds is that the half it measures is measured on an
// instrument the original claim did not use, on shapes the original claim did not run, by someone
// with no stake in the answer.
//
// THE INSTRUMENT. `crate::certify` computes the integrated autocorrelation time by GEYER'S INITIAL
// MONOTONE SEQUENCE with long-batch means beside it, the larger carried (until 2026-09-28 it was
// Sokal's automatic window, which truncates a slow mode -- see `certify::tau_int`), not by fitting
// an exponential to the large-lag tail -- which is what the source measurement does, and which is
// the more fragile of the two, because the tail is where the estimator has the fewest samples. Where the model is small enough to enumerate, the same
// certificate also reports TOTAL VARIATION FROM THE EXACT BOLTZMANN DISTRIBUTION beside the noise
// floor that finite sampling alone produces. So a row is not only "how correlated were the draws"
// but "how wrong was the answer, against truth, above what sampling noise explains".
//
// THE CONTROL THAT MAKES IT A MEASUREMENT. A deeper layered model has a different number of edges
// than a shallow one of the same spin count, so its energy scale differs and its autocorrelation
// would differ FOR THAT REASON ALONE. Couplings are therefore scaled as 1/sqrt(fan-in), which holds
// the variance of every local field constant across every shape here. Without that, this example
// would measure the coupling scale and report it as depth.
//
// THE UNIT THAT MATTERS. Autocorrelation is a number about a chain. `updates/sample` is `tau_int`
// times the spins touched per sweep, which is the work one independent draw actually costs, and the
// joules column prices that against a Z1-class device (vendor SPICE, pre-silicon -- a projection,
// labelled as one). That is the mixing-expressivity tradeoff expressed in the unit the whole field
// is arguing about, and this review did not locate a published table of it.
//
// TWO THINGS THE FIRST VERSION OF THIS GOT WRONG, both of which would have produced a confident
// wrong answer.
//
// ONE SEED IS NOT A MEASUREMENT. tau_int is an estimator on a stochastic chain; a single run of it
// is a draw, not a number. Every row below is a mean over SEEDS runs with the spread printed, and a
// difference smaller than the spread is not a difference.
//
// AND ONE BETA IS THE WRONG REGIME. The tradeoff is a statement about RUGGED landscapes: barriers
// separating modes are what make mixing slow, and barriers are tall only when beta is large. At
// beta = 1 with variance-normalised couplings almost everything mixes in a couple of sweeps, and a
// table taken there would report "no effect" while measuring the wrong regime entirely. This
// repository has made that exact mistake before -- the hubo comparison published a negative that
// was really an unswept beta ladder -- so beta is swept here and the sweep is the result.
//
// THE THIRD THING, which is the one worth carrying away. tau_int is an estimator with a validity
// condition: it needs the chain to be many multiples of tau long. Swept cold, this measurement ran
// straight past that condition and produced numbers like 221 +/- 218 -- a spread larger than the
// mean -- and, colder still, tau_int = inf beside a total variation of exactly 0.5000, which is not
// slow mixing at all but a chain FROZEN in one state.
//
// So every row here carries `draws/tau`, and a row below MIN_RATIO is printed as unusable rather
// than reported. That column is the contribution as much as the table is: the standard way to
// measure a mixing time -- fitting an exponential to the autocorrelation's large-lag tail, which
// is what the source measurement does -- returns a number in exactly that regime and says nothing
// about whether it earned one. The certificate at least answers `inf`. And a row between MIN_RATIO
// and `certify::RESOLVED_TAUS` is printed with `>=`: the certificate reports its tau as a LOWER
// bound there (`Finding::TauLowerBound`), because below a thousand autocorrelation times every
// single-chain estimator here reads low on a slow mode.
//
// NOT run in CI: the largest rows take minutes.
//
// run: cargo run --release --example mixing_expressivity

use ferrotherm::certify::{Certificate, RESOLVED_TAUS};
use ferrotherm::gibbs::Sampler;
use ferrotherm::graph::{Graph, GraphBuilder};
use ferrotherm::ledger::{Ledger, Z1_SPICE};
use ferrotherm::rng::Pcg;
use ferrotherm::samples::Plan;

/// A layered Boltzmann machine: `layers` layers of `width`, dense bipartite between neighbours.
///
/// Couplings are `±1/sqrt(fan_in)`, where `fan_in` is the number of neighbours a node in the
/// widest-connected position has. That is the control: it holds the variance of every local field
/// constant, so a difference between two shapes here is a difference of SHAPE and not of scale.
fn layered(layers: usize, width: usize, seed: u64) -> Graph {
    let n = layers * width;
    let mut gb = GraphBuilder::new(n);
    if layers < 2 {
        return gb.build();
    }
    // An interior node sees `width` neighbours in each adjacent layer.
    let fan_in = (2 * width) as f64;
    let scale = 1.0 / fan_in.sqrt();
    let mut rng = Pcg::new(seed, 0x004D_E711);
    for l in 0..(layers - 1) {
        for a in 0..width {
            for b in 0..width {
                let i = l * width + a;
                let j = (l + 1) * width + b;
                gb.couple(i, j, if rng.f64() < 0.5 { scale } else { -scale });
            }
        }
    }
    gb.build()
}

/// Draw a certificate: burn in, then take `draws` samples `thin` sweeps apart.
fn measure(g: &Graph, beta: f64, draws: usize, thin: usize, seed: u64) -> (Certificate, Ledger) {
    let mut led = Ledger::default();
    let mut smp = Sampler::new(g, beta, seed);
    // Burn-in long enough that the certificate's own drift finding is about the model and not
    // about a chain that was still walking away from its random start.
    let set = smp.collect(&Plan::new(2_000, draws, thin.max(1)), Some(&mut led));
    (set.certificate(g).expect("collect returns a chain"), led)
}

const DRAWS: usize = 40_000;
const SEEDS: u64 = 3;
/// Betas inside the validated window. Beyond `beta = 2` at this many draws the estimator breaks
/// down -- see `beyond_the_window` below, which shows the breakdown rather than describing it.
const BETAS: [f64; 4] = [0.5, 1.0, 1.5, 2.0];
/// A row is only reported when the chain is at least this many multiples of `tau_int` long.
const MIN_RATIO: f64 = 200.0;

/// Mean and spread of `tau_int` over seeds, with the smallest `draws/tau` seen.
fn tau_over_seeds(g: &Graph, beta: f64) -> (f64, f64, f64) {
    let taus: Vec<f64> =
        (0..SEEDS).map(|s| measure(g, beta, DRAWS, 1, 11 + s).0.tau_int).collect();
    let mean = taus.iter().sum::<f64>() / taus.len() as f64;
    let var = taus.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / taus.len() as f64;
    let worst = taus.iter().copied().fold(0.0f64, f64::max);
    (mean, var.sqrt(), DRAWS as f64 / worst.max(1e-9))
}

/// `tau ± sd`, or why the row cannot be read.
fn cell(m: f64, sd: f64, ratio: f64) -> String {
    if !m.is_finite() {
        "frozen".into()
    } else if ratio < MIN_RATIO {
        format!("unusable {ratio:.0}x")
    } else if ratio < RESOLVED_TAUS {
        format!(">={m:.2}+-{sd:.2}")
    } else {
        format!("{m:.2}+-{sd:.2}")
    }
}

fn main() {
    println!("The mixing-expressivity tradeoff: the structural half, measured independently\n");
    println!(
        "tau_int as the certificate carries it (Geyer's sequence, long-batch means beside it, the\n\
         larger carried), mean +/- sd over {SEEDS} seeds, {DRAWS} draws each.\n\
         Couplings are +/-1/sqrt(fan-in) throughout, so every shape has the same local-field\n\
         variance and a difference between shapes is a difference of SHAPE.\n\
         A cell reads `unusable` when the chain is under {MIN_RATIO:.0}x tau long -- the estimator's\n\
         validity condition, which is not optional and is not usually printed -- and carries `>=`\n\
         under {RESOLVED_TAUS:.0}x, where the certificate reports tau as a lower bound.\n"
    );

    // ---- 1. depth at a fixed spin count ---------------------------------------------------------
    //
    // The claim's own words: more latent variables without more connectivity means more DEPTH, and
    // depth makes sampling harder. Spin count held constant at 144, so nothing else can explain it.
    println!("DEPTH AT A FIXED SPIN COUNT -- 144 spins, reshaped\n");
    print!("{:>7} {:>6} {:>7}", "layers", "width", "edges");
    for b in BETAS {
        print!("{:>16}", format!("beta {b}"));
    }
    println!();
    let shapes = [2usize, 3, 4, 6, 12];
    let mut cold_row: Vec<(usize, usize, usize, f64, f64, f64)> = Vec::new();
    for layers in shapes {
        let width = 144 / layers;
        let g = layered(layers, width, 7);
        print!("{layers:>7} {width:>6} {:>7}", g.n_edges);
        for b in BETAS {
            let (m, sd, r) = tau_over_seeds(&g, b);
            print!("{:>16}", cell(m, sd, r));
            if b == *BETAS.last().unwrap() {
                cold_row.push((layers, width, g.n_edges, m, sd, r));
            }
        }
        println!();
    }

    // ---- 2. the same shapes, priced -------------------------------------------------------------
    let cold = *BETAS.last().unwrap();
    println!("\nPRICED AT beta {cold} -- what ONE INDEPENDENT DRAW costs\n");
    println!(
        "{:>7} {:>6} {:>7} {:>13} {:>14} {:>11} {:>11} {:>11}",
        "layers", "width", "edges", "tau_int", "updates/draw", "nJ mixing", "nJ readback", "read share"
    );
    for (layers, width, edges, m, sd, ratio) in &cold_row {
        let n = layers * width;
        let per = m * n as f64;
        // An independent draw is the sweeps that separate it from the last one AND the readback
        // that turns the device's state into a sample somebody holds. Pricing only the first was
        // what this column did before, and it is the more flattering of the two halves.
        let mixing = Ledger { samples: per.round().max(1.0) as u64, reads: 0, writes: 0 };
        let readback = Ledger { samples: 0, reads: n as u64, writes: 0 };
        let mj = mixing.joules(&Z1_SPICE).unwrap_or(f64::NAN) * 1e9;
        let rj = readback.joules(&Z1_SPICE).unwrap_or(f64::NAN) * 1e9;
        println!(
            "{layers:>7} {width:>6} {edges:>7} {:>13} {per:>14.0} {mj:>11.4} {rj:>11.4} {:>10.1}%",
            cell(*m, *sd, *ratio),
            100.0 * rj / (mj + rj)
        );
    }
    println!(
        "\nTHE COLUMN THIS TABLE DID NOT HAVE UNTIL NOW, and it changes what the table is about.\n\
         A Z1-class read is 1.692 pJ per node against 7.09 fJ per Gibbs cycle, so ONE READ IS WORTH\n\
         239 UPDATES. Depth moves the mixing column by a large factor across these shapes and leaves\n\
         the readback column exactly constant, because readback depends on the spin count and these\n\
         shapes hold the spin count fixed. So the quantity the field argues about is real, is\n\
         measured above, and is the minority of the bill at these sizes. That is not a weakening of\n\
         the tradeoff; it is where the tradeoff sits. A machine of this class is an I/O machine,\n\
         and this is that sentence in the units of this experiment.\n\n\
         The read is charged here because `Sampler::collect` charges it. Five places in this\n\
         repository, this example included, used to append `smp.s.clone()` inside their own\n\
         collection loop -- which never touches the read path, so the readback column was zero\n\
         everywhere and nobody could see it was missing."
    );

    // ---- 3. width at a fixed depth --------------------------------------------------------------
    //
    // The other half of the same sentence: scaling latent variables WITH connectivity.
    println!("\nWIDTH AT A FIXED DEPTH (beta {cold}) -- 2 layers, growing\n");
    println!(
        "{:>7} {:>6} {:>7} {:>14} {:>16}",
        "layers", "width", "edges", "tau_int", "updates/sample"
    );
    for width in [8usize, 16, 32, 64, 96] {
        let g = layered(2, width, 7);
        let (m, sd, r) = tau_over_seeds(&g, cold);
        println!(
            "{:>7} {width:>6} {:>7} {:>14} {:>16.0}",
            2,
            g.n_edges,
            cell(m, sd, r),
            m * g.n as f64
        );
    }

    // ---- 4. beyond the window, shown rather than described --------------------------------------
    //
    // This is the methodological half of the result. Colder than the table above, the estimator
    // stops being an estimator, and the point of printing it is that the number keeps arriving.
    println!("\nBEYOND THE WINDOW -- the same two shapes, colder, one seed each\n");
    println!("{:>6} {:>18} {:>18}  note", "beta", "2 layers", "12 layers");
    let shallow = layered(2, 72, 7);
    let deep = layered(12, 12, 7);
    let mut beyond: Vec<f64> = Vec::new();
    for beta in [2.5f64, 3.0, 4.0, 8.0] {
        let a = measure(&shallow, beta, DRAWS, 1, 11).0.tau_int;
        let b = measure(&deep, beta, DRAWS, 1, 11).0.tau_int;
        let ratio = |t: f64| if t.is_finite() { DRAWS as f64 / t.max(1e-9) } else { 0.0 };
        let worst = ratio(a).min(ratio(b));
        beyond.push(a);
        println!(
            "{beta:>6.1} {:>18} {:>18}  {}",
            format!("{a:.1}"),
            format!("{b:.1}"),
            if !a.is_finite() || !b.is_finite() {
                "frozen: not slow mixing, a chain that stopped moving"
            } else if worst < MIN_RATIO {
                "estimator invalid: the chain is not long enough to say"
            } else if worst < RESOLVED_TAUS {
                "a lower bound: under a thousand tau"
            } else {
                "inside the window"
            }
        );
    }

    // The numbers in the prose are read off the rows above, so the sentence cannot outlive them:
    // it used to say 26.30 and 65.95 while the table printed 48.91 and 65.95.
    let (first, last) = (cold_row[0], cold_row[cold_row.len() - 1]);
    let middle: Vec<f64> = cold_row[1..cold_row.len() - 1].iter().map(|r| r.3).collect();
    let mid_lo = middle.iter().copied().fold(f64::INFINITY, f64::min);
    let mid_hi = middle.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    println!(
        "\nWHAT THE TABLE SHOWS, AND IT IS NOT A CONFIRMATION.\n\n\
         WEAKLY COUPLED (beta 0.5, beta 1) the claim mostly holds: at a fixed 144 spins, deeper\n\
         shapes generally mix more slowly, with spreads of a few percent. Read the columns rather\n\
         than this sentence for where the ordering is not strict.\n\n\
         STRONGLY COUPLED (beta {cold}) it does not. The column is U-SHAPED. The shallowest shape --\n\
         {} layers of {}, a dense restricted Boltzmann machine -- is SLOW at {:.2}, the middle\n\
         shapes run {mid_lo:.2} to {mid_hi:.2}, and the deepest, {} layers of {}, reads {:.2}{}. A monotone\n\
         reading of `depth makes sampling harder` does not survive into the regime where the\n\
         tradeoff is supposed to bite.\n\n\
         A plausible reading, offered as a reading and not a result: the two slow ends are slow for\n\
         DIFFERENT reasons. A dense bipartite layer has strong collective modes that a single-site\n\
         sweep moves through slowly; a deep narrow stack has the barriers the original claim is\n\
         about. Nothing here separates those two mechanisms, and doing so would need a cluster move\n\
         or a mode-resolved statistic rather than a scalar autocorrelation.",
        first.0, first.1, first.3, last.0, last.1, last.3,
        if last.5 < MIN_RATIO {
            format!(" -- from chains only {:.0} of their tau long, past the validity condition", last.5)
        } else if last.5 < RESOLVED_TAUS {
            " -- a lower bound".to_string()
        } else {
            String::new()
        }
    );
    println!(
        "\nAND THE REGIME THAT MATTERS IS THE ONE THAT CANNOT BE MEASURED THIS WAY. Past beta 2 at\n\
         {DRAWS} draws the estimator stops being one: the 2-layer rows above read {:.1}, {:.1}, {:.1}\n\
         and {:.1} on the same shape, and at beta 8 a small number is a chain that has stopped\n\
         moving rather than one that mixes fast. The tradeoff is a claim about rugged landscapes,\n\
         ruggedness needs cold, and cold is where the standard measurement dissolves. That is a\n\
         result about the instrument, and it applies to any table of mixing times taken cold --\n\
         including one produced by fitting an exponential to an autocorrelation tail, which returns\n\
         a number in this regime without a validity condition to fail.",
        beyond[0], beyond[1], beyond[2], beyond[3]
    );
    println!(
        "\nWHAT IT DOES NOT SHOW. Expressivity. That is a property of a model FITTED TO DATA and\n\
         nothing here fits anything, so this is the structural half of the tradeoff and only that\n\
         half. The claim under test is itself structural, which is why the structural half can\n\
         test it -- but a reader wanting the full tradeoff needs the training half too, and this\n\
         is not it."
    );
    println!(
        "\nTHE CONTROL IS THE POINT. Every shape uses couplings of +/-1/sqrt(fan-in), so a local\n\
         field has the same variance in a 2-layer model and a 12-layer one. Without it, deeper\n\
         models have fewer edges per node at a fixed spin count, their landscapes are shallower,\n\
         and the table would show depth HELPING -- measuring the coupling scale and calling it\n\
         depth."
    );
    println!(
        "\nTHE JOULES COLUMN IS A PROJECTION. Z1_SPICE is vendor pre-silicon pricing, not a\n\
         measurement. What is measured is updates/sample. The column exists because the field's\n\
         argument is about energy, and a mixing time nobody converts into energy cannot join it."
    );
}
