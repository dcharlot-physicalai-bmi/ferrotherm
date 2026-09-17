//! The number that decides whether a dense coupled-oscillator architecture can be built.
//!
//! ```text
//!   cargo run --release --example sparse_k
//! ```
//!
//! `docs/ABSORPTION.md` prices a published coupled-oscillator generator and finds the decoder is
//! 99.997% of its energy. This asks the other question, the one that is about the fabric rather
//! than the bill: the trained coupling is **dense** and every fabric of physically coupled
//! oscillators is **sparse**, so something has to give, and there are exactly three things that can.
//!
//! | route | keeps | costs | priced by |
//! |---|---|---|---|
//! | truncate | the node count | accuracy, in nats | `kuramoto::Truncated::kl_bound` |
//! | sparsify | the ground states, exactly | nodes, as copies | `sparsify::copies_for` |
//! | embed | the model, on a named machine | physical sites, as chains | `embed::site_lower_bound` |
//!
//! Sections 1 and 2 measure the first route exactly, by enumerating both laws — which caps the
//! system at a handful of oscillators, and the output says so. Section 3 prices the second at the
//! published scale, where the arithmetic is closed-form and no enumeration is needed.
use ferrotherm::diffuse::kl;
use ferrotherm::kuramoto::{dense_coupling_bytes, Kuramoto};
use ferrotherm::potts::enumerate;
use ferrotherm::rng::Pcg;
use ferrotherm::sparsify::copies_for;

/// Oscillators in the enumerable model. `q^n` states have to be enumerable twice.
const N: usize = 5;
/// Phase grid for the exact laws.
const Q: usize = 6;
/// Inverse temperature everything is measured at.
const BETA: f64 = 0.6;
/// The published dense architecture's oscillator count.
const PUBLISHED_N: usize = 16_384;

/// A dense coupling with a heavy tail: most pairs weak, a few strong.
///
/// The shape matters more than the values, and it is the whole answer: whether truncation is cheap
/// or ruinous is a question about how much of the mass sits in how few pairs, not about the
/// architecture. `flat` is the control.
fn heavy_tailed(n: usize, seed: u64) -> Kuramoto {
    let mut rng = Pcg::new(seed, 0x0DE5_5E11);
    let mut k = vec![0.0; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            let u = rng.f64();
            let w = 0.05 + 1.2 * u * u * u;
            k[i * n + j] = if rng.f64() < 0.5 { w } else { -w };
            k[j * n + i] = k[i * n + j];
        }
    }
    Kuramoto::gradient(k, n).expect("symmetric by construction")
}

/// Every pair the same magnitude: the coupling for which no truncation is cheap.
fn flat(n: usize, seed: u64) -> Kuramoto {
    let mut rng = Pcg::new(seed, 0x0F1A_7000);
    let mut k = vec![0.0; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            k[i * n + j] = if rng.f64() < 0.5 { 0.4 } else { -0.4 };
            k[j * n + i] = k[i * n + j];
        }
    }
    Kuramoto::gradient(k, n).expect("symmetric by construction")
}

/// One profile's truncation table, printed and returned as the worst KL a fabric degree costs.
fn table(label: &str, sys: &Kuramoto) {
    println!("  -- {label} --");
    println!(
        "  {:>6} {:>7} {:>9} {:>11} {:>11} {:>8}",
        "degree", "pairs", "mass kept", "KL measured", "KL bound", "slack"
    );
    let full = enumerate(&sys.to_clock(Q), BETA).expect("small enough to enumerate");
    for d in 1..N {
        let t = sys.truncate_to_degree(d);
        let cut = enumerate(&t.system.to_clock(Q), BETA).expect("small enough to enumerate");
        let measured = kl(&full.p, &cut.p);
        let bound = t.kl_bound(BETA);
        let slack = if measured > 0.0 { bound / measured } else { f64::INFINITY };
        println!(
            "  {:>6} {:>7} {:>8.1}% {:>11.4} {:>11.4} {:>7.1}x",
            t.degree, t.kept, 100.0 * t.mass_kept(), measured, bound, slack
        );
    }
}

fn main() {
    let sys = heavy_tailed(N, 20_260_917);
    let pairs = N * (N - 1) / 2;
    println!("== 1. the model ==");
    println!("  {N} oscillators, all-to-all: {pairs} pairs, coupling mass {:.4}", sys.coupling_mass());
    println!(
        "  at the published {PUBLISHED_N} that same density is {} bytes of couplings in f64",
        dense_coupling_bytes(PUBLISHED_N)
    );

    println!("\n== 2. truncating to a fabric's degree, measured exactly ==");
    println!("  both laws enumerated on a {Q}-point grid at beta = {BETA}; nothing is sampled");
    table("heavy-tailed weights", &sys);
    table("flat weights (the control)", &flat(N, 4242));
    println!("  The two tables are the answer: truncation is cheap when the mass is concentrated");
    println!("  and ruinous when it is not, and that is a property of the TRAINED WEIGHTS rather");
    println!("  than of the architecture. `slack` is bound over measured -- the bound uses the");
    println!("  worst case pointwise, so it is loose wherever the dropped couplings cancel.");
    println!("  (the bound is the same 2 beta epsilon argument the phase grid uses, so the two");
    println!("   mismatches -- discrete phases and sparse couplings -- add in one unit)");

    println!("\n== 3. keeping every coupling instead, at the published scale ==");
    println!("  copy-splitting preserves the ground states exactly and pays in nodes:");
    println!("  {:>7} {:>12} {:>16}", "degree", "copies/osc", "oscillators");
    for d in [3usize, 4, 6, 8, 16, 100] {
        let c = copies_for(PUBLISHED_N - 1, d);
        println!("  {:>7} {:>12} {:>16}", d, c, PUBLISHED_N * c);
    }
    let six = PUBLISHED_N * copies_for(PUBLISHED_N - 1, 6);
    println!();
    println!("  A degree-6 fabric holding the dense coupling exactly needs {six} oscillators");
    println!("  for a model of {PUBLISHED_N}: a factor of {}.", six / PUBLISHED_N);
    println!("  That factor is the choice. Pay it in silicon, or pay the nats in section 2, or");
    println!("  train a coupling that was sparse to begin with -- which nothing published does.");

    println!("\n== what this shows and what it does not ==");
    println!("  shown:     the truncation cost is exact here, and bounded everywhere by a quantity");
    println!("             that needs no enumeration -- 2 beta times the dropped coupling mass.");
    println!("  not shown: the truncation cost of a REAL trained coupling. That needs the weights,");
    println!("             and the published models ship a decoder rather than a K. The two");
    println!("             profiles above bracket it, and the measurement to demand of any such");
    println!("             architecture is one number: what fraction of the coupling mass sits in");
    println!("             the heaviest d couplings per oscillator.");
}
