// Z1T's headline, taken apart with its own arithmetic.
//
// On 2026-09-04 Extropic published Z1T -- transformer-like models for the Z1 sparse probabilistic
// chip -- with "up to 140x energy efficiency gains over GPUs". The post is unusually complete: it
// gives the per-token split, the baseline, the utilisation assumption, and, in a separate section,
// the loss penalty its own architecture pays. Everything below is THEIR numbers and our arithmetic.
//
// Three questions the multiplier does not answer on its own, and this program does:
//
//   1. How much of it is the sampler?          -> Split::ceiling, with the accelerator set to ZERO
//   2. Which opponent was it measured against? -> Split::with_baseline_scaled over the MFU axis
//   3. Is it the same unit of work?            -> Split::with_work_multiplier, iso-loss not iso-param
//
// run: cargo run --release --example z1t_ledger

use ferrotherm::hybrid::{Split, Z1T_PUBLISHED};

fn main() {
    let s = Z1T_PUBLISHED;
    println!("Z1T per token, as published (extropic.ai/writing/z1t, 2026-09-04)");
    println!("  source: {}\n", s.source);
    println!(
        "  Z1 {:.2} nJ + FPGA {:.2} nJ = {:.2} nJ/token; H100 baseline {:.1} uJ/token @ 10% MFU",
        s.accel_joules * 1e9,
        s.host_joules * 1e9,
        s.total() * 1e9,
        s.baseline_joules * 1e6
    );
    let f = s.factor().expect("the published split prices a real unit of work");
    println!("  -> {f:.1}x   (the post says \"up to 140x\"; this reproduces it)\n");

    // ---- 1. the ceiling: what the thermodynamic half is worth, at most, ever ----
    let ceiling = s.ceiling().expect("the FPGA half is nonzero, which is the whole point");
    let headroom = s.headroom().expect("likewise");
    println!("1. WHAT THE SAMPLER IS WORTH");
    println!("   Z1's share of the hybrid's energy:        {:.1}%", s.accel_share().unwrap() * 100.0);
    println!("   headline with Z1 at EXACTLY zero joules:  {ceiling:.1}x");
    println!("   so every future improvement to the sampler, summed over all time, is worth");
    println!("   {:.1}% -- and the remaining orders of magnitude are the FPGA's to find.\n",
             (headroom - 1.0) * 100.0);

    // ---- 2. the opponent: the multiplier is linear in the baseline, so quote the axis ----
    println!("2. WHICH OPPONENT (the post gives all three; the headline quotes the first)");
    for (mfu, k) in [(10.0, 1.0), (50.0, 0.2), (100.0, 0.1)] {
        let at = s.with_baseline_scaled(k);
        println!(
            "   H100 @ {mfu:>5.0}% MFU  baseline {:>6.2} uJ/token   ->  {:>6.1}x   (ceiling {:>6.1}x)",
            at.baseline_joules * 1e6,
            at.factor().unwrap(),
            at.ceiling().unwrap()
        );
    }
    println!("   Low MFU is not a strawman -- batch-1 sequential decode really does idle a GPU --");
    println!("   but it is a fact about the opponent, not about the substrate.\n");

    // ---- 3. the unit of work: iso-parameter is not iso-quality ----
    // The post: the H100 runs "the same next-token step run densely, with no sparsity exploited".
    // Also the post: "we need about an order of magnitude more FLOPs with our sparser Z1T model to
    // achieve the same loss as a GPT-2 model". Both cannot be inside the same multiplier.
    const ISO_LOSS_FLOPS: f64 = 10.0;
    println!("3. WHICH UNIT OF WORK  (charging the post's own ~{ISO_LOSS_FLOPS:.0}x iso-loss FLOP penalty)");
    println!("   {:>16}  {:>12}  {:>12}", "baseline", "iso-parameter", "iso-loss");
    for (label, k) in [("H100 @ 10% MFU", 1.0), ("H100 @ 50% MFU", 0.2), ("H100 @100% MFU", 0.1)] {
        let iso_param = s.with_baseline_scaled(k);
        let iso_loss = iso_param.with_work_multiplier(ISO_LOSS_FLOPS);
        println!(
            "   {label:>16}  {:>11.1}x  {:>11.2}x",
            iso_param.factor().unwrap(),
            iso_loss.factor().unwrap()
        );
    }
    println!("   The corner a datacentre would actually buy -- same quality, GPU kept busy -- is");
    println!("   the bottom right. The headline is the top left. Same hardware, same post.\n");

    // ---- 4. the roadmap, as an engineering statement about a specific chip ----
    println!("4. THE STATED PATH TO 1000x, INVERTED");
    for target in [500.0, 1000.0] {
        match s.host_speedup_for(target) {
            Ok(k) => {
                let budget = s.host_joules_for(target).unwrap();
                println!(
                    "   {target:>6.0}x needs the FPGA half at {:>6.2} nJ/token -- {k:.2}x better than today",
                    budget * 1e9
                );
            }
            Err(e) => println!("   {target:>6.0}x: {e}"),
        }
    }
    let free = Split { accel_joules: 0.0, ..s };
    println!(
        "   ... and with Z1 running FREE it is still {:.2}x. The sampler is not what is in the way.",
        free.host_speedup_for(1000.0).unwrap()
    );
    println!("\n   The post says this itself: \"the FPGA consumes the vast majority (>95%) of the");
    println!("   energy... potentially reaching up to 1000x greater energy efficiency than GPUs\"");
    println!("   with a chip designed for these models. That is a correct diagnosis of a bottleneck");
    println!("   in a part that is not the thermodynamic one, and it is the honest version of 140x.");

    // ---- 5. what is measured, and what is not ----
    println!("\n5. PROVENANCE");
    println!("   MEASURED: the H100 baseline (2026-08-12, torch 2.7.0, batch-1 decode).");
    println!("   PROJECTED: both halves of the numerator. Z1 is at tapeout; the post says its");
    println!("   energy is \"theoretical... anchored to reality from our experiments with similar");
    println!("   pbits in X0\", and the FPGA half is a 0.2 pJ/MAC + 3.0 pJ/scalar + 1.5 W model.");
    println!("   So the ratio is one measurement over two estimates, and the 3%-share term is the");
    println!("   only one with any silicon behind it at all.");
    println!(
        "   Its price also moved: 1.3e-14 J/sample here vs 7.09e-15 in the Thermalizers appendix\n   six weeks earlier -- {:.2}x apart, for one quantity. This crate carries both, asserts neither.",
        1.3e-14 / ferrotherm::ledger::Z1_SPICE.e_sample
    );
}
