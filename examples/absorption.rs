//! The absorption gate: every number `docs/ABSORPTION.md` quotes, computed by the crate.
//!
//! Run with `cargo run --release --example absorption`. Nothing here is transcribed; if a figure in
//! the document disagrees with this output, the document is wrong.

use ferrotherm::dsisa::{Assembled, Insn, Program, Substrate};
use ferrotherm::kuramoto::{Kuramoto, dense_coupling_bytes};
use ferrotherm::ledger::Z1_SPICE;
use ferrotherm::nonrev::{
    balance_defect, exact_tau_int, lifted_matrix, reversible_matrix, site_measure,
    stationary_defect,
};
use ferrotherm::phasegen::PhaseGen;
use ferrotherm::potts::{Interaction, PottsBuilder};
use ferrotherm::precision::{
    EmulationCost, GeneratorBudget, ROOM_TEMPERATURE_K, analog_bit_energy,
    analog_to_landauer_ratio, readout_break_even,
};

fn main() {
    println!("== 1. the analogue/Landauer exchange rate ==");
    for b in [1u32, 2, 4, 8, 12, 16] {
        println!(
            "  {b:>2} bits: analogue floor {:.3e} J, Landauer {:.3e} J, ratio {:.1}",
            analog_bit_energy(b, ROOM_TEMPERATURE_K),
            f64::from(b) * 1.380_649e-23 * ROOM_TEMPERATURE_K * core::f64::consts::LN_2,
            analog_to_landauer_ratio(b)
        );
    }

    println!("\n== 2. emulating a continuous phase system with bits ==");
    let n = 8;
    let mut k = vec![0.0; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            k[i * n + j] = 0.5;
            k[j * n + i] = 0.5;
        }
    }
    let sys = Kuramoto::gradient(k, n).expect("symmetric");
    for budget in [1.0f64, 0.1, 0.01] {
        let c = EmulationCost::for_budget(&sys, 1.0, budget);
        println!(
            "  KL <= {budget:>5} nats: q = {:>6}, {:>2} bits/phase, achieved {:.4} nats, \
             read floor {:.3e} J",
            c.q, c.bits, c.kl_nats, c.read_joules
        );
    }

    println!("\n== 3. where a published generator's joules go ==");
    let g = GeneratorBudget {
        oscillators: 16_384,
        steps: 10,
        readout_bits: 8,
        decoder_params: 37_000_000,
        e_update: 7.09e-15,
        e_mac: 1e-12,
    };
    let b = g.breakdown();
    let (d, r, dec) = b.shares();
    println!("  dynamics {:.3e} J  ({:.4}%)", b.dynamics, 100.0 * d);
    println!("  readout  {:.3e} J  ({:.4}%)", b.readout, 100.0 * r);
    println!("  decoder  {:.3e} J  ({:.4}%)", b.decoder, 100.0 * dec);
    println!("  total    {:.3e} J", b.total());
    println!(
        "  a FREE substrate improves the total by {:.5}x  <- Amdahl on the claim",
        b.free_substrate_speedup()
    );
    println!(
        "  readout break-even per update at 10 steps, 8 bits: {:.3e} J (Z1 cycle 7.09e-15)",
        readout_break_even(10, 8)
    );
    println!("  a dense 16,384-oscillator K is {} bytes", dense_coupling_bytes(16_384));

    println!("\n== 4. what an asymmetric coupling is and is not ==");
    let kk = vec![0.0, 1.0, -0.5, 0.25, 0.0, 0.75, -0.5, 0.75, 0.0];
    let asym = Kuramoto::new(vec![0.1, -0.2, 0.05], kk).expect("well formed");
    let mut worst_defect = 0.0f64;
    let mut worst_sol = 0.0f64;
    for a in 0..24 {
        for c in 0..24 {
            let phi = [
                0.0,
                core::f64::consts::TAU * f64::from(a) / 24.0,
                core::f64::consts::TAU * f64::from(c) / 24.0,
            ];
            worst_defect = worst_defect.max(asym.gibbs_defect(&phi).abs());
            worst_sol = worst_sol.max(asym.solenoidal_divergence(&phi).abs());
        }
    }
    println!("  asymmetry |A| = {:.4}, winding |omega| = {:.4}", asym.asymmetry(), asym.winding());
    println!("  max |div F_A|            = {worst_sol:.3e}   <- solenoidal, identically zero");
    println!("  max |grad E . F_A|       = {worst_defect:.4}       <- NOT zero: no Boltzmann law");

    println!("\n== 5. non-reversible acceleration with an invariant measure ==");
    let q = 16;
    let mut pb = PottsBuilder::new(q, 1, Interaction::Clock);
    for c in 0..q {
        let t = core::f64::consts::TAU * c as f64 / q as f64;
        pb.field(0, c as u8, 1.4 * t.cos() + 0.35 * (2.0 * t).cos());
    }
    let m = pb.build();
    let beta = 1.5;
    let pl = lifted_matrix(&m, beta);
    let pr = reversible_matrix(&m, beta);
    let pi = site_measure(&m, beta);
    let mu: Vec<f64> = pi.iter().flat_map(|&x| [x / 2.0, x / 2.0]).collect();
    let obs: Vec<f64> =
        (0..q).map(|c| (core::f64::consts::TAU * c as f64 / q as f64).cos()).collect();
    let lobs: Vec<f64> = obs.iter().flat_map(|&x| [x, x]).collect();
    let tl = exact_tau_int(&pl, &mu, &lobs, 4_000);
    let tr = exact_tau_int(&pr, &pi, &obs, 4_000);
    println!("  lifted:     stationary defect {:.2e}, balance defect {:.4}, tau_int {tl:.3}",
        stationary_defect(&pl, &mu), balance_defect(&pl, &mu));
    println!("  reversible: stationary defect {:.2e}, balance defect {:.2e}, tau_int {tr:.3}",
        stationary_defect(&pr, &pi), balance_defect(&pr, &pi));
    println!("  speed-up from lifting alone: {:.2}x", tr / tl);

    println!("\n== 6. pricing their instruction set ==");
    for sweeps in [1usize, 1_000, 100_000, 390_000, 1_000_000] {
        let asm = ring(sweeps);
        let share = asm
            .configuration_share(&Z1_SPICE, Substrate::Spins, 1.0)
            .expect("Z1_SPICE states every price");
        println!("  Evolve {sweeps:>9} sweeps -> configuration is {:.2}% of the energy", 100.0 * share);
    }

    println!("\n== 7. an exactly certifiable in-fabric generator ==");
    let pg = PhaseGen::new(
        2,
        2,
        vec![2.0, 0.3, 0.3, 1.5],
        vec![0.2, -0.1],
        vec![0.9, 0.1, -0.4, 0.5, 0.2, -0.7, 0.6, 0.3],
        vec![0.0, 0.8, 0.8, 0.0],
    )
    .expect("well formed");
    let e = pg.enumerate_grid(16, 1.0).expect("small enough");
    println!("  exact ln Z of the WHOLE generator (outputs included) = {:.6}", e.log_z);
    println!("  most probable of {} grid states carries p = {:.4}", e.p.len(),
        e.p.iter().copied().fold(0.0f64, f64::max));
}

fn ring(sweeps: usize) -> Assembled {
    let mut p = Program::new(4);
    for i in 0..4 {
        p.push(Insn::Connect { i, j: (i + 1) % 4, w: 1.0 });
    }
    p.push(Insn::Evolve { steps: sweeps });
    p.push(Insn::Store { i: 0 });
    p.assemble().expect("well formed")
}
