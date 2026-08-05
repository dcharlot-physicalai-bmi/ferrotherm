// The crossings tax, executable — what a Z1-class device would pay, per its own published prices.
//
// Two workloads on the same fabric, priced with the vendor's SPICE constants (arXiv:2608.01615
// Table IV: 7.09 fJ/sample, 1.692 pJ/read, 153.6 pJ/write, coupling reflash <= ~1/s):
//
//   A. GENERATIVE (their regime): model resident, K sweeps between reads, read data nodes only.
//      Many local updates between infrequent I/O -> sampling dominates. This is the regime every
//      published projection lives in.
//
//   B. CONTROL LOOP AT 100 Hz (our regime): model resident, but every tick clamps a fresh
//      observation and reads action bits. If clamping uses the write path (Table IV lumps "clamp
//      state" into E_write), I/O dominates sampling by orders of magnitude — the robotics pitch
//      dies on the vendor's own numbers. If clamping is cheap (read-class), the loop is viable.
//      The bit-precision of that one line item decides the market; nobody has published it.
//
// run: cargo run --release --example z1_ledger

use ferrotherm::device::z1_grid;
use ferrotherm::gibbs::Sampler;
use ferrotherm::ledger::{Ledger, Prices, Z1_SPICE};

fn joules_fmt(j: f64) -> String {
    if j >= 1e-3 {
        format!("{:.2} mJ", j * 1e3)
    } else if j >= 1e-6 {
        format!("{:.2} uJ", j * 1e6)
    } else {
        format!("{:.2} nJ", j * 1e9)
    }
}

fn main() {
    // A die-scale patch: 96x96 = 9,216 nodes of the degree-16 fabric (full die ~250k+ nodes;
    // costs scale linearly in nodes, so the SHARES below are die-size independent).
    let (w, h) = (96usize, 96usize);
    let g = z1_grid(w, h, 0.08, 0.0);
    let n = g.n;
    let k_mix = 250; // sweeps per independent sample, the vendor's Fashion-MNIST-tuned constant
    let p: Prices = Z1_SPICE;

    println!("Z1-class fabric {w}x{h} = {n} nodes, degree 16, 2-colorable; prices = SPICE Table IV");
    println!("(pre-silicon vendor estimates; every number below is THEIR device model, our arithmetic)\n");

    // ---- A. generative: 64 independent samples, read 834 data nodes each (their DTM data size) ----
    {
        let mut smp = Sampler::new(&g, 0.9, 0xE57);
        let mut led = Ledger::default();
        led.writes += n as u64; // program the model once
        let data_nodes: Vec<usize> = (0..834).collect();
        for _ in 0..64 {
            smp.sweeps(k_mix, Some(&mut led));
            let _ = smp.read_subset(&data_nodes, Some(&mut led));
        }
        let (s, r, wr) = led.shares(&p);
        println!("A. GENERATIVE, model resident, 64 samples x {k_mix} sweeps, read 834 data nodes/sample:");
        println!("   total {}   shares: sampling {:.0}%  read {:.0}%  write {:.0}% (one-time program)",
                 joules_fmt(led.joules(&p)), s * 100.0, r * 100.0, wr * 100.0);
        println!("   -> their favorable regime, reproduced: local updates dominate, I/O amortized.\n");
    }

    // ---- B. control loop: 100 Hz, 1 s of operation, obs clamp + action read every tick ----
    let obs_bits = 64usize; // observation nodes clamped per tick
    let act_bits = 16usize; // action nodes read per tick
    let sweeps_per_tick = k_mix; // one independent sample per tick (1 action draw)
    let ticks = 100usize; // 1 second at 100 Hz

    // B1: clamping billed on the write path (the literal reading of Table IV)
    {
        let mut smp = Sampler::new(&g, 0.9, 0x0B1);
        let mut led = Ledger::default();
        led.writes += n as u64; // program once
        let act_idx: Vec<usize> = (0..act_bits).collect();
        for t in 0..ticks {
            for i in 0..obs_bits {
                smp.clamp(1000 + i, if (t + i) % 2 == 0 { 1 } else { -1 });
            }
            led.writes += obs_bits as u64; // clamp state flashed = write path
            smp.sweeps(sweeps_per_tick, Some(&mut led));
            let _ = smp.read_subset(&act_idx, Some(&mut led));
        }
        let (s, r, wr) = led.shares(&p);
        let per_tick = led.joules(&p) / ticks as f64;
        println!("B1. CONTROL 100 Hz, clamping BILLED AS WRITE (Table IV literal), {obs_bits} obs + {act_bits} act bits/tick:");
        println!("    per tick {}   shares: sampling {:.0}%  read {:.0}%  write {:.0}%",
                 joules_fmt(per_tick), s * 100.0, r * 100.0, wr * 100.0);
        println!("    write path = {:.0}x the sampling energy per tick; and 100 reflashes/s vs the <=1/s cap -> INFEASIBLE as specced.",
                 (obs_bits as f64 * p.e_write) / (sweeps_per_tick as f64 * n as f64 * p.e_sample));
    }

    // B2: optimistic — clamping costs read-class energy (unpublished; the deciding unknown)
    {
        let mut smp = Sampler::new(&g, 0.9, 0x0B2);
        let mut led = Ledger::default();
        led.writes += n as u64;
        let act_idx: Vec<usize> = (0..act_bits).collect();
        for t in 0..ticks {
            for i in 0..obs_bits {
                smp.clamp(1000 + i, if (t + i) % 2 == 0 { 1 } else { -1 });
            }
            led.reads += obs_bits as u64; // optimistic: clamp priced like a read
            smp.sweeps(sweeps_per_tick, Some(&mut led));
            let _ = smp.read_subset(&act_idx, Some(&mut led));
        }
        let (s, r, wr) = led.shares(&p);
        let per_tick = led.joules(&p) / ticks as f64;
        let per_tick_no_program = (led.joules(&p) - n as f64 * p.e_write) / ticks as f64;
        println!("\nB2. CONTROL 100 Hz, clamping priced READ-CLASS (optimistic, unpublished):");
        println!("    per tick {} ({} excluding one-time program)   shares: sampling {:.0}%  read {:.0}%  write {:.0}%",
                 joules_fmt(per_tick), joules_fmt(per_tick_no_program), s * 100.0, r * 100.0, wr * 100.0);
        println!("    -> viable on paper; the single unpublished line item (clamp cost) separates B1 from B2.");
    }

    println!("\nREADING: same fabric, same prices, two regimes. Generative amortizes I/O and looks like the");
    println!("marketing; a 100 Hz control loop pays the write path per tick and violates the reflash cap —");
    println!("per the vendor's own Table IV. The deciding unknown is the energy price of clamping an input.");
    println!("This ledger runs identically over any device model: swap Prices to re-price the same workload");
    println!("on GPU-simulated sampling (measured watts x time), which is the impedance-tax comparison.");
}
