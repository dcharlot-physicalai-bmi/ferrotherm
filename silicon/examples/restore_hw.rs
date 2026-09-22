#![allow(missing_docs)]
// IS A CAPTURED BIT REAL FLIP-FLOP STATE, OR JUST A CELL THAT CHANGED?
//
// `examples/capture_hw` established (2026-09-19) that GCAPTURE latches something live into
// configuration memory on this part. It also recorded what it could not establish: that a captured
// bit is the value of a flip-flop. Two readbacks differing is equally consistent with a capture
// that scrambles, or writes a counter, or reports some function of the cells.
//
// This drives the state the OTHER WAY and asks for it back. GRESTORE loads every flip-flop FROM
// its configuration cell; GCAPTURE writes every flip-flop back INTO it. So capture a state, command
// it back into the flip-flops, capture again, and the result is predictable: it must come back.
//
//   C1              capture the running design's state into the cells
//   R = restore+capture   force the flip-flops from those cells, then re-capture them
//   C2              a plain capture after the same delay -- the control
//
// If the captured bits are flip-flop state, `R` reproduces `C1` and `C2` does not, because the
// design kept running in both cases and only `R` had the state imposed on it first. If capture were
// any read-only function of configuration memory, `R` and `C2` would behave alike.
//
// GRESTORE and GCAPTURE ride in ONE packet stream (`capture::restore_capture_and_read`): the fabric
// clock stops for neither, so a USB round trip between them would be hundreds of fabric cycles.
//
// NON-DESTRUCTIVE TO CONFIGURATION: nothing is reconfigured. It does perturb the RUNNING design's
// flip-flops, which is the experiment; a power cycle reloads the board from flash.
//
// run on the board host:
//   cargo run --release -p ferrotherm-silicon --features flash --example restore_hw

use ferrotherm_silicon::bitstream::reg;
use ferrotherm_silicon::flash::{Ftdi, Stat, Tap};
use ferrotherm_silicon::tilegrid::Far;

const FRAMES: usize = 36;
const WANTED: usize = 6;
const PASSES: usize = 8;

fn differing_bits(a: &[u32], b: &[u32]) -> u32 {
    a.iter().zip(b).map(|(x, y)| (x ^ y).count_ones()).sum()
}

fn main() {
    let (ftdi, product) = match Ftdi::open("Alchitry") {
        Ok(v) => v,
        Err(e) => {
            println!("no board: {e}");
            return;
        }
    };
    let mut tap = Tap::new(ftdi).expect("tap init");
    let stat = Stat(tap.read_config_reg(reg::STAT).expect("stat"));
    println!("board: {product}   {}", stat.describe());
    if !stat.done() {
        println!("no design is running (DONE low), so there is no flip-flop state to impose.");
        return;
    }

    println!();
    println!("  finding columns whose state moves on its own");
    let mut live: Vec<u32> = Vec::new();
    'scan: for bottom_half in [false, true] {
        for row in 0..3u8 {
            for column in 0..96u16 {
                let far = Far { block_type: 0, bottom_half, row, column, minor: 0 }.encode();
                let Ok(a) = tap.capture_frames(far, FRAMES) else { continue };
                std::thread::sleep(std::time::Duration::from_millis(30));
                let Ok(b) = tap.capture_frames(far, FRAMES) else { continue };
                if differing_bits(&a, &b) > 0 {
                    live.push(far);
                    println!("    live column at FAR 0x{far:08X}");
                    if live.len() >= WANTED {
                        break 'scan;
                    }
                }
            }
        }
    }
    if live.is_empty() {
        println!("  nothing moved, so there is no free-running state to impose one on.");
        return;
    }

    // FIRST, WHETHER THE EXPERIMENT CAN WORK AT ALL. Imposing a value only means something if the
    // design can hold it for the handful of configuration clocks between GRESTORE and GCAPTURE. A
    // free runner advances every fabric clock and shows state moving even at the shortest delay
    // this transport can produce; an occasional event shows little at a short delay and more at a
    // long one. This is measured before the verdict rather than offered after it as an excuse.
    // FIRST, WHETHER THE EXPERIMENT CAN WORK AT ALL. Imposing a value only means something if the
    // design can hold it for the handful of configuration clocks between GRESTORE and GCAPTURE. A
    // free runner advances every fabric clock and shows state moving even at the shortest delay
    // this transport can produce; an occasional event shows little at a short delay and more at a
    // long one. Measured before the verdict rather than offered after it as an excuse.
    println!();
    println!("  can these columns hold a value? bits moving between two captures, by delay");
    println!("  {:>10}  {:>7}  {:>7}  {:>7}  {:>7}   reading", "FAR", "0 ms", "1 ms", "10 ms", "100 ms");
    let mut free_running = 0usize;
    for &far in &live {
        let mut d = Vec::new();
        for ms in [0u64, 1, 10, 100] {
            let a = tap.capture_frames(far, FRAMES).expect("a");
            if ms > 0 {
                std::thread::sleep(std::time::Duration::from_millis(ms));
            }
            let b = tap.capture_frames(far, FRAMES).expect("b");
            d.push(f64::from(differing_bits(&a, &b)));
        }
        // Flat in the delay means it had already moved as far as it goes before the shortest delay
        // this host can produce: it is reclocking, not accumulating.
        let flat = d[0] > 0.0 && d[3] < 3.0 * d[0];
        if flat {
            free_running += 1;
        }
        println!(
            "  0x{far:08X}  {:>7.0}  {:>7.0}  {:>7.0}  {:>7.0}   {}",
            d[0], d[1], d[2], d[3],
            if flat { "free-running" } else { "accumulates with time" }
        );
    }

    // THE COMPARISON IS PAIRED, AND IT HAS TO BE. One restore+capture against one plain capture is
    // two draws from counts that swing by half their own value: an earlier version of this example
    // reported "state came back" on a column where the same pair measured the opposite way minutes
    // later. So each pass contributes ONE difference under identical conditions, and the standard
    // error is taken over the passes, not over the bits.
    println!();
    println!("  imposing a captured state and asking for it back, {PASSES} paired passes");
    println!("  {:>10}  {:>10}  {:>10}  {:>14}  {}", "FAR", "restore", "control", "diff +- se", "verdict");
    let (mut resolved, mut tried) = (0usize, 0usize);
    for &far in &live {
        let mut diffs = Vec::new();
        let (mut sum_r, mut sum_c) = (0.0f64, 0.0f64);
        for _ in 0..PASSES {
            let c1 = tap.capture_frames(far, FRAMES).expect("c1");
            let r = tap.restore_capture_frames(far, FRAMES).expect("restore+capture");
            let d_restore = f64::from(differing_bits(&c1, &r));

            let c2 = tap.capture_frames(far, FRAMES).expect("c2");
            let c3 = tap.capture_frames(far, FRAMES).expect("c3");
            let d_control = f64::from(differing_bits(&c2, &c3));

            sum_r += d_restore;
            sum_c += d_control;
            diffs.push(d_restore - d_control);
        }
        let n = diffs.len() as f64;
        let mean = diffs.iter().sum::<f64>() / n;
        let var = diffs.iter().map(|d| (d - mean) * (d - mean)).sum::<f64>() / (n - 1.0);
        let se = (var / n).sqrt();
        let sigma = if se > 0.0 { mean.abs() / se } else { 0.0 };
        tried += 1;
        // A restore that imposed the captured state would make `restore` SMALLER than `control`.
        let held = mean < 0.0 && sigma > 3.0;
        if held {
            resolved += 1;
        }
        println!(
            "  0x{far:08X}  {:>10.1}  {:>10.1}  {:>7.1} +-{:>4.1}  {}",
            sum_r / n, sum_c / n, mean, se,
            if held { "state came back" } else if sigma > 3.0 { "restore made it WORSE" } else { "no effect at 3 sigma" }
        );
    }

    println!();
    println!("  {resolved} of {tried} columns returned the state they were given.");
    if resolved == tried && tried > 0 {
        println!("  The imposed state survived the round trip and free-running state did not, so the");
        println!("  captured bits are written INTO the flip-flops by GRESTORE and read back OUT of them");
        println!("  by GCAPTURE. A read-only function of configuration memory cannot do that.");
    } else if free_running == tried {
        println!("  AND EVERY COLUMN IS FREE-RUNNING, which is why. A value imposed on a flip-flop that");
        println!("  reclocks every fabric cycle is gone before the next command reaches it, so this");
        println!("  design cannot answer the question however GRESTORE behaves. The negative is a");
        println!("  property of the fixture, not evidence against the opcode.");
    } else {
        println!("  Some columns can hold a value and still did not return it, which IS evidence about");
        println!("  GRESTORE rather than about the fixture. Worth a closer look at those columns.");
    }
    println!();
    println!("  WHAT WOULD SETTLE IT: a design whose state HOLDS between two commands, so a value can");
    println!("  be imposed and read back. That is our own sequential fabric, which this crate does not");
    println!("  yet assemble -- `lib.rs` says so where a Device would otherwise be returned.");
}
