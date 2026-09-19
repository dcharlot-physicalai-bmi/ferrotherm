// `missing_docs` is denied workspace-wide and is right to be: it guards the API surface, and
// every public item in every library here carries a doc. An EXAMPLE has no API surface -- it is
// a program, and its helpers are private to it -- so the lint has nothing to guard and asks for
// doc comments on `fn main`'s scaffolding instead. Scoped off here rather than weakened there.
#![allow(missing_docs)]
// DOES GCAPTURE LATCH ON A BOARD? The device is the witness, and no placement is assumed.
//
// `examples/capture` builds the capture stream, names the bits, and then says what it cannot
// show: "that GCAPTURE latches on a board. That needs a part, a running design." This is that.
//
// The test needs no knowledge of whose design is running or where it put anything, because it is
// DIFFERENTIAL. Configuration memory is static; a running design's flip-flops are not. So, for
// every column of whatever is configured right now:
//
//   plain, plain          two ordinary readbacks MUST be identical -- it is the same memory
//   capture, wait, capture two captured readbacks should DIFFER wherever a counter lives,
//                          because GCAPTURE copied live flip-flop state and the state moved
//   capture, then plain    a plain readback AFTER a capture must equal that capture: the
//                          latched values persist in configuration memory until the next one
//
// The first and third are what rule out the dull explanations. If captures differed because the
// transport is flaky, plain readbacks would differ too. If they differed for any reason other
// than state being latched into configuration memory, the plain readback afterwards would not
// reproduce them.
//
// NON-DESTRUCTIVE TO THE DESIGN: nothing is configured, the design keeps running, and the only
// thing written is the command and control sequence `capture::capture_and_read` documents.
//
// run on the board host:
//   cargo run --release -p ferrotherm-silicon --features flash --example capture_hw

use ferrotherm_silicon::bitstream::reg;
use ferrotherm_silicon::flash::{Ftdi, Stat, Tap};
use ferrotherm_silicon::tilegrid::Far;

const FRAMES: usize = 36; // a CLB or interconnect column

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
        println!("no design is running (DONE low), so there is no flip-flop state to latch.");
        return;
    }

    println!();
    println!("  pass 1: two captures per column, 30 ms apart -- where does state MOVE?");
    let mut live: Vec<(u32, u32)> = Vec::new();
    let mut scanned = 0usize;
    for bottom_half in [false, true] {
        for row in 0..3u8 {
            for column in 0..96u16 {
                let far = Far { block_type: 0, bottom_half, row, column, minor: 0 }.encode();
                let Ok(first) = tap.capture_frames(far, FRAMES) else { continue };
                std::thread::sleep(std::time::Duration::from_millis(30));
                let Ok(second) = tap.capture_frames(far, FRAMES) else { continue };
                scanned += 1;
                let moved = differing_bits(&first, &second);
                if moved > 0 {
                    live.push((far, moved));
                }
            }
        }
    }
    live.sort_by_key(|&(_, moved)| core::cmp::Reverse(moved));
    println!("  scanned {scanned} columns; {} show flip-flop state moving between captures", live.len());
    if live.is_empty() {
        println!("  nothing moved. Either this design holds no free-running state, or GCAPTURE did not");
        println!("  latch. This run cannot tell those apart, and does not claim to.");
        return;
    }

    println!();
    println!("  pass 2: the three-way check on the liveliest columns");
    println!("  {:>10}  {:>13}  {:>15}  {:>21}", "FAR", "plain vs plain", "capture vs capture", "plain-after vs capture");
    let (mut ok, mut tried) = (0usize, 0usize);
    for &(far, _) in live.iter().take(8) {
        let p1 = tap.read_frames(far, FRAMES).expect("plain 1");
        let p2 = tap.read_frames(far, FRAMES).expect("plain 2");
        let c1 = tap.capture_frames(far, FRAMES).expect("capture 1");
        std::thread::sleep(std::time::Duration::from_millis(30));
        let c2 = tap.capture_frames(far, FRAMES).expect("capture 2");
        let after = tap.read_frames(far, FRAMES).expect("plain after");
        let (pp, cc, ac) = (differing_bits(&p1, &p2), differing_bits(&c1, &c2), differing_bits(&after, &c2));
        tried += 1;
        let pass = pp == 0 && cc > 0 && ac == 0;
        if pass {
            ok += 1;
        }
        println!("  0x{far:08X}  {pp:>13}  {cc:>15}  {ac:>21}   {}", if pass { "PASS" } else { "" });
    }
    println!();
    println!("  {ok} of {tried} columns pass all three: static memory reads back identically, captured");
    println!("  state moves, and what a capture latched is what a plain readback then finds.");
    if ok > 0 {
        println!("  GCAPTURE latches live flip-flop state into configuration memory on this part.");
    }
}
