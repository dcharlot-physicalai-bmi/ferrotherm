#![allow(missing_docs)]
// How a fabric with no output pins is read: the capture stream, the pad frame, and the file that
// turns 3,232 anonymous bits into named flip-flops.
//
//   cargo run -p ferrotherm-silicon --example capture [design.ll]
//
// Runs with no arguments. With a Vivado logic-location file it uses that listing's real cells
// instead of the built-in one.
//
// Stage 4 is a ROUND TRIP THROUGH OUR OWN ARITHMETIC, and says so where it prints. It shows that
// the naming path recovers what was put in; it is not evidence about a device, because both ends
// use the same addressing and would agree wherever that addressing is wrong. The witness for the
// addressing is the logic-location file itself -- see `logic_location::cross_check`.
use ferrotherm_silicon::bitstream::{cmd, reg, type1_read, type1_write, type2_read, SYNC};
use ferrotherm_silicon::capture::{
    capture_and_read, readback_words, Readout, GLUTMASK, PAD_FRAMES, STREAM_CTL0, STREAM_MASK,
};
use ferrotherm_silicon::frame::WORDS_PER_FRAME;
use ferrotherm_silicon::logic_location::{parse_ll, LlBit};

/// The frame the built-in listing places its cells in, and the one after it.
const BASE: u32 = 0x0042_0100;

/// A listing in the shape Vivado writes, for when no real one is given.
const BUILT_IN: &str = "\
Bit 1 0x00420100 0 SLR0 0 Block=SLICE_X0Y0 Latch=AQ Net=spin[0]
Bit 2 0x00420100 33 SLR0 0 Block=SLICE_X0Y0 Latch=BQ Net=spin[1]
Bit 3 0x00420100 70 SLR0 0 Block=SLICE_X0Y0 Latch=CQ Net=spin[2]
Bit 4 0x00420101 5 SLR0 0 Block=SLICE_X0Y1 Latch=AQ Net=spin[3]
Bit 5 0x00420101 96 SLR0 0 Block=SLICE_X0Y1 Latch=BQ Net=spin[4]
";

fn name_of(w: u32) -> &'static str {
    match w {
        0xFFFF_FFFF => "dummy, flushes the pipeline",
        SYNC => "sync",
        0x2000_0000 => "no-op",
        _ => "",
    }
}

fn describe(w: u32, prev: u32) -> String {
    if prev == type1_write(reg::CMD, 1) {
        return match w {
            cmd::GCAPTURE => "  <- GCAPTURE: latch every flip-flop into configuration memory".into(),
            cmd::RCFG => "  <- RCFG: the next FDRO read streams frames out".into(),
            _ => format!("  <- command 0x{w:02X}"),
        };
    }
    if prev == type1_write(reg::FAR, 1) {
        return format!("  <- frame address 0x{w:08X}");
    }
    if w == type1_write(reg::CMD, 1) {
        return "  write 1 word to CMD".into();
    }
    if w == type1_write(reg::FAR, 1) {
        return "  write 1 word to FAR".into();
    }
    if w == type1_read(reg::FDRO, 0) {
        return "  read FDRO, count 0 -- the length follows in a type-2".into();
    }
    let n = name_of(w);
    if n.is_empty() { String::new() } else { format!("  {n}") }
}

fn main() {
    let cells: Vec<LlBit> = match std::env::args().nth(1) {
        Some(path) => match std::fs::read_to_string(&path) {
            Ok(text) => {
                let c = parse_ll(&text);
                println!("listing: {path} -- {} located cells\n", c.len());
                c
            }
            Err(e) => {
                println!("cannot read {path}: {e}");
                return;
            }
        },
        None => parse_ll(BUILT_IN),
    };
    if cells.is_empty() {
        println!("the listing located no cells; nothing to read back");
        return;
    }

    // ---- 1. the stream ---------------------------------------------------------------------
    let frames = 2usize;
    println!("== 1. what goes in ==");
    let stream = capture_and_read(BASE, frames);
    let mut prev = 0u32;
    for w in &stream {
        println!("  0x{w:08X}{}", describe(*w, prev));
        prev = *w;
    }

    // ---- 2. what comes back ----------------------------------------------------------------
    println!("\n== 2. what comes back ==");
    let words = readback_words(frames);
    println!("  {frames} frames asked for, {words} words returned");
    println!(
        "  {PAD_FRAMES} of those frames ({} words) is pipeline contents and belongs to no address",
        PAD_FRAMES * WORDS_PER_FRAME
    );
    println!("  take it at face value and every frame reads one address late, with real bits");
    assert!(stream.contains(&type2_read(words as u32)));

    // ---- 3. the control bit that does not arrive -------------------------------------------
    println!("\n== 3. the write that does nothing ==");
    println!("  the generated stream writes CTL0 = 0x{STREAM_CTL0:08X} under MASK = 0x{STREAM_MASK:08X}");
    println!(
        "  bit 8 (0x{GLUTMASK:08X}) is set in the value and absent from the mask, so it never lands"
    );
    println!("  reading lookup-table RAM back needs it; capturing flip-flops does not");

    // ---- 4. naming the bits ----------------------------------------------------------------
    println!("\n== 4. naming the bits (round trip through our own arithmetic) ==");
    let mut buffer = vec![0u32; readback_words(frames)];
    for word in &mut buffer[..WORDS_PER_FRAME] {
        *word = 0xA5A5_A5A5; // a pad frame that is not zero, so mistaking it is visible
    }
    // put a known pattern where the listing says each cell lives
    let mut expected = Vec::new();
    for (i, cell) in cells.iter().enumerate() {
        let on = i % 2 == 0;
        expected.push(on);
        if !on {
            continue;
        }
        let Some(frame) = cell.frame_address.checked_sub(BASE) else { continue };
        if frame as usize >= frames {
            continue;
        }
        let at = (PAD_FRAMES + frame as usize) * WORDS_PER_FRAME;
        buffer[at + (cell.frame_offset / 32) as usize] |= 1 << (cell.frame_offset % 32);
    }
    match Readout::from_words(BASE, &buffer, frames) {
        Ok(readout) => {
            let sample = readout.sample(&cells);
            for o in &sample.values {
                println!("  {}.{:<3} = {}", o.block, o.latch, u8::from(o.value));
            }
            println!(
                "  {} cells observed, {} outside the run, {} reading high",
                sample.values.len(),
                sample.uncovered,
                sample.ones()
            );
            if !sample.observed() {
                println!("  NOTHING WAS OBSERVED -- a sample with no values is not a clean result");
            }
        }
        Err(e) => println!("  {e}"),
    }

    println!("\n== what this does and does not show ==");
    println!("  shown:     the packet order, the pad-frame arithmetic, and that a located cell");
    println!("             round-trips through the frame addressing.");
    println!("  not shown: that GCAPTURE latches on a board. That needs a part, a running");
    println!("             design, and a value known by other means.");
}
