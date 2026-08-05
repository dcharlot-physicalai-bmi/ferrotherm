// OUTSIDE-WITNESS CHECK on the generator: assemble a bitstream with our own code and compare its
// configuration header, word for word, against a bitstream a real device accepted. If our packet
// order or register values differ, this prints the first divergence.
//
// usage: cargo run -p ferrotherm-silicon --example header_compare -- <known-good.bit>
use ferrotherm_silicon::bitstream::{find_sync, parse_bit, to_words};
use ferrotherm_silicon::frame::assemble;
use ferrotherm_silicon::framebuf::FrameBuf;

fn main() {
    let path = std::env::args().nth(1).expect("usage: header_compare <file.bit>");
    let raw = std::fs::read(&path).expect("read");
    let bit = parse_bit(&raw);
    let sync = find_sync(bit.config).expect("sync");
    let theirs = to_words(&bit.config[sync..]);

    // one contiguous run so the header is directly comparable
    let mut fb = FrameBuf::new();
    for f in 0..256u32 {
        fb.frame_mut(f);
    }
    let ours_all = assemble(0x0363_1093, &fb);
    let ours_sync = ours_all.iter().position(|&w| w == 0xAA99_5566).expect("our sync") + 1;
    let ours = &ours_all[ours_sync..];

    // compare up to the first FDRI payload (headers only)
    let stop = theirs
        .iter()
        .position(|&w| w >> 29 == 2 || (w >> 13) & 0x3FFF == 0x02 && w >> 29 == 1)
        .unwrap_or(theirs.len().min(64))
        .min(ours.len());
    let mut diffs = 0;
    for i in 0..stop {
        if ours[i] != theirs[i] {
            if diffs < 8 {
                println!("  word {i:3}: ours 0x{:08X}  theirs 0x{:08X}", ours[i], theirs[i]);
            }
            diffs += 1;
        }
    }
    println!("compared {stop} header words after sync: {diffs} differences");
    println!(
        "verdict: {}",
        if diffs == 0 {
            "our generated header is word-identical to a stream real silicon accepted"
        } else {
            "header differs - our packet order or register values are not what the device saw"
        }
    );
}
