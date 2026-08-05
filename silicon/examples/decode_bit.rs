// Validate our configuration-packet layer against an OUTSIDE WITNESS: a real bitstream that a
// real device accepts. If our decoder walks the whole stream cleanly and the registers it
// reports match what a 7-series part expects, the encoder half is trustworthy too.
//
// usage: cargo run -p ferrotherm-silicon --example decode_bit -- <file.bit|file.bin>
use ferrotherm_silicon::bitstream::{decode, find_sync, parse_bit, reg, to_words, Packet};

fn reg_name(r: u32) -> &'static str {
    match r {
        reg::CRC => "CRC", reg::FAR => "FAR", reg::FDRI => "FDRI", reg::FDRO => "FDRO",
        reg::CMD => "CMD", reg::CTL0 => "CTL0", reg::MASK => "MASK", reg::STAT => "STAT",
        reg::LOUT => "LOUT", reg::COR0 => "COR0", reg::IDCODE => "IDCODE", reg::COR1 => "COR1",
        reg::WBSTAR => "WBSTAR", reg::TIMER => "TIMER", reg::BOOTSTS => "BOOTSTS",
        reg::CTL1 => "CTL1", _ => "?",
    }
}

fn main() {
    let path = std::env::args().nth(1).expect("usage: decode_bit <file>");
    let raw = std::fs::read(&path).expect("read");
    let bit = parse_bit(&raw);
    if !bit.part.is_empty() {
        println!("container: design={:?} part={:?} {} {}", bit.design, bit.part, bit.date, bit.time);
    }
    let sync = find_sync(bit.config).expect("no sync word — not a configuration stream");
    println!("{} bytes, sync at offset {}", bit.config.len(), sync - 4);
    let words = to_words(&bit.config[sync..]);
    let packets = decode(&words);

    let mut frame_words = 0usize;
    let mut idcode = None;
    let mut far_writes = 0;
    let mut crc_writes = 0;
    let mut nops = 0;
    let mut unknown = 0;
    for p in &packets {
        match p {
            Packet::Nop => nops += 1,
            Packet::Continue { words } => frame_words += words,
            Packet::Unknown(_) => unknown += 1,
            Packet::Read { .. } => {}
            Packet::Write { reg: r, data } => {
                match *r {
                    reg::IDCODE => idcode = data.first().copied(),
                    reg::FAR => far_writes += 1,
                    reg::FDRI => frame_words += data.len(),
                    reg::CRC => crc_writes += 1,
                    _ => {}
                }
                if data.len() <= 1 && !matches!(*r, reg::FDRI) {
                    println!("  {:>7} <= {}", reg_name(*r),
                             data.first().map(|d| format!("0x{d:08X}")).unwrap_or_default());
                }
            }
        }
    }
    println!("\npackets: {} ({} NOP, {} unknown)", packets.len(), nops, unknown);
    println!("FAR writes: {far_writes}   frame data words: {frame_words}   CRC writes: {crc_writes}");
    match idcode {
        Some(id) => println!("IDCODE stamped in stream: 0x{id:08X}"),
        None => println!("no IDCODE packet found"),
    }
    println!(
        "verdict: {}",
        if unknown == 0 && idcode.is_some() {
            "decoder walked the entire stream with no unknown packets — packet layer agrees with a device-accepted bitstream"
        } else {
            "stream did not decode cleanly — our packet layer disagrees with real hardware"
        }
    );
}
