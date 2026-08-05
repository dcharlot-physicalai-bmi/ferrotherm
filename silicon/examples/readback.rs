// Ask the silicon where the bits are: read configuration frames back from the device and compare
// them against a bitstream WE DID NOT GENERATE.
//
// This is the only version of this test that means anything. Comparing a readback against our own
// writes proves nothing — both would use the same frame arithmetic, so a wrong address would
// agree with itself. Checking against a foreign bitstream makes the device the witness.
//
// usage: readback <reference.bit> [n_frames]
use ferrotherm_silicon::bitstream::{decode, find_sync, parse_bit, reg, to_words, Packet};
use ferrotherm_silicon::flash::{Ftdi, Tap};

fn main() {
    let path = std::env::args().nth(1).expect("usage: readback <reference.bit> [n_frames]");
    let n: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(4);
    let raw = std::fs::read(&path).expect("read");
    let bit = parse_bit(&raw);
    let sync = find_sync(bit.config).expect("sync");
    let packets = decode(&to_words(&bit.config[sync..]));

    // the reference's frame payload and the address it was written to
    let mut far = None;
    let mut payload: Vec<u32> = Vec::new();
    let words = to_words(&bit.config[sync..]);
    let mut idx = 0usize;
    for p in &packets {
        match p {
            Packet::Write { reg: r, data } if *r == reg::FAR => far = data.first().copied(),
            Packet::Continue { words: n } => {
                let avail = words.len().saturating_sub(idx + 1);
                let take = (*n).min(avail);
                payload.extend_from_slice(&words[idx + 1..idx + 1 + take]);
            }
            _ => {}
        }
        idx += match p {
            Packet::Nop => 1,
            Packet::Write { data, .. } => 1 + data.len(),
            Packet::Read { .. } => 1,
            Packet::Continue { words } => 1 + words,
            Packet::Unknown(_) => 1,
        };
    }
    let far = far.unwrap_or(0);
    println!("reference: FAR 0x{far:08X}, {} payload words ({} frames)", payload.len(), payload.len() / 101);
    if payload.len() < n * 101 {
        println!("reference has fewer than {n} frames");
        return;
    }

    let (ftdi, product) = Ftdi::open("Alchitry").expect("no board");
    println!("board: {product}");
    let mut tap = Tap::new(ftdi).expect("tap");
    println!("reading {n} frames from FAR 0x{far:08X}...");
    let got = match tap.read_frames(far, n) {
        Ok(w) => w,
        Err(e) => {
            println!("readback failed: {e}");
            return;
        }
    };
    println!("read {} words", got.len());

    let mut matches = 0usize;
    let mut diffs = 0usize;
    for i in 0..(n * 101).min(got.len()).min(payload.len()) {
        if got[i] == payload[i] {
            matches += 1;
        } else {
            if diffs < 5 {
                println!("  word {i:4} (frame {}, w{}): device 0x{:08X}  reference 0x{:08X}",
                         i / 101, i % 101, got[i], payload[i]);
            }
            diffs += 1;
        }
    }
    let total = matches + diffs;
    println!("\n{matches}/{total} words match ({:.1}%)", 100.0 * matches as f64 / total.max(1) as f64);
    println!(
        "verdict: {}",
        if total > 0 && matches * 100 / total >= 95 {
            "THE DEVICE AGREES - frames read back from silicon match a bitstream we did not \
             generate, so our frame addressing is right in the device's own terms"
        } else {
            "mismatch - our frame addressing disagrees with the device"
        }
    );
}
