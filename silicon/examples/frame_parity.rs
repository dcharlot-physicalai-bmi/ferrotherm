#![allow(missing_docs)]
// Survey any bitstream's frames for the parity every vendor frame carries.
//
//   cargo run -p ferrotherm-silicon --example frame_parity -- <file>...
//
// Takes 7-series `.bit`/`.bin` (big-endian, one file header) and UltraScale+ `.bit.bin` (a
// little-endian boot image whose first sync word is a decoy). Which it is, it works out from
// the file: both geometries are tried and the one whose frame count comes out whole wins.
//
// This is the check that showed the library was writing frames no vendor tool would: 20 of the
// 40 frames in `lab_fabric.bit` had odd parity, against 0 of 21,680 in four Vivado designs.
use ferrotherm_silicon::bitstream::{find_sync, to_words};
use ferrotherm_silicon::ecc::{survey, Survey};
use ferrotherm_silicon::usplus;

/// One reading of a file: the geometry tried and what it found.
struct Reading {
    family: &'static str,
    words_per_frame: usize,
    survey: Survey,
}

/// The frame-data words of a big-endian 7-series stream.
fn seven_series(bytes: &[u8]) -> Vec<u32> {
    let Some(at) = find_sync(bytes) else { return Vec::new() };
    frame_data(&to_words(&bytes[at..]))
}

/// The frame-data words of a little-endian `UltraScale+` boot image.
fn ultrascale(bytes: &[u8]) -> Vec<u32> {
    let w = usplus::words(bytes);
    let Some(at) = usplus::config_start(&w) else { return Vec::new() };
    frame_data(&w[at..])
}

/// Walk the packet stream and collect everything written to the frame-data-in register.
///
/// Written out here rather than through `bitstream::decode` because a frame-data burst is the
/// one packet whose payload must stay in stream order across a type-2 continuation.
fn frame_data(w: &[u32]) -> Vec<u32> {
    const FDRI: u32 = 0x02;
    let mut out = Vec::new();
    let mut p = 1usize;
    while p < w.len() {
        let h = w[p];
        p += 1;
        match h >> 29 {
            1 => {
                let reg = (h >> 13) & 0x1F;
                let count = (h & 0x7FF) as usize;
                let end = (p + count).min(w.len());
                if (h >> 27) & 3 == 2 && reg == FDRI {
                    out.extend_from_slice(&w[p..end]);
                }
                p = end;
            }
            2 => {
                let count = (h & 0x07FF_FFFF) as usize;
                let end = (p + count).min(w.len());
                out.extend_from_slice(&w[p..end]);
                p = end;
            }
            _ => {}
        }
    }
    out
}

fn readings(bytes: &[u8]) -> Vec<Reading> {
    let mut out = Vec::new();
    let x7 = seven_series(bytes);
    if !x7.is_empty() {
        out.push(Reading {
            family: "7-series",
            words_per_frame: ferrotherm_silicon::frame::WORDS_PER_FRAME,
            survey: survey(&x7, ferrotherm_silicon::frame::WORDS_PER_FRAME),
        });
    }
    let usp = ultrascale(bytes);
    if !usp.is_empty() {
        out.push(Reading {
            family: "UltraScale+",
            words_per_frame: usplus::WORDS_PER_FRAME,
            survey: survey(&usp, usplus::WORDS_PER_FRAME),
        });
    }
    out
}

fn main() {
    let files: Vec<String> = std::env::args().skip(1).collect();
    if files.is_empty() {
        println!("usage: frame_parity <file>...");
        println!();
        println!("  a 7-series .bit, or an UltraScale+ .bit.bin the FPGA manager loads");
        return;
    }
    for path in &files {
        println!("{path}");
        let Ok(bytes) = std::fs::read(path) else {
            println!("  cannot read it");
            continue;
        };
        let found = readings(&bytes);
        if found.is_empty() {
            println!("  no configuration stream: no sync word a no-op follows");
            println!("  (an UltraScale+ boot image carries a decoy sync word in its header)");
            continue;
        }
        for r in &found {
            let s = r.survey;
            let fit = if s.remainder == 0 { "" } else { " <- does not divide, wrong family" };
            println!(
                "  {:<12} {:>3} words/frame: {} frames, {} occupied, {} odd, {} words left over{fit}",
                r.family, r.words_per_frame, s.frames, s.occupied, s.odd, s.remainder
            );
        }
        match found.iter().find(|r| r.survey.remainder == 0) {
            Some(r) if r.survey.vendor_shaped() => {
                println!("  VENDOR-SHAPED: every frame carries its parity");
            }
            Some(r) => println!(
                "  {} of {} frames have the parity no vendor frame has",
                r.survey.odd, r.survey.frames
            ),
            None => println!("  no geometry divided this stream evenly"),
        }
    }
}
