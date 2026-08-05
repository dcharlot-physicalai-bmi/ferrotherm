//! Configuration-frame assembly and the bitstream CRC.
//!
//! Everything a generated bitstream needs below the fabric database: the CRC the device checks
//! before it will start, and the packet sequence that writes configuration frames. Offline
//! tests only — the CRC is anchored to the published CRC-32C test vector, so the implementation
//! is verified against a value nobody in this project chose.

use crate::bitstream::{cmd, reg, type1_write, type2_write, DUMMY, NOOP, SYNC};

/// 7-series frame geometry.
pub const WORDS_PER_FRAME: usize = 101;

/// CRC-32C (Castagnoli, reflected, poly 0x1EDC6F41 -> reflected 0x82F63B78) — the function the
/// 7-series configuration CRC is built from.
pub fn crc32c(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &b in data {
        crc ^= b as u32;
        for _ in 0..8 {
            crc = if crc & 1 != 0 { (crc >> 1) ^ 0x82F6_3B78 } else { crc >> 1 };
        }
    }
    !crc
}

/// The running configuration CRC: every write of `data` to register `addr` folds a 37-bit
/// quantity (5-bit address, then 32-bit data, LSB first) into the register.
#[derive(Default, Clone, Copy)]
pub struct ConfigCrc {
    pub value: u32,
}

impl ConfigCrc {
    pub fn update(&mut self, addr: u32, data: u32) {
        // shift the 32 data bits then the 5 address bits through the CRC-32C polynomial
        for i in 0..37 {
            let bit = if i < 32 { (data >> i) & 1 } else { (addr >> (i - 32)) & 1 };
            let top = (self.value ^ bit) & 1;
            self.value >>= 1;
            if top != 0 {
                self.value ^= 0x82F6_3B78;
            }
        }
    }
    pub fn reset(&mut self) {
        self.value = 0;
    }
}

/// Assemble the standard 7-series configuration preamble: dummy words, sync, and the header
/// packets that set the device IDCODE and clear the CRC.
pub fn preamble(idcode: u32) -> Vec<u32> {
    vec![
        DUMMY, DUMMY,
        SYNC,
        NOOP,
        type1_write(reg::CMD, 1), cmd::RCRC,
        NOOP, NOOP,
        type1_write(reg::IDCODE, 1), idcode,
        type1_write(reg::CMD, 1), cmd::WCFG,
        NOOP,
    ]
}

/// Frames written at `far` as one FDRI burst: sets the frame address, then streams the data.
/// A type-1 FDRI write carries at most 2047 words, so longer bursts use a type-2 continuation.
pub fn frame_write(far: u32, frames: &[u32]) -> Vec<u32> {
    let mut v = vec![type1_write(reg::FAR, 1), far];
    if frames.len() <= 0x7FF {
        v.push(type1_write(reg::FDRI, frames.len() as u32));
    } else {
        v.push(type1_write(reg::FDRI, 0));
        v.push(type2_write(frames.len() as u32));
    }
    v.extend_from_slice(frames);
    v
}

/// The closing sequence: CRC check, startup, desync.
pub fn epilogue(crc: u32) -> Vec<u32> {
    vec![
        type1_write(reg::CRC, 1), crc,
        type1_write(reg::CMD, 1), cmd::START,
        NOOP,
        type1_write(reg::CMD, 1), cmd::DESYNC,
        NOOP, NOOP,
    ]
}

/// Words -> big-endian bytes, the order the configuration port expects.
pub fn words_to_bytes(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|w| w.to_be_bytes()).collect()
}


/// Bus-width auto-detect pattern that precedes the sync word.
pub const BUS_WIDTH_SYNC: u32 = 0x0000_00BB;
pub const BUS_WIDTH_DETECT: u32 = 0x1122_0044;

/// Assemble a complete 7-series configuration stream for a sparse set of frames.
///
/// The packet order reproduces a stream this device family actually accepts (decoded from a
/// known-good XC7A100T bitstream), rather than an order invented from the register list:
/// bus-width detect, sync, RCRC, TIMER/WBSTAR/COR0/COR1/IDCODE, switch, MASK/CTL0/CTL1, then per
/// contiguous frame run a FAR write and an FDRI burst, then GRESTORE/LFRM/START/RCRC/DESYNC.
/// No CRC value is written — RCRC resets the register and the device does not compare.
pub fn assemble(idcode: u32, buf: &crate::framebuf::FrameBuf) -> Vec<u32> {
    let mut w = Vec::new();
    w.extend_from_slice(&[DUMMY; 8]);
    w.push(BUS_WIDTH_SYNC);
    w.push(BUS_WIDTH_DETECT);
    w.extend_from_slice(&[DUMMY; 4]);
    w.push(SYNC);
    w.push(NOOP);
    w.extend_from_slice(&[type1_write(reg::CMD, 1), cmd::RCRC]);
    w.extend_from_slice(&[NOOP, NOOP]);
    w.extend_from_slice(&[type1_write(reg::TIMER, 1), 0]);
    w.extend_from_slice(&[type1_write(reg::WBSTAR, 1), 0]);
    w.extend_from_slice(&[type1_write(reg::CMD, 1), cmd::NULL]);
    w.extend_from_slice(&[type1_write(reg::COR0, 1), 0x0200_3FE5]);
    w.extend_from_slice(&[type1_write(reg::COR1, 1), 0]);
    w.extend_from_slice(&[type1_write(reg::IDCODE, 1), idcode]);
    w.extend_from_slice(&[type1_write(reg::CMD, 1), cmd::SWITCH]);
    // TWO noops here, not one — read off a stream the device accepted rather than guessed.
    w.extend_from_slice(&[NOOP, NOOP]);
    w.extend_from_slice(&[type1_write(reg::MASK, 1), 0x0000_0401]);
    w.extend_from_slice(&[type1_write(reg::CTL0, 1), 0x0000_0501]);
    w.extend_from_slice(&[type1_write(reg::MASK, 1), 0]);
    w.extend_from_slice(&[type1_write(reg::CTL1, 1), 0]);

    for (start, words) in buf.runs() {
        w.extend_from_slice(&[type1_write(reg::FAR, 1), start]);
        w.extend_from_slice(&[type1_write(reg::CMD, 1), cmd::WCFG]);
        w.push(NOOP);
        // (FDRI follows immediately; the witness stream has exactly one NOOP here)
        w.extend_from_slice(&frame_write_payload(&words));
    }

    w.extend_from_slice(&[type1_write(reg::CMD, 1), cmd::GRESTORE]);
    w.push(NOOP);
    w.extend_from_slice(&[type1_write(reg::CMD, 1), cmd::LFRM]);
    w.extend_from_slice(&[NOOP; 100]);
    w.extend_from_slice(&[type1_write(reg::CMD, 1), cmd::START]);
    w.push(NOOP);
    w.extend_from_slice(&[type1_write(reg::CMD, 1), cmd::RCRC]);
    w.extend_from_slice(&[NOOP, NOOP]);
    w.extend_from_slice(&[type1_write(reg::CMD, 1), cmd::DESYNC]);
    w.extend_from_slice(&[NOOP; 16]);
    w
}

/// The FDRI packet for a burst of frame words (type-2 continuation when it exceeds a type-1).
fn frame_write_payload(words: &[u32]) -> Vec<u32> {
    let mut v = Vec::with_capacity(words.len() + 2);
    if words.len() <= 0x7FF {
        v.push(type1_write(reg::FDRI, words.len() as u32));
    } else {
        v.push(type1_write(reg::FDRI, 0));
        v.push(type2_write(words.len() as u32));
    }
    v.extend_from_slice(words);
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The published CRC-32C check value: CRC32C("123456789") = 0xE3069283. An independent
    /// anchor — if the polynomial or reflection were wrong, this would not land.

    /// THE WITNESS TEST: the 34 configuration words after the sync word, as they appear in a
    /// bitstream real XC7A100T silicon accepted. These are not values we chose — they were read
    /// off a working stream, and our assembler must reproduce them exactly. If a future change
    /// reorders a packet or drops a NOOP, this fails before anything reaches a board.
    #[test]
    fn header_matches_the_accepted_stream() {
        const WITNESS: [u32; 34] = [
            NOOP,
            0x3000_8001, cmd::RCRC,      // CMD <= RCRC
            NOOP, NOOP,
            0x3002_2001, 0x0000_0000,    // TIMER  <= 0
            0x3002_0001, 0x0000_0000,    // WBSTAR <= 0
            0x3000_8001, cmd::NULL,      // CMD <= NULL
            0x3001_2001, 0x0200_3FE5,    // COR0
            0x3001_C001, 0x0000_0000,    // COR1
            0x3001_8001, 0x0363_1093,    // IDCODE (XC7A100T)
            0x3000_8001, cmd::SWITCH,    // CMD <= SWITCH
            NOOP, NOOP,
            0x3000_C001, 0x0000_0401,    // MASK
            0x3000_A001, 0x0000_0501,    // CTL0
            0x3000_C001, 0x0000_0000,    // MASK
            0x3003_0001, 0x0000_0000,    // CTL1
            0x3000_2001, 0x0000_0000,    // FAR <= 0
            0x3000_8001, cmd::WCFG,      // CMD <= WCFG
            NOOP,
        ];
        let mut fb = crate::framebuf::FrameBuf::new();
        for f in 0..4u32 {
            fb.frame_mut(f);
        }
        let words = assemble(0x0363_1093, &fb);
        let sync_at = words.iter().position(|&w| w == SYNC).expect("sync") + 1;
        assert_eq!(
            &words[sync_at..sync_at + WITNESS.len()],
            &WITNESS[..],
            "generated header diverged from the stream silicon accepted"
        );
        // and the register encodings the witness implies
        assert_eq!(type1_write(reg::TIMER, 1), 0x3002_2001);
        assert_eq!(type1_write(reg::CTL1, 1), 0x3003_0001);
    }

    /// Frames stream as one FDRI burst per contiguous run, with the count matching the words.
    #[test]
    fn frame_runs_become_fdri_bursts() {
        let mut fb = crate::framebuf::FrameBuf::new();
        for f in [0u32, 1, 2, 100] {
            fb.frame_mut(f);
        }
        let words = assemble(0x0363_1093, &fb);
        let far_writes = words
            .windows(2)
            .filter(|w| w[0] == type1_write(reg::FAR, 1))
            .count();
        assert_eq!(far_writes, 2, "two runs -> two FAR writes");
        let total_frame_words: usize = fb.runs().iter().map(|(_, w)| w.len()).sum();
        assert_eq!(total_frame_words, 4 * WORDS_PER_FRAME);
    }

    #[test]
    fn crc32c_standard_vector() {
        assert_eq!(crc32c(b"123456789"), 0xE306_9283);
    }

    #[test]
    fn config_crc_accumulates_and_resets() {
        let mut c = ConfigCrc::default();
        c.update(reg::IDCODE, 0x1363_1093);
        let after_one = c.value;
        assert_ne!(after_one, 0);
        c.update(reg::CMD, cmd::WCFG);
        assert_ne!(c.value, after_one);
        // order matters: the same pair applied in the other order gives a different register
        let mut d = ConfigCrc::default();
        d.update(reg::CMD, cmd::WCFG);
        d.update(reg::IDCODE, 0x1363_1093);
        assert_ne!(c.value, d.value);
        c.reset();
        assert_eq!(c.value, 0);
    }

    #[test]
    fn preamble_stamps_the_idcode() {
        let p = preamble(0x1363_1093);
        assert_eq!(p[2], SYNC);
        let idx = p.iter().position(|&w| w == 0x1363_1093).expect("idcode present");
        assert_eq!(p[idx - 1], type1_write(reg::IDCODE, 1));
    }

    /// A burst longer than a type-1 packet can address must switch to a type-2 continuation.
    #[test]
    fn long_bursts_use_type2() {
        let short = frame_write(0, &vec![0u32; WORDS_PER_FRAME]);
        assert_eq!(short[2], type1_write(reg::FDRI, WORDS_PER_FRAME as u32));
        let long = frame_write(0, &vec![0u32; 5000]);
        assert_eq!(long[2], type1_write(reg::FDRI, 0));
        assert_eq!(long[3], type2_write(5000));
        assert_eq!(long.len(), 4 + 5000);
    }
}
