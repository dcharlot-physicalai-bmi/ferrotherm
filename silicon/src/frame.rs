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

#[cfg(test)]
mod tests {
    use super::*;

    /// The published CRC-32C check value: CRC32C("123456789") = 0xE3069283. An independent
    /// anchor — if the polynomial or reflection were wrong, this would not land.
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
