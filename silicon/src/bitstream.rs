//! 7-series bitstream containers and configuration packets (UG470).
//!
//! Two jobs: strip a Xilinx `.bit` container down to its raw configuration payload, and build
//! the type-1/type-2 packets the configuration port speaks. Both are pure data handling with
//! offline tests — nothing here touches hardware.

/// Configuration register addresses (UG470 Table 5-23).
pub mod reg {
    pub const CRC: u32 = 0x00;
    pub const FAR: u32 = 0x01;
    pub const FDRI: u32 = 0x02;
    pub const FDRO: u32 = 0x03;
    pub const CMD: u32 = 0x04;
    pub const CTL0: u32 = 0x05;
    pub const MASK: u32 = 0x06;
    pub const STAT: u32 = 0x07;
    pub const LOUT: u32 = 0x08;
    pub const COR0: u32 = 0x09;
    pub const IDCODE: u32 = 0x0C;
    pub const COR1: u32 = 0x0E;
    pub const WBSTAR: u32 = 0x10;
    pub const TIMER: u32 = 0x11;
    pub const BOOTSTS: u32 = 0x16;
    pub const CTL1: u32 = 0x18;
}

/// CMD register opcodes (UG470 Table 5-24).
pub mod cmd {
    pub const NULL: u32 = 0x00;
    pub const WCFG: u32 = 0x01;
    pub const LFRM: u32 = 0x03;
    pub const RCFG: u32 = 0x04;
    pub const START: u32 = 0x05;
    pub const RCAP: u32 = 0x06;
    pub const GRESTORE: u32 = 0x0A;
    pub const SWITCH: u32 = 0x09;
    pub const RCRC: u32 = 0x07;
    pub const DESYNC: u32 = 0x0D;
    pub const IPROG: u32 = 0x0F;
}

pub const DUMMY: u32 = 0xFFFF_FFFF;
pub const SYNC: u32 = 0xAA99_5566;
pub const NOOP: u32 = 0x2000_0000;

/// Type-1 packet header: read `count` words from `reg`.
pub fn type1_read(reg: u32, count: u32) -> u32 {
    0x2800_0000 | ((reg & 0x3FFF) << 13) | (count & 0x7FF)
}

/// Type-1 packet header: write `count` words to `reg`.
pub fn type1_write(reg: u32, count: u32) -> u32 {
    0x3000_0000 | ((reg & 0x3FFF) << 13) | (count & 0x7FF)
}

/// Type-2 packet header: continue the previous register with a long word count.
pub fn type2_write(count: u32) -> u32 {
    0x5000_0000 | (count & 0x07FF_FFFF)
}

/// The payload of a Xilinx `.bit` container, plus whatever metadata the header carried.
#[derive(Debug, Clone)]
pub struct BitFile<'a> {
    pub design: String,
    pub part: String,
    pub date: String,
    pub time: String,
    pub config: &'a [u8],
}

fn be16(d: &[u8], p: usize) -> usize {
    ((d[p] as usize) << 8) | d[p + 1] as usize
}

/// Parse a `.bit` container. Falls back to treating the whole buffer as raw configuration data
/// (a `.bin`) when no recognizable header is present.
pub fn parse_bit(data: &[u8]) -> BitFile<'_> {
    let mut out = BitFile {
        design: String::new(),
        part: String::new(),
        date: String::new(),
        time: String::new(),
        config: data,
    };
    if data.len() < 4 {
        return out;
    }
    // field 0: 2-byte length + payload, then a 2-byte 0x0001 marker
    let mut p = 2 + be16(data, 0) + 2;
    while p + 3 <= data.len() {
        let tag = data[p];
        p += 1;
        if tag == b'e' {
            if p + 4 > data.len() {
                return out;
            }
            let len = u32::from_be_bytes([data[p], data[p + 1], data[p + 2], data[p + 3]]) as usize;
            p += 4;
            out.config = &data[p..(p + len).min(data.len())];
            return out;
        }
        if !matches!(tag, b'a'..=b'd') || p + 2 > data.len() {
            return out; // unrecognized header -> assume already-raw config
        }
        let len = be16(data, p);
        p += 2;
        let s = String::from_utf8_lossy(&data[p..(p + len).min(data.len())])
            .trim_end_matches('\0')
            .to_string();
        match tag {
            b'a' => out.design = s,
            b'b' => out.part = s,
            b'c' => out.date = s,
            _ => out.time = s,
        }
        p += len;
    }
    out
}

/// Locate the sync word in a configuration payload, returning the offset of the word AFTER it.
pub fn find_sync(config: &[u8]) -> Option<usize> {
    config
        .windows(4)
        .position(|w| w == SYNC.to_be_bytes())
        .map(|i| i + 4)
}


/// A decoded configuration packet.
#[derive(Debug, Clone, PartialEq)]
pub enum Packet {
    Nop,
    Write { reg: u32, data: Vec<u32> },
    Read { reg: u32, count: u32 },
    /// Type-2 continuation of the previous register.
    Continue { words: usize },
    Unknown(u32),
}

/// Walk a configuration word stream from the sync word onward. Returns the packets in order;
/// stops at the end of the buffer. Used to validate our encoders against real bitstreams.
pub fn decode(words: &[u32]) -> Vec<Packet> {
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < words.len() {
        let w = words[i];
        i += 1;
        if w == NOOP {
            out.push(Packet::Nop);
            continue;
        }
        let typ = w >> 29;
        match typ {
            1 => {
                let op = (w >> 27) & 0x3;
                let reg = (w >> 13) & 0x3FFF;
                let count = (w & 0x7FF) as usize;
                match op {
                    1 => out.push(Packet::Read { reg, count: count as u32 }),
                    2 => {
                        let end = (i + count).min(words.len());
                        out.push(Packet::Write { reg, data: words[i..end].to_vec() });
                        i = end;
                    }
                    _ => out.push(Packet::Unknown(w)),
                }
            }
            2 => {
                let count = (w & 0x07FF_FFFF) as usize;
                let end = (i + count).min(words.len());
                out.push(Packet::Continue { words: end - i });
                i = end;
            }
            _ => out.push(Packet::Unknown(w)),
        }
    }
    out
}

/// Read a big-endian word stream out of raw configuration bytes.
pub fn to_words(config: &[u8]) -> Vec<u32> {
    config
        .chunks_exact(4)
        .map(|c| u32::from_be_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_bit(design: &str, part: &str, payload: &[u8]) -> Vec<u8> {
        let mut v = Vec::new();
        v.extend_from_slice(&[0x00, 0x09]);
        v.extend_from_slice(&[0x0F, 0xF0, 0x0F, 0xF0, 0x0F, 0xF0, 0x0F, 0xF0, 0x00]);
        v.extend_from_slice(&[0x00, 0x01]);
        for (tag, s) in [(b'a', design), (b'b', part), (b'c', "2026/08/05"), (b'd', "12:00:00")] {
            let mut b = s.as_bytes().to_vec();
            b.push(0);
            v.push(tag);
            v.extend_from_slice(&(b.len() as u16).to_be_bytes());
            v.extend_from_slice(&b);
        }
        v.push(b'e');
        v.extend_from_slice(&(payload.len() as u32).to_be_bytes());
        v.extend_from_slice(payload);
        v
    }

    #[test]
    fn bit_container_roundtrip() {
        let payload: Vec<u8> = [DUMMY, SYNC, NOOP].iter().flat_map(|w| w.to_be_bytes()).collect();
        let file = synthetic_bit("fabric;UserID=0X0", "7a100tfgg484", &payload);
        let parsed = parse_bit(&file);
        assert_eq!(parsed.design, "fabric;UserID=0X0");
        assert_eq!(parsed.part, "7a100tfgg484");
        assert_eq!(parsed.config, &payload[..]);
        assert_eq!(find_sync(parsed.config), Some(8));
    }

    #[test]
    fn raw_bin_passes_through() {
        // A raw payload with no container must be returned untouched rather than mangled.
        let raw: Vec<u8> = [DUMMY, DUMMY, SYNC].iter().flat_map(|w| w.to_be_bytes()).collect();
        let parsed = parse_bit(&raw);
        assert_eq!(parsed.config, &raw[..]);
    }

    // Packet encodings against the values documented in UG470.
    /// Our own encoders must decode back to themselves — and the decoder must agree with the
    /// packet layout the device actually accepts (see examples/decode_bit.rs, run against a
    /// real generated bitstream).
    #[test]
    fn encode_decode_roundtrip() {
        let mut words = vec![DUMMY, SYNC];
        words.extend_from_slice(&[type1_write(reg::CMD, 1), cmd::RCRC]);
        words.push(NOOP);
        words.extend_from_slice(&[type1_write(reg::IDCODE, 1), 0x0363_1093]);
        words.extend_from_slice(&[type1_read(reg::STAT, 1)]);
        let sync_at = find_sync(&crate::frame::words_to_bytes(&words)).unwrap();
        let decoded = decode(&to_words(&crate::frame::words_to_bytes(&words))[sync_at / 4..]);
        assert_eq!(decoded[0], Packet::Write { reg: reg::CMD, data: vec![cmd::RCRC] });
        assert_eq!(decoded[1], Packet::Nop);
        assert_eq!(decoded[2], Packet::Write { reg: reg::IDCODE, data: vec![0x0363_1093] });
        assert_eq!(decoded[3], Packet::Read { reg: reg::STAT, count: 1 });
    }

    #[test]
    fn packet_headers() {
        assert_eq!(type1_read(reg::STAT, 1), 0x2800_E001);
        assert_eq!(type1_read(reg::IDCODE, 1), 0x2801_8001);
        assert_eq!(type1_write(reg::CMD, 1), 0x3000_8001);
        assert_eq!(type1_write(reg::FAR, 1), 0x3000_2001);
        assert_eq!(type2_write(0x1234), 0x5000_1234);
    }
}
