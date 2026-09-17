//! 7-series bitstream containers and configuration packets (UG470).
//!
//! Two jobs: strip a Xilinx `.bit` container down to its raw configuration payload, and build
//! the type-1/type-2 packets the configuration port speaks. Both are pure data handling with
//! offline tests — nothing here touches hardware.

/// Configuration register addresses (UG470 Table 5-23).
pub mod reg {
    /// CRC check.
    pub const CRC: u32 = 0x00;
    /// Frame Address Register.
    pub const FAR: u32 = 0x01;
    /// Frame Data Register, input.
    pub const FDRI: u32 = 0x02;
    /// Frame Data Register, output.
    pub const FDRO: u32 = 0x03;
    /// Command register.
    pub const CMD: u32 = 0x04;
    /// Control register 0.
    pub const CTL0: u32 = 0x05;
    /// Mask for CTL0/CTL1 writes.
    pub const MASK: u32 = 0x06;
    /// Status register.
    pub const STAT: u32 = 0x07;
    /// Legacy output for daisy chains.
    pub const LOUT: u32 = 0x08;
    /// Configuration Option Register 0.
    pub const COR0: u32 = 0x09;
    /// Device ID, checked against the target part.
    pub const IDCODE: u32 = 0x0C;
    /// Configuration Option Register 1.
    pub const COR1: u32 = 0x0E;
    /// Warm Boot Start Address.
    pub const WBSTAR: u32 = 0x10;
    /// Watchdog timer.
    pub const TIMER: u32 = 0x11;
    /// Boot history.
    pub const BOOTSTS: u32 = 0x16;
    /// Control register 1.
    pub const CTL1: u32 = 0x18;
}

/// CMD register opcodes (UG470 Table 5-24).
pub mod cmd {
    /// No operation.
    pub const NULL: u32 = 0x00;
    /// Write configuration data.
    pub const WCFG: u32 = 0x01;
    /// Last frame.
    pub const LFRM: u32 = 0x03;
    /// Read configuration data.
    pub const RCFG: u32 = 0x04;
    /// Begin the startup sequence.
    pub const START: u32 = 0x05;
    /// Reset the CAPTURE signal.
    pub const RCAP: u32 = 0x06;
    /// Pulse GCAPTURE: latch the fabric's live flip-flop state into configuration memory so a
    /// frame readback returns what the design is doing rather than what was configured.
    ///
    /// This opcode comes from UG470's command table, not from a stream this project has watched a
    /// device accept, and it is the only constant here in that position. Everything the
    /// [`crate::capture`] module builds around it is checked offline; the latch itself is not yet
    /// exercised on silicon.
    pub const GCAPTURE: u32 = 0x0C;
    /// Pulse GRESTORE, restoring flip-flop initial values.
    pub const GRESTORE: u32 = 0x0A;
    /// Switch to the configured clock rate.
    pub const SWITCH: u32 = 0x09;
    /// Reset the CRC register.
    pub const RCRC: u32 = 0x07;
    /// Leave the synchronised state.
    pub const DESYNC: u32 = 0x0D;
    /// Internal reconfiguration.
    pub const IPROG: u32 = 0x0F;
}

/// Dummy word, clocked in before the sync word to flush the configuration pipeline.
pub const DUMMY: u32 = 0xFFFF_FFFF;
/// The sync word. Everything before it is ignored; everything after is packets.
pub const SYNC: u32 = 0xAA99_5566;
/// A type-1 packet with zero payload: the padding between real commands.
pub const NOOP: u32 = 0x2000_0000;

/// Type-1 packet header: read `count` words from `reg`.
#[must_use]
pub fn type1_read(reg: u32, count: u32) -> u32 {
    0x2800_0000 | ((reg & 0x3FFF) << 13) | (count & 0x7FF)
}

/// Type-1 packet header: write `count` words to `reg`.
#[must_use]
pub fn type1_write(reg: u32, count: u32) -> u32 {
    0x3000_0000 | ((reg & 0x3FFF) << 13) | (count & 0x7FF)
}

/// Type-2 packet header: continue the previous register with a long word count.
#[must_use]
pub fn type2_write(count: u32) -> u32 {
    0x5000_0000 | (count & 0x07FF_FFFF)
}

/// Type-2 packet header for a READ continuation: the form a long frame readback uses.
///
/// A write continuation and a read continuation differ by one bit of the opcode field, and the
/// JTAG readback path shifted the read form as a bare literal. Naming it here is what keeps the
/// two from being confused: a readback issued with the write header returns nothing and looks
/// like a board that is not answering.
#[must_use]
pub fn type2_read(count: u32) -> u32 {
    0x4800_0000 | (count & 0x07FF_FFFF)
}

/// The payload of a Xilinx `.bit` container, plus whatever metadata the header carried.
#[derive(Debug, Clone)]
pub struct BitFile<'a> {
    /// Design name from the header.
    pub design: String,
    /// Target part, which must match the `IDCODE` the stream writes.
    pub part: String,
    /// Build date from the header.
    pub date: String,
    /// Build time from the header.
    pub time: String,
    /// The raw configuration stream, still borrowing the input.
    pub config: &'a [u8],
}

fn be16(d: &[u8], p: usize) -> usize {
    ((d[p] as usize) << 8) | d[p + 1] as usize
}

/// Parse a `.bit` container. Falls back to treating the whole buffer as raw configuration data
/// (a `.bin`) when no recognizable header is present.
#[must_use]
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

/// The bytes every `.bit` container opens with: a two-byte length, a nine-byte field-0 payload,
/// and the `0x0001` marker that closes it. [`parse_bit`] skips exactly this much before reading
/// the first tagged field.
pub const BIT_PREAMBLE: [u8; 13] = [
    0x00, 0x09, 0x0f, 0xf0, 0x0f, 0xf0, 0x0f, 0xf0, 0x0f, 0xf0, 0x00, 0x00, 0x01,
];

/// Wrap a raw configuration stream in a `.bit` container.
///
/// The inverse of [`parse_bit`], and the reason it exists is a crash rather than a preference.
/// A raw stream is a perfectly good configuration — the device takes it and asserts DONE — but it
/// is a `.bin`, and writing one under a `.bit` name hands every tool that reads the extension a
/// file with no header where it expects one. `openFPGALoader` does not report that as an error:
/// it **segfaults**. Emitting the container costs about a hundred bytes and makes the output
/// loadable by name rather than by flag.
///
/// The part string is written for the reader's benefit only; what the device actually checks is
/// the `IDCODE` the stream itself writes, so the two must agree and only one of them is enforced
/// by hardware.
///
/// # Panics
///
/// If any header string is 65,535 bytes or longer, or the configuration exceeds `u32::MAX`
/// bytes — neither is reachable with a real design, and a silent truncation would produce a
/// container whose declared length disagrees with its contents.
#[must_use]
pub fn write_bit(design: &str, part: &str, date: &str, time: &str, config: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(config.len() + 128);
    out.extend_from_slice(&BIT_PREAMBLE);
    for (tag, text) in [(b'a', design), (b'b', part), (b'c', date), (b'd', time)] {
        out.push(tag);
        // Each field is NUL-terminated inside its own declared length, which is how the vendor
        // writes it and what `parse_bit` trims back off.
        let len = text.len() + 1;
        let len = u16::try_from(len).expect("a header field shorter than 65,535 bytes");
        out.extend_from_slice(&len.to_be_bytes());
        out.extend_from_slice(text.as_bytes());
        out.push(0);
    }
    out.push(b'e');
    let len = u32::try_from(config.len()).expect("a configuration stream under 4 GiB");
    out.extend_from_slice(&len.to_be_bytes());
    out.extend_from_slice(config);
    out
}

/// Locate the sync word in a configuration payload, returning the offset of the word AFTER it.
#[must_use]
pub fn find_sync(config: &[u8]) -> Option<usize> {
    config
        .windows(4)
        .position(|w| w == SYNC.to_be_bytes())
        .map(|i| i + 4)
}


/// A decoded configuration packet.
#[derive(Debug, Clone, PartialEq)]
pub enum Packet {
    /// A no-op word.
    Nop,
    /// A write to a configuration register.
    Write {
        /// Register address, from [`reg`].
        reg: u32,
        /// Payload words.
        data: Vec<u32>,
    },
    /// A read request.
    Read {
        /// Register address.
        reg: u32,
        /// Words requested.
        count: u32,
    },
    /// Type-2 continuation of the previous register.
    Continue {
        /// Payload words that follow.
        words: usize,
    },
    /// A header word this decoder does not recognise, carried rather than dropped.
    Unknown(u32),
}

/// Walk a configuration word stream from the sync word onward. Returns the packets in order;
/// stops at the end of the buffer. Used to validate our encoders against real bitstreams.
#[must_use]
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
#[must_use]
pub fn to_words(config: &[u8]) -> Vec<u32> {
    config.as_chunks::<4>().0.iter().map(|&c| u32::from_be_bytes(c)).collect()
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

    /// The writer must produce exactly the container this module's parser has always been tested
    /// against — byte for byte, not merely something that parses back. A writer checked only
    /// against its own reader agrees with itself and with nothing else.
    #[test]
    fn the_writer_emits_the_container_the_parser_was_tested_against() {
        let payload: Vec<u8> = [DUMMY, SYNC, NOOP].iter().flat_map(|w| w.to_be_bytes()).collect();
        let hand = synthetic_bit("fabric;UserID=0X0", "7a100tfgg484", &payload);
        let written =
            write_bit("fabric;UserID=0X0", "7a100tfgg484", "2026/08/05", "12:00:00", &payload);
        assert_eq!(written, hand, "the writer and the hand-built container must agree");
        assert_eq!(&written[..BIT_PREAMBLE.len()], &BIT_PREAMBLE[..]);
        let parsed = parse_bit(&written);
        assert_eq!(parsed.design, "fabric;UserID=0X0");
        assert_eq!(parsed.part, "7a100tfgg484");
        assert_eq!(parsed.date, "2026/08/05");
        assert_eq!(parsed.time, "12:00:00");
        assert_eq!(parsed.config, &payload[..]);
        // The headerless case still parses as raw. That fallback is correct and is exactly what
        // made the crash possible elsewhere: the stream was fine, the extension was the lie.
        assert_eq!(parse_bit(&payload).config, &payload[..]);
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
    /// packet layout the device actually accepts (see `examples/decode_bit.rs`, run against a
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
        // the read continuation is a DIFFERENT header, and this is the literal the JTAG
        // readback path shifts; if the two were interchangeable this assertion would not hold
        assert_eq!(type2_read(0x1234), 0x4800_1234);
        assert_ne!(type2_read(0x1234), type2_write(0x1234));
    }
}
