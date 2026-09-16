//! `Zynq UltraScale+` geometry: the frame, the frame address, and the container the board loads.
//!
//! The Kria KV260 carries a ZU5EV, which is the same configuration protocol as the Artix-7 in
//! [`crate::bitstream`] — same sync word, same register file, same packet encoding — over
//! different geometry and inside a different file. Three things change, and each one is a way to
//! read a valid bitstream and conclude it is empty.
//!
//! **The frame is 93 words, not 101.** Measured, not assumed: the frame data of a Kria K26
//! reference image is 1,948,908 words and of its partial template 15,531. Both divide by 93
//! exactly (20,956 and 167 frames) and neither divides by 101 or by 123.
//!
//! **The minor field is eight bits wide, not seven.** Over the 20,905 frame-address writes in
//! that image the minor runs to 255, and the block-RAM content columns are exactly 256 frames
//! deep — a seven-bit field could not address them. Masking a 7-series `0x7F` across this family
//! silently folds the top half of every wide column onto the bottom half.
//!
//! **The file is little-endian and starts with a decoy.** What the Linux FPGA manager loads is a
//! boot image: 32-bit words in little-endian byte order, opening with a `ZynqMP` boot header that
//! itself contains the word `0xAA995566` as its bus-width pattern, followed by the ASCII tag
//! `XLNX`. A reader that stops at the first sync word parses the boot header as configuration
//! and finds nothing. [`config_start`] takes the sync word that a no-op follows, which is where
//! the configuration stream actually begins — word 2,580 in both images here.
//!
//! The geometry this module read off that image, for one ZU5EV:
//!
//! | block type | clock rows | columns per row | column depths seen |
//! |---|---|---|---|
//! | 0 — logic, I/O and clocking | 4 | 134 | 4, 6, 8, 9, 10, 12, 16, 76 |
//! | 1 — block RAM contents | 4 | 3 | 256 |
//!
//! Frame parity holds here as it does on 7-series: none of those 21,123 frames has an odd number
//! of set bits. See [`crate::ecc`], which is where the check lives; the position of the parity
//! bit within an `UltraScale+` frame this sweep did not locate, so [`crate::ecc::set_parity`] takes
//! the position from its caller and there is no `UltraScale+` default.

/// Words in one `UltraScale+` configuration frame.
pub const WORDS_PER_FRAME: usize = 93;

/// Bits in one `UltraScale+` configuration frame.
pub const BITS_PER_FRAME: usize = WORDS_PER_FRAME * 32;

/// The IDCODE a Kria K26 module's ZU5EV declares, read from its own reference image.
///
/// The low 28 bits are the device; the top four are a silicon revision, so a comparison that
/// includes them can fail on a part that is otherwise the right one.
pub const K26_IDCODE: u32 = 0x04A4_9093;

/// Mask for the part of an IDCODE that identifies the device rather than the revision.
pub const IDCODE_DEVICE: u32 = 0x0FFF_FFFF;

/// A no-op word, which is what follows the configuration stream's sync word.
pub const NOOP: u32 = 0x2000_0000;

/// An `UltraScale+` frame address, split into its fields.
///
/// The field widths are the measured ones, not the 7-series ones: three bits of block type, six
/// of row, ten of column and **eight** of minor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Far {
    /// Block type: 0 is logic, I/O and clocking; 1 is block RAM content.
    pub block_type: u8,
    /// Clock row.
    pub row: u8,
    /// Major column within the row.
    pub column: u16,
    /// Which frame within the column.
    pub minor: u8,
}

impl Far {
    /// Split a packed frame address into its fields.
    #[must_use]
    pub fn decode(far: u32) -> Far {
        Far {
            block_type: ((far >> 24) & 0x7) as u8,
            row: ((far >> 18) & 0x3F) as u8,
            column: ((far >> 8) & 0x3FF) as u16,
            minor: (far & 0xFF) as u8,
        }
    }

    /// Pack the fields back into a frame address. Inverse of [`Far::decode`].
    #[must_use]
    pub fn encode(&self) -> u32 {
        let block = (self.block_type as u32 & 0x7) << 24;
        let row = (self.row as u32 & 0x3F) << 18;
        let column = (self.column as u32 & 0x3FF) << 8;
        block + row + column + (self.minor as u32 & 0xFF)
    }

    /// The address of the first frame in this address's column.
    #[must_use]
    pub fn column_base(&self) -> u32 {
        let mut base = *self;
        base.minor = 0;
        base.encode()
    }
}

/// Read a boot image's bytes as 32-bit words, in the little-endian order the file stores them.
///
/// A trailing partial word is dropped; the caller sees the effect through the word count.
#[must_use]
pub fn words(bytes: &[u8]) -> Vec<u32> {
    let whole = bytes.len() / 4;
    let mut out = Vec::with_capacity(whole);
    for i in 0..whole {
        let b = i * 4;
        out.push(u32::from_le_bytes([bytes[b], bytes[b + 1], bytes[b + 2], bytes[b + 3]]));
    }
    out
}

/// Write words back in the byte order the FPGA manager expects.
#[must_use]
pub fn to_bytes(config: &[u32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(config.len() * 4);
    for w in config {
        out.extend_from_slice(&w.to_le_bytes());
    }
    out
}

/// Index of the sync word that begins the configuration stream, skipping the boot header's.
///
/// The rule is what separates the two on real images: the configuration sync is followed by a
/// no-op, and the boot header's is followed by its `XLNX` identification tag. Matching on the
/// sync word alone lands on the header, 2,572 words early, where every later packet decodes as
/// nonsense.
#[must_use]
pub fn config_start(config: &[u32]) -> Option<usize> {
    for i in 0..config.len() {
        if config[i] != crate::bitstream::SYNC {
            continue;
        }
        if config.get(i + 1) == Some(&NOOP) {
            return Some(i);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The frame addresses this test uses are real ones, taken from the last frames a Kria K26
    /// reference image writes.
    const REAL_FARS: [u32; 4] = [0x010C_02F9, 0x010C_02FF, 0x0000_0000, 0x0000_0005];

    /// Every field survives a round trip, on addresses a vendor tool emitted.
    #[test]
    fn real_addresses_round_trip() {
        for far in REAL_FARS {
            assert_eq!(Far::decode(far).encode(), far, "0x{far:08X} did not survive decode");
        }
    }

    /// The measured split of a real address, by hand.
    #[test]
    fn a_real_address_splits_as_measured() {
        let f = Far::decode(0x010C_02F9);
        assert_eq!(f.block_type, 1, "block RAM content");
        assert_eq!(f.row, 3);
        assert_eq!(f.column, 2);
        assert_eq!(f.minor, 0xF9);
    }

    /// The eight-bit minor is the difference from 7-series, so it gets its own test: a minor of
    /// 255 must come back as 255 and must not disturb the column.
    #[test]
    fn the_minor_field_is_eight_bits() {
        let deep = Far { block_type: 1, row: 3, column: 2, minor: 255 };
        let round = Far::decode(deep.encode());
        assert_eq!(round, deep);
        assert_eq!(round.minor, 255);
        // the 7-series mask would fold this frame onto minor 121 and change nothing else
        let folded = Far { minor: (255u32 & 0x7F) as u8, ..deep };
        assert_ne!(folded.encode(), deep.encode());
        // and a whole block-RAM column is 256 frames deep, so the fold loses half of it
        assert_eq!(deep.column_base(), Far { minor: 0, ..deep }.encode());
    }

    /// `column_base` clears the minor and nothing else.
    #[test]
    fn column_base_clears_only_the_minor() {
        let f = Far::decode(0x010C_02F9);
        let base = Far::decode(f.column_base());
        assert_eq!(base.minor, 0);
        assert_eq!(base.block_type, f.block_type);
        assert_eq!(base.row, f.row);
        assert_eq!(base.column, f.column);
    }

    /// The boot header carries a decoy sync word; the first match is the wrong one.
    #[test]
    fn the_first_sync_word_is_not_the_configuration_stream() {
        // the shape of a real image: boot-header sync, the ASCII tag, padding, then the real one
        let mut image = vec![0x1400_0000u32; 8];
        image.push(crate::bitstream::SYNC);
        image.push(0x584C_4E58); // "XLNX", byte-swapped as the file stores it
        image.extend_from_slice(&[0u32; 6]);
        let real = image.len();
        image.push(crate::bitstream::SYNC);
        image.push(NOOP);
        assert_eq!(config_start(&image), Some(real));
        assert_ne!(config_start(&image), Some(8), "matched the boot header's decoy");
    }

    /// An image with no configuration stream is refused rather than answered with zero.
    #[test]
    fn an_image_without_a_configuration_stream_has_no_start() {
        let mut image = vec![0u32; 12];
        image[8] = crate::bitstream::SYNC;
        image[9] = 0x584C_4E58;
        assert_eq!(config_start(&image), None);
    }

    /// Words go out in the byte order they came in.
    #[test]
    fn bytes_round_trip_little_endian() {
        let raw: Vec<u8> = vec![0x66, 0x55, 0x99, 0xAA, 0x00, 0x00, 0x00, 0x20];
        let w = words(&raw);
        assert_eq!(w, vec![crate::bitstream::SYNC, NOOP]);
        assert_eq!(to_bytes(&w), raw);
        // read big-endian, as a 7-series stream is, the same bytes are not the sync word
        assert_ne!(u32::from_be_bytes([raw[0], raw[1], raw[2], raw[3]]), crate::bitstream::SYNC);
    }

    /// A trailing partial word is dropped, not padded into a false one.
    #[test]
    fn a_partial_trailing_word_is_dropped() {
        assert_eq!(words(&[1, 2, 3]).len(), 0);
        assert_eq!(words(&[1, 2, 3, 4, 5]).len(), 1);
    }

    /// The frame geometry is what the measurement said, and it is not the 7-series one.
    #[test]
    fn frame_geometry_differs_from_seven_series() {
        assert_eq!(WORDS_PER_FRAME, 93);
        assert_eq!(BITS_PER_FRAME, 2976);
        assert_ne!(WORDS_PER_FRAME, crate::frame::WORDS_PER_FRAME);
        // the two measured images divide by this geometry and by no other tried
        for total in [1_948_908usize, 15_531] {
            assert_eq!(total % WORDS_PER_FRAME, 0);
            assert_ne!(total % crate::frame::WORDS_PER_FRAME, 0);
        }
    }

    /// The IDCODE comparison that survives a silicon revision.
    #[test]
    fn idcode_compares_below_the_revision() {
        let revised = K26_IDCODE + 0x1000_0000;
        assert_ne!(revised, K26_IDCODE);
        assert_eq!(revised & IDCODE_DEVICE, K26_IDCODE & IDCODE_DEVICE);
    }
}
