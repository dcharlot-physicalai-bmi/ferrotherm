//! Frame parity: the invariant every vendor frame holds, and the one ours did not.
//!
//! A configuration frame is not only a bag of feature bits. Xilinx parts carry a per-frame error
//! correcting code so the device can scrub its own configuration memory for upsets while the
//! design runs. The code's field is reserved inside the frame; a generator that leaves it zero
//! writes frames that differ structurally from anything the vendor emits.
//!
//! This module implements the part of that code this project can back with evidence: the overall
//! parity bit. Two independent lines of evidence fix it, and both are reproducible from published
//! data rather than from a datasheet reading.
//!
//! **The field is reserved.** In the Project X-Ray Artix-7 database, frame word 50 is occupied
//! only by the `HCLK` family of tiles (`HCLK_L`, `HCLK_R` and `HCLK_IOI3` sit at word offset 50
//! with a one-word span; `HCLK_CMT` spans words 45 to 54). Sweeping every `segbits_*.db` in that
//! database for a bit that resolves into frame word 50, bits 0 to 13, finds **none** — out of
//! 1,404 segbits in the three one-word tile types, the lowest bit index is 14, and `HCLK_CMT`'s
//! lowest bit inside word 50 is 14 as well. Four tile types, reverse-engineered separately,
//! agree on the same boundary. A fourteen-bit hole that no feature ever uses is the ECC field.
//!
//! **Real frames have even parity.** Counting set bits over whole frames of vendor bitstreams:
//!
//! | source | frames | frames with odd parity |
//! |---|---|---|
//! | Four Vivado designs, XC7A35T (`arty-a7` pmod / swbut / uart, `basys3` swbut) | 21,680 | 0 |
//! | Kria K26 reference image, ZU5EV | 20,956 | 0 |
//! | Kria K26 partial template, ZU5EV | 167 | 0 |
//! | **this library, before this module** (`lab_fabric.bit`) | **40** | **20** |
//!
//! The four 7-series designs are independent: taken as vectors over GF(2) their frames span a
//! 472-dimensional subspace, and each design adds directions the others do not (106, 119, 192 and
//! 55 respectively). An all-ones vector orthogonal to 472 independent directions by chance is a
//! `2^-472` event, so the parity relation is structural and the check is not vacuous.
//!
//! **What this module does not claim.** The other twelve bits of the field are a Hamming code
//! whose bit-to-syndrome map this sweep did not recover: the raw linear position labelling and
//! the power-of-two-skipping Hamming labelling were both tried, under forward and reversed word
//! order, forward and reversed bit order, and 66 offsets, and none reproduced the vendor frames.
//! They are left zero, and [`Survey`] reports parity alone. Configuration does not depend on
//! them — a board took a stream from this library with the whole field zero and raised `DONE` —
//! so the code gates readback and upset detection, not loading.

/// The word of a 7-series frame that holds the error correcting code.
pub const X7_ECC_WORD: usize = 50;
/// How many low bits of [`X7_ECC_WORD`] no configuration feature uses.
pub const X7_ECC_BITS: u32 = 14;
/// The bit within [`X7_ECC_WORD`] that carries the frame's overall parity.
pub const X7_PARITY_BIT: u32 = 12;

/// Why a parity write could not be applied.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EccError {
    /// The frame is shorter than the word the field lives in.
    ShortFrame {
        /// Words the frame actually has.
        words: usize,
        /// The word index the field needs.
        needed: usize,
    },
    /// The requested bit is outside a 32-bit word.
    BitOutOfRange {
        /// The bit index asked for.
        bit: u32,
    },
}

impl std::fmt::Display for EccError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EccError::ShortFrame { words, needed } => {
                write!(f, "frame has {words} words, the ECC field needs word {needed}")
            }
            EccError::BitOutOfRange { bit } => write!(f, "bit {bit} is outside a 32-bit word"),
        }
    }
}

impl std::error::Error for EccError {}

/// Set bits across a whole frame.
#[must_use]
pub fn weight(frame: &[u32]) -> u32 {
    let mut total = 0;
    for w in frame {
        total += w.count_ones();
    }
    total
}

/// Whether a frame's set-bit count is even, which every vendor frame measured here is.
#[must_use]
pub fn is_even(frame: &[u32]) -> bool {
    weight(frame).is_multiple_of(2)
}

/// Set or clear the parity bit so the frame's set-bit count is even, and say whether the bit
/// ended up set.
///
/// The bit is part of the count it corrects, so this is not "count, then write": clearing the
/// bit first is what makes the result independent of what was there before. Writing the parity
/// of a frame that already had its parity bit set would flip the answer on every second call.
///
/// # Errors
///
/// [`EccError`] when the frame is too short for the field, or the bit index is not a bit.
pub fn set_parity(frame: &mut [u32], word: usize, bit: u32) -> Result<bool, EccError> {
    if bit >= 32 {
        return Err(EccError::BitOutOfRange { bit });
    }
    if frame.len() <= word {
        return Err(EccError::ShortFrame { words: frame.len(), needed: word });
    }
    let mask = 1u32 << bit;
    frame[word] &= !mask;
    if is_even(frame) {
        return Ok(false);
    }
    frame[word] |= mask;
    Ok(true)
}

/// Set 7-series frame parity at its published position.
///
/// # Errors
///
/// [`EccError`] when the frame is shorter than 51 words.
pub fn set_x7_parity(frame: &mut [u32]) -> Result<bool, EccError> {
    set_parity(frame, X7_ECC_WORD, X7_PARITY_BIT)
}

/// What a scan of a frame stream found.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Survey {
    /// Frames the stream divides into.
    pub frames: usize,
    /// Frames holding at least one set bit.
    pub occupied: usize,
    /// Frames whose set-bit count is odd.
    pub odd: usize,
    /// Words left over after the last whole frame — nonzero means the geometry is wrong.
    pub remainder: usize,
}

impl Survey {
    /// Whether this stream carries frame parity the way vendor streams do.
    ///
    /// The `frames > 0` clause is the point. A survey of an empty stream has nothing odd in it
    /// and would otherwise report agreement, which is the shape of check that passes because it
    /// compared nothing.
    #[must_use]
    pub fn vendor_shaped(&self) -> bool {
        self.frames > 0 && self.odd == 0 && self.remainder == 0
    }
}

/// Split a frame-data stream into frames and count the ones whose parity is wrong.
///
/// `words_per_frame` is the geometry: 101 on 7-series, 93 on `UltraScale+`. A stream that does not
/// divide evenly is reported through [`Survey::remainder`] rather than truncated, because the
/// usual cause is the wrong device family and silently dropping the tail hides it.
#[must_use]
pub fn survey(stream: &[u32], words_per_frame: usize) -> Survey {
    if words_per_frame == 0 {
        return Survey { frames: 0, occupied: 0, odd: 0, remainder: stream.len() };
    }
    let mut s = Survey {
        frames: stream.len() / words_per_frame,
        occupied: 0,
        odd: 0,
        remainder: stream.len() % words_per_frame,
    };
    for chunk in stream.chunks_exact(words_per_frame) {
        let w = weight(chunk);
        if w > 0 {
            s.occupied += 1;
        }
        if w % 2 == 1 {
            s.odd += 1;
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::WORDS_PER_FRAME;

    /// The field the database leaves empty is where the parity bit goes.
    #[test]
    fn the_reserved_field_is_where_we_write() {
        let (parity, field) = (X7_PARITY_BIT, X7_ECC_BITS);
        assert!(parity < field, "the parity bit lives inside the reserved field");
        assert_eq!(X7_ECC_WORD, 50);
    }

    /// An odd frame gains the bit; an even frame does not.
    #[test]
    fn parity_is_set_only_when_the_count_is_odd() {
        let mut odd = vec![0u32; WORDS_PER_FRAME];
        odd[3] = 0b1;
        assert!(set_x7_parity(&mut odd).unwrap(), "one set bit is odd, so parity is written");
        assert!(is_even(&odd));
        assert_eq!(odd[X7_ECC_WORD], 1 << X7_PARITY_BIT);

        let mut even = vec![0u32; WORDS_PER_FRAME];
        even[3] = 0b11;
        assert!(!set_x7_parity(&mut even).unwrap(), "two set bits are already even");
        assert!(is_even(&even));
        assert_eq!(even[X7_ECC_WORD], 0);
    }

    /// Running it twice must not toggle. The bit is inside the count it corrects, so a naive
    /// implementation that counts before clearing alternates between right and wrong.
    #[test]
    fn setting_parity_is_idempotent() {
        let mut f = vec![0u32; WORDS_PER_FRAME];
        f[7] = 0xDEAD_BEEF;
        let first = set_x7_parity(&mut f).unwrap();
        let snapshot = f.clone();
        let second = set_x7_parity(&mut f).unwrap();
        assert_eq!(first, second);
        assert_eq!(f, snapshot, "a second pass changed the frame");
        assert!(is_even(&f));
    }

    /// An all-zero frame is already even, so it keeps the field clear -- which is what the
    /// vendor streams show for their unoccupied frames.
    #[test]
    fn an_empty_frame_stays_empty() {
        let mut f = vec![0u32; WORDS_PER_FRAME];
        assert!(!set_x7_parity(&mut f).unwrap());
        assert_eq!(weight(&f), 0);
    }

    /// A frame too short for the field is refused rather than silently skipped.
    #[test]
    fn a_short_frame_is_an_error() {
        let mut f = vec![0u32; 8];
        assert_eq!(
            set_x7_parity(&mut f),
            Err(EccError::ShortFrame { words: 8, needed: X7_ECC_WORD })
        );
        assert_eq!(set_parity(&mut f, 0, 32), Err(EccError::BitOutOfRange { bit: 32 }));
    }

    /// A survey of nothing must not read as agreement.
    #[test]
    fn an_empty_survey_is_not_vendor_shaped() {
        let s = survey(&[], WORDS_PER_FRAME);
        assert_eq!(s.frames, 0);
        assert!(!s.vendor_shaped(), "a check that compared nothing claimed a pass");
    }

    /// The survey separates the two failures: wrong parity, and the wrong device family.
    #[test]
    fn survey_reports_parity_and_geometry_apart() {
        let mut stream = vec![0u32; WORDS_PER_FRAME * 3];
        stream[0] = 1; // frame 0: one set bit, odd
        stream[WORDS_PER_FRAME] = 3; // frame 1: two set bits, even
        let s = survey(&stream, WORDS_PER_FRAME);
        assert_eq!(s.frames, 3);
        assert_eq!(s.occupied, 2);
        assert_eq!(s.odd, 1);
        assert_eq!(s.remainder, 0);
        assert!(!s.vendor_shaped());

        // the same words read as UltraScale+ frames: the geometry does not divide
        let wrong = survey(&stream, crate::usplus::WORDS_PER_FRAME);
        assert_ne!(wrong.remainder, 0, "101-word frames read as 93-word ones must not divide");
        assert!(!wrong.vendor_shaped());
    }

    /// Correcting every frame in a stream makes the stream vendor-shaped.
    #[test]
    fn correcting_a_stream_makes_it_vendor_shaped() {
        let mut stream = vec![0u32; WORDS_PER_FRAME * 4];
        for i in 0..4 {
            stream[i * WORDS_PER_FRAME + i + 1] = 0x8000_0001 ^ (i as u32);
        }
        let before = survey(&stream, WORDS_PER_FRAME);
        assert!(before.odd > 0, "the fixture must start with frames to fix");
        for i in 0..4 {
            let at = i * WORDS_PER_FRAME;
            set_x7_parity(&mut stream[at..at + WORDS_PER_FRAME]).unwrap();
        }
        assert!(survey(&stream, WORDS_PER_FRAME).vendor_shaped());
    }

    #[test]
    fn weight_counts_the_whole_frame() {
        let mut f = vec![0u32; WORDS_PER_FRAME];
        f[0] = 0xFFFF_FFFF;
        f[100] = 0b101;
        assert_eq!(weight(&f), 34);
    }
}
