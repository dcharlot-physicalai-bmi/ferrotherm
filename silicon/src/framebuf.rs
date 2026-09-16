//! The frame buffer: a sparse set of configuration frames, written bit by bit.
//!
//! A 7-series frame is 101 words of 32 bits. Designs touch a small fraction of the device, so
//! frames are held sparsely and only the touched ones are emitted — matching how real generated
//! streams work (a blink design writes 256 frames out of the part's many thousands).
//!
//! Two safety properties are enforced here because both have burned real projects:
//!  * writing a bit into a frame the buffer has never seen ALLOCATES it (an all-zero frame is a
//!    legitimate thing to write into, and refusing it silently drops the bit — the "empty frame"
//!    trap that makes a correct write look like a failed one);
//!  * a segbit that says a bit must be CLEAR is applied as a clear, not ignored.

use crate::segbits::SegBit;
use crate::tilegrid::{BitAddr, BitsBlock};
use std::collections::BTreeMap;

/// Words in one 7-series configuration frame.
pub const WORDS_PER_FRAME: usize = 101;

#[derive(Debug, Default, Clone)]
/// A sparse set of configuration frames, keyed by frame address and kept in address order.
pub struct FrameBuf {
    /// frame address -> 101 words
    pub frames: BTreeMap<u32, Vec<u32>>,
}

impl FrameBuf {
    #[must_use]
    /// An empty buffer.
    pub fn new() -> FrameBuf {
        FrameBuf { frames: BTreeMap::new() }
    }

    /// Number of frames the buffer will emit.
    #[must_use]
    pub fn len(&self) -> usize {
        self.frames.len()
    }
    #[must_use]
    /// Whether it holds no frames.
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// Ensure a frame exists (all zeros) and return it.
    pub fn frame_mut(&mut self, addr: u32) -> &mut Vec<u32> {
        self.frames.entry(addr).or_insert_with(|| vec![0u32; WORDS_PER_FRAME])
    }

    /// Apply one resolved bit. Returns an error when the word index is outside the frame, which
    /// means the tilegrid and the segbits disagree about the part.
    ///
    /// # Errors
    ///
    /// A message when the word or bit index is outside a frame.
    pub fn apply(&mut self, a: BitAddr) -> Result<(), String> {
        if a.word as usize >= WORDS_PER_FRAME {
            return Err(format!(
                "word {} outside a {WORDS_PER_FRAME}-word frame (frame 0x{:08X})",
                a.word, a.frame
            ));
        }
        let f = self.frame_mut(a.frame);
        let mask = 1u32 << a.bit;
        if a.set {
            f[a.word as usize] |= mask;
        } else {
            f[a.word as usize] &= !mask;
        }
        Ok(())
    }

    /// Apply a feature's segbits within a tile's configuration block.
    ///
    /// # Errors
    ///
    /// A message when a bit falls outside the block's frames or words.
    pub fn apply_feature(&mut self, block: &BitsBlock, bits: &[SegBit]) -> Result<(), String> {
        for sb in bits {
            let a = block
                .resolve(*sb)
                .ok_or_else(|| format!("segbit {}_{} outside the tile's extent", sb.frame, sb.bit))?;
            self.apply(a)?;
        }
        Ok(())
    }

    /// Write a LUT truth table: `init_bits` are the 64 INIT segbits in index order, `init` is
    /// the truth table (bit i = output for input combination i).
    ///
    /// # Errors
    ///
    /// A message when the LUT init does not fit the block, or the block is not a logic block.
    pub fn write_lut_init(
        &mut self,
        block: &BitsBlock,
        init_bits: &[SegBit],
        init: u64,
    ) -> Result<(), String> {
        if init_bits.len() != 64 {
            return Err(format!("expected 64 INIT bits, got {}", init_bits.len()));
        }
        for (i, sb) in init_bits.iter().enumerate() {
            let want = (init >> i) & 1 == 1;
            let a = block
                .resolve(SegBit { set: want, ..*sb })
                .ok_or_else(|| format!("INIT[{i}] outside the tile's extent"))?;
            self.apply(a)?;
        }
        Ok(())
    }

    /// Read a LUT truth table back out of the buffer — used to check what we wrote, never as
    /// evidence about hardware.
    #[must_use]
    pub fn read_lut_init(&self, block: &BitsBlock, init_bits: &[SegBit]) -> Option<u64> {
        let mut init = 0u64;
        for (i, sb) in init_bits.iter().enumerate() {
            let a = block.resolve(*sb)?;
            let f = self.frames.get(&a.frame)?;
            if f[a.word as usize] >> a.bit & 1 == 1 {
                init |= 1u64 << i;
            }
        }
        Some(init)
    }

    /// Give every frame the parity bit its set-bit count calls for, and say how many got one.
    ///
    /// Call this last, after the design's bits are in. Vendor frames all carry it and, before
    /// this existed, half of ours did not; [`crate::ecc`] has the measurements. It is safe to
    /// call on a design placed in logic: nothing in the Project X-Ray Artix-7 database resolves
    /// into the field it writes.
    ///
    /// The one design that must not call it is one that configures `HCLK` tiles by hand at a bit
    /// index below 14, which no published segbit does.
    pub fn set_x7_parity(&mut self) -> usize {
        let mut written = 0;
        for f in self.frames.values_mut() {
            if crate::ecc::set_x7_parity(f).unwrap_or(false) {
                written += 1;
            }
        }
        written
    }

    /// Frames whose set-bit count is odd, which is the shape no vendor frame has.
    #[must_use]
    pub fn odd_frames(&self) -> usize {
        let mut odd = 0;
        for f in self.frames.values() {
            if !crate::ecc::is_even(f) {
                odd += 1;
            }
        }
        odd
    }

    /// Contiguous runs of frames, as (start address, words). Frames stream through FDRI with the
    /// address auto-incrementing, so each run needs exactly one FAR write.
    #[must_use]
    pub fn runs(&self) -> Vec<(u32, Vec<u32>)> {
        let mut out: Vec<(u32, Vec<u32>)> = Vec::new();
        let mut cur_start: Option<u32> = None;
        let mut cur_end = 0u32;
        let mut words: Vec<u32> = Vec::new();
        for (&addr, f) in &self.frames {
            match cur_start {
                Some(_) if addr == cur_end + 1 => {
                    words.extend_from_slice(f);
                    cur_end = addr;
                }
                Some(s) => {
                    out.push((s, std::mem::take(&mut words)));
                    words.extend_from_slice(f);
                    cur_start = Some(addr);
                    cur_end = addr;
                }
                None => {
                    words.extend_from_slice(f);
                    cur_start = Some(addr);
                    cur_end = addr;
                }
            }
        }
        if let Some(s) = cur_start {
            out.push((s, words));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn block() -> BitsBlock {
        BitsBlock { baseaddr: 0x0042_0100, frames: 36, offset: 0, words: 2 }
    }

    /// Writing into a frame that does not exist yet must allocate it, not drop the bit.
    #[test]
    fn writing_an_absent_frame_allocates_it() {
        let mut fb = FrameBuf::new();
        assert!(fb.is_empty());
        fb.apply(BitAddr { frame: 0x0042_0120, word: 0, bit: 15, set: true }).unwrap();
        assert_eq!(fb.len(), 1);
        assert_eq!(fb.frames[&0x0042_0120][0], 1 << 15);
    }

    /// A clear-bit segbit must actually clear.
    #[test]
    fn clear_bits_are_applied() {
        let mut fb = FrameBuf::new();
        fb.apply(BitAddr { frame: 1, word: 0, bit: 3, set: true }).unwrap();
        fb.apply(BitAddr { frame: 1, word: 0, bit: 3, set: false }).unwrap();
        assert_eq!(fb.frames[&1][0], 0);
    }

    #[test]
    fn word_out_of_frame_is_refused() {
        let mut fb = FrameBuf::new();
        assert!(fb.apply(BitAddr { frame: 1, word: 101, bit: 0, set: true }).is_err());
    }

    /// A truth table round-trips through the physical bit positions.
    #[test]
    fn lut_init_round_trip() {
        // the real interleave: even INIT indices in frame 32, odd in 33, bit walking down
        let init_bits: Vec<SegBit> = (0..64)
            .map(|i| SegBit { frame: 32 + (i % 2) as u16, bit: (15 - (i / 2).min(15)) as u16, set: true })
            .collect();
        let mut fb = FrameBuf::new();
        // NOTE: this synthetic map reuses coordinates beyond i=31, so use a table that only
        // exercises distinct positions: the low 32 indices map to distinct (frame, bit) pairs.
        let distinct: Vec<SegBit> = (0..64)
            .map(|i| SegBit { frame: 32 + (i % 2) as u16, bit: (i / 2) as u16, set: true })
            .collect();
        let table = 0xDEAD_BEEF_1234_5678u64;
        fb.write_lut_init(&block(), &distinct, table).unwrap();
        assert_eq!(fb.read_lut_init(&block(), &distinct), Some(table));
        let _ = init_bits;
    }

    /// Contiguous frames coalesce into one run; a gap starts a new one.
    #[test]
    fn runs_coalesce_contiguous_frames() {
        let mut fb = FrameBuf::new();
        for addr in [10u32, 11, 12, 20, 21] {
            fb.frame_mut(addr);
        }
        let runs = fb.runs();
        assert_eq!(runs.len(), 2);
        assert_eq!(runs[0].0, 10);
        assert_eq!(runs[0].1.len(), 3 * WORDS_PER_FRAME);
        assert_eq!(runs[1].0, 20);
        assert_eq!(runs[1].1.len(), 2 * WORDS_PER_FRAME);
    }

    /// Sealing a buffer leaves no frame with the parity no vendor frame has -- and the fixture
    /// asserts it started with some, so a sealer that did nothing would not pass.
    #[test]
    fn sealing_removes_every_odd_frame() {
        let mut fb = FrameBuf::new();
        for addr in 0..8u32 {
            fb.apply(BitAddr { frame: addr, word: 3, bit: (addr % 32) as u16, set: true }).unwrap();
        }
        assert_eq!(fb.odd_frames(), 8, "the fixture must start with frames to seal");
        let written = fb.set_x7_parity();
        assert_eq!(written, 8);
        assert_eq!(fb.odd_frames(), 0);
        // and the design's own bits are untouched
        for f in fb.frames.values() {
            assert_eq!(f[3].count_ones(), 1);
        }
    }

    /// Sealing twice must not undo itself.
    #[test]
    fn sealing_is_idempotent() {
        let mut fb = FrameBuf::new();
        fb.apply(BitAddr { frame: 7, word: 2, bit: 9, set: true }).unwrap();
        fb.set_x7_parity();
        let once = fb.frames.clone();
        fb.set_x7_parity();
        assert_eq!(fb.frames, once);
        assert_eq!(fb.odd_frames(), 0);
    }
}
