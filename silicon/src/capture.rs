//! Reading a running fabric: latch the live state into configuration memory, stream it back, and
//! name what came out.
//!
//! A sampler on this fabric has no output pins. Its spins are flip-flops inside slices, and the
//! only wire out of the part is JTAG. Xilinx parts answer exactly this: `GCAPTURE` copies every
//! flip-flop's current value into the configuration memory cell that held its initial value, and
//! a frame readback then returns it. The design keeps running throughout — capture reads, it does
//! not stop the clock.
//!
//! Three things have to be right, and each is a way to get a plausible wrong answer.
//!
//! **The readback returns a pad frame first.** The device streams one frame of pipeline contents
//! before the frame you asked for. Take the buffer at face value and every frame is one late:
//! frame *n*'s bits are read as frame *n+1*'s, the values are all real bits from the real device,
//! and nothing anywhere looks wrong. [`strip_pad`] refuses a short buffer rather than returning
//! whatever is there.
//!
//! **A captured bit is anonymous.** A 7-series frame is 3,232 bits with nothing in it to say
//! which flip-flop is which. The name comes from Vivado's logic-location file, which states the
//! frame address, the bit offset within the frame, and the latch — see [`crate::logic_location`].
//! Without that file a readback is a wall of bits; with it, [`Readout::latch`] answers by name.
//!
//! **A masked control bit does nothing.** `CTL0` is written through `MASK`, and only the bits set
//! in `MASK` reach the register. The configuration stream this project generates writes
//! `CTL0 = 0x0000_0501` under `MASK = 0x0000_0401`, so bit 8 — set in the value — never lands.
//! [`reaches`] is that arithmetic, and [`GLUTMASK`] is the bit in question: reading lookup-table
//! RAM and shift-register contents back needs it, and the current stream does not deliver it.
//! Flip-flop capture does not depend on it.
//!
//! # What is checked and what is not
//!
//! The packet sequence, the pad-frame rule, the address walk and the naming all have offline
//! tests. Whether `GCAPTURE` latches is not a thing a test in this crate can establish, so it was
//! put to a part: `examples/capture_hw`, on 2026-09-19, against the design an Alchitry Pt V2
//! (XC7A100T) boots from its flash. The test is differential, because nobody here knows what that
//! design's flip-flops should hold. Over 576 columns, two columns moved between two captures
//! 30 ms apart (11 bits and 2 bits). On both: two plain readbacks were identical, so the movement
//! is not readback noise; two captures differed, so it is live state; and a plain readback after
//! a capture equalled that capture bit for bit, so the latch writes configuration memory and the
//! value stays. `STAT` read `DONE=1 CRC_ERR=0` before and after.
//!
//! WHAT THAT RUN DOES NOT ESTABLISH: that a captured bit is the value of a NAMED flip-flop. No
//! captured bit has yet been compared with a value known by other means, so the naming path
//! ([`Readout::latch`] through a `.ll` listing) is still checked offline only. Our own emitted
//! fabric cannot supply that value: it is combinational and has no flip-flop to latch.
//!
//! The same run found a deadlock in the JTAG driver that no earlier readback was long enough to
//! meet; see [`crate::mpsse`].

use crate::bitstream::{cmd, reg, type1_read, type1_write, type2_read, DUMMY, NOOP, SYNC};
use crate::frame::WORDS_PER_FRAME;
use crate::logic_location::LlBit;
use crate::tilegrid::Far;

/// Frames of pipeline contents the device streams before the first frame you asked for.
pub const PAD_FRAMES: usize = 1;

/// The `CTL0` bit that decides whether lookup-table RAM and shift-register contents appear in a
/// readback.
pub const GLUTMASK: u32 = 1 << 8;

/// The `MASK` value the generated configuration stream writes before its `CTL0`.
pub const STREAM_MASK: u32 = 0x0000_0401;

/// The `CTL0` value the generated configuration stream writes.
pub const STREAM_CTL0: u32 = 0x0000_0501;

/// Frames in one 7-series configuration column: the minor field is seven bits wide.
pub const FRAMES_PER_COLUMN: usize = 128;

/// Why a readback could not be turned into frames.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CaptureError {
    /// Fewer words came back than the request implies, so the pad frame cannot be identified.
    Short {
        /// Words received.
        got: usize,
        /// Words the request needs.
        want: usize,
    },
    /// The run would leave the column it starts in, where addresses stop incrementing by one.
    CrossesColumn {
        /// Minor address the run starts at.
        minor: u8,
        /// Frames requested.
        frames: usize,
    },
    /// No frames were asked for, so there is nothing a readback could establish.
    Empty,
}

impl std::fmt::Display for CaptureError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CaptureError::Short { got, want } => {
                write!(f, "readback returned {got} words, the request needs {want}")
            }
            CaptureError::CrossesColumn { minor, frames } => write!(
                f,
                "{frames} frames from minor {minor} leave the column; read one column at a time"
            ),
            CaptureError::Empty => write!(f, "a readback of zero frames establishes nothing"),
        }
    }
}

impl std::error::Error for CaptureError {}

/// Words the device returns for a request of `n_frames`, pad frame included.
#[must_use]
pub fn readback_words(n_frames: usize) -> usize {
    (n_frames + PAD_FRAMES) * WORDS_PER_FRAME
}

/// Which bits of a `CTL0` write actually reach the register, given the `MASK` that precedes it.
///
/// A control bit set in the value and clear in the mask is the shape of change that looks applied
/// and is not: the write succeeds, the register does not move, and the behaviour it was meant to
/// enable stays off.
#[must_use]
pub fn reaches(mask: u32, value: u32) -> u32 {
    mask & value
}

/// The words to shift into the configuration port to latch the fabric and read `n_frames` back
/// starting at `far`.
///
/// `GCAPTURE` comes before `RCFG`, because the latch has to happen while the design is still the
/// thing in the configuration memory. Ordering them the other way reads out the configuration and
/// captures into a frame nobody looks at — which returns the bitstream you loaded, exactly, and
/// so reads as a design that never changes state.
#[must_use]
pub fn capture_and_read(far: u32, n_frames: usize) -> Vec<u32> {
    let want = readback_words(n_frames) as u32;
    vec![
        DUMMY,
        SYNC,
        NOOP,
        type1_write(reg::CMD, 1),
        cmd::GCAPTURE,
        NOOP,
        NOOP,
        type1_write(reg::CMD, 1),
        cmd::RCFG,
        type1_write(reg::FAR, 1),
        far,
        type1_read(reg::FDRO, 0),
        type2_read(want),
        NOOP,
        NOOP,
    ]
}

/// Drop the pad frame, or say why the buffer cannot carry the frames requested.
///
/// # Errors
///
/// [`CaptureError::Empty`] for a request of nothing, [`CaptureError::Short`] when fewer words came
/// back than the request implies.
pub fn strip_pad(words: &[u32], n_frames: usize) -> Result<&[u32], CaptureError> {
    if n_frames == 0 {
        return Err(CaptureError::Empty);
    }
    let want = readback_words(n_frames);
    if words.len() < want {
        return Err(CaptureError::Short { got: words.len(), want });
    }
    Ok(&words[PAD_FRAMES * WORDS_PER_FRAME..want])
}

/// Frames read back from a device, with the address each one came from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Readout {
    /// The frame address the run started at.
    pub first: u32,
    /// The frames, in address order, pad removed.
    pub frames: Vec<Vec<u32>>,
}

impl Readout {
    /// Build a readout from the raw words a readback returned.
    ///
    /// Addresses are `first + i`, which holds inside one column and stops holding at its end, so a
    /// run that would leave the column is refused rather than mislabelled.
    ///
    /// # Errors
    ///
    /// [`CaptureError`] when the request is empty, the buffer is short, or the run crosses a
    /// column boundary.
    pub fn from_words(first: u32, words: &[u32], n_frames: usize) -> Result<Readout, CaptureError> {
        let minor = Far::decode(first).minor;
        if minor as usize + n_frames > FRAMES_PER_COLUMN {
            return Err(CaptureError::CrossesColumn { minor, frames: n_frames });
        }
        let body = strip_pad(words, n_frames)?;
        let mut frames = Vec::with_capacity(n_frames);
        for i in 0..n_frames {
            let at = i * WORDS_PER_FRAME;
            frames.push(body[at..at + WORDS_PER_FRAME].to_vec());
        }
        Ok(Readout { first, frames })
    }

    /// The frame read from `addr`, if the run covered it.
    #[must_use]
    pub fn frame(&self, addr: u32) -> Option<&[u32]> {
        let offset = addr.checked_sub(self.first)? as usize;
        self.frames.get(offset).map(Vec::as_slice)
    }

    /// One bit, by frame address and bit offset within the frame.
    #[must_use]
    pub fn bit(&self, addr: u32, offset: u32) -> Option<bool> {
        let frame = self.frame(addr)?;
        let word = (offset / 32) as usize;
        Some(frame.get(word)? >> (offset % 32) & 1 == 1)
    }

    /// The value of a cell Vivado located, by the frame address and offset it states.
    #[must_use]
    pub fn latch(&self, cell: &LlBit) -> Option<bool> {
        self.bit(cell.frame_address, cell.frame_offset)
    }

    /// Read every cell in a listing that this readout covers.
    #[must_use]
    pub fn sample(&self, cells: &[LlBit]) -> Sample {
        let mut s = Sample::default();
        for cell in cells {
            match self.latch(cell) {
                Some(value) => s.values.push(Observed {
                    block: cell.block.clone(),
                    latch: cell.latch.clone().unwrap_or_default(),
                    value,
                }),
                None => s.uncovered += 1,
            }
        }
        s
    }
}

/// One named cell and the value the device returned for it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Observed {
    /// The physical block, for example `SLICE_X0Y0`.
    pub block: String,
    /// The latch within it, for example `CQ`.
    pub latch: String,
    /// What the readback said.
    pub value: bool,
}

/// What a readout could say about a listing.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Sample {
    /// Cells this readout covered, with their values.
    pub values: Vec<Observed>,
    /// Cells the listing names that the run did not reach.
    pub uncovered: usize,
}

impl Sample {
    /// Whether this sample observed anything at all.
    ///
    /// A run that covered none of the listing's cells produces a sample with no disagreement in
    /// it, which reads like a clean result. It is the absence of a result.
    #[must_use]
    pub fn observed(&self) -> bool {
        !self.values.is_empty()
    }

    /// How many of the observed cells read as one.
    #[must_use]
    pub fn ones(&self) -> usize {
        let mut n = 0;
        for o in &self.values {
            if o.value {
                n += 1;
            }
        }
        n
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn listing() -> Vec<LlBit> {
        crate::logic_location::parse_ll(
            "Bit 1 0x00420100 0 SLR0 0 Block=SLICE_X0Y0 Latch=AQ Net=spin[0]\n\
             Bit 2 0x00420100 33 SLR0 0 Block=SLICE_X0Y0 Latch=BQ Net=spin[1]\n\
             Bit 3 0x00420101 64 SLR0 0 Block=SLICE_X0Y1 Latch=AQ Net=spin[2]\n",
        )
    }

    /// A readback buffer with the pad frame in front, and known bits where the listing says.
    ///
    /// The offsets here are LITERAL, not `PAD_FRAMES * WORDS_PER_FRAME`. A fixture written in
    /// terms of the constant it is testing moves when the constant does: with `PAD_FRAMES = 0`
    /// the marks slide up into the pad frame and the test still passes, which is exactly what
    /// happened when this was first written.
    fn buffer() -> Vec<u32> {
        let mut w = vec![0u32; 3 * 101];
        // the pad frame is not zero: if it were, taking it by mistake would look like an
        // all-clear fabric rather than like a bug
        for word in &mut w[..101] {
            *word = 0xA5A5_A5A5;
        }
        let f0 = 101;
        let f1 = 202;
        w[f0] = 1; // frame 0x00420100, offset 0 -> AQ of SLICE_X0Y0 is high
        w[f0 + 1] = 1 << 1; // offset 33 -> BQ is high
        w[f1 + 2] = 0; // frame 0x00420101, offset 64 -> AQ of SLICE_X0Y1 is low
        w
    }

    /// The capture packet sequence latches before it reads, and asks for the pad frame too.
    #[test]
    fn the_stream_captures_before_it_reads() {
        let w = capture_and_read(0x0042_0100, 4);
        let gcap = w.iter().position(|&x| x == cmd::GCAPTURE).expect("GCAPTURE");
        let rcfg = w.iter().position(|&x| x == cmd::RCFG).expect("RCFG");
        assert!(gcap < rcfg, "capturing after the read latches into a frame nobody reads");
        assert_eq!(w[gcap - 1], type1_write(reg::CMD, 1));
        assert_eq!(w[rcfg - 1], type1_write(reg::CMD, 1));
        // the FAR write is what the read starts from, and it follows RCFG
        let far = w.iter().position(|&x| x == type1_write(reg::FAR, 1)).expect("FAR");
        assert!(far > rcfg);
        assert_eq!(w[far + 1], 0x0042_0100);
        // the frame-data register is READ, not written: the write header would load the
        // frames it was meant to fetch
        assert!(w.contains(&type1_read(reg::FDRO, 0)));
        assert!(!w.contains(&type1_write(reg::FDRO, 0)));
        // and the request covers the pad frame
        assert!(w.contains(&type2_read(readback_words(4) as u32)));
        assert_eq!(readback_words(4), 5 * WORDS_PER_FRAME);
    }

    /// The read continuation is the read header, not the write one.
    #[test]
    fn the_continuation_is_a_read() {
        let w = capture_and_read(0, 2);
        let want = readback_words(2) as u32;
        assert!(w.contains(&type2_read(want)));
        assert!(!w.contains(&crate::bitstream::type2_write(want)));
    }

    /// The pad frame comes off the front, and a short buffer is an error rather than a shift.
    #[test]
    fn the_pad_frame_is_dropped_and_a_short_read_refused() {
        let w = buffer();
        let body = strip_pad(&w, 2).unwrap();
        assert_eq!(body.len(), 2 * WORDS_PER_FRAME);
        assert_eq!(body[0], 1, "the first real frame, not the pad");
        assert_ne!(body[0], 0xA5A5_A5A5);

        let short = &w[..w.len() - 1];
        assert_eq!(
            strip_pad(short, 2),
            Err(CaptureError::Short { got: short.len(), want: readback_words(2) })
        );
        assert_eq!(strip_pad(&w, 0), Err(CaptureError::Empty));
    }

    /// Frames are addressed from the start of the run, and a cell is read by its stated offset.
    #[test]
    fn cells_are_read_by_the_address_vivado_states() {
        let r = Readout::from_words(0x0042_0100, &buffer(), 2).unwrap();
        assert_eq!(r.frames.len(), 2);
        assert_eq!(r.bit(0x0042_0100, 0), Some(true));
        assert_eq!(r.bit(0x0042_0100, 33), Some(true));
        assert_eq!(r.bit(0x0042_0101, 64), Some(false));
        assert_eq!(r.bit(0x0042_0102, 0), None, "outside the run");

        let s = r.sample(&listing());
        assert!(s.observed());
        assert_eq!(s.values.len(), 3);
        assert_eq!(s.uncovered, 0);
        assert_eq!(s.ones(), 2);
        assert_eq!(s.values[0].block, "SLICE_X0Y0");
        assert_eq!(s.values[0].latch, "AQ");
        assert!(s.values[0].value);
        assert_eq!(s.values[2].latch, "AQ");
        assert!(!s.values[2].value);
    }

    /// Reading the pad frame as the first real frame is silent, so the test says what it costs.
    #[test]
    fn taking_the_pad_frame_changes_every_answer() {
        let w = buffer();
        let honest = Readout::from_words(0x0042_0100, &w, 2).unwrap();
        // what a reader that forgot the pad frame would build
        let mut naive = Vec::new();
        for i in 0..2 {
            let at = i * WORDS_PER_FRAME;
            naive.push(w[at..at + WORDS_PER_FRAME].to_vec());
        }
        let naive = Readout { first: 0x0042_0100, frames: naive };
        assert_ne!(naive, honest);
        assert_eq!(naive.bit(0x0042_0100, 0), Some(true), "the pad frame is full of ones too");
        assert_eq!(naive.bit(0x0042_0101, 0), Some(true));
        assert_eq!(honest.bit(0x0042_0101, 0), Some(false));
    }

    /// A sample that covered nothing must not read as a clean result.
    #[test]
    fn a_sample_that_covered_nothing_is_not_an_observation() {
        let r = Readout::from_words(0x0050_0000, &buffer(), 2).unwrap();
        let s = r.sample(&listing());
        assert_eq!(s.uncovered, 3);
        assert!(!s.observed(), "a run that reached none of the cells reported success");
        assert_eq!(s.ones(), 0);
    }

    /// A run that would leave its column is refused: past the column end the address stops
    /// incrementing by one and every frame after it would be mislabelled.
    #[test]
    fn a_run_past_the_column_end_is_refused() {
        let far = Far { block_type: 0, bottom_half: false, row: 0, column: 3, minor: 120 }.encode();
        let words = vec![0u32; readback_words(64)];
        assert_eq!(
            Readout::from_words(far, &words, 64),
            Err(CaptureError::CrossesColumn { minor: 120, frames: 64 })
        );
        // eight frames fit exactly, and are allowed
        assert!(Readout::from_words(far, &vec![0u32; readback_words(8)], 8).is_ok());
    }

    /// The control bit a readback of lookup-table RAM needs is outside the mask the generated
    /// stream writes, so setting it in the value changes nothing.
    #[test]
    fn the_stream_does_not_deliver_glutmask() {
        assert_eq!(STREAM_CTL0 & GLUTMASK, GLUTMASK, "the value has the bit");
        assert_eq!(STREAM_MASK & GLUTMASK, 0, "the mask does not admit it");
        assert_eq!(reaches(STREAM_MASK, STREAM_CTL0), 0x0000_0401 & 0x0000_0501);
        assert_eq!(reaches(STREAM_MASK, STREAM_CTL0) & GLUTMASK, 0);
        // and the value that would deliver it
        assert_eq!(reaches(STREAM_MASK | GLUTMASK, STREAM_CTL0) & GLUTMASK, GLUTMASK);
    }

    #[test]
    fn readback_words_counts_the_pad() {
        assert_eq!(readback_words(0), WORDS_PER_FRAME);
        assert_eq!(readback_words(1), 2 * WORDS_PER_FRAME);
        assert_eq!(readback_words(40), 41 * WORDS_PER_FRAME);
    }
}
