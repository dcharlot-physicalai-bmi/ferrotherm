//! How a JTAG shift that READS is cut into pieces an FTDI chip can actually carry.
//!
//! # The deadlock this exists to prevent
//!
//! An MPSSE "clock bytes in and out" command returns one byte for every byte it is given. The
//! first version of `flash::Tap::shift_dr` (no link: that module exists only with the hardware
//! feature, which is the point of this one) built the whole command for a captured shift,
//! sent all of it, and only then began to read. That works until the reply is larger than the
//! chip's return buffer. Then the MPSSE engine stalls with its buffer full, stops draining the OUT
//! endpoint, and the host's blocking write never completes — while the host is not reading,
//! because it has not finished writing. Nothing errors. On 2026-09-19 a 36-frame readback (14.5 KB)
//! against a real XC7A100T sat in `Ftdi::write` for twenty-nine minutes on 0.01 s of CPU.
//!
//! It had never been seen because every earlier readback was a few frames, and it could not have
//! been caught by a test, because the module that talks to the chip only compiles with the
//! hardware feature on. So the PLAN lives here, behind no feature, as a pure function: write a
//! piece, read its reply, write the next. The TAP stays in Shift-DR between MPSSE commands, so a
//! shift may be cut anywhere on a byte boundary.

/// Clock bytes out on TDI and in on TDO, LSB first, on the falling/rising edges JTAG uses.
pub const CLK_BYTES_IO: u8 = 0x39;
/// The same for a count of BITS, which is how a shift's final partial byte is sent.
pub const CLK_BITS_IO: u8 = 0x3B;
/// Clock TMS with no read: a state-machine move.
pub const CLK_TMS: u8 = 0x4B;
/// Clock TMS and sample TDO: the move that also carries a shift's last data bit.
pub const CLK_TMS_IO: u8 = 0x6B;
/// Flush the chip's return buffer to the host now rather than at the latency timer.
pub const SEND_IMMEDIATE: u8 = 0x87;

/// The FT2232H's return buffer, per channel, in bytes (its datasheet's 4 KB).
pub const FTDI_RETURN_BUFFER: usize = 4096;

/// The most reply bytes any one step may generate. A quarter of the chip's buffer: the two-byte
/// status header on every USB packet and whatever the host has not yet collected both eat into
/// the rest, and there is no prize for running the buffer close to full.
pub const RETURN_BUFFER_SAFE: usize = 1024;

// The margin is a claim about the chip, so it is held against the chip's number -- at compile
// time, where raising one constant without the other cannot get as far as a board.
const _: () = assert!(RETURN_BUFFER_SAFE * 2 <= FTDI_RETURN_BUFFER, "no margin under the FTDI's buffer");

/// One write to the chip and the number of reply bytes to collect before the next.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Step {
    /// MPSSE command bytes, ending in [`SEND_IMMEDIATE`].
    pub cmd: Vec<u8>,
    /// Reply bytes this command generates. Never more than [`RETURN_BUFFER_SAFE`].
    pub reads: usize,
}

/// Cut a data-register shift of `tx` that samples TDO into steps the chip can carry.
///
/// From Run-Test/Idle: enter Shift-DR, shift every byte but the last in pieces of at most
/// [`RETURN_BUFFER_SAFE`], then the last byte as seven bits plus one bit riding the TMS clock that
/// leaves Shift-DR, and return to Run-Test/Idle. The reply is `tx.len() - 1` whole bytes, then
/// one byte holding seven samples in `[7:1]`, then one holding the last sample in `[7]`.
///
/// # Panics
///
/// If `tx` is empty: a shift of zero bits is a caller error, not a bus condition.
#[must_use]
pub fn plan_capturing_shift(tx: &[u8]) -> Vec<Step> {
    assert!(!tx.is_empty(), "a data-register shift of zero bits");
    let n = tx.len();
    let mut steps = Vec::new();
    let mut prefix = vec![CLK_TMS, 0x02, 0x01]; // RTI -> Select-DR -> Capture-DR -> Shift-DR
    for chunk in tx[..n - 1].chunks(RETURN_BUFFER_SAFE) {
        let mut cmd = core::mem::take(&mut prefix);
        // MPSSE length fields are (count - 1).
        let len = (chunk.len() - 1) as u16;
        cmd.push(CLK_BYTES_IO);
        cmd.push(len as u8);
        cmd.push((len >> 8) as u8);
        cmd.extend_from_slice(chunk);
        cmd.push(SEND_IMMEDIATE);
        steps.push(Step { cmd, reads: chunk.len() });
    }
    let last = tx[n - 1];
    let mut cmd = prefix; // still holds the Shift-DR entry when `tx` was a single byte
    cmd.push(CLK_BITS_IO);
    cmd.push(6); // (count - 1): seven bits
    cmd.push(last & 0x7F);
    cmd.push(CLK_TMS_IO);
    cmd.push(0x00); // one clock, so the single TDO sample lands unambiguously in bit 7
    cmd.push(((last >> 7) << 7) | 0x01); // the last data bit, with TMS = 1 -> Exit1-DR
    cmd.push(CLK_TMS);
    cmd.push(0x01);
    cmd.push(0x01); // Update-DR, Run-Test/Idle
    cmd.push(SEND_IMMEDIATE);
    steps.push(Step { cmd, reads: 2 });
    steps
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The invariant the deadlock broke, on the shift that hit it: 36 frames plus the pad and
    /// dummy frames of a 7-series readback. The first version of this plan was ONE step asking for
    /// the whole reply; the chip cannot hold it, and nothing anywhere said so.
    #[test]
    fn no_step_asks_the_chip_to_hold_more_than_it_can() {
        let tx = vec![0u8; 38 * 101 * 4];
        let steps = plan_capturing_shift(&tx);
        // Two steps is what the uncut plan had too (the body, then the last byte), so "more than
        // one" would have passed on the defect. The bound below is the claim; this is its shadow.
        assert!(steps.len() > 2, "a reply larger than the buffer must be cut, not sent whole");
        for (i, s) in steps.iter().enumerate() {
            assert!(s.reads <= RETURN_BUFFER_SAFE, "step {i} generates {} reply bytes", s.reads);
            assert_eq!(s.cmd.last(), Some(&SEND_IMMEDIATE), "step {i} must flush its reply");
        }
        let total: usize = steps.iter().map(|s| s.reads).sum();
        assert_eq!(total, tx.len() - 1 + 2, "whole bytes, then the 7-bit byte, then the TMS byte");
    }

    /// Cutting the shift must not change what is shifted. Walk the commands as the chip would
    /// and recover the payload.
    #[test]
    fn the_plan_shifts_exactly_the_bytes_it_was_given() {
        let mut tx: Vec<u8> = (0..2_500u32).map(|i| (i * 37 + 11) as u8).collect();
        // THE LAST BYTE'S TOP BIT MUST BE SET. It is the one bit that does not travel as data: it
        // rides the TMS clock that leaves Shift-DR. The arithmetic fixture above happens to end in
        // 0x3A, and with that a plan that drops the bit entirely walked back to the same payload.
        *tx.last_mut().expect("non-empty") = 0xC3;
        let steps = plan_capturing_shift(&tx);
        assert_eq!(&steps[0].cmd[..3], &[CLK_TMS, 0x02, 0x01], "the shift is entered exactly once");
        let mut payload = Vec::new();
        let mut entries = 0;
        for s in &steps {
            let c = &s.cmd;
            let mut i = 0;
            while i < c.len() {
                match c[i] {
                    CLK_BYTES_IO => {
                        let len = usize::from(c[i + 1]) + (usize::from(c[i + 2]) << 8) + 1;
                        payload.extend_from_slice(&c[i + 3..i + 3 + len]);
                        i += 3 + len;
                    }
                    CLK_BITS_IO => {
                        assert_eq!(c[i + 1], 6, "seven bits, written as count - 1");
                        payload.push(c[i + 2]);
                        i += 3;
                    }
                    CLK_TMS_IO => {
                        let top = c[i + 2] & 0x80;
                        let low7 = payload.pop().expect("the seven bits came first");
                        payload.push(low7 | top);
                        i += 3;
                    }
                    CLK_TMS => {
                        if c[i + 1] == 0x02 {
                            entries += 1;
                        }
                        i += 3;
                    }
                    SEND_IMMEDIATE => i += 1,
                    other => panic!("an opcode this plan never emits: {other:#04x}"),
                }
            }
        }
        assert_eq!(entries, 1, "Shift-DR must be entered once, not once per piece");
        assert_eq!(payload, tx);
    }

    /// A one-byte shift has no whole bytes at all, and must still enter and leave Shift-DR.
    #[test]
    fn a_single_byte_is_the_last_byte() {
        let steps = plan_capturing_shift(&[0xA5]);
        assert_eq!(steps.len(), 1);
        assert_eq!(steps[0].reads, 2);
        assert_eq!(&steps[0].cmd[..3], &[CLK_TMS, 0x02, 0x01]);
        assert_eq!(steps[0].cmd[5], 0xA5 & 0x7F);
        assert_eq!(steps[0].cmd[8], 0x80 | 0x01, "bit 7 rides the TMS clock with TMS high");
    }
}
