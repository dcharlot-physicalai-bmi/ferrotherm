//! Board access over USB — pure Rust, no vendor tools, no C libraries.
//!
//! FT2232H MPSSE JTAG for the Alchitry Artix boards (Au V2 / Pt V2): FTDI vendor-request init,
//! a real TAP state machine, IDCODE, and the 7-series configuration port (CFG_IN / CFG_OUT).
//! Ported with permission from Open Interface Engineering's openie-fpga and re-implemented
//! independently here; the USB transport is the pure-Rust `nusb` crate.
//!
//! Pin map (FT2232H interface A): TCK=AD0, TDI=AD1, TDO=AD2 (input), TMS=AD3 (idle high).

use nusb::transfer::{ControlOut, ControlType, Recipient, RequestBuffer};

const FTDI_EP_OUT: u8 = 0x02;
const FTDI_EP_IN: u8 = 0x81;
const FTDI_SIO_RESET: u8 = 0x00;
const FTDI_SIO_SET_LATENCY: u8 = 0x09;
const FTDI_SIO_SET_BITMODE: u8 = 0x0B;
const FTDI_INDEX_A: u16 = 1;
const BITMODE_RESET: u16 = 0x0000;
const BITMODE_MPSSE: u16 = 0x0200;

// MPSSE opcode bits: 0x01 -ve write, 0x02 bit mode, 0x04 -ve read, 0x08 LSB first,
// 0x10 do-write, 0x20 do-read, 0x40 TMS mode.
const CLK_BYTES_OUT: u8 = 0x19;
const CLK_BITS_OUT: u8 = 0x1B;
const CLK_BYTES_IO: u8 = 0x39;
const CLK_BITS_IO: u8 = 0x3B;
const CLK_TMS: u8 = 0x4B;
const CLK_TMS_IO: u8 = 0x6B;
const MPSSE_LOOPBACK_DIS: u8 = 0x85;
const MPSSE_DIS_DIV5: u8 = 0x8A;
const MPSSE_SET_DIVISOR: u8 = 0x86;
const MPSSE_SET_LOW: u8 = 0x80;
const MPSSE_SEND_IMMEDIATE: u8 = 0x87;

const PIN_DIR: u8 = 0x0B; // TCK, TDI, TMS out; TDO in
const PIN_IDLE: u8 = 0x08; // TMS high

// 7-series 6-bit IR opcodes (UG470).
pub const IR_LEN: u8 = 6;
pub const IR_IDCODE: u8 = 0x09;
pub const IR_JPROGRAM: u8 = 0x0B;
pub const IR_CFG_IN: u8 = 0x05;
pub const IR_CFG_OUT: u8 = 0x04;
pub const IR_JSTART: u8 = 0x0C;
pub const IR_BYPASS: u8 = 0x3F;

pub struct Ftdi {
    iface: nusb::Interface,
}

impl Ftdi {
    /// Find and claim an FT2232H whose product string contains `needle` (e.g. "Alchitry").
    pub fn open(needle: &str) -> Result<(Ftdi, String), String> {
        let devs = nusb::list_devices().map_err(|e| format!("usb list: {e}"))?;
        for d in devs {
            if d.vendor_id() == 0x0403 && d.product_id() == 0x6010 {
                let prod = d.product_string().unwrap_or("").to_string();
                if prod.contains(needle) {
                    let dev = d.open().map_err(|e| format!("open: {e}"))?;
                    let iface = dev.claim_interface(0).map_err(|e| format!("claim: {e}"))?;
                    let f = Ftdi { iface };
                    f.enable_mpsse()?;
                    return Ok((f, prod));
                }
            }
        }
        Err(format!("no FT2232H with product containing '{needle}' found"))
    }

    fn ctrl(&self, request: u8, value: u16, index: u16) -> Result<(), String> {
        let c = pollster::block_on(self.iface.control_out(ControlOut {
            control_type: ControlType::Vendor,
            recipient: Recipient::Device,
            request,
            value,
            index,
            data: &[],
        }));
        c.status.map_err(|e| format!("ctrl 0x{request:02X}: {e:?}"))
    }

    fn enable_mpsse(&self) -> Result<(), String> {
        self.ctrl(FTDI_SIO_RESET, 0, FTDI_INDEX_A)?;
        self.ctrl(FTDI_SIO_SET_BITMODE, BITMODE_RESET, FTDI_INDEX_A)?;
        self.ctrl(FTDI_SIO_SET_BITMODE, BITMODE_MPSSE, FTDI_INDEX_A)?;
        self.ctrl(FTDI_SIO_SET_LATENCY, 16, FTDI_INDEX_A)?;
        self.ctrl(FTDI_SIO_RESET, 1, FTDI_INDEX_A)?;
        self.ctrl(FTDI_SIO_RESET, 2, FTDI_INDEX_A)?;
        Ok(())
    }

    pub fn write(&self, data: &[u8]) -> Result<(), String> {
        for chunk in data.chunks(4096) {
            let c = pollster::block_on(self.iface.bulk_out(FTDI_EP_OUT, chunk.to_vec()));
            c.status.map_err(|e| format!("bulk out: {e:?}"))?;
        }
        Ok(())
    }

    /// Read exactly `n` MPSSE payload bytes (each USB packet carries a 2-byte status header).
    pub fn read(&self, n: usize) -> Result<Vec<u8>, String> {
        let mut out = Vec::with_capacity(n);
        let mut spins = 0;
        while out.len() < n {
            let c = pollster::block_on(
                self.iface.bulk_in(FTDI_EP_IN, RequestBuffer::new(512)),
            );
            c.status.map_err(|e| format!("bulk in: {e:?}"))?;
            if c.data.len() > 2 {
                out.extend_from_slice(&c.data[2..]);
            }
            spins += 1;
            if spins > 400 {
                return Err(format!("read timeout: got {} of {n}", out.len()));
            }
        }
        out.truncate(n);
        Ok(out)
    }
}

/// Reverse the bit order of a byte — the JTAG config port shifts LSB-first while configuration
/// packets are defined MSB-first, so every payload byte is reversed on the way out and back.
pub fn reverse_byte(mut b: u8) -> u8 {
    b = (b & 0xF0) >> 4 | (b & 0x0F) << 4;
    b = (b & 0xCC) >> 2 | (b & 0x33) << 2;
    (b & 0xAA) >> 1 | (b & 0x55) << 1
}

/// A JTAG TAP driver over MPSSE. Tracks nothing implicitly: every operation starts and ends in
/// Run-Test/Idle, so sequences that must not be interrupted (CFG_IN then CFG_OUT) stay valid.
pub struct Tap {
    pub ftdi: Ftdi,
}

impl Tap {
    pub fn new(ftdi: Ftdi) -> Result<Tap, String> {
        // 60 MHz base clock, loopback off, TCK = 60/((1+14)*2) = 2 MHz, pins idle.
        ftdi.write(&[
            MPSSE_DIS_DIV5,
            MPSSE_LOOPBACK_DIS,
            MPSSE_SET_DIVISOR, 14, 0x00,
            MPSSE_SET_LOW, PIN_IDLE, PIN_DIR,
        ])?;
        let mut t = Tap { ftdi };
        t.reset()?;
        Ok(t)
    }

    /// Test-Logic-Reset, then Run-Test/Idle.
    pub fn reset(&mut self) -> Result<(), String> {
        self.ftdi.write(&[CLK_TMS, 0x05, 0xFF, CLK_TMS, 0x00, 0x00])
    }

    /// Shift an instruction (6 bits on 7-series), returning to Run-Test/Idle.
    pub fn shift_ir(&mut self, ir: u8) -> Result<(), String> {
        let last = (ir >> (IR_LEN - 1)) & 1;
        self.ftdi.write(&[
            CLK_TMS, 0x03, 0x03,                       // RTI -> Shift-IR (TMS 1,1,0,0)
            CLK_BITS_OUT, IR_LEN - 2, ir & 0x1F,       // first 5 bits
            CLK_TMS, 0x02, (last << 7) | 0x03,         // last bit + Exit1,Update,RTI
        ])
    }

    /// Shift `tx` (whole bytes) through DR. When `capture`, the same number of bytes is read
    /// back. Returns to Run-Test/Idle.
    pub fn shift_dr(&mut self, tx: &[u8], capture: bool) -> Result<Vec<u8>, String> {
        assert!(!tx.is_empty());
        let n = tx.len();
        let mut cmds = vec![CLK_TMS, 0x02, 0x01]; // RTI -> Shift-DR (TMS 1,0,0)
        if n > 1 {
            let len = (n - 2) as u16; // bytes except the last one, minus 1
            cmds.push(if capture { CLK_BYTES_IO } else { CLK_BYTES_OUT });
            cmds.push(len as u8);
            cmds.push((len >> 8) as u8);
            cmds.extend_from_slice(&tx[..n - 1]);
        }
        let last = tx[n - 1];
        // Bits 0..=6 of the final byte here; bit 7 must ride the TMS clock that exits Shift-DR,
        // because the last data bit and the Exit1 transition happen on the same edge.
        // LENGTH FIELDS ARE (count - 1): 6 means SEVEN bits. (Writing 5 here shifted 31 bits of
        // a 32-bit register and silently corrupted the MSB — caught by an IDCODE whose top
        // nibble changed between runs.)
        cmds.push(if capture { CLK_BITS_IO } else { CLK_BITS_OUT });
        cmds.push(6);
        cmds.push(last & 0x7F);
        if capture {
            // ONE clock, so the single TDO sample is unambiguously in bit 7 of the returned
            // byte (FTDI left-aligns bit-mode reads: n samples occupy [7 : 8-n]).
            cmds.push(CLK_TMS_IO);
            cmds.push(0x00);
            cmds.push(((last >> 7) << 7) | 0x01); // last data bit, TMS=1 -> Exit1-DR
            cmds.push(CLK_TMS);
            cmds.push(0x01);
            cmds.push(0x01); // Update-DR, Run-Test/Idle
            cmds.push(MPSSE_SEND_IMMEDIATE);
        } else {
            cmds.push(CLK_TMS);
            cmds.push(0x02);
            cmds.push(((last >> 7) << 7) | 0x03); // last data bit + Exit1, Update, RTI
        }
        self.ftdi.write(&cmds)?;
        if !capture {
            return Ok(Vec::new());
        }
        // reads: (n-1) whole bytes, one 7-bit byte, one 1-bit TMS byte
        let want = if n > 1 { n - 1 } else { 0 } + 2;
        let raw = self.ftdi.read(want)?;
        let mut out = Vec::with_capacity(n);
        if n > 1 {
            out.extend_from_slice(&raw[..n - 1]);
        }
        let bits7 = raw[want - 2] >> 1; // 7 samples land in [7:1]
        let tms_bit = (raw[want - 1] >> 7) & 1; // 1 sample lands in [7]
        out.push((bits7 & 0x7F) | (tms_bit << 7));
        Ok(out)
    }

    /// Read the 32-bit IDCODE (the DR selected after Test-Logic-Reset).
    pub fn idcode(&mut self) -> Result<u32, String> {
        self.reset()?;
        let rx = self.shift_dr(&[0; 4], true)?;
        Ok(u32::from_le_bytes([rx[0], rx[1], rx[2], rx[3]]))
    }

    /// Write configuration words through CFG_IN (words are sent MSB-first, bit-reversed per byte).
    pub fn cfg_in_words(&mut self, words: &[u32]) -> Result<(), String> {
        let mut payload = Vec::with_capacity(words.len() * 4);
        for w in words {
            for b in w.to_be_bytes() {
                payload.push(reverse_byte(b));
            }
        }
        self.shift_ir(IR_CFG_IN)?;
        self.shift_dr(&payload, false)?;
        Ok(())
    }

    /// Read one 32-bit word back through CFG_OUT.
    pub fn cfg_out_word(&mut self) -> Result<u32, String> {
        self.shift_ir(IR_CFG_OUT)?;
        let rx = self.shift_dr(&[0; 4], true)?;
        Ok(u32::from_be_bytes([
            reverse_byte(rx[0]),
            reverse_byte(rx[1]),
            reverse_byte(rx[2]),
            reverse_byte(rx[3]),
        ]))
    }

    /// Read a 7-series configuration register (non-destructive; STAT = 7).
    pub fn read_config_reg(&mut self, reg: u32) -> Result<u32, String> {
        let read_cmd = 0x2800_0000 | ((reg & 0x3FFF) << 13) | 1; // type-1 read, 1 word
        self.reset()?;
        self.cfg_in_words(&[
            0xFFFF_FFFF, // dummy
            0xAA99_5566, // sync
            0x2000_0000, // NOOP
            read_cmd,
            0x2000_0000,
            0x2000_0000,
        ])?;
        self.cfg_out_word()
    }
}

/// An explicit acknowledgement that an operation will clear the design currently running on the
/// fabric. Reconfiguration cannot be requested by accident: the caller must name the consequence.
pub enum Consequence {
    ClearsTheRunningDesign,
}

impl Tap {
    /// Load a raw configuration payload into fabric SRAM: JPROGRAM (clear), CFG_IN (the payload),
    /// JSTART (release the startup sequence), then report the status register.
    ///
    /// This ERASES whatever is currently configured. On a board that boots from SPI flash the
    /// stored image is untouched, but the live fabric stays as loaded here until the next power
    /// cycle — which is why the caller must pass [`Consequence::ClearsTheRunningDesign`].
    pub fn configure(&mut self, config: &[u8], _ack: Consequence) -> Result<Stat, String> {
        self.reset()?;
        // clear configuration memory
        self.shift_ir(IR_JPROGRAM)?;
        self.run_clocks(120_000)?; // config-clear settle
        // shift the payload through the configuration port
        let mut payload = Vec::with_capacity(config.len());
        for b in config {
            payload.push(reverse_byte(*b));
        }
        self.shift_ir(IR_CFG_IN)?;
        self.shift_dr(&payload, false)?;
        // release the startup sequencer
        self.shift_ir(IR_JSTART)?;
        self.run_clocks(2_000)?;
        self.shift_ir(IR_BYPASS)?;
        Ok(Stat(self.read_config_reg(crate::bitstream::reg::STAT)?))
    }

    /// Hold the TAP in Run-Test/Idle for `n` TCK cycles.
    pub fn run_clocks(&mut self, n: usize) -> Result<(), String> {
        let mut left = n;
        while left > 0 {
            let batch = left.min(6);
            self.ftdi.write(&[CLK_TMS, (batch - 1) as u8, 0x00])?;
            left -= batch;
        }
        Ok(())
    }
}

/// 7-series STAT register (UG470 Table 5-25) — the bits worth naming.
pub struct Stat(pub u32);

impl Stat {
    pub fn crc_error(&self) -> bool { self.0 & 1 != 0 }
    pub fn dec_error(&self) -> bool { self.0 >> 1 & 1 != 0 }
    pub fn id_error(&self) -> bool { self.0 >> 2 & 1 != 0 }
    pub fn done(&self) -> bool { self.0 >> 3 & 1 != 0 }
    pub fn release_done(&self) -> bool { self.0 >> 4 & 1 != 0 }
    pub fn init_b(&self) -> bool { self.0 >> 5 & 1 != 0 }
    pub fn init_complete(&self) -> bool { self.0 >> 6 & 1 != 0 }
    pub fn mode(&self) -> u8 { (self.0 >> 7 & 0x7) as u8 }
    pub fn eos(&self) -> bool { self.0 >> 13 & 1 != 0 }
    pub fn part_secured(&self) -> bool { self.0 >> 16 & 1 != 0 }
    pub fn describe(&self) -> String {
        format!(
            "STAT=0x{:08X}  DONE={} EOS={} INIT_B={} INIT_COMPLETE={} MODE={:03b} CRC_ERR={}",
            self.0,
            self.done() as u8,
            self.eos() as u8,
            self.init_b() as u8,
            self.init_complete() as u8,
            self.mode(),
            self.crc_error() as u8
        )
    }
}

/// Decode a 7-series IDCODE (version nibble masked) to a part name.
pub fn xc7_part(idcode: u32) -> Option<&'static str> {
    match idcode & 0x0FFF_FFFF {
        0x0362D093 => Some("XC7A35T"),
        0x03631093 => Some("XC7A100T"),
        0x03636093 => Some("XC7A200T"),
        0x03622093 => Some("XC7S25"),
        0x0364C093 => Some("XC7K325T"),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_reversal_is_an_involution() {
        for b in 0..=255u8 {
            assert_eq!(reverse_byte(reverse_byte(b)), b);
        }
        assert_eq!(reverse_byte(0b1000_0000), 0b0000_0001);
        assert_eq!(reverse_byte(0xAA), 0x55);
    }

    /// The type-1 read packet for STAT must be the documented 0x2800E001.
    #[test]
    fn stat_read_command_word() {
        let reg = 7u32;
        let cmd = 0x2800_0000 | ((reg & 0x3FFF) << 13) | 1;
        assert_eq!(cmd, 0x2800_E001);
    }

    #[test]
    fn idcode_decode() {
        assert_eq!(xc7_part(0x13631093), Some("XC7A100T"));
    }
}
