//! Board access over USB — pure Rust, no vendor tools, no libusb.
//!
//! FT2232H MPSSE JTAG for the Alchitry Artix boards (Au V2 / Pt V2): the FTDI vendor-request
//! init sequence, the MPSSE JTAG TAP primitives, and IDCODE readout. Ported with permission from
//! Open Interface Engineering's openie-fpga and re-implemented independently here; the USB
//! transport is the pure-Rust `nusb` crate (no C library).
//!
//! Pin map (FT2232H interface A): TCK=AD0, TDI=AD1, TDO=AD2 (input), TMS=AD3 (idle high).

use nusb::transfer::{ControlOut, ControlType, Recipient};

const FTDI_EP_OUT: u8 = 0x02;
const FTDI_EP_IN: u8 = 0x81;
const FTDI_SIO_RESET: u8 = 0x00;
const FTDI_SIO_SET_LATENCY: u8 = 0x09;
const FTDI_SIO_SET_BITMODE: u8 = 0x0B;
const FTDI_INDEX_A: u16 = 1;
const BITMODE_RESET: u16 = 0x0000;
const BITMODE_MPSSE: u16 = 0x0200;

const MPSSE_LOOPBACK_DIS: u8 = 0x85;
const MPSSE_DIS_DIV5: u8 = 0x8A;
const MPSSE_SET_DIVISOR: u8 = 0x86;
const MPSSE_SET_LOW: u8 = 0x80;
const MPSSE_SEND_IMMEDIATE: u8 = 0x87;
const CLK_TMS: u8 = 0x4B; // clock TMS bits out, LSB first, -ve edge
const CLK_BYTES_IN: u8 = 0x28; // clock data bytes in, +ve edge, LSB order per bit

const PIN_DIR: u8 = 0x0B; // TCK, TDI, TMS out; TDO in
const PIN_IDLE: u8 = 0x08; // TMS high

pub struct Ftdi {
    iface: nusb::Interface,
}

impl Ftdi {
    /// Find and claim an FT2232H whose product string contains `needle` (e.g. "Alchitry").
    /// Returns the device product string alongside the handle.
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
        self.ctrl(FTDI_SIO_RESET, 1, FTDI_INDEX_A)?; // purge RX
        self.ctrl(FTDI_SIO_RESET, 2, FTDI_INDEX_A)?; // purge TX
        Ok(())
    }

    pub fn write(&self, data: &[u8]) -> Result<(), String> {
        let c = pollster::block_on(self.iface.bulk_out(FTDI_EP_OUT, data.to_vec()));
        c.status.map_err(|e| format!("bulk out: {e:?}"))
    }

    /// Read exactly `n` MPSSE payload bytes (each USB packet carries a 2-byte modem status
    /// header, stripped here).
    pub fn read(&self, n: usize) -> Result<Vec<u8>, String> {
        let mut out = Vec::with_capacity(n);
        let mut spins = 0;
        while out.len() < n {
            let c = pollster::block_on(self.iface.bulk_in(FTDI_EP_IN, nusb::transfer::RequestBuffer::new(512)));
            c.status.map_err(|e| format!("bulk in: {e:?}"))?;
            let data = c.data;
            if data.len() > 2 {
                out.extend_from_slice(&data[2..]);
            }
            spins += 1;
            if spins > 200 {
                return Err(format!("read timeout: got {} of {n}", out.len()));
            }
        }
        out.truncate(n);
        Ok(out)
    }
}

/// Minimal JTAG TAP over MPSSE, sufficient for chain identification.
pub struct Tap {
    pub ftdi: Ftdi,
}

impl Tap {
    pub fn new(ftdi: Ftdi) -> Result<Tap, String> {
        // 60 MHz base, loopback off, TCK = 60/((1+14)*2) = 2 MHz, pins idle
        ftdi.write(&[
            MPSSE_DIS_DIV5,
            MPSSE_LOOPBACK_DIS,
            MPSSE_SET_DIVISOR, 14, 0x00,
            MPSSE_SET_LOW, PIN_IDLE, PIN_DIR,
        ])?;
        Ok(Tap { ftdi })
    }

    /// TAP reset then read the 32-bit IDCODE (the default DR after reset on Xilinx parts).
    pub fn idcode(&mut self) -> Result<u32, String> {
        let mut cmds = vec![
            CLK_TMS, 0x05, 0xFF, // 6 clocks TMS=1: Test-Logic-Reset
            CLK_TMS, 0x00, 0x00, // 1 clock TMS=0: Run-Test/Idle
            CLK_TMS, 0x02, 0x01, // 3 clocks TMS=1,0,0: Select-DR, Capture-DR, Shift-DR
            CLK_BYTES_IN, 0x03, 0x00, // clock 4 bytes in
            MPSSE_SEND_IMMEDIATE,
        ];
        self.ftdi.write(&cmds)?;
        let raw = self.ftdi.read(4)?;
        cmds.clear();
        Ok(u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]))
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
