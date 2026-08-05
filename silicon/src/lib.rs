//! ferrotherm-silicon — the IPAI-owned silicon layer for the ferrotherm deployment ladder.
//!
//! Pure Rust, no vendor tools, no third-party stacks. Selected algorithms ported with permission
//! from Open Interface Engineering's openie-fpga; this crate is fully independent of it.

pub mod bitstream;
pub mod frame;
pub mod framebuf;
pub mod json;
pub mod pips;
pub mod route;
pub mod segbits;
pub mod tilegrid;
pub mod lut;

#[cfg(feature = "flash")]
pub mod flash;
