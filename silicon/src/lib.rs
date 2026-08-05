//! ferrotherm-silicon — the IPAI-owned silicon layer for the ferrotherm deployment ladder.
//!
//! Pure Rust, no vendor tools, no third-party stacks. Selected algorithms ported with permission
//! from Open Interface Engineering's openie-fpga; this crate is fully independent of it.

pub mod lut;

#[cfg(feature = "flash")]
pub mod flash;
