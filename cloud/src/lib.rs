//! Cloud fabrics, reached through the same [`ferrotherm::fabric::Device`] trait as a CPU.
//!
//! A separate crate because it needs a TLS client and the core crate has no dependencies. Nothing
//! here is named by the core; deleting this crate leaves ferrotherm intact.

pub mod hitachi;
