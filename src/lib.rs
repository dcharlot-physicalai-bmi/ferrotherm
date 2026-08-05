//! # ferrotherm — thermodynamic sampling for Physical AI
//!
//! The IPAI @ BMI equivalent of the thermodynamic-computing software stack, in pure Rust: sparse
//! energy-based models, chromatic block-Gibbs sampling, and — first-class, not an appendix — the
//! device energy ledger that prices every sample, read, and write in joules.
//!
//! The physics is old and open: Ising (1925), Glauber dynamics (1963), Gibbs sampling
//! (Geman & Geman 1984), graph-colored parallel sweeps (standard checkerboard decomposition).
//! What a "thermodynamic sampling unit" accelerates is exactly this loop; what it charges for is
//! I/O. Both belong in the open commons, runnable on every compute fabric — CPU today, WebGPU next,
//! physics-native silicon when it exists to measure.
//!
//! Design positions, each earned from a verified source:
//!  * **The ledger is first-class.** Extropic's Thermalizers appendix (arXiv:2608.01615, Table IV)
//!    prices a Z1-class node at 7.09 fJ per Gibbs cycle, 1.692 pJ per read, 153.6 pJ per write —
//!    a write costs ~21,700 samples. Any honest account of this hardware class is an I/O story,
//!    so every `ferrotherm` simulation carries a [`ledger::Ledger`] and reports what the device
//!    WOULD pay, crossings included.
//!  * **Sparse and 2-colorable is the native shape.** The published Z1 topology is a planar grid
//!    with odd-Manhattan couplings (degree 16, longest edge sqrt(17)), which is bipartite: one
//!    full sweep = two parallel half-sweeps. [`device::z1_grid`] reproduces it.
//!  * **Verification against exact physics before any claim.** The sampler must reproduce the
//!    exact Boltzmann distribution on enumerable systems and the Onsager magnetization on the 2D
//!    lattice before it is used for anything else. See `examples/ring_tv.rs`, `examples/onsager.rs`.
//!
//! ## Quickstart
//!
//! ```
//! use ferrotherm::{ising, gibbs::Sampler, ledger::{Ledger, Z1_SPICE}};
//!
//! // a 16x16 Ising magnet below its critical temperature
//! let g = ising::lattice2d(16, 1.0);
//! let mut led = Ledger::default();
//! let mut smp = Sampler::new(&g, 0.6, 42);
//! smp.sweeps(500, Some(&mut led));
//!
//! let m = (smp.s.iter().map(|&v| v as i64).sum::<i64>().abs() as f64) / g.n as f64;
//! assert!(m > 0.9, "ordered phase: |M| = {m}");
//! // what those sweeps WOULD cost on a Z1-class device (pre-silicon vendor prices)
//! assert!(led.joules(&Z1_SPICE) > 0.0);
//! ```
//!
//! Scope note: binary (pbit) nodes with pairwise couplings are the sampling core. Categorical
//! and continuous nodes arrive through the program layer ([`program`]) and the thermodynamic
//! linear-algebra module ([`tla`]); the compiler ([`compile`]) targets device topologies.

pub mod rng;
pub mod graph;
pub mod gibbs;
pub mod ising;
pub mod device;
pub mod ledger;
pub mod program;
pub mod compile;
pub mod tempering;
pub mod tla;
pub mod linalg;
pub mod het;
pub mod lrw;
pub mod sbm;
pub mod dtm;
pub mod targets;
pub mod hdl;
pub mod ffi;
