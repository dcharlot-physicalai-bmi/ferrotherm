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
//! // What those sweeps WOULD cost on a Z1-class device. `joules` returns an Option because a
//! // device whose per-operation energy nobody has published has no answer here -- and borrowing
//! // another device's prices produces a figure indistinguishable from a measured one.
//! assert!(led.joules(&Z1_SPICE).unwrap() > 0.0);
//! assert_eq!(led.joules(&ferrotherm::ledger::Prices::UNSTATED), None);
//! ```
//!
//! # What "deterministic by seed" does and does not promise
//!
//! Measured across three machines -- macOS/arm64 (Apple M5 Max), `Linux/x86_64` (AMD EPYC 9R14) and
//! Linux/aarch64 (Graviton3), all on rustc 1.97.1 -- running the identical program:
//!
//! | | macOS arm64 | Linux `x86_64` | Linux aarch64 |
//! |---|---|---|---|
//! | compiled `.ftp` program | identical | identical | identical |
//! | CSR neighbour order | identical | identical | identical |
//! | **sampled state** | identical | identical | identical |
//! | `exp()` and the sigmoid | identical | identical | identical |
//! | energy computed from that state | `..a7b3` | `..a7b2` | `..a7b2` |
//!
//! **The answer is bit-reproducible.** The state a seed produces is the same on every platform
//! tested, which is what the promise is for: a run can be repeated and a result checked.
//!
//! **A derived float may differ by one ULP across operating systems.** The two Linux boxes agree
//! with each other across DIFFERENT architectures, and macOS disagrees with Linux on the SAME
//! architecture -- so it is not architecture. It is not libm either: `exp` and the sigmoid were
//! measured bit-identical on both. It is floating-point contraction, `w * s_i * s_j` accumulating
//! through an `fma` on one target and a separate multiply and add on another, which round
//! differently. Same values, same order, one bit apart.
//!
//! So: compare states, hashes and programs with `==`; compare energies with a tolerance. A test
//! asserting bit-equality of a derived float across platforms asserts something this crate does not
//! promise and could only deliver by disabling contraction everywhere, which costs more than the
//! property is worth.
//!
//! Scope note: binary (pbit) nodes with pairwise couplings are the sampling core. Categorical
//! and continuous nodes arrive through the program layer ([`program`]) and the thermodynamic
//! linear-algebra module ([`tla`]); the compiler ([`compile`]) targets device topologies.

pub mod access;
pub mod adaptive;
pub mod analog;
pub mod apps;
pub mod autocorr;
pub mod barrier;
pub mod bayes;
pub mod bls;
pub mod bound;
pub mod branch;
pub mod calibration;
pub mod categorical;
pub mod certify;
pub mod cftp;
pub mod cim;
pub mod cluster;
pub mod compile;
pub mod conform;
pub mod continuous;
pub mod corpora;
pub mod cuts;
pub mod decomp;
pub mod delay;
pub mod dense;
pub mod dense_memory;
pub mod device;
pub mod diffuse;
pub mod dimacs;
pub mod dimod;
pub mod dsisa;
pub mod dtm;
pub mod duty;
pub mod ebm;
pub mod embed;
pub mod encode;
pub mod ensembles;
pub mod entropy;
pub mod ep;
pub mod eqprop;
pub mod exact;
pub mod fabric;
pub mod factor;
pub mod ffi;
pub mod fft;
pub mod floors;
pub mod free_energy;
pub mod freefermion;
pub mod fss;
pub mod ftp;
pub mod gauge;
pub mod gbp;
pub mod gf2;
pub mod gibbs;
pub mod graph;
pub mod grid;
pub mod gset;
pub mod hdl;
pub mod het;
pub mod hfs;
pub mod hmc;
pub mod hopfield;
pub mod host;
pub mod hubo;
pub mod hybrid;
pub mod icm;
pub mod informed;
pub mod invertible;
pub mod ising;
pub mod kernel;
pub mod kmc;
pub mod kuramoto;
pub mod kwsample;
pub mod landauer;
pub mod ldpc;
pub mod ledger;
pub mod length;
pub mod linalg;
pub mod lns;
pub mod logdomain;
pub mod logit;
pub mod lp;
pub mod lrw;
pub mod matching;
pub mod mbar;
pub mod meanfield;
pub mod memcomp;
pub mod model;
pub mod mpemba;
pub mod mppi;
pub mod mps;
pub mod mrf;
pub mod multiflip;
pub mod multispin;
pub mod nce;
pub mod nested;
pub mod nonrev;
pub mod npising;
pub mod oim;
pub mod ommx;
pub mod oracle;
pub mod partition;
pub mod pathwise;
pub mod pdit;
pub mod perceptron;
pub mod perturb;
pub mod pfaffian;
pub mod phasegen;
pub mod planar;
pub mod planarcut;
pub mod planted;
pub mod pointproc;
pub mod popanneal;
pub mod portfolio;
pub mod potts;
pub mod precision;
pub mod program;
pub mod qaoa;
pub mod receipt;
pub mod reduce;
pub mod region;
pub mod relax;
pub mod relink;
pub mod restart;
pub mod restore;
pub mod rhat;
pub mod rld;
pub mod rng;
pub mod roofdual;
pub mod round;
pub mod samples;
pub mod sbm;
pub mod schedule;
pub mod sdp;
pub mod sos;
pub mod sourlas;
pub mod sparsify;
pub mod sqa;
pub mod sse;
pub mod stein;
pub mod structured;
pub mod survey;
pub mod syndrome;
pub mod tabu;
pub mod targets;
pub mod tempered_cd;
pub mod tempering;
pub mod tensor;
pub mod tla;
pub mod tmmc;
pub mod torque;
pub mod trw;
pub mod tts;
pub mod umbrella;
pub mod vmc;
pub mod wanglandau;
pub mod wgsl;
pub mod wire;
pub mod working;
pub mod writable;

/// The README's own code blocks, compiled by `cargo test`.
///
/// Its headline "Use it" snippet did not compile: `Ledger::joules` returns `Option<f64>` and the
/// snippet formatted it with `{:.2e}`, which `Option` does not implement. That is the first thing
/// anyone copies, and nothing was checking it -- a README is documentation the compiler can read,
/// so it should.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeDoc;
