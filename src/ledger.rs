//! The device energy ledger — first-class, because on this hardware class the story IS the I/O.
//!
//! Prices are per-node-operation costs of a device model. For the Z1-class costs (SPICE-derived,
//! pre-silicon; arXiv:2608.01615 Table IV) a WRITE costs ~21,700 Gibbs cycles and a READ ~239.
//! The vendor's own conclusion follows from these three numbers: the architecture wins where
//! "many local updates are performed between infrequent I/O operations" — and the ledger makes
//! that arithmetic executable instead of promotional.

/// How much a per-operation energy figure is worth as evidence.
///
/// This exists because the field it describes has stopped distinguishing. A 2026 sweep of
/// specialised AI silicon found **no tokens-per-watt figure on a language model verified by any
/// independent third party, for any vendor** — the one power-measured result in the space is a
/// ResNet-50 submission from a company that has since shut down, and the only rule-governed harness
/// with a power methodology recorded zero power submissions in its most recent round. What
/// circulates instead is a mixture of metered, simulated, derived and projected numbers, quoted in
/// the same units and compared to one another without remark.
///
/// A `f64` cannot tell you which it is holding. [`Prices::source`] says so in prose, and prose does
/// not stop a ratio being taken. This does: [`weaker`] gives the grade any comparison inherits, so
/// a metered number divided by a projection is typed as a projection.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Evidence {
    /// No published or measured figure at all. The weakest grade, and the honest one.
    Unstated,
    /// A roadmap number or design target for hardware that does not exist yet.
    Projected,
    /// Computed analytically from stated parameters. Nothing was instrumented.
    Derived,
    /// Circuit simulation of a design — SPICE or equivalent. No silicon was measured.
    Simulated,
    /// Measured on physical hardware, without a fully stated measurement protocol.
    Measured,
    /// Metered on physical silicon with the protocol stated: instrument, baseline, and a
    /// reproduced control. The strongest grade, and the rarest.
    Metered,
}

impl core::fmt::Display for Evidence {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let s = match self {
            Evidence::Unstated => "unstated",
            Evidence::Projected => "projected",
            Evidence::Derived => "derived",
            Evidence::Simulated => "simulated",
            Evidence::Measured => "measured",
            Evidence::Metered => "metered",
        };
        f.write_str(s)
    }
}

/// The grade a comparison between two figures can carry: the weaker of the two.
///
/// A measured number divided by a projected one is a projection. Reporting the ratio at the
/// stronger grade is the single most common error in the literature this type was written against.
#[must_use]
pub fn weaker(a: Evidence, b: Evidence) -> Evidence {
    if a <= b { a } else { b }
}

/// The energy advantage that survives once the answer is **read out**.
///
/// `baseline_j / (dynamics_j + reads * e_read)`. An energy-advantage claim for a physical computer
/// is almost always a ratio of two *dynamics* figures: what the digital baseline burns computing,
/// against what the physical device dissipates evolving. The device's answer is then still inside
/// the device. Getting it out is a third term, and it is charged nowhere in the claim.
///
/// This is the forward direction. [`read_budget_for_advantage`] is its inverse, and
/// `the_readout_budget_and_the_advantage_are_inverses` round-trips them.
///
/// # Panics
///
/// If any energy is negative or non-finite, or if `dynamics_j + reads * e_read` is zero — a ratio
/// against nothing is not an advantage.
#[must_use]
pub fn advantage_after_readout(baseline_j: f64, dynamics_j: f64, reads: u64, e_read: f64) -> f64 {
    assert!(baseline_j.is_finite() && baseline_j >= 0.0, "baseline must be a non-negative energy, got {baseline_j}");
    assert!(dynamics_j.is_finite() && dynamics_j >= 0.0, "dynamics must be a non-negative energy, got {dynamics_j}");
    assert!(e_read.is_finite() && e_read >= 0.0, "a read price must be a non-negative energy, got {e_read}");
    let total = dynamics_j + reads as f64 * e_read;
    assert!(total > 0.0, "a claimed advantage needs something in the denominator");
    baseline_j / total
}

/// **The largest per-read energy at which a claimed advantage still holds** — the bound an
/// unpriced readout has to satisfy for the headline to be true.
///
/// Inverting [`advantage_after_readout`]: `(baseline_j / advantage - dynamics_j) / reads`. Divided
/// by [`crate::floors::readout_floor`] this becomes the useful, constant-free form — *"every value
/// read out must cost no more than N times its own Landauer floor"* — which can be checked against
/// any real device without agreeing on what that device is.
///
/// `None` when the claim cannot hold **at any readout cost**, free readout included: the dynamics
/// alone already exceed the budget the advantage allows.
///
/// # Panics
///
/// If `reads` is zero (a claim with no readout has no readout bound to report), if the advantage is
/// not positive and finite, or if either energy is negative or non-finite.
#[must_use]
pub fn read_budget_for_advantage(baseline_j: f64, dynamics_j: f64, reads: u64, advantage: f64) -> Option<f64> {
    assert!(reads > 0, "a claim that reads nothing has no readout bound");
    assert!(advantage > 0.0 && advantage.is_finite(), "an advantage is positive and finite, got {advantage}");
    assert!(baseline_j.is_finite() && baseline_j >= 0.0, "baseline must be a non-negative energy, got {baseline_j}");
    assert!(dynamics_j.is_finite() && dynamics_j >= 0.0, "dynamics must be a non-negative energy, got {dynamics_j}");
    let budget = baseline_j / advantage - dynamics_j;
    if budget > 0.0 { Some(budget / reads as f64) } else { None }
}

/// Per-operation energy prices, in joules. These describe a DEVICE MODEL, not measured silicon,
/// unless the source says otherwise; keep the provenance in the name.
#[derive(Clone, Copy, Debug)]
pub struct Prices {
    /// One Gibbs update of one node.
    pub e_sample: f64,
    /// One node value read out to the chip edge.
    pub e_read: f64,
    /// One node's couplings/bias/clamp state flashed.
    pub e_write: f64,
    /// Maximum sustainable full-graph reflash rate, Hz (None = unstated).
    ///
    /// Read by [`Ledger::reflash_seconds`]: a workload that reflashes the whole graph faster than
    /// the device can sustain is not a fast workload, it is an unphysical one, and a joules figure
    /// computed for it prices a run that could not happen.
    pub reflash_hz_cap: Option<f64>,
    /// WHAT these numbers describe, and where they came from.
    ///
    /// Not documentation. A joules figure is a claim about a machine, and a `Prices` without a
    /// subject can be applied to any machine at all — which is exactly what happened here: every
    /// fabric in the tree declared `Z1_SPICE`, so a Hitachi CMOS annealer and an FPGA both reported
    /// Extropic's pre-silicon SPICE estimates, and the HTTP surface reported them for a plain CPU
    /// run on a laptop. Nothing was lying; nothing had been asked to say whose numbers these were.
    pub source: &'static str,
    /// What kind of evidence [`Prices::source`] describes.
    ///
    /// Typed rather than left to the prose, so that a ratio across grades can be caught by the
    /// compiler's user rather than by a careful reader.
    pub evidence: Evidence,
}

impl Prices {
    /// No published or measured per-operation energy for this machine.
    ///
    /// [`Ledger::joules`] returns `None` for these rather than a number, because the alternative —
    /// borrowing another device's prices — produces a figure that looks exactly like a real one.
    /// An unstated cost is a fact about the world, and reporting it is more useful than a guess
    /// wearing a decimal point.
    pub const UNSTATED: Prices = Prices {
        e_sample: f64::NAN,
        e_read: f64::NAN,
        e_write: f64::NAN,
        reflash_hz_cap: None,
        source: "no published or measured per-operation energy for this machine",
        evidence: Evidence::Unstated,
    };

    /// Whether these prices describe anything.
    #[must_use = "false means these prices describe no machine, and pricing a run against them produces a figure that looks exactly like a real one"]
    pub fn is_stated(&self) -> bool {
        self.e_sample.is_finite() && self.e_read.is_finite() && self.e_write.is_finite()
    }
}

/// Z1-class prices from arXiv:2608.01615 Table IV (SPICE estimates for taped-out, uncharacterized
/// silicon; "measured" in the paper's prose is a misnomer the appendix itself contradicts).
pub const Z1_SPICE: Prices = Prices {
    e_sample: 7.09e-15,
    e_read: 1.692e-12,
    e_write: 153.6e-12,
    reflash_hz_cap: Some(1.0),
    // THE VENDOR PUBLISHES TWO. Table IV gives 7.09e-15 J per pbit per Gibbs cycle; the later Z1T
    // write-up gives 1.3e-14 J per sample, 1.83x apart, and neither is metered. The smaller, older
    // and more favourable-to-them figure is the one taken here, because a price set against a
    // competitor should be the one that flatters the competitor.
    source: "Z1-class SPICE estimates, arXiv:2608.01615 Table IV — taped-out but uncharacterised \
             silicon, not measured. Applies to that device model and to nothing else.",
    evidence: Evidence::Simulated,
};

/// **MEASURED** per-sample energy of this crate's own p-bit fabric on real silicon.
///
/// Every other `Prices` in this crate describes a device MODEL. This one describes a board that ran
/// the workload, on a wattmeter, and it is the only figure here that was not projected by anybody.
///
/// # What was measured
///
/// A Kria KV260 (`xck26-sfvc784-2LV-c`), 2026-09-06. [`crate::hdl::FixedFabric::emit_verilog`]
/// emitted a 1,024 p-bit fabric — `lattice2d(32)`, Q.8 weights, 1024-entry sigmoid ROM, one
/// xorshift32 per node — implemented in Vivado 2026.1 against the PS's `pl_clk0` at 100 MHz and
/// flashed through `fpga_manager`. Board power came from the SOM's own INA260 on the 5 V rail, so
/// the scope is **whole board including regulator loss**: an upper bound on the die, and the same
/// convention `ferrotherm_meter` uses everywhere else.
///
/// ```text
///   PL idle      3.1362 W  +- 0.0848   (24 samples)
///   fabric on    3.6917 W  +- 0.0815   (24 samples)
///   delta        0.5554 W  se 0.0240   -> 23.1 sigma
///   throughput   1024 p-bits x 100 MHz / 2 clocks per sweep = 51.2 flips/ns
///   ==>          10.85 +- 0.47 pJ per single-node update
/// ```
///
/// # It survived a re-measurement
///
/// A single reading is not reproducible until it has been taken twice. A second, independently
/// dmesg-verified flash of the same bitstream read **3.6993 W +- 0.0847** against the first run's
/// **3.6917 W +- 0.0815** — 7.6 mW apart, inside the noise on either. Across the two flashes the
/// delta is **0.559 +- 0.008 W**, a 1.4% run-to-run spread, and the per-flip figure moves from
/// 10.85 to 10.9 pJ. The constant keeps the first run's value; the second is what says it is real.
///
/// # Three things that make it a measurement rather than a plausible number
///
/// 1. **The instrument was validated before it was trusted.** Loading the four A53 cores moved the
///    same sensor by +0.778 W against an idle spread of 0.087 W. A sensor that does not respond to
///    load makes every figure downstream unfalsifiable, and that check is cheap.
/// 2. **The flash was verified by `dmesg`, not by `state`.** Writing a filename to
///    `/sys/class/fpga_manager/fpga0/firmware` **silently no-ops** when `/lib/firmware/<f>` is
///    missing: the old bitstream keeps running and `state` still reads `operating`. The run that
///    produced this number carries `writing ft_fabric.bit.bin to Xilinx ZynqMP FPGA Manager` with no
///    error. An earlier attempt here loaded nothing and honestly reported `-0.006 W`.
/// 3. **The logic had to be kept alive.** With no observable sink the synthesiser dead-code-
///    eliminates the entire fabric to about one LUT, and the board then measures a correct zero for
///    a design that is not there. The vehicle carries `DONT_TOUCH` and folds all 1,024 state bits
///    into one registered bit.
///
/// # The finding worth carrying
///
/// Vivado's own estimator, run on the implemented design, put the PL at **0.100 W** — clocks 0.077,
/// CLB logic 0.016, signals 0.007. The board drew **0.5554 W**. The vendor tool under-predicted by
/// **5.6x**, and the reason generalises: without a switching-activity file it assumes a default
/// toggle rate near 12.5%, while this fabric IS a randomness engine — 1,024 xorshift32 generators
/// each flipping about half of 32 bits every clock. Every projected p-bit energy in this field is a
/// model of that kind, and here is one caught under-predicting a stochastic sampler in the direction
/// that flatters it.
///
/// Against [`Z1_SPICE`]'s projected `7.09e-15` J per sample, this measured `1.085e-11` is **~1,530x**
/// more per flip. That is the honest shape of the gap between 16 nm FPGA CMOS and a thermodynamic
/// p-bit.
///
/// **CORRECTED 2026-09-18.** This paragraph used to end "and it is the first entry in that
/// comparison that is not itself a projection." A global literature sweep retired that sentence.
/// [`PEGASUS_28NM_ASIC`] is fabricated silicon whose authors state `1.2e-12` J per update
/// including I/O — about **9x below** this figure — and arXiv:2606.25313 (June 2026, before this
/// measurement) reports the wall power of two multi-FPGA p-bit machines. What this entry still is:
/// the one with a stated metering protocol — instrument, idle baseline, a reproduced control —
/// and an open path from a library call to the bitstream that was metered.
///
/// It is also an INCREMENT: `0.5554 W` above an idle PL. The whole board drew `3.6917 W` with the
/// fabric on, which over `5.12e10` flips per second is **`7.21e-11` J per flip** — the figure to
/// hold against anyone else's wall power, and nearly seven times this one.
///
/// # Why reads and writes are unstated
///
/// They were not measured. This run clocked a free-running fabric with no host traffic, so
/// [`Ledger::joules`] **refuses** any workload that touches them rather than pricing them at zero.
/// That refusal is the point of the type. It was not repaired by editing this constant: the
/// bitstream it describes had no read path to meter. [`KV260_AXI_METERED`] is a different
/// bitstream on the same board, with one, and states the read.
pub const KV260_MEASURED: Prices = Prices {
    e_sample: 1.0848e-11,
    e_read: f64::NAN,
    e_write: f64::NAN,
    reflash_hz_cap: None,
    source: "MEASURED on a Kria KV260 (xck26), 2026-09-06: 1,024 p-bits at 100 MHz drew 0.5554 W \
             above an idle PL on the SOM's INA260 (5 V rail, whole board), 23.1 sigma, over 51.2 \
             flips/ns. Reads and writes were not exercised and are unstated, not zero.",
    evidence: Evidence::Metered,
};

/// A fabricated 28 nm four-chip digital Ising accelerator: the lowest MEASURED energy per update this
/// project has located, and the authors' own stated figure rather than a quotient computed here.
///
/// arXiv:2609.07907 (7 Sept 2026; Wu, Raut, Khor, Alswaidan, Lee, Kuang, Aadit, Raju, Chinmay, Das,
/// Mai, Camsari and Srimani): 27,648 spins, degree-15 Pegasus connectivity, 10-bit coefficients,
/// 30.24e9 peak updates per second at 140 MHz.
///
/// # Why it is here
///
/// Because it is the number that disciplines every other one. Against [`KV260_MEASURED`] it is
/// about 9x lower, which makes the FPGA-to-ASIC gap for this job a measured quantity instead of a
/// rule of thumb. Against [`Z1_SPICE`] it is about **169x higher** — so the best measured update
/// energy located anywhere is still more than two orders of magnitude above the projected one,
/// and that gap, not any vendor multiplier, is the state of the field.
///
/// Graded [`Evidence::Measured`] and not [`Evidence::Metered`]: the abstract was read and the
/// measurement protocol was not, which is exactly what that grade is for. The figure INCLUDES I/O
/// power, so reads are not separable from it and stay unstated rather than zero.
pub const PEGASUS_28NM_ASIC: Prices = Prices {
    e_sample: 1.2e-12,
    e_read: f64::NAN,
    e_write: f64::NAN,
    reflash_hz_cap: None,
    source: "STATED BY ITS AUTHORS, arXiv:2609.07907 (2026-09-07): a fabricated 28 nm four-chip \
             digital Ising accelerator, 27,648 spins, Pegasus degree 15, '1.2 pJ/update including \
             I/O power' at 140 MHz. Abstract read; metering protocol not. Reads and writes are \
             folded into that figure and are unstated here, not zero.",
    evidence: Evidence::Measured,
};

/// The Kria KV260 again, thirteen days on, WITH A READ PATH — the first price in this crate for
/// moving a spin off a fabric, and a second metering of the flip by a different method.
///
/// [`KV260_MEASURED`] could not price a read because its bitstream had no way to perform one: the
/// block design disabled every PS-to-PL port. This one hangs
/// [`crate::hdl::FixedFabric::emit_axi_shell`] off `M_AXI_HPM0_FPD`, and `examples/board_build`
/// (`kv260-axi`) emits the design, the host reader and the protocol. The sensor logs are in
/// `measurements/kv260-read-2026-09-19/`, and a test below re-derives both constants from them.
///
/// # The read: `5.81e-10` J per spin value
///
/// One A53 core reading the fabric's 32 state words round-robin over single-beat AXI4-Lite moved
/// `1.3205e8` spin values a second and raised the board `+0.0768 W` over the same fabric running
/// unread (8 paired passes, se `0.0074`, 10.3 sigma). That is **581 ± 56 pJ per spin read — about
/// 64 flips.** One full-state readout takes 7.75 µs, during which the fabric completes 388 sweeps:
/// on this board the read path, not the sampler, bounds any chain whose correlation time is
/// shorter than that.
///
/// It INCLUDES the core that issues the read, because a read needs a master. The attempt to
/// subtract the core failed and is kept on the record: an arm spinning the same loop over cached
/// memory drew `0.0714 W` MORE than the arm reading the fabric, so a core stalled on a bus is not
/// a busy core and the difference bounds nothing. And single-beat AXI4-Lite through a CPU is the
/// most expensive way to move a bit off this fabric; a burst or DMA path was not built.
///
/// # The flip: `9.13e-12` J, by run/halt on one placement
///
/// The shell's `run` bit holds spins and generators while the clock keeps toggling. Running minus
/// held is `+0.4675 W` (se `0.0067`, 69 sigma) over `5.1199e10` flips a second. No reload between
/// arms, so placement, routing and clock tree are identical and cancel. It sits 16% under
/// [`KV260_MEASURED`]'s `1.0848e-11`, whose baseline was an idle PL and so included the clock tree
/// and the configured logic's static draw. Two methods, two definitions, one fabric; neither
/// replaces the other, and the older constant is unchanged.
///
/// Writes were not exercised — the couplings are in the bitstream — and stay unstated.
pub const KV260_AXI_METERED: Prices = Prices {
    e_sample: 9.1316e-12,
    e_read: 5.8126e-10,
    e_write: f64::NAN,
    reflash_hz_cap: None,
    source: "METERED on a Kria KV260 (xck26), 2026-09-19, one bitstream: 1,024 p-bits at 100 MHz \
             behind an AXI4-Lite shell on M_AXI_HPM0_FPD. Flip: run minus halt, +0.4675 W over \
             5.12e10 flips/s (69 sigma), datapath only, clock tree cancelled. Read: one A53 core, \
             single-beat reads, +0.0768 W over 1.32e8 spin values/s (10.3 sigma), INCLUDING the \
             core. INA260, 5 V rail, 8 paired passes each. Writes not exercised: unstated, not zero.",
    evidence: Evidence::Metered,
};

/// A fabricated 16 nm sampling accelerator, and **the closest thing in print to a competitor for
/// the figures this crate metered itself.**
///
/// arXiv:2606.16148 (Zhao, Verhelst et al., KU Leuven): *"AIA can generate 1277 MSample/s at 0.9V
/// and 20 GSamples/s/W at 0.7V which is up to 2× faster and 1.45x more energy efficient compared to
/// the previous state-of-the-art Markov Random Field (MRF) accelerator"*, implemented in Intel
/// 16 nm on a 4 mm² die. `20 GSamples/s/W` is `5.0e-11` J per sample, which is the number here.
///
/// # ⚠ Read this before comparing it with [`KV260_AXI_METERED`]
///
/// **A "sample" here may not be one node update.** The figure comes from a sampler microbenchmark
/// over distributions of varying entropy, on an architecture built for general approximate
/// inference — so the operation being priced is this crate's `e_sample` only if their sample and
/// our single-node redraw are the same work. That has not been established, and until it is, the
/// apparent ratio against our `9.13e-12` is **not** a like-for-like claim about silicon. It is
/// recorded here so the comparison is available and fenced, rather than made casually elsewhere.
///
/// Graded [`Evidence::Measured`] rather than [`Evidence::Metered`]: the chip is real and the figure
/// is its authors', but the metering protocol — what was held fixed, what the idle baseline was —
/// is not stated in the terms [`KV260_AXI_METERED`] states them. Reads and writes are not
/// separated from the figure and stay unstated, not zero.
pub const AIA_16NM_SAMPLER: Prices = Prices {
    e_sample: 5.0e-11,
    e_read: f64::NAN,
    e_write: f64::NAN,
    reflash_hz_cap: None,
    source: "STATED BY ITS AUTHORS, arXiv:2606.16148: AIA, a 16 nm multicore SoC for approximate \
             inference, '20 GSamples/s/W at 0.7V' = 5.0e-11 J per sample, from a sampler \
             microbenchmark. Silicon, 4 mm2, Intel 16 nm. WHETHER THEIR 'SAMPLE' IS THIS CRATE'S \
             NODE UPDATE IS NOT ESTABLISHED. Reads and writes unstated, not zero.",
    evidence: Evidence::Measured,
};

/// **The first METERED WRITE in this crate** — Kria KV260, 2026-09-22, on a fabric whose couplings
/// are registers rather than bitstream.
///
/// [`KV260_AXI_METERED`] cannot price a write and says so: its couplings live in the configuration
/// bitstream, so there is no write to perform. Pricing one needed
/// [`crate::writable::WritableFabric`], and this is the first time that fabric has run on silicon.
///
/// # The write: `4.407e-8` J per node
///
/// One A53 core rewrites every node's three configuration words, round and round, alternating
/// `J = 256` and `J = 255` in Q.8 so the register bits really toggle while the physics stays where
/// it was. That raised the board `+0.1114 W` over the same fabric running unwritten (8 paired
/// passes, se `0.0077`, **14.5 sigma**) at `2,527,457` node writes a second. **44.07 ± 3.04 nJ per
/// node written**, or `14.69 nJ` per 32-bit word.
///
/// **A write costs 2,302 node updates on this same fabric.** That is what a tempering ladder pays
/// per node per rung to rewrite its couplings, and it is the number that decides whether a schedule
/// belongs in the fabric or in the problem.
///
/// # The per-beat cost dominates, which corrects an easy misreading
///
/// [`KV260_AXI_METERED`] quotes `581 pJ` per spin READ, which makes this write look 25x dearer. It
/// is not. A read word carries 32 spin values, so per AXI beat that read is `18.6 nJ` against this
/// write's `14.7 nJ`. **A single-beat AXI4-Lite transaction costs 15 to 19 nJ on this board in
/// either direction**; `581 pJ` is low only because one word carries 32 spins. Compare beats to
/// beats, or compare a read of 32 spins against a write of one coupling and be wrong by 25x.
///
/// # The flip: `1.915e-11` J, and what writability costs
///
/// Run minus held on one placement, `+0.1225 W` (se `0.0052`, 23.5 sigma) over `6.3999e9` flips a
/// second. That is **2.1x** [`KV260_AXI_METERED`]'s `9.1316e-12` on the same board by the same
/// method — the price of a fabric with 137 LUTs per p-bit against 44, taking four clocks a sweep
/// against two. Writability is not free and this is what it costs.
///
/// Reads were not exercised on this design and stay unstated.
pub const KV260_WRITABLE_METERED: Prices = Prices {
    e_sample: 1.914510e-11,
    e_read: f64::NAN,
    e_write: 4.407325e-08,
    reflash_hz_cap: None,
    source: "METERED on a Kria KV260 (xck26), 2026-09-22, one bitstream: 256 p-bits at 100 MHz \
             behind writable::WritableFabric's AXI4-Lite shell on M_AXI_HPM0_FPD, couplings as \
             12-bit Q.8 registers. Write: one A53 core rewriting every node, +0.1114 W over \
             2.527e6 node writes/s (14.5 sigma), INCLUDING the core. Flip: run minus halt, \
             +0.1225 W over 6.40e9 flips/s (23.5 sigma), clock tree cancelled. INA260, 5 V rail, \
             8 paired passes. Reads not exercised on this design: unstated, not zero.",
    evidence: Evidence::Metered,
};

/// Every machine this crate states prices for, in a stable order, each beside its name.
///
/// The reason this is a table and not three constants a caller looks up by hand: a joules figure
/// belongs to a machine, and the binding surfaces had no way to *name* one. Python, Julia and Zig
/// could ask what a run cost but had to supply the prices themselves, which in practice means
/// pasting numbers out of this file — and a pasted number arrives without its
/// [`Prices::source`], which is the half that says whose silicon it describes and whether anyone
/// measured it. Enumerating the table over the ABI carries the provenance with the number.
///
/// [`Prices::UNSTATED`] is deliberately in here. It is the honest answer for a machine nobody has
/// characterised, and a caller who can select it by name can demonstrate that pricing a run
/// against it yields no figure rather than a zero.
pub const CATALOGUE: [(&str, Prices); 7] = [
    ("UNSTATED", Prices::UNSTATED),
    ("Z1_SPICE", Z1_SPICE),
    ("KV260_MEASURED", KV260_MEASURED),
    // APPENDED, never inserted: the binding surfaces enumerate this table by index.
    ("PEGASUS_28NM_ASIC", PEGASUS_28NM_ASIC),
    ("KV260_AXI_METERED", KV260_AXI_METERED),
    ("AIA_16NM_SAMPLER", AIA_16NM_SAMPLER),
    ("KV260_WRITABLE_METERED", KV260_WRITABLE_METERED),
];

/// Operation counts accumulated by a run.
///
/// # These three fields are physical events, not work counters
///
/// Each field counts one kind of device operation, priced by the matching field of [`Prices`],
/// and a sampler may charge a field ONLY for that event. This is stated as a rule because it was
/// broken: until 2026-09-13 `cluster.rs` charged `reads` for every BOND it examined and `writes`
/// for every spin its cluster FLIPPED, while `gibbs.rs` charged `reads` for every node READ OUT
/// and `writes` never -- so one field, priced at 1.692 pJ, named two different things in two
/// samplers and a joules figure compared across them was not a comparison. Found by an agent
/// writing a third sampler, who had to pick one convention and noticed there were two.
///
/// - `samples`: one node's state updated by the device. What a sweep does.
/// - `reads`: one node's value leaving the chip. What `collect` does per kept draw, and nothing
///   a sweep does internally -- neighbour lookups, bond tests, cluster growth -- is a read.
/// - `writes`: one node's couplings, bias or clamp reprogrammed. A spin flipping is not a write;
///   a temperature change on a bitstream fabric is one per node.
///
/// Work a move does that is none of these -- a bond tested, a cluster grown -- is measured in that
/// sampler's own stats and is UNPRICED, because no device model here states a price for it, and a
/// ledger field is not the place to keep an unpriced count.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Ledger {
    /// Single-node Gibbs updates performed.
    pub samples: u64,
    /// Node values read out to the chip edge.
    pub reads: u64,
    /// Node couplings, biases or clamp state flashed.
    pub writes: u64,
}

impl Ledger {
    /// Energy under `p`, or `None` when `p` states no prices.
    ///
    /// `Option`, not a number, and not zero. A machine whose per-operation energy nobody has
    /// published does not cost nothing, and a caller that has to unwrap this cannot accidentally
    /// print a figure for a device that has none.
    #[must_use]
    pub fn joules(&self, p: &Prices) -> Option<f64> {
        // Per-OPERATION, not all-or-nothing. A price set that states what this run actually did can
        // price it, and one that does not cannot -- which is a finer and more useful question than
        // "are all three stated". It matters because real measurements arrive partial:
        // `KV260_MEASURED` came off a wattmeter running a free-running fabric with no host traffic,
        // so it states a per-sample energy and nothing else. Under the old rule that measurement
        // could price NOTHING, including the very workload it was taken from. Under this one it
        // prices sampling exactly and still refuses the reads and writes nobody metered.
        //
        // A zero count needs no price: a run that performed no writes is not made unpriceable by an
        // unstated write cost. An unstated price for an operation the run DID perform still yields
        // `None`, which is the refusal this type exists for.
        let charge = |count: u64, price: f64| -> Option<f64> {
            if count == 0 {
                Some(0.0)
            } else if price.is_finite() {
                Some(count as f64 * price)
            } else {
                None
            }
        };
        Some(
            charge(self.samples, p.e_sample)?
                + charge(self.reads, p.e_read)?
                + charge(self.writes, p.e_write)?,
        )
    }

    /// The wall-clock floor this many full-graph reflashes implies, or `None` if the device states
    /// no cap.
    ///
    /// A run that reflashes faster than the hardware sustains is not fast; it is unphysical, and
    /// pricing it describes something that could not have happened.
    #[must_use]
    pub fn reflash_seconds(&self, p: &Prices, graph_nodes: u64) -> Option<f64> {
        let hz = p.reflash_hz_cap?;
        if graph_nodes == 0 || hz <= 0.0 {
            return None;
        }
        Some(self.writes as f64 / graph_nodes as f64 / hz)
    }

    /// Fractional breakdown (sample, read, write) of total energy under prices `p`.
    #[must_use]
    pub fn shares(&self, p: &Prices) -> (f64, f64, f64) {
        if !p.is_stated() {
            return (f64::NAN, f64::NAN, f64::NAN);
        }
        let s = self.samples as f64 * p.e_sample;
        let r = self.reads as f64 * p.e_read;
        let w = self.writes as f64 * p.e_write;
        let t = (s + r + w).max(1e-300);
        (s / t, r / t, w / t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A ratio across evidence grades inherits the weaker one.
    ///
    /// The error this prevents is the field's most common: our own metered KV260 figure divided by
    /// a vendor's pre-silicon SPICE estimate is not a measurement of anything, and the number that
    /// comes out of that division must not be reported as though it were.
    #[test]
    fn a_comparison_is_only_as_good_as_its_weaker_side() {
        assert_eq!(KV260_MEASURED.evidence, Evidence::Metered);
        assert_eq!(Z1_SPICE.evidence, Evidence::Simulated);
        assert_eq!(
            weaker(KV260_MEASURED.evidence, Z1_SPICE.evidence),
            Evidence::Simulated,
            "metered over simulated is simulated, whichever way round it is written"
        );
        assert_eq!(weaker(Z1_SPICE.evidence, KV260_MEASURED.evidence), Evidence::Simulated);
        // It is an order, and the order is the argument.
        assert!(Evidence::Metered > Evidence::Measured);
        assert!(Evidence::Measured > Evidence::Simulated);
        assert!(Evidence::Simulated > Evidence::Derived);
        assert!(Evidence::Derived > Evidence::Projected);
        assert!(Evidence::Projected > Evidence::Unstated);
        // An unstated price drags any comparison to the bottom, which is the point of grading it.
        assert_eq!(weaker(KV260_MEASURED.evidence, Prices::UNSTATED.evidence), Evidence::Unstated);
    }

    /// The one price in this crate that was metered says so in the type, not only in the prose.
    #[test]
    fn only_the_metered_price_claims_to_be_metered() {
        let graded = [
            ("UNSTATED", Evidence::Unstated),
            ("Z1_SPICE", Evidence::Simulated),
            ("KV260_MEASURED", Evidence::Metered),
            ("PEGASUS_28NM_ASIC", Evidence::Measured),
            ("KV260_AXI_METERED", Evidence::Metered),
            ("AIA_16NM_SAMPLER", Evidence::Measured),
            ("KV260_WRITABLE_METERED", Evidence::Metered),
        ];
        // OVER THE CATALOGUE, not over a list written here. This test used to walk its own three
        // entries and conclude "exactly one price in this crate was metered" -- a count of the
        // test's array, which two later entries joined the crate without ever entering.
        assert_eq!(CATALOGUE.len(), graded.len(), "a machine joined the table without a grade here");
        let mut metered = Vec::new();
        for ((name, p), (want_name, want)) in CATALOGUE.iter().zip(graded) {
            assert_eq!(*name, want_name);
            assert_eq!(p.evidence, want, "{name} was graded wrong");
            // THE NAME MUST BE BOUND TO THE CONSTANT IT NAMES. Nothing checked this for any entry:
            // `("KV260_AXI_METERED", KV260_MEASURED)` passed every test in the crate, because the
            // two share a grade and an instrument, and a binding that asked for the board with a
            // metered read would have been handed the one that refuses reads, by name.
            let constant = match *name {
                "UNSTATED" => Prices::UNSTATED,
                "Z1_SPICE" => Z1_SPICE,
                "KV260_MEASURED" => KV260_MEASURED,
                "PEGASUS_28NM_ASIC" => PEGASUS_28NM_ASIC,
                "KV260_AXI_METERED" => KV260_AXI_METERED,
                "AIA_16NM_SAMPLER" => AIA_16NM_SAMPLER,
                "KV260_WRITABLE_METERED" => KV260_WRITABLE_METERED,
                other => panic!("{other} is in the table and has no constant named here"),
            };
            assert_eq!(p.source, constant.source, "{name} is bound to another machine's prices");
            for (got, want) in [(p.e_sample, constant.e_sample), (p.e_read, constant.e_read), (p.e_write, constant.e_write)] {
                assert_eq!(got.to_bits(), want.to_bits(), "{name}: a price that is not its constant's");
            }
            if p.evidence == Evidence::Metered {
                // Metered means this project's protocol on this project's instrument, and the
                // source has to say which instrument.
                assert!(p.source.contains("INA260"), "{name} claims a meter and names none");
                metered.push(*name);
            }
        }
        assert_eq!(
            metered,
            ["KV260_MEASURED", "KV260_AXI_METERED", "KV260_WRITABLE_METERED"],
            "all three on the one board we own, and no other machine may claim the grade"
        );
        // and the grade must agree with the prose, or one of them is lying
        assert!(KV260_MEASURED.source.contains("MEASURED"));
        assert!(Z1_SPICE.source.contains("SPICE"));
    }

    #[test]
    fn evidence_prints_as_the_word_it_is() {
        assert_eq!(Evidence::Metered.to_string(), "metered");
        assert_eq!(Evidence::Projected.to_string(), "projected");
        assert_ne!(Evidence::Derived.to_string(), Evidence::Simulated.to_string());
    }


    #[test]
    fn write_to_sample_ratio_is_the_finding() {
        // The structural number behind the robotics verdict: one write = how many samples?
        let ratio = Z1_SPICE.e_write / Z1_SPICE.e_sample;
        assert!((ratio - 21664.0).abs() < 100.0, "write/sample = {ratio}");
        let rr = Z1_SPICE.e_read / Z1_SPICE.e_sample;
        assert!((rr - 238.6).abs() < 5.0, "read/sample = {rr}");
    }

    /// **A PRICE MUST BE RE-DERIVABLE FROM THE QUOTATION IT CITES**, or it is a number somebody
    /// typed. [`KV260_AXI_METERED`] is re-derived from its sensor logs; this does the same for the
    /// entries whose evidence is a published figure, by reading the figure back out of the
    /// `source` string and recomputing the price from it.
    ///
    /// Written after a mutant moved [`AIA_16NM_SAMPLER`]'s energy by a factor of ten and every
    /// test in this file stayed green: the catalogue check compares each entry against its
    /// constant, so changing the constant changes both sides and they still agree.
    #[test]
    fn a_cited_price_is_recomputed_from_the_figure_its_own_source_quotes() {
        // "20 GSamples/s/W" -> 1 / 20e9 J per sample.
        let src = AIA_16NM_SAMPLER.source;
        assert!(src.contains("20 GSamples/s/W"), "the source no longer quotes the figure it is built from");
        let want = 1.0 / 20e9;
        assert!(
            (AIA_16NM_SAMPLER.e_sample - want).abs() < 1e-18,
            "source says 20 GSamples/s/W = {want:e} J per sample; the constant says {:e}",
            AIA_16NM_SAMPLER.e_sample
        );
        // And the arithmetic in the source text must agree with itself.
        assert!(src.contains("5.0e-11"), "the source states a converted value that must match");

        // The same discipline for the other cited figure: the ASIC's stated pJ per update.
        assert!(PEGASUS_28NM_ASIC.source.contains("1.2 pJ/update"));
        assert!((PEGASUS_28NM_ASIC.e_sample - 1.2e-12).abs() < 1e-24);

        // The board figures are re-derived from sensor logs elsewhere; here, just that they still
        // quote the protocol their grade depends on.
        for p in [KV260_MEASURED, KV260_AXI_METERED] {
            assert!(p.source.contains("INA260"), "a metered grade must name its instrument");
        }
    }

    /// The measured board figure, and the gap it puts a number on.
    ///
    /// This is the only price in the crate that came off a wattmeter rather than out of a model, so
    /// the test that guards it checks the two things that would make it a lie: that it prices
    /// sampling, and that it REFUSES to price the reads and writes nobody measured.
    #[test]
    fn the_measured_board_price_prices_sampling_and_refuses_the_rest() {
        // 1,024 p-bits at 100 MHz, two clocks per sweep, one second of running.
        let flips = 1024u64 * 50_000_000;
        let l = Ledger { samples: flips, reads: 0, writes: 0 };
        let joules = l.joules(&KV260_MEASURED).expect("sampling alone is priced");
        // 0.5554 W for one second, reproduced from the per-flip figure.
        assert!((joules - 0.5554).abs() < 0.005, "one second of the measured fabric = {joules} J");

        // The gap against the projection this crate exists to test, stated as a ratio rather than
        // as a slogan. Extropic's SPICE table says 7.09 fJ; this board says 10.85 pJ.
        let ratio = KV260_MEASURED.e_sample / Z1_SPICE.e_sample;
        assert!((1500.0..1560.0).contains(&ratio), "measured / projected = {ratio}");

        // A workload that reads or writes cannot be priced from a sampling-only measurement, and
        // the ledger says so instead of charging zero for the parts nobody metered.
        assert!(!KV260_MEASURED.is_stated(), "reads and writes are unstated, not free");
        let with_io = Ledger { samples: flips, reads: 1, writes: 0 };
        assert_eq!(
            with_io.joules(&KV260_MEASURED),
            None,
            "pricing a read against a run that never performed one would be a fabricated number"
        );
    }

    #[test]
    fn unstated_prices_produce_no_number_rather_than_a_wrong_one() {
        // The failure this prevents: every fabric in the tree once declared Z1_SPICE, so a Hitachi
        // CMOS annealer and a laptop CPU both reported Extropic's pre-silicon SPICE estimates as
        // their own energy. Nothing was lying; nothing had been asked whose numbers those were.
        let l = Ledger { samples: 1_000, reads: 10, writes: 1 };
        assert!(l.joules(&Z1_SPICE).unwrap() > 0.0);
        assert_eq!(
            l.joules(&Prices::UNSTATED),
            None,
            "a device with no published per-operation energy has no joules figure, and zero would \
             be a claim that it costs nothing"
        );
        assert!(!Prices::UNSTATED.is_stated());
        assert!(Z1_SPICE.is_stated());
        assert!(
            Z1_SPICE.source.contains("SPICE") && Z1_SPICE.source.contains("not measured"),
            "the provenance has to travel WITH the numbers: {}",
            Z1_SPICE.source
        );
    }

    #[test]
    fn a_program_load_is_charged_as_a_write() {
        // On this hardware class a write costs ~21,700 samples, which is the ledger's whole thesis.
        // No implementation charged it, so every figure the stack produced was a sample-and-read
        // story with the expensive term silently zero.
        use crate::fabric::{Cpu, Device};
        use crate::ftp::Program;
        let mut cpu = Cpu::default();
        assert_eq!(cpu.ledger().writes, 0);
        let sched = crate::schedule::Schedule::default();
        let p = Program::from_graph(&crate::ising::lattice2d(4, 1.0), &sched);
        assert!(cpu.program(&p).is_empty(), "a pairwise lattice loads on the CPU device");
        assert_eq!(cpu.ledger().writes, 16, "one write per node flashed");
    }

    #[test]
    fn the_reflash_cap_turns_writes_into_a_wall_clock_floor() {
        // A workload that reflashes the whole graph faster than the device sustains is not fast,
        // it is unphysical -- and pricing it describes a run that could not have happened.
        // `reflash_hz_cap` was declared and read by nothing.
        let l = Ledger { samples: 0, reads: 0, writes: 500 };
        // 500 node-writes over a 100-node graph is 5 full reflashes; at 1 Hz that is 5 seconds.
        assert_eq!(l.reflash_seconds(&Z1_SPICE, 100), Some(5.0));
        assert_eq!(
            l.reflash_seconds(&Prices::UNSTATED, 100),
            None,
            "a device that states no cap implies no floor"
        );
    }

    /// The comparison this module exists to keep honest, on STATED figures only.
    ///
    /// Three machines, three grades, in the order the evidence ladder puts them — and the ordering
    /// of the ENERGIES is not the ordering of the GRADES, which is the whole point: the weakest
    /// evidence carries the most flattering number. A fabricated ASIC beats this project's FPGA
    /// fabric by about nine, and the projection beats the ASIC by about a hundred and seventy.
    /// A sentence in this file once called the KV260 figure the first non-projected entry in the
    /// comparison; it was retired by a literature sweep, and this test is what replaces it.
    #[test]
    fn the_best_measured_update_sits_between_the_projection_and_our_fabric() {
        let (z1, asic, kv) = (Z1_SPICE, PEGASUS_28NM_ASIC, KV260_MEASURED);
        assert!(z1.e_sample < asic.e_sample && asic.e_sample < kv.e_sample, "energies ascend");
        assert!(z1.evidence < asic.evidence && asic.evidence < kv.evidence, "and so do the grades");

        let fpga_over_asic = kv.e_sample / asic.e_sample;
        assert!((fpga_over_asic - 9.04).abs() < 0.01, "FPGA fabric over ASIC: {fpga_over_asic}");
        let asic_over_projection = asic.e_sample / z1.e_sample;
        assert!((asic_over_projection - 169.25).abs() < 0.01, "ASIC over projection: {asic_over_projection}");

        // a comparison inherits the weaker grade, in either direction
        assert_eq!(weaker(asic.evidence, kv.evidence), Evidence::Measured);
        assert_eq!(weaker(asic.evidence, z1.evidence), Evidence::Simulated);

        // The like-for-like figure against anyone's WALL power is the whole board, not the
        // increment: 3.6917 W with the fabric on, over 1,024 p-bits at 50 MHz of updates each.
        let whole_board: f64 = 3.6917 / 5.12e10;
        assert!((whole_board - 7.21e-11).abs() < 1e-13, "whole-board J per flip: {whole_board}");
        assert!(whole_board / kv.e_sample > 6.0, "the increment flatters by more than six");

        // the table: appended, named once each, and the ASIC priced like the KV260 -- samples only
        assert_eq!(CATALOGUE.len(), 7);
        assert_eq!(CATALOGUE[3].0, "PEGASUS_28NM_ASIC");
        assert_eq!(CATALOGUE[4].0, "KV260_AXI_METERED", "appended, so every earlier index holds");
        assert_eq!(CATALOGUE[6].0, "KV260_WRITABLE_METERED", "and the metered write appended after it");
        for (i, (a, _)) in CATALOGUE.iter().enumerate() {
            for (b, _) in &CATALOGUE[i + 1..] {
                assert_ne!(a, b, "a machine named twice");
            }
        }
        assert!(asic.e_read.is_nan() && asic.e_write.is_nan(), "folded into the figure, not zero");
    }

    /// Per-pass mean power of each arm, and the host program's own account of each arm.
    type ArmMeans = std::collections::BTreeMap<(u32, String), f64>;
    type ArmFacts = std::collections::BTreeMap<(u32, String), std::collections::BTreeMap<String, f64>>;

    fn parse_sensor_log(text: &str) -> (ArmMeans, ArmFacts, usize) {
        let mut sums: std::collections::BTreeMap<(u32, String), (f64, usize)> = Default::default();
        let mut facts = ArmFacts::new();
        let mut flags = 0;
        for line in text.lines() {
            let t: Vec<&str> = line.split_whitespace().collect();
            match t.first().copied() {
                Some("S") => {
                    let e = sums.entry((t[1].parse().unwrap(), t[2].to_string())).or_insert((0.0, 0));
                    e.0 += t[3].parse::<f64>().unwrap() * 1e-6;
                    e.1 += 1;
                }
                Some("A") => {
                    let pass: u32 = t[1].parse().unwrap();
                    let mut arm = String::new();
                    let mut kv = std::collections::BTreeMap::new();
                    for f in &t[2..] {
                        let (k, v) = f.split_once('=').unwrap();
                        if k == "arm" {
                            arm = v.to_string();
                        } else if let Ok(x) = v.parse::<f64>() {
                            kv.insert(k.to_string(), x);
                        }
                    }
                    facts.insert((pass, arm), kv);
                }
                Some("#") if line.contains("FOREIGN") || line.contains("ABORT") => flags += 1,
                _ => {}
            }
        }
        (sums.into_iter().map(|(k, (s, n))| (k, s / n as f64)).collect(), facts, flags)
    }

    /// Mean and standard error of one difference PER PASS: the 768 samples of an arm share sensor
    /// conversion windows and are not 768 measurements, but the 8 passes are 8.
    fn paired(means: &ArmMeans, hi: &str, lo: &str) -> (f64, f64, usize) {
        let d: Vec<f64> = (1..=64)
            .filter_map(|p| Some(means.get(&(p, hi.to_string()))? - means.get(&(p, lo.to_string()))?))
            .collect();
        let n = d.len() as f64;
        let m = d.iter().sum::<f64>() / n;
        let var = d.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1.0);
        (m, (var / n).sqrt(), d.len())
    }

    /// The constants in [`KV260_WRITABLE_METERED`] are RE-DERIVED here from the sensor log they came
    /// from — **the first metered write in this crate**, so it is the one number most worth tying to
    /// its evidence.
    #[test]
    fn the_metered_write_is_rederived_from_its_own_sensor_log() {
        let (means, facts, flags) =
            parse_sensor_log(include_str!("../measurements/kv260-write-2026-09-22/write_meter_1.dat"));
        assert_eq!(flags, 0, "a pass during which someone else loaded the FPGA is not evidence");

        // CONTROLS WITH KNOWN ANSWERS, taken before either constant is computed.
        let mut node_writes_per_s = 0.0;
        let mut words_per_s = 0.0;
        let (mut wrote, mut held) = (0usize, 0usize);
        for ((_, arm), kv) in &facts {
            match arm.as_str() {
                "halt" => {
                    // Held means held: no sweeps, and a popcount that never moves.
                    assert_eq!(kv["sweeps_per_s"], 0.0, "a held fabric does not sweep");
                    assert_eq!(kv["pop_min"], kv["pop_max"], "a held fabric does not change state");
                    held += 1;
                }
                arm => {
                    // 100 MHz at FOUR clocks a sweep is 2.5e7 -- what a writable fabric costs where
                    // `FixedFabric` takes two. Measured on the part, not assumed from the RTL.
                    let sweeps = kv["sweeps_per_s"];
                    assert!((sweeps / 2.5e7 - 1.0).abs() < 1e-4, "{arm}: {sweeps} sweeps/s is not this fabric");
                    if arm == "write" {
                        // The write path is lossless: the shell accepted exactly what was sent.
                        assert_eq!(kv["words_sent"], kv["words_accepted"], "a dropped word is a wrong bill");
                        assert!(kv["words_sent"] > 0.0, "the write arm must actually write");
                        node_writes_per_s += kv["node_writes_per_s"] / 8.0;
                        words_per_s += kv["words_per_s"] / 8.0;
                        wrote += 1;
                    } else {
                        assert_eq!(kv["words_sent"], 0.0, "only the write arm writes");
                    }
                }
            }
        }
        assert_eq!((wrote, held), (8, 8), "8 passes of each arm");
        // Three words a node, as `ft_write` programs them.
        assert!((words_per_s / node_writes_per_s - 3.0).abs() < 1e-6, "a node is three config words");

        let (watts, se, passes) = paired(&means, "write", "idle");
        assert_eq!(passes, 8);
        assert!(watts / se > 5.0, "a write must clear the sensor's noise: {watts} W, se {se}");
        let e_write = watts / node_writes_per_s;
        assert!(
            (e_write / KV260_WRITABLE_METERED.e_write - 1.0).abs() < 1e-3,
            "the log says {e_write:e} J per node written, the constant says {:e}",
            KV260_WRITABLE_METERED.e_write
        );

        // The flip on the same placement, by run/halt: no reload between arms, so routing and
        // clock tree are identical and cancel.
        let (watts, se, passes) = paired(&means, "idle", "halt");
        assert_eq!(passes, 8);
        assert!(watts / se > 5.0);
        let flips_per_s = 2.5e7 * 256.0 * 0.99997854; // sweeps/s x 256 nodes, at the measured rate
        let e_sample = watts / flips_per_s;
        assert!(
            (e_sample / KV260_WRITABLE_METERED.e_sample - 1.0).abs() < 1e-3,
            "the log says {e_sample:e} J per flip, the constant says {:e}",
            KV260_WRITABLE_METERED.e_sample
        );

        // WHAT THE WRITE COSTS IN THE UNIT THE FABRIC IS PRICED IN. This is the number that decides
        // whether a schedule belongs in the fabric or in the problem.
        let ratio = e_write / e_sample;
        assert!((2250.0..2350.0).contains(&ratio), "a node write is {ratio} node updates");

        // WRITABILITY IS NOT FREE, and this is the only pair of constants that can say so: two
        // fabrics, one board, one method, one sensor.
        let price = KV260_WRITABLE_METERED.e_sample / KV260_AXI_METERED.e_sample;
        assert!((2.0..2.2).contains(&price), "a writable p-bit flips for {price}x a fixed one");

        // AND THE MISREADING THE TWO CONSTANTS INVITE. `KV260_AXI_METERED.e_read` is per SPIN, and
        // a read word carries 32 of them, so comparing it to a per-word write makes the write look
        // 25x dearer. Per AXI BEAT the two are within a factor of two, which is the honest
        // statement: on this board a single-beat transaction costs 15-19 nJ in either direction.
        let read_per_beat = KV260_AXI_METERED.e_read * 32.0;
        let write_per_beat = e_write / 3.0;
        let beats = read_per_beat / write_per_beat;
        assert!((1.0..2.0).contains(&beats), "read beat over write beat is {beats}, not a factor of two");
        let per_spin = KV260_AXI_METERED.e_read / write_per_beat;
        assert!(per_spin < 0.1, "and per SPIN the read looks {per_spin}x a write, which is the trap");
    }

    /// The constants in [`KV260_AXI_METERED`] are RE-DERIVED here from the sensor logs they came
    /// from, so they cannot drift from their evidence, and the evidence cannot be lost with a
    /// scratch directory the way an earlier audit's was.
    #[test]
    fn the_metered_read_is_rederived_from_its_own_sensor_log() {
        let (means, facts, flags) =
            parse_sensor_log(include_str!("../measurements/kv260-read-2026-09-19/read_meter_1.dat"));
        assert_eq!(flags, 0, "a pass during which someone else loaded the FPGA is not evidence");
        let (watts, se, passes) = paired(&means, "axi", "idle");
        assert_eq!(passes, 8);
        assert!(watts / se > 5.0, "a read must clear the sensor's noise: {watts} W, se {se}");

        // CONTROLS WITH KNOWN ANSWERS. 100 MHz and two clocks a sweep is 5.0e7; the PS clock is
        // 20 ppm slow. A port whose width disagreed with the PS would return words that fail this.
        let mut words_per_s = 0.0;
        for ((_, arm), kv) in &facts {
            let sweeps = kv["sweeps_per_s"];
            assert!((sweeps / 5.0e7 - 1.0).abs() < 1e-4, "{arm}: {sweeps} sweeps/s is not this fabric");
            if arm == "axi" {
                words_per_s += kv["reads_per_s"] / 8.0;
                assert!(kv["word0_changes"] > 1000.0, "the words read must be a fabric that is moving");
            }
        }
        let e_read = watts / (words_per_s * 32.0);
        assert!(
            (e_read / KV260_AXI_METERED.e_read - 1.0).abs() < 1e-3,
            "the log says {e_read:e} J per spin read, the constant says {:e}",
            KV260_AXI_METERED.e_read
        );

        // THE CONTROL THAT FAILED stays failed. If a later edit starts reporting `axi - cpu` as
        // the bus's share of a read, this is the measurement that says it is not one.
        let (bus, bus_se, _) = paired(&means, "axi", "cpu");
        assert!(bus < -5.0 * bus_se, "a core stalled on AXI drew LESS than a busy one: {bus} W");

        // The flip, by run/halt on the same placement.
        let (means, facts, flags) =
            parse_sensor_log(include_str!("../measurements/kv260-read-2026-09-19/read_meter_2.dat"));
        assert_eq!(flags, 0);
        let (watts, se, passes) = paired(&means, "idle", "halt");
        assert_eq!(passes, 8);
        assert!(watts / se > 5.0);
        let mut flips_per_s = 0.0;
        for ((_, arm), kv) in &facts {
            if arm == "halt" {
                // Held means held: no sweeps, and a popcount that never moves.
                assert_eq!(kv["sweeps_per_s"], 0.0);
                assert_eq!(kv["pop_min"], kv["pop_max"]);
                assert_eq!(kv["word0_changes"], 0.0);
            } else {
                flips_per_s += kv["sweeps_per_s"] * 1024.0 / 8.0;
            }
        }
        let e_sample = watts / flips_per_s;
        assert!((e_sample / KV260_AXI_METERED.e_sample - 1.0).abs() < 1e-3, "{e_sample:e} J per flip");

        // Two meterings of one fabric, thirteen days and one method apart. They must not agree
        // exactly -- the older baseline was an idle PL and includes the clock tree -- and they
        // must not be far apart either.
        let older = KV260_MEASURED.e_sample / KV260_AXI_METERED.e_sample;
        assert!((1.05..1.35).contains(&older), "idle-PL baseline over run/halt = {older}");

        // What the read costs in the unit the fabric is priced in, and what that does to a run
        // that reads its whole state every sweep: the reads are the bill.
        let ratio = KV260_AXI_METERED.e_read / KV260_AXI_METERED.e_sample;
        assert!((55.0..72.0).contains(&ratio), "a spin read costs {ratio} flips");
        let every_sweep = Ledger { samples: 1024 * 1000, reads: 1024 * 1000, writes: 0 };
        let never = Ledger { samples: 1024 * 1000, reads: 0, writes: 0 };
        let a = every_sweep.joules(&KV260_AXI_METERED).expect("samples and reads are both metered");
        let b = never.joules(&KV260_AXI_METERED).expect("samples are metered");
        assert!(a / b > 56.0, "reading every sweep multiplies the bill by {}", a / b);
        // And it still refuses what nobody metered.
        assert!(!KV260_AXI_METERED.is_stated());
        assert_eq!(Ledger { samples: 1, reads: 1, writes: 1 }.joules(&KV260_AXI_METERED), None);
    }

    /// The two directions of the readout question are inverses, checked as a round trip rather than
    /// as a restatement of either formula.
    ///
    /// [`read_budget_for_advantage`] answers "how cheap must a read be for this claim to hold";
    /// [`advantage_after_readout`] answers "what is left of the claim at this read price". Feeding
    /// each into the other must return the input. Written as two separate expressions on purpose: a
    /// single formula asserted against itself is the repo's most common vacuity, and a round trip
    /// through two independently written ones is not.
    #[test]
    fn the_readout_budget_and_the_advantage_are_inverses() {
        let cases = [
            (2.0833e-3f64, 1.2083e-14f64, 2_592_000u64, 1e11f64),
            (1.0, 1e-9, 1_000, 5.0),
            (7.5e-6, 2.5e-9, 42, 137.0),
        ];
        let mut checked = 0usize;
        for &(baseline, dynamics, reads, want) in &cases {
            let budget = read_budget_for_advantage(baseline, dynamics, reads, want).expect("attainable");
            let back = advantage_after_readout(baseline, dynamics, reads, budget);
            assert!((back - want).abs() / want < 1e-9, "round trip: asked {want}, budget {budget:e}, got back {back}");
            // and the budget is binding: a read ten percent dearer must lose the claim.
            let dearer = advantage_after_readout(baseline, dynamics, reads, budget * 1.1);
            assert!(dearer < want, "the budget must be a real ceiling: {dearer} vs {want}");
            checked += 1;
        }
        assert_eq!(checked, 3, "every case must be exercised");

        // A claim the dynamics alone already break has no readout budget, free readout included.
        assert_eq!(read_budget_for_advantage(1.0, 1.0, 10, 2.0), None, "dynamics already exceed the budget");
        // and one that is only just attainable still reports a positive budget.
        assert!(read_budget_for_advantage(1.0, 0.4, 10, 2.0).is_some(), "0.5 > 0.4 leaves room");
    }

    /// **Whitelam\'s ten orders of magnitude require a readout at the Landauer floor.**
    ///
    /// arXiv:2506.15121 (v3) prices a generative Langevin computer against a digital denoiser. Both
    /// figures are the paper\'s own, verbatim: the digital budget is *"not less than
    /// 5\u{d7}10^{14} k_BT"* per denoising trajectory, and *"Over 1000 independent denoising
    /// trajectories of the trained computer we calculate a mean heat emission of
    /// \u{27e8}Q\u{27e9}=2.9\u{d7}10^{3} k_BT"*. Their ratio is *"more than 10^{11}"*.
    ///
    /// **Both numbers are dynamics.** The heat is defined as
    /// `Q = V(x(0)) - V(x(t_f))` — the potential energy at the two ends of a trajectory — so
    /// obtaining the paper\'s own reported quantity requires reading the full state twice, over
    /// `N_v + N_h = 784 + 512 = 1296` units. That is 2,592 values per trajectory and 2,592,000 over
    /// the 1000 trajectories the mean is taken over. The claim charges none of them.
    ///
    /// | per-value readout cost | advantage that survives |
    /// |---|---|
    /// | free | 1.72e11 — the paper\'s own ratio, reproduced |
    /// | 1.18 x the Landauer floor | 1e11 — the last point the headline holds |
    /// | 10 x the floor | 2.4e10 |
    /// | 26.4 x the floor | 1e10 |
    /// | one metered KV260 AXI read | **1.38** |
    ///
    /// The middle rows need no device and no agreement about one: **for "more than 10^11" to
    /// survive, every one of those 2,592 values must be read out for no more than 1.18 times
    /// `kT ln 2`** — essentially at the thermodynamic floor for reading one bit, with no room for a
    /// wire, an amplifier or an ADC. That is a bound on any readout, not a complaint about a
    /// particular one.
    ///
    /// The free-readout row is the control: with reads priced at zero this machinery returns the
    /// paper\'s ratio exactly, so the collapse in the last row is attributable to the readout and
    /// not to the arithmetic here.
    #[test]
    fn whitelams_ten_orders_of_magnitude_need_a_readout_at_the_landauer_floor() {
        // The paper\'s OWN kT, from its own sentence: 1 pJ per MAC "or 2.4e8 k_BT".
        let kt = 1e-12 / 2.4e8;
        let trajectories = 1000.0f64;
        let units = 784u64 + 512;
        let reads = 2 * units * 1000; // both ends of each trajectory, every unit
        assert_eq!(reads, 2_592_000, "1296 units read at both ends of 1000 trajectories");

        let baseline = 5e14 * kt * trajectories;
        let dynamics = 2.9e3 * kt * trajectories;

        // CONTROL: with a free readout this reproduces the paper\'s own ratio, which is a closed
        // form in its two published numbers and carries no physical constant at all.
        let free = advantage_after_readout(baseline, dynamics, reads, 0.0);
        let paper = 5e14 / 2.9e3;
        assert!((free - paper).abs() / paper < 1e-12, "free readout must reproduce the paper: {free} vs {paper}");
        assert!(free > 1e11, "and the paper says more than 1e11: {free}");

        // THE FINDING, in units nobody has to agree on a device to check.
        let budget = read_budget_for_advantage(baseline, dynamics, reads, 1e11).expect("attainable");
        for &t in &[300.0f64, 301.8] {
            // 300 K is this crate\'s constant; 301.8 K is what the paper\'s own kT implies. The
            // conclusion must not turn on which is used, so both are asserted.
            let floor = crate::floors::readout_floor(1.0, t);
            let in_floors = budget / floor;
            assert!(
                (1.0..1.25).contains(&in_floors),
                "at {t} K the readout budget is {in_floors} Landauer floors, and the claim is that it is barely above 1"
            );
        }

        // Ten times the floor already costs an order of magnitude of the headline.
        let floor = crate::floors::readout_floor(1.0, 300.0);
        let at_ten = advantage_after_readout(baseline, dynamics, reads, 10.0 * floor);
        assert!((2e10..3e10).contains(&at_ten), "at 10x the Landauer floor: {at_ten}");
        let at_264 = advantage_after_readout(baseline, dynamics, reads, 26.4 * floor);
        assert!((0.9e10..1.1e10).contains(&at_264), "at 26.4x the Landauer floor: {at_264}");

        // And what a real, metered read does to it. KV260_AXI_METERED is the only read price in the
        // catalogue graded Metered, and it is one single-beat AXI4-Lite read of one node.
        let metered = advantage_after_readout(baseline, dynamics, reads, KV260_AXI_METERED.e_read);
        assert!(metered < 10.0, "a metered readout leaves {metered}, not 1e11");
        assert!(metered > 1.0, "though the computer is still ahead, barely: {metered}");
        // The collapse is eleven orders of magnitude, and it is the readout, not the dynamics:
        // the dynamics term is negligible against the readout term at this price.
        let readout_only = reads as f64 * KV260_AXI_METERED.e_read;
        assert!(readout_only > 1e10 * dynamics, "the readout dwarfs the dynamics: {readout_only} vs {dynamics}");
        assert!(free / metered > 1e10, "the readout costs eleven orders of magnitude of the claim");
    }

}
