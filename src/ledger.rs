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
pub const CATALOGUE: [(&str, Prices); 5] = [
    ("UNSTATED", Prices::UNSTATED),
    ("Z1_SPICE", Z1_SPICE),
    ("KV260_MEASURED", KV260_MEASURED),
    // APPENDED, never inserted: the binding surfaces enumerate this table by index.
    ("PEGASUS_28NM_ASIC", PEGASUS_28NM_ASIC),
    ("KV260_AXI_METERED", KV260_AXI_METERED),
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
        assert_eq!(metered, ["KV260_MEASURED", "KV260_AXI_METERED"], "both on the one board we own");
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
        assert_eq!(CATALOGUE.len(), 5);
        assert_eq!(CATALOGUE[3].0, "PEGASUS_28NM_ASIC");
        assert_eq!(CATALOGUE[4].0, "KV260_AXI_METERED", "appended, so every earlier index holds");
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
}
