//! The device energy ledger — first-class, because on this hardware class the story IS the I/O.
//!
//! Prices are per-node-operation costs of a device model. For the Z1-class costs (SPICE-derived,
//! pre-silicon; arXiv:2608.01615 Table IV) a WRITE costs ~21,700 Gibbs cycles and a READ ~239.
//! The vendor's own conclusion follows from these three numbers: the architecture wins where
//! "many local updates are performed between infrequent I/O operations" — and the ledger makes
//! that arithmetic executable instead of promotional.

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
    source: "Z1-class SPICE estimates, arXiv:2608.01615 Table IV — taped-out but uncharacterised \
             silicon, not measured. Applies to that device model and to nothing else.",
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
/// p-bit — and it is the first entry in that comparison that is not itself a projection.
///
/// # Why reads and writes are unstated
///
/// They were not measured. This run clocked a free-running fabric with no host traffic, so
/// [`Ledger::joules`] **refuses** any workload that touches them rather than pricing them at zero.
/// That refusal is the point of the type.
pub const KV260_MEASURED: Prices = Prices {
    e_sample: 1.0848e-11,
    e_read: f64::NAN,
    e_write: f64::NAN,
    reflash_hz_cap: None,
    source: "MEASURED on a Kria KV260 (xck26), 2026-09-06: 1,024 p-bits at 100 MHz drew 0.5554 W \
             above an idle PL on the SOM's INA260 (5 V rail, whole board), 23.1 sigma, over 51.2 \
             flips/ns. Reads and writes were not exercised and are unstated, not zero.",
};

/// Operation counts accumulated by a run.
#[derive(Clone, Copy, Debug, Default)]
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
}
