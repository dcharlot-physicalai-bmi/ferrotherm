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

/// Operation counts accumulated by a run.
#[derive(Clone, Copy, Debug, Default)]
pub struct Ledger {
    pub samples: u64,
    pub reads: u64,
    pub writes: u64,
}

impl Ledger {
    /// Energy under `p`, or `None` when `p` states no prices.
    ///
    /// `Option`, not a number, and not zero. A machine whose per-operation energy nobody has
    /// published does not cost nothing, and a caller that has to unwrap this cannot accidentally
    /// print a figure for a device that has none.
    pub fn joules(&self, p: &Prices) -> Option<f64> {
        p.is_stated().then(|| {
            self.samples as f64 * p.e_sample
                + self.reads as f64 * p.e_read
                + self.writes as f64 * p.e_write
        })
    }

    /// The wall-clock floor this many full-graph reflashes implies, or `None` if the device states
    /// no cap.
    ///
    /// A run that reflashes faster than the hardware sustains is not fast; it is unphysical, and
    /// pricing it describes something that could not have happened.
    pub fn reflash_seconds(&self, p: &Prices, graph_nodes: u64) -> Option<f64> {
        let hz = p.reflash_hz_cap?;
        if graph_nodes == 0 || hz <= 0.0 {
            return None;
        }
        Some(self.writes as f64 / graph_nodes as f64 / hz)
    }

    /// Fractional breakdown (sample, read, write) of total energy under prices `p`.
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
