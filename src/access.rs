//! What a submission costs before a single spin flips: the published QPU access-time model of a
//! quantum annealer, its published per-solver constants, and the joules a wall-power figure turns
//! them into — beside the ledger's price for one Gibbs update.
//!
//! # Why this exists
//!
//! [`crate::ledger`] prices a p-bit fabric per operation: femtojoules per update, picojoules per
//! read. A quantum annealer is priced the other way round, by the second: D-Wave documents the QPU
//! access time of a submission as
//!
//! ```text
//!   T = T_p + Delta + R (T_a + T_r + T_d)
//! ```
//!
//! programming time (which "includes the postprogramming thermalization time", typically 1 ms),
//! an initialisation overhead Delta ("roughly 10–20 ms for Advantage systems"), and per read the
//! anneal, readout and delay times — and publishes, per solver, the programming time, the
//! readout-time range and the delay time. The whole system draws a stated wall power: "a mere 12.5
//! kilowatts", the same "over six generations", at Advantage2's 2025 general availability. The
//! vendor's 2017 whitepaper (14-1005A) put "the power draw of a D-Wave system" at "nominally 16
//! kW"; the two figures are both the vendor's and disagree by a quarter, so the constant here is
//! the later one and `wall_watts` is a field a caller can set to the other.
//! Multiplying the two is the only honest way to put an annealer on the same axis as a fabric, and
//! it puts the fixed cost first: on `Advantage_system4` a single-read submission with a 20 µs
//! anneal and a 100 µs readout occupies the QPU for about 29 ms and the system for about 370 J —
//! before the one anneal it asked for. Ten thousand reads amortise that to about 1.8 J per read. A
//! Z1-class Gibbs update is 7 fJ in the same ledger, fifty quadrillion times less.
//!
//! # What this is not
//!
//! Not a claim about anneal quality, quantum advantage, or the useful work in a read. It is the
//! published timing model with the published constants, so that a cost-per-sample comparison that
//! includes an annealer states the annealer's fixed cost rather than its marginal one. The wall
//! power is the whole-system figure the vendor publishes, cooling included; the per-solver
//! constants are the ones on the solver-properties page on 2026-09-14, and they change with every
//! solver revision, so [`Qpu`] carries the date it was read.

/// A quantum annealer's published timing constants.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Qpu {
    /// The solver name as published.
    pub name: &'static str,
    /// Working qubits.
    pub qubits: u32,
    /// Programming time, microseconds, as published ("∼ 14100 µs").
    pub programming_us: f64,
    /// Readout time per sample, microseconds: the published low and high ends of the range.
    pub readout_us: (f64, f64),
    /// QPU delay time per sample, microseconds.
    pub delay_us: f64,
    /// Wall power of the whole system, watts, from the vendor figure named in `power_source`.
    pub wall_watts: f64,
    /// Where the wall power comes from.
    pub power_source: &'static str,
    /// The date the solver-properties page was read.
    pub read_on: &'static str,
}

/// The vendor's whole-system power figure, which it states has held over six generations.
pub const WALL_POWER_SOURCE: &str =
    "D-Wave, Advantage2 general availability, 2025-05-20: 'the same amount of electricity over six generations -- a mere 12.5 kilowatts'";

/// `Advantage_system4`: 5,627 qubits, programming ∼14,100 µs, readout 17–235 µs, delay 20.5 µs.
pub const ADVANTAGE_SYSTEM4: Qpu = Qpu {
    name: "Advantage_system4",
    qubits: 5627,
    programming_us: 14_100.0,
    readout_us: (17.0, 235.0),
    delay_us: 20.5,
    wall_watts: 12_500.0,
    power_source: WALL_POWER_SOURCE,
    read_on: "2026-09-14",
};

/// `Advantage_system6`: 5,612 qubits, programming ∼14,200 µs, readout 18–173 µs, delay 20.5 µs.
pub const ADVANTAGE_SYSTEM6: Qpu = Qpu {
    name: "Advantage_system6",
    qubits: 5612,
    programming_us: 14_200.0,
    readout_us: (18.0, 173.0),
    delay_us: 20.5,
    wall_watts: 12_500.0,
    power_source: WALL_POWER_SOURCE,
    read_on: "2026-09-14",
};

/// `Advantage2_system1`: 4,577 qubits, programming ∼33,600 µs, readout 17–101 µs, delay 60.6 µs.
pub const ADVANTAGE2_SYSTEM1: Qpu = Qpu {
    name: "Advantage2_system1",
    qubits: 4577,
    programming_us: 33_600.0,
    readout_us: (17.0, 101.0),
    delay_us: 60.6,
    wall_watts: 12_500.0,
    power_source: WALL_POWER_SOURCE,
    read_on: "2026-09-14",
};

/// `Advantage2_system2`: 4,514 qubits, programming ∼24,500 µs, readout 17–95 µs, delay 20.6 µs.
pub const ADVANTAGE2_SYSTEM2: Qpu = Qpu {
    name: "Advantage2_system2",
    qubits: 4514,
    programming_us: 24_500.0,
    readout_us: (17.0, 95.0),
    delay_us: 20.6,
    wall_watts: 12_500.0,
    power_source: WALL_POWER_SOURCE,
    read_on: "2026-09-14",
};

/// `Advantage2_system4`: 1,202 qubits, programming ∼8,000 µs, readout 17–45 µs, delay 20.6 µs.
pub const ADVANTAGE2_SYSTEM4: Qpu = Qpu {
    name: "Advantage2_system4",
    qubits: 1202,
    programming_us: 8_000.0,
    readout_us: (17.0, 45.0),
    delay_us: 20.6,
    wall_watts: 12_500.0,
    power_source: WALL_POWER_SOURCE,
    read_on: "2026-09-14",
};

/// The published QPU access overhead for Advantage systems, "roughly 10–20 ms": its midpoint.
pub const ACCESS_OVERHEAD_US: f64 = 15_000.0;

/// Post-programming thermalisation, "typically, 1 ms" -- already inside the published programming
/// time, which "includes the postprogramming thermalization time", so it is not added again.
pub const PROGRAMMING_THERMALIZATION_US: f64 = 1_000.0;

/// One submission: how many reads, at what anneal time, with which readout time within the
/// solver's range, and the overheads.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Submission {
    /// Reads requested.
    pub reads: u32,
    /// Anneal time per read, microseconds.
    pub anneal_us: f64,
    /// Readout time per read, microseconds; a problem-size-dependent value inside the solver's range.
    pub readout_us: f64,
    /// Access overhead, microseconds; [`ACCESS_OVERHEAD_US`] unless measured.
    pub overhead_us: f64,
}

impl Submission {
    /// `reads` reads at `anneal_us` with a readout of `readout_us`, the documented overheads.
    #[must_use]
    pub fn new(reads: u32, anneal_us: f64, readout_us: f64) -> Submission {
        Submission {
            reads,
            anneal_us,
            readout_us,
            overhead_us: ACCESS_OVERHEAD_US,
        }
    }

    /// The per-read term `T_a + T_r + T_d`, microseconds.
    #[must_use]
    pub fn per_read_us(&self, qpu: &Qpu) -> f64 {
        self.anneal_us + self.readout_us + qpu.delay_us
    }

    /// The fixed term `T_p + Delta`, microseconds: what is paid before any read.
    #[must_use]
    pub fn fixed_us(&self, qpu: &Qpu) -> f64 {
        qpu.programming_us + self.overhead_us
    }

    /// `T = T_p + Delta + R (T_a + T_r + T_d)`, microseconds.
    #[must_use]
    pub fn access_time_us(&self, qpu: &Qpu) -> f64 {
        self.fixed_us(qpu) + f64::from(self.reads) * self.per_read_us(qpu)
    }

    /// Joules the whole system draws over the access time.
    #[must_use]
    pub fn joules(&self, qpu: &Qpu) -> f64 {
        qpu.wall_watts * self.access_time_us(qpu) * 1e-6
    }

    /// Joules per read, the fixed cost amortised over the reads. `NaN` for zero reads.
    #[must_use]
    pub fn joules_per_read(&self, qpu: &Qpu) -> f64 {
        self.joules(qpu) / f64::from(self.reads)
    }

    /// Joules per qubit read: [`Self::joules_per_read`] over the solver's working qubits.
    #[must_use]
    pub fn joules_per_qubit_read(&self, qpu: &Qpu) -> f64 {
        self.joules_per_read(qpu) / f64::from(qpu.qubits)
    }

    /// The reads at which the fixed cost per read falls to `fraction` of the per-read cost:
    /// `fixed / (fraction x per_read)`, rounded up. At 1.0 the two are equal; at 0.01 the
    /// submission is 99% sampling.
    #[must_use]
    pub fn reads_to_amortise(&self, qpu: &Qpu, fraction: f64) -> u32 {
        (self.fixed_us(qpu) / (fraction * self.per_read_us(qpu)))
            .ceil()
            .max(1.0) as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ledger::Z1_SPICE;

    /// The identity, and the pinned headline: one read on `Advantage_system4` at a 20 µs anneal and
    /// a 100 µs readout is about 29 ms of QPU and about 370 J of system -- numbers that follow from
    /// the published constants and move when they do.
    #[test]
    fn one_read_costs_a_third_of_a_kilojoule_before_it_anneals() {
        let q = &ADVANTAGE_SYSTEM4;
        let s = Submission::new(1, 20.0, 100.0);
        let t = s.access_time_us(q);
        let want = 14_100.0 + 15_000.0 + (20.0 + 100.0 + 20.5);
        assert!((t - want).abs() < 1e-9, "access time {t} vs {want}");
        let j = s.joules(q);
        assert!((j - 365.5).abs() / 365.5 < 0.01, "one read: {j} J");
        let against = j / Z1_SPICE.e_sample;
        assert!(
            (against - 5.16e16).abs() / 5.16e16 < 0.01,
            "against a Z1 SPICE Gibbs update: {against:e}"
        );
    }

    /// Amortisation: ten thousand reads bring the per-read cost to about 1.8 J, and the closed form
    /// for the reads at which fixed equals per-read is the published fixed time over the per-read
    /// time.
    #[test]
    fn reads_amortise_the_fixed_cost_as_the_closed_form_says() {
        let q = &ADVANTAGE_SYSTEM4;
        let s = Submission::new(10_000, 20.0, 100.0);
        let per = s.joules_per_read(q);
        assert!(
            per > 1.7 && per < 1.9,
            "per read at ten thousand reads: {per} J"
        );
        // (14,100 + 15,000) / (20 + 100 + 20.5) = 207.1, rounded up; a hundred times that at 1%.
        assert_eq!(s.reads_to_amortise(q, 1.0), 208);
        assert_eq!(s.reads_to_amortise(q, 0.01), 20_712);
        // The per-qubit read on the small Advantage2_system4 is the most expensive of the five at
        // ten thousand reads: a quarter of the qubits at the same wall power.
        let mut per_qubit: Vec<(f64, &str)> = [
            ADVANTAGE_SYSTEM4,
            ADVANTAGE_SYSTEM6,
            ADVANTAGE2_SYSTEM1,
            ADVANTAGE2_SYSTEM2,
            ADVANTAGE2_SYSTEM4,
        ]
        .iter()
        .map(|q| {
            (
                Submission::new(10_000, 20.0, 40.0).joules_per_qubit_read(q),
                q.name,
            )
        })
        .collect();
        per_qubit.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        assert_eq!(per_qubit.last().map(|x| x.1), Some("Advantage2_system4"));
    }
}
