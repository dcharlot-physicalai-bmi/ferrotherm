//! What a quantum annealer charges before it anneals: the published access-time model with the
//! published per-solver constants, priced at the vendor's wall power, beside the ledger's Gibbs
//! update.
//!
//! ```text
//!   cargo run --release --example qpu_access_cost
//! ```
//!
//! Every number here follows from constants a vendor publishes and changes when they do; the
//! module records the date they were read.

use ferrotherm::access::{
    Submission, ADVANTAGE2_SYSTEM1, ADVANTAGE2_SYSTEM2, ADVANTAGE2_SYSTEM4, ADVANTAGE_SYSTEM4,
    ADVANTAGE_SYSTEM6,
};
use ferrotherm::ledger::{KV260_MEASURED, Z1_SPICE};

fn main() {
    let qpus = [
        ADVANTAGE_SYSTEM4,
        ADVANTAGE_SYSTEM6,
        ADVANTAGE2_SYSTEM1,
        ADVANTAGE2_SYSTEM2,
        ADVANTAGE2_SYSTEM4,
    ];
    println!(
        "{:<20} {:>6} {:>10} {:>9} {:>9} {:>11} {:>11} {:>11} {:>9}",
        "solver", "qubits", "fixed ms", "J/read@1", "@100", "@10k", "uJ/qubit@10k", "amortise", "at 1%"
    );
    for q in &qpus {
        // A 20 us anneal and a readout at the middle of the solver's published range.
        let readout = 0.5 * (q.readout_us.0 + q.readout_us.1);
        let one = Submission::new(1, 20.0, readout);
        let hundred = Submission::new(100, 20.0, readout);
        let ten_k = Submission::new(10_000, 20.0, readout);
        println!(
            "{:<20} {:>6} {:>10.1} {:>9.1} {:>9.2} {:>11.3} {:>11.0} {:>11} {:>9}",
            q.name,
            q.qubits,
            one.fixed_us(q) * 1e-3,
            one.joules_per_read(q),
            hundred.joules_per_read(q),
            ten_k.joules_per_read(q),
            ten_k.joules_per_qubit_read(q) * 1e6,
            one.reads_to_amortise(q, 1.0),
            one.reads_to_amortise(q, 0.01),
        );
    }
    let q = &ADVANTAGE_SYSTEM4;
    let ten_k = Submission::new(10_000, 20.0, 100.0);
    let per_qubit = ten_k.joules_per_qubit_read(q);
    println!();
    println!(
        "{} at ten thousand reads: {:.3} J per read, {:.1} uJ per qubit read.",
        q.name,
        ten_k.joules_per_read(q),
        per_qubit * 1e6
    );
    println!(
        "  against one Gibbs update: {:.2e}x a Z1-class SPICE update ({:.2} fJ), {:.2e}x a KV260 flip ({:.2} pJ, measured)",
        per_qubit / Z1_SPICE.e_sample,
        Z1_SPICE.e_sample * 1e15,
        per_qubit / KV260_MEASURED.e_sample,
        KV260_MEASURED.e_sample * 1e12
    );
    println!("  wall power: {} W ({})", q.wall_watts, q.power_source);
    println!("  constants read on {}", q.read_on);
}
