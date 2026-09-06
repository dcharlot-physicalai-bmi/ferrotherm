//! Run the conformance suite on ourselves, on every backend the core crate can reach, and print
//! the results unedited.
//!
//! A conformance suite whose author exempts themselves is worthless. Two backends here rather than
//! one, because the interesting question is not whether a fabric passes but whether **one program
//! means the same thing on two different arithmetics**: `Cpu` works in `f64`, and `hdl::RtlFabric`
//! works in the Q.8 fixed point, 1024-entry sigmoid ROM and per-node xorshift32 that were
//! implemented for `xck26` and metered on a Kria KV260 at 10.85 pJ per node update.
//!
//! The GPU backend scores through the same suite; it lives in `ferrotherm-gpu` because it needs
//! wgpu, and `conform_can_finally_score_the_gpu_path` there is where it runs.
use ferrotherm::fabric::Device;

fn report(name: &str, dev: &mut dyn Device) -> bool {
    let r = ferrotherm::conform::run(dev);
    println!("{r}\n");
    let l = dev.ledger();
    // What the suite COST that backend, not merely how it scored. The counts are the
    // machine-independent half; `ledger::CATALOGUE` holds the machines that can price them.
    println!(
        "  ledger: {} node updates, {} reads, {} writes",
        l.samples, l.reads, l.writes
    );
    let p = dev.fabric().prices;
    match l.joules(&p) {
        Some(j) => println!("  {j:.3e} J on {}", p.source),
        None => {
            // Name the operation that has no price, rather than only that one does not. The
            // difference matters: KV260_MEASURED is a real wattmeter reading for SAMPLING and
            // states nothing about reads or writes, because those were never exercised on that
            // board. Unstated, and not zero.
            let unpriced: Vec<&str> = [
                ("sampling", l.samples, p.e_sample),
                ("reads", l.reads, p.e_read),
                ("writes", l.writes, p.e_write),
            ]
            .iter()
            .filter(|(_, count, price)| *count > 0 && !price.is_finite())
            .map(|(what, _, _)| *what)
            .collect();
            let (list, verb) = (unpriced.join(", "), if unpriced.len() == 1 { "is" } else { "are" });
            println!("  no joules figure for {name}: it did {list}, and those prices {verb} unstated.");
            println!("  Borrowing another machine's would look exactly like a measurement.");
        }
    }
    println!();
    r.passed()
}

fn main() {
    let cpu = report("cpu", &mut ferrotherm::fabric::Cpu::default());
    let rtl = report("ferrotherm-pbit-rtl", &mut ferrotherm::hdl::RtlFabric::new());

    if !cpu {
        println!("WE FAIL OUR OWN SUITE ON THE CPU. That is published as-is.");
    }
    if !rtl {
        // Expected today, and worth printing rather than hiding: the v1 fabric updates two colour
        // classes in two clock cycles, and the suite's frustrated 5-ring is an odd cycle. A
        // refusal that names the reason is a better answer than a wrong one.
        println!(
            "The RTL fabric declines at least one case. Read the FAIL line: a refusal that names \
             its reason is the fabric telling you what it cannot express."
        );
    }
}
