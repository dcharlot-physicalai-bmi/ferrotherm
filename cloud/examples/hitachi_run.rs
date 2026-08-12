//! Run a program on Hitachi's CMOS annealing ASIC through the Device trait.
//!
//! Needs ACW_TOKEN. The token is never committed; it comes from the environment.
use ferrotherm::fabric::Device;
use ferrotherm::ftp::Program;
use ferrotherm::schedule::Schedule;
use ferrotherm_cloud::hitachi::{Hitachi, Machine};

fn main() {
    let mut d = match Hitachi::from_env(Machine::Asic) {
        Ok(d) => d,
        Err(e) => { eprintln!("{e}"); return; }
    };
    let f = d.fabric();
    println!("fabric        {} | {:?} | {} sites | degree {} | coupling {:?}",
             f.name, f.topology, f.max_spins.unwrap(), f.max_degree.unwrap(), f.coupling_precision);

    // A 4x4 antiferromagnetic block, laid out row-major so every coupling is King-adjacent.
    let side = Machine::Asic.side();
    let mut src = format!("ftp 1\nname acw-4x4-antiferro\nspins {}\n", 3 * side + 4);
    let mut edges = 0;
    for y in 0..4usize {
        for x in 0..4usize {
            let i = y * side + x;
            if x + 1 < 4 { src.push_str(&format!("factor -1 {i} {}\n", i + 1)); edges += 1; }
            if y + 1 < 4 { src.push_str(&format!("factor -1 {i} {}\n", i + side)); edges += 1; }
        }
    }
    let p = Program::from_ftp(&src).expect("program");
    println!("program       {} spins declared, {edges} couplings, digest {:016x}", p.spins, p.digest());

    let bad = d.program(&p);
    if !bad.is_empty() {
        for u in &bad { println!("REFUSED: {u}"); }
        return;
    }

    match d.run(&Schedule::geometric(0.1, 10.0, 20, 50), 1) {
        Err(e) => println!("run failed: {e}"),
        Ok(state) => {
            let g = p.to_graph().unwrap();
            println!("machine energy (their sign) {:?}", d.last_energies);
            println!("execution     {:.3} ms on the ASIC", d.last_execution_ns as f64 / 1e6);
            println!("our energy    {}", g.energy(&state));
            let ok = (0..4).all(|y| (0..4).all(|x| {
                let i = y * side + x;
                state[i] == if (x + y) % 2 == 0 { state[0] } else { -state[0] }
            }));
            println!("checkerboard  {}", if ok { "yes - every bond satisfied" } else { "no" });
            println!("ledger        {} node updates", d.ledger().samples);
        }
    }
}
