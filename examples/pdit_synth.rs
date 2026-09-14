//! What a p-trit costs in cells: synthesise the emitted p-dit fabric and a p-bit fabric of the
//! same site count with yosys's generic flow, and report cells per unit -- the embodiment's price
//! in logic, beside the noise draws the module already counts.
//!
//! ```text
//!   cargo run --release --example pdit_synth
//! ```
//!
//! Needs `yosys` on the path; says so and stops if it is not there. Generic cells (`synth` with
//! no target library) are a relative measure: the same flow on both fabrics, so the ratio means
//! something and the absolute count does not.

use ferrotherm::hdl::FixedFabric;
use ferrotherm::ising::lattice2d;
use ferrotherm::pdit::FixedPdit;
use ferrotherm::potts::{ring, Interaction};
use std::process::Command;

/// Run yosys on one module and return `(cells, breakdown lines)` from its `stat`.
fn synth(dir: &std::path::Path, file: &str, top: &str) -> Option<(usize, Vec<String>)> {
    let script = format!("read_verilog {file}; synth -top {top}; stat");
    let out = Command::new("yosys")
        .current_dir(dir)
        .args(["-q", "-p", &script])
        .output()
        .ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    let mut cells = None;
    let mut lines = Vec::new();
    let mut in_stat = false;
    for line in text.lines() {
        let t = line.trim();
        if t.starts_with("Number of cells:") {
            cells = t.split(':').nth(1).and_then(|v| v.trim().parse().ok());
            in_stat = true;
            continue;
        }
        if in_stat {
            if t.is_empty() {
                in_stat = false;
                continue;
            }
            if t.starts_with("$_") {
                lines.push(t.to_string());
            }
        }
    }
    cells.map(|c| (c, lines))
}

fn main() {
    if Command::new("yosys").arg("-V").output().is_err() {
        println!("yosys is not on the path; nothing to synthesise.");
        return;
    }
    let dir = std::env::temp_dir().join(format!("ferrotherm_pdit_synth_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let sites = 16usize;
    // A p-trit ring: 16 three-state sites, each a Gumbel-max unit with a ten-bit ROM.
    let potts = ring(sites, 3, 1.0, Interaction::Potts);
    let trit = FixedPdit::new(&potts, 1.0, 1, 10).expect("Potts ring");
    std::fs::write(dir.join("pdit.v"), trit.emit_verilog("pdit")).expect("write");
    // A p-bit lattice of the same site count: 4 x 4, bipartite, one sigmoid ROM per node.
    let g = lattice2d(4, 1.0);
    let bit = FixedFabric::new(&g, 1.0, 1);
    std::fs::write(dir.join("pbit.v"), bit.emit_verilog("pbit")).expect("write");
    println!("== generic yosys synthesis, {sites} sites each ==");
    for (file, top, units, what) in [
        ("pdit.v", "pdit", sites, "p-trit (q = 3, ten-bit Gumbel ROM, three draws, three-way argmax)"),
        ("pbit.v", "pbit", sites, "p-bit (ten-bit sigmoid ROM, one draw, one comparator)"),
    ] {
        match synth(&dir, file, top) {
            Some((cells, lines)) => {
                println!("{what}: {cells} cells, {:.0} per unit", cells as f64 / units as f64);
                for l in lines.iter().take(12) {
                    println!("    {l}");
                }
            }
            None => println!("{what}: yosys gave no cell count"),
        }
    }
    println!();
    println!("A three-state variable is one p-trit, or three one-hot p-bits plus three penalty couplings,");
    println!("or two domain-wall p-bits plus one; the cells above price the units, not the couplings.");
    let _ = std::fs::remove_dir_all(&dir);
}
