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
use ferrotherm::pdit::{FixedCumulative, FixedPdit};
use ferrotherm::potts::{ring, Interaction};
use std::process::Command;

/// Run yosys on one module and return `(cells, breakdown lines)` from the last `stat` it prints:
/// yosys 0.69 writes `      47 cells` and `       1   $_AND_` under "Printing statistics", and its
/// quiet flag suppresses exactly those lines, so the run is not quiet.
fn synth(dir: &std::path::Path, file: &str, top: &str) -> Option<(usize, Vec<String>)> {
    let script = format!("read_verilog {file}; synth -top {top}; stat");
    let out = Command::new("yosys")
        .current_dir(dir)
        .args(["-p", &script])
        .output()
        .ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    let start = text.rfind("Printing statistics")?;
    let mut cells = None;
    let mut lines = Vec::new();
    for line in text[start..].lines() {
        let t = line.trim();
        let mut parts = t.split_whitespace();
        let (Some(count), Some(what), None) = (parts.next(), parts.next(), parts.next()) else {
            continue;
        };
        let Ok(count) = count.parse::<usize>() else {
            continue;
        };
        if what == "cells" {
            cells = Some(count);
        } else if what.starts_with("$_") {
            lines.push(format!("{count:>6}  {what}"));
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
    // The same ring as cumulative units: one draw, an exponential ROM addressed by the field gap,
    // one multiply.
    let cumulative = FixedCumulative::new(&potts, 1.0, 1, 10).expect("Potts ring");
    std::fs::write(dir.join("cpdit.v"), cumulative.emit_verilog("cpdit")).expect("write");
    // A p-bit lattice of the same site count: 4 x 4, bipartite, one sigmoid ROM per node.
    let g = lattice2d(4, 1.0);
    let bit = FixedFabric::new(&g, 1.0, 1);
    std::fs::write(dir.join("pbit.v"), bit.emit_verilog("pbit")).expect("write");
    println!("== generic yosys synthesis, {sites} sites each ==");
    for (file, top, units, what) in [
        (
            "pdit.v",
            "pdit",
            sites,
            "p-trit (q = 3, ten-bit Gumbel ROM, three draws, three-way argmax)",
        ),
        (
            "cpdit.v",
            "cpdit",
            sites,
            "p-trit (q = 3, cumulative: gap-addressed exponential ROM, one draw, one multiply)",
        ),
        (
            "pbit.v",
            "pbit",
            sites,
            "p-bit (ten-bit sigmoid ROM, one draw, one comparator)",
        ),
    ] {
        match synth(&dir, file, top) {
            Some((cells, lines)) => {
                println!(
                    "{what}: {cells} cells, {:.0} per unit",
                    cells as f64 / units as f64
                );
                for l in lines.iter().take(12) {
                    println!("    {l}");
                }
            }
            None => println!("{what}: yosys gave no cell count"),
        }
    }
    println!();
    println!("A three-state variable is one p-trit, or three one-hot p-bits plus three penalty couplings,");
    println!(
        "or two domain-wall p-bits plus one; the cells above price the units, not the couplings."
    );
    println!("Weights are constants in these netlists, so a ROM addressed by a field folds to the field's");
    println!("reachable values (the sigmoid, the exponential); a ROM addressed by noise cannot (the Gumbel).");
    let _ = std::fs::remove_dir_all(&dir);
}
