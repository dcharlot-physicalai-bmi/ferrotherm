//! What a native p-trit buys over p-bits, exactly: Kemeny's constant of the native heat-bath
//! chain against the one-hot and domain-wall spellings of the same three-state model at several
//! penalties, in steps, draws and sweeps, with the stationary mass each encoding leaves on codes
//! that decode to nothing -- and the fixed-point p-trit's floor against its ROM width.
//!
//! ```text
//!   cargo run --release --example pdit_exact
//! ```

use ferrotherm::autocorr::total_variation;
use ferrotherm::hdl::FRAC;
use ferrotherm::pdit::{
    bits_per_charge, encoded_row, gumbel_rom, native_row, rom_conditional, rom_span, softmax,
    Charge, Embodiment, Row, MIXING_SWEEPS_CAP,
};
use ferrotherm::potts::{ring, Interaction};

fn print_row(r: &Row) {
    let mix = r
        .mixing_sweeps
        .map_or("> cap".to_string(), |t| t.to_string());
    let mix_draws = r
        .mixing_draws
        .map_or("> cap".to_string(), |t| t.to_string());
    println!(
        "{:<20} {:>7.2} {:>7} {:>6} {:>7} {:>9} {:>9.3} {:>9.2} {:>8.4}",
        r.embodiment.label(),
        r.penalty,
        r.states,
        r.draws_per_sweep,
        mix,
        mix_draws,
        r.tau_int_sweeps,
        r.tau_int_draws,
        r.valid_mass
    );
}

fn main() {
    println!("== the radix economy, three charges (bits per unit charged) ==");
    for q in 2..=5usize {
        let unit = if q == 2 {
            Embodiment::Comparator
        } else {
            Embodiment::GumbelMax
        };
        println!(
            "  q = {q}: radix {:.3}  comparators {:.3}  draws {:.3}",
            bits_per_charge(q, unit, Charge::Radix),
            bits_per_charge(q, unit, Charge::Comparators),
            bits_per_charge(q, unit, Charge::Draws)
        );
    }
    println!();
    println!("== native p-trit vs its encodings: mixing time to TV 1/4 and tau_int of [x_0 = 0], exact ==");
    for &(n, beta) in &[
        (3usize, 0.5f64),
        (3, 1.0),
        (3, 2.0),
        (4, 0.5),
        (4, 1.0),
        (4, 2.0),
    ] {
        let m = ring(n, 3, 1.0, Interaction::Potts);
        println!("-- three-state Potts ring, n = {n}, J = 1, beta = {beta} --");
        println!(
            "{:<20} {:>7} {:>7} {:>6} {:>7} {:>9} {:>9} {:>9} {:>8}",
            "embodiment",
            "penalty",
            "states",
            "dr/sw",
            "mix sw",
            "mix dr",
            "tau sw",
            "tau dr",
            "valid"
        );
        match native_row(&m, beta, MIXING_SWEEPS_CAP) {
            Ok(r) => print_row(&r),
            Err(e) => println!("native: {e}"),
        }
        for embodiment in [Embodiment::DomainWall, Embodiment::OneHot] {
            for penalty in [0.5f64, 1.0, 2.0, 4.0] {
                match encoded_row(&m, beta, embodiment, penalty, MIXING_SWEEPS_CAP) {
                    Ok(r) => print_row(&r),
                    Err(e) => println!("{} at {penalty}: {e}", embodiment.label()),
                }
            }
        }
    }
    println!();
    println!("== the fixed-point p-trit's floor: TV from the softmax at fields (0, -g, -g), and the span ==");
    let scale = f64::from(1u32 << FRAC);
    print!("{:<6} {:>7}", "bits", "span");
    for g in [1.0f64, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0] {
        print!(" {:>9}", format!("g={g}"));
    }
    println!();
    for bits in [8u32, 10, 12, 14, 16] {
        let rom = gumbel_rom(bits);
        print!("{bits:<6} {:>7.2}", rom_span(&rom));
        for g in [1.0f64, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0] {
            let fields = [0.0, -g, -g];
            let fields_q: Vec<i32> = fields.iter().map(|f| (f * scale).round() as i32).collect();
            let law = rom_conditional(&fields_q, &rom);
            let tv = total_variation(&law, &softmax(&fields, 1.0));
            if law[1] == 0.0 {
                print!(" {:>9}", format!("{tv:.1e}*"));
            } else {
                print!(" {tv:>9.1e}");
            }
        }
        println!();
    }
    println!("  (* the trailing state has probability exactly zero: the floor)");
}
