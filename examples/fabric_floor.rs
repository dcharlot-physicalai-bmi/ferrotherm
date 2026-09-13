#![allow(missing_docs)]
// WHY THE FABRIC RELAXES FASTER THAN THE EXACT KERNEL WHEN COLD: the comparator floor, measured.
//
// `fabric_exact` at beta 3 on the 4x3 grid: Kemeny's constant is 1.15e8 for the fabric against
// 4.48e9 for the exact chromatic kernel. K is the sum of every mode's relaxation time and a
// property of the kernel alone, so the fabric's slowest mode really is 39x faster -- this is NOT
// the variance weighting that can make a tau_int ratio lie, and the note that first read the
// tau ratio that way was wrong. The suspect is the 16-bit comparator. A flip probability is
// round(p * 65535) / 65536, so the probability of the UNLIKELY state is quantised to multiples of
// 2^-16 = 1.53e-5: rounded to exactly 0 below 7.6e-6 when the unlikely state is +1, and never
// below 1.53e-5 when it is -1, because the entry cannot exceed 65535. The unlikely state's
// probability is sigma(-2 beta f), the heat bath's factor of two included, so the zero arrives once
// 2 beta f > 11.8: at beta 2 every field above 2.95 forbids its flip, and at beta 3 a flip against a
// field of 4.2 (exact 1e-11) is 0 on one side and 1.53e-5 -- a million times too likely -- on the
// other; an escape from a metastable valley that needs several such flips compounds the factor.
// If that is the mechanism, the K ratio must return to 1 as the
// comparator gains bits with the field and ROM bits held at the shipped 8 and 10 -- and it must
// not as the field or ROM bits grow with the comparator held at 16.
//
// run: cargo run --release --example fabric_floor

use ferrotherm::autocorr::{boltzmann, kemeny_constant, stationary_solved, total_variation, Kernel};
use ferrotherm::graph::{Graph, GraphBuilder};
use ferrotherm::rng::Pcg;

fn grid_glass(w: usize, h: usize, seed: u64) -> Graph {
    let mut rng = Pcg::new(seed, 0x6A);
    let mut b = GraphBuilder::new(w * h);
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            if x + 1 < w {
                b.couple(i, i + 1, if rng.f64() < 0.5 { -1.0 } else { 1.0 });
            }
            if y + 1 < h {
                b.couple(i, i + w, if rng.f64() < 0.5 { -1.0 } else { 1.0 });
            }
        }
    }
    for i in 0..w * h {
        b.bias(i, (rng.f64() - 0.5) * 0.4);
    }
    b.build()
}

fn main() {
    let g = grid_glass(4, 3, 7);
    println!("THE COMPARATOR FLOOR AND THE FABRIC'S RELAXATION, EXACTLY, ON ALL 2^{} STATES\n", g.n);
    println!("  model    4x3 open grid, random +-1 couplings, fields in [-0.2, 0.2] (the fabric_exact fixture)");
    println!("  measure  Kemeny's constant K of the Quantised kernel against the exact chromatic kernel; TV of its law from Boltzmann\n");
    for &beta in &[2.0f64, 3.0] {
        let pi = boltzmann(&g, beta).expect("small");
        let k_exact = kemeny_constant(&g, beta, Kernel::ChromaticGibbs).expect("small");
        println!("  beta {beta:.1}: exact kernel K = {k_exact:.4e}\n");
        println!("  {:<26} {:>10} {:>10} {:>12} {:>9}", "kernel {frac, rom, prob}", "floor", "TV(law,B)", "K", "K/K_exact");
        let rows: Vec<(&str, Kernel)> = vec![
            ("{8, 10, 12}", Kernel::Quantised { frac_bits: 8, lut_bits: 10, prob_bits: 12 }),
            ("{8, 10, 16} shipped", Kernel::Quantised { frac_bits: 8, lut_bits: 10, prob_bits: 16 }),
            ("{8, 10, 20}", Kernel::Quantised { frac_bits: 8, lut_bits: 10, prob_bits: 20 }),
            ("{8, 10, 24}", Kernel::Quantised { frac_bits: 8, lut_bits: 10, prob_bits: 24 }),
            ("{8, 10, 32}", Kernel::Quantised { frac_bits: 8, lut_bits: 10, prob_bits: 32 }),
            ("{12, 12, 16}", Kernel::Quantised { frac_bits: 12, lut_bits: 12, prob_bits: 16 }),
        ];
        for (name, k) in rows {
            let Kernel::Quantised { prob_bits, .. } = k else { unreachable!() };
            let floor = 2f64.powi(-(prob_bits as i32));
            let law = stationary_solved(&g, beta, k).expect("small");
            let tv = total_variation(&law, &pi);
            let kk = kemeny_constant(&g, beta, k).expect("small");
            println!("  {name:<26} {floor:>10.2e} {tv:>10.3e} {kk:>12.4e} {:>9.3}", kk / k_exact);
        }
        println!();
    }
    println!("  WHAT THE TABLE SAYS.\n");
    println!("  'floor' is 2^-prob: the smallest nonzero probability the comparator can express, and the granularity of");
    println!("  every probability near 0 or 1. If K/K_exact climbs toward 1 down the comparator column while the {{12, 12, 16}}");
    println!("  row stays where the shipped fabric is, the fabric's fast relaxation when cold is the comparator's floor");
    println!("  inflating its rarest flips, not the field or ROM precision -- and not the weighting of any observable.");
}
