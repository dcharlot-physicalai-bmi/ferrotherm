//! Decoding a Sourlas code on the fixed-point p-bit fabric at the Nishimori temperature, across
//! channel noise levels: where `beta_N` grows past the comparator floor, what happens to the bit
//! error -- measured against the exact finite-temperature decoder and the exact Gibbs kernel.
//!
//! ```text
//!   cargo run --release --example sourlas_fabric
//! ```
//!
//! Counts over sampled noise patterns, so valid on a busy machine; each pattern's decoders are
//! exact (enumeration and direct solves), so the only randomness is the channel's.

use ferrotherm::autocorr::Kernel;
use ferrotherm::sourlas::{
    bit_error, bsc, kernel_marginals, mpm_decode, nishimori_beta, posterior, Code,
};

fn main() {
    let code = Code::random(10, 15, 2, 77)
        .expect("a code")
        .with_message_bits();
    let xi: Vec<i8> = (0..10).map(|i| if i % 3 == 0 { -1 } else { 1 }).collect();
    let word = code.encode(&xi).expect("length");
    let degree = (0..code.k)
        .map(|i| code.checks.iter().filter(|s| s.contains(&i)).count())
        .max()
        .unwrap_or(0);
    println!(
        "Sourlas code: k = {}, m = {} (message bits sent too), C = 2, rate {:.2}, max degree {degree}; the fabric floors once 2 beta f > 11.8",
        code.k,
        code.m(),
        code.rate()
    );
    println!(
        "{:>7} {:>8} {:>9} {:>10} {:>10} {:>10} {:>8}",
        "p", "beta_N", "2bN*deg", "BER exact", "BER gibbs", "BER fabric", "frozen"
    );
    let patterns = 40u64;
    for &p in &[0.3f64, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005] {
        let beta = nishimori_beta(p);
        let (mut exact_ber, mut gibbs_ber, mut fabric_ber) = (0.0, 0.0, 0.0);
        let mut frozen = 0usize;
        let mut decoded_by_fabric = 0usize;
        for n in 0..patterns {
            let received = bsc(&word, p, 1000 + n);
            let post = posterior(&code, &received, beta).expect("small");
            exact_ber += bit_error(&mpm_decode(&post.marginals), &xi);
            let gibbs =
                kernel_marginals(&code, &received, beta, Kernel::SequentialGibbs).expect("dense");
            gibbs_ber += bit_error(&mpm_decode(&gibbs), &xi);
            match kernel_marginals(&code, &received, beta, Kernel::FixedFabric) {
                Ok(marg) => {
                    fabric_ber += bit_error(&mpm_decode(&marg), &xi);
                    decoded_by_fabric += 1;
                }
                Err(_) => frozen += 1,
            }
        }
        let n = patterns as f64;
        let fabric_mean = if decoded_by_fabric > 0 {
            fabric_ber / decoded_by_fabric as f64
        } else {
            f64::NAN
        };
        println!(
            "{p:>7.3} {beta:>8.3} {:>9.1} {:>10.4} {:>10.4} {:>10.4} {frozen:>5}/{patterns}",
            2.0 * beta * degree as f64,
            exact_ber / n,
            gibbs_ber / n,
            fabric_mean
        );
    }
    println!();
    println!("BER fabric averages the patterns the fabric's chain could decode (its law has one closed class);");
    println!("frozen counts the patterns where the direct solve found more than one, and no decoder at all.");
}
