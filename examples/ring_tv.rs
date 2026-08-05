// The 8-site Ising ring, exact vs sampled — the honest version of the only hardware demo in the
// Aug-2026 thermodynamic-computing release (there, ONE calibrated silicon pbit fed randomness to a
// software Metropolis loop; here, the whole sampler is software and says so).
//
// PASS = TV(sampled, exact) is statistically indistinguishable from TV(sampled, sampled') at the
// same sample count: the sampler's residual is sampling noise, not bias.
//
// run: cargo run --release --example ring_tv

use ferrotherm::gibbs::Sampler;
use ferrotherm::ising::{exact_boltzmann, ring, tv};

fn hist(smp: &mut Sampler, n_samples: usize, thin: usize) -> Vec<f64> {
    let m = 1usize << smp.g.n;
    let mut counts = vec![0u64; m];
    for _ in 0..n_samples {
        smp.sweeps(thin, None);
        let mut mask = 0usize;
        for b in 0..smp.g.n {
            if smp.s[b] == 1 {
                mask |= 1 << b;
            }
        }
        counts[mask] += 1;
    }
    counts.iter().map(|&c| c as f64 / n_samples as f64).collect()
}

fn main() {
    let beta = 1.5; // the beta used in the published X0 demo
    let g = ring(8, 1.0, 0.15); // small field to break symmetry: harder than the symmetric case
    let p_exact = exact_boltzmann(&g, beta);

    let n_samples = 100_000;
    let thin = 3;

    let mut a = Sampler::new(&g, beta, 0xA11CE);
    a.sweeps(500, None);
    let pa = hist(&mut a, n_samples, thin);

    let mut b = Sampler::new(&g, beta, 0xB0B);
    b.sweeps(500, None);
    let pb = hist(&mut b, n_samples, thin);

    let tv_exact = tv(&pa, &p_exact);
    let tv_noise = tv(&pa, &pb); // two independent runs: the pure-noise floor at this sample count

    println!("8-site Ising ring, beta = {beta}, J = 1, h = 0.15, {n_samples} samples (thin {thin})");
    println!("  TV(sampled, exact Boltzmann) = {tv_exact:.4}");
    println!("  TV(sampled, sampled')        = {tv_noise:.4}   <- noise floor at this sample count");
    let ok = tv_exact < 1.5 * tv_noise + 0.005;
    println!("  verdict: {}", if ok { "PASS — residual is sampling noise, not bias" } else { "FAIL — sampler is biased" });
    std::process::exit(if ok { 0 } else { 1 });
}
