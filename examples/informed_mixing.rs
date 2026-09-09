#![allow(missing_docs)]
// Does weighting the proposal by the energy actually mix faster? Measured, flips against flips.
//
// THE CLAIM UNDER TEST is the one Zanella (JASA 2020) and Grathwohl et al. (ICML 2021) make for
// locally-informed proposals: choosing WHICH move to make by how much it would change the target
// beats choosing uniformly, by enough to pay for the bookkeeping. Every other single-flip sampler
// in this crate picks uniformly or in order, so if the claim holds here it is a gap worth having
// closed, and if it does not, that is worth writing down too.
//
// THE CONTROL THAT MAKES IT A MEASUREMENT. A locally-informed step flips ONE spin; a Gibbs sweep
// updates n. Comparing steps to sweeps would report the ratio n and nothing else. Both are
// therefore counted in FLIPS, and the per-flip cost is the same order: flipping k moves the field
// at its neighbours and nowhere else, so an informed step is O(deg), which is what one site of a
// Gibbs sweep costs. `updates/sample` is tau_int times the flips per draw, which is the work one
// independent draw actually costs -- the same unit `mixing_expressivity` prices in.
//
// WHERE IT SHOULD HELP AND WHERE IT SHOULD NOT. An informed proposal has something to say only when
// the weights are UNEQUAL. On a flat or high-temperature landscape every flip is about as good as
// every other, the proposal is nearly uniform, and this is Gibbs with extra arithmetic. The sweep
// below is over inverse temperature for that reason: the interesting column is the cold end.

use ferrotherm::certify::tau_int;
use ferrotherm::gibbs::Sampler;
use ferrotherm::graph::{Graph, GraphBuilder};
use ferrotherm::informed::{Balance, Informed};
use ferrotherm::rng::Pcg;

/// A frustrated ring with chords and fields — sparse, and not solvable by inspection.
fn frustrated(n: usize, seed: u64) -> Graph {
    let mut rng = Pcg::new(seed, 0xF5);
    let mut b = GraphBuilder::new(n);
    for i in 0..n {
        b.couple(i, (i + 1) % n, if rng.f64() < 0.5 { -1.0 } else { 1.0 });
    }
    for _ in 0..n / 4 {
        let (i, j) = ((rng.f64() * n as f64) as usize % n, (rng.f64() * n as f64) as usize % n);
        if i != j {
            b.couple(i, j, if rng.f64() < 0.5 { -1.0 } else { 1.0 });
        }
    }
    for i in 0..n {
        b.bias(i, (rng.f64() - 0.5) * 0.4);
    }
    b.build()
}

/// `tau_int` of the energy trace, in units of FLIPS, for a chain given `flips` total spin flips.
fn gibbs_tau(g: &Graph, beta: f64, flips: usize, seed: u64) -> (f64, f64) {
    let mut s = Sampler::new(g, beta, seed);
    let sweeps = flips / g.n;
    s.sweeps(sweeps / 10, None); // burn in
    let mut trace = Vec::with_capacity(sweeps);
    for _ in 0..sweeps {
        s.sweeps(1, None);
        trace.push(g.energy(&s.s));
    }
    let mean = trace.iter().sum::<f64>() / trace.len() as f64;
    // tau in sweeps -> tau in flips, since one sweep is n flips.
    (tau_int(&trace) * g.n as f64, mean)
}

fn informed_tau(g: &Graph, beta: f64, flips: usize, seed: u64, b: Balance) -> (f64, f64, f64) {
    let mut it = Informed::new(g, beta, seed).with_balance(b);
    let draws = flips / g.n;
    it.steps(flips / 10);
    let mut trace = Vec::with_capacity(draws);
    for _ in 0..draws {
        it.steps(g.n); // one draw per n flips, the same opportunity a sweep gets
        trace.push(it.energy());
    }
    let mean = trace.iter().sum::<f64>() / trace.len() as f64;
    (tau_int(&trace) * g.n as f64, mean, it.acceptance())
}

fn main() {
    let n = 256;
    let flips = 4_000 * n;
    println!(
        "Locally-informed proposals against Gibbs, on a frustrated ring with chords\n\n\
         {n} spins, {flips} flips per chain, tau_int in FLIPS (lower is better).\n\
         Both arms get the same number of spin flips, which is the same order of work.\n"
    );
    println!(
        "{:>6} {:>12} {:>12} {:>12} {:>12} {:>10}",
        "beta", "gibbs", "sqrt", "barker", "metropolis", "sqrt acc"
    );

    for &beta in &[0.2f64, 0.5, 1.0, 2.0, 4.0] {
        let mut row = [0.0f64; 4];
        let mut acc = 0.0;
        let seeds = 4u64;
        for seed in 0..seeds {
            let g = frustrated(n, seed);
            row[0] += gibbs_tau(&g, beta, flips, seed).0;
            for (k, b) in [Balance::Sqrt, Balance::Barker, Balance::Metropolis].iter().enumerate() {
                let (t, _, a) = informed_tau(&g, beta, flips, seed, *b);
                row[k + 1] += t;
                if k == 0 {
                    acc += a;
                }
            }
        }
        for v in &mut row {
            *v /= f64::from(u32::try_from(seeds).unwrap());
        }
        acc /= f64::from(u32::try_from(seeds).unwrap());
        println!(
            "{beta:>6.1} {:>12.0} {:>12.0} {:>12.0} {:>12.0} {:>10.3}",
            row[0], row[1], row[2], row[3], acc
        );
    }
    println!(
        "\nThe informed columns are worth their bookkeeping only where they beat the first one.\n\
         An acceptance near 1.0 means the proposal landscape barely moved -- which is the regime\n\
         where this has nothing to offer, and is why it is printed beside the times."
    );
}
