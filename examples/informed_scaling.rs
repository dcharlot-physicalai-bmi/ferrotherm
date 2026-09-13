#![allow(missing_docs)]
// DOES THE INFORMED PROPOSAL'S PER-FLIP ADVANTAGE SURVIVE THE SIZE OF THE MODEL, AND THE COST OF
// CHOOSING?
//
// `informed_mixing` reports that a locally-informed (Barker) proposal mixes 41x faster than Gibbs
// per FLIP at beta = 2 on a 256-spin frustrated ring. Two things that number does not say:
//
//   1. How it moves with n. Zanella (JASA 2020) and Grathwohl et al. (ICML 2021) report
//      "order of magnitude" gains as though the factor were a property of the method. On a glass
//      the slow modes are extended, and a proposal that chooses one site at a time may buy less as
//      there are more sites to choose among -- or more, if the landscape is sharper. That is a
//      measurement, not a derivation.
//
//   2. What a flip COSTS. Until 2026-09-13 an informed step drew its site by a linear scan of all
//      n weights, so each "flip" did O(n) work against a Gibbs flip's O(deg) -- and the module's
//      own doc said O(deg). The selection is now a Fenwick tree, O(log n). tau_int in flips is a
//      property of the chain's law and did not move; what this example does is price the same
//      flips under BOTH cost models, so the reader can see what the 41x was worth before the fix
//      and what it is worth after.
//
// UNITS. tau_int is in flips for every arm (a Gibbs sweep is n flips; one informed draw is taken
// per n steps, the same opportunity a sweep gets), so `gibbs/barker` is the per-flip advantage.
// A Gibbs flip costs deg+1 field terms. An informed flip reweighs the flipped site and its
// neighbours (deg+1 exponentials) and then chooses: n weight reads under the scan, about log2(n)
// tree reads under the tree. `work` columns multiply tau by those, so they are the flips-per-
// independent-sample of `informed_mixing` restated as WORK per independent sample, and the
// advantage columns are the ratios that survive each model. Nothing here is wall-clock, so this
// runs on a busy machine and the numbers are the same on any machine.
//
// run: cargo run --release --example informed_scaling

use ferrotherm::certify::tau_int;
use ferrotherm::gibbs::Sampler;
use ferrotherm::graph::{Graph, GraphBuilder};
use ferrotherm::informed::{Balance, Informed};
use ferrotherm::rng::Pcg;

/// The `informed_mixing` fixture at any size: a frustrated ring with n/4 random chords and fields.
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

fn mean_degree(g: &Graph) -> f64 {
    2.0 * g.n_edges as f64 / g.n as f64
}

/// `tau_int` of the energy in FLIPS for Gibbs, given `flips` spin flips after a 10% burn-in.
fn gibbs_tau(g: &Graph, beta: f64, flips: usize, seed: u64) -> f64 {
    let mut s = Sampler::new(g, beta, seed);
    let sweeps = flips / g.n;
    s.sweeps(sweeps / 10, None);
    let mut trace = Vec::with_capacity(sweeps);
    for _ in 0..sweeps {
        s.sweeps(1, None);
        trace.push(g.energy(&s.s));
    }
    tau_int(&trace) * g.n as f64
}

/// `tau_int` of the energy in FLIPS for the Barker-informed chain, one draw per `n` steps.
fn informed_tau(g: &Graph, beta: f64, flips: usize, seed: u64) -> (f64, f64) {
    let mut it = Informed::new(g, beta, seed).with_balance(Balance::Barker);
    let draws = flips / g.n;
    it.steps(flips / 10);
    let mut trace = Vec::with_capacity(draws);
    for _ in 0..draws {
        it.steps(g.n);
        trace.push(it.energy());
    }
    (tau_int(&trace) * g.n as f64, it.acceptance())
}

fn main() {
    let beta = 2.0;
    let seeds = 4u64;
    let flips_per_spin = 4_000usize;
    println!("THE INFORMED PROPOSAL'S ADVANTAGE AGAINST n, AND AGAINST WHAT A FLIP COSTS\n");
    println!("  fixture   frustrated ring with n/4 chords and fields (the informed_mixing fixture), beta = {beta}");
    println!("  budget    {flips_per_spin} flips per spin per chain, {seeds} seeds per size, tau_int in FLIPS\n");
    println!(
        "  {:>5} {:>5}   {:>9} {:>9} {:>7}   {:>10} {:>10} {:>10}   {:>8} {:>8}  {:>6}",
        "n", "deg", "tau G", "tau B", "G/B", "work G", "work B:scan", "work B:tree", "adv:scan", "adv:tree", "acc"
    );
    let mut per_flip: Vec<(usize, f64)> = Vec::new();
    for &n in &[64usize, 128, 256, 512, 1024, 2048] {
        let flips = flips_per_spin * n;
        let (mut tg, mut tb, mut acc, mut deg) = (0.0, 0.0, 0.0, 0.0);
        for seed in 0..seeds {
            let g = frustrated(n, seed);
            deg += mean_degree(&g);
            tg += gibbs_tau(&g, beta, flips, seed);
            let (t, a) = informed_tau(&g, beta, flips, seed);
            tb += t;
            acc += a;
        }
        let s = seeds as f64;
        let (tg, tb, acc, deg) = (tg / s, tb / s, acc / s, deg / s);
        // Work per flip under each model. Gibbs: the field, deg+1 terms. Informed: deg+1 reweighs
        // plus the choice -- n reads for the scan, log2(n) for the tree.
        let per_g = deg + 1.0;
        let per_b_scan = deg + 1.0 + n as f64;
        let per_b_tree = deg + 1.0 + (n as f64).log2();
        let (wg, wbs, wbt) = (tg * per_g, tb * per_b_scan, tb * per_b_tree);
        println!(
            "  {n:>5} {deg:>5.2}   {tg:>9.0} {tb:>9.0} {:>7.2}   {wg:>10.3e} {wbs:>10.3e} {wbt:>10.3e}   {:>8.2} {:>8.2}  {acc:>6.3}",
            tg / tb,
            wg / wbs,
            wg / wbt
        );
        per_flip.push((n, tg / tb));
    }
    println!("\n  WHAT THE TABLE SAYS.\n");
    println!("  G/B is the per-flip advantage `informed_mixing` reports for one size. adv:scan is what");
    println!("  that advantage was worth as WORK before 2026-09-13, when choosing a site scanned every");
    println!("  weight; adv:tree is what it is worth now. A value below 1 means the informed chain costs");
    println!("  MORE work per independent sample than Gibbs, whatever its flips say.");
    if let (Some(first), Some(last)) = (per_flip.first(), per_flip.last()) {
        let trend = last.1 / first.1;
        println!(
            "\n  Per-flip advantage from n = {} to n = {}: {:.2}x -> {:.2}x, a factor of {trend:.2} across a {}x\n  range in n. {}",
            first.0,
            last.0,
            first.1,
            last.1,
            last.0 / first.0,
            if trend > 2.0 {
                "It GROWS with size on this fixture."
            } else if trend < 0.5 {
                "It SHRINKS with size on this fixture: 'order of magnitude' is a small-n statement here."
            } else {
                "It is roughly size-independent on this fixture."
            }
        );
    }
    println!("\n  tau_int's relative error is about 1/sqrt(ESS); at the largest n each chain holds");
    println!("  {flips_per_spin} draws, so a tau above ~200 draws (= 200 n flips) is not resolved and the");
    println!("  row should be read as a bound. Four seeds are averaged; the spread between them is the");
    println!("  reproducibility of every ratio above, and no ratio is tighter than it.");
}
