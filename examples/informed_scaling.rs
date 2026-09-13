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

use ferrotherm::certify::{tau_int, tau_int_batch};
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
fn gibbs_tau(g: &Graph, beta: f64, flips: usize, seed: u64) -> (f64, bool) {
    let mut s = Sampler::new(g, beta, seed);
    let sweeps = flips / g.n;
    s.sweeps(sweeps / 10, None);
    let mut trace = Vec::with_capacity(sweeps);
    for _ in 0..sweeps {
        s.sweeps(1, None);
        trace.push(g.energy(&s.s));
    }
    let (t, truncated) = checked_tau(&trace);
    (t * g.n as f64, truncated)
}

/// Sokal's window cross-checked by batch means, exactly as `certify()` does it since 2026-09-13:
/// the larger of the two is carried, and `true` marks a row where the window closed on a fast
/// mode while batch means saw a slow one. Even the larger value is then a lower bound.
fn checked_tau(trace: &[f64]) -> (f64, bool) {
    let s = tau_int(trace);
    let b = tau_int_batch(trace, 20);
    if b.is_finite() && b > 2.0 * s { (b, true) } else { (s, false) }
}

/// `tau_int` of the energy in FLIPS for the Barker-informed chain, one draw per `n` steps.
fn informed_tau(g: &Graph, beta: f64, flips: usize, seed: u64) -> (f64, f64, bool) {
    let mut it = Informed::new(g, beta, seed).with_balance(Balance::Barker);
    let draws = flips / g.n;
    it.steps(flips / 10);
    let mut trace = Vec::with_capacity(draws);
    for _ in 0..draws {
        it.steps(g.n);
        trace.push(it.energy());
    }
    let (t, truncated) = checked_tau(&trace);
    (t * g.n as f64, it.acceptance(), truncated)
}

/// An effective sample size below this cannot support a ratio: `tau_int`'s relative error is about
/// `1/sqrt(ESS)`, so 25 is +-20%. The FIRST run of this example printed a "roughly size-independent"
/// verdict from rows whose Gibbs chain held 9 to 14 effective samples, and the verdict was an
/// artefact of the budget: every row is now escalated until both arms clear this floor, or the
/// cap says how far it got and the row is printed as unresolved.
const MIN_ESS: f64 = 25.0;

fn main() {
    let beta = 2.0;
    let seeds = 4u64;
    let start_flips_per_spin = 4_000usize;
    let cap_flips_per_spin = 128_000usize;
    println!("THE INFORMED PROPOSAL'S ADVANTAGE AGAINST n, AND AGAINST WHAT A FLIP COSTS\n");
    println!("  fixture   frustrated ring with n/4 chords and fields (the informed_mixing fixture), beta = {beta}");
    println!("  budget    from {start_flips_per_spin} flips per spin per chain, doubled until both arms hold ESS >= {MIN_ESS:.0}");
    println!("            (cap {cap_flips_per_spin}), {seeds} seeds per size, tau_int in FLIPS\n");
    println!(
        "  {:>5} {:>5} {:>7}   {:>9} {:>9} {:>7}   {:>10} {:>10} {:>10}   {:>8} {:>8}  {:>6}  {:>5} {:>5}",
        "n", "deg", "f/spin", "tau G", "tau B", "G/B", "work G", "work B:scan", "work B:tree", "adv:scan", "adv:tree", "acc", "ESS G", "ESS B"
    );
    let mut resolved: Vec<(usize, f64, f64, f64)> = Vec::new();
    let mut unresolved: Vec<(usize, f64, f64)> = Vec::new();
    for &n in &[64usize, 128, 256, 512, 1024, 2048] {
        let mut flips_per_spin = start_flips_per_spin;
        let (tg, tb, acc, deg, ess_g, ess_b, trunc_g, trunc_b) = loop {
            let flips = flips_per_spin * n;
            let draws = flips / n;
            let (mut tg, mut tb, mut acc, mut deg) = (0.0, 0.0, 0.0, 0.0);
            let (mut trunc_g, mut trunc_b) = (0u32, 0u32);
            for seed in 0..seeds {
                let g = frustrated(n, seed);
                deg += mean_degree(&g);
                let (t, tr) = gibbs_tau(&g, beta, flips, seed);
                tg += t;
                trunc_g += u32::from(tr);
                let (t, a, tr) = informed_tau(&g, beta, flips, seed);
                tb += t;
                acc += a;
                trunc_b += u32::from(tr);
            }
            let s = seeds as f64;
            let (tg, tb, acc, deg) = (tg / s, tb / s, acc / s, deg / s);
            // tau is in flips and the trace holds one draw per n flips, so ESS = draws / (2 tau/n).
            let ess_g = draws as f64 / (2.0 * tg / n as f64);
            let ess_b = draws as f64 / (2.0 * tb / n as f64);
            if (ess_g >= MIN_ESS && ess_b >= MIN_ESS) || flips_per_spin >= cap_flips_per_spin {
                break (tg, tb, acc, deg, ess_g, ess_b, trunc_g, trunc_b);
            }
            flips_per_spin = (flips_per_spin * 2).min(cap_flips_per_spin);
        };
        // Work per flip under each model. Gibbs: the field, deg+1 terms. Informed: deg+1 reweighs
        // plus the choice -- n reads for the scan, log2(n) for the tree.
        let per_g = deg + 1.0;
        let per_b_scan = deg + 1.0 + n as f64;
        let per_b_tree = deg + 1.0 + (n as f64).log2();
        let (wg, wbs, wbt) = (tg * per_g, tb * per_b_scan, tb * per_b_tree);
        let ok = ess_g >= MIN_ESS && ess_b >= MIN_ESS;
        // A truncation count is how many of the seeds had batch means exceed twice Sokal's window
        // on that arm; those rows carry the batch-means tau, which is itself a lower bound.
        println!(
            "  {n:>5} {deg:>5.2} {flips_per_spin:>7}   {tg:>9.0} {tb:>9.0} {:>7.2}   {wg:>10.3e} {wbs:>10.3e} {wbt:>10.3e}   {:>8.2} {:>8.2}  {acc:>6.3}  {ess_g:>5.0} {ess_b:>5.0}  trunc G {trunc_g}/{seeds} B {trunc_b}/{seeds}{}",
            tg / tb,
            wg / wbs,
            wg / wbt,
            if ok { "" } else { "  UNRESOLVED" }
        );
        if ok {
            resolved.push((n, tg / tb, wg / wbs, wg / wbt));
        } else {
            unresolved.push((n, ess_g, ess_b));
        }
    }
    println!("\n  WHAT THE TABLE SAYS.\n");
    println!("  G/B is the per-flip advantage `informed_mixing` reports for one size. adv:scan is what");
    println!("  that advantage was worth as WORK before 2026-09-13, when choosing a site scanned every");
    println!("  weight; adv:tree is what it is worth now. A value below 1 means the informed chain costs");
    println!("  MORE work per independent sample than Gibbs, whatever its flips say.");
    let scan_wins = resolved.iter().filter(|r| r.2 > 1.0).count();
    let tree_wins = resolved.iter().filter(|r| r.3 > 1.0).count();
    println!(
        "\n  Over the {} resolved sizes: under the SCAN model the informed chain beat Gibbs in work at {} of them;\n  under the TREE model at {} of them.",
        resolved.len(),
        scan_wins,
        tree_wins
    );
    if let (Some(first), Some(last)) = (resolved.first(), resolved.last()) {
        let trend = last.1 / first.1;
        let peak = resolved.iter().map(|r| r.1).fold(0.0f64, f64::max);
        println!(
            "  Per-flip advantage over the resolved sizes, n = {} to n = {}: {:.2}x -> {:.2}x (peak {peak:.2}x), a factor of\n  {trend:.2} across a {}x range in n. {}",
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
    for (n, eg, eb) in &unresolved {
        println!(
            "  n = {n}: UNRESOLVED at the cap -- Gibbs ESS {eg:.0}, informed ESS {eb:.0}; its ratios are printed and not counted."
        );
    }
    println!("\n  tau_int's relative error is about 1/sqrt(ESS), which is why the ESS columns are there and why");
    println!("  a row is escalated until both clear {MIN_ESS:.0}. Four seeds are averaged; the spread between");
    println!("  them is the reproducibility of every ratio above, and no ratio is tighter than it.");
}
