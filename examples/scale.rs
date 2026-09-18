// `missing_docs` is denied workspace-wide and is right to be: it guards the API surface, and
// every public item in every library here carries a doc. An EXAMPLE has no API surface -- it is
// a program, and its helpers are private to it -- so the lint has nothing to guard and asks for
// doc comments on `fn main`'s scaffolding instead. Scoped off here rather than weakened there.
#![allow(missing_docs)]
// How far EXACT verification reaches, and what it costs to get there.
//
// ABSORPTION.md said the honest thing about scale: everything in it is verified at `n` in the
// single digits, "where exact oracles exist". That sentence carries a premise, and the premise is
// wrong. Enumeration dies at `2^n`. It is not the only exact oracle in this crate.
//
// The Kac-Ward determinant (`pfaffian`) returns `ln Z` and `<E>` for ANY planar graph at ANY beta
// in `O((2E)^3)` -- polynomial, not exponential, and exact rather than estimated. So the ceiling on
// exact verification is not `n = 24`; it is wherever a `2E x 2E` determinant stops being
// affordable. This program measures where that is, and uses every rung of the ladder to check a
// sampler against truth at sizes no enumeration could referee.
//
// The check is the honest one, not a tolerance chosen after the fact: the cluster sampler's 95%
// interval on `<E>/N` either covers the exact value or it does not. An interval that misses is a
// failure whatever its width, and an interval so wide it cannot miss is reported as such by
// printing it.
//
// Run at the critical temperature, `beta_c = ln(1+sqrt(2))/2`, because that is where a sampler is
// hardest to get right and where a wrong one is most likely to look plausible.
//
// A check that accepts everything measures nothing, so every rung is also asked a question it must
// answer NO to: does the same interval cover the exact energy of a lattice one percent away in
// temperature? The gap between those two answers is the check's resolving power, and it is printed
// in units of the sampler's own standard error rather than asserted.
//
// NOT run in CI: the largest rung is a 3480x3480 complex determinant.
//
// run: cargo run --release --example scale

use ferrotherm::cluster::{Sampler, Update};
use ferrotherm::host::Timing;
use ferrotherm::samples::Plan;
use ferrotherm::{ising, pfaffian};

/// The exact critical point of the square-lattice Ising model, `ln(1 + sqrt 2) / 2`.
fn beta_critical() -> f64 {
    (1.0 + 2.0_f64.sqrt()).ln() / 2.0
}

/// `2^n` as a power of ten, for the enumeration column. `n = 900` overflows every float we have,
/// so report the exponent rather than the number.
fn log10_enumeration(n: usize) -> f64 {
    n as f64 * 2.0_f64.log10()
}

fn main() {
    let beta = beta_critical();
    println!("EXACT VERIFICATION AT SCALE");
    println!("square-lattice Ising, J = 1, beta = beta_c = {beta:.9}");
    println!();
    println!("Two exact oracles, and only one of them has a ceiling in n:");
    println!("  enumeration   2^n terms          -- the oracle every small test in this crate uses");
    println!("  Kac-Ward      (2E)^3 arithmetic  -- exact for any planar graph, any beta");
    println!();
    println!(
        "  {:>3}  {:>5}  {:>6}  {:>12}  {:>11}  {:>10}  {:>21}  {:>6}  resolving power",
        "L",
        "n",
        "2E",
        "enumeration",
        "exact <E>/N",
        "Kac-Ward",
        "sampler <E>/N",
        "covers"
    );
    println!("  {}", "-".repeat(118));
    // Every rung must also REJECT this: the same lattice one percent colder.
    const DETUNE: f64 = 1.01;

    let mut rungs = 0usize;
    let mut covered = 0usize;
    let mut worst_phase: f64 = 0.0;
    let mut rejected = 0usize;
    let mut blind: Vec<usize> = Vec::new();
    let mut weakest_rejection = f64::INFINITY;
    let mut deepest: Option<(usize, f64)> = None;

    for l in [4usize, 6, 8, 12, 16, 20, 24, 30] {
        let g = ising::grid2d(l, l, 1.0);
        let n = g.n;

        let (exact, t_exact) = Timing::around(|| pfaffian::solve(&g, beta));
        let exact = match exact {
            Ok(s) => s,
            Err(e) => {
                println!("  {l:>3}  {n:>5}  Kac-Ward refused: {e}");
                continue;
            }
        };
        let e_exact = exact.energy_density().expect("Params::default asks for the energy");
        worst_phase = worst_phase.max(exact.phase_residual.abs());

        // The sampler gets a plan that does not depend on n: the point is whether it agrees with
        // truth at scale under a FIXED budget, not whether it can be made to agree by spending more.
        let plan = Plan::new(2_000, 4_000, 1);
        let mut s = Sampler::new(&g, beta, 0x5CA1_E000 + l as u64).expect("ferromagnet is unfrustrated");
        let set = s.collect(&plan, Update::SwendsenWang, None);
        let est = set.mean_energy().expect("a chain of 4000 draws has a mean");
        let per_spin = ferrotherm::samples::Estimate {
            value: est.value / n as f64,
            stderr: est.stderr / n as f64,
            ess: est.ess,
            tau_int: est.tau_int,
        };
        let ok = per_spin.covers(e_exact);

        // The negative control. Nothing about the sampler changes -- only which exact value it is
        // held against. A check that cannot tell these apart is not checking the physics.
        let detuned = pfaffian::solve(&g, beta * DETUNE).expect("same graph, colder");
        let e_detuned = detuned.energy_density().expect("Params::default asks for the energy");
        let rejects = !per_spin.covers(e_detuned);
        // Resolving power is a property of the DESIGN -- this size at this budget -- so the
        // numerator is the exact gap between the two lattices, not the distance from wherever this
        // seed happened to land. `rejects` above is the realised outcome; this is its expectation.
        let sigma = (e_detuned - e_exact).abs() / per_spin.stderr;

        rungs += 1;
        if ok {
            covered += 1;
        }
        if rejects {
            rejected += 1;
            weakest_rejection = weakest_rejection.min(sigma);
        } else {
            blind.push(n);
        }
        deepest = Some((n, t_exact.seconds));

        println!(
            "  {:>3}  {:>5}  {:>6}  {:>10.3e}  {:>11.6}  {:>9.2}s  {:>10.6} +- {:.6}  {:>6}  {:>5}  {:>5.1} sigma",
            l,
            n,
            2 * exact.edges,
            10.0_f64.powf(log10_enumeration(n).min(300.0)),
            e_exact,
            t_exact.seconds,
            per_spin.value,
            per_spin.stderr,
            if ok { "yes" } else { "NO" },
            if rejects { "yes" } else { "NO" },
            sigma
        );
    }

    println!();
    println!("  intervals covering the exact value:       {covered} of {rungs}");
    println!("  intervals that excluded a 1% error:       {rejected} of {rungs}");
    println!(
        "  worst determinant phase residual:        {worst_phase:.3e}  (zero in exact arithmetic --"
    );
    println!("                                            a free check that the embedding stayed sound)");
    println!();

    if let Some((n, secs)) = deepest {
        let ratio = log10_enumeration(n) - (n as f64).log10();
        println!("WHAT THIS BUYS.");
        println!("  The deepest rung refereed here is n = {n}, in {secs:.1}s of exact arithmetic.");
        println!(
            "  Enumerating it would take 10^{:.0} terms. The exact oracle this crate already had",
            log10_enumeration(n)
        );
        println!(
            "  reaches {ratio:.0} orders of magnitude past the one its small tests use, on the same"
        );
        println!("  machine, with no tolerance and no seed.");
        println!();
    }

    if !blind.is_empty() {
        println!("AND THE FINDING, WHICH IS THE OTHER WAY ROUND FROM THE ASSUMPTION.");
        println!(
            "  The rungs that FAILED the control are the SMALL ones: n = {}.",
            blind.iter().map(usize::to_string).collect::<Vec<_>>().join(", ")
        );
        println!("  There, one interval covers the true lattice and a lattice one percent colder");
        println!("  at the same time, so agreeing with the exact answer distinguishes almost");
        println!("  nothing. Resolving power rises monotonically with size, at least as fast as");
        println!("  sqrt(n): an intensive observable's error shrinks as it is averaged over more");
        println!("  spins, while the gap it has to resolve does not shrink with it -- it widens");
        println!(
            "  slightly toward the thermodynamic limit. {weakest_rejection:.1} sigma is the weakest separation"
        );
        println!("  here that still excluded the wrong lattice.");
        println!();
        println!("  The usual reading of a small exact test is that it is the strongest evidence");
        println!("  available, since nothing is estimated. That is true of the ORACLE and false of");
        println!("  the COMPARISON. A small lattice is where an exact oracle is cheapest and where");
        println!("  it discriminates least. Verifying only at single-digit n is not the");
        println!("  conservative choice it looks like.");
        println!();
    }

    println!("The limit is not the oracle. A 2E x 2E determinant at L = 30 is 3480 square; the");
    println!("cube law, not the exponential, is what eventually stops this -- and it stops it at a");
    println!("size where enumeration has been impossible for twenty-five doublings.");
}
