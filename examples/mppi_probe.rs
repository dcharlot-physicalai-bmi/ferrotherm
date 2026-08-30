//! What does MPPI actually do on a system whose optimum we know? Measure before asserting.
use ferrotherm::mppi::{Lqr, Mppi, System};

/// The step count the three stable rows of the published table reproduce at. Named, because the
/// number the table reports depends on it and the table never said so.
const STEPS: usize = 200;
const SEED: u64 = 7;
const STABLE: System = System { a: 0.9, b: 1.0, q: 1.0, r: 0.5 };
const UNSTABLE: System = System { a: 1.1, b: 1.0, q: 1.0, r: 0.5 };

fn main() {
    for (label, s) in [("stable a=0.9", System { a: 0.9, b: 1.0, q: 1.0, r: 0.5 }),
                       ("unstable a=1.1", System { a: 1.1, b: 1.0, q: 1.0, r: 0.5 })] {
    println!("\n### {label}");
    let l = Lqr::solve(&s);
    let opt = l.cost_to_go(1.0);
    println!("LQR: p = {:.4}, k = {:.4}, optimal cost from x0=1 is {:.4}", l.p, l.k, opt);
    println!("{:>8} {:>7} {:>7} {:>6} {:>10} {:>10}", "horizon", "sigma", "lambda", "iters", "cost", "excess");
    for &h in &[5usize, 15] {
        for &sig in &[0.2f64, 0.4] {
            for &lam in &[0.3f64, 1.0] {
                for &it in &[1usize, 10] {
                    let m = Mppi { horizon: h, rollouts: 300, sigma: sig, lambda: lam, iters: it };
                    let c = m.run(&s, 1.0, 200, 7);
                    println!("{h:>8} {sig:>7.2} {lam:>7.2} {it:>6} {:>10.4} {:>9.2}%", c, (c - opt) / opt * 100.0);
                }
            }
        }
    }
    }

    // THE PUBLISHED TABLE, printed by the command that is cited as its source.
    //
    // It was not, until now. The doc in `src/mppi.rs` and the entry in `WORKLOADS.md` both point
    // here for a five-row table, and the sweep above produces neither unstable row -- it never runs
    // horizon 30 or iters 30. Two numbers were therefore published that no command in the
    // repository could print, and one of them (729%) was wrong by a factor of two.
    println!("\n\nTHE PUBLISHED TABLE, from this command, at steps = {STEPS} and seed {SEED}:\n");
    println!("{:>10} {:>9} {:>7} {:>7} {:>11} {:>11}", "plant", "horizon", "iters", "excess", "@100 steps", "@800 steps");
    for (label, sys, h, it) in [
        ("stable", &STABLE, 5usize, 10usize),
        ("stable", &STABLE, 5, 1),
        ("stable", &STABLE, 15, 10),
        ("unstable", &UNSTABLE, 10, 30),
        ("unstable", &UNSTABLE, 30, 30),
    ] {
        let m = Mppi { horizon: h, rollouts: 300, sigma: 0.2, lambda: 0.3, iters: it };
        let o = Lqr::solve(sys).cost_to_go(1.0);
        let at = |steps: usize| (m.run(sys, 1.0, steps, SEED) - o) / o * 100.0;
        println!("{label:>10} {h:>9} {it:>7} {:>6.1}% {:>10.1}% {:>10.1}%", at(STEPS), at(100), at(800));
    }

    println!(
        "\nTHE LAST TWO COLUMNS ARE THE POINT, AND THEY WERE NOT THERE BEFORE. `excess over the\n\
         provable optimum` IS NOT A PROPERTY OF THE METHOD. It grows without bound in the number of\n\
         steps, because MPPI injects sigma noise at every step forever while the LQR oracle's\n\
         `cost_to_go` is a finite infinite-horizon cost from x0 = 1. The flagship 7.1% is 1.0% at 25\n\
         steps and 22.6% at 800. It is a COORDINATE -- a number plus the horizon it was taken over --\n\
         and it was published as though it were a property, in WORKLOADS.md, in this module's doc\n\
         and in ROADMAP.md."
    );
    println!(
        "\nAND THE TWO UNSTABLE ROWS COULD NOT BOTH BE TRUE. The published pair was 15.7% at\n\
         horizon 10 and 729% at horizon 30. At {STEPS} steps -- where all three STABLE rows\n\
         reproduce to the printed digit -- they are 15.1% and 1446.0%. At 100 steps the second\n\
         becomes 733.5%, which is the published 729%, but the first becomes 7.2% rather than 15.7%.\n\
         There is no step count at which both published numbers appear, so they were not measured\n\
         in one run."
    );
}
