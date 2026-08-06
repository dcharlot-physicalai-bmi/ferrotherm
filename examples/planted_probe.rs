//! How does difficulty vary with planted-loop density? Measure, do not assume.
use ferrotherm::oracle::{Annealer, Solver, SteepestDescent};
use ferrotherm::planted::{frustrated_loops, wishart};
use ferrotherm::schedule::Schedule;

fn main() {
    let l = 8;
    let edges = 2 * l * l; // periodic square lattice
    println!("8x8 lattice, {edges} edges. excess above the planted optimum, mean over 5 seeds:");
    println!("{:>7} {:>11} {:>12} {:>10} {:>10}", "loops", "hits/edge", "greedy solved", "worst", "anneal worst");
    println!("(greedy: solve rate and worst excess over a 4x4 instance/solver seed grid)");
    for loops in [8, 16, 32, 64, 128, 256, 512, 1024, 4096] {
        let sched = Schedule::geometric(0.05, 6.0, 60, 20);
        let (mut solved, mut total, mut worst, mut aworst) = (0, 0, 0.0f64, 0.0f64);
        for iseed in 1..=4u64 {
            let p = frustrated_loops(l, loops, iseed);
            for sseed in 1..=4u64 {
                let e = p.excess(&SteepestDescent { restarts: 50, seed: sseed }.solve(&p.graph).0);
                total += 1;
                if e <= 1e-9 { solved += 1; }
                worst = worst.max(e);
                aworst = aworst
                    .max(p.excess(&Annealer { schedule: sched.clone(), seed: sseed }.solve(&p.graph).0));
            }
        }
        println!(
            "{loops:>7} {:>11.2} {:>7}/{:<4} {:>10.4} {:>10.4}",
            4.0 * loops as f64 / edges as f64, solved, total, worst, aworst
        );
    }

    println!();
    println!("Wishart planted ensemble, n=24, greedy with 100 restarts over a 4x4 seed grid:");
    println!("{:>7} {:>14} {:>10} {:>12}", "alpha", "greedy solved", "worst", "anneal worst");
    for alpha in [0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 3.0] {
        let sched = Schedule::geometric(0.05, 6.0, 60, 20);
        let (mut solved, mut total, mut worst, mut aworst) = (0, 0, 0.0f64, 0.0f64);
        for iseed in 1..=4u64 {
            let p = wishart(24, alpha, iseed);
            for sseed in 1..=4u64 {
                let e = p.excess(&SteepestDescent { restarts: 100, seed: sseed }.solve(&p.graph).0);
                total += 1;
                if e <= 1e-9 { solved += 1; }
                worst = worst.max(e);
                aworst = aworst
                    .max(p.excess(&Annealer { schedule: sched.clone(), seed: sseed }.solve(&p.graph).0));
            }
        }
        println!("{alpha:>7.2} {:>9}/{:<4} {:>10.4} {:>12.4}", solved, total, worst, aworst);
    }
}
