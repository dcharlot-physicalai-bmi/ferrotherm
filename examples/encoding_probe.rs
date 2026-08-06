//! Does the encoding choice actually pay? Measure the feasibility rate, not the counting argument.
use ferrotherm::categorical::Categorical;
use ferrotherm::encode::Encoding;
use ferrotherm::schedule::Schedule;

fn rate(n: usize, k: usize, enc: Encoding, p: f64) -> f64 {
    let sched = Schedule::geometric(0.05, 8.0, 80, 30);
    (1..=8u64).map(|s| Categorical::new(n, k, enc, p).anneal_feasibility(&sched, s)).sum::<f64>() / 8.0
}

fn main() {
    println!("40 independent k-valued variables, no objective. Feasible fraction after annealing,");
    println!("mean over 8 seeds. Penalty strength 2.0.\n");
    println!("{:>4} {:>8} {:>8} {:>12} {:>12} {:>10}", "k", "dw spins", "oh spins", "domain-wall", "one-hot", "gap");
    for k in [3usize, 4, 6, 8, 12, 16, 24, 32] {
        let dw = rate(40, k, Encoding::DomainWall, 2.0);
        let oh = rate(40, k, Encoding::OneHot, 2.0);
        println!("{k:>4} {:>8} {:>8} {:>12.4} {:>12.4} {:>+10.4}",
                 Encoding::DomainWall.spins(k) * 40, Encoding::OneHot.spins(k) * 40, dw, oh, dw - oh);
    }
    println!("\nThe gap is not in feasibility at an adequate penalty -- both reach 1.0. It is in HOW");
    println!("WEAK a penalty each tolerates, which matters because a large penalty distorts any");
    println!("objective sitting beside it.\n");
    println!("{:>8} {:>12} {:>12}", "penalty", "domain-wall", "one-hot");
    for p in [0.05f64, 0.1, 0.15, 0.2, 0.25, 0.4, 0.6, 1.0] {
        println!("{p:>8.2} {:>12.4} {:>12.4}", rate(40, 12, Encoding::DomainWall, p),
                 rate(40, 12, Encoding::OneHot, p));
    }
    println!("\nsmallest penalty reaching 0.99 feasible, by k:");
    println!("{:>4} {:>14} {:>14}", "k", "domain-wall", "one-hot");
    for k in [4usize, 8, 16, 32] {
        let find = |e: Encoding| {
            let mut lo = 0.02f64;
            while lo < 4.0 { if rate(40, k, e, lo) >= 0.99 { return lo } lo *= 1.3 }
            f64::NAN
        };
        println!("{k:>4} {:>14.3} {:>14.3}", find(Encoding::DomainWall), find(Encoding::OneHot));
    }
}
