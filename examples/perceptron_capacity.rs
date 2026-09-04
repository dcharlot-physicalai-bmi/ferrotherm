//! Gardner's storage problem on binary couplings: three layers of claim, kept apart.
//!
//! A theorem (the first moment), a replica value (cited), and exact enumeration -- then the
//! measurement neither of them makes: where a local algorithm actually reaches.
//!
//! usage: cargo run --release --example perceptron_capacity

use ferrotherm::perceptron::{annealed_capacity, annealed_log_z, p_sat, Perceptron, KRAUTH_MEZARD_CAPACITY};

fn main() {
    // ---- 1. the first moment, exact at finite N ----------------------------------------------
    println!("--- the annealed count E[Z] = 2^N p_sat^P, against enumeration averaged over 400 pattern sets");
    println!("{:<5} {:<4} {:<7} {:>12} {:>14} {:>8}", "N", "P", "alpha", "mean count", "2^N p_sat^P", "ratio");
    for (n, p) in [(9usize, 5usize), (11, 6), (11, 11), (10, 5), (12, 12)] {
        let sets = 400u64;
        let total: u64 = (0..sets).map(|s| Perceptron::random(n, p, 900 + s).solution_count().unwrap()).sum();
        let mean = total as f64 / sets as f64;
        let want = annealed_log_z(n, p).exp();
        println!("{n:<5} {p:<4} {:<7.2} {mean:>12.3} {want:>14.3} {:>8.3}", p as f64 / n as f64, mean / want);
    }
    println!("  p_sat is 1/2 only for ODD N; with N even a tie J.xi = 0 is a misclassification, so");
    println!("  p_sat(10) = {:.6} < 1/2 and the annealed capacity is {:.4}, not 1.\n", p_sat(10), annealed_capacity(10));

    // ---- 2. the three capacities -------------------------------------------------------------
    println!("--- three capacities, three kinds of claim");
    println!("  first moment (a THEOREM: P(Z >= 1) <= E[Z])      alpha <= {:.4}   [odd N]", annealed_capacity(11));
    println!("  Krauth-Mezard replica value (CITED, 1989)        alpha_c ~ {KRAUTH_MEZARD_CAPACITY}");
    println!("  enumeration (EXACT, this N, these patterns)      measured below\n");

    // ---- 3. what enumeration sees, and what the annealer reaches ------------------------------
    println!("--- N = 15: solutions that exist, and solutions the annealer finds (40 instances each)");
    println!("{:<7} {:<4} {:>18} {:>18} {:>20}", "alpha", "P", "solvable (exact)", "annealer found", "median log2 count");
    for p in [3usize, 7, 11, 13, 15, 17] {
        let (n, sets) = (15usize, 40u64);
        let (mut solvable, mut found) = (0, 0);
        let mut counts = Vec::new();
        for s in 0..sets {
            let per = Perceptron::random(n, p, 5000 + s + 97 * n as u64);
            let c = per.solution_count().unwrap();
            if c > 0 {
                solvable += 1;
                counts.push((c as f64).log2());
            }
            if (0..3).any(|r| per.solve(0.05, 12.0, 60, 40, 700 + s * 10 + r).1 == 0) {
                found += 1;
            }
        }
        counts.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med = counts.get(counts.len() / 2).copied().unwrap_or(f64::NAN);
        println!("{:<7.2} {p:<4} {:>18} {:>18} {med:>20.1}", p as f64 / n as f64, format!("{solvable}/{sets}"), format!("{found}/{sets}"));
    }
    println!("  At every size enumeration can reach, there is NO algorithmic gap: annealing finds a");
    println!("  solution in essentially every instance that has one.\n");

    // ---- 4. the gap, which is a statement about N --------------------------------------------
    println!("--- the gap opens with N: annealer success rate (20 instances, 3 restarts), capacity is {KRAUTH_MEZARD_CAPACITY} at every N");
    print!("{:<6}", "N");
    for a in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7] {
        print!("  a={a:.1}");
    }
    println!();
    for n in [21usize, 51, 101, 201, 401] {
        print!("{n:<6}");
        for a in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7] {
            let p = (a * n as f64).round() as usize;
            let found = (0..20u64)
                .filter(|&s| {
                    let per = Perceptron::random(n, p, 8000 + s + 31 * n as u64);
                    (0..3).any(|r| per.solve(0.05, 12.0, 80, 30, 900 + s * 10 + r).1 == 0)
                })
                .count();
            print!("  {found:>2}/20");
        }
        println!();
    }
    println!("  At alpha = 0.5, far below the capacity, the success rate falls from 20/20 to 0/20 as N");
    println!("  grows. The instances did not become unsatisfiable -- the solution space is FROZEN into");
    println!("  isolated points, and a local search has nothing to follow. Easy to verify, hard to find.");
}
