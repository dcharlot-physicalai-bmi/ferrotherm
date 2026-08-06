//! What does MPPI actually do on a system whose optimum we know? Measure before asserting.
use ferrotherm::mppi::{Lqr, Mppi, System};

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
}
