//! What do real chains actually look like? Measure before asserting.
use ferrotherm::gibbs::Sampler;
use ferrotherm::graph::Graph;
use ferrotherm::samples::{Plan, SampleSet};

fn run(g: &Graph, beta: f64, thin: usize, burn: usize, draws: usize, seed: u64) -> SampleSet {
    Sampler::new(g, beta, seed).collect(&Plan::new(burn, draws, thin), None)
}

fn main() {
    println!("{:<34} {:>8} {:>8} {:>7} {:>6}  findings", "case", "beta_eff", "tau", "ess", "n");
    let cases: Vec<(&str, Graph, f64, usize, usize, usize)> = vec![
        ("ring12 b2.5 burn0 thin1",   ferrotherm::ising::ring(12, 1.0, 0.0), 2.5, 1, 0, 400),
        ("lat6  b1.2 burn200 thin1",  ferrotherm::ising::lattice2d(6, 1.0),  1.2, 1, 200, 2000),
        ("lat6  b1.2 burn200 thin40", ferrotherm::ising::lattice2d(6, 1.0),  1.2, 40, 200, 2000),
        ("lat16 b0.8 burn0 thin1",    ferrotherm::ising::lattice2d(16, 1.0), 0.8, 1, 0, 400),
        ("lat24 b0.7 burn0 thin1",    ferrotherm::ising::lattice2d(24, 1.0), 0.7, 1, 0, 600),
        ("lat24 b0.7 burn500 thin1",  ferrotherm::ising::lattice2d(24, 1.0), 0.7, 1, 500, 600),
        ("lat24 b0.7 burn500 thin50", ferrotherm::ising::lattice2d(24, 1.0), 0.7, 50, 500, 600),
        ("lat12 b0.44 burn500 thin20",ferrotherm::ising::lattice2d(12, 1.0), 0.44, 20, 500, 3000),
    ];
    for (name, g, beta, thin, burn, draws) in cases {
        let c = run(&g, beta, thin, burn, draws, 1).certificate(&g).expect("a chain");
        let f: Vec<String> = c.findings.iter().map(|x| format!("{x:?}").split(' ').next().unwrap().to_string()).collect();
        println!("{name:<34} {:>8.3} {:>8.1} {:>7.0} {:>6}  {}",
                 c.beta_eff, c.tau_int, c.ess, g.n,
                 if f.is_empty() { "PASSED".to_string() } else { f.join(",") });
    }
}
