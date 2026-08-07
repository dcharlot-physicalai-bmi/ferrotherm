fn main() {
    let g = ferrotherm::ising::lattice2d(10, 1.0);
    let mut s = ferrotherm::gibbs::Sampler::new(&g, 0.05, 0);
    s.sweeps(600, None);
    let hot = g.energy(&s.s);
    for bmax in [0.2f64, 0.3, 0.5, 1.0, 4.0] {
        let ladder = ferrotherm::tempering::geometric_ladder(0.05, bmax, 30);
        let sched: Vec<(f64, usize)> = ladder.iter().map(|&b| (b, 20)).collect();
        let (_, e) = ferrotherm::tempering::anneal(&g, &sched, 0, None);
        println!("(I) beta_max={bmax:<4} annealed = {e:>8.1}  hot = {hot}  ground = -200  \
                  `annealed<hot` -> {}", if e < hot { "PASSES" } else { "fails" });
    }
}
