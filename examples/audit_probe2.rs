fn main() {
    unsafe { probe() }
}
use ferrotherm::ffi::*;

unsafe fn probe() {
    // (F) a_planted_instance_carries_its_optimum: assertion is only `e >= known - 1e-9`.
    let sim = ft_planted_frustrated(6, 40, 3, 1.0);
    let known = ft_ground_energy(sim);
    let good = ft_anneal(sim, 0.05, 6.0, 80, 40);
    ft_free(sim);
    // deliberately useless ladder: 2 stages, 1 sweep
    let sim2 = ft_planted_frustrated(6, 40, 3, 1.0);
    let bad = ft_anneal(sim2, 0.05, 0.06, 2, 1);
    ft_free(sim2);
    println!("(F) known={known}  real anneal={good}  crippled anneal={bad}  \
              assertion `e >= known-1e-9` on crippled -> {}",
             if bad >= known - 1e-9 { "STILL PASSES" } else { "fails" });

    // (G) a_wishart_instance ... "carries its optimum": only checks is_finite().
    //     Is the reported ground energy actually the optimum? Brute-force a small one.
    for n in [14u32, 16] {
        let sim = ft_planted_wishart(n, 0.5, 1, 1.0);
        let claimed = ft_ground_energy(sim);
        let exact = ft_exact_ground(sim, 30);
        println!("(G) wishart n={n}: claimed ground {claimed:.6}, exact ground {exact:.6}, \
                  delta {:.6}", claimed - exact);
        ft_free(sim);
    }

    // (H) builds_and_samples_an_arbitrary_graph: `assert!(ft_energy(sim).is_finite())`
    let b = ft_builder_new(4);
    ft_builder_couple(b, 0, 1, 1.0);
    let sim = ft_builder_build(b, 1.0, 7);
    ft_sweep(sim, 50);
    println!("(H) energy = {} ; is_finite() = {}", ft_energy(sim), ft_energy(sim).is_finite());
    ft_free(sim);
}
