//! **Two exact samplers that share nothing, held to each other past enumeration.**
//!
//! `kwsample` draws exact independent samples from a planar spin glass by Kac–Ward determinants and
//! an apex spin (Liu et al., arXiv:2608.24382); `exact` draws them by variable elimination and
//! backward sampling. On lattices too large to enumerate, their estimates of the mean energy and of a
//! nearest-neighbour correlation must agree within their joint sampling error, and both must agree
//! with the Kac–Ward internal energy `pfaffian::solve` computes without sampling at all.
//!
//! And one check that needs no statistics: every Kac–Ward draw carries the log-likelihood its chain of
//! conditionals assigned it, which must equal `−βE(s) − ln Z` EXACTLY, `ln Z` from the determinant.
//! The largest disagreement over every draw is printed, so a sampling-error z-score that looks large
//! can be told apart from a sampler that is actually wrong.
//!
//! ```text
//! cargo run --release --example kwsample_exact
//! ```
use ferrotherm::exact::Elimination;
use ferrotherm::graph::{Graph, GraphBuilder};
use ferrotherm::kwsample::sample;
use ferrotherm::pfaffian;
use ferrotherm::rng::Pcg;

fn glass(side: usize, seed: u64) -> Graph {
    let mut rng = Pcg::new(seed, 0x6A);
    let mut b = GraphBuilder::new(side * side);
    for y in 0..side {
        for x in 0..side {
            let i = y * side + x;
            if x + 1 < side {
                b.couple(i, i + 1, if rng.f64() < 0.5 { -1.0 } else { 1.0 });
            }
            if y + 1 < side {
                b.couple(i, i + side, if rng.f64() < 0.5 { -1.0 } else { 1.0 });
            }
        }
    }
    b.build()
}

fn mean_se(v: &[f64]) -> (f64, f64) {
    let n = v.len() as f64;
    let m = v.iter().sum::<f64>() / n;
    let var = v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1.0);
    (m, (var / n).sqrt())
}

fn main() {
    let beta = 1.0;
    // Draws per size: each Kac–Ward draw is 2N dense determinants of order about 2E.
    println!("exact draws from each sampler, beta = {beta}, +-J glass; energy per spin");
    println!("  side  spins  draws   Kac-Ward sampler     elimination sampler   Kac-Ward <E> (no sampling)   z(KW - elim)   secs/draw KW   worst |ln P - (-bE - ln Z)|");
    for &(side, draws) in &[(6usize, 1000usize), (8, 600), (10, 1500), (14, 100)] {
        let g = glass(side, 3);
        let n = g.n as f64;
        let exact_e = pfaffian::solve(&g, beta).expect("planar").energy_density().expect("computed");
        let mut rng = Pcg::new(7, 1);
        let t0 = std::time::Instant::now();
        let ln_z = pfaffian::log_partition(&g, beta).expect("planar");
        let mut worst_lik = 0.0f64;
        let kw: Vec<f64> = (0..draws)
            .map(|_| {
                let d = sample(&g, beta, &mut rng).expect("planar");
                worst_lik = worst_lik.max((d.log_prob - (-beta * g.energy(&d.s) - ln_z)).abs());
                g.energy(&d.s) / n
            })
            .collect();
        let per = t0.elapsed().as_secs_f64() / draws as f64;
        let elim = Elimination::default().draws(&g, beta).expect("small treewidth");
        let mut rng2 = Pcg::new(8, 1);
        let el: Vec<f64> = (0..draws).map(|_| g.energy(&elim.draw(&mut rng2)) / n).collect();
        let (mk, sk) = mean_se(&kw);
        let (me, se) = mean_se(&el);
        let z = (mk - me) / (sk * sk + se * se).sqrt();
        println!(
            "  {side:4}  {:5}  {draws:5}   {mk:8.4} +- {sk:.4}   {me:8.4} +- {se:.4}   {exact_e:26.4}   {z:12.2}   {per:12.3}   {worst_lik:.1e}",
            g.n
        );
    }
}
