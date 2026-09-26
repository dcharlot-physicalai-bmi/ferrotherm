//! **Does a late read wash a p-bit network's correlations out?**
//!
//! Zhang, Gibeault et al. (arXiv:2607.15215) report, from coupled superparamagnetic tunnel
//! junctions, that *"sufficiently long delays drive the steady-state probabilities toward equal
//! state occupations even in strongly coupled systems"*. Their spins flip at a rate set by their OWN
//! current state and their neighbours' delayed states. A heat-bath p-bit ignores its own state. This
//! computes both, exactly, on the chain over the last `d` frames: the probability that two coupled
//! spins agree (`1/2` is uniform), and the distance of the current frame from the Boltzmann law.
//!
//! ```text
//! cargo run --release --example delay_exact
//! ```
use ferrotherm::autocorr::{boltzmann, total_variation};
use ferrotherm::delay::{aligned, stationary_current, Rule};
use ferrotherm::graph::GraphBuilder;

fn main() {
    let beta = 1.0;
    println!("two spins, J = 1, beta = 1; the probability they agree (1/2 is uniform), and TV from Boltzmann");
    println!("  delay   heat-bath agree   TV     | Arrhenius p0=0.05  TV     | p0=0.2  TV     | p0=0.2, h=0.5 agree");
    let pair = |h: f64| {
        let mut b = GraphBuilder::new(2);
        b.couple(0, 1, 1.0);
        b.bias(0, h);
        b.bias(1, h);
        b.build()
    };
    let g = pair(0.0);
    let gh = pair(0.5);
    let bolt = boltzmann(&g, beta).expect("small");
    for d in 1..=6 {
        let (hb, _) = stationary_current(&g, beta, Rule::HeatBath, d, 1e-15, 50_000);
        let (a05, _) = stationary_current(&g, beta, Rule::Arrhenius { p0: 0.05 }, d, 1e-15, 2_000_000);
        let (a20, _) = stationary_current(&g, beta, Rule::Arrhenius { p0: 0.2 }, d, 1e-15, 500_000);
        let (ah, _) = stationary_current(&gh, beta, Rule::Arrhenius { p0: 0.2 }, d, 1e-15, 500_000);
        println!(
            "  {d:5}   {:15.4}   {:.4} | {:17.4}  {:.4} | {:6.4}  {:.4} | {:18.4}",
            aligned(&g, &hb),
            total_variation(&hb, &bolt),
            aligned(&g, &a05),
            total_variation(&a05, &bolt),
            aligned(&g, &a20),
            total_variation(&a20, &bolt),
            aligned(&gh, &ah)
        );
    }
    println!("\nBoltzmann: agree {:.4}", aligned(&g, &bolt));
}
