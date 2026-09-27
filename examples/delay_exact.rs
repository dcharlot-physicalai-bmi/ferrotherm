//! **Does a late read wash a p-bit network's correlations out?**
//!
//! Zhang, Gibeault et al. (arXiv:2607.15215) report, from coupled superparamagnetic tunnel
//! junctions, that *"sufficiently long delays drive the steady-state probabilities toward equal
//! state occupations even in strongly coupled systems"*. Their spins flip at a rate set by their OWN
//! current state and their neighbours' delayed states. A heat-bath p-bit ignores its own state. This
//! computes both, exactly, on the chain over the last `d` frames: the probability that two coupled
//! spins agree (`1/2` is uniform), and the distance of the current frame from the Boltzmann law.
//!
//! The third rule is the one a p-bit fabric might actually build: the stochastic cellular automaton
//! (`Rule::Sca`, STATICA and Amorphica's all-at-once update), whose pinning `q` pulls each spin toward
//! its own CURRENT value. That own-state dependence lets the delay in, so the second half of this
//! prints what a `d`-tick read does to the pinned automaton -- the first-order law
//! `TV e^{2q} -> c_d` against the exact chain on four graphs -- and what it costs: to hold the law
//! within `eps`, pinning harder at `q*(eps, d)` against WAITING, running the one-tick automaton with
//! fresh reads on every `d`-th tick. Cost is spins moved per tick, exactly, as PAI-310's
//! `every-spin-at-once` counts it.
//!
//! The third table is the other repair for a shared clock, COLOURING: class `t mod K` redraws on tick
//! `t` from the frame `d` ticks back and every other spin holds (`delay::stationary_coloured`). Read
//! fresh it is exact Gibbs; holding is own-state dependence, so a late read reaches it too.
//!
//! ```text
//! cargo run --release --example delay_exact
//! ```
use ferrotherm::autocorr::{boltzmann, total_variation};
use ferrotherm::delay::{
    aligned, sca_rate_constant, stationary_coloured, stationary_current, stationary_solved, Rule, MAX_BITS,
};
use ferrotherm::graph::{Graph, GraphBuilder};

fn pair(h: f64) -> Graph {
    let mut b = GraphBuilder::new(2);
    b.couple(0, 1, 1.0);
    b.bias(0, h);
    b.bias(1, h);
    b.build()
}

/// PAI-310's frustrated triangle.
fn frustrated() -> Graph {
    let mut b = GraphBuilder::new(3);
    b.couple(0, 1, 1.0);
    b.couple(1, 2, 1.0);
    b.couple(0, 2, -1.0);
    b.bias(0, 0.2);
    b.bias(1, -0.1);
    b.bias(2, 0.05);
    b.build()
}

/// Four spins, every pair coupled, `J ~ N(0, 1/4)` and `h ~ N(0, 0.01)` from numpy's `default_rng(7)`,
/// rounded to three decimals: the literals are the instance (the same one `delay`'s tests use).
fn sk4() -> Graph {
    let mut b = GraphBuilder::new(4);
    for &(i, j, w) in &[(0, 1, 0.001), (0, 2, 0.149), (0, 3, -0.137), (1, 2, -0.445), (1, 3, -0.227), (2, 3, -0.496)] {
        b.couple(i, j, w);
    }
    for (i, h) in [0.006, 0.134, -0.049, -0.062].into_iter().enumerate() {
        b.bias(i, h);
    }
    b.build()
}

fn distance(g: &Graph, q: f64, d: usize) -> f64 {
    total_variation(&stationary_solved(g, 1.0, Rule::Sca { q }, d).law, &boltzmann(g, 1.0).expect("small"))
}

/// The pinning at which the delayed automaton comes within `eps` of Boltzmann: the Illinois variant of
/// regula falsi on `ln TV − ln eps`, which is nearly linear in `q` (slope −2), in a bracket of half a
/// unit either side of the first-order `ln(c_d / eps) / 2`.
fn pinning_for(g: &Graph, d: usize, eps: f64) -> f64 {
    let f = |q: f64| distance(g, q, d).ln() - eps.ln();
    let guess = 0.5 * (sca_rate_constant(g, 1.0, d) / eps).ln();
    let (mut a, mut b) = (guess - 0.5, guess + 0.5);
    let (mut fa, mut fb) = (f(a), f(b));
    assert!(fa > 0.0 && fb <= 0.0, "q* is not within 0.5 of the first-order {guess}");
    for _ in 0..100 {
        let c = b - fb * (b - a) / (fb - fa);
        let fc = f(c);
        if fc * fb < 0.0 {
            (a, fa) = (b, fb);
        } else {
            fa *= 0.5;
        }
        (b, fb) = (c, fc);
        if (b - a).abs() < 1e-12 || fc == 0.0 {
            break;
        }
    }
    b
}

fn main() {
    let beta = 1.0;
    println!("two spins, J = 1, beta = 1; the probability they agree (1/2 is uniform), and TV from Boltzmann");
    println!("  delay   heat-bath agree   TV     | Arrhenius p0=0.05  TV     | p0=0.2  TV     | p0=0.2, h=0.5 agree | SCA q=2 TV");
    let g = pair(0.0);
    let gh = pair(0.5);
    let bolt = boltzmann(&g, beta).expect("small");
    for d in 1..=6 {
        let (hb, _) = stationary_current(&g, beta, Rule::HeatBath, d, 1e-15, 50_000);
        let (a05, _) = stationary_current(&g, beta, Rule::Arrhenius { p0: 0.05 }, d, 1e-15, 2_000_000);
        let (a20, _) = stationary_current(&g, beta, Rule::Arrhenius { p0: 0.2 }, d, 1e-15, 500_000);
        let (ah, _) = stationary_current(&gh, beta, Rule::Arrhenius { p0: 0.2 }, d, 1e-15, 500_000);
        let sca = stationary_solved(&g, beta, Rule::Sca { q: 2.0 }, d);
        println!(
            "  {d:5}   {:15.4}   {:.4} | {:17.4}  {:.4} | {:6.4}  {:.4} | {:19.4} | {:10.4}",
            aligned(&g, &hb),
            total_variation(&hb, &bolt),
            aligned(&g, &a05),
            total_variation(&a05, &bolt),
            aligned(&g, &a20),
            total_variation(&a20, &bolt),
            aligned(&gh, &ah),
            total_variation(&sca.law, &bolt)
        );
    }
    println!("\nBoltzmann: agree {:.4}", aligned(&g, &bolt));

    let fixtures = [("pair h=0", pair(0.0)), ("pair h=0.3", pair(0.3)), ("triangle", frustrated()), ("SK4", sk4())];

    println!("\na coloured fabric, one class per spin: class t mod K redraws from the frame d ticks back, the rest");
    println!("hold. TV from Boltzmann of the law at a random tick, and after each class moves; the every-tick");
    println!("heat bath beside it is the same at every d (Peretto's law)");
    println!("  fixture       d   agree    TV      after class 0, 1, ...          moves/tick   every-tick TV");
    for (name, g) in &fixtures {
        let bolt = boltzmann(g, beta).expect("small");
        let classes: Vec<Vec<usize>> = (0..g.n).map(|i| vec![i]).collect();
        for d in (1..).take_while(|&d| g.n << (g.n * d) <= 1 << MAX_BITS) {
            let s = stationary_coloured(g, beta, &classes, d);
            let phases: Vec<String> = s.after.iter().map(|l| format!("{:.4}", total_variation(l, &bolt))).collect();
            let every = total_variation(&stationary_solved(g, beta, Rule::HeatBath, d).law, &bolt);
            let agree = if g.n == 2 { format!("{:.4}", aligned(g, &s.law)) } else { "  -   ".to_string() };
            println!(
                "  {name:11} {d:3}   {agree}   {:.4}   {:30}   {:10.4}   {every:.4}",
                total_variation(&s.law, &bolt),
                phases.join(", "),
                s.moves
            );
        }
    }
    println!("\nthe pinned automaton read late: TV e^2q -> c_d as q grows. Exact chain at q = 6 against the");
    println!("first-order constant from 2^n states (sca_rate_constant), both divided by c_1; pinning beats");
    println!("waiting, to first order, where c_d/c_1 < d");
    println!("  fixture       d   exact e^2q TV/c_1 (q=6)   c_d/c_1   exact/first-order - 1");
    for (name, g) in &fixtures {
        let c1 = sca_rate_constant(g, beta, 1);
        for d in 1..=MAX_BITS / g.n {
            let c = sca_rate_constant(g, beta, d);
            let exact = distance(g, 6.0, d) * 12f64.exp();
            println!("  {name:11} {d:3}   {:23.6}   {:7.4}   {:+.1e}", exact / c1, c / c1, exact / c - 1.0);
        }
    }

    // The pin-or-wait search solves the chain a dozen times per cell, and at n d = 12 one solve is
    // 0.3 s (the pair), 0.6 s (the triangle) and 1.2 s (SK4) on Apple silicon: the three cells there
    // would take this example from seconds to most of a minute. They were run once, to the same
    // conclusion (WORKLOADS entry 13), and are left out of the per-push run.
    const PIN_BITS: usize = 10;
    println!("\npin or wait: hold TV <= eps with reads d ticks old. PIN runs the delayed automaton every tick at");
    println!("q*(eps, d); WAIT runs the one-tick automaton on every d-th tick at q*(eps, 1). Spins moved per tick,");
    println!("exact; PIN/WAIT > 1 means pinning moves more and wins. First order predicts d c_1 / c_d. n d <= {PIN_BITS}.");
    println!("  fixture       eps     d   q*(eps,d)   PIN moves/tick   WAIT moves/tick   PIN/WAIT   first order");
    for eps in [1e-2, 1e-3] {
        for (name, g) in &fixtures {
            let c1 = sca_rate_constant(g, beta, 1);
            let q1 = pinning_for(g, 1, eps);
            let m1 = stationary_solved(g, beta, Rule::Sca { q: q1 }, 1).moves;
            for d in 1..=PIN_BITS / g.n {
                let qd = if d == 1 { q1 } else { pinning_for(g, d, eps) };
                let pin = stationary_solved(g, beta, Rule::Sca { q: qd }, d).moves;
                let wait = m1 / d as f64;
                let first = d as f64 * c1 / sca_rate_constant(g, beta, d);
                println!(
                    "  {name:11} {eps:6.0e} {d:4}   {qd:9.4}   {pin:14.6}   {wait:15.6}   {:8.4}   {first:11.4}",
                    pin / wait
                );
            }
        }
    }
}
