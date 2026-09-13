#![allow(missing_docs)]
// TRAINING THROUGH THE DEVICE'S ARITHMETIC: what the precision costs in likelihood, exactly.
//
// A global effective temperature costs a Boltzmann machine nothing: if the device samples at
// beta_dev, the couplings the learning rule converges to are the true ones divided by beta_dev,
// and the device's distribution is then the true one -- a reparametrisation, latents included.
// That is algebra, and this example does not spend compute on it. What is NOT a reparametrisation
// is the fabric's arithmetic: Q.8 couplings, a 1,024-entry sigmoid ROM, a 16-bit comparator. The
// device samples a law that is not in the Boltzmann family at all (`fabric_exact` measures how far),
// so the questions below have answers only a device can give, and on a small model the device's
// law is exact (`autocorr::Kernel::Quantised`). Everything is a population quantity over the
// target's exact distribution on an eight-spin ring: no data noise, no sampling.
//
//   loaded     Run the TRUE couplings on the device: the train-in-software, run-on-hardware path.
//              KL from the target is what the arithmetic costs when nothing is done about it.
//   rule       Train THROUGH the device the way every on-device learning loop does: the
//              Boltzmann learning rule, positive phase from the data, negative phase from the
//              device's own (exact) stationary law, step size decaying. Its fixed point is where
//              the DEVICE's moments match the data's -- which is the maximum of the likelihood
//              only inside the Boltzmann family, and the device is outside it.
//   lattice    The best loadable couplings the device can reach at all: a coordinate search on
//              the device's own parameter lattice (one quantum per move, accept only improvements,
//              until no single-quantum move helps). A local optimum of the device's true
//              likelihood, and the floor the rule is measured against.
//
// A first version of this example ascended a central-difference gradient of the device's exact
// likelihood and recovered exactly 0.0% at every precision. That was not a result about the
// device; it was the derivative of a staircase. The device's law is PIECEWISE CONSTANT in the
// loaded parameters -- every coupling is rounded to a quantum before it touches the field -- so
// its gradient with respect to the parameters is zero almost everywhere, and any scheme that
// differentiates through a quantised device's law is differentiating a step function. The two
// paths kept here move by moments and by quanta, neither of which needs that derivative. For the
// same reason the learning rule's moment gap can never close: the device's moments are constant
// on each cell of its lattice, so the rule ends chattering between cells, and its decaying step
// picks one. Which one depends on the schedule, which is why the second table runs several.
//
// run: cargo run --release --example device_train_exact

use ferrotherm::autocorr::{boltzmann, spins, stationary, Kernel};
use ferrotherm::graph::{Graph, GraphBuilder};
use ferrotherm::rng::Pcg;

const N: usize = 8;

/// Ring edges plus two chords, with fields: ten parameters on edges, eight on sites.
fn edges() -> Vec<(usize, usize)> {
    let mut e: Vec<(usize, usize)> = (0..N).map(|i| (i, (i + 1) % N)).collect();
    e.push((0, 4));
    e.push((2, 6));
    e
}

fn build(theta: &[f64]) -> Graph {
    let e = edges();
    let mut b = GraphBuilder::new(N);
    for (k, &(i, j)) in e.iter().enumerate() {
        b.couple(i, j, theta[k]);
    }
    for i in 0..N {
        b.bias(i, theta[e.len() + i]);
    }
    b.build()
}

/// KL(target || law).
fn kl(target: &[f64], law: &[f64]) -> f64 {
    target
        .iter()
        .zip(law)
        .filter(|(t, _)| **t > 0.0)
        .map(|(t, l)| t * (t / l.max(1e-300)).ln())
        .sum()
}

/// The device's law for these parameters, exactly.
fn device_law(theta: &[f64], kernel: Kernel) -> Vec<f64> {
    let g = build(theta);
    match kernel {
        Kernel::ChromaticGibbs => boltzmann(&g, 1.0).expect("small"),
        k => stationary(&g, 1.0, k, 1e-13, 200_000).expect("small").0,
    }
}

/// Edge and site moments of a distribution over the 2^N states, in the parameter layout.
fn moments(law: &[f64]) -> Vec<f64> {
    let e = edges();
    let mut m = vec![0.0f64; e.len() + N];
    for (x, &p) in law.iter().enumerate() {
        if p == 0.0 {
            continue;
        }
        let s = spins(x, N);
        for (k, &(i, j)) in e.iter().enumerate() {
            m[k] += p * f64::from(s[i] * s[j]);
        }
        for i in 0..N {
            m[e.len() + i] += p * f64::from(s[i]);
        }
    }
    m
}

/// The Boltzmann learning rule with the negative phase from the device's exact law and the step
/// `lr0 / (1 + t / decay)`. Returns the KL at the final parameters and the largest moment gap
/// there -- the mismatch the device's arithmetic leaves at that cell of its lattice.
fn moment_rule(target: &[f64], kernel: Kernel, start: &[f64], steps: usize, lr0: f64, decay: f64) -> (f64, f64) {
    let m_target = moments(target);
    let mut theta = start.to_vec();
    for t in 0..steps {
        let lr = lr0 / (1.0 + t as f64 / decay);
        let m_dev = moments(&device_law(&theta, kernel));
        for ((th, mt), md) in theta.iter_mut().zip(&m_target).zip(&m_dev) {
            *th += lr * (mt - md);
        }
    }
    let law = device_law(&theta, kernel);
    let gap = m_target.iter().zip(&moments(&law)).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
    (kl(target, &law), gap)
}

/// Coordinate search on the device's parameter lattice: one quantum per move, accept each
/// improving move, stop when a pass improves nothing. Returns the KL and the accepted moves.
fn lattice_search(target: &[f64], kernel: Kernel, start: &[f64], quantum: f64, max_passes: usize) -> (f64, usize) {
    let mut theta: Vec<f64> = start.iter().map(|t| (t / quantum).round() * quantum).collect();
    let mut best = kl(target, &device_law(&theta, kernel));
    let mut moves = 0usize;
    for _ in 0..max_passes {
        let mut improved = false;
        for p in 0..theta.len() {
            for dir in [1.0f64, -1.0] {
                let mut trial = theta.clone();
                trial[p] += dir * quantum;
                let k = kl(target, &device_law(&trial, kernel));
                if k < best - 1e-15 {
                    best = k;
                    theta = trial;
                    moves += 1;
                    improved = true;
                }
            }
        }
        if !improved {
            break;
        }
    }
    (best, moves)
}

/// A target: +-1 couplings on the ring-with-chords, fields in [-0.3, 0.3], from a seed.
fn target_from(seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut rng = Pcg::new(seed, 0xD0);
    let e = edges();
    let mut theta: Vec<f64> = Vec::with_capacity(e.len() + N);
    for _ in 0..e.len() {
        theta.push(if rng.f64() < 0.5 { -1.0 } else { 1.0 });
    }
    for _ in 0..N {
        theta.push((rng.f64() - 0.5) * 0.6);
    }
    let target = boltzmann(&build(&theta), 1.0).expect("small");
    (theta, target)
}

fn main() {
    let e = edges();
    let kernels = [
        ("exact (Boltzmann)", Kernel::ChromaticGibbs, None),
        ("{12, 12, 16}", Kernel::Quantised { frac_bits: 12, lut_bits: 12, prob_bits: 16 }, Some(12u32)),
        ("{8, 10, 16} shipped", Kernel::FixedFabric, Some(8)),
        ("{6, 10, 16}", Kernel::Quantised { frac_bits: 6, lut_bits: 10, prob_bits: 16 }, Some(6)),
        ("{4, 8, 16}", Kernel::Quantised { frac_bits: 4, lut_bits: 8, prob_bits: 16 }, Some(4)),
        ("{4, 6, 8}", Kernel::Quantised { frac_bits: 4, lut_bits: 6, prob_bits: 8 }, Some(4)),
    ];

    // ONE TARGET, IN FULL.
    let (true_theta, target) = target_from(11);
    let entropy: f64 = -target.iter().filter(|p| **p > 0.0).map(|p| p * p.ln()).sum::<f64>();
    println!("TRAINING THROUGH THE DEVICE'S ARITHMETIC, EXACTLY, ON AN {N}-SPIN RING\n");
    println!("  target     +-1 couplings on a ring with two chords, fields in [-0.3, 0.3], beta 1; entropy {entropy:.4} nats");
    println!("  precision  Quantised {{frac, rom, prob}} kernels; the shipped fabric is {{8, 10, 16}}; ChromaticGibbs is the exact ceiling");
    println!("  rule       Boltzmann learning rule, negative phase = device's exact law, 600 steps, lr 0.5/(1+t/100)");
    println!("  lattice    coordinate search at one quantum (2^-frac) per move from the loaded couplings, {} parameters\n", e.len() + N);
    println!(
        "  {:<18}   {:>11} {:>11} {:>9} {:>9}   {:>11} {:>6} {:>9}",
        "device", "KL loaded", "KL rule", "mom. gap", "recov.", "KL lattice", "moves", "recov."
    );
    for (name, k, frac) in kernels {
        let kl_loaded = kl(&target, &device_law(&true_theta, k));
        let (kl_rule, gap) = moment_rule(&target, k, &true_theta, 600, 0.5, 100.0);
        let recov = |k_after: f64| if kl_loaded > 0.0 { 100.0 * (kl_loaded - k_after) / kl_loaded } else { 0.0 };
        let lattice = frac.map(|f| lattice_search(&target, k, &true_theta, 1.0 / f64::from(1u32 << f), 40));
        let (lat, moves, lat_recov) = match lattice {
            Some((kl_lat, moves)) => (format!("{kl_lat:.3e}"), moves.to_string(), format!("{:.1}%", recov(kl_lat))),
            None => ("-".to_string(), "-".to_string(), "-".to_string()),
        };
        println!(
            "  {name:<18}   {kl_loaded:>11.3e} {kl_rule:>11.3e} {gap:>9.1e} {:>8.1}%   {lat:>11} {moves:>6} {lat_recov:>9}",
            recov(kl_rule)
        );
    }

    // SEVERAL TARGETS AND SCHEDULES: is the rule's verdict a property of one cell it happened to
    // land in, or of training through a quantised device?
    let seeds = [11u64, 12, 13, 14];
    let schedules = [(600usize, 0.5f64, 100.0f64), (2000, 0.1, 500.0)];
    println!("\n  ROBUSTNESS -- {} targets x {} schedules per device; 'beats loading' counts (target, schedule) pairs", seeds.len(), schedules.len());
    println!("  schedules  600 steps at 0.5/(1+t/100); 2000 steps at 0.1/(1+t/500)\n");
    println!(
        "  {:<18}   {:>13} {:>13} {:>13}   {:>13} {:>13}",
        "device", "rule beats", "rule worst", "rule best", "lattice beats", "lattice best"
    );
    println!(
        "  {:<18}   {:>13} {:>13} {:>13}   {:>13} {:>13}",
        "", "loading", "(KL ratio)", "(KL ratio)", "loading", "(KL ratio)"
    );
    for (name, k, frac) in kernels {
        let Some(f) = frac else { continue };
        let quantum = 1.0 / f64::from(1u32 << f);
        let mut rule_wins = 0usize;
        let mut rule_ratio = (f64::INFINITY, 0.0f64);
        let mut lat_wins = 0usize;
        let mut lat_ratio = f64::INFINITY;
        for &seed in &seeds {
            let (theta, target) = target_from(seed);
            let kl_loaded = kl(&target, &device_law(&theta, k));
            for &(steps, lr0, decay) in &schedules {
                let (kl_rule, _) = moment_rule(&target, k, &theta, steps, lr0, decay);
                let r = kl_rule / kl_loaded;
                if r < 1.0 {
                    rule_wins += 1;
                }
                rule_ratio = (rule_ratio.0.min(r), rule_ratio.1.max(r));
            }
            let (kl_lat, _) = lattice_search(&target, k, &theta, quantum, 40);
            let r = kl_lat / kl_loaded;
            if r < 1.0 - 1e-9 {
                lat_wins += 1;
            }
            lat_ratio = lat_ratio.min(r);
        }
        println!(
            "  {name:<18}   {:>8} of {:>2} {:>13.2} {:>13.3}   {:>8} of {:>2} {:>13.3}",
            rule_wins,
            seeds.len() * schedules.len(),
            rule_ratio.1,
            rule_ratio.0,
            lat_wins,
            seeds.len(),
            lat_ratio
        );
    }
    println!("\n  WHAT THE TABLES SAY.\n");
    println!("  'KL loaded' is the likelihood the arithmetic costs when the exact couplings are simply loaded --");
    println!("  the train-in-software, run-on-hardware path. 'KL rule' is where the Boltzmann learning rule");
    println!("  settles when its negative phase is the device; 'mom. gap' is the moment mismatch the device has");
    println!("  at that cell, which no cell can close. 'KL lattice' is the best the device can do with any");
    println!("  loadable couplings (a local optimum on its own lattice) and 'moves' how many quanta from the");
    println!("  loaded couplings it lies. Each 'recov.' is the fraction of the loaded loss that path wins back;");
    println!("  negative means the path made the device WORSE than loading the truth. The exact row is the");
    println!("  control: nothing lost, nothing to recover, no lattice. A KL ratio below 1 in the second table");
    println!("  is a path that beat loading; 'rule worst' is the largest ratio the rule produced anywhere.");
}
