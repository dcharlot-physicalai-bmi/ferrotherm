// `missing_docs` is denied workspace-wide and is right to be: it guards the API surface, and
// every public item in every library here carries a doc. An EXAMPLE has no API surface -- it is
// a program, and its helpers are private to it -- so the lint has nothing to guard and asks for
// doc comments on `fn main`'s scaffolding instead. Scoped off here rather than weakened there.
#![allow(missing_docs)]
// IS SEGMENTED PROBABILISTIC SAMPLING A BOLTZMANN SAMPLER? Exactly.
//
// Rhee, Jang, ... and K. M. Kim, "NbOx Mott Memristor-Based Oscillatory P-trit for Ternary Potts
// Machine", Advanced Science 2026, e76754 (doi:10.1002/advs.76754) introduce a hardware p-trit and,
// to drive it, the Segmented Probabilistic Sampling (SPS) rule -- described there as "an
// approximate probabilistic sampling method". A ternary spin is a unit vector at 0, 2pi/3 or
// 4pi/3; one scalar input theta_i^min per node is mapped linearly onto the device voltage; and in
// each 2pi/3 phase interval ONLY THE TWO ADJACENT STATES have non-zero probability, with a sigmoid
// between them. The paper validates SPS by SOLUTION QUALITY on Max-3-Cut and number partitioning.
// It never asks what distribution the chain samples. This does, with no sampling anywhere.
//
// WHAT IS READ FROM THE PAPER AND WHAT IS OURS. The main text fixes the three angles, the single
// scalar input, the two-adjacent-states support and the sigmoid. It does NOT settle, without the
// supplementary notes this was written without, whether the sigmoid's sharpness scales with the
// local field's MAGNITUDE. So that is a parameter here, not a guess, and the rule is taken apart
// into the approximations it stacks:
//
//   K0  heat bath            all three states, exact odds          -- the control
//   K3  pair only            third state excluded, EXACT pair odds  sigma(b sqrt3 |F| sin(x))
//   K2  pair + linearised    ... and sin(x) -> x, the most charitable magnitude-aware Eq. 4
//   K1  angle only           ... and |F| ignored: a fixed device curve, sharpness s
//   K4  angle only, ordinal  ... and no on<->off interval (their stated hardware limit), with a
//                            linear, non-wrapping voltage map that SATURATES there.
//   K5  ... nearest clamp    the same limit read as gently as a deterministic rule allows: the
//                            omitted interval goes to whichever of on/off is NEARER.
//
// K4 AND K5 ARE A BRACKET, NOT A CLAIM ABOUT THEIR MACHINE. The paper says the ordinal pathway
// "can lead to slower convergence and larger fluctuations" (its Fig. S11, unread here) and offers a
// look-up-table remap as the cure -- which is K1. Its own simulations report near-ideal
// convergence WITH the constraint, so K4 is almost certainly harsher than what they ran. The two
// rows show what an UNMITIGATED restriction costs, and how much the choice of reading matters.
//
// where x = theta - (interval midpoint) and F_i = sum_j J_ij x_j is the local field vector. The
// exact two-state odds follow from cos(u) - cos(u - 2pi/3) = -sqrt3 sin(u - pi/3).
//
// Every law below is a direct solve of pi (I - P) = 0 on all 3^n states for a random-scan
// single-site kernel, held against the enumerated Boltzmann law of the vector (clock) Potts model.
//
// run: cargo run --release --example sps_exact

use core::f64::consts::{PI, TAU};
use ferrotherm::autocorr::total_variation;
use ferrotherm::pdit::stationary_of;
use ferrotherm::potts::{enumerate, Interaction, Potts, PottsBuilder};
use ferrotherm::rng::Pcg;

#[derive(Clone, Copy)]
enum Rule {
    HeatBath,
    PairExact,
    PairLinear,
    AngleOnly(f64),
    AngleOnlyOrdinal(f64),
    AngleOnlyNearest(f64),
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// The local field vector at `i`: `sum_j J_ij (cos a_j, sin a_j)`.
fn field(adj: &[Vec<(usize, f64)>], s: &[u8], i: usize) -> (f64, f64) {
    let (mut fx, mut fy) = (0.0, 0.0);
    for &(j, w) in &adj[i] {
        let a = TAU * f64::from(s[j]) / 3.0;
        fx += w * a.cos();
        fy += w * a.sin();
    }
    (fx, fy)
}

/// The three update probabilities at one site under `rule`.
fn update(rule: Rule, beta: f64, f: (f64, f64)) -> [f64; 3] {
    let mag = (f.0 * f.0 + f.1 * f.1).sqrt();
    if let Rule::HeatBath = rule {
        let mut w = [0.0f64; 3];
        for (k, wk) in w.iter_mut().enumerate() {
            let a = TAU * k as f64 / 3.0;
            *wk = beta * (f.0 * a.cos() + f.1 * a.sin());
        }
        let top = w.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let z: f64 = w.iter().map(|x| (x - top).exp()).sum();
        return [(w[0] - top).exp() / z, (w[1] - top).exp() / z, (w[2] - top).exp() / z];
    }
    // No field, no preferred direction: every rule is uniform here, as the heat bath is.
    if mag < 1e-12 {
        return [1.0 / 3.0; 3];
    }
    let theta = f.1.atan2(f.0).rem_euclid(TAU);
    let seg = ((theta / (TAU / 3.0)).floor() as usize).min(2);
    let (lo, hi) = (seg, (seg + 1) % 3);
    let x = theta - (TAU * seg as f64 / 3.0 + PI / 3.0);
    let mut p = [0.0f64; 3];
    let up = match rule {
        Rule::PairExact => sigmoid(beta * 3.0_f64.sqrt() * mag * x.sin()),
        Rule::PairLinear => sigmoid(beta * 3.0_f64.sqrt() * mag * x),
        Rule::AngleOnly(sharp) => sigmoid(sharp * x),
        Rule::AngleOnlyOrdinal(sharp) => {
            if seg == 2 {
                // The interval between `on` (4pi/3) and `off` (0) does not exist on the device. A
                // linear voltage map that does not wrap sends all of it past the `on` threshold.
                p[2] = 1.0;
                return p;
            }
            sigmoid(sharp * x)
        }
        Rule::AngleOnlyNearest(sharp) => {
            if seg == 2 {
                // x < 0 is the half of the omitted interval nearer `on`; the rest is nearer `off`.
                p[if x < 0.0 { 2 } else { 0 }] = 1.0;
                return p;
            }
            sigmoid(sharp * x)
        }
        Rule::HeatBath => unreachable!("handled above"),
    };
    p[hi] = up;
    p[lo] = 1.0 - up;
    p
}

/// The stationary law of the random-scan single-site chain under `rule`.
fn law(m: &Potts, adj: &[Vec<(usize, f64)>], rule: Rule, beta: f64) -> Vec<f64> {
    let n = m.n();
    let states = 3usize.pow(n as u32);
    let mut kernel = vec![0.0f64; states * states];
    for x in 0..states {
        let s = m.state_at(x);
        for i in 0..n {
            let p = update(rule, beta, field(adj, &s, i));
            let mut t = s.clone();
            for (a, pa) in p.iter().enumerate() {
                t[i] = a as u8;
                let y = m.index_of(&t).expect("a state of this model");
                kernel[x * states + y] += pa / n as f64;
            }
        }
    }
    stationary_of(&kernel, states).expect("an irreducible chain")
}

/// The beta whose Boltzmann law is nearest `pi` in total variation, and that distance.
fn nearest_beta(m: &Potts, pi: &[f64]) -> (f64, f64) {
    let tv_at = |b: f64| total_variation(pi, &enumerate(m, b).expect("small").p);
    let mut best = (0.0, tv_at(0.0));
    for k in 1..=160 {
        let b = 0.05 * k as f64;
        let t = tv_at(b);
        if t < best.1 {
            best = (b, t);
        }
    }
    let (mut lo, mut hi) = ((best.0 - 0.05).max(0.0), best.0 + 0.05);
    for _ in 0..40 {
        let (a, c) = (lo + (hi - lo) / 3.0, hi - (hi - lo) / 3.0);
        if tv_at(a) < tv_at(c) {
            hi = c;
        } else {
            lo = a;
        }
    }
    let b = 0.5 * (lo + hi);
    (b, tv_at(b))
}

fn ground_mass(m: &Potts, beta_for_energies: f64, pi: &[f64]) -> f64 {
    let e = enumerate(m, beta_for_energies).expect("small").energies;
    let min = e.iter().copied().fold(f64::INFINITY, f64::min);
    e.iter().zip(pi).filter(|(x, _)| **x < min + 1e-9).map(|(_, p)| *p).sum()
}

fn build(n: usize, edges: &[(usize, usize, f64)]) -> (Potts, Vec<Vec<(usize, f64)>>) {
    let mut b = PottsBuilder::new(3, n, Interaction::Clock);
    let mut adj = vec![Vec::new(); n];
    for &(i, j, w) in edges {
        b.couple(i, j, w);
        adj[i].push((j, w));
        adj[j].push((i, w));
    }
    (b.build(), adj)
}

fn main() {
    let n = 6usize;
    let mut rng = Pcg::new(0x5B5_E7AC, 3);
    let mut glass = Vec::new();
    let mut cut = Vec::new();
    for i in 0..n {
        for j in i + 1..n {
            if rng.f64() < 0.6 {
                glass.push((i, j, 2.0 * rng.f64() - 1.0));
            }
            if rng.f64() < 0.5 {
                cut.push((i, j, -1.0)); // Max-3-Cut: every edge wants its ends to differ
            }
        }
    }
    let systems = [("random couplings, n = 6", glass), ("Max-3-Cut, 50% density, n = 6", cut)];

    println!("IS SEGMENTED PROBABILISTIC SAMPLING A BOLTZMANN SAMPLER?  (3^{n} = {} states, exact)", 3usize.pow(n as u32));
    for (name, edges) in &systems {
        let (m, adj) = build(n, edges);
        println!();
        println!("== {name}, {} edges", edges.len());
        println!("   rule                         beta    TV to Boltzmann(beta)   nearest beta   TV there   ground mass (exact)");
        for &beta in &[0.5f64, 1.0, 2.0, 4.0] {
            let exact = enumerate(&m, beta).expect("small");
            let g_exact = ground_mass(&m, beta, &exact.p);
            for (label, rule) in [
                ("K0 heat bath (control)", Rule::HeatBath),
                ("K3 pair only, exact odds", Rule::PairExact),
                ("K2 pair, linearised", Rule::PairLinear),
            ] {
                let pi = law(&m, &adj, rule, beta);
                let tv = total_variation(&pi, &exact.p);
                let (nb, ntv) = nearest_beta(&m, &pi);
                println!(
                    "   {label:<28} {beta:>4.1}    {tv:>10.3e}              {nb:>6.3}       {ntv:>9.3e}   {:.4} ({g_exact:.4})",
                    ground_mass(&m, beta, &pi)
                );
            }
        }
        println!("   -- angle only: the device curve is fixed, so there is no beta to hold it to; nearest beta is the fair test");
        for &sharp in &[2.0f64, 4.0, 8.0, 16.0] {
            for (label, rule) in [
                ("K1 angle only", Rule::AngleOnly(sharp)),
                ("K4 ordinal, saturating", Rule::AngleOnlyOrdinal(sharp)),
                ("K5 ordinal, nearest", Rule::AngleOnlyNearest(sharp)),
            ] {
                let pi = law(&m, &adj, rule, 0.0);
                let (nb, ntv) = nearest_beta(&m, &pi);
                let g_there = ground_mass(&m, nb, &enumerate(&m, nb).expect("small").p);
                println!(
                    "   {label:<22} s={sharp:<4}    -               -                  {nb:>6.3}       {ntv:>9.3e}   {:.4} ({g_there:.4})",
                    ground_mass(&m, nb, &pi)
                );
            }
        }
        // WHERE THE ORDINAL CHAIN'S ERROR LIVES. The clock model is invariant under rotating every
        // spin by 2pi/3, so ground states come in rotated triples of EQUAL Boltzmann weight. An
        // ordinal rule has no such symmetry. If it finds the ground states and still sits far from
        // Boltzmann, the distance should be in how it SHARES mass among them -- shown, not assumed.
        let e = enumerate(&m, 1.0).expect("small").energies;
        let min = e.iter().copied().fold(f64::INFINITY, f64::min);
        let grounds: Vec<usize> = (0..e.len()).filter(|&k| e[k] < min + 1e-9).collect();
        let k1 = law(&m, &adj, Rule::AngleOnly(16.0), 0.0);
        let k5 = law(&m, &adj, Rule::AngleOnlyNearest(16.0), 0.0);
        println!("   -- mass on each of the {} ground states at s = 16 (Boltzmann shares them EQUALLY: {:.4} each)", grounds.len(), 1.0 / grounds.len() as f64);
        println!("      state          K1 angle only     K5 ordinal, nearest");
        for &k in grounds.iter().take(9) {
            println!("      {:?}   {:>12.4}      {:>12.4}", m.state_at(k), k1[k], k5[k]);
        }
    }
}
