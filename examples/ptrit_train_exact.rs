//! Training through a p-trit device, exactly: does the Potts learning rule, run with the
//! fixed-point p-trit fabric's own stationary law as its negative phase, absorb the fabric's
//! arithmetic -- or does it end, as it did for p-bits, chattering on a staircase?
//!
//! ```text
//!   cargo run --release --example ptrit_train_exact
//! ```
//!
//! Everything is a population quantity: the data law is the exact Boltzmann law of a random
//! three-state model, the device's law is the exact stationary law of its sweep kernel
//! ([`ferrotherm::pdit::FixedPdit::sweep_kernel`]), and the moments on both sides are sums over
//! all `3^5 = 243` states. No sampling anywhere, so a difference is the arithmetic's.

use ferrotherm::autocorr::total_variation;
use ferrotherm::pdit::{stationary_of, FixedPdit};
use ferrotherm::potts::{enumerate, Interaction, Potts, PottsBuilder};

const Q: usize = 3;
const N: usize = 5;

/// A parameter set: ring couplings and per-site, per-state fields.
#[derive(Clone, Debug)]
struct Params {
    j: Vec<f64>,
    h: Vec<f64>,
}

impl Params {
    fn model(&self) -> Potts {
        let mut b = PottsBuilder::new(Q, N, Interaction::Potts);
        for i in 0..N {
            b.couple(i, (i + 1) % N, self.j[i]);
        }
        for i in 0..N {
            for a in 0..Q {
                b.field(i, a as u8, self.h[i * Q + a]);
            }
        }
        b.build()
    }
}

fn state_of(mut index: usize) -> [u8; N] {
    let mut s = [0u8; N];
    for v in &mut s {
        *v = (index % Q) as u8;
        index /= Q;
    }
    s
}

/// `<delta(s_i, s_{i+1})>` per ring edge and `<delta(s_i, a)>` per site and state under a law.
fn moments(law: &[f64]) -> Params {
    let mut j = vec![0.0; N];
    let mut h = vec![0.0; N * Q];
    for (x, &p) in law.iter().enumerate() {
        let s = state_of(x);
        for i in 0..N {
            if s[i] == s[(i + 1) % N] {
                j[i] += p;
            }
            h[i * Q + usize::from(s[i])] += p;
        }
    }
    Params { j, h }
}

fn kl(from: &[f64], to: &[f64]) -> f64 {
    from.iter()
        .zip(to)
        .filter(|(a, _)| **a > 0.0)
        .map(|(a, b)| a * (a / b).ln())
        .sum()
}

fn device_law(p: &Params, rom_bits: u32) -> Option<Vec<f64>> {
    let m = p.model();
    let fab = FixedPdit::new(&m, 1.0, 1, rom_bits).ok()?;
    let kernel = fab.sweep_kernel().ok()?;
    stationary_of(&kernel, law_size()).ok()
}

fn law_size() -> usize {
    Q.pow(N as u32)
}

fn boltzmann_law(p: &Params) -> Vec<f64> {
    enumerate(&p.model(), 1.0).expect("243 states").p
}

/// The learning rule with a chosen negative phase, `steps` steps of a decaying rate.
fn train(
    target: &Params,
    start: &Params,
    steps: usize,
    negative: impl Fn(&Params) -> Option<Vec<f64>>,
) -> (Params, usize) {
    let data = moments(&boltzmann_law(target));
    let mut p = start.clone();
    let mut failed = 0;
    for t in 0..steps {
        let eta = 0.6 / (1.0 + t as f64 / 60.0);
        let Some(law) = negative(&p) else {
            failed += 1;
            continue;
        };
        let dev = moments(&law);
        for i in 0..N {
            p.j[i] += eta * (data.j[i] - dev.j[i]);
        }
        for k in 0..N * Q {
            p.h[k] += eta * (data.h[k] - dev.h[k]);
        }
    }
    (p, failed)
}

fn report(label: &str, truth: &[f64], law: Option<Vec<f64>>) {
    match law {
        Some(l) => println!(
            "  {label:<44} TV {:.4}   KL(device || data) {:.2e}",
            total_variation(&l, truth),
            kl(&l, truth)
        ),
        None => println!("  {label:<44} no unique law (frozen)"),
    }
}

fn main() {
    let mut seed = 0x9E37_79B9_7F4A_7C15u64;
    let mut unit = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 11) as f64 / (1u64 << 53) as f64
    };
    for scale in [0.6f64, 1.2] {
        let target = Params {
            j: (0..N).map(|_| scale * (2.0 * unit() - 1.0)).collect(),
            h: (0..N * Q)
                .map(|_| 0.5 * scale * (2.0 * unit() - 1.0))
                .collect(),
        };
        let truth = boltzmann_law(&target);
        let start = Params {
            j: vec![0.0; N],
            h: vec![0.0; N * Q],
        };
        println!("== three-state ring, n = {N}, couplings and fields at scale {scale} ==");
        let (fit, _) = train(&target, &start, 300, |p| Some(boltzmann_law(p)));
        report(
            "Boltzmann rule (control, exact negative phase)",
            &truth,
            Some(boltzmann_law(&fit)),
        );
        for rom_bits in [8u32, 10, 12, 14] {
            println!("  -- Gumbel ROM of {rom_bits} bits --");
            report(
                "load the true parameters",
                &truth,
                device_law(&target, rom_bits),
            );
            let (fit, failed) = train(&target, &start, 300, |p| device_law(p, rom_bits));
            report(
                &format!("rule with the device as negative phase ({failed} frozen steps)"),
                &truth,
                device_law(&fit, rom_bits),
            );
        }
    }
    println!();
    println!(
        "TV and KL are of the device's exact stationary law against the data's Boltzmann law."
    );
}
