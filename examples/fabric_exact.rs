#![allow(missing_docs)]
// WHAT THE FABRIC'S ARITHMETIC COSTS IN DISTRIBUTION, EXACTLY: Q.8 weights, a 1,024-entry sigmoid
// ROM, a 16-bit comparator -- and an ideal random number.
//
// The KV260 p-bit fabric (`hdl::FixedFabric`, cycle-exact emulator of the RTL that was metered at
// 10.85 pJ per flip) samples with fixed-point arithmetic. Every joules-per-sample figure built on
// it prices a draw from SOME distribution; the question is which. `conform` checks the fabric
// against Boltzmann statistics within a tolerance on sampled traces; nothing has ever computed the
// distance exactly. On a model small enough to enumerate it can be:
//
//   * `autocorr::Kernel::FixedFabric` is the fabric's one-sweep kernel with its arithmetic
//     reproduced step by step and its uniforms taken as ideal, so its stationary law is what the
//     fabric samples if only the arithmetic is wrong;
//   * total variation from the Boltzmann distribution at the programmed beta is then a number,
//     not an estimate; so is the beta' whose Boltzmann distribution the fabric's law is nearest
//     to -- the "effective temperature" the certificate's beta_eff fits from samples;
//   * and the fabric's exact tau_int on its own law, against f64 chromatic Gibbs on Boltzmann,
//     says whether the emulator's mixing-time numbers (which `fabric_vs_cpu` prices) can be
//     trusted as a stand-in for the exact kernel's.
//
// The 4x3 grid used here is bipartite, as the v1 fabric requires. The RNG's own departure from
// uniform (xorshift32 per node, all on one cycle) is a separate question and is not in these
// numbers; `autocorr`'s own test bounds it on a six-spin ring.
//
// Count-based throughout; valid on a busy machine.
//
// run: cargo run --release --example fabric_exact

use ferrotherm::autocorr::{boltzmann, stationary, tau_int_exact, total_variation, Kernel};
use ferrotherm::graph::{Graph, GraphBuilder};
use ferrotherm::rng::Pcg;

fn grid_glass(w: usize, h: usize, seed: u64) -> Graph {
    let mut rng = Pcg::new(seed, 0x6A);
    let mut b = GraphBuilder::new(w * h);
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            if x + 1 < w {
                b.couple(i, i + 1, if rng.f64() < 0.5 { -1.0 } else { 1.0 });
            }
            if y + 1 < h {
                b.couple(i, i + w, if rng.f64() < 0.5 { -1.0 } else { 1.0 });
            }
        }
    }
    for i in 0..w * h {
        b.bias(i, (rng.f64() - 0.5) * 0.4);
    }
    b.build()
}

/// The beta' at which the Boltzmann distribution is nearest (in TV) to `law`: golden section on
/// `[lo, hi]`, the objective being unimodal enough here for that to find it.
fn nearest_beta(g: &Graph, law: &[f64], lo: f64, hi: f64) -> (f64, f64) {
    let f = |b: f64| total_variation(&boltzmann(g, b).expect("small"), law);
    let phi = (5f64.sqrt() - 1.0) / 2.0;
    let (mut a, mut b) = (lo, hi);
    let (mut c, mut d) = (b - phi * (b - a), a + phi * (b - a));
    let (mut fc, mut fd) = (f(c), f(d));
    for _ in 0..60 {
        if fc < fd {
            b = d;
            d = c;
            fd = fc;
            c = b - phi * (b - a);
            fc = f(c);
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + phi * (b - a);
            fd = f(d);
        }
    }
    let best = 0.5 * (a + b);
    (best, f(best))
}

fn main() {
    let (w, h) = (4usize, 3usize);
    let g = grid_glass(w, h, 7);
    let n = g.n;
    println!("THE FABRIC'S STATIONARY LAW AGAINST BOLTZMANN, EXACTLY, ON ALL 2^{n} STATES\n");
    println!("  model    {w}x{h} open grid, random +-1 couplings, fields in [-0.2, 0.2]; {} colour classes", g.classes.len());
    println!("  fabric   Q.{} weights, {}-entry ROM, 16-bit comparator, ideal uniforms (hdl::FixedFabric's arithmetic)",
             ferrotherm::hdl::FRAC, 1usize << ferrotherm::hdl::LUT_BITS);
    println!("  oracle   autocorr::stationary by pushing mass through the exact kernel; tau_int_exact on each law\n");
    println!(
        "  {:>5}   {:>10} {:>9} {:>10}   {:>10} {:>10} {:>7}",
        "beta", "TV(fab,B)", "beta_eff", "TV at eff", "tau fabric", "tau f64", "ratio"
    );
    let mut worst_tv = 0.0f64;
    let mut worst_ratio = 1.0f64;
    for &beta in &[0.5f64, 1.0, 1.5, 2.0, 3.0] {
        let (fab, steps) = stationary(&g, beta, Kernel::FixedFabric, 1e-13, 500_000).expect("small");
        assert!(steps < 500_000, "the fabric law must converge at beta {beta}");
        let pi = boltzmann(&g, beta).expect("small");
        let tv = total_variation(&fab, &pi);
        let (b_eff, tv_eff) = nearest_beta(&g, &fab, 0.25 * beta, 2.0 * beta);
        let t_fab = tau_int_exact(&g, beta, Kernel::FixedFabric, |s| g.energy(s), 1e-12, 500_000)
            .expect("small")
            .tau_int;
        let t_f64 = tau_int_exact(&g, beta, Kernel::ChromaticGibbs, |s| g.energy(s), 1e-12, 500_000)
            .expect("small")
            .tau_int;
        println!(
            "  {beta:>5.2}   {tv:>10.3e} {b_eff:>9.4} {tv_eff:>10.3e}   {t_fab:>10.2} {t_f64:>10.2} {:>7.3}",
            t_fab / t_f64
        );
        worst_tv = worst_tv.max(tv);
        worst_ratio = worst_ratio.max((t_fab / t_f64).max(t_f64 / t_fab));
    }
    println!("\n  WHAT THE TABLE SAYS.\n");
    println!("  TV(fab,B) is the exact distance between what the fabric's arithmetic samples and the Boltzmann");
    println!("  distribution it was programmed for; beta_eff is the temperature whose Boltzmann law it is nearest");
    println!("  to, and TV at eff what remains after that shift -- the part no temperature correction removes.");
    println!("  tau fabric / tau f64 is how far the emulator's mixing time sits from the exact kernel's; a ratio");
    println!("  near 1 is what lets `fabric_vs_cpu` use the emulator's tau in the fabric's price per independent");
    println!("  sample.");
    println!("\n  Worst TV over the five temperatures: {worst_tv:.3e}; worst tau ratio either way: {worst_ratio:.3}.");
}
