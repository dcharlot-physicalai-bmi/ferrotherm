#![allow(missing_docs)]
// HOW FAR IS SOKAL'S tau_int FROM THE EXACT ONE -- the instrument behind every ESS in this crate.
//
// Every "joules per independent sample", every "flips per independent sample", every
// `Finding::Undermixed`, and both tables in `informed.rs` divide by an effective sample size that
// comes from `certify::tau_int`: Sokal's automatic windowing over an empirical autocorrelation,
// truncated at the first window W >= 5 tau. That estimator has a known bias -- it truncates a tail
// that is positive, and on a trace not much longer than tau it sees the slow mode only partly --
// and nothing here has ever measured the bias against a value that does not itself come from a
// trace.
//
// THE ORACLE. On a model small enough to enumerate, the chromatic Gibbs sweep is an explicit
// linear operator P on the 2^n states: a product over colour classes of independent single-site
// heat-bath kernels, in the order `gibbs::Sampler::sweep` applies them. The stationary
// autocovariance of the energy at lag k is then EXACTLY
//
//     C(k) = sum_x pi(x) e(x) (P^k e)(x),     e = E - <E>_pi,
//
// and tau_exact = 1/2 + sum_{k>=1} C(k)/C(0), summed until the tail is below floating point. No
// sampling, no windowing, no trace: linear algebra on pi and P. Then the SAME kernel is run as a
// chain, its energy trace handed to `certify::tau_int`, and the ratio tau_sokal / tau_exact
// reported as a function of the trace length in units of tau. The question with an answer:
//
//     at how many tau of trace does Sokal's estimate land within 10% of the truth, and on which
//     side of it does it sit before that?
//
// If the ratio sits well below 1 at the lengths this crate's examples actually run, every ESS
// they print is too large by that factor, and every joules-per-independent-sample too small.
//
// Count-based throughout; valid on a busy machine.
//
// run: cargo run --release --example tau_exactness

use ferrotherm::certify::tau_int;
use ferrotherm::gibbs::Sampler;
use ferrotherm::graph::{Graph, GraphBuilder};
use ferrotherm::kernel::p_up;
use ferrotherm::rng::Pcg;

/// A w x h open grid with random +-1 couplings and small random fields: frustrated, bipartite,
/// and small enough that 2^n states are cheap.
fn grid_glass(w: usize, h: usize, seed: u64) -> Graph {
    let mut rng = Pcg::new(seed, 0x6A);
    let n = w * h;
    let mut b = GraphBuilder::new(n);
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
    for i in 0..n {
        b.bias(i, (rng.f64() - 0.5) * 0.4);
    }
    b.build()
}

/// State `x` as spins: bit `i` set means `s_i = +1`.
fn spins(x: usize, n: usize) -> Vec<i8> {
    (0..n).map(|i| if (x >> i) & 1 == 1 { 1 } else { -1 }).collect()
}

/// The exact one-sweep operator applied to a function over states: `v <- P v`.
///
/// `P = P_{c_last} ... P_{c_1}` in class order. `(P_c v)(x) = sum_y P_c(x -> y) v(y)` where `y`
/// differs from `x` only on the sites of class `c`, and `P_c(x -> y) = prod_{i in c} p_i(y_i | x)`
/// with `p_i` the heat-bath probability from the field at `x` -- the sites of a class are
/// pairwise non-adjacent, so their fields do not see each other and the class update factorises.
/// Applying `P v` means applying the classes in REVERSE order to `v` (operators compose right to
/// left), which is what the loop below does.
fn apply_sweep(g: &Graph, beta: f64, v: &[f64]) -> Vec<f64> {
    let n = g.n;
    let m = 1usize << n;
    let mut cur = v.to_vec();
    for class in g.classes.iter().rev() {
        let sites: Vec<usize> = class.iter().map(|&i| i as usize).collect();
        let k = sites.len();
        let mut next = vec![0.0f64; m];
        for x in 0..m {
            let s = spins(x, n);
            // p_up for each class site given x's OTHER sites (its own class does not enter).
            let p: Vec<f64> = sites.iter().map(|&i| p_up(g.field(i, &s), beta)).collect();
            // Base state with every class site cleared, then every assignment of the class.
            let mut base = x;
            for &i in &sites {
                base &= !(1usize << i);
            }
            let mut acc = 0.0;
            for a in 0..(1usize << k) {
                let mut y = base;
                let mut w = 1.0;
                for (j, &i) in sites.iter().enumerate() {
                    if (a >> j) & 1 == 1 {
                        y |= 1usize << i;
                        w *= p[j];
                    } else {
                        w *= 1.0 - p[j];
                    }
                }
                acc += w * cur[y];
            }
            next[x] = acc;
        }
        cur = next;
    }
    cur
}

/// Exact `tau_int` of the energy under the chromatic Gibbs sweep, and the exact energy variance.
fn tau_exact(g: &Graph, beta: f64) -> (f64, usize) {
    let n = g.n;
    let m = 1usize << n;
    let energy: Vec<f64> = (0..m).map(|x| g.energy(&spins(x, n))).collect();
    // pi from the crate's own energy, normalised here: the oracle is the linear algebra, not
    // another sampler.
    let emin = energy.iter().copied().fold(f64::INFINITY, f64::min);
    let mut pi: Vec<f64> = energy.iter().map(|e| (-beta * (e - emin)).exp()).collect();
    let z: f64 = pi.iter().sum();
    for p in &mut pi {
        *p /= z;
    }
    let mean: f64 = pi.iter().zip(&energy).map(|(p, e)| p * e).sum();
    let e: Vec<f64> = energy.iter().map(|x| x - mean).collect();
    let c0: f64 = pi.iter().zip(&e).map(|(p, x)| p * x * x).sum();
    let mut v = e.clone();
    let mut tau = 0.5;
    let mut k = 0usize;
    loop {
        v = apply_sweep(g, beta, &v);
        k += 1;
        let ck: f64 = pi.iter().zip(&e).zip(&v).map(|((p, x), y)| p * x * y).sum();
        let rho = ck / c0;
        tau += rho;
        if rho.abs() < 1e-13 || k >= 200_000 {
            break;
        }
    }
    (tau, k)
}

/// THE CANDIDATE REPLACEMENT, measured here against the oracle before it is allowed near
/// `certify.rs`. Sokal's window closes at the first lag `W >= 5 tau(W)`; on a chain whose
/// autocorrelation is a large fast mode plus a SMALL slow one, `tau(W)` is still small when the
/// window closes, so it closes early and the slow mode -- most of the true tau -- is never summed.
/// This keeps summing past Sokal's window for as long as the empirical autocorrelation is still
/// resolvably positive: above twice its own noise, `sqrt((1 + 2 tau(k)) / L)` (Bartlett's large-lag
/// standard error for a stationary sequence), and above a floor of 0.01. A single-mode chain is
/// left exactly where Sokal left it, because at `W = 5 tau` its `rho` is `e^-5`, under both.
fn tau_extended(trace: &[f64]) -> f64 {
    let n = trace.len();
    if n < 16 {
        return f64::NAN;
    }
    let mean = trace.iter().sum::<f64>() / n as f64;
    let var = trace.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
    if var <= 0.0 {
        return f64::INFINITY;
    }
    let max_lag = (n / 4).max(1);
    let mut tau = 0.5f64;
    let mut closed = false;
    for k in 1..=max_lag {
        let mut c = 0.0;
        for t in 0..(n - k) {
            c += (trace[t] - mean) * (trace[t + k] - mean);
        }
        let rho = c / ((n - k) as f64 * var);
        if !closed && (k as f64) >= 5.0 * tau.max(0.5) {
            closed = true;
        }
        if closed {
            let noise = ((1.0 + 2.0 * tau) / n as f64).sqrt();
            if rho < (2.0 * noise).max(0.01) {
                break;
            }
        }
        tau += rho;
    }
    tau.max(0.5)
}

/// A third opinion with no window at all: batch means. With `b` batches of length `L/b`,
/// `Var(batch mean) ~ Var(x) * 2 tau / (L/b)`, so `tau = (L/b) * Var(batch means) / (2 Var(x))`.
/// Unbiased only when the batch is much longer than the slow mode, so it reads LOW on a short
/// trace -- which is why it is a check and not the estimator.
fn tau_batch(trace: &[f64], b: usize) -> f64 {
    let n = trace.len();
    let len = n / b;
    if len < 2 {
        return f64::NAN;
    }
    let mean = trace.iter().sum::<f64>() / n as f64;
    let var = trace.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
    if var <= 0.0 {
        return f64::INFINITY;
    }
    let means: Vec<f64> =
        (0..b).map(|i| trace[i * len..(i + 1) * len].iter().sum::<f64>() / len as f64).collect();
    let vb = means.iter().map(|m| (m - mean).powi(2)).sum::<f64>() / (b as f64 - 1.0);
    (len as f64 * vb / (2.0 * var)).max(0.5)
}

struct Estimates {
    sokal: (f64, f64),
    extended: (f64, f64),
    batch: (f64, f64),
}

fn mean_sd(v: &[f64]) -> (f64, f64) {
    let m = v.iter().sum::<f64>() / v.len() as f64;
    let sd = (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64).sqrt();
    (m, sd)
}

/// All three estimators on `reps` independent traces of `len` sweeps from the same kernel, after
/// a burn-in of `burn` sweeps. Means and standard deviations over the reps.
fn tau_sampled(g: &Graph, beta: f64, len: usize, burn: usize, reps: u64) -> Estimates {
    let (mut s_v, mut e_v, mut b_v) = (Vec::new(), Vec::new(), Vec::new());
    for r in 0..reps {
        let mut s = Sampler::new(g, beta, 1000 + r);
        s.sweeps(burn, None);
        let mut trace = Vec::with_capacity(len);
        for _ in 0..len {
            s.sweep(None);
            trace.push(g.energy(&s.s));
        }
        s_v.push(tau_int(&trace));
        e_v.push(tau_extended(&trace));
        b_v.push(tau_batch(&trace, 20));
    }
    Estimates { sokal: mean_sd(&s_v), extended: mean_sd(&e_v), batch: mean_sd(&b_v) }
}

fn main() {
    let (w, h) = (4usize, 3usize);
    let g = grid_glass(w, h, 7);
    let n = g.n;
    println!("SOKAL'S tau_int AGAINST THE EXACT tau_int OF THE SAME KERNEL\n");
    println!(
        "  model    {w}x{h} open grid, {n} spins, random +-1 couplings, fields in [-0.2, 0.2]; {} colour classes",
        g.classes.len()
    );
    println!("  oracle   C(k) = <pi, e * P^k e> on all 2^{n} states, tau = 1/2 + sum rho(k) to 1e-13");
    println!("  chain    gibbs::Sampler, the same sweep, traces of L sweeps after a burn-in of 20 tau, 16 reps\n");
    println!("  estimators   S = certify::tau_int (Sokal, window at 5 tau); X = the same sum extended past the");
    println!("               window while rho stays above twice its noise; B = batch means, 20 batches.");
    println!("               Each cell is estimate / exact, mean over 16 reps, with sd/mean in brackets.\n");
    let mults = [30.0f64, 100.0, 1_000.0, 10_000.0];
    println!(
        "  {:>5} {:>10} {:>6}   {}",
        "beta",
        "tau exact",
        "lags",
        mults.iter().map(|m| format!("{:>26}", format!("L = {m:.0} tau: S / X / B"))).collect::<Vec<_>>().join("  ")
    );
    let mut worst: Vec<(f64, &str, f64)> = Vec::new(); // (mult, estimator, worst |ratio - 1|)
    for &beta in &[0.5f64, 1.0, 1.5, 2.0, 3.0] {
        let (te, lags) = tau_exact(&g, beta);
        let mut cells = Vec::new();
        for &mlt in &mults {
            let len = ((mlt * te).ceil() as usize).max(64);
            let burn = ((20.0 * te).ceil() as usize).max(16);
            let e = tau_sampled(&g, beta, len, burn, 16);
            let f = |(m, sd): (f64, f64)| format!("{:.2}({:.0}%)", m / te, 100.0 * sd / m.max(1e-300));
            cells.push(format!("{:>8} {:>8} {:>8}", f(e.sokal), f(e.extended), f(e.batch)));
            for (name, (m, _)) in [("S", e.sokal), ("X", e.extended), ("B", e.batch)] {
                worst.push((mlt, name, (m / te - 1.0).abs()));
            }
        }
        println!("  {beta:>5.2} {te:>10.2} {lags:>6}   {}", cells.join("  "));
    }
    println!("\n  WHAT THE TABLE SAYS.\n");
    println!("  A cell below 1 UNDERSTATES the true autocorrelation time at that trace length, so every ESS");
    println!("  derived from it is too large by the inverse and every cost per independent sample too small.");
    println!("  The exact column is linear algebra on the transition operator and pi: no sampling error, no");
    println!("  window. An S cell that stays flat as L grows is not a short-trace bias -- it is the window");
    println!("  closing on a fast mode and never summing a slow one, and no trace length repairs it.");
    println!("\n  Worst |estimate/exact - 1| over the five temperatures:");
    for &mlt in &mults {
        let row: Vec<String> = ["S", "X", "B"]
            .iter()
            .map(|name| {
                let w = worst
                    .iter()
                    .filter(|(m, n, _)| *m == mlt && n == name)
                    .map(|(_, _, r)| *r)
                    .fold(0.0f64, f64::max);
                format!("{name} {:>4.0}%", 100.0 * w)
            })
            .collect();
        println!("    L = {mlt:>5.0} tau:   {}", row.join("   "));
    }
    println!("\n  This crate's examples typically hold 4,000 draws; at a tau of 100 draws that is L = 40 tau.");
    println!("  Read the column nearest that for what those examples' ESS figures were worth.");
}
