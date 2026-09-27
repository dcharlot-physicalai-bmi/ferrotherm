//! **Does a spike that lives a fixed time sample faster than one that lives a random time?**
//!
//! Stewart & Sahani (arXiv:2603.09089) report that their point-process sampler *"always outperforms
//! these birth-death processes"* in multivariate effective sample size on 63 targets. The two differ in
//! one line: a spike's lifetime is fixed (`m`) or exponential with mean `m`. Same birth rates, same
//! limiting law.
//!
//! First ONE unit, where the answer is exact: the exponential lifetime has exactly twice the long-run
//! variance of the fixed one, at every field (`pointproc::unit_variance`). The window-free estimator
//! the table below uses is printed against it, and so is what Sokal's window reads off the exact
//! autocorrelation -- the estimator this example used to use, which the fixed lifetime's negative
//! lobe throws off.
//!
//! Then Sherrington–Kirkpatrick instances: the integrated autocorrelation time of the energy and the
//! magnetisation IN UNITS OF THE LIFETIME `m`, `tau = sigma^2 / (2 Var)`, where `sigma^2` is the
//! long-run variance of the time average by batch means over exact path integrals
//! (`PointProcess::batch_means`, `pointproc::long_run_variance`) and `Var` is the exact Boltzmann
//! variance by enumeration. Each lifetime runs on 16 independent seeds; the value is their mean and
//! the error their standard error, and the ratio's error is propagated from both. Two checks on the
//! estimator ride along: `b/4 : b` compares batches a quarter as long, from the same runs, so a slow
//! mode the batches were too short for shows as a ratio below one; and for the exponential lifetime at
//! `n = 8`, a 256-state Markov chain, the exact `tau` by a dense solve.
//!
//! A MEASUREMENT, not a gate, and in `examples/SLOW`: about 380 s of CPU over 16 threads.
//!
//! ```text
//! cargo run --release --example pointproc_exact
//! ```
use ferrotherm::autocorr::{boltzmann, spins};
use ferrotherm::certify::sokal_window;
use ferrotherm::graph::{Graph, GraphBuilder};
use ferrotherm::pointproc::{long_run_variance, unit_autocorrelation, unit_tau, unit_variance, Lifetime, PointProcess};
use ferrotherm::rng::Pcg;
use ferrotherm::tla::{solve_exact, Spd};

fn sk(n: usize, seed: u64) -> Graph {
    let mut rng = Pcg::new(seed, 0x6A);
    let mut normal = || {
        let a = rng.f64().max(1e-300);
        let b = rng.f64();
        (-2.0 * a.ln()).sqrt() * (std::f64::consts::TAU * b).cos()
    };
    let mut b = GraphBuilder::new(n);
    for i in 0..n {
        for j in i + 1..n {
            b.couple(i, j, normal() / (n as f64).sqrt());
        }
    }
    for i in 0..n {
        b.bias(i, 0.1 * normal());
    }
    b.build()
}

fn magnetisation(s: &[i8]) -> f64 {
    s.iter().map(|&v| f64::from(v)).sum()
}

/// Mean and standard error across independent replicas.
fn mean_se(xs: &[f64]) -> (f64, f64) {
    let k = xs.len() as f64;
    let mean = xs.iter().sum::<f64>() / k;
    let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (k - 1.0);
    (mean, (var / k).sqrt())
}

/// Pool consecutive batch means `f` at a time: the batch means of batches `f` times as long.
fn pool(means: &[f64], f: usize) -> Vec<f64> {
    means.chunks_exact(f).map(|c| c.iter().sum::<f64>() / f as f64).collect()
}

/// The EXACT integrated autocorrelation time, in units of `m`, of an observable under the
/// exponential lifetime, which is a continuous-time Markov chain on the `2^n` states: births
/// `(1/m) e^{2 beta f_i}`, deaths `1/m`, reversible with respect to the Boltzmann law `pi`. With
/// `S = D^{1/2} (-Q) D^{-1/2}` (symmetric, `S_xy = -sqrt(q_xy q_yx)`) and `v = sqrt(pi)`, its null
/// vector, `int_0^inf C(t) dt = w^T (S + v v^T)^{-1} w` for `w = D^{1/2} (f - <f>)`. One dense
/// solve; a check on the batch means that shares nothing with the simulation.
fn exact_birth_death_tau(g: &Graph, beta: f64, m: f64, f: &dyn Fn(&[i8]) -> f64) -> f64 {
    let n = g.n;
    let size = 1usize << n;
    let pi = boltzmann(g, beta).expect("enumerable");
    let vals: Vec<f64> = (0..size).map(|x| f(&spins(x, n))).collect();
    let mean: f64 = pi.iter().zip(&vals).map(|(p, v)| p * v).sum();
    let var: f64 = pi.iter().zip(&vals).map(|(p, v)| p * (v - mean).powi(2)).sum();
    let rate = |x: usize, i: usize| {
        let s = spins(x, n);
        if s[i] > 0 { 1.0 / m } else { (2.0 * beta * g.field(i, &s)).exp() / m }
    };
    let mut a = vec![0.0f64; size * size];
    for x in 0..size {
        for i in 0..n {
            let y = x ^ (1 << i);
            let q = rate(x, i);
            a[x * size + x] += q;
            a[x * size + y] -= (q * rate(y, i)).sqrt();
        }
    }
    for x in 0..size {
        for y in 0..size {
            a[x * size + y] += (pi[x] * pi[y]).sqrt();
        }
    }
    let w: Vec<f64> = (0..size).map(|x| pi[x].sqrt() * (vals[x] - mean)).collect();
    let z = solve_exact(&Spd::new(size, a, w.clone()));
    w.iter().zip(&z).map(|(a, b)| a * b).sum::<f64>() / var
}

struct Replica {
    sigma2: [f64; 2],
    sigma2_short: [f64; 2],
    births_per_m: f64,
}

fn one_unit() {
    let (beta, m, batch, batches) = (1.0, 1.0, 200.0, 20_000usize);
    println!("ONE UNIT, beta = 1, m = 1: long-run variance of the time-averaged on-indicator");
    println!("   h   lifetime     exact      batch means (+- se)    tau exact   Sokal at 0.1 m (error)");
    for &h in &[0.0f64, 0.7] {
        let mut g = GraphBuilder::new(1);
        g.bias(0, h);
        let g = g.build();
        let on = |s: &[i8]| if s[0] > 0 { 1.0 } else { 0.0 };
        let mut got = Vec::new();
        for (seed, lifetime) in [(21u64, Lifetime::Fixed), (22, Lifetime::Exponential)] {
            let mut pp = PointProcess::new(&g, beta, m, lifetime, seed);
            let means = pp.batch_means(50.0, batch, batches, &[&on]);
            let (v, se) = long_run_variance(&means[0], batch);
            let exact = unit_variance(beta, h, m, lifetime);
            let tau = unit_tau(beta, h, m, lifetime);
            let window = 0.1 * sokal_window(100_000, |k| unit_autocorrelation(beta, h, m, lifetime, k as f64 * 0.1));
            println!(
                "  {h:3}  {:11}  {exact:.6}   {v:.6} +- {se:.6}      {tau:.6}    {window:.6} ({:+.2}%)",
                format!("{lifetime:?}"),
                100.0 * (window / tau - 1.0)
            );
            got.push((v, se, window));
        }
        let ((vf, sf, wf), (ve, s_e, we)) = (got[0], got[1]);
        let r = ve / vf;
        let se_r = r * ((sf / vf).powi(2) + (s_e / ve).powi(2)).sqrt();
        println!("       exp / fixed: exact 2, batch means {r:.3} +- {se_r:.3}, Sokal's window {:.3}", we / wf);
        // The dense birth-death solve used on SK below, on this one unit: it must give unit_tau.
        let dense = exact_birth_death_tau(&g, beta, m, &on);
        println!("       the dense birth-death solve on this unit: tau {dense:.12} (unit_tau {:.12})", unit_tau(beta, h, m, Lifetime::Exponential));
    }
}

fn main() {
    one_unit();

    // Batches must be long against the slowest mode, and the cold cells have one of tens of m: a
    // first run at 50 m for every cell read sigma^2 up to 36% higher from batches four times as
    // long. So each temperature gets its own batch length, and the last columns hold it to account.
    let (burn, seeds, batches, quarter) = (500.0f64, 16u64, 320usize, 4usize);
    let batch_for = |beta: f64| if beta < 0.75 { 100.0 } else if beta < 1.25 { 500.0 } else { 1500.0 };
    println!();
    println!("SK, tau in units of m by batch means: {seeds} independent seeds per lifetime, each {burn} m of burn-in then");
    println!("{batches} batches of b = 100 / 500 / 1500 m at beta 0.5 / 1 / 1.5. se is the standard error across seeds.");
    println!("b/4 : b is sigma^2 from batches a quarter as long over sigma^2 at b; the bias at b is about a third of");
    println!("its shortfall from 1 (sigma^2(b) ~ sigma^2 (1 - kappa / b)).");
    println!("   n  beta   lifetime   births/m      tau_E (se)         tau_M (se)      b/4 : b   E      M      exact tau_E, tau_M");
    for &n in &[8usize, 16] {
        for &beta in &[0.5f64, 1.0, 1.5] {
            let batch = batch_for(beta);
            let g = sk(n, 11);
            let law = boltzmann(&g, beta).expect("enumerable");
            let var = |f: &dyn Fn(&[i8]) -> f64| {
                let vals: Vec<f64> = (0..law.len())
                    .map(|x| f(&(0..n).map(|i| if x >> i & 1 == 1 { 1 } else { -1 }).collect::<Vec<i8>>()))
                    .collect();
                let mean: f64 = law.iter().zip(&vals).map(|(p, v)| p * v).sum();
                law.iter().zip(&vals).map(|(p, v)| p * (v - mean).powi(2)).sum::<f64>()
            };
            let energy = |s: &[i8]| g.energy(s);
            let var_obs = [var(&energy), var(&magnetisation)];
            let mut taus = Vec::new();
            for lifetime in [Lifetime::Fixed, Lifetime::Exponential] {
                let reps: Vec<Replica> = std::thread::scope(|scope| {
                    let handles: Vec<_> = (0..seeds)
                        .map(|seed| {
                            let g = &g;
                            scope.spawn(move || {
                                let energy = |s: &[i8]| g.energy(s);
                                let mut pp = PointProcess::new(g, beta, 1.0, lifetime, 1000 + seed);
                                let short = batch / quarter as f64;
                                let rows = pp.batch_means(burn, short, batches * quarter, &[&energy, &magnetisation]);
                                let births_per_m = pp.births as f64 / pp.t;
                                let at_b = |o: usize| long_run_variance(&pool(&rows[o], quarter), batch).0;
                                let at_quarter = |o: usize| long_run_variance(&rows[o], short).0;
                                Replica { sigma2: [at_b(0), at_b(1)], sigma2_short: [at_quarter(0), at_quarter(1)], births_per_m }
                            })
                        })
                        .collect();
                    handles.into_iter().map(|h| h.join().expect("replica")).collect()
                });
                let stat = |o: usize| mean_se(&reps.iter().map(|r| r.sigma2[o] / (2.0 * var_obs[o])).collect::<Vec<_>>());
                let short = |o: usize| {
                    reps.iter().map(|r| r.sigma2_short[o]).sum::<f64>() / reps.iter().map(|r| r.sigma2[o]).sum::<f64>()
                };
                let births = reps.iter().map(|r| r.births_per_m).sum::<f64>() / reps.len() as f64;
                let (te, se_e) = stat(0);
                let (tm, se_m) = stat(1);
                let exact = if lifetime == Lifetime::Exponential && n <= 8 {
                    format!(
                        "   exact {:.3}, {:.3}",
                        exact_birth_death_tau(&g, beta, 1.0, &energy),
                        exact_birth_death_tau(&g, beta, 1.0, &magnetisation)
                    )
                } else {
                    String::new()
                };
                println!(
                    "  {n:2}  {beta:4}   {:11} {births:7.3}   {te:7.3} ({se_e:.3})   {tm:8.3} ({se_m:.3})      {:5.3}  {:5.3}{exact}",
                    format!("{lifetime:?}"),
                    short(0),
                    short(1)
                );
                taus.push([(te, se_e), (tm, se_m)]);
            }
            let ratio = |o: usize| {
                let ((f, sf), (e, s_e)) = (taus[0][o], taus[1][o]);
                let r = e / f;
                (r, r * ((sf / f).powi(2) + (s_e / e).powi(2)).sqrt())
            };
            let (re, sre) = ratio(0);
            let (rm, srm) = ratio(1);
            println!("                 ratio (exp / fixed)       {re:5.3} ({sre:.3})    {rm:6.3} ({srm:.3})");
        }
    }
}
