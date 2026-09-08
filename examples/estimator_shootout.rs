// `missing_docs` is denied workspace-wide and is right to be: it guards the API surface, and
// every public item in every library here carries a doc. An EXAMPLE has no API surface -- it is
// a program, and its helpers are private to it -- so the lint has nothing to guard and asks for
// doc comments on `fn main`'s scaffolding instead. Scoped off here rather than weakened there.
#![allow(missing_docs)]
// Six ways to fit an energy-based model, scored against the one that cannot be run.
//
// THE PROBLEM WITH EVERY PUBLISHED COMPARISON of EBM training methods is that it has no ceiling.
// A paper reports that its method reached log-likelihood -3.0 and a baseline reached -3.4, and the
// reader cannot tell whether -3.0 is nearly everything the model could have done or half of it,
// because the best achievable value for that structure on that data is not computed. Without it,
// "better than the baseline" is the only sayable thing, and it is a statement about the baseline.
//
// `ebm::train_exact` computes the ceiling. It ascends the TRUE maximum-likelihood gradient
//
//     d log L / d J_ij  =  <s_i s_j>_data  -  <s_i s_j>_model
//
// with BOTH averages taken by enumeration over all 2^n states -- no sampler, no bias, no k. It is
// useless on a real model and that is fine: its job is to be the number the useful methods are a
// fraction of. Every row below is reported as that fraction.
//
// THE SIX METHODS split into two families that fail differently.
//
//   CD-1, CD-10                 approximate the model average by k sweeps FROM THE DATA
//   PCD-10                      ... from a chain that is never reset
//   pseudolikelihood            does not target the likelihood at all: a product of conditionals
//   minimum probability flow    ... nor this: outflow from the data in the first instant
//   ratio matching              ... nor this: the ratio between a state and its one-bit flips
//
// The first three are the right objective computed wrongly. The last three are a different
// objective computed exactly -- and they are the SAME different objective three ways, all three
// being a sum over (row, site) of a loss on the flip margin u = s_i f_i. That is why they are here
// together: they share a gradient, a loop, and a cost, and differ only in the price they put on a
// small margin. See `FlipLoss` in src/ebm.rs for the table.
//
// THE AXIS THIS SWEEPS, AND WHY IT IS THE RIGHT ONE. A single fixture cannot separate these
// methods, and an earlier version of this program is the evidence: at coupling strengths where the
// model mixes freely, all six reach 100.0% of the ceiling to one decimal place and the comparison
// says nothing. The sampler-free methods can only earn their place where the SAMPLER IS THE
// PROBLEM, so the sweep here is over coupling strength -- the same axis the mixing-expressivity
// tradeoff is stated on, and the one that decides whether a negative phase is affordable. Every
// method is given its own best step size at every point on it, because a shared step would report
// which method one number happened to suit.
//
// WHAT IS MEASURED AND WHAT IS NOT. This is 10 spins. Contrastive divergence's bias is a claim
// about a chain that cannot reach the model's modes, and ten sweeps of a ten-spin machine can
// reach anywhere; the regime where CD genuinely fails is larger than anything `train_exact` can
// price. So a small gap here is a LOWER BOUND on the gap at scale, not a measurement of it, and
// this program says which of the two it is reporting rather than leaving it to be read either way.

use ferrotherm::ebm::{
    Dataset, Error, FitParams, Params, Trained, exact_log_likelihood, minimum_probability_flow,
    pseudo_log_likelihood, ratio_matching, train, train_exact, train_mpf, train_pseudolikelihood,
    train_ratio_matching,
};
use ferrotherm::graph::{Graph, GraphBuilder};
use ferrotherm::rng::Pcg;

/// A model with a definite correlation structure to recover: a ring with two chords, mixed signs,
/// biases on two sites, and every coupling scaled by `scale`. Mixed signs matter -- a ferromagnet
/// is nearly the easiest thing any of these methods can be asked to fit and would flatter all six
/// equally; the frustration is what makes the landscape hard as `scale` rises.
fn truth(n: usize, scale: f64) -> Graph {
    let mut b = GraphBuilder::new(n);
    for i in 0..n {
        b.couple(i, (i + 1) % n, scale * if i % 2 == 0 { 0.8 } else { -0.6 });
    }
    b.couple(0, n / 2, scale * 0.5);
    b.couple(1, n / 2 + 1, scale * -0.45);
    b.bias(0, scale * 0.4);
    b.bias(n / 2, scale * -0.35);
    b.build()
}

/// The same edge set with every weight and bias at zero: what each method is handed to start from.
fn blank(g: &Graph) -> Graph {
    let mut b = GraphBuilder::new(g.n);
    for i in 0..g.n {
        for k in g.offset[i]..g.offset[i + 1] {
            let j = g.nbr[k] as usize;
            if j > i {
                b.couple(i, j, 0.0);
            }
        }
    }
    b.build()
}

/// Exact draws from `g`, by inverting the cumulative distribution over all `2^n` states. The data
/// is therefore from the model EXACTLY, with no burn-in and no autocorrelation -- otherwise a
/// sampler's mixing would enter the comparison on the data side as well as the method side, and
/// this program's whole subject is what a sampler costs.
fn draw_from(g: &Graph, rows: usize, seed: u64) -> Dataset {
    let p = ferrotherm::ising::exact_boltzmann(g, 1.0);
    let mut cdf = Vec::with_capacity(p.len());
    let mut acc = 0.0;
    for &x in &p {
        acc += x;
        cdf.push(acc);
    }
    let mut rng = Pcg::new(seed, 11);
    let rows = (0..rows)
        .map(|_| {
            let u = rng.f64() * acc;
            let m = cdf.partition_point(|&c| c < u).min(p.len() - 1);
            (0..g.n).map(|i| if (m >> i) & 1 == 1 { 1i8 } else { -1 }).collect()
        })
        .collect();
    Dataset { visible: g.n, rows }
}

/// Replace `frac` of the rows with a uniformly random state. Not noise on top of the data -- a row
/// the model should find very improbable, which is what separates a bounded loss from an unbounded
/// one. A corrupted row's flip margins go NEGATIVE, and that is the whole region the three
/// sampler-free losses disagree in.
fn corrupt(data: &Dataset, frac: f64, seed: u64) -> Dataset {
    let mut rng = Pcg::new(seed, 909);
    let rows = data
        .rows
        .iter()
        .map(|r| {
            if rng.f64() < frac {
                (0..r.len()).map(|_| if rng.f64() < 0.5 { 1i8 } else { -1 }).collect()
            } else {
                r.clone()
            }
        })
        .collect();
    Dataset { visible: data.visible, rows }
}

const NAMES: [&str; 6] =
    ["CD-1", "CD-10", "PCD-10", "pseudolikelihood", "min prob flow", "ratio matching"];

/// Ascend until the method's OWN objective stops improving, and report how many epochs that took.
///
/// A FIXED EPOCH COUNT MEASURES A BUDGET, NOT A METHOD, and this program has the receipts. Its
/// first version gave every method 800 epochs; pseudolikelihood and ratio matching then won at the
/// largest step in the grid in six of thirty cells, which means their best was outside the grid and
/// their shortfall against contrastive divergence was partly an artifact of it. Raising the budget
/// to 8000 moved minimum probability flow at the strongest coupling from 97.2% to 88.0% -- worse,
/// because the earlier number was an UNCONVERGED fit passing through a better-likelihood region on
/// its way to its own optimum. Early stopping was flattering it, and no amount of widening the grid
/// fixes that.
///
/// So the budget is removed instead. Each method ascends its own objective in chunks until the
/// improvement falls below `TOL`, which for the two concave objectives is the optimum and for
/// ratio matching is a local one. The method is NEVER shown the likelihood it will be scored on --
/// that would be early stopping on the test statistic, which is the same error in a smarter suit.
///
/// The returned epoch count is reported: a method that hits `CAP` has not converged and its row is
/// a lower bound.
fn to_convergence(
    fit: Fit,
    objective: Objective,
    structure: &Graph,
    data: &Dataset,
    lr: f64,
) -> (Graph, usize) {
    const CHUNK: usize = 250;
    const CAP: usize = 40_000;
    const TOL: f64 = 1e-9;
    let mut g = fit(structure, data, &FitParams { epochs: 0, lr, l2: 0.0 }).unwrap().graph;
    let mut prev = objective(&g, data).unwrap();
    let mut used = 0;
    while used < CAP {
        g = fit(&g, data, &FitParams { epochs: CHUNK, lr, l2: 0.0 }).unwrap().graph;
        used += CHUNK;
        let now = objective(&g, data).unwrap();
        // A step too large makes this NEGATIVE rather than small, and stopping on it is correct:
        // that step has overshot and another chunk of it will not help. The grid's other steps
        // decide the row.
        if now - prev < TOL {
            break;
        }
        prev = now;
    }
    (g, used)
}

/// A trainer, by the signature the three sampler-free ones share.
type Fit = fn(&Graph, &Dataset, &FitParams) -> Result<Trained, Error>;
/// The objective one of them ascends, by the signature those three share.
type Objective = fn(&Graph, &Dataset) -> Result<f64, Error>;

/// The three sampler-free methods, each paired with the objective it actually ascends.
const SAMPLER_FREE: [(Fit, Objective); 3] = [
    (train_pseudolikelihood, pseudo_log_likelihood),
    (train_mpf, minimum_probability_flow),
    (train_ratio_matching, ratio_matching),
];

/// Contrastive divergence at a fixed budget, because it has no fixed point to converge to: its
/// gradient is sampled, so the parameters random-walk around the optimum forever and "stopped
/// improving" is a statement about the noise. Its step decays internally, which is what settles it.
fn fit_cd(structure: &Graph, data: &Dataset, epochs: usize, lr: f64) -> [Graph; 3] {
    let cd1 = Params {
        epochs,
        k: 1,
        positive_sweeps: 1,
        learning_rate: lr,
        batch: 32,
        persistent: false,
    };
    let cd10 = Params { k: 10, ..cd1 };
    let pcd = Params { persistent: true, ..cd10 };
    [
        train(structure, data, &cd1, 7).unwrap().graph,
        train(structure, data, &cd10, 7).unwrap().graph,
        train(structure, data, &pcd, 7).unwrap().graph,
    ]
}

/// Every method at its own best step, scored on `score` -- which is the data it was fitted on
/// everywhere except the robustness table, where a fit on corrupted rows is scored on clean ones.
///
/// Returns the exact log-likelihood, the winning step, and the epochs the sampler-free methods
/// needed to converge (zero for the three that run to a budget).
fn all_methods(
    structure: &Graph,
    data: &Dataset,
    score: &Dataset,
    cd_epochs: usize,
    grid: &[f64],
) -> [(f64, f64, usize); 6] {
    let mut best = [(f64::NEG_INFINITY, 0.0, 0usize); 6];
    for &lr in grid {
        for (k, g) in fit_cd(structure, data, cd_epochs, lr).iter().enumerate() {
            let v = exact_log_likelihood(g, score).unwrap();
            if v > best[k].0 {
                best[k] = (v, lr, 0);
            }
        }
        for (k, &(fit, obj)) in SAMPLER_FREE.iter().enumerate() {
            let (g, used) = to_convergence(fit, obj, structure, data, lr);
            let v = exact_log_likelihood(&g, score).unwrap();
            if v > best[3 + k].0 {
                best[3 + k] = (v, lr, used);
            }
        }
    }
    best
}

/// The ceiling: exact maximum likelihood, run to convergence on the likelihood itself -- which is
/// legitimate here and only here, because for this one method the objective IS the score.
fn ceiling_of(structure: &Graph, data: &Dataset, grid: &[f64]) -> (f64, usize) {
    let mut best = (f64::NEG_INFINITY, 0);
    for &lr in grid {
        let (g, used) = to_convergence(train_exact, exact_log_likelihood, structure, data, lr);
        let v = exact_log_likelihood(&g, data).unwrap();
        if v > best.0 {
            best = (v, used);
        }
    }
    best
}

/// The reachable fraction: where a fit landed between an untrained model and the ceiling.
///
/// A raw likelihood ratio would be useless -- the absolute log-likelihood of a ten-spin model is
/// bounded well away from zero, so a fit that learned almost nothing still reads as 90-odd percent
/// of the ceiling. Measured against the RANGE instead, an untrained model is 0% by construction.
fn pct(v: f64, floor: f64, ceiling: f64) -> f64 {
    100.0 * (v - floor) / (ceiling - floor)
}

fn main() {
    let n = 10;
    let rows = 1_000;
    // Contrastive divergence runs to a budget and the rest run to convergence; see `to_convergence`
    // for why a shared epoch count measures neither.
    let cd_epochs = 800;
    let grid = [0.005f64, 0.02, 0.05, 0.15, 0.4];
    let base = truth(n, 1.0);
    let start = blank(&base);

    println!(
        "Six estimators against the exact-maximum-likelihood ceiling\n\n\
         {n} spins, {} edges, {rows} exact draws, step swept over {grid:?}\n\
         contrastive divergence {cd_epochs} epochs ({} updates); every other method to convergence\n",
        base.n_edges,
        cd_epochs * rows / 32
    );

    // THE HEADLINE. Coupling strength is the axis the sampler-free case is made on: a negative
    // phase is only a liability where the chain cannot get anywhere, and raising the couplings is
    // what stops it getting anywhere. Reported as percent of reachable so rows compare across
    // scales, with the ceiling and the range in nats beside them so a percent can be converted back
    // into the model's own currency.
    let scales = [0.5f64, 1.0, 1.5, 2.0, 2.5];
    println!("PERCENT OF REACHABLE, AS THE LANDSCAPE HARDENS\n");
    print!("{:<20}", "coupling scale");
    for s in scales {
        print!("{s:>9.1}");
    }
    println!();
    let mut table = vec![vec![0.0f64; scales.len()]; 6];
    let mut steps = vec![vec![0.0f64; scales.len()]; 6];
    let mut epochs = vec![vec![0usize; scales.len()]; 6];
    let (mut ceilings, mut floors) = (Vec::new(), Vec::new());
    for (si, &s) in scales.iter().enumerate() {
        let t = truth(n, s);
        let data = draw_from(&t, rows, 3);
        let floor = exact_log_likelihood(&start, &data).unwrap();
        let (ceiling, _) = ceiling_of(&start, &data, &grid);
        for (k, &(v, lr, used)) in
            all_methods(&start, &data, &data, cd_epochs, &grid).iter().enumerate()
        {
            table[k][si] = pct(v, floor, ceiling);
            steps[k][si] = lr;
            epochs[k][si] = used;
        }
        ceilings.push(ceiling);
        floors.push(floor);
    }
    for (k, name) in NAMES.iter().enumerate() {
        print!("{name:<20}");
        for v in &table[k] {
            print!("{v:>8.1}%");
        }
        println!();
    }
    print!("{:<20}", "ceiling (nats)");
    for c in &ceilings {
        print!("{c:>9.3}");
    }
    println!();
    print!("{:<20}", "range (nats)");
    for (c, f) in ceilings.iter().zip(&floors) {
        print!("{:>9.3}", c - f);
    }
    println!();

    // A SWEEP WHOSE WINNER IS THE LAST GRID POINT HAS NOT FOUND THE BEST. Marked rather than
    // silently reported -- reading a number off the edge of a grid is reading the grid.
    println!("\n{:<20}   winning step (* = at the edge of the grid)", "");
    let edge = *grid.last().unwrap();
    let mut truncated = 0;
    for (k, name) in NAMES.iter().enumerate() {
        print!("{name:<20}");
        for v in &steps[k] {
            if (v - edge).abs() < 1e-12 {
                truncated += 1;
                print!("{v:>8.3}*");
            } else {
                print!("{v:>9.3}");
            }
        }
        println!();
    }
    if truncated > 0 {
        println!("  {truncated} of {} cells won at the grid edge", 6 * scales.len());
    }

    // The convergence cost, which is the thing a fixed-epoch table hides. Contrastive divergence
    // is not here because it does not converge -- see `to_convergence`.
    println!("\n{:<20}   epochs to convergence (sampler-free methods only)", "");
    let mut capped = 0;
    for k in 3..6 {
        print!("{:<20}", NAMES[k]);
        for e in &epochs[k] {
            if *e >= 40_000 {
                capped += 1;
                print!("{e:>8}+");
            } else {
                print!("{e:>9}");
            }
        }
        println!();
    }
    if capped > 0 {
        println!(
            "\n  READ THE TABLE ABOVE WITH THIS ONE. {capped} cells hit the convergence cap, so\n\
             \x20 their percentages are LOWER BOUNDS on what the method reaches, not measurements\n\
             \x20 of it -- these objectives are cheap per epoch and slow in epochs, and the epoch\n\
             \x20 count is what rises with coupling strength."
        );
    }

    // ROBUSTNESS. The three sampler-free losses differ out in the negative-margin tail: exp(-u) is
    // unbounded, sigma(-2u)^2 saturates at 1, log sigma(2u) is linear in -u between them. That is a
    // prediction about which one a hopeless data point can drag furthest, and it is tested at TWO
    // data sizes, because a consistent estimator with plenty of data converges to the same place
    // whatever the shape of its tail -- so if the tail is ever going to matter, it matters where
    // the data is thin. Fits are scored on the CLEAN likelihood: the question is what a fit on
    // corrupted data costs on the real distribution, not how well it fitted the corruption.
    println!("\nROBUSTNESS -- rows replaced by uniform noise, scored on the CLEAN data\n");
    let fracs = [0.0f64, 0.05, 0.15, 0.30];
    for &(size, label) in &[(200usize, "200 rows"), (4_000usize, "4000 rows")] {
        let t = truth(n, 1.5);
        let clean = draw_from(&t, size, 3);
        let floor = exact_log_likelihood(&start, &clean).unwrap();
        let (ceiling, _) = ceiling_of(&start, &clean, &grid);
        println!("{label} (ceiling {ceiling:.3} nats, range {:.3})\n", ceiling - floor);
        print!("{:<20}", "corrupted");
        for f in fracs {
            print!("{:>9.0}%", 100.0 * f);
        }
        println!("     lost");
        let mut got = vec![vec![0.0f64; fracs.len()]; 6];
        for (fi, &f) in fracs.iter().enumerate() {
            let dirty = corrupt(&clean, f, 41);
            for (k, &(v, _, _)) in
                all_methods(&start, &dirty, &clean, cd_epochs, &grid).iter().enumerate()
            {
                got[k][fi] = pct(v, floor, ceiling);
            }
        }
        for (k, name) in NAMES.iter().enumerate() {
            print!("{name:<20}");
            for v in &got[k] {
                print!("{v:>9.1}%");
            }
            println!("{:>9.1}", got[k][0] - got[k][fracs.len() - 1]);
        }
        println!();
    }
}
