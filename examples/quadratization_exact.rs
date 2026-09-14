#![allow(missing_docs)]
// THE EXACT QUADRATIZATION PENALTY P*, and what the default costs against it at a matched budget.
//
// Rosenberg's reduction defines an ancilla y = x_a x_b and enforces it with a penalty; the crate's
// default is twice the sum of every coefficient's magnitude, more than the whole model can pay, and
// `hubo_vs_reduction` measured that the reduced arm stays worse than the native one at 1024x the
// budget with no ancilla ever broken -- mechanism: the penalty makes the landscape rigid. The
// contested question is whether that cost is the penalty's or the ancilla dimension's. With the
// exact least penalty in hand it separates:
//
//   P*     the least penalty at which every ground state of the reduced model respects every
//          ancilla, found by bisection on an exact elimination of the reduced graph against the
//          enumerated optimum of the original higher-order model (12 variables, 4,096 states)
//   arms   native anneal of the higher-order model; the reduction annealed at P*, 1.5 P*, 2 P*,
//          and at the default, all at the same number of single-spin flips and the same ladder,
//          scored on the ORIGINAL energy of the projected state
//
// run: cargo run --release --example quadratization_exact

use ferrotherm::exact::Elimination;
use ferrotherm::ftp::Program;
use ferrotherm::hubo::{self, Hubo};
use ferrotherm::reduce::{to_pairwise_with, Reduction};
use ferrotherm::rng::Pcg;
use ferrotherm::tempering::{anneal, geometric_ladder};

const N: usize = 12;

/// `t` random three-body +-1 terms on distinct triples.
fn instance(t: usize, seed: u64) -> Vec<([usize; 3], f64)> {
    let mut rng = Pcg::new(seed, 0x3B);
    let mut terms: Vec<([usize; 3], f64)> = Vec::new();
    while terms.len() < t {
        let mut v = [0usize; 3];
        for x in &mut v {
            *x = (rng.f64() * N as f64) as usize;
        }
        v.sort_unstable();
        if v[0] == v[1] || v[1] == v[2] || terms.iter().any(|(u, _)| *u == v) {
            continue;
        }
        terms.push((v, if rng.f64() < 0.5 { 1.0 } else { -1.0 }));
    }
    terms
}

fn hubo_of(terms: &[([usize; 3], f64)]) -> Hubo {
    let mut h = Hubo::new(N);
    for (v, w) in terms {
        h.add(v, *w).expect("distinct variables");
    }
    h
}

fn program_of(terms: &[([usize; 3], f64)]) -> Program {
    let mut ftp = format!("ftp 1\nspins {N}\n");
    for (v, w) in terms {
        ftp.push_str(&format!("factor {w} {} {} {}\n", v[0], v[1], v[2]));
    }
    Program::from_ftp(&ftp).expect("a well-formed program")
}

/// The original optimum by enumeration.
fn optimum(h: &Hubo) -> f64 {
    (0..(1usize << N))
        .map(|x| {
            let s: Vec<i8> = (0..N).map(|i| if (x >> i) & 1 == 1 { 1 } else { -1 }).collect();
            h.energy(&s)
        })
        .fold(f64::INFINITY, f64::min)
}

/// The reduced model's exact ground energy in the original's units, at this penalty.
fn reduced_ground(prog: &Program, penalty: f64) -> Option<(Reduction, f64)> {
    let red = to_pairwise_with(prog, Some(penalty)).ok()?;
    let g = red.program.to_graph().ok()?;
    let e = Elimination::default().ground_state(&g).ok()?.ground_energy? + red.offset;
    Some((red, e))
}

/// P* by bisection: the least penalty at which the reduced ground energy equals the optimum.
fn threshold(prog: &Program, e_star: f64, default: f64) -> Option<f64> {
    let ok = |p: f64| -> Option<bool> { Some(reduced_ground(prog, p)?.1 >= e_star - 1e-9) };
    let mut hi = default;
    if !ok(hi)? {
        return None;
    }
    let mut lo = 0.0;
    for _ in 0..40 {
        let mid = 0.5 * (lo + hi);
        if ok(mid)? {
            hi = mid;
        } else {
            lo = mid;
        }
        if hi - lo < 1e-4 * hi {
            break;
        }
    }
    Some(hi)
}

fn main() {
    let instances = 20u64;
    let seeds = 100u64;
    let (stages, sweeps) = (40usize, 20usize);
    println!("THE EXACT QUADRATIZATION PENALTY, AND THE DEFAULT AGAINST IT AT A MATCHED FLIP BUDGET\n");
    println!("  model    {N} variables, t random three-body +-1 terms on distinct triples; {instances} instances per t; optimum by enumeration");
    println!("  P*       least Rosenberg penalty at which the reduced ground energy equals the optimum (exact elimination), relative 1e-4");
    println!("  anneal   ladder 0.05..8.0, {stages} stages; native {sweeps} sweeps per stage; reduced arms scaled to the same flips; {seeds} seeds");
    println!("  score    P(GS): the projected state attains the optimum on the ORIGINAL energy; 'broken' = ancilla violated in the returned state\n");
    println!(
        "  {:>3} {:>4} {:>6} {:>8} {:>8} {:>9}   {:>7}   {:>7} {:>7} {:>7} {:>7} {:>8}   {:>7} {:>7}",
        "t", "inst", "anc", "P* mean", "default", "P*/def", "native", "at P*", "1.5 P*", "2 P*", "default", "def/scal", "brk P*", "brk def"
    );
    for &t in &[12usize, 18, 24] {
        let mut count = 0usize;
        let (mut anc, mut pstar, mut pdef) = (0.0f64, 0.0f64, 0.0f64);
        let mut pgs = [0.0f64; 6];
        let (mut brk_star, mut brk_def) = (0.0f64, 0.0f64);
        for seed in 0..instances {
            let terms = instance(t, 300 + seed);
            let h = hubo_of(&terms);
            let prog = program_of(&terms);
            let e_star = optimum(&h);
            let Some((red_default, _)) = reduced_ground(&prog, f64::NAN) else { continue };
            let default = red_default.penalty;
            let Some(p_star) = threshold(&prog, e_star, default) else { continue };
            count += 1;
            anc += red_default.ancillas as f64;
            pstar += p_star;
            pdef += default;
            let native_p = hubo::Params { beta_min: 0.05, beta_max: 8.0, stages, sweeps_per_stage: sweeps };
            let ladder = geometric_ladder(0.05, 8.0, stages);
            for seed2 in 0..seeds {
                if hubo::anneal(&h, &native_p, 7_000 + seed2).energy <= e_star + 1e-9 {
                    pgs[0] += 1.0 / (seeds * instances) as f64;
                }
            }
            // Arms 0..3: P*, 1.5 P*, 2 P*, default, all on the objective's ladder. Arm 4: the default
            // penalty on a ladder scaled by P*/default, so the penalty terms see the temperatures the
            // P* arm's do and the objective terms see a ladder that many times hotter -- the other
            // way of matching, and the reason there is no single fair ladder for a rigid landscape.
            for (ai, mult) in [1.0f64, 1.5, 2.0, f64::NAN, f64::INFINITY].iter().enumerate() {
                let penalty = if mult.is_finite() { mult * p_star } else { default };
                let Some((red, _)) = reduced_ground(&prog, penalty) else { continue };
                let g = red.program.to_graph().expect("pairwise");
                let scaled = ((sweeps * N) as f64 / g.n as f64).round().max(1.0) as usize;
                let beta_scale = if mult.is_infinite() { p_star / default } else { 1.0 };
                let sched: Vec<(f64, usize)> = ladder.iter().map(|&b| (b * beta_scale, scaled)).collect();
                for seed2 in 0..seeds {
                    let (state, e_red) = anneal(&g, &sched, 7_000 + seed2, None);
                    let original = h.energy(red.project(&state));
                    if original <= e_star + 1e-9 {
                        pgs[ai + 1] += 1.0 / (seeds * instances) as f64;
                    }
                    let broke = ((e_red + red.offset) - original).abs() > 1e-6;
                    if broke && ai == 0 {
                        brk_star += 1.0 / (seeds * instances) as f64;
                    }
                    if broke && ai == 3 {
                        brk_def += 1.0 / (seeds * instances) as f64;
                    }
                }
            }
        }
        let c = count.max(1) as f64;
        // P(GS) accumulators were normalised by the planned instance count; rescale to those run.
        let f = instances as f64 / c;
        println!(
            "  {t:>3} {count:>4} {:>6.1} {:>8.3} {:>8.1} {:>9.4}   {:>7.3}   {:>7.3} {:>7.3} {:>7.3} {:>7.3} {:>8.3}   {:>7.3} {:>7.3}",
            anc / c,
            pstar / c,
            pdef / c,
            (pstar / c) / (pdef / c),
            pgs[0] * f,
            pgs[1] * f,
            pgs[2] * f,
            pgs[3] * f,
            pgs[4] * f,
            pgs[5] * f,
            brk_star * f,
            brk_def * f
        );
    }
    println!("\n  WHAT THE TABLE SAYS.\n");
    println!("  P*/def is how much of the default penalty an instance actually needs. The four reduced columns are");
    println!("  the same flips and the same ladder at four penalties; if the reduction at P* matches the native");
    println!("  arm, the measured quadratization cost was the penalty's, and if it does not, it is the ancilla");
    println!("  dimension's. 'brk' is the fraction of returned states with a violated ancilla: at P* a broken");
    println!("  ancilla can tie the optimum, so some breakage there is the threshold's own definition. 'def/scal' is");
    println!("  the default penalty on a ladder scaled by P*/default: the penalty terms then see the P* arm's");
    println!("  temperatures and the objective terms a ladder that many times hotter -- the other way to match.");
}
