#![allow(missing_docs)]
// DOES RAMPING THE ENCODING PENALTY HELP? The crate shipped `Schedule::ramp_domain_wall` on the
// premise that a penalty which starts weak lets the sampler move and finishes strong so the
// constraint binds. Until 2026-09-13 no solver applied it (see `Penalties`); now that one does,
// the premise can be measured at an EQUAL sweep budget against holding the penalty constant.
//
// Model: eight one-hot categoricals over eight values under `all_different` -- a permutation --
// with an objective of twenty random +-1 value literals and ten random +-1 pairs, so the optimum is
// a specific permutation found by enumerating all 40,320. The compiled penalty is the certified
// one; a schedule's scale multiplies it. (The value literals an objective needs are one-hot's, so
// the ramp is exercised on the one-hot codeword penalty; the channel scales every encoding's.)
//
// Six schedules on the same ladder, 0.05..8.0 over 120 stages of 40 sweeps: the penalty constant
// at the certified value, constant at half and at twice it, and ramped 0.25 -> 1, 0.5 -> 1 and
// 0.25 -> 2. Two hundred seeds per instance, the same seeds for every schedule, so every
// comparison is paired.
//
// run: cargo run --release --example penalty_ramp

use ferrotherm::encode::Encoding;
use ferrotherm::model::{Expr, Lit, Model, Sense, Var};
use ferrotherm::rng::Pcg;
use ferrotherm::schedule::Schedule;
use ferrotherm::tts::wilson;

const NV: usize = 8;
const K: i64 = 8;

/// A pair term: two value literals and a coefficient.
type Pair = ((usize, i64), (usize, i64), f64);

struct Instance {
    singles: Vec<(usize, i64, f64)>,
    pairs: Vec<Pair>,
}

fn instance(seed: u64) -> Instance {
    let mut rng = Pcg::new(seed, 0x9A);
    let sign = |r: &mut Pcg| if r.f64() < 0.5 { 1.0 } else { -1.0 };
    let singles = (0..20).map(|_| ((rng.f64() * NV as f64) as usize, (rng.f64() * K as f64) as i64, sign(&mut rng))).collect();
    let mut pairs = Vec::new();
    while pairs.len() < 10 {
        let a = (rng.f64() * NV as f64) as usize;
        let b = (rng.f64() * NV as f64) as usize;
        if a == b {
            continue;
        }
        pairs.push(((a, (rng.f64() * K as f64) as i64), (b, (rng.f64() * K as f64) as i64), sign(&mut rng)));
    }
    Instance { singles, pairs }
}

impl Instance {
    fn value(&self, assignment: &[i64]) -> f64 {
        let mut v = 0.0;
        for &(var, val, c) in &self.singles {
            if assignment[var] == val {
                v += c;
            }
        }
        for &((a, va), (b, vb), c) in &self.pairs {
            if assignment[a] == va && assignment[b] == vb {
                v += c;
            }
        }
        v
    }

    /// The exact optimum over all permutations.
    fn optimum(&self) -> f64 {
        let mut best = f64::INFINITY;
        let mut perm: Vec<i64> = (0..K).collect();
        fn heap(k: usize, perm: &mut Vec<i64>, f: &mut dyn FnMut(&[i64])) {
            if k == 1 {
                f(perm);
                return;
            }
            heap(k - 1, perm, f);
            for i in 0..k - 1 {
                if k.is_multiple_of(2) {
                    perm.swap(i, k - 1);
                } else {
                    perm.swap(0, k - 1);
                }
                heap(k - 1, perm, f);
            }
        }
        heap(NV, &mut perm, &mut |p| best = best.min(self.value(p)));
        best
    }

    fn model(&self) -> (Model, Vec<Var>) {
        let mut m = Model::new();
        let vars: Vec<Var> = (0..NV).map(|i| m.categorical_as(&format!("v{i}"), K as usize, Encoding::OneHot)).collect();
        m.all_different(vars.iter().copied());
        for &(var, val, c) in &self.singles {
            m.objective(Sense::Minimize, Expr::lit(c, Lit::Is(vars[var], val)));
        }
        for &((a, va), (b, vb), c) in &self.pairs {
            m.objective(Sense::Minimize, Expr::pair(c, Lit::Is(vars[a], va), Lit::Is(vars[b], vb)));
        }
        (m, vars)
    }
}

/// A named schedule.
type Named = (&'static str, Schedule);

fn main() {
    let instances = 20u64;
    let tries = 200u64;
    let base = Schedule::geometric(0.05, 8.0, 120, 40);
    let schedules: Vec<Named> = vec![
        ("constant 1", base.clone()),
        ("constant 0.5", base.clone().ramp_domain_wall(0.5, 0.5)),
        ("constant 2", base.clone().ramp_domain_wall(2.0, 2.0)),
        ("ramp 0.25->1", base.clone().ramp_domain_wall(0.25, 1.0)),
        ("ramp 0.5->1", base.clone().ramp_domain_wall(0.5, 1.0)),
        ("ramp 0.25->2", base.clone().ramp_domain_wall(0.25, 2.0)),
    ];
    println!("RAMPING THE ENCODING PENALTY AGAINST HOLDING IT, AT AN EQUAL SWEEP BUDGET\n");
    println!("  model      {NV} one-hot categoricals over {K} values, all_different, 20 random +-1 value literals + 10 random +-1 pairs; optimum over 40,320 permutations");
    println!("  penalty    the model's certified penalty is the schedule's 1.0; a schedule scales it per stage");
    println!("  ladder     0.05..8.0, 120 stages x 40 sweeps = {} sweeps for every schedule; {instances} instances x {tries} paired seeds\n", base.total_sweeps());
    let mut feas = vec![0usize; schedules.len()];
    let mut opt = vec![0usize; schedules.len()];
    let mut wins = vec![0usize; schedules.len()]; // instances on which this schedule's P(opt) beats constant 1's
    let mut ties = vec![0usize; schedules.len()];
    let mut penalties = Vec::new();
    for seed in 0..instances {
        let inst = instance(500 + seed);
        let (mut m, _) = inst.model();
        let p = m.certified_penalty().unwrap_or_else(|_| 2.0 * (inst.singles.len() + inst.pairs.len()) as f64);
        m.fixed_penalty(p);
        penalties.push(p);
        let c = m.compile().expect("a compilable permutation model");
        let e_star = inst.optimum();
        let mut per_inst = vec![0usize; schedules.len()];
        for (si, (_, sched)) in schedules.iter().enumerate() {
            for sol in c.solve_all_with(sched, tries) {
                if !sol.feasible() {
                    continue;
                }
                feas[si] += 1;
                let assignment: Vec<i64> = (0..NV).map(|i| sol.get(&format!("v{i}")).expect("decoded")).collect();
                if inst.value(&assignment) <= e_star + 1e-9 {
                    opt[si] += 1;
                    per_inst[si] += 1;
                }
            }
        }
        for si in 0..schedules.len() {
            if per_inst[si] > per_inst[0] {
                wins[si] += 1;
            } else if per_inst[si] == per_inst[0] {
                ties[si] += 1;
            }
        }
    }
    let total = (instances * tries) as usize;
    let mean_p = penalties.iter().sum::<f64>() / penalties.len() as f64;
    println!("  certified penalty: mean {mean_p:.2} over instances\n");
    println!("  {:<14}   {:>10}   {:>10} {:>18}   {:>22}", "schedule", "P(feasible)", "P(optimum)", "95% Wilson", "vs constant 1 (inst)");
    for (si, (name, _)) in schedules.iter().enumerate() {
        let (lo, hi) = wilson(opt[si], total, 1.96);
        println!(
            "  {name:<14}   {:>10.3}   {:>10.3} {:>8.3} .. {:>6.3}   {:>2} better, {:>2} tied, {:>2} worse",
            feas[si] as f64 / total as f64,
            opt[si] as f64 / total as f64,
            lo,
            hi,
            wins[si],
            ties[si],
            instances as usize - wins[si] - ties[si]
        );
    }
    println!("\n  WHAT THE TABLE SAYS.\n");
    println!("  Every schedule spends the same sweeps on the same ladder with the same seeds; only the penalty's");
    println!("  path differs. P(optimum) is the fraction of returned answers that are feasible AND attain the exact");
    println!("  optimum. A ramp that beats 'constant 1' with non-overlapping intervals is the shipped premise");
    println!("  confirmed; one that does not is a knob that should default to off. The constant half and double");
    println!("  say whether the certified penalty itself was the binding choice.");
}
