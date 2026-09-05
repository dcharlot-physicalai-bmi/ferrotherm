// JOULES PER INDEPENDENT SAMPLE -- the number this field does not report, measured on one machine.
//
// Every efficiency claim in thermodynamic computing is quoted per FLIP: flips per second, joules
// per flip, flips per watt. That prices the operation, and nobody buys operations. What a sampler
// is for is independent draws from a distribution, and a sampler that flips twice as fast while
// decorrelating three times more slowly has made the bill worse while improving every number in
// the press release.
//
//   J/flip                = joules / (nodes x sweeps)      -- what the field reports
//   J/independent sample  = joules / ESS,  ESS = draws / 2*tau_int
//
// So this sweeps the temperature and measures both, on the same graph, on this machine, with a
// real wattmeter, for two samplers that the two metrics rank differently:
//
//   * chromatic Gibbs is the cheapest possible flip, and at low temperature it stops moving;
//   * parallel tempering pays for a whole tuned ladder to produce one cold draw, so per flip it
//     can only lose -- and it keeps producing independent samples after Gibbs has frozen.
//
// The crossing is the point. J/flip barely moves with temperature; J/independent-sample diverges,
// and somewhere between the two ends the cheaper sampler changes identity. Same structural error as
// `z1t_ledger`'s iso-parameter comparison, one layer down: a ratio of per-operation costs is not a
// ratio of the costs of getting an answer.
//
// WHAT THIS REFUSES, each because an earlier cut of this example printed it as though it meant
// something:
//
//   * A DEAD LADDER. A hand-picked 12-rung geometric ladder over this range has its coldest pair at
//     acceptance 0.000 and completes ZERO round trips -- measured here, first run. Pricing it would
//     price twelve independent chains with an overhead. The ladder is now tuned by
//     `adaptive::adapt` and the replica count escalated until the WORST pair is alive; evenness
//     alone cannot say that, since a ladder too short for its range is evenly dead.
//   * tau MEASURED IN DIFFERENT UNITS. The tempering trace has one entry per round and the Gibbs
//     trace one per several sweeps; both are converted to COLD-REPLICA SWEEPS before anything is
//     divided by anything.
//   * AN UNRESOLVED tau. Its relative error is about 1/sqrt(ESS) -- at ESS 4 that is 50%, and the
//     first run of this example reported a four-significant-figure verdict on exactly that. Below
//     `MIN_ESS` the ratio is refused rather than printed.
//
// And what it does NOT refuse, because it is the answer rather than a failure: a sampler that
// produces LESS THAN ONE independent sample in the whole budget. That is reported as a one-sided
// bound -- J/independent-sample is at least the entire run's joules -- which is a fact, not an
// estimate, and it is the sharpest form the finding takes.
//
// run: cargo run --release -p ferrotherm-meter --example joules_per_sample

use ferrotherm::certify::tau_int;
use ferrotherm::graph::{Graph, GraphBuilder};
use ferrotherm::gibbs::Sampler;
use ferrotherm::host;
use ferrotherm::rng::Pcg;
use ferrotherm::tempering::parallel_tempering_observed;

/// A hardware-native instance: the Advantage's own P_8 fabric, random +-1 couplings, no fields.
///
/// Not a lattice. The point of a device graph is that it is the graph the machine has, and a
/// spin glass on it is the workload the machine exists for.
fn pegasus_glass(m: usize, seed: u64) -> Graph {
    let topo = ferrotherm::device::pegasus(m, 1.0);
    let mut rng = Pcg::new(seed, 0x9E37);
    let mut b = GraphBuilder::new(topo.graph.n);
    for i in 0..topo.graph.n {
        for k in topo.graph.offset[i]..topo.graph.offset[i + 1] {
            let j = topo.graph.nbr[k] as usize;
            if j > i {
                b.couple(i, j, if rng.f64() < 0.5 { -1.0 } else { 1.0 });
            }
        }
    }
    b.build()
}

/// An ESS below this cannot support a comparison, so this example refuses instead of printing one.
///
/// `tau_int`'s relative error is about `sqrt(2*tau/n)` = `1/sqrt(ESS)`; at ESS 25 that is 20%, which
/// is enough to rank two samplers that differ by more than that and honest about not resolving less.
const MIN_ESS: f64 = 25.0;

struct Measured {
    label: &'static str,
    flips: f64,
    seconds: f64,
    joules: f64,
    joules_total: f64,
    /// Integrated autocorrelation time, in COLD-REPLICA SWEEPS -- the same unit for both rows.
    tau: f64,
    draws: usize,
}

impl Measured {
    fn j_per_flip(&self) -> f64 {
        self.joules / self.flips
    }
    /// Effective sample size of the cold trace: `n / 2*tau_int`, the standard definition.
    fn ess(&self) -> f64 {
        self.draws as f64 / (2.0 * self.tau)
    }
    fn j_per_independent(&self) -> f64 {
        self.joules / self.ess()
    }
    /// What `tau_int` itself is worth here: `1/sqrt(ESS)`, which every number derived from it
    /// inherits.
    fn tau_rel_err(&self) -> f64 {
        1.0 / self.ess().sqrt()
    }
}

fn main() {
    let quiet = match host::require_quiet("a power and timing measurement") {
        Ok(q) => q,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(3);
        }
    };
    let mut meter = match ferrotherm_meter::Meter::detect() {
        Some(m) => m,
        None => {
            eprintln!(
                "no power backend on this machine, so there are no joules to report and this \
                 example refuses to print a modelled number in their place."
            );
            std::process::exit(2);
        }
    };

    let m = 8usize;
    let g = pegasus_glass(m, 0xC0FFEE);
    let n = g.n;
    let rounds = 8_000usize;
    let burn_in = 800usize;

    println!("JOULES PER INDEPENDENT SAMPLE, ACROSS TEMPERATURE");
    println!("  graph      Pegasus P_{m} fabric, {n} nodes, random +-1 couplings, no fields");
    println!("  machine    {}", meter.machine());
    println!("  budget     {rounds} rounds per point, {burn_in} burn-in, iso-flip between samplers");

    let idle = match meter.idle(std::time::Duration::from_secs(3)) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("no usable idle baseline: {e}");
            std::process::exit(2);
        }
    };
    println!(
        "  idle       {:.2} W +- {:.2} over {} readings, load1 {}\n",
        idle.watts,
        idle.sigma,
        idle.samples,
        idle.load1.map_or("unknown".to_string(), |l| format!("{l:.2}"))
    );

    println!("  {:>5} {:>6} {:>7}   {:>11} {:>11}   {:>13} {:>13}   {:>7} {:>8}  cheaper",
             "beta", "rungs", "trips", "J/flip Gibbs", "J/flip PT", "J/ind Gibbs", "J/ind PT",
             "flip", "indep");
    println!("  {:>62}   {:>7} {:>8}", "", "Gibbs by", "PT by");
    let mut rows: Vec<(f64, String)> = Vec::new();
    let mut flips_seen: Vec<f64> = Vec::new();

    for &beta_cold in &[0.5f64, 1.0, 1.5, 2.0, 3.0] {
        // ---- the ladder is TUNED, not guessed, and escalated until the WORST pair is alive -----
        let mut betas = Vec::new();
        let mut worst_tuned = 0.0f64;
        for replicas in [8usize, 16, 24, 48] {
            let p = ferrotherm::adaptive::Params {
                replicas,
                epochs: 5,
                rounds: 300,
                swap_every: 1,
                beta_min: 0.2_f64.min(beta_cold / 2.0),
                beta_max: beta_cold,
            };
            let out = ferrotherm::adaptive::adapt(&g, &p, 0x1AD);
            worst_tuned = out.swap_rates.iter().copied().fold(f64::INFINITY, f64::min);
            betas = out.betas;
            if worst_tuned >= 0.10 {
                break;
            }
        }
        let rungs = betas.len();
        if worst_tuned < 0.05 {
            println!("  {beta_cold:>5.2} {rungs:>6} {:>7}   ladder never came alive (worst pair {worst_tuned:.3}); nothing priced", "-");
            continue;
        }

        // ---- A. plain chromatic Gibbs, given the ladder's whole flip budget --------------------
        let sweeps_a = rounds * rungs;
        let mut trace_a: Vec<f64> = Vec::with_capacity(rounds);
        let run_a = meter
            .measure(idle, || {
                let mut s = Sampler::new(&g, beta_cold, 0xA11CE);
                for r in 0..sweeps_a {
                    s.sweep(None);
                    if r % rungs == 0 && r / rungs >= burn_in {
                        trace_a.push(g.energy(&s.s));
                    }
                }
            })
            .expect("the meter was open a moment ago");
        let a = Measured {
            label: "chromatic Gibbs",
            flips: (n * sweeps_a) as f64,
            seconds: run_a.seconds,
            joules: run_a.joules_above_idle,
            joules_total: run_a.joules_total,
            tau: tau_int(&trace_a) * rungs as f64, // draws -> cold-replica sweeps
            draws: trace_a.len(),
        };

        // ---- B. parallel tempering on the tuned ladder, same flip budget -----------------------
        let mut out = None;
        let run_b = meter
            .measure(idle, || {
                out = Some(parallel_tempering_observed(&g, &betas, rounds, 1, burn_in, 0xB0B, None));
            })
            .expect("the meter was open a moment ago");
        let (_res, tr) = out.expect("the closure ran");
        let cold = tr.energies.last().expect("a ladder has rungs");
        let b = Measured {
            label: "parallel tempering",
            flips: (n * rounds * rungs) as f64,
            seconds: run_b.seconds,
            joules: run_b.joules_above_idle,
            joules_total: run_b.joules_total,
            tau: tau_int(cold),
            draws: cold.len(),
        };

        // ---- report ----------------------------------------------------------------------------
        // A sampler below one effective sample did not produce an independent draw at all. That is
        // reported as a BOUND -- J/indep is at least the whole run's joules -- rather than refused,
        // because it is a fact and it is the sharpest form this finding takes.
        let cell = |x: &Measured| -> String {
            if x.ess() < 1.0 {
                format!(">={:.2}", x.joules)
            } else if x.ess() < MIN_ESS {
                "unresolved".to_string()
            } else {
                format!("{:.4}", x.j_per_independent())
            }
        };
        let verdict = if a.ess() < 1.0 && b.ess() >= MIN_ESS {
            "PT only"
        } else if a.ess() < MIN_ESS || b.ess() < MIN_ESS {
            "unresolved"
        } else if b.j_per_independent() < a.j_per_independent() {
            "PT"
        } else {
            "Gibbs"
        };
        // The ratio in each metric, stated rather than left to be divided by the reader. When Gibbs
        // never decorrelated its J/indep is a lower bound, so the ratio is a lower bound too -- and
        // it is the sharpest number this example produces.
        let flip_x = b.j_per_flip() / a.j_per_flip();
        let indep_x = if a.ess() < 1.0 && b.ess() >= MIN_ESS {
            format!(">={:.0}x", a.joules / b.j_per_independent())
        } else if a.ess() >= MIN_ESS && b.ess() >= MIN_ESS {
            format!("{:.2}x", a.j_per_independent() / b.j_per_independent())
        } else {
            "-".to_string()
        };
        println!(
            "  {beta_cold:>5.2} {rungs:>6} {:>7}   {:>11.3e} {:>11.3e}   {:>13} {:>13}   {:>7} {indep_x:>8}  {verdict}",
            tr.round_trips,
            a.j_per_flip(),
            b.j_per_flip(),
            cell(&a),
            cell(&b),
            format!("{:.2}x", 1.0 / flip_x)
        );
        flips_seen.push(a.j_per_flip());
        rows.push((
            beta_cold,
            format!(
                "beta {beta_cold}: {} tau {:.0} sweeps, ESS {:.1} (+-{:.0}% on tau); {} tau {:.0}, \
                 ESS {:.1} (+-{:.0}%). Above idle {:.2} J and {:.2} J against an idle drift of \
                 +-{:.2} J; at the wall, idle included, {:.2} J and {:.2} J.",
                a.label,
                a.tau,
                a.ess(),
                100.0 * a.tau_rel_err(),
                b.label,
                b.tau,
                b.ess(),
                100.0 * b.tau_rel_err(),
                a.joules,
                b.joules,
                idle.sigma * a.seconds.max(b.seconds),
                a.joules_total,
                b.joules_total
            ),
        ));
    }

    println!("\n  WHAT THE TABLE SAYS.\n");
    println!("  J/flip is nearly flat in temperature and always favours plain Gibbs: it is the same");
    println!("  kernel, and tempering runs a whole ladder to produce one cold draw. That is the");
    println!("  number this field publishes, and on it Gibbs wins everywhere.");
    println!();
    println!("  J/independent-sample is not flat. As the glass freezes, Gibbs' autocorrelation runs");
    println!("  away and its cost per ANSWER diverges while its cost per flip does not move. Where");
    println!("  the verdict column says PT, the two metrics disagree about which sampler is cheaper,");
    println!("  and only one of them is about the cost of getting an answer.");
    println!();
    println!("  PER-POINT EVIDENCE, so no row is read as tighter than it is:");
    for (_, line) in &rows {
        println!("    {line}");
    }
    println!("    tau_int's relative error is about 1/sqrt(ESS); every J/indep figure inherits it,");
    println!("    and a row below ESS {MIN_ESS:.0} is printed as `unresolved` rather than estimated.");
    if let (Some(lo), Some(hi)) = (
        flips_seen.iter().copied().reduce(f64::min),
        flips_seen.iter().copied().reduce(f64::max),
    ) {
        println!(
            "    The SAME Gibbs kernel measured {lo:.3e} to {hi:.3e} J/flip across these rows, a \
             spread of"
        );
        println!(
            "    {:.0}%. That is this machine's reproducibility on a J/flip figure, and no claim in",
            100.0 * (hi / lo - 1.0)
        );
        println!("    that column -- ours or anyone's -- is tighter than it.");
    }
    println!("    `>=X` means the sampler produced LESS THAN ONE independent draw in the budget, so");
    println!("    the whole run's joules is a lower bound on the cost of its first one.");
    if let Some(c) = quiet.caveat() {
        println!("\n  {c}");
    }
}
