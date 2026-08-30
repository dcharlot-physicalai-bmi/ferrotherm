// Does the error bar mean what it says?
//
// THE CLAIM UNDER TEST is this crate's own. `samples::Estimate` quotes a standard error computed as
// `sqrt(var / ess)` with `ess = N / (2 tau_int)`, rather than the `sqrt(var / N)` that a caller
// would write by hand. That is the textbook correction for autocorrelated draws and it is easy to
// state; whether the interval it produces actually contains the right answer 95% of the time is a
// measurement, and this is it.
//
// THE ORACLE. Every model here is small enough to enumerate, so `samples::enumerate` gives the
// EXACT `<s_i>` for every site -- not a longer run of the same sampler, which would only measure
// self-consistency. An interval either contains the exact number or it does not, and counting is
// the whole method.
//
// WHAT IS BEING COMPARED. For each chain and each site, two intervals around the SAME estimate:
//
//   corrected   value +- 1.96 * sqrt(var / ess)      what this crate returns
//   naive       value +- 1.96 * sqrt(var / N)        what the correction is undone to give
//
// They differ by exactly sqrt(2 tau), so this isolates the correction and nothing else.
//
// THE THIRD COLUMN IS THE POINT. `certify` reports a finding called `Undermixed` when tau is large
// relative to the run. The question worth answering is not "is the corrected interval always
// right" -- no interval built from an estimated tau can be -- but "does the certificate know when
// to stop believing it". That is what the last two columns test, and the answer at the bottom of
// the table is a qualified yes with one honest exception.

use ferrotherm::certify::Finding;
use ferrotherm::gibbs::Sampler;
use ferrotherm::graph::{Graph, GraphBuilder};
use ferrotherm::rng::Pcg;
use ferrotherm::samples::{enumerate, Plan};

const DRAWS: usize = 20_000;
const BURN: usize = 2_000;
const SEEDS: u64 = 24;

/// A random +-1 glass on a circulant graph: frustrated, and small enough to enumerate.
fn glass(n: usize, seed: u64, reach: usize) -> Graph {
    let mut r = Pcg::new(seed, 7);
    let mut gb = GraphBuilder::new(n);
    for i in 0..n {
        for k in 1..=reach {
            let j = (i + k) % n;
            if i < j {
                gb.couple(i, j, if r.f64() < 0.5 { 1.0 } else { -1.0 });
            }
        }
    }
    gb.build()
}

fn main() {
    let models: Vec<(&str, Graph)> = vec![
        ("ring12", ferrotherm::ising::ring(12, 1.0, 0.0)),
        ("glass14", glass(14, 3, 2)),
        ("glass16", glass(16, 5, 3)),
    ];
    println!(
        "{} chains of {} draws after {} burn-in, every site, against exact enumeration\n",
        SEEDS, DRAWS, BURN
    );
    println!(
        "{:>8} {:>5} {:>9} {:>10} {:>8} {:>12} {:>14}",
        "model", "beta", "tau_int", "corrected", "naive", "mixed ok", "corrected|ok"
    );

    for (name, g) in &models {
        for beta in [0.3f64, 0.5, 0.8, 1.2] {
            let exact = enumerate(g, beta).expect("every model here enumerates");
            let truth: Vec<f64> =
                (0..g.n).map(|i| exact.mean_spin(i).expect("enumeration is exact").value).collect();

            let (mut hit, mut naive_hit, mut total) = (0usize, 0usize, 0usize);
            let (mut mixed, mut hit_mixed, mut total_mixed) = (0usize, 0usize, 0usize);
            let mut taus = Vec::new();

            for seed in 0..SEEDS {
                let mut smp = Sampler::new(g, beta, seed * 7919 + 1);
                let set = smp.collect(&Plan::new(BURN, DRAWS, 1), None);
                let cert = set.certificate(g).expect("collect returns a chain");
                // Only the Undermixed finding, deliberately. A certificate on a 16-spin model at
                // 20,000 draws also reports TooFewSamples -- the TV noise floor for a 65,536-state
                // space needs far more draws than a MARGINAL does -- and that finding is about a
                // different question than the one this column asks.
                let mixed_ok = !cert.findings.iter().any(|f| matches!(f, Finding::Undermixed { .. }));
                if mixed_ok {
                    mixed += 1;
                }
                taus.push(set.chain_tau());

                for i in 0..g.n {
                    let e = set.mean_spin(i).expect("a chain is distributional");
                    // Undo exactly the autocorrelation factor and nothing else.
                    let naive_se = e.stderr * (e.ess / set.len() as f64).sqrt();
                    if e.covers(truth[i]) {
                        hit += 1;
                        if mixed_ok {
                            hit_mixed += 1;
                        }
                    }
                    if (e.value - truth[i]).abs() <= 1.96 * naive_se {
                        naive_hit += 1;
                    }
                    if mixed_ok {
                        total_mixed += 1;
                    }
                    total += 1;
                }
            }

            let pct = |a: usize, b: usize| {
                if b == 0 {
                    "--".to_string()
                } else {
                    format!("{:.1}%", 100.0 * a as f64 / b as f64)
                }
            };
            println!(
                "{name:>8} {beta:>5.1} {:>9.1} {:>10} {:>8} {:>12} {:>14}",
                taus.iter().sum::<f64>() / taus.len() as f64,
                pct(hit, total),
                pct(naive_hit, total),
                format!("{mixed}/{SEEDS}"),
                pct(hit_mixed, total_mixed)
            );
        }
    }

    println!(
        "\nWHAT THE TABLE SAYS.\n\n\
         THE CORRECTION IS NOT COSMETIC. Read the two coverage columns against each other, not\n\
         against 95%. They are built from the SAME estimate and differ only by sqrt(2 tau), and at\n\
         beta 1.2 on the ring the naive interval contains the exact answer for about one site in\n\
         four while announcing 95%. An interval that is wrong three times out of four is not a\n\
         conservative interval; it is a wrong number with a decoration.\n\n\
         THE CORRECTED INTERVAL IS CONSERVATIVE, ON PURPOSE. Several rows read 100%, which is\n\
         over-coverage: intervals wider than they need to be. That is the direction chosen. Each\n\
         estimate is deflated by the SLOWEST autocorrelation the chain showed, not by the site's\n\
         own -- see `SampleSet::chain_tau` -- because a single site sitting in a metastable mode\n\
         reports a fast-looking trace while the mode that decides the answer never moves.\n\n\
         AND THE HONEST EXCEPTION, which is the reason this example prints the last two columns\n\
         rather than stopping at the first two. Where tau reaches the hundreds, tau is ITSELF an\n\
         estimate from a chain barely long enough to make it, and a seed that happens to\n\
         under-estimate tau passes the Undermixed check with an interval that is still too narrow.\n\
         Look at glass16 at beta 1.2: some seeds clear the check, and coverage among exactly those\n\
         seeds is well under 95%. So the rule this measurement supports is narrower than 'the\n\
         correction fixes it':\n\n\
         The corrected interval is calibrated wherever the chain is long against its own tau, and\n\
         the certificate's Undermixed finding is a good but not sufficient test of that. Where tau\n\
         runs to hundreds, no interval computed from an estimated tau is trustworthy, and the\n\
         answer is a longer chain or a better move -- not a wider bar."
    );
}
