// Does adapting the ladder help, and is a second tempering axis worth its replicas?
//
// `tempering::parallel_tempering` has reported `swap_rates` since the beginning -- the acceptance
// of each adjacent pair, which is what decides whether a ladder is a ladder or eight independent
// anneals. Nothing ever acted on it. `adaptive::adapt` closes that loop by respacing the interior
// betas so every pair accepts alike, and `adaptive::adapt_2d` adds a second axis that scales the
// COUPLINGS while leaving the fields alone.
//
// Both are claims, and this measures them, on a MATCHED REPLICA BUDGET. A 2D grid of 4x2 has the
// same eight replicas as a 1D ladder of eight, so the comparison is of arrangements rather than of
// resources -- which is the only version of the question anyone should care about.
//
// THREE FAMILIES, chosen so the second axis has somewhere to succeed and somewhere to fail:
//
//   ferromagnet   2D lattice, uniform J, no fields. Easy, and every method should find the ground
//                 state; it is here to catch a method that is broken rather than to rank one.
//   glass         +/-1 couplings, no fields. Hard, and the ordinary reason to reach for tempering.
//   glass+fields  the same glass with strong random fields. THIS is where scaling the couplings is
//                 supposed to differ from raising beta: warming enough to cross a coupling barrier
//                 also erases the field that says which side to land on, and scaling the couplings
//                 does not.
//
// run: cargo run --release --example adaptive_ladder

use ferrotherm::adaptive::{self, Params};
use ferrotherm::graph::{Graph, GraphBuilder};
use ferrotherm::rng::Pcg;
use ferrotherm::tempering;

const SEEDS: u64 = 6;
const REPLICAS: usize = 8;
const ROUNDS: usize = 900;
const SWAP_EVERY: usize = 2;
const BETA_MIN: f64 = 0.02;
const BETA_MAX: f64 = 6.0;

fn instance(family: usize, l: usize, seed: u64) -> Graph {
    let mut rng = Pcg::new(seed, 0x0AD0_1ADD);
    let mut b = GraphBuilder::new(l * l);
    for y in 0..l {
        for x in 0..l {
            let i = y * l + x;
            let (r, d) = match family {
                0 => (1.0, 1.0),
                _ => (
                    if rng.f64() < 0.5 { 1.0 } else { -1.0 },
                    if rng.f64() < 0.5 { 1.0 } else { -1.0 },
                ),
            };
            b.couple(i, y * l + (x + 1) % l, r);
            b.couple(i, ((y + 1) % l) * l + x, d);
        }
    }
    if family == 2 {
        for i in 0..l * l {
            b.bias(i, if rng.f64() < 0.5 { 1.5 } else { -1.5 });
        }
    }
    b.build()
}

fn stat(v: &[f64]) -> (f64, f64) {
    let m = v.iter().sum::<f64>() / v.len() as f64;
    let s = (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64).sqrt();
    (m, s)
}

fn main() {
    let names = ["ferromagnet", "glass", "glass+fields"];
    println!("Adapting a tempering ladder, and a second axis, on a matched replica budget\n");
    println!(
        "{REPLICAS} replicas everywhere: the 1D arms are a ladder of {REPLICAS}, the 2D arm is a\n\
         4x2 grid over (beta, coupling scale). {ROUNDS} swap rounds, {SWAP_EVERY} sweeps between\n\
         attempts, mean over {SEEDS} seeds. Lower energy is better; lower spread is a healthier\n\
         ladder.\n"
    );
    println!(
        "{:>14} {:>6} {:>16} {:>16} {:>16} {:>9}",
        "family", "spins", "geometric", "adapted", "2D (beta,scale)", "spread"
    );

    for family in 0..3 {
        let l = 16usize;
        let (mut geo, mut ada, mut two) = (Vec::new(), Vec::new(), Vec::new());
        let (mut spread0, mut spread1) = (Vec::new(), Vec::new());

        for seed in 0..SEEDS {
            let g = instance(family, l, seed);

            // A plain geometric ladder, the same total swap rounds.
            let betas = tempering::geometric_ladder(BETA_MIN, BETA_MAX, REPLICAS);
            let base = tempering::parallel_tempering(&g, &betas, ROUNDS, SWAP_EVERY, seed, None);
            geo.push(base.best_e);
            let hi = base.swap_rates.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let lo = base.swap_rates.iter().cloned().fold(f64::INFINITY, f64::min);
            spread0.push(hi - lo);

            // Adapted, with the SAME total rounds split across epochs, so adaptation is not
            // simply being handed more sampling than the arm it is compared with.
            const EPOCHS: usize = 6;
            let p = Params {
                replicas: REPLICAS,
                epochs: EPOCHS,
                rounds: ROUNDS / EPOCHS,
                swap_every: SWAP_EVERY,
                beta_min: BETA_MIN,
                beta_max: BETA_MAX,
            };
            let out = adaptive::adapt(&g, &p, seed);
            ada.push(out.best_e);
            spread1.push(*out.spread.last().unwrap());

            // 4 betas x 2 scales = the same 8 replicas. `1.0` must be present or the answer is
            // about a different model, and the library refuses a grid without it.
            let b4 = tempering::geometric_ladder(BETA_MIN, BETA_MAX, 4);
            let out2 = adaptive::adapt_2d(&g, &b4, &[0.5, 1.0], ROUNDS, SWAP_EVERY, seed).unwrap();
            two.push(out2.best_e);
        }

        let (gm, gs) = stat(&geo);
        let (am, asd) = stat(&ada);
        let (tm, ts) = stat(&two);
        let (s0, _) = stat(&spread0);
        let (s1, _) = stat(&spread1);
        println!(
            "{:>14} {:>6} {gm:>10.1}+-{gs:<4.1} {am:>10.1}+-{asd:<4.1} {tm:>10.1}+-{ts:<4.1} {:>4.2}->{:<4.2}",
            names[family],
            l * l,
            s0,
            s1
        );
    }

    // A SECOND TABLE, because the first one answers only half the question. Above, every ladder
    // was long enough that no pair was truly dead, and an evener ladder found nothing lower. The
    // case adaptation is FOR is a ladder that is actually broken, and the honest way to find out
    // whether it helps there is to break one on purpose.
    println!(
        "\n\nAnd now a ladder that is genuinely broken: HALF THE REPLICAS over the same range,\n\
         so an adjacent pair is far enough apart to stop swapping altogether.\n"
    );
    println!(
        "{:>14} {:>6} {:>16} {:>16} {:>10} {:>12}",
        "family", "replicas", "geometric", "adapted", "spread", "worst pair"
    );
    for family in 0..3 {
        let l = 16usize;
        let (mut geo, mut ada) = (Vec::new(), Vec::new());
        let (mut s0, mut s1) = (Vec::new(), Vec::new());
        let (mut w0, mut w1) = (Vec::new(), Vec::new());
        for seed in 0..SEEDS {
            let g = instance(family, l, seed);
            let betas = tempering::geometric_ladder(BETA_MIN, BETA_MAX, 4);
            let base = tempering::parallel_tempering(&g, &betas, ROUNDS, SWAP_EVERY, seed, None);
            geo.push(base.best_e);
            let hi = base.swap_rates.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let lo = base.swap_rates.iter().cloned().fold(f64::INFINITY, f64::min);
            s0.push(hi - lo);
            w0.push(lo);

            const EPOCHS: usize = 6;
            let p = Params {
                replicas: 4,
                epochs: EPOCHS,
                rounds: ROUNDS / EPOCHS,
                swap_every: SWAP_EVERY,
                beta_min: BETA_MIN,
                beta_max: BETA_MAX,
            };
            let out = adaptive::adapt(&g, &p, seed);
            ada.push(out.best_e);
            s1.push(*out.spread.last().unwrap());
            w1.push(out.swap_rates.iter().cloned().fold(f64::INFINITY, f64::min));
        }
        let (gm, gs) = stat(&geo);
        let (am, asd) = stat(&ada);
        println!(
            "{:>14} {:>6} {gm:>10.1}+-{gs:<4.1} {am:>10.1}+-{asd:<4.1} {:>4.2}->{:<4.2} {:>5.3}->{:<5.3}",
            names[family], 4, stat(&s0).0, stat(&s1).0, stat(&w0).0, stat(&w1).0
        );
    }

    println!(
        "\n\nTHE MECHANISM WORKS AND THE PAYOFF IS NOT THERE. Two claims, and only the first\n\
         survives. Acceptance spread falls on every family and by a lot -- 0.99 to 0.47, 0.77 to\n\
         0.22, 0.56 to 0.18 -- so respacing does exactly what it says: it moves the interior betas\n\
         until every adjacent pair accepts alike. And the energies do not move. On the glass,\n\
         -360.3 against -360.3, inside a between-seed spread of 6 to 8. An evener ladder found\n\
         nothing lower, and printing the spread column without this sentence would let a reader\n\
         assume it had."
    );
    println!(
        "\nTHE SECOND TABLE FOUND SOMETHING BETTER THAN WHAT IT WAS LOOKING FOR. It was built to\n\
         show adaptation rescuing a dead pair. It does not: at four replicas over this range the\n\
         worst pair is 0.000 before AND after, because no respacing of two interior betas can make\n\
         a range this wide crossable -- the ladder is too short for the question, and that is not a\n\
         placement problem. What it shows instead is a trap:\n\n\
         \x20   THE SPREAD FELL ANYWAY. 0.07 to 0.01 on the glass, 0.75 to 0.12 on the ferromagnet.\n\n\
         A spread near zero means every pair accepts ALIKE. It does not mean every pair accepts.\n\
         All-dead is perfectly even, and it scores better on that column than a healthy ladder with\n\
         one weak link. Read the spread WITH the worst-pair rate or it will tell you a broken\n\
         ladder is a good one -- which is the same shape of error as a frozen chain returning a\n\
         small autocorrelation time, and this repository has now met it twice."
    );
    println!(
        "\nSO WHAT ADAPTATION IS. On these families it is a diagnostic that can now fix itself\n\
         rather than a faster optimiser: it removes the need to hand-tune a ladder that was already\n\
         working, and it cannot save one whose range needs more replicas than it was given. That is\n\
         worth having and it is less than the module's own docstring would have implied without\n\
         this table."
    );
    println!(
        "\nTHE SECOND AXIS DID NOT EARN ITS REPLICAS, AND THE HYPOTHESIS THAT SAID IT WOULD WAS\n\
         MINE. The argument was that scaling couplings while leaving fields intact is a different\n\
         move from warming, so a model with strong fields should be where it pays -- warming enough\n\
         to cross a coupling barrier also erases the field that says which side to land on. On\n\
         glass+fields the 2D arm came back at -502.3 against the 1D -505.3, which is WORSE, inside\n\
         a between-seed spread of 13 either way. It is not better anywhere in this table. Four\n\
         betas over the same range means each gap is wider and accepts less, and the coupling axis\n\
         did not buy back what the shorter ladder cost. `adapt_2d` stays because the move is\n\
         correct and the refusal it carries -- a grid that never visits scale 1.0 is answering\n\
         about a different model -- is worth having. It is not a recommendation."
    );
    println!(
        "\nTHE ENDS DO NOT MOVE, which is what keeps this comparison fair. Adaptation respaces the\n\
         interior only: a method that quietly widened the hot end would buy acceptance by answering\n\
         an easier question, and no column here would show it."
    );
}
