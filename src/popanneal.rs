//! Population annealing: a **sequential Monte Carlo** annealer that reports how much to believe it.
//!
//! Simulated annealing runs one chain down a temperature ladder and hands back the best state it
//! saw. Whether that state is the ground state, and whether the chain was ever in equilibrium, are
//! questions its output cannot answer. Population annealing (Hukushima–Iba 2003, Machta 2010) runs
//! `R` chains down the same ladder and **resamples** them at each step, so replicas that landed in
//! low-energy regions are copied and replicas that did not are culled. Two things fall out that a
//! single chain cannot produce:
//!
//! * **The partition function.** Each resampling step's normalisation `Q_k` is an unbiased
//!   estimator of `Z(β_k)/Z(β_{k−1})`, so the whole ladder telescopes into `ln Z`. Starting at
//!   `β = 0`, where `Z = 2ⁿ` exactly, makes it an absolute free energy rather than a ratio.
//! * **A diagnostic that can say "do not trust this run."** [`Outcome::rho`] is the family-size
//!   statistic `ρ_t = (Σ_f n_f²)/R`, where `n_f` counts the descendants of ancestor `f`. It is
//!   exactly `1` when every ancestor still has one descendant and exactly `R` when the population
//!   has collapsed onto one. A run whose `ρ` spiked has explored one basin with `R` copies of the
//!   same history, and its `ln Z` is worth nothing — which is a thing you can only learn from a
//!   method that tracks lineage.
//!
//! # Overflow is not a detail here
//!
//! The reweighting factor is `exp(−Δβ·E_i)`, and on the instances this crate targets that argument
//! is large: G1 has energies near `−2·10⁴`, so a ladder step of `Δβ = 0.03` gives `exp(600)`, and
//! `f64` overflows at `exp(709.78)`. Computed directly, `Q` becomes `inf`, every `τ_i` becomes
//! `NaN`, and the population silently dies. Every exponential here is therefore shifted by the
//! running maximum first — `ln Q = m + ln((1/R)Σexp(−Δβ E_i − m))` — which is exact in the ratios
//! because the shift cancels.
//!
//! # What it does not claim
//!
//! `ln Z` from a finite population is biased **low**, by `O(1/R)`. That direction is stated rather
//! than corrected: the estimator's bias is a theorem about the mean of a product of ratios, and the
//! honest report is the number, the population it came from, and `ρ`. [`Outcome::ln_z_is_absolute`]
//! is false when the ladder did not start at `β = 0`, in which case `ln_z` is `ln(Z(β_end)/Z(β_0))`
//! and nothing more.

use crate::graph::Graph;
use crate::rng::Pcg;

/// How to run the annealer.
#[derive(Clone, Debug, PartialEq)]
pub struct Params {
    /// Target population `R`. Held roughly constant: resampling normalises back to it each step.
    pub population: usize,
    /// Chromatic Gibbs sweeps applied to every replica after each resampling.
    pub sweeps: usize,
    /// The inverse-temperature ladder, in ascending order.
    ///
    /// Starting at `0.0` is what makes [`Outcome::ln_z`] an absolute free energy: `Z(0) = 2ⁿ` for
    /// every graph, with no sampling involved.
    pub betas: Vec<f64>,
}

impl Params {
    /// A linear ladder from `β = 0` to `beta_max` in `stages` steps.
    ///
    /// Linear, not geometric, and that is forced: a geometric ladder cannot contain zero, and
    /// without `β = 0` the free energy is only known up to an unmeasured constant.
    pub fn linear_from_zero(population: usize, sweeps: usize, beta_max: f64, stages: usize) -> Params {
        let stages = stages.max(1);
        let betas = (0..=stages).map(|i| beta_max * i as f64 / stages as f64).collect();
        Params { population, sweeps, betas }
    }
}

/// What the run produced, and what it is worth.
#[derive(Clone, Debug)]
pub struct Outcome {
    /// The lowest-energy state seen anywhere in the population, at any temperature.
    pub state: Vec<i8>,
    /// Its energy, recomputed from `state` rather than accumulated.
    pub energy: f64,
    /// `ln Z(β_end)`, or `ln(Z(β_end)/Z(β_0))` when the ladder did not start at zero.
    pub ln_z: f64,
    /// Whether [`Outcome::ln_z`] is an absolute free energy.
    pub ln_z_is_absolute: bool,
    /// The family statistic after each resampling. `1.0` is ideal; `population` is total collapse.
    pub rho: Vec<f64>,
    /// The worst `ρ` over the ladder — the single number that says whether to believe `ln_z`.
    pub rho_max: f64,
    /// Population size after each resampling. Fluctuates by `O(√R)` around the target.
    pub sizes: Vec<usize>,
}

impl Outcome {
    /// The free energy per spin at the final `β`, or `None` when `β` is zero or `ln_z` is relative.
    pub fn free_energy_per_spin(&self, beta_end: f64, n: usize) -> Option<f64> {
        (self.ln_z_is_absolute && beta_end > 0.0 && n > 0)
            .then(|| -self.ln_z / (beta_end * n as f64))
    }
}

/// Run population annealing.
///
/// Deterministic in `seed`. Returns immediately with an empty outcome for an empty graph or an
/// empty ladder, rather than dividing by a population of zero.
pub fn run(g: &Graph, p: &Params, seed: u64) -> Outcome {
    let n = g.n;
    let r_target = p.population.max(1);
    if n == 0 || p.betas.is_empty() {
        return Outcome {
            state: vec![0i8; n],
            energy: 0.0,
            ln_z: 0.0,
            ln_z_is_absolute: false,
            rho: Vec::new(),
            rho_max: 1.0,
            sizes: Vec::new(),
        };
    }
    let mut rng = Pcg::new(seed, 0x9A_11E4);
    let mut smp = crate::gibbs::Sampler::new(g, p.betas[0], seed ^ 0x9E37_79B9);

    // The population, its lineage, and its energies. Energies are carried rather than recomputed
    // per step: `Graph::energy` is O(edges), and the reweighting needs one per replica per step.
    let mut pop: Vec<Vec<i8>> = Vec::with_capacity(r_target);
    let mut fam: Vec<u32> = Vec::with_capacity(r_target);
    for i in 0..r_target {
        pop.push((0..n).map(|_| rng.spin(0.5)).collect());
        fam.push(i as u32);
    }
    let mut energy: Vec<f64> = pop.iter().map(|s| g.energy(s)).collect();

    let mut best_i = 0usize;
    for i in 1..energy.len() {
        if energy[i] < energy[best_i] {
            best_i = i;
        }
    }
    let mut best_state = pop[best_i].clone();
    let mut best_energy = energy[best_i];

    // `Z(0) = 2^n` exactly: at infinite temperature every one of the 2^n states has weight 1. This
    // is the whole reason the ladder starts at zero.
    let absolute = p.betas[0] == 0.0;
    let mut ln_z = if absolute { n as f64 * core::f64::consts::LN_2 } else { 0.0 };

    // Equilibrate at the first rung before any reweighting, so step 1 resamples states that belong
    // to `betas[0]` rather than to the uniform draw.
    smp.beta = p.betas[0];
    for i in 0..pop.len() {
        smp.s.copy_from_slice(&pop[i]);
        smp.sweeps(p.sweeps, None);
        pop[i].copy_from_slice(&smp.s);
        energy[i] = g.energy(&pop[i]);
        if energy[i] < best_energy {
            best_energy = energy[i];
            best_state.copy_from_slice(&pop[i]);
        }
    }

    let mut rho = Vec::with_capacity(p.betas.len());
    let mut sizes = Vec::with_capacity(p.betas.len());

    for k in 1..p.betas.len() {
        let d_beta = p.betas[k] - p.betas[k - 1];
        let r_now = pop.len();
        if r_now == 0 {
            break;
        }

        // --- reweight, shifted -----------------------------------------------------------------
        //
        // `x_i = -Δβ·E_i` can be several hundred on a G-set instance; `exp` of it is `inf`. The
        // shift by the running maximum is exact in every quantity used below, because `Q` and the
        // `τ_i` that divide by it are shifted by the same constant.
        let x: Vec<f64> = energy.iter().map(|e| -d_beta * e).collect();
        let m = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        if !m.is_finite() {
            break;
        }
        let ex: Vec<f64> = x.iter().map(|v| (v - m).exp()).collect();
        let sum_ex: f64 = ex.iter().sum();
        if !(sum_ex > 0.0) || !sum_ex.is_finite() {
            break;
        }
        // ln(Z_k / Z_{k-1}) = ln( (1/R) Σ exp(-Δβ E_i) ), evaluated through the shift.
        ln_z += m + (sum_ex / r_now as f64).ln();

        // --- resample --------------------------------------------------------------------------
        //
        // Expected copies of replica i, normalised so the population returns to its target:
        //   τ_i = (R_target / R_now) · exp(-Δβ E_i) / Q  =  R_target · exp(x_i - m) / Σ exp(x - m)
        // Integer copies by systematic rounding: floor plus a Bernoulli on the fraction, which is
        // unbiased in expectation and keeps `Σ n_i` within O(√R) of the target.
        let scale = r_target as f64 / sum_ex;
        let mut next: Vec<Vec<i8>> = Vec::with_capacity(r_target);
        let mut next_fam: Vec<u32> = Vec::with_capacity(r_target);
        let mut next_e: Vec<f64> = Vec::with_capacity(r_target);
        for i in 0..r_now {
            let tau = ex[i] * scale;
            let mut copies = tau.floor();
            if rng.f64() < tau - copies {
                copies += 1.0;
            }
            // A single replica may not eat the whole allocation: `tau` is bounded by R_target, and
            // an unbounded `copies` here would be a silent memory blowup on a degenerate step.
            let copies = (copies as usize).min(r_target * 4);
            for _ in 0..copies {
                next.push(pop[i].clone());
                next_fam.push(fam[i]);
                next_e.push(energy[i]);
            }
        }
        if next.is_empty() {
            // Everything rounded to zero. Keep the single best replica rather than returning a
            // population of nothing: the run is already suspect, and `rho` will say so.
            let mut bi = 0usize;
            for i in 1..r_now {
                if energy[i] < energy[bi] {
                    bi = i;
                }
            }
            next.push(pop[bi].clone());
            next_fam.push(fam[bi]);
            next_e.push(energy[bi]);
        }
        pop = next;
        fam = next_fam;
        energy = next_e;

        // --- the diagnostic --------------------------------------------------------------------
        //
        // ρ = (Σ_f n_f²)/R over ancestors f. One descendant each gives exactly 1; one ancestor
        // owning the whole population gives exactly R. It is computed BEFORE the sweeps, because it
        // describes the resampling that just happened.
        let mut counts = vec![0u32; r_target];
        for &f in &fam {
            let idx = f as usize;
            if idx < counts.len() {
                counts[idx] += 1;
            }
        }
        let sq: f64 = counts.iter().map(|&c| (c as f64) * (c as f64)).sum();
        rho.push(sq / pop.len() as f64);
        sizes.push(pop.len());

        // --- equilibrate at the new rung -------------------------------------------------------
        smp.beta = p.betas[k];
        for i in 0..pop.len() {
            smp.s.copy_from_slice(&pop[i]);
            smp.sweeps(p.sweeps, None);
            pop[i].copy_from_slice(&smp.s);
            energy[i] = g.energy(&pop[i]);
            if energy[i] < best_energy {
                best_energy = energy[i];
                best_state.copy_from_slice(&pop[i]);
            }
        }
    }

    let rho_max = rho.iter().cloned().fold(1.0f64, f64::max);
    // Recomputed from the state, not carried: the one number a caller acts on should not depend on
    // an accumulator being right.
    let energy = g.energy(&best_state);
    Outcome { state: best_state, energy, ln_z, ln_z_is_absolute: absolute, rho, rho_max, sizes }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::GraphBuilder;

    fn random_graph(n: usize, p: f64, seed: u64, fields: bool) -> Graph {
        let mut rng = Pcg::new(seed, 0xC0FFEE);
        let mut gb = GraphBuilder::new(n);
        for i in 0..n {
            if fields {
                gb.bias(i, rng.f64() * 2.0 - 1.0);
            }
            for j in (i + 1)..n {
                if rng.f64() < p {
                    gb.couple(i, j, rng.f64() * 2.0 - 1.0);
                }
            }
        }
        gb.build()
    }

    /// `ln Z` and the minimum, by enumeration. Only usable up to about 20 spins.
    fn brute(g: &Graph, beta: f64) -> (f64, f64) {
        let n = g.n;
        let mut s = vec![1i8; n];
        let mut min = f64::INFINITY;
        let mut xs = Vec::with_capacity(1usize << n);
        for mask in 0..(1u64 << n) {
            for i in 0..n {
                s[i] = if mask >> i & 1 == 1 { 1 } else { -1 };
            }
            let e = g.energy(&s);
            min = min.min(e);
            xs.push(-beta * e);
        }
        let m = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let ln_z = m + xs.iter().map(|x| (x - m).exp()).sum::<f64>().ln();
        (ln_z, min)
    }

    /// A graph with no edges and no fields makes every step **exact**, so this asserts equality.
    ///
    /// Every state has energy zero, so `Z(β) = 2ⁿ` at every temperature and every reweighting
    /// factor is exactly 1. That drives the arithmetic down a path with no rounding anywhere:
    /// `Σexp = R` exactly, `ln(R/R) = 0` exactly, `τ_i = 1` exactly, so no Bernoulli is drawn and
    /// no family is ever duplicated. Anything that perturbs the reweighting — a missing
    /// normalisation, a shift applied to one side only, an off-by-one in the population target —
    /// moves at least one of these off its exact value.
    #[test]
    fn a_flat_landscape_is_reproduced_exactly() {
        let g = GraphBuilder::new(9).build();
        let p = Params::linear_from_zero(64, 2, 4.0, 12);
        let o = run(&g, &p, 5);
        assert_eq!(o.ln_z, 9.0 * core::f64::consts::LN_2, "ln Z must be exactly n ln 2");
        assert!(o.ln_z_is_absolute);
        assert_eq!(o.rho, vec![1.0; 12], "no replica may be copied when all weights are equal");
        assert_eq!(o.sizes, vec![64; 12], "the population is preserved exactly");
        assert_eq!(o.energy, 0.0);
    }

    /// The free energy of a small graph, against enumeration.
    ///
    /// The estimator is biased **low** by `O(1/R)`, so the tolerance is one-sided in spirit; it is
    /// written two-sided anyway, because a bound that only fails in one direction would not catch a
    /// reweighting that over-counts.
    #[test]
    fn ln_z_matches_exact_enumeration_on_a_small_graph() {
        for seed in 0..3u64 {
            let g = random_graph(8, 0.5, seed, seed % 2 == 0);
            let beta_end = 1.5;
            let (exact, _) = brute(&g, beta_end);
            let p = Params::linear_from_zero(3000, 4, beta_end, 30);
            let o = run(&g, &p, 100 + seed);
            let err = (o.ln_z - exact).abs();
            assert!(
                err < 0.15,
                "seed {seed}: ln Z estimate {:.4} vs exact {exact:.4} (err {err:.4}), rho_max {:.1}",
                o.ln_z,
                o.rho_max
            );
        }
    }

    /// The ladder must survive energies whose reweighting factor overflows `f64`.
    ///
    /// This is the test the shift exists for. With `|E|` of order `4·10⁴` and `Δβ = 0.05`, the
    /// unshifted factor is `exp(2000)`, which is `inf`; `Q` becomes `inf`, every `τ` becomes `NaN`,
    /// and the loop breaks out with a truncated ladder. Asserting the ladder RAN TO THE END is what
    /// makes that visible — a finite `ln_z` alone would pass on a run that gave up at step one.
    #[test]
    fn a_reweighting_factor_that_would_overflow_does_not_end_the_run() {
        let n = 200;
        let mut gb = GraphBuilder::new(n);
        for i in 0..n {
            gb.couple(i, (i + 1) % n, 100.0);
            gb.couple(i, (i + 7) % n, -100.0);
        }
        let g = gb.build();
        let stages = 20;
        let p = Params::linear_from_zero(64, 1, 1.0, stages);
        let o = run(&g, &p, 3);
        assert_eq!(o.rho.len(), stages, "the ladder stopped early: overflow was not handled");
        assert!(o.ln_z.is_finite(), "ln Z = {}", o.ln_z);
        assert!(o.energy.is_finite() && o.energy < 0.0);
    }

    /// A ladder that jumps straight to low temperature collapses the population, and `ρ` says so.
    ///
    /// The diagnostic is the point of the method, so it has to be shown FAILING, not only passing.
    /// One step from `β = 0` to `β = 40` puts essentially all the weight on whichever random
    /// replica happened to be lowest, and every survivor descends from it.
    #[test]
    fn rho_reports_a_collapsed_population() {
        let g = random_graph(30, 0.3, 11, false);
        let r = 128;
        let p = Params { population: r, sweeps: 1, betas: vec![0.0, 40.0] };
        let o = run(&g, &p, 9);
        assert_eq!(o.rho.len(), 1);
        assert!(
            o.rho_max > 0.5 * r as f64,
            "rho {:.1} on a one-step quench of {r} replicas -- expected near-total collapse",
            o.rho_max
        );
        // And the healthy case, for contrast: the same graph on a gentle ladder stays near 1.
        let gentle = Params::linear_from_zero(r, 4, 4.0, 60);
        let o2 = run(&g, &gentle, 9);
        assert!(o2.rho_max < 0.25 * r as f64, "gentle ladder rho_max {:.1}", o2.rho_max);
    }

    /// Population annealing has to actually find the ground state of something small.
    #[test]
    fn it_reaches_the_true_minimum_on_an_enumerable_graph() {
        for seed in 0..4u64 {
            let g = random_graph(14, 0.45, seed, true);
            let (_, min) = brute(&g, 1.0);
            let p = Params::linear_from_zero(400, 4, 6.0, 40);
            let o = run(&g, &p, 200 + seed);
            assert!(
                o.energy <= min + 1e-9,
                "seed {seed}: found {:.6}, true minimum {min:.6}",
                o.energy
            );
            assert_eq!(o.energy, g.energy(&o.state), "energy must match the state returned");
        }
    }

    /// A ladder that does not start at zero yields a RATIO, and says so rather than pretending.
    #[test]
    fn a_ladder_that_skips_infinite_temperature_reports_a_relative_free_energy() {
        let g = random_graph(8, 0.5, 4, false);
        let p = Params { population: 200, sweeps: 2, betas: vec![0.2, 0.6, 1.0] };
        let o = run(&g, &p, 4);
        assert!(!o.ln_z_is_absolute);
        assert!(o.free_energy_per_spin(1.0, g.n).is_none(), "a ratio is not a free energy");
        // The ratio itself is still meaningful, and must be positive: Z grows as beta rises.
        assert!(o.ln_z > 0.0, "ln(Z(1.0)/Z(0.2)) = {}", o.ln_z);
    }

    /// Degenerate inputs return rather than dividing by an empty population.
    #[test]
    fn an_empty_graph_or_ladder_returns() {
        let g = GraphBuilder::new(0).build();
        let o = run(&g, &Params::linear_from_zero(10, 1, 1.0, 3), 1);
        assert!(o.state.is_empty() && o.rho.is_empty());
        let g2 = random_graph(6, 0.5, 1, false);
        let o2 = run(&g2, &Params { population: 10, sweeps: 1, betas: Vec::new() }, 1);
        assert!(o2.rho.is_empty() && !o2.ln_z_is_absolute);
    }
}
