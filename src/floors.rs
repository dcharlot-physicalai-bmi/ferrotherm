//! The floors every substrate is measured against, and the distance a real run sits above its own.
//!
//! # One axis
//!
//! A computing paradigm is a claim about how you carry a number. There are three ways and they do
//! not cost the same:
//!
//! | carrier | floor per value at `b` bits | growth |
//! |---|---|---|
//! | **digital**, any encoding | `b kT ln 2` — [`crate::precision::digital_bit_energy`] | **linear** |
//! | **analogue** node | `12 kT 4^b` — [`crate::precision::analog_bit_energy`] | **exponential** |
//! | **samples** | **zero**, quasistatically — see below | none |
//!
//! The middle row's exponential is not one field's quirk. This crate derives `4^b` from `kT/C`
//! sampling noise against a quantiser's SNR; the analog compute-in-memory literature derives the
//! same `4^ENOB` from ADC circuit design; and photonic shot noise gives a steeper one still —
//! [`OPTICAL_SHOT_MAC`] records three published points whose ratio is `7.98` per bit. Charge,
//! voltage and photons; three mechanisms, one conclusion. **An analogue representation pays
//! exponentially for precision whatever physically carries it, and a digital one pays linearly.**
//! A number-format argument is an argument about the first row's coefficient
//! ([`crate::logdomain::format_floor`] asserts the encoding cancels), and it is bounded.
//!
//! # The third row, which is the whole bet
//!
//! **Sampling is free in the quasistatic limit, and this is a theorem rather than a hope.** Doty,
//! Kornerup, Luchsinger, Orshansky, Soloveichik & Woods, *Harvesting Brownian Motion: Zero Energy
//! Computational Sampling* (arXiv:2309.06957): a device can perpetually generate samples from **any
//! computable** distribution, driven entirely by Brownian motion, **with zero energy dissipation**.
//! The textbook statement of the same thing: a system's nonequilibrium free energy is
//! `F[p] = <E>_p − kT H(p)`, the least work to drive `p0 → p1` is `F[p1] − F[p0]`, and if the
//! target is the Gibbs law of the machine's *own* Hamiltonian then `ΔF = 0` and the reversible work
//! of drawing that sample is **zero**.
//!
//! The bill arrives somewhere else, and it is worth being exact about where:
//!
//! **The readout register is where Landauer bites.** A sample that is copied into a register which
//! must later be reset costs `kT ln 2` per bit of the sample's entropy — [`readout_floor`]. This
//! crate measured that empirically before it knew it was the theory: 78–98% of the cost of an
//! independent draw on its own fabric is readback, and [`crate::ledger`] has carried the number for
//! months. It is not an artefact of an FPGA. It is where the floor lives.
//!
//! # Two floors that are positive, and which you should actually budget against
//!
//! Zero is the quasistatic answer. Demand a finite time or a finite accuracy and it stops being
//! zero.
//!
//! **Energy–time–accuracy.** Rolandi, Abiuso, Lipka-Bartosik, Aifer & Coles, *Energy-Time-Accuracy
//! Tradeoffs in Thermodynamic Computing* (arXiv:2601.04358, January 2026):
//!
//! ```text
//!   W_diss * tau * (1 - Q)  >=  W^2(rho_0, rho_1) / (beta gamma)
//! ```
//!
//! with `Q` in `[0,1]` a quality factor, `W` the Bures–Wasserstein distance between initial and
//! target, and `gamma` the Langevin friction. [`eddp_floor`] evaluates it. In the slow limit the
//! dissipated work vanishes like `1/tau` but **the product does not** — energy, time and accuracy
//! trade against one another and you cannot have all three. [`bures_wasserstein_diagonal`] computes
//! the distance for the case this crate can verify.
//!
//! **Per effective sample.** From the thermodynamic uncertainty relation (Barato & Seifert 2015;
//! Gingrich, Horowitz, Perunov & England 2016), for a physical Markov sampler estimating an
//! observable `f`:
//!
//! ```text
//!   joules per effective sample  >=  2 kT * (<f> / sigma_f)^2
//! ```
//!
//! [`tur_floor`] is that, and **the sign of it is the most useful fact in this module**: the floor
//! scales with the *squared signal-to-noise ratio of what you are measuring*, not with the number
//! of samples. **You pay for the precision you extract, not for the samples you draw.** A machine
//! that computes by sampling and needs only coarse answers sits in the cheap regime by
//! construction — which is exactly the regime every analogue substrate is priced out of, because
//! it pays `4^b` for precision whether the task needs it or not.
//!
//! This composition — a TUR bound expressed as joules per effective sample — is a derivation from
//! those theorems and not a citation; a sweep of the literature did not locate it stated this way.
//! It is labelled as such at [`tur_floor`], and its caveats are recorded there.
//!
//! # What this module is for
//!
//! [`cost_per_effective_sample`] takes a real run — a [`crate::ledger::Ledger`], a price set, the
//! sweeps performed and the measured integrated autocorrelation time — and reports what that run
//! cost per *independent* sample, beside the floor for the observable it was estimating. The ratio
//! is the headroom: how much a native thermodynamic fabric could in principle win over the machine
//! that produced the number. A digital emulation sits `10^9`–`10^10` above the floor because its
//! cost is set by CMOS rather than by the bath, and saying so with a number is more useful than
//! either claiming the win or ignoring it.

use crate::ledger::{Evidence, Ledger, Prices};
use crate::landauer::BOLTZMANN_CONSTANT;
use crate::round::sum_up;

/// Room temperature, kelvin, as the rest of this crate uses it.
pub const ROOM_TEMPERATURE_K: f64 = 300.0;

/// Published shot-noise floors for one optical multiply-accumulate, in joules, at 4, 7 and 8 bits.
///
/// From Tait, *Quantifying Power in Silicon Photonic Neural Networks*, Phys. Rev. Applied 17,
/// 054029 (2022). Recorded rather than derived: this crate does not re-derive the optical bound,
/// and the three points are here so that the **scaling** can be checked against the other two
/// carriers. Their ratio is `7.98` per bit, steeper than the `4^b` of a capacitive node — see
/// `the_three_carriers_scale_as_measured`.
pub const OPTICAL_SHOT_MAC: [(u32, f64); 3] = [(4, 0.96e-15), (7, 490e-15), (8, 3.9e-12)];

/// The work needed to draw one sample quasistatically from a machine's own Gibbs law: **zero**.
///
/// Present as a named constant because it is a theorem and the whole argument turns on it, and
/// because a zero that is written down is harder to forget than a zero that is merely true.
pub const QUASISTATIC_SAMPLING_WORK: f64 = 0.0;

/// `kT ln 2` per bit of a sample's entropy: what the readout register costs.
///
/// The sampling is free; copying the result into something that must later be reset is not. This is
/// the only unavoidable cost in the quasistatic picture, and it is proportional to the entropy
/// actually extracted rather than to the machine's size.
///
/// # Panics
///
/// If the temperature is not positive or the entropy is negative.
#[must_use]
pub fn readout_floor(entropy_bits: f64, temperature_k: f64) -> f64 {
    assert!(temperature_k > 0.0, "temperature must be positive, got {temperature_k}");
    assert!(entropy_bits >= 0.0, "a sample cannot carry negative entropy, got {entropy_bits}");
    entropy_bits * BOLTZMANN_CONSTANT * temperature_k * core::f64::consts::LN_2
}

/// The thermodynamic-uncertainty floor on joules per **effective** sample: `2 kT (mean/sigma)^2`.
///
/// # Provenance, stated because it matters
///
/// The thermodynamic uncertainty relation itself is a theorem — Barato & Seifert, *PRL* 114,
/// 158101 (2015), proved in general by Gingrich, Horowitz, Perunov & England, *PRL* 116, 120601
/// (2016). Expressing it as *joules per effective sample* by identifying the time-integrated
/// current with an MCMC estimator and substituting `Var = sigma^2 / N_eff` is a **derivation**; a
/// literature sweep did not locate it published in this form. Treat the inequality as sound and the
/// packaging as ours.
///
/// Three caveats travel with it. It is a steady-state statement, so an annealed or burning-in
/// sampler needs the finite-time TUR instead. It floors a **physical** Markov process, so a digital
/// emulation's cost is set by CMOS and sits far above. And it can be loosened under asymmetric or
/// time-dependent driving, so it bounds a classical, time-symmetric sampler.
///
/// # Panics
///
/// If the temperature is not positive.
#[must_use]
pub fn tur_floor(snr: f64, temperature_k: f64) -> f64 {
    assert!(temperature_k > 0.0, "temperature must be positive, got {temperature_k}");
    2.0 * BOLTZMANN_CONSTANT * temperature_k * snr * snr
}

/// The squared Bures–Wasserstein distance between two Gaussians with **diagonal** covariances.
///
/// `|mu0 - mu1|^2 + sum_i (sqrt(s0_i) - sqrt(s1_i))^2`, which is the computable moment lower bound
/// the energy–time–accuracy theorem offers.
///
/// Diagonal only, and refused otherwise rather than approximated: the general case needs the matrix
/// square root of `Σ0^{1/2} Σ1 Σ0^{1/2}`, this crate has no verified eigendecomposition to build it
/// from, and a distance that is quietly wrong would make the floor below it quietly wrong too.
///
/// # Errors
///
/// A message when the lengths disagree or a variance is negative.
pub fn bures_wasserstein_diagonal(
    mu0: &[f64],
    mu1: &[f64],
    s0: &[f64],
    s1: &[f64],
) -> Result<f64, String> {
    if mu0.len() != mu1.len() || s0.len() != s1.len() || mu0.len() != s0.len() {
        return Err(format!(
            "means and variances must share one dimension: {} {} {} {}",
            mu0.len(),
            mu1.len(),
            s0.len(),
            s1.len()
        ));
    }
    let mut terms = Vec::with_capacity(2 * mu0.len());
    for i in 0..mu0.len() {
        if s0[i] < 0.0 || s1[i] < 0.0 {
            return Err(format!("variance {i} is negative: {} {}", s0[i], s1[i]));
        }
        let d = mu0[i] - mu1[i];
        terms.push(d * d);
        let r = s0[i].sqrt() - s1[i].sqrt();
        terms.push(r * r);
    }
    Ok(sum_up(&terms))
}

/// The least dissipated work a drive of duration `tau_drive` can do at quality `quality`.
///
/// `W_diss >= W^2 / (beta gamma tau_drive)`, the fixed-duration corollary of the energy–time–accuracy
/// inequality. The full product form — dissipation times total time times inaccuracy — is what does
/// not vanish in the slow limit; this is the form a caller with a deadline actually needs.
///
/// # Panics
///
/// If `beta`, `gamma` or `tau_drive` is not positive, or `quality` is outside `[0, 1)`.
#[must_use]
pub fn eddp_floor(wasserstein_sq: f64, beta: f64, gamma: f64, tau_drive: f64, quality: f64) -> f64 {
    assert!(beta > 0.0 && beta.is_finite(), "beta must be positive and finite, got {beta}");
    assert!(gamma > 0.0 && gamma.is_finite(), "friction must be positive and finite, got {gamma}");
    assert!(tau_drive > 0.0 && tau_drive.is_finite(), "a drive takes time, got {tau_drive}");
    assert!(
        (0.0..1.0).contains(&quality),
        "a quality factor lives in [0, 1); at 1 the bound says nothing, got {quality}"
    );
    wasserstein_sq / (beta * gamma * tau_drive * (1.0 - quality))
}

/// What one run cost per independent sample, and how far that is above the floor.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SamplerCost {
    /// Independent samples the run actually produced: `sweeps / (2 tau_int)`.
    pub effective_samples: f64,
    /// Joules the run spent, from its ledger and its price set.
    pub joules: f64,
    /// Joules per effective sample.
    pub joules_per_effective_sample: f64,
    /// The thermodynamic floor for the observable being estimated.
    pub floor: f64,
    /// How many times the floor this run cost. **The headroom, and the size of the prize.**
    pub ratio_to_floor: f64,
    /// The weakest evidence anywhere in this figure — the grade the ratio inherits.
    pub evidence: Evidence,
    /// How convergence was established. A cost carrying [`Convergence::SingleChain`] rests on a
    /// diagnostic that cannot detect its own failure; the number is real, the claim is weak.
    pub convergence: Convergence,
}

/// How a run's convergence was established — which [`cost_per_effective_sample`] now requires,
/// because the diagnostic it used to rely on alone cannot report its own failure.
///
/// `tau_int` is computed from one chain's own trace, and a trace is only evidence about the
/// timescales it CONTAINS. A chain that never leaves the basin it started in fluctuates only
/// inside that basin, so its autocorrelation decays fast and `tau_int` comes back **small** —
/// which prices the run as though it bought MANY independent samples, at a LOW cost per sample.
/// The error flatters the run, on the one number this crate exists to report honestly.
///
/// `examples/burnin` measures that: a chain below the critical temperature reports a tight
/// interval around an answer that is wrong by construction, with an autocorrelation time that
/// says it is well mixed. Both instruments named below catch it, and until now neither was wired
/// to the pricing.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Convergence {
    /// Coupling from the past coalesced, so the draw is exactly stationary and the burn-in is
    /// known rather than assumed. See [`crate::cftp`].
    Certified {
        /// Sweeps back coalescence actually required — `cftp::Draw::coalesced_at`.
        coalesced_at: f64,
        /// Sweeps the run actually discarded. Less than `coalesced_at` is a refusal, not a
        /// warning: the draws are from the starting state, not from the model.
        burn_in: f64,
    },
    /// Several chains from dispersed starts agree. See [`crate::rhat`].
    MultiChain {
        /// Split, rank-normalised, folded R-hat. Vehtari, Gelman, Simpson, Carpenter and Bürkner
        /// (*Bayesian Analysis*, 2021) recommend refusing above [`Convergence::RHAT_LIMIT`].
        rhat: f64,
    },
    /// One chain's own autocorrelation, and nothing else.
    ///
    /// Named rather than defaulted, because this is the case that cannot fail loudly. It is
    /// priced, since sometimes it is all a caller has, and it is recorded on the result so a
    /// reader can see what the figure rests on.
    SingleChain,
}

impl Convergence {
    /// The R-hat above which [`Convergence::MultiChain`] is refused (Vehtari et al. 2021).
    pub const RHAT_LIMIT: f64 = 1.01;

    /// The R-hat below which [`Convergence::MultiChain`] is refused — and it is NOT one.
    ///
    /// A sample R-hat is `sqrt((N - 1)/N + B/(N W))` for split chains of `N` draws, so when the
    /// chains agree well (`B -> 0`) it sits slightly BELOW one: four chains of 2,000 draws from
    /// one law give `0.99985`. The first version of this type refused anything under one as "not
    /// real", which refused exactly the best-converged runs, and the test written beside it
    /// checked `1.0` and `0.5` and never a realistic value. What the formula does bound is
    /// `R^2 >= (N - 1)/N`: an R-hat under `0.99` means fewer than about fifty draws per split
    /// chain, which is too little evidence to call agreement — so that is where the floor is.
    pub const RHAT_FLOOR: f64 = 0.99;

    /// From the chains themselves: one observable's trace per chain, started from DISPERSED states.
    ///
    /// This exists so that the strong claim is as easy to make as the weak one. Without it a
    /// caller has to assemble traces, pick a diagnostic and build the variant by hand, while
    /// [`Convergence::SingleChain`] needs nothing — and the path of least resistance would be
    /// the one diagnostic that cannot fail.
    ///
    /// It takes the **largest** of [`crate::rhat`]'s three potential scale reductions — split,
    /// rank-normalised and folded — because each is blind where another sees: plain R-hat passes
    /// chains whose means agree and whose spreads do not, which the folded one catches, and the
    /// rank-normalised one survives tails the plain one does not. If any of the three is `NaN` —
    /// fewer than two chains, fewer than four draws, a constant chain — the result is `NaN` and
    /// [`Convergence::refusal`] refuses it: too little evidence is not agreement. (`f64::max`
    /// silently DROPS a `NaN`, so that case is handled by hand.)
    #[must_use]
    pub fn from_chains(chains: &[Vec<f64>]) -> Convergence {
        let d = crate::rhat::diagnose(chains);
        let all = [d.rhat, d.rhat_rank, d.rhat_folded];
        let mut worst = f64::NEG_INFINITY;
        for r in all {
            if r.is_nan() {
                return Convergence::MultiChain { rhat: f64::NAN };
            }
            if r > worst {
                worst = r;
            }
        }
        Convergence::MultiChain { rhat: worst }
    }

    /// From perfect draws of the same model: the burn-in a chain used, held against the WORST
    /// coalescence time coupling from the past needed across `draws`.
    ///
    /// No draws at all is an infinite requirement, which [`Convergence::refusal`] refuses: an
    /// empty measurement certifies nothing. Note what is certified — [`crate::cftp`] couples the
    /// single-site heat-bath chain, so this bounds a Gibbs sampler's burn-in exactly and is only
    /// a loose upper bound for a cluster sampler, which mixes faster than the chain measured.
    #[must_use]
    pub fn from_perfect_draws(draws: &[crate::cftp::Draw], burn_in: usize) -> Convergence {
        let mut worst: Option<usize> = None;
        for d in draws {
            worst = Some(worst.map_or(d.coalesced_at, |w| w.max(d.coalesced_at)));
        }
        Convergence::Certified {
            coalesced_at: worst.map_or(f64::INFINITY, |w| w as f64),
            burn_in: burn_in as f64,
        }
    }

    /// Why this run must not be priced, or `None` if it may be. The `Option` IS the refusal, so
    /// there is no `Result` here and no error section: `None` is the permission to proceed.
    #[must_use]
    pub fn refusal(&self) -> Option<String> {
        match *self {
            Convergence::Certified { coalesced_at, burn_in } => {
                if !coalesced_at.is_finite() || !burn_in.is_finite() {
                    Some("a coalescence time and a burn-in must both be finite".to_string())
                } else if coalesced_at < 0.0 || burn_in < 0.0 {
                    // Without this a NEGATIVE coalescence time passes the comparison below --
                    // `200 < -5` is false -- so nonsense would read as a certification.
                    Some(format!(
                        "sweep counts cannot be negative: coalesced_at {coalesced_at}, burn_in {burn_in}"
                    ))
                } else if burn_in < coalesced_at {
                    Some(format!(
                        "the burn-in was {burn_in} sweeps and coalescence required {coalesced_at}: \
                         these draws carry the starting state, so there is nothing here to price"
                    ))
                } else {
                    None
                }
            }
            Convergence::MultiChain { rhat } => {
                if !rhat.is_finite() {
                    Some(format!("an R-hat must be finite, got {rhat}"))
                } else if rhat < Convergence::RHAT_FLOOR {
                    Some(format!(
                        "R-hat is {rhat}, below the {} floor: since R^2 >= (N - 1)/N that means \
                         fewer than about fifty draws per split chain, too few to call agreement",
                        Convergence::RHAT_FLOOR
                    ))
                } else if rhat > Convergence::RHAT_LIMIT {
                    Some(format!(
                        "R-hat is {rhat}, above the {} limit: the chains do not agree, so they \
                         are not all sampling the same distribution",
                        Convergence::RHAT_LIMIT
                    ))
                } else {
                    None
                }
            }
            Convergence::SingleChain => None,
        }
    }

    /// Whether convergence was established by an instrument that could have said no.
    #[must_use]
    pub fn is_certified(&self) -> bool {
        !matches!(self, Convergence::SingleChain)
    }
}

/// Price a real sampling run per independent sample, against its own thermodynamic floor.
///
/// `sweeps` is what the chain performed and `tau_int` its measured integrated autocorrelation time
/// ([`crate::certify::tau_int`] or [`crate::autocorr::tau_int_exact`]), so the effective sample
/// count is `sweeps / (2 tau_int)` — the standard definition, and the one that makes the result a
/// per-*independent*-sample cost rather than a per-sweep one.
///
/// The evidence grade is carried through and cannot improve: a metered price against a floor is
/// still only as good as the price.
///
/// # Errors
///
/// A message when the price set cannot price this run, `tau_int` is not positive, or no sweeps were
/// performed — each of which would otherwise produce a number that looks like a measurement.
pub fn cost_per_effective_sample(
    ledger: &Ledger,
    prices: &Prices,
    sweeps: f64,
    tau_int: f64,
    convergence: Convergence,
    snr: f64,
    temperature_k: f64,
) -> Result<SamplerCost, String> {
    if sweeps <= 0.0 {
        return Err("a run of no sweeps produced no samples to price".into());
    }
    if !(tau_int > 0.0) || !tau_int.is_finite() {
        return Err(format!(
            "an integrated autocorrelation time must be positive and finite, got {tau_int}"
        ));
    }
    // CONVERGENCE IS CHECKED BEFORE ANYTHING IS DIVIDED. A chain that did not converge has no
    // effective samples to be cheap per, and `tau_int` is structurally unable to say so: it
    // summarises the timescales the trace contains, and a barrier never crossed contributes none.
    if let Some(why) = convergence.refusal() {
        return Err(why);
    }
    let joules = ledger
        .joules(prices)
        .ok_or_else(|| format!("these prices cannot price this run: {}", prices.source))?;
    let effective_samples = sweeps / (2.0 * tau_int);
    if effective_samples <= 0.0 {
        return Err("that chain produced no effective samples".into());
    }
    let per = joules / effective_samples;
    let floor = tur_floor(snr, temperature_k);
    Ok(SamplerCost {
        effective_samples,
        joules,
        joules_per_effective_sample: per,
        floor,
        ratio_to_floor: if floor > 0.0 { per / floor } else { f64::INFINITY },
        evidence: prices.evidence,
        convergence,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ledger::{KV260_MEASURED, Z1_SPICE};

    /// The three carriers scale as the three literatures say, and the scalings differ.
    ///
    /// This is the axis the module exists to state, so it is asserted rather than described.
    #[test]
    fn the_three_carriers_scale_as_measured() {
        // digital: linear in bits, whatever the encoding
        let d8 = crate::precision::digital_bit_energy(8, ROOM_TEMPERATURE_K);
        let d16 = crate::precision::digital_bit_energy(16, ROOM_TEMPERATURE_K);
        assert!((d16 / d8 - 2.0).abs() < 1e-12, "digital must double with the bits");
        assert_eq!(d8, crate::logdomain::format_floor(8, ROOM_TEMPERATURE_K));

        // analogue: four times per bit
        let a8 = crate::precision::analog_bit_energy(8, ROOM_TEMPERATURE_K);
        let a9 = crate::precision::analog_bit_energy(9, ROOM_TEMPERATURE_K);
        assert!((a9 / a8 - 4.0).abs() < 1e-9, "a capacitive node quadruples per bit");

        // optical: steeper still, and the published points say so
        let [(b0, e0), _, (b2, e2)] = OPTICAL_SHOT_MAC;
        let per_bit = (e2 / e0).powf(1.0 / f64::from(b2 - b0));
        assert!(
            (per_bit - 7.98).abs() < 0.05,
            "the published optical points scale at {per_bit} per bit, not ~7.98"
        );
        assert!(per_bit > 4.0, "optical must be steeper than capacitive, or the axis is wrong");
    }

    /// Sampling is free and the readout is the bill.
    #[test]
    fn the_readout_is_the_only_unavoidable_cost() {
        assert_eq!(QUASISTATIC_SAMPLING_WORK, 0.0);
        // one bit of extracted entropy costs exactly one Landauer erasure
        let one = readout_floor(1.0, ROOM_TEMPERATURE_K);
        assert!((one - crate::landauer::landauer_bound(ROOM_TEMPERATURE_K)).abs() < 1e-30);
        // and it is proportional to what was extracted, not to the machine
        assert!((readout_floor(64.0, ROOM_TEMPERATURE_K) / one - 64.0).abs() < 1e-9);
        assert_eq!(readout_floor(0.0, ROOM_TEMPERATURE_K), 0.0, "extract nothing, pay nothing");
    }

    /// The TUR floor scales with the SQUARE of the signal-to-noise ratio, not with sample count.
    ///
    /// The direction is the point: a coarse answer is cheap and a sharp one is expensive, in exact
    /// proportion to how sharp. That is the opposite of an analogue carrier, which pays for
    /// precision whether the task needs it or not.
    #[test]
    fn you_pay_for_the_precision_you_extract() {
        let coarse = tur_floor(1.0, ROOM_TEMPERATURE_K);
        let sharp = tur_floor(10.0, ROOM_TEMPERATURE_K);
        assert!((sharp / coarse - 100.0).abs() < 1e-9, "ten times the SNR is a hundred times the cost");
        // at SNR 1 the floor is exactly 2kT
        assert!((coarse - 2.0 * BOLTZMANN_CONSTANT * ROOM_TEMPERATURE_K).abs() < 1e-30);
        // and it does not depend on how many samples were drawn -- that is the whole distinction
        assert_eq!(tur_floor(3.0, ROOM_TEMPERATURE_K), tur_floor(3.0, ROOM_TEMPERATURE_K));
    }

    /// The Bures–Wasserstein distance, against its closed form and its own identities.
    #[test]
    fn the_wasserstein_distance_matches_its_closed_form() {
        // identical laws are at distance zero
        let z = bures_wasserstein_diagonal(&[1.0, 2.0], &[1.0, 2.0], &[3.0, 4.0], &[3.0, 4.0]);
        assert!(z.unwrap().abs() < 1e-15);
        // pure mean shift: the Euclidean distance squared
        let m = bures_wasserstein_diagonal(&[0.0], &[3.0], &[1.0], &[1.0]).unwrap();
        assert!((m - 9.0).abs() < 1e-12, "got {m}");
        // pure variance change, scalar closed form (sqrt(a) - sqrt(b))^2
        let v = bures_wasserstein_diagonal(&[0.0], &[0.0], &[4.0], &[9.0]).unwrap();
        assert!((v - 1.0).abs() < 1e-12, "(2 - 3)^2 = 1, got {v}");
        // and the two add
        let both = bures_wasserstein_diagonal(&[0.0], &[3.0], &[4.0], &[9.0]).unwrap();
        assert!((both - 10.0).abs() < 1e-12, "got {both}");
        // shapes and signs are refused rather than coerced
        assert!(bures_wasserstein_diagonal(&[0.0], &[0.0, 1.0], &[1.0], &[1.0]).is_err());
        assert!(bures_wasserstein_diagonal(&[0.0], &[0.0], &[-1.0], &[1.0]).is_err());
    }

    /// Dissipation falls like 1/tau, and the quality factor is what stops the bound being vacuous.
    #[test]
    fn the_energy_time_accuracy_bound_trades_three_ways() {
        let w2 = 4.0;
        let slow = eddp_floor(w2, 1.0, 1.0, 100.0, 0.5);
        let fast = eddp_floor(w2, 1.0, 1.0, 10.0, 0.5);
        assert!((fast / slow - 10.0).abs() < 1e-9, "ten times the hurry is ten times the floor");
        // demanding more quality raises the floor
        let sloppy = eddp_floor(w2, 1.0, 1.0, 10.0, 0.5);
        let exact = eddp_floor(w2, 1.0, 1.0, 10.0, 0.99);
        // 1/(1-0.99) over 1/(1-0.5) is exactly 50, so this is pinned rather than bounded -- a
        // `> 50x` assertion sits exactly on the boundary and fails.
        assert!((exact / sloppy - 50.0).abs() < 1e-9, "got {}", exact / sloppy);
        // a quality of one would divide by zero, so it is refused rather than returned as infinity
        assert!(std::panic::catch_unwind(|| eddp_floor(w2, 1.0, 1.0, 1.0, 1.0)).is_err());
    }

    /// The measurement this module exists for, on the one metered price this crate owns.
    #[test]
    fn a_metered_run_can_be_priced_against_its_own_floor() {
        // A million sweeps of a 1,024-spin fabric, correlation time 10 sweeps. Samples only:
        // the KV260 measurement exercised no reads or writes and states no price for them, so a
        // run that performed one could not be priced by it -- asserted at the end of this test.
        let ledger = Ledger { samples: 1_024_000_000, reads: 0, writes: 0 };
        let got = cost_per_effective_sample(
            &ledger,
            &KV260_MEASURED,
            1_000_000.0,
            10.0,
            Convergence::SingleChain,
            1.0,
            ROOM_TEMPERATURE_K,
        )
        .expect("a metered price can price a sampled run");
        assert!((got.effective_samples - 50_000.0).abs() < 1e-6, "1e6 / (2*10)");
        assert!(got.joules > 0.0);
        assert_eq!(got.evidence, Evidence::Metered, "a metered price yields a metered cost");
        // The headroom. A digital emulation sits many orders above the bath's own floor, and
        // saying which order with a number is the point of computing it at all.
        assert!(
            got.ratio_to_floor > 1e6,
            "a CMOS emulation should sit far above the floor, got {}",
            got.ratio_to_floor
        );
        assert!((got.joules_per_effective_sample / got.floor - got.ratio_to_floor).abs() < 1e-6);

        // And the price set's own honesty is load-bearing here: this measurement exercised no
        // reads, states no read price, and therefore REFUSES a run that reads rather than
        // treating the unmeasured cost as zero. That refusal is what stops a sampling machine
        // reporting a per-sample energy that silently omits its readout -- the one term the
        // module documentation says the whole floor lives in.
        let with_a_read = Ledger { samples: 1_024_000_000, reads: 1, writes: 0 };
        assert!(
            cost_per_effective_sample(
                &with_a_read,
                &KV260_MEASURED,
                1_000_000.0,
                10.0,
                Convergence::SingleChain,
                1.0,
                ROOM_TEMPERATURE_K
            )
            .is_err(),
            "a sample-only price must not silently price a run that read something out"
        );
    }

    /// JOULES PER CERTIFIED SAMPLE, WITH ITS READOUT, ON METERED SILICON -- the figure this module
    /// was built to state and could not, because until 2026-09-19 no price set in the crate had a
    /// metered read and a sample that is never read out is not a sample anyone received.
    #[test]
    fn a_certified_draw_is_priced_with_its_readout_on_metered_silicon() {
        use crate::ledger::KV260_AXI_METERED;
        // A thousand independent draws on the 1,024-spin metered fabric. Each is `per_draw` sweeps
        // from a certified start and ONE readout of the whole state.
        let (n, draws, per_draw) = (1_024u64, 1_000u64, 64u64);
        let ledger = Ledger { samples: draws * per_draw * n, reads: draws * n, writes: 0 };
        let certified = Convergence::Certified { coalesced_at: 48.0, burn_in: per_draw as f64 };
        // One independent sample per `per_draw` sweeps is `tau_int = per_draw / 2` in this
        // function's units, since it counts `sweeps / (2 tau_int)`.
        let price = |prices| {
            cost_per_effective_sample(
                &ledger,
                prices,
                (draws * per_draw) as f64,
                per_draw as f64 / 2.0,
                certified,
                1.0,
                ROOM_TEMPERATURE_K,
            )
        };
        let got = price(&KV260_AXI_METERED).expect("samples and reads are both metered");
        assert!((got.effective_samples - draws as f64).abs() < 1e-9);
        assert_eq!(got.evidence, Evidence::Metered);
        assert!(got.convergence.is_certified());

        // THE FINDING. At 64 sweeps a draw the readout is HALF THE BILL: a spin read costs about
        // 64 flips on this board, so the break-even readout interval is about 64 sweeps, and any
        // sampler that mixes faster than that is paying mostly to be looked at.
        let sampling = (draws * per_draw * n) as f64 * KV260_AXI_METERED.e_sample;
        let share = 1.0 - sampling / got.joules;
        assert!((0.45..0.55).contains(&share), "the readout's share of the energy = {share}");
        let break_even = KV260_AXI_METERED.e_read / KV260_AXI_METERED.e_sample;
        assert!((55.0..72.0).contains(&break_even), "break-even sweeps per readout = {break_even}");
        // about 1.2 microjoules a certified, delivered sample
        assert!((1.0e-6..1.4e-6).contains(&got.joules_per_effective_sample), "{:e}", got.joules_per_effective_sample);

        // The older measurement still refuses this run, and should: it never metered a read.
        assert!(price(&KV260_MEASURED).is_err());
        // And a draw taken before its chain coalesced is refused whatever the price set knows.
        let early = Convergence::Certified { coalesced_at: 48.0, burn_in: 40.0 };
        assert!(cost_per_effective_sample(&ledger, &KV260_AXI_METERED, 64_000.0, 32.0, early, 1.0, ROOM_TEMPERATURE_K).is_err());
    }

    /// The grade cannot improve by being divided by a floor.
    #[test]
    fn a_simulated_price_yields_a_simulated_cost() {
        let ledger = Ledger { samples: 1_000_000, reads: 0, writes: 0 };
        let got =
            cost_per_effective_sample(
                &ledger,
                &Z1_SPICE,
                1_000.0,
                2.0,
                Convergence::SingleChain,
                1.0,
                ROOM_TEMPERATURE_K,
            )
                .expect("Z1_SPICE states a sample price");
        assert_eq!(got.evidence, Evidence::Simulated);
        assert_ne!(got.evidence, Evidence::Metered, "a projection cannot become a measurement");
    }

    /// Every way of producing a number that is not a measurement is refused.
    #[test]
    fn a_run_that_measured_nothing_is_refused_rather_than_priced() {
        let ledger = Ledger { samples: 1_000, reads: 0, writes: 0 };
        let at = |sweeps, tau| {
            cost_per_effective_sample(
                &ledger,
                &KV260_MEASURED,
                sweeps,
                tau,
                Convergence::SingleChain,
                1.0,
                ROOM_TEMPERATURE_K,
            )
        };
        assert!(at(0.0, 1.0).is_err(), "no sweeps");
        assert!(at(-1.0, 1.0).is_err(), "negative sweeps");
        assert!(at(10.0, 0.0).is_err(), "a zero autocorrelation time");
        assert!(at(10.0, f64::NAN).is_err(), "a NaN autocorrelation time");
        assert!(at(10.0, 1.0).is_ok(), "and a real run is not refused");
        // an unpriceable run is refused by name rather than defaulted to zero
        let unpriced = cost_per_effective_sample(
            &ledger,
            &Prices::UNSTATED,
            10.0,
            1.0,
            Convergence::SingleChain,
            1.0,
            ROOM_TEMPERATURE_K,
        );
        assert!(unpriced.is_err(), "an unstated price must not yield a number");
    }

    /// The pricing used to divide by `tau_int` alone, and `tau_int` cannot report the one failure
    /// that matters most here.
    ///
    /// A chain that never leaves its starting basin fluctuates only inside it, so its measured
    /// autocorrelation is SMALL — which bought it MANY effective samples and a LOW cost per
    /// sample. `examples/burnin` measures the whole thing: at `beta = 0.6` on an 8x8 ferromagnet,
    /// a chain started all-up reports `<m> = +0.974 +- 0.0009` against a truth of exactly zero —
    /// 1060 standard errors out — with `tau_int = 1.13`, which is near the ideal value of one.
    /// R-hat over dispersed chains is 27.2 and coupling from the past puts the required burn-in
    /// between `2^19` and `2^20` sweeps against the 200 that were used.
    ///
    /// So the refusal is not conservatism. The number it refuses to produce has no denominator:
    /// a run that did not sample the target has no independent samples of it to be cheap per.
    #[test]
    fn a_run_that_did_not_converge_is_refused_rather_than_priced() {
        let ledger = Ledger { samples: 1_000_000, reads: 0, writes: 0 };
        let price = |c: Convergence| {
            cost_per_effective_sample(
                &ledger,
                &KV260_MEASURED,
                1_000_000.0,
                10.0,
                c,
                1.0,
                ROOM_TEMPERATURE_K,
            )
        };

        // THE PREMISE, asserted rather than assumed: everything else about this run prices, so a
        // refusal below is about convergence and not about the ledger, the prices or the sweeps.
        let ok = price(Convergence::SingleChain).expect("this run is otherwise priceable");
        assert!(ok.joules_per_effective_sample > 0.0);
        assert_eq!(ok.convergence, Convergence::SingleChain);
        assert!(!Convergence::SingleChain.is_certified(), "one chain cannot certify itself");

        // A burn-in shorter than coalescence required.
        let short = Convergence::Certified { coalesced_at: 1_048_576.0, burn_in: 200.0 };
        assert!(price(short).is_err(), "a burn-in below coalescence must be refused");
        // And the control: the same instrument, with a burn-in that was long enough.
        let long = Convergence::Certified { coalesced_at: 100.0, burn_in: 200.0 };
        let good = price(long).expect("an adequate burn-in is priced");
        assert!(good.convergence.is_certified());

        // Chains from dispersed starts that disagree.
        assert!(price(Convergence::MultiChain { rhat: 27.199 }).is_err(), "R-hat 27 must refuse");
        assert!(
            price(Convergence::MultiChain { rhat: 1.005 }).is_ok(),
            "an R-hat inside the limit is not refused"
        );
        // The limit itself is Vehtari et al. 2021, and it must be the limit that decides.
        assert!(price(Convergence::MultiChain { rhat: 1.0101 }).is_err(), "just above the limit");
        assert!(price(Convergence::MultiChain { rhat: 1.0 }).is_ok(), "exactly one is perfect");
        // SLIGHTLY BELOW ONE IS THE BEST OUTCOME, NOT AN IMPOSSIBLE ONE: R^2 = (N-1)/N + B/(N W),
        // so well-mixed chains land just under 1. This value is measured, from four real chains,
        // and the first version of `refusal` rejected it.
        assert!(
            price(Convergence::MultiChain { rhat: 0.999_849_864 }).is_ok(),
            "an R-hat just under one is excellent agreement and must be priced"
        );
        assert!(price(Convergence::MultiChain { rhat: 0.98 }).is_err(), "under the floor: too few draws");
        assert!(price(Convergence::MultiChain { rhat: 0.5 }).is_err(), "far under the floor");
        assert!(price(Convergence::MultiChain { rhat: f64::NAN }).is_err(), "NaN R-hat");
    }

    /// The refusal must not be reachable by accident from a non-finite input, which would make the
    /// gate look present while the real check never ran.
    #[test]
    fn a_convergence_claim_made_of_non_finite_numbers_is_refused() {
        let inf = Convergence::Certified { coalesced_at: f64::INFINITY, burn_in: 1.0 };
        assert!(inf.refusal().is_some(), "an infinite coalescence time is not a certification");
        let nan = Convergence::Certified { coalesced_at: 1.0, burn_in: f64::NAN };
        assert!(nan.refusal().is_some(), "a NaN burn-in is not a certification");
        // A NEGATIVE coalescence time would otherwise pass `burn_in < coalesced_at`, since
        // `200 < -5` is false, and nonsense would read as a certification.
        let neg = Convergence::Certified { coalesced_at: -5.0, burn_in: 200.0 };
        assert!(neg.refusal().is_some(), "a negative coalescence time is not a certification");
        // and the ordinary case still passes, so the guard did not swallow everything
        let fine = Convergence::Certified { coalesced_at: 1.0, burn_in: 2.0 };
        assert!(fine.refusal().is_none(), "a real certification is not refused");
    }

    /// Standard normals by Box-Muller, so the fixtures below need nothing the crate does not have.
    fn normals(seed: u64, n: usize, scale: f64, shift: f64) -> Vec<f64> {
        let mut rng = crate::rng::Pcg::new(seed, 0xC0_17E5);
        let mut out = Vec::with_capacity(n);
        while out.len() < n {
            let u1 = rng.f64().max(1e-300);
            let u2 = rng.f64();
            let r = (-2.0 * u1.ln()).sqrt();
            out.push(shift + scale * r * (core::f64::consts::TAU * u2).cos());
        }
        out
    }

    /// The constructor exists so the strong claim is as easy as the weak one — so it has to be
    /// STRONG. It takes the largest of three R-hats because each is blind where another sees,
    /// and the fixture that proves it is the one `rhat`'s own header describes: chains whose
    /// means agree and whose spreads do not. Plain split R-hat passes them. The folded one does
    /// not, and neither may this.
    #[test]
    fn convergence_from_chains_takes_the_strictest_diagnostic_and_refuses_too_little_evidence() {
        // Four chains of one distribution: agreement, and it must not be refused.
        let same: Vec<Vec<f64>> = (0..4).map(|k| normals(k, 2_000, 1.0, 0.0)).collect();
        let c = Convergence::from_chains(&same);
        assert!(c.refusal().is_none(), "four chains of one law must agree: {c:?}");
        assert!(c.is_certified());

        // Two up, two down: the burn-in failure itself. Every diagnostic sees it.
        let split = vec![
            normals(10, 2_000, 0.1, 1.0),
            normals(11, 2_000, 0.1, -1.0),
            normals(12, 2_000, 0.1, 1.0),
            normals(13, 2_000, 0.1, -1.0),
        ];
        assert!(Convergence::from_chains(&split).refusal().is_some(), "chains in two basins");

        // SAME MEAN, DIFFERENT SPREAD. The premise first: plain split R-hat really does pass this,
        // or the assertion after it would prove nothing about taking the maximum.
        let spread = vec![
            normals(20, 2_000, 1.0, 0.0),
            normals(21, 2_000, 1.0, 0.0),
            normals(22, 2_000, 3.0, 0.0),
            normals(23, 2_000, 3.0, 0.0),
        ];
        let plain = crate::rhat::split_rhat(&spread);
        assert!(
            plain < Convergence::RHAT_LIMIT,
            "the fixture must fool plain R-hat or it tests nothing: {plain}"
        );
        assert!(
            Convergence::MultiChain { rhat: plain }.refusal().is_none(),
            "and built by hand from plain R-hat it WOULD be priced"
        );
        let strict = Convergence::from_chains(&spread);
        assert!(strict.refusal().is_some(), "the folded diagnostic must catch it: {strict:?}");

        // Too little evidence is not agreement: one chain, and chains too short to diagnose.
        let one = vec![normals(30, 2_000, 1.0, 0.0)];
        assert!(Convergence::from_chains(&one).refusal().is_some(), "one chain certifies nothing");
        let short = vec![vec![0.1, 0.2, 0.3], vec![0.2, 0.1, 0.3]];
        assert!(Convergence::from_chains(&short).refusal().is_some(), "three draws certify nothing");
        assert!(Convergence::from_chains(&[]).refusal().is_some(), "no chains certify nothing");
    }

    /// The other instrument, from real perfect draws rather than typed numbers.
    #[test]
    fn convergence_from_perfect_draws_holds_the_burn_in_against_the_worst_coalescence() {
        let g = crate::ising::grid2d(4, 4, 1.0);
        let draws = crate::cftp::exact_draws(&g, 0.2, 7, 50).expect("a hot ferromagnet coalesces");
        let worst = draws.iter().map(|d| d.coalesced_at).max().expect("fifty draws");
        assert!(worst >= 1, "coalescence takes at least one sweep");

        let enough = Convergence::from_perfect_draws(&draws, worst);
        assert!(enough.refusal().is_none(), "a burn-in equal to the worst coalescence is adequate");
        let short = Convergence::from_perfect_draws(&draws, worst - 1);
        assert!(short.refusal().is_some(), "one sweep short of the worst draw must be refused");
        // it is the WORST draw that binds, not the first or the mean
        assert_eq!(enough, Convergence::Certified { coalesced_at: worst as f64, burn_in: worst as f64 });
        // and an empty measurement certifies nothing
        assert!(Convergence::from_perfect_draws(&[], 1_000_000).refusal().is_some());
    }
}
