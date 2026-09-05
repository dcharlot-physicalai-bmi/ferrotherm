//! What the part you did NOT accelerate does to the headline.
//!
//! Every thermodynamic system that has been benchmarked end to end is a hybrid: some layers run on
//! the new substrate and the rest run on a GPU, an FPGA or a host CPU. The vendor reports one
//! multiplier for the whole system, and that multiplier is bounded — hard — by the unaccelerated
//! remainder, whatever the sampler does. This module is that bound, made executable.
//!
//! The arithmetic is Amdahl's law in joules, and it has a consequence worth stating on its own:
//!
//! > The system multiplier is LINEAR in the baseline you chose to compare against and LINEAR in the
//! > work your architecture needs to reach the same quality. It is bounded above, by a constant, in
//! > the efficiency of the accelerator. Two of those three knobs belong to whoever writes the press
//! > release, and the third — the physics — is the one that saturates.
//!
//! Concretely, on the numbers Extropic published for Z1T on 2026-09-04 ([`Z1T_PUBLISHED`]): the
//! system spends 8.74 nJ/token on Z1 and 285.78 nJ/token on the FPGA that carries the rest. Setting
//! the Z1 half to **exactly zero joules** moves the headline from 138.9x to 143.1x. The entire
//! thermodynamic contribution is worth 3.1% of the claim; the other two knobs are worth 10x each.
//!
//! That is not an argument against the substrate. It is the argument for [`Split::host_speedup_for`]:
//! the vendor's own stated path to 1000x asks the NON-thermodynamic half to improve 8.9x, and until
//! it does, the sampler's efficiency is not what the number is measuring.
//!
//! ```
//! use ferrotherm::hybrid::Z1T_PUBLISHED;
//!
//! let s = Z1T_PUBLISHED;
//! assert!((s.factor().unwrap() - 138.87).abs() < 0.01);   // as published: "up to 140x"
//! assert!((s.ceiling().unwrap() - 143.12).abs() < 0.01);  // ... with a FREE accelerator
//! assert!((s.headroom().unwrap() - 1.0306).abs() < 1e-4); // 3.1% still on the table
//! ```

/// A unit of work split across an accelerated part and an unaccelerated remainder, priced against
/// the incumbent that does the whole thing itself.
///
/// All three energies are per the SAME unit of work — one token, one control tick, one sample.
/// Mixing units here is the mistake the type cannot catch, so name the unit in `source`.
#[derive(Clone, Copy, Debug)]
pub struct Split {
    /// Energy on the new substrate, per unit of work.
    pub accel_joules: f64,
    /// Energy on everything else the workload still needs — FPGA, host CPU, GPU, DRAM.
    ///
    /// This is the term the headline is bounded by, and the one vendor posts report last.
    pub host_joules: f64,
    /// What the incumbent spends on the same unit of work, doing all of it.
    ///
    /// A CHOSEN number: the same GPU quoted at 10% and 100% model-FLOPs utilisation differs by 10x,
    /// and the multiplier inherits that factor exactly. See [`Split::with_baseline_scaled`].
    pub baseline_joules: f64,
    /// WHAT this splits and where the three numbers came from, unit included.
    pub source: &'static str,
}

/// Why a split cannot be priced.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum HybridError {
    /// An energy that is negative, infinite or NaN.
    NotPhysical(&'static str),
    /// The unaccelerated remainder is zero, so no finite ceiling exists.
    ///
    /// Not an error about arithmetic. A system with NO unaccelerated remainder is the thing this
    /// module says nobody has built yet; if you have one, the bound does not apply and you should
    /// be reporting the sampler's efficiency directly.
    NothingLeftOnTheHost,
    /// The whole system spends nothing, so there is no multiplier to report.
    NoWorkPriced,
    /// The target multiplier is unreachable even with the accelerator running free.
    ///
    /// Carries the ceiling so the caller can say by how much: this is the variant that turns
    /// "we are going to 1000x" into an arithmetic statement about a different chip.
    AboveCeiling {
        /// The multiplier requested.
        target: f64,
        /// The most the system can reach with `accel_joules` set to zero.
        ceiling: f64,
    },
}

impl core::fmt::Display for HybridError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            HybridError::NotPhysical(s) => write!(f, "{s}"),
            HybridError::NothingLeftOnTheHost => write!(
                f,
                "this split leaves nothing on the host, so it has no finite ceiling -- which is a \
                 claim about a system nobody has demonstrated, not a rounding case"
            ),
            HybridError::NoWorkPriced => write!(
                f,
                "the system spends nothing on this unit of work, so there is no multiplier to report"
            ),
            HybridError::AboveCeiling { target, ceiling } => write!(
                f,
                "{target:.4}x is unreachable here: with the accelerator running at EXACTLY zero \
                 joules this system reaches {ceiling:.4}x. The remaining distance is a claim about \
                 the unaccelerated half, not about the sampler"
            ),
        }
    }
}

impl Split {
    /// Refuses anything that is not a physical split.
    pub fn new(
        accel_joules: f64,
        host_joules: f64,
        baseline_joules: f64,
        source: &'static str,
    ) -> Result<Split, HybridError> {
        for (v, what) in [
            (accel_joules, "accelerator energy must be finite and non-negative"),
            (host_joules, "host energy must be finite and non-negative"),
            (baseline_joules, "baseline energy must be finite and non-negative"),
        ] {
            if !v.is_finite() || v < 0.0 {
                return Err(HybridError::NotPhysical(what));
            }
        }
        Ok(Split { accel_joules, host_joules, baseline_joules, source })
    }

    /// What the hybrid spends per unit of work.
    #[must_use]
    pub fn total(&self) -> f64 {
        self.accel_joules + self.host_joules
    }

    /// The reported multiplier: `baseline / total`.
    pub fn factor(&self) -> Result<f64, HybridError> {
        let t = self.total();
        if t <= 0.0 {
            return Err(HybridError::NoWorkPriced);
        }
        Ok(self.baseline_joules / t)
    }

    /// Fraction of the hybrid's energy that the accelerator is responsible for.
    ///
    /// On [`Z1T_PUBLISHED`] this is 3.0%, which is the whole finding restated: two thirds of the
    /// press release is about a term worth three percent of the arithmetic.
    pub fn accel_share(&self) -> Result<f64, HybridError> {
        let t = self.total();
        if t <= 0.0 {
            return Err(HybridError::NoWorkPriced);
        }
        Ok(self.accel_joules / t)
    }

    /// The multiplier this system reaches with the accelerator running at EXACTLY zero joules.
    ///
    /// The hard ceiling. No improvement to the sampler — none, ever, including a perfect one —
    /// takes the system past this, because the host still has to do its half. Report it next to
    /// [`Split::factor`] and the reader can see immediately whether the claim is about the physics
    /// or about the plumbing.
    pub fn ceiling(&self) -> Result<f64, HybridError> {
        if self.host_joules <= 0.0 {
            return Err(HybridError::NothingLeftOnTheHost);
        }
        Ok(self.baseline_joules / self.host_joules)
    }

    /// `ceiling / factor` — the multiplier still available from the accelerated half.
    ///
    /// Equivalently `total / host`, so it does not depend on the baseline at all: it is a property
    /// of the machine, not of the comparison. A headroom of 1.03 means the accelerator has already
    /// given the system 97% of everything it will ever give it.
    pub fn headroom(&self) -> Result<f64, HybridError> {
        if self.host_joules <= 0.0 {
            return Err(HybridError::NothingLeftOnTheHost);
        }
        let t = self.total();
        if t <= 0.0 {
            return Err(HybridError::NoWorkPriced);
        }
        Ok(t / self.host_joules)
    }

    /// Host energy that would have to be reached for the system to hit `target`, accelerator fixed.
    ///
    /// Errors with [`HybridError::AboveCeiling`] when the target is unreachable at any accelerator
    /// efficiency, which is the answer more often than not.
    pub fn host_joules_for(&self, target: f64) -> Result<f64, HybridError> {
        if !target.is_finite() || target <= 0.0 {
            return Err(HybridError::NotPhysical("a target multiplier must be finite and positive"));
        }
        let budget = self.baseline_joules / target;
        let left = budget - self.accel_joules;
        if left <= 0.0 {
            return Err(HybridError::AboveCeiling { target, ceiling: self.ceiling()? });
        }
        Ok(left)
    }

    /// How much better the UNACCELERATED half has to get for the system to hit `target`.
    ///
    /// The number that turns a roadmap into an engineering statement about a specific chip that is
    /// not the sampler.
    pub fn host_speedup_for(&self, target: f64) -> Result<f64, HybridError> {
        let need = self.host_joules_for(target)?;
        if self.host_joules <= 0.0 {
            return Err(HybridError::NothingLeftOnTheHost);
        }
        Ok(self.host_joules / need)
    }

    /// The same system compared against a baseline scaled by `k`.
    ///
    /// The utilisation axis. A GPU quoted at 10% model-FLOPs utilisation costs 10x what the same
    /// GPU costs at 100%, and every multiplier computed against it is 10x larger — exactly, because
    /// [`Split::factor`] is linear in the baseline. Sweeping `k` is how a reader separates the
    /// hardware result from the choice of opponent.
    ///
    /// This is not an accusation. Batch-1 sequential decode really does run a GPU at low
    /// utilisation, so a low-MFU baseline is defensible; it is just not a fact about the sampler,
    /// and it should be visible rather than folded in.
    #[must_use]
    pub fn with_baseline_scaled(mut self, k: f64) -> Split {
        self.baseline_joules *= k;
        self
    }

    /// The same system charged `m` times the work — the iso-QUALITY correction.
    ///
    /// A sparse architecture that needs an order of magnitude more operations to reach the same
    /// loss is not doing the same unit of work as the dense model it is timed against. Charging the
    /// hybrid `m x` its energy makes the comparison iso-quality instead of iso-parameter, and since
    /// [`Split::factor`] is linear in `1/m`, an order-of-magnitude work penalty costs an order of
    /// magnitude of headline.
    ///
    /// [`Split::headroom`] is INVARIANT under this: scaling both halves does not change which one
    /// binds.
    #[must_use]
    pub fn with_work_multiplier(mut self, m: f64) -> Split {
        self.accel_joules *= m;
        self.host_joules *= m;
        self
    }
}

/// Z1T as published: per-token energy split across Z1 and the FPGA that carries the rest.
///
/// From Extropic's Z1T post (2026-09-04, extropic.ai/writing/z1t): 8.74 nJ/token on Z1, 285.78
/// nJ/token on the FPGA, 294.52 nJ/token total, against an H100 measured 2026-08-12 at batch-1
/// sequential decode and quoted at 10% model-FLOPs utilisation, ~40.9 uJ/token.
///
/// **PROVENANCE, stated once and carried in the constant:** the Z1 term is not measured. The post
/// says so — "theoretical chip energy consumption of Z1 based on our best estimates, which are
/// anchored to reality from our experiments with similar pbits in X0". Z1 is at tapeout. The H100
/// term IS measured. So the ratio is a measurement divided by a projection, and the projection is
/// the 3% one.
///
/// Two further facts, both from the same post, that the multiplier does not contain:
///
/// 1. The comparison is ISO-PARAMETER — the H100 runs "the same next-token step run densely, with
///    no sparsity exploited" — while the post separately states "we need about an order of magnitude
///    more FLOPs with our sparser Z1T model to achieve the same loss as a GPT-2 model". Apply
///    [`Split::with_work_multiplier`] with `10.0` for the iso-quality figure.
/// 2. The baseline utilisation is a choice; the post itself gives 10%, 50% and 100%. Apply
///    [`Split::with_baseline_scaled`] with `5.0` or `10.0`.
///
/// The Z1 sampling constant behind the 8.74 nJ is `1.3e-14 J/sample`, which is **1.83x** the
/// `7.09e-15` in [`Z1_SPICE`](crate::ledger::Z1_SPICE) from the Thermalizers appendix six weeks
/// earlier. Two published prices for one quantity, a factor of 1.8 apart; this crate carries both
/// and asserts neither.
pub const Z1T_PUBLISHED: Split = Split {
    accel_joules: 8.74e-9,
    host_joules: 285.78e-9,
    baseline_joules: 40.9e-6,
    source: "Extropic Z1T post 2026-09-04, per token: 8.74 nJ Z1 (PROJECTED from X0 pbits; Z1 is at \
             tapeout) + 285.78 nJ FPGA (estimated at 0.2 pJ/MAC, 3.0 pJ/scalar, 1.5 W static) vs an \
             H100 MEASURED 2026-08-12 at batch-1 decode, quoted at 10% MFU. Iso-parameter, not \
             iso-loss.",
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_published_headline_reproduces_from_the_published_parts() {
        let s = Z1T_PUBLISHED;
        assert!((s.total() - 294.52e-9).abs() < 1e-13, "total = {} nJ", s.total() * 1e9);
        let f = s.factor().unwrap();
        assert!((f - 138.87).abs() < 0.01, "factor = {f}");
    }

    #[test]
    fn a_free_accelerator_buys_three_percent() {
        // The bound the module exists for. Setting the ENTIRE thermodynamic contribution to zero
        // moves the headline 138.87 -> 143.12. Every remaining order of magnitude in the vendor's
        // roadmap has to come from the FPGA.
        let s = Z1T_PUBLISHED;
        let c = s.ceiling().unwrap();
        assert!((c - 143.12).abs() < 0.01, "ceiling = {c}");
        let h = s.headroom().unwrap();
        assert!((h - 1.0306).abs() < 1e-4, "headroom = {h}");
        assert!(s.accel_share().unwrap() < 0.03, "accel share = {}", s.accel_share().unwrap());
        // headroom is a property of the machine, not of the opponent it was pointed at
        let quiet = s.with_baseline_scaled(0.1);
        assert!((quiet.headroom().unwrap() - h).abs() < 1e-12);
    }

    #[test]
    fn the_thousand_x_roadmap_is_a_claim_about_the_fpga() {
        let s = Z1T_PUBLISHED;
        let k = s.host_speedup_for(1000.0).unwrap();
        assert!((k - 8.886).abs() < 0.01, "host must improve {k}x");
        // and with the sampler free it is still 7x, so the sampler is not what is in the way
        let free = Split { accel_joules: 0.0, ..s };
        assert!((free.host_speedup_for(1000.0).unwrap() - 6.987).abs() < 0.01);
    }

    #[test]
    fn the_two_vendor_knobs_are_each_worth_an_order_of_magnitude() {
        let s = Z1T_PUBLISHED;
        // baseline utilisation: 10% MFU -> 100% MFU divides the headline by ten, exactly
        let full_mfu = s.with_baseline_scaled(0.1);
        assert!((full_mfu.factor().unwrap() - 13.887).abs() < 0.01);
        // iso-quality: the post's own ~10x FLOP penalty for reaching GPT-2's loss
        let iso_loss = s.with_work_multiplier(10.0);
        assert!((iso_loss.factor().unwrap() - 13.887).abs() < 0.01);
        // both at once, which is the number a datacentre would actually be buying
        let both = s.with_baseline_scaled(0.1).with_work_multiplier(10.0);
        assert!((both.factor().unwrap() - 1.3887).abs() < 1e-3, "{}", both.factor().unwrap());
        // ... while the sampler's own share never moved
        assert!((both.headroom().unwrap() - s.headroom().unwrap()).abs() < 1e-12);
    }

    #[test]
    fn unreachable_targets_report_the_ceiling_rather_than_a_negative_budget() {
        let s = Z1T_PUBLISHED;
        match s.host_joules_for(5000.0) {
            Err(HybridError::AboveCeiling { target, ceiling }) => {
                assert!((target - 5000.0).abs() < 1e-9);
                assert!((ceiling - 143.12).abs() < 0.01);
            }
            other => panic!("expected AboveCeiling, got {other:?}"),
        }
    }

    #[test]
    fn a_split_with_no_host_half_has_no_ceiling_rather_than_an_infinite_one() {
        let s = Split::new(1e-9, 0.0, 1e-6, "hypothetical fully-offloaded system").unwrap();
        assert_eq!(s.ceiling(), Err(HybridError::NothingLeftOnTheHost));
        assert_eq!(s.headroom(), Err(HybridError::NothingLeftOnTheHost));
        assert!((s.factor().unwrap() - 1000.0).abs() < 1e-9);
    }

    #[test]
    fn negative_and_nonfinite_energies_are_refused() {
        assert!(matches!(
            Split::new(-1.0, 1.0, 1.0, "x"),
            Err(HybridError::NotPhysical(_))
        ));
        assert!(matches!(
            Split::new(1.0, f64::NAN, 1.0, "x"),
            Err(HybridError::NotPhysical(_))
        ));
        assert!(matches!(
            Split::new(0.0, 0.0, 1.0, "x").unwrap().factor(),
            Err(HybridError::NoWorkPriced)
        ));
    }

    #[test]
    fn the_two_published_sampling_prices_differ_by_a_factor_this_crate_does_not_resolve() {
        // Thermalizers Table IV: 7.09e-15 J. Z1T post: 1.3e-14 J. Same quantity, six weeks apart.
        let ratio = 1.3e-14 / crate::ledger::Z1_SPICE.e_sample;
        assert!((ratio - 1.834).abs() < 0.01, "ratio = {ratio}");
    }
}
