//! Accuracy parity for one sampler on one open Ising side.
//!
//! The numbers are the gates already in this harness's tests, applied per side and per
//! sampler. A and B are judged separately. Neither is the other's reference.
//!
//! * Sides with a transfer-matrix mean: the chain's mean is within 0.05 per site of it.
//!   That is `an_8x8_reference_chain_lands_near_the_transfer_matrix`.
//! * Larger sides: the per-site gap to Onsager is strictly smaller than the exact 16×16
//!   gap. That is `ising_16_is_exact_and_larger_sides_use_onsager_plus_a_chain`. The
//!   constant 0.08 in that test is a sanity bound on the 16×16 gap, not a wider gate.
//! * Two seeds agree to strictly less than 0.05 per site. Same test.
//!
//! A size that fails is refused. The other sampler's row is not rewritten to fill it.

/// Per-site agreement used by the existing Ising reference tests. Not a knob.
pub const PER_SITE: f64 = 0.05;

#[derive(Clone, Debug, PartialEq)]
pub struct Verdict {
    pub ok: bool,
    /// Empty when the gate holds. Otherwise the numbers that failed.
    pub reason: String,
}

/// Judge one sampler's mean energy on an open grid.
///
/// `exact` is the transfer-matrix mean when this side has one. `other_seed` is a second
/// chain's mean. `gap16` is the absolute per-site gap of the exact 16×16 mean to Onsager.
pub fn judge_ising_open(
    side: usize,
    mean: f64,
    other_seed: Option<f64>,
    exact: Option<f64>,
    onsager_density: f64,
    gap16: f64,
) -> Verdict {
    let n = (side * side) as f64;
    if !(mean.is_finite() && n > 0.0 && gap16.is_finite() && onsager_density.is_finite()) {
        return Verdict {
            ok: false,
            reason: format!(
                "side {side} parity refused: a mean was not finite (mean {mean}, gap16 {gap16})"
            ),
        };
    }
    if let Some(exact) = exact {
        if !exact.is_finite() {
            return Verdict {
                ok: false,
                reason: format!(
                    "side {side} parity refused: the transfer-matrix mean was not finite"
                ),
            };
        }
        let gap = (mean - exact).abs() / n;
        if gap >= PER_SITE {
            return Verdict {
                ok: false,
                reason: format!(
                    "side {side} parity refused: |mean - transfer matrix| / n = {gap} is not below {PER_SITE} (mean {mean}, transfer {exact})"
                ),
            };
        }
    } else {
        let density = mean / n;
        let gap = (density - onsager_density).abs();
        if gap >= gap16 {
            return Verdict {
                ok: false,
                reason: format!(
                    "side {side} parity refused: per-site gap to Onsager {gap} is not strictly below the exact 16x16 gap {gap16} (density {density}, Onsager {onsager_density})"
                ),
            };
        }
    }
    if let Some(other) = other_seed {
        if !other.is_finite() {
            return Verdict {
                ok: false,
                reason: format!(
                    "side {side} parity refused: the second seed's mean was not finite"
                ),
            };
        }
        let agree = (mean - other).abs() / n;
        if agree >= PER_SITE {
            return Verdict {
                ok: false,
                reason: format!(
                    "side {side} parity refused: two seeds differ by {agree} per site, not below {PER_SITE} ({mean} vs {other})"
                ),
            };
        }
    }
    Verdict {
        ok: true,
        reason: String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::oracle::ising_open_transfer;
    use ferrotherm::free_energy::onsager_energy_density;

    #[test]
    fn a_transfer_matrix_hit_passes_and_half_a_unit_per_site_fails() {
        let side = 8usize;
        let n = (side * side) as f64;
        let exact = -100.0;
        let ok = judge_ising_open(side, exact + 0.01, Some(exact), Some(exact), 0.0, 0.01);
        assert!(ok.ok, "{}", ok.reason);
        let bad = judge_ising_open(
            side,
            exact + PER_SITE * n,
            Some(exact),
            Some(exact),
            0.0,
            0.01,
        );
        assert!(!bad.ok, "the 0.05 per-site line is not a pass");
        assert!(bad.reason.contains("transfer"), "{}", bad.reason);
    }

    #[test]
    fn two_seeds_past_the_line_are_refused_on_their_own() {
        let side = 8usize;
        let n = (side * side) as f64;
        let exact = -40.0;
        let bad = judge_ising_open(
            side,
            exact,
            Some(exact + PER_SITE * n),
            Some(exact),
            -1.0,
            0.2,
        );
        assert!(!bad.ok);
        assert!(bad.reason.contains("two seeds"), "{}", bad.reason);
    }

    #[test]
    fn the_16x16_gap_is_the_threshold_not_a_wider_constant() {
        let j = 1.0;
        let beta = 0.3;
        let tm = ising_open_transfer(16, j, beta).unwrap().mean_energy;
        let onsager = onsager_energy_density(beta, j);
        let gap16 = (tm / 256.0 - onsager).abs();
        assert!(
            gap16 < 0.08,
            "the existing sanity bound still holds, gap16 {gap16}"
        );
        assert!(gap16 > 0.0);
        let n32 = (32 * 32) as f64;
        // Exactly the 16x16 gap. The gate is strict, so this is a refusal.
        let mean = (onsager + gap16) * n32;
        let on_the_line = judge_ising_open(32, mean, Some(mean), None, onsager, gap16);
        assert!(!on_the_line.ok, "landing on the 16x16 gap must be refused");
        assert!(
            on_the_line.reason.contains("16x16"),
            "{}",
            on_the_line.reason
        );
        // A looser constant of 0.08 would accept this. The gate must not.
        assert!(gap16 < 0.08);
        let inside = (onsager + gap16 * 0.5) * n32;
        let held = judge_ising_open(32, inside, Some(inside), None, onsager, gap16);
        assert!(held.ok, "{}", held.reason);
    }

    #[test]
    fn a_missing_second_seed_does_not_invent_agreement() {
        // One chain can still pass the reference. The caller has to pass the second seed
        // when it has one. Omitting it is not a fabricated pass of the two-seed clause.
        let v = judge_ising_open(8, -10.0, None, Some(-10.0), -1.0, 0.2);
        assert!(v.ok, "{}", v.reason);
    }
}
