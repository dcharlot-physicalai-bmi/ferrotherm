//! The thermodynamic floor under the ledger — Landauer's bound, the fluctuation theorems a
//! stationary kernel obeys exactly, and how far each priced device sits above the floor.
//!
//! # Why this exists
//!
//! [`crate::ledger`] prices a device's operations in joules from published or measured figures:
//! 7.09 fJ per update in a Z1-class SPICE model, 10.85 pJ per flip metered on a KV260. Neither
//! number says how far the device sits above physics. Landauer (1961): erasing one bit of
//! information into a bath at temperature `T` dissipates at least `k_B T ln 2` — `2.87e-21 J` at
//! 300 K, from the SI-exact `k_B = 1.380649e-23 J/K`. A heat-bath update replaces a spin by a
//! fresh draw, so it forgets the old value: an erasure of at most one bit, and `k_B T ln 2` is the
//! floor per update. [`floor_ratio`] is what a price set costs in those units: the KV260 update is
//! `3.8e9` Landauer bits, the Z1 SPICE update `2.5e6`. That is the gap the whole field's "physics
//! does the sampling" claim has to close, stated in the one unit that does not depend on a
//! technology.
//!
//! # Fluctuation theorems, exactly
//!
//! For a stationary Markov chain in which every transition has a reverse, the one-step total
//! entropy production `Δs = ln[π(x) P(x, y) / (π(y) P(y, x))]` obeys the detailed fluctuation
//! theorem `P(Δs = a) = e^a P(Δs = -a)` (Evans–Searles 1994, Gallavotti–Cohen 1995, Crooks 1999)
//! and therefore the integral one, `<e^{-Δs}> = 1` (Seifert 2005) — identities, not approximations,
//! which [`fluctuation`] verifies on the dense kernel and which fail, by exactly the mass of the
//! one-way transitions, when a kernel has some. The mean of `Δs` is
//! [`crate::autocorr::entropy_production`], and the two must agree to rounding.
//!
//! # What the floor is not
//!
//! It is a floor on dissipation per erased bit, not a prediction of any device's cost, and a
//! sampler that erases less than a bit per update — a Gibbs update whose conditional is nearly
//! deterministic forgets almost nothing — sits above a smaller floor. The ratio reported here uses
//! the full bit; a tighter one would need the conditional entropy per update, which the exact
//! kernel can supply and which is a later addition.

use crate::autocorr::{apply_distribution, own_law, AutocorrError, Kernel, MAX_DENSE_SPINS};
use crate::graph::Graph;
use crate::ledger::Prices;
use std::collections::BTreeMap;

/// Boltzmann's constant, joules per kelvin, exact since the 2019 SI.
pub const BOLTZMANN_CONSTANT: f64 = 1.380_649e-23;

/// Landauer's bound: the least heat, in joules, that erasing one bit dissipates into a bath at
/// `temperature_k` kelvin. `2.87e-21 J` at 300 K.
#[must_use]
pub fn landauer_bound(temperature_k: f64) -> f64 {
    BOLTZMANN_CONSTANT * temperature_k * core::f64::consts::LN_2
}

/// How many Landauer bits one update costs under `prices` at `temperature_k`: `e_sample` over
/// [`landauer_bound`]. A device at the floor would read 1.
#[must_use]
pub fn floor_ratio(prices: &Prices, temperature_k: f64) -> f64 {
    prices.e_sample / landauer_bound(temperature_k)
}

/// The one-step entropy production of a kernel in its steady state, and the two theorems it obeys.
#[derive(Clone, Debug, PartialEq)]
pub struct Fluctuation {
    /// `<Δs>`: the entropy production rate, nats per step; equals
    /// [`crate::autocorr::entropy_production`].
    pub mean: f64,
    /// `<e^{-Δs}>` over the transitions that have a reverse: exactly 1 when all do.
    pub integral: f64,
    /// The worst relative departure of `P(Δs = a) / P(Δs = -a)` from `e^a` over every value `a`
    /// the chain produces: zero to rounding when every transition has a reverse.
    pub symmetry_defect: f64,
    /// Steady-state probability mass of transitions with no reverse, which the theorems exclude
    /// and which is what the integral falls short of 1 by.
    pub one_way_mass: f64,
}

/// [`Fluctuation`] of `kernel` at `beta` on `g`, from the dense operator and the kernel's own law.
///
/// # Errors
///
/// As [`own_law`], and [`AutocorrError::TooManyForDense`] above [`MAX_DENSE_SPINS`].
pub fn fluctuation(g: &Graph, beta: f64, kernel: Kernel) -> Result<Fluctuation, AutocorrError> {
    if g.n > MAX_DENSE_SPINS {
        return Err(AutocorrError::TooManyForDense { n: g.n, max: MAX_DENSE_SPINS });
    }
    let pi = own_law(g, beta, kernel)?;
    let m = 1usize << g.n;
    let mut p = vec![0.0f64; m * m];
    let mut point = vec![0.0f64; m];
    for x in 0..m {
        point[x] = 1.0;
        let row = apply_distribution(g, beta, kernel, &point);
        point[x] = 0.0;
        p[x * m..(x + 1) * m].copy_from_slice(&row);
    }
    let mut mean = 0.0;
    let mut integral = 0.0;
    let mut one_way = 0.0;
    // P(Δs = a) for a > 0 and P(Δs = -a), keyed by the bits of a: the reverse transition
    // produces exactly the negated difference of the same two logarithms.
    let mut plus: BTreeMap<u64, f64> = BTreeMap::new();
    let mut minus: BTreeMap<u64, f64> = BTreeMap::new();
    for x in 0..m {
        for y in 0..m {
            let fwd = pi[x] * p[x * m + y];
            if fwd <= 0.0 {
                continue;
            }
            let bwd = pi[y] * p[y * m + x];
            if bwd <= 0.0 {
                one_way += fwd;
                continue;
            }
            let a = fwd.ln() - bwd.ln();
            mean += fwd * a;
            integral += fwd * (-a).exp();
            if a > 0.0 {
                *plus.entry(a.to_bits()).or_insert(0.0) += fwd;
            } else if a < 0.0 {
                *minus.entry((-a).to_bits()).or_insert(0.0) += fwd;
            }
        }
    }
    let mut defect = 0.0f64;
    for (key, p_plus) in &plus {
        let a = f64::from_bits(*key);
        let want = a.exp();
        let ratio = minus.get(key).map_or(f64::INFINITY, |p_minus| p_plus / p_minus);
        defect = defect.max((ratio - want).abs() / want);
    }
    Ok(Fluctuation { mean, integral, symmetry_defect: defect, one_way_mass: one_way })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autocorr::entropy_production;
    use crate::graph::GraphBuilder;
    use crate::ledger::{KV260_MEASURED, Z1_SPICE};
    use crate::rng::Pcg;

    fn grid_glass(w: usize, h: usize, seed: u64) -> Graph {
        let mut rng = Pcg::new(seed, 0x6A);
        let mut b = GraphBuilder::new(w * h);
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                if x + 1 < w {
                    b.couple(i, i + 1, if rng.f64() < 0.5 { -1.0 } else { 1.0 });
                }
                if y + 1 < h {
                    b.couple(i, i + w, if rng.f64() < 0.5 { -1.0 } else { 1.0 });
                }
            }
        }
        for i in 0..w * h {
            b.bias(i, (rng.f64() - 0.5) * 0.4);
        }
        b.build()
    }

    /// `k_B T ln 2` at 300 K from the SI-exact constant, and the two priced devices in that unit.
    #[test]
    fn the_floor_at_room_temperature_and_the_devices_above_it() {
        let floor = landauer_bound(300.0);
        assert!((floor - 2.870_978_e-21).abs() < 1e-26, "kT ln 2 at 300 K: {floor:e}");
        let kv = floor_ratio(&KV260_MEASURED, 300.0);
        let z1 = floor_ratio(&Z1_SPICE, 300.0);
        assert!((kv / 3.778e9 - 1.0).abs() < 1e-3, "KV260: {kv:e} Landauer bits per flip");
        assert!((z1 / 2.469e6 - 1.0).abs() < 1e-3, "Z1 SPICE: {z1:e} Landauer bits per update");
    }

    /// On a kernel whose every transition has a reverse the integral theorem holds to rounding,
    /// the detailed one to rounding at every value, no mass is one-way, and the mean is the
    /// entropy production the autocorrelation module computes.
    #[test]
    fn the_fluctuation_theorems_hold_exactly_on_a_reversible_support_kernel() {
        let g = grid_glass(3, 3, 5);
        for kernel in [Kernel::ChromaticGibbs, Kernel::SequentialGibbs, Kernel::Synchronous] {
            let f = fluctuation(&g, 1.1, kernel).unwrap();
            assert!((f.integral - 1.0).abs() < 1e-12, "{kernel:?}: <exp(-ds)> = {}", f.integral);
            assert!(f.symmetry_defect < 1e-9, "{kernel:?}: symmetry defect {:e}", f.symmetry_defect);
            assert_eq!(f.one_way_mass, 0.0, "{kernel:?}: one-way mass");
            let sigma = entropy_production(&g, 1.1, kernel).unwrap();
            assert!((f.mean - sigma).abs() < 1e-12 * sigma.max(1.0), "{kernel:?}: mean {} vs Sigma {sigma}", f.mean);
        }
    }

    /// The shipped fabric at beta 2 on this grid forbids flips once `2 beta f > 11.8`, so some of
    /// its transitions have no reverse: the integral falls short of 1 by their mass, which the
    /// theorem excludes rather than the fabric violates.
    #[test]
    fn one_way_transitions_are_what_the_integral_theorem_falls_short_by() {
        let g = grid_glass(3, 3, 5);
        // The one-way mass is small -- 1.4e-7 here, the steady-state weight of the states whose
        // exits are forbidden -- and the integral falls short by exactly that much.
        let f = fluctuation(&g, 2.0, Kernel::FixedFabric).unwrap();
        assert!(f.one_way_mass > 1e-9, "the fabric at beta 2 has forbidden flips: one-way mass {:e}", f.one_way_mass);
        assert!(f.integral < 1.0 - 1e-9, "<exp(-ds)> over the reversible support = {}", f.integral);
        assert!((1.0 - f.integral - f.one_way_mass).abs() < 1e-9, "the shortfall must be the one-way mass: {} vs {}", 1.0 - f.integral, f.one_way_mass);
    }
}
