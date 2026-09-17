//! What an analogue variable costs, in bits and in joules — the exchange rate between a continuous
//! dynamical substrate and this crate's categorical one.
//!
//! # The question
//!
//! A coupled-oscillator or other continuous-state machine claims an advantage over a stochastic-bit
//! machine because its variable is continuous: one phase, it is said, carries what many bits would.
//! That claim is checkable, and this module checks it from both ends. [`crate::kuramoto`] supplies
//! the information half — a `q`-point readout of a phase costs at most `2 beta epsilon(q)` nats,
//! falling like `2^-b` in the bit depth. This module supplies the energy half, and the two together
//! settle the argument.
//!
//! # The thermal bound on an analogue variable
//!
//! Resolving `b` bits from an analogue node is limited by `kT/C` sampling noise, and the derivation
//! is short enough to write out rather than cite:
//!
//! * a full-scale sinusoid on a swing `V` has signal power `V^2 / 8`;
//! * the sampled noise power on capacitance `C` is `kT / C`, independent of `C`'s value in the sense
//!   that it is fixed once `C` is;
//! * an ideal `b`-bit quantiser has signal-to-noise ratio `(3/2) 2^{2b}`;
//! * equating them gives `C = 12 kT 2^{2b} / V^2`, so the energy to drive that node over its full
//!   swing, `C V^2`, is **`12 kT 2^{2b}` and does not depend on the voltage at all**.
//!
//! [`ANALOG_SNR_KAPPA`] is that 12, named and derived rather than borrowed, and
//! [`analog_bit_energy`] evaluates the bound. The voltage dropping out is what makes this a bound on
//! physics rather than on a process node: no supply-voltage scaling, no device scaling and no
//! architectural cleverness moves it, because it is the noise of the bath.
//!
//! # The comparison that decides it
//!
//! Erasing `b` bits digitally costs at least `b kT ln 2` (Landauer 1961; [`crate::landauer`]).
//! Carrying the same `b` bits on an analogue node costs at least `12 kT 2^{2b}`. The ratio
//!
//! ```text
//!   12 * 2^{2b} / (b ln 2)
//! ```
//!
//! is **69 at one bit** and grows without bound: 138 at two bits, 1,108 at four, 141,823 at eight.
//! [`analog_to_landauer_ratio`] returns it, and `an_analog_variable_never_undercuts_the_landauer_
//! floor` asserts it exceeds one at every depth from 1 to 32. So an analogue variable is never the
//! cheap way to carry information — it is the cheap way to carry *very little* information, and at
//! the precision where it is cheapest it is worth a handful of bits, which is what a handful of
//! stochastic bits already costs less to provide.
//!
//! That is the emulation argument in its quantitative form, and [`EmulationCost`] states it as a
//! constructive result: given a target KL budget, [`EmulationCost::for_budget`] returns the `q` and
//! the bit depth at which a clock model reproduces a continuous phase system inside that budget —
//! after which every tool in this crate applies, certificates included.
//!
//! # Where the joules actually are, which is the point of the module
//!
//! [`GeneratorBudget`] prices a published coupled-oscillator image generator end to end: `n`
//! oscillators, `s` Euler steps, a `b`-bit readout of every phase, and a conventional neural decoder
//! of `p` parameters that turns the readout into pixels. At the published shape — 16,384
//! oscillators, ten steps, eight-bit readout, a decoder of roughly 37 million parameters — the
//! arithmetic is not close:
//!
//! ```text
//!   oscillator dynamics   16,384 x 10 updates                 ~ 1 nJ  at a Z1-class 7.09 fJ update
//!   phase readout         16,384 x 12 kT 2^16                 ~ 53 pJ at the thermal floor
//!   digital decoder       37e6 MACs                           ~ 37 uJ at 1 pJ per MAC
//! ```
//!
//! The decoder is **four orders of magnitude** above everything the substrate does, and
//! `the_decoder_dominates_the_published_architecture` asserts its share exceeds 99.9%. A substrate
//! that makes the oscillator part free changes the total by less than a tenth of a percent.
//!
//! This is the same finding as the crate's Z1T decomposition, arrived at from the other side: the
//! sampler is 3% of that system's energy and the fabric around it is the rest. A thousand-fold
//! claim for a generative model whose decoder is digital is a claim about a component that was never
//! the cost. Lane B in `docs/ABSORPTION.md` — the in-fabric generator in [`crate::phasegen`], with
//! no digital decoder at all — exists because it is the only version of this architecture where the
//! substrate's efficiency could reach the total.

use crate::kuramoto::Kuramoto;
use crate::landauer::BOLTZMANN_CONSTANT;
use crate::ledger::Prices;

/// Room temperature, kelvin, as the rest of this crate uses it.
pub const ROOM_TEMPERATURE_K: f64 = 300.0;

/// The constant in the `kT/C` bound on an analogue node's energy: `E >= kappa kT 2^{2b}`.
///
/// Twelve, from equating a full-scale sinusoid's `V^2/8` signal power against `kT/C` noise at the
/// `(3/2) 2^{2b}` signal-to-noise ratio of an ideal `b`-bit quantiser, and then charging `C V^2` to
/// drive the node. The voltage cancels. Exposed as a named constant because the figure varies in the
/// literature with the convention chosen — half-swing against full, `CV^2` against `CV^2/2`, a
/// sinusoid against a uniform signal — and a bound whose constant arrives without its convention is
/// a bound nobody can check. [`analog_bit_energy_with`] takes a different one for a caller who
/// prefers another.
pub const ANALOG_SNR_KAPPA: f64 = 12.0;

/// The thermal floor on the energy of one `b`-bit analogue sample, in joules.
///
/// # Panics
///
/// If `bits` is zero or the temperature is not positive.
#[must_use]
pub fn analog_bit_energy(bits: u32, temperature_k: f64) -> f64 {
    analog_bit_energy_with(bits, temperature_k, ANALOG_SNR_KAPPA)
}

/// [`analog_bit_energy`] with an explicit convention constant.
///
/// # Panics
///
/// If `bits` is zero, the temperature is not positive, or `kappa` is not positive.
#[must_use]
pub fn analog_bit_energy_with(bits: u32, temperature_k: f64, kappa: f64) -> f64 {
    assert!(bits > 0, "a zero-bit readout resolves nothing");
    assert!(temperature_k > 0.0, "temperature must be positive, got {temperature_k}");
    assert!(kappa > 0.0, "the convention constant must be positive, got {kappa}");
    kappa * BOLTZMANN_CONSTANT * temperature_k * 2.0f64.powi(2 * bits as i32)
}

/// The Landauer cost of carrying `b` bits digitally: `b kT ln 2`, in joules.
///
/// # Panics
///
/// If the temperature is not positive.
#[must_use]
pub fn digital_bit_energy(bits: u32, temperature_k: f64) -> f64 {
    assert!(temperature_k > 0.0, "temperature must be positive, got {temperature_k}");
    f64::from(bits) * BOLTZMANN_CONSTANT * temperature_k * core::f64::consts::LN_2
}

/// How many times the Landauer floor an analogue node at `b` bits costs.
///
/// Above one at every depth, and rising like `2^{2b} / b`. The temperature cancels, so this is a
/// pure function of the bit depth — which is why it is the right form of the comparison.
///
/// # Panics
///
/// If `bits` is zero.
#[must_use]
pub fn analog_to_landauer_ratio(bits: u32) -> f64 {
    analog_bit_energy(bits, ROOM_TEMPERATURE_K) / digital_bit_energy(bits, ROOM_TEMPERATURE_K)
}

/// The bit depth and grid size at which a clock model reproduces a continuous phase system inside a
/// stated KL budget, and what that costs to read.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EmulationCost {
    /// Grid points per phase.
    pub q: usize,
    /// `ceil(log2 q)` — bits per phase.
    pub bits: u32,
    /// The KL ceiling this grid actually achieves, in nats. At or below the budget.
    pub kl_nats: f64,
    /// The thermal floor on reading one phase at this depth, joules.
    pub read_joules: f64,
}

impl EmulationCost {
    /// The cheapest grid meeting a KL budget for `sys` at inverse temperature `beta`.
    ///
    /// From [`crate::kuramoto::Kuramoto::quantisation_kl_bound`]: the ceiling is
    /// `4 pi beta * mass / q`, so `q >= 4 pi beta * mass / budget`, and the bit depth is its
    /// logarithm. Because the requirement falls like `1/q`, **one extra bit halves the KL** — the
    /// exchange rate is exponential in the reader's favour, which is why a continuous variable can
    /// always be bought out with a small number of bits.
    ///
    /// # Panics
    ///
    /// If the budget or `beta` is not positive.
    #[must_use]
    pub fn for_budget(sys: &Kuramoto, beta: f64, budget_nats: f64) -> EmulationCost {
        assert!(budget_nats > 0.0, "a KL budget must be positive, got {budget_nats}");
        assert!(beta > 0.0, "beta must be positive, got {beta}");
        let mass = sys.coupling_mass();
        let needed = 2.0 * core::f64::consts::TAU * beta * mass / budget_nats;
        let mut q = 2usize;
        let mut bits = 1u32;
        while (q as f64) < needed && bits < 60 {
            q *= 2;
            bits += 1;
        }
        EmulationCost {
            q,
            bits,
            kl_nats: sys.quantisation_kl_bound(beta, q),
            read_joules: analog_bit_energy(bits, ROOM_TEMPERATURE_K),
        }
    }
}

/// A device model for a fabric of coupled analogue oscillators whose readout is at the thermal
/// floor.
///
/// The per-update energy is the caller's, because **nobody has published one**: the oscillator
/// machines in the field ship papers and simulators, not characterised silicon, and inventing a
/// number here would produce exactly the kind of figure [`Prices::UNSTATED`] exists to refuse. What
/// this function does supply is the read price, which is not a guess: it is the `kT/C` floor at the
/// stated bit depth, and therefore a number no such device can come in under.
///
/// The write price is left unstated for the same reason as the update price. A run that reprograms
/// couplings therefore prices to `None`, which is the honest answer.
///
/// # Panics
///
/// If `bits` is zero or `e_update` is not finite and positive.
#[must_use]
pub fn oscillator_fabric_floor(bits: u32, e_update: f64) -> Prices {
    assert!(e_update > 0.0 && e_update.is_finite(), "state a positive per-update energy");
    Prices {
        e_sample: e_update,
        e_read: analog_bit_energy(bits, ROOM_TEMPERATURE_K),
        e_write: f64::NAN,
        reflash_hz_cap: None,
        source: "MODEL, and a FLOOR on only one of its three numbers: the read price is the kT/C \
                 thermal bound at the stated bit depth (12 kT 2^{2b} at 300 K), which no analogue \
                 readout can undercut. The per-update energy is the caller's, because no coupled- \
                 oscillator machine has published one. Writes are unstated, not zero.",
        evidence: crate::ledger::Evidence::Derived,
    }
}

/// The energy break-even point: the per-oscillator-update energy at which the dynamics of a run cost
/// the same as reading it out once.
///
/// Below this figure the run is a readout problem; above it, a dynamics problem. At the published
/// generative architecture — ten Euler steps, eight-bit phases — it lands at 0.33 femtojoules,
/// **below** the Z1-class Gibbs-cycle estimate of 7.09 fJ. So *at the thermal floor* the dynamics
/// dominate the substrate's own budget, and this crate's usual readout-dominance finding does
/// **not** transfer to it: one eight-bit conversion at the `kT/C` bound is 3.26 fJ, under half a
/// single published Gibbs update, so no step count from one upward makes readout the larger half.
///
/// That holds at the floor, and real converters are not at the floor. The analog compute-in-memory
/// literature, which arrives at this same `4^b` scaling from ADC circuit design rather than from
/// `kT/C` — *"the power consumption of a DAC and ADC scales exponentially with precision … steepening
/// to `4^N` once thermal noise limits necessitate larger sampling capacitors"* — anchors a 28 nm
/// converter at **30 fJ/Op at 35 dB SQNR** (about 5.7 bits) with **100 fJ/Op** as its modelled
/// ceiling. Against this module's floor those are **147x** at six bits and about **30x** at eight,
/// so the "tens of times" below is the sourced figure rather than an estimate, and at that multiple
/// the two halves are comparable again at ten steps. The honest summary is that readout dominance is a
/// property of *this crate's* fabric, which takes thousands of updates between reads, and not of an
/// architecture that reads after ten. Both halves, in any case, are dwarfed by the decoder; see
/// [`GeneratorBudget`].
///
/// # Panics
///
/// If `steps` is zero or `bits` is zero.
#[must_use]
pub fn readout_break_even(steps: u64, bits: u32) -> f64 {
    assert!(steps > 0, "a run with no steps has no dynamics to price");
    analog_bit_energy(bits, ROOM_TEMPERATURE_K) / steps as f64
}

/// An end-to-end energy account of a coupled-oscillator generative model.
///
/// Every field is an input the caller states, and [`GeneratorBudget::breakdown`] is arithmetic on
/// them. The reason it is a type rather than a function is that the *shape* of the account is the
/// argument: a substrate claim covers the first two lines and the third is where the joules are.
#[derive(Clone, Copy, Debug)]
pub struct GeneratorBudget {
    /// Oscillators in the fabric.
    pub oscillators: u64,
    /// Integration steps per generated sample.
    pub steps: u64,
    /// Bit depth of the phase readout.
    pub readout_bits: u32,
    /// Parameters of the conventional decoder that turns the readout into output.
    pub decoder_params: u64,
    /// Energy of one oscillator update, joules. The caller's number.
    pub e_update: f64,
    /// Energy of one decoder multiply-accumulate, joules. The caller's number.
    pub e_mac: f64,
}

/// Where a generated sample's energy went.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Breakdown {
    /// Joules spent integrating the oscillator dynamics.
    pub dynamics: f64,
    /// Joules spent reading the phases out, at the thermal floor.
    pub readout: f64,
    /// Joules spent in the digital decoder.
    pub decoder: f64,
}

impl Breakdown {
    /// Total joules.
    #[must_use]
    pub fn total(&self) -> f64 {
        self.dynamics + self.readout + self.decoder
    }

    /// Fractional shares `(dynamics, readout, decoder)`.
    #[must_use]
    pub fn shares(&self) -> (f64, f64, f64) {
        let t = self.total().max(f64::MIN_POSITIVE);
        (self.dynamics / t, self.readout / t, self.decoder / t)
    }

    /// The factor by which the total would improve if the substrate became **free** — dynamics and
    /// readout both to zero.
    ///
    /// Amdahl's law, applied to the claim. A number near one is a claim about a component that was
    /// never the cost.
    #[must_use]
    pub fn free_substrate_speedup(&self) -> f64 {
        let t = self.total();
        if self.decoder <= 0.0 {
            return f64::INFINITY;
        }
        t / self.decoder
    }
}

impl GeneratorBudget {
    /// The energy account of one generated sample.
    ///
    /// # Panics
    ///
    /// If the readout depth is zero.
    #[must_use]
    pub fn breakdown(&self) -> Breakdown {
        let dynamics = self.oscillators as f64 * self.steps as f64 * self.e_update;
        let readout =
            self.oscillators as f64 * analog_bit_energy(self.readout_bits, ROOM_TEMPERATURE_K);
        let decoder = self.decoder_params as f64 * self.e_mac;
        Breakdown { dynamics, readout, decoder }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::landauer::landauer_bound;

    #[test]
    fn the_thermal_bound_is_independent_of_voltage_and_quadruples_per_bit() {
        // 2^{2b} means one extra bit costs 4x, which is the whole shape of the argument.
        let a = analog_bit_energy(8, ROOM_TEMPERATURE_K);
        let b = analog_bit_energy(9, ROOM_TEMPERATURE_K);
        assert!((b / a - 4.0).abs() < 1e-12, "{b} / {a} should be exactly 4");
        // And the value is what the derivation says: 12 kT 2^16 at 300 K.
        let want = 12.0 * BOLTZMANN_CONSTANT * 300.0 * 65_536.0;
        assert!((a - want).abs() < 1e-30, "got {a}, derivation says {want}");
    }

    #[test]
    fn an_analog_variable_never_undercuts_the_landauer_floor() {
        // The comparison that settles the substrate argument. Every depth, not a chosen one.
        for b in 1..=32u32 {
            let r = analog_to_landauer_ratio(b);
            assert!(r > 1.0, "an analogue node at {b} bits should not beat Landauer, got {r}");
        }
        // The headline figures quoted in the module documentation, checked rather than asserted.
        assert!((analog_to_landauer_ratio(1) - 69.249).abs() < 0.01);
        assert!((analog_to_landauer_ratio(2) - 138.499).abs() < 0.01);
        assert!((analog_to_landauer_ratio(4) - 1_107.99).abs() < 0.05);
        assert!((analog_to_landauer_ratio(8) - 141_822.7).abs() < 1.0);
        // Monotone from two bits up: the gap only widens.
        for b in 2..32u32 {
            assert!(analog_to_landauer_ratio(b + 1) > analog_to_landauer_ratio(b));
        }
    }

    #[test]
    fn one_extra_bit_halves_the_emulation_kl() {
        let n = 6;
        let mut k = vec![0.0; n * n];
        for i in 0..n {
            for j in (i + 1)..n {
                k[i * n + j] = 0.4;
                k[j * n + i] = 0.4;
            }
        }
        let sys = Kuramoto::gradient(k, n).expect("symmetric fixture");
        let loose = EmulationCost::for_budget(&sys, 1.0, 1.0);
        let tight = EmulationCost::for_budget(&sys, 1.0, 0.5);
        assert_eq!(tight.bits, loose.bits + 1, "halving the budget should cost exactly one bit");
        assert!(loose.kl_nats <= 1.0, "the returned grid must meet the budget it was asked for");
        assert!(tight.kl_nats <= 0.5);
        // And the readout price of that extra bit is a factor of four.
        assert!((tight.read_joules / loose.read_joules - 4.0).abs() < 1e-9);
    }

    #[test]
    fn a_modest_budget_needs_only_a_handful_of_bits() {
        // The constructive half of the emulation claim: continuous phases are bought out cheaply.
        let n = 4;
        let mut k = vec![0.0; n * n];
        for i in 0..n {
            for j in (i + 1)..n {
                k[i * n + j] = 0.25;
                k[j * n + i] = 0.25;
            }
        }
        let sys = Kuramoto::gradient(k, n).expect("symmetric fixture");
        let cost = EmulationCost::for_budget(&sys, 1.0, 0.1);
        assert!(cost.bits <= 12, "a 0.1-nat emulation should not need {} bits", cost.bits);
    }

    #[test]
    fn the_decoder_dominates_the_published_architecture() {
        // 16,384 oscillators, ten Euler steps, eight-bit phase readout, a ~37M-parameter decoder.
        // Oscillator updates are priced at the Z1-class SPICE Gibbs cycle, which is the most
        // favourable published figure in the field and still generous to a device nobody has
        // characterised. Decoder MACs at 1 pJ, which is a competent digital accelerator.
        let g = GeneratorBudget {
            oscillators: 16_384,
            steps: 10,
            readout_bits: 8,
            decoder_params: 37_000_000,
            e_update: 7.09e-15,
            e_mac: 1e-12,
        };
        let b = g.breakdown();
        let (dyn_share, read_share, dec_share) = b.shares();
        assert!(dec_share > 0.999, "decoder share was only {dec_share}");
        assert!(dyn_share < 1e-3 && read_share < 1e-3);
        // Amdahl: make the entire substrate free and the total barely moves.
        let speedup = b.free_substrate_speedup();
        assert!(
            speedup < 1.01,
            "a free substrate should buy under 1% here, it bought {speedup}x"
        );
    }

    #[test]
    fn readout_dominance_does_not_transfer_at_the_thermal_floor() {
        // Worth stating precisely rather than repeating the crate's usual finding. At the kT/C
        // bound an eight-bit conversion is 3.26 fJ -- under half of one published Gibbs update --
        // so the break-even per-update energy sits below 7.09 fJ at EVERY step count from one up,
        // and the dynamics are the larger half throughout.
        for steps in [1u64, 10, 100, 1_000] {
            let be = readout_break_even(steps, 8);
            assert!(
                be < 7.09e-15,
                "at {steps} steps the break-even {be} should sit below the Z1-class cycle"
            );
        }
        // A real converter is not at the floor. At thirty times the bound -- a competent eight-bit
        // SAR -- readout and a ten-step evolution are the same order, which is the regime the
        // published architecture actually lives in.
        let real = 30.0 * analog_bit_energy(8, ROOM_TEMPERATURE_K);
        let dynamics = 10.0 * 7.09e-15;
        assert!(real > dynamics, "a realistic converter should outweigh ten updates");
        assert!(real / dynamics < 2.0, "and not by more than a small factor: {}", real / dynamics);
    }

    #[test]
    fn the_fabric_floor_prices_reads_and_refuses_writes() {
        let p = oscillator_fabric_floor(8, 1e-15);
        assert!(p.e_read > 0.0 && p.e_read.is_finite());
        assert!(!p.is_stated(), "an unstated write price must make the price set incomplete");
        // A run that only samples and reads is priceable; one that writes is not.
        let mut l = crate::ledger::Ledger { samples: 1_000, reads: 100, writes: 0 };
        assert!(l.joules(&p).is_some());
        l.writes = 1;
        assert!(l.joules(&p).is_none(), "an unstated write price must refuse, not assume zero");
    }

    #[test]
    fn the_landauer_bound_this_module_compares_against_is_the_crate_s_own() {
        // One floor, one place. If `landauer` ever changes its constant this test fails here too.
        let want = landauer_bound(ROOM_TEMPERATURE_K);
        assert!((digital_bit_energy(1, ROOM_TEMPERATURE_K) - want).abs() < 1e-30);
    }
}
