//! What a number format can and cannot buy: Mitchell's logarithm, measured, and the floor it
//! does not move.
//!
//! # The bet this prices
//!
//! A logarithmic number system stores `x` as `log2 x`, so a multiply becomes an add. Adders are
//! cheap and multipliers are not — the published range is roughly five to thirty times — so an
//! accelerator built this way claims a large win on the one operation a neural network does most.
//! The technique is not new: Mitchell, *Computer multiplication and division using binary
//! logarithms*, IRE Transactions on Electronic Computers EC-11(4) (1962) 512–517.
//!
//! Mitchell's observation is that a binary float already carries its own logarithm. Write
//! `x = 2^E * m` with `m` in `[1, 2)`. Then
//!
//! ```text
//!   log2(x) = E + log2(m)  ~=  E + (m - 1),
//! ```
//!
//! and `m - 1` **is the mantissa field**. No table, no circuit: drop the leading one and read the
//! bits as a fraction. The inverse is the same trick backwards. So the conversion is free and the
//! multiply is an add.
//!
//! # What it costs, in closed form
//!
//! The approximation is a chord across `log2` on `[1, 2)`, and its error
//! `e(m) = (m - 1) - log2(m)` vanishes at both ends. It is extremal where `e'(m) = 1 - 1/(m ln 2)`
//! vanishes, at `m = 1/ln 2`, giving [`MAX_LOG_ERROR`] — about `0.0861` in exponent units.
//!
//! For a *multiply* the worst case is sharper and exactly rational. At `a = b = 3/2` the true
//! product is `9/4`, Mitchell gives `1/2 + 1/2 = 1` and reconstructs `2`, and
//!
//! ```text
//!   (2 - 9/4) / (9/4)  =  -1/9.
//! ```
//!
//! [`MAX_MUL_RELATIVE_ERROR`] is that `1/9` — **11.111%**, always an underestimate. A dense scan
//! confirms no pair does worse, and the test pins both the value and the point.
//!
//! # What correcting it costs, measured
//!
//! Eleven percent is not FP16. Closing that gap is where a vendor's intellectual property lives,
//! and the shape of the bill can be measured without knowing anyone's circuit. Two corrections are
//! needed and they are different functions — the forward `log2(m) - (m - 1)` and the inverse
//! `2^f - (1 + f)`. [`Correction`] tabulates each at `bits` of mantissa and measures what remains:
//!
//! | table bits | entries per table | worst relative multiply error |
//! |---|---|---|
//! | 0 (raw Mitchell) | — | 11.111% |
//! | 4 | 16 | 2.077% |
//! | 6 | 64 | 0.624% |
//! | 8 | 256 | 0.163% |
//! | **10** | **1,024** | **0.038%** |
//! | 12 | 4,096 | 0.011% |
//!
//! So reaching the FP16-level `0.08%` a logarithmic accelerator needs takes about **ten bits of
//! correction table in each direction**, and the error falls by roughly a factor of two per bit.
//! Correcting only the operands and not the reconstruction — the obvious first implementation —
//! **plateaus near 1.2% and never arrives**, which is worth knowing before concluding a correction
//! is cheap.
//!
//! **The literature agrees on the shape and on the size of the prize.** Parhami states the floor
//! qualitatively — *"LNS addition and subtraction require lookup tables whose size grows
//! exponentially with the logarithm width"* — and the table above is that sentence with numbers in
//! it. On the payoff: every published LNS result that names a **tuned integer datapath** as its
//! baseline is worth tens of percent rather than multiples (53.5% energy against fixed-point in one
//! 2025 study, 42.61% against integer quantisation in another), and the most direct comparison of
//! all finds an 8-bit log-float multiply-add at *"0.96x the power and 1.12x the area of 8/32-bit
//! integer multiply-add"* — a tie with `int8`, from Johnson's `deepfloat` work at FAIR. The
//! multiples reported elsewhere come from baselines other than a tuned integer unit.
//!
//! **Two companies took this thesis at silicon, and one left.** Lemurian Labs' PAL — "parallel
//! adaptive logarithm", which extends the logarithmic number system with multiple bases and
//! interleaved exponents — was the other funded attempt; the company turned software-first in 2024
//! and now builds a hardware-agnostic stack instead. The surviving silicon effort has **taped out,
//! not shipped**: beta is stated for Q1 2027, and every performance figure it publishes is labelled
//! by its own whitepaper *"based on modeling and simulation and will be verified upon silicon
//! availability."* A format argument that has not yet met a wafer is exactly the kind this module
//! exists to price from first principles instead.
//!
//! That is the number this module exists to supply. Whether two 1,024-entry tables cost back the
//! multiplier they saved depends on how they are shared across a systolic array, which is a
//! circuit question nobody outside the vendor can answer — but it is the question, and it is the
//! same shape as every other compression win in this project: the saving is real and **the unpack
//! is the bound**.
//!
//! # The floor it does not move
//!
//! None of this touches the thermodynamics. Landauer's bound is `b kT ln 2` per `b` bits erased
//! and it does not care whether those bits encode a mantissa, a logarithm, a posit or a tally —
//! [`format_floor`] is that quantity and `the_floor_does_not_care_what_the_bits_encode` asserts
//! the two agree exactly. A format change moves an implementation along the enormous gap between
//! present silicon and the floor; it does not lower the floor.
//!
//! Set against [`crate::precision::analog_bit_energy`], which prices an *analogue* variable at
//! `12 kT 2^{2b}`, the three representations line up on one axis:
//!
//! | representation | floor per value at `b` bits |
//! |---|---|
//! | digital, any encoding — LNS included | `b kT ln 2`, **linear** in `b` |
//! | an analogue node | `12 kT 2^{2b}`, **exponential** in `b` |
//!
//! A format argument is an argument about the first row's coefficient. It is a real argument and a
//! bounded one.

use crate::landauer::BOLTZMANN_CONSTANT;
use crate::round::sum_up;

/// The largest error Mitchell's approximation makes on `log2`, in exponent units.
///
/// `e(m) = (m - 1) - log2(m)` at `m = 1/ln 2`, where `e'` vanishes. Mitchell always
/// **underestimates** the logarithm, so the stored constant is the magnitude.
pub const MAX_LOG_ERROR: f64 = 0.086_071_332_055_934_9;

/// The mantissa at which [`MAX_LOG_ERROR`] is attained: `1 / ln 2`.
pub const WORST_MANTISSA: f64 = core::f64::consts::LOG2_E;

/// The largest relative error a Mitchell **multiply** makes: exactly `1/9`.
///
/// Attained at `a = b = 3/2`, where the true product `9/4` is reconstructed as `2`. Rational, not
/// a fit: the module documentation works it out in three lines.
pub const MAX_MUL_RELATIVE_ERROR: f64 = 1.0 / 9.0;

/// Split `x > 0` into `(E, m)` with `x = 2^E * m` and `m` in `[1, 2)`.
///
/// # Panics
///
/// If `x` is not finite and strictly positive.
#[must_use]
pub fn decompose(x: f64) -> (i32, f64) {
    assert!(x > 0.0 && x.is_finite(), "a logarithm needs a positive finite argument, got {x}");
    let e = x.log2().floor();
    let m = x / 2.0f64.powf(e);
    // Guard the boundary: rounding can leave the mantissa at exactly 2.0, which is not in [1, 2)
    // and would put the exponent one out.
    if m >= 2.0 { (e as i32 + 1, m / 2.0) } else { (e as i32, m) }
}

/// Mitchell's approximation to `log2(x)`: the exponent plus the mantissa field, read as a fraction.
///
/// # Panics
///
/// If `x` is not finite and strictly positive.
#[must_use]
pub fn mitchell_log2(x: f64) -> f64 {
    let (e, m) = decompose(x);
    f64::from(e) + (m - 1.0)
}

/// Mitchell's approximation to `2^y`: the inverse trick, a fraction read back as a mantissa.
#[must_use]
pub fn mitchell_exp2(y: f64) -> f64 {
    let e = y.floor();
    let f = y - e;
    2.0f64.powf(e) * (1.0 + f)
}

/// A multiply done the way a logarithmic datapath does it: two conversions and an **add**.
///
/// # Panics
///
/// If either argument is not finite and strictly positive.
#[must_use]
pub fn mitchell_mul(a: f64, b: f64) -> f64 {
    mitchell_exp2(mitchell_log2(a) + mitchell_log2(b))
}

/// A tabulated correction to Mitchell's approximation, at a stated table width.
///
/// Two tables, because the forward and inverse errors are different functions and correcting only
/// the operands leaves the reconstruction wrong — the plateau the module documentation names.
/// Each is indexed by the top `bits` of the mantissa and holds the error at the bucket's midpoint.
#[derive(Clone, Debug)]
pub struct Correction {
    /// Mantissa bits the tables are indexed by.
    pub bits: u32,
    /// `log2(m) - (m - 1)` at each bucket's midpoint, for `m` in `[1, 2)`.
    pub forward: Vec<f64>,
    /// `2^f - (1 + f)` at each bucket's midpoint, for `f` in `[0, 1)`.
    pub inverse: Vec<f64>,
}

impl Correction {
    /// Build both tables at `bits` of index.
    ///
    /// # Panics
    ///
    /// If `bits` is zero or above 20, where the table stops being a table.
    #[must_use]
    pub fn new(bits: u32) -> Correction {
        assert!(bits > 0 && bits <= 20, "a correction table of {bits} bits is not one");
        let n = 1usize << bits;
        let mut forward = Vec::with_capacity(n);
        let mut inverse = Vec::with_capacity(n);
        for i in 0..n {
            let mid = (i as f64 + 0.5) / n as f64;
            let m = 1.0 + mid;
            forward.push(m.log2() - (m - 1.0));
            inverse.push(2.0f64.powf(mid) - (1.0 + mid));
        }
        Correction { bits, forward, inverse }
    }

    /// Entries in each table.
    #[must_use]
    pub fn entries(&self) -> usize {
        1usize << self.bits
    }

    /// Which bucket a fraction in `[0, 1)` falls in.
    fn bucket(&self, frac: f64) -> usize {
        let n = self.entries();
        let i = (frac * n as f64) as usize;
        if i >= n { n - 1 } else { i }
    }

    /// Corrected `log2`.
    ///
    /// # Panics
    ///
    /// If `x` is not finite and strictly positive.
    #[must_use]
    pub fn log2(&self, x: f64) -> f64 {
        let (e, m) = decompose(x);
        f64::from(e) + (m - 1.0) + self.forward[self.bucket(m - 1.0)]
    }

    /// Corrected `2^y`.
    #[must_use]
    pub fn exp2(&self, y: f64) -> f64 {
        let e = y.floor();
        let f = y - e;
        2.0f64.powf(e) * ((1.0 + f) + self.inverse[self.bucket(f)])
    }

    /// A corrected multiply: still two conversions and an add, plus two table reads.
    ///
    /// # Panics
    ///
    /// If either argument is not finite and strictly positive.
    #[must_use]
    pub fn mul(&self, a: f64, b: f64) -> f64 {
        self.exp2(self.log2(a) + self.log2(b))
    }

    /// The worst relative multiply error over an `n`-by-`n` grid of mantissas.
    ///
    /// A scan rather than a bound: the corrected error has no closed form the way raw Mitchell's
    /// does, and a measured worst case over a dense grid is the honest substitute.
    ///
    /// # Panics
    ///
    /// If `n` is zero.
    #[must_use]
    pub fn worst_relative_error(&self, n: usize) -> f64 {
        assert!(n > 0, "a scan of no points measures nothing");
        let mut worst = 0.0f64;
        for i in 0..n {
            let a = 1.0 + i as f64 / n as f64;
            for j in 0..n {
                let b = 1.0 + j as f64 / n as f64;
                let got = self.mul(a, b);
                worst = worst.max((got - a * b).abs() / (a * b));
            }
        }
        worst
    }
}

/// The worst relative error a **raw** Mitchell multiply makes over an `n`-by-`n` mantissa grid.
///
/// The comparator for [`Correction::worst_relative_error`], and the scan that confirms
/// [`MAX_MUL_RELATIVE_ERROR`] is the maximum rather than merely a value.
///
/// # Panics
///
/// If `n` is zero.
#[must_use]
pub fn worst_mitchell_error(n: usize) -> f64 {
    assert!(n > 0, "a scan of no points measures nothing");
    let mut worst = 0.0f64;
    for i in 0..n {
        let a = 1.0 + i as f64 / n as f64;
        for j in 0..n {
            let b = 1.0 + j as f64 / n as f64;
            let got = mitchell_mul(a, b);
            worst = worst.max((got - a * b).abs() / (a * b));
        }
    }
    worst
}

/// Landauer's floor for a `b`-bit value, in joules at `temperature_k`: `b kT ln 2`.
///
/// **The encoding does not appear.** A logarithmic word, a float, a posit and a tally of the same
/// width erase the same number of bits and cost the same at the floor. This function exists to make
/// that statement executable, because the whole of a number-format efficiency argument is about the
/// distance between an implementation and this quantity — never about the quantity itself.
///
/// # Panics
///
/// If the temperature is not positive.
#[must_use]
pub fn format_floor(bits: u32, temperature_k: f64) -> f64 {
    assert!(temperature_k > 0.0, "temperature must be positive, got {temperature_k}");
    f64::from(bits) * BOLTZMANN_CONSTANT * temperature_k * core::f64::consts::LN_2
}

/// How many multiplier-equivalents a format saving is worth, given what fraction of a device's
/// energy the multipliers were.
///
/// Amdahl, in the one form that matters for a datapath claim: if multipliers are `share` of the
/// energy and the format makes them `factor` times cheaper, the whole device improves by this much.
/// Setting `factor` to infinity — a free multiplier — gives the ceiling on any format argument, and
/// that ceiling is `1 / (1 - share)` no matter how good the format is.
///
/// # Panics
///
/// If `share` is outside `[0, 1]` or `factor` is not above zero.
#[must_use]
pub fn device_speedup(share: f64, factor: f64) -> f64 {
    assert!((0.0..=1.0).contains(&share), "a share must be a fraction, got {share}");
    assert!(factor > 0.0, "a saving factor must be positive, got {factor}");
    let terms = [1.0 - share, share / factor];
    1.0 / sum_up(&terms)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The worst multiply is exactly one ninth, at three halves, and nothing does worse.
    #[test]
    fn mitchells_worst_multiply_is_exactly_one_ninth() {
        let got = mitchell_mul(1.5, 1.5);
        assert!((got - 2.0).abs() < 1e-15, "Mitchell should reconstruct exactly 2, got {got}");
        let rel = (got - 2.25) / 2.25;
        assert!((rel + 1.0 / 9.0).abs() < 1e-15, "the worst case should be -1/9, got {rel}");
        assert!((MAX_MUL_RELATIVE_ERROR - 1.0 / 9.0).abs() < 1e-18);
        // And it IS the worst: a dense scan finds nothing above it.
        let scanned = worst_mitchell_error(600);
        assert!(
            scanned <= MAX_MUL_RELATIVE_ERROR + 1e-12,
            "a scan found {scanned}, above the claimed maximum"
        );
        assert!(scanned > 0.9 * MAX_MUL_RELATIVE_ERROR, "the scan should approach it, got {scanned}");
    }

    /// The log error peaks where the derivative vanishes, and the constant is that value.
    #[test]
    fn the_log_error_peaks_where_the_derivative_vanishes() {
        let m = WORST_MANTISSA;
        assert!((m - 1.0 / core::f64::consts::LN_2).abs() < 1e-15, "1/ln2 is the stationary point");
        let e = (m - 1.0) - m.log2();
        assert!((e.abs() - MAX_LOG_ERROR).abs() < 1e-12, "got {e}, constant says {MAX_LOG_ERROR}");
        assert!(e < 0.0, "Mitchell underestimates the logarithm");
        // Nothing in [1, 2) beats it.
        for i in 0..=10_000 {
            let x = 1.0 + f64::from(i) / 10_000.0;
            if x >= 2.0 {
                break;
            }
            let here = ((x - 1.0) - x.log2()).abs();
            assert!(here <= MAX_LOG_ERROR + 1e-12, "m = {x} gave {here}");
        }
    }

    /// Correcting the operands but not the reconstruction plateaus, which is the trap.
    ///
    /// A first implementation corrects each input's logarithm and stops, because that is where the
    /// error obviously is. The reconstruction carries its own, different error, and leaving it
    /// uncorrected puts a floor near a percent that more table bits do not move.
    #[test]
    fn correcting_only_the_operands_does_not_converge() {
        let mut previous = f64::INFINITY;
        let mut stalled = 0;
        for bits in [4u32, 8, 12] {
            let c = Correction::new(bits);
            // operands corrected, reconstruction left raw
            let mut worst = 0.0f64;
            for i in 0..300 {
                let a = 1.0 + f64::from(i) / 300.0;
                for j in 0..300 {
                    let b = 1.0 + f64::from(j) / 300.0;
                    let got = mitchell_exp2(c.log2(a) + c.log2(b));
                    worst = worst.max((got - a * b).abs() / (a * b));
                }
            }
            if worst > 0.5 * previous {
                stalled += 1;
            }
            previous = worst;
            assert!(worst > 0.005, "half-corrected error {worst} at {bits} bits fell below 0.5%");
        }
        assert!(stalled >= 1, "the half correction was expected to stop improving, and did not");
    }

    /// Correcting both directions converges, and reaching FP16 accuracy takes ten bits.
    ///
    /// The measured ladder is 2.077%, 0.624%, 0.163%, 0.038% at 4, 6, 8 and 10 bits — roughly a
    /// factor of two per bit. Ten is where it crosses the 0.08% a logarithmic accelerator needs to
    /// claim FP16 parity, and that is the size of the thing a vendor has to pay for.
    #[test]
    fn correcting_mitchell_to_fp16_accuracy_costs_ten_bits_of_table() {
        let coarse = Correction::new(4).worst_relative_error(400);
        let mid = Correction::new(8).worst_relative_error(400);
        let ten = Correction::new(10).worst_relative_error(400);
        assert!(coarse > mid && mid > ten, "more table must mean less error: {coarse} {mid} {ten}");
        assert!(ten < 0.0008, "ten bits should reach FP16 level, got {ten}");
        assert!(mid > 0.0008, "eight bits should NOT reach it, got {mid} -- else ten is not the answer");
        assert_eq!(Correction::new(10).entries(), 1024);
        // and it is a real correction, not a rounding artefact of the scan
        assert!(worst_mitchell_error(400) > 100.0 * ten, "the correction bought two orders");
    }

    /// The floor does not care what the bits encode.
    #[test]
    fn the_floor_does_not_care_what_the_bits_encode() {
        // An LNS word and a float of the same width cost the same at the floor, exactly.
        for b in [4u32, 8, 16, 32] {
            let lns = format_floor(b, 300.0);
            let fp = format_floor(b, 300.0);
            assert_eq!(lns, fp, "two encodings of {b} bits must share a floor");
            assert!((lns - crate::precision::digital_bit_energy(b, 300.0)).abs() < 1e-30);
        }
        // Fewer bits wins LINEARLY. An analogue node at the same depth loses EXPONENTIALLY, which
        // is the contrast the whole axis exists to make.
        assert!((format_floor(8, 300.0) / format_floor(16, 300.0) - 0.5).abs() < 1e-12);
        let analog_ratio = crate::precision::analog_bit_energy(16, 300.0)
            / crate::precision::analog_bit_energy(8, 300.0);
        assert!(analog_ratio > 60_000.0, "analogue should blow up by 2^16, got {analog_ratio}");
    }

    /// Amdahl's ceiling on any format argument, whatever the format.
    #[test]
    fn a_free_multiplier_is_still_bounded_by_what_it_was_a_share_of() {
        // A five-fold cheaper multiplier that was half the energy buys 1.67x, not 5x.
        let five = device_speedup(0.5, 5.0);
        assert!((five - 1.0 / 0.6).abs() < 1e-12, "got {five}");
        // Free is the ceiling, and the ceiling is 1/(1 - share) no matter the factor.
        let free = device_speedup(0.5, f64::INFINITY);
        assert!((free - 2.0).abs() < 1e-12, "a free multiplier at half the energy is 2x, got {free}");
        assert!(device_speedup(0.3, 1e9) < 1.0 / 0.7 + 1e-9);
        // And a format that touches a tenth of the budget cannot buy more than 1.12x, ever.
        assert!(device_speedup(0.1, f64::INFINITY) < 1.12);
    }

    #[test]
    fn decomposition_round_trips_and_holds_the_boundary() {
        for x in [1.0f64, 1.5, 2.0, 3.999, 1e-9, 1e9, core::f64::consts::PI] {
            let (e, m) = decompose(x);
            assert!((1.0..2.0).contains(&m), "mantissa {m} out of range for {x}");
            let back = 2.0f64.powi(e) * m;
            assert!((back - x).abs() <= 1e-12 * x, "{x} came back as {back}");
        }
        // Powers of two are the boundary case and must give mantissa exactly one.
        let (e, m) = decompose(8.0);
        assert_eq!(e, 3);
        assert!((m - 1.0).abs() < 1e-15);
        assert!((mitchell_log2(8.0) - 3.0).abs() < 1e-15, "Mitchell is EXACT on powers of two");
    }
}
