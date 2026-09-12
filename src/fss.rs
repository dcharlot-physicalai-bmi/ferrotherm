//! Finite-size scaling — locating a phase transition, and measuring its exponents, from lattices
//! that are every one of them too small to have one.
//!
//! Binder, "Finite size scaling analysis of Ising model block distribution functions", Z. Phys. B
//! **43**, 119 (1981); Onsager, Phys. Rev. **65**, 117 (1944); Yang, Phys. Rev. **85**, 808 (1952).
//! The collapse quality measure is Houdayer & Hartmann, Phys. Rev. B **70**, 014418 (2004), after
//! Bhattacharjee & Seno, J. Phys. A **34**, 6375 (2001). The fit is generalised pattern search
//! (Kolda, Lewis & Torczon, SIAM Review **45**, 385 (2003)).
//!
//! # Why this module exists
//!
//! A finite system has no singularity. Every observable on an `L × L` lattice is an analytic
//! function of temperature, so nothing this crate can sample is a phase transition, and reading
//! `T_c` off "where the magnetisation drops" gives a different answer for every `L`. This crate
//! quotes Onsager throughout — [`crate::ising::onsager_m`] is one of its oldest oracles — and had
//! no machinery for the thing Onsager's exact solution is usually USED for: recovering the
//! infinite-volume critical point, and the exponents, from a handful of small lattices.
//!
//! # The Binder cumulant, and which normalisation this is
//!
//! With `m = (1/N) Σ s_i` the magnetisation per spin,
//!
//! ```text
//!     U_4 = 1 − ⟨m⁴⟩ / (3 ⟨m²⟩²)
//! ```
//!
//! Two exact limits fall out of that definition, and they are what make it useful:
//!
//! * **Ordered.** The distribution of `m` concentrates on `±m₀`, so `⟨m⁴⟩ = ⟨m²⟩² = m₀⁴` and
//!   `U_4 → 2/3`.
//! * **Disordered.** `m` goes Gaussian about zero, so `⟨m⁴⟩ → 3⟨m²⟩²` and `U_4 → 0`. For `N`
//!   genuinely INDEPENDENT spins the value is not merely small but **exactly `2/(3N)`**, because
//!   `⟨M⁴⟩ = 3N² − 2N` for a sum of `N` Rademacher variables. That closed form is the oracle in
//!   `the_binder_cumulant_of_free_spins_is_exactly_two_over_three_n`, and it is what the `3` in the
//!   definition stands or falls by.
//!
//! Both limits are reached in a real Swendsen–Wang run on an `8 × 8` lattice: `0.66585` at
//! `T = 1.5` (that is `2/3 − 0.0008`, and the spread over six seeds is `6e-5`) and `0.019…0.055` at
//! `T = 6` against a free-spin floor of `2/(3·64) = 0.0104`.
//!
//! `U_4` is dimensionless, and — Binder's observation — at the critical point it is also
//! **size-independent** to leading order, because the scaling form makes `m L^{β/ν}` a function of
//! `(T − T_c) L^{1/ν}` alone and the ratio `⟨m⁴⟩/⟨m²⟩²` cancels the prefactor. Curves of `U_4`
//! against `T` for different `L` therefore pass through one point, and that point is `T_c`.
//!
//! `⟨m⁴⟩ ≥ ⟨m²⟩²` for any distribution whatever (Cauchy–Schwarz), so `U_4 ≤ 2/3` always.
//! [`Curve::new`] enforces it: a pair of moments that violates it did not come from one sample of
//! one observable — two different draw counts, say — and is refused rather than turned into a
//! cumulant above the ordered limit.
//!
//! # What the crossing actually costs, measured
//!
//! The crossing is `T_c` only in the limit. At finite `L` it drifts by the corrections to scaling,
//! and the size of that drift is the honest resolution of the method. Measured on the periodic
//! square lattice with **no Monte Carlo error anywhere in it** — exact moments by row transfer
//! matrix, itself pinned against [`crate::ising::exact_boltzmann`] at `L = 3, 4`:
//!
//! | pair | crossing `T` | error against `2/ln(1+√2) = 2.269185` | `U*` |
//! |---|---|---|---|
//! | `L = 3, 4` | 2.169236 | −4.40% | 0.62924 |
//! | `L = 4, 5` | 2.212409 | −2.50% | 0.62430 |
//! | `L = 5, 6` | 2.233094 | −1.59% | 0.62122 |
//! | `L = 6, 7` | 2.244747 | −1.08% | 0.61907 |
//!
//! Monotone, from below, onto Onsager. `U*` marches the same way onto `0.6107`, the universal value
//! for this class on a periodic square (Kamieniarz & Blöte, J. Phys. A **26**, 201 (1993)).
//!
//! **The mean over pairs is the number not to quote.** [`Transition::t_c`] averages every pair and
//! the small ones drag it down; [`Transition::largest`] is the least-corrected estimate and
//! [`Transition::spread`] is the drift, which does not shrink by taking more draws.
//!
//! # Data collapse, and the normalisation without which the fit runs away
//!
//! The scaling hypothesis is `m(L, T) = L^{−β/ν} f((T − T_c) L^{1/ν})`, so plotting
//! `y = m L^{β/ν}` against `x = (T − T_c) L^{1/ν}` should put every size on one curve.
//! [`collapse_residual`] scores that: each point is compared against the linear interpolation of
//! the OTHER sizes' curves at the same `x`, and the misfit is divided by the spread of `y` itself,
//!
//! ```text
//!     S = Σ (y_i − Y_i)² / Σ (y_i − ȳ)²
//! ```
//!
//! **The division is not cosmetic and the bare sum is unusable.** `y = m L^{β/ν}`, so driving
//! `β/ν` negative sends every `y` towards zero and a bare `Σ(y − Y)²` to zero with it: the best
//! collapse would be the one that flattens the data to nothing. Measured on the exact `L = 3…7`
//! set, at `β/ν = −5` the largest `|y|` is `3.3e-3` times the largest at Onsager's exponents, so a
//! bare sum of squares is about `1.1e-5` of its honest value — while `S`, which divides that
//! shrinkage out of both ends, goes the other way and rises by a factor of **90**.
//!
//! `S = 0` is an exact collapse and `S = 1` means the scaled points are no closer to each other
//! than to their own mean. The numerator accumulates through [`crate::round::sum_up`] and the
//! denominator through [`crate::round::sum_down`], so the reported `S` is an UPPER bound on the
//! exact one and `S ≤ tol` really implies a collapse that good. At the magnitudes this module
//! works at that guard is far below the interpolation error and no test here can see it; it is
//! carried because [`crate::bound::forest`] shipped a "lower bound" that sat above its own optimum
//! for want of exactly this, and it is recorded as invisible rather than claimed as load-bearing.
//!
//! # Two collapses, and only one of them measures `ν` at small `L`
//!
//! [`Observable::Cumulant`] collapses `U_4`, which carries no power of `L` at all, so that fit
//! contains `T_c` and `ν` and nothing else. [`Observable::OrderParameter`] collapses
//! `√⟨m²⟩ L^{β/ν}` and has `β` as well. On the same exact `L = 4…7` data:
//!
//! | fit | result | against exact |
//! |---|---|---|
//! | cumulant, `T_c` and `ν` free | `ν = 1.055`, `T_c = 2.2268` | `+5.5%`, `−1.9%` |
//! | order parameter, all three free | `ν = 0.782`, `β = 0.090` | **`−21.8%`**, `−28%` |
//! | order parameter, `β` alone at exact `T_c`, `ν` | `β = 0.1172` | `−6.2%` |
//!
//! The middle row is not the optimiser failing. At `ν = 0.782` the order-parameter residual is
//! `8.9e-5` against `6.2e-3` at Onsager's own exponents — seventy times better — so on lattices
//! this small the data genuinely collapses better on the wrong exponent. `β/ν = 1/8` separates
//! `L = 4` from `L = 7` by only `(7/4)^{1/8} = 7%` vertically, and the fit buys back an error in
//! `ν` with a small change in `β`. **So do it in two stages**: `T_c` and `ν` from the cumulant,
//! then `β` from the order parameter with those pinned ([`FitOptions`] freezes a coordinate with a
//! step of zero).
//!
//! `γ` is not fitted at all. [`Scaling::gamma`] derives it from the other two by hyperscaling,
//! `γ = d·ν − 2β`, which at the exact 2D values returns `7/4` — Onsager's own susceptibility
//! exponent — with no arithmetic error at all. Fitting it independently from the same collapse
//! would be reporting one number twice.
//!
//! # Why the exponents are several percent out, and why that is the right answer
//!
//! The pure scaling form is the leading term of an expansion in `L^{−ω}`, and at `L ≤ 7` the next
//! term is not small. Measured directly from the exact data at `T_c`, with no fitting involved —
//! `β/ν` from the ratio of `m_rms` between successive sizes and `1/ν` from the ratio of
//! `dU_4/dT` — the EFFECTIVE exponents between `L = 6` and `L = 7` are `0.1198` and `1.0275`,
//! already 4% and 3% from their own limits. A fit returning `ν = 1.000` from these lattices would
//! be reporting something other than what the data contains.
//!
//! # What this module does not do
//!
//! There is no `L^{−ω}` correction term in the fit, no automatic discarding of small sizes, and no
//! error bar on `T_c` beyond [`Transition::spread`]. Quoting a crossing from five small lattices to
//! four digits is the error this module is most able to cause.

use crate::round;

/// The exact critical temperature of the 2D square-lattice Ising ferromagnet at `J = 1`:
/// `T_c = 2 / ln(1 + √2) = 2.269185…`.
///
/// Onsager (1944). Computed rather than transcribed, so it carries every bit `f64` has.
#[must_use]
pub fn ising_2d_tc() -> f64 {
    2.0 / (1.0 + core::f64::consts::SQRT_2).ln()
}

/// The Binder cumulant, `U_4 = 1 − ⟨m⁴⟩ / (3 ⟨m²⟩²)`.
///
/// `None` when `⟨m²⟩` is not strictly positive or either moment is not finite — a cumulant with a
/// zero denominator is not a large number, it is an absent measurement, and `m ≡ 0` happens
/// (a graph with no spins, a clamped model) rather than being hypothetical.
#[must_use]
pub fn binder(m2: f64, m4: f64) -> Option<f64> {
    if !(m2 > 0.0) || !m4.is_finite() {
        return None;
    }
    Some(1.0 - m4 / (3.0 * m2 * m2))
}

/// Moments of the magnetisation per spin at one temperature.
///
/// `m2` and `m4` must come from the SAME set of draws of the SAME observable; [`Curve::new`] checks
/// the one consequence of that which is checkable, `⟨m⁴⟩ ≥ ⟨m²⟩²`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Point {
    /// Temperature, in the same units as the couplings (`k_B = 1`).
    pub temperature: f64,
    /// `⟨m²⟩`, with `m = (1/N) Σ s_i` — magnetisation PER SPIN, so the value is `O(1)` and not
    /// `O(N)`. A curve built from unnormalised `M` scales differently with `L` and will not
    /// collapse.
    pub m2: f64,
    /// `⟨m⁴⟩`, per spin likewise.
    pub m4: f64,
}

impl Point {
    /// Moments from the per-draw magnetisations of one run.
    ///
    /// `m[k]` is the magnetisation per spin of draw `k`, in `[−1, 1]`. The sign is irrelevant —
    /// only even moments are taken — which is what makes the cumulant usable in zero field, where
    /// a long enough chain visits both signs and `⟨m⟩` is zero by symmetry.
    ///
    /// # Errors
    ///
    /// [`Malformed::NoDraws`] on an empty slice, and [`Malformed::NotFinite`] naming the first
    /// value that is not finite.
    pub fn from_magnetisations(temperature: f64, m: &[f64]) -> Result<Point, Malformed> {
        if m.is_empty() {
            return Err(Malformed::NoDraws { temperature });
        }
        for (index, &v) in m.iter().enumerate() {
            if !v.is_finite() {
                return Err(Malformed::NotFinite { temperature, index, value: v });
            }
        }
        let n = m.len() as f64;
        let m2 = m.iter().map(|v| v * v).sum::<f64>() / n;
        let m4 = m.iter().map(|v| v * v * v * v).sum::<f64>() / n;
        Ok(Point { temperature, m2, m4 })
    }
}

/// An input that is not a scaling data set, naming what it actually was.
#[derive(Clone, Debug, PartialEq)]
pub enum Malformed {
    /// A size of zero. `L` enters the scaling form as `L^{1/ν}` and `L^{β/ν}`; zero has no powers.
    ZeroSize,
    /// A curve with nothing on it.
    NoPoints {
        /// The linear size the empty curve claimed.
        size: usize,
    },
    /// Temperatures that do not strictly increase. Interpolation and the crossing search both walk
    /// the grid in order, and an unsorted grid makes both of them silently answer for the wrong
    /// interval.
    OutOfOrder {
        /// Linear size of the offending curve.
        size: usize,
        /// Index of the point that did not advance.
        index: usize,
        /// The temperature before it.
        previous: f64,
        /// The temperature at `index`.
        temperature: f64,
    },
    /// A pair of moments no single sample can produce.
    NotAMoment {
        /// Linear size of the offending curve.
        size: usize,
        /// Where on the grid.
        temperature: f64,
        /// `⟨m²⟩` as given.
        m2: f64,
        /// `⟨m⁴⟩` as given.
        m4: f64,
    },
    /// No draws to average.
    NoDraws {
        /// The temperature the empty run claimed.
        temperature: f64,
    },
    /// A magnetisation that is not a number.
    NotFinite {
        /// The temperature of the run it came from.
        temperature: f64,
        /// Which draw.
        index: usize,
        /// What was actually there.
        value: f64,
    },
}

impl core::fmt::Display for Malformed {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Malformed::ZeroSize => write!(
                f,
                "a linear size of zero: the scaling form is L^(1/nu) and L^(beta/nu), and zero has \
                 no powers to take"
            ),
            Malformed::NoPoints { size } => {
                write!(f, "the curve for L = {size} has no points on it, so it has no crossing with anything")
            }
            Malformed::OutOfOrder { size, index, previous, temperature } => write!(
                f,
                "L = {size}: temperature {temperature} at index {index} does not exceed {previous} \
                 before it -- the crossing search and every interpolation walk this grid in order, \
                 and an unsorted grid answers for the wrong interval without saying so"
            ),
            Malformed::NotAMoment { size, temperature, m2, m4 } => write!(
                f,
                "L = {size} at T = {temperature}: <m^2> = {m2}, <m^4> = {m4}. Cauchy-Schwarz gives \
                 <m^4> >= <m^2>^2 for ANY distribution, and <m^2> > 0 for any non-degenerate one, \
                 so this pair did not come from one sample of one observable -- two different draw \
                 counts, or a fourth moment of something else"
            ),
            Malformed::NoDraws { temperature } => {
                write!(f, "no draws at T = {temperature}, so there is nothing to take moments of")
            }
            Malformed::NotFinite { temperature, index, value } => write!(
                f,
                "draw {index} at T = {temperature} has magnetisation {value}, which is not a \
                 finite number"
            ),
        }
    }
}

impl core::error::Error for Malformed {}

/// One lattice size's measured curve: moments on a strictly increasing temperature grid, with the
/// Binder cumulant and the order parameter already formed.
#[derive(Clone, Debug, PartialEq)]
pub struct Curve {
    size: usize,
    points: Vec<Point>,
    temps: Vec<f64>,
    u4: Vec<f64>,
    order: Vec<f64>,
}

impl Curve {
    /// Freeze a size's measurements, checking that they are measurements.
    ///
    /// `size` is the LINEAR size `L`, not the spin count: it is what enters `L^{1/ν}`. Passing
    /// `N = L²` fits a `ν` that is out by a factor of two and nothing else notices.
    ///
    /// # Errors
    ///
    /// [`Malformed::ZeroSize`], [`Malformed::NoPoints`], [`Malformed::OutOfOrder`] for a grid that
    /// does not strictly increase, and [`Malformed::NotAMoment`] for a pair of moments that no
    /// single sample can produce.
    pub fn new(size: usize, points: Vec<Point>) -> Result<Curve, Malformed> {
        if size == 0 {
            return Err(Malformed::ZeroSize);
        }
        if points.is_empty() {
            return Err(Malformed::NoPoints { size });
        }
        let mut temps = Vec::with_capacity(points.len());
        let mut u4 = Vec::with_capacity(points.len());
        let mut order = Vec::with_capacity(points.len());
        for (index, p) in points.iter().enumerate() {
            if !p.temperature.is_finite()
                || (index > 0 && !(p.temperature > points[index - 1].temperature))
            {
                return Err(Malformed::OutOfOrder {
                    size,
                    index,
                    previous: if index > 0 { points[index - 1].temperature } else { f64::NAN },
                    temperature: p.temperature,
                });
            }
            // Cauchy-Schwarz, with a relative slack for the rounding of two means that were each
            // accumulated over many draws. The slack is 1e-12 of the fourth moment: four orders
            // above the error of a million-term mean and far below any real violation.
            //
            // Note what is NOT checked. `m2 <= 1` holds for a magnetisation per spin and is left
            // alone deliberately: this module's arithmetic is the scaling form, which is the same
            // for any order parameter, and refusing one that happens not to be bounded by one
            // would be refusing a correct data set on a convention it never agreed to.
            let sound = p.m4 >= p.m2 * p.m2 - 1e-12 * p.m4.abs();
            let Some(u) = binder(p.m2, p.m4).filter(|_| sound) else {
                return Err(Malformed::NotAMoment {
                    size,
                    temperature: p.temperature,
                    m2: p.m2,
                    m4: p.m4,
                });
            };
            temps.push(p.temperature);
            u4.push(u);
            order.push(p.m2.sqrt());
        }
        Ok(Curve { size, points, temps, u4, order })
    }

    /// The linear size `L`.
    #[must_use]
    pub fn size(&self) -> usize {
        self.size
    }

    /// The measured moments, in grid order.
    #[must_use]
    pub fn points(&self) -> &[Point] {
        &self.points
    }

    /// The temperature grid, strictly increasing.
    #[must_use]
    pub fn temperatures(&self) -> &[f64] {
        &self.temps
    }

    /// `U_4` at each grid point.
    #[must_use]
    pub fn binder(&self) -> &[f64] {
        &self.u4
    }

    /// The order parameter used for collapse: `√⟨m²⟩`, the root-mean-square magnetisation per spin.
    ///
    /// `⟨|m|⟩` is the other common choice and carries the same exponent `β/ν`; this one is taken
    /// because it is already implied by the moments the cumulant needs, so one data set answers
    /// every question in this module and there is no second convention to get wrong.
    #[must_use]
    pub fn order_parameter(&self) -> &[f64] {
        &self.order
    }

    /// Lowest and highest temperature measured.
    #[must_use]
    pub fn temperature_range(&self) -> (f64, f64) {
        (self.temps[0], self.temps[self.temps.len() - 1])
    }

    /// `U_4` at an arbitrary temperature by linear interpolation; `None` outside the measured grid.
    ///
    /// Refusing to extrapolate is the point. A crossing located outside every measured window is
    /// an artefact of the straight line drawn past the last point, and it reads exactly like a
    /// real one.
    #[must_use]
    pub fn binder_at(&self, temperature: f64) -> Option<f64> {
        interpolate(&self.temps, &self.u4, temperature)
    }

    /// `√⟨m²⟩` at an arbitrary temperature by linear interpolation; `None` outside the grid.
    #[must_use]
    pub fn order_at(&self, temperature: f64) -> Option<f64> {
        interpolate(&self.temps, &self.order, temperature)
    }
}

/// Linear interpolation on a strictly increasing grid; `None` outside it.
fn interpolate(xs: &[f64], ys: &[f64], x: f64) -> Option<f64> {
    if xs.is_empty() || !x.is_finite() || x < xs[0] || x > xs[xs.len() - 1] {
        return None;
    }
    match xs.binary_search_by(|p| p.partial_cmp(&x).unwrap_or(core::cmp::Ordering::Equal)) {
        Ok(i) => Some(ys[i]),
        Err(i) => {
            let (x0, x1) = (xs[i - 1], xs[i]);
            let (y0, y1) = (ys[i - 1], ys[i]);
            Some(y0 + (y1 - y0) * (x - x0) / (x1 - x0))
        }
    }
}

/// Where two sizes' Binder curves meet.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Crossing {
    /// The smaller linear size.
    pub small: usize,
    /// The larger linear size.
    pub large: usize,
    /// Temperature of the crossing, from the piecewise-linear interpolants of both curves.
    pub temperature: f64,
    /// The common value `U*` there. Universal in the limit — for the 2D Ising class on a periodic
    /// square it is about `0.61` — so a crossing at a wildly different height is a sign that the
    /// two curves met for some reason other than criticality.
    pub u4: f64,
    /// The two GRID temperatures the sign change actually lies between.
    ///
    /// Carried because it is the honest resolution of the measurement: the interpolated
    /// `temperature` is one straight line's worth of detail finer than the data, and a reader
    /// comparing two crossings a tenth of a bracket apart is comparing interpolation.
    pub bracket: (f64, f64),
}

/// Why no crossing was reported, rather than a number where there is none.
#[derive(Clone, Debug, PartialEq)]
pub enum NoCrossing {
    /// Two curves of the same size cannot cross in the sense that matters — the whole content of a
    /// Binder crossing is that `L` differs.
    SameSize {
        /// The size both curves claimed.
        size: usize,
    },
    /// Two sizes measured at the same `L`, so a pair of them is not a pair.
    DuplicateSize {
        /// The repeated size.
        size: usize,
    },
    /// Fewer than two sizes: nothing to cross.
    TooFewSizes {
        /// How many curves were offered.
        sizes: usize,
    },
    /// The two temperature windows do not overlap in more than a point.
    Disjoint {
        /// The smaller lattice's window.
        small: (f64, f64),
        /// The larger lattice's window.
        large: (f64, f64),
    },
    /// `U_large − U_small` keeps one sign across the whole shared window. **This is the answer for
    /// a model with no transition**, and it is a result rather than a failure: a 1D chain's curves
    /// are ordered by size at every temperature above zero and never meet.
    NeverCrosses {
        /// The smaller linear size.
        small: usize,
        /// The larger linear size.
        large: usize,
        /// Smallest value of `U_large − U_small` on the shared window.
        lowest: f64,
        /// Largest value of it. Both on the same side of zero, which is the statement.
        highest: f64,
    },
    /// The difference changed sign more than once. Two curves meeting several times in one window
    /// is noise or a grid too coarse to resolve, and averaging the roots would hide both.
    Ambiguous {
        /// The smaller linear size.
        small: usize,
        /// The larger linear size.
        large: usize,
        /// Every root found, in temperature order.
        temperatures: Vec<f64>,
    },
}

impl core::fmt::Display for NoCrossing {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            NoCrossing::SameSize { size } => write!(
                f,
                "both curves are L = {size}: a Binder crossing is a statement about two DIFFERENT \
                 sizes, and two measurements of one size cross wherever their noise does"
            ),
            NoCrossing::DuplicateSize { size } => write!(
                f,
                "L = {size} appears twice among the curves, so the pairs are not pairs and the \
                 spread across them would count one measurement as two"
            ),
            NoCrossing::TooFewSizes { sizes } => write!(
                f,
                "{sizes} curve(s): finite-size scaling needs at least two sizes, because every \
                 single-size curve is analytic and has no transition on it to find"
            ),
            NoCrossing::Disjoint { small, large } => write!(
                f,
                "the windows {small:?} and {large:?} do not overlap, and a crossing outside both \
                 is a property of the straight line drawn past the last point"
            ),
            NoCrossing::NeverCrosses { small, large, lowest, highest } => write!(
                f,
                "U(L={large}) - U(L={small}) stays within [{lowest}, {highest}] across the shared \
                 window, all of one sign: these curves are ordered by size everywhere measured and \
                 do not meet. That is what a model with no finite-temperature transition looks like"
            ),
            NoCrossing::Ambiguous { small, large, temperatures } => write!(
                f,
                "U(L={large}) - U(L={small}) changes sign {} times, at {temperatures:?}: two curves \
                 meeting repeatedly in one window is noise or a grid too coarse to resolve it, and \
                 a mean of the roots would report a crossing sharper than the data",
                temperatures.len()
            ),
        }
    }
}

impl core::error::Error for NoCrossing {}

/// Every root of `U_large − U_small` inside the shared temperature window.
///
/// # Errors
///
/// [`NoCrossing::SameSize`], [`NoCrossing::Disjoint`] and [`NoCrossing::NeverCrosses`]. A curve
/// pair that meets several times returns every root here and is refused by [`crossing`].
pub fn crossings(a: &Curve, b: &Curve) -> Result<Vec<Crossing>, NoCrossing> {
    if a.size == b.size {
        return Err(NoCrossing::SameSize { size: a.size });
    }
    let (small, large) = if a.size < b.size { (a, b) } else { (b, a) };
    let (slo, shi) = small.temperature_range();
    let (llo, lhi) = large.temperature_range();
    let lo = slo.max(llo);
    let hi = shi.min(lhi);
    if !(hi > lo) {
        return Err(NoCrossing::Disjoint { small: (slo, shi), large: (llo, lhi) });
    }

    // Nodes: every breakpoint of either curve inside the window, plus the two ends. Between two
    // consecutive nodes BOTH interpolants are affine, so their difference is affine there and its
    // root is exact rather than bisected.
    let mut nodes: Vec<f64> = vec![lo, hi];
    for t in small.temps.iter().chain(large.temps.iter()) {
        if *t > lo && *t < hi {
            nodes.push(*t);
        }
    }
    nodes.sort_by(|x, y| x.partial_cmp(y).unwrap_or(core::cmp::Ordering::Equal));
    nodes.dedup_by(|x, y| (*x - *y).abs() <= 1e-12 * x.abs().max(1.0));

    // Both curves' values AT the nodes, and their difference. Every node lies inside both windows
    // by construction, so no interpolation here can fail to answer -- and carrying the two values
    // rather than re-interpolating at the root keeps `U*` exact: on one interval both curves are
    // affine, so the height at the root is the same linear blend of the node values for each.
    let mut us = Vec::with_capacity(nodes.len());
    let mut ul = Vec::with_capacity(nodes.len());
    for &t in &nodes {
        match (small.binder_at(t), large.binder_at(t)) {
            (Some(a), Some(b)) => {
                us.push(a);
                ul.push(b);
            }
            // Unreachable: `t` is in `[lo, hi]`, which is inside both grids. A shared window that
            // an interpolant declines is a disjoint window, and is reported as one rather than
            // filled in with a number.
            _ => return Err(NoCrossing::Disjoint { small: (slo, shi), large: (llo, lhi) }),
        }
    }

    let mut out: Vec<Crossing> = Vec::new();
    for k in 0..nodes.len() - 1 {
        let (d0, d1) = (ul[k] - us[k], ul[k + 1] - us[k + 1]);
        let (t, u4, bracket) = if d0 == 0.0 {
            (nodes[k], 0.5 * (us[k] + ul[k]), (nodes[k], nodes[k]))
        } else if d0 * d1 < 0.0 {
            // Opposite signs, so the fraction is in (0, 1) exactly; the clamp is the mathematics
            // held against rounding, not a repair of an out-of-range answer.
            let f = (d0 / (d0 - d1)).clamp(0.0, 1.0);
            let t = nodes[k] + f * (nodes[k + 1] - nodes[k]);
            let u = 0.5
                * ((us[k] + f * (us[k + 1] - us[k])) + (ul[k] + f * (ul[k + 1] - ul[k])));
            (t, u, (nodes[k], nodes[k + 1]))
        } else {
            continue;
        };
        out.push(Crossing { small: small.size, large: large.size, temperature: t, u4, bracket });
    }
    let last = nodes.len() - 1;
    if ul[last] - us[last] == 0.0 {
        out.push(Crossing {
            small: small.size,
            large: large.size,
            temperature: nodes[last],
            u4: 0.5 * (us[last] + ul[last]),
            bracket: (nodes[last], nodes[last]),
        });
    }

    if out.is_empty() {
        let d = || ul.iter().zip(&us).map(|(b, a)| b - a);
        let lowest = d().fold(f64::INFINITY, f64::min);
        let highest = d().fold(f64::NEG_INFINITY, f64::max);
        return Err(NoCrossing::NeverCrosses {
            small: small.size,
            large: large.size,
            lowest,
            highest,
        });
    }
    Ok(out)
}

/// The single crossing of two sizes' Binder curves.
///
/// # Errors
///
/// Everything [`crossings`] returns, plus [`NoCrossing::Ambiguous`] when the difference changes
/// sign more than once in the shared window.
pub fn crossing(a: &Curve, b: &Curve) -> Result<Crossing, NoCrossing> {
    let all = crossings(a, b)?;
    if all.len() > 1 {
        return Err(NoCrossing::Ambiguous {
            small: all[0].small,
            large: all[0].large,
            temperatures: all.iter().map(|c| c.temperature).collect(),
        });
    }
    Ok(all[0])
}

/// The crossing-point estimate of `T_c`, with every pair it was averaged over.
#[derive(Clone, Debug, PartialEq)]
pub struct Transition {
    /// Mean crossing temperature over the pairs that crossed.
    pub t_c: f64,
    /// Sample standard deviation of those crossings, `0` for a single pair.
    ///
    /// **This is a drift, not an error bar.** The crossings move systematically with size — that
    /// is what corrections to scaling are — so the spread measures how far from the limit the
    /// smallest sizes are, and it does not shrink by taking more draws.
    pub spread: f64,
    /// Mean `U*` at the crossings.
    pub u_star: f64,
    /// The crossing of the two largest sizes: the least-corrected single estimate, and the one to
    /// quote when the pairs disagree.
    pub largest: Crossing,
    /// Every pair, so the drift can be read rather than trusted.
    pub pairs: Vec<Crossing>,
    /// Pairs that produced no crossing, by size. A non-empty list is a real finding about those
    /// sizes and is not averaged away.
    pub uncrossed: Vec<(usize, usize)>,
}

/// Locate `T_c` from the crossings of every pair of sizes.
///
/// # Errors
///
/// [`NoCrossing::TooFewSizes`], [`NoCrossing::DuplicateSize`], and — when NO pair crosses — the
/// reason the largest pair gave. A model with no transition returns [`NoCrossing::NeverCrosses`]
/// here rather than a number.
pub fn critical_temperature(curves: &[Curve]) -> Result<Transition, NoCrossing> {
    if curves.len() < 2 {
        return Err(NoCrossing::TooFewSizes { sizes: curves.len() });
    }
    let mut sizes: Vec<usize> = curves.iter().map(Curve::size).collect();
    sizes.sort_unstable();
    for w in sizes.windows(2) {
        if w[0] == w[1] {
            return Err(NoCrossing::DuplicateSize { size: w[0] });
        }
    }

    let mut pairs: Vec<Crossing> = Vec::new();
    let mut uncrossed: Vec<(usize, usize)> = Vec::new();
    let mut last_error: Option<NoCrossing> = None;
    // Ordered so the final failure reported is the LARGEST pair's, which is the informative one.
    let mut order: Vec<(usize, usize)> = Vec::new();
    for i in 0..curves.len() {
        for j in (i + 1)..curves.len() {
            order.push((i, j));
        }
    }
    order.sort_by_key(|&(i, j)| {
        let (a, b) = (curves[i].size, curves[j].size);
        (a.max(b), a.min(b))
    });
    for &(i, j) in &order {
        match crossing(&curves[i], &curves[j]) {
            Ok(c) => pairs.push(c),
            Err(e) => {
                let (a, b) = (curves[i].size, curves[j].size);
                uncrossed.push((a.min(b), a.max(b)));
                last_error = Some(e);
            }
        }
    }
    let Some(&largest) = pairs.iter().max_by_key(|c| (c.large, c.small)) else {
        // Nothing crossed anywhere. Two or more distinct sizes make at least one pair, and a pair
        // that is not in `pairs` is in `last_error`, so the `unwrap_or` arm is unreachable rather
        // than a default standing in for a missing reason. `order` runs smallest pair to largest,
        // so the error carried here is the LARGEST pair's, which is the informative one.
        return Err(last_error.unwrap_or(NoCrossing::TooFewSizes { sizes: curves.len() }));
    };

    let n = pairs.len() as f64;
    let t_c = pairs.iter().map(|c| c.temperature).sum::<f64>() / n;
    let u_star = pairs.iter().map(|c| c.u4).sum::<f64>() / n;
    let spread = if pairs.len() < 2 {
        0.0
    } else {
        (pairs.iter().map(|c| (c.temperature - t_c).powi(2)).sum::<f64>() / (n - 1.0)).sqrt()
    };
    Ok(Transition { t_c, spread, u_star, largest, pairs, uncrossed })
}

/// The three numbers a collapse is a claim about.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Scaling {
    /// Critical temperature.
    pub t_c: f64,
    /// Correlation-length exponent `ν`, `ξ ~ |T − T_c|^{−ν}`. Strictly positive.
    pub nu: f64,
    /// Order-parameter exponent `β`, `m ~ (T_c − T)^β`.
    pub beta: f64,
}

impl Scaling {
    /// The exact 2D square-lattice Ising values: `T_c = 2/ln(1+√2)`, `ν = 1`, `β = 1/8`.
    ///
    /// Onsager (1944) for `T_c` and `ν`; Yang (1952) for `β`.
    #[must_use]
    pub fn ising_2d() -> Scaling {
        Scaling { t_c: ising_2d_tc(), nu: 1.0, beta: 0.125 }
    }

    /// The susceptibility exponent `γ`, from `ν` and `β` by hyperscaling: `γ = d·ν − 2β`.
    ///
    /// Josephson's relation, equivalently `γ/ν = d − 2β/ν`. It is a CONSEQUENCE of the other two
    /// and not a third measurement, which is why it is a method here and not a field: fitting `γ`
    /// independently from the same collapse would be reporting one number twice.
    ///
    /// At the exact 2D Ising values it returns `2·1 − 2/8 = 7/4` — Onsager's own `γ` — and that
    /// exact `7/4` is asserted in
    /// `hyperscaling_turns_onsagers_nu_and_yangs_beta_into_onsagers_gamma`.
    ///
    /// **It fails above the upper critical dimension** (`d > 4` for this universality class),
    /// where mean-field exponents take over and hyperscaling no longer holds. `dimension` is the
    /// caller's to get right; nothing in a set of curves says what `d` was.
    #[must_use]
    pub fn gamma(&self, dimension: usize) -> f64 {
        dimension as f64 * self.nu - 2.0 * self.beta
    }
}

/// Which quantity is collapsed, and therefore which exponents the collapse can see.
///
/// **These are not two views of one fit.** The order parameter carries `L^{β/ν}` and the cumulant
/// carries no power of `L` at all, so the cumulant collapse contains `T_c` and `ν` and nothing
/// else. That makes it the better-conditioned of the two by a wide margin, and the difference is
/// measured rather than asserted: on exact `L = 4…7` transfer-matrix data for the 2D Ising model,
/// the cumulant collapse returns `ν = 1.056` against the exact `1`, and the order-parameter
/// collapse on the SAME data returns `ν = 0.75`. See
/// `the_cumulant_collapse_recovers_nu_where_the_order_parameter_collapse_cannot`.
///
/// The reason is `β/ν = 1/8`. From `L = 4` to `L = 7` that rescales the vertical axis by
/// `(7/4)^{1/8} = 1.07`, so the sizes are separated by seven percent in `y` and the fit can buy
/// back almost any error in `ν` with a small change in `β`. The cumulant has no such knob.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Observable {
    /// `y = √⟨m²⟩ · L^{β/ν}` — the collapse the scaling hypothesis is usually written for.
    OrderParameter,
    /// `y = U_4` — dimensionless, so no power of `L` and no `β`. The fit varies `T_c` and `ν`
    /// only, and returns `β` exactly as it was handed in.
    Cumulant,
}

/// One data point in collapsed coordinates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Scaled {
    /// The lattice it came from.
    pub size: usize,
    /// `(T − T_c) L^{1/ν}`.
    pub x: f64,
    /// `√⟨m²⟩ · L^{β/ν}`, or `U_4`, according to the [`Observable`].
    pub y: f64,
}

/// Why a collapse could not be scored.
#[derive(Clone, Debug, PartialEq)]
pub enum CollapseError {
    /// Fewer than two sizes. A single curve collapses onto itself for every parameter value.
    TooFewSizes {
        /// How many curves were offered.
        sizes: usize,
    },
    /// One size given twice, which would score a curve against a copy of itself.
    DuplicateSize {
        /// The repeated size.
        size: usize,
    },
    /// `ν ≤ 0` or not finite. `L^{1/ν}` with a negative `ν` reverses the temperature axis, so the
    /// scaled grids are no longer sorted and every interpolation answers for the wrong interval.
    NonPositiveNu {
        /// What was passed.
        nu: f64,
    },
    /// A parameter that is not a number.
    NotFinite {
        /// `T_c` as passed.
        t_c: f64,
        /// `ν` as passed.
        nu: f64,
        /// `β` as passed.
        beta: f64,
    },
    /// No point of any size fell inside another size's scaled window, so nothing was compared.
    /// Usually `1/ν` so large that the sizes' windows have flown apart.
    NoOverlap {
        /// Points in the data set.
        points: usize,
    },
    /// The scaled `y` values have no spread, so the normalised residual has no denominator. A
    /// constant order parameter is not a perfect collapse; it is a data set with nothing in it.
    NoSpread {
        /// Points that were compared.
        covered: usize,
    },
}

impl core::fmt::Display for CollapseError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            CollapseError::TooFewSizes { sizes } => write!(
                f,
                "{sizes} curve(s): a collapse compares sizes against each other, and one curve \
                 collapses onto itself for every value of every parameter"
            ),
            CollapseError::DuplicateSize { size } => write!(
                f,
                "L = {size} appears twice, so a curve would be scored against a copy of itself and \
                 the residual would fall for a reason that is not a collapse"
            ),
            CollapseError::NonPositiveNu { nu } => write!(
                f,
                "nu = {nu}: the scaled axis is (T - T_c) L^(1/nu), and a non-positive nu either \
                 reverses it -- leaving every grid unsorted and every interpolation answering for \
                 the wrong interval -- or is not a number at all"
            ),
            CollapseError::NotFinite { t_c, nu, beta } => write!(
                f,
                "scaling (T_c = {t_c}, nu = {nu}, beta = {beta}) is not three finite numbers"
            ),
            CollapseError::NoOverlap { points } => write!(
                f,
                "none of the {points} points fell inside another size's scaled window, so nothing \
                 was compared with anything: 1/nu is large enough that the sizes have flown apart"
            ),
            CollapseError::NoSpread { covered } => write!(
                f,
                "the {covered} scaled values have no spread about their own mean, so the \
                 normalised residual has no denominator -- a flat order parameter is an empty data \
                 set, not a perfect collapse"
            ),
        }
    }
}

impl core::error::Error for CollapseError {}

/// Every point in collapsed coordinates, size by size and grid order preserved.
///
/// # Errors
///
/// [`CollapseError::TooFewSizes`], [`CollapseError::DuplicateSize`],
/// [`CollapseError::NonPositiveNu`] and [`CollapseError::NotFinite`].
pub fn collapse(
    curves: &[Curve],
    p: Scaling,
    what: Observable,
) -> Result<Vec<Scaled>, CollapseError> {
    check(curves, p)?;
    let mut out = Vec::new();
    for c in curves {
        let (xs, ys) = axes(c, p, what);
        for (k, &t) in c.temps.iter().enumerate() {
            out.push(Scaled { size: c.size, x: (t - p.t_c) * xs, y: ys[k] });
        }
    }
    Ok(out)
}

/// The horizontal stretch `L^{1/ν}` and the vertical values, for one curve.
///
/// The only place the [`Observable`] is interpreted: the order parameter carries `L^{β/ν}` and the
/// cumulant carries nothing, which is the whole difference between the two fits.
fn axes(c: &Curve, p: Scaling, what: Observable) -> (f64, Vec<f64>) {
    let l = c.size as f64;
    let y = match what {
        Observable::OrderParameter => {
            let s = l.powf(p.beta / p.nu);
            c.order.iter().map(|m| m * s).collect()
        }
        Observable::Cumulant => c.u4.clone(),
    };
    (l.powf(1.0 / p.nu), y)
}

/// The conditions both collapse entry points need before they touch a number: at least two
/// DISTINCT sizes, and three finite parameters with a positive `nu`.
fn check(curves: &[Curve], p: Scaling) -> Result<(), CollapseError> {
    if curves.len() < 2 {
        return Err(CollapseError::TooFewSizes { sizes: curves.len() });
    }
    let mut sizes: Vec<usize> = curves.iter().map(Curve::size).collect();
    sizes.sort_unstable();
    for w in sizes.windows(2) {
        if w[0] == w[1] {
            return Err(CollapseError::DuplicateSize { size: w[0] });
        }
    }
    if !p.t_c.is_finite() || !p.nu.is_finite() || !p.beta.is_finite() {
        return Err(CollapseError::NotFinite { t_c: p.t_c, nu: p.nu, beta: p.beta });
    }
    if !(p.nu > 0.0) {
        return Err(CollapseError::NonPositiveNu { nu: p.nu });
    }
    Ok(())
}

/// How well a set of curves collapses, and on how much of itself.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Quality {
    /// `S = Σ(y − Y)² / Σ(y − ȳ)²`, an UPPER bound on the exact value (see the module docs). Zero
    /// is an exact collapse; one means the sizes are no closer to each other than to their mean.
    pub residual: f64,
    /// Points that fell inside another size's window and were therefore compared.
    pub covered: usize,
    /// Points that did not, and contributed nothing. A large number here means the parameters have
    /// pulled the sizes apart and the residual is being computed on a fragment.
    pub skipped: usize,
}

/// Score a collapse: the master-curve residual, normalised by the spread of the scaled data.
///
/// Each point is compared against the linear interpolation of every OTHER size's scaled curve at
/// the same `x`, averaged. Points outside every other size's window are skipped and counted.
///
/// # Errors
///
/// Everything [`collapse`] returns, plus [`CollapseError::NoOverlap`] when no point could be
/// compared and [`CollapseError::NoSpread`] when the scaled values are constant.
pub fn collapse_residual(
    curves: &[Curve],
    p: Scaling,
    what: Observable,
) -> Result<Quality, CollapseError> {
    check(curves, p)?;
    // Scaled coordinates per curve. x is increasing in T because L^(1/nu) > 0, so each scaled grid
    // inherits the sortedness `Curve::new` enforced and the interpolants stay valid.
    let mut xs: Vec<Vec<f64>> = Vec::with_capacity(curves.len());
    let mut ys: Vec<Vec<f64>> = Vec::with_capacity(curves.len());
    for c in curves {
        let (sx, y) = axes(c, p, what);
        xs.push(c.temps.iter().map(|t| (t - p.t_c) * sx).collect());
        ys.push(y);
    }

    let mut misfit: Vec<f64> = Vec::new();
    let mut used: Vec<f64> = Vec::new();
    let mut skipped = 0usize;
    let mut total = 0usize;
    for i in 0..curves.len() {
        for k in 0..xs[i].len() {
            total += 1;
            let x = xs[i][k];
            let mut acc = 0.0;
            let mut count = 0usize;
            for j in 0..curves.len() {
                if j == i {
                    continue;
                }
                if let Some(v) = interpolate(&xs[j], &ys[j], x) {
                    acc += v;
                    count += 1;
                }
            }
            if count == 0 {
                skipped += 1;
                continue;
            }
            let dy = ys[i][k] - acc / count as f64;
            misfit.push(dy * dy);
            used.push(ys[i][k]);
        }
    }
    if used.is_empty() {
        return Err(CollapseError::NoOverlap { points: total });
    }
    let mean = used.iter().sum::<f64>() / used.len() as f64;
    let spread: Vec<f64> = used.iter().map(|y| (y - mean) * (y - mean)).collect();
    // A ratio of two sums that is reported as a bound: the numerator up and the denominator down,
    // so the quotient is never BELOW the exact residual. `crate::bound::forest` shipped a lower
    // bound above its own optimum for want of exactly this.
    let num = round::sum_up(&misfit);
    let den = round::sum_down(&spread);
    if !(den > 0.0) {
        return Err(CollapseError::NoSpread { covered: used.len() });
    }
    Ok(Quality { residual: num / den, covered: used.len(), skipped })
}

/// How hard to look for the best collapse.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FitOptions {
    /// Initial step in `T_c`, in temperature units.
    pub step_t_c: f64,
    /// Initial step in `1/ν`.
    pub step_inv_nu: f64,
    /// Initial step in `β/ν`.
    pub step_beta_nu: f64,
    /// Stop once every step has shrunk below this. The search is exact to about this much.
    pub tolerance: f64,
    /// Hard cap on objective evaluations.
    pub max_evaluations: usize,
}

impl Default for FitOptions {
    /// Steps sized for a transition already bracketed to about a tenth of a degree and exponents
    /// of order one — which is what the crossing estimate hands over.
    fn default() -> FitOptions {
        FitOptions {
            step_t_c: 0.05,
            step_inv_nu: 0.1,
            step_beta_nu: 0.05,
            tolerance: 1e-9,
            max_evaluations: 20_000,
        }
    }
}

/// What the fit came to, and everything needed to disbelieve it.
#[derive(Clone, Debug, PartialEq)]
pub struct CollapseFit {
    /// Which quantity was collapsed. For [`Observable::Cumulant`] the form contains no `β`, so
    /// `scaling.beta` is the value handed in and was not fitted.
    pub observable: Observable,
    /// The fitted `(T_c, ν, β)`.
    pub scaling: Scaling,
    /// `S` at the optimum.
    pub residual: f64,
    /// `S` at the starting point, so a fit that went nowhere is visible.
    pub start_residual: f64,
    /// Objective evaluations spent.
    pub evaluations: usize,
    /// Whether every step shrank below [`FitOptions::tolerance`] before the evaluation cap.
    pub converged: bool,
    /// Points compared at the optimum.
    pub covered: usize,
}

/// Fit `(T_c, ν, β)` by minimising the collapse residual, with the default search.
///
/// # Errors
///
/// Everything [`collapse_residual`] returns, evaluated at `start`.
pub fn fit_collapse(
    curves: &[Curve],
    start: Scaling,
    what: Observable,
) -> Result<CollapseFit, CollapseError> {
    fit_collapse_with(curves, start, what, FitOptions::default())
}

/// Fit `(T_c, ν, β)` by minimising the collapse residual.
///
/// # The search runs in `(T_c, 1/ν, β/ν)`
///
/// Those are the three numbers the scaling form contains — `x = (T − T_c) L^{1/ν}` and
/// `y = m L^{β/ν}` — and `ν` and `β` are recovered at the end as `ν = 1/(1/ν)` and
/// `β = (β/ν)·ν`. Searching in `(T_c, ν, β)` directly would move `β/ν` every time `ν` moved, which
/// couples two axes through a ratio that the objective does not actually contain.
///
/// # The method
///
/// Compass search: try `±step` on each coordinate in turn, take any improvement, halve every step
/// when a full round finds none. Deterministic, derivative-free, no RNG — the objective has kinks
/// wherever the linear interpolation switches interval, so a gradient method would be differentiating
/// through a corner. Generalised pattern search converges to a stationary point for continuously
/// differentiable objectives (Torczon 1997); this one is piecewise smooth, so what the tolerance
/// promises is a coordinate-wise local minimum to within the final step, which is what
/// [`CollapseFit::converged`] reports and no more.
///
/// **A local minimum is all it is.** Start from [`critical_temperature`]'s crossing estimate and a
/// plausible pair of exponents, not from nothing.
///
/// # Freezing an axis
///
/// A step of zero pins that coordinate: `step_t_c = 0.0` fits the exponents at a `T_c` known from
/// elsewhere, which is the two-stage workflow FSS is usually done with. For
/// [`Observable::Cumulant`] the `β` step is ignored, because the cumulant collapse does not contain
/// `β` and moving it would change nothing while reporting that it had.
///
/// # Errors
///
/// Everything [`collapse_residual`] returns, evaluated at `start`.
pub fn fit_collapse_with(
    curves: &[Curve],
    start: Scaling,
    what: Observable,
    opts: FitOptions,
) -> Result<CollapseFit, CollapseError> {
    let start_quality = collapse_residual(curves, start, what)?;

    // v = [T_c, 1/nu, beta/nu]
    let mut v = [start.t_c, 1.0 / start.nu, start.beta / start.nu];
    let beta_step = match what {
        Observable::OrderParameter => opts.step_beta_nu.abs(),
        Observable::Cumulant => 0.0,
    };
    let mut step = [opts.step_t_c.abs(), opts.step_inv_nu.abs(), beta_step];
    let mut evaluations = 1usize;
    let mut best = start_quality.residual;

    let score = |v: &[f64; 3], curves: &[Curve]| -> f64 {
        if !(v[1] > 0.0) {
            return f64::INFINITY;
        }
        let nu = 1.0 / v[1];
        let s = Scaling { t_c: v[0], nu, beta: v[2] * nu };
        collapse_residual(curves, s, what).map_or(f64::INFINITY, |q| q.residual)
    };

    let converged = loop {
        if step.iter().all(|s| *s <= opts.tolerance) {
            break true;
        }
        if evaluations >= opts.max_evaluations {
            break false;
        }
        let mut improved = false;
        for axis in 0..3 {
            // A step of zero is a frozen coordinate, not a move worth scoring twice.
            if step[axis] == 0.0 {
                continue;
            }
            for sign in [1.0f64, -1.0] {
                let mut trial = v;
                trial[axis] += sign * step[axis];
                let s = score(&trial, curves);
                evaluations += 1;
                if s < best {
                    best = s;
                    v = trial;
                    improved = true;
                    break;
                }
            }
        }
        if !improved {
            for s in &mut step {
                *s *= 0.5;
            }
        }
    };

    let nu = 1.0 / v[1];
    let scaling = Scaling {
        t_c: v[0],
        nu,
        // The search carries `beta/nu`, so recovering `beta = (beta/nu) * nu` would MOVE `beta`
        // whenever `nu` moved -- even on a cumulant fit, which does not contain `beta` at all and
        // never scored a single trial against it. Handing back a number the fit did not choose,
        // computed from one it did, is precisely the shape of a reported search that never
        // happened. On the cumulant path `beta` is therefore returned exactly as it arrived.
        beta: match what {
            Observable::OrderParameter => v[2] * nu,
            Observable::Cumulant => start.beta,
        },
    };
    let final_quality = collapse_residual(curves, scaling, what)?;
    Ok(CollapseFit {
        observable: what,
        scaling,
        residual: final_quality.residual,
        start_residual: start_quality.residual,
        evaluations,
        converged,
        covered: final_quality.covered,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::cluster::{Sampler, Update};
    use crate::graph::Graph;
    use crate::ising::{exact_boltzmann, lattice2d, ring};
    use crate::samples::Plan;

    /// Exact `⟨m²⟩` and `⟨m⁴⟩` by exhaustive enumeration, through the crate's own Boltzmann oracle.
    ///
    /// [`crate::ising::exact_boltzmann`] has its own verification against the low-temperature
    /// split of a ferromagnetic ring, so this is a reference and not a second implementation of
    /// the thing under test.
    fn enumerated_point(g: &Graph, t: f64) -> Point {
        let p = exact_boltzmann(g, 1.0 / t);
        let n = g.n as f64;
        let (mut m2, mut m4) = (0.0, 0.0);
        for (mask, &w) in p.iter().enumerate() {
            let m = (2.0 * f64::from((mask as u64).count_ones()) - n) / n;
            m2 += w * m * m;
            m4 += w * m * m * m * m;
        }
        Point { temperature: t, m2, m4 }
    }

    /// Exact moments on the periodic `L × L` Ising lattice by ROW TRANSFER MATRIX, which reaches
    /// sizes exhaustive enumeration cannot.
    ///
    /// The weight factorises over rows, `w = Π_r W(a_r) V(a_r, a_{r+1})` with `W` the intra-row
    /// bonds and `V` the inter-row ones, so the partition function is a product of `2^L × 2^L`
    /// matrices. The magnetisation is a SUM over rows, so its moments propagate with it: carrying
    /// `(Z, ZM, ZM², ZM³, ZM⁴)` and expanding `(M + μ)^k` by the binomial theorem gives every
    /// moment in one pass, with no need to resolve the whole distribution of `M`.
    ///
    /// The periodic boundary in the row direction is closed by running one propagation per
    /// starting row configuration and multiplying back by `V(last, first)` — a trace, written out.
    ///
    /// `L = 7` is 2²¹ inner steps and costs about 40 ms per temperature; `L = 8` is eight times
    /// that and is why these tests stop at seven.
    ///
    /// Verified against `enumerated_point` at `L = 3` and `L = 4` in
    /// `the_transfer_matrix_oracle_agrees_with_exhaustive_enumeration`.
    fn transfer_point(l: usize, t: f64) -> Point {
        let beta = 1.0 / t;
        let rows = 1usize << l;
        let spin = |c: usize, x: usize| if (c >> x) & 1 == 1 { 1.0f64 } else { -1.0f64 };
        let mut w = vec![0.0f64; rows];
        let mut mu = vec![0.0f64; rows];
        for c in 0..rows {
            let (mut bond, mut m) = (0.0, 0.0);
            for x in 0..l {
                bond += spin(c, x) * spin(c, (x + 1) % l);
                m += spin(c, x);
            }
            w[c] = (beta * bond).exp();
            mu[c] = m;
        }
        let mut v = vec![0.0f64; rows * rows];
        for a in 0..rows {
            for b in 0..rows {
                let mut bond = 0.0;
                for x in 0..l {
                    bond += spin(a, x) * spin(b, x);
                }
                v[a * rows + b] = (beta * bond).exp();
            }
        }
        // One row's weight is at most exp(2 beta L); dividing it out per row keeps the product inside
        // f64 at every size and cancels out of the ratios below.
        let scale = (-2.0 * beta * l as f64).exp();
        let (mut z, mut z2, mut z4) = (0.0f64, 0.0f64, 0.0f64);
        let mut cur = vec![[0.0f64; 5]; rows];
        let mut next = vec![[0.0f64; 5]; rows];
        for a0 in 0..rows {
            cur.fill([0.0; 5]);
            let (m0, w0) = (mu[a0], w[a0] * scale);
            cur[a0] = [w0, w0 * m0, w0 * m0 * m0, w0 * m0 * m0 * m0, w0 * m0 * m0 * m0 * m0];
            for _ in 1..l {
                next.fill([0.0; 5]);
                for c in 0..rows {
                    let k = cur[c];
                    if k[0] == 0.0 {
                        continue;
                    }
                    for b in 0..rows {
                        let f = v[c * rows + b] * w[b] * scale;
                        let m = mu[b];
                        let (m2, m3, m4) = (m * m, m * m * m, m * m * m * m);
                        let n = &mut next[b];
                        n[0] += f * k[0];
                        n[1] += f * (k[1] + m * k[0]);
                        n[2] += f * (k[2] + 2.0 * m * k[1] + m2 * k[0]);
                        n[3] += f * (k[3] + 3.0 * m * k[2] + 3.0 * m2 * k[1] + m3 * k[0]);
                        n[4] += f
                            * (k[4]
                                + 4.0 * m * k[3]
                                + 6.0 * m2 * k[2]
                                + 4.0 * m3 * k[1]
                                + m4 * k[0]);
                    }
                }
                core::mem::swap(&mut cur, &mut next);
            }
            for c in 0..rows {
                let f = v[c * rows + a0];
                z += f * cur[c][0];
                z2 += f * cur[c][2];
                z4 += f * cur[c][4];
            }
        }
        let n = (l * l) as f64;
        Point { temperature: t, m2: z2 / z / (n * n), m4: z4 / z / (n * n * n * n) }
    }

    /// The temperature grid the exact curves are measured on: 21 points at 0.02 through the
    /// crossing region, which straddles `T_c = 2.2692` on both sides.
    fn grid() -> Vec<f64> {
        (0..21).map(|k| 2.07 + 0.02 * k as f64).collect()
    }

    /// `L = 3…7` exact Binder curves, built once — `L = 7` costs about 0.8 s and five tests want it.
    fn exact_curves() -> &'static [Curve] {
        static CACHE: std::sync::OnceLock<Vec<Curve>> = std::sync::OnceLock::new();
        CACHE.get_or_init(|| {
            let g = grid();
            [3usize, 4, 5, 6, 7]
                .iter()
                .map(|&l| Curve::new(l, g.iter().map(|&t| transfer_point(l, t)).collect()).unwrap())
                .collect()
        })
    }

    /// THE ORACLE'S OWN ORACLE. The transfer matrix reaches `L = 5, 6, 7`, where nothing else in
    /// this crate can check it, so it is pinned against exhaustive enumeration where both answer.
    ///
    /// A wrong boundary closure, a dropped binomial term or a transposed `V` all leave a plausible
    /// number here; agreeing with `exact_boltzmann` to `1e-12` at two sizes and four temperatures
    /// does not.
    #[test]
    fn the_transfer_matrix_oracle_agrees_with_exhaustive_enumeration() {
        for l in [3usize, 4] {
            let g = lattice2d(l, 1.0);
            for t in [1.8f64, 2.269_185, 3.0, 8.0] {
                let (a, b) = (enumerated_point(&g, t), transfer_point(l, t));
                assert!(
                    (a.m2 - b.m2).abs() < 1e-12 && (a.m4 - b.m4).abs() < 1e-12,
                    "L={l} T={t}: enumeration ({}, {}) against transfer matrix ({}, {})",
                    a.m2,
                    a.m4,
                    b.m2,
                    b.m4
                );
            }
        }
    }

    /// CLOSED FORM. For `N` INDEPENDENT spins `U_4` is exactly `2/(3N)`, and nothing about this
    /// number comes from this module.
    ///
    /// `M = Σ s_i` of `N` independent `±1` variables has `⟨M²⟩ = N` and `⟨M⁴⟩ = 3N² − 2N`
    /// (the fourth moment of a Rademacher sum: `N` terms `s_i⁴` and `3N(N−1)` pairs
    /// `s_i² s_j²`). So
    ///
    /// ```text
    ///     ⟨m⁴⟩ / (3⟨m²⟩²) = (3N² − 2N)/N⁴ ÷ (3/N²) = 1 − 2/(3N)      and      U_4 = 2/(3N).
    /// ```
    ///
    /// This is the test the `3` in the definition lives or dies by: the disordered limit is `0`
    /// only under this normalisation, and any other constant puts the free-spin value somewhere
    /// else entirely.
    #[test]
    fn the_binder_cumulant_of_free_spins_is_exactly_two_over_three_n() {
        for n in [1usize, 2, 6, 11, 16] {
            let free = crate::graph::GraphBuilder::new(n).build();
            // Uncoupled and unbiased: every temperature is the same distribution, the uniform one.
            let p = enumerated_point(&free, 1.234);
            let want = 2.0 / (3.0 * n as f64);
            let got = binder(p.m2, p.m4).unwrap();
            assert!(
                (got - want).abs() < 1e-12,
                "N={n}: U_4 = {got}, closed form 2/(3N) = {want}"
            );
            // And the same number through the raw-draw constructor, over all 2^N states.
            let ms: Vec<f64> = (0..1u64 << n)
                .map(|mask| (2.0 * f64::from(mask.count_ones()) - n as f64) / n as f64)
                .collect();
            let q = Point::from_magnetisations(1.234, &ms).unwrap();
            assert!((binder(q.m2, q.m4).unwrap() - want).abs() < 1e-12, "N={n} from draws");
        }
        // Cauchy-Schwarz: no distribution whatever exceeds the ordered limit, and a two-valued
        // m attains it. <m^4> = <m^2>^2 is exactly the equality case.
        assert!((binder(0.4, 0.16).unwrap() - 2.0 / 3.0).abs() < 1e-15, "two-valued m attains it");
        // A degenerate second moment is an absent measurement, not a large cumulant.
        assert_eq!(binder(0.0, 0.0), None);
        assert_eq!(binder(-1.0, 1.0), None);
        assert_eq!(binder(0.5, f64::NAN), None);
    }

    /// BOTH EXACT LIMITS, FROM A REAL RUN — Swendsen–Wang on an `8 × 8` periodic lattice.
    ///
    /// Ordered (`T = 1.5`, well below `T_c = 2.269`): `m` sits on `±m₀`, so `⟨m⁴⟩ = ⟨m²⟩²` and
    /// `U_4 → 2/3`. Measured `0.66585`, which is `2/3 − 0.00081`, and the spread over six seeds is
    /// `6e-5` — the ordered limit is reached to four digits by a lattice of 64 spins.
    ///
    /// Disordered (`T = 6`): `m` is near-Gaussian about zero, `⟨m⁴⟩ → 3⟨m²⟩²`, and `U_4 → 0`. The
    /// free-spin floor is `2/(3·64) = 0.0104`. **The tolerance here is forty times looser than the
    /// ordered one and the reason is structural**: `U_4 = 1 − r` with `r → 1`, so an absolute error
    /// in `U_4` is a RELATIVE error in a ratio of fourth moments, and the fourth moment of a
    /// near-Gaussian is the noisiest thing in the estimator. Measured across six seeds at 20,000
    /// draws: `0.019` to `0.055`; at 80,000 draws: `0.032` to `0.048`.
    #[test]
    fn both_exact_limits_of_u4_appear_in_a_swendsen_wang_run_on_the_2d_lattice() {
        let g = lattice2d(8, 1.0);
        let n = g.n as f64;
        let run = |t: f64, seed: u64| -> f64 {
            let mut s = Sampler::new(&g, 1.0 / t, seed).unwrap();
            let set = s.collect(&Plan::new(500, 20_000, 2), Update::SwendsenWang);
            let ms: Vec<f64> = set
                .states()
                .iter()
                .map(|st| st.iter().map(|&v| f64::from(v)).sum::<f64>() / n)
                .collect();
            let p = Point::from_magnetisations(t, &ms).unwrap();
            binder(p.m2, p.m4).unwrap()
        };

        for seed in [7u64, 11, 23] {
            let ordered = run(1.5, seed);
            assert!(
                (ordered - 2.0 / 3.0).abs() < 2e-3,
                "seed {seed}: ordered U_4 = {ordered}, limit 2/3"
            );
            assert!(ordered <= 2.0 / 3.0, "seed {seed}: {ordered} is above the Cauchy-Schwarz cap");

            let hot = run(6.0, seed);
            assert!(hot > 0.0 && hot < 0.08, "seed {seed}: disordered U_4 = {hot}, limit 0");
            // ASYMMETRIC: the two limits must be told apart, not merely both be "close to
            // something". A sampler stuck in one phase would put both on the same side.
            assert!(
                ordered - hot > 0.55,
                "seed {seed}: the two limits are {ordered} and {hot}, which is not 2/3 apart"
            );
            assert!(
                (hot - 2.0 / (3.0 * n)).abs() < 0.06,
                "seed {seed}: disordered U_4 = {hot} against the free-spin floor {}",
                2.0 / (3.0 * n)
            );
        }
    }

    /// HEADLINE. The Binder crossings of exact `L = 3…7` lattices march onto Onsager's
    /// `T_c = 2/ln(1+√2) = 2.269185`.
    ///
    /// Measured, with no Monte Carlo error anywhere in it:
    ///
    /// | pair | crossing `T` | error | `U*` |
    /// |---|---|---|---|
    /// | 3, 4 | 2.169236 | −4.40% | 0.62924 |
    /// | 4, 5 | 2.212409 | −2.50% | 0.62430 |
    /// | 5, 6 | 2.233094 | −1.59% | 0.62122 |
    /// | 6, 7 | 2.244747 | −1.08% | 0.61907 |
    ///
    /// So the window is **one percent at `L = 6, 7`**, and it is approached from BELOW and
    /// monotonically. That monotone march is the asymmetric half of this test: a crossing routine
    /// that returned the middle of the window, or the minimum of one curve, would sit inside a
    /// percent-wide band and would not order itself by size.
    ///
    /// `U*` drifts onto `0.6107`, the universal value for this class on a periodic square
    /// (Kamieniarz & Blöte, J. Phys. A 26, 201 (1993)), from above and by the same monotone march.
    #[test]
    fn the_binder_crossing_of_exact_lattices_converges_on_the_onsager_temperature() {
        let curves = exact_curves();
        let tc = ising_2d_tc();
        assert!((tc - 2.269_185_314_213_022).abs() < 1e-14, "Onsager's own number: {tc}");

        let mut previous = f64::INFINITY;
        let mut previous_u = f64::INFINITY;
        for k in 0..4 {
            let c = crossing(&curves[k], &curves[k + 1]).unwrap();
            let err = (c.temperature - tc) / tc;
            assert_eq!((c.small, c.large), (k + 3, k + 4));
            assert!(err < 0.0, "({},{}) crossed ABOVE T_c at {}", c.small, c.large, c.temperature);
            assert!(err.abs() < 0.05, "({},{}) is {err:+.4} from T_c", c.small, c.large);
            assert!(
                err.abs() < previous,
                "({},{}) at {err:+.5} is no closer to T_c than the pair below it ({previous:+.5})",
                c.small,
                c.large
            );
            previous = err.abs();
            // The crossing is bracketed by two GRID points, and the interpolated value is inside.
            assert!(c.bracket.0 <= c.temperature && c.temperature <= c.bracket.1);
            assert!(c.bracket.1 - c.bracket.0 <= 0.0201, "bracket {:?}", c.bracket);
            // Universal U*, drifting DOWN onto 0.6107 and, like the temperature, monotonically.
            assert!((c.u4 - 0.6107).abs() < 0.02, "({},{}) U* = {}", c.small, c.large, c.u4);
            assert!(
                c.u4 > 0.6107 && c.u4 < previous_u,
                "({},{}) U* = {} is not between 0.6107 and the pair below it ({previous_u})",
                c.small,
                c.large,
                c.u4
            );
            previous_u = c.u4;
        }
        assert!(previous_u - 0.6107 < 0.009, "the largest pair's U* is {previous_u}");
        assert!(previous < 0.012, "the largest pair is {previous:+.5} from T_c");

        let t = critical_temperature(curves).unwrap();
        assert_eq!(t.pairs.len(), 10, "every pair of five sizes");
        assert!(t.uncrossed.is_empty());
        assert_eq!((t.largest.small, t.largest.large), (6, 7));
        assert!((t.largest.temperature - 2.244_747).abs() < 1e-5, "{:?}", t.largest);
        // The mean over pairs is DRAGGED DOWN by the small ones, and the spread says so. This is
        // the number not to quote, and the field says so too.
        assert!(t.t_c < t.largest.temperature, "{} against {}", t.t_c, t.largest.temperature);
        assert!(t.spread > 0.02, "the pairs disagree by {}, which is the drift", t.spread);
    }

    /// WHY THE TOLERANCES ARE WHAT THEY ARE, measured with no fitting in it at all.
    ///
    /// The pure scaling form is the leading term of an expansion in `L^{−ω}`. How far the next term
    /// reaches is not a matter of opinion: take the EFFECTIVE exponents straight off the exact data
    /// at `T_c`, from ratios between successive sizes,
    ///
    /// ```text
    ///     β/ν = −ln(m_rms(L')/m_rms(L)) / ln(L'/L)          1/ν = ln(|U'(L')|/|U'(L)|) / ln(L'/L)
    /// ```
    ///
    /// and read how fast they are still moving:
    ///
    /// | `L → L'` | `β/ν` (exact `0.125`) | `1/ν` (exact `1`) | `U_4(T_c)` |
    /// |---|---|---|---|
    /// | 3 → 4 | 0.1064 | 1.0762 | 0.62015 → 0.61720 |
    /// | 4 → 5 | 0.1137 | 1.0471 | 0.61548 |
    /// | 5 → 6 | 0.1176 | 1.0352 | 0.61436 |
    /// | 6 → 7 | 0.1198 | 1.0275 | 0.61359 |
    ///
    /// So at `L = 7` the data itself is still 4% and 3% from the limit BEFORE any fit is asked to
    /// find it, and `U_4(T_c)` is still 0.003 above the universal `0.6107`. An 8% tolerance on a
    /// fitted exponent is not slack; it is the size of the thing being measured.
    #[test]
    fn the_effective_exponents_are_still_moving_at_side_seven() {
        let tc = ising_2d_tc();
        let d = 0.01;
        let mut previous: Option<(f64, f64, f64)> = None;
        let mut at_tc = Vec::new();
        for l in [3usize, 4, 5, 6, 7] {
            let here = transfer_point(l, tc);
            let m = here.m2.sqrt();
            let u = binder(here.m2, here.m4).unwrap();
            let (up, dn) = (transfer_point(l, tc + d), transfer_point(l, tc - d));
            let slope = (binder(up.m2, up.m4).unwrap() - binder(dn.m2, dn.m4).unwrap()) / (2.0 * d);
            assert!(slope < 0.0, "L={l}: U_4 must fall with temperature, not rise ({slope})");
            at_tc.push(u);
            if let Some((pl, pm, ps)) = previous {
                let r = (l as f64 / pl).ln();
                let eff_beta_nu = -(m / pm).ln() / r;
                let eff_inv_nu = (slope / ps).abs().ln() / r;
                // Approaching the exact values, and from below / above respectively.
                assert!(
                    eff_beta_nu > 0.10 && eff_beta_nu < 0.125,
                    "L={l}: effective beta/nu = {eff_beta_nu}"
                );
                assert!(
                    eff_inv_nu > 1.0 && eff_inv_nu < 1.10,
                    "L={l}: effective 1/nu = {eff_inv_nu}"
                );
            }
            previous = Some((l as f64, m, slope));
        }
        // Monotone onto the universal U* from above, and still short of it at L = 7.
        for w in at_tc.windows(2) {
            assert!(w[1] < w[0], "U_4(T_c) must fall with size: {at_tc:?}");
        }
        let last = at_tc[at_tc.len() - 1];
        assert!(last > 0.6107 && last - 0.6107 < 0.004, "U_4(T_c) at L = 7 is {last}");
        // ASYMMETRIC: the exponents are still MOVING, so a claim that L = 7 is converged is false.
        // The gap at 6 -> 7 must be a real fraction of the gap at 3 -> 4, not a rounding artefact.
        assert!((0.125 - 0.1198f64).abs() > 0.003, "the L=7 data is not at the limit");
    }

    /// The exponents, against Onsager's `ν = 1` and Yang's `β = 1/8`, from lattices of side 4 to 7.
    ///
    /// **Two stages, because three correlated parameters do not come out of four small lattices at
    /// once.** The cumulant collapse carries no `β` and no power of `L` on its vertical axis, so it
    /// determines `T_c` and `ν` alone; `β` is then fitted from the order-parameter collapse with
    /// those two held.
    ///
    /// Measured: `ν = 1.0491` (**+4.9%**), `T_c = 2.22684` (−1.9%), and with `T_c` and `ν` at
    /// their exact values `β = 0.11723` (**−6.2%**).
    ///
    /// **The tolerances are 8% and 8%, and the size of them is corrections to scaling, not
    /// sloppiness.** The pure scaling form is the leading term of an expansion in `L^{−ω}`, and at
    /// `L ≤ 7` the next term is not small: the EFFECTIVE exponents measured directly from the
    /// exact data at `T_c` — `β/ν` from `m_rms(L)` and `1/ν` from `dU_4/dT` — are `0.1198` and
    /// `1.0275` between `L = 6` and `L = 7`, already 4% and 3% off their own limits before any
    /// fitting happens. A fit that returned `ν = 1.000` from these lattices would be reporting
    /// something other than what the data contains.
    #[test]
    fn the_exponents_come_out_near_onsagers_from_exact_lattices_of_side_four_to_seven() {
        let curves: Vec<Curve> = exact_curves()[1..].to_vec();
        assert_eq!(curves.iter().map(Curve::size).collect::<Vec<_>>(), vec![4, 5, 6, 7]);

        // Stage one: T_c and nu, from a collapse that contains nothing else.
        let start = Scaling { t_c: 2.24, nu: 1.3, beta: 0.0 };
        let cum = fit_collapse(&curves, start, Observable::Cumulant).unwrap();
        assert!(cum.converged);
        assert_eq!(cum.observable, Observable::Cumulant);
        assert_eq!(cum.scaling.beta, 0.0, "the cumulant collapse must not touch beta");
        assert!(cum.residual < cum.start_residual, "the fit must go somewhere");
        assert!(
            (cum.scaling.nu - 1.0).abs() < 0.08,
            "nu = {} against Onsager's 1",
            cum.scaling.nu
        );
        assert!(
            (cum.scaling.t_c - ising_2d_tc()).abs() / ising_2d_tc() < 0.025,
            "T_c = {} against 2.269185",
            cum.scaling.t_c
        );

        // Stage two: beta alone, at the EXACT T_c and nu, which is what the tolerance is about.
        let pinned = FitOptions { step_t_c: 0.0, step_inv_nu: 0.0, ..FitOptions::default() };
        let b = fit_collapse_with(
            &curves,
            Scaling { t_c: ising_2d_tc(), nu: 1.0, beta: 0.3 },
            Observable::OrderParameter,
            pinned,
        )
        .unwrap();
        assert_eq!(b.scaling.t_c, ising_2d_tc(), "a zero step pins its coordinate");
        assert_eq!(b.scaling.nu, 1.0);
        assert!(
            (b.scaling.beta - 0.125).abs() < 0.125 * 0.08,
            "beta = {} against Yang's 1/8",
            b.scaling.beta
        );
        // And it is a real minimum rather than the start: 0.3 was a long way off.
        assert!(b.residual < 0.2 * b.start_residual, "{} against {}", b.residual, b.start_residual);
    }

    /// ASYMMETRIC, and the finding is that ONE of these two collapses works at this size.
    ///
    /// The same exact `L = 4…7` data, fitted two ways from the same starting point. The cumulant
    /// collapse returns `ν = 1.0555`; the free three-parameter order-parameter collapse returns
    /// `ν = 0.7816`, 22% out.
    ///
    /// **The miss is the data's, not the search's**, and that is asserted rather than assumed: the
    /// order-parameter residual at its own answer is `8.9e-5` against `6.2e-3` at Onsager's exact
    /// exponents — SEVENTY times better on the same objective — so on lattices this small the
    /// order parameter genuinely collapses better on the wrong `ν`. `β/ν = 1/8` separates `L = 4`
    /// from `L = 7` by only `(7/4)^{1/8} = 7%` in the vertical, and the fit buys back an error in
    /// `ν` with a small change in `β`. The cumulant has no such knob.
    ///
    /// A test that only asserted "the cumulant fit lands near 1" would also pass for an
    /// implementation that ignored the observable and fitted the cumulant both times.
    #[test]
    fn the_cumulant_collapse_recovers_nu_where_the_order_parameter_collapse_cannot() {
        let curves: Vec<Curve> = exact_curves()[1..].to_vec();
        let start = Scaling { t_c: ising_2d_tc(), nu: 1.0, beta: 0.125 };
        let cum = fit_collapse(&curves, start, Observable::Cumulant).unwrap();
        let ord = fit_collapse(&curves, start, Observable::OrderParameter).unwrap();

        assert!((cum.scaling.nu - 1.0).abs() < 0.08, "cumulant nu = {}", cum.scaling.nu);
        assert!(
            (ord.scaling.nu - 1.0).abs() > 0.15,
            "the order-parameter fit is supposed to MISS here; it gave nu = {}",
            ord.scaling.nu
        );
        // Same objective, two parameter sets: the fit's is far better than Onsager's own, which is
        // what says the search did its job and the DATA prefers the wrong exponent.
        let at_exact =
            collapse_residual(&curves, Scaling::ising_2d(), Observable::OrderParameter)
                .unwrap()
                .residual;
        assert!(
            ord.residual < 0.1 * at_exact,
            "the miss must be the data's and not the search's: S = {} at nu = {}, and {at_exact} \
             at Onsager's exact exponents",
            ord.residual,
            ord.scaling.nu
        );
        // The cumulant fit beats Onsager's exponents on ITS objective too -- same corrections to
        // scaling, one tenth the damage.
        let cum_at_exact =
            collapse_residual(&curves, Scaling::ising_2d(), Observable::Cumulant).unwrap().residual;
        assert!(cum.residual < 0.1 * cum_at_exact, "{} against {cum_at_exact}", cum.residual);
        // Robust to where it starts: the cumulant basin is one basin.
        for far in [
            Scaling { t_c: 2.10, nu: 0.6, beta: 0.4 },
            Scaling { t_c: 2.40, nu: 1.6, beta: 0.05 },
        ] {
            let again = fit_collapse(&curves, far, Observable::Cumulant).unwrap();
            assert!(
                (again.scaling.nu - cum.scaling.nu).abs() < 0.02,
                "from {:?} the cumulant fit went to nu = {}, not {}",
                far,
                again.scaling.nu,
                cum.scaling.nu
            );
        }
    }

    /// NEGATIVE CONTROL, and its asymmetric twin in the same test.
    ///
    /// The 1D Ising chain has its transition at `T = 0`, so at every positive temperature a longer
    /// chain is strictly more disordered than a shorter one and the `U_4` curves are ordered by
    /// size everywhere. They never meet, and the estimator must say so rather than produce a
    /// number. Measured exactly by enumeration over `n = 8, 12, 16` spins: `U(16) − U(8)` stays in
    /// `[−0.195, −0.0008]` across `T ∈ [0.4, 5.2]`, all of one sign.
    ///
    /// The same three lines of code on the 2D lattices, over the same kind of window, DO find a
    /// crossing. Without that half, a `crossing` that always returned `NeverCrosses` would pass.
    #[test]
    fn a_model_with_no_transition_reports_no_crossing_and_a_2d_lattice_still_does() {
        let g: Vec<f64> = (0..25).map(|k| 0.4 + 0.2 * k as f64).collect();
        let chains: Vec<Curve> = [8usize, 12, 16]
            .iter()
            .map(|&n| {
                let r = ring(n, 1.0, 0.0);
                Curve::new(n, g.iter().map(|&t| enumerated_point(&r, t)).collect()).unwrap()
            })
            .collect();

        match crossing(&chains[0], &chains[2]) {
            Err(NoCrossing::NeverCrosses { small, large, lowest, highest }) => {
                assert_eq!((small, large), (8, 16));
                assert!(highest < 0.0, "the difference must keep ONE sign: [{lowest}, {highest}]");
                assert!(lowest > -0.25, "[{lowest}, {highest}]");
            }
            other => panic!("a chain has no finite-temperature transition, but: {other:?}"),
        }
        let refused = critical_temperature(&chains).unwrap_err();
        assert!(
            matches!(refused, NoCrossing::NeverCrosses { .. }),
            "the aggregate must refuse too, not average nothing: {refused:?}"
        );
        assert!(
            refused.to_string().contains("no finite-temperature transition"),
            "{refused}"
        );

        // ASYMMETRIC HALF: the same call on lattices that DO have a transition.
        let ok = critical_temperature(exact_curves()).unwrap();
        assert!(ok.t_c > 2.1 && ok.t_c < 2.3, "{}", ok.t_c);
    }

    /// CLOSED FORM for the collapse machinery itself: an AFFINE master curve collapses EXACTLY.
    ///
    /// Data is generated from `m(L, T) = L^{−β/ν}(a + c·(T − T_c) L^{1/ν})` with
    /// `T_c = 2.0, ν = 0.8, β = 0.2`. Because the master curve is a straight line and the residual
    /// interpolates linearly, there is no interpolation error anywhere: at the true parameters the
    /// scaled points lie on one line to the last bit, and `S` must be zero to rounding.
    ///
    /// The three sizes are given DIFFERENT grids, offset from each other, so the interpolation is
    /// actually exercised rather than evaluated at its own nodes.
    ///
    /// The solution is unique: at `T = T_c` every size sits at `x = 0`, so a different `β/ν` would
    /// have to make `a L^{β'/ν' − β/ν}` independent of `L`, and matching the `T` coefficient forces
    /// `ν' = ν`.
    #[test]
    fn an_affine_master_curve_collapses_exactly_and_the_fit_recovers_its_parameters() {
        let truth = Scaling { t_c: 2.0, nu: 0.8, beta: 0.2 };
        let (a, c) = (0.5f64, -0.15f64);
        let build = |l: usize, xs: &[f64]| -> Curve {
            let lf = l as f64;
            let pts: Vec<Point> = xs
                .iter()
                .map(|&x| {
                    let m = (a + c * x) * lf.powf(-truth.beta / truth.nu);
                    let m2 = m * m;
                    Point {
                        temperature: truth.t_c + x * lf.powf(-1.0 / truth.nu),
                        m2,
                        m4: 1.5 * m2 * m2,
                    }
                })
                .collect();
            Curve::new(l, pts).unwrap()
        };
        let curves = vec![
            build(4, &(0..11).map(|k| -1.0 + 0.2 * k as f64).collect::<Vec<f64>>()),
            build(6, &(0..11).map(|k| -0.95 + 0.19 * k as f64).collect::<Vec<f64>>()),
            build(9, &(0..13).map(|k| -0.9 + 0.15 * k as f64).collect::<Vec<f64>>()),
        ];

        let q = collapse_residual(&curves, truth, Observable::OrderParameter).unwrap();
        assert!(q.residual < 1e-24, "an affine master curve must collapse exactly: S = {}", q.residual);
        assert!(q.residual >= 0.0, "a sum of squares over a spread is never negative");
        assert!(q.covered >= 30 && q.skipped <= 5, "{q:?}");

        // ASYMMETRIC: every single-parameter perturbation is strictly worse, by orders.
        for (name, p) in [
            ("T_c", Scaling { t_c: 2.01, ..truth }),
            ("nu", Scaling { nu: 0.84, ..truth }),
            ("beta", Scaling { beta: 0.24, ..truth }),
        ] {
            let s = collapse_residual(&curves, p, Observable::OrderParameter).unwrap().residual;
            assert!(s > 1e-8, "moving {name} left S at {s}, which is no penalty at all");
        }

        // And the search finds its way back from a start that is wrong in all three.
        let fit = fit_collapse(
            &curves,
            Scaling { t_c: 2.06, nu: 1.05, beta: 0.35 },
            Observable::OrderParameter,
        )
        .unwrap();
        assert!(fit.converged, "{fit:?}");
        assert!((fit.scaling.t_c - 2.0).abs() < 1e-4, "T_c = {}", fit.scaling.t_c);
        assert!((fit.scaling.nu - 0.8).abs() < 1e-3, "nu = {}", fit.scaling.nu);
        assert!((fit.scaling.beta - 0.2).abs() < 1e-3, "beta = {}", fit.scaling.beta);
        assert!(fit.residual < 1e-12, "S = {}", fit.residual);
    }

    /// THE NORMALISATION, on the input that motivates it.
    ///
    /// `y = m L^{β/ν}`, so `β/ν → −∞` sends every scaled value to zero and a BARE `Σ(y − Y)²` to
    /// zero with it — the best collapse would be the one that flattens the data to nothing.
    /// Measured on the exact `L = 3…7` set: at `β/ν = −5` the largest `|y|` is `3.3e-3` times the
    /// largest at Onsager's exponents, so a bare sum of squares would be about `1.1e-5` of its
    /// honest value. Dividing by the spread of `y` itself removes exactly that trade, and the
    /// normalised residual goes the other way — UP by a factor of **90**.
    #[test]
    fn the_normalised_residual_refuses_the_runaway_that_flattens_the_data() {
        let curves = exact_curves();
        let truth = Scaling::ising_2d();
        let runaway = Scaling { beta: -5.0, ..truth };

        let peak = |p: Scaling| {
            collapse(curves, p, Observable::OrderParameter)
                .unwrap()
                .iter()
                .map(|s| s.y.abs())
                .fold(0.0f64, f64::max)
        };
        let (big, small) = (peak(truth), peak(runaway));
        assert!(
            small < 0.005 * big,
            "the runaway must actually flatten the data: peak {small} against {big}"
        );

        let s_true = collapse_residual(curves, truth, Observable::OrderParameter).unwrap().residual;
        let s_away =
            collapse_residual(curves, runaway, Observable::OrderParameter).unwrap().residual;
        assert!(
            s_away > 10.0 * s_true,
            "the normalised residual must PUNISH the flattening: {s_away} against {s_true}"
        );
        // And the fit does not walk there from a sane start.
        let fit = fit_collapse(curves, truth, Observable::OrderParameter).unwrap();
        assert!(fit.scaling.beta / fit.scaling.nu > 0.0, "{fit:?}");
    }

    /// Moments no single sample can produce are refused, by name and with the numbers.
    #[test]
    fn malformed_curves_are_refused_by_name() {
        let ok = Point { temperature: 1.0, m2: 0.5, m4: 0.3 };
        assert_eq!(Curve::new(0, vec![ok]).unwrap_err(), Malformed::ZeroSize);
        assert_eq!(Curve::new(4, vec![]).unwrap_err(), Malformed::NoPoints { size: 4 });

        // Cauchy-Schwarz: <m^4> >= <m^2>^2 for ANY distribution. 0.2 < 0.25 is impossible.
        let bad = Curve::new(4, vec![Point { temperature: 1.0, m2: 0.5, m4: 0.2 }]).unwrap_err();
        assert_eq!(bad, Malformed::NotAMoment { size: 4, temperature: 1.0, m2: 0.5, m4: 0.2 });
        assert!(bad.to_string().contains("Cauchy-Schwarz"), "{bad}");
        // A hair inside the slack is accepted; a real violation is not.
        assert!(Curve::new(4, vec![Point { temperature: 1.0, m2: 0.5, m4: 0.25 - 1e-17 }]).is_ok());

        // A grid that does not strictly increase.
        let flat = Curve::new(
            4,
            vec![
                Point { temperature: 1.0, m2: 0.5, m4: 0.3 },
                Point { temperature: 1.0, m2: 0.5, m4: 0.3 },
            ],
        )
        .unwrap_err();
        assert_eq!(
            flat,
            Malformed::OutOfOrder { size: 4, index: 1, previous: 1.0, temperature: 1.0 }
        );
        assert!(flat.to_string().contains("wrong interval"), "{flat}");

        assert_eq!(
            Point::from_magnetisations(2.0, &[]).unwrap_err(),
            Malformed::NoDraws { temperature: 2.0 }
        );
        let nf = Point::from_magnetisations(2.0, &[0.1, f64::NAN]).unwrap_err();
        assert!(matches!(nf, Malformed::NotFinite { index: 1, .. }), "{nf:?}");
        // Every variant is an error and prints something with the numbers in it.
        for e in [bad, flat, nf] {
            let _: &dyn core::error::Error = &e;
            assert!(e.to_string().len() > 40, "{e}");
        }
    }

    /// Every way two curves fail to cross, refused by name rather than answered with a number.
    #[test]
    fn crossings_that_are_not_crossings_are_refused_by_name() {
        let pts = |ts: &[f64], u: &[f64]| -> Vec<Point> {
            // Build moments that realise a chosen U_4: m4 = 3 m2^2 (1 - U).
            ts.iter()
                .zip(u)
                .map(|(&t, &uu)| {
                    let m2 = 0.5;
                    Point { temperature: t, m2, m4: 3.0 * m2 * m2 * (1.0 - uu) }
                })
                .collect()
        };
        let ts = [1.0, 2.0, 3.0];
        let a = Curve::new(4, pts(&ts, &[0.6, 0.5, 0.4])).unwrap();
        let b = Curve::new(8, pts(&ts, &[0.65, 0.45, 0.25])).unwrap();

        let c = crossing(&a, &b).unwrap();
        assert_eq!((c.small, c.large), (4, 8));
        // d(T) = U_8 - U_4 is +0.05 at T=1 and -0.05 at T=2, so the root is the midpoint and both
        // interpolants read 0.55 there. Worked by hand, which is the point of a fixture this small.
        assert!((c.temperature - 1.5).abs() < 1e-12, "{c:?}");
        assert!((c.u4 - 0.55).abs() < 1e-12, "{c:?}");
        assert_eq!(c.bracket, (1.0, 2.0), "the honest resolution is the grid interval");
        // Order of the arguments does not change the answer.
        assert_eq!(crossing(&b, &a).unwrap(), c);

        assert_eq!(
            crossing(&a, &Curve::new(4, pts(&ts, &[0.1, 0.2, 0.3])).unwrap()).unwrap_err(),
            NoCrossing::SameSize { size: 4 }
        );
        let far = Curve::new(8, pts(&[9.0, 10.0, 11.0], &[0.6, 0.5, 0.4])).unwrap();
        assert!(matches!(crossing(&a, &far).unwrap_err(), NoCrossing::Disjoint { .. }));

        // Two sign changes in one window: a number would be a lie about the resolution.
        let wiggle = Curve::new(
            8,
            pts(&[1.0, 1.5, 2.0, 2.5, 3.0], &[0.65, 0.45, 0.55, 0.40, 0.25]),
        )
        .unwrap();
        match crossings(&a, &wiggle) {
            Ok(v) => assert!(v.len() >= 2, "the fixture must actually wiggle: {v:?}"),
            Err(e) => panic!("{e}"),
        }
        let amb = crossing(&a, &wiggle).unwrap_err();
        assert!(matches!(amb, NoCrossing::Ambiguous { .. }), "{amb:?}");
        assert!(amb.to_string().contains("changes sign"), "{amb}");

        assert_eq!(
            critical_temperature(std::slice::from_ref(&a)).unwrap_err(),
            NoCrossing::TooFewSizes { sizes: 1 }
        );
        assert_eq!(
            critical_temperature(&[a.clone(), b.clone(), a.clone()]).unwrap_err(),
            NoCrossing::DuplicateSize { size: 4 }
        );
        for e in [NoCrossing::SameSize { size: 4 }, NoCrossing::TooFewSizes { sizes: 1 }] {
            let _: &dyn core::error::Error = &e;
            assert!(e.to_string().len() > 40, "{e}");
        }
    }

    /// Collapses that cannot be scored are refused, including the two that are easy to answer
    /// wrongly: a negative `ν` (which unsorts every grid) and a flat data set (which is an empty
    /// measurement, not a perfect collapse).
    #[test]
    fn collapses_that_cannot_be_scored_are_refused_by_name() {
        let flat = |l: usize, m2: f64| {
            Curve::new(
                l,
                (0..5)
                    .map(|k| Point {
                        temperature: 1.0 + f64::from(k),
                        m2,
                        m4: 1.5 * m2 * m2,
                    })
                    .collect(),
            )
            .unwrap()
        };
        let ok = Scaling { t_c: 2.0, nu: 1.0, beta: 0.1 };
        let cs = vec![flat(4, 0.4), flat(8, 0.4)];

        assert_eq!(
            collapse_residual(&cs[..1], ok, Observable::OrderParameter).unwrap_err(),
            CollapseError::TooFewSizes { sizes: 1 }
        );
        assert_eq!(
            collapse_residual(&[flat(4, 0.4), flat(4, 0.3)], ok, Observable::OrderParameter)
                .unwrap_err(),
            CollapseError::DuplicateSize { size: 4 }
        );
        let neg = collapse_residual(&cs, Scaling { nu: -1.0, ..ok }, Observable::OrderParameter)
            .unwrap_err();
        assert_eq!(neg, CollapseError::NonPositiveNu { nu: -1.0 });
        assert!(neg.to_string().contains("wrong interval"), "{neg}");
        assert!(matches!(
            collapse_residual(&cs, Scaling { t_c: f64::NAN, ..ok }, Observable::OrderParameter),
            Err(CollapseError::NotFinite { .. })
        ));

        // A flat order parameter with beta = 0 gives every point the same y: no spread, no
        // denominator, and NOT a perfect collapse.
        let none = collapse_residual(&cs, Scaling { beta: 0.0, ..ok }, Observable::Cumulant)
            .unwrap_err();
        assert!(matches!(none, CollapseError::NoSpread { .. }), "{none:?}");
        assert!(none.to_string().contains("empty data set"), "{none}");

        // Windows that do not overlap in the scaled coordinate: nothing is compared with anything.
        let near = Curve::new(
            4,
            (0..5)
                .map(|k| Point { temperature: 1.0 + 0.1 * f64::from(k), m2: 0.4, m4: 0.3 })
                .collect(),
        )
        .unwrap();
        let far = Curve::new(
            8,
            (0..5)
                .map(|k| Point { temperature: 9.0 + 0.1 * f64::from(k), m2: 0.4, m4: 0.3 })
                .collect(),
        )
        .unwrap();
        let apart = collapse_residual(
            &[near, far],
            Scaling { t_c: 0.0, nu: 1.0, beta: 0.1 },
            Observable::OrderParameter,
        );
        assert_eq!(apart, Err(CollapseError::NoOverlap { points: 10 }), "{apart:?}");
        for e in [neg, none] {
            let _: &dyn core::error::Error = &e;
        }
    }

    /// A frozen coordinate stays frozen, and the cumulant collapse cannot be talked into moving
    /// `β` — it does not contain one.
    ///
    /// The second half is the asymmetric one: handing [`Observable::Cumulant`] a large `β` step is
    /// exactly how a fit could report that it had searched an axis it never touched.
    #[test]
    fn a_frozen_step_pins_its_coordinate_and_the_cumulant_never_moves_beta() {
        let curves: Vec<Curve> = exact_curves()[1..].to_vec();
        let start = Scaling { t_c: 2.2, nu: 1.2, beta: 0.4 };

        let shouted = FitOptions { step_beta_nu: 5.0, ..FitOptions::default() };
        let cum = fit_collapse_with(&curves, start, Observable::Cumulant, shouted).unwrap();
        assert_eq!(cum.scaling.beta, 0.4, "the cumulant collapse has no beta to fit");
        assert_eq!(cum.observable, Observable::Cumulant);
        assert!(cum.scaling.nu != start.nu && cum.scaling.t_c != start.t_c, "{:?}", cum.scaling);
        // Identical to the run that never mentioned a beta step: the axis really is unused.
        let quiet = FitOptions { step_beta_nu: 0.0, ..FitOptions::default() };
        let same = fit_collapse_with(&curves, start, Observable::Cumulant, quiet).unwrap();
        assert_eq!(same.scaling, cum.scaling);
        assert_eq!(same.evaluations, cum.evaluations);

        // A zero step on the order parameter pins that coordinate and nothing else.
        let pinned = FitOptions { step_t_c: 0.0, ..FitOptions::default() };
        let f = fit_collapse_with(&curves, start, Observable::OrderParameter, pinned).unwrap();
        assert_eq!(f.scaling.t_c, 2.2);
        assert!(f.scaling.nu != start.nu, "{:?}", f.scaling);
        assert!(f.residual <= f.start_residual);

        // An evaluation cap is reported, not silently treated as convergence.
        let capped = FitOptions { max_evaluations: 12, ..FitOptions::default() };
        let short = fit_collapse_with(&curves, start, Observable::Cumulant, capped).unwrap();
        assert!(!short.converged, "{short:?}");
        assert!(short.evaluations <= 12 + 6, "{}", short.evaluations);
    }

    /// PUBLISHED NUMBER. Hyperscaling turns `ν = 1` and `β = 1/8` into `γ = 7/4`, exactly.
    ///
    /// `γ = d·ν − 2β` with `d = 2` gives `2 − 1/4`, and `7/4` is Onsager's susceptibility exponent
    /// for this model. The value is representable in `f64` and the arithmetic is exact, so this is
    /// asserted with `==` rather than a tolerance — an epsilon here would not soften the test, it
    /// would delete it.
    ///
    /// The two-stage fit of the exact `L = 4…7` lattices carries its own error through: with
    /// `ν = 1.0555` from the cumulant collapse and `β = 0.1055` from the order parameter at that
    /// `T_c` and `ν`, `γ` comes out `1.900`, **+8.6%**, against `ν`'s own +5.5%.
    ///
    /// **It is worse than `ν` and better than `β`, and the first version of this test asserted it
    /// was worse than both.** `γ − 7/4 = 2(ν − 1) − 2(β − 1/8)` exactly, so the two errors combine
    /// in ABSOLUTE terms — `2(+0.0555) − 2(−0.0195) = +0.150` — and they push the same way here
    /// rather than cancelling. But `β = 1/8` is a small number, so its 15.6% is only `0.0195` of
    /// absolute error, and measured against `7/4` the total is 8.6%. A relative error is not a
    /// quantity that propagates, and writing as if it were is how that assertion came to be
    /// wrong.
    #[test]
    fn hyperscaling_turns_onsagers_nu_and_yangs_beta_into_onsagers_gamma() {
        assert_eq!(Scaling::ising_2d().gamma(2), 1.75);
        assert_eq!(Scaling::ising_2d().gamma(2), 7.0 / 4.0);
        // gamma/nu = d - 2 beta/nu is the same statement, and is 7/4 here because nu is 1.
        let s = Scaling::ising_2d();
        assert_eq!(s.gamma(2) / s.nu, 2.0 - 2.0 * s.beta / s.nu);
        // Mean field (nu = 1/2, beta = 1/2) at its own upper critical dimension gives gamma = 1.
        assert_eq!(Scaling { t_c: 1.0, nu: 0.5, beta: 0.5 }.gamma(4), 1.0);

        // And through the two-stage fit of the exact lattices.
        let curves: Vec<Curve> = exact_curves()[1..].to_vec();
        let cum =
            fit_collapse(&curves, Scaling::ising_2d(), Observable::Cumulant).unwrap();
        let pinned = FitOptions { step_t_c: 0.0, step_inv_nu: 0.0, ..FitOptions::default() };
        let both =
            fit_collapse_with(&curves, cum.scaling, Observable::OrderParameter, pinned).unwrap();
        let g = both.scaling.gamma(2);
        assert!(
            (g - 1.75).abs() < 1.75 * 0.10,
            "gamma = {g} from nu = {} and beta = {}, against Onsager's 7/4",
            both.scaling.nu,
            both.scaling.beta
        );
        // The errors ADD in absolute terms -- an identity, since gamma is linear in both -- and
        // here they push the same way rather than cancelling.
        let (d_nu, d_beta) = (both.scaling.nu - 1.0, both.scaling.beta - 0.125);
        assert!(d_nu > 0.0 && d_beta < 0.0, "both push gamma up: {d_nu}, {d_beta}");
        assert!(
            ((g - 1.75) - (2.0 * d_nu - 2.0 * d_beta)).abs() < 1e-12,
            "gamma - 7/4 must be exactly 2(nu-1) - 2(beta-1/8): {} against {}",
            g - 1.75,
            2.0 * d_nu - 2.0 * d_beta
        );
        // ASYMMETRIC, and it is the RELATIVE errors that do not simply compound: gamma is worse
        // than nu, because d*nu doubles nu's error and beta's does not cancel it -- but it is
        // BETTER than beta's 15.6%, because beta is a small number and its relative error is a
        // small absolute one. A relative error is not a quantity that propagates.
        let e_gamma = (g - 1.75).abs() / 1.75;
        let (e_nu, e_beta) = (d_nu.abs(), d_beta.abs() / 0.125);
        assert!(e_gamma > e_nu, "gamma {e_gamma} should be worse than nu {e_nu}");
        assert!(e_gamma < e_beta, "gamma {e_gamma} should be better than beta {e_beta}");
    }

    /// The scaled coordinates themselves: `x = 0` at `T_c` for every size whatever `ν` is, and the
    /// vertical carries `L^{β/ν}` only for the order parameter.
    #[test]
    fn scaled_coordinates_are_what_the_scaling_form_says() {
        let exact = Scaling::ising_2d();
        assert_eq!(exact.nu, 1.0);
        assert_eq!(exact.beta, 0.125);
        assert_eq!(exact.t_c, ising_2d_tc());

        let g = [2.0, ising_2d_tc(), 2.5];
        let curves: Vec<Curve> = [4usize, 9]
            .iter()
            .map(|&l| {
                Curve::new(
                    l,
                    g.iter()
                        .map(|&t| Point { temperature: t, m2: 0.4, m4: 0.3 })
                        .collect(),
                )
                .unwrap()
            })
            .collect();

        let pts = collapse(&curves, exact, Observable::OrderParameter).unwrap();
        assert_eq!(pts.len(), 6);
        for p in pts.iter().filter(|p| p.size == 4) {
            // y = sqrt(0.4) * 4^(1/8) for every point of this size.
            assert!((p.y - 0.4f64.sqrt() * 4f64.powf(0.125)).abs() < 1e-15, "{p:?}");
        }
        let at_tc: Vec<&Scaled> = pts.iter().filter(|p| p.x.abs() < 1e-15).collect();
        assert_eq!(at_tc.len(), 2, "both sizes sit at x = 0 when T = T_c");
        assert!(at_tc[0].y < at_tc[1].y, "9^(1/8) is above 4^(1/8)");

        // The cumulant carries no power of L at all, so the two sizes land on top of each other.
        let cu = collapse(&curves, exact, Observable::Cumulant).unwrap();
        let tops: Vec<f64> = cu.iter().filter(|p| p.x.abs() < 1e-15).map(|p| p.y).collect();
        assert_eq!(tops[0], tops[1], "the cumulant is dimensionless");
        // And x stretches with L^(1/nu): at nu = 1 the ratio of the two sizes' x is 9/4.
        let far = |size: usize, v: &[Scaled]| {
            v.iter().filter(|p| p.size == size).map(|p| p.x).fold(0.0f64, f64::max)
        };
        assert!((far(9, &cu) / far(4, &cu) - 9.0 / 4.0).abs() < 1e-12);
    }

    /// Interpolation answers inside the measured grid and REFUSES outside it, because a crossing
    /// found on an extrapolated straight line reads exactly like a real one.
    #[test]
    fn interpolation_refuses_to_extrapolate() {
        let c = Curve::new(
            4,
            vec![
                Point { temperature: 1.0, m2: 0.5, m4: 3.0 * 0.25 * 0.4 },
                Point { temperature: 2.0, m2: 0.5, m4: 3.0 * 0.25 * 0.8 },
            ],
        )
        .unwrap();
        assert!((c.binder()[0] - 0.6).abs() < 1e-15);
        assert!((c.binder()[1] - 0.2).abs() < 1e-15);
        assert!((c.binder_at(1.5).unwrap() - 0.4).abs() < 1e-15, "the midpoint of a straight line");
        assert_eq!(c.binder_at(1.0), Some(c.binder()[0]), "a grid point returns its own value");
        assert_eq!(c.binder_at(2.0), Some(c.binder()[1]));
        assert_eq!(c.binder_at(0.999), None);
        assert_eq!(c.binder_at(2.001), None);
        assert_eq!(c.binder_at(f64::NAN), None);
        assert_eq!(c.temperature_range(), (1.0, 2.0));
        assert_eq!(c.order_at(1.0), Some(0.5f64.sqrt()));
        assert_eq!(c.order_at(3.0), None);
        assert_eq!(c.size(), 4);
        assert_eq!(c.points().len(), 2);
        assert_eq!(c.temperatures(), &[1.0, 2.0]);
        assert_eq!(c.order_parameter().len(), 2);
    }
}
