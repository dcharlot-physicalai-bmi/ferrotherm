//! Umbrella sampling, WHAM and metadynamics — free energy along a chosen coordinate.
//!
//! [`crate::wanglandau`] and [`crate::tmmc`] both estimate a density of states, which is the free
//! energy along ENERGY. Energy is the one coordinate a spin model hands you for free, and it is
//! usually not the one the question is about. "How hard is it to turn this magnet over" is a
//! question about MAGNETISATION: the barrier lives in `F(m)`, and `g(E)` cannot answer it, because
//! states with the magnet up and states with it down sit at the same energies and are pooled into
//! the same bin.
//!
//! So this module runs a BIASED ensemble along a chosen reaction coordinate — here `M(s) = sum_i
//! s_i`, the coordinate an Ising model actually has — and unbiases the result. Three tools, in the
//! order they were invented, each the answer to the previous one's problem.
//!
//! # Umbrella sampling (Torrie & Valleau, J. Comput. Phys. 23:187, 1977)
//!
//! Add a harmonic restraint `V_w(m) = (k/2)(m − m_0)^2` to the Hamiltonian and sample
//! `exp(−beta (E(s) + V_w(M(s))))`. The chain is then held in a slice of the coordinate it would
//! otherwise never visit, so a barrier ten `kT` high is sampled as well as the well beside it. One
//! run per window, windows chosen to overlap, and every window's histogram is a biased estimate of
//! the same `P(m)`.
//!
//! # WHAM (Kumar, Bouzida, Swendsen, Kollman & Rosenberg, J. Comput. Chem. 13:1011, 1992)
//!
//! Unbiasing one window is one line — divide out `exp(−beta V_w)` — and stitching the windows is
//! not, because each is unbiased only up to its own unknown constant `f_w`, and those constants are
//! exactly what naive overlap-matching gets wrong. WHAM solves for all of them at once, by
//! minimising the variance of the pooled estimate. The result is a pair of coupled equations,
//!
//! ```text
//!   P(m)           = sum_w H_w(m)  /  sum_w N_w exp(−beta (V_w(m) − f_w))
//!   exp(−beta f_w) = sum_m P(m) exp(−beta V_w(m))
//! ```
//!
//! with `H_w(m)` the counts window `w` recorded in bin `m` and `N_w` its total. Iterated to a fixed
//! point, then `F(m) = −(1/beta) ln P(m)`. Only differences matter, and the gauge is fixed by
//! normalising `P` to sum to one over the bins.
//!
//! **The fixed point is exact, which is what makes it testable without statistics.** Substitute the
//! true `P` and the true `f_w` into the right-hand side, with each `H_w` set to its own exact
//! expectation, and every term cancels: the numerator becomes `P(m)` times the denominator. So a
//! WHAM solver fed exact biased histograms must return the exact profile, and
//! `wham_recovers_the_enumerated_profile_from_exact_histograms` runs that against a sum over all
//! `2^n` states.
//!
//! # Metadynamics (Laio & Parrinello, PNAS 99:12562, 2002)
//!
//! Umbrella sampling needs the windows placed before the run, which needs a guess at where the
//! barrier is. Metadynamics needs no guess: deposit a Gaussian hill on the coordinate wherever the
//! walk currently is, so the system fills the well it is in and is pushed out of it. The bias is
//! the history of the walk, and it grows into the profile's negative:
//!
//! ```text
//!   V(m, t)  ->  −F(m) + const
//! ```
//!
//! which is the statement `metadynamics_bias_is_minus_the_enumerated_profile` checks, after
//! removing the mean from both sides so the additive constant is not what makes it pass.
//!
//! The 2002 algorithm deposits hills of constant height and does not settle — it oscillates around
//! `−F` forever, with an amplitude set by the hill height. Well-tempered metadynamics (Barducci,
//! Bussi & Parrinello, Phys. Rev. Lett. 100:020603, 2008) scales each hill by
//! `exp(−beta V(m)/(gamma − 1))`, so deposition slows where the bias is already deep and the bias
//! converges to `−(1 − 1/gamma) F(m)`. Both are here: [`Hills::bias_factor`] of `None` is the
//! original, `Some(gamma)` the tempered form, and [`Meta::profile`] applies whichever rescaling the
//! choice implies.
//!
//! # The coordinate is discrete, so there is no binning choice to get wrong
//!
//! `M(s)` on `n` spins takes exactly `n + 1` values, `−n, −n+2, ..., n`. [`Magnetisation`] is that
//! grid. Same decision [`crate::wanglandau`] made for energy levels and for the same reason: a bin
//! edge that splits a level produces a histogram no amount of sampling can flatten. It also makes
//! hill deposition exact rather than interpolated — every hill is evaluated at every grid point,
//! and the walk is never anywhere else.
//!
//! # Where the errors are, and where the asserts are
//!
//! A parameter the caller chose wrong — a negative force constant, a hill of zero width, a bias
//! factor below one — is a programmer error and panics at the constructor, as
//! [`crate::graph::GraphBuilder::couple`] does. A DATA shape the solver cannot use — a window with
//! no samples, a ragged histogram, a model too big to enumerate — is an [`Invalid`], because a
//! caller stitching histograms it did not produce cannot check those in advance.

use crate::graph::Graph;
use crate::rng::Pcg;

/// The most spins [`exact_profile`] will enumerate over.
///
/// A memory statement as much as a time one: the enumeration holds one `f64` per state for each end
/// of its bracket, so twenty spins is 16 MB and every further spin doubles it.
pub const MAX_ENUM: usize = 20;

/// Default convergence tolerance for [`solve`]: the largest change in any `beta f_w` between
/// iterations, which is zero at the fixed point.
pub const TOL: f64 = 1e-13;

/// Default iteration cap for [`solve`]. WHAM's fixed point converges linearly, so this is generous
/// where the windows overlap and a real limit where they barely do.
pub const MAX_ITERS: usize = 200_000;

/// Why a set of histograms, or a model, could not be turned into a profile.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Invalid {
    /// No windows were given. There is nothing to stitch.
    NoWindows,
    /// The bias matrix and the histograms describe different numbers of windows.
    Mismatched {
        /// Rows in the bias matrix.
        windows: usize,
        /// Histograms supplied.
        histograms: usize,
    },
    /// A row does not cover every bin of the coordinate.
    Ragged {
        /// The offending window.
        window: usize,
        /// Its length.
        len: usize,
        /// The bin count every row must match.
        bins: usize,
    },
    /// This window recorded no visits, so nothing determines its free energy.
    EmptyWindow(usize),
    /// A bias value was not finite, at this window and bin.
    NotFinite {
        /// The window whose bias was not finite.
        window: usize,
        /// The bin it was evaluated at.
        bin: usize,
    },
    /// An inverse temperature that is not finite and positive. `F = −(1/beta) ln P` divides by it.
    BadBeta(f64),
    /// A model with more spins than [`MAX_ENUM`], which no enumeration can reach.
    TooLarge {
        /// Spins in the model.
        n: usize,
        /// The cap.
        cap: usize,
    },
}

impl core::fmt::Display for Invalid {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Invalid::NoWindows => write!(f, "WHAM needs at least one window, got none"),
            Invalid::Mismatched { windows, histograms } => {
                write!(f, "{windows} windows of bias but {histograms} histograms")
            }
            Invalid::Ragged { window, len, bins } => {
                write!(f, "window {window} covers {len} bins, not the {bins} the coordinate has")
            }
            Invalid::EmptyWindow(w) => write!(f, "window {w} recorded no visits"),
            Invalid::NotFinite { window, bin } => {
                write!(f, "the bias of window {window} at bin {bin} is not finite")
            }
            Invalid::BadBeta(b) => write!(
                f,
                "a free energy is divided by beta, which must be finite and positive, not {b}"
            ),
            Invalid::TooLarge { n, cap } => {
                write!(f, "enumerating {n} spins means 2^{n} states, past a cap of 2^{cap}")
            }
        }
    }
}

impl core::error::Error for Invalid {}

// ---- the coordinate ----------------------------------------------------------------------------

/// The reaction coordinate: total magnetisation `M(s) = sum_i s_i`, and the grid of values it takes.
///
/// On `n` spins that is exactly `n + 1` values, `−n, −n+2, ..., n`, so bin `b` holds `2b − n` and
/// nothing falls between two bins. Flipping one spin moves `M` by `±2`, which is one bin.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Magnetisation {
    n: usize,
}

impl Magnetisation {
    /// The coordinate of an `n`-spin model.
    #[must_use]
    pub fn new(n: usize) -> Self {
        Magnetisation { n }
    }

    /// Spins in the model.
    #[must_use]
    pub fn spins(&self) -> usize {
        self.n
    }

    /// How many values the coordinate takes: `n + 1`.
    #[must_use]
    pub fn bins(&self) -> usize {
        self.n + 1
    }

    /// The coordinate value of bin `b`: `2b − n`.
    #[must_use]
    pub fn value(&self, b: usize) -> f64 {
        2.0 * b as f64 - self.n as f64
    }

    /// Every bin's coordinate value, in bin order.
    #[must_use]
    pub fn values(&self) -> Vec<f64> {
        (0..self.bins()).map(|b| self.value(b)).collect()
    }

    /// The bin holding magnetisation `m`.
    ///
    /// # Panics
    ///
    /// If `m` is outside `−n..=n` or has the wrong parity. Both are impossible for a magnetisation
    /// actually computed from `n` spins, so reaching either means the caller is binning something
    /// that is not this coordinate.
    #[must_use]
    pub fn bin(&self, m: i64) -> usize {
        let n = self.n as i64;
        assert!(m >= -n && m <= n, "magnetisation {m} is outside -{n}..={n}");
        assert!((m + n) % 2 == 0, "magnetisation {m} has the wrong parity for {n} spins");
        ((m + n) / 2) as usize
    }
}

/// A harmonic umbrella window: the restraint `V(m) = (k/2)(m − m_0)^2`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Window {
    /// Force constant, in energy per squared coordinate unit. Zero is the unbiased ensemble.
    pub k: f64,
    /// Where the restraint is centred, in coordinate units. Need not be on the grid.
    pub m0: f64,
}

impl Window {
    /// A window of stiffness `k` centred at `m0`.
    ///
    /// # Panics
    ///
    /// If `k` is negative or either argument is not finite. A negative force constant is an
    /// anti-restraint: it pushes the chain to whichever end of the coordinate it started nearest
    /// and never brings it back, which looks like a converged window and is one state.
    #[must_use]
    pub fn new(k: f64, m0: f64) -> Self {
        assert!(
            k.is_finite() && k >= 0.0,
            "a force constant must be finite and non-negative, got {k}"
        );
        assert!(m0.is_finite(), "a window centre must be finite, got {m0}");
        Window { k, m0 }
    }

    /// The restraint energy at coordinate value `m`.
    #[must_use]
    pub fn bias(&self, m: f64) -> f64 {
        let d = m - self.m0;
        0.5 * self.k * d * d
    }

    /// The restraint evaluated at every bin of `coord`, which is the form [`solve`] takes.
    #[must_use]
    pub fn on_grid(&self, coord: &Magnetisation) -> Vec<f64> {
        (0..coord.bins()).map(|b| self.bias(coord.value(b))).collect()
    }
}

/// A ladder of `count` windows evenly spanning the coordinate, all of stiffness `k`.
///
/// The ends sit exactly on `−n` and `n`, so the extreme bins — where an unbiased chain of a cold
/// ferromagnet spends all of its time — are covered by a window centred on them rather than by the
/// tail of one that is not.
///
/// # Panics
///
/// If `count` is zero, or through [`Window::new`].
#[must_use]
pub fn ladder(coord: &Magnetisation, count: usize, k: f64) -> Vec<Window> {
    assert!(count > 0, "a ladder needs at least one window");
    let n = coord.spins() as f64;
    if count == 1 {
        return vec![Window::new(k, 0.0)];
    }
    (0..count).map(|w| Window::new(k, -n + 2.0 * n * w as f64 / (count - 1) as f64)).collect()
}

// ---- profiles ----------------------------------------------------------------------------------

/// A free-energy profile along the coordinate: `F(m)` for every bin.
///
/// `log_p` is normalised to sum to one over the bins, so `free_energy` is absolute rather than
/// defined up to a constant, and two profiles over the same bins compare bin by bin. A bin no
/// estimator ever reached has `log_p` of `−inf` and `free_energy` of `+inf`, which is the honest
/// answer: nothing was measured there.
#[derive(Clone, Debug)]
pub struct Profile {
    /// Coordinate value of each bin.
    pub coord: Vec<f64>,
    /// `ln P(m)`, normalised so `sum_m P(m) = 1`.
    pub log_p: Vec<f64>,
    /// `F(m) = −(1/beta) ln P(m)`, in energy units.
    pub free_energy: Vec<f64>,
    /// The inverse temperature the profile is at. A free energy without one is a shape.
    pub beta: f64,
}

impl Profile {
    /// A profile from unnormalised log-weights, normalised on the way in.
    ///
    /// # Panics
    ///
    /// If the two slices differ in length, or `beta` is not finite and positive.
    #[must_use]
    pub fn new(coord: Vec<f64>, log_w: &[f64], beta: f64) -> Self {
        assert_eq!(coord.len(), log_w.len(), "one log-weight per bin");
        assert!(beta.is_finite() && beta > 0.0, "beta must be finite and positive, got {beta}");
        let norm = log_sum_exp(log_w);
        let log_p: Vec<f64> = log_w.iter().map(|x| x - norm).collect();
        let free_energy = log_p.iter().map(|x| -x / beta).collect();
        Profile { coord, log_p, free_energy, beta }
    }

    /// `F(m)` with its mean over the reachable bins removed.
    ///
    /// The comparison to make when a profile is known only up to an additive constant — which is
    /// what a metadynamics bias is, and what any two estimators normalised over different bin sets
    /// are. Bins that were never reached stay `+inf` and are left out of the mean.
    #[must_use]
    pub fn centred(&self) -> Vec<f64> {
        let finite: Vec<f64> = self.free_energy.iter().copied().filter(|x| x.is_finite()).collect();
        if finite.is_empty() {
            return self.free_energy.clone();
        }
        let mean = finite.iter().sum::<f64>() / finite.len() as f64;
        self.free_energy.iter().map(|x| x - mean).collect()
    }

    /// The tallest barrier on the profile: `max F − min F` over the reachable bins.
    ///
    /// The number umbrella sampling exists to measure, and the one that says whether a tolerance on
    /// the profile is tight or decorative.
    #[must_use]
    pub fn barrier(&self) -> f64 {
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for &x in &self.free_energy {
            if x.is_finite() {
                lo = lo.min(x);
                hi = hi.max(x);
            }
        }
        if hi < lo { 0.0 } else { hi - lo }
    }
}

/// The enumerated profile, and the partition function it was normalised by.
#[derive(Clone, Debug)]
pub struct Enumerated {
    /// `F(m)` from a sum over all `2^n` states.
    pub profile: Profile,
    /// `ln Z`, bracketed: `(lower, upper)`, each certainly on its own side of the truth.
    pub log_z: (f64, f64),
}

/// `F(m)` for every bin by summing over all `2^n` states — the oracle everything here is measured
/// against.
///
/// # The bracket on `ln Z` is a bound, so it is accumulated as one
///
/// The bin weights are summed through [`crate::round::sum_down`] and [`crate::round::sum_up`],
/// because this crate has already shipped a "lower bound" that sat above the optimum for summing in
/// round-to-nearest. Two roundings sit below the accumulation and are handled rather than glossed:
/// forming `−beta E` costs a relative `|beta E| eps`, which `exp` carries through as a relative
/// error of the same size, and `exp` costs one more ulp — so each term is widened by
/// `(1 + |beta E|) eps` on both sides before it is summed. What remains assumed is that `exp` is
/// faithfully rounded, which is true of every libm this crate has been run on and is not something
/// `std` promises. `the_enumerated_bracket_contains_the_elimination_log_z` checks the result
/// against [`crate::exact::Elimination::log_partition`], and checks that it is still NARROW — a
/// bound wide enough to be useless is the other way to get this wrong, and the one
/// [`crate::round`]'s own tests were written after.
///
/// # Errors
///
/// [`Invalid::BadBeta`] for an inverse temperature a free energy cannot be divided by, and
/// [`Invalid::TooLarge`] past [`MAX_ENUM`] spins.
pub fn exact_profile(g: &Graph, beta: f64) -> Result<Enumerated, Invalid> {
    if !(beta.is_finite() && beta > 0.0) {
        return Err(Invalid::BadBeta(beta));
    }
    if g.n > MAX_ENUM {
        return Err(Invalid::TooLarge { n: g.n, cap: MAX_ENUM });
    }
    let coord = Magnetisation::new(g.n);
    let bins = coord.bins();
    let mut lo: Vec<Vec<f64>> = vec![Vec::new(); bins];
    let mut hi: Vec<Vec<f64>> = vec![Vec::new(); bins];
    let mut s = vec![-1i8; g.n];
    for mask in 0u64..(1u64 << g.n) {
        let mut m = 0i64;
        for i in 0..g.n {
            let v = if (mask >> i) & 1 == 1 { 1i8 } else { -1 };
            s[i] = v;
            m += i64::from(v);
        }
        let (a, b) = weight_bracket(beta, g.energy(&s));
        let k = coord.bin(m);
        lo[k].push(a);
        hi[k].push(b);
    }
    let bin_lo: Vec<f64> = lo.iter().map(|v| crate::round::sum_down(v)).collect();
    let bin_hi: Vec<f64> = hi.iter().map(|v| crate::round::sum_up(v)).collect();
    // A lower bound on every bin, summed downward, is a lower bound on their total.
    let z_lo = crate::round::sum_down(&bin_lo);
    let z_hi = crate::round::sum_up(&bin_hi);
    let log_w: Vec<f64> = (0..bins).map(|b| (0.5 * (bin_lo[b] + bin_hi[b])).ln()).collect();
    Ok(Enumerated {
        profile: Profile::new(coord.values(), &log_w, beta),
        // `ln` rounds too, in an unknown direction; one ulp outward keeps each end on its own side.
        log_z: (z_lo.ln().next_down(), z_hi.ln().next_up()),
    })
}

/// `exp(−beta E)`, widened to a bracket that contains the exact value.
///
/// See [`exact_profile`] for what is and is not assumed.
fn weight_bracket(beta: f64, e: f64) -> (f64, f64) {
    let x = -beta * e;
    let w = x.exp();
    let rel = (1.0 + x.abs()) * f64::EPSILON;
    (w * (1.0 - rel), w * (1.0 + rel))
}

// ---- the biased walk ---------------------------------------------------------------------------

/// A single-spin Metropolis walk under an arbitrary bias on the coordinate.
///
/// Shared by [`Umbrella`] and [`Meta`]: the only thing that differs between them is where the bias
/// grid comes from, so it is passed in per step rather than owned.
struct Walk<'g> {
    g: &'g Graph,
    beta: f64,
    coord: Magnetisation,
    s: Vec<i8>,
    m: i64,
    rng: Pcg,
}

impl<'g> Walk<'g> {
    /// A walk started at the reachable magnetisation nearest `m0`, with WHICH spins point up chosen
    /// by the seed rather than by index — a block of contiguous up-spins is a domain wall, and
    /// starting every window inside one biases the burn-in the same way in every window.
    fn new(g: &'g Graph, beta: f64, m0: f64, seed: u64) -> Self {
        assert!(beta.is_finite() && beta > 0.0, "beta must be finite and positive, got {beta}");
        let n = g.n;
        let mut rng = Pcg::new(seed, 0x_11B4);
        let up = (((m0 + n as f64) / 2.0).round() as i64).clamp(0, n as i64) as usize;
        let mut order: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let j = ((rng.f64() * (i + 1) as f64) as usize).min(i);
            order.swap(i, j);
        }
        let mut s = vec![-1i8; n];
        for &i in order.iter().take(up) {
            s[i] = 1;
        }
        let m = s.iter().map(|&v| i64::from(v)).sum();
        Walk { g, beta, coord: Magnetisation::new(n), s, m, rng }
    }

    /// One proposed flip, accepted with `min(1, exp(−beta (dE + dV)))`; `bias[b]` is the bias energy
    /// at bin `b`. Returns whether the flip was taken.
    fn step(&mut self, bias: &[f64]) -> bool {
        let n = self.g.n;
        let i = ((self.rng.f64() * n as f64) as usize).min(n - 1);
        let si = f64::from(self.s[i]);
        // E = -f_i s_i + (terms without i), so flipping s_i moves the energy by 2 f_i s_i.
        let de = 2.0 * self.g.field(i, &self.s) * si;
        let m_new = self.m - 2 * i64::from(self.s[i]);
        let dv = bias[self.coord.bin(m_new)] - bias[self.coord.bin(self.m)];
        let arg = -self.beta * (de + dv);
        if arg >= 0.0 || self.rng.f64() < arg.exp() {
            self.s[i] = -self.s[i];
            self.m = m_new;
            true
        } else {
            false
        }
    }
}

/// A biased Metropolis walk inside one umbrella window, recording a histogram of the coordinate.
pub struct Umbrella<'g> {
    walk: Walk<'g>,
    window: Window,
    bias: Vec<f64>,
    hist: Vec<u64>,
    proposed: u64,
    accepted: u64,
}

impl<'g> Umbrella<'g> {
    /// A walk in `window` at `beta`, started near the window's centre.
    ///
    /// # Panics
    ///
    /// If `beta` is not finite and positive.
    #[must_use]
    pub fn new(g: &'g Graph, beta: f64, window: Window, seed: u64) -> Self {
        let walk = Walk::new(g, beta, window.m0, seed);
        let bias = window.on_grid(&walk.coord);
        let bins = bias.len();
        Umbrella { walk, window, bias, hist: vec![0; bins], proposed: 0, accepted: 0 }
    }

    /// `steps` proposals whose visits are NOT recorded, to forget the starting state.
    pub fn burn(&mut self, steps: u64) {
        for _ in 0..steps {
            self.walk.step(&self.bias);
        }
    }

    /// `steps` proposals, each recording the bin the walk is in afterwards.
    ///
    /// A REJECTED proposal is still a visit. The histogram estimates the time the chain spends in
    /// each bin, and a rejection leaves it in a bin for another step; counting only accepted moves
    /// would weight bins by how easy they are to leave, which is the same mistake
    /// [`crate::wanglandau`] documents on its own histogram.
    pub fn steps(&mut self, steps: u64) {
        for _ in 0..steps {
            if self.walk.step(&self.bias) {
                self.accepted += 1;
            }
            self.proposed += 1;
            self.hist[self.walk.coord.bin(self.walk.m)] += 1;
        }
    }

    /// Visits per bin.
    #[must_use]
    pub fn histogram(&self) -> &[u64] {
        &self.hist
    }

    /// The window being sampled.
    #[must_use]
    pub fn window(&self) -> Window {
        self.window
    }

    /// Fraction of proposals accepted, or zero before any were made.
    #[must_use]
    pub fn acceptance(&self) -> f64 {
        if self.proposed == 0 { 0.0 } else { self.accepted as f64 / self.proposed as f64 }
    }

    /// The current spin state.
    #[must_use]
    pub fn state(&self) -> &[i8] {
        &self.walk.s
    }

    /// The current coordinate value.
    #[must_use]
    pub fn coordinate(&self) -> f64 {
        self.walk.m as f64
    }
}

/// Run every window and collect its histogram, one independent chain per window.
///
/// Each window gets its own RNG stream derived from `seed` and its index, so the windows are
/// independent and the whole set is reproducible from one seed.
#[must_use]
pub fn sample(
    g: &Graph,
    beta: f64,
    windows: &[Window],
    burn: u64,
    steps: u64,
    seed: u64,
) -> Vec<Vec<u64>> {
    windows
        .iter()
        .enumerate()
        .map(|(w, &win)| {
            let stream = seed ^ (w as u64).wrapping_mul(0x9E3779B97F4A7C15);
            let mut u = Umbrella::new(g, beta, win, stream);
            u.burn(burn);
            u.steps(steps);
            u.hist
        })
        .collect()
}

// ---- WHAM --------------------------------------------------------------------------------------

/// A stitched profile and the window constants that produced it.
#[derive(Clone, Debug)]
pub struct Wham {
    /// The unbiased profile.
    pub profile: Profile,
    /// Each window's free energy `f_w`, in energy units — the offset umbrella sampling has to solve
    /// for, and the reason matching two windows by hand on their overlap is not enough.
    pub f: Vec<f64>,
    /// Iterations taken before the tolerance was met or the cap was reached.
    pub iters: usize,
    /// The final residual: the largest change in any `beta f_w` over the last iteration.
    pub residual: f64,
    /// Whether `residual` reached the requested tolerance.
    pub converged: bool,
}

/// Solve the WHAM equations for arbitrary biases.
///
/// `coord[b]` is the coordinate value of bin `b`, `bias[w][b]` the bias energy window `w` applies
/// there, and `hist[w][b]` the visits window `w` recorded. A window of all-zero bias is the
/// unbiased ensemble and is a legitimate input — with one such window the answer is exactly the
/// normalised histogram, which is what `one_unbiased_window_returns_the_raw_histogram` pins down.
///
/// # Errors
///
/// [`Invalid`] for no windows, a mismatched or ragged input, a window with no visits, a non-finite
/// bias, or a `beta` a free energy cannot be divided by.
pub fn solve(
    beta: f64,
    coord: &[f64],
    bias: &[Vec<f64>],
    hist: &[Vec<u64>],
    tol: f64,
    max_iters: usize,
) -> Result<Wham, Invalid> {
    if !(beta.is_finite() && beta > 0.0) {
        return Err(Invalid::BadBeta(beta));
    }
    let nw = bias.len();
    if nw == 0 {
        return Err(Invalid::NoWindows);
    }
    if hist.len() != nw {
        return Err(Invalid::Mismatched { windows: nw, histograms: hist.len() });
    }
    let bins = coord.len();
    for w in 0..nw {
        if bias[w].len() != bins {
            return Err(Invalid::Ragged { window: w, len: bias[w].len(), bins });
        }
        if hist[w].len() != bins {
            return Err(Invalid::Ragged { window: w, len: hist[w].len(), bins });
        }
        for (b, v) in bias[w].iter().enumerate() {
            if !v.is_finite() {
                return Err(Invalid::NotFinite { window: w, bin: b });
            }
        }
    }
    let totals: Vec<u64> = hist.iter().map(|h| h.iter().sum()).collect();
    for (w, &t) in totals.iter().enumerate() {
        if t == 0 {
            return Err(Invalid::EmptyWindow(w));
        }
    }

    let ln_n: Vec<f64> = totals.iter().map(|&t| (t as f64).ln()).collect();
    // The numerator is the pooled histogram and never changes. An empty bin is -inf, which
    // propagates to ln P = -inf and F = +inf without ever becoming a NaN.
    let ln_num: Vec<f64> = (0..bins)
        .map(|b| {
            let c: u64 = hist.iter().map(|h| h[b]).sum();
            if c == 0 { f64::NEG_INFINITY } else { (c as f64).ln() }
        })
        .collect();

    let mut f = vec![0.0f64; nw];
    let mut log_p = vec![0.0f64; bins];
    let mut term = vec![0.0f64; nw.max(bins)];
    let mut iters = 0;
    let mut residual = f64::INFINITY;
    while iters < max_iters {
        p_from_f(beta, &ln_n, &ln_num, bias, &f, &mut term, &mut log_p);
        residual = 0.0;
        for w in 0..nw {
            for b in 0..bins {
                term[b] = log_p[b] - beta * bias[w][b];
            }
            let next = -log_sum_exp(&term[..bins]) / beta;
            residual = residual.max((beta * (next - f[w])).abs());
            f[w] = next;
        }
        iters += 1;
        if residual <= tol {
            break;
        }
    }
    // Report the profile the returned `f` implies, not the one the previous iterate did.
    p_from_f(beta, &ln_n, &ln_num, bias, &f, &mut term, &mut log_p);
    Ok(Wham {
        profile: Profile::new(coord.to_vec(), &log_p, beta),
        f,
        iters,
        residual,
        converged: residual <= tol,
    })
}

/// One half of the fixed point: `ln P(m)` given the window constants.
///
/// `P(m) = sum_w H_w(m) / sum_w N_w exp(−beta (V_w(m) − f_w))`, normalised to sum to one.
///
/// # The sign on `f_w` is the load-bearing line in this module
///
/// Flipping it — `ln_n[w] - beta * f[w]` — is the mistake a careful person writes, because the
/// equation is printed with a minus in front of the bracket. Measured: the iteration then has no
/// fixed point it can reach, stalling at a residual of 1.2e-14 after the full 200,000 iterations,
/// and the umbrella profile comes out 4.78 wrong at its worst bin against an enumeration — on a
/// barrier of 7.38, so the shape is gone rather than shifted.
/// `wham_recovers_the_enumerated_profile_from_exact_histograms` and
/// `umbrella_and_wham_match_the_enumerated_profile_and_beat_an_unbiased_chain` both go red.
/// `one_unbiased_window_returns_the_raw_histogram` does NOT, and that is worth knowing: with a
/// single window the normalisation absorbs `f_0` entirely, so the degenerate case is necessary and
/// nowhere near sufficient.
fn p_from_f(
    beta: f64,
    ln_n: &[f64],
    ln_num: &[f64],
    bias: &[Vec<f64>],
    f: &[f64],
    term: &mut [f64],
    log_p: &mut [f64],
) {
    let nw = bias.len();
    for b in 0..log_p.len() {
        for w in 0..nw {
            // exp(-beta (V_w - f_w)) = exp(-beta V_w) exp(+beta f_w): the SIGN on f_w is the whole
            // stitching. Backwards, and every window's offset is applied the wrong way.
            term[w] = ln_n[w] + beta * f[w] - beta * bias[w][b];
        }
        log_p[b] = ln_num[b] - log_sum_exp(&term[..nw]);
    }
    let norm = log_sum_exp(log_p);
    for x in log_p.iter_mut() {
        *x -= norm;
    }
}

/// `ln sum_i exp(v_i)`, shifted by the maximum so a long sum of small weights does not underflow.
fn log_sum_exp(v: &[f64]) -> f64 {
    let mx = v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !mx.is_finite() {
        return mx;
    }
    let acc: f64 = v.iter().map(|x| (x - mx).exp()).sum();
    mx + acc.ln()
}

/// WHAM on harmonic windows over the magnetisation — the ordinary case, with the bias matrix built
/// from the windows rather than by the caller.
///
/// # Errors
///
/// Everything [`solve`] refuses.
pub fn stitch(
    beta: f64,
    coord: &Magnetisation,
    windows: &[Window],
    hist: &[Vec<u64>],
    tol: f64,
    max_iters: usize,
) -> Result<Wham, Invalid> {
    let bias: Vec<Vec<f64>> = windows.iter().map(|w| w.on_grid(coord)).collect();
    solve(beta, &coord.values(), &bias, hist, tol, max_iters)
}

/// Sample every window and stitch the result: the whole umbrella pipeline in one call.
///
/// # Errors
///
/// Everything [`solve`] refuses — in practice [`Invalid::EmptyWindow`] when `steps` is zero.
pub fn run(
    g: &Graph,
    beta: f64,
    windows: &[Window],
    burn: u64,
    steps: u64,
    seed: u64,
) -> Result<Wham, Invalid> {
    let hist = sample(g, beta, windows, burn, steps, seed);
    stitch(beta, &Magnetisation::new(g.n), windows, &hist, TOL, MAX_ITERS)
}

// ---- metadynamics ------------------------------------------------------------------------------

/// How hills are deposited.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hills {
    /// Height of a hill, in energy units, before any tempering.
    pub height: f64,
    /// Width of a hill, in coordinate units. The magnetisation grid is spaced by two, so a width
    /// below that deposits on single bins and fills the profile as a comb rather than a surface.
    pub sigma: f64,
    /// Proposals between depositions. The walk must move between hills or it buries itself.
    pub stride: u64,
    /// `None` is Laio & Parrinello 2002: constant height, and a bias that oscillates around `−F`
    /// rather than settling on it. `Some(gamma)` is well-tempered (Barducci, Bussi & Parrinello
    /// 2008): each hill is scaled by `exp(−beta V/(gamma − 1))`, so the bias converges to
    /// `−(1 − 1/gamma) F`, which [`Meta::profile`] undoes.
    pub bias_factor: Option<f64>,
}

impl Hills {
    /// Deposition parameters.
    ///
    /// # Panics
    ///
    /// If the height or width is not finite and positive, the stride is zero, or a bias factor is
    /// given that is not finite and strictly above one. `gamma = 1` deposits nothing at all — the
    /// scaling divides by `gamma − 1` — and below one it deposits HARDER where the bias is already
    /// deep, which runs away.
    #[must_use]
    pub fn new(height: f64, sigma: f64, stride: u64, bias_factor: Option<f64>) -> Self {
        assert!(
            height.is_finite() && height > 0.0,
            "a hill height must be finite and positive, got {height}"
        );
        assert!(
            sigma.is_finite() && sigma > 0.0,
            "a hill width must be finite and positive, got {sigma}"
        );
        assert!(stride > 0, "a stride of zero deposits without ever proposing a move");
        if let Some(gamma) = bias_factor {
            assert!(
                gamma.is_finite() && gamma > 1.0,
                "a bias factor must be finite and above one, got {gamma}"
            );
        }
        Hills { height, sigma, stride, bias_factor }
    }

    /// What the converged bias must be multiplied by to become `−F`: `gamma/(gamma − 1)`, or one
    /// for the untempered algorithm.
    #[must_use]
    pub fn rescale(&self) -> f64 {
        self.bias_factor.map_or(1.0, |gamma| gamma / (gamma - 1.0))
    }
}

/// A metadynamics run: a biased walk that writes its own bias as it goes.
pub struct Meta<'g> {
    walk: Walk<'g>,
    hills: Hills,
    bias: Vec<f64>,
    steps: u64,
    deposited: u64,
}

impl<'g> Meta<'g> {
    /// A run at `beta` with these deposition parameters, started from a state of zero magnetisation.
    ///
    /// # Panics
    ///
    /// If `beta` is not finite and positive.
    #[must_use]
    pub fn new(g: &'g Graph, beta: f64, hills: Hills, seed: u64) -> Self {
        let walk = Walk::new(g, beta, 0.0, seed);
        let bins = walk.coord.bins();
        Meta { walk, hills, bias: vec![0.0; bins], steps: 0, deposited: 0 }
    }

    /// `steps` proposals, depositing a hill every [`Hills::stride`] of them.
    pub fn run(&mut self, steps: u64) {
        for _ in 0..steps {
            let Meta { walk, bias, .. } = self;
            walk.step(bias);
            self.steps += 1;
            if self.steps.is_multiple_of(self.hills.stride) {
                self.deposit();
            }
        }
    }

    /// One Gaussian hill at the walk's current position, evaluated on every grid point.
    ///
    /// Exact rather than interpolated: the coordinate is discrete, so a hill's centre is always a
    /// grid point and the walk is never between two of them.
    fn deposit(&mut self) {
        let here = self.walk.coord.bin(self.walk.m);
        let h = match self.hills.bias_factor {
            Some(gamma) => {
                self.hills.height * (-self.walk.beta * self.bias[here] / (gamma - 1.0)).exp()
            }
            None => self.hills.height,
        };
        let c = self.walk.coord.value(here);
        let two_sigma_sq = 2.0 * self.hills.sigma * self.hills.sigma;
        for b in 0..self.bias.len() {
            let d = self.walk.coord.value(b) - c;
            self.bias[b] += h * (-d * d / two_sigma_sq).exp();
        }
        self.deposited += 1;
    }

    /// The accumulated bias `V(m)` on the grid.
    #[must_use]
    pub fn bias(&self) -> &[f64] {
        &self.bias
    }

    /// Hills deposited so far.
    #[must_use]
    pub fn deposited(&self) -> u64 {
        self.deposited
    }

    /// Proposals made so far.
    #[must_use]
    pub fn steps(&self) -> u64 {
        self.steps
    }

    /// The current coordinate value.
    #[must_use]
    pub fn coordinate(&self) -> f64 {
        self.walk.m as f64
    }

    /// The free-energy profile the bias implies: `F(m) = −rescale * V(m)`.
    ///
    /// Determined only up to an additive constant — [`Profile::new`] fixes one by normalising, and
    /// [`Profile::centred`] is the comparison that does not depend on which constant was chosen.
    #[must_use]
    pub fn profile(&self) -> Profile {
        let beta = self.walk.beta;
        let scale = self.hills.rescale();
        // ln P = -beta F = beta * scale * V, up to the constant Profile::new normalises away.
        let log_w: Vec<f64> = self.bias.iter().map(|v| beta * scale * v).collect();
        Profile::new(self.walk.coord.values(), &log_w, beta)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exact::Elimination;
    use crate::ising;

    /// Worst disagreement between two profiles, bin by bin. The SHAPE, not the minimum: a profile
    /// that gets the well right and the barrier wrong is the failure this module exists to avoid,
    /// and a test that looked only at `argmin F` would pass it.
    fn worst(a: &[f64], b: &[f64]) -> f64 {
        assert_eq!(a.len(), b.len(), "profiles must be over the same bins");
        a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0f64, f64::max)
    }

    /// Histograms whose empirical distribution IS the biased distribution of each window, to the
    /// resolution of an integer count.
    ///
    /// Window `w` sees `P(m) exp(−beta V_w(m))` normalised, and that is computed here from the
    /// ENUMERATED profile — so nothing in this construction has been near the sampler or the
    /// solver. `scale` is the count the largest bin of each window is given; the rest follow in
    /// proportion, so the residual discretisation is one part in `scale`.
    fn exact_histograms(
        exact: &Profile,
        windows: &[Window],
        coord: &Magnetisation,
        beta: f64,
        scale: f64,
    ) -> Vec<Vec<u64>> {
        windows
            .iter()
            .map(|w| {
                let bias = w.on_grid(coord);
                let lw: Vec<f64> =
                    (0..coord.bins()).map(|b| exact.log_p[b] - beta * bias[b]).collect();
                let top = lw.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                lw.iter().map(|x| ((x - top).exp() * scale).round() as u64).collect()
            })
            .collect()
    }

    /// THE ORACLE: WHAM's fixed point is exact, so fed exact histograms it must return the exact
    /// profile — checked bin by bin against a sum over all `2^n` states, on three models.
    ///
    /// No statistics anywhere in the claim. Two things keep it off zero, and both were measured
    /// rather than assumed:
    ///
    /// * a histogram holds integers, so each window's counts are rounded at one part in `2^50`;
    /// * the solver stops on a residual, and the fixed point converges LINEARLY — rate about 0.89
    ///   on these windows — so whatever residual it stops at is amplified roughly ninefold in `f`.
    ///
    /// The second dominates, which is why this runs the solver to `1e-15` rather than to the
    /// default [`TOL`] of `1e-13`: at the default the worst bin came out at 2.3e-13, 8.0e-13 and
    /// 2.1e-12 on the three models, and the test would have been measuring the stopping rule. At
    /// `1e-15` they are **2.9e-14, 2.1e-14 and 2.4e-13**, the last on the model whose profile spans
    /// 33 and whose absolute error is therefore the largest while its relative error is the
    /// smallest. The assertion is 1e-12 for all three.
    ///
    /// The window constants `f_w` are checked too, against their own closed form
    /// `f_w = −(1/beta) ln sum_m P(m) exp(−beta V_w(m))` evaluated on the enumerated `P`. A solver
    /// can get `P` right while the `f` it reports is a different quantity, and the profile alone
    /// would not notice.
    #[test]
    fn wham_recovers_the_enumerated_profile_from_exact_histograms() {
        let cases: [(Graph, f64, f64); 3] = [
            (ising::grid2d(3, 4, 1.0), 0.7, 0.25),
            (ising::ring(10, 1.0, 0.25), 0.9, 0.25),
            // A frustrated 3x3 antiferromagnet, whose profile spans 32.8 — and whose windows
            // therefore need a stiffness of a different order. At k = 0.25 the restraint never
            // dominates the profile's own variation, the "window" is not a window, and the residual
            // rises from 1e-13 to 1e-10 because the count resolution now falls on bins whose
            // biased probability is 1e-7 rather than 1e-1.
            (ising::lattice2d(3, -1.0), 0.45, 4.0),
        ];
        let scale = (1u64 << 50) as f64;
        for (g, beta, k) in &cases {
            let coord = Magnetisation::new(g.n);
            let exact = exact_profile(g, *beta).unwrap();
            let windows = ladder(&coord, coord.bins(), *k);
            let hist = exact_histograms(&exact.profile, &windows, &coord, *beta, scale);
            let got = stitch(*beta, &coord, &windows, &hist, 1e-15, MAX_ITERS).unwrap();
            assert!(got.converged, "residual {} after {} iters", got.residual, got.iters);

            let e = worst(&got.profile.free_energy, &exact.profile.free_energy);
            println!(
                "n={} beta={beta} barrier {:.3} worst bin {e:.3e} in {} iters",
                g.n,
                exact.profile.barrier(),
                got.iters
            );
            assert!(e < 1e-12, "n={} beta={beta}: worst bin off by {e:e}", g.n);

            for (w, win) in windows.iter().enumerate() {
                let bias = win.on_grid(&coord);
                let lw: Vec<f64> =
                    (0..coord.bins()).map(|b| exact.profile.log_p[b] - beta * bias[b]).collect();
                let want = -log_sum_exp(&lw) / beta;
                assert!(
                    (got.f[w] - want).abs() < 1e-12,
                    "n={} window {w}: f {} vs closed form {want}",
                    g.n,
                    got.f[w]
                );
            }
        }
    }

    /// THE DEGENERATE CASE: one window with no bias must hand back the histogram it was given.
    ///
    /// This is what catches a reweighting applied twice, or not at all, or with the sign of `f_w`
    /// reversed — every one of which leaves the general case looking plausible and this one exactly
    /// wrong. `P(m)` here is `H(m)/N` with nothing left to solve for, so it is asserted to 1e-15
    /// rather than to a tolerance: the only arithmetic between the input and the output is one
    /// normalisation.
    #[test]
    fn one_unbiased_window_returns_the_raw_histogram() {
        let g = ising::grid2d(3, 4, 1.0);
        let beta = 0.7;
        let coord = Magnetisation::new(g.n);
        let flat = [Window::new(0.0, 0.0)];
        let hist = sample(&g, beta, &flat, 20_000, 200_000, 4);
        let total: u64 = hist[0].iter().sum();
        assert_eq!(total, 200_000, "every step is one visit, accepted or not");

        let got = stitch(beta, &coord, &flat, &hist, TOL, MAX_ITERS).unwrap();
        assert_eq!(got.iters, 1, "an unbiased window is already the answer");
        assert!(got.f[0].abs() < 1e-15, "the one window's constant is zero, got {}", got.f[0]);
        for b in 0..coord.bins() {
            let want = (hist[0][b] as f64 / total as f64).ln();
            assert!(
                (got.profile.log_p[b] - want).abs() < 1e-15,
                "bin {b}: ln P {} vs ln(H/N) {want}",
                got.profile.log_p[b]
            );
            assert!(
                (got.profile.free_energy[b] + want / beta).abs() < 1e-14,
                "bin {b}: F is not -(1/beta) ln(H/N)"
            );
        }
    }

    /// The BIASED WALK itself, against the biased distribution computed by enumeration.
    ///
    /// WHAM can be perfect and the sampler wrong; then every profile test fails and none of them
    /// says which half. This one holds the solver out entirely: run one window, normalise its
    /// histogram, and compare with `P(m) exp(−beta V(m))` from a sum over all `2^n` states.
    ///
    /// Total variation against that distribution, worst of four seeds, each case with its own
    /// tolerance — one tolerance for all of them would be the loosest case's, and would assert
    /// nothing about the other four.
    ///
    /// ```text
    ///   k     m0   steps    worst TV   tolerance
    ///   0      0     400k    0.0462      0.11
    ///   0      0       4M    0.0156      0.04
    ///   0.25  -6     400k    0.0047      0.012
    ///   0.25   0     400k    0.0064      0.016
    ///   1      4     400k    0.0025      0.007
    /// ```
    ///
    /// `k = 0` is the unbiased Metropolis chain and is in the list on purpose — twice. It is the
    /// case where the bias must do NOTHING, and a `dV` with the wrong sign still produces something
    /// shaped like a distribution. It is also ten times further from the truth than any restrained
    /// window at the same budget, which is the module's reason to exist stated in the sampler's own
    /// terms: the unbiased chain has to cross the barrier to get its weight right, and the windows
    /// never do. Its error falls as `1/sqrt(t)` — 0.0462 to 0.0156 for ten times the steps, against
    /// 3.16 predicted — so what it has is variance, not bias, and only patience fixes it.
    #[test]
    fn the_biased_walk_samples_the_enumerated_biased_distribution() {
        let g = ising::grid2d(3, 4, 1.0);
        let beta = 0.7;
        let coord = Magnetisation::new(g.n);
        let exact = exact_profile(&g, beta).unwrap();
        let cases: [(f64, f64, u64, f64); 5] = [
            (0.0, 0.0, 400_000, 0.11),
            (0.0, 0.0, 4_000_000, 0.04),
            (0.25, -6.0, 400_000, 0.012),
            (0.25, 0.0, 400_000, 0.016),
            (1.0, 4.0, 400_000, 0.007),
        ];
        let mut unbiased_tv = 0.0f64;
        let mut biased_tv = 0.0f64;
        for (k, m0, steps, tol) in cases {
            let win = Window::new(k, m0);
            let bias = win.on_grid(&coord);
            let lw: Vec<f64> =
                (0..coord.bins()).map(|b| exact.profile.log_p[b] - beta * bias[b]).collect();
            let norm = log_sum_exp(&lw);
            let want: Vec<f64> = lw.iter().map(|x| (x - norm).exp()).collect();

            let mut tv_worst = 0.0f64;
            for seed in 0..4u64 {
                let mut u = Umbrella::new(&g, beta, win, seed);
                u.burn(20_000);
                u.steps(steps);
                let n: u64 = u.histogram().iter().sum();
                let got: Vec<f64> = u.histogram().iter().map(|&c| c as f64 / n as f64).collect();
                let tv: f64 =
                    0.5 * got.iter().zip(want.iter()).map(|(a, b)| (a - b).abs()).sum::<f64>();
                tv_worst = tv_worst.max(tv);
            }
            println!("k={k} m0={m0} steps={steps} worst TV over 4 seeds {tv_worst:.4}");
            assert!(tv_worst < tol, "k={k} m0={m0} steps={steps}: total variation {tv_worst}");
            if k == 0.0 && steps == 400_000 {
                unbiased_tv = tv_worst;
            } else if k > 0.0 {
                biased_tv = biased_tv.max(tv_worst);
            }
        }
        assert!(
            biased_tv * 5.0 < unbiased_tv,
            "a restrained window was no closer than the unbiased chain at the same budget: \
             {biased_tv} against {unbiased_tv}"
        );
    }

    /// Umbrella sampling plus WHAM against the enumerated profile, every bin — and against an
    /// unbiased chain given the SAME total number of steps, which is the comparison that says
    /// whether the windows bought anything.
    ///
    /// A 4x5 ferromagnet at `beta = 0.9`: twenty spins, a barrier of 7.38 in energy units, which is
    /// 6.6 `kT`. Measured over four seeds at two million steps each way — worst bin 0.154 for the
    /// twenty-one windows, 0.503 for the unbiased chain. The margin is a factor of three, not an
    /// order of magnitude, and the honest reason is that 6.6 `kT` over twenty spins is a barrier an
    /// unbiased chain can still cross a few thousand times in two million steps. The windows win by
    /// spending their steps where the error is, not by reaching somewhere unreachable.
    #[test]
    fn umbrella_and_wham_match_the_enumerated_profile_and_beat_an_unbiased_chain() {
        let g = ising::grid2d(4, 5, 1.0);
        let beta = 0.9;
        let coord = Magnetisation::new(g.n);
        let exact = exact_profile(&g, beta).unwrap();
        let total = 2_000_000u64;

        let windows = ladder(&coord, coord.bins(), 0.25);
        let per = total / windows.len() as u64;
        let mut umbrella_worst = 0.0f64;
        for seed in 0..4u64 {
            let r = run(&g, beta, &windows, 20_000, per, seed).unwrap();
            assert!(r.converged, "seed {seed}: residual {} after {} iters", r.residual, r.iters);
            umbrella_worst =
                umbrella_worst.max(worst(&r.profile.free_energy, &exact.profile.free_energy));
        }

        let flat = [Window::new(0.0, 0.0)];
        let mut unbiased_worst = 0.0f64;
        for seed in 0..4u64 {
            let r = run(&g, beta, &flat, 20_000, total, seed).unwrap();
            unbiased_worst =
                unbiased_worst.max(worst(&r.profile.free_energy, &exact.profile.free_energy));
        }
        println!(
            "barrier {:.3}: worst bin, umbrella {umbrella_worst:.4}, unbiased {unbiased_worst:.4}",
            exact.profile.barrier()
        );
        assert!(umbrella_worst < 0.25, "worst bin {umbrella_worst} against the enumeration");
        assert!(
            umbrella_worst < 0.6 * unbiased_worst,
            "the windows bought nothing: {umbrella_worst} against {unbiased_worst} unbiased"
        );
    }

    /// THE CONVERGED BIAS IS MINUS THE PROFILE, checked every bin against the enumeration after the
    /// mean is removed from BOTH sides — so the additive constant metadynamics is defined up to is
    /// not what makes this pass.
    ///
    /// Well-tempered, `gamma = 5`, hills of height 0.02 and width 0.5 every 20 proposals, four
    /// million proposals, on a 3x4 ferromagnet at `beta = 0.55` whose centred profile spans 2.48.
    /// Worst bin over eight seeds: **0.077**, which is 3% of the span. The tolerance here is 0.15,
    /// twice the measured worst, and it is a statistical tolerance rather than a mathematical one —
    /// stated that way because a reader is entitled to know which kind they are looking at.
    ///
    /// The untempered 2002 algorithm is run on the same model with the same hills and is held to a
    /// much weaker bound, because it does not converge: it oscillates around `−F` with an amplitude
    /// set by the hill height. Worst bin over the same eight seeds: **0.515**, nearly seven times
    /// worse. That ratio is asserted, since it is the entire reason [`Hills::bias_factor`] exists.
    ///
    /// # The hills are NARROWER than the grid spacing, and that is not a mistake
    ///
    /// Textbook advice for a continuous coordinate is a width around the feature size. This
    /// coordinate is discrete with spacing two, and a broad hill makes the bias a CONVOLUTION of
    /// the visit histogram; recovering `F` then means deconvolving, which rings — the measured
    /// error alternates in sign from bin to bin, which is what deconvolution ringing looks like and
    /// what statistical noise does not. Measured, worst bin against the enumeration, hills of
    /// height 0.05 every 20 proposals, 2M proposals, eight seeds:
    ///
    /// ```text
    ///   sigma            0.5     1.0     1.5
    ///   Laio-Parrinello  0.68    0.94    1.54
    ///   well-tempered    0.15    0.34    0.76
    /// ```
    ///
    /// Every column is worse than the one before it. At `sigma = 0.5` a neighbouring bin receives
    /// `exp(−8)` of a hill, so deposition is effectively per-bin and there is nothing to deconvolve.
    #[test]
    fn metadynamics_bias_is_minus_the_enumerated_profile() {
        let g = ising::grid2d(3, 4, 1.0);
        let beta = 0.55;
        let exact = exact_profile(&g, beta).unwrap();
        let want = exact.profile.centred();
        let span = exact.profile.barrier();

        let mut tempered = 0.0f64;
        let mut plain = 0.0f64;
        for seed in 0..8u64 {
            let mut wt = Meta::new(&g, beta, Hills::new(0.02, 0.5, 20, Some(5.0)), seed);
            wt.run(4_000_000);
            assert_eq!(wt.deposited(), 200_000, "one hill per stride, and the stride is 20");
            tempered = tempered.max(worst(&wt.profile().centred(), &want));

            let mut lp = Meta::new(&g, beta, Hills::new(0.02, 0.5, 20, None), seed);
            lp.run(4_000_000);
            plain = plain.max(worst(&lp.profile().centred(), &want));
        }
        println!(
            "span {span:.3}: worst bin, well-tempered {tempered:.4}, Laio-Parrinello {plain:.4}"
        );
        assert!(tempered < 0.15, "well-tempered bias is off by {tempered} on a span of {span}");
        assert!(
            tempered * 3.0 < plain,
            "tempering bought nothing: {tempered} against {plain} untempered"
        );
    }

    /// The `ln Z` bracket contains what exact variable elimination computes, and is still NARROW.
    ///
    /// Two ways to get a bound wrong, and this checks both. Summing in round-to-nearest produces
    /// ends on the wrong side of the truth, which the containment catches; scaling the guard by
    /// `sum |x|` instead of by the answer produces a bracket wide enough to be useless, which the
    /// width catches — [`crate::round`]'s own history is that the second was the first version of
    /// that module. Measured width here: 7e-15 in `ln Z`, on models whose `ln Z` is around 10.
    #[test]
    fn the_enumerated_bracket_contains_the_elimination_log_z() {
        let el = Elimination::default();
        let cases: [(Graph, f64); 4] = [
            (ising::grid2d(3, 4, 1.0), 0.7),
            (ising::ring(10, 1.0, 0.25), 0.9),
            (ising::lattice2d(3, -1.0), 0.45),
            (ising::ring(4, 2.0, -0.5), 1.6),
        ];
        for (g, beta) in &cases {
            let e = exact_profile(g, *beta).unwrap();
            let want = el.log_partition(g, *beta).unwrap().log_z.unwrap();
            let (lo, hi) = e.log_z;
            assert!(lo <= want && want <= hi, "n={}: {want} outside [{lo}, {hi}]", g.n);
            assert!(hi - lo < 1e-12, "n={}: bracket width {} is not a useful bound", g.n, hi - lo);
            // The profile is normalised, so its own weights must re-sum to one exactly enough that
            // `free_energy` is a free energy and not a shape.
            let z: f64 = e.profile.log_p.iter().map(|x| x.exp()).sum();
            assert!((z - 1.0).abs() < 1e-14, "n={}: P sums to {z}", g.n);
        }
    }

    /// The bin map is a bijection on every state of every small model, checked exhaustively.
    ///
    /// An off-by-one here is silent: the profile still has the right shape, shifted by one bin, and
    /// every comparison against another profile built the same way still passes.
    #[test]
    fn the_coordinate_grid_round_trips_every_state_by_enumeration() {
        for n in 1..=10usize {
            let coord = Magnetisation::new(n);
            assert_eq!(coord.bins(), n + 1);
            let mut seen = vec![0u32; coord.bins()];
            for mask in 0u64..(1u64 << n) {
                let m: i64 = (0..n).map(|i| if mask >> i & 1 == 1 { 1i64 } else { -1 }).sum();
                let b = coord.bin(m);
                assert_eq!(coord.value(b), m as f64, "n={n} m={m} went to bin {b}");
                seen[b] += 1;
            }
            // Every bin is a binomial coefficient, and they must sum to 2^n.
            assert_eq!(seen.iter().sum::<u32>() as u64, 1u64 << n);
            assert!(seen.iter().all(|&c| c > 0), "n={n}: a bin no state reaches");
            assert_eq!(seen[0], 1, "n={n}: exactly one state is all-down");
            assert_eq!(seen[n], 1, "n={n}: exactly one state is all-up");
        }
    }

    /// A ladder spans the coordinate end to end, and one window sits in the middle.
    #[test]
    fn a_ladder_covers_the_whole_coordinate() {
        let coord = Magnetisation::new(12);
        let l = ladder(&coord, 13, 0.25);
        assert_eq!(l.len(), 13);
        assert_eq!(l[0].m0, -12.0);
        assert_eq!(l[12].m0, 12.0);
        for (b, w) in l.iter().enumerate() {
            assert_eq!(w.m0, coord.value(b), "window {b} is not on bin {b}");
            assert_eq!(w.bias(w.m0), 0.0, "a restraint is zero at its own centre");
        }
        assert_eq!(ladder(&coord, 1, 0.25)[0].m0, 0.0, "a single window is central");
        // An unbiased window is flat, and the harmonic form is symmetric about its centre.
        let w = Window::new(0.5, 3.0);
        assert_eq!(w.bias(5.0), w.bias(1.0));
        assert_eq!(w.bias(5.0), 1.0, "(k/2) d^2 with k = 1/2 and d = 2");
        assert!(Window::new(0.0, 7.0).on_grid(&coord).iter().all(|&x| x == 0.0));
    }

    /// The rescaling that turns a well-tempered bias into a free energy, at its two limits.
    #[test]
    fn the_well_tempered_rescaling_has_the_right_limits() {
        assert_eq!(Hills::new(1.0, 1.0, 1, None).rescale(), 1.0, "untempered needs no rescaling");
        assert_eq!(Hills::new(1.0, 1.0, 1, Some(2.0)).rescale(), 2.0);
        assert_eq!(Hills::new(1.0, 1.0, 1, Some(5.0)).rescale(), 1.25);
        // gamma -> infinity IS the untempered algorithm, and the rescaling must agree.
        assert!((Hills::new(1.0, 1.0, 1, Some(1e12)).rescale() - 1.0).abs() < 1e-11);
    }

    #[test]
    fn refusals() {
        let coord = Magnetisation::new(4);
        let vals = coord.values();
        let bias = vec![vec![0.0; 5]];
        let hist = vec![vec![1u64; 5]];
        let err = |r: Result<Wham, Invalid>| r.err().unwrap();

        assert_eq!(err(solve(0.0, &vals, &bias, &hist, TOL, 10)), Invalid::BadBeta(0.0));
        assert_eq!(
            err(solve(f64::NAN, &vals, &bias, &hist, TOL, 10)).to_string(),
            "a free energy is divided by beta, which must be finite and positive, not NaN"
        );
        assert_eq!(err(solve(1.0, &vals, &[], &[], TOL, 10)), Invalid::NoWindows);
        assert_eq!(
            err(solve(1.0, &vals, &bias, &[hist[0].clone(), hist[0].clone()], TOL, 10)),
            Invalid::Mismatched { windows: 1, histograms: 2 }
        );
        assert_eq!(
            err(solve(1.0, &vals, &[vec![0.0; 3]], &hist, TOL, 10)),
            Invalid::Ragged { window: 0, len: 3, bins: 5 }
        );
        assert_eq!(
            err(solve(1.0, &vals, &bias, &[vec![1u64; 4]], TOL, 10)),
            Invalid::Ragged { window: 0, len: 4, bins: 5 }
        );
        assert_eq!(
            err(solve(1.0, &vals, &bias, &[vec![0u64; 5]], TOL, 10)),
            Invalid::EmptyWindow(0)
        );
        assert_eq!(
            err(solve(1.0, &vals, &[vec![0.0, 0.0, f64::INFINITY, 0.0, 0.0]], &hist, TOL, 10)),
            Invalid::NotFinite { window: 0, bin: 2 }
        );

        let big = ising::ring(21, 1.0, 0.0);
        assert_eq!(
            exact_profile(&big, 1.0).err().unwrap(),
            Invalid::TooLarge { n: 21, cap: MAX_ENUM }
        );
        let small = ising::ring(4, 1.0, 0.0);
        assert_eq!(exact_profile(&small, -1.0).err().unwrap(), Invalid::BadBeta(-1.0));
        // Every refusal must print something a caller can act on.
        for e in [
            Invalid::NoWindows,
            Invalid::Mismatched { windows: 1, histograms: 2 },
            Invalid::Ragged { window: 0, len: 3, bins: 5 },
            Invalid::EmptyWindow(3),
            Invalid::NotFinite { window: 1, bin: 2 },
            Invalid::TooLarge { n: 21, cap: 20 },
        ] {
            assert!(e.to_string().len() > 20, "{e:?} prints nothing useful");
        }
    }

    /// A profile whose bins were never reached says so, rather than reporting a number for them.
    #[test]
    fn an_unreached_bin_is_infinite_rather_than_a_guess() {
        let coord = Magnetisation::new(4);
        let bias = vec![vec![0.0; 5]];
        let hist = vec![vec![3u64, 0, 5, 0, 2]];
        let got = solve(1.0, &coord.values(), &bias, &hist, TOL, 100).unwrap();
        assert!(got.profile.free_energy[1].is_infinite());
        assert!(got.profile.free_energy[3].is_infinite());
        assert!(got.profile.log_p[1] == f64::NEG_INFINITY);
        // The barrier and the centring must both ignore them rather than become NaN.
        assert!(got.profile.barrier().is_finite());
        assert!(got.profile.centred().iter().filter(|x| x.is_finite()).count() == 3);
    }

    /// Seeded runs repeat exactly, which is the crate's headline and not free: the walk draws a
    /// permutation at construction and then one or two uniforms per proposal.
    #[test]
    fn a_run_is_reproducible_from_its_seed() {
        let g = ising::grid2d(3, 4, 1.0);
        let w = ladder(&Magnetisation::new(g.n), 5, 0.25);
        let a = sample(&g, 0.7, &w, 1_000, 20_000, 77);
        let b = sample(&g, 0.7, &w, 1_000, 20_000, 77);
        assert_eq!(a, b, "same seed, same histograms");
        let c = sample(&g, 0.7, &w, 1_000, 20_000, 78);
        assert_ne!(a, c, "and a different seed is a different run");
        // Windows must not share a stream: two windows of the SAME shape still differ.
        let same = [Window::new(0.25, 0.0), Window::new(0.25, 0.0)];
        let h = sample(&g, 0.7, &same, 1_000, 20_000, 5);
        assert_ne!(h[0], h[1], "two windows drawing the same numbers are one window counted twice");
    }
}

