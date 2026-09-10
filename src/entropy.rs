//! Entropy and mutual information estimated from samples, with the bias named and corrected.
//!
//! Counting outcomes and feeding the frequencies to `-sum p ln p` does not estimate the entropy;
//! it estimates something smaller. With `K` occupied bins and `N` draws the plug-in estimator sits
//! roughly `(K - 1) / (2N)` nats **below** the truth, on nearly every draw, and the deficit does
//! not average away — it is a bias, not noise. Measured exactly by enumerating every count vector:
//! a uniform four-bin source at `N = 8` returns `1.165` against `ln 4 = 1.386`.
//!
//! Three classical corrections, all aimed at that same term:
//!
//! * [`Estimator::MillerMadow`] adds `(K_obs - 1) / (2N)` back, with the *observed* support
//!   standing in for the true one — itself an undercount whenever a bin goes unseen.
//! * [`Estimator::Grassberger`] replaces `ln n_i` with `psi(n_i) + (-1)^n_i / (n_i + 1)`
//!   (Grassberger 1988). Since `psi(n) = ln n - 1/(2n) + ...`, to leading order this is the same
//!   correction with `K_obs` in place of `K_obs - 1`, plus an oscillating term.
//! * [`Estimator::ChaoShen`] corrects the *probabilities* rather than the estimate:
//!   Horvitz-Thompson weighting under a Good-Turing coverage estimate, so the mass never seen is
//!   charged for.
//!
//! None is uniformly best and none is unbiased — no unbiased entropy estimator exists at fixed
//! `N`. What is true, and what the tests here measure against exact enumeration rather than
//! assert, is that all three shrink the bias at small `N`.
//!
//! # Mutual information, and the correction that works for entropy but not here
//!
//! `I = H(X) + H(Y) - H(X, Y)` with each term estimated separately, so the three biases combine:
//!
//! ```text
//!   -(Kx - 1)/2N  -  (Ky - 1)/2N  +  (Kx Ky - 1)/2N  =  +(Kx - 1)(Ky - 1) / 2N
//! ```
//!
//! Plug-in mutual information is biased **upward**, so two independent variables read as
//! dependent. Miller-Madow's `-1` per alphabet is exactly what cancels that, and on a fully
//! occupied two-by-two table the cancellation is an algebraic identity: `I_mm = I_plugin - 1/(2N)`.
//! Grassberger's does not: its per-bin `+1/(2N)` combines as `(Kx + Ky - Kx Ky) / 2N`, which is
//! **zero** for a two-by-two table — the estimator that corrects entropy best leaves a binary
//! mutual information essentially uncorrected. Exact enumeration at `N = 32`, true `I = 0`:
//! plug-in `+0.01650`, Grassberger `+0.01572`, Miller-Madow `+0.00088`.
//!
//! # What `N` means here
//!
//! Every correction divides by the number of draws, so each is right for *independent* draws.
//! [`crate::samples::Provenance::Chain`] draws are not: `N` of them are worth `N / (2 tau_int)`
//! (see [`crate::samples::SampleSet::chain_tau`]), so these corrections under-correct on a chain.
//! [`draws`] refuses the provenances whose states are not draws at all.

use crate::samples::{ENUMERATION_LIMIT, Provenance, SampleSet};

/// Which bias correction an entropy estimate carries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Estimator {
    /// Maximum likelihood: `-sum (n_i/N) ln(n_i/N)`. Biased low by about `(K - 1) / (2N)`.
    PlugIn,
    /// Plug-in plus `(K_obs - 1) / (2N)`, the leading bias at the observed support (Miller 1955).
    MillerMadow,
    /// `ln N - (1/N) sum n_i [psi(n_i) + (-1)^n_i / (n_i + 1)]` (Grassberger 1988).
    Grassberger,
    /// Horvitz-Thompson under a Good-Turing coverage estimate (Chao and Shen 2003).
    ChaoShen,
}

impl Estimator {
    /// Every estimator here, in the order the module documents them.
    pub const ALL: [Estimator; 4] =
        [Estimator::PlugIn, Estimator::MillerMadow, Estimator::Grassberger, Estimator::ChaoShen];

    /// Entropy in nats of the distribution `counts` were drawn from. Empty bins are ignored.
    #[must_use]
    pub fn entropy(self, counts: &[usize]) -> f64 {
        match self {
            Estimator::PlugIn => plug_in(counts),
            Estimator::MillerMadow => miller_madow(counts),
            Estimator::Grassberger => grassberger(counts),
            Estimator::ChaoShen => chao_shen(counts),
        }
    }

    /// A short name, for tables and messages.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Estimator::PlugIn => "plug-in",
            Estimator::MillerMadow => "miller-madow",
            Estimator::Grassberger => "grassberger",
            Estimator::ChaoShen => "chao-shen",
        }
    }
}

/// Total draws in a count vector.
#[must_use]
pub fn total(counts: &[usize]) -> usize {
    counts.iter().sum()
}

/// Occupied bins: the observed support, at most the true one and usually less.
#[must_use]
pub fn support(counts: &[usize]) -> usize {
    counts.iter().filter(|&&c| c > 0).count()
}

/// The leading plug-in bias in nats, `-(k - 1) / (2n)`: what every correction here aims at.
///
/// # Panics
/// If `n` is zero.
#[must_use]
pub fn plug_in_bias(k: usize, n: usize) -> f64 {
    assert!(n > 0, "a bias per draw is undefined with no draws");
    -((k as f64) - 1.0) / (2.0 * n as f64)
}

/// Maximum-likelihood entropy in nats. Zero for an empty sample.
#[must_use]
pub fn plug_in(counts: &[usize]) -> f64 {
    let n = total(counts);
    if n == 0 {
        return 0.0;
    }
    let nf = n as f64;
    let mut h = 0.0;
    for &c in counts {
        if c > 0 {
            let p = c as f64 / nf;
            h -= p * p.ln();
        }
    }
    h
}

/// Miller-Madow entropy in nats: plug-in plus `(support - 1) / (2N)`.
#[must_use]
pub fn miller_madow(counts: &[usize]) -> f64 {
    let n = total(counts);
    if n == 0 {
        return 0.0;
    }
    plug_in(counts) - plug_in_bias(support(counts), n)
}

/// Grassberger 1988 entropy in nats: `ln N - (1/N) sum n_i [psi(n_i) + (-1)^n_i / (n_i + 1)]`.
///
/// Not zero on a degenerate sample — one bin holding all `N` draws returns `O(1/N)` rather than
/// exactly `0`, which is the price of a correction that never looks at the support count.
#[must_use]
pub fn grassberger(counts: &[usize]) -> f64 {
    let n = total(counts);
    if n == 0 {
        return 0.0;
    }
    let nf = n as f64;
    let mut acc = 0.0;
    for &c in counts {
        if c == 0 {
            continue;
        }
        let cf = c as f64;
        let sign = if c % 2 == 0 { 1.0 } else { -1.0 };
        acc += cf * (digamma(cf) + sign / (cf + 1.0));
    }
    nf.ln() - acc / nf
}

/// Chao-Shen entropy in nats: Horvitz-Thompson under a Good-Turing coverage estimate.
///
/// Coverage is `C = 1 - f1/N` for `f1` singletons; when every draw is a singleton `f1` is lowered
/// by one, since `C = 0` would send every adjusted probability to zero and the estimate to `NaN`.
#[must_use]
pub fn chao_shen(counts: &[usize]) -> f64 {
    let n = total(counts);
    if n == 0 {
        return 0.0;
    }
    let nf = n as f64;
    let mut f1 = counts.iter().filter(|&&c| c == 1).count();
    if f1 == n {
        f1 = n - 1;
    }
    let coverage = 1.0 - f1 as f64 / nf;
    let mut h = 0.0;
    for &c in counts {
        if c == 0 {
            continue;
        }
        let p = coverage * (c as f64) / nf;
        // `1 - (1 - p)^N`, the chance this bin was seen at all, written through `ln_1p`/`exp_m1`:
        // the direct form loses every significant digit when `p` is small, and small `p` is the
        // regime the correction exists for. At `p == 1` the direct form is exact and the log is not.
        let seen = if p >= 1.0 { 1.0 } else { -(nf * (-p).ln_1p()).exp_m1() };
        h -= p * p.ln() / seen;
    }
    h
}

/// Shannon entropy in nats of a probability vector: the exact oracle, not an estimate.
///
/// # Panics
/// If `p` does not sum to one within `1e-9`, or holds a negative entry.
#[must_use]
pub fn shannon(p: &[f64]) -> f64 {
    let mut s = 0.0;
    let mut h = 0.0;
    for &v in p {
        assert!(v >= 0.0, "a probability vector has no negative entry, got {v}");
        s += v;
        if v > 0.0 {
            h -= v * v.ln();
        }
    }
    assert!((s - 1.0).abs() < 1e-9, "probabilities must sum to 1, these sum to {s}");
    h
}

/// The digamma function `psi(x) = d/dx ln Gamma(x)`, for `x > 0`.
///
/// Recurrence up to `x >= 15`, then the Bernoulli asymptotic series through `x^-10`, which is
/// good to about one part in `1e16` there.
///
/// # Panics
/// If `x` is not positive.
#[must_use]
pub fn digamma(x: f64) -> f64 {
    assert!(x > 0.0, "digamma is taken on positive arguments only, got {x}");
    let mut z = x;
    let mut acc = 0.0;
    while z < 15.0 {
        acc -= 1.0 / z;
        z += 1.0;
    }
    let inv = 1.0 / z;
    let u = inv * inv;
    let tail =
        u * (1.0 / 12.0 - u * (1.0 / 120.0 - u * (1.0 / 252.0 - u * (1.0 / 240.0 - u / 132.0))));
    acc + z.ln() - 0.5 * inv - tail
}

/// Counts of `symbols` over an alphabet of size `k`.
///
/// # Panics
/// If any symbol is `k` or larger.
#[must_use]
pub fn histogram(symbols: &[usize], k: usize) -> Vec<usize> {
    let mut c = vec![0usize; k];
    for &s in symbols {
        assert!(s < k, "symbol {s} is outside an alphabet of {k}");
        c[s] += 1;
    }
    c
}

/// A joint count table over two discrete variables, stored row-major.
#[derive(Clone, Debug)]
pub struct Joint {
    rows: usize,
    cols: usize,
    cells: Vec<usize>,
}

impl Joint {
    /// An empty `rows` by `cols` table.
    ///
    /// # Panics
    /// If either dimension is zero.
    #[must_use]
    pub fn new(rows: usize, cols: usize) -> Joint {
        assert!(rows > 0 && cols > 0, "a joint table needs both alphabets non-empty");
        Joint { rows, cols, cells: vec![0; rows * cols] }
    }

    /// A table from cells already counted, row-major.
    ///
    /// # Panics
    /// If `cells` is not `rows * cols` long, or either dimension is zero.
    #[must_use]
    pub fn from_cells(rows: usize, cols: usize, cells: Vec<usize>) -> Joint {
        assert!(rows > 0 && cols > 0, "a joint table needs both alphabets non-empty");
        assert_eq!(cells.len(), rows * cols, "{rows}x{cols} needs {} cells", rows * cols);
        Joint { rows, cols, cells }
    }

    /// A table counted from observed pairs.
    ///
    /// # Panics
    /// If any pair is outside the alphabets, or either dimension is zero.
    #[must_use]
    pub fn from_pairs(rows: usize, cols: usize, pairs: &[(usize, usize)]) -> Joint {
        let mut j = Joint::new(rows, cols);
        for &(x, y) in pairs {
            j.observe(x, y);
        }
        j
    }

    /// Count one observed pair.
    ///
    /// # Panics
    /// If `x` or `y` is outside its alphabet.
    pub fn observe(&mut self, x: usize, y: usize) {
        assert!(x < self.rows && y < self.cols, "({x},{y}) is outside {}x{}", self.rows, self.cols);
        self.cells[x * self.cols + y] += 1;
    }

    /// Rows in the table: the size of `X`'s alphabet.
    #[must_use]
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Columns in the table: the size of `Y`'s alphabet.
    #[must_use]
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// The cells, row-major — the joint count vector an estimator takes.
    #[must_use]
    pub fn cells(&self) -> &[usize] {
        &self.cells
    }

    /// Draws counted.
    #[must_use]
    pub fn total(&self) -> usize {
        total(&self.cells)
    }

    /// Marginal counts of `X`.
    #[must_use]
    pub fn marginal_x(&self) -> Vec<usize> {
        (0..self.rows).map(|r| self.cells[r * self.cols..(r + 1) * self.cols].iter().sum()).collect()
    }

    /// Marginal counts of `Y`.
    #[must_use]
    pub fn marginal_y(&self) -> Vec<usize> {
        (0..self.cols).map(|c| (0..self.rows).map(|r| self.cells[r * self.cols + c]).sum()).collect()
    }

    /// `I(X;Y) = H(X) + H(Y) - H(X,Y)` in nats, every term from `est`.
    ///
    /// May come out negative: a correction applied three times need not leave a non-negative
    /// remainder, and clamping at zero would hide exactly the over-correction the module documents.
    #[must_use]
    pub fn mutual_information(&self, est: Estimator) -> f64 {
        est.entropy(&self.marginal_x()) + est.entropy(&self.marginal_y())
            - est.entropy(&self.cells)
    }

    /// `H(Y|X) = H(X,Y) - H(X)` in nats, both terms from `est`.
    #[must_use]
    pub fn conditional_entropy(&self, est: Estimator) -> f64 {
        est.entropy(&self.cells) - est.entropy(&self.marginal_x())
    }
}

/// Why a set of states cannot be counted into a histogram.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NotSamples {
    /// The states were visited by a search, so their frequencies describe the search.
    NotDistributional {
        /// The search that produced them.
        method: &'static str,
    },
    /// Every state appears exactly once by construction, so the histogram is uniform whatever the
    /// model is; the distribution lives in the weights, not the counts.
    Enumerated,
    /// `2^spins` bins is more than a state histogram will allocate.
    TooManyBins {
        /// Spins in each state.
        spins: usize,
        /// The largest allowed.
        limit: usize,
    },
}

impl core::fmt::Display for NotSamples {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            NotSamples::NotDistributional { method } => write!(
                f,
                "{method} is a search: how often it saw a state is a fact about where it walked, \
                 and an entropy estimated from those frequencies estimates nothing"
            ),
            NotSamples::Enumerated => write!(
                f,
                "an enumerated set holds every state exactly once, so its histogram is uniform \
                 whatever the model is; the exact entropy is `shannon` of the Boltzmann weights"
            ),
            NotSamples::TooManyBins { spins, limit } => {
                write!(f, "2^{spins} bins exceeds the 2^{limit} cap on a state histogram")
            }
        }
    }
}

/// The states of `set`, if they are draws from a distribution at all.
///
/// The corrections here divide by the number of draws, so they are right for independent ones and
/// conservative for a chain; they are meaningless for the provenances this refuses.
///
/// # Errors
/// [`NotSamples::NotDistributional`] for a search, [`NotSamples::Enumerated`] for an enumeration.
pub fn draws(set: &SampleSet) -> Result<&[Vec<i8>], NotSamples> {
    match set.provenance() {
        Provenance::Search { method } => Err(NotSamples::NotDistributional { method }),
        Provenance::Enumerated { .. } => Err(NotSamples::Enumerated),
        Provenance::Chain { .. } | Provenance::Population { .. } => Ok(set.states()),
    }
}

/// Counts of whole `n`-spin states, indexed by the bitmask that sets bit `b` where spin `b` is up.
///
/// # Errors
/// [`NotSamples::TooManyBins`] above [`crate::samples::ENUMERATION_LIMIT`] spins.
///
/// # Panics
/// If any state is not `n` spins wide.
pub fn state_counts(states: &[Vec<i8>], n: usize) -> Result<Vec<usize>, NotSamples> {
    if n > ENUMERATION_LIMIT {
        return Err(NotSamples::TooManyBins { spins: n, limit: ENUMERATION_LIMIT });
    }
    let mut c = vec![0usize; 1usize << n];
    for s in states {
        assert_eq!(s.len(), n, "a state of {} spins in an {n}-spin set", s.len());
        let mut mask = 0usize;
        for (b, &v) in s.iter().enumerate() {
            if v > 0 {
                mask |= 1 << b;
            }
        }
        c[mask] += 1;
    }
    Ok(c)
}

/// The two-by-two joint count table of spins `i` and `j`, with down as index 0.
///
/// # Panics
/// If `i` or `j` is outside a state.
#[must_use]
pub fn spin_pair(states: &[Vec<i8>], i: usize, j: usize) -> Joint {
    let mut t = Joint::new(2, 2);
    for s in states {
        assert!(i < s.len() && j < s.len(), "spins {i},{j} outside a state of {}", s.len());
        t.observe(usize::from(s[i] > 0), usize::from(s[j] > 0));
    }
    t
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{Graph, GraphBuilder};
    use crate::rng::Pcg;

    // Every test here checks against something exactly known: a closed form (ln K, m*h(q),
    // psi(n) = -gamma + H_{n-1}), an exhaustive expectation over every count vector a multinomial
    // can produce, or an algebraic identity between two estimators on the same table. Nowhere is
    // one estimator's answer treated as another's truth -- a sampler checked against a sampler
    // agrees or disagrees and settles nothing. Where two estimators ARE compared, the claim is
    // about their bias, and both biases are measured against an exact entropy first.

    const GAMMA: f64 = 0.5772156649015329;

    /// Enumerate every way `n` draws can land in `k` ordered bins.
    fn each_composition(n: usize, k: usize, buf: &mut Vec<usize>, f: &mut dyn FnMut(&[usize])) {
        if k == 1 {
            buf.push(n);
            f(buf);
            buf.pop();
            return;
        }
        for first in 0..=n {
            buf.push(first);
            each_composition(n - first, k - 1, buf, f);
            buf.pop();
        }
    }

    fn ln_factorials(n: usize) -> Vec<f64> {
        let mut v = vec![0.0; n + 1];
        for i in 1..=n {
            v[i] = v[i - 1] + (i as f64).ln();
        }
        v
    }

    /// The exact expectation of `f` under the multinomial with cell probabilities `p` and `n`
    /// draws. No Monte Carlo: every count vector is visited with its own weight, so a bias this
    /// returns is the bias itself and not an estimate of one.
    fn exact_mean<F: Fn(&[usize]) -> f64>(p: &[f64], n: usize, f: F) -> f64 {
        let lnf = ln_factorials(n);
        let mut acc = 0.0;
        let mut mass = 0.0;
        let mut buf = Vec::new();
        each_composition(n, p.len(), &mut buf, &mut |c: &[usize]| {
            let mut lw = lnf[n];
            for (idx, &x) in c.iter().enumerate() {
                lw -= lnf[x];
                if x > 0 {
                    lw += x as f64 * p[idx].ln();
                }
            }
            let w = lw.exp();
            mass += w;
            acc += w * f(c);
        });
        assert!((mass - 1.0).abs() < 1e-9, "the multinomial did not sum to 1: {mass}");
        acc / mass
    }

    fn ring(n: usize, j: f64, h: f64) -> Graph {
        let mut b = GraphBuilder::new(n);
        for i in 0..n {
            b.couple(i, (i + 1) % n, j);
            b.bias(i, h);
        }
        b.build()
    }

    /// Draw `n` independent states from an exact Boltzmann table by inverse transform.
    ///
    /// Independent on purpose: the claim under test is about finite-sample bias, and a Gibbs chain
    /// would fold its autocorrelation into the same number and make the two indistinguishable.
    fn iid_states(p: &[f64], spins: usize, n: usize, seed: u64) -> Vec<Vec<i8>> {
        let mut cdf = Vec::with_capacity(p.len());
        let mut acc = 0.0;
        for &v in p {
            acc += v;
            cdf.push(acc);
        }
        let mut r = Pcg::new(seed, 1);
        (0..n)
            .map(|_| {
                let u = r.f64();
                let (mut lo, mut hi) = (0usize, cdf.len() - 1);
                while lo < hi {
                    let mid = (lo + hi) / 2;
                    if u <= cdf[mid] {
                        hi = mid;
                    } else {
                        lo = mid + 1;
                    }
                }
                (0..spins).map(|b| if lo >> b & 1 == 1 { 1i8 } else { -1 }).collect()
            })
            .collect()
    }

    fn mi_2x2(c: &[usize], e: Estimator) -> f64 {
        Joint::from_cells(2, 2, c.to_vec()).mutual_information(e)
    }

    #[test]
    fn digamma_matches_its_closed_form() {
        // psi(1) = -gamma, and psi(n+1) = psi(n) + 1/n makes psi(n) = -gamma + H_{n-1} exactly.
        assert!((digamma(1.0) + GAMMA).abs() < 1e-14, "psi(1) = {}", digamma(1.0));
        let mut harmonic = 0.0;
        for n in 1..200 {
            let want = -GAMMA + harmonic;
            let got = digamma(n as f64);
            assert!((got - want).abs() < 1e-13, "psi({n}) = {got}, closed form {want}");
            harmonic += 1.0 / n as f64;
        }
        // psi(1/2) = -gamma - 2 ln 2 is the half-integer anchor, which the recurrence cannot reach
        // from psi(1) and which therefore tests the asymptotic tail on its own.
        let want = -GAMMA - 2.0 * core::f64::consts::LN_2;
        assert!((digamma(0.5) - want).abs() < 1e-14, "psi(1/2) = {}", digamma(0.5));
        for &x in &[0.1, 0.7, 2.3, 9.9, 14.999, 15.001, 1e3] {
            let d = digamma(x + 1.0) - digamma(x);
            assert!((d - 1.0 / x).abs() < 1e-12, "psi({x}+1) - psi({x}) = {d}, want {}", 1.0 / x);
        }
    }

    /// `G(n)` as the shipped [`grassberger`] actually uses it, read back through its own API:
    /// a one-bin sample gives `ln n - (1/n) * n * G(n)`, so `G(n) = ln n - grassberger(&[n])`.
    fn shipped_g(n: usize) -> f64 {
        (n as f64).ln() - grassberger(&[n])
    }

    /// `E[n G(n)]` for `n` Poisson with mean `lam`, summed until the tail is beneath `f64`.
    fn poisson_mean_n_g(lam: f64, g: &dyn Fn(usize) -> f64) -> f64 {
        let nmax = (lam + 12.0 * lam.sqrt() + 40.0) as usize;
        let mut log_p = -lam;
        let mut acc = 0.0;
        for n in 1..=nmax {
            log_p += lam.ln() - (n as f64).ln();
            acc += log_p.exp() * n as f64 * g(n);
        }
        acc
    }

    #[test]
    fn the_grassberger_correction_moves_toward_its_own_exact_target() {
        // Two anchors on the term itself, because the bias tests below are far too loose to see
        // its SIGN: negating it leaves every one of them passing.
        //
        // First, arithmetic. G(1) = psi(1) - 1/2 and G(2) = psi(2) + 1/3, closed forms.
        assert!((shipped_g(1) - (-GAMMA - 0.5)).abs() < 1e-14, "G(1) = {}", shipped_g(1));
        assert!((shipped_g(2) - (1.0 - GAMMA + 1.0 / 3.0)).abs() < 1e-14, "G(2) = {}", shipped_g(2));

        // Second, the construction. `ln N - (1/N) sum n_i G(n_i)` is unbiased exactly when
        // E[n G(n)] = lam*ln(lam) for n ~ Poisson(lam) -- write H as ln N - (1/N) sum lam_i ln lam_i
        // and the two match term by term. That target is a closed form, so the residual is exact.
        // psi alone leaves a positive residual; the oscillating term exists to shrink it. Flip its
        // sign and the residual GROWS, drop it and the residual is unchanged: both are caught here.
        for &lam in &[0.5f64, 1.0, 2.0, 3.0, 5.0] {
            let target = lam * lam.ln();
            let bare = poisson_mean_n_g(lam, &|n| digamma(n as f64)) - target;
            let full = poisson_mean_n_g(lam, &shipped_g) - target;
            assert!(bare > 0.0, "lam = {lam}: psi alone should overshoot, got {bare:+}");
            assert!(full > 0.0, "lam = {lam}: the 1988 term must not overshoot past 0, got {full:+}");
            assert!(full < 0.9 * bare, "lam = {lam}: residual {full:+} against psi's {bare:+}");
        }
    }

    #[test]
    fn shannon_reproduces_the_closed_forms_it_is_the_oracle_for() {
        for k in [2usize, 3, 7, 64] {
            let p = vec![1.0 / k as f64; k];
            assert!((shannon(&p) - (k as f64).ln()).abs() < 1e-14, "uniform {k}");
        }
        // m independent Bernoulli(q) spins have H = m*h(q), and the 2^m-entry product distribution
        // must reproduce it -- the known Bernoulli product the estimators are checked against.
        for &q in &[0.1f64, 0.3, 0.5] {
            let m = 5usize;
            let mut p = vec![0.0; 1 << m];
            for (mask, cell) in p.iter_mut().enumerate() {
                let mut v = 1.0;
                for b in 0..m {
                    v *= if mask >> b & 1 == 1 { q } else { 1.0 - q };
                }
                *cell = v;
            }
            let h1 = -q * q.ln() - (1.0 - q) * (1.0 - q).ln();
            assert!((shannon(&p) - m as f64 * h1).abs() < 1e-13, "q = {q}");
        }
        assert_eq!(shannon(&[1.0, 0.0, 0.0]), 0.0);
    }

    #[test]
    fn balanced_counts_give_the_exact_closed_form() {
        // Perfectly balanced counts are the one case where plug-in is exactly right, so
        // Miller-Madow is then exactly plug-in plus the term it claims to add.
        for k in [2usize, 4, 8] {
            let c = vec![16usize; k];
            let n = 16 * k;
            assert!((plug_in(&c) - (k as f64).ln()).abs() < 1e-14, "k = {k}");
            let want = (k as f64).ln() + (k as f64 - 1.0) / (2.0 * n as f64);
            assert!((miller_madow(&c) - want).abs() < 1e-14, "k = {k}");
        }
        // A degenerate sample: one bin, all the mass. Plug-in and Miller-Madow are exactly zero;
        // Grassberger is not, and is documented not to be, since it never counts the support.
        assert_eq!(plug_in(&[40, 0, 0]), 0.0);
        assert_eq!(miller_madow(&[40, 0, 0]), 0.0);
        assert!(grassberger(&[40, 0, 0]).abs() < 0.05, "{}", grassberger(&[40, 0, 0]));
        for e in Estimator::ALL {
            assert_eq!(e.entropy(&[]), 0.0, "{} on an empty sample", e.label());
            assert_eq!(e.entropy(&[0, 0, 0]), 0.0, "{} on no draws", e.label());
        }
    }

    #[test]
    fn chao_shen_survives_a_sample_that_is_all_singletons() {
        // Coverage is 1 - f1/N, so N draws in N distinct bins give C = 0, every adjusted
        // probability zero, and 0*ln(0)/0 -- a NaN that walks straight into a published number.
        // The guard lowers f1 by one instead.
        let c = vec![1usize; 12];
        let h = chao_shen(&c);
        assert!(h.is_finite() && h > 0.0, "all-singleton Chao-Shen = {h}");
        // Twelve draws in twelve bins: the truth is at least ln 12, which is what plug-in returns
        // here, and a coverage correction must not push the estimate below it.
        assert!(h >= plug_in(&c) - 1e-12, "{h} against plug-in {}", plug_in(&c));
        assert!(chao_shen(&[1]).is_finite(), "a single draw");
    }

    #[test]
    fn the_plug_in_bias_is_minus_support_minus_one_over_twice_the_draws() {
        // The expectation is exact: every count vector, weighted by its multinomial probability.
        // The claim is that the leading term of E[H_ml] - H is -(K-1)/(2N), so the ratio of the
        // measured bias to that term must fall toward 1 from above as N grows.
        let p = vec![0.25; 4];
        let truth = 4.0f64.ln();
        let mut ratios = Vec::new();
        for n in [8usize, 16, 32, 64] {
            let bias = exact_mean(&p, n, plug_in) - truth;
            assert!(bias < 0.0, "N = {n}: plug-in must sit low, got {bias:+}");
            let ratio = bias / plug_in_bias(4, n);
            assert!((1.0..1.2).contains(&ratio), "N = {n}: bias over leading term = {ratio}");
            ratios.push(ratio);
        }
        for w in ratios.windows(2) {
            assert!(w[1] < w[0], "the ratio must fall toward 1: {ratios:?}");
        }
        assert!((ratios[3] - 1.0).abs() < 0.03, "at N = 64 the leading term is it: {}", ratios[3]);
    }

    #[test]
    fn every_correction_reduces_the_bias_at_small_n() {
        // The point of the module, measured rather than asserted -- and measured as an exact
        // expectation, so "reduces the bias" means the bias and not one lucky draw.
        let cases: [(Vec<f64>, usize); 5] = [
            (vec![0.25; 4], 8),
            (vec![0.25; 4], 16),
            (vec![1.0 / 6.0; 6], 12),
            (vec![0.49, 0.21, 0.21, 0.09], 16),
            (vec![0.7, 0.2, 0.07, 0.03], 16),
        ];
        for (p, n) in cases {
            let truth = shannon(&p);
            let base = (exact_mean(&p, n, plug_in) - truth).abs();
            assert!(base > 0.05, "the plug-in bias should be worth correcting: {base}");
            for e in [Estimator::MillerMadow, Estimator::Grassberger, Estimator::ChaoShen] {
                let bias = exact_mean(&p, n, |c: &[usize]| e.entropy(c)) - truth;
                assert!(
                    bias.abs() < 0.7 * base,
                    "{} at K = {}, N = {n}: |bias| {} against plug-in {base}",
                    e.label(),
                    p.len(),
                    bias.abs()
                );
            }
        }
    }

    #[test]
    fn an_outer_product_table_has_exactly_zero_plug_in_mutual_information() {
        // An algebraic identity, not a limit: when n_xy = a_x * b_y exactly, the joint plug-in
        // entropy splits into the two marginal ones and the estimate is zero to the arithmetic's
        // last bits. Anything else is an indexing mistake in the marginals.
        let a = [3usize, 5, 2];
        let b = [4usize, 1, 6, 9];
        let cells: Vec<usize> =
            (0..a.len()).flat_map(|x| (0..b.len()).map(move |y| a[x] * b[y])).collect();
        let t = Joint::from_cells(a.len(), b.len(), cells);
        // row sums are a_x * sum(b) = a_x * 20, column sums are sum(a) * b_y = 10 * b_y
        assert_eq!(t.marginal_x(), vec![60, 100, 40]);
        assert_eq!(t.marginal_y(), vec![40, 10, 60, 90]);
        assert!(t.mutual_information(Estimator::PlugIn).abs() < 1e-13);
        // and the chain rule, which is the same identity read the other way round
        let cond = t.conditional_entropy(Estimator::PlugIn);
        let want = plug_in(t.cells()) - plug_in(&t.marginal_x());
        assert!((cond - want).abs() < 1e-15);
    }

    #[test]
    fn miller_madow_removes_the_mutual_information_bias_and_grassberger_does_not() {
        // Oracle: X and Y independent, so I = 0 exactly, whatever the marginals are.
        //
        // On a fully occupied 2x2 table the Miller-Madow correction is an exact identity,
        // (1 + 1 - 3)/(2N) = -1/(2N), and -1/(2N) is precisely -(Kx-1)(Ky-1)/(2N), the leading
        // plug-in bias. Grassberger's is (2 + 2 - 4)/(2N) = 0: it corrects nothing here.
        let t = Joint::from_cells(2, 2, vec![7, 3, 4, 6]);
        let n = t.total() as f64;
        let d =
            t.mutual_information(Estimator::PlugIn) - t.mutual_information(Estimator::MillerMadow);
        assert!((d - 1.0 / (2.0 * n)).abs() < 1e-14, "the 2x2 identity: {d} vs {}", 1.0 / (2.0 * n));

        // Now the bias itself, exactly, over every 2x2 table that 32 draws can produce.
        let p = vec![0.25; 4];
        let plug = exact_mean(&p, 32, |c: &[usize]| mi_2x2(c, Estimator::PlugIn));
        let mm = exact_mean(&p, 32, |c: &[usize]| mi_2x2(c, Estimator::MillerMadow));
        let gr = exact_mean(&p, 32, |c: &[usize]| mi_2x2(c, Estimator::Grassberger));
        let leading = 1.0 / (2.0 * 32.0);
        assert!(plug > 0.0, "plug-in mutual information is biased UP, got {plug}");
        assert!((plug / leading - 1.0).abs() < 0.15, "plug-in bias {plug} against {leading}");
        assert!(mm.abs() < 0.15 * plug, "Miller-Madow should cancel it: {mm} against {plug}");
        assert!(gr > 0.8 * plug, "Grassberger cancels nothing on a 2x2: {gr} against {plug}");
    }

    #[test]
    fn entropy_of_an_ising_model_against_its_exact_boltzmann_entropy() {
        let (n, beta) = (6usize, 0.8);
        let g = ring(n, 0.7, 0.3);
        let p = crate::ising::exact_boltzmann(&g, beta);
        let truth = shannon(&p);

        // Cross-check the oracle by a second exact route: S = ln Z + beta<E>, with ln Z from
        // variable elimination and <E> from the same table. Two independent computations of one
        // number, so a mistake in either shows up here rather than inside an estimator test.
        let ex = crate::exact::Elimination::default().log_partition(&g, beta).unwrap();
        let mean_e: f64 = (0..p.len())
            .map(|m| {
                let s: Vec<i8> = (0..n).map(|b| if m >> b & 1 == 1 { 1i8 } else { -1 }).collect();
                p[m] * g.energy(&s)
            })
            .sum();
        let identity = ex.log_z.unwrap() + beta * mean_e;
        assert!((truth - identity).abs() < 1e-9, "S = {truth}, ln Z + b<E> = {identity}");

        // 64 bins from 200 draws: the regime the corrections exist for. Averaged over 200
        // independent replicates, so what is compared is the bias, against the exact entropy.
        let (reps, per) = (200usize, 200usize);
        let mut mean = [0.0f64; 4];
        for r in 0..reps {
            let states = iid_states(&p, n, per, 900 + r as u64);
            let counts = state_counts(&states, n).unwrap();
            assert_eq!(total(&counts), per);
            for (k, e) in Estimator::ALL.iter().enumerate() {
                mean[k] += e.entropy(&counts) / reps as f64;
            }
        }
        let base = (mean[0] - truth).abs();
        assert!(mean[0] < truth - 0.05, "plug-in must sit low: {} vs {truth}", mean[0]);
        for (k, e) in Estimator::ALL.iter().enumerate().skip(1) {
            assert!(
                (mean[k] - truth).abs() < 0.8 * base,
                "{} bias {:+} against plug-in {:+}",
                e.label(),
                mean[k] - truth,
                mean[0] - truth
            );
        }
        // and with enough draws every one of them converges on the closed form
        let states = iid_states(&p, n, 200_000, 7);
        let counts = state_counts(&states, n).unwrap();
        for e in Estimator::ALL {
            let h = e.entropy(&counts);
            assert!((h - truth).abs() < 0.01, "{} at 200k draws: {h} vs {truth}", e.label());
        }
    }

    #[test]
    fn spin_pair_mutual_information_against_the_exact_two_spin_joint() {
        let (n, beta) = (6usize, 0.8);
        let g = ring(n, 0.7, 0.3);
        let p = crate::ising::exact_boltzmann(&g, beta);
        // The exact two-spin joint, by marginalising the exact Boltzmann table.
        let exact_mi = |i: usize, j: usize| {
            let mut q = [[0.0f64; 2]; 2];
            for (m, &pm) in p.iter().enumerate() {
                q[(m >> i) & 1][(m >> j) & 1] += pm;
            }
            let px = [q[0][0] + q[0][1], q[1][0] + q[1][1]];
            let py = [q[0][0] + q[1][0], q[0][1] + q[1][1]];
            let mut mi = 0.0;
            for a in 0..2 {
                for b in 0..2 {
                    if q[a][b] > 0.0 {
                        mi += q[a][b] * (q[a][b] / (px[a] * py[b])).ln();
                    }
                }
            }
            mi
        };
        let states = iid_states(&p, n, 200_000, 4242);
        for (i, j) in [(0usize, 1usize), (0, 3)] {
            let truth = exact_mi(i, j);
            assert!(truth > 0.0, "a coupled ring correlates its spins: I({i};{j}) = {truth}");
            let t = spin_pair(&states, i, j);
            assert_eq!(t.total(), states.len());
            for e in Estimator::ALL {
                let mi = t.mutual_information(e);
                assert!((mi - truth).abs() < 0.005, "{} I({i};{j}) = {mi}, exact {truth}", e.label());
            }
        }
    }

    #[test]
    fn states_that_are_not_draws_are_refused() {
        let g = ring(4, 1.0, 0.0);
        let enumerated = crate::samples::enumerate(&g, 0.5).unwrap();
        assert_eq!(draws(&enumerated), Err(NotSamples::Enumerated));
        let searched = SampleSet::from_search(vec![vec![1i8, -1, 1, 1]], vec![0.0], "tabu");
        assert_eq!(draws(&searched), Err(NotSamples::NotDistributional { method: "tabu" }));
        let chain = SampleSet::from_chain(
            vec![vec![1i8, -1, 1, 1], vec![-1, -1, 1, 1]],
            vec![0.0, 0.0],
            0.5,
            10,
            2,
        );
        assert_eq!(draws(&chain).unwrap().len(), 2);
        assert_eq!(
            state_counts(&[], ENUMERATION_LIMIT + 1),
            Err(NotSamples::TooManyBins { spins: ENUMERATION_LIMIT + 1, limit: ENUMERATION_LIMIT })
        );
        assert!(NotSamples::Enumerated.to_string().contains("shannon"));
    }

    #[test]
    fn histograms_and_joint_tables_count_what_they_are_given() {
        assert_eq!(histogram(&[0, 2, 2, 1, 2], 4), vec![1, 1, 3, 0]);
        let t = Joint::from_pairs(2, 3, &[(0, 0), (0, 2), (1, 1), (1, 1), (0, 0)]);
        assert_eq!(t.cells(), &[2, 0, 1, 0, 2, 0]);
        assert_eq!(t.marginal_x(), vec![3, 2]);
        assert_eq!(t.marginal_y(), vec![2, 2, 1]);
        assert_eq!((t.rows(), t.cols(), t.total()), (2, 3, 5));
        // a state histogram indexes by the bitmask that sets a bit where the spin is up
        let states = vec![vec![-1i8, -1], vec![1, -1], vec![1, 1], vec![1, 1]];
        assert_eq!(state_counts(&states, 2).unwrap(), vec![1, 1, 0, 2]);
        assert_eq!(spin_pair(&states, 0, 1).cells(), &[1, 0, 1, 2]);
        assert_eq!(support(&[0, 3, 0, 1]), 2);
        assert_eq!(total(&[0, 3, 0, 1]), 4);
    }
}
