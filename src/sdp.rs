//! A **certified** SDP lower bound on the ground energy, in std-only Rust.
//!
//! The max-cut SDP relaxation is the standard strong bound, and the obvious way to compute it —
//! Burer-Monteiro / the mixing method — produces a *primal feasible* point whose value bounds the
//! SDP optimum **from the wrong side**. Measured on G1: a random `V` yields a naive "upper bound"
//! of 9579.94 against a true max cut of 11624. Unsound by 2044.
//!
//! The way out is not to certify an eigenvalue. It is to exhibit a **dual** point and check one
//! thing about it.
//!
//! # Weak duality does all the work
//!
//! Writing `C` for the cost matrix with zero diagonal, the primal is
//! `p* = min { <C,X> : X ⪰ 0, X_aa = 1 }` and its dual is
//! `d* = max { eᵀy : C − Diag(y) ⪰ 0 }`. For **any** primal-feasible `X` and **any** dual-feasible
//! `y`,
//!
//! ```text
//! <C,X> − eᵀy = <C − Diag(y), X> ≥ 0
//! ```
//!
//! because the trace inner product of two PSD matrices is non-negative. Every `s ∈ {−1,+1}ⁿ` gives
//! a feasible `X = ssᵀ`, so
//!
//! ```text
//! eᵀy  ≤  p*  ≤  min_s E(s)      for every dual-feasible y
//! ```
//!
//! No optimality, convergence, or rank assumption appears anywhere. **The mixing method, the rank,
//! the Lanczos estimate and the seed are heuristics for choosing `y`.** They can be arbitrarily
//! bad and only move the bound down — never make it invalid.
//!
//! # The one claim that has to be true
//!
//! `C − Diag(y')` is PSD. That is discharged by [Rump 2006]'s criterion rather than by an
//! eigensolver: shift the diagonal down by a computable constant `c`, run a plain `f64` Cholesky,
//! and **completion proves definiteness**. It is a theorem about what a finished floating-point
//! factorisation implies under IEEE-754 round-to-nearest, so it transfers to Rust verbatim — with
//! one caveat that is enforced here: no `mul_add` anywhere in the Cholesky, because fusion changes
//! the error model the theorem assumes.
//!
//! Lanczos is therefore demoted to a *search heuristic* for the shift. Rayleigh-Ritz gives
//! `θ ≥ λ_min`, so the estimate is optimistic and using it unshifted would be unsound; the shift is
//! grown until the Cholesky verifies, and bisected back to recover the overshoot.
//!
//! # What it is worth
//!
//! Measured against the other bounds in this crate, by `cargo run --release --example gset_gap`:
//!
//! ```text
//!   instance   odd_cycle UB   sdp UB    cut found    gap
//!   G1            14958       12083        11624    3.8%
//!   G14            3602        3192         3058    4.2%
//!   G11             579         629          564    2.6%
//! ```
//!
//! The SDP takes G1's gap from 22.3% to **3.8%** and G14's from 15.1% to **4.2%** — and *loses* on
//! G11, the degree-4 torus, where the cycle bound is stronger and sets the 2.6%. Both are sound, so
//! [`crate::bound`] users should take the maximum. That is also why [`certified`] never returns
//! worse than [`crate::bound::decoupled`]: the Gershgorin dual point is exactly that bound, and it
//! is used as a floor.
//!
//! **The sweep count is not the dial it looks like.** Twenty times the work moves nothing: 733 /
//! 731 / 733 on G11 and 12223 / 12224 / 12224 on G1 at 200 / 1,000 / 4,000 sweeps. The mixing
//! method reaches a stationary point early, and what the bound is worth after that is decided by
//! the shift — which is why `lanczos_min` returning something that was not an eigenvalue cost
//! every number in the table above, and why it now has a test.
//!
//! [Rump 2006]: S. M. Rump, "Verification of positive definiteness", BIT Numerical Mathematics.

use crate::bound::Bound;
use crate::graph::Graph;
use crate::rng::Pcg;

/// A dual point, and the bound it certifies.
///
/// The artefact a sceptic runs. [`Certificate::verify`] rebuilds the cost matrix from the graph,
/// re-checks positive definiteness and re-sums `y` — with no reference to the mixing method, the
/// rank, Lanczos, or the seed that produced it.
#[derive(Clone, Debug)]
pub struct Certificate {
    /// The dual point. Length `n`, or `n+1` when fields forced homogenisation.
    pub y: Vec<f64>,
    /// `eᵀy`, computed exactly by the snap-down grid.
    pub value: f64,
    /// Whether a gauge spin was prepended to absorb non-zero fields.
    pub homogenised: bool,
    /// Rump's constant for the verified matrix, so the slack given up is visible.
    pub rump_c: f64,
    /// Mixing sweeps run, and the rank used.
    pub sweeps: usize,
    pub rank: usize,
}

/// Why a certificate failed to re-verify.
#[derive(Clone, Debug, PartialEq)]
pub enum CertError {
    /// `y` is not the length this graph implies.
    Shape { got: usize, want: usize },
    /// The Cholesky did not complete: `C − Diag(y)` is not provably PSD.
    NotPsd,
    /// A non-finite entry.
    NotFinite,
}

impl core::fmt::Display for CertError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            CertError::Shape { got, want } => write!(f, "y has {got} entries and this graph needs {want}"),
            CertError::NotPsd => write!(
                f,
                "C - Diag(y) did not verify as positive definite, so this y is not dual feasible \
                 and certifies nothing"
            ),
            CertError::NotFinite => write!(f, "the certificate contains a non-finite value"),
        }
    }
}

/// The cost matrix: symmetric, **zero diagonal**, with `E(s) = sᵀCs`.
///
/// Held **twice**, because the two consumers want opposite things. The Cholesky factors the dense
/// copy and has to: verification is a claim about a completed factorisation, and fill-in is not
/// optional. The mixing sweep and Lanczos read the CSR copy, and they exist on the other side of a
/// `sweeps × n` loop where the Cholesky runs a handful of times.
///
/// The CSR is built by **scanning the dense rows in ascending column order**, so a sparse sweep
/// accumulates the same terms in the same order as the dense loop it replaced — bit-identical
/// rather than merely equivalent, which is the only version of this optimisation worth having in a
/// file whose output is a certificate. [`Cost::gather`] is the shared kernel, and the tests hold it
/// against a dense reference.
struct Cost {
    n: usize,
    dense: Vec<f64>,
    /// Row starts into `nz_col` / `nz_val`. Length `n + 1`.
    nz_off: Vec<usize>,
    nz_col: Vec<u32>,
    nz_val: Vec<f64>,
    homogenised: bool,
}

impl Cost {
    /// `C[i][j] = -J_ij/2`, and a gauge spin at index 0 when any field is non-zero.
    ///
    /// The zero diagonal is load-bearing twice over: it makes `sᵀCs` equal the energy with no
    /// constant term, and it makes the mixing update an exact coordinate minimiser.
    fn build(g: &Graph) -> Cost {
        let homog = g.h.iter().any(|&h| h != 0.0);
        let n = if homog { g.n + 1 } else { g.n };
        let off = usize::from(homog);
        let mut dense = vec![0.0; n * n];
        for i in 0..g.n {
            if homog && g.h[i] != 0.0 {
                let v = -g.h[i] / 2.0;
                dense[i + off] = v;
                dense[(i + off) * n] = v;
            }
            for k in g.offset[i]..g.offset[i + 1] {
                let j = g.nbr[k] as usize;
                let v = -g.w[k] / 2.0;
                // Written to both triangles from the same value, never re-read from the CSR, so
                // the matrix is exactly symmetric rather than symmetric to rounding.
                dense[(i + off) * n + (j + off)] = v;
                dense[(j + off) * n + (i + off)] = v;
            }
        }
        for a in 0..n {
            dense[a * n + a] = 0.0;
        }
        // Ascending column order, read out of the dense row itself rather than rebuilt from the
        // graph's CSR -- whose neighbour lists are not required to be sorted. See `Cost`.
        let mut nz_off = Vec::with_capacity(n + 1);
        let mut nz_col = Vec::new();
        let mut nz_val = Vec::new();
        nz_off.push(0);
        for a in 0..n {
            for b in 0..n {
                let v = dense[a * n + b];
                if v != 0.0 {
                    nz_col.push(b as u32);
                    nz_val.push(v);
                }
            }
            nz_off.push(nz_col.len());
        }
        Cost { n, dense, nz_off, nz_col, nz_val, homogenised: homog }
    }

    #[inline]
    fn row(&self, a: usize) -> &[f64] {
        &self.dense[a * self.n..(a + 1) * self.n]
    }

    /// `g <- C[a,:] · V`, over the non-zeros of row `a`, with `V` stored row-major at width `k`.
    ///
    /// The hot kernel of the whole module: the mixing method calls it `sweeps × n` times and
    /// Lanczos once per step per row. Skipping the zeros is the entire optimisation — on a
    /// degree-4 instance at n = 800 the dense form did 800 column visits to find 4 that mattered.
    #[inline]
    fn gather(&self, a: usize, v: &[f64], k: usize, g: &mut [f64]) {
        g.iter_mut().for_each(|x| *x = 0.0);
        let (s, e) = (self.nz_off[a], self.nz_off[a + 1]);
        for (&b, &c) in self.nz_col[s..e].iter().zip(&self.nz_val[s..e]) {
            let vb = &v[b as usize * k..b as usize * k + k];
            for t in 0..k {
                g[t] += c * vb[t];
            }
        }
    }
}

/// Rump's constant `c` for a symmetric `f64` matrix given as `C − Diag(y)`.
fn rump_c(cost: &Cost, y: &[f64]) -> f64 {
    let n = cost.n;
    let eps = f64::EPSILON / 2.0; // 2^-53
    let eta = f64::from_bits(1); // 2^-1074
    let k = (n + 1) as f64;
    let gam = k * eps / (1.0 - k * eps);
    let tr: f64 = (0..n).map(|a| -y[a]).sum();
    let maxd = (0..n).map(|a| -y[a]).fold(f64::NEG_INFINITY, f64::max);
    let m = 3.0 * (2.0 * n as f64 + maxd);
    let c = gam / (1.0 - gam) * tr + n as f64 * m * eta;
    // Doubled, plus a floor: this also covers the rounding of `c` itself, at a cost of a factor two
    // on a quantity measured near 1e-10.
    2.0 * c + f64::from_bits(0x03d0_0000_0000_0000)
}

/// Is `C − Diag(y)` positive definite? Proven by a completed Cholesky, not estimated.
///
/// Returns `false` on any doubt. A false negative costs a looser bound; a false positive would
/// invalidate every number this module produces, which is why the diagonal is shifted **down** by
/// Rump's `c` first and every pivot is checked for strict positivity and finiteness.
fn verify_psd(cost: &Cost, y: &[f64], c: f64) -> bool {
    let n = cost.n;
    let mut a = vec![0.0f64; n * n];
    for i in 0..n {
        let row = cost.row(i);
        a[i * n..i * n + n].copy_from_slice(row);
        // S = C - Diag(y), then the verification shift, then one ULP down for the rounding of the
        // subtraction itself.
        let d = -y[i] - c;
        a[i * n + i] = if d.is_finite() { next_down(d) } else { return false };
    }
    // Plain right-looking Cholesky. NO `mul_add`: fusion changes the error model Rump's lemma
    // assumes, and a "harmless" optimisation here would silently void the proof.
    for j in 0..n {
        let mut d = a[j * n + j];
        for k in 0..j {
            let v = a[j * n + k];
            d -= v * v;
        }
        if !(d > 0.0) || !d.is_finite() {
            return false;
        }
        let l = d.sqrt();
        a[j * n + j] = l;
        for i in (j + 1)..n {
            let mut s = a[i * n + j];
            for k in 0..j {
                s -= a[i * n + k] * a[j * n + k];
            }
            let v = s / l;
            if !v.is_finite() {
                return false;
            }
            a[i * n + j] = v;
        }
    }
    true
}

fn next_down(x: f64) -> f64 {
    if x.is_nan() || x == f64::NEG_INFINITY {
        return x;
    }
    if x == 0.0 {
        return -f64::from_bits(1);
    }
    if x > 0.0 {
        f64::from_bits(x.to_bits() - 1)
    } else {
        f64::from_bits(x.to_bits() + 1)
    }
}

/// The Gershgorin dual point: [`crate::bound::decoupled`], nudged until it actually verifies.
///
/// `y_a = -Σ_{b≠a}|C[a][b]|` makes `C − Diag(y)` diagonally dominant with the row sums **exactly
/// balanced**, which is positive SEMI-definite and singular. The verifier proves positive
/// DEFINITENESS via a Cholesky, and a singular matrix has a zero pivot — so the textbook Gershgorin
/// point does not verify. PSD is not PD, and that gap is the whole difficulty of certifying by
/// factorisation.
///
/// A *relative* nudge `r_a·2⁻⁴⁰` fixes the balanced rows and still fails, because a node with **no
/// edges** has `r_a = 0`: its diagonal is `MIN_POSITIVE`, Rump's shift `c` drives it negative, and
/// the Cholesky stops. Measured at n=10, seed 107: `c = 1.6e-14` against a diagonal of `5e-324`.
///
/// So the nudge is grown until the verifier accepts, rather than set to a constant chosen by
/// argument. Termination is not in doubt — a large enough diagonal makes any symmetric matrix
/// strictly dominant — and the loop reports how far it had to go.
fn gershgorin_verified(cost: &Cost) -> Option<(Vec<f64>, f64)> {
    let rows: Vec<f64> =
        (0..cost.n).map(|a| cost.row(a).iter().map(|v| v.abs()).sum::<f64>()).collect();
    // A scale for the ADDITIVE part, so an edgeless row is nudged by something meaningful rather
    // than by the smallest positive f64.
    let scale = rows.iter().cloned().fold(0.0f64, f64::max).max(1.0);
    let mut nudge = scale * (2.0f64).powi(-40);
    for _ in 0..80 {
        let y: Vec<f64> = rows.iter().map(|r| -(r + nudge)).collect();
        let ys = snap_down(&y, 0.0);
        let c = rump_c(cost, &ys);
        if verify_psd(cost, &ys, c) {
            return Some((ys, c));
        }
        nudge *= 4.0;
    }
    None
}

/// Mixing-method sweeps, returning the dual read-off `y_a = ⟨v_a, g_a⟩`.
fn mixing(cost: &Cost, rank: usize, sweeps: usize, seed: u64) -> Vec<f64> {
    let n = cost.n;
    let k = rank.clamp(1, n.max(1));
    let mut rng = Pcg::new(seed, 0x5D_9A);
    let mut v = vec![0.0f64; n * k];
    for a in 0..n {
        loop {
            // Box-Muller, so the start is isotropic rather than cube-shaped.
            for t in 0..k {
                let u1 = rng.f64().max(1e-12);
                let u2 = rng.f64();
                v[a * k + t] = (-2.0 * u1.ln()).sqrt() * (core::f64::consts::TAU * u2).cos();
            }
            let nrm = v[a * k..a * k + k].iter().map(|x| x * x).sum::<f64>().sqrt();
            if nrm > 0.0 && nrm.is_finite() {
                for t in 0..k {
                    v[a * k + t] /= nrm;
                }
                break;
            }
            // Redraw with fresh RNG output rather than perturbing, so the run stays reproducible
            // from the seed alone.
        }
    }
    let mut g = vec![0.0f64; k];
    for _ in 0..sweeps {
        for a in 0..n {
            cost.gather(a, &v, k, &mut g);
            let nrm = g.iter().map(|x| x * x).sum::<f64>().sqrt();
            if nrm > 0.0 && nrm.is_finite() {
                for t in 0..k {
                    v[a * k + t] = -g[t] / nrm;
                }
            }
            // Degenerate row: leave v_a alone. Documented, not an error.
        }
    }
    // One final pass so `y` matches the final `V` exactly.
    (0..n)
        .map(|a| {
            cost.gather(a, &v, k, &mut g);
            (0..k).map(|t| v[a * k + t] * g[t]).sum::<f64>()
        })
        .collect()
}

/// Snap `y + shift` down onto a power-of-two grid.
///
/// Two jobs. Feasibility: `y' ≤ y + shift` componentwise, and adding a non-negative diagonal to a
/// PSD matrix keeps it PSD, so rounding down can only help. Exactness: every `y'_a` is a multiple
/// of the same power of two, so `Σ y'_a` is computed with **zero** rounding error and the reported
/// bound is the number that was verified.
fn snap_down(y: &[f64], shift: f64) -> Vec<f64> {
    let n = y.len() as f64;
    let maxa = y.iter().map(|v| (v + shift).abs()).fold(0.0f64, f64::max);
    if maxa == 0.0 || !maxa.is_finite() {
        return y.iter().map(|v| v + shift).collect();
    }
    let e = (n * maxa).log2().ceil() - 52.0;
    let gr = (2.0f64).powf(e);
    y.iter().map(|v| ((v + shift) / gr).floor() * gr).collect()
}

/// Lanczos estimate of `λ_min(C − Diag(y))`. A **heuristic** for choosing the shift.
///
/// Rayleigh-Ritz gives `θ ≥ λ_min`, so this is optimistic and must never be used unshifted.
fn lanczos_min(cost: &Cost, y: &[f64], steps: usize, seed: u64) -> f64 {
    let n = cost.n;
    if n == 0 {
        return 0.0;
    }
    let m = steps.min(n).max(1);
    let mut rng = Pcg::new(seed, 0x1A_9C05);
    let mut q: Vec<Vec<f64>> = Vec::with_capacity(m + 1);
    let mut v: Vec<f64> = (0..n).map(|_| rng.f64() * 2.0 - 1.0).collect();
    let nrm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if !(nrm > 0.0) {
        return 0.0;
    }
    v.iter_mut().for_each(|x| *x /= nrm);
    q.push(v);
    let (mut alpha, mut beta) = (Vec::new(), Vec::new());
    for j in 0..m {
        let qj = &q[j];
        let mut w: Vec<f64> = (0..n)
            .map(|a| {
                let (s, e) = (cost.nz_off[a], cost.nz_off[a + 1]);
                let mut acc = -y[a] * qj[a];
                for (&b, &c) in cost.nz_col[s..e].iter().zip(&cost.nz_val[s..e]) {
                    acc += c * qj[b as usize];
                }
                acc
            })
            .collect();
        let aj: f64 = (0..n).map(|i| w[i] * q[j][i]).sum();
        alpha.push(aj);
        // FULL reorthogonalisation, twice. Lanczos loses orthogonality fast, and a ghost eigenvalue
        // here would only mislead the shift search -- but a search that starts in the wrong place
        // costs iterations of dense Cholesky, which is the expensive part.
        for _ in 0..2 {
            for qi in q.iter() {
                let d: f64 = (0..n).map(|i| w[i] * qi[i]).sum();
                for i in 0..n {
                    w[i] -= d * qi[i];
                }
            }
        }
        let bj = w.iter().map(|x| x * x).sum::<f64>().sqrt();
        if bj < 1e-14 * aj.abs().max(1.0) {
            break;
        }
        beta.push(bj);
        w.iter_mut().for_each(|x| *x /= bj);
        q.push(w);
    }
    let k = alpha.len();
    if k == 0 {
        return 0.0;
    }
    let mut t = vec![0.0f64; k * k];
    for i in 0..k {
        t[i * k + i] = alpha[i];
        if i + 1 < k {
            t[i * k + i + 1] = beta[i];
            t[(i + 1) * k + i] = beta[i];
        }
    }
    // `jacobi_eig` RETURNS the eigenVECTOR matrix and leaves the eigenVALUES on the diagonal of the
    // matrix it was given. This line used to fold `min` over the return value, which is the most
    // negative eigenvector COMPONENT -- a number in [-1, 0] on every instance, with no relation to
    // the spectrum. Nothing failed, because the Cholesky is what makes the bound sound and a bad
    // `theta` only costs tightness; the bound was quietly loose instead of wrong.
    let _vectors = crate::linalg::jacobi_eig(&mut t, k);
    (0..k).map(|i| t[i * k + i]).fold(f64::INFINITY, f64::min)
}

/// How hard to try.
#[derive(Clone, Copy, Debug)]
pub struct Params {
    /// Mixing sweeps. More is tighter and never changes soundness.
    pub sweeps: usize,
    /// Rank of `V`. `None` uses the Barvinok-Pataki value `⌈√(2n)⌉ + 1`, below which the
    /// relaxation can have spurious stationary points.
    pub rank: Option<usize>,
    /// Lanczos steps for the shift search.
    pub lanczos: usize,
}

impl Default for Params {
    fn default() -> Self {
        Params { sweeps: 200, rank: None, lanczos: 64 }
    }
}

/// A certified lower bound on `min_s E(s)`, with the dual point that proves it.
///
/// Never worse than [`crate::bound::decoupled`], because the Gershgorin point is used as a floor.
pub fn certified(g: &Graph, p: &Params, seed: u64) -> (Bound, Certificate) {
    let cost = Cost::build(g);
    let n = cost.n;
    if n == 0 {
        let b = Bound { value: 0.0, parts: 0, method: "sdp: empty graph", rounds: 0, best_round: 0 };
        let c = Certificate {
            y: Vec::new(), value: 0.0, homogenised: false, rump_c: 0.0, sweeps: 0, rank: 0,
        };
        return (b, c);
    }
    let rank = p.rank.unwrap_or_else(|| ((2.0 * n as f64).sqrt().ceil() as usize + 1).min(n));

    // The floor, grown until it verifies. `None` would mean even a hugely dominant diagonal failed
    // the Cholesky, which cannot happen for a finite matrix -- but it is returned rather than
    // unwrapped, because a panic here would be a panic inside a soundness check.
    let Some((mut best_y, mut rc)) = gershgorin_verified(&cost) else {
        let b = Bound { value: f64::NEG_INFINITY, parts: 0, method: "sdp: no dual point verified", rounds: 0, best_round: 0 };
        let c = Certificate { y: Vec::new(), value: f64::NEG_INFINITY, homogenised: cost.homogenised, rump_c: 0.0, sweeps: 0, rank };
        return (b, c);
    };
    let mut best_val: f64 = best_y.iter().sum();

    let y = mixing(&cost, rank, p.sweeps, seed);
    let theta = lanczos_min(&cost, &y, p.lanczos, seed);

    // Grow the shift until the Cholesky verifies, then bisect back to recover the overshoot.
    let mut delta = (theta.abs() * 1e-6).max(1e-13);
    let mut accepted: Option<(Vec<f64>, f64, f64)> = None;
    let mut last_fail = 0.0f64;
    for _ in 0..64 {
        let cand = theta - delta;
        let ys = snap_down(&y, cand);
        let c = rump_c(&cost, &ys);
        if verify_psd(&cost, &ys, c) {
            let v: f64 = ys.iter().sum();
            accepted = Some((ys, v, c));
            break;
        }
        last_fail = delta;
        delta *= 2.0;
    }
    if let Some((_, _, _)) = &accepted {
        let mut lo = last_fail;
        let mut hi = delta;
        for _ in 0..8 {
            let mid = 0.5 * (lo + hi);
            let ys = snap_down(&y, theta - mid);
            let c = rump_c(&cost, &ys);
            if verify_psd(&cost, &ys, c) {
                hi = mid;
                let v: f64 = ys.iter().sum();
                accepted = Some((ys, v, c));
            } else {
                lo = mid;
            }
        }
    }

    let mut sweeps_used = 0usize;
    if let Some((ys, v, c)) = accepted {
        if v > best_val {
            best_val = v;
            best_y = ys;
            rc = c;
            sweeps_used = p.sweeps;
        }
    }

    let cert = Certificate {
        y: best_y,
        value: best_val,
        homogenised: cost.homogenised,
        rump_c: rc,
        sweeps: sweeps_used,
        rank,
    };
    let b = Bound {
        value: best_val,
        parts: 1,
        method: "sdp: mixing-method primal, dual point verified positive definite (Rump)",
        rounds: p.sweeps,
        best_round: p.sweeps,
    };
    (b, cert)
}

impl Certificate {
    /// Re-check this certificate against the graph, from scratch.
    ///
    /// Rebuilds the cost matrix, re-verifies positive definiteness and re-sums `y`. It never looks
    /// at the mixing method, the rank, Lanczos, or the seed — which is the point: the bound stands
    /// on the dual point alone.
    pub fn verify(&self, g: &Graph) -> Result<f64, CertError> {
        let cost = Cost::build(g);
        if self.y.len() != cost.n {
            return Err(CertError::Shape { got: self.y.len(), want: cost.n });
        }
        if self.y.iter().any(|v| !v.is_finite()) || !self.value.is_finite() {
            return Err(CertError::NotFinite);
        }
        let c = rump_c(&cost, &self.y);
        if !verify_psd(&cost, &self.y, c) {
            return Err(CertError::NotPsd);
        }
        Ok(self.y.iter().sum())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::GraphBuilder;
    use crate::ising::lattice2d;

    fn random_graph(n: usize, p: f64, seed: u64, fields: bool) -> Graph {
        let mut rng = Pcg::new(seed, 0xD0);
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

    fn brute_min(g: &Graph) -> f64 {
        (0..(1u32 << g.n))
            .map(|m| {
                let s: Vec<i8> = (0..g.n).map(|i| if m >> i & 1 == 1 { 1 } else { -1 }).collect();
                g.energy(&s)
            })
            .fold(f64::INFINITY, f64::min)
    }

    #[test]
    fn the_bound_never_exceeds_the_true_minimum() {
        // SOUNDNESS, which is the only property that matters. Checked with AND without fields, so
        // the homogenisation path is exercised rather than assumed.
        let p = Params { sweeps: 60, rank: None, lanczos: 24 };
        for seed in 0..60u64 {
            for fields in [false, true] {
                let g = random_graph(9, 0.45, seed, fields);
                let truth = brute_min(&g);
                let (b, _) = certified(&g, &p, seed);
                assert!(
                    b.value <= truth + 1e-9,
                    "seed {seed} fields={fields}: sdp gave {} above the true minimum {truth}",
                    b.value
                );
            }
        }
    }

    #[test]
    fn a_certificate_verifies_independently_of_how_it_was_found() {
        // The artefact a sceptic runs. `verify` rebuilds C from the graph and re-checks the one
        // load-bearing claim, touching nothing from the search that produced y.
        let g = random_graph(12, 0.4, 5, true);
        let (b, cert) = certified(&g, &Params::default(), 5);
        let v = cert.verify(&g).expect("its own certificate must re-verify");
        assert!((v - b.value).abs() < 1e-12, "verify gave {v}, bound said {}", b.value);
        assert!(v <= brute_min(&g) + 1e-9);
    }

    #[test]
    fn a_tampered_certificate_is_refused() {
        // Raising y makes the bound look better and breaks dual feasibility. If `verify` accepted
        // that, the certificate would prove nothing at all -- so this is the test that gives the
        // whole module its meaning.
        let g = random_graph(10, 0.5, 2, false);
        let (_, mut cert) = certified(&g, &Params::default(), 2);
        let honest = cert.verify(&g).unwrap();
        for a in 0..cert.y.len() {
            cert.y[a] += 1.0;
        }
        assert_eq!(cert.verify(&g), Err(CertError::NotPsd), "an inflated y must not verify");
        // And the shape check is real too.
        cert.y.push(0.0);
        assert!(matches!(cert.verify(&g), Err(CertError::Shape { .. })));
        assert!(honest.is_finite());
    }

    #[test]
    fn it_is_never_worse_than_the_trivial_floor() {
        // The Gershgorin dual point IS bound::decoupled, and it always verifies, so it is used as a
        // floor. Without that, an unlucky mixing run could report worse than the one-pass bound.
        for seed in 0..25u64 {
            let g = random_graph(11, 0.5, seed + 40, seed % 2 == 0);
            let (b, _) = certified(&g, &Params { sweeps: 5, rank: Some(2), lanczos: 8 }, seed);
            let dec = crate::bound::decoupled(&g);
            assert!(
                b.value >= dec.value - 1e-9,
                "seed {seed}: sdp {} fell below decoupled {}",
                b.value,
                dec.value
            );
        }
    }

    #[test]
    fn a_bad_search_loosens_the_bound_without_invalidating_it() {
        // The claim the module doc rests on: everything upstream of the psd check is a heuristic.
        // One sweep at rank 1 is a deliberately terrible choice of y, and it must still be SOUND.
        let p = Params { sweeps: 1, rank: Some(1), lanczos: 4 };
        for seed in 0..30u64 {
            let g = random_graph(10, 0.45, seed + 90, false);
            let truth = brute_min(&g);
            let (b, cert) = certified(&g, &p, seed);
            assert!(b.value <= truth + 1e-9, "seed {seed}: {} > {truth}", b.value);
            assert!(cert.verify(&g).is_ok(), "seed {seed}: certificate must still verify");
        }
    }

    #[test]
    fn the_verifier_rejects_a_matrix_that_is_not_definite() {
        // Directly: an all-zero y leaves S = C, which for any graph with a non-zero coupling has a
        // negative eigenvalue (its diagonal is zero and its trace is zero).
        let g = lattice2d(4, 1.0);
        let cost = Cost::build(&g);
        let zero = vec![0.0; cost.n];
        let c = rump_c(&cost, &zero);
        assert!(!verify_psd(&cost, &zero, c), "a zero-diagonal C with edges cannot be PSD");
        // And the Gershgorin floor, which is grown until it does. The textbook point is only
        // SEMI-definite (row sums exactly balanced), so this is the nudged one.
        let (yg, cg) = gershgorin_verified(&cost).expect("a dominant diagonal must verify");
        assert!(verify_psd(&cost, &yg, cg));
        // Nudging can only lower y, so the floor never claims more than `decoupled`.
        let dec = crate::bound::decoupled(&g).value;
        let got: f64 = yg.iter().sum();
        assert!(got <= dec + 1e-9, "the floor {got} must not exceed decoupled {dec}");
    }

    #[test]
    fn homogenisation_is_used_exactly_when_there_are_fields() {
        let no_h = random_graph(8, 0.5, 1, false);
        let with_h = random_graph(8, 0.5, 1, true);
        let (_, c1) = certified(&no_h, &Params { sweeps: 10, rank: Some(4), lanczos: 8 }, 1);
        let (_, c2) = certified(&with_h, &Params { sweeps: 10, rank: Some(4), lanczos: 8 }, 1);
        assert!(!c1.homogenised && c1.y.len() == no_h.n);
        assert!(c2.homogenised && c2.y.len() == with_h.n + 1, "a gauge spin is prepended");
    }

    #[test]
    fn an_empty_graph_returns_rather_than_panicking() {
        let g = GraphBuilder::new(0).build();
        let (b, c) = certified(&g, &Params::default(), 1);
        assert_eq!(b.value, 0.0);
        assert!(c.y.is_empty());
    }

    /// The dense gather the CSR one replaced. The reference, kept because the optimisation's whole
    /// claim is that it changed nothing.
    fn gather_dense(cost: &Cost, a: usize, v: &[f64], k: usize, g: &mut [f64]) {
        g.iter_mut().for_each(|x| *x = 0.0);
        let row = cost.row(a);
        for b in 0..cost.n {
            let c = row[b];
            if c != 0.0 {
                for t in 0..k {
                    g[t] += c * v[b * k + t];
                }
            }
        }
    }

    /// Skipping the zeros must reproduce the dense sweep **bit for bit**, not approximately.
    ///
    /// The sparse loop visits the same columns in the same ascending order, so every partial sum is
    /// identical and equality is the right assertion. `assert!((a-b).abs() < 1e-12)` would pass just
    /// as happily for a reordering that changed the arithmetic — and in a module that emits a
    /// certificate, "close enough" is the assertion that lets a real change through.
    #[test]
    fn the_sparse_gather_is_bit_identical_to_the_dense_one() {
        for seed in 0..6u64 {
            let g = random_graph(26, 0.35, seed, seed % 2 == 0);
            let cost = Cost::build(&g);
            let k = 5;
            let mut rng = Pcg::new(seed, 0xC5B);
            let v: Vec<f64> = (0..cost.n * k).map(|_| rng.f64() * 2.0 - 1.0).collect();
            let (mut sparse, mut dense) = (vec![0.0; k], vec![0.0; k]);
            for a in 0..cost.n {
                cost.gather(a, &v, k, &mut sparse);
                gather_dense(&cost, a, &v, k, &mut dense);
                for t in 0..k {
                    assert_eq!(
                        sparse[t].to_bits(),
                        dense[t].to_bits(),
                        "seed {seed} row {a} component {t}: sparse {:e}, dense {:e}",
                        sparse[t],
                        dense[t]
                    );
                }
            }
        }
    }

    /// Lanczos must return a **Ritz value of the matrix**, and this is the test that did not exist.
    ///
    /// The module documents Lanczos as "only a heuristic for choosing the shift", and that licence
    /// is real: the Cholesky is what makes the bound sound, so a bad `theta` costs tightness and
    /// nothing else. It is also exactly how this went unchecked. `jacobi_eig` returns the
    /// eigenVECTOR matrix and leaves the eigenVALUES on the diagonal of its input, so folding `min`
    /// over the return value produced the most negative eigenvector COMPONENT — always in `[-1, 0]`,
    /// never the spectrum. Every certificate still verified. The bound was simply looser than it
    /// needed to be, on every instance, for as long as the line existed.
    ///
    /// An untested heuristic does not stay a heuristic; it becomes a different function.
    #[test]
    fn the_lanczos_estimate_is_a_ritz_value_of_the_actual_matrix() {
        for seed in 0..5u64 {
            let g = random_graph(16, 0.4, seed, seed % 2 == 0);
            let cost = Cost::build(&g);
            let n = cost.n;
            // A `y` with spread-out entries, so `S` is not close to `C` and the diagonal genuinely
            // participates -- with `y = 0` a bug that ignored `y` would pass too.
            let y: Vec<f64> = (0..n).map(|a| -0.5 - 0.1 * a as f64).collect();

            let mut dense = vec![0.0f64; n * n];
            for a in 0..n {
                dense[a * n..a * n + n].copy_from_slice(cost.row(a));
                dense[a * n + a] = -y[a]; // S = C - Diag(y), and C has a zero diagonal
            }
            let _ = crate::linalg::jacobi_eig(&mut dense, n);
            let truth = (0..n).map(|i| dense[i * n + i]).fold(f64::INFINITY, f64::min);

            let est = lanczos_min(&cost, &y, n, seed);
            // Rayleigh-Ritz: a Ritz value is never below the true minimum.
            assert!(est >= truth - 1e-6, "seed {seed}: Ritz value {est} sits BELOW lambda_min {truth}");
            // And a full-length Krylov space reaches it. The old bug returned a number near -1
            // against a true minimum several times that, so this is the assertion that catches it.
            assert!(
                (est - truth).abs() <= 1e-3 * truth.abs().max(1.0),
                "seed {seed}: Lanczos says {est}, the dense eigensolver says {truth}"
            );
        }
    }

    /// The CSR index has to agree with the dense matrix it was read out of — every non-zero
    /// present, in ascending column order, and nothing else.
    #[test]
    fn the_sparse_index_enumerates_exactly_the_non_zeros() {
        let g = random_graph(20, 0.3, 7, true);
        let cost = Cost::build(&g);
        let mut counted = 0usize;
        for a in 0..cost.n {
            let (s, e) = (cost.nz_off[a], cost.nz_off[a + 1]);
            let cols = &cost.nz_col[s..e];
            assert!(cols.windows(2).all(|w| w[0] < w[1]), "row {a} is not strictly ascending");
            for (&b, &val) in cols.iter().zip(&cost.nz_val[s..e]) {
                assert_eq!(val.to_bits(), cost.row(a)[b as usize].to_bits());
                assert!(val != 0.0);
            }
            counted += e - s;
        }
        let dense_nz = cost.dense.iter().filter(|v| **v != 0.0).count();
        assert_eq!(counted, dense_nz, "the index missed a non-zero or invented one");
    }
}
