//! MBAR — the multistate Bennett acceptance ratio (Shirts & Chodera, J. Chem. Phys. 129:124105).
//!
//! [`crate::free_energy::bar_ladder`] walks a ladder one rung at a time: each step is a pairwise
//! Bennett estimate from the two rungs it joins, and the steps are summed. That throws information
//! away twice. A sample drawn at `beta_3` carries a perfectly good weight under `beta_7` whenever
//! the two overlap, and the pairwise chain never looks at it; and the summed steps are correlated,
//! because neighbouring steps share a rung's samples, so adding their error bars in quadrature
//! understates the result's spread.
//!
//! MBAR solves every state at once. With `N_k` samples drawn from state `k`, reduced potentials
//! `u_k(x)` (here `beta_k E(x)`), and dimensionless free energies `f_k = −ln Z_k`, the estimator is
//! the self-consistent solution of
//!
//! ```text
//!   exp(-f_i) = sum_n exp(-u_i(x_n)) / sum_k N_k exp(f_k - u_k(x_n)),     n over ALL samples.
//! ```
//!
//! Every sample appears in every equation, weighted by how well the mixture of all `K` states
//! explains it. Shirts & Chodera show this is the asymptotically minimum-variance estimator built
//! from those samples, and that at `K = 2` it reduces EXACTLY to Bennett's acceptance ratio —
//! [`crate::free_energy::bar_pair`] — which is the identity `two_states_reduce_to_bennett` checks.
//!
//! # How much it is worth, measured
//!
//! On the same samples, against the exact `ln Z`, MBAR's root-mean-square error is a few per cent
//! below the chained one — 0.967 of it on an even five-rung ladder, 0.921 when two rungs are
//! starved of samples. That is the honest size of the estimate's improvement, and it is small
//! because a ladder with good adjacent overlap is a case the chain already handles. The error BAR
//! is where the two part company: MBAR's covariance predicts its own spread to within 5%, the
//! quadrature of pairwise steps comes out 26% low on the even ladder and 62% low on the starved
//! one. The advantage is also asymptotic — at a handful of samples per state the joint solution's
//! bias can eat the pooling gain, and was measured to do so at eight draws per rung.
//!
//! # What is exact here, and what is estimated
//!
//! The solution is a fixed point, not a sample average, so it can be checked exactly. If every
//! state's samples have the empirical distribution `p_k` EXACTLY — take each configuration with
//! multiplicity proportional to `exp(-u_k(x))` — then substituting the true `f` satisfies the
//! equations term by term, and the solver must return the true free energies. That is a closed
//! oracle with no statistics in it, and `exact_empirical_distributions_are_solved_exactly` runs it
//! against [`crate::exact::Elimination::log_partition`] to `1e-9`.
//!
//! With real samples the answer is an estimate, and [`Mbar::theta`] carries the asymptotic
//! covariance of `f` (Shirts & Chodera eq. 12), from which the error on any difference follows:
//! `var(f_j - f_i) = Theta_ii + Theta_jj - 2 Theta_ij`.
//!
//! # Gauge
//!
//! Only differences are determined — shifting every `f_k` by a constant leaves the equations
//! unchanged — so the solution is reported with `f[0] = 0`. On a temperature ladder anchored at
//! `beta = 0`, [`ladder`] turns that into absolute `ln Z` using the one value known in closed form,
//! `ln Z(0) = n ln 2`.
//!
//! # Solver
//!
//! Self-consistent iteration converges linearly and can crawl; Newton–Raphson on the convex
//! objective converges quadratically but can overshoot far from the solution. Each iteration
//! computes both candidates and keeps the one with the smaller residual `max_k |sum_n W_nk - 1|`,
//! which is zero at the solution. The Hessian is singular by construction (the gauge direction),
//! so the Newton step goes through a pseudo-inverse; [`crate::linalg::jacobi_eig`] does both
//! symmetric decompositions this module needs.

use crate::linalg::jacobi_eig;

/// Default residual the solver must reach: `max_k |sum_n W_nk - 1|`, which is zero at the solution.
pub const TOL: f64 = 1e-12;

/// Default iteration cap. Newton converges quadratically once close, so this is slack, not a budget.
pub const MAX_ITERS: usize = 500;

/// Why a set of reduced potentials could not be solved.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Invalid {
    /// Fewer than two states. MBAR estimates differences, and one state has none.
    TooFewStates(usize),
    /// `counts` names a different number of states than the potential matrix has.
    Mismatched {
        /// Rows in the potential matrix.
        states: usize,
        /// Entries in `counts`.
        counts: usize,
    },
    /// This state contributed no samples, so nothing constrains its free energy.
    EmptyState(usize),
    /// A row of the potential matrix does not cover every sample.
    Ragged {
        /// The offending row.
        state: usize,
        /// Its length.
        len: usize,
        /// The total sample count every row must match.
        total: usize,
    },
    /// A reduced potential was not finite, at this state and sample.
    NotFinite {
        /// The state whose potential was not finite.
        state: usize,
        /// The sample it was evaluated on.
        sample: usize,
    },
    /// A warm start with the wrong number of entries.
    BadInit {
        /// Entries given.
        given: usize,
        /// Entries needed.
        want: usize,
    },
    /// A temperature ladder whose first rung is not `beta = 0`, where `ln Z` is known exactly.
    NotAnchored(f64),
}

impl core::fmt::Display for Invalid {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Invalid::TooFewStates(k) => write!(f, "MBAR needs at least two states, got {k}"),
            Invalid::Mismatched { states, counts } => {
                write!(f, "{states} rows of reduced potentials but {counts} sample counts")
            }
            Invalid::EmptyState(k) => write!(f, "state {k} contributed no samples"),
            Invalid::Ragged { state, len, total } => {
                write!(f, "row {state} covers {len} samples, not the {total} drawn")
            }
            Invalid::NotFinite { state, sample } => {
                write!(f, "reduced potential u[{state}][{sample}] is not finite")
            }
            Invalid::BadInit { given, want } => {
                write!(f, "warm start has {given} entries, needs {want}")
            }
            Invalid::NotAnchored(b) => {
                write!(f, "the ladder must start at beta = 0 where ln Z is exact, not {b}")
            }
        }
    }
}

/// The solved multistate estimator.
#[derive(Clone, Debug)]
pub struct Mbar {
    /// Number of thermodynamic states.
    pub states: usize,
    /// Samples drawn from each state, in the order the potential matrix's rows run.
    pub counts: Vec<usize>,
    /// Dimensionless free energies `f_k = -ln Z_k`, gauge-fixed so that `f[0] = 0`.
    pub f: Vec<f64>,
    /// Asymptotic covariance of `f`, row-major `states` by `states` (Shirts & Chodera eq. 12).
    ///
    /// Singular by construction: the gauge direction has zero variance because it was fixed.
    pub theta: Vec<f64>,
    /// Iterations taken before the residual met the tolerance or the cap was reached.
    pub iters: usize,
    /// The final residual `max_k |sum_n W_nk - 1|`.
    pub residual: f64,
    /// Whether `residual` reached the requested tolerance.
    pub converged: bool,
}

impl Mbar {
    /// `f_j - f_i`, which is `ln(Z_i / Z_j)`.
    ///
    /// # Panics
    ///
    /// If either index is past the last state.
    #[must_use]
    pub fn delta(&self, i: usize, j: usize) -> f64 {
        self.f[j] - self.f[i]
    }

    /// Standard error of `f_j - f_i`: `sqrt(Theta_ii + Theta_jj - 2 Theta_ij)`.
    ///
    /// # Panics
    ///
    /// If either index is past the last state.
    #[must_use]
    pub fn stderr_pair(&self, i: usize, j: usize) -> f64 {
        let k = self.states;
        assert!(i < k && j < k, "state index out of range");
        let v = self.theta[i * k + i] + self.theta[j * k + j] - 2.0 * self.theta[i * k + j];
        v.max(0.0).sqrt()
    }

    /// Standard error of every `f_k - f_0`; the gauge state's own entry is therefore zero.
    #[must_use]
    pub fn stderr(&self) -> Vec<f64> {
        (0..self.states).map(|k| self.stderr_pair(0, k)).collect()
    }

    /// Absolute `ln Z_k` given the anchor `ln Z_0`: `ln Z_k = ln Z_0 - f_k`.
    #[must_use]
    pub fn log_z(&self, log_z0: f64) -> Vec<f64> {
        self.f.iter().map(|fk| log_z0 - fk).collect()
    }
}

/// `ln Z` at every rung of a temperature ladder, with MBAR's own error bars.
#[derive(Clone, Debug)]
pub struct Profile {
    /// Inverse temperature of every rung, in the order given.
    pub beta: Vec<f64>,
    /// `ln Z(beta)` at each rung, anchored at the exact `ln Z(0) = n ln 2`.
    pub log_z: Vec<f64>,
    /// Standard error of `ln Z(beta) - ln Z(0)`; the anchor is exact, so its entry is zero.
    pub stderr: Vec<f64>,
    /// The solved estimator the profile came from.
    pub mbar: Mbar,
}

/// `ln Z` at every rung by MBAR, from the energies of samples drawn at each.
///
/// Same input as [`crate::free_energy::bar_ladder`] — `(beta, energies)` per rung, anchored at
/// `beta = 0` where the zero rung's samples must be uniform draws — so the two estimators can be
/// run on the SAME samples and compared.
///
/// # Errors
///
/// [`Invalid::NotAnchored`] unless the first rung is `beta = 0`, plus everything [`solve`] refuses.
pub fn ladder(
    n: usize,
    traces: &[(f64, Vec<f64>)],
    tol: f64,
    max_iters: usize,
) -> Result<Profile, Invalid> {
    if traces.len() < 2 {
        return Err(Invalid::TooFewStates(traces.len()));
    }
    if traces[0].0 != 0.0 {
        return Err(Invalid::NotAnchored(traces[0].0));
    }
    let mbar = from_traces(traces, tol, max_iters)?;
    let anchor = n as f64 * core::f64::consts::LN_2;
    Ok(Profile {
        beta: traces.iter().map(|(b, _)| *b).collect(),
        log_z: mbar.log_z(anchor),
        stderr: mbar.stderr(),
        mbar,
    })
}

/// MBAR on a temperature ladder: `u_k(x) = beta_k E(x)`, so the energies are all that is needed.
///
/// # Errors
///
/// Everything [`solve`] refuses.
pub fn from_traces(traces: &[(f64, Vec<f64>)], tol: f64, max_iters: usize) -> Result<Mbar, Invalid> {
    if traces.len() < 2 {
        return Err(Invalid::TooFewStates(traces.len()));
    }
    let counts: Vec<usize> = traces.iter().map(|(_, e)| e.len()).collect();
    let all: Vec<f64> = traces.iter().flat_map(|(_, e)| e.iter().copied()).collect();
    let u: Vec<Vec<f64>> =
        traces.iter().map(|(b, _)| all.iter().map(|e| b * e).collect()).collect();
    solve(&u, &counts, None, tol, max_iters)
}

/// Solve the MBAR equations for arbitrary reduced potentials.
///
/// `u[k][n]` is the reduced potential of sample `n` evaluated in state `k`, over the samples of
/// every state concatenated in state order; `counts[k]` is how many of them state `k` drew.
/// `init` is an optional warm start for `f`.
///
/// # Errors
///
/// [`Invalid`] for fewer than two states, a state with no samples, a row that does not cover every
/// sample, a non-finite potential, a mismatched `counts`, or a wrongly sized warm start.
pub fn solve(
    u: &[Vec<f64>],
    counts: &[usize],
    init: Option<&[f64]>,
    tol: f64,
    max_iters: usize,
) -> Result<Mbar, Invalid> {
    let k = u.len();
    if k < 2 {
        return Err(Invalid::TooFewStates(k));
    }
    if counts.len() != k {
        return Err(Invalid::Mismatched { states: k, counts: counts.len() });
    }
    let total: usize = counts.iter().sum();
    for (i, &c) in counts.iter().enumerate() {
        if c == 0 {
            return Err(Invalid::EmptyState(i));
        }
    }
    for (i, row) in u.iter().enumerate() {
        if row.len() != total {
            return Err(Invalid::Ragged { state: i, len: row.len(), total });
        }
        for (n, v) in row.iter().enumerate() {
            if !v.is_finite() {
                return Err(Invalid::NotFinite { state: i, sample: n });
            }
        }
    }
    let mut f = match init {
        Some(v) if v.len() != k => return Err(Invalid::BadInit { given: v.len(), want: k }),
        Some(v) => v.to_vec(),
        None => vec![0.0; k],
    };
    let shift = f[0];
    for v in &mut f {
        *v -= shift;
    }

    let ln_n: Vec<f64> = counts.iter().map(|&c| (c as f64).ln()).collect();
    let mut denom = vec![0.0; total];
    let mut scratch = vec![0.0; total];
    let mut g = vec![0.0; k];
    let mut gg = vec![0.0; k * k];
    let mut iters = 0;
    let residual;
    loop {
        log_denominators(u, &ln_n, &f, &mut denom);
        weight_moments(u, &f, &denom, &mut g, &mut gg);
        let r = g.iter().map(|x| (x - 1.0).abs()).fold(0.0f64, f64::max);
        if r <= tol || iters >= max_iters {
            residual = r;
            break;
        }
        let mut sci = vec![0.0; k];
        sci_step(u, &denom, &mut sci);
        let r_sci = residual_at(u, &ln_n, &sci, &mut scratch);
        f = match newton_step(counts, &g, &gg, &f, total as f64) {
            Some(nw) if residual_at(u, &ln_n, &nw, &mut scratch) < r_sci => nw,
            _ => sci,
        };
        iters += 1;
    }

    let theta = covariance(&gg, counts, k);
    Ok(Mbar {
        states: k,
        counts: counts.to_vec(),
        f,
        theta,
        iters,
        residual,
        converged: residual <= tol,
    })
}

// ---- the iteration -----------------------------------------------------------------------------

/// `ln sum_k N_k exp(f_k - u_k(x_n))` for every sample, the denominator every weight shares.
fn log_denominators(u: &[Vec<f64>], ln_n: &[f64], f: &[f64], out: &mut [f64]) {
    let k = u.len();
    for n in 0..out.len() {
        let mut mx = f64::NEG_INFINITY;
        for j in 0..k {
            let t = ln_n[j] + f[j] - u[j][n];
            if t > mx {
                mx = t;
            }
        }
        let mut acc = 0.0;
        for j in 0..k {
            acc += (ln_n[j] + f[j] - u[j][n] - mx).exp();
        }
        out[n] = mx + acc.ln();
    }
}

/// `g_k = sum_n W_nk` and `G = W^T W`, the two moments of the weight matrix everything downstream
/// needs — accumulated one sample at a time, so `W` itself is never materialised.
fn weight_moments(u: &[Vec<f64>], f: &[f64], denom: &[f64], g: &mut [f64], gg: &mut [f64]) {
    let k = u.len();
    g.fill(0.0);
    gg.fill(0.0);
    let mut w = vec![0.0; k];
    for n in 0..denom.len() {
        for j in 0..k {
            w[j] = (f[j] - u[j][n] - denom[n]).exp();
            g[j] += w[j];
        }
        for a in 0..k {
            for b in 0..k {
                gg[a * k + b] += w[a] * w[b];
            }
        }
    }
}

/// One self-consistent update: `f_i = -ln sum_n exp(-u_i(x_n) - ln D_n)`, re-gauged to `f[0] = 0`.
fn sci_step(u: &[Vec<f64>], denom: &[f64], out: &mut [f64]) {
    let k = u.len();
    for i in 0..k {
        let mut mx = f64::NEG_INFINITY;
        for n in 0..denom.len() {
            let t = -u[i][n] - denom[n];
            if t > mx {
                mx = t;
            }
        }
        let mut acc = 0.0;
        for n in 0..denom.len() {
            acc += (-u[i][n] - denom[n] - mx).exp();
        }
        out[i] = -(mx + acc.ln());
    }
    let shift = out[0];
    for v in &mut *out {
        *v -= shift;
    }
}

/// One Newton step on the convex MBAR objective, through the Hessian's pseudo-inverse.
///
/// Gradient `grad_i = (N_i/N)(g_i - 1)`, Hessian `H_ij = (1/N)(delta_ij N_i g_i - N_i N_j G_ij)`.
/// `H` annihilates the all-ones vector exactly — that is the gauge — so the step is taken in the
/// complement of that null space. `None` if the Hessian is degenerate or the step is not finite.
fn newton_step(counts: &[usize], g: &[f64], gg: &[f64], f: &[f64], total: f64) -> Option<Vec<f64>> {
    let k = g.len();
    let mut h = vec![0.0; k * k];
    let mut grad = vec![0.0; k];
    for i in 0..k {
        let ni = counts[i] as f64;
        grad[i] = ni * (g[i] - 1.0) / total;
        for j in 0..k {
            let nj = counts[j] as f64;
            let diag = if i == j { ni * g[i] } else { 0.0 };
            h[i * k + j] = (diag - ni * nj * gg[i * k + j]) / total;
        }
    }
    let v = jacobi_eig(&mut h, k);
    let lam: Vec<f64> = (0..k).map(|c| h[c * k + c]).collect();
    let scale = lam.iter().fold(0.0f64, |a, &l| a.max(l.abs()));
    if !(scale > 0.0) {
        return None;
    }
    let cut = 1e-10 * scale;
    let mut out = f.to_vec();
    for c in 0..k {
        if lam[c].abs() <= cut {
            continue;
        }
        let dot: f64 = (0..k).map(|i| v[i * k + c] * grad[i]).sum();
        let coef = dot / lam[c];
        for i in 0..k {
            out[i] -= coef * v[i * k + c];
        }
    }
    if !out.iter().all(|x| x.is_finite()) {
        return None;
    }
    let shift = out[0];
    for x in &mut out {
        *x -= shift;
    }
    Some(out)
}

/// `max_k |sum_n W_nk - 1|` at a candidate `f`. Zero exactly at the solution, so it is the
/// scale-free thing to compare two candidates by.
fn residual_at(u: &[Vec<f64>], ln_n: &[f64], f: &[f64], denom: &mut [f64]) -> f64 {
    log_denominators(u, ln_n, f, denom);
    let k = u.len();
    let mut worst = 0.0f64;
    for i in 0..k {
        let mut s = 0.0;
        for n in 0..denom.len() {
            s += (f[i] - u[i][n] - denom[n]).exp();
        }
        worst = worst.max((s - 1.0).abs());
    }
    worst
}

// ---- asymptotic covariance ---------------------------------------------------------------------

/// `Theta = W^T (I_N - W N W^T)^+ W`, the covariance of `f` (Shirts & Chodera appendix D).
///
/// Computed in `K` by `K` rather than `N` by `N`. With `W = U S V^T` the `N`-sized projector splits
/// and `Theta = V S (I_K - S V^T N V S)^+ S V^T`, where `V` and `S` come from the
/// eigendecomposition of `G = W^T W` — which the iteration already accumulated. The pseudo-inverse
/// is what handles the exact null direction: `W N W^T` has eigenvalue one on the all-ones vector,
/// because `sum_k N_k W_nk = 1` for every sample.
fn covariance(gg: &[f64], counts: &[usize], k: usize) -> Vec<f64> {
    let mut a = gg.to_vec();
    let v = jacobi_eig(&mut a, k);
    let s: Vec<f64> = (0..k).map(|c| a[c * k + c].max(0.0).sqrt()).collect();

    // b = I - S V^T N V S, whose pseudo-inverse is the middle factor.
    let mut b = vec![0.0; k * k];
    for c in 0..k {
        for d in 0..k {
            let mut acc = 0.0;
            for j in 0..k {
                acc += v[j * k + c] * counts[j] as f64 * v[j * k + d];
            }
            b[c * k + d] = if c == d { 1.0 } else { 0.0 } - s[c] * s[d] * acc;
        }
    }
    let q = jacobi_eig(&mut b, k);
    let lam: Vec<f64> = (0..k).map(|c| b[c * k + c]).collect();
    let scale = lam.iter().fold(0.0f64, |acc, &l| acc.max(l.abs())).max(f64::MIN_POSITIVE);
    let cut = 1e-10 * scale;
    let mut pinv = vec![0.0; k * k];
    for c in 0..k {
        for d in 0..k {
            let mut acc = 0.0;
            for e in 0..k {
                if lam[e].abs() > cut {
                    acc += q[c * k + e] * q[d * k + e] / lam[e];
                }
            }
            pinv[c * k + d] = acc;
        }
    }

    // Theta = V S pinv S V^T.
    let mut theta = vec![0.0; k * k];
    for i in 0..k {
        for j in 0..k {
            let mut acc = 0.0;
            for c in 0..k {
                if s[c] == 0.0 {
                    continue;
                }
                for d in 0..k {
                    acc += v[i * k + c] * s[c] * pinv[c * k + d] * s[d] * v[j * k + d];
                }
            }
            theta[i * k + j] = acc;
        }
    }
    theta
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exact::Elimination;
    use crate::free_energy::{bar_ladder, bar_pair, exact_log_z};
    use crate::graph::{Graph, GraphBuilder};
    use crate::ising;
    use crate::rng::Pcg;

    const LN2: f64 = core::f64::consts::LN_2;

    fn glass6() -> Graph {
        let mut b = GraphBuilder::new(6);
        let j = [1.0, -1.0, 1.0, 1.0, -1.0, 1.0];
        for i in 0..6 {
            b.couple(i, (i + 1) % 6, j[i]);
        }
        b.build()
    }

    fn ring_with_field(n: usize) -> Graph {
        let mut b = GraphBuilder::new(n);
        for i in 0..n {
            b.couple(i, (i + 1) % n, 1.0);
            b.bias(i, 1.0);
        }
        b.build()
    }

    fn energies_of(g: &Graph) -> Vec<f64> {
        (0..1usize << g.n)
            .map(|mask| {
                let s: Vec<i8> = (0..g.n).map(|b| if mask >> b & 1 == 1 { 1 } else { -1 }).collect();
                g.energy(&s)
            })
            .collect()
    }

    /// The sample list whose empirical distribution IS the Boltzmann distribution: configuration
    /// `x` appears `exp(beta (E_max - E(x)))` times. Integral only for a beta whose weights are
    /// rational, which is asserted rather than assumed.
    fn exact_multiset(g: &Graph, beta: f64) -> Vec<f64> {
        let es = energies_of(g);
        let emax = es.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let mut out = Vec::new();
        for &e in &es {
            let w = (beta * (emax - e)).exp();
            let m = w.round();
            assert!(
                (w - m).abs() < 1e-6 * m.max(1.0),
                "multiplicity {w} is not an integer -- pick a beta with rational Boltzmann weights"
            );
            for _ in 0..(m as usize) {
                out.push(e);
            }
        }
        out
    }

    /// Independent exact draws from the Boltzmann distribution by inverse CDF over the enumerated
    /// states. No Markov chain, so what the tests below measure is the ESTIMATOR's variance and not
    /// a sampler's autocorrelation.
    fn iid_energies(g: &Graph, beta: f64, m: usize, rng: &mut Pcg) -> Vec<f64> {
        let es = energies_of(g);
        let p = ising::exact_boltzmann(g, beta);
        let mut cdf = Vec::with_capacity(p.len());
        let mut acc = 0.0;
        for &v in &p {
            acc += v;
            cdf.push(acc);
        }
        (0..m)
            .map(|_| {
                let x = rng.f64();
                let i = cdf.partition_point(|&c| c < x).min(es.len() - 1);
                es[i]
            })
            .collect()
    }

    fn iid_traces(g: &Graph, betas: &[f64], m: usize, seed: u64) -> Vec<(f64, Vec<f64>)> {
        betas
            .iter()
            .enumerate()
            .map(|(k, &b)| {
                let mut rng = Pcg::new(seed, k as u64);
                (b, iid_energies(g, b, m, &mut rng))
            })
            .collect()
    }

    /// THE oracle: with the empirical distributions exact, MBAR's fixed point is the exact free
    /// energy, term by term, with no statistics anywhere in the claim. Checked against exact
    /// variable elimination on three models.
    #[test]
    fn exact_empirical_distributions_are_solved_exactly() {
        let el = Elimination::default();
        let cases: [(Graph, Vec<f64>); 3] = [
            (ising::ring(4, 1.0, 0.0), vec![0.0, 0.5 * LN2, LN2]),
            (ring_with_field(4), vec![0.0, 0.5 * LN2]),
            (glass6(), vec![0.0, 0.5 * LN2]),
        ];
        for (g, betas) in &cases {
            let traces: Vec<(f64, Vec<f64>)> =
                betas.iter().map(|&b| (b, exact_multiset(g, b))).collect();
            let p = ladder(g.n, &traces, 1e-13, MAX_ITERS).unwrap();
            assert!(p.mbar.converged, "residual {} after {} iters", p.mbar.residual, p.mbar.iters);
            for (i, &b) in betas.iter().enumerate() {
                let want = el.log_partition(g, b).unwrap().log_z.unwrap();
                assert!(
                    (p.log_z[i] - want).abs() < 1e-9,
                    "n={} beta={b}: ln Z {} vs exact {want}",
                    g.n,
                    p.log_z[i]
                );
            }
        }
    }

    /// Shirts & Chodera's own reduction: at two states MBAR IS Bennett. Not a similar answer to
    /// two decimal places -- the same root of the same equation, so it holds on any samples at all.
    ///
    /// The error bars must agree too, and that is a real check on the covariance code rather than a
    /// restatement: [`crate::free_energy::bar_pair`] gets its bar from Bennett's own two-sample
    /// formula, this module gets the same number out of a `K` by `K` pseudo-inverse. They land
    /// within 4%, the gap being that Bennett's divides by `N / 2 tau_int` where the asymptotic form
    /// divides by `N`, and `tau_int` of an independent trace is only near a half, not exactly one.
    #[test]
    fn two_states_reduce_to_bennett() {
        let g = glass6();
        for (ba, bb) in [(0.0, 0.4), (0.2, 0.9), (0.5, 0.51)] {
            let traces = iid_traces(&g, &[ba, bb], 400, 7);
            let m = from_traces(&traces, TOL, MAX_ITERS).unwrap();
            let b = bar_pair(ba, &traces[0].1, bb, &traces[1].1);
            // BAR reports ln(Z_b/Z_a); MBAR reports f_1 - f_0 = -ln(Z_1/Z_0).
            assert!(
                (b.delta + m.delta(0, 1)).abs() < 1e-8,
                "BAR {} vs MBAR {}",
                b.delta,
                -m.delta(0, 1)
            );
            let r = m.stderr_pair(0, 1) / b.stderr;
            assert!((r - 1.0).abs() < 0.05, "stderr {} vs Bennett's {}", m.stderr_pair(0, 1), b.stderr);
        }
    }

    /// Adding a constant to ONE state's potential must move only that state's free energy, by
    /// exactly that constant. An algebraic identity of the fixed point, checked to `1e-10`.
    #[test]
    fn shifting_one_potential_shifts_only_that_free_energy() {
        let mut rng = Pcg::new(11, 3);
        let counts = [7usize, 11, 5];
        let total: usize = counts.iter().sum();
        let u: Vec<Vec<f64>> =
            (0..3).map(|_| (0..total).map(|_| 4.0 * rng.f64() - 2.0).collect()).collect();
        let base = solve(&u, &counts, None, TOL, MAX_ITERS).unwrap();

        let a = 0.75;
        let mut shifted = u.clone();
        for v in &mut shifted[1] {
            *v += a;
        }
        let got = solve(&shifted, &counts, None, TOL, MAX_ITERS).unwrap();
        assert!((got.f[0] - base.f[0]).abs() < 1e-10);
        assert!((got.f[1] - (base.f[1] + a)).abs() < 1e-10, "{} vs {}", got.f[1], base.f[1] + a);
        assert!((got.f[2] - base.f[2]).abs() < 1e-10);
    }

    /// Adding a per-SAMPLE constant to every state's potential must change nothing: it multiplies
    /// numerator and denominator of each weight by the same factor.
    #[test]
    fn a_per_sample_shift_of_every_potential_changes_nothing() {
        let mut rng = Pcg::new(29, 1);
        let counts = [9usize, 4, 6, 3];
        let total: usize = counts.iter().sum();
        let u: Vec<Vec<f64>> =
            (0..4).map(|_| (0..total).map(|_| 3.0 * rng.f64() - 1.5).collect()).collect();
        let base = solve(&u, &counts, None, TOL, MAX_ITERS).unwrap();

        let c: Vec<f64> = (0..total).map(|_| 5.0 * rng.f64() - 2.5).collect();
        let mut shifted = u.clone();
        for row in &mut shifted {
            for (n, v) in row.iter_mut().enumerate() {
                *v += c[n];
            }
        }
        let got = solve(&shifted, &counts, None, TOL, MAX_ITERS).unwrap();
        for i in 0..4 {
            assert!(
                (got.f[i] - base.f[i]).abs() < 1e-10,
                "state {i}: {} vs {}",
                got.f[i],
                base.f[i]
            );
        }
    }

    /// The estimator does not depend on the order the states are listed in, and a deliberately bad
    /// warm start reaches the same fixed point.
    #[test]
    fn the_solution_is_independent_of_state_order_and_warm_start() {
        let g = glass6();
        let betas = [0.0, 0.3, 0.7];
        let traces = iid_traces(&g, &betas, 300, 5);
        let straight = from_traces(&traces, TOL, MAX_ITERS).unwrap();

        let mut flipped: Vec<(f64, Vec<f64>)> = traces.clone();
        flipped.reverse();
        let rev = from_traces(&flipped, TOL, MAX_ITERS).unwrap();
        for i in 0..3 {
            let want = straight.delta(0, i);
            let got = rev.delta(2, 2 - i);
            assert!((got - want).abs() < 1e-9, "order changed f: {got} vs {want}");
        }

        let all: Vec<f64> = traces.iter().flat_map(|(_, e)| e.iter().copied()).collect();
        let u: Vec<Vec<f64>> =
            betas.iter().map(|b| all.iter().map(|e| b * e).collect()).collect();
        let far = solve(&u, &[300, 300, 300], Some(&[0.0, -40.0, 90.0]), TOL, MAX_ITERS).unwrap();
        for i in 0..3 {
            assert!((far.f[i] - straight.f[i]).abs() < 1e-9, "warm start moved f[{i}]");
        }
    }

    /// The claim this module exists for: on the SAME samples, pooling every state beats chaining
    /// adjacent pairs, and MBAR's error bar knows how big its own error is.
    ///
    /// Root-mean-square error against the exact `ln Z` at the cold end, pooled over three models
    /// and 200 independent seeds each, in two ladder shapes: an even one, and one where two rungs
    /// are starved of samples so the chain must cross a weak link the pooled solution routes
    /// around.
    ///
    /// The margin is a few per cent, not an order of magnitude, and that is its honest size on a
    /// well-overlapping ladder: the ratio is nearest one exactly where every adjacent pair already
    /// overlaps well, and widest where a rung is starved. What is NOT a few per cent is the error
    /// bar. MBAR's joint covariance predicts its own spread to within a few per cent; summing the
    /// pairwise steps in quadrature ignores that adjacent steps share a rung's samples, and comes
    /// out ~20% low on the even ladder and about half on the starved one — which is the optimism
    /// [`crate::free_energy::ThermoRung::stderr`] documents about itself.
    #[test]
    fn mbar_beats_chained_bar_on_the_same_samples() {
        let models = [glass6(), ising::ring(6, 1.0, 0.3), ring_with_field(6)];
        let even: Vec<f64> = (0..5).map(|k| 0.2 * k as f64).collect();
        let fine: Vec<f64> = (0..9).map(|k| 0.1 * k as f64).collect();
        let shapes: [(&str, &[f64], Vec<usize>); 2] = [
            ("even", &even, vec![100; 5]),
            ("starved", &fine, vec![60, 60, 5, 60, 5, 60, 60, 60, 60]),
        ];
        for (name, betas, ns) in &shapes {
            let seeds = 200;
            let (mut sm, mut sb, mut pm, mut pb) = (0.0, 0.0, 0.0, 0.0);
            let mut runs = 0.0;
            for g in &models {
                let truth = exact_log_z(g, *betas.last().unwrap());
                for seed in 0..seeds {
                    let traces: Vec<(f64, Vec<f64>)> = betas
                        .iter()
                        .zip(ns.iter())
                        .enumerate()
                        .map(|(k, (&b, &m))| {
                            let mut rng = Pcg::new(7000 + seed as u64, k as u64);
                            (b, iid_energies(g, b, m, &mut rng))
                        })
                        .collect();
                    let mb = ladder(g.n, &traces, TOL, MAX_ITERS).unwrap();
                    let br = bar_ladder(g.n, &traces, 1.96);
                    let em = mb.log_z[betas.len() - 1] - truth;
                    let eb = br.top().log_z - truth;
                    sm += em * em;
                    sb += eb * eb;
                    pm += *mb.stderr.last().unwrap();
                    pb += br.top().stderr;
                    runs += 1.0;
                }
            }
            let (rm, rb) = ((sm / runs).sqrt(), (sb / runs).sqrt());
            let (cm, cb) = (pm / runs / rm, pb / runs / rb);
            println!(
                "{name:8} rmse MBAR {rm:.5} chained BAR {rb:.5} ratio {:.3} | \
                 error bar over true spread: MBAR {cm:.3} BAR {cb:.3}",
                rm / rb
            );
            assert!(rm < rb, "{name}: MBAR rmse {rm} did not beat chained BAR {rb}");
            assert!(
                (cm - 1.0).abs() < 0.15,
                "{name}: MBAR's covariance predicted {cm} of its own spread"
            );
            assert!(
                (cm - 1.0).abs() < (cb - 1.0).abs(),
                "{name}: the joint covariance ({cm}) was no better calibrated than the quadrature \
                 of pairwise steps ({cb})"
            );
        }
    }

    #[test]
    fn refusals() {
        let err = |r: Result<Mbar, Invalid>| r.err().unwrap();
        assert_eq!(err(solve(&[vec![0.0]], &[1], None, TOL, 10)), Invalid::TooFewStates(1));
        let u = vec![vec![0.0; 4], vec![1.0; 4]];
        assert_eq!(
            err(solve(&u, &[2, 1, 1], None, TOL, 10)),
            Invalid::Mismatched { states: 2, counts: 3 }
        );
        assert_eq!(err(solve(&u, &[4, 0], None, TOL, 10)), Invalid::EmptyState(1));
        assert_eq!(
            err(solve(&[vec![0.0; 4], vec![1.0; 3]], &[2, 2], None, TOL, 10)),
            Invalid::Ragged { state: 1, len: 3, total: 4 }
        );
        let mut bad = u.clone();
        bad[1][2] = f64::NAN;
        assert_eq!(
            err(solve(&bad, &[2, 2], None, TOL, 10)),
            Invalid::NotFinite { state: 1, sample: 2 }
        );
        assert_eq!(
            err(solve(&u, &[2, 2], Some(&[0.0]), TOL, 10)),
            Invalid::BadInit { given: 1, want: 2 }
        );
        assert_eq!(
            ladder(2, &[(0.5, vec![0.0; 2]), (1.0, vec![0.0; 2])], TOL, 10).err().unwrap(),
            Invalid::NotAnchored(0.5)
        );
    }
}
