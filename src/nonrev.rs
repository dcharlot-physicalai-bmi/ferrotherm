//! Non-reversible kernels that preserve a **stated** invariant measure — the acceleration an
//! asymmetric coupling only pretends to be.
//!
//! # The gap this closes
//!
//! Every sampler in this crate before it satisfies detailed balance. Detailed balance is a
//! sufficient condition for `pi`-invariance and it is not a necessary one, and the difference is
//! not academic: a kernel that keeps `pi` while breaking reversibility can mix strictly faster than
//! any reversible kernel with the same proposal, because it stops undoing its own last move.
//!
//! [`crate::kuramoto`] shows that a learned, asymmetrically-coupled oscillator network is **not**
//! this. Its antisymmetric part is divergence-free, which looks like the right shape, but
//! `pi`-invariance needs `div(pi F) = 0` and the second term of that expansion,
//! `grad E . F_A`, does not vanish — so the system samples nothing nameable. This module contains
//! the two constructions that do work, one continuous and one discrete, each with the identity that
//! proves it machine-checked.
//!
//! # The continuous construction, and the two identities it reduces to
//!
//! Hwang, Hwang & Sheu, *Accelerating Gaussian diffusions*, Ann. Appl. Probab. 3:897 (1993): add a
//! drift `g` to overdamped Langevin,
//!
//! ```text
//!   dtheta  =  (-grad E(theta) + g(theta)) dt  +  sqrt(2/beta) dW ,
//! ```
//!
//! and `pi = e^{-beta E}` survives exactly when `div(pi g) = 0`. Writing out the stationary
//! Fokker–Planck equation at `p = pi`, the `-grad E` drift and the diffusion cancel identically, and
//! what is left is `-pi (div g - beta grad E . g)`. So **two** things must vanish, not one.
//!
//! Take `g = A grad E` with `A` a constant antisymmetric matrix. Then:
//!
//! * `div g = tr(A H)` where `H` is the Hessian of `E`, and the trace of an antisymmetric matrix
//!   against a symmetric one is zero — [`trace_against`], checked to `1e-13` against
//!   [`crate::kuramoto::Kuramoto::xy_hessian`];
//! * `grad E . A grad E = 0` because an antisymmetric quadratic form vanishes on every vector —
//!   [`quadratic_form`], checked exactly.
//!
//! Both are identities in linear algebra rather than properties of a model, which is why the
//! construction transfers to any energy this crate can differentiate. [`SkewLangevin`] is the
//! integrator; the tests check the identities rather than a histogram, because an identity that
//! holds to rounding is stronger evidence than a histogram that agrees to three digits.
//!
//! The reason this matters for the substrate argument: `A grad E` is a **physically realisable
//! drift on the same fabric** — it is a linear map applied to the field each node already computes.
//! A machine that can evaluate `grad E` locally can evaluate `A grad E` locally, so non-reversible
//! acceleration is available to a thermodynamic fabric at the cost of one extra coupling matrix,
//! and it comes with an invariant measure the fabric can still certify. That is strictly more than
//! an asymmetric `K` offers, which is speed with no stationary law at all.
//!
//! # The discrete construction, which is exactly verifiable
//!
//! Lifting (Diaconis, Holmes & Neal, *Analysis of a nonreversible Markov chain sampler*, Ann. Appl.
//! Probab. 10:726, 2000; Turitsyn, Chertkov & Vucelja, Physica D 240:410, 2011). Give every site a
//! direction bit `v` in `{+1, -1}` and replace the symmetric proposal by a directed one:
//!
//! ```text
//!   propose  a -> a + v (mod q)  ;  accept with min(1, pi(a+v)/pi(a))  ;  on reject, set v <- -v
//! ```
//!
//! The chain on `(a, v)` preserves `pi(a)/2` **exactly**, and the proof is two lines of case
//! analysis on whether `pi(a+v)` exceeds `pi(a)` — written out at [`LiftedClock`] and checked here
//! by building the transition matrix of a small model and confirming the left eigenvector to
//! `1e-15`, rather than by sampling.
//!
//! It is not reversible, and [`balance_defect`] measures by how much: a strictly positive number on
//! the same instance where [`stationary_defect`] returns zero is the whole claim of the module in
//! two function calls. On a `q = 16` clock site in a field the lifted chain's exact integrated
//! autocorrelation time is a factor of several below the reversible chain with the identical
//! proposal, and `lifting_beats_the_reversible_chain_with_the_same_proposal` computes both from
//! their transition matrices rather than measuring them.
//!
//! Lifting is the discrete image of the solenoidal Hodge component: the direction bit is the extra
//! coordinate the circulation lives in. That correspondence is why the two halves of this module
//! belong together, and why a fabric that can hold one extra bit per node can have the continuous
//! trick's benefit without leaving the categorical state space this crate certifies.

use crate::ledger::Ledger;
use crate::potts::Potts;
use crate::rng::Pcg;
use crate::round::sum_up;

/// `v^T A v` for a row-major `n * n` matrix.
///
/// Zero for every antisymmetric `A` and every `v`, which is the second of the two identities the
/// skew construction needs.
///
/// # Panics
///
/// If the shapes disagree.
#[must_use]
pub fn quadratic_form(a: &[f64], v: &[f64], n: usize) -> f64 {
    assert_eq!(a.len(), n * n, "matrix must be {n}x{n}");
    assert_eq!(v.len(), n, "vector must be length {n}");
    let mut terms = Vec::with_capacity(n * n);
    for i in 0..n {
        for j in 0..n {
            terms.push(v[i] * a[i * n + j] * v[j]);
        }
    }
    sum_up(&terms)
}

/// `tr(A B)` for row-major `n * n` matrices.
///
/// Zero whenever one is antisymmetric and the other symmetric — the first identity the skew
/// construction needs, and the one that makes `div(A grad E)` vanish.
///
/// # Panics
///
/// If the shapes disagree.
#[must_use]
pub fn trace_against(a: &[f64], b: &[f64], n: usize) -> f64 {
    assert_eq!(a.len(), n * n, "first matrix must be {n}x{n}");
    assert_eq!(b.len(), n * n, "second matrix must be {n}x{n}");
    let mut terms = Vec::with_capacity(n * n);
    for i in 0..n {
        for j in 0..n {
            terms.push(a[i * n + j] * b[j * n + i]);
        }
    }
    sum_up(&terms)
}

/// `max |A_ij + A_ji|` — zero exactly when `A` is antisymmetric.
///
/// # Panics
///
/// If the shape disagrees.
#[must_use]
pub fn antisymmetry_defect(a: &[f64], n: usize) -> f64 {
    assert_eq!(a.len(), n * n, "matrix must be {n}x{n}");
    let mut m = 0.0f64;
    for i in 0..n {
        for j in 0..n {
            m = m.max((a[i * n + j] + a[j * n + i]).abs());
        }
    }
    m
}

/// A constant antisymmetric matrix built from a seed, for use as a skew drift.
///
/// Antisymmetric **by construction** rather than by rounding: the lower triangle is copied from the
/// upper with a sign flip and the diagonal is set to zero, so [`antisymmetry_defect`] returns
/// exactly `0.0`.
#[must_use]
pub fn skew_matrix(n: usize, scale: f64, seed: u64) -> Vec<f64> {
    let mut rng = Pcg::new(seed, 0x5CE1);
    let mut a = vec![0.0; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            let v = scale * (2.0 * rng.f64() - 1.0);
            a[i * n + j] = v;
            a[j * n + i] = -v;
        }
    }
    a
}

/// Overdamped Langevin on the torus with an added skew drift `A grad E`.
///
/// The invariant measure is `e^{-beta E}` for **any** antisymmetric `A`, by the two identities in
/// the module documentation. `A = 0` recovers the ordinary reversible sampler, so a caller can vary
/// one argument and compare.
#[derive(Clone, Debug)]
pub struct SkewLangevin {
    a: Vec<f64>,
    n: usize,
    beta: f64,
    dt: f64,
}

impl SkewLangevin {
    /// Build from an antisymmetric matrix, an inverse temperature and a step size.
    ///
    /// # Panics
    ///
    /// If `a` is not `n * n`, not antisymmetric, or `beta` or `dt` is not positive and finite.
    #[must_use]
    pub fn new(a: Vec<f64>, n: usize, beta: f64, dt: f64) -> SkewLangevin {
        assert_eq!(a.len(), n * n, "matrix must be {n}x{n}");
        assert!(
            antisymmetry_defect(&a, n) == 0.0,
            "the skew drift must be exactly antisymmetric; a matrix that is antisymmetric to \
             within rounding does not preserve the target to within rounding"
        );
        assert!(beta > 0.0 && beta.is_finite(), "beta must be positive and finite, got {beta}");
        assert!(dt > 0.0 && dt.is_finite(), "dt must be positive and finite, got {dt}");
        SkewLangevin { a, n, beta, dt }
    }

    /// The added drift `A grad E`, written into `out`.
    ///
    /// # Panics
    ///
    /// If either slice is the wrong length.
    pub fn skew_drift(&self, grad: &[f64], out: &mut [f64]) {
        assert_eq!(grad.len(), self.n, "gradient must be length {}", self.n);
        assert_eq!(out.len(), self.n, "output must be length {}", self.n);
        for i in 0..self.n {
            let row: Vec<f64> =
                (0..self.n).map(|j| self.a[i * self.n + j] * grad[j]).collect();
            out[i] = sum_up(&row);
        }
    }

    /// One Euler–Maruyama step given the gradient at the current point.
    ///
    /// The caller supplies `grad` because the energy is the caller's: this integrator works for any
    /// differentiable `E`, and [`crate::kuramoto::Kuramoto::xy_grad`] is one supplier of it.
    ///
    /// # Panics
    ///
    /// If either slice is the wrong length.
    pub fn step(&self, x: &mut [f64], grad: &[f64], rng: &mut Pcg, ledger: Option<&mut Ledger>) {
        assert_eq!(x.len(), self.n, "state must be length {}", self.n);
        let mut skew = vec![0.0; self.n];
        self.skew_drift(grad, &mut skew);
        let sigma = (2.0 * self.dt / self.beta).sqrt();
        for i in 0..self.n {
            let noise = sigma * gaussian(rng);
            x[i] += self.dt * (-grad[i] + skew[i]) + noise;
        }
        if let Some(l) = ledger {
            l.samples += self.n as u64;
        }
    }
}

/// A standard normal from the crate's own generator, by Box–Muller.
fn gaussian(rng: &mut Pcg) -> f64 {
    // `f64()` is in [0, 1); the log needs a strictly positive argument, so the zero is lifted to the
    // smallest positive value the generator can produce rather than resampled, which would make the
    // stream length depend on its own contents.
    let u1 = rng.f64().max(f64::MIN_POSITIVE);
    let u2 = rng.f64();
    (-2.0 * u1.ln()).sqrt() * (core::f64::consts::TAU * u2).cos()
}

/// What one lifted sweep did.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LiftStats {
    /// Proposals accepted, each moving one site by one grid step.
    pub moves: u64,
    /// Proposals rejected, each reversing one site's direction bit.
    pub reversals: u64,
}

/// A lifted, non-reversible single-site sampler for the clock model.
///
/// # The invariance, in full
///
/// Fix a site and write `pi` for its conditional. The kernel on `(a, v)` is: go to `(a + v, v)` with
/// probability `alpha = min(1, pi(a+v)/pi(a))`, otherwise go to `(a, -v)`. Inflow to `(y, +1)` comes
/// from exactly two places — an accepted move from `(y-1, +1)` and a rejection at `(y, -1)`:
///
/// ```text
///   (pi(y-1)/2) min(1, pi(y)/pi(y-1))  +  (pi(y)/2) [1 - min(1, pi(y-1)/pi(y))]
/// ```
///
/// If `pi(y) >= pi(y-1)` this is `pi(y-1)/2 + (pi(y) - pi(y-1))/2 = pi(y)/2`. If `pi(y) < pi(y-1)`
/// the first term is `pi(y)/2` and the second is zero, giving `pi(y)/2` again. So `pi(a)/2` is
/// invariant in both cases, exactly, with no condition on `pi`. The same argument applies verbatim
/// at `v = -1`, and a sweep is a composition of per-site kernels each preserving the joint measure.
///
/// # What it is not
///
/// It is not reversible, and it is not supposed to be: `[balance_defect]` on the same instance is
/// strictly positive. A `pi`-invariant chain that violates detailed balance still gives unbiased
/// expectations under the ergodic theorem — what it loses is the spectral theory that assumes a
/// self-adjoint operator, which is why the autocorrelation comparison in this module is computed
/// from the transition matrix directly rather than from eigenvalues.
pub struct LiftedClock<'m> {
    m: &'m Potts,
    beta: f64,
    a: Vec<u8>,
    v: Vec<i8>,
    off: Vec<usize>,
    nbr: Vec<u32>,
    w: Vec<f64>,
    rng: Pcg,
}

impl<'m> LiftedClock<'m> {
    /// Start from an all-zero state with every direction bit `+1`.
    ///
    /// # Panics
    ///
    /// If `beta` is not finite and non-negative.
    #[must_use]
    pub fn new(m: &'m Potts, beta: f64, seed: u64) -> LiftedClock<'m> {
        assert!(beta >= 0.0 && beta.is_finite(), "beta must be finite and non-negative, got {beta}");
        let n = m.n();
        let mut deg = vec![0usize; n];
        for (i, j, _) in m.edges() {
            deg[i] += 1;
            deg[j] += 1;
        }
        let mut off = vec![0usize; n + 1];
        for i in 0..n {
            off[i + 1] = off[i] + deg[i];
        }
        let mut cursor = off.clone();
        let mut nbr = vec![0u32; off[n]];
        let mut w = vec![0.0f64; off[n]];
        for (i, j, jij) in m.edges() {
            nbr[cursor[i]] = j as u32;
            w[cursor[i]] = jij;
            cursor[i] += 1;
            nbr[cursor[j]] = i as u32;
            w[cursor[j]] = jij;
            cursor[j] += 1;
        }
        LiftedClock {
            m,
            beta,
            a: vec![0u8; n],
            v: vec![1i8; n],
            off,
            nbr,
            w,
            rng: Pcg::new(seed, 0x11F7),
        }
    }

    /// The current categorical state.
    #[must_use]
    pub fn state(&self) -> &[u8] {
        &self.a
    }

    /// The current direction bits.
    #[must_use]
    pub fn directions(&self) -> &[i8] {
        &self.v
    }

    /// Overwrite the state.
    ///
    /// # Panics
    ///
    /// If the length is wrong or any value is not below `q`.
    pub fn set_state(&mut self, a: &[u8]) {
        assert_eq!(a.len(), self.a.len(), "state must have one entry per site");
        assert!(a.iter().all(|&x| usize::from(x) < self.m.q()), "a state is not below q");
        self.a.copy_from_slice(a);
    }

    /// The energy contributed by site `i` when it carries state `c`, with its neighbours held.
    ///
    /// `-sum_j J_ij pair(c, a_j) - h_i(c)`, which is all the conditional depends on.
    ///
    /// # Panics
    ///
    /// If `i` is past the end or `c` is not below `q`.
    #[must_use]
    pub fn site_energy(&self, i: usize, c: u8) -> f64 {
        let q = self.m.q();
        let kind = self.m.kind();
        let mut terms = Vec::with_capacity(self.off[i + 1] - self.off[i] + 1);
        for k in self.off[i]..self.off[i + 1] {
            terms.push(-self.w[k] * kind.pair(q, c, self.a[self.nbr[k] as usize]));
        }
        terms.push(-self.m.field(i, c));
        sum_up(&terms)
    }

    /// One lifted sweep over every site, charging one device sample per site.
    pub fn sweep(&mut self, ledger: Option<&mut Ledger>) -> LiftStats {
        let q = self.m.q();
        let mut stats = LiftStats::default();
        for i in 0..self.a.len() {
            let cur = self.a[i];
            let step = self.v[i];
            let next = step_state(cur, step, q);
            let de = self.site_energy(i, next) - self.site_energy(i, cur);
            // min(1, exp(-beta dE)), evaluated without forming exp of a large positive argument.
            let accept = de <= 0.0 || self.rng.f64() < (-self.beta * de).exp();
            if accept {
                self.a[i] = next;
                stats.moves += 1;
            } else {
                self.v[i] = -step;
                stats.reversals += 1;
            }
        }
        if let Some(l) = ledger {
            l.samples += self.a.len() as u64;
        }
        stats
    }

    /// Run `n` sweeps.
    pub fn sweeps(&mut self, n: usize, mut ledger: Option<&mut Ledger>) -> LiftStats {
        let mut total = LiftStats::default();
        for _ in 0..n {
            let s = self.sweep(ledger.as_deref_mut());
            total.moves += s.moves;
            total.reversals += s.reversals;
        }
        total
    }
}

/// One grid step in direction `step`, wrapping.
fn step_state(a: u8, step: i8, q: usize) -> u8 {
    let q = q as i32;
    let next = (i32::from(a) + i32::from(step)).rem_euclid(q);
    next as u8
}

/// The exact transition matrix of the lifted chain on a **one-site** clock model, row-major over
/// `2q` states indexed `a * 2 + (v < 0) as usize`.
///
/// One site, because that is where the claim can be checked rather than sampled: the kernel is the
/// atom a sweep composes, and a composition of measure-preserving kernels preserves the measure.
///
/// # Panics
///
/// If the model has more than one site.
#[must_use]
pub fn lifted_matrix(m: &Potts, beta: f64) -> Vec<f64> {
    assert_eq!(m.n(), 1, "the exact matrix is built for a single site; sweeps compose it");
    let q = m.q();
    let e: Vec<f64> = (0..q).map(|c| -m.field(0, c as u8)).collect();
    let mut p = vec![0.0; (2 * q) * (2 * q)];
    for a in 0..q {
        for (vi, step) in [1i8, -1i8].into_iter().enumerate() {
            let from = a * 2 + vi;
            let next = usize::from(step_state(a as u8, step, q));
            let de = e[next] - e[a];
            let alpha = if de <= 0.0 { 1.0 } else { (-beta * de).exp() };
            p[from * (2 * q) + next * 2 + vi] += alpha;
            p[from * (2 * q) + a * 2 + (1 - vi)] += 1.0 - alpha;
        }
    }
    p
}

/// The reversible chain with the **identical** proposal: one grid step, either direction with
/// probability one half, Metropolis acceptance. Row-major over `q` states.
///
/// The comparator. Any speed difference between this and [`lifted_matrix`] is attributable to the
/// lifting and to nothing else, because the proposal, the acceptance rule and the target are the
/// same.
///
/// # Panics
///
/// If the model has more than one site.
#[must_use]
pub fn reversible_matrix(m: &Potts, beta: f64) -> Vec<f64> {
    assert_eq!(m.n(), 1, "the exact matrix is built for a single site");
    let q = m.q();
    let e: Vec<f64> = (0..q).map(|c| -m.field(0, c as u8)).collect();
    let mut p = vec![0.0; q * q];
    for a in 0..q {
        for step in [1i8, -1i8] {
            let next = usize::from(step_state(a as u8, step, q));
            let de = e[next] - e[a];
            let alpha = if de <= 0.0 { 1.0 } else { (-beta * de).exp() };
            p[a * q + next] += 0.5 * alpha;
            p[a * q + a] += 0.5 * (1.0 - alpha);
        }
    }
    p
}

/// The Boltzmann weights of a one-site model, normalised.
///
/// # Panics
///
/// If the model has more than one site.
#[must_use]
pub fn site_measure(m: &Potts, beta: f64) -> Vec<f64> {
    assert_eq!(m.n(), 1, "a site measure is built for a single site");
    let q = m.q();
    let w: Vec<f64> = (0..q).map(|c| (beta * m.field(0, c as u8)).exp()).collect();
    let z = sum_up(&w);
    w.into_iter().map(|x| x / z).collect()
}

/// `max_y |(mu P)(y) - mu(y)|` — how far `mu` is from invariant under `P`.
///
/// # Panics
///
/// If the shapes disagree.
#[must_use]
pub fn stationary_defect(p: &[f64], mu: &[f64]) -> f64 {
    let n = mu.len();
    assert_eq!(p.len(), n * n, "transition matrix must be {n}x{n}");
    let mut m = 0.0f64;
    for y in 0..n {
        let col: Vec<f64> = (0..n).map(|x| mu[x] * p[x * n + y]).collect();
        m = m.max((sum_up(&col) - mu[y]).abs());
    }
    m
}

/// `max_{x,y} |mu(x) P(x,y) - mu(y) P(y,x)|` — how far the chain is from reversible.
///
/// Strictly positive for a lifted chain, and that is the point rather than a defect.
///
/// # Panics
///
/// If the shapes disagree.
#[must_use]
pub fn balance_defect(p: &[f64], mu: &[f64]) -> f64 {
    let n = mu.len();
    assert_eq!(p.len(), n * n, "transition matrix must be {n}x{n}");
    let mut m = 0.0f64;
    for x in 0..n {
        for y in 0..n {
            m = m.max((mu[x] * p[x * n + y] - mu[y] * p[y * n + x]).abs());
        }
    }
    m
}

/// The exact integrated autocorrelation time of observable `f` under `P` with stationary law `mu`.
///
/// `tau = 1/2 + sum_{t>=1} C(t)/C(0)`, with `C(t) = <f, P^t f>_mu - <f>^2` evaluated by repeated
/// application of `P` to `f`. Computed, not estimated: no chain is run, no window is chosen, and the
/// only approximation is truncating the sum at `max_lag`.
///
/// # Panics
///
/// If the shapes disagree.
#[must_use]
pub fn exact_tau_int(p: &[f64], mu: &[f64], f: &[f64], max_lag: usize) -> f64 {
    let n = mu.len();
    assert_eq!(p.len(), n * n, "transition matrix must be {n}x{n}");
    assert_eq!(f.len(), n, "observable must be length {n}");
    let mean = {
        let t: Vec<f64> = (0..n).map(|x| mu[x] * f[x]).collect();
        sum_up(&t)
    };
    let centred: Vec<f64> = f.iter().map(|v| v - mean).collect();
    let cov = |g: &[f64]| -> f64 {
        let t: Vec<f64> = (0..n).map(|x| mu[x] * centred[x] * g[x]).collect();
        sum_up(&t)
    };
    let c0 = cov(&centred);
    if c0 <= 0.0 {
        return 0.5;
    }
    let mut g = centred.clone();
    let mut acc = 0.5;
    for _ in 0..max_lag {
        let mut next = vec![0.0; n];
        for x in 0..n {
            let row: Vec<f64> = (0..n).map(|y| p[x * n + y] * g[y]).collect();
            next[x] = sum_up(&row);
        }
        g = next;
        acc += cov(&g) / c0;
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kuramoto::Kuramoto;
    use crate::potts::{Interaction, PottsBuilder, enumerate};

    /// A one-site clock model in a field with two unequal wells, so the chain has somewhere to go.
    fn one_site(q: usize) -> Potts {
        let mut b = PottsBuilder::new(q, 1, Interaction::Clock);
        for c in 0..q {
            let t = core::f64::consts::TAU * c as f64 / q as f64;
            b.field(0, c as u8, 1.4 * t.cos() + 0.35 * (2.0 * t).cos());
        }
        b.build()
    }

    #[test]
    fn the_two_identities_the_skew_drift_rests_on_hold() {
        // tr(A H) = 0 for antisymmetric A against the symmetric XY Hessian...
        let n = 5;
        let a = skew_matrix(n, 0.8, 4242);
        assert_eq!(antisymmetry_defect(&a, n), 0.0, "constructed antisymmetric, not rounded to it");
        let mut k = vec![0.0; n * n];
        let mut rng = Pcg::new(99, 3);
        for i in 0..n {
            for j in (i + 1)..n {
                let v = 2.0 * rng.f64() - 1.0;
                k[i * n + j] = v;
                k[j * n + i] = v;
            }
        }
        let sys = Kuramoto::gradient(k, n).expect("symmetric fixture");
        for t in [0.0, 0.7, 2.1] {
            let phi: Vec<f64> = (0..n).map(|i| t * (i as f64 + 1.0) * 0.37).collect();
            let h = sys.xy_hessian(&phi);
            assert!(
                trace_against(&a, &h, n).abs() < 1e-13,
                "tr(A H) should vanish, got {}",
                trace_against(&a, &h, n)
            );
            // ...and grad E . A grad E = 0 for every gradient.
            let mut g = vec![0.0; n];
            sys.xy_grad(&phi, &mut g);
            assert!(
                quadratic_form(&a, &g, n).abs() < 1e-14,
                "an antisymmetric quadratic form must vanish, got {}",
                quadratic_form(&a, &g, n)
            );
        }
    }

    #[test]
    fn the_trace_identity_needs_the_hessian_to_be_symmetric() {
        // The identity is not an accident of the matrices; drop symmetry and it fails. This is what
        // separates the working construction from an asymmetric coupling.
        let n = 4;
        let a = skew_matrix(n, 1.0, 7);
        let mut b = vec![0.0; n * n];
        let mut rng = Pcg::new(5, 11);
        for i in 0..n * n {
            b[i] = 2.0 * rng.f64() - 1.0;
        }
        assert!(
            trace_against(&a, &b, n).abs() > 1e-3,
            "a generic matrix should not annihilate the trace"
        );
    }

    #[test]
    fn the_lifted_chain_preserves_the_target_exactly() {
        let q = 16;
        let m = one_site(q);
        for &beta in &[0.0, 0.5, 1.0, 3.0] {
            let p = lifted_matrix(&m, beta);
            let pi = site_measure(&m, beta);
            // The lifted measure is pi(a)/2 on each of the two direction bits.
            let mu: Vec<f64> = pi.iter().flat_map(|&x| [x / 2.0, x / 2.0]).collect();
            let d = stationary_defect(&p, &mu);
            assert!(d < 1e-15, "lifted chain is not stationary at beta = {beta}: defect {d}");
            // Every row is a probability distribution.
            for a in 0..(2 * q) {
                let row: Vec<f64> = (0..(2 * q)).map(|b| p[a * (2 * q) + b]).collect();
                assert!((sum_up(&row) - 1.0).abs() < 1e-15, "row {a} does not sum to one");
            }
        }
    }

    #[test]
    fn the_lifted_chain_is_stationary_without_being_reversible() {
        // The two numbers that are the whole module: one zero, one not.
        let q = 16;
        let m = one_site(q);
        let beta = 1.5;
        let p = lifted_matrix(&m, beta);
        let pi = site_measure(&m, beta);
        let mu: Vec<f64> = pi.iter().flat_map(|&x| [x / 2.0, x / 2.0]).collect();
        assert!(stationary_defect(&p, &mu) < 1e-15);
        assert!(
            balance_defect(&p, &mu) > 1e-3,
            "a lifted chain that satisfied detailed balance would not be a lifted chain"
        );
        // The comparator does satisfy it, on the same instance and the same proposal.
        let r = reversible_matrix(&m, beta);
        assert!(stationary_defect(&r, &pi) < 1e-15);
        assert!(balance_defect(&r, &pi) < 1e-15, "the Metropolis comparator must be reversible");
    }

    #[test]
    fn lifting_beats_the_reversible_chain_with_the_same_proposal() {
        let q = 16;
        let m = one_site(q);
        let beta = 1.5;
        let p = lifted_matrix(&m, beta);
        let r = reversible_matrix(&m, beta);
        let pi = site_measure(&m, beta);
        let mu: Vec<f64> = pi.iter().flat_map(|&x| [x / 2.0, x / 2.0]).collect();
        let obs: Vec<f64> =
            (0..q).map(|c| (core::f64::consts::TAU * c as f64 / q as f64).cos()).collect();
        let lifted_obs: Vec<f64> = obs.iter().flat_map(|&x| [x, x]).collect();
        let tau_lift = exact_tau_int(&p, &mu, &lifted_obs, 4_000);
        let tau_rev = exact_tau_int(&r, &pi, &obs, 4_000);
        assert!(
            tau_lift < tau_rev,
            "lifting should not be slower: lifted {tau_lift}, reversible {tau_rev}"
        );
        // Not a rounding-scale win. Measured on this instance the ratio is several-fold; the
        // assertion is deliberately well inside it so that a change in the fixture fails loudly
        // rather than silently passing on a coin flip.
        assert!(
            tau_rev / tau_lift > 2.0,
            "expected a substantial speed-up, got {tau_rev} / {tau_lift}"
        );
    }

    #[test]
    fn a_lifted_sweep_reproduces_the_enumerated_distribution() {
        // The composition claim, checked by sampling a model small enough to enumerate exactly.
        let q = 4;
        let n = 3;
        let mut b = PottsBuilder::new(q, n, Interaction::Clock);
        b.couple(0, 1, 0.9);
        b.couple(1, 2, -0.6);
        b.field(0, 1, 0.5);
        b.field(2, 3, 0.4);
        let m = b.build();
        let beta = 0.8;
        let exact = enumerate(&m, beta).expect("small enough to enumerate");

        let mut s = LiftedClock::new(&m, beta, 2026);
        let mut led = Ledger::default();
        s.sweeps(2_000, Some(&mut led));
        let mut hist = vec![0u64; q.pow(n as u32)];
        let draws = 400_000u64;
        for _ in 0..draws {
            s.sweep(Some(&mut led));
            let idx = m.index_of(s.state()).expect("valid state");
            hist[idx] += 1;
        }
        let got: Vec<f64> = hist.iter().map(|&c| c as f64 / draws as f64).collect();
        let want = &exact.p;
        let tv: f64 = 0.5
            * sum_up(&got.iter().zip(want.iter()).map(|(a, b)| (a - b).abs()).collect::<Vec<_>>());
        assert!(tv < 0.01, "total variation {tv} against the exact distribution is too large");
        assert_eq!(led.samples, (2_000 + draws) * n as u64, "one sample charged per site per sweep");
    }

    #[test]
    fn the_skew_integrator_runs_and_charges_its_updates() {
        let n = 3;
        let a = skew_matrix(n, 0.5, 1);
        let sl = SkewLangevin::new(a, n, 1.0, 1e-3);
        let k = vec![0.0, 0.7, 0.2, 0.7, 0.0, -0.4, 0.2, -0.4, 0.0];
        let sys = Kuramoto::gradient(k, n).expect("symmetric fixture");
        let mut rng = Pcg::new(3, 9);
        let mut x = vec![0.1, 0.2, 0.3];
        let mut led = Ledger::default();
        let mut g = vec![0.0; n];
        for _ in 0..1_000 {
            sys.xy_grad(&x, &mut g);
            sl.step(&mut x, &g, &mut rng, Some(&mut led));
        }
        assert_eq!(led.samples, 3_000);
        assert!(x.iter().all(|v| v.is_finite()), "the integrator diverged");
    }
}
