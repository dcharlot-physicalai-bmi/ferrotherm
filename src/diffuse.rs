//! Discrete diffusion — D3PM kernels, the exact posterior, the variational bound, the masked
//! objective, and SEDD score entropy.
//!
//! # What this is, and why it is separate from [`crate::dtm`]
//!
//! A diffusion model over a FINITE ALPHABET. The forward process corrupts a sequence of tokens by a
//! Markov chain with an explicit transition matrix `Q_t`, and the reverse process — learned —
//! generates by undoing it. This is the lane that actually produces text, code and bytes, and none
//! of it was here: [`crate::dtm`] is the same idea in continuous TIME over a BINARY alphabet with a
//! Boltzmann machine as the denoiser, and it carries no likelihood machinery at all.
//!
//! The two meet where they should. The uniform D3PM cumulative kernel at `alpha_bar = e^{-gamma t}`
//! reproduces [`crate::dtm::keep_prob`] for every alphabet size, and the test that says so is one
//! of the oracles below — an existing module with its own closed-form check, used as the reference
//! for this one.
//!
//! # The pieces, and the papers they come from
//!
//! * **D3PM forward kernels** (Austin, Johnson, Ho, Tarlow & van den Berg, "Structured Denoising
//!   Diffusion Models in Discrete State-Spaces", `NeurIPS` 2021). [`Kernel::Uniform`] mixes each token
//!   toward the uniform law; [`Kernel::Absorbing`] sends it to a `[MASK]` token it can never leave.
//!   Both have a closed-form cumulative `Q_bar_t`, which is what makes `x_t` samplable in one step
//!   from `x_0` — see [`Forward::sample_at`].
//! * **The exact posterior** `q(x_{t-1} | x_t, x_0)`, Austin et al. Eq. 3. This is the target the
//!   reverse model is fitted against, and it is the one place a transposed index produces numbers
//!   that still sum to one — see the Bayes oracle below.
//! * **The variational bound** on `log p(x_0)`, Austin et al. Eq. 4: a prior term, `T-1` posterior
//!   KLs, and a reconstruction term.
//! * **The masked-diffusion simplified objective** (Sahoo et al., "Simple and Effective Masked
//!   Diffusion Language Models"; Shi et al., "Simplified and Generalized Masked Diffusion", both
//!   `NeurIPS` 2024). Under the absorbing kernel and the SUBS parameterisation the whole bound
//!   collapses to a weighted cross-entropy over the MASKED positions only. That collapse is a
//!   theorem, so it is testable: [`masked_nelbo`] and [`SiteReverse::elbo`] are independent code
//!   paths that must return the same number.
//! * **SEDD score entropy** (Lou, Meng & Ermon, "Discrete Diffusion Modeling by Estimating the
//!   Ratios of the Data Distribution", ICML 2024). The network learns the ratios
//!   `p_t(y) / p_t(x_t)` rather than a distribution, and the loss that identifies them is a Bregman
//!   divergence — zero exactly at the true ratio, positive everywhere else. [`score_entropy_term`]
//!   is written in the form that makes that zero EXACT in `f64` rather than merely small.
//!
//! # Signs, indices and the two traps
//!
//! `Q_t` is ROW-STOCHASTIC here: `q_step(t, a, b)` is `P(x_t = b | x_{t-1} = a)`. The posterior
//! numerator is therefore `q_step(t, j, x_t) * q_bar(t-1, x_0, j)` — the step matrix read down its
//! COLUMN `x_t`, and the cumulative read at `t-1`, not `t`. Both mistakes leave a vector that
//! normalises to one and is wrong, which is why the posterior is checked against a dense product of
//! the single-step matrices rather than against itself.
//!
//! The second trap is the prior. The masked objective drops the prior KL term because the absorbing
//! schedule is supposed to reach `alpha_bar_T = 0` — everything masked. If it does not, that term
//! is INFINITE rather than small, so [`masked_nelbo`] refuses the schedule instead of returning a
//! number that silently is not a bound.
//!
//! # What is a bound here
//!
//! [`SiteReverse::elbo`] and [`masked_nelbo`] claim to LOWER-bound `log p(x_0)`. Their terms are
//! accumulated through [`crate::round::sum_up`] and negated, so the reported bound is never above
//! the exact one — the discipline this crate adopted after a "lower bound" shipped above the
//! optimum for want of a directed sum.

use crate::rng::Pcg;
use crate::round::sum_up;

/// Which D3PM transition family the forward process uses.
///
/// The two differ in what "fully noised" means, and they fail differently when a schedule is short:
/// the absorbing chain reaches its stationary law EXACTLY the moment `alpha_bar` hits zero, while
/// the uniform chain only ever approaches it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kernel {
    /// Each token is kept with probability `1 - beta_t` and otherwise redrawn uniformly over the
    /// vocabulary. Stationary law: uniform. The state space is the vocabulary itself.
    Uniform,
    /// Each token is kept with probability `1 - beta_t` and otherwise replaced by `[MASK]`, which
    /// has no way out. Stationary law: the point mass on `[MASK]`. The state space is the
    /// vocabulary plus one extra state, indexed last.
    Absorbing,
}

/// Why a discrete-diffusion object was refused.
#[derive(Clone, Debug, PartialEq)]
pub enum DiffuseError {
    /// A diffusion with no steps has no forward process and no bound to take.
    NoSteps,
    /// Fewer than two tokens is a constant, not an alphabet.
    TinyVocab {
        /// The vocabulary size asked for.
        vocab: usize,
    },
    /// A noise rate outside `[0, 1]`, which is not a probability.
    BadBeta {
        /// One-based step index.
        at: usize,
        /// The offending value.
        beta: f64,
    },
    /// This `(x_0, x_t)` pair has probability zero under the forward process, so the posterior
    /// `q(x_{t-1} | x_t, x_0)` conditions on an event that cannot happen.
    ///
    /// Under the absorbing kernel this is the ordinary case rather than an exotic one: a data token
    /// at time `t` can only have come from ITSELF, so every other `x_0` is unreachable. Returning
    /// zeros there would look like a distribution and be none.
    Unreachable {
        /// The clean token conditioned on.
        x0: usize,
        /// The noised token conditioned on.
        xt: usize,
        /// One-based step index.
        t: usize,
    },
    /// A token index outside the state space.
    BadToken {
        /// Position within the sequence.
        at: usize,
        /// The offending token.
        token: u32,
        /// States the process has, so the token had to be below this.
        states: usize,
    },
    /// A sequence, distribution or prediction of the wrong length.
    Width {
        /// The length supplied.
        got: usize,
        /// The length required.
        want: usize,
    },
    /// A row that is not a probability distribution.
    NotStochastic {
        /// Which row.
        row: usize,
        /// What it summed to.
        sum: f64,
    },
    /// The table this model would need is larger than [`MAX_TABLE`] cells.
    ///
    /// An exhaustive table over `states^positions` contexts is a REFERENCE denoiser, not a
    /// practical one. It exists so the learned object can be compared against exact enumeration;
    /// past this size that comparison is no longer what anyone is doing.
    TooLarge {
        /// Cells the table would need.
        cells: usize,
        /// The ceiling.
        limit: usize,
    },
    /// The schedule does not drive `alpha_bar_T` to zero, so the masked objective's dropped prior
    /// term is infinite rather than absent.
    ///
    /// The masked-diffusion bound assumes `q(x_T | x_0)` is the point mass on `[MASK]`. When it is
    /// not, `p(x_T) = delta_MASK` assigns probability zero to something `q` gives positive mass,
    /// the prior KL is `+inf`, and a routine that quietly omits it returns a number that is not a
    /// bound on anything.
    PriorNotAbsorbed {
        /// The offending `alpha_bar_T`, which had to be exactly zero.
        alpha_bar_t: f64,
    },
    /// Every clean token the forward process could have produced was assigned probability zero by
    /// the denoiser, so the reverse step has nothing to sample from.
    Degenerate {
        /// One-based step index.
        t: usize,
        /// The noised token whose reverse step collapsed.
        xt: usize,
    },
}

impl core::fmt::Display for DiffuseError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            DiffuseError::NoSteps => write!(f, "a diffusion needs at least one step"),
            DiffuseError::TinyVocab { vocab } => {
                write!(f, "a vocabulary of {vocab} is a constant; two tokens is the minimum")
            }
            DiffuseError::BadBeta { at, beta } => {
                write!(f, "beta_{at} is {beta}, which is not a probability in [0, 1]")
            }
            DiffuseError::Unreachable { x0, xt, t } => write!(
                f,
                "the forward process cannot take x_0 = {x0} to x_{t} = {xt}, so there is no \
                 posterior to condition on"
            ),
            DiffuseError::BadToken { at, token, states } => {
                write!(f, "token {token} at position {at} is not below the {states} states")
            }
            DiffuseError::Width { got, want } => {
                write!(f, "length {got} where {want} was required")
            }
            DiffuseError::NotStochastic { row, sum } => {
                write!(f, "row {row} sums to {sum} rather than one")
            }
            DiffuseError::TooLarge { cells, limit } => {
                write!(f, "an exhaustive table of {cells} cells exceeds the {limit} ceiling")
            }
            DiffuseError::PriorNotAbsorbed { alpha_bar_t } => write!(
                f,
                "the masked bound drops a prior KL that is only zero when alpha_bar_T is, and this \
                 schedule ends at {alpha_bar_t:e}"
            ),
            DiffuseError::Degenerate { t, xt } => write!(
                f,
                "at step {t} the denoiser gave every reachable clean token zero probability for \
                 x_t = {xt}, so the reverse step has no support"
            ),
        }
    }
}

impl core::error::Error for DiffuseError {}

/// Ceiling on an exhaustive [`TableDenoiser`], in cells.
///
/// A table over every context is the reference object that exact enumeration can check; it is not a
/// model anybody scales. Four million cells is already far past anything a test enumerates and well
/// short of a surprise allocation.
pub const MAX_TABLE: usize = 1 << 22;

/// A discrete forward diffusion: an alphabet, a kernel family, and a noise schedule.
///
/// Times run `1 ..= T` for the schedule and `0 ..= T` for the cumulative kernel, with
/// `alpha_bar_0 = 1` so that `Q_bar_0` is the identity — which is what makes the posterior at
/// `t = 1` come out as the reconstruction term rather than a special case.
pub struct Forward {
    kernel: Kernel,
    v: usize,
    betas: Vec<f64>,
    abar: Vec<f64>,
}

impl Forward {
    /// A forward process over `v` data tokens with the given per-step noise rates.
    ///
    /// `betas[i]` is `beta_{i+1}`. The cumulative `alpha_bar_t` is the running product of
    /// `1 - beta_s`, computed once here: a schedule that ends at `beta_T = 1` therefore lands on
    /// `alpha_bar_T = 0.0` EXACTLY, because the last factor is exactly zero and zero times anything
    /// is zero. That exactness is what the absorbing stationary test asserts on.
    ///
    /// # Errors
    ///
    /// [`DiffuseError::NoSteps`], [`DiffuseError::TinyVocab`] or [`DiffuseError::BadBeta`].
    pub fn new(kernel: Kernel, v: usize, betas: Vec<f64>) -> Result<Forward, DiffuseError> {
        if betas.is_empty() {
            return Err(DiffuseError::NoSteps);
        }
        if v < 2 {
            return Err(DiffuseError::TinyVocab { vocab: v });
        }
        for (i, &b) in betas.iter().enumerate() {
            // `contains` is false for NaN, so negating it REJECTS NaN -- which is the behaviour
            // the BadBeta test pins, and the reason this reads as a negation rather than a range.
            if !(0.0..=1.0).contains(&b) {
                return Err(DiffuseError::BadBeta { at: i + 1, beta: b });
            }
        }
        let mut abar = Vec::with_capacity(betas.len() + 1);
        abar.push(1.0);
        let mut a = 1.0f64;
        for &b in &betas {
            a *= 1.0 - b;
            abar.push(a);
        }
        Ok(Forward { kernel, v, betas, abar })
    }

    /// A constant-rate schedule: `beta_t = beta` at every step.
    ///
    /// The uniform kernel under this schedule reaches its stationary law only in the limit —
    /// `alpha_bar_T = (1 - beta)^T` is small and never zero — which is exactly the case the
    /// stationary oracle uses to separate "exact" from "tight".
    ///
    /// # Errors
    ///
    /// As [`Forward::new`].
    pub fn constant(
        kernel: Kernel,
        v: usize,
        t_steps: usize,
        beta: f64,
    ) -> Result<Forward, DiffuseError> {
        Forward::new(kernel, v, vec![beta; t_steps])
    }

    /// A linear schedule from `b_first` at `t = 1` to `b_last` at `t = T`.
    ///
    /// # Errors
    ///
    /// As [`Forward::new`].
    pub fn linear(
        kernel: Kernel,
        v: usize,
        t_steps: usize,
        b_first: f64,
        b_last: f64,
    ) -> Result<Forward, DiffuseError> {
        if t_steps == 0 {
            return Err(DiffuseError::NoSteps);
        }
        let d = if t_steps == 1 { 0.0 } else { (b_last - b_first) / (t_steps as f64 - 1.0) };
        let betas = (0..t_steps).map(|i| b_first + d * i as f64).collect();
        Forward::new(kernel, v, betas)
    }

    /// The standard absorbing schedule of Austin et al.: `beta_t = 1 / (T - t + 1)`, giving
    /// `alpha_bar_t = 1 - t/T`.
    ///
    /// Its last rate is exactly `1.0`, so `alpha_bar_T` is exactly `0.0` and the forward marginal at
    /// `T` is the point mass on `[MASK]` to the BIT rather than to a tolerance. Every masked
    /// objective depends on that, which is why this constructor exists rather than a comment
    /// telling callers to arrange it.
    ///
    /// # Errors
    ///
    /// As [`Forward::new`].
    pub fn absorbing_linear(v: usize, t_steps: usize) -> Result<Forward, DiffuseError> {
        if t_steps == 0 {
            return Err(DiffuseError::NoSteps);
        }
        let betas = (0..t_steps).map(|i| 1.0 / (t_steps - i) as f64).collect();
        Forward::new(Kernel::Absorbing, v, betas)
    }

    /// The transition family in use.
    #[must_use]
    pub fn kernel(&self) -> Kernel {
        self.kernel
    }

    /// Data tokens, EXCLUDING the mask state if there is one.
    #[must_use]
    pub fn vocab(&self) -> usize {
        self.v
    }

    /// Steps `T` in the schedule.
    #[must_use]
    pub fn steps(&self) -> usize {
        self.betas.len()
    }

    /// States the chain moves over: the vocabulary, plus one for `[MASK]` under the absorbing
    /// kernel.
    #[must_use]
    pub fn states(&self) -> usize {
        self.v + usize::from(self.kernel == Kernel::Absorbing)
    }

    /// The index of `[MASK]`, or `None` for a kernel that has none.
    #[must_use]
    pub fn mask(&self) -> Option<usize> {
        match self.kernel {
            Kernel::Uniform => None,
            Kernel::Absorbing => Some(self.v),
        }
    }

    /// The noise rate `beta_t`, one-based.
    ///
    /// # Panics
    ///
    /// If `t` is zero or past `T`.
    #[must_use]
    pub fn beta(&self, t: usize) -> f64 {
        assert!(t >= 1 && t <= self.betas.len(), "beta is indexed 1..={}", self.betas.len());
        self.betas[t - 1]
    }

    /// The cumulative keep weight `alpha_bar_t = prod_{s<=t} (1 - beta_s)`, with
    /// `alpha_bar_0 = 1`.
    ///
    /// # Panics
    ///
    /// If `t` is past `T`.
    #[must_use]
    pub fn alpha_bar(&self, t: usize) -> f64 {
        assert!(t < self.abar.len(), "alpha_bar is indexed 0..={}", self.betas.len());
        self.abar[t]
    }

    /// The stationary law of the kernel, as a distribution over [`Forward::states`].
    #[must_use]
    pub fn stationary(&self) -> Vec<f64> {
        let s = self.states();
        match self.kernel {
            Kernel::Uniform => vec![1.0 / self.v as f64; self.v],
            Kernel::Absorbing => {
                let mut p = vec![0.0; s];
                p[s - 1] = 1.0;
                p
            }
        }
    }

    /// One step of the transition matrix: `P(x_t = b | x_{t-1} = a)`, ROW-stochastic in `a`.
    ///
    /// # Panics
    ///
    /// If `t` is not in `1..=T`, or either state is out of range.
    #[must_use]
    pub fn q_step(&self, t: usize, a: usize, b: usize) -> f64 {
        let s = self.states();
        assert!(a < s && b < s, "states are 0..{s}");
        let beta = self.beta(t);
        match self.kernel {
            Kernel::Uniform => {
                (1.0 - beta) * f64::from(u8::from(a == b)) + beta / self.v as f64
            }
            Kernel::Absorbing => {
                let m = s - 1;
                if a == m {
                    // The mask state is absorbing: this row is `delta_MASK` for every schedule,
                    // exactly, and that exactness is the whole reason the kernel is called
                    // absorbing.
                    f64::from(u8::from(b == m))
                } else if b == a {
                    1.0 - beta
                } else if b == m {
                    beta
                } else {
                    0.0
                }
            }
        }
    }

    /// The cumulative kernel `P(x_t = b | x_0 = a)` in closed form, for `t` in `0..=T`.
    ///
    /// Closed form rather than a product of [`Forward::q_step`] matrices — which is what lets the
    /// posterior test use that product as an INDEPENDENT reference.
    ///
    /// # Panics
    ///
    /// If `t` is past `T`, or either state is out of range.
    #[must_use]
    pub fn q_bar(&self, t: usize, a: usize, b: usize) -> f64 {
        let s = self.states();
        assert!(a < s && b < s, "states are 0..{s}");
        let abar = self.alpha_bar(t);
        match self.kernel {
            Kernel::Uniform => abar * f64::from(u8::from(a == b)) + (1.0 - abar) / self.v as f64,
            Kernel::Absorbing => {
                let m = s - 1;
                if a == m {
                    f64::from(u8::from(b == m))
                } else if b == a {
                    abar
                } else if b == m {
                    1.0 - abar
                } else {
                    0.0
                }
            }
        }
    }

    /// The whole single-step matrix, row-major over [`Forward::states`].
    ///
    /// # Panics
    ///
    /// If `t` is not in `1..=T`.
    #[must_use]
    pub fn step_matrix(&self, t: usize) -> Vec<f64> {
        let s = self.states();
        let mut m = vec![0.0; s * s];
        for a in 0..s {
            for b in 0..s {
                m[a * s + b] = self.q_step(t, a, b);
            }
        }
        m
    }

    /// The whole cumulative matrix, row-major over [`Forward::states`].
    ///
    /// # Panics
    ///
    /// If `t` is past `T`.
    #[must_use]
    pub fn cumulative_matrix(&self, t: usize) -> Vec<f64> {
        let s = self.states();
        let mut m = vec![0.0; s * s];
        for a in 0..s {
            for b in 0..s {
                m[a * s + b] = self.q_bar(t, a, b);
            }
        }
        m
    }

    /// The forward marginal `q(x_t | x_0)` as a distribution over [`Forward::states`].
    ///
    /// # Panics
    ///
    /// If `t` is past `T` or `x0` is out of range.
    #[must_use]
    pub fn marginal(&self, x0: usize, t: usize) -> Vec<f64> {
        (0..self.states()).map(|b| self.q_bar(t, x0, b)).collect()
    }

    /// The exact posterior `q(x_{t-1} | x_t, x_0)` (Austin et al. Eq. 3), as a distribution over
    /// [`Forward::states`].
    ///
    /// Bayes on the two kernels: the numerator is `q(x_t | x_{t-1}) q(x_{t-1} | x_0)` and the
    /// denominator is `q(x_t | x_0)`. The step matrix is read down its COLUMN `x_t` and the
    /// cumulative at `t-1`; transposing either leaves a vector that still normalises to one.
    ///
    /// # Errors
    ///
    /// [`DiffuseError::Unreachable`] when the forward process gives this `(x_0, x_t)` pair zero
    /// probability, which under the absorbing kernel is most pairs.
    ///
    /// # Panics
    ///
    /// If `t` is not in `1..=T`, or either state is out of range.
    pub fn posterior(&self, x0: usize, xt: usize, t: usize) -> Result<Vec<f64>, DiffuseError> {
        let den = self.q_bar(t, x0, xt);
        if !(den > 0.0) {
            return Err(DiffuseError::Unreachable { x0, xt, t });
        }
        let s = self.states();
        let mut q = vec![0.0; s];
        for j in 0..s {
            let num = self.q_step(t, j, xt) * self.q_bar(t - 1, x0, j);
            q[j] = num / den;
        }
        Ok(q)
    }

    /// Advance a whole sequence one forward step, in place, position by position.
    ///
    /// # Panics
    ///
    /// If `t` is not in `1..=T`, or a token is out of range.
    pub fn step(&self, x: &mut [u32], t: usize, rng: &mut Pcg) {
        let s = self.states();
        let mut row = vec![0.0; s];
        for tok in x.iter_mut() {
            let a = *tok as usize;
            assert!(a < s, "token {a} is not below the {s} states");
            for (b, r) in row.iter_mut().enumerate() {
                *r = self.q_step(t, a, b);
            }
            *tok = draw(&row, rng) as u32;
        }
    }

    /// Sample `x_t` from `x_0` in ONE draw per position, through the cumulative kernel.
    ///
    /// This is the whole practical point of having `Q_bar` in closed form: training never walks the
    /// chain.
    ///
    /// # Panics
    ///
    /// If `t` is past `T`, or a token is out of range.
    #[must_use]
    pub fn sample_at(&self, x0: &[u32], t: usize, rng: &mut Pcg) -> Vec<u32> {
        let s = self.states();
        let mut row = vec![0.0; s];
        x0.iter()
            .map(|&tok| {
                let a = tok as usize;
                assert!(a < s, "token {a} is not below the {s} states");
                for (b, r) in row.iter_mut().enumerate() {
                    *r = self.q_bar(t, a, b);
                }
                draw(&row, rng) as u32
            })
            .collect()
    }

    /// The concrete score SEDD learns: `p_t(y | x_0) / p_t(x_t | x_0)` for every state `y`.
    ///
    /// The entry at `y = x_t` is one by construction and carries no information; the loss skips it.
    ///
    /// # Errors
    ///
    /// [`DiffuseError::Unreachable`] when the denominator is zero.
    ///
    /// # Panics
    ///
    /// If `t` is past `T`, or either state is out of range.
    pub fn conditional_ratios(
        &self,
        x0: usize,
        xt: usize,
        t: usize,
    ) -> Result<Vec<f64>, DiffuseError> {
        let den = self.q_bar(t, x0, xt);
        if !(den > 0.0) {
            return Err(DiffuseError::Unreachable { x0, xt, t });
        }
        Ok((0..self.states()).map(|y| self.q_bar(t, x0, y) / den).collect())
    }

    /// SEDD's per-transition weights at `x_t`: the forward rate-matrix row `Q(x_t, y)`.
    ///
    /// The rate matrices are the paper's: `Q_uniform = 11^T - N I`, so every off-diagonal weight is
    /// one; `Q_absorb = e_MASK 1^T - I`, so the only transitions with weight are those OUT of the
    /// mask state — at an unmasked position the reverse process has nothing to decide and the
    /// weights are all zero, which is the carry-over structure falling out of the rate matrix
    /// rather than being imposed on top of it.
    #[must_use]
    pub fn sedd_weights(&self, xt: usize) -> Vec<f64> {
        let s = self.states();
        let mut w = vec![0.0; s];
        match self.kernel {
            Kernel::Uniform => {
                for (y, e) in w.iter_mut().enumerate() {
                    if y != xt {
                        *e = 1.0;
                    }
                }
            }
            Kernel::Absorbing => {
                if xt == s - 1 {
                    for e in w.iter_mut().take(s - 1) {
                        *e = 1.0;
                    }
                }
            }
        }
        w
    }
}

/// Draw an index from a distribution, by inverse CDF.
fn draw(p: &[f64], rng: &mut Pcg) -> usize {
    let u = rng.f64();
    let mut c = 0.0;
    for (i, &q) in p.iter().enumerate() {
        c += q;
        if u < c {
            return i;
        }
    }
    // Only reachable when the row sums fractionally below one; the last support point is the
    // honest answer rather than a panic on a rounding artefact.
    p.iter().rposition(|&q| q > 0.0).unwrap_or(0)
}

/// `KL(q || p)` in nats over a finite alphabet, with `0 log 0 = 0`.
///
/// Infinite when `p` starves a state `q` gives mass to. That is the correct answer and it is
/// load-bearing: the masked bound's dropped prior term is infinite, not small, when the schedule
/// fails to absorb, and a routine that returned a finite number there would be reporting a bound
/// that is not one.
///
/// # Panics
///
/// If the two distributions have different lengths.
#[must_use]
pub fn kl(q: &[f64], p: &[f64]) -> f64 {
    assert_eq!(q.len(), p.len(), "KL needs two laws on the same alphabet");
    let mut terms = Vec::with_capacity(q.len());
    for (&qi, &pi) in q.iter().zip(p) {
        if qi <= 0.0 {
            continue;
        }
        if !(pi > 0.0) {
            return f64::INFINITY;
        }
        terms.push(qi * (qi / pi).ln());
    }
    terms.iter().sum()
}

/// Turn accumulated NELBO terms into an ELBO that is never ABOVE the exact one.
///
/// [`crate::round::sum_up`] is never below the exact total, so its negation is never above the
/// exact ELBO, and a lower bound stays a lower bound. Non-finite terms are answered directly
/// because the compensated summation has nothing to say about `inf - inf`.
fn elbo_from_nelbo(terms: &[f64]) -> f64 {
    if terms.iter().any(|t| t.is_nan()) {
        return f64::NAN;
    }
    if terms.iter().any(|t| t.is_infinite()) {
        return f64::NEG_INFINITY;
    }
    -sum_up(terms)
}

/// A fully explicit single-site reverse process: a prior over `x_T` and one transition table per
/// step.
///
/// This is the REFERENCE reverse model — the most general one a single site admits, and small
/// enough that both its exact likelihood and its exact variational bound can be enumerated. Every
/// parameterised denoiser in this module is checked against it.
///
/// One site, not a sequence, deliberately. A sequence model's exact likelihood needs the joint over
/// `states^positions`, and the absorbing kernel does not tensor into a single absorbing kernel on
/// that product alphabet, so the honest reference is the case where the enumeration is exact rather
/// than the case that looks more like a language model.
pub struct SiteReverse {
    s: usize,
    prior: Vec<f64>,
    step: Vec<Vec<f64>>,
}

impl SiteReverse {
    /// A reverse process from an explicit prior and per-step tables.
    ///
    /// `step[t-1]` is row-major over `states`, indexed `x_t * states + x_{t-1}`: each ROW is a
    /// distribution over the previous state.
    ///
    /// # Errors
    ///
    /// [`DiffuseError::Width`] for a table of the wrong shape, [`DiffuseError::NotStochastic`] for
    /// a row that is not a distribution.
    pub fn new(
        s: usize,
        prior: Vec<f64>,
        step: Vec<Vec<f64>>,
    ) -> Result<SiteReverse, DiffuseError> {
        if prior.len() != s {
            return Err(DiffuseError::Width { got: prior.len(), want: s });
        }
        let psum: f64 = prior.iter().sum();
        if (psum - 1.0).abs() > 1e-9 {
            return Err(DiffuseError::NotStochastic { row: 0, sum: psum });
        }
        for (t, m) in step.iter().enumerate() {
            if m.len() != s * s {
                return Err(DiffuseError::Width { got: m.len(), want: s * s });
            }
            for r in 0..s {
                let rs: f64 = m[r * s..(r + 1) * s].iter().sum();
                if (rs - 1.0).abs() > 1e-9 {
                    return Err(DiffuseError::NotStochastic { row: t * s + r, sum: rs });
                }
            }
        }
        Ok(SiteReverse { s, prior, step })
    }

    /// States the chain moves over.
    #[must_use]
    pub fn states(&self) -> usize {
        self.s
    }

    /// Steps `T`.
    #[must_use]
    pub fn steps(&self) -> usize {
        self.step.len()
    }

    /// The exact reverse of `fwd` started from the data law `data`, which is the model that makes
    /// the variational bound TIGHT.
    ///
    /// Built by Bayes on the forward MARGINALS, which is valid because the forward chain is Markov:
    /// `P(x_{t-1} = j | x_t = i)` is `P(x_{t-1} = j) q_step(t, j, i) / P(x_t = i)`. A state the
    /// forward process never reaches gets a uniform row, which no expectation ever weighs.
    ///
    /// Two claims follow and both are asserted in the tests: the likelihood of this chain is
    /// exactly the data law, and the bound against it has zero gap.
    ///
    /// # Errors
    ///
    /// [`DiffuseError::Width`] if `data` is not one entry per DATA token,
    /// [`DiffuseError::NotStochastic`] if it is not a distribution.
    pub fn exact(fwd: &Forward, data: &[f64]) -> Result<SiteReverse, DiffuseError> {
        if data.len() != fwd.vocab() {
            return Err(DiffuseError::Width { got: data.len(), want: fwd.vocab() });
        }
        let dsum: f64 = data.iter().sum();
        if (dsum - 1.0).abs() > 1e-9 {
            return Err(DiffuseError::NotStochastic { row: 0, sum: dsum });
        }
        let s = fwd.states();
        let t_steps = fwd.steps();
        let marg: Vec<Vec<f64>> = (0..=t_steps)
            .map(|t| {
                (0..s)
                    .map(|b| (0..data.len()).map(|m| data[m] * fwd.q_bar(t, m, b)).sum())
                    .collect()
            })
            .collect();
        let mut step = Vec::with_capacity(t_steps);
        for t in 1..=t_steps {
            let mut m = vec![0.0; s * s];
            for xt in 0..s {
                let den = marg[t][xt];
                if den > 0.0 {
                    for j in 0..s {
                        m[xt * s + j] = marg[t - 1][j] * fwd.q_step(t, j, xt) / den;
                    }
                } else {
                    for j in 0..s {
                        m[xt * s + j] = 1.0 / s as f64;
                    }
                }
            }
            step.push(m);
        }
        SiteReverse::new(s, marg[t_steps].clone(), step)
    }

    /// The SUBS parameterisation of a masked reverse process from an explicit per-step prediction
    /// of `x_0` (Sahoo et al.).
    ///
    /// `x0_pred[t-1]` is the model's distribution over DATA tokens given `x_t = [MASK]`. Two
    /// structural facts are imposed rather than learned, and they are what make the bound collapse:
    /// an unmasked token CARRIES OVER unchanged, and the last step unmasks with probability one.
    ///
    /// # Errors
    ///
    /// [`DiffuseError::Width`] for the wrong number of steps or the wrong prediction width,
    /// [`DiffuseError::PriorNotAbsorbed`] if the schedule does not end fully masked, and
    /// [`DiffuseError::NotStochastic`] if a prediction is not a distribution.
    ///
    /// # Panics
    ///
    /// If `fwd` is not an absorbing kernel.
    pub fn subs(fwd: &Forward, x0_pred: &[Vec<f64>]) -> Result<SiteReverse, DiffuseError> {
        let m = fwd.mask().expect("SUBS is the masked parameterisation and needs a mask state");
        let t_steps = fwd.steps();
        if x0_pred.len() != t_steps {
            return Err(DiffuseError::Width { got: x0_pred.len(), want: t_steps });
        }
        let abar_t = fwd.alpha_bar(t_steps);
        if abar_t != 0.0 {
            return Err(DiffuseError::PriorNotAbsorbed { alpha_bar_t: abar_t });
        }
        let s = fwd.states();
        let mut step = Vec::with_capacity(t_steps);
        for t in 1..=t_steps {
            let pred = &x0_pred[t - 1];
            if pred.len() != fwd.vocab() {
                return Err(DiffuseError::Width { got: pred.len(), want: fwd.vocab() });
            }
            let psum: f64 = pred.iter().sum();
            if (psum - 1.0).abs() > 1e-9 {
                return Err(DiffuseError::NotStochastic { row: t, sum: psum });
            }
            let (a_prev, a_now) = (fwd.alpha_bar(t - 1), fwd.alpha_bar(t));
            let unmask = (a_prev - a_now) / (1.0 - a_now);
            let mut mat = vec![0.0; s * s];
            for xt in 0..s {
                if xt == m {
                    mat[xt * s + m] = 1.0 - unmask;
                    for (j, &p) in pred.iter().enumerate() {
                        mat[xt * s + j] = unmask * p;
                    }
                } else {
                    mat[xt * s + xt] = 1.0;
                }
            }
            step.push(mat);
        }
        let mut prior = vec![0.0; s];
        prior[m] = 1.0;
        SiteReverse::new(s, prior, step)
    }

    /// Exact `log p_theta(x_0)` by marginalising the whole reverse chain.
    ///
    /// Dense, `O(T s^2)`, and exact up to the arithmetic — this is the quantity the bound is a
    /// bound ON, and there is no sampling anywhere in it.
    ///
    /// # Panics
    ///
    /// If `x0` is not a state of this chain.
    #[must_use]
    pub fn log_likelihood(&self, x0: usize) -> f64 {
        assert!(x0 < self.s, "states are 0..{}", self.s);
        let mut p = self.prior.clone();
        for t in (1..=self.step.len()).rev() {
            let m = &self.step[t - 1];
            let mut next = vec![0.0; self.s];
            for xt in 0..self.s {
                let w = p[xt];
                if w == 0.0 {
                    continue;
                }
                for (j, e) in next.iter_mut().enumerate() {
                    *e += w * m[xt * self.s + j];
                }
            }
            p = next;
        }
        p[x0].ln()
    }

    /// The D3PM variational bound on `log p_theta(x_0)` (Austin et al. Eq. 4), with every
    /// expectation taken EXACTLY by enumeration.
    ///
    /// Three groups of terms: the prior `KL(q(x_T | x_0) || p(x_T))`, the posterior KLs for
    /// `t = 2..T`, and the reconstruction term `-E log p(x_0 | x_1)`. They are accumulated as a
    /// NELBO through [`crate::round::sum_up`] and negated, so the returned number is never ABOVE
    /// the exact bound — see this module's header.
    ///
    /// Returns `-inf` when the model starves a state the forward process reaches, which is the
    /// honest value rather than a large finite one.
    ///
    /// # Panics
    ///
    /// If `x0` is not a state, or `fwd` and this chain disagree on the state count or step count.
    #[must_use]
    pub fn elbo(&self, fwd: &Forward, x0: usize) -> f64 {
        assert_eq!(fwd.states(), self.s, "the bound needs one alphabet, not two");
        assert_eq!(fwd.steps(), self.step.len(), "the bound needs one schedule, not two");
        assert!(x0 < self.s, "states are 0..{}", self.s);
        let t_steps = self.step.len();
        let mut terms = Vec::new();
        terms.push(kl(&fwd.marginal(x0, t_steps), &self.prior));
        for t in (2..=t_steps).rev() {
            for xt in 0..self.s {
                let w = fwd.q_bar(t, x0, xt);
                if w <= 0.0 {
                    continue;
                }
                let q = fwd.posterior(x0, xt, t).expect("weight is positive, so this is reachable");
                let p = &self.step[t - 1][xt * self.s..(xt + 1) * self.s];
                terms.push(w * kl(&q, p));
            }
        }
        for x1 in 0..self.s {
            let w = fwd.q_bar(1, x0, x1);
            if w <= 0.0 {
                continue;
            }
            terms.push(-w * self.step[0][x1 * self.s + x0].ln());
        }
        elbo_from_nelbo(&terms)
    }
}

/// A denoiser: given a noised sequence and a time, a distribution over the DATA vocabulary for
/// every position — the `x_0` parameterisation that D3PM, MDLM and SEDD all share.
pub trait Denoiser {
    /// Positions in a sequence this denoiser takes.
    fn positions(&self) -> usize;

    /// Data tokens it predicts over, excluding any mask state.
    fn vocab(&self) -> usize;

    /// Fill `out` — length `positions * vocab`, row-major by position — with `x_theta(x_t, t)`.
    fn predict_x0(&self, x_t: &[u32], t: usize, out: &mut [f64]);
}

/// The masked-diffusion simplified objective at one noised sequence (Sahoo et al.; Shi et al.).
///
/// A weighted cross-entropy over the MASKED positions only, with weight
/// `(alpha_bar_{t-1} - alpha_bar_t) / (1 - alpha_bar_t)`. Unmasked positions contribute exactly
/// nothing — not "approximately nothing": under carry-over the reverse step there is a point mass
/// that matches the posterior, so its KL is zero.
///
/// # Errors
///
/// [`DiffuseError::Width`] for a prediction of the wrong shape, [`DiffuseError::BadToken`] for a
/// token out of range.
///
/// # Panics
///
/// If `fwd` is not an absorbing kernel, or `t` is not in `1..=T`.
pub fn masked_term(
    fwd: &Forward,
    t: usize,
    x_t: &[u32],
    x0: &[u32],
    x0_pred: &[f64],
) -> Result<f64, DiffuseError> {
    let m = fwd.mask().expect("the masked objective is the absorbing kernel's");
    let v = fwd.vocab();
    if x_t.len() != x0.len() {
        return Err(DiffuseError::Width { got: x_t.len(), want: x0.len() });
    }
    if x0_pred.len() != x0.len() * v {
        return Err(DiffuseError::Width { got: x0_pred.len(), want: x0.len() * v });
    }
    let a_now = fwd.alpha_bar(t);
    if !(1.0 - a_now > 0.0) {
        // Nothing is masked at this time, so there is no cross-entropy to weight -- and the weight
        // itself is 0/0 there. Answering zero is the limit, not a dodge.
        return Ok(0.0);
    }
    let w = (fwd.alpha_bar(t - 1) - a_now) / (1.0 - a_now);
    let mut acc = Vec::new();
    for (l, (&xt, &clean)) in x_t.iter().zip(x0).enumerate() {
        if clean as usize >= v {
            return Err(DiffuseError::BadToken { at: l, token: clean, states: v });
        }
        if xt as usize == m {
            acc.push(-x0_pred[l * v + clean as usize].ln());
        }
    }
    Ok(w * acc.iter().sum::<f64>())
}

/// The masked-diffusion NELBO for one clean sequence, with the expectation over the forward process
/// taken EXACTLY by enumerating the `2^positions` mask patterns.
///
/// The prior term is dropped, which is only legitimate when the schedule absorbs completely — so
/// this refuses a schedule that does not, rather than returning a number that is not a bound.
///
/// # Errors
///
/// [`DiffuseError::PriorNotAbsorbed`] for a schedule that does not end fully masked,
/// [`DiffuseError::TooLarge`] for a sequence too long to enumerate mask patterns over,
/// [`DiffuseError::Width`] or [`DiffuseError::BadToken`] for a malformed sequence.
///
/// # Panics
///
/// If `fwd` is not an absorbing kernel.
pub fn masked_nelbo<D: Denoiser>(fwd: &Forward, d: &D, x0: &[u32]) -> Result<f64, DiffuseError> {
    let m = fwd.mask().expect("the masked objective is the absorbing kernel's");
    let l = x0.len();
    if l != d.positions() {
        return Err(DiffuseError::Width { got: l, want: d.positions() });
    }
    let v = fwd.vocab();
    if v != d.vocab() {
        return Err(DiffuseError::Width { got: d.vocab(), want: v });
    }
    let t_steps = fwd.steps();
    let abar_t = fwd.alpha_bar(t_steps);
    if abar_t != 0.0 {
        return Err(DiffuseError::PriorNotAbsorbed { alpha_bar_t: abar_t });
    }
    if l >= 24 {
        let cells = 1usize.checked_shl(l as u32).unwrap_or(usize::MAX);
        return Err(DiffuseError::TooLarge { cells, limit: 1 << 24 });
    }
    for (i, &tok) in x0.iter().enumerate() {
        if tok as usize >= v {
            return Err(DiffuseError::BadToken { at: i, token: tok, states: v });
        }
    }
    let mut pred = vec![0.0; l * v];
    let mut xt = vec![0u32; l];
    let mut terms = Vec::new();
    for t in 1..=t_steps {
        let a_now = fwd.alpha_bar(t);
        if !(1.0 - a_now > 0.0) {
            continue;
        }
        for pattern in 0..(1usize << l) {
            let mut w = 1.0;
            for (i, slot) in xt.iter_mut().enumerate() {
                if pattern >> i & 1 == 1 {
                    *slot = m as u32;
                    w *= 1.0 - a_now;
                } else {
                    *slot = x0[i];
                    w *= a_now;
                }
            }
            if w <= 0.0 {
                continue;
            }
            d.predict_x0(&xt, t, &mut pred);
            terms.push(w * masked_term(fwd, t, &xt, x0, &pred)?);
        }
    }
    Ok(sum_up(&terms))
}

/// One denoising-score-entropy term (Lou, Meng & Ermon, ICML 2024), in the form whose optimum is
/// EXACT.
///
/// The paper writes the term as `s - a log s + K(a)` with `K(a) = a(log a - 1)`, which is a Bregman
/// divergence of `-log` and is therefore non-negative with equality exactly at `s = a`. Written
/// that way the zero is NOT exact in `f64`: the two logarithms cancel analytically and not
/// numerically, so the optimum comes out at `1e-17` and any test asserting exactness has to be
/// loosened until it stops testing.
///
/// Regrouped as `s - a - a log(s/a)` the same quantity is identical mathematically and exact at the
/// optimum: `s - a` is exactly zero and `log(1)` is exactly zero, so the term is exactly zero. It
/// is also the better-conditioned form, since `s/a` is the ratio the loss is actually about.
///
/// `weight` is the rate-matrix entry from [`Forward::sedd_weights`]; a zero weight contributes
/// nothing and short-circuits, which is what makes an unmasked position free.
#[must_use]
pub fn score_entropy_term(weight: f64, score: f64, ratio: f64) -> f64 {
    if weight <= 0.0 {
        return 0.0;
    }
    if !(ratio > 0.0) {
        // The data ratio is zero: the loss reduces to the score itself, minimised at zero.
        return weight * score;
    }
    weight * (score - ratio - ratio * (score / ratio).ln())
}

/// The full denoising score entropy at one noised token.
///
/// `scores[y]` is the network's estimate of `p_t(y) / p_t(x_t)`. The entry at `y = x_t` is ignored,
/// as it is one by definition.
///
/// # Errors
///
/// [`DiffuseError::Width`] if `scores` is not one entry per state, and whatever
/// [`Forward::conditional_ratios`] returns.
///
/// # Panics
///
/// If `t` is past `T`, or either state is out of range.
pub fn denoising_score_entropy(
    fwd: &Forward,
    x0: usize,
    xt: usize,
    t: usize,
    scores: &[f64],
) -> Result<f64, DiffuseError> {
    let s = fwd.states();
    if scores.len() != s {
        return Err(DiffuseError::Width { got: scores.len(), want: s });
    }
    let ratios = fwd.conditional_ratios(x0, xt, t)?;
    let w = fwd.sedd_weights(xt);
    let terms: Vec<f64> = (0..s)
        .filter(|&y| y != xt)
        .map(|y| score_entropy_term(w[y], scores[y], ratios[y]))
        .collect();
    Ok(terms.iter().sum())
}

/// The reverse marginal `p_theta(x_{t-1} | x_t)` at one position, from an `x_0` prediction.
///
/// The D3PM `x_0` parameterisation: mix the exact posterior over the model's belief about `x_0`.
/// Clean tokens the forward process could NOT have produced are dropped and the rest renormalised,
/// which is not a patch — it is the projection that turns an arbitrary predictor into a valid
/// reverse kernel, and under the absorbing kernel it is exactly the SUBS carry-over rule: at an
/// unmasked `x_t` only `x_0 = x_t` survives, so the step is a point mass whatever the model said.
///
/// # Errors
///
/// [`DiffuseError::Width`] if `pred` is not one entry per data token, and
/// [`DiffuseError::Degenerate`] if the model gave every reachable clean token zero probability.
///
/// # Panics
///
/// If `t` is not in `1..=T`, or `xt` is out of range.
pub fn reverse_marginal(
    fwd: &Forward,
    t: usize,
    xt: usize,
    pred: &[f64],
) -> Result<Vec<f64>, DiffuseError> {
    if pred.len() != fwd.vocab() {
        return Err(DiffuseError::Width { got: pred.len(), want: fwd.vocab() });
    }
    let s = fwd.states();
    let mut out = vec![0.0; s];
    let mut mass = 0.0;
    for (x0, &p) in pred.iter().enumerate() {
        if p <= 0.0 || !(fwd.q_bar(t, x0, xt) > 0.0) {
            continue;
        }
        let q = fwd.posterior(x0, xt, t).expect("checked reachable");
        for (j, e) in out.iter_mut().enumerate() {
            *e += p * q[j];
        }
        mass += p;
    }
    if !(mass > 0.0) {
        return Err(DiffuseError::Degenerate { t, xt });
    }
    for e in &mut out {
        *e /= mass;
    }
    Ok(out)
}

/// Ancestral sampling from a trained reverse process: start at the stationary law and walk
/// `t = T` down to `1`.
///
/// Every position is stepped independently given `x_t`, which is the standard sampler and the
/// standard approximation: it is exact only in the limit of many steps, because two positions that
/// change in the SAME step do so independently and the model's correlation between them is lost.
/// That error is not folklore — it is measurable, and the histogram test below predicts its size in
/// closed form and asserts it.
///
/// # Errors
///
/// Whatever [`reverse_marginal`] returns.
pub fn ancestral_sample<D: Denoiser>(
    fwd: &Forward,
    d: &D,
    rng: &mut Pcg,
) -> Result<Vec<u32>, DiffuseError> {
    let l = d.positions();
    let s = fwd.states();
    let stat = fwd.stationary();
    let mut x: Vec<u32> = (0..l).map(|_| draw(&stat, rng) as u32).collect();
    let mut pred = vec![0.0; l * d.vocab()];
    for t in (1..=fwd.steps()).rev() {
        d.predict_x0(&x, t, &mut pred);
        for i in 0..l {
            let row = &pred[i * d.vocab()..(i + 1) * d.vocab()];
            let p = reverse_marginal(fwd, t, x[i] as usize, row)?;
            debug_assert_eq!(p.len(), s);
            x[i] = draw(&p, rng) as u32;
        }
    }
    Ok(x)
}

/// An exhaustive-table denoiser: one softmax per `(context, position)`, where the context is the
/// whole noised sequence.
///
/// Not a model anybody scales — a REFERENCE. Because it can represent any function of `x_t` it is
/// the right object for asking what the objective's minimiser actually is, and because the loss is
/// a cross-entropy in the logits it is convex, so "the fit converged" is a claim about arithmetic
/// rather than about luck.
///
/// Time-independent by construction, which is not a shortcut: under the absorbing kernel
/// `P(x_0 | x_t)` depends on `t` only through `x_t`, so the Bayes-optimal denoiser IS
/// time-independent, and that is the observation the masked-diffusion papers turn into a simpler
/// architecture.
pub struct TableDenoiser {
    l: usize,
    v: usize,
    s: usize,
    logit: Vec<f64>,
}

impl TableDenoiser {
    /// A table over `l` positions with every logit at zero, so every prediction starts uniform.
    ///
    /// # Errors
    ///
    /// [`DiffuseError::TooLarge`] past [`MAX_TABLE`] cells, [`DiffuseError::Width`] for zero
    /// positions.
    pub fn new(fwd: &Forward, l: usize) -> Result<TableDenoiser, DiffuseError> {
        if l == 0 {
            return Err(DiffuseError::Width { got: 0, want: 1 });
        }
        let (s, v) = (fwd.states(), fwd.vocab());
        let mut ctx = 1usize;
        for _ in 0..l {
            ctx = ctx.saturating_mul(s);
        }
        let cells = ctx.saturating_mul(l).saturating_mul(v);
        if cells > MAX_TABLE {
            return Err(DiffuseError::TooLarge { cells, limit: MAX_TABLE });
        }
        Ok(TableDenoiser { l, v, s, logit: vec![0.0; cells] })
    }

    /// The row index of a noised sequence, base [`Forward::states`], least significant position
    /// first.
    fn context(&self, x_t: &[u32]) -> usize {
        let mut c = 0usize;
        let mut p = 1usize;
        for &tok in x_t {
            c += tok as usize * p;
            p *= self.s;
        }
        c
    }

    /// Softmax of one `(context, position)` cell into `out`.
    fn softmax_into(&self, ctx: usize, pos: usize, out: &mut [f64]) {
        let base = (ctx * self.l + pos) * self.v;
        let row = &self.logit[base..base + self.v];
        let mx = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mut z = 0.0;
        for (o, &g) in out.iter_mut().zip(row) {
            let e = (g - mx).exp();
            *o = e;
            z += e;
        }
        for o in out.iter_mut() {
            *o /= z;
        }
    }

    /// Fit the masked objective by stochastic gradient descent on the logits.
    ///
    /// Each step draws a row of `data`, a time `t` uniform in `1..=T`, and `x_t` from the cumulative
    /// kernel, then takes a cross-entropy gradient step at every MASKED position.
    ///
    /// # The per-time weights are deliberately not applied, and it does not move the answer
    ///
    /// The objective's weight `(alpha_bar_{t-1} - alpha_bar_t) / (1 - alpha_bar_t)` is strictly
    /// positive and — this is the part worth writing down — it factorises out of every cell. Under
    /// the absorbing kernel `P(x_t | x_0)` is `alpha_bar^{kept} (1 - alpha_bar)^{masked}` times the
    /// indicator that `x_t` agrees with `x_0` wherever it is unmasked, and those counts depend only
    /// on `x_t`. So each cell's total weight is a constant times that indicator, and the minimiser
    /// is `P_data(x_0[pos] | x_t)` whatever the schedule weights are. Applying them would only
    /// rescale each cell's step size — by a factor of `1/t` on the standard schedule, which starves
    /// exactly the fully-masked context that matters most. [`masked_nelbo`] reports the weighted
    /// objective itself; this routine finds its minimiser.
    ///
    /// # Constant-rate SGD on a cross-entropy table does NOT converge, and the averaging is why
    /// this fit is trustworthy
    ///
    /// With a fixed rate the logit difference of a cell whose target is genuinely random does not
    /// settle — it mixes, around the right answer, with a stationary spread of about `sqrt(lr)`.
    /// Measured here on the two-correlated-bits fixture at `lr = 0.35` and 400,000 steps: the
    /// fully-masked context, whose correct marginal is one half, came out at **0.649** — a logit
    /// difference of 0.61 against the predicted `sqrt(0.35) = 0.59`. Nothing about that is a bug and
    /// no amount of extra steps fixes it, because the noise floor does not depend on the step count.
    ///
    /// So the last half of the trajectory is AVERAGED (Polyak & Juditsky, SIAM J. Control Optim.
    /// 1992), which is what turns mixing into an estimate: the same run, same seed, then gives
    /// **0.4951**. Cells whose target is deterministic diverge logarithmically rather than mix, and
    /// the averaging costs them almost nothing — the measured leak went from `4.3e-5` to `5.9e-5`.
    ///
    /// The bias is not cosmetic and the histogram test sees it. Sampling 20,000 sequences from the
    /// unaveraged fit of the two-correlated-bits law gives **12,920 / 6,952** on the diagonal — a
    /// 65/35 split of a distribution that is 50/50 — against **9,772 / 10,096** after averaging.
    ///
    /// # Errors
    ///
    /// [`DiffuseError::Width`] for a row of the wrong length, [`DiffuseError::BadToken`] for a
    /// token out of range, [`DiffuseError::NoSteps`] for empty data.
    ///
    /// # Panics
    ///
    /// If `fwd` and this table disagree on the alphabet.
    pub fn fit_masked(
        &mut self,
        fwd: &Forward,
        data: &[Vec<u32>],
        steps: usize,
        lr: f64,
        seed: u64,
    ) -> Result<(), DiffuseError> {
        assert_eq!(fwd.states(), self.s, "the fit needs one alphabet, not two");
        assert_eq!(fwd.vocab(), self.v, "the fit needs one vocabulary, not two");
        let m = fwd.mask().expect("the masked objective is the absorbing kernel's");
        if data.is_empty() {
            return Err(DiffuseError::NoSteps);
        }
        for row in data {
            if row.len() != self.l {
                return Err(DiffuseError::Width { got: row.len(), want: self.l });
            }
            for (i, &tok) in row.iter().enumerate() {
                if tok as usize >= self.v {
                    return Err(DiffuseError::BadToken { at: i, token: tok, states: self.v });
                }
            }
        }
        let mut rng = Pcg::new(seed, 0xD1FF);
        let mut p = vec![0.0; self.v];
        let burn = steps / 2;
        let mut avg = vec![0.0f64; self.logit.len()];
        let mut taken = 0usize;
        for k in 0..steps {
            let row = &data[rng.next_u32() as usize % data.len()];
            let t = 1 + rng.next_u32() as usize % fwd.steps();
            let xt = fwd.sample_at(row, t, &mut rng);
            let ctx = self.context(&xt);
            for pos in 0..self.l {
                if xt[pos] as usize != m {
                    continue;
                }
                self.softmax_into(ctx, pos, &mut p);
                let base = (ctx * self.l + pos) * self.v;
                let target = row[pos] as usize;
                for j in 0..self.v {
                    let g = p[j] - f64::from(u8::from(j == target));
                    self.logit[base + j] -= lr * g;
                }
            }
            if k >= burn {
                for (a, &g) in avg.iter_mut().zip(&self.logit) {
                    *a += g;
                }
                taken += 1;
            }
        }
        if taken > 0 {
            for (g, &a) in self.logit.iter_mut().zip(&avg) {
                *g = a / taken as f64;
            }
        }
        Ok(())
    }
}

impl Denoiser for TableDenoiser {
    fn positions(&self) -> usize {
        self.l
    }

    fn vocab(&self) -> usize {
        self.v
    }

    fn predict_x0(&self, x_t: &[u32], _t: usize, out: &mut [f64]) {
        let ctx = self.context(x_t);
        for pos in 0..self.l {
            self.softmax_into(ctx, pos, &mut out[pos * self.v..(pos + 1) * self.v]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// ORACLE 1, and the two halves fail differently ON PURPOSE.
    ///
    /// The absorbing chain's stationary law is the point mass on `[MASK]`, and the standard
    /// schedule's last rate is exactly `1.0`, so `alpha_bar_T` is exactly `0.0` and the forward
    /// marginal is that point mass TO THE BIT. Asserted with `assert_eq!` on `f64`, because
    /// anything looser would pass for a schedule that merely got close.
    ///
    /// The uniform chain never reaches its stationary law. The closed form says the deviation is
    /// exactly `alpha_bar_T (delta - 1/V)`, so the tolerance is not a guess: it is `alpha_bar_T`,
    /// and the test also asserts the deviation is NOT zero — otherwise this arm would be passing
    /// for the same reason as the other one and would test nothing.
    #[test]
    fn forward_marginal_is_exactly_stationary_when_absorbing_and_alpha_bar_close_when_uniform() {
        let v = 5;
        let f = Forward::absorbing_linear(v, 64).unwrap();
        assert_eq!(f.alpha_bar(64), 0.0, "the standard schedule's last factor is exactly zero");
        let stat = f.stationary();
        for x0 in 0..v {
            assert_eq!(f.marginal(x0, 64), stat, "x_0 = {x0} is not exactly absorbed");
        }
        // The mask row is stationary at EVERY time, not only the last.
        for t in 0..=64 {
            assert_eq!(f.marginal(v, t), stat, "the mask row moved at t = {t}");
        }

        let g = Forward::constant(Kernel::Uniform, v, 200, 0.05).unwrap();
        let abar = g.alpha_bar(200);
        assert!(abar > 0.0, "this arm only means something while the chain has NOT converged");
        let ustat = g.stationary();
        let mut worst = 0.0f64;
        for x0 in 0..v {
            let p = g.marginal(x0, 200);
            for (j, (&got, &want)) in p.iter().zip(&ustat).enumerate() {
                let d = (got - want).abs();
                // Closed form: the deviation is alpha_bar times (delta_{j,x0} - 1/V).
                let exact = abar * (f64::from(u8::from(j == x0)) - 1.0 / v as f64).abs();
                // The deviation is a DIFFERENCE of two numbers near 1/V, so it carries the ulp of
                // 1/V and not its own -- eight of those, not eight of 7e-6.
                let tol = 8.0 * f64::EPSILON / v as f64;
                assert!((d - exact).abs() <= tol, "x_0 {x0} j {j}: {d:e} vs {exact:e}");
                worst = worst.max(d);
            }
        }
        assert!(worst <= abar, "deviation {worst:e} must be within alpha_bar {abar:e}");
        assert!(worst > 0.0, "a uniform chain that converged EXACTLY would make this arm vacuous");
    }

    /// ORACLE: an existing module with its own closed-form check.
    ///
    /// [`crate::dtm::keep_prob`] is the continuous-time uniform jump kernel's keep probability,
    /// verified there against the semigroup property and the sign of the coupling. Our D3PM uniform
    /// cumulative kernel at `alpha_bar = e^{-gamma t}` must reproduce it for every alphabet size,
    /// and the off-diagonal must be the complement spread evenly.
    #[test]
    fn uniform_cumulative_kernel_matches_the_dtm_keep_prob_closed_form() {
        for v in [2usize, 3, 7, 16] {
            for gamma in [0.3f64, 1.0, 2.5] {
                for t in [0.1f64, 0.5, 1.0, 3.0] {
                    let abar = (-gamma * t).exp();
                    // One step whose cumulative keep weight is exactly this alpha_bar.
                    let f = Forward::new(Kernel::Uniform, v, vec![1.0 - abar]).unwrap();
                    let want = crate::dtm::keep_prob(gamma, t, v);
                    let got = f.q_bar(1, 0, 0);
                    assert!(
                        (got - want).abs() < 1e-15,
                        "v {v} gamma {gamma} t {t}: {got} vs dtm {want}"
                    );
                    let off = f.q_bar(1, 0, 1);
                    assert!(
                        (off - (1.0 - want) / (v as f64 - 1.0)).abs() < 1e-15,
                        "v {v}: the off-diagonal must be the complement spread evenly"
                    );
                }
            }
        }
    }

    /// ORACLE 2: Bayes, brute-forced from the two kernels.
    ///
    /// The reference is a DENSE PRODUCT of the single-step matrices — `Q_1 Q_2 ... Q_{t-1}` and
    /// `Q_1 ... Q_t` multiplied out — normalised by an explicit sum over `x_{t-1}`. Our posterior
    /// uses the closed-form cumulative and divides by `q_bar(t, x_0, x_t)` instead. The two agree
    /// only if the closed form, the index order and the denominator are all right; a transposed
    /// step matrix or a cumulative read at `t` instead of `t-1` still normalises to one and still
    /// fails here.
    #[test]
    fn posterior_matches_brute_force_bayes_from_dense_kernel_products() {
        for kernel in [Kernel::Uniform, Kernel::Absorbing] {
            let v = 4;
            let f = Forward::linear(kernel, v, 6, 0.12, 0.55).unwrap();
            let s = f.states();
            // Dense cumulative by repeated multiplication, independent of the closed form.
            let mut prod: Vec<Vec<f64>> = Vec::new();
            let mut cur: Vec<f64> = (0..s * s)
                .map(|k| f64::from(u8::from(k / s == k % s)))
                .collect();
            prod.push(cur.clone());
            for t in 1..=f.steps() {
                let q = f.step_matrix(t);
                let mut next = vec![0.0; s * s];
                for a in 0..s {
                    for b in 0..s {
                        let mut acc = 0.0;
                        for c in 0..s {
                            acc += cur[a * s + c] * q[c * s + b];
                        }
                        next[a * s + b] = acc;
                    }
                }
                cur = next;
                prod.push(cur.clone());
            }
            // The closed form must agree with the product to begin with.
            for t in 0..=f.steps() {
                for a in 0..s {
                    for b in 0..s {
                        assert!(
                            (f.q_bar(t, a, b) - prod[t][a * s + b]).abs() < 5e-16,
                            "{kernel:?} t {t} ({a},{b}): closed form and matrix product differ"
                        );
                    }
                }
            }
            for t in 1..=f.steps() {
                let q = f.step_matrix(t);
                for x0 in 0..s {
                    for xt in 0..s {
                        let den: f64 = (0..s).map(|j| q[j * s + xt] * prod[t - 1][x0 * s + j]).sum();
                        match f.posterior(x0, xt, t) {
                            Err(DiffuseError::Unreachable { .. }) => {
                                assert!(den <= 0.0, "{kernel:?}: refused a reachable pair");
                            }
                            Err(e) => panic!("{kernel:?}: unexpected {e}"),
                            Ok(got) => {
                                assert!(den > 0.0, "{kernel:?}: accepted an unreachable pair");
                                for j in 0..s {
                                    let want = q[j * s + xt] * prod[t - 1][x0 * s + j] / den;
                                    assert!(
                                        (got[j] - want).abs() < 5e-15,
                                        "{kernel:?} t {t} x0 {x0} xt {xt} j {j}: \
                                         {} vs brute force {want}",
                                        got[j]
                                    );
                                }
                                let sum: f64 = got.iter().sum();
                                assert!((sum - 1.0).abs() < 1e-14, "posterior sums to {sum}");
                            }
                        }
                    }
                }
            }
        }
    }

    /// The published STRUCTURE of the absorbing posterior, which normalisation alone cannot give.
    ///
    /// An unmasked token can only have come from itself, so `q(x_{t-1} | x_t, x_0)` is the point
    /// mass on `x_t` — exactly one, not approximately. A masked token came from `[MASK]` with
    /// probability `(1 - alpha_bar_{t-1})/(1 - alpha_bar_t)` and from `x_0` otherwise. Both numbers
    /// are written here from the papers rather than read out of the implementation.
    #[test]
    fn absorbing_posterior_has_the_published_carry_over_structure() {
        let v = 5;
        let f = Forward::absorbing_linear(v, 12).unwrap();
        let m = f.mask().unwrap();
        for t in 1..=12 {
            for x0 in 0..v {
                if f.alpha_bar(t) > 0.0 {
                    let q = f.posterior(x0, x0, t).unwrap();
                    for (j, &p) in q.iter().enumerate() {
                        assert_eq!(p, f64::from(u8::from(j == x0)), "carry-over must be EXACT");
                    }
                } else {
                    // Once the schedule has absorbed completely, NO data token survives at time t
                    // -- including the one we started from. Refusing the pair is the right answer
                    // and a posterior that returned `delta_{x_t}` here would be conditioning on an
                    // event of probability zero.
                    assert_eq!(
                        f.posterior(x0, x0, t),
                        Err(DiffuseError::Unreachable { x0, xt: x0, t }),
                        "t {t} is fully absorbed, so no data token is reachable"
                    );
                }
                let q = f.posterior(x0, m, t).unwrap();
                let (a_prev, a_now) = (f.alpha_bar(t - 1), f.alpha_bar(t));
                let stay = (1.0 - a_prev) / (1.0 - a_now);
                assert!((q[m] - stay).abs() < 1e-15, "t {t}: stay-masked {} vs {stay}", q[m]);
                assert!(
                    (q[x0] - (a_prev - a_now) / (1.0 - a_now)).abs() < 1e-15,
                    "t {t}: unmask-to-x0 probability"
                );
                for (j, &p) in q.iter().enumerate() {
                    if j != m && j != x0 {
                        assert_eq!(p, 0.0, "a masked token cannot become some OTHER data token");
                    }
                }
                // Other data tokens are unreachable, not zero-probability-but-defined.
                for other in (0..v).filter(|&o| o != x0) {
                    assert_eq!(
                        f.posterior(x0, other, t),
                        Err(DiffuseError::Unreachable { x0, xt: other, t })
                    );
                }
            }
        }
    }

    /// ORACLE 3: the bound must sit BELOW an exactly enumerated likelihood, and must be TIGHT
    /// exactly where the theory says it is.
    ///
    /// Three claims, and the third is what stops the first being vacuous:
    ///
    ///  1. for arbitrary reverse models the ELBO never exceeds `log p_theta(x_0)`, which is computed
    ///     by marginalising the whole reverse chain densely — no sampling, no shared code with the
    ///     bound;
    ///  2. against the EXACT reverse of the forward chain the gap is zero to `1e-12`, and the
    ///     likelihood is the data law itself;
    ///  3. against a random reverse model the gap is at least a tenth of a nat, so the test would
    ///     notice a bound that had quietly become an equality — a dropped prior term is worth about
    ///     that much and shows up here as a NEGATIVE gap.
    #[test]
    fn elbo_lower_bounds_the_enumerated_log_likelihood_and_is_tight_at_the_exact_reverse() {
        let mut rng = Pcg::new(0xE1B0, 5);
        for kernel in [Kernel::Uniform, Kernel::Absorbing] {
            let v = 3;
            let f = Forward::linear(kernel, v, 5, 0.15, 0.6).unwrap();
            let s = f.states();

            // (1) and (3): random reverse models.
            let mut worst_gap = f64::INFINITY;
            for _ in 0..12 {
                let mut prior: Vec<f64> = (0..s).map(|_| rng.f64() + 0.05).collect();
                let z: f64 = prior.iter().sum();
                for p in &mut prior {
                    *p /= z;
                }
                let step: Vec<Vec<f64>> = (0..f.steps())
                    .map(|_| {
                        let mut m = vec![0.0; s * s];
                        for r in 0..s {
                            let mut acc = 0.0;
                            for c in 0..s {
                                m[r * s + c] = rng.f64() + 0.05;
                                acc += m[r * s + c];
                            }
                            for c in 0..s {
                                m[r * s + c] /= acc;
                            }
                        }
                        m
                    })
                    .collect();
                let rev = SiteReverse::new(s, prior, step).unwrap();
                for x0 in 0..v {
                    let gap = rev.log_likelihood(x0) - rev.elbo(&f, x0);
                    assert!(gap >= -1e-12, "{kernel:?} x_0 {x0}: ELBO is ABOVE the likelihood by {gap:e}");
                    worst_gap = worst_gap.min(gap);
                }
            }
            assert!(
                worst_gap > 0.1,
                "{kernel:?}: the smallest gap over random models was {worst_gap:e}; a test whose \
                 gaps are all tiny cannot tell a bound from an identity"
            );

            // (2): the exact reverse chain of a data law makes the bound an equality.
            let data = [0.5, 0.3, 0.2];
            let rev = SiteReverse::exact(&f, &data).unwrap();
            for x0 in 0..v {
                let ll = rev.log_likelihood(x0);
                assert!(
                    (ll - data[x0].ln()).abs() < 1e-12,
                    "{kernel:?} x_0 {x0}: the exact reverse must reproduce the data law, got {ll}"
                );
                let gap = ll - rev.elbo(&f, x0);
                assert!(
                    gap.abs() < 1e-12,
                    "{kernel:?} x_0 {x0}: the bound must be TIGHT here, gap {gap:e}"
                );
            }
        }
    }

    /// The masked simplified objective and the general D3PM bound are two derivations of ONE
    /// number, and that is a theorem, so it is testable.
    ///
    /// The left side is a weighted cross-entropy over masked positions only, with no matrices in it
    /// at all. The right side builds the full SUBS reverse kernel, takes `T-1` KL divergences over
    /// the whole `V+1` alphabet plus a prior term and a reconstruction term, and negates. They agree
    /// to `1e-12` or one of the two derivations is wrong.
    #[test]
    fn the_masked_simplified_objective_equals_the_general_d3pm_elbo() {
        let v = 4;
        let f = Forward::absorbing_linear(v, 9).unwrap();
        let mut rng = Pcg::new(0x5AB0, 11);
        for _ in 0..6 {
            // A different x_0 prediction at every step, so a routine that silently used step 1's
            // prediction everywhere would not survive.
            let x0_pred: Vec<Vec<f64>> = (0..f.steps())
                .map(|_| {
                    let mut p: Vec<f64> = (0..v).map(|_| rng.f64() + 0.05).collect();
                    let z: f64 = p.iter().sum();
                    for e in &mut p {
                        *e /= z;
                    }
                    p
                })
                .collect();
            let rev = SiteReverse::subs(&f, &x0_pred).unwrap();
            for x0 in 0..v {
                // The simplified objective at one site: sum_t (abar_{t-1} - abar_t) * -ln pred.
                let mut simple = 0.0;
                for t in 1..=f.steps() {
                    let w = f.alpha_bar(t - 1) - f.alpha_bar(t);
                    simple += w * -x0_pred[t - 1][x0].ln();
                }
                let general = -rev.elbo(&f, x0);
                assert!(
                    (simple - general).abs() < 1e-12,
                    "x_0 {x0}: simplified {simple} vs general {general}"
                );
            }
        }
    }

    /// The same equality through the SEQUENCE path, which is the one a sampler uses.
    ///
    /// [`masked_nelbo`] enumerates mask patterns and calls [`masked_term`]; the reference is the
    /// single-site general bound. At one position they must agree exactly, and the test also runs
    /// two positions to prove the enumeration weights are right — there the reference is the SUM of
    /// the per-position bounds, which is correct only because the forward process factorises and
    /// the denoiser here does too.
    #[test]
    fn the_sequence_masked_nelbo_agrees_with_the_single_site_bound() {
        let v = 3;
        let f = Forward::absorbing_linear(v, 7).unwrap();
        let m = f.mask().unwrap() as u32;

        /// A denoiser that ignores its context entirely, so its sequence bound must be the sum of
        /// single-site bounds.
        struct Fixed {
            l: usize,
            v: usize,
            p: Vec<f64>,
        }
        impl Denoiser for Fixed {
            fn positions(&self) -> usize {
                self.l
            }
            fn vocab(&self) -> usize {
                self.v
            }
            fn predict_x0(&self, _x: &[u32], _t: usize, out: &mut [f64]) {
                for pos in 0..self.l {
                    out[pos * self.v..(pos + 1) * self.v].copy_from_slice(&self.p);
                }
            }
        }
        let p = vec![0.5, 0.3, 0.2];
        let d1 = Fixed { l: 1, v, p: p.clone() };
        let per_site: Vec<f64> = (0..v)
            .map(|x0| {
                let pred: Vec<Vec<f64>> = (0..f.steps()).map(|_| p.clone()).collect();
                -SiteReverse::subs(&f, &pred).unwrap().elbo(&f, x0)
            })
            .collect();
        for x0 in 0..v {
            let got = masked_nelbo(&f, &d1, &[x0 as u32]).unwrap();
            assert!(
                (got - per_site[x0]).abs() < 1e-12,
                "one position, x_0 {x0}: {got} vs {}",
                per_site[x0]
            );
        }
        let d2 = Fixed { l: 2, v, p };
        for a in 0..v {
            for b in 0..v {
                let got = masked_nelbo(&f, &d2, &[a as u32, b as u32]).unwrap();
                let want = per_site[a] + per_site[b];
                assert!((got - want).abs() < 1e-12, "two positions ({a},{b}): {got} vs {want}");
            }
        }
        // And a schedule that does not absorb is REFUSED rather than silently missing a prior term.
        let g = Forward::constant(Kernel::Absorbing, v, 5, 0.3).unwrap();
        assert!(matches!(
            masked_nelbo(&g, &d1, &[0]),
            Err(DiffuseError::PriorNotAbsorbed { .. })
        ));
        assert_eq!(f.mask().unwrap() as u32, m, "the mask index is the last state");
    }

    /// SEDD's score entropy is a Bregman divergence: EXACTLY zero at the true ratio, positive
    /// elsewhere, and quadratic around the optimum with curvature `1/a`.
    ///
    /// The zero is asserted with `assert_eq!` because the implementation is written in the form
    /// that makes it exact — `s - a - a log(s/a)` rather than the paper's `s - a log s + K(a)`,
    /// whose logs cancel analytically and not in `f64`.
    ///
    /// The curvature check is the independent one: `f(a + eps)` must be `eps^2 / (2a)` to leading
    /// order, and that number comes from differentiating the paper's expression, not from this
    /// code.
    #[test]
    fn score_entropy_is_exactly_zero_at_the_true_ratio_and_quadratic_around_it() {
        for a in [1e-3f64, 0.1, 1.0, 7.5, 300.0] {
            assert_eq!(score_entropy_term(1.0, a, a), 0.0, "the optimum must be EXACTLY zero");
            assert_eq!(score_entropy_term(2.5, a, a), 0.0, "and stay exact under any weight");
            for f in [0.5f64, 0.9, 1.1, 2.0] {
                let val = score_entropy_term(1.0, a * f, a);
                assert!(val > 0.0, "a {a} factor {f}: {val} must be strictly positive");
            }
            let eps = 1e-5 * a;
            let got = score_entropy_term(1.0, a + eps, a);
            let want = eps * eps / (2.0 * a);
            assert!(
                ((got - want) / want).abs() < 1e-4,
                "a {a}: curvature {got:e} vs closed form {want:e}"
            );
        }
        // A zero data ratio leaves the score itself, minimised at zero.
        assert_eq!(score_entropy_term(1.0, 0.0, 0.0), 0.0);
        assert_eq!(score_entropy_term(1.0, 0.4, 0.0), 0.4);
        // A zero weight is free.
        assert_eq!(score_entropy_term(0.0, 9.0, 1.0), 0.0);
    }

    /// The full SEDD loss is minimised by the D3PM conditional ratios, and its weights carry the
    /// absorbing kernel's carry-over structure without being told to.
    #[test]
    fn denoising_score_entropy_is_minimised_by_the_d3pm_conditional_ratios() {
        for kernel in [Kernel::Uniform, Kernel::Absorbing] {
            let v = 4;
            let f = Forward::linear(kernel, v, 8, 0.1, 0.5).unwrap();
            let s = f.states();
            for t in 1..=f.steps() {
                for x0 in 0..v {
                    for xt in 0..s {
                        let Ok(ratios) = f.conditional_ratios(x0, xt, t) else { continue };
                        let at_truth = denoising_score_entropy(&f, x0, xt, t, &ratios).unwrap();
                        assert_eq!(at_truth, 0.0, "{kernel:?} t {t}: the optimum must be EXACT");
                        for scale in [0.5f64, 1.7] {
                            let off: Vec<f64> = ratios.iter().map(|r| r * scale).collect();
                            let val = denoising_score_entropy(&f, x0, xt, t, &off).unwrap();
                            let w = f.sedd_weights(xt);
                            let any = (0..s).any(|y| y != xt && w[y] > 0.0 && ratios[y] > 0.0);
                            if any {
                                assert!(val > 0.0, "{kernel:?} t {t} scale {scale}: {val}");
                            }
                        }
                    }
                }
            }
            // Structure: under the absorbing kernel an unmasked position has NOTHING to learn.
            if kernel == Kernel::Absorbing {
                let m = f.mask().unwrap();
                assert!(f.sedd_weights(0).iter().all(|&w| w == 0.0), "carry-over is weightless");
                assert_eq!(f.sedd_weights(m)[m], 0.0, "no self-transition");
                assert_eq!(f.sedd_weights(m).iter().sum::<f64>(), v as f64);
            }
        }
    }

    /// ORACLE 4: train on a synthetic law, then check the SAMPLES against it — and against the
    /// closed-form size of the sampler's own approximation.
    ///
    /// The target is two perfectly correlated bits: `(0,0)` and `(1,1)` with probability one half
    /// each, `(0,1)` and `(1,0)` never. It is chosen because NO factorised model can represent it,
    /// so a sampler that merely learned per-position marginals would produce all four outcomes
    /// evenly and fail by a factor of two.
    ///
    /// # The off-diagonal rate is predicted, not tolerated
    ///
    /// Ancestral sampling steps positions independently given `x_t`, so the two bits can unmask in
    /// the SAME step, and when they do they are drawn independently from the marginal — one time in
    /// two that is an off-diagonal pair. On the standard schedule a masked position unmasks at step
    /// `t` with probability `1/t`, and the probability both are still masked entering step `t` is
    /// `(t/T)^2`, so the chance of a simultaneous unmasking is `sum_t (t/T)^2 / t^2 = 1/T`, and the
    /// off-diagonal rate is exactly `1/(2T)`. The schedule makes that exact rather than asymptotic:
    /// `(alpha_bar_{t-1} - alpha_bar_t) / (1 - alpha_bar_t)` equals `1/t` to the BIT here, and the
    /// recursion sums to `1/T = 0.015625` with nothing left over.
    ///
    /// At `T = 64` and 20,000 samples that predicts **156.2** off-diagonal draws against a binomial
    /// sigma of 12.5 — a twelve-sigma effect — while the trained model's own leak contributes about
    /// ONE. So this asserts a number, not an inequality. Measured over eight sampling seeds the
    /// z-scores were −1.95, −0.58, 0.14, −0.58, −0.98, 1.51, −0.42, −0.66.
    #[test]
    fn a_trained_masked_model_reproduces_the_synthetic_histogram_and_its_predicted_sampling_error() {
        let (v, l, t_steps) = (2usize, 2usize, 64usize);
        let f = Forward::absorbing_linear(v, t_steps).unwrap();
        let data = vec![vec![0u32, 0], vec![1u32, 1]];
        let mut d = TableDenoiser::new(&f, l).unwrap();
        d.fit_masked(&f, &data, 400_000, 0.35, 0x51EED).unwrap();

        // What the model learned, against the exact Bayes posteriors of the target law.
        let m = f.mask().unwrap() as u32;
        let mut pred = vec![0.0; l * v];
        d.predict_x0(&[m, m], t_steps, &mut pred);
        for pos in 0..l {
            assert!(
                (pred[pos * v] - 0.5).abs() < 5e-3,
                "nothing observed: position {pos} must predict the marginal, got {}",
                pred[pos * v]
            );
        }
        let mut leak = 0.0f64;
        for bit in 0..2u32 {
            d.predict_x0(&[bit, m], t_steps, &mut pred);
            leak = leak.max(pred[v + (1 - bit) as usize]);
            d.predict_x0(&[m, bit], t_steps, &mut pred);
            leak = leak.max(pred[(1 - bit) as usize]);
        }
        assert!(leak < 1e-3, "one bit observed pins the other; leak was {leak:e}");

        // The histogram.
        let n = 20_000usize;
        let mut rng = Pcg::new(0xD1CE, 7);
        let mut counts = [0usize; 4];
        for _ in 0..n {
            let x = ancestral_sample(&f, &d, &mut rng).unwrap();
            counts[(x[0] * 2 + x[1]) as usize] += 1;
        }
        let off = counts[1] + counts[2];
        let p_off = 1.0 / (2.0 * t_steps as f64);
        let predicted = n as f64 * p_off;
        let sigma = (n as f64 * p_off * (1.0 - p_off)).sqrt();
        assert!(
            (off as f64 - predicted).abs() < 4.0 * sigma + n as f64 * leak,
            "off-diagonal {off} against the derived 1/(2T) prediction {predicted:.1} \
             (4 sigma = {:.1}, model leak = {:.1})",
            4.0 * sigma,
            n as f64 * leak
        );
        // And the diagonal is the target's even split, less the mass the sampler misplaced.
        let want = 0.5 * (1.0 - p_off);
        let split = (n as f64 * 0.25).sqrt() / n as f64;
        for (i, c) in [(0usize, counts[0]), (3, counts[3])] {
            let got = c as f64 / n as f64;
            assert!(
                (got - want).abs() < 4.0 * split,
                "outcome {i}: {got:.4} vs target {want:.4}, tolerance {:.4}",
                4.0 * split
            );
        }
    }

    /// Sampling the EXACT reverse chain reproduces the data law, which separates "the sampler is
    /// right" from "the model is right".
    ///
    /// The previous test trains a model and then samples it; a bug in either half could hide in the
    /// other. Here the reverse process is the exact one of a known law, so anything but the law is
    /// the sampler's fault. One position, so there is no factorisation error to account for.
    #[test]
    fn ancestral_sampling_of_an_exactly_known_reverse_reproduces_the_data_law() {
        let v = 3;
        let f = Forward::absorbing_linear(v, 24).unwrap();
        let data = [0.5, 0.3, 0.2];

        /// The Bayes-optimal denoiser for this one-site law: with nothing observed it predicts the
        /// data law, and with the token observed it is not consulted.
        struct Exact {
            p: Vec<f64>,
        }
        impl Denoiser for Exact {
            fn positions(&self) -> usize {
                1
            }
            fn vocab(&self) -> usize {
                self.p.len()
            }
            fn predict_x0(&self, _x: &[u32], _t: usize, out: &mut [f64]) {
                out.copy_from_slice(&self.p);
            }
        }
        let d = Exact { p: data.to_vec() };
        let n = 40_000usize;
        let mut rng = Pcg::new(0xF1DE, 3);
        let mut counts = vec![0usize; v];
        for _ in 0..n {
            let x = ancestral_sample(&f, &d, &mut rng).unwrap();
            counts[x[0] as usize] += 1;
        }
        for (j, &c) in counts.iter().enumerate() {
            let got = c as f64 / n as f64;
            let sigma = (data[j] * (1.0 - data[j]) / n as f64).sqrt();
            assert!(
                (got - data[j]).abs() < 4.0 * sigma,
                "token {j}: {got:.4} vs {:.4}, 4 sigma = {:.4}",
                data[j],
                4.0 * sigma
            );
        }
    }

    /// Forward sampling agrees with the closed-form marginal it is supposed to draw from.
    #[test]
    fn sampled_forward_marginals_match_the_closed_form_cumulative() {
        for kernel in [Kernel::Uniform, Kernel::Absorbing] {
            let v = 4;
            let f = Forward::linear(kernel, v, 10, 0.08, 0.4).unwrap();
            let s = f.states();
            let n = 20_000usize;
            let mut rng = Pcg::new(0xC0FFEE, 1);
            for t in [1usize, 5, 10] {
                for x0 in 0..v {
                    let mut counts = vec![0usize; s];
                    for _ in 0..n {
                        counts[f.sample_at(&[x0 as u32], t, &mut rng)[0] as usize] += 1;
                    }
                    let want = f.marginal(x0, t);
                    for (j, &c) in counts.iter().enumerate() {
                        let got = c as f64 / n as f64;
                        let sigma = (want[j] * (1.0 - want[j]) / n as f64).sqrt().max(1e-9);
                        assert!(
                            (got - want[j]).abs() < 5.0 * sigma,
                            "{kernel:?} t {t} x_0 {x0} state {j}: {got:.4} vs {:.4}",
                            want[j]
                        );
                    }
                }
            }
            // And walking the chain one step at a time must land in the same place.
            let mut rng = Pcg::new(0xC0FFEE, 2);
            let n = 20_000usize;
            let mut counts = vec![0usize; s];
            for _ in 0..n {
                let mut x = [0u32];
                for t in 1..=f.steps() {
                    f.step(&mut x, t, &mut rng);
                }
                counts[x[0] as usize] += 1;
            }
            let want = f.marginal(0, f.steps());
            for (j, &c) in counts.iter().enumerate() {
                let got = c as f64 / n as f64;
                let sigma = (want[j] * (1.0 - want[j]) / n as f64).sqrt().max(1e-9);
                assert!(
                    (got - want[j]).abs() < 5.0 * sigma,
                    "{kernel:?} stepped state {j}: {got:.4} vs {:.4}",
                    want[j]
                );
            }
        }
    }

    /// The bound is accumulated through [`crate::round::sum_up`], so it can only err DOWNWARD.
    ///
    /// Round-to-nearest would let the reported bound drift either way, and a "lower bound" above
    /// the quantity it bounds is the defect this crate wrote a whole module to prevent. Asserted by
    /// comparing against a plain summation of the same terms: the directed one is never larger.
    #[test]
    fn the_reported_bound_is_never_above_a_round_to_nearest_accumulation() {
        let mut rng = Pcg::new(0xB0DD, 2);
        for _ in 0..200 {
            let terms: Vec<f64> = (0..37).map(|_| rng.f64() * 1e3).collect();
            let naive: f64 = -terms.iter().sum::<f64>();
            assert!(
                elbo_from_nelbo(&terms) <= naive,
                "the directed sum must not be the optimistic one"
            );
        }
        assert_eq!(elbo_from_nelbo(&[1.0, f64::INFINITY]), f64::NEG_INFINITY);
        assert!(elbo_from_nelbo(&[f64::NAN]).is_nan());
    }

    /// Malformed input is refused with the variant that names what went wrong.
    #[test]
    fn bad_schedules_and_bad_shapes_are_refused_with_a_named_variant() {
        assert!(matches!(Forward::new(Kernel::Uniform, 4, vec![]), Err(DiffuseError::NoSteps)));
        assert!(matches!(
            Forward::new(Kernel::Uniform, 1, vec![0.5]),
            Err(DiffuseError::TinyVocab { vocab: 1 })
        ));
        assert!(matches!(
            Forward::new(Kernel::Uniform, 3, vec![0.5, 1.4]),
            Err(DiffuseError::BadBeta { at: 2, .. })
        ));
        assert!(matches!(
            Forward::new(Kernel::Uniform, 3, vec![f64::NAN]),
            Err(DiffuseError::BadBeta { at: 1, .. })
        ));
        let f = Forward::absorbing_linear(3, 4).unwrap();
        assert!(SiteReverse::new(4, vec![0.5, 0.5, 0.0, 0.0], vec![]).is_ok());
        assert!(matches!(
            SiteReverse::new(4, vec![0.5, 0.4, 0.0, 0.0], vec![]),
            Err(DiffuseError::NotStochastic { .. })
        ));
        assert!(matches!(
            SiteReverse::exact(&f, &[0.5, 0.5]),
            Err(DiffuseError::Width { got: 2, want: 3 })
        ));
        // A SUBS chain over a schedule that does not absorb is refused.
        let g = Forward::constant(Kernel::Absorbing, 3, 3, 0.4).unwrap();
        let pred: Vec<Vec<f64>> = (0..3).map(|_| vec![1.0 / 3.0; 3]).collect();
        assert!(matches!(
            SiteReverse::subs(&g, &pred),
            Err(DiffuseError::PriorNotAbsorbed { .. })
        ));
        // A table the machine should not be asked to allocate.
        assert!(matches!(
            TableDenoiser::new(&f, 32),
            Err(DiffuseError::TooLarge { .. })
        ));
        // Every variant prints something that names the problem.
        for e in [
            DiffuseError::NoSteps,
            DiffuseError::TinyVocab { vocab: 1 },
            DiffuseError::BadBeta { at: 1, beta: 2.0 },
            DiffuseError::Unreachable { x0: 0, xt: 1, t: 2 },
            DiffuseError::BadToken { at: 0, token: 7, states: 3 },
            DiffuseError::Width { got: 1, want: 2 },
            DiffuseError::NotStochastic { row: 0, sum: 0.9 },
            DiffuseError::TooLarge { cells: 9, limit: 4 },
            DiffuseError::PriorNotAbsorbed { alpha_bar_t: 0.1 },
            DiffuseError::Degenerate { t: 1, xt: 0 },
        ] {
            assert!(!e.to_string().is_empty());
        }
    }
}
