//! Gaussian belief propagation — and the quantity it gets **wrong** on every loop.
//!
//! This is the algorithm the robotics side of this field actually runs. A factor graph of Gaussian
//! factors, messages passed locally between neighbours, no global solve: it is how bundle
//! adjustment is distributed across an IPU's tiles, how the Robot Web has robots agree without a
//! server, and how a pose graph is kept up to date incrementally. A 2026-09-20 survey of this
//! field's applications did not locate **any** attempt to run it on thermodynamic, analog-OU or
//! p-bit hardware, which is why it is here.
//!
//! # The model
//!
//! A Gaussian in INFORMATION form: `p(x) ∝ exp(-½ xᵀ Λ x + ηᵀ x)`, with `Λ` symmetric positive
//! definite and sparse. The posterior is `N(Λ⁻¹η, Λ⁻¹)`, so the mean is a linear solve and the
//! **marginal variance of node `i` is `(Λ⁻¹)_ii`** — the number a robot reports as its uncertainty.
//!
//! # The two classical facts this module exists to hold on to
//!
//! On a TREE, belief propagation is exact: both the means and the variances.
//!
//! On a graph with loops, if it converges, **the means are still exact and the variances are
//! not** (Weiss and Freeman, *Correctness of belief propagation in Gaussian graphical models of
//! arbitrary topology*, Neural Computation 13:2173, 2001). The reason is structural rather than
//! numerical: the fixed point computes a quantity on the graph's unrolled computation tree, which
//! reproduces every walk that contributes to the mean and only some of the walks that contribute
//! to the variance. **No amount of iterating removes it.** `loopy_belief_propagation_gets_the_mean_
//! right_and_the_variance_wrong` is that statement, measured against an exact inverse, in both
//! directions — a test that only checked "close to exact" would pass an implementation that had
//! quietly become a dense solve.
//!
//! # Why this belongs in a thermodynamic-computing crate
//!
//! Because the failure is in exactly the quantity the sampling route gets right. An
//! Ornstein-Uhlenbeck network's stationary covariance **is** `Λ⁻¹` ([`crate::tla`]), so its
//! variance estimates are unbiased and shrink like `1/√samples`, while GBP's error is a constant
//! of the graph. [`crate::apps::marginals`] is the entry point that runs both and prices them, and
//! `the_sampler_converges_past_the_message_passers_floor` is the crossover measured rather than
//! asserted.
//!
//! What this does not claim: that sampling is the cheaper way to get a Gaussian mean. It is not,
//! and arXiv:2608.09743 makes that case at length — the OU dynamics' mean is preconditioned
//! gradient descent, and a deterministic digital method does that better. The claim here is about
//! the second moment.

use crate::tla::Spd;

/// A sparse Gaussian in information form, with scalar nodes.
#[derive(Clone, Debug)]
pub struct Info {
    /// Node count.
    pub n: usize,
    /// The diagonal of `Λ`. Positive; a node with no prior at all leaves the model improper.
    pub diag: Vec<f64>,
    /// Off-diagonal entries `(i, j, Λ_ij)` with `i < j`. Each undirected pair appears once.
    pub edges: Vec<(usize, usize, f64)>,
    /// The information vector `η = Λ μ`.
    pub eta: Vec<f64>,
}

/// Why a model could not be built or solved.
#[derive(Clone, Debug, PartialEq)]
pub enum Ill {
    /// A vector's length disagrees with the node count.
    Shape,
    /// An edge names a node outside the model, or joins a node to itself.
    Edge(usize, usize),
    /// The same pair appears twice; `Λ_ij` would be ambiguous.
    DuplicateEdge(usize, usize),
    /// A diagonal entry is not positive and finite, so `Λ` cannot be positive definite.
    Diagonal(usize, f64),
    /// The messages did not settle. Carries the largest change on the last iteration: Gaussian BP
    /// does not converge on every model, and a fixed point that was never reached is not one.
    DidNotConverge(f64),
}

impl core::fmt::Display for Ill {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Ill::Shape => write!(f, "a vector's length disagrees with the node count"),
            Ill::Edge(i, j) => write!(f, "the edge ({i}, {j}) is not between two distinct nodes"),
            Ill::DuplicateEdge(i, j) => write!(f, "the pair ({i}, {j}) appears more than once"),
            Ill::Diagonal(i, v) => {
                write!(f, "node {i}'s precision {v} is not positive and finite")
            }
            Ill::DidNotConverge(d) => {
                write!(f, "the messages did not settle; last change {d:e}")
            }
        }
    }
}

impl core::error::Error for Ill {}

/// Per-node beliefs, and what it took to reach them.
#[derive(Clone, Debug)]
pub struct Beliefs {
    /// Posterior mean per node.
    pub mean: Vec<f64>,
    /// Posterior **marginal variance** per node — the one the loop corrupts.
    pub variance: Vec<f64>,
    /// Message-passing sweeps performed.
    pub iterations: usize,
    /// Largest message change on the last sweep.
    pub residual: f64,
}

impl Info {
    /// Check the model, or say which part of it is ill-formed.
    ///
    /// # Errors
    ///
    /// An [`Ill`] naming the offending vector, edge or diagonal entry.
    pub fn check(&self) -> Result<(), Ill> {
        if self.diag.len() != self.n || self.eta.len() != self.n {
            return Err(Ill::Shape);
        }
        for (i, &d) in self.diag.iter().enumerate() {
            if !(d > 0.0) || !d.is_finite() {
                return Err(Ill::Diagonal(i, d));
            }
        }
        let mut seen = std::collections::BTreeSet::new();
        for &(i, j, _) in &self.edges {
            if i == j || i >= self.n || j >= self.n {
                return Err(Ill::Edge(i, j));
            }
            let key = (i.min(j), i.max(j));
            if !seen.insert(key) {
                return Err(Ill::DuplicateEdge(key.0, key.1));
            }
        }
        Ok(())
    }

    /// The model as a dense system `Λ x = η`, for the exact route and for [`crate::tla`].
    ///
    /// # Errors
    ///
    /// An [`Ill`], from [`Info::check`].
    pub fn to_spd(&self) -> Result<Spd, Ill> {
        self.check()?;
        let n = self.n;
        let mut a = vec![0.0; n * n];
        for (i, &d) in self.diag.iter().enumerate() {
            a[i * n + i] = d;
        }
        for &(i, j, w) in &self.edges {
            a[i * n + j] = w;
            a[j * n + i] = w;
        }
        Ok(Spd::new(n, a, self.eta.clone()))
    }

    /// The exact posterior: `Λ⁻¹η` and the diagonal of `Λ⁻¹`, by Gauss-Jordan.
    ///
    /// The oracle everything in this module is held to. `O(n³)`, so it is a reference and not a
    /// method — which is the whole reason message passing exists.
    ///
    /// # Errors
    ///
    /// An [`Ill`], or a message if `Λ` turned out to be singular.
    pub fn exact(&self) -> Result<(Vec<f64>, Vec<f64>), String> {
        self.check().map_err(|e| e.to_string())?;
        let n = self.n;
        let spd = self.to_spd().map_err(|e| e.to_string())?;
        // [A | I] -> [I | A^-1], with partial pivoting.
        let mut aug = vec![0.0; n * 2 * n];
        for i in 0..n {
            for j in 0..n {
                aug[i * 2 * n + j] = spd.a[i * n + j];
            }
            aug[i * 2 * n + n + i] = 1.0;
        }
        for col in 0..n {
            let mut piv = col;
            for r in col + 1..n {
                if aug[r * 2 * n + col].abs() > aug[piv * 2 * n + col].abs() {
                    piv = r;
                }
            }
            if aug[piv * 2 * n + col].abs() < 1e-300 {
                return Err(format!("the precision matrix is singular at column {col}"));
            }
            if piv != col {
                for k in 0..2 * n {
                    aug.swap(col * 2 * n + k, piv * 2 * n + k);
                }
            }
            let p = aug[col * 2 * n + col];
            for k in 0..2 * n {
                aug[col * 2 * n + k] /= p;
            }
            for r in 0..n {
                if r == col {
                    continue;
                }
                let f = aug[r * 2 * n + col];
                if f == 0.0 {
                    continue;
                }
                for k in 0..2 * n {
                    aug[r * 2 * n + k] -= f * aug[col * 2 * n + k];
                }
            }
        }
        let variance: Vec<f64> = (0..n).map(|i| aug[i * 2 * n + n + i]).collect();
        let mean: Vec<f64> = (0..n)
            .map(|i| (0..n).map(|j| aug[i * 2 * n + n + j] * self.eta[j]).sum())
            .collect();
        Ok((mean, variance))
    }

    /// Gaussian belief propagation to a fixed point.
    ///
    /// `damping` in `[0, 1)` mixes each new message with the last; `0` is the undamped update.
    /// Damping changes which models converge and **does not change the fixed point**, which is why
    /// it cannot rescue the variance.
    ///
    /// # Errors
    ///
    /// [`Ill::DidNotConverge`] if the residual is still above `tol` after `max_iters` sweeps. Not
    /// converging is a real outcome for this algorithm, and reporting the last iterate as a belief
    /// would be reporting a number that means nothing.
    pub fn belief_propagation(
        &self,
        max_iters: usize,
        tol: f64,
        damping: f64,
    ) -> Result<Beliefs, Ill> {
        self.check()?;
        let n = self.n;
        // Directed message slots: 2 per undirected edge. `out[i]` lists (slot, other end).
        let mut adj: Vec<Vec<(usize, usize, f64)>> = vec![Vec::new(); n];
        let mut m_prec = Vec::new();
        let mut m_info = Vec::new();
        for &(i, j, w) in &self.edges {
            let s_ij = m_prec.len();
            m_prec.push(0.0);
            m_info.push(0.0);
            let s_ji = m_prec.len();
            m_prec.push(0.0);
            m_info.push(0.0);
            // (slot this node SENDS on, the other end, the coupling)
            adj[i].push((s_ij, j, w));
            adj[j].push((s_ji, i, w));
        }
        // The slots a node RECEIVES on. Slot `2k` carries edges[k].0 -> edges[k].1 and `2k+1` the
        // reverse, so a message and its reply are siblings under `^ 1` -- which is what makes the
        // cavity below one subtraction instead of a loop over the neighbourhood.
        let mut incoming: Vec<Vec<usize>> = vec![Vec::new(); n];
        for (k, &(i, j, _)) in self.edges.iter().enumerate() {
            incoming[j].push(2 * k);
            incoming[i].push(2 * k + 1);
        }
        let mut residual = f64::INFINITY;
        let mut iterations = 0;
        for _ in 0..max_iters {
            iterations += 1;
            residual = 0.0;
            for i in 0..n {
                // Totals over everything arriving at i, so each message costs O(1) rather than
                // O(degree) to exclude one term.
                let mut tot_p = self.diag[i];
                let mut tot_e = self.eta[i];
                for &s in &incoming[i] {
                    tot_p += m_prec[s];
                    tot_e += m_info[s];
                }
                for &(send, _, w) in &adj[i] {
                    let back = send ^ 1; // the message arriving at i from that same neighbour
                    let cav_p = tot_p - m_prec[back];
                    let cav_e = tot_e - m_info[back];
                    // A cavity precision at or below zero is not a Gaussian; leave the message
                    // where it is and let the residual report that nothing settled.
                    if !(cav_p > 0.0) || !cav_p.is_finite() {
                        residual = f64::INFINITY;
                        continue;
                    }
                    let np = -w * w / cav_p;
                    let ne = -w * cav_e / cav_p;
                    let dp = np - m_prec[send];
                    let de = ne - m_info[send];
                    residual = residual.max(dp.abs()).max(de.abs());
                    m_prec[send] += (1.0 - damping) * dp;
                    m_info[send] += (1.0 - damping) * de;
                }
            }
            if residual <= tol {
                break;
            }
        }
        if !(residual <= tol) {
            return Err(Ill::DidNotConverge(residual));
        }
        let mut mean = vec![0.0; n];
        let mut variance = vec![0.0; n];
        for i in 0..n {
            let mut p = self.diag[i];
            let mut e = self.eta[i];
            for &s in &incoming[i] {
                p += m_prec[s];
                e += m_info[s];
            }
            mean[i] = e / p;
            variance[i] = 1.0 / p;
        }
        Ok(Beliefs { mean, variance, iterations, residual })
    }
}

/// A `w × h` grid of scalar nodes: each node coupled to its neighbours, each with a unit prior.
///
/// The shape a pose graph or an occupancy map has, and the smallest thing with loops in it.
/// `coupling` is the off-diagonal `Λ_ij`; `prior` the diagonal. Diagonal dominance —
/// `prior > degree · |coupling|` — is what keeps `Λ` positive definite and message passing
/// convergent, and [`grid`] does not enforce it, because the models where it fails are worth
/// building on purpose.
#[must_use]
pub fn grid(w: usize, h: usize, prior: f64, coupling: f64) -> Info {
    let n = w * h;
    let mut edges = Vec::new();
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            if x + 1 < w {
                edges.push((i, i + 1, coupling));
            }
            if y + 1 < h {
                edges.push((i, i + w, coupling));
            }
        }
    }
    Info { n, diag: vec![prior; n], edges, eta: vec![0.0; n] }
}

/// A chain of `n` scalar nodes — a tree, so belief propagation is exact on it.
#[must_use]
pub fn chain(n: usize, prior: f64, coupling: f64) -> Info {
    let edges = (0..n.saturating_sub(1)).map(|i| (i, i + 1, coupling)).collect();
    Info { n, diag: vec![prior; n], edges, eta: vec![0.0; n] }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn with_information(mut m: Info, seed: u64) -> Info {
        let mut rng = crate::rng::Pcg::new(seed, 0x6B9);
        for e in &mut m.eta {
            *e = rng.f64() * 2.0 - 1.0;
        }
        m
    }

    /// The control. On a TREE, belief propagation is exact in BOTH moments — so an implementation
    /// that is merely wrong everywhere cannot pass the loopy test below by accident.
    #[test]
    fn on_a_tree_belief_propagation_is_exact_in_both_moments() {
        for (n, prior, coupling) in [(2usize, 1.0, -0.4), (8, 1.5, -0.6), (25, 2.0, 0.9)] {
            let m = with_information(chain(n, prior, coupling), 0x11 + n as u64);
            let (mean, var) = m.exact().expect("a chain is well conditioned here");
            let b = m.belief_propagation(500, 1e-13, 0.0).expect("a tree converges");
            for i in 0..n {
                assert!(
                    (b.mean[i] - mean[i]).abs() < 1e-9,
                    "chain of {n}, node {i}: mean {} vs exact {}",
                    b.mean[i],
                    mean[i]
                );
                assert!(
                    (b.variance[i] - var[i]).abs() < 1e-9,
                    "chain of {n}, node {i}: variance {} vs exact {}",
                    b.variance[i],
                    var[i]
                );
            }
        }
    }

    /// **THE FINDING.** On a graph with loops, a converged Gaussian BP has the mean EXACTLY right
    /// and the variance wrong — and the error does not shrink with more iterations, because it is
    /// a property of the fixed point and not of the path to it.
    ///
    /// Asserted in both directions on purpose. "BP is close to exact" would pass for a dense solve
    /// wearing a message-passing costume, and "BP is wrong" would pass for an implementation whose
    /// means were wrong too — which is the ordinary way to get this algorithm wrong.
    #[test]
    fn loopy_belief_propagation_gets_the_mean_right_and_the_variance_wrong() {
        let m = with_information(grid(5, 5, 1.0, -0.22), 0xA11);
        let (mean, var) = m.exact().expect("diagonally dominant");

        let mut worst_mean = 0.0f64;
        let mut worst_var = 0.0f64;
        let mut underestimates = 0usize;
        // Ten times the iterations, and the same answer: the error is the fixed point's.
        let mut at = Vec::new();
        for iters in [200usize, 2_000, 20_000] {
            let b = m.belief_propagation(iters, 1e-14, 0.0).expect("this grid converges");
            worst_mean = 0.0;
            worst_var = 0.0;
            underestimates = 0;
            for i in 0..m.n {
                worst_mean = worst_mean.max((b.mean[i] - mean[i]).abs());
                worst_var = worst_var.max((b.variance[i] - var[i]).abs());
                if b.variance[i] < var[i] {
                    underestimates += 1;
                }
            }
            at.push(worst_var);
        }
        assert!(worst_mean < 1e-9, "the means must be EXACT on a converged run: {worst_mean:e}");
        assert!(
            worst_var > 1e-3,
            "the variances must be wrong, and by a margin no tolerance explains: {worst_var:e}"
        );
        // A hundred times the iterations changes it by nothing: this is a floor, not a residual.
        assert!(
            (at[2] - at[0]).abs() < 1e-9,
            "the variance error moved with the iteration count, so it was a residual: {at:?}"
        );
        // And the direction, which is the walk-sum statement: the computation tree omits walks
        // that carry variance, so what is left is too confident.
        assert_eq!(
            underestimates, m.n,
            "every node's variance should be an UNDERESTIMATE on this model"
        );
        eprintln!(
            "5x5 grid: worst mean error {worst_mean:e}, worst variance error {worst_var:.5}, \
             {underestimates}/{} nodes overconfident",
            m.n
        );
    }

    /// **AND THE CROSSOVER.** The sampling route's variance error is a standard error and falls
    /// like `1/sqrt(samples)`; message passing's is a constant of the graph. So there is a budget
    /// past which the sampler is simply more accurate on the quantity a robot reports, and this
    /// test finds it rather than assuming it.
    #[test]
    fn the_sampler_converges_past_the_message_passers_floor() {
        let m = with_information(grid(5, 5, 1.0, -0.22), 0xA11);
        let (_, var) = m.exact().expect("diagonally dominant");
        let spd = m.to_spd().expect("well formed");
        let bp = m.belief_propagation(20_000, 1e-14, 0.0).expect("converges");
        let bp_err = (0..m.n)
            .map(|i| (bp.variance[i] - var[i]).abs())
            .fold(0.0f64, f64::max);

        // The exact-transition OU sampler: unbiased, so only its standard error stands between it
        // and the truth. `stride_h` of a few relaxation times gives near-independent samples.
        let mut errs = Vec::new();
        for samples in [1_000usize, 10_000, 100_000] {
            let r = crate::tla::solve_spd_exact_ou(&spd, 1.0, 3.0, 200, samples, 0xC0FFEE);
            let e = (0..m.n)
                .map(|i| (r.a_inv[i * m.n + i] - var[i]).abs())
                .fold(0.0f64, f64::max);
            errs.push(e);
        }
        // It must actually be converging, not merely small: ten times the samples buys about
        // sqrt(10) = 3.16x, and asserting only the last would pass for a biased estimator that
        // happened to sit near the answer.
        assert!(errs[0] > errs[2] * 2.0, "the sampler is not converging: {errs:?}");
        // The crossover itself.
        assert!(
            errs[2] < bp_err,
            "at 1e5 samples the sampler's worst variance error {:.5} should be under message \
             passing's floor {bp_err:.5}",
            errs[2]
        );
        eprintln!(
            "worst variance error -- message passing {bp_err:.5} (a floor), \
             sampling {:.5} / {:.5} / {:.5} at 1e3 / 1e4 / 1e5 samples",
            errs[0], errs[1], errs[2]
        );
    }

    /// Not converging is an outcome, and it is reported rather than returned as a belief. A grid
    /// coupled past diagonal dominance is the ordinary way a real pose graph does this.
    #[test]
    fn a_model_that_does_not_settle_is_refused_rather_than_reported() {
        let m = with_information(grid(6, 6, 1.0, -0.9), 0xBAD);
        let out = m.belief_propagation(500, 1e-12, 0.0);
        assert!(
            matches!(out, Err(Ill::DidNotConverge(_))),
            "an over-coupled grid must not return a fixed point it never reached"
        );
        // Damping changes WHICH models converge; when it rescues one, the fixed point it reaches
        // is the same one, so the variance error is unchanged. That is the claim damping cannot
        // touch, checked on a model where damping does make the difference.
        let hard = with_information(grid(5, 5, 1.0, -0.45), 0xD00D);
        if hard.belief_propagation(2_000, 1e-13, 0.0).is_err()
            && let Ok(damped) = hard.belief_propagation(200_000, 1e-13, 0.7)
        {
            {
                let (_, var) = hard.exact().expect("still positive definite");
                let err = (0..hard.n)
                    .map(|i| (damped.variance[i] - var[i]).abs())
                    .fold(0.0f64, f64::max);
                assert!(err > 1e-3, "damping must not fix the variance: {err:e}");
            }
        }
    }

    /// Every way the model itself can be ill-formed.
    #[test]
    fn an_ill_formed_model_is_named_rather_than_solved() {
        let base = grid(3, 3, 1.0, -0.2);
        let mut bad = base.clone();
        bad.eta.pop();
        assert_eq!(bad.check(), Err(Ill::Shape));
        let mut bad = base.clone();
        bad.diag[4] = 0.0;
        assert_eq!(bad.check(), Err(Ill::Diagonal(4, 0.0)));
        let mut bad = base.clone();
        bad.diag[1] = f64::NAN;
        assert!(matches!(bad.check(), Err(Ill::Diagonal(1, _))));
        let mut bad = base.clone();
        bad.edges.push((2, 2, 0.1));
        assert_eq!(bad.check(), Err(Ill::Edge(2, 2)));
        let mut bad = base.clone();
        bad.edges.push((0, 99, 0.1));
        assert_eq!(bad.check(), Err(Ill::Edge(0, 99)));
        // A duplicated pair would make `Lambda_ij` ambiguous: the dense build takes the last and
        // message passing takes both, so the two routes would silently solve different models.
        let mut bad = base.clone();
        bad.edges.push((0, 1, 0.3));
        assert_eq!(bad.check(), Err(Ill::DuplicateEdge(0, 1)));
        assert!(base.check().is_ok());
    }
}
