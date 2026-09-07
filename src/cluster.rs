//! Cluster updates: Swendsen–Wang and Wolff, and the gauge that says when they apply.
//!
//! Every sampler in this crate flips one spin at a time. [`crate::gibbs::Sampler`] is chromatic
//! single-spin Gibbs, and [`crate::tempering`], [`crate::adaptive`], [`crate::popanneal`] and
//! [`crate::sqa`] all schedule that same kernel; the only genuine alternative move is Houdayer's in
//! [`crate::icm`], which is isoenergetic and restricted to zero-field glasses. So on the attractive
//! models most of this crate's verification is built from, there was no cluster update at all.
//!
//! That costs where it is most expensive to pay. Near a critical point a single-spin sampler's
//! autocorrelation time grows as `L^z` with `z ≈ 2.17` in two dimensions — critical slowing down —
//! because flipping a domain of linear size `ξ` one spin at a time is a random walk against the
//! surface tension holding it together. A cluster update builds the correlated region and flips it
//! whole. `examples/critical_slowdown.rs` measures both at `beta_c` rather than citing them: `z` of
//! 2.06 for single-spin Gibbs against 0.29 for Swendsen–Wang and 0.46 for Wolff, and at `L = 32` an
//! independent sample costs 103,274 spin visits under Gibbs and 1,352 under Wolff.
//!
//! # When these algorithms are valid, and it is not "no negative couplings"
//!
//! Swendsen–Wang and Wolff need a **ferromagnetic** model. The usual statement is that they do not
//! work on a spin glass, which is true and too weak: what is actually required is that the model can
//! be *gauged* to a ferromagnet. Flipping the sign convention of a spin, `s_i → σ_i s_i` with
//! `σ_i = ±1`, maps `J_ij → σ_i σ_j J_ij` and `h_i → σ_i h_i` and leaves every energy alone — it is
//! a relabelling of the state space, not a different model. A gauge making every coupling positive
//! exists exactly when the signed graph is **balanced** (Harary 1953): every cycle has a positive
//! product of couplings.
//!
//! That is decidable in one pass. Two-colour the sign structure by breadth-first search, forcing
//! `σ_j = σ_i · sign(J_ij)` along every edge; a conflict is a cycle with a negative product, and the
//! search hands back the cycle itself. So [`gauge`] both decides validity and produces the
//! relabelling, which is why this module covers every balanced instance rather than only the
//! literally-ferromagnetic ones — and it costs nothing extra, because the decision procedure and the
//! construction are the same traversal.
//!
//! Balance and frustration are different properties, and confusing them is the easy mistake here. A
//! model can have wildly mixed signs and be balanced: apply a random gauge to a ferromagnet and the
//! result *looks* like a spin glass while being one relabelling away from trivial.
//! `a_relabelled_ferromagnet_is_still_balanced` is that instance, and a validity test that checked
//! for negative couplings would refuse a model this samples exactly.
//!
//! # Fields are an extra vertex, not a special case
//!
//! The moves as stated need a zero-field model, because a field breaks the up/down symmetry a
//! probability-one-half cluster flip relies on. [`with_ghost`] supplies one by a change of
//! variables: couple every biased site to one extra spin at `J_ig = h_i`, and read the physical
//! configuration back as `ŝ_i = s_i · s_g`. That is exact, not approximate — `s_g² = 1` cancels out
//! of both energy terms — and it means the balance test above decides the field case too, with no
//! second algorithm and no second code path.
//!
//! It also gives a sharper answer than the folklore. A cycle through the ghost has product
//! `J_ij h_i h_j`, so:
//!
//!   - a ferromagnet under a field of ONE sign is balanced, and is sampled here — "Swendsen–Wang
//!     cannot handle a field" is true of the move as literally stated and false of the model;
//!   - a ferromagnet under MIXED-sign fields is genuinely unbalanced, and is refused with a cycle
//!     through the ghost naming the two sites whose fields disagree across a coupling.
//!
//! # The witness for a refusal is a cycle, not an edge
//!
//! When no gauge exists, the reason is a specific cycle, and [`Frustrated`] carries it. A single
//! edge is never the obstruction — any one edge can be satisfied by choosing a sign — so naming one
//! would be pointing at an innocent bystander. A cycle is a proof, and a caller can check it without
//! trusting this module: multiply the couplings around it and the product is negative. No gauge can
//! change that, because a closed walk enters and leaves every vertex once and so multiplies each
//! `σ_i` in exactly twice.

use crate::graph::Graph;
use crate::rng::Pcg;

/// No gauge makes this model ferromagnetic, and here is the cycle that proves it.
///
/// Balance is exactly the absence of a negative-product cycle (Harary), so this is a certificate
/// rather than a complaint. See the module documentation for why a cycle and not an edge.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Frustrated {
    /// The cycle, as vertices in order. The last is adjacent to the first.
    pub cycle: Vec<usize>,
    /// The ghost spin's index, when the cycle runs through it.
    ///
    /// A cycle through the ghost is a statement about FIELDS: the ghost's incident couplings are
    /// the `h_i`, so such a cycle says two sites' fields disagree in sign around a loop of real
    /// couplings. The index is one past the caller's last node and is not a node of their graph,
    /// which is why it is named here rather than left to be inferred from a number that looks like
    /// an ordinary vertex. See [`Frustrated::product`].
    pub ghost: Option<usize>,
}

impl Frustrated {
    /// The product of the couplings around the cycle. Negative whenever this type exists.
    ///
    /// The witness made executable. A refusal that hands back a cycle is only worth more than a
    /// complaint if the caller can check it without trusting the code that produced it, and asking
    /// every caller to reimplement "multiply the couplings, and use `h_i` for the ghost's edges" is
    /// how a checkable claim becomes an unchecked one.
    ///
    /// # Errors
    ///
    /// `None` if the cycle is not a cycle of `g` — consecutive vertices that are not adjacent, or a
    /// walk that does not close. That is a bug in this module rather than a property of the model,
    /// and returning `None` lets a test say so.
    #[must_use]
    pub fn product(&self, g: &Graph) -> Option<f64> {
        let coupling = |i: usize, j: usize| -> Option<f64> {
            match self.ghost {
                // A ghost edge's weight IS the field of the site at the other end.
                Some(gh) if i == gh => g.h.get(j).copied(),
                Some(gh) if j == gh => g.h.get(i).copied(),
                _ => (*g.offset.get(i)?..*g.offset.get(i + 1)?)
                    .find(|&k| g.nbr[k] as usize == j)
                    .map(|k| g.w[k]),
            }
        };
        let mut prod = 1.0f64;
        for w in self.cycle.windows(2) {
            prod *= coupling(w[0], w[1])?;
        }
        Some(prod * coupling(*self.cycle.last()?, self.cycle[0])?)
    }
}

impl core::fmt::Display for Frustrated {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "no gauge makes this model ferromagnetic: the {}-cycle {:?} has a negative coupling \
             product, and a gauge multiplies each vertex's sign in twice around a closed walk, so \
             no relabelling can change it",
            self.cycle.len(),
            self.cycle
        )
    }
}

impl core::error::Error for Frustrated {}

/// The gauge that makes every coupling of `g` positive, or the cycle proving there is none.
///
/// Returns `σ` with `σ_i = ±1` such that `σ_i σ_j J_ij ≥ 0` on every edge. Pass it to
/// [`apply_gauge`] to get the relabelled model, and multiply elementwise to carry a state back.
///
/// Runs in `O(n + m)`: one breadth-first search over the sign structure.
///
/// # Errors
///
/// [`Frustrated`], carrying a cycle whose coupling product is negative.
///
/// # A zero coupling imposes nothing
///
/// An edge of weight zero constrains no signs and is skipped. Treating it as a constraint would
/// invent a frustration the model does not have, and refuse instances that are perfectly samplable.
pub fn gauge(g: &Graph) -> Result<Vec<i8>, Frustrated> {
    let mut sigma = vec![0i8; g.n];
    // Parent pointers, so a conflict can be turned back into the cycle that caused it.
    let mut parent = vec![usize::MAX; g.n];
    let mut queue = std::collections::VecDeque::new();

    for root in 0..g.n {
        if sigma[root] != 0 {
            continue;
        }
        sigma[root] = 1;
        queue.push_back(root);
        while let Some(i) = queue.pop_front() {
            for k in g.offset[i]..g.offset[i + 1] {
                let j = g.nbr[k] as usize;
                let w = g.w[k];
                if w == 0.0 {
                    continue;
                }
                // For σ_i σ_j J_ij to be positive: σ_j is σ_i when J > 0, and −σ_i when J < 0.
                let want = if w > 0.0 { sigma[i] } else { -sigma[i] };
                if sigma[j] == 0 {
                    sigma[j] = want;
                    parent[j] = i;
                    queue.push_back(j);
                } else if sigma[j] != want {
                    return Err(Frustrated { cycle: cycle_through(&parent, i, j), ghost: None });
                }
            }
        }
    }
    Ok(sigma)
}

/// The cycle closed by the edge `i–j` in a search tree: up from each end to where they meet.
fn cycle_through(parent: &[usize], i: usize, j: usize) -> Vec<usize> {
    let path_from_root = |mut v: usize| -> Vec<usize> {
        let mut p = vec![v];
        while parent[v] != usize::MAX {
            v = parent[v];
            p.push(v);
        }
        p.reverse();
        p
    };
    let (ri, rj) = (path_from_root(i), path_from_root(j));
    // The deepest shared ancestor: walk both down from the root and stop where they part.
    let mut meet = 0usize;
    while meet + 1 < ri.len() && meet + 1 < rj.len() && ri[meet + 1] == rj[meet + 1] {
        meet += 1;
    }
    // Down-then-up: i back to the meeting point, then out to j. The edge j–i closes it.
    let mut out: Vec<usize> = ri[meet..].iter().rev().copied().collect();
    out.extend(rj[meet + 1..].iter().copied());
    out
}

/// `g` with `sigma` applied: every coupling positive, every energy unchanged.
///
/// `s → σ ⊙ s` is a bijection on states under which `E` is invariant, so this is the same model in
/// different coordinates. Fields transform as `h_i → σ_i h_i`, which is what keeps it a relabelling
/// rather than a different Hamiltonian — `a_gauge_changes_coordinates_and_not_energies` checks that
/// over the whole state space.
#[must_use]
pub fn apply_gauge(g: &Graph, sigma: &[i8]) -> Graph {
    let mut b = crate::graph::GraphBuilder::new(g.n);
    for i in 0..g.n {
        if g.h[i] != 0.0 {
            b.set_bias(i, g.h[i] * f64::from(sigma[i]));
        }
        for k in g.offset[i]..g.offset[i + 1] {
            let j = g.nbr[k] as usize;
            if j > i {
                b.couple(i, j, g.w[k] * f64::from(sigma[i]) * f64::from(sigma[j]));
            }
        }
    }
    b.build()
}

/// `g` with its fields turned into couplings to one extra spin, or `g` unchanged if it has none.
///
/// The ghost-spin construction. Add a spin `g` and couple it to every biased site at `J_ig = h_i`,
/// giving a zero-field model on `n + 1` spins whose energy is
/// `E'(s) = -Σ J_ij s_i s_j - Σ h_i s_i s_g`. Read the physical configuration back as
/// `ŝ_i = s_i · s_g` and `E(ŝ) = E'(s)` exactly, because `s_g² = 1` cancels out of both terms. So
/// this is not an approximation of a field, it is a change of variables that removes one.
///
/// Returns the ghost's index, which is one past the caller's last node, or `None` when the model
/// carried no field and none was added — a ghost coupled at zero would be an isolated spin doing
/// nothing but perturbing the random stream.
///
/// # What this buys, and what it does not
///
/// The point is that the SAME balance test then decides validity, with no second algorithm for the
/// field case. And it does not simply wave fields through: a cycle through the ghost has product
/// `J_ij h_i h_j`, so a ferromagnet under a field of ONE sign is balanced and samplable — which the
/// usual "Swendsen–Wang cannot handle a field" denies — while a ferromagnet under mixed-sign fields
/// is genuinely unbalanced and is refused, with a cycle through the ghost naming the two sites whose
/// fields disagree.
#[must_use]
pub fn with_ghost(g: &Graph) -> (Graph, Option<usize>) {
    let ghost = g.h.iter().any(|&h| h != 0.0).then_some(g.n);
    let mut b = crate::graph::GraphBuilder::new(g.n + usize::from(ghost.is_some()));
    for i in 0..g.n {
        for k in g.offset[i]..g.offset[i + 1] {
            let j = g.nbr[k] as usize;
            if j > i {
                b.couple(i, j, g.w[k]);
            }
        }
        if let Some(gh) = ghost
            && g.h[i] != 0.0
        {
            b.couple(i, gh, g.h[i]);
        }
    }
    (b.build(), ghost)
}

/// Which cluster move to make.
///
/// Both build clusters from the same bond probability `1 − exp(−2βJ)` on aligned edges, and both
/// leave the Boltzmann distribution invariant with acceptance one. They differ in what they build.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Update {
    /// Swendsen–Wang: decompose the whole lattice into clusters, flip each with probability one
    /// half, independently. Every spin is touched every sweep.
    SwendsenWang,
    /// Wolff: grow one cluster from a random seed and flip it with probability one.
    ///
    /// The seed is uniform over spins, so a cluster is reached in proportion to its size. That bias
    /// is the point: the single cluster is drawn from the size-weighted distribution over
    /// Swendsen–Wang's clusters, so Wolff spends its work on the large ones and skips the
    /// singletons Swendsen–Wang pays to enumerate. It usually wins at criticality for that reason.
    Wolff,
}

/// What one sweep did, for the ledger and for judging whether the move earned its cost.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ClusterStats {
    /// Clusters the sweep formed. For [`Update::Wolff`], the number of single-cluster steps taken.
    pub clusters: usize,
    /// Size of the largest, which is the whole reason to run a cluster algorithm.
    pub largest: usize,
    /// Spins whose value changed.
    pub flipped: u64,
    /// Bonds whose two endpoints the sweep compared, which is what a sweep actually costs.
    ///
    /// The same thing under both moves, deliberately. Swendsen–Wang examines every edge once;
    /// Wolff examines every edge from a cluster member to a site not already in the cluster,
    /// whether or not it turns out to be aligned. Counting only the edges that reached an RNG draw
    /// would make the field mean one thing for one move and another for the other, and a cost
    /// comparison built on it would be measuring the definition.
    pub bonds_tested: u64,
    /// Spins visited. A Swendsen–Wang sweep visits every spin by construction; a Wolff sweep visits
    /// at least `n` by the definition in [`Sampler::sweep_with`], and this reports what it did.
    pub visited: u64,
}

/// A cluster sampler over a balanced model.
///
/// # The acceptance probability, and where it went
///
/// For each edge whose two spins are **aligned**, open a bond with probability `1 − exp(−2βJ)`.
/// Clusters are the connected components of the opened bonds. Flipping a cluster costs no energy
/// across its open bonds, and the bond probability is exactly the number that makes the forward and
/// reverse proposals balance across the *closed* ones — which is why there is no accept/reject step
/// and a cluster of any size moves in a single move. That is Fortuin–Kasteleyn: the algorithm is a
/// Markov chain on a joint spin-and-bond model whose spin marginal is Boltzmann.
///
/// A misaligned edge never opens, so a cluster never crosses a domain wall, and at criticality the
/// clusters *are* the correlated regions.
///
/// # Fields, and why there is no separate code path for them
///
/// A field breaks the up/down symmetry a probability-one-half cluster flip relies on, so the moves
/// as stated need a zero-field model. [`with_ghost`] supplies one by a change of variables rather
/// than an approximation, and the SAME balance test then decides validity on the augmented graph.
/// A field is therefore not a special case here; it is an extra vertex.
pub struct Sampler<'g> {
    graph: &'g Graph,
    beta: f64,
    /// The gauge that made the couplings positive, kept so states are reported in the caller's own
    /// convention rather than in the one the algorithm needed.
    sigma: Vec<i8>,
    gauged: Graph,
    /// Current state, in GAUGED coordinates.
    s: Vec<i8>,
    rng: Pcg,
    /// Scratch reused across sweeps so a sweep does not allocate.
    stack: Vec<usize>,
    in_cluster: Vec<bool>,
    /// Wolff steps per sweep. A CONSTANT, for the reason in [`Sampler::with_wolff_steps`].
    wolff_steps: usize,
    /// The ghost spin's index, when the model had fields. See [`with_ghost`].
    ghost: Option<usize>,
}

impl<'g> Sampler<'g> {
    /// A sampler over `g` at inverse temperature `beta`.
    ///
    /// # Errors
    ///
    /// [`Frustrated`] when no gauge makes `g` ferromagnetic once its fields are absorbed into a
    /// ghost spin — the algorithm is not valid there, and running it anyway would sample a
    /// distribution that is not the model's. The cycle is the witness, and
    /// [`Frustrated::product`] checks it against `g` directly.
    pub fn new(g: &'g Graph, beta: f64, seed: u64) -> Result<Sampler<'g>, Frustrated> {
        let (augmented, ghost) = with_ghost(g);
        let sigma = gauge(&augmented).map_err(|mut f| {
            f.ghost = ghost;
            f
        })?;
        let gauged = apply_gauge(&augmented, &sigma);
        let n = augmented.n;
        let mut rng = Pcg::new(seed, 0x0C_1057);
        let s = (0..n).map(|_| rng.spin(0.5)).collect();
        Ok(Sampler {
            graph: g,
            beta,
            sigma,
            gauged,
            s,
            rng,
            stack: Vec::new(),
            in_cluster: vec![false; n],
            wolff_steps: 1,
            ghost,
        })
    }

    /// How many Wolff steps make one sweep. Default 1.
    ///
    /// # This number must not depend on the state, and finding that out cost a wrong answer
    ///
    /// A Wolff step flips one cluster, so to compare Wolff against a single-spin sweep you want
    /// enough steps to touch every spin. The obvious way to write that is to keep stepping until
    /// `n` spins have been visited — and it is wrong. The number of steps is then a function of the
    /// cluster sizes, which are a function of the state: an ordered configuration produces large
    /// clusters and ends the sweep in fewer steps. Sampling at a stopping time that depends on the
    /// trajectory does not preserve the stationary distribution, however correct the kernel being
    /// stopped. It is optional stopping, and it is invisible — every individual move is exactly
    /// right.
    ///
    /// It is not a small effect. On `ring(10)` at `beta = 0.4`, where enumeration gives
    /// `<E> = -3.8009`, that rule returned `-4.3262`: too ordered, because sweeps end preferentially
    /// just after a large cluster flip. Any FIXED count reproduces the exact value — 1, 3 and 7
    /// steps give -3.7905, -3.7899 and -3.7982. `certify` caught it as a sampled `beta` of 0.4492
    /// against a requested 0.4000, six standard errors out, while the total-variation distance
    /// stayed under its noise floor and reported nothing.
    ///
    /// So the count is frozen here, and comparability is bought from the ledger instead: `visited`
    /// and `bonds_tested` are what the sweep actually did, which is a measurement rather than a
    /// convention, and cannot be biased by what it measures.
    ///
    /// # Panics
    ///
    /// If `k` is zero: a sweep that takes no steps is not a sweep.
    #[must_use]
    pub fn with_wolff_steps(mut self, k: usize) -> Sampler<'g> {
        assert!(k > 0, "a Wolff sweep of zero steps never moves");
        self.wolff_steps = k;
        self
    }

    /// The current state, in the CALLER's sign convention and over the CALLER's spins.
    ///
    /// Two changes of variable are undone here. The gauge, by multiplying through by `σ`; and the
    /// ghost, by reading every spin relative to it — `ŝ_i = s_i · s_g` — and dropping it, so the
    /// result has the caller's width whether or not their model had fields.
    #[must_use]
    pub fn state(&self) -> Vec<i8> {
        let a = self.s.iter().zip(&self.sigma).map(|(&v, &sg)| v * sg);
        match self.ghost {
            None => a.collect(),
            Some(gh) => {
                let all: Vec<i8> = a.collect();
                let g_val = all[gh];
                all[..gh].iter().map(|&v| v * g_val).collect()
            }
        }
    }

    /// Its energy under the original graph.
    #[must_use]
    pub fn energy(&self) -> f64 {
        self.graph.energy(&self.state())
    }

    /// The probability a bond opens on an edge of weight `w`.
    #[inline]
    fn p_open(&self, w: f64) -> f64 {
        1.0 - (-2.0 * self.beta * w).exp()
    }

    /// One Swendsen–Wang sweep, charged to `ledger`.
    pub fn sweep(&mut self, ledger: Option<&mut crate::ledger::Ledger>) -> ClusterStats {
        self.sweep_with(Update::SwendsenWang, ledger)
    }

    /// One sweep with the given move.
    ///
    /// For [`Update::SwendsenWang`] that is one decomposition of the lattice, which touches every
    /// spin. For [`Update::Wolff`] it is [`Sampler::with_wolff_steps`] single-cluster steps, a count
    /// fixed in advance — read that method for why it cannot be derived from the clusters the sweep
    /// happens to build.
    pub fn sweep_with(
        &mut self,
        update: Update,
        ledger: Option<&mut crate::ledger::Ledger>,
    ) -> ClusterStats {
        let stats = match update {
            Update::SwendsenWang => self.sw_sweep(),
            Update::Wolff => self.wolff_sweep(),
        };
        if let Some(l) = ledger {
            // Charged for what it touched, not for the sweep count. A cluster update's win is in
            // mixing per sweep, and the ledger exists to stop a method being flattered by the unit
            // it reports in: `visited` is spins actually read and `bonds_tested` is bonds actually
            // examined, so a move that does less work is billed less and one that does more cannot
            // hide it.
            l.samples += stats.visited;
            l.reads += stats.bonds_tested;
            l.writes += stats.flipped;
        }
        stats
    }

    /// Swendsen–Wang: decompose everything, flip each component on its own coin.
    fn sw_sweep(&mut self) -> ClusterStats {
        let n = self.gauged.n;
        let mut uf = Uf::new(n);
        let mut bonds = 0u64;

        for i in 0..n {
            for k in self.gauged.offset[i]..self.gauged.offset[i + 1] {
                let j = self.gauged.nbr[k] as usize;
                if j <= i {
                    continue;
                }
                bonds += 1;
                let w = self.gauged.w[k];
                // Aligned only: a misaligned edge is a domain wall and never joins a cluster.
                if w > 0.0 && self.s[i] == self.s[j] && self.rng.spin(self.p_open(w)) > 0 {
                    uf.union(i, j);
                }
            }
        }

        // One coin per cluster, taken by its root, so every member reads the same decision.
        let mut flip = vec![0i8; n];
        for v in 0..n {
            let r = uf.find(v);
            if flip[r] == 0 {
                flip[r] = self.rng.spin(0.5);
            }
        }
        let mut flipped = 0u64;
        let mut sizes = vec![0usize; n];
        for v in 0..n {
            let r = uf.find(v);
            sizes[r] += 1;
            if flip[r] < 0 {
                self.s[v] = -self.s[v];
                flipped += 1;
            }
        }
        ClusterStats {
            clusters: sizes.iter().filter(|&&c| c > 0).count(),
            largest: sizes.iter().copied().max().unwrap_or(0),
            flipped,
            bonds_tested: bonds,
            visited: n as u64,
        }
    }

    /// Wolff: a fixed number of single-cluster steps. See [`Sampler::with_wolff_steps`].
    fn wolff_sweep(&mut self) -> ClusterStats {
        let mut acc = ClusterStats::default();
        for _ in 0..self.wolff_steps {
            let step = self.wolff_step();
            acc.clusters += 1;
            acc.largest = acc.largest.max(step.largest);
            acc.flipped += step.flipped;
            acc.bonds_tested += step.bonds_tested;
            acc.visited += step.visited;
        }
        acc
    }

    /// One Wolff step: grow a cluster from a random seed and flip it.
    ///
    /// Unlike Swendsen–Wang there is no coin at the end — the single cluster flips with probability
    /// one, which is why the move never wastes a proposal.
    pub fn wolff_step(&mut self) -> ClusterStats {
        let n = self.gauged.n;
        let seed = ((self.rng.f64() * n as f64) as usize).min(n - 1);
        let spin = self.s[seed];

        self.stack.clear();
        self.stack.push(seed);
        self.in_cluster[seed] = true;
        let mut members = vec![seed];
        let mut bonds = 0u64;

        while let Some(i) = self.stack.pop() {
            for k in self.gauged.offset[i]..self.gauged.offset[i + 1] {
                let j = self.gauged.nbr[k] as usize;
                if self.in_cluster[j] {
                    continue;
                }
                bonds += 1;
                let w = self.gauged.w[k];
                if w <= 0.0 || self.s[j] != spin {
                    continue;
                }
                if self.rng.spin(self.p_open(w)) > 0 {
                    self.in_cluster[j] = true;
                    members.push(j);
                    self.stack.push(j);
                }
            }
        }

        for &v in &members {
            self.s[v] = -self.s[v];
            self.in_cluster[v] = false;
        }
        ClusterStats {
            clusters: 1,
            largest: members.len(),
            flipped: members.len() as u64,
            bonds_tested: bonds,
            visited: members.len() as u64,
        }
    }

    /// Draw a chain, so [`crate::certify::certify`] applies to it unchanged.
    #[must_use]
    pub fn collect(
        &mut self,
        plan: &crate::samples::Plan,
        update: Update,
    ) -> crate::samples::SampleSet {
        for _ in 0..plan.burn_in {
            self.sweep_with(update, None);
        }
        let thin = plan.thin.max(1);
        let mut states = Vec::with_capacity(plan.draws);
        let mut energies = Vec::with_capacity(plan.draws);
        for _ in 0..plan.draws {
            for _ in 0..thin {
                self.sweep_with(update, None);
            }
            let st = self.state();
            energies.push(self.graph.energy(&st));
            states.push(st);
        }
        crate::samples::SampleSet::from_chain(states, energies, self.beta, plan.burn_in, thin)
    }
}

/// Union-find with path halving and union by size. Enough for one sweep's clusters.
struct Uf {
    parent: Vec<usize>,
    size: Vec<usize>,
}

impl Uf {
    fn new(n: usize) -> Uf {
        Uf { parent: (0..n).collect(), size: vec![1; n] }
    }
    fn find(&mut self, mut v: usize) -> usize {
        while self.parent[v] != v {
            self.parent[v] = self.parent[self.parent[v]];
            v = self.parent[v];
        }
        v
    }
    fn union(&mut self, a: usize, b: usize) {
        let (mut ra, mut rb) = (self.find(a), self.find(b));
        if ra == rb {
            return;
        }
        if self.size[ra] < self.size[rb] {
            core::mem::swap(&mut ra, &mut rb);
        }
        self.parent[rb] = ra;
        self.size[ra] += self.size[rb];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn coupling(g: &Graph, i: usize, j: usize) -> Option<f64> {
        (g.offset[i]..g.offset[i + 1]).find(|&k| g.nbr[k] as usize == j).map(|k| g.w[k])
    }

    /// The product of the couplings around a cycle, which is what balance is about.
    fn cycle_product(g: &Graph, cycle: &[usize]) -> f64 {
        let mut prod = 1.0f64;
        for w in cycle.windows(2) {
            prod *= coupling(g, w[0], w[1]).expect("consecutive cycle vertices must be adjacent");
        }
        prod * coupling(g, *cycle.last().unwrap(), cycle[0]).expect("a cycle must close")
    }

    /// A ferromagnet is balanced, and so is any relabelling of one.
    ///
    /// The second half is the point. Applying a random gauge to a ferromagnet produces a graph of
    /// mixed signs that LOOKS like a spin glass, and balance sees through it. A validity test that
    /// checked for "no negative couplings" would refuse a model this samples exactly, so this is
    /// the test that separates the implemented condition from the one it is often confused with.
    #[test]
    fn a_relabelled_ferromagnet_is_still_balanced() {
        let base = crate::ising::lattice2d(6, 1.0);
        let mut rng = Pcg::new(7, 1);
        let disguise: Vec<i8> = (0..base.n).map(|_| rng.spin(0.5)).collect();
        let disguised = apply_gauge(&base, &disguise);

        assert!(disguised.w.iter().any(|&w| w < 0.0), "the disguise did nothing");

        let sigma = gauge(&disguised).expect("a relabelled ferromagnet is balanced");
        let regauged = apply_gauge(&disguised, &sigma);
        assert!(
            regauged.w.iter().all(|&w| w >= 0.0),
            "the gauge must make every coupling non-negative"
        );
    }

    /// An odd antiferromagnetic ring is not balanced, and the witness is checkable.
    #[test]
    fn an_odd_antiferromagnetic_ring_is_refused_with_a_cycle() {
        for n in [5usize, 7, 9] {
            let g = crate::ising::ring(n, -1.0, 0.0);
            let err = gauge(&g).expect_err("an odd AF ring cannot be gauged ferromagnetic");
            assert!(err.cycle.len() >= 3, "a cycle needs three vertices: {:?}", err.cycle);
            let prod = cycle_product(&g, &err.cycle);
            assert!(prod < 0.0, "the witness cycle must have a negative product, got {prod}");
        }
    }

    /// An even antiferromagnetic ring IS balanced: alternate the gauge and it is a ferromagnet.
    ///
    /// The companion to the odd case. Together they say the refusal tracks the cycle parity balance
    /// is about, not the sign of the couplings, which is identical in both.
    #[test]
    fn an_even_antiferromagnetic_ring_is_balanced() {
        let g = crate::ising::ring(8, -1.0, 0.0);
        let sigma = gauge(&g).expect("an even AF ring is bipartite, so balanced");
        let regauged = apply_gauge(&g, &sigma);
        assert!(regauged.w.iter().all(|&w| w >= 0.0));
    }

    /// The witness is a real cycle in the graph: adjacent, distinct, and closing.
    ///
    /// Without this the refusal could carry any vector of vertices and the product test above would
    /// still pass on a lucky pair of edges.
    #[test]
    fn the_witness_is_a_real_cycle() {
        let g = crate::planted::frustrated_loops(6, 40, 3).graph;
        let err = gauge(&g).expect_err("a planted-loop instance is frustrated by construction");
        let c = &err.cycle;
        assert!(c.len() >= 3, "a cycle needs three vertices");
        let unique: std::collections::BTreeSet<_> = c.iter().collect();
        assert_eq!(unique.len(), c.len(), "a cycle does not repeat a vertex: {c:?}");
        for w in c.windows(2) {
            assert!(coupling(&g, w[0], w[1]).is_some(), "{}-{} is not an edge", w[0], w[1]);
        }
        assert!(coupling(&g, *c.last().unwrap(), c[0]).is_some(), "the cycle does not close");
        assert!(cycle_product(&g, c) < 0.0);
    }

    /// A gauge is a relabelling: it maps states to states and leaves every energy alone.
    ///
    /// Checked over the WHOLE state space, and with a field, because the field's transformation is
    /// the half that is easy to forget and impossible to notice at zero field.
    #[test]
    fn a_gauge_changes_coordinates_and_not_energies() {
        let g = crate::ising::ring(8, 1.0, 0.3);
        let sigma: Vec<i8> = (0..g.n).map(|i| if i % 3 == 0 { -1 } else { 1 }).collect();
        let gauged = apply_gauge(&g, &sigma);
        for m in 0u64..(1u64 << g.n) {
            let s: Vec<i8> = (0..g.n).map(|i| if m >> i & 1 == 1 { 1i8 } else { -1 }).collect();
            let mapped: Vec<i8> = s.iter().zip(&sigma).map(|(&v, &sg)| v * sg).collect();
            assert!(
                (g.energy(&mapped) - gauged.energy(&s)).abs() < 1e-12,
                "the gauge moved an energy"
            );
        }
    }

    /// Both moves sample the Boltzmann distribution, checked against exact enumeration.
    ///
    /// The claim a cluster algorithm has to earn: it moves large regions at once and must still
    /// leave the same distribution invariant.
    ///
    /// Scored by `cert.passed()` rather than by comparing `tv_exact` against `noise_floor` by hand.
    /// That is not a stylistic preference. A hand-rolled `tv < floor` silently switches ITSELF off
    /// on a run that is too short: the floor is `0.5 sqrt(2^n / ess)`, which at `n = 16` and 4000
    /// draws is 2.31, and total variation between two distributions can never exceed 1 — so the
    /// comparison passes for every sampler, including one that returns a constant. This test was
    /// written that way and two of its three models were decorative. [`crate::certify`] already
    /// knows: it raises `TooFewSamples` exactly when the floor reaches 1, and `passed()` reads both
    /// findings, so the power of the test is checked by the same call that checks the result.
    ///
    /// The disguised case is here on purpose: mixed signs exercise the gauge round trip — sample in
    /// gauged coordinates, report in the caller's — and catch a gauge applied on the way in and
    /// forgotten on the way out.
    #[test]
    fn both_moves_sample_the_distribution_they_claim_to() {
        let disguised = {
            let base = crate::ising::lattice2d(3, 1.0);
            let mut rng = Pcg::new(99, 2);
            let d: Vec<i8> = (0..base.n).map(|_| rng.spin(0.5)).collect();
            let g = apply_gauge(&base, &d);
            assert!(g.w.iter().any(|&w| w < 0.0), "the disguise did nothing");
            g
        };
        for (name, g) in [
            ("ferro ring 10", crate::ising::ring(10, 1.0, 0.0)),
            ("ferro 3x3", crate::ising::lattice2d(3, 1.0)),
            ("disguised 3x3", disguised),
        ] {
            for update in [Update::SwendsenWang, Update::Wolff] {
                let beta = 0.4;
                let mut c = Sampler::new(&g, beta, 11).expect("balanced and unbiased");
                let set = c.collect(&crate::samples::Plan::new(500, 6000, 4), update);
                let cert = set.certificate(&g).expect("collect returns a chain");
                assert!(
                    cert.passed(),
                    "{name} under {update:?} is not a Boltzmann sample:\n{cert}"
                );
            }
        }
    }

    /// The clusters are real: at low temperature a sweep moves a spanning region, not a spin.
    ///
    /// Without this the sampler could be single-spin Gibbs in disguise — correct, and pointless.
    #[test]
    fn a_cold_sweep_moves_a_cluster_rather_than_a_spin() {
        let g = crate::ising::lattice2d(8, 1.0);
        for update in [Update::SwendsenWang, Update::Wolff] {
            let mut c = Sampler::new(&g, 1.0, 3).unwrap();
            let mut biggest = 0usize;
            for _ in 0..20 {
                biggest = biggest.max(c.sweep_with(update, None).largest);
            }
            assert!(
                biggest > g.n / 2,
                "{update:?}: at beta 1 on a ferromagnet the largest cluster should span the \
                 lattice; got {biggest} of {}",
                g.n
            );
        }
    }

    /// A hot sweep does not, which is the other end of the previous test.
    ///
    /// Without it, a sampler that simply flipped everything every sweep would pass. `1 - exp(-2βJ)`
    /// at β = 0.01 is about 0.02, so clusters are singletons and the move degenerates to a
    /// single-spin flip — which is correct, and is what a cluster algorithm should do when there is
    /// no correlation to exploit.
    #[test]
    fn a_hot_sweep_finds_no_cluster_to_move() {
        let g = crate::ising::lattice2d(8, 1.0);
        let mut c = Sampler::new(&g, 0.01, 5).unwrap();
        let mut biggest = 0usize;
        for _ in 0..20 {
            biggest = biggest.max(c.sweep_with(Update::SwendsenWang, None).largest);
        }
        assert!(biggest < g.n / 4, "at beta 0.01 clusters should be tiny; got {biggest}");
    }

    /// The ledger reports what the sweep touched, and the stats and the bill are the same number.
    ///
    /// The bill is not derived from the sweep count, because a sweep is a convention and the two
    /// moves do not mean the same thing by it. `visited` and `bonds_tested` are measurements, so a
    /// comparison built on them cannot be rigged by redefining a sweep — which is the failure this
    /// module already made once, in the other direction. See [`Sampler::with_wolff_steps`].
    #[test]
    fn the_ledger_is_billed_exactly_what_the_stats_report() {
        let g = crate::ising::lattice2d(6, 1.0);
        for update in [Update::SwendsenWang, Update::Wolff] {
            let mut c = Sampler::new(&g, 0.44, 2).unwrap();
            let mut l = crate::ledger::Ledger::default();
            let (mut visited, mut bonds, mut flipped) = (0u64, 0u64, 0u64);
            for _ in 0..10 {
                let st = c.sweep_with(update, Some(&mut l));
                visited += st.visited;
                bonds += st.bonds_tested;
                flipped += st.flipped;
            }
            assert_eq!((l.samples, l.reads, l.writes), (visited, bonds, flipped), "{update:?}");
            assert!(bonds > 0, "{update:?}: bonds were tested and not counted");
        }
    }

    /// A Swendsen–Wang sweep visits every spin; a Wolff sweep takes exactly the steps it was given.
    ///
    /// The step count is a constant and this is what pins it. If it ever again became a function of
    /// the clusters the sweep built, `clusters` would stop equalling `k` and this would say so.
    #[test]
    fn a_sweep_takes_the_number_of_steps_it_was_told_to() {
        let g = crate::ising::lattice2d(6, 1.0);
        let mut sw = Sampler::new(&g, 0.44, 2).unwrap();
        for _ in 0..5 {
            assert_eq!(sw.sweep_with(Update::SwendsenWang, None).visited, g.n as u64);
        }
        for k in [1usize, 4, 9] {
            let mut c = Sampler::new(&g, 0.44, 2).unwrap().with_wolff_steps(k);
            for _ in 0..5 {
                assert_eq!(c.sweep_with(Update::Wolff, None).clusters, k, "at k = {k}");
            }
        }
    }

    /// A Wolff step flips its cluster with probability one, so the seed spin always changes.
    ///
    /// The distribution test cannot see this. Flipping with probability one half is still a valid
    /// kernel — a lazy one — that leaves the same distribution invariant and simply mixes half as
    /// fast, so it survives every check on the distribution while quietly halving the algorithm's
    /// reason to exist. This is the claim stated exactly, over the whole state space it touches.
    #[test]
    fn a_wolff_step_always_flips_the_cluster_it_grew() {
        let g = crate::ising::lattice2d(5, 1.0);
        for beta in [0.05, 0.44, 1.5] {
            let mut c = Sampler::new(&g, beta, 8).unwrap();
            for step in 0..200 {
                let before = c.state();
                let st = c.wolff_step();
                let after = c.state();
                let moved = before.iter().zip(&after).filter(|(a, b)| a != b).count();
                assert_eq!(
                    moved as u64, st.flipped,
                    "beta {beta}, step {step}: reported {} flips and made {moved}",
                    st.flipped
                );
                assert!(st.flipped > 0, "beta {beta}, step {step}: a Wolff step declined to move");
            }
        }
    }

    /// Every component gets sampled, which a fixed seed would not deliver.
    ///
    /// Wolff with a deterministic seed still satisfies detailed balance, so it passes the
    /// distribution test on a connected graph — and never touches a second component at all. Two
    /// disjoint rings make that visible, and exercise the outer loop of [`gauge`] over roots at the
    /// same time, since a disconnected graph has one gauge per component.
    #[test]
    fn a_disconnected_model_is_sampled_in_every_component() {
        let mut b = crate::graph::GraphBuilder::new(12);
        for r in 0..2 {
            for i in 0..6 {
                b.couple(r * 6 + i, r * 6 + (i + 1) % 6, 1.0);
            }
        }
        let g = b.build();
        let mut c = Sampler::new(&g, 0.4, 4).unwrap();
        let start = c.state();
        let mut moved = [false; 12];
        for _ in 0..400 {
            c.wolff_step();
            let now = c.state();
            for i in 0..12 {
                moved[i] |= now[i] != start[i];
            }
        }
        assert!(
            moved.iter().all(|&m| m),
            "these sites never moved: {:?}",
            (0..12).filter(|&i| !moved[i]).collect::<Vec<_>>()
        );
    }

    /// A coupling of zero constrains nothing, and a model carrying one is not frustrated by it.
    ///
    /// A zero edge is not hypothetical. `GraphBuilder` sums duplicate pairs, so `couple(0, 2, 1.0)`
    /// followed by `couple(0, 2, -1.0)` leaves a stored edge of weight zero — the ordinary way a
    /// user bias and a penalty cancel — and `build` keeps it in the CSR. Drop the guard in [`gauge`]
    /// and `w > 0.0` is false, so the sign rule reads the edge as antiferromagnetic and demands
    /// `σ_2 = -σ_0`. On the triangle below that contradicts the two real couplings, and a model with
    /// no frustration in it anywhere is refused with a "frustrated" cycle whose product is zero.
    ///
    /// So this checks both halves: the gauge accepts, and the sampler then reproduces the
    /// distribution — which also pins the two places the sweep must ignore a zero bond, since a
    /// cluster joined across one would be joining across nothing.
    #[test]
    fn a_zero_coupling_constrains_nothing() {
        let mut b = crate::graph::GraphBuilder::new(9);
        b.couple(0, 1, 1.0);
        b.couple(1, 2, 1.0);
        // The third side of the triangle, cancelled to exactly zero by two opposing calls.
        b.couple(0, 2, 1.0);
        b.couple(0, 2, -1.0);
        // The rest of the sites make the model big enough for `certify` to have power.
        for i in 3..9 {
            b.couple(i, (i + 1 - 3) % 6 + 3, 1.0);
        }
        let g = b.build();
        let k = (g.offset[0]..g.offset[1]).find(|&k| g.nbr[k] == 2).expect("edge 0-2 was kept");
        assert_eq!(g.w[k], 0.0, "the two couplings must cancel to a stored zero");

        let sigma = gauge(&g).expect("a zero coupling cannot frustrate anything");
        let regauged = apply_gauge(&g, &sigma);
        assert!(regauged.w.iter().all(|&w| w >= 0.0));

        for update in [Update::SwendsenWang, Update::Wolff] {
            let mut c = Sampler::new(&g, 0.4, 6).expect("balanced");
            let set = c.collect(&crate::samples::Plan::new(500, 6000, 4), update);
            let cert = set.certificate(&g).expect("collect returns a chain");
            assert!(cert.passed(), "{update:?} on a graph with a zero edge:\n{cert}");
        }
    }

    /// A uniform field is sampled, not refused — the ghost spin makes it an ordinary edge.
    ///
    /// This is the case the folklore gets wrong. "Swendsen–Wang cannot handle a field" is true of
    /// the move as literally stated and false of the model: a cycle through the ghost has product
    /// `J_ij h_i h_j`, so a ferromagnet under fields of one sign is balanced, and the same code that
    /// samples the zero-field model samples this one.
    ///
    /// Scored against enumeration by `certify`, at two field strengths and both signs, because a
    /// ghost applied on the way in and dropped on the way out would leave the distribution
    /// symmetric — the field's whole effect is to break that symmetry.
    #[test]
    fn a_uniform_field_is_sampled_rather_than_refused() {
        for h in [0.3, -0.3, 0.8] {
            for update in [Update::SwendsenWang, Update::Wolff] {
                let g = crate::ising::ring(10, 1.0, h);
                let mut c = Sampler::new(&g, 0.4, 12).expect("a uniform field keeps it balanced");
                let set = c.collect(&crate::samples::Plan::new(500, 6000, 4), update);
                let cert = set.certificate(&g).expect("collect returns a chain");
                assert!(cert.passed(), "h = {h} under {update:?}:\n{cert}");
            }
        }
    }

    /// The field actually biases the magnetisation, by the amount enumeration says.
    ///
    /// `certify` scores the distribution, but this is the statement a reader wants to see made
    /// directly: a field has an effect, it has the right sign, and it has the right size. Drop the
    /// ghost and `<m>` collapses to zero by symmetry, which is a much louder failure than a
    /// distribution that is subtly off.
    #[test]
    fn the_field_biases_the_magnetisation_by_the_enumerated_amount() {
        let g = crate::ising::ring(10, 1.0, 0.4);
        let beta = 0.4;
        let (mut z, mut mz) = (0.0f64, 0.0f64);
        for mask in 0u64..(1u64 << g.n) {
            let st: Vec<i8> = (0..g.n).map(|i| if mask >> i & 1 == 1 { 1i8 } else { -1 }).collect();
            let w = (-beta * g.energy(&st)).exp();
            z += w;
            mz += w * st.iter().map(|&x| f64::from(x)).sum::<f64>() / g.n as f64;
        }
        let exact = mz / z;
        // The guard is on the FIXTURE, not the result: dropping the ghost sends `<m>` to zero by
        // symmetry, so the fixture only earns its keep if the true value is far from zero compared
        // with the 0.01 tolerance below. It is 0.337 here, a margin of about thirty.
        assert!(exact > 0.2, "the fixture's field is too weak to distinguish: {exact}");

        for update in [Update::SwendsenWang, Update::Wolff] {
            let mut c = Sampler::new(&g, beta, 21).unwrap();
            for _ in 0..2_000 {
                c.sweep_with(update, None);
            }
            let (mut m, mut n) = (0.0f64, 0.0f64);
            for _ in 0..40_000 {
                c.sweep_with(update, None);
                m += c.state().iter().map(|&x| f64::from(x)).sum::<f64>() / g.n as f64;
                n += 1.0;
            }
            let got = m / n;
            assert!(
                (got - exact).abs() < 0.01,
                "{update:?}: <m> {got:.4} against an enumerated {exact:.4}"
            );
        }
    }

    /// Mixed-sign fields are genuinely unbalanced, and the witness runs through the ghost.
    ///
    /// The other half of the previous test, and the one that stops the ghost from being a way to
    /// wave every field through. A cycle `i - j - ghost - i` has product `J_ij h_j h_i`, so two
    /// sites whose fields disagree in sign across a positive coupling cannot be gauged
    /// ferromagnetic — and the refusal names them.
    #[test]
    fn mixed_sign_fields_are_refused_with_a_cycle_through_the_ghost() {
        let mut b = crate::graph::GraphBuilder::new(6);
        for i in 0..6 {
            b.couple(i, (i + 1) % 6, 1.0);
            b.set_bias(i, if i % 2 == 0 { 0.5 } else { -0.5 });
        }
        let g = b.build();
        let Err(f) = Sampler::new(&g, 0.4, 1) else {
            panic!("mixed-sign fields are not balanced and must be refused")
        };
        assert_eq!(f.ghost, Some(g.n), "the ghost must be named, not left to be inferred");
        assert!(f.cycle.contains(&g.n), "the witness must run through the ghost: {:?}", f.cycle);
        let prod = f.product(&g).expect("the witness is a cycle of g plus the ghost");
        assert!(prod < 0.0, "the witness product must be negative, got {prod}");
    }

    /// A frustrated model is refused with its cycle.
    #[test]
    fn a_spin_glass_is_refused_with_the_cycle_that_makes_it_one() {
        let g = crate::planted::frustrated_loops(6, 40, 3).graph;
        match Sampler::new(&g, 0.5, 1) {
            Err(f) => {
                assert!(f.cycle.len() >= 3);
                assert_eq!(f.ghost, None, "a zero-field glass needs no ghost");
                assert!(f.product(&g).expect("a cycle of g") < 0.0);
            }
            Ok(_) => panic!("a frustrated glass must be refused"),
        }
    }
}
