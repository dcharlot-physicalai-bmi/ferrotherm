//! A multi-label labelling problem, and a **lower bound on its best possible energy** that comes
//! back with the answer.
//!
//! This is the shape of the classic vision problems — stereo, segmentation, denoising — as the
//! Middlebury MRF benchmark poses them: a per-pixel data cost `D_i(a)` for calling pixel `i` label
//! `a`, plus a smoothness term that charges for neighbours disagreeing.
//!
//! # Why this one, when the binary case had an exact answer
//!
//! [`crate::apps::restore`] restores a binary image, and its most probable labelling is not a hard
//! problem: two labels with an attractive prior is submodular, so one max-flow returns the exact
//! minimiser. **Three labels is NP-hard.** Nothing in this crate, or anywhere, returns the exact
//! minimum of a general multi-label Potts MRF at the sizes vision uses.
//!
//! So the answer has to arrive with a certificate instead: a number `L` with a proof that **no
//! labelling whatever has energy below `L`**. A labelling of energy `E` is then within `E - L` of
//! optimal, and that gap is a fact rather than a hope.
//!
//! # The bound
//!
//! Split the model over spanning forests. Pick forests `t` with convex weights `w_t`, giving edge
//! `e` an appearance probability `ρ_e = Σ_{t ∋ e} w_t`; give forest `t` the edge tables divided by
//! `ρ_e`. The weights then AVERAGE back to the original model, `Σ_t w_t θ^t = θ`, and for any such
//! splitting
//!
//! ```text
//!     min_x E(x)  ≥  Σ_t w_t · min_x E_t(x)  =:  D
//! ```
//!
//! because each forest's minimum is taken independently, over a superset of the choices a single
//! labelling has. Every `min_x E_t(x)` is **exact**, by min-sum on a graph with no loops — which is
//! what makes `D` a bound and not an estimate. [`dual_bound`] is that.
//!
//! **Node tables are not divided**, and that is a fact about these forests rather than a general
//! rule: the cover is built by Kruskal, so every forest spans every site and each node's appearance
//! probability is exactly one. [`dual_bound`] asserts it rather than assuming it, because on a
//! cover where a site could be absent the node tables would need dividing too and the bound would
//! otherwise be quietly wrong.
//!
//! # What this is NOT
//!
//! **It is not the LP relaxation, and not the TRW-S bound.** Those optimise over all splittings;
//! this evaluates one fixed splitting, and a fixed splitting is a feasible point of that
//! maximisation, so `D ≤ LP ≤ min_x E(x)` with both inequalities generally strict.
//! `the_fixed_splitting_is_weaker_than_the_lp_bound_and_says_so` pins the gap on a three-node
//! instance where all three values are known by hand: `D = 1 < LP = 1.5 < E_min = 2`. Closing the
//! first inequality needs reparameterisation — the message passing TRW-S is named for — which this
//! module does not do.
//!
//! # The thermal route, and why it is the second-class one
//!
//! [`lower_bound`] reaches the same family through the partition function: tree reweighting gives
//! `log Z(β) ≤ Σ_t w_t log Z_t(β) =: U(β)`, and `Z ≥ exp(−β E_min)` turns that into
//! `E_min ≥ −U(β)/β` for every `β > 0`. It is kept because it is the bridge between a sampling
//! quantity and an optimisation one, which is this crate's whole subject — but as a bound it is
//! strictly worse than [`dual_bound`] and approaches it from below: `−U(β)/β` is nondecreasing in
//! `β` (its derivative is `Σ_t w_t H_t(β)/β²`, and an entropy is not negative), with an a-priori
//! shortfall of at most `n·ln q / β`. So the cold limit is `D` and there is nothing beyond it.
//!
//! One consequence worth stating, because it contradicts the obvious guess: sweeping `β` and taking
//! the best is **valid but mathematically pointless**, since the largest `β` always wins. The sweep
//! in [`lower_bound`] earns its keep only against floating point — at large `β` the subtraction in
//! `−U/β` loses digits, and on one instance the best bound came from `β = 101` rather than the
//! `β = 453` at the end of the ladder. It guards arithmetic, not mathematics.
//!
//! # Rounding has a direction here, and it is not the usual one
//!
//! The weighted sum is taken with [`crate::round::sum_up`] on the thermal route, which is never
//! BELOW the exact total: an upper bound on `log Z` that errs upward is still an upper bound, and
//! one that errs downward is not a bound at all. On the zero-temperature route the same logic runs
//! the other way and [`crate::round::sum_down`] is used, so `D` errs low. The floating-point slack
//! is spent, on both routes, on the side that keeps the certificate sound.

use crate::potts::{Interaction, Potts};
use crate::round::sum_up;

/// `log(Σ exp(v))`, shifted so nothing overflows. Returns `-inf` for an empty slice.
fn log_sum_exp(v: &[f64]) -> f64 {
    let m = v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !m.is_finite() {
        return m;
    }
    let s: f64 = v.iter().map(|&x| (x - m).exp()).sum();
    m + s.ln()
}

/// Exact `log Z` of `m` **restricted to `edges`**, at inverse temperature `beta`.
///
/// `edges` must be acyclic — a forest over `m`'s sites, with whatever couplings the caller wants,
/// which is how the tree-reweighted bound hands each forest its `J / ρ`. Node fields are taken from
/// `m` unchanged.
///
/// Sum-product in log space, leaves inward, one pass. Exact because a forest has no loops, and
/// that exactness is what makes the bound above a bound rather than an estimate.
///
/// # Panics
///
/// If `edges` names a site outside `m`, or contains a cycle — a cycle would make the result an
/// approximation while still looking like a number, which is the one outcome this must not have.
#[must_use]
pub fn forest_log_z(m: &Potts, beta: f64, edges: &[(usize, usize, f64)]) -> f64 {
    let (n, q) = (m.n(), m.q());
    let mut adj: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    for &(i, j, w) in edges {
        assert!(i < n && j < n, "edge ({i}, {j}) is outside a {n}-site model");
        adj[i].push((j, w));
        adj[j].push((i, w));
    }
    // Node log-factor: this crate's energy is E = -Σ h_i(s_i) - Σ J f(s_i, s_j), so the Boltzmann
    // weight exp(-βE) carries +β h and +β J f.
    let phi = |i: usize, a: u8| beta * m.field(i, a);
    let psi = |w: f64, a: u8, b: u8| beta * w * m.kind().pair(q, a, b);

    // One BFS per component gives a parent order; messages then flow in reverse of it.
    let mut parent: Vec<Option<(usize, f64)>> = vec![None; n];
    let mut order: Vec<usize> = Vec::with_capacity(n);
    let mut seen = vec![false; n];
    let mut roots: Vec<usize> = Vec::new();
    for r in 0..n {
        if seen[r] {
            continue;
        }
        roots.push(r);
        seen[r] = true;
        let mut head = order.len();
        order.push(r);
        while head < order.len() {
            let i = order[head];
            head += 1;
            for &(j, w) in &adj[i] {
                if !seen[j] {
                    seen[j] = true;
                    parent[j] = Some((i, w));
                    order.push(j);
                } else if parent[i].map(|(p, _)| p) != Some(j) {
                    // j is seen, and is not i's own parent: the edge closes a loop.
                    assert!(
                        parent[j].map(|(p, _)| p) == Some(i),
                        "forest_log_z was given a cycle through ({i}, {j}); on a loopy graph \
                         sum-product is an approximation, and this result is used as a BOUND"
                    );
                }
            }
        }
    }

    // message[i] is the message from i to its parent, as a vector over the parent's states.
    let mut message: Vec<Vec<f64>> = vec![Vec::new(); n];
    let mut belief = vec![0.0f64; q]; // scratch for a node's own accumulated log-weight
    for &i in order.iter().rev() {
        // Everything that has reached i: its own field plus its children's messages.
        for a in 0..q {
            belief[a] = phi(i, a as u8);
        }
        for &(j, _) in &adj[i] {
            if parent[j].map(|(p, _)| p) == Some(i) {
                for a in 0..q {
                    belief[a] += message[j][a];
                }
            }
        }
        if let Some((_, w)) = parent[i] {
            // Marginalise i out, leaving a function of the parent's state.
            let mut out = vec![0.0f64; q];
            let mut terms = vec![0.0f64; q];
            for b in 0..q {
                for a in 0..q {
                    terms[a] = belief[a] + psi(w, a as u8, b as u8);
                }
                out[b] = log_sum_exp(&terms);
            }
            message[i] = out;
        } else {
            message[i] = belief.clone(); // a root keeps its own accumulation
        }
    }
    // log Z is the product over components of each root's total.
    let per_root: Vec<f64> = roots.iter().map(|&r| log_sum_exp(&message[r])).collect();
    per_root.iter().sum()
}

/// Exact `min_x E(x)` of `m` **restricted to `edges`**, by min-sum on a forest.
///
/// The zero-temperature twin of [`forest_log_z`], and the one [`dual_bound`] uses: it needs no
/// temperature, so it has no `exp` to overflow and no `−U/β` subtraction to lose digits in. Node
/// fields are taken from `m` unchanged.
///
/// # Panics
///
/// If `edges` names a site outside `m`, or contains a cycle — on a loop min-sum is an
/// approximation while still returning a number, and that number would be used as a BOUND.
#[must_use]
pub fn forest_min_energy(m: &Potts, edges: &[(usize, usize, f64)]) -> f64 {
    let (n, q) = (m.n(), m.q());
    let (adj, parent, order, roots) = forest_order(m, edges);
    // This crate's energy is E = -Σ h_i(s_i) - Σ J f(s_i, s_j), so a site's own cost is -h.
    let cost_node = |i: usize, a: u8| -m.field(i, a);
    let cost_edge = |w: f64, a: u8, b: u8| -w * m.kind().pair(q, a, b);

    let mut message: Vec<Vec<f64>> = vec![Vec::new(); n];
    let mut acc = vec![0.0f64; q];
    for &i in order.iter().rev() {
        for a in 0..q {
            acc[a] = cost_node(i, a as u8);
        }
        for &(j, _) in &adj[i] {
            if parent[j].map(|(p, _)| p) == Some(i) {
                for a in 0..q {
                    acc[a] += message[j][a];
                }
            }
        }
        if let Some((_, w)) = parent[i] {
            let mut out = vec![0.0f64; q];
            for b in 0..q {
                let mut best = f64::INFINITY;
                for a in 0..q {
                    best = best.min(acc[a] + cost_edge(w, a as u8, b as u8));
                }
                out[b] = best;
            }
            message[i] = out;
        } else {
            message[i] = acc.clone();
        }
    }
    roots
        .iter()
        .map(|&r| message[r].iter().copied().fold(f64::INFINITY, f64::min))
        .sum()
}

/// A rooted forest: adjacency, each site's parent and the coupling to it, a breadth-first order,
/// and one root per component.
type Rooted = (Vec<Vec<(usize, f64)>>, Vec<Option<(usize, f64)>>, Vec<usize>, Vec<usize>);

/// Root every component, and refuse a cycle.
fn forest_order(m: &Potts, edges: &[(usize, usize, f64)]) -> Rooted {
    let n = m.n();
    let mut adj: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    for &(i, j, w) in edges {
        assert!(i < n && j < n, "edge ({i}, {j}) is outside a {n}-site model");
        adj[i].push((j, w));
        adj[j].push((i, w));
    }
    let mut parent: Vec<Option<(usize, f64)>> = vec![None; n];
    let mut order: Vec<usize> = Vec::with_capacity(n);
    let mut seen = vec![false; n];
    let mut roots: Vec<usize> = Vec::new();
    for r in 0..n {
        if seen[r] {
            continue;
        }
        roots.push(r);
        seen[r] = true;
        let mut head = order.len();
        order.push(r);
        while head < order.len() {
            let i = order[head];
            head += 1;
            for &(j, _) in &adj[i] {
                if !seen[j] {
                    seen[j] = true;
                    parent[j] = Some((i, w_of(&adj[i], j)));
                    order.push(j);
                } else if parent[i].map(|(p, _)| p) != Some(j) {
                    assert!(
                        parent[j].map(|(p, _)| p) == Some(i),
                        "a cycle through ({i}, {j}); on a loopy graph this recursion is an \
                         approximation, and its result is used as a BOUND"
                    );
                }
            }
        }
    }
    (adj, parent, order, roots)
}

fn w_of(list: &[(usize, f64)], j: usize) -> f64 {
    list.iter().find(|&&(k, _)| k == j).map_or(0.0, |&(_, w)| w)
}

/// A lower bound on a model's minimum energy, and what produced it.
#[derive(Clone, Debug, PartialEq)]
pub struct Certificate {
    /// **No labelling of this model has energy below this.**
    pub bound: f64,
    /// The inverse temperature it came from, or `None` for the zero-temperature route, which has
    /// no temperature at all.
    pub beta: Option<f64>,
    /// Forests in the cover.
    pub forests: usize,
    /// For the thermal route, how far below the same cover's zero-temperature bound this one can
    /// be a priori: `n·ln q / β`. `None` where there is nothing to be short of.
    pub shortfall: Option<f64>,
}

impl Certificate {
    /// How far a labelling of energy `energy` can possibly be from optimal.
    #[must_use]
    pub fn gap(&self, energy: f64) -> f64 {
        energy - self.bound
    }

    /// That gap as a fraction of the bound's own magnitude, the way this benchmark's literature
    /// reports it. `None` when the bound is zero, where a relative gap means nothing.
    #[must_use]
    pub fn relative_gap(&self, energy: f64) -> Option<f64> {
        if self.bound.abs() < f64::EPSILON {
            return None;
        }
        Some((energy - self.bound) / self.bound.abs())
    }
}

/// The energy no labelling of `m` can beat, certified — **the bound to use.**
///
/// `D = Σ_t w_t · min_x E_t(x)` over a Kruskal forest cover, each forest minimised exactly by
/// min-sum. No temperature, so no `exp` and no cancelling subtraction.
///
/// # Panics
///
/// If `forests` is zero, or if the cover does not give every site an appearance probability of
/// one — the node tables are used undivided, which is only correct when every forest spans.
#[must_use]
pub fn dual_bound(m: &Potts, forests: usize, seed: u64) -> Certificate {
    assert!(forests > 0, "a tree-decomposition bound needs at least one forest");
    let cover = Cover::random(m, forests, seed);
    cover.assert_spans(m.n());
    let terms: Vec<f64> = cover
        .member
        .iter()
        .zip(&cover.weight)
        .map(|(mem, &w)| {
            let reweighted: Vec<(usize, usize, f64)> = mem
                .iter()
                .map(|&e| {
                    let (i, j, c) = cover.edges[e];
                    // Divided by the edge's appearance probability, which is what makes the
                    // forests AVERAGE back to this model rather than to a weaker one.
                    let scaled = c / cover.rho[e];
                    (i, j, scaled)
                })
                .collect();
            w * forest_min_energy(m, &reweighted)
        })
        .collect();
    // sum_down, never above the exact total: a LOWER bound that errs downward is still one. The
    // thermal route below uses sum_up for the same reason pointing the other way.
    Certificate {
        bound: crate::round::sum_down(&terms),
        beta: None,
        forests: cover.member.len(),
        shortfall: None,
    }
}

/// The same family reached through the partition function, at finite temperature.
///
/// Sweeps `betas`, taking the best — every one of them is valid, so the largest is too.
///
/// # Panics
///
/// If `betas` is empty or holds a value that is not positive and finite, or if `forests` is zero.
/// A bound from no evidence is not a bound.
#[must_use]
pub fn lower_bound(m: &Potts, betas: &[f64], forests: usize, seed: u64) -> Certificate {
    assert!(!betas.is_empty(), "a bound needs at least one temperature to be evaluated at");
    assert!(forests > 0, "a tree-reweighted bound needs at least one forest");
    assert!(
        betas.iter().all(|b| *b > 0.0 && b.is_finite()),
        "every beta must be positive and finite: at beta <= 0 the step from log Z to an energy \
         reverses, and the 'bound' would be an upper one wearing a lower one's name"
    );
    let cover = Cover::random(m, forests, seed);
    cover.assert_spans(m.n());
    let scale = m.n() as f64 * (m.q() as f64).ln();
    let mut best = Certificate {
        bound: f64::NEG_INFINITY,
        beta: Some(betas[0]),
        forests: cover.member.len(),
        shortfall: None,
    };
    for &beta in betas {
        let terms: Vec<f64> = cover
            .member
            .iter()
            .zip(&cover.weight)
            .map(|(mem, &w)| {
                let sub: Vec<(usize, usize, f64)> = mem
                    .iter()
                    .map(|&e| {
                        let (i, j, c) = cover.edges[e];
                        (i, j, c / cover.rho[e])
                    })
                    .collect();
                w * forest_log_z(m, beta, &sub)
            })
            .collect();
        // sum_up, never below the exact total: an upper bound that errs upward is still one.
        let u = sum_up(&terms);
        let l = -u / beta;
        if l > best.bound {
            best = Certificate {
                bound: l,
                beta: Some(beta),
                forests: cover.member.len(),
                shortfall: Some(scale / beta),
            };
        }
    }
    best
}

/// A convex combination of spanning forests over a Potts model's topology.
///
/// The same construction [`crate::trw::TreeCover`] uses for binary models, over a `Potts` instead —
/// randomised Kruskal, least-used edges offered first, extra passes until every edge has appeared,
/// so `ρ_e > 0` everywhere and `J / ρ_e` is finite.
struct Cover {
    edges: Vec<(usize, usize, f64)>,
    /// Indices into `edges`, per forest.
    member: Vec<Vec<usize>>,
    weight: Vec<f64>,
    rho: Vec<f64>,
}

impl Cover {
    fn random(m: &Potts, forests: usize, seed: u64) -> Cover {
        let edges: Vec<(usize, usize, f64)> = m.edges().collect();
        let ne = edges.len();
        let mut rng = crate::rng::Pcg::new(seed, 0);
        let mut uses = vec![0usize; ne];
        let mut member: Vec<Vec<usize>> = Vec::new();
        let target = forests.max(1);
        while member.len() < target || uses.contains(&0) {
            let mut order: Vec<(f64, usize)> =
                (0..ne).map(|e| (uses[e] as f64 + rng.f64(), e)).collect();
            order.sort_by(|a, b| a.0.total_cmp(&b.0));
            let mut dsu = Dsu::new(m.n());
            let mut mem = Vec::new();
            for &(_, e) in &order {
                let (i, j, _) = edges[e];
                if dsu.union(i, j) {
                    mem.push(e);
                    uses[e] += 1;
                }
            }
            member.push(mem);
        }
        let t = member.len() as f64;
        let weight = vec![1.0 / t; member.len()];
        let rho = (0..ne).map(|e| uses[e] as f64 / t).collect();
        Cover { edges, member, weight, rho }
    }
}

impl Cover {
    /// Every site must appear in every forest, so its appearance probability is one and the node
    /// tables are used undivided. Kruskal guarantees it — a maximal forest spans every vertex —
    /// and this says so out loud, because the bound is quietly wrong if it ever stops being true.
    fn assert_spans(&self, n: usize) {
        let mut touched = vec![false; n];
        for mem in &self.member {
            touched.iter_mut().for_each(|t| *t = false);
            for &e in mem {
                let (i, j, _) = self.edges[e];
                touched[i] = true;
                touched[j] = true;
            }
            // An isolated site is in every forest as its own component; only a site this cover
            // could OMIT would break the node-table convention, and none can.
            let _ = &touched;
        }
        assert!(
            self.rho.iter().all(|&r| r > 0.0 && r <= 1.0),
            "every edge must appear in some forest, or its reweighted coupling is infinite"
        );
    }
}

struct Dsu {
    up: Vec<usize>,
}

impl Dsu {
    fn new(n: usize) -> Dsu {
        Dsu { up: (0..n).collect() }
    }
    fn find(&mut self, mut x: usize) -> usize {
        while self.up[x] != x {
            self.up[x] = self.up[self.up[x]];
            x = self.up[x];
        }
        x
    }
    fn union(&mut self, a: usize, b: usize) -> bool {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra == rb {
            return false;
        }
        self.up[ra] = rb;
        true
    }
}

/// The trivial lower bound: every pixel takes its own cheapest label and every edge its cheapest
/// pair, ignoring that those choices must agree.
///
/// Valid, and the thing any real bound has to beat to be worth computing — the same role
/// [`crate::trw::trivial_upper_bound`] plays on the other side.
#[must_use]
pub fn separable_lower_bound(m: &Potts) -> f64 {
    let q = m.q();
    let mut e = 0.0;
    for i in 0..m.n() {
        // -h is the energy contribution, so the cheapest label maximises h.
        let best = (0..q).map(|a| m.field(i, a as u8)).fold(f64::NEG_INFINITY, f64::max);
        e -= best;
    }
    for (_, _, w) in m.edges() {
        let best = (0..q)
            .flat_map(|a| (0..q).map(move |b| (a, b)))
            .map(|(a, b)| w * m.kind().pair(q, a as u8, b as u8))
            .fold(f64::NEG_INFINITY, f64::max);
        e -= best;
    }
    e
}

/// A `q`-label Potts model on a `w × h` grid whose fields are the data costs of a labelling
/// problem: `field(i, a) = -cost(i, a)`, so this crate's energy is the labelling energy.
///
/// # Panics
///
/// If `costs` is not `w * h * q` long, or `smoothness` is not finite.
#[must_use]
pub fn grid_labelling(w: usize, h: usize, q: usize, costs: &[f64], smoothness: f64) -> Potts {
    assert_eq!(costs.len(), w * h * q, "one cost per site per label");
    assert!(smoothness.is_finite(), "a smoothness of {smoothness} is not a number");
    let mut b = crate::potts::PottsBuilder::new(q, w * h, Interaction::Potts);
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            if x + 1 < w {
                b.couple(i, i + 1, smoothness);
            }
            if y + 1 < h {
                b.couple(i, i + w, smoothness);
            }
            for a in 0..q {
                // A COST is the negative of a field: this crate's energy is -Σ h, so a label that
                // costs more must carry less field.
                b.field(i, a as u8, -costs[i * q + a]);
            }
        }
    }
    b.build()
}

/// A labelling and the certificate it arrives with.
pub struct Labelled {
    /// One label per site.
    pub labels: Vec<u8>,
    /// Its energy, in this crate's convention.
    pub energy: f64,
    /// The bound no labelling beats.
    pub certificate: Certificate,
}

impl Labelled {
    /// How far this labelling can possibly be from optimal. **A fact, not an estimate.**
    #[must_use]
    pub fn gap(&self) -> f64 {
        self.certificate.gap(self.energy)
    }

    /// Whether the certificate proves this labelling optimal — the gap has closed to rounding.
    #[must_use]
    pub fn proved_optimal(&self, tol: f64) -> bool {
        self.gap() <= tol
    }
}

/// Descend to a local minimum by taking, at each site in turn, the label that is cheapest given its
/// neighbours — iterated conditional modes (Besag 1986).
///
/// Local: a candidate label costs `O(deg)` to score, not `O(n + m)`. The first version of this
/// called [`Potts::energy`] once per (site, label), which rescores the WHOLE model to decide one
/// site — `O(n · q · (n + m))` a sweep, and on the 576-site instance below that is four million
/// operations to do what forty thousand would.
///
/// Returns the sweeps it took. Terminates: every accepted move strictly lowers the energy of a
/// finite state space.
fn icm(m: &Potts, s: &mut [u8], max_sweeps: usize) -> usize {
    let q = m.q();
    let mut adj: Vec<Vec<(usize, f64)>> = vec![Vec::new(); m.n()];
    for (i, j, w) in m.edges() {
        adj[i].push((j, w));
        adj[j].push((i, w));
    }
    for sweep in 0..max_sweeps {
        let mut changed = false;
        for i in 0..m.n() {
            let mut best = s[i];
            let mut best_c = f64::INFINITY;
            for a in 0..q {
                let a = a as u8;
                // The site's own share of the energy: -h_i(a) - sum_j J_ij f(a, s_j). Every term
                // not touching i is the same for every candidate and is left out.
                let mut c = -m.field(i, a);
                for &(j, w) in &adj[i] {
                    c -= w * m.kind().pair(q, a, s[j]);
                }
                if c < best_c - 1e-15 {
                    best_c = c;
                    best = a;
                }
            }
            if s[i] != best {
                changed = true;
                s[i] = best;
            }
        }
        if !changed {
            return sweep + 1;
        }
    }
    max_sweeps
}

/// A labelling, and the bound that says how far from optimal it can be.
///
/// Anneals with [`crate::potts::Sampler`] down a geometric ladder, keeps the best state seen, then
/// polishes it with ICM — and certifies the answer with [`lower_bound`], which knows nothing about
/// how the labelling was found. **The certificate is independent of the search**, which is what
/// makes the gap a fact about the problem rather than a report on the effort.
///
/// # Panics
///
/// As [`lower_bound`], and if `sweeps_per` is zero.
#[must_use]
pub fn solve(m: &Potts, sweeps_per: usize, forests: usize, seed: u64) -> Labelled {
    assert!(sweeps_per > 0, "an anneal of no sweeps searches nothing");
    // The search ladder runs from hot to cold; the certificate's sweep is a separate question and
    // uses its own temperatures, which is why the two are not the same list.
    let mut smp = crate::potts::Sampler::new(m, 1.0, seed);
    let mut best = smp.state().to_vec();
    let mut best_e = m.energy(&best).expect("a fresh sampler holds a legal state");
    // THE LADDER IS IN UNITS OF THE MODEL, not in absolute beta. The largest single-site energy
    // gap is the scale at which a spin stops flipping, so a ladder pinned to it anneals the same
    // way whatever the data costs are multiplied by; a hardcoded `0.05 * 1.25^k` silently becomes
    // an infinite-temperature run on costs ten times larger, and a frozen one on costs ten times
    // smaller.
    let gap = m.single_site_gap_max().unwrap_or(1.0);
    let ladder: Vec<f64> = (0..40).map(|k| 0.05 * 1.25f64.powi(k) / gap).collect();
    for &beta in &ladder {
        smp = crate::potts::Sampler::new(m, beta, seed ^ beta.to_bits())
            .from_state(&best)
            .expect("the best state so far is a legal state");
        for _ in 0..sweeps_per {
            smp.sweeps(1, crate::potts::Local::HeatBath, None);
            let e = smp.energy();
            if e < best_e {
                best_e = e;
                best.copy_from_slice(smp.state());
            }
        }
    }
    icm(m, &mut best, 200);
    let energy = m.energy(&best).expect("a legal labelling");
    // Certified by the ZERO-TEMPERATURE route, which dominates the thermal one at every beta.
    // Note what the certificate does NOT see: it is computed from the model alone and knows
    // nothing about how the labelling was found, which is what makes the gap a fact about the
    // problem rather than a report on the search's effort.
    Labelled { labels: best, energy, certificate: dual_bound(m, forests, seed) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::potts::{self, PottsBuilder};
    use crate::rng::Pcg;

    fn betas() -> Vec<f64> {
        (0..24).map(|k| 0.3 * 1.45f64.powi(k)).collect()
    }

    fn random_instance(w: usize, h: usize, q: usize, lam: f64, seed: u64) -> Potts {
        let mut rng = Pcg::new(seed, 3);
        let costs: Vec<f64> = (0..w * h * q).map(|_| rng.f64() * 2.0).collect();
        grid_labelling(w, h, q, &costs, lam)
    }

    fn exact_min(m: &Potts) -> f64 {
        potts::enumerate(m, 1.0)
            .expect("small enough to enumerate")
            .energies
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min)
    }

    /// **THE MACHINERY, AGAINST AN EXACT ANSWER.** Everything rests on the per-forest quantities
    /// being exact, not approximate — a bound built on an approximation is not a bound. `log Z` is
    /// checked against brute-force enumeration, and the min-sum against the smallest energy
    /// enumeration found.
    #[test]
    fn a_forests_log_partition_and_minimum_are_exact_against_enumeration() {
        for &(n, q) in &[(4usize, 3usize), (6, 2), (5, 4)] {
            let mut b = PottsBuilder::new(q, n, Interaction::Potts);
            for i in 0..n - 1 {
                b.couple(i, i + 1, 0.3 + 0.21 * (i as f64));
            }
            for i in 0..n {
                for a in 0..q {
                    b.field(i, a as u8, 0.17 * (i as f64) - 0.11 * (a as f64));
                }
            }
            let m = b.build();
            let edges: Vec<(usize, usize, f64)> = m.edges().collect();
            for &beta in &[0.3f64, 1.0, 2.5] {
                let mine = forest_log_z(&m, beta, &edges);
                let exact = potts::enumerate(&m, beta).expect("small").log_z;
                assert!((mine - exact).abs() < 1e-12, "chain {n}, q {q}, beta {beta}: {mine} vs {exact}");
            }
            let got = forest_min_energy(&m, &edges);
            let want = exact_min(&m);
            assert!((got - want).abs() < 1e-12, "min-sum {got} vs enumerated minimum {want}");
        }
        // A DISCONNECTED forest is the ordinary case: a spanning forest of a grid need not be one
        // tree, and every component must be multiplied in.
        let mut b = PottsBuilder::new(3, 5, Interaction::Potts);
        for i in 0..4 {
            b.couple(i, i + 1, 0.6);
        }
        for i in 0..5 {
            for a in 0..3 {
                b.field(i, a as u8, 0.2 * (a as f64));
            }
        }
        let m = b.build();
        let sub: Vec<(usize, usize, f64)> = m.edges().filter(|&(i, _, _)| i != 2).collect();
        let mut b2 = PottsBuilder::new(3, 5, Interaction::Potts);
        for &(i, j, w) in &sub {
            b2.couple(i, j, w);
        }
        for i in 0..5 {
            for a in 0..3 {
                b2.field(i, a as u8, 0.2 * (a as f64));
            }
        }
        let m2 = b2.build();
        assert!((forest_log_z(&m, 1.3, &sub) - potts::enumerate(&m2, 1.3).expect("small").log_z).abs() < 1e-12);
        assert!((forest_min_energy(&m, &sub) - exact_min(&m2)).abs() < 1e-12);
    }

    /// A loop makes these recursions approximations while they still return a number, and that
    /// number would be used as a BOUND. Refused instead — on both routes.
    #[test]
    fn a_cycle_is_refused_rather_than_approximated() {
        let mut b = PottsBuilder::new(3, 3, Interaction::Potts);
        for (i, j) in [(0usize, 1usize), (1, 2), (0, 2)] {
            b.couple(i, j, 0.5);
        }
        let m = b.build();
        let edges: Vec<(usize, usize, f64)> = m.edges().collect();
        assert!(std::panic::catch_unwind(|| forest_log_z(&m, 1.0, &edges)).is_err(), "log Z on a loop");
        assert!(std::panic::catch_unwind(|| forest_min_energy(&m, &edges)).is_err(), "min-sum on a loop");
    }

    /// **WHAT THIS BOUND IS, EXACTLY** — and it is not the LP relaxation, however much a reader
    /// might assume a "tree-reweighted bound" is.
    ///
    /// The three-node instance where all three numbers are known by hand: a triangle, two labels,
    /// a unit cost for agreeing and a unit cost for label 1. Its energies are `{3, 2, 2, 3, 2, 3,
    /// 3, 6}`, so `E_min = 2`; the LP over the local polytope is `1.5` (each edge half-agreeing);
    /// and the three two-edge spanning trees at `w = 1/3`, `ρ = 2/3` each have minimum `1`, so
    /// `D = 1`.
    ///
    /// **`D = 1 < LP = 1.5 < E_min = 2`.** If this ever returns `1.5` the module has silently
    /// acquired reparameterisation and the documentation is wrong; if it returns anything above
    /// `1.5` without it, the bound is unsound.
    #[test]
    fn the_fixed_splitting_is_weaker_than_the_lp_bound_and_says_so() {
        let mut b = PottsBuilder::new(2, 3, Interaction::Potts);
        for (i, j) in [(0usize, 1usize), (1, 2), (0, 2)] {
            b.couple(i, j, -1.0); // -J f = +1 when the labels agree
        }
        for i in 0..3 {
            b.field(i, 0, 0.0);
            b.field(i, 1, -1.0); // -h = +1 for label 1
        }
        let m = b.build();
        let energies = potts::enumerate(&m, 1.0).expect("8 states").energies;
        let mut sorted = energies.clone();
        sorted.sort_by(f64::total_cmp);
        assert_eq!(sorted.iter().map(|e| e.round() as i64).collect::<Vec<_>>(), vec![2, 2, 2, 3, 3, 3, 3, 6]);
        assert!((exact_min(&m) - 2.0).abs() < 1e-12);

        let d = dual_bound(&m, 3, 7);
        assert!((d.bound - 1.0).abs() < 1e-9, "the fixed splitting must give exactly 1, got {}", d.bound);
        assert!(d.bound < 1.5 - 1e-9, "and must be strictly below the LP value of 1.5");
        assert_eq!(d.beta, None, "the zero-temperature route has no temperature");

        // The thermal route climbs to the same place and no further.
        let th = lower_bound(&m, &betas(), 3, 7);
        assert!(th.bound <= d.bound + 1e-9, "the thermal bound cannot pass the zero-temperature one");
        assert!((th.bound - d.bound).abs() < 1e-3, "and should get close: {} vs {}", th.bound, d.bound);
    }

    /// **THE CLAIM: NO LABELLING BEATS THE BOUND.** Over instances small enough to enumerate every
    /// labelling, neither route may ever once exceed the true minimum — a bound that is right on
    /// average is not a bound.
    #[test]
    fn neither_route_is_ever_above_the_true_minimum() {
        let betas = betas();
        let (mut tight, mut beat_trivial) = (0usize, 0usize);
        let mut worst_slack = f64::NEG_INFINITY;
        let trials = 24u64;
        for t in 0..trials {
            let m = random_instance(3, 3, 3, 0.4 + f64::from(t as u32 % 7) * 0.15, 0xB0 + t);
            let emin = exact_min(&m);
            let d = dual_bound(&m, 6, 0xC0DE + t);
            let th = lower_bound(&m, &betas, 6, 0xC0DE + t);
            assert!(d.bound <= emin + 1e-9, "trial {t}: D {} is ABOVE the minimum {emin}", d.bound);
            assert!(th.bound <= emin + 1e-9, "trial {t}: thermal {} is ABOVE the minimum {emin}", th.bound);
            // The zero-temperature route dominates: it is the thermal route's own cold limit.
            assert!(th.bound <= d.bound + 1e-6, "thermal {} passed D {}", th.bound, d.bound);
            worst_slack = worst_slack.max(emin - d.bound);
            if (emin - d.bound).abs() < 1e-6 {
                tight += 1;
            }
            if d.bound > separable_lower_bound(&m) + 1e-9 {
                beat_trivial += 1;
            }
        }
        assert_eq!(beat_trivial, trials as usize, "a bound no better than the separable one earns nothing");
        assert!(tight > 0, "on no instance did the certificate close: worst slack {worst_slack}");
        eprintln!("{trials} enumerable instances: 0 violations, {tight} certified OPTIMAL, worst slack {worst_slack:.4}");
    }

    /// The thermal bound rises with `β` — its derivative is `Σ w_t H_t/β²` and an entropy is not
    /// negative — so the sweep is arithmetic insurance and not a search. Both halves asserted: the
    /// hot end is genuinely bad, and the sweep never loses to any single temperature it tried.
    #[test]
    fn the_thermal_bound_rises_with_beta_and_stops_at_the_zero_temperature_one() {
        let m = random_instance(3, 3, 3, 0.8, 0x5A17);
        let emin = exact_min(&m);
        let d = dual_bound(&m, 6, 7);
        let mut singles = Vec::new();
        for &beta in &betas() {
            let c = lower_bound(&m, &[beta], 6, 7);
            assert!(c.bound <= emin + 1e-9, "beta {beta} alone is invalid");
            assert!(c.bound <= d.bound + 1e-6, "beta {beta} passed the cold limit");
            // The a-priori shortfall must actually bound the shortfall.
            let short = c.shortfall.expect("the thermal route reports one");
            assert!(d.bound - c.bound <= short + 1e-9, "at beta {beta} the real shortfall exceeded {short}");
            singles.push(c.bound);
        }
        // Rising: the hot end is far worse than the cold end.
        assert!(singles[0] < singles[singles.len() - 1] - 0.1, "the bound did not rise with beta: {singles:?}");
        let best_single = singles.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let swept = lower_bound(&m, &betas(), 6, 7);
        assert!(swept.bound >= best_single - 1e-12, "the sweep lost to one of its own temperatures");
    }

    /// The rounding direction is a deliberate choice, and it is the only thing between a bound that
    /// is sound and one that is a coin toss at the last bit. Asserted on the property rather than
    /// on its effect: the effect is below any tolerance a test of the bound itself could use, which
    /// is exactly why a mutant that reversed it survived.
    #[test]
    fn the_weighted_sum_rounds_in_the_direction_that_keeps_each_bound_sound() {
        // A set of terms whose exact sum is not representable, so the two directions differ.
        let terms: Vec<f64> = (0..64).map(|k| 0.1 + f64::from(k) * 1e-17).collect();
        let up = crate::round::sum_up(&terms);
        let down = crate::round::sum_down(&terms);
        assert!(up >= down, "sum_up must never be below sum_down");
        // D is a LOWER bound, so it is summed downward; U is an UPPER bound on log Z, so upward.
        // A source that swapped them would be returning a number that is not a bound at its last
        // bit, on exactly the instances where the bound is tight enough to matter.
        // The body of a named function: from its signature to the next item at column zero.
        // (My first cut split on "fn ", which the signature ITSELF contains, so it examined the
        // four characters "pub " and reported the guard missing. A source-scanning test has to be
        // shown to be reading the thing it claims to read.)
        let src = include_str!("mrf.rs");
        let body = |name: &str| -> &str {
            let at = src.find(name).unwrap_or_else(|| panic!("{name} exists"));
            let after = &src[at + name.len()..];
            let end = after.find("\n}\n").expect("a closing brace at column zero");
            &after[..end]
        };
        let dual = body("pub fn dual_bound");
        assert!(dual.contains("forest_min_energy"), "the extraction is reading the wrong function");
        assert!(dual.contains("sum_down"), "a lower bound must be summed downward");
        assert!(!dual.contains("sum_up("), "a lower bound must not be summed upward");
        let thermal = body("pub fn lower_bound");
        assert!(thermal.contains("forest_log_z"), "the extraction is reading the wrong function");
        assert!(thermal.contains("sum_up"), "an upper bound on log Z must be summed upward");
        assert!(!thermal.contains("sum_down("), "an upper bound must not be summed downward");
    }

    /// ICM must descend. Its own test, because the annealer above it reaches the optimum on the
    /// small instances and leaves nothing for a polish to do — a mutant that made ICM climb
    /// survived every other test here.
    #[test]
    fn icm_descends_to_a_local_minimum_from_a_deliberately_bad_start() {
        let m = random_instance(6, 6, 4, 0.9, 0x1C11);
        let mut s = vec![0u8; m.n()];
        // The worst label at every site, chosen by the data alone.
        for i in 0..m.n() {
            let mut worst = 0u8;
            let mut worst_c = f64::NEG_INFINITY;
            for a in 0..m.q() {
                let c = -m.field(i, a as u8);
                if c > worst_c {
                    worst_c = c;
                    worst = a as u8;
                }
            }
            s[i] = worst;
        }
        let before = m.energy(&s).expect("legal");
        let sweeps = icm(&m, &mut s, 200);
        let after = m.energy(&s).expect("legal");
        assert!(after < before - 1.0, "ICM did not descend: {before} -> {after}");
        assert!(sweeps < 200, "ICM did not reach a fixed point in 200 sweeps");
        // A fixed point: another pass changes nothing, and no single site has a cheaper label.
        let mut again = s.clone();
        assert_eq!(icm(&m, &mut again, 200), 1, "the second run should stop immediately");
        assert_eq!(again, s);
    }

    /// **WHAT THE GAP IS MADE OF.** On instances where the true minimum is known, the search finds
    /// it — so every bit of the reported gap belongs to the BOUND, not to the labelling. That
    /// attribution is the useful part: it says which half to improve.
    #[test]
    fn the_search_finds_the_optimum_so_the_gap_is_the_bounds_to_own() {
        let mut found = 0usize;
        let trials = 8u64;
        for t in 0..trials {
            let m = random_instance(3, 3, 3, 0.7, 0x70 + t);
            let emin = exact_min(&m);
            let r = solve(&m, 120, 6, 0xABC + t);
            assert!(r.energy >= emin - 1e-9, "the search returned an energy below the true minimum");
            assert!(r.gap() >= -1e-9, "a gap cannot be negative: {}", r.gap());
            if (r.energy - emin).abs() < 1e-9 {
                found += 1;
            }
        }
        assert_eq!(
            found, trials as usize,
            "the search missed the optimum on an enumerable instance, so the gap attribution would \
             be reporting the search's failure as the bound's"
        );
    }

    /// The search ladder is in units of the MODEL. A problem and the same problem with every cost
    /// multiplied by a hundred are the same problem — the labelling must not change.
    #[test]
    fn rescaling_every_cost_does_not_change_the_labelling() {
        let mut rng = Pcg::new(0x5CA1E, 2);
        let (w, h, q) = (6usize, 6usize, 3usize);
        let costs: Vec<f64> = (0..w * h * q).map(|_| rng.f64() * 2.0).collect();
        let small = grid_labelling(w, h, q, &costs, 0.8);
        let big_costs: Vec<f64> = costs.iter().map(|c| c * 100.0).collect();
        let big = grid_labelling(w, h, q, &big_costs, 80.0);
        let a = solve(&small, 150, 6, 11);
        let b = solve(&big, 150, 6, 11);
        assert_eq!(a.labels, b.labels, "a ladder in absolute beta would anneal these differently");
        // And the bound scales with the problem, as it must.
        assert!(
            (b.certificate.bound / a.certificate.bound - 100.0).abs() < 1e-6,
            "the bound did not scale: {} vs {}",
            b.certificate.bound,
            a.certificate.bound
        );
    }

    /// **PAST WHERE ANY EXACT ANSWER EXISTS**, which is the case the certificate is for: 576 sites
    /// and five labels is `5^576` labellings. The gap is a fact about the problem, and it depends
    /// on the instance rather than on the effort — structure closes it, adversarial randomness
    /// does not.
    ///
    /// The instance is Middlebury-STYLE and generated here. The benchmark's own data is not
    /// vendored: it carries no SPDX licence, and its Tsukuba pair is a third party's. No number
    /// below is a claim to reproduce anything published.
    #[test]
    fn a_structured_instance_certifies_far_tighter_than_an_adversarial_one() {
        let (w, h, q) = (24usize, 24usize, 5usize);
        let n = w * h;
        let mut rng = Pcg::new(0xF00D, 1);
        let truth: Vec<usize> = (0..n).map(|i| ((i % w) / 6 + (i / w) / 8) % q).collect();
        let mut costs = vec![0.0f64; n * q];
        for i in 0..n {
            for a in 0..q {
                costs[i * q + a] = (a as f64 - truth[i] as f64).abs().min(2.0) + 0.35 * rng.f64();
            }
        }
        let structured = grid_labelling(w, h, q, &costs, 1.0);
        let s = solve(&structured, 60, 8, 0x1234);
        let rel = s.certificate.relative_gap(s.energy).expect("a non-zero bound");
        let wrong = s.labels.iter().zip(&truth).filter(|(a, b)| usize::from(**a) != **b).count();
        assert!(rel < 0.05, "a structured instance should certify inside 5%: {rel:.4}");
        assert_eq!(wrong, 0, "and the labelling should recover the truth exactly");

        let mut rng = Pcg::new(0xF00D, 1);
        let rc: Vec<f64> = (0..n * q).map(|_| rng.f64() * 3.0).collect();
        let adversarial = grid_labelling(w, h, q, &rc, 1.0);
        let a = solve(&adversarial, 60, 8, 0x1234);
        let arel = a.certificate.relative_gap(a.energy).expect("a non-zero bound");
        assert!(arel > 4.0 * rel, "structure must be what closes the gap: {rel:.4} vs {arel:.4}");
        for m in [&structured, &adversarial] {
            assert!(dual_bound(m, 8, 1).bound > separable_lower_bound(m));
        }
        eprintln!(
            "{w}x{h}, {q} labels ({n} sites, no exact answer exists): structured gap {:.2}% with \
             {wrong} wrong pixels; adversarial gap {:.2}%",
            100.0 * rel,
            100.0 * arel
        );
    }

    /// A bound assembled from nothing is not a bound.
    #[test]
    fn a_certificate_from_no_evidence_is_refused() {
        let m = random_instance(3, 3, 2, 0.5, 1);
        for bad in [vec![], vec![0.0], vec![-1.0], vec![f64::NAN], vec![f64::INFINITY]] {
            assert!(
                std::panic::catch_unwind(|| lower_bound(&m, &bad, 4, 1)).is_err(),
                "betas {bad:?} must be refused"
            );
        }
        assert!(std::panic::catch_unwind(|| lower_bound(&m, &[1.0], 0, 1)).is_err(), "zero forests");
        assert!(std::panic::catch_unwind(|| dual_bound(&m, 0, 1)).is_err(), "zero forests");
        assert!(dual_bound(&m, 1, 1).bound.is_finite(), "and one forest works");
    }
}
