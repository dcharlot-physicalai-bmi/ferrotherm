//! Structured mean field — a variational bound over a tractable subgraph, not a product.
//!
//! [`crate::meanfield`] optimises over fully factorised `q(s) = Π_i q_i(s_i)`. Every coupling in
//! the model is thrown away by that family: a product distribution cannot represent a correlation,
//! so `⟨s_i s_j⟩` is forced to `m_i m_j` and the Gibbs–Bogoliubov bound pays for it.
//!
//! Structured mean field (Saul & Jordan, *Exploiting tractable substructures in intractable
//! networks*, NIPS 1995; Xing, Jordan & Russell, UAI 2003) partitions the spins into disjoint
//! **parts**, keeps every coupling INSIDE a part exactly, and factorises only ACROSS parts:
//!
//! ```text
//!   q(s) = Π_p q_p(s_p),   q_p(s_p) ∝ exp( β Σ_{i∈p} b_i s_i + β Σ_{(ij) intra p} w_ij s_i s_j )
//! ```
//!
//! The variational parameters are the fields `b`. Each part is solved EXACTLY by
//! [`crate::exact::Elimination`], so a part may be a chain, a subtree, a lattice row or any block
//! narrow enough to eliminate — the same trade [`crate::bound::forest`] makes against
//! [`crate::bound::decoupled`], one level up: keep a tractable substructure whole instead of
//! shredding it.
//!
//! ```
//! use ferrotherm::{exact::Elimination, ising, meanfield, structured};
//!
//! let g = ising::ring(24, 1.0, 0.0);           // a ferromagnetic ring, no fields
//! let beta = 0.6;
//! let smf = structured::StructuredMeanField::new(beta);
//! let s = smf.tightest(&g, &structured::contiguous(24, 8)).unwrap();   // three arcs of eight
//!
//! let exact = Elimination::default().log_partition(&g, beta).unwrap().log_z.unwrap();
//! let naive = meanfield::naive_mean_field(&g, beta, 5000, 0.5).log_z;
//! assert!(s.log_z <= exact);            // a bound, always
//! assert!(s.log_z > naive);             // and a tighter one here: 20.21 against 17.21
//! ```
//!
//! # The bound, and why it is valid at every `b`
//!
//! Gibbs–Bogoliubov gives `ln Z ≥ β⟨−E⟩_q + H(q)` for any `q`. Substituting the family above and
//! cancelling — the intra-part pair moments `⟨s_i s_j⟩` appear once from `⟨−E⟩` and once from the
//! part entropies, with opposite signs — leaves
//!
//! ```text
//!   F(b) = Σ_p ln Z_p(b) + β Σ_i (h_i − b_i) m_i(b) + β Σ_{(ij) across} w_ij m_i(b) m_j(b)
//! ```
//!
//! where `ln Z_p` and `m_i` are the exact log partition function and magnetisations of part `p`
//! under fields `b`. Nothing in that derivation asked `b` to be a fixed point, so **`F(b)` is a
//! lower bound on `ln Z` for any `b` whatsoever** — [`StructuredMeanField::bound_at`] evaluates it
//! at fields a caller supplies, and the tests hold the inequality at random `b`. The iteration
//! `b_i ← h_i + Σ_{j ∉ part(i)} w_ij m_j` is only how the bound is made tight, and
//! [`StructuredMeanField::run`] returns the **best** iterate, so a run that oscillates or is cut
//! short still returns a sound number.
//!
//! Two limits fix the scale, and both are tested as exact identities rather than as inequalities:
//!
//! * **[`singletons`]** — every part one spin, no intra couplings, `q` a product. The bound is
//!   naive mean field's, term for term.
//! * **one part holding every spin** — nothing is across parts, `q` is the model itself, `b = h`
//!   is a fixed point immediately, and `F = ln Z` **exactly**.
//!
//! # Where it is tighter, measured
//!
//! On a ferromagnet with no fields, naive mean field sits at `m = 0` and returns `n ln 2` — it
//! cannot see a single coupling. Structured mean field at the same `m = 0` returns `Σ_p ln Z_p`,
//! which exceeds `n ln 2` by exactly the correlation each part keeps. Against exact `ln Z` from
//! [`crate::exact::Elimination`], each of these asserted per model rather than on an average:
//!
//! | model | β | exact | naive | structured |
//! |---|---|---|---|---|
//! | ferromagnetic ring of 24, arcs of 8 | 0.6 | 20.719 | 17.214 | **20.208** |
//! | periodic lattice 6x6, split into rows | 0.2 | 26.445 | 24.953 | **25.669** |
//! | glass ring of 20, arcs of 5 | 1.5 | 24.857 | 22.952 | **24.296** |
//! | dense glass of 18, blocks of 6 | 0.5 | 17.742 | 16.012 | **16.544** |
//!
//! And the bound rises monotonically with the parts, from naive mean field at one spin per part to
//! exact when one part holds them all — a 24-spin ring at `β = 0.4`, exact `ln Z` 18.6022:
//!
//! ```text
//!   part size    1        2        3        4        6        8       12       24
//!   bound     16.8385  17.7022  18.0003  18.1503  18.3008  18.3761  18.4515  18.6022
//! ```
//!
//! # Where it is NOT tighter, and this is not hypothetical
//!
//! The structured family does **not contain** the factorised one: a part's `q_p` carries the true
//! intra couplings and no setting of `b` can switch them off. So there is no theorem saying this
//! beats [`crate::meanfield::naive_mean_field`], and from the cold `m = 0.01` start it can lose
//! badly — a frustrated model leaves the field iteration in a local optimum well below where the
//! cheap method stopped. Measured over 1800 dense-glass cases (60 instances x 6 temperatures x 5
//! partitions): the cold start lost **99** times, and a start seeded from naive mean field's own
//! fixed point lost **none**. Neither seed dominates the other, so
//! [`StructuredMeanField::tightest`] runs both and keeps the larger bound — legitimate because
//! both are sound. `a_cold_start_can_lose_to_naive_mean_field` exhibits one such instance and
//! asserts the loss, because a module claiming a tighter bound must say when it has none.

use crate::exact::{Elimination, TooWide};
use crate::graph::{Graph, GraphBuilder};

/// A structured mean-field run: the fields, the magnetisations, and the bound.
#[derive(Clone, Debug)]
pub struct Structured {
    /// Inverse temperature the bound was computed at.
    pub beta: f64,
    /// The **best** lower bound on `ln Z` over every iterate, the seed's included.
    pub log_z: f64,
    /// Magnetisations of the iterate that produced [`Structured::log_z`], which near convergence
    /// may be a few steps before the last — the bound wobbles in its last digits once the fields
    /// have stopped moving.
    pub m: Vec<f64>,
    /// Variational fields of that same iterate: `bound_at` on them returns [`Structured::log_z`].
    pub fields: Vec<f64>,
    /// Largest change in any field on the last iteration.
    pub residual: f64,
    /// Iterations actually run, which is the cap when it did not converge.
    pub iterations: usize,
    /// Which iterate produced [`Structured::log_z`]; `0` is the seed, before any field update.
    pub best_iteration: usize,
    /// Widest induced width among the parts. Each part cost `2^width` per elimination.
    pub width: usize,
}

impl Structured {
    /// Whether the last iteration moved every field by less than `tol`.
    #[must_use]
    pub fn converged(&self, tol: f64) -> bool {
        self.residual < tol
    }
}

/// Configuration for a structured mean-field run.
#[derive(Clone, Debug)]
pub struct StructuredMeanField {
    /// Inverse temperature.
    pub beta: f64,
    /// Cap on field updates; the iteration also stops when the field residual falls below `1e-13`.
    pub iters: usize,
    /// Damping in `[0, 1)`: the new field is `damping * old + (1 − damping) * update`.
    pub damping: f64,
    /// Refuse a part whose elimination order needs a table wider than this, as
    /// [`crate::exact::Elimination::max_width`].
    pub max_width: usize,
}

impl StructuredMeanField {
    /// A run at inverse temperature `beta`: 200 iterations, damping 0.5, width cap 20.
    #[must_use]
    pub fn new(beta: f64) -> Self {
        StructuredMeanField { beta, iters: 200, damping: 0.5, max_width: 20 }
    }

    /// The bound `F(b)` at fields `b` of the caller's choosing — valid at every `b`, so this never
    /// needs a fixed point.
    ///
    /// # Errors
    ///
    /// [`TooWide`] if some part's elimination order exceeds [`StructuredMeanField::max_width`].
    ///
    /// # Panics
    ///
    /// If `parts` is not a partition of `0..g.n`, or `fields` is not one field per spin.
    pub fn bound_at(&self, g: &Graph, parts: &[Vec<usize>], fields: &[f64]) -> Result<f64, TooWide> {
        assert_eq!(fields.len(), g.n, "one variational field per spin");
        let (mut built, of) = split(g, parts);
        self.evaluate(g, &mut built, &of, fields).map(|(f, _)| f)
    }

    /// Iterate the fields towards a fixed point from `m = 0.01`, returning the best bound seen.
    ///
    /// Seeded off zero exactly as [`crate::meanfield::naive_mean_field`] is, and for the same
    /// reason: `m = 0` is a symmetric fixed point on a fieldless ferromagnet, so an unnudged
    /// iteration sits there however cold the model gets.
    ///
    /// # Errors
    ///
    /// [`TooWide`] if some part's elimination order exceeds [`StructuredMeanField::max_width`].
    ///
    /// # Panics
    ///
    /// If `parts` is not a partition of `0..g.n`.
    pub fn run(&self, g: &Graph, parts: &[Vec<usize>]) -> Result<Structured, TooWide> {
        self.run_from(g, parts, &vec![0.01; g.n])
    }

    /// [`StructuredMeanField::run`] started from magnetisations of the caller's choosing.
    ///
    /// The obvious seed is [`crate::meanfield::naive_mean_field`]'s fixed point: this family has
    /// many local optima on a frustrated model, and starting where the cheap method stopped is what
    /// keeps the expensive one from landing somewhere worse.
    ///
    /// # Errors
    ///
    /// [`TooWide`] if some part's elimination order exceeds [`StructuredMeanField::max_width`].
    ///
    /// # Panics
    ///
    /// If `parts` is not a partition of `0..g.n`, or `m0` is not one magnetisation per spin.
    pub fn run_from(&self, g: &Graph, parts: &[Vec<usize>], m0: &[f64]) -> Result<Structured, TooWide> {
        assert_eq!(m0.len(), g.n, "one starting magnetisation per spin");
        let (mut built, of) = split(g, parts);
        let el = Elimination { max_width: self.max_width };
        let width = built.iter().map(|b| el.width(&b.sub)).max().unwrap_or(0);

        let mut b = vec![0.0f64; g.n];
        fields_from(g, &of, m0, &mut b);
        let (mut f, mut m) = self.evaluate(g, &mut built, &of, &b)?;
        let (mut best_f, mut best_m, mut best_b, mut best_iteration) = (f, m.clone(), b.clone(), 0);
        let mut residual = f64::INFINITY;
        let mut it = 0;
        while it < self.iters && residual > 1e-13 {
            residual = 0.0;
            for i in 0..g.n {
                let mut update = g.h[i];
                for e in g.offset[i]..g.offset[i + 1] {
                    let j = g.nbr[e] as usize;
                    if of[j] != of[i] {
                        update += g.w[e] * m[j];
                    }
                }
                let next = self.damping * b[i] + (1.0 - self.damping) * update;
                residual = residual.max((next - b[i]).abs());
                b[i] = next;
            }
            it += 1;
            let (nf, nm) = self.evaluate(g, &mut built, &of, &b)?;
            f = nf;
            m = nm;
            if f > best_f {
                best_f = f;
                best_m.copy_from_slice(&m);
                best_b.copy_from_slice(&b);
                best_iteration = it;
            }
        }
        Ok(Structured {
            beta: self.beta,
            log_z: best_f,
            m: best_m,
            fields: best_b,
            residual,
            iterations: it,
            best_iteration,
            width,
        })
    }

    /// Both seeds — `m = 0.01` and naive mean field's fixed point — with the better bound kept.
    ///
    /// Neither seed dominates: the cold one climbs higher on models the cheap method misreads, the
    /// warm one rescues the frustrated models where the cold iteration lands in a bad optimum. Two
    /// sound bounds, so their maximum is sound, and this is the entry point to prefer.
    ///
    /// Naive mean field is iterated at this configuration's damping for `max(iters, 5000)` steps,
    /// since it costs nothing beside one part elimination.
    ///
    /// # Errors
    ///
    /// [`TooWide`] if some part's elimination order exceeds [`StructuredMeanField::max_width`].
    ///
    /// # Panics
    ///
    /// If `parts` is not a partition of `0..g.n`.
    pub fn tightest(&self, g: &Graph, parts: &[Vec<usize>]) -> Result<Structured, TooWide> {
        let cold = self.run(g, parts)?;
        let mf = crate::meanfield::naive_mean_field(g, self.beta, self.iters.max(5000), self.damping);
        let warm = self.run_from(g, parts, &mf.m)?;
        Ok(if warm.log_z > cold.log_z { warm } else { cold })
    }

    /// `F(b)` and the part magnetisations it was computed from.
    fn evaluate(
        &self,
        g: &Graph,
        built: &mut [Block],
        of: &[usize],
        b: &[f64],
    ) -> Result<(f64, Vec<f64>), TooWide> {
        let el = Elimination { max_width: self.max_width };
        let mut m = vec![0.0f64; g.n];
        let mut parts_log_z = 0.0;
        for blk in built.iter_mut() {
            for (k, &i) in blk.nodes.iter().enumerate() {
                blk.sub.h[k] = b[i];
            }
            parts_log_z += el
                .log_partition(&blk.sub, self.beta)?
                .log_z
                .expect("sum-product was run, so it reports log Z");
            let p = el.marginals(&blk.sub, self.beta)?;
            for (k, &i) in blk.nodes.iter().enumerate() {
                m[i] = 2.0 * p[k] - 1.0;
            }
        }
        let mut linear = 0.0;
        for i in 0..g.n {
            linear += (g.h[i] - b[i]) * m[i];
        }
        let mut across = 0.0;
        for i in 0..g.n {
            for e in g.offset[i]..g.offset[i + 1] {
                let j = g.nbr[e] as usize;
                if j > i && of[j] != of[i] {
                    across += g.w[e] * m[i] * m[j];
                }
            }
        }
        Ok((parts_log_z + self.beta * (linear + across), m))
    }
}

/// `b_i = h_i + Σ_{j ∉ part(i)} w_ij m_j`, written into `b`.
fn fields_from(g: &Graph, of: &[usize], m: &[f64], b: &mut [f64]) {
    for i in 0..g.n {
        let mut field = g.h[i];
        for e in g.offset[i]..g.offset[i + 1] {
            let j = g.nbr[e] as usize;
            if of[j] != of[i] {
                field += g.w[e] * m[j];
            }
        }
        b[i] = field;
    }
}

/// One part: its spins in global order, and the subgraph of its intra couplings.
struct Block {
    nodes: Vec<usize>,
    sub: Graph,
}

/// Build one subgraph per part, and the part index of every spin.
fn split(g: &Graph, parts: &[Vec<usize>]) -> (Vec<Block>, Vec<usize>) {
    let mut of = vec![usize::MAX; g.n];
    for (p, part) in parts.iter().enumerate() {
        for &i in part {
            assert!(i < g.n, "part {p} names spin {i}, past the graph's {}", g.n);
            assert_eq!(of[i], usize::MAX, "spin {i} is in two parts");
            of[i] = p;
        }
    }
    assert!(of.iter().all(|&p| p != usize::MAX), "every spin must be in a part");

    let mut local = vec![usize::MAX; g.n];
    let built = parts
        .iter()
        .map(|part| {
            for (k, &i) in part.iter().enumerate() {
                local[i] = k;
            }
            let mut gb = GraphBuilder::new(part.len());
            for (k, &i) in part.iter().enumerate() {
                for e in g.offset[i]..g.offset[i + 1] {
                    let j = g.nbr[e] as usize;
                    if j > i && of[j] == of[i] {
                        gb.couple(k, local[j], g.w[e]);
                    }
                }
            }
            Block { nodes: part.clone(), sub: gb.build() }
        })
        .collect();
    (built, of)
}

/// Every spin its own part: the family is fully factorised, and the bound is naive mean field's.
#[must_use]
pub fn singletons(n: usize) -> Vec<Vec<usize>> {
    (0..n).map(|i| vec![i]).collect()
}

/// Contiguous index blocks of at most `len` spins — the rows of a lattice built row-major.
///
/// # Panics
///
/// If `len` is zero.
#[must_use]
pub fn contiguous(n: usize, len: usize) -> Vec<Vec<usize>> {
    assert!(len > 0, "a part must hold at least one spin");
    (0..n).step_by(len).map(|s| (s..(s + len).min(n)).collect()).collect()
}

/// Greedy breadth-first parts of at most `max_size` **connected** spins.
///
/// Connectivity is what makes a part worth keeping: a part whose spins share no coupling has
/// nothing to model exactly, and the bound falls back to the factorised one on those spins.
///
/// # Panics
///
/// If `max_size` is zero.
#[must_use]
pub fn blocks(g: &Graph, max_size: usize) -> Vec<Vec<usize>> {
    assert!(max_size > 0, "a part must hold at least one spin");
    let mut taken = vec![false; g.n];
    let mut out = Vec::new();
    for seed in 0..g.n {
        if taken[seed] {
            continue;
        }
        let mut part = vec![seed];
        taken[seed] = true;
        let mut head = 0;
        while head < part.len() && part.len() < max_size {
            let i = part[head];
            head += 1;
            for e in g.offset[i]..g.offset[i + 1] {
                let j = g.nbr[e] as usize;
                if !taken[j] && part.len() < max_size {
                    taken[j] = true;
                    part.push(j);
                }
            }
        }
        part.sort_unstable();
        out.push(part);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exact::Elimination;
    use crate::graph::GraphBuilder;
    use crate::ising;
    use crate::meanfield::naive_mean_field;
    use crate::rng::Pcg;

    fn truth(g: &Graph, beta: f64) -> f64 {
        Elimination { max_width: 24 }
            .log_partition(g, beta)
            .expect("test models are narrow")
            .log_z
            .expect("sum-product was run")
    }

    fn random_tree(n: usize, seed: u64) -> Graph {
        let mut rng = Pcg::new(seed, 0);
        let mut gb = GraphBuilder::new(n);
        for i in 1..n {
            gb.couple((rng.f64() * i as f64) as usize, i, 2.0 * rng.f64() - 1.0);
        }
        for i in 0..n {
            gb.bias(i, rng.f64() - 0.5);
        }
        gb.build()
    }

    /// A random-coupling ring: frustrated, and narrow enough that `ln Z` is exact.
    fn glass_ring(n: usize, seed: u64) -> Graph {
        let mut rng = Pcg::new(seed, 0);
        let mut gb = GraphBuilder::new(n);
        for i in 0..n {
            gb.couple(i, (i + 1) % n, 2.0 * rng.f64() - 1.0);
            gb.bias(i, rng.f64() - 0.5);
        }
        gb.build()
    }

    /// A dense frustrated model, still narrow enough to eliminate at width 24.
    fn dense_glass(n: usize, seed: u64) -> Graph {
        let mut rng = Pcg::new(seed, 1);
        let mut gb = GraphBuilder::new(n);
        for i in 0..n {
            for j in (i + 1)..n {
                if rng.f64() < 0.35 {
                    gb.couple(i, j, 3.0 * rng.f64() - 1.5);
                }
            }
            gb.bias(i, 2.0 * rng.f64() - 1.0);
        }
        gb.build()
    }

    /// One part holding every spin makes `q` the model itself, so the bound IS `ln Z`.
    ///
    /// The exact oracle for this module, and the only test here that asserts equality with the
    /// truth rather than an inequality against it: with nothing across parts there is no
    /// approximation left to make, `b = h` is a fixed point at once, and any discrepancy is a bug
    /// in the free-energy algebra rather than a looseness in the family.
    #[test]
    fn one_part_holding_everything_is_exactly_log_z() {
        for (name, g, beta) in [
            ("ring", ising::ring(14, 1.0, 0.3), 0.7),
            ("tree", random_tree(18, 4), 1.1),
            ("glass ring", glass_ring(16, 9), 2.0),
            ("lattice", ising::lattice2d(4, 1.0), 0.45),
        ] {
            let all: Vec<usize> = (0..g.n).collect();
            let st = StructuredMeanField::new(beta).run(&g, &[all]).expect("narrow");
            let exact = truth(&g, beta);
            assert!(
                (st.log_z - exact).abs() < 1e-9,
                "{name}: one part gave {} for an exact ln Z of {exact}",
                st.log_z
            );
            assert!(st.converged(1e-12), "{name}: no across couplings, so nothing can move");
        }
    }

    /// Singleton parts ARE naive mean field: `F(b) = gibbs_bogoliubov(tanh(β b))`, at every `b`.
    ///
    /// The other end of the same identity, and stated at arbitrary `b` rather than at a fixed
    /// point so that nothing about convergence enters it. With no couplings inside a part, `q_p`
    /// is one free spin, `m_i = tanh(β b_i)`, and the three terms of `F` collapse term for term
    /// onto [`crate::meanfield::gibbs_bogoliubov`]. Held to `1e-12` at random fields — an
    /// inequality here would pass for an implementation off by a constant.
    #[test]
    fn singleton_parts_are_naive_mean_field_at_every_field() {
        let mut rng = Pcg::new(17, 0);
        for (name, g) in [
            ("lattice", ising::lattice2d(4, 1.0)),
            ("ring", ising::ring(12, 1.0, 0.2)),
            ("glass ring", glass_ring(16, 3)),
            ("tree", random_tree(16, 2)),
        ] {
            let parts = singletons(g.n);
            for beta in [0.1, 0.5, 1.7] {
                let smf = StructuredMeanField { beta, iters: 1, damping: 0.5, max_width: 4 };
                for _ in 0..8 {
                    let b: Vec<f64> = (0..g.n).map(|_| 4.0 * rng.f64() - 2.0).collect();
                    let m: Vec<f64> = b.iter().map(|&x| (beta * x).tanh()).collect();
                    let f = smf.bound_at(&g, &parts, &b).expect("a singleton has no width");
                    let gb = crate::meanfield::gibbs_bogoliubov(&g, beta, &m);
                    assert!(
                        (f - gb).abs() < 1e-12,
                        "{name} beta {beta}: structured {f} vs Gibbs-Bogoliubov {gb}"
                    );
                }
            }
        }
    }

    /// And the singleton ITERATION lands where naive mean field's does.
    ///
    /// Separate from the algebraic identity above, because this one is about two different
    /// iterations — Gauss–Seidel on `m`, Jacobi on `b` — reaching the same fixed point. Run at
    /// temperatures where that fixed point is unique, so agreement is the identity rather than
    /// luck, and to `1e-9` in `ln Z`; the magnetisations are held looser because the returned
    /// iterate is the best-bound one rather than the last.
    #[test]
    fn the_singleton_iteration_matches_naive_mean_field() {
        let mut biased = ising::lattice2d(4, 1.0);
        for i in 0..biased.n {
            biased.h[i] = 0.2 * ((i % 3) as f64 - 1.0);
        }
        for (name, g) in [
            ("biased lattice", biased),
            ("ring", ising::ring(12, 1.0, 0.2)),
            ("glass ring", glass_ring(16, 3)),
            ("tree", random_tree(16, 2)),
        ] {
            for beta in [0.1, 0.2, 0.35] {
                let mf = naive_mean_field(&g, beta, 20000, 0.5);
                let smf = StructuredMeanField { beta, iters: 20000, damping: 0.5, max_width: 4 };
                let st = smf.run(&g, &singletons(g.n)).expect("a singleton has no width");
                assert!(mf.converged(1e-10) && st.converged(1e-10), "{name} beta {beta}");
                assert!(
                    (st.log_z - mf.log_z).abs() < 1e-9,
                    "{name} beta {beta}: structured {} vs naive {}",
                    st.log_z,
                    mf.log_z
                );
                for i in 0..g.n {
                    assert!(
                        (st.m[i] - mf.m[i]).abs() < 1e-7,
                        "{name} beta {beta} site {i}: {} vs {}",
                        st.m[i],
                        mf.m[i]
                    );
                }
            }
        }
    }

    /// The bound is below `ln Z` at fields nobody optimised — which is the theorem, not the run.
    ///
    /// Gibbs–Bogoliubov holds at every `q` in the family, so `F(b)` is sound at arbitrary `b`. A
    /// test that only ever evaluated converged fields would pass for an implementation whose
    /// validity depended on convergence, and this crate's callers cut iterations short.
    #[test]
    fn the_bound_is_below_log_z_at_arbitrary_fields() {
        let mut rng = Pcg::new(31, 0);
        for (name, g) in [
            ("ring", ising::ring(16, 1.0, 0.25)),
            ("lattice", ising::lattice2d(4, 1.0)),
            ("glass ring", glass_ring(18, 5)),
            ("tree", random_tree(18, 8)),
            ("dense glass", dense_glass(14, 3)),
        ] {
            for beta in [0.15, 0.6, 1.4, 3.0] {
                let exact = truth(&g, beta);
                for parts in [singletons(g.n), contiguous(g.n, 3), blocks(&g, 5), blocks(&g, 8)] {
                    let smf = StructuredMeanField { beta, iters: 100, damping: 0.5, max_width: 20 };
                    for _ in 0..6 {
                        let b: Vec<f64> = (0..g.n).map(|_| 6.0 * rng.f64() - 3.0).collect();
                        let f = smf.bound_at(&g, &parts, &b).expect("narrow parts");
                        assert!(
                            f <= exact + 1e-9,
                            "{name} beta {beta}: F(b) = {f} above ln Z = {exact}"
                        );
                    }
                    let st = smf.tightest(&g, &parts).expect("narrow parts");
                    assert!(
                        st.log_z <= exact + 1e-9,
                        "{name} beta {beta}: iterated bound {} above ln Z = {exact}",
                        st.log_z
                    );
                }
            }
        }
    }

    /// The bound rises monotonically as the parts grow, and reaches `ln Z` when one part holds all.
    ///
    /// A refinement ladder on one model: parts of 1, 2, 3, 4, 6, 8, 12, 24 spins of a 24-ring. The
    /// bottom rung is naive mean field and the top rung is exact, so the ladder measures what the
    /// extra structure buys rather than asserting that it buys something.
    #[test]
    fn the_bound_rises_as_the_parts_grow() {
        let g = ising::ring(24, 1.0, 0.15);
        for beta in [0.4, 1.0] {
            let exact = truth(&g, beta);
            let mut previous = f64::NEG_INFINITY;
            for len in [1usize, 2, 3, 4, 6, 8, 12, 24] {
                let st = StructuredMeanField::new(beta)
                    .tightest(&g, &contiguous(24, len))
                    .expect("a chain of any length is width 1");
                assert!(
                    st.log_z > previous,
                    "beta {beta}: parts of {len} gave {}, not above the previous rung {previous}",
                    st.log_z
                );
                assert!(st.log_z <= exact + 1e-9, "beta {beta}: parts of {len} above ln Z");
                previous = st.log_z;
            }
            assert!((previous - exact).abs() < 1e-9, "beta {beta}: the top rung must be exact");
        }
    }

    /// Tighter than naive mean field, on every family measured.
    ///
    /// The whole point of the module, so it is asserted per model rather than on an average, and
    /// against exact `ln Z` so a reader can see how much of the remaining gap each one closes.
    #[test]
    fn tighter_than_naive_mean_field() {
        let cases: Vec<(&str, Graph, f64, Vec<Vec<usize>>)> = vec![
            ("ferro ring, arcs of 8", ising::ring(24, 1.0, 0.0), 0.6, contiguous(24, 8)),
            ("ferro ring cold", ising::ring(24, 1.0, 0.0), 1.2, contiguous(24, 8)),
            ("biased ring, arcs of 5", ising::ring(20, 1.0, 0.4), 0.3, contiguous(20, 5)),
            ("lattice 6x6, rows", ising::lattice2d(6, 1.0), 0.2, contiguous(36, 6)),
            ("lattice 6x6, rows, critical", ising::lattice2d(6, 1.0), 0.44, contiguous(36, 6)),
            ("glass ring, arcs of 5", glass_ring(20, 7), 1.5, contiguous(20, 5)),
            ("glass ring, arcs of 5, cold", glass_ring(20, 7), 3.0, contiguous(20, 5)),
            ("dense glass, blocks of 6", dense_glass(18, 3), 0.5, blocks(&dense_glass(18, 3), 6)),
        ];
        for (name, g, beta, parts) in cases {
            let exact = truth(&g, beta);
            let mf = naive_mean_field(&g, beta, 20000, 0.5).log_z;
            let st = StructuredMeanField::new(beta).tightest(&g, &parts).expect("narrow parts");
            assert!(st.log_z <= exact + 1e-9, "{name}: {} above ln Z {exact}", st.log_z);
            assert!(
                st.log_z > mf,
                "{name}: structured {} did NOT beat naive {mf} (exact {exact})",
                st.log_z
            );
        }
    }

    /// A cold start CAN lose to naive mean field, and the naive seed is what recovers it.
    ///
    /// There is no theorem here: the structured family does not contain the factorised one, so a
    /// frustrated model can leave the field iteration in a local optimum well below where naive
    /// mean field stopped. Measured across 1800 dense-glass cases (60 instances x 6 temperatures x
    /// 5 partitions), the `m = 0.01` start lost 99 times and the naive-seeded start lost none — so
    /// this asserts BOTH halves on one instance, since a module claiming a tighter bound must say
    /// exactly when it does not have one.
    #[test]
    fn a_cold_start_can_lose_to_naive_mean_field() {
        let g = dense_glass(16, 8);
        let beta = 3.0;
        let parts = contiguous(16, 4);
        let exact = truth(&g, beta);
        let mf = naive_mean_field(&g, beta, 20000, 0.5);
        let smf = StructuredMeanField::new(beta);
        let cold = smf.run(&g, &parts).expect("narrow parts");
        let warm = smf.run_from(&g, &parts, &mf.m).expect("narrow parts");
        assert!(
            cold.log_z < mf.log_z - 1.0,
            "this test is about a LOSS: cold {} vs naive {}",
            cold.log_z,
            mf.log_z
        );
        assert!(
            warm.log_z >= mf.log_z - 1e-9,
            "the naive seed should recover it: warm {} vs naive {}",
            warm.log_z,
            mf.log_z
        );
        // Both are sound whichever wins, which is what makes taking the maximum legitimate.
        assert!(cold.log_z <= exact + 1e-9 && warm.log_z <= exact + 1e-9);
        assert!(smf.tightest(&g, &parts).expect("narrow parts").log_z >= cold.log_z.max(warm.log_z) - 1e-9);
    }

    /// [`blocks`] and [`contiguous`] return partitions: every spin exactly once, parts within size.
    #[test]
    fn the_partition_helpers_partition() {
        for g in [ising::lattice2d(5, 1.0), glass_ring(17, 2), random_tree(23, 6)] {
            for size in [1usize, 2, 3, 7, 40] {
                for (tag, parts) in [("blocks", blocks(&g, size)), ("contiguous", contiguous(g.n, size))] {
                    let mut seen = vec![0usize; g.n];
                    for part in &parts {
                        assert!(!part.is_empty() && part.len() <= size, "{tag} {size}: bad part size");
                        for &i in part {
                            seen[i] += 1;
                        }
                    }
                    assert!(seen.iter().all(|&c| c == 1), "{tag} {size}: not a partition");
                }
            }
            // A `blocks` part is connected, which is the property that makes it worth keeping.
            for part in blocks(&g, 6) {
                if part.len() > 1 {
                    let inside: Vec<usize> = part.clone();
                    for &i in &part {
                        let touches = (g.offset[i]..g.offset[i + 1])
                            .any(|e| inside.contains(&(g.nbr[e] as usize)));
                        assert!(touches, "spin {i} shares no coupling with its own part");
                    }
                }
            }
        }
    }
}
