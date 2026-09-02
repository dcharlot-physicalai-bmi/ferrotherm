//! COPY-gate sparsification: bound a model's degree by splitting its heavy variables.
//!
//! A sampling fabric has a fixed, sparse topology. A model denser than the fabric has two routes
//! onto it, and they are not the same thing:
//!
//! * **Minor embedding** ([`crate::embed`]) *places* the model onto one specific hardware graph,
//!   mapping each variable to a chain of physical sites. It needs the machine in hand and its
//!   answer is about that machine.
//! * **Sparsification**, here, *rewrites the model* so that no variable has more than `d`
//!   neighbours, with no machine involved at all. A variable of degree `k` becomes `c` COPIES bound
//!   into a path by a strong ferromagnetic coupling `W0`, its edges shared out among them and its
//!   bias split evenly. The result is a different, larger, sparser model with the SAME ground
//!   states, and any degree-`d` fabric can then take it.
//!
//! The field states this as an open problem in exactly those terms — OPUSLab's answer is one MATLAB
//! file from June 2025, and their repository named `SparsifyDenseGraph` is empty — so it is worth
//! owning, and worth owning with the correctness property checked rather than asserted.
//!
//! # The copy count is the embedding bound, and that is not a coincidence
//!
//! A path of `c` copies offers `c(d−2) + 2` free ports: the two ends spend one coupling each on the
//! path and every interior copy spends two. So a variable of degree `k` needs
//! `c ≥ ⌈(k−2)/(d−2)⌉` copies — which is character for character the count
//! [`crate::embed::site_lower_bound`] derives for a chain, because it is the same port-counting
//! argument seen from the other side. A chain of hardware sites and a path of logical copies are
//! the same object; only who owns it differs.
//!
//! # What makes it ground-state preserving
//!
//! Every ground state of the sparsified model has all copies of every variable agreeing, provided
//! `W0` is large enough — and "large enough" is derivable rather than tuned. Suppose some variable's
//! copies disagree. Flipping a contiguous block of them repairs at least one broken copy edge, worth
//! `2·W0`, while changing every logical term on that block by at most `2·W_v`, where `W_v` is the
//! variable's own bias plus every coupling on it. So any `W0 > W_v` makes disagreement strictly
//! unprofitable, and [`copy_strength`] returns that bound with a margin.
//!
//! This is **checked by enumeration**, not argued: `sparsification_preserves_the_ground_states`
//! enumerates the sparsified model in full, and requires every one of its ground states to have all
//! copies agreeing and to project onto a ground state of the original — and requires every ground
//! state of the original to be reached. A companion test drops `W0` below the bound and requires the
//! property to FAIL, so the derivation is doing work rather than decorating a test that would pass
//! at any strength.

use crate::graph::{Graph, GraphBuilder};

/// A model rewritten to fit a degree budget, and the map back.
pub struct Sparsified {
    /// The sparse model. Larger than the original: one node per copy.
    pub graph: Graph,
    /// `copies[v]` are the nodes representing logical variable `v`, in path order.
    pub copies: Vec<Vec<u32>>,
    /// The copy coupling used to bind each path.
    pub w0: f64,
    /// The degree budget asked for.
    pub budget: usize,
    /// The maximum degree actually reached, which is at most `budget`.
    pub achieved: usize,
    /// `E_logical(s) = E_sparse(s') + offset` when every copy set agrees.
    ///
    /// The copy couplings are pure bookkeeping: they contribute `−W0` per copy edge in any agreeing
    /// state, the same constant for every such state, so they order answers identically and shift
    /// every energy by this amount. Reporting a sparsified energy without it compares a number from
    /// one model against a number from another.
    pub offset: f64,
}

/// Why a model could not be sparsified to a budget.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refused {
    /// A budget of two or less: a path of any length offers only two ports, so no number of copies
    /// makes a variable of degree three fit. The bound divides by `d − 2` and this is that.
    BudgetTooSmall { budget: usize },
}

impl core::fmt::Display for Refused {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Refused::BudgetTooSmall { budget } => write!(
                f,
                "a degree budget of {budget} cannot be met by splitting: a path of copies spends \
                 one coupling at each end and two in the middle, so it offers c(d-2)+2 ports and \
                 that is not increasing in c below d = 3. Raise the budget to at least 3"
            ),
        }
    }
}

/// Copies a variable of degree `k` needs to fit a degree budget of `d`.
///
/// One when it already fits. Otherwise the smallest `c` with `c(d−2) + 2 ≥ k`, which is the port
/// count of a path of `c` copies. Identical to the per-variable term of
/// [`crate::embed::site_lower_bound`], for the same reason.
pub fn copies_for(k: usize, d: usize) -> usize {
    if d < 3 {
        return 1; // meaningless; callers are refused before reaching here
    }
    if k <= d {
        1
    } else {
        (k - 2).div_ceil(d - 2).max(2)
    }
}

/// How much heavier than the model a copy coupling has to be.
///
/// The derivation needs `W0` strictly greater than the heaviest variable's total weight; anything
/// above that works and nothing at or below it does. Two, rather than a hair over one, because at
/// exactly the bound a broken copy set is EQUAL in energy to an intact one, and a sampler is
/// entitled to return either — floating-point rounding puts real models at that edge.
///
/// Mirrors [`crate::embed::DEFAULT_CHAIN_MULTIPLE`], which solves the same problem for chains and
/// sits higher because a chain's neighbours are the hardware's, not the model's, and cannot be
/// bounded as tightly.
pub const DEFAULT_COPY_MULTIPLE: f64 = 2.0;

/// A copy coupling strong enough to hold every copy set together, from the model's own weights.
///
/// `max over v of (|h_v| + sum of |w| on v)`, times [`DEFAULT_COPY_MULTIPLE`]. Returns 1.0 for a
/// model with no weight at all, so an empty or zeroed graph still gets a coupling that binds.
pub fn copy_strength(g: &Graph) -> f64 {
    let heaviest = (0..g.n)
        .map(|v| {
            g.h[v].abs()
                + (g.offset[v]..g.offset[v + 1]).map(|k| g.w[k].abs()).sum::<f64>()
        })
        .fold(0.0f64, f64::max);
    if heaviest > 0.0 {
        heaviest * DEFAULT_COPY_MULTIPLE
    } else {
        1.0
    }
}

/// Rewrite `g` so no variable exceeds degree `budget`, using [`copy_strength`].
pub fn sparsify(g: &Graph, budget: usize) -> Result<Sparsified, Refused> {
    sparsify_with(g, budget, copy_strength(g))
}

/// As [`sparsify`], with the copy coupling stated.
///
/// A `w0` below [`copy_strength`] does not error: it produces a model whose ground states may have
/// copies disagreeing, which is a real thing to want to study and exactly what
/// `a_copy_coupling_below_the_bound_stops_preserving_the_ground_states` does. [`project`] reports
/// which variables broke, so the caller can see it happen rather than read a silently wrong answer.
pub fn sparsify_with(g: &Graph, budget: usize, w0: f64) -> Result<Sparsified, Refused> {
    if budget < 3 {
        return Err(Refused::BudgetTooSmall { budget });
    }
    // Lay out the copies: variable v owns a contiguous block, so `copies` is also the inverse map.
    let mut copies: Vec<Vec<u32>> = Vec::with_capacity(g.n);
    let mut next = 0u32;
    for v in 0..g.n {
        let k = g.offset[v + 1] - g.offset[v];
        let c = copies_for(k, budget);
        copies.push((0..c).map(|i| next + i as u32).collect());
        next += c as u32;
    }
    let mut gb = GraphBuilder::new(next as usize);

    // Free ports per copy: the budget less what the path itself spends. A lone copy spends nothing.
    let mut cap: Vec<usize> = Vec::with_capacity(next as usize);
    let mut offset = 0.0f64;
    for v in 0..g.n {
        let c = copies[v].len();
        for i in 0..c {
            let spent = if c == 1 {
                0
            } else if i == 0 || i + 1 == c {
                1
            } else {
                2
            };
            cap.push(budget - spent);
        }
        // The path, and the bias split evenly across it so the total field is unchanged.
        for i in 0..c.saturating_sub(1) {
            gb.couple(copies[v][i] as usize, copies[v][i + 1] as usize, w0);
            offset += w0;
        }
        let share = g.h[v] / c as f64;
        for &node in &copies[v] {
            gb.bias(node as usize, share);
        }
    }

    // Hand each logical edge to the first copy of each endpoint with a port left. Total capacity is
    // at least the degree at both ends by construction, so first-fit cannot fail -- and the
    // assertion says so rather than leaving a silent miscount to surface as a wrong graph.
    let mut used = vec![0usize; next as usize];
    for u in 0..g.n {
        for k in g.offset[u]..g.offset[u + 1] {
            let v = g.nbr[k] as usize;
            if v <= u {
                continue; // each undirected edge once
            }
            let pick = |copies: &Vec<Vec<u32>>, used: &Vec<usize>, x: usize| -> u32 {
                *copies[x]
                    .iter()
                    .find(|&&n| used[n as usize] < cap[n as usize])
                    .expect("a copy set is sized to hold every edge on its variable")
            };
            let (a, b) = (pick(&copies, &used, u), pick(&copies, &used, v));
            used[a as usize] += 1;
            used[b as usize] += 1;
            gb.couple(a as usize, b as usize, g.w[k]);
        }
    }

    let graph = gb.build();
    let achieved = graph.max_degree();
    debug_assert!(achieved <= budget, "the budget is what the copy count was chosen for");
    Ok(Sparsified { graph, copies, w0, budget, achieved, offset })
}

impl core::fmt::Debug for Sparsified {
    /// `Graph` carries no `Debug` and printing a CSR block would be pages nobody reads. The shape
    /// and what it cost are what a caller wants at a breakpoint.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "Sparsified {{ {} variables -> {} copies, degree {} of a {} budget, w0 {}, offset {} }}",
            self.copies.len(),
            self.graph.n,
            self.achieved,
            self.budget,
            self.w0,
            self.offset
        )
    }
}

impl Sparsified {
    /// The logical energy of a sparsified state, or `None` when a copy set disagrees.
    ///
    /// `None` rather than a number, because there is no logical state to price: a variable whose
    /// copies disagree has not been assigned a value, and returning the sparse energy would be
    /// answering a question about a different model.
    pub fn logical_energy(&self, g: &Graph, state: &[i8]) -> Option<f64> {
        let (s, broken) = project(self, state);
        broken.is_empty().then(|| g.energy(&s))
    }
}

/// Read a sparsified state back as a logical one, and say which variables broke.
///
/// The value of a variable is the value its copies agree on. Where they do not, the majority is
/// returned so the caller still has a complete state to look at, and the index is reported — a
/// broken copy set means the coupling lost, and a caller that ignores the list is reading a
/// majority vote as though it were an answer.
pub fn project(s: &Sparsified, state: &[i8]) -> (Vec<i8>, Vec<usize>) {
    let mut out = Vec::with_capacity(s.copies.len());
    let mut broken = Vec::new();
    for (v, set) in s.copies.iter().enumerate() {
        let up = set.iter().filter(|&&n| state[n as usize] > 0).count();
        if up != 0 && up != set.len() {
            broken.push(v);
        }
        out.push(if up * 2 >= set.len() { 1i8 } else { -1i8 });
    }
    (out, broken)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Pcg;

    fn clique(k: usize) -> Graph {
        let mut gb = GraphBuilder::new(k);
        for i in 0..k {
            for j in (i + 1)..k {
                gb.couple(i, j, 1.0);
            }
        }
        gb.build()
    }

    /// A random dense graph with biases, so the property is tested on weights and not only on a
    /// symmetric special case.
    fn glass(n: usize, p: f64, seed: u64) -> Graph {
        let mut r = Pcg::new(seed, 0x5A17);
        let mut gb = GraphBuilder::new(n);
        for i in 0..n {
            gb.bias(i, r.f64() * 2.0 - 1.0);
            for j in (i + 1)..n {
                if r.f64() < p {
                    gb.couple(i, j, r.f64() * 2.0 - 1.0);
                }
            }
        }
        gb.build()
    }

    /// Every state of a graph, in linear index order.
    fn states(n: usize) -> impl Iterator<Item = Vec<i8>> {
        (0..(1usize << n)).map(move |mask| {
            (0..n).map(|b| if mask >> b & 1 == 1 { 1i8 } else { -1 }).collect()
        })
    }

    /// The ground states of a graph, by enumeration.
    fn ground(g: &Graph) -> (f64, std::collections::BTreeSet<Vec<i8>>) {
        let mut best = f64::INFINITY;
        let mut set = std::collections::BTreeSet::new();
        for s in states(g.n) {
            let e = g.energy(&s);
            if e < best - 1e-9 {
                best = e;
                set.clear();
            }
            if e <= best + 1e-9 {
                set.insert(s);
            }
        }
        (best, set)
    }

    /// THE CORRECTNESS PROPERTY, checked exhaustively rather than argued.
    ///
    /// For each model: sparsify it, enumerate the whole sparsified state space, and require
    ///   1. every ground state of the sparsified model has all copies agreeing,
    ///   2. each projects onto a ground state of the original,
    ///   3. every ground state of the original is reached, and
    ///   4. the energies line up through `offset`.
    ///
    /// Item 3 is the one an implementation can quietly fail: a rewrite that LOST a ground state
    /// would satisfy the first two and still be wrong.
    #[test]
    fn sparsification_preserves_the_ground_states() {
        // Sizes chosen so the SPARSIFIED model stays enumerable, which is the binding constraint
        // and not an obvious one: at budget 3 the port count is c(d-2)+2 = c+2, so copies grow
        // like the degree itself and a 7-node glass becomes 28 spins. `LIMIT` below is checked
        // before anything is enumerated, so a future case that blows past it fails in a second
        // instead of running for a week -- which is how this list was first written.
        const LIMIT: usize = 18;
        let cases: [(&str, Graph, usize); 4] = [
            ("K_5 -> deg 3", clique(5), 3),
            ("K_6 -> deg 4", clique(6), 4),
            ("glass(6, 0.9) -> deg 4", glass(6, 0.9, 11), 4),
            ("glass(8, 0.5) -> deg 4", glass(8, 0.5, 5), 4),
        ];
        for (name, g, budget) in cases {
            let s = sparsify(&g, budget).expect("budget is at least 3");
            assert!(s.achieved <= budget, "{name}: degree {} exceeds {budget}", s.achieved);
            assert!(g.max_degree() > budget, "{name} is not dense enough to be a test");
            assert!(
                s.graph.n <= LIMIT,
                "{name}: sparsifies to {} spins, past the {LIMIT} this test can enumerate",
                s.graph.n
            );

            let (log_e, log_gs) = ground(&g);
            let (sp_e, sp_gs) = ground(&s.graph);

            let mut reached = std::collections::BTreeSet::new();
            for state in &sp_gs {
                let (proj, broken) = project(&s, state);
                assert!(broken.is_empty(), "{name}: a ground state has copies disagreeing: {broken:?}");
                assert!(log_gs.contains(&proj), "{name}: projects outside the ground manifold");
                reached.insert(proj);
            }
            assert_eq!(reached, log_gs, "{name}: a ground state of the original was LOST");
            assert!(
                (sp_e + s.offset - log_e).abs() < 1e-9,
                "{name}: energies do not line up: {sp_e} + {} vs {log_e}",
                s.offset
            );
            // And the convenience accessor agrees with the arithmetic above.
            let any = sp_gs.iter().next().expect("non-empty");
            assert!((s.logical_energy(&g, any).expect("agrees") - log_e).abs() < 1e-9);
        }
    }

    /// The derivation is doing work: below the bound, the property FAILS.
    ///
    /// Without this the correctness test would pass at any coupling strength and would be checking
    /// that enumeration works rather than that `copy_strength` is right.
    #[test]
    fn a_copy_coupling_below_the_bound_stops_preserving_the_ground_states() {
        // Budget 4, not 3: at 3 a path offers c+2 ports, so a 7-node glass sparsifies to 28 spins
        // and enumerating it twice is not a test, it is a week. The property does not depend on
        // the budget.
        let g = glass(6, 0.9, 3);
        let full = sparsify(&g, 4).unwrap();
        assert!(full.w0 > 0.0);
        assert!(full.graph.n <= 18, "{} spins is too many to enumerate", full.graph.n);

        // A coupling so weak the copies are barely tied together at all.
        let weak = sparsify_with(&g, 4, 0.01).unwrap();
        let (_, sp_gs) = ground(&weak.graph);
        let any_broken = sp_gs.iter().any(|s| !project(&weak, s).1.is_empty());
        assert!(
            any_broken,
            "at w0 = 0.01 some ground state must have a copy set disagreeing, or this test is not \
             exercising the bound it exists to justify"
        );

        // And at the derived strength, none does.
        let (_, ok_gs) = ground(&full.graph);
        assert!(ok_gs.iter().all(|s| project(&full, s).1.is_empty()));
    }

    /// The copy count is the embedding bound. Same argument, so it must be the same number.
    #[test]
    fn the_copy_count_is_the_embedding_site_bound() {
        for d in 3..8usize {
            // A degree-d hardware graph to ask `site_lower_bound` against: a d-regular-ish clique.
            let hw = clique(d + 1);
            assert_eq!(hw.max_degree(), d);
            for k in 2..14usize {
                let logical = clique(k + 1); // one variable of degree k, and k of degree k
                let mine: usize = (0..logical.n).map(|_| copies_for(k, d)).sum();
                assert_eq!(
                    mine,
                    crate::embed::site_lower_bound(&logical, &hw),
                    "degree {k} onto budget {d}"
                );
            }
        }
    }

    #[test]
    fn a_budget_below_three_is_refused_with_the_reason() {
        for budget in [0usize, 1, 2] {
            let e = sparsify(&clique(5), budget).unwrap_err();
            assert_eq!(e, Refused::BudgetTooSmall { budget });
            assert!(format!("{e}").contains("c(d-2)+2"), "the refusal shows the arithmetic");
        }
        // Three is the smallest budget that can work, and it does.
        assert!(sparsify(&clique(5), 3).is_ok());
    }

    /// A model already inside the budget is left alone: one copy each, no copy edges, no offset.
    #[test]
    fn a_model_that_already_fits_is_not_rewritten() {
        let g = crate::ising::lattice2d(6, 1.0);
        assert_eq!(g.max_degree(), 4);
        let s = sparsify(&g, 4).unwrap();
        assert_eq!(s.graph.n, g.n, "no copies added");
        assert!(s.copies.iter().all(|c| c.len() == 1));
        assert_eq!(s.offset, 0.0, "no copy edges to pay for");
        assert_eq!(s.graph.n_edges, g.n_edges);
    }
}

// ---- machine-checked theorems ---------------------------------------------------------------------
//
// These are PROOFS, not tests: `cargo kani` explores every value in the stated ranges by bounded
// model checking, so a pass is exhaustive over the domain rather than a sample of it. They compile
// only under the Kani toolchain (`cfg(kani)`), so the crate's zero-dependency promise is untouched.
// `scripts/check-proofs.sh` runs them.
#[cfg(kani)]
mod proofs {
    use super::*;

    /// The copy count is sufficient AND minimal, for every degree and budget in range.
    ///
    /// Sufficiency: a path of `copies_for(k, d)` copies offers at least `k` ports. Minimality: one
    /// copy fewer offers too few. Together these say the count is exactly right, not merely safe —
    /// and the whole ground-state argument stands on sufficiency, while the site economics stand
    /// on minimality.
    #[kani::proof]
    fn copies_for_is_sufficient_and_minimal() {
        let k: usize = kani::any();
        let d: usize = kani::any();
        kani::assume((3..=32).contains(&d));
        kani::assume((1..=64).contains(&k));
        let c = copies_for(k, d);
        let capacity = |c: usize| if c <= 1 { d } else { c * (d - 2) + 2 };
        assert!(capacity(c) >= k, "sufficient");
        if c > 1 {
            assert!(capacity(c - 1) < k, "minimal");
        }
    }

    /// The `.max(2)` in `copies_for` is provably redundant: for `k > d ≥ 3` the ceiling is already
    /// at least two. Kept in the code as belt-and-braces; proved here so the belt is known to be
    /// decorative rather than load-bearing.
    #[kani::proof]
    fn the_ceiling_already_exceeds_one_past_the_budget() {
        let k: usize = kani::any();
        let d: usize = kani::any();
        kani::assume((3..=32).contains(&d));
        kani::assume(k > d && k <= 64);
        assert!((k - 2).div_ceil(d - 2) >= 2);
        assert_eq!(copies_for(k, d), (k - 2).div_ceil(d - 2));
    }
}
