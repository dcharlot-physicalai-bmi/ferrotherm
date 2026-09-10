//! Uniform torque compensation: a chain strength read off the model's own statistics.
//!
//! Boothby, Raymond and King, *Programming the D-Wave QPU: Setting the Chain Strength*, D-Wave
//! technical report 14-1041A-A (2020) — the rule `dwave-system` ships as its default
//! `uniform_torque_compensation`. A degree-`d` variable's chain is pulled on by `d` couplings whose
//! signs are effectively independent, so the torque trying to tear it apart grows like
//! `sqrt(<J^2> d)`: not like `max|J|`, and not like a constant. [`strength`] is that number, with
//! the shipped prefactor `sqrt(2)`.
//!
//! [`sweep`] is what it costs, since a chain-strength rule is only worth what a sweep says it is:
//! break rate and achieved logical energy against an EXACT optimum, per strength.
//!
//! # Measured, on ten-variable cliques embedded in `chimera(4, 4, 4)`
//!
//! Against [`crate::embed::DEFAULT_CHAIN_MULTIPLE`] `* max|coefficient|`, which is what
//! [`crate::embed::apply`] uses. On uniform +-1 couplings the two rules agree to 6% (4.24 against
//! 4.0) and neither breaks a chain nor misses an optimum — this rule is NOT better there, and that
//! family is what the constant was tuned on. They part off it. One coupling of 8 among 45 moves
//! `max|J|` eightfold and the RMS by half: the max rule holds chains at 32 where the knee is 4, and
//! that costs 3.75 in mean energy and 6 of 8 optima. Give the clique +-5 FIELDS, which no torque
//! argument mentions and `max|coefficient|` reads anyway, and it overshoots to 20, 4.7x the knee,
//! while this rule does not move.
//!
//! Both over-strong rows report zero broken chains. Too weak announces itself; too strong is
//! silent, which is why the sweep is scored against a proof rather than against its own best row.

use crate::exact::{Elimination, TooWide};
use crate::graph::Graph;
use crate::{embed, tempering};

/// The prefactor `dwave-system` ships, `sqrt(2)` to four figures.
///
/// The report's recommended range to try around it is `[0.5, 2]`.
pub const UTC_PREFACTOR: f64 = 1.414;

/// Root-mean-square coupling, `sqrt(<J^2>)` over the model's undirected edges. `0.0` if there are
/// none.
#[must_use]
pub fn rms_coupling(logical: &Graph) -> f64 {
    if logical.n_edges == 0 {
        return 0.0;
    }
    // Each undirected edge appears twice in CSR, so the mean over `w` is the mean over edges.
    let sq: f64 = logical.w.iter().map(|j| j * j).sum();
    (sq / logical.w.len() as f64).sqrt()
}

/// Mean degree over all variables, `2 * n_edges / n`. Isolated variables count, and lower it.
///
/// `0.0` for an empty model.
#[must_use]
pub fn mean_degree(logical: &Graph) -> f64 {
    if logical.n == 0 {
        return 0.0;
    }
    2.0 * logical.n_edges as f64 / logical.n as f64
}

/// The uniform-torque-compensation chain strength: `1.414 * sqrt(<J^2>) * sqrt(mean degree)`.
///
/// `1.0` for a model with no couplings, which is what the reference implementation returns and
/// keeps a chain held together rather than collapsing it to zero.
#[must_use]
pub fn strength(logical: &Graph) -> f64 {
    strength_with(logical, UTC_PREFACTOR)
}

/// [`strength`] with the prefactor chosen; `1.0` for a model with no couplings.
///
/// # Panics
///
/// If `prefactor` is not finite and positive — a chain strength of zero or NaN holds nothing, and
/// silently substituting the default would hide the caller's mistake.
#[must_use]
pub fn strength_with(logical: &Graph, prefactor: f64) -> f64 {
    assert!(prefactor.is_finite() && prefactor > 0.0, "prefactor must be finite and positive");
    if logical.n_edges == 0 {
        return 1.0;
    }
    prefactor * rms_coupling(logical) * mean_degree(logical).sqrt()
}

/// Rewrite a logical model onto hardware sites, holding chains at [`strength`].
///
/// The same call as [`crate::embed::apply`] with the rule swapped for the model's own statistics.
#[must_use]
pub fn apply(logical: &Graph, hardware: &Graph, e: &embed::Embedding) -> embed::Embedded {
    embed::apply_with(logical, hardware, e, strength(logical))
}

/// One embedded instance and the **exact** optimum of its logical model.
pub struct Case {
    /// The logical model, small enough that its optimum is proved rather than found.
    pub logical: Graph,
    /// Where it sits on the machine.
    pub embedding: embed::Embedding,
    /// Exact ground energy of `logical`. A sweep scored against its own best answer cannot tell
    /// "this row is bad" from "every row is bad", so this must not come from a sampler.
    pub optimum: f64,
}

/// A [`Case`] whose optimum is computed by exact variable elimination.
///
/// # Errors
///
/// [`TooWide`] when the logical model's treewidth exceeds [`Elimination::max_width`].
///
/// # Panics
///
/// If elimination returns no ground state, which `ground_state` cannot do without erroring first.
pub fn case(logical: Graph, embedding: embed::Embedding) -> Result<Case, TooWide> {
    let exact = Elimination::default().ground_state(&logical)?;
    let optimum = exact.ground_energy.expect("ground_state computed no ground energy");
    Ok(Case { logical, embedding, optimum })
}

/// What one chain strength did across a set of cases.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Row {
    /// The chain coupling under test.
    pub strength: f64,
    /// Mean share of variables whose chain disagreed with itself. Read this column first: a run
    /// with broken chains has not answered the question, whatever energy it reports.
    pub broken: f64,
    /// Mean logical energy above the exact optimum, after unembedding.
    pub gap: f64,
    /// Cases whose exact optimum was reached.
    pub found: usize,
}

/// Sweep chain strength over embedded cases: break rate and achieved energy per strength.
///
/// One anneal per case, the same geometric ladder (beta 0.05 to 6.0, 120 rungs of 60 sweeps) at
/// every strength, so the only thing varying down a column is the strength itself.
///
/// # Panics
///
/// If `cases` is empty — every column would be a division by zero.
#[must_use]
pub fn sweep(cases: &[Case], hardware: &Graph, strengths: &[f64], seed: u64) -> Vec<Row> {
    assert!(!cases.is_empty(), "a sweep over no cases measures nothing");
    let ladder: Vec<(f64, usize)> =
        tempering::geometric_ladder(0.05, 6.0, 120).into_iter().map(|b| (b, 60)).collect();
    let n = cases.len() as f64;
    strengths
        .iter()
        .map(|&s| {
            let (mut broken, mut gap, mut found) = (0.0, 0.0, 0usize);
            for (i, c) in cases.iter().enumerate() {
                let em = embed::apply_with(&c.logical, hardware, &c.embedding, s);
                let case_seed = seed ^ (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
                let (state, _) = tempering::anneal(&em.graph, &ladder, case_seed, None);
                let (values, broke) = embed::unembed(&c.embedding, &state);
                broken += broke.len() as f64 / c.logical.n as f64;
                let e = c.logical.energy(&values);
                gap += e - c.optimum;
                if (e - c.optimum).abs() < 1e-9 {
                    found += 1;
                }
            }
            Row { strength: s, broken: broken / n, gap: gap / n, found }
        })
        .collect()
}

/// The knee: the weakest strength swept at which no chain broke, if any did hold.
///
/// `rows` must be in ascending strength — as [`sweep`] returns them for an ascending input.
#[must_use]
pub fn knee(rows: &[Row]) -> Option<f64> {
    rows.iter().find(|r| r.broken == 0.0).map(|r| r.strength)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::GraphBuilder;
    use crate::rng::Pcg;
    use crate::{ising, samples};

    /// Closed form, computed by hand: a triangle with couplings 1, 2, 2 has `<J^2> = 3` and every
    /// degree 2, so the rule is `1.414 * sqrt(3) * sqrt(2) = 1.414 * sqrt(6)`.
    #[test]
    fn the_rule_is_its_closed_form() {
        let mut b = GraphBuilder::new(3);
        b.couple(0, 1, 1.0);
        b.couple(1, 2, 2.0);
        b.couple(0, 2, -2.0);
        let g = b.build();
        assert!((rms_coupling(&g) - 3f64.sqrt()).abs() < 1e-12, "{}", rms_coupling(&g));
        assert!((mean_degree(&g) - 2.0).abs() < 1e-12);
        assert!((strength(&g) - UTC_PREFACTOR * 6f64.sqrt()).abs() < 1e-12, "{}", strength(&g));
    }

    /// On a `k`-regular model with every `|J| = j`, the rule collapses to `prefactor * j * sqrt(k)`
    /// exactly. The 2D lattice is 4-regular under periodic boundaries; the ring is 2-regular.
    #[test]
    fn regular_models_reduce_to_j_root_degree() {
        for (g, k, j) in [
            (ising::lattice2d(6, 1.5), 4.0, 1.5),
            (ising::ring(9, -0.75, 0.0), 2.0, 0.75),
            (embed::topology::complete(7), 6.0, 1.0),
        ] {
            assert!((mean_degree(&g) - k).abs() < 1e-12, "degree {}", mean_degree(&g));
            assert!((rms_coupling(&g) - j).abs() < 1e-12, "rms {}", rms_coupling(&g));
            let want = UTC_PREFACTOR * j * k.sqrt();
            assert!((strength(&g) - want).abs() < 1e-12, "{} vs {want}", strength(&g));
        }
    }

    /// `mean_degree` is `2 E / n` and counts isolated variables. Hand-checked on a path of three
    /// plus a variable nobody couples to: degrees 1, 2, 1, 0, mean 1.
    #[test]
    fn mean_degree_counts_the_variables_nobody_coupled() {
        let mut b = GraphBuilder::new(4);
        b.couple(0, 1, 1.0);
        b.couple(1, 2, 1.0);
        let g = b.build();
        assert!((mean_degree(&g) - 1.0).abs() < 1e-12, "{}", mean_degree(&g));
        assert_eq!(g.n_edges, 2);
    }

    /// A model with no couplings has no torque to compensate; the reference implementation returns
    /// 1 and so does this, rather than a zero that holds nothing together.
    #[test]
    fn no_couplings_holds_at_one() {
        let mut b = GraphBuilder::new(3);
        b.bias(0, 5.0);
        let g = b.build();
        assert_eq!(rms_coupling(&g), 0.0);
        assert_eq!(strength(&g), 1.0);
        assert_eq!(strength_with(&g, 7.0), 1.0);
    }

    /// The prefactor scales the rule linearly and nothing else — the statistics are the model's.
    #[test]
    fn the_prefactor_only_scales() {
        let g = clique_pm1(11, 7);
        let one = strength_with(&g, 1.0);
        for p in [0.5, 1.414, 2.0, 10.0] {
            assert!((strength_with(&g, p) - p * one).abs() < 1e-12);
        }
    }

    #[test]
    #[should_panic(expected = "prefactor must be finite and positive")]
    fn a_zero_prefactor_is_refused() {
        let _ = strength_with(&embed::topology::complete(4), 0.0);
    }

    /// THE DIFFERENCE BETWEEN THE TWO RULES, as an exact ratio rather than a story. One edge in
    /// forty-five carries `J = 8`; the rest are +-1. `max|J|` is 8 and follows the outlier
    /// one-for-one, so `embed`'s default is 32. The rule here reads `sqrt(<J^2>) = sqrt(107/45)`,
    /// which the outlier moves by half.
    #[test]
    fn one_outlier_moves_max_eightfold_and_the_rms_by_half() {
        let plain = clique_pm1(10, 3);
        let spiked = spiked_clique(10, 3, 8.0, 1);
        assert!((embed::worst_coefficient(&plain) - 1.0).abs() < 1e-12);
        assert!((embed::worst_coefficient(&spiked) - 8.0).abs() < 1e-12);
        // 44 edges at J^2 = 1, one at 64, over 45 edges.
        let want = (108.0f64 / 45.0).sqrt();
        assert!((rms_coupling(&spiked) - want).abs() < 1e-12, "{}", rms_coupling(&spiked));
        assert!((rms_coupling(&plain) - 1.0).abs() < 1e-12);
        let ratio = strength(&spiked) / strength(&plain);
        assert!((ratio - want).abs() < 1e-12 && ratio < 1.6, "rms rule moved {ratio}x");
    }

    // ---- the sweep: what the rule costs ---------------------------------------------------

    fn clique_pm1(n: usize, seed: u64) -> Graph {
        let mut rng = Pcg::new(seed, 0x7043_0001);
        let mut b = GraphBuilder::new(n);
        for i in 0..n {
            for j in (i + 1)..n {
                b.couple(i, j, if rng.f64() < 0.5 { 1.0 } else { -1.0 });
            }
        }
        b.build()
    }

    /// A +-1 clique with `spikes` edges replaced by +-`big`: the family where the two rules part.
    fn spiked_clique(n: usize, seed: u64, big: f64, spikes: usize) -> Graph {
        let mut rng = Pcg::new(seed, 0x7043_0002);
        let mut b = GraphBuilder::new(n);
        let edges = n * (n - 1) / 2;
        let mut k = 0;
        for i in 0..n {
            for j in (i + 1)..n {
                let sign = if rng.f64() < 0.5 { 1.0 } else { -1.0 };
                // Spread the spikes across distinct variables rather than onto one.
                let mag = if k % (edges / spikes).max(1) == 0 && k / (edges / spikes).max(1) < spikes
                {
                    big
                } else {
                    1.0
                };
                b.couple(i, j, sign * mag);
                k += 1;
            }
        }
        b.build()
    }

    /// A +-1 clique with +-5 fields: the scale lives where no torque does.
    fn biased_clique(n: usize, seed: u64) -> Graph {
        let mut rng = Pcg::new(seed, 0x7043_0003);
        let mut b = GraphBuilder::new(n);
        for i in 0..n {
            for j in (i + 1)..n {
                b.couple(i, j, if rng.f64() < 0.5 { 1.0 } else { -1.0 });
            }
        }
        for i in 0..n {
            b.bias(i, if rng.f64() < 0.5 { 5.0 } else { -5.0 });
        }
        b.build()
    }

    /// Build cases on a Chimera machine, skipping seeds the placer fails to embed.
    fn cases(make: impl Fn(u64) -> Graph, hardware: &Graph, seeds: u64) -> Vec<Case> {
        let mut out = Vec::new();
        for seed in 0..seeds {
            let g = make(seed);
            let Some(emb) = embed::embed(&g, hardware, seed) else { continue };
            emb.verify(&g, hardware).expect("the placer returned a broken embedding");
            out.push(case(g, emb).expect("a small clique is narrow enough to eliminate"));
        }
        out
    }

    fn show(name: &str, rows: &[Row], marks: &[(&str, f64)]) {
        println!("\n{name}");
        for r in rows {
            let mark = marks
                .iter()
                .filter(|(_, s)| (s - r.strength).abs() < 1e-9)
                .map(|(m, _)| *m)
                .collect::<Vec<_>>()
                .join(" ");
            println!(
                "  strength {:>7.3}  broken {:>6.2}%  gap {:>7.3}  found {:>3}   {mark}",
                r.strength,
                r.broken * 100.0,
                r.gap,
                r.found
            );
        }
    }

    /// The optimum an exact elimination reports is the optimum exhaustive enumeration reports.
    /// Cheap, and it is the number every row of every sweep is scored against.
    #[test]
    fn the_oracle_agrees_with_exhaustive_enumeration() {
        for seed in 0..4 {
            let g = clique_pm1(10, seed);
            let elim =
                Elimination::default().ground_state(&g).expect("K10 is narrow").ground_energy;
            let set = samples::enumerate(&g, 1.0).expect("2^10 states");
            let brute = set.energies().iter().copied().fold(f64::INFINITY, f64::min);
            assert!((elim.expect("min-sum ran") - brute).abs() < 1e-9, "seed {seed}");
        }
    }

    /// UNIFORM FAMILY. +-1 cliques of ten on Chimera. The rule must land at or above the knee —
    /// the weakest strength whose chains all hold — and must not be so far above it that the
    /// answer degrades. It is NOT expected to beat `embed`'s tuned constant here; on this family
    /// the two are within 20% of each other by construction.
    #[test]
    fn on_uniform_cliques_the_rule_sits_at_the_knee() {
        let hw = ising::chimera(4, 4, 4, 1.0);
        let cs = cases(|s| clique_pm1(10, s), &hw, 8);
        assert!(cs.len() >= 6, "only {} of 8 embedded", cs.len());
        let utc = strength(&cs[0].logical);
        let maxj = embed::DEFAULT_CHAIN_MULTIPLE * embed::worst_coefficient(&cs[0].logical);
        let mut strengths = vec![0.5, 1.0, 2.0, 3.0, maxj, utc, 6.0, 9.0, 16.0, 32.0];
        strengths.sort_by(f64::total_cmp);
        let rows = sweep(&cs, &hw, &strengths, 0xC4A1);
        show("uniform +-1 K10 on chimera(4,4,4)", &rows, &[("<- UTC", utc), ("<- 4 max|J|", maxj)]);

        let knee = knee(&rows).expect("some strength must hold every chain");
        let at_utc = rows.iter().find(|r| r.strength == utc).expect("UTC row");
        assert_eq!(at_utc.broken, 0.0, "the rule must not break chains: {at_utc:?}");
        assert!(utc >= knee && utc <= 2.0 * knee, "UTC {utc} against knee {knee}");
        // Weak chains break and strong ones do not: the failure is one-sided, which is why the
        // broken column alone cannot pick a strength.
        let weakest = rows[0];
        let strongest = rows[rows.len() - 1];
        assert!(weakest.broken > 0.05, "0.5x should break chains: {weakest:?}");
        assert_eq!(strongest.broken, 0.0, "32x should break none: {strongest:?}");
        // And over-strong is worse on the only column that matters.
        assert!(strongest.gap > at_utc.gap, "{strongest:?} vs {at_utc:?}");
    }

    /// SPIKED FAMILY, and the point of the module. One coupling of eight per clique. `max|J|`
    /// follows the outlier to 32; the rule reads the RMS and lands near the knee. The claim under
    /// test is the ordering of their gaps, not a target number.
    #[test]
    fn on_spiked_cliques_the_rms_rule_beats_the_max_rule() {
        let hw = ising::chimera(4, 4, 4, 1.0);
        let cs = cases(|s| spiked_clique(10, s, 8.0, 1), &hw, 8);
        assert!(cs.len() >= 6, "only {} of 8 embedded", cs.len());
        let utc = strength(&cs[0].logical);
        let maxj = embed::DEFAULT_CHAIN_MULTIPLE * embed::worst_coefficient(&cs[0].logical);
        let mut strengths = vec![2.0, 4.0, 6.0, utc, 12.0, 16.0, 24.0, maxj, 48.0, 64.0];
        strengths.sort_by(f64::total_cmp);
        let rows = sweep(&cs, &hw, &strengths, 0xC4A2);
        show("spiked K10 (one J = 8) on chimera(4,4,4)", &rows, &[
            ("<- UTC", utc),
            ("<- 4 max|J|", maxj),
        ]);

        let knee = knee(&rows).expect("some strength must hold every chain");
        let at_utc = rows.iter().find(|r| r.strength == utc).expect("UTC row");
        let at_max = rows.iter().find(|r| r.strength == maxj).expect("max|J| row");
        assert!(maxj > 3.0 * knee, "the max rule should overshoot: {maxj} vs knee {knee}");
        assert!(utc <= 2.0 * knee, "the rms rule should not: {utc} vs knee {knee}");
        assert_eq!(at_max.broken, 0.0, "over-strong chains never break — that is the trap");
        assert!(at_utc.gap < at_max.gap, "rms {at_utc:?} should beat max {at_max:?}");
        assert!(at_utc.found >= at_max.found, "rms {at_utc:?} vs max {at_max:?}");
    }

    /// FIELD-HEAVY FAMILY. Same +-1 clique, plus +-5 fields. A field pulls every site of a chain
    /// the same way, so it exerts no torque and this rule ignores it — while
    /// `embed::worst_coefficient` reads it and quadruples the max rule to 20, 4.7x the knee.
    #[test]
    fn fields_move_the_max_rule_and_not_this_one() {
        let hw = ising::chimera(4, 4, 4, 1.0);
        let cs = cases(|s| biased_clique(10, s), &hw, 8);
        assert!(cs.len() >= 6, "only {} of 8 embedded", cs.len());
        let utc = strength(&cs[0].logical);
        let maxj = embed::DEFAULT_CHAIN_MULTIPLE * embed::worst_coefficient(&cs[0].logical);
        assert!((rms_coupling(&cs[0].logical) - 1.0).abs() < 1e-12, "couplings are still +-1");
        assert!((maxj - 20.0).abs() < 1e-12, "the max rule read the field: {maxj}");

        let mut strengths = vec![1.0, 2.0, 3.0, utc, 6.0, 8.0, 12.0, maxj, 32.0];
        strengths.sort_by(f64::total_cmp);
        let rows = sweep(&cs, &hw, &strengths, 0xC4A3);
        show("field-heavy K10 on chimera(4,4,4)", &rows, &[
            ("<- UTC", utc),
            ("<- 4 max|coef|", maxj),
        ]);

        let knee = knee(&rows).expect("some strength must hold every chain");
        let at_utc = rows.iter().find(|r| r.strength == utc).expect("UTC row");
        assert_eq!(at_utc.broken, 0.0, "the rule must not break chains: {at_utc:?}");
        assert!((utc - knee).abs() < 1e-12, "UTC {utc} should BE the knee, which is {knee}");
        assert!(maxj > 3.0 * knee, "the max rule should overshoot: {maxj} vs {knee}");
        let strongest = rows[rows.len() - 1];
        assert_eq!(strongest.broken, 0.0, "over-strong chains never break — that is the trap");
        assert!(strongest.gap > at_utc.gap, "{strongest:?} vs {at_utc:?}");
    }
}
