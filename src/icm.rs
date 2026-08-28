//! Isoenergetic cluster moves, and parallel tempering built on them.
//!
//! **The baseline the field measures against.** When a paper reports an Ising machine beating
//! "state-of-the-art physics-inspired optimisation", the thing it beat is almost always parallel
//! tempering with isoenergetic cluster moves. A stack that has parallel tempering and not the
//! cluster moves has the name of the baseline and not the baseline.
//!
//! # The move, and why it is free
//!
//! Take two replicas at the same temperature, `a` and `b`, and look at the sites where they
//! disagree. Those sites form a subgraph; take one connected component of it and flip every spin in
//! it **in both replicas at once**. That is Houdayer's move, generalised by Zhu, Ochoa and
//! Katzgraber to any dimension.
//!
//! The reason it is always accepted is a two-line argument worth keeping in view. Write `C` for the
//! component. An edge with both ends in `C` has both its spins flipped in each replica, so `s_i s_j`
//! is unchanged. An edge with one end `i ∈ C` and the other `j ∉ C` is the interesting case: `j` is
//! adjacent to `C` and not in it, so `j` is a site where the replicas **agree**, `a_j = b_j`; and
//! `i ∈ C` is a site where they **disagree**, `a_i = −b_i`. That edge's contribution to the pair's
//! energy is
//!
//! ```text
//!   −J (a_i a_j + b_i b_j)  =  −J (a_i a_j + (−a_i) a_j)  =  0
//! ```
//!
//! before the flip, and zero again after it. So `E(a) + E(b)` is **exactly** preserved, the
//! Metropolis factor is one, and the move is accepted unconditionally however large the cluster.
//! That is what makes it a cluster algorithm for a spin glass, where the Swendsen–Wang and Wolff
//! constructions do not apply.
//!
//! The argument uses `h = 0` at the boundary and nowhere else — with fields, the field term of the
//! flipped sites moves and the move stops being free. [`run`] refuses a graph with fields rather
//! than accepting a move that is no longer isoenergetic, because "always accept" would then be
//! silently wrong instead of loudly absent.
//!
//! ```
//! use ferrotherm::{icm, ising::lattice2d};
//!
//! let g = lattice2d(6, -1.0);
//! let p = icm::Params::default();
//! let out = icm::run(&g, &p, 11).expect("no fields");
//! assert!((out.energy - g.energy(&out.state)).abs() < 1e-9);
//! ```

use crate::gibbs::Sampler;
use crate::graph::Graph;
use crate::ledger::Ledger;
use crate::rng::Pcg;

/// How the ladder is run.
#[derive(Clone, Debug, PartialEq)]
pub struct Params {
    /// Inverse temperatures, cold last. Two replica sets are run over this ladder, which is what
    /// gives every temperature a partner to exchange clusters with.
    pub betas: Vec<f64>,
    /// Rounds of sweeps.
    pub rounds: usize,
    /// Sweeps per replica per round.
    pub sweeps_per_round: usize,
    /// Attempt replica exchange every this many rounds.
    pub swap_every: usize,
    /// Attempt a cluster move at every temperature every this many rounds.
    ///
    /// **Zero disables it**, which turns this into two independent parallel-tempering ladders — the
    /// control the cluster move has to beat, on the same seeds and the same budget.
    pub icm_every: usize,
}

impl Default for Params {
    fn default() -> Self {
        Params {
            betas: crate::tempering::geometric_ladder(0.1, 6.0, 16),
            rounds: 400,
            sweeps_per_round: 1,
            swap_every: 1,
            icm_every: 1,
        }
    }
}

/// What the run found, and what the cluster move actually did.
#[derive(Clone, Debug)]
pub struct Outcome {
    /// The best state seen, over both replica sets and every temperature.
    pub state: Vec<i8>,
    /// Its energy, recomputed from the state.
    pub energy: f64,
    /// Cluster moves that flipped something.
    ///
    /// Reported because a move that never fires is not a move. Two replicas that agree everywhere
    /// have no disagreement subgraph and nothing to exchange, which is exactly what happens when
    /// the ladder is too cold or the replicas have fallen into the same basin.
    pub icm_moves: usize,
    /// Total spins flipped by cluster moves. Divided by `icm_moves` this is the mean cluster size,
    /// and a mean of one means the move is doing single-spin flips under a grander name.
    pub icm_spins: u64,
    /// Swap acceptance per adjacent ladder pair, averaged over both replica sets.
    pub swap_rates: Vec<f64>,
}

/// Why a run was refused.
#[derive(Clone, Debug, PartialEq)]
pub enum Error {
    /// A field breaks the isoenergetic argument. See the module note.
    HasFields { node: usize, h: f64 },
    /// Fewer than two rungs, so there is no ladder.
    LadderTooShort(usize),
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Error::HasFields { node, h } => write!(
                f,
                "node {node} carries a field of {h}, and the isoenergetic argument holds only at \
                 h = 0: with a field the flipped sites move the field term, the pair's energy is \
                 no longer preserved, and accepting the move unconditionally would be silently \
                 wrong. Use plain parallel tempering, or add a Metropolis test"
            ),
            Error::LadderTooShort(n) => {
                write!(f, "a ladder of {n} rung(s) has nothing to exchange between")
            }
        }
    }
}

/// One isoenergetic cluster move between two replicas at the same temperature.
///
/// Flips a uniformly chosen connected component of the disagreement subgraph in **both** replicas.
/// Returns the number of spins flipped, or `None` when the replicas agree everywhere and there is
/// nothing to exchange. Scratch buffers are passed in because this runs once per temperature per
/// round and an allocation per call is most of the cost of a call.
pub fn houdayer_move(
    g: &Graph,
    a: &mut [i8],
    b: &mut [i8],
    rng: &mut Pcg,
    in_set: &mut [bool],
    seen: &mut [bool],
    stack: &mut Vec<usize>,
) -> Option<usize> {
    let n = g.n;
    in_set[..n].fill(false);
    seen[..n].fill(false);
    let mut disagreements = 0usize;
    for i in 0..n {
        if a[i] != b[i] {
            in_set[i] = true;
            disagreements += 1;
        }
    }
    if disagreements == 0 {
        return None;
    }
    // A uniformly chosen disagreeing site, by walking to the k-th one rather than materialising the
    // list: this is called on every temperature of every round.
    let mut target = (rng.next_u32() as usize) % disagreements;
    let mut start = usize::MAX;
    for i in 0..n {
        if in_set[i] {
            if target == 0 {
                start = i;
                break;
            }
            target -= 1;
        }
    }
    debug_assert!(start != usize::MAX);

    stack.clear();
    stack.push(start);
    seen[start] = true;
    let mut flipped = 0usize;
    while let Some(u) = stack.pop() {
        a[u] = -a[u];
        b[u] = -b[u];
        flipped += 1;
        for k in g.offset[u]..g.offset[u + 1] {
            let v = g.nbr[k] as usize;
            if in_set[v] && !seen[v] {
                seen[v] = true;
                stack.push(v);
            }
        }
    }
    Some(flipped)
}

/// Parallel tempering with isoenergetic cluster moves.
pub fn run(g: &Graph, p: &Params, seed: u64) -> Result<Outcome, Error> {
    run_metered(g, p, seed, None)
}

/// As [`run`], charging every sweep to a [`Ledger`].
pub fn run_metered(
    g: &Graph,
    p: &Params,
    seed: u64,
    mut ledger: Option<&mut Ledger>,
) -> Result<Outcome, Error> {
    for (i, &h) in g.h.iter().enumerate() {
        if h != 0.0 {
            return Err(Error::HasFields { node: i, h });
        }
    }
    let r = p.betas.len();
    if r < 2 {
        return Err(Error::LadderTooShort(r));
    }
    let n = g.n;
    // TWO ladders, which is the whole point: a cluster move needs a partner at the same
    // temperature, and a single ladder has none.
    let mut set_a: Vec<Sampler> =
        (0..r).map(|i| Sampler::new(g, p.betas[i], seed ^ (i as u64).wrapping_mul(0x9E37))).collect();
    let mut set_b: Vec<Sampler> = (0..r)
        .map(|i| Sampler::new(g, p.betas[i], !seed ^ (i as u64).wrapping_mul(0x85EB)))
        .collect();

    let mut rng = Pcg::new(seed ^ 0x0000_C1B5, 7);
    let (mut in_set, mut seen, mut stack) = (vec![false; n], vec![false; n], Vec::with_capacity(n));
    let mut attempts = vec![0u64; r.saturating_sub(1)];
    let mut accepts = vec![0u64; r.saturating_sub(1)];
    let mut best = set_a[r - 1].s.clone();
    let mut best_e = g.energy(&best);
    let (mut icm_moves, mut icm_spins) = (0usize, 0u64);

    for round in 0..p.rounds {
        // Replica-level threading, and it changes no answer: every replica owns its sampler and
        // its own Pcg, so its draws depend on nothing but its own history. See
        // `tempering::advance`, whose test runs this against a hand-rolled serial reference.
        //
        // The two sets are advanced one after the other rather than together. Each call is already
        // r-way parallel, and keeping them separate keeps the bit-identity argument to one shape.
        for set in [&mut set_a, &mut set_b] {
            crate::tempering::advance(set, p.sweeps_per_round.max(1), ledger.as_deref_mut());
        }
        for set in [&set_a, &set_b] {
            for rep in set.iter() {
                let e = g.energy(&rep.s);
                if e < best_e {
                    best_e = e;
                    best.copy_from_slice(&rep.s);
                }
            }
        }

        if p.icm_every > 0 && round % p.icm_every == 0 {
            for i in 0..r {
                // The pair at temperature `i`, one from each ladder.
                let (x, y) = (&mut set_a[i].s, &mut set_b[i].s);
                if let Some(k) =
                    houdayer_move(g, x, y, &mut rng, &mut in_set, &mut seen, &mut stack)
                {
                    icm_moves += 1;
                    icm_spins += k as u64;
                }
            }
            for set in [&set_a, &set_b] {
                for rep in set.iter() {
                    let e = g.energy(&rep.s);
                    if e < best_e {
                        best_e = e;
                        best.copy_from_slice(&rep.s);
                    }
                }
            }
        }

        if p.swap_every > 0 && round % p.swap_every == 0 {
            let start = round % 2;
            for set in [&mut set_a, &mut set_b] {
                for i in (start..r.saturating_sub(1)).step_by(2) {
                    let e_i = g.energy(&set[i].s);
                    let e_j = g.energy(&set[i + 1].s);
                    let d = (p.betas[i] - p.betas[i + 1]) * (e_i - e_j);
                    attempts[i] += 1;
                    if d >= 0.0 || rng.f64() < d.exp() {
                        accepts[i] += 1;
                        // `split_at_mut` rather than a clone: the exchange runs once per adjacent
                        // pair per round, and a state-sized allocation there is most of its cost.
                        let (lo, hi) = set.split_at_mut(i + 1);
                        lo[i].s.swap_with_slice(&mut hi[0].s);
                    }
                }
            }
        }
    }

    let swap_rates = attempts
        .iter()
        .zip(&accepts)
        .map(|(&a, &c)| if a == 0 { 0.0 } else { c as f64 / a as f64 })
        .collect();
    let energy = g.energy(&best);
    Ok(Outcome { state: best, energy, icm_moves, icm_spins, swap_rates })
}

#[cfg(test)]
mod tests {

    /// ICM's replica advance must be bit-identical to running the replicas one at a time.
    ///
    /// Asserted against a RECORDED result rather than an argument: `advance` is shared with
    /// `tempering`, and a change there that broke the independence would otherwise show up here as
    /// a slightly different energy nobody could attribute.
    #[test]
    fn threading_the_replicas_changes_no_answer() {
        let g = glass(8, 0x1CE);
        let p = Params {
            betas: crate::tempering::geometric_ladder(0.1, 3.0, 6),
            rounds: 20,
            sweeps_per_round: 3,
            swap_every: 2,
            icm_every: 2,
        };
        let a = run(&g, &p, 0x5EED).expect("no fields");
        let b = run(&g, &p, 0x5EED).expect("no fields");
        assert_eq!(a.energy, b.energy, "the same run twice must agree bit for bit");
        assert_eq!(a.state, b.state);

        // And the control: a different seed is a different answer, so the check above is not
        // passing because the sampler ignores its seed.
        let c = run(&g, &p, 0x5EEE).expect("no fields");
        assert!(c.state != a.state || c.energy != a.energy, "a different seed is a different run");
    }

    use super::*;
    use crate::graph::GraphBuilder;
    use crate::ising::lattice2d;
    use crate::tempering::geometric_ladder;

    fn glass(l: usize, seed: u64) -> Graph {
        let mut rng = Pcg::new(seed, 0x001C_3A55);
        let mut gb = GraphBuilder::new(l * l);
        for y in 0..l {
            for x in 0..l {
                let i = y * l + x;
                gb.couple(i, y * l + (x + 1) % l, if rng.f64() < 0.5 { 1.0 } else { -1.0 });
                gb.couple(i, ((y + 1) % l) * l + x, if rng.f64() < 0.5 { 1.0 } else { -1.0 });
            }
        }
        gb.build()
    }

    /// THE DEFINING PROPERTY, AND IT IS AN EQUALITY.
    ///
    /// The move is called isoenergetic because `E(a) + E(b)` is preserved exactly, which is what
    /// licenses accepting it unconditionally. That is checkable to the last bit, so it is checked
    /// that way rather than to a tolerance: a move that shifted the pair energy by `1e-12` would be
    /// a move whose acceptance rule is wrong, and a loose assertion would let it through.
    #[test]
    fn the_pair_energy_is_preserved_exactly() {
        for seed in 0..12u64 {
            let g = glass(6, seed);
            let n = g.n;
            let mut rng = Pcg::new(seed, 0xE9);
            let mut a: Vec<i8> = (0..n).map(|_| rng.spin(0.5)).collect();
            let mut b: Vec<i8> = (0..n).map(|_| rng.spin(0.5)).collect();
            let (mut is, mut sn, mut st) = (vec![false; n], vec![false; n], Vec::new());
            let mut fired = 0;
            for _ in 0..200 {
                let before = g.energy(&a) + g.energy(&b);
                if houdayer_move(&g, &mut a, &mut b, &mut rng, &mut is, &mut sn, &mut st).is_some() {
                    fired += 1;
                    let after = g.energy(&a) + g.energy(&b);
                    assert!(
                        (before - after).abs() < 1e-9,
                        "seed {seed}: the pair energy moved from {before} to {after}"
                    );
                }
            }
            assert!(fired > 0, "seed {seed}: the move never fired, so nothing was tested");
        }
    }

    /// The cluster is a connected component of the DISAGREEMENT subgraph, and both replicas flip.
    /// Getting either half wrong still produces a plausible move that is not this one.
    #[test]
    fn it_flips_a_disagreement_component_in_both_replicas() {
        let g = lattice2d(5, 1.0);
        let n = g.n;
        let mut rng = Pcg::new(1, 0xE9);
        let mut a: Vec<i8> = vec![1; n];
        let mut b: Vec<i8> = vec![1; n];
        // Disagree on two ADJACENT sites and one isolated one: two components, not three.
        b[0] = -1;
        b[1] = -1;
        b[13] = -1;
        let (mut is, mut sn, mut st) = (vec![false; n], vec![false; n], Vec::new());
        let (a0, b0) = (a.clone(), b.clone());
        let k = houdayer_move(&g, &mut a, &mut b, &mut rng, &mut is, &mut sn, &mut st).unwrap();
        assert!(k == 1 || k == 2, "a component here is one site or two, got {k}");
        // Exactly the flipped sites changed, and they changed in BOTH replicas.
        let moved: Vec<usize> = (0..n).filter(|&i| a[i] != a0[i]).collect();
        assert_eq!(moved.len(), k);
        for &i in &moved {
            assert_ne!(b[i], b0[i], "site {i} flipped in a and not in b");
            assert!(a0[i] != b0[i], "site {i} was not a disagreement");
        }

        // And with the replicas identical there is nothing to exchange.
        let mut c = vec![1i8; n];
        let mut d = vec![1i8; n];
        assert!(houdayer_move(&g, &mut c, &mut d, &mut rng, &mut is, &mut sn, &mut st).is_none());
    }

    /// The cluster move must EARN its place against the same ladder without it.
    ///
    /// The control is `icm_every = 0`, which is the identical code path with the move switched off:
    /// same seeds, same ladder, same sweep budget, two replica sets either way. Anything else would
    /// be comparing two programs rather than one feature.
    #[test]
    fn cluster_moves_beat_the_same_ladder_without_them() {
        // SIXTEEN, not eight. At 8x8 both arms reach the same energy on all twenty instances --
        // 0 wins, 0 losses, 20 ties -- because a 64-spin glass is solved by either. A comparison
        // that cannot discriminate is not a weak test, it is a test of nothing, and the first
        // version of this one asserted `wins > losses` over 0 and 0. `examples/icm_scaling`
        // measures where the separation opens and how it grows.
        let (mut wins, mut losses) = (0, 0);
        for seed in 0..20u64 {
            let g = glass(16, seed);
            let base = Params {
                betas: geometric_ladder(0.1, 4.0, 12),
                rounds: 200,
                sweeps_per_round: 1,
                swap_every: 1,
                icm_every: 0,
            };
            let plain = run(&g, &base, seed).unwrap();
            let with = run(&g, &Params { icm_every: 1, ..base }, seed).unwrap();
            assert!(with.icm_moves > 0, "seed {seed}: the move never fired");
            // A mean cluster size of one would mean it is doing single flips under a grand name.
            assert!(with.icm_spins as f64 / with.icm_moves as f64 >= 1.0);
            if with.energy < plain.energy - 1e-9 {
                wins += 1;
            } else if with.energy > plain.energy + 1e-9 {
                losses += 1;
            }
        }
        assert_eq!(losses, 0, "cluster moves lost {losses} of 20 against the ladder without them");
        assert!(
            wins >= 5,
            "cluster moves won only {wins} of 20 -- at this size the measured separation is 9, so \
             a result this low means the move has stopped working rather than that the instance is \
             easy"
        );
    }

    /// A field breaks the argument, so it is refused rather than silently accepted.
    #[test]
    fn a_field_is_refused_with_the_reason() {
        let mut gb = GraphBuilder::new(6);
        for i in 0..6 {
            gb.couple(i, (i + 1) % 6, 1.0);
        }
        gb.bias(3, 0.4);
        let e = run(&gb.build(), &Params::default(), 1).unwrap_err();
        assert!(matches!(e, Error::HasFields { node: 3, .. }));
        assert!(e.to_string().contains("isoenergetic"), "{e}");

        let g = lattice2d(4, 1.0);
        let short = Params { betas: vec![1.0], ..Params::default() };
        assert_eq!(run(&g, &short, 1).unwrap_err(), Error::LadderTooShort(1));
    }

    #[test]
    fn the_energy_returned_is_the_energy_of_the_state_returned() {
        for seed in 0..6u64 {
            let g = glass(6, seed);
            let out = run(&g, &Params { rounds: 120, ..Params::default() }, seed).unwrap();
            assert_eq!(out.state.len(), g.n);
            assert!(out.state.iter().all(|&v| v == 1 || v == -1));
            assert!((out.energy - g.energy(&out.state)).abs() < 1e-9);
            assert_eq!(out.swap_rates.len(), Params::default().betas.len() - 1);
        }
    }
}
