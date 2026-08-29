//! Hamze–de Freitas–Selby: solve a low-treewidth *block* exactly, over and over.
//!
//! Every other local search in this crate flips **one** spin and asks whether that helped:
//! [`crate::tabu`], [`crate::bls`], [`crate::gibbs`]. HFS flips a whole subgraph at once, and it
//! does not ask — it takes the **exact best** assignment of that subgraph given everything outside
//! it held fixed. A single-flip method escapes a barrier only by paying for it; a block move steps
//! over any barrier that lives entirely inside the block.
//!
//! Both halves this needs were already here and had never been put together.
//! [`crate::exact::Elimination`] solves a graph exactly in `2^w` where `w` is the induced width, and
//! [`crate::branch`] already computes the residual field a fixed neighbourhood exerts. HFS is the
//! observation that those two compose: condition on the complement, and what is left is small
//! enough to solve outright.
//!
//! # Why this algorithm and not another
//!
//! It is the reason the field stopped believing the first generation of quantum-annealer speedup
//! claims. Selby's implementation solved the same Chimera-structured instances on one core faster
//! than the hardware they were reported against, and the structure it exploits — that a Chimera
//! graph decomposes into low-treewidth blocks — was exactly the structure the benchmark instances
//! had. A stack that means to make honest comparisons has to be able to run the algorithm the
//! comparisons turned on.
//!
//! # The conditioning, written out
//!
//! With `E(s) = −Σ h_i s_i − Σ J_ij s_i s_j` and a block `B` whose complement is held fixed,
//!
//! ```text
//!   E(s) = − Σ_{i∈B} ( h_i + Σ_{j∉B} J_ij s_j ) s_i  −  Σ_{i,j∈B} J_ij s_i s_j  +  const
//! ```
//!
//! so the residual problem is an ordinary Ising model on `|B|` spins whose fields have absorbed
//! what the frozen neighbours say. That bracket is the same `λ` [`crate::branch`] carries down its
//! tree, and it is the only place a sign error could hide — so [`step`] is tested against brute
//! force over the block rather than against itself.
//!
//! # Blocks are grown as TREES, and the reason is not speed
//!
//! [`tree_block`] grows an induced subtree: a node joins only if exactly one of its neighbours is
//! already in the block. That makes the induced subgraph acyclic **by construction**, so its width
//! is 1 and no width has to be computed, searched for, or hoped for. The alternative — grow an
//! arbitrary subset and measure — is available as [`grown_block`], which measures the width and
//! **refuses** rather than approximating when it is too large.
//!
//! Selby's Chimera implementation uses two interleaved trees for the same reason. A tree is not a
//! compromise here; it is the largest block whose exact solution is free.
//!
//! # What it does not do
//!
//! **It is a descent, not a sampler.** [`step`] takes the minimum, so the energy never increases
//! and the method has no temperature and no detailed balance. Restarts and block variety are what
//! keep it moving; [`run`] does both. For a *sampling* version — the exact conditional at finite
//! beta rather than its minimum — the same conditioning would call
//! [`crate::exact::Elimination::marginals`] instead, which is a different algorithm and is not this
//! one.

use crate::exact::{Elimination, TooWide};
use crate::graph::{Graph, GraphBuilder};
use crate::rng::Pcg;

/// Solve `block` exactly with everything outside it held fixed, writing the answer into `s`.
///
/// Returns the change in total energy, which is `<= 0` up to floating point: the current assignment
/// of the block is itself a candidate, so the exact minimum can never be worse than it.
///
/// Refuses, rather than approximating, when the block's induced subgraph is wider than
/// `el.max_width` — the same [`TooWide`] the exact solver returns, from the same order.
///
/// Duplicate entries in `block` are ignored rather than refused: a block is a SET, and a grower that
/// offers the same node twice has said nothing new.
pub fn step(g: &Graph, s: &mut [i8], block: &[usize], el: &Elimination) -> Result<f64, TooWide> {
    // `map[i]` is the residual index of `i`, or usize::MAX when `i` is outside the block.
    let mut map = vec![usize::MAX; g.n];
    let mut vars: Vec<usize> = Vec::with_capacity(block.len());
    for &i in block {
        if i < g.n && map[i] == usize::MAX {
            map[i] = vars.len();
            vars.push(i);
        }
    }
    if vars.is_empty() {
        return Ok(0.0);
    }

    let mut gb = GraphBuilder::new(vars.len());
    for (a, &i) in vars.iter().enumerate() {
        // The bracket from the module doc: the node's own field plus what every FROZEN neighbour
        // says to it. A neighbour inside the block says nothing here -- its say is the coupling.
        let mut field = g.h[i];
        for k in g.offset[i]..g.offset[i + 1] {
            let j = g.nbr[k] as usize;
            if map[j] == usize::MAX {
                field += g.w[k] * s[j] as f64;
            } else if j > i {
                // Each in-block edge once: the CSR holds both directions.
                gb.couple(a, map[j], g.w[k]);
            }
        }
        if field != 0.0 {
            gb.bias(a, field);
        }
    }

    let residual = gb.build();
    let solved = el.ground_state(&residual)?;
    let best = solved.ground_state.expect("min-sum was run");

    let before = g.energy(s);
    for (a, &i) in vars.iter().enumerate() {
        s[i] = best[a];
    }
    Ok(g.energy(s) - before)
}

/// Grow an induced SUBTREE from `seed`, up to `target` nodes.
///
/// A node joins only when exactly one of its neighbours is already in the block, so the induced
/// subgraph is acyclic by construction and its width is 1. Nothing here computes a width, searches
/// for an order, or can be surprised by one.
///
/// The frontier is shuffled by `rng`, so successive calls cover different parts of the graph — which
/// is the whole mechanism by which a descent that never increases energy still keeps moving.
pub fn tree_block(g: &Graph, seed: usize, target: usize, rng: &mut Pcg) -> Vec<usize> {
    if g.n == 0 || target == 0 {
        return Vec::new();
    }
    let seed = seed % g.n;
    let mut inside = vec![false; g.n];
    let mut block = vec![seed];
    inside[seed] = true;

    // Candidates are (node, how many of its neighbours are already inside). A node is admissible at
    // exactly one; at two or more it would close a cycle and the block would stop being a tree.
    let mut frontier: Vec<usize> = Vec::new();
    let push_nbrs = |i: usize, frontier: &mut Vec<usize>| {
        for k in g.offset[i]..g.offset[i + 1] {
            frontier.push(g.nbr[k] as usize);
        }
    };
    push_nbrs(seed, &mut frontier);

    while block.len() < target && !frontier.is_empty() {
        // Pick at random rather than in order: a deterministic frontier walks the same shape of
        // block out of every seed, and block VARIETY is what makes the descent escape anything.
        let at = (rng.f64() * frontier.len() as f64) as usize % frontier.len();
        let cand = frontier.swap_remove(at);
        if inside[cand] {
            continue;
        }
        let touching = (g.offset[cand]..g.offset[cand + 1])
            .filter(|&k| inside[g.nbr[k] as usize])
            .count();
        if touching != 1 {
            continue;
        }
        inside[cand] = true;
        block.push(cand);
        push_nbrs(cand, &mut frontier);
    }
    block
}

/// Grow a block without the tree restriction, and MEASURE its width.
///
/// Returns `None` when the block that grew is wider than `max_width`. A wider block is a strictly
/// stronger move — it can step over barriers a tree cannot — and the cost is `2^w`, so this is the
/// knob that trades the two against each other with the width measured rather than assumed.
pub fn grown_block(
    g: &Graph,
    seed: usize,
    target: usize,
    max_width: usize,
    rng: &mut Pcg,
) -> Option<Vec<usize>> {
    if g.n == 0 || target == 0 {
        return None;
    }
    let seed = seed % g.n;
    let mut inside = vec![false; g.n];
    let mut block = vec![seed];
    inside[seed] = true;
    let mut frontier: Vec<usize> = (g.offset[seed]..g.offset[seed + 1])
        .map(|k| g.nbr[k] as usize)
        .collect();

    while block.len() < target && !frontier.is_empty() {
        let at = (rng.f64() * frontier.len() as f64) as usize % frontier.len();
        let cand = frontier.swap_remove(at);
        if inside[cand] {
            continue;
        }
        inside[cand] = true;
        block.push(cand);
        for k in g.offset[cand]..g.offset[cand + 1] {
            frontier.push(g.nbr[k] as usize);
        }
    }

    // Measure. The induced subgraph is built as a graph in its own right, because the width of a
    // SUBGRAPH is not any property of the whole one.
    let mut map = vec![usize::MAX; g.n];
    for (a, &i) in block.iter().enumerate() {
        map[i] = a;
    }
    let mut gb = GraphBuilder::new(block.len());
    for (a, &i) in block.iter().enumerate() {
        for k in g.offset[i]..g.offset[i + 1] {
            let j = g.nbr[k] as usize;
            if map[j] != usize::MAX && j > i {
                gb.couple(a, map[j], g.w[k]);
            }
        }
    }
    let induced = gb.build();
    if Elimination::default().width(&induced) > max_width {
        return None;
    }
    Some(block)
}

/// How to run the descent.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Params {
    /// Block moves to attempt.
    pub steps: usize,
    /// Nodes per block. Larger blocks step over larger barriers and cost `2^w` to solve, but a
    /// tree block's width is 1 whatever its size, so this is bounded by memory and not by width.
    pub block: usize,
    /// Width ceiling for the exact solve. Only bites for blocks from [`grown_block`]; a tree is
    /// width 1.
    pub max_width: usize,
    /// Attempt blocks that are grown freely and measured, rather than grown as trees.
    ///
    /// A wider block is a strictly stronger move. It is also refusable: a grown block that measures
    /// too wide is skipped, and `Outcome::refused` counts how often that happened, so a run that
    /// mostly skipped is visible rather than merely disappointing.
    pub grown: bool,
}

impl Default for Params {
    fn default() -> Self {
        Params { steps: 400, block: 64, max_width: 12, grown: false }
    }
}

/// What the descent did.
#[derive(Clone, Debug)]
pub struct Outcome {
    pub state: Vec<i8>,
    /// Recomputed from the state, not accumulated: an energy carried along a run of deltas drifts,
    /// and a drifting energy is indistinguishable from a search that is doing well.
    pub energy: f64,
    /// Block moves that actually ran.
    pub moves: usize,
    /// Moves that strictly lowered the energy. A descent whose blocks all land on a minimum they
    /// already sit in has stopped, and this is how a caller can tell.
    pub improving: usize,
    /// Blocks refused for width. Always 0 unless `grown` is set.
    pub refused: usize,
}

/// Run the descent from a random start.
pub fn run(g: &Graph, p: &Params, seed: u64) -> Outcome {
    let mut rng = Pcg::new(seed, 0x004F_5300);
    let s: Vec<i8> = (0..g.n).map(|_| rng.spin(0.5)).collect();
    run_from(g, s, p, seed)
}

/// Run the descent from a state you already have — from an anneal, or from tabu.
///
/// The composable form. HFS is a descent: it never raises the energy, so it cannot undo whatever
/// found the state it starts from, and starting it from a good state is strictly better than
/// starting it from noise.
pub fn run_from(g: &Graph, start: Vec<i8>, p: &Params, seed: u64) -> Outcome {
    let el = Elimination { max_width: p.max_width.max(1) };
    let mut rng = Pcg::new(seed, 0x0048_4653);
    let mut s = start;
    let (mut moves, mut improving, mut refused) = (0usize, 0usize, 0usize);

    for _ in 0..p.steps {
        if g.n == 0 {
            break;
        }
        let seed_node = (rng.f64() * g.n as f64) as usize % g.n;
        let block = if p.grown {
            match grown_block(g, seed_node, p.block, p.max_width, &mut rng) {
                Some(b) => b,
                None => {
                    refused += 1;
                    continue;
                }
            }
        } else {
            tree_block(g, seed_node, p.block, &mut rng)
        };
        match step(g, &mut s, &block, &el) {
            Ok(d) => {
                moves += 1;
                if d < -1e-12 {
                    improving += 1;
                }
            }
            Err(_) => refused += 1,
        }
    }

    Outcome { energy: g.energy(&s), state: s, moves, improving, refused }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::GraphBuilder;

    fn glass(l: usize, seed: u64) -> Graph {
        let mut rng = Pcg::new(seed, 0x0F5A);
        let mut b = GraphBuilder::new(l * l);
        for y in 0..l {
            for x in 0..l {
                let i = y * l + x;
                b.couple(i, y * l + (x + 1) % l, if rng.f64() < 0.5 { 1.0 } else { -1.0 });
                b.couple(i, ((y + 1) % l) * l + x, if rng.f64() < 0.5 { 1.0 } else { -1.0 });
            }
        }
        b.build()
    }

    /// The conditioning, against BRUTE FORCE over the block.
    ///
    /// Enumerating `2^|B|` assignments of the block and scoring each with the WHOLE graph's energy
    /// is a completely different computation from folding frozen neighbours into fields and
    /// eliminating variables. If the residual field had the wrong sign, or counted an in-block
    /// neighbour as frozen, this is what would catch it — and nothing else here would.
    #[test]
    fn a_block_move_finds_exactly_what_enumerating_the_block_finds() {
        let el = Elimination::default();
        for seed in [1u64, 42, 777] {
            let g = glass(5, seed);
            let mut rng = Pcg::new(seed ^ 0xB10C, 5);
            let mut s: Vec<i8> = (0..g.n).map(|_| rng.spin(0.5)).collect();

            // A block small enough to enumerate, and NOT a tree, so the in-block couplings matter.
            let block: Vec<usize> = vec![0, 1, 2, 5, 6, 7, 10, 11];
            let start = s.clone();

            let mut best_e = f64::INFINITY;
            let mut best = start.clone();
            for mask in 0..(1u32 << block.len()) {
                let mut cand = start.clone();
                for (bit, &i) in block.iter().enumerate() {
                    cand[i] = if mask >> bit & 1 == 1 { 1 } else { -1 };
                }
                let e = g.energy(&cand);
                if e < best_e {
                    best_e = e;
                    best = cand;
                }
            }

            let d = step(&g, &mut s, &block, &el).expect("a width-8 block is narrow");
            assert!(
                (g.energy(&s) - best_e).abs() < 1e-9,
                "seed {seed}: block move got {} where enumeration gets {best_e}",
                g.energy(&s)
            );
            assert!(d <= 1e-12, "a block move can never raise the energy: {d}");
            // The states themselves, not only the energies -- a degenerate optimum would let two
            // different states share one energy, and both are correct, so compare the energy of
            // the enumerated best rather than demanding the same assignment.
            assert!((g.energy(&best) - g.energy(&s)).abs() < 1e-9);
        }
    }

    /// A tree block is acyclic by construction, so its width is 1 and no solve can be refused for
    /// width. If the "exactly one neighbour inside" rule ever slips, this is where it shows.
    #[test]
    fn a_tree_block_is_a_tree() {
        let g = glass(8, 3);
        let mut rng = Pcg::new(9, 1);
        for target in [4usize, 16, 40] {
            for seed in 0..12usize {
                let block = tree_block(&g, seed, target, &mut rng);
                let inside: std::collections::BTreeSet<usize> = block.iter().copied().collect();
                assert_eq!(inside.len(), block.len(), "a block is a set");

                // A tree on k nodes has exactly k-1 edges. Count the induced ones.
                let mut edges = 0usize;
                for &i in &block {
                    for k in g.offset[i]..g.offset[i + 1] {
                        let j = g.nbr[k] as usize;
                        if j > i && inside.contains(&j) {
                            edges += 1;
                        }
                    }
                }
                assert_eq!(
                    edges,
                    block.len() - 1,
                    "target {target}, seed {seed}: {} nodes and {edges} induced edges is not a tree",
                    block.len()
                );

                // And the exact solver agrees it is width 1.
                let mut map = vec![usize::MAX; g.n];
                for (a, &i) in block.iter().enumerate() {
                    map[i] = a;
                }
                let mut gb = GraphBuilder::new(block.len());
                for (a, &i) in block.iter().enumerate() {
                    for k in g.offset[i]..g.offset[i + 1] {
                        let j = g.nbr[k] as usize;
                        if map[j] != usize::MAX && j > i {
                            gb.couple(a, map[j], g.w[k]);
                        }
                    }
                }
                if block.len() > 1 {
                    assert!(Elimination::default().width(&gb.build()) <= 1);
                }
            }
        }
    }

    /// The energy never rises. A descent that can raise it is not a descent, and every claim about
    /// composing this after an anneal rests on it.
    #[test]
    fn the_descent_never_raises_the_energy() {
        let g = glass(10, 11);
        let el = Elimination::default();
        let mut rng = Pcg::new(4, 1);
        let mut s: Vec<i8> = (0..g.n).map(|_| rng.spin(0.5)).collect();
        let mut e = g.energy(&s);
        for k in 0..60usize {
            let block = tree_block(&g, k * 7, 24, &mut rng);
            step(&g, &mut s, &block, &el).unwrap();
            let now = g.energy(&s);
            assert!(now <= e + 1e-9, "step {k}: {e} -> {now}");
            e = now;
        }
    }

    /// It has to actually be worth having. A block move sees barriers a single flip cannot, so on a
    /// frustrated glass it should beat plain steepest descent given the same starting states.
    #[test]
    fn block_moves_beat_single_flip_descent_from_the_same_starts() {
        let g = glass(12, 0xC0DE);
        let el = Elimination::default();
        let (mut hfs_wins, mut ties, mut losses) = (0, 0, 0);

        for seed in 0..12u64 {
            let mut rng = Pcg::new(seed, 0xA1);
            let start: Vec<i8> = (0..g.n).map(|_| rng.spin(0.5)).collect();

            // HFS from this start.
            let mut a = start.clone();
            let mut r = Pcg::new(seed, 0xB2);
            for k in 0..40usize {
                let block = tree_block(&g, k * 13, 48, &mut r);
                step(&g, &mut a, &block, &el).unwrap();
            }

            // Steepest descent from the SAME start: flip whatever helps most, until nothing does.
            let mut b = start.clone();
            loop {
                let mut best = (0.0f64, usize::MAX);
                for i in 0..g.n {
                    let mut f = g.h[i];
                    for k in g.offset[i]..g.offset[i + 1] {
                        f += g.w[k] * b[g.nbr[k] as usize] as f64;
                    }
                    // Flipping i changes E by +2 * f * s_i.
                    let d = 2.0 * f * b[i] as f64;
                    if d < best.0 - 1e-12 {
                        best = (d, i);
                    }
                }
                if best.1 == usize::MAX {
                    break;
                }
                b[best.1] = -b[best.1];
            }

            let (ea, eb) = (g.energy(&a), g.energy(&b));
            if ea < eb - 1e-9 {
                hfs_wins += 1;
            } else if ea > eb + 1e-9 {
                losses += 1;
            } else {
                ties += 1;
            }
        }
        assert!(
            hfs_wins > losses,
            "block moves must beat single flips on a frustrated glass: {hfs_wins} wins, \
             {losses} losses, {ties} ties"
        );
    }

    /// A grown block is measured, not assumed, and a wide one is refused rather than attempted.
    #[test]
    fn a_grown_block_is_refused_when_it_measures_too_wide() {
        // Dense on 24 nodes: any sizeable induced subgraph is far past width 3.
        let mut b = GraphBuilder::new(24);
        for i in 0..24 {
            for j in (i + 1)..24 {
                b.couple(i, j, 0.5);
            }
        }
        let g = b.build();
        let mut rng = Pcg::new(1, 1);
        assert!(grown_block(&g, 0, 16, 3, &mut rng).is_none(), "16 dense nodes are not width 3");
        // And a block of two is width 1 whatever the graph, so the ceiling is about the BLOCK.
        assert!(grown_block(&g, 0, 2, 3, &mut rng).is_some());
    }

    #[test]
    fn a_run_reports_what_it_did_and_reproduces_itself() {
        let g = glass(8, 5);
        let p = Params { steps: 30, block: 20, ..Params::default() };
        let a = run(&g, &p, 7);
        let b = run(&g, &p, 7);
        assert_eq!(a.state, b.state, "same seed, same run");
        assert_eq!(a.energy, b.energy);
        assert!(a.moves > 0 && a.refused == 0, "{a:?}");
        // The energy is recomputed from the state, not accumulated from deltas.
        assert!((g.energy(&a.state) - a.energy).abs() < 1e-12);
        // A different seed is a different run, so the check above is not passing on a constant.
        assert!(run(&g, &p, 8).state != a.state);
    }

    #[test]
    fn an_empty_block_and_an_empty_graph_are_no_ops_rather_than_panics() {
        let g = glass(4, 1);
        let el = Elimination::default();
        let mut s = vec![1i8; g.n];
        assert_eq!(step(&g, &mut s, &[], &el).unwrap(), 0.0);
        // Out-of-range entries are dropped rather than panicking: a grower is allowed to be wrong.
        assert_eq!(step(&g, &mut s, &[9999], &el).unwrap(), 0.0);
        let empty = GraphBuilder::new(0).build();
        assert_eq!(run(&empty, &Params::default(), 1).moves, 0);
    }
}
