//! Minor embedding: put a program on a machine whose graph is not the program's graph.
//!
//! Almost no annealer is fully connected. A model saying "these two variables interact" needs those
//! two variables to sit on sites the hardware actually couples, and when the model asks for more
//! neighbours than a site has, one variable becomes a **chain** of sites held together strongly
//! enough that they act as one.
//!
//! That is the layer this crate has been missing, and its absence made the declared fabrics
//! academic: [`crate::fabric`] could tell you a program needed embedding and could not perform it,
//! and the Hitachi driver refused any coupling that was not already King-adjacent rather than
//! placing it.
//!
//! # The algorithm
//!
//! The published heuristic (Cai, Macready and Roy, 2014), which is what `minorminer` implements.
//! Place variables one at a time; to place one, find the cheapest connected set of sites that
//! touches every already-placed neighbour, where "cheapest" charges more for sites already in use.
//! Then rip up each variable in turn and re-place it, which lets an early greedy choice be undone
//! once its neighbours exist. Repeat until no site is shared.
//!
//! Three details in that sentence are the whole algorithm rather than trimmings, and getting each
//! of them wrong is a distinct way for the placer to fail:
//!
//! * **Overlap is priced, not forbidden.** A variable may stand on a neighbour's site and pay for
//!   it. That is the only move that makes a chain longer than the paths that built it, and without
//!   it a variable with more neighbours than the hardware has degree can never grow.
//! * **A route keeps the sites it travelled through.** Paths cut through other variables' chains
//!   constantly; those crossings are a temporary, priced overlap for the rounds to resolve.
//!   Dropping them severs the chain from the neighbour the route was built to reach.
//! * **A round that does not converge is not a verdict.** "No site is shared" is weaker than "is an
//!   embedding", so a round can pass the first test and fail the second; that is a round to redo,
//!   not a search to abandon. A run that stops improving is started over from an empty machine.
//!
//! # What is guaranteed
//!
//! [`Embedding::verify`] checks the two properties that make an embedding an embedding: every chain
//! is connected in the hardware graph, and every logical edge has a hardware edge between its
//! endpoints' chains. It is checked on every embedding this module returns, so a broken embedding
//! is a panic here rather than wrong answers on a machine.
//!
//! # What is not
//!
//! Finding a minor is NP-hard, and this is a heuristic: failing to find one does **not** mean none
//! exists. [`embed`] returns `None`, which almost always means "not found" rather than
//! "impossible". A program already laid out for the hardware is checked for first and returned as
//! itself.
//!
//! **There is exactly one exception, and it is a proof.** [`site_lower_bound`] counts the sites any
//! embedding would need — a chain of `L` sites on degree-`d` hardware can offer at most `L(d-2)+2`
//! ports to other chains, so a variable of degree `k` needs a chain of at least
//! `ceil((k-2)/(d-2))` — and when that sum exceeds the machine, no embedding exists. `embed_with`
//! checks it before searching. That is what stands between a caller and the ninety-five seconds a
//! hopeless dense program used to spend proving nothing: `K_60` and `K_100` on a 512-site Chimera
//! are now refused in microseconds, while `K_33` and `K_40`, which the counting argument cannot
//! rule out, are still searched for properly.
//!
//! The search is also bounded — see [`DEFAULT_SEARCH_BUDGET`]. Saying "no" used to be free because
//! the old placer abandoned the whole search on the first variable it could not route; repairing
//! that is most of why cliques embed at all, and it also means a hopeless input now runs the search
//! it was always meant to run. [`embed_bounded`] takes the ceiling explicitly.
//!
//! # How far it actually gets
//!
//! Measured on `chimera(8, 8, 4)` — 512 sites, degree 6 except for the 128 boundary sites, which
//! have five — with the default twenty rip-up rounds, sixteen seeds each, and every result checked
//! by [`Embedding::verify`]:
//!
//! | program | was | is |
//! |---|---|---|
//! | a star of 8, 12 or 20 leaves | 0 of 16 seeds | 16 of 16 |
//! | a star of 6 leaves | 9 of 16 | 16 of 16 |
//! | `K_8`, `K_12`, `K_20` | 0 of 16 | 16 of 16 |
//! | `K_16` | 0 of 16 | 15 of 16 |
//! | `K_24` | 0 of 16 | 15 of 16 |
//! | `K_26` | 0 of 16 | 5 of 16 |
//! | `K_28` | 0 of 16 | 2 of 16 |
//!
//! It is not a repair aimed at cliques and stars. On 120 random graphs per machine — 6 to 18
//! variables, edge probability 0.2 to 0.7, one seed each, every success verified — the count went
//! from 56 to **115** on `chimera(8, 8, 4)`, 43 to **68** on a 64-site King's graph, and 26 to
//! **48** on a 64-site grid.
//!
//! The largest clique that fits `chimera(8, 8, 4)` at all is `K_33`, so `K_26` upward is where this
//! heuristic starts to be the binding constraint rather than the machine. More rounds buy some of
//! that back — `embed_with(.., 80)` reaches `K_26` on 14 of 16 seeds and `K_28` on 10 — at a cost
//! paid mostly by programs that were never going to fit, since a search that will not succeed now
//! spends its whole budget finding that out.
//!
//! Chains are longer than they need to be, and that is the next thing to fix rather than a
//! footnote: `K_24` embeds with a longest chain around 20 where a hand construction uses about 12.
//! Chain length is what dilutes the model on a real machine, and it is also what stops `K_28` — 28
//! chains of 26 sites do not fit in 512.
//!
//! # ⛔ What this section used to say
//!
//! Until this repair, a star with eight leaves — the simplest graph that cannot fit a degree-6
//! machine without one chain — was **not embedded onto a 512-site Chimera**, and every clique past
//! `K_7` failed with it. Two independent defects, both in the placement step:
//!
//! * The cheapest set of sites touching every neighbour was chosen and then every site belonging to
//!   a neighbour was **subtracted from it**. For a variable whose neighbours are all one hop away —
//!   a star's centre, exactly — that subtraction leaves a single site, every round, for ever, so a
//!   chain could never grow. Forbidding a variable from standing on a neighbour's site also deleted
//!   the only move in the published heuristic that lengthens a chain under pressure.
//! * The same subtraction removed sites that were merely **on the way** to one neighbour because
//!   they belonged to another. That severed the chain from the neighbour the path had been built to
//!   reach, producing a placement in which no site was shared and which was still not an embedding.
//!   The loop's only test was "is any site shared", so it declared victory, and the verify that
//!   followed turned one bad round into `None` for the entire search. Cliques never reached round
//!   one. That is why more rounds and a bigger machine were both measured to change nothing.
//!
//! Ramping the overlap penalty, which is the published fix for a rip-up loop that will not
//! converge, was tried and measured not to help, and that measurement stands: both options a
//! congested variable has pay the penalty, so scaling it leaves their order unchanged. The price
//! was never the defect.

use crate::graph::{Graph, GraphBuilder};
use std::cmp::Reverse;
use std::collections::{BTreeSet, BinaryHeap, VecDeque};

/// A placement of logical variables onto hardware sites.
#[derive(Clone, Debug, PartialEq)]
pub struct Embedding {
    /// For each logical variable, the sites it occupies. Never empty for a variable that appears.
    pub chains: Vec<Vec<usize>>,
    /// How many sites the hardware has.
    pub sites: usize,
}

impl Embedding {
    /// The longest chain. Chains cost coupling budget and dilute the model, so this is the number
    /// to compare two embeddings by.
    pub fn longest_chain(&self) -> usize {
        self.chains.iter().map(|c| c.len()).max().unwrap_or(0)
    }

    /// Total sites used.
    pub fn used(&self) -> usize {
        self.chains.iter().map(|c| c.len()).sum()
    }

    /// Is this actually an embedding of `logical` into `hardware`?
    ///
    /// Two properties, and both matter: a chain that is not connected is not one variable, and a
    /// logical edge with no hardware edge between the chains is an interaction the machine cannot
    /// represent. Returns the first failure in words.
    pub fn verify(&self, logical: &Graph, hardware: &Graph) -> Result<(), String> {
        let mut seen = vec![usize::MAX; hardware.n];
        for (v, chain) in self.chains.iter().enumerate() {
            if chain.is_empty() {
                return Err(format!("variable {v} has no sites"));
            }
            for &s in chain {
                if s >= hardware.n {
                    return Err(format!("variable {v} uses site {s}, past the {} the machine has", hardware.n));
                }
                if seen[s] != usize::MAX {
                    return Err(format!("site {s} is used by both {} and {v}", seen[s]));
                }
                seen[s] = v;
            }
            if !connected(chain, hardware) {
                return Err(format!("variable {v}'s chain {chain:?} is not connected in the hardware"));
            }
        }
        for i in 0..logical.n {
            for k in logical.offset[i]..logical.offset[i + 1] {
                let j = logical.nbr[k] as usize;
                if j <= i {
                    continue;
                }
                if !touching(&self.chains[i], &self.chains[j], hardware) {
                    return Err(format!(
                        "variables {i} and {j} interact, and no site of one is adjacent to a site \
                         of the other"
                    ));
                }
            }
        }
        Ok(())
    }
}

fn connected(chain: &[usize], h: &Graph) -> bool {
    if chain.len() <= 1 {
        return true;
    }
    let set: BTreeSet<usize> = chain.iter().copied().collect();
    let mut seen = BTreeSet::new();
    let mut q = VecDeque::from([chain[0]]);
    seen.insert(chain[0]);
    while let Some(u) = q.pop_front() {
        for k in h.offset[u]..h.offset[u + 1] {
            let v = h.nbr[k] as usize;
            if set.contains(&v) && seen.insert(v) {
                q.push_back(v);
            }
        }
    }
    seen.len() == chain.len()
}

fn touching(a: &[usize], b: &[usize], h: &Graph) -> bool {
    let bs: BTreeSet<usize> = b.iter().copied().collect();
    a.iter().any(|&u| {
        (h.offset[u]..h.offset[u + 1]).any(|k| bs.contains(&(h.nbr[k] as usize)))
    })
}

/// Find an embedding of `logical` into `hardware`, or `None` if this heuristic could not.
///
/// `None` means not found. Deciding whether a minor exists is NP-hard, so an honest answer here is
/// never "impossible" — a different seed, a longer run, or a better heuristic may succeed.
pub fn embed(logical: &Graph, hardware: &Graph, seed: u64) -> Option<Embedding> {
    embed_with(logical, hardware, seed, 20)
}

/// The fewest sites any embedding of `logical` into `hardware` could possibly use.
///
/// A COUNTING ARGUMENT, not a heuristic. A chain of `L` sites in a graph of maximum degree `d` has
/// at most `L*d` edge endpoints in total, and being connected it spends at least `2(L-1)` of them
/// on itself — so it can offer at most `L(d-2) + 2` ports to other chains. A variable of logical
/// degree `k` needs `k` of those ports, so its chain cannot be shorter than
/// `ceil((k - 2) / (d - 2))`. Summing that floor over every variable bounds the whole embedding.
///
/// **When this exceeds `hardware.n`, no embedding exists** — and that is a proof rather than a
/// failure to find one, which is the only place in this module where `None` means "impossible"
/// rather than "not found". [`embed_with`] checks it before searching, which is what turns a
/// hopeless dense input from ninety-five seconds of futile rip-up into a few microseconds of
/// arithmetic: K_60 and K_100 on a 512-site Chimera are refused instantly, while K_33 and K_40 —
/// which the counting argument cannot rule out — are still searched for properly.
///
/// The bound is loose by design. It ignores that ports must reach DISTINCT chains, that chains
/// compete for the same sites, and every question of geometry, so passing it says nothing at all
/// about whether an embedding exists.
pub fn site_lower_bound(logical: &Graph, hardware: &Graph) -> usize {
    let d = (0..hardware.n)
        .map(|s| hardware.offset[s + 1] - hardware.offset[s])
        .max()
        .unwrap_or(0);
    // With d <= 2 a chain offers at most 2 ports however long it is, so the argument says nothing
    // beyond "one site each" and this reports exactly that rather than dividing by zero.
    if d <= 2 {
        return logical.n;
    }
    (0..logical.n)
        .map(|v| {
            let k = degree(logical, v);
            if k <= d {
                1
            } else {
                (k - 2).div_ceil(d - 2)
            }
        })
        .sum()
}

/// How many shortest-path searches [`embed_with`] will run before giving up.
///
/// SAYING "NO" USED TO BE FREE AND IS NOT ANY MORE, and that is a direct consequence of the repair
/// rather than an oversight in it. The old placer aborted the entire search on the first variable it
/// could not route, so a hopeless input returned `None` in microseconds — it never spent its round
/// budget because it never reached round 1. Fixing that abort is most of why cliques embed at all
/// now, and it also means a hopeless input runs the search it was always supposed to run. Measured
/// on chimera(8,8,4), unbounded: K_33 1.9 s, K_40 2.5 s, K_60 16.9 s, K_100 95.3 s to answer "no".
///
/// Ninety-five seconds with no output is not an answer a library may give a caller, and
/// [`crate::fabric`] and the Hitachi driver both reach this path. So the work is bounded.
///
/// The number is one Dijkstra per placed neighbour per variable per round. 200,000 of them is
/// roughly a second on the hardware this was measured on, and it is far above what any input that
/// SUCCEEDS has been observed to need — the whole K_8..K_24 sweep on chimera(8,8,4) fits inside
/// 5,000. A bound that cut off a case that would have succeeded would be worse than the latency it
/// prevents, so it is set an order of magnitude clear of the worst success, not at it.
pub const DEFAULT_SEARCH_BUDGET: u64 = 200_000;

/// As [`embed_with`], with an explicit ceiling on shortest-path searches.
///
/// Returns `None` when the budget runs out, which is the same `None` as "not found" and means the
/// same thing: this heuristic did not find an embedding. It never means none exists. Raise the
/// budget for a large machine, or pass `u64::MAX` for the unbounded search — and see
/// [`DEFAULT_SEARCH_BUDGET`] for what unbounded costs on a dense input that cannot be placed.
pub fn embed_bounded(
    logical: &Graph,
    hardware: &Graph,
    seed: u64,
    rounds: usize,
    budget: u64,
) -> Option<Embedding> {
    embed_inner(logical, hardware, seed, rounds, budget)
}

/// As [`embed`], with an explicit number of rip-up rounds.
pub fn embed_with(logical: &Graph, hardware: &Graph, seed: u64, rounds: usize) -> Option<Embedding> {
    embed_inner(logical, hardware, seed, rounds, DEFAULT_SEARCH_BUDGET)
}

fn embed_inner(
    logical: &Graph,
    hardware: &Graph,
    seed: u64,
    rounds: usize,
    budget: u64,
) -> Option<Embedding> {
    let mut spent: u64 = 0;
    if logical.n == 0 {
        return Some(Embedding { chains: Vec::new(), sites: hardware.n });
    }
    if logical.n > hardware.n {
        return None; // more variables than sites, before any structure is considered
    }
    // A PROOF OF IMPOSSIBILITY, taken before any search. See `site_lower_bound`: this is the one
    // `None` in this module that means "no embedding exists" rather than "this heuristic did not
    // find one". It costs a pass over the degrees and it is what stands between a caller and the
    // ninety-five seconds a hopeless dense input used to spend proving nothing.
    if site_lower_bound(logical, hardware) > hardware.n {
        return None;
    }

    // A program already laid out for this machine embeds as itself, instantly and with no chains.
    // That is the common case for anyone who placed their model on the grid by hand -- which is
    // what the Hitachi driver demanded of every caller before this module existed -- and it is also
    // the case a rip-up heuristic handles worst, because a perfect packing leaves it no room to
    // move anything.
    let identity = Embedding {
        chains: (0..logical.n).map(|i| vec![i]).collect(),
        sites: hardware.n,
    };
    if identity.verify(logical, hardware).is_ok() {
        return Some(identity);
    }

    let mut rng = Pcg::new(seed);

    // Place the most-connected variables first: they are the hardest to fit, and fitting them last
    // means fitting them into whatever is left. Later rounds shuffle, because a variable that was
    // placed well against an empty machine is exactly the one that has to move once the machine is
    // not empty, and a fixed order re-derives the same bad arrangement every round.
    let mut order: Vec<usize> = (0..logical.n).collect();
    order.sort_by_key(|&v| core::cmp::Reverse(degree(logical, v)));

    let mut chains: Vec<Vec<usize>> = vec![Vec::new(); logical.n];
    // How far a round is from being an embedding: the extra occupants sites carry, then the sites
    // used at all. This is the measure the published heuristic improves on, and a run that stops
    // improving on it is stuck rather than slow -- so it is restarted rather than ground on.
    let mut best = (usize::MAX, usize::MAX);
    let mut stale = 0usize;

    for round in 0..=rounds {
        // Checked per ROUND rather than per variable, so a round is never abandoned half-placed:
        // the loop below rips up each variable and re-places it, and stopping between those two
        // leaves a variable with no sites at all -- which every later round would then read as an
        // unplaced neighbour. Giving up on a whole-round boundary leaves a consistent layout that
        // simply is not an embedding yet, which is exactly what `None` means here.
        if spent >= budget {
            return None;
        }
        for idx in 0..order.len() {
            let v = order[idx];
            chains[v].clear(); // rip up, then re-place against everyone else's current position

            // How busy each site is, counting everyone except v.
            let mut load = vec![0usize; hardware.n];
            for (u, c) in chains.iter().enumerate() {
                if u != v {
                    for &s in c {
                        load[s] += 1;
                    }
                }
            }

            let neighbours: Vec<usize> = (logical.offset[v]..logical.offset[v + 1])
                .map(|k| logical.nbr[k] as usize)
                .filter(|&u| !chains[u].is_empty())
                .collect();

            let mut chain = if neighbours.is_empty() {
                // Nothing to be near yet: take the emptiest site, preferring one with the most
                // hardware neighbours. Preferring is not cosmetic -- a quarter of Chimera's sites
                // sit on a boundary and have five neighbours rather than six, and a first
                // placement that lands on one of those is a variable that starts a whole run one
                // neighbour short. Ties break randomly so a restart explores somewhere new.
                match seed_site(hardware, &load, &mut rng) {
                    Some(s) => vec![s],
                    None => continue,
                }
            } else {
                match vertex_model(hardware, &neighbours, &chains, &load, &mut rng, &mut spent) {
                    Some(c) => c,
                    // Nowhere reachable from every neighbour at once. That is this PLACEMENT
                    // failing, not the search: leave v unplaced and let the next round try it
                    // against a different arrangement.
                    None => continue,
                }
            };
            // A chain is also the frontage every neighbour has to land on, not just a route
            // between the ones already placed. A single site on a degree-6 machine seats six
            // neighbours and no more, so a degree-8 variable placed on one site strands two of
            // them wherever it goes and however many rounds it is given.
            //
            // The full degree, not the neighbours still waiting: sizing to the waiting ones was
            // measured and is worse -- K_16 15/16 -> 9/16, K_20 16/16 -> 4/16, K_24 15/16 -> 1/16
            // on chimera(8, 8, 4) -- because a chain sized to exactly the neighbours it has leaves
            // them nowhere to move to when it is their own turn to be ripped up.
            grow_to_fit(hardware, &mut chain, &load, degree(logical, v));
            chains[v] = chain;
        }

        // Every variable has now moved at least once, so a chain built early in the round was
        // built to reach neighbours that are no longer where they were. Drop whatever it turned
        // out not to need. This only ever removes sites, so it can lower an overlap and can never
        // create one, and the sites it hands back are what the next round has to work with.
        for v in 0..logical.n {
            if chains[v].is_empty() {
                continue;
            }
            let nbrs: Vec<usize> = (logical.offset[v]..logical.offset[v + 1])
                .map(|k| logical.nbr[k] as usize)
                .filter(|&u| !chains[u].is_empty())
                .collect();
            let mut c = core::mem::take(&mut chains[v]);
            prune(hardware, &mut c, &nbrs, &chains);
            chains[v] = c;
        }

        let mut load = vec![0usize; hardware.n];
        for c in &chains {
            for &s in c {
                load[s] += 1;
            }
        }
        let excess: usize = load.iter().map(|&n| n.saturating_sub(1)).sum();
        let all_placed = chains.iter().all(|c| !c.is_empty());

        if excess == 0 && all_placed {
            let e = Embedding { chains: chains.clone(), sites: hardware.n };
            // An embedding this module returns is always checked. A wrong one does not fail
            // loudly on a machine; it returns plausible answers to a different problem.
            //
            // A failure here is NOT the search failing. "No site is shared" is a weaker property
            // than "is an embedding", so a round can pass that test on a placement verify rejects;
            // treating that as a verdict on the whole search is what used to abandon cliques in
            // round zero with the entire round budget unspent.
            if e.verify(logical, hardware).is_ok() {
                return Some(e);
            }
        }
        if round == rounds {
            break;
        }

        let here = (excess, chains.iter().map(|c| c.len()).sum::<usize>());
        if here < best {
            best = here;
            stale = 0;
        } else {
            stale += 1;
        }
        if stale >= STALL {
            // Not converging. Rip the whole placement up rather than spending the remaining
            // rounds refining an arrangement that has stopped getting better.
            for c in chains.iter_mut() {
                c.clear();
            }
            order.sort_by_key(|&v| core::cmp::Reverse(degree(logical, v)));
            best = (usize::MAX, usize::MAX);
            stale = 0;
        } else {
            shuffle(&mut order, &mut rng);
        }
    }
    None
}

/// Rounds without improvement before [`embed_with`] starts over from an empty machine.
const STALL: usize = 6;

/// What one more occupant of a site adds to the price of routing through it.
const OCCUPIED_BASE: u64 = 8;

/// How many logical neighbours a variable has.
fn degree(g: &Graph, v: usize) -> usize {
    g.offset[v + 1] - g.offset[v]
}

/// What it costs to take a site: one, plus a penalty for everyone already there.
fn site_cost(load: &[usize], s: usize) -> u64 {
    1 + load[s] as u64 * OCCUPIED_BASE
}

/// The emptiest site, preferring hardware degree, breaking ties at random.
fn seed_site(h: &Graph, load: &[usize], rng: &mut Pcg) -> Option<usize> {
    (0..h.n).min_by_key(|&s| {
        (load[s], core::cmp::Reverse(h.offset[s + 1] - h.offset[s]), rng.next() % 1024)
    })
}

/// Fisher-Yates, on the crate's own PCG so an order is reproducible from a seed.
fn shuffle(order: &mut [usize], rng: &mut Pcg) {
    for i in (1..order.len()).rev() {
        let j = (rng.next() % (i as u64 + 1)) as usize;
        order.swap(i, j);
    }
}

/// The cheapest connected set of sites touching every placed neighbour's chain.
///
/// This is `findMinimalVertexModel` from Cai, Macready and Roy (2014), and the two details that
/// look like details are the whole algorithm:
///
/// * **The root may sit on a neighbour's site**, priced at what that overlap costs rather than
///   forbidden. That is the only move in the heuristic that makes a chain longer than the paths
///   themselves, and it is how a variable with more neighbours than the hardware has degree ever
///   gets a second site: a neighbour with nowhere left lands *on* it, the site's price rises, and
///   the next re-placement is pushed one hop off it and has to reach back.
/// * **A path keeps every site except the neighbour's own.** Shortest paths run *through* other
///   variables' chains all the time; those sites are a legal, priced, temporary overlap that the
///   rip-up rounds resolve. Deleting them severs the chain from the very neighbour the path was
///   built to reach, which produces a placement that shares no site and is still not an embedding.
fn vertex_model(
    h: &Graph,
    neighbours: &[usize],
    chains: &[Vec<usize>],
    load: &[usize],
    rng: &mut Pcg,
    spent: &mut u64,
) -> Option<Vec<usize>> {
    let mut total = vec![0u64; h.n];
    let mut parents: Vec<Vec<usize>> = Vec::with_capacity(neighbours.len());

    for &u in neighbours {
        *spent += 1;
        let (dist, parent) = dijkstra(h, &chains[u], load);
        for s in 0..h.n {
            total[s] = total[s].saturating_add(dist[s]);
        }
        // A site of u's own is reachable at distance zero, which would make u's own chain the
        // free winner. Charge it what standing on it costs instead.
        for &s in &chains[u] {
            total[s] = total[s].saturating_add(site_cost(load, s));
        }
        parents.push(parent);
    }

    // The Steiner score ranks ROOTS; it is not what the chain costs. Paths share a prefix, and the
    // score charges that prefix once per path while the chain holds it once -- so the best-scoring
    // root is regularly not the cheapest chain. Rank by the score, then BUILD the leading few and
    // keep whichever really is cheapest once its paths are merged and the sites it does not need
    // are dropped. On a clique that is the difference between chains that sprawl across the
    // machine and chains that fit on it.
    let jitter: Vec<u64> = (0..h.n).map(|_| rng.next() % 8).collect();
    let mut ranked: Vec<usize> = (0..h.n).filter(|&s| total[s] < u64::MAX / 4).collect();
    ranked.sort_unstable_by_key(|&s| (total[s], jitter[s]));
    ranked.truncate(ROOT_CANDIDATES);

    let mut best: Option<(u64, Vec<usize>)> = None;
    for &root in &ranked {
        // Walk back from the root toward each neighbour, keeping every site on the way but the
        // neighbour's own. The last site kept is adjacent to one of theirs, so the chain touches
        // every neighbour by construction, and every site kept has its whole ancestry kept too, so
        // it is connected by construction.
        let mut set = BTreeSet::new();
        set.insert(root);
        let mut reachable = true;
        for parent in &parents {
            if parent[root] == usize::MAX {
                reachable = false; // not reachable from this neighbour at all
                break;
            }
            let mut at = root;
            while parent[at] != at {
                let p = parent[at];
                if parent[p] == p {
                    break; // p is the neighbour's own site: touch it, do not take it
                }
                set.insert(p);
                at = p;
            }
        }
        if !reachable {
            continue;
        }
        let mut chain: Vec<usize> = set.into_iter().collect();
        prune(h, &mut chain, neighbours, chains);
        let cost: u64 = chain.iter().map(|&s| site_cost(load, s)).sum();
        if best.as_ref().is_none_or(|(c, _)| cost < *c) {
            best = Some((cost, chain));
        }
    }
    best.map(|(_, chain)| chain)
}

/// How many of the best-scoring roots [`vertex_model`] builds a chain for before choosing.
const ROOT_CANDIDATES: usize = 8;

/// Drop every site a chain does not need, smallest first.
///
/// The union of shortest paths from one root is a tree, and a tree grown to reach several targets
/// routinely reaches one of them twice: a branch built for neighbour `a` ends next to `b` as well,
/// leaving `b`'s own branch redundant. Removing a site is allowed only when what is left is still
/// connected and still reaches every neighbour it reached before, so this can shorten a chain and
/// can never break one.
fn prune(h: &Graph, chain: &mut Vec<usize>, neighbours: &[usize], chains: &[Vec<usize>]) {
    let reaches = |c: &[usize], u: usize| -> bool {
        touching(c, &chains[u], h) || c.iter().any(|s| chains[u].contains(s))
    };
    let before: Vec<bool> = neighbours.iter().map(|&u| reaches(chain, u)).collect();
    let mut i = 0;
    while i < chain.len() {
        if chain.len() == 1 {
            break;
        }
        let trial: Vec<usize> =
            chain.iter().enumerate().filter(|&(k, _)| k != i).map(|(_, &s)| s).collect();
        let keeps = connected(&trial, h)
            && neighbours
                .iter()
                .zip(&before)
                .all(|(&u, &had)| !had || reaches(&trial, u));
        if keeps {
            *chain = trial;
            i = 0; // dropping one site can make another droppable
        } else {
            i += 1;
        }
    }
}

/// Extend a chain until it has as many free sites next to it as the variable has neighbours.
///
/// A chain is not just a route between the neighbours already placed; it is the frontage the
/// neighbours still to come have to land on. A single site on a degree-6 machine can seat six
/// neighbours and no more, so a degree-8 variable placed on one site strands two of them wherever
/// it goes and however many rounds it is given.
///
/// Stops as soon as there is room, and stops anyway when there is no free site adjacent, so a full
/// machine degrades to the shortest chain rather than looping.
fn grow_to_fit(h: &Graph, chain: &mut Vec<usize>, load: &[usize], want: usize) {
    let mut inside: BTreeSet<usize> = chain.iter().copied().collect();
    loop {
        let mut free: Vec<usize> = Vec::new();
        for &s in &inside {
            for k in h.offset[s]..h.offset[s + 1] {
                let v = h.nbr[k] as usize;
                if load[v] == 0 && !inside.contains(&v) && !free.contains(&v) {
                    free.push(v);
                }
            }
        }
        if free.len() >= want || free.is_empty() {
            break;
        }
        // Grow into the site that opens the most new frontage, so the chain gets wider rather
        // than merely longer.
        let pick = free
            .iter()
            .copied()
            .max_by_key(|&v| {
                (h.offset[v]..h.offset[v + 1])
                    .filter(|&k| {
                        let w = h.nbr[k] as usize;
                        load[w] == 0 && !inside.contains(&w) && !free.contains(&w)
                    })
                    .count()
            })
            .expect("free is not empty");
        inside.insert(pick);
    }
    *chain = inside.into_iter().collect();
}

/// Shortest paths from a set of sources, charging [`site_cost`] to enter a site.
fn dijkstra(h: &Graph, sources: &[usize], load: &[usize]) -> (Vec<u64>, Vec<usize>) {
    let mut dist = vec![u64::MAX; h.n];
    let mut parent = vec![usize::MAX; h.n];
    let mut heap: BinaryHeap<Reverse<(u64, usize)>> = BinaryHeap::new();
    for &s in sources {
        dist[s] = 0;
        parent[s] = s;
        heap.push(Reverse((0, s)));
    }
    while let Some(Reverse((d, u))) = heap.pop() {
        if d > dist[u] {
            continue;
        }
        for k in h.offset[u]..h.offset[u + 1] {
            let v = h.nbr[k] as usize;
            let nd = d.saturating_add(site_cost(load, v));
            if nd < dist[v] {
                dist[v] = nd;
                parent[v] = u;
                heap.push(Reverse((nd, v)));
            }
        }
    }
    (dist, parent)
}

/// A program rewritten onto the hardware's sites, with chains held together.
pub struct Embedded {
    /// The graph to run, over `hardware.n` sites.
    pub graph: Graph,
    /// The placement it came from, for reading answers back.
    pub embedding: Embedding,
    /// The coupling holding each chain together.
    ///
    /// Too weak and a chain breaks, so a variable has two values at once and means nothing. Too
    /// strong and it swamps the problem the model was actually about. Chosen as twice the largest
    /// coefficient in the logical model, which is the standard first guess and is reported rather
    /// than hidden so it can be tuned.
    pub chain_strength: f64,
}

/// Rewrite a logical model onto hardware sites under an embedding.
///
/// Couplings are shared out across the hardware edges that realise each logical edge, and fields
/// are shared out along each chain, so the total weight is unchanged however long a chain is.
pub fn apply(logical: &Graph, hardware: &Graph, e: &Embedding) -> Embedded {
    apply_with(logical, hardware, e, DEFAULT_CHAIN_MULTIPLE * worst_coefficient(logical))
}

/// The multiple of the largest logical coefficient [`apply`] holds chains together with.
///
/// **Four, and it was two until it was measured.** Two is the standard first guess in the
/// literature and it is wrong here. `examples/chain_strength` sweeps this against an optimum branch
/// and bound PROVED, on 24 twelve-variable cliques embedded into Chimera with chains up to 18 sites:
///
/// ```text
///   chain x   broken   gap above optimum   optimum found
///      1.00    32.6%          5.42             7/24
///      2.00     9.7%          1.50            15/24     <- the old default
///      3.00     2.1%          0.42            20/24
///      4.00     0.0%          0.50            20/24     <- this
///      8.00     0.0%          1.83            14/24
///     16.00     0.0%          4.67             5/24
/// ```
///
/// At two, a tenth of chains BREAK — one logical variable holding two values, resolved by a
/// majority vote that is a coin toss wearing a number. Four is the first multiple that breaks none,
/// and it ties the best gap and the best hit rate. Both failure modes are visible in that table and
/// they are not symmetric: too weak announces itself in the broken column, and too strong is
/// SILENT — sixteen breaks nothing, reports clean, and is nine times further from the optimum.
///
/// This was measured on one logical family, one machine and one annealing schedule, so it
/// calibrates a default rather than establishing a law; [`apply_with`] takes the number when a
/// model's own scale calls for a different one. It could not be measured at all until the placer
/// was repaired, because exhibiting the silent half needs chains long enough to swamp a search.
pub const DEFAULT_CHAIN_MULTIPLE: f64 = 4.0;

/// The largest absolute coupling or field in a model — the scale a chain has to outrank.
///
/// Returns `0.5` for a model with no weights at all, so the default chain strength stays `1.0`
/// there rather than collapsing to zero and holding nothing together.
pub fn worst_coefficient(logical: &Graph) -> f64 {
    let worst = (0..logical.n)
        .flat_map(|i| (logical.offset[i]..logical.offset[i + 1]).map(move |k| logical.w[k].abs()))
        .chain(logical.h.iter().map(|x| x.abs()))
        .fold(0.0f64, f64::max);
    if worst > 0.0 {
        worst
    } else {
        0.5
    }
}

/// Rewrite a logical model onto hardware sites, choosing the chain coupling yourself.
///
/// [`apply`] picks `2 x` the largest logical coefficient, which is the standard first guess and is
/// a GUESS: a chain has to outrank the couplings it carries or it breaks, and it has to not swamp
/// them or the machine spends its search holding chains together instead of solving anything. That
/// is the same trade-off [`crate::hubo`] measured for a reduction penalty, where the standard
/// choice turned out to make the landscape rigid enough to change the answer.
///
/// `examples/chain_strength` sweeps this against a proved optimum and reports where the trade-off
/// actually sits. Read it before overriding the default, and before trusting it.
///
/// A non-finite or non-positive `chain_strength` falls back to the default rather than building a
/// model whose chains do not hold.
pub fn apply_with(
    logical: &Graph,
    hardware: &Graph,
    e: &Embedding,
    chain_strength: f64,
) -> Embedded {
    let chain_strength = if chain_strength.is_finite() && chain_strength > 0.0 {
        chain_strength
    } else {
        DEFAULT_CHAIN_MULTIPLE * worst_coefficient(logical)
    };

    let mut b = GraphBuilder::new(hardware.n);

    // Hold each chain together.
    for chain in &e.chains {
        for a in 0..chain.len() {
            for c in (a + 1)..chain.len() {
                let (u, v) = (chain[a], chain[c]);
                if (hardware.offset[u]..hardware.offset[u + 1])
                    .any(|k| hardware.nbr[k] as usize == v)
                {
                    b.couple(u, v, chain_strength);
                }
            }
        }
    }

    // Every logical edge, spread over the hardware edges that realise it.
    for i in 0..logical.n {
        for k in logical.offset[i]..logical.offset[i + 1] {
            let j = logical.nbr[k] as usize;
            if j <= i {
                continue;
            }
            let mut links = Vec::new();
            for &u in &e.chains[i] {
                for kk in hardware.offset[u]..hardware.offset[u + 1] {
                    let v = hardware.nbr[kk] as usize;
                    if e.chains[j].contains(&v) {
                        links.push((u, v));
                    }
                }
            }
            if links.is_empty() {
                continue; // verify() rules this out; belt and braces
            }
            let share = logical.w[k] / links.len() as f64;
            for (u, v) in links {
                b.couple(u, v, share);
            }
        }
    }

    // Fields, spread along the chain.
    for i in 0..logical.n {
        if logical.h[i] == 0.0 {
            continue;
        }
        let share = logical.h[i] / e.chains[i].len() as f64;
        for &s in &e.chains[i] {
            b.bias(s, share);
        }
    }

    Embedded { graph: b.build(), embedding: e.clone(), chain_strength }
}

/// Read a hardware state back as logical values, by majority vote along each chain.
///
/// Returns the values and the variables whose chains **broke** — disagreed with themselves. A
/// broken chain means the answer for that variable is a coin toss dressed as a result, so it is
/// reported rather than silently resolved.
pub fn unembed(e: &Embedding, state: &[i8]) -> (Vec<i8>, Vec<usize>) {
    let mut out = vec![0i8; e.chains.len()];
    let mut broken = Vec::new();
    for (v, chain) in e.chains.iter().enumerate() {
        let up = chain.iter().filter(|&&s| state.get(s).copied().unwrap_or(0) > 0).count();
        let down = chain.len() - up;
        if up != 0 && down != 0 {
            broken.push(v);
        }
        out[v] = if up >= down { 1 } else { -1 };
    }
    (out, broken)
}

/// The tiny PCG used elsewhere in this crate, so an embedding is reproducible from its seed.
struct Pcg(u64);

impl Pcg {
    fn new(seed: u64) -> Pcg {
        Pcg(seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407))
    }
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        let x = self.0;
        (x >> 33) ^ x
    }
}

/// Hardware graphs to embed into.
pub mod topology {
    use super::*;

    /// A King's graph: an `l` by `l` grid where each site couples to its eight neighbours.
    ///
    /// Hitachi's CMOS annealer. Site `(x, y)` is index `y * l + x`.
    pub fn king(l: usize) -> Graph {
        let mut b = GraphBuilder::new(l * l);
        let at = |x: usize, y: usize| y * l + x;
        for y in 0..l {
            for x in 0..l {
                for (dx, dy) in [(1i64, 0i64), (0, 1), (1, 1), (1, -1)] {
                    let (nx, ny) = (x as i64 + dx, y as i64 + dy);
                    if nx >= 0 && ny >= 0 && (nx as usize) < l && (ny as usize) < l {
                        b.couple(at(x, y), at(nx as usize, ny as usize), 1.0);
                    }
                }
            }
        }
        b.build()
    }

    /// A plain `l` by `l` grid, four neighbours per site.
    pub fn grid(l: usize) -> Graph {
        let mut b = GraphBuilder::new(l * l);
        for y in 0..l {
            for x in 0..l {
                if x + 1 < l {
                    b.couple(y * l + x, y * l + x + 1, 1.0);
                }
                if y + 1 < l {
                    b.couple(y * l + x, (y + 1) * l + x, 1.0);
                }
            }
        }
        b.build()
    }

    /// A complete graph on `n` sites, for testing that embedding into something generous is easy.
    pub fn complete(n: usize) -> Graph {
        let mut b = GraphBuilder::new(n);
        for i in 0..n {
            for j in (i + 1)..n {
                b.couple(i, j, 1.0);
            }
        }
        b.build()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use topology::{complete, grid, king};

    /// A triangle: three variables, all interacting. Needs a chain on a grid, not on a King's graph.
    fn triangle() -> Graph {
        let mut b = GraphBuilder::new(3);
        b.couple(0, 1, 1.0);
        b.couple(1, 2, 1.0);
        b.couple(0, 2, 1.0);
        b.build()
    }

    fn clique(n: usize, w: f64) -> Graph {
        let mut b = GraphBuilder::new(n);
        for i in 0..n {
            for j in (i + 1)..n {
                b.couple(i, j, w);
            }
        }
        b.build()
    }

    #[test]
    fn a_graph_embeds_into_itself_with_no_chains() {
        let g = king(4);
        let e = embed(&g, &king(4), 1).expect("a graph is its own minor");
        assert_eq!(e.longest_chain(), 1, "nothing needs a chain");
        e.verify(&g, &king(4)).unwrap();
    }

    #[test]
    fn a_triangle_needs_no_chain_on_a_kings_graph_and_does_on_a_grid() {
        // A King's graph has triangles; a square grid has none, so one variable must stretch.
        let k = king(4);
        let ek = embed(&triangle(), &k, 7).expect("a King's graph has triangles");
        ek.verify(&triangle(), &k).unwrap();
        assert_eq!(ek.longest_chain(), 1, "no chain needed: {:?}", ek.chains);

        let g = grid(4);
        let eg = embed(&triangle(), &g, 7).expect("a triangle is a minor of a 4x4 grid");
        eg.verify(&triangle(), &g).unwrap();
        assert!(eg.longest_chain() > 1, "a square grid has no triangle: {:?}", eg.chains);
    }

    #[test]
    fn verify_rejects_an_embedding_that_is_not_one() {
        let g = grid(4);
        // a disconnected "chain"
        let broken = Embedding { chains: vec![vec![0, 15], vec![1], vec![2]], sites: g.n };
        let e = broken.verify(&triangle(), &g).unwrap_err();
        assert!(e.contains("not connected"), "{e}");

        // two variables on one site
        let shared = Embedding { chains: vec![vec![0], vec![0], vec![2]], sites: g.n };
        let e = shared.verify(&triangle(), &g).unwrap_err();
        assert!(e.contains("used by both"), "{e}");

        // an interaction with nowhere to happen
        let apart = Embedding { chains: vec![vec![0], vec![3], vec![12]], sites: g.n };
        let e = apart.verify(&triangle(), &g).unwrap_err();
        assert!(e.contains("interact"), "{e}");
    }

    #[test]
    fn a_clique_embeds_into_a_kings_graph_with_chains() {
        // Six variables all interacting, on a machine where a site has eight neighbours. This is
        // the case the Hitachi driver refused outright rather than placing.
        let k = king(8);
        let c = clique(6, 1.0);
        let e = embed(&c, &k, 3).expect("K6 is a minor of an 8x8 King's graph");
        e.verify(&c, &k).unwrap();
        assert!(e.used() >= 6);
    }

    #[test]
    fn more_variables_than_sites_is_refused_immediately() {
        assert_eq!(embed(&clique(10, 1.0), &grid(3), 1), None, "9 sites cannot hold 10 variables");
    }

    #[test]
    fn an_embedded_program_has_the_same_ground_state_as_the_one_it_came_from() {
        // The property that makes embedding worth anything: solve the placed model, read it back,
        // and hold an optimum of the model that was written. Checked exhaustively on the logical
        // side and by exact elimination on the hardware side.
        let mut b = GraphBuilder::new(4);
        b.couple(0, 1, -1.0);
        b.couple(1, 2, -1.0);
        b.couple(2, 3, -1.0);
        b.couple(0, 3, -1.0);
        b.couple(0, 2, 1.0);
        b.set_bias(0, 0.3);
        b.set_bias(3, -0.2);
        let logical = b.build();

        let hw = grid(5);
        let e = embed(&logical, &hw, 11).expect("a small graph fits a 5x5 grid");
        e.verify(&logical, &hw).unwrap();
        let emb = apply(&logical, &hw, &e);

        let best = crate::exact::Elimination::default()
            .ground_state(&emb.graph)
            .expect("elimination")
            .ground_state
            .expect("a ground state");
        let (values, broken) = unembed(&e, &best);
        assert!(broken.is_empty(), "the chain coupling should hold at the optimum: {broken:?}");

        let want = (0u32..(1 << logical.n))
            .map(|m| {
                let s: Vec<i8> =
                    (0..logical.n).map(|i| if m & (1 << i) != 0 { 1 } else { -1 }).collect();
                (logical.energy(&s) * 1e9) as i64
            })
            .min()
            .unwrap();
        assert_eq!(
            (logical.energy(&values) * 1e9) as i64,
            want,
            "unembedded {values:?} must minimise the logical model"
        );
    }

    #[test]
    fn a_broken_chain_is_reported_rather_than_resolved_silently() {
        // A chain that disagrees with itself has no value for that variable. Majority vote gives an
        // answer; saying which variables needed one is what stops it being mistaken for a result.
        let e = Embedding { chains: vec![vec![0, 1, 2], vec![3]], sites: 4 };
        let (v, broken) = unembed(&e, &[1, 1, -1, 1]);
        assert_eq!(v, vec![1, 1], "majority is up");
        assert_eq!(broken, vec![0], "and variable 0 is the one that disagreed");

        let (_, none) = unembed(&e, &[1, 1, 1, -1]);
        assert!(none.is_empty(), "an agreeing chain is not reported");
    }

    /// A star with eight leaves is the simplest graph that cannot sit on a degree-6 machine
    /// without exactly one chain: the centre has degree 8 and a Chimera site has six neighbours.
    ///
    /// It is embedded, on every seed, and the centre really does become a chain rather than the
    /// placer finding some way around the arithmetic. About ten sites, which is what it should
    /// cost. This test spent a release asserting the opposite, because the placer could not build
    /// a chain to relieve a neighbour with nowhere left to sit; it is pinned in this direction now
    /// so that regression is loud.
    #[test]
    fn a_star_that_needs_one_chain_is_placed() {
        let hardware = crate::ising::chimera(8, 8, 4, 1.0);
        let mut b = GraphBuilder::new(9);
        for i in 1..9 {
            b.couple(0, i, 1.0);
        }
        let star = b.build();
        assert!(hardware.n > 50 * star.n, "the machine is nowhere near full");
        for seed in 0..8 {
            let e = embed(&star, &hardware, seed)
                .unwrap_or_else(|| panic!("the star must embed, seed {seed}"));
            e.verify(&star, &hardware).expect("and the embedding must be one");
            assert!(e.longest_chain() >= 2, "the centre must be a chain: {:?}", e.chains);
            assert!(e.used() <= 20, "about ten sites, not a sprawl: {:?}", e.chains);
        }
        // And it is not a matter of budget either way: one round finds it.
        let e = embed_with(&star, &hardware, 0, 0).expect("the first round already places it");
        e.verify(&star, &hardware).unwrap();
    }

    /// Cliques past the point where a chain becomes necessary.
    ///
    /// `K_7` is the largest clique a degree-6 site can hold with one site per variable, so `K_8` is
    /// the first that forces the placer to grow a chain to relieve a variable with nowhere left to
    /// sit. That used to be a cliff — nothing past it embedded at any seed, machine size or round
    /// budget. The interesting number now is how far past it the placer reaches, and on a 512-site
    /// Chimera that is at least `K_16`.
    #[test]
    fn cliques_past_the_first_one_needing_a_chain_are_placed() {
        let hardware = crate::ising::chimera(8, 8, 4, 1.0);
        for n in [6usize, 8, 12] {
            let c = clique(n, 1.0);
            for seed in 0..4 {
                let e = embed(&c, &hardware, seed)
                    .unwrap_or_else(|| panic!("K_{n} must embed, seed {seed}"));
                e.verify(&c, &hardware).expect("and the embedding must be one");
                assert!(e.longest_chain() > 1, "K_{n} cannot fit without a chain: {:?}", e.chains);
            }
        }
        // Stated as "at least this many of these seeds" on purpose. This is a heuristic, and a
        // claim that it always succeeds is one it cannot make. Measured here: 8 of 8.
        let c = clique(16, 1.0);
        let ok = (0..8u64)
            .filter(|&s| {
                embed(&c, &hardware, s).map(|e| e.verify(&c, &hardware).is_ok()).unwrap_or(false)
            })
            .count();
        assert!(ok >= 6, "K_16 embedded on only {ok} of 8 seeds");
    }

    /// THE BOUND MUST NEVER EXCEED A REAL EMBEDDING, because `embed_with` refuses outright when it
    /// exceeds the machine. A bound that overshot by one site would turn a solvable program into a
    /// permanent "impossible" with no way to tell from the outside -- the worst failure available
    /// to this module, and the reason the counting argument is checked against actual embeddings
    /// rather than only against itself.
    #[test]
    fn the_site_lower_bound_never_exceeds_an_embedding_it_admits() {
        let machines = [
            crate::ising::chimera(8, 8, 4, 1.0),
            crate::ising::lattice2d(12, 1.0),
            crate::ising::grid2d(10, 8, 1.0),
        ];
        let clique = |n: usize| {
            let mut b = GraphBuilder::new(n);
            for i in 0..n {
                for j in (i + 1)..n {
                    b.couple(i, j, 1.0);
                }
            }
            b.build()
        };
        let star = |leaves: usize| {
            let mut b = GraphBuilder::new(leaves + 1);
            for i in 1..=leaves {
                b.couple(0, i, 1.0);
            }
            b.build()
        };
        let path = |n: usize| {
            let mut b = GraphBuilder::new(n);
            for i in 0..n - 1 {
                b.couple(i, i + 1, 1.0);
            }
            b.build()
        };

        let mut checked = 0;
        for hw in &machines {
            for g in [clique(4), clique(6), clique(8), clique(12), star(3), star(8), star(20),
                      path(5), path(40)] {
                let lb = site_lower_bound(&g, hw);
                for seed in 0..4u64 {
                    if let Some(e) = embed(&g, hw, seed) {
                        e.verify(&g, hw).expect("embed only returns verified embeddings");
                        assert!(
                            lb <= e.used(),
                            "the bound claims {lb} sites are needed and an embedding used {} \
                             -- the bound is UNSOUND and embed_with refuses on it",
                            e.used()
                        );
                        checked += 1;
                    }
                }
            }
        }
        // A floor: if nothing embedded, the loop above asserted nothing.
        assert!(checked > 30, "only {checked} embeddings to check the bound against");
    }

    /// And it has to actually bite, or it is decoration on the hot path.
    #[test]
    fn the_site_lower_bound_refuses_what_cannot_fit() {
        let hw = crate::ising::chimera(8, 8, 4, 1.0);
        let clique = |n: usize| {
            let mut b = GraphBuilder::new(n);
            for i in 0..n {
                for j in (i + 1)..n {
                    b.couple(i, j, 1.0);
                }
            }
            b.build()
        };
        // Chimera has max degree 6, so a chain of L sites offers at most 4L + 2 ports. K_60 needs
        // chains of 15 and 60 of them: 900 sites against 512.
        assert_eq!(site_lower_bound(&clique(60), &hw), 900);
        assert!(embed(&clique(60), &hw, 0).is_none());
        assert!(site_lower_bound(&clique(100), &hw) > hw.n);
        // And it must NOT refuse what the argument cannot rule out: K_24 embeds, K_33 is inside
        // the bound and is searched for properly rather than dismissed.
        assert!(site_lower_bound(&clique(24), &hw) <= hw.n);
        assert!(site_lower_bound(&clique(33), &hw) <= hw.n);
        assert!(embed(&clique(24), &hw, 0).is_some());
        // Degenerate machines: degree <= 2 makes the argument vacuous, and it says so rather than
        // dividing by zero.
        let ring = crate::ising::ring(16, 1.0, 0.0);
        assert_eq!(site_lower_bound(&clique(5), &ring), 5);
    }

    #[test]
    fn chain_strength_outweighs_the_model_it_holds_together() {
        let mut b = GraphBuilder::new(3);
        b.couple(0, 1, 5.0);
        b.couple(1, 2, 5.0);
        b.couple(0, 2, 5.0);
        let logical = b.build();
        let hw = grid(4);
        let e = embed(&logical, &hw, 5).unwrap();
        let emb = apply(&logical, &hw, &e);
        assert!(emb.chain_strength > 5.0, "a chain must outrank the couplings it carries: {}", emb.chain_strength);
    }

    #[test]
    fn embedding_is_reproducible_from_its_seed() {
        let c = clique(5, 1.0);
        let k = king(6);
        let a = embed(&c, &k, 42).unwrap();
        let b = embed(&c, &k, 42).unwrap();
        assert_eq!(a, b, "same seed, same placement");
    }

    #[test]
    fn a_generous_machine_makes_it_easy() {
        // Sanity: into a complete graph, everything fits with chains of one.
        let c = clique(8, 1.0);
        let e = embed(&c, &complete(8), 1).expect("K8 into K8");
        assert_eq!(e.longest_chain(), 1);
        e.verify(&c, &complete(8)).unwrap();
    }
}
