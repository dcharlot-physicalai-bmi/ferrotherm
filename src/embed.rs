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
//! exists. [`embed`] returns `None`, which means "not found", never "impossible". It is weakest
//! exactly where the machine is fullest — a packing that uses every site leaves the rip-up rounds
//! nowhere to move anything — so a program already laid out for the hardware is checked for first
//! and returned as itself.

use crate::graph::{Graph, GraphBuilder};
use std::collections::{BTreeMap, BTreeSet, VecDeque};

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

/// As [`embed`], with an explicit number of rip-up rounds.
pub fn embed_with(logical: &Graph, hardware: &Graph, seed: u64, rounds: usize) -> Option<Embedding> {
    if logical.n == 0 {
        return Some(Embedding { chains: Vec::new(), sites: hardware.n });
    }
    if logical.n > hardware.n {
        return None; // more variables than sites, before any structure is considered
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
    // means fitting them into whatever is left.
    let mut order: Vec<usize> = (0..logical.n).collect();
    order.sort_by_key(|&v| core::cmp::Reverse(logical.offset[v + 1] - logical.offset[v]));

    let mut chains: Vec<Vec<usize>> = vec![Vec::new(); logical.n];

    for round in 0..=rounds {
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

            let chain = if neighbours.is_empty() {
                // Nothing to be near yet: take the emptiest site, breaking ties randomly so that a
                // second round explores somewhere new rather than repeating itself.
                let start = (0..hardware.n)
                    .min_by_key(|&s| (load[s], rng.next() % 1024))
                    .expect("a machine has sites");
                vec![start]
            } else {
                steiner_ish(hardware, &neighbours, &chains, &load, &mut rng)?
            };
            chains[v] = chain;
        }

        let overlapped = {
            let mut load = vec![0usize; hardware.n];
            for c in &chains {
                for &s in c {
                    load[s] += 1;
                }
            }
            load.iter().any(|&n| n > 1)
        };
        if !overlapped {
            let e = Embedding { chains, sites: hardware.n };
            // An embedding this module returns is always checked. A wrong one does not fail
            // loudly on a machine; it returns plausible answers to a different problem.
            e.verify(logical, hardware).ok()?;
            return Some(e);
        }
        if round == rounds {
            break;
        }
    }
    None
}

/// The cheapest connected set of sites touching every neighbour's chain.
///
/// Grown by running a Dijkstra outward from each neighbour's chain and taking the site whose total
/// distance is least — the standard heuristic for a Steiner tree, which is what a chain is. Cost
/// rises with how many other variables already occupy a site, so an overlap is possible but
/// expensive, and the rip-up rounds get a chance to resolve it.
fn steiner_ish(
    h: &Graph,
    neighbours: &[usize],
    chains: &[Vec<usize>],
    load: &[usize],
    rng: &mut Pcg,
) -> Option<Vec<usize>> {
    const OCCUPIED: u64 = 64; // a site in use costs this much more than an empty one

    let mut total = vec![0u64; h.n];
    let mut parents: Vec<Vec<usize>> = Vec::with_capacity(neighbours.len());

    for &u in neighbours {
        let (dist, parent) = dijkstra(h, &chains[u], load, OCCUPIED);
        for s in 0..h.n {
            total[s] = total[s].saturating_add(dist[s]);
        }
        parents.push(parent);
    }

    // The root cannot be a site belonging to a neighbour we are routing TO: this variable needs
    // sites of its own, and choosing one of theirs leaves nothing after the subtraction below --
    // which returned None and failed embeddings that plainly exist, like a triangle in a grid.
    let theirs: BTreeSet<usize> =
        neighbours.iter().flat_map(|&u| chains[u].iter().copied()).collect();

    let root = (0..h.n)
        .filter(|s| !theirs.contains(s))
        .filter(|&s| total[s] < u64::MAX / 4)
        .min_by_key(|&s| (total[s], rng.next() % 8))?;

    // Walk back from the root toward each neighbour, collecting the path.
    let mut chain = BTreeSet::new();
    chain.insert(root);
    for parent in &parents {
        let mut at = root;
        while parent[at] != usize::MAX && parent[at] != at {
            at = parent[at];
            chain.insert(at);
        }
    }
    // The paths end ON the neighbours' chains; those sites belong to them, not to us.
    let mine: Vec<usize> = chain.difference(&theirs).copied().collect();
    if mine.is_empty() { None } else { Some(mine) }
}

/// Shortest paths from a set of sources, with a per-site cost.
fn dijkstra(h: &Graph, sources: &[usize], load: &[usize], occupied: u64) -> (Vec<u64>, Vec<usize>) {
    let mut dist = vec![u64::MAX; h.n];
    let mut parent = vec![usize::MAX; h.n];
    // A binary heap by hand: this crate has no dependencies, and a Vec-backed heap over sites is
    // not the bottleneck next to the rip-up rounds around it.
    let mut heap: Vec<(u64, usize)> = Vec::new();
    for &s in sources {
        dist[s] = 0;
        parent[s] = s;
        heap.push((0, s));
    }
    while let Some(pos) = heap
        .iter()
        .enumerate()
        .min_by_key(|(_, (d, _))| *d)
        .map(|(i, _)| i)
    {
        let (d, u) = heap.swap_remove(pos);
        if d > dist[u] {
            continue;
        }
        for k in h.offset[u]..h.offset[u + 1] {
            let v = h.nbr[k] as usize;
            let step = 1 + load[v] as u64 * occupied;
            let nd = d.saturating_add(step);
            if nd < dist[v] {
                dist[v] = nd;
                parent[v] = u;
                heap.push((nd, v));
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
    let worst = (0..logical.n)
        .flat_map(|i| (logical.offset[i]..logical.offset[i + 1]).map(move |k| logical.w[k].abs()))
        .chain(logical.h.iter().map(|x| x.abs()))
        .fold(0.0f64, f64::max);
    let chain_strength = if worst > 0.0 { 2.0 * worst } else { 1.0 };

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
