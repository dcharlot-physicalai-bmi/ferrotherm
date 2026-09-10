//! Working graphs: the machine on the floor, not the one in the datasheet.
//!
//! Every annealer ships with dead qubits and dead couplers, so an embedding computed against the
//! ideal lattice can name a qubit the machine does not have — and it fails at submission time, or
//! worse, quietly programs a coupler that carries nothing. [`Working`] deletes the defects from an
//! ideal [`Topology`] and hands [`crate::embed`] the graph that is left.
//!
//! Defects are stated in the **machine's own qubit numbering** ([`Topology::qubits`]), because that
//! is the only numbering deleting a qubit does not change: dense indices renumber the moment
//! anything is removed. [`Working::audit_qubits`] checks a placement in that numbering against the
//! live fabric alone — it never consults the working graph — so it is an independent answer to
//! "does this touch anything dead" rather than a restatement of how the working graph was built.

use crate::device::Topology;
use crate::embed::{Embedded, Embedding};
use crate::graph::{Graph, GraphBuilder};
use crate::rng::Pcg;
use std::collections::{BTreeMap, BTreeSet, VecDeque};

/// A coupler as an unordered pair, so `(a, b)` and `(b, a)` are one key.
fn key(a: u32, b: u32) -> (u32, u32) {
    if a <= b { (a, b) } else { (b, a) }
}

/// Wrap a plain hardware graph as a topology numbered `0..n`.
///
/// [`crate::embed::topology`] and [`crate::ising::chimera`] return bare graphs with no vendor
/// numbering; this gives them the identity one so they can be defect-mapped like a real machine.
#[must_use]
pub fn identity_numbered(graph: Graph) -> Topology {
    let n = graph.n as u32;
    Topology { graph, qubits: (0..n).collect() }
}

/// The qubits and couplers a machine has lost, in the machine's own numbering.
///
/// A coupler may be listed dead even though one of its qubits is dead too; that is redundant, not
/// contradictory, and [`Defects::sample`] produces it on purpose.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Defects {
    qubits: BTreeSet<u32>,
    couplers: BTreeSet<(u32, u32)>,
}

impl Defects {
    /// A machine with nothing wrong with it.
    #[must_use]
    pub fn none() -> Defects {
        Defects::default()
    }

    /// The named qubits and couplers, dead.
    #[must_use]
    pub fn from_lists(qubits: &[u32], couplers: &[(u32, u32)]) -> Defects {
        Defects {
            qubits: qubits.iter().copied().collect(),
            couplers: couplers.iter().map(|&(a, b)| key(a, b)).collect(),
        }
    }

    /// Mark one qubit dead.
    pub fn kill_qubit(&mut self, q: u32) {
        self.qubits.insert(q);
    }

    /// Mark one coupler dead. Order of the endpoints does not matter.
    pub fn kill_coupler(&mut self, a: u32, b: u32) {
        self.couplers.insert(key(a, b));
    }

    /// Is this qubit dead?
    #[must_use]
    pub fn qubit_dead(&self, q: u32) -> bool {
        self.qubits.contains(&q)
    }

    /// Is this coupler dead? True also when either endpoint is dead — a coupler to nowhere.
    #[must_use]
    pub fn coupler_dead(&self, a: u32, b: u32) -> bool {
        self.couplers.contains(&key(a, b)) || self.qubits.contains(&a) || self.qubits.contains(&b)
    }

    /// The dead qubits, ascending.
    #[must_use]
    pub fn dead_qubits(&self) -> &BTreeSet<u32> {
        &self.qubits
    }

    /// The couplers explicitly marked dead, ascending. Excludes ones dead only by a dead endpoint.
    #[must_use]
    pub fn dead_couplers(&self) -> &BTreeSet<(u32, u32)> {
        &self.couplers
    }

    /// Draw a defect map at a yield rate: each qubit survives with probability `qubit_yield`, each
    /// coupler with `coupler_yield`. Rates outside `[0, 1]` are clamped; a non-finite rate is 1.
    ///
    /// One draw per qubit and one per coupler, in the machine's index order, from streams that do
    /// not depend on the rates — so lowering a yield can only ever kill more, never something else.
    ///
    /// # Panics
    ///
    /// If `ideal` is malformed: its qubit list must have one entry per node.
    #[must_use]
    pub fn sample(ideal: &Topology, qubit_yield: f64, coupler_yield: f64, seed: u64) -> Defects {
        assert_eq!(ideal.qubits.len(), ideal.graph.n, "topology qubit list does not match its graph");
        let clamp = |p: f64| if p.is_finite() { p.clamp(0.0, 1.0) } else { 1.0 };
        let (pq, pc) = (clamp(qubit_yield), clamp(coupler_yield));

        let mut rq = Pcg::new(seed, 0x00DE_AD00);
        let qubits: BTreeSet<u32> =
            ideal.qubits.iter().copied().filter(|_| rq.f64() >= pq).collect();

        let mut rc = Pcg::new(seed, 0x00DE_ADC0);
        let couplers: BTreeSet<(u32, u32)> =
            vendor_edges(ideal).into_iter().filter(|_| rc.f64() >= pc).collect();

        Defects { qubits, couplers }
    }
}

/// Every coupler of a topology as a sorted set of vendor pairs.
fn vendor_edges(t: &Topology) -> BTreeSet<(u32, u32)> {
    let g = &t.graph;
    let mut out = BTreeSet::new();
    for i in 0..g.n {
        for k in g.offset[i]..g.offset[i + 1] {
            let j = g.nbr[k] as usize;
            if j > i {
                out.insert(key(t.qubits[i], t.qubits[j]));
            }
        }
    }
    out
}

/// How much of the ideal machine survived.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Yield {
    /// Live qubits.
    pub qubits: usize,
    /// Qubits the ideal topology has.
    pub qubits_ideal: usize,
    /// Live couplers.
    pub couplers: usize,
    /// Couplers the ideal topology has.
    pub couplers_ideal: usize,
}

impl Yield {
    /// Fraction of qubits alive; `1.0` for an empty machine.
    #[must_use]
    pub fn qubit_rate(&self) -> f64 {
        if self.qubits_ideal == 0 { 1.0 } else { self.qubits as f64 / self.qubits_ideal as f64 }
    }

    /// Fraction of couplers alive; `1.0` for a machine with none.
    #[must_use]
    pub fn coupler_rate(&self) -> f64 {
        if self.couplers_ideal == 0 { 1.0 } else { self.couplers as f64 / self.couplers_ideal as f64 }
    }
}

/// An ideal topology with its defects removed — the graph an embedder must actually target.
pub struct Working {
    /// The working graph in fresh dense indices, keeping the machine's own qubit numbering.
    pub topology: Topology,
    /// What is dead, in the machine's numbering.
    pub defects: Defects,
    /// The ideal machine's qubits, ascending. A placement may not name anything outside this.
    ideal_qubits: Vec<u32>,
    /// The ideal machine's couplers as vendor pairs. A placement may not use anything outside this.
    ideal_couplers: BTreeSet<(u32, u32)>,
}

impl core::fmt::Debug for Working {
    /// The shape, since [`Graph`] has no `Debug` and the index arithmetic would be pages.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let y = self.stats();
        write!(
            f,
            "Working {{ {}/{} qubits, {}/{} couplers }}",
            y.qubits, y.qubits_ideal, y.couplers, y.couplers_ideal
        )
    }
}

impl Working {
    /// Delete `defects` from `ideal` and renumber what is left.
    ///
    /// Live qubits with no live coupler are **kept** as isolated nodes, matching what a vendor
    /// reports as a working qubit; only [`crate::embed`] decides they are useless.
    ///
    /// # Panics
    ///
    /// If `ideal` is malformed: its qubit list must have one entry per node.
    #[must_use]
    pub fn new(ideal: &Topology, defects: &Defects) -> Working {
        let g = &ideal.graph;
        assert_eq!(ideal.qubits.len(), g.n, "topology qubit list does not match its graph");
        let live: Vec<u32> =
            ideal.qubits.iter().copied().filter(|q| !defects.qubits.contains(q)).collect();

        let mut b = GraphBuilder::new(live.len());
        let mut ideal_couplers = BTreeSet::new();
        for i in 0..g.n {
            let qi = ideal.qubits[i];
            for k in g.offset[i]..g.offset[i + 1] {
                let j = g.nbr[k] as usize;
                if j <= i {
                    continue;
                }
                let qj = ideal.qubits[j];
                ideal_couplers.insert(key(qi, qj));
                if defects.coupler_dead(qi, qj) {
                    continue;
                }
                if let (Ok(a), Ok(c)) = (live.binary_search(&qi), live.binary_search(&qj)) {
                    b.couple(a, c, g.w[k]);
                }
            }
        }
        for i in 0..g.n {
            if g.h[i] != 0.0
                && let Ok(d) = live.binary_search(&ideal.qubits[i])
            {
                b.bias(d, g.h[i]);
            }
        }

        Working {
            topology: Topology { graph: b.build(), qubits: live },
            defects: defects.clone(),
            ideal_qubits: ideal.qubits.clone(),
            ideal_couplers,
        }
    }

    /// The working graph, in dense indices. This is what to pass an embedder.
    #[must_use]
    pub fn graph(&self) -> &Graph {
        &self.topology.graph
    }

    /// The machine's qubit index for one of our nodes.
    #[must_use]
    pub fn qubit(&self, node: usize) -> Option<u32> {
        self.topology.qubit(node)
    }

    /// Our node index for one of the machine's qubits, or `None` if that qubit is dead or absent.
    #[must_use]
    pub fn node(&self, qubit: u32) -> Option<usize> {
        self.topology.node(qubit)
    }

    /// Live and ideal counts.
    #[must_use]
    pub fn stats(&self) -> Yield {
        Yield {
            qubits: self.topology.graph.n,
            qubits_ideal: self.ideal_qubits.len(),
            couplers: self.topology.graph.n_edges,
            couplers_ideal: self.ideal_couplers.len(),
        }
    }

    /// Does the ideal machine have this coupler, and is it and both its qubits still alive?
    #[must_use]
    pub fn coupler_live(&self, a: u32, b: u32) -> bool {
        self.ideal_couplers.contains(&key(a, b)) && !self.defects.coupler_dead(a, b)
    }

    /// Does the ideal machine have this qubit, and is it still alive?
    #[must_use]
    pub fn qubit_live(&self, q: u32) -> bool {
        self.ideal_qubits.binary_search(&q).is_ok() && !self.defects.qubit_dead(q)
    }

    /// Rewrite an embedding's chains in the machine's own qubit numbering.
    ///
    /// # Errors
    ///
    /// If a chain names a site past the end of the working graph.
    pub fn chain_qubits(&self, e: &Embedding) -> Result<Vec<Vec<u32>>, String> {
        let mut out = Vec::with_capacity(e.chains.len());
        for (v, chain) in e.chains.iter().enumerate() {
            let mut c = Vec::with_capacity(chain.len());
            for &s in chain {
                match self.qubit(s) {
                    Some(q) => c.push(q),
                    None => {
                        return Err(format!(
                            "variable {v} uses site {s}, past the {} this machine has",
                            self.topology.graph.n
                        ));
                    }
                }
            }
            out.push(c);
        }
        Ok(out)
    }

    /// The couplers a run would program, as machine qubit pairs: every live coupler inside a chain,
    /// plus every live coupler realising a logical edge. Sorted, deduplicated.
    ///
    /// Chains are in machine numbering — see [`Working::chain_qubits`].
    #[must_use]
    pub fn couplers_used(&self, logical: &Graph, chains: &[Vec<u32>]) -> Vec<(u32, u32)> {
        let mut out = BTreeSet::new();
        for chain in chains {
            for a in 0..chain.len() {
                for b in (a + 1)..chain.len() {
                    if self.coupler_live(chain[a], chain[b]) {
                        out.insert(key(chain[a], chain[b]));
                    }
                }
            }
        }
        for i in 0..logical.n.min(chains.len()) {
            for k in logical.offset[i]..logical.offset[i + 1] {
                let j = logical.nbr[k] as usize;
                if j <= i || j >= chains.len() {
                    continue;
                }
                for &u in &chains[i] {
                    for &v in &chains[j] {
                        if self.coupler_live(u, v) {
                            out.insert(key(u, v));
                        }
                    }
                }
            }
        }
        out.into_iter().collect()
    }

    /// Check chains written in the machine's own qubit numbering against the live fabric.
    ///
    /// Consults only the ideal coupler list and the defect map, never the working graph, so it is
    /// an independent verdict on "does this touch anything dead".
    ///
    /// # Errors
    ///
    /// A message naming the first defect: a wrong number of chains, an empty chain, a qubit the
    /// machine does not have, a dead qubit, a qubit used twice, a chain no live coupler holds
    /// together, or a logical edge with no live coupler between its chains.
    pub fn audit_qubits(&self, logical: &Graph, chains: &[Vec<u32>]) -> Result<(), String> {
        if chains.len() != logical.n {
            return Err(format!("{} chains for {} variables", chains.len(), logical.n));
        }
        let mut seen: BTreeMap<u32, usize> = BTreeMap::new();
        for (v, chain) in chains.iter().enumerate() {
            if chain.is_empty() {
                return Err(format!("variable {v} has no qubits"));
            }
            for &q in chain {
                if self.ideal_qubits.binary_search(&q).is_err() {
                    return Err(format!("variable {v} uses qubit {q}, which this machine never had"));
                }
                if self.defects.qubit_dead(q) {
                    return Err(format!("variable {v} uses qubit {q}, which is dead"));
                }
                if let Some(&o) = seen.get(&q) {
                    return Err(format!("qubit {q} is used by both {o} and {v}"));
                }
                seen.insert(q, v);
            }
            if !self.held_together(chain) {
                return Err(format!(
                    "variable {v}'s chain {chain:?} is not connected by live couplers"
                ));
            }
        }
        for i in 0..logical.n {
            for k in logical.offset[i]..logical.offset[i + 1] {
                let j = logical.nbr[k] as usize;
                if j <= i {
                    continue;
                }
                let joined = chains[i]
                    .iter()
                    .any(|&u| chains[j].iter().any(|&v| self.coupler_live(u, v)));
                if !joined {
                    return Err(format!(
                        "variables {i} and {j} interact and no live coupler joins their chains"
                    ));
                }
            }
        }
        Ok(())
    }

    /// Both checks an embedding onto a defective machine has to pass: nothing dead is touched, and
    /// it is still an embedding of `logical` into the working graph.
    ///
    /// # Errors
    ///
    /// The first failure, from [`Working::audit_qubits`] or [`Embedding::verify`].
    pub fn audit(&self, logical: &Graph, e: &Embedding) -> Result<(), String> {
        let chains = self.chain_qubits(e)?;
        self.audit_qubits(logical, &chains)?;
        e.verify(logical, self.graph())
    }

    /// Embed onto the working graph, never onto the ideal one.
    ///
    /// # Panics
    ///
    /// If what it found touches a dead qubit or coupler, or fails [`Embedding::verify`] against the
    /// working graph. Either is a bug here rather than a caller error.
    #[must_use]
    pub fn embed(&self, logical: &Graph, seed: u64) -> Option<Embedding> {
        self.embed_with(logical, seed, 20)
    }

    /// As [`Working::embed`], with an explicit number of rip-up rounds.
    ///
    /// # Panics
    ///
    /// If what it found touches a dead qubit or coupler, or fails [`Embedding::verify`].
    #[must_use]
    pub fn embed_with(&self, logical: &Graph, seed: u64, rounds: usize) -> Option<Embedding> {
        let e = crate::embed::embed_with(logical, self.graph(), seed, rounds)?;
        if let Err(why) = self.audit(logical, &e) {
            panic!("embedded onto a defective machine and it does not hold: {why}");
        }
        Some(e)
    }

    /// Rewrite a logical model onto the working graph's sites under an embedding.
    #[must_use]
    pub fn apply(&self, logical: &Graph, e: &Embedding) -> Embedded {
        crate::embed::apply(logical, self.graph(), e)
    }

    /// Is every qubit of this chain reachable from the first by live couplers?
    fn held_together(&self, chain: &[u32]) -> bool {
        if chain.len() <= 1 {
            return true;
        }
        let mut seen = vec![false; chain.len()];
        seen[0] = true;
        let mut q = VecDeque::from([0usize]);
        let mut count = 1;
        while let Some(a) = q.pop_front() {
            for b in 0..chain.len() {
                if !seen[b] && self.coupler_live(chain[a], chain[b]) {
                    seen[b] = true;
                    count += 1;
                    q.push_back(b);
                }
            }
        }
        count == chain.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embed::topology::{grid, king};
    use crate::embed::{chimera_clique, topology::complete};
    use crate::ising::chimera;

    fn triangle() -> Graph {
        let mut b = GraphBuilder::new(3);
        b.couple(0, 1, 1.0);
        b.couple(1, 2, 1.0);
        b.couple(0, 2, 1.0);
        b.build()
    }

    /// Every coupler of a graph as a sorted set of dense pairs, recomputed by hand.
    fn edge_set(g: &Graph) -> BTreeSet<(u32, u32)> {
        let mut out = BTreeSet::new();
        for i in 0..g.n {
            for k in g.offset[i]..g.offset[i + 1] {
                let j = g.nbr[k] as usize;
                if j > i {
                    out.insert(key(i as u32, j as u32));
                }
            }
        }
        out
    }

    /// With nothing dead, the working graph is the ideal one: same nodes, same numbering, same
    /// couplers. An identity, so it is checked as one.
    #[test]
    fn no_defects_leaves_the_machine_alone() {
        let ideal = identity_numbered(king(5));
        let want_edges = edge_set(&ideal.graph);
        let w = Working::new(&ideal, &Defects::none());

        assert_eq!(w.graph().n, ideal.graph.n);
        assert_eq!(w.graph().n_edges, ideal.graph.n_edges);
        assert_eq!(w.topology.qubits, ideal.qubits);
        assert_eq!(edge_set(w.graph()), want_edges);
        assert_eq!(w.stats().qubit_rate(), 1.0);
        assert_eq!(w.stats().coupler_rate(), 1.0);
    }

    /// Node and coupler counts are the ideal ones minus exactly what died. The oracle is the ideal
    /// edge list filtered by hand, not the builder's own arithmetic.
    #[test]
    fn counts_are_the_ideal_ones_minus_what_died() {
        let ideal = identity_numbered(chimera(3, 3, 4, 1.0));
        let dead_q = [0u32, 7, 40, 71];
        let dead_c = [(1u32, 5u32), (8, 9), (2, 6), (0, 4)];
        let d = Defects::from_lists(&dead_q, &dead_c);
        let w = Working::new(&ideal, &d);

        let want_nodes = ideal.graph.n - dead_q.len();
        let want_edges = edge_set(&ideal.graph)
            .iter()
            .filter(|&&(a, b)| {
                !dead_q.contains(&a) && !dead_q.contains(&b) && !dead_c.contains(&key(a, b))
            })
            .count();
        // (0,4) is a real chimera coupler and 0 is also a dead qubit; both filters agree on it.
        assert_eq!(w.graph().n, want_nodes);
        assert_eq!(w.graph().n_edges, want_edges);
        assert!(want_edges < ideal.graph.n_edges, "the test must actually remove something");
    }

    /// The working graph is exactly the live-induced subgraph minus the dead couplers — both
    /// directions, over every edge, on a machine with a real vendor numbering.
    #[test]
    fn the_working_graph_is_exactly_what_survived() {
        let ideal = crate::device::pegasus(3, 1.0);
        let d = Defects::sample(&ideal, 0.9, 0.95, 7);
        let w = Working::new(&ideal, &d);
        let ideal_edges = vendor_edges(&ideal);

        // Nothing extra: every working coupler is an ideal one, alive, between live qubits.
        for (a, b) in edge_set(w.graph()) {
            let (qa, qb) = (w.qubit(a as usize).unwrap(), w.qubit(b as usize).unwrap());
            assert!(ideal_edges.contains(&key(qa, qb)), "coupler {qa}-{qb} is not on the machine");
            assert!(!d.coupler_dead(qa, qb), "coupler {qa}-{qb} is dead");
            assert!(!d.qubit_dead(qa) && !d.qubit_dead(qb));
        }
        // Nothing missing: every surviving ideal coupler is in the working graph.
        let live_pairs: BTreeSet<(u32, u32)> = edge_set(w.graph())
            .iter()
            .map(|&(a, b)| key(w.qubit(a as usize).unwrap(), w.qubit(b as usize).unwrap()))
            .collect();
        for &(qa, qb) in &ideal_edges {
            if !d.coupler_dead(qa, qb) {
                assert!(live_pairs.contains(&key(qa, qb)), "coupler {qa}-{qb} was lost");
            }
        }
        assert!(!d.dead_qubits().is_empty(), "the draw must actually kill something");
    }

    /// Renumbering preserves adjacency: every node's neighbour set, read back in machine numbering,
    /// is the ideal neighbour set minus what died. Exhaustive over the machine.
    #[test]
    fn renumbering_preserves_every_neighbourhood() {
        let ideal = crate::device::pegasus(3, 1.0);
        let d = Defects::sample(&ideal, 0.92, 0.9, 3);
        let w = Working::new(&ideal, &d);

        for node in 0..w.graph().n {
            let q = w.qubit(node).unwrap();
            let i = ideal.node(q).expect("a live qubit is an ideal qubit");
            let want: BTreeSet<u32> = (ideal.graph.offset[i]..ideal.graph.offset[i + 1])
                .map(|k| ideal.qubits[ideal.graph.nbr[k] as usize])
                .filter(|&nq| !d.coupler_dead(q, nq))
                .collect();
            let got: BTreeSet<u32> = (w.graph().offset[node]..w.graph().offset[node + 1])
                .map(|k| w.qubit(w.graph().nbr[k] as usize).unwrap())
                .collect();
            assert_eq!(got, want, "qubit {q}");
        }
    }

    /// Full yield kills nothing and zero yield kills everything. Both are exact, because
    /// `Pcg::f64` is in `[0, 1)`: never `>= 1.0`, always `>= 0.0`.
    #[test]
    fn the_two_ends_of_the_yield_range_are_exact() {
        let ideal = identity_numbered(king(4));
        let full = Defects::sample(&ideal, 1.0, 1.0, 11);
        assert_eq!(full, Defects::none());
        assert_eq!(Working::new(&ideal, &full).graph().n, ideal.graph.n);

        let nothing = Defects::sample(&ideal, 0.0, 0.0, 11);
        assert_eq!(nothing.dead_qubits().len(), ideal.graph.n);
        assert_eq!(nothing.dead_couplers().len(), ideal.graph.n_edges);
        let w = Working::new(&ideal, &nothing);
        assert_eq!(w.graph().n, 0);
        assert_eq!(w.graph().n_edges, 0);
        assert!(w.embed(&triangle(), 1).is_none(), "no machine, no embedding");
    }

    /// Lowering a yield can only kill more. One draw per item from a rate-independent stream makes
    /// the dead sets nest, and a generator that reordered or skipped draws would break it.
    #[test]
    fn a_lower_yield_kills_a_superset() {
        let ideal = identity_numbered(chimera(2, 2, 4, 1.0));
        let rates = [0.99, 0.95, 0.9, 0.7, 0.4, 0.1];
        for pair in rates.windows(2) {
            let (hi, lo) = (pair[0], pair[1]);
            let a = Defects::sample(&ideal, hi, hi, 5);
            let b = Defects::sample(&ideal, lo, lo, 5);
            assert!(a.dead_qubits().is_subset(b.dead_qubits()), "qubits at {hi} vs {lo}");
            assert!(a.dead_couplers().is_subset(b.dead_couplers()), "couplers at {hi} vs {lo}");
        }
    }

    /// Same seed, same machine, same defects; a different seed is a different machine.
    #[test]
    fn a_defect_map_is_reproducible_from_its_seed() {
        let ideal = identity_numbered(chimera(2, 2, 4, 1.0));
        assert_eq!(Defects::sample(&ideal, 0.9, 0.9, 42), Defects::sample(&ideal, 0.9, 0.9, 42));
        assert_ne!(Defects::sample(&ideal, 0.9, 0.9, 42), Defects::sample(&ideal, 0.9, 0.9, 43));
    }

    /// A hand-built embedding on `grid(3)`, checked against the grid's own definition, and then the
    /// three ways a defect can break it. The clean case passing is what stops the rest being
    /// vacuous: each mutation kills exactly one thing and the audit must name it.
    ///
    /// `grid(3)` is `y*3 + x`. Chain `{0, 1}` is held by the coupler 0-1; it reaches `{3}` through
    /// 0-3 and `{4}` through 1-4, and `{3}` reaches `{4}` through 3-4.
    #[test]
    fn the_audit_names_the_defect_that_breaks_a_placement() {
        let ideal = identity_numbered(grid(3));
        let logical = triangle();
        let chains = vec![vec![0u32, 1], vec![3], vec![4]];
        let e = Embedding { chains: vec![vec![0, 1], vec![3], vec![4]], sites: 9 };

        let clean = Working::new(&ideal, &Defects::none());
        clean.audit_qubits(&logical, &chains).expect("this is an embedding of the triangle");
        clean.audit(&logical, &e).expect("and it verifies against the working graph");
        assert_eq!(
            clean.couplers_used(&logical, &chains),
            vec![(0, 1), (0, 3), (1, 4), (3, 4)],
            "exactly the four couplers a run would program"
        );

        // A dead qubit under a chain.
        let mut d = Defects::none();
        d.kill_qubit(4);
        let why = Working::new(&ideal, &d).audit_qubits(&logical, &chains).unwrap_err();
        assert!(why.contains("qubit 4") && why.contains("dead"), "{why}");

        // A dead coupler holding a chain together.
        let mut d = Defects::none();
        d.kill_coupler(0, 1);
        let why = Working::new(&ideal, &d).audit_qubits(&logical, &chains).unwrap_err();
        assert!(why.contains("not connected by live couplers"), "{why}");

        // A dead coupler realising a logical edge.
        let mut d = Defects::none();
        d.kill_coupler(3, 4);
        let why = Working::new(&ideal, &d).audit_qubits(&logical, &chains).unwrap_err();
        assert!(why.contains("variables 1 and 2"), "{why}");

        // A qubit that is not on the machine at all.
        let off = vec![vec![0u32, 1], vec![3], vec![99]];
        let why = clean.audit_qubits(&logical, &off).unwrap_err();
        assert!(why.contains("never had"), "{why}");
    }

    /// The reason this module exists: an embedding computed against the ideal lattice names dead
    /// qubits on the real machine, and the audit says so.
    ///
    /// `chimera_clique(3, 4)` is a construction, not a search, so its chains are known exactly —
    /// qubit 0 is the first site of variable 0's chain.
    #[test]
    fn an_ideal_lattice_embedding_is_rejected_by_the_real_machine() {
        let ideal = identity_numbered(chimera(3, 3, 4, 1.0));
        let clique = complete(12);
        let e = chimera_clique(3, 4).expect("K_12 on a 3x3 chimera");
        e.verify(&clique, &ideal.graph).expect("the construction is right on the ideal lattice");

        let victim = e.chains[0][0] as u32;
        let mut d = Defects::none();
        d.kill_qubit(victim);
        let w = Working::new(&ideal, &d);
        let chains: Vec<Vec<u32>> =
            e.chains.iter().map(|c| c.iter().map(|&s| s as u32).collect()).collect();
        let why = w.audit_qubits(&clique, &chains).unwrap_err();
        assert!(why.contains(&format!("qubit {victim}")) && why.contains("dead"), "{why}");
    }

    /// An embedding found against a working graph uses no dead qubit and no dead coupler — read off
    /// the returned chains in machine numbering, not inferred from how the graph was built — and it
    /// still verifies against the working graph.
    #[test]
    fn an_embedding_onto_a_defective_machine_touches_nothing_dead() {
        let ideal = identity_numbered(chimera(4, 4, 4, 1.0));
        let logical = complete(5);
        let mut found = 0;
        let mut defective = 0;

        for seed in 0..8u64 {
            let d = Defects::sample(&ideal, 0.92, 0.95, seed);
            let w = Working::new(&ideal, &d);
            assert!(w.stats().qubit_rate() < 1.0);
            defective += 1;

            let Some(e) = w.embed(&logical, seed) else { continue };
            found += 1;

            let chains = w.chain_qubits(&e).unwrap();
            for (v, chain) in chains.iter().enumerate() {
                for &q in chain {
                    assert!(!d.qubit_dead(q), "variable {v} stands on dead qubit {q}");
                    assert!(w.qubit_live(q));
                }
            }
            for (a, b) in w.couplers_used(&logical, &chains) {
                assert!(!d.coupler_dead(a, b), "the run would program dead coupler {a}-{b}");
                assert!(w.coupler_live(a, b));
            }
            // The chains have to still be an embedding of the program into the machine that is left.
            e.verify(&logical, w.graph()).expect("verify against the working graph");
            w.audit(&logical, &e).expect("audit against the live fabric");
        }
        assert_eq!(defective, 8);
        assert!(found >= 6, "only {found} of 8 defective machines took a K_5");
    }

    /// A model applied under an embedding programs only live sites, and its total weight is the
    /// model's — an identity that has to survive the renumbering.
    #[test]
    fn applying_a_model_to_a_working_graph_conserves_its_weight() {
        let ideal = identity_numbered(king(6));
        let d = Defects::sample(&ideal, 0.9, 0.9, 2);
        let w = Working::new(&ideal, &d);
        let logical = triangle();
        let e = w.embed(&logical, 4).expect("a triangle fits a defective 6x6 king");
        let out = w.apply(&logical, &e);

        assert_eq!(out.graph.n, w.graph().n);
        let chain_weight: f64 = e
            .chains
            .iter()
            .map(|c| {
                let mut s = 0.0;
                for a in 0..c.len() {
                    for b in (a + 1)..c.len() {
                        let (u, v) = (c[a], c[b]);
                        if (w.graph().offset[u]..w.graph().offset[u + 1])
                            .any(|k| w.graph().nbr[k] as usize == v)
                        {
                            s += out.chain_strength;
                        }
                    }
                }
                s
            })
            .sum();
        let total: f64 = (0..out.graph.n)
            .flat_map(|i| (out.graph.offset[i]..out.graph.offset[i + 1]).map(move |k| (i, k)))
            .filter(|&(i, k)| out.graph.nbr[k] as usize > i)
            .map(|(_, k)| out.graph.w[k])
            .sum();
        assert!((total - (3.0 + chain_weight)).abs() < 1e-9, "total {total}");
    }
}
