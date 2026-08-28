//! Planar embedding — the rotation system every surface algorithm is written against.
//!
//! "Planar" is not a property you can use; an **embedding** is. Max-cut on a planar graph is
//! polynomial because a cut in `G` is a cycle in the dual `G*`, and there is no dual until the
//! faces are known — which needs the cyclic order of edges around each vertex, not merely the
//! knowledge that one exists. This module produces that order.
//!
//! # Path addition, not Left-Right
//!
//! The linear-time planarity tests (Hopcroft–Tarjan, Left-Right, Boyer–Myrvold) are the right
//! choice at scale and the wrong choice here: they are long, subtle, and their intermediate state
//! is not checkable against anything. Demoucron's path-addition method is `O(n²)`, which at the
//! sizes an exact max-cut can be run on is not the binding cost, and it builds the embedding
//! directly as it goes:
//!
//! 1. Embed any cycle. It has two faces, inside and out.
//! 2. Take the subgraph embedded so far. Every remaining piece is a **fragment** — a connected
//!    component of the rest, plus the embedded vertices it touches.
//! 3. A fragment can only be drawn inside a face that contains all of its contact vertices. Count
//!    those faces. **If any fragment has none, the graph is not planar.** If one has exactly one,
//!    it must go there, so do that first — the standard argument that the greedy choice is safe.
//! 4. Embed a path through the chosen fragment, splitting the chosen face in two. Repeat.
//!
//! # It is checked, not trusted
//!
//! [`Embedding::faces`] traces faces by walking darts, and [`Embedding::is_consistent`] asserts
//! **Euler's formula** on the result: `V − E + F = 2` for a connected planar embedding. A rotation
//! system that is not planar fails that, so an embedding this module returns has been checked
//! against an invariant it did not construct itself. Every test here runs it.
//!
//! ```
//! use ferrotherm::{planar, ising::grid2d};
//!
//! let g = grid2d(4, 4, 1.0);                   // a 4x4 grid is planar
//! let e = planar::embed(&g).expect("a grid embeds");
//! assert!(e.is_consistent());
//! assert_eq!(e.faces().len(), 10);             // 9 squares and the outer face
//! ```

use crate::graph::Graph;

/// A combinatorial embedding: for each vertex, its neighbours in cyclic order.
///
/// Two rotation systems that differ only by reflecting every vertex describe the same embedding on
/// the sphere, and nothing here distinguishes them. What matters is that the faces traced from it
/// are the faces of a genuine planar drawing, which [`Embedding::is_consistent`] checks.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Embedding {
    n: usize,
    adj: Vec<Vec<usize>>,
}

impl Embedding {
    /// Neighbours of `v`, in cyclic order.
    pub fn rotation(&self, v: usize) -> &[usize] {
        &self.adj[v]
    }

    /// Nodes.
    pub fn len(&self) -> usize {
        self.n
    }

    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Undirected edges.
    pub fn edges(&self) -> usize {
        self.adj.iter().map(|a| a.len()).sum::<usize>() / 2
    }

    /// Trace the faces, each as the sequence of directed edges bounding it.
    ///
    /// The standard walk: from dart `(u, v)`, the next dart is `(v, w)` where `w` is the neighbour
    /// **before** `u` in `v`'s rotation. Every dart belongs to exactly one face, so the traces
    /// partition the `2E` darts — which is what makes the Euler check below meaningful rather than
    /// circular.
    pub fn faces(&self) -> Vec<Vec<(usize, usize)>> {
        let mut seen: Vec<Vec<bool>> = self.adj.iter().map(|a| vec![false; a.len()]).collect();
        let mut out = Vec::new();
        for u in 0..self.n {
            for k in 0..self.adj[u].len() {
                if seen[u][k] {
                    continue;
                }
                let mut face = Vec::new();
                let (mut a, mut i) = (u, k);
                loop {
                    if seen[a][i] {
                        break;
                    }
                    seen[a][i] = true;
                    let b = self.adj[a][i];
                    face.push((a, b));
                    // Position of `a` in `b`'s rotation, stepped one backwards.
                    let j = self.adj[b].iter().position(|&x| x == a).expect("the reverse dart");
                    let deg = self.adj[b].len();
                    i = (j + deg - 1) % deg;
                    a = b;
                }
                out.push(face);
            }
        }
        out
    }

    /// `V − E + F` for this rotation system: the Euler characteristic of the surface it embeds in.
    ///
    /// 2 is the sphere, 0 the torus, −2 the genus-2 surface. This is the one number that says
    /// *which surface* a list of lists actually describes, and everything downstream turns on it:
    /// the dual argument behind [`crate::planarcut`] gives an exact maximum on the sphere and only
    /// an upper bound above it.
    pub fn euler(&self) -> i64 {
        if self.n == 0 {
            return 2;
        }
        self.n as i64 - self.edges() as i64 + self.faces().len() as i64
    }

    /// Orientable genus, from Euler's formula `χ = 2 − 2g`. `None` if `χ` is odd or positive.
    pub fn genus(&self) -> Option<usize> {
        let c = self.euler();
        if c > 2 || (2 - c) % 2 != 0 {
            return None;
        }
        Some(((2 - c) / 2) as usize)
    }

    /// Does this rotation system describe a **planar** embedding of a connected graph?
    ///
    /// Euler: `V − E + F = 2`. Checked rather than assumed, because a rotation system is just a
    /// list of lists — nothing about its type says the faces it traces close up into a sphere.
    pub fn is_consistent(&self) -> bool {
        if self.n == 0 {
            return true;
        }
        self.euler() == 2
    }
}

/// The rotation system of a `w × h` **toroidal** grid, laid out row-major.
///
/// A torus is not a plane and [`embed`] refuses it, correctly. But an embedding on any surface has
/// faces, and faces are all the dual argument needs — so this exists to hand
/// [`crate::planarcut::bound_on_surface`] the one thing it cannot derive.
///
/// The cyclic order at every vertex is right, up, left, down, which is the standard embedding: it
/// traces `w·h` square faces, and `V − E + F = wh − 2wh + wh = 0`, the Euler characteristic of the
/// torus. Both are asserted by the tests rather than argued here.
///
/// `w` and `h` must be at least 3. At 2 the periodic grid has a doubled edge between neighbours —
/// two distinct edges the graph type cannot tell apart — and an embedding of a multigraph is a
/// different object from an embedding of its underlying simple graph.
pub fn torus_grid(w: usize, h: usize) -> Option<Embedding> {
    if w < 3 || h < 3 {
        return None;
    }
    let n = w * h;
    let mut adj = vec![Vec::with_capacity(4); n];
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let right = y * w + (x + 1) % w;
            let up = ((y + h - 1) % h) * w + x;
            let left = y * w + (x + w - 1) % w;
            let down = ((y + 1) % h) * w + x;
            adj[i] = vec![right, up, left, down];
        }
    }
    Some(Embedding { n, adj })
}

/// Recover a toroidal grid embedding from an edge list, if the graph is one.
///
/// A toroidal grid carries its embedding in its structure, but a file does not say so — G-set's
/// toroidal instances are edge lists like any other, and "toroidal" is a word in the accompanying
/// prose. This tries every factorisation `n = w · h` and returns the embedding for the first whose
/// **entire edge set** matches, so a match is a proof rather than a guess: `2n` edges all in the
/// right places cannot happen by accident.
///
/// Measured on the three toroidal G-set instances: G11 is 8 × 100, G12 is 16 × 50, G13 is 32 × 25.
pub fn torus_grid_of(g: &Graph) -> Option<Embedding> {
    let n = g.n;
    if n < 9 {
        return None;
    }
    let mut have: Vec<(usize, usize)> = Vec::new();
    for u in 0..n {
        for k in g.offset[u]..g.offset[u + 1] {
            let v = g.nbr[k] as usize;
            if v > u {
                have.push((u, v));
            }
        }
    }
    have.sort_unstable();
    have.dedup();
    if have.len() != 2 * n {
        return None; // a toroidal grid is 4-regular
    }
    for w in 3..=(n / 3) {
        // `is_multiple_of` rather than `% == 0`: stable since 1.87 and what clippy asks for. No
        // MSRV is declared here and CI builds on stable, so there is nothing to be cautious about.
        if !n.is_multiple_of(w) {
            continue;
        }
        let h = n / w;
        if h < 3 {
            continue;
        }
        let mut want: Vec<(usize, usize)> = Vec::with_capacity(2 * n);
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                let a = y * w + (x + 1) % w;
                let b = ((y + 1) % h) * w + x;
                want.push((i.min(a), i.max(a)));
                want.push((i.min(b), i.max(b)));
            }
        }
        want.sort_unstable();
        want.dedup();
        if want == have {
            return torus_grid(w, h);
        }
    }
    None
}

/// Build an [`Embedding`] from a rotation system supplied by the caller.
///
/// Validated, not trusted: every list must be a permutation of a set of distinct neighbours, the
/// relation must be symmetric, and every dart must land on exactly one traced face. A rotation
/// system that fails any of those is not an embedding of anything, and the faces it would trace
/// would be a dual of nothing.
pub fn from_rotation(adj: Vec<Vec<usize>>) -> Option<Embedding> {
    let n = adj.len();
    for (v, list) in adj.iter().enumerate() {
        let mut s = list.clone();
        s.sort_unstable();
        let before = s.len();
        s.dedup();
        if s.len() != before {
            return None; // a repeated neighbour
        }
        if s.iter().any(|&u| u >= n || u == v) {
            return None;
        }
        for &u in list {
            if !adj[u].contains(&v) {
                return None; // not symmetric
            }
        }
    }
    let e = Embedding { n, adj };
    // The dart count is the check that the traced faces really partition the embedding.
    let darts: usize = e.faces().iter().map(|f| f.len()).sum();
    (darts == 2 * e.edges()).then_some(e)
}

/// Embed `g` in the plane, or return `None`.
///
/// `None` means one of three things, and [`why`] distinguishes them:
///
/// * the graph is **not planar**, which is the interesting answer;
/// * it is **disconnected** — refused rather than half-embedded, because the caller's next step is
///   a dual and a dual of one component is a wrong answer that looks like a right one;
/// * it is connected but **not 2-connected**, which path addition cannot handle. A fragment hanging
///   off a single cut vertex has only one contact, so there is no path between two contacts to add,
///   and the method has nothing to say. That is a limitation of this algorithm rather than a fact
///   about the graph, and it is reported as one. Trees are the exception and embed directly.
///
/// The distinction matters to a caller: max-cut decomposes exactly across biconnected components,
/// so "not 2-connected" is an instruction to split the graph, where "not planar" is not.
pub fn embed(g: &Graph) -> Option<Embedding> {
    let n = g.n;
    if n == 0 {
        return Some(Embedding { n: 0, adj: Vec::new() });
    }
    // Simple undirected adjacency; the Graph CSR may hold both directions and may repeat.
    let mut nbr: Vec<Vec<usize>> = vec![Vec::new(); n];
    for u in 0..n {
        for k in g.offset[u]..g.offset[u + 1] {
            let v = g.nbr[k] as usize;
            if v != u && !nbr[u].contains(&v) {
                nbr[u].push(v);
            }
        }
    }
    if !connected(&nbr) {
        return None;
    }
    // A tree or a single edge has no cycle to start from, and is planar with one face.
    let m: usize = nbr.iter().map(|a| a.len()).sum::<usize>() / 2;
    if m + 1 == n {
        return Some(tree_embedding(&nbr));
    }
    // Kuratowski's bound. Not an optimisation: it makes the O(n^2) loop below safe to enter on a
    // dense graph that could never be planar anyway.
    if n >= 3 && m > 3 * n - 6 {
        return None;
    }
    if !biconnected(&nbr) {
        // Refused HERE, with a reason, rather than three hundred lines later when `path` cannot
        // find a second contact and the whole run unwinds into an indistinguishable `None`. That
        // is how this limitation presented the first time, and a limitation that arrives as a
        // mystery is one nobody can route around.
        return None;
    }
    Demoucron::new(nbr).run()
}

/// Why [`embed`] would refuse this graph. `None` means it would not.
///
/// Separated out so a caller can act: "not 2-connected" is an instruction to split into blocks,
/// "not planar" is a fact about the instance, and "disconnected" is usually a bug upstream.
pub fn why(g: &Graph) -> Option<Refusal> {
    let n = g.n;
    if n == 0 {
        return None;
    }
    let mut nbr: Vec<Vec<usize>> = vec![Vec::new(); n];
    for u in 0..n {
        for k in g.offset[u]..g.offset[u + 1] {
            let v = g.nbr[k] as usize;
            if v != u && !nbr[u].contains(&v) {
                nbr[u].push(v);
            }
        }
    }
    if !connected(&nbr) {
        return Some(Refusal::Disconnected);
    }
    let m: usize = nbr.iter().map(|a| a.len()).sum::<usize>() / 2;
    if m + 1 == n {
        return None; // a tree, which embeds
    }
    if n >= 3 && m > 3 * n - 6 {
        return Some(Refusal::NotPlanar);
    }
    if !biconnected(&nbr) {
        return Some(Refusal::NotBiconnected);
    }
    Demoucron::new(nbr).run().is_none().then_some(Refusal::NotPlanar)
}

/// Why an embedding was refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refusal {
    /// More than one component. Embed them separately, or fix the caller.
    Disconnected,
    /// A cut vertex. Split into biconnected components — max-cut decomposes across them exactly.
    NotBiconnected,
    /// A fact about the graph: it contains a subdivision of `K5` or `K3,3`.
    NotPlanar,
}

impl core::fmt::Display for Refusal {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Refusal::Disconnected => write!(f, "the graph is disconnected"),
            Refusal::NotBiconnected => write!(
                f,
                "the graph has a cut vertex, and path addition needs two contacts to add a path \
                 between. Split it into biconnected components: max-cut decomposes across them \
                 exactly, because the two sides of a block can be flipped independently"
            ),
            Refusal::NotPlanar => write!(f, "the graph is not planar"),
        }
    }
}

/// No cut vertices. Hopcroft-Tarjan low-points, iteratively.
fn biconnected(nbr: &[Vec<usize>]) -> bool {
    let n = nbr.len();
    if n < 3 {
        return true;
    }
    let (mut disc, mut low) = (vec![usize::MAX; n], vec![0usize; n]);
    let mut timer = 0usize;
    let mut root_children = 0usize;
    // (vertex, parent, next neighbour index)
    let mut stack: Vec<(usize, usize, usize)> = vec![(0, usize::MAX, 0)];
    disc[0] = timer;
    low[0] = timer;
    timer += 1;
    while let Some(&mut (u, parent, ref mut k)) = stack.last_mut() {
        if *k < nbr[u].len() {
            let v = nbr[u][*k];
            *k += 1;
            if v == parent {
                continue;
            }
            if disc[v] != usize::MAX {
                low[u] = low[u].min(disc[v]);
            } else {
                disc[v] = timer;
                low[v] = timer;
                timer += 1;
                if u == 0 {
                    root_children += 1;
                }
                stack.push((v, u, 0));
            }
        } else {
            stack.pop();
            if let Some(&mut (p, _, _)) = stack.last_mut() {
                low[p] = low[p].min(low[u]);
                // A non-root cut vertex: some child cannot reach above `p` without going through it.
                if p != 0 && low[u] >= disc[p] {
                    return false;
                }
            }
        }
    }
    root_children <= 1 && disc.iter().all(|&d| d != usize::MAX)
}

fn connected(nbr: &[Vec<usize>]) -> bool {
    let n = nbr.len();
    let mut seen = vec![false; n];
    let mut stack = vec![0usize];
    seen[0] = true;
    let mut count = 1;
    while let Some(u) = stack.pop() {
        for &v in &nbr[u] {
            if !seen[v] {
                seen[v] = true;
                count += 1;
                stack.push(v);
            }
        }
    }
    count == n
}

/// A tree embeds with any rotation, and every dart is on the single face.
fn tree_embedding(nbr: &[Vec<usize>]) -> Embedding {
    Embedding { n: nbr.len(), adj: nbr.to_vec() }
}

/// Demoucron's path-addition planarity test, carrying the embedding as it goes.
struct Demoucron {
    n: usize,
    nbr: Vec<Vec<usize>>,
    /// Which edges are already drawn. Indexed by `(u, v)` with `u < v`.
    drawn: Vec<Vec<bool>>,
    /// Faces of the drawing so far, each a cyclic vertex sequence.
    faces: Vec<Vec<usize>>,
    /// Vertices in the drawing so far.
    inside: Vec<bool>,
}

impl Demoucron {
    fn new(nbr: Vec<Vec<usize>>) -> Demoucron {
        let n = nbr.len();
        Demoucron {
            n,
            nbr,
            drawn: vec![vec![false; n]; n],
            faces: Vec::new(),
            inside: vec![false; n],
        }
    }

    fn edge_count(&self) -> usize {
        self.nbr.iter().map(|a| a.len()).sum::<usize>() / 2
    }

    fn drawn_count(&self) -> usize {
        let mut c = 0;
        for u in 0..self.n {
            for v in (u + 1)..self.n {
                if self.drawn[u][v] {
                    c += 1;
                }
            }
        }
        c
    }

    /// Any cycle, found by DFS. `None` on a forest, which the caller has already excluded.
    fn find_cycle(&self) -> Option<Vec<usize>> {
        let mut parent = vec![usize::MAX; self.n];
        let mut state = vec![0u8; self.n]; // 0 unseen, 1 on the stack, 2 done
        let mut stack: Vec<(usize, usize)> = vec![(0, 0)];
        state[0] = 1;
        while let Some(&mut (u, ref mut k)) = stack.last_mut() {
            if *k < self.nbr[u].len() {
                let v = self.nbr[u][*k];
                *k += 1;
                if v == parent[u] {
                    continue;
                }
                if state[v] == 1 {
                    // Back edge: walk the tree path from u up to v.
                    let mut cyc = vec![v];
                    let mut x = u;
                    while x != v {
                        cyc.push(x);
                        x = parent[x];
                    }
                    cyc.reverse();
                    return Some(cyc);
                }
                if state[v] == 0 {
                    state[v] = 1;
                    parent[v] = u;
                    stack.push((v, 0));
                }
            } else {
                state[u] = 2;
                stack.pop();
            }
        }
        None
    }

    /// The fragments of the graph relative to the current drawing.
    ///
    /// Two kinds, and both matter: a single undrawn edge between two drawn vertices, and a
    /// connected component of undrawn vertices together with every drawn vertex it touches.
    fn fragments(&self) -> Vec<(Vec<usize>, Vec<usize>)> {
        let mut out: Vec<(Vec<usize>, Vec<usize>)> = Vec::new();
        // Single undrawn edges between two drawn vertices.
        for u in 0..self.n {
            if !self.inside[u] {
                continue;
            }
            for &v in &self.nbr[u] {
                if v > u && self.inside[v] && !self.drawn[u][v] {
                    out.push((vec![u, v], vec![u, v]));
                }
            }
        }
        // Components of undrawn vertices.
        let mut seen = vec![false; self.n];
        for s in 0..self.n {
            if self.inside[s] || seen[s] {
                continue;
            }
            let mut comp = Vec::new();
            let mut contacts = Vec::new();
            let mut stack = vec![s];
            seen[s] = true;
            while let Some(u) = stack.pop() {
                comp.push(u);
                for &v in &self.nbr[u] {
                    if self.inside[v] {
                        if !contacts.contains(&v) {
                            contacts.push(v);
                        }
                    } else if !seen[v] {
                        seen[v] = true;
                        stack.push(v);
                    }
                }
            }
            comp.extend_from_slice(&contacts);
            out.push((comp, contacts));
        }
        out
    }

    /// Faces that can hold every contact vertex of this fragment.
    fn admissible(&self, contacts: &[usize]) -> Vec<usize> {
        (0..self.faces.len())
            .filter(|&f| contacts.iter().all(|c| self.faces[f].contains(c)))
            .collect()
    }

    /// A path across the fragment from one contact to another, through undrawn vertices only.
    fn path(&self, fragment: &[usize], contacts: &[usize]) -> Option<Vec<usize>> {
        let start = contacts[0];
        let inset: Vec<bool> = {
            let mut s = vec![false; self.n];
            for &v in fragment {
                s[v] = true;
            }
            s
        };
        // BFS from `start` through undrawn interior vertices, stopping at another contact. A single
        // undrawn edge between two drawn vertices is the degenerate case and comes out as length 2.
        let mut prev = vec![usize::MAX; self.n];
        let mut seen = vec![false; self.n];
        let mut queue = std::collections::VecDeque::new();
        seen[start] = true;
        queue.push_back(start);
        while let Some(u) = queue.pop_front() {
            for &v in &self.nbr[u] {
                if !inset[v] || seen[v] || (self.inside[u] && self.inside[v] && self.drawn[u][v]) {
                    continue;
                }
                if self.inside[u] && u != start {
                    continue; // a drawn vertex other than the start ends the path
                }
                seen[v] = true;
                prev[v] = u;
                if self.inside[v] {
                    let mut p = vec![v];
                    let mut x = u;
                    while x != start {
                        p.push(x);
                        x = prev[x];
                    }
                    p.push(start);
                    p.reverse();
                    return Some(p);
                }
                queue.push_back(v);
            }
        }
        None
    }

    /// Split face `f` along `path`, whose endpoints are both on it.
    fn split_face(&mut self, f: usize, path: &[usize]) {
        let cycle = self.faces[f].clone();
        let a = cycle.iter().position(|&x| x == path[0]).expect("endpoint on the face");
        let b = cycle.iter().position(|&x| x == path[path.len() - 1]).expect("endpoint on the face");
        // Two arcs of the face between the endpoints; the path plus each arc is a new face.
        let mut arc1 = Vec::new();
        let mut i = a;
        while i != b {
            arc1.push(cycle[i]);
            i = (i + 1) % cycle.len();
        }
        arc1.push(cycle[b]);
        let mut arc2 = Vec::new();
        let mut i = b;
        while i != a {
            arc2.push(cycle[i]);
            i = (i + 1) % cycle.len();
        }
        arc2.push(cycle[a]);

        let mut f1 = arc1;
        f1.extend(path[1..path.len() - 1].iter().rev());
        let mut f2 = arc2;
        f2.extend(path[1..path.len() - 1].iter());
        self.faces[f] = f1;
        self.faces.push(f2);
    }

    fn draw_path(&mut self, path: &[usize]) {
        for w in path.windows(2) {
            let (a, b) = (w[0], w[1]);
            self.drawn[a][b] = true;
            self.drawn[b][a] = true;
            self.inside[a] = true;
            self.inside[b] = true;
        }
    }

    fn run(mut self) -> Option<Embedding> {
        let cycle = self.find_cycle()?;
        for w in cycle.windows(2) {
            self.drawn[w[0]][w[1]] = true;
            self.drawn[w[1]][w[0]] = true;
        }
        let (a, b) = (cycle[0], cycle[cycle.len() - 1]);
        self.drawn[a][b] = true;
        self.drawn[b][a] = true;
        for &v in &cycle {
            self.inside[v] = true;
        }
        // A cycle has exactly two faces, and they are the same cyclic sequence traversed each way.
        self.faces.push(cycle.clone());
        let mut rev = cycle.clone();
        rev.reverse();
        self.faces.push(rev);

        let total = self.edge_count();
        while self.drawn_count() < total {
            let frags = self.fragments();
            if frags.is_empty() {
                break;
            }
            // The greedy step, and the one place non-planarity is detected: a fragment with no
            // admissible face cannot be drawn anywhere, now or later. A fragment with exactly one
            // has no choice, so taking it first cannot be wrong -- which is what makes the greedy
            // choice safe rather than merely convenient.
            let mut chosen: Option<(usize, usize)> = None;
            for (i, (_, contacts)) in frags.iter().enumerate() {
                let adm = self.admissible(contacts);
                if adm.is_empty() {
                    return None;
                }
                if adm.len() == 1 {
                    chosen = Some((i, adm[0]));
                    break;
                }
                if chosen.is_none() {
                    chosen = Some((i, adm[0]));
                }
            }
            let (fi, face) = chosen?;
            let (frag, contacts) = &frags[fi];
            let path = self.path(frag, contacts)?;
            self.draw_path(&path);
            self.split_face(face, &path);
        }
        self.into_embedding()
    }

    /// Turn the face list into a rotation system.
    ///
    /// Each face gives, at every vertex on it, a pair of consecutive darts. Chaining those pairs
    /// around a vertex reconstructs its cyclic order — and if they do not chain into a single
    /// cycle, the face structure was not planar and this returns `None` rather than an embedding
    /// nothing would check.
    fn into_embedding(self) -> Option<Embedding> {
        let n = self.n;
        // successor[u][v] = w means: around u, the dart to w follows the dart to v.
        let mut succ: Vec<std::collections::BTreeMap<usize, usize>> =
            vec![std::collections::BTreeMap::new(); n];
        for face in &self.faces {
            let k = face.len();
            for i in 0..k {
                let prev = face[(i + k - 1) % k];
                let v = face[i];
                let next = face[(i + 1) % k];
                if prev == next {
                    continue;
                }
                succ[v].insert(prev, next);
            }
        }
        let mut adj = vec![Vec::new(); n];
        for v in 0..n {
            let deg = self.nbr[v].len();
            if deg == 0 {
                return None;
            }
            if succ[v].len() != deg {
                return None;
            }
            let start = *succ[v].keys().next().expect("non-empty");
            let mut order = Vec::with_capacity(deg);
            let mut cur = start;
            for _ in 0..deg {
                order.push(cur);
                cur = *succ[v].get(&cur)?;
            }
            if cur != start || order.len() != deg {
                return None;
            }
            let mut sorted = order.clone();
            sorted.sort_unstable();
            let mut want = self.nbr[v].clone();
            want.sort_unstable();
            if sorted != want {
                return None;
            }
            adj[v] = order;
        }
        let e = Embedding { n, adj };
        e.is_consistent().then_some(e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::GraphBuilder;
    use crate::ising::{grid2d, lattice2d, ring};

    fn complete(n: usize) -> Graph {
        let mut gb = GraphBuilder::new(n);
        for i in 0..n {
            for j in (i + 1)..n {
                gb.couple(i, j, 1.0);
            }
        }
        gb.build()
    }

    fn bipartite(a: usize, b: usize) -> Graph {
        let mut gb = GraphBuilder::new(a + b);
        for i in 0..a {
            for j in 0..b {
                gb.couple(i, a + j, 1.0);
            }
        }
        gb.build()
    }

    /// THE TWO GRAPHS PLANARITY IS DEFINED BY. Kuratowski's theorem says every non-planar graph
    /// contains a subdivision of one of them, so a test that accepts either is not a planarity test
    /// at all — and would hand [`crate::planarcut`] a dual that does not exist.
    #[test]
    fn k5_and_k33_are_refused() {
        assert!(embed(&complete(5)).is_none(), "K5 is not planar");
        assert!(embed(&bipartite(3, 3)).is_none(), "K3,3 is not planar");
        // And the reason is the interesting one, not an incidental refusal.
        assert_eq!(why(&complete(5)), Some(Refusal::NotPlanar));
        assert_eq!(why(&bipartite(3, 3)), Some(Refusal::NotPlanar));
        // And the largest planar complete graph still embeds, so the refusal is not blanket.
        let k4 = embed(&complete(4)).expect("K4 is planar");
        assert!(k4.is_consistent());
        assert_eq!(k4.faces().len(), 4);
    }

    /// Every embedding this module returns satisfies Euler's formula, on graphs whose face count is
    /// known independently.
    #[test]
    fn the_faces_are_the_faces() {
        // A ring: two faces, inside and out.
        let r = embed(&ring(9, 1.0, 0.0)).expect("a cycle is planar");
        assert!(r.is_consistent());
        assert_eq!(r.faces().len(), 2);

        // A w x h grid: (w-1)(h-1) squares plus the outer face.
        for (w, h) in [(2usize, 2usize), (3, 3), (4, 4), (5, 3), (6, 6)] {
            let e = embed(&grid2d(w, h, 1.0)).unwrap_or_else(|| panic!("{w}x{h} grid is planar"));
            assert!(e.is_consistent(), "{w}x{h} fails Euler");
            assert_eq!(
                e.faces().len(),
                (w - 1) * (h - 1) + 1,
                "{w}x{h}: expected {} faces",
                (w - 1) * (h - 1) + 1
            );
            assert_eq!(e.len(), w * h);
            assert_eq!(e.edges(), w * (h - 1) + h * (w - 1));
        }
    }

    /// A torus is not a plane, and the wrap-around edges are exactly what breaks it. This is the
    /// distinction [`crate::planarcut`] rests on, so it is asserted rather than assumed: a periodic
    /// lattice of side 3 or more is non-planar and must be refused.
    #[test]
    fn a_periodic_lattice_is_not_planar() {
        for l in [3usize, 4, 5] {
            assert!(
                embed(&lattice2d(l, 1.0)).is_none(),
                "the {l}x{l} PERIODIC lattice is a torus and must not embed in the plane"
            );
        }
        // The same size with open boundaries does embed, which is the control.
        assert!(embed(&grid2d(4, 4, 1.0)).is_some());
    }

    /// The rotation system has to be a permutation of the real neighbours at every vertex — not a
    /// subset, not a multiset. A rotation that dropped an edge would trace faces that close up
    /// perfectly and describe a different graph.
    #[test]
    fn the_rotation_is_a_permutation_of_the_neighbours() {
        let g = grid2d(5, 4, 1.0);
        let e = embed(&g).unwrap();
        for v in 0..e.len() {
            let mut got = e.rotation(v).to_vec();
            got.sort_unstable();
            let mut want: Vec<usize> =
                (g.offset[v]..g.offset[v + 1]).map(|k| g.nbr[k] as usize).collect();
            want.sort_unstable();
            want.dedup();
            assert_eq!(got, want, "vertex {v}");
        }
        // And every dart is on exactly one face, so the traces partition 2E.
        let darts: usize = e.faces().iter().map(|f| f.len()).sum();
        assert_eq!(darts, 2 * e.edges());
    }

    /// The torus is not a plane, and the rotation system says so in the one number that can.
    #[test]
    fn the_toroidal_grid_has_euler_characteristic_zero() {
        for (w, h) in [(3usize, 3usize), (4, 4), (5, 7), (20, 40)] {
            let e = torus_grid(w, h).expect("3 or more each way");
            assert_eq!(e.len(), w * h);
            assert_eq!(e.edges(), 2 * w * h, "a 4-regular graph on {w}x{h}");
            assert_eq!(e.faces().len(), w * h, "every face of a toroidal grid is a square");
            assert_eq!(e.euler(), 0, "{w}x{h}: the torus has chi = 0");
            assert_eq!(e.genus(), Some(1));
            assert!(!e.is_consistent(), "it is a valid embedding, and NOT a planar one");
            // Every dart on exactly one face, which is what makes the dual well defined.
            assert_eq!(e.faces().iter().map(|f| f.len()).sum::<usize>(), 2 * e.edges());
        }
        // Below 3 the periodic grid has doubled edges, and an embedding of a multigraph is a
        // different object from an embedding of its underlying simple graph.
        assert!(torus_grid(2, 5).is_none());
        assert!(torus_grid(5, 2).is_none());
    }

    /// The structure is recovered from the edge list alone, and only when it is really there.
    #[test]
    fn a_toroidal_grid_is_recognised_and_nothing_else_is() {
        use crate::ising::lattice2d;
        for (w, h) in [(3usize, 4usize), (5, 8), (8, 100), (16, 50)] {
            let mut gb = GraphBuilder::new(w * h);
            for y in 0..h {
                for x in 0..w {
                    let i = y * w + x;
                    gb.couple(i, y * w + (x + 1) % w, 1.0);
                    gb.couple(i, ((y + 1) % h) * w + x, 1.0);
                }
            }
            let e = torus_grid_of(&gb.build())
                .unwrap_or_else(|| panic!("{w}x{h} is a toroidal grid"));
            assert_eq!(e.euler(), 0);
            assert_eq!(e.len(), w * h);
        }
        // A square periodic lattice is the same object by another constructor.
        assert!(torus_grid_of(&lattice2d(6, 1.0)).is_some());
        // And things that are not: an open grid, and a graph of the right degree that is not a grid.
        assert!(torus_grid_of(&grid2d(6, 6, 1.0)).is_none(), "open boundaries are not a torus");
        assert!(torus_grid_of(&ring(12, 1.0, 0.0)).is_none(), "degree 2, not 4");
    }

    /// A rotation system supplied from outside is validated, not trusted.
    #[test]
    fn a_supplied_rotation_is_checked_rather_than_believed() {
        // The planar 4-cycle, by hand.
        let ok = from_rotation(vec![vec![1, 3], vec![0, 2], vec![1, 3], vec![0, 2]])
            .expect("a valid 4-cycle rotation");
        assert_eq!(ok.euler(), 2);

        // Asymmetric: 0 lists 1, but 1 does not list 0.
        assert!(from_rotation(vec![vec![1], vec![]]).is_none());
        // A repeated neighbour is a multigraph, which this cannot represent.
        assert!(from_rotation(vec![vec![1, 1], vec![0, 0]]).is_none());
        // A self-loop, and an index off the end.
        assert!(from_rotation(vec![vec![0], vec![]]).is_none());
        assert!(from_rotation(vec![vec![9], vec![0]]).is_none());
        // And a genuine embedding round-trips: the torus rotation validates as one.
        let t = torus_grid(4, 4).unwrap();
        let back = from_rotation((0..t.len()).map(|v| t.rotation(v).to_vec()).collect())
            .expect("the torus rotation is a valid embedding");
        assert_eq!(back.euler(), 0);
    }

    /// A disconnected graph is refused rather than half-embedded: the caller's next step is a dual,
    /// and a dual of one component is a wrong answer that looks like a right one. A cut vertex is
    /// refused too, and the two are told apart — one is an instruction to split, the other a bug.
    #[test]
    fn disconnected_and_degenerate_inputs_are_answered() {
        let mut gb = GraphBuilder::new(6);
        gb.couple(0, 1, 1.0);
        gb.couple(2, 3, 1.0);
        let g = gb.build();
        assert!(embed(&g).is_none(), "two components plus isolated vertices");
        assert_eq!(why(&g), Some(Refusal::Disconnected), "and it says which of the three it is");

        assert!(embed(&GraphBuilder::new(0).build()).unwrap().is_consistent());

        // A tree has no cycle to start from and is planar with a single face.
        let mut gb = GraphBuilder::new(5);
        for i in 1..5 {
            gb.couple(0, i, 1.0);
        }
        let star = embed(&gb.build()).expect("a star is planar");
        assert_eq!(star.faces().len(), 1);

        // A cut vertex: two triangles sharing one vertex. Planar, and beyond path addition.
        let mut gb = GraphBuilder::new(5);
        for (a, b) in [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 2)] {
            gb.couple(a, b, 1.0);
        }
        let bowtie = gb.build();
        assert!(embed(&bowtie).is_none());
        assert_eq!(
            why(&bowtie),
            Some(Refusal::NotBiconnected),
            "a bowtie IS planar; refusing it as non-planar would be a wrong answer"
        );
    }

    /// Random planar instances, built by construction so planarity is not in question, embedded and
    /// checked against Euler. A sweep rather than a handful, because the path-addition step has
    /// cases that only show up on irregular fragments.
    #[test]
    fn randomly_built_planar_graphs_all_embed_and_all_satisfy_euler() {
        use crate::rng::Pcg;
        for seed in 0..40u64 {
            let mut rng = Pcg::new(seed, 0x091A_4A00);
            let (w, h) = (3 + (rng.next_u32() % 4) as usize, 3 + (rng.next_u32() % 4) as usize);
            // A grid with some edges deleted is still planar, and its face count is not known in
            // advance -- so Euler is the only check available, which is the point.
            let mut gb = GraphBuilder::new(w * h);
            let mut kept = 0;
            for y in 0..h {
                for x in 0..w {
                    let i = y * w + x;
                    if x + 1 < w && rng.f64() < 0.85 {
                        gb.couple(i, i + 1, 1.0);
                        kept += 1;
                    }
                    if y + 1 < h && rng.f64() < 0.85 {
                        gb.couple(i, i + w, 1.0);
                        kept += 1;
                    }
                }
            }
            let g = gb.build();
            if kept == 0 {
                continue;
            }
            match embed(&g) {
                Some(e) => assert!(e.is_consistent(), "seed {seed}: {w}x{h} embedding fails Euler"),
                // Deleting edges can disconnect the grid or leave a cut vertex, and both are
                // documented refusals. What must NOT happen is a refusal reported as NotPlanar:
                // every one of these is planar by construction, so that would be a wrong answer.
                None => {
                    let r = why(&g).expect("a refusal must have a reason");
                    assert_ne!(
                        r,
                        Refusal::NotPlanar,
                        "seed {seed}: a grid subgraph is planar by construction, refused as {r}"
                    );
                }
            }
        }
    }
}
