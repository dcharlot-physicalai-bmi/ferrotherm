//! Minimum-weight perfect matching on a general graph — Edmonds' blossom algorithm.
//!
//! The piece every polynomial max-cut algorithm on a surface eventually needs. Planar max-cut
//! reduces to a minimum-weight `T`-join in the planar dual ([`crate::planarcut`]), and a `T`-join
//! reduces to a minimum-weight perfect matching over the odd-degree vertices. Nothing else in this
//! crate can do that, and no amount of sampling substitutes for it: this is where "exact, in
//! polynomial time" comes from.
//!
//! # Why the weights are integers
//!
//! The algorithm is primal-dual. It maintains vertex duals `u_i` and blossom duals `z_B`, and
//! progress is made by shifting them by a `δ` chosen as the minimum of several slacks — then
//! relying on some slack becoming **exactly** zero so an edge becomes tight and the structure
//! changes. In floating point that equality is a coin toss: a slack that should be zero comes back
//! at `1e-17`, no edge becomes tight, `δ` is computed again, and the algorithm either loops or
//! terminates with a matching that is not optimal and says nothing about it.
//!
//! So this takes `i64` and the caller scales. Max-cut on an integer-weighted graph — which is every
//! G-set instance and most real ones — needs no scaling at all. [`crate::planarcut`] documents what
//! it does with weights that are not integers, and it is not "round and hope".
//!
//! # What it returns
//!
//! A perfect matching, or `None` when the graph has no perfect matching at all (odd order, or a
//! disconnected component of odd order once infinite-cost edges are excluded). `None` is a real
//! answer here rather than a failure: a `T`-join always exists when `|T|` is even, so a `None` from
//! this module means the caller's reduction is wrong, not that the instance is hard.
//!
//! ```
//! use ferrotherm::matching::min_weight_perfect;
//!
//! // A square: matching the two cheap opposite sides beats the two expensive ones.
//! let cost = vec![
//!     0, 1, 9, 9,
//!     1, 0, 9, 9,
//!     9, 9, 0, 1,
//!     9, 9, 1, 0,
//! ];
//! let (mate, total) = min_weight_perfect(4, &cost).expect("a perfect matching exists");
//! assert_eq!(total, 2);
//! assert_eq!(mate[0], 1);
//! assert_eq!(mate[2], 3);
//! ```

/// Cost meaning "this pair may not be matched". Large enough to lose to any real matching, small
/// enough that summing `n/2` of them cannot overflow `i64`.
pub const FORBIDDEN: i64 = 1 << 50;

/// Minimum-weight perfect matching over `n` vertices with a dense symmetric cost matrix.
///
/// `cost` is row-major `n × n`; only the off-diagonal entries are read, and `cost[i*n+j]` must
/// equal `cost[j*n+i]`. Returns `(mate, total)` where `mate[i]` is the partner of `i`, or `None`
/// when `n` is odd or no perfect matching exists over the finite-cost edges.
pub fn min_weight_perfect(n: usize, cost: &[i64]) -> Option<(Vec<usize>, i64)> {
    if n == 0 {
        return Some((Vec::new(), 0));
    }
    if n % 2 == 1 || cost.len() != n * n {
        return None;
    }
    // Maximum-weight is the form the blossom dual is usually written in, so the problem is
    // reflected rather than the algorithm rewritten: `w = M - cost` for a constant `M` above every
    // cost makes the maximum-weight perfect matching the minimum-cost one, since every perfect
    // matching has exactly `n/2` edges and the constant cancels.
    let mut hi = i64::MIN;
    for i in 0..n {
        for j in (i + 1)..n {
            if cost[i * n + j] < FORBIDDEN {
                hi = hi.max(cost[i * n + j]);
            }
        }
    }
    if hi == i64::MIN {
        return None; // every pair is forbidden: no perfect matching over the finite edges
    }
    let mut w = vec![0i64; (n + 1) * (n + 1)];
    for i in 0..n {
        for j in 0..n {
            // `+ 1`, and it is load-bearing. Weight 0 means "no edge" inside the solver, so
            // `hi - cost` silently DELETED every maximum-cost edge -- which on a complete graph
            // with repeated costs removed enough of them to destroy the perfect matching, and the
            // solver then correctly reported that none existed. The constant cancels because every
            // perfect matching has exactly `n / 2` edges. FORBIDDEN really means no edge.
            if i != j && cost[i * n + j] < FORBIDDEN {
                w[(i + 1) * (n + 1) + (j + 1)] = hi + 1 - cost[i * n + j];
            }
        }
    }
    let mate = Blossom::new(n, w).solve()?;
    let mut total = 0i64;
    for i in 0..n {
        // Each edge counted once; a vertex matched to itself would mean the matching is not perfect
        // and is rejected rather than scored.
        if mate[i] == i {
            return None;
        }
        if mate[i] > i {
            total += cost[i * n + mate[i]];
        }
    }
    Some((mate, total))
}

/// The maximum-weight perfect matching machine, 1-indexed internally.
///
/// One-indexed because 0 is the "no such node" sentinel throughout, and a sentinel that is also a
/// valid index is how this algorithm is usually got wrong.
///
/// The first version of this module was a loose paraphrase of the standard primal-dual blossom and
/// it **hung**. Four divergences, each individually plausible and none of them detectable from the
/// output: unlabelled and outer nodes shared a value so the tree could not tell a free vertex from
/// one it had already reached; `augment` walked `slack` where the structure is threaded through
/// `pa`; slacks were taken between node pairs rather than between the representative real edges
/// that blossom nodes stand for; and `add_blossom` reversed the whole petal list instead of the
/// tail after the base. This version keeps the edge triple explicit for exactly that reason —
/// `(u, v, w)` with `w = 0` meaning "no edge" — because the bugs all lived in the places where a
/// blossom node was treated as if it were a vertex.
struct Blossom {
    n: usize,
    /// Total node count including blossom nodes: `n` real plus at most `n / 2` nested blossoms.
    n_x: usize,
    /// `g[u][v] = (a, b, w)`: the representative real edge between `u` and `v`, and its weight.
    /// `w == 0` means there is no edge.
    g: Vec<Vec<(usize, usize, i64)>>,
    /// Dual variable per node.
    lab: Vec<i64>,
    /// `match_[x]` is the node matched to `x`, or 0.
    match_: Vec<usize>,
    /// `st[x]` is the outermost blossom containing `x`.
    st: Vec<usize>,
    /// Tree parent, as a real vertex.
    pa: Vec<usize>,
    /// Label. Three values, and the whole algorithm turns on them being three.
    s: Vec<u8>,
    /// For each node, the node whose edge currently gives it its least slack. 0 for none.
    slack: Vec<usize>,
    flower: Vec<Vec<usize>>,
    flower_from: Vec<Vec<usize>>,
    q: Vec<usize>,
}

const OUTER: u8 = 0;
const INNER: u8 = 1;
const UNLABELLED: u8 = 2;

impl Blossom {
    fn new(n: usize, w: Vec<i64>) -> Blossom {
        let m = n * 2 + 1;
        let mut g = vec![vec![(0usize, 0usize, 0i64); m]; m];
        for u in 1..=n {
            for v in 1..=n {
                g[u][v] = (u, v, w[u * (n + 1) + v]);
            }
        }
        Blossom {
            n,
            n_x: n,
            g,
            lab: vec![0; m],
            match_: vec![0; m],
            st: vec![0; m],
            pa: vec![0; m],
            s: vec![UNLABELLED; m],
            slack: vec![0; m],
            flower: vec![Vec::new(); m],
            flower_from: vec![vec![0; n + 1]; m],
            q: Vec::new(),
        }
    }

    /// Slack of an edge, **doubled**, so every quantity in the dual update stays an integer: the
    /// blossom dual moves by `δ/2`, and halving an odd integer is where an integer implementation
    /// silently becomes a wrong one.
    #[inline]
    fn e_delta(&self, e: (usize, usize, i64)) -> i64 {
        self.lab[e.0] + self.lab[e.1] - e.2 * 2
    }

    fn update_slack(&mut self, u: usize, x: usize) {
        if self.slack[x] == 0 || self.e_delta(self.g[u][x]) < self.e_delta(self.g[self.slack[x]][x])
        {
            self.slack[x] = u;
        }
    }

    fn set_slack(&mut self, x: usize) {
        self.slack[x] = 0;
        for u in 1..=self.n {
            if self.g[u][x].2 > 0 && self.st[u] != x && self.s[self.st[u]] == OUTER {
                self.update_slack(u, x);
            }
        }
    }

    fn q_push(&mut self, x: usize) {
        if x <= self.n {
            self.q.push(x);
        } else {
            for i in 0..self.flower[x].len() {
                let f = self.flower[x][i];
                self.q_push(f);
            }
        }
    }

    fn set_st(&mut self, x: usize, b: usize) {
        self.st[x] = b;
        if x > self.n {
            for i in 0..self.flower[x].len() {
                let f = self.flower[x][i];
                self.set_st(f, b);
            }
        }
    }

    /// Position of `xr` in blossom `b`'s petal list, along the ALTERNATING direction.
    ///
    /// A blossom is an odd cycle, so exactly one of the two ways round leaves an even number of
    /// edges between `xr` and the base. When the direct index is odd the list is reversed from the
    /// base onward — `[1..]`, never `[0..]`, because the base must stay first.
    fn get_pr(&mut self, b: usize, xr: usize) -> usize {
        let pr = self.flower[b].iter().position(|&v| v == xr).expect("xr is a petal of b");
        if pr % 2 == 1 {
            let k = self.flower[b].len();
            self.flower[b][1..].reverse();
            k - pr
        } else {
            pr
        }
    }

    fn set_match(&mut self, u: usize, v: usize) {
        let e = self.g[u][v];
        self.match_[u] = e.1;
        if u <= self.n {
            return;
        }
        let xr = self.flower_from[u][e.0];
        let pr = self.get_pr(u, xr);
        for i in 0..pr {
            let a = self.flower[u][i];
            let b = self.flower[u][i ^ 1];
            self.set_match(a, b);
        }
        self.set_match(xr, v);
        self.flower[u].rotate_left(pr);
    }

    fn augment(&mut self, mut u: usize, mut v: usize) {
        loop {
            let xnv = self.st[self.match_[u]];
            self.set_match(u, v);
            if xnv == 0 {
                return;
            }
            // The CONTAINING BLOSSOM, not the real vertex. `g[xnv][·]` is indexed by the outer
            // node on the far side, so passing `pa[xnv]` reads a different edge -- one whose
            // endpoint need not lie in `xnv` at all, which surfaces as `xr is a petal of b`
            // hundreds of nodes later. Invisible below ~400 vertices, where blossoms rarely nest.
            let t = self.st[self.pa[xnv]];
            self.set_match(xnv, t);
            u = t;
            v = xnv;
        }
    }

    fn lca(&mut self, mut u: usize, mut v: usize) -> usize {
        // A fresh sweep rather than a shared timestamp: the stamp is one more piece of state to get
        // wrong, and this runs O(n) times, not O(n^3).
        let mut seen = vec![false; self.n_x + 1];
        loop {
            while u != 0 {
                if seen[u] {
                    return u;
                }
                seen[u] = true;
                u = self.st[self.match_[u]];
                if u != 0 {
                    u = self.st[self.pa[u]];
                }
                core::mem::swap(&mut u, &mut v);
            }
            if v == 0 {
                return 0;
            }
            core::mem::swap(&mut u, &mut v);
        }
    }

    fn add_blossom(&mut self, u: usize, lca: usize, v: usize) {
        let mut b = self.n + 1;
        while b <= self.n_x && self.st[b] != 0 {
            b += 1;
        }
        if b > self.n_x {
            self.n_x += 1;
        }
        self.lab[b] = 0;
        self.s[b] = OUTER;
        self.match_[b] = self.match_[lca];
        self.flower[b].clear();
        self.flower[b].push(lca);

        let mut x = u;
        while x != lca {
            let y = self.st[self.match_[x]];
            self.flower[b].push(x);
            self.flower[b].push(y);
            self.q_push(y);
            x = self.st[self.pa[y]];
        }
        // From index 1, not 0: the base stays at the front and only the arm is reversed.
        self.flower[b][1..].reverse();
        let mut x = v;
        while x != lca {
            let y = self.st[self.match_[x]];
            self.flower[b].push(x);
            self.flower[b].push(y);
            self.q_push(y);
            x = self.st[self.pa[y]];
        }
        self.set_st(b, b);
        for x in 1..=self.n_x {
            self.g[b][x].2 = 0;
            self.g[x][b].2 = 0;
        }
        for x in 1..=self.n {
            self.flower_from[b][x] = 0;
        }
        let petals = self.flower[b].clone();
        for &xs in &petals {
            for x in 1..=self.n_x {
                if self.g[b][x].2 == 0 || self.e_delta(self.g[xs][x]) < self.e_delta(self.g[b][x]) {
                    self.g[b][x] = self.g[xs][x];
                    self.g[x][b] = self.g[x][xs];
                }
            }
            for x in 1..=self.n {
                if self.flower_from[xs][x] != 0 {
                    self.flower_from[b][x] = xs;
                }
            }
        }
        self.set_slack(b);
    }

    fn expand_blossom(&mut self, b: usize) {
        let petals = self.flower[b].clone();
        for &f in &petals {
            self.set_st(f, f);
        }
        let xr = self.flower_from[b][self.g[b][self.pa[b]].0];
        let pr = self.get_pr(b, xr);
        let mut i = 0;
        while i < pr {
            let xs = self.flower[b][i];
            let xns = self.flower[b][i + 1];
            self.pa[xs] = self.g[xns][xs].0;
            self.s[xs] = INNER;
            self.s[xns] = OUTER;
            self.slack[xs] = 0;
            self.set_slack(xns);
            self.q_push(xns);
            i += 2;
        }
        self.s[xr] = INNER;
        self.pa[xr] = self.pa[b];
        for i in (pr + 1)..self.flower[b].len() {
            let xs = self.flower[b][i];
            self.s[xs] = UNLABELLED;
            self.set_slack(xs);
        }
        self.st[b] = 0;
    }

    fn on_found_edge(&mut self, e: (usize, usize, i64)) -> bool {
        let (xu, xv) = (self.st[e.0], self.st[e.1]);
        if self.s[xv] == UNLABELLED {
            self.pa[xv] = e.0;
            self.s[xv] = INNER;
            let nu = self.st[self.match_[xv]];
            self.slack[xv] = 0;
            self.slack[nu] = 0;
            self.s[nu] = OUTER;
            self.q_push(nu);
        } else if self.s[xv] == OUTER {
            let lca = self.lca(xu, xv);
            if lca == 0 {
                self.augment(xu, xv);
                self.augment(xv, xu);
                return true;
            }
            self.add_blossom(xu, lca, xv);
        }
        false
    }

    fn matching(&mut self) -> bool {
        for x in 0..=self.n_x {
            self.s[x] = UNLABELLED;
            self.slack[x] = 0;
        }
        self.q.clear();
        for x in 1..=self.n_x {
            if self.st[x] == x && self.match_[x] == 0 {
                self.pa[x] = 0;
                self.s[x] = OUTER;
                self.q_push(x);
            }
        }
        if self.q.is_empty() {
            return false;
        }
        loop {
            while let Some(u) = self.q.pop() {
                if self.s[self.st[u]] == INNER {
                    continue;
                }
                for v in 1..=self.n {
                    if self.g[u][v].2 > 0 && self.st[u] != self.st[v] {
                        if self.e_delta(self.g[u][v]) == 0 {
                            if self.on_found_edge(self.g[u][v]) {
                                return true;
                            }
                        } else {
                            let x = self.st[v];
                            self.update_slack(u, x);
                        }
                    }
                }
            }
            // ---- dual adjustment ------------------------------------------------------------
            let mut d = i64::MAX;
            for b in (self.n + 1)..=self.n_x {
                if self.st[b] == b && self.s[b] == INNER {
                    d = d.min(self.lab[b] / 2);
                }
            }
            for x in 1..=self.n_x {
                if self.st[x] == x && self.slack[x] != 0 {
                    let sl = self.e_delta(self.g[self.slack[x]][x]);
                    match self.s[x] {
                        UNLABELLED => d = d.min(sl),
                        OUTER => d = d.min(sl / 2),
                        _ => {}
                    }
                }
            }
            for u in 1..=self.n {
                match self.s[self.st[u]] {
                    OUTER => {
                        // A dual that would go non-positive means no perfect matching exists over
                        // the finite-weight edges. Returning here is the only exit that is not an
                        // augmentation, and it has to be checked BEFORE the subtraction.
                        if self.lab[u] <= d {
                            return false;
                        }
                        self.lab[u] -= d;
                    }
                    INNER => self.lab[u] += d,
                    _ => {}
                }
            }
            for b in (self.n + 1)..=self.n_x {
                if self.st[b] == b {
                    match self.s[b] {
                        OUTER => self.lab[b] += d * 2,
                        INNER => self.lab[b] -= d * 2,
                        _ => {}
                    }
                }
            }
            self.q.clear();
            for x in 1..=self.n_x {
                if self.st[x] == x
                    && self.slack[x] != 0
                    && self.st[self.slack[x]] != x
                    && self.e_delta(self.g[self.slack[x]][x]) == 0
                    && self.on_found_edge(self.g[self.slack[x]][x])
                {
                    return true;
                }
            }
            for b in (self.n + 1)..=self.n_x {
                if self.st[b] == b && self.s[b] == INNER && self.lab[b] == 0 {
                    self.expand_blossom(b);
                }
            }
        }
    }

    fn solve(mut self) -> Option<Vec<usize>> {
        let n = self.n;
        for u in 0..=n {
            self.match_[u] = 0;
            self.st[u] = u;
        }
        self.n_x = n;
        let mut w_max = 0i64;
        for u in 1..=n {
            for v in 1..=n {
                self.flower_from[u][v] = if u == v { u } else { 0 };
                w_max = w_max.max(self.g[u][v].2);
            }
        }
        for u in 1..=n {
            self.lab[u] = w_max;
        }
        // One augmentation per two vertices, and no more: an extra round would mean the invariant
        // "every phase augments" is broken, and looping forever is how that presents.
        for _ in 0..(n / 2) {
            if !self.matching() {
                return None;
            }
        }
        let mut mate = vec![usize::MAX; n];
        for u in 1..=n {
            let m = self.match_[u];
            if m == 0 || m > n {
                return None;
            }
            mate[u - 1] = m - 1;
        }
        for u in 0..n {
            if mate[mate[u]] != u {
                return None; // not an involution: not a matching
            }
        }
        Some(mate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Pcg;

    /// Every perfect matching, by brute force. The only thing that can check a blossom
    /// implementation without being one.
    fn brute(n: usize, cost: &[i64]) -> Option<i64> {
        fn go(n: usize, cost: &[i64], used: &mut Vec<bool>, k: usize, acc: i64, best: &mut i64) {
            if k == n {
                *best = (*best).min(acc);
                return;
            }
            if used[k] {
                go(n, cost, used, k + 1, acc, best);
                return;
            }
            used[k] = true;
            for j in (k + 1)..n {
                if !used[j] && cost[k * n + j] < FORBIDDEN {
                    used[j] = true;
                    go(n, cost, used, k + 1, acc + cost[k * n + j], best);
                    used[j] = false;
                }
            }
            used[k] = false;
        }
        if n % 2 == 1 {
            return None;
        }
        let mut best = i64::MAX;
        go(n, cost, &mut vec![false; n], 0, 0, &mut best);
        (best != i64::MAX).then_some(best)
    }

    fn random_cost(n: usize, seed: u64, hi: i64) -> Vec<i64> {
        let mut rng = Pcg::new(seed, 0x0B10_5503);
        let mut c = vec![0i64; n * n];
        for i in 0..n {
            for j in (i + 1)..n {
                let v = (rng.next_u32() as i64) % hi;
                c[i * n + j] = v;
                c[j * n + i] = v;
            }
        }
        c
    }

    /// THE TEST THIS MODULE STANDS ON. A wrong blossom does not raise anything: it returns a
    /// perfect matching that is merely not the cheapest, and every reduction built on top inherits
    /// a wrong answer that looks exactly like a right one.
    #[test]
    fn it_agrees_with_exhaustive_enumeration() {
        for n in [2usize, 4, 6, 8, 10] {
            for seed in 0..40u64 {
                let c = random_cost(n, seed + n as u64 * 1000, 100);
                let (mate, total) = min_weight_perfect(n, &c).expect("complete graph, even order");
                let truth = brute(n, &c).expect("complete graph, even order");
                assert_eq!(
                    total, truth,
                    "n={n} seed={seed}: blossom {total}, exhaustive {truth}"
                );
                // And the thing returned really is a perfect matching, scored by its own edges.
                let mut sum = 0;
                for i in 0..n {
                    assert_eq!(mate[mate[i]], i, "not an involution at {i}");
                    assert_ne!(mate[i], i);
                    if mate[i] > i {
                        sum += c[i * n + mate[i]];
                    }
                }
                assert_eq!(sum, total, "the reported total is not the matching's weight");
            }
        }
    }

    /// Blossoms exist because odd cycles break the naive augmenting-path argument. An instance
    /// whose optimum forces one is the case a non-blossom implementation gets wrong, so it is worth
    /// its own test rather than trusting the random sweep to have hit one.
    #[test]
    fn an_odd_cycle_does_not_defeat_it() {
        // Two triangles joined by a cheap bridge: the optimum has to use the bridge and one edge
        // from each triangle, which is exactly the configuration a blossom contracts.
        let n = 6;
        let mut c = vec![50i64; n * n];
        for i in 0..n {
            c[i * n + i] = 0;
        }
        let set = |a: usize, b: usize, v: i64, c: &mut Vec<i64>| {
            c[a * n + b] = v;
            c[b * n + a] = v;
        };
        for (a, b) in [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)] {
            set(a, b, 10, &mut c);
        }
        set(2, 3, 1, &mut c);
        let (_, total) = min_weight_perfect(n, &c).unwrap();
        assert_eq!(total, brute(n, &c).unwrap());
        assert_eq!(total, 21, "bridge 1 + one edge from each triangle 10 + 10");
    }

    /// Forbidden edges must actually forbid, and an instance with no perfect matching must come
    /// back as `None` rather than as a matching that quietly uses one.
    #[test]
    fn a_missing_perfect_matching_is_reported_rather_than_invented() {
        // Three vertices reachable only through one hub: no perfect matching exists.
        let n = 4;
        let mut c = vec![FORBIDDEN; n * n];
        for i in 0..n {
            c[i * n + i] = 0;
        }
        // Written out rather than `0 * n + 1` and `1 * n + 0`: the row-major form reads better
        // beside its neighbours, and clippy is right that those two expressions are noise.
        c[1] = 1;
        c[n] = 1;
        // 2 and 3 are isolated from everything, including each other.
        assert!(min_weight_perfect(n, &c).is_none(), "there is no perfect matching here");

        // And with the last pair connected, there is one.
        c[2 * n + 3] = 5;
        c[3 * n + 2] = 5;
        let (_, total) = min_weight_perfect(n, &c).unwrap();
        assert_eq!(total, 6);
    }

    #[test]
    fn odd_order_and_empty_are_answered_not_attempted() {
        assert!(min_weight_perfect(3, &[0i64; 9]).is_none(), "odd order has no perfect matching");
        assert_eq!(min_weight_perfect(0, &[]), Some((Vec::new(), 0)));
        assert!(min_weight_perfect(4, &[0i64; 3]).is_none(), "a cost matrix of the wrong shape");
    }

    /// A PLANTED optimum, at sizes exhaustive enumeration cannot reach.
    ///
    /// Brute force stops at about 14 vertices; the defect this test exists for first appeared at
    /// roughly 700, in `augment`, where the containing blossom and the real vertex were confused.
    /// Nothing below 400 exercised nested blossoms hard enough to notice. So: pair the vertices up
    /// at cost 0 and price every other pair at 1. The optimum is exactly 0, any deviation is
    /// visible, and the instance is as large as we like.
    #[test]
    fn a_planted_optimum_is_found_at_sizes_brute_force_cannot_reach() {
        for n in [50usize, 200, 500] {
            let mut rng = Pcg::new(n as u64, 0x091A_47ED);
            // A random pairing, so the answer is not the identity ordering.
            let mut perm: Vec<usize> = (0..n).collect();
            for i in (1..n).rev() {
                let j = (rng.next_u32() as usize) % (i + 1);
                perm.swap(i, j);
            }
            let mut c = vec![1i64; n * n];
            for i in 0..n {
                c[i * n + i] = 0;
            }
            for k in (0..n).step_by(2) {
                let (a, b) = (perm[k], perm[k + 1]);
                c[a * n + b] = 0;
                c[b * n + a] = 0;
            }
            let (mate, total) = min_weight_perfect(n, &c).expect("a perfect matching is planted");
            assert_eq!(total, 0, "n={n}: the planted matching costs 0 and this found {total}");
            for i in 0..n {
                assert_eq!(mate[mate[i]], i);
                assert_eq!(c[i * n + mate[i]], 0, "n={n}: vertex {i} matched off the plant");
            }
        }
    }

    /// Negative weights are legal and the reduction to maximum-weight has to survive them: the
    /// constant it subtracts is chosen from the largest cost, not from zero.
    #[test]
    fn negative_costs_are_handled_rather_than_assumed_away() {
        for n in [4usize, 6, 8] {
            for seed in 0..20u64 {
                let mut c = random_cost(n, seed + 77, 40);
                for v in c.iter_mut() {
                    *v -= 20;
                }
                for i in 0..n {
                    c[i * n + i] = 0;
                }
                let (_, total) = min_weight_perfect(n, &c).unwrap();
                assert_eq!(total, brute(n, &c).unwrap(), "n={n} seed={seed}");
            }
        }
    }
}
