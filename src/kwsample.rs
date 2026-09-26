//! Exact, independent draws from a planar Ising spin glass with exact likelihoods: the Kac–Ward
//! determinant turned into an autoregressive sampler.
//!
//! # The algorithm, and where it comes from
//!
//! Liu, Chen, Che, Wang, Deng and Zhang (arXiv:2608.24382, 2026): *"Under the chain-rule factorization,
//! sequentially fixing spins induces boundary-localized external fields, which destroy the zero-field
//! structure required for exact evaluation. By encoding these fields with a planarity-preserving
//! auxiliary spin construction, the conditional partition functions are mapped to an extended
//! zero-field Ising model and exactly evaluated using the Kac–Ward determinant formula."*
//!
//! Here: fix spins in breadth-first order, so the fixed set `F` is connected at every step. A
//! connected `F` sits inside ONE face of the drawing of the rest, so every free spin it touches lies
//! on that face's boundary, and one apex spin placed in the face can be joined to all of them without
//! a crossing. The fields the fixed spins exert become the apex's couplings; with the apex free,
//! global flip symmetry halves its partition function, so `Z(fields) = Z(apex graph) / 2`, a
//! field-free planar Ising model that [`crate::pfaffian::log_partition`] evaluates exactly. Each spin's
//! conditional is the ratio of two such evaluations, and the product of the conditionals is the exact
//! likelihood of the draw.
//!
//! # What this is for, and what it is not
//!
//! This is the paper's algorithm with a DENSE determinant: `O(N (2E)³)` per draw, against the
//! paper's `O(N^{5/2})`, which needs a nested-dissection sparse factorisation this crate does not
//! have. At sizes a dense determinant reaches, [`crate::exact`]'s backward sampler already draws
//! exact samples by elimination. What this adds is a SECOND exact sampler that shares nothing with
//! the first — planar determinants against variable elimination — so each can be held to the other
//! on graphs too large to enumerate. `examples/kwsample_exact.rs` does that.

use crate::graph::{Graph, GraphBuilder};
use crate::pfaffian::{self, Error};
use crate::rng::Pcg;

/// `ln Σ exp(−βE)` over the states agreeing with `fixed` (`None` is free), for a planar, field-free
/// graph, by the apex construction. Refused with [`Error::HasFields`] if any node carries a bias, and
/// with [`Error::NotEmbeddable`] if the apex graph is not planar -- which cannot happen when the
/// fixed set is connected in `g`, and is reported rather than guessed at when it is not.
///
/// # Errors
///
/// As [`crate::pfaffian::log_partition`], plus [`Error::HasFields`].
///
/// # Panics
///
/// If `fixed` is not one entry per node.
pub fn conditional_log_z(g: &Graph, beta: f64, fixed: &[Option<i8>]) -> Result<f64, Error> {
    assert_eq!(fixed.len(), g.n, "one entry per node");
    if let Some(i) = (0..g.n).find(|&i| g.h[i] != 0.0) {
        return Err(Error::HasFields { node: i, h: g.h[i] });
    }
    let free: Vec<usize> = (0..g.n).filter(|&i| fixed[i].is_none()).collect();
    let mut index = vec![usize::MAX; g.n];
    for (k, &i) in free.iter().enumerate() {
        index[i] = k;
    }
    let apex = free.len();
    let mut field = vec![0.0f64; free.len()];
    let mut touches = vec![false; free.len()];
    let mut constant = 0.0;
    for i in 0..g.n {
        for k in g.offset[i]..g.offset[i + 1] {
            let j = g.nbr[k] as usize;
            let w = g.w[k];
            match (fixed[i], fixed[j]) {
                (Some(a), Some(b)) if j > i => constant += w * f64::from(a) * f64::from(b),
                (None, Some(b)) => {
                    field[index[i]] += w * f64::from(b);
                    touches[index[i]] = true;
                }
                _ => {}
            }
        }
    }
    // Connected components of the free spins plus the apex, which joins every touched spin.
    let total = apex + 1;
    let mut parent: Vec<usize> = (0..total).collect();
    fn find(p: &mut [usize], x: usize) -> usize {
        let mut r = x;
        while p[r] != r {
            r = p[r];
        }
        let mut y = x;
        while p[y] != r {
            let nx = p[y];
            p[y] = r;
            y = nx;
        }
        r
    }
    let union = |p: &mut Vec<usize>, a: usize, b: usize| {
        let (ra, rb) = (find(p, a), find(p, b));
        if ra != rb {
            p[ra] = rb;
        }
    };
    for (k, &i) in free.iter().enumerate() {
        for e in g.offset[i]..g.offset[i + 1] {
            let j = g.nbr[e] as usize;
            if fixed[j].is_none() {
                union(&mut parent, k, index[j]);
            }
        }
        if touches[k] {
            union(&mut parent, k, apex);
        }
    }
    let mut members: std::collections::BTreeMap<usize, Vec<usize>> = std::collections::BTreeMap::new();
    for v in 0..total {
        let r = find(&mut parent, v);
        members.entry(r).or_default().push(v);
    }
    let mut log_z = beta * constant;
    for comp in members.values() {
        let has_apex = comp.contains(&apex);
        if comp.len() == 1 {
            if !has_apex {
                log_z += std::f64::consts::LN_2; // an isolated free spin, no field
            }
            continue;
        }
        let mut local = vec![usize::MAX; total];
        for (c, &v) in comp.iter().enumerate() {
            local[v] = c;
        }
        let mut b = GraphBuilder::new(comp.len());
        for &v in comp {
            if v == apex {
                continue;
            }
            let i = free[v];
            for e in g.offset[i]..g.offset[i + 1] {
                let j = g.nbr[e] as usize;
                if fixed[j].is_none() && index[j] > v {
                    b.couple(local[v], local[index[j]], g.w[e]);
                }
            }
            if touches[v] {
                b.couple(local[v], local[apex], field[v]);
            }
        }
        let sub = b.build();
        let lz = pfaffian::log_partition(&sub, beta)?;
        log_z += if has_apex { lz - std::f64::consts::LN_2 } else { lz };
    }
    Ok(log_z)
}

/// Breadth-first order over every component, lowest index first: each prefix is connected within
/// its component, which is what keeps the apex graph planar.
#[must_use]
pub fn connected_order(g: &Graph) -> Vec<usize> {
    let mut seen = vec![false; g.n];
    let mut order = Vec::with_capacity(g.n);
    for root in 0..g.n {
        if seen[root] {
            continue;
        }
        seen[root] = true;
        let mut queue = std::collections::VecDeque::from([root]);
        while let Some(i) = queue.pop_front() {
            order.push(i);
            for k in g.offset[i]..g.offset[i + 1] {
                let j = g.nbr[k] as usize;
                if !seen[j] {
                    seen[j] = true;
                    queue.push_back(j);
                }
            }
        }
    }
    order
}

/// One exact draw and its exact log-likelihood.
#[derive(Clone, Debug)]
pub struct Draw {
    /// The spins.
    pub s: Vec<i8>,
    /// `ln P(s)` under the Boltzmann law, as the product of the conditionals.
    pub log_prob: f64,
    /// The largest `|ln Z(prefix) − logsumexp(ln Z(prefix, +), ln Z(prefix, −))|` over the steps:
    /// the construction's own consistency, which exact arithmetic would make zero.
    pub residual: f64,
}

fn chain(g: &Graph, beta: f64, mut choose: impl FnMut(usize, f64) -> i8) -> Result<Draw, Error> {
    let mut fixed = vec![None; g.n];
    let mut parent = conditional_log_z(g, beta, &fixed)?;
    let (mut log_prob, mut residual) = (0.0, 0.0f64);
    for v in connected_order(g) {
        fixed[v] = Some(1);
        let up = conditional_log_z(g, beta, &fixed)?;
        fixed[v] = Some(-1);
        let down = conditional_log_z(g, beta, &fixed)?;
        let m = up.max(down);
        let lse = m + ((up - m).exp() + (down - m).exp()).ln();
        residual = residual.max((lse - parent).abs());
        let s = choose(v, (up - lse).exp());
        fixed[v] = Some(s);
        let chosen = if s > 0 { up } else { down };
        log_prob += chosen - lse;
        parent = chosen;
    }
    Ok(Draw { s: fixed.iter().map(|x| x.expect("every spin fixed")).collect(), log_prob, residual })
}

/// An exact independent draw.
///
/// # Errors
///
/// As [`conditional_log_z`].
pub fn sample(g: &Graph, beta: f64, rng: &mut Pcg) -> Result<Draw, Error> {
    chain(g, beta, |_, p_up| if rng.f64() < p_up { 1 } else { -1 })
}

/// The chain-rule log-likelihood the sampler assigns to `s`, with every choice forced to `s`.
///
/// # Errors
///
/// As [`conditional_log_z`].
pub fn chain_log_prob(g: &Graph, beta: f64, s: &[i8]) -> Result<f64, Error> {
    chain(g, beta, |v, _| s[v]).map(|d| d.log_prob)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ising::grid2d;

    fn glass(w: usize, h: usize, seed: u64) -> Graph {
        let mut rng = Pcg::new(seed, 0x6A);
        let mut b = GraphBuilder::new(w * h);
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                if x + 1 < w {
                    b.couple(i, i + 1, if rng.f64() < 0.5 { -1.0 } else { 1.0 } * (0.5 + rng.f64()));
                }
                if y + 1 < h {
                    b.couple(i, i + w, if rng.f64() < 0.5 { -1.0 } else { 1.0 } * (0.5 + rng.f64()));
                }
            }
        }
        b.build()
    }

    fn enumerate_log_z(g: &Graph, beta: f64, fixed: &[Option<i8>]) -> f64 {
        let mut terms = Vec::new();
        for x in 0..(1usize << g.n) {
            let s: Vec<i8> = (0..g.n).map(|i| if (x >> i) & 1 == 1 { 1 } else { -1 }).collect();
            if fixed.iter().zip(&s).all(|(f, v)| f.is_none_or(|a| a == *v)) {
                terms.push(-beta * g.energy(&s));
            }
        }
        let m = terms.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        m + terms.iter().map(|t| (t - m).exp()).sum::<f64>().ln()
    }

    /// **Every conditional partition function the sampler uses is the enumerated one.** On a 4x3
    /// glass, every breadth-first prefix with every assignment of its values: the apex construction
    /// against brute force. Also a scattered fixing, whose apex graph may not be planar: it is either
    /// exact or refused, never a plausible wrong number.
    #[test]
    fn conditional_partition_functions_match_enumeration() {
        let g = glass(4, 3, 5);
        let beta = 0.8;
        let order = connected_order(&g);
        let mut checked = 0usize;
        for len in 0..=6usize {
            for bits in 0..(1usize << len) {
                let mut fixed = vec![None; g.n];
                for (k, &v) in order[..len].iter().enumerate() {
                    fixed[v] = Some(if (bits >> k) & 1 == 1 { 1 } else { -1 });
                }
                let got = conditional_log_z(&g, beta, &fixed).expect("a connected prefix is planar");
                let want = enumerate_log_z(&g, beta, &fixed);
                assert!((got - want).abs() < 1e-9, "prefix {len}, bits {bits:b}: {got} vs {want}");
                checked += 1;
            }
        }
        assert_eq!(checked, 127);
        let mut scattered = vec![None; g.n];
        scattered[5] = Some(1);
        scattered[6] = Some(-1);
        match conditional_log_z(&g, beta, &scattered) {
            Ok(v) => assert!((v - enumerate_log_z(&g, beta, &scattered)).abs() < 1e-9),
            Err(Error::NotEmbeddable(_)) => {}
            Err(e) => panic!("unexpected {e:?}"),
        }
    }

    /// **The sampler's likelihood is the Boltzmann probability, for every state.** The product of its
    /// conditionals, forced to each of the 4,096 states of a 4x3 glass, against `−βE − ln Z` by
    /// enumeration; and the construction's own residual stays at rounding.
    #[test]
    fn the_chain_rule_likelihood_is_the_boltzmann_probability_of_every_state() {
        let g = glass(4, 3, 9);
        let beta = 1.1;
        let fixed = vec![None; g.n];
        let ln_z = enumerate_log_z(&g, beta, &fixed);
        let mut worst = 0.0f64;
        for x in 0..(1usize << g.n) {
            let s: Vec<i8> = (0..g.n).map(|i| if (x >> i) & 1 == 1 { 1 } else { -1 }).collect();
            let lp = chain_log_prob(&g, beta, &s).expect("planar");
            worst = worst.max((lp - (-beta * g.energy(&s) - ln_z)).abs());
        }
        assert!(worst < 1e-9, "worst log-likelihood error {worst:e}");
        let mut rng = Pcg::new(3, 1);
        let d = sample(&g, beta, &mut rng).expect("planar");
        assert!(d.residual < 1e-9, "residual {:e}", d.residual);
        assert!((d.log_prob - (-beta * g.energy(&d.s) - ln_z)).abs() < 1e-9);
    }

    /// Refusals: a field anywhere, and a non-planar graph.
    #[test]
    fn fields_and_non_planar_graphs_are_refused() {
        let mut g = grid2d(3, 3, 1.0);
        g.h[4] = 0.3;
        assert!(matches!(conditional_log_z(&g, 1.0, &[None; 9]), Err(Error::HasFields { node: 4, .. })));
        let mut b = GraphBuilder::new(5);
        for i in 0..5 {
            for j in i + 1..5 {
                b.couple(i, j, 1.0);
            }
        }
        let k5 = b.build();
        assert!(matches!(conditional_log_z(&k5, 1.0, &[None; 5]), Err(Error::NotEmbeddable(_))));
    }
}
