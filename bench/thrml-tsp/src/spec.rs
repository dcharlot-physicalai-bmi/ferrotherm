//! The workloads both runners execute.
//!
//! The numbers live here, once. The process writes them out as text and the Python runner
//! reads that text, so a THRML chain cannot quietly be a different model from the ferrotherm
//! chain it is scored against. Couplings are written in full. A seed is not a specification
//! when the two processes do not share a generator.

use ferrotherm::graph::GraphBuilder;
use ferrotherm::ising::grid2d;
use ferrotherm::rng::Pcg;

pub const WARMUP: usize = 400;
pub const DRAWS: usize = 2000;
pub const THIN: usize = 1;
pub const TRIALS: usize = 5;
pub const SEED0: u64 = 260_801_615;

/// Inverse-temperature and topology of one pinned case.
#[derive(Clone, Debug)]
pub struct Workload {
    pub id: &'static str,
    pub kind: Kind,
    pub beta: f64,
    pub warmup: usize,
    pub draws: usize,
    pub thin: usize,
    pub trials: usize,
    pub seed0: u64,
    /// Where the case comes from, in a form a reader can check.
    pub source: &'static str,
    pub n: usize,
    pub q: usize,
    pub biases: Vec<f64>,
    /// Pairwise terms `(i, j, J)` with `i < j`.
    pub edges: Vec<(usize, usize, f64)>,
    /// Three-body terms `(i, j, k, W)` with `i < j < k`. Empty for pairwise models.
    pub triples: Vec<(usize, usize, usize, f64)>,
    /// `None` asks the scorer to enumerate the Boltzmann law.
    /// `Some` is a mean energy the state space is too large to histogram:
    /// a transfer-matrix mean, or Onsager's infinite-lattice energy times `n`.
    pub exact_energy: Option<f64>,
    /// Gibbs blocks, outer index is the sweep order. Nodes inside a block share no edge
    /// (and, for a hypergraph, no term), so a block update is one conditional.
    pub blocks: Vec<Vec<usize>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Ising,
    Potts,
    Hubo,
}

impl Kind {
    pub fn as_str(self) -> &'static str {
        match self {
            Kind::Ising => "ising",
            Kind::Potts => "potts",
            Kind::Hubo => "hubo",
        }
    }
}

pub fn canonical(trials: usize, warmup: usize, draws: usize) -> Vec<Workload> {
    let mut out = vec![
        ising_chain(trials, warmup, draws),
        ising_grid(trials, warmup, draws),
        potts_path(trials, warmup, draws),
        three_body(trials, warmup, draws),
    ];
    for w in &mut out {
        w.trials = trials;
        w.warmup = warmup;
        w.draws = draws;
    }
    out
}

/// The five-spin open chain in the THRML 0.1.4 README and the docs quickstart:
/// `J = 0.5`, zero field, `beta = 1`, even sites then odd sites.
fn ising_chain(trials: usize, warmup: usize, draws: usize) -> Workload {
    let n = 5;
    let j = 0.5;
    let mut gb = GraphBuilder::new(n);
    let mut edges = Vec::new();
    for i in 0..n - 1 {
        gb.couple(i, i + 1, j);
        edges.push((i, i + 1, j));
    }
    let g = gb.build();
    Workload {
        id: "thrml_readme_ising_chain_n5",
        kind: Kind::Ising,
        beta: 1.0,
        warmup,
        draws,
        thin: THIN,
        trials,
        seed0: SEED0,
        source: "THRML 0.1.4 README quickstart and https://docs.thrml.ai/en/latest/ \
                 (5-spin Ising chain, J=0.5, h=0, beta=1, two-colour block Gibbs)",
        n,
        q: 2,
        biases: vec![0.0; n],
        edges,
        triples: Vec::new(),
        blocks: classes_of(&g),
        exact_energy: None,
    }
}

/// Open 3×3 Ising grid (2^9 = 512 states). A 4×4 grid is enumerable too, and a few thousand
/// draws still miss almost every state, so total variation sits near 1 for a correct sampler.
/// 3×3 is the largest open grid on which that distance is a statement about the chain.
/// `J = 0.5`, `beta = 0.8`, so `beta * J = 0.4`. Onsager's critical value on the infinite
/// lattice is about 0.4407 and is not the oracle for this finite open grid.
fn ising_grid(trials: usize, warmup: usize, draws: usize) -> Workload {
    let j = 0.5;
    let g = grid2d(3, 3, j);
    let mut edges = Vec::new();
    for i in 0..g.n {
        for k in g.offset[i]..g.offset[i + 1] {
            let jn = g.nbr[k] as usize;
            if jn > i {
                edges.push((i, jn, g.w[k]));
            }
        }
    }
    Workload {
        id: "ising_grid2d_open_3x3",
        kind: Kind::Ising,
        beta: 0.8,
        warmup,
        draws,
        thin: THIN,
        trials,
        seed0: SEED0 + 100,
        source: "Finite open 2D Ising, the lattice both codebases sample (ferrotherm::ising::grid2d; \
                 THRML IsingEBM). 3x3 rather than 4x4 so a few thousand draws can resolve total \
                 variation. Not Onsager's infinite lattice, and not a Z1-topology graph.",
        n: g.n,
        q: 2,
        biases: vec![0.0; g.n],
        edges,
        triples: Vec::new(),
        blocks: classes_of(&g),
        exact_energy: None,
    }
}

/// q=3 path of 6 sites. 3^6 = 729 states, so the Boltzmann law is enumerable.
/// The codon-optimisation walkthrough in the THRML docs is a Potts model of a real gene and is
/// not enumerable; this is that interaction (equal states coupled, unequal states not), at a
/// size where total variation against the exact law is a number rather than a hope.
fn potts_path(trials: usize, warmup: usize, draws: usize) -> Workload {
    let n = 6;
    let j = 0.4;
    let mut edges = Vec::new();
    for i in 0..n - 1 {
        edges.push((i, i + 1, j));
    }
    // ferrotherm's Potts sampler is a systematic index-order heat bath, one site at a time.
    // The blocks match that scan so the two chains are the same Gibbs schedule.
    let blocks: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
    Workload {
        id: "potts_path_q3_n6",
        kind: Kind::Potts,
        beta: 1.0,
        warmup,
        draws,
        thin: THIN,
        trials,
        seed0: SEED0 + 200,
        source: "Potts interaction as in Extropic's codon-optimisation note \
                 (https://docs.thrml.ai/en/latest/papers/codon-optimization/) and \
                 arXiv:2510.23972 Appendix C.1.1, reduced to an enumerable path. \
                 ferrotherm::potts heat-bath, index order; THRML one node per block in that order.",
        n,
        q: 3,
        biases: vec![0.0; n],
        edges,
        triples: Vec::new(),
        blocks,
        exact_energy: None,
    }
}

/// TSP §IV.4 / Appendix J.3, with two substitutions the measurement forces.
/// The paper's draw seed is not published. The paper uses d = 12 because it propagates the
/// full 4096-state law exactly; a histogram of a few thousand Gibbs draws cannot resolve that
/// space, so this pin uses d = 8 (256 states), 12 of the 28 pairs, and 10 of the 56 triples.
/// Each field and each coupling is an independent N(0, 0.6^2) draw, beta = 1, and single-site
/// conditionals are grouped into a proper colouring of the hypergraph.
///
/// This samples the target energy directly. It is not the paper's variational compilation of
/// those conditionals onto a pairwise substrate, which is what Figure 12 measures.
fn three_body(trials: usize, warmup: usize, draws: usize) -> Workload {
    const D: usize = 8;
    const N_PAIRS: usize = 12;
    const N_TRIPLES: usize = 10;
    const SIGMA: f64 = 0.6;
    let mut rng = Pcg::new(SEED0, 0x5453_5031);
    let mut pairs = combinations(D, 2);
    let mut triples = combinations(D, 3);
    shuffle(&mut rng, &mut pairs);
    shuffle(&mut rng, &mut triples);
    pairs.truncate(N_PAIRS);
    triples.truncate(N_TRIPLES);
    let mut biases = Vec::with_capacity(D);
    for _ in 0..D {
        biases.push(SIGMA * standard_normal(&mut rng));
    }
    let mut edges = Vec::with_capacity(N_PAIRS);
    for p in pairs {
        let (i, j) = (p[0], p[1]);
        edges.push((i, j, SIGMA * standard_normal(&mut rng)));
    }
    edges.sort_by_key(|e| (e.0, e.1));
    let mut trips = Vec::with_capacity(N_TRIPLES);
    for t in triples {
        let (i, j, k) = (t[0], t[1], t[2]);
        trips.push((i, j, k, SIGMA * standard_normal(&mut rng)));
    }
    trips.sort_by_key(|t| (t.0, t.1, t.2));
    let blocks = colour_hyper(D, &edges, &trips);
    Workload {
        id: "tsp_iv4_three_body_d8",
        kind: Kind::Hubo,
        beta: 1.0,
        warmup,
        draws,
        thin: THIN,
        trials,
        seed0: SEED0 + 300,
        source: "arXiv:2608.01615 §IV.4 and Appendix J.3, reduced from d=12 to d=8 so a histogram \
                 of a few thousand draws can be compared with the exact law. 12 pairs, 10 triples, \
                 weights i.i.d. N(0, 0.6^2). The paper does not publish the draw seed; this pin uses \
                 ferrotherm::rng::Pcg::new(260801615, 0x54535031), Box-Muller, and rejection-sampled \
                 Fisher-Yates. Native chromatic Gibbs of the three-body energy, not Figure 12.",
        n: D,
        q: 2,
        biases,
        edges,
        triples: trips,
        blocks,
        exact_energy: None,
    }
}

/// Scaled workloads. Not part of [`canonical`]: the pinned four stay the default comparison.
///
/// Ising sides 4, 8 and 16 carry a transfer-matrix mean (4 is also enumerable). Sides 32 and
/// 64 carry Onsager's infinite-lattice energy times `n`, which is not the finite-lattice mean.
/// Potts `q = 3` and `q = 4` carry a transfer-matrix mean. The three-body models are the
/// paper's `d = 12` law and a larger `d = 16` draw, both enumerable.
///
/// Couplings: Ising `J = 1`, `beta = 0.3` (`K = 0.3`, below Onsager's `K_c ≈ 0.4407`).
/// Potts `J = 1`, `beta = 0.5`, below [`ferrotherm::potts::critical_beta`].
pub fn scaling(trials: usize, warmup: usize, draws: usize) -> Vec<Workload> {
    let mut out = Vec::new();
    for side in [4usize, 8, 16, 32, 64] {
        out.push(ising_square(side, trials, warmup, draws));
    }
    out.push(potts_square(4, 3, trials, warmup, draws));
    out.push(potts_square(8, 3, trials, warmup, draws));
    out.push(potts_square(4, 4, trials, warmup, draws));
    out.push(potts_square(8, 4, trials, warmup, draws));
    out.push(three_body_sized(12, 18, 20, 0x5453_5032, trials, warmup, draws));
    out.push(three_body_sized(16, 24, 24, 0x5453_5033, trials, warmup, draws));
    out
}

fn ising_square(side: usize, trials: usize, warmup: usize, draws: usize) -> Workload {
    let j = 1.0;
    let beta = 0.3;
    let g = grid2d(side, side, j);
    let mut edges = Vec::new();
    for i in 0..g.n {
        for k in g.offset[i]..g.offset[i + 1] {
            let jn = g.nbr[k] as usize;
            if jn > i {
                edges.push((i, jn, g.w[k]));
            }
        }
    }
    let computed = if side <= crate::oracle::ISING_TM_MAX_SIDE {
        let tm = crate::oracle::ising_open_transfer(side, j, beta).expect("side is inside the transfer-matrix range");
        assert!(tm.log_z.is_finite(), "transfer matrix log Z");
        if side == 4 {
            let enumerated = crate::oracle::ising_enumerated_mean(side, j, beta).expect("4x4 enumerates");
            assert!((tm.mean_energy - enumerated).abs() < 1e-6, "4x4 transfer matrix left enumeration");
        }
        if side == 8 {
            let kac = crate::oracle::ising_kac_ward_energy(side, j, beta).expect("8x8 Kac-Ward");
            assert!((tm.mean_energy - kac).abs() < 1e-5, "8x8 transfer matrix left Kac-Ward");
        }
        tm.mean_energy
    } else {
        crate::oracle::onsager_total(side, j, beta)
    };
    // 4×4 is enumerable. Leave the mean unset so the scorer builds the full law.
    let exact_energy = if side <= 4 { None } else { Some(computed) };
    Workload {
        id: match side {
            4 => "ising_open_4x4",
            8 => "ising_open_8x8",
            16 => "ising_open_16x16",
            32 => "ising_open_32x32",
            64 => "ising_open_64x64",
            _ => "ising_open_scaled",
        },
        kind: Kind::Ising,
        beta,
        warmup,
        draws,
        thin: THIN,
        trials,
        seed0: SEED0 + 1_000 * side as u64,
        source: source_ising(side),
        n: g.n,
        q: 2,
        biases: vec![0.0; g.n],
        edges,
        triples: Vec::new(),
        blocks: classes_of(&g),
        exact_energy,
    }
}

fn source_ising(side: usize) -> &'static str {
    match side {
        4 => "Open 4x4 Ising, J=1, beta=0.3. Exact by enumeration (2^16) and by the transfer matrix. Checkerboard blocks are the two-colour Gibbs THRML's Ising quickstart uses (https://docs.thrml.ai/en/latest/).",
        8 => "Open 8x8 Ising, J=1, beta=0.3. exact_energy is the transfer-matrix mean, cross-checked against the Kac-Ward determinant. Checkerboard block Gibbs, as in THRML's Ising quickstart.",
        16 => "Open 16x16 Ising, J=1, beta=0.3. exact_energy is the transfer-matrix mean. A dense 2^32 transfer matrix is not this number. Checkerboard block Gibbs.",
        32 => "Open 32x32 Ising, J=1, beta=0.3. Past the transfer matrix. exact_energy is Onsager's infinite-lattice internal energy times N, not the finite-lattice mean. Checkerboard block Gibbs. The spin-model notebook (https://docs.thrml.ai/en/latest/02_spin_models.html) scales a Pegasus Ising model for throughput on GPUs; that graph is not an enumerable law.",
        _ => "Open 64x64 Ising, J=1, beta=0.3. Past the transfer matrix. exact_energy is Onsager's infinite-lattice internal energy times N, not the finite-lattice mean. A reference Gibbs chain is checked against that limit and against a second seed. Checkerboard block Gibbs.",
    }
}

fn potts_square(side: usize, q: usize, trials: usize, warmup: usize, draws: usize) -> Workload {
    let j = 1.0;
    let beta = 0.5;
    let n = side * side;
    let mut edges = Vec::new();
    for y in 0..side {
        for x in 0..side {
            let i = y * side + x;
            if x + 1 < side {
                edges.push((i, i + 1, j));
            }
            if y + 1 < side {
                edges.push((i, i + side, j));
            }
        }
    }
    let tm = crate::oracle::potts_open_transfer(side, q, j, beta).expect("potts side is inside the transfer-matrix range");
    assert!(tm.log_z.is_finite(), "potts transfer matrix log Z");
    if q == 3 && side == 4 {
        let small = crate::oracle::potts_open_transfer(3, 3, j, beta).expect("3x3 q=3");
        let enumerated = crate::oracle::potts_enumerated_mean(3, 3, j, beta).expect("3x3 q=3 enumerates");
        assert!((small.mean_energy - enumerated).abs() < 1e-6, "q=3 3x3 transfer matrix left enumeration");
    }
    let mut even = Vec::new();
    let mut odd = Vec::new();
    for i in 0..n {
        let x = i % side;
        let y = i / side;
        if (x + y).is_multiple_of(2) { even.push(i); } else { odd.push(i); }
    }
    let id: &'static str = match (q, side) {
        (3, 4) => "potts_open_q3_4x4",
        (3, 8) => "potts_open_q3_8x8",
        (4, 4) => "potts_open_q4_4x4",
        (4, 8) => "potts_open_q4_8x8",
        _ => "potts_open_scaled",
    };
    let source: &'static str = match (q, side) {
        (3, 4) => "Open 4x4 q=3 Potts, J=1, beta=0.5, below potts::critical_beta. exact_energy is the row transfer matrix. Blocks are the checkerboard, which is THRML's parallel colouring; ferrotherm's Potts heat bath still scans sites in index order.",
        (3, 8) => "Open 8x8 q=3 Potts, J=1, beta=0.5. exact_energy is the row transfer matrix (3^8 states per row). Not the codon-optimisation gene, which is not enumerable.",
        (4, 4) => "Open 4x4 q=4 Potts, J=1, beta=0.5, below potts::critical_beta(4). exact_energy is the row transfer matrix.",
        _ => "Open 8x8 q=4 Potts, J=1, beta=0.5. exact_energy is the row transfer matrix (4^8 states per row).",
    };
    Workload {
        id,
        kind: Kind::Potts,
        beta,
        warmup,
        draws,
        thin: THIN,
        trials,
        seed0: SEED0 + 4_000 + (q * 100 + side) as u64,
        source,
        n,
        q,
        biases: vec![0.0; n],
        edges,
        triples: Vec::new(),
        blocks: vec![even, odd],
        exact_energy: Some(tm.mean_energy),
    }
}

fn three_body_sized(
    d: usize,
    n_pairs: usize,
    n_triples: usize,
    stream: u64,
    trials: usize,
    warmup: usize,
    draws: usize,
) -> Workload {
    const SIGMA: f64 = 0.6;
    let mut rng = Pcg::new(SEED0, stream);
    let mut pairs = combinations(d, 2);
    let mut triples = combinations(d, 3);
    shuffle(&mut rng, &mut pairs);
    shuffle(&mut rng, &mut triples);
    pairs.truncate(n_pairs);
    triples.truncate(n_triples);
    let mut biases = Vec::with_capacity(d);
    for _ in 0..d {
        biases.push(SIGMA * standard_normal(&mut rng));
    }
    let mut edges = Vec::new();
    for p in pairs {
        edges.push((p[0], p[1], SIGMA * standard_normal(&mut rng)));
    }
    edges.sort_by_key(|e| (e.0, e.1));
    let mut trips = Vec::new();
    for t in triples {
        trips.push((t[0], t[1], t[2], SIGMA * standard_normal(&mut rng)));
    }
    trips.sort_by_key(|t| (t.0, t.1, t.2));
    let blocks = colour_hyper(d, &edges, &trips);
    let (id, source): (&'static str, &'static str) = if d == 12 {
        (
            "tsp_iv4_three_body_d12",
            "arXiv:2608.01615 §IV.4 at the paper's d=12 (4096 states), 18 pairs and 20 triples, N(0, 0.6^2), stream Pcg::new(260801615, 0x54535032). Enumerable. This is the native three-body energy, not Figure 12's compiled kernels. THRML's spin-model notebook is the same polynomial Gibbs conditional on a different graph.",
        )
    } else {
        (
            "tsp_iv4_three_body_d16",
            "Larger three-body model, d=16 (65536 states), 24 pairs and 24 triples, N(0, 0.6^2), stream Pcg::new(260801615, 0x54535033). Enumerable, so the oracle is the Boltzmann mean, not a histogram resolution of every state.",
        )
    };
    Workload {
        id,
        kind: Kind::Hubo,
        beta: 1.0,
        warmup,
        draws,
        thin: THIN,
        trials,
        seed0: SEED0 + d as u64 * 10,
        source,
        n: d,
        q: 2,
        biases,
        edges,
        triples: trips,
        blocks,
        exact_energy: None,
    }
}

fn classes_of(g: &ferrotherm::graph::Graph) -> Vec<Vec<usize>> {
    g.classes.iter().map(|c| c.iter().map(|&i| i as usize).collect()).collect()
}

fn combinations(n: usize, k: usize) -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    let mut cur = Vec::new();
    fn rec(start: usize, n: usize, k: usize, cur: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
        if cur.len() == k {
            out.push(cur.clone());
            return;
        }
        for i in start..n {
            cur.push(i);
            rec(i + 1, n, k, cur, out);
            cur.pop();
        }
    }
    rec(0, n, k, &mut cur, &mut out);
    out
}

fn shuffle(rng: &mut Pcg, xs: &mut [Vec<usize>]) {
    let n = xs.len();
    for i in (1..n).rev() {
        let j = uniform_below(rng, i + 1);
        xs.swap(i, j);
    }
}

/// Uniform in `0..n` by rejection, so the low residues are not over-represented.
fn uniform_below(rng: &mut Pcg, n: usize) -> usize {
    let n = u32::try_from(n).expect("index bound fits in u32");
    let zone = u32::MAX - (u32::MAX % n);
    loop {
        let x = rng.next_u32();
        if x < zone {
            return (x % n) as usize;
        }
    }
}

/// Standard normal via Box-Muller, using two `Pcg::f64` draws. `u1` is kept off zero so `ln` is finite.
fn standard_normal(rng: &mut Pcg) -> f64 {
    let u1 = rng.f64().max(1e-12);
    let u2 = rng.f64();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Greedy colouring in index order. Two sites that share a pair or a triple cannot share a colour.
fn colour_hyper(
    n: usize,
    edges: &[(usize, usize, f64)],
    triples: &[(usize, usize, usize, f64)],
) -> Vec<Vec<usize>> {
    let mut colour = vec![0usize; n];
    for i in 0..n {
        let mut banned = vec![false; n];
        for &(a, b, _) in edges {
            if a == i && b < i {
                banned[colour[b]] = true;
            }
            if b == i && a < i {
                banned[colour[a]] = true;
            }
        }
        for &(a, b, c, _) in triples {
            if a == i || b == i || c == i {
                for &j in &[a, b, c] {
                    if j < i {
                        banned[colour[j]] = true;
                    }
                }
            }
        }
        let mut c = 0;
        while banned[c] {
            c += 1;
        }
        colour[i] = c;
    }
    let n_c = colour.iter().copied().max().map_or(1, |c| c + 1);
    let mut blocks = vec![Vec::new(); n_c];
    for i in 0..n {
        blocks[colour[i]].push(i);
    }
    blocks
}

/// Text the Python runner parses. One workload, then its arrays, then a blank line.
pub fn write_spec(work: &[Workload]) -> String {
    let mut s = String::new();
    s.push_str("# ferrotherm thrml-tsp spec 1\n");
    for w in work {
        s.push_str(&format!(
            "WORK id={} kind={} beta={:.16} n={} q={} warmup={} draws={} thin={} trials={} seed0={}\n",
            w.id, w.kind.as_str(), w.beta, w.n, w.q, w.warmup, w.draws, w.thin, w.trials, w.seed0
        ));
        s.push_str("SOURCE ");
        s.push_str(&w.source.replace('\n', " "));
        s.push('\n');
        s.push_str("BIASES");
        for b in &w.biases {
            s.push_str(&format!(" {b:.16}"));
        }
        s.push('\n');
        s.push_str("EDGES");
        for (i, j, wj) in &w.edges {
            s.push_str(&format!(" {i},{j},{wj:.16}"));
        }
        s.push('\n');
        s.push_str("TRIPLES");
        for (i, j, k, wj) in &w.triples {
            s.push_str(&format!(" {i},{j},{k},{wj:.16}"));
        }
        s.push('\n');
        s.push_str("BLOCKS");
        for block in &w.blocks {
            s.push(' ');
            for (t, i) in block.iter().enumerate() {
                if t > 0 {
                    s.push(',');
                }
                s.push_str(&format!("{i}"));
            }
            if block.is_empty() {
                s.push('-');
            }
        }
        s.push_str("\nEND\n");
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_readme_chain_is_the_readme_chain() {
        let w = ising_chain(1, 1, 1);
        assert_eq!(w.n, 5);
        assert_eq!(w.edges.len(), 4);
        for (i, e) in w.edges.iter().enumerate() {
            assert_eq!(*e, (i, i + 1, 0.5));
        }
        assert!(w.biases.iter().all(|h| *h == 0.0));
        assert!((w.beta - 1.0).abs() < 1e-15);
        // A path is 2-colourable. Even sites first is what the README builds.
        assert_eq!(w.blocks.len(), 2);
        assert_eq!(w.blocks[0], vec![0, 2, 4]);
        assert_eq!(w.blocks[1], vec![1, 3]);
    }

    #[test]
    fn the_grid_is_the_open_grid_not_the_torus() {
        let w = ising_grid(1, 1, 1);
        assert_eq!(w.n, 9);
        // Open 3×3 has 12 edges. A torus of the same side has 18.
        assert_eq!(w.edges.len(), 12, "open grid, not a torus");
        assert_eq!(w.blocks.len(), 2, "a grid is bipartite");
    }

    #[test]
    fn the_three_body_pin_matches_the_paper_protocol() {
        let a = three_body(1, 1, 1);
        let b = three_body(1, 1, 1);
        assert_eq!(a.n, 8);
        assert_eq!(a.edges.len(), 12);
        assert_eq!(a.triples.len(), 10);
        assert_eq!(a.biases.len(), 8);
        assert_eq!(a.edges, b.edges, "the pin is a function of the seed, not of the clock");
        assert_eq!(a.triples, b.triples);
        assert_eq!(a.biases, b.biases);
        let mut seen_edges = std::collections::BTreeSet::new();
        for (i, j, _) in &a.edges {
            assert!(i < j && *j < 8);
            assert!(seen_edges.insert((*i, *j)));
        }
        let mut seen_triples = std::collections::BTreeSet::new();
        for (i, j, k, _) in &a.triples {
            assert!(i < j && j < k && *k < 8);
            assert!(seen_triples.insert((*i, *j, *k)));
        }
        // A proper colouring: no term sits inside one block.
        let colour_of = |site: usize| -> usize {
            a.blocks.iter().position(|b| b.contains(&site)).expect("site coloured")
        };
        for (i, j, _) in &a.edges {
            assert_ne!(colour_of(*i), colour_of(*j));
        }
        for (i, j, k, _) in &a.triples {
            let c = [colour_of(*i), colour_of(*j), colour_of(*k)];
            assert_ne!(c[0], c[1]);
            assert_ne!(c[0], c[2]);
            assert_ne!(c[1], c[2]);
        }
    }

    #[test]
    fn scaled_workloads_name_their_oracle() {
        let scale = scaling(2, 40, 64);
        let ids: Vec<&str> = scale.iter().map(|w| w.id).collect();
        assert_eq!(
            ids,
            vec![
                "ising_open_4x4",
                "ising_open_8x8",
                "ising_open_16x16",
                "ising_open_32x32",
                "ising_open_64x64",
                "potts_open_q3_4x4",
                "potts_open_q3_8x8",
                "potts_open_q4_4x4",
                "potts_open_q4_8x8",
                "tsp_iv4_three_body_d12",
                "tsp_iv4_three_body_d16",
            ]
        );
        let four = scale.iter().find(|w| w.id == "ising_open_4x4").unwrap();
        assert_eq!(four.n, 16);
        assert_eq!(four.edges.len(), 24, "open 4x4 has 24 edges");
        assert!(four.exact_energy.is_none(), "4x4 is enumerated");
        assert_eq!(four.blocks.len(), 2);
        let eight = scale.iter().find(|w| w.id == "ising_open_8x8").unwrap();
        assert!(eight.exact_energy.is_some());
        let big = scale.iter().find(|w| w.id == "ising_open_64x64").unwrap();
        assert_eq!(big.n, 4096);
        assert_eq!(big.edges.len(), 2 * 64 * 63);
        assert!(big.source.contains("Onsager"));
        let d12 = scale.iter().find(|w| w.id == "tsp_iv4_three_body_d12").unwrap();
        assert_eq!(d12.n, 12);
        assert_eq!(d12.edges.len(), 18);
        assert_eq!(d12.triples.len(), 20);
        let q4 = scale.iter().find(|w| w.id == "potts_open_q4_8x8").unwrap();
        assert_eq!(q4.q, 4);
        assert_eq!(q4.n, 64);
        assert!(q4.exact_energy.unwrap() < 0.0);
    }

    #[test]
    fn potts_blocks_are_the_index_scan() {
        let w = potts_path(1, 1, 1);
        assert_eq!(w.q, 3);
        assert_eq!(w.blocks, vec![vec![0], vec![1], vec![2], vec![3], vec![4], vec![5]]);
    }
}
