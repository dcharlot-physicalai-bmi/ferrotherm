//! **Exact** finite-temperature Ising on a planar graph: the Kac–Ward determinant.
//!
//! [`crate::planarcut`] computes the maximum cut of a planar graph exactly, in polynomial time.
//! That is the **zero-temperature** answer — the ground state and nothing else. This module is the
//! finite-temperature companion: the full partition function
//!
//! ```text
//!   Z(β) = Σ_s exp(−β E(s)),        E(s) = − Σ_(i,j) J_ij s_i s_j
//! ```
//!
//! exactly, in `O((2E)³)`, for **any** planar graph, **any** real couplings and **any** β — no
//! sampling, no seed, no budget, no incumbent. Everything else in this crate that reports `ln Z`
//! either enumerates (`2^n`), eliminates (`2^treewidth`) or estimates it from a chain. On a planar
//! graph this is polynomial in `n` regardless of treewidth: a 30×30 lattice has treewidth 30 and is
//! hopeless for [`crate::exact`], and is a 3480×3480 determinant here.
//!
//! # The formula, and its provenance
//!
//! The high-temperature expansion turns `Z` into a sum over **even subgraphs** (every vertex of
//! even degree), with `x_e = tanh(β J_e)`:
//!
//! ```text
//!   Z = 2^V ( ∏_e cosh β J_e ) · Σ_{P even} ∏_{e∈P} x_e .
//! ```
//!
//! Kac and Ward's theorem is that on a **planar** graph the last factor is a determinant over the
//! `2E` **darts** (directed edges). Let `Λ` be indexed by darts, with
//!
//! ```text
//!   Λ_(e,f) = exp(i α(e,f) / 2) · x_f     when head(e) = tail(f) and f ≠ reverse(e), else 0,
//! ```
//!
//! where `α(e,f) ∈ (−π, π)` is the **turning angle** of a walker who arrives along `e` and leaves
//! along `f`. Then
//!
//! ```text
//!   det(I − Λ) = ( Σ_{P even} ∏_{e∈P} x_e )² .
//! ```
//!
//! * Onsager, "Crystal statistics I", Phys. Rev. 65 (1944) 117 — the square lattice solved.
//! * Kac & Ward, "A combinatorial solution of the two-dimensional Ising model", Phys. Rev. 88
//!   (1952) 1332 — the determinant.
//! * Sherman, "Combinatorial aspects of the Ising model", J. Math. Phys. 1 (1960) 202 — the loop
//!   identity that makes the sign cancellation work.
//! * Kager, Lis & Werner, "The phase transition for planar Gaussian free fields and the Kac–Ward
//!   formula", Ann. Probab. 41 (2013), arXiv:1011.3494 — the modern proof, and the statement of the
//!   theorem this module implements.
//!
//! # Why Kac–Ward and not Fisher/Kasteleyn dimers
//!
//! The other exact route is Kasteleyn's: map the even subgraphs to perfect matchings of a decorated
//! graph (Fisher's three-nodes-per-edge-end gadget), find a Pfaffian orientation, and take a
//! Pfaffian. It is equally exact and it costs three objects this crate does not have: the decorated
//! graph, **its** planar embedding (not the original one — the gadget has to be embedded inside each
//! vertex, respecting the rotation), and a Pfaffian orientation built by walking a dual spanning
//! tree. Kac–Ward needs exactly one object, the rotation system that [`crate::planar`] already
//! produces and already checks against Euler's formula, and it carries its own alarm: the
//! determinant of a correct `Λ` is **real and positive**, so its argument is a free, sensitive
//! check that the whole construction is consistent. [`Solution::phase_residual`] reports it and
//! [`Error::DeterminantNotReal`] refuses on it.
//!
//! # Where the angles come from, since a rotation system has none
//!
//! `α(e,f)` is a geometric quantity and an [`Embedding`] is a list of lists. Producing an actual
//! straight-line drawing (Fáry's theorem) is a different algorithm entirely, and it is not needed,
//! because of what the determinant is made of:
//!
//! ```text
//!   ln det(I − Λ) = − Σ_k tr(Λ^k)/k
//! ```
//!
//! and `tr(Λ^k)` sums over **closed** non-backtracking walks. So only the product of phases around
//! a closed walk ever enters, which is `exp(i Σα/2) = (−1)^turning number`. The turning number is a
//! topological invariant of the drawing, so any assignment of **corner angles** consistent with
//! discrete Gauss–Bonnet gives the same determinant:
//!
//! ```text
//!   Σ corners at a vertex   = 2π                  (the drawing closes up around each vertex)
//!   Σ corners of a face     = (c − 2)π            (an internal c-gon)
//!   Σ corners of the outer face = (c + 2)π        (it is traced the other way round)
//! ```
//!
//! Corners are in bijection with darts, each corner lies on exactly one vertex and exactly one
//! face, so this is a `b`-flow on the **radial graph** (vertices against faces), and it is solvable
//! by a spanning tree in `O(E)` — the two sides' demands agree by Euler's formula, which is the
//! solvability condition. That is [`corner_angles`], and it is the whole of the geometry.
//!
//! **The face equations are not optional, and the cheap alternative is wrong by a measurable
//! amount.** The obvious thing to do with a rotation system alone is to make every corner at a
//! degree-`d` vertex `2π/d`: it satisfies the vertex equations, it needs no solve, and it is what a
//! rotation system looks like it is telling you. It is wrong, because it ignores the edges — a
//! triangle drawn that way turns by `3 · (π − π) = 0` where the truth is `2π`. Measured on a 3×4
//! lattice at β = 0.3: equiangular gives `ln Z = 9.089027267953` against the true
//! `9.119035824602`, an error of 3.0e-2 in the second decimal — **and its determinant is still
//! real** (`arg = 3.5e-18`), so the alarm below does not fire on it. That is the reason this module
//! solves a linear system for six lines' worth of angles rather than writing down a fraction.
//!
//! **Which face is called "outer" does not matter, and that is a theorem rather than a hope.**
//! Moving the outer label changes the corner demands by `±4π` at two faces, so it changes any
//! closed walk's turning sum by a multiple of `4π` and its phase `exp(iΣα/2)` by `exp(2πik) = 1`.
//! `the_outer_face_choice_cannot_move_log_z` asserts it on every face of three graphs.
//!
//! # What it refuses
//!
//! **Fields.** The dart expansion is an expansion in `tanh β J` over even subgraphs; `h ≠ 0` is not
//! that sum. Refused by [`Error::HasFields`] naming the node, exactly as [`crate::planarcut`] does.
//!
//! **Non-planar graphs.** `K₅` and `K₃,₃` are refused by [`Error::NotEmbeddable`], which is the
//! point: the theorem is false off the sphere, and a plausible number is worse than a refusal.
//! A caller with a toroidal lattice gets the same refusal from the embedding, and passing
//! [`crate::planar::torus_grid`] to [`with_embedding`] is refused by its Euler characteristic.
//!
//! **Disconnected graphs.** [`crate::planar::embed`] refuses them, and [`with_embedding`] refuses
//! them again by Euler's formula. A face TRACE cannot know that two components share one outer
//! face — it walks darts, and each component closes its own — so a `c`-component rotation system
//! reports `χ = 2c`, and two triangles measure 4 rather than the 3 a drawing on one page would
//! have. Either way it is not 2, which is all the refusal needs. The caller's move is the easy
//! one: `ln Z` is ADDITIVE over components, so solve each and add. That is left to the caller
//! rather than done silently, because a component decomposition changes what the `N` in
//! `ln Z / N` means.
//!
//! # No quantity here is a bound
//!
//! Nothing in this module accumulates through [`crate::round`], and that is deliberate rather than
//! an oversight: `ln Z` here is an exact quantity computed in floating point, not a proved bracket,
//! and dressing it in `sum_down` would claim a guarantee that an `O((2E)³)` elimination does not
//! carry. The honest error statement is the determinant's own phase residual, which is reported.
//!
//! ```
//! use ferrotherm::{pfaffian, ising::grid2d, free_energy::exact_log_z};
//!
//! let g = grid2d(4, 4, 1.0);                       // 16 spins, planar, no field
//! let s = pfaffian::solve(&g, 0.35).expect("planar and unbiased");
//! // The same number by brute-force enumeration of all 2^16 states.
//! assert!((s.log_z - exact_log_z(&g, 0.35)).abs() < 1e-11);
//! ```

use crate::graph::Graph;
use crate::planar::{self, Embedding, Refusal};
use std::collections::BTreeMap;
use std::f64::consts::PI;

/// Why a partition function could not be computed this way.
#[derive(Clone, Debug, PartialEq)]
pub enum Error {
    /// The graph carries a field. The dart expansion is over even subgraphs and has no room for
    /// one; see the module note.
    HasFields {
        /// The first node carrying a field.
        node: usize,
        /// Its bias.
        h: f64,
    },
    /// No embedding, with the reason [`crate::planar::why`] gave.
    NotEmbeddable(Refusal),
    /// A caller-supplied embedding does not describe the sphere. `2` is planar; `0` is the torus.
    NotPlanarEmbedding {
        /// `V − E + F` for the rotation system that was passed.
        euler: i64,
    },
    /// A caller-supplied embedding is not an embedding of this graph.
    EmbeddingMismatch {
        /// Node counts: the graph's, then the embedding's.
        nodes: (usize, usize),
        /// Undirected edge counts: the graph's, then the embedding's.
        edges: (usize, usize),
    },
    /// `beta` or a coupling is not finite, so the matrix would not be either.
    NotFinite {
        /// What was passed: `"beta"`, or the two endpoints of the offending edge.
        what: &'static str,
        /// The value seen.
        value: f64,
    },
    /// The requested outer face does not exist in this embedding.
    NoSuchFace {
        /// The index asked for.
        given: usize,
        /// How many faces the embedding traces.
        faces: usize,
    },
    /// The corner-angle system did not close. An internal inconsistency, reported rather than used.
    AnglesInconsistent {
        /// The residual at the last node of the spanning tree, in radians.
        residual: f64,
    },
    /// `I − Λ` is singular, so `Z` would be zero. Only reachable at `|x| ≥ 1`, which `tanh` cannot
    /// produce for finite arguments.
    Singular,
    /// The determinant came out complex. It cannot be: it is the square of a real number. This is
    /// the construction's own alarm, and it fires on a wrong rotation system or a wrong angle.
    DeterminantNotReal {
        /// `arg det(I − Λ)` folded into `(−π, π]`, in radians.
        arg: f64,
    },
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Error::HasFields { node, h } => write!(
                f,
                "node {node} carries a field h = {h}, and the Kac-Ward expansion runs over even \
                 subgraphs of the couplings only. Absorb the field into an apex vertex (which \
                 destroys planarity) or use exact elimination"
            ),
            Error::NotEmbeddable(r) => write!(f, "no planar embedding: {r}"),
            Error::NotPlanarEmbedding { euler } => write!(
                f,
                "the embedding has Euler characteristic {euler}, not 2, so it is not a sphere; \
                 Kac-Ward is a plane statement and is false on any other surface"
            ),
            Error::EmbeddingMismatch { nodes, edges } => write!(
                f,
                "the embedding is of a different graph: {} nodes and {} edges against the graph's \
                 {} and {}",
                nodes.1, edges.1, nodes.0, edges.0
            ),
            Error::NotFinite { what, value } => {
                write!(f, "{what} is {value}, and a matrix entry cannot be built from it")
            }
            Error::NoSuchFace { given, faces } => {
                write!(f, "face {given} was requested as the outer face; the embedding traces {faces}")
            }
            Error::AnglesInconsistent { residual } => write!(
                f,
                "the corner-angle system left a residual of {residual} radians, so discrete \
                 Gauss-Bonnet did not close on this embedding"
            ),
            Error::Singular => write!(f, "I - Lambda is singular, so Z would be zero"),
            Error::DeterminantNotReal { arg } => write!(
                f,
                "det(I - Lambda) has argument {arg} radians; it is the square of a real number and \
                 must be real and positive. The rotation system or the turning angles are wrong"
            ),
        }
    }
}

impl std::error::Error for Error {}

/// The exact partition function of a planar Ising model, and what was computed to get it.
#[derive(Clone, Debug)]
pub struct Solution {
    /// `ln Z(β)`. **Exact** up to floating point, not an estimate and not a bound.
    pub log_z: f64,
    /// `⟨E⟩ = −∂ ln Z / ∂β`, by the analytic derivative `tr((I−Λ)⁻¹ ∂_β Λ)`. `None` when the
    /// caller asked only for `ln Z` — the inverse costs as much again as the determinant.
    pub energy: Option<f64>,
    /// The inverse temperature this was computed at.
    pub beta: f64,
    /// Spins.
    pub n: usize,
    /// Undirected edges. The matrix is `2 · edges` square.
    pub edges: usize,
    /// Faces the embedding traces, which is `E − V + 2`.
    pub faces: usize,
    /// `ln |det(I − Λ)|`, the whole content of the planarity argument. `ln Z` is
    /// `n ln 2 + Σ ln cosh βJ + ½` of this.
    pub log_det: f64,
    /// `arg det(I − Λ)`, folded into `(−π, π]`. **Zero in exact arithmetic**: a correct Kac–Ward
    /// determinant is the square of a real number. Carried out because it is the one number that
    /// says the angles and the rotation system were consistent, and it is a free check.
    pub phase_residual: f64,
}

impl Solution {
    /// `ln Z / N`, which is what [`crate::free_energy::onsager_log_z_density`] returns.
    #[must_use]
    pub fn log_z_density(&self) -> f64 {
        self.log_z / self.n as f64
    }

    /// The Helmholtz free energy per spin, `−ln Z / (β N)`. `None` at `β = 0`, where it diverges.
    #[must_use]
    pub fn free_energy_density(&self) -> Option<f64> {
        (self.beta != 0.0).then(|| -self.log_z / (self.beta * self.n as f64))
    }

    /// `⟨E⟩ / N`, comparable with [`crate::free_energy::onsager_energy_density`]. `None` when the
    /// internal energy was not asked for.
    #[must_use]
    pub fn energy_density(&self) -> Option<f64> {
        self.energy.map(|e| e / self.n as f64)
    }
}

/// What to compute, and on which embedding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Params {
    /// Also compute `⟨E⟩` by the analytic derivative. Costs a full inverse of the `2E × 2E` matrix,
    /// so roughly triples the work; `ln Z` alone needs only the elimination.
    pub internal_energy: bool,
    /// Which traced face to treat as the unbounded one. `None` takes the largest, which is the
    /// outer face of any drawing this crate builds. **The answer does not depend on it** (module
    /// note), and this exists so a test can prove that rather than assert it.
    pub outer_face: Option<usize>,
}

impl Default for Params {
    fn default() -> Self {
        Params { internal_energy: true, outer_face: None }
    }
}

/// `ln Z` and `⟨E⟩` for a planar, field-free Ising model at inverse temperature `beta`.
///
/// # Errors
///
/// [`Error::HasFields`] for a biased node, [`Error::NotEmbeddable`] carrying
/// [`crate::planar::Refusal`] for a graph with no planar embedding — which is the negative control
/// the module exists to pass — [`Error::NotFinite`] for a non-finite `beta` or coupling, and the
/// two self-checks [`Error::AnglesInconsistent`] and [`Error::DeterminantNotReal`].
pub fn solve(g: &Graph, beta: f64) -> Result<Solution, Error> {
    solve_with(g, beta, &Params::default())
}

/// `ln Z` alone, skipping the inverse that the internal energy needs.
///
/// # Errors
///
/// As [`solve`].
pub fn log_partition(g: &Graph, beta: f64) -> Result<f64, Error> {
    let p = Params { internal_energy: false, outer_face: None };
    solve_with(g, beta, &p).map(|s| s.log_z)
}

/// [`solve`], with the choices in [`Params`].
///
/// # Errors
///
/// As [`solve`], plus [`Error::NoSuchFace`] for an out-of-range [`Params::outer_face`].
pub fn solve_with(g: &Graph, beta: f64, p: &Params) -> Result<Solution, Error> {
    let emb = planar::embed(g)
        .ok_or_else(|| Error::NotEmbeddable(planar::why(g).unwrap_or(Refusal::NotPlanar)))?;
    with_embedding(g, &emb, beta, p)
}

/// The same, on an embedding the caller already has.
///
/// The embedding is **checked, not trusted**: it must be of this graph (same nodes, same edge set)
/// and it must describe the sphere. Handing this [`crate::planar::torus_grid`] is refused by its
/// Euler characteristic rather than answered with a number that is wrong by a factor nobody could
/// estimate.
///
/// # Errors
///
/// As [`solve_with`], plus [`Error::EmbeddingMismatch`] and [`Error::NotPlanarEmbedding`].
pub fn with_embedding(
    g: &Graph,
    emb: &Embedding,
    beta: f64,
    p: &Params,
) -> Result<Solution, Error> {
    for (i, &h) in g.h.iter().enumerate() {
        if h != 0.0 {
            return Err(Error::HasFields { node: i, h });
        }
    }
    if !beta.is_finite() {
        return Err(Error::NotFinite { what: "beta", value: beta });
    }
    let model = Model::new(g, emb)?;
    if model.darts == 0 {
        // A single isolated spin: no edges, no walks, Z = 2.
        return Ok(Solution {
            log_z: (g.n as f64) * std::f64::consts::LN_2,
            energy: p.internal_energy.then_some(0.0),
            beta,
            n: g.n,
            edges: 0,
            faces: 0,
            log_det: 0.0,
            phase_residual: 0.0,
        });
    }
    let outer = match p.outer_face {
        None => model.largest_face(),
        Some(k) if k < model.faces.len() => k,
        Some(k) => return Err(Error::NoSuchFace { given: k, faces: model.faces.len() }),
    };
    let angles = model.corner_angles(outer)?;
    let nd = model.darts;
    let x: Vec<f64> = model.weight.iter().map(|&j| (beta * j).tanh()).collect();
    let fac = model.factor(&angles, &x)?;

    let mut log_cosh = 0.0;
    for &j in &model.edge_weight {
        log_cosh += (beta * j).cosh().ln();
    }
    let log_z = (g.n as f64) * std::f64::consts::LN_2 + log_cosh + 0.5 * fac.log_det;

    let energy = if p.internal_energy {
        // ⟨E⟩ = −∂β ln Z. The determinant's share is tr(M⁻¹ ∂β M) with M = I − Λ, and
        // ∂β Λ_(e,f) = phase · J_f · (1 − x_f²), so ∂β M = −∂β Λ.
        let inv = invert(&fac.lu, &fac.perm, nd);
        let mut d_log_det = Cx::ZERO;
        for (&(e, f), &ph) in &fac.phases {
            let dx = model.weight[f] * (1.0 - x[f] * x[f]);
            d_log_det = d_log_det.sub(inv[f * nd + e].mul(ph.scale(dx)));
        }
        let mut bonds = 0.0;
        for &j in &model.edge_weight {
            bonds += j * (beta * j).tanh();
        }
        Some(-bonds - 0.5 * d_log_det.re)
    } else {
        None
    };

    Ok(Solution {
        log_z,
        energy,
        beta,
        n: g.n,
        edges: model.edge_weight.len(),
        faces: model.faces.len(),
        log_det: fac.log_det,
        phase_residual: fac.arg,
    })
}

// ---- the embedded model ------------------------------------------------------------------------

/// Darts, faces and corners: everything the determinant is indexed by.
struct Model {
    n: usize,
    darts: usize,
    /// `start[v]` is the dart id of `v`'s first neighbour in rotation order.
    start: Vec<usize>,
    /// Tail of each dart.
    tail: Vec<usize>,
    /// Head of each dart.
    head: Vec<usize>,
    /// The opposite dart.
    rev: Vec<usize>,
    /// Position of each dart within its tail's rotation.
    pos: Vec<usize>,
    /// Degree of each vertex.
    deg: Vec<usize>,
    /// `J` of the edge each dart runs along, so `weight[d] == weight[rev[d]]`.
    weight: Vec<f64>,
    /// `J` of each undirected edge, once.
    edge_weight: Vec<f64>,
    /// Faces, as dart ids in trace order.
    faces: Vec<Vec<usize>>,
    /// The face each dart belongs to.
    face_of: Vec<usize>,
}

impl Model {
    fn new(g: &Graph, emb: &Embedding) -> Result<Model, Error> {
        // The embedding must be of THIS graph and must be of the sphere.
        let mut nbr: Vec<Vec<usize>> = vec![Vec::new(); g.n];
        let mut wmap: BTreeMap<(usize, usize), f64> = BTreeMap::new();
        for u in 0..g.n {
            for k in g.offset[u]..g.offset[u + 1] {
                let v = g.nbr[k] as usize;
                if !nbr[u].contains(&v) {
                    nbr[u].push(v);
                }
                if !g.w[k].is_finite() {
                    return Err(Error::NotFinite { what: "a coupling", value: g.w[k] });
                }
                wmap.insert((u, v), g.w[k]);
            }
        }
        let g_edges: usize = nbr.iter().map(std::vec::Vec::len).sum::<usize>() / 2;
        if emb.len() != g.n || emb.edges() != g_edges {
            return Err(Error::EmbeddingMismatch {
                nodes: (g.n, emb.len()),
                edges: (g_edges, emb.edges()),
            });
        }
        for u in 0..g.n {
            let mut a = emb.rotation(u).to_vec();
            let mut b = nbr[u].clone();
            a.sort_unstable();
            b.sort_unstable();
            if a != b {
                return Err(Error::EmbeddingMismatch {
                    nodes: (g.n, emb.len()),
                    edges: (g_edges, emb.edges()),
                });
            }
        }
        if g_edges > 0 && emb.euler() != 2 {
            return Err(Error::NotPlanarEmbedding { euler: emb.euler() });
        }

        let deg: Vec<usize> = (0..g.n).map(|v| emb.rotation(v).len()).collect();
        let mut start = vec![0usize; g.n + 1];
        for v in 0..g.n {
            start[v + 1] = start[v] + deg[v];
        }
        let darts = start[g.n];
        let (mut tail, mut head, mut pos) = (vec![0; darts], vec![0; darts], vec![0; darts]);
        let mut weight = vec![0.0; darts];
        let mut id: BTreeMap<(usize, usize), usize> = BTreeMap::new();
        for v in 0..g.n {
            for (k, &u) in emb.rotation(v).iter().enumerate() {
                let d = start[v] + k;
                tail[d] = v;
                head[d] = u;
                pos[d] = k;
                weight[d] = wmap[&(v, u)];
                id.insert((v, u), d);
            }
        }
        let rev: Vec<usize> = (0..darts).map(|d| id[&(head[d], tail[d])]).collect();
        let edge_weight: Vec<f64> =
            (0..darts).filter(|&d| tail[d] < head[d]).map(|d| weight[d]).collect();

        let traced = emb.faces();
        let mut faces: Vec<Vec<usize>> = Vec::with_capacity(traced.len());
        let mut face_of = vec![usize::MAX; darts];
        for (fi, face) in traced.iter().enumerate() {
            let mut ids = Vec::with_capacity(face.len());
            for &(a, b) in face {
                let d = id[&(a, b)];
                face_of[d] = fi;
                ids.push(d);
            }
            faces.push(ids);
        }
        Ok(Model {
            n: g.n,
            darts,
            start,
            tail,
            head,
            rev,
            pos,
            deg,
            weight,
            edge_weight,
            faces,
            face_of,
        })
    }

    /// The face with the most darts; ties to the lowest index. The outer face of any drawing this
    /// crate builds, and immaterial anyway — see the module note.
    fn largest_face(&self) -> usize {
        let mut best = 0;
        for (i, f) in self.faces.iter().enumerate() {
            if f.len() > self.faces[best].len() {
                best = i;
            }
        }
        best
    }

    /// Solve discrete Gauss–Bonnet for one angle per corner.
    ///
    /// Corners are in bijection with darts: corner `d` is the wedge at `tail(d)` running
    /// counter-clockwise from `d` to the next dart in rotation. It belongs to the face whose trace
    /// contains `d`, which is exactly the face-tracing rule in [`crate::planar`].
    ///
    /// The system is a `b`-flow on the radial graph — every corner joins one vertex node to one
    /// face node — so a spanning tree solves it: fix every non-tree corner at zero and walk the
    /// tree from the leaves in. The root's equation then holds by Euler's formula, and it is
    /// checked rather than assumed.
    fn corner_angles(&self, outer: usize) -> Result<Vec<f64>, Error> {
        let nodes = self.n + self.faces.len();
        let mut demand = vec![2.0 * PI; nodes];
        for (fi, f) in self.faces.iter().enumerate() {
            let c = f.len() as f64;
            demand[self.n + fi] = if fi == outer { (c + 2.0) * PI } else { (c - 2.0) * PI };
        }
        // Radial graph: corner d joins tail(d) to n + face_of(d).
        let ends = |d: usize| (self.tail[d], self.n + self.face_of[d]);
        let mut incident: Vec<Vec<usize>> = vec![Vec::new(); nodes];
        for d in 0..self.darts {
            let (a, b) = ends(d);
            incident[a].push(d);
            incident[b].push(d);
        }
        // BFS spanning tree from node 0.
        let mut parent_corner = vec![usize::MAX; nodes];
        let mut order = Vec::with_capacity(nodes);
        let mut seen = vec![false; nodes];
        seen[0] = true;
        order.push(0);
        let mut qi = 0;
        while qi < order.len() {
            let u = order[qi];
            qi += 1;
            for &d in &incident[u] {
                let (a, b) = ends(d);
                let v = if a == u { b } else { a };
                if !seen[v] {
                    seen[v] = true;
                    parent_corner[v] = d;
                    order.push(v);
                }
            }
        }
        let mut theta = vec![0.0f64; self.darts];
        for &u in order.iter().skip(1).rev() {
            let d = parent_corner[u];
            let rest: f64 = incident[u].iter().filter(|&&c| c != d).map(|&c| theta[c]).sum();
            theta[d] = demand[u] - rest;
        }
        let root: f64 = incident[0].iter().map(|&c| theta[c]).sum::<f64>() - demand[0];
        // The root equation is the sum of every other one, so a residual here is a defect in the
        // embedding or in this routine, never in the arithmetic. The tolerance is a rounding
        // budget over 2E additions of numbers of order pi.
        let tol = 1e-9 * (self.darts as f64 + 1.0);
        if root.abs() > tol {
            return Err(Error::AnglesInconsistent { residual: root });
        }
        Ok(theta)
    }

    /// `exp(i α(e,f) / 2)` for every legal dart transition.
    ///
    /// `α` is the turn: a walker arriving along `e` at `v` is heading in the direction of
    /// `reverse(e)` reversed, so the angle it sweeps to leave along `f` is the counter-clockwise
    /// wedge from `reverse(e)` to `f`, less a straight angle. Backtracking (`f = reverse(e)`) is
    /// excluded, which is exactly the case where that wedge is the whole `2π` and the turn would be
    /// an ambiguous `±π`.
    fn phases(&self, theta: &[f64]) -> BTreeMap<(usize, usize), Cx> {
        // Prefix sums of the corner angles around each vertex, so a wedge is one subtraction.
        let mut prefix = vec![0.0f64; self.darts + self.n];
        for v in 0..self.n {
            let mut acc = 0.0;
            for k in 0..self.deg[v] {
                prefix[self.start[v] + v + k] = acc;
                acc += theta[self.start[v] + k];
            }
            prefix[self.start[v] + v + self.deg[v]] = acc;
        }
        let full = |v: usize| prefix[self.start[v] + v + self.deg[v]];
        let mut out = BTreeMap::new();
        for e in 0..self.darts {
            let v = self.head[e];
            let back = self.rev[e];
            let pb = self.pos[back];
            for k in 0..self.deg[v] {
                let f = self.start[v] + k;
                if f == back {
                    continue;
                }
                let mut wedge = prefix[self.start[v] + v + k] - prefix[self.start[v] + v + pb];
                if k < pb {
                    wedge += full(v);
                }
                // THE HALF ANGLE. `exp(i alpha / 2)`, not `exp(i alpha)`: a closed walk's turning
                // sum is 2 pi times its turning number, and it is the HALF of that which reads off
                // as the sign (-1)^turning that Sherman's identity needs.
                let turn = 0.5 * (wedge - PI);
                out.insert((e, f), Cx { re: turn.cos(), im: turn.sin() });
            }
        }
        out
    }

    /// Build `I − Λ` from an angle assignment and the activities `x_e = tanh β J_e`, and
    /// eliminate it.
    ///
    /// The phase check lives here rather than at the call site because it is a statement about
    /// THIS matrix: a correct Kac–Ward determinant is the square of a real number, so its argument
    /// is zero, and an angle assignment that violates discrete Gauss–Bonnet makes it complex. That
    /// is what `a_corrupted_corner_angle_is_caught_by_the_determinants_argument` damages on purpose.
    fn factor(&self, theta: &[f64], x: &[f64]) -> Result<Factored, Error> {
        let nd = self.darts;
        let phases = self.phases(theta);
        let mut m = vec![Cx::ZERO; nd * nd];
        for (&(e, f), &ph) in &phases {
            m[e * nd + f] = ph.scale(-x[f]);
        }
        for d in 0..nd {
            m[d * nd + d] = m[d * nd + d].add(Cx::ONE);
        }
        let (perm, swaps) = lu_factor(&mut m, nd).ok_or(Error::Singular)?;
        let (log_det, arg) = log_det_of_u(&m, nd, swaps);
        // The tolerance is a rounding budget for 2E complex eliminations, not a fudge: 1e-12 per
        // dart, against a measured residual of 1.5e-16 on a 224-dart lattice at criticality.
        if arg.abs() > 1e-12 * nd as f64 {
            return Err(Error::DeterminantNotReal { arg });
        }
        Ok(Factored { lu: m, perm, log_det, arg, phases })
    }
}

/// `I − Λ`, eliminated: everything the partition function and its derivative need.
struct Factored {
    /// The `L` and `U` factors, packed, as [`lu_factor`] leaves them.
    lu: Vec<Cx>,
    /// The row permutation partial pivoting chose.
    perm: Vec<usize>,
    /// `ln |det(I − Λ)|`.
    log_det: f64,
    /// `arg det(I − Λ)`, which is zero for a correct construction.
    arg: f64,
    /// `exp(i α / 2)` per legal transition, kept for the derivative.
    phases: BTreeMap<(usize, usize), Cx>,
}

// ---- complex arithmetic, and dense elimination over it -----------------------------------------

/// A complex number. Private: the crate has no complex type and this module needs six operations.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Cx {
    re: f64,
    im: f64,
}

impl Cx {
    const ZERO: Cx = Cx { re: 0.0, im: 0.0 };
    const ONE: Cx = Cx { re: 1.0, im: 0.0 };

    fn add(self, o: Cx) -> Cx {
        Cx { re: self.re + o.re, im: self.im + o.im }
    }

    fn sub(self, o: Cx) -> Cx {
        Cx { re: self.re - o.re, im: self.im - o.im }
    }

    fn mul(self, o: Cx) -> Cx {
        Cx { re: self.re * o.re - self.im * o.im, im: self.re * o.im + self.im * o.re }
    }

    fn scale(self, k: f64) -> Cx {
        Cx { re: self.re * k, im: self.im * k }
    }

    /// Smith's division, which scales by the larger denominator part so neither branch squares a
    /// number that could overflow.
    fn div(self, o: Cx) -> Cx {
        if o.im.abs() <= o.re.abs() {
            let r = o.im / o.re;
            let d = o.re + o.im * r;
            Cx { re: (self.re + self.im * r) / d, im: (self.im - self.re * r) / d }
        } else {
            let r = o.re / o.im;
            let d = o.re * r + o.im;
            Cx { re: (self.re * r + self.im) / d, im: (self.im * r - self.re) / d }
        }
    }

    fn norm(self) -> f64 {
        self.re.hypot(self.im)
    }
}

/// LU with partial pivoting, in place. Returns the row permutation and the number of swaps, or
/// `None` if a pivot column is entirely zero.
fn lu_factor(a: &mut [Cx], nd: usize) -> Option<(Vec<usize>, usize)> {
    let mut perm: Vec<usize> = (0..nd).collect();
    let mut swaps = 0usize;
    for k in 0..nd {
        let mut piv = k;
        let mut best = a[k * nd + k].norm();
        for r in (k + 1)..nd {
            let v = a[r * nd + k].norm();
            if v > best {
                best = v;
                piv = r;
            }
        }
        if best == 0.0 {
            return None;
        }
        if piv != k {
            for c in 0..nd {
                a.swap(k * nd + c, piv * nd + c);
            }
            perm.swap(k, piv);
            swaps += 1;
        }
        let d = a[k * nd + k];
        for r in (k + 1)..nd {
            let m = a[r * nd + k].div(d);
            a[r * nd + k] = m;
            if m == Cx::ZERO {
                continue;
            }
            for c in (k + 1)..nd {
                a[r * nd + c] = a[r * nd + c].sub(m.mul(a[k * nd + c]));
            }
        }
    }
    Some((perm, swaps))
}

/// `ln |det|` and `arg det` from a factored matrix, as sums rather than a product — the modulus of
/// a 2E-fold product overflows long before its logarithm does.
fn log_det_of_u(a: &[Cx], nd: usize, swaps: usize) -> (f64, f64) {
    let mut log_abs = 0.0;
    let mut arg = if swaps % 2 == 1 { PI } else { 0.0 };
    for k in 0..nd {
        let d = a[k * nd + k];
        log_abs += d.norm().ln();
        arg += d.im.atan2(d.re);
    }
    // Fold into (-pi, pi]: the sum of 2E arguments winds many times and only the residue is a
    // statement about the determinant.
    let tau = 2.0 * PI;
    let mut r = arg % tau;
    if r > PI {
        r -= tau;
    } else if r <= -PI {
        r += tau;
    }
    (log_abs, r)
}

/// The inverse, from the factorisation: one forward and one back substitution per column.
fn invert(lu: &[Cx], perm: &[usize], nd: usize) -> Vec<Cx> {
    let mut inv = vec![Cx::ZERO; nd * nd];
    let mut col = vec![Cx::ZERO; nd];
    for j in 0..nd {
        for r in 0..nd {
            col[r] = if perm[r] == j { Cx::ONE } else { Cx::ZERO };
        }
        for r in 1..nd {
            let mut s = col[r];
            for c in 0..r {
                s = s.sub(lu[r * nd + c].mul(col[c]));
            }
            col[r] = s;
        }
        for r in (0..nd).rev() {
            let mut s = col[r];
            for c in (r + 1)..nd {
                s = s.sub(lu[r * nd + c].mul(col[c]));
            }
            col[r] = s.div(lu[r * nd + r]);
        }
        for r in 0..nd {
            inv[r * nd + j] = col[r];
        }
    }
    inv
}

/// Corner angles for one embedding, exposed because they are the module's only geometry.
///
/// One angle per dart: the wedge at `tail(d)` from `d` counter-clockwise to the next neighbour in
/// rotation. Any solution of discrete Gauss–Bonnet gives the same partition function (module note),
/// and this returns the spanning-tree solution, which is **not** a drawing — individual angles can
/// be negative or exceed `2π`. What is guaranteed is the three sums.
///
/// # Errors
///
/// [`Error::EmbeddingMismatch`] or [`Error::NotPlanarEmbedding`] if the embedding is not one of
/// this graph on the sphere, [`Error::NoSuchFace`] for an out-of-range outer face, and
/// [`Error::AnglesInconsistent`] if the system does not close.
pub fn corner_angles(g: &Graph, emb: &Embedding, outer: Option<usize>) -> Result<Vec<f64>, Error> {
    let model = Model::new(g, emb)?;
    let face = match outer {
        None => model.largest_face(),
        Some(k) if k < model.faces.len() => k,
        Some(k) => return Err(Error::NoSuchFace { given: k, faces: model.faces.len() }),
    };
    model.corner_angles(face)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exact::Elimination;
    use crate::free_energy::{exact_log_z, onsager_energy_density, onsager_log_z_density};
    use crate::graph::GraphBuilder;
    use crate::ising::{grid2d, lattice2d};
    use crate::rng::Pcg;

    fn k5() -> Graph {
        let mut b = GraphBuilder::new(5);
        for i in 0..5 {
            for j in (i + 1)..5 {
                b.couple(i, j, 1.0);
            }
        }
        b.build()
    }

    fn k33() -> Graph {
        let mut b = GraphBuilder::new(6);
        for i in 0..3 {
            for j in 3..6 {
                b.couple(i, j, 1.0);
            }
        }
        b.build()
    }

    /// Random planar: a grid with a random subset of its edges given random couplings.
    fn random_grid(w: usize, h: usize, seed: u64) -> Graph {
        let mut r = Pcg::new(seed, 11);
        let mut b = GraphBuilder::new(w * h);
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                if x + 1 < w {
                    b.couple(i, i + 1, 2.0 * r.f64() - 1.0);
                }
                if y + 1 < h {
                    b.couple(i, i + w, 2.0 * r.f64() - 1.0);
                }
            }
        }
        b.build()
    }

    /// THE ORACLE: `ln Z` by variable elimination, which is a completely different algorithm for
    /// the same quantity — no embedding, no determinant, no complex arithmetic, just sum-product
    /// over a tree decomposition. Four temperatures, ferromagnetic and antiferromagnetic and
    /// random, on three planar shapes.
    #[test]
    fn log_z_matches_exact_elimination_on_planar_graphs() {
        let elim = Elimination::default();
        let graphs: Vec<(&str, Graph)> = vec![
            ("3x4 ferro", grid2d(3, 4, 1.0)),
            ("4x4 antiferro", grid2d(4, 4, -1.0)),
            ("5x4 random", random_grid(5, 4, 7)),
            ("6x3 random", random_grid(6, 3, 19)),
        ];
        for (name, g) in &graphs {
            for &beta in &[0.05f64, 0.3, 0.75, 1.6] {
                let got = solve(g, beta).unwrap_or_else(|e| panic!("{name} at {beta}: {e}"));
                let want = elim.log_partition(g, beta).expect("narrow enough").log_z.unwrap();
                assert!(
                    (got.log_z - want).abs() < 1e-9 * want.abs().max(1.0),
                    "{name} at beta {beta}: Kac-Ward {} vs elimination {want}",
                    got.log_z
                );
                // And the determinant really was real, which is the construction's own alarm.
                assert!(
                    got.phase_residual.abs() < 1e-12,
                    "{name} at beta {beta}: arg det = {}",
                    got.phase_residual
                );
            }
        }
    }

    /// The same against brute-force enumeration of all `2^n` states, which shares no code with
    /// elimination either — and on a graph that is NOT a lattice, so the face degrees vary.
    #[test]
    fn log_z_matches_2_to_the_n_enumeration_on_an_irregular_planar_graph() {
        // A wheel: a 7-cycle with a hub. Faces are seven triangles and one 7-gon; the hub has
        // degree 7 and the rim degree 3, so no two corners of the angle system are alike.
        let mut b = GraphBuilder::new(8);
        for i in 0..7 {
            b.couple(i, (i + 1) % 7, 0.8);
            b.couple(i, 7, -1.3);
        }
        let g = b.build();
        for &beta in &[0.2f64, 0.9] {
            let got = solve(&g, beta).expect("a wheel is planar");
            assert!(
                (got.log_z - exact_log_z(&g, beta)).abs() < 1e-11,
                "beta {beta}: {} vs {}",
                got.log_z,
                exact_log_z(&g, beta)
            );
        }
    }

    /// NEGATIVE CONTROL, and it is asymmetric: the two Kuratowski graphs are refused **by name**,
    /// and the same call on `K₅` with one edge deleted — still 3-connected, still dense, still
    /// nothing like a lattice — succeeds and agrees with enumeration. So the refusal is about
    /// planarity and not about difficulty.
    #[test]
    fn non_planar_graphs_are_refused_and_k5_minus_an_edge_is_not() {
        for (name, g) in [("K5", k5()), ("K3,3", k33())] {
            let err = solve(&g, 0.4).expect_err("must refuse");
            assert_eq!(err, Error::NotEmbeddable(Refusal::NotPlanar), "{name}");
            assert!(format!("{err}").contains("not planar"), "{name}: {err}");
        }
        let mut b = GraphBuilder::new(5);
        for i in 0..5 {
            for j in (i + 1)..5 {
                if (i, j) != (0, 1) {
                    b.couple(i, j, 0.7);
                }
            }
        }
        let g = b.build();
        let got = solve(&g, 0.4).expect("K5 minus an edge is planar");
        assert!((got.log_z - exact_log_z(&g, 0.4)).abs() < 1e-12, "{}", got.log_z);
    }

    /// A torus is not a plane, and the embedding says so. `lattice2d` is periodic, so
    /// [`crate::planar::embed`] refuses it — and if a caller hands the toroidal rotation system
    /// over anyway, Euler's formula refuses it again with the characteristic it measured.
    #[test]
    fn a_toroidal_embedding_is_refused_by_its_euler_characteristic() {
        let g = lattice2d(4, 1.0);
        assert!(matches!(solve(&g, 0.3), Err(Error::NotEmbeddable(_))), "periodic is not planar");
        let torus = planar::torus_grid(4, 4).expect("4x4 torus");
        let err = with_embedding(&g, &torus, 0.3, &Params::default()).expect_err("genus 1");
        assert_eq!(err, Error::NotPlanarEmbedding { euler: 0 });
        // The open lattice of the same size, on its own embedding, is accepted — so the refusal is
        // the surface and not the shape.
        let open = grid2d(4, 4, 1.0);
        let emb = planar::embed(&open).expect("planar");
        assert!(with_embedding(&open, &emb, 0.3, &Params::default()).is_ok());
    }

    /// A disconnected graph is refused twice over: by the embedder, which will not half-embed one,
    /// and by Euler's formula if a caller builds the rotation system themselves.
    ///
    /// The number in the assertion is MEASURED, and it is not the textbook one. A plane DRAWING of
    /// `c` components has `χ = 1 + c`, which is 3 here, because the components share one outer
    /// face. A face TRACE has no page to share: it walks darts, each component closes its own
    /// outer walk, and `χ` comes out `2c = 4`. Both refuse; the test states what the code does.
    #[test]
    fn a_disconnected_graph_is_refused_by_both_routes() {
        let mut b = GraphBuilder::new(6);
        for t in 0..2 {
            for i in 0..3 {
                b.couple(3 * t + i, 3 * t + (i + 1) % 3, 1.0);
            }
        }
        let g = b.build();
        let err = solve(&g, 0.4).expect_err("two triangles are not one graph");
        assert_eq!(err, Error::NotEmbeddable(Refusal::Disconnected));
        let rot: Vec<Vec<usize>> = (0..6)
            .map(|v| {
                let t = v / 3;
                let i = v % 3;
                vec![3 * t + (i + 1) % 3, 3 * t + (i + 2) % 3]
            })
            .collect();
        let emb = planar::from_rotation(rot).expect("a valid rotation system, just not connected");
        let err = with_embedding(&g, &emb, 0.4, &Params::default()).expect_err("two components");
        assert_eq!(err, Error::NotPlanarEmbedding { euler: 4 });
    }

    /// Which face is called "outer" is a labelling, and the module's argument says it cannot move
    /// the answer: it shifts every closed walk's turning sum by a multiple of `4π`. Asserted on
    /// EVERY face of three graphs, to machine precision.
    #[test]
    fn the_outer_face_choice_cannot_move_log_z() {
        for g in [grid2d(3, 3, 1.0), random_grid(4, 3, 5), {
            let mut b = GraphBuilder::new(6);
            for i in 0..5 {
                b.couple(i, (i + 1) % 5, 1.1);
            }
            b.couple(0, 5, -0.4);
            b.couple(2, 5, 0.9);
            b.couple(3, 5, 0.5);
            b.build()
        }] {
            let emb = planar::embed(&g).expect("planar");
            let nf = emb.faces().len();
            let base = with_embedding(&g, &emb, 0.45, &Params::default()).unwrap().log_z;
            for k in 0..nf {
                let p = Params { internal_energy: false, outer_face: Some(k) };
                let got = with_embedding(&g, &emb, 0.45, &p).unwrap().log_z;
                assert!((got - base).abs() < 1e-12, "outer face {k}: {got} vs {base}");
            }
            let p = Params { internal_energy: false, outer_face: Some(nf) };
            let err = with_embedding(&g, &emb, 0.45, &p).expect_err("there is no face nf");
            assert_eq!(err, Error::NoSuchFace { given: nf, faces: nf });
        }
    }

    /// The corner-angle system is what the whole construction stands on, so its three sums are
    /// asserted directly rather than inferred from the answer being right.
    #[test]
    fn the_corner_angles_satisfy_discrete_gauss_bonnet() {
        let g = random_grid(4, 4, 3);
        let emb = planar::embed(&g).expect("planar");
        let theta = corner_angles(&g, &emb, None).expect("solvable");
        let model = Model::new(&g, &emb).expect("valid");
        let outer = model.largest_face();
        for v in 0..model.n {
            let s: f64 = (0..model.deg[v]).map(|k| theta[model.start[v] + k]).sum();
            assert!((s - 2.0 * PI).abs() < 1e-9, "vertex {v} sums to {s}");
        }
        for (fi, f) in model.faces.iter().enumerate() {
            let s: f64 = f.iter().map(|&d| theta[d]).sum();
            let want = if fi == outer { f.len() as f64 + 2.0 } else { f.len() as f64 - 2.0 } * PI;
            assert!((s - want).abs() < 1e-9, "face {fi} sums to {s}, wanted {want}");
        }
    }

    /// `⟨E⟩` by the analytic derivative of the determinant, against a central difference of the
    /// INDEPENDENT oracle — `exact::Elimination`'s `ln Z`, differenced numerically. Two different
    /// algorithms and two different definitions of the derivative.
    #[test]
    fn internal_energy_matches_a_central_difference_of_exact_elimination() {
        let elim = Elimination::default();
        for g in [grid2d(4, 4, 1.0), random_grid(5, 3, 23)] {
            for &beta in &[0.25f64, 0.7] {
                let hstep = 1e-4;
                let up = elim.log_partition(&g, beta + hstep).unwrap().log_z.unwrap();
                let dn = elim.log_partition(&g, beta - hstep).unwrap().log_z.unwrap();
                let want = -(up - dn) / (2.0 * hstep);
                let got = solve(&g, beta).unwrap().energy.expect("asked for");
                assert!(
                    (got - want).abs() < 1e-6 * want.abs().max(1.0),
                    "beta {beta}: analytic {got} vs differenced {want}"
                );
            }
        }
    }

    /// ONSAGER. The free energy per site of a finite open lattice is `f_∞ + a/L + O(1/L²)` — the
    /// missing boundary bonds are a surface term, and they are `O(1/L)`, not `O(1/L²)`. So the test
    /// that bites is not "L is close to infinity" but Richardson's: `2 f_2L − f_L` kills the `1/L`
    /// term, and the extrapolation must land an order of magnitude closer to Onsager's closed form
    /// than either lattice it was built from.
    ///
    /// Measured at β = 0.3, J = 1 (below `β_c ≈ 0.4407`), on L = 6 and L = 12.
    #[test]
    fn the_free_energy_density_extrapolates_to_onsagers_closed_form() {
        let beta = 0.3;
        let small = log_partition(&grid2d(6, 6, 1.0), beta).unwrap() / 36.0;
        let large = log_partition(&grid2d(12, 12, 1.0), beta).unwrap() / 144.0;
        let onsager = onsager_log_z_density(beta, 1.0, 2048);
        let richardson = 2.0 * large - small;
        let (e_small, e_large, e_rich) = (
            (small - onsager).abs(),
            (large - onsager).abs(),
            (richardson - onsager).abs(),
        );
        // The raw densities are BELOW the infinite lattice, because a finite lattice is missing
        // bonds. Asserting the direction as well as the size is what makes this a physics test.
        assert!(small < onsager && large < onsager, "{small} {large} vs {onsager}");
        assert!(e_large < e_small, "a bigger lattice must be closer: {e_large} vs {e_small}");
        assert!(
            e_rich < 0.1 * e_large,
            "Richardson {richardson} (err {e_rich:.2e}) must beat L=12 (err {e_large:.2e}) by an \
             order of magnitude; if it does not, the finite-size law is not 1/L"
        );
        assert!(e_rich < 2.5e-4, "extrapolated {richardson} vs Onsager {onsager}");
    }

    /// THE THEOREM ITSELF, against an enumeration that shares no line of code with it.
    ///
    /// Kac and Ward's claim is not about `Z`, it is about the determinant: `det(I − Λ)` is the
    /// SQUARE of the sum over even subgraphs of `∏ tanh βJ_e`. That sum can be computed directly
    /// on a small graph by walking all `2^E` edge subsets and keeping the ones where every vertex
    /// has even degree — nine lines of bookkeeping, no embedding, no angles, no complex numbers.
    ///
    /// Checked on a wheel, whose 14 edges give 16384 subsets: seven triangles and a 7-gon, so the
    /// faces and the vertex degrees both vary, and the couplings are mixed-sign so the even
    /// subgraphs do not all pull the same way. Agreement is to 15 digits.
    #[test]
    fn the_determinant_is_the_squared_even_subgraph_sum_by_enumeration() {
        let mut b = GraphBuilder::new(8);
        for i in 0..7 {
            b.couple(i, (i + 1) % 7, 0.8);
            b.couple(i, 7, -1.3);
        }
        let g = b.build();
        let mut edges: Vec<(usize, usize, f64)> = Vec::new();
        for u in 0..g.n {
            for k in g.offset[u]..g.offset[u + 1] {
                let v = g.nbr[k] as usize;
                if v > u {
                    edges.push((u, v, g.w[k]));
                }
            }
        }
        assert_eq!(edges.len(), 14, "the fixture must be the wheel it claims to be");
        let mut sums = Vec::new();
        for &beta in &[0.15f64, 0.55, 1.2] {
            let mut want = 0.0f64;
            for mask in 0u32..(1 << edges.len()) {
                let mut deg = vec![0u32; g.n];
                let mut prod = 1.0f64;
                for (bit, &(u, v, w)) in edges.iter().enumerate() {
                    if mask >> bit & 1 == 1 {
                        deg[u] += 1;
                        deg[v] += 1;
                        prod *= (beta * w).tanh();
                    }
                }
                if deg.iter().all(|d| d % 2 == 0) {
                    want += prod;
                }
            }
            let got = (0.5 * solve(&g, beta).expect("planar").log_det).exp();
            assert!(
                (got - want).abs() < 1e-12 * want.abs(),
                "beta {beta}: sqrt(det) {got} vs the even-subgraph sum {want}"
            );
            sums.push(want);
        }
        // And the agreement is not the trivial one. The empty subgraph alone contributes exactly 1,
        // so a determinant that had lost every loop would read 1 at every temperature; these run
        // from 1.06 to 6.8, and they must also be increasing, because a colder model weights the
        // loops more heavily.
        assert!(sums[2] > 2.0, "the fixture must have live loops: {sums:?}");
        assert!(sums[0] < sums[1] && sums[1] < sums[2], "colder must weight loops more: {sums:?}");
    }

    /// ONSAGER AGAIN, on the internal energy this time, which is the derivative rather than the
    /// value — a different closed form ([`crate::free_energy::onsager_energy_density`], through the
    /// AGM rather than a quadrature) checking a different part of this module (the inverse of
    /// `I − Λ` rather than its determinant).
    ///
    /// Measured at β = 0.3: `u/N` is `−0.87/L` short of the infinite lattice at every size from
    /// L = 4 to L = 16, so the law is `1/L` and Richardson's `2 u_2L − u_L` applies. At L = 8, 16
    /// it lands 1.5e-3 from the closed form, thirty-five times closer than L = 16 alone.
    #[test]
    fn the_internal_energy_extrapolates_to_onsagers_closed_form() {
        let beta = 0.3;
        let want = onsager_energy_density(beta, 1.0);
        let small = solve(&grid2d(8, 8, 1.0), beta).unwrap().energy_density().unwrap();
        let large = solve(&grid2d(16, 16, 1.0), beta).unwrap().energy_density().unwrap();
        let rich = 2.0 * large - small;
        // A finite lattice is missing boundary bonds, so its energy per site is LESS negative.
        assert!(small > want && large > want, "{small} {large} vs {want}");
        assert!((large - want).abs() < (small - want).abs(), "a bigger lattice must be closer");
        assert!(
            (rich - want).abs() < 0.1 * (large - want).abs(),
            "Richardson {rich} must beat L=16 ({large}) by an order of magnitude against {want}"
        );
        assert!((rich - want).abs() < 2.5e-3, "extrapolated {rich} vs Onsager {want}");
    }

    /// THE ALARM MUST BITE. `Error::DeterminantNotReal` is unreachable through the public API by
    /// construction — a rotation system with `χ = 2` is a planar embedding, and everything else is
    /// refused earlier — so the only way to know it works is to break the angles on purpose.
    ///
    /// Asymmetric on both sides: the correct angles factor cleanly, and moving ONE corner by a
    /// tenth of a radian (which breaks Gauss–Bonnet at one vertex and one face, and nowhere else)
    /// is caught with an argument nine orders of magnitude above the tolerance.
    #[test]
    fn a_corrupted_corner_angle_is_caught_by_the_determinants_argument() {
        let g = grid2d(4, 4, 1.0);
        let emb = planar::embed(&g).expect("planar");
        let model = Model::new(&g, &emb).expect("of this graph");
        let theta = model.corner_angles(model.largest_face()).expect("solvable");
        let x: Vec<f64> = model.weight.iter().map(|&j| (0.4 * j).tanh()).collect();
        let good = model.factor(&theta, &x).expect("correct angles are accepted");
        assert!(good.arg.abs() < 1e-14, "a correct determinant is real: {}", good.arg);

        let mut bad = theta.clone();
        bad[3] += 0.1;
        match model.factor(&bad, &x) {
            Err(Error::DeterminantNotReal { arg }) => {
                assert!(arg.abs() > 1e-3, "caught, but only just: {arg}");
            }
            Err(e) => panic!("a broken angle must be refused by its argument, got {e}"),
            Ok(f) => panic!("a broken angle must be refused; arg came out {}", f.arg),
        }
    }

    /// THE COLD LIMIT, against the module this one is the finite-temperature companion to.
    ///
    /// `ln Z(β) + β E₀ → ln g₀` as `β → ∞`, where `E₀` is the ground energy and `g₀` the number of
    /// states attaining it. [`crate::planarcut`] computes `E₀` exactly by minimum-weight perfect
    /// matching on the dual — a completely different polynomial algorithm, with its own oracle —
    /// and a field-free Ising model's ground state always comes in a `±` pair, so `g₀ = 2` here.
    ///
    /// **Asymmetric, and that is the point.** At β = 2 the residual is 0.694591, which is NOT
    /// `ln 2`: a routine that had quietly returned `−βE₀ + ln 2` at every temperature would pass
    /// the cold assertions and fail the warm one. And this is where the matrix is worst
    /// conditioned — at β = 30, `tanh βJ` is one to 26 decimals and `I − Λ` is on the edge of
    /// singular — so it is also the numerical stress test: 9 digits survive.
    #[test]
    fn the_cold_limit_reproduces_planarcuts_exact_ground_energy_and_degeneracy() {
        let ln2 = std::f64::consts::LN_2;
        for g in [grid2d(4, 4, -1.0), grid2d(4, 4, 1.0), grid2d(3, 5, 1.0)] {
            let cut = crate::planarcut::solve(&g, &crate::planarcut::Params::default())
                .expect("planar and integral");
            for &beta in &[8.0f64, 16.0, 30.0] {
                let s = solve(&g, beta).expect("planar");
                let resid = s.log_z + beta * cut.energy;
                assert!(
                    (resid - ln2).abs() < 1e-8,
                    "beta {beta}: ln Z + beta E0 = {resid}, wanted ln 2 for the +- pair"
                );
                // The internal energy must have collapsed onto the ground energy itself.
                assert!(
                    (s.energy.unwrap() - cut.energy).abs() < 1e-6,
                    "beta {beta}: <E> {:?} vs the exact ground energy {}",
                    s.energy,
                    cut.energy
                );
            }
            // Warm, the same expression is NOT ln 2 -- the excited states are still paying.
            let warm = solve(&g, 2.0).expect("planar").log_z + 2.0 * cut.energy;
            assert!(
                warm - ln2 > 1e-4,
                "at beta 2 the residual {warm} must exceed ln 2; if it does not, the cold \
                 assertions above are checking an identity rather than a limit"
            );
        }
    }

    /// A tree has no closed non-backtracking walk at all, so `Λ` is nilpotent and the determinant
    /// is **exactly** one: `ln Z = n ln 2 + Σ ln cosh βJ`, with no planarity content whatever.
    /// Worth pinning because it is the one case where the hard part of this module contributes
    /// nothing, and an implementation that silently returned the trivial answer everywhere would
    /// pass it — which is why it is here beside the lattice tests and not instead of them.
    #[test]
    fn on_a_tree_the_determinant_is_exactly_one() {
        let mut b = GraphBuilder::new(7);
        for (i, j, w) in [(0, 1, 0.5), (0, 2, -0.9), (1, 3, 1.2), (1, 4, 0.3), (2, 5, 0.7), (2, 6, -0.2)]
        {
            b.couple(i, j, w);
        }
        let g = b.build();
        let s = solve(&g, 0.6).expect("a tree is planar");
        assert_eq!(s.log_det, 0.0, "ln det must be exactly zero on a tree");
        assert!((s.log_z - exact_log_z(&g, 0.6)).abs() < 1e-13);
    }

    /// Fields are a different problem and are named, not absorbed.
    #[test]
    fn a_field_is_refused_by_name() {
        let mut b = GraphBuilder::new(4);
        for i in 0..4 {
            b.couple(i, (i + 1) % 4, 1.0);
        }
        b.bias(2, -0.75);
        let g = b.build();
        let err = solve(&g, 0.3).expect_err("a biased node is a different problem");
        assert_eq!(err, Error::HasFields { node: 2, h: -0.75 });
        assert!(format!("{err}").contains("carries a field"), "{err}");
    }

    /// Infinite temperature is `Z = 2^n` and nothing else, for any couplings.
    #[test]
    fn beta_zero_is_n_ln_two() {
        let g = random_grid(4, 4, 41);
        let s = solve(&g, 0.0).expect("planar");
        assert!((s.log_z - 16.0 * std::f64::consts::LN_2).abs() < 1e-12, "{}", s.log_z);
        assert!(s.energy.unwrap().abs() < 1e-12, "no correlation at infinite temperature");
        assert_eq!(s.free_energy_density(), None, "free energy diverges at beta = 0");
    }

    /// How far an exact check REACHES is one question; how much it DISCRIMINATES is another, and
    /// the two move in opposite directions with size.
    ///
    /// Enumeration stops at `2^n`, so every small exact test in this crate lives where the oracle
    /// is cheapest. This determinant does not stop there, and running the same fixed sampler budget
    /// against it at two sizes shows what the small test was actually buying: at `n = 16` one
    /// interval covers the true lattice AND a lattice one percent colder, so agreeing with the
    /// exact answer separates almost nothing. At `n = 144` the same budget tells them apart.
    ///
    /// The mechanism is that `<E>/N` is intensive -- averaging it over more spins shrinks its error
    /// while the quantity it estimates stays put -- so signal-to-noise RISES with `n`. A small
    /// lattice is therefore the WEAKEST place to compare against an exact answer, not the
    /// strongest, which is the reverse of how a small exact test is usually read.
    ///
    /// `examples/scale` runs the full ladder to `n = 900`, where the separation reaches 8.7 sigma
    /// against an exact answer that takes 30 seconds to compute and `10^271` terms to enumerate.
    #[test]
    fn an_exact_check_discriminates_better_as_the_lattice_grows() {
        let beta = (1.0 + 2.0_f64.sqrt()).ln() / 2.0;
        let detune = 1.01;
        let plan = crate::samples::Plan::new(2_000, 4_000, 1);
        let mut separation: Vec<(usize, f64, bool)> = Vec::new();

        for l in [4usize, 12] {
            let g = crate::ising::grid2d(l, l, 1.0);
            let truth = solve(&g, beta).expect("a grid is planar");
            let colder = solve(&g, beta * detune).expect("same grid, colder");
            let e_true = truth.energy_density().expect("Params::default asks for the energy");
            let e_cold = colder.energy_density().expect("Params::default asks for the energy");

            // The premise the comparison rests on: the two lattices really are different. If the
            // detuned energy equalled the true one there would be nothing to resolve and the test
            // would pass by describing an identity.
            assert!(
                (e_cold - e_true).abs() > 1e-4,
                "1% in beta must move <E>/N at l={l}: {e_true} vs {e_cold}"
            );

            let mut s = crate::cluster::Sampler::new(&g, beta, 0x5CA1_E000 + l as u64)
                .expect("a ferromagnet is unfrustrated");
            let set = s.collect(&plan, crate::cluster::Update::SwendsenWang, None);
            let est = set.mean_energy().expect("1500 draws have a mean");
            let value = est.value / g.n as f64;
            let stderr = est.stderr / g.n as f64;

            let per_spin = crate::samples::Estimate {
                value,
                stderr,
                ess: est.ess,
                tau_int: est.tau_int,
            };
            assert!(
                per_spin.covers(e_true),
                "the sampler must agree with the exact answer at l={l}: {value} +- {stderr} vs {e_true}"
            );
            // Resolving power is a property of the DESIGN -- the size and the budget -- not of
            // one draw's luck, so the numerator is the exact gap between the two lattices rather
            // than the distance from wherever this seed happened to land. Whether the realised
            // interval also excludes the colder lattice is checked separately, below.
            separation.push((l, (e_cold - e_true).abs() / stderr, per_spin.covers(e_cold)));
        }

        let small = separation[0].1;
        let large = separation[1].1;
        assert!(
            separation[0].2,
            "at n=16 the interval is expected to cover the colder lattice too -- to be BLIND"
        );
        assert!(
            !separation[1].2,
            "at n=144 the interval must exclude the colder lattice"
        );
        assert!(
            small < 1.96,
            "at n=16 the interval is expected to be BLIND to a 1% error, and was not: {small} sigma"
        );
        assert!(
            large > 1.96,
            "at n=144 the interval must resolve a 1% error, and did not: {large} sigma"
        );
        assert!(
            large > 3.0 * small,
            "discrimination must grow substantially with size: {small} -> {large} sigma"
        );
    }
}
