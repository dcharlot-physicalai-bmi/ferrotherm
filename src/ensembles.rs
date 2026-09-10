//! The instance families the benchmarking literature actually uses.
//!
//! [`crate::planted`] has frustrated loops and the Wishart ensemble. This module adds the other
//! four standard generators, in two groups that answer different questions.
//!
//! **No planted optimum.** [`sherrington_kirkpatrick`] (dense Gaussian couplings, the mean-field
//! spin glass) and [`edwards_anderson_3d`] (cubic lattice, `+/-J` or Gaussian). These are the
//! canonical hard instances, and their ground states are *not* known — which is exactly the
//! problem: past about twenty-five spins nothing here can say whether a solver found the optimum.
//! They are for measuring cost, scaling and agreement between methods, not correctness.
//!
//! **Ground state known by construction.** [`tile_planted_2d`] and [`deceptive_cluster_loops`],
//! the two remaining pairwise families of Chook (Perera, Hamze, Raymond, Weigel & Katzgraber, and
//! Hamze, Jacob, Ochoa, Perera, Wang & Katzgraber); its `k`-local family is not pairwise and does
//! not belong here. Both work the same way: write the energy as a sum of
//! local terms, choose a state first, and build every term so that state minimises it. A sum of
//! terms each individually minimised by one state is minimised by that state, whatever the terms
//! overlap. The ground energy is then a closed form.
//!
//! The tests do not take that argument on trust. Every planted claim here is checked against
//! [`crate::exact::Elimination::ground_state`] or exhaustive enumeration at sizes where those run.

use crate::graph::{Graph, GraphBuilder};
use crate::planted::Planted;
use crate::rng::Pcg;

/// Standard normal, Box-Muller. Private in [`crate::planted`], so duplicated rather than exported.
fn gauss(rng: &mut Pcg) -> f64 {
    let u = rng.f64().max(1e-15);
    let v = rng.f64();
    (-2.0 * u.ln()).sqrt() * (core::f64::consts::TAU * v).cos()
}

/// How a bond is drawn: the two standard Edwards-Anderson disorder distributions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bonds {
    /// Uniform on `{-1, +1}`. Discrete, so the spectrum is integer-valued and highly degenerate.
    PlusMinusJ,
    /// Standard normal. Continuous, so the ground state is almost surely unique up to global flip.
    Gaussian,
}

/// The Sherrington-Kirkpatrick model: every pair coupled, `J_ij` drawn `N(0, 1/n)`, no field.
///
/// The mean-field spin glass, and the standard dense benchmark. The `1/n` variance is the scaling
/// that makes the energy extensive, so the ground energy per spin has a finite limit
/// (`-0.7633...`, Parisi) rather than growing with `n`.
///
/// The optimum is NOT known — that is the point of the family. For a dense instance whose optimum
/// is known, use [`crate::planted::wishart`].
///
/// # Panics
///
/// If `n` is below 2, where there is no pair to couple.
#[must_use]
pub fn sherrington_kirkpatrick(n: usize, seed: u64) -> Graph {
    assert!(n >= 2, "a Sherrington-Kirkpatrick instance needs at least one pair");
    let mut rng = Pcg::new(seed, 0x5B_0000);
    let scale = 1.0 / (n as f64).sqrt();
    let mut b = GraphBuilder::new(n);
    for i in 0..n {
        for j in (i + 1)..n {
            b.couple(i, j, gauss(&mut rng) * scale);
        }
    }
    b.build()
}

/// The 3D Edwards-Anderson model: an `l x l x l` cubic lattice with periodic boundaries.
///
/// Site `(x, y, z)` is index `(z * l + y) * l + x`, coupled to its six axis neighbours modulo `l`,
/// so there are `l^3` spins and exactly `3 l^3` bonds, each of degree six. The short-range spin
/// glass in the dimension where a finite-temperature transition exists, and the family the large
/// annealing comparisons are run on.
///
/// The optimum is NOT known, and unlike the two-dimensional case it is hard in the complexity
/// sense rather than merely inconvenient.
///
/// # Panics
///
/// If `l` is below 3. At `l = 2` the periodic wrap maps both `x+1` and `x-1` onto the same site,
/// so the two bonds would silently sum into one of double weight.
#[must_use]
pub fn edwards_anderson_3d(l: usize, bonds: Bonds, seed: u64) -> Graph {
    assert!(l >= 3, "a periodic cubic lattice below 3x3x3 doubles its own bonds");
    let n = l * l * l;
    let mut rng = Pcg::new(seed, 0x3D_0000);
    let at = |x: usize, y: usize, z: usize| ((z % l) * l + (y % l)) * l + (x % l);
    let mut b = GraphBuilder::new(n);
    let draw = |rng: &mut Pcg| match bonds {
        Bonds::PlusMinusJ => {
            if rng.f64() < 0.5 {
                1.0
            } else {
                -1.0
            }
        }
        Bonds::Gaussian => gauss(rng),
    };
    for z in 0..l {
        for y in 0..l {
            for x in 0..l {
                let i = at(x, y, z);
                b.couple(i, at(x + 1, y, z), draw(&mut rng));
                b.couple(i, at(x, y + 1, z), draw(&mut rng));
                b.couple(i, at(x, y, z + 1), draw(&mut rng));
            }
        }
    }
    b.build()
}

// ---- tile planting ------------------------------------------------------------------------------

/// A four-spin plaquette class, named by its ground-state degeneracy in isolation.
///
/// The couplings are given in the gauge where the all-up state is a ground state; planting gauges
/// them by the chosen state. Degeneracies are verified by enumerating each tile in the tests.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tile {
    /// Unfrustrated, all bonds satisfied, ground energy `-4`, two ground states.
    C1,
    /// Frustrated with unequal bonds, ground energy `-4`, four ground states.
    C2,
    /// Frustrated with equal bonds, ground energy `-2`, eight ground states.
    C3,
}

impl Tile {
    /// Energy this tile contributes at the planted state, which is its minimum in isolation.
    #[must_use]
    pub fn ground_energy(self) -> f64 {
        match self {
            Tile::C1 | Tile::C2 => -4.0,
            Tile::C3 => -2.0,
        }
    }

    /// How many states of the isolated four-cycle attain that minimum.
    #[must_use]
    pub fn degeneracy(self) -> usize {
        match self {
            Tile::C1 => 2,
            Tile::C2 => 4,
            Tile::C3 => 8,
        }
    }

    /// The four bonds around the cycle, in the gauge where all-up is a ground state.
    fn bonds(self, rng: &mut Pcg) -> [f64; 4] {
        let base: [f64; 4] = match self {
            Tile::C1 => [1.0, 1.0, 1.0, 1.0],
            // Two heavy bonds and two light ones, one of the light ones reversed. The minimum
            // breaks a light bond, and there are two of those.
            Tile::C2 => {
                if rng.f64() < 0.5 {
                    [2.0, 2.0, 1.0, -1.0]
                } else {
                    [2.0, 1.0, 2.0, -1.0]
                }
            }
            // One reversed bond among equals: any of the four may be the broken one.
            Tile::C3 => [1.0, 1.0, 1.0, -1.0],
        };
        let r = (rng.f64() * 4.0) as usize % 4;
        let mut out = [0.0; 4];
        for k in 0..4 {
            out[k] = base[(k + r) % 4];
        }
        out
    }
}

/// Tile planting on a periodic `l x l` square lattice, `l` even.
///
/// The lattice's `2 l^2` bonds partition **exactly** into the `l^2 / 2` plaquettes whose lower-left
/// corner has even `x + y`: each bond lies in one of them and no other. Every plaquette is given a
/// [`Tile`], gauged so the planted state minimises it, and because the plaquettes are bond-disjoint
/// the planted state minimises the sum. Neighbouring tiles still share spins, so the instance is
/// not a product of independent problems.
///
/// `mix` weights the three classes, in order `[C1, C2, C3]`; it is normalised, so `[1.0, 0.0, 0.0]`
/// and `[3.0, 0.0, 0.0]` are the same all-`C1` instance. `C1` alone is a gauged ferromagnet, and
/// difficulty comes from the degenerate classes.
///
/// # Panics
///
/// If `l` is odd or below 4 — the bond-disjoint tiling needs an even side, and at `l = 2` the
/// periodic wrap doubles the lattice's own bonds. Or if `mix` has a negative entry or sums to
/// zero.
#[must_use]
pub fn tile_planted_2d(l: usize, mix: [f64; 3], seed: u64) -> Planted {
    assert!(l >= 4 && l.is_multiple_of(2), "tile planting needs an even side of at least 4, got {l}");
    let total: f64 = mix.iter().sum();
    assert!(mix.iter().all(|&p| p >= 0.0) && total > 0.0, "mix must be non-negative and non-zero");

    let n = l * l;
    let mut rng = Pcg::new(seed, 0x71_1E00);
    let sigma: Vec<i8> = (0..n).map(|_| if rng.f64() < 0.5 { 1 } else { -1 }).collect();
    let at = |x: usize, y: usize| (y % l) * l + (x % l);

    let mut b = GraphBuilder::new(n);
    let mut energy = 0.0;
    let mut frustrated = 0;
    for y in 0..l {
        for x in 0..l {
            if (x + y) % 2 != 0 {
                continue; // the other half of the checkerboard; its bonds belong to these tiles
            }
            let u = rng.f64() * total;
            let tile = if u < mix[0] {
                Tile::C1
            } else if u < mix[0] + mix[1] {
                Tile::C2
            } else {
                Tile::C3
            };
            let c = [at(x, y), at(x + 1, y), at(x + 1, y + 1), at(x, y + 1)];
            let bonds = tile.bonds(&mut rng);
            for k in 0..4 {
                let (i, j) = (c[k], c[(k + 1) % 4]);
                b.couple(i, j, bonds[k] * f64::from(sigma[i]) * f64::from(sigma[j]));
            }
            energy += tile.ground_energy();
            if tile != Tile::C1 {
                frustrated += 1;
            }
        }
    }

    Planted { graph: b.build(), ground_state: sigma, ground_energy: energy, loops: frustrated }
}

// ---- deceptive cluster loops --------------------------------------------------------------------

/// Deceptive cluster loops on a Chimera graph `C_{m,n,t}`.
///
/// Frustrated loops planted between *cells* rather than between spins. Each cell's `K_{t,t}` bonds
/// are ferromagnetic in the planted gauge with strength `lambda`, so a cell behaves as one rigid
/// logical spin; every planted loop then runs around four neighbouring cells, entering and leaving
/// each through a single intra-cell bond, with one of its four inter-cell bonds reversed.
///
/// That eight-bond cycle is frustrated and its lightest bonds are the inter-cell ones, so its
/// minimum breaks exactly one of them — which is what the planted state does. The deception is that
/// every bond a single-spin move can see is satisfied and heavy, while the frustration lives at the
/// cluster level, where escaping costs a whole cell of `2t` spins flipped together.
///
/// Vertices are labelled as [`crate::ising::chimera`], and the bonds are a subset of that graph's:
/// all `m n t^2` intra-cell bonds, plus the inter-cell bonds the loops actually route through.
///
/// The ground energy is `-lambda * m n t^2 - loops * (4 lambda + 2)`.
///
/// # Panics
///
/// If the cell grid is smaller than `2 x 2` (no four-cell loop exists), if `t` or `loops` is zero,
/// or if `lambda` is below 1 — a cluster bond lighter than an inter-cell bond would be the one the
/// loop's minimum breaks, and the planted state would no longer attain it.
#[must_use]
pub fn deceptive_cluster_loops(
    m: usize,
    n: usize,
    t: usize,
    loops: usize,
    lambda: f64,
    seed: u64,
) -> Planted {
    assert!(m >= 2 && n >= 2, "a cluster loop needs a 2x2 block of cells, got {m}x{n}");
    assert!(t >= 1 && loops >= 1, "need at least one qubit per shore and one loop");
    assert!(lambda >= 1.0, "cluster bonds must be at least as heavy as inter-cell bonds");

    let nodes = 2 * t * m * n;
    let idx = |i: usize, j: usize, u: usize, k: usize| ((i * n) + j) * 2 * t + u * t + k;
    let mut rng = Pcg::new(seed, 0xDC_1000);
    let sigma: Vec<i8> = (0..nodes).map(|_| if rng.f64() < 0.5 { 1 } else { -1 }).collect();
    let mut b = GraphBuilder::new(nodes);
    let gauge = |i: usize, j: usize| f64::from(sigma[i]) * f64::from(sigma[j]);

    // Rigid clusters: every intra-cell bond, satisfied by the planted state.
    for i in 0..m {
        for j in 0..n {
            for a in 0..t {
                for c in 0..t {
                    let (p, q) = (idx(i, j, 0, a), idx(i, j, 1, c));
                    b.couple(p, q, lambda * gauge(p, q));
                }
            }
        }
    }

    for _ in 0..loops {
        let i = (rng.f64() * (m - 1) as f64) as usize % (m - 1);
        let j = (rng.f64() * (n - 1) as f64) as usize % (n - 1);
        let k: Vec<usize> = (0..4).map(|_| (rng.f64() * t as f64) as usize % t).collect();
        // Cells A=(i,j) B=(i+1,j) C=(i+1,j+1) D=(i,j+1). Vertical hops use shore 0, horizontal
        // hops shore 1, so the two ends inside a cell always sit on opposite shores and the
        // intra-cell step is a single K_{t,t} bond.
        let cycle = [
            (idx(i, j, 0, k[0]), idx(i + 1, j, 0, k[0])),         // A -> B, inter
            (idx(i + 1, j, 0, k[0]), idx(i + 1, j, 1, k[1])),     // inside B, intra
            (idx(i + 1, j, 1, k[1]), idx(i + 1, j + 1, 1, k[1])), // B -> C, inter
            (idx(i + 1, j + 1, 1, k[1]), idx(i + 1, j + 1, 0, k[2])), // inside C, intra
            (idx(i + 1, j + 1, 0, k[2]), idx(i, j + 1, 0, k[2])), // C -> D, inter
            (idx(i, j + 1, 0, k[2]), idx(i, j + 1, 1, k[3])),     // inside D, intra
            (idx(i, j + 1, 1, k[3]), idx(i, j, 1, k[3])),         // D -> A, inter
            (idx(i, j, 1, k[3]), idx(i, j, 0, k[0])),             // inside A, intra
        ];
        let inter = [0usize, 2, 4, 6];
        let broken = inter[(rng.f64() * 4.0) as usize % 4];
        for (step, &(p, q)) in cycle.iter().enumerate() {
            let w = if step % 2 == 0 { 1.0 } else { lambda };
            b.couple(p, q, if step == broken { -w } else { w } * gauge(p, q));
        }
    }

    let intra = (m * n * t * t) as f64;
    let energy = -lambda * intra - loops as f64 * (4.0 * lambda + 2.0);
    Planted { graph: b.build(), ground_state: sigma, ground_energy: energy, loops }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exact::Elimination;
    use crate::oracle::{Exhaustive, RandomGuess, Solver, SteepestDescent};

    // What these tests check against, and what they refuse to.
    //
    // Every planted claim is checked against an EXACT optimum -- exhaustive enumeration below 24
    // spins, variable elimination above it -- and against a closed form computed before the graph
    // exists. The unplanted families have no optimum to check, so they are checked against exact
    // identities instead: the edge set, and the first two moments of the whole energy spectrum,
    // both of which are computable in closed form and neither of which any sampler is involved in.
    //
    // There is deliberately no test comparing one solver against another as evidence of
    // correctness. Two heuristics agreeing on a state neither can prove optimal is not evidence,
    // and on these families it is the specific mistake the planted construction exists to avoid.

    /// Weight of edge `(i, j)`, or `None` if there is none.
    fn bond(g: &Graph, i: usize, j: usize) -> Option<f64> {
        (g.offset[i]..g.offset[i + 1]).find(|&k| g.nbr[k] as usize == j).map(|k| g.w[k])
    }

    /// Mean and mean-square of the energy over ALL `2^n` states, by enumeration.
    fn spectrum_moments(g: &Graph) -> (f64, f64) {
        assert!(g.n <= 20);
        let (mut sum, mut sq) = (0.0, 0.0);
        let mut s = vec![-1i8; g.n];
        for mask in 0..(1u64 << g.n) {
            for i in 0..g.n {
                s[i] = if mask >> i & 1 == 1 { 1 } else { -1 };
            }
            let e = g.energy(&s);
            sum += e;
            sq += e * e;
        }
        let count = (1u64 << g.n) as f64;
        (sum / count, sq / count)
    }

    /// Sum of squared couplings, each edge once.
    fn sum_j_squared(g: &Graph) -> f64 {
        let mut t = 0.0;
        for i in 0..g.n {
            for k in g.offset[i]..g.offset[i + 1] {
                if g.nbr[k] as usize > i {
                    t += g.w[k] * g.w[k];
                }
            }
        }
        t
    }

    // ---- Sherrington-Kirkpatrick ---------------------------------------------------------------

    #[test]
    fn sk_is_the_complete_graph_and_reproduces_from_its_seed() {
        for n in [2, 5, 40] {
            let g = sherrington_kirkpatrick(n, 3);
            assert_eq!(g.n_edges, n * (n - 1) / 2, "n={n}");
            assert_eq!(g.max_degree(), n - 1, "n={n}");
            assert!(g.h.iter().all(|&h| h == 0.0), "the model has no field");
        }
        let a = sherrington_kirkpatrick(30, 8);
        let b = sherrington_kirkpatrick(30, 8);
        let c = sherrington_kirkpatrick(30, 9);
        assert_eq!(a.w, b.w, "same seed must reproduce");
        assert_ne!(a.w, c.w, "different seeds must differ");
    }

    #[test]
    fn sk_couplings_carry_the_variance_that_makes_the_energy_extensive() {
        // J ~ N(0, 1/n) is the scaling, so n * mean(J^2) -> 1. With 19,900 couplings the standard
        // error of that estimate is sqrt(2/19900) = 0.010, so 0.05 is five sigma: this fails on a
        // wrong scale factor, not on luck.
        let n = 200;
        let g = sherrington_kirkpatrick(n, 17);
        let m = g.n_edges as f64;
        let mean = sum_j_squared(&g) / m; // the second moment
        let first: f64 = {
            let mut t = 0.0;
            for i in 0..g.n {
                for k in g.offset[i]..g.offset[i + 1] {
                    if g.nbr[k] as usize > i {
                        t += g.w[k];
                    }
                }
            }
            t / m
        };
        assert!((mean * n as f64 - 1.0).abs() < 0.05, "n*var was {}", mean * n as f64);
        assert!(first.abs() < 0.003, "mean coupling {first} should be zero");
    }

    #[test]
    fn sk_spectrum_moments_match_their_closed_form() {
        // An exact identity, not a sample: averaged over all 2^n states, <E> = 0 and <E^2> is the
        // sum of squared couplings, because every cross term pairs distinct spins and averages
        // away. It pins the whole spectrum -- centred, and exactly as wide as the couplings make it
        // -- so a stray field, a self-coupling folded into the energy, or a sign convention that
        // did not survive the build would move it.
        //
        // What it does NOT catch is a doubled edge: both sides read the merged weight back out of
        // the same graph. The edge count in the structural test above is what covers that.
        for n in [8, 12, 16] {
            let g = sherrington_kirkpatrick(n, 5);
            let (mean, ms) = spectrum_moments(&g);
            assert!(mean.abs() < 1e-9, "n={n}: <E> was {mean}");
            let want = sum_j_squared(&g);
            assert!((ms - want).abs() / want < 1e-12, "n={n}: <E^2> {ms} vs sum J^2 {want}");
        }
    }

    #[test]
    fn sk_ground_energy_agrees_between_two_exact_methods() {
        // Enumeration and variable elimination are independent exact routes to the same number.
        // Neither is a sampler and neither can be tuned into agreement.
        for n in [10, 14] {
            for seed in 1..=3u64 {
                let g = sherrington_kirkpatrick(n, seed);
                let (_s, brute) = Exhaustive.solve(&g);
                let e = Elimination::default().ground_state(&g).expect("dense but small");
                let elim = e.ground_energy.expect("min-sum was run");
                assert!((brute - elim).abs() < 1e-9, "n={n} seed={seed}: {brute} vs {elim}");
            }
        }
    }


    #[test]
    fn the_ground_energy_density_approaches_the_published_limit() {
        // NOT an exact oracle, and labelled so: Parisi's -0.7632 is a thermodynamic limit and these
        // are sixteen-spin instances. It is here because it catches the one mistake this generator
        // can actually make. The N(0, 1/n) variance is what keeps the energy extensive; drop the
        // 1/n and the density grows like sqrt(n) instead of settling. Exact ground energies,
        // averaged over instances, must sit above the limit and deepen toward it with n.
        let density = |n: usize, instances: u64| {
            let t: f64 = (1..=instances)
                .map(|seed| Exhaustive.solve(&sherrington_kirkpatrick(n, seed)).1 / n as f64)
                .sum();
            t / instances as f64
        };
        const PARISI: f64 = -0.7632;
        let small = density(10, 24);
        let large = density(16, 24);
        assert!(small > PARISI && large > PARISI, "{small} and {large} must sit above {PARISI}");
        assert!(large < small - 0.01, "the density must deepen with n: {small} then {large}");
        assert!(large - PARISI < 0.2, "n=16 should be within 0.2 of the limit, was {large}");
    }

    // ---- 3D Edwards-Anderson --------------------------------------------------------------------

    #[test]
    fn the_cubic_lattice_has_exactly_its_axis_neighbours() {
        // The whole content of the generator is its edge set, so the edge set is checked against
        // the definition rather than against a count.
        for l in [3, 4, 5] {
            let g = edwards_anderson_3d(l, Bonds::PlusMinusJ, 1);
            assert_eq!(g.n, l * l * l);
            assert_eq!(g.n_edges, 3 * l * l * l, "l={l}");
            let at = |x: usize, y: usize, z: usize| ((z % l) * l + (y % l)) * l + (x % l);
            for z in 0..l {
                for y in 0..l {
                    for x in 0..l {
                        let i = at(x, y, z);
                        let mut want = vec![
                            at(x + 1, y, z),
                            at(x + l - 1, y, z),
                            at(x, y + 1, z),
                            at(x, y + l - 1, z),
                            at(x, y, z + 1),
                            at(x, y, z + l - 1),
                        ];
                        want.sort_unstable();
                        let mut got: Vec<usize> =
                            (g.offset[i]..g.offset[i + 1]).map(|k| g.nbr[k] as usize).collect();
                        got.sort_unstable();
                        assert_eq!(got, want, "site ({x},{y},{z}) at l={l}");
                    }
                }
            }
        }
    }

    #[test]
    fn plus_minus_bonds_are_unit_and_gaussian_bonds_are_not() {
        let pm = edwards_anderson_3d(6, Bonds::PlusMinusJ, 4);
        assert!(pm.w.iter().all(|&w| w == 1.0 || w == -1.0), "+/-J must be exactly +/-1");
        let neg = pm.w.iter().filter(|&&w| w < 0.0).count() / 2;
        let m = pm.n_edges;
        // Binomial(648, 1/2): four standard deviations is 2*sqrt(m).
        let dev = (neg as f64 - m as f64 / 2.0).abs();
        assert!(dev < 2.0 * (m as f64).sqrt(), "{neg} of {m} bonds negative");

        let gs = edwards_anderson_3d(6, Bonds::Gaussian, 4);
        assert!(gs.w.iter().any(|&w| (w.abs() - 1.0).abs() > 1e-6), "Gaussian bonds are continuous");
        let var = sum_j_squared(&gs) / gs.n_edges as f64;
        assert!((var - 1.0).abs() < 0.25, "unit variance, got {var}");
    }

    #[test]
    fn exactly_half_the_plaquettes_are_frustrated_in_expectation() {
        // The defining statistic of the +/-J model: a plaquette is frustrated iff the product of
        // its four bonds is negative, which for independent fair bonds has probability exactly 1/2.
        // A generator that correlated bonds -- reusing a draw, or seeding per site -- would move
        // this, and no degree or count check would notice.
        let l = 8;
        let g = edwards_anderson_3d(l, Bonds::PlusMinusJ, 21);
        let at = |x: usize, y: usize, z: usize| ((z % l) * l + (y % l)) * l + (x % l);
        let mut frustrated = 0;
        let mut total = 0;
        for z in 0..l {
            for y in 0..l {
                for x in 0..l {
                    // the three plaquettes with a corner here, one per plane
                    let faces = [
                        [at(x, y, z), at(x + 1, y, z), at(x + 1, y + 1, z), at(x, y + 1, z)],
                        [at(x, y, z), at(x + 1, y, z), at(x + 1, y, z + 1), at(x, y, z + 1)],
                        [at(x, y, z), at(x, y + 1, z), at(x, y + 1, z + 1), at(x, y, z + 1)],
                    ];
                    for f in faces {
                        let mut prod = 1.0;
                        for k in 0..4 {
                            prod *= bond(&g, f[k], f[(k + 1) % 4]).expect("lattice face");
                        }
                        total += 1;
                        if prod < 0.0 {
                            frustrated += 1;
                        }
                    }
                }
            }
        }
        assert_eq!(total, 3 * l * l * l);
        let dev = (frustrated as f64 - total as f64 / 2.0).abs();
        assert!(dev < 2.0 * (total as f64).sqrt(), "{frustrated} of {total} faces frustrated");
    }

    #[test]
    fn the_lattice_ground_energy_is_gauge_invariant() {
        // No optimum is known for this family, so the check is an identity the optimum must obey:
        // J_ij -> g_i g_j J_ij is a relabelling of states, so the ground energy cannot move. Run
        // through exact elimination, at the one size where a 3D torus still fits under the width
        // cap.
        let l = 3;
        for bonds in [Bonds::PlusMinusJ, Bonds::Gaussian] {
            let g = edwards_anderson_3d(l, bonds, 6);
            let mut rng = Pcg::new(99, 1);
            let gauge: Vec<i8> = (0..g.n).map(|_| if rng.f64() < 0.5 { 1 } else { -1 }).collect();
            let mut b = GraphBuilder::new(g.n);
            for i in 0..g.n {
                for k in g.offset[i]..g.offset[i + 1] {
                    let j = g.nbr[k] as usize;
                    if j > i {
                        b.couple(i, j, g.w[k] * f64::from(gauge[i]) * f64::from(gauge[j]));
                    }
                }
            }
            let gauged = b.build();
            let el = Elimination::default();
            let a = el.ground_state(&g).expect("3x3x3 fits").ground_energy.expect("min-sum");
            let c = el.ground_state(&gauged).expect("same shape").ground_energy.expect("min-sum");
            assert!((a - c).abs() < 1e-9, "{bonds:?}: {a} vs gauged {c}");

            // and the state elimination returns really is optimal there: no single flip improves.
            let s = el.ground_state(&g).expect("3x3x3 fits").ground_state.expect("min-sum");
            for i in 0..g.n {
                let delta = 2.0 * f64::from(s[i]) * g.field(i, &s);
                assert!(delta > -1e-9, "flipping {i} lowers the reported ground energy by {delta}");
            }
        }
    }

    // ---- tile planting --------------------------------------------------------------------------

    /// Every checkerboard plaquette of a periodic `l x l` lattice, in cycle order.
    fn tiles_of(l: usize) -> Vec<[usize; 4]> {
        let at = |x: usize, y: usize| (y % l) * l + (x % l);
        let mut out = Vec::new();
        for y in 0..l {
            for x in 0..l {
                if (x + y) % 2 == 0 {
                    out.push([at(x, y), at(x + 1, y), at(x + 1, y + 1), at(x, y + 1)]);
                }
            }
        }
        out
    }

    #[test]
    fn the_tiling_covers_every_bond_exactly_once() {
        // The premise of the construction. If a bond were in two tiles their couplings would sum
        // and neither tile's minimum would survive; if it were in none the lattice would be
        // incomplete.
        for l in [4, 6, 8] {
            let p = tile_planted_2d(l, [1.0, 1.0, 1.0], 2);
            assert_eq!(p.graph.n_edges, 2 * l * l, "l={l}");
            let mut seen = std::collections::BTreeSet::new();
            for t in tiles_of(l) {
                for k in 0..4 {
                    let (a, b) = (t[k], t[(k + 1) % 4]);
                    let key = if a < b { (a, b) } else { (b, a) };
                    assert!(seen.insert(key), "bond {key:?} is in two tiles at l={l}");
                }
            }
            assert_eq!(seen.len(), 2 * l * l, "l={l}");
            assert_eq!(tiles_of(l).len(), l * l / 2);
        }
    }

    #[test]
    fn every_tile_is_minimised_by_the_planted_state_with_its_class_degeneracy() {
        // Each plaquette read back out of the built graph and enumerated over its sixteen states.
        // This is the per-term claim the global argument rests on, and it also pins the class
        // degeneracies 2/4/8 that name the classes.
        for mix in [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]] {
            let p = tile_planted_2d(6, mix, 12);
            for t in tiles_of(6) {
                let w: Vec<f64> =
                    (0..4).map(|k| bond(&p.graph, t[k], t[(k + 1) % 4]).expect("tile bond")).collect();
                let e = |s: [i8; 4]| -> f64 {
                    -(0..4).map(|k| w[k] * f64::from(s[k]) * f64::from(s[(k + 1) % 4])).sum::<f64>()
                };
                let mut best = f64::INFINITY;
                let mut hits = 0;
                for mask in 0..16u32 {
                    let s: [i8; 4] =
                        std::array::from_fn(|k| if mask >> k & 1 == 1 { 1 } else { -1 });
                    let v = e(s);
                    if v < best - 1e-9 {
                        best = v;
                        hits = 1;
                    } else if (v - best).abs() < 1e-9 {
                        hits += 1;
                    }
                }
                let planted: [i8; 4] = std::array::from_fn(|k| p.ground_state[t[k]]);
                assert!((e(planted) - best).abs() < 1e-9, "tile {t:?} planted {} vs {best}", e(planted));
                let class = match (best, hits) {
                    (b, 2) if (b + 4.0).abs() < 1e-9 => Tile::C1,
                    (b, 4) if (b + 4.0).abs() < 1e-9 => Tile::C2,
                    (b, 8) if (b + 2.0).abs() < 1e-9 => Tile::C3,
                    _ => panic!("tile {t:?} has minimum {best} with {hits} minimisers"),
                };
                assert_eq!(class.ground_energy(), best);
                assert_eq!(class.degeneracy(), hits);
            }
        }
    }

    #[test]
    fn the_planted_state_really_is_a_ground_state_of_the_tiled_lattice() {
        // Against enumeration where it runs, and against exact elimination past it. The closed-form
        // energy is predicted before the graph is built, so this checks the arithmetic too.
        for mix in [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [2.0, 1.0, 1.0], [0.0, 1.0, 3.0]] {
            for seed in 1..=3u64 {
                let p = tile_planted_2d(4, mix, seed);
                let (_s, brute) = Exhaustive.solve(&p.graph);
                let planted = p.graph.energy(&p.ground_state);
                assert!((planted - brute).abs() < 1e-9, "{mix:?} seed={seed}: {planted} vs {brute}");
                assert!((planted - p.ground_energy).abs() < 1e-9, "closed form {}", p.ground_energy);
            }
        }
        for l in [6, 8] {
            let p = tile_planted_2d(l, [1.0, 1.0, 2.0], 4);
            let e = Elimination::default()
                .ground_state(&p.graph)
                .expect("a torus of this side fits the width cap")
                .ground_energy
                .expect("min-sum was run");
            assert!((p.ground_energy - e).abs() < 1e-9, "l={l}: planted {} vs exact {e}", p.ground_energy);
        }
    }

    #[test]
    fn tile_instances_are_reproducible_and_noise_never_solves_one() {
        let a = tile_planted_2d(8, [1.0, 1.0, 1.0], 42);
        let b = tile_planted_2d(8, [1.0, 1.0, 1.0], 42);
        let c = tile_planted_2d(8, [1.0, 1.0, 1.0], 43);
        assert_eq!(a.ground_state, b.ground_state);
        assert_eq!(a.ground_energy, b.ground_energy);
        assert_ne!(a.ground_state, c.ground_state);

        let (s, _) = RandomGuess { tries: 20_000, seed: 2 }.solve(&a.graph);
        assert!(!a.solved(&s), "random guessing solved a planted instance");
    }


    #[test]
    fn the_mixture_is_what_makes_a_tiled_lattice_hard() {
        // Measured, and it corrected two guesses. All-C3 is the MOST degenerate class and the
        // EASIEST instance: eight ground states per tile means exponentially many global ones, and
        // any of them counts as solved. All-C1 is a gauged ferromagnet, which greedy also misses
        // often -- but for the classic domain-wall reason, not because the instance is hard. It is
        // the MIXTURE that makes the optimum scarce: at l = 8, 16/16 solved for all-C3 against
        // 0/16 for an even mix.
        let rate = |mix: [f64; 3]| {
            let (mut solved, mut total) = (0, 0);
            for iseed in 1..=4u64 {
                let p = tile_planted_2d(8, mix, iseed);
                for sseed in 1..=4u64 {
                    let (s, _) = SteepestDescent { restarts: 50, seed: sseed }.solve(&p.graph);
                    total += 1;
                    if p.solved(&s) {
                        solved += 1;
                    }
                }
            }
            solved as f64 / total as f64
        };
        let degenerate = rate([0.0, 0.0, 1.0]);
        let mixed = rate([1.0, 1.0, 1.0]);
        assert!(degenerate > 0.9, "all-C3 is the easy end of the family, got {degenerate}");
        assert!(mixed < 0.25, "an even mix should defeat greedy, got {mixed}");
    }

    // ---- deceptive cluster loops ------------------------------------------------------------------

    #[test]
    fn the_planted_state_really_is_a_ground_state_of_the_cluster_instance() {
        for lambda in [1.0, 2.0, 5.0] {
            for loops in [1, 3, 9] {
                for seed in 1..=2u64 {
                    let p = deceptive_cluster_loops(2, 2, 2, loops, lambda, seed);
                    let (_s, brute) = Exhaustive.solve(&p.graph);
                    let planted = p.graph.energy(&p.ground_state);
                    assert!(
                        (planted - brute).abs() < 1e-9,
                        "lambda={lambda} loops={loops} seed={seed}: planted {planted} vs {brute}"
                    );
                    assert!(
                        (planted - p.ground_energy).abs() < 1e-9,
                        "closed form {} vs measured {planted}",
                        p.ground_energy
                    );
                }
            }
        }
    }

    #[test]
    fn the_cluster_instance_is_a_subgraph_of_chimera_with_rigid_cells() {
        let (m, n, t, lambda) = (3, 3, 4, 3.0);
        let p = deceptive_cluster_loops(m, n, t, 40, lambda, 7);
        let c = crate::ising::chimera(m, n, t, 1.0);
        assert_eq!(p.graph.n, c.n);
        let idx = |i: usize, j: usize, u: usize, k: usize| ((i * n) + j) * 2 * t + u * t + k;

        for i in 0..p.graph.n {
            for k in p.graph.offset[i]..p.graph.offset[i + 1] {
                let j = p.graph.nbr[k] as usize;
                assert!(bond(&c, i, j).is_some(), "edge ({i},{j}) is not a Chimera edge");
            }
        }
        // every intra-cell bond present, at least lambda, and satisfied by the planted state
        for i in 0..m {
            for j in 0..n {
                for a in 0..t {
                    for b in 0..t {
                        let (u, v) = (idx(i, j, 0, a), idx(i, j, 1, b));
                        let w = bond(&p.graph, u, v).expect("intra-cell bond");
                        assert!(w.abs() >= lambda - 1e-9, "cluster bond {w} below lambda");
                        let sat = w * f64::from(p.ground_state[u]) * f64::from(p.ground_state[v]);
                        assert!(sat > 0.0, "cluster bond ({u},{v}) is broken at the planted state");
                    }
                }
            }
        }
    }

    #[test]
    fn cluster_instances_are_exactly_solved_past_the_size_enumeration_reaches() {
        // Chimera is narrow -- a 3x3 block of size-4 cells is 72 spins at induced width 8 -- so
        // elimination checks the planted claim two magnitudes past where enumeration stops.
        for &(m, n, t) in &[(3usize, 3usize, 4usize), (4, 4, 4)] {
            for lambda in [1.0, 3.0] {
                let p = deceptive_cluster_loops(m, n, t, 20, lambda, 5);
                let e = Elimination::default()
                    .ground_state(&p.graph)
                    .expect("Chimera is narrow")
                    .ground_energy
                    .expect("min-sum was run");
                assert!(
                    (p.ground_energy - e).abs() < 1e-9,
                    "{m}x{n}x{t} lambda={lambda}: planted {} vs exact {e}",
                    p.ground_energy
                );
            }
        }
    }

    #[test]
    fn heavy_clusters_defeat_single_spin_descent_where_light_ones_do_not() {
        // The deception, measured rather than asserted. Same cells, same loop count, same planted
        // optimum: the only change is how heavy the bonds are that a single-spin move must climb.
        // Measured as a SOLVE RATE against the known optimum over a seed grid -- a mean excess
        // would hide it, because a solver that leaves one cluster flipped is close in energy and
        // still wrong. At 3x3 cells of size 4 with 20 loops: 12/16 at lambda 1, 0/16 at lambda 8.
        let rate = |lambda: f64| {
            let (mut solved, mut total) = (0, 0);
            for iseed in 1..=4u64 {
                let p = deceptive_cluster_loops(3, 3, 4, 24, lambda, iseed);
                for sseed in 1..=4u64 {
                    let (s, _) = SteepestDescent { restarts: 40, seed: sseed }.solve(&p.graph);
                    total += 1;
                    if p.solved(&s) {
                        solved += 1;
                    }
                }
            }
            solved as f64 / total as f64
        };
        let light = rate(1.0);
        let heavy = rate(8.0);
        assert!(light > 0.5, "light clusters should mostly be solved, got {light}");
        assert!(heavy < 0.1, "heavy clusters should defeat single-spin descent, got {heavy}");
    }

    #[test]
    fn descent_stops_with_every_cluster_bond_satisfied_and_the_answer_still_wrong() {
        // This is what "deceptive" means, stated as a property of the returned state rather than
        // as a comparison. Greedy ends having satisfied EVERY bond inside every cell -- everything
        // a single-spin move can see is already paid for -- and is still above the optimum, because
        // the frustration it has not resolved lives between cells and costs a whole cell to move.
        let (m, n, t, lambda) = (3usize, 3usize, 4usize, 8.0);
        let idx = |i: usize, j: usize, u: usize, k: usize| ((i * n) + j) * 2 * t + u * t + k;
        for iseed in 1..=4u64 {
            let p = deceptive_cluster_loops(m, n, t, 24, lambda, iseed);
            let (s, _) = SteepestDescent { restarts: 40, seed: 3 }.solve(&p.graph);
            assert!(!p.solved(&s), "seed {iseed} was solved; the test size no longer deceives");
            for i in 0..m {
                for j in 0..n {
                    for a in 0..t {
                        for b in 0..t {
                            let (u, v) = (idx(i, j, 0, a), idx(i, j, 1, b));
                            let w = bond(&p.graph, u, v).expect("intra-cell bond");
                            let sat = w * f64::from(s[u]) * f64::from(s[v]);
                            assert!(sat > 0.0, "descent left cluster bond ({u},{v}) broken");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn the_difficulty_is_the_move_scale_and_nothing_else() {
        // The other half of the same claim: take the state single-spin descent gave up on and keep
        // descending, changing only the MOVE -- whole cells at a time instead of single spins.
        // Same instance, same greedy rule, same planted optimum as judge. 4/16 becomes 13/16.
        //
        // This is a comparison of two solvers, which proves nothing on its own; what makes it
        // evidence is that both are scored against a ground state known by construction rather than
        // against each other.
        let (m, n, t) = (4usize, 4usize, 4usize);
        let idx = |i: usize, j: usize, u: usize, k: usize| ((i * n) + j) * 2 * t + u * t + k;
        let (mut single, mut cluster, mut total) = (0, 0, 0);
        for iseed in 1..=4u64 {
            let p = deceptive_cluster_loops(m, n, t, 8, 4.0, iseed);
            for sseed in 1..=4u64 {
                let (mut s, _) = SteepestDescent { restarts: 40, seed: sseed }.solve(&p.graph);
                total += 1;
                if p.solved(&s) {
                    single += 1;
                }
                loop {
                    let mut moved = false;
                    for i in 0..m {
                        for j in 0..n {
                            let cell: Vec<usize> = (0..2)
                                .flat_map(|u| (0..t).map(move |k| (u, k)))
                                .map(|(u, k)| idx(i, j, u, k))
                                .collect();
                            let before = p.graph.energy(&s);
                            for &q in &cell {
                                s[q] = -s[q];
                            }
                            if p.graph.energy(&s) < before - 1e-9 {
                                moved = true;
                            } else {
                                for &q in &cell {
                                    s[q] = -s[q];
                                }
                            }
                        }
                    }
                    if !moved {
                        break;
                    }
                }
                if p.solved(&s) {
                    cluster += 1;
                }
            }
        }
        assert!(single * 2 < total, "single-spin descent should mostly fail, got {single}/{total}");
        assert!(
            cluster * 2 > total && cluster > single,
            "cluster moves should mostly succeed, got {cluster}/{total} against {single}/{total}"
        );
    }

    // ---- the guards that keep a planted optimum true --------------------------------------------

    #[test]
    #[should_panic(expected = "cluster bonds must be at least as heavy")]
    fn a_cluster_lighter_than_its_loops_is_refused() {
        // Below lambda = 1 the loop's minimum breaks an INTRA-cell bond instead of the inter-cell
        // one, the planted state stops attaining it, and the instance would ship with an advertised
        // optimum that is not one. This assertion is the only thing standing between the two.
        let _ = deceptive_cluster_loops(2, 2, 2, 1, 0.5, 1);
    }

    #[test]
    #[should_panic(expected = "even side")]
    fn an_odd_side_is_refused_for_tile_planting() {
        // On an odd side the checkerboard does not close around the periodic wrap: some bonds land
        // in two tiles, where the couplings would sum, and some in none.
        let _ = tile_planted_2d(5, [1.0, 1.0, 1.0], 1);
    }

    #[test]
    #[should_panic(expected = "doubles its own bonds")]
    fn a_two_site_cube_is_refused() {
        let _ = edwards_anderson_3d(2, Bonds::PlusMinusJ, 1);
    }
}
