//! Simulated quantum annealing: path-integral Monte Carlo on the transverse-field Ising model.
//!
//! The sampler [`OpenJij`] ships beside its classical one, and the reason a stack that has
//! simulated annealing and not this one is incomplete: they explore differently. Classical
//! annealing crosses a barrier by going **over** it, at a cost set by the barrier's height. This
//! goes **through**, at a cost set by the barrier's width — so a landscape of tall thin walls,
//! which is what a frustrated glass has, is the case where the two methods genuinely differ.
//!
//! # What is actually simulated
//!
//! Not a quantum computer. The Suzuki–Trotter decomposition turns a `d`-dimensional quantum Ising
//! model in a transverse field into a `d+1`-dimensional **classical** one: `M` copies of the spins,
//! the Trotter slices, each carrying the original couplings at strength `J/M`, and every site
//! coupled to itself in the neighbouring slices at
//!
//! ```text
//!   J⊥ = −(1 / 2β) · ln tanh(βΓ / M)     equivalently   βJ⊥ = ½ ln coth(βΓ / M)
//! ```
//!
//! which is large when the transverse field `Γ` is small — slices locked together, the classical
//! limit — and small when `Γ` is large, letting the slices disagree and the system explore several
//! classical states at once. Annealing `Γ` downwards is what closes them back into one answer.
//!
//! Everything here is a Metropolis update on that classical `d+1`-dimensional system. The word
//! "quantum" describes what is being modelled, not what is being run, and the distinction matters
//! because the result is a classical state and carries no quantum claim.
//!
//! # Two things this does not hide
//!
//! **`Γ` is never annealed to exactly zero.** `tanh(0) = 0` makes `J⊥` infinite, and an infinite
//! coupling is not a strong classical limit but a division. [`Params::gamma_min`] stops short, and
//! the value is a parameter rather than a constant because how short is a modelling choice.
//!
//! **One Trotter slice is not the classical limit.** With `M = 1` the two Trotter neighbours of a
//! site are the site itself, so the coupling term becomes a constant that suppresses every flip
//! equally. That is not classical annealing; it is a frozen system. `M = 1` therefore drops the
//! term entirely, which *is* classical annealing, and is the honest control to compare against.
//!
//! ```
//! use ferrotherm::{sqa, ising::lattice2d};
//!
//! let g = lattice2d(6, 1.0);
//! let out = sqa::run(&g, &sqa::Params::default(), 5);
//! assert!((out.energy - g.energy(&out.state)).abs() < 1e-9);
//! assert!(out.energy <= -2.0 * 36.0 + 1e-9, "a ferromagnet has an all-aligned ground state");
//! ```
//!
//! [`OpenJij`]: https://www.openjij.org/

use crate::graph::Graph;
use crate::ledger::Ledger;
use crate::rng::Pcg;

/// How the anneal is run.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Params {
    /// Trotter slices `M`. **One means classical**: see the module note.
    pub trotter: usize,
    /// Inverse temperature, held fixed while the transverse field anneals. This is the standard
    /// discrete-time formulation — the temperature is a simulation parameter, not the schedule.
    pub beta: f64,
    /// Transverse field at the start. Large enough that the slices are effectively free.
    pub gamma_max: f64,
    /// Transverse field at the end. **Not zero**, which would make `J⊥` infinite.
    pub gamma_min: f64,
    /// Anneal steps. Each step lowers `Γ` geometrically and runs `sweeps_per_step` sweeps.
    pub steps: usize,
    /// Sweeps of the whole `M × n` system per step.
    pub sweeps_per_step: usize,
}

impl Default for Params {
    fn default() -> Self {
        Params {
            trotter: 4,
            beta: 10.0,
            gamma_max: 3.0,
            gamma_min: 0.05,
            steps: 200,
            sweeps_per_step: 1,
        }
    }
}

/// What the anneal found.
#[derive(Clone, Debug)]
pub struct Outcome {
    /// The best **classical** state seen, over every slice and every step.
    pub state: Vec<i8>,
    /// Its energy, recomputed from the state by the graph.
    pub energy: f64,
    /// Single-spin proposals made, over the whole `M × n` system.
    pub proposals: u64,
    /// Proposals accepted. A rate near zero means the field annealed faster than the system could
    /// follow, and a rate near one means it never got cold enough to decide anything.
    pub accepted: u64,
    /// The largest `J⊥` reached, at `gamma_min`. Reported because it is the quantity that goes to
    /// infinity if `gamma_min` is set to zero, and a reader should be able to see how close it got.
    pub max_j_perp: f64,
}

/// Run simulated quantum annealing.
pub fn run(g: &Graph, p: &Params, seed: u64) -> Outcome {
    run_metered(g, p, seed, None)
}

/// As [`run`], charging every proposal to a [`Ledger`].
pub fn run_metered(g: &Graph, p: &Params, seed: u64, mut ledger: Option<&mut Ledger>) -> Outcome {
    let n = g.n;
    let m = p.trotter.max(1);
    if n == 0 {
        return Outcome { state: Vec::new(), energy: 0.0, proposals: 0, accepted: 0, max_j_perp: 0.0 };
    }
    let mut rng = Pcg::new(seed, 0x0005_9A11);
    // `s[k * n + i]` is site `i` in Trotter slice `k`.
    let mut s: Vec<i8> = (0..m * n).map(|_| rng.spin(0.5)).collect();

    let mut best: Vec<i8> = s[..n].to_vec();
    let mut best_e = g.energy(&best);
    let (mut proposals, mut accepted) = (0u64, 0u64);
    let mut max_j_perp = 0.0f64;

    let steps = p.steps.max(1);
    let (gmax, gmin) = (p.gamma_max.max(1e-9), p.gamma_min.max(1e-9));
    for step in 0..steps {
        // Geometric in Γ, which is the schedule the transverse-field literature uses: the
        // interesting dynamics are at small Γ and a linear ramp spends its budget at large Γ.
        let f = if steps == 1 { 1.0 } else { step as f64 / (steps - 1) as f64 };
        let gamma = gmax * (gmin / gmax).powf(f);
        // J⊥ = −(1 / 2β) ln tanh(βΓ / M). Only meaningful for M > 1: see the module note.
        let j_perp = if m == 1 {
            0.0
        } else {
            let x = (p.beta * gamma / m as f64).tanh().max(1e-300);
            // NO FACTOR OF M. See the module note: with one, this whole module sampled a different
            // model. The Trotter coupling that belongs beside an intra-slice term scaled by 1/M and
            // a Boltzmann factor at full beta is J_perp = (1/2beta) ln coth(beta*Gamma/M), so that
            // beta*J_perp = (1/2) ln coth(beta*Gamma/M) -- the dimensionless number Suzuki-Trotter
            // actually fixes.
            -(1.0 / (2.0 * p.beta)) * x.ln()
        };
        max_j_perp = max_j_perp.max(j_perp);

        for _ in 0..p.sweeps_per_step.max(1) {
            for k in 0..m {
                for i in 0..n {
                    let cur = s[k * n + i];
                    // Intra-slice field, at strength 1/M: this slice's own copy of the problem.
                    let mut field = g.h[i];
                    for e in g.offset[i]..g.offset[i + 1] {
                        field += g.w[e] * s[k * n + g.nbr[e] as usize] as f64;
                    }
                    let mut d = 2.0 * cur as f64 * field / m as f64;
                    if m > 1 {
                        // Periodic in the Trotter direction, which is what makes the slices a ring
                        // rather than a chain with two special ends.
                        let up = s[((k + m - 1) % m) * n + i] as f64;
                        let down = s[((k + 1) % m) * n + i] as f64;
                        d += 2.0 * cur as f64 * j_perp * (up + down);
                    }
                    proposals += 1;
                    if d <= 0.0 || rng.f64() < (-p.beta * d).exp() {
                        s[k * n + i] = -cur;
                        accepted += 1;
                    }
                }
            }
            if let Some(l) = ledger.as_deref_mut() {
                l.samples += (m * n) as u64;
            }
        }

        // Every slice is a classical state, and the best of them is the answer. Reading only the
        // first slice would throw away M-1 samples per step for no reason.
        for k in 0..m {
            let e = g.energy(&s[k * n..(k + 1) * n]);
            if e < best_e {
                best_e = e;
                best.copy_from_slice(&s[k * n..(k + 1) * n]);
            }
        }
    }
    let energy = g.energy(&best);
    Outcome { state: best, energy, proposals, accepted, max_j_perp }
}

#[cfg(test)]
mod tests {

    /// THE TROTTER COUPLING HAS AN EXACT ORACLE, AND THE SHIPPED ONE FAILED IT.
    ///
    /// For a single spin the transverse-field Ising model is solvable in closed form:
    /// `<sz> = (h/E) tanh(beta E)` with `E = sqrt(h^2 + Gamma^2)`. And the classical (d+1)
    /// system the Suzuki-Trotter mapping produces is, for n = 1, a ring of `M` spins with field
    /// `h/M` and coupling `J_perp` -- which a 2x2 transfer matrix solves exactly, for any `M`, with
    /// no sampling at all. So the mapping itself can be checked against the model it claims to
    /// represent, without a sampler and without a stopwatch.
    ///
    /// It did not hold. With `J_perp = (M/2beta) ln coth(beta Gamma / M)` the ring's magnetisation
    /// is `tanh(beta h)` -- the CLASSICAL value, identical at every `M` and completely independent
    /// of `Gamma`. The slices were locked so rigidly that the transverse field did nothing, so this
    /// module was running classical annealing on `M` redundant copies of the spins and calling it
    /// quantum. The correct coupling satisfies `beta J_perp = (1/2) ln coth(beta Gamma / M)`, and
    /// with it the ring converges to the quantum answer as `M` grows, which is what this asserts.
    #[test]
    fn the_trotter_mapping_converges_to_the_quantum_model_it_claims_to_represent() {
        // Magnetisation of a ring of M spins, field h/M and coupling j_perp, by transfer matrix.
        fn ring_mean(h: f64, beta: f64, m: usize, j_perp: f64) -> f64 {
            let b = h / m as f64;
            let sp = [1.0f64, -1.0];
            let mut tm = [[0.0f64; 2]; 2];
            for i in 0..2 {
                for j in 0..2 {
                    tm[i][j] = (beta * (j_perp * sp[i] * sp[j] + b * (sp[i] + sp[j]) / 2.0)).exp();
                }
            }
            // T^m by repeated squaring, renormalised each step so large m cannot overflow -- the
            // shipped coupling overflowed f64 at m = 64, which is its own signal.
            let mut acc = [[1.0f64, 0.0], [0.0, 1.0]];
            let mut base = tm;
            let mut k = m;
            while k > 0 {
                if k & 1 == 1 {
                    acc = mul_norm(acc, base);
                }
                base = mul_norm(base, base);
                k >>= 1;
            }
            let z = acc[0][0] + acc[1][1];
            (sp[0] * acc[0][0] + sp[1] * acc[1][1]) / z
        }
        fn mul_norm(a: [[f64; 2]; 2], b: [[f64; 2]; 2]) -> [[f64; 2]; 2] {
            let mut c = [[0.0f64; 2]; 2];
            for i in 0..2 {
                for j in 0..2 {
                    c[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j];
                }
            }
            let s = c.iter().flatten().fold(0.0f64, |m, &x| m.max(x.abs())).max(1e-300);
            for row in c.iter_mut() {
                for v in row.iter_mut() {
                    *v /= s;
                }
            }
            c
        }
        // The coupling this module computes, extracted so the test uses the shipped formula.
        fn j_perp(beta: f64, gamma: f64, m: usize) -> f64 {
            let x = (beta * gamma / m as f64).tanh().max(1e-300);
            -(1.0 / (2.0 * beta)) * x.ln()
        }

        for &(h, gamma, beta) in &[(0.5f64, 1.0f64, 1.0f64), (1.0, 2.0, 1.0), (0.5, 0.5, 2.0)] {
            let e = (h * h + gamma * gamma).sqrt();
            let quantum = (h / e) * (beta * e).tanh();
            let classical = (beta * h).tanh();
            // The two answers must be far apart, or converging to one says nothing about the other.
            assert!(
                (quantum - classical).abs() > 0.05,
                "this case cannot tell the two apart: quantum {quantum:.4} classical {classical:.4}"
            );

            let mut prev = f64::INFINITY;
            for &m in &[8usize, 32, 128, 512] {
                let got = ring_mean(h, beta, m, j_perp(beta, gamma, m));
                let err = (got - quantum).abs();
                assert!(
                    err < prev + 1e-12,
                    "error must not grow with M: M={m} gave {got:.6} (err {err:.2e}), \
                     previous error {prev:.2e}"
                );
                prev = err;
            }
            assert!(
                prev < 2e-4,
                "M=512 must reach the quantum answer {quantum:.6}, and the error was {prev:.2e}"
            );
        }
    }
    use super::*;
    use crate::graph::GraphBuilder;
    use crate::ising::lattice2d;

    fn glass(l: usize, seed: u64) -> Graph {
        let mut rng = Pcg::new(seed, 0x0005_9AC0);
        let mut gb = GraphBuilder::new(l * l);
        for y in 0..l {
            for x in 0..l {
                let i = y * l + x;
                gb.couple(i, y * l + (x + 1) % l, if rng.f64() < 0.5 { 1.0 } else { -1.0 });
                gb.couple(i, ((y + 1) % l) * l + x, if rng.f64() < 0.5 { 1.0 } else { -1.0 });
            }
        }
        gb.build()
    }

    #[test]
    fn the_energy_returned_is_the_energy_of_the_state_returned() {
        for seed in 0..8u64 {
            let g = glass(6, seed);
            let out = run(&g, &Params::default(), seed);
            assert_eq!(out.state.len(), g.n);
            assert!(out.state.iter().all(|&v| v == 1 || v == -1));
            assert!((out.energy - g.energy(&out.state)).abs() < 1e-9);
            assert_eq!(out.proposals, (Params::default().trotter * g.n * Params::default().steps) as u64);
            assert!(out.accepted <= out.proposals);
        }
    }

    /// A ferromagnet has a known ground state, and any annealer that cannot reach it is broken
    /// rather than merely unlucky.
    #[test]
    fn it_reaches_a_known_ground_state() {
        for l in [4usize, 6, 8] {
            let g = lattice2d(l, 1.0);
            let out = run(&g, &Params::default(), 3);
            let bonds = 2.0 * (l * l) as f64;
            assert!(
                (out.energy + bonds).abs() < 1e-9,
                "{l}x{l}: reached {} against a ground energy of {}",
                out.energy,
                -bonds
            );
        }
    }

    /// The transverse field has to DO something, and the control is the same code with one slice.
    ///
    /// `M = 1` drops the Trotter term and is exactly classical annealing at the same temperature,
    /// same seed and the same number of proposals per step — so a difference between the arms is
    /// the quantum term and nothing else. On a frustrated glass, which is where tunnelling is
    /// supposed to matter, more slices must not be worse.
    #[test]
    fn more_trotter_slices_are_not_worse_on_a_frustrated_glass() {
        let (mut wins, mut losses) = (0, 0);
        for seed in 0..24u64 {
            let g = glass(10, seed);
            // Matched WORK, not matched steps: one slice at M sweeps per step does the same number
            // of proposals as M slices at one. Comparing at equal steps would hand the quantum arm
            // four times the budget and prove nothing.
            let classical = Params { trotter: 1, sweeps_per_step: 4, ..Params::default() };
            let quantum = Params { trotter: 4, sweeps_per_step: 1, ..Params::default() };
            let c = run(&g, &classical, seed);
            let q = run(&g, &quantum, seed);
            assert_eq!(c.proposals, q.proposals, "the arms must do equal work");
            if q.energy < c.energy - 1e-9 {
                wins += 1;
            } else if q.energy > c.energy + 1e-9 {
                losses += 1;
            }
        }
        assert!(
            wins >= losses,
            "with four Trotter slices the anneal won {wins} and lost {losses} of 24 against the \
             same code at one slice and equal work"
        );
    }

    /// `Γ` must not be annealed to zero, and the reported `J⊥` is how a reader checks that.
    #[test]
    fn the_transverse_field_stops_short_of_zero() {
        let g = glass(5, 1);
        let out = run(&g, &Params::default(), 1);
        assert!(out.max_j_perp.is_finite(), "J_perp went to infinity: gamma reached zero");
        assert!(out.max_j_perp > 0.0);

        // A caller who asks for zero is clamped rather than divided by, and the run still returns.
        let zeroed = run(&g, &Params { gamma_min: 0.0, ..Params::default() }, 1);
        assert!(zeroed.max_j_perp.is_finite() && zeroed.energy.is_finite());

        // One slice has no Trotter coupling at all, which is the documented classical case.
        let one = run(&g, &Params { trotter: 1, ..Params::default() }, 1);
        assert_eq!(one.max_j_perp, 0.0);
    }

    #[test]
    fn an_empty_graph_returns_rather_than_panicking() {
        let out = run(&GraphBuilder::new(0).build(), &Params::default(), 1);
        assert!(out.state.is_empty() && out.energy == 0.0 && out.proposals == 0);
    }
}
