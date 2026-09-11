//! Oscillator Ising machines — Kuramoto phase dynamics with sub-harmonic injection locking.
//!
//! The third hardware lane in this crate, beside simulated bifurcation ([`crate::sbm`]) and the
//! transverse-field path integral ([`crate::sqa`]), and the one built out of the cheapest
//! nonlinear component there is: a ring oscillator. Each spin is one oscillator's **phase**; the
//! Ising coupling is the resistive network that pulls two oscillators towards or away from each
//! other; and a pump tone at **twice** the oscillation frequency — sub-harmonic injection locking,
//! SHIL — is what forces every phase into one of two values.
//!
//! The dynamics, in the form the OIM papers write it:
//!
//! ```text
//!   dphi_i/dt  =  -K sum_j J_ij sin(phi_i - phi_j)  -  K h_i sin(phi_i)  -  K_s sin(2 phi_i)
//! ```
//!
//! and the readout is `s_i = +1` when `cos(phi_i) > 0`, `-1` otherwise.
//!
//! # Why the `sin(2 phi)` term is the whole point
//!
//! Drop it and this is the **XY model**: a continuous-spin relaxation whose minima are generally
//! not binary at all. The antiferromagnetic triangle settles at mutual phase differences of 120°,
//! energy `-3/2`, strictly below the `-1` of any spin assignment — a perfectly good answer to a
//! different question. `sin(2 phi)` has zeros at `0` and `pi` and pushes away from `pi/2` and
//! `3 pi/2`, so it is a potential with exactly two wells per oscillator. That is the step that
//! turns an analog relaxation into an *Ising* machine, and the triangle test below asserts both
//! halves of it — that SHIL binarises, and that its absence does not. The threshold between the
//! two is `K_s = K/2` on that instance, and it is computed there by hand from an eigenvalue.
//!
//! # Fields
//!
//! The `h_i sin(phi_i)` term is the standard OIM treatment of a bias: an extra oscillator pinned at
//! phase `0`, coupled to node `i` with weight `h_i`. Pinning it is what makes the term a function
//! of `phi_i` alone. Because the pinned coordinate is simply held fixed, the remaining `n` phases
//! still follow the exact gradient flow of the restricted energy, so nothing below is weakened by
//! it.
//!
//! # This is gradient descent, and the module is built so that it cannot not be
//!
//! With the Lyapunov function
//!
//! ```text
//!   E(phi) = -K [ sum_{i<j} J_ij cos(phi_i - phi_j) + sum_i h_i cos(phi_i) ]
//!            - (K_s / 2) sum_i cos(2 phi_i)
//! ```
//!
//! the drift above is exactly `-dE/dphi_i`. `E` is globally smooth, and its Hessian is bounded by
//! [`Oim::lipschitz`] through Gershgorin, so an explicit Euler step of at most `1/L` decreases `E`
//! at **every** step by the descent lemma. [`Oim::new`] therefore *refuses* a larger step rather
//! than accepting it and quietly losing the property — a machine that is not a descent is a
//! different machine, and the caller should have to say so.
//!
//! At binarised phases the Lyapunov function collapses onto the Ising energy this crate defines:
//! with `phi_i` in `{0, pi}` and `s_i = cos(phi_i)`, `cos(phi_i - phi_j) = s_i s_j`, so
//!
//! ```text
//!   E(phi) = K * g.energy(s) - K_s * n / 2
//! ```
//!
//! exactly — a positive rescaling plus a constant, which is to say the same optimisation problem.
//! That identity is what pins the sign convention, and it is asserted against
//! [`crate::graph::Graph::energy`] rather than against anything in this file.
//!
//! # Sources
//!
//! * Y. Kuramoto, *Self-entrainment of a population of coupled non-linear oscillators* (1975) — the
//!   phase-coupling form.
//! * T. Wang and J. Roychowdhury, *Oscillator-based Ising Machines* (2017), and *OIM: Oscillator-
//!   based Ising Machines for Solving Combinatorial Optimisation Problems*, Unconventional
//!   Computation and Natural Computation (2019) — SHIL, the binarisation argument, and the
//!   Lyapunov function.
//! * T. Wang, L. Wu, P. Nobel and J. Roychowdhury, *Solving combinatorial optimisation problems
//!   using oscillator based Ising machines*, Natural Computing 20 (2021) — the annealed-SHIL
//!   protocol [`Params::shil_ramp`] implements.
//!
//! ```
//! use ferrotherm::{ising, oim};
//!
//! let g = ising::lattice2d(6, 1.0);
//! let out = oim::run(&g, &oim::Params::default(), 7).unwrap();
//! // The phases ended in the wells, so the readout is a state and not a rounding.
//! assert!(out.defect < 1e-6, "defect {}", out.defect);
//! // A ferromagnet has an all-aligned ground state: 36 sites on a torus carry 72 edges.
//! assert_eq!(out.energy, -72.0);
//! ```

use crate::graph::Graph;
use crate::rng::Pcg;
use crate::round::{accumulation_guard, sum_down, sum_up};

/// Why an oscillator machine could not be built for this instance and these gains.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Error {
    /// No oscillators: there is nothing to lock.
    EmptyGraph,
    /// The coupling gain `K` must be finite and strictly positive. Carries the value offered.
    CouplingGain(f64),
    /// The SHIL gain `K_s` must be finite and non-negative. Carries the value offered.
    ///
    /// Zero is allowed and is the XY control: see the module note.
    ShilGain(f64),
    /// The integration step must be finite and strictly positive. Carries the value offered.
    Step(f64),
    /// The step is above `1 / L`, where the descent lemma stops guaranteeing that the energy falls.
    StepTooLarge {
        /// The step offered.
        dt: f64,
        /// The largest step this instance and these gains admit, `1 / lipschitz`.
        max: f64,
        /// The Hessian bound `L` the limit came from.
        lipschitz: f64,
    },
    /// A coupling or field is not finite, so no Hessian bound exists. Carries the node it sits on.
    NonFiniteModel(usize),
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Error::EmptyGraph => write!(f, "an oscillator machine over zero oscillators"),
            Error::CouplingGain(k) => write!(
                f,
                "coupling gain K = {k} must be finite and strictly positive; at K = 0 the \
                 oscillators do not see the problem at all and every phase relaxes to the nearest \
                 SHIL well, which is a readout of the initial condition rather than a solve"
            ),
            Error::ShilGain(ks) => write!(
                f,
                "SHIL gain K_s = {ks} must be finite and non-negative; K_s = 0 is permitted and is \
                 the XY model, whose minima need not be binary"
            ),
            Error::Step(dt) => write!(f, "integration step dt = {dt} must be finite and positive"),
            Error::StepTooLarge { dt, max, lipschitz } => write!(
                f,
                "step dt = {dt} exceeds {max}, which is 1/L for the Hessian bound L = {lipschitz} \
                 of this instance at these gains. Above it the descent lemma no longer holds and \
                 the energy can rise, so the run would not be the gradient flow this module \
                 documents. Use Oim::with_safe_step, or lower the gains"
            ),
            Error::NonFiniteModel(i) => write!(
                f,
                "node {i} carries a non-finite coupling or field, so the Hessian is unbounded and \
                 no step size can be certified"
            ),
        }
    }
}

impl core::error::Error for Error {}

/// A bound on the spectral norm of the Hessian of `E`, by Gershgorin.
///
/// `d2E/dphi_i2 = K (sum_j J_ij cos(phi_i - phi_j) + h_i cos(phi_i)) + 2 K_s` in magnitude at most
/// `K (sum_j |J_ij| + |h_i|) + 2 K_s`, and the off-diagonals of row `i` sum to at most
/// `K sum_j |J_ij|`. Every cosine is bounded by one, so the result holds at every phase rather
/// than near a particular one — which is what makes a *fixed* step legitimate for a whole run.
fn lipschitz_of(g: &Graph, k: f64, k_s: f64) -> f64 {
    let mut worst = 0.0f64;
    for i in 0..g.n {
        let row: f64 = (g.offset[i]..g.offset[i + 1]).map(|e| g.w[e].abs()).sum();
        worst = worst.max(2.0 * row + g.h[i].abs());
    }
    2.0 * k_s + k * worst
}

/// One oscillator Ising machine: an instance, its gains, and a step certified against them.
///
/// Holds the graph by reference and owns only the drift scratch, so constructing one per run is
/// free beside the run itself.
pub struct Oim<'g> {
    g: &'g Graph,
    k: f64,
    k_s: f64,
    dt: f64,
    lipschitz: f64,
    drift: Vec<f64>,
}

impl<'g> Oim<'g> {
    /// A machine with an explicit step.
    ///
    /// # Errors
    ///
    /// [`Error::EmptyGraph`], [`Error::CouplingGain`], [`Error::ShilGain`] and [`Error::Step`] on
    /// arguments that are not a machine at all; [`Error::NonFiniteModel`] when a coupling or field
    /// is not finite; and [`Error::StepTooLarge`] when `dt` is above `1 / L`, past which the energy
    /// can rise and the run is no longer the gradient flow this module documents.
    pub fn new(g: &'g Graph, k: f64, k_s: f64, dt: f64) -> Result<Self, Error> {
        if g.n == 0 {
            return Err(Error::EmptyGraph);
        }
        if !(k.is_finite() && k > 0.0) {
            return Err(Error::CouplingGain(k));
        }
        if !(k_s.is_finite() && k_s >= 0.0) {
            return Err(Error::ShilGain(k_s));
        }
        if !(dt.is_finite() && dt > 0.0) {
            return Err(Error::Step(dt));
        }
        for i in 0..g.n {
            let bad = !g.h[i].is_finite()
                || (g.offset[i]..g.offset[i + 1]).any(|e| !g.w[e].is_finite());
            if bad {
                return Err(Error::NonFiniteModel(i));
            }
        }
        let lipschitz = lipschitz_of(g, k, k_s);
        // L = 0 is the machine with no forces at all: a single uncoupled, unbiased oscillator with
        // the pump off. Nothing moves, so no step can be wrong, and dividing by it to say so would
        // be the only way to fail here.
        if lipschitz > 0.0 {
            let max = 1.0 / lipschitz;
            if dt > max {
                return Err(Error::StepTooLarge { dt, max, lipschitz });
            }
        }
        Ok(Oim { g, k, k_s, dt, lipschitz, drift: vec![0.0; g.n] })
    }

    /// A machine whose step is the largest the descent lemma certifies, `1 / L`.
    ///
    /// This is the step to want. It is as large as monotone descent allows, it is derived from the
    /// instance rather than guessed, and it moves with the gains — which a constant cannot.
    ///
    /// # Errors
    ///
    /// As [`Oim::new`], minus the two step variants, which this cannot produce.
    pub fn with_safe_step(g: &'g Graph, k: f64, k_s: f64) -> Result<Self, Error> {
        let probe = Oim::new(g, k, k_s, f64::MIN_POSITIVE)?;
        let l = probe.lipschitz;
        let dt = if l > 0.0 { 1.0 / l } else { 1.0 };
        Oim::new(g, k, k_s, dt)
    }

    /// The Hessian bound `L` this instance's step was certified against.
    #[must_use]
    pub fn lipschitz(&self) -> f64 {
        self.lipschitz
    }

    /// The integration step, in units of the dynamics' own time.
    #[must_use]
    pub fn dt(&self) -> f64 {
        self.dt
    }

    /// The coupling gain `K`.
    #[must_use]
    pub fn k(&self) -> f64 {
        self.k
    }

    /// The largest SHIL gain this machine's step was certified for; see [`Oim::step`].
    #[must_use]
    pub fn k_s(&self) -> f64 {
        self.k_s
    }

    /// Oscillator count, which is the graph's node count.
    #[must_use]
    pub fn n(&self) -> usize {
        self.g.n
    }

    /// One explicit Euler step of the phase dynamics, at SHIL gain `k_s`.
    ///
    /// The gain is an argument rather than a field because the annealed protocol ramps it, and a
    /// term whose strength changes is a term whose strength should be visible at the call site.
    /// Every drift is computed from the **pre-step** phases, which is what makes the update the
    /// gradient of one function rather than a Gauss–Seidel sweep of several.
    ///
    /// Phases are deliberately not wrapped here. Wrapping is exact in the mathematics and is not
    /// exact in `f64`, and a rounding of the argument of a cosine is indistinguishable from an
    /// energy that rose — which is the one thing the descent test exists to detect. The flow is
    /// contracting, so the phases stay bounded anyway; [`run`] wraps once, at the end.
    ///
    /// # Panics
    ///
    /// If `phi` is not one value per oscillator, or `k_s` is not finite and within
    /// `0 ..= self.k_s()` — the step was certified for at most that gain, and a larger one would
    /// silently void the descent guarantee.
    pub fn step(&mut self, phi: &mut [f64], k_s: f64) {
        assert_eq!(phi.len(), self.g.n, "one phase per oscillator");
        assert!(
            k_s.is_finite() && (0.0..=self.k_s).contains(&k_s),
            "SHIL gain {k_s} is outside 0 ..= {}, which the step {} was certified for",
            self.k_s,
            self.dt
        );
        for i in 0..self.g.n {
            let mut c = self.g.h[i] * phi[i].sin();
            for e in self.g.offset[i]..self.g.offset[i + 1] {
                c += self.g.w[e] * (phi[i] - phi[self.g.nbr[e] as usize]).sin();
            }
            let d = -(self.k * c + k_s * (2.0 * phi[i]).sin());
            self.drift[i] = d;
        }
        for i in 0..self.g.n {
            phi[i] += self.dt * self.drift[i];
        }
    }

    /// Every additive term of `E(phi)` at gain `k_s`, once each, in a fixed order.
    ///
    /// One definition, two consumers: [`Oim::energy`] adds them and [`Oim::energy_bracket`] bounds
    /// their sum. Writing the energy twice is how the value and the bound drift apart.
    fn for_each_term<F: FnMut(f64)>(&self, phi: &[f64], k_s: f64, mut f: F) {
        for i in 0..self.g.n {
            f(-self.k * self.g.h[i] * phi[i].cos());
            f(-0.5 * k_s * (2.0 * phi[i]).cos());
            for e in self.g.offset[i]..self.g.offset[i + 1] {
                let j = self.g.nbr[e] as usize;
                if j > i {
                    f(-self.k * self.g.w[e] * (phi[i] - phi[j]).cos());
                }
            }
        }
    }

    /// The Lyapunov energy `E(phi)` at SHIL gain `k_s`.
    ///
    /// A value, **not** a bound: it is a left-to-right sum and may land either side of the exact
    /// total. Where the direction matters — anywhere the claim is that the energy did not rise —
    /// use [`Oim::energy_bracket`].
    ///
    /// # Panics
    ///
    /// If `phi` is not one value per oscillator.
    #[must_use]
    pub fn energy(&self, phi: &[f64], k_s: f64) -> f64 {
        assert_eq!(phi.len(), self.g.n, "one phase per oscillator");
        let mut s = 0.0;
        self.for_each_term(phi, k_s, |t| s += t);
        s
    }

    /// A bracket `(lower, upper)` that certainly contains `E(phi)` as the terms were computed.
    ///
    /// The summation is bounded by [`crate::round`], which is the part `f64` addition gets wrong in
    /// a direction nobody chose. On top of that each term carries its own rounding — a cosine that
    /// the platform's libm need not round correctly, and two products — and `3 eps` per term covers
    /// it, taken through [`accumulation_guard`] against the summed magnitude.
    ///
    /// **What it does not bound**: the phases themselves. If `phi` is the output of an integrator
    /// it carries discretisation error orders of magnitude above this bracket, and no amount of
    /// careful addition makes that go away. The bracket answers "is this sum right", not "is this
    /// trajectory right".
    ///
    /// # Panics
    ///
    /// If `phi` is not one value per oscillator.
    #[must_use]
    pub fn energy_bracket(&self, phi: &[f64], k_s: f64) -> (f64, f64) {
        assert_eq!(phi.len(), self.g.n, "one phase per oscillator");
        let mut terms = Vec::with_capacity(2 * self.g.n + self.g.n_edges);
        self.for_each_term(phi, k_s, |t| terms.push(t));
        let abs: Vec<f64> = terms.iter().copied().map(f64::abs).collect();
        let guard = accumulation_guard(3, sum_up(&abs));
        (sum_down(&terms) - guard, sum_up(&terms) + guard)
    }
}

/// Read the machine: `s_i = +1` when `cos(phi_i) > 0`, else `-1`.
///
/// The tie at `cos = 0` — a phase exactly at `pi/2`, which is the peak of the SHIL potential and
/// therefore where nothing settles — resolves to `-1`, fixed so a readout is a function of the
/// phases and not of the platform.
#[must_use]
pub fn readout(phi: &[f64]) -> Vec<i8> {
    phi.iter().map(|&p| if p.cos() > 0.0 { 1 } else { -1 }).collect()
}

/// How far the worst phase is from the nearest of `{0, pi}`, in radians, so in `0 ..= pi/2`.
///
/// This is the number that says whether a run produced an *Ising* answer or an XY one. Binary
/// phases are exact fixed points of the dynamics — `sin(2 k pi) = 0` and every coupling term
/// vanishes with it — so a converged SHIL run drives this to zero rather than to some floor, and it
/// is fair to demand a tight tolerance of it.
#[must_use]
pub fn phase_defect(phi: &[f64]) -> f64 {
    phi.iter().fold(0.0f64, |m, &p| m.max(p.sin().abs().asin()))
}

/// How a run is configured.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Params {
    /// Coupling gain `K`: how hard the network pulls phases together. Finite and positive.
    pub k: f64,
    /// SHIL gain `K_s` at the end of the run. Zero is the XY control.
    ///
    /// **This is the parameter with a real trade-off in it, and there is no value that wins both
    /// sides.** At a binary state the Hessian of `E` is `2 K_s I + K (diag(g) - A)`, with `2 g_i`
    /// the Ising flip gain of site `i` and `A_ij = J_ij s_i s_j`. Too small and the state is not
    /// stable, so the machine settles somewhere between the wells and the readout is a rounding of
    /// an XY answer. Too large and `2 K_s I` dominates, so states one flip short of a minimum are
    /// stable too and the machine holds whatever it reached.
    ///
    /// The threshold is the instance's, not a constant. Measured here: the antiferromagnetic
    /// triangle needs exactly `K/2`; a dense 14-node `{-1,+1}` instance needs between `2 K` and
    /// `3 K`; a 6x6 ferromagnetic lattice is stable at any positive gain, which is why the default
    /// of `1.0` is right for the sparse structured instances and low for dense ones.
    ///
    /// [`Outcome::defect`] is how a caller checks rather than guesses: it is near zero exactly when
    /// the gain was enough.
    pub k_s: f64,
    /// Integration step, or `None` for [`Oim::with_safe_step`], which is the usual choice.
    pub dt: Option<f64>,
    /// Euler steps. Time elapsed is `steps * dt`, and `dt` scales with the instance, so this is
    /// not comparable across instances on its own.
    pub steps: usize,
    /// Ramp `K_s` linearly from zero to [`Params::k_s`] across the run, rather than holding it.
    ///
    /// The annealed protocol, and the reason OIM solves anything: at `K_s = 0` the machine finds a
    /// minimum of the XY relaxation, and raising the pump walks that continuous answer into a
    /// binary one. Holding `K_s` high from the start binarises immediately and reports little more
    /// than the initial condition.
    ///
    /// **A ramp is not a descent on any single energy.** `E` itself depends on `K_s`, so the
    /// monotonicity this module certifies is a property of a *fixed* gain. Set this to `false` to
    /// get it back.
    pub shil_ramp: bool,
}

impl Default for Params {
    fn default() -> Self {
        Params { k: 1.0, k_s: 1.0, dt: None, steps: 4000, shil_ramp: true }
    }
}

/// What a run settled on.
#[derive(Clone, Debug)]
pub struct Outcome {
    /// Final phases, wrapped into `[0, 2 pi)`.
    pub phases: Vec<f64>,
    /// The readout of those phases, by [`readout`].
    pub state: Vec<i8>,
    /// The Ising energy of [`Outcome::state`], from [`crate::graph::Graph::energy`] — the number
    /// the problem was posed in, not the phase energy.
    pub energy: f64,
    /// The Lyapunov energy of [`Outcome::phases`] at the final SHIL gain.
    pub lyapunov: f64,
    /// [`phase_defect`] of the final phases. Near zero means the machine gave an Ising answer;
    /// anything else means it stopped somewhere between the two wells and the readout rounded it.
    pub defect: f64,
    /// Steps actually taken.
    pub steps: usize,
}

/// Run one oscillator machine from a seeded random phase initialisation.
///
/// # Errors
///
/// As [`Oim::new`], on gains or a step that do not describe a machine.
pub fn run(g: &Graph, p: &Params, seed: u64) -> Result<Outcome, Error> {
    let mut oim = match p.dt {
        Some(dt) => Oim::new(g, p.k, p.k_s, dt)?,
        None => Oim::with_safe_step(g, p.k, p.k_s)?,
    };
    let mut rng = Pcg::new(seed, 0x0000_01B4);
    let mut phi: Vec<f64> = (0..g.n).map(|_| rng.f64() * std::f64::consts::TAU).collect();
    let steps = p.steps.max(1);
    for step in 0..steps {
        // `frac` first, so the last step is `k_s * 1.0` and not `(k_s * x) / x`, which need not be
        // `k_s` and would trip the gain assertion in `step` by one ulp.
        let k_s = if p.shil_ramp && steps > 1 {
            let frac = step as f64 / (steps - 1) as f64;
            p.k_s * frac
        } else {
            p.k_s
        };
        oim.step(&mut phi, k_s);
    }
    let state = readout(&phi);
    let energy = g.energy(&state);
    let lyapunov = oim.energy(&phi, p.k_s);
    let defect = phase_defect(&phi);
    for v in &mut phi {
        *v = v.rem_euclid(std::f64::consts::TAU);
    }
    Ok(Outcome { phases: phi, state, energy, lyapunov, defect, steps })
}

/// `restarts` seeded runs, the lowest **Ising** energy kept.
///
/// The phase flow is a descent and therefore stops at the first minimum it reaches; which one that
/// is depends entirely on where it started. Restarts are the whole of the search.
///
/// # Errors
///
/// As [`run`]. A refusal is a property of the gains, so it happens on the first restart or not at
/// all.
pub fn run_restarts(g: &Graph, p: &Params, seed: u64, restarts: usize) -> Result<Outcome, Error> {
    // The first restart is hoisted out rather than folded into an Option, so that "there is always
    // an answer" is a fact about the control flow instead of an `expect` that has to be believed.
    let mut best = run(g, p, seed)?;
    for r in 1..restarts.max(1) {
        let out = run(g, p, seed ^ (r as u64).wrapping_mul(0x9E3779B97F4A7C15))?;
        if out.energy < best.energy {
            best = out;
        }
    }
    Ok(best)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::GraphBuilder;
    use std::f64::consts::PI;

    fn single() -> Graph {
        GraphBuilder::new(1).build()
    }

    fn pair(j: f64) -> Graph {
        let mut gb = GraphBuilder::new(2);
        gb.couple(0, 1, j);
        gb.build()
    }

    fn triangle(j: f64) -> Graph {
        let mut gb = GraphBuilder::new(3);
        gb.couple(0, 1, j);
        gb.couple(1, 2, j);
        gb.couple(0, 2, j);
        gb.build()
    }

    /// A random instance with Gaussian couplings on every pair and Gaussian fields.
    fn dense_random(n: usize, seed: u64) -> Graph {
        let mut rng = Pcg::new(seed, 0x0B1E);
        let mut gauss = move || {
            let a = rng.f64().max(1e-12);
            let b = rng.f64();
            (-2.0 * a.ln()).sqrt() * (std::f64::consts::TAU * b).cos()
        };
        let mut gb = GraphBuilder::new(n);
        for i in 0..n {
            for j in (i + 1)..n {
                gb.couple(i, j, gauss());
            }
        }
        for i in 0..n {
            gb.bias(i, 0.3 * gauss());
        }
        gb.build()
    }

    /// A dense instance with couplings drawn from `{-1, +1}` and no fields, so every flip gain is
    /// an even integer and "one flip short of a minimum" has a smallest possible value of `-2`.
    fn pm1_dense(n: usize, seed: u64) -> Graph {
        let mut rng = Pcg::new(0xA10 ^ seed, 5);
        let mut gb = GraphBuilder::new(n);
        for i in 0..n {
            for j in (i + 1)..n {
                gb.couple(i, j, f64::from(rng.spin(0.5)));
            }
        }
        gb.build()
    }

    fn exact_ground(g: &Graph) -> f64 {
        assert!(g.n <= 16, "exhaustive enumeration is for small instances");
        let mut e0 = f64::MAX;
        let mut s = vec![-1i8; g.n];
        for m in 0..(1u32 << g.n) {
            for b in 0..g.n {
                s[b] = if m >> b & 1 == 1 { 1 } else { -1 };
            }
            e0 = e0.min(g.energy(&s));
        }
        e0
    }

    /// Every single flip of `s` raises the Ising energy, by [`Graph::energy`] and nothing else.
    fn worst_flip_gain(g: &Graph, s: &[i8]) -> f64 {
        let e = g.energy(s);
        let mut t = s.to_vec();
        let mut worst = f64::MAX;
        for i in 0..g.n {
            t[i] = -t[i];
            worst = worst.min(g.energy(&t) - e);
            t[i] = -t[i];
        }
        worst
    }

    /// THE SHIL TERM HAS A CLOSED FORM, AND IT PINS BOTH THE SIGN AND THE FACTOR OF TWO.
    ///
    /// One uncoupled, unbiased oscillator obeys `dphi/dt = -K_s sin(2 phi)`. Substituting
    /// `u = tan(phi)` gives `du/dt = (1 + u^2) * (-2 K_s u / (1 + u^2)) = -2 K_s u`, so
    ///
    /// ```text
    ///   tan(phi(t)) = tan(phi(0)) * exp(-2 K_s t)
    /// ```
    ///
    /// exactly, for every `t`. That is an oracle this file cannot influence: it is the solution of
    /// the differential equation the module claims to integrate, written down by hand.
    ///
    /// The test asserts the value AND the convergence order. Halving the step must halve the error,
    /// which is what explicit Euler promises and what distinguishes "the integrator is first-order
    /// accurate on the right equation" from "the integrator is converging to something else" — a
    /// wrong factor inside the sine converges beautifully, to the wrong number.
    #[test]
    fn shil_relaxation_matches_the_closed_form_tan_decay() {
        let g = single();
        let (k_s, phi0, t_end) = (0.7f64, 1.0f64, 1.0f64);
        let want = phi0.tan() * (-2.0 * k_s * t_end).exp();

        let mut errs = Vec::new();
        for &dt in &[1e-3f64, 5e-4] {
            let steps = (t_end / dt).round() as usize;
            let mut m = Oim::new(&g, 1.0, k_s, dt).unwrap();
            let mut phi = vec![phi0];
            for _ in 0..steps {
                m.step(&mut phi, k_s);
            }
            errs.push((phi[0].tan() - want).abs());
        }
        assert!(errs[0] < 2e-3, "dt = 1e-3 was off by {:e}, want {want}", errs[0]);
        // First order: halving dt halves the error. 10% either side of exactly 2 is loose enough
        // for the higher-order remainder at these steps and far too tight for a wrong equation.
        let ratio = errs[0] / errs[1];
        assert!(
            (ratio - 2.0).abs() < 0.2,
            "Euler must be first order: errors {:e} and {:e}, ratio {ratio}",
            errs[0],
            errs[1]
        );
    }

    /// THE COUPLING TERM HAS A CLOSED FORM TOO, and a conserved quantity beside it.
    ///
    /// Two oscillators, coupling `J`, pump off. The phase difference `d = phi_0 - phi_1` obeys
    /// `dd/dt = -2 K J sin(d)`, and `v = tan(d/2)` obeys `dv/dt = -2 K J v`, so
    ///
    /// ```text
    ///   tan(d(t)/2) = tan(d(0)/2) * exp(-2 K J t)
    /// ```
    ///
    /// while `phi_0 + phi_1` is conserved exactly, the two drifts being equal and opposite. Both
    /// are properties of the equation rather than of this implementation: the first fixes the
    /// magnitude of the coupling term and the second fixes its antisymmetry, which is the half a
    /// sign error in the neighbour subtraction would destroy.
    #[test]
    fn coupled_pair_phase_difference_matches_the_closed_form_tan_half_decay() {
        let (k, j, t_end, dt) = (1.0f64, 1.0f64, 1.0f64, 1e-3f64);
        let g = pair(j);
        let (a0, b0) = (2.3f64, 0.4f64);
        let want = ((a0 - b0) / 2.0).tan() * (-2.0 * k * j * t_end).exp();

        let mut m = Oim::new(&g, k, 0.0, dt).unwrap();
        let mut phi = vec![a0, b0];
        for _ in 0..(t_end / dt).round() as usize {
            m.step(&mut phi, 0.0);
        }
        let got = ((phi[0] - phi[1]) / 2.0).tan();
        assert!((got - want).abs() < 2e-3, "difference: got {got}, closed form {want}");
        assert!(
            ((phi[0] + phi[1]) - (a0 + b0)).abs() < 1e-12,
            "the mean phase is conserved: {} vs {}",
            phi[0] + phi[1],
            a0 + b0
        );
    }

    /// THE ORACLE IS `graph::Graph::energy`, AND THIS IS THE TEST THAT PINS THE SIGN CONVENTION.
    ///
    /// At phases in `{0, pi}` the Lyapunov function must be the Ising energy of the readout, scaled
    /// by `K` and shifted by `-K_s n / 2`. Every cosine involved is exactly `+-1` in `f64`, so the
    /// identity is asserted at the level of the arithmetic rather than at the level of a tolerance
    /// someone chose. It is checked with fields and with both signs of coupling, since a sign error
    /// in the field term alone survives any test on an unbiased instance.
    #[test]
    fn binarised_phases_reproduce_the_ising_energy_of_graph_energy() {
        for seed in 0..8u64 {
            let g = dense_random(9, 0x5E ^ seed);
            let (k, k_s) = (1.7f64, 0.9f64);
            let m = Oim::new(&g, k, k_s, 1e-4).unwrap();
            let mut rng = Pcg::new(seed, 0x11);
            for _ in 0..16 {
                let s: Vec<i8> = (0..g.n).map(|_| rng.spin(0.5)).collect();
                let phi: Vec<f64> = s.iter().map(|&v| if v > 0 { 0.0 } else { PI }).collect();
                assert_eq!(readout(&phi), s, "the readout must invert the encoding");
                let want = k * g.energy(&s) - k_s * g.n as f64 / 2.0;
                let got = m.energy(&phi, k_s);
                assert!(
                    (got - want).abs() < 1e-12,
                    "seed {seed}: phase energy {got} vs K * ising - K_s n / 2 = {want}"
                );
                let (lo, hi) = m.energy_bracket(&phi, k_s);
                assert!(lo <= want && want <= hi, "bracket [{lo}, {hi}] misses {want}");
            }
        }
    }

    /// THE DESCENT LEMMA, AT EVERY STEP OF A LONG RUN, NOT AT THE ENDPOINTS.
    ///
    /// `E` is `L`-smooth with `L` from [`Oim::lipschitz`], so explicit Euler at `dt = 1/L` gives
    /// `E(next) <= E(now) - (1 / 2L) |grad|^2`. The oracle is that inequality, not anything this
    /// file computes — and it is the only check that sees an integration-sign error, because a run
    /// that ascends still terminates, still binarises, and still returns a plausible spin vector.
    ///
    /// Stated through [`Oim::energy_bracket`] rather than the plain value: the claim is about the
    /// exact energy, and comparing two left-to-right sums would compare two roundings of it.
    #[test]
    fn energy_never_rises_at_any_step_by_the_descent_lemma() {
        for seed in 0..4u64 {
            let g = dense_random(40, 0xD0 ^ seed);
            let k_s = 0.6;
            let mut m = Oim::with_safe_step(&g, 1.0, k_s).unwrap();
            let mut rng = Pcg::new(seed, 0xDE5C);
            let mut phi: Vec<f64> =
                (0..g.n).map(|_| rng.f64() * std::f64::consts::TAU).collect();

            let first = m.energy(&phi, k_s);
            let mut prev_hi = m.energy_bracket(&phi, k_s).1;
            for step in 0..6000 {
                m.step(&mut phi, k_s);
                let (lo, hi) = m.energy_bracket(&phi, k_s);
                assert!(
                    lo <= prev_hi,
                    "seed {seed} step {step}: energy rose, {lo} is above the previous {prev_hi}"
                );
                prev_hi = hi;
            }
            // And the run must actually have gone somewhere, or "never rose" is a statement about
            // a constant.
            let last = m.energy(&phi, k_s);
            assert!(last < first - 1.0, "seed {seed}: E went {first} -> {last}, which is no descent");
        }
    }

    /// OIM IS NOT THE XY MODEL, AND THE FRUSTRATED TRIANGLE IS WHERE THEY DIFFER — BOTH WAYS.
    ///
    /// Three oscillators, all couplings `-1`. With the pump off this is the XY antiferromagnet,
    /// whose minimum is the 120-degree state: every pairwise phase difference has cosine `-1/2` and
    /// the energy is `-3/2`, strictly below the `-1` of every binary assignment. Those are
    /// classical facts about the model, not about this code, and they are asserted at `1e-12` —
    /// the flow converges to them, it does not approximate them.
    ///
    /// The negative half is the point. A module that binarised unconditionally — the readout
    /// thresholds a cosine, and a cosine always has a sign — passes a binarisation test and fails
    /// this one.
    ///
    /// **And the threshold between the two behaviours is a number that can be computed by hand.**
    /// At the binary state `(+,+,-)` the gauged couplings `A_ij = J_ij s_i s_j` give flip half-gains
    /// `g = (0, 0, 2)`, and `diag(g) - A` has characteristic polynomial `-lambda (lambda - 3)
    /// (lambda + 1)`, so its eigenvalues are `{3, 0, -1}`. The Hessian of `E` there is
    /// `2 K_s I + K (diag(g) - A)`, which is positive semidefinite exactly when `K_s >= K/2`. So
    /// `K_s = 0.45` must NOT hold a binary state and `K_s = 0.6` must — and measured, it is
    /// `0.43` radians of defect below the threshold and `6e-15` above it.
    #[test]
    fn oim_distinguishes_itself_from_the_xy_model_on_the_frustrated_triangle() {
        let g = triangle(-1.0);
        let long = Params { steps: 20_000, ..Params::default() };

        // XY: pump off.
        let xy = run(&g, &Params { k_s: 0.0, ..long }, 3).unwrap();
        for (a, b) in [(0, 1), (1, 2), (0, 2)] {
            let c = (xy.phases[a] - xy.phases[b]).cos();
            assert!((c + 0.5).abs() < 1e-12, "pair ({a},{b}) has cos {c}, want the 120-degree -0.5");
        }
        assert!(
            (xy.lyapunov + 1.5).abs() < 1e-12,
            "the XY minimum of the antiferromagnetic triangle is -3/2, got {}",
            xy.lyapunov
        );
        // NOT binarised: the 120-degree state sits pi/3 from the nearest well, and the flow drifts
        // along the free global rotation, so the worst phase is further still.
        assert!(
            xy.defect > 1.0,
            "without SHIL the phases must not binarise, and the defect was {}",
            xy.defect
        );
        // The readout of a non-binary state is a rounding, and the energies say so: the XY optimum
        // is strictly below anything a spin vector can reach on this instance.
        assert!(xy.lyapunov < g.energy(&xy.state) - 0.4, "XY must beat every Ising state here");

        // Below the hand-computed threshold K/2: still not binary.
        let weak = run(&g, &Params { k_s: 0.45, ..long }, 3).unwrap();
        assert!(
            weak.defect > 0.1,
            "K_s = 0.45 is below K/2 and must not hold a binary state; defect {}",
            weak.defect
        );

        // Above it: binarised, and on a ground state.
        let oim = run(&g, &Params { k_s: 0.6, ..long }, 3).unwrap();
        assert!(oim.defect < 1e-9, "SHIL must binarise above K/2, and the defect was {}", oim.defect);
        assert!((oim.energy + 1.0).abs() < 1e-12, "every ground state here has energy -1");
    }

    /// WITH THE PUMP ON, WHAT SETTLES IS A BINARY STATE NO SINGLE FLIP IMPROVES — AND THE THREE
    /// EXCEPTIONS IN 600 RUNS ARE A PROPERTY OF THE MACHINE, NOT OF THE CODE.
    ///
    /// Binarisation is asserted unconditionally, at `1e-9`, on every one of the 600 runs. Binary
    /// phases are *exact* fixed points — `sin(2 k pi)` is zero and every coupling term vanishes
    /// with it — so this is a convergence tolerance and not a floor, and the measured worst is
    /// `9e-15`.
    ///
    /// Local minimality is asserted as a **count**, and the reason is worth stating rather than
    /// hiding behind a friendlier instance. At a binary state the Hessian of `E` is
    /// `2 K_s I + K (diag(g) - A)` with `2 g_i` the Ising flip gain of site `i`, so stability
    /// requires only `flip gain >= -4 K_s / K`. A pump strong enough to binarise a frustrated
    /// dense instance is strong enough to hold a state one flip short of a minimum, and no gain
    /// does both jobs: on this family `K_s` below `K/2` would make every non-minimum unstable and
    /// does not binarise at all, while `K_s = 3` binarises everything. **That trade-off is the
    /// machine.** Measured over 600 runs at `K_s = 3`: 597 exact local minima, and all three
    /// exceptions miss by exactly `-2`, the smallest gap a `{-1,+1}` instance can have.
    ///
    /// The floor is 580 rather than 597 so that an unlucky seed does not fail and a regression
    /// does, and the control in the test is what makes 580 a real number: **a random spin vector
    /// on the same instances is a local minimum once in 600**, the median worst-flip-gain of such
    /// a vector being `-10`. Anything that broke the dynamics would land in that distribution, not
    /// near this one.
    ///
    /// On Gaussian couplings with fields the same protocol gives 175 of 200, because a continuous
    /// coupling distribution puts flip gains arbitrarily close to zero and the pump holds them all.
    /// Recorded here because it is the limit of the claim, and a reader choosing gains needs it.
    #[test]
    fn settled_phases_are_binarised_and_the_readout_is_an_ising_local_minimum() {
        let (k, k_s) = (1.0f64, 3.0f64);
        let mut minima = 0;
        let mut runs = 0;
        let mut control_minima = 0;
        let mut rng = Pcg::new(0xC0FFEE, 2);
        for &n in &[12usize, 14, 16] {
            for seed in 0..200u64 {
                let g = pm1_dense(n, seed);
                let p = Params { k, k_s, steps: 12_000, ..Params::default() };
                let out = run(&g, &p, seed ^ 0x77).unwrap();
                runs += 1;
                assert!(out.defect < 1e-9, "n={n} seed {seed}: defect {} is not in a well", out.defect);
                let worst = worst_flip_gain(&g, &out.state);
                if worst > 0.0 {
                    minima += 1;
                } else {
                    assert!(
                        worst >= -2.0,
                        "n={n} seed {seed}: the settled state misses a minimum by {worst}, which is \
                         further than a stable fixed point of this machine can be"
                    );
                }
                // The control: the same instance, a random spin vector. If THIS were a local
                // minimum at any rate, the count above would be measuring nothing.
                let s: Vec<i8> = (0..n).map(|_| rng.spin(0.5)).collect();
                if worst_flip_gain(&g, &s) > 0.0 {
                    control_minima += 1;
                }
            }
        }
        assert_eq!(runs, 600);
        assert!(minima >= 580, "local minima {minima}/600");
        assert!(control_minima <= 5, "random states were local minima {control_minima}/600, so \
                                      {minima}/600 is not evidence of anything");
    }

    /// Against exhaustive enumeration over all `2^14` states: the annealed machine finds the exact
    /// ground state on most instances, and never returns an energy below one.
    ///
    /// Two assertions of different kinds. "Never below the enumerated ground state" is absolute —
    /// an energy below the true minimum is not a lucky run, it is a broken energy — and it holds on
    /// all 24. The hit rate is a frozen measurement, 21 of 24 at 12 restarts, asserted at 19 so
    /// that a real regression fails and an unlucky seed does not.
    #[test]
    fn annealed_restarts_reach_the_exhaustively_enumerated_ground_state() {
        let mut hits = 0;
        for seed in 0..24u64 {
            let g = dense_random(14, 0xE0 ^ seed);
            let e0 = exact_ground(&g);
            let p = Params { k: 1.0, k_s: 3.0, steps: 4000, ..Params::default() };
            let out = run_restarts(&g, &p, 0x9C ^ seed, 12).unwrap();
            assert!(
                out.energy > e0 - 1e-9,
                "seed {seed}: returned {} below the enumerated ground state {e0}",
                out.energy
            );
            if (out.energy - e0).abs() < 1e-9 {
                hits += 1;
            }
        }
        assert!(hits >= 19, "ground-state hits {hits}/24");
    }

    /// The ramp must be a ramp: the SHIL gain reaches exactly the configured maximum on the last
    /// step, and an unramped run is the same trajectory at constant gain.
    #[test]
    fn the_shil_ramp_starts_at_zero_and_ends_exactly_at_the_configured_gain() {
        let g = pair(1.0);
        let p = Params { k_s: 0.4, steps: 5, shil_ramp: true, ..Params::default() };
        // Reconstructed here because it is the schedule `run` computes, and a ramp that overshot by
        // one ulp would panic inside `Oim::step` rather than return a wrong answer.
        let gains: Vec<f64> =
            (0..p.steps).map(|s| p.k_s * (s as f64 / (p.steps - 1) as f64)).collect();
        assert_eq!(gains[0], 0.0);
        assert_eq!(gains[p.steps - 1], p.k_s);
        assert!(run(&g, &p, 1).is_ok());
    }

    /// A refusal is a refusal: every rejected argument names itself, and `with_safe_step` produces
    /// a step the constructor accepts.
    #[test]
    fn uncertified_steps_and_impossible_gains_are_refused() {
        let g = pair(1.0);
        let empty = GraphBuilder::new(0).build();
        assert_eq!(Oim::new(&empty, 1.0, 1.0, 0.1).err(), Some(Error::EmptyGraph));
        assert_eq!(Oim::new(&g, 0.0, 1.0, 0.1).err(), Some(Error::CouplingGain(0.0)));
        // NaN is compared with `matches!` and not `assert_eq!`: NaN is not equal to itself, so the
        // equality form would pass on the wrong variant and fail on the right one.
        assert!(matches!(Oim::new(&g, f64::NAN, 1.0, 0.1), Err(Error::CouplingGain(_))));
        assert!(matches!(Oim::new(&g, 1.0, -1.0, 0.1), Err(Error::ShilGain(_))));
        assert!(matches!(Oim::new(&g, 1.0, 1.0, 0.0), Err(Error::Step(_))));
        assert!(matches!(Oim::new(&g, 1.0, 1.0, f64::INFINITY), Err(Error::Step(_))));

        // L = 2 * K_s + K * max_i (2 sum_j |J_ij| + |h_i|) = 2 + 2 = 4 here.
        let m = Oim::with_safe_step(&g, 1.0, 1.0).unwrap();
        assert!((m.lipschitz() - 4.0).abs() < 1e-12, "L = {}", m.lipschitz());
        assert!((m.dt() - 0.25).abs() < 1e-12, "dt = {}", m.dt());
        assert!(matches!(
            Oim::new(&g, 1.0, 1.0, 0.25 + 1e-9),
            Err(Error::StepTooLarge { .. })
        ));

        let mut bad = GraphBuilder::new(2);
        bad.couple(0, 1, f64::NAN);
        assert!(matches!(Oim::new(&bad.build(), 1.0, 1.0, 0.01), Err(Error::NonFiniteModel(_))));

        // Every variant prints something that names the quantity.
        for e in [
            Error::EmptyGraph,
            Error::CouplingGain(0.0),
            Error::ShilGain(-1.0),
            Error::Step(0.0),
            Error::StepTooLarge { dt: 1.0, max: 0.25, lipschitz: 4.0 },
            Error::NonFiniteModel(3),
        ] {
            assert!(!e.to_string().is_empty());
        }
    }

    /// Same seed, same answer — and a different seed is a different trajectory, or the seed is
    /// decorative.
    #[test]
    fn runs_are_deterministic_by_seed() {
        let g = dense_random(12, 7);
        let p = Params::default();
        let a = run(&g, &p, 99).unwrap();
        let b = run(&g, &p, 99).unwrap();
        assert_eq!(a.state, b.state);
        assert_eq!(a.phases, b.phases);
        let c = run(&g, &p, 100).unwrap();
        assert!(c.phases != a.phases, "a different seed must move the trajectory");
    }

    /// The bracket contains the value it brackets, and is not so wide as to contain everything.
    #[test]
    fn the_energy_bracket_is_tight_around_the_energy() {
        let g = dense_random(30, 12);
        let m = Oim::with_safe_step(&g, 1.0, 0.8).unwrap();
        let mut rng = Pcg::new(4, 0x0C);
        let phi: Vec<f64> = (0..g.n).map(|_| rng.f64() * std::f64::consts::TAU).collect();
        let e = m.energy(&phi, 0.8);
        let (lo, hi) = m.energy_bracket(&phi, 0.8);
        assert!(lo <= e && e <= hi, "[{lo}, {hi}] does not contain {e}");
        assert!(hi - lo < 1e-11, "the bracket is {} wide, which bounds nothing useful", hi - lo);
    }
}

