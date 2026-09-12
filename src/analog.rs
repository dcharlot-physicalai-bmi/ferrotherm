//! The Hopfield–Tank analog network, and the reason it stopped being used.
//!
//! Hopfield & Tank, "Neural computation of decisions in optimization problems", *Biological
//! Cybernetics* **52**:141–152 (1985). A combinatorial problem is written as a quadratic penalty
//! energy, the discrete variables are relaxed onto the unit hypercube, and an analog circuit is
//! allowed to slide downhill:
//!
//! ```text
//!   du_i/dt = −u_i/τ + Σ_j T_ij V_j + I_i ,        V_i = g(u_i) = ½(1 + tanh(β u_i))
//! ```
//!
//! `u` is a capacitor voltage, `T` a resistor matrix, `I` an injected current, `g` an amplifier of
//! gain `β`. Hopfield's 1984 companion paper gives the Lyapunov function
//!
//! ```text
//!   E(V) = −½ Σ_ij T_ij V_i V_j − Σ_i I_i V_i + (1/τ) Σ_i ∫₀^{V_i} g⁻¹(v) dv
//! ```
//!
//! whose time derivative is `−Σ_i g′(u_i) (du_i/dt)²` **when `T` is symmetric** — non-positive,
//! because `g′ > 0`. That single inequality is the entire theoretical claim of the method, and
//! [`AnalogNet::lyapunov`] is it. The integral term has a closed form for this `g`,
//!
//! ```text
//!   ∫₀^V g⁻¹(v) dv = (1/2β) [ V ln V + (1−V) ln(1−V) ]
//! ```
//!
//! (a negative entropy: zero at the corners, `−ln2/2β` at the centre), so the high-gain limit
//! `β → ∞` deletes it and leaves a pure quadratic, whose minima over a box with zero diagonal sit
//! at the **corners** — the relaxation becomes the discrete problem again. [`corner_bound`] states
//! how fast: a fixed point is within `exp(−2βτ m)` of a corner, `m` the smallest field magnitude.
//!
//! # Why this module exists next to [`crate::hopfield`]
//!
//! [`crate::hopfield`] is the *statistical mechanics* of the DISCRETE Hopfield model: Hebbian
//! couplings, the Amit–Gutfreund–Sompolinsky replica equations, storage capacity `α_c ≈ 0.138`.
//! Its state is spins and its dynamics is this crate's Gibbs sampler; its question is "how many
//! memories fit". This module is the *analog optimiser*: a deterministic ODE on real-valued
//! neurons with no temperature at all, whose question is "does downhill motion on a relaxed
//! objective land on a feasible answer". They share a surname and nothing else, and nothing here
//! duplicates anything there.
//!
//! # The documented failure, which is the point
//!
//! Wilson & Pawley, "On the stability of the travelling salesman problem algorithm of Hopfield and
//! Tank", *Biological Cybernetics* **58**:63–70 (1988), re-ran Hopfield & Tank's own 10-city
//! example and found that with the published parameters the great majority of random starts do not
//! converge to a valid tour at all — they settle on a corner of the hypercube that is not a
//! permutation matrix. The relaxation is exact only on the feasible set, the penalty terms make
//! infeasibility expensive rather than impossible, and the dynamics is free to trade a constraint
//! against a shorter path. That result is why the continuous-relaxation Ising machines that came
//! after it — [`crate::sbm`], [`crate::continuous`], [`crate::eqprop`] — either keep the state ON
//! the feasible set or read out a discrete state at every step and keep the best one seen.
//!
//! [`Tsp`] builds Hopfield & Tank's own energy and [`valid_fraction`] measures it. MEASURED here,
//! simulated, on one 10-city Euclidean instance (`Tsp::random_euclidean(10, 12345)`), 64 seeded
//! starts each, `β = 50`, `τ = 1`, `dt = 1e-5`, 4000 steps, `C = 200`:
//!
//! | A = B | D | `n′` | valid tours | mean neurons on |
//! |---|---|---|---|---|
//! | 500 | 500 | 15 | **9 / 64** | 10.4 |
//! | 1000 | 500 | 15 | 56 / 64 | — |
//! | 2000 | 500 | 15 | 64 / 64 | 10.0 |
//! | 500 | 0 | 10 | 64 / 64 | 10.0 |
//! | 500 | 500 | 10 | **0 / 64** | 7.8 |
//! | 20000 | 500 | 10 | **0 / 64** | 7.1 |
//!
//! Row 1 is Hopfield & Tank's published parameter set and reproduces Wilson & Pawley: 86% of starts
//! are invalid. Rows 1–3 are the regime the literature names — validity returns when the penalties
//! dominate the distance term.
//!
//! **The last two rows are the surprise, and they explain a number in the 1985 paper.** For a
//! 10-city problem Hopfield & Tank set the activity target `n′` to 15, not 10, and say only that it
//! encourages neurons to turn on. Set it to the obvious 10 and the failure is not double-booking at
//! all — it is UNDER-activation: the mean number of active neurons falls to 7.8, rows end up empty,
//! and no tour exists. Raising the exclusion penalties `A` and `B` forty-fold makes it **worse**
//! (7.1 active, still 0/64), because they are not the terms being outbid. The distance term is
//! beating the ACTIVITY term `C`: switching a tenth neuron on buys `(C/2)(ΔΣV)² = 100` and costs
//! about `D·d̄·2 ≈ 500`. Inflating `n′` past `n` is what keeps `C(ΣV − n′)` pushing at `ΣV = n`, and
//! that is what the 15 is for.
//!
//! A run that only ever succeeds has got the dynamics wrong; so has one that never does.
//!
//! # `unconverged` is not a bug, and the step budget is not arbitrary
//!
//! Every TSP run above reports `unconverged`: `max|du/dt|` never falls below `1e-6`. That is
//! correct and is a property of the circuit, not of the integrator. The activations `V` saturate
//! within a few hundred steps and never move again, while `u` keeps drifting toward `τ f` on the
//! `τ` timescale, which at `dt = 1e-5` takes 10⁵ steps it does not need. The readout is what is
//! being measured, and it is stationary: the valid count is **9/64 at 2000, 4000, 12000 and 40000
//! steps**, identical, which `the_tsp_readout_is_stationary_in_the_step_budget` asserts.

use crate::graph::Graph;
use crate::rng::Pcg;

/// Why an analog network, a set of dynamics parameters or a distance matrix was refused.
///
/// A default is not a fallback: a `T` of the wrong length, a non-finite entry or a zero gain each
/// name what was actually seen, because every one of them silently produces a trajectory that looks
/// like a relaxation and is not one.
#[derive(Clone, Debug, PartialEq)]
pub enum Refused {
    /// `T` is not `n × n`.
    MatrixShape {
        /// Entries supplied.
        got: usize,
        /// Entries required, `n * n`.
        want: usize,
    },
    /// The bias/current vector is not length `n`.
    BiasShape {
        /// Entries supplied.
        got: usize,
        /// Entries required.
        want: usize,
    },
    /// An entry is NaN or infinite. A non-finite `T` poisons every field in one step.
    NotFinite {
        /// Flat index into whichever array carried it.
        index: usize,
        /// The value seen.
        value: f64,
    },
    /// Gain must be finite and strictly positive: `g′ > 0` is what makes `dE/dt ≤ 0`.
    BadGain {
        /// The gain seen.
        gain: f64,
    },
    /// The membrane time constant must be finite and strictly positive.
    BadTau {
        /// The time constant seen.
        tau: f64,
    },
    /// The Euler step must be finite and strictly positive.
    BadStep {
        /// The step seen.
        dt: f64,
    },
    /// A distance matrix that is not square, or not the size the city count implies.
    CityShape {
        /// Entries supplied.
        got: usize,
        /// Entries required, `cities * cities`.
        want: usize,
    },
    /// A distance matrix with a non-zero diagonal: `d(X, X)` is zero by definition, and a non-zero
    /// one silently adds a constant to every tour.
    SelfDistance {
        /// The city whose self-distance was non-zero.
        city: usize,
        /// The value seen.
        value: f64,
    },
}

impl core::fmt::Display for Refused {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Refused::MatrixShape { got, want } => {
                write!(f, "the coupling matrix has {got} entries and this network needs {want}")
            }
            Refused::BiasShape { got, want } => {
                write!(f, "the input-current vector has {got} entries and this network needs {want}")
            }
            Refused::NotFinite { index, value } => {
                write!(f, "entry {index} is {value}, and a non-finite entry poisons every field in one step")
            }
            Refused::BadGain { gain } => write!(
                f,
                "gain {gain} is not finite and positive: the Lyapunov derivative is \
                 -sum g'(u) (du/dt)^2, which is non-positive only because g' > 0"
            ),
            Refused::BadTau { tau } => write!(
                f,
                "time constant {tau} is not finite and positive: the leak -u/tau is what bounds the \
                 trajectory, and the energy's integral term is divided by it"
            ),
            Refused::BadStep { dt } => write!(
                f,
                "step {dt} is not finite and positive: the Lyapunov theorem is about the ODE, and a \
                 non-positive step does not integrate it"
            ),
            Refused::CityShape { got, want } => {
                write!(f, "the distance matrix has {got} entries and {want} are needed for this many cities")
            }
            Refused::SelfDistance { city, value } => write!(
                f,
                "d({city}, {city}) is {value} and must be zero; a non-zero self-distance adds a \
                 constant to every tour and changes nothing else, which makes it invisible"
            ),
        }
    }
}

impl core::error::Error for Refused {}

/// The sigmoid `g(u) = ½(1 + tanh(β u))`, the amplifier of Hopfield & Tank's circuit.
///
/// Range `(0, 1)`, midpoint `g(0) = ½`, and `g′(u) = β/(2 cosh²(βu)) > 0` everywhere — the strict
/// positivity the Lyapunov argument rests on.
#[must_use]
pub fn g(gain: f64, u: f64) -> f64 {
    0.5 * (1.0 + (gain * u).tanh())
}

/// `g′(u) = β (1 − tanh²(βu)) / 2`, strictly positive.
#[must_use]
pub fn g_prime(gain: f64, u: f64) -> f64 {
    let t = (gain * u).tanh();
    0.5 * gain * (1.0 - t * t)
}

/// `g⁻¹(V) = atanh(2V − 1) / β`, infinite at the corners.
#[must_use]
pub fn g_inverse(gain: f64, v: f64) -> f64 {
    (2.0 * v - 1.0).atanh() / gain
}

/// `∫₀^V g⁻¹(v) dv = (1/2β)[V ln V + (1−V) ln(1−V)]`, the term that vanishes at high gain.
///
/// The closed form is a negative entropy. It is zero at both corners and `−ln2/(2β)` at `V = ½`,
/// which is the sense in which high gain "deletes" it: the whole term is bounded by `ln2/(2β)`,
/// so as `β → ∞` the Lyapunov function becomes the bare quadratic and its minima move to the
/// corners of the box.
///
/// Evaluated as written rather than by quadrature, and checked against an independent evaluation
/// in `the_integral_term_matches_independent_quadrature`. Outside `[0, 1]` the integral is not defined
/// and this returns `f64::INFINITY`, matching [`crate::continuous::Potential::HopfieldTanh`]'s
/// treatment of the same hard box.
#[must_use]
pub fn integral_g_inverse(gain: f64, v: f64) -> f64 {
    if !(0.0..=1.0).contains(&v) {
        return f64::INFINITY;
    }
    let term = |x: f64| if x <= 0.0 { 0.0 } else { x * x.ln() };
    (term(v) + term(1.0 - v)) / (2.0 * gain)
}

/// Dynamics parameters: gain, membrane time constant, and the Euler step.
///
/// Separated from the network because the SAME circuit under a different gain is a different
/// answer — that is the whole content of the high-gain limit — and because the step size belongs
/// to the integrator, not to the physics.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Dynamics {
    gain: f64,
    tau: f64,
    dt: f64,
}

impl Dynamics {
    /// Gain `β`, time constant `τ`, Euler step `dt`, all finite and positive.
    ///
    /// # Errors
    ///
    /// [`Refused::BadGain`], [`Refused::BadTau`] or [`Refused::BadStep`] naming the value seen.
    pub fn new(gain: f64, tau: f64, dt: f64) -> Result<Self, Refused> {
        if !(gain.is_finite() && gain > 0.0) {
            return Err(Refused::BadGain { gain });
        }
        if !(tau.is_finite() && tau > 0.0) {
            return Err(Refused::BadTau { tau });
        }
        if !(dt.is_finite() && dt > 0.0) {
            return Err(Refused::BadStep { dt });
        }
        Ok(Dynamics { gain, tau, dt })
    }

    /// Amplifier gain `β`.
    #[must_use]
    pub fn gain(&self) -> f64 {
        self.gain
    }

    /// Membrane time constant `τ`.
    #[must_use]
    pub fn tau(&self) -> f64 {
        self.tau
    }

    /// Euler step `dt`.
    #[must_use]
    pub fn dt(&self) -> f64 {
        self.dt
    }
}

/// A Hopfield–Tank analog network: dense couplings `T` and input currents `I`.
///
/// Dense rather than [`crate::graph::Graph`]'s CSR on purpose. The problems this method was
/// proposed for — the travelling salesman, assignment, graph bisection — produce `T` matrices that
/// are dense by construction (Hopfield & Tank's `−C` "row sum" term couples every neuron to every
/// other), so a sparse representation would store the same thing with an index beside it.
///
/// Symmetry is NOT required, and that is deliberate: an asymmetric `T` is a perfectly legal
/// circuit and is exactly the hypothesis the Lyapunov theorem needs. The network reports its own
/// [`AnalogNet::asymmetry`] so a caller can check rather than assume, and
/// `lyapunov_descends_only_for_symmetric_coupling` is what that accessor is for.
#[derive(Clone, Debug, PartialEq)]
pub struct AnalogNet {
    n: usize,
    t: Vec<f64>,
    i_ext: Vec<f64>,
}

impl AnalogNet {
    /// A network from a row-major `n × n` matrix and `n` input currents.
    ///
    /// # Errors
    ///
    /// [`Refused::MatrixShape`], [`Refused::BiasShape`] or [`Refused::NotFinite`].
    pub fn new(n: usize, t: Vec<f64>, i_ext: Vec<f64>) -> Result<Self, Refused> {
        if t.len() != n * n {
            return Err(Refused::MatrixShape { got: t.len(), want: n * n });
        }
        if i_ext.len() != n {
            return Err(Refused::BiasShape { got: i_ext.len(), want: n });
        }
        for (k, &v) in t.iter().enumerate() {
            if !v.is_finite() {
                return Err(Refused::NotFinite { index: k, value: v });
            }
        }
        for (k, &v) in i_ext.iter().enumerate() {
            if !v.is_finite() {
                return Err(Refused::NotFinite { index: k, value: v });
            }
        }
        Ok(AnalogNet { n, t, i_ext })
    }

    /// Neuron count.
    #[must_use]
    pub fn n(&self) -> usize {
        self.n
    }

    /// `T_ij`.
    ///
    /// # Panics
    ///
    /// If either index is at or past [`AnalogNet::n`].
    #[must_use]
    pub fn t(&self, i: usize, j: usize) -> f64 {
        assert!(i < self.n && j < self.n, "({i},{j}) outside a {}-neuron network", self.n);
        self.t[i * self.n + j]
    }

    /// Input current `I_i`.
    ///
    /// # Panics
    ///
    /// If `i` is at or past [`AnalogNet::n`].
    #[must_use]
    pub fn current(&self, i: usize) -> f64 {
        assert!(i < self.n, "neuron {i} outside a {}-neuron network", self.n);
        self.i_ext[i]
    }

    /// `max_ij |T_ij − T_ji|`: zero exactly when the Lyapunov theorem applies.
    #[must_use]
    pub fn asymmetry(&self) -> f64 {
        let mut worst = 0.0f64;
        for i in 0..self.n {
            for j in (i + 1)..self.n {
                worst = worst.max((self.t[i * self.n + j] - self.t[j * self.n + i]).abs());
            }
        }
        worst
    }

    /// `max_i |T_ii|`: zero is the condition for the high-gain minima to be at corners.
    ///
    /// A negative diagonal is a self-inhibition, `−½T_ii V_i²` becomes a convex `+½|T_ii| V_i²`,
    /// and it pushes the minimum INTO the box. Hopfield & Tank's own TSP network has `T_ii = −C`,
    /// which is one of the reasons its answers are not always corners.
    #[must_use]
    pub fn max_diagonal(&self) -> f64 {
        (0..self.n).map(|i| self.t[i * self.n + i].abs()).fold(0.0f64, f64::max)
    }

    /// The total input to neuron `i`: `Σ_j T_ij V_j + I_i`.
    ///
    /// The diagonal `T_ii V_i` is included, which is what makes this exactly `−∂E/∂V_i + u_i/τ`
    /// for the [`AnalogNet::lyapunov`] written below, diagonal or no diagonal.
    ///
    /// # Panics
    ///
    /// If `i` is at or past [`AnalogNet::n`], or `v` is shorter than that.
    #[must_use]
    pub fn field(&self, v: &[f64], i: usize) -> f64 {
        let row = &self.t[i * self.n..(i + 1) * self.n];
        let mut s = self.i_ext[i];
        for (j, &tij) in row.iter().enumerate() {
            s += tij * v[j];
        }
        s
    }

    /// The activations `V = g(u)`.
    #[must_use]
    pub fn activations(&self, d: &Dynamics, u: &[f64]) -> Vec<f64> {
        u.iter().map(|&x| g(d.gain, x)).collect()
    }

    /// Hopfield's Lyapunov function `E(V) = −½ VᵀTV − IᵀV + (1/τ) Σ ∫₀^{V_i} g⁻¹`.
    ///
    /// Summed over ALL ordered pairs including `i = j`, which is what makes `∂E/∂V_i =
    /// −Σ_j T_ij V_j − I_i + u_i/τ` hold for a symmetric `T` with any diagonal. That partial is
    /// exactly `−du_i/dt`, so `dE/dt = −Σ_i g′(u_i)(du_i/dt)² ≤ 0`.
    ///
    /// # Panics
    ///
    /// If `v` is not length [`AnalogNet::n`].
    #[must_use]
    pub fn lyapunov(&self, d: &Dynamics, v: &[f64]) -> f64 {
        assert_eq!(v.len(), self.n, "state length");
        let mut quad = 0.0;
        for i in 0..self.n {
            let row = &self.t[i * self.n..(i + 1) * self.n];
            let mut s = 0.0;
            for (j, &tij) in row.iter().enumerate() {
                s += tij * v[j];
            }
            quad += v[i] * s;
        }
        let lin: f64 = (0..self.n).map(|i| self.i_ext[i] * v[i]).sum();
        let ent: f64 = v.iter().map(|&x| integral_g_inverse(d.gain, x)).sum();
        -0.5 * quad - lin + ent / d.tau
    }

    /// One synchronous forward-Euler step of `du/dt = −u/τ + TV + I`, returning `max_i |du_i/dt|`.
    ///
    /// Synchronous: every field is read from the PRE-step `u`, which is what a circuit of
    /// capacitors does and what the Lyapunov argument assumes. Updating in place neuron by neuron
    /// is a different dynamical system (Gauss–Seidel rather than Jacobi) and converges to different
    /// fixed points.
    ///
    /// # Panics
    ///
    /// If `u` is not length [`AnalogNet::n`].
    pub fn step(&self, d: &Dynamics, u: &mut [f64]) -> f64 {
        assert_eq!(u.len(), self.n, "state length");
        let v = self.activations(d, u);
        let mut worst = 0.0f64;
        let mut du = vec![0.0f64; self.n];
        for i in 0..self.n {
            let rate = -u[i] / d.tau + self.field(&v, i);
            du[i] = rate;
            worst = worst.max(rate.abs());
        }
        for i in 0..self.n {
            u[i] += d.dt * du[i];
        }
        worst
    }

    /// Integrate until `max_i |du_i/dt|` falls below `tol`, or `max_steps` steps have run.
    ///
    /// # Panics
    ///
    /// If `u` is not length [`AnalogNet::n`].
    #[must_use]
    pub fn relax(&self, d: &Dynamics, u: &mut [f64], max_steps: usize, tol: f64) -> Relaxation {
        let mut steps = 0;
        let mut rate = f64::INFINITY;
        while steps < max_steps {
            rate = self.step(d, u);
            steps += 1;
            if rate < tol {
                break;
            }
        }
        let v = self.activations(d, u);
        Relaxation { energy: self.lyapunov(d, &v), v, steps, rate, converged: rate < tol }
    }
}

/// What one relaxation produced.
#[derive(Clone, Debug)]
pub struct Relaxation {
    /// Final activations, each in `(0, 1)`.
    pub v: Vec<f64>,
    /// Steps actually taken.
    pub steps: usize,
    /// `max_i |du_i/dt|` at the last step — the evidence for or against `converged`.
    pub rate: f64,
    /// Whether `rate` fell below the tolerance before the step cap.
    pub converged: bool,
    /// The Lyapunov function at `v`.
    pub energy: f64,
}

impl Relaxation {
    /// `max_i min(V_i, 1 − V_i)`: how far the worst neuron is from a corner of the hypercube.
    #[must_use]
    pub fn corner_distance(&self) -> f64 {
        self.v.iter().map(|&x| x.min(1.0 - x)).fold(0.0f64, f64::max)
    }

    /// The activations rounded to the nearest corner, as `0`/`1` bits.
    #[must_use]
    pub fn bits(&self) -> Vec<u8> {
        self.v.iter().map(|&x| u8::from(x > 0.5)).collect()
    }

    /// The activations rounded to spins, `V > ½ ↦ +1`.
    #[must_use]
    pub fn spins(&self) -> Vec<i8> {
        self.v.iter().map(|&x| if x > 0.5 { 1i8 } else { -1 }).collect()
    }
}

/// A certified bound on how far a FIXED POINT of these dynamics can be from a hypercube corner.
///
/// At a fixed point `u_i = τ(Σ_j T_ij V_j + I_i) = τ f_i`, so `min(V_i, 1−V_i) = 1/(1 + e^{2β τ
/// |f_i|}) < exp(−2βτ|f_i|)`. Taking `m = min_i |f_i|` gives one number for the whole state:
///
/// ```text
///   max_i min(V_i, 1 − V_i)  <  exp(−2 β τ m)
/// ```
///
/// This is the quantitative form of Hopfield's high-gain statement, and it says what "approach the
/// corners" means: exponentially in `β τ m`, and **not at all** where the field vanishes. A
/// neuron whose total input is zero sits at `V = ½` at any gain, which is why this returns
/// `1.0` — a vacuous bound, honestly — rather than pretending.
///
/// `m` is accumulated through [`crate::round`]: the field is a sum of `n + 1` floats and a bound
/// that used round-to-nearest would be a bound only up to its own rounding error — this crate
/// shipped a "lower bound" above the optimum for exactly that reason. Each field is bracketed
/// `[lo, hi]` by [`crate::round::sum_down`] and [`crate::round::sum_up`], widened by the rounding
/// of the `T_ij V_j` PRODUCTS (which the summation bracket does not see: it brackets the sum of
/// the rounded products, not of the exact ones), and the magnitude taken from the end of that
/// interval nearest zero. `m` is then a certified lower bound on the true smallest field
/// magnitude, which is the direction that keeps `exp(−2βτm)` an upper bound on the distance.
///
/// `2βτm` is itself rounded down before the exponential, and the result stepped up twice for
/// `exp`'s own error, so the number returned is on the safe side at both ends. When nothing keeps
/// a neuron off the centre the answer is the trivial `0.5`, which is what `min(V, 1−V)` can never
/// exceed — a vacuous bound, stated honestly rather than papered over.
#[must_use]
pub fn corner_bound(net: &AnalogNet, d: &Dynamics, v: &[f64]) -> f64 {
    /// The largest value `min(V, 1 − V)` can take, and so the bound when nothing is known.
    const TRIVIAL: f64 = 0.5;
    let mut m = f64::INFINITY;
    let mut terms = vec![0.0f64; net.n() + 1];
    for i in 0..net.n() {
        terms[net.n()] = net.current(i);
        for j in 0..net.n() {
            terms[j] = net.t(i, j) * v[j];
        }
        // Each product is rounded to nearest, so it is within eps/2 of the exact product in
        // relative terms; n + 1 of them contribute at most (eps/2) * sum|terms| to the sum.
        let scale: f64 = crate::round::sum_up(&terms.iter().map(|x| x.abs()).collect::<Vec<_>>());
        let slop = (0.5 * f64::EPSILON * scale).next_up();
        let lo = crate::round::sum_down(&terms) - slop;
        let hi = crate::round::sum_up(&terms) + slop;
        // The interval straddles zero: nothing keeps this neuron away from the centre.
        let mag = if lo > 0.0 {
            lo
        } else if hi < 0.0 {
            -hi
        } else {
            0.0
        };
        m = m.min(mag);
    }
    if !(m > 0.0) {
        return TRIVIAL;
    }
    // Three roundings in the product, taken at four eps for margin, and rounded DOWN so the
    // exponent is never overstated.
    let z = 2.0 * d.gain() * d.tau() * m;
    let z_lo = z - 4.0 * f64::EPSILON * z.abs();
    (-z_lo).exp().next_up().next_up().min(TRIVIAL)
}

/// Hopfield & Tank's relaxation of an Ising ground state, exactly.
///
/// Substituting `s_i = 2V_i − 1` into `E(s) = −Σ_{i<j} J_ij s_i s_j − Σ_i h_i s_i` gives
///
/// ```text
///   T_ij = 4 J_ij  (i ≠ j),   T_ii = 0,   I_i = 2 h_i − 2 Σ_{j≠i} J_ij
/// ```
///
/// and a constant `Σ_i h_i − Σ_{i<j} J_ij`, which is returned alongside. At any CORNER of the
/// hypercube the integral term is exactly zero, so
///
/// ```text
///   E_ising(s) = lyapunov(V) + offset        for every one of the 2ⁿ corners
/// ```
///
/// is an identity, not an approximation. `the_corner_energies_equal_graph_energy_exactly` checks
/// it against [`crate::graph::Graph::energy`] at all `2ⁿ` corners.
///
/// The zero diagonal is not a choice here: the `s²` terms of the substitution are constants on
/// `{±1}`, so the Ising energy has no diagonal to relax, and its absence is what puts the
/// high-gain minima at the corners.
///
/// # Panics
///
/// Never: the shapes are built here, so the only refusal [`AnalogNet::new`] could raise is a
/// non-finite entry, and that would be a non-finite weight the graph was already carrying.
#[must_use]
pub fn from_ising(gr: &Graph) -> (AnalogNet, f64) {
    let n = gr.n;
    let mut t = vec![0.0f64; n * n];
    let mut i_ext = vec![0.0f64; n];
    let mut offset = 0.0;
    for i in 0..n {
        let mut row_sum = 0.0;
        for k in gr.offset[i]..gr.offset[i + 1] {
            let j = gr.nbr[k] as usize;
            t[i * n + j] = 4.0 * gr.w[k];
            row_sum += gr.w[k];
            if j > i {
                offset -= gr.w[k];
            }
        }
        i_ext[i] = 2.0 * gr.h[i] - 2.0 * row_sum;
        offset += gr.h[i];
    }
    // Shapes are correct by construction here, so the only way `new` can refuse is a non-finite
    // weight, which the graph would already have carried.
    let net = AnalogNet::new(n, t, i_ext).expect("from_ising builds its own shapes");
    (net, offset)
}

/// The four penalty weights of Hopfield & Tank's travelling-salesman energy.
///
/// `a` forbids a city appearing twice, `b` forbids a tour slot holding two cities, `c` pulls the
/// total activity to the target count, and `d` is the distance term — the only one that is the
/// actual objective. The first three are penalties, and a penalty makes a constraint expensive
/// rather than impossible.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Penalties {
    /// One city per row.
    pub a: f64,
    /// One city per tour slot.
    pub b: f64,
    /// Total activity target.
    pub c: f64,
    /// Tour length.
    pub d: f64,
}

/// Hopfield & Tank 1985's published values for their 10-city problem: `A = B = D = 500`,
/// `C = 200`.
pub const HOPFIELD_TANK_1985: Penalties = Penalties { a: 500.0, b: 500.0, c: 200.0, d: 500.0 };

/// A symmetric travelling-salesman instance: `cities × cities` distances, zero diagonal.
#[derive(Clone, Debug, PartialEq)]
pub struct Tsp {
    cities: usize,
    d: Vec<f64>,
}

impl Tsp {
    /// From a row-major distance matrix.
    ///
    /// # Errors
    ///
    /// [`Refused::CityShape`] if the matrix is not `cities × cities`, [`Refused::NotFinite`] for a
    /// NaN or infinite distance, [`Refused::SelfDistance`] for a non-zero diagonal.
    pub fn new(cities: usize, d: Vec<f64>) -> Result<Self, Refused> {
        if d.len() != cities * cities {
            return Err(Refused::CityShape { got: d.len(), want: cities * cities });
        }
        for (k, &v) in d.iter().enumerate() {
            if !v.is_finite() {
                return Err(Refused::NotFinite { index: k, value: v });
            }
        }
        for x in 0..cities {
            let v = d[x * cities + x];
            if v != 0.0 {
                return Err(Refused::SelfDistance { city: x, value: v });
            }
        }
        Ok(Tsp { cities, d })
    }

    /// `cities` points uniform in the unit square, with Euclidean distances.
    ///
    /// Hopfield & Tank's own instance construction. Deterministic in `seed`.
    ///
    /// # Panics
    ///
    /// Never: the matrix it builds is square, finite and zero-diagonal by construction.
    #[must_use]
    pub fn random_euclidean(cities: usize, seed: u64) -> Self {
        let mut rng = Pcg::new(seed, 0xA7);
        let pts: Vec<(f64, f64)> = (0..cities).map(|_| (rng.f64(), rng.f64())).collect();
        let mut d = vec![0.0f64; cities * cities];
        for x in 0..cities {
            for y in 0..cities {
                if x != y {
                    let (dx, dy) = (pts[x].0 - pts[y].0, pts[x].1 - pts[y].1);
                    d[x * cities + y] = dx.hypot(dy);
                }
            }
        }
        Tsp::new(cities, d).expect("a Euclidean matrix is square, finite and zero-diagonal")
    }

    /// City count.
    #[must_use]
    pub fn cities(&self) -> usize {
        self.cities
    }

    /// `d(x, y)`.
    ///
    /// # Panics
    ///
    /// If either index is at or past [`Tsp::cities`].
    #[must_use]
    pub fn distance(&self, x: usize, y: usize) -> f64 {
        assert!(x < self.cities && y < self.cities, "city ({x},{y}) of {}", self.cities);
        self.d[x * self.cities + y]
    }

    /// The neuron index of "city `x` occupies tour slot `i`".
    #[must_use]
    pub fn neuron(&self, x: usize, i: usize) -> usize {
        x * self.cities + i
    }

    /// Hopfield & Tank's network, equations (11)–(12) of the 1985 paper:
    ///
    /// ```text
    ///   T_{Xi,Yj} = −A δ_XY (1−δ_ij) − B δ_ij (1−δ_XY) − C − D d_XY (δ_{j,i+1} + δ_{j,i−1})
    ///   I_Xi      = C n_target
    /// ```
    ///
    /// with slot arithmetic modulo the city count, so the tour closes. `n_target` is the activity
    /// the `C` term pulls towards; Hopfield & Tank set it ABOVE the city count (15 for 10 cities)
    /// to bias the network towards turning neurons on, and that choice is the caller's.
    ///
    /// Note `T_{Xi,Xi} = −C`: the `C` term is `(C/2)(Σ V − n)²`, whose expansion has a diagonal,
    /// and Hopfield & Tank keep it. [`AnalogNet::max_diagonal`] reports it.
    ///
    /// The result is symmetric for any symmetric distance matrix — `d_XY = d_YX` and the
    /// `δ_{j,i+1} + δ_{j,i−1}` pair is its own transpose — so the Lyapunov theorem applies to it,
    /// which `the_tsp_network_is_symmetric_so_the_theorem_applies` checks rather than assumes. Two
    /// cities is the one degenerate size: `i+1` and `i−1` are the same slot modulo 2, so the single
    /// edge is counted twice, which is what a two-city closed tour is.
    ///
    /// # Panics
    ///
    /// Never: shapes are built here and the distances were validated at construction.
    #[must_use]
    pub fn network(&self, p: &Penalties, n_target: f64) -> AnalogNet {
        let c = self.cities;
        let n = c * c;
        let mut t = vec![0.0f64; n * n];
        for x in 0..c {
            for i in 0..c {
                let a_idx = self.neuron(x, i);
                for y in 0..c {
                    for j in 0..c {
                        let b_idx = self.neuron(y, j);
                        let mut val = -p.c;
                        if x == y && i != j {
                            val -= p.a;
                        }
                        if i == j && x != y {
                            val -= p.b;
                        }
                        if x != y {
                            let next = (i + 1) % c;
                            let prev = (i + c - 1) % c;
                            if j == next {
                                val -= p.d * self.d[x * c + y];
                            }
                            if j == prev {
                                val -= p.d * self.d[x * c + y];
                            }
                        }
                        t[a_idx * n + b_idx] = val;
                    }
                }
            }
        }
        let i_ext = vec![p.c * n_target; n];
        AnalogNet::new(n, t, i_ext).expect("the TSP network builds its own shapes")
    }

    /// Hopfield & Tank's initial condition: every neuron at the voltage whose activation is
    /// `1/cities`, plus a small uniform jitter that breaks the symmetry.
    ///
    /// Without the jitter every neuron is identical, the dynamics is identical at every site
    /// forever, and the network converges to the uniform state — which is not a tour and is not
    /// what anyone measured. The jitter is what makes a seed mean something.
    #[must_use]
    pub fn initial_u(&self, d: &Dynamics, jitter: f64, seed: u64) -> Vec<f64> {
        let n = self.cities * self.cities;
        let u0 = g_inverse(d.gain(), 1.0 / self.cities as f64);
        let mut rng = Pcg::new(seed, 0x717);
        (0..n).map(|_| u0 + jitter * (2.0 * rng.f64() - 1.0)).collect()
    }

    /// Decode activations into a tour, or say precisely why they are not one.
    ///
    /// The criterion is Wilson & Pawley's: threshold at `½` and demand a PERMUTATION MATRIX —
    /// exactly one active neuron in every city row and in every tour slot. Anything else is an
    /// invalid tour, and the variant returned names the first row or slot that failed.
    ///
    /// # Errors
    ///
    /// [`TourFault`] naming the row or slot whose active count was not one.
    ///
    /// # Panics
    ///
    /// If `v` is not `cities²` long.
    pub fn tour(&self, v: &[f64]) -> Result<Vec<usize>, TourFault> {
        let c = self.cities;
        assert_eq!(v.len(), c * c, "activation length");
        let on = |x: usize, i: usize| v[x * c + i] > 0.5;
        for x in 0..c {
            let count = (0..c).filter(|&i| on(x, i)).count();
            if count != 1 {
                return Err(TourFault::CityCount { city: x, active: count });
            }
        }
        let mut tour = vec![usize::MAX; c];
        for i in 0..c {
            let count = (0..c).filter(|&x| on(x, i)).count();
            if count != 1 {
                return Err(TourFault::SlotCount { slot: i, active: count });
            }
            tour[i] = (0..c).find(|&x| on(x, i)).expect("count is one");
        }
        Ok(tour)
    }

    /// Closed-tour length `Σ_i d(tour[i], tour[i+1 mod c])`.
    ///
    /// # Panics
    ///
    /// If the tour is not `cities` long or names a city out of range.
    #[must_use]
    pub fn length(&self, tour: &[usize]) -> f64 {
        let c = self.cities;
        assert_eq!(tour.len(), c, "tour length");
        (0..c).map(|i| self.distance(tour[i], tour[(i + 1) % c])).sum()
    }
}

/// Why a converged analog state is not a tour.
///
/// Carried as a typed error rather than a bare `None` because WHICH constraint broke is the
/// finding: Wilson & Pawley's point is not that the method sometimes fails, it is that the failed
/// states are corners of the hypercube that violate the permutation constraints the penalties were
/// supposed to enforce.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TourFault {
    /// A city occupies a number of slots other than one.
    CityCount {
        /// The city.
        city: usize,
        /// Neurons active in its row.
        active: usize,
    },
    /// A tour slot holds a number of cities other than one.
    SlotCount {
        /// The slot.
        slot: usize,
        /// Neurons active in its column.
        active: usize,
    },
}

impl core::fmt::Display for TourFault {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            TourFault::CityCount { city, active } => {
                write!(f, "city {city} is in {active} tour slots, and a tour puts it in exactly one")
            }
            TourFault::SlotCount { slot, active } => {
                write!(f, "tour slot {slot} holds {active} cities, and a tour puts exactly one there")
            }
        }
    }
}

impl core::error::Error for TourFault {}

/// What one sweep of Hopfield–Tank runs produced.
#[derive(Clone, Debug, PartialEq)]
pub struct ValidFraction {
    /// Runs attempted.
    pub runs: usize,
    /// Runs whose converged state decoded to a permutation matrix.
    pub valid: usize,
    /// Runs whose `max |du/dt|` never fell below the tolerance. A run can be non-convergent and
    /// still decode, and the two failures are counted separately on purpose.
    pub unconverged: usize,
    /// Mean tour length over the valid runs, or `None` when there were none.
    pub mean_length: Option<f64>,
    /// Mean number of neurons above `½` at readout, over ALL runs.
    ///
    /// Carried because it is what distinguishes the two ways this method fails, and they call for
    /// opposite fixes. Above the city count the network has double-booked and the exclusion
    /// penalties `A`, `B` are too weak; below it the network has left rows empty and the ACTIVITY
    /// term `C`, or the target `n′`, is too weak. Hopfield & Tank's 10-city network at `n′ = 10`
    /// fails the second way, and raising `A` and `B` drives it further down — see the module doc.
    pub mean_active: f64,
}

impl ValidFraction {
    /// Valid runs as a fraction of runs attempted.
    #[must_use]
    pub fn fraction(&self) -> f64 {
        if self.runs == 0 { 0.0 } else { self.valid as f64 / self.runs as f64 }
    }
}

/// Run `runs` seeded Hopfield–Tank relaxations on `tsp` and count how many decode to a valid tour.
///
/// This is the Wilson & Pawley experiment. Seeds are `seed, seed+1, …`, each giving a different
/// jitter on the same initial voltage, which is exactly the "random initial conditions" of the
/// 1988 paper.
#[must_use]
pub fn valid_fraction(
    tsp: &Tsp,
    p: &Penalties,
    d: &Dynamics,
    n_target: f64,
    jitter: f64,
    max_steps: usize,
    tol: f64,
    runs: usize,
    seed: u64,
) -> ValidFraction {
    let net = tsp.network(p, n_target);
    let mut valid = 0;
    let mut unconverged = 0;
    let mut total = 0.0;
    let mut active = 0usize;
    for r in 0..runs {
        let mut u = tsp.initial_u(d, jitter, seed.wrapping_add(r as u64));
        let out = net.relax(d, &mut u, max_steps, tol);
        if !out.converged {
            unconverged += 1;
        }
        active += out.bits().iter().filter(|&&b| b == 1).count();
        if let Ok(t) = tsp.tour(&out.v) {
            valid += 1;
            total += tsp.length(&t);
        }
    }
    let mean_length = (valid > 0).then(|| total / valid as f64);
    let mean_active = if runs == 0 { 0.0 } else { active as f64 / runs as f64 };
    ValidFraction { runs, valid, unconverged, mean_length, mean_active }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exact::Elimination;
    use crate::graph::GraphBuilder;

    /// A small symmetric zero-diagonal network with fields, built from a seed.
    fn random_symmetric(n: usize, seed: u64) -> AnalogNet {
        let mut rng = Pcg::new(seed, 11);
        let mut t = vec![0.0f64; n * n];
        for i in 0..n {
            for j in (i + 1)..n {
                let w = 2.0 * rng.f64() - 1.0;
                t[i * n + j] = w;
                t[j * n + i] = w;
            }
        }
        let i_ext: Vec<f64> = (0..n).map(|_| 0.6 * (2.0 * rng.f64() - 1.0)).collect();
        AnalogNet::new(n, t, i_ext).expect("shapes")
    }

    /// A random Ising graph with couplings and fields.
    fn random_graph(n: usize, seed: u64) -> Graph {
        let mut rng = Pcg::new(seed, 5);
        let mut gb = GraphBuilder::new(n);
        for i in 0..n {
            for j in (i + 1)..n {
                if rng.f64() < 0.6 {
                    gb.couple(i, j, 2.0 * rng.f64() - 1.0);
                }
            }
            gb.bias(i, 0.4 * (2.0 * rng.f64() - 1.0));
        }
        gb.build()
    }

    /// ORACLE: an independent evaluation of `∫₀^V g⁻¹` that never touches the closed form under
    /// test — the textbook antiderivative `∫ ln v dv = v(ln v − 1)` for the singular half, and
    /// Simpson's rule for the smooth half.
    ///
    /// The closed form `(1/2β)[V ln V + (1−V) ln(1−V)]` is derived by hand in the module doc, and a
    /// dropped factor of two there would be invisible everywhere else: the Lyapunov function would
    /// still descend, just to a different value, and every corner identity would still hold because
    /// this term is zero at the corners. Only an independent evaluation of the integral catches it.
    ///
    /// `g⁻¹(v) = ½ln(v/(1−v))/β` splits into a `ln v` piece, whose integral from 0 is the standard
    /// `V(ln V − 1)`, and a `ln(1−v)` piece, which is smooth on `[0, V]` for `V < 1` and is
    /// integrated numerically. Naive quadrature of the whole integrand is not an option: the
    /// integrand diverges at `v = 0`, and the only closed form for the missing sliver IS the one
    /// under test, which is how the first version of this test came to be circular.
    ///
    /// A second, independent check ties the closed form to the ACTUAL integrand `g_inverse`:
    /// Simpson over an interior interval, where nothing is singular, must equal the difference of
    /// the closed form at its endpoints. The interval is asymmetric about `½` on purpose — over
    /// `[0.1, 0.9]` that difference is exactly zero by symmetry, and a test whose expected value is
    /// zero would pass for a closed form scaled by anything at all.
    #[test]
    fn the_integral_term_matches_independent_quadrature() {
        let beta = 1.7;
        for &v in &[0.05, 0.25, 0.5, 0.75, 0.95] {
            let m = 200_000usize; // even, for Simpson
            let h = v / m as f64;
            let smooth = |x: f64| (1.0 - x).ln();
            let mut acc = smooth(0.0) + smooth(v);
            for k in 1..m {
                acc += if k % 2 == 1 { 4.0 } else { 2.0 } * smooth(k as f64 * h);
            }
            let q = acc * h / 3.0;
            let reference = (v * (v.ln() - 1.0) - q) / (2.0 * beta);
            let got = integral_g_inverse(beta, v);
            assert!(
                (got - reference).abs() < 1e-12,
                "V={v}: closed form {got:e} vs independent reference {reference:e}"
            );
        }

        // Interior: Simpson on g_inverse itself over [0.15, 0.8], asymmetric about 1/2.
        let (a, b) = (0.15f64, 0.8f64);
        let m = 200_000usize;
        let h = (b - a) / m as f64;
        let mut acc = g_inverse(beta, a) + g_inverse(beta, b);
        for k in 1..m {
            acc += if k % 2 == 1 { 4.0 } else { 2.0 } * g_inverse(beta, a + k as f64 * h);
        }
        let q = acc * h / 3.0;
        let got = integral_g_inverse(beta, b) - integral_g_inverse(beta, a);
        assert!(q.abs() > 1e-3, "the interior reference must not be a symmetric zero: {q:e}");
        assert!((got - q).abs() < 1e-12, "interior: closed {got:e} vs quadrature {q:e}");

        // Endpoints are exactly zero, which is what makes the corner identity in `from_ising` hold.
        assert_eq!(integral_g_inverse(beta, 0.0), 0.0);
        assert_eq!(integral_g_inverse(beta, 1.0), 0.0);
        assert!(integral_g_inverse(beta, 1.5).is_infinite());
        // g and g_inverse really are inverse, which is what makes the integrand the right one.
        for &v in &[0.05, 0.3, 0.71] {
            assert!((g(beta, g_inverse(beta, v)) - v).abs() < 1e-14);
        }
    }

    /// ORACLE: [`crate::graph::Graph::energy`], at every one of the `2ⁿ` corners.
    ///
    /// The relaxation is only worth running if it relaxes the RIGHT function, and the claim that it
    /// does is an exact identity at the corners rather than an approximation. A factor of 4 or a
    /// sign in `from_ising` would leave the dynamics perfectly well-behaved and descending to the
    /// wrong answer, which is the failure no monotonicity test can see.
    #[test]
    fn the_corner_energies_equal_graph_energy_exactly_vs_graph_energy() {
        for seed in 0..6u64 {
            let gr = random_graph(10, seed);
            let (net, offset) = from_ising(&gr);
            assert_eq!(net.max_diagonal(), 0.0, "the Ising relaxation has no diagonal");
            assert_eq!(net.asymmetry(), 0.0, "J is symmetric, so T must be");
            let d = Dynamics::new(1.0, 1.0, 1e-3).expect("dynamics");
            let mut worst = 0.0f64;
            for mask in 0..(1u32 << gr.n) {
                let v: Vec<f64> = (0..gr.n).map(|i| f64::from(mask >> i & 1)).collect();
                let s: Vec<i8> = v.iter().map(|&x| if x > 0.5 { 1i8 } else { -1 }).collect();
                let got = net.lyapunov(&d, &v) + offset;
                worst = worst.max((got - gr.energy(&s)).abs());
            }
            assert!(worst < 1e-12, "seed {seed}: corner energies off by {worst}");
        }
    }

    /// The theorem, and its hypothesis, in one test: descent under a symmetric `T` and a MEASURED
    /// violation under an asymmetric one.
    ///
    /// `dE/dt = −Σ_i g′(u_i)(du_i/dt)²` needs `∂E/∂V_i = −du_i/dt`, which needs `T = Tᵀ`. Break
    /// the symmetry and the extra term is `Σ_i (ΩV)_i g′(u_i) u̇_i` with `Ω` the antisymmetric
    /// part, which has no sign. The asymmetric half is the load-bearing one: a descent test alone
    /// passes for an implementation that simply symmetrised its input.
    #[test]
    fn lyapunov_descends_only_for_symmetric_coupling() {
        let n = 8;
        let d = Dynamics::new(2.0, 1.0, 2e-3).expect("dynamics");
        let sym = random_symmetric(n, 3);
        let mut rng = Pcg::new(99, 2);
        let mut u: Vec<f64> = (0..n).map(|_| 0.4 * (2.0 * rng.f64() - 1.0)).collect();
        let u_start = u.clone();

        let mut prev = sym.lyapunov(&d, &sym.activations(&d, &u));
        let first = prev;
        for k in 0..20_000 {
            sym.step(&d, &mut u);
            let e = sym.lyapunov(&d, &sym.activations(&d, &u));
            assert!(
                e <= prev + 1e-12,
                "symmetric T rose at step {k}: {prev} -> {e} (by {})",
                e - prev
            );
            prev = e;
        }
        assert!(first - prev > 0.05, "the run must actually descend, not merely not rise");

        // Same symmetric part, plus a strong antisymmetric circulation.
        let mut t = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                t[i * n + j] = sym.t(i, j);
            }
        }
        for i in 0..n {
            let j = (i + 1) % n;
            t[i * n + j] += 3.0;
            t[j * n + i] -= 3.0;
        }
        let asym = AnalogNet::new(n, t, (0..n).map(|i| sym.current(i)).collect()).expect("shapes");
        assert!(asym.asymmetry() > 5.0, "the counterexample must actually be asymmetric");

        let mut u = u_start;
        let mut prev = asym.lyapunov(&d, &asym.activations(&d, &u));
        let mut worst_rise = 0.0f64;
        for _ in 0..20_000 {
            asym.step(&d, &mut u);
            let e = asym.lyapunov(&d, &asym.activations(&d, &u));
            worst_rise = worst_rise.max(e - prev);
            prev = e;
        }
        assert!(
            worst_rise > 1e-6,
            "an asymmetric T must break the descent; worst single-step rise was {worst_rise}, \
             which is rounding and not a violation"
        );
    }

    /// Descent is a claim about the ODE, and forward Euler is not the ODE. This is where the
    /// difference is, in numbers.
    ///
    /// A SPEC-LEVEL QUALIFICATION, measured rather than assumed: "the Lyapunov function is
    /// non-increasing at every step" is true of `du/dt`, not of `u ← u + dt·du/dt`. On this
    /// network (`β = 2`, `τ = 1`) the worst single-step rise over 4000 steps is pure rounding —
    /// `≤ 1.4e-15` — at every `dt` up to **0.8**, and becomes `0.218` at `dt = 1.0`, `10.5` at
    /// `dt = 2.0`. The monotonicity test above runs at `dt = 2e-3`, four hundred times below the
    /// boundary, which is why it can assert `≤ prev + 1e-12` rather than a soft tolerance.
    ///
    /// Both ends are asserted. A test that only showed descent at a small step would pass for an
    /// implementation with no integrator at all — one that returned the fixed point directly.
    #[test]
    fn a_large_euler_step_breaks_descent_on_a_symmetric_network() {
        let n = 8;
        let sym = random_symmetric(n, 3);
        let mut rng = Pcg::new(99, 2);
        let start: Vec<f64> = (0..n).map(|_| 0.4 * (2.0 * rng.f64() - 1.0)).collect();
        let worst_rise = |dt: f64| {
            let d = Dynamics::new(2.0, 1.0, dt).expect("dynamics");
            let mut u = start.clone();
            let mut prev = sym.lyapunov(&d, &sym.activations(&d, &u));
            let mut worst = 0.0f64;
            for _ in 0..4000 {
                sym.step(&d, &mut u);
                let e = sym.lyapunov(&d, &sym.activations(&d, &u));
                worst = worst.max(e - prev);
                prev = e;
            }
            worst
        };
        for &dt in &[0.002, 0.05, 0.2, 0.5, 0.8] {
            let w = worst_rise(dt);
            assert!(w < 1e-12, "dt={dt} must descend to rounding; worst rise {w:e}");
        }
        for &dt in &[1.0, 2.0, 4.0] {
            let w = worst_rise(dt);
            assert!(w > 0.1, "dt={dt} must overshoot and RISE; worst rise was only {w:e}");
        }
    }

    /// ORACLE: the closed form `exp(−2βτm)`, checked at high gain and shown VACUOUS at low gain.
    ///
    /// The bound is a statement about fixed points, so the run is taken to convergence first and
    /// the residual rate is asserted. At `β = 50` the state must be at the corners to within the
    /// bound; at `β = 0.02` the same network's fixed point must be nowhere near one, which is what
    /// makes this a test of the HIGH-GAIN LIMIT rather than of rounding.
    #[test]
    fn high_gain_fixed_points_sit_inside_the_closed_form_corner_bound() {
        let gr = random_graph(12, 21);
        let (net, _) = from_ising(&gr);

        let hot = Dynamics::new(50.0, 1.0, 2e-3).expect("dynamics");
        let mut rng = Pcg::new(7, 1);
        let mut u: Vec<f64> = (0..gr.n).map(|_| 0.02 * (2.0 * rng.f64() - 1.0)).collect();
        let out = net.relax(&hot, &mut u, 200_000, 1e-9);
        assert!(out.converged, "high-gain run did not settle: rate {}", out.rate);
        let bound = corner_bound(&net, &hot, &out.v);
        assert!(bound < 1e-6, "the bound itself must be tight at beta=50, got {bound}");
        assert!(
            out.corner_distance() <= bound,
            "corner distance {} exceeds its own bound {bound}",
            out.corner_distance()
        );

        // The bound is not a promise that high gain is enough. A neuron whose total input is
        // exactly zero sits at V = 1/2 at EVERY gain, and `corner_bound` must say so — return the
        // trivial 1/2 rather than a small number. Without this the test would pass for a function
        // that simply returned exp(-2 beta tau) and ignored the fields.
        let flat = AnalogNet::new(2, vec![0.0; 4], vec![0.0; 2]).expect("shapes");
        let mut u = vec![0.0, 0.0];
        let out = flat.relax(&hot, &mut u, 100, 1e-12);
        assert_eq!(out.v, vec![0.5, 0.5], "a zero field holds V at 1/2 at any gain");
        assert_eq!(corner_bound(&flat, &hot, &out.v), 0.5, "a vanishing field bounds nothing");

        let cold = Dynamics::new(0.02, 1.0, 2e-3).expect("dynamics");
        let mut u: Vec<f64> = (0..gr.n).map(|_| 0.02 * (2.0 * rng.f64() - 1.0)).collect();
        let out = net.relax(&cold, &mut u, 200_000, 1e-9);
        assert!(out.converged, "low-gain run did not settle: rate {}", out.rate);
        assert!(
            out.corner_distance() > 0.4,
            "at beta=0.02 the fixed point must be INTERIOR, not a corner; distance {}",
            out.corner_distance()
        );
    }

    /// ORACLE: [`crate::exact::Elimination::ground_state`], and the asymmetric half is that a
    /// single run must NOT always find it.
    ///
    /// Hopfield–Tank is descent on a relaxed energy with many local minima. Three claims at once,
    /// on one frustrated 14-spin instance, at a gain high enough (`β = 100`) that the settled state
    /// is a corner to `2.3e-6`:
    ///
    /// * no run may report an energy BELOW the exact ground energy — that would be an energy bug,
    ///   and it is the one failure a "did it find the optimum" test cannot see;
    /// * the Lyapunov value at the settled state, plus the [`from_ising`] offset, must equal the
    ///   Ising energy of the rounded spins, which is the relaxation being the right relaxation;
    /// * restarts must find the ground state, and single runs must not always find it. MEASURED:
    ///   11 of 48 starts reach it. A run that always succeeded would mean this instance has no
    ///   local minima, and the test would be certifying nothing.
    #[test]
    fn restarts_reach_the_elimination_ground_state_and_single_runs_do_not() {
        let gr = random_graph(14, 4);
        let exact = Elimination::default().ground_state(&gr).expect("14 spins eliminate");
        let e0 = exact.ground_energy.expect("ground energy");
        let (net, offset) = from_ising(&gr);
        let d = Dynamics::new(100.0, 1.0, 3e-4).expect("dynamics");

        let mut best = f64::INFINITY;
        let mut hits = 0;
        let runs = 48u64;
        for r in 0..runs {
            let mut rng = Pcg::new(1000 + r, 3);
            let mut u: Vec<f64> = (0..gr.n).map(|_| 0.05 * (2.0 * rng.f64() - 1.0)).collect();
            let out = net.relax(&d, &mut u, 200_000, 1e-9);
            assert!(out.converged, "run {r} did not settle: rate {}", out.rate);
            let e = gr.energy(&out.spins());
            assert!(e >= e0 - 1e-9, "run {r} reported {e}, below the exact ground energy {e0}");
            assert!(
                (out.energy + offset - e).abs() < 1e-6,
                "run {r}: lyapunov {} plus offset {offset} is not the Ising energy {e}",
                out.energy
            );
            if e < best {
                best = e;
            }
            if e <= e0 + 1e-9 {
                hits += 1;
            }
        }
        assert!((best - e0).abs() < 1e-9, "restarts must find the ground state: {best} vs {e0}");
        assert!(hits > 0, "no restart reached the ground state at all");
        assert!(
            (hits as u64) < runs,
            "every one of {runs} runs found the ground state, so this instance has no local \
             minima and the test is vacuous"
        );
    }

    /// THE POINT OF THE MODULE, and both halves are asserted.
    ///
    /// Wilson & Pawley 1988: at Hopfield & Tank's published parameters most random starts do not
    /// produce a valid tour. MEASURED here, 64 starts on one 10-city Euclidean instance: **9 valid,
    /// 55 invalid**. The literature's regime — validity returns when the penalty terms dominate the
    /// distance term — is the `A = B` sweep, MEASURED 9 → 56 → 64 of 64 as `A = B` goes
    /// 500 → 1000 → 2000 against `D = 500`.
    ///
    /// An implementation that only ever succeeds has the dynamics wrong: the descent is on a
    /// penalised objective, and a penalty is a price, not a wall. An implementation that never
    /// succeeds has it wrong too — this network CAN produce tours, and the top of the sweep is
    /// where it does.
    #[test]
    fn hopfield_tank_tsp_produces_invalid_tours_at_the_published_parameters() {
        let tsp = Tsp::random_euclidean(10, 12345);
        let d = Dynamics::new(50.0, 1.0, 1e-5).expect("dynamics");
        let runs = 64;
        let sweep: Vec<ValidFraction> = [500.0, 1000.0, 2000.0]
            .iter()
            .map(|&ab| {
                let p = Penalties { a: ab, b: ab, ..HOPFIELD_TANK_1985 };
                valid_fraction(&tsp, &p, &d, 15.0, 0.002, 4000, 1e-6, runs, 900)
            })
            .collect();

        // The published point is the first of the sweep, and it must mostly fail.
        assert_eq!(
            sweep[0],
            valid_fraction(&tsp, &HOPFIELD_TANK_1985, &d, 15.0, 0.002, 4000, 1e-6, runs, 900),
            "the sweep's first point is Hopfield & Tank's published parameter set"
        );
        assert!(
            sweep[0].fraction() < 0.25,
            "Wilson & Pawley: the published parameters must fail on most starts; got {}/{}",
            sweep[0].valid,
            sweep[0].runs
        );
        // ... and the top of the sweep must mostly succeed, or the failure above is not a failure
        // OF THE PARAMETERS, it is a broken network.
        assert!(
            sweep[2].fraction() > 0.8,
            "with the penalties four times the distance term the network must usually succeed; \
             got {}/{}",
            sweep[2].valid,
            sweep[2].runs
        );
        assert!(
            sweep[0].valid < sweep[1].valid && sweep[1].valid < sweep[2].valid,
            "validity must rise with the penalty-to-distance ratio: {} {} {}",
            sweep[0].valid,
            sweep[1].valid,
            sweep[2].valid
        );
        let len = sweep[2].mean_length.expect("valid tours have lengths");
        assert!(len.is_finite() && len > 0.0, "mean valid tour length {len}");
    }

    /// The failure at the obvious activity target is UNDER-activation, and the exclusion penalties
    /// do not fix it — which is what Hopfield & Tank's `n′ = 15` for ten cities is for.
    ///
    /// With `n′ = 10` and no distance term the network is a perfect feasibility solver: MEASURED
    /// 64/64 valid, mean 10.00 neurons on. Switch Hopfield & Tank's own `D = 500` back on and it is
    /// 0/64, mean 7.8 — too FEW neurons on, so rows are left empty. Raising `A` and `B` forty-fold
    /// makes it worse (7.1 on, still 0/64), because `A` and `B` punish double-booking and nothing
    /// here is double-booked. This is the asymmetric test the module is worth writing for: it
    /// asserts that a plausible fix DOES NOT WORK, which no success-only test can express.
    #[test]
    fn the_tsp_failure_at_target_n_is_under_activation_and_bigger_penalties_worsen_it() {
        let tsp = Tsp::random_euclidean(10, 12345);
        let d = Dynamics::new(50.0, 1.0, 1e-5).expect("dynamics");
        let runs = 64;
        let run = |a: f64, dist: f64| {
            let p = Penalties { a, b: a, c: 200.0, d: dist };
            valid_fraction(&tsp, &p, &d, 10.0, 0.002, 4000, 1e-6, runs, 900)
        };

        let feasibility = run(500.0, 0.0);
        assert!(
            feasibility.fraction() > 0.95,
            "with no distance term the network must solve the permutation constraints; got {}/{}",
            feasibility.valid,
            feasibility.runs
        );
        assert!(
            (feasibility.mean_active - 10.0).abs() < 0.1,
            "and must turn on one neuron per city: mean active {}",
            feasibility.mean_active
        );

        let published_d = run(500.0, 500.0);
        assert_eq!(published_d.valid, 0, "Hopfield & Tank's D at n' = 10 must give no valid tour");
        assert!(
            published_d.mean_active < 9.0,
            "the failure must be UNDER-activation, not double-booking: mean active {}",
            published_d.mean_active
        );

        let heavier = run(20000.0, 500.0);
        assert_eq!(
            heavier.valid, 0,
            "raising the exclusion penalties forty-fold must NOT fix an under-activation failure"
        );
        assert!(
            heavier.mean_active < published_d.mean_active,
            "and must make it worse: {} active at A=B=20000 against {} at A=B=500",
            heavier.mean_active,
            published_d.mean_active
        );
    }

    /// The step budget is not doing the work: the readout is stationary long before `du/dt` is.
    ///
    /// Every TSP run reports `unconverged`, because `u` keeps drifting toward `τ f` on the `τ`
    /// timescale while `V` saturated hundreds of steps in. If that mattered, the valid count would
    /// move with the budget. MEASURED: 9/64 at 2000 steps and 9/64 at 12000, and the `unconverged`
    /// count is the full 64 at both — the number being reported is a settled readout of an
    /// unsettled voltage, and this is what says so.
    #[test]
    fn the_tsp_readout_is_stationary_in_the_step_budget() {
        let tsp = Tsp::random_euclidean(10, 12345);
        let d = Dynamics::new(50.0, 1.0, 1e-5).expect("dynamics");
        let short = valid_fraction(&tsp, &HOPFIELD_TANK_1985, &d, 15.0, 0.002, 2000, 1e-6, 64, 900);
        let long = valid_fraction(&tsp, &HOPFIELD_TANK_1985, &d, 15.0, 0.002, 12000, 1e-6, 64, 900);
        assert_eq!(short.valid, long.valid, "the valid count must not depend on the step budget");
        assert_eq!(short.unconverged, 64, "no run settles du/dt inside 2000 steps");
        assert_eq!(long.unconverged, 64, "nor inside 12000, which is the point");
        // The readout is stationary but not frozen: MEASURED, exactly ONE neuron of the 6400 in
        // this sweep (64 runs x 100 neurons) is still moving between step 2000 and step 12000, and
        // it is not one that changes any tour's validity. Asserting bit-equality here would be
        // asserting something the circuit does not do; asserting nothing would miss a readout that
        // drifts.
        let moved = (short.mean_active - long.mean_active).abs() * short.runs as f64;
        assert!(
            moved <= 1.0,
            "at most one neuron of {} may still be moving; {moved} were",
            short.runs * tsp.cities() * tsp.cities()
        );
    }

    /// The TSP network satisfies the Lyapunov theorem's hypothesis, and its diagonal is Hopfield
    /// & Tank's `−C` rather than zero.
    ///
    /// Asserted rather than assumed, because the whole module rests on it: an asymmetric `T` has no
    /// Lyapunov function, so a TSP network that was accidentally asymmetric would be measuring the
    /// wrong failure for the wrong reason.
    #[test]
    fn the_tsp_network_is_symmetric_so_the_theorem_applies() {
        let tsp = Tsp::random_euclidean(6, 77);
        let net = tsp.network(&HOPFIELD_TANK_1985, 9.0);
        assert_eq!(net.n(), 36);
        assert_eq!(net.asymmetry(), 0.0, "a symmetric distance matrix gives a symmetric T");
        assert_eq!(net.max_diagonal(), 200.0, "Hopfield & Tank keep the C term's diagonal");
        assert_eq!(net.current(0), 200.0 * 9.0, "I = C n'");
        // The A term: two slots of the same city repel by A (plus the ubiquitous C).
        assert_eq!(net.t(tsp.neuron(2, 0), tsp.neuron(2, 3)), -700.0);
        // The B term: two cities in the same slot repel by B (plus C).
        assert_eq!(net.t(tsp.neuron(1, 4), tsp.neuron(5, 4)), -700.0);
        // The D term: adjacent slots repel in proportion to distance.
        let want = -200.0 - 500.0 * tsp.distance(1, 3);
        assert!((net.t(tsp.neuron(1, 2), tsp.neuron(3, 3)) - want).abs() < 1e-12);
        // Non-adjacent slots of different cities see only C.
        assert_eq!(net.t(tsp.neuron(1, 0), tsp.neuron(3, 3)), -200.0);
    }

    /// The decoder names WHICH constraint broke, and a permutation matrix decodes to itself.
    #[test]
    fn the_tour_decoder_names_the_constraint_that_broke() {
        let tsp = Tsp::random_euclidean(4, 1);
        let c = 4;
        let mut v = vec![0.0f64; c * c];
        // the identity permutation: city x in slot x
        for x in 0..c {
            v[tsp.neuron(x, x)] = 1.0;
        }
        assert_eq!(tsp.tour(&v), Ok(vec![0, 1, 2, 3]));
        let closed: f64 = (0..c).map(|i| tsp.distance(i, (i + 1) % c)).sum();
        assert!((tsp.length(&[0, 1, 2, 3]) - closed).abs() < 1e-15);

        // city 2 in two slots
        let mut bad = v.clone();
        bad[tsp.neuron(2, 3)] = 1.0;
        assert_eq!(tsp.tour(&bad), Err(TourFault::CityCount { city: 2, active: 2 }));

        // city 1 in no slot, and slot 1 then empty
        let mut bad = v.clone();
        bad[tsp.neuron(1, 1)] = 0.0;
        assert_eq!(tsp.tour(&bad), Err(TourFault::CityCount { city: 1, active: 0 }));

        // Every ROW still holds exactly one neuron, so only the column check can catch this: city 0
        // has moved into slot 1, which already held city 1. Slot 0 is now empty.
        let mut bad = v;
        bad[tsp.neuron(0, 0)] = 0.0;
        bad[tsp.neuron(0, 1)] = 1.0;
        for x in 0..c {
            assert_eq!((0..c).filter(|&i| bad[tsp.neuron(x, i)] > 0.5).count(), 1, "row {x}");
        }
        assert_eq!(tsp.tour(&bad), Err(TourFault::SlotCount { slot: 0, active: 0 }));
    }

    /// A default is not a fallback: every malformed input names what it saw.
    #[test]
    fn malformed_inputs_are_refused_by_name() {
        assert_eq!(
            AnalogNet::new(3, vec![0.0; 8], vec![0.0; 3]),
            Err(Refused::MatrixShape { got: 8, want: 9 })
        );
        assert_eq!(
            AnalogNet::new(3, vec![0.0; 9], vec![0.0; 2]),
            Err(Refused::BiasShape { got: 2, want: 3 })
        );
        let mut t = vec![0.0; 9];
        t[4] = f64::NAN;
        // NaN never compares equal to itself, so the variant is matched and the payload read.
        assert!(matches!(
            AnalogNet::new(3, t, vec![0.0; 3]),
            Err(Refused::NotFinite { index: 4, value }) if value.is_nan()
        ));
        assert_eq!(
            AnalogNet::new(2, vec![0.0; 4], vec![0.0, f64::INFINITY]),
            Err(Refused::NotFinite { index: 1, value: f64::INFINITY })
        );
        assert!(matches!(Dynamics::new(0.0, 1.0, 1e-3), Err(Refused::BadGain { gain }) if gain == 0.0));
        assert!(matches!(Dynamics::new(1.0, -1.0, 1e-3), Err(Refused::BadTau { tau }) if tau == -1.0));
        assert!(matches!(Dynamics::new(1.0, 1.0, 0.0), Err(Refused::BadStep { dt }) if dt == 0.0));
        assert_eq!(Tsp::new(3, vec![0.0; 8]), Err(Refused::CityShape { got: 8, want: 9 }));
        let mut d = vec![1.0; 9];
        d[0] = 0.0;
        d[4] = 0.5;
        assert_eq!(Tsp::new(3, d), Err(Refused::SelfDistance { city: 1, value: 0.5 }));
    }
}
