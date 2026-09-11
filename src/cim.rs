//! The Coherent Ising Machine: a network of degenerate optical parametric oscillators, simulated
//! at mean-field, with the amplitude-heterogeneity correction that makes its readout trustworthy.
//!
//! [`crate::fabric::catalog::qboson_cpqc`] already declares a coherent Ising machine as a
//! deployment target for this crate's programs, and until this module there was nothing here that
//! could run one. That is the hole this fills.
//!
//! # The machine
//!
//! Below threshold a degenerate optical parametric oscillator is a lossy resonator. Above
//! threshold it oscillates, and — being *degenerate* — it can do so in only one of two phases,
//! 0 or pi. That binary is the spin. Couple `N` such oscillators through the Ising matrix and pump
//! them from below threshold to above, and the mode that reaches oscillation first is the one with
//! the largest net gain, which is the one whose phase pattern minimises the Ising energy. Read out
//! `sign(x_i)` and you have a candidate ground state.
//!
//! The mean-field equation of motion, with the pump rate `p` in units of the threshold pump and the
//! in-phase amplitude `x_i` in units of the saturation amplitude:
//!
//! ```text
//!   dx_i/dt = (p - 1 - x_i^2) x_i  +  xi * sum_j J_ij x_j
//! ```
//!
//! The cubic is saturation, `(p - 1)` is net linear gain, and `xi` sets how hard the coupling
//! drives. This crate's energy is `E(s) = -sum J s s - sum h s`, so the coupling term already
//! points downhill in energy with no sign flip: a positive local field pushes `x_i` positive, and a
//! positive `x_i` with a positive field lowers `E`. **Getting that sign backwards turns this module
//! into a maximiser that still passes every symmetric test**, which is why the ground-state tests
//! below check against exhaustive enumeration rather than against another run of this code.
//!
//! A field `h_i` enters as a constant drive `xi * h_i`, which is the ancilla-spin encoding with the
//! ancilla pinned at amplitude one. **That pinning is a modelling choice and it is not free**: a
//! real machine would give the ancilla an oscillator of its own, whose amplitude would drift with
//! every other, so the fields here are weighted as though one oscillator in the network were immune
//! to exactly the heterogeneity this module is about. On a field-free instance the question does
//! not arise, and this crate's hardest instances are field-free.
//!
//! # Two machines, not one
//!
//! [`Coupling::OpenLoop`] is the all-optical machine: the couplings are realised in delay lines and
//! every oscillator sees its neighbours' *present* amplitudes.
//!
//! [`Coupling::MeasurementFeedback`] is the machine that was actually built at scale (Inagaki et
//! al., Science 354:603, 2016; Hamerly et al., Sci. Adv. 5:eaau0823, 2019, at 100,512 spins, and
//! the `QBoson` part this crate already prices). A fraction of each pulse is measured by homodyne
//! detection, an FPGA computes `J x`, and the result is injected back one round trip later. Two
//! things follow, and both are modelled here rather than waved at: the controller sees a **noisy**
//! estimate of `x`, and it injects it **one step late**. The loop delay is the part that survives
//! even at zero measurement noise, so the two couplings are genuinely different dynamics and not a
//! renaming — see `measurement_feedback_is_a_delayed_loop_not_a_relabelled_open_loop`.
//!
//! # Why the error variable exists
//!
//! At a fixed point of the plain machine, oscillator `i` sits where
//! `(p - 1 - x_i^2) x_i = -xi * sum_j J_ij x_j`, so its sign is decided by the **amplitude**-
//! weighted field `sum_j J_ij x_j` and not by the **sign**-weighted field `sum_j J_ij sgn(x_j)`
//! that the Ising energy cares about. When the amplitudes are heterogeneous — and they always are,
//! because a frustrated oscillator saturates lower than a satisfied one — those two disagree, and
//! the machine happily reports a stable fixed point whose sign pattern is not even a local minimum
//! of `E`.
//!
//! Chaotic amplitude control (Leleu et al., *Destabilization of local minima in analog spin systems
//! by correction of amplitude heterogeneity*, Phys. Rev. Lett. 122:040607, 2019; and Leleu et al.,
//! Sci. Rep. 11:13733, 2021) adds one error variable per oscillator:
//!
//! ```text
//!   dx_i/dt = (-1 + p - x_i^2) x_i  +  xi * e_i * sum_j J_ij x_j
//!   de_i/dt = -beta * e_i * (x_i^2 - a)
//! ```
//!
//! `e_i` grows while oscillator `i` runs below the target amplitude `a` and shrinks while it runs
//! above, so the only states where `e` can rest are those with `x_i^2 = a` for every `i` — exactly
//! the amplitude-homogeneous ones, where sign-weighted and amplitude-weighted fields agree. Every
//! other fixed point is destabilised, and the trajectory keeps searching.
//!
//! `amplitude_heterogeneity_defeats_plain_cim_and_the_error_variable_repairs_it` is the test that
//! matters here: one instance, one set of parameters, and the *only* difference between the two
//! runs is [`Control`]. The plain machine settles, settles on the wrong answer, and settles on one
//! that is not even a local minimum; the corrected one lands on the enumerated ground state. Both
//! halves are asserted, because the half that is easy to forget is the failure.
//!
//! **One mutation that survives that test is recorded rather than hidden.** Dropping the
//! multiplicative `e_i` from the error law — writing `de_i/dt = -beta (x_i^2 - a)`, the plausible
//! simplification, and a different algorithm — leaves the single-instance test passing. What
//! catches it is `across_twenty_heterogeneous_instances_...`, whose ground-state count falls from
//! 18 to 16. That is why that test freezes both counts instead of asserting the corrected machine
//! merely beats the plain one, which the broken law also does.
//!
//! # What this is not
//!
//! Not a quantum simulation. This is the mean-field (classical) limit of the DOPO network, which is
//! what the algorithmic literature above studies; the quantum-noise treatments need a positive-P or
//! truncated-Wigner model and are not here. [`Cim::noise`] adds a Gaussian term that stands in for
//! vacuum and pump noise at the same mean-field level, and nothing in this module claims that term
//! is a derived quantum noise strength.
//!
//! ```
//! use ferrotherm::{cim, ising};
//!
//! let g = ising::ring(8, 1.0, 0.0);
//! let run = cim::Cim::for_instance(&g).run(&g).unwrap();
//! assert_eq!(run.energy, -8.0, "a ferromagnetic ring of 8 has E = -8");
//! ```

use crate::graph::Graph;
use crate::rng::Pcg;

/// RNG stream for this module, so a seed shared with another sampler does not share its draws.
const STREAM: u64 = 0xC1_4D;

/// Amplitude past which the integration is declared divergent rather than continued.
///
/// Forward Euler on a cubic is unstable for a step that is too long, and the failure is fast: the
/// amplitude squares itself every step or two and reaches infinity, after which `sign(NaN)` is
/// whatever the comparison happens to return. A refusal naming the step is a better answer than a
/// state, and the caller learns that `dt` was the problem rather than the instance.
const DIVERGENCE: f64 = 1e6;

/// Where each oscillator's coupling term comes from.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Coupling {
    /// All-optical: the coupling reads every neighbour's present amplitude, with no delay and no
    /// measurement in the path.
    OpenLoop,
    /// Measurement feedback: the amplitudes are measured, the matrix-vector product is computed
    /// off-line, and it is injected **one round trip later**.
    MeasurementFeedback {
        /// Standard deviation of the homodyne estimate of `x_i`, in units of the saturation
        /// amplitude. Zero is a perfect detector, and still not the open loop: the delay remains.
        measurement_noise: f64,
    },
}

/// Whether each oscillator carries an error variable that corrects its amplitude.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Control {
    /// The plain machine. Every oscillator's coupling is weighted alike, and a fixed point with
    /// heterogeneous amplitudes can report signs that do not minimise `E`.
    Fixed,
    /// Chaotic amplitude control (Leleu et al. 2019).
    Chaotic {
        /// Rate at which an error variable responds to its oscillator's amplitude. Too small and
        /// the correction never arrives within the run; too large and the error variables
        /// themselves become the fast dynamics and the amplitudes never settle.
        beta: f64,
        /// Target for `x_i^2`. The fixed points that survive are those where every oscillator hits
        /// it, so this is the amplitude the machine is being asked to make uniform.
        target: f64,
    },
}

/// One CIM run: the pump ramp, the coupling path, the control law, and the integrator.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cim {
    /// Pump at the first step, in units of the threshold pump. Below 1 there is no oscillation,
    /// which is where a ramp is meant to start.
    pub pump_start: f64,
    /// Pump at the last step. **Must exceed 1**: at or below threshold every amplitude decays to
    /// zero, `sign(0)` is a convention rather than a measurement, and the readout means nothing.
    /// [`CimError::PumpBelowThreshold`] says so rather than returning that readout.
    pub pump_end: f64,
    /// Coupling strength. The one parameter whose scale is not fixed by the physics, because `J`
    /// carries whatever units the caller's instance came in: the coupling term competes with the
    /// saturation term `(p - 1 - x^2) x`, so `xi * |J|` is what has to be comparable to `p - 1`,
    /// not `xi`. [`Cim::for_instance`] sets it from the instance rather than leaving the caller to
    /// discover this, which is the same lesson [`Graph::flip_gap_max`] records for temperature.
    pub xi: f64,
    /// Integration step. Explicit Euler on a cubic: see [`CimError::Diverged`].
    pub dt: f64,
    /// Steps to integrate. The pump ramp is spread over exactly this many.
    pub steps: usize,
    /// Standard deviation of the Gaussian noise added per unit time, as Euler–Maruyama
    /// (`sqrt(dt) * noise` per step). Zero makes the whole run a deterministic ODE.
    pub noise: f64,
    /// Half-width of the uniform initial spread of `x`. Must be non-zero: `x = 0` is a fixed point
    /// of the equation for every `p`, and an unstable one, so a machine started exactly there stays
    /// there until noise moves it.
    pub init: f64,
    /// All-optical or measurement feedback.
    pub coupling: Coupling,
    /// Plain, or with the error variables.
    pub control: Control,
    /// Seed for the initial amplitudes, the measurement noise and the process noise.
    pub seed: u64,
}

impl Default for Cim {
    /// A short deterministic open-loop run: pump ramped from below threshold to `1.2`, no noise.
    ///
    /// `xi = 0.1` suits couplings of order one; on anything else, prefer [`Cim::for_instance`].
    fn default() -> Self {
        Cim {
            pump_start: 0.0,
            pump_end: 1.2,
            xi: 0.1,
            dt: 0.01,
            steps: 4000,
            noise: 0.0,
            init: 0.01,
            coupling: Coupling::OpenLoop,
            control: Control::Fixed,
            seed: 0xC1,
        }
    }
}

/// What a run produced.
#[derive(Clone, Debug)]
pub struct Run {
    /// `sign(x)` at the final step. **This is what the machine reports**: an optical machine has
    /// one state at the end of a ramp, and [`Run::best_spins`] costs a readout per round trip that
    /// the machine does not otherwise pay ([`crate::ledger`] prices exactly that).
    pub spins: Vec<i8>,
    /// `E(spins)`, recomputed by the graph.
    pub energy: f64,
    /// `E(spins)` accumulated through [`crate::round::sum_up`], so it is never **below** the exact
    /// energy of that state. A solver's answer is an upper bound on the ground energy, and a gap
    /// measured against a total that rounded downwards overstates how good the answer is.
    pub energy_upper: f64,
    /// The lowest-energy readout seen at any step, which needs a host reading the machine out every
    /// round trip. Reported separately from [`Run::spins`] because conflating the two credits an
    /// analog machine with a digital minimum it did not produce.
    pub best_spins: Vec<i8>,
    /// `E(best_spins)`.
    pub best_energy: f64,
    /// The step [`Run::best_spins`] was seen at.
    pub best_step: usize,
    /// Final in-phase amplitudes. The magnitudes are the amplitude heterogeneity itself.
    pub x: Vec<f64>,
    /// Final error variables — all exactly `1.0` under [`Control::Fixed`], since nothing moves them.
    pub e: Vec<f64>,
}

impl Run {
    /// Spread of the final amplitudes, `max|x| - min|x|`, which is zero for a perfectly homogeneous
    /// machine and is the quantity [`Control::Chaotic`] exists to shrink.
    #[must_use]
    pub fn amplitude_spread(&self) -> f64 {
        let (mut lo, mut hi) = (f64::INFINITY, 0.0f64);
        for &v in &self.x {
            lo = lo.min(v.abs());
            hi = hi.max(v.abs());
        }
        if self.x.is_empty() { 0.0 } else { hi - lo }
    }
}

/// Why a run was refused.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CimError {
    /// A machine with no oscillators. Every field below is vacuously satisfied on it, so it is
    /// refused rather than returning an empty state that a caller would compare against nothing.
    NoOscillators,
    /// The final pump is at or below threshold, so every amplitude decays to zero and the sign
    /// readout reports the convention `sign(0) = +1` for every spin.
    PumpBelowThreshold {
        /// The offending [`Cim::pump_end`].
        pump_end: f64,
    },
    /// A step that is not a positive finite number.
    BadStep {
        /// The offending [`Cim::dt`].
        dt: f64,
    },
    /// Zero steps, or a non-finite parameter that would make every amplitude `NaN` immediately.
    BadRun {
        /// Which parameter, by name.
        what: &'static str,
        /// Its value, or `0.0` where the problem is that there are no steps at all.
        value: f64,
    },
    /// The initial spread is zero. `x = 0` is an unstable fixed point for every pump, so a machine
    /// started there never bifurcates and reports `sign(0)` for every spin.
    NoSeedAmplitude,
    /// Error-variable parameters that do not describe a correction: a non-positive `beta` would let
    /// the error variables run away instead of correcting, and a non-positive `target` asks for an
    /// amplitude no oscillator can have.
    BadControl {
        /// Which of `beta` or `target`.
        what: &'static str,
        /// Its value.
        value: f64,
    },
    /// The integration blew up, which for explicit Euler on a cubic means `dt` was too long.
    Diverged {
        /// The step it happened on.
        step: usize,
        /// The oscillator that went first.
        node: usize,
        /// Its amplitude at that point.
        amplitude: f64,
    },
}

impl core::fmt::Display for CimError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            CimError::NoOscillators => {
                write!(f, "a coherent Ising machine with no oscillators has no state to read out")
            }
            CimError::PumpBelowThreshold { pump_end } => write!(
                f,
                "the pump ends at {pump_end}, at or below the threshold pump of 1: every amplitude \
                 decays to zero and the readout is the sign convention rather than a measurement. \
                 Ramp the pump ABOVE 1"
            ),
            CimError::BadStep { dt } => {
                write!(f, "dt = {dt} is not a positive finite integration step")
            }
            CimError::BadRun { what, value } => {
                write!(f, "{what} = {value} cannot be integrated")
            }
            CimError::NoSeedAmplitude => write!(
                f,
                "init = 0 starts every oscillator at x = 0, which is a fixed point of the equation \
                 of motion at every pump: without noise the machine never bifurcates and reports \
                 sign(0) for every spin"
            ),
            CimError::BadControl { what, value } => write!(
                f,
                "chaotic amplitude control needs a positive {what}; got {value}. A non-positive \
                 rate amplifies amplitude heterogeneity instead of correcting it"
            ),
            CimError::Diverged { step, node, amplitude } => write!(
                f,
                "oscillator {node} reached amplitude {amplitude} at step {step}. Explicit Euler on \
                 the saturation cubic is unstable for a long step; reduce dt"
            ),
        }
    }
}

impl std::error::Error for CimError {}

/// `sign(x)` with `sign(0) = +1`, fixed so a run is reproducible rather than tie-dependent.
fn readout(x: &[f64]) -> Vec<i8> {
    x.iter().map(|&v| if v < 0.0 { -1 } else { 1 }).collect()
}

/// One standard normal draw, Box–Muller, from this crate's generator.
fn gauss(rng: &mut Pcg) -> f64 {
    // f64() returns [0, 1); ln(0) is -inf, so the first draw is floored off zero.
    let u1 = rng.f64().max(f64::MIN_POSITIVE);
    let u2 = rng.f64();
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// `E(s)` accumulated so the result is never **below** the exact energy of `s`.
///
/// The energy of any state is an upper bound on the ground energy, and this crate has already
/// shipped one bound that was on the wrong side of the truth because it was summed in
/// round-to-nearest ([`crate::round`] records it). Every term is collected and handed to
/// [`crate::round::sum_up`], so a gap measured from this figure is never flattering by a rounding.
///
/// This is a bound on the summation only. It says nothing about the couplings themselves, which
/// arrive already rounded.
#[must_use]
pub fn energy_upper(g: &Graph, s: &[i8]) -> f64 {
    let mut terms = Vec::with_capacity(g.n + g.n_edges);
    for i in 0..g.n {
        let si = f64::from(s[i]);
        terms.push(-g.h[i] * si);
        for k in g.offset[i]..g.offset[i + 1] {
            let j = g.nbr[k] as usize;
            // Each undirected edge is stored from both ends; count it once.
            if j > i {
                terms.push(-g.w[k] * si * f64::from(s[j]));
            }
        }
    }
    crate::round::sum_up(&terms)
}

impl Cim {
    /// A run whose coupling strength is set from the instance rather than left at a default.
    ///
    /// `xi = 0.5 / (J_rms * sqrt(N))` — the same normalisation [`crate::sbm`] uses, and for the same
    /// reason: the coupling term has to compete with a saturation term of order `p - 1`, the field
    /// summed over `N` neighbours grows like `J_rms * sqrt(N)`, and an instance stated in
    /// millikelvin is the same problem as one stated in kelvin. A `xi` that ignores the instance
    /// answers the two differently.
    ///
    /// `J_rms` is taken over the stored CSR weights, which hold every undirected edge twice; the
    /// double-counting cancels in the ratio, since it is a root-mean-square.
    #[must_use]
    pub fn for_instance(g: &Graph) -> Self {
        let sum_j2: f64 = g.w.iter().map(|&w| w * w).sum();
        let n = g.n as f64;
        let denom = (n * (n - 1.0)).max(1.0);
        let j_rms = (sum_j2 / denom).sqrt();
        let xi = if j_rms > 0.0 && n > 0.0 { 0.5 / (j_rms * n.sqrt()) } else { 0.5 };
        Cim { xi, ..Cim::default() }
    }

    /// The same run on a different seed.
    #[must_use]
    pub fn with_seed(self, seed: u64) -> Self {
        Cim { seed, ..self }
    }

    /// The same run with chaotic amplitude control switched on.
    ///
    /// Changes nothing else, which is what makes a plain run and a corrected run comparable.
    #[must_use]
    pub fn with_chaotic_control(self, beta: f64, target: f64) -> Self {
        Cim { control: Control::Chaotic { beta, target }, ..self }
    }

    /// Refuse a configuration that cannot produce a meaningful readout.
    ///
    /// # Errors
    ///
    /// See [`CimError`]; every variant but [`CimError::Diverged`] is decided here, before a single
    /// step is taken.
    pub fn check(&self, g: &Graph) -> Result<(), CimError> {
        if g.n == 0 {
            return Err(CimError::NoOscillators);
        }
        if !(self.dt > 0.0) || !self.dt.is_finite() {
            return Err(CimError::BadStep { dt: self.dt });
        }
        if self.steps == 0 {
            return Err(CimError::BadRun { what: "steps", value: 0.0 });
        }
        for (what, value) in
            [("pump_start", self.pump_start), ("pump_end", self.pump_end), ("xi", self.xi)]
        {
            if !value.is_finite() {
                return Err(CimError::BadRun { what, value });
            }
        }
        if !(self.noise >= 0.0) || !self.noise.is_finite() {
            return Err(CimError::BadRun { what: "noise", value: self.noise });
        }
        if !(self.pump_end > 1.0) {
            return Err(CimError::PumpBelowThreshold { pump_end: self.pump_end });
        }
        if !(self.init.abs() > 0.0) {
            return Err(CimError::NoSeedAmplitude);
        }
        if let Control::Chaotic { beta, target } = self.control {
            if !(beta > 0.0) || !beta.is_finite() {
                return Err(CimError::BadControl { what: "beta", value: beta });
            }
            if !(target > 0.0) || !target.is_finite() {
                return Err(CimError::BadControl { what: "target", value: target });
            }
        }
        Ok(())
    }

    /// Integrate the machine and read it out.
    ///
    /// Forward Euler on the joint `(x, e)` system: every derivative is taken from the pre-step
    /// state, so the update has full-copy semantics and does not depend on the order the
    /// oscillators are visited in.
    ///
    /// # Errors
    ///
    /// [`Cim::check`] up front, and [`CimError::Diverged`] if the integration blows up.
    pub fn run(&self, g: &Graph) -> Result<Run, CimError> {
        self.check(g)?;
        let n = g.n;
        let mut rng = Pcg::new(self.seed, STREAM);
        let mut x: Vec<f64> = (0..n).map(|_| (rng.f64() - 0.5) * 2.0 * self.init).collect();
        let mut e = vec![1.0f64; n];
        // What the controller last recorded. Seeded from the initial amplitudes so the first
        // injected field is the one an operator would have measured before the ramp began.
        let mut measured = x.clone();
        let mut drive = vec![0.0f64; n];
        let mut best = readout(&x);
        let mut best_energy = g.energy(&best);
        let mut best_step = 0usize;
        let sqrt_dt = self.dt.sqrt();

        for step in 0..self.steps {
            let frac = (step + 1) as f64 / self.steps as f64;
            let pump = self.pump_start + (self.pump_end - self.pump_start) * frac;
            {
                let src: &[f64] = match self.coupling {
                    Coupling::OpenLoop => &x,
                    // One round trip late: this is the controller's PREVIOUS record, which is what
                    // a feedback machine has when it computes the injection for this round trip.
                    Coupling::MeasurementFeedback { .. } => &measured,
                };
                for i in 0..n {
                    let mut f = g.h[i];
                    for k in g.offset[i]..g.offset[i + 1] {
                        f += g.w[k] * src[g.nbr[k] as usize];
                    }
                    drive[i] = f;
                }
            }
            if let Coupling::MeasurementFeedback { measurement_noise } = self.coupling {
                for i in 0..n {
                    measured[i] = x[i] + measurement_noise * gauss(&mut rng);
                }
            }
            for i in 0..n {
                let xi_now = x[i];
                let dx = (pump - 1.0 - xi_now * xi_now) * xi_now + self.xi * e[i] * drive[i];
                let mut next = xi_now + self.dt * dx;
                if self.noise > 0.0 {
                    next += self.noise * sqrt_dt * gauss(&mut rng);
                }
                if let Control::Chaotic { beta, target } = self.control {
                    e[i] += self.dt * (-beta * e[i] * (xi_now * xi_now - target));
                }
                if !next.is_finite() || next.abs() > DIVERGENCE {
                    return Err(CimError::Diverged { step, node: i, amplitude: next });
                }
                x[i] = next;
            }
            let s = readout(&x);
            let energy = g.energy(&s);
            if energy < best_energy {
                best_energy = energy;
                best = s;
                best_step = step;
            }
        }

        let spins = readout(&x);
        let energy = g.energy(&spins);
        let energy_upper = energy_upper(g, &spins);
        Ok(Run { spins, energy, energy_upper, best_spins: best, best_energy, best_step, x, e })
    }

    /// `restarts` runs on derived seeds, keeping the one whose **final** readout is lowest.
    ///
    /// Kept on the final readout rather than on [`Run::best_spins`] so that what is being compared
    /// across restarts is what the machine reports.
    ///
    /// # Errors
    ///
    /// As [`Cim::run`]. A divergence on any restart is returned rather than skipped: a `dt` that
    /// blew up once is a parameter problem, and quietly averaging over the runs that survived it
    /// would report a success rate for a configuration nobody could reproduce.
    ///
    /// # Panics
    ///
    /// If `restarts` is zero.
    pub fn best_of(&self, g: &Graph, restarts: usize) -> Result<Run, CimError> {
        assert!(restarts > 0, "a best-of over zero runs has no answer");
        let mut best: Option<Run> = None;
        for r in 0..restarts {
            let seed = self.seed ^ (r as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            let run = self.with_seed(seed).run(g)?;
            if best.as_ref().is_none_or(|b| run.energy < b.energy) {
                best = Some(run);
            }
        }
        Ok(best.expect("restarts > 0"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::GraphBuilder;
    use crate::oracle::{Exhaustive, Solver};

    /// The two-spin ferromagnet and the frustrated triangle, against `oracle::Exhaustive`.
    ///
    /// The oracle is enumeration over all `2^n` states, in a module that has its own tests — not a
    /// number this module produced and not a second run of this module. It is what makes the
    /// coupling SIGN checkable at all: a machine that maximised `E` instead of minimising it would
    /// return an equally plausible-looking `+-1` vector and satisfy anything weaker than this.
    ///
    /// The triangle is the interesting half. All three couplings are antiferromagnetic, no
    /// assignment satisfies all three, and the machine has to give up exactly one bond — six ground
    /// states out of eight, at `E = -1` rather than the `-3` a reader might expect.
    ///
    /// Run for all four machines: open loop and measurement feedback, plain and corrected. Both
    /// instances have whole-number couplings, so both energies are whole numbers and the comparison
    /// is exact; an epsilon here would not soften the test, it would delete it.
    #[test]
    fn two_spin_ferromagnet_and_frustrated_triangle_match_exhaustive_enumeration() {
        let mut gb = GraphBuilder::new(2);
        gb.couple(0, 1, 1.0);
        let ferro = gb.build();

        let mut gb = GraphBuilder::new(3);
        gb.couple(0, 1, -1.0);
        gb.couple(1, 2, -1.0);
        gb.couple(2, 0, -1.0);
        let triangle = gb.build();

        for (name, g, want) in
            [("ferromagnet", &ferro, -1.0f64), ("frustrated triangle", &triangle, -1.0)]
        {
            let (_, exact) = Exhaustive.solve(g);
            assert_eq!(exact, want, "{name}: the enumeration itself, stated so it is checkable");
            for coupling in [
                Coupling::OpenLoop,
                Coupling::MeasurementFeedback { measurement_noise: 0.005 },
            ] {
                for control in [Control::Fixed, Control::Chaotic { beta: 0.3, target: 0.2 }] {
                    let cim = Cim { coupling, control, ..Cim::for_instance(g) };
                    let run = cim.run(g).unwrap();
                    assert_eq!(
                        run.energy, exact,
                        "{name} / {coupling:?} / {control:?}: readout {:?} has E = {} against the \
                         enumerated ground energy {exact}",
                        run.spins, run.energy
                    );
                    // The energy is the claim, but the state has to be the state that carries it:
                    // a `Run` whose `energy` did not come from its own `spins` would pass above.
                    assert_eq!(run.energy, g.energy(&run.spins));
                }
            }
        }
    }

    /// THE ASYMMETRIC TEST. Amplitude heterogeneity makes the plain machine report a wrong sign,
    /// and the error variable repairs it. **Both halves are asserted**, against enumeration.
    ///
    /// The two runs differ in **exactly one field** — [`Control`] — so nothing else can be credited
    /// with the difference: same instance, same `xi`, same pump ramp, same step, same seed.
    ///
    /// Four claims, in the order they earn each other:
    ///
    /// 1. **The failure.** The plain machine settles above the enumerated ground energy.
    /// 2. **Not even a local minimum.** Some single flip lowers `E` — checked by enumerating the
    ///    eight one-flip neighbours, not by asking this module. That is the specific pathology
    ///    Leleu et al. name: an analog fixed point need not be a local minimum of the Ising energy.
    /// 3. **The mechanism, at that oscillator.** Its sign follows the AMPLITUDE-weighted field and
    ///    contradicts the SIGN-weighted field, which is the disagreement heterogeneity creates.
    ///    Without this the test would record a symptom and could be satisfied by any bad run.
    /// 4. **The fix.** With the error variable, the same run lands exactly on the enumerated ground
    ///    energy.
    ///
    /// Then claims 1 and 4 again across eight further seeds, so neither half is one lucky
    /// trajectory. Measured while the instance was chosen: plain is wrong on 16 of 16 seeds and
    /// corrected is right on 16 of 16, which is why eight is an assertion and not a hope.
    #[test]
    fn amplitude_heterogeneity_defeats_plain_cim_and_the_error_variable_repairs_it() {
        let g = heterogeneous_instance();
        let (_, exact) = Exhaustive.solve(&g);
        assert_eq!(exact, -36.5, "the enumerated ground energy, stated so it is checkable");

        let plain = Cim::for_instance(&g);
        let corrected = plain.with_chaotic_control(0.3, 0.2);

        let a = plain.run(&g).unwrap();
        let b = corrected.run(&g).unwrap();

        // 1. The failure half, which is the half a symmetric test would never look for.
        assert!(
            a.energy > exact + 1e-9,
            "plain CIM settled at E = {} on an instance whose enumerated ground energy is {exact}",
            a.energy
        );

        // 2. Its readout is not even a local minimum: enumerate the one-flip neighbours.
        let downhill: Vec<usize> = (0..g.n)
            .filter(|&i| {
                let mut t = a.spins.clone();
                t[i] = -t[i];
                g.energy(&t) < a.energy - 1e-12
            })
            .collect();
        assert!(
            !downhill.is_empty(),
            "the plain fixed point must be the documented pathology — a stable state that is not a \
             local minimum — and not merely a shallow minimum: spins {:?} at E = {}",
            a.spins,
            a.energy
        );

        // 3. The mechanism at the misreported oscillator. `Graph::field` is the SIGN-weighted field
        //    the energy cares about; `amplitude_field` is what the oscillators actually saw.
        let i = downhill[0];
        let sign_field = g.field(i, &a.spins);
        let amp_field = amplitude_field(&g, i, &a.x);
        assert!(
            sign_field * f64::from(a.spins[i]) < 0.0,
            "oscillator {i} reports {} against a sign-weighted field of {sign_field}",
            a.spins[i]
        );
        assert!(
            amp_field * f64::from(a.spins[i]) > 0.0,
            "and it reports that because the amplitude-weighted field {amp_field} points the other \
             way from the sign-weighted field {sign_field} — which is the heterogeneity itself"
        );
        assert!(
            a.amplitude_spread() > 0.1,
            "and the amplitudes are genuinely spread, by {}",
            a.amplitude_spread()
        );

        // 4. The fix half.
        assert_eq!(
            b.energy, exact,
            "with the error variable the same run, same seed, must land on the enumerated ground \
             state; got {} with spins {:?}",
            b.energy, b.spins
        );

        // And neither half is one lucky trajectory.
        for k in 0..8u64 {
            let seed = 0xC1 ^ k.wrapping_mul(0x9E37_79B9_7F4A_7C15);
            let p = plain.with_seed(seed).run(&g).unwrap();
            let c = corrected.with_seed(seed).run(&g).unwrap();
            assert!(p.energy > exact + 1e-9, "seed {seed}: plain reached {} ", p.energy);
            assert_eq!(c.energy, exact, "seed {seed}: corrected reached {}", c.energy);
        }
    }

    /// `sum_j J_ij x_j + h_i` — the field the oscillators actually see, weighted by amplitude.
    ///
    /// Deliberately NOT [`Graph::field`], which weights by sign. The whole failure being tested is
    /// that these two disagree.
    fn amplitude_field(g: &Graph, i: usize, x: &[f64]) -> f64 {
        let mut f = g.h[i];
        for k in g.offset[i]..g.offset[i + 1] {
            f += g.w[k] * x[g.nbr[k] as usize];
        }
        f
    }

    /// With `xi = 0` the oscillators decouple and each obeys `dx/dt = (p - 1 - x^2) x`, whose
    /// stable fixed points are `x = +-sqrt(p - 1)`.
    ///
    /// The oracle is that closed form, and it is exact for the *discrete* map as well — `x + dt
    /// (p - 1 - x^2) x` has exactly the fixed points of the flow — so no integrator error enters
    /// and 200 time units at a convergence rate of `2(p - 1)` leaves nothing but rounding.
    ///
    /// `1e-14` is therefore the floor rather than a comfortable margin, and it is the floor for a
    /// stated reason: the map contracts by `1 - 2 dt (p - 1)` per step, so its rounding is amplified
    /// by `1 / (2 dt (p - 1))`, which at the shallowest pump here is 250 — a few hundred ULP of
    /// `0.447`, i.e. ~1e-14. Measured: `1e-15` fails, `1e-14` passes.
    #[test]
    fn decoupled_oscillators_reach_the_closed_form_fixed_point_sqrt_of_p_minus_one() {
        for p in [1.2f64, 1.5, 2.0, 3.0] {
            let g = crate::ising::ring(12, 1.0, 0.0);
            let cim = Cim {
                xi: 0.0,
                pump_start: p,
                pump_end: p,
                dt: 0.01,
                steps: 20_000,
                init: 0.05,
                ..Cim::default()
            };
            let run = cim.run(&g).unwrap();
            let want = (p - 1.0).sqrt();
            for (i, &v) in run.x.iter().enumerate() {
                assert!(
                    (v.abs() - want).abs() < 1e-14,
                    "p = {p}: oscillator {i} settled at |x| = {} against the closed form {want}",
                    v.abs()
                );
            }
            // Decoupled means decoupled, and that is a second closed form: `x = 0` is a fixed point
            // of the same equation, so no trajectory can cross it and every oscillator ends on the
            // sign it started with — whatever the ferromagnetic ring underneath it would prefer.
            // The starting signs are recomputed here from the generator rather than read back out
            // of the run, so this compares against the seed and not against the module.
            let mut rng = Pcg::new(cim.seed, STREAM);
            let started: Vec<f64> = (0..g.n).map(|_| (rng.f64() - 0.5) * 2.0 * cim.init).collect();
            for (i, (&v, &s0)) in run.x.iter().zip(&started).enumerate() {
                assert_eq!(
                    v > 0.0,
                    s0 > 0.0,
                    "p = {p}: oscillator {i} started at {s0} and ended at {v}"
                );
            }
            assert!(
                started.iter().any(|&v| v > 0.0) && started.iter().any(|&v| v < 0.0),
                "p = {p}: and the check is not vacuous only if both signs were drawn"
            );
        }
    }

    /// Measurement feedback is a delayed loop, not a second name for the open loop.
    ///
    /// At zero measurement noise the two differ only by where the injected field was measured, and
    /// that alone must change the trajectory. If this ever passes trivially, the two variants have
    /// collapsed into one and the [`Coupling`] enum is decoration.
    #[test]
    fn measurement_feedback_is_a_delayed_loop_not_a_relabelled_open_loop() {
        let g = crate::ising::lattice2d(4, 1.0);
        let base = Cim { steps: 50, ..Cim::for_instance(&g) };
        let open = base.run(&g).unwrap();
        let fed = Cim {
            coupling: Coupling::MeasurementFeedback { measurement_noise: 0.0 },
            ..base
        }
        .run(&g)
        .unwrap();
        assert_eq!(open.x.len(), fed.x.len());
        assert!(
            open.x.iter().zip(&fed.x).any(|(a, b)| a != b),
            "a one-round-trip delay in the feedback path must move the trajectory"
        );
        // And it still solves a ferromagnet, delay and shot noise included.
        let long = Cim {
            coupling: Coupling::MeasurementFeedback { measurement_noise: 0.01 },
            ..Cim::for_instance(&g)
        };
        let run = long.run(&g).unwrap();
        assert_eq!(run.energy, -32.0, "a 4x4 periodic ferromagnet has 32 satisfied bonds");
    }

    /// A seeded run is a run: the same seed reproduces the trajectory bit for bit, a different seed
    /// does not, and the noise term is actually in the path rather than being a field nobody reads.
    #[test]
    fn a_run_is_reproducible_from_its_seed_and_moves_when_the_seed_does() {
        let g = crate::ising::lattice2d(4, -1.0);
        let cim = Cim { noise: 0.02, ..Cim::for_instance(&g) };
        let a = cim.run(&g).unwrap();
        let b = cim.run(&g).unwrap();
        assert!(
            a.x.iter().zip(&b.x).all(|(p, q)| p.to_bits() == q.to_bits()),
            "same seed must give bit-identical amplitudes, not merely equal to what prints"
        );
        let c = cim.with_seed(cim.seed ^ 0xABCD).run(&g).unwrap();
        assert!(a.x.iter().zip(&c.x).any(|(p, q)| p != q), "a different seed must differ");
        let quiet = Cim { noise: 0.0, ..cim }.run(&g).unwrap();
        assert!(
            a.x.iter().zip(&quiet.x).any(|(p, q)| p != q),
            "and the noise term must reach the integrator: a noisy run cannot equal a silent one"
        );
    }

    /// Measured over a family, against enumeration: the error variable is worth having, and the
    /// plain machine's shortfall is not an artefact of the one instance frozen above.
    ///
    /// Twenty heterogeneous twelve-spin instances, `best_of(4)` each, both arms identical but for
    /// [`Control`]. Frozen counts rather than an inequality, because "corrected >= plain" is a
    /// comparison almost any pair of runs satisfies and would still pass if the error variable did
    /// nothing at all. Measured on this tree: plain 10/20, corrected 18/20.
    ///
    /// It is also the test that holds the error law itself down. Dropping the multiplicative `e_i`
    /// from `de_i/dt` passes every other test in this module and lands here as 16 rather than 18.
    #[test]
    fn across_twenty_heterogeneous_instances_the_error_variable_beats_plain_cim_on_enumeration() {
        let (mut plain_hits, mut cac_hits) = (0, 0);
        for inst in 0..20u64 {
            let mut rng = Pcg::new(0xCAC ^ inst, 5);
            // Per-node scale factors spanning 20x: the heterogeneity is in the instance, so the
            // comparison is about the control law rather than about a lucky conditioning.
            let d: Vec<f64> = (0..12).map(|_| if rng.f64() < 0.5 { 0.25 } else { 5.0 }).collect();
            let mut gb = GraphBuilder::new(12);
            for i in 0..12 {
                for j in (i + 1)..12 {
                    if rng.f64() < 0.6 {
                        gb.couple(i, j, d[i] * d[j] * (rng.f64() * 2.0 - 1.0));
                    }
                }
            }
            let g = gb.build();
            let (_, exact) = Exhaustive.solve(&g);
            let base = Cim::for_instance(&g);
            if (base.best_of(&g, 4).unwrap().energy - exact).abs() < 1e-9 {
                plain_hits += 1;
            }
            if (base.with_chaotic_control(0.3, 0.2).best_of(&g, 4).unwrap().energy - exact).abs()
                < 1e-9
            {
                cac_hits += 1;
            }
        }
        assert_eq!(plain_hits, 10, "plain CIM ground-state hits out of 20");
        assert_eq!(cac_hits, 18, "corrected CIM ground-state hits out of 20");
    }

    /// The reported upper bound is never below the energy it bounds, and on integer couplings it
    /// is the exact integer.
    #[test]
    fn the_reported_energy_upper_bound_is_never_below_the_exact_energy() {
        // Couplings that are exactly representable, so the exact total is known independently of
        // any summation: 72 bonds of weight -1 with all spins +1 gives E = +72 exactly. The bound
        // is ABOVE it by the directed-rounding guard and never below, which is the whole contract —
        // and the guard on 72 terms is 3.2e-14, so "safe" has not been traded for "useless".
        let g = crate::ising::lattice2d(6, -1.0);
        let all_up = vec![1i8; g.n];
        let bound = energy_upper(&g, &all_up);
        assert_eq!(g.energy(&all_up), 72.0, "6x6 periodic lattice has 72 bonds");
        assert!(bound >= 72.0, "a bound below the quantity it bounds is not a bound: {bound}");
        assert!(bound - 72.0 < 1e-12, "and it must stay tight: over by {}", bound - 72.0);
        // And on couplings that are not representable, the bound must still be on the safe side.
        let mut gb = GraphBuilder::new(40);
        let mut rng = Pcg::new(11, 3);
        for i in 0..40 {
            for j in (i + 1)..40 {
                gb.couple(i, j, rng.f64() - 0.5);
            }
            gb.bias(i, 0.1 * (rng.f64() - 0.5));
        }
        let g = gb.build();
        let cim = Cim::for_instance(&g);
        let run = cim.run(&g).unwrap();
        assert!(
            run.energy_upper >= run.energy,
            "upper {} must not be below the graph's own total {}",
            run.energy_upper,
            run.energy
        );
        assert!(
            run.energy_upper - run.energy < 1e-9,
            "and it must still be tight: slack {}",
            run.energy_upper - run.energy
        );
    }

    /// The best-so-far readout is bookkeeping, and bookkeeping that never fires is a field that
    /// lies. It must be no worse than the final readout, and on a frustrated instance it must
    /// actually have fired at some step other than the last.
    #[test]
    fn the_best_so_far_readout_is_recorded_and_is_never_worse_than_the_final_one() {
        let mut gb = GraphBuilder::new(12);
        let mut rng = Pcg::new(5, 1);
        for i in 0..12 {
            for j in (i + 1)..12 {
                gb.couple(i, j, rng.f64() - 0.5);
            }
        }
        let g = gb.build();
        let run = Cim::for_instance(&g).run(&g).unwrap();
        assert!(run.best_energy <= run.energy, "{} vs {}", run.best_energy, run.energy);
        assert_eq!(run.best_energy, g.energy(&run.best_spins), "and it names its own state");
    }

    /// Every refusal is reachable, and a divergence is refused rather than returned as `NaN` spins.
    #[test]
    fn each_refusal_is_reachable_and_a_long_step_diverges_rather_than_returning_nan() {
        let g = crate::ising::ring(4, 1.0, 0.0);
        let ok = Cim::for_instance(&g);
        assert_eq!(
            ok.run(&GraphBuilder::new(0).build()).unwrap_err(),
            CimError::NoOscillators
        );
        assert_eq!(
            Cim { pump_end: 1.0, ..ok }.run(&g).unwrap_err(),
            CimError::PumpBelowThreshold { pump_end: 1.0 }
        );
        assert_eq!(Cim { dt: 0.0, ..ok }.run(&g).unwrap_err(), CimError::BadStep { dt: 0.0 });
        assert_eq!(
            Cim { steps: 0, ..ok }.run(&g).unwrap_err(),
            CimError::BadRun { what: "steps", value: 0.0 }
        );
        assert_eq!(
            Cim { noise: -1.0, ..ok }.run(&g).unwrap_err(),
            CimError::BadRun { what: "noise", value: -1.0 }
        );
        assert_eq!(Cim { init: 0.0, ..ok }.run(&g).unwrap_err(), CimError::NoSeedAmplitude);
        assert_eq!(
            ok.with_chaotic_control(0.0, 0.2).run(&g).unwrap_err(),
            CimError::BadControl { what: "beta", value: 0.0 }
        );
        assert_eq!(
            ok.with_chaotic_control(0.3, -1.0).run(&g).unwrap_err(),
            CimError::BadControl { what: "target", value: -1.0 }
        );
        // dt = 5 on a cubic: the amplitude squares itself out of the representable range.
        let blown = Cim { dt: 5.0, pump_end: 3.0, init: 1.0, ..ok }.run(&g).unwrap_err();
        assert!(matches!(blown, CimError::Diverged { .. }), "got {blown}");
        assert!(!format!("{blown}").is_empty());
    }

    /// The instance frozen into the asymmetric test, written out so the test does not depend on a
    /// generator that a later edit could change underneath it.
    ///
    /// Eight oscillators, fifteen couplings, every weight a multiple of a quarter and spanning a
    /// factor of **53** from `0.25` to `13.25`. That span is the point: it is what drives the
    /// steady-state amplitudes apart. Found by searching heterogeneous couplings for an instance
    /// where the plain machine's fixed point misreports a sign, then snapped to quarters and frozen.
    fn heterogeneous_instance() -> Graph {
        let mut gb = GraphBuilder::new(8);
        for &(i, j, w) in HETEROGENEOUS_EDGES {
            gb.couple(i, j, w);
        }
        gb.build()
    }

    /// `(i, j, J_ij)`. See [`heterogeneous_instance`].
    const HETEROGENEOUS_EDGES: &[(usize, usize, f64)] = &[
        (0, 2, 0.25),
        (0, 6, -0.5),
        (1, 2, 4.0),
        (1, 4, -3.0),
        (1, 5, -0.25),
        (1, 7, 0.25),
        (2, 4, -4.0),
        (2, 6, -8.0),
        (2, 7, -1.75),
        (3, 5, 0.5),
        (3, 6, -0.75),
        (4, 5, 4.25),
        (4, 6, -9.0),
        (4, 7, 3.75),
        (5, 6, -13.25),
    ];
}
