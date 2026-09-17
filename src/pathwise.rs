//! Training *through* the dynamics: the adjoint of an Euler map, and what the integrator costs it.
//!
//! # The gap this closes
//!
//! This crate's learning estimators are all score-function or kernel estimators — REINFORCE in
//! [`crate::ebm`], parameter-shift in [`crate::qaoa`], the contrastive-divergence family. Every one
//! of them differentiates a *log-probability* and needs no derivative of the dynamics at all. The
//! published coupled-oscillator generators are trained the other way: noise enters once, the
//! trajectory that follows is deterministic, and the gradient goes straight back through the
//! integrator. That is a *pathwise* or reparameterisation gradient, and without it this crate can
//! price their architecture ([`crate::precision`]) and reproduce it
//! ([`crate::kuramoto::Pushforward`]) but not train it.
//!
//! [`euler_adjoint`] is that gradient, exactly. For the map
//!
//! ```text
//!   phi^{t+1}_i  =  phi^t_i  +  dt ( omega_i + sum_j K_ij sin(phi^t_j - phi^t_i) )
//! ```
//!
//! the reverse pass carries a cotangent `lambda` backwards through
//! `lambda^t = lambda^{t+1} + dt J^T lambda^{t+1}`, with `J` the drift Jacobian
//! [`crate::kuramoto::Kuramoto::jacobian`] already computes, and accumulates
//!
//! ```text
//!   dL/dK_ij  +=  dt * lambda^{t+1}_i * sin(phi^t_j - phi^t_i),
//!   dL/domega_i  +=  dt * lambda^{t+1}_i.
//! ```
//!
//! No wrapping appears in the tape and none is needed: the drift and the readout both see the
//! phases only through `sin` and `cos`, so reducing modulo `2 pi` changes stored numbers and
//! nothing else.
//!
//! # The finding this module exists to state
//!
//! **A trajectory that agrees does not mean a gradient that agrees.** A trained oscillator model
//! is a `dt` and a step count as much as it is a `K`: what the optimiser sees is the Euler map,
//! and what a chip would run is the differential equation. [`integrator_gap`] measures both halves
//! of that mismatch against a fourth-order reference on the same trajectory, each error relative to
//! the magnitude of the quantity it belongs to, and [`IntegratorGap::gradient_penalty`] is their
//! ratio.
//!
//! Measured over five systems and five step sizes — the table `examples/pathwise` prints — the
//! penalty exceeds one in **22 of 25** cases and reaches **32**. It is largest exactly where it
//! matters: on the strongly coupled and the larger systems, where a step size chosen by watching
//! the trajectory leaves the gradient wrong by an order of magnitude more.
//!
//! The three exceptions are worth as much as the rule and are kept in the tests, because the
//! temptation is to state this as a theorem. They are the smallest, most weakly coupled fixture at
//! its finest steps, where both errors have already fallen below a percent and neither is
//! deciding anything. So: **the effect is not universal and it is not an inequality — it is that
//! the two errors are not proxies for one another, and the case where they diverge is the case a
//! real model is trained in.**
//!
//! The consequence is not that Euler is wrong. It is that `dt` is a trained parameter in disguise:
//! a model fitted through a coarse Euler map has absorbed the integrator's error into its weights,
//! and the silicon it is then loaded onto does not have that error to cancel.

use crate::kuramoto::Kuramoto;
use crate::round::sum_up;

/// Two pi.
const TAU: f64 = core::f64::consts::TAU;

/// The gradient of a scalar loss with respect to a coupled-oscillator system's parameters.
#[derive(Clone, Debug, PartialEq)]
pub struct Grads {
    /// `dL/dK_ij`, row-major, the same shape as the coupling.
    ///
    /// Not symmetrised. A caller training a symmetric model adds this to its own transpose; a
    /// caller training the general system, which is what the published models do, does not. Doing
    /// it here would silently halve one of them.
    pub d_k: Vec<f64>,
    /// `dL/domega_i`.
    pub d_omega: Vec<f64>,
}

/// Run the Euler map and keep every intermediate state.
///
/// The trajectory is returned rather than the endpoint because the adjoint needs all of it: the
/// Jacobian at step `t` is a function of `phi^t`, so a reverse pass without the tape would have to
/// integrate backwards and accumulate a second error.
///
/// # Panics
///
/// If `phi0` is not length `n` or `dt` is not finite.
#[must_use]
pub fn trajectory(sys: &Kuramoto, phi0: &[f64], dt: f64, steps: usize) -> Vec<Vec<f64>> {
    assert_eq!(phi0.len(), sys.n(), "initial phases must have one entry per oscillator");
    assert!(dt.is_finite(), "the step size must be finite, got {dt}");
    let n = sys.n();
    let mut tape = Vec::with_capacity(steps + 1);
    let mut phi = phi0.to_vec();
    tape.push(phi.clone());
    let mut f = vec![0.0; n];
    for _ in 0..steps {
        sys.drift(&phi, &mut f);
        for i in 0..n {
            phi[i] += dt * f[i];
        }
        tape.push(phi.clone());
    }
    tape
}

/// The `2n` readout features `(cos phi_0, sin phi_0, ...)` of a phase vector.
#[must_use]
pub fn features_of(phi: &[f64]) -> Vec<f64> {
    let mut u = Vec::with_capacity(2 * phi.len());
    for t in phi {
        u.push(t.cos());
        u.push(t.sin());
    }
    u
}

/// Turn a cotangent on the `2n` features into one on the `n` terminal phases.
///
/// `dL/dphi_i = -g_{2i} sin(phi_i) + g_{2i+1} cos(phi_i)`, which is the chain rule through the
/// readout the fabric performs.
///
/// # Panics
///
/// If the cotangent is not length `2n`.
#[must_use]
pub fn feature_cotangent(phi: &[f64], g: &[f64]) -> Vec<f64> {
    assert_eq!(g.len(), 2 * phi.len(), "a feature cotangent has two entries per oscillator");
    (0..phi.len()).map(|i| -g[2 * i] * phi[i].sin() + g[2 * i + 1] * phi[i].cos()).collect()
}

/// The exact gradient of a loss on the terminal phases, back through every Euler step.
///
/// `cotangent` is `dL/dphi` at the end of the trajectory; `tape` is [`trajectory`]'s output. The
/// cost is one Jacobian per step, so `O(steps * n^2)` — the same order as the forward map, which
/// is what makes reverse mode the right choice here rather than differentiating forwards through
/// `n^2 + n` parameters.
///
/// # Panics
///
/// If the tape is empty, its states are the wrong length, or the cotangent is not length `n`.
#[must_use]
pub fn euler_adjoint(sys: &Kuramoto, tape: &[Vec<f64>], dt: f64, cotangent: &[f64]) -> Grads {
    let n = sys.n();
    assert!(!tape.is_empty(), "an empty tape carries no trajectory to differentiate");
    assert_eq!(cotangent.len(), n, "the cotangent must have one entry per oscillator");
    let mut lam = cotangent.to_vec();
    let mut d_k = vec![0.0; n * n];
    let mut d_omega = vec![0.0; n];
    // Walk the tape backwards. `tape[t]` is the state the step from t to t+1 was taken FROM, so
    // the Jacobian and the sines both belong to `tape[t]` while the cotangent belongs to t+1.
    for t in (0..tape.len().saturating_sub(1)).rev() {
        let phi = &tape[t];
        assert_eq!(phi.len(), n, "a taped state has the wrong length");
        // parameters first, while `lam` is still lambda^{t+1}
        for i in 0..n {
            d_omega[i] += dt * lam[i];
            for j in 0..n {
                if j == i {
                    continue;
                }
                d_k[i * n + j] += dt * lam[i] * (phi[j] - phi[i]).sin();
            }
        }
        // then the state: lambda^t = lambda^{t+1} + dt * J^T lambda^{t+1}
        let jac = sys.jacobian(phi);
        let mut next = vec![0.0; n];
        for j in 0..n {
            // Written as a loop rather than a closure on purpose: the mutation suite's rows are
            // pipe-separated, so a line carrying a Rust `|` cannot be one of its targets, and this
            // is the line a transposed Jacobian would be caught on.
            let mut terms = Vec::with_capacity(n + 1);
            for i in 0..n {
                terms.push(jac[i * n + j] * lam[i]);
            }
            terms.push(lam[j] / dt);
            next[j] = dt * sum_up(&terms);
        }
        lam = next;
    }
    Grads { d_k, d_omega }
}

/// One classical fourth-order Runge–Kutta step of the same drift.
///
/// The reference the Euler map is measured against. Its local error is `O(dt^5)` against Euler's
/// `O(dt^2)`, so at a step small enough for both it stands in for the differential equation a chip
/// would run.
///
/// # Panics
///
/// If `phi` is not length `n`.
pub fn step_rk4(sys: &Kuramoto, phi: &mut [f64], dt: f64) {
    let n = sys.n();
    assert_eq!(phi.len(), n, "phase vector must have one entry per oscillator");
    let mut k1 = vec![0.0; n];
    let mut k2 = vec![0.0; n];
    let mut k3 = vec![0.0; n];
    let mut k4 = vec![0.0; n];
    let mut tmp = vec![0.0; n];
    sys.drift(phi, &mut k1);
    for i in 0..n {
        tmp[i] = phi[i] + 0.5 * dt * k1[i];
    }
    sys.drift(&tmp, &mut k2);
    for i in 0..n {
        tmp[i] = phi[i] + 0.5 * dt * k2[i];
    }
    sys.drift(&tmp, &mut k3);
    for i in 0..n {
        tmp[i] = phi[i] + dt * k3[i];
    }
    sys.drift(&tmp, &mut k4);
    for i in 0..n {
        phi[i] += dt / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
    }
}

/// Integrate to time `t` with fourth-order steps of size `dt / refine`.
///
/// # Panics
///
/// If `refine` is zero or `phi0` is the wrong length.
#[must_use]
pub fn rk4_to(sys: &Kuramoto, phi0: &[f64], dt: f64, steps: usize, refine: usize) -> Vec<f64> {
    assert!(refine > 0, "a refinement of zero takes no steps");
    let h = dt / refine as f64;
    let mut phi = phi0.to_vec();
    for _ in 0..(steps * refine) {
        step_rk4(sys, &mut phi, h);
    }
    phi
}

/// How far an Euler map is from the differential equation, in the trajectory and in the gradient.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IntegratorGap {
    /// Step size the Euler map used.
    pub dt: f64,
    /// `max_i |phi_euler_i - phi_reference_i|`, reduced onto the circle.
    pub phase_error: f64,
    /// That error relative to the reference trajectory's own displacement.
    pub phase_relative: f64,
    /// `max |dL/dK_euler - dL/dK_reference|`.
    pub grad_error: f64,
    /// That error relative to the reference gradient's largest entry.
    pub grad_relative: f64,
}

impl IntegratorGap {
    /// How many times worse the gradient's relative error is than the trajectory's.
    ///
    /// Above one means the training signal is the more delicate of the two. It usually is —
    /// 22 of the 25 measurements in `examples/pathwise`, by up to 32 — but not always, and the
    /// ratio is between two errors with different denominators, so it is an instrument rather
    /// than a dimensionless law. Infinite when the trajectory agrees exactly.
    #[must_use]
    pub fn gradient_penalty(&self) -> f64 {
        if self.phase_relative <= 0.0 {
            return f64::INFINITY;
        }
        self.grad_relative / self.phase_relative
    }
}

/// Measure both halves of the integrator's error at one step size.
///
/// The reference is the same system integrated by [`rk4_to`] at `refine` times the resolution, and
/// its gradient is the adjoint of *that* trajectory — so the comparison is like for like and the
/// only difference between the two sides is the integrator.
///
/// # Panics
///
/// If `phi0` or `cotangent` is the wrong length, or `refine` is zero.
#[must_use]
pub fn integrator_gap(
    sys: &Kuramoto,
    phi0: &[f64],
    dt: f64,
    steps: usize,
    refine: usize,
    cotangent: &[f64],
) -> IntegratorGap {
    let n = sys.n();
    assert_eq!(cotangent.len(), 2 * n, "a feature cotangent has two entries per oscillator");
    let coarse = trajectory(sys, phi0, dt, steps);
    let fine_end = rk4_to(sys, phi0, dt, steps, refine);
    let coarse_end = coarse.last().expect("a tape always has its initial state");

    let mut worst = 0.0f64;
    let mut travel = 0.0f64;
    for i in 0..n {
        worst = worst.max(circle_distance(coarse_end[i], fine_end[i]));
        travel = travel.max(circle_distance(fine_end[i], phi0[i]));
    }

    let g_coarse = euler_adjoint(sys, &coarse, dt, &feature_cotangent(coarse_end, cotangent));
    let h = dt / refine as f64;
    let fine = trajectory(sys, phi0, h, steps * refine);
    let fine_last = fine.last().expect("a tape always has its initial state").clone();
    let g_fine = euler_adjoint(sys, &fine, h, &feature_cotangent(&fine_last, cotangent));

    let mut gerr = 0.0f64;
    let mut gmax = 0.0f64;
    for (a, b) in g_coarse.d_k.iter().zip(&g_fine.d_k) {
        gerr = gerr.max((a - b).abs());
        gmax = gmax.max(b.abs());
    }
    IntegratorGap {
        dt,
        phase_error: worst,
        phase_relative: if travel > 0.0 { worst / travel } else { 0.0 },
        grad_error: gerr,
        grad_relative: if gmax > 0.0 { gerr / gmax } else { 0.0 },
    }
}

/// The distance between two angles on the circle, in `[0, pi]`.
#[must_use]
pub fn circle_distance(a: f64, b: f64) -> f64 {
    let d = (a - b).rem_euclid(TAU);
    if d > core::f64::consts::PI { TAU - d } else { d }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kuramoto::two_oscillator_locked;

    fn fixture(n: usize, seed: u64) -> Kuramoto {
        let mut rng = crate::rng::Pcg::new(seed, 0x9A7_4EED);
        let mut k = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    k[i * n + j] = 1.4 * (2.0 * rng.f64() - 1.0);
                }
            }
        }
        let omega: Vec<f64> = (0..n).map(|_| 0.6 * (2.0 * rng.f64() - 1.0)).collect();
        Kuramoto::new(omega, k).expect("well formed")
    }

    /// THE FORWARD MAP IS PINNED FIRST, to a closed form this module did not produce.
    ///
    /// A finite-difference check validates the derivative of whatever function is implemented, so
    /// a tape that integrates the wrong dynamics passes its gradient test and fails nothing. Two
    /// oscillators below threshold lock at `asin(d_omega / 2k)`, which
    /// [`crate::kuramoto::two_oscillator_locked`] states in closed form, so the trajectory is
    /// checked against that before any gradient is taken.
    #[test]
    fn the_taped_trajectory_reaches_the_closed_form_before_any_gradient_is_taken() {
        let k = 1.0;
        let d_omega = 1.2;
        let want = two_oscillator_locked(d_omega, k).expect("below threshold");
        let sys = Kuramoto::new(vec![0.0, d_omega], vec![0.0, k, k, 0.0]).expect("well formed");
        let tape = trajectory(&sys, &[0.0, 0.0], 1e-3, 200_000);
        let end = tape.last().expect("a tape always has its initial state");
        let delta = {
            let d = (end[1] - end[0]).rem_euclid(TAU);
            if d > core::f64::consts::PI { d - TAU } else { d }
        };
        assert!((delta - want).abs() < 1e-6, "taped trajectory locked at {delta}, closed form {want}");
        assert_eq!(tape.len(), 200_001, "the tape must carry every state the adjoint needs");
    }

    /// The adjoint against central differences, entry by entry, on a system with no symmetry left
    /// to hide an index error.
    #[test]
    fn the_adjoint_matches_central_differences() {
        let n = 4;
        let sys = fixture(n, 4242);
        let dt = 0.02;
        let steps = 25;
        let phi0: Vec<f64> = (0..n).map(|i| 0.3 * (i as f64 + 1.0)).collect();
        // An asymmetric cotangent: a uniform one is invariant under swaps and would not see a
        // transposed index.
        let g: Vec<f64> = (0..2 * n).map(|i| 0.2 + 0.11 * i as f64).collect();

        let loss = |s: &Kuramoto| -> f64 {
            let tape = trajectory(s, &phi0, dt, steps);
            let u = features_of(tape.last().expect("tape"));
            let terms: Vec<f64> = (0..2 * n).map(|i| g[i] * u[i]).collect();
            sum_up(&terms)
        };

        let tape = trajectory(&sys, &phi0, dt, steps);
        let end = tape.last().expect("tape").clone();
        let grads = euler_adjoint(&sys, &tape, dt, &feature_cotangent(&end, &g));

        let h = 1e-6;
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                let mut up = vec![0.0; n * n];
                let mut down = vec![0.0; n * n];
                for a in 0..n * n {
                    up[a] = sys.coupling(a / n, a % n);
                    down[a] = up[a];
                }
                up[i * n + j] += h;
                down[i * n + j] -= h;
                let fd = (loss(&Kuramoto::new(sys.omega().to_vec(), up).expect("well formed"))
                    - loss(&Kuramoto::new(sys.omega().to_vec(), down).expect("well formed")))
                    / (2.0 * h);
                assert!(
                    (grads.d_k[i * n + j] - fd).abs() < 1e-6,
                    "dL/dK[{i}][{j}]: adjoint {} against finite difference {fd}",
                    grads.d_k[i * n + j]
                );
            }
        }
        for i in 0..n {
            let mut up = sys.omega().to_vec();
            let mut down = up.clone();
            up[i] += h;
            down[i] -= h;
            let k: Vec<f64> = (0..n * n).map(|a| sys.coupling(a / n, a % n)).collect();
            let fd = (loss(&Kuramoto::new(up, k.clone()).expect("well formed"))
                - loss(&Kuramoto::new(down, k).expect("well formed")))
                / (2.0 * h);
            assert!(
                (grads.d_omega[i] - fd).abs() < 1e-6,
                "dL/domega[{i}]: adjoint {} against finite difference {fd}",
                grads.d_omega[i]
            );
        }
    }

    /// A gradient that is not zero and not symmetric: the check above would pass on an all-zero
    /// gradient if the loss happened not to depend on the parameters, and this says it does.
    #[test]
    fn the_gradient_is_neither_zero_nor_symmetric() {
        let n = 4;
        let sys = fixture(n, 77);
        let phi0: Vec<f64> = (0..n).map(|i| 0.5 * i as f64).collect();
        let g: Vec<f64> = (0..2 * n).map(|i| 1.0 + 0.3 * i as f64).collect();
        let tape = trajectory(&sys, &phi0, 0.02, 30);
        let end = tape.last().expect("tape").clone();
        let grads = euler_adjoint(&sys, &tape, 0.02, &feature_cotangent(&end, &g));
        let biggest = grads.d_k.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        assert!(biggest > 1e-3, "the loss does not depend on the coupling at all: {biggest}");
        let mut asym = 0.0f64;
        for i in 0..n {
            for j in 0..n {
                asym = asym.max((grads.d_k[i * n + j] - grads.d_k[j * n + i]).abs());
            }
        }
        assert!(asym > 1e-3, "an asymmetric system gave a symmetric gradient, which is an index bug");
    }

    /// Runge–Kutta converges at fourth order and Euler at first, which is what makes the fine
    /// trajectory a stand-in for the differential equation.
    #[test]
    fn the_reference_integrator_is_fourth_order() {
        let sys = fixture(3, 5);
        let phi0 = [0.2, 1.1, -0.4];
        let truth = rk4_to(&sys, &phi0, 0.4, 5, 4_000);
        let mut previous: Option<f64> = None;
        for refine in [1usize, 2, 4, 8] {
            let got = rk4_to(&sys, &phi0, 0.4, 5, refine);
            let err = (0..3).fold(0.0f64, |m, i| m.max(circle_distance(got[i], truth[i])));
            if let (Some(p), true) = (previous, err > 1e-14) {
                let ratio = p / err;
                assert!(ratio > 8.0, "halving the step should cut a 4th-order error by ~16, got {ratio}");
            }
            previous = Some(err);
        }
    }

    /// Where a real model is trained — strong coupling, many oscillators — the gradient's error
    /// runs several times the trajectory's at every step size.
    ///
    /// The bound is 2, and the measured values on this fixture are 24.5, 4.0, 2.4, 3.8 and 3.1.
    #[test]
    fn the_integrator_costs_a_strongly_coupled_gradient_more_than_its_trajectory() {
        let n = 8;
        let sys = fixture(n, 7);
        let phi0: Vec<f64> = (0..n).map(|i| 0.4 * (i as f64 + 1.0)).collect();
        let g: Vec<f64> = (0..2 * n).map(|i| 0.5 + 0.2 * i as f64).collect();
        for dt in [0.2f64, 0.1, 0.05, 0.025, 0.0125] {
            let gap = integrator_gap(&sys, &phi0, dt, 50, 64, &g);
            assert!(gap.phase_error > 0.0, "Euler at dt = {dt} agreed with RK4 exactly");
            assert!(gap.grad_error > 0.0, "the gradients agreed exactly at dt = {dt}");
            assert!(
                gap.gradient_penalty() > 2.0,
                "at dt = {dt} the gradient's relative error {} was not twice the trajectory's {}",
                gap.grad_relative,
                gap.phase_relative
            );
        }
    }

    /// AND IT IS NOT A LAW. On the smallest, most weakly coupled fixture at a fine step the
    /// penalty drops below one.
    ///
    /// This test exists so that the claim above cannot quietly widen into a theorem. Both errors
    /// here are already under a percent and neither is deciding anything, which is the honest
    /// reading of the exception — but the exception is real and it is pinned.
    #[test]
    fn the_gradient_penalty_is_not_always_above_one() {
        let n = 4;
        let sys = fixture(n, 2026);
        let phi0: Vec<f64> = (0..n).map(|i| 0.4 * (i as f64 + 1.0)).collect();
        let g: Vec<f64> = (0..2 * n).map(|i| 0.5 + 0.2 * i as f64).collect();
        let gap = integrator_gap(&sys, &phi0, 0.0125, 20, 64, &g);
        assert!(
            gap.gradient_penalty() < 1.0,
            "the documented exception has disappeared: penalty {} at dt = 0.0125",
            gap.gradient_penalty()
        );
        assert!(gap.phase_relative < 0.01 && gap.grad_relative < 0.01, "and both should be small");
        // The same system at a coarse step is the other side of it.
        let coarse = integrator_gap(&sys, &phi0, 0.2, 20, 64, &g);
        assert!(coarse.gradient_penalty() > 1.0, "at a coarse step the penalty returns");
    }

    /// Euler's trajectory error falls when the step does, which is what makes `dt` the knob.
    #[test]
    fn halving_the_step_cuts_eulers_error() {
        let n = 4;
        let sys = fixture(n, 2026);
        let phi0: Vec<f64> = (0..n).map(|i| 0.4 * (i as f64 + 1.0)).collect();
        let g: Vec<f64> = (0..2 * n).map(|i| 0.5 + 0.2 * i as f64).collect();
        let coarse = integrator_gap(&sys, &phi0, 0.1, 20, 64, &g);
        let fine = integrator_gap(&sys, &phi0, 0.05, 20, 64, &g);
        assert!(
            coarse.phase_error > 1.5 * fine.phase_error,
            "halving dt should cut Euler's error: {} then {}",
            coarse.phase_error,
            fine.phase_error
        );
        assert!(coarse.grad_error > 1.5 * fine.grad_error, "and the gradient's with it");
    }

    #[test]
    fn the_circle_distance_wraps() {
        assert!((circle_distance(0.1, TAU - 0.1) - 0.2).abs() < 1e-12);
        assert!((circle_distance(0.0, core::f64::consts::PI) - core::f64::consts::PI).abs() < 1e-12);
        assert_eq!(circle_distance(1.3, 1.3), 0.0);
    }
}
