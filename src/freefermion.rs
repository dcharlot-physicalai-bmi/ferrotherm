//! The transverse-field Ising chain, solved exactly: Pfeuty's free-fermion spectrum as an oracle
//! for every quantum sampler in this crate, checked against exact diagonalisation.
//!
//! # Why this exists
//!
//! [`crate::sse`] and [`crate::sqa`] sample `H = -J sum sz_i sz_{i+1} - Gamma sum sx_i`, and their
//! tests check against four decoupled spins and against each other. The chain itself has been
//! solved since 1970 (Pfeuty, *Ann. Phys.* 57:79, after Lieb, Schultz and Mattis 1961): a
//! Jordan–Wigner transformation turns it into free fermions with single-particle energies
//!
//! ```text
//!   eps(k) = 2 sqrt(J^2 + Gamma^2 - 2 J Gamma cos k),
//! ```
//!
//! and for `N` even with periodic couplings the ground state lies in the even-fermion sector, whose
//! momenta are the antiperiodic `k_m = (2m + 1) pi / N`, with energy `E_0 = -(1/2) sum_k eps(k)`.
//! At `Gamma = 0` that is `-N J`, at `J = 0` it is `-N Gamma`, and at the critical point
//! `Gamma = J` the energy per site is `-4J / pi` in the thermodynamic limit. The transverse
//! magnetisation is `<sx> = (1/N) sum_k 2 (Gamma - J cos k) / eps(k)`.
//!
//! At finite temperature the chain is still free fermions, but both parity sectors contribute
//! (Katsura, *Phys. Rev.* 127:1508, 1962): the even sector with antiperiodic momenta, the odd
//! sector with periodic ones, each projected onto its parity by a `(1 +- P)/2` that turns the
//! trace into a sum of four products,
//!
//! ```text
//!   2 Z = prod_AP 2cosh(beta eps/2) + prod_AP 2sinh(beta eps/2)
//!       + prod_P  2cosh(beta eps/2) - prod_P  2sinh(beta eps/2),
//! ```
//!
//! where in the periodic sector the `k = 0` mode keeps its sign, `eps_0 = 2 (Gamma - J)`, which is
//! what makes the last term change sign across the transition and reproduce the classical
//! `(2cosh beta J)^N + (2sinh beta J)^N` at `Gamma = 0`. [`log_partition`] and [`thermal_energy`]
//! evaluate this in log space, so `N = 64` at `beta = 100` is a number and not an overflow.
//!
//! Every closed form here is checked against a dense Hamiltonian on 4, 6 and 8 spins -- its ground
//! state by a sector-preserving power iteration, its whole spectrum by Jacobi -- an oracle for the
//! oracle -- so a quantum sampler scored against [`ground_energy`] or [`thermal_energy`] is scored
//! against the exact answer for its size and temperature, not a limit.

use core::f64::consts::PI;

/// The single-particle energy at momentum `k`: `2 sqrt(J^2 + Gamma^2 - 2 J Gamma cos k)`.
#[must_use]
pub fn epsilon(j: f64, gamma: f64, k: f64) -> f64 {
    2.0 * (j * j + gamma * gamma - 2.0 * j * gamma * k.cos()).sqrt()
}

/// The antiperiodic momenta of the even-fermion sector on `n` sites, `(2m + 1) pi / n`.
#[must_use]
pub fn momenta(n: usize) -> Vec<f64> {
    (0..n)
        .map(|m| PI * (2.0 * m as f64 + 1.0) / n as f64)
        .collect()
}

/// The exact ground energy of the periodic chain on `n` sites (`n` even): `-(1/2) sum_k eps(k)`
/// over the antiperiodic momenta.
///
/// # Panics
///
/// If `n` is odd or zero: the even-sector momenta are the ground state's only for even `n`.
#[must_use]
pub fn ground_energy(n: usize, j: f64, gamma: f64) -> f64 {
    assert!(
        n >= 2 && n.is_multiple_of(2),
        "the periodic chain's closed form needs an even number of sites, not {n}"
    );
    -0.5 * momenta(n)
        .iter()
        .map(|&k| epsilon(j, gamma, k))
        .sum::<f64>()
}

/// The ground energy per site in the thermodynamic limit, `-(1/(4 pi)) integral_{-pi}^{pi} eps(k) dk`,
/// by Simpson's rule on `1e5` panels: `-4J / pi` at the critical point `Gamma = J`.
#[must_use]
pub fn ground_energy_density(j: f64, gamma: f64) -> f64 {
    let panels = 100_000usize;
    let h = 2.0 * PI / panels as f64;
    let f = |i: usize| epsilon(j, gamma, -PI + i as f64 * h);
    let mut s = f(0) + f(panels);
    for i in 1..panels {
        s += if i % 2 == 1 { 4.0 } else { 2.0 } * f(i);
    }
    -(s * h / 3.0) / (4.0 * PI)
}

/// The excitation gap in the thermodynamic limit, `2 |J - Gamma|`, closing at the critical point.
#[must_use]
pub fn gap_infinite(j: f64, gamma: f64) -> f64 {
    2.0 * (j - gamma).abs()
}

/// The transverse magnetisation `<sx>` of the ground state on `n` sites (`n` even):
/// `(1/n) sum_k 2 (Gamma - J cos k) / eps(k)`. Exactly 1 at `J = 0`, 0 at `Gamma = 0`.
///
/// # Panics
///
/// As [`ground_energy`].
#[must_use]
pub fn transverse_magnetisation(n: usize, j: f64, gamma: f64) -> f64 {
    assert!(
        n >= 2 && n.is_multiple_of(2),
        "the periodic chain's closed form needs an even number of sites, not {n}"
    );
    if j == 0.0 && gamma == 0.0 {
        return 0.0;
    }
    momenta(n)
        .iter()
        .map(|&k| 2.0 * (gamma - j * k.cos()) / epsilon(j, gamma, k))
        .sum::<f64>()
        / n as f64
}

/// One parity sector's contribution to the trace: the sign of its product of `2sinh`, and the
/// log-magnitude and `beta`-derivative sums of both products.
#[derive(Clone, Copy, Debug)]
struct Sector {
    /// `sum_k ln 2cosh(beta eps_k / 2)`.
    log_cosh: f64,
    /// `sum_k (eps_k / 2) tanh(beta eps_k / 2)`: minus the derivative of `log_cosh` in `beta`.
    d_cosh: f64,
    /// `sum_k ln |2sinh(beta eps_k / 2)|`, `-inf` when a mode sits at zero energy.
    log_sinh: f64,
    /// `sum_k (eps_k / 2) coth(beta eps_k / 2)`, the zero-energy mode excluded.
    d_sinh: f64,
    /// The sign of `prod_k 2sinh(beta eps_k / 2)`: one flip per negative mode, zero if any vanishes.
    sign: f64,
}

/// `ln 2cosh x` and `ln |2sinh x|` without overflow: `|x| + ln(1 +- e^(-2|x|))`.
fn sector(eps: &[f64], beta: f64) -> Sector {
    let mut out = Sector {
        log_cosh: 0.0,
        d_cosh: 0.0,
        log_sinh: 0.0,
        d_sinh: 0.0,
        sign: 1.0,
    };
    for &e in eps {
        let x = 0.5 * beta * e;
        let a = x.abs();
        out.log_cosh += a + (-2.0 * a).exp().ln_1p();
        out.d_cosh += 0.5 * e * x.tanh();
        if e < 0.0 {
            out.sign = -out.sign;
        }
        if e == 0.0 {
            out.sign = 0.0;
            out.log_sinh = f64::NEG_INFINITY;
        } else {
            out.log_sinh += a + (-(-2.0 * a).exp_m1()).ln();
            out.d_sinh += 0.5 * e / x.tanh();
        }
    }
    out
}

/// The single-particle energies of both sectors: antiperiodic momenta for the even sector, and
/// periodic ones for the odd sector with the `k = 0` mode signed, `eps_0 = 2 (Gamma - J)`.
fn spectra(n: usize, j: f64, gamma: f64) -> (Vec<f64>, Vec<f64>) {
    let even = momenta(n).iter().map(|&k| epsilon(j, gamma, k)).collect();
    let odd = (0..n)
        .map(|m| {
            if m == 0 {
                2.0 * (gamma - j)
            } else {
                epsilon(j, gamma, 2.0 * PI * m as f64 / n as f64)
            }
        })
        .collect();
    (even, odd)
}

/// The four signed terms of `2Z` as `(sign, log-magnitude, -d/d beta of the log-magnitude)`.
fn terms(n: usize, j: f64, gamma: f64, beta: f64) -> [(f64, f64, f64); 4] {
    let (even, odd) = spectra(n, j, gamma);
    let a = sector(&even, beta);
    let p = sector(&odd, beta);
    let mut odd_sign = -1.0;
    odd_sign *= p.sign;
    [
        (1.0, a.log_cosh, a.d_cosh),
        (a.sign, a.log_sinh, a.d_sinh),
        (1.0, p.log_cosh, p.d_cosh),
        (odd_sign, p.log_sinh, p.d_sinh),
    ]
}

/// `ln Z` of the periodic chain on `n` sites (`n` even) at inverse temperature `beta`, both parity
/// sectors included, evaluated in log space.
///
/// # Panics
///
/// As [`ground_energy`], and if `beta` is not positive.
#[must_use]
pub fn log_partition(n: usize, j: f64, gamma: f64, beta: f64) -> f64 {
    assert!(
        n >= 2 && n.is_multiple_of(2),
        "the periodic chain's closed form needs an even number of sites, not {n}"
    );
    assert!(
        beta > 0.0,
        "a partition function needs a positive inverse temperature, not {beta}"
    );
    let t = terms(n, j, gamma, beta);
    let top = t.iter().map(|x| x.1).fold(f64::NEG_INFINITY, f64::max);
    let z2: f64 = t.iter().map(|&(s, l, _)| s * (l - top).exp()).sum();
    top + z2.ln() - core::f64::consts::LN_2
}

/// `<H> = -d ln Z / d beta` of the periodic chain on `n` sites (`n` even) at inverse temperature
/// `beta`: the exact thermal energy, which [`crate::sse`] estimates with an error bar.
///
/// # Panics
///
/// As [`log_partition`].
#[must_use]
pub fn thermal_energy(n: usize, j: f64, gamma: f64, beta: f64) -> f64 {
    assert!(
        n >= 2 && n.is_multiple_of(2),
        "the periodic chain's closed form needs an even number of sites, not {n}"
    );
    assert!(
        beta > 0.0,
        "a thermal energy needs a positive inverse temperature, not {beta}"
    );
    let t = terms(n, j, gamma, beta);
    let top = t.iter().map(|x| x.1).fold(f64::NEG_INFINITY, f64::max);
    let z2: f64 = t.iter().map(|&(s, l, _)| s * (l - top).exp()).sum();
    let dz2: f64 = t.iter().map(|&(s, l, d)| s * (l - top).exp() * d).sum();
    -dz2 / z2
}

/// The free energy `-ln Z / beta` of the periodic chain on `n` sites (`n` even).
///
/// # Panics
///
/// As [`log_partition`].
#[must_use]
pub fn free_energy(n: usize, j: f64, gamma: f64, beta: f64) -> f64 {
    -log_partition(n, j, gamma, beta) / beta
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The dense Hamiltonian on `n` spins, `sz` diagonal in the computational basis, `sx` a flip.
    fn dense(n: usize, j: f64, gamma: f64) -> Vec<f64> {
        let dim = 1usize << n;
        let mut h = vec![0.0f64; dim * dim];
        for x in 0..dim {
            let s = |i: usize| if (x >> (i % n)) & 1 == 1 { 1.0 } else { -1.0 };
            let mut diag = 0.0;
            for i in 0..n {
                diag -= j * s(i) * s(i + 1);
            }
            h[x * dim + x] = diag;
            for i in 0..n {
                let y = x ^ (1 << i);
                h[x * dim + y] -= gamma;
            }
        }
        h
    }

    /// Ground energy and state of the dense Hamiltonian in the even sector of the global spin flip,
    /// by power iteration on `c I - H` from a flip-symmetric start. `H` commutes with the flip, so
    /// the sector is preserved and convergence is governed by the gap within it -- not by the
    /// splitting between the sectors, which is exponentially small in `n` in the ordered phase and
    /// left a plain power iteration unconverged after twenty thousand steps. The residual
    /// `|H v - E v|` certifies the answer: an oracle for the oracle must not be taken on trust.
    fn ground_even(h: &[f64], n: usize, shift: f64) -> (f64, Vec<f64>) {
        let dim = 1usize << n;
        let flip = dim - 1;
        let mut v: Vec<f64> = (0..dim)
            .map(|i| 1.0 + (i.min(i ^ flip) as f64 * 0.37).sin())
            .collect();
        let norm = |v: &[f64]| v.iter().map(|x| x * x).sum::<f64>().sqrt();
        let nv = norm(&v);
        for x in &mut v {
            *x /= nv;
        }
        for _ in 0..50_000 {
            let mut w = vec![0.0f64; dim];
            for x in 0..dim {
                let mut acc = shift * v[x];
                for y in 0..dim {
                    acc -= h[x * dim + y] * v[y];
                }
                w[x] = acc;
            }
            for x in 0..dim {
                if x < (x ^ flip) {
                    let m = 0.5 * (w[x] + w[x ^ flip]);
                    w[x] = m;
                    w[x ^ flip] = m;
                }
            }
            let nw = norm(&w);
            for x in &mut w {
                *x /= nw;
            }
            let moved = v
                .iter()
                .zip(&w)
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();
            v = w;
            if moved < 1e-13 {
                break;
            }
        }
        let mut hv = vec![0.0f64; dim];
        for x in 0..dim {
            hv[x] = (0..dim).map(|y| h[x * dim + y] * v[y]).sum();
        }
        let e: f64 = v.iter().zip(&hv).map(|(a, b)| a * b).sum();
        let residual = v
            .iter()
            .zip(&hv)
            .map(|(a, b)| (b - e * a).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(
            residual < 1e-8,
            "the dense oracle did not converge: residual {residual:e} at n {n}"
        );
        (e, v)
    }

    /// The two decoupled limits are exact, and the critical energy density is `-4 J / pi`.
    #[test]
    fn the_limits_and_the_critical_density_are_the_known_ones() {
        assert!((ground_energy(8, 1.5, 0.0) + 12.0).abs() < 1e-12);
        assert!((ground_energy(8, 0.0, 0.7) + 5.6).abs() < 1e-12);
        assert!(
            (ground_energy_density(1.0, 1.0) + 4.0 / PI).abs() < 1e-7,
            "{}",
            ground_energy_density(1.0, 1.0)
        );
        assert!(
            (gap_infinite(1.0, 1.0)).abs() < 1e-15 && (gap_infinite(1.0, 0.25) - 1.5).abs() < 1e-15
        );
        assert!((transverse_magnetisation(10, 0.0, 1.0) - 1.0).abs() < 1e-12);
        assert!(transverse_magnetisation(10, 1.0, 0.0).abs() < 1e-12);
    }

    /// Against exact diagonalisation on 4, 6 and 8 spins, ordered, critical and disordered: the
    /// ground energy to `1e-9` and the transverse magnetisation to `1e-7`.
    #[test]
    fn pfeutys_closed_form_is_the_exact_ground_state_of_the_dense_hamiltonian() {
        for n in [4usize, 6, 8] {
            for &(j, gamma) in &[(1.0f64, 0.5f64), (1.0, 1.0), (1.0, 2.0), (0.7, 1.3)] {
                let dim = 1usize << n;
                let h = dense(n, j, gamma);
                let (e, v) = ground_even(&h, n, n as f64 * (j + gamma) + 1.0);
                let want = ground_energy(n, j, gamma);
                assert!(
                    (e - want).abs() < 1e-9,
                    "n {n} J {j} Gamma {gamma}: dense {e} vs closed form {want}"
                );
                // <sx> = sum over sites of <v| flip_i |v>, over n.
                let mut sx = 0.0;
                for i in 0..n {
                    for x in 0..dim {
                        sx += v[x] * v[x ^ (1 << i)];
                    }
                }
                let sx = sx / v.iter().map(|a| a * a).sum::<f64>() / n as f64;
                let want = transverse_magnetisation(n, j, gamma);
                assert!(
                    (sx - want).abs() < 1e-7,
                    "n {n} J {j} Gamma {gamma}: <sx> dense {sx} vs closed form {want}"
                );
            }
        }
    }

    /// Against the whole spectrum of the dense Hamiltonian by Jacobi on 4, 6 and 8 spins, in the
    /// ordered, critical and disordered phases, at three temperatures: `ln Z` to `1e-9` and the
    /// thermal energy to `1e-8`. The classical limit is the textbook `(2cosh)^N + (2sinh)^N`, and
    /// at `beta = 20_000` the thermal energy is the ground energy to `1e-10` -- not at `beta = 100`,
    /// where the ordered chain at `Gamma = 0.5` still sits `7e-4` above it, because its two parity
    /// sectors' ground states split by only `1.46e-3` (of the order of `(Gamma/J)^N`), and
    /// `e^(-100 x 0.00146)` is `0.86`; even at `beta = 10_000` the residual is `7e-10`. The finite
    /// chain in the ordered phase is cold only when `beta` beats that splitting by thirty.
    #[test]
    fn katsuras_finite_temperature_solution_is_the_whole_dense_spectrum() {
        for n in [4usize, 6, 8] {
            for &(j, gamma) in &[(1.0f64, 0.5f64), (1.0, 1.0), (1.0, 2.0), (0.7, 1.3)] {
                let dim = 1usize << n;
                let mut h = dense(n, j, gamma);
                crate::linalg::jacobi_eig(&mut h, dim);
                let levels: Vec<f64> = (0..dim).map(|c| h[c * dim + c]).collect();
                let floor = levels.iter().copied().fold(f64::INFINITY, f64::min);
                for beta in [0.3f64, 1.0, 4.0] {
                    let weights: Vec<f64> = levels
                        .iter()
                        .map(|&e| (-beta * (e - floor)).exp())
                        .collect();
                    let z: f64 = weights.iter().sum();
                    let ln_z = z.ln() - beta * floor;
                    let energy = levels.iter().zip(&weights).map(|(e, w)| e * w).sum::<f64>() / z;
                    let got = log_partition(n, j, gamma, beta);
                    assert!(
                        (got - ln_z).abs() < 1e-9,
                        "n {n} J {j} Gamma {gamma} beta {beta}: ln Z {got} vs dense {ln_z}"
                    );
                    let got = thermal_energy(n, j, gamma, beta);
                    assert!(
                        (got - energy).abs() < 1e-8,
                        "n {n} J {j} Gamma {gamma} beta {beta}: <H> {got} vs dense {energy}"
                    );
                    assert!((free_energy(n, j, gamma, beta) + ln_z / beta).abs() < 1e-9);
                }
            }
        }
        let classical = ((2.0 * 0.8f64.cosh()).powi(8) + (2.0 * 0.8f64.sinh()).powi(8)).ln();
        assert!((log_partition(8, 1.0, 0.0, 0.8) - classical).abs() < 1e-12);
        for &(j, gamma) in &[(1.0f64, 0.5f64), (1.0, 1.0), (1.0, 2.0)] {
            let warm = thermal_energy(8, j, gamma, 100.0);
            let cold = thermal_energy(8, j, gamma, 20_000.0);
            let want = ground_energy(8, j, gamma);
            assert!(
                (cold - want).abs() < 1e-10,
                "beta 20000: {cold} vs ground {want}"
            );
            if gamma < 1.0 {
                assert!(
                    warm - want > 1e-4,
                    "beta 100 in the ordered phase: {warm} vs ground {want}"
                );
            }
        }
        assert!(
            log_partition(64, 1.0, 1.0, 100.0).is_finite(),
            "log space, not overflow"
        );
    }

    /// At the critical point the finite-size correction to the energy per site is the Casimir term
    /// of a `c = 1/2` conformal field theory with velocity `v = 2J`: `E_0/N = -4J/pi - pi c v/(6 N^2)`
    /// (Bloete, Cardy and Nightingale 1986; Affleck 1986). At 64 sites the coefficient is `-pi/6` to
    /// a part in ten thousand, and it converges from 8 sites. The periodic momenta give `+1.047`
    /// instead, so this is also the test that knows which sector the ground state is in.
    #[test]
    fn the_critical_chains_finite_size_correction_is_the_casimir_term_of_c_one_half() {
        let coefficient =
            |n: usize| (ground_energy(n, 1.0, 1.0) / n as f64 + 4.0 / PI) * (n as f64).powi(2);
        let want = -PI / 6.0;
        let at_64 = coefficient(64);
        let at_8 = coefficient(8);
        assert!(
            (at_64 - want).abs() / want.abs() < 1e-3,
            "at 64 sites: {at_64} vs {want}"
        );
        assert!(
            (at_8 - want).abs() / want.abs() < 1e-2,
            "at 8 sites: {at_8} vs {want}"
        );
        assert!(
            (at_64 - want).abs() < (at_8 - want).abs(),
            "no convergence: {at_8} then {at_64}"
        );
    }

    /// The module header says this spectrum is "an oracle for every quantum sampler in this crate".
    /// It was not one: until this test, `freefermion` was named by exactly one file outside itself,
    /// `lib.rs`, which only declares the module. Nothing in `sse`, `sqa`, `vmc`, `qaoa`, `mps` or
    /// `tensor` referred to it, and `thermal_energy`'s own line — "the exact thermal energy, which
    /// `crate::sse` estimates with an error bar" — described a pairing that had never been made.
    ///
    /// # What it buys
    ///
    /// `sse`'s non-trivial quantum checks are exact diagonalisation of a **two-site chain** and a
    /// **four-site ring**; the rest are limits where the model stops being quantum (zero transverse
    /// field) or stops being coupled (decoupled sites). Jordan–Wigner has no such ceiling: the
    /// chain is free fermions at every size, so this scores the sampler at `n = 32`, where a dense
    /// Hamiltonian would be `2^32` on a side.
    ///
    /// # Why the mean over seeds
    ///
    /// One seed is one draw. Scoring `n = 64` at a much larger budget, the eight seeds `19..97`
    /// gave `z` from `-0.91` to `+2.79` and averaged `+0.10` — so a single-seed threshold either
    /// has to be loose enough to be nearly vacuous or it is flaky. The mean over seeds has a
    /// standard error of `1/sqrt(k)` and the threshold can be tight.
    #[test]
    fn the_quantum_sampler_is_scored_against_the_exact_chain_beyond_diagonalisation() {
        let (n, beta, gamma) = (32usize, 1.2f64, 1.8f64);
        let seeds = [19u64, 23, 31];
        let g = crate::ising::ring(n, 1.0, 0.0);

        let mean_z = |gamma_truth: f64| -> f64 {
            let exact = thermal_energy(n, 1.0, gamma_truth, beta);
            let mut total = 0.0;
            for &seed in &seeds {
                let p = crate::sse::Params {
                    beta,
                    gamma,
                    equilibrate: 2_000,
                    measure: 20_000,
                    cutoff: 16,
                };
                let out = crate::sse::run(&g, &p, seed).expect("these parameters are valid");
                // The crate's own warning: a truncated string is biased, so a disagreement below
                // would have two possible causes and this test could not tell them apart.
                assert_eq!(out.saturated, 0, "the operator string truncated at seed {seed}");
                assert!(out.energy.stderr > 0.0, "a measured energy has an error bar");
                total += (out.energy.value - exact) / out.energy.stderr;
            }
            total / seeds.len() as f64
        };

        let agree = mean_z(gamma);
        assert!(
            agree.abs() < 2.5,
            "the sampler must agree with the exact chain at n={n}: mean z = {agree:+.2}"
        );

        // THE CONTROL. A comparison that cannot reject anything is not a comparison. Two percent in
        // the transverse field, which is a small enough error to be a plausible defect.
        let wrong = mean_z(gamma * 1.02);
        assert!(
            wrong > 6.0,
            "a 2% error in Gamma must be caught: mean z = {wrong:+.2}, and agreement was {agree:+.2}"
        );
    }
}
