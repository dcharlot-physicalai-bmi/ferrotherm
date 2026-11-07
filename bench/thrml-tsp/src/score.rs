//! One scoring path for both runners.
//!
//! Total variation and KL are against the exact Boltzmann law on the whole state space.
//! `tau` and the effective sample size are `ferrotherm::potts::estimate` on the energy trace,
//! which is Geyer's initial monotone sequence cross-checked against long-batch means, floored
//! at 1/2, with `ESS = N / (2 tau)`. The same function scores both chains, so a difference in
//! `tau` is a difference in the samples and not a difference in the estimator.
//!
//! KL(empirical || exact) is finite whenever the exact law has full support, which it does at
//! finite beta and finite couplings. KL(exact || empirical) is infinite when the chain missed
//! a state; that infinity is reported as null plus the number of missed states, not as a large
//! finite stand-in.

use ferrotherm::ising::tv;
use ferrotherm::potts::estimate;

#[derive(Clone, Debug)]
pub struct Score {
    pub tv: f64,
    pub kl_empirical_exact: f64,
    pub kl_exact_empirical: Option<f64>,
    pub missed_states: usize,
    pub tau_int_energy: f64,
    pub ess_energy: f64,
    pub ess_per_second: Option<f64>,
    pub mean_energy: f64,
    pub exact_mean_energy: f64,
    pub energy_abs_err: f64,
    pub magnetization_mae: Option<f64>,
    pub seconds: f64,
    pub draws: usize,
}

/// Energy, autocorrelation and ESS, without a histogram.
///
/// Used when the state space is not enumerated. `tv` and the KLs are non-finite and the
/// report prints them as null. `exact_mean_energy` is whatever the caller proved: a
/// transfer-matrix mean, or Onsager's infinite-lattice energy times `N`.
pub fn score_mean_only(energies: &[f64], exact_mean_energy: f64, seconds: f64) -> Score {
    let est = estimate(energies);
    let ess_per_second = if seconds > 0.0 && est.ess.is_finite() { Some(est.ess / seconds) } else { None };
    Score {
        tv: f64::NAN,
        kl_empirical_exact: f64::NAN,
        kl_exact_empirical: None,
        missed_states: 0,
        tau_int_energy: est.tau_int,
        ess_energy: est.ess,
        ess_per_second,
        mean_energy: est.value,
        exact_mean_energy,
        energy_abs_err: (est.value - exact_mean_energy).abs(),
        magnetization_mae: None,
        seconds,
        draws: energies.len(),
    }
}

pub fn score_chain(
    hist: &[f64],
    exact: &[f64],
    energies: &[f64],
    exact_mean_energy: f64,
    mag_mae: Option<f64>,
    seconds: f64,
) -> Score {
    let est = estimate(energies);
    let ess_per_second = if seconds > 0.0 && est.ess.is_finite() { Some(est.ess / seconds) } else { None };
    let (kl_ee, missed) = kl_exact_to_empirical(exact, hist);
    Score {
        tv: tv(hist, exact),
        kl_empirical_exact: kl(hist, exact),
        kl_exact_empirical: kl_ee,
        missed_states: missed,
        tau_int_energy: est.tau_int,
        ess_energy: est.ess,
        ess_per_second,
        mean_energy: est.value,
        exact_mean_energy,
        energy_abs_err: (est.value - exact_mean_energy).abs(),
        magnetization_mae: mag_mae,
        seconds,
        draws: energies.len(),
    }
}

pub fn histogram_spins(states: &[Vec<i8>]) -> Vec<f64> {
    let n = states.first().map_or(0, Vec::len);
    let m = 1usize << n;
    let mut h = vec![0.0f64; m];
    for s in states {
        let mut mask = 0usize;
        for (b, &v) in s.iter().enumerate() {
            if v == 1 {
                mask |= 1 << b;
            }
        }
        h[mask] += 1.0;
    }
    let inv = 1.0 / states.len() as f64;
    for v in &mut h {
        *v *= inv;
    }
    h
}

pub fn histogram_potts(states: &[Vec<u8>], q: usize) -> Vec<f64> {
    let n = states.first().map_or(0, Vec::len);
    let m = q.pow(u32::try_from(n).expect("n fits u32"));
    let mut h = vec![0.0f64; m];
    for s in states {
        let mut idx = 0usize;
        for &v in s.iter().rev() {
            idx = idx * q + usize::from(v);
        }
        h[idx] += 1.0;
    }
    let inv = 1.0 / states.len() as f64;
    for v in &mut h {
        *v *= inv;
    }
    h
}

/// Mean of each spin under `p`, bitmask convention (bit b set => +1).
pub fn exact_magnetization(p: &[f64], n: usize) -> Vec<f64> {
    let mut m = vec![0.0f64; n];
    for (mask, &pk) in p.iter().enumerate() {
        for b in 0..n {
            let s = if mask >> b & 1 == 1 { 1.0 } else { -1.0 };
            m[b] += pk * s;
        }
    }
    m
}

pub fn empirical_magnetization(states: &[Vec<i8>]) -> Vec<f64> {
    let n = states.first().map_or(0, Vec::len);
    let mut m = vec![0.0f64; n];
    for s in states {
        for (b, &v) in s.iter().enumerate() {
            m[b] += f64::from(v);
        }
    }
    let inv = 1.0 / states.len() as f64;
    for v in &mut m {
        *v *= inv;
    }
    m
}

pub fn mae(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum::<f64>() / a.len() as f64
}

/// `sum p ln(p/q)` over `p > 0`. Infinite when some such `q` is 0.
fn kl(p: &[f64], q: &[f64]) -> f64 {
    let mut k = 0.0;
    for (i, &pi) in p.iter().enumerate() {
        if pi > 0.0 {
            let qi = q[i];
            if !(qi > 0.0) {
                return f64::INFINITY;
            }
            k += pi * (pi.ln() - qi.ln());
        }
    }
    k
}

fn kl_exact_to_empirical(exact: &[f64], hist: &[f64]) -> (Option<f64>, usize) {
    let mut missed = 0usize;
    for (i, &pk) in exact.iter().enumerate() {
        if pk > 0.0 && !(hist[i] > 0.0) {
            missed += 1;
        }
    }
    if missed > 0 {
        (None, missed)
    } else {
        (Some(kl(exact, hist)), 0)
    }
}

/// Student-t 95% interval. `None` when fewer than two finite numbers were supplied:
/// one draw has no interval, and a non-finite draw is not a draw.
pub fn mean_ci_95(xs: &[f64]) -> Option<(f64, f64, f64, usize)> {
    let v: Vec<f64> = xs.iter().copied().filter(|x| x.is_finite()).collect();
    if v.len() < 2 {
        return None;
    }
    let n = v.len();
    let mean = v.iter().sum::<f64>() / n as f64;
    let var = v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
    let half = t_crit_95(n - 1) * (var / n as f64).sqrt();
    Some((mean, mean - half, mean + half, n))
}

/// Two-sided Student t critical value at 97.5%, `df` degrees of freedom.
/// Values for `df <= 30` are the standard table to 10 digits. Above that, 1.9599639845,
/// which is `z_0.975` and is already the table at `df = 30` to three digits.
pub fn t_crit_95(df: usize) -> f64 {
    const T: [f64; 30] = [
        12.7062047364,
        4.3026527299,
        3.1824463053,
        2.7764451052,
        2.5705818366,
        2.4469118510,
        2.3646242520,
        2.3060041352,
        2.2621571627,
        2.2281388520,
        2.2009851601,
        2.1788128297,
        2.1603686565,
        2.1447866879,
        2.1314495456,
        2.1199052992,
        2.1098155778,
        2.1009220402,
        2.0930240544,
        2.0859634473,
        2.0796138447,
        2.0738730679,
        2.0686576104,
        2.0638985616,
        2.0595385528,
        2.0555294386,
        2.0518305165,
        2.0484071418,
        2.0452296421,
        2.0422724563,
    ];
    if df == 0 {
        f64::NAN
    } else if df <= 30 {
        T[df - 1]
    } else {
        1.9599639845
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_laws_score_zero() {
        let p = vec![0.25, 0.25, 0.25, 0.25];
        let energies = vec![0.0; 64];
        // A constant trace has infinite tau and ess 1. TV and KL are the claim here.
        let s = score_chain(&p, &p, &energies, 0.0, None, 1.0);
        assert!(s.tv.abs() < 1e-15);
        assert!(s.kl_empirical_exact.abs() < 1e-15);
        assert!(s.kl_exact_empirical.unwrap().abs() < 1e-15);
        assert_eq!(s.missed_states, 0);
    }

    #[test]
    fn a_missed_state_is_not_a_finite_kl() {
        let exact = vec![0.5, 0.5];
        let hist = vec![1.0, 0.0];
        let (kl, missed) = super::kl_exact_to_empirical(&exact, &hist);
        assert!(kl.is_none());
        assert_eq!(missed, 1);
    }

    #[test]
    fn the_t_table_has_the_one_degree_value() {
        assert!((t_crit_95(1) - 12.7062047364).abs() < 1e-12);
        assert!((t_crit_95(2) - 4.3026527299).abs() < 1e-12);
        let (mean, lo, hi, n) = mean_ci_95(&[1.0, 3.0]).unwrap();
        assert_eq!(n, 2);
        assert!((mean - 2.0).abs() < 1e-12);
        assert!(lo < mean && mean < hi);
        assert!(mean_ci_95(&[1.0]).is_none());
        assert!(mean_ci_95(&[f64::NAN, f64::NAN]).is_none());
    }

    #[test]
    fn spin_histogram_uses_the_bitmask_the_oracle_uses() {
        // Two states: all -1 (mask 0) and spin 0 up (mask 1), one each.
        let states = vec![vec![-1, -1], vec![1, -1]];
        let h = histogram_spins(&states);
        assert_eq!(h.len(), 4);
        assert!((h[0] - 0.5).abs() < 1e-15);
        assert!((h[1] - 0.5).abs() < 1e-15);
        assert_eq!(h[2], 0.0);
    }
}
