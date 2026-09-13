//! Radix-2 fast Fourier transform, and the autocovariance it makes cheap.
//!
//! [`crate::certify::tau_int`] computes each lag of the autocorrelation directly, `O(L)` per lag,
//! which is right for Sokal's window — a few `tau` lags — and hopeless for the extended sums and
//! long traces that MEASURING the window's failure needs: `examples/tau_exactness.rs` at `beta = 1.5`
//! sat on a `1.3e8`-sweep trace for five hours (2026-09-13) and was stopped. The Wiener–Khinchin
//! route — centre, zero-pad to at least twice the length, transform, square the magnitudes,
//! transform back — is `O(L log L)` for every lag at once, and [`autocovariance`] returns exactly
//! the per-lag-normalised sequence `certify::tau_int` sums, to rounding, so an estimator built on
//! it is the same estimator with a different clock.
//!
//! Pure Rust, iterative, in place, power-of-two lengths; the caller pads. The twiddle recurrence is
//! re-anchored to a direct `cos`/`sin` every 32 butterflies, so a `2^28`-point transform stays at
//! `1e-13` instead of drifting with its length.

use std::f64::consts::TAU;

/// In-place forward transform of the complex sequence `(re, im)`: `X_k = sum_n x_n e^{-2 pi i n k / N}`,
/// unnormalised. The length must be a power of two.
///
/// # Panics
///
/// If `re` and `im` differ in length, or the length is not a power of two.
pub fn fft(re: &mut [f64], im: &mut [f64]) {
    transform(re, im, false);
}

/// In-place inverse transform, normalised by `1 / N`, so `ifft(fft(x)) = x`.
///
/// # Panics
///
/// If `re` and `im` differ in length, or the length is not a power of two.
pub fn ifft(re: &mut [f64], im: &mut [f64]) {
    transform(re, im, true);
    let n = re.len() as f64;
    for v in re.iter_mut() {
        *v /= n;
    }
    for v in im.iter_mut() {
        *v /= n;
    }
}

fn transform(re: &mut [f64], im: &mut [f64], inverse: bool) {
    let n = re.len();
    assert_eq!(n, im.len(), "real and imaginary parts must have the same length");
    assert!(n.is_power_of_two(), "the transform length must be a power of two, not {n}");
    if n < 2 {
        return;
    }
    let bits = n.trailing_zeros();
    for i in 0..n {
        let j = i.reverse_bits() >> (usize::BITS - bits);
        if j > i {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= n {
        let ang = if inverse { TAU / len as f64 } else { -TAU / len as f64 };
        let (wr, wi) = (ang.cos(), ang.sin());
        let half = len / 2;
        for start in (0..n).step_by(len) {
            let (mut cr, mut ci) = (1.0f64, 0.0f64);
            for k in 0..half {
                if k % 32 == 0 {
                    let a = ang * k as f64;
                    cr = a.cos();
                    ci = a.sin();
                }
                let (a, b) = (start + k, start + k + half);
                let tr = re[b] * cr - im[b] * ci;
                let ti = re[b] * ci + im[b] * cr;
                re[b] = re[a] - tr;
                im[b] = im[a] - ti;
                re[a] += tr;
                im[a] += ti;
                let next = cr * wr - ci * wi;
                ci = cr * wi + ci * wr;
                cr = next;
            }
        }
        len <<= 1;
    }
}

/// The autocovariance of `x` at lags `0..=max_lag`, each lag normalised by the number of pairs
/// that formed it: `c(k) = sum_{t < n - k} (x_t - mean)(x_{t + k} - mean) / (n - k)`, which is the
/// quantity [`crate::certify::tau_int`] divides by `c(0)` and sums. Computed for every lag at once
/// by one transform of the centred sequence zero-padded to at least twice its length, so nothing
/// wraps. `max_lag` is clamped to `n - 1`; an empty `x` gives an empty result.
#[must_use]
pub fn autocovariance(x: &[f64], max_lag: usize) -> Vec<f64> {
    let n = x.len();
    if n == 0 {
        return Vec::new();
    }
    let mean = x.iter().sum::<f64>() / n as f64;
    let m = (2 * n).next_power_of_two();
    let mut re = vec![0.0f64; m];
    let mut im = vec![0.0f64; m];
    for (r, v) in re.iter_mut().zip(x) {
        *r = v - mean;
    }
    fft(&mut re, &mut im);
    for (r, i) in re.iter_mut().zip(im.iter_mut()) {
        *r = *r * *r + *i * *i;
        *i = 0.0;
    }
    ifft(&mut re, &mut im);
    let top = max_lag.min(n - 1);
    (0..=top).map(|k| re[k] / (n - k) as f64).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Pcg;

    /// The direct `O(N^2)` transform, written independently, is the oracle; the inverse must undo
    /// the forward to floating point; and a unit impulse transforms to all ones, exactly.
    #[test]
    fn the_transform_matches_the_direct_dft_and_inverts() {
        let n = 64;
        let mut rng = Pcg::new(3, 1);
        let xr: Vec<f64> = (0..n).map(|_| rng.f64() - 0.5).collect();
        let xi: Vec<f64> = (0..n).map(|_| rng.f64() - 0.5).collect();
        let (mut re, mut im) = (xr.clone(), xi.clone());
        fft(&mut re, &mut im);
        for k in 0..n {
            let (mut dr, mut di) = (0.0f64, 0.0f64);
            for t in 0..n {
                let a = -TAU * (t * k) as f64 / n as f64;
                dr += xr[t] * a.cos() - xi[t] * a.sin();
                di += xr[t] * a.sin() + xi[t] * a.cos();
            }
            assert!((re[k] - dr).abs() < 1e-12 && (im[k] - di).abs() < 1e-12, "bin {k}: fft ({}, {}) vs dft ({dr}, {di})", re[k], im[k]);
        }
        ifft(&mut re, &mut im);
        for t in 0..n {
            assert!((re[t] - xr[t]).abs() < 1e-13 && (im[t] - xi[t]).abs() < 1e-13, "inverse at {t}");
        }
        let mut dr = vec![0.0f64; 16];
        let mut di = vec![0.0f64; 16];
        dr[0] = 1.0;
        fft(&mut dr, &mut di);
        assert!(dr.iter().all(|v| *v == 1.0) && di.iter().all(|v| *v == 0.0), "an impulse must transform to ones");
    }

    /// Lag by lag, the transform route must reproduce the direct sum with the same `n - k`
    /// normalisation on a length that is not a power of two (so the padding is exercised), and
    /// summing its ratios under Sokal's window must give `certify::tau_int` to rounding -- the
    /// same estimator with a different clock.
    #[test]
    fn the_autocovariance_matches_the_direct_sum_lag_by_lag_and_sokal_agrees() {
        let n = 1000;
        let mut rng = Pcg::new(7, 2);
        let mut x = Vec::with_capacity(n);
        let mut v = 0.0f64;
        for _ in 0..n {
            v = 0.9 * v + (rng.f64() - 0.5);
            x.push(v);
        }
        let c = autocovariance(&x, 200);
        let mean = x.iter().sum::<f64>() / n as f64;
        for (k, ck) in c.iter().enumerate() {
            let direct: f64 = (0..n - k).map(|t| (x[t] - mean) * (x[t + k] - mean)).sum::<f64>() / (n - k) as f64;
            assert!((ck - direct).abs() < 1e-12 * c[0], "lag {k}: fft {ck} vs direct {direct}");
        }
        let mut tau = 0.5;
        for (k, ck) in c.iter().enumerate().skip(1) {
            tau += ck / c[0];
            if (k as f64) >= 5.0 * tau.max(0.5) {
                break;
            }
        }
        let sokal = crate::certify::tau_int(&x);
        assert!((tau - sokal).abs() < 1e-12, "windowed sum {tau} vs certify::tau_int {sokal}");
    }
}
