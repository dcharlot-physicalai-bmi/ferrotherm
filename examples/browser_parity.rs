//! The native half of the browser certificate comparison: one JSON line, same calls.
//!
//! Deliberately driven through the **C ABI** rather than the Rust API, because that is what the
//! browser can reach. A comparison whose two sides enter the library by different doors measures
//! the doors as much as the library.
//!
//! Paired with `scripts/cert-wasm.mjs` by `scripts/check-browser-certificate.sh`.
use ferrotherm::ffi::*;

fn main() {
    // `ising::ring(10, 1.0, 0.3)`, built the way the browser must build it. Ten spins, so the
    // exact Boltzmann distribution is enumerable and `tv` is a distance rather than an estimate.
    const N: u32 = 10;
    let b = ft_builder_new(N);
    for i in 0..N {
        assert_eq!(ft_builder_couple(b, i, (i + 1) % N, 1.0), 1, "couple {i}");
        assert_eq!(ft_builder_bias(b, i, 0.3), 1, "bias {i}");
    }
    let sim = ft_builder_build(b, 0.5, 11);
    assert!(!sim.is_null(), "the builder produced no simulation");
    assert_eq!(ft_certify(sim, 3000, 8), 1, "certify refused");

    println!(
        "{{\"beta_eff\":{},\"beta_lo\":{},\"beta_hi\":{},\"tau\":{},\"ess\":{},\"tv\":{},\
         \"floor\":{},\"passed\":{},\"findings\":{}}}",
        ft_cert_beta_eff(sim),
        ft_cert_beta_lo(sim),
        ft_cert_beta_hi(sim),
        ft_cert_tau(sim),
        ft_cert_ess(sim),
        ft_cert_tv(sim),
        ft_cert_floor(sim),
        ft_cert_passed(sim) == 1,
        ft_cert_findings(sim),
    );
    ft_free(sim);
}
