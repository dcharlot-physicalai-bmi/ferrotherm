//! C ABI for WebAssembly and host-language bindings.
//!
//! Build with `crate-type = ["cdylib"]` for a `.wasm` (wasm32-unknown-unknown) or a native shared
//! library. The surface is deliberately small, stateless-per-handle, and copy-free where it
//! matters: create a simulation, sweep it, read the spin field pointer, read the ledger. It is
//! what the in-browser workbench binds to, and it is designed so an AI agent can drive it from
//! the function names alone.
//!
//! Safety: handles are opaque pointers owned by the library; every function checks for null.
//! One simulation is single-threaded; concurrent calls on one handle are the caller's bug.

use crate::device::z1_grid;
use crate::gibbs::Sampler;
use crate::graph::Graph;
use crate::ising::{lattice2d, onsager_m};
use crate::ledger::{Ledger, Z1_SPICE};

pub struct Sim {
    graph: Box<Graph>,
    sampler_state: Vec<i8>,
    beta: f64,
    seed: u64,
    sweeps_done: u64,
    ledger: Ledger,
}

impl Sim {
    fn new(graph: Graph, beta: f64, seed: u64) -> *mut Sim {
        let g = Box::new(graph);
        // SAFETY of the self-reference dance avoided: store state, rebuild Sampler per call.
        let sampler = Sampler::new(&g, beta, seed);
        Box::into_raw(Box::new(Sim { sampler_state: sampler.s.clone(), graph: g, beta, seed, sweeps_done: 0, ledger: Ledger::default() }))
    }
}

/// New 2D nearest-neighbour Ising lattice (periodic), side `l`, coupling `j`.
#[no_mangle]
pub extern "C" fn ft_ising2d_new(l: u32, j: f64, beta: f64, seed: u64) -> *mut Sim {
    Sim::new(lattice2d(l as usize, j), beta, seed)
}

/// New Z1-topology grid (degree 16, open boundaries), `w` x `h`, uniform coupling `j`, bias `hb`.
#[no_mangle]
pub extern "C" fn ft_z1_new(w: u32, h: u32, j: f64, hb: f64, beta: f64, seed: u64) -> *mut Sim {
    Sim::new(z1_grid(w as usize, h as usize, j, hb), beta, seed)
}

/// Run `n` chromatic Gibbs sweeps. Returns the total sweeps done so far, or 0 on null.
#[no_mangle]
pub extern "C" fn ft_sweep(sim: *mut Sim, n: u32) -> u64 {
    let Some(s) = (unsafe { sim.as_mut() }) else { return 0 };
    let mut smp = Sampler::new(&s.graph, s.beta, s.seed ^ s.sweeps_done.wrapping_mul(0x9E3779B97F4A7C15));
    smp.s.copy_from_slice(&s.sampler_state);
    for _ in 0..n {
        smp.sweep(Some(&mut s.ledger));
    }
    s.sampler_state.copy_from_slice(&smp.s);
    s.sweeps_done += n as u64;
    s.sweeps_done
}

/// Set the inverse temperature (annealing from the host side).
#[no_mangle]
pub extern "C" fn ft_set_beta(sim: *mut Sim, beta: f64) {
    if let Some(s) = unsafe { sim.as_mut() } {
        s.beta = beta;
    }
}

/// Number of spins.
#[no_mangle]
pub extern "C" fn ft_len(sim: *const Sim) -> u32 {
    unsafe { sim.as_ref() }.map_or(0, |s| s.graph.n as u32)
}

/// Pointer to the spin field (i8 per site, values -1/+1), valid until the next ft_ call.
#[no_mangle]
pub extern "C" fn ft_spins(sim: *const Sim) -> *const i8 {
    unsafe { sim.as_ref() }.map_or(std::ptr::null(), |s| s.sampler_state.as_ptr())
}

/// Mean magnetization of the current state.
#[no_mangle]
pub extern "C" fn ft_magnetization(sim: *const Sim) -> f64 {
    unsafe { sim.as_ref() }.map_or(0.0, |s| {
        s.sampler_state.iter().map(|&v| v as i64).sum::<i64>() as f64 / s.graph.n as f64
    })
}

/// Energy of the current state.
#[no_mangle]
pub extern "C" fn ft_energy(sim: *const Sim) -> f64 {
    unsafe { sim.as_ref() }.map_or(0.0, |s| s.graph.energy(&s.sampler_state))
}

/// Joules this simulation WOULD have cost on a Z1-class device (vendor SPICE prices, pre-silicon).
#[no_mangle]
pub extern "C" fn ft_ledger_joules_z1(sim: *const Sim) -> f64 {
    unsafe { sim.as_ref() }.map_or(0.0, |s| s.ledger.joules(&Z1_SPICE))
}

/// Onsager's exact spontaneous magnetization for the 2D lattice at this beta (J = 1).
#[no_mangle]
pub extern "C" fn ft_onsager(beta: f64) -> f64 {
    onsager_m(beta)
}

#[no_mangle]
pub extern "C" fn ft_free(sim: *mut Sim) {
    if !sim.is_null() {
        drop(unsafe { Box::from_raw(sim) });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The FFI path must reproduce the same physics as the library path.
    #[test]
    fn ffi_roundtrip_matches_onsager() {
        let sim = ft_ising2d_new(32, 1.0, 0.6, 42);
        assert_eq!(ft_len(sim), 1024);
        ft_sweep(sim, 2000);
        let mut acc = 0.0;
        let reads = 200;
        for _ in 0..reads {
            ft_sweep(sim, 10);
            acc += ft_magnetization(sim).abs();
        }
        let m = acc / reads as f64;
        let exact = ft_onsager(0.6);
        assert!((m - exact).abs() < 0.02, "FFI |M| {m} vs Onsager {exact}");
        assert!(ft_ledger_joules_z1(sim) > 0.0);
        assert!(!ft_spins(sim).is_null());
        ft_free(sim);
    }
}
