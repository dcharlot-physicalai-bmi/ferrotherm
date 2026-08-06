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
    /// Built on first request; a GPU model is pure derived data and most runs never ask for one.
    gpu: Option<crate::wgsl::GpuModel>,
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
        Box::into_raw(Box::new(Sim { sampler_state: sampler.s.clone(), graph: g, beta, seed, sweeps_done: 0, ledger: Ledger::default(), gpu: None }))
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

// ---- arbitrary graphs -------------------------------------------------------------------------
//
// The two constructors above cover the shapes this crate ships. A workbench needs to build a model
// the caller invented, so the builder is exposed as its own handle: create it, add couplings and
// biases one at a time, then consume it into a simulation. Incremental calls keep the ABI free of
// array marshalling, which is the part that goes wrong across a language boundary.

use crate::graph::GraphBuilder;
use crate::tempering::{anneal, geometric_ladder};

/// New graph builder over `n` nodes. Consume it with [`ft_builder_build`] or release it with
/// [`ft_builder_free`]; dropping the handle without either leaks it.
#[no_mangle]
pub extern "C" fn ft_builder_new(n: u32) -> *mut GraphBuilder {
    if n == 0 {
        return core::ptr::null_mut();
    }
    Box::into_raw(Box::new(GraphBuilder::new(n as usize)))
}

/// Add a coupling. Returns 1 on success, 0 if the handle is null, an index is out of range, `i`
/// equals `j`, or the weight is not finite.
#[no_mangle]
pub extern "C" fn ft_builder_couple(b: *mut GraphBuilder, i: u32, j: u32, w: f64) -> u32 {
    let Some(b) = (unsafe { b.as_mut() }) else { return 0 };
    if i == j || !w.is_finite() || i as usize >= b.n() || j as usize >= b.n() {
        return 0;
    }
    b.couple(i as usize, j as usize, w);
    1
}

/// Add a bias. Returns 1 on success, 0 on a null handle, an out-of-range index, or a non-finite h.
#[no_mangle]
pub extern "C" fn ft_builder_bias(b: *mut GraphBuilder, i: u32, h: f64) -> u32 {
    let Some(bb) = (unsafe { b.as_mut() }) else { return 0 };
    if !h.is_finite() || i as usize >= bb.n() {
        return 0;
    }
    bb.bias(i as usize, h);
    1
}

/// Consume the builder into a simulation. The builder handle is invalid after this call.
#[no_mangle]
pub extern "C" fn ft_builder_build(b: *mut GraphBuilder, beta: f64, seed: u64) -> *mut Sim {
    if b.is_null() {
        return core::ptr::null_mut();
    }
    let b = unsafe { Box::from_raw(b) };
    Sim::new(b.build(), beta, seed)
}

/// Release a builder that was never built.
#[no_mangle]
pub extern "C" fn ft_builder_free(b: *mut GraphBuilder) {
    if !b.is_null() {
        drop(unsafe { Box::from_raw(b) });
    }
}

/// Anneal down a geometric ladder from `beta_min` to `beta_max`, leaving the simulation holding the
/// lowest-energy state found and returning that energy. Returns NaN on a null handle or bad ladder.
#[no_mangle]
pub extern "C" fn ft_anneal(
    sim: *mut Sim,
    beta_min: f64,
    beta_max: f64,
    stages: u32,
    sweeps_per_stage: u32,
) -> f64 {
    let Some(s) = (unsafe { sim.as_mut() }) else { return f64::NAN };
    if !(beta_min > 0.0 && beta_max > beta_min) || stages < 2 || sweeps_per_stage == 0 {
        return f64::NAN;
    }
    let ladder = geometric_ladder(beta_min, beta_max, stages as usize);
    let schedule: Vec<(f64, usize)> =
        ladder.iter().map(|&b| (b, sweeps_per_stage as usize)).collect();
    let seed = s.seed ^ s.sweeps_done.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    let (best, e) = anneal(&s.graph, &schedule, seed, Some(&mut s.ledger));
    s.sampler_state.copy_from_slice(&best);
    s.sweeps_done += (stages as u64) * (sweeps_per_stage as u64);
    s.beta = beta_max;
    e
}

/// Node count of a simulation's graph, or 0 on null.
#[no_mangle]
pub extern "C" fn ft_nodes(sim: *const Sim) -> u32 {
    match unsafe { sim.as_ref() } {
        Some(s) => s.graph.n as u32,
        None => 0,
    }
}

/// Total node updates charged to the ledger so far, or 0 on null.
#[no_mangle]
pub extern "C" fn ft_ledger_updates(sim: *const Sim) -> u64 {
    match unsafe { sim.as_ref() } {
        Some(s) => s.ledger.samples,
        None => 0,
    }
}

#[cfg(test)]
mod builder_tests {
    use super::*;

    #[test]
    fn builds_and_samples_an_arbitrary_graph() {
        let b = ft_builder_new(4);
        assert!(!b.is_null());
        assert_eq!(ft_builder_couple(b, 0, 1, 1.0), 1);
        assert_eq!(ft_builder_couple(b, 1, 2, 1.0), 1);
        assert_eq!(ft_builder_bias(b, 0, 0.5), 1);
        let sim = ft_builder_build(b, 1.0, 7);
        assert_eq!(ft_nodes(sim), 4);
        ft_sweep(sim, 50);
        assert!(ft_energy(sim).is_finite());
        assert!(ft_ledger_updates(sim) >= 200);
        ft_free(sim);
    }

    #[test]
    fn rejects_bad_edges_without_crashing() {
        let b = ft_builder_new(3);
        assert_eq!(ft_builder_couple(b, 0, 9, 1.0), 0, "out of range");
        assert_eq!(ft_builder_couple(b, 1, 1, 1.0), 0, "self coupling");
        assert_eq!(ft_builder_couple(b, 0, 1, f64::NAN), 0, "non-finite");
        assert_eq!(ft_builder_bias(b, 7, 1.0), 0, "out of range");
        ft_builder_free(b);
        // null handles are inert, not a crash
        assert_eq!(ft_builder_couple(core::ptr::null_mut(), 0, 1, 1.0), 0);
        assert_eq!(ft_nodes(core::ptr::null()), 0);
        assert!(ft_anneal(core::ptr::null_mut(), 0.1, 1.0, 4, 4).is_nan());
    }

    #[test]
    fn anneal_finds_the_frustrated_optimum() {
        // odd antiferromagnetic ring: one bond must stay unsatisfied, so -3 is the floor
        let b = ft_builder_new(5);
        for i in 0..5u32 {
            ft_builder_couple(b, i, (i + 1) % 5, -1.0);
        }
        let sim = ft_builder_build(b, 0.1, 1);
        let e = ft_anneal(sim, 0.05, 6.0, 40, 30);
        assert_eq!(e, -3.0, "frustrated 5-cycle optimum");
        assert_eq!(ft_energy(sim), -3.0, "sim must hold the best state");
        ft_free(sim);
    }
}

// ---- the GPU path ------------------------------------------------------------------------------
//
// A browser needs three things to run the sweep on a GPU: the shader, the padded interaction
// rectangle, and the colour classes. All three come from here rather than being rebuilt in
// JavaScript, so there is one source of truth and the tested Rust layout is the one that ships.

use crate::wgsl::{sweep_shader, GpuModel};

fn ensure_gpu(s: &mut Sim) -> &GpuModel {
    if s.gpu.is_none() {
        s.gpu = Some(GpuModel::from_graph(&s.graph));
    }
    s.gpu.as_ref().unwrap()
}

/// Row width of the padded interaction rectangle, or 0 on null.
#[no_mangle]
pub extern "C" fn ft_gpu_k(sim: *mut Sim) -> u32 {
    match unsafe { sim.as_mut() } {
        Some(s) => ensure_gpu(s).k,
        None => 0,
    }
}

/// `n * k` neighbour indices.
#[no_mangle]
pub extern "C" fn ft_gpu_nbr(sim: *mut Sim) -> *const u32 {
    match unsafe { sim.as_mut() } {
        Some(s) => ensure_gpu(s).nbr.as_ptr(),
        None => core::ptr::null(),
    }
}

/// `n * k` couplings as f32, the width a GPU actually has.
#[no_mangle]
pub extern "C" fn ft_gpu_w(sim: *mut Sim) -> *const f32 {
    match unsafe { sim.as_mut() } {
        Some(s) => ensure_gpu(s).w.as_ptr(),
        None => core::ptr::null(),
    }
}

/// `n` biases as f32.
#[no_mangle]
pub extern "C" fn ft_gpu_h(sim: *mut Sim) -> *const f32 {
    match unsafe { sim.as_mut() } {
        Some(s) => ensure_gpu(s).h.as_ptr(),
        None => core::ptr::null(),
    }
}

/// Number of colour classes. Nodes within one class share no edge and update together.
#[no_mangle]
pub extern "C" fn ft_gpu_classes(sim: *mut Sim) -> u32 {
    match unsafe { sim.as_mut() } {
        Some(s) => ensure_gpu(s).classes.len() as u32,
        None => 0,
    }
}

/// Length of colour class `c`.
#[no_mangle]
pub extern "C" fn ft_gpu_class_len(sim: *mut Sim, c: u32) -> u32 {
    match unsafe { sim.as_mut() } {
        Some(s) => ensure_gpu(s).classes.get(c as usize).map_or(0, |v| v.len() as u32),
        None => 0,
    }
}

/// Node indices of colour class `c`.
#[no_mangle]
pub extern "C" fn ft_gpu_class_ptr(sim: *mut Sim, c: u32) -> *const u32 {
    match unsafe { sim.as_mut() } {
        Some(s) => ensure_gpu(s).classes.get(c as usize).map_or(core::ptr::null(), |v| v.as_ptr()),
        None => core::ptr::null(),
    }
}

/// Overwrite the simulation's state, so a GPU result can be read back into it and then scored,
/// certified or annealed by exactly the same code that handles a CPU result.
#[no_mangle]
pub extern "C" fn ft_set_spins(sim: *mut Sim, ptr: *const i8, len: u32) -> u32 {
    let Some(s) = (unsafe { sim.as_mut() }) else { return 0 };
    if ptr.is_null() || len as usize != s.sampler_state.len() {
        return 0;
    }
    let src = unsafe { core::slice::from_raw_parts(ptr, len as usize) };
    if src.iter().any(|&v| v != 1 && v != -1) {
        return 0; // states are -1/+1; refusing beats silently sampling nonsense
    }
    s.sampler_state.copy_from_slice(src);
    1
}

/// Pointer to the WGSL sweep shader, NUL-free. Pair with [`ft_shader_len`].
///
/// The browser takes the shader from here rather than carrying its own copy, so the emitted
/// arithmetic and the tested arithmetic cannot drift apart.
#[no_mangle]
pub extern "C" fn ft_shader() -> *const u8 {
    shader_bytes().as_ptr()
}

#[no_mangle]
pub extern "C" fn ft_shader_len() -> u32 {
    shader_bytes().len() as u32
}

fn shader_bytes() -> &'static [u8] {
    use std::sync::OnceLock;
    static SRC: OnceLock<String> = OnceLock::new();
    SRC.get_or_init(sweep_shader).as_bytes()
}

#[cfg(test)]
mod gpu_tests {
    use super::*;

    #[test]
    fn the_gpu_view_matches_the_graph() {
        let sim = ft_ising2d_new(8, 1.0, 0.44, 1);
        assert_eq!(ft_gpu_k(sim), 4, "a square lattice has degree 4");
        assert_eq!(ft_gpu_classes(sim), 2, "a bipartite lattice has two colours");
        let total: u32 = (0..ft_gpu_classes(sim)).map(|c| ft_gpu_class_len(sim, c)).sum();
        assert_eq!(total, ft_len(sim), "every node belongs to exactly one class");
        assert!(!ft_gpu_nbr(sim).is_null() && !ft_gpu_w(sim).is_null());
        ft_free(sim);
    }

    #[test]
    fn the_shader_crosses_the_boundary_intact() {
        let len = ft_shader_len() as usize;
        let src = unsafe { core::slice::from_raw_parts(ft_shader(), len) };
        let s = core::str::from_utf8(src).expect("the shader must be valid UTF-8");
        assert!(s.contains("@compute"), "not a compute shader");
        assert!(s.contains("1.0 / (1.0 + exp(-2.0 * P.beta * f))"), "the update must survive");
    }

    #[test]
    fn a_state_can_be_read_back_in() {
        let sim = ft_ising2d_new(4, 1.0, 1.0, 1);
        let n = ft_len(sim) as usize;
        let up = vec![1i8; n];
        assert_eq!(ft_set_spins(sim, up.as_ptr(), n as u32), 1);
        assert_eq!(ft_energy(sim), -2.0 * n as f64, "all aligned on a degree-4 lattice");
        // and malformed input is refused rather than absorbed
        let bad = vec![0i8; n];
        assert_eq!(ft_set_spins(sim, bad.as_ptr(), n as u32), 0);
        assert_eq!(ft_set_spins(sim, up.as_ptr(), 3), 0, "wrong length");
        ft_free(sim);
    }

    #[test]
    fn null_handles_stay_inert() {
        assert_eq!(ft_gpu_k(core::ptr::null_mut()), 0);
        assert_eq!(ft_gpu_classes(core::ptr::null_mut()), 0);
        assert!(ft_gpu_nbr(core::ptr::null_mut()).is_null());
        assert_eq!(ft_set_spins(core::ptr::null_mut(), core::ptr::null(), 0), 0);
    }
}

/// Local field at node `i`: `sum_j J_ij s_j + h_i`, with beta excluded. NaN on null or out of range.
///
/// Exposed so a GPU result can be compared against the field the CPU computes for the same state,
/// which is a far sharper instrument than comparing the states that come out the other end.
#[no_mangle]
pub extern "C" fn ft_field(sim: *const Sim, i: u32) -> f64 {
    match unsafe { sim.as_ref() } {
        Some(s) if (i as usize) < s.graph.n => s.graph.field(i as usize, &s.sampler_state),
        _ => f64::NAN,
    }
}
