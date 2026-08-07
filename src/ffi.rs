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
    /// Known optimum, when this simulation came from a planted instance.
    ground: Option<f64>,
    /// The last certificate, if `ft_certify` has been called.
    cert: Option<crate::certify::Certificate>,
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
        Box::into_raw(Box::new(Sim { sampler_state: sampler.s.clone(), graph: g, beta, seed, sweeps_done: 0, ledger: Ledger::default(), gpu: None, ground: None, cert: None }))
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
        assert!(s.contains("1.0 / (1.0 + exp(-2.0 * P.ctl.x * f))"), "the update must survive");
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

/// A planted instance whose optimum is known by construction.
///
/// Exposed because a node graph that reports an energy is showing a number nobody can judge. With a
/// planted instance the same graph reports how far it is from the true optimum, which is the
/// difference between a demo and a measurement.
#[no_mangle]
pub extern "C" fn ft_planted_frustrated(l: u32, loops: u32, seed: u64, beta: f64) -> *mut Sim {
    if l < 3 || loops == 0 {
        return core::ptr::null_mut();
    }
    let p = crate::planted::frustrated_loops(l as usize, loops as usize, seed);
    let sim = Sim::new(p.graph, beta, seed);
    if let Some(s) = unsafe { sim.as_mut() } {
        s.ground = Some(p.ground_energy);
    }
    sim
}

/// The Wishart planted ensemble: dense, and genuinely hard below alpha = 1.
#[no_mangle]
pub extern "C" fn ft_planted_wishart(n: u32, alpha: f64, seed: u64, beta: f64) -> *mut Sim {
    if n < 3 || !(alpha > 0.0) {
        return core::ptr::null_mut();
    }
    let p = crate::planted::wishart(n as usize, alpha, seed);
    let sim = Sim::new(p.graph, beta, seed);
    if let Some(s) = unsafe { sim.as_mut() } {
        s.ground = Some(p.ground_energy);
    }
    sim
}

/// The known optimum of a planted instance, or NaN if this simulation is not one.
#[no_mangle]
pub extern "C" fn ft_ground_energy(sim: *const Sim) -> f64 {
    match unsafe { sim.as_ref() } {
        Some(s) => s.ground.unwrap_or(f64::NAN),
        None => f64::NAN,
    }
}

#[cfg(test)]
mod planted_ffi_tests {
    use super::*;

    #[test]
    fn a_planted_instance_carries_its_optimum() {
        let sim = ft_planted_frustrated(6, 40, 3, 1.0);
        assert!(!sim.is_null());
        let known = ft_ground_energy(sim);
        assert_eq!(known, -80.0, "40 plaquettes contribute -2 each");
        // annealing should reach it, and the FFI should agree with the crate
        let e = ft_anneal(sim, 0.05, 6.0, 80, 40);
        assert!(e >= known - 1e-9, "nothing can beat the planted optimum");
        ft_free(sim);
    }

    #[test]
    fn a_wishart_instance_is_dense_and_carries_its_optimum() {
        let sim = ft_planted_wishart(24, 0.5, 1, 1.0);
        assert!(ft_ground_energy(sim).is_finite());
        assert_eq!(ft_gpu_k(sim), 23, "dense: every spin couples to every other");
        ft_free(sim);
    }

    #[test]
    fn an_ordinary_simulation_has_no_known_optimum() {
        let sim = ft_ising2d_new(8, 1.0, 0.44, 1);
        assert!(ft_ground_energy(sim).is_nan(), "only planted instances know their optimum");
        ft_free(sim);
        assert!(ft_planted_frustrated(2, 1, 0, 1.0).is_null(), "too small to have plaquettes");
    }
}

// ---- certificate and exact inference -----------------------------------------------------------
//
// The bindings could build models and sample them, but not check the result or compare it against
// truth -- so Python and Zig could do the easy half of what this crate is for. These close that.

/// Sample `draws` states with `thin` sweeps between them and certify the run.
///
/// The certificate is stored on the simulation; read it with the `ft_cert_*` accessors. Returns 1
/// on success, 0 on a null handle or a degenerate request.
#[no_mangle]
pub extern "C" fn ft_certify(sim: *mut Sim, draws: u32, thin: u32) -> u32 {
    let Some(s) = (unsafe { sim.as_mut() }) else { return 0 };
    if draws < 16 {
        return 0; // certifying 15 samples is theatre; certify::TooFewSamples says so too
    }
    let mut smp = Sampler::new(&s.graph, s.beta, s.seed ^ s.sweeps_done.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    smp.s.copy_from_slice(&s.sampler_state);
    let mut samples = Vec::with_capacity(draws as usize);
    let mut trace = Vec::with_capacity(draws as usize);
    for _ in 0..draws {
        for _ in 0..thin.max(1) {
            smp.sweep(Some(&mut s.ledger));
        }
        samples.push(smp.s.clone());
        trace.push(s.graph.energy(&smp.s));
    }
    s.sampler_state.copy_from_slice(&smp.s);
    s.sweeps_done += draws as u64 * thin.max(1) as u64;
    s.cert = Some(crate::certify::certify(&s.graph, s.beta, &samples, &trace));
    1
}

macro_rules! cert_field {
    ($name:ident, $f:expr) => {
        #[no_mangle]
        pub extern "C" fn $name(sim: *const Sim) -> f64 {
            match unsafe { sim.as_ref() }.and_then(|s| s.cert.as_ref()) {
                Some(c) => $f(c),
                None => f64::NAN,
            }
        }
    };
}

cert_field!(ft_cert_beta_eff, |c: &crate::certify::Certificate| c.beta_eff);
cert_field!(ft_cert_beta_lo, |c: &crate::certify::Certificate| c.beta_ci.0);
cert_field!(ft_cert_beta_hi, |c: &crate::certify::Certificate| c.beta_ci.1);
cert_field!(ft_cert_tau, |c: &crate::certify::Certificate| c.tau_int);
cert_field!(ft_cert_ess, |c: &crate::certify::Certificate| c.ess);
cert_field!(ft_cert_tv, |c: &crate::certify::Certificate| c.tv_exact.unwrap_or(f64::NAN));
cert_field!(ft_cert_floor, |c: &crate::certify::Certificate| c.noise_floor.unwrap_or(f64::NAN));

/// 1 if the run certified clean, 0 if it has findings, and 0 with no certificate present.
#[no_mangle]
pub extern "C" fn ft_cert_passed(sim: *const Sim) -> u32 {
    match unsafe { sim.as_ref() }.and_then(|s| s.cert.as_ref()) {
        Some(c) if c.passed() => 1,
        _ => 0,
    }
}

/// Number of findings. Zero is the only value that means the run is sound.
#[no_mangle]
pub extern "C" fn ft_cert_findings(sim: *const Sim) -> u32 {
    match unsafe { sim.as_ref() }.and_then(|s| s.cert.as_ref()) {
        Some(c) => c.findings.len() as u32,
        None => 0,
    }
}

/// Copy finding `i` into `buf` as UTF-8. Returns the byte length written, or the length needed if
/// `buf` is null, or 0 if there is no such finding.
#[no_mangle]
pub extern "C" fn ft_cert_finding(sim: *const Sim, i: u32, buf: *mut u8, cap: u32) -> u32 {
    let Some(c) = unsafe { sim.as_ref() }.and_then(|s| s.cert.as_ref()) else { return 0 };
    let Some(f) = c.findings.get(i as usize) else { return 0 };
    let text = f.to_string();
    let bytes = text.as_bytes();
    if buf.is_null() {
        return bytes.len() as u32;
    }
    let n = bytes.len().min(cap as usize);
    unsafe { core::ptr::copy_nonoverlapping(bytes.as_ptr(), buf, n) };
    n as u32
}

/// Exact ground energy by variable elimination, or NaN if the induced width exceeds `max_width`.
///
/// This is the oracle that makes a claim checkable on graphs far too large to enumerate.
#[no_mangle]
pub extern "C" fn ft_exact_ground(sim: *const Sim, max_width: u32) -> f64 {
    let Some(s) = (unsafe { sim.as_ref() }) else { return f64::NAN };
    crate::exact::Elimination { max_width: max_width as usize }
        .ground_state(&s.graph)
        .ok()
        .and_then(|e| e.ground_energy)
        .unwrap_or(f64::NAN)
}

/// Exact `log Z` at `beta`, or NaN if too wide.
#[no_mangle]
pub extern "C" fn ft_exact_log_z(sim: *const Sim, beta: f64, max_width: u32) -> f64 {
    let Some(s) = (unsafe { sim.as_ref() }) else { return f64::NAN };
    crate::exact::Elimination { max_width: max_width as usize }
        .log_partition(&s.graph, beta)
        .ok()
        .and_then(|e| e.log_z)
        .unwrap_or(f64::NAN)
}

/// Induced width of the elimination order. Cost of exact inference is `2^width`, so this is the
/// number that decides whether to ask for it at all.
#[no_mangle]
pub extern "C" fn ft_exact_width(sim: *const Sim) -> u32 {
    match unsafe { sim.as_ref() } {
        Some(s) => crate::exact::Elimination::default().width(&s.graph) as u32,
        None => 0,
    }
}

/// Exact ground state by variable elimination, written into `out` as -1/+1.
///
/// Returns 1 on success, 0 on a null handle, a wrong length, or a graph wider than `max_width`.
/// The energy alone is not enough for a caller that has to return a solution rather than a bound.
#[no_mangle]
pub extern "C" fn ft_exact_ground_state(
    sim: *const Sim,
    max_width: u32,
    out: *mut i8,
    len: u32,
) -> u32 {
    let Some(s) = (unsafe { sim.as_ref() }) else { return 0 };
    if out.is_null() || len as usize != s.graph.n {
        return 0;
    }
    let el = crate::exact::Elimination { max_width: max_width as usize };
    match el.ground_state(&s.graph) {
        Ok(e) => match e.ground_state {
            Some(st) => {
                unsafe { core::ptr::copy_nonoverlapping(st.as_ptr(), out, st.len()) };
                1
            }
            None => 0,
        },
        Err(_) => 0,
    }
}

#[cfg(test)]
mod exact_state_ffi {
    use super::*;

    #[test]
    fn the_recovered_state_attains_the_energy() {
        let sim = ft_planted_frustrated(4, 12, 3, 1.0);
        let n = ft_len(sim) as usize;
        let mut out = vec![0i8; n];
        assert_eq!(ft_exact_ground_state(sim, 20, out.as_mut_ptr(), n as u32), 1);
        assert!(out.iter().all(|&v| v == 1 || v == -1));
        assert_eq!(ft_set_spins(sim, out.as_ptr(), n as u32), 1);
        let e = ft_energy(sim);
        assert!((e - ft_exact_ground(sim, 20)).abs() < 1e-9, "state {e} vs energy");
        assert!((e - ft_ground_energy(sim)).abs() < 1e-9, "and it is the planted optimum");
        ft_free(sim);
    }

    #[test]
    fn a_wrong_length_is_refused() {
        let sim = ft_ising2d_new(4, 1.0, 1.0, 1);
        let mut out = vec![0i8; 3];
        assert_eq!(ft_exact_ground_state(sim, 20, out.as_mut_ptr(), 3), 0);
        assert_eq!(ft_exact_ground_state(sim, 20, core::ptr::null_mut(), 16), 0);
        ft_free(sim);
    }
}

// ---- the modelling layer -------------------------------------------------------------------------
//
// Variables are referred to by index across this boundary and names are kept by the caller. A node
// graph already knows what it called each node, and marshalling strings both ways to tell it
// something it knows would be work for nothing.

use crate::model::{Compiled, Constraint, Expr, Lit, Model, Sense, Solution};

/// A model under construction, plus whatever it last compiled and solved.
pub struct ModelHandle {
    model: Model,
    compiled: Option<Compiled>,
    solution: Option<Solution>,
    objective: Expr,
    sense: Sense,
    last_error: String,
    cert: Option<crate::certify::Certificate>,
}

#[no_mangle]
pub extern "C" fn ft_model_new() -> *mut ModelHandle {
    Box::into_raw(Box::new(ModelHandle {
        model: Model::new(),
        compiled: None,
        solution: None,
        objective: Expr::zero(),
        sense: Sense::Minimize,
        last_error: String::new(),
        cert: None,
    }))
}

#[no_mangle]
pub extern "C" fn ft_model_free(m: *mut ModelHandle) {
    if !m.is_null() {
        drop(unsafe { Box::from_raw(m) });
    }
}

/// Declare a `k`-valued variable. Returns its index, or `u32::MAX` on failure.
#[no_mangle]
pub extern "C" fn ft_model_categorical(m: *mut ModelHandle, k: u32) -> u32 {
    let Some(h) = (unsafe { m.as_mut() }) else { return u32::MAX };
    if k < 2 {
        return u32::MAX;
    }
    let n = h.model.len();
    h.model.categorical(&format!("v{n}"), k as usize);
    n as u32
}

/// Declare an integer in `lo..=hi`.
#[no_mangle]
pub extern "C" fn ft_model_integer(m: *mut ModelHandle, lo: i64, hi: i64) -> u32 {
    let Some(h) = (unsafe { m.as_mut() }) else { return u32::MAX };
    if hi <= lo {
        return u32::MAX;
    }
    let n = h.model.len();
    h.model.integer(&format!("v{n}"), lo, hi);
    n as u32
}

/// Declare a 0/1 variable.
#[no_mangle]
pub extern "C" fn ft_model_binary(m: *mut ModelHandle) -> u32 {
    let Some(h) = (unsafe { m.as_mut() }) else { return u32::MAX };
    let n = h.model.len();
    h.model.binary(&format!("v{n}"));
    n as u32
}

fn var_of(h: &ModelHandle, i: u32) -> Option<crate::model::Var> {
    (( i as usize) < h.model.len()).then(|| h.model.var_at(i as usize))
}

/// `a != b`. Returns 1 on success.
#[no_mangle]
pub extern "C" fn ft_model_not_equal(m: *mut ModelHandle, a: u32, b: u32) -> u32 {
    let Some(h) = (unsafe { m.as_mut() }) else { return 0 };
    match (var_of(h, a), var_of(h, b)) {
        (Some(x), Some(y)) if a != b => {
            h.model.constrain(Constraint::NotEqual(x, y));
            1
        }
        _ => 0,
    }
}

/// `a == b`.
#[no_mangle]
pub extern "C" fn ft_model_equal(m: *mut ModelHandle, a: u32, b: u32) -> u32 {
    let Some(h) = (unsafe { m.as_mut() }) else { return 0 };
    match (var_of(h, a), var_of(h, b)) {
        (Some(x), Some(y)) if a != b => {
            h.model.constrain(Constraint::Equal(x, y));
            1
        }
        _ => 0,
    }
}

/// Pin a variable to a value.
#[no_mangle]
pub extern "C" fn ft_model_fix(m: *mut ModelHandle, v: u32, value: u32) -> u32 {
    let Some(h) = (unsafe { m.as_mut() }) else { return 0 };
    match var_of(h, v) {
        Some(x) => {
            h.model.constrain(Constraint::Fix(x, value as usize));
            1
        }
        None => 0,
    }
}

/// Add `coeff · [var == value]` to the objective.
#[no_mangle]
pub extern "C" fn ft_model_objective_term(
    m: *mut ModelHandle,
    maximize: u32,
    coeff: f64,
    v: u32,
    value: u32,
) -> u32 {
    let Some(h) = (unsafe { m.as_mut() }) else { return 0 };
    let Some(x) = var_of(h, v) else { return 0 };
    if !coeff.is_finite() {
        return 0;
    }
    h.sense = if maximize != 0 { Sense::Maximize } else { Sense::Minimize };
    let e = core::mem::take(&mut h.objective);
    h.objective = e.plus(Expr::lit(coeff, Lit::Is(x, value as usize)));
    // Push it into the model as we go, so `ft_model_penalty` answers correctly at any point rather
    // than only after compiling. An editor showing a penalty that is about to change is worse than
    // showing none.
    let (obj, sense) = (h.objective.clone(), h.sense);
    h.model.objective(sense, obj);
    1
}

/// Add `coeff · [a == av] · [b == bv]` to the objective.
#[no_mangle]
pub extern "C" fn ft_model_objective_pair(
    m: *mut ModelHandle,
    maximize: u32,
    coeff: f64,
    a: u32,
    av: u32,
    b: u32,
    bv: u32,
) -> u32 {
    let Some(h) = (unsafe { m.as_mut() }) else { return 0 };
    let (Some(x), Some(y)) = (var_of(h, a), var_of(h, b)) else { return 0 };
    if !coeff.is_finite() || a == b {
        return 0;
    }
    h.sense = if maximize != 0 { Sense::Maximize } else { Sense::Minimize };
    let e = core::mem::take(&mut h.objective);
    h.objective =
        e.plus(Expr::pair(coeff, Lit::Is(x, av as usize), Lit::Is(y, bv as usize)));
    let (obj, sense) = (h.objective.clone(), h.sense);
    h.model.objective(sense, obj);
    1
}

/// Compile. Returns the spin count, or 0 on failure; the reason is available from
/// [`ft_model_error`].
#[no_mangle]
pub extern "C" fn ft_model_compile(m: *mut ModelHandle) -> u32 {
    let Some(h) = (unsafe { m.as_mut() }) else { return 0 };
    let obj = h.objective.clone();
    h.model.objective(h.sense, obj);
    match h.model.compile() {
        Ok(c) => {
            let n = c.spins() as u32;
            h.compiled = Some(c);
            h.last_error.clear();
            n
        }
        Err(e) => {
            h.last_error = e.to_string();
            h.compiled = None;
            0
        }
    }
}

/// Anneal the compiled model, keeping the best of `tries`. Returns 1 on success.
#[no_mangle]
pub extern "C" fn ft_model_solve(m: *mut ModelHandle, tries: u32) -> u32 {
    let Some(h) = (unsafe { m.as_mut() }) else { return 0 };
    let Some(c) = h.compiled.as_ref() else { return 0 };
    h.solution = Some(c.solve_best_of(tries.max(1) as u64));
    1
}

/// The solved value of variable `v`, or `i64::MIN` if it did not decode.
#[no_mangle]
pub extern "C" fn ft_model_value(m: *const ModelHandle, v: u32) -> i64 {
    let Some(h) = (unsafe { m.as_ref() }) else { return i64::MIN };
    let Some(s) = h.solution.as_ref() else { return i64::MIN };
    s.get(&format!("v{v}")).unwrap_or(i64::MIN)
}

/// 1 if every variable decoded.
#[no_mangle]
pub extern "C" fn ft_model_feasible(m: *const ModelHandle) -> u32 {
    match unsafe { m.as_ref() }.and_then(|h| h.solution.as_ref()) {
        Some(s) if s.feasible() => 1,
        _ => 0,
    }
}

/// Energy of the solution.
#[no_mangle]
pub extern "C" fn ft_model_energy(m: *const ModelHandle) -> f64 {
    match unsafe { m.as_ref() }.and_then(|h| h.solution.as_ref()) {
        Some(s) => s.energy,
        None => f64::NAN,
    }
}

/// The penalty actually used, after scaling against the objective.
#[no_mangle]
pub extern "C" fn ft_model_penalty(m: *const ModelHandle) -> f64 {
    match unsafe { m.as_ref() } {
        Some(h) => h.model.effective_penalty(),
        None => f64::NAN,
    }
}

/// Copy the last compile error into `buf`; returns bytes written, or the length needed if null.
#[no_mangle]
pub extern "C" fn ft_model_error(m: *const ModelHandle, buf: *mut u8, cap: u32) -> u32 {
    let Some(h) = (unsafe { m.as_ref() }) else { return 0 };
    let b = h.last_error.as_bytes();
    if buf.is_null() {
        return b.len() as u32;
    }
    let n = b.len().min(cap as usize);
    unsafe { core::ptr::copy_nonoverlapping(b.as_ptr(), buf, n) };
    n as u32
}

/// The compiled program as `.ftp` text; same buffer protocol as [`ft_model_error`].
#[no_mangle]
pub extern "C" fn ft_model_ftp(m: *const ModelHandle, buf: *mut u8, cap: u32) -> u32 {
    let Some(c) = unsafe { m.as_ref() }.and_then(|h| h.compiled.as_ref()) else { return 0 };
    let text = c.program.to_ftp();
    let b = text.as_bytes();
    if buf.is_null() {
        return b.len() as u32;
    }
    let n = b.len().min(cap as usize);
    unsafe { core::ptr::copy_nonoverlapping(b.as_ptr(), buf, n) };
    n as u32
}

#[cfg(test)]
mod model_ffi_tests {
    use super::*;

    fn text(m: *const ModelHandle, f: unsafe extern "C" fn(*const ModelHandle, *mut u8, u32) -> u32) -> String {
        let need = unsafe { f(m, core::ptr::null_mut(), 0) } as usize;
        let mut buf = vec![0u8; need];
        let got = unsafe { f(m, buf.as_mut_ptr(), need as u32) } as usize;
        String::from_utf8_lossy(&buf[..got]).into_owned()
    }

    #[test]
    fn a_colouring_model_goes_through_the_boundary() {
        // The graph editor's whole vocabulary, exercised as it would drive it.
        let m = ft_model_new();
        let a = ft_model_categorical(m, 3);
        let b = ft_model_categorical(m, 3);
        let c = ft_model_categorical(m, 3);
        assert_eq!((a, b, c), (0, 1, 2));
        assert_eq!(ft_model_not_equal(m, a, b), 1);
        assert_eq!(ft_model_not_equal(m, b, c), 1);
        assert_eq!(ft_model_not_equal(m, a, c), 1);

        assert_eq!(ft_model_compile(m), 9, "three one-hot variables of three values");
        assert_eq!(ft_model_solve(m, 12), 1);
        assert_eq!(ft_model_feasible(m), 1);

        let (va, vb, vc) = (ft_model_value(m, a), ft_model_value(m, b), ft_model_value(m, c));
        assert!(va != vb && vb != vc && va != vc, "a triangle needs three colours: {va} {vb} {vc}");
        ft_model_free(m);
    }

    #[test]
    fn an_objective_crosses_and_the_penalty_scales() {
        let m = ft_model_new();
        let x = ft_model_categorical(m, 5);
        for v in 0..5u32 {
            assert_eq!(ft_model_objective_term(m, 1, v as f64, x, v), 1);
        }
        assert_eq!(ft_model_penalty(m), 8.0, "twice the largest coefficient, across the boundary");
        assert!(ft_model_compile(m) > 0);
        ft_model_solve(m, 10);
        assert_eq!(ft_model_value(m, x), 4, "maximising picks the top value");
        ft_model_free(m);
    }

    #[test]
    fn a_compile_error_crosses_as_text() {
        // A graph editor has to show the user why, not just that it failed.
        let m = ft_model_new();
        let x = ft_model_categorical(m, 4);
        assert_eq!(ft_model_objective_term(m, 0, 1.0, x, 99), 1, "accepted at build time");
        assert_eq!(ft_model_compile(m), 0, "and refused at compile time");
        let e = text(m, ft_model_error);
        assert!(e.contains("99") && e.contains("not one of them"), "{e}");
        ft_model_free(m);
    }

    #[test]
    fn the_compiled_program_comes_back_as_ftp() {
        let m = ft_model_new();
        let a = ft_model_categorical(m, 3);
        let b = ft_model_categorical(m, 3);
        ft_model_not_equal(m, a, b);
        ft_model_compile(m);
        let ftp = text(m, ft_model_ftp);
        assert!(ftp.starts_with("ftp 1"));
        assert!(ftp.contains("encode 0 3 onehot"), "the layout travels with it: {ftp}");
        assert!(crate::ftp::Program::from_ftp(&ftp).is_ok());
        ft_model_free(m);
    }

    #[test]
    fn malformed_calls_are_inert() {
        let m = ft_model_new();
        assert_eq!(ft_model_categorical(m, 1), u32::MAX, "k below 2 is a constant");
        assert_eq!(ft_model_integer(m, 5, 5), u32::MAX, "an empty range");
        assert_eq!(ft_model_not_equal(m, 0, 0), 0, "a variable differs from nothing but itself");
        assert_eq!(ft_model_value(m, 0), i64::MIN, "no solution yet");
        assert_eq!(ft_model_categorical(core::ptr::null_mut(), 3), u32::MAX);
        assert_eq!(ft_model_solve(core::ptr::null_mut(), 1), 0);
        ft_model_free(m);
        ft_model_free(core::ptr::null_mut());
    }
}

/// A scratch buffer in wasm memory, for callers with no allocator of their own.
///
/// A browser can read wasm memory but cannot allocate in it, so the two-call text protocol —
/// ask the length, then fill a buffer — needs somewhere to write. This grows on demand and is
/// reused; it is single-threaded like the rest of this ABI, and the caller must copy out before the
/// next call.
#[no_mangle]
pub extern "C" fn ft_scratch(len: u32) -> *mut u8 {
    use std::cell::RefCell;
    thread_local! {
        static BUF: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
    }
    BUF.with(|b| {
        let mut b = b.borrow_mut();
        if b.len() < len as usize {
            b.resize(len as usize, 0);
        }
        b.as_mut_ptr()
    })
}

#[cfg(test)]
mod scratch_tests {
    use super::*;

    #[test]
    fn the_scratch_buffer_grows_and_is_writable() {
        let p = ft_scratch(16);
        assert!(!p.is_null());
        unsafe { core::ptr::write_bytes(p, 0xAB, 16) };
        let big = ft_scratch(4096);
        assert!(!big.is_null());
        unsafe { core::ptr::write_bytes(big, 0x01, 4096) };
    }

    #[test]
    fn text_round_trips_through_the_scratch_protocol() {
        // Exactly how the browser reads a compile error.
        let m = ft_model_new();
        let x = ft_model_categorical(m, 3);
        ft_model_objective_term(m, 0, 1.0, x, 99);
        assert_eq!(ft_model_compile(m), 0);

        let need = ft_model_error(m, core::ptr::null_mut(), 0);
        assert!(need > 0);
        let buf = ft_scratch(need);
        let got = ft_model_error(m, buf, need);
        let s = unsafe { core::slice::from_raw_parts(buf, got as usize) };
        assert!(core::str::from_utf8(s).unwrap().contains("not one of them"));
        ft_model_free(m);
    }
}

/// Exactly `k` of the given variables take `value`.
///
/// Up to four variables, passed positionally with `u32::MAX` for the unused slots — a node graph
/// has a fixed number of ports, and a variadic call across this boundary would need an allocator on
/// the caller's side that a browser does not have.
#[no_mangle]
pub extern "C" fn ft_model_cardinality(
    m: *mut ModelHandle,
    count: u32,
    k: u32,
    value: u32,
    a: u32,
    b: u32,
    c: u32,
    d: u32,
) -> u32 {
    let Some(h) = (unsafe { m.as_mut() }) else { return 0 };
    let mut lits = Vec::new();
    for v in [a, b, c, d].iter().take(count.min(4) as usize) {
        match var_of(h, *v) {
            Some(x) => lits.push(Lit::Is(x, value as usize)),
            None => return 0,
        }
    }
    if lits.len() < 2 || k as usize > lits.len() {
        return 0;
    }
    h.model.constrain(Constraint::Cardinality { lits, k: k as usize });
    1
}

#[cfg(test)]
mod cardinality_ffi {
    use super::*;

    #[test]
    fn exactly_k_crosses_the_boundary() {
        let m = ft_model_new();
        let v: Vec<u32> = (0..4).map(|_| ft_model_binary(m)).collect();
        assert_eq!(ft_model_cardinality(m, 4, 2, 1, v[0], v[1], v[2], v[3]), 1);
        assert!(ft_model_compile(m) > 0);
        ft_model_solve(m, 24);
        assert_eq!(ft_model_feasible(m), 1);
        let on = v.iter().filter(|&&i| ft_model_value(m, i) == 1).count();
        assert_eq!(on, 2, "exactly two should be on");
        ft_model_free(m);
    }

    #[test]
    fn a_degenerate_cardinality_is_refused() {
        let m = ft_model_new();
        let a = ft_model_binary(m);
        let b = ft_model_binary(m);
        assert_eq!(ft_model_cardinality(m, 1, 1, 1, a, u32::MAX, u32::MAX, u32::MAX), 0,
                   "one variable is not a cardinality constraint");
        assert_eq!(ft_model_cardinality(m, 2, 5, 1, a, b, u32::MAX, u32::MAX), 0,
                   "k cannot exceed the number of variables");
        ft_model_free(m);
    }
}

/// Certify a compiled model: sample its energy landscape and check the run.
///
/// The same instrument the rest of the stack uses, reachable from a model rather than from a raw
/// graph. A solved answer says *what*; a certificate says whether the machine that produced it was
/// sampling the distribution it claimed. Returns 1 on success.
#[no_mangle]
pub extern "C" fn ft_model_certify(m: *mut ModelHandle, beta: f64, draws: u32, thin: u32) -> u32 {
    let Some(h) = (unsafe { m.as_mut() }) else { return 0 };
    let Some(c) = h.compiled.as_ref() else { return 0 };
    if draws < 16 || !(beta > 0.0) {
        return 0;
    }
    let g = &c.graph;
    let mut smp = Sampler::new(g, beta, 1);
    smp.sweeps(200, None);
    let mut samples = Vec::with_capacity(draws as usize);
    let mut trace = Vec::with_capacity(draws as usize);
    for _ in 0..draws {
        smp.sweeps(thin.max(1) as usize, None);
        samples.push(smp.s.clone());
        trace.push(g.energy(&smp.s));
    }
    h.cert = Some(crate::certify::certify(g, beta, &samples, &trace));
    1
}

macro_rules! model_cert_field {
    ($name:ident, $f:expr) => {
        #[no_mangle]
        pub extern "C" fn $name(m: *const ModelHandle) -> f64 {
            match unsafe { m.as_ref() }.and_then(|h| h.cert.as_ref()) {
                Some(c) => $f(c),
                None => f64::NAN,
            }
        }
    };
}

model_cert_field!(ft_model_cert_beta, |c: &crate::certify::Certificate| c.beta_eff);
model_cert_field!(ft_model_cert_ess, |c: &crate::certify::Certificate| c.ess);
model_cert_field!(ft_model_cert_tau, |c: &crate::certify::Certificate| c.tau_int);
model_cert_field!(ft_model_cert_tv, |c: &crate::certify::Certificate| c
    .tv_exact
    .unwrap_or(f64::NAN));
model_cert_field!(ft_model_cert_floor, |c: &crate::certify::Certificate| c
    .noise_floor
    .unwrap_or(f64::NAN));

/// Number of findings; zero is the only value meaning the run is sound.
#[no_mangle]
pub extern "C" fn ft_model_cert_findings(m: *const ModelHandle) -> u32 {
    match unsafe { m.as_ref() }.and_then(|h| h.cert.as_ref()) {
        Some(c) => c.findings.len() as u32,
        None => 0,
    }
}

/// Copy finding `i` as UTF-8; same two-call protocol as the other text getters.
#[no_mangle]
pub extern "C" fn ft_model_cert_finding(
    m: *const ModelHandle,
    i: u32,
    buf: *mut u8,
    cap: u32,
) -> u32 {
    let Some(c) = unsafe { m.as_ref() }.and_then(|h| h.cert.as_ref()) else { return 0 };
    let Some(fnd) = c.findings.get(i as usize) else { return 0 };
    let text = fnd.to_string();
    let b = text.as_bytes();
    if buf.is_null() {
        return b.len() as u32;
    }
    let n = b.len().min(cap as usize);
    unsafe { core::ptr::copy_nonoverlapping(b.as_ptr(), buf, n) };
    n as u32
}

#[cfg(test)]
mod model_cert_tests {
    use super::*;

    #[test]
    fn a_compiled_model_can_be_certified() {
        let m = ft_model_new();
        let a = ft_model_categorical(m, 3);
        let b = ft_model_categorical(m, 3);
        ft_model_not_equal(m, a, b);
        assert!(ft_model_compile(m) > 0);
        assert_eq!(ft_model_certify(m, 0.5, 800, 4), 1);
        let beta = ft_model_cert_beta(m);
        assert!((beta - 0.5).abs() < 0.15, "beta_eff {beta} should be near the 0.5 asked for");
        assert!(ft_model_cert_ess(m) > 0.0);
        ft_model_free(m);
    }

    #[test]
    fn certifying_before_compiling_is_refused() {
        let m = ft_model_new();
        ft_model_categorical(m, 3);
        assert_eq!(ft_model_certify(m, 0.5, 800, 1), 0, "nothing compiled yet");
        assert_eq!(ft_model_certify(m, 0.5, 4, 1), 0, "and 4 draws certifies nothing");
        assert!(ft_model_cert_beta(m).is_nan());
        ft_model_free(m);
    }
}
