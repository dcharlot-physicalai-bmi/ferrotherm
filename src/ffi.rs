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
//!
//! # Why `not_unsafe_ptr_arg_deref` is allowed here, and what was checked before allowing it
//!
//! Clippy's `not_unsafe_ptr_arg_deref` is deny-by-default, and these 93 `extern "C"` entry points
//! all trip it. That mattered far more than the lint itself: `cargo clippy --workspace
//! --all-targets` **aborted on this file with exit 101**, so `ferrotherm-gpu`, `-meter`, `-cloud`,
//! `-serve`, `-silicon` and all 21 examples were never linted at all. Suppressing a lint to make a
//! run go green is exactly the move this project distrusts, so the property the lint points at was
//! audited first rather than assumed:
//!
//! - Every handle argument is dereferenced through `as_ref()` / `as_mut()`, which return `Option`
//!   and so are null-checked by construction: 39 `as_mut`, 37 `as_ref`.
//! - Every caller-supplied **out**-pointer is explicitly null-checked before it is written, and a
//!   null buffer is answered with the length the caller needs rather than a write (`ft_ommx_error`,
//!   `ft_model_ftp`, and the rest of the two-call sizing pairs).
//! - Every copy into a caller buffer is clamped to the caller's own capacity with `.min(cap)`
//!   before `copy_nonoverlapping`.
//! - Every slice built from a caller pointer goes through `from_raw_parts` only after a null check
//!   on the same pointer.
//!
//! What remains is the part no Rust signature can fix: a caller may hand over a **non-null dangling**
//! pointer, or a capacity larger than the buffer it owns. Marking these `unsafe fn` would move that
//! obligation onto callers who are C, Python `ctypes`, Julia `ccall`, Zig and JavaScript — none of
//! which have Rust's `unsafe` to move it to. The contract lives in `include/ferrotherm.h`, where
//! those callers can read it.
//!
//! So the allow is scoped to this module, and the lint is enforced everywhere else: CI runs
//! `cargo clippy --workspace --all-targets -D warnings`, which before this comment existed had
//! never linted five of the six published crates.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

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
    /// The last tabu outcome, for [`ft_tabu_iterations`].
    tb: Option<crate::tabu::Outcome>,
    /// The last breakout-local-search outcome, for the `ft_bls_*` accessors.
    bl: Option<crate::bls::Outcome>,
    /// The last planar exact solve, or the reason it was refused.
    pc: Option<Result<crate::planarcut::Outcome, String>>,
    /// The last toroidal bound, for [`ft_toroidal_attained`].
    tor: Option<crate::planarcut::SurfaceBound>,
    /// The last Goemans-Williamson rounding, for [`ft_gw_guaranteed`].
    gw: Option<crate::sdp::Rounding>,
    /// The last cluster-move run, for [`ft_icm_moves`].
    ic: Option<crate::icm::Outcome>,
    /// The last population-annealing outcome, for the `ft_popanneal_*` accessors.
    pa: Option<crate::popanneal::Outcome>,
    /// The last branch-and-bound outcome, for the `ft_branch_*` accessors.
    bb: Option<crate::branch::Outcome>,
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
        Box::into_raw(Box::new(Sim { sampler_state: sampler.s.clone(), graph: g, beta, seed, sweeps_done: 0, ledger: Ledger::default(), gpu: None, ground: None, cert: None, tb: None, bl: None, pc: None, tor: None, gw: None, ic: None, pa: None, bb: None }))
    }
}

/// New 2D nearest-neighbour Ising lattice (periodic), side `l`, coupling `j`.
#[no_mangle]
pub extern "C" fn ft_ising2d_new(l: u32, j: f64, beta: f64, seed: u64) -> *mut Sim {
    Sim::new(lattice2d(l as usize, j), beta, seed)
}

/// Read an `ommx.v1.Instance` and return a simulation over it, or null if it cannot be read.
///
/// The direction that makes this a bridge rather than an exporter: a problem someone else compiled
/// to OMMX -- from jijmodeling, say -- becomes something this sampler can run.
///
/// `constant_out`, when non-null, receives the offset the 0/1 to +/-1 substitution introduces:
/// `ommx_objective(x) == ft_energy(sim) + constant`. Dropping it leaves an energy that ranks states
/// correctly and reports the wrong number.
///
/// On null, [`ft_ommx_error`] says why in the caller's own terms -- a continuous variable, a bound
/// that is not `[0,1]`, an objective of degree three or more. This sampler samples spins, and a
/// bridge that silently dropped what it could not represent would return a model that solves a
/// different problem.
#[no_mangle]
pub extern "C" fn ft_ommx_read(
    bytes: *const u8,
    len: u32,
    beta: f64,
    seed: u64,
    constant_out: *mut f64,
) -> *mut Sim {
    if bytes.is_null() {
        set_ommx_error("no bytes were given");
        return core::ptr::null_mut();
    }
    let raw = unsafe { core::slice::from_raw_parts(bytes, len as usize) };
    match crate::ommx::import(raw) {
        Ok((g, constant)) => {
            set_ommx_error("");
            if !constant_out.is_null() {
                unsafe { *constant_out = constant };
            }
            Sim::new(g, beta, seed)
        }
        Err(e) => {
            set_ommx_error(&e.to_string());
            core::ptr::null_mut()
        }
    }
}

/// Why the last [`ft_ommx_read`] on this thread returned null. Empty when it did not.
#[no_mangle]
pub extern "C" fn ft_ommx_error(buf: *mut u8, cap: u32) -> u32 {
    OMMX_ERROR.with(|e| {
        let e = e.borrow();
        let b = e.as_bytes();
        if buf.is_null() {
            return b.len() as u32;
        }
        let n = b.len().min(cap as usize);
        unsafe { core::ptr::copy_nonoverlapping(b.as_ptr(), buf, n) };
        n as u32
    })
}

thread_local! {
    /// Per-thread, because `ft_ommx_read` is a free function with no handle to hang an error on,
    /// and a global would let one thread's failure explain another thread's success.
    static OMMX_ERROR: core::cell::RefCell<String> = const { core::cell::RefCell::new(String::new()) };
}

fn set_ommx_error(s: &str) {
    OMMX_ERROR.with(|e| *e.borrow_mut() = s.to_string());
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
    // Z1_SPICE always states prices, so the NaN branch is unreachable -- but joules()
    // returns Option now precisely so a caller cannot forget that some devices have none.
    unsafe { sim.as_ref() }.map_or(0.0, |s| s.ledger.joules(&Z1_SPICE).unwrap_or(f64::NAN))
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
    // `!(alpha > 0.0)` rejects NaN and non-positives but ADMITS +inf, which reaches an allocation
    // sized from it and aborts with "capacity overflow" -- a non-unwinding panic across the C ABI.
    // Measured: NaN returned null correctly and +inf killed the process, from one guard. Every
    // other non-finite check in this file uses `is_finite`; this one did not.
    if n < 3 || !alpha.is_finite() || !(alpha > 0.0) {
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
    last_error: String,
    /// Literals accumulating for the next variable-length counting constraint.
    lits: Vec<Lit>,
    cert: Option<crate::certify::Certificate>,
}

#[no_mangle]
pub extern "C" fn ft_model_new() -> *mut ModelHandle {
    Box::into_raw(Box::new(ModelHandle {
        model: Model::new(),
        compiled: None,
        solution: None,
        last_error: String::new(),
        lits: Vec::new(),
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
            h.last_error.clear();
            1
        }
        // The header says "0 on refusal; ft_model_error says why", and this arm said nothing --
        // leaving whatever error happened to be there from an earlier call, which is worse than
        // empty. The two refusals are different mistakes and read differently.
        _ => {
            h.last_error = if a == b {
                format!("'not_equal' needs two DIFFERENT variables; both arguments are variable {a}")
            } else {
                format!(
                    "'not_equal' names variable {}, which is not declared; {} exist",
                    if var_of(h, a).is_none() { a } else { b },
                    h.model.len()
                )
            };
            0
        }
    }
}

/// `a == b`.
#[no_mangle]
pub extern "C" fn ft_model_equal(m: *mut ModelHandle, a: u32, b: u32) -> u32 {
    let Some(h) = (unsafe { m.as_mut() }) else { return 0 };
    match (var_of(h, a), var_of(h, b)) {
        (Some(x), Some(y)) if a != b => {
            h.model.constrain(Constraint::Equal(x, y));
            h.last_error.clear();
            1
        }
        // The header says "0 on refusal; ft_model_error says why", and this arm said nothing --
        // leaving whatever error happened to be there from an earlier call, which is worse than
        // empty. The two refusals are different mistakes and read differently.
        _ => {
            h.last_error = if a == b {
                format!("'equal' needs two DIFFERENT variables; both arguments are variable {a}")
            } else {
                format!(
                    "'equal' names variable {}, which is not declared; {} exist",
                    if var_of(h, a).is_none() { a } else { b },
                    h.model.len()
                )
            };
            0
        }
    }
}

/// Pin a variable to a value.
#[no_mangle]
pub extern "C" fn ft_model_fix(m: *mut ModelHandle, v: u32, value: i64) -> u32 {
    let Some(h) = (unsafe { m.as_mut() }) else { return 0 };
    match var_of(h, v) {
        Some(x) if check_value(h, x, value) => {
            h.model.constrain(Constraint::Fix(x, value));
            h.last_error.clear();
            1
        }
        // ONLY the undeclared case. `check_value` already sets a better message for an
        // out-of-domain value -- it names the variable the caller declared and describes the
        // domain, e.g. "'temperature' takes 10..=20; 3 is not one of them" -- and a first cut of
        // this arm clobbered it with a worse one keyed by handle index. Two existing tests caught
        // that immediately. An audit that reads a function BODY for `last_error` cannot see an
        // error set inside a helper it calls; the tests could.
        _ => {
            if var_of(h, v).is_none() {
                h.last_error = format!(
                    "'fix' names variable {v}, which is not declared; {} exist",
                    h.model.len()
                );
            }
            0
        }
    }
}

/// Add `coeff · [var == value]` to the objective.
#[no_mangle]
pub extern "C" fn ft_model_objective_term(
    m: *mut ModelHandle,
    maximize: u32,
    coeff: f64,
    v: u32,
    value: i64,
) -> u32 {
    let Some(h) = (unsafe { m.as_mut() }) else { return 0 };
    let Some(x) = var_of(h, v) else { return 0 };
    if !coeff.is_finite() || !check_value(h, x, value) {
        return 0;
    }
    // Straight into the model, which accumulates and folds the sense in per term. This used to keep
    // a second copy here and re-push the whole thing under the latest call's sense, so a minimising
    // term arriving after maximising ones re-interpreted every one of them.
    let sense = if maximize != 0 { Sense::Maximize } else { Sense::Minimize };
    h.model.objective(sense, Expr::lit(coeff, Lit::Is(x, value)));
    1
}

/// Add `coeff · [a == av] · [b == bv]` to the objective.
#[no_mangle]
pub extern "C" fn ft_model_objective_pair(
    m: *mut ModelHandle,
    maximize: u32,
    coeff: f64,
    a: u32,
    av: i64,
    b: u32,
    bv: i64,
) -> u32 {
    let Some(h) = (unsafe { m.as_mut() }) else { return 0 };
    let (Some(x), Some(y)) = (var_of(h, a), var_of(h, b)) else { return 0 };
    if !coeff.is_finite() || a == b || !check_value(h, x, av) || !check_value(h, y, bv) {
        return 0;
    }
    let sense = if maximize != 0 { Sense::Maximize } else { Sense::Minimize };
    h.model.objective(sense, Expr::pair(coeff, Lit::Is(x, av), Lit::Is(y, bv)));
    1
}

/// Add `coeff · l₁ · l₂ · … · lₖ` to the objective, over the pending literal list.
///
/// Build the list with [`ft_model_lit`] exactly as for a counting constraint, then close it here
/// instead of with [`ft_model_close`]. The list is cleared either way, so a refused term cannot
/// bleed into the next one.
///
/// Three or more literals is a higher-order term. `ft_model_compile` lowers it with an ancilla spin
/// per substituted pair — see `ferrotherm::reduce` — and the count is not reported through this
/// ABI, so a caller who needs it should compare `ft_model_compile`'s spin count against what the
/// declared variables require.
///
/// A product of one literal is an ordinary linear term and a product of two is
/// [`ft_model_objective_pair`]; both are accepted here so a caller building terms in a loop does
/// not need three code paths.
#[no_mangle]
pub extern "C" fn ft_model_objective_product(
    m: *mut ModelHandle,
    maximize: u32,
    coeff: f64,
) -> u32 {
    let Some(h) = (unsafe { m.as_mut() }) else { return 0 };
    let lits = core::mem::take(&mut h.lits);
    if lits.is_empty() {
        h.last_error = "an objective term needs at least one literal".into();
        return 0;
    }
    if !coeff.is_finite() {
        h.last_error = format!("an objective coefficient must be a real number, not {coeff}");
        return 0;
    }
    let sense = if maximize != 0 { Sense::Maximize } else { Sense::Minimize };
    h.model.objective(sense, Expr::product(coeff, &lits));
    1
}

/// Compile. Returns the spin count, or 0 on failure; the reason is available from
/// [`ft_model_error`].
#[no_mangle]
pub extern "C" fn ft_model_compile(m: *mut ModelHandle) -> u32 {
    let Some(h) = (unsafe { m.as_mut() }) else { return 0 };
    // Nothing to push: every objective term went straight into the model as it arrived, which is
    // also why `ft_model_penalty` answers correctly before compiling rather than only after.
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

/// Solve on a caller's own annealing ladder.
///
/// `beta0` to `beta1` over `stages`, `sweeps` per stage, best of `tries`. Zero for any of the four
/// ladder parameters means "use the default", so a caller can override only what they measured.
/// A harder model wants a longer ladder than the default, and this is how it says so.
#[no_mangle]
pub extern "C" fn ft_model_solve_with(
    m: *mut ModelHandle,
    tries: u32,
    beta0: f64,
    beta1: f64,
    stages: u32,
    sweeps: u32,
) -> u32 {
    let Some(h) = (unsafe { m.as_mut() }) else { return 0 };
    let Some(c) = h.compiled.as_ref() else { return 0 };
    // NaN must be refused, not defaulted. `NaN > 0.0` is false, so a naive "positive means the
    // caller meant it" test would send a NaN quietly down the default path and return an answer
    // computed on a ladder the caller never asked for.
    if beta0.is_nan() || beta1.is_nan() {
        return 0;
    }
    let (dlo, dhi, dn, dw) = crate::model::Compiled::DEFAULT_LADDER;
    let lo = if beta0 > 0.0 { beta0 } else { dlo };
    let hi = if beta1 > 0.0 { beta1 } else { dhi };
    if !lo.is_finite() || !hi.is_finite() || hi <= lo {
        return 0;
    }
    let n = if stages > 0 { stages as usize } else { dn };
    let w = if sweeps > 0 { sweeps as usize } else { dw };
    let sched = crate::schedule::Schedule::geometric(lo, hi, n, w);
    h.solution = Some(c.solve_best_with(&sched, tries.max(1) as u64));
    1
}

/// The solved value of variable `v`, or `i64::MIN` if it did not decode.
#[no_mangle]
pub extern "C" fn ft_model_value(m: *const ModelHandle, v: u32) -> i64 {
    let Some(h) = (unsafe { m.as_ref() }) else { return i64::MIN };
    let Some(s) = h.solution.as_ref() else { return i64::MIN };
    // By the variable's CURRENT name, which `ft_model_name` may have changed. Reconstructing the
    // synthetic `v{index}` here would silently stop finding anything the caller had renamed.
    if (v as usize) >= h.model.len() {
        return i64::MIN;
    }
    let name = h.model.name_of(h.model.var_at(v as usize));
    s.get(name).unwrap_or(i64::MIN)
}

/// 1 if every variable decoded.
#[no_mangle]
pub extern "C" fn ft_model_feasible(m: *const ModelHandle) -> u32 {
    match unsafe { m.as_ref() }.and_then(|h| h.solution.as_ref()) {
        Some(s) if s.feasible() => 1,
        _ => 0,
    }
}

/// Serialise the compiled model as an `ommx.v1.Instance`, the interchange format this corner of the
/// field converged on.
///
/// Same two-call protocol as the text getters, except the payload is BINARY protobuf rather than
/// UTF-8: call with a null buffer for the length, then again with a buffer that size. Returns 0
/// before a successful compile.
///
/// The objective needs no correction: the substitution's constant is written INTO the instance, so
/// `ommx_objective(x) == ferrotherm_energy(s)`. [`ft_model_ommx_constant`] reports that value for
/// inspection and must not be added on top. See [`crate::ommx`].
#[no_mangle]
pub extern "C" fn ft_model_ommx(m: *const ModelHandle, buf: *mut u8, cap: u32) -> u32 {
    let Some(h) = (unsafe { m.as_ref() }) else { return 0 };
    let Some(c) = h.compiled.as_ref() else { return 0 };
    let e = crate::ommx::export(&c.graph);
    if buf.is_null() {
        return e.bytes.len() as u32;
    }
    let n = e.bytes.len().min(cap as usize);
    unsafe { core::ptr::copy_nonoverlapping(e.bytes.as_ptr(), buf, n) };
    n as u32
}

/// The offset the +/-1 to 0/1 substitution produced, ALREADY FOLDED INTO the instance.
/// Read it, do not add it: ommx_objective(x) == ferrotherm_energy(s) exactly, and adding it again double-counts.
/// Reported so the substitution is visible, not because anything downstream must apply it.
#[no_mangle]
pub extern "C" fn ft_model_ommx_constant(m: *const ModelHandle) -> f64 {
    match unsafe { m.as_ref() }.and_then(|h| h.compiled.as_ref()) {
        Some(c) => crate::ommx::export(&c.graph).constant,
        None => 0.0,
    }
}

/// How many compile-time caveats the model carries.
///
/// A caveat is something the compiler KNOWS is wrong with the model and cannot fix: today, an
/// encoding no penalty can make exact. Zero before a successful compile.
#[no_mangle]
pub extern "C" fn ft_model_caveats(m: *const ModelHandle) -> u32 {
    match unsafe { m.as_ref() }.and_then(|h| h.compiled.as_ref()) {
        Some(c) => c.caveats.len() as u32,
        None => 0,
    }
}

/// Copy caveat `i` as UTF-8; same two-call protocol as the other text getters.
#[no_mangle]
pub extern "C" fn ft_model_caveat(
    m: *const ModelHandle,
    i: u32,
    buf: *mut u8,
    cap: u32,
) -> u32 {
    let Some(h) = (unsafe { m.as_ref() }) else { return 0 };
    let Some(c) = h.compiled.as_ref() else { return 0 };
    let Some(text) = c.caveats.get(i as usize) else { return 0 };
    let b = text.as_bytes();
    if buf.is_null() {
        return b.len() as u32;
    }
    let n = b.len().min(cap as usize);
    unsafe { core::ptr::copy_nonoverlapping(b.as_ptr(), buf, n) };
    n as u32
}

/// Spins the higher-order lowering added, or 0 if no term named three or more variables.
///
/// Zero after a failed compile too, so read it beside a non-zero `ft_model_compile`.
#[no_mangle]
pub extern "C" fn ft_model_ancillas(m: *const ModelHandle) -> u32 {
    match unsafe { m.as_ref() }.and_then(|h| h.compiled.as_ref()) {
        Some(c) => c.ancillas as u32,
        None => 0,
    }
}

/// How many constraints the answer breaks.
///
/// Zero when the answer keeps everything it was asked to. Distinct from a variable that did not
/// decode: a broken constraint means every value read cleanly and one of them is not what was
/// asked for, which nothing in the values themselves reveals.
#[no_mangle]
pub extern "C" fn ft_model_violations(m: *const ModelHandle) -> u32 {
    match unsafe { m.as_ref() }.and_then(|h| h.solution.as_ref()) {
        Some(s) => s.violated.len() as u32,
        None => 0,
    }
}

/// Copy violation `i` as UTF-8; same two-call protocol as the other text getters.
#[no_mangle]
pub extern "C" fn ft_model_violation(
    m: *const ModelHandle,
    i: u32,
    buf: *mut u8,
    cap: u32,
) -> u32 {
    let Some(s) = unsafe { m.as_ref() }.and_then(|h| h.solution.as_ref()) else { return 0 };
    let Some(v) = s.violated.get(i as usize) else { return 0 };
    let b = v.detail.as_bytes();
    if buf.is_null() {
        return b.len() as u32;
    }
    let n = b.len().min(cap as usize);
    unsafe { core::ptr::copy_nonoverlapping(b.as_ptr(), buf, n) };
    n as u32
}

/// How far outside constraint `i` the answer sits, in that constraint's own units.
///
/// Places over a ceiling, places under a floor, distance from a fixed value. Always positive; NaN
/// if there is no violation `i`. A description says a constraint broke; this says whether it was a
/// near miss or a rout, which is what a caller ranking repairs or deciding whether a larger penalty
/// would be enough actually needs.
#[no_mangle]
pub extern "C" fn ft_model_violation_amount(m: *const ModelHandle, i: u32) -> f64 {
    match unsafe { m.as_ref() }.and_then(|h| h.solution.as_ref()) {
        Some(s) => s.violated.get(i as usize).map(|v| v.amount).unwrap_or(f64::NAN),
        None => f64::NAN,
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
    fn a_compile_error_crosses_as_text() {
        // A graph editor has to show the user why, not just that it failed.
        let m = ft_model_new();
        assert_eq!(ft_model_compile(m), 0, "a model with nothing in it");
        let e = text(m, ft_model_error);
        assert!(e.contains("no variables"), "{e}");
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
        // Exactly how the browser reads an error from the library.
        let m = ft_model_new();
        let x = ft_model_categorical(m, 3);
        assert_eq!(ft_model_objective_term(m, 0, 1.0, x, 99), 0, "99 is not one of three values");

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
    value: i64,
    a: u32,
    b: u32,
    c: u32,
    d: u32,
) -> u32 {
    counting(m, count, k, value, [a, b, c, d], |lits, k| Constraint::Cardinality { lits, k })
}

/// At most `k` of up to four variables take `value`.
///
/// Costs more spins than the exact form: an inequality needs a slack variable to become an equality
/// the sampler can square. See [`crate::model::Constraint::AtMost`].
#[no_mangle]
pub extern "C" fn ft_model_at_most(
    m: *mut ModelHandle,
    count: u32,
    k: u32,
    value: i64,
    a: u32,
    b: u32,
    c: u32,
    d: u32,
) -> u32 {
    counting(m, count, k, value, [a, b, c, d], |lits, k| Constraint::AtMost { lits, k })
}

/// At least `k` of up to four variables take `value`.
#[no_mangle]
pub extern "C" fn ft_model_at_least(
    m: *mut ModelHandle,
    count: u32,
    k: u32,
    value: i64,
    a: u32,
    b: u32,
    c: u32,
    d: u32,
) -> u32 {
    counting(m, count, k, value, [a, b, c, d], |lits, k| Constraint::AtLeast { lits, k })
}

/// Declare a categorical with a chosen encoding.
///
/// `encoding` is 0 for one-hot, 1 for binary, 2 for domain-wall. The trade is real and worth
/// stating, because it is the difference between a model that fits a machine and one that does not:
///
/// | encoding | spins for `k` values | usable in an objective |
/// |---|---|---|
/// | one-hot | `k` | yes |
/// | domain-wall | `k - 1` | yes |
/// | binary | `ceil(log2 k)` | **no** |
///
/// Only a one-hot or domain-wall indicator is linear in the spins. A binary-encoded variable is
/// cheapest and can appear in constraints alone; putting it in an objective is refused at compile
/// time rather than approximated.
#[no_mangle]
pub extern "C" fn ft_model_categorical_as(m: *mut ModelHandle, k: u32, encoding: u32) -> u32 {
    let Some(h) = (unsafe { m.as_mut() }) else { return u32::MAX };
    let Some(enc) = encoding_of(encoding, h) else { return u32::MAX };
    if k < 2 {
        return u32::MAX;
    }
    let n = h.model.len();
    h.model.categorical_as(&format!("v{n}"), k as usize, enc);
    n as u32
}

/// Declare an integer with a chosen encoding. See [`ft_model_categorical_as`] for the codes.
#[no_mangle]
pub extern "C" fn ft_model_integer_as(
    m: *mut ModelHandle,
    lo: i64,
    hi: i64,
    encoding: u32,
) -> u32 {
    let Some(h) = (unsafe { m.as_mut() }) else { return u32::MAX };
    let Some(enc) = encoding_of(encoding, h) else { return u32::MAX };
    if hi <= lo {
        return u32::MAX;
    }
    let n = h.model.len();
    h.model.integer_as(&format!("v{n}"), lo, hi, enc);
    n as u32
}

fn encoding_of(code: u32, h: &mut ModelHandle) -> Option<crate::encode::Encoding> {
    use crate::encode::Encoding;
    match code {
        0 => Some(Encoding::OneHot),
        1 => Some(Encoding::Binary),
        2 => Some(Encoding::DomainWall),
        other => {
            h.last_error =
                format!("unknown encoding {other}; 0 one-hot, 1 binary, 2 domain-wall");
            None
        }
    }
}

/// Start a fresh list of literals for a counting constraint.
///
/// The positional forms below take four variables and one shared value, which is what a node graph
/// with a fixed number of ports needs and what a scheduling problem does not: "at most two of these
/// nine shifts", or a list whose literals name DIFFERENT values, cannot be said that way at all.
/// Build the list with `ft_model_lit`, then close it with one of the `_n` forms.
///
/// The list lives on the model and is cleared by every `_n` call, so two constraints cannot bleed
/// into each other.
#[no_mangle]
pub extern "C" fn ft_model_lits_clear(m: *mut ModelHandle) -> u32 {
    let Some(h) = (unsafe { m.as_mut() }) else { return 0 };
    h.lits.clear();
    1
}

/// Append a VARIABLE to the pending list, for constraints that are about variables rather than
/// literals -- `all_different` is the only one today.
///
/// It picks a value from the variable's own domain, because the caller has no reason to know one
/// and should not have to. Passing a placeholder through [`ft_model_lit`] instead is what the first
/// version of this did, and it refused every variable whose domain did not happen to contain the
/// placeholder -- correctly, since that function's whole job is to reject a value a variable cannot
/// take. The fix belongs here, where the domain is already known.
#[no_mangle]
pub extern "C" fn ft_model_var(m: *mut ModelHandle, var: u32) -> u32 {
    let Some(h) = (unsafe { m.as_mut() }) else { return 0 };
    if var as usize >= h.model.len() {
        h.last_error = format!("no variable {var}; {} declared", h.model.len());
        return 0;
    }
    let v = h.model.var_at(var as usize);
    let Some(value) = h.model.domain_of(v).values().next() else {
        h.last_error = format!("variable {var} has an empty domain");
        return 0;
    };
    h.lits.push(Lit::Is(v, value));
    1
}

/// Append "`var` takes `value`" to the pending list. Refuses a value the variable cannot take.
#[no_mangle]
pub extern "C" fn ft_model_lit(m: *mut ModelHandle, var: u32, value: i64) -> u32 {
    let Some(h) = (unsafe { m.as_mut() }) else { return 0 };
    match var_of(h, var) {
        Some(x) if check_value(h, x, value) => {
            h.lits.push(Lit::Is(x, value));
            1
        }
        _ => 0,
    }
}

/// How many literals are pending, so a caller can check its own bookkeeping.
#[no_mangle]
pub extern "C" fn ft_model_lits(m: *const ModelHandle) -> u32 {
    match unsafe { m.as_ref() } {
        Some(h) => h.lits.len() as u32,
        None => 0,
    }
}

/// Close the pending list as a counting constraint. See [`ft_model_cardinality`] for the meanings.
///
/// `kind` is 0 for exactly, 1 for at-most, 2 for at-least, 3 for exactly-one, 4 for at-most-one.
/// The last two ignore `k`. Clears the pending list whether it succeeds or not, so a refused
/// constraint cannot silently join the next one.
#[no_mangle]
pub extern "C" fn ft_model_close(m: *mut ModelHandle, kind: u32, k: u32) -> u32 {
    close_counting(m, kind, k, None)
}

fn close_counting(m: *mut ModelHandle, kind: u32, k: u32, soft: Option<f64>) -> u32 {
    let Some(h) = (unsafe { m.as_mut() }) else { return 0 };
    if let Some(w) = soft {
        if !(w > 0.0) || !w.is_finite() {
            h.last_error = format!("a soft constraint needs a positive price, not {w}");
            h.lits.clear();
            return 0;
        }
    }
    let lits = core::mem::take(&mut h.lits);
    if lits.len() < 2 {
        h.last_error = format!(
            "a counting constraint needs at least two literals; {} were given",
            lits.len()
        );
        return 0;
    }
    if kind <= 2 && k as usize > lits.len() {
        h.last_error = format!(
            "k is {k} and only {} literals were given, so the constraint cannot be met",
            lits.len()
        );
        return 0;
    }
    let c = match kind {
        0 => Constraint::Cardinality { lits, k: k as usize },
        1 => Constraint::AtMost { lits, k: k as usize },
        2 => Constraint::AtLeast { lits, k: k as usize },
        3 => Constraint::ExactlyOne(lits),
        4 => Constraint::AtMostOne(lits),
        // 5 reads the VARIABLES out of the pending literals and ignores their values, so
        // all_different needs no second list on any of the eight surfaces. A caller writes
        // ft_model_lit(m, v, 0) per variable and closes with kind 5.
        5 => {
            let mut vars: Vec<crate::model::Var> = Vec::new();
            for l in &lits {
                // Lit::Spin carries no variable to make different from anything, so it is skipped
                // rather than silently treated as one -- an all_different built from spin literals
                // would otherwise constrain fewer variables than the caller listed and still
                // report success.
                if let crate::model::Lit::Is(v, _) = l {
                    if !vars.contains(v) {
                        vars.push(*v);
                    }
                }
            }
            Constraint::AllDifferent(vars)
        }
        other => {
            h.last_error = format!(
                "unknown counting kind {other}; 0 exactly, 1 at-most, 2 at-least, \
                 3 exactly-one, 4 at-most-one"
            );
            return 0;
        }
    };
    match soft {
        Some(w) => h.model.soft(c, w),
        None => h.model.constrain(c),
    };
    1
}

/// Close the pending literal list as a SOFT counting constraint, at a price.
///
/// Same `kind` codes as [`ft_model_close`]. The difference is what breaking it means: a hard
/// constraint says which answers are answers at all, so breaking one makes
/// [`ft_model_feasible`] zero; a soft one is a preference with a number on it, and breaking it
/// costs `weight` and leaves the answer feasible. [`ft_model_soft_cost`] totals what was traded.
///
/// The weight is absolute, not scaled. Automatic scaling exists to stop a hard constraint being
/// outbid by the objective; a soft one is meant to be traded against it.
#[no_mangle]
pub extern "C" fn ft_model_close_soft(
    m: *mut ModelHandle,
    kind: u32,
    k: u32,
    weight: f64,
) -> u32 {
    close_counting(m, kind, k, Some(weight))
}

/// Make the last constraint added a soft one, at `weight`.
///
/// For the pairwise constraints — `not_equal`, `equal`, `fix` — which take their arguments
/// directly rather than through the literal list. Returns 0 if no constraint has been added or the
/// weight is not a positive number.
#[no_mangle]
pub extern "C" fn ft_model_soften_last(m: *mut ModelHandle, weight: f64) -> u32 {
    let Some(h) = (unsafe { m.as_mut() }) else { return 0 };
    if !(weight > 0.0) || !weight.is_finite() {
        h.last_error = format!("a soft constraint needs a positive price, not {weight}");
        return 0;
    }
    if !h.model.soften_last(weight) {
        h.last_error = "there is no constraint to soften yet".into();
        return 0;
    }
    1
}

/// What the broken soft constraints cost. Zero when none broke, or before solving.
#[no_mangle]
pub extern "C" fn ft_model_soft_cost(m: *const ModelHandle) -> f64 {
    match unsafe { m.as_ref() }.and_then(|h| h.solution.as_ref()) {
        Some(s) => s.soft_cost(),
        None => 0.0,
    }
}

/// 1 if violation `i` is a hard one, 0 if it is a preference that was traded away.
#[no_mangle]
pub extern "C" fn ft_model_violation_is_hard(m: *const ModelHandle, i: u32) -> u32 {
    match unsafe { m.as_ref() }.and_then(|h| h.solution.as_ref()) {
        Some(s) => s.violated.get(i as usize).map(|v| v.hard as u32).unwrap_or(1),
        None => 1,
    }
}

/// Use exactly this penalty, disabling the automatic scaling.
///
/// By default the penalty rises to twice the largest objective coefficient, because a constraint
/// that merely ties with the objective gets traded away. When `feasible` comes back 0 the remedy is
/// to raise it, and until now the C surface -- and so Python, Zig, Julia and the editor -- had no
/// way to. A non-finite or non-positive value is refused.
#[no_mangle]
pub extern "C" fn ft_model_fixed_penalty(m: *mut ModelHandle, p: f64) -> u32 {
    let Some(h) = (unsafe { m.as_mut() }) else { return 0 };
    if !p.is_finite() || p <= 0.0 {
        h.last_error = format!("a penalty must be a positive number, not {p}");
        return 0;
    }
    h.model.fixed_penalty(p);
    1
}

/// Give a variable the caller's own name, so errors and answers use it.
///
/// Optional: a variable declared without one is called `v0`, `v1` and so on. Returns 1 on success,
/// 0 if the index is unknown or the bytes are not UTF-8. `len` is a byte count, not a terminator.
#[no_mangle]
pub extern "C" fn ft_model_name(m: *mut ModelHandle, v: u32, name: *const u8, len: u32) -> u32 {
    let Some(h) = (unsafe { m.as_mut() }) else { return 0 };
    let Some(x) = var_of(h, v) else { return 0 };
    if name.is_null() {
        return 0;
    }
    let bytes = unsafe { core::slice::from_raw_parts(name, len as usize) };
    let Ok(s) = core::str::from_utf8(bytes) else {
        h.last_error = "a variable name must be UTF-8".into();
        return 0;
    };
    // Refused here rather than at compile, because a caller naming variables in a loop wants to
    // learn about the collision at the call that caused it. An answer is keyed by name, so the
    // second of two identical names does not shadow the first -- it replaces it.
    let clash = (0..h.model.len())
        .map(|i| h.model.var_at(i))
        .any(|v| v != x && h.model.name_of(v) == s);
    if clash {
        h.last_error = format!("'{s}' is already the name of another variable");
        return 0;
    }
    h.model.rename(x, s);
    1
}

/// Reject a value the variable cannot take, at the call that wrote it.
///
/// The compiler catches this too, but by then the caller is several statements away from the
/// mistake and holds only "the model did not compile". A C caller has no stack trace to fall back
/// on, so the error has to arrive while the call that caused it is still the current one.
fn check_value(h: &mut ModelHandle, var: crate::model::Var, value: i64) -> bool {
    let d = h.model.domain_of(var);
    if d.index_of(value).is_some() {
        return true;
    }
    h.last_error = format!(
        "'{}' takes {}; {value} is not one of them",
        h.model.name_of(var),
        d.describe()
    );
    false
}

/// The shared body of the three counting constraints, which differ only in the comparison.
fn counting(
    m: *mut ModelHandle,
    count: u32,
    k: u32,
    value: i64,
    vars: [u32; 4],
    build: impl FnOnce(Vec<Lit>, usize) -> Constraint,
) -> u32 {
    let Some(h) = (unsafe { m.as_mut() }) else { return 0 };
    let mut lits = Vec::new();
    for v in vars.iter().take(count.min(4) as usize) {
        match var_of(h, *v) {
            Some(x) if check_value(h, x, value) => lits.push(Lit::Is(x, value)),
            _ => return 0,
        }
    }
    if lits.len() < 2 || k as usize > lits.len() {
        return 0;
    }
    h.model.constrain(build(lits, k as usize));
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
    fn an_encoding_can_be_chosen_and_costs_what_it_says() {
        // The trade is the difference between a model that fits a machine and one that does not,
        // and it was reachable from Rust alone until now.
        // No constraint on the variable: a BINARY-encoded one cannot appear in a literal at all,
        // so fixing it -- as this first did -- measures whether it compiles rather than what it
        // costs, and reported 0 for the encoding with the smallest cost of the three.
        let spins_for = |enc: u32| {
            let m = ft_model_new();
            let v = ft_model_categorical_as(m, 8, enc);
            assert_ne!(v, u32::MAX, "encoding {enc} should be accepted");
            let n = ft_model_compile(m);
            ft_model_free(m);
            n
        };
        assert_eq!(spins_for(0), 8, "one-hot: one spin per value");
        assert_eq!(spins_for(2), 7, "domain-wall: one fewer");
        assert_eq!(spins_for(1), 3, "binary: log2 of the domain, and the cheapest by far");

        // and the two that CAN carry a literal both do
        for enc in [0u32, 2] {
            let m = ft_model_new();
            let v = ft_model_categorical_as(m, 8, enc);
            assert_eq!(ft_model_fix(m, v, 3), 1);
            assert!(ft_model_compile(m) > 0, "encoding {enc} must work in a constraint");
            assert_eq!(ft_model_solve(m, 16), 1);
            assert_eq!(ft_model_value(m, v), 3, "encoding {enc} decodes to what it was fixed to");
            ft_model_free(m);
        }

        let m = ft_model_new();
        assert_eq!(ft_model_categorical_as(m, 8, 9), u32::MAX, "an unknown encoding is refused");
        let mut buf = [0u8; 256];
        let n = ft_model_error(m, buf.as_mut_ptr(), buf.len() as u32) as usize;
        let e = core::str::from_utf8(&buf[..n]).unwrap();
        assert!(e.contains("domain-wall"), "and lists the ones it knows: {e}");
        ft_model_free(m);
    }

    #[test]
    fn a_binary_encoded_variable_cannot_appear_in_an_objective() {
        // Only a one-hot or domain-wall indicator is linear in the spins. A binary one is not, and
        // approximating it would answer a different question -- so it is refused at compile time.
        let m = ft_model_new();
        let v = ft_model_categorical_as(m, 8, 1);
        ft_model_objective_term(m, 1, 1.0, v, 3);
        assert_eq!(ft_model_compile(m), 0, "a binary variable in an objective must not compile");
        let mut buf = [0u8; 512];
        let n = ft_model_error(m, buf.as_mut_ptr(), buf.len() as u32) as usize;
        let e = core::str::from_utf8(&buf[..n]).unwrap();
        assert!(e.contains("OneHot") || e.contains("one-hot"), "{e}");
        ft_model_free(m);
    }

    #[test]
    fn a_soft_constraint_crosses_the_c_abi_as_a_price() {
        // Both would rather be on shift 0; the clash is priced. Cheap, they take it and the answer
        // is still feasible; dear, they do not.
        let run = |price: f64| {
            let m = ft_model_new();
            let a = ft_model_categorical(m, 2);
            let b = ft_model_categorical(m, 2);
            ft_model_not_equal(m, a, b);
            assert_eq!(ft_model_soften_last(m, price), 1);
            ft_model_objective_term(m, 1, 5.0, a, 0);
            ft_model_objective_term(m, 1, 5.0, b, 0);
            assert!(ft_model_compile(m) > 0);
            assert_eq!(ft_model_solve(m, 24), 1);
            let out = (
                ft_model_value(m, a),
                ft_model_value(m, b),
                ft_model_feasible(m),
                ft_model_soft_cost(m),
                ft_model_violations(m),
            );
            ft_model_free(m);
            out
        };

        let (a, b, feasible, cost, n) = run(1.0);
        assert_eq!((a, b), (0, 0), "a cheap clash is worth having");
        assert_eq!(feasible, 1, "and a soft violation is not an infeasible answer");
        assert_eq!(cost, 1.0);
        assert_eq!(n, 1, "it is still reported");

        let (a, b, _, cost, _) = run(50.0);
        assert_ne!(a, b, "a dear one is not");
        assert_eq!(cost, 0.0);
    }

    #[test]
    fn hard_and_soft_are_distinguishable_over_the_abi() {
        let m = ft_model_new();
        let a = ft_model_categorical(m, 2);
        let b = ft_model_categorical(m, 2);
        ft_model_not_equal(m, a, b);
        ft_model_fixed_penalty(m, 1.0);            // hard, and deliberately outbid
        ft_model_objective_term(m, 1, 40.0, a, 0);
        ft_model_objective_term(m, 1, 40.0, b, 0);
        assert!(ft_model_compile(m) > 0);
        ft_model_solve(m, 16);
        assert_eq!(ft_model_feasible(m), 0, "a broken hard constraint is infeasible");
        assert_eq!(ft_model_violation_is_hard(m, 0), 1);
        assert_eq!(ft_model_soft_cost(m), 0.0, "a hard constraint has no price");
        ft_model_free(m);
    }

    #[test]
    fn a_soft_counting_constraint_and_a_bad_price_are_both_handled() {
        let m = ft_model_new();
        let v: Vec<u32> = (0..4).map(|_| ft_model_binary(m)).collect();
        for &i in &v {
            ft_model_lit(m, i, 1);
            // Worth more than the clash costs. The penalty is SQUARED, so taking all four is
            // priced at 1·(4-2)² = 4 against 1·(3-2)² = 1 for taking three: a reward of 3 apiece
            // makes those exactly equal, which is a tie rather than a test.
            ft_model_objective_term(m, 1, 4.0, i, 1);
        }
        // "prefer at most two", priced below what taking the other two is worth
        assert_eq!(ft_model_close_soft(m, 1, 2, 1.0), 1);
        assert!(ft_model_compile(m) > 0);
        ft_model_solve(m, 24);
        assert_eq!(v.iter().filter(|&&i| ft_model_value(m, i) == 1).count(), 4, "all four taken");
        assert_eq!(ft_model_feasible(m), 1, "and the answer is still an answer");
        assert_eq!(ft_model_violation_is_hard(m, 0), 0, "the violation is a traded preference");
        assert!(ft_model_soft_cost(m) > 0.0);
        ft_model_free(m);

        let m = ft_model_new();
        let a = ft_model_binary(m);
        let b = ft_model_binary(m);
        ft_model_lit(m, a, 1);
        ft_model_lit(m, b, 1);
        assert_eq!(ft_model_close_soft(m, 1, 1, 0.0), 0, "a price must be positive");
        assert_eq!(ft_model_lits(m), 0, "and a refused constraint clears the list");
        assert_eq!(ft_model_soften_last(m, 1.0), 0, "with nothing to soften");
        ft_model_free(m);
    }


    #[test]
    fn a_higher_order_objective_term_crosses_the_c_abi() {
        // Three literals, built with the same list machinery a counting constraint uses. The whole
        // point is that a C caller can express "these three together" at all -- there was no way
        // to, since the ABI offered one literal or two and nothing else.
        let m = ft_model_new();
        let v: Vec<u32> = (0..3).map(|_| ft_model_categorical(m, 3)).collect();
        for &i in &v {
            assert_eq!(ft_model_lit(m, i, 2), 1);
        }
        assert_eq!(ft_model_objective_product(m, 1, 9.0), 1);
        assert_eq!(ft_model_lits(m), 0, "closing clears the list");

        let spins = ft_model_compile(m);
        assert!(spins > 9, "three categoricals are 9 spins; the ancilla makes it more: {spins}");
        assert_eq!(ft_model_solve(m, 24), 1);
        for &i in &v {
            assert_eq!(ft_model_value(m, i), 2, "the reward is only paid when all three hold");
        }
        ft_model_free(m);
    }

    #[test]
    fn an_objective_product_refuses_what_it_cannot_mean() {
        let m = ft_model_new();
        let x = ft_model_categorical(m, 3);
        assert_eq!(ft_model_objective_product(m, 1, 1.0), 0, "no literals is not a term");
        ft_model_lit(m, x, 1);
        assert_eq!(ft_model_objective_product(m, 1, f64::NAN), 0, "NaN is not a coefficient");
        assert_eq!(ft_model_lits(m), 0, "and a refused term does not bleed into the next");
        ft_model_free(m);
    }

    #[test]
    fn a_counting_constraint_can_be_any_length_and_name_different_values() {
        // Nine shifts, at most two of them taken. The positional form tops out at four, so this
        // could not be said through the C ABI at all.
        let m = ft_model_new();
        let v: Vec<u32> = (0..9).map(|_| ft_model_binary(m)).collect();
        for &i in &v {
            assert_eq!(ft_model_lit(m, i, 1), 1);
            ft_model_objective_term(m, 1, 1.0, i, 1); // reward taking every one
        }
        assert_eq!(ft_model_lits(m), 9);
        assert_eq!(ft_model_close(m, 1, 2), 1, "at most 2 of nine");
        assert_eq!(ft_model_lits(m), 0, "closing clears the list");
        assert!(ft_model_compile(m) > 0);
        assert_eq!(ft_model_solve(m, 24), 1);
        assert_eq!(ft_model_feasible(m), 1);
        assert_eq!(v.iter().filter(|&&i| ft_model_value(m, i) == 1).count(), 2);
        ft_model_free(m);

        // and the literals may name DIFFERENT values, which the shared-value form cannot express
        let m = ft_model_new();
        let a = ft_model_categorical(m, 4);
        let b = ft_model_integer(m, 10, 20);
        ft_model_lit(m, a, 3);
        ft_model_lit(m, b, 17);
        assert_eq!(ft_model_close(m, 0, 2), 1, "exactly both");
        assert!(ft_model_compile(m) > 0);
        assert_eq!(ft_model_solve(m, 16), 1);
        assert_eq!((ft_model_value(m, a), ft_model_value(m, b)), (3, 17));
        ft_model_free(m);
    }

    #[test]
    fn exactly_one_and_at_most_one_are_reachable() {
        for (kind, want) in [(3u32, 1usize), (4u32, 0usize)] {
            let m = ft_model_new();
            let v: Vec<u32> = (0..5).map(|_| ft_model_binary(m)).collect();
            for &i in &v {
                ft_model_lit(m, i, 1);
                // push everything OFF, so at-most-one takes none and exactly-one still takes one
                ft_model_objective_term(m, 0, 1.0, i, 1);
            }
            assert_eq!(ft_model_close(m, kind, 0), 1);
            assert!(ft_model_compile(m) > 0);
            assert_eq!(ft_model_solve(m, 24), 1);
            let on = v.iter().filter(|&&i| ft_model_value(m, i) == 1).count();
            assert_eq!(on, want, "kind {kind}");
            assert_eq!(ft_model_feasible(m), 1);
            ft_model_free(m);
        }
    }

    #[test]
    fn a_refused_counting_constraint_does_not_bleed_into_the_next() {
        let m = ft_model_new();
        let a = ft_model_binary(m);
        let b = ft_model_binary(m);
        ft_model_lit(m, a, 1);
        assert_eq!(ft_model_close(m, 1, 1), 0, "one literal is not a counting constraint");
        assert_eq!(ft_model_lits(m), 0, "and the list is cleared even so");

        ft_model_lit(m, a, 1);
        ft_model_lit(m, b, 1);
        assert_eq!(ft_model_close(m, 0, 5), 0, "k cannot exceed the literal count");
        assert_eq!(ft_model_lits(m), 0);
        assert_eq!(ft_model_close(m, 9, 1), 0, "and an unknown kind is refused by name");

        // a bad literal is refused at the push, not silently carried
        assert_eq!(ft_model_lit(m, 99, 1), 0, "no such variable");
        let t = ft_model_integer(m, 10, 20);
        assert_eq!(ft_model_lit(m, t, 3), 0, "3 is not a temperature in 10..=20");
        ft_model_free(m);
    }

    #[test]
    fn a_penalty_can_be_raised_when_a_constraint_loses() {
        // The remedy the error text recommends, which the C surface could not perform. A constraint
        // against an objective ten times its weight loses; raising the penalty wins it back.
        let build = |p: f64| {
            let m = ft_model_new();
            let a = ft_model_categorical(m, 3);
            let b = ft_model_categorical(m, 3);
            ft_model_not_equal(m, a, b);
            // both want value 1, hard
            ft_model_objective_term(m, 1, 40.0, a, 1);
            ft_model_objective_term(m, 1, 40.0, b, 1);
            if p > 0.0 {
                assert_eq!(ft_model_fixed_penalty(m, p), 1);
            }
            assert!(ft_model_compile(m) > 0);
            ft_model_solve(m, 16);
            let out = (ft_model_feasible(m), ft_model_value(m, a), ft_model_value(m, b));
            ft_model_free(m);
            out
        };
        // pinned low, the constraint is outbid and both take 1
        let (_, a, b) = build(1.0);
        assert_eq!((a, b), (1, 1), "a penalty of 1 against a weight of 40 loses, as it should");
        // raised, it holds
        let (feasible, a, b) = build(200.0);
        assert_eq!(feasible, 1);
        assert_ne!(a, b, "a raised penalty wins the constraint back");

        // and a penalty that is not a positive number is refused
        let m = ft_model_new();
        assert_eq!(ft_model_fixed_penalty(m, 0.0), 0);
        assert_eq!(ft_model_fixed_penalty(m, -1.0), 0);
        assert_eq!(ft_model_fixed_penalty(m, f64::NAN), 0);
        ft_model_free(m);
    }

    #[test]
    fn objective_terms_accumulate_and_a_later_sense_does_not_rewrite_earlier_ones() {
        // The C ABI takes a maximize flag PER CALL. It used to write that flag onto the whole
        // accumulated objective and re-push it, so one minimising term arriving after three
        // maximising ones inverted all four -- silently, with feasible still true.
        let m = ft_model_new();
        let v: Vec<u32> = (0..4).map(|_| ft_model_binary(m)).collect();
        for &i in &v[..3] {
            assert_eq!(ft_model_objective_term(m, 1, 1.0, i, 1), 1); // maximise: want these ON
        }
        assert_eq!(ft_model_objective_term(m, 0, 1.0, v[3], 1), 1); // minimise: want this OFF
        assert!(ft_model_compile(m) > 0);
        assert_eq!(ft_model_solve(m, 16), 1);
        let on: Vec<usize> = (0..4).filter(|&i| ft_model_value(m, v[i]) == 1).collect();
        assert_eq!(on, vec![0, 1, 2], "three rewarded, one penalised, and no flipping");
        ft_model_free(m);
    }

    #[test]
    fn every_objective_term_survives_to_the_answer() {
        // And each term counts once. Three separate calls used to be pushed into the model three
        // times over, each call re-adding everything before it.
        let m = ft_model_new();
        let x = ft_model_categorical(m, 4);
        // 1 to value 1, 2 to value 2, 3 to value 3: the largest must win, and would not if the
        // earlier terms were re-added on top of it.
        for value in 1..4i64 {
            assert_eq!(ft_model_objective_term(m, 1, value as f64, x, value), 1);
        }
        assert!(ft_model_compile(m) > 0);
        assert_eq!(ft_model_solve(m, 16), 1);
        assert_eq!(ft_model_value(m, x), 3);
        ft_model_free(m);
    }

    #[test]
    fn a_name_already_taken_is_refused_at_the_call() {
        let m = ft_model_new();
        let a = ft_model_binary(m);
        let b = ft_model_binary(m);
        let n = "shift";
        assert_eq!(ft_model_name(m, a, n.as_ptr(), n.len() as u32), 1);
        assert_eq!(ft_model_name(m, b, n.as_ptr(), n.len() as u32), 0, "already taken");
        let mut buf = [0u8; 256];
        let k = ft_model_error(m, buf.as_mut_ptr(), buf.len() as u32) as usize;
        let e = core::str::from_utf8(&buf[..k]).unwrap();
        assert!(e.contains("'shift' is already"), "{e}");

        // renaming a variable to its OWN name is not a collision
        assert_eq!(ft_model_name(m, a, n.as_ptr(), n.len() as u32), 1);
        ft_model_free(m);
    }

    #[test]
    fn a_renamed_variable_can_still_be_read_back() {
        // The answer is keyed by name, so renaming a variable and then reading it by index has to
        // keep working. It did not: the reader rebuilt the synthetic name and found nothing.
        let m = ft_model_new();
        let x = ft_model_categorical(m, 3);
        let n = "west";
        assert_eq!(ft_model_name(m, x, n.as_ptr(), n.len() as u32), 1);
        ft_model_fix(m, x, 2);
        assert!(ft_model_compile(m) > 0);
        assert_eq!(ft_model_solve(m, 4), 1);
        assert_eq!(ft_model_value(m, x), 2, "a renamed variable still reads back by index");
        assert_eq!(ft_model_value(m, 99), i64::MIN, "and an index that does not exist does not");
        ft_model_free(m);
    }

    #[test]
    fn a_name_pushed_down_shows_up_in_the_error() {
        let m = ft_model_new();
        let t = ft_model_integer(m, 10, 20);
        let n = "temperature";
        assert_eq!(ft_model_name(m, t, n.as_ptr(), n.len() as u32), 1);
        assert_eq!(ft_model_fix(m, t, 3), 0);
        let mut buf = [0u8; 256];
        let n = ft_model_error(m, buf.as_mut_ptr(), buf.len() as u32) as usize;
        let e = core::str::from_utf8(&buf[..n]).unwrap();
        assert!(e.contains("'temperature'"), "should name the variable the caller knows: {e}");
        assert!(!e.contains("v0"), "and not the handle they never saw: {e}");
        ft_model_free(m);
    }

    #[test]
    fn a_value_outside_the_domain_is_refused_at_the_call_that_wrote_it() {
        // Not at compile time, several statements later, holding only "it did not compile". A C
        // caller has no stack to fall back on.
        let m = ft_model_new();
        let t = ft_model_integer(m, 10, 20);
        assert_eq!(ft_model_fix(m, t, 3), 0, "3 is a slot, not a temperature in 10..=20");
        let mut buf = [0u8; 256];
        let n = ft_model_error(m, buf.as_mut_ptr(), buf.len() as u32) as usize;
        let e = core::str::from_utf8(&buf[..n]).unwrap();
        assert!(e.contains("10..=20") && e.contains("3 is not"), "{e}");

        assert_eq!(ft_model_fix(m, t, 13), 1, "13 is one");
        assert_eq!(ft_model_objective_term(m, 1, 1.0, t, 99), 0, "and 99 is not");
        ft_model_free(m);
    }

    #[test]
    fn a_caller_supplied_ladder_is_used_and_a_bad_one_refused() {
        let m = ft_model_new();
        let a = ft_model_categorical(m, 3);
        let b = ft_model_categorical(m, 3);
        ft_model_not_equal(m, a, b);
        assert!(ft_model_compile(m) > 0);
        assert_eq!(ft_model_solve_with(m, 4, 0.05, 6.0, 60, 20), 1);
        assert_eq!(ft_model_feasible(m), 1);
        assert_ne!(ft_model_value(m, a), ft_model_value(m, b));
        // zeros mean "default", so a caller can override only what they measured
        assert_eq!(ft_model_solve_with(m, 4, 0.0, 0.0, 0, 0), 1);
        assert_eq!(ft_model_feasible(m), 1);
        // a ladder that runs backwards is not a ladder
        assert_eq!(ft_model_solve_with(m, 4, 8.0, 0.05, 60, 20), 0, "hot-to-cold only");
        assert_eq!(ft_model_solve_with(m, 4, f64::NAN, 6.0, 60, 20), 0, "NaN is not a temperature");
        ft_model_free(m);
    }

    #[test]
    fn ffi_inequalities_bound_without_forcing() {
        // The distinction the C surface has to preserve: at_most 2 permits fewer than two, where
        // an exact cardinality would not. Rewarding every variable proves the ceiling still binds.
        let m = ft_model_new();
        let v: Vec<u32> = (0..4).map(|_| ft_model_binary(m)).collect();
        assert_eq!(ft_model_at_most(m, 4, 2, 1, v[0], v[1], v[2], v[3]), 1);
        for &i in &v {
            ft_model_objective_term(m, 1, 1.0, i, 1);
        }
        assert!(ft_model_compile(m) > 0);
        ft_model_solve(m, 24);
        assert_eq!(ft_model_feasible(m), 1);
        let on = v.iter().filter(|&&i| ft_model_value(m, i) == 1).count();
        assert_eq!(on, 2, "the ceiling binds against a reward pushing past it");
        ft_model_free(m);

        // and at_least holds a floor against a reward pushing the other way
        let m = ft_model_new();
        let v: Vec<u32> = (0..4).map(|_| ft_model_binary(m)).collect();
        assert_eq!(ft_model_at_least(m, 4, 3, 1, v[0], v[1], v[2], v[3]), 1);
        for &i in &v {
            ft_model_objective_term(m, 0, 1.0, i, 1);
        }
        assert!(ft_model_compile(m) > 0);
        ft_model_solve(m, 24);
        let on = v.iter().filter(|&&i| ft_model_value(m, i) == 1).count();
        assert_eq!(on, 3, "the floor holds against a reward pushing below it");
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

// ---- solvers and bounds ------------------------------------------------------------------------
//
// A GAP THAT HAD BEEN OPEN SINCE `bound` LANDED: the C ABI could build a graph and sample it, but
// could not ask how far from optimal the sample was. Optimality-gap certificates are the headline
// claim in this crate's README, and until now they were reachable from exactly one of the six
// surfaces. `check-parity.sh` exists to catch a capability that stops at Rust, and it did not,
// because a symbol that was never exported is not a parity failure -- it is a thing nobody can say.
//
// Each solver leaves its best state as the simulation's state, so `ft_spins` reads the answer and
// `ft_energy` recomputes the energy from it rather than trusting the number returned here. That
// also makes them compose: anneal, then tabu from where annealing stopped, then branch and bound
// with that as its incumbent.

/// Tabu search. Returns the energy of the best state found, or NaN on a null handle.
///
/// `tenure = 0` means "scale to the graph", matching [`crate::tabu::Params`]; `restart_after = 0`
/// means never restart.
#[no_mangle]
pub extern "C" fn ft_tabu(sim: *mut Sim, iterations: u32, tenure: u32, restart_after: u32) -> f64 {
    let Some(s) = (unsafe { sim.as_mut() }) else { return f64::NAN };
    let p = crate::tabu::Params {
        iterations: iterations.max(1) as usize,
        tenure: tenure as usize,
        restart_after: (restart_after > 0).then_some(restart_after as usize),
    };
    let out = crate::tabu::search_metered(&s.graph, &p, s.seed, Some(&mut s.ledger));
    if out.state.len() == s.sampler_state.len() {
        s.sampler_state.copy_from_slice(&out.state);
    }
    let e = out.energy;
    s.tb = Some(out);
    e
}

/// Iterations tabu actually ran, which is not always the budget it was given.
///
/// Exported rather than left implicit because truncation is invisible from outside otherwise --
/// the defect that shipped in the first version of that module, where a run that spent 9 of 50,000
/// iterations returned a result shaped exactly like a completed one.
#[no_mangle]
pub extern "C" fn ft_tabu_iterations(sim: *const Sim) -> u64 {
    unsafe { sim.as_ref() }.and_then(|s| s.tb.as_ref()).map_or(0, |o| o.iterations_run as u64)
}

/// Population annealing. Returns the best energy found, or NaN on a null handle.
///
/// The ladder is linear from `β = 0` to `beta_max` in `stages` steps, which is what makes
/// [`ft_popanneal_ln_z`] an absolute free energy rather than a ratio.
#[no_mangle]
pub extern "C" fn ft_popanneal(
    sim: *mut Sim,
    population: u32,
    sweeps: u32,
    beta_max: f64,
    stages: u32,
) -> f64 {
    let Some(s) = (unsafe { sim.as_mut() }) else { return f64::NAN };
    if !beta_max.is_finite() || beta_max < 0.0 {
        return f64::NAN;
    }
    let p = crate::popanneal::Params::linear_from_zero(
        population.max(1) as usize,
        sweeps.max(1) as usize,
        beta_max,
        stages.max(1) as usize,
    );
    let out = crate::popanneal::run(&s.graph, &p, s.seed);
    if out.state.len() == s.sampler_state.len() {
        s.sampler_state.copy_from_slice(&out.state);
    }
    let e = out.energy;
    s.pa = Some(out);
    e
}

/// `ln Z` at the final β from the last [`ft_popanneal`], or NaN if there was none.
#[no_mangle]
pub extern "C" fn ft_popanneal_ln_z(sim: *const Sim) -> f64 {
    match unsafe { sim.as_ref() }.and_then(|s| s.pa.as_ref()) {
        Some(o) if o.ln_z_is_absolute => o.ln_z,
        _ => f64::NAN,
    }
}

/// The worst family statistic `ρ` over the ladder — **the number that says whether to believe
/// [`ft_popanneal_ln_z`]**.
///
/// `1.0` means every ancestor still has one descendant; the population size means the population
/// collapsed onto a single ancestor and explored one basin with N copies of one history. NaN if no
/// run has happened.
#[no_mangle]
pub extern "C" fn ft_popanneal_rho(sim: *const Sim) -> f64 {
    match unsafe { sim.as_ref() }.and_then(|s| s.pa.as_ref()) {
        Some(o) => o.rho_max,
        None => f64::NAN,
    }
}

/// Branch and bound, starting from this simulation's current state as its incumbent.
///
/// Returns the lowest energy found. **Whether it is the minimum is a separate question**, answered
/// by [`ft_branch_proved`]: a run that exhausted its node budget returns the best it saw and says
/// the proof is missing.
#[no_mangle]
pub extern "C" fn ft_branch(sim: *mut Sim, max_nodes: u64) -> f64 {
    let Some(s) = (unsafe { sim.as_mut() }) else { return f64::NAN };
    let p = crate::branch::Params {
        max_nodes: max_nodes.max(1),
        incumbent: Some(s.sampler_state.clone()),
        ..crate::branch::Params::default()
    };
    let out = crate::branch::solve(&s.graph, &p);
    if out.state.len() == s.sampler_state.len() {
        s.sampler_state.copy_from_slice(&out.state);
    }
    let e = out.energy;
    s.bb = Some(out);
    e
}

/// 1 if the last [`ft_branch`] exhausted the tree and its answer is the proved minimum, else 0.
#[no_mangle]
pub extern "C" fn ft_branch_proved(sim: *const Sim) -> u32 {
    match unsafe { sim.as_ref() }.and_then(|s| s.bb.as_ref()) {
        Some(o) => u32::from(o.proved_optimal),
        None => 0,
    }
}

/// Nodes the last [`ft_branch`] visited. 0 if there was none.
#[no_mangle]
pub extern "C" fn ft_branch_nodes(sim: *const Sim) -> u64 {
    unsafe { sim.as_ref() }.and_then(|s| s.bb.as_ref()).map_or(0, |o| o.nodes)
}

/// An **upper bound** on the maximum cut of a toroidal grid, from the same dual reduction.
///
/// A torus is not a plane and [`ft_planar_cut`] refuses it, correctly. But the dual argument needs
/// only faces, and an embedding on any surface has them. What changes is what the answer means: on
/// a torus the cycle space of the dual is four times the cut space, so the relaxation ranges over
/// sets that are not cuts and its optimum can only bound the maximum from above.
///
/// That is the side of G-set nobody publishes. Every figure in the table is a best cut **found** —
/// a lower bound. This is the other end of the bracket. Measured: it closes the bracket on G11,
/// proving the twenty-five-year-old best-known cut of 564 optimal.
///
/// Returns NaN unless the graph is a toroidal grid, whose structure is recovered from the edge list
/// — a match on all `2n` edges rather than a guess. [`ft_toroidal_attained`] says whether the bound
/// happens to be achieved by a genuine cut, in which case it is the maximum rather than a bound.
#[no_mangle]
pub extern "C" fn ft_toroidal_bound(sim: *mut Sim, scale: f64) -> f64 {
    let Some(s) = (unsafe { sim.as_mut() }) else { return f64::NAN };
    s.tor = None;
    let Some(emb) = crate::planar::torus_grid_of(&s.graph) else { return f64::NAN };
    let p = crate::planarcut::Params { scale };
    match crate::planarcut::bound_on_surface(&s.graph, &emb, &p) {
        Ok(b) => {
            // A bound that is attained comes with the state that attains it, and leaving it behind
            // is what makes `ft_energy` the proved minimum in that case.
            if let Some(st) = &b.state {
                if st.len() == s.sampler_state.len() {
                    s.sampler_state.copy_from_slice(st);
                }
            }
            let c = b.cut;
            s.tor = Some(b);
            c
        }
        Err(_) => f64::NAN,
    }
}

/// 1 if the last [`ft_toroidal_bound`] was **attained** by a genuine cut, else 0.
///
/// Attained means the relaxation's optimum two-coloured the graph, so it is a cut and the bound is
/// the maximum — proved, not bounded. Not attained still leaves the bound standing: every cut is
/// such a subgraph, so a maximum over the larger set can only be larger.
#[no_mangle]
pub extern "C" fn ft_toroidal_attained(sim: *const Sim) -> u32 {
    match unsafe { sim.as_ref() }.and_then(|s| s.tor.as_ref()) {
        Some(b) => u32::from(b.attained),
        None => 0,
    }
}

/// Goemans–Williamson: round the semidefinite relaxation to a state.
///
/// **The only worst-case guarantee in max-cut.** [`ft_bound_sdp`] uses the relaxation from the dual
/// side to produce a bound; this uses it from the primal side to produce a solution, by cutting the
/// sphere the relaxation placed the nodes on with a random hyperplane. Returns the cut under
/// `w = −J`, or NaN on a null handle, and leaves the state on the simulation.
///
/// **The 0.87856 ratio does not apply in general** — it is stated for non-negative edge weights,
/// which here means non-positive couplings and no fields. [`ft_gw_guaranteed`] says which case this
/// was, because a guarantee that is always claimed is not a guarantee.
#[no_mangle]
pub extern "C" fn ft_gw_round(sim: *mut Sim, hyperplanes: u32, seed: u64) -> f64 {
    let Some(s) = (unsafe { sim.as_mut() }) else { return f64::NAN };
    let r = crate::sdp::goemans_williamson(&s.graph, &crate::sdp::Params::default(), seed, hyperplanes.max(1) as usize);
    if r.state.len() == s.sampler_state.len() {
        s.sampler_state.copy_from_slice(&r.state);
    }
    let c = r.cut;
    s.gw = Some(r);
    c
}

/// 1 if the last [`ft_gw_round`] was inside the hypothesis of the 0.87856 guarantee, else 0.
#[no_mangle]
pub extern "C" fn ft_gw_guaranteed(sim: *const Sim) -> u32 {
    match unsafe { sim.as_ref() }.and_then(|s| s.gw.as_ref()) {
        Some(r) => u32::from(r.guaranteed),
        None => 0,
    }
}

/// Parallel tempering with **isoenergetic cluster moves** — the baseline the field measures against.
///
/// Two ladders of `rungs` replicas from `beta_min` to `beta_max`; every round, a connected component
/// of the disagreement subgraph between the two replicas at each temperature is flipped in both. The
/// move preserves the pair's energy exactly and is therefore always accepted, which is what makes it
/// a cluster algorithm for a spin glass.
///
/// Returns the best energy found, or NaN when the graph carries a **field** — the isoenergetic
/// argument holds only at `h = 0`, and accepting the move anyway would be silently wrong.
#[no_mangle]
pub extern "C" fn ft_icm(sim: *mut Sim, rungs: u32, rounds: u32, beta_min: f64, beta_max: f64) -> f64 {
    let Some(s) = (unsafe { sim.as_mut() }) else { return f64::NAN };
    if !(beta_min > 0.0 && beta_max > beta_min) {
        return f64::NAN;
    }
    let p = crate::icm::Params {
        betas: crate::tempering::geometric_ladder(beta_min, beta_max, rungs.max(2) as usize),
        rounds: rounds.max(1) as usize,
        sweeps_per_round: 1,
        swap_every: 1,
        icm_every: 1,
    };
    match crate::icm::run_metered(&s.graph, &p, s.seed, Some(&mut s.ledger)) {
        Ok(o) => {
            if o.state.len() == s.sampler_state.len() {
                s.sampler_state.copy_from_slice(&o.state);
            }
            let e = o.energy;
            s.ic = Some(o);
            e
        }
        Err(_) => f64::NAN,
    }
}

/// Cluster moves that actually fired in the last [`ft_icm`]. 0 if there was none.
///
/// Reported because a move that never fires is not a move: two replicas that agree everywhere have
/// no disagreement subgraph and nothing to exchange.
#[no_mangle]
pub extern "C" fn ft_icm_moves(sim: *const Sim) -> u64 {
    unsafe { sim.as_ref() }.and_then(|s| s.ic.as_ref()).map_or(0, |o| o.icm_moves as u64)
}

/// Simulated quantum annealing: path-integral Monte Carlo on the transverse-field Ising model.
///
/// `trotter` slices at fixed `beta`, with the transverse field annealed from `gamma_max` down to
/// `gamma_min` over `steps`. **One slice is classical**, which is the honest control rather than a
/// degenerate case. `gamma_min` must not be zero: `J⊥` diverges there, and it is clamped rather than
/// divided by. Returns the best classical energy found and leaves that state on the simulation.
#[no_mangle]
pub extern "C" fn ft_sqa(
    sim: *mut Sim,
    trotter: u32,
    beta: f64,
    gamma_max: f64,
    gamma_min: f64,
    steps: u32,
) -> f64 {
    let Some(s) = (unsafe { sim.as_mut() }) else { return f64::NAN };
    if !(beta > 0.0 && gamma_max > 0.0 && gamma_min >= 0.0 && gamma_max >= gamma_min) {
        return f64::NAN;
    }
    let p = crate::sqa::Params {
        trotter: trotter.max(1) as usize,
        beta,
        gamma_max,
        gamma_min,
        steps: steps.max(1) as usize,
        sweeps_per_step: 1,
    };
    let o = crate::sqa::run_metered(&s.graph, &p, s.seed, Some(&mut s.ledger));
    if o.state.len() == s.sampler_state.len() {
        s.sampler_state.copy_from_slice(&o.state);
    }
    o.energy
}

/// Breakout local search — the algorithm that holds the max-cut record on most of G-set.
///
/// Steepest descent with an adaptive perturbation between local optima; see [`crate::bls`] for what
/// makes it different from [`ft_tabu`]. Returns the best energy found, or NaN on a null handle.
///
/// One iteration is one **spin flip**, which is also what [`ft_tabu`] counts — so passing the same
/// number to both is a matched-budget comparison, and it is the only comparison this ABI can offer
/// honestly: a wall-clock one needs a quiet machine.
#[no_mangle]
pub extern "C" fn ft_bls(sim: *mut Sim, iterations: u32) -> f64 {
    let Some(s) = (unsafe { sim.as_mut() }) else { return f64::NAN };
    let p = crate::bls::Params {
        iterations: iterations.max(1) as usize,
        ..crate::bls::Params::default()
    };
    let out = crate::bls::search_metered(&s.graph, &p, s.seed, Some(&mut s.ledger));
    if out.state.len() == s.sampler_state.len() {
        s.sampler_state.copy_from_slice(&out.state);
    }
    let e = out.energy;
    s.bl = Some(out);
    e
}

/// Local optima the last [`ft_bls`] visited. 0 if there was none.
///
/// The number that says whether the search had room to work: a run with a handful of descents spent
/// its budget inside one basin and is a descent, not a breakout search.
#[no_mangle]
pub extern "C" fn ft_bls_descents(sim: *const Sim) -> u64 {
    unsafe { sim.as_ref() }.and_then(|s| s.bl.as_ref()).map_or(0, |o| o.descents as u64)
}

/// Flips the last [`ft_bls`] actually made, which is not always the budget it was given.
#[no_mangle]
pub extern "C" fn ft_bls_iterations(sim: *const Sim) -> u64 {
    unsafe { sim.as_ref() }.and_then(|s| s.bl.as_ref()).map_or(0, |o| o.iterations_run as u64)
}

/// The largest jump magnitude the last [`ft_bls`] reached — how hard it had to work to escape.
///
/// It grows only when a descent returns to the immediately previous local optimum, so a value above
/// the initial `L0` is direct evidence the adaptive rule fired rather than idled.
#[no_mangle]
pub extern "C" fn ft_bls_max_jump(sim: *const Sim) -> u32 {
    unsafe { sim.as_ref() }.and_then(|s| s.bl.as_ref()).map_or(0, |o| o.max_jump as u32)
}

/// **Exact** max-cut on a planar graph, in polynomial time. Not a search.
///
/// Returns the maximum cut weight under `w = −J`, or NaN when this graph cannot be solved this way
/// — in which case [`ft_planar_error`] says which of the four reasons it was, because they are four
/// different instructions to the caller. The simulation's state is set to the optimal partition, so
/// [`ft_energy`] returns the **proved minimum** energy.
///
/// `scale` multiplies every coupling before it is rounded to an integer; pass 1.0 for whole-number
/// couplings. The matching underneath is exact only in exact arithmetic, so a weight that does not
/// land on an integer is refused rather than rounded.
#[no_mangle]
pub extern "C" fn ft_planar_cut(sim: *mut Sim, scale: f64) -> f64 {
    let Some(s) = (unsafe { sim.as_mut() }) else { return f64::NAN };
    let p = crate::planarcut::Params { scale };
    match crate::planarcut::solve(&s.graph, &p) {
        Ok(o) => {
            if o.state.len() == s.sampler_state.len() {
                s.sampler_state.copy_from_slice(&o.state);
            }
            let c = o.cut;
            s.pc = Some(Ok(o));
            c
        }
        Err(e) => {
            s.pc = Some(Err(e.to_string()));
            f64::NAN
        }
    }
}

/// Faces in the planar embedding from the last [`ft_planar_cut`] — the dual's vertex count.
#[no_mangle]
pub extern "C" fn ft_planar_faces(sim: *const Sim) -> u64 {
    match unsafe { sim.as_ref() }.and_then(|s| s.pc.as_ref()) {
        Some(Ok(o)) => o.faces as u64,
        _ => 0,
    }
}

/// Odd-degree dual vertices from the last [`ft_planar_cut`].
///
/// The size of the matching problem, and the real cost driver: this is what makes the method
/// `O(n³)` rather than `O(2ⁿ)`.
#[no_mangle]
pub extern "C" fn ft_planar_odd_faces(sim: *const Sim) -> u64 {
    match unsafe { sim.as_ref() }.and_then(|s| s.pc.as_ref()) {
        Some(Ok(o)) => o.odd_faces as u64,
        _ => 0,
    }
}

/// Why the last [`ft_planar_cut`] refused, in the caller's own terms.
///
/// Two-call text protocol: pass a null buffer to learn the length, then a buffer of that size.
/// Empty when the last call succeeded or none has happened. Exported because "not planar", "has a
/// cut vertex", "has fields" and "weights are not integral" are four different things to do next,
/// and a bare NaN collapses them into one.
#[no_mangle]
pub extern "C" fn ft_planar_error(sim: *const Sim, buf: *mut u8, cap: u32) -> u32 {
    let msg = match unsafe { sim.as_ref() }.and_then(|s| s.pc.as_ref()) {
        Some(Err(e)) => e.as_str(),
        _ => "",
    };
    let bytes = msg.as_bytes();
    if buf.is_null() {
        return bytes.len() as u32;
    }
    let n = bytes.len().min(cap as usize);
    unsafe { core::ptr::copy_nonoverlapping(bytes.as_ptr(), buf, n) };
    n as u32
}

/// A **lower bound on the ground energy** from decoupling every term. The cheapest and weakest.
///
/// `min_s E(s) ≥ −Σ|h| − Σ|J|`, in `O(edges)`. Every bound here is sound on its own, so a caller
/// should take the maximum of the ones it can afford.
#[no_mangle]
pub extern "C" fn ft_bound_decoupled(sim: *const Sim) -> f64 {
    unsafe { sim.as_ref() }.map_or(f64::NAN, |s| crate::bound::decoupled(&s.graph).value)
}

/// Lagrangian decomposition into forests, tightened by `rounds` of subgradient ascent.
///
/// **Worth nothing on an instance with no fields**: a tree is never frustrated, so every part
/// minimises to `−Σ|J|` and this degenerates to [`ft_bound_decoupled`]. Exported anyway, with the
/// caveat, because a caller comparing bounds should be able to see that for themselves.
#[no_mangle]
pub extern "C" fn ft_bound_forest(sim: *const Sim, rounds: u32) -> f64 {
    unsafe { sim.as_ref() }
        .map_or(f64::NAN, |s| crate::bound::forest(&s.graph, rounds as usize).value)
}

/// Charges `2·min|J|` for every edge-disjoint frustrated cycle up to length `max_len`.
///
/// Edge-disjointness is what makes the penalties add: two cycles sharing an edge could be paid for
/// by the same single violation.
#[no_mangle]
pub extern "C" fn ft_bound_odd_cycle(sim: *const Sim, max_len: u32) -> f64 {
    unsafe { sim.as_ref() }
        .map_or(f64::NAN, |s| crate::bound::odd_cycle(&s.graph, max_len as usize).value)
}

/// The certified semidefinite bound — **re-verified at this boundary before it is returned**.
///
/// [`crate::sdp::certified`] hands back a dual point and the bound it certifies; this rebuilds the
/// cost matrix from the graph and re-runs the positive-definiteness proof before letting the number
/// cross, and returns NaN if that fails. A bound that only its own author can reproduce is not a
/// bound, and a bound crossing a language boundary is exactly the case where the caller cannot
/// check it themselves.
#[no_mangle]
pub extern "C" fn ft_bound_sdp(sim: *const Sim, sweeps: u32, seed: u64) -> f64 {
    let Some(s) = (unsafe { sim.as_ref() }) else { return f64::NAN };
    let p = crate::sdp::Params { sweeps: sweeps.max(1) as usize, ..crate::sdp::Params::default() };
    let (_, cert) = crate::sdp::certified(&s.graph, &p, seed);
    cert.verify(&s.graph).unwrap_or(f64::NAN)
}

#[cfg(test)]
mod solver_ffi_tests {
    use super::*;

    /// Everything a caller can reach through this boundary has to agree with the state it left
    /// behind — the number returned is a claim about `ft_spins`, not a separate answer.
    #[test]
    fn every_solver_leaves_the_state_its_energy_belongs_to() {
        for (name, run) in [
            ("tabu", (|s: *mut Sim| ft_tabu(s, 4_000, 0, 1_000)) as fn(*mut Sim) -> f64),
            ("bls", |s: *mut Sim| ft_bls(s, 4_000)),
            ("popanneal", |s: *mut Sim| ft_popanneal(s, 64, 2, 6.0, 20)),
            ("branch", |s: *mut Sim| ft_branch(s, 2_000_000)),
        ] {
            let sim = ft_planted_frustrated(4, 8, 7, 1.0);
            assert!(!sim.is_null());
            let e = run(sim);
            assert!(e.is_finite(), "{name} returned {e}");
            if name == "tabu" {
                assert_eq!(ft_tabu_iterations(sim), 4_000, "the whole budget, not a truncated run");
            }
            if name == "bls" {
                assert_eq!(ft_bls_iterations(sim), 4_000, "the whole budget, not a truncated run");
                assert!(ft_bls_descents(sim) > 0, "a search with no descents is not a search");
                assert!(ft_bls_max_jump(sim) >= 1);
            }
            assert!(
                (ft_energy(sim) - e).abs() < 1e-9,
                "{name}: returned {e}, the state it left has {}",
                ft_energy(sim)
            );
            let known = ft_ground_energy(sim);
            assert!(e >= known - 1e-9, "{name} beat the planted optimum {known} with {e}");
            ft_free(sim);
        }
    }

    /// Branch and bound has to report a PROOF, and has to withhold one when the budget ran out.
    #[test]
    fn the_proof_flag_crosses_the_boundary_and_can_say_no() {
        let sim = ft_planted_frustrated(3, 4, 1, 1.0);
        let e = ft_branch(sim, 5_000_000);
        assert_eq!(ft_branch_proved(sim), 1, "a 9-spin tree fits in five million nodes");
        assert!(ft_branch_nodes(sim) > 0);
        assert!((e - ft_ground_energy(sim)).abs() < 1e-9, "a proved minimum IS the planted optimum");
        ft_free(sim);

        // A budget that genuinely runs out needs a genuinely hard instance. The obvious choice --
        // a 64-spin Z1 grid -- turned out to prove itself in under 200 nodes, and that is not a
        // weak test, it is a fact about the instance: a FERROMAGNET is unfrustrated, every coupling
        // and every field can be satisfied at once, so `decoupled` is EXACTLY tight there and the
        // root bound already equals the incumbent. Asserted below rather than discarded.
        let hard = ft_planted_wishart(40, 0.5, 5, 1.0);
        ft_branch(hard, 200);
        assert_eq!(ft_branch_proved(hard), 0, "200 nodes cannot exhaust a dense 40-spin tree");
        assert!(ft_branch_nodes(hard) <= 201);
        ft_free(hard);
    }

    /// An unfrustrated instance is proved almost immediately, and the reason is worth stating.
    ///
    /// On a ferromagnet with aligned fields every term is satisfiable at once, so
    /// `-Σ|h| - Σ|J|` is not a relaxation at all -- it is the ground energy. The root bound equals
    /// the incumbent and the whole tree prunes. This is the case where the cheapest bound in the
    /// crate is also the best one available, which is easy to forget after measuring it on G-set.
    #[test]
    fn a_ferromagnet_is_proved_at_once_because_the_cheap_bound_is_exact_there() {
        let sim = ft_z1_new(8, 8, 0.5, 0.1, 1.0, 3);
        let e = ft_branch(sim, 100_000);
        assert_eq!(ft_branch_proved(sim), 1);
        assert!(ft_branch_nodes(sim) < 300, "took {} nodes", ft_branch_nodes(sim));
        let d = ft_bound_decoupled(sim);
        assert!((e - d).abs() < 1e-9, "ground {e} should equal the decoupled bound {d}");
        ft_free(sim);
    }

    /// Population annealing's diagnostic is the point of the method, so it has to cross too.
    #[test]
    fn population_annealing_hands_over_its_free_energy_and_its_warning() {
        let sim = ft_z1_new(4, 4, 0.4, 0.0, 1.0, 5);
        assert!(ft_popanneal_ln_z(sim).is_nan(), "no run yet, so no free energy");
        assert!(ft_popanneal_rho(sim).is_nan());
        ft_popanneal(sim, 128, 2, 3.0, 25);
        let ln_z = ft_popanneal_ln_z(sim);
        // Z(0) = 2^n and Z is non-decreasing in beta, so ln Z at any beta is at least n ln 2.
        let floor = 16.0 * core::f64::consts::LN_2;
        assert!(ln_z >= floor - 1e-9, "ln Z {ln_z} below n ln 2 = {floor}");
        let rho = ft_popanneal_rho(sim);
        assert!((1.0..=128.0).contains(&rho), "rho {rho} outside [1, population]");
        ft_free(sim);
    }

    /// Every bound must be a bound, and the boundary must not be able to invent one.
    ///
    /// Checked against a PLANTED optimum, which is the only ground truth available on both sides of
    /// this boundary without enumerating anything.
    #[test]
    fn no_bound_crossing_this_boundary_exceeds_a_known_optimum() {
        let sim = ft_planted_frustrated(4, 12, 11, 1.0);
        let known = ft_ground_energy(sim);
        assert!(known.is_finite());
        let bounds = [
            ("decoupled", ft_bound_decoupled(sim)),
            ("forest", ft_bound_forest(sim, 20)),
            ("odd_cycle", ft_bound_odd_cycle(sim, 6)),
            ("sdp", ft_bound_sdp(sim, 100, 1)),
        ];
        for (name, v) in bounds {
            assert!(v.is_finite(), "{name} returned {v}");
            assert!(v <= known + 1e-9, "{name} bound {v} EXCEEDS the planted optimum {known}");
        }
        // And the SDP, which is the expensive one, has to earn that by beating the trivial floor.
        assert!(
            bounds[3].1 >= bounds[0].1 - 1e-9,
            "sdp {} is worse than decoupled {}",
            bounds[3].1,
            bounds[0].1
        );
        ft_free(sim);
    }

    /// The exact planar solver crosses the boundary, and so does the REASON it refuses.
    ///
    /// Four refusals, four different things for a caller to do next. A bare NaN collapses them into
    /// "it did not work", which is the least useful sentence available.
    #[test]
    fn the_planar_solver_and_its_four_refusals_cross_the_boundary() {
        // A 4x4 antiferromagnetic grid: bipartite, so every one of its 24 edges is cut.
        let b = ft_builder_new(16);
        for y in 0..4u32 {
            for x in 0..4u32 {
                let i = y * 4 + x;
                if x + 1 < 4 {
                    ft_builder_couple(b, i, i + 1, -1.0);
                }
                if y + 1 < 4 {
                    ft_builder_couple(b, i, i + 4, -1.0);
                }
            }
        }
        let sim = ft_builder_build(b, 1.0, 1);
        assert_eq!(ft_planar_cut(sim, 1.0), 24.0);
        assert_eq!(ft_planar_faces(sim), 10);
        // ZERO odd faces, and that is a fact rather than a failure: with uniform weights every
        // square face of a grid has degree 4 and the outer face degree 12, so the T-join is empty
        // and the whole cut is free. Asserting `> 0` here -- as the first version of this test did
        // -- asserts that the easy case does not occur.
        assert_eq!(ft_planar_odd_faces(sim), 0);
        assert_eq!(ft_planar_error(sim, core::ptr::null_mut(), 0), 0, "no error on success");
        // The state left behind is the optimum, so `ft_energy` is the PROVED minimum.
        assert_eq!(ft_energy(sim), -24.0);
        ft_free(sim);

        // A frustrated grid: mixed signs make face degrees odd, and the matching has work to do.
        let b = ft_builder_new(16);
        // A fixed, irregular sign pattern. Cycled rather than computed from a modulus, because the
        // point is only that the signs are mixed and the pattern is reproducible.
        let signs = [-1.0f64, -1.0, 1.0, -1.0, 1.0];
        let mut k = 0usize;
        for y in 0..4u32 {
            for x in 0..4u32 {
                let i = y * 4 + x;
                if x + 1 < 4 {
                    ft_builder_couple(b, i, i + 1, signs[k % signs.len()]);
                    k += 1;
                }
                if y + 1 < 4 {
                    ft_builder_couple(b, i, i + 4, signs[k % signs.len()]);
                    k += 1;
                }
            }
        }
        let frus = ft_builder_build(b, 1.0, 1);
        let c = ft_planar_cut(frus, 1.0);
        assert!(c.is_finite() && c < 24.0, "a frustrated grid cannot cut every edge: {c}");
        assert!(ft_planar_odd_faces(frus) > 0, "frustration makes face degrees odd");
        ft_free(frus);

        // A torus is genus 1, and the reduction is a plane statement.
        let torus = ft_ising2d_new(4, 1.0, 1.0, 1);
        assert!(ft_planar_cut(torus, 1.0).is_nan());
        let need = ft_planar_error(torus, core::ptr::null_mut(), 0);
        assert!(need > 0, "a refusal must carry a reason");
        let mut buf = vec![0u8; need as usize];
        let got = ft_planar_error(torus, buf.as_mut_ptr(), need);
        let msg = String::from_utf8_lossy(&buf[..got as usize]).to_string();
        assert!(msg.contains("not planar"), "{msg}");
        ft_free(torus);
    }

    /// The toroidal bound crosses, and it bounds -- checked against a search that cannot beat it.
    #[test]
    fn the_toroidal_bound_crosses_and_is_never_beaten() {
        // A 6x6 periodic lattice: a torus, refused by the planar solver and answered by this one.
        let torus = ft_ising2d_new(6, -1.0, 1.0, 3);
        let bound = ft_toroidal_bound(torus, 1.0);
        assert!(bound.is_finite(), "a periodic lattice IS a toroidal grid");
        assert!(ft_planar_cut(torus, 1.0).is_nan(), "and it is not planar");
        // Every edge of a bipartite torus can be cut, and 6x6 is bipartite: 72 edges.
        assert_eq!(bound, 72.0);
        assert_eq!(ft_toroidal_attained(torus), 1, "a bound that is achieved says so");
        ft_free(torus);

        // A frustrated torus: a search must never exceed the bound, which is what makes it one.
        let hard = ft_ising2d_new(5, 1.0, 1.0, 3);
        let b = ft_toroidal_bound(hard, 1.0);
        assert!(b.is_finite());
        let e = ft_bls(hard, 200_000);
        // cut = (W - E) / 2 with W = sum of -J over 50 edges of J = +1, so W = -50.
        let cut = (-50.0 - e) / 2.0;
        assert!(cut <= b + 1e-9, "breakout local search reached {cut}, above the bound {b}");
        ft_free(hard);

        // An open grid is planar, not toroidal, and this must decline rather than answer.
        let b2 = ft_builder_new(9);
        for y in 0..3u32 {
            for x in 0..3u32 {
                let i = y * 3 + x;
                if x + 1 < 3 {
                    ft_builder_couple(b2, i, i + 1, -1.0);
                }
                if y + 1 < 3 {
                    ft_builder_couple(b2, i, i + 3, -1.0);
                }
            }
        }
        let planar = ft_builder_build(b2, 1.0, 1);
        assert!(ft_toroidal_bound(planar, 1.0).is_nan(), "an open grid is not a torus");
        ft_free(planar);
    }

    /// The three algorithms the toolchain survey named as missing, crossing the boundary.
    #[test]
    fn the_closed_gaps_reach_this_boundary_and_report_their_own_caveats() {
        // A 6x6 ANTIferromagnet: non-positive couplings, so the GW guarantee applies.
        let anti = ft_ising2d_new(6, -1.0, 1.0, 3);
        let cut = ft_gw_round(anti, 64, 5);
        assert!(cut.is_finite());
        assert_eq!(ft_gw_guaranteed(anti), 1, "an antiferromagnet is inside the hypothesis");
        // Bipartite: every one of the 72 edges can be cut, and GW should find it.
        assert_eq!(cut, 72.0);
        assert_eq!(ft_energy(anti), -72.0, "the state left behind is the one that cut them");
        ft_free(anti);

        // A ferromagnet is OUTSIDE the hypothesis, and the flag has to say so.
        let ferro = ft_ising2d_new(6, 1.0, 1.0, 3);
        assert!(ft_gw_round(ferro, 16, 5).is_finite());
        assert_eq!(ft_gw_guaranteed(ferro), 0, "positive couplings are outside the theorem");

        // Cluster moves fire, and simulated quantum annealing finds the ferromagnetic ground state.
        let e = ft_icm(ferro, 8, 200, 0.1, 4.0);
        assert!((e + 72.0).abs() < 1e-9, "icm reached {e}");
        assert!(ft_icm_moves(ferro) > 0, "the cluster move never fired");
        let q = ft_sqa(ferro, 4, 10.0, 3.0, 0.05, 200);
        assert!((q + 72.0).abs() < 1e-9, "sqa reached {q}");
        ft_free(ferro);

        // A field breaks the isoenergetic argument, and ICM must decline rather than accept.
        let b = ft_builder_new(6);
        for i in 0..6u32 {
            ft_builder_couple(b, i, (i + 1) % 6, 1.0);
        }
        ft_builder_bias(b, 2, 0.5);
        let fielded = ft_builder_build(b, 1.0, 1);
        assert!(ft_icm(fielded, 4, 50, 0.1, 4.0).is_nan(), "a field is not isoenergetic");
        ft_free(fielded);
    }

    /// A null handle is a caller error, not a crash, and NaN is how this ABI says so.
    #[test]
    fn a_null_handle_returns_rather_than_dereferencing() {
        let n: *mut Sim = core::ptr::null_mut();
        assert!(ft_tabu(n, 10, 0, 0).is_nan());
        assert!(ft_bls(n, 10).is_nan());
        assert!(ft_planar_cut(n, 1.0).is_nan());
        assert!(ft_toroidal_bound(n, 1.0).is_nan());
        assert!(ft_gw_round(n, 8, 1).is_nan());
        assert_eq!(ft_gw_guaranteed(n), 0);
        assert!(ft_icm(n, 8, 10, 0.1, 4.0).is_nan());
        assert_eq!(ft_icm_moves(n), 0);
        assert!(ft_sqa(n, 4, 1.0, 3.0, 0.05, 10).is_nan());
        assert_eq!(ft_toroidal_attained(n), 0);
        assert_eq!(ft_planar_faces(n), 0);
        assert_eq!(ft_planar_odd_faces(n), 0);
        assert_eq!(ft_planar_error(n, core::ptr::null_mut(), 0), 0);
        assert_eq!(ft_bls_descents(n), 0);
        assert_eq!(ft_bls_iterations(n), 0);
        assert_eq!(ft_bls_max_jump(n), 0);
        assert!(ft_popanneal(n, 8, 1, 1.0, 4).is_nan());
        assert!(ft_branch(n, 10).is_nan());
        assert!(ft_bound_decoupled(n).is_nan());
        assert!(ft_bound_forest(n, 4).is_nan());
        assert!(ft_bound_odd_cycle(n, 6).is_nan());
        assert!(ft_bound_sdp(n, 10, 0).is_nan());
        assert_eq!(ft_branch_proved(n), 0);
        assert_eq!(ft_branch_nodes(n), 0);
        assert!(ft_popanneal_ln_z(n).is_nan());
        assert!(ft_popanneal_rho(n).is_nan());
    }
}
