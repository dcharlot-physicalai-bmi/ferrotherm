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
//! Clippy's `not_unsafe_ptr_arg_deref` is deny-by-default, and these `extern "C"` entry points
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
    /// The last annealed-importance-sampling run, for the `ft_ln_z_ais_*` accessors.
    fe: Option<crate::free_energy::Ais>,
    /// The last branch-and-bound outcome, for the `ft_branch_*` accessors.
    bb: Option<crate::branch::Outcome>,
    /// The last HFS descent, for the `ft_hfs_*` accessors.
    hf: Option<crate::hfs::Outcome>,
    /// The last collected sample set, for the `ft_samples_*` accessors.
    sm: Option<crate::samples::SampleSet>,
    sampler_state: Vec<i8>,
    beta: f64,
    seed: u64,
    sweeps_done: u64,
    /// Threads the last `ft_sweep_par` actually used. See [`ft_threads_used`].
    threads_used: u32,
    ledger: Ledger,
    /// The vendor's linear qubit index per node, for a simulation built on a real device topology.
    /// Empty for every graph that has no such numbering. See [`ft_qubit`].
    qubits: Vec<u32>,
    /// For a simulation produced by [`ft_sparsify`]: which nodes represent each logical variable,
    /// and the constant that turns a sparse energy back into a logical one. Empty otherwise.
    copies: Vec<Vec<u32>>,
    sparsify_offset: f64,
    /// The placement [`ft_embed`] found, or the one an embedded simulation was built from.
    emb: Option<crate::embed::Embedding>,
}

impl Sim {
    fn new(graph: Graph, beta: f64, seed: u64) -> *mut Sim {
        let g = Box::new(graph);
        // SAFETY of the self-reference dance avoided: store state, rebuild Sampler per call.
        let sampler = Sampler::new(&g, beta, seed);
        Box::into_raw(Box::new(Sim { sampler_state: sampler.s.clone(), graph: g, beta, seed, sweeps_done: 0,
            threads_used: 0, ledger: Ledger::default(), gpu: None, ground: None, cert: None, tb: None, bl: None, pc: None, tor: None, gw: None, ic: None, pa: None,
            fe: None, bb: None, hf: None, sm: None, qubits: Vec::new(), copies: Vec::new(), sparsify_offset: 0.0, emb: None }))
    }

    /// As [`Sim::new`], keeping the device topology's own qubit numbering. See [`ft_qubit`].
    fn with_qubits(t: crate::device::Topology, beta: f64, seed: u64) -> *mut Sim {
        let p = Sim::new(t.graph, beta, seed);
        // SAFETY: `Sim::new` just handed back a live, uniquely-owned allocation.
        unsafe { (*p).qubits = t.qubits };
        p
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
/// that is not `[0,1]`, an objective of degree three or more, or CONSTRAINTS -- ferrotherm
/// expresses a constraint as a penalty whose weight changes the answer, so reading the objective
/// alone would hand back the relaxation, which is a different problem.
///
/// This sampler samples spins, and a
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

/// New **Pegasus** `P_m` graph — the topology of every D-Wave *Advantage* processor.
///
/// `m = 16` is the Advantage: 5,640 qubits, 40,484 couplers, degree 15. Uniform coupling `j`.
/// Null for `m < 2`, which has no qubits.
///
/// The nominal full-yield graph, not a particular machine's working graph. Read
/// [`ft_qubit`] to get the vendor's own qubit number for a node — this crate indexes densely and
/// Pegasus does not, so the two disagree and programming a machine with our indices would drive
/// the wrong qubits.
#[no_mangle]
pub extern "C" fn ft_pegasus_new(m: u32, j: f64, beta: f64, seed: u64) -> *mut Sim {
    let t = crate::device::pegasus(m as usize, j);
    if t.graph.n == 0 {
        return core::ptr::null_mut();
    }
    Sim::with_qubits(t, beta, seed)
}

/// New **Zephyr** `Z_{m,t}` graph — the topology of D-Wave's *Advantage2* processors.
///
/// `m = 15, t = 4` is the Advantage2: 7,440 qubits, 71,736 couplers, degree 20. `t` is 4 on every
/// shipped machine; 0 for either parameter returns null.
///
/// Zephyr's higher degree is what it is for: the same problem embeds with shorter chains, and a
/// chain that breaks leaves a variable with no value at all. See `examples/embedding_tax.rs`.
#[no_mangle]
pub extern "C" fn ft_zephyr_new(m: u32, t: u32, j: f64, beta: f64, seed: u64) -> *mut Sim {
    let z = crate::device::zephyr(m as usize, t as usize, j);
    if z.graph.n == 0 {
        return core::ptr::null_mut();
    }
    Sim::with_qubits(z, beta, seed)
}

/// The **vendor's** linear qubit index for node `i`, or `0xFFFFFFFF` if there is no mapping.
///
/// Two numbering systems meet at this boundary and conflating them is silent. Every sampler here
/// indexes `0..n` densely; Pegasus's fabric drops the qubits outside its largest component, so its
/// own numbering is sparse — a `P16` spreads 5,640 qubits over indices 30 to 5,729. A chain written
/// in our indices and handed to a machine programs different qubits, and the answer comes back
/// looking like a bad embedding rather than like the mistake it is.
///
/// `0xFFFFFFFF` — not 0, which is a valid qubit — for a simulation built from a graph with no
/// vendor numbering at all, which is every one except [`ft_pegasus_new`] and [`ft_zephyr_new`].
#[no_mangle]
pub extern "C" fn ft_qubit(sim: *const Sim, i: u32) -> u32 {
    match unsafe { sim.as_ref() } {
        Some(s) => s.qubits.get(i as usize).copied().unwrap_or(u32::MAX),
        None => u32::MAX,
    }
}

#[cfg(test)]
mod topology_ffi_tests {
    use super::*;

    /// The machines you can actually rent, across the boundary, with their own numbering intact.
    #[test]
    fn the_abi_builds_the_advantage_and_advantage2_graphs() {
        // P16 is the Advantage: 5,640 qubits at degree 15.
        let p = ft_pegasus_new(16, 1.0, 0.5, 3);
        assert!(!p.is_null());
        assert_eq!(ft_len(p), 5640);

        // The vendor numbering is SPARSE and must survive the crossing. Our node 0 is their
        // qubit 30, not their qubit 0 -- which is the whole reason this accessor exists.
        assert_eq!(ft_qubit(p, 0), 30);
        assert_eq!(ft_qubit(p, 5639), 5729);
        assert!((0..5640).all(|i| ft_qubit(p, i) < ft_qubit(p, i + 1)), "strictly increasing");
        assert_eq!(ft_qubit(p, 5640), u32::MAX, "past the end is not qubit zero");

        // It samples like any other graph.
        ft_sweep(p, 20);
        assert!(ft_energy(p).is_finite());
        ft_free(p);

        // Z15 is the Advantage2: 7,440 qubits at degree 20, densely numbered.
        let z = ft_zephyr_new(15, 4, 1.0, 0.5, 3);
        assert!(!z.is_null());
        assert_eq!(ft_len(z), 7440);
        assert_eq!(ft_qubit(z, 0), 0);
        assert_eq!(ft_qubit(z, 7439), 7439);
        ft_free(z);
    }

    /// A graph with no vendor numbering says so, rather than answering zero.
    #[test]
    fn a_graph_without_a_device_numbering_refuses_the_question() {
        let s = ft_ising2d_new(4, 1.0, 0.5, 1);
        assert_eq!(ft_len(s), 16);
        assert_eq!(ft_qubit(s, 0), u32::MAX, "a lattice has no vendor qubit numbers");
        ft_free(s);
        assert_eq!(ft_qubit(core::ptr::null(), 0), u32::MAX);
    }

    #[test]
    fn a_size_with_no_qubits_is_null_rather_than_an_empty_simulation() {
        for m in [0u32, 1] {
            assert!(ft_pegasus_new(m, 1.0, 0.5, 1).is_null(), "P{m} has no qubits");
        }
        assert!(ft_zephyr_new(0, 4, 1.0, 0.5, 1).is_null());
        assert!(ft_zephyr_new(4, 0, 1.0, 0.5, 1).is_null());
    }
}

/// Rewrite this simulation's model so no variable exceeds `budget` neighbours, and return the
/// rewritten one.
///
/// A model denser than a fabric has two routes onto it. Minor embedding PLACES it onto one specific
/// machine; this REWRITES it, with no machine involved, by splitting each heavy variable into copies
/// bound by a strong coupling. The result is a larger, sparser model with the same ground states,
/// and any degree-`budget` fabric can take it.
///
/// The other route is **not on this ABI**: [`crate::embed`] is Rust-only, so a caller here cannot
/// compare the two and has to take the measurement on trust. That gap is real and worth stating
/// where someone would look for the missing function rather than leaving them to grep for it.
///
/// The original simulation is untouched and still owned by the caller. The returned one must be
/// freed with [`ft_free`] like any other.
///
/// NULL for a null handle or a budget below 3 — a path of copies spends one coupling at each end
/// and two in the middle, so it offers `c(d-2)+2` ports and that does not grow with `c` below 3.
///
/// Read the result back with [`ft_sparsify_project`], which reports which variables' copies
/// disagreed, and price it with [`ft_sparsify_offset`].
#[no_mangle]
pub extern "C" fn ft_sparsify(sim: *const Sim, budget: u32) -> *mut Sim {
    let Some(s) = (unsafe { sim.as_ref() }) else { return core::ptr::null_mut() };
    let Ok(sp) = crate::sparsify::sparsify(&s.graph, budget as usize) else {
        return core::ptr::null_mut();
    };
    let (copies, offset) = (sp.copies, sp.offset);
    let p = Sim::new(sp.graph, s.beta, s.seed);
    // SAFETY: `Sim::new` just handed back a live, uniquely-owned allocation.
    unsafe {
        (*p).copies = copies;
        (*p).sparsify_offset = offset;
    }
    p
}

/// Logical variables a sparsified simulation stands for, or 0 if it was not produced by
/// [`ft_sparsify`].
#[no_mangle]
pub extern "C" fn ft_sparsify_variables(sim: *const Sim) -> u32 {
    unsafe { sim.as_ref() }.map_or(0, |s| s.copies.len() as u32)
}

/// Copy the nodes representing logical variable `v` into `out`. Returns how many were written, or
/// the count needed when `out` is NULL.
#[no_mangle]
pub extern "C" fn ft_sparsify_copies(sim: *const Sim, v: u32, out: *mut u32, cap: u32) -> u32 {
    let Some(s) = (unsafe { sim.as_ref() }) else { return 0 };
    let Some(set) = s.copies.get(v as usize) else { return 0 };
    if out.is_null() {
        return set.len() as u32;
    }
    let n = set.len().min(cap as usize);
    unsafe { core::ptr::copy_nonoverlapping(set.as_ptr(), out, n) };
    n as u32
}

/// `E_logical = E_sparse + offset` when every copy set agrees. 0 for a simulation that was not
/// sparsified.
///
/// The copy couplings contribute the same constant in every agreeing state, so they order answers
/// identically and shift every energy by this amount. Reporting a sparsified energy without it
/// compares a number from one model against a number from another.
#[no_mangle]
pub extern "C" fn ft_sparsify_offset(sim: *const Sim) -> f64 {
    unsafe { sim.as_ref() }.map_or(0.0, |s| s.sparsify_offset)
}

/// Read the current state back as a LOGICAL one. Returns the number of variables whose copies
/// disagreed, and `0xFFFFFFFF` on a null handle or a buffer too small.
///
/// A variable whose copies disagree has not been assigned a value; the majority is written so the
/// caller still has a complete state to look at, and the count says how much of it to distrust. A
/// non-zero return means the copy coupling lost, and reading the state as an answer without
/// checking it is reading a majority vote as though it were one.
#[no_mangle]
pub extern "C" fn ft_sparsify_project(sim: *const Sim, out: *mut i8, cap: u32) -> u32 {
    let Some(s) = (unsafe { sim.as_ref() }) else { return u32::MAX };
    if s.copies.is_empty() || out.is_null() || (cap as usize) < s.copies.len() {
        return u32::MAX;
    }
    let mut broken = 0u32;
    for (v, set) in s.copies.iter().enumerate() {
        let up = set.iter().filter(|&&n| s.sampler_state[n as usize] > 0).count();
        if up != 0 && up != set.len() {
            broken += 1;
        }
        unsafe { *out.add(v) = if up * 2 >= set.len() { 1 } else { -1 } };
    }
    broken
}

#[cfg(test)]
mod sparsify_ffi_tests {
    use super::*;

    /// Rewriting a model across the boundary, and reading it back as the model it stands for.
    #[test]
    fn a_dense_model_crosses_the_abi_and_comes_back_logical() {
        // A 12-node clique: degree 11, well past the budget below.
        let b = ft_builder_new(12);
        for i in 0..12u32 {
            for j in (i + 1)..12 {
                ft_builder_couple(b, i, j, 1.0);
            }
        }
        let dense = ft_builder_build(b, 0.5, 7);
        assert_eq!(ft_len(dense), 12);
        assert_eq!(ft_sparsify_variables(dense), 0, "a model nobody sparsified has no copy map");
        assert_eq!(ft_sparsify_offset(dense), 0.0);

        let sparse = ft_sparsify(dense, 4);
        assert!(!sparse.is_null());
        assert!(ft_len(sparse) > 12, "splitting adds nodes");
        assert_eq!(ft_sparsify_variables(sparse), 12, "one entry per logical variable");
        assert!(ft_sparsify_offset(sparse) > 0.0, "copy couplings cost a constant");

        // The copy sets partition the sparsified nodes: every node belongs to exactly one variable.
        let mut owned = vec![0u32; ft_len(sparse) as usize];
        let mut total = 0usize;
        for v in 0..12u32 {
            let need = ft_sparsify_copies(sparse, v, core::ptr::null_mut(), 0);
            assert!(need >= 1, "variable {v} has at least one copy");
            let mut buf = vec![0u32; need as usize];
            assert_eq!(ft_sparsify_copies(sparse, v, buf.as_mut_ptr(), need), need);
            for n in buf {
                owned[n as usize] += 1;
            }
            total += need as usize;
        }
        assert_eq!(total, ft_len(sparse) as usize);
        assert!(owned.iter().all(|&c| c == 1), "every node belongs to exactly one variable");
        assert_eq!(ft_sparsify_copies(sparse, 12, core::ptr::null_mut(), 0), 0, "no variable 12");

        // Anneal it cold, then read it back as twelve logical spins.
        ft_anneal(sparse, 0.05, 8.0, 200, 40);
        let mut logical = vec![0i8; 12];
        let broken = ft_sparsify_project(sparse, logical.as_mut_ptr(), 12);
        assert!(broken != u32::MAX, "a real answer, not a refusal");
        assert!(logical.iter().all(|&s| s == 1 || s == -1));
        // Annealed at the derived copy strength, nothing should have come apart.
        assert_eq!(broken, 0, "{broken} of 12 variables had their copies disagree");

        // A buffer too small is refused rather than written past.
        assert_eq!(ft_sparsify_project(sparse, logical.as_mut_ptr(), 11), u32::MAX);
        assert_eq!(ft_sparsify_project(dense, logical.as_mut_ptr(), 12), u32::MAX,
                   "a model that was never sparsified has no logical state to project to");
        ft_free(sparse);
        ft_free(dense);
    }

    #[test]
    fn a_budget_that_cannot_work_is_null_rather_than_a_wrong_model() {
        let s = ft_ising2d_new(4, 1.0, 0.5, 1);
        for budget in [0u32, 1, 2] {
            assert!(ft_sparsify(s, budget).is_null(), "budget {budget} cannot be met by splitting");
        }
        assert!(ft_sparsify(core::ptr::null(), 4).is_null());
        // A lattice already fits a budget of 4, so this is the identity and still valid.
        let same = ft_sparsify(s, 4);
        assert!(!same.is_null());
        assert_eq!(ft_len(same), ft_len(s));
        assert_eq!(ft_sparsify_offset(same), 0.0, "no copy edges to pay for");
        ft_free(same);
        ft_free(s);
    }
}

// ---- minor embedding ------------------------------------------------------------------------------
//
// The other route onto a sparse fabric, and until now the one this ABI did not have. `sparsify`
// rewrites a model to fit a degree budget with no machine involved; embedding PLACES the model as it
// stands onto one specific hardware graph, giving each variable a chain of physical sites.
//
// `examples/sparsify_vs_embed.rs` measures which is cheaper and the answer is not close -- placing
// wins wherever both apply. A caller on this ABI could previously only sparsify, and so had to take
// that measurement on trust. These close it.

/// Store a CLOSED-FORM structured clique embedding on `hardware`, if one is known for its topology.
///
/// Where [`ft_embed`] searches, this writes the answer down: a `K_n` clique embedding built from the
/// topology's own minor structure, with uniform chains and no search. Supported today for
/// **Pegasus** (`ft_pegasus_new`: `K_{12(m-2)}`, chains `m+1` -- `K_168` on the Advantage's P_16)
/// and **Zephyr** (`ft_zephyr_new`: `K_{2t*m}`, chains `m+1`).
///
/// The clique size is FIXED by the machine, not chosen: it is the largest this construction places,
/// `K_{2t*m}` on `Z_{m,t}`. `n_out`, when non-null, receives it. The placement is stored on
/// `logical` exactly as [`ft_embed`] stores its own — so the caller builds their `K_n` problem as
/// `logical`, and `ft_embed_apply`, `ft_unembed` and the `ft_embed_*` accessors then read it back
/// unchanged. (The placement is topology-only, so `logical` need not be a clique for this call to
/// succeed; it is the caller's problem to solve on those `n` variables.)
///
/// Returns 1 on success, 0 on a null handle or a topology with no known construction — in which case
/// [`ft_embed`] is the fallback, and `ft_site_lower_bound` still answers whether any embedding of a
/// given clique can exist.
#[no_mangle]
pub extern "C" fn ft_clique_embed(logical: *mut Sim, hardware: *const Sim, n_out: *mut u32) -> u32 {
    let Some(hw) = (unsafe { hardware.as_ref() }) else { return 0 };
    let Some(lg) = (unsafe { logical.as_mut() }) else { return 0 };
    // The construction needs the DEVICE parameters, which the graph alone does not carry -- a
    // Zephyr graph and an arbitrary graph of the same size are indistinguishable here. So it is
    // gated on the vendor numbering being present and the site count matching a Z_{m,t}. This is
    // deliberately conservative: a graph that merely looks Zephyr-shaped is refused rather than
    // mis-embedded, because Embedding::verify would catch a wrong guess but a wrong guess should
    // not reach it.
    let Some(e) = structured_clique_for(hw) else { return 0 };
    let n = e.chains.len() as u32;
    if !n_out.is_null() {
        unsafe { *n_out = n };
    }
    lg.emb = Some(e);
    1
}

/// The structured clique for a simulation, when its shape and numbering say it is a device topology.
///
/// Recovers the device parameter from the site count -- |Z_{m,4}| = 16m(2m+1), |P_m| fabric =
/// 8(m-1)(3m-1) -- and requires an exact match AND the vendor numbering, so an arbitrary graph of
/// the same size cannot be mistaken for a machine. The last line of defence is unconditional
/// either way: the construction must verify against THIS graph before it is returned.
fn structured_clique_for(s: &Sim) -> Option<crate::embed::Embedding> {
    if s.qubits.is_empty() {
        return None;
    }
    let n = s.graph.n;
    let sealed = |built: crate::embed::Embedding| -> Option<crate::embed::Embedding> {
        let k = built.chains.len();
        let mut gb = crate::graph::GraphBuilder::new(k);
        for i in 0..k {
            for j in (i + 1)..k {
                gb.couple(i, j, 1.0);
            }
        }
        built.verify(&gb.build(), &s.graph).ok().map(|()| built)
    };
    for m in 1..=64usize {
        if 16 * m * (2 * m + 1) == n {
            if let Some(e) = crate::embed::zephyr_clique(m, 4).and_then(sealed) {
                return Some(e);
            }
        }
        if m >= 3 && 8 * (m - 1) * (3 * m - 1) == n {
            if let Some(e) = crate::embed::pegasus_clique(m).and_then(sealed) {
                return Some(e);
            }
        }
        if 16 * m * (2 * m + 1) > n && 8 * m * (3 * m + 2) > n {
            break;
        }
    }
    None
}

/// Place `logical` onto `hardware`, storing the placement on `logical` for the accessors below.
///
/// Returns 1 on success, 0 on a null handle or when the search did not find a placement.
///
/// **Zero never means "impossible."** It means this heuristic did not find one, which is a fact
/// about the search. [`ft_site_lower_bound`] is the question with a proof behind it: when it
/// exceeds the machine's site count, no embedding exists at all, and asking it first costs nothing.
///
/// `rounds` of rip-up and reroute, 0 for the default; `budget` shortest-path searches before giving
/// up, 0 for the default. A large machine wants a larger budget — saying "no" is not free, and on a
/// hopeless dense input the unbounded search runs for minutes.
#[no_mangle]
pub extern "C" fn ft_embed(
    logical: *mut Sim,
    hardware: *const Sim,
    seed: u64,
    rounds: u32,
    budget: u64,
) -> u32 {
    let Some(hw) = (unsafe { hardware.as_ref() }) else { return 0 };
    let Some(lg) = (unsafe { logical.as_mut() }) else { return 0 };
    let rounds = if rounds == 0 { 10 } else { rounds as usize };
    let budget = if budget == 0 { crate::embed::DEFAULT_SEARCH_BUDGET } else { budget };
    match crate::embed::embed_bounded(&lg.graph, &hw.graph, seed, rounds, budget) {
        Some(e) => {
            lg.emb = Some(e);
            1
        }
        None => 0,
    }
}

/// Physical sites the placement uses in total, or 0 if there is none.
#[no_mangle]
pub extern "C" fn ft_embed_sites(sim: *const Sim) -> u32 {
    unsafe { sim.as_ref() }
        .and_then(|s| s.emb.as_ref())
        .map_or(0, |e| e.chains.iter().map(|c| c.len()).sum::<usize>() as u32)
}

/// The longest chain, which is the number that decides whether an answer survives.
///
/// Sites are a budget and you either have them or you do not. A chain is a FAILURE MODE: it is held
/// together by a coupling, and when that coupling loses, the sites of one variable disagree and the
/// variable has no value at all. Halving this is worth more than halving [`ft_embed_sites`].
#[no_mangle]
pub extern "C" fn ft_embed_longest(sim: *const Sim) -> u32 {
    unsafe { sim.as_ref() }
        .and_then(|s| s.emb.as_ref())
        .map_or(0, |e| e.chains.iter().map(|c| c.len()).max().unwrap_or(0) as u32)
}

/// Copy the sites holding logical variable `v` into `out`; entries written, or the count needed
/// when `out` is NULL.
#[no_mangle]
pub extern "C" fn ft_embed_chain(sim: *const Sim, v: u32, out: *mut u32, cap: u32) -> u32 {
    let Some(e) = unsafe { sim.as_ref() }.and_then(|s| s.emb.as_ref()) else { return 0 };
    let Some(chain) = e.chains.get(v as usize) else { return 0 };
    if out.is_null() {
        return chain.len() as u32;
    }
    let n = chain.len().min(cap as usize);
    for (i, &site) in chain.iter().take(n).enumerate() {
        unsafe { *out.add(i) = site as u32 };
    }
    n as u32
}

/// The fewest sites ANY embedding of `logical` onto `hardware` could use.
///
/// A proof, not a heuristic. A chain of `L` sites on a degree-`d` machine offers at most
/// `L(d−2) + 2` ports, so a variable of degree `k` needs `⌈(k−2)/(d−2)⌉` sites however cleverly it
/// is placed. When the sum exceeds the machine, **no embedding exists** — and this answers in
/// microseconds where [`ft_embed`] would spend its whole budget discovering the same thing.
#[no_mangle]
pub extern "C" fn ft_site_lower_bound(logical: *const Sim, hardware: *const Sim) -> u32 {
    let (Some(lg), Some(hw)) = (unsafe { logical.as_ref() }, unsafe { hardware.as_ref() }) else {
        return 0;
    };
    crate::embed::site_lower_bound(&lg.graph, &hw.graph) as u32
}

/// Build the model to actually RUN on the hardware, from a placement already found.
///
/// The result is a simulation over the hardware's sites, with each chain bound by a coupling strong
/// enough to hold it. `chain_strength` of 0 takes the derived default — a multiple of the largest
/// coefficient in the logical model, which is the standard first guess. The placement rides along,
/// so [`ft_unembed`] works on the result.
///
/// NULL on a null handle or when `logical` carries no placement.
#[no_mangle]
pub extern "C" fn ft_embed_apply(
    logical: *const Sim,
    hardware: *const Sim,
    chain_strength: f64,
) -> *mut Sim {
    let (Some(lg), Some(hw)) = (unsafe { logical.as_ref() }, unsafe { hardware.as_ref() }) else {
        return core::ptr::null_mut();
    };
    let Some(e) = lg.emb.as_ref() else { return core::ptr::null_mut() };
    let out = crate::embed::apply_with(&lg.graph, &hw.graph, e, chain_strength);
    let embedding = out.embedding;
    let p = Sim::new(out.graph, lg.beta, lg.seed);
    // SAFETY: `Sim::new` just handed back a live, uniquely-owned allocation.
    unsafe { (*p).emb = Some(embedding) };
    p
}

/// Read an embedded state back as a LOGICAL one, returning how many chains BROKE.
///
/// `0xFFFFFFFF` — not 0, which is a valid count — on a null handle, a simulation carrying no
/// placement, or a buffer smaller than the logical variable count.
///
/// A variable whose chain broke has two values at once and therefore none. The majority is written
/// so there is still a complete state to look at, and the count says how much of it to distrust:
/// non-zero means the chain coupling lost to the problem, and the answer is a stronger coupling or
/// a shorter chain.
#[no_mangle]
pub extern "C" fn ft_unembed(sim: *const Sim, out: *mut i8, cap: u32) -> u32 {
    let Some(s) = (unsafe { sim.as_ref() }) else { return u32::MAX };
    let Some(e) = s.emb.as_ref() else { return u32::MAX };
    if out.is_null() || (cap as usize) < e.chains.len() {
        return u32::MAX;
    }
    let (state, broken) = crate::embed::unembed(e, &s.sampler_state);
    for (i, &v) in state.iter().enumerate() {
        unsafe { *out.add(i) = v };
    }
    broken.len() as u32
}

#[cfg(test)]
mod free_energy_ffi_tests {
    use super::*;

    /// The three ABI routes agree with the closed form, and the bound holds.
    #[test]
    fn ln_z_crosses_the_boundary_three_ways() {
        let s = ft_ising2d_new(4, 1.0, 0.5, 3); // a 4x4 torus, 16 spins
        let beta = 0.5;
        let truth = crate::free_energy::exact_log_z(unsafe { &(*s).graph }, beta);
        let exact = ft_ln_z_exact(s, beta);
        assert!((exact - truth).abs() < 1e-9, "elimination {exact} vs enumeration {truth}");

        assert!(ft_ln_z_ais_lower(s, 0.05).is_nan(), "no run yet");
        let a = ft_ln_z_ais(s, beta, 0, 0, 0);
        assert!((a - truth).abs() < 0.3, "ais {a} vs {truth}");
        assert!(ft_ln_z_ais_lower(s, 1e-6) <= truth);
        assert!(ft_ln_z_ais_ess(s) > 8.0);
        assert!(ft_ln_z_ais_lower(s, 0.0).is_nan() && ft_ln_z_ais_lower(s, 1.0).is_nan());

        let (mut lo, mut hi) = (f64::NAN, f64::NAN);
        let mid = ft_ln_z_ti(s, beta, 16, 100, 500, 3.0, &mut lo, &mut hi);
        assert!(lo <= truth && truth <= hi, "[{lo}, {hi}] misses {truth}");
        assert!((mid - truth).abs() < 0.5);

        assert!(ft_ln_z_exact(s, -1.0).is_nan() && ft_ln_z_ais(s, 0.0, 0, 0, 0).is_nan());
        assert!(ft_ln_z_exact(core::ptr::null(), beta).is_nan());
        ft_free(s);
    }
}

#[cfg(test)]
mod embed_ffi_tests {
    use super::*;

    /// A structured Zephyr clique, across the boundary, reading back with no broken chains.
    #[test]
    fn a_structured_clique_places_on_zephyr_over_the_abi() {
        let hw = ft_zephyr_new(4, 4, 1.0, 0.5, 3); // Z_4: 576 sites
        assert!(!hw.is_null());
        assert_eq!(ft_len(hw), 576);

        // The caller's problem is a K_56 clique; the placement is stored on IT, as ft_embed does.
        let logical = {
            let b = ft_builder_new(56);
            for i in 0..56u32 {
                for j in (i + 1)..56 {
                    ft_builder_couple(b, i, j, 1.0);
                }
            }
            ft_builder_build(b, 0.5, 3)
        };
        let mut n: u32 = 0;
        assert_eq!(ft_clique_embed(logical, hw, &mut n), 1);
        assert_eq!(n, 56, "K_{{2t(2m-1)}} = K_56 on Z_4, the busclique frontier size");
        assert_eq!(ft_embed_sites(logical), 56 * 5, "56 chains of 5");
        assert_eq!(ft_embed_longest(logical), 5, "uniform m+1 = 5");

        // Build the runnable model, anneal it, read back the 56 logical spins.
        let embedded = ft_embed_apply(logical, hw, 0.0);
        assert!(!embedded.is_null());
        ft_anneal(embedded, 0.05, 8.0, 300, 40);
        let mut out = vec![0i8; 56];
        let broken = ft_unembed(embedded, out.as_mut_ptr(), 56);
        assert!(broken != u32::MAX);
        assert!(out.iter().all(|&s| s == 1 || s == -1));

        ft_free(embedded);
        ft_free(logical);
        ft_free(hw);
    }

    /// The Advantage fabric places its structured clique over the boundary too.
    #[test]
    fn a_structured_clique_places_on_pegasus_over_the_abi() {
        let hw = ft_pegasus_new(4, 1.0, 0.5, 3); // P_4 fabric: 8*3*11 = 264 sites
        assert!(!hw.is_null());
        let lg = {
            let b = ft_builder_new(28);
            for i in 0..28u32 {
                for j in (i + 1)..28 {
                    ft_builder_couple(b, i, j, 1.0);
                }
            }
            ft_builder_build(b, 0.5, 3)
        };
        let mut n: u32 = 0;
        assert_eq!(ft_clique_embed(lg, hw, &mut n), 1);
        assert_eq!(n, 28, "K_{{12(m-2)+4}} = K_28 on P_4");
        assert_eq!(ft_embed_longest(lg), 5, "ells at m+1 = 5; the universal wires are shorter");
        ft_free(lg);
        ft_free(hw);
    }

    /// A graph that is not a device topology has no known construction, and is refused, not guessed.
    #[test]
    fn a_plain_graph_has_no_structured_clique() {
        let s = ft_ising2d_new(8, 1.0, 0.5, 1); // 64 sites, no vendor numbering
        let lg = ft_ising2d_new(2, 1.0, 0.5, 1);
        let mut n: u32 = 7;
        assert_eq!(ft_clique_embed(lg, s, &mut n), 0, "a lattice is not Zephyr-shaped");
        assert_eq!(ft_clique_embed(lg, s, core::ptr::null_mut()), 0);
        assert_eq!(ft_clique_embed(core::ptr::null_mut(), s, &mut n), 0);
        assert_eq!(ft_clique_embed(lg, core::ptr::null(), &mut n), 0);
        ft_free(lg);
        ft_free(s);
    }

    fn clique_sim(k: u32, beta: f64) -> *mut Sim {
        let b = ft_builder_new(k);
        for i in 0..k {
            for j in (i + 1)..k {
                ft_builder_couple(b, i, j, 1.0);
            }
        }
        ft_builder_build(b, beta, 7)
    }

    /// Place a clique on a real machine, run the placed model, and read it back by variable.
    #[test]
    fn a_model_places_onto_pegasus_and_the_answer_comes_back_logical() {
        let logical = clique_sim(12, 0.5);
        let hw = ft_pegasus_new(6, 1.0, 0.5, 3);
        assert!(!hw.is_null());
        assert_eq!(ft_embed_sites(logical), 0, "nothing placed yet");
        assert_eq!(ft_embed_longest(logical), 0);

        // The proof-carrying question first: K_12 needs far fewer sites than P6 has.
        let lb = ft_site_lower_bound(logical, hw);
        assert!(lb >= 12, "at least one site per variable, got {lb}");
        assert!(lb <= ft_len(hw), "K_12 is not impossible on a 680-site machine");

        assert_eq!(ft_embed(logical, hw, 7, 0, 0), 1);
        let sites = ft_embed_sites(logical);
        let longest = ft_embed_longest(logical);
        assert!(sites >= lb, "a placement cannot beat the lower bound: {sites} < {lb}");
        assert!(longest >= 1 && (longest as usize) <= sites as usize);

        // Chains partition the sites they use: no site holds two variables.
        let mut seen = std::collections::BTreeSet::new();
        let mut total = 0usize;
        for v in 0..12u32 {
            let need = ft_embed_chain(logical, v, core::ptr::null_mut(), 0);
            assert!(need >= 1, "variable {v} has no chain");
            let mut buf = vec![0u32; need as usize];
            assert_eq!(ft_embed_chain(logical, v, buf.as_mut_ptr(), need), need);
            for s in buf {
                assert!(s < ft_len(hw), "site {s} is off the machine");
                assert!(seen.insert(s), "site {s} is in two chains");
            }
            total += need as usize;
        }
        assert_eq!(total, sites as usize);
        assert_eq!(ft_embed_chain(logical, 12, core::ptr::null_mut(), 0), 0, "no variable 12");

        // Now build the model that actually runs on the machine, and solve it.
        let embedded = ft_embed_apply(logical, hw, 0.0);
        assert!(!embedded.is_null());
        assert_eq!(ft_len(embedded), ft_len(hw), "it runs over the hardware's sites");
        ft_anneal(embedded, 0.05, 8.0, 300, 40);

        let mut out = vec![0i8; 12];
        let broken = ft_unembed(embedded, out.as_mut_ptr(), 12);
        assert!(broken != u32::MAX, "a real answer, not a refusal");
        assert!(out.iter().all(|&s| s == 1 || s == -1));
        assert!(broken <= 12);

        // A buffer too small, and a simulation with no placement, are refusals rather than writes.
        assert_eq!(ft_unembed(embedded, out.as_mut_ptr(), 11), u32::MAX);
        assert_eq!(ft_unembed(hw, out.as_mut_ptr(), 12), u32::MAX, "the machine holds no placement");

        ft_free(embedded);
        ft_free(hw);
        ft_free(logical);
    }

    /// The bound proves impossibility; the search only reports failure.
    ///
    /// A P2 has 40 sites at degree 13. A K_24 variable has degree 23, and a chain of L sites offers
    /// L(13-2)+2 ports, so it needs 2 sites -- 48 for the twenty-four of them, which is more than
    /// the machine has. No embedding exists, and the library says so in microseconds without
    /// searching. That is the whole point of having a bound beside a heuristic.
    #[test]
    fn the_lower_bound_answers_impossible_where_the_search_only_says_not_found() {
        let logical = clique_sim(24, 0.5);
        let hw = ft_pegasus_new(2, 1.0, 0.5, 1);
        assert!(!hw.is_null());
        assert_eq!(ft_len(hw), 40);
        let lb = ft_site_lower_bound(logical, hw);
        assert_eq!(lb, 48, "24 variables at 2 sites each");
        assert!(lb > ft_len(hw), "K_24 needs {lb} sites of 40, so no embedding exists");
        // And the search agrees, by failing -- but its failure is the weaker statement.
        assert_eq!(ft_embed(logical, hw, 1, 0, 5_000), 0);
        ft_free(hw);
        ft_free(logical);
    }

    #[test]
    fn null_and_unplaced_handles_are_inert() {
        let s = ft_ising2d_new(4, 1.0, 0.5, 1);
        assert_eq!(ft_embed(core::ptr::null_mut(), s, 1, 0, 0), 0);
        assert_eq!(ft_embed(s, core::ptr::null(), 1, 0, 0), 0);
        assert_eq!(ft_site_lower_bound(core::ptr::null(), s), 0);
        assert_eq!(ft_embed_sites(core::ptr::null()), 0);
        assert!(ft_embed_apply(s, s, 0.0).is_null(), "no placement to apply");
        ft_free(s);
    }
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

/// How many threads this machine can actually run at once, or 1 when that cannot be known.
///
/// So a caller does not have to guess. An 18-core machine running a sampler on one core is the
/// commonest way this library is left slow, and the fix is a number the caller has no way to obtain
/// from the C ABI otherwise. Returns 1 in a browser, which is the truth there.
#[no_mangle]
pub extern "C" fn ft_hardware_threads() -> u32 {
    #[cfg(not(target_arch = "wasm32"))]
    {
        std::thread::available_parallelism().map_or(1, |n| n.get() as u32)
    }
    #[cfg(target_arch = "wasm32")]
    {
        1
    }
}

/// Sweep across `threads` OS threads, returning total sweeps done. Same contract as [`ft_sweep`].
///
/// Within a colour class no two nodes are adjacent, so the class splits into disjoint chunks and
/// each thread reads other-colour spins nobody is writing. The result is bit-reproducible for a
/// fixed `(seed, threads)` -- and a DIFFERENT thread count is a different, equally valid sample
/// path, so record the thread count next to the seed or the run is not reproducible from what you
/// wrote down. [`ft_threads_used`] reports what actually ran.
///
/// `threads` of 0 means "ask the machine", which is [`ft_hardware_threads`].
#[no_mangle]
pub extern "C" fn ft_sweep_par(sim: *mut Sim, n: u32, threads: u32) -> u64 {
    let Some(s) = (unsafe { sim.as_mut() }) else { return 0 };
    let threads = if threads == 0 { ft_hardware_threads() } else { threads }.max(1) as usize;
    let mut smp = Sampler::new(
        &s.graph,
        s.beta,
        s.seed ^ s.sweeps_done.wrapping_mul(0x9E3779B97F4A7C15),
    );
    smp.s.copy_from_slice(&s.sampler_state);
    smp.sweeps_par(n as usize, threads, Some(&mut s.ledger));
    s.sampler_state.copy_from_slice(&smp.s);
    s.threads_used = smp.threads_used() as u32;
    s.sweeps_done += n as u64;
    s.sweeps_done
}

/// How many threads the last [`ft_sweep_par`] actually used, or 0 before one.
///
/// Not the number you passed in. A browser has no threads to spread across and answers 1 whatever
/// was asked, and a colour class with three nodes cannot occupy eight workers. A caller reporting
/// throughput per thread needs the number that ran, and this is the only place it exists.
#[no_mangle]
pub extern "C" fn ft_threads_used(sim: *const Sim) -> u32 {
    unsafe { sim.as_ref() }.map_or(0, |s| s.threads_used)
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
    // NaN on a null handle, for the reason spelled out on [`ft_energy`]: zero magnetisation is the
    // ordinary state of any unmagnetised model, so 0.0 cannot mean "there is no handle".
    unsafe { sim.as_ref() }.map_or(f64::NAN, |s| {
        s.sampler_state.iter().map(|&v| v as i64).sum::<i64>() as f64 / s.graph.n as f64
    })
}

/// Energy of the current state, or NaN if the handle is null.
///
/// NaN rather than 0.0, which is what this returned until it was noticed: zero is a legal energy —
/// it is the energy of any state of an empty model, and of a balanced one — so a caller could not
/// tell a null handle from an answer. Every later section of this file already answered NaN for a
/// real-valued result on a refusal; this one and [`ft_magnetization`] were the two that did not.
#[no_mangle]
pub extern "C" fn ft_energy(sim: *const Sim) -> f64 {
    unsafe { sim.as_ref() }.map_or(f64::NAN, |s| s.graph.energy(&s.sampler_state))
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
/// on success, 0 on a null handle or a degenerate request. Exactly [`ft_collect`] with no burn-in;
/// the states it drew are kept and reachable through the `ft_samples_*` accessors.
#[no_mangle]
pub extern "C" fn ft_certify(sim: *mut Sim, draws: u32, thin: u32) -> u32 {
    ft_collect(sim, 0, draws, thin)
}

/// Draw `draws` states `thin` sweeps apart after `burn_in`, keep them, and certify the run.
///
/// This is [`ft_certify`] with the burn-in exposed and the states kept. Read the states back with
/// the `ft_samples_*` accessors and the certificate with the `ft_cert_*` ones; both describe the
/// same run. Returns 1 on success, 0 on a null handle or fewer than 16 draws.
///
/// # It charges the device for the readback, and that is a change
///
/// The loop this replaced appended the sampler's state directly, which never touches the read
/// path, so every run certified over this ABI reported its readback energy as exactly zero. On a
/// Z1-class device a read is 1.692 pJ per node against 7.09 fJ per Gibbs cycle -- one read is
/// worth 239 updates -- so [`ft_ledger_joules_z1`] after this call is now LARGER than it was, and
/// the earlier figure was the one that was wrong.
#[no_mangle]
pub extern "C" fn ft_collect(sim: *mut Sim, burn_in: u32, draws: u32, thin: u32) -> u32 {
    let Some(s) = (unsafe { sim.as_mut() }) else { return 0 };
    if draws < 16 {
        return 0; // certifying 15 samples is theatre; certify::TooFewSamples says so too
    }
    let mut smp = Sampler::new(&s.graph, s.beta, s.seed ^ s.sweeps_done.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    smp.s.copy_from_slice(&s.sampler_state);
    let plan = crate::samples::Plan::new(burn_in as usize, draws as usize, thin.max(1) as usize);
    let set = smp.collect(&plan, Some(&mut s.ledger));
    s.sampler_state.copy_from_slice(&smp.s);
    s.sweeps_done += plan.sweeps() as u64;
    s.cert = Some(set.certificate(&s.graph).expect("collect returns a chain"));
    s.sm = Some(set);
    1
}

/// States held by the last [`ft_collect`]. Zero when nothing has been collected.
#[no_mangle]
pub extern "C" fn ft_samples_len(sim: *const Sim) -> u32 {
    unsafe { sim.as_ref() }.and_then(|s| s.sm.as_ref()).map_or(0, |m| m.len() as u32)
}

/// How many of those states were DISTINCT.
///
/// The number a sampler is usually not asked for and usually should be: a run returning 10,000
/// draws of which 3 are distinct has told you about 3 states, whatever its draw count says.
#[no_mangle]
pub extern "C" fn ft_samples_distinct(sim: *const Sim) -> u32 {
    unsafe { sim.as_ref() }.and_then(|s| s.sm.as_ref()).map_or(0, |m| m.distinct().len() as u32)
}

/// Lowest energy in the collected set, or NaN if nothing has been collected.
#[no_mangle]
pub extern "C" fn ft_samples_best_energy(sim: *const Sim) -> f64 {
    unsafe { sim.as_ref() }
        .and_then(|s| s.sm.as_ref())
        .and_then(|m| m.best().map(|(_, e)| e))
        .unwrap_or(f64::NAN)
}

/// Distinct states within `tol` of the lowest energy seen.
///
/// This is EVIDENCE of degeneracy and not a count of it: a chain proves the states it visited
/// exist and can prove nothing about the ones it did not. Only exhaustive enumeration counts a
/// ground manifold, and this ABI does not expose one.
#[no_mangle]
pub extern "C" fn ft_samples_degeneracy(sim: *const Sim, tol: f64) -> u32 {
    unsafe { sim.as_ref() }
        .and_then(|s| s.sm.as_ref())
        .map_or(0, |m| m.ground_states(tol.max(0.0)).len() as u32)
}

/// The slowest autocorrelation time the chain showed, which every estimate below is deflated by.
/// NaN when nothing has been collected.
#[no_mangle]
pub extern "C" fn ft_samples_chain_tau(sim: *const Sim) -> f64 {
    unsafe { sim.as_ref() }.and_then(|s| s.sm.as_ref()).map_or(f64::NAN, |m| m.chain_tau())
}

/// Copy state `k` into `out`, which must hold at least `cap` entries. Returns the number written.
///
/// With a NULL `out` it returns the width and writes nothing, so a caller can size its buffer.
#[no_mangle]
pub extern "C" fn ft_samples_state(sim: *const Sim, k: u32, out: *mut i8, cap: u32) -> u32 {
    let Some(m) = unsafe { sim.as_ref() }.and_then(|s| s.sm.as_ref()) else { return 0 };
    let Some(st) = m.states().get(k as usize) else { return 0 };
    if out.is_null() {
        return st.len() as u32;
    }
    let n = st.len().min(cap as usize);
    unsafe { core::ptr::copy_nonoverlapping(st.as_ptr(), out, n) };
    n as u32
}

/// Write `[value, stderr, ess, tau_int]` for one estimate into `out`, which must hold 4 doubles.
///
/// # The writes are UNALIGNED, deliberately
///
/// `copy_nonoverlapping` to a `*mut f64` requires the destination to be 8-byte aligned, and the
/// caller this was written for cannot promise that: the browser passes a pointer from
/// [`ft_scratch`], which hands back the buffer of a `Vec<u8>` and is therefore aligned to 1. The
/// same page already learned this on the other side of the boundary — it reads these bytes through
/// a `DataView` rather than a `Float64Array`, because the typed-array constructor REFUSES a
/// byte offset that is not a multiple of eight. An aligned write into a buffer with no alignment
/// is undefined behaviour that would work on every machine anyone tested it on.
fn write_estimate(
    e: Result<crate::samples::Estimate, crate::samples::Refused>,
    out: *mut f64,
) -> u32 {
    let Ok(e) = e else { return 0 };
    if out.is_null() {
        return 0;
    }
    let v = [e.value, e.stderr, e.ess, e.tau_int];
    for (k, x) in v.iter().enumerate() {
        unsafe { out.add(k).write_unaligned(*x) };
    }
    1
}

/// `<s_i>` with its error bar: writes `[value, stderr, ess, tau_int]` into `out` (4 doubles).
///
/// The standard error is `sqrt(var/ess)`, NOT `sqrt(var/N)`: chain draws are correlated and the
/// naive interval understates the error by `sqrt(2*tau)`. Measured against exact enumeration in
/// `examples/interval_calibration.rs`, the naive interval contains the true value for one site in
/// four on a chain with `tau = 32`, while announcing 95%.
///
/// Returns 0 if nothing has been collected, `i` is out of range, or `out` is NULL.
#[no_mangle]
pub extern "C" fn ft_samples_mean_spin(sim: *const Sim, i: u32, out: *mut f64) -> u32 {
    let Some(m) = unsafe { sim.as_ref() }.and_then(|s| s.sm.as_ref()) else { return 0 };
    if i as usize >= m.n_spins() {
        return 0;
    }
    write_estimate(m.mean_spin(i as usize), out)
}

/// `<s_i s_j>` with its error bar, in the same four-double layout as [`ft_samples_mean_spin`].
///
/// This and the single-site mean are the two moments contrastive divergence matches.
#[no_mangle]
pub extern "C" fn ft_samples_correlation(sim: *const Sim, i: u32, j: u32, out: *mut f64) -> u32 {
    let Some(m) = unsafe { sim.as_ref() }.and_then(|s| s.sm.as_ref()) else { return 0 };
    if i as usize >= m.n_spins() || j as usize >= m.n_spins() {
        return 0;
    }
    write_estimate(m.correlation(i as usize, j as usize), out)
}

/// `<E>` with its error bar, in the same four-double layout as [`ft_samples_mean_spin`].
///
/// The internal energy, which is the expectation this field asks for most and the one a single
/// returned state cannot give: [`ft_energy`] reports the energy of the ONE configuration the
/// machine is holding, and a draw from a distribution is not an estimate of its mean.
#[no_mangle]
pub extern "C" fn ft_samples_mean_energy(sim: *const Sim, out: *mut f64) -> u32 {
    let Some(m) = unsafe { sim.as_ref() }.and_then(|s| s.sm.as_ref()) else { return 0 };
    write_estimate(m.mean_energy(), out)
}

/// The order parameter `(1/n) sum_i <s_i>`, in the same four-double layout.
#[no_mangle]
pub extern "C" fn ft_samples_magnetization(sim: *const Sim, out: *mut f64) -> u32 {
    let Some(m) = unsafe { sim.as_ref() }.and_then(|s| s.sm.as_ref()) else { return 0 };
    write_estimate(m.magnetization(), out)
}

#[cfg(test)]
mod sample_tests {
    use super::*;

    /// The ABI must hand back estimates that agree with exact enumeration, and refuse cleanly
    /// where it has nothing to say.
    #[test]
    fn the_abi_returns_estimates_that_cover_the_exact_answer() {
        let sim = ft_ising2d_new(3, 1.0, 0.4, 7); // 9 spins: small enough to enumerate exactly
        assert_eq!(ft_samples_len(sim), 0, "nothing collected yet");
        assert!(ft_samples_best_energy(sim).is_nan());
        let mut out = [0.0f64; 4];
        assert_eq!(ft_samples_mean_spin(sim, 0, out.as_mut_ptr()), 0, "and no estimate either");

        assert_eq!(ft_collect(sim, 1_000, 8_000, 1), 1);
        assert_eq!(ft_samples_len(sim), 8_000);
        assert!(ft_samples_distinct(sim) > 1, "a chain that visits one state is not sampling");
        assert!(ft_samples_distinct(sim) <= 8_000);
        assert!(ft_samples_chain_tau(sim) >= 0.5);

        let g = unsafe { &*sim }.graph.as_ref();
        let truth = crate::samples::enumerate(g, 0.4).expect("9 spins enumerates");
        for i in 0..9u32 {
            assert_eq!(ft_samples_mean_spin(sim, i, out.as_mut_ptr()), 1);
            let e = crate::samples::Estimate { value: out[0], stderr: out[1], ess: out[2], tau_int: out[3] };
            let exact = truth.mean_spin(i as usize).unwrap().value;
            assert!(e.covers(exact), "site {i}: ABI {e} does not cover the enumerated {exact:.5}");
        }
        assert_eq!(ft_samples_correlation(sim, 0, 1, out.as_mut_ptr()), 1);
        assert_eq!(ft_samples_magnetization(sim, out.as_mut_ptr()), 1);
        // <E> is a mean over draws, checked against the exactly enumerated internal energy --
        // the expectation this field asks for most, and the one a single returned state cannot
        // give: `ft_energy` reports the energy of the ONE configuration the machine is holding.
        assert_eq!(ft_samples_mean_energy(sim, out.as_mut_ptr()), 1);
        let mean_e = crate::samples::Estimate { value: out[0], stderr: out[1], ess: out[2], tau_int: out[3] };
        assert!(mean_e.stderr > 0.0, "an error bar of exactly zero on a moving chain is a bug");
        let exact_e = truth.mean_energy().unwrap().value;
        assert!(mean_e.covers(exact_e), "<E>: ABI {mean_e} does not cover the enumerated {exact_e:.5}");
        assert!(out[1] > 0.0, "an error bar of exactly zero on a moving chain is a bug");

        // Out of range and null are refusals, not answers.
        assert_eq!(ft_samples_mean_spin(sim, 9, out.as_mut_ptr()), 0);
        assert_eq!(ft_samples_correlation(sim, 0, 99, out.as_mut_ptr()), 0);
        assert_eq!(ft_samples_mean_spin(sim, 0, core::ptr::null_mut()), 0);
        assert_eq!(ft_samples_len(core::ptr::null()), 0);

        // States come back at the graph's width, and a NULL buffer reports that width.
        assert_eq!(ft_samples_state(sim, 0, core::ptr::null_mut(), 0), 9);
        let mut st = [0i8; 9];
        assert_eq!(ft_samples_state(sim, 0, st.as_mut_ptr(), 9), 9);
        assert!(st.iter().all(|&v| v == 1 || v == -1));
        assert_eq!(ft_samples_state(sim, 8_000, st.as_mut_ptr(), 9), 0, "no such draw");

        ft_free(sim);
    }

    /// The browser hands these functions a pointer with no alignment, and they must survive it.
    ///
    /// `ft_scratch` returns the buffer of a `Vec<u8>`, aligned to ONE. An aligned `f64` write into
    /// that is undefined behaviour of the kind that works on every machine anyone tests it on, so
    /// the writes are unaligned; this deliberately hands over a pointer that is NOT 8-byte aligned
    /// and requires the same four numbers back.
    #[test]
    fn an_estimate_writes_correctly_to_a_pointer_with_no_alignment() {
        let sim = ft_ising2d_new(3, 1.0, 0.4, 11);
        assert_eq!(ft_collect(sim, 200, 1_000, 1), 1);

        let mut aligned = [0.0f64; 4];
        assert_eq!(ft_samples_mean_spin(sim, 0, aligned.as_mut_ptr()), 1);

        // Land the destination four bytes off an eight-byte boundary, which is the worst case a
        // byte buffer can produce.
        let mut raw = [0u8; 48];
        let base = raw.as_mut_ptr();
        let off = (4 + 8 - (base as usize % 8)) % 8;
        let odd = unsafe { base.add(off) }.cast::<f64>();
        assert_ne!(odd as usize % 8, 0, "this test is vacuous unless the pointer is misaligned");
        assert_eq!(ft_samples_mean_spin(sim, 0, odd), 1);

        for k in 0..4 {
            let got = unsafe { odd.add(k).read_unaligned() };
            let want = aligned[k];
            assert!(
                got == want || (got.is_nan() && want.is_nan()),
                "slot {k}: unaligned wrote {got} where aligned wrote {want}"
            );
        }
        ft_free(sim);
    }

    /// The defect the collection path was built to close: certifying used to be free.
    #[test]
    fn certifying_now_charges_for_the_readback_it_performs() {
        let sim = ft_ising2d_new(8, 1.0, 0.5, 3);
        let before = ft_ledger_joules_z1(sim);
        assert_eq!(ft_certify(sim, 200, 1), 1);
        let after = ft_ledger_joules_z1(sim);

        // 200 draws of 64 nodes at 1.692 pJ is 21.7 nJ of readback; 200 sweeps of 64 nodes at
        // 7.09 fJ is 0.09 nJ of sampling. If this ABI ever stops charging for reads again, the
        // total drops by more than two orders of magnitude and this catches it.
        let reads = 200.0 * 64.0 * crate::ledger::Z1_SPICE.e_read;
        assert!(
            after - before > 0.9 * reads,
            "certify charged {:.3e} J where readback alone is {reads:.3e} J",
            after - before
        );
        assert_eq!(ft_samples_len(sim), 200, "and the states it read are kept");
        ft_free(sim);
    }
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

/// Exact single-site marginals `P(s_i = +1)`, written into `out`.
///
/// Returns 1 on success, 0 on a null handle, a wrong length, or a graph wider than `max_width`.
///
/// This is the referee. A sampler's histogram can be compared against these on a graph far past
/// where enumeration stops -- a 42-spin strip is 2^42 states and width 3 -- which is the only way
/// to check a sampler at a size anyone actually runs. The exhaustive referee and the certificate compare
/// against exhaustive enumeration and stop at about twenty spins; this does not.
///
/// COST: `2n` eliminations, so `O(n * 2^width)` rather than the single `O(2^width)` of
/// [`ft_exact_log_z`]. Check [`ft_exact_width`] first.
#[no_mangle]
pub extern "C" fn ft_exact_marginals(
    sim: *const Sim,
    beta: f64,
    max_width: u32,
    out: *mut f64,
    len: u32,
) -> u32 {
    let Some(s) = (unsafe { sim.as_ref() }) else { return 0 };
    if out.is_null() || len as usize != s.graph.n || !beta.is_finite() {
        return 0;
    }
    let el = crate::exact::Elimination { max_width: max_width as usize };
    match el.marginals(&s.graph, beta) {
        Ok(m) => {
            unsafe { core::ptr::copy_nonoverlapping(m.as_ptr(), out, m.len()) };
            1
        }
        Err(_) => 0,
    }
}

#[cfg(test)]
mod exact_marginal_ffi {
    use super::*;

    #[test]
    fn the_marginals_are_a_referee_a_sampler_can_be_checked_against() {
        // A 5-ring, small enough that the ABI's own sampler can be run against it here.
        let b = ft_builder_new(5);
        for i in 0..5u32 {
            assert_eq!(ft_builder_couple(b, i, (i + 1) % 5, -1.0), 1);
        }
        assert_eq!(ft_builder_bias(b, 0, 0.4), 1);
        let sim = ft_builder_build(b, 0.7, 11);
        let n = ft_len(sim) as usize;

        let mut m = vec![0.0f64; n];
        assert_eq!(ft_exact_marginals(sim, 0.7, 24, m.as_mut_ptr(), n as u32), 1);
        assert!(m.iter().all(|p| (0.0..=1.0).contains(p)), "{m:?}");
        // The biased node must lean the way its field points, or the sign convention is inverted
        // and every comparison built on this would inherit it.
        assert!(m[0] > 0.5, "a positive field must favour +1: {}", m[0]);

        ft_sweep(sim, 2000);
        let draws = 20_000;
        let mut up = vec![0u64; n];
        for _ in 0..draws {
            ft_sweep(sim, 1);
            let st = unsafe { core::slice::from_raw_parts(ft_spins(sim), n) };
            for i in 0..n {
                if st[i] == 1 {
                    up[i] += 1;
                }
            }
        }
        for i in 0..n {
            let got = up[i] as f64 / draws as f64;
            assert!((got - m[i]).abs() < 0.03, "node {i}: sampled {got:.4} vs exact {:.4}", m[i]);
        }
        ft_free(sim);
    }

    #[test]
    fn a_wrong_length_or_a_too_wide_graph_is_refused_rather_than_partly_written() {
        let sim = ft_ising2d_new(4, 1.0, 0.5, 1);
        let n = ft_len(sim) as usize;
        let mut m = vec![0.0f64; n];
        assert_eq!(ft_exact_marginals(sim, 0.5, 24, m.as_mut_ptr(), (n - 1) as u32), 0);
        assert_eq!(ft_exact_marginals(sim, 0.5, 24, core::ptr::null_mut(), n as u32), 0);
        assert_eq!(ft_exact_marginals(sim, f64::NAN, 24, m.as_mut_ptr(), n as u32), 0);
        // max_width 0 refuses everything that is not already trivial.
        assert_eq!(ft_exact_marginals(sim, 0.5, 0, m.as_mut_ptr(), n as u32), 0);
        assert!(m.iter().all(|&x| x == 0.0), "a refusal must not write");
        assert_eq!(ft_exact_marginals(core::ptr::null(), 0.5, 24, m.as_mut_ptr(), n as u32), 0);
        ft_free(sim);
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

use crate::model::{Compiled, Constraint, Expr, Lit, Model, Rel, Sense, Solution};

/// A model under construction, plus whatever it last compiled and solved.
pub struct ModelHandle {
    model: Model,
    compiled: Option<Compiled>,
    solution: Option<Solution>,
    last_error: String,
    /// Literals accumulating for the next variable-length counting constraint.
    lits: Vec<Lit>,
    /// The coefficient each pending literal carries, kept exactly in step with `lits`.
    ///
    /// A parallel vector rather than a second list: `ft_model_lit` pushes 1.0, so a list built the
    /// way every existing caller builds it closes as a WEIGHTED row that means exactly the
    /// counting row -- and no binding has to learn two ways to append a literal.
    coeffs: Vec<f64>,
    cert: Option<crate::certify::Certificate>,
    /// Every answer the last solve produced, not only the best. See [`ft_model_optima`].
    answers: Vec<crate::model::Solution>,
}

#[no_mangle]
pub extern "C" fn ft_model_new() -> *mut ModelHandle {
    Box::into_raw(Box::new(ModelHandle {
        model: Model::new(),
        compiled: None,
        solution: None,
        last_error: String::new(),
        lits: Vec::new(),
        coeffs: Vec::new(),
        cert: None,
        answers: Vec::new(),
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
    // FIVE CAUSES USED TO SHARE ONE SILENT `return 0`. Python surfaced them all as the fallback
    // text "the library refused that objective", which names nothing a caller can act on, while
    // this function's own sibling `ft_model_objective_product` sets a reason for every refusal.
    let (x, y) = match (var_of(h, a), var_of(h, b)) {
        (Some(x), Some(y)) => (x, y),
        (None, _) => {
            h.last_error = format!("there is no variable {a} in this model");
            return 0;
        }
        (_, None) => {
            h.last_error = format!("there is no variable {b} in this model");
            return 0;
        }
    };
    if !coeff.is_finite() {
        h.last_error = format!("an objective coefficient must be a real number, not {coeff}");
        return 0;
    }
    // THE SAME VARIABLE TWICE IS LEGAL AND USED TO BE REFUSED HERE.
    //
    // `a == b` was rejected outright, and the Rust path has always handled it correctly: the square
    // of an indicator IS the indicator, so `5.0 * x.is(1) * x.is(1)` scores 5 when x is 1; and
    // `x.is(1) * x.is(2)` contributes 0, because one variable cannot hold two values. Both compile
    // and solve to the right answer through `Model` today. The guard made a term expressible from
    // Rust and inexpressible from C, Python, Zig and Julia -- and said nothing about why.
    if !check_value(h, x, av) {
        h.last_error = format!("{av} is not a value variable {a} can take");
        return 0;
    }
    if !check_value(h, y, bv) {
        h.last_error = format!("{bv} is not a value variable {b} can take");
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
    h.coeffs.clear();
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
            // Answers belong to the model they were solved from. Recompiling means the constraints
            // or the objective moved, so a kept list would have `ft_model_optima` counting the
            // optima of a model that no longer exists -- the same stale-surface failure the
            // certificate accessors and the workbench's panels each learned separately.
            h.answers.clear();
            h.solution = None;
            h.last_error.clear();
            n
        }
        Err(e) => {
            h.last_error = e.to_string();
            h.compiled = None;
            h.answers.clear();
            h.solution = None;
            0
        }
    }
}

/// Anneal the compiled model, keeping the best of `tries`. Returns 1 on success.
#[no_mangle]
pub extern "C" fn ft_model_solve(m: *mut ModelHandle, tries: u32) -> u32 {
    let Some(h) = (unsafe { m.as_mut() }) else { return 0 };
    let Some(c) = h.compiled.as_ref() else { return 0 };
    let (lo, hi, n, w) = crate::model::Compiled::DEFAULT_LADDER;
    // The default ladder, spelled out, so this keeps every try the way `ft_model_solve_with` does.
    // `solve_best_of` runs the same seeds on the same schedule and returns only the winner; going
    // through `solve_all_with` here is what makes `ft_model_optima` answer after EITHER entry
    // point. A surface where the same question works after one solve and silently returns zero
    // after the other is worse than one where it does not exist.
    let sched = crate::schedule::Schedule::geometric(lo, hi, n, w);
    let all = c.solve_all_with(&sched, tries.max(1) as u64);
    h.solution = Some(best_answer(&all));
    h.answers = all;
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
    // Every try, not only the winner. `solve_best_with` runs exactly these seeds and keeps one,
    // which answers "what should I do" and cannot answer "how many ways could I have done it" --
    // and a model with a symmetry has several. Picking the best from the kept list rather than
    // calling `solve_best_with` again means the two cannot disagree.
    let all = c.solve_all_with(&sched, tries.max(1) as u64);
    h.solution = Some(best_answer(&all));
    h.answers = all;
    1
}

/// The answer `solve_best_with` would have returned: feasible beats infeasible, then lowest energy.
fn best_answer(all: &[crate::model::Solution]) -> crate::model::Solution {
    let mut best = &all[0];
    for cand in &all[1..] {
        let better = match (best.feasible(), cand.feasible()) {
            (false, true) => true,
            (true, false) => false,
            _ => cand.energy < best.energy,
        };
        if better {
            best = cand;
        }
    }
    best.clone()
}

/// How many answers the last solve kept — one per try.
#[no_mangle]
pub extern "C" fn ft_model_answers(m: *const ModelHandle) -> u32 {
    unsafe { m.as_ref() }.map_or(0, |h| h.answers.len() as u32)
}

/// How many DISTINCT optimal assignments those answers found, within `tol` of the best.
///
/// The question a modeller has and no surface in this field answers: **how many different ways are
/// there to do the job**. A schedule that returns one answer cannot say whether it was the only
/// one, and a problem with a symmetry usually has several.
///
/// Distinctness is on the DECODED VALUES, never on the spins — a compiled model carries slack and
/// ancilla bits no variable reads, and the count has to be a statement about the model rather than
/// about how the compiler chose to represent it.
///
/// This is **evidence**, not a count of the ground manifold: `tries` independent anneals prove the
/// optima they landed on exist and prove nothing about the ones they missed. Only feasible answers
/// are counted, because an assignment that breaks a hard row is not a way to do the job. `tol` is
/// on the compiled Ising energy, which folds in every penalty; `1e-9` is the value for exact ties.
#[no_mangle]
pub extern "C" fn ft_model_optima(m: *const ModelHandle, tol: f64) -> u32 {
    let Some(h) = (unsafe { m.as_ref() }) else { return 0 };
    let tol = if tol.is_finite() && tol >= 0.0 { tol } else { 0.0 };
    crate::model::distinct_optima(&h.answers, tol).len() as u32
}

#[cfg(test)]
mod optima_tests {
    use super::*;

    /// The question the node editor could not ask: how many ways are there to do the job.
    #[test]
    fn the_abi_counts_every_way_to_do_the_job_and_can_read_each_one() {
        let m = ft_model_new();
        // Three binaries, exactly one of them on. Three assignments, known in advance.
        let a = ft_model_binary(m);
        let b = ft_model_binary(m);
        let c = ft_model_binary(m);
        for v in [a, b, c] {
            assert_eq!(ft_model_lit(m, v, 1), 1);
        }
        assert_eq!(ft_model_close(m, 3, 0), 1); // kind 3 = exactly-one
        assert!(ft_model_compile(m) > 0);

        assert_eq!(ft_model_optima(m, 1e-9), 0, "nothing solved yet");
        assert_eq!(ft_model_answers(m), 0);

        assert_eq!(ft_model_solve_with(m, 40, 0.0, 0.0, 0, 0), 1);
        assert_eq!(ft_model_answers(m), 40, "one answer per try, kept");
        assert_eq!(ft_model_optima(m, 1e-9), 3, "exactly-one over three has three ways");

        // Each one reads back by name through the accessors that already exist, and each is a
        // DIFFERENT assignment with exactly one variable on.
        let mut seen = std::collections::BTreeSet::new();
        for i in 0..3u32 {
            assert_eq!(ft_model_select_optimum(m, i, 1e-9), 1);
            assert_eq!(ft_model_feasible(m), 1);
            let vals: Vec<i64> = (0..3).map(|v| ft_model_value(m, v)).collect();
            assert_eq!(vals.iter().filter(|&&x| x == 1).count(), 1, "exactly one on: {vals:?}");
            assert!(seen.insert(vals), "the same assignment was listed twice");
        }
        assert_eq!(ft_model_select_optimum(m, 3, 1e-9), 0, "there is no fourth");

        // The solve's own answer is ONE OF the optima -- not necessarily the head of the list.
        // These three tie on energy, so the list orders them by assignment while the solve returns
        // whichever seed reached the minimum first. An earlier version of this test asserted they
        // were the same and passed by coincidence until a colouring change moved the trajectory.
        assert_eq!(ft_model_solve_with(m, 40, 0.0, 0.0, 0, 0), 1);
        let solved = (0..3).map(|v| ft_model_value(m, v)).collect::<Vec<_>>();
        assert!(seen.contains(&solved), "solve returned {solved:?}, which is not among the optima");
        // And the list itself is deterministic: same tries, same order.
        let mut again = Vec::new();
        for i in 0..3u32 {
            assert_eq!(ft_model_select_optimum(m, i, 1e-9), 1);
            again.push((0..3).map(|v| ft_model_value(m, v)).collect::<Vec<_>>());
        }
        assert_eq!(again.len(), 3);
        assert!(again.windows(2).all(|w| w[0] != w[1]), "three distinct assignments");

        // Recompiling invalidates them: an optimum belongs to the model it was solved from.
        assert!(ft_model_compile(m) > 0);
        assert_eq!(ft_model_optima(m, 1e-9), 0);
        assert_eq!(ft_model_answers(m), 0);
        assert_eq!(ft_model_select_optimum(m, 0, 1e-9), 0);

        ft_model_free(m);
    }

    /// Both solve entry points must answer, or the surface is confidently inconsistent.
    #[test]
    fn the_default_ladder_keeps_its_tries_too() {
        let m = ft_model_new();
        let a = ft_model_binary(m);
        let b = ft_model_binary(m);
        for v in [a, b] {
            assert_eq!(ft_model_lit(m, v, 1), 1);
        }
        assert_eq!(ft_model_close(m, 3, 0), 1); // kind 3 = exactly-one
        assert!(ft_model_compile(m) > 0);
        assert_eq!(ft_model_solve(m, 24), 1);
        assert_eq!(ft_model_answers(m), 24);
        assert_eq!(ft_model_optima(m, 1e-9), 2, "exactly-one over two has two ways");

        // A single-method solve replaces the list rather than leaving the previous one behind.
        assert_eq!(ft_model_solve_by(m, 3, 0), 1, "branch and bound");
        assert_eq!(ft_model_answers(m), 1);
        assert_eq!(ft_model_optima(m, 1e-9), 1, "one run found one assignment");

        // A negative or NaN tolerance is coerced, not obeyed.
        assert_eq!(ft_model_optima(m, -1.0), 1);
        assert_eq!(ft_model_optima(m, f64::NAN), 1);
        assert_eq!(ft_model_optima(core::ptr::null(), 1e-9), 0);
        ft_model_free(m);
    }
}

/// Make optimum `i` the current answer, so `ft_model_value` and friends report it.
///
/// Enumerating the alternatives needs no second decode surface: select one, read it by name
/// through the accessors that already exist, and select the next.
///
/// # Index 0 is the best optimum, NOT necessarily the one the solve returned
///
/// The list is ordered by energy and then by the assignment itself, which makes it deterministic.
/// When several optima TIE on energy — which is the whole case this function exists for — the head
/// of the list is the lexicographically first of them, and the solve returned whichever seed
/// happened to reach the minimum first. Both are optimal and neither is more correct.
///
/// An earlier version of this documentation claimed selecting 0 put the handle back where it was.
/// It was true by coincidence on the graphs that were tested and stopped being true the moment the
/// colouring changed, which is how it was found. Re-solve if you need the solve's own answer.
///
/// Returns 1 on success, 0 for a null handle or an index past the count.
#[no_mangle]
pub extern "C" fn ft_model_select_optimum(m: *mut ModelHandle, i: u32, tol: f64) -> u32 {
    let Some(h) = (unsafe { m.as_mut() }) else { return 0 };
    let tol = if tol.is_finite() && tol >= 0.0 { tol } else { 0.0 };
    let opt = crate::model::distinct_optima(&h.answers, tol);
    match opt.into_iter().nth(i as usize) {
        Some(s) => {
            h.solution = Some(s);
            1
        }
        None => 0,
    }
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

/// Solve the compiled model by a chosen METHOD, rather than always annealing.
///
/// `method` is 0 anneal, 1 tabu, 2 breakout, 3 branch and bound. `effort` is the method's budget --
/// iterations for tabu and breakout, a node ceiling for branch -- and 0 takes a default.
///
/// Returns 1 on success, 0 on a null handle, an unknown method, or a model that has not compiled.
/// Read [`ft_model_proved`] afterwards: only branch can prove anything, and it is the reason this
/// exists. Every other solver in this crate takes a graph of spins, so the modelling layer -- the
/// one every document here tells a caller to reach for first -- was the one layer that could not
/// certify its own answer.
#[no_mangle]
pub extern "C" fn ft_model_solve_by(m: *mut ModelHandle, method: u32, effort: u64) -> u32 {
    let Some(h) = (unsafe { m.as_mut() }) else { return 0 };
    let Some(c) = h.compiled.as_ref() else {
        h.last_error = "compile the model before solving it".into();
        return 0;
    };
    let meth = match method {
        0 => crate::model::Method::Anneal,
        1 => crate::model::Method::Tabu {
            iterations: if effort == 0 { 50_000 } else { effort as usize },
        },
        2 => crate::model::Method::Breakout {
            iterations: if effort == 0 { 50_000 } else { effort as usize },
        },
        3 => crate::model::Method::Branch {
            max_nodes: if effort == 0 { 20_000_000 } else { effort },
        },
        other => {
            h.last_error =
                format!("unknown method {other}; 0 anneal, 1 tabu, 2 breakout, 3 branch");
            return 0;
        }
    };
    let sol = c.solve_by(meth, 1);
    // One run, one answer -- and `answers` is REPLACED rather than left alone. A stale list would
    // make `ft_model_optima` describe a previous solve of a possibly different model, which is the
    // shape of a surface that is confidently wrong.
    h.answers = vec![sol.clone()];
    h.solution = Some(sol);
    h.last_error.clear();
    1
}

/// Whether the last solve PROVED its answer optimal, rather than merely finding it.
///
/// Only [`ft_model_solve_by`] with method 3 can set it, and only when branch and bound exhausted
/// the tree inside its node budget.
///
/// **Read it together with [`ft_model_feasible`].** Branch proves a statement about the compiled
/// energy; it becomes a statement about the caller's MODEL exactly when the answer is also feasible,
/// because a feasible assignment pays no penalty and its compiled energy is the objective plus a
/// constant. Proved and feasible is a real optimality proof for the model as written, and the
/// argument uses nothing about the penalty being large enough. Proved and INFEASIBLE proves
/// something else and still useful: the penalty is too small, and no longer search will fix it.
#[no_mangle]
pub extern "C" fn ft_model_proved(m: *const ModelHandle) -> u32 {
    u32::from(
        unsafe { m.as_ref() }
            .and_then(|h| h.solution.as_ref())
            .is_some_and(|s| s.proved_optimal),
    )
}

/// The objective's value in the modeller's own units, in the direction they wrote it.
///
/// NaN when no objective was written, when both senses were used and there is no single direction
/// to report, or when a variable did not decode and there is only half an answer to score.
///
/// Distinct from [`ft_model_energy`], which is the compiled Ising energy with every penalty and
/// the constant folded in. That number is about SPINS: it compares two answers to one model and
/// nothing else, and it moves when the penalty does. A modeller who wrote `maximize 5*mon + 4*tue`
/// reads their schedule's worth here and reads a number in the hundreds there.
#[no_mangle]
pub extern "C" fn ft_model_objective(m: *const ModelHandle) -> f64 {
    unsafe { m.as_ref() }
        .and_then(|h| h.solution.as_ref())
        .and_then(|s| s.objective)
        .unwrap_or(f64::NAN)
}

/// Whether the answer carries an objective value at all, so a caller need not test for NaN.
#[no_mangle]
pub extern "C" fn ft_model_has_objective(m: *const ModelHandle) -> u32 {
    u32::from(
        unsafe { m.as_ref() }
            .and_then(|h| h.solution.as_ref())
            .is_some_and(|s| s.objective.is_some()),
    )
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
    h.coeffs.clear();
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
    h.coeffs.push(1.0);
    1
}

/// Append "`var` takes `value`" to the pending list. Refuses a value the variable cannot take.
#[no_mangle]
pub extern "C" fn ft_model_lit(m: *mut ModelHandle, var: u32, value: i64) -> u32 {
    let Some(h) = (unsafe { m.as_mut() }) else { return 0 };
    match var_of(h, var) {
        Some(x) if check_value(h, x, value) => {
            h.lits.push(Lit::Is(x, value));
            h.coeffs.push(1.0);
            1
        }
        _ => 0,
    }
}

/// Append "`var` takes `value`, weighted by `coeff`" to the pending list.
///
/// The only thing this adds over [`ft_model_lit`] is the coefficient, and it is the whole reason
/// [`ft_model_close_linear`] exists: `3a + 4b + 5c <= 7` could not be stated anywhere in this
/// library, because every counting constraint counts UNWEIGHTED literals and the LP reader refused
/// a weighted row by name. A list built with `ft_model_lit` carries a coefficient of 1 on every
/// literal, so the two can be mixed freely and an unweighted list closed as a linear row means
/// exactly the counting row it looks like.
///
/// Refuses a value the variable cannot take, and a coefficient that is not a real number.
#[no_mangle]
pub extern "C" fn ft_model_lit_weighted(
    m: *mut ModelHandle,
    var: u32,
    value: i64,
    coeff: f64,
) -> u32 {
    let Some(h) = (unsafe { m.as_mut() }) else { return 0 };
    if !coeff.is_finite() {
        h.last_error = format!("a linear coefficient must be a real number, not {coeff}");
        return 0;
    }
    match var_of(h, var) {
        Some(x) if check_value(h, x, value) => {
            h.lits.push(Lit::Is(x, value));
            h.coeffs.push(coeff);
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
/// `kind` is 0 for exactly, 1 for at-most, 2 for at-least, 3 for exactly-one, 4 for at-most-one,
/// 5 for all-different. The last three ignore `k`; 5 reads the VARIABLES out of the pending
/// literals and ignores their values, so `ft_model_var` is the natural way to build its list.
/// Clears the pending list whether it succeeds or not, so a refused constraint cannot silently join
/// the next one.
///
/// Kind 5 shipped without appearing in this comment or in the refusal below, so the ABI's own
/// error message told callers that the constraint it implements does not exist.
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
            h.coeffs.clear();
            return 0;
        }
    }
    let lits = core::mem::take(&mut h.lits);
    h.coeffs.clear();
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
                 3 exactly-one, 4 at-most-one, 5 all-different"
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

/// Close the pending literal list as a **weighted linear row**: `Σ wᵢ·lᵢ ≤ rhs`, `≥` or `=`.
///
/// `rel` is 0 for `<=`, 1 for `>=`, 2 for `=`. Each literal carries the coefficient it was appended
/// with -- 1.0 through [`ft_model_lit`], anything finite through [`ft_model_lit_weighted`].
///
/// The constraint none of the counting kinds can express, and the one whose absence made the LP
/// reader tell callers to "add it to the objective" -- which is not a constraint, so
/// [`ft_model_feasible`] and the violation list stop knowing about the row.
///
/// Costs `ceil(log2(S+1))` slack spins for an inequality and none for an equality; see
/// `Constraint::Linear` for the whole cost model and for what it refuses. Clears the pending list
/// whether it succeeds or not, so a refused row cannot silently join the next one. The refusals
/// that depend on the row's arithmetic -- a fractional coefficient on an inequality, a row nothing
/// can satisfy -- are raised by `ft_model_compile`, and `ft_model_error` carries the reason.
#[no_mangle]
pub extern "C" fn ft_model_close_linear(m: *mut ModelHandle, rel: u32, rhs: f64) -> u32 {
    close_linear(m, rel, rhs, None)
}

/// Close the pending literal list as a **soft** weighted linear row, at a price.
///
/// Same `rel` codes as [`ft_model_close_linear`]. The difference is what breaking it means: a hard
/// row says which answers are answers at all, so breaking one makes [`ft_model_feasible`] zero; a
/// soft one is a preference with a number on it, and breaking it costs `weight × amount²` in the
/// modeller's own units -- exactly the energy the compiled row contributes -- and leaves the answer
/// feasible. [`ft_model_soft_cost`] totals what was traded.
#[no_mangle]
pub extern "C" fn ft_model_close_linear_soft(
    m: *mut ModelHandle,
    rel: u32,
    rhs: f64,
    weight: f64,
) -> u32 {
    close_linear(m, rel, rhs, Some(weight))
}

fn close_linear(m: *mut ModelHandle, rel: u32, rhs: f64, soft: Option<f64>) -> u32 {
    let Some(h) = (unsafe { m.as_mut() }) else { return 0 };
    let lits = core::mem::take(&mut h.lits);
    let coeffs = core::mem::take(&mut h.coeffs);
    debug_assert_eq!(lits.len(), coeffs.len());
    if let Some(w) = soft {
        if !(w > 0.0) || !w.is_finite() {
            h.last_error = format!("a soft constraint needs a positive price, not {w}");
            return 0;
        }
    }
    if lits.is_empty() {
        h.last_error = "a linear row needs at least one term".into();
        return 0;
    }
    if !rhs.is_finite() {
        h.last_error = format!("a right-hand side must be a real number, not {rhs}");
        return 0;
    }
    let relation = match rel {
        0 => Rel::Le,
        1 => Rel::Ge,
        2 => Rel::Eq,
        other => {
            h.last_error = format!("unknown relation {other}; 0 is <=, 1 is >=, 2 is =");
            return 0;
        }
    };
    let terms: Vec<(Lit, f64)> = lits.into_iter().zip(coeffs).collect();
    match soft {
        Some(w) => h.model.linear_soft(terms, relation, rhs, w),
        None => h.model.linear(terms, relation, rhs),
    };
    1
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

    /// The last error a model handle recorded, as text.
    fn model_error(m: *mut ModelHandle) -> String {
        let mut buf = [0u8; 1024];
        let n = ft_model_error(m, buf.as_mut_ptr(), buf.len() as u32) as usize;
        String::from_utf8_lossy(&buf[..n.min(buf.len())]).into_owned()
    }

    /// The row that could not be stated across this boundary at all.
    ///
    /// Every counting kind counts UNWEIGHTED literals, so `3a + 4b + 5c <= 7` had no form here.
    #[test]
    fn a_weighted_linear_row_crosses_the_boundary_and_constrains_the_answer() {
        let m = ft_model_new();
        let v: Vec<u32> = (0..3).map(|_| ft_model_binary(m)).collect();
        for &i in &v {
            ft_model_objective_term(m, 1, 1.0, i, 1); // maximise how many are taken
        }
        for (i, w) in v.iter().zip([3.0, 4.0, 5.0]) {
            assert_eq!(ft_model_lit_weighted(m, *i, 1, w), 1);
        }
        assert_eq!(ft_model_lits(m), 3);
        assert_eq!(ft_model_close_linear(m, 0, 7.0), 1); // 0 is <=
        assert_eq!(ft_model_lits(m), 0, "closing clears the pending list");
        assert!(ft_model_compile(m) > 0);
        ft_model_solve(m, 32);
        assert_eq!(ft_model_feasible(m), 1, "{}", model_error(m));
        let taken: f64 = v
            .iter()
            .zip([3.0, 4.0, 5.0])
            .filter(|(i, _)| ft_model_value(m, **i) == 1)
            .map(|(_, w)| w)
            .sum();
        assert!(taken <= 7.0, "the row is a CONSTRAINT, not a preference: got {taken}");
        assert_eq!(taken, 7.0, "and 3 + 4 is the best that fits");
        ft_model_free(m);

        // A list built with the unweighted append means exactly the counting row it looks like.
        let m = ft_model_new();
        let a = ft_model_binary(m);
        let b = ft_model_binary(m);
        ft_model_lit(m, a, 1);
        ft_model_lit(m, b, 1);
        assert_eq!(ft_model_close_linear(m, 1, 2.0), 1, "a + b >= 2");
        assert!(ft_model_compile(m) > 0);
        ft_model_solve(m, 16);
        assert_eq!((ft_model_value(m, a), ft_model_value(m, b)), (1, 1));
        ft_model_free(m);

        // Every refusal returns zero and says why, rather than aborting or compiling something else.
        let m = ft_model_new();
        let a = ft_model_binary(m);
        assert_eq!(ft_model_lit_weighted(m, a, 1, f64::NAN), 0, "a coefficient must be real");
        assert_eq!(ft_model_close_linear(m, 0, 1.0), 0, "an empty row is not a row");
        ft_model_lit_weighted(m, a, 1, 2.5);
        assert_eq!(ft_model_close_linear(m, 7, 1.0), 0, "7 is not a relation");
        assert_eq!(ft_model_lits(m), 0, "and a refused row clears the list");
        ft_model_lit_weighted(m, a, 1, 2.5);
        assert_eq!(ft_model_close_linear(m, 0, 4.0), 1);
        assert_eq!(ft_model_compile(m), 0, "a fractional coefficient on an inequality is refused");
        assert!(model_error(m).contains("common denominator"), "{}", model_error(m));
        ft_model_free(m);

        // And a SOFT row is priced rather than enforced.
        let m = ft_model_new();
        let a = ft_model_binary(m);
        let b = ft_model_binary(m);
        ft_model_objective_term(m, 1, 10.0, a, 1);
        ft_model_objective_term(m, 1, 10.0, b, 1);
        ft_model_lit_weighted(m, a, 1, 3.0);
        ft_model_lit_weighted(m, b, 1, 4.0);
        assert_eq!(ft_model_close_linear_soft(m, 0, 3.0, 0.5), 1);
        assert!(ft_model_compile(m) > 0);
        ft_model_solve(m, 32);
        assert_eq!((ft_model_value(m, a), ft_model_value(m, b)), (1, 1), "the price is worth paying");
        assert_eq!(ft_model_feasible(m), 1, "a soft row leaves the answer an answer");
        // 3 + 4 = 7 against a bound of 3 is 4 over, priced at 0.5 × 4² = 8 -- less than the 10
        // that taking the second one is worth, which is why the trade happens at all.
        assert_eq!(ft_model_soft_cost(m), 8.0);
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
    let set = smp.collect(&crate::samples::Plan::new(200, draws as usize, thin.max(1) as usize), None);
    h.cert = Some(set.certificate(g).expect("collect returns a chain"));
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
        // Start from THIS SIMULATION'S state, so tabu composes the way every other solver here
        // does. It used to discard it and start from noise, which meant anneal-then-tabu threw the
        // anneal away without saying so.
        start: Some(s.sampler_state.clone()),
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

// ---- free energy ----------------------------------------------------------------------------------
//
// What a sampler owes: ln Z, with the guarantee each route actually carries. Reverse AIS is
// Rust-only -- it needs caller-supplied draws from the target and a statement of how they were
// made, which is a contract the flat ABI cannot express honestly. The three below are
// self-contained: exact, an UNCONDITIONAL lower bound, and a monotonicity bracket.

/// Exact `ln Z(beta)` by variable elimination, or NaN if the graph is too wide (induced width
/// above 24) or the handle is null. Bounded by treewidth, not by spin count.
#[no_mangle]
pub extern "C" fn ft_ln_z_exact(sim: *const Sim, beta: f64) -> f64 {
    let Some(s) = (unsafe { sim.as_ref() }) else { return f64::NAN };
    if !beta.is_finite() || beta < 0.0 {
        return f64::NAN;
    }
    match crate::exact::Elimination::default().log_partition(&s.graph, beta) {
        Ok(e) => e.log_z.unwrap_or(f64::NAN),
        Err(_) => f64::NAN,
    }
}

/// `ln Z(beta)` by annealed importance sampling up a linear ladder of `rungs` rungs, `sweeps`
/// palindromic sweeps per rung, `runs` independent walks. Returns the point estimate and keeps the
/// run for [`ft_ln_z_ais_lower`] and [`ft_ln_z_ais_ess`]. NaN on a null handle or `beta <= 0`.
/// Zero arguments take the defaults 64 / 2 / 128.
#[no_mangle]
pub extern "C" fn ft_ln_z_ais(sim: *mut Sim, beta: f64, rungs: u32, sweeps: u32, runs: u32) -> f64 {
    let Some(s) = (unsafe { sim.as_mut() }) else { return f64::NAN };
    if !beta.is_finite() || beta <= 0.0 {
        return f64::NAN;
    }
    let rungs = if rungs < 2 { 64 } else { rungs as usize };
    let sweeps = if sweeps == 0 { 2 } else { sweeps as usize };
    let runs = if runs == 0 { 128 } else { runs as usize };
    let ladder = crate::free_energy::linear_ladder(beta, rungs);
    let a = crate::free_energy::ais(&s.graph, &ladder, sweeps, runs, s.seed);
    let v = a.log_z;
    s.fe = Some(a);
    v
}

/// `ln Z >= ft_ln_z_ais_lower(delta)` with probability at least `1 - delta`, unconditionally --
/// Markov's inequality on the unbiased estimator of the last [`ft_ln_z_ais`]. NaN if there was
/// no run or `delta` is outside `(0, 1)`.
#[no_mangle]
pub extern "C" fn ft_ln_z_ais_lower(sim: *const Sim, delta: f64) -> f64 {
    if !(delta > 0.0 && delta < 1.0) {
        return f64::NAN;
    }
    match unsafe { sim.as_ref() }.and_then(|s| s.fe.as_ref()) {
        Some(a) => a.lower_bound(delta),
        None => f64::NAN,
    }
}

/// Effective sample size of the last [`ft_ln_z_ais`]'s weights; near 1 means one walk dominated
/// and the bound, still valid, is loose. NaN if there was no run.
#[no_mangle]
pub extern "C" fn ft_ln_z_ais_ess(sim: *const Sim) -> f64 {
    match unsafe { sim.as_ref() }.and_then(|s| s.fe.as_ref()) {
        Some(a) => a.ess,
        None => f64::NAN,
    }
}

/// `ln Z(beta)` by thermodynamic integration: `rungs` rungs, each measured by a chain of
/// `burn_in + draws` palindromic sweeps. Returns the bracket midpoint and writes the bracket --
/// each mean widened by `z` standard errors -- to `lower_out` / `upper_out` when non-null. The
/// bracket rests on `d<E>/dbeta <= 0`, a theorem, and on each rung being at equilibrium, which is
/// not. NaN on a null handle or `beta <= 0`; zero arguments take the defaults 32 / 200 / 2000 / 3.
#[no_mangle]
pub extern "C" fn ft_ln_z_ti(
    sim: *const Sim,
    beta: f64,
    rungs: u32,
    burn_in: u32,
    draws: u32,
    z: f64,
    lower_out: *mut f64,
    upper_out: *mut f64,
) -> f64 {
    let Some(s) = (unsafe { sim.as_ref() }) else { return f64::NAN };
    if !beta.is_finite() || beta <= 0.0 {
        return f64::NAN;
    }
    let rungs = if rungs < 2 { 32 } else { rungs as usize };
    let burn_in = if burn_in == 0 { 200 } else { burn_in as usize };
    let draws = if draws < 4 { 2000 } else { draws as usize };
    let z = if z.is_finite() && z > 0.0 { z } else { 3.0 };
    let ladder = crate::free_energy::linear_ladder(beta, rungs);
    let t = crate::free_energy::thermodynamic_integration(&s.graph, &ladder, burn_in, draws, z, s.seed);
    if !lower_out.is_null() {
        unsafe { *lower_out = t.lower_widened };
    }
    if !upper_out.is_null() {
        unsafe { *upper_out = t.upper_widened };
    }
    t.midpoint()
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
        // Same as `ft_tabu`: start from this simulation's state rather than discarding it.
        start: Some(s.sampler_state.clone()),
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

/// Hamze-de Freitas-Selby: solve a low-treewidth BLOCK exactly, repeatedly.
///
/// Every other local search here flips one spin and asks whether that helped. This takes the exact
/// best assignment of a whole subgraph given everything outside it held fixed, so it steps over any
/// barrier living entirely inside the block rather than paying to climb it. It is the algorithm
/// that turned the first generation of quantum-annealer speedup claims, and a stack that means to
/// make honest comparisons has to be able to run it.
///
/// Starts from THIS SIMULATION'S CURRENT STATE, so it composes: anneal, then tabu, then this. It is
/// a descent -- the energy never rises -- so it cannot undo whatever found the state it starts from.
///
/// `block` of 0 takes the default. Blocks are grown as induced TREES, whose width is 1 by
/// construction, so nothing here can be refused for width. Returns the best energy found, or NaN on
/// a null handle.
#[no_mangle]
pub extern "C" fn ft_hfs(sim: *mut Sim, steps: u32, block: u32) -> f64 {
    let Some(s) = (unsafe { sim.as_mut() }) else { return f64::NAN };
    let p = crate::hfs::Params {
        steps: steps.max(1) as usize,
        block: if block == 0 { crate::hfs::Params::default().block } else { block as usize },
        ..crate::hfs::Params::default()
    };
    let out = crate::hfs::run_from(&s.graph, s.sampler_state.clone(), &p, s.seed);
    if out.state.len() == s.sampler_state.len() {
        s.sampler_state.copy_from_slice(&out.state);
    }
    let e = out.energy;
    s.hf = Some(out);
    e
}

/// Block moves the last [`ft_hfs`] actually ran. 0 if there was none.
#[no_mangle]
pub extern "C" fn ft_hfs_moves(sim: *const Sim) -> u64 {
    unsafe { sim.as_ref() }.and_then(|s| s.hf.as_ref()).map_or(0, |o| o.moves as u64)
}

/// Block moves that strictly LOWERED the energy. 0 if there was none.
///
/// The number that says whether the descent is still going: a run whose blocks all land on a
/// minimum they already sit in has stopped, and no energy figure shows that.
#[no_mangle]
pub extern "C" fn ft_hfs_improving(sim: *const Sim) -> u64 {
    unsafe { sim.as_ref() }.and_then(|s| s.hf.as_ref()).map_or(0, |o| o.improving as u64)
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

// ---- higher-order models -------------------------------------------------------------------
//
// Everything above this line is pairwise, or becomes pairwise. `crate::hubo` is the one module that
// is neither: a term of any width contributes `-w * prod(s_i)` and the change from one flip is a
// sum over the terms containing that spin, so nothing about it needs an ancilla. Until now it had
// reached exactly one surface -- Rust -- and every other caller wanting a k-body model went through
// `Model` and `reduce`, which is a different computation with a measurable cost.
//
// How much cost was an open question and is not one any more. `examples/hubo_vs_reduction` runs the
// two paths on the same terms and gives the reduced arm its best ladder and up to 1024x the budget:
// on 60 three-body terms over 40 spins the native path reaches -48.12 and the reduced path reaches
// -34.00 at a thousand times the work. The mechanism is `Reduction::penalty`, chosen as the sum of
// every coefficient's magnitude and therefore ~1300 against term weights of 1, which makes the
// landscape rigid rather than merely larger. That is why this section exists: not so a k-body model
// can be EXPRESSED from C -- `ft_model_objective_product` already allows that -- but so it can be
// SOLVED the way `hubo` solves it.
//
// `ft_hubo_ancillas_avoided` is a CEILING and its doc says so. `reduce` shares one ancilla across
// every term containing the same pair, so it usually spends fewer.

use crate::hubo::{Hubo, Outcome as HuboOutcome, Params as HuboParams};

/// A higher-order model under construction, plus whatever it last annealed.
pub struct HuboHandle {
    hubo: Hubo,
    /// The last run, or `None` before one. Read by the counters and by `ft_hubo_energy`.
    out: Option<HuboOutcome>,
    /// The current state, which a caller may also write with `ft_hubo_set_spins`.
    state: Vec<i8>,
    ledger: Ledger,
    last_error: String,
    /// Variables accumulating for the next term. Closed by `ft_hubo_add`.
    vars: Vec<u32>,
}

/// A model over `n` spins. NULL if `n` is zero, since a model with no variables can hold no term.
#[no_mangle]
pub extern "C" fn ft_hubo_new(n: u32) -> *mut HuboHandle {
    if n == 0 {
        return core::ptr::null_mut();
    }
    Box::into_raw(Box::new(HuboHandle {
        hubo: Hubo::new(n as usize),
        out: None,
        state: vec![1; n as usize],
        ledger: Ledger::default(),
        last_error: String::new(),
        vars: Vec::new(),
    }))
}

#[no_mangle]
pub extern "C" fn ft_hubo_free(h: *mut HuboHandle) {
    if !h.is_null() {
        drop(unsafe { Box::from_raw(h) });
    }
}

/// Lift a pairwise simulation into a higher-order model, unchanged.
///
/// The only way the native-versus-reduced comparison this module exists to settle can be set up
/// from outside Rust: build a graph, lift it, and check that both paths score it identically.
#[no_mangle]
pub extern "C" fn ft_hubo_from_sim(sim: *const Sim) -> *mut HuboHandle {
    let Some(s) = (unsafe { sim.as_ref() }) else { return core::ptr::null_mut() };
    let hubo = Hubo::from_graph(&s.graph);
    Box::into_raw(Box::new(HuboHandle {
        hubo,
        out: None,
        state: s.sampler_state.clone(),
        ledger: Ledger::default(),
        last_error: String::new(),
        vars: Vec::new(),
    }))
}

/// Start a fresh variable list for the next term.
#[no_mangle]
pub extern "C" fn ft_hubo_vars_clear(h: *mut HuboHandle) -> u32 {
    let Some(hh) = (unsafe { h.as_mut() }) else { return 0 };
    hh.vars.clear();
    1
}

/// Append a variable to the pending term. Refuses one out of range, or one already pending.
///
/// The repeat is caught HERE rather than at `ft_hubo_add`, because `s * s = 1` silently changes a
/// term's order and a caller that learns about it several calls later has to work out which call
/// was wrong. `Hubo::add` refuses it too; this is the earlier of the two.
#[no_mangle]
pub extern "C" fn ft_hubo_var(h: *mut HuboHandle, var: u32) -> u32 {
    let Some(hh) = (unsafe { h.as_mut() }) else { return 0 };
    if var as usize >= hh.hubo.len() {
        hh.last_error = format!("no variable {var}; {} declared", hh.hubo.len());
        return 0;
    }
    if hh.vars.contains(&var) {
        hh.last_error =
            format!("variable {var} is already in this term; s*s = 1, so a repeat would change its order");
        return 0;
    }
    hh.vars.push(var);
    hh.last_error.clear();
    1
}

/// How many variables are pending, so a caller can check its own bookkeeping.
#[no_mangle]
pub extern "C" fn ft_hubo_vars(h: *const HuboHandle) -> u32 {
    match unsafe { h.as_ref() } {
        Some(hh) => hh.vars.len() as u32,
        None => 0,
    }
}

/// Close the pending variables as one term of the given weight.
///
/// Clears the list whether it succeeds or not, and clears it FIRST -- a refused term that left its
/// variables pending would be silently absorbed by the next one.
#[no_mangle]
pub extern "C" fn ft_hubo_add(h: *mut HuboHandle, weight: f64) -> u32 {
    let Some(hh) = (unsafe { h.as_mut() }) else { return 0 };
    let vars: Vec<usize> = core::mem::take(&mut hh.vars).iter().map(|&v| v as usize).collect();
    match hh.hubo.add(&vars, weight) {
        Ok(()) => {
            hh.last_error.clear();
            1
        }
        Err(e) => {
            hh.last_error = e.to_string();
            0
        }
    }
}

/// A term of up to four variables, positionally, for a node graph with a fixed number of ports.
///
/// `u32::MAX` in a slot means "no variable there". `count` says how many of `a b c d` to read, so a
/// caller cannot accidentally add a term of the wrong order by leaving a stale argument in place.
/// Everything past four goes through `ft_hubo_var` + `ft_hubo_add`, which has no arity ceiling.
#[no_mangle]
pub extern "C" fn ft_hubo_term(
    h: *mut HuboHandle,
    count: u32,
    weight: f64,
    a: u32,
    b: u32,
    c: u32,
    d: u32,
) -> u32 {
    let Some(hh) = (unsafe { h.as_mut() }) else { return 0 };
    if count == 0 || count > 4 {
        hh.last_error = format!("this form takes one to four variables, not {count}");
        return 0;
    }
    hh.vars.clear();
    for &v in [a, b, c, d].iter().take(count as usize) {
        if ft_hubo_var(h, v) == 0 {
            // ft_hubo_var left the reason behind; clear the partial term so it cannot bleed.
            let Some(hh) = (unsafe { h.as_mut() }) else { return 0 };
            hh.vars.clear();
            return 0;
        }
    }
    ft_hubo_add(h, weight)
}

/// Spins in the model.
#[no_mangle]
pub extern "C" fn ft_hubo_len(h: *const HuboHandle) -> u32 {
    match unsafe { h.as_ref() } {
        Some(hh) => hh.hubo.len() as u32,
        None => 0,
    }
}

/// Terms in the model.
#[no_mangle]
pub extern "C" fn ft_hubo_terms(h: *const HuboHandle) -> u32 {
    match unsafe { h.as_ref() } {
        Some(hh) => hh.hubo.terms() as u32,
        None => 0,
    }
}

/// The widest term, or 0 for a model with none.
#[no_mangle]
pub extern "C" fn ft_hubo_max_arity(h: *const HuboHandle) -> u32 {
    match unsafe { h.as_ref() } {
        Some(hh) => hh.hubo.max_arity() as u32,
        None => 0,
    }
}

/// An UPPER BOUND on the ancillas a pairwise reduction would have spent, and this path did not.
///
/// A ceiling rather than a cost: `reduce::to_pairwise` substitutes the commonest pair first, so one
/// ancilla serves every term containing that pair, and on three terms sharing one it spends one
/// where this returns three. See [`crate::hubo::Hubo::ancillas_avoided`].
#[no_mangle]
pub extern "C" fn ft_hubo_ancillas_avoided(h: *const HuboHandle) -> u32 {
    match unsafe { h.as_ref() } {
        Some(hh) => hh.hubo.ancillas_avoided() as u32,
        None => 0,
    }
}

/// Anneal, returning the best energy found, or NaN on a refusal.
///
/// Zero for any ladder parameter means "use the default for that one". NaN is refused explicitly
/// BEFORE that test, because `NaN > 0.0` is false and would otherwise be read as a zero and
/// silently answered on a ladder the caller never asked for.
#[no_mangle]
pub extern "C" fn ft_hubo_anneal(
    h: *mut HuboHandle,
    beta_min: f64,
    beta_max: f64,
    stages: u32,
    sweeps_per_stage: u32,
    seed: u64,
) -> f64 {
    let Some(hh) = (unsafe { h.as_mut() }) else { return f64::NAN };
    if !beta_min.is_finite() || !beta_max.is_finite() || beta_min < 0.0 || beta_max < 0.0 {
        hh.last_error =
            format!("a beta ladder needs two finite non-negative numbers, not {beta_min} and {beta_max}");
        return f64::NAN;
    }
    let d = HuboParams::default();
    let p = HuboParams {
        beta_min: if beta_min > 0.0 { beta_min } else { d.beta_min },
        beta_max: if beta_max > 0.0 { beta_max } else { d.beta_max },
        stages: if stages > 0 { stages as usize } else { d.stages },
        sweeps_per_stage: if sweeps_per_stage > 0 { sweeps_per_stage as usize } else { d.sweeps_per_stage },
    };
    if p.beta_max <= p.beta_min {
        hh.last_error =
            format!("beta_max must exceed beta_min; got {} and {}", p.beta_max, p.beta_min);
        return f64::NAN;
    }
    let out = crate::hubo::anneal_metered(&hh.hubo, &p, seed, Some(&mut hh.ledger));
    hh.state.clear();
    hh.state.extend_from_slice(&out.state);
    let e = out.energy;
    hh.out = Some(out);
    hh.last_error.clear();
    e
}

/// The current state, or NULL. Valid until the next `ft_hubo_*` call on this handle.
#[no_mangle]
pub extern "C" fn ft_hubo_spins(h: *const HuboHandle) -> *const i8 {
    match unsafe { h.as_ref() } {
        Some(hh) if !hh.state.is_empty() => hh.state.as_ptr(),
        _ => core::ptr::null(),
    }
}

/// Copy the state out. Refuses a length that is not exactly the model's, never writing partially.
#[no_mangle]
pub extern "C" fn ft_hubo_read(h: *const HuboHandle, out: *mut i8, len: u32) -> u32 {
    let Some(hh) = (unsafe { h.as_ref() }) else { return 0 };
    if out.is_null() || len as usize != hh.state.len() {
        return 0;
    }
    unsafe { core::ptr::copy_nonoverlapping(hh.state.as_ptr(), out, hh.state.len()) };
    1
}

/// Put a state IN, so something computed elsewhere can be scored by this library.
///
/// Refuses any element that is not -1 or +1, and refuses the whole write rather than part of it: a
/// model half-set from a bad buffer would score a state that never existed anywhere.
#[no_mangle]
pub extern "C" fn ft_hubo_set_spins(h: *mut HuboHandle, ptr: *const i8, len: u32) -> u32 {
    let Some(hh) = (unsafe { h.as_mut() }) else { return 0 };
    if ptr.is_null() || len as usize != hh.hubo.len() {
        hh.last_error =
            format!("this model has {} spins; {len} were offered", hh.hubo.len());
        return 0;
    }
    let src = unsafe { core::slice::from_raw_parts(ptr, len as usize) };
    if let Some(bad) = src.iter().position(|&v| v != -1 && v != 1) {
        hh.last_error = format!("spin {bad} is {}, and a spin is -1 or +1", src[bad]);
        return 0;
    }
    hh.state.clear();
    hh.state.extend_from_slice(src);
    hh.last_error.clear();
    1
}

/// Energy of the current state, or NaN if the handle is null.
#[no_mangle]
pub extern "C" fn ft_hubo_energy(h: *const HuboHandle) -> f64 {
    match unsafe { h.as_ref() } {
        Some(hh) if hh.state.len() == hh.hubo.len() => hh.hubo.energy(&hh.state),
        _ => f64::NAN,
    }
}

/// The energy change from flipping spin `i`, or NaN if the handle is null or `i` is out of range.
///
/// The higher-order twin of [`ft_field`]: what lets another language, or a GPU, check this
/// library's arithmetic term by term rather than only comparing a final number.
#[no_mangle]
pub extern "C" fn ft_hubo_delta(h: *const HuboHandle, i: u32) -> f64 {
    match unsafe { h.as_ref() } {
        Some(hh) if (i as usize) < hh.hubo.len() && hh.state.len() == hh.hubo.len() => {
            hh.hubo.delta(&hh.state, i as usize)
        }
        _ => f64::NAN,
    }
}

/// Flips proposed by the last run. Without it a run that moved nothing looks like a completed one.
#[no_mangle]
pub extern "C" fn ft_hubo_proposals(h: *const HuboHandle) -> u64 {
    unsafe { h.as_ref() }.and_then(|hh| hh.out.as_ref()).map_or(0, |o| o.proposals)
}

/// Flips accepted by the last run.
#[no_mangle]
pub extern "C" fn ft_hubo_accepted(h: *const HuboHandle) -> u64 {
    unsafe { h.as_ref() }.and_then(|hh| hh.out.as_ref()).map_or(0, |o| o.accepted)
}

/// Joules this model WOULD have cost on a Z1-class device (vendor SPICE prices, pre-silicon).
#[no_mangle]
pub extern "C" fn ft_hubo_joules_z1(h: *const HuboHandle) -> f64 {
    unsafe { h.as_ref() }.map_or(f64::NAN, |hh| hh.ledger.joules(&Z1_SPICE).unwrap_or(f64::NAN))
}

/// The last refusal, as UTF-8. Same two-call protocol as [`ft_model_error`].
#[no_mangle]
pub extern "C" fn ft_hubo_error(h: *const HuboHandle, buf: *mut u8, cap: u32) -> u32 {
    let Some(hh) = (unsafe { h.as_ref() }) else { return 0 };
    let b = hh.last_error.as_bytes();
    if buf.is_null() {
        return b.len() as u32;
    }
    let n = b.len().min(cap as usize);
    unsafe { core::ptr::copy_nonoverlapping(b.as_ptr(), buf, n) };
    n as u32
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

#[cfg(test)]
mod hubo_ffi_tests {
    use super::*;

    /// Build a model through the ABI exactly as a C caller would.
    fn model(n: u32, terms: &[(&[u32], f64)]) -> *mut HuboHandle {
        let h = ft_hubo_new(n);
        assert!(!h.is_null());
        for (vars, w) in terms {
            assert_eq!(ft_hubo_vars_clear(h), 1);
            for &v in *vars {
                assert_eq!(ft_hubo_var(h, v), 1, "variable {v}");
            }
            assert_eq!(ft_hubo_add(h, *w), 1, "term {vars:?}");
        }
        h
    }

    #[test]
    fn the_module_doc_example_solves_through_the_abi() {
        // src/hubo.rs's own doctest: a three-body parity term, minimised when the product is +1.
        let h = model(3, &[(&[0, 1, 2], 1.0)]);
        let e = ft_hubo_anneal(h, 0.0, 0.0, 0, 0, 7);
        assert_eq!(e, -1.0, "the three-body parity term");

        let mut out = [0i8; 3];
        assert_eq!(ft_hubo_read(h, out.as_mut_ptr(), 3), 1);
        assert_eq!(out[0] as i32 * out[1] as i32 * out[2] as i32, 1, "{out:?}");

        // The energy the run returned is a claim about the state it left behind, so read it back.
        assert!((ft_hubo_energy(h) - e).abs() < 1e-9, "{} against {e}", ft_hubo_energy(h));
        assert_eq!(ft_hubo_terms(h), 1);
        assert_eq!(ft_hubo_max_arity(h), 3);
        assert_eq!(ft_hubo_ancillas_avoided(h), 1, "one substitution for one 3-body term");
        assert!(ft_hubo_proposals(h) > 0, "a run that proposed nothing is not a run");
        ft_hubo_free(h);
    }

    #[test]
    fn a_refused_variable_does_not_bleed_into_the_next_term() {
        // The failure this prevents: a half-built term left pending, silently absorbed by the term
        // after it, producing a model nobody wrote and an answer to it.
        let h = ft_hubo_new(4);
        assert_eq!(ft_hubo_var(h, 0), 1);
        assert_eq!(ft_hubo_var(h, 9), 0, "out of range");
        assert_eq!(ft_hubo_vars(h), 1, "the good one is still pending; only the bad one was refused");

        assert_eq!(ft_hubo_var(h, 0), 0, "a repeat, because s*s = 1 changes the order silently");
        let need = ft_hubo_error(h, core::ptr::null_mut(), 0);
        let mut buf = vec![0u8; need as usize];
        ft_hubo_error(h, buf.as_mut_ptr(), need);
        let msg = String::from_utf8(buf).unwrap();
        assert!(msg.contains("already in this term"), "{msg}");

        // A refused ADD clears the list, so nothing survives into the next term.
        assert_eq!(ft_hubo_add(h, f64::NAN), 0, "a non-finite weight poisons every energy");
        assert_eq!(ft_hubo_vars(h), 0, "cleared even though the add failed");
        assert_eq!(ft_hubo_terms(h), 0, "nothing malformed was recorded");
        ft_hubo_free(h);
    }

    #[test]
    fn the_positional_form_matches_the_list_form() {
        let a = model(4, &[(&[0, 1, 2], 1.5), (&[1, 2, 3], -2.0)]);
        let b = ft_hubo_new(4);
        assert_eq!(ft_hubo_term(b, 3, 1.5, 0, 1, 2, u32::MAX), 1);
        assert_eq!(ft_hubo_term(b, 3, -2.0, 1, 2, 3, u32::MAX), 1);

        let state: [i8; 4] = [1, -1, 1, -1];
        assert_eq!(ft_hubo_set_spins(a, state.as_ptr(), 4), 1);
        assert_eq!(ft_hubo_set_spins(b, state.as_ptr(), 4), 1);
        assert_eq!(ft_hubo_energy(a), ft_hubo_energy(b), "two ways to say one model");
        for i in 0..4 {
            assert_eq!(ft_hubo_delta(a, i), ft_hubo_delta(b, i), "flip {i}");
        }

        // A count that does not match the arguments is refused rather than read partially.
        assert_eq!(ft_hubo_term(b, 5, 1.0, 0, 1, 2, 3), 0);
        assert_eq!(ft_hubo_term(b, 0, 1.0, 0, 1, 2, 3), 0);
        assert_eq!(ft_hubo_vars(b), 0, "a refused positional term leaves nothing pending");
        ft_hubo_free(a);
        ft_hubo_free(b);
    }

    #[test]
    fn a_lifted_graph_scores_exactly_as_the_pairwise_path_does() {
        // The comparison this section exists to make possible from outside Rust: the same state,
        // scored by both paths, must give the same number. If it does not, one of them has the
        // sign convention wrong and every later comparison inherits it.
        let b = ft_builder_new(6);
        assert!(!b.is_null());
        for i in 0..5u32 {
            assert_eq!(ft_builder_couple(b, i, i + 1, if i % 2 == 0 { 1.0 } else { -1.0 }), 1);
        }
        assert_eq!(ft_builder_bias(b, 0, 0.5), 1);
        let sim = ft_builder_build(b, 0.9, 11);
        assert!(!sim.is_null());
        ft_sweep(sim, 20);

        let h = ft_hubo_from_sim(sim);
        assert!(!h.is_null());
        assert_eq!(ft_hubo_len(h), 6);
        assert_eq!(ft_hubo_max_arity(h), 2, "a lifted pairwise graph is still pairwise");
        assert_eq!(ft_hubo_ancillas_avoided(h), 0, "nothing wider than two needs a substitution");

        let pairwise = ft_energy(sim);
        let native = ft_hubo_energy(h);
        assert!((pairwise - native).abs() < 1e-9, "{pairwise} against {native}");

        // And the incremental update agrees with recomputing, node by node, through the ABI --
        // which is the check another language or a GPU would run against this library.
        let mut state = vec![0i8; 6];
        assert_eq!(ft_hubo_read(h, state.as_mut_ptr(), 6), 1);
        for i in 0..6usize {
            let before = ft_hubo_energy(h);
            let d = ft_hubo_delta(h, i as u32);
            state[i] = -state[i];
            assert_eq!(ft_hubo_set_spins(h, state.as_ptr(), 6), 1);
            let after = ft_hubo_energy(h);
            assert!((after - before - d).abs() < 1e-9, "flip {i}: {d} against {}", after - before);
            state[i] = -state[i];
            assert_eq!(ft_hubo_set_spins(h, state.as_ptr(), 6), 1);
        }
        ft_hubo_free(h);
        ft_free(sim);
    }

    #[test]
    fn a_bad_ladder_is_refused_by_name_and_a_nan_is_not_read_as_a_default() {
        let h = model(3, &[(&[0, 1, 2], 1.0)]);
        assert!(ft_hubo_anneal(h, 8.0, 0.05, 10, 10, 1).is_nan(), "backwards");
        assert!(ft_hubo_anneal(h, f64::NAN, 8.0, 10, 10, 1).is_nan(), "NaN is not a zero");
        assert!(ft_hubo_anneal(h, -1.0, 8.0, 10, 10, 1).is_nan(), "negative");
        // Zeros DO mean "use the default", which is the whole reason NaN has to be refused first.
        assert_eq!(ft_hubo_anneal(h, 0.0, 0.0, 0, 0, 1), -1.0);
        ft_hubo_free(h);
    }

    #[test]
    fn a_state_is_refused_whole_or_taken_whole() {
        let h = model(3, &[(&[0, 1, 2], 1.0)]);
        let good: [i8; 3] = [1, 1, 1];
        assert_eq!(ft_hubo_set_spins(h, good.as_ptr(), 3), 1);
        assert_eq!(ft_hubo_energy(h), -1.0);

        let bad: [i8; 3] = [1, 0, 1];
        assert_eq!(ft_hubo_set_spins(h, bad.as_ptr(), 3), 0, "0 is not a spin");
        assert_eq!(ft_hubo_energy(h), -1.0, "the refused write changed nothing");
        assert_eq!(ft_hubo_set_spins(h, good.as_ptr(), 2), 0, "wrong length");
        assert_eq!(ft_hubo_read(h, core::ptr::null_mut(), 3), 0);
        ft_hubo_free(h);
    }

    #[test]
    fn every_call_is_inert_on_a_null_handle() {
        let n: *mut HuboHandle = core::ptr::null_mut();
        ft_hubo_free(n);
        assert!(ft_hubo_from_sim(core::ptr::null()).is_null());
        assert_eq!(ft_hubo_vars_clear(n), 0);
        assert_eq!(ft_hubo_var(n, 0), 0);
        assert_eq!(ft_hubo_vars(n), 0);
        assert_eq!(ft_hubo_add(n, 1.0), 0);
        assert_eq!(ft_hubo_term(n, 2, 1.0, 0, 1, u32::MAX, u32::MAX), 0);
        assert_eq!(ft_hubo_len(n), 0);
        assert_eq!(ft_hubo_terms(n), 0);
        assert_eq!(ft_hubo_max_arity(n), 0);
        assert_eq!(ft_hubo_ancillas_avoided(n), 0);
        assert!(ft_hubo_anneal(n, 0.05, 8.0, 10, 10, 1).is_nan());
        assert!(ft_hubo_spins(n).is_null());
        assert_eq!(ft_hubo_read(n, core::ptr::null_mut(), 0), 0);
        assert_eq!(ft_hubo_set_spins(n, core::ptr::null(), 0), 0);
        assert!(ft_hubo_energy(n).is_nan());
        assert!(ft_hubo_delta(n, 0).is_nan());
        assert_eq!(ft_hubo_proposals(n), 0);
        assert_eq!(ft_hubo_accepted(n), 0);
        assert!(ft_hubo_joules_z1(n).is_nan());
        assert_eq!(ft_hubo_error(n, core::ptr::null_mut(), 0), 0);
        assert!(ft_hubo_new(0).is_null(), "a model with no variables can hold no term");
    }
}

#[cfg(test)]
mod parallel_sweep_tests {
    use super::*;

    fn lattice(l: u32, beta: f64, seed: u64) -> *mut Sim {
        let s = ft_ising2d_new(l, 1.0, beta, seed);
        assert!(!s.is_null());
        s
    }

    #[test]
    fn a_parallel_sweep_reproduces_bit_for_bit_at_a_fixed_thread_count() {
        // The promise is per (seed, threads), not per seed. Two runs at the same pair must agree;
        // asserting only the first would pass on a sampler that ignored `threads` entirely.
        let a = lattice(16, 0.5, 0xABC);
        let b = lattice(16, 0.5, 0xABC);
        assert_eq!(ft_sweep_par(a, 40, 4), 40);
        assert_eq!(ft_sweep_par(b, 40, 4), 40);
        let (na, nb) = (ft_len(a) as usize, ft_len(b) as usize);
        let sa = unsafe { core::slice::from_raw_parts(ft_spins(a), na) };
        let sb = unsafe { core::slice::from_raw_parts(ft_spins(b), nb) };
        assert_eq!(sa, sb, "same (seed, threads) must reproduce bit-identically");
        ft_free(a);
        ft_free(b);
    }

    #[test]
    fn the_thread_count_is_part_of_the_run_and_the_abi_says_which_ran() {
        // A different thread count is a different sample path -- WHEN THE THREADS ACTUALLY RUN. The
        // graph has to clear the MIN_CHUNK floor for that to be true, which a 16x16 lattice (128
        // per class) does not: below the floor `sweep_par` IS the serial path, and asking for four
        // threads there gives the same answer as asking for one because it is the same code. That
        // is the whole point of the floor and it is worth pinning both halves of.
        let small_a = lattice(16, 0.5, 0xABC);
        let small_b = lattice(16, 0.5, 0xABC);
        ft_sweep_par(small_a, 40, 1);
        ft_sweep_par(small_b, 40, 4);
        let n = ft_len(small_a) as usize;
        let sa = unsafe { core::slice::from_raw_parts(ft_spins(small_a), n) }.to_vec();
        let sb = unsafe { core::slice::from_raw_parts(ft_spins(small_b), n) }.to_vec();
        assert_eq!(sa, sb, "below the floor, four threads and one are the same serial path");
        assert_eq!(ft_threads_used(small_a), 1);
        assert_eq!(ft_threads_used(small_b), 1, "the ABI reports what RAN, not what was asked");
        ft_free(small_a);
        ft_free(small_b);

        // Above the floor they diverge, and the reproducibility note on ft_sweep_par depends on it.
        let a = lattice(96, 0.5, 0xABC);
        let b = lattice(96, 0.5, 0xABC);
        ft_sweep_par(a, 40, 1);
        ft_sweep_par(b, 40, 4);
        let n = ft_len(a) as usize;
        let sa = unsafe { core::slice::from_raw_parts(ft_spins(a), n) }.to_vec();
        let sb = unsafe { core::slice::from_raw_parts(ft_spins(b), n) }.to_vec();
        assert_ne!(sa, sb, "one thread and four are different paths once four actually run");
        assert_eq!(ft_threads_used(a), 1);
        assert_eq!(ft_threads_used(b), 4, "4608 per class clears the floor four times over");
        ft_free(a);
        ft_free(b);
    }

    #[test]
    fn threads_used_reports_the_chunks_that_ran_not_the_number_asked_for() {
        // A ring of 5 two-colours into classes of 3 and 2. Asked for 4 threads, the larger class
        // splits into chunks of ceil(3/4) = 1, so THREE run -- and the first version of this
        // accessor answered 4, because it computed min(threads, biggest class). An accessor whose
        // doc says "what actually ran" and returns what was asked for is worse than none.
        let b = ft_builder_new(5);
        for i in 0..5u32 {
            assert_eq!(ft_builder_couple(b, i, (i + 1) % 5, -1.0), 1);
        }
        let s = ft_builder_build(b, 0.5, 1);
        assert!(!s.is_null());
        ft_sweep_par(s, 5, 4);
        let used = ft_threads_used(s);
        assert!(used <= 4, "cannot use more threads than were asked for: {used}");
        assert!(used <= 3, "5 nodes over 2 colour classes cannot occupy 4 threads: {used}");
        ft_free(s);

        // A class big enough to fill them reports the full count. "Big enough" is now a floor of
        // MIN_CHUNK = 1024 nodes per thread, not one node per thread: a ring of 400 gives classes
        // of 200, which four threads would slice into fifties -- and a fifty-node slice finishes
        // faster than the barrier it then waits at, which is how asking for threads used to make a
        // caller up to thirty-three times SLOWER. Below the floor the answer is one, and one is the
        // truth: the serial path is what ran.
        let ring = |n: u32, threads: u32| {
            let b = ft_builder_new(n);
            for i in 0..n {
                assert_eq!(ft_builder_couple(b, i, (i + 1) % n, -1.0), 1);
            }
            let s = ft_builder_build(b, 0.5, 1);
            ft_sweep_par(s, 2, threads);
            let used = ft_threads_used(s);
            ft_free(s);
            used
        };
        assert_eq!(ring(400, 4), 1, "50 nodes a thread is below the floor, so it runs serially");
        assert_eq!(ring(8192, 4), 4, "1024 a thread is exactly the floor, so all four run");
        assert_eq!(ring(4096, 4), 2, "512 a thread is under it; two threads at 1024 each is not");
        assert_eq!(ring(65536, 4), 4, "and the floor never asks for MORE than was requested");
    }

    #[test]
    fn zero_threads_asks_the_machine_and_the_answer_is_at_least_one() {
        assert!(ft_hardware_threads() >= 1, "a machine has at least one thread");
        let s = lattice(12, 0.4, 5);
        assert_eq!(ft_threads_used(s), 0, "nothing parallel has run yet");
        ft_sweep_par(s, 10, 0);
        assert!(ft_threads_used(s) >= 1, "0 means ask the machine, not run on nothing");
        ft_free(s);
    }

    #[test]
    fn the_parallel_path_samples_the_same_physics_as_the_serial_one() {
        // Not bit-identical -- the RNG streams differ by construction -- but the same DISTRIBUTION.
        // Onsager is the referee, so a parallel sweep that raced would show up as a wrong
        // magnetisation rather than as a crash nobody sees.
        let beta = 0.6;
        let want = ft_onsager(beta);
        for threads in [1u32, 4] {
            let s = lattice(48, beta, 0x9A7);
            // Start ORDERED, as src/gibbs.rs's own version of this test does. Below the critical
            // point a random start on 48x48 coarsens into domains and stays there for far longer
            // than any test will wait: the first draft of this measured |M| = 0.12 against
            // Onsager's 0.97 and was measuring domain walls, not a broken sampler.
            let up = vec![1i8; ft_len(s) as usize];
            assert_eq!(ft_set_spins(s, up.as_ptr(), up.len() as u32), 1);
            ft_sweep_par(s, 2000, threads);
            let mut acc = 0.0;
            for _ in 0..400 {
                ft_sweep_par(s, 1, threads);
                acc += ft_magnetization(s).abs();
            }
            let m = acc / 400.0;
            assert!((m - want).abs() < 0.02, "threads={threads}: |M| {m:.4} vs Onsager {want:.4}");
            ft_free(s);
        }
    }

    #[test]
    fn every_parallel_call_is_inert_on_a_null_handle() {
        let n: *mut Sim = core::ptr::null_mut();
        assert_eq!(ft_sweep_par(n, 10, 4), 0);
        assert_eq!(ft_threads_used(core::ptr::null()), 0);
    }
}

#[cfg(test)]
mod hfs_ffi {
    use super::*;

    #[test]
    fn a_block_descent_composes_after_annealing_and_never_undoes_it() {
        // The composition claim on ft_hfs, tested rather than asserted: it starts from the
        // simulation's CURRENT state, and being a descent it cannot make that state worse.
        let sim = ft_planted_frustrated(6, 40, 3, 1.0);
        assert!(!sim.is_null());
        ft_anneal(sim, 0.05, 4.0, 60, 40);
        let after_anneal = ft_energy(sim);

        let e = ft_hfs(sim, 200, 32);
        assert!(e <= after_anneal + 1e-9, "a descent cannot rise: {after_anneal} -> {e}");
        // The returned energy is the energy of the state left behind, not a number carried along.
        assert!((ft_energy(sim) - e).abs() < 1e-9);
        assert!(ft_hfs_moves(sim) > 0, "a run that made no move is not a run");
        assert!(ft_hfs_improving(sim) <= ft_hfs_moves(sim));
        ft_free(sim);
    }

    #[test]
    fn block_moves_reach_lower_energy_than_the_same_budget_of_sweeps() {
        // Not a speed claim -- the machine is loaded and no rate here is quotable. A claim about
        // REACH: a block move sees barriers a single flip cannot, so from the same start it should
        // land lower on a frustrated instance more often than not.
        let (mut better, mut worse) = (0, 0);
        for seed in 0..8u64 {
            let a = ft_planted_frustrated(6, 40, 3 + seed, 1.0);
            let b = ft_planted_frustrated(6, 40, 3 + seed, 1.0);
            ft_anneal(a, 0.05, 4.0, 40, 20);
            ft_anneal(b, 0.05, 4.0, 40, 20);
            let hfs = ft_hfs(a, 150, 32);
            let swept = {
                ft_sweep(b, 150 * 32);
                ft_energy(b)
            };
            if hfs < swept - 1e-9 {
                better += 1;
            } else if hfs > swept + 1e-9 {
                worse += 1;
            }
            ft_free(a);
            ft_free(b);
        }
        assert!(better >= worse, "block moves reached lower {better} times, higher {worse}");
    }

    #[test]
    fn every_hfs_call_is_inert_on_a_null_handle() {
        let n: *mut Sim = core::ptr::null_mut();
        assert!(ft_hfs(n, 10, 8).is_nan());
        assert_eq!(ft_hfs_moves(core::ptr::null()), 0);
        assert_eq!(ft_hfs_improving(core::ptr::null()), 0);
    }
}

#[cfg(test)]
mod warm_start_ffi {
    use super::*;

    #[test]
    fn tabu_and_breakout_build_on_the_state_they_are_given() {
        // Both used to DISCARD the simulation's state and start from noise, so anneal-then-tabu
        // threw the anneal away without saying so. Every other solver here composes; these did not.
        //
        // The assertion is that the handed state is never lost, which is the property that makes
        // composition safe: both searches track the best state ever seen, and the handed one is the
        // first they see.
        for solver in 0..2 {
            let sim = ft_planted_frustrated(6, 40, 7, 1.0);
            assert!(!sim.is_null());
            ft_anneal(sim, 0.05, 4.0, 60, 40);
            let annealed = ft_energy(sim);

            let after = if solver == 0 {
                ft_tabu(sim, 5_000, 0, 0)
            } else {
                ft_bls(sim, 5_000)
            };
            assert!(
                after <= annealed + 1e-9,
                "solver {solver}: handed {annealed}, returned {after} -- the start was discarded"
            );
            ft_free(sim);
        }
    }
}

#[cfg(test)]
mod model_method_ffi {
    use super::*;

    /// The claim this crate leads with, reachable from the modelling layer for the first time.
    #[test]
    fn the_model_layer_can_prove_an_answer_through_the_abi() {
        let m = ft_model_new();
        let a = ft_model_categorical(m, 3);
        let b = ft_model_categorical(m, 3);
        assert_eq!(ft_model_not_equal(m, a, b), 1);
        assert_eq!(ft_model_objective_term(m, 1, 5.0, a, 1), 1);
        assert_eq!(ft_model_objective_term(m, 1, 4.0, b, 2), 1);
        assert!(ft_model_compile(m) > 0);

        // Annealing cannot prove, whatever it finds.
        assert_eq!(ft_model_solve_by(m, 0, 0), 1);
        assert_eq!(ft_model_proved(m), 0, "an anneal proves nothing");

        // Branch can.
        assert_eq!(ft_model_solve_by(m, 3, 5_000_000), 1);
        assert_eq!(ft_model_proved(m), 1, "the tree is tiny");
        assert_eq!(ft_model_feasible(m), 1);
        // a != b permits a = 1 and b = 2, so the optimum is 9 in the modeller's units.
        assert!((ft_model_objective(m) - 9.0).abs() < 1e-9, "{}", ft_model_objective(m));
        ft_model_free(m);
    }

    #[test]
    fn every_method_runs_and_an_unknown_one_is_refused_by_name() {
        let m = ft_model_new();
        let v = ft_model_categorical(m, 3);
        assert_eq!(ft_model_fix(m, v, 1), 1);
        assert!(ft_model_compile(m) > 0);
        for method in 0..4u32 {
            assert_eq!(ft_model_solve_by(m, method, 2_000), 1, "method {method}");
            assert_eq!(ft_model_feasible(m), 1, "method {method}");
        }
        assert_eq!(ft_model_solve_by(m, 9, 0), 0);
        let need = ft_model_error(m, core::ptr::null_mut(), 0);
        let mut buf = vec![0u8; need as usize];
        ft_model_error(m, buf.as_mut_ptr(), need);
        let msg = String::from_utf8(buf).unwrap();
        assert!(msg.contains("unknown method 9"), "{msg}");
        ft_model_free(m);
    }

    #[test]
    fn solving_before_compiling_is_refused_with_the_reason() {
        let m = ft_model_new();
        let _ = ft_model_categorical(m, 3);
        assert_eq!(ft_model_solve_by(m, 3, 0), 0, "nothing has been compiled");
        let need = ft_model_error(m, core::ptr::null_mut(), 0);
        let mut buf = vec![0u8; need as usize];
        ft_model_error(m, buf.as_mut_ptr(), need);
        assert!(String::from_utf8(buf).unwrap().contains("compile the model"));
        ft_model_free(m);

        let n: *mut ModelHandle = core::ptr::null_mut();
        assert_eq!(ft_model_solve_by(n, 0, 0), 0);
        assert_eq!(ft_model_proved(core::ptr::null()), 0);
    }
}

// ---------------------------------------------------------------------------------------------
// FITTING A MODEL TO DATA
//
// Every other family here takes a model as given: it samples one, optimises one, bounds one. This
// family PRODUCES one, and the reason it belongs on the ABI rather than in Rust alone is that a
// thermodynamic stack that can only consume models is half a paradigm. The argument for this class
// of hardware is that it samples Boltzmann distributions cheaply; the distributions anyone actually
// wants are FITTED, and a caller in C, Python, Zig or Julia who cannot fit one has to leave for
// PyTorch and come back, which is exactly the seam the hardware is supposed to remove.
//
// The composition is the point. `ft_ebm_train` REPLACES the simulation's graph with the fitted one,
// so every solver, sampler, certificate and bound already on this ABI immediately applies to a
// trained model. Fit an RBM, then anneal it, certify it, or hand it to branch and bound -- with no
// new API and no export step.

thread_local! {
    /// Per-thread for the same reason [`ft_ommx_error`]'s is: these are free-standing calls whose
    /// failures must not explain another thread's success.
    static EBM_ERROR: core::cell::RefCell<String> = const { core::cell::RefCell::new(String::new()) };
}

fn set_ebm_error(s: &str) {
    EBM_ERROR.with(|e| *e.borrow_mut() = s.to_string());
}

/// Why the last `ft_ebm_*` call failed, in the caller's own terms. Empty after a success.
///
/// Copies at most `cap` bytes into `buf` and returns how many were written; with a null `buf`,
/// returns the length needed and writes nothing. Not null-terminated. Same shape as
/// [`ft_ommx_error`], because a second convention for the same job is a second thing to get wrong.
#[no_mangle]
pub extern "C" fn ft_ebm_error(buf: *mut u8, cap: u32) -> u32 {
    EBM_ERROR.with(|e| {
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

/// A restricted Boltzmann machine's STRUCTURE: `visible` + `hidden` spins, complete bipartite,
/// every weight zero. Feed it to [`ft_ebm_train`] to give the weights meaning.
///
/// Visible units are spins `0..visible`, which is what [`ft_ebm_train`] and
/// [`ft_ebm_log_likelihood`] assume when they clamp a data row on.
#[no_mangle]
pub extern "C" fn ft_ebm_rbm(visible: u32, hidden: u32, beta: f64, seed: u64) -> *mut Sim {
    set_ebm_error("");
    if visible == 0 {
        set_ebm_error("an RBM needs at least one visible unit");
        return core::ptr::null_mut();
    }
    Sim::new(crate::ebm::rbm(visible as usize, hidden as usize), beta, seed)
}

/// A deep Boltzmann machine's structure: `visible` spins, then each layer in `layers`, chained.
///
/// One layer is exactly [`ft_ebm_rbm`]. More layers add latent units WITHOUT scaling any unit's
/// connectivity, which is the arrangement the mixing-expressivity tradeoff is a claim about --
/// `examples/trained_tradeoff` measures the two against each other and finds the claim's two halves
/// do not both survive.
#[no_mangle]
pub extern "C" fn ft_ebm_dbm(
    visible: u32,
    layers: *const u32,
    n_layers: u32,
    beta: f64,
    seed: u64,
) -> *mut Sim {
    set_ebm_error("");
    if visible == 0 {
        set_ebm_error("a Boltzmann machine needs at least one visible unit");
        return core::ptr::null_mut();
    }
    if layers.is_null() || n_layers == 0 {
        set_ebm_error("no hidden layers were given");
        return core::ptr::null_mut();
    }
    let widths: Vec<usize> = unsafe { core::slice::from_raw_parts(layers, n_layers as usize) }
        .iter()
        .map(|&w| w as usize)
        .collect();
    Sim::new(crate::ebm::dbm(visible as usize, &widths), beta, seed)
}

/// Fit the simulation's graph to `rows` by contrastive divergence. Returns 1, or 0 with the reason
/// in [`ft_ebm_error`].
///
/// `rows` is `n_rows * visible` entries of `-1` or `+1`, row-major. The graph's EDGE SET is kept and
/// its weights are overwritten, so the structure comes from [`ft_ebm_rbm`], [`ft_ebm_dbm`], or any
/// graph the caller built.
///
/// **This replaces the simulation's model, so every cached result about the old one is dropped** --
/// certificates, tabu and branch outcomes, the GPU model. A certificate proved against the weights
/// before training is a true statement about a model that no longer exists, and returning it after a
/// fit would be the most confident way this ABI could lie. The spin state survives: it is a state of
/// the same spins, and it is a perfectly good starting point for sampling the fitted model.
///
/// `epochs`, `k`, `positive_sweeps` and `batch` clamp up from 0 to the documented defaults of
/// [`crate::ebm::Params`]. The learning rate DECAYS to a tenth of `learning_rate` across training;
/// without that decay the fit has a noise floor and never reaches its own fixed point.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn ft_ebm_train(
    sim: *mut Sim,
    visible: u32,
    rows: *const i8,
    n_rows: u32,
    epochs: u32,
    k: u32,
    positive_sweeps: u32,
    learning_rate: f64,
    batch: u32,
    seed: u64,
) -> u32 {
    set_ebm_error("");
    let Some(s) = (unsafe { sim.as_mut() }) else {
        set_ebm_error("no simulation was given");
        return 0;
    };
    let Some(data) = read_dataset(visible, rows, n_rows) else { return 0 };
    let d = crate::ebm::Params::default();
    let p = crate::ebm::Params {
        epochs: if epochs == 0 { d.epochs } else { epochs as usize },
        k: if k == 0 { d.k } else { k as usize },
        positive_sweeps: if positive_sweeps == 0 {
            d.positive_sweeps
        } else {
            positive_sweeps as usize
        },
        learning_rate: if learning_rate == 0.0 { d.learning_rate } else { learning_rate },
        batch: if batch == 0 { d.batch } else { batch as usize },
    };
    match crate::ebm::train(&s.graph, &data, &p, seed) {
        Ok(t) => {
            *s.graph = t.graph;
            // Everything derived from the OLD weights is now false. Dropping it is not tidiness.
            s.gpu = None;
            s.cert = None;
            s.tb = None;
            s.bl = None;
            s.pc = None;
            s.tor = None;
            s.gw = None;
            s.ic = None;
            s.pa = None;
            s.bb = None;
            s.hf = None;
            s.ground = None;
            1
        }
        Err(e) => {
            set_ebm_error(&e.to_string());
            0
        }
    }
}

/// Mean log-likelihood per row under the simulation's current model, EXACT, by enumeration.
///
/// Returns NaN with the reason in [`ft_ebm_error`] -- including when the model has more than
/// [`crate::ebm::MAX_ENUMERATED`] spins, where it refuses rather than returning something cheaper.
/// That refusal is deliberate: an ELBO, a reconstruction error or a pseudo-likelihood is worst
/// exactly where sampling is worst, so a caller comparing models on one would be reading the
/// proxy's failure and calling it expressivity.
///
/// The scale has fixed ends and needs no calibration. A model that has learned nothing scores
/// `-visible * ln 2`; one that reproduces `n` equiprobable rows scores `-ln n`.
#[no_mangle]
pub extern "C" fn ft_ebm_log_likelihood(
    sim: *mut Sim,
    visible: u32,
    rows: *const i8,
    n_rows: u32,
) -> f64 {
    set_ebm_error("");
    let Some(s) = (unsafe { sim.as_ref() }) else {
        set_ebm_error("no simulation was given");
        return f64::NAN;
    };
    let Some(data) = read_dataset(visible, rows, n_rows) else { return f64::NAN };
    match crate::ebm::exact_log_likelihood(&s.graph, &data) {
        Ok(v) => v,
        Err(e) => {
            set_ebm_error(&e.to_string());
            f64::NAN
        }
    }
}

/// The `side` x `side` bars-and-stripes dataset, the standard tiny benchmark for fitting an EBM.
///
/// Writes `2^(side+1) - 2` rows of `side*side` entries each into `out`, row-major, and returns the
/// row count. Returns the row count WITHOUT writing when `out` is null, so a caller can size its
/// buffer first; returns 0 if `cap` is too small to hold every row.
#[no_mangle]
pub extern "C" fn ft_ebm_bars_and_stripes(side: u32, out: *mut i8, cap: u32) -> u32 {
    set_ebm_error("");
    if side == 0 || side > 8 {
        set_ebm_error("side must be between 1 and 8");
        return 0;
    }
    let d = crate::ebm::bars_and_stripes(side as usize);
    let need = d.rows.len() * d.visible;
    if out.is_null() {
        return d.rows.len() as u32;
    }
    if (cap as usize) < need {
        set_ebm_error("the buffer is too small for every row");
        return 0;
    }
    let dst = unsafe { core::slice::from_raw_parts_mut(out, need) };
    for (r, row) in d.rows.iter().enumerate() {
        dst[r * d.visible..(r + 1) * d.visible].copy_from_slice(row);
    }
    d.rows.len() as u32
}

/// Shared argument check for the two calls that take a dataset. Sets the error and returns `None`.
fn read_dataset(visible: u32, rows: *const i8, n_rows: u32) -> Option<crate::ebm::Dataset> {
    if rows.is_null() {
        set_ebm_error("no data rows were given");
        return None;
    }
    if visible == 0 || n_rows == 0 {
        set_ebm_error("a dataset needs at least one row and one visible unit");
        return None;
    }
    let flat =
        unsafe { core::slice::from_raw_parts(rows, n_rows as usize * visible as usize) };
    Some(crate::ebm::Dataset {
        visible: visible as usize,
        rows: flat.chunks(visible as usize).map(|c| c.to_vec()).collect(),
    })
}

#[cfg(test)]
mod ebm_ffi_tests {
    use super::*;

    /// The whole ABI family, end to end, and the invalidation that makes it safe to compose.
    #[test]
    fn fitting_through_the_abi_learns_and_drops_what_it_invalidates() {
        // Size the buffer first, the way a C caller must.
        let n_rows = ft_ebm_bars_and_stripes(2, core::ptr::null_mut(), 0);
        assert_eq!(n_rows, 6, "2x2 bars and stripes is 2*2^2 - 2 rows");
        let mut rows = vec![0i8; n_rows as usize * 4];
        assert_eq!(ft_ebm_bars_and_stripes(2, rows.as_mut_ptr(), rows.len() as u32), n_rows);
        assert!(rows.iter().all(|&v| v == 1 || v == -1));

        let sim = ft_ebm_rbm(4, 4, 1.0, 11);
        assert!(!sim.is_null());
        assert_eq!(ft_len(sim), 8);

        // An untrained model with every weight zero is uniform, so its likelihood is exactly
        // -visible * ln 2 and nothing else it could be.
        let before = ft_ebm_log_likelihood(sim, 4, rows.as_ptr(), n_rows);
        assert!((before - (-4.0 * 2f64.ln())).abs() < 1e-9, "{before}");

        // Prove something about the OLD weights, so there is a cached result to invalidate.
        ft_sweep(sim, 50);
        assert!(ft_tabu(sim, 2000, 0, 0).is_finite());
        assert!(unsafe { sim.as_ref() }.unwrap().tb.is_some());

        assert_eq!(ft_ebm_train(sim, 4, rows.as_ptr(), n_rows, 600, 10, 5, 0.05, 6, 3), 1);
        let after = ft_ebm_log_likelihood(sim, 4, rows.as_ptr(), n_rows);
        assert!(after > before + 0.05, "training must help: {before:.4} -> {after:.4}");
        assert!(after < 0.0, "a log-likelihood is negative: {after}");

        // A tabu outcome proved against weights that no longer exist is the most confident way
        // this ABI could lie, so the fit drops it.
        let s = unsafe { sim.as_ref() }.unwrap();
        assert!(s.tb.is_none(), "the fit must drop results about the old weights");
        assert!(s.cert.is_none() && s.gpu.is_none() && s.ground.is_none());
        // The spin state survives: same spins, and a fine start for sampling the fitted model.
        assert_eq!(s.sampler_state.len(), 8);

        // And the fitted model composes with everything already on this ABI.
        assert!(ft_tabu(sim, 2000, 0, 0).is_finite());
        ft_free(sim);
    }

    /// EVERY REFUSAL FROM `ft_model_objective_pair` MUST NAME ITSELF, and none of them did.
    ///
    /// Five causes shared one silent `return 0`, so Python raised the fallback "the library refused
    /// that objective" for all of them. And one of the five was not a refusal at all: the same
    /// variable twice is legal, the Rust path solves it correctly, and the C ABI rejected it -- a
    /// term expressible from Rust and from nowhere else.
    #[test]
    fn every_objective_pair_refusal_names_itself_and_the_legal_case_is_allowed() {
        let read = |m: *mut ModelHandle| {
            let n = ft_model_error(m, core::ptr::null_mut(), 0) as usize;
            let mut b = vec![0u8; n];
            let got = ft_model_error(m, b.as_mut_ptr(), n as u32) as usize;
            String::from_utf8_lossy(&b[..got]).to_string()
        };
        let m = ft_model_new();
        let a = ft_model_categorical(m, 3);
        let b = ft_model_categorical(m, 3);

        // A variable that does not exist, named on each side separately.
        assert_eq!(ft_model_objective_pair(m, 1, 1.0, 99, 0, b, 0), 0);
        assert!(read(m).contains("99"), "names the missing variable: {}", read(m));
        assert_eq!(ft_model_objective_pair(m, 1, 1.0, a, 0, 99, 0), 0);
        assert!(read(m).contains("99"), "on the second side too: {}", read(m));

        // A coefficient that is not a real number.
        assert_eq!(ft_model_objective_pair(m, 1, f64::NAN, a, 0, b, 0), 0);
        assert!(read(m).contains("real number"), "{}", read(m));

        // A value the variable cannot take, named on each side separately.
        assert_eq!(ft_model_objective_pair(m, 1, 1.0, a, 7, b, 0), 0);
        assert!(read(m).contains('7'), "names the bad value: {}", read(m));
        assert_eq!(ft_model_objective_pair(m, 1, 1.0, a, 0, b, 9), 0);
        assert!(read(m).contains('9'), "on the second side too: {}", read(m));

        // AND THE LEGAL CASE IS NOW ACCEPTED. x.is(1)*x.is(1) is the square of an indicator, which
        // is the indicator; maximising it at 5.0 must drive the variable to 1.
        assert_eq!(
            ft_model_objective_pair(m, 1, 5.0, a, 1, a, 1),
            1,
            "the same variable twice is a legal term and Model solves it correctly"
        );
        // compile returns the SPIN COUNT: two 3-value one-hots is six.
        assert_eq!(ft_model_compile(m), 6);
        assert_eq!(ft_model_solve(m, 0), 1);
        assert_eq!(ft_model_value(m, a), 1, "the square rewards a = 1");
        assert!((ft_model_objective(m) - 5.0).abs() < 1e-9, "and it is worth 5");
        ft_model_free(m);
    }

    #[test]
    fn a_refusal_says_why_in_the_callers_terms() {
        let read = || {
            let n = ft_ebm_error(core::ptr::null_mut(), 0) as usize;
            let mut b = vec![0u8; n];
            let got = ft_ebm_error(b.as_mut_ptr(), n as u32) as usize;
            String::from_utf8_lossy(&b[..got]).to_string()
        };
        assert!(ft_ebm_rbm(0, 4, 1.0, 1).is_null());
        assert!(read().contains("visible"));

        let sim = ft_ebm_rbm(4, 2, 1.0, 1);
        let rows = [1i8, -1, 1, -1];
        assert_eq!(ft_ebm_train(sim, 4, core::ptr::null(), 1, 10, 1, 1, 0.05, 1, 1), 0);
        assert!(read().contains("no data rows"));
        assert_eq!(ft_ebm_train(core::ptr::null_mut(), 4, rows.as_ptr(), 1, 10, 1, 1, 0.05, 1, 1), 0);
        assert!(read().contains("simulation"));
        // A successful call clears it.
        assert_eq!(ft_ebm_train(sim, 4, rows.as_ptr(), 1, 10, 1, 1, 0.05, 1, 1), 1);
        assert_eq!(read(), "");
        ft_free(sim);

        // Too large to enumerate is a refusal, not a cheaper answer.
        let big = ft_ebm_rbm(20, 8, 1.0, 1);
        let wide = [1i8; 20];
        assert!(ft_ebm_log_likelihood(big, 20, wide.as_ptr(), 1).is_nan());
        assert!(read().contains("28"), "the message names the size it refused: {}", read());
        ft_free(big);

        // A dbm with no layers is refused rather than silently becoming a bare visible layer.
        assert!(ft_ebm_dbm(4, core::ptr::null(), 0, 1.0, 1).is_null());
        assert!(read().contains("hidden layers"));
        let layers = [3u32, 3];
        let deep = ft_ebm_dbm(4, layers.as_ptr(), 2, 1.0, 1);
        assert_eq!(ft_len(deep), 10);
        ft_free(deep);
    }
}
