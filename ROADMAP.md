# ferrotherm: the implementation plan

Institute for Physical AI @ BMI. Written 2026-08-05, after a nine-angle global survey of how every
existing thermodynamic, Ising and probabilistic compute stack is programmed.

---

## 0. The position

The field has converged on what a program *is* on a sampling fabric, and we already have that
object. What nobody has is the rest of the sentence.

Every player returns "best found." Not one commercial machine — American, Japanese or Chinese —
exposes calibrated finite-temperature sampling with a stated distribution error. The standard
benchmark harness has four metrics and no energy axis. No library anywhere has a public path to
silicon. No node-graph editor exists for any Ising or factor-graph stack on Earth. The community
with the best hardware results ships MATLAB scripts and a `colorMap.csv`.

So SOTA here is not a faster sampler. **It is the only stack that can prove what it sampled, price
what it cost, and run the same program from a browser tab to a bitstream.** Speed is table stakes
and we will measure it honestly; the certificate and the ledger are the product.

Three properties are non-negotiable and constrain everything below.

| Property | Consequence |
|---|---|
| **Zero dependencies in the core** | No protobuf, no serde, no rayon in `ferrotherm`. Anything that needs a dependency is a separate crate the core never names. |
| **Runs everywhere, browser first** | Every core path must compile to `wasm32-unknown-unknown` and run without threads. Native and GPU are accelerations, never the only path. |
| **Nothing is quoted below its noise floor** | Every number ships with the floor it was measured against, or it does not ship. |

### What we are not doing

Routing, scheduling and portfolio optimisation. They are MILP in a QUBO costume and they lose to
Gurobi. Chasing them is how this field has burned its credibility for fifteen years. We go where
the substrate is native and the oracle is exact.

---

## 1. Architecture

One IR. Everything above is a lowering pass, everything below is a backend.

```
  Python │ Rust │ Node graph │ Zig │ MCP │ HTTP          surfaces
                        │
              lowering passes                     gates→factors, Potts→Ising (domain wall),
                        │                         dense→sparse (COPY gate), higher-order→pairwise
                        ▼
   Model × Blocks × Kernels × Schedule × Observers        THE IR  (.ftp)
                        │
                        ▼
        Sampler trait ── Certificate ── Ledger            proof + price, always
                        │
   CPU │ WebGPU │ FPGA (Pt V2 →) │ device                 backends
```

The IR is the converged abstraction, validated independently by THRML's decomposition and by the
p-bit community's `(J, h, colorMap.csv)` convention. `Ebm::gibbs_chromatic` is already it. Our work
is to name it properly, free it of the two mistakes we currently share with the incumbents, and put
a certificate on its output.

---

## 2. Phases

Each item lists its **acceptance criterion** — the thing that must be true, measured, before it is
called done. No item is complete because code exists.

### Phase 0 — Pay the debt before it compounds (v0.6.0, breaking)

We currently carry two of the exact defects the survey found in the incumbents. Both are cheap now
and expensive after four surfaces depend on them.

**0.1 One kernel.** ✅ **DONE** — `src/kernel.rs`.

*Diagnosis corrected after reading the code.* This plan first said `program::Gate` was "a second IR
beside the EBM graph," Extropic's mistake. That was wrong, and reading `program.rs` closely showed
something more specific and more fixable. `program.rs` is not a rival spelling of the sampler; it is
a differentiable stochastic program with continuous state and score-function gradients, a capability
the IR genuinely does not have. What it *was* doing wrong is that it carried its own copy of the
sweep, because it needs the score alongside the draw.

The real defect was that `P(s_i=+1) = sigma(2 beta f_i)` — the one equation this crate computes —
was written out **six times across five files in three spellings of beta**:

| Site | Spelling | Problem |
|---|---|---|
| `gibbs.rs` ×2 | `sigma(2.0 * self.beta * f)` | — |
| `program.rs` | `sigma(2.0 * beta * f)` | beta baked into the gate |
| `compile.rs` | `1/(1+exp(-2.0*self.beta*f))` | different spelling |
| `hdl.rs` | `1/(1+exp(-2.0*beta*arg))` | RTL emitter |
| `dtm.rs` ×2 | `1/(1+exp(-2.0*f))` | **no beta at all** — folded into the weights |

That last row is the same footgun this document criticises THRML for: with beta inside the weights,
annealing a DTM means rewriting every coupling.

**Done:** every Rust caller now routes through `kernel::{p_up, draw, score_dh, delta_e}`.
`program.rs` keeps its differentiable layer and loses its private sweep, because `score_dh` gives it
the score without the copy. `dtm` gained `gibbs_at`/`gibbs_chromatic_at` taking beta as a parameter,
defaulting to 1.0 so behaviour is unchanged. `hdl.rs` is the one legitimate duplicate — emitted
Verilog cannot call Rust — so it builds its threshold table *from* `kernel::p_up` and a test asserts
the 16-bit table agrees with the kernel to within one LSB across five betas.

**Accepted:** `grep` finds zero spin-kernel implementations outside `kernel.rs`; 52 tests pass
including a numerical-derivative check that `score_dh` matches `p_up`, and a detailed-balance check
tying `delta_e` to the acceptance ratio; wasm, Python and Zig all still green, with Python still
reproducing Onsager (|m| 0.919 vs 0.911 exact).

**0.2 Free the temperature.** ✅ **DONE** — `src/schedule.rs`.

`Schedule` is an ordered list of `Stage { beta, sweeps, penalties }`, where `Penalties` carries the
domain-wall and copy-agreement strengths that lowering passes introduce. Temperature and penalty
ramps are specified independently (`geometric(..).ramp_domain_wall(..).ramp_copy(..)`), because they
are separate physics: the ladder is geometric since the interesting behaviour is spread evenly in
log β, and a penalty ramps so the sampler can move freely early and the constraint binds late.
`total_sweeps` and `node_updates(n)` let a run be sized before it starts.

**Accepted, mechanically:** `graph::graph_builds()` counts programs built, and
`anneal_never_rebuilds_the_program` runs a **4,000-stage** ladder and asserts the counter does not
move. Three further contract tests: a reused graph agrees with a freshly built one, a cold ladder
cannot contaminate a later hot one, and the ledger matches what the schedule predicted.

*One finding from writing it.* The counter was first a process-global `AtomicU64` and the test
failed at 4 rebuilds — which was the test being right and the counter being wrong. `cargo test` runs
tests in parallel within one process, so a global counter answers "did anything in this process
build a graph", not "did this run rebuild". It is thread-local now, which is the scope the question
actually has.

**0.3 An encoding layer, not a type system.** The IR has exactly one variable: the spin, ±1. That
is what the fabric is, and nothing else may appear below the lowering passes.

Above the IR, the modelling layer accepts a discrete variable and the compiler **eliminates** it:

| Modelling variable | Encoding | Spins | Penalty couplings |
|---|---|---|---|
| `Categorical(k)` | one-hot | k | k(k−1)/2, **quadratic** |
| `Categorical(k)` | binary | ⌈log₂k⌉ | none, *only if k is a power of two* |
| `Categorical(k)` | **domain wall** | k−1 | k−2, **linear**, a chain |
| integer range | categorical over values, or binary expansion | same machinery | per encoding |

*Claim corrected while implementing.* This table first said domain-wall encoding needs **no**
penalty. That is the usual shorthand and it is wrong: it still needs terms to suppress states with
more than one wall. What it actually does is replace one-hot's all-to-all penalty with a *chain*,
taking the penalty cost from quadratic to linear in k, and save a spin. That is the honest claim and
the one worth having.

Binary is the trap: fewest spins, but it only excludes surplus codes for free when k is a power of
two. Otherwise the leftover codes are invalid states no pairwise penalty removes in general, so
`add_penalty` reports that it cannot be exact rather than pretending, and `decode` returns `None`
rather than rounding to a nearest valid guess. Silently rounding is how a constraint violation
becomes a wrong answer nobody notices.

*Why this framing:* an integer is not a thing the hardware has. Writing `Int` beside `Spin` implies
a register that does not exist, and the p-int literature proposing such units is three papers with
no code and no silicon. What is real is that problems have discrete variables and someone must
choose how to spell them in spins. That someone is the compiler.

**Accepted** — `src/encode.rs`. `Encoding::{OneHot, Binary, DomainWall}` with `Slot` placing a
variable's spins in a graph. Domain-wall uses strictly fewer spins than one-hot for every k in
2..=32, and its penalty is a chain of exactly k−2 couplings against one-hot's k(k−1)/2. The
construction is proved by **exhaustive enumeration**, not algebra: for each k the penalty's ground
states are enumerated over every spin configuration and must be exactly the k valid codewords, each
decoding to a distinct value. Below the lowering passes there is still no variable but the spin.

*A real bug fell out of that test.* `GraphBuilder::bias` **replaced** where `couple` **sums**.
Domain-wall at k = 2 puts both boundary terms on its single spin, where they must cancel — instead
the second erased the first, and the enumeration found one ground state where there should be two.
Any two passes touching one node hit this, and a user bias plus a penalty bias is the ordinary case.
`bias` now accumulates; `set_bias` replaces when that is actually wanted.

**0.4 Types that make the footguns unrepresentable.** ✅ **DONE** — `src/factor.rs`, `src/dense.rs`.

**The repeated variable** (`factor.rs`). THRML documents that a variable repeated in a factor
silently breaks Boltzmann correctness because "this condition has not been enforced in the code" —
`s_i * s_i = 1`, so an even multiplicity collapses the term to a constant and an odd one lowers its
order, and the model quietly stops being the one that was written. `Factor::new` returns
`Err(RepeatedVariable { var, times })`, naming *which* variable and how often, because "duplicate
found" is not actionable in a factor of eight arguments. A test demonstrates the harm rather than
asserting it: `[0,1,0]` is shown to be numerically identical to the order-1 factor `[1]`.

**The padding sentinel** (`dense.rs`). The `[n,k]` rectangle the CPU, GPU and RTL emitters share.
THRML pads with `INVALID_BIAS = -1e10`; a sentinel is fine until something does arithmetic on it,
whereupon it silently dominates every real term. Padding here is an explicit `active` mask *and*
inert twice over — padded slots carry weight `0.0` and index in range — so a kernel that forgot the
mask still computes the correct field. A test substitutes the sentinel into the padding and confirms
it destroys the field, so the design choice is demonstrated rather than argued.

**The sign convention** is pinned by test instead of by newtype: a positive weight prefers the
product to be +1, aligned is low energy, and arity-2 factors must agree exactly with the coupling
they replace or the IR has two energies.

*One planned item deliberately dropped.* Distinct `Weight` and `Energy` newtypes were in this plan
and are not built. They would not catch the error that actually happens — sign inversion — since
both sides of `E = -J s s` are `f64` either way, and the tests above pin the convention properly.
Ceremony that looks like safety is worse than no ceremony, because it is trusted.

**Accepted:** each documented incumbent footgun has a test proving it is a constructor `Err` or is
inert by construction. 79 lib tests pass.

**0.5 The `.ftp` program format.** Author the spec ourselves: spins, factors including
higher-order, the block coloring, the schedule with β and penalty ramps, the observer set, the
device topology target, the encoding provenance for each group of spins, and the price table used
for the ledger.
*Why:* the survey's sharpest finding is that `(J, h, coloring, schedule)` is the field's de facto
interchange and **nobody has specified it** — it lives as scattered `.mat` files. That is a
standard-shaped hole. Being the format is durable; conforming to someone else's two-year-old one is
a bet on their survival.
**Accept:** spec published; round-trips every model in the test suite byte-exactly; a `.ftp` written
by the browser runs unchanged on the CPU, and its hash matches.

### Phase 1 — The certificate and the oracles

This is the differentiator. It ships before any new surface, because everything we will later claim
depends on it.

**1.1 `Certificate`, returned from every `sample()`.** Never optional, never a separate call.

| Field | What it answers |
|---|---|
| `beta_eff` + CI | Did we sample at the temperature we asked for? |
| `tau_int`, `ess` | How many *independent* samples do we actually have? |
| `tv_exact` | On enumerable systems, distance from ground truth |
| `bias_floor` / `burn_in_bias` | Equilibrium bias and burn-in bias, separated |
| `copy_agreement` | Did sparsified copies agree? (mandatory) |
| `feasible` | Constraint satisfaction |
| `kl_bound` | Accumulated compile-pass KL, per Thermalizers Eq. 17 |
| `noise_floor` | Below this the API refuses to quote a number |

**Accept:** `verify` on a known-good sampler reports TV under the floor; a *deliberately broken*
sampler (wrong β, too-short burn-in, correlated draws) is caught by the certificate in each case.
A certificate that cannot fail is not a certificate.

**1.2 The oracle set**, all behind the same `Sampler` trait: exact enumeration, tree-decomposition
exact, planar exact, steepest descent, and a literal random-noise floor. Plus planted-instance
generators (Wishart, 3R3X XORSAT, frustrated loops) and physics oracles: Onsager magnetisation, the
SK transition, and 3D Edwards–Anderson against OPUSLab's published result files.
*Why:* build the oracle before the thing it judges. This is the artifact nobody in the world
publishes and it is the spine of every claim we make.
**Accept:** each oracle agrees with its published reference within a stated tolerance, and the
random-noise oracle *fails* every quality test — proving the tests discriminate.

### Phase 2 — Backends and the honest benchmark

**2.1 Chromatic block-Gibbs on WebGPU.** The padded-interaction + active-mask WGSL kernel behind
`Backend::Wgpu`.
*Why:* open GPU Ising sampling is an empty lane — OpenJij formally dropped GPGPU in 2023 and is
CPU-only, while every commercial engine is GPU, FPGA or ASIC. There is no open, permissive,
browser-capable sampler. That is our lane and it is unoccupied.
**Accept:** bit-reproducible per seed and workgroup count; matches the CPU backend exactly on every
oracle.

**2.2 The benchmark, reported the way nobody reports it.** Port THRML notebook 02's Pegasus
flips/ns benchmark exactly, and publish **flips/ns, joules/flip, and joules per *independent*
sample** (ESS-corrected — nobody reports this) against the three honest comparators the field hands
us: Extropic's own 8×B200 JAX figure, the ~60 flips/ns baked-in FPGA prior art (arXiv:2303.10728),
and the 185 flips/ns at 9.168 W Alveo result.
**Accept:** every number carries its noise floor and a named, *tuned* classical baseline. An
untuned baseline is a fabricated win.

**2.3 The path to silicon.** Sequential fabric on the Alchitry Pt V2 — flip-flops, clock, a running
sampler, state read back — then the same `.ftp` running on CPU, browser and FPGA with matching
results.
*Why:* **no library in this field has a public path to any silicon.** Extropic's packages contain
zero device code; every "backend" is a simulator. Closing library→bitstream→board→readback in the
open is a first, and we already own the whole toolchain.
**Accept:** one `.ftp`, three backends, same distribution within the certificate's floor. Published
with the bitstream.

### Phase 3 — The embedding layer

The admitted open wound of the field, and unowned.

**3.1 COPY-gate sparsification** ported to Rust (exact, ground-state preserving, degree budget
`ceil(max_deg/copies)+1`, bias split across copies) with **DSATUR** coloring as a first-class pass.

**3.2 2D adaptive parallel tempering over (β, W0)**, so nobody hand-tunes copy strength.
*Why:* every sparsification introduces a penalty that must be tuned by hand. OPUSLab states the
problem in exactly those terms; their answer is one MATLAB file from June 2025 with no adoption, and
their repo literally named `SparsifyDenseGraph` is empty.
**Accept:** across the planted-instance suite, adaptive tempering matches or beats the best
hand-tuned `W0`, with copy-agreement certified at readout.

**3.3 Publish the crossover.** At what `N` does dense-all-to-all beat sparsify-plus-embed?
*Why:* pc-COP runs 2048 fully-connected p-bits with zero embedding tax; the physics-ASIC
prescription is dense-within-tile, sparse-between-tile. Nobody has measured where the line is.
Publishing that crossover honestly is worth more than any speedup claim we could make.

### Phase 4 — Surfaces

Adapters, not a DSL. The field does not need a fifth modelling language.

| Surface | Design | Priority |
|---|---|---|
| **Rust** | Native. `Model` → lowering → `Program` → `Schedule`; traits for `Factor`/`Kernel`/`Sampler`. One variable below the lowering passes: the spin. Newtype indices, not THRML's metaclass ID hack. | now |
| **Python** | `ferrotherm.linalg` in thermox's signatures + `estimate/stderr/certificate/ledger` on every result. A PyMC `BlockedStep` with a `Competence` rule, so a discrete-latent model gets our block-Gibbs step in one line. | now |
| **Node graph** | The unoccupied lane. Nodes are IR objects, not widgets: variables, factors, blocks, schedule, observers, backend. Executes the same `.ftp`. Builds on the shipped workbench. | next |
| **Zig** | Already binds the C ABI. Extend as the IR lands. | next |
| **Adapters** (`dimod.Sampler`, `ommx`) | Separate, optional, deletable crates the core never names. Written when a user asks, not before. | later |

**Accept (node graph):** a graph built by dragging produces a `.ftp` byte-identical to the
hand-written one, and the round trip is a test.

### Phase 5 — Workloads

Ship where the substrate is native and the oracle is exact.

| Workload | Why it is ours | Oracle |
|---|---|---|
| **EBM / DTM training** | Already at published flagship scale on real Fashion-MNIST | Held-out likelihood; per-pixel stats vs data |
| **Spin-glass physics** | The substrate *is* the model | Onsager, SK, 3D EA reference files |
| **Thermodynamic linear algebra** | Solve/invert/sample by equilibration | Exact `A⁻¹b`, condition-number sweep |
| **Domain-wall categorical optimisation** | Where the type system pays | Planted instances with known optima |
| **Sampling-based control (MPPI)** | The Institute's own centre of gravity: on-device policy sampling for embodied AI | Known-optimal trajectories |

The last row is the one no thermodynamic vendor is pursuing and the one that connects this stack to
physical AI. It should become the flagship once Phase 2 lands.

---

## 3. The ingest list

What we port to Rust, and why each is worth owning.

| Source | State today | Our port |
|---|---|---|
| COPY-gate sparsification (OPUSLab) | ~102 lines MATLAB, no adoption | First-class pass |
| DSATUR coloring | Scattered | First-class pass |
| 2D adaptive PT over (β, W0) | One MATLAB file, June 2025 | Core capability |
| Lattice Random Walk integrator (arXiv:2508.20883) | **Paper, no code** | Binary/ternary SDE increments, no Gaussian RNG in the datapath, robust to quantisation |
| Thermalizers KL chain-rule bound (arXiv:2608.01615) | **Published without code** | The compile-pass error bound; hardware-independent mathematics |
| Domain-wall encoding (Chancellor 2019) | Published, scattered implementations | Compiler pass: k−1 spins, no penalty term |
| Chook planted-instance generators | Python | Test fixtures with known ground truth |
| Tree-decomposition & planar exact solvers | Scattered literature | Oracles |
| thermox signatures (Normal) | Python, no device hook | `ferrotherm.linalg`, plus certificate and ledger |
| THRML's decomposition | JAX, simulator-only | Inspiration, not a port — Rust traits do it better |

Three of these are papers with no code. Publishing working Rust for them is, on its own, a
contribution the field currently lacks.

---

## 4. Answering the critique aimed at us

Normal Computing argues that building 32-bit state variables from single-bit p-bits "requires 1024×
as many coupling terms," and that dense interaction matrices cause frustrated sampling and degrade
sample quality. That is aimed squarely at our design centre and we answer it with measurement, not
rhetoric: a domain-wall encoding pass is the structural reply to the first, and the
dense-vs-sparsify crossover of Phase 3.3 is the empirical reply to the second. If they are right at
some scale, we publish where.

---

## 5. Claims we will not make

The survey catalogued every unsupported number in this field. The short version of what we avoid:

- Any headline multiplier without a **named, tuned** classical baseline.
- Any energy figure that is a projection, quoted as a measurement. Extropic's "10,000×" was revised
  down roughly tenfold by their own later SPICE table; Normal's "1000×" appears in a chip paper
  containing zero watts.
- Any device specification quoted as measured before characterised silicon exists.
- Any benchmark reported without its noise floor.

We hold ourselves to the same standard we apply to them, in public, including when it costs us a
number we would rather quote.

---

## 6. Sequence

```
v0.6  Phase 0        one kernel ✅, β freed ✅, encoding passes, .ftp   breaking
v0.7  Phase 1        Certificate + oracle set + planted instances
v0.8  Phase 2.1/2.2  WebGPU backend + the honest benchmark
v0.9  Phase 3        sparsifier, DSATUR, 2D adaptive PT, crossover
v0.10 Phase 2.3      Pt V2 sequential fabric; one .ftp on three backends
v1.0  Phase 4/5      node graph, Python adapters, MPPI flagship
```

Phase 0 first, and alone, because every later phase inherits its mistakes. Phase 1 before any new
surface, because the certificate is what the surfaces are for.
