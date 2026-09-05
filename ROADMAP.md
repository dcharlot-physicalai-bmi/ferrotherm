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

**0.5 The `.ftp` program format.** ✅ **DONE** — `src/ftp.rs`, spec in the module docs.

Line-oriented UTF-8 text, one directive per line, `#` comments. Directives: `ftp`, `name`, `spins`,
`bias`, `factor` (any arity), `color`, `encode` (encoding provenance), `stage` (β, sweeps, and both
penalty strengths), `observe`, `target`, `price`. Text rather than binary because the field this is
meant to serve currently exchanges MATLAB files nobody outside the sending lab can read; something
you can `grep`, `diff` and read in a terminal is worth more here than saved bytes.

It describes a **program**, not a problem — which spins update together, at what temperatures, with
which penalties ramping, what to observe, what an operation costs. That difference is the whole
reason it exists rather than an adoption of somebody's instance format.

**Accepted:**
- Round-trips every model in the test suite **byte-exactly**, and floats survive **bit-for-bit**
  (verified on 0.1, 1/3, 1e-300, 1e300, π and a denormal-adjacent value via `to_bits()`).
- The worked example in the module documentation is parsed by a test *and annealed*, confirming it
  is the frustrated 5-ring it claims to be at energy −3. Documentation that does not parse is a bug
  report waiting to happen.
- Nine malformed inputs produce errors carrying a line number and a fix.
- **A `.ftp` written by the browser runs unchanged on the CPU.** `docs/ide.html` gained an
  independent JavaScript writer; `tests/fixtures/browser-lattice32.ftp` is its real output,
  captured from a browser run and committed verbatim. Four tests confirm the CPU parses it, agrees
  with `ising::lattice2d(32, 1.0)` on 200 random energies and on the −2048 ground state, round-trips
  it through the Rust writer with a stable digest, and anneals it. A format with one implementation
  is a data structure, not a format.

---

### Phase 0 is complete.

One kernel, β freed into a schedule, encodings as compiler passes, the footguns unrepresentable, and
a program format with two independent implementations. Three real bugs surfaced on the way, each
found by a test written to check a claim rather than to pass: `bias` silently replacing where
`couple` sums, a build counter measuring the wrong scope, and a TV threshold set tighter than its
own noise floor.

### Phase 1 — The certificate and the oracles

This is the differentiator. It ships before any new surface, because everything we will later claim
depends on it.

**1.1 `Certificate`.** ✅ **DONE** — `src/certify.rs`.

Computed **from samples alone**, never from the sampler's own account of itself — a sampler cannot
certify itself any more than a witness can corroborate their own testimony, and taking only samples
means a deliberately broken sampler can be handed to the same function. It is, in the tests.

| Field | How |
|---|---|
| `beta_eff` + CI | Pseudolikelihood MLE by **bisection**; CI widened for autocorrelation |
| `tau_int`, `ess` | Sokal windowing over **energy *and* magnetization**, worst reported |
| `tv_exact` | Exact enumeration where n ≤ 20 |
| `noise_floor` | Always beside the distance |
| `findings` | Empty is the only thing that means passed |

**Wired into every surface.** `/v1/sample` and `ferrotherm_sample` return a certificate on every
call; it is not a separate endpoint and cannot be skipped. Sampling now runs `sweeps` as burn-in and
then records `draws` (default 128, `thin` between them), which costs more than returning a single
state — and the ledger says so, because a run that returns one state cannot be checked at all.
Retained draws are capped to about 20 MB of spins on large graphs, reporting a thinner certificate
rather than fabricating one or allocating a gigabyte.

**Accepted — a certificate that cannot fail is not a certificate.** Deliberately broken samplers are
caught in each mode: one running at β = 1.4 while claiming 0.6 (`beta_eff` recovers 1.4), an
unburned 24×24 lattice still coarsening out domains (τ = 20 against τ = 0.8 burned in), critical
slowing on a 12×12 lattice at β ≈ β_c, and uniform noise, which fits β ≈ 0 as it should.

Live through the API, a 24×24 lattice with no burn-in comes back with both diagnostics firing:
*"400 draws are worth about 3 independent samples"* and *"early draws average −0.0197 and late ones
−0.9846, a gap of 6.2 standard errors"*.

**1.2 `SampleSet`.** ✅ **DONE** — `src/samples.rs`.

The certificate had no producer. Its signature takes `samples: &[Vec<i8>]` and a trace, and nothing
in the crate returned either — five places hand-wrote the same burn-in/thin/collect loop, and every
one of them appended the sampler's state directly rather than READING it, so every certified run in
the repository priced its readback at exactly zero on a device where one read is worth 239 Gibbs
cycles. Section 0 says every player returns "best found"; this crate did too.

A sample set carries the distribution its states came from and **refuses** where there is none:
averaging spins over a tabu search's trajectory yields a number the same shape as `⟨s_i⟩` and
estimates nothing. Three provenances answer, each deflated by its own correlation structure —
a chain by `tau_int`, population annealing's final population by its family statistic `ρ`,
exhaustive enumeration by nothing at all, because nothing was sampled.

| | independent draws | measured coverage of a 95% interval |
|---|---|---|
| corrected `sqrt(var/ess)` | `N / (2·τ_int)` | 94.6%–100% where the chain is long against its own τ |
| naive `sqrt(var/N)` | `N` | **24.0%** at τ = 32 |

`examples/interval_calibration.rs`, 24 chains × 20,000 draws × every site × three models × four
temperatures, against exactly enumerated marginals. The limit is printed with the result: where τ
runs to hundreds, τ is itself an estimate from a chain barely long enough to make it, and 11 of 24
seeds clear the `Undermixed` finding with 80.7% coverage among exactly those.

Reaches Rust, C, Python, Zig, Julia, HTTP, MCP and the browser workbench, which is where the
interval stops being an argument and becomes a reading: at `β_c` on a 16×16 lattice the workbench
reports 2,000 draws, 2,000 distinct, `tau_int` 245 and an effective sample size of **4**, so
`⟨M⟩ = 0.103 ± 0.71`.

**1.3 The same gap one layer up.** ✅ **DONE** — `model::distinct_optima`.

`docs/graph.html` edits models rather than samplers, so a sample set on a `Sim` was the wrong shape
for it — but the gap it had was the same one: `solve_best_with` runs `tries` independent anneals and
keeps one, and a modeller reading a single assignment cannot tell a unique optimum from one of four.
Every try is kept now, and the editor reports the distinct optimal assignments by name with the
sentence that stops a count reading as a census: independent anneals are evidence the optima they
found exist, not a proof there are no others.

*Five findings from building it, each from a test that failed for a real reason:*

1. **Newton diverged**; uniform noise pinned at the clamp instead of reporting β ≈ 0. Far from the
   optimum the logistic saturates and the second derivative underflows. The log-likelihood is
   concave, so its derivative is monotone and **bisection is exact** — the robust method was also
   the correct one.
2. **`beta_eff` is a *local* statistic and cannot see burn-in.** A chain trapped in a metastable
   configuration still has locally correct conditionals. That needs a separate Geweke-style
   comparison of early against late draws.
3. **Energy autocorrelation misses frozen configurations.** An ordered lattice sits in one basin
   while its energy jitters quickly around a fixed value, so an energy trace reports fast mixing for
   a chain that has not moved. Magnetization sees exactly what energy misses; both are measured.
4. **Two break-mode tests asserted on systems that do not break.** A 12-node 1D ring has no ordered
   phase, so it equilibrates fast at every temperature — certifying it clean was correct behaviour,
   not a missed detection. Replaced with 2D lattices below and at criticality, chosen from measured
   τ rather than assumed.
5. **An `ess < 30` threshold let a genuinely undermixed run through** — τ = 43, ess = 35 out of
   3,000 draws. Thresholds are now round numbers chosen against measured chains rather than picked
   to make the suite pass.

`examples/certify_probe.rs` prints the table those decisions were made from.

**1.2 The oracle set.** ◐ **MOSTLY DONE** — `src/oracle.rs`, `src/planted.rs`.

A `Solver` trait with `Exhaustive` (exact to 26 spins), `SteepestDescent` (restarts are a
*parameter*, because comparing against greedy-from-one-start flatters everything), `Annealer`, and
`RandomGuess` — which **exists to fail**. Every quality test runs against it, and any test it passes
is a test that measures nothing.

`planted.rs` gives the true optimum at any size by construction: choose the ground state first, then
build frustrated cycles around it. The argument is in the module docs and the tests do not take it
on trust — they enumerate small instances and confirm the planted state really is the optimum, over
a grid of sizes, densities and seeds.

*The finding worth keeping.* Difficulty on this family is **not monotonic** — it is an
easy–hard–easy transition peaking near four planted loops per edge, where greedy solves only 4 of 16
seed pairs against 16 of 16 at both extremes. Too few loops means no competing constraints; too many
means the accumulated couplings concentrate toward their mean and the instance relaxes back into a
gauged ferromagnet.

Both obvious guesses were made here in turn and both were wrong: first that denser is harder, then
that the family is uniformly easy. The second came from a probe averaging over matched seeds, which
hid the peak completely — the shape only appears when the solve **rate** is measured across a seed
grid rather than the mean excess. `examples/planted_probe.rs` prints the table.

One caveat carried in the docs: 2D spin-glass ground states in no field are polynomial-time
computable by minimum-weight matching, so nothing here is hard in the complexity sense. It is a
benchmark for *heuristics*, and must be described that way.

**The Wishart planted ensemble** is in, and fills exactly the gap the lattice family leaves. Columns
drawn Gaussian but projected orthogonal to the planted state give `E(s) = (‖Wᵀs‖² − tr(WWᵀ))/2n`, so
the planted state drives a non-negative quantity to zero and is a ground state by construction, with
the optimum known in closed form. Enumeration confirms it at n ≤ 16; the closed form matches to 1e-9
at n = 80.

Its ruggedness knob is **monotonic**, unlike the lattice family: greedy solves 4/16 at α = 0.2 and
16/16 at α ≥ 1, matching the published ensemble. And the two families **fail differently**, which is
the more useful half of the measurement — a lattice miss can be 17% above the optimum, a Wishart
miss under 2%. The Wishart landscape is dense with near-degenerate minima, so a solver gets very
close and still misses. **Any benchmark reporting mean excess would call it easy.** Report the solve
rate.

**Variable elimination** (`src/exact.rs`) closes the tree-decomposition item and does more than the
plan asked. Cost is `2^width` in the induced width rather than `2^n` in the size, so exactness
becomes a property of a graph's *shape*: a 2,000-spin chain is instant, a 240-spin lattice strip is
exact, and a dense graph is refused with a message saying to use a planted instance instead.
Min-sum gives the exact ground state, sum-product the exact `log Z` — both checked against
enumeration, and back-substitution checked separately, since a recovered state can be wrong
independently of its energy being right.

Measured rather than assumed: min-fill ordering is **optimal up to width 5** and drifts two or three
above beyond it (8 where treewidth is 6, 11 where it is 8). At `2^width`, three over is an eightfold
price. The docs carry the table so nobody assumes the heuristic is exact — a bad order makes this
slow or refused, never wrong.

**Still to come:** a planar exact solver (minimum-weight perfect matching), 3R3X XORSAT, and the
physics oracles beyond Onsager — the SK transition and 3D Edwards–Anderson against OPUSLab's
published files.

### Phase 2 — Backends and the honest benchmark

**2.1 Chromatic block-Gibbs on WebGPU.** ✅ **CORRECT** — `src/wgsl.rs`, dispatched from the
workbench. Emitted from Rust the way `hdl.rs` emits Verilog, with **no new dependency**; the FFI
hands the browser the shader text and the padded layout so there is one source of truth.

**Verified on real hardware**: the local field every GPU lane computes, against the field this crate
computes on the CPU for the same state, agrees **exactly** across all 512 nodes of the first colour
class. Only the first class can be compared — by the time the second computes its fields the first
has been resampled, which is what chromatic Gibbs does, so comparing it would measure the schedule
rather than the arithmetic.

*Getting there took four measurements and the first three of my hypotheses were wrong.*

The shader had **two WGSL compile errors**, and their symptom was nothing at all: `class` is a
reserved keyword and WGSL refuses to guess a precedence between `*` and `^`. An invalid shader module
makes an invalid pipeline, and an invalid pipeline's dispatches are a **silent no-op** — the sweep
appeared to run and changed no state, which reads as a sampling bug. I diagnosed it as an RNG
problem, then as a buffer problem, then as a uniform-alignment problem, and it was none of those.
What found it was making failure loud: `createComputePipelineAsync` plus `getCompilationInfo`, which
the workbench now always checks. Two tests pin the language rules, since compiling WGSL in the test
suite would mean taking a dependency.

**The crossover, measured** across a size sweep on one machine, in-browser:

| nodes | CPU upd/s | GPU upd/s | speedup | cpu E/n | gpu E/n |
|---|---|---|---|---|---|
| 256 | 1.97e7 | 3.03e6 | 0.15× | −1.484 | −0.969 |
| 1,024 | 3.20e7 | 1.77e7 | 0.55× | −1.453 | −1.258 |
| **4,096** | 3.20e7 | 6.50e7 | **2.03×** | −1.309 | −1.418 |
| 16,384 | 1.92e7 | 2.09e8 | 10.9× | −1.286 | −1.331 |
| 65,536 | 1.85e7 | 6.14e8 | 33× | −1.284 | −1.285 |
| **160,000** | 1.90e7 | **1.23e9** | **65×** | −1.254 | −1.255 |
| 360,000 | 1.91e7 | 1.16e9 | 61× | −1.193 | −1.191 |
| 810,000 | 1.98e7 | 1.03e9 | 52× | −1.135 | −1.138 |

The GPU **loses below about 4,000 nodes**, where two dispatches per sweep are fixed cost against
too little work. It peaks near 65× at 160k and then declines as it goes memory-bound.

The right-hand columns matter more than the speedup: energy per node agrees between backends to
three decimals at every size above the crossover, so the distributional agreement holds across the
sweep rather than at one point.

**Calibrate the headline before quoting it.** The peak is 1.23e9 node updates/s, which is
**1.2 flips/ns** — an order of magnitude *below* the ~60 flips/ns baked-in FPGA prior art
(arXiv:2303.10728) and two below the 185 flips/ns at 9.168 W Alveo figure. What this is, is an open
sampler that runs at that speed **in a browser tab with no install**, which is a different claim and
the only one we should make.

**2.2 The benchmark, reported the way nobody reports it.** ◐ **The metric exists and is measured;
the cross-vendor port is still open** — `meter/examples/joules_per_sample.rs`.

Joules per **independent** sample is now a measured quantity on real silicon, not a proposal:
`J/indep = joules_above_idle / ESS`, `ESS = draws / 2·τ_int`, with τ in one unit for both arms and
every figure carrying its own error. Measured on a Pegasus P₈ spin glass on an Apple M5 Max, plain
chromatic Gibbs against parallel tempering at an **iso-flip** budget on a ladder tuned by
`adaptive::adapt` until its worst pair is alive:

| β | J/flip | J/independent sample |
|---|---|---|
| 0.5 | Gibbs cheaper by **2.6×** | **PT cheaper by 1.9×** (ESS 252 / 1252) |
| 1.0 | Gibbs cheaper by 1.9× | **PT cheaper by ≥604×** — Gibbs never decorrelated once |

**The two metrics name different winners on the same run**, and only one of them is about the cost
of getting an answer. Below β = 1.5 tempering's own ESS falls under 25 and those rows print
`unresolved` rather than an estimate. A sampler that produced less than one independent draw is
reported as a *bound* (`≥X J`), which is a fact rather than an extrapolation.

Two things it measures about the instrument, not the samplers: the same Gibbs kernel spans
1.56e-7 to 2.27e-7 J/flip across rows — **46%**, this machine's reproducibility floor on any J/flip
claim, ours or anyone's — and a hand-picked 12-rung geometric ladder over this range has its coldest
pair at acceptance **0.000** with zero round trips, which is why the ladder is tuned and escalated
rather than chosen.

*Still open:* the port of THRML notebook 02 itself, and the cross-vendor comparators — Extropic's
8×B200 JAX figure, the ~60 flips/ns baked-in FPGA prior art (arXiv:2303.10728), and the 185 flips/ns
at 9.168 W Alveo result.
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

**3.0 A machine to embed ONTO.** ✅ **DONE** — `device::pegasus`, `device::zephyr`.

Everything below assumed a target and the crate had the wrong one: `embed` did honest minor
embedding onto **Chimera**, which D-Wave retired, while `fabric` described "a 5,640-qubit Pegasus"
it could not build. `P₁₆` now comes out at 5,640 qubits / 40,484 couplers at degree 15 and `Z₁₅` at
7,440 / 71,736 at degree 20 — the Advantage's and Advantage2's published figures, produced from the
coordinate rules and checked at five sizes each against D-Wave's own generator on counts *and* full
degree histograms. `Topology` carries the vendor's qubit numbering beside the graph, because
Pegasus's is sparse and a chain in our indices would program different qubits.

`examples/embedding_tax` prices the generations in counts, never seconds: at `K₁₆` Chimera spends
126 sites and an 18-qubit chain against Pegasus's 49 and 7 and Zephyr's 48 and 6.

**3.1 COPY-gate sparsification.** ✅ **DONE** — `src/sparsify.rs`, and **DSATUR** is in
`graph::color_for`.

Exact and ground-state preserving, with the property *enumerated* rather than argued: the whole
sparsified state space is walked, and every ground state must have its copies agreeing, must project
onto a ground state of the original, and every ground state of the original must be reached. A
companion test drops the copy coupling below the derived bound and requires the property to fail.

**The degree budget in the line above was wrong and the correction is the interesting part.** A path
of `c` copies spends one coupling at each end and two in the middle, so it offers `c(d−2) + 2` ports
and a variable of degree `k` needs `c ≥ ⌈(k−2)/(d−2)⌉` — not `⌈k/c⌉ + 1`. That is character for
character `embed::site_lower_bound`'s per-variable term, because a chain of hardware sites and a path
of logical copies are the same object seen from opposite ends. A test asserts the two agree.

DSATUR was added the day the crate acquired a graph greedy colours badly: it saves a sweep pass on
Zephyr (6 → 5) and on compiled counting constraints (4 → 3), ties on Pegasus where greedy already
matches the clique bound, and is adopted only when it strictly wins.

**4.1 Free energy as a certified quantity.** ✅ **DONE** — `src/free_energy.rs`,
`ebm::log_likelihood_ais`.

The certificate said the chain mixed; nothing said what the distribution is. `ln Z` now comes
three ways — AIS with an *unconditional* Markov lower bound, reverse AIS with the conditional upper,
thermodynamic integration with a monotonicity bracket, Bennett acceptance ratio for `ln Z` at every
rung (and with it entropy and heat capacity), clamped AIS for the EBM numerator past enumeration —
cross-checked against enumeration, elimination, the transfer matrix, Onsager and `popanneal`.
✅ **The observer is in**: `tempering::parallel_tempering_observed` records per-rung energy traces,
so one run gives the optimiser's best AND the free-energy curve, with recording proved bit-identical
to the unobserved loop and warm ladders refused an absolute anchor. `adaptive::adapt_observed`
records the final (respaced) epoch: measured at EQUAL RECORDED SAMPLES, respacing evens the swap
rates as designed (spread 0.40 → 0.15) and tightens the reported step error in 12 of 12 seeds, but
the ACTUAL error is unchanged (0.047 either way) — the ladder got better conditioned and the
estimate did not get better.

And having the traces immediately caught a defect: `bar_ladder`'s quadrature error bar understates
the true spread by ~30% (`sd(z) = 1.28`), because adjacent steps share the samples at their common
rung. `LadderTraces::log_z_total` block-jackknifes the total instead and comes out conservative
(`0.81`). Open: calibration is not established across scales (900 samples → 1.5, 9,000 → 1.4) and
per-rung `tau_int` is only 1.4, so the residual is the ladder's ROUND-TRIP time, which no current
measurement resolves.

**4.2 Learning theory as oracles.** ✅ **DONE** — `src/meanfield.rs`, `src/hopfield.rs`.

Mean field with the Gibbs–Bogoliubov bound (a deterministic lower bound on `ln Z`), TAP, belief
propagation exact on trees; the Hopfield memory against Curie–Weiss and the AGS replica theory,
capacity `α_c = 0.1379` reproduced by bisection on the crate's own numerics, retrieval and its
absence confirmed by the samplers. Recorded gaps: replica-symmetry-breaking corrections near
`α_c`; the Gardner/perceptron storage problem (a higher-order model, not pairwise — needs the
`hubo` path); modern (dense associative) Hopfield and equilibrium propagation are path item 5.

**4.3 Modern memory and equilibrium propagation.** ✅ **DONE** — `src/dense_memory.rs`,
`src/eqprop.rs`.

Dense associative memory with the degree-2 identity to the Hebbian energy, exact fixed-point
capacity, attention as the exponential memory's update; equilibrium propagation for Boltzmann
machines with its theorem held at both rates. The memory is a program (`to_hubo`, `to_program`):
native annealing retrieves; the pairwise reduction is exact and measured dynamically frozen (0 of
5 anneals, 21 ancillas behind a penalty 50× the signal) and unplaceable at max degree 28; degree 2
reaches its exact ground state on a Chimera at a chain strength set relative to the couplings.

**4.4 Gardner's storage problem.** ✅ **DONE** — `src/perceptron.rs`.

The binary perceptron: first-moment bound (exact at finite `N`, both parities), Krauth–Mézard
cited, enumeration by Gray code. Measured: no algorithmic gap at enumerable sizes (313 of 314),
and the gap opening with `N` (20/20 → 0/20 at `α = 0.5` from `N = 21` to `401`). The spherical
case is done too — `gardner_capacity` computed in closed form (exactly 2 at `κ = 0`, which needed
a machine-precision `erf`), minover measured to fail on a budget below the capacity and on the
model above it. Recorded gap: continuous-state units *inside the EBM graph*, which the spherical
perceptron did not need (its couplings are a standalone vector) but continuous Hopfield and the
original EqProp formulation do.

**4.5 Continuous units.** ✅ **DONE, including the general nonlinear case** — `src/continuous.rs`.

`Gbm` is a Gaussian–Bernoulli Boltzmann machine with exact oracles at every level: `N(A⁻¹b, (βA)⁻¹)`
and closed-form `ln Z` with no spins, exact enumerated `ln Z` and marginals with them. Chosen
because it is checkable; a continuous sampler that cannot be checked should not be trusted.

The recorded gap — continuous Hopfield's graded response and EqProp's original formulation need a
general nonlinear unit, whose conditional is not Gaussian and which therefore has none of those
oracles — **closed at 0.39.0.** `ContinuousEbm` carries a `Potential` (quadratic, Hopfield-tanh,
double-well) sampled by Metropolis-within-Gibbs, and `chain_log_z` is the oracle it was missing: a
transfer-operator `ln Z`, `O(n · grid²)`, exact to the grid at any chain length, against which the
nonlinear sampler is verified at twelve units. **Still open, and smaller than it was:** non-chain
topologies past three units, where the transfer operator does not apply and quadrature is the only
oracle left. This entry said "open" for a release after it was closed, which is its own small lesson
about roadmaps.

**3.4a Pegasus's ceiling, proved.** `K_{12(m−2)+4}` is optimal for chains made of one vertical and
one horizontal wire segment: the interior segments are forced (`α = 0`, `β = 1`, so `w ∈ [1, m−2]`)
and the boundary admits exactly two tracks at each end (10 and 11 hot, 0 and 1 cold), proved in Kani
and measured on P₄/P₅/P₈. Every connected candidate is already mutually adjacent, so adjacency is
not the limit — connectivity is. Routing at FRAGMENT granularity (six per qubit) closes the gap
entirely; see `embed::pegasus_clique_fragment`.

**3.4 Structured clique constructions.** ✅ **All three fabrics; Zephyr AT the frontier** —
`embed::pegasus_clique`, `embed::zephyr_clique`, `embed::chimera_clique`.

The crossover tables ran the heuristic embedder on both sides; for cliques the frontier is a
*construction*, not a search. `zephyr_clique` places `K_{2t(2m−1)}` at uniform chain `m+1` — `K_232`
on Z₁₅, **exactly busclique's size and chain**: the measured crossing law (`zv = (wh−jv)/2`, no
offset dependence) makes the two `j` phases first-class tracks, no odd-coupler fusion needed.
`pegasus_clique` places `K_{12(m−2)+4}` — `K_172` on the Advantage's P₁₆, ells at `m+1` plus the
four universal wires the offsets provably admit (exactly four: tracks {0,1} at `w = m−1`, {10,11} at
`w = 0`). `pegasus_clique_fragment` then builds on the fragment grid and reaches **`K_{12(m−1)}` = `K_180` at
chain 17 — busclique's frontier exactly, size and chain both**, at every size from P₃ to P₁₆, against
the heuristic's `K_80`. All coverage and quantifier arguments are Kani theorems (exhaustive to
`m = 2¹⁶`); every size passes `Embedding::verify`, and every Pegasus size is checked against
`minorminer.busclique` on the same graph. **No structured clique bar remains on either fabric.**

**3.2 2D adaptive parallel tempering over (β, W0)**, so nobody hand-tunes copy strength.
*Partly answered another way:* `sparsify::copy_strength` derives a sufficient `W0` from the model's
own weights rather than tuning one, and the derivation is checked by enumeration — every ground
state of the sparsified model has its copies agreeing, and dropping `W0` below the bound breaks that.
Adaptive tempering would find a SMALLER working `W0` than the derivation's; whether that is worth
having is now a question about the margin rather than about correctness.
*Why:* every sparsification introduces a penalty that must be tuned by hand. OPUSLab states the
problem in exactly those terms; their answer is one MATLAB file from June 2025 with no adoption, and
their repo literally named `SparsifyDenseGraph` is empty.
**Accept:** across the planted-instance suite, adaptive tempering matches or beats the best
hand-tuned `W0`, with copy-agreement certified at readout.

**3.3 Publish the crossover.** ✅ **DONE, and the answer is NO CROSSOVER** —
`examples/sparsify_vs_embed.rs`.

At what `N` does sparsify-plus-embed beat placing the model directly? **Nowhere, on either machine.**
`K₂₄` onto a Pegasus P16 costs 130 sites and a 14-site chain placed directly, against 758 sites and a
55-site run through sparsification — 5.8× the qubits and 3.9× the length of the thing that has to
agree. At `K₃₂` the sparsified model does not embed at all while the direct route places it in 237.
Below `K₂₄` the model already fits the machine's degree, so `sparsify` returns it unchanged and the
two routes are one route.

It is the same tax paid twice: copies are chosen before the machine is looked at, and the embedder
then chains every one of them. So **where a placer exists, place** — and sparsification is for a
fabric with a fixed sparse topology and no placer at all, where the question is not which is cheaper
but whether the model runs.

That is the opposite of what a paper introducing a sparsifier would conclude, which is why it was
worth measuring rather than assuming. *Still unmeasured:* the pc-COP comparison — 2048 fully
connected p-bits with zero embedding tax — needs a dense fabric this crate does not model yet.

### Phase 4 — Surfaces

Adapters, not a DSL. The field does not need a fifth modelling language.

| Surface | Design | Priority |
|---|---|---|
| **Rust** | Native. `Model` → lowering → `Program` → `Schedule`; traits for `Factor`/`Kernel`/`Sampler`. One variable below the lowering passes: the spin. Newtype indices, not THRML's metaclass ID hack. | now |
| **Python** | `ferrotherm.linalg` in thermox's signatures + `estimate/stderr/certificate/ledger` on every result. A PyMC `BlockedStep` with a `Competence` rule, so a discrete-latent model gets our block-Gibbs step in one line. | now |
| **Node graph** | ✅ `docs/graph.html`. Nodes are IR objects, not widgets — model, schedule, run, report — and the ports carry those kinds, so a graph that type-checks is a program that compiles. Runs on the wasm module and exports `.ftp`. The survey found no node editor exists for **any** Ising or factor-graph stack anywhere; this is the first. | done |
| **Zig** | Already binds the C ABI. Extend as the IR lands. | next |
| **Embedding, everywhere** | ✅ Both routes onto a sparse fabric now cross the ABI: `ft_embed`/`ft_embed_apply`/`ft_unembed` beside `ft_sparsify`, with `ft_site_lower_bound` as the proof-carrying refusal. Until this landed a non-Rust caller could rewrite a model but not place one, and so could not run the 3.3 comparison themselves. 201 symbols, four surfaces. | done |
| **Adapters** (`dimod.Sampler`, `ommx`) | Separate, optional, deletable crates the core never names. Written when a user asks, not before. | later |

**Accept (node graph):** a graph built by dragging produces a `.ftp` byte-identical to the
hand-written one, and the round trip is a test.

### Phase 5 — Workloads

◐ **CATALOGUE PUBLISHED** — [`WORKLOADS.md`](WORKLOADS.md). Five entries, each naming its oracle and
reporting what was measured against it, including where the method stops.

| Workload | Oracle | Measured |
|---|---|---|
| **Sampling-based control** (`mppi.rs`) | closed-form LQR optimum | **7.1%** above the provable optimum **over 200 steps** — the metric grows without bound in run length (1.0% at 25, 22.6% at 800), so it is a coordinate and not a property |
| **Categorical optimisation** (`categorical.rs`) | exact feasibility | domain wall needs a **3× weaker** penalty |
| **Thermodynamic linear algebra** (`tla.rs`) | Gaussian elimination | exact-transition integrator unbiased; EM bias law confirmed |
| **Spin-glass physics** (`ising.rs`, `planted.rs`) | Onsager; planted optima | agrees to **0.0086**, annealed in |
| **EBM training** (`dtm.rs`) | data statistics vs noise | ⛔ **72.9%** closer to data than noise — but taken under a 120-second WALL-CLOCK training budget, so it is a function of machine speed and cannot be reproduced. See WORKLOADS.md; the example is step-bounded now |

The last row is calibrated in the file: per-pixel marginals are a weak metric and this is not the
published FID. Reaching that is a known and affordable run, not a research problem.

Two entries shipped a first test that passed **vacuously**, and both are recorded in place rather
than quietly fixed. That, plus "report the rate, not the mean" — earned four separate times — are
the two rules the file opens with.

**Explicitly not pursued:** routing, scheduling and portfolio optimisation. MILP in a QUBO costume,
and they lose to Gurobi.

---

## 3. The ingest list

What we port to Rust, and why each is worth owning.

| Source | State today | Our port |
|---|---|---|
| COPY-gate sparsification (OPUSLab) | ~102 lines MATLAB, no adoption | ❌ NOT PORTED — subsumed by `embed`, which already splits a high-degree variable into a chain. Going after it uncovered a larger problem: **the placer cannot embed a graph that needs a chain**, see below |
| DSATUR coloring | Scattered | ❌ NOT PORTED, and measured to be the wrong tool: every graph this crate builds is bipartite, greedy already hits the optimum on all but Chimera, and the fix there was a bipartiteness check (3 colours → 2, +32–56% on the parallel path). DSATUR wins on dense irregular graphs; this review did not locate one here that greedy colours suboptimally |
| 2D adaptive PT over (β, W0) | One MATLAB file, June 2025 | ✅ `adaptive` — ladder respacing measured to work; the second axis measured NOT to earn its replicas, including where it was predicted to |
| Lattice Random Walk integrator (arXiv:2508.20883) | **Paper, no code** | Binary/ternary SDE increments, no Gaussian RNG in the datapath, robust to quantisation |
| Thermalizers KL chain-rule bound (arXiv:2608.01615) | **Published without code** | The compile-pass error bound; hardware-independent mathematics |
| Domain-wall encoding (Chancellor 2019) | Published, scattered implementations | Compiler pass: k−1 spins, no penalty term |
| Chook planted-instance generators | Python | Test fixtures with known ground truth |
| Tree-decomposition & planar exact solvers | Scattered literature | Oracles |
| thermox signatures (Normal) | Python, no device hook | `ferrotherm.linalg`, plus certificate and ledger |
| THRML's decomposition | JAX, simulator-only | Inspiration, not a port — Rust traits do it better |

Three of these are papers with no code. Publishing working Rust for them is, on its own, a
contribution the field currently lacks.

### ✅ Closed: the minor-embedding placer could not build a chain when one was needed

`src/embed.rs` did not embed a star with eight leaves onto a 512-site Chimera — the simplest graph
needing exactly one chain — and failed on every clique past `K_7`, at every machine size and round
budget. Two independent defects, both now fixed:

**The root choice collapsed the chain.** `steiner_ish` picked its root by minimum total
shortest-path distance, which is precisely the site whose placed neighbours all sit one hop away —
so every back-walk was a single edge and subtracting the neighbours' own sites left the singleton
`{root}`. The union was seven sites; the *subtraction* collapsed it. The runner-up root would have
given the two-site hub that is the fix, sitting second in the candidate list and unreachable.

**And cliques never reached round 1 at all.** The same subtraction removed sites merely *interior*
to a path routed through a third neighbour's chain, severing the chain from the neighbour it was
built to reach. The round then passed its only test — "is any site shared" — `verify` correctly
rejected it, and `e.verify(...).ok()?` turned that rejection into an immediate `None` for the whole
function. More rounds cannot help a search that quits in round 0, which is why enlarging the machine
and multiplying the round budget were both measured to be irrelevant.

Measured on `chimera(8,8,4)`, 16 seeds, every result verified: a star of 8/12/20 leaves 0→16 of 16;
`K_8`, `K_12`, `K_20` 0→16 of 16; `K_24` 0→15 of 16. On a 736-instance paired corpus the repair
solves 141 instances the original could not and loses 2, with chain length shorter on 68 and longer
on 43 (mean −0.19).

**Saying "no" is no longer free, so it is now bounded twice.** The old placer abandoned the search
on the first unroutable variable, so a hopeless input returned instantly; repairing that meant
`K_100` spent 95 s proving nothing. `site_lower_bound` is a counting argument — a chain of `L` sites
on degree-`d` hardware offers at most `L(d−2)+2` ports — and when it exceeds the machine **no
embedding exists**, which is the one place `None` is a proof. `K_60` and `K_100` are refused in
microseconds; `K_33` and `K_40`, which it cannot rule out, are still searched properly.
`DEFAULT_SEARCH_BUDGET` bounds the rest.

**And repairing it unblocked a measurement that changed a shipped default.**
`examples/chain_strength` could only run at six logical variables, which barely need chains, so the
rigidity half of the trade-off never appeared. At twelve variables with 18-site chains it appears
immediately, and the standard `2×` first guess turns out to break 9.7% of chains: the default is
now `4×`, the first multiple that breaks none.

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
v0.6  Phase 0 ✅     one kernel, β freed, encoding passes, .ftp format  breaking
v0.7  Phase 1        Certificate + oracle set + planted instances
v0.8  Phase 2.1/2.2  WebGPU backend + the honest benchmark
v0.9  Phase 3        sparsifier, DSATUR, 2D adaptive PT, crossover
v0.10 Phase 2.3      Pt V2 sequential fabric; one .ftp on three backends
v1.0  Phase 4/5      node graph, Python adapters, MPPI flagship
```

Phase 0 first, and alone, because every later phase inherits its mistakes. Phase 1 before any new
surface, because the certificate is what the surfaces are for.
