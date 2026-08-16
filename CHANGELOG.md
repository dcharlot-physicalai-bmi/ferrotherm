# Changelog

## 0.10.0 (2026-08-15)

**Every capability the landscape survey measured, and one side of the energy claim now measured
rather than borrowed.** Breaking: `Ledger::joules` returns `Option<f64>`, `Prices` gained a `source`
field, `Constraint` and `CompileError` gained variants. Two new sibling crates: `ferrotherm-gpu`
(native GPU sampling) and `ferrotherm-meter` (measured wall power).

This release closes the last four rows in `docs/LANDSCAPE.md` where any surveyed competitor —
Extropic, D-Wave Ocean, Fixstars Amplify, Jij, PyQUBO, the open-source Ising layer, or the hardware
vendors' own SDKs — had something this stack did not. Nine of the fifteen capabilities are ones no
other surveyed stack has at all.

### `ferrotherm-meter` — measured wall power, so a joules figure describes the machine that produced it

The ledger counted operations exactly and priced them against the one table in the tree: `Z1_SPICE`,
pre-silicon estimates for an accelerator nobody has characterised. `Prices::UNSTATED` made borrowing
it honest. This makes borrowing it unnecessary.

- **Measured on an Apple M5 Max: 4.261e-7 J per node update** — whole-system wall power above idle,
  over an 8.29 s window with 75 power readings. `Z1_SPICE` estimates 7.09e-15 J. The ratio is 6.0e7,
  and it is *the size of the prize being claimed, not a measured speedup*: one side is a
  general-purpose CPU at the wall, the other a per-device SPICE estimate for silicon that has not
  been fabricated. One side is now measured.
- Std-only, no dependencies — the backend is a subprocess (`macmon`), and one field out of a JSON
  line does not justify a parser.
- **Four things it refuses**, each found by it happening: a workload too short for the backend's tick
  (a mean of two readings is not a mean); a run drawing *less* than its baseline (which means the
  baseline was taken on a busy machine — the first version clamped this to zero and thereby reported
  that computing is free); a delta inside the baseline's own 3σ wander (two runs of one workload
  reported 2.34e-8 and 1.81e-7 J/update, an 8× spread, because one added ~1 W to a ~60 W baseline);
  and a mixed workload turned into a per-sample price (three unknowns, one equation).
- Settling is part of the protocol, measured rather than guessed: a baseline taken right after a
  heavy run read 68.5 W while the run itself averaged 64.1 W. Fans and thermal management lag the
  workload, so idle straight after load is systematically *higher* than idle.
- Jetson/Linux INA3221 rails are the same measurement and are **not implemented**: that host has been
  offline for a week, and a backend nobody can run is a backend nobody has tested.


### A joules figure now has to say whose machine it describes

**Breaking:** `Ledger::joules` returns `Option<f64>`, and `Prices` gained a `source` field.

Every fabric in the tree declared `Z1_SPICE`. So a Hitachi CMOS annealing ASIC and a laptop CPU both
reported Extropic's pre-silicon SPICE estimates as their own energy, and the HTTP surface reported
them for every run including a plain CPU sample. Nothing was lying — nothing had been asked to say
whose numbers these were.

- `Prices::source` travels with the numbers. `Prices::UNSTATED` is what "nobody has published this"
  looks like in the type system; Hitachi and the CPU device declare it.
- `joules` returns `None` rather than zero for an uncharacterised device. Zero is a claim that it
  costs nothing.
- HTTP reports exact **counts** always, `joules` as null when unpriced, and `priced_as` generated
  from the prices — a hardcoded note is how a figure ends up labelled with the wrong machine.

### The write is charged

`Ledger::writes` was incremented by **nothing** in the library, so every figure the stack produced
was a sample-and-read story with the expensive term silently zero — on this hardware class a write
costs roughly 21,700 samples, which is the module's entire thesis. `Device::program` *is* the write;
the trait says so now, and the CPU and Hitachi implementations charge one per node. A demonstration
run reports the write at **100%** of the projected energy.

### `reflash_hz_cap` is read

It was declared and consulted by nothing. `Ledger::reflash_seconds` turns it into a wall-clock floor:
a workload that reflashes the whole graph faster than the device sustains is not fast, it is
unphysical, and pricing it describes a run that could not have happened.


### The compiler says what it knows

`Slot::add_penalty` has always returned whether the encoding it added can be exact, and **both
callers threw it away**. `Compiled::caveats` now carries it, on every surface.

- A binary encoding of *k* values uses ⌈log₂ k⌉ spins, spelling 2^⌈log₂ k⌉ codewords. When *k* is not
  a power of two the spare codewords decode to nothing — and **no penalty removes them**. Measured
  on k = 6: the cheapest invalid state costs `0.00` and so does the cheapest valid one, so the
  sampler has no reason at all to prefer an answer. A test enumerates all eight codewords, so the
  message cannot drift from the physics it describes.
- `ft_model_caveats`/`ft_model_caveat` (C ABI and header), `Answer.caveats` (Python),
  `Problem.caveats`/`caveat` (Zig), `caveats(a)` (Julia), `"caveats"` (HTTP, MCP), and the editor,
  where a caveat now outranks "every constraint holds" in the status line — that is exactly the case
  where the answer looks clean and the model is not.

### Encoding selection reaches the last two surfaces

HTTP and the node editor **silently ignored `"encoding"`**: a document asking for binary got one-hot,
with a different spin count, a different penalty and no error — the reply's own `ftp` said `onehot`
and nothing else did. Both now honour it and refuse an unknown name by listing the ones that exist.


### Native GPU sampling — `ferrotherm-gpu`

The last row in `docs/LANDSCAPE.md` where a surveyed competitor had a sampler this stack did not.
GPU sampling existed only in the browser; the native core was CPU-only.

- A **separate crate**, beside `silicon`, `serve` and `cloud`. `ferrotherm` stays std-only with zero
  dependencies, which is what lets the same source compile to `wasm32-unknown-unknown` and to a
  microcontroller — a property worth keeping rather than a slogan.
- It has **no shader of its own**. The WGSL comes from `ferrotherm::wgsl::sweep_shader`, the same
  string the browser fetches through `ft_shader`, so a native run and a browser run cannot disagree
  about the arithmetic — only about the hardware.
- **Measured on an Apple M5 Max (Metal), median of 5 runs, 200 sweeps:** crossover at ~4k nodes,
  3.5× at 4,096, 14× at 16,384, 42× at 65,536, 54× at 262,144. The GPU **loses** below ~1k nodes and
  the benchmark says so: fixed cost per run does not shrink, so the crossover is the number worth
  quoting rather than the peak.
- `Gpu::is_hardware()` and the benchmark refuse to quote a speedup against a software rasteriser.
  lavapipe, SwiftShader and WARP all run this shader correctly and say nothing about a GPU.
- A readback that is not ±1 is **refused, not coerced** — the same discipline the browser path
  learned, where `> 0 ? 1 : -1` laundered a dropped dispatch into a believable energy.

### Fixed before it shipped

- **One command submit per dispatch made the GPU slower than the CPU at every size**, at a near
  constant ~60 ms regardless of node count — constant time under a growing workload is the signature
  of paying for driver round trips rather than arithmetic. Now one encoder, one pass and one submit
  for the whole run, with per-dispatch parameters selected by dynamic uniform offset. 60 ms → 2.5 ms,
  and the ratio at 65k nodes went from 0.75× to 42×.
- `Limits::downlevel_defaults()` caps storage buffers at 4 per stage and this shader binds 6, so the
  device compiled nothing. The WebGPU baseline is the right floor: anything that runs the page runs
  this.


### all-different, the constraint no pair of variables can express

`Model::all_different` (Rust), `ft_model_var` + close kind 5 (C ABI and header), `all_different`
(Python), `allDifferent` (Zig), `all_different!` (Julia), `{"type": "all_different", "of": [{"var":
name}, ...]}` (HTTP, MCP, node editor). The last gap in the constraint vocabulary against every
other modelling layer surveyed in `docs/LANDSCAPE.md`.

- **Lowered per shared value, not per pair.** For each value two of them could both take, the
  indicators are excluded pairwise — the `AtMostOne` lowering repeated over shared values. No slack,
  no ancillas, and *nothing at all* where the domains do not overlap: `all_different` over `0..=3`
  and `10..=13` adds zero couplings, which a sweep of n(n−1)/2 `not_equal`s would not notice.
- **The violation names which value collided and who took it** — "a and b both take 2" is a repair;
  "all-different was violated" is something the modeller already suspects.
- **The pigeonhole case is refused at compile time, by name.** More variables than the values they
  share between them has no answer at any penalty. Annealing it and reporting `feasible: false`
  reads as "raise the penalty" or "lengthen the ladder", and neither can work. Counting is cheap, so
  it is counted.
- `ft_model_var` appends a variable rather than a literal, picking a value from the variable's own
  domain. The first version passed a placeholder `0` through `ft_model_lit` and was correctly
  refused for every variable whose domain did not contain 0 — that function's whole job is to reject
  a value a variable cannot take, so the fix belonged in the library, where the domain is known.


## 0.9.0 (2026-08-15)

**A constraint can be a price, and surface parity is checked rather than remembered.** Breaking:
`Violation` gained fields, Julia's `Answer` gained three, and Zig's `Error` gained `BadState`.

### Soft constraints — a preference, priced, on every surface

A constraint has always been either a rule or nothing. A soft one is a **preference the solver may
trade away**: breaking it costs, and the answer stays feasible. The distinction is not a shade of
penalty strength — it changes what the answer *means*. A broken rule says the answer is not an
answer; a traded preference says the solver made the choice it was asked to price.

- `Model::soft(c, weight)` and `Model::soften_last(weight)` (Rust), `ft_model_close_soft` /
  `ft_model_soften_last` / `ft_model_soft_cost` / `ft_model_violation_is_hard` (C ABI and header),
  `soft=` on every constraint plus `soften_last` (Python), `countSoft` / `softenLast` / `softCost` /
  `violationIsHard` (Zig), `soft =` keyword plus `soften_last!` / `soft_cost` / `traded` (Julia),
  `"soft": w` on any constraint (HTTP, MCP and the node editor). Eight surfaces, one release.
- `Solution::feasible` ignores soft violations by design; `Solution::soft_cost` totals what the
  trades cost, and each `Violation` carries `hard` and `cost` so the two are never confused.
- **The price is `weight × amount²`.** A constraint becomes an energy term by squaring how far
  outside it sits, so missing by two costs *four* times missing by one. Pricing a preference chooses
  that curve as well as its scale, and reporting a linear price would misstate what was traded.
- The weight is **absolute, not scaled**. Automatic penalty scaling exists to stop a hard constraint
  being outbid by the objective; a soft one is meant to be traded against it.
- A `"soft"` that is not a positive finite number is refused **by name** on every JSON surface
  rather than falling back to a hard constraint — the same shape as five earlier silent-wrongness
  bugs, where an unreadable field became the opposite instruction with `feasible: true` and nothing
  to suggest anything had gone wrong. The editor checks `typeof`, not coercion, so the same document
  gets the same verdict there as over HTTP.
- The editor's status line no longer claims "every constraint holds" over an answer that traded one.

### Parity is now checked, not remembered

`scripts/check-parity.sh` asks, for each of the 84 exported C ABI symbols, whether the header
declares it and Python, Zig and Julia call it. A gap passes only with a written reason in its EXEMPT
table — which is what AGENTS.md has always meant by "the gap is written down", except that until now
nothing read it. It is in CI, and its first run found sixteen real ones:

- **`ft_model_ancillas` reached no binding.** It reports the spins the higher-order lowering added,
  which is the number that says whether *sampling* from a solved model is sound at all — the
  reduction is exact for optimisation and not for the Boltzmann distribution. Zig even had a test
  named for ancillas that could not read the count. Now on Python (`Answer.ancillas`), Zig
  (`Problem.ancillas`) and Julia (`ancillas(a)`).
- **`ft_set_spins` was written, unit-tested and called by nothing.** It puts a state computed
  elsewhere into a simulation so the same code scores it — and it refuses a wrong length or a value
  that is not ±1. The browser's WebGPU path wrote straight into wasm memory after coercing every
  value with `> 0 ? 1 : -1`, which launders a dropped dispatch or a short readback into a plausible
  state that is then scored with confidence. The page now goes through the validation; `sim.spins =`
  (Python), `setSpins` (Zig) and `spins!` (Julia) expose it.
- **Violation magnitude never reached Zig or Julia.** `violationAmount` and `amounts(a)`: "at most
  two, and four hold" is over by two, which is a different problem from being over by one.
- **A solved `Problem` could not be certified from Zig.** `Problem.certify` and
  `ProblemCertificate`, separate from the simulation certificate because the exact bounds are not
  available there and half a struct of NaNs would claim they were computed.
- **`exact_ground_state` was Julia-only.** Python and Zig had the ground *energy* and not the
  assignment that reaches it, which is what checking a sampler against the truth needs.
- The header now declares `ft_field`, `ft_shader`/`ft_shader_len` and the seven `ft_gpu_*`
  device-buffer accessors. A header is the ABI contract; selectivity belongs in the bindings above
  it.

### Fixed

- `scripts/check-versions.sh` named `serve` as the only crate pinning the library, so this release's
  bump left `cloud` and `silicon` at `0.8` while the script reported "all agree". It now *finds*
  every dependent instead of listing them, with a floor so a search that stops matching fails rather
  than passing over nothing. An exemption in `check-parity.sh` naming a symbol that does not exist
  now fails too — a table of reasons nobody can check is the thing that script replaced.
- **`ferrotherm_jll` has never been installable, and its check passed anyway.** The Julia artifacts
  were built, hashed, and uploaded to a CI artifact that expires — while the `Artifacts.toml` shipped
  in the package named `releases/download/vX/…` URLs that no release ever carried. Every one has
  been a 404 since 0.7.0. The verification step rewrote those URLs to a `localhost` server before
  testing them, so it reported on the stand-in rather than on what a user would fetch. The release
  job now creates the release and attaches the tarballs, then fetches each **published** URL and
  checks the bytes against the hash the manifest commits them to — with a floor, so finding no URLs
  fails instead of passing over nothing. The manifest is also committed back into the package: the
  one in the repository sat at 0.8.0 hashes while the library moved on.
- `Solution::soft_cost` returned `-0.0` when nothing was traded. Rust's `Sum for f64` folds from
  `-0.0`, which is the correct additive identity and prints as a minus sign through every binding
  that formats a float. A price with a minus sign in front of it reads as a credit.

## 0.8.0 (2026-08-13)

**A fabric describes a real machine, and a model can be wider than pairwise.** Breaking: the
precision field changed type, `Fabric` gained fields, `Unsupported` gained variants.

### Higher-order models

- `reduce::to_pairwise` lowers a k-body model onto pairwise hardware. An ancilla spin is defined as
  the product of two existing ones and substituted wherever that pair appears; the pair chosen each
  round is the commonest, so three 3-body terms sharing a pair cost **one** ancilla, not three.
  It goes through binary because in spin space *"t equals s_a·s_b"* is itself a three-body
  statement — Rosenberg's `3y + x_a·x_b − 2x_a·y − 2x_b·y` is quadratic throughout.
- **Verified by enumerating every state of both models.** For each assignment of the original
  spins, the reduced energy minimised over the ancillas equals the original plus one constant, so
  nothing is reordered and the ground states correspond exactly.
- **The guarantee is about optimisation.** The ancillas add states, so the Boltzmann distribution
  over the original variables is not preserved at finite temperature. Read `reduce`'s docs before
  sampling from a reduced model.
- `Model::compile` accepts a cubic or wider objective and applies the pass; `Compiled::ancillas`
  reports the price. Whatever expands to degree 0, 1 or 2 goes straight into the graph, so only
  what is genuinely wider is charged.
- Reachable everywhere: `Expr::product` (Rust), `ft_model_objective_product` (C),
  `a.is_(1) * b.is_(1) * c.is_(1)` (Python), `preferAll` (Zig), `(w, (l1, l2, l3))` (Julia),
  `"of": [...]` (HTTP/MCP), and a variadic **Prefer all together** node in the editor.

### Fabrics

Six machines declared from vendor documentation, each number cited where it is used:

| fabric | from | notable |
|---|---|---|
| D-Wave Advantage2 | topology docs, GA announcement | Zephyr, 20-way, `j_range [-1,1]` |
| D-Wave Advantage | topology docs | Pegasus, 5,640 qubits, 15-way |
| Fujitsu DA3 | Fujitsu's API documentation | 100,000 bits, **fully connected**, 64-bit integers |
| Toshiba SQBM+ (QUBO) | Toshiba's user manual | 10M variables, **float32** |
| Toshiba SQBM+ (PUBO) | same | **order 4 natively** — no reduction needed |
| QBoson CPQC | Kaiwu SDK docs | **8-bit integers, [-128, 127]** — the hardest limit here |

- `Range { lo, hi, integral }` says what magnitudes a coefficient may take and whether they must be
  whole. D-Wave's couplings are continuous over `[-1, 1]`; Hitachi's are four-bit integers over
  `-7..=7`. `J = 0.5` fits the first exactly and the second not at all, and a bit count cannot say
  that. `Fabric::scale_to_fit` returns the factor that makes a program fit, or `None` when no
  factor helps.
- `Precision::{Exact, Fixed, Float, Unstated}` replaces `Option<u32>`. Fixed point spreads one step
  across the range and loses a small coefficient; floating point keeps significant digits and does
  not. Over `[1e8, 1.0]` the two answers are `1.0` and `6e-8`.
- `Verdict` carries **every** caveat rather than the first one checked. D-Wave has two: it places by
  minor embedding *and* its precision is unpublished.
- `Fabric::unstated` names what a vendor does not publish. `None` used to mean both "no limit" and
  "not documented", so a machine of unknown size looked exactly like a simulator with no size.

### Fixed

- `feasible` now means the constraints hold, not merely that every variable decoded. A penalty makes
  a constraint expensive, not impossible; a sampler whose objective outbids it returned an answer
  that read perfectly and broke the request, reported as feasible. `violated` names what broke.
- Two variables may not share a name. An answer is keyed by name, so the second replaced the first
  and one of them vanished from the result.
- A unary factor is a **field**, as it is everywhere else in the crate. It was range-checked as a
  coupling, so fields written that way walked past a fabric that has none.
- Degree counts edges, not factor mentions. `uniform_couplings` compares values, not bit patterns,
  so `0.0` and `-0.0` are one weight rather than two.
- `check` reports the worst out-of-range coefficient, which is the one that sets the scale factor.
  It reported the first while its comment said worst.
- The node-update ceiling counts the whole run. A request declaring 1,024 node updates did 262
  million, because every handler runs `draws × thin` further sweeps afterwards.
- Every Hitachi layout failure was reported as `TooHighDegree { degree: 0, limit: 8 }`. It says what
  failed now.
- `field_bits` was declared by every fabric and read by nothing.
- The Pt V2 declared a negative coupling its cell cannot express — it counts active neighbours, so a
  coupling is present or absent.

### Verification

- `scripts/mutation-check.sh` breaks the code on purpose and requires a named test to notice. It
  refuses to run on uncommitted work, calls out a mutation that did not apply, and distinguishes a
  build failure from a genuine red — each of which had already produced a false pass here.
- `include/check.c` compiles against the header and links against the library, so the header cannot
  drift from the ABI. It found a missing declaration on its first run.
- `web-tests/` drives both browser pages through the same API an agent would use, and
  `npm run live` drives the deployed copy rather than a local build.

**403 Rust, 18 Python, 19 Zig, 123 Julia, 42 browser, 18 C.**



## 0.7.0 (2026-08-08)

**The modelling layer, on every surface.** And an ABI break to make it honest: a `value` is now
`int64_t` everywhere and means the modeller's own value, never a slot index.

### The layer

`model` states a problem in its own vocabulary — variables with domains, constraints that must
hold, an objective — and compiles to spins, answering in the names you gave.

- Variables: `categorical`, `integer`, `binary`, `spin`. Constraints: `not_equal`, `equal`, `fix`,
  `cardinality`, `at_most`, `at_least`, `exactly_one`, `at_most_one`. Objectives read like
  arithmetic: `5.0 * x.is(2) + 2.0 * y.is(1) - a.is(1) * b.is(1)`.
- Inequalities compile through a **slack variable**, because squaring "at most three" would punish
  choosing two exactly as hard as choosing four. The slack costs spins and never appears in the
  answer: a solver artefact is not a result.
- Counting constraints take **any number of literals**, each naming its own variable and its own
  value. "At most two of these nine shifts" and "at most one of a = 3, b = 17" are both sayable.
- Reachable from Rust, C, Python, Zig, Julia, wasm, the node editor, HTTP and MCP. Every one
  compiles through the same code.

### Breaking

- **Every `value` widened from `uint32_t` to `int64_t`** and carries the modeller's own value. An
  integer over `10..=20` takes 13; passing 3 is an error naming the range. It used to be a slot
  index, so `x.is(13)` rewarded **18** — and `not_equal` compared two variables slot by slot, so an
  integer over `5..=10` and one over `0..=5` were held to agree in six places when they share one.
- **`feasible` now means the constraints hold**, not merely that every variable decoded into a
  valid codeword. A penalty makes a constraint expensive, not impossible; a sampler whose objective
  outbids it returns an answer that reads perfectly and breaks the request, and that reported as
  feasible. `violated` describes each broken constraint in the caller's own names.
- `Model::objective` **accumulates** rather than replaces, and folds its sense in per term. Writing
  one term per option in a loop kept only the last; a minimising term after maximising ones
  re-interpreted all of them. `set_objective` replaces.
- `Domain::Spin` speaks in −1 and +1. It reported 0 and 1 to the literal reader while the decoder
  handed back −1 and +1, because both folded Spin into a `_ =>` catch-all.
- Two variables may not share a name. An answer is keyed by name, so the second did not shadow the
  first — it replaced it, and one of the two vanished from the result.

### Fixed, all of them silent

Each returned a plausible answer with `feasible: true` and no error.

- `"maximize": 1` **minimized**. `as_bool` returned `None` for a JSON number and `unwrap_or(false)`
  made that `Minimize` — not a degraded answer, the opposite one.
- A `"value"` the reader could not parse became 0, or 1. `"13"` as a string pinned a variable to 0.
- `at most 0 of these` compiled to **nothing**. Slack is only allocated when the range has room in
  it, and "needs no slack" was taken to mean "needs no constraint".
- The node-update ceiling counted burn-in only. A request declaring 1,024 node updates did 262
  million, because every handler runs `draws × thin` further sweeps afterwards. `verify` had no
  ceiling at all.
- The node editor discarded every refusal code, so a constraint the library rejected vanished from
  the model and the editor answered a different problem.

### Added

- `ft_model_*`: the whole modelling layer over the C ABI, declared in `include/ferrotherm.h`.
- `ft_model_fixed_penalty`, the remedy every error message recommends and no surface could perform.
- `ft_model_solve_with` and a `schedule` on HTTP/MCP: a caller who measured their instance can say
  so. A ladder that runs backwards is refused rather than substituted.
- `ft_model_violations` / `ft_model_violation`, and `violated` on every surface.
- Certification of a compiled model from Python and Julia, and the five `ft_model_cert_*`
  accessors declared at last.
- `exactly_one` / `at_most_one` on every surface. They lower pairwise with no slack, so they are
  measurably cheaper than `k = 1`.

### Verification

- `include/check.c` compiles against the header and links against the library, because nothing was
  checking that the header describes the ABI. It found a defect on its first run.
- `web-tests/`: 24 browser tests driving `window.ferrotherm`, the same surface an agent uses.
  `npm run live` drives the deployed copy rather than a local build — a stale deploy served a build
  missing a full day of exports while the page still loaded and still answered questions.
- `scripts/check-wasm-exports.sh` derives its requirement from the pages themselves, every
  `W.ft_...` in `docs/*.html`, rather than a list kept by hand beside them.
- The agent harness drives `ferrotherm_solve`, which nothing had, which is how a whole family of
  defects on it lived behind a green suite.

355 Rust, 18 Python, 17 Zig, 117 Julia, 24 browser, 18 C.

## 0.5.1 (2026-08-05)

Deployment-ladder facts finalized at datasheet grade (verified Aug 2026).

- Alchitry V2 lineup: Cu V2 $59.99 (iCE40-HX8K, the full-open-flow rung), Au V2 $149.99
  (XC7A35T-2), Pt V2 $349.99 (XC7A100T-2, 4x GTP = PCIe Gen2 x4 capable; the vendor listings'
  "FGG84I" package is a typo, confirmed FGG484 from the Rev A schematic).
- Kria KV260: XCK26 exact fabric numbers; corrected the widely repeated FALSE claim that it needs
  Vivado Enterprise (free ML Standard covers Kria, per AMD's licensing FAQ).
- Numato Aller: XC7A200T-2 in M.2 2280 — the only first-party 2280 M.2 FPGA still manufactured
  (LiteFury/NiteFury/Acorn dead); ~$500 quote-only.
- AWS f2.48xlarge, architecture-critical [AWS re:Post]: NO FPGA-to-FPGA links (no P2P, no ring —
  F1 had both). The x8 tier therefore runs replica-exchange parallel tempering (scalar energies
  per swap fit host-mediated topology) rather than DSIM-2-style lattice partitioning.

## 0.5.0 (2026-08-05)

The hardware backend and the named deployment ladder.

- `hdl`: lower any bipartite sampling graph to a fixed-point p-bit fabric (Q.8 weights, 1024-entry
  sigmoid ROM, per-node xorshift32, two-phase chromatic schedule) and emit synthesizable Verilog.
  The contract: `FixedFabric` is a CYCLE-EXACT Rust emulator of the emitted RTL — the generated
  self-checking testbench replays the emulator's per-sweep state trace and must match BIT-EXACTLY
  in icarus-verilog simulation (gated in `cargo test`; CI installs iverilog). The quantized
  fabric also re-passes the Onsager physics gate within quantization tolerance. Software model ==
  emulator == RTL, verified.
- `targets`: the named deployment ladder added — Alchitry Au/Au+, AMD Kria KV260 (K26 SOM),
  Numato Aller (XC7A200T in M.2: the compute-stick class, buyable today), and AWS f2.48xlarge
  (8x VU47P, the multi-chip tier for DSIM-2-style distributed Gibbs). 19-entry database.
- CI: icarus-verilog installed so the RTL gate runs on every push.

## 0.4.0 (2026-08-05)

The performance core, the same-machine parity measurement, and the FPGA deployment-target database.

- `gibbs::Sampler::sweep_par` / `sweeps_par`: multithreaded chromatic sweeps (scoped threads,
  race-free by coloring, per-(sweep, class, chunk) RNG streams; bit-reproducible for a fixed
  (seed, threads)). Passes the same Onsager physics gate as the sequential path.
- Same-machine, same-model parity vs THRML (JAX 0.11, Python 3.14, CPU), measured quiet:
  at 16,384 nodes ferrotherm 6.3e7 flips/s single-thread vs THRML 1.68e7 (3.7x); at ~270k nodes
  ferrotherm 3.8e8 at 18 threads vs THRML 1.05e8 (3.6x; THRML's vectorization beats our single
  thread at that size, 9.5 vs 13.6 ns). Browser WebGPU on the same machine: 9.35e9 (89x THRML-CPU;
  THRML has no GPU path on non-CUDA hardware). Scripts: `scripts/thrml_bench.py`,
  `examples/parity_bench.rs`.
- Corrected a published number: the earlier 86 ns/flip CPU figure was measured while background
  jobs shared the machine; quiet re-measurement gives 13.6 ns/flip. Recorded as a discipline rule.
- `targets`: the FPGA deployment-target database — edge parts (iCE40/ECP5/Gowin/Artix, with open-
  toolchain status), buyable cards (Alveo U55C = AWS F2 silicon twin; V80 flagship with a
  first-mover slot), cloud instances (AWS F2 active at $1.98/hr; Azure NP sunsetting May 2027;
  Alibaba/Huawei FPGA clouds verified dead), academic clusters (AMD HACC, NSF OCT: F2-class
  silicon at $0), and a $200 salvage CI tier. Capacity model anchored to the measured DSIM-2
  machine (arXiv:2606.25313: 18x VP1902, 1e12 flips/s) after the anchor test caught the first
  version ~10x optimistic. Large-part and calibration sweeps queued.

## 0.3.0 (2026-08-05)

Wave 3 of the field ingest: the flagship architecture and two more hardware-algorithm lines.

- `dtm`: Denoising Thermodynamic Models (arXiv:2510.23972) — closed-form forward jump kernels,
  pattern grids (G8/G12/G16 with degree and bipartiteness tests), contrastive chain training with
  latent marginalization, the TC penalty (closed form, h-component cancels exactly), the ACP
  controller (scripted-sequence unit test), and reverse-chain sampling. THE GOLD TEST: on a fully
  enumerable DTM the Eq.-14 gradient with exact conditional expectations matches central finite
  differences of the exact NLL for every parameter; sampled-gradient training must reduce the
  EXACT NLL. Recorded: the paper's printed Eq. D1 sign is wrong — the energy-form keep
  probability test pins the negative sign and shows the printed sign yields exactly the
  complement (indistinguishable only in the noise-saturation limit).
- `lrw`: Lattice Random Walk SDE discretisation (arXiv:2508.20883, the algorithm behind Normal
  Computing's CN101) — ternary increments with exact conditional moments (algebraic identity
  test, no Monte Carlo), validity clipping, and the stability mechanism demonstrated: a cubic
  drift that provably diverges under Euler-Maruyama from x0 = 5 stays bounded under the walk.
- `sbm`: Simulated bifurcation (ballistic and discrete variants, Goto et al.) — symplectic
  momentum-first updates, inelastic walls, best-so-far readout. Verified against exhaustively
  enumerated ground states: K8, C7, Petersen exact on both variants; 20/20 and 20/20 on seeded
  N = 16 Gaussian instances. Recorded: with x initialized exactly to zero, symmetric graphs
  synchronize, hit the walls together, and the momentum reset erases the symmetry-breaking
  (measured: dSB converged to the WORST K8 state); a small random x-init removes the trap.

## 0.2.0 (2026-08-05)

Wave 2 of the field ingest. Breaking: `Program::run` gains a trajectory-trace parameter.

- `het`: heterogeneous factor-graph Gibbs — spin and categorical nodes, energy-table factors of
  arbitrary arity (subsumes pairwise), proper-coloring block sweeps, clamping. Verified: a mixed
  spin+categorical model with a 3-ary factor matches exact Boltzmann enumeration (TV < 0.02); on
  pure-spin pairwise models the het and spin engines agree to 1e-12; clamped categorical
  conditionals match exact rows. The spin engine remains the fast path.
- `linalg`: cyclic Jacobi eigensolver for symmetric matrices (reconstruction-verified).
- `tla::solve_spd_exact_ou`: the bias-free exact Ornstein-Uhlenbeck transition integrator in the
  eigenbasis. New tests pin BOTH facts: the Euler-Maruyama chain's stationary covariance is
  biased per eigenmode by exactly 2/(2 - dt*alpha) (the test that catches silently absorbing it),
  and the exact integrator lands on beta^-1 alpha^-1 unbiased.
- `program::Gate::BoltzExact` + `Program::ebm_kernel_grad`: the third gradient estimator
  (EBM-kernel decomposition: one trajectory plus one auxiliary clamped draw, arXiv:2608.01612
  Sec III C). Cross-validated against exact-score REINFORCE and an in-test full-enumeration
  reference. Recorded along the way: finite differences with common random numbers is NOT a
  usable referee across a discrete re-draw (CRN decorrelates; noise floor exceeded the gradient),
  so the test enumerates instead.

## 0.1.0 (2026-08-05)

First public release. Pure Rust, zero dependencies, std-only, wasm-clean, deterministic by seed.

- `graph` + `gibbs`: sparse pairwise energy-based models, chromatic block-Gibbs with clamping.
  Verified: exact Boltzmann on enumerable systems; Onsager's 2D closed form to 4 decimals.
- `device`: the published degree-16 planar topology of Z1-class thermodynamic chips
  (displacement rules (1,0),(2,1),(2,3),(4,1); proven bipartite, longest edge sqrt(17)).
- `ledger`: first-class joules accounting; `Z1_SPICE` prices (pre-silicon vendor estimates,
  arXiv:2608.01615 Table IV) with the write/sample ratio (~21,700x) as a tested invariant.
- `tempering`: simulated annealing + parallel tempering with ladder diagnostics. Verified:
  finds exhaustively-enumerated ground states of frustrated glasses.
- `tla`: thermodynamic linear algebra (Ornstein-Uhlenbeck network; Aifer et al.,
  arXiv:2308.05660). Verified: SPD solves match Gaussian elimination; covariance estimates A^-1.
- `program`: stochastic differentiable circuits (flip, Gaussian-policy, linear-dynamics,
  stage-cost, Gibbs-kernel gates); REINFORCE, parameter-shift, and finite-difference gradients
  cross-validated; a trained stochastic controller reaches a provably optimal LQR gain.
- `compile`: variational compilation of conditional kernels onto device graph patches with
  hidden-unit marginalization; exact positive/negative-phase gradients (FD-verified to 1e-6);
  the chain-rule KL error bound verified exactly on compiled programs.
- `ffi`: C ABI (`ft_*`) for WebAssembly and host languages; FFI path re-verified against Onsager.
- Examples double as verification gates (exit codes); `web/gibbs_bench.html` is the WebGPU
  instrument (verifies against Onsager on the visitor's GPU before reporting throughput).
