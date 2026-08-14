# Changelog

## Unreleased

- `model`: an objective term of three or more literals compiles, via `reduce`. `Expr::product`
  writes one; `Compiled::ancillas` reports what it cost. Whatever expands to degree 0, 1 or 2 goes
  straight into the graph, so only what is genuinely wider is charged.
- `reduce`: higher-order models run on pairwise hardware. Verified by enumerating every state of
  both models rather than by sampling.
- `fabric`: six machines declared from vendor documentation — D-Wave Advantage and Advantage2,
  Fujitsu DA3, Toshiba SQBM+ (QUBO and PUBO), QBoson CPQC. `Precision` distinguishes fixed point
  from floating point from **unstated**; `Verdict` carries every caveat rather than the first one
  checked; `Range` says what magnitudes a coefficient may take and whether they must be whole.
- `scripts/mutation-check.sh`: break the code on purpose, require a named test to notice.

**This is a breaking API change and is not released yet**, deliberately — see PACKAGING.md. The
fabric API is still moving and two breaking releases a week apart help nobody.

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
