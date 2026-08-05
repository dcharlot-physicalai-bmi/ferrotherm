# Changelog

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
