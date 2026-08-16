# ferrotherm

Thermodynamic computing in pure Rust. Sparse energy-based models, chromatic block-Gibbs, parallel
tempering, thermodynamic linear algebra, stochastic differentiable programs, a variational
compiler onto device topologies, and a first-class joules ledger — zero dependencies, std-only,
wasm-clean, deterministic by seed, verified against exact physics before anything else.

The physics is open and old: Ising (1925), Glauber dynamics (1963), Gibbs sampling (Geman & Geman
1984), checkerboard parallel sweeps, Ornstein-Uhlenbeck relaxation. A "thermodynamic sampling
unit" accelerates exactly these loops and charges for I/O. Both the loops and the ledger belong in
the open commons, runnable on every compute fabric: CPU today, WebGPU and wasm in the browser,
physics-native silicon when there is silicon to measure.

## Use it

```sh
cargo add ferrotherm
```

```rust
use ferrotherm::{ising, gibbs::Sampler, ledger::{Ledger, Z1_SPICE}};

let g = ising::lattice2d(16, 1.0);            // a magnet below critical temperature
let mut led = Ledger::default();
let mut smp = Sampler::new(&g, 0.6, 42);
smp.sweeps(500, Some(&mut led));               // sample it, and meter it
println!("|M| = {:.3}", smp.s.iter().map(|&v| v as f64).sum::<f64>().abs() / g.n as f64);
let j = led.joules(&Z1_SPICE).expect("Z1_SPICE states its prices; Prices::UNSTATED would not");
println!("device-model cost: {j:.2e} J");     // pre-silicon vendor prices, labelled
```

`AGENTS.md` carries the invariants and task recipes for AI agents; `llms.txt` is the machine
summary. Seven of the twenty examples are verification gates that exit non-zero when their check
fails; the rest are probes that print what they measured and always exit 0.

## The crates

The core is std-only with **zero dependencies**, and stays that way. Anything needing a dependency —
a GPU driver, a TLS client, a power sensor — is a sibling crate you opt into, and deleting any of
them leaves `ferrotherm` intact.

| crate | what it adds | why it is separate |
|---|---|---|
| [`ferrotherm`](https://crates.io/crates/ferrotherm) | the physics, the compiler, the ledger, the C ABI | — |
| [`ferrotherm-gpu`](https://crates.io/crates/ferrotherm-gpu) | the same WGSL sweep the browser runs, natively | needs `wgpu` |
| [`ferrotherm-meter`](https://crates.io/crates/ferrotherm-meter) | joules **measured on the machine that ran it**, not borrowed from a vendor datasheet | needs a power sensor |
| [`ferrotherm-cloud`](https://crates.io/crates/ferrotherm-cloud) | real fabricated Ising silicon: Hitachi's CMOS annealing ASIC | needs a TLS client |
| [`ferrotherm-silicon`](https://crates.io/crates/ferrotherm-silicon) | FPGA fabrics — stochastic-neuron LUTs, chip databases, bitstream emission | needs the FPGA toolchain |
| [`ferrotherm-serve`](https://crates.io/crates/ferrotherm-serve) | an HTTP sampling API and an MCP server | it is a binary, not a library |

The two that drive *someone else's* hardware — `-cloud` and `-silicon` — reach it through the same
[`fabric::Device`] trait, which is what makes "runs on any fabric" a thing you can check rather than
a thing we say. `-gpu` is a sampler rather than a fabric and implements no `Device`.

## Field map

| Thermodynamic-computing field | ferrotherm module | status |
|---|---|---|
| THRML — block-Gibbs on sparse EBM graphs (Extropic) | `graph` + `gibbs` + `device` | **shipped, verified** |
| THRML — heterogeneous graphs (categorical nodes, arbitrary-arity factors) | `het` — mixed-kind factor-graph Gibbs | **shipped, verified** |
| Torx — stochastic differentiable programming (Extropic) | `program` — typed wires, stochastic gates, 3 gradient routes | **shipped, verified** |
| Thermalizers — variational compilation (Extropic) | `compile` — exact per-factor KL fit onto device patches | **shipped, verified** |
| p-computer optimization line (Camsari et al.) | `tempering` — annealing + parallel tempering, ladder diagnostics | **shipped, verified** |
| Thermodynamic linear algebra (Aifer et al. / Normal Computing) | `tla` — OU-network SPD solves + bias-free exact-transition integrator | **shipped, verified** |
| Torx gradient estimators (Extropic) | `program` — REINFORCE + parameter-shift + **EBM-kernel** (one trajectory + one auxiliary draw) | **shipped, verified** |
| DTM — denoising thermodynamic models (Extropic's flagship architecture) | `dtm` — forward kernels, pattern grids, contrastive chain training, ACP, TC penalty | **shipped, verified** |
| Lattice Random Walk (Normal Computing CN101 algorithm) | `lrw` — ternary-increment SDE integration, exact-moment identities | **shipped, verified** |
| Simulated bifurcation (Toshiba bSB/dSB) | `sbm` — symplectic Ising machines vs enumerated ground states | **shipped, verified** |
| Hosted simulator APIs (extropic.dev) | `web/gibbs_bench.html` + `ffi` (wasm C ABI) — on YOUR device | **shipped**; the page verifies itself against Onsager in your browser before reporting a rate |
| **Fabricated CMOS annealing silicon (Hitachi)** | `ferrotherm-cloud::hitachi` — 384×384 King's graph, four-bit coefficients, over a free public API | **shipped, conventions measured** |
| Device hardware (Z1 tapeout 2027; SPU/CN101) | `ledger::Prices` device models — priced, not owned | n/a |

Focus: **embodied and Physical AI** — sampling-based control (MPPI needs thousands of samples per
tick), implicit/energy-based policies, world-model sampling — the workload domain the entire
thermodynamic-computing corpus currently leaves empty.

## Verification (all reproducible, seeds fixed)

- `cargo test --workspace` — 506 tests across the six crates, including: exact-Boltzmann TV on an
  enumerable system, clamped-conditional exactness,
  proper coloring, degree-16 bipartite Z1 grid (longest edge √17), write/sample price ratio.
- `cargo run --release --example ring_tv` — 8-site Ising ring: TV(sampled, exact) = 0.0031 vs
  noise floor 0.0057 at 100k samples. Residual is sampling noise, not bias.
- `cargo run --release --example onsager` — 2D Ising 64×64 vs Onsager/Yang closed form:
  |M| matches to 4 decimals at β = 0.5/0.6/0.7; disordered above β_c.
- `cargo run --release --example z1_ledger` — the crossings tax, executable, at the vendor's own
  SPICE prices (arXiv:2608.01615 Table IV): the generative regime amortizes I/O; a 100 Hz control
  loop is decided by the reflash-rate cap and the unpublished price of clamping an input.
- `cargo run --release --example grad_check` — three independent gradient routes (REINFORCE,
  parameter-shift, finite-difference referee) agree on the same stochastic circuit: −0.1922 /
  −0.1922 / −0.1926 on the flip logit.
- `cargo run --release --example gibbs_grad` — REINFORCE **through the Gibbs kernel** (exact
  trajectory log-density, no approximation) matches the FD referee at three bias points; training
  the biases of a ferromagnetic ring against E[(Σs)²/n] drives 2.21 → 0.20.
- `cargo run --release --example lqr_energy` — a stochastic-program controller trained by gradient
  descent lands on the provable optimum: k = 1.996 vs exact k* = 1.997, expected-cost excess 0.00%.
  Control effort (R·E[Σu²]) is the actuation-proxy term — the E_task frame at the program level.
- `cargo run --release --example compile_chain` — the compilation error bound (arXiv:2608.01615
  Eq. 17, the chain rule of KL) verified **exactly**: readout KL 0.0054 ≤ Σε = 1.42 nats on a
  3-stage compiled program, and context-matched compilation beats uniform-input compilation on the
  inputs the program actually feeds it (ε 0.721 vs 0.750).
- `cargo run --release --example reach_on_z1` — the flagship, and **the boundary is the result**:
  a coherent quantized reach target exists (gate 90%, reached only after applying our
  capacity-vs-basis lesson — raw-angle bins gate-fail at 32%, error-vector log-bins pass), but the
  capacity ladder plateaus far below it: single patch kernel 15–30% closed-loop, per-joint
  factorization 32–35%, and trajectory-level post-training added ~3 points in an earlier run that
  this example does not re-measure. The reach law is J(q)ᵀe — products
  of state bits that sparse local pairwise energies with a few hidden spins cannot route. A control
  workload does **not yet** map onto the degree-16 fabric at patch scale; this review did not locate
  published work demonstrating
  otherwise. The ledger stands regardless: at gate quality the device's compute would sit ~7 orders
  below Jetson watts×time and E_task becomes actuation-dominated, while 9,600 clamp ops/s against
  the ≤1/s reflash cap remains the unpriced feasibility wall.

- `cargo test` also verifies: `tempering` finds the **exhaustively-enumerated ground state** of a
  random frustrated 16-spin glass (and its ladder diagnostics catch dead replica pairs); `tla`
  matches **Gaussian elimination** on SPD solves and recovers A⁻¹ from sample covariance; the
  `ffi` path re-reproduces Onsager end to end through the C ABI.
- `cargo test -p ferrotherm-gpu` — the native WGSL sampler, 6/6 on **three graphics APIs**: Apple
  M5 Max (Metal), NVIDIA L4 (Vulkan 1.4), and DX12. All three reproduce the exact mean energy from
  variable elimination — a shader can pass on Metal and fail on Vulkan, whose validation is stricter
  and whose f32 behaviour differs, so this was worth checking rather than assuming. **The DX12 run
  was WARP, a software rasteriser**: it establishes that the shader compiles under DX12 and that the
  physics is right, and says nothing about DX12 on hardware. `Gpu::is_hardware()` reported `Cpu` and
  the benchmark declined to quote a speedup on its own.
- `cargo build --release --lib --target wasm32-unknown-unknown` — compiles with **zero changes**;
  the cdylib is a **356 KB .wasm** (128 KB gzipped) exposing the `ft_*` C ABI: the run-everywhere
  claim is a build,
  not a slogan.
- `web/gibbs_bench.html` — the impedance-tax instrument. The WGSL sampler **verifies itself against
  Onsager on the visitor's GPU before reporting throughput** — and note that this page runs its
  **own** shader, a dense degree-16 lattice kernel, not the general CSR sweep that `ferrotherm-gpu`
  exposes and that the Metal/Vulkan/DX12 table above was measured on. Two shaders, two scopes: the
  page's is checked by the page, against the closed form, on whatever GPU you open it with (measured here: |M| 0.9143 vs 0.9113,
  0.9750 vs 0.9736 on Apple metal-3). Measured: **9.35e9 flips/s** at full die scale (269,568
  nodes, degree 16; 0.107 ns/flip). CPU on the same machine, measured quiet: 7.3e7 flips/s
  single-thread (13.6 ns/flip), 3.8e8 flips/s at 18 threads via `sweeps_par` (an earlier
  published 86 ns/flip figure was contaminated by concurrent background load and is corrected).
  Energy per flip at package watts / measured rate: 10 W → 1.07 nJ (151× the Z1 SPICE projection),
  25 W → 2.67 nJ (377×), 60 W → 6.4 nJ (905×). So the measured gap between a first-pass browser
  sampler on consumer silicon and the vendor's pre-silicon projection is **2–3 orders of
  magnitude**, not the marketed four — with both biases stated: package watts cover the whole
  platform; the SPICE figure excludes I/O and its own appendix revised the coarse model ~10× worse.

## Positions this crate takes

1. **The ledger is not an appendix.** Every simulation carries joules: samples, reads, writes,
   priced by a swappable `Prices` device model. Re-price the same workload on GPU-measured
   watts×time and you have the impedance-tax comparison that decides whether standalone sampling
   hardware is worth buying.
2. **Determinism.** Same seed, same draws, on every platform. Published numbers are reproducible
   or they are not published.
3. **Verify against exact physics first.** Onsager before opinions.
