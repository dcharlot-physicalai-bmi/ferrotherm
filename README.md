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
a thing we say. As of 0.19.0 `-gpu` reaches it too, through `GpuDevice`: it was a sampler and not a
fabric for five releases, which meant the fastest path here was the only one `conform` could not
score. Scoring it found three defects on the first run.

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

- `cargo test --workspace` — 635 tests across the six crates, including: exact-Boltzmann TV on an
  enumerable system, clamped-conditional exactness,
  proper coloring, degree-16 bipartite Z1 grid (longest edge √17), write/sample price ratio.
- `cargo test --lib bound::` — **optimality-gap certificates**. `bound::forest` splits the energy into forests,
  minimises each exactly at induced width 1, and tightens the split by subgradient ascent —
  Lagrangian dual decomposition. `min_s E(s) >= Σ_k min_s E_k(s)` for **any** split, which is what
  makes optimising the split safe. A sampler holding a state of energy `E` is then within `E - L` of
  optimal whatever it found; at gap zero the answer is *proven* optimal without trusting the
  sampler. Soundness checked against brute force on 200 random instances, and both ways it could
  silently stop being a bound are recorded mutations. **Not a first**: D-Wave's
  `dwave-preprocessing` has shipped `roof_duality()` — a lower bound plus persistent variable
  assignments — for years, and 0.20.0 claimed this lane was empty, which was wrong. What is ours is
  a different relaxation (Lagrangian decomposition, not roof duality's max-flow), in a std-only Rust
  stack, and *anytime*: every round is a valid bound. Which is tighter on which instances is
  unmeasured; both are sound, so their maximum is too.
- **Every one of those is reachable from every surface.** `bound` had never been on the C ABI:
  optimality-gap certificates are the headline claim above, and until 0.25.0 Python, Julia, Zig, the
  HTTP server and the MCP tools could build a graph and sample it but could not ask how far from
  optimal the sample was. `scripts/check-parity.sh` exists to catch a capability that stops at Rust
  and did not catch this one — **it checks that every exported symbol reaches every binding, and a
  capability that was never exported is not a parity failure, it is a thing nobody can say.** Twelve
  C ABI symbols close it (`ft_tabu`, `ft_popanneal`, `ft_branch`, `ft_bound_*` and their
  accessors), plus `bound` and `optimize` on HTTP/MCP. Each solver leaves its best state as the
  simulation's state, so the returned number is a claim about `spins` that every binding's tests
  check, and they compose: anneal, then tabu, then branch and bound with that as its incumbent.
  `ft_bound_sdp` **re-verifies the certificate before the number crosses** — a bound crossing a
  language boundary is exactly the case where the caller cannot check it themselves. Python and
  Julia get a one-line `gap()`.
- `cargo run --release --example exact_reach` — **how far the exact solver actually goes**, which
  `exact_bracket` cannot say because its size is chosen to always prove. Measured, 40M-node budget,
  tabu incumbent, median of 3 seeds:

  | family | mean degree | cheap bound proves | with the SDP bound | nodes at the cheap ceiling |
  |---|---|---|---|---|
  | sparse | 6.0 | 76 spins | **84 spins** | 8,277,603 → 156,793 (53×) |
  | dense | 22.1 | 44 spins | **52 spins** | 12,173,789 → 192,501 (63×) |

  Density costs far more than node count: the cheap bound charges for every edge with both ends
  still free, and a sparse graph has `O(n)` of those — a few fixings retire most of them — where a
  dense one has `O(n²)` and stays loose for many levels.
- `cargo run --release --example sdp_in_tree` — **the sweep that corrected the previous line.** A
  certified SDP bound on the residual problem inside the tree is now on by default, and the first
  measurement of it said it did nothing: at depth 2 it fired ~21 times, pruned 0–4, and left the
  node count unchanged on 17 of 19 sizes. That was a property of the setting, not the method —
  depth 2 is at most seven nodes. Swept, on dense instances:

  | spins | cheap | d4 | d8 | d12 | d16 | saturates |
  |---|---|---|---|---|---|---|
  | 32 | 94,809 | 68,769 | 17,465 | 17,465 | 17,465 | d8 |
  | 36 | 242,943 | 160,381 | 13,963 | 1,731 | 1,731 | d12 |
  | 40 | 2,181,007 | 1,869,399 | 379,181 | 17,231 | **2,451** | d16 |

  It saturates because **the tree closes above that depth** once the bound is on — which means
  depth was never the real control. `sdp_min_free` and `sdp_max_free` are: too small to be worth a
  Cholesky, or too large to afford one.
- `cargo test --lib tabu:: popanneal:: branch::` — **the three solvers a max-cut result is expected
  to be measured against.** `tabu` is the mandatory baseline in the literature, with the incremental
  gain `Δ_i = 2 s_i (h_i + Σ_j J_ij s_j)` updating in `O(degree)` per flip. `popanneal` is
  population annealing: `R` chains down one ladder with resampling, which yields two things a single
  annealed chain cannot — `ln Z` from the telescoping product of resampling normalisations (absolute
  when the ladder starts at `β = 0`, where `Z = 2ⁿ` exactly), and `ρ = (Σ_f n_f²)/R` over ancestor
  families, which is exactly 1 when every ancestor still has a descendant and exactly `R` when the
  population has collapsed onto one — **a run that can say "do not trust me"**. Every exponential is
  shifted by the running maximum, because `exp(−Δβ·E)` on a G-set instance asks for `exp(600)` and
  `f64` overflows at `exp(709.78)`; the test for it asserts the ladder ran to the END, not merely
  that `ln Z` came back finite. `branch` is branch and bound, and the only thing here that returns a
  **proof**: `proved_optimal` is true only when the tree was exhausted inside the node budget, and a
  run that hit the limit says so. Nothing in it is undone by arithmetic — `x + d − d` is not `x`,
  and a bound that drifts upward prunes the subtree containing the optimum while still reporting
  success — so scalars are restored by returning from the frame and touched entries are written back
  verbatim.
- `cargo run --release --example gset_gap -- <G-set file> [best-known]` — **the standard max-cut
  benchmark, reported as a gap rather than a league-table entry.** G-set has been the comparison
  set for twenty-five years and every published figure is a *best cut found* — a lower bound, which
  ranks how hard people looked. `bound` supplies the other side, so the true optimum is bracketed:

  | instance | mean degree | cut found | best known | | forest | odd-cycle | sdp | gap |
  |---|---|---|---|---|---|---|---|---|
  | G11 | 4.0 | 564 | 564 | **100.00%** | 817 | **579** | 629 | **2.6%** |
  | G14 | 11.7 | 3058 | 3064 | 99.80% | 4694 | 3602 | **3192** | **4.2%** |
  | G1 | 47.9 | 11624 | 11624 | **100.00%** | 19176 | 14958 | **12083** | **3.8%** |

  800 nodes, 8 restarts. Bold is the bound that won; all three are sound, so the harness takes the
  maximum. G11's optimum is provably in **[564, 579]**. **`bound::forest` contributes nothing here
  and the module says so**: a tree is never frustrated and G-set carries no fields, so it
  degenerates to the trivial `-Σ|w|` on every instance — measured, `decoupled -1600 / forest -1600`
  on G11. `bound::odd_cycle` charges `2·min|J|` per edge-disjoint frustrated cycle, which is the
  only thing that makes max-cut hard, and takes G11's bound from 817 to 579. `sdp` exhibits a
  **dual point** and proves it positive definite by a completed Cholesky (Rump 2006), so weak
  duality alone makes it a bound — no optimality, convergence or rank assumption anywhere — and it
  wins by more the denser the instance is, where decomposition bounds suffer most.
- `cargo run --release --example exact_bracket` — **a gate: every bound checked against a PROVED
  optimum on every push.** `branch` returns the true minimum with a proof at 22 spins, 256× past
  what a unit test can enumerate, so `decoupled`, `odd_cycle` and `sdp` are held against ground
  truth on six independent instances rather than against a published cut that is itself only a
  lower bound. The check is one-sided: a bound may be loose by any amount and may never exceed the
  optimum. It found a real defect on its first run — the `sdp` column came back *identical to
  `decoupled`* on all six, because `lanczos_min` had been folding `min` over `jacobi_eig`'s
  eigenVECTOR matrix instead of reading the eigenvalues off the diagonal. Every certificate still
  verified, because the Cholesky is what makes the bound sound; the bound was simply loose on every
  instance. Fixing it moved G1 from 12223 to 12083 and closed a mean 88% of the gap at 22 spins.
- `cargo run --release --example ring_tv` — 8-site Ising ring: TV(sampled, exact) = 0.0031 vs
  noise floor 0.0057 at 100k samples. Residual is sampling noise, not bias.
- `cargo run --release --example onsager` — 2D Ising 64×64 vs Onsager/Yang closed form:
  |M| matches to 4 decimals at β = 0.5/0.6/0.7; disordered above β_c.
- `cargo run --release --example z1_ledger` — the crossings tax, executable, at the vendor's own
  SPICE prices (arXiv:2608.01615 Table IV): the generative regime amortizes I/O; a 100 Hz control
  loop is decided by the reflash-rate cap and the unpublished price of clamping an input.
- `cargo run --release -p ferrotherm-gpu --example duty_cycle` — **the bill for being switched
  on**, and the only place this stack prices the wait rather than subtracting it. Every energy
  comparison in this field, this project's own included, divides joules *above idle* by work done.
  That prices a machine kept busy, and the case a sampling substrate is supposed to win is the
  opposite: intermittent, low-duty work where the machine spends most of its life waiting.

  **Measured** on an idle i9-13900H (RAPL, package scope), 1024×1024, 200 sweeps, one task:

  | cadence | duty | above idle | true total | understated |
  |---|---|---|---|---|
  | continuous | 100% | 41.4 J | 43.7 J | 1× |
  | once a minute | 0.86% | 41.4 J | 309.0 J | **7×** |
  | once an hour | 0.014% | 41.4 J | 16,095 J | **389×** |

  Idle 4.5 W against 80.5 W marginal, so idle is most of the bill below a **5.5%** duty cycle.
  Inverted, that gives the number a challenger must beat — the **standby budget**,
  `idle + marginal × duty`, which grants the challenger perfectly free computation and so cannot be
  argued down by a better sampler. It settles at **4.47 W**, the idle draw, with nothing about
  sampling left in it. `ledger::Prices` carries no standby term because **no thermodynamic vendor
  publishes one**; `DeviceRun::with_standby_at_most` therefore substitutes a published ACTIVE figure,
  which bounds standby from above since CMOS active is leakage plus switching. Extropic's Z1 spec of
  `<1 W` sampling clears the 4.47 W budget — a real but **~4.5×** margin, not the 20× an assumed
  20 W incumbent suggests.

  Two scope facts decide how to read it. RAPL package scope **omits** RAM, storage, fans and supply
  losses, so it understates the incumbent's idle — the term the argument leans on — making the
  conclusion conservative. And the GPU arm **refused to report**: the RTX 4050 is discrete, RAPL
  reads the CPU package, and the card's draw is outside the counter. It first reported 5.5 W
  marginal, which was the cost of *feeding* the card. `Meter::scope()` and `Scope::covers()` now
  refuse rather than divide.
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
  the benchmark declined to quote a speedup on its own. **Four backends now, and CI executes one**:
  Apple Metal, NVIDIA Vulkan, Intel Iris Xe Vulkan, and lavapipe (software Vulkan) — 12/12 on each.
  CI used to run this crate on a runner with no adapter, where every hardware-gated test skips, so
  the fastest sampler in the stack had *zero* CI coverage and its correctness rested on whichever
  machine somebody remembered to test by hand. It now installs lavapipe and runs the real shader,
  and **a skip there is a failure** — a driver was installed on purpose, so "no GPU adapter" means
  it did not load and the shader went unverified while the job stayed green.
  **A second vendor found what one could not**:
  on an RTX 4050 the default `cargo test -p ferrotherm-gpu` SIGSEGVs — parallel Vulkan device
  creation crashes that driver stack, where single-threaded it passes 12/12. The shader was never
  implicated; adapter acquisition is now serialised behind the same lock the meter uses, and the
  suite passes under default parallelism there. Since 0.19.0 `GpuDevice` implements
  `Device`, so **`conform` scores the GPU path** — for five releases the fastest sampler here was
  the one path the conformance suite could not reach, runnable but uncheckable against the fabric
  it claims to be. Pointing `conform::run` at it found three defects that being unscoreable had
  hidden: it returned the schedule's last state where every other implementation returns the best
  seen (−57 against variable elimination's exact −59, on a ladder the CPU solves); `Gpu::sweep` had
  no seed, so a `Device` honouring the trait signature would have accepted one and dropped it —
  which **no determinism check can catch, because an ignored seed is perfectly reproducible**; and a
  run inherited the previous run's state instead of starting from a seed-drawn configuration, so a
  second run began at the first's answer and handed it back. The fabric now also declares
  `Precision::Float { mantissa: 24 }`: the shader's buffers are f32 while the CPU path is f64, and
  an undeclared difference is one nothing downstream can reason about.
- `cargo build --release --lib --target wasm32-unknown-unknown` — compiles with **zero changes**;
  the cdylib is a **437 KB .wasm** (156 KB gzipped) exposing the `ft_*` C ABI: the run-everywhere
  claim is a build,
  not a slogan.
- `web/gibbs_bench.html` — the impedance-tax instrument. The WGSL sampler **verifies itself against
  Onsager on the visitor's GPU before reporting throughput** — and note that this page runs its
  **own** shader, a dense degree-16 lattice kernel, not the general CSR sweep that `ferrotherm-gpu`
  exposes and that the Metal/Vulkan/DX12 table above was measured on. Two shaders, two scopes: the
  page's is checked by the page, against the closed form, on whatever GPU you open it with (measured here: |M| 0.9143 vs 0.9113,
  0.9750 vs 0.9736 on Apple metal-3). Measured: **9.35e9 flips/s** at full die scale (269,568
  nodes, degree 16; 0.107 ns/flip). CPU on the same machine, measured quiet: 7.3e7 flips/s
  single-thread (13.6 ns/flip), and **3.8e8 flips/s at 18 threads via `sweeps_par` — at a lattice
  size that figure never stated, which is a defect in the figure**: `sweeps_par` spawns its threads
  *inside* each sweep (`gibbs.rs`), so parallel efficiency is set by how much work one sweep carries
  and the same call reports different speedups at different problem sizes. A multithreaded
  throughput number without its problem size is not reproducible; re-measuring it is pending a quiet
  machine. (An earlier published 86 ns/flip figure was contaminated by concurrent background load
  and is corrected — the same failure the load guard now refuses outright.) **`host` is that guard,
  and it is now the whole class rather than one file.** The energy side has refused an idle baseline
  above a load average of 2 since 0.17.0; the timing side had nothing, and `gset_gap` reported
  85.7 s for a G1 search that takes about 14 s on a quiet machine, in the same format as every
  honest timing beside it. The distinction the module is built on is that a **result** — a cut, a
  bound, an energy — is the same number whoever else is on the CPU, while a **rate** — flips/s,
  ns/flip, J/flip, a speedup column, a head-to-head — is a division by wall-clock time and measures
  the run queue. So `gset_gap` annotates and `flips_bench` / `parity_bench` exit non-zero.
  `Timing::as_measurement()` returns `Option<f64>`, because the defect was never a missing check —
  it was a check whose result nothing was obliged to consult.
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
2. **Idle is part of the bill.** Every energy comparison in this field, this stack's own included
   until now, divides joules *above idle* by work done — which is the right question only for a
   machine kept busy, and most places a sampling substrate would go do not keep one busy. So `duty`
   prices the wait, and reports both halves.
3. **A busy machine has no idle.** `Meter::idle` reads the load average and refuses to call a
   baseline idle above 2 runnable threads. This is not hypothetical hygiene: one published figure in
   this README was already corrected for exactly this contamination, and the first run of
   `duty_cycle` was refused by the new guard on a machine at load 24. The bias runs one way — other
   people's work *inflates* a baseline — so the guard protects against overstatement, which is the
   direction that would have flattered this project's own argument.
4. **Determinism.** Same seed, same draws, on every platform. Published numbers are reproducible
   or they are not published.
5. **Verify against exact physics first.** Onsager before opinions.
