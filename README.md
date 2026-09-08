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
use ferrotherm::{ising, gibbs::Sampler, samples::Plan, ledger::{Ledger, Z1_SPICE}};

let g = ising::lattice2d(16, 1.0);            // a magnet below critical temperature
let mut led = Ledger::default();
let mut smp = Sampler::new(&g, 0.6, 42);

// Burn in 500 sweeps, then keep 2,000 states two sweeps apart -- charging the device for the
// sweeps AND the readback, which is the larger half of the bill on hardware of this class.
let set = smp.collect(&Plan::new(500, 2_000, 2), Some(&mut led));

let m = set.magnetization().expect("a chain is distributional; a search would refuse here");
println!("|M| = {m}");                        // value, error bar, effective sample size, tau
println!("{}", set.certificate(&g).unwrap()); // and at what temperature it REALLY sampled

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
| 2D adaptive parallel tempering over (β, W₀) — *one MATLAB file, June 2025* | `adaptive` — respacing to equal acceptance, plus a (β, coupling-scale) grid; replica count derived from the model | **shipped, verified** — the default ladder was severed on a glass and the payoff is now measured |
| Thermodynamic linear algebra (Aifer et al. / Normal Computing) | `tla` — OU-network SPD solves + bias-free exact-transition integrator | **shipped, verified** |
| Torx gradient estimators (Extropic) | `program` — REINFORCE + parameter-shift + **EBM-kernel** (one trajectory + one auxiliary draw) | **shipped, verified** |
| DTM — denoising thermodynamic models (Extropic's flagship architecture) | `dtm` — forward kernels, pattern grids, contrastive chain training, ACP, TC penalty | **shipped, verified** |
| **Fitting an EBM to data (the training half every EBM stack needs)** | `ebm` — contrastive divergence + **exact** likelihood by enumeration | **shipped, verified** — the fixed point is moment matching, checked against enumeration rather than against more sampling |
| Lattice Random Walk (Normal Computing CN101 algorithm) | `lrw` — ternary-increment SDE integration, exact-moment identities | **shipped, verified** |
| Simulated bifurcation (Toshiba bSB/dSB) | `sbm` — symplectic Ising machines vs enumerated ground states | **shipped, verified** |
| Hosted simulator APIs (extropic.dev) | `web/gibbs_bench.html` + `ffi` (wasm C ABI) — on YOUR device | **shipped**; the page verifies itself against Onsager in your browser before reporting a rate |
| **Fabricated CMOS annealing silicon (Hitachi)** | `ferrotherm-cloud::hitachi` — 384×384 King's graph, four-bit coefficients, over a free public API | **shipped, conventions measured** |
| Optimality **gap** in the modeller's units (D-Wave, Amplify, Jij: not found) | `Solution::gap` + `branch::Outcome::bound` | **shipped, verified** — invariant under the penalty, checked against enumeration |
| Penalty sufficiency **proved**, not scaled (D-Wave `penaltymodel` is per-constraint) | `Model::certified_penalty` | **shipped, verified** — and refuses where no penalty suffices |
| Tensor networks (quimb, cotengra, ITensor, GenericTensorNetworks.jl) | `tensor` — general-rank contraction, any index dimension, order priced before it runs | **shipped, verified** — contraction agrees with variable elimination on `log Z` and marginals |
| Propagation presolve + conflict set | `model::presolve` — sound fixpoint over the constraints, with a witness that contradicts itself in isolation | **shipped, verified** — soundness by enumeration, minimality under propagation |
| Solver portfolio under one budget (what commercial solvers ship) | `portfolio` — a `Search` trait, a budget in spin proposals, four arms | **shipped, verified** — every arm's budget conversion is measured, not asserted |
| Slicing / cutset conditioning (the standard answer to bounded treewidth) | `exact::log_partition_sliced` + `ground_state_sliced` — pin variables, solve the pieces | **shipped, verified** — sliced equals direct where both run, and beats brute force where direct refuses |
| Vector-state LQR oracle (matrix DARE) | `mppi::MatSystem` + `MatLqr` — the exact optimum for a robot-shaped system, not a one-dimensional one | **shipped, verified** — residual, cost-to-go, and no perturbation beats it |
| Exact 2D Ising energy density (Onsager 1944, via AGM elliptic `K`) | `free_energy::onsager_energy_density` — closed form, machine precision, no grid | **shipped, verified** — `U/N = −J√2` exactly at criticality |
| DIMACS CNF / WCNF — the MAX-SAT benchmark corpus | `dimacs` — clauses to a `Hubo` by exact subset expansion, no penalty | **shipped, verified** — the energy IS the unsatisfied weight at every assignment |
| Pseudolikelihood training (Besag 1975; the standard sampling-free fit) | `ebm::train_pseudolikelihood` — closed-form objective and gradient, no sampler | **shipped, verified** — gradient checked against finite differences, consistency measured |
| Penalty-free higher-order reduction (Freedman–Drineas; Ishikawa 2011) | `reduce::to_pairwise_exact` — exact minima, no penalty coefficient anywhere | **shipped, verified** — the identities checked exhaustively at every arity to eight, both signs |
| Density of states / flat-histogram sampling (Wang–Landau; Belardinelli–Pereyra 1/t) | `wanglandau` — one bin per energy LEVEL, `1/t` schedule, exact-enumeration oracle | **shipped, verified** — one run reproduces the whole `log Z(beta)` curve against exact elimination |
| Multi-spin coding (Isakov et al.; the Janus line) | `multispin` — 64 replicas per `u64`, bit-sliced ripple-carry field, refuses non-uniform `|J|` by name | **shipped, verified** — every lane certified, and lane INDEPENDENCE tested separately |
| Cluster updates (OpenJij `Algorithm_SwendsenWang_run`) | `cluster` — Swendsen–Wang + Wolff, validity decided by **signed-graph balance**, fields absorbed by a ghost spin | **shipped, verified** — `z` measured against the literature, and a refusal carries a frustrated cycle the caller can multiply out |
| Exact ground-state **counting** (GenericTensorNetworks.jl) | `exact::ground_degeneracy` — the cold limit of `log Z` | **shipped, verified** against a closed form at 101 spins |
| Device hardware (Z1 tapeout 2027; SPU/CN101) | `ledger::Prices` device models — priced, not owned | n/a |

Focus: **embodied and Physical AI** — sampling-based control (MPPI needs thousands of samples per
tick), implicit/energy-based policies, world-model sampling — the workload domain the entire
thermodynamic-computing corpus currently leaves empty.

## Verification (all reproducible, seeds fixed)

### Machine-checked theorems

Where a statement is load-bearing and its domain is finite, it is **proved**, not tested: four Kani
harnesses (bounded model checking, exhaustive over the stated ranges) verify that `copies_for` is
sufficient *and minimal* for every degree and budget in range, and that the Pegasus and Zephyr
linear indices are injective and in range at the shipped machine sizes — injectivity being the
difference between programming a qubit and programming *some* qubit. The harnesses live beside the
code under `cfg(kani)` (no dependency added); `scripts/check-proofs.sh` runs them, and its selftest
feeds Kani a false theorem and requires the refutation.

- `cargo test --workspace` — 699 tests across the six crates, including: exact-Boltzmann TV on an
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
- `cargo run --release --example planar_exact` — **exact max-cut at 10,000 spins.** Everything else
  here searches. This does not: max-cut is NP-hard *in general* and polynomial *on a planar graph*,
  and the difference is a theorem rather than an engineering margin. A cut in the graph is a cycle
  in the dual, so the problem becomes a minimum-weight `T`-join and then a minimum-weight perfect
  matching — Edmonds' blossom, in `matching`, with a Demoucron embedding in `planar`. Measured on
  planar spin glasses with couplings uniform in `{−1, +1}`:

  | grid | spins | odd dual faces | **exact cut** | breakout local search | BLS short by |
  |---|---|---|---|---|---|
  | 10×10 | 100 | 42 | **75** | 74 | 1.33% |
  | 20×20 | 400 | 180 | **270** | 268 | 0.74% |
  | 40×40 | 1,600 | 742 | **1,115** | 1,089 | 2.33% |
  | 100×100 | 10,000 | 4,848 | **7,040** | 6,864 | 2.50% |

  For scale: branch and bound with a certified SDP bound *proves* 76 spins. This proves 10,000,
  because the structure is there — and that clause is the whole result. Mandrà, Katzgraber and Thomas
  showed in 2017 that quantum-annealer speedup claims on planar gadget problems were measured on
  instances **minimum-weight perfect matching solves exactly in polynomial time**, which is to say on
  instances that are easy. This module is that observation implemented, so read the table the same
  way: breakout local search falling 2.5% short is not evidence that our search is behind the field,
  it is evidence that a heuristic which does not know the graph is planar cannot use the one fact
  that makes it tractable. A planar result is a statement about structure, never a benchmark of
  solvers. The whole pipeline — blossom, embedding, dual, `T`-join,
  two-colouring — is five pieces none of which raises anything when subtly wrong, so it is checked
  against `branch::solve` on small instances (a completely different argument, enumeration in the
  spin domain), and it **checks itself twice** on every run: the recovered edge set must two-colour,
  and two disjoint computations of the cut must agree. It refuses rather than reports — on fields,
  on non-planarity, on a cut vertex, on weights that do not scale to integers — and says which,
  because those are four different things to do next. A periodic lattice is a torus and is refused.
- `cargo run --release --example toroidal_bound -- G11.txt 564` — **G11's best-known cut is
  optimal, and this proves it.** G-set's toroidal instances are the case the exact planar solver
  refuses: a torus is not a plane. But the dual argument needs only *faces*, and an embedding on any
  surface has them — so the same reduction runs on a toroidal embedding, where the cycle space of
  the dual is four times the cut space and its optimum is therefore an **upper bound**. That is the
  side of the table nobody publishes: every G-set figure is a best cut *found*, a lower bound.

  | instance | torus | odd dual faces | best known (lower) | **upper bound** | verdict |
  |---|---|---|---|---|---|
  | G11 | 8×100 | 434 | 564 | **564** | **the bracket closes: 564 is OPTIMAL** |
  | G12 | 16×50 | 394 | 556 | **558** | optimum in [556, 558] |
  | G13 | 32×25 | 384 | 582 | **583** | optimum in [582, 583] |

  The grid dimensions are **recovered from the edge list**, not assumed — a match on all 1,600 edges
  is a proof of structure. `bound_on_surface` also reports whether the bound is *attained* (its
  optimum is itself a cut, so it is the maximum by construction rather than a bound); on the sphere
  that always holds, and asserting it is how the planar path knows the reduction is right.

  **Exact genus-1 max-cut is not implemented and is not claimed.** Barahona's algorithm needs
  modular arithmetic over a nested-dissection solve; the 2026 toroidal survey offers a heuristic and
  this same relaxation as the bound. What is here is the bound, and the honest verdict beside it.
- `cargo run --release --example maxcut_shootout -- G1.txt 11624` — **the head-to-head this crate did
  not have.** Three solvers on one instance at the same number of spin flips, 8 seeds each:

  | instance | degree | parallel tempering | tabu search | **breakout local search** | best known |
  |---|---|---|---|---|---|
  | G11 | 4.0 | 556 | 560 | **562** | 564 |
  | G14 | 11.7 | 3045 | **3057** | 3054 | 3064 |
  | G1 | 47.9 | 11612 | 11622 | **11624** | 11624 |

  BLS matches the world best-known cut on G1 and wins two of three; that is the result the
  literature predicts, and it is the first time this crate has been able to check it. **The budget
  is flips, not seconds** — a wall-clock comparison needs a quiet machine, and the asymmetry it
  hides is stated in the example: tempering pays `O(degree)` to make a flip where tabu and BLS pay
  `O(n)` to choose one.
- `cargo test --lib icm:: sqa:: hubo:: sdp::` — **the four gaps the toolchain survey named, closed.**
  `icm` is parallel tempering with **isoenergetic cluster moves**, the baseline the Ising-machine
  literature measures against. The move flips a whole connected component of the disagreement
  between two replicas at once and is *always accepted*, because the pair's energy is preserved
  exactly: a boundary edge joins a site where the replicas disagree to one where they agree, so its
  contribution `−J(a_i a_j + b_i b_j)` is zero before and after. That equality is asserted to `1e-9`
  on every move rather than argued. It holds only at `h = 0`, so a graph with fields is refused with
  the reason. Measured against the identical ladder with the move switched off, on periodic 2D
  glasses — and the advantage **grows with size**, which is the literature's actual claim:

  | lattice | spins | ICM wins | loses | mean ΔE |
  |---|---|---|---|---|
  | 8×8 | 64 | 0 | 0 | 0.00 |
  | 16×16 | 256 | 9 | 0 | −1.80 |
  | 24×24 | 576 | **19** | 0 | **−8.00** |

  At 8×8 both arms tie on all twenty instances — a 64-spin glass is solved by either, so the unit
  test runs at 16 and `examples/icm_scaling` measures where the separation opens. `sqa` is simulated
  quantum annealing by Suzuki–Trotter: `M` classical slices coupled at
  `J⊥ = −(1/2β)·ln tanh(βΓ/M)`, with `Γ` annealed down but **never to zero**, where `J⊥` diverges.
  One slice drops the coupling and *is* classical annealing — the honest control, compared at
  matched work rather than matched steps. `sdp::goemans_williamson` rounds the relaxation from the
  primal side: **the only worst-case guarantee in max-cut**, and `guaranteed` is false on most
  instances people care about, because 0.87856 needs non-negative edge weights. Checked against
  proved optima from `branch` on 24 instances where it does apply. `hubo` solves higher-order models
  **without quadratising**: `ΔE_i = 2·Σ_{T∋i} w_T·Π s_j` costs `O(terms containing i)`, so a `k`-body
  model is no harder to sample — only harder to put on pairwise hardware, and those are different
  problems. Verified against exhaustive enumeration over `2¹⁴`, with the ancillas it avoided
  reported as a number.
- `cargo test --lib tabu:: bls:: popanneal:: branch::` — **the four solvers a max-cut result is expected
  to be measured against.** `tabu` is the mandatory baseline in the literature, with the incremental
  gain `Δ_i = 2 s_i (h_i + Σ_j J_ij s_j)` updating in `O(degree)` per flip. `bls` is breakout local
  search (Benlic & Hao 2013), which improved the best-known cut on 33 of 71 G-set instances and is
  the record holder on most of them: descent with **no tabu list at all** — the paper argues
  diversification during descent is the mistake — and an adaptive perturbation between local optima.
  The jump `L` grows only when a descent lands on *the same* optimum as last time, and the mix of
  directed and random perturbations follows `P = max(e^(−ω/T), P0)` in the count of consecutive
  non-improving descents. Its published pseudo-code is genuinely ambiguous about whether an
  *improving* descent is also followed by a random perturbation — `ω ← 0` means both "just improved"
  and "just stagnated" by the time the perturbation procedure sees it — so both readings are a
  parameter and a test asserts they are different searches. `popanneal` is
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
- `cargo run --release --example z1t_ledger` — **the field's newest efficiency headline, taken
  apart with its own arithmetic.** Extropic's Z1T (2026-09-04) reports 294.52 nJ/token — 8.74 on Z1,
  285.78 on the FPGA carrying the rest — against an H100 measured at 40.9 µJ/token, and that
  reproduces here as 138.9x. What the headline does not contain: setting the *entire thermodynamic
  contribution to exactly zero joules* moves it to 143.1x, so the sampler is worth 3.1% of the claim
  and the stated path to 1000x needs the FPGA half 8.9x better. The `hybrid::Split` bound is general
  — Amdahl's law in joules — and applies to every hybrid this field has benchmarked.
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
**wasm SIMD does not help this sampler, and it was measured before it was believed.** Building with
`-C target-feature=+simd128` produced 110.4 M flips/s against the baseline's 110.3 on a 128×128
lattice — indistinguishable, on a machine noisy enough that a single baseline run dipped to 65.8 —
and cost 5 KB. The mechanism is plain in hindsight: chromatic block-Gibbs is a scatter/gather over a
CSR neighbour list with one RNG draw and one transcendental per spin, so there is no wide arithmetic
for an autovectoriser to find. The flag is not enabled. Energy was bit-identical either way, which
is the check that says the comparison was of the same computation.

- `RUSTFLAGS='-C strip=symbols' cargo build --release --lib --target wasm32-unknown-unknown` —
  compiles with **zero changes**; the cdylib is a **740 KB .wasm** (268 KB gzipped) exposing the
  `ft_*` C ABI: the run-everywhere
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

### The error bar, checked against the exact answer

A standard error is a claim, and this is the measurement of it. `examples/interval_calibration`
runs 24 chains of 20,000 draws on three models at four temperatures, takes `⟨s_i⟩` at every site,
and asks how often the interval contains the *exactly enumerated* marginal. Two intervals, built
from the same estimate, differing only by `sqrt(2τ)`:

| model | β | τ_int | corrected `sqrt(var/ess)` | naive `sqrt(var/N)` |
|---|---|---|---|---|
| ring12 | 0.5 | 2.1 | 99.7% | 83.3% |
| ring12 | 0.8 | 6.5 | 100.0% | 67.0% |
| ring12 | 1.2 | 31.6 | 100.0% | **24.0%** |
| glass14 | 0.8 | 68.1 | 94.6% | 27.7% |
| glass16 | 0.8 | 23.6 | 97.9% | 30.7% |

An interval announcing 95% and containing the truth for one site in four is not conservative; it is
a wrong number with a decoration. The corrected one over-covers on several rows and that is the
direction chosen — each estimate is deflated by the *slowest* autocorrelation the chain showed, not
the site's own, because a site sitting in a metastable mode reports a fast-looking trace while the
mode that decides the answer never moves.

**And the limit is printed with the result.** Where τ runs to hundreds, τ is itself an estimate from
a chain barely long enough to make it. On glass16 at β = 1.2, 11 of 24 seeds clear `certify`'s
`Undermixed` finding and coverage among exactly those seeds is 80.7%. The correction is a large
improvement and not a guarantee; past that point the answer is a longer chain, not a wider bar.

### Readback is 78–98% of what an independent draw costs

The mixing-expressivity table below prices one independent draw. It used to price only the sweeps
between draws, because the collection loop appended the sampler's state directly instead of reading
it — five places in this repository did the same, so the readback column was zero everywhere and
nothing could show it was missing. `Sampler::collect` reads.

```text
 layers  width   edges       tau_int   updates/draw   nJ mixing nJ readback  read share
      2     72    5184   26.30+-1.18           3787      0.0268      0.2436       90.1%
      3     48    4608    4.91+-0.31            707      0.0050      0.2436       98.0%
     12     12    1584  65.95+-20.75           9497      0.0673      0.2436       78.3%
```

A Z1-class read is 1.692 pJ per node against 7.09 fJ per Gibbs cycle: **one read is worth 239
updates.** The mixing column spans 13× across these shapes and the total spans 1.25×, because
readback depends on the spin count and these shapes hold it fixed. The tradeoff the field argues
about is real, is measured below, and is the minority of the bill at these sizes — which is what it
means to say a machine of this class is an I/O machine.

### The mixing-expressivity tradeoff, measured on both halves

The field states one sentence as its central open problem — *"scaling the number of latent variables
only improves performance if the connectivity of the graph is also scaled; otherwise... increasing
latent variables increases the depth of the Boltzmann machine, making sampling more difficult."*
This review did not locate an independent, cross-topology measurement of it. There are now two.

`examples/mixing_expressivity` is the **structural** half: shapes of a fixed spin count, random
couplings, τ_int by Sokal windowing rather than an exponential fit. The claim **holds weakly coupled
and goes U-shaped strongly coupled** — at β = 2 the shallowest shape is slow (26.30), the middle
shapes are fast (~5), the deepest slower still (65.95). And past β = 2 the estimator stops being
one: the same shape returns 285.6, 18.7, 42.6, and at β = 8 returns *small* numbers from a chain
that has stopped moving. Ruggedness needs cold; cold is where the measurement dissolves. Every row
carries `draws/τ` and prints `unusable` below 200×.

`examples/trained_tradeoff` is the **fitted** half, and it splits the sentence in two. Same latent
count wired one, two or three layers deep; both axes exact.

- **Latents without connectivity buy less expressivity — confirmed**, monotone in depth at every
  latent count.
- **They therefore cost more mixing — not as stated.** The deep arms mix *faster*; at six latents
  the *wide* model is the slowest thing in the table.

Spearman of τ_int against **what the model learned: ρ = +0.81**; against **how deep it is: −0.17**.
τ_int = 0.5 is the floor — independent draws — and the deep arms sit on it. **They are fast because
they failed.** Depth does not make sampling harder; depth makes *learning* harder, and what a model
learned is what makes sampling harder.

### Structured cliques, written down instead of searched

For a clique on a structured fabric the frontier is a construction, not a search — D-Wave's tooling
places its cliques by writing the answer down. This crate does the same on all three fabrics, and on
Zephyr it reaches the frontier exactly: `embed::zephyr_clique` places **K_{16m−8} at uniform chain
m+1 — K_232 on Z₁₅ — the same size and chain length D-Wave's busclique reaches on a perfect
fabric**. **and on Pegasus it now does the same**: `embed::pegasus_clique_fragment` places
**K_{12(m−1)} = K_180 on the Advantage's P₁₆ at chain 17**, which is busclique's size and busclique's
chain length, at every size from P₃ to P₁₆ — against this crate's own heuristic search at K_80.
`embed::pegasus_clique` keeps the closed form (K_172) with a machine-checked ceiling, as the
fallback. `embed::chimera_clique` is the classic `K_{t·m}`. Each is verified at every size by
`Embedding::verify` against the shipped fabric, the interval and quantifier arithmetic is
machine-checked by Kani exhaustively, and every Pegasus size is checked against
`minorminer.busclique` run as an oracle on the same graph. **No structured clique gap remains on
either fabric.** `ft_clique_embed` carries the constructions to Python, Zig and Julia;
`examples/embedding_tax` shows them beside the search and the frontier.

### Where this stands against the field

`LANDSCAPE.md` compares ferrotherm to every other thermodynamic stack on the axis that computation
can settle: **does it reproduce what is independently known, and how much of the field does it
cover.** Eighteen external sources of truth reproduced in CI — Onsager, transfer matrix, exact
elimination, planar max-cut, Gardner, AGS, Curie–Weiss, Bethe-on-trees, `busclique` — and 117 of the
911 tests compare against an exact answer rather than against the code's own behaviour. Searching
the four largest competitors' repositories for `onsager` or `gardner` returns zero hits in every one;
their suites test their API surface. That is the difference between a sampler and a reference.

### Free energy, certified

`ln Z` is the number every thermodynamic-computing paper quotes and no sampling stack certifies.
`free_energy` computes it three ways with exactly the guarantee each carries: **annealed importance
sampling** gives `ln Z ≥ ln Ẑ − ln(1/δ)` with probability `1 − δ` and *no equilibrium assumption*
(Markov on an unbiased estimator), reverse AIS the mirror upper bound, and **thermodynamic
integration** a bracket from `d⟨E⟩/dβ ≤ 0` that is ~9× tighter at the price of assuming each rung
equilibrated; **Bennett's acceptance ratio** steps the precise estimate up the ladder from the
exact anchor, giving `ln Z`, entropy and heat capacity at every rung from the same chains. All three are checked against enumeration, exact elimination, the transfer matrix
and Onsager before they are trusted; outward rounding of every published bound is a Kani theorem.
`ebm::log_likelihood_ais` turns it into a likelihood for models past enumeration, with an
unconditional upper bound. On every surface as `ft_ln_z_*`. And the bars themselves are checked:
`calibration::calibrate` forms `z = (estimate − truth)/stderr` against exact answers over many
seeds, so "the error bar is honest" is a test result rather than an assumption — it found one bar
30% too small, and established that the sampled estimators are never optimistic. And it comes free with an optimisation
run: `tempering::parallel_tempering_observed` records the ladder it was already sampling, so one
parallel-tempering run returns the best state *and* the whole free-energy curve — the recording
proved bit-identical to the unobserved loop.

### Learning theory as oracles

`meanfield` gives the fast approximations with their standing stated: the Gibbs–Bogoliubov bound
(a theorem — a deterministic lower bound on `ln Z` at any magnetisation), TAP, and belief propagation
with the Bethe free energy, exact on trees and measured to degrade past criticality on loops.
`hopfield` is the statistical mechanics of learning with its closed forms as the check: one pattern
is Curie–Weiss and the sampler retrieves at `m = tanh(βm)`; at finite load the
Amit–Gutfreund–Sompolinsky replica equations are solved in the crate, and bisection on them gives
the capacity **`α_c = 0.1379`** (AGS: 0.138), with retrieval present at `α = 0.02` and absent at
`0.10` in the samplers' own runs. `examples/learning_theory` shows the table. `dense_memory` is the
modern Hopfield network — degree-2 provably the classical one minus a constant, degree 3 and the
exponential memory holding hundreds of patterns in 100 spins where the classical one holds 14, and
Ramsauer's attention as the exponential memory's one-step update — and it is a *program*:
`to_hubo`/`to_program` write the memory as a higher-order `.ftp` model (an identity checked to
1e-9), from which the native annealer retrieves, the pairwise reduction is measured exact but
dynamically frozen, and the degree-2 memory reaches its exact ground state on a Chimera. `eqprop`
is equilibrium propagation for Boltzmann machines, its gradient theorem held by enumeration at
both of its convergence rates. `perceptron` is Gardner's storage problem on binary couplings, with
its first-moment bound (a theorem), the Krauth–Mézard capacity (a citation) and exact enumeration
kept apart — and the measured algorithmic gap: at `α = 0.5`, well under the capacity, annealing
succeeds in 20 of 20 instances at `N = 21` and 0 of 20 at `N = 401`. The spherical case is there
too, with Gardner's `α_c(κ)` *computed* in closed form (exactly 2 at zero margin), where the
convexity means a failure below the capacity is a budget and above it is the model — an
attribution the binary problem cannot make.

### The machines you can actually rent

`embed` did honest minor embedding onto **Chimera**, which D-Wave retired. `device::pegasus` and
`device::zephyr` build the topologies of the Advantage and Advantage2: `P₁₆` is **5,640 qubits /
40,484 couplers** at degree 15, `Z₁₅` is **7,440 / 71,736** at degree 20 — the vendor's published
figures, produced here from the coordinate rules. Transcribed from D-Wave's own generator and
checked against it at five sizes each on node count, coupler count and the *full degree histogram*,
because two different graphs can share a total.

`examples/embedding_tax` measures what a topology generation is worth, in counts rather than
seconds — the same table on any machine:

```text
--- K_16
hardware        sites   deg     used   longest    mean
Chimera C8        512     6      126        18    7.88
Pegasus P16      5640    15       49         7    3.06
Zephyr Z15       7440    20       48         6    3.00
```

Two and a half times the qubits and three times the chain length for the same sixteen variables.
**The chain column is the one to read**: sites are a budget, but a chain is a failure mode — held
together by a penalty, and when that penalty loses, the qubits of one variable disagree and the
variable has no value at all.

`Topology` carries the vendor's qubit numbering beside the graph, because Pegasus's is *sparse*: a
`P₁₆` spreads 5,640 qubits over indices 30 to 5,729, and a chain written in our dense indices would
program different qubits on a real machine.

### Sparsification, with the correctness property enumerated

A model denser than the fabric has two routes onto it. `embed` **places** it onto one specific
machine; `sparsify` **rewrites** it so no variable exceeds degree *d*, with no machine involved — a
variable of degree *k* becomes *c* copies bound into a path by a strong coupling, its edges shared
out and its bias split. The field names this as an open problem and the reference answer is one
unmaintained MATLAB file.

The copy count is `⌈(k−2)/(d−2)⌉` — character for character what `embed::site_lower_bound` derives
for a chain, because it is the same port-counting argument from the other side. A test asserts the
two agree.

**Ground-state preservation is checked by enumeration, not argued.** The whole sparsified state
space is enumerated and required to satisfy three things: every ground state has all copies
agreeing, each projects onto a ground state of the original, and **every** ground state of the
original is reached — the last being the one a rewrite can quietly fail while passing the first two.
A companion test drops the copy coupling below the derived bound and requires the property to break,
so the derivation is load-bearing rather than decorative.

### Both routes, from every language

`embed` places a model onto a specific machine; `sparsify` rewrites it to fit a degree budget. Both
now cross the C ABI, so Python, Zig and Julia can run the comparison below rather than take it on
trust — 201 symbols across four surfaces.

```python
k  = model.build(beta=0.5, seed=7)      # a K_12
hw = ferrotherm.pegasus(6)
k.site_lower_bound(hw)                  # 12 -- a PROOF, in microseconds
k.embed(hw, seed=7)                     # True; 25 sites, longest chain 4
run = k.embed_apply(hw); run.anneal()
state, broken = run.unembed(12)         # 0 broken chains
```

`site_lower_bound` is the question with a proof behind it: `embed` returning false means *this
search* did not find a placement, while a bound exceeding the machine's site count means **none
exists**.

### The crossover, and sparsification loses

*At what N does sparsify-plus-embed beat placing the model directly?* **Nowhere.**
`examples/sparsify_vs_embed` measures it in counts on both machines:

```text
=== Pegasus P16: 5640 sites, degree 15
  K_n       direct sites     direct longest       sparse sites     sparse longest
   16                 49                  7                 49                  7
   24                130                 14                758                 55
   32                237                 16          not found          not found
```

`K₂₄` costs 130 sites and a 14-site chain placed directly, against 758 sites and a 55-site run
through sparsification. It is the same tax paid twice — copies are chosen before the machine is
looked at, and the embedder then chains every one of them. The rows that tie do so because the model
already fits the machine's degree and `sparsify` returns it unchanged.

**Where a placer exists, place.** Sparsification is for a fabric with a fixed sparse topology and no
placer at all, where the question is not which is cheaper but whether the model runs.

### Fewer sequential barriers per sweep

A chromatic sweep runs one pass per colour, so the colour count *is* the number of sequential
barriers — and on the GPU path, the number of dispatches. `graph` now tries **DSATUR** after greedy
and after the bipartite check, and keeps it only when it strictly wins, because a different
colouring moves every seeded trajectory on that graph:

| graph | greedy | DSATUR | clique bound |
|---|---|---|---|
| lattice, Chimera, Z1 grid | 2 | 2 | 2 |
| Pegasus P₄ … P₁₆ | 4 | 4 | 4 |
| **Zephyr Z₆, Z₁₅** | 6 | **5** | 4 |
| **a compiled exactly-one model** | 4 | **3** | 3 |

Greedy is already optimal on Pegasus — it matches the clique bound. Zephyr and compiled counting
constraints each get a pass cheaper, on every fabric.

### How many ways are there to do the job

`model` answers a problem by name; it could not say whether the answer was the only one. A solve
runs `tries` independent anneals and keeps the best, so a model with a symmetry hands back one of
several optima and nothing distinguishes that from a unique answer. Every try is kept now, and the
node editor lists the alternatives:

```text
3 distinct ways to do this, all at energy -5.0000:
  1.  a=0  b=0  c=1
  2.  a=0  b=1  c=0
  3.  a=1  b=0  c=0
  (found by 40 independent tries -- evidence that these exist, not a
   proof there are no others. Raise the Solve node's tries to look harder.)
```

Distinctness is on the **decoded values**, never on the spins. The obvious justification — slack
bits float, so counting states over-reports — is wrong, and the test is named after the claim it
refuted: enumerating `at most two of four` exactly gives eleven assignments and eleven
minimum-energy states, because the penalty that makes a row hold also pins its slack. The real
reason is that the count must be a statement about the model rather than about how the compiler
chose to represent it.

### Elimination orders, and the models that were refused for having a bad one

Cost is `2^width`, so the elimination order is the whole price of exact inference. Finding the
optimal one is NP-hard; `Elimination::order_for` builds two heuristics and keeps the narrower —
min-fill, which is greedy and local, and nested dissection, which splits by a BFS level set (every
level of a breadth-first search is a separator, and on a grid the levels are its rows).

```text
       graph   spins  min-fill    kept   treewidth
  6x40 strip     240         8       7           6
  8x50 strip     400        11       9           8
 10x10 grid      100        13      10          10
 torus 10x10     100        23      20          20
 torus 12x12     144        26      24          24
 torus 14x14     196        34      28          28
```

The grid and every torus now land **on** the treewidth. `max_width` defaults to 24, so a 144-spin
torus ordered at 26 was refused for its *order* rather than its shape, and a 10×10 torus cost 8×
more than it had to.

Keeping the narrower of two heuristics is what makes this safe to adopt: the width can only go down,
so a model that was accepted stays accepted. Ties go to min-fill, the incumbent — an order change
with no width change is churn that moves which ground state comes back from a degenerate model.

### An answer that says how far from optimal it might be

Every other solver here, and every commercial machine in this field, returns "best found" and
nothing. `Method::Branch` returns a bracket:

```text
  bound  ≤  true optimum  ≤  energy          gap = energy − bound
```

The search already evaluated `fixed_energy − free_h_abs − free_abs` at every node to decide prunes
and discarded it. The minimum over the subtrees the node budget forced it to **abandon** is a valid
lower bound on the whole problem, so a truncated run reports how far off it might be. When the tree
is exhausted nothing is abandoned, `bound == energy`, and `proved_optimal` becomes a corollary of a
zero gap rather than a flag to be trusted on its own.

`Solution::gap` carries it up in the modeller's own units. The compiled energy is the objective plus
soft costs plus a constant offset that is identical for every feasible fully-decoded state, so the
offset cancels in a difference — no offset is computed and no convention is chosen:

```text
  objective(this) − objective(best)  =  energy(this) − energy(best)
```

**The falsifier is penalty invariance**, and it needs no enumeration. Compile the same model at
penalty 2, 20 and 200: the energies move by hundreds, the gap does not move at all. A sign error, a
dropped constant, or a bound in the wrong units all survive an agreement test against one
compilation, and none survive this.

The first version of the bound was not a bound, and exhaustive enumeration said so — −13.675 against
a true minimum of −14.913. When the budget runs out deep inside the first branch, the loop returns
before entering the second, so that sibling is abandoned and unrecorded.

### A penalty proved sufficient

`Model::effective_penalty` takes twice the largest pull on one literal set. Relaxing a constraint
frees every term touching its *support*, and those live in different literal sets:

```text
  Fix(a,0)  against  maximize a.is(1) + Σᵢ a.is(1)·bᵢ.is(1)

  n=2  auto_penalty=2  feasible=false  invalid=["a"]
  n=3  auto_penalty=2  feasible=false  violated=1
  n=4  auto_penalty=2  feasible=false  invalid=["a"]
```

Satisfiable models whose compiled optimum is not a solution, failing in two different shapes.

`Model::certified_penalty` proves one sufficient. The objective is a sum of terms contributing `c`
or `0`, so its whole range is `Σ|c|` and no change of assignment can gain more. Breaking a hard row
or an encoding costs `p·d`, where `d` is the smallest violation the compiled form admits — 1 for a
hard row and one-hot, 2 for domain-wall, since a violation there is an extra wall and every wall
costs `2p`. So `p·d > Σ|c|`, returned with one ULP added because the argument needs a strict
inequality and a tie is something the sampler may take.

It **refuses** rather than guessing where the argument does not reach. A binary encoding whose `k`
is not a power of two has invalid codewords costing exactly what valid ones cost — `add_penalty`
returns `false` — so `d = 0` and no penalty certifies it at any size. That is a fact about the
encoding, not a limit of the argument.

### How many ground states are there — counted, not estimated

`samples` reports "evidence of degeneracy, not a count of it", and `oracle::Exhaustive` stops at 26
spins. `exact::ground_degeneracy` counts, on anything narrow enough to eliminate, which is a claim
about the graph's **shape** rather than its size.

With `g₀` states at the ground energy `E₀` and a spectral gap `D`,

```text
  Z(β) = g₀ e^{-β E₀} (1 + (g₁/g₀) e^{-β D} + …)
  exp(ln Z(β) + β E₀) → g₀        error O((g₁/g₀) e^{-β D})
```

Every excited level contributes a positive term, so the estimate is an **upper bound that tightens
as β grows**. Two temperatures are evaluated, both are returned, and an integer is named only when
they are distinct, positive, finite, and agree — equal temperatures agree for a reason unrelated to
convergence, and at `(0, 0)` the estimate is `2ⁿ`, the total state count.

The oracle needs no enumeration. An odd antiferromagnetic N-ring is frustrated: one bond must break,
it can be any of the N, in either global orientation — exactly **2N** ground states.

```text
  N = 21   counted 42     enumeration would need 2^21 states, elimination needs 2^2
  N = 51   counted 102
  N = 101  counted 202
```

Finding it corrected an older defect. `initial_tables` emits nothing for a spin with no field and no
edges, and elimination then skipped a variable no table mentions — right for min-sum, which owes a
free spin zero energy, wrong for sum-product, which owes it a factor of two. `log_partition` was
short by `ln 2` per free spin. `tensor::Network::from_ising` had the same hole for the same reason,
so the two engines agreed on the wrong number to the last ulp; the regression test is scored against
brute-force enumeration instead.

### Contraction and elimination are the same computation

Markov & Shi (2008): contracting a tensor network is polynomial in its size and exponential in its
treewidth — which is `exact`'s `2^width` in the other field's notation. `tensor` says it out loud and
generalises it: any rank, any index dimension, indices summed when the contracting pair holds their
last live copies, and indices left **open** so a contraction yields a marginal instead of a scalar.

An order's price is reported before it is paid, matching `Elimination::width`:

```text
  Plan { peak_entries, flops }      →  Uncontractable::TooWide { entries, max }
```

Checked against the engine the crate already had — `from_ising` + `contract` against
`Elimination::log_partition`, and an open index against `Elimination::marginals` — on chains, rings
and 4×4/5×5 lattices at four temperatures. The lattice rows are the load-bearing ones: every site has
degree four, so they exercise summing an index carried by several tensors.

It is real-valued and exact: it contracts probability and partition-function networks, **not
amplitudes**, and performs no bond-dimension truncation.

### The certificate was accusing correct samplers, and now it does not

`certify` is what every sampler in this crate is scored by, so a defect in it is a defect in every
verification claim the crate makes. It had one.

It fits `beta` by maximum **pseudolikelihood** and took the interval from `sqrt(1/H)`, the inverse
Hessian. That is the variance of a real likelihood. Pseudolikelihood is a **composite** likelihood —
a product of conditionals that is not the likelihood of anything — and its asymptotic variance is the
Godambe sandwich `H⁻¹ J H⁻¹`. The two agree only when the multiplied terms are independent, and the
`n` conditionals from one configuration all read the same spins.

Measured on **exact independent draws**, where there is no sampler and so no sampler can be wrong:

| fixture | naive `H⁻¹` | Godambe sandwich |
|---|---|---|
| `ring(10, +1, 0)` | 12.0% | **4.0%** |
| `ring(10, −1, 0)` | 18.0% | **5.0%** |
| `ring(10, +1, .35)` | 12.5% | **6.5%** |
| `lattice2d(3)` | 2.0% | **4.0%** |
| `ring(10, +1, .2)`, β = 0.6 | 8.0% | **2.5%** |

against the 5% a 95% interval is allowed. `beta_eff` was unbiased throughout — to about 0.02% — so the
point estimate was never the problem. On identical data the naive standard error is a stable **0.76**
of the right one at every draw count tried: an interval 32% too narrow, reporting `BetaMismatch` on
correct samplers about one run in seven. A verification tool with that false-alarm rate is one people
learn to argue with.

`J` is now estimated by clustering on the configuration — one score term per draw — which is exactly
the dependence the naive form ignores. `the_interval_covers_at_the_rate_it_claims` measures the
calibration over 250 runs of exact draws and is two-sided, because an interval that never fires is as
broken as one that always does.

**How it surfaced, and what else it exposed.** A mutation of `multispin` that should have cost only
performance also failed the distribution test, with a 95% interval missing the truth by four
ten-thousandths. Chasing that showed 1.7%–13.3% false-failure rates across fixtures, and three
unrelated algorithms — Swendsen–Wang, Wolff, multi-spin — all failing at ~7–13% on `ring(10)` while a
3×3 lattice failed at 1.7%. That pattern indicts the instrument, not the samplers.

The same investigation measured the autocorrelation inflation, which had never been tested for its
purpose. It **over**-widens: on `ring(10, 1.0, 0.2)` at 500 draws over 300 seeds, coverage with it is
0.0% / 0.7% / 2.3% miss at thin 1 / 2 / 5, against 2.3% / 4.7% / 3.0% without. It scales by the
autocorrelation of the energy trace as a proxy for that of the estimator's score, and a Newey–West
estimate on the score itself asks for 1.05x where the proxy asks for 1.74x. It is kept — over-wide is
the conservative direction for an instrument whose job is to accuse, and one fixture is not grounds to
swap a known-conservative heuristic for a differently-wrong one — but it is now tested for the
contract it actually has: a more correlated chain gets a wider interval.

### Finding what a model already forces, and proving what it cannot satisfy

A model often forces its own variables before anything is solved: a `Fix`, an `Equal` to something
already fixed, an `ExactlyOne` whose other literals are all excluded. `model::presolve` is a fixpoint
over the constraint list that finds those, shrinking what the sampler has to explore — and when two
constraints demand different values of one variable, it returns the subset that proves it.

**Sound, deliberately not complete.** Every value it reports is one that *every* satisfying
assignment agrees on, checked by enumeration. The converse does not hold: there are forced variables
it will not find, and finding all of them is as hard as solving the model. A presolve claiming
completeness would be claiming to have solved the problem it is preparing. Concretely, exclusions
propagate only on two-valued domains, where "not this" is "that" — ruling one value out of a wide
categorical leaves a variable it does not track, and that is written down rather than discovered.

**Soft constraints are not propagated.** A solution is allowed to break one, so fixing a variable
from a soft row is as unsound as inventing a constraint — and it would be invisible in any model
whose soft rows happen to hold at the optimum.

**The conflict set is a witness, not a list.** Propagation over *only* the reported constraints must
still contradict itself, and that test caught a real defect: the `ExactlyOne` and `Cardinality`
closures pushed an EMPTY provenance, so a fix that followed from three other literals being false
forgot whatever made them false. The reported set then did not stand on its own. Both closure
directions carry their chain now, and both are tested — a fixture through one says nothing about the
other, which is exactly how the second survived a mutation pass after the first was fixed.

The claim is **minimal under propagation**: drop any constraint and the contradiction is no longer
found. That is weaker than an irreducible infeasible subset, which needs a complete solver, and it is
what the test checks — the property in the code and the claim in the documentation are the same
sentence.

### Asking every solver the same question

The crate has many solvers and had no way to ask them one question. Each takes its own parameters,
counts its own work, and returns its own `Outcome`, so "which is better here" could only be answered
by hand-tuning every arm and hoping the comparison was fair.

**The unit is the whole problem.** `tabu` counts iterations, `bls` counts moves, `sqa` counts Trotter
slices times steps times sweeps, `tempering` counts replicas times rounds — comparing those is
comparing labels. What all of them do underneath is propose a single-spin change and accept or reject
it, which is also what the ledger prices. So `Budget` is a proposal count and each arm converts it
into its own knobs.

Those conversions are measured rather than asserted: every arm must spend **within** its budget and
**at least a quarter of it**. An arm that overspends turns the portfolio into a measurement of who
cheated; one that spends a hundredth is not answering the same question as the others. The budget is
split, not handed to each arm in full — which is the mistake that makes a portfolio look free, beating
any single arm at "the same" cost while spending `k` times as much.

Every arm gets the same seed, deliberately: a portfolio whose arms are seeded differently measures
the seeds as much as the methods.

The claim is stated against the **worst** arm, not the best. Beating the best would require knowing
which that is, and not having to know is the entire reason a portfolio exists.

The tempering arm sizes its ladder with `adaptive::replicas_for` rather than a constant — today's
severed-ladder fix paying off in a second place. A fixed rung count still returns answers, just worse
ones, so no comparison *between* arms would catch it; it is tested as the property it is.

Eight mutations, and the three that survived a first pass are the useful part. Two were real gaps —
nothing checked that the ladder was model-sized, and the budget test bounded spending only from
above, so reporting zero passed. The third is recorded rather than papered over: every solver here
already recomputes its own energy, so `Found`'s recompute is defensive and no test can see it. What
it buys is that the portfolio does not *depend* on that staying true.

### Turning a width refusal into a slower answer

Elimination costs `2^width` in memory, so `TooWide` is a wall rather than a slowdown: past the cap
there is no answer at any price. Slicing changes the price. Pin a variable and it leaves the graph,
narrowing what remains; solve every assignment of the pinned set and combine, and the exact answer
comes back at `2^k` times the work for `k` pins. The caller sets the rate with `max_slices`, and a
budget too small to reach the cap is refused with the width rather than answered approximately.

**The difficulty is entirely in two constants, and both are invisible on a model without fields.**
`pin` was written for marginals, where it is called twice and its dropped terms cancel in the ratio.
Slicing calls it once per slice and sums, so nothing cancels:

- it zeroes the pinned node's own field, dropping `−h_i v` from every energy, so `log Z` needs
  `+ β h_i v` back — and `h_i` must be read from the graph **at the time of the pin**, because an
  earlier pin folds its couplings into later nodes' fields;
- it keeps the node rather than deleting it, so the pinned spin survives as a *free* spin worth a
  factor of two in `Z`, which is `− ln 2` per pin.

Min-sum takes only the first: a free spin costs no energy.

The fixtures therefore carry a field on **every** node. On a field-free model both corrections
multiply by zero, the bookkeeping could be entirely wrong, and every test would still pass — the same
shape as the MAX-SAT constant. And `log Z` is checked rather than the ground energy alone, because
`log Z` is where the constants land.

The returned ground state has its pinned values **written back**. Elimination solves a graph with
those nodes stripped and cannot know what they were, so without that step the state is a correct
answer to a different question — and a test asserts the state's own energy equals the number reported
beside it, which is what would catch it.

Checked against the direct computation where both can run, against brute force where the cap refuses,
and on a graph narrow enough that nothing is pinned at all. Eight mutations, eight killed.

### The control oracle, at the dimension a robot actually has

`mppi` is the workload this crate says connects a thermodynamic sampler to a robot, and its whole
claim is that a sampling controller can be scored against a provable optimum rather than against a
rival heuristic. That oracle was **scalar** — one state, one control — which is not a system anyone
controls.

`MatSystem` and `MatLqr` are the vector form: `x' = Ax + Bu` with cost `xᵀQx + uᵀRu`, and the exact
optimum from the discrete algebraic Riccati equation

```text
  P = Q + AᵀPA − AᵀPB (R + BᵀPB)⁻¹ BᵀPA,     K = (R + BᵀPB)⁻¹ BᵀPA,     u* = −Kx
```

solved by iterating from `P = Q`, with the small dense `matmul`, `transpose` and pivoted Gaussian
`solve` it needs — `linalg` carried only `jacobi_eig`, and the crate is std-only.

Four checks, and the last two are what make it an oracle rather than a routine:

- **`residual`** substitutes the answer back into the equation. An iteration that stopped early, or a
  derivation with a transpose in the wrong place, converges to *something*; only this says whether
  that something solves it. Under 1e-9 across five shapes and twelve seeds, with `P` symmetric —
  which the equation forces and a transpose error breaks.
- **The scalar `Lqr` agrees at n = m = 1**, two independently written implementations of the same
  equation.
- **`x₀ᵀPx₀` is what the policy spends, not a bound on it.** Roll the optimal gain forward four
  thousand steps and the summed stage costs converge to the number `P` named before the rollout
  began.
- **No perturbation of the gain costs less** — two hundred random perturbations, none cheaper. That
  is what "optimal" has to mean, and a residual check alone does not establish it.

Nine mutations, nine killed: the dropped correction term, an untransposed `A`, `R` omitted from the
inverse, a sign flip on the gain, a missing factor of `x` in the cost-to-go, a self-transposing
`transpose`, no pivoting, no singularity check, and back-substitution without its divide.

### An exact oracle where the old one was weakest

`onsager_log_z_density` integrates a `grid × grid` mesh, and its integrand has a logarithmic
singularity at criticality — so exactly where a two-dimensional sampler is hardest to check, the
oracle is least accurate. The **energy** density has a closed form:

```text
  U/N = −J coth(2K) [ 1 + (2/π)(2 tanh²(2K) − 1) K(κ) ],   K = βJ,  κ = 2 sinh(2K) / cosh²(2K)
```

`K(κ)` comes from Gauss's AGM identity, `K(k) = π / (2·AGM(1, √(1−k²)))` — quadratic convergence, so
machine precision in about five iterations, with no grid and no accuracy that degrades anywhere.

At criticality it does not even need that. `sinh(2K_c) = 1` makes `tanh²(2K_c) = 1/2`, the second
term vanishes, and **`U/N = −J√2` exactly**, the elliptic integral never consulted. That is the
sharpest single number available for testing a 2D sampler.

**The pole and the zero are the same float.** `κ = 1` exactly when `sinh(2K) = 1`, which is also
where the coefficient `2tanh²(2K) − 1` vanishes — so the naive expression is `0 × ∞`, and forming `κ`
directly puts it a rounding *above* one, outside the domain where `K` is real. The fix is to work
through the complementary modulus `|1 − sinh²|/cosh²`, which has no cancellation. The limit is zero
because the coefficient falls linearly while `K` grows only logarithmically.

That edge is reachable, not theoretical: at `beta_c` plus **one ulp**, `sinh(2K)` rounds to exactly
1.0 and the guard is the only thing between the caller and a `NaN`. At `beta_c` itself `sinh` lands a
rounding below one and the arithmetic works by luck — which is why a mutation removing the guard
survived every other test here until a fixture was written at the ulp that reaches it.

Checked four ways: the closed form at `K_c`, both temperature limits forced by bond-counting
(`−2J` and `0`), `K(k)` against its own defining integral by fine quadrature, and against
`−d(lnZ/N)/dβ` from the independent grid implementation. 8 of 8 mutations killed.

### A certified lower bound that cannot round upward

`sdp::Certificate::verify` re-checks a certificate from scratch and returns `eᵀy`. It summed with
`iter().sum()`, and left-to-right addition can round **up**: `[1.0, 3·2⁻⁵⁴, 3·2⁻⁵⁴]` has true sum
`1 + 1.5·2⁻⁵²` and sums in `f64` to `1 + 2·2⁻⁵²`. Over by 1.1e-16 — trivial in size, wrong in
direction, and a lower bound that exceeds the truth is not a bound.

The dual points `certified` produces are on `snap_down`'s power-of-two grid, where the sum is exact
and this changes nothing. But `verify` exists to check certificates it did **not** produce —
deserialised, hand-built, or from another implementation — and `y` is a public field. On that path
the grid is an assumption, and a re-verification that assumes what it is checking has stopped being
one.

`sum_down` uses Kahan–Babuška compensation, then subtracts `2ε|total| + n²ε²Σ|y|` so the direction is
certain. The first draft scaled the whole guard by `Σ|y|`, which is equally sound and useless under
cancellation: on `[1e16, 1, −1e16]`, true sum one, that guard is **26.6** and the function returns
−25.6. The bound above gives 9e-15 on the same input, because its leading term follows the answer
rather than the arithmetic that produced it.

Four mutations killed. The fifth — deleting the second-order term — **survives, and is recorded
rather than papered over**: it overtakes the first-order term only past `n ≈ 9.5e7`, so catching it
would need a dual point with a hundred million entries. The test asserts the two guards are
bit-identical at four thousand terms, which is the true statement, and says the crossover is what
would have to move for that to change.

### The MAX-SAT corpus, without a penalty or a modelling choice

`gset` brings in the max-cut benchmarks. `dimacs` brings in the other standard family, and a far
larger one: thirty years of published SAT and MAX-SAT competition instances, every one of them an
energy model this crate already solves.

The translation is exact. A clause is violated exactly when all its literals are false, so its cost
is the indicator of that event:

```text
  violated(C) = Π_{l ∈ C} [l is false] = Π_{l ∈ C} (1 − σ_l s_i) / 2
```

with `σ_l = ±1` for a positive or negated literal, since `l` is false exactly at `s_i = −σ_l`.
Expanding over subsets gives a `k`-body spin polynomial, which is what `Hubo` holds — and `reduce`
lowers it to pairwise hardware from there. No penalty, no scaling, nothing left for the caller to
choose.

**The constant is returned rather than dropped.** The expansion has an empty-subset term, `w / 2^k`
per clause, and `Factor::new` correctly refuses a term with no variables — so `to_hubo` hands it back
separately. Add it and the energy IS the unsatisfied weight, at *every* assignment, which is what
makes the check a proof rather than a spot test. Drop it and every optimum is still right and every
reported number is off by a fixed amount: an error that passes a solver test and fails only when
someone compares against a published result.

Real files carry clauses that are not clauses, and each is handled rather than rejected — a tautology
`x ∨ ¬x` has no energy and is dropped, a repeated literal collapses because `x ∨ x` is `x`, and an
empty clause is violated by everything and joins the constant. A clause count disagreeing with the
header is refused, because a truncated download otherwise parses into a perfectly valid smaller
instance whose optimum nobody can compare.

Verified exhaustively against the clause list at every assignment, against brute force for the
optimum, and end to end through `hubo::anneal`. Nine mutations, nine killed.

### Training an energy model without a sampler

`ebm::train` is contrastive divergence, and a negative phase needs a sampler — so its cost, its bias
and its seed all enter the fit. `train_pseudolikelihood` replaces the intractable normaliser with a
product of conditionals `P(s_i | rest)`, each a logistic function of the local field and computable
exactly from the data. Objective and gradient are both closed-form, the fit is deterministic, and
there is nothing to tune but the step.

The price is stated rather than hidden: the objective is *not* the likelihood. Its case rests on
**consistency** — the maximiser goes to the true parameters as data grows — so that is what the test
measures, and it measures the error *shrinking* rather than clearing a threshold, because a
threshold would be testing the fixture. Twenty times the data at least halves the mean parameter
error, landing under 0.1. This crate already leans on the same consistency: `certify` fits an inverse
temperature by pseudolikelihood, and its interval needed the Godambe sandwich for exactly the reason
PL is not a likelihood.

A model with hidden units is refused by name. "The rest" has to be observed, and a latent unit is
not — so the conditional does not exist, which is a different thing from being slower or looser.

**The gradient is checked against finite differences**, because a hand-derived gradient is where this
kind of code goes quietly wrong. An edge weight enters the field of *both* its endpoints, so its
derivative has two terms; drop either and the fit still converges to something, just not to the
maximiser of the stated objective — and no accuracy check on the result would reveal it. That
mutation, and seven others, are killed.

### The adaptive ladder's default was severed, and adapting did not repair it

`tempering::TemperingResult::swap_rates` states the criterion: healthy adjacent pairs sit roughly in
`[0.2, 0.6]`, and near-zero pairs mean "the ladder has a gap replicas cannot cross". `adaptive`
shipped eight replicas over `β ∈ [0.05, 4.0]`, and on a planted 14×14 glass that gives

```text
  0.17  0.01  0.00  0.01  0.07  0.54  0.28
```

Three pairs at or below 0.01 — a ladder cut in half, where cold replicas never reach the hot side
and parallel tempering silently degenerates into independent chains that still return a
tempering-shaped answer.

**And `adapt` does not fix it**, which is the module's whole purpose. On a 16×16 it went from three
severed pairs to four. The respaced ladder shows why:

```text
  0.050  0.175  0.249  0.295  0.350  0.502  0.976  4.000
```

Six of eight rungs crowded below `β = 1`, and an unbridgeable jump from 0.976 to the cold endpoint.
The endpoints are held deliberately — they are the physics the caller asked for — so with too few
rungs for the span there is no arrangement that connects. Respacing cannot manufacture a replica.

The count a span needs grows as `sqrt(n)`: swap acceptance is governed by `Δβ · ΔE`, and the energy
fluctuation grows as the square root of the model. `replicas_for(n, β_min, β_max)` is
`⌈0.35 · sqrt(n) · ln(β_max/β_min)⌉` — the **shape** from that argument, the constant measured. What
`adapt` needs, at five seeds each:

| spins | 100 | 196 | 256 |
|---|---|---|---|
| needed | 12 | 16 | 24 |
| shipped | 8 | 8 | 8 |

On five sizes the constant was *not* fitted to — 36, 64, 144, 324 and 400 spins — the rule produced
no severed pair and no pair below 0.2, with worst-pair acceptance falling gently from 0.50 to 0.30 as
the models grew.

`Params::for_graph(&g)` applies it. `Params::default()` still ships eight, because a `Default` impl
has no model to read and a replica count needs `n` — the doc says so and points at `for_graph`, since
any constant is wrong at some size and a constant wrong at a *different* size is not a fix. That is
the structural difference from `sqa`, where the derived value depends only on fields `Params` already
carries and so could go straight into `Default`.

`Outcome::gaps()` returns the severed pairs. `swap_rates` always carried the evidence; what was
missing was the judgement, so a caller could receive a disconnected ladder without a way to ask.

### The simulated-quantum default was running classical annealing

`sqa` shipped `M = 4` Trotter slices with `beta = 10` and `gamma_max = 3`. The Suzuki–Trotter error
is governed by `beta * Gamma / M`, and that triple is a ratio of **7.5** — where `tanh(7.5) ≈ 1`
makes the slice coupling `J⊥ ≈ 0`, the slices decouple, and what runs is independent classical
annealing on four copies of the spins.

The module's own single-spin oracle says so. At `Gamma = 3`, `beta = 10`, for one spin in a
longitudinal field:

| | magnetisation |
|---|---|
| exact quantum, `(h/E)·tanh(βE)` | **0.316** |
| what `M = 4` simulates | 0.987 |
| purely classical, `tanh(βh)` | 1.000 |

The error is larger than the quantity being measured, and the simulation sits fifty times closer to
classical than to quantum. The transverse field does essentially nothing until `Gamma` falls below
about 0.2, near the end of the anneal. The module's convergence test checks `M` = 8, 32, 128 and 512
— it never checked the value it shipped.

**And it costs answers.** Mean excess over the planted optimum on four instances, 30 seeds each, at
*identical* proposal counts — annealing steps traded for slices, so every row does the same work:

| ratio `βΓ/M` | M | 8×8 | 10×10 | 12×12 | 14×14 | total |
|---|---|---|---|---|---|---|
| 1.00 | 30 | 0.49 | 2.49 | 2.65 | 2.72 | 8.35 |
| 1.50 | 20 | 0.62 | 2.67 | 2.35 | 2.18 | 7.81 |
| 1.88 | 16 | 0.28 | 2.13 | 2.59 | 2.27 | **7.27** |
| 2.00 | 15 | 0.35 | 2.18 | 2.90 | 2.43 | 7.85 |
| 2.50 | 12 | 0.42 | 2.40 | 3.36 | 2.49 | 8.68 |
| **7.50** | **4** | 4.51 | 6.93 | 7.50 | 5.65 | **24.59** ← shipped |

A broad plateau from about 1 to 2.5, with the shipped value three times worse than any of it — and
on the 8×8 the default found the planted optimum 3 times in 30 where a plateau setting found it 27.

`trotter` is now derived: `M = round(beta * gamma_max / TROTTER_RATIO)` with the ratio at 2.0, a
round number inside the plateau rather than the argmin of one experiment. `beta`, `gamma_max` and
`trotter` are one choice, not three, and a test asserts the default is what its own rule gives — the
drift that produced 7.5 cannot recur silently. `serve` carried the same literal `4` and now derives
it too, so an HTTP caller passing their own `beta` gets a slice count that tracks it.

**How the measurement went wrong three times first.** The comparison has to hold proposals fixed, and
that is easy to get wrong in three different ways: comparing at equal `steps` hands the higher-`M`
arm proportionally more work; comparing quantum against classical within a row while varying `M`
across rows matches the wrong pair; and comparing at a budget eight times the default's answers a
question about budget rather than about `M`. All three produced confident, wrong conclusions —
including one that said more slices are monotonically *worse*.

### A higher-order reduction with no penalty to get wrong

`to_pairwise` is Rosenberg's reduction: define an ancilla, then **bribe** the model into respecting
the definition with a penalty larger than the whole model is worth. It works, and it costs something
that does not show up in a correctness test. The penalty enters the energy, so the reduced model's
scale is set by the enforcement rather than by the problem — and `Schedule::for_instance` reads
exactly that scale to choose an annealing ladder, while a fabric with four-bit coefficients has to
quantise it.

`to_pairwise_exact` needs no penalty, because its identities are exact minima rather than
constrained definitions. For binary `x` and a **negative** coefficient,

```text
  c · x_1 ⋯ x_k  =  min_y  c · y · (x_1 + ⋯ + x_k − (k−1))        (Freedman–Drineas)
```

one auxiliary, every term quadratic. For a **positive** coefficient, Ishikawa (2011) does it with
`⌊(k−1)/2⌋` auxiliaries. Neither has a coefficient that has to be "large enough".

**The trade, measured** — a single factor of each arity, ancillas and `Graph::flip_gap_max`:

| arity | ancillas (Rosenberg / free) | energy scale (Rosenberg / free) |
|---|---|---|
| 3 | 1 / 1 | 162 / **16** |
| 4 | 2 / 5 | 502 / **48** |
| 5 | 5 / 16 | 2926 / **210** |
| 6 | 12 / 48 | 11670 / **782** |
| 7 | 15 / 106 | 35018 / **2906** |

Up to seven times the ancillas for an energy scale about twelve times tighter. On multi-term models
the scale ratios are 24×, 9.3× and 16.9× at 1×, 3.3× and 3.5× the ancillas. Which way that trade goes
is a property of the hardware, not of the reduction, so both are shipped and neither is a default.

Past arity nine the penalty-free cost stops being a trade — 1490 ancillas against 68 at arity ten —
and `MAX_ANCILLAS` refuses there, naming `to_pairwise` as the alternative rather than allocating.

**Verified by exhaustive minimisation.** For every assignment of the original variables, the reduced
energy minimised over every auxiliary assignment equals the original — the same oracle the Rosenberg
path uses, plus a check that the reported offset has the right *sign*, which no state-ordering test
can see: a reduction reporting it backwards lands the caller exactly twice the offset away.

The identities are also checked directly, one monomial at a time, at every arity to eight and both
signs. That separation matters and was found by mutation: replacing Ishikawa's `⌊(k−1)/2⌋`
auxiliaries with a bare `1` survives every end-to-end test, because the two agree at arities three
and four and an arity-five *factor* needs sixteen auxiliaries — past what exhaustive minimisation can
reach. An arity-five *monomial* needs two.

### Every temperature from one run, and two textbook recipes that do not converge

Every other sampler here runs AT a temperature, so a curve — energy against temperature, a heat
capacity, an entropy — costs one run per point. `wanglandau` estimates `g(E)`, the number of states
at each energy, which is a property of the model and carries no temperature at all. Then
`Z(beta) = sum_E g(E) exp(-beta E)` answers every temperature by a sum, including the cold ones where
an ordinary chain is stuck behind a barrier.

Checked against exact variable elimination at five temperatures spanning a factor of forty, from one
run that never saw a temperature — and against enumeration level by level, as an absolute count
rather than a shape, because anchoring to `sum_E g(E) = 2^n` turns the estimate into a number of
states rather than a number of states times an unknown constant.

**Equal-width energy bins are wrong on a spin model.** A 12-spin ring with `J = 1, h = 0.2` has 148
distinct energies; cut that into 24 equal bins and several energy LEVELS land exactly on a bin edge.
Those bins collect ~72,000 visits where their neighbours collect ~188,000 — a ratio of 0.39,
permanently under any flatness threshold — and the walk cannot converge at any budget. It reached
`ln f = 1e-3` in 1.35 million steps and made no further progress in **four hundred million**. The
symptom is indistinguishable from slow mixing. So this bins by level: one bin per energy the model
actually has. No empty bins, no split levels, no width to choose. A continuous spectrum is refused by
name instead.

**Halving the modification factor does not converge either.** The error SATURATES — whatever
statistical error a stage ends with is frozen into `ln g` when `f` drops. Worst error in `ln g` on a
4×4 lattice over four seeds:

| target `ln f` | halving | `1/t` |
|---|---|---|
| 1e-4 | 0.2392 | 0.2392 |
| 1e-5 | 0.2147 | **0.0881** |
| 1e-6 | 0.2110 | **0.0292** |
| 1e-7 | 0.2104 | **0.0049** |

Three more orders of magnitude of work buys nothing under halving. Switching to `ln f = 1/t` once `f`
falls below `1/t` (Belardinelli & Pereyra 2007) removes the saturation: 43× better at the tightest
target, falling as `1/sqrt(t)`. The test tolerance is **0.15** precisely because `1/t` reaches 0.088
and halving saturates at 0.215 — a number between them makes the schedule itself the thing under test.

The order is a handover, not a replacement. Starting in the `1/t` regime rather than arriving at it
gives worst errors of 7.7, 7.0, 12.1 and 34.7 across the fixtures: `1/t` refines an estimate and
cannot build one, because early on it is enormous and writes noise that later, tinier increments
cannot repair.

**Three guards were written for a problem that did not exist.** Flatness is judged over levels found
so far, which at the first check may be one — and a histogram over one level is trivially flat, so
`ln f` could halve before the walk has been anywhere. A discovery phase, a histogram reset on late
discovery, and a minimum-visit bar were each added to prevent that. Measured, all three were inert:

```text
                       discovery on / off      reset on / off
  ring(12, 1, 0.2)      0.0536 / 0.0552        0.0590 / 0.0590
  ring(16, 1, 0.2)      0.0978 / 0.0606        0.0746 / 0.0761
  4x4 lattice           0.0292 / 0.0226        0.0190 / 0.0214
  5x5 lattice           0.0193 / 0.0220        0.0209 / 0.0209
```

No direction, no fixture outside noise, complete level sets either way. All three are gone, and the
flatness ratio alone does the job. Each removal was decided by measurement after a plausible
mechanism argument turned out to be wrong — three times in a row, which is the honest reason this
section exists.

### 64 replicas in one word

`gibbs` visits one site of one replica at a time, which is the wrong shape for what this crate mostly
does with chains — a tempering ladder, a population, a disorder average. `multispin` packs 64
replicas into one `u64` per site and updates them with bitwise operations.

With every `|J_ij|` equal to one value `J`, the local field needs no multiplication:

```text
  f_i = J * sum_j sign(J_ij) s_j + h_i  =  J * (2k - deg_i) + h_i
```

`k` counts the neighbours whose signed contribution is `+1`; that contribution is `s_j XOR (J_ij < 0)`
— one XOR — and `k` is a per-lane popcount computed by a ripple-carry adder over bit planes. The field
then takes only `deg_i + 1` values, so acceptance is a small per-site table. Mixed magnitudes are
refused carrying the magnitudes found, which includes `planted::frustrated_loops`: it overlaps its
planted loops, so their couplings add to 1, 2 and 3.

**The randomness is drawn from the top and stops when it stops mattering.** Bit-sliced, each lane
needs its own uniform, naively 32 words per site — *more* RNG than the 64 scalar draws it replaces, so
the optimisation would have been a pessimisation. A comparison is decided by its leading bits, so
drawing from the most significant end and stopping once every lane has resolved costs **under 12 words
per site, measured**, and is exact rather than approximate: the bits not drawn could not have changed
any answer. `sweep` returns the count, so the claim is a measurement.

**The test that matters is the one about independence.** Share one random word across the lanes and all
64 replicas become the same chain — and every one of them is still an exactly correct sample of the
model. Certification passes, energies match enumeration, and the sampler delivers one replica's worth
of information while reporting sixty-four. No per-replica check can see it, so lane independence is
tested directly, by cross-lane magnetisation correlation. A 12-mutation battery kills 12 of 12,
including that one.

The per-lane spread is also what makes the correctness test writable. Checking each of 64 lanes against
a fixed tolerance cannot be done correctly — the spread is sampling noise, so any threshold tight enough
to catch a biased lane is one an honest lane will cross, and the first draft failed on lane 29 at 1.9
sigma while the mean over all 64 sat 0.00004 from the enumerated value. The tolerance is derived from
the observed spread instead, which yields a second guard on the duplication trap for free: identical
lanes have a spread of exactly zero.

### Critical slowing down, measured on both sides of it

Every other sampler in this crate flips one spin at a time — `gibbs` is the kernel and `tempering`,
`adaptive`, `popanneal` and `sqa` all schedule it. `icm` is the one alternative move, and it is
isoenergetic and restricted to zero-field glasses. So the failure mode single-spin dynamics is worst
at was, until now, unaddressed and unmeasured: at a critical point the correlation length diverges
with the lattice and a single-spin sampler has to move a domain of size `L` one spin at a time.

`cargo run --release --example critical_slowdown` — `tau_int` of `|m|` by Sokal windowing at
`beta_c = ln(1+sqrt2)/2`, 40,000 draws after 4,000 burn-in sweeps:

| L | Gibbs τ/sweep | Gibbs τ/visit | SW τ/sweep | SW τ/visit | Wolff τ/sweep | Wolff τ/visit |
|---|---|---|---|---|---|---|
| 8 | 6.48 | 415 | 2.43 | 156 | 1.48 | **61** |
| 12 | 15.13 | 2,178 | 2.64 | 380 | 1.96 | **166** |
| 16 | 32.39 | 8,292 | 2.93 | 749 | 2.10 | **297** |
| 24 | 78.41 | 45,166 | 3.13 | 1,803 | 2.56 | **732** |
| 32 | 100.85 | 103,274 | 3.73 | 3,823 | 2.87 | **1,352** |

```text
  Gibbs  z = 2.06 per sweep (literature 2.17)     SW  z = 0.29 (literature ~0.25)     Wolff  z = 0.46
```

At L = 32 an independent sample costs 103,274 spin visits under Gibbs and 1,352 under Wolff — **76×**,
widening as `L^1.84`. `<|m|>` agrees across all three methods at every size, which is what makes the
comparison a speedup rather than a different answer arrived at faster.

Two cost columns, because a sweep is a convention and the two moves do not mean the same thing by it.
The per-visit column comes from the ledger, and it is the one that survives contact with hardware:
per sweep Wolff beats SW, per spin visited their exponents are the same (2.22 vs 2.29) and Wolff wins
on the prefactor alone.

### Validity is balance, not the sign of the couplings

Swendsen–Wang and Wolff need a ferromagnet, and "has no negative couplings" is the wrong test for
that. A gauge `s_i → σ_i s_i` maps `J_ij → σ_i σ_j J_ij` and leaves every energy alone, so what is
actually required is that **some** gauge makes the model ferromagnetic — signed-graph *balance*
(Harary): every cycle has a positive coupling product. Apply a random gauge to a 6×6 ferromagnet and
28 of its couplings go negative; it is still the same model, and `cluster::gauge` samples it exactly.

One BFS decides balance and constructs the gauge in the same traversal, so the general case costs
nothing over the special one. A refusal carries the **frustrated cycle**, not an edge — any single
edge can be satisfied by choosing a sign, so an edge is never the obstruction, while a cycle is a
proof the caller can check by multiplying its couplings.

```text
  gauge(g) -> Result<Vec<i8>, Frustrated { cycle }>      apply_gauge(g, σ) -> Graph
```

Scored against `certify` on a ring, a 3×3 lattice, a *disguised* 3×3 lattice and a graph carrying a
coupling of exactly zero, for both moves; refusals checked on odd antiferromagnetic rings (where the
witness's product is verified negative) and on `planted::frustrated_loops`.

**Fields are an extra vertex, not a special case.** `with_ghost` couples every biased site to one
added spin at `J_ig = h_i` and reads the physical state back as `ŝ_i = s_i · s_g` — exact, since
`s_g² = 1` cancels out of both energy terms. The same balance test then decides the field case, and
gives a sharper answer than the folklore: a cycle through the ghost has product `J_ij h_i h_j`, so

| model | balance | this crate |
|---|---|---|
| ferromagnet, **uniform** field (either sign) | balanced | **sampled** — "Swendsen–Wang cannot handle a field" is true of the move as stated and false of the model |
| ferromagnet, **mixed-sign** fields | unbalanced | refused, with a cycle through the ghost naming the two sites whose fields disagree |
| antiferromagnet, uniform field | unbalanced | refused |

Checked by `certify` at three field strengths and both signs, and separately by `<m>` against exact
enumeration: 0.337 on `ring(10, 1.0, 0.4)` at `beta = 0.4`, reproduced to 0.01 by both moves. That
second check is there because dropping the ghost on the way out sends `<m>` to zero by symmetry —
a loud failure where a subtly wrong distribution would be a quiet one.

`Frustrated::product(&g)` multiplies the couplings around the witness cycle, using `h_i` for the
ghost's edges, so a caller can check a refusal without trusting the code that produced it.

**Two defects this found in itself.** An earlier `wolff_sweep` ran single-cluster steps "until `n`
spins have been visited", to make a Wolff sweep comparable to a Gibbs sweep. That makes the number of
steps a function of the cluster sizes, hence of the state: ordered configurations produce large
clusters and end the sweep sooner, so sweeps end preferentially just after a large flip. It is
optional stopping, every individual move is exactly correct, and on `ring(10)` at `beta = 0.4` it
returned `<E> = -4.3262` where enumeration gives `-3.8009`. Any fixed step count reproduces the exact
value. `certify` caught it as a sampled `beta` of 0.4492 against a requested 0.4000 — six standard
errors — while the total-variation distance stayed under its noise floor and reported nothing.

The reason TV reported nothing is the second defect, and it was in the test: the check was a
hand-written `tv < noise_floor`. The floor is `0.5 sqrt(2^n / ess)`, which at n = 16 and 4,000 draws
is **2.31**, and total variation between two distributions cannot exceed 1 — so the comparison passed
for every possible sampler, including one returning a constant. Two of that test's three models were
decorative. `certify` already knew, raising `TooFewSamples` exactly when the floor reaches 1; the test
now asserts `cert.passed()`, so the power of the check and the result of the check are the same call.

A 12-mutation battery covers the module: bond probability, both alignment tests, the SW half-coin,
the Wolff flip-with-probability-one, seed uniformity (checked on a disconnected graph, since a fixed
seed is still a valid kernel and only fails to be ergodic), the gauge round trip, the field
transformation, the zero-coupling guard, and the state-dependent stopping rule above. All 12 are
killed.

### The ladder knows the instance's energy scale

`β` and energy enter the Boltzmann weight only as the product `βE`, so a ladder in absolute `β` is a
claim about the units the modeller happened to write in. On `planted::frustrated_loops(8, 96, 3)`,
worst excess over the planted optimum across five seeds:

| coefficient scale | fixed ladder | `Schedule::for_instance` |
|---|---|---|
| 1e-3 | **52.08%** | 2.08% |
| 1e-2 | 43.75% | 2.08% |
| 1e0 | 2.08% | 2.08% |
| 1e2 | 20.83% | 2.08% |
| 1e3 | 20.83% | 2.08% |

`Graph::flip_gap_max` is the instance's energy scale — `max_i 2(Σ_j |w_ij| + |h_i|)`, an exact bound
on what one flip can change. The oracle is a theorem rather than a benchmark: scale the Hamiltonian
by `k` and the ladder by `1/k` and, for `k` a power of two, every multiplication is exact in
IEEE-754, so the trajectory is **bit-identical** and the energy scales by exactly `k`.

## Positions this crate takes

0. **A sampler returns samples.** Every commercial machine in this field returns "best found", and
   so did this crate until `samples` landed. A `SampleSet` carries the distribution its states came
   from and *refuses* an expectation value where there is none — averaging over a tabu search's
   trajectory produces a number the same shape as `⟨s_i⟩` and estimates nothing. Every estimate
   carries an error bar deflated by the chain's slowest autocorrelation, checked against exact
   enumeration above. The workbench shows it: at `β_c` on a 16×16 lattice, 2,000 draws are worth
   **4** independent ones, so `⟨M⟩ = 0.103 ± 0.71` — and that width is the answer, not a defect.
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
