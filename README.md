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
| 2D adaptive parallel tempering over (β, W₀) — *one MATLAB file, June 2025* | `adaptive` — respacing to equal acceptance, plus a (β, coupling-scale) grid | **shipped; mechanism verified, payoff measured absent** |
| Thermodynamic linear algebra (Aifer et al. / Normal Computing) | `tla` — OU-network SPD solves + bias-free exact-transition integrator | **shipped, verified** |
| Torx gradient estimators (Extropic) | `program` — REINFORCE + parameter-shift + **EBM-kernel** (one trajectory + one auxiliary draw) | **shipped, verified** |
| DTM — denoising thermodynamic models (Extropic's flagship architecture) | `dtm` — forward kernels, pattern grids, contrastive chain training, ACP, TC penalty | **shipped, verified** |
| **Fitting an EBM to data (the training half every EBM stack needs)** | `ebm` — contrastive divergence + **exact** likelihood by enumeration | **shipped, verified** — the fixed point is moment matching, checked against enumeration rather than against more sampling |
| Lattice Random Walk (Normal Computing CN101 algorithm) | `lrw` — ternary-increment SDE integration, exact-moment identities | **shipped, verified** |
| Simulated bifurcation (Toshiba bSB/dSB) | `sbm` — symplectic Ising machines vs enumerated ground states | **shipped, verified** |
| Hosted simulator APIs (extropic.dev) | `web/gibbs_bench.html` + `ffi` (wasm C ABI) — on YOUR device | **shipped**; the page verifies itself against Onsager in your browser before reporting a rate |
| **Fabricated CMOS annealing silicon (Hitachi)** | `ferrotherm-cloud::hitachi` — 384×384 King's graph, four-bit coefficients, over a free public API | **shipped, conventions measured** |
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
fabric**. `embed::pegasus_clique` places K_172 on the Advantage's P₁₆ — ells at chain 17 plus the fabric's
four provably-universal wires at chain 15 (busclique's K_180 is within 5%; the heuristic search
reaches K_80) — and `embed::chimera_clique` the classic `K_{t·m}`. Each is verified at every size by
`Embedding::verify` against the shipped fabric, and the interval and quantifier arithmetic each
rests on is machine-checked by Kani, exhaustively. The one remaining structured gap is exact and
recorded: eight chains on Pegasus (busclique's staggered-fragment diagonal). `ft_clique_embed` carries the constructions to Python, Zig and Julia;
`examples/embedding_tax` shows them beside the search and the frontier.

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
unconditional upper bound. On every surface as `ft_ln_z_*`.

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
Ramsauer's attention as the exponential memory's one-step update — and `eqprop` is equilibrium
propagation for Boltzmann machines, its gradient theorem held by enumeration at both of its
convergence rates.

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
