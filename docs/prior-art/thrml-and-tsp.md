# THRML and Thermalizing Stochastic Programs, as prior art

ferrotherm is independent and wholly original. The works below are prior art and
inspiration. This repository is not a port, a fork, or a derivative of their
source, and it does not ship their text. The Boltzmann distribution, Gibbs
sampling, and the chain rule for relative entropy are older than both projects.
Where a comparison turns on one of those, the section says so.

Every third-party name below is tied to its owner and to a link. A mark is
written only where the owner has a live filing or a registration. THRML™ is a
pending mark of Extropic Corp. (USPTO serial 99120424, filed 3 April 2025,
notice of allowance 9 December 2025, statement of use received 27 April 2026,
no registration number on the record consulted 7 November 2026:
<https://www.trademarkia.com/thrml-99120424>). It is written with ™, not ®.
"Extropic" is the company name of Extropic Corporation (also Extropic Corp. on
the USPTO record). No separate registered word mark was confirmed for it, so
none is printed. JAX is the numerical library maintained by Google at
<https://github.com/jax-ml/jax>. The repository's Apache-2.0 notice grants no
trademark licence beyond customary identification of origin, and the repository
does not print a ™ or ® on the name, so this note does not invent one.
NVIDIA® is a registered trademark of NVIDIA Corporation. Intel® is a registered
trademark of Intel Corporation. macOS is a trademark of Apple Inc. arXiv® is a
trademark of Cornell University. ML.ENERGY is the name of the initiative that
publishes Zeus; Zeus is their library, not a mark this note adds a symbol to.

## Sources, and the date they were read

Read on 7 November 2026.

| Work | Owner | What was read | Link |
| --- | --- | --- | --- |
| THRML™ 0.1.4 | Extropic (authors listed as Extropic AI on the package) | The PyPI sdist, 2,515 lines under `thrml/`, the tests, `pyproject.toml`. Wheel sha256 `6e2f38cecb562589d230ca063b5fcb5d2a6533201e37bb70c1f2dac4a63a0858`, uploaded 4 August 2026. Apache-2.0. | <https://github.com/extropic-ai/thrml>, <https://pypi.org/project/thrml/0.1.4/> |
| THRML™ documentation | Extropic | The docs front page and the quickstart it prints; the codon-optimisation walkthrough. | <https://docs.thrml.ai/en/latest/>, <https://docs.thrml.ai/en/latest/papers/codon-optimization/> |
| THRML™ paper | Extropic Corp., with Isaac L. Chuang (MIT EECS) | arXiv:2510.23972, HTML dated 24 August 2026, asset path `v2`. Title: *An efficient probabilistic hardware architecture for diffusion-like models*. | <https://arxiv.org/abs/2510.23972> |
| Thermalizing Stochastic Programs | Extropic Corporation (one author also CUNY Graduate Center) | arXiv:2608.01615, the canonical abstract page, HTML dated 24 August 2026. Sections I–V and Appendices A, C–M, including Table 4. | <https://arxiv.org/abs/2608.01615> |
| Companion PSC paper | Extropic authors, cited by the TSP paper as Ref. [84] | Abstract and the role TSP assigns it. Not re-derived here. | <https://arxiv.org/abs/2608.01612> |
| Zeus | ML.ENERGY Initiative; Jie You, Jae-Won Chung, Mosharaf Chowdhury | The measurement discipline (hardware counters, idle baseline, repeated trials), from the project page and the NSDI 2023 paper. The harness does not depend on the Zeus package. | <https://ml.energy/zeus/>, <https://github.com/ml-energy/zeus>, <https://www.usenix.org/conference/nsdi23/presentation/you> |
| Public Extropic organisation | Extropic | GitHub API listing, 7 November 2026: `thrml`, `torx`, `codon_opt`, `thrml-skill`, `sparse-transformers`. | <https://github.com/extropic-ai> |

`torx` is Apache-2.0 (<https://github.com/extropic-ai/torx>). `codon_opt` and
`thrml-skill` had a null license field on that API response. There is no
repository named thermalizers. TSP §V hopes that thermalizers "provides a
practical place" to express a process in torx, compile it, and sample it with
THRML™. That layer is a paper, not a release this comparison could run.

GitHub `thrml` `main` still declares version 0.1.4 in `pyproject.toml` (fetched
the same day; the repository was pushed 3 November 2026). The runner executes
the PyPI wheel named above, not an unreleased checkout.

## What each artefact is

**THRML™** is a JAX library for block Gibbs sampling of hypergraphical
energy-based models. The 0.1.4 README states the design point: blocked Gibbs on
sparse heterogeneous graphs, arbitrary PyTree node states, discrete EBM
utilities, and a place to prototype against future Extropic hardware. The docs
front page says the same structure is what Extropic's hardware is built to
accelerate. Installation is `pip install thrml`, Python ≥ 3.10. Declared
dependencies are `equinox` and `jaxtyping`. JAX is not pinned.

**The THRML™ paper** (arXiv:2510.23972) proposes denoising thermodynamic models
(DTMs): a chain of energy-based models used as denoising steps, on a denoising
thermodynamic computer architecture (DTCA) of sparse locally connected Boltzmann
machines and an all-transistor random-number generator. Section III and
Appendix E are a system-level energy model. The abstract's 10,000× figure is
that model, set against GPU baselines on binarized Fashion-MNIST by Fréchet
Inception Distance. Appendix E.5 says the model "almost certainly underestimates"
a complete device and is useful because the claimed gap is "at least several
orders of magnitude."

**Thermalizing Stochastic Programs** (arXiv:2608.01615) is a compilation paper.
A stochastic program is a directed factor graph of Markov kernels, equivalently
a parametrized stochastic circuit (TSP §II.3–II.4, pointing at arXiv:2608.01612
for the torx framework). Each factor is replaced by a thermodynamic kernel: the
conditional of a thermodynamic hypergraphical model obtained by clamping inputs,
thermalizing the rest, and reading outputs (TSP Eq. 19, §III.1). The loss is the
trajectory KL, which the chain rule splits into per-kernel conditional KLs
(Eq. 17). Two refinements follow: context matching (§III.2) and trajectory-level
REINFORCE (§III.4, Appendix I). Section IV demonstrates the framework in
software on the device model of Ref. [45], which is arXiv:2510.23972.

**ferrotherm** is a zero-dependency Rust library: sparse energy-based models,
chromatic block-Gibbs, annealing and parallel tempering, thermodynamic linear
algebra, stochastic differentiable programs, a variational compiler onto device
graphs, and a joules ledger whose evidence grade is part of the value. The
sampler, the compiler, and the ledger are checked against exact distributions
or closed forms. That is the original work. The papers above are what it is
compared with.

## Model classes

TSP §II.1 defines a thermodynamic hypergraphical model as a Boltzmann law on a
finite hypergraph,

`E(s) = −Σ_e J_e Π_{v in e} s_v`, `s ∈ {−1,+1}^n`,

with the order equal to the largest edge. A pairwise model is called a
Boltzmann machine (Eq. 4). A kernel is the same object with vertices partitioned
into input, hidden, and output; clamping the inputs and marginalising the hidden
spins yields the forward conditional (Eq. 5). The forward is invariant under
energy shifts that depend only on the input; the backward conditional is not
(Eqs. 7–9). Appendix J.4 extends the target to a continuous vector by a linear
encoding into p-bits, which preserves polynomial order.

THRML™ 0.1.4 implements that discrete fragment.

- `SpinNode` is a variable in `{−1,+1}`, stored as a JAX `bool` (`True` = +1,
  `False` = −1). `CategoricalNode` is an integer in `[0, K)`. Both classes have
  an empty body. Identity is an auto-incrementing integer on `_UniqueID`
  (`thrml/pgm.py`). There is no name, no domain object, and no constraint.
  `K` for a categorical node is not stored on the node. It is the trailing
  dimension of a weight tensor and the `n_categories` field of
  `CategoricalGibbsConditional`.
- `IsingEBM` (`thrml/models/ising.py`) is the energy
  `E(s) = −β (Σ b_i s_i + Σ J_ij s_i s_j)`. `β` is an attribute of the model
  and is multiplied into the factor weights when `factors` is read.
- `DiscreteEBMFactor` (`thrml/models/discrete_ebm.py`) is a batch of terms
  `s_1 … s_M · W[c_1, …, c_N]`. The factor's `energy` returns the negative of
  that sum, so the Boltzmann weight is `exp(−E)` with `E` already containing
  whatever temperature the caller folded into `W`. `SpinEBMFactor` and
  `CategoricalEBMFactor` are the pure-spin and pure-categorical specialisations.
  The docstring states that a variable repeated inside one interaction is not
  rejected, and that the resulting chain can disagree with the Boltzmann law.
- `BernoulliConditional` samples `P(S = s) ∝ exp(γ s)` by `sigmoid(2γ)`.
  `SoftmaxConditional` samples `P(X = k) ∝ exp(θ_k)`. Those are the Gibbs
  conditionals of the energies above, and they match ferrotherm's sign
  convention: `Graph::energy` is `−Σ J s s − Σ h s`, and `kernel::p_up` is
  `sigmoid(2 β field)`.

The THRML™ paper's Appendix C.1 restricts the hardware-natural families to
quadratic energy-based models: Boltzmann machines (main text Eq. 10–11), Potts
models in one-hot form (Appendix C.1.1, Eqs. 65–72), and Gaussian-Bernoulli
energy-based models (Appendix C.1.2). Potts conditionals are a softmax of a
linear function of the neighbours. THRML™ can express a Potts model as a
`CategoricalEBMFactor` whose weight tensor is diagonal in the two state indices.
The codon walkthrough on the docs site is that construction, at gene scale, plus
a domain-wall encoding down to Ising. The encoding lives in application code
(`codon_opt`), not in `thrml/__init__.py`.

TSP §IV.4's target is a three-body Ising energy (Eq. 40) that the Z1 substrate
cannot host: Z1 is pairwise, bipartite, and capped (Eq. 39, §II.2). The paper
compiles each single-site conditional into a pairwise kernel with hidden spins
(Eq. 42, Appendix J.2). On a fully connected substrate with unbounded couplings
the compilation is exact. With a cap `|J| ≤ J_max` it is not.

ferrotherm's model surface is wider in a different direction and narrower in
another.

- Pairwise Ising graphs (`ising`, `graph`) use the same `±1` energy as
  `IsingEBM` with `β` held outside the couplings, on the sampler.
- Potts and clock models (`potts`) use labels in `0 .. q−1`, energy
  `−Σ J f(s_i, s_j) − Σ h`, and refuse a state outside the label set rather
  than reducing it modulo `q`.
- Higher-order spin polynomials (`hubo`) use `E = −Σ_T w_T Π s_i`, the same
  sign as `Graph::energy`, solved and (in the benchmark harness) heat-bathed
  without ancillas. `reduce` introduces ancillas when the target device is
  pairwise. Those are different problems. THRML™ does not need the reduction
  for sampling, because a k-body factor is native on the CPU/GPU simulator.
  TSP's point is that the *hardware* is pairwise, so the k-body target has to
  become a kernel on that hardware. ferrotherm has both passes and names which
  one is running.
- The modelling layer (`model`) takes named variables with domains
  (categorical, integer, binary, spin), constraints, and an objective, and
  returns values in the modeller's own units. THRML™ has no equivalent. Torx
  names circuit sites (`DFG` / `Site` in the TSP §II.3 sense), which is a
  different object: a wire in a program, not a decision variable with a domain.
- `program::Gate` includes `PNot` (a bit flip with a sigmoid probability),
  linear-Gaussian control noise, and a chromatic Gibbs gate `GibbsK`. TSP §II.4
  names `PNOT(p)` as an elementary parametrized-stochastic-circuit gate. The
  two gates are the same mathematical object, written independently, in
  different programs.

## Samplers

TSP §II.1 defines a Gibbs update and a chromatic block update. A proper
colouring of the hypergraph makes same-colour sites conditionally independent,
so one block draw resamples the colour class from its exact conditional. One
Gibbs iteration on Z1 is the two colour classes of the bipartite lattice
(§II.2, Eq. 10). Appendix J.1 records the schedules a meta-EBM may run:
systematic scan, random scan, or chromatic scan. Single-site Gibbs is the
schedule of singletons. Each exact block kernel leaves the target invariant;
a composition of them does too.

THRML™ implements that schedule as data.

- A `Block` is an ordered list of nodes resampled together
  (`thrml/block_management.py`).
- A `BlockGibbsSpec` takes free superblocks and clamped blocks
  (`thrml/block_sampling.py`). Nodes in one superblock are updated from the
  same global state. The caller supplies the blocks. Nothing in `thrml/`
  computes a colouring. The docs tell the reader to divide the model into
  blocks via graph colouring; the examples' optional extra pulls in NetworkX
  for that, and that extra is not a library dependency.
- `SamplingSchedule(n_warmup, n_samples, steps_per_sample)`: one step is
  `sample_blocks`, which visits every free block once (`sample_blocks`'s
  docstring: "one iteration of sampling, visiting every block"). The state at
  the end of warmup is sample 0. Later samples are separated by
  `steps_per_sample` steps (`sample_with_observation`).
- Clamped blocks are first-class. That is the mechanism TSP §II.1 uses to turn
  an energy into a kernel.
- `hinton_init` draws each spin from `sigmoid(β h_i)`, independently
  (`ising.py`, citing Hinton 2012).

ferrotherm's Ising sampler (`gibbs::Sampler`) is chromatic block-Gibbs: one
sweep updates every free node once, colour class by colour class, and the
colouring is computed when the graph is built (greedy, then a 2-colouring when
greedy spent three or more colours, then DSATUR when that spends fewer).
`Sampler::collect` burns in, thins, and charges the ledger one read per site
per kept state. Potts sampling (`potts::Sampler`) is a systematic index-order
heat bath or Metropolis sweep, not a caller-supplied colouring. The benchmark
tells THRML™ to use one node per block in that same index order, so the Potts
comparison is two systematic scans. The Ising and three-body comparisons pass
the same colour classes to both sides.

One schedule detail is easy to get wrong and is pinned in the runner. THRML™
keeps the post-warmup state. `Sampler::collect` discards the post-burn-in state
and keeps a state only after a further `thin` sweeps. The Python runner sets
`n_warmup = warmup + thin` so the first kept state and the spacing match.

Neither library's random stream is the other's. Equal seeds name equal
experiments. They do not name equal trajectories. The comparison is each chain
against the exact law, scored by one function.

`hubo::anneal` is simulated annealing toward a minimum. It is not an
equilibrium sampler. The harness heat-baths a `Hubo` with `Hubo::delta`, which
is the exact single-spin energy gap, and says so in the trace. That is the
ferrotherm-side sampler for the TSP §IV.4 target energy. It is not Figure 12.
Figure 12 is the bias of a *compiled* chain.

## Compilation

TSP §III is the compilation theory.

- Per-kernel conditional KL `J_ℓ(x)`, averaged under the target input marginal,
  sums to the trajectory KL, which upper-bounds any readout KL (Eq. 17,
  proved in Appendix C). A generic training input inflates each term by
  `c_ℓ = max q/μ`.
- The gradient cases are enumerated in Appendix D: exact conditional,
  conditional sampler, or trajectory data, crossed with a tractable or an
  intractable kernel. The intractable case is conditional contrastive
  divergence (Eq. 23).
- Context matching re-optimises a kernel under target inputs or under the
  inputs the compiled circuit actually feeds it (§III.2). The random-walk
  demonstration (§IV.1) is the existence proof that training under the wrong
  inputs leaks probability mass.
- Pre-compiled libraries (§III.3) are kernels trained under a generic input,
  repaired by context matching after placement. The paper leaves the choice of
  library as future work.
- Trajectory REINFORCE (§III.4, Appendix I) trains every kernel against one
  scalar on the whole trajectory. The score uses the spin products `Φ = −∇E`,
  one extra reference sample per factor, and a baseline that is not optional
  (Appendix I.3). It is applied after per-kernel compilation because the
  score-function gradient has high variance until samples land where the reward
  varies.
- The meta-EBM (§IV.4, Appendix J) compiles the Gibbs conditionals of a
  non-native energy. On the published d = 12 instance the stationary total
  variation tracks `ε̄ / (1 − ρ_0)` from about 0.46 at `J_max = 0.3` down to
  `2.4×10⁻²` at `J_max = 10`, with `ρ_0 = 0.28` and the measured floor near
  `0.6×` the bound (Appendix J.3 gives `ε̄ = 0.5052` and the floor `0.70215`
  at the worst cap). The Z1 topology residual is analysed and then left out of
  that sweep on purpose.

ferrotherm's `compile` fits one device-native conditional,
`P̃(y | x) = Σ_h exp(−β E) / Z(x)`, by exact cross-entropy on kernels small
enough to enumerate, and states the same chain-rule shape as Eq. 17 in the
module docs, cited there as prior art. `factor_eps` is the per-factor
conditional KL averaged under a caller-supplied `μ`. That is the middle term of
Eq. 17 for a single factor. It is not context matching, it is not a library of
pre-compiled gates, and it is not trajectory REINFORCE on a composed program.
`program` has its own score-function estimator, `E[(L − baseline) · score]`,
on gates whose log-density is known, including an exact Boltzmann resample
whose score is `β (s_i − ⟨s_i⟩)`. Appendix I's estimator is the energy-gradient
form for a thermodynamic kernel whose density is not evaluated. Both are
REINFORCE. They are not the same estimator, and ferrotherm does not claim the
TSP experiments.

The Gaussian design loop (TSP §IV.5, Appendix K) compiles a hierarchical
Gaussian Markov random field by reading the precision and the interpolation
operators into a pairwise energy, minor-embeds onto Z1, and runs uncertainty
sampling. Reported posterior RMSE over 24 fields and 30 measurements:
`0.259 ± 0.005` for the compiled model, `0.260 ± 0.006` for the exact posterior
loop, `0.305 ± 0.008` for random placement. Top-cell agreement with the exact
loop is 35%, top-three 63%. ferrotherm's thermodynamic linear algebra
(`tla`) solves symmetric positive-definite systems by an Ornstein-Uhlenbeck
network and checks them against elimination. That is a different algorithm for
a related continuous quadratic energy. It is not this design loop, and this
comparison does not run the design loop: the embedding is thousands of p-bits
and the target is a Gaussian, not one of the enumerable discrete laws below.

## Hardware assumptions

TSP §II.2 specifies the substrate the demonstrations compile toward, called Z1,
and points at arXiv:2510.23972 for the implementation. The lattice is a 2D grid
with connection rules `(1,0)`, `(2,1)`, `(2,3)`, `(4,1)`, degree 16 in the
bulk, longest edge `√17` (§II.2.1). It is 2-colourable. Couplings may be
asymmetric, but the rest of the paper restricts itself to the symmetric case
because asymmetric Gibbs has a stationary law with no known closed form and no
known training rule (§II.2, after Eq. 12). The chip description in §II.2 is
about 250,000 nodes, a Gibbs iteration estimated at about `3×10⁻¹⁰ J`, and
`10⁶` to `10⁷` iterations per second. Readout is estimated at `10²`–`10³`
iterations of energy. Coupling flashing is more expensive than readout and is
assumed at most about once per second (§II.2.2).

Table 4, which sits in the HTML between the glossary and Appendix C and is the
table ferrotherm's ledger calls Table IV, gives a per-p-bit breakdown at
50 MHz: sampling `≥ 7.09 fJ`, read `1.692 pJ`, write `153.6 pJ`. The prose
above the table says the breakdown "updates the high-level estimates reported
in [45] by incorporating … SPICE-based energy estimates." A later sentence says
"retaining the measured Z1 sampling and read/write energies per node." Both
sentences are in the paper. ferrotherm stores the three numbers as `Z1_SPICE`
with evidence grade `Simulated`, source string naming the table, and does not
treat them as a meter reading. Appendix K.2 of TSP, on the Gaussian loop, says
outright that the device-side number "charges energy to the sweeps alone and
carries no readout or idle term," and that the GPU side is "converted
wall-clock" at board power. Section V asks the reader to keep simulator
results, calibrated projections, and physical measurements distinct. This
comparison follows that instruction.

The THRML™ paper's own cell-energy estimate is different and earlier:
`E_cell ≈ 2 fJ` in §III, and about `1.6 T nJ` for the Fashion-MNIST DTM in
Appendix E.4 (`N = 4900`, `K = 250`, `G_12` connectivity). TSP's Table 4 is the
refined Z1 account and is larger (the appendix computes a factor of about 9.8
on the same DTM workload, `15.6 T nJ` against `1.6 T nJ`). Quoting either
number as a measured joule on a GPU host is a category error. The benchmark
harness has no parameter that accepts them.

ferrotherm's device models (`device`, including `z1_grid`) describe topologies
and, separately, price tables with an evidence grade. A run on a CPU is priced
only by a meter on that CPU, or it is not priced. `duty` refuses a comparison
against a device whose standby power was never published. Extropic's Table 4
has no standby row. That gap is unchanged by this note.

## Correctness guarantees

TSP's guarantees are about compilation error, and they are theorems plus exact
finite checks.

- Eq. 17 is an equality for the trajectory KL under the target input marginals,
  and an inequality for a readout and for a mismatched training measure.
- Markov kernels are non-expansive in total variation. A contracting kernel
  washes an early residual out at a geometric rate; the deployed error then
  sits near `ε̄ / (1 − ρ)` rather than growing with depth (§III.1, Appendix G.2,
  Eq. 43). The d = 12 meta-EBM checks this by enumerating all `2¹²` states
  (Appendix J.3: "All quantities entering Fig. 12 are computed exactly").
- The paper is explicit about what it does not guarantee. Section V: every
  compiled program remains approximate, because of finite coupling range,
  sparse topology, optimisation residual, and finite thermalisation. Path-space
  KL is called a conservative diagnostic. Task metrics still have to be checked
  per application. The market simulator (§IV.3, Table 2) has no target
  conditional to compare, so it reports stylised-fact errors on held-out
  rollouts. REINFORCE there is judged by transfer from 20-day training
  rollouts to 1,200-day held-out rollouts, and the paper says the reward was
  chosen to parallel the scored terms.

THRML™ 0.1.4's guarantees are tests on tiny enumerable models, not a
certificate API.

- `tests/test_ising.py` and `tests/test_discrete_ebm.py` draw on the order of
  `10⁴` samples and assert that the maximum bin-wise absolute error, divided
  by the maximum exact probability, is `< 0.02`. The Ising test clamps one
  spin of a 5-node line. The categorical test includes a bias, a pairwise
  tensor, and one triple. That is a real check against an exact law. It is a
  relative max-bin error, not total variation, not KL, and not an error bar.
- A search of the 0.1.4 tree finds no autocorrelation, effective-sample-size,
  R-hat, total-variation, or joule-counter symbol in library code. (The paper's
  Appendix G defines autocorrelation and mixing time for the DTM experiments.
  That definition is not a function in the 0.1.4 package. A third-party DTM
  replication the docs link to has its own autocorrelation helper. It is not
  part of THRML™.)
- `estimate_kl_grad` estimates the contrastive-divergence gradient for Ising
  training. It is a training utility. It does not score a sampler.

ferrotherm's guarantees are part of the library, and the tests are required to
be able to fail.

- `ising::exact_boltzmann` and `potts::enumerate` are the oracles, with the
  log-weight shift that keeps `exp` finite.
- `certify::tau_estimate` is Geyer's initial monotone sequence, cross-checked
  against long-batch means, floored at `1/2`. `potts::estimate` reports
  `ESS = N / (2 τ)` and a standard error `sqrt(var / ESS)`. The crate's
  calibration example shows a `sqrt(var / N)` interval covering about 24% of
  draws while announcing 95%. The benchmark uses this estimator on both
  chains, so a gap in `τ` is a gap in the samples.
- A sample set carries the distribution it came from. An expectation is refused
  when the states are not from that distribution.
- The compiler's exact gradients are checked against finite differences on
  enumerable kernels.
- Kani harnesses, run by `scripts/check-proofs.sh`, cover the arithmetic a
  capacity or an index formula stands on.

What ferrotherm does not have, and should not be described as having: a
published check of TSP's Eq. 43 on their d = 12 draw (the draw seed is not in
Appendix J.3), a market-simulator evaluation, or a proof of their contraction
bound. The bound is theirs. The citation in `compile`'s module docs is a
citation.

## Energy

The discipline this repository copies from the ML.ENERGY Zeus work (NSDI 2023,
and the later counter coverage described at <https://ml.energy/zeus/>) is:

- read a hardware energy counter, not a datasheet;
- subtract an idle baseline taken on a quiet machine;
- report gross joules and incremental joules as different numbers;
- repeat, and put an interval on the repeats;
- keep the raw trace.

Zeus reads NVIDIA® GPU energy through the NVIDIA Management Library and CPU
energy through Intel® RAPL. ferrotherm-meter already reads RAPL from
`/sys/class/powercap` (world-readable `energy_uj`, no elevated privileges),
rejects a `psys` domain that fails to exceed the package inside it, reads
INA3221 board rails without summing nested channels, and on macOS shells out
only to `macmon pipe`, which reads Apple's IOReport counters without root
(<https://github.com/vladkens/macmon>). `powermetrics` needs root. The harness
does not call it. The harness adds the NVIDIA Management Library total-energy
counter (`nvmlDeviceGetTotalEnergyConsumption`, millijoules) by `dlopen` of
`libnvidia-ml.so.1`, summed across devices and labelled as a GPU sum, not as
whole-machine energy.

The quiet-host gate is unchanged: a 1-minute load average above 2 is not quiet.
The harness waits and retries for at most thirty seconds, then records
`load_gated` and takes no energy figure, instead of omitting the field.
`--contended` is a separate mode. It still subtracts an idle baseline and
reports gross joules, incremental joules, joules per effective sample, and 95%
intervals, with `load1` and other-process CPU on every trial. Every number from
that mode is evidence grade `contended host`. It is not quiet-host evidence and
it is not pooled with a quiet run.

RAPL on hostname `framework-128gb-0000` is the primary energy evidence. That
machine reads package `energy_uj` without root. macmon is secondary and its
scope is SoC telemetry (`sys_power`), not wall power. Package joules, core
joules, and SoC-telemetry joules are not added together, and a summary that
mixed them is refused.

An earlier fleet attempt on 7 November 2026 obtained no joules. The machine
that answered was `framework-32gb-0001`, not `framework-128gb-0000`. On
`framework-32gb-0001`, RAPL `energy_uj` was permission denied and `perf`
`power/energy-pkg` failed at `perf_event_paranoid` 4. The M5 Max 48 GB run of
that attempt had rootless macmon traces and a load average of 6.45, so the
quiet gate refused a baseline. Those traces were not turned into joules.

The first measured-joules bundle is `bench/thrml-tsp/results/fleet/189502d/`,
revision `189502d`, both hosts evidence grade `contended host`.
`framework-128gb-0000` measured RAPL package energy. The M5 Max measured macmon
SoC telemetry (`sys_power`), not wall power. Quiet mode on the Framework host stayed above load 2 and
recorded `load_gated`. The ferrotherm windows in that bundle are about 1 ms,
below the RAPL update cadence. One trial records gross 0 J, a negative
incremental, and `attributable: true`. Those values stay in the bundle. The
harness now repeats each sampler for at least 2 s, divides by the repetition
count, and refuses attribution when the counter advanced fewer than 8 times.
Summaries are per workload. The two scopes stay in separate tables.

A delta inside three standard deviations of the baseline's own wander is kept
in the trace and is not divided into a per-sample price. So is a window whose
counter did not keep updating. Absence of a counter is a null joule with a
reason. It is not filled from `Z1_SPICE`, from Table 4, or from Appendix E.

## Where each one is stronger

**THRML™ is stronger** as a GPU array implementation of block Gibbs on
heterogeneous discrete factors. One `DiscreteEBMFactor` batches an arbitrary
order of spin and categorical interactions. Node state is a PyTree, so a new
node type is a dtype plus a conditional, not a new sampler. The Ising training
helpers (positive phase, negative phase, moment accumulator) sit next to the
sampler. The docs describe a spin-model notebook aimed at block-Gibbs
throughput on eight B200 GPUs. That throughput is their report. It was not
re-measured here. The library is the natural place to prototype a program whose
blocks, clamps, and factors already match the TSP kernel.

**THRML™ is weaker** as a modelling layer, as a certificate, and as an energy
account. Variables are anonymous. Colouring is not computed. Temperature on an
`IsingEBM` is baked into the factors. There is no effective-sample-size, no
error bar, and no joule. The distribution tests are real and they are a single
relative max-bin threshold on one or two schedules. A caller who wants `τ` writes
it themselves. The package is Python and JAX only.

**The TSP paper is stronger** as a theory of compiling a directed stochastic
program onto an equilibrium device, and as a set of worked error budgets.
Eq. 17, the input-distribution factor `c_ℓ`, context matching, the REINFORCE
post-pass, and the contraction floor are a coherent account, and §IV.4 checks
the floor against an enumeration. The demonstrations cover a conserved random
walk, a birth-death process, a market simulator with held-out stylised facts
(Table 2: composite error 0.667 after variational compilation, 0.486 after
REINFORCE, against baselines at 2.961 and 1.651), a non-native three-body
sampler, and a Gaussian experimental-design loop whose RMSE matches the exact
loop. Section V is unusually careful about what those demonstrations are not.

**The TSP paper is weaker** as something a third party can run. thermalizers is
not in the public organisation. The d = 12 couplings are specified in
distribution and not by seed, so Figure 12 cannot be regenerated bit for bit
from the paper. The hardware joules are a SPICE and cost-model projection, and
the paper's own Appendix K flags a missing readout term and a missing idle
term. Asymmetric couplings are left untrained on purpose. Pre-compiled kernel
libraries are left as future work.

**ferrotherm is stronger** where a number has to survive contact with an exact
law or with a counter. Enumerable Boltzmann laws, Potts laws, and (in the
harness) three-body laws are scored with total variation, both KLs, and an
autocorrelation estimator whose failure modes are documented in the crate.
The ledger distinguishes simulated, projected, and metered evidence, and the
meter refuses the failure modes that have already produced a flattering zero in
this repository: a dead `psys` domain, a nested INA3221 sum, a negative delta
clamped to zero, a baseline taken under foreign load. The same model is
reachable from Rust, the C ABI, Python, Zig, Julia, HTTP, MCP, and the node
editor, with parity gates that fail when a surface drops a symbol. There are
no dependencies. A seed fixes the stream on every platform the crate claims.

**ferrotherm is weaker** as a large-scale array sampler and as a compiled
stochastic-program stack. It is not JAX, and its GPU backend is a separate
wgpu crate with a different execution model from THRML™'s XLA block program.
Potts sampling is a fixed systematic scan. There is no market simulator, no
codon-optimisation application, and no implementation of TSP's context-matching
loop or of Appendix I's energy-gradient estimator on a composed thermodynamic
kernel. The chain-rule bound is cited. It is not re-proved, and Eq. 43 is not
re-measured on their unpublished draw. Nobody here has metered a Z1.

## What the benchmark measures

`bench/thrml-tsp` pins four workloads and scores both runners with
`potts::estimate` and `ising::tv`.

| Id | Law | Why it is here | Oracle |
| --- | --- | --- | --- |
| `thrml_readme_ising_chain_n5` | Open chain, `n = 5`, `J = 0.5`, `h = 0`, `β = 1`, blocks `(0,2,4)` then `(1,3)` | The model in the THRML™ 0.1.4 README and the docs quickstart | `2^5` enumeration |
| `ising_grid2d_open_3x3` | Open 3×3 grid, `J = 0.5`, `β = 0.8` (`βJ = 0.4`), 12 edges | A finite 2D Ising both codebases can build. 4×4 is enumerable and still too large for a histogram of a few thousand draws (total variation stays near 1 for a correct sampler). Onsager's critical value is for the infinite lattice and is not the oracle | `2^9` enumeration |
| `potts_path_q3_n6` | Path, `q = 3`, `J = 0.4`, `β = 1`, index-order heat bath | The Potts interaction of arXiv:2510.23972 Appendix C.1.1 and the codon walkthrough, at `3^6 = 729` states. The gene-scale problem is not enumerable | `potts::enumerate` |
| `tsp_iv4_three_body_d8` | `d = 8`, 12 pairs, 10 triples, fields and couplings i.i.d. `N(0, 0.6²)`, `β = 1` | TSP §IV.4 and Appendix J.3's weight law. Their d = 12 is what makes the full chain law enumerable for Figure 12; a histogram cannot resolve 4096 states at this draw count, so the pin is d = 8 (256 states). Seed `Pcg::new(260801615, 0x54535031)` replaces the unpublished draw | `2^8` enumeration of the three-body energy |

The three-body case is native chromatic Gibbs of the target energy on both
sides. It is not the compiled pairwise kernel, and it does not reproduce
Figure 12's total-variation floor. Doing so would require thermalizers, or a
reimplementation of Appendix J.2's constructive compilation, which is a
different experiment and is not claimed.

Reported for each trial: total variation; `KL(empirical ‖ exact)`;
`KL(exact ‖ empirical)` or null when the chain missed a state, with the miss
count; `τ` and ESS on the energy trace; ESS per second of the timed call,
burn-in included, JIT excluded and reported separately for THRML™; absolute
error of the mean energy; mean absolute error of the per-site magnetisation
for spin models. Trials are summarised with a Student-t 95% interval. Two
trials is the minimum the binary accepts, because one trial has no interval.

Joules, when a counter exists, come from that counter. Per-trial chains of a
few thousand sweeps on at most 16 sites finish in milliseconds, under
ferrotherm-meter's minimum window, so a separate repeat of at least two
seconds is what the meter is asked about, and it is labelled as that repeat
rather than divided into the scored ESS. On a host with no counter the field
is null and the probe says which backends were absent.

Versions locked for the external runner, resolved together on 7 November 2026
and smoke-tested on the README chain: THRML™ 0.1.4, JAX 0.12.0.1 (CPU extra),
jaxlib 0.12.0.1, equinox 0.13.8. THRML™ does not pin JAX. The pin is the
resolution this harness ran. `bench/thrml-tsp/python/requirements.txt` records it.

## Gaps

- This harness does not run on eight B200s, on an H100, on Apple silicon, or
  on a Z1. A VM without `/sys/class/powercap` and without `libnvidia-ml.so.1`
  produces accuracy and wall-clock only. The report says which machine that was.
- BMI fleet hardware is what would supply a quiet RAPL host, an NVIDIA® GPU
  with the management library loaded, and a macOS host with `macmon` on `PATH`,
  each as its own instrument. Those three counters must not be substituted for
  each other, and none of them meters Extropic silicon.
- Figure 12, Table 2, the codon-optimisation energy projection, and the
  Fashion-MNIST 10,000× estimate are not results of this harness. They are
  cited above as the authors' results, with the grade the authors' own caveats
  support.
- Context matching, trajectory REINFORCE on a compiled kernel, and the Gaussian
  design loop are described here and not executed. Executing them faithfully is
  a compiler project, gated on either a thermalizers release or a line-by-line
  reimplementation of Appendices H, I, J, and K, checked against the enumerated
  d = 12 floor before any larger claim.
