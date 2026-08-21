# Where this stack sits, layer by layer

## Re-survey, 2026-08-16 — and a defect in how the first one was run

**The original survey ran on Python 3.9.6**, the interpreter macOS ships. Three of the packages it
assessed could not install their current versions there: `jijmodeling 2.7.1` needs >=3.11,
`amplify 1.6.2` and `ommx 2.6.2` need >=3.10. So pip resolved the newest 3.9-compatible release and
the survey recorded those as the state of the art. They were not. An outdated interpreter turns
into a wrong competitive assessment, quietly, and every "surveyed from a live install" line below
inherits that caveat until re-checked.

Re-run on Python 3.13, 8 of 8 spot-checked packages had moved, one by a major version:

| | surveyed | actual |
|---|---|---|
| jijmodeling | 1.14.2 | **2.7.1** |
| amplify | 1.3.1 | 1.6.2 |
| ommx | 2.0.12 | 2.6.2 |
| openjij | 0.11.6 | 0.12.1 |
| dwave-ocean-sdk | 9.0.0 | 9.4.0 |
| dimod / minorminer / dwave-system | 0.12.21 / 0.2.19 / 1.33.0 | 0.12.22 / 0.2.22 / 1.35.0 |

### Two things this changes

**1. jijmodeling 2.x detects constraint patterns; ferrotherm does not.** `ConstraintDetectionConfig`
and `ConstraintHintName` (`OneHot`, `Sos1`) let it RECOGNISE that a set of constraints forms a
one-hot pattern and hint the solver accordingly. Ferrotherm requires the modeller to say
`exactly_one`. **Corrected 2026-08-16 by measuring rather than inferring:** the gap is far smaller
than it first looked. `cardinality(lits, 1)` and `exactly_one` compile to *identical* graphs here
(10 spins, 15 factors on five literals), and so do six pairwise `not_equal`s and one `all_different`
(16 spins, 48 factors). The only longhand form that measures more expensive is `at_most(lits, 1)` —
12 spins and 26 factors against `at_most_one`'s 10 and 15, because an inequality needs a slack
variable. Ferrotherm now detects that one, plus constraints that constrain nothing, and **reports
them as caveats rather than rewriting the model**: silently compiling something other than what was
written is the opposite of this compiler's discipline. The remaining difference is that Jij hints a
*solver* while we advise a *modeller*.

**2. Amplify ships 16 vendor clients; ferrotherm declares 7 fabrics.** D-Wave (three), Fujitsu DA3c
and DA4, Gurobi, Hitachi, NEC VA2, Toshiba SQBM2, Fixstars AE, plus a `CustomClientProtocol` for
user backends. That is broader vendor reach than ours (`dwave_advantage`, `dwave_advantage2`,
`fujitsu_da3`, `toshiba_sqbm`, `toshiba_sqbm_pubo`, `qboson_cpqc`, `unconstrained`, plus CPU, GPU
and the Hitachi driver). **They win on breadth.**

What they do not have, checked by introspection rather than assumed: no energy, joule, watt or
power surface anywhere in `amplify`, and no certificate or sampling-fidelity surface. Those rows
stand.

### The sub-lane inside layer 10 that is empty for EVERYONE, this project included until 2026-08-20

Layer 10 (energy / cost accounting) is marked **not found** for every group surveyed here, and this
project has been claiming it. That claim was too generous to itself. Every energy figure in this
tree — and every one anywhere in the field — is **joules above idle, divided by work done**. That
prices a machine kept busy, and a sampling substrate's best case is the opposite: intermittent,
low-duty edge work where the machine spends most of its life waiting.

Price the wait and the argument inverts into a single hardware number:

```text
standby budget = idle + marginal × duty
```

which is what a challenger must come in under, **granting it perfectly free computation** — so no
better sampler can argue it down. As the cadence slackens it collapses onto the incumbent's idle
draw, and the entire case for a thermodynamic fabric reduces to its own standby power.

**No thermodynamic vendor publishes a standby figure.** Extropic's Table IV states e_sample, e_read,
e_write and a reflash cap; there is no standby row. Neither is there one in any other device model
this survey located. So the comparison that would actually decide the field's strongest argument
cannot be completed by anybody today — not by us, and not by them, and it is one number away from
being decidable.

That is where `src/duty.rs` and `Machine::beaten_by_device` sit: the arithmetic is public and takes
the vendor's number as an argument, so the party that knows it can finish the sum. `DeviceRun`
carries `standby_watts: Option<f64>` and the comparison REFUSES on `None` rather than defaulting it
to zero, which is the difference between exposing the gap and quietly stepping over it. Applied to
Z1_SPICE run model-resident at a sustainable cadence — every per-operation energy published, the
computation priced at 1.77e-8 J per period — the verdict is `StandbyUnpublished`. Whether this project
can fill in its OWN side of the table is a measurement question, and as of this note it is pending a
quiet machine — the first attempt was refused by `Meter::idle`'s load guard.

### And a strategic fact

jijmodeling 2.x's `Compiler` converts problems into **OMMX** instances, and it round-trips through
protobuf. Jij has committed to OMMX as the shared IR. The field converging on a program IR is the
context `.ftp` sits in, and it is worth deciding deliberately whether `.ftp` should read and write
OMMX rather than only its own text.


Surveyed 2026-08-14 by installing each stack and interrogating the shipped API — not by
reading marketing. Every claim below carries the source the surveying agent used. Six stacks:
Extropic (THRML + Torx), D-Wave Ocean 9.0.0, the Japanese cluster (Amplify, Jij, PyQUBO,
OpenJij), the open-source layer, the hardware vendors' own SDKs (Fujitsu, Toshiba, QBoson,
Hitachi, NEC), and ferrotherm itself, assessed harshly on purpose.

This file is a snapshot. It will go stale; the date above is the only thing that makes it
useful later.

## Extropic — THRML (v0.1.4, Apache-2.0) + Torx (extro-torx v0.0.1, Apache-2.0), both pure-Python/JAX GPU simulators; Thermalizers (the compilation layer that would bridge them to hardware) is a paper only, no code released. Verified 2026-08-14 by cloning github.com/extropic-ai/{thrml,torx,codon_opt,thrml-skill} and reading source; org has exactly 4 repos (GitHub API).

### 1. Modelling layer (named variables, domains, constraints, objective, answers by name) — **not found**

THRML has no modelling layer above raw graphs. A variable is an anonymous object: `SpinNode()`
and `CategoricalNode()` are empty classes whose bodies are literally `pass`, identified only by
an auto-incrementing integer from `_UniqueID._counter` (thrml/pgm.py:30-56, 73-81). There is no
name, no user-facing identifier, and no objective/constraint declaration. Crucially,
CategoricalNode does NOT carry its own K: the docstring says 'an integer in [0,K)' but K is
stored nowhere on the node — it is implicit in the shape of the weight tensor passed to
CategoricalEBMFactor ([b, x_1..x_N], thrml/models/discrete_ebm.py:115-118). Answers come back
positionally, indexed by `Block` order, never by name — you rebuild the mapping yourself. Torx
is one step better but for a different object: `DFG`/`Site` carry `sites_by_name` and `PortSpec`
(torx/dfg.py:105-122, torx/factor.py), naming circuit sites, not decision variables with
domains. Nearest thing to a domain in THRML is `DEFAULT_NODE_SHAPE_DTYPES` mapping
SpinNode->bool scalar, CategoricalNode->uint8 scalar (thrml/pgm.py:84-87).

> https://github.com/extropic-ai/thrml/blob/main/thrml/pgm.py (lines 30-87); https://github.com/extropic-ai/torx/blob/main/torx/dfg.py

### 2. Encodings (one-hot, domain-wall, binary/log; is the choice exposed?) — partial

Domain-wall encoding EXISTS but as hand-written application code, never as a library API — the
choice is a code fork you author, not a parameter you set. Implementation:
`codon_opt/ising/dwc.py` (194 lines: `compute_spin_layout`, `compute_ising_biases`; Potts K
states -> K-1 thermometer spins, unary bias becomes a first difference across adjacent states,
pairwise coupling a second difference) and re-derived inline as `compile_dwc` in
thrml/examples/03_codon_optimization.ipynb, which tells the reader to 'treat compile_dwc below
as a black box'. One-hot: NOT FOUND, and explicitly rejected — the same notebook argues 'domain-
wall encoding keeps the constraint graph a sparse chain (rather than a dense clique) and mixes
well even when P is large'. Binary/log encoding: not found anywhere. Ragged domains are handled
by a manual masking hack, not an encoding: pad every categorical to K_MAX=6 and give invalid
states `INVALID_BIAS` = -1e10 so softmax never selects them. THRML's own library exports zero
encoding symbols (thrml/__init__.py, 33 lines).

> https://github.com/extropic-ai/codon_opt/blob/main/ising/dwc.py; https://github.com/extropic-ai/thrml/blob/main/examples/03_codon_optimization.ipynb

### 3. Higher-order reduction (k-body -> pairwise, ancillas) — **not found**

No reduction machinery, no ancilla generation — but note the reason is architectural, not a
plain gap: THRML is 'hypergraphical' and k-body factors are FIRST CLASS, so nothing needs
lowering at this level. `DiscreteEBMFactor.__init__(spin_node_groups, categorical_node_groups,
weights)` accepts an arbitrary-length list of node groups and asserts `len(weights.shape) == 1 +
len(categorical_node_groups)`, i.e. a rank-(N+1) weight tensor for an N-body factor
(thrml/models/discrete_ebm.py:57-118). Torx likewise composes arbitrary-arity factors
(torx/factor.py, torx/composite_factors.py). The gap is that Extropic's own silicon is pairwise
p-bit Ising, so the k-body -> 2-body lowering must happen somewhere, and that somewhere is the
unreleased Thermalizers layer: arXiv:2608.01615 'Thermalizing Stochastic Programs' compiles
Directed Factor Graphs to hardware-native EBMs by per-factor variational fitting. No code. The
only k->2 lowering in public code is the bespoke Potts->Ising domain-wall compile in
codon_opt/ising/dwc.py, which is not general.

> https://github.com/extropic-ai/thrml/blob/main/thrml/models/discrete_ebm.py (lines 57-118); https://arxiv.org/abs/2608.01615

### 4. Constraint vocabulary (equality, inequality/slack, cardinality, exactly-one, all-different) — **not found**

NOT FOUND — none of the five. A case-insensitive grep for
constraint|penalt|slack|cardinal|all.differ|exactly.one|feasib across thrml/ and torx/ library
source returns ZERO hits (the only near-hits are the word 'feasible' in an unrelated comment at
thrml/block_sampling.py:37 and 'extract'/'Slack' substring matches). There is no equality, no
inequality, no slack-variable generator, no cardinality or exactly-one helper, no all-different.
Where the word 'constraint' does appear it is user-authored example code meaning 'ferromagnetic
chain edges I built by hand': `constraint_edges = [(offset[p]+j, offset[p]+j+1) for p in
range(L) for j in range(Ks[p]-2)]` in the codon notebook. The user writes the energy function;
the library only samples it.

> grep over https://github.com/extropic-ai/thrml/tree/main/thrml and https://github.com/extropic-ai/torx/tree/main/torx; example usage https://github.com/extropic-ai/thrml/blob/main/examples/03_codon_optimization.ipynb

### 5. Penalty handling (auto scaling, feasibility check, WHICH constraint broke) — **not found**

NOT FOUND on all three counts. No automatic penalty scaling: the codon notebook anneals
constraint strength P on a second, hand-tuned schedule alongside beta and admits 'these
schedules were chosen empirically and you can tune them'. No feasibility checker: validity of
the thermometer encoding is confirmed by ad-hoc notebook code after the fact. No violated-
constraint reporting of any kind. WARNING on a misleading name: THRML exports
`verify_block_state` (thrml/block_management.py), which sounds like a feasibility check but is a
pytree shape/dtype compatibility assert — its own docstring says it guards against 'unintended
casting/other weird silent errors' and it raises RuntimeError only on 'Number of states not
equal to number of blocks' or 'State shape did not match detected block length'. It knows
nothing about constraints or energies.

> https://github.com/extropic-ai/thrml/blob/main/thrml/block_management.py (verify_block_state); https://github.com/extropic-ai/thrml/blob/main/examples/03_codon_optimization.ipynb

### 6. Embedding / placement onto hardware topology — **not found**

NOT FOUND. No minor-embedding, no chain strength, no ancilla chains, no topology object. Two
things could be mistaken for it and are not: (a) GRAPH COLORING for parallel block updates —
real and necessary, but not in the library either; the user calls networkx, `coloring =
nx.coloring.greedy_color(graph, strategy='DSATUR')` (thrml/examples/02_spin_models.ipynb; also
prescribed in thrml-skill/SKILL.md, which supplies the DSATUR -> Block recipe). A grep for
color|dsatur|partition across thrml/ library source returns one unrelated comment. (b)
`dwave_networkx.pegasus_graph(14)` appears in example 02, but purely as a BENCHMARK GRAPH
GENERATOR — the notebook states the purpose verbatim: 'We will use DWave's Pegasus graph
topology to allow us to directly compare the speed of our GPU-based sampler to results obtained
using other hardware accelerators.' dwave_networkx is an `examples` extra, not a runtime
dependency, and no embedder is imported.

> https://github.com/extropic-ai/thrml/blob/main/examples/02_spin_models.ipynb; https://github.com/extropic-ai/thrml/blob/main/pyproject.toml (optional-dependencies.examples); https://github.com/extropic-ai/thrml-skill/blob/main/SKILL.md

### 7. Samplers / solvers (algorithms, CPU/GPU/hardware) — **yes**

The strongest layer, and genuinely good. THRML: blocked/chromatic Gibbs over sparse
heterogeneous factor graphs — `sample_states`, `sample_blocks`, `sample_single_block`,
`sample_with_observation`, driven by `SamplingSchedule(n_warmup, n_samples, steps_per_sample)`
and `BlockGibbsSpec(free_blocks, clamped_blocks, ...)` (thrml/block_sampling.py, 531 lines).
Per-block conditional kernels are pluggable via `AbstractConditionalSampler`; shipped concretes
are `BernoulliConditional` (spin) and `SoftmaxConditional` (categorical/Potts), specialised as
`SpinGibbsConditional` / `CategoricalGibbsConditional` (thrml/conditional_samplers.py,
thrml/models/discrete_ebm.py). Clamping is first-class (clamped_blocks), which is what makes the
contrastive positive phase work. Simulated annealing is NOT provided — the user rebuilds the
program per beta (codon notebook). Torx adds three simulators behind `AbstractSimulator`:
`StateVectorSimulator` (exact enumeration of the output distribution), `BranchingSimulator`
(sampling), `AffineGaussianSimulator` (Gaussian moments), plus a gate set including
PNOT/PCNOT/PISING/PJUMP/PditCycle/AffineGaussianGate. Fabric: CPU/GPU/TPU via JAX only. NO
hardware execution path exists in either library.

> https://github.com/extropic-ai/thrml/blob/main/thrml/block_sampling.py; https://github.com/extropic-ai/thrml/blob/main/thrml/conditional_samplers.py; https://github.com/extropic-ai/torx/blob/main/torx/psc/simulation/

### 8. Device abstraction (one interface over multiple vendors, capability declaration) — **not found**

NOT FOUND. There is no device object, no vendor interface, and no capability declaration
anywhere in either library. A grep for device|hardware|Z1|XTR|TSU|topolog across thrml/ and
torx/ library source yields only JAX-generic and unrelated matches; the word 'backend' in
torx/psc/simulation/base.py means a NUMERICAL method (state-vector vs branching vs Gaussian),
not a machine. Nothing declares degree, precision, pbit count, coupling range, or reflash cost —
so a program cannot ask what it is running on. THRML's README frames hardware only
aspirationally: 'a natural place to prototype today and experiment with future Extropic
hardware.' The layer that would provide this is Thermalizers, and as of 2026-08-14 it does not
exist as code: the extropic-ai org contains exactly 4 repos (thrml, torx, codon_opt, thrml-skill
— GitHub API), and Extropic's own launch post says Thermalizers is 'whitepaper today', with
'open-source release in the coming weeks'. A login-gated console exists at extropic.dev (HTTP
200, an 831-byte SPA shell titled 'Extropic // Console'); no public API schema, endpoint list,
or auth model was located for it.

> https://api.github.com/orgs/extropic-ai/repos (4 repos, retrieved 2026-08-14); https://extropic.ai/writing/from-one-to-one-billion/; https://github.com/extropic-ai/torx/blob/main/torx/psc/simulation/base.py; https://extropic.dev

### 9. Verification (certificates, effective temperature, ESS, TV vs exact, conformance) — partial

NO library API — not found. A grep for total variation|TV distance|effective
sample|autocorrelat|ESS|R-hat|convergen|certificat across BOTH thrml/ and torx/ library source
returns ZERO hits. What exists is per-notebook hand-written NumPy: torx example 06 computes TV
against the exact Boltzmann law (chromatic Gibbs 0.0634 vs a naive host PISING-matrix walker
0.4020) and asserts a hardcoded threshold — 'inside the 0.08 bound we asserted'; torx example 16
computes TV 0.0222 on the magnetization histogram against exact enumeration. These are good
practice but they are notebook prose, not a callable check, and the bound is a literal a user
typed. Autocorrelation exists only in the THIRD-PARTY dtm-replication
(thrmlDenoising/autocorr_fun.py). No effective temperature, no ESS, no R-hat, no conformance
suite anywhere. THRML's `MomentAccumulatorObserver` accumulates moments for TRAINING gradients,
not for diagnostics. The nearest thing to a certificate in the whole stack is the Thermalizers
paper's analysis of 'how the error of the compiled DFG accumulates from the per-factor errors' —
a mathematical result on paper (arXiv:2608.01615), with no implementation to run.

> https://github.com/extropic-ai/torx/blob/main/examples/06_ising_sampling_contrastive_divergence.ipynb; https://github.com/extropic-ai/torx/blob/main/examples/16_gibbs_sampling_factor_graph.ipynb; https://arxiv.org/abs/2608.01615; https://github.com/pschilliOrange/dtm-replication/blob/main/thrmlDenoising/autocorr_fun.py

### 10. Energy / cost accounting (joules or price per operation) — **not found**

NOT FOUND in either library: zero occurrences of joule|watt|pJ|fJ|nJ|energy_cost in thrml/ or
torx/ source. ('Energy' in THRML always means the EBM energy function E(x), never electrical
energy.) The ONLY executable energy accounting in Extropic's entire public codebase is
codon_opt/qodon/fast_ga/energy_comparison.py, and it is an estimator with no instrument behind
it — three specifics, since this repo's README advertises '~5-9 orders of magnitude less energy
than a GPU': (a) the chip figure is one hardcoded literal, line 299: `potts_chip_energy = 10 *
256 * 1e-10  # 10 steps x 256 chains x 1e-10 J/sweep` — a 1e-10 J/sweep constant with no
provenance cited in the code; (b) the CPU/GPU baselines are TDP divided by peak FP32 TFLOPS
(`spec['j_per_flop'] = spec['tdp_w'] / (spec['tflops']*1e12)`), which the file itself labels
'idealized' and 'generous to the CPU/GPU'; (c) even the section printed as 'SECTION 3: MEASURED
BENCHMARK' measures only WALL TIME and multiplies it by an assumed constant — `system_power_w =
300` for GPU, `100` for CPU. No power meter, no per-operation price, no cost model API. Note
this is the un-licensed repo, not THRML.

> https://github.com/extropic-ai/codon_opt/blob/main/qodon/fast_ga/energy_comparison.py (lines ~275-300, ~355-365)

### 11. Language surfaces (which languages; native vs FFI vs subprocess) — partial

PYTHON ONLY, pure-Python source, no second language and no FFI. THRML: `requires-python =
'>=3.10'`; declared dependencies are just `equinox>=0.11.2` and `jaxtyping>=0.2.23` — note JAX
is NOT declared directly, it arrives transitively via equinox. Torx: `requires-python =
'>=3.11'`, deps `equinox>=0.13.0`, `jax>=0.4.31`, `jaxtyping`, `ihoop[equinox]>=0.1.4`. Both are
pip/uv installable (`pip install thrml`, `pip install extro-torx`). A grep for
pybind|ctypes|cffi|maturin|wasm|grpc across both packages and their pyproject files returns ZERO
hits — there is no C/C++/Rust core, no bindings for other languages, no WASM or browser target,
no CLI or subprocess surface. Everything is one JAX-shaped Python surface. Codebases are small:
THRML library source is 2,515 lines across 11 files; Torx is 6,185 lines.

> https://github.com/extropic-ai/thrml/blob/main/pyproject.toml; https://github.com/extropic-ai/torx/blob/main/pyproject.toml

### 12. Agent / AI surfaces (MCP, HTTP API, structured tool schemas) — partial

One real surface, of an unusual kind, and it post-dates most write-ups: `extropic-ai/thrml-
skill` is a first-party portable AGENT SKILL — a 27,777-byte SKILL.md with YAML frontmatter plus
references/source.md (115,094 B) and references/jax_equinox.md — shipped in the open Agent
Skills format and documented to install into Claude Code (.claude/skills/thrml), opencode, Codex
CLI, and Gemini CLI (.agents/skills/thrml). Its content is prescriptive engineering guidance: a
top-level-imports-only rule, the DSATUR colouring recipe, and the warning that an IsingEBM is a
pytree whose leaves are SpinNode objects so it cannot be passed as a traced argument into a
jitted function. Everything else is NOT FOUND: no MCP server, no HTTP/REST API, no structured
tool schemas, no OpenAPI — grep for fastapi|flask|uvicorn|mcp|grpc across both libraries returns
zero. A login-gated console at extropic.dev is referenced by extropic.ai as an 'API to simulate
workloads running on TSUs' / early-access API, but no public schema or docs for it were located.

> https://github.com/extropic-ai/thrml-skill (README.md, SKILL.md); https://extropic.ai/software; https://extropic.dev

### 13. Training (EBM training, gradient estimators) — **yes**

Real and specific, but narrow. THRML ships contrastive (positive/negative phase) KL-gradient
training for Ising/Boltzmann machines only: `IsingTrainingSpec` (bundles the EBM with separate
positive and negative sampling programs and schedules), `estimate_kl_grad`, `estimate_moments`,
`hinton_init`, backed by `MomentAccumulatorObserver` — all in thrml/models/ising.py (309 lines).
The estimator is the textbook two-term form, documented in the docstring as dW = -beta(<s_i
s_j>_+ - <s_i s_j>_-) and db = -beta(<s_i>_+ - <s_i>_-), with the positive phase data-clamped.
LIMIT: `estimate_kl_grad` is typed to `IsingTrainingSpec`/`IsingEBM` — there is no general
factor-graph training entry point. The optimizer is bring-your-own: optax is a TESTING extra,
not a runtime dependency. Torx is differentiable end-to-end through JAX, with gradient
estimators developed in arXiv:2608.01612 'A Framework for Stochastic Differentiable
Programming', and Thermalizers (paper) adds context matching and trajectory-level REINFORCE
post-training. DTM training itself is NOT in THRML — see notes.

> https://github.com/extropic-ai/thrml/blob/main/thrml/models/ising.py (estimate_kl_grad, IsingTrainingSpec); https://github.com/extropic-ai/thrml/blob/main/pyproject.toml; https://arxiv.org/abs/2608.01612

### 14. Visual / graph programming — **not found**

NOT FOUND. THRML has no visualization module whatsoever. Torx has exactly one, and it is ASCII:
`torx.psc.visualization.draw_circuit` / `TextDrawing` (torx/psc/visualization/text.py), which
renders a gate list as text. There is no node editor, no drag-and-drop graph builder, no GUI, no
browser canvas, no interactive surface of any kind. Graph pictures in the documentation are
matplotlib/networkx figures written by hand in notebook helper files
(examples/helpers/_plots_schematics.py).

> https://github.com/extropic-ai/torx/blob/main/torx/psc/visualization/text.py; https://github.com/extropic-ai/torx/tree/main/examples/helpers

### 15. Licence, openness, hardware requirement — **yes**

HARDWARE IS NOT REQUIRED — and, more strongly, hardware is not even reachable: THRML and Torx
are pure CPU/GPU/TPU simulators with no hardware code path, and Z1 early access is stated as
2027 (Z1 Stick, M.2, 'over half a million pbits'; Z1 Card, PCIe, 'over 4 million pbits').
Everything Extropic has published today runs on a laptop. Licences are mixed and the gaps
matter: THRML Apache-2.0 (LICENSE present, GitHub API spdx Apache-2.0, 1,137 stars); Torx
Apache-2.0 (54 stars, PyPI `extro-torx` v0.0.1). BUT codon_opt has NO licence (GitHub API
license: null) and thrml-skill has NO licence — both are publicly visible but not open-source-
licensed, so the domain-wall compiler and the energy model are look-don't-reuse. The third-party
dtm-replication has no LICENSE file either. Thermalizers is unreleased, so the compilation layer
is closed in practice.

> https://api.github.com/orgs/extropic-ai/repos (retrieved 2026-08-14); https://github.com/extropic-ai/thrml/blob/main/LICENSE; https://github.com/extropic-ai/torx/blob/main/LICENSE; https://extropic.ai/hardware

**Notes.** SHAPE OF THE STACK: three tiers, and only two of them exist as code. Torx (high-level stochastic
programs, DFGs of kernels) -> Thermalizers (variational compilation to hardware-native EBMs) ->
THRML (block Gibbs on the factor graph). The MIDDLE TIER IS MISSING: Thermalizers is
arXiv:2608.01615 only, described by Extropic's own launch post as 'whitepaper today' with open-
source release 'in the coming weeks'; no repo exists (extropic-ai org has exactly 4 repos,
GitHub API 2026-08-14). Everything the survey calls 'not found' — device abstraction, k-body
lowering to pairwise silicon, placement, certificates — is precisely what that unbuilt tier is
supposed to supply.  DTM IS NOT IN THRML. This is a common misattribution worth stating plainly:
THRML ships no denoising-thermodynamic-model code. DTM lives in a THIRD-PARTY repo,
pschilliOrange/dtm-replication (thrmlDenoising/DTM.py, trained on MNIST/Fashion-MNIST with FID
eval, no LICENSE file), which THRML's README and docs link out to as 'a larger project built on
THRML'. THRML's own examples are four notebooks: probabilistic computing, all-of-thrml, spin
models, codon optimization.  LANGUAGE AND DEPENDENCIES, confirmed from pyproject.toml: Python
only, pure-Python source, JAX ecosystem throughout. THRML requires Python >=3.10 and declares
only equinox + jaxtyping (JAX arrives transitively via equinox); Torx requires Python >=3.11 and
declares equinox, jax, jaxtyping, ihoop. No C/C++/Rust core, no FFI, no WASM, no browser, no
CLI. THRML library source is only 2,515 lines across 11 files; Torx is 6,185 lines. THRML is at
v0.1.4, Torx at v0.0.1 — Torx is brand new (it has grown substantially and fast: it now carries
16 example notebooks and 54 stars, up from a single commit in my prior survey).  THE HONEST ONE-
LINE COMPARISON: THRML is an excellent SAMPLER KERNEL and a decent TRAINER for Ising/Boltzmann
machines, deliberately positioned at the graph level. It is not a modelling stack. Layers 1-6
and 8-10 and 14 are absent by design or by not-yet-built: you hand-author the energy function,
hand-author the encoding, hand-colour the graph with networkx, and read answers back
positionally. That is the ferrotherm opportunity surface, and it is wide — but note the two
places Extropic is genuinely ahead and should not be dismissed: (i) verification PRACTICE in
Torx notebooks (TV against exact enumeration, with the naive-sampler null control at 0.4020 vs
0.0634 — a real null control, even if computed by hand rather than by API), and (ii) the first-
party AGENT SKILL (thrml-skill, SKILL.md format, cross-agent), which is a distribution channel
most stacks in this field do not have.  MEASUREMENT CAVEAT WORTH CARRYING FORWARD: the '~5-9
orders of magnitude less energy than a GPU' in codon_opt's README rests on a single hardcoded
literal, `potts_chip_energy = 10 * 256 * 1e-10` (energy_comparison.py:299), with TDP/peak-TFLOPS
quotients as the baseline and an assumed 300 W constant even in the section labelled 'MEASURED
BENCHMARK'. No power instrument appears in any Extropic repo. Any ferrotherm comparison should
therefore compete on MEASURED joules, where the bar is currently unclaimed rather than high.
TWO SPEC UPDATES observed on extropic.ai/hardware that differ from earlier notes: Z1 is now
presented in two form factors — Z1 Stick (M.2, 'over half a million pbits') and Z1 Card (PCIe,
'over 4 million pbits'), both early access 2027 — rather than a single ~269k-pbit part. Treat as
vendor claims; the parts are pre-characterization.

## D-Wave Ocean SDK 9.0.0 (Python). Surveyed from a live install (`pip install dwave-ocean-sdk` into a venv at /private/tmp/claude-501/-Users-dcharlot-vibe-coding-bmi-concept/ec64a91f-fbcc-442a-8e9b-f2f378c7a081/scratchpad/oceanenv), giving component versions dimod 0.12.21, dwave-system 1.33.0, dwave-samplers 1.6.0, minorminer 0.2.19, dwave-hybrid 0.6.14, dwave-preprocessing 0.6.10, dwave-optimization 0.6.4, penaltymodel 1.3.0, dwave-cloud-client 0.14.0, dwave-inspector 0.5.5, dwave-networkx 0.8.18. All claims below are from that source tree (absolute paths given) or from docs.dwavequantum.com; several were verified by executing code, marked "verified live".

### 1. Modelling layer — **yes**

Deepest layer in the survey. dimod ships four model classes plus a fifth in dwave-optimization:
BinaryQuadraticModel (dimod/binary/binary_quadratic_model.py), ConstrainedQuadraticModel
(dimod/constrained/constrained.py, C++/Cython core in dimod/constrained/cyconstrained.pyx),
QuadraticModel, DiscreteQuadraticModel, and dwave.optimization.Model (nonlinear, 67 symbol
classes). Named variables with domains: dimod.Binary/Spin/Integer/Real and the plural
dimod.Binaries/Integers/Reals construct symbolic single-variable models; domains are per-
variable via lower_bound=/upper_bound= and readable back with
cqm.lower_bound(v)/cqm.upper_bound(v)/cqm.vartype(v) (cyconstrained.pyx:479,703,724). Objective
via cqm.set_objective() (cyconstrained.pyx:605); operator overloading builds expressions
(dimod/sym.py) and dimod.quicksum aggregates. Answers come back BY NAME: samples are dicts keyed
by the user's variable label. VERIFIED LIVE: built a CQM over Binaries ['a','b','c'],
Integer('i', lower_bound=2, upper_bound=9), Real('r', lower_bound=0, upper_bound=1.5) and read
violations back keyed by constraint label. Caveat: plain Python sum() over a Binaries list then
compared with == raised TypeError('unexpected data format') from constrained.py:205;
dimod.quicksum works.

> /private/tmp/claude-501/-Users-dcharlot-vibe-coding-bmi-concept/ec64a91f-fbcc-442a-8e9b-f2f378c7a081/scratchpad/oceanenv/lib/python3.9/site-packages/dimod/constrained/constrained.py and https://docs.dwavequantum.com/en/latest/ocean/api_ref_dimod/models.html

### 2. Encodings (one-hot, domain-wall, binary/log; choice exposed?) — partial

ONE-HOT: present but hard-wired.
cqm.add_discrete()/add_discrete_from_iterable()/add_discrete_from_model()
(constrained.py:445,509,565,631) add a sum(cases)==1 equality and flag it via
constraint.mark_discrete(); DQM->CQM conversion in cyconstrained.pyx:395
from_discrete_quadratic_model() emits exactly sum-of-cases==1 with no alternative. BINARY/LOG:
dimod.generators.binary_encoding(v, upper_bound) (dimod/generators/integer.py:24) implements the
Karimi & Ronagh (arXiv:1706.01945) bounded-coefficient binary expansion, labelling bits
('i',1),('i',2),('i',3,'msb'). It is invoked AUTOMATICALLY and unconditionally by
dimod.cqm_to_bqm (constrained.py:2143) — the caller cannot choose a different integer encoding
at that entry point. DOMAIN-WALL: NOT FOUND — a case-insensitive grep for 'domain.wall' across
dimod, dwave, hybrid, penaltymodel, minorminer and dwave_networkx returned zero hits. The one
genuinely EXPOSED encoding choice is for inequality penalties:
BinaryQuadraticModel.add_linear_inequality_constraint(..., penalization_method:
Literal['slack','unbalanced']) (dimod/binary/binary_quadratic_model.py:707-755), where
'unbalanced' cites arXiv:2211.13914 and avoids slack variables entirely. So: one-hot yes (not
selectable), binary/log yes (not selectable), domain-wall not found, and choice exposed only for
slack-vs-unbalanced inequality penalization.

> /private/tmp/claude-501/-Users-dcharlot-vibe-coding-bmi-concept/ec64a91f-fbcc-442a-8e9b-f2f378c7a081/scratchpad/oceanenv/lib/python3.9/site-packages/dimod/binary/binary_quadratic_model.py:707 and dimod/generators/integer.py:24

### 3. Higher-order reduction (k-body -> pairwise with ancillas) — **yes**

dimod.reduce_binary_polynomial(poly) (dimod/higherorder/utils.py:102) lowers a BinaryPolynomial
to linear+quadratic terms, returning (reduced_terms, constraints) where each constraint is
(pair, product_ancilla). The algorithm is a greedy most-frequent-pair substitution: it indexes
every pair appearing in a >2-body term, keeps a queue bucketed by occurrence count, and
repeatedly pops the pair appearing in the MOST terms, introducing one ancilla per substitution
via _new_product() (utils.py:67). dimod.make_quadratic(poly, strength, vartype) (utils.py:272)
then materialises the ancilla definitions as AND-gate penalty terms at a user-supplied scalar
`strength`; dimod.make_quadratic_cqm(poly) (utils.py:222) is the constraint-based alternative
that adds them as hard CQM constraints instead of penalties. Supporting pieces:
dimod.higherorder.polynomial.BinaryPolynomial, dimod.ExactPolySolver, dimod.HigherOrderComposite
and PolyFixedVariableComposite / PolyScaleComposite
(dimod/reference/composites/higherordercomposites.py), dimod.poly_energy/poly_energies
(utils.py:340,368). penaltymodel.generate() (penaltymodel/generation.py:118) and
penaltymodel.get_penalty_model() (penaltymodel/interface.py:34) synthesise penalty BQMs (via
LP/MIP/maxgap, penaltymodel/lp.py, mip.py, maxgap.py) for arbitrary small truth tables — but
penaltymodel emits a DeprecationWarning on import: 'penaltymodel is deprecated and will be
removed in Ocean 10.'

> /private/tmp/claude-501/-Users-dcharlot-vibe-coding-bmi-concept/ec64a91f-fbcc-442a-8e9b-f2f378c7a081/scratchpad/oceanenv/lib/python3.9/site-packages/dimod/higherorder/utils.py:102

### 4. Constraint vocabulary — partial

VERIFIED LIVE on one CQM (num_constraints()==6, num_soft_constraints()==1). EQUALITY: yes,
`a+b+c == 2`. INEQUALITY: yes, both senses, `a + 2*b <= 2` and `i - c >= 3`; Sense enum is
Eq/Le/Ge only (no strict inequalities, no !=). QUADRATIC constraints: yes in CQM — `a*b + i*c <=
4` was accepted and evaluated. CARDINALITY: expressible as a linear equality (`a+b+c == 2`
worked) but there is NO dedicated cardinality primitive. EXACTLY-ONE: yes,
cqm.add_discrete(['d','e','f']) (constrained.py:445). SOFT constraints: yes, add_constraint(...,
weight=5.0, penalty='linear'|'quadratic'); weight=None/inf means hard. ALL-DIFFERENT: NOT FOUND
in dimod — `[n for n in dir(cqm) if 'diff' in n.lower() or 'cardinal' in n.lower()]` returned
[]. The nearest equivalent lives in a different model class:
dwave.optimization.symbols.ListVariable and .Permutation give a permutation decision variable
that is implicitly all-different, and Model.list()/Model.set()/Model.disjoint_lists() expose
combinatorial decisions — but these belong to the nonlinear model consumed only by
LeapHybridNLSampler, not to CQM/BQM. IMPORTANT ASYMMETRY: dimod.cqm_to_bqm raises
ValueError('CQM must not have any quadratic constraints') (constrained.py:2193) and ValueError
if any integer variable has a nonzero lower bound (constrained.py:2158) — so the rich CQM
vocabulary is only fully consumable by the cloud Leap CQM solver, not by the local BQM path.

> /private/tmp/claude-501/-Users-dcharlot-vibe-coding-bmi-concept/ec64a91f-fbcc-442a-8e9b-f2f378c7a081/scratchpad/oceanenv/lib/python3.9/site-packages/dimod/constrained/constrained.py:186-703 (live-executed)

### 5. Penalty handling (auto scaling, feasibility, WHICH constraint broke) — partial

WHICH CONSTRAINT BROKE — YES, this is a genuine strength and it reports by LABEL.
cqm.violations(sample) returns a dict {label: violation}; VERIFIED LIVE output {'eq_card2': 0.0,
'le': 1.0, 'ge': 1.0, 'quadratic_con': -3.0, 'soft': 0.9, 'one_hot': 1.0}.
cqm.violations(sample, skip_satisfied=True) filters to only the broken ones; clip=True floors
negative slack at 0. cqm.iter_constraint_data(sample) (constrained.py:1162) yields
ConstraintData(label, lhs_energy, rhs_energy, sense, activity, violation) — a NamedTuple at
constrained.py:86 — so you get the magnitude AND the two sides, not just a boolean. At sampleset
level, SampleSet.from_samples_cqm (dimod/sampleset.py:860-929) attaches an is_satisfied boolean
MATRIX (samples x constraints) plus an is_feasible vector, with info['constraint_labels'] giving
the column ordering; sampleset.filter(lambda d: d.is_feasible) selects feasible rows.
FEASIBILITY CHECK — yes: cqm.check_feasible(sample, rtol=1e-6, atol=1e-8) (constrained.py:762),
tolerance rule violation <= atol + rtol*|rhs_energy|. Note is_feasible ignores soft constraints
(sampleset.py:921 restricts to hard columns). AUTOMATIC SCALING — WEAK. dimod.cqm_to_bqm uses
ONE GLOBAL lagrange_multiplier for ALL constraints, defaulting to 10x the largest absolute bias
in the objective (constrained.py:2174-2185); there is no per-constraint scaling and no
feasibility-driven re-scaling loop. Automatic scaling that IS principled applies to CHAIN
penalties, not constraint penalties:
dwave.embedding.chain_strength.uniform_torque_compensation(bqm, embedding, prefactor=1.414)
computes sqrt(mean degree) * RMS coupling (Ray2020 torque argument), and .scaled(prefactor=1.0)
uses the max absolute bias. GAP: dwave.preprocessing.Presolver detects infeasibility but does
NOT name the culprit — VERIFIED LIVE, a CQM with 'impossible_lower' (i>=8 on i in [0,5])
returned Feasibility.Infeasible from p.feasibility() with no constraint label and no exception.

> /private/tmp/claude-501/-Users-dcharlot-vibe-coding-bmi-concept/ec64a91f-fbcc-442a-8e9b-f2f378c7a081/scratchpad/oceanenv/lib/python3.9/site-packages/dimod/constrained/constrained.py:1232,1847 and dimod/sampleset.py:860 (live-executed)

### 6. Embedding / placement onto hardware topology — **yes**

The most mature layer in Ocean and the hardest for any competitor to match. FINDING:
minorminer.find_embedding (Cai-Macready-Roy heuristic, C++ core minorminer/_minorminer.pyx),
minorminer.busclique (clique/biclique embedder with cache, busclique.pyx), minorminer.subgraph
(Glasgow subgraph solver for exact subgraph embedding, subgraph.pyx),
minorminer.layout.{layout,placement} for layout-aware placement,
minorminer.utils.{chimera,pegasus,zephyr} topology-specific embedders,
minorminer.utils.parallel_embeddings and minorminer.utils.feasibility. DIAGNOSTICS:
minorminer.utils.diagnostic.diagnose_embedding (line 25), is_valid_embedding (131),
verify_embedding (152). APPLY/UNAPPLY: dwave.embedding.embed_bqm/embed_ising/embed_qubo and
unembed_sampleset (dwave/embedding/transforms.py), dwave.embedding.EmbeddedStructure,
chain_to_quadratic, target_to_source. CHAIN-BREAK RESOLUTION as a pluggable policy:
majority_vote, discard, weighted_random (dwave/embedding/chain_breaks.py:156,96,227) and the
MinimizeEnergy callable class; broken_chains (line 32) and
dwave.embedding.utils.chain_break_frequency (line 153) quantify the damage. COMPOSITES that
automate it: EmbeddingComposite, FixedEmbeddingComposite, LazyEmbeddingComposite,
LazyFixedEmbeddingComposite, AutoEmbeddingComposite, ParallelEmbeddingComposite,
TilingComposite, VirtualGraphComposite, LinearAncillaComposite (dwave/system/composites/), plus
DWaveCliqueSampler for dense all-to-all problems. Topology graph generators in dwave_networkx
(chimera_graph/pegasus_graph/zephyr_graph) — but note dwave-networkx is now DEPRECATED and
superseded by dwave-graphs 1.0.0 per the official package list.

> https://docs.dwavequantum.com/en/latest/ocean/packages.html and /private/tmp/claude-501/-Users-dcharlot-vibe-coding-bmi-concept/ec64a91f-fbcc-442a-8e9b-f2f378c7a081/scratchpad/oceanenv/lib/python3.9/site-packages/dwave/embedding/

### 7. Samplers / solvers (algorithms, CPU/GPU/hardware) — **yes**

CPU (dwave-samplers 1.6.0, Cython over C++; enumerated live from dir(dwave.samplers)):
SimulatedAnnealingSampler (aliased as `Neal` — the old standalone `neal` package is folded in),
TabuSampler (MST2 tabu search),
SteepestDescentSampler/SteepestDescentSolver/SteepestDescentComposite (discrete steepest
descent, dwave/samplers/greedy/), TreeDecompositionSampler/TreeDecompositionSolver (EXACT on
bounded-treewidth graphs), PlanarGraphSolver (EXACT ground state for planar zero-field Ising),
RandomSampler, and two quantum-emulating samplers: PathIntegralAnnealingSampler (path-
integral/worldline QMC, dwave/samplers/sqa/pimc_annealing.pyx) and RotorModelAnnealingSampler
(rotor/coherent-spin relaxation, arXiv:1401.7087, rotormc_annealing.pyx). EXACT/reference in
dimod: ExactSolver, ExactCQMSolver, ExactPolySolver, IdentitySampler, NullSampler. HYBRID meta-
heuristics (dwave-hybrid, pure-Python workflow framework): KerberosSampler and
LatticeLNLSSampler (hybrid/reference/kerberos.py:110, lattice_lnls.py:179), plus parallel
tempering (FixedTemperatureSampler, SwapReplicaPairRandom, SwapReplicasDownsweep,
SpawnParallelTemperingReplicas in hybrid/reference/pt.py), population annealing
(EnergyWeightedResampler, CalculateAnnealingBetaSchedule in hybrid/reference/pa.py), and a
qbsolv-style decomposer (hybrid/reference/qbsolv.py). HARDWARE: DWaveSampler (QPU),
DWaveCliqueSampler, LeapHybridSampler/LeapHybridBQMSampler, LeapHybridCQMSampler,
LeapHybridDQMSampler, LeapHybridNLSampler (dwave/system/samplers/). GPU: NOT FOUND — no CUDA,
ROCm, OpenCL or GPU backend anywhere in the installed tree; every classical sampler is CPU
C++/Cython.

> /private/tmp/claude-501/-Users-dcharlot-vibe-coding-bmi-concept/ec64a91f-fbcc-442a-8e9b-f2f378c7a081/scratchpad/oceanenv/lib/python3.9/site-packages/dwave/samplers/ and hybrid/reference/ (enumerated live)

### 8. Device abstraction (one interface over MULTIPLE VENDORS, capability declaration) — partial

Capability declaration is strong; multi-vendor is absent. CAPABILITY DECLARATION — yes: every
sampler exposes .properties and .parameters dicts
(dwave/system/samplers/dwave_sampler.py:267,291); structured samplers add .nodelist, .edgelist,
.adjacency and .to_networkx_graph() (dwave_sampler.py:324,348,606). Leap hybrid samplers publish
limits used for pre-flight validation — LeapHybridCQMSampler checks
maximum_number_of_constraints, maximum_number_of_variables and maximum_number_of_biases before
submitting (leap_hybrid_sampler.py:736,743,750) and exposes min_time_limit(cqm) (line 770).
DISCOVERY/SELECTION — yes: dwave.cloud.Client.get_solvers(refresh, order_by='avg_load',
**filters) and get_solver(name, **filters) (dwave/cloud/client/base.py:841,1195) support
feature-based filtering on qpu/hybrid/software/online/num_active_qubits/avg_load with operator
suffixes. The cloud layer models solver kinds as classes: BQMSolver, DQMSolver, CQMSolver,
NLSolver, StructuredSolver (dwave/cloud/solver.py:546,613,717,796,931). MULTI-VENDOR — NOT
FOUND. A case-insensitive grep across dwave/cloud and dwave/system for ibm, rigetti, fujitsu,
toshiba, qatalyst, hitachi, nec, 'amazon braket' and 'azure quantum' returned ZERO hits. Ocean
is one interface over D-Wave's OWN family of solvers (QPU + four Leap hybrid solver types), not
over multiple vendors' machines. Third parties can implement the dimod.Sampler ABC (that is what
makes dimod 'a shared API for samplers' per the dimod README), so the abstraction is extensible
in principle — but no non-D-Wave device backend ships in the SDK.

> /private/tmp/claude-501/-Users-dcharlot-vibe-coding-bmi-concept/ec64a91f-fbcc-442a-8e9b-f2f378c7a081/scratchpad/oceanenv/lib/python3.9/site-packages/dwave/cloud/client/base.py:841 and dwave/cloud/solver.py

### 9. Verification (certificates, effective temperature, ESS, TV vs exact, conformance) — partial

EFFECTIVE TEMPERATURE — YES, and it is the most serious sampling-fidelity instrument found in
any incumbent stack. The whole module dwave/system/temperatures.py exists for it:
maximum_pseudolikelihood_temperature(bqm, sampleset, ...) (line 360) solves the convex MPL root
equation 0 = sum_i sum_s f_i(s) exp(f_i(s)/T) over effective fields via scipy.optimize, and
returns bootstrap error bars when num_bootstrap_samples>0; maximum_pseudolikelihood() (line 701)
generalises to fitting Hamiltonian parameters; fast_effective_temperature(sampler, ...) (line
1336) submits single-qubit problems at h_range=(-1/6.1, 1/6.1) and infers T from the excitation
rate — it works on ANY dimod.Sampler, not just a QPU; freezeout_effective_temperature() (line
1228) derives T from freeze-out time and anneal schedule. Supporting physics: effective_field()
(91), background_susceptibility_ising/bqm (198, 299), Ip_in_units_of_B (995),
h_to_fluxbias/fluxbias_to_h (1092, 1159). The docstring is honest about the limit: 'If the
distribution is not Boltzmann with respect to the BQM provided, as may be the case for heuristic
samplers (such as annealers), the temperature estimate can be interpreted as characterizing only
a rate of LOCAL excitations.' ESS — NOT FOUND. TV DISTANCE vs EXACT — NOT FOUND. AUTOCORRELATION
— NOT FOUND. KL DIVERGENCE — NOT FOUND. A grep for effective_sample_size, ESS, autocorrelation,
total_variation, kl_divergence across dimod, dwave, hybrid, penaltymodel, minorminer and
dwave_networkx returned zero hits. SAMPLING CERTIFICATE — NOT FOUND: nothing bounds the distance
between the returned samples and the target Boltzmann distribution. CONFORMANCE — yes but
INTERFACE-level only: dimod.testing provides assert_sampler_api, assert_composite_api,
assert_structured_api, assert_sampleset_energies (+ _cqm/_dqm variants), assert_consistent_bqm,
assert_bqm_almost_equal and load_sampler_bqm_tests (dimod/testing/sampler.py, asserts.py) —
these check that a sampler obeys the dimod ABC and that reported energies match recomputed
energies, not that its distribution is correct. EMBEDDING-level conformance is real:
verify_embedding/diagnose_embedding/is_valid_embedding plus chain_break_frequency quantify
embedding fidelity. FEASIBILITY-level: check_feasible / is_feasible / is_satisfied (see layer
5).

> /private/tmp/claude-501/-Users-dcharlot-vibe-coding-bmi-concept/ec64a91f-fbcc-442a-8e9b-f2f378c7a081/scratchpad/oceanenv/lib/python3.9/site-packages/dwave/system/temperatures.py:360,701,1228,1336 and dimod/testing/

### 10. Energy / cost accounting (joules or price per operation) — **not found**

NOT FOUND for joules and NOT FOUND for price. What exists is TIME accounting only:
Solver.estimate_qpu_access_time(num_qubits, num_reads, annealing_time, anneal_schedule,
initial_state, reverse_anneal, reinitialize_state, programming_thermalization,
readout_thermalization, reduce_intersample_correlation) (dwave/cloud/solver.py:1410) returns
estimated QPU access time IN MICROSECONDS, computed from the solver's problem_timing_data
property and the number of qubits the embedding consumes; after a run,
computation.timing['qpu_access_time'] gives the measured microseconds, and
problem_run_duration_range (solver.py:1071) bounds it. Leap bills against QPU/solver access
TIME, so microseconds are the de facto cost unit — but no monetary conversion, quota balance or
usage ledger ships in the SDK: `dwave leap --help` exposes only a `project` subcommand, and a
grep for quota/solver_access_time/remaining_time/usage across dwave/cloud/api and
dwave/cloud/client returned nothing. IMPORTANT DISAMBIGUATION: the strings 'Joule'/'Joules' DO
appear in the tree
(dwave/system/temperatures.py:1020,1046,1063,1075,1139,1209,1288,1316,1323,1331) but they are
PHYSICS units — converting the annealing energy scale B and the temperature between GHz, Joules
and milli-Kelvin using Planck's constant and k_B. They are not an energy-consumption ledger. No
watts, no power draw, no kWh, no price-per-operation anywhere in the SDK.

> /private/tmp/claude-501/-Users-dcharlot-vibe-coding-bmi-concept/ec64a91f-fbcc-442a-8e9b-f2f378c7a081/scratchpad/oceanenv/lib/python3.9/site-packages/dwave/cloud/solver.py:1410

### 11. Language surfaces (native vs FFI vs subprocess) — partial

PYTHON ONLY as a public API. The official package list describes all 14 Ocean packages as
Python. There IS substantial C++ underneath, exposed as installable headers but not documented
as a supported public API: dimod ships dimod/include/dimod/{abc.h, binary_quadratic_model.h,
constrained_quadratic_model.h, constraint.h, expression.h, iterators.h, quadratic_model.h,
utils.h, vartypes.h} and dwave-preprocessing ships dwave/preprocessing/include/. These are
consumed by Cython extension modules IN-PROCESS (native, not FFI or subprocess):
dimod/constrained/cyconstrained.pyx, dimod/cyqmbase/, dimod/cylp.pyx,
minorminer/_minorminer.pyx, minorminer/busclique.pyx, minorminer/subgraph.pyx,
dwave/samplers/greedy/descent.pyx, dwave/samplers/sqa/pimc_annealing.pyx,
dwave/preprocessing/cyfix_variables.pyx. The dimod README mentions a meson C++ test setup but
makes no claim of a supported C++ or header-only public library. NOT FOUND: Rust, Julia,
JavaScript/WASM, Java, C, Go — a find for *.rs, *.jl, *.js, *.java across dimod and dwave
returned nothing. Interchange formats partly substitute for bindings: cqm.to_file()/from_file(),
dimod.lp / cqm.from_lp_file() (LP format), and dimod/serialization/. One non-Python surface
exists: the `dwave` CLI (click-based console script from dwave-cloud-client) with subcommands
auth, cache, config, install, leap, ping, sample, setup, solvers, upload.

> /private/tmp/claude-501/-Users-dcharlot-vibe-coding-bmi-concept/ec64a91f-fbcc-442a-8e9b-f2f378c7a081/scratchpad/oceanenv/lib/python3.9/site-packages/dimod/include/dimod/ and https://docs.dwavequantum.com/en/latest/ocean/packages.html

### 12. Agent / AI surfaces (MCP, HTTP API, structured tool schemas) — **not found**

MCP — NOT FOUND. A case-insensitive grep for 'modelcontextprotocol', 'model context protocol'
and '\bmcp\b' across dimod, dwave and hybrid returned zero hits, and a web search for a D-Wave
Ocean MCP server surfaced only official docs pages with no MCP integration. STRUCTURED TOOL
SCHEMAS — NOT FOUND: no JSON-schema tool definitions or function-calling descriptors ship in the
SDK. HTTP API — present only in the WRONG DIRECTION for agent use: dwave/cloud/api/ makes Ocean
an HTTP CLIENT of D-Wave's Solver API (SAPI) and Leap API; there is no server you can host to
expose your own models. The single HTTP SERVER in the tree is incidental —
dwave/inspector/server.py runs a local Flask app purely to serve the problem-visualiser UI to a
browser, and even that depends on the closed-source viewer bundle. What DOES exist and is
machine-consumable, if an agent harness wants to build on it: serialisation via
cqm.to_file()/ConstrainedQuadraticModel.from_file(), LP-format round-trip via cqm.from_lp_file()
and dimod/lp.py, and the `dwave` CLI (dwave solvers, dwave sample, dwave ping) which an agent
could drive as a subprocess. None of that is an agent-facing contract; it is a general-purpose
SDK.

> /private/tmp/claude-501/-Users-dcharlot-vibe-coding-bmi-concept/ec64a91f-fbcc-442a-8e9b-f2f378c7a081/scratchpad/oceanenv/lib/python3.9/site-packages/dwave/cloud/api/ and dwave/inspector/server.py (grep for MCP returned nothing)

### 13. Training (energy-based model training, gradient estimators) — **not found**

NOT FOUND. No energy-based-model training, no Boltzmann-machine training loop, no contrastive
divergence, no gradient estimator ships in Ocean. A grep for 'gradient', 'boltzmann machine' and
'contrastive divergence' across dimod, dwave and hybrid produced only two kinds of false
positive: (a) dwave/samplers/greedy/sampler.py and decl.pxd, where steepest_gradient_descent is
a DISCRETE LOCAL-SEARCH optimiser explicitly described as 'the discrete analogue of gradient
descent, but the best move is computed using a local minimization rather than computing a
gradient' (sampler.py:34-36) — an optimiser, not a learning rule; and (b)
dwave/system/temperatures.py:570,892, where 'gradient' refers to the derivative of the pseudo-
likelihood objective inside the effective-temperature root-finder. The closest thing to a
learning primitive in the whole stack is maximum_pseudolikelihood() (temperatures.py:701), which
fits Hamiltonian parameters to an observed sampleset by MPL — that is a parameter-estimation
routine for CHARACTERISING a device, not a training API for building generative models, and it
is not documented or positioned as one. Ocean is a modelling-and-sampling SDK; training is left
entirely to the user.

> /private/tmp/claude-501/-Users-dcharlot-vibe-coding-bmi-concept/ec64a91f-fbcc-442a-8e9b-f2f378c7a081/scratchpad/oceanenv/lib/python3.9/site-packages/dwave/samplers/greedy/sampler.py:34 and dwave/system/temperatures.py:701

### 14. Visual / graph programming — partial

VISUALISATION yes; visual PROGRAMMING not found. dwave-inspector 0.5.5 is 'Visualizer for
problems submitted to quantum computers' per the official package list — it shows the logical
problem, the embedding onto the QPU graph, chains and the returned samples in a browser
(dwave/inspector/{server.py, adapters.py, viewers.py, proxies.py}). CRITICAL LICENCE CAVEAT: the
pip-installed dwave-inspector is only an Apache-2.0 WRAPPER;
dwave/inspector/package_info.py:32-45 registers 'the non-open-source packages required for
dwave-inspector to work' — namely dwave-inspectorapp==0.3.3 under a licence named literally
'D-Wave EULA' (https://docs.dwavequantum.com/en/latest/licenses.html) — and it is obtained
separately via `dwave install` ('Install optional non-open-source Ocean packages'). Additional
drawing helpers: dwave/embedding/drawing.py, and dwave_networkx's chimera/pegasus/zephyr layout
drawing (though dwave-networkx is deprecated in favour of dwave-graphs 1.0.0). NOT FOUND: any
node-graph or block-based AUTHORING surface — you cannot construct a model by wiring nodes;
dwave-hybrid's flow.py composes Runnables (Branch, RacingBranches, ParallelBranches, Loop) into
dataflow workflows, but that is a Python combinator API, not a visual editor.

> /private/tmp/claude-501/-Users-dcharlot-vibe-coding-bmi-concept/ec64a91f-fbcc-442a-8e9b-f2f378c7a081/scratchpad/oceanenv/lib/python3.9/site-packages/dwave/inspector/package_info.py:32-45

### 15. Licence, openness, hardware requirement — **yes**

LICENCE: Apache 2.0 across the board — confirmed from installed dist-info METADATA for dimod
0.12.21, dwave-system 1.33.0, dwave-samplers 1.6.0, dwave-hybrid 0.6.14, minorminer 0.2.19,
dwave-optimization 0.6.4, penaltymodel 1.3.0 and dwave-inspector 0.5.5. Sources are on GitHub
under github.com/dwavesystems/. ONE CLOSED COMPONENT: dwave-inspectorapp==0.3.3 under a 'D-Wave
EULA' (see layer 14) — the only non-open piece, and it is optional. HARDWARE NOT REQUIRED for
most of the stack: `pip install dwave-ocean-sdk` gave a fully working local environment with no
account and no token — I built CQMs, evaluated violations, ran the Presolver and enumerated all
samplers offline. dimod, dwave-samplers (SA, tabu, steepest descent, tree decomposition, planar,
SQA), minorminer, dwave-preprocessing, dwave-hybrid and penaltymodel all run purely on CPU.
HARDWARE/CLOUD IS REQUIRED for: DWaveSampler, DWaveCliqueSampler and all four Leap hybrid
samplers (LeapHybridSampler/BQM, CQM, DQM, NL) — these need a Leap account and an API token,
configured via `dwave auth` / `dwave config`, and are billed by QPU/solver access time. Note
also that the LARGEST-capacity solvers are cloud-only: the Leap hybrid CQM solver is the only
thing in the stack that consumes the full CQM vocabulary (quadratic constraints, nonzero integer
lower bounds), since the local dimod.cqm_to_bqm path rejects both. DEPRECATIONS to flag:
penaltymodel is deprecated with removal announced for Ocean 10 (emits DeprecationWarning on
import), and dwave-networkx is deprecated in favour of dwave-graphs 1.0.0.

> https://docs.dwavequantum.com/en/latest/ocean/packages.html and dist-info METADATA under /private/tmp/claude-501/-Users-dcharlot-vibe-coding-bmi-concept/ec64a91f-fbcc-442a-8e9b-f2f378c7a081/scratchpad/oceanenv/lib/python3.9/site-packages/

**Notes.** METHOD: Installed dwave-ocean-sdk 9.0.0 into a Python 3.9 venv and read the actual source;
several claims were verified by EXECUTING code, not just reading docstrings (marked "verified
live" in the layer detail). Base path for all local citations: /private/tmp/claude-501/-Users-
dcharlot-vibe-coding-bmi-
concept/ec64a91f-fbcc-442a-8e9b-f2f378c7a081/scratchpad/oceanenv/lib/python3.9/site-packages/.
Note the installed versions lag the current published ones (docs list dimod 0.12.22, dwave-
samplers 1.8.0, dwave-system 1.35.0, dwave-optimization 0.7.1, dwave-ocean-sdk 9.4.0) because
the sandbox pip is old; the API surface claims should hold but exact line numbers are pinned to
the versions listed under `stack`.  LICENCE/LANGUAGE/HARDWARE SUMMARY: Apache 2.0, Python-only
public API (C++ headers + Cython internally, no supported non-Python binding), and hardware is
NOT required for the classical two-thirds of the stack — but IS required for DWaveSampler and
all Leap hybrid solvers, which are the only path to the full CQM vocabulary.  SIX FINDINGS THAT
MATTER MOST FOR A FERROTHERM COMPARISON:  1. Ocean's real moat is layer 6 (embedding), not layer
1. minorminer + dwave.embedding + the composite family is ~15 years of tooling — heuristic and
exact embedders, topology-specific embedders, chain-break policies as pluggable strategies, and
embedding-validity diagnostics. Anything competing on modelling alone concedes this.  2. Layer 5
is genuinely strong on "WHICH constraint broke" and this should NOT be scored as a gap for
Ocean. cqm.violations() returns {label: violation}, iter_constraint_data() returns a NamedTuple
with lhs_energy/rhs_energy/sense/activity/violation, and the SampleSet carries an is_satisfied
MATRIX plus info['constraint_labels']. Ferrotherm needs at least parity here.  3. But Ocean's
penalty SCALING is weak and that is the exploitable seam: cqm_to_bqm applies ONE GLOBAL Lagrange
multiplier (default 10x max objective bias) to every constraint, with no per-constraint scaling
and no feasibility-driven re-scaling. The sophisticated auto-scaling in the stack
(uniform_torque_compensation) applies to CHAIN strength, a different problem.  4. Layer 9 splits
sharply. Effective temperature is a REAL, well-engineered capability (dwave.system.temperatures:
MPL estimator with bootstrap error bars, plus a freeze-out physical estimator, and
fast_effective_temperature works on any dimod.Sampler). But ESS, autocorrelation, TV-vs-exact,
KL divergence and any sampling CERTIFICATE are all NOT FOUND. dimod.testing is interface
conformance only. This is the clearest empty lane.  5. Layers 12 (MCP/agent), 13 (EBM training)
and 10 (joules/price) are outright empty. On layer 10 specifically, beware a false positive:
"Joules" appears ~10 times in temperatures.py but purely as a PHYSICS unit (Planck-constant
conversion of the annealing energy scale), never as an energy-consumption ledger. Ocean's only
cost currency is estimated QPU access time in microseconds.  6. Two capacity/vocabulary
asymmetries worth naming explicitly, because they are easy to mis-report: (a) CQM accepts
QUADRATIC constraints and nonzero integer lower bounds, but the LOCAL dimod.cqm_to_bqm path
raises ValueError on both — the full vocabulary is cloud-only; (b) all-different does not exist
in dimod at all, only as ListVariable/Permutation in dwave-optimization's nonlinear model, which
is consumed solely by the cloud LeapHybridNLSampler.  DEPRECATIONS: penaltymodel emits a
DeprecationWarning and is slated for removal in Ocean 10 (D-Wave's stated replacement is "use
the Leap hybrid solvers" — i.e. the open constraint-to-penalty compiler is being retired in
favour of a cloud service, which is strategically relevant). dwave-networkx is also deprecated,
superseded by dwave-graphs 1.0.0.  DOMAIN-WALL ENCODING: this review did not locate any domain-
wall encoding in the installed Ocean tree (grep across dimod, dwave, hybrid, penaltymodel,
minorminer, dwave_networkx returned zero hits). I did not exhaustively search D-Wave's examples
repositories or published notebooks, so state this as "not present in the Ocean SDK packages"
rather than "D-Wave does not support it".

## Japanese optimisation-software cluster — Fixstars Amplify SDK v1.3.1 (Fixstars Amplify Corp.), Jij Inc. (JijModeling 1.14.2, OMMX 2.0.12 / Rust crate 2.6.1, OpenJij 0.11.6, jij-cimod 1.7.4, MINTO, Qamomile), and PyQUBO 1.5.0 (Recruit Co., Ltd.). Surveyed by installing all packages and introspecting/executing the shipped APIs, cross-checked against vendor documentation.

### 1. Modelling layer — named variables with domains, constraints, objective; answers by name — **yes**

STRONGEST IN THE FIELD, and JijModeling and Amplify are strong in DIFFERENT ways. JijModeling
separates model from data:
`jm.BinaryVar/IntegerVar/ContinuousVar/SemiIntegerVar/SemiContinuousVar` carry `name, shape,
lower_bound, upper_bound, description, set_latex`; `jm.Placeholder(name, ndim, dtype, jagged)`
stands for not-yet-bound data; `jm.Element(name, belong_to=(0,n))` is a genuine index;
`jm.Constraint(name, expr, forall=[i])` and `jm.Problem` whose `.constraints` is a dict KEYED BY
NAME. One model text compiles at any size —
`jm.Interpreter({'n':3,'C':array}).eval_problem(problem)`. VERIFIED END TO END: I built an
assignment problem, and constraints came back as `onehot_row[0] feasible=False / onehot_row[1]
feasible=True / onehot_row[2] feasible=False` — name AND subscript preserved through lowering,
sampling and decode. Amplify instead materialises eagerly with NumPy-style array programming:
`VariableGenerator().array('Binary',3,3)` returns a `PolyArray` supporting `Dim0..Dim4`,
`einsum`, `matmul`, `sum`, slicing; `Model(objective)`; `Constraint.label` is settable but
DEFAULTS TO EMPTY STRING (measured: `label=''` on freshly built one_hot constraints), so name-
addressable answers are opt-in. PyQUBO has names (`Constraint(expr, label=...)`, `Array.create`,
`Placeholder`) but no abstract index/forall. OpenJij is NOT a modelling layer at all —
dict/dimod BQM only.

> https://jij-inc.github.io/JijModeling-Tutorials/ ; https://amplify.fixstars.com/en/docs/amplify/v1/index.html ; introspected /private/tmp/claude-501/-Users-dcharlot-vibe-coding-bmi-concept/ec64a91f-fbcc-442a-8e9b-f2f378c7a081/scratchpad/qv/lib/python3.9/site-packages/jijmodeling/__init__.pyi and .../amplify/__init__.pyi

### 2. Encodings — one-hot, domain-wall, binary/log, and whether the choice is exposed — **yes**

Amplify exposes the most choice, as keyword arguments to `solve()`: `IntegerEncodingMethod`
{Unary, Linear, Binary, Default}, `RealEncodingMethod` {Random4, Random8, Random16, Random32},
`InequalityFormulation` {Default, Relaxation, RelaxationLinear, RelaxationQuadra},
`PenaltyFormulation` {Default, IntegerVariable, LinearRelaxation, QuadraticRelaxation,
RealVariable, Relaxation}. Domain-wall is a first-class constraint builder
`amplify.domain_wall()`, alongside `one_hot()`. PyQUBO exposes encodings as CLASSES:
`OneHotEncInteger`, `LogEncInteger`, `UnaryEncInteger`, `OrderEncInteger` (order encoding = the
domain-wall family), in .../pyqubo/integer/order_enc_integer.py. OMMX provides
`Instance.log_encode(decision_variable_ids)` documented with the exact formula x = sum 2^i b_i +
(u-l-2^{m-1}+1) b_{m-1} + l. GAP: JijModeling itself exposes NO encoding choice — it delegates
entirely to OMMX and the backend adapter. Domain-wall NOT FOUND in JijModeling, OMMX or OpenJij
(grep for domain_wall|order_enc across all five hit only amplify and pyqubo).

> https://amplify.fixstars.com/en/docs/amplify/v1/intermediate.html ; .../site-packages/pyqubo/integer/order_enc_integer.py ; ommx.v1.Instance.log_encode docstring

### 3. Higher-order reduction — k-body terms lowered to pairwise, with ancillas — **yes**

Amplify is explicit and configurable: `QuadratizationMethod` {Substitute, IshikawaKZFD} plus
`Model.to_intermediate_model(AcceptableDegrees(objective={'Binary':'Quadratic'}))`. MEASURED:
objective q0*q1*q2*q3 (degree 4, 4 vars) lowered to degree 2 with 5 vars, the new ancilla
printed as `q'_0` in `q0q1 + q0q2 + q0q3 - 2q0q'_0 + ... + 3q'_0`. Documented difference:
Substitute adds quadratic equality penalties and works on binary only but can also reduce
CONSTRAINT degree; IshikawaKZFD handles binary and Ising and preserves values at optima but
cannot reduce constraint degrees. OpenJij takes the opposite route and does NOT reduce at all —
`SASampler.sample_hubo`, `.sample_huio`, `.sample_quio` solve higher-order natively; VERIFIED
`sample_hubo({(0,1,2):-1}, vartype='BINARY')` returned energy -1.0. OMMX preserves the choice at
the IR level with `as_hubo_format()` / `to_hubo()` beside `as_qubo_format()` / `to_qubo()`.
PyQUBO accepts higher-order expressions and `Model.to_qubo()` returns quadratic, but the
reduction happens inside the compiled `cpp_pyqubo` core — this review did not locate an exposed
method choice equivalent to QuadratizationMethod.

> https://amplify.fixstars.com/en/docs/amplify/v1/intermediate.html ; measured with amplify 1.3.1 and openjij 0.11.6 in the survey venv

### 4. Constraint vocabulary — equality, inequality/slack, cardinality, exactly-one, all-different — partial

Equality, inequality and exactly-one are well covered; cardinality is expressible but not named;
ALL-DIFFERENT IS ABSENT EVERYWHERE. Amplify builders: `equal_to`, `less_equal`, `greater_equal`,
`clamp` (two-sided range), `one_hot`, `domain_wall`. JijModeling: `jm.Constraint` over ==, <=,
>= with `forall`, plus `jm.CustomPenaltyTerm(name, expression, forall)` as an escape hatch for
penalties that are not (in)equalities. OMMX normalises everything to `Equality` = {EqualToZero,
LessThanOrEqualToZero} and supplies slack machinery:
`convert_inequality_to_equality_with_integer_slack(constraint_id, max_integer_range)` and
`add_integer_slack_to_inequality(constraint_id, slack_upper_bound)`. PyQUBO adds a logic
vocabulary absent from the others: `AndConst`, `OrConst`, `NotConst`, `XorConst`, `SubH`,
`WithPenalty`. NOT FOUND: any all-different / alldiff constraint in any of the five (grep for
all_different|alldifferent|alldiff across amplify, jijmodeling, ommx, openjij, pyqubo returned
zero hits). NOT FOUND: a named cardinality helper — you write `equal_to(sum, k)` or `sum(...) ==
k` yourself. Amplify's `one_hot` is the exactly-one special case only.

> https://amplify.fixstars.com/en/docs/amplify/v1/constraint.html ; ommx.v1.Instance method docstrings ; grep over the five installed packages

### 5. Penalty handling — automatic scaling, feasibility checking, reporting WHICH constraint broke — partial

Feasibility checking and which-constraint-broke reporting are EXCELLENT and verified by
execution; AUTOMATIC SCALING IS THE CLUSTER'S REAL GAP. Verified reporting: (a) OMMX
`Solution.constraints` yields `EvaluatedConstraint` objects with `.name, .subscripts, .id,
.evaluated_value, .feasible, .dual_variable, .removed_reason` — I fed a deliberately infeasible
state and got `onehot_row[0]: eval=1.0 feasible=False`, `onehot_row[1]: eval=0.0 feasible=True`,
`onehot_row[2]: eval=-1.0 feasible=False`. (b) PyQUBO `model.decode_sample(bad,
vartype='BINARY').constraints(only_broken=True)` returned `{'exactly_one': (False, 1.0)}` — name
plus violation magnitude. (c) JijModeling `Evaluation.constraint_violations` is a dict keyed by
constraint name, with `SampleSet.feasible` / `.infeasible` partitions. (d) Amplify
`Constraint.is_satisfied(values) -> bool` per constraint (measured True/False correctly),
`SolverSolution.is_feasible`, `Result.filter_solution`. Weight control: Amplify
`Constraint.weight` (default 1.0), `c *= 2`, and `Constraint.penalty` exposes the generated
penalty polynomial; OMMX `to_qubo(uniform_penalty_weight=..., penalty_weights={id: w})`,
`penalty_method()` (per-constraint parametric weights) and `uniform_penalty_method()`. NOT
FOUND: a principled automatic penalty scaler that derives weights from objective coefficient
magnitudes or bound analysis. Every stack defaults to weight 1.0 or requires you to supply the
number; OMMX's `penalty_method()` returns a ParametricInstance you must still parameterise.

> executed against ommx 2.0.12, pyqubo 1.5.0, amplify 1.3.1 in the survey venv ; ommx.v1.Instance.to_qubo signature

### 6. Embedding / placement onto hardware topology — partial

Amplify ONLY. Signature measured verbatim: `amplify.embed(poly: Poly, client_graph: Graph,
embedding_method: Literal['Default','Minor','Clique','Parallel'] | EmbeddingMethod =
EmbeddingMethod.Default, embedding_timeout: timedelta = 10s, chain_strength: float = 1.0) ->
tuple[Poly, list[ndarray[uint32]], list[tuple[int,int]]]`. Supporting types: `EmbeddingMethod`
{Default, Minor, Clique, Parallel}, `Graph` with `.nodes, .edges, .adjacency, .shape, .type`,
helper `to_edges`, and `Result.embedding` returns the embedding actually used. `HitachiClient`
carries a `.graph` attribute declaring its CMOS topology (measured present). `Parallel`
embedding — packing several independent copies of a small problem onto one large chip — is a
capability I did not find elsewhere in this cluster. NOT FOUND in JijModeling, OMMX, OpenJij or
PyQUBO: none carries a topology model or a minor-embedding routine; users fall back to D-Wave's
external `minorminer`.

> amplify.embed signature introspected from amplify 1.3.1 ; https://amplify.fixstars.com/en/docs/amplify/v1/index.html

### 7. Samplers/solvers — what algorithms, CPU/GPU/hardware — **yes**

OpenJij is the open local engine: `SASampler` (simulated annealing), `SQASampler` (path-integral
simulated quantum annealing), `CSQASampler` (continuous-time SQA). Its C++ core
`openjij.cxxjij.algorithm` exposes `Algorithm_SingleSpinFlip_run`, `Algorithm_SwendsenWang_run`,
`Algorithm_ContinuousTimeSwendsenWang_run`, `Algorithm_KLocal_run` plus `UpdateMethod` and
`RandomNumberEngine`. Verified a real QUBO solve (energy -1.0) and a real HUBO solve. CPU ONLY
in the shipped wheel — grep for cuda|gpu across openjij 0.11.6 returned nothing and
`cxxjij.system` exposed no GPU class; older OpenJij had Chimera GPU classes, so treat this as a
regression/removal in 0.11.6 rather than a permanent absence. Amplify ships NO local solver:
`solve()` against `FixstarsClient()` raised `RuntimeError: 401: Unauthorized`. It dispatches
instead to Fixstars Annealing Engine, D-Wave (`DWaveSamplerClient`, `LeapHybridSamplerClient`,
`LeapHybridCQMSamplerClient`), Fujitsu Digital Annealer (`FujitsuDA4Client`,
`FujitsuDA3cClient`), Toshiba SQBM+ v2 (`ToshibaSQBM2Client`), NEC Vector Annealing
(`NECVA2Client`), Hitachi CMOS (`HitachiClient`), `GurobiClient`, and gate-model backends
(`IBMClient`, `AerClient`, `QulacsClient`, `BraketSimulatorClient`). PyQUBO depends on `dwave-
neal` 0.6.0 / `dwave-samplers` 1.6.0 and ships `solve_qubo` / `solve_ising`. Jij's Qamomile
covers gate-model: QAOA, FQAOA, QRAO transpiled to Qiskit, QURI-Parts, CUDA-Q, HUGR, qBraid.

> executed against openjij 0.11.6, amplify 1.3.1 ; https://amplify.fixstars.com/en/docs/amplify/v1/features.html ; https://github.com/Jij-Inc/Qamomile

### 8. Device abstraction — one interface over multiple vendors' machines, capability declaration — **yes**

THE STANDOUT FINDING OF THIS SURVEY. Amplify has a genuine machine-readable capability
declaration, which most stacks in this field lack entirely. Every client exposes
`client.acceptable_degrees`, an `AcceptableDegrees` object holding THREE dicts — `.objective`,
`.equality_constraints`, `.inequality_constraints` — each mapping `VariableType` {Binary, Ising,
Integer, Real} to a `Degree` {Zero, Linear, Quadratic, Cubic, Quartic, HighOrder}. Measured
values: FixstarsClient = objective Binary:Quadratic, all constraints Zero (no native constraint
support); DWaveSamplerClient = objective Binary AND Ising Quadratic, constraints Zero;
FujitsuDA4Client = objective Binary:Quadratic plus INEQUALITY Binary:Linear (it natively accepts
linear inequalities, unlike the others); ToshibaSQBM2Client and NECVA2Client = objective
Binary:Quadratic only; HitachiClient = Ising:Quadratic only, Binary Zero, plus a `.graph`
topology; GurobiClient = Binary/Integer/Real Quadratic across objective AND both constraint
classes; LeapHybridCQMSamplerClient = Quadratic in Binary/Ising/Integer and Linear in Real for
objective and both constraint classes. `solve(model, client)` then applies exactly the
transforms — variable encoding, quadratization, penalty conversion, embedding — needed to land
the model inside that declared envelope, and inverts them on the way back. Clients also carry
`.token, .url, .proxy, .version, .parameters, .write_request_data, .write_response_data`. OMMX
provides the open-source counterpart at a coarser grain: `ommx.adapter.SolverAdapter` and
`SamplerAdapter` ABCs with abstract `solve`, `sample`, `decode`, `decode_to_sampleset`,
`solver_input`, `sampler_input`, plus typed `InfeasibleDetected` / `UnboundedDetected`
exceptions. Published adapters include ommx-openjij-adapter, ommx-pyscipopt-adapter, ommx-highs-
adapter, ommx-python-mip-adapter, ommx-gurobi-adapter, ommx-dwave-adapter, ommx-da4-adapter and
ommx-fixstars-amplify-adapter. NOT FOUND in OMMX: an `acceptable_degrees` analogue — an adapter
either handles an instance or raises, rather than declaring its envelope up front.

> executed capability probe across 8 amplify 1.3.1 clients ; ommx.adapter module introspection ; https://pypi.org/project/ommx-openjij-adapter

### 9. Verification — sampling certificates, effective temperature, ESS, TV vs exact, conformance — partial

Benchmarking exists; STATISTICAL VERIFICATION OF THE SAMPLER DOES NOT. What is present:
`openjij.utils.benchmark` supplies `solver_benchmark`, `residual_energy`, `success_probability`,
`time_to_solution` AND their standard errors `se_residual_energy`, `se_success_probability`,
`se_lower_tts`, `se_upper_tts` — so time-to-solution and success probability come with error
bars, which is more discipline than most. `amplify-benchmark` (MIT, github.com/fixstars/amplify-
benchmark) computes TTS, feasibility rate, success rate and objective-vs-time against TSPLIB,
QAPLIB, Gset, CVRPLIB, QPLIB and Sudoku. OMMX carries solution provenance flags `Optimality`
{Optimal, NotOptimal, Unspecified} and `Relaxation` {LpRelaxed, Unspecified}, ships reference
datasets `ommx.dataset.miplib2017` and `ommx.dataset.qplib`, and
`ommx.testing.SingleFeasibleLPGenerator` for constructing instances with known feasible points.
MINTO records runs with `collect_environment` (measured: os_name, os_version, platform_info,
cpu_info, cpu_count, memory_total, architecture, python_version, package_versions, timestamp),
`save_as_ommx_archive`, `push_github`. NOT FOUND, across all five packages: sampling
certificates, effective-temperature estimation, effective sample size (ESS), integrated
autocorrelation time, total-variation distance against exact enumeration, KL divergence, or any
conformance/acceptance suite a third-party sampler must pass. A grep for
effective_temperature|effective_sample|autocorrel|total_variation|tv_distance|kl_diverg across
amplify, jijmodeling, ommx, openjij and pyqubo returned ZERO hits. Everything measured is about
SOLUTION QUALITY (did we reach the best-known objective, how fast); nothing measures whether the
sampler is drawing from the Boltzmann distribution it claims. This is the cluster's largest
single gap.

> openjij.utils.benchmark introspection ; https://github.com/fixstars/amplify-benchmark ; ommx.testing / ommx.dataset introspection ; minto Experiment.get_environment_info executed

### 10. Energy/cost accounting — joules or price per operation — **not found**

NOT FOUND anywhere in the cluster. A grep for joule|watt|kwh|power_consum across all five
installed packages (amplify, jijmodeling, ommx, openjij, pyqubo) returned ZERO hits; the same
grep over MINTO returned only the string 'Cost' inside a CVRP problem generator, which is a
routing objective, not a price. WALL-CLOCK TIME is however first class everywhere, which is what
these stacks account for INSTEAD of energy: OpenJij `response.info` carries `sampling_time`,
`execution_time`, `list_exec_times`, `schedule` (measured); Amplify `Result` carries
`execution_time`, `response_time`, `total_time`, `num_solves`; JijModeling ships dedicated
`MeasuringTime`, `SolvingTime`, `SystemTime` classes and
`SampleSet.get_backend_calculation_time()`, with the JijZept timing breakdown itemised down to
`post_problem_and_instance_data`, `request_queue`, `fetch_problem_and_instance_data`,
`fetch_result`, `deserialize_solution`. MINTO records the machine's identity (CPU model, core
count, memory) but attaches no power figure to it. So: for a QPU-second or a DA4-second you can
learn the seconds and, off-platform, the price — but no joules, and no per-operation cost model,
is exposed by any API in this cluster.

> grep over the five installed packages and minto ; openjij Response.info measured ; jijmodeling MeasuringTime/SolvingTime/SystemTime docstrings

### 11. Language surfaces — which languages, native vs FFI vs subprocess — partial

PYTHON IS THE ONLY USER-FACING SURFACE for all five, but the cores differ and one has a real
second surface. OMMX is the exception and the one directly relevant to a Rust stack: a genuine
Rust crate `ommx` on crates.io — v2.6.1 stable, v3.0.0-beta.3 in flight, MIT OR Apache-2.0,
45,079 downloads — with the Python binding built on top via the `ommx-pyo3-bridge` crate.
Because the format is Protocol Buffers (`ommx.v1.Instance`, with `instance_pb2`,
`constraint_pb2`, `function_pb2`, `polynomial_pb2`, `quadratic_pb2`, `linear_pb2`,
`one_hot_pb2`, `sos1_pb2` all present in the wheel), any language with a protobuf compiler can
read and write OMMX without going through Python. OpenJij is C++ at the core (`openjij.cxxjij`
extension module) with the header-only `cimod`/`jij-cimod` library usable directly from C++.
Amplify is a compiled C++ core (`amplify.cpython-39-darwin.so`) with a Python-only public
surface and no source. JijModeling is a RUST core compiled to `_jijmodeling.abi3.so`, again
Python-only and closed. PyQUBO is a C++ core (`cpp_pyqubo`) with Python bindings. All bindings
are in-process FFI, not subprocess. NOT FOUND: any JavaScript, WebAssembly or browser surface
for any of the five; no C API is documented for Amplify or JijModeling.

> https://crates.io/api/v1/crates/ommx (queried) ; https://github.com/Jij-Inc/ommx ; site-packages binary inspection of amplify, jijmodeling, openjij, pyqubo

### 12. Agent/AI surfaces — MCP, HTTP API, structured tool schemas — partial

WEAKEST LAYER RELATIVE TO THE CLUSTER'S SOPHISTICATION ELSEWHERE. MCP: this review did not
locate an MCP server for Fixstars Amplify, JijZept, JijModeling, OMMX, OpenJij or PyQUBO —
targeted searches returned only generic MCP literature with no hits on these projects, and no
MCP module is present in any installed wheel. Structured tool schemas for LLM function calling:
not located. HTTP: Amplify Cloud IS an HTTP service and the SDK is its client — clients expose
`.token`, `.url`, `.proxy`, `.write_request_data`, `.write_response_data`, and `solve()` without
a token raised `RuntimeError: 401: Unauthorized`, confirming remote HTTP transport — but the
supported surface is the Python SDK, not a documented public REST contract an agent could call
directly. JijZept likewise runs as a cloud service behind the SDK. What IS unusually agent-
ready, though never marketed that way: (a) OMMX Artifact, an OCI-standard container format —
`ArtifactBuilder.new/new_archive/for_github/temp`, `.add_instance/.add_solution/.add_sample_set/
.add_parametric_instance/.add_dataframe/.add_ndarray/.add_json/.add_annotation/.build`,
`Artifact.push/.load/.get_instance/.get_solution/.get_layer_descriptor/.annotations` — meaning a
problem, its data, its solutions and their provenance annotations can be pushed to and pulled
from any container registry including GHCR; (b) JijModeling `Problem.get_problem_schema()`,
which returns a machine-readable schema of the model's own placeholders, and
`Problem.generate_random_instance()` / `generate_random_dataset()`, which let a program
synthesise well-formed data for a model it did not write. Those two are the natural attachment
points for an agent surface that nobody in this cluster has yet built.

> searches for MCP across all five projects returned no project hits ; ommx.artifact introspection ; jijmodeling Problem member introspection ; amplify FixstarsClient 401 measured

### 13. Training — energy-based model training, gradient estimators — **not found**

NOT FOUND in any of the five. A precise grep for
contrastive_divergence|backprop|\.backward\(|autograd|torch|jax\.|gradient_estimator across
amplify, jijmodeling, ommx, openjij and pyqubo returned ZERO hits, and none of the five declares
a deep-learning framework dependency. (An earlier looser grep for 'train' appeared to hit — that
was the substring inside 'consTRAINt', a false positive worth recording.) These are OPTIMISATION
stacks, not energy-based-model learning stacks: the Ising machine is used to MINIMISE a hand-
written or compiler-generated Hamiltonian, never to draw the negative-phase samples that would
train one. The nearest adjacent things, and neither is EBM training: OpenJij's documentation
includes a QBoost tutorial that uses quantum annealing to select weak classifiers for an
ensemble — annealing applied TO machine learning, not learning a Boltzmann distribution; and
Qamomile performs classical outer-loop optimisation of variational QAOA/QRAO circuit parameters,
which is parameter fitting over a fixed ansatz rather than a gradient estimator for an energy
model. No Boltzmann-machine trainer, no contrastive divergence, no persistent-contrastive-
divergence, no score matching, no reparameterised or REINFORCE-style estimator located anywhere
in the cluster.

> grep over the five installed packages ; https://openjij.github.io/OpenJij/tutorial/en/machine_learning/qboost.html ; https://github.com/Jij-Inc/Qamomile

### 14. Visual/graph programming — **not found**

NOT FOUND in any of the five. No node-graph editor, visual model builder, block-based composer
or drag-and-drop canvas is present in amplify, jijmodeling, ommx, openjij or pyqubo, and none is
documented on the Fixstars Amplify or JijZept product pages this review examined. The cluster's
entire authoring story is textual Python. Two things sit adjacent to visual work without being
it: JijModeling renders models to LaTeX for display (every variable, placeholder and element
carries a `set_latex()` method and models render as mathematical notation in Jupyter), which is
a READ-ONLY visual output rather than a programming surface; and amplify-benchmark ships a
report/dashboard viewer for benchmark results, which visualises outcomes rather than composing
models. Given that JijModeling's whole design premise is an abstract AST over indexed variables
— precisely the structure a node graph would represent — this is a conspicuous empty lane.

> package introspection of all five ; https://www.jijzept.com/en/products/sdk/ ; https://amplify.fixstars.com/en/product

### 15. Licence, openness, and whether the hardware is required to use it — partial

THE CLUSTER IS SPLIT DOWN THE MIDDLE — the open half is Jij's and Recruit's engines and formats,
the closed half is the two sophisticated modelling layers. OPEN, no hardware, runs locally:
OpenJij 0.11.6 Apache-2.0 (Jij Inc.); jij-cimod 1.7.4 Apache-2.0; PyQUBO 1.5.0 Apache-2.0
(Recruit Co., Ltd., github.com/recruit-communications/pyqubo); OMMX MIT OR Apache-2.0 dual, both
Python and Rust; amplify-benchmark MIT. CLOSED: JijModeling 1.14.2 declares `Classifier: License
:: Other/Proprietary License` and ships only as a compiled `_jijmodeling.abi3.so` — free to `pip
install`, no hardware needed to build and lower a model (the `Interpreter` runs locally and
emits an OMMX Instance you can then solve with open-source OpenJij), but you cannot read, fork
or audit the compiler. Amplify SDK 1.3.1 declares its licence literally as 'Terms of Use'
(Fixstars Amplify Corp.); the shipped LICENSE.txt states the user 'acknowledges that agreeing to
this license does not entitle them to receive provision or disclosure of source code for any
part of Fixstars Amplify SDK', offers a restricted Free plan and an unrestricted Paid plan, and
reserves the right to change the terms. HARDWARE/CLOUD REQUIRED FOR AMPLIFY: measured —
`solve(model, FixstarsClient())` without a token raised `RuntimeError: 401: Unauthorized`.
Modelling and lowering are local (`solve(model, client, dry_run=True)` succeeded offline and
returned a `Result`), but every actual sample is a cloud call to Fixstars or a vendor machine.
So the most sophisticated modelling layer in the field is closed-source and its solver is
unreachable without an account, while the open engine (OpenJij) has no modelling layer at all —
the gap OMMX exists to bridge.

> pip metadata for all five ; .../amplify-1.3.1.dist-info/LICENSE.txt read directly ; .../jijmodeling-1.14.2.dist-info/METADATA ; FixstarsClient 401 and dry_run measured

**Notes.** DIRECT ANSWERS TO THE TWO QUESTIONS ASKED.  (1) WHAT JIJMODELING AND AMPLIFY OFFER THAT A RAW
QUBO LIBRARY DOES NOT — four things, all verified by execution, not just read in docs:  (a)
ABSTRACT INDEXED VARIABLES OVER UNBOUND DATA. JijModeling's `Placeholder` + `Element` + `forall`
means the model is written ONCE, independent of instance size, and
`Interpreter(data).eval_problem(p)` compiles it at any n. A raw QUBO library forces you to write
the Python loop that emits coefficients, so the model and the data are fused and the size is
baked in. Amplify takes the other route — eager NumPy-style `PolyArray` with
`einsum`/`matmul`/slicing over `Dim0..Dim4` — which is less abstract but far more ergonomic than
dict-of-tuples.  (b) NAMES THAT SURVIVE THE LOWERING AND COME BACK. This is the property that
most distinguishes the cluster. I fed a deliberately infeasible assignment solution through
JijModeling → OMMX → evaluation and got back `onehot_row[0] eval=1.0 feasible=False`,
`onehot_row[1] feasible=True`, `onehot_row[2] eval=-1.0 feasible=False`. The constraint name AND
its forall subscript round-tripped through quadratization, penalty folding, sampling and decode.
PyQUBO does the same at flat scope: `.constraints(only_broken=True)` → `{'exactly_one': (False,
1.0)}`. A raw QUBO library hands you a bit vector and an energy scalar and leaves the
attribution to you.  (c) AUTOMATIC LOWERING AGAINST A DECLARED DEVICE ENVELOPE. Amplify's
`client.acceptable_degrees` is the single most interesting artefact I found — a per-vendor
matrix over {Binary, Ising, Integer, Real} × {Zero, Linear, Quadratic, Cubic, Quartic,
HighOrder}, given separately for objective, equality constraints and inequality constraints.
`solve()` reads it and applies exactly the needed transforms. Measured differences are real and
consequential: FujitsuDA4Client declares native Binary:Linear INEQUALITY support, so a linear
inequality is passed through rather than converted to a slack-and-penalty; FixstarsClient
declares Zero for all constraints, so the same inequality gets encoded; HitachiClient is Ising-
only and additionally carries a `.graph` topology; GurobiClient accepts quadratic Integer and
Real constraints outright. One model, one `solve()` call, and the compiler's behaviour changes
per machine.  (d) MULTI-BACKEND DISPATCH. Amplify: ~15 vendor clients spanning Fixstars AE,
D-Wave, Fujitsu DA3c/DA4, Toshiba SQBM+ v2, NEC VA2, Hitachi CMOS, Gurobi, plus gate-model. Jij:
the OMMX adapter set (openjij, pyscipopt, highs, python-mip, gurobi, dwave, da4, fixstars-
amplify) behind `SolverAdapter`/`SamplerAdapter` ABCs.  (2) WHAT OMMX IS AND WHAT IT SOLVES.
OMMX (Open Mathematical prograMming eXchange) is Jij's protobuf-defined intermediate
representation and OCI-based artifact format, MIT OR Apache-2.0, shipping as BOTH a Python SDK
and a Rust crate (crates.io `ommx` v2.6.1 stable, v3.0.0-beta.3 in flight, 45,079 downloads). It
solves three problems at once. First, the N×M adapter explosion: N modellers × M solvers becomes
N+M adapters against one IR. Second, LOSS OF STRUCTURE ACROSS THE HANDOFF — and this is the
subtle one. `ommx.v1.ConstraintHints` carries `one_hot_constraints` and `sos1_constraints` as
machine-readable structural facts; I built a plain `sum(j, x[i,j]) == 1` model and OMMX AUTO-
DETECTED it, emitting `OneHot(id=0, variables=[0,1,2])`, `OneHot(id=1, variables=[3,4,5])`,
`OneHot(id=2, variables=[6,7,8])` without being told. A solver that can exploit one-hot
structure is handed that fact instead of having to rediscover it from an anonymised penalty
matrix. Third, PROVENANCE: OMMX Artifact packages instance + parameters + solutions + sample
sets + dataframes + annotations into an OCI container pushable to any registry including GHCR,
so a result is transportable with the problem that produced it.  THE STRUCTURAL FACT WORTH
CARRYING FORWARD: OMMX is not a side-format Jij also publishes — it IS JijModeling's compiler
IR. I verified this directly: `type(jm.Interpreter(data).eval_problem(problem))` is literally
`ommx.v1.Instance`. So Jij open-sourced the IR and the engine (OpenJij) under permissive
licences while keeping the modelling front-end (JijModeling) proprietary. Fixstars kept the
entire stack closed and additionally gated the solver behind a cloud token.  HOW THIS BEARS ON
FERROTHERM (positions, not verdicts). Four lanes in this cluster are genuinely empty and none is
empty for lack of engineering capacity — they are empty because this cluster's binding
constraint is COMMERCIAL POSITIONING around cloud/hardware access, not algorithms:  • SAMPLING
VERIFICATION. Zero hits for effective temperature, ESS, autocorrelation time, TV-vs-exact or KL
across all five packages. Everything measured is solution quality (TTS, success probability,
residual energy — creditably WITH standard errors in `openjij.utils.benchmark`). Nothing asks
whether the sampler draws from the Boltzmann distribution it claims. Ferrotherm's certificate
work has no competitor here.  • JOULES. Zero hits for joule|watt|kwh|power_consum across all
five. Wall-clock is exhaustively instrumented — JijZept itemises timing down to
`deserialize_solution` — but energy is absent everywhere. Note the asymmetry this creates: a
cloud-gated QPU stack has structural reason NOT to publish joules per solve.  • AGENT SURFACES.
No MCP server located for any of the six projects. The attachment points already exist and are
unused: `Problem.get_problem_schema()`, `generate_random_instance()`, and OMMX Artifact's
registry push.  • VISUAL/NODE-GRAPH. Absent, despite JijModeling's AST being exactly the
structure a node graph would render. Two lanes are NOT empty and it would be an error to claim
them: capability declaration (Amplify's `acceptable_degrees` is more precise than most stacks
manage) and named-constraint feasibility attribution (OMMX and PyQUBO both do this well). Also
worth noting for a Rust stack specifically: OMMX ALREADY OCCUPIES the Rust IR lane under
MIT/Apache-2.0, so the honest framing is interoperate-with rather than displace — an OMMX
adapter is probably a cheaper route to the cluster's model corpus than a competing IR.  METHOD
AND CAVEATS. All five SDKs were pip-installed and introspected/executed rather than read about:
amplify 1.3.1, jijmodeling 1.14.2, ommx 2.0.12, openjij 0.11.6, pyqubo 1.5.0, jij-cimod 1.7.4,
plus ommx-openjij-adapter and (on a separate Python 3.12 venv, since they need >=3.10) minto.
Venv at /private/tmp/claude-501/-Users-dcharlot-vibe-coding-bmi-
concept/ec64a91f-fbcc-442a-8e9b-f2f378c7a081/scratchpad/qv (and qv2). Scripts retained:
t_jij.py, t_amp3.py through t_amp6.py, t_ommx.py, t_rt.py, t_rt2.py, t_hints.py, t_oj2.py,
t_pq.py, t_caps.py, t_art.py, t_ver.py. CAVEATS: (i) every Amplify claim about SOLVER behaviour
is from documentation and capability metadata, never from an executed cloud solve, because I
have no token — the 401 is itself the evidence for the gating claim but it means no Amplify
result quality was verified; (ii) `qamomile` has no PyPI distribution reachable from here, so
Qamomile claims are from its GitHub/PyPI pages, not introspection; (iii) OpenJij's CPU-only
finding is specific to the 0.11.6 wheel on darwin-arm64 — earlier OpenJij carried Chimera GPU
classes, so read that as removed-or-not-built here, not as never-existed; (iv) I did not verify
MINTO's licence; (v) absence claims throughout are scoped as 'this review did not locate X in
the installed packages and the documentation examined', which is the correct strength — a
private roadmap or an unreleased branch could hold any of them.

## Open-source / general-purpose Ising-QUBO layer: qubovert 1.2.5, PyQUBO 1.5.0, OpenJij 0.11.6, dimod 0.12.21 (standalone), Google OR-Tools 9.15.6755 (CP-SAT + MathOpt, non-Ising control), NVIDIA cuOpt 26.08/26.10 (non-Ising control), plus the Rust crates: ommx 3.0.0-beta.3, rustqubo 0.1.0, quantrs2-anneal 0.2.0, quantrs2-tytan 0.2.0, hercules 0.5.0, separatrix 0.1.0, problemreductions 0.6.0, annealers 0.1.0. All Python packages installed into a venv and interrogated by execution; all Rust crates downloaded from crates.io, read, and where load-bearing compiled and run. Sources under /private/tmp/claude-501/-Users-dcharlot-vibe-coding-bmi-concept/ec64a91f-fbcc-442a-8e9b-f2f378c7a081/scratchpad/{py,crates,buildtest,qtest}.

### 1. Modelling layer (named variables, domains, constraints, objective; answers by name) — **yes**

dimod: BEST-IN-CLASS and open. `dimod.ConstrainedQuadraticModel` + symbolic
`Binary/Spin/Integer/Real` + `quicksum`; `set_objective`, `add_constraint(expr <= 7,
label='capacity')`, `add_discrete`. RAN IT: labels ['capacity','pick_one_of_ab'], answer
returned keyed by name {'take_a':0.0,'take_b':1.0,'take_c':0.0}. PyQUBO: `Binary('a')`,
`Constraint(expr, label=...)`, `Placeholder`, `Array`; `compile()` -> `Model`;
`decode_sampleset` returns names. qubovert: `boolean_var('x')`/`spin_var`, `PCBO`/`PCSO` carry
constraints; answers by name. OMMX (Rust): `Instance` with `DecisionVariable` of
`Kind::{Continuous,Integer,Binary,SemiContinuous,SemiInteger}`, bounds, names + subscripts,
`Constraint` collections, objective. rustqubo (Rust):
`Expr::{Binary,Spin,Number,Constraint{label,expr},WithPenalty,Placeholder}` generic over label
type; RAN IT, returned {"b":true,"c":true,"a":false}. OpenJij: NOT FOUND — `dir(openjij)` has
zero constraint/variable-declaration API; it is a sampler that eats dicts/BQMs. Its modelling
layer JijModeling is a separate PROPRIETARY package (PyPI classifier 'License ::
Other/Proprietary License').

> https://github.com/dwavesystems/dimod ; https://github.com/recruit-communications/pyqubo ; https://github.com/jtiosue/qubovert ; https://github.com/Jij-Inc/ommx (src/decision_variable.rs:76 Kind enum) ; https://github.com/yasuo-ozu/rustqubo (src/expr.rs:12) ; https://pypi.org/project/jijmodeling/ ; verified by running scratchpad/py/run_dimod.py, run_pyqubo.py, run_qv.py, run_oj.py and scratchpad/buildtest/src/bin/demo.rs

### 2. Encodings (one-hot, domain-wall, binary/log; choice exposed) — partial

PyQUBO: choice exposed as four distinct CLASSES — `OneHotEncInteger`, `LogEncInteger`,
`UnaryEncInteger`, `OrderEncInteger` (all instantiated successfully). DOMAIN-WALL: not found in
pyqubo's exports. OMMX (Rust): choice exposed as instance-level passes — `Instance::log_encode`,
`log_encode_all_used_integers` (src/instance/log_encode.rs:182,232), `unary_encode`
(unary_encode.rs:74), `convert_one_hot_to_constraint` (one_hot.rs:28); domain-wall NOT FOUND.
quantrs2-tytan: richest ENUM —
`EncodingScheme::{OneHot,Binary,GrayCode,DomainWall,Unary,OrderEncoding,Direct}`
(src/encoding.rs:14) — the only stack surveyed that names domain-wall; but see layer 5, its
penalty path is self-documented as incomplete. dimod: integers are first-class (`Integer`), so
the user never picks an encoding at model time; the encoding is chosen inside `cqm_to_bqm` and
is not user-selectable. qubovert: `integer_var(prefix, num_bits, log_trick=True)` — binary/log
only, exposed via one boolean. rustqubo, hercules, separatrix, OpenJij: NOT FOUND, no integer-
encoding layer at all.

> scratchpad/py/run_pyqubo.py output ; scratchpad/crates/ommx/ommx-3.0.0-beta.3/src/instance/{log_encode,unary_encode,one_hot,slack}.rs ; scratchpad/crates/quantrs2-tytan/quantrs2-tytan-0.2.0/src/encoding.rs:14-30

### 3. Higher-order reduction (k-body lowered to pairwise, with ancillas) — **yes**

This is the best-covered layer in the ecosystem. dimod: `dimod.higherorder.BinaryPolynomial` +
`make_quadratic(poly, strength, vartype)`; RAN IT on {('p','q','r'):1} -> variables
['p','q','q*p','r'] — ancilla named after the pair it replaces. PyQUBO: automatic inside
`compile()`; RAN a*b*c -> QUBO keys include ('a','a * b'), ancilla 'a * b'. qubovert:
`PUBO.to_qubo()`; RAN {('x','y','z'):-1} -> index 3 is an ancilla, and `P.convert_solution(sol)`
correctly strips it and returns {'x':1,'y':1,'z':1}; `PCBO.num_ancillas` and
`remove_ancilla_from_solution` expose them by name ('__a0','__a1'). rustqubo:
`CompiledModel::reduce_order(2)` called automatically by `Expr::compile()`
(src/compiled.rs:186), greedy max-count subset substitution with a generated penalty constraint.
quantrs2-anneal:
`HoboProblem::to_qubo(ReductionMethod::{SubstitutionMethod,MinimumVertexCover,BooleanProduct})`
returning `QuboReduction` with `auxiliary_vars: Vec<AuxiliaryVariable>` and
`extract_original_solution` (src/hobo.rs:146,330). OpenJij: sidesteps reduction entirely —
`SASampler.sample_hubo` / `sample_huio` sample k-body directly. NOT FOUND: hercules, separatrix,
ommx (ommx has reduce_binary_power but its target class is quadratic instances, not general
k-body lowering).

> scratchpad/py/run_dimod.py, run_pyqubo.py, qv2.py outputs ; scratchpad/crates/rustqubo/rustqubo-0.1.0/src/compiled.rs:186 ; scratchpad/crates/quantrs2-anneal/quantrs2-anneal-0.2.0/src/hobo.rs:146

### 4. Constraint vocabulary (equality, inequality/slack, cardinality, exactly-one, all-different) — partial

dimod: equality + inequality via natural Python comparison on symbolic expressions,
`add_discrete` = exactly-one over a named set. All-different: NOT FOUND (must be hand-built from
one-hots). qubovert: widest boolean vocabulary — `add_constraint_{eq,ne,lt,le,gt,ge}_zero` plus
gate constraints `add_constraint_{AND,OR,NAND,NOR,XOR,XNOR,NOT,BUFFER}` and
`add_constraint_eq_*` variants. All-different / cardinality: NOT FOUND as named primitives.
PyQUBO: `Constraint(expr,label)` is a wrapper around any expression — inequality has no first-
class form; logic gates `AndConst/OrConst/NotConst/XorConst`; all-different NOT FOUND. OMMX
(Rust): equality + inequality with senses, plus SPECIAL kinds
`SpecialConstraintKind::{Indicator,OneHot,Sos1}` (src/instance.rs:71) and real slack machinery:
`convert_inequality_to_equality_with_integer_slack`, `add_integer_slack_to_inequality`
(src/instance/slack.rs:61,174). All-different: NOT FOUND. OR-Tools CP-SAT (the control): by far
the richest — `add_all_different`, `add_exactly_one`, `add_at_most_one`, `add_at_least_one`,
`add_cumulative`, `add_circuit`, `add_automaton`, `add_element`, `add_no_overlap`,
`add_reservoir_constraint`, `add_allowed_assignments`. NOTHING in the Ising ecosystem comes
close. quantrs2-tytan declares the vocabulary as data (`GlobalConstraint::{AllDifferent,Cumulati
ve,GlobalCardinality,Regular,Element,Table,Circuit,BinPacking}`, src/constraints.rs:14) but I
found NO lowering from `GlobalConstraint` to a QUBO — grepping `GlobalConstraint` outside
constraints.rs returns zero hits, so the enum is only consumed by a CP-style
`AllDifferentPropagator` and by `ConstraintLibrary::{n_queens,graph_coloring,sudoku}` which
return the enum again.

> scratchpad/py/qv2.py, run_ortools.py outputs ; scratchpad/crates/ommx/ommx-3.0.0-beta.3/src/instance.rs:71 and src/instance/slack.rs ; scratchpad/crates/quantrs2-tytan/quantrs2-tytan-0.2.0/src/constraints.rs:14-46

### 5. Penalty handling (auto scaling, feasibility check, reporting WHICH constraint broke) — partial

dimod: STRONGEST. RAN IT — `cqm.iter_constraint_data(sample)` yields per-constraint (label,
lhs_energy, rhs_energy, sense, violation): 'capacity lhs=12.0 rhs=7.0 sense=Sense.Le
violation=5.0'; plus `violations()`, `check_feasible()`, `num_soft_constraints`.
`cqm_to_bqm(cqm, lagrange_multiplier=...)` auto-generated 3 log-encoded slack bits for the <=7
row (vars slack_v8b446...0/1/2). PyQUBO: `DecodedSample.constraints(only_broken=True)` — RAN IT,
returned {'exactly_one_ab': (False, 1.0)}, i.e. label + broken flag + violation energy. Penalty
scaling is manual-but-symbolic via `Placeholder` and `feed_dict`. qubovert:
`is_solution_valid(sol)` is a BOOLEAN ONLY; constraints are stored in a dict keyed by TYPE
('eq','le') with list indices — there is NO label parameter on any `add_constraint_*` method (I
checked every signature: zero contain 'label'), so the best you can do is iterate
`H.constraints['le'][i].value(sol)`. Auto-scaling NOT FOUND — `lam=1` default with a
`default_lam`. rustqubo: `solve_with_constraints()` returns a third element listing UNSATISFIED
CONSTRAINT LABELS; RAN IT, returned []. OMMX: per-constraint feasibility as a first-class type —
`EvaluatedConstraint`, `SampleSet::{feasible_ids,feasible_relaxed_ids,is_sample_feasible}`,
`Solution::feasible_relaxed()` aggregating over regular/indicator/one-hot/SOS1 collections;
penalty scaling via `Instance::{penalty_method, uniform_penalty_method,
penalty_method_with_fixed_weights}` which return a ParametricInstance whose weight is a named
parameter. quantrs2-anneal: penalty is a HARDCODED LITERAL — `let penalty_weight = if
constraint.is_hard { 1000.0 }` (src/dsl.rs). CRITICAL, REPRODUCED:
`OptimizationModel::compile_to_qubo` handles only True/False/Equal/ExactlyOne/AtMostOne/And;
every inequality falls through a `_ =>` catch-all. I built and RAN it: `less_than_or_equal(...)`
is ACCEPTED by `add_constraint` (summary reports num_constraints:1) and then `compile_to_qubo`
returns Err('Unsupported constraint type in QUBO compilation'), while ExactlyOne compiles fine.
The sibling `csp_compiler.rs` is the same — `add_linear_constraint` is a stub whose body reads
'// This would require implementing slack variables and penalty methods / For now, return
unsupported'. quantrs2-tytan's `constraints_to_penalties` carries the comment '// This is a
simplified version / Real implementation would handle inequality properly'.

> scratchpad/py/run_dimod.py, run_pyqubo.py, qv2.py outputs ; scratchpad/qtest/src/main.rs (built and run) ; scratchpad/crates/quantrs2-anneal/quantrs2-anneal-0.2.0/src/dsl.rs:583-643 and src/csp_compiler.rs:653-667 ; scratchpad/crates/quantrs2-tytan/quantrs2-tytan-0.2.0/src/constraints.rs:512-548 ; scratchpad/crates/ommx/ommx-3.0.0-beta.3/src/instance/penalty.rs:77,183,246,359

### 6. Embedding / placement onto hardware topology — partial

Open and standalone: `minorminer` 0.2.19 (Apache-2.0, pip-installable with no hardware and no
D-Wave account) — `minorminer.find_embedding(S, T, **params)`; topology generators live in
`dwave-networkx`. dimod itself has NO embedder; it supplies the ABSTRACTION for one:
`dimod.Structured` (nodelist/edgelist/adjacency/to_networkx_graph/valid_bqm_graph) and
`StructureComposite`. The actual `EmbeddingComposite` lives in `dwave-system`, which is
installable but only useful against a Leap QPU. quantrs2-anneal (Rust): a genuine native
embedder — `HardwareGraph::new_chimera(m,n,t)`, `HardwareTopology`,
`Embedding::{add_chain,verify}`, `MinorMiner::find_embedding`
(src/embedding.rs:104,230,250,327), plus layout_embedding.rs, multi_chip_embedding.rs,
chain_break.rs, flux_bias.rs. NOT FOUND: qubovert, PyQUBO, OpenJij, OMMX, rustqubo, hercules,
separatrix, OR-Tools, cuOpt — none place onto a fixed sparse topology.

> scratchpad/py (pip show minorminer -> Apache License, find_embedding signature verified) ; https://github.com/dwavesystems/minorminer ; scratchpad/crates/quantrs2-anneal/quantrs2-anneal-0.2.0/src/embedding.rs

### 7. Samplers / solvers (algorithms, CPU/GPU/hardware) — **yes**

OpenJij: C++ core (cxxjij.cpython-39-darwin.so) — `SASampler`, `SQASampler` (path-integral
quantum SA), `CSQASampler`; methods sample_ising/qubo/hubo/huio/quio. RAN it; `response.info`
carries sampling_time, execution_time, per-read list_exec_times and the schedule {beta_min,
beta_max, num_sweeps}. dimod reference samplers are DELIBERATELY TOY: `ExactSolver`,
`ExactCQMSolver`, `ExactPolySolver`, `RandomSampler`, `NullSampler`, `IdentitySampler`, and a
`SimulatedAnnealingSampler` (real annealing lives in `dwave-neal`/`dwave-samplers`). qubovert:
`qubovert.sim.anneal_{qubo,quso,pubo,puso}` backed by a C extension (_canneal...so) +
`solve_bruteforce`. PyQUBO ships no sampler; it hands a BQM to dimod/neal. hercules (Rust,
BSD-3): the most serious open Rust QUBO SOLVER — branch & bound with Clarabel LP/QP subproblem
solvers, presolve, variable probing, warm starting, k-opt, multithreaded; author states it 'can
generally solve dense and sparse problems below 80 binaries'. separatrix (Rust, MIT): ballistic
+ discrete Simulated Bifurcation, SA, parallel tempering, Gray-code exact ground truth refusing
n>26. quantrs2-anneal/tytan (Rust, Apache-2.0): SA, SQA/path-integral, population annealing,
coherent Ising machine, tabu, GA, simulated bifurcation, GPU samplers, and hardware clients for
D-Wave, Fujitsu DA, Hitachi CMOS, NEC, Amazon Braket, Azure Quantum, IBM, FPGA. rustqubo's
default `SimpleSolver` is UNRELIABLE at trivial size: I ran min(-p·q·r) ten times and it
returned energy 0 instead of -1 on 4/10 trials, with no repetition/confidence metadata in the
return type. OR-Tools: CP-SAT (LCG); MathOpt dispatches to GLOP, PDLP, HIGHS, SCIP, GUROBI,
CP_SAT, OSQP, ECOS, SCS, GLPK. cuOpt: GPU LP/QP/QCQP/SOCP/MIP + TSP/VRP/PDP.

> scratchpad/py/run_oj.py, run_dimod.py, qv2.py, run_ortools.py outputs ; scratchpad/buildtest/src/bin/hobo.rs (built and run, 10 trials) ; https://github.com/DKenefake/hercules/ ; scratchpad/crates/separatrix/separatrix-0.1.0/README.md ; scratchpad/crates/quantrs2-tytan/quantrs2-tytan-0.2.0/src/sampler/hardware/

### 8. Device abstraction (one interface over multiple vendors, capability declaration) — partial

TWO real designs exist, and they solve different halves. (a) dimod's `Sampler` ABC is the de-
facto interface the whole annealing ecosystem implements — `sample`/`sample_ising`/`sample_qubo`
plus `.parameters` and `.properties` dicts for capability declaration, `Structured` for
topology, and ~16 Composites for cross-cutting behaviour. WEAKNESS I MEASURED: the declaration
is by convention only, not schema — `SimulatedAnnealingSampler().properties` is `{}` and
`.parameters` is `{'num_reads': [], 'beta_range': [], 'num_sweeps': []}`, i.e. keys with empty
value-lists; `ExactSolver` declares both as `{}`. Nothing states a variable-count limit, a
coupling range, or a topology. (b) OMMX is the vendor-neutral EXCHANGE format, and it is the
only thing spanning MIP solvers AND Ising machines: adapters published on PyPI for OpenJij,
SCIP, python-mip, HiGHS, D-Wave, Gurobi, Fixstars Amplify and Fujitsu DA4 (all HTTP 200; ommx-
qiskit-adapter 404 = not found). OMMX also has `InstanceClass` with `DegreeBound`,
`allowed_variable_kinds`, `allows_one_hot`, `allows_sos1`, `allowed_senses` — a real machine-
readable capability declaration, which is exactly what dimod's `.properties` is missing.
quantrs2-tytan has a single `Sampler` trait implemented by ~19 backends including 8 hardware
vendors, but capability is declared ad hoc (e.g. Fujitsu's `self.max_variables` checked inline).
NOT FOUND: qubovert, PyQUBO, rustqubo, hercules, separatrix.

> scratchpad/py/probe_dimod.py output and inline dimod capability probe ; PyPI HTTP status sweep of ommx-*-adapter ; scratchpad/crates/ommx/ommx-3.0.0-beta.3/src/instance_class.rs:30-224 ; https://github.com/Jij-Inc/ommx

### 9. Verification (sampling certificates, effective temperature, ESS, TV vs exact, conformance) — **not found**

NOT FOUND as a sampling-correctness layer anywhere in this stack. I grepped every installed
package (dimod, pyqubo, qubovert, openjij) for
effective_temperature|effective_sample|total_variation|autocorrel|gelman|r_hat: ZERO hits in all
four. What DOES exist is adjacent and should not be mistaken for it: (i) OpenJij ships a SOLVER-
BENCHMARK layer — `openjij.utils.{solver_benchmark, time_to_solution, success_probability,
residual_energy, se_upper_tts, se_lower_tts, se_success_probability, se_residual_energy}`; TTS =
tau*log(1-p_r)/log(1-p_s). That measures whether an optimiser finds a known ground state, not
whether a sampler reproduces the Boltzmann distribution, and it REQUIRES the true solution as an
input argument. (ii) separatrix ships exact enumeration to n<=26 as ground truth and reports
optimality gap, with an explicit 'honesty contract' in its README refusing quantum-advantage
claims — the most disciplined verification posture I found, but it is optimality-gap, not
distributional. (iii) quantrs2-anneal has `effective_sample_size` but only as an internal
resampling trigger inside population_annealing.rs, never surfaced as a diagnostic; the
`total_variation` hit in quantrs2-tytan is a path-length accumulator in a convergence PLOT, not
a distance to a reference distribution. (iv) dimod `sampleset.info` came back `{}` from
SimulatedAnnealingSampler — no schedule, no acceptance rate, no temperature. NOT FOUND anywhere:
measured effective temperature, TV distance against exact enumeration, ESS as a reported
statistic, or a conformance suite a third-party sampler must pass.

> grep sweep over scratchpad/py/venv/lib/python3.9/site-packages/{dimod,pyqubo,qubovert,openjij} ; scratchpad/py/verif.py output (openjij.utils listing + docstrings) ; scratchpad/crates/quantrs2-anneal/quantrs2-anneal-0.2.0/src/population_annealing.rs:189,450 ; scratchpad/crates/separatrix/separatrix-0.1.0/README.md

### 10. Energy / cost accounting (joules or price per operation) — **not found**

NOT FOUND in every stack surveyed. I grepped all five installed Python packages for
joule|watt|kwh|energy_consum|power_draw|nvml|carbon: ZERO matching files in dimod, pyqubo,
qubovert, openjij, ortools. Note the word 'energy' is everywhere in these libraries but always
means the OBJECTIVE VALUE of a spin configuration, never a physical joule — that collision is
worth naming explicitly. ommx, rustqubo, hercules, separatrix, problemreductions: zero hits. The
one apparent exception is a FALSE POSITIVE and I verified it: quantrs2-tytan's gpu_benchmark.rs
declares `EnergyMetrics { avg_power, energy_per_sample, perf_per_watt }` with the field
documented '/// Energy per sample (joules)', but the function body reads '// This would require
GPU power monitoring capabilities / // Using placeholder values for demonstration' followed by
`let avg_power = 150.0; // Watts` — a hardcoded literal multiplied by wall time. Its sibling
benchmark/metrics.rs and benchmark/runner.rs set `power_consumption: None`. quantrs2-anneal has
one `power_consumption_estimates: Vec<f64>` field and one '/// Energy consumption (Joules)' doc
comment, both in types-only files. Nothing in this ecosystem reads a power rail, an
RAPL/IOReport counter, or NVML. Nothing prices a solve.

> grep sweep over scratchpad/py/venv/lib/python3.9/site-packages/{dimod,pyqubo,qubovert,openjij,ortools} and scratchpad/crates/*/*/src ; scratchpad/crates/quantrs2-tytan/quantrs2-tytan-0.2.0/src/gpu_benchmark.rs:361-390 (verbatim body read)

### 11. Language surfaces (which languages; native vs FFI vs subprocess) — partial

The Ising/QUBO modelling ecosystem is PYTHON-ONLY at the surface, with C++ underneath. Measured
compiled-extension counts in the venv: dimod 12 .so, ortools 18, openjij 1 (cxxjij) + jij-cimod,
qubovert 1 (sim/_canneal), pyqubo 0 in-package but a top-level cpp_pyqubo.cpython-39-darwin.so.
So all five are Python-facing FFI over C/C++; NONE of them expose a C ABI, a CLI, or a non-
Python binding as a supported surface. Rust natives: ommx is genuinely multi-surface — Rust
crate, `ommx-pyo3-bridge`, a Python package, and a CLI binary (src/bin/ommx.rs) — and reads MPS
(`mps::parse/load/save`) and QPLIB (`qplib::parse/load`), which are the real cross-language
lingua franca. hercules is Rust + PyO3 (src/python_interopt.rs). rustqubo is Rust + PyO3
(src/python.rs) but see the packaging defect below. quantrs2-* are Rust-only. NOT FOUND across
the whole surveyed set: any JS/WASM, Julia, Zig, or browser surface; and no HTTP/subprocess
surface except cuOpt's server.

> find over scratchpad/py/venv/lib/python3.9/site-packages for *.so ; scratchpad/crates/ommx/ommx-3.0.0-beta.3/src/{bin/ommx.rs,mps.rs,qplib.rs} ; scratchpad/crates/hercules/hercules-0.5.0/src/python_interopt.rs

### 12. Agent / AI surfaces (MCP, HTTP API, structured tool schemas) — **not found**

NOT FOUND for every Ising/QUBO stack surveyed. Grepped dimod, pyqubo, qubovert, openjij and
ortools for 'model context protocol|modelcontextprotocol|mcp_server': ZERO hits in all five. No
HTTP API in any of them (D-Wave's `dwave-cloud-client` is a REST client to Leap, i.e. a client
of someone else's hardware endpoint, not an API these libraries serve). No structured tool
schemas. The only agent surface in the whole survey is on the NON-Ising control: NVIDIA cuOpt
ships agent skills at repo root (AGENTS.md, skills/, .claude-plugin/marketplace.json, .cursor-
plugin, .opencode, .windsurf) and a Server API (REST/OpenAPI) plus gRPC and cuopt_cli — its
README says cuOpt 'seamlessly extends into agent-first optimization workflows through open-
source cuOpt agent skills'. I searched the cuOpt repo for an MCP server and did not locate one
(a code search for 'mcp+in:path' returned no files). problemreductions is the only Ising-
adjacent Rust crate with an agent-facing story, and it is a CLI (`pred`) plus README prompts for
Claude Code/Codex, not MCP.

> grep sweep over scratchpad/py/venv/lib/python3.9/site-packages ; gh api repos/NVIDIA/cuopt/contents and repos/NVIDIA/cuopt/contents/AGENTS.md ; https://github.com/NVIDIA/cuopt ; https://github.com/CodingThrust/problem-reductions

### 13. Training (energy-based model training, gradient estimators) — partial

NOT FOUND in the Python stacks: grepped dimod, pyqubo, qubovert and openjij for
contrastive_divergence|boltzmann_machine|log_likelihood — ZERO hits in all four. These libraries
model and sample; they do not fit. The only implementation located is Rust: quantrs2-anneal's
`quantum_boltzmann_machine.rs` provides `QuantumRestrictedBoltzmannMachine` with
`QbmTrainingConfig`, `train(&[TrainingSample])`, `infer`, `generate_samples`,
`save_model`/`load_model`, and constructors `create_binary_rbm` /
`create_gaussian_bernoulli_rbm`. I read the `train` body: it is a real CD implementation with
minibatching, shuffling, momentum terms, an optional persistent-chain path gated on
`training_config.persistent_cd` (PCD), and a `contrastive_divergence_batch` call returning
reconstruction error and free-energy difference. quantrs2-tytan has a parallel
`quantum_ml_integration.rs`. NOT FOUND: ommx, rustqubo, hercules, separatrix, problemreductions,
OR-Tools, cuOpt.

> grep sweep over scratchpad/py/venv/lib/python3.9/site-packages/{dimod,pyqubo,qubovert,openjij} ; scratchpad/crates/quantrs2-anneal/quantrs2-anneal-0.2.0/src/quantum_boltzmann_machine.rs:105,133,347-420

### 14. Visual / graph programming — **not found**

NOT FOUND in every Python stack: dimod, PyQUBO, qubovert, OpenJij, OR-Tools and cuOpt ship no
node-graph or visual authoring surface. The closest things located are (a) quantrs2-tytan's
`visual_problem_builder` module (a directory with mod.rs, types.rs, functions.rs and trait files
codegenerator_traits.rs, gridsettings_traits.rs, theme_traits.rs, problemvalidator_traits.rs,
visualproblem_traits.rs) — a headless builder + code-generator model with grid/theme concepts,
but no renderer and no interactive surface in the crate; and (b) its `problem_dsl` module, which
is a genuine TEXTUAL DSL (lexer.rs, parser.rs, ast.rs, compiler.rs, stdlib.rs, optimizer.rs)
with `ProblemDSL::{parse, tokenize, compile_to_qubo}` — that is a language, not a graph.
quantrs2-tytan also has a `visualization/` module (convergence, solution_analysis) but that is
output plotting, not programming.

> scratchpad/crates/quantrs2-tytan/quantrs2-tytan-0.2.0/src/visual_problem_builder/ and src/problem_dsl/mod.rs:36-105

### 15. Licence, openness, hardware requirement — **yes**

The open layer is genuinely open and needs NO hardware. Verified from package metadata: dimod
Apache-2.0, PyQUBO Apache-2.0, qubovert Apache-2.0, OpenJij Apache-2.0, OR-Tools Apache-2.0,
minorminer Apache-2.0, Pyomo BSD-3-Clause, PuLP MIT. Rust: ommx 'MIT OR Apache-2.0', rustqubo
MIT, annealers MIT, quantrs2-anneal/tytan Apache-2.0, hercules BSD-3-Clause, separatrix MIT,
problemreductions MIT. cuOpt is Apache-2.0 (github.com/NVIDIA/cuopt, also hosted as a COIN-OR
project) but REQUIRES NVIDIA GPU: CUDA 12.0+/13.0+, Volta or better (CC>=7.0), driver
>=525.60.13. THE IMPORTANT EXCEPTION: OpenJij's modelling layer is NOT open — JijModeling 2.7.1
carries the PyPI classifier 'License :: Other/Proprietary License'. So the open half of the Jij
stack is the sampler (OpenJij) and the exchange format (OMMX); the ergonomic modelling layer is
commercial. PACKAGING DEFECT FOUND: rustqubo 0.1.0 does NOT build as a plain dependency — its
Cargo.toml sets `default = ["python"]` pulling pyo3 with `extension-module`, combined with
`crate-type = ["rlib","dylib"]`; on macOS arm64 this fails at link with hundreds of undefined
_Py* symbols. It builds and runs correctly with `default-features = false`, which I verified.
That is a one-line fix nobody has made since 2023-06-03.

> pip show output for all Python packages ; Cargo.toml of each downloaded crate ; https://pypi.org/project/jijmodeling/ (classifiers) ; gh api repos/NVIDIA/cuopt (license Apache-2.0) ; scratchpad/crates/rustqubo/rustqubo-0.1.0/Cargo.toml.orig:9-14 ; scratchpad/buildtest (link failure then success with default-features=false)

### ANSWER: what does the Python ecosystem consider the standard way to express a constrained optimisation problem — and how does that compare to sending a QUBO matrix? — **yes**

The standard is an ALGEBRAIC MODEL OF NAMED COMPONENTS, serialised to a solver-agnostic FILE
FORMAT, then dispatched to any of ~30 solvers. I demonstrated it: in Pyomo 6.9.5 a ConcreteModel
carries `Set`, `Param`, `Var(domain=Binary)`, `Objective`, `Constraint`, each an addressable
named component — `m.component_objects(Constraint)` returned ['capacity','pick_one_of_ab'] — and
`m.write('knap.lp')` emitted an LP file naming every row (`c_u_capacity_`,
`c_e_pick_one_of_ab_`), every column (`take(a)`), the bounds block and the `binary` section.
Pyomo's WriterFactory offers lp, mps, nl, bar, gams, gurobi_minlp; SolverFactory covers
cbc/cplex/glpk/gurobi/ipopt/highs/baron/neos and more. PuLP (MIT) and OR-Tools MathOpt are the
same shape — MathOpt's `mathopt.solve(model, SolverType.X)` accepts CP_SAT, GLOP, PDLP, HIGHS,
GSCIP, GUROBI, OSQP, ECOS, SCS, GLPK behind one model object. THE COMPARISON, precisely: a QUBO
matrix is the model AFTER four irreversible losses. (1) NAMES — an LP row is `capacity`, a QUBO
row is index 3; PyQUBO/qubovert/dimod all keep a side mapping to undo this, which is exactly the
admission that the matrix alone is insufficient. (2) STRUCTURE — `<=` does not exist in a QUBO;
it must be re-encoded as slack bits (dimod's cqm_to_bqm turned one `<=7` row into three log-
encoded slack variables) and the row identity is gone. (3) DOMAINS — integer and continuous
variables must be encoded to bits, and the encoding (one-hot/log/unary/order) is a modelling
decision the matrix cannot record. (4) FEASIBILITY — an LP/MPS solver returns INFEASIBLE as a
status and CP-SAT will even return a minimal explanation (I ran
`sufficient_assumptions_for_infeasibility()` and got [2,3], the two contradictory assumption
literals); a QUBO sampler returns a low-energy bitstring that may silently violate constraints,
and 'which constraint broke' is only recoverable if the modelling layer kept the labels. That
fourth loss is the substantive one, and it is why dimod's `iter_constraint_data` and PyQUBO's
`constraints(only_broken=True)` exist. NET: the ecosystem standard is strictly richer than a
QUBO matrix; a QUBO matrix is a compilation target, and every serious QUBO library is really a
compiler that retains a symbol table so it can decompile the answer.

> scratchpad/py Pyomo probe (component_objects, WriterFactory, generated /tmp/knap.lp) ; scratchpad/py/run_ortools.py (CP-SAT status + sufficient_assumptions_for_infeasibility) ; scratchpad/py/run_dimod.py (cqm_to_bqm slack generation, iter_constraint_data) ; https://www.pyomo.org

### ANSWER: is there ANY Rust-native library for Ising/QUBO modelling with a constraint layer? — **yes**

YES — three, with very different maturity, and none is a complete equivalent of dimod's CQM. (1)
OMMX 3.0.0-beta.3 (Jij, MIT OR Apache-2.0, actively developed — published 2026-08-12) is the
strongest and I would call it the real answer. Rust-native `Instance` with named/subscripted
decision variables over five Kinds with bounds, constraint collections with senses, special
kinds Indicator/OneHot/SOS1, encoding passes (log/unary/one-hot/slack),
`penalty_method`/`uniform_penalty_method`, `as_qubo_format() -> (BTreeMap<BinaryIdPair,f64>,
f64)`, per-constraint `EvaluatedConstraint` and `SampleSet::feasible_ids`, MPS + QPLIB readers,
a CLI, a PyO3 bridge, and a machine-readable `InstanceClass` capability model. It compiles clean
(I built it). Its gap versus a thermodynamic stack: it is an EXCHANGE format — it has no
sampler, no embedder, no verification, no joules. (2) rustqubo 0.1.0 (MIT, yasuo-ozu, MITOU
TARGET 2020) is the closest thing to a Rust PyQUBO and is explicitly positioned as such:
`Expr::Constraint{label, expr}`, `Placeholder`, automatic order reduction, and
`solve_with_constraints()` returning the labels of BROKEN constraints. I built and ran it
successfully. But: last published 2023-06-03, 1513 downloads, no integer encodings, no
inequality/slack, no cardinality, a stochastic default solver that missed the optimum of a
3-variable problem 4 times in 10, and it does not build with default features (see layer 15).
(3) quantrs2-anneal 0.2.0 (Apache-2.0) has the broadest ambition — dsl.rs OptimizationModel with
binary/integer/spin variables and comparison builders, csp_compiler.rs with CSP domains and
global constraints, hobo.rs reduction with three methods, embedding.rs, penalty_optimization.rs
— but its constraint layer has a REACHABILITY HOLE I reproduced: `less_than_or_equal` is exposed
and accepted at model-build time and then rejected at compile time, and the CSP linear-
constraint path is an explicit stub. Treat its README's '3,895 public items, 0 stubs' as
unverified; the crate compiles, but compiling is not the same as the path being reachable. ALSO
WORTH NAMING, though not modelling layers: problemreductions 0.6.0 (MIT) is a Rust REDUCTION
layer with `QUBO` and `SpinGlass` models and concrete rules ilp_qubo, coloring_qubo,
graphpartitioning_qubo, minimumvertexcover_qubo, spinglass_maxcut — a different and interesting
way to get a constrained problem into Ising form. hercules 0.5.0 (BSD-3) and separatrix 0.1.0
(MIT) are solvers with NO modelling layer — hercules' `Constraint{x_i, x_j, ConstraintType}` is
a two-variable PRESOLVE inference device (`make_inference`, `check`) for fixing variables, not a
user-facing constraint vocabulary, and it is easy to misread as one. annealers 0.1.0 is solver
traits only.

> scratchpad/crates/ommx/ommx-3.0.0-beta.3/src/ (instance.rs:71, instance/{penalty,slack,log_encode,unary_encode,one_hot,qubo}.rs, instance_class.rs, solution.rs:256-288) ; scratchpad/crates/rustqubo/rustqubo-0.1.0/src/{lib.rs,expr.rs,compiled.rs:186,solve.rs:126} + scratchpad/buildtest (built and run) ; scratchpad/crates/quantrs2-anneal/quantrs2-anneal-0.2.0/src/{dsl.rs:583-643,csp_compiler.rs:653-667,hobo.rs} + scratchpad/qtest (built and run) ; scratchpad/crates/problemreductions/problemreductions-0.6.0/src/models/{algebraic/qubo.rs,graph/spin_glass.rs} and src/rules/ ; scratchpad/crates/hercules/hercules-0.5.0/src/constraint.rs

**Notes.** METHOD. Everything above is from primary sources. The five Python packages were installed into a
venv (dimod 0.12.21, pyqubo 1.5.0, qubovert 1.2.5, openjij 0.11.6, ortools 9.15.6755, plus pyomo
6.9.5, pulp 3.3.1, minorminer 0.2.19) and interrogated by EXECUTION, not by reading docs — every
"RAN IT" above has a script in scratchpad/py. Every Rust crate was downloaded from crates.io and
read at source; rustqubo, ommx and quantrs2-anneal were additionally compiled and run in
scratchpad/buildtest and scratchpad/qtest. Where I could not establish something I searched a
named place and report what I searched, so the negatives are "not located by this review in
<place>", not "does not exist".  ONE DOC SOURCE WAS DISCARDED. A WebFetch summary of the cuOpt
user guide asserted cuOpt is "built on NVIDIA's Claude Agent SDK". That is a summariser
artefact, not in the source. I dropped it and re-derived every cuOpt claim from
github.com/NVIDIA/cuopt via the gh API. Worth flagging because the same failure mode could have
contaminated any doc-derived claim in this survey.  THREE FINDINGS I'D PUT FIRST IN ANY WRITE-
UP.  1. The ecosystem splits into modellers and samplers, and the split is licence-shaped.
dimod, PyQUBO, qubovert and OMMX model; OpenJij, hercules and separatrix sample. OpenJij has NO
constraint API at all — I checked `dir(openjij)` and there is nothing. Its modelling layer,
JijModeling, is proprietary (PyPI classifier "License :: Other/Proprietary License"). So the
most-cited open Ising sampler in the world ships without an open modelling layer, and Jij's open
contribution at that layer is OMMX, an exchange format rather than an authoring DSL.  2. Layers
9 and 10 are empty across the board, and this is the cleanest gap in the survey. Grepping all
four Python QUBO packages for
effective_temperature|effective_sample|total_variation|autocorrel|gelman|r_hat returned ZERO
hits; grepping all five for joule|watt|kwh|nvml|carbon returned ZERO files. Nobody verifies that
a sampler reproduces a Boltzmann distribution, and nobody counts a joule. Two traps here: (a)
"energy" in these libraries always means objective value, never a physical joule — the word
collision will mislead a reader; (b) OpenJij's TTS/success-probability/residual-energy utilities
look like verification but are optimisation benchmarking and require the true ground state as an
input argument. The single apparent counterexample is fake and I read the body: quantrs2-tytan
documents a field as "Energy per sample (joules)" and computes it from `let avg_power = 150.0;
// Watts` under the comment "Using placeholder values for demonstration".  3. A capability can
be written, exposed, documented and unreachable. quantrs2-anneal's DSL advertises
`less_than_or_equal()`; `add_constraint` accepts it and `summary()` reports num_constraints:1;
then `compile_to_qubo()` returns Err("Unsupported constraint type in QUBO compilation") because
every inequality falls into a `_ =>` catch-all. I built the crate and reproduced this — the code
is in scratchpad/qtest/src/main.rs. Its CSP path is the same, with the body reading "This would
require implementing slack variables and penalty methods / For now, return unsupported". Take
this as a warning about the crate's own README claims ("3,895 public items: 0 stubs — complete
production-ready implementation"): the crate does compile, but compiling is not evidence that a
path is reachable, and README counts are not evidence of anything. I did not audit the other ~60
modules, so I cannot say how widespread the pattern is — only that I found it on the first
constraint path I tried.  FOR THE FERROTHERM COMPARISON, the honest reading of where the empty
lanes are: verification (layer 9) and energy accounting (layer 10) are unoccupied by every stack
surveyed, open or vendor. Agent surfaces (layer 12) are unoccupied by every Ising stack — the
only agent surface in the survey is cuOpt's, and cuOpt does not do Ising at all (I checked: the
single "QUBO" hit in the NVIDIA/cuopt repo is a base64 blob inside a TLS test certificate, so
that negative is solid). Visual/graph programming (layer 14) is unoccupied. Conversely, do NOT
claim the modelling or reduction lanes are empty: dimod's CQM is a strong, open, well-designed
modelling layer with per-constraint violation reporting and automatic log-encoded slack, higher-
order reduction is present in six independent implementations, and OMMX is a live, actively-
developed, Apache/MIT Rust-native constraint layer with encodings, penalty methods and a
capability model. The defensible Rust-native gap is narrower than "there is no Rust constraint
layer" — it is that no Rust stack combines a constraint layer with sampling, verification and
joules in one codebase.  CAVEATS ON MY OWN COVERAGE. The `present` enum on each row is an
ecosystem-wide verdict (does ANY open general-purpose stack provide this); the per-stack
breakdown with explicit not-founds is in each `detail` field, because per-stack rows would have
exceeded the schema's item limit. I did not benchmark performance anywhere — no timing claim
appears above. I did not audit quantrs2-tytan's ~60 modules or quantrs2-anneal's ~60 modules
beyond the constraint, encoding, embedding, HOBO, energy and sampler paths named. I did not test
the hardware sampler backends (D-Wave, Fujitsu, Hitachi, NEC, Braket, Azure) because they need
credentials, so I can say the code paths exist and issue REST calls but not that they work. I
did not install jijmodeling (proprietary) and rely on its PyPI classifier for the licence claim.

## Hardware vendors' own SDKs, non-D-Wave: (A) Fujitsu Digital Annealer — Computing-as-a-Service QUBO API V3c/V4 (FujitsuDA3Solver); (B) Toshiba SQBM+ V2 (Simulated Bifurcation, REST); (C) QBoson Kaiwu SDK 1.4.1 + kaiwu-community 1.0.7 + kaiwu-pytorch-plugin 0.2.0 (CIM); (D) Hitachi CMOS Annealing via Annealing Cloud Web (Web API v2, operated by Fixstars); (E) NEC Vector Annealing — cloud service 2.0 (SX-Aurora TSUBASA) and V4.0.0x x86 on-prem engine.

### 1. Modelling layer — FUJITSU DA — **not found**

The public API has no modelling layer. A problem is `binary_polynomial.terms[]` with
`{coefficient|c, polynomials|p}` where `p` is an array of integer VARIABLE NUMBERS (uint64), not
names. Answers return as `solutions[].configuration = {"0": false, "1": true, "2": false}` —
indices, not names. Fujitsu's DADK (Digital Annealer Development Kit, `dadk.BinPol`,
`reduce_higher_degree_to_qubo`) is referenced in third-party literature as the Python modelling
layer, but this review did not locate a public primary source for it: `dadk` returns 404 on PyPI
(`/pypi/dadk/json` and `/simple/dadk/`), and it is absent from the apidoc index which lists only
the User's Guide, QUBO API V3c, QUBO API V4, V3c/V4-Premium and Storage API.

> https://portal.aispf.global.fujitsu.com/apidoc/da/jp/api-ref/da-qubo-v4-en.yaml

### 2. Encodings — FUJITSU DA — partial

One-hot is exposed, but as a solver mode rather than a variable-domain encoding choice:
`one_way_one_hot_groups: {numbers:[4,3,5]}` (one True per consecutive group) and
`two_way_one_hot_groups: {numbers:[16,25,36]}` (one True per row and per column of a square
block). With `internal_penalty: 1` the solver generates the one-hot penalty internally so the
user's polynomial need not contain it. Domain-wall: not found. Binary/log encoding of integers:
not found — the caller must do it.

> https://portal.aispf.global.fujitsu.com/apidoc/da/jp/api-ref/da-qubo-v4-en.yaml

### 3. Higher-order reduction — FUJITSU DA — **yes**

Dedicated endpoint `POST /v1/qubo/hobo2qubo` (operationId `post-v1-qubo-hobo2qubo`), 'Converts a
HOBO (Higher Order Binary Optimization) to a QUBO'. Worked example in the spec: input `{c:1,
p:[1,2,3]}` returns `2x1x2 − 4x1x4 − 4x2x4 + x3x4 + 6x4`, i.e. ancilla variable 4 is introduced
with a quadratic penalty. Ancillas appear as new integer variable numbers; there is no named
record of which ancilla stands for which product. Concurrency limit: 2 simultaneous synchronous
hobo2qubo requests per account.

> https://portal.aispf.global.fujitsu.com/apidoc/da/jp/api-ref/da-qubo-v4-en.yaml

### 4. Constraint vocabulary — FUJITSU DA — partial

Inequality is a first-class field: `inequalities[]`, each `{terms:[...], lambda:int}`
interpreted as `A0*x0 + ... + An*xn − C <= 0`, with per-inequality weight `lambda` (1..1e9,
default 1). Equality/general constraints go in `penalty_binary_polynomial`, which the user must
write already squared (spec example: `(x1+x2+x3−2)^2`). Exactly-one via `one_way_one_hot_groups`
/ `two_way_one_hot_groups`. Cardinality: expressible only as an inequality. All-different: not
found.

> https://portal.aispf.global.fujitsu.com/apidoc/da/jp/api-ref/da-qubo-v4-en.yaml

### 5. Penalty handling — FUJITSU DA — partial

Automatic scaling exists: `penalty_auto_mode` (0 = fixed at `penalty_coef`; 1..10000 = internal
autofit using `penalty_coef` as initial value), `penalty_inc_rate` (100..200),
`max_penalty_coef`. Coefficient range is handled by an automatic 'Scaling and Rounding' feature
(64-bit signed integer quadratic, 76-bit linear). Feasibility is reported ONLY in aggregate:
`penalty_energy` per solution and per progress point — 'When penalty_energy is not 0, one or
more constraints are not satisfied'. WHICH constraint broke: not found.

> https://portal.aispf.global.fujitsu.com/apidoc/da/jp/da-guide-en.html

### 6. Embedding / placement — FUJITSU DA — **not found**

No embedding step and none needed: DA3/DA4 is fully coupled up to 100,000 bits ('The maximum
problem scale can be solved is 100K (100,000) bits'). The analogous fitting problem is numeric
precision, handled by the automatic Scaling and Rounding feature rather than by a topology
mapper.

> https://portal.aispf.global.fujitsu.com/apidoc/da/jp/da-guide-en.html

### 7. Samplers/solvers — FUJITSU DA — partial

Exactly one solver key, `fujitsuDA3` → `FujitsuDA3Solver` ('The 3rd Generation and 4th
Generation Digital Annealer Solver'). Parameters: `time_limit_sec` 1..3600, `target_energy`,
`num_run` 1..1024, `num_group` 1..16, `num_output_solution`, `gs_level` 0..100, `gs_cutoff`,
`one_hot_level`, `one_hot_cutoff`, `guidance_config` (per-variable initial values),
`fixed_config` (per-variable fixing). Hardware is Fujitsu's own cloud; no CPU/GPU choice
exposed.

> https://portal.aispf.global.fujitsu.com/apidoc/da/jp/api-ref/da-qubo-v4-en.yaml

### 8. Device abstraction — FUJITSU DA — **not found**

Single-vendor, single solver. No capability-declaration endpoint beyond `GET /v1/healthcheck`
(returns `{}`); bit-count and coefficient-precision limits are documented in prose, not machine-
readable.

> https://portal.aispf.global.fujitsu.com/apidoc/da/jp/api-ref/da-qubo-v4-en.yaml

### 9. Verification — FUJITSU DA — **not found**

The response carries `solutions[].{configuration, energy, frequency, penalty_energy}`,
`progress[].{energy, penalty_energy, time}` and `timing.{solve_time, total_elapsed_time}`.
`frequency` is a raw occurrence count. No effective-temperature estimate, no ESS, no TV-
distance-vs-exact, no sampling certificate, no conformance suite.

> https://portal.aispf.global.fujitsu.com/apidoc/da/jp/api-ref/da-qubo-v4-en.yaml

### 10. Energy/cost accounting — FUJITSU DA — **not found**

A grep of the full OpenAPI 3.0.2 spec (1408 lines) for watt/joule/power/price/cost/billing
returns zero hits. Only wall-clock is reported (`solve_time`, `total_elapsed_time`, in
milliseconds as strings).

> https://portal.aispf.global.fujitsu.com/apidoc/da/jp/api-ref/da-qubo-v4-en.yaml

### 11. Language surfaces — FUJITSU DA — partial

REST/JSON over HTTPS, language-agnostic; auth via `X-Access-Token` or `X-Api-Key` header.
Problems >2 GB must go via Azure Blob Storage (`bucket_name`, `binary_polynomial_object_name`,
`penalty_binary_polynomial_object_name`, `inequalities_object_name`), up to 20 GB. No first-
party client library located in this review (`fujitsu-quantum` 2.2.5 on PyPI is the Fujitsu
Quantum Cloud SDK for gate-based machines, not DA).

> https://portal.aispf.global.fujitsu.com/apidoc/da/jp/api-ref/da-qubo-v4-en.yaml

### 12. Agent/AI surfaces — FUJITSU DA — partial

A machine-readable OpenAPI 3.0.2 document is published at a stable URL, which is directly usable
as a structured tool schema — the best agent-facing artefact of the five. No MCP server located
for Fujitsu DA.

> https://portal.aispf.global.fujitsu.com/apidoc/da/jp/api-ref/da-qubo-v4-en.yaml

### 13. Training (EBM / gradient estimators) — FUJITSU DA — **not found**

Nothing in the API or User's Guide relates to model training, Boltzmann machines, or gradient
estimation.

> https://portal.aispf.global.fujitsu.com/apidoc/da/jp/da-guide-en.html

### 14. Visual/graph programming — FUJITSU DA — **not found**

No visual or node-graph surface located in the public documentation set.

> https://portal.aispf.global.fujitsu.com/apidoc/da/jp/index.html

### 15. Licence / openness / hardware required — FUJITSU DA — **not found**

Proprietary and hosted; requires a Fujitsu Computing-as-a-Service contract (job IDs are of the
form `contract-a001-...`), plus a separate Microsoft Azure contract for large problems. No open-
source component. Structural note checked 2026-08-14:
`https://www.fujitsu.com/global/digitalannealer/` now 302s to the Fujitsu global home page,
while the apidoc portal (V3c/V4 references, User's Guide) still returns 200.

> https://portal.aispf.global.fujitsu.com/apidoc/da/jp/index.html

### 1. Modelling layer — TOSHIBA SQBM+ — **not found**

No modelling layer whatsoever. Six REST solvers each take a file or matrix: `qubo` (MatrixMarket
or QUBO-compatible HDF5), `qplib` (QPLIB text or HDF5), `pubo` (QPLIB), `tsp`, `qap`, `shift`
(JSON). Results are positional: 'An array of variables. If an array has five variables, its
"result" would be [0,1,1,0,0]'. No variable names anywhere.

> https://www.global.toshiba/content/dam/toshiba/ww/products-solutions/ai-iot/sbm/pdf/User_Manual-SQBM_for_AWS_V2.pdf

### 2. Encodings — TOSHIBA SQBM+ — **not found**

No encoding concept and no encoding selector; the caller supplies Q, B, A, LHS, RHS
matrices/vectors directly. Data-type choices exist (`format: dense` / `csr` sparse in HDF5) but
those are storage layouts, not variable encodings.

> https://www.global.toshiba/content/dam/toshiba/ww/products-solutions/ai-iot/sbm/pdf/User_Manual-SQBM_for_AWS_V2.pdf

### 3. Higher-order reduction — TOSHIBA SQBM+ — partial

There is no reduction utility — instead the `pubo` solver API accepts higher-order terms
NATIVELY up to order 4 ('a solver that solves a problem with higher order terms up to order 4'),
via qplib-format problem types QBB/QCB/QGB, ≤100,000 variables, ≥16 GB GPU memory. Degree >4 and
any k-body-to-pairwise lowering: not found.

> https://www.global.toshiba/content/dam/toshiba/ww/products-solutions/ai-iot/sbm/pdf/User_Manual-SQBM_for_AWS_V2.pdf

### 4. Constraint vocabulary — TOSHIBA SQBM+ — partial

The `qplib` solver takes LINEAR CONSTRAINTS NATIVELY, not as penalties: minimize ½·xᵀQx + B·x
subject to LHS ≤ A·x ≤ RHS, with A an M×N matrix, up to 1,000,000 constraints and 10,000,000
variables. Equality when LHS_m = RHS_m; one-sided inequality via ±inf. Cardinality is
expressible as a row of A. Exactly-one and all-different as named primitives: not found.
Separately, the `shift` solver has a hard-coded domain vocabulary — `empcaps` (executable jobs),
`jobconn` (job connection), `emptime` (total work hours), `dayplan` (human-resource planning),
`empplan` (number of jobs) — each with its own weight.

> https://www.global.toshiba/content/dam/toshiba/ww/products-solutions/ai-iot/sbm/pdf/User_Manual-SQBM_for_AWS_V2.pdf

### 5. Penalty handling — TOSHIBA SQBM+ — partial

No user-facing penalty scaling for qplib — constraints are handled inside the algorithm (PD3O,
tuned by `phi` ∈ [0,1]: near 0 behaves like plain SB, near 1 narrows the search). Infeasibility
reporting is unusually good but split across two channels: `detail_level=1` returns HTTP 400
with `main_info.message`, `additional.runs` and `additional_info[]` (up to 10 most recent
results with `value` and `result`); `detail_log=1` makes SQBM+ 'write value of each constraint
violation rate to log file'. So PER-CONSTRAINT violation is available — but only in the server
log, never in the API response.

> https://www.global.toshiba/content/dam/toshiba/ww/products-solutions/ai-iot/sbm/pdf/User_Manual-SQBM_for_AWS_V2.pdf

### 6. Embedding / placement — TOSHIBA SQBM+ — **not found**

No embedding step and none needed — all-to-all coupling on GPU. The real placement constraint is
memory: ≥16 GB GPU memory, and 'For problem data with more than 45,000 variables or 300 million
interaction terms, GPU memory is required 32 GB or larger for each GPU core'.

> https://www.global.toshiba/content/dam/toshiba/ww/products-solutions/ai-iot/sbm/pdf/User_Manual-SQBM_for_AWS_V2.pdf

### 7. Samplers/solvers — TOSHIBA SQBM+ — partial

Simulated Bifurcation, selected by the `algo` parameter: bSB (ballistic — fast, good solution)
and dSB (discrete — more accurate); PD3O for constrained qplib. GPU-resident: the `blocks`
parameter 'Specify the number of blocks of GPUs used to find a solution', and the API iterates
until 'loops × number of GPU devices × number of multi-processors' solutions are obtained.
Tuning knobs: `steps`, `loops`, `C`, `dt`, `timeout`, `maxout` (number of solutions returned).
The product page also lists an FPGA version ('Under Test marketing') and a GPU version ('Under
development').

> https://www.global.toshiba/content/dam/toshiba/ww/products-solutions/ai-iot/sbm/pdf/User_Manual-SQBM_for_AWS_V2.pdf

### 8. Device abstraction — TOSHIBA SQBM+ — **not found**

Single vendor, single engine. `GET /healthcheck` and `GET /version` are the only introspection
endpoints; neither declares capabilities such as variable limits or supported degrees.

> https://www.global.toshiba/content/dam/toshiba/ww/products-solutions/ai-iot/sbm/pdf/User_Manual-SQBM_for_AWS_V2.pdf

### 9. Verification — TOSHIBA SQBM+ — **not found**

Responses contain `value` (objective), `result` (variable array), `param` (echo of
algo/steps/dt/C), timing and `runs`. `maxout` returns multiple solutions. No effective
temperature, no ESS, no TV-vs-exact, no sampling certificate, no conformance suite. SB is a
deterministic-dynamics heuristic rather than a thermal sampler, so no distributional claim is
made.

> https://www.global.toshiba/content/dam/toshiba/ww/products-solutions/ai-iot/sbm/pdf/User_Manual-SQBM_for_AWS_V2.pdf

### 10. Energy/cost accounting — TOSHIBA SQBM+ — **not found**

A grep of the 2,926-line manual text for watt/joule/kWh/power-consumption/price/billing returns
no matches; every occurrence of 'cost' is the problem-domain objective in the shift and qap
solvers (e.g. 'Total cost of flow'). The intro page's only power claim is qualitative:
implementable 'on the general servers - normal temperature, normal power supply'.

> https://www.global.toshiba/ww/products-solutions/ai-iot/sbm/intro.html

### 11. Language surfaces — TOSHIBA SQBM+ — partial

REST only, hence language-agnostic: `POST
http://{sqbmplus_server}:8000/solver/{qubo|qplib|pubo|tsp|shift|qap}` with `Content-Type:
application/octet-stream` (or JSON for shift), plus `GET /healthcheck` and `GET /version`. All
examples are curl; one Python 3 snippet appears solely to build the HDF5 input file. No first-
party SDK in any language located in this review.

> https://www.global.toshiba/content/dam/toshiba/ww/products-solutions/ai-iot/sbm/pdf/User_Manual-SQBM_for_AWS_V2.pdf

### 12. Agent/AI surfaces — TOSHIBA SQBM+ — partial

A plain REST API with a documented request/response contract, so it is callable from an agent,
but this review did not locate an OpenAPI/Swagger document or an MCP server for SQBM+; the
contract exists only as PDF prose tables.

> https://www.global.toshiba/content/dam/toshiba/ww/products-solutions/ai-iot/sbm/pdf/User_Manual-SQBM_for_AWS_V2.pdf

### 13. Training (EBM / gradient estimators) — TOSHIBA SQBM+ — **not found**

No training, Boltzmann-machine or gradient-estimator functionality in the manual or product
pages.

> https://www.global.toshiba/ww/products-solutions/ai-iot/sbm/intro.html

### 14. Visual/graph programming — TOSHIBA SQBM+ — **not found**

No visual or node-graph surface located.

> https://www.global.toshiba/ww/products-solutions/ai-iot/sbm/intro.html

### 15. Licence / openness / hardware required — TOSHIBA SQBM+ — **not found**

Proprietary, closed-source. Delivered as a virtual machine image (AWS AMI; also offered via
partners such as Strangeworks), plus on-prem FPGA/GPU variants. No dedicated annealing hardware
is required — it runs on commodity GPU servers — but a Toshiba software licence is. Version 2.2
is stated to support 'up to 1 billion variables'. Export-controlled: 'This service is subject to
the Foreign Exchange and Foreign Trade Control Act and all United States export laws'. Only
third-party components are openly licensed (QPLIB under CC-BY 4.0).

> https://www.global.toshiba/content/dam/toshiba/ww/products-solutions/ai-iot/sbm/pdf/User_Manual-SQBM_for_AWS_V2.pdf

### 1. Modelling layer — QBOSON KAIWU — **yes**

A genuine modelling layer, the strongest of the five. `kaiwu.core` gives named variables with
domains: `Binary(name)` ∈{0,1}, `Spin(name)` ∈{−1,1}, `Integer(name, min_value=0,
max_value=127)`, `Placeholder(name)` (symbolic parameter, resolved by `.feed({'p':2})`), plus
`ndarray(shape, name, var_func)` / `zeros` / `dot` / `quicksum` for vectorised construction over
`BinaryExpressionNDArray`. Models: `BinaryModel` → `QuboModel` (`set_objective`,
`add_constraint`, `make()`, `get_matrix()`, `get_offset()`, `get_sol_dict()`) and `IsingModel`.
`__str__` prints 'Minimize <obj>\nSubject to (hard constraints):' — a real symbolic model.
Answers by name: `get_sol_dict(solution, vars_dict)` returns a dict whose 'Keys are variable
names'.

> https://kaiwu-sdk-docs.qboson.com/en/latest/source/modules/kaiwu.core.html

### 2. Encodings — QBOSON KAIWU — partial

Exactly one integer encoding, hard-coded and NOT selectable. `Integer.__init__` builds a
bounded-coefficient binary (log) encoding: `num_bits = int(math.log2(max_value - min_value))`,
coefficients `2**j` for j<num_bits and a final capped bit `max_value - min_value - 2**num_bits +
1`, with `offset = min_value`. One-hot, domain-wall and unary: NOT FOUND — a recursive grep of
the whole kaiwu-community 1.0.7 source tree for one.?hot / domain.?wall / unary / all.?different
/ cardinality / exactly.?one / embed returns zero matches. You build one-hot yourself as
`(quicksum(x) - 1)**2`.

> https://files.pythonhosted.org/packages/24/6f/cdb10b501a72c483fe6f44d08af9f36baaf00a5f9be76b7bf340236f2bb3/kaiwu_community-1.0.7.tar.gz (src/kaiwu/core/_binary_expression.py, class Integer)

### 3. Higher-order reduction — QBOSON KAIWU — **yes**

This is what the `hobo` module is: `kaiwu.hobo.HoboModel(objective=None,
hobo_default_penalty=1)`, subclass of `BinaryModel`, with `reduce(predefined_pairs=None) ->
QuboModel` ('Reduce the order of high-order expressions in the HOBO Model (to second-order)').
The algorithm is Rosenberg quadratisation with ancillas: introduce y with y = x0·x1 and add
p(x0,x1,y) = x0x1 − 2x0y − 2x1y + 3y, giving f(x,y) + k·Σ p(xi,xj,yij). `predefined_pairs` lets
you steer which products get merged. Crucially it also ships the checker:
`verify_hobo_constraint(solution_dict) -> (int, dict)` 'Verify whether the HOBO reduction
constraints are satisfied', so broken ancilla consistency is detectable, per-ancilla.

> https://kaiwu-sdk-docs.qboson.com/en/latest/source/advanced/hobo_reduce.html

### 4. Constraint vocabulary — QBOSON KAIWU — partial

`Constraint(expr_left, relation, expected_value, slack_var_expr)` supports the full relational
set `== != > >= < <=` (dispatched through an `ops` dict of `operator.eq/ne/gt/ge/lt/le`).
`BinaryModel.add_constraint(constraint_in, name=None, constr_type='soft'|'hard', penalty=1,
slack_var_expr=None)` accepts a single constraint or a list/tuple/ndarray (auto-named
`name[00]`, `name[01]`…). Equality is squared; inequalities get an AUTO-GENERATED integer slack
— `_create_slack_variable` sets slack_min = 0 for >=/<= and 1 for >/<, computes the range from
the sum of negative coefficients minus the offset, discretises at `_find_min_interval` (smallest
positive gap between coefficients), then emits `Integer(f"_slack_{name}", slack_min,
precision_steps+slack_min)` scaled by range/steps; `_adjust_inequality_direction` negates for
>/>=. Cardinality, exactly-one and all-different as named primitives: NOT FOUND — write them as
sums.

> https://files.pythonhosted.org/packages/24/6f/cdb10b501a72c483fe6f44d08af9f36baaf00a5f9be76b7bf340236f2bb3/kaiwu_community-1.0.7.tar.gz (src/kaiwu/core/_penalty_method_constraint.py, src/kaiwu/core/_constraint.py)

### 5. Penalty handling — QBOSON KAIWU — **yes**

Automatic scaling AND per-constraint feasibility reporting — the only stack of the five to do
both. `BinaryModel.initialize_penalties()` sets each hard constraint via
`get_min_penalty_from_min_diff(cons, negative_delta, positive_delta)` (max single-flip objective
delta from `objective.get_max_deltas()` divided by the smallest non-zero increment of the
constraint expression) and each soft constraint via `get_soft_penalty` (ratio of average
coefficients). Variants: `get_min_penalty`, `get_min_penalty_for_equal_constraint` (guarantees a
feasible solution is a 1-flip local optimum), `get_min_penalty_from_deltas(...,
min_delta_method='diff'|'exhaust')`. Runtime adaptation:
`PenaltyMethodConstraint.penalize_more()` (×2) / `penalize_less()` (bisect), driven by
`kaiwu.hybrid.PenaltyMethodOptimizer(optimizer, controller)`. WHICH CONSTRAINT BROKE:
`verify_constraint(solution_dict, constr_type='hard'|'soft')` returns `(unsatisfied_count,
{constraint_name: constraint_value})` — keyed by constraint name.

> https://files.pythonhosted.org/packages/24/6f/cdb10b501a72c483fe6f44d08af9f36baaf00a5f9be76b7bf340236f2bb3/kaiwu_community-1.0.7.tar.gz (src/kaiwu/core/_binary_model.py)

### 6. Embedding / placement — QBOSON KAIWU — partial

No graph-topology embedding (the CIM is fully connected), so `preprocess` solves the OTHER
hardware-fitting problem: coefficient PRECISION for an 8-bit device. It provides
`calculate_qubo_matrix_bit_width(m, bit_width=8)` / `calculate_ising_matrix_bit_width` (returns
precision + multiplier), `adjust_qubo_matrix_precision` / `adjust_ising_matrix_precision`,
`get_dynamic_range_metric(mat)` (log ratio of max to min coefficient difference),
`get_min_diff(mat)`, `perform_precision_adaption_mutate(ising_matrix, iterations=100,
heuristic='greedy', decision='heuristic')` ('iteratively reduce the dynamic range … while
preserving the optimal solution'), `perform_precision_adaption_split(ising_matrix, param_bit=8,
method='var')` (SPLITS VARIABLES to shrink coefficient range) with `restore_split_solution` /
`construct_split_solution`, bounds `lower_bound_parameters`, `upper_bound_sample`,
`upper_bound_simulated_annealing`, and the `PrecisionReducer(component, precision=8,
target_bits=None, only_feasible_solution=False)` decorator that wraps any solver. Explicit gaps:
adjacency/embedding and graph partitioning are NOT in this module.

> https://kaiwu-sdk-docs.qboson.com/en/latest/source/modules/kaiwu.preprocess.html

### 7. Samplers/solvers — QBOSON KAIWU — **yes**

`kaiwu.classical`: `SimulatedAnnealingOptimizer(initial_temperature=100, alpha=0.99,
cutoff_temperature=0.001, iterations_per_t=10, size_limit=100, flag_evolution_history,
rand_seed, process_num)` (CPU, `process_num=-1` uses all cores), `TabuSearchOptimizer(max_iter,
recency_size, kmax=3, span_control_p1=3, span_control_p2=7)`, `BruteForceOptimizer` ('slow but
accurate'). `kaiwu.sampler.SimulatedAnnealingSampler` (same knobs; 'Each solution is solved
independently'). `kaiwu.cim.CIMOptimizer(task_name, wait=False, interval=1, project_no=None,
task_mode='optimization'|'sampling', sample_number=10..2000)` submits to 'a special-purpose
quantum computer (SPQC)' — a coherent Ising machine built on degenerate optical parametric
oscillators; the compiled `_cim_optimizer.so` contains the endpoints
`/api/system/software_task_manager_pro/create_sdk_task/`,
`/api/system/software_task_manager_pro/get_sdk_task_result/`,
`/api/system/file/oss_signed_url/`. `kaiwu.hybrid.PenaltyMethodOptimizer`. No GPU backend
documented.

> https://kaiwu-sdk-docs.qboson.com/en/latest/source/modules/kaiwu.classical.html

### 8. Device abstraction — QBOSON KAIWU — partial

One interface spans classical and hardware: `kaiwu.core.IsingSolver` (`set_matrix`,
`on_matrix_change`, `solve(ising_matrix, negtail_flip, sort_solutions)`, `get_hamiltonian`) and
`QuboSolver(optimizer)` (`solve_qubo`) are implemented by SA, tabu, brute force,
`PrecisionReducer` and `CIMOptimizer` alike — so a CPU baseline and the CIM are drop-in swaps.
But it abstracts only QBoson's own machine; no other vendor's device, and there is no
capability-declaration call — the device's 8-bit coefficient precision is a number you type into
`preprocess` yourself.

> https://kaiwu-sdk-docs.qboson.com/en/latest/source/modules/kaiwu.core.html

### 9. Verification — QBOSON KAIWU — partial

Feasibility verification is real (`verify_constraint`, `verify_hobo_constraint`,
`PenaltyMethodConstraint.is_satisfied`, tolerance 1e-5). Solution bookkeeping:
`get_sorted_solutions(matrix, solutions, bias, negtail_ff, sort_solutions)`, `HeapUniquePool` /
`ArgpartitionUniquePool` for de-duplicated pools, `get_ha_history()` for the SA Hamiltonian
trace, and device-side `TaskMode.SAMPLING` with `sample_number` 10..2000. But SAMPLING
CERTIFICATES ARE NOT FOUND: no effective-temperature estimation, no ESS, no TV distance against
an exact distribution, no conformance suite. The mode.html page describes sampling mode only
qualitatively — it 'continuously samples the solution space through special-purpose quantum
energy evolution and eventually converges to a low-energy state' — with no claim about the
resulting distribution, even though it names Boltzmann-machine training as the target
application.

> https://kaiwu-sdk-docs.qboson.com/en/latest/source/advanced/mode.html

### 10. Energy/cost accounting — QBOSON KAIWU — **not found**

No joules, watts or price in any module of kaiwu 1.4.1 or kaiwu-community 1.0.7. The nearest
thing is telemetry flowing the other way: `kaiwu/license/_post_sdk_data.so` posts to
`api/sdk_data/`, and the PyTorch plugin has
`usage_stats.enable_usage_stats/disable_usage_stats/is_usage_stats_enabled` which tags CIM tasks
with `task_source_detail` — vendor usage reporting, not energy accounting.

> https://files.pythonhosted.org/packages/4e/32/ab30471d7da64a20b4b727218ed7df0c0b2442a169722b7d226b723d8147/kaiwu-1.4.1-cp310-none-macosx_11_0_arm64.whl

### 11. Language surfaces — QBOSON KAIWU — partial

Python only, `>=3.10`. The enterprise `kaiwu` wheel is native-compiled and platform-pinned —
`.so`/`.pyd` per module (cp310 only; macOS x86_64 and arm64, manylinux1 x86_64, win_amd64) with
thin `__init__.py` re-export shims. The pure-Python part is the separate dependency `kaiwu-
community==1.0.7` (`src/kaiwu/core`, `src/kaiwu/common`), which is where the whole modelling
layer actually lives. Hard-pinned deps (numpy==2.2.6, pandas==2.2.3, matplotlib==3.10.3,
httpx==0.28.1, …). No C, C++, Rust, Julia or browser surface located.

> https://pypi.org/pypi/kaiwu/json

### 12. Agent/AI surfaces — QBOSON KAIWU — **not found**

No MCP server and no documented public HTTP API for third-party use — the
`/api/system/software_task_manager_pro/...` endpoints are internal to the compiled CIM client.
The closest structured surface is `JsonSerializableMixin` (`to_json_dict(exclude_fields)` /
`load_json_dict`) on optimizers and loop controllers, plus `CheckpointManager.save_dir` for
resumable runs; that is serialisation, not a tool schema.

> https://kaiwu-sdk-docs.qboson.com/en/latest/source/modules/kaiwu.common.html

### 13. Training (EBM / gradient estimators) — QBOSON KAIWU — **yes**

The only real EBM training stack among the five, in a SEPARATE Apache-2.0 repo: qboson/kaiwu-
pytorch-plugin v0.2.0, `src/kaiwu/torch_plugin/` exporting `RestrictedBoltzmannMachine`,
`BoltzmannMachine`, `UnsupervisedDBN`, `QVAE`, `QDiffusion`/`EnergyModel`/`QDiffusionConfig`
(plus gbrbm.py, empty qgan.py). Gradient estimator is the standard positive/negative-phase
surrogate: `AbstractBoltzmannMachine.objective(s_positive, s_negative)` returns
`self(s_positive).mean() - self(s_negative).mean()`, documented as 'Objective function whose
gradient is equivalent to the gradient of negative log-likelihood'. The negative phase comes
from `sample(sampler)`, which calls `get_ising_matrix()` then `sampler.solve(ising_mat)` and
maps back with `(solution[:,:-1]*solution[:,[-1]]+1)/2` — so `SimulatedAnnealingOptimizer` and
`CIMOptimizer` are interchangeable backends. Their own tutorial names the gap honestly (CD/PCD
trade speed for approximate gradients) but the plugin ships NO effective-temperature calibration
between the device's output and the model's β.

> https://github.com/qboson/kaiwu-pytorch-plugin (src/kaiwu/torch_plugin/abstract_boltzmann_machine.py)

### 14. Visual/graph programming — QBOSON KAIWU — partial

A web cloud platform GUI exists but it is a submission form, not a programming surface: select
the hardware → Create Task → fill in the task name and UPLOAD THE MATRIX (documented as
`pd.DataFrame(qubo_mat).to_csv('tsp.csv', index=False, header=False)`) → confirm → submit;
results show 'qubo solution vector, qubo value evolution curve, task execution time'. No node-
graph model builder, no visual constraint editor located.

> https://kaiwu-sdk-docs.qboson.com/en/latest/source/getting_started/tutorial4_cloud_platform_usage_cases.html

### 15. Licence / openness / hardware required — QBOSON KAIWU — partial

Two-tier and mostly proprietary. The wheel METADATA states: 'This package is the enterprise
distribution. A valid license is required before using licensed features.'
`kaiwu.license.init(user_id, sdk_code)` generates a licence file and `ensure_license()` prompts
for credentials if absent. The docs split it as: community edition = basic capabilities; 'the
enterprise edition adds SPQC hardware solving, classical optimizers, precision adaptation, HOBO
modeling, license authentication, and usage-data reporting' — i.e. even the LOCAL CPU solvers
and the HOBO reducer are licence-gated, while `kaiwu.core` modelling ships in `kaiwu-community`.
Neither PyPI package declares a licence field. Hardware is NOT required for
modelling/classical/HOBO/preprocess; it IS required for `kaiwu.cim` ('register an account on the
Qboson Platform and contact official staff to obtain a real-machine quota'). The PyTorch plugin
alone is Apache-2.0 and runs fully classically.

> https://kaiwu-sdk-docs.qboson.com/en/latest/source/getting_started/introduction.html

### 1. Modelling layer — HITACHI CMOS / ANNEALING CLOUD WEB — **not found**

The interface is a bare Ising model laid out on a King's graph: spins are addressed by lattice
coordinates (x, y) with `id = x + y * graph_size`, and objectives 'consist only of Ising
variables of degree two or less'. No named variables, no domains, no objective/constraint
separation. Third-party SDKs supply the modelling (Fixstars Amplify `HitachiClient`; OpenJij
`cmos.sample_ising(king_graph=[[0,0,0,0,1],[0,0,1,0,-1],...])`). Caveat: the ACW Web API v2
reference itself is client-side rendered and this review could NOT retrieve its body — the page
renders 'Loading...', `_payload.json` is empty, and probes of /api/v2/solve,
/api/v2/solver/list, /openapi.json all return 404 — so the claims here rest on the operator's
SDK documentation, not on the v2 reference text.

> https://amplify.fixstars.com/en/docs/amplify/v1/clients/hitachi.html

### 2. Encodings — HITACHI CMOS / ANNEALING CLOUD WEB — **not found**

No encoding layer; spin values only. Integer/one-hot/domain-wall encoding is entirely the
caller's or the third-party SDK's responsibility.

> https://amplify.fixstars.com/en/docs/amplify/v1/clients/hitachi.html

### 3. Higher-order reduction — HITACHI CMOS / ANNEALING CLOUD WEB — **not found**

Degree ≤ 2 only ('objectives of 2nd degree'). No HOBO/PUBO endpoint and no reduction utility
located.

> https://amplify.fixstars.com/en/docs/amplify/v1/clients/hitachi.html

### 4. Constraint vocabulary — HITACHI CMOS / ANNEALING CLOUD WEB — **not found**

None. 'Constraints must be embedded as penalty functions within the objective.' No equality,
inequality, cardinality, exactly-one or all-different primitive.

> https://amplify.fixstars.com/en/docs/amplify/v1/clients/hitachi.html

### 5. Penalty handling — HITACHI CMOS / ANNEALING CLOUD WEB — **not found**

No penalty scaling, no feasibility check, no per-constraint reporting — there is no constraint
object for the service to reason about.

> https://amplify.fixstars.com/en/docs/amplify/v1/clients/hitachi.html

### 6. Embedding / placement — HITACHI CMOS / ANNEALING CLOUD WEB — partial

This is the one vendor of the five where embedding is UNAVOIDABLE — and where the vendor's own
API does not do it. Quadratic term indices 'must be adjacent vertically, horizontally, or
diagonally on the King's graph', so any dense problem needs minor-embedding onto the 512×512 (or
384×384) lattice. That work is done by third-party SDKs: Amplify's model conversion, with
`HitachiClient.solve()` offered as a bypass 'for topology-optimized problems' you have already
placed yourself.

> https://amplify.fixstars.com/en/docs/amplify/v1/clients/hitachi.html

### 7. Samplers/solvers — HITACHI CMOS / ANNEALING CLOUD WEB — partial

CMOS annealing, selected by a `type` parameter: type 3 = GPU 32-bit integer, 256k spins (512×512
King's graph); type 4 = GPU 32-bit float, 256k spins (default); type 5 = ASIC 4-bit, 147,456
spins (384×384). Types 3 and 5 require integer coefficients; type 4 accepts floats — so the
machine's coefficient precision is part of the device selection, not a modelling concern.

> https://amplify.fixstars.com/en/docs/amplify/v1/clients/hitachi.html

### 8. Device abstraction — HITACHI CMOS / ANNEALING CLOUD WEB — **not found**

Single vendor. The multi-vendor abstraction that exists over it (Amplify) is a third party, not
Hitachi.

> https://amplify.fixstars.com/en/docs/amplify/v1/clients/hitachi.html

### 9. Verification — HITACHI CMOS / ANNEALING CLOUD WEB — **not found**

No sampling certificate, effective temperature, ESS, TV-vs-exact, or conformance suite located.
Note the retrieval caveat: the v2 API reference body could not be retrieved in this review, so
this is an absence in the material examined, not a proven absence from the API.

> https://annealing-cloud.com/en/web-api/reference/v2.html

### 10. Energy/cost accounting — HITACHI CMOS / ANNEALING CLOUD WEB — **not found**

No per-job joule or price figure located, despite low power being CMOS annealing's headline
claim. Same retrieval caveat as layer 9.

> https://annealing-cloud.com/en/about/cmos-annealing-machine.html

### 11. Language surfaces — HITACHI CMOS / ANNEALING CLOUD WEB — partial

HTTP/JSON REST with an access token (`/en/web-api/token-request.html`), hence language-agnostic.
No first-party client library located; in practice access is through third-party Python SDKs
(Fixstars Amplify `HitachiClient`, OpenJij's CMOS annealer sampler).

> https://annealing-cloud.com/en/web-api/token-request.html

### 12. Agent/AI surfaces — HITACHI CMOS / ANNEALING CLOUD WEB — **not found**

No MCP server located, and no machine-readable API description retrievable — the reference page
is a client-rendered SPA fed from a headless CMS, so there is no OpenAPI document an agent could
consume.

> https://annealing-cloud.com/en/web-api/reference/v2.html

### 13. Training (EBM / gradient estimators) — HITACHI CMOS / ANNEALING CLOUD WEB — **not found**

No training functionality located.

> https://annealing-cloud.com/en/

### 14. Visual/graph programming — HITACHI CMOS / ANNEALING CLOUD WEB — partial

ACW ships browser demos with real visual editing — an 'Ising editor' for laying out spins and
couplings on the lattice, plus image-noise-reduction, network-robustness and traffic-signal
demos with tutorials. These are pedagogical surfaces (the site's 'Play' and 'Learn' sections, an
'ACW Skills Roadmap' and a 'Fitness Diagnostic Tool'), not a programming layer that emits
deployable models. Of the five vendors this is by far the most developed teaching surface.

> https://annealing-cloud.com/en/

### 15. Licence / openness / hardware required — HITACHI CMOS / ANNEALING CLOUD WEB — partial

Free, registration-based cloud access — the most open of the five in terms of getting hands on
real hardware. 'Annealing Cloud Web is operated by Fixstars Corporation using research and
development results from Hitachi, Ltd., etc. which was entrusted by New Energy and Industrial
Technology Development Organization (NEDO).' Terms of use and an agreement apply; the software
itself is not open source. Hardware is not purchasable by the user — it is only reachable
through the service.

> https://annealing-cloud.com/en/about/service.html

### 1. Modelling layer — NEC VECTOR ANNEALING — **not found**

NEC ships no modelling layer of its own and says so explicitly — it DELEGATES to a third-party
DSL. The 2.0 service spec lists PyQUBO under 'what the customer must prepare' ('定式化した組合せ最適問題を
QUBO 変換するライブラリ') and states that only QUBO data created with PyQUBO or numpy can be computed.
The V4 x86 guide requires `pyqubo>=1.4.0` and defines it as 'a domain-specific language (DSL)
that converts a formulated expression into QUBO form'. The consequence is that variables ARE
name-addressed downstream — constraints are written as `'x[0][1]'` strings and results come back
as `spin = {'x':0, 'y':0, 'z':1}` — but the names originate in PyQUBO, not in anything NEC
provides. `VectorAnnealing.model(qubo, offset, ...)` takes the already-compiled dict.

> https://www.hpc.cmc.osaka-u.ac.jp/wp-content/uploads/2025/10/Vector_Annealing_x86%E7%89%88__%E3%83%A6%E3%83%BC%E3%82%B5%E3%82%99%E3%83%BC%E3%82%B9%E3%82%99%E3%82%AB%E3%82%99%E3%82%A4%E3%83%88%E3%82%99_rev2.pdf

### 2. Encodings — NEC VECTOR ANNEALING — **not found**

No encoding selector and no encoding concept in the VA API; PyQUBO decides. What VA does instead
is give the solver knowledge of the resulting STRUCTURE (see layer 4) so it can flip whole one-
hot groups atomically rather than needing the encoding penalty to be well-scaled.

> https://amplify.fixstars.com/ja/docs/amplify/v1/_downloads/f90ee151bb5d360d5b6dadd8db400663/nec_vector_annealing_service_2.0_user_guide_v4.pdf

### 3. Higher-order reduction — NEC VECTOR ANNEALING — **yes**

V4 removes the need for reduction rather than performing it: `VectorAnnealing.model(qubo,
offset, high_order={('x[0][0]','x[0][1]','x[0][2]'): 3.0, ('x[1][0]',...,'x[1][3]'): 4.0, (5
spins): -5.0})`. The guide states that where cubic-and-above terms previously required auxiliary
spins plus constraint expressions before annealing, they can now be annealed AS HIGHER-ORDER
TERMS; degree is unbounded ('3次項以上が設定可能'), with the idempotence caveat that repeated spins must
be collapsed (`('x','y','z','z')` must be written `('x','y','z')`). Classic ancilla-style
reduction is still available if wanted, as the `spl` / 'cubic supplement' flip option ([[y, x1,
x2], ...] enforcing y = x1·x2).

> https://www.hpc.cmc.osaka-u.ac.jp/wp-content/uploads/2025/10/Vector_Annealing_x86%E7%89%88__%E3%83%A6%E3%83%BC%E3%82%B5%E3%82%99%E3%83%BC%E3%82%B9%E3%82%99%E3%82%AB%E3%82%99%E3%82%A4%E3%83%88%E3%82%99_rev2.pdf

### 4. Constraint vocabulary — NEC VECTOR ANNEALING — **yes**

By far the richest of the five, delivered as solver-side 'flip options' keyed on PyQUBO variable
NAMES. Service 2.0 `solve_param` already has: `onehot` [['x[0]','x[1]'],...] (exactly-one),
`fixed` {'x[0]':1,...}, `andzero` (AND of a group = 0), `orone` (OR of a group = 1),
`supplement` [['y[0]','x[0]','x[1]'],...], `maxone` [[1, ['x[0]','x[1]','x[2]']],...] (at-most-k
cardinality), `minmaxone` [[1,2,[...]],...] (k1 ≤ count ≤ k2), `init_spin`, `spin_list`. V4 x86
adds: conditioned min-max-one `{'condition':('cnd0',1), 'min':2, 'max':4, 'spin_set':{...}}`;
`pattern` in four forms — required pattern
`{'condition':('cnd0',1),'pattern':{'a':0,'b':0,'c':0}}`, prohibited pattern
`{'prohibit':True,'pattern':{...}}`, conditioned prohibited pattern, and reification
`{'equi_spin':'equi','pattern':{...}}` (indicator spin = 1 iff the pattern holds, composable
with conditioned constraints); and `weighted_sum` with
`'comparison':(VectorAnnealing.COMPARISON_OPERATOR_LESS_OR_EQUAL|EQUAL|GREATER_OR_EQUAL,
value)`. So equality ✓, weighted inequality ✓, cardinality ✓, exactly-one ✓, fixing ✓, forbidden
assignment ✓, indicator/reification ✓. All-different as a named primitive: not found
(expressible as a family of one-hots).

> https://amplify.fixstars.com/ja/docs/amplify/v1/_downloads/f90ee151bb5d360d5b6dadd8db400663/nec_vector_annealing_service_2.0_user_guide_v4.pdf

### 5. Penalty handling — NEC VECTOR ANNEALING — partial

NEC's approach sidesteps penalty weighting rather than automating it: the constraint-aware
search restricts moves to the feasible subspace — the guide's example is that under a one-hot
constraint on x1..x10, flipping one spin flips the set together so the constraint is never
violated — which is why no penalty coefficient is asked for. Two important caveats, both stated
in the guide: (a) 'いずれのオプションもハミルトニアンの定式にも別途含める必要があります' — every flip option must ALSO be encoded
in the Hamiltonian separately, EXCEPT under `vector_mode=VECTOR_MODE_CONSTRAINT` /
`constraint_only`; (b) `constraint_only` does not anneal at all — it terminates as soon as the
constraints are satisfied, and `pattern`, `weighted_sum` and conditioned min-max-one are usable
ONLY in those two modes. Automatic penalty scaling: not found. Feasibility reporting is a single
boolean per result: `result.constraint` — 'Satisfy constraint if constraint is True, Broken
constraint if constraint is False'. WHICH constraint broke: NOT FOUND.

> https://www.hpc.cmc.osaka-u.ac.jp/wp-content/uploads/2025/10/Vector_Annealing_x86%E7%89%88__%E3%83%A6%E3%83%BC%E3%82%B5%E3%82%99%E3%83%BC%E3%82%B9%E3%82%99%E3%82%AB%E3%82%99%E3%82%A4%E3%83%88%E3%82%99_rev2.pdf

### 6. Embedding / placement — NEC VECTOR ANNEALING — **not found**

No embedding and none needed: '結合：解像度 32bit 階調の全結合' — full all-to-all coupling at 32-bit
coefficient resolution, up to 300,000 bits (300k requires 8 multi-plan contracts). The only
placement-like knob is `dense` (True = dense matrix mode, False = sparse, None = auto-select by
QUBO density).

> https://amplify.fixstars.com/ja/docs/amplify/v1/_downloads/f90ee151bb5d360d5b6dadd8db400663/nec_vector_annealing_service_2.0_user_guide_v4.pdf

### 7. Samplers/solvers — NEC VECTOR ANNEALING — partial

One proprietary annealing engine, two deployments. Cloud service 2.0 runs on SX-Aurora TSUBASA
vector engines (`ve_num` 1..8 VE cards); V4.0.0x is an x86 build using OpenMP (`num_threads`,
`OMP_NUM_THREADS`) with a `precision` switch (PRECISION_COMPUTE_SINGLE / _DOUBLE). Shared
parameters: `num_reads` (1..20 in the cloud service), `num_results`, `num_sweeps` (1..100,000,
default 500), `beta_range [start, end, steps]` or explicit `beta_list`, `dense`, `vector_mode`
(speed/accuracy in 2.0; SPEED / CONSTRAINT / CONSTRAINT_ONLY in V4), `timeout` (1..7200 s
standard, 0 = unlimited on premium/dedicated).

> https://amplify.fixstars.com/ja/docs/amplify/v1/_downloads/f90ee151bb5d360d5b6dadd8db400663/nec_vector_annealing_service_2.0_user_guide_v4.pdf

### 8. Device abstraction — NEC VECTOR ANNEALING — partial

Same Python model/sampler API covers both the SX-Aurora vector-engine cloud service and the x86
on-prem engine, so the compute substrate is swappable within NEC's own line. No other vendor's
machine, and no capability-declaration call — limits (300k bits, 32-bit coupling resolution,
core counts allowed by the licence) are documented in prose only.

> https://amplify.fixstars.com/ja/docs/amplify/v1/_downloads/f90ee151bb5d360d5b6dadd8db400663/nec_vector_annealing_service_2.0_user_guide_v4.pdf

### 9. Verification — NEC VECTOR ANNEALING — **not found**

Each result carries `spin` (named dict), `energy`, `time` (seconds), `constraint` (bool),
`memory_usage` (GiB). That is it. No effective temperature, no ESS, no TV-vs-exact, no sampling
certificate, no conformance suite — and `num_reads` caps at 20 on the 2.0 cloud service, which
limits statistical assessment by construction.

> https://amplify.fixstars.com/ja/docs/amplify/v1/_downloads/f90ee151bb5d360d5b6dadd8db400663/nec_vector_annealing_service_2.0_user_guide_v4.pdf

### 10. Energy/cost accounting — NEC VECTOR ANNEALING — **not found**

A grep of both NEC documents (the 2.0 service spec and the V4 x86 user guide) for 電力 / 消費電力 /
joule / watt returns no per-job figure. Cost is metered by contract shape instead: 'multi-plan'
contracts for bit count, standard (shared) vs premium (dedicated) timeouts, and on-prem
licensing by usable CPU core count. `memory_usage` (GiB) is the only resource number reported
per run.

> https://www.hpc.cmc.osaka-u.ac.jp/wp-content/uploads/2025/10/Vector_Annealing_x86%E7%89%88__%E3%83%A6%E3%83%BC%E3%82%B5%E3%82%99%E3%83%BC%E3%82%B9%E3%82%99%E3%82%AB%E3%82%99%E3%82%A4%E3%83%88%E3%82%99_rev2.pdf

### 11. Language surfaces — NEC VECTOR ANNEALING — partial

The V4 architecture diagram shows BOTH a 'Python API' and a 'C++ API' over the VA engine — the
only native non-Python surface among the five vendor SDKs. Python entry points: `import
VectorAnnealing; VectorAnnealing.model(qubo, offset, **flip_options)`,
`VectorAnnealing.sampler()`, `sampler.sample(model, num_reads=5)`. The 2.0 cloud service is
Python-only through a client module: `from SACService import SACServiceClient;
sac.init_sac(init_param); sac.solve_qubo(qubo, solve_param)` over HTTPS/JSON against
`https://api.sac-service.aurora-xaas.com/login`, requiring the locally installed 'SAC service
client'. Requirements: Python ≥3.11, numpy ≥2.0.0, pyqubo ≥1.4.0.

> https://www.hpc.cmc.osaka-u.ac.jp/wp-content/uploads/2025/10/Vector_Annealing_x86%E7%89%88__%E3%83%A6%E3%83%BC%E3%82%B5%E3%82%99%E3%83%BC%E3%82%B9%E3%82%99%E3%82%AB%E3%82%99%E3%82%A4%E3%83%88%E3%82%99_rev2.pdf

### 12. Agent/AI surfaces — NEC VECTOR ANNEALING — partial

An HTTPS/JSON API exists (content-type application/json, UTF-8) with a fully enumerated error-
code table — S100..S131 for auth/body/requestId problems and S200..S222 for per-parameter type
errors, e.g. 'S214 onehot is not of type list' — which is unusually good machine-actionable
feedback. But it is reachable only through the installed SAC client, and no OpenAPI document or
MCP server was located.

> https://amplify.fixstars.com/ja/docs/amplify/v1/_downloads/f90ee151bb5d360d5b6dadd8db400663/nec_vector_annealing_service_2.0_user_guide_v4.pdf

### 13. Training (EBM / gradient estimators) — NEC VECTOR ANNEALING — **not found**

No Boltzmann-machine, energy-based-model or gradient-estimator functionality located in either
NEC document.

> https://www.hpc.cmc.osaka-u.ac.jp/wp-content/uploads/2025/10/Vector_Annealing_x86%E7%89%88__%E3%83%A6%E3%83%BC%E3%82%B5%E3%82%99%E3%83%BC%E3%82%B9%E3%82%99%E3%82%AB%E3%82%99%E3%82%A4%E3%83%88%E3%82%99_rev2.pdf

### 14. Visual/graph programming — NEC VECTOR ANNEALING — **not found**

No visual or node-graph surface located; the workflow is entirely code plus a file upload.

> https://amplify.fixstars.com/ja/docs/amplify/v1/_downloads/f90ee151bb5d360d5b6dadd8db400663/nec_vector_annealing_service_2.0_user_guide_v4.pdf

### 15. Licence / openness / hardware required — NEC VECTOR ANNEALING — **not found**

Proprietary and licence-enforced at runtime. Cloud service 2.0 requires a signed application,
tenant ID / user ID / password and a locally installed SAC service client; the document itself
forbids redistribution. On-prem V4 ships as RPMs —
`VectorAnnealing-4.0.0X-4.0.0X-1.el9.x86_64.rpm` plus
`va_license_manager-4.0.0X-1.el9.x86_64.rpm` (el8 variants for RHEL/Rocky 8.8/8.10) — installed
under `/opt/va/V4.0.0X`, with 'the VA engine depends on the license service' and the licence
capping usable cores (node licence = all server cores; basic licence = min(licensed cores,
server cores)). Dedicated NEC hardware is required only for the SX-Aurora path; the x86 build
runs on commodity CPUs but still needs the licence. The only open component is PyQUBO itself
(Apache-2.0, recruit-communications/pyqubo), which is not NEC's.

> https://www.hpc.cmc.osaka-u.ac.jp/wp-content/uploads/2025/10/Vector_Annealing_x86%E7%89%88__%E3%83%A6%E3%83%BC%E3%82%B5%E3%82%99%E3%83%BC%E3%82%B9%E3%82%99%E3%82%AB%E3%82%99%E3%82%A4%E3%83%88%E3%82%99_rev2.pdf

**Notes.** DIRECT ANSWERS TO THE TWO QUESTIONS ASKED.  (a) "Is there a modelling layer, or only a matrix
submission API?" Only ONE of the five vendors ships a real modelling layer: QBoson's Kaiwu
(`kaiwu.core`: named Binary/Spin/Integer/Placeholder variables, symbolic Expression algebra,
QuboModel with named constraints, answers keyed by variable name). Fujitsu, Toshiba and Hitachi
are matrix/polynomial submission APIs with positional or numeric variable identity. NEC is the
interesting middle case: no modelling layer of its own, but it explicitly imports one — the
service spec and the V4 guide both REQUIRE PyQUBO (Apache-2.0, third party) and state that only
PyQUBO- or numpy-produced QUBO data is accepted. Because PyQUBO names variables, NEC's
constraint options and results are name-addressed even though NEC wrote no modelling code.  (b)
"What does each offer above 'send me a QUBO matrix'?" Ranked by how much: NEC VA — a solver-side
constraint DSL (onehot, fixed, andzero, orone, supplement/spl, maxone, minmaxone + conditioned
form, pattern with prohibit/condition/equi_spin reification, weighted_sum with a comparison
operator) plus native unbounded higher-order terms via `high_order`, plus constraint-priority
search modes. Fujitsu DA — native `inequalities[]` with per-constraint `lambda`, one-/two-way
one-hot groups with internal penalty generation, automatic penalty autofit, a `hobo2qubo`
reduction endpoint, warm-start (`guidance_config`) and variable fixing (`fixed_config`). QBoson
Kaiwu — the modelling layer, Rosenberg HOBO reduction with a verifier, auto-slack inequalities,
automatic penalty derivation, per-constraint feasibility reporting, and a precision-adaptation
toolkit. Toshiba SQBM+ — native linear constraints via the qplib solver (LHS ≤ Ax ≤ RHS, up to
1e6 constraints / 1e7 variables) and native degree-4 PUBO, plus canned TSP/QAP/shift solvers.
Hitachi ACW — essentially nothing above the matrix; it is below it, since you must also place
the problem on a King's graph yourself.  WHAT `hobo` AND `preprocess` ACTUALLY DO IN KAIWU
(asked specifically). `kaiwu.hobo` is one class, `HoboModel(objective, hobo_default_penalty=1)`,
a BinaryModel subclass whose `reduce(predefined_pairs=None) -> QuboModel` performs Rosenberg
quadratisation: substitute y = x_i·x_j and add p = x_i x_j − 2 x_i y − 2 x_j y + 3y, scaled by a
penalty; `verify_hobo_constraint(solution_dict)` then checks that every ancilla actually equals
its product. `kaiwu.preprocess` is NOT preprocessing in the graph sense — it is entirely
COEFFICIENT-PRECISION ADAPTATION for a limited-precision device (the CIM's ~8-bit coefficients):
bit-width measurement (`calculate_ising_matrix_bit_width`), rescaling
(`adjust_ising_matrix_precision`), dynamic-range metrics (`get_dynamic_range_metric`,
`get_min_diff`), Hamiltonian bounds (`lower_bound_parameters`, `upper_bound_sample`,
`upper_bound_simulated_annealing`), a coefficient-mutation heuristic that shrinks dynamic range
while preserving the optimum (`perform_precision_adaption_mutate`), a variable-SPLITTING scheme
with round-trip helpers (`perform_precision_adaption_split` / `restore_split_solution` /
`construct_split_solution`), and the `PrecisionReducer` solver decorator. It contains no
embedding, adjacency or graph-partitioning function — I checked the module's own `__init__.py`
export list.  CROSS-CUTTING FINDINGS RELEVANT TO A RUST STACK LIKE FERROTHERM. 1. Layer 9
(verification) is empty across all five. Not one vendor reports an effective temperature, ESS,
TV distance against an exact distribution, or any sampling certificate. Fujitsu returns a raw
`frequency` count; NEC returns a boolean; QBoson's own docs describe sampling mode only as
"converges to a low-energy state" while naming Boltzmann-machine training as its application —
and their Apache-2.0 PyTorch plugin trains RBMs against a CIM with no temperature calibration
between device output and model β. This is the single largest open lane. 2. Layer 10
(energy/cost) is empty across all five. Zero joule, watt or price fields in any spec or manual I
read, despite low power being the marketing claim for CMOS annealing and SX-Aurora alike.
Second-largest open lane. 3. Layer 5 (which constraint broke) is nearly empty. Only Kaiwu
returns a name-keyed dict (`verify_constraint` → `(count, {name: value})`). Toshiba can log per-
constraint violation rates but only to a server log file (`detail_log=1`), never in the
response. Fujitsu and NEC give one aggregate number/boolean. 4. Layer 8 (device abstraction) is
empty in the cross-vendor sense — every SDK is single-vendor. Only Kaiwu abstracts classical and
hardware behind one interface (IsingSolver/QuboSolver), and only over QBoson's own machine. No
vendor declares device capabilities programmatically; bit-width and size limits live in prose.
5. Layer 12 (agent surfaces): no MCP server located for any of the five. Fujitsu is the only one
publishing a machine-readable OpenAPI 3.0.2 document at a stable URL. NEC's exhaustive per-
parameter error-code table (S200-S222) is the best machine-actionable failure reporting. 6.
Layer 11 (languages): Python dominates. NEC is the only vendor with a documented native C++ API.
Kaiwu's enterprise wheel is compiled `.so` pinned to cp310 and four platforms — no Rust, C,
Julia or browser surface exists anywhere in this set, so a portable Rust core is uncontested
ground. 7. Layer 2 (encodings): no vendor exposes an encoding CHOICE. Kaiwu hard-codes bounded-
coefficient binary for `Integer`; Fujitsu exposes one-hot only as a solver mode; the rest push
it to the caller. Domain-wall encoding appears nowhere in any of the five.  HARDWARE REQUIREMENT
AND LICENCE, SUMMARISED. All five cores are proprietary. Hardware is strictly required for
Fujitsu (CaaS contract), Hitachi (service only) and NEC's SX-Aurora path; NOT required for
Toshiba (commodity GPU servers under licence), NEC's x86 build (commodity CPUs under licence) or
Kaiwu's classical path (though the enterprise licence still gates the classical optimizers, HOBO
and preprocess). Only two openly licensed artefacts exist in the whole survey: qboson/kaiwu-
pytorch-plugin (Apache-2.0, and the only EBM training stack of the five) and PyQUBO (Apache-2.0,
third party, which NEC mandates). Hitachi's Annealing Cloud Web is free to register and has by
far the best teaching surface (Ising editor, demos, skills roadmap) but the least capable API.
RETRIEVAL GAPS I AM DECLARING RATHER THAN GUESSING. (i) Fujitsu DADK: `dadk.BinPol` /
`reduce_higher_degree_to_qubo` is referenced in third-party literature as Fujitsu's Python
modelling layer, but THIS REVIEW DID NOT LOCATE a public primary source — `dadk` 404s on both
PyPI JSON and simple indexes, and it is absent from Fujitsu's own apidoc index. Treat every DADK
capability as unverified. (ii) Hitachi ACW Web API v2 reference: the page body is client-side
rendered and could not be retrieved (renders "Loading...", `_payload.json` returns an empty
shell, and /api/v2/solve, /api/v2/solver/list, /openapi.json all 404). Hitachi rows therefore
rest on the operator-adjacent Fixstars Amplify `HitachiClient` documentation and the ACW site
structure, not on the v2 reference text. (iii) I found no MCP server, no energy metric and no
sampling certificate for any vendor — in each case that is an absence in the material examined,
not a proof of absence.  LOCAL ARTEFACTS. All downloaded primary sources are in
/private/tmp/claude-501/-Users-dcharlot-vibe-coding-bmi-
concept/ec64a91f-fbcc-442a-8e9b-f2f378c7a081/scratchpad/vendors/ — fujitsu_da_v4.yaml (the
OpenAPI spec), fj_guide.txt, sqbm_manual.pdf + sqbm_manual.txt (2,926 lines), nec_va20.pdf +
nec_va20.txt, nec_va_x86.pdf + nec_va_x86.txt, kaiwu_pkg/ (unpacked 1.4.1 wheel) and
kaiwu_community-1.0.7/ (the pure-Python modelling source).

## ferrotherm 0.9.0 (rows marked 0.9.0 updated 2026-08-15; the rest surveyed at 0.8.0) — a pure-Rust, zero-dependency (std-only) thermodynamic/Ising computing stack: sparse pairwise EBMs, chromatic block-Gibbs, parallel tempering, a named-variable modelling layer, a `.ftp` program IR with a normative spec, a capability-declaring `Device`/`Fabric` abstraction over real and declared annealing hardware, sampler certificates, and a joules ledger. Workspace = core `.` + `silicon` (Xilinx 7-series bitstream/JTAG) + `serve` (HTTP + MCP) + `cloud` (Hitachi driver). ~15k lines in src/, ~2.7k silicon, ~2.4k serve. Apache-2.0, github.com/dcharlot-physicalai-bmi/ferrotherm.

### 1. Modelling layer (named variables, domains, constraints, objective, answers by name) — **yes**

`model::Model` with `Domain::{Spin, Binary, Categorical(k), Integer{lo,hi}}`, `Var`,
`Lit::{Spin, Is}`, `Expr` (full operator overloads: `5.0 * x.is(2) + y.is(1)*z.is(0)`),
`Sense::{Minimize,Maximize}` folded in at term-arrival so a second call cannot flip an
accumulated objective. `Model::compile() -> Compiled`; `Compiled::decode(&[i8]) -> Solution`;
`Solution::value("temperature")` returns the domain's OWN units via `Compiled::reify` (an
Integer over 10..=20 answers 13, not slot 3). `Model::rename` exists so FFI/node-graph callers
push their names down and errors read `'temperature' takes the integers 10..=20`. Duplicate
names are a hard `CompileError::DuplicateName` because an answer is a name-keyed map. HARSH: no
continuous/real variables at this layer at all — `tla.rs` (Ornstein-Uhlenbeck SPD solves) is a
separate, unconnected module; only ONE objective expression; no sets, permutations, intervals,
or sub-models. Compare against dimod's CQM (REAL vars) or PyQUBO/Amplify.

> /Users/dcharlot/vibe-coding/bmi-concept/research/ferrotherm/src/model.rs (lines 44-111 Domain, 419-747 Model::compile, 958-1156 Compiled, 1179-1234 Solution)

### 2. Encodings (one-hot, domain-wall, binary/log; is the choice exposed?) — **yes** (0.9.0)

**Updated 2026-08-15.** All three findings closed. (a) The choice reaches all nine surfaces now: `categorical_as` (Rust), `ft_model_categorical_as`/`ft_model_integer_as` (C), `encoding=` (Python), `categoricalAs` (Zig), `encoding =` (Julia), and `"encoding"` on HTTP, MCP and the node editor — the last two were still hard-defaulting to one-hot and IGNORING the field, so a document asking for binary got one-hot with a different spin count and no error. An unknown name is now refused by listing the ones that exist. (b) `add_penalty`'s exactness bool is no longer discarded: `Compiled::caveats` names every variable whose encoding no penalty can make exact, and it reaches every surface. Measured, not asserted — for a k=6 binary slot the cheapest INVALID state costs exactly what the cheapest valid one does, and a test enumerates all eight codewords to keep the message true. (c) `Encoding::is_exact` is what feeds it.

All three exist and are correct: `encode::Encoding::{OneHot, Binary, DomainWall}` with
`spins(k)`, `penalty_couplings(k)` (k(k-1)/2 vs 0 vs k-2), `is_exact(k)`, and `Slot::{encode,
decode, add_penalty, width, range}`. Tests enumerate every spin configuration and prove the
penalty's ground states are EXACTLY the k codewords
(`domain_wall_ground_states_are_exactly_the_codewords`). `decode` returns `None` for an invalid
codeword rather than rounding to a guess. Doc corrects the usual mis-statement: domain-wall is
not 'no penalty', it is a CHAIN instead of all-to-all. HARSH, three ways: (a) the choice is
exposed ONLY in Rust, via `Model::categorical_as`; the C ABI, Python `Problem.categorical`, Zig,
Julia, HTTP `solve`, MCP `ferrotherm_solve` and the node-graph editor all hard-default to OneHot
with no parameter — eight of the nine surfaces cannot select an encoding. (b) `Model::compile`
calls `s.add_penalty(&mut b, penalty)` and DISCARDS the returned 'is this penalty exact' bool,
so a Binary-encoded k=6 variable compiles with literally no penalty terms and no warning;
`Encoding::is_exact` is referenced by nothing outside its own tests. (c) any variable appearing
in an expression must be OneHot (enforced with a good error, `CompileError::NeedsOneHot`), so
domain-wall's advantage is available only to constraint-only variables.

> /Users/dcharlot/vibe-coding/bmi-concept/research/ferrotherm/src/encode.rs; src/model.rs:458-461 (categorical_as), src/model.rs:663-665 (return value discarded), src/model.rs:808-813

### 3. Higher-order reduction (k-body → pairwise, with ancillas) — **yes**

`reduce::to_pairwise(&Program) -> Result<Reduction, ReduceError>`: Rosenberg substitution via a
binary multilinear polynomial (`P = 3y + x_a x_b - 2 x_a y - 2 x_b y`), greedy on the pair
appearing in the most wide monomials, converting s=2x-1 in and out. Returns `Reduction{program,
ancillas, original_spins, penalty, offset}` plus `project()` to slice the ancillas off an
answer. `MAX_ARITY = 20` refuses 2^k blowup loudly (`ReduceError::TooWide`). Wired into the
modelling layer: `Expr::product(c, &[lits])` of degree ≥3 is expanded by `Model::expand_product`
(with the s²=1 parity collapse), anything still wider than 2 goes through `to_pairwise`, and the
cost is reported as `Compiled::ancillas`. Correctness is proved by exhaustive enumeration of
BOTH models (`agrees_everywhere`: reduced-minimised-over-ancillas equals original plus ONE
constant for every state). The module doc states the limit honestly: this is a statement about
OPTIMISATION; the Boltzmann distribution over the original variables is not preserved at finite
temperature. HARSH: the ancilla penalty is `2 × Σ|every coefficient in the whole model|`, which
includes every encoding and constraint penalty — on a large model that number dwarfs the
objective landscape and will hurt the annealer; and it is a DIFFERENT policy from the one
`Model::effective_penalty` uses (2 × max objective coefficient), two inconsistent heuristics
inside one compile path. Greedy pair choice with no bound on the ancilla count. Only Toshiba's
PUBO fabric declares `max_arity: 4`; everything else is 2.

> /Users/dcharlot/vibe-coding/bmi-concept/research/ferrotherm/src/reduce.rs (to_pairwise 132-198, penalty 150-152, tests 262-481); src/model.rs:678-728

### 4. Constraint vocabulary (equality, inequality/slack, cardinality, exactly-one, all-different) — **yes** (0.9.0)

**Updated 2026-08-15 (0.9.0).** ALL-DIFFERENT shipped as `Constraint::AllDifferent`, lowered per shared value rather than per pair: it emits nothing where two domains do not overlap, needs no slack and no ancillas, names WHICH value collided in its violation, and refuses the pigeonhole case (more variables than the values they share) at compile time by name rather than annealing a model that has no answer at any penalty. On all eight surfaces. This closes the last row where a surveyed competitor had a modelling capability this stack did not.

`model::Constraint::{NotEqual, Equal, Fix, ExactlyOne, AtMostOne, Cardinality{lits,k},
AtMost{lits,k}, AtLeast{lits,k}}`. Inequalities are real: `Model::compile` declares its own one-
hot slack variables (`Domain::Categorical(k+1)` for AtMost, `lits.len()-k+1` for AtLeast),
constrains `Σ lits ± slack = k` and squares it via `add_squared`; slack is laid out after the
user's variables and never reported in the answer. NotEqual/Equal iterate over
`shared_values(domain_a, domain_b)` rather than slot indices — an Integer 5..=10 and an Integer
0..=5 correctly agree only at 5. The zero-slack edge case (`at most 0 of these`) is handled
rather than silently dropped, with a comment recording that it used to compile to nothing while
reporting feasible. ALL-DIFFERENT: NOT FOUND — no `AllDifferent` variant anywhere in the tree; a
modeller writes O(n²) pairwise `not_equal` by hand. ALSO NOT FOUND: weighted linear constraints
(`Σ aᵢxᵢ ≤ b` with non-unit coefficients — every counting constraint is unit-weighted, even
though the private `add_squared` takes weights), reified/implication constraints, logical
and/or/xor, element/table/global constraints. That is a thin vocabulary next to a decade-old
CP/QUBO modelling layer.

> /Users/dcharlot/vibe-coding/bmi-concept/research/ferrotherm/src/model.rs (Constraint 321-348, slack declaration 624-640, apply_constraint 821-903, add_squared 916-925)

### 5. Penalty handling (auto scaling, feasibility checking, reporting WHICH constraint broke) — **yes**

Strongest part of the modelling layer. `Model::effective_penalty()` = `max(model.penalty, 2 ×
largest |objective coefficient|)`, applied at compile time so a constraint added before the
objective still gets the scaled value (recorded as NaN and resolved late); `fixed_penalty(p)`
opts out; `constrain_at(c, p)` overrides per constraint. Feasibility is checked TWICE and the
two are kept distinct: `Solution::invalid` lists variables whose spins are not a valid codeword,
and `Compiled::check()` re-evaluates every stored `Constraint` against the DECODED values,
returning `Vec<Violation>` where each `Violation` carries a prose `detail` in the caller's own
variable names ('at most 2 of 5 may hold, and 4 do') AND a numeric `amount` (how far outside, in
the constraint's own units) so a caller can rank near-misses. `Solution::feasible()` requires
both to be empty — the comment records that it used to mean only decodability, 'the answer to a
question nobody asked'. `Solution::value()` panics with a message that distinguishes a typo from
an under-weighted penalty. `Compiled::best_of` prefers a feasible answer over a lower-energy
infeasible one. HARSH: the auto-scale is 2×MAX coefficient, not a bound on the objective's
achievable RANGE — 100 terms of weight 1 can outbid a penalty of 2; and `check()` is skipped
entirely when any variable failed to decode (`if invalid.is_empty()`), so the worst failures
report which variables broke but not which constraints. No automatic escalate-and-retry loop.

> /Users/dcharlot/vibe-coding/bmi-concept/research/ferrotherm/src/model.rs (effective_penalty 536-551, decode 988-1002, check 1005-1082, Violation 1164-1177, feasible 1227-1229)

### 6. Embedding / placement onto hardware topology — **yes** (0.9.0)

**Updated 2026-08-15 (0.9.0).** `src/embed.rs` implements the Cai-Macready-Roy heuristic (what minorminer implements): chains, chain strength, rip-up rounds, an identity fast path, and `Embedding::verify`. `None` means not found, never impossible.


NOT FOUND. There is no minor-embedding pass, no clique embedder, no chain-strength calculator,
no working-graph/yield handling anywhere in the tree — `grep -rn embed` over src/, cloud/src/,
silicon/src/ returns only doc prose and the `Caveat::NeedsEmbedding` variant. The crate is
explicit that it does not have one: `Caveat::NeedsEmbedding` Display literally says 'run an
embedder against the machine's own working graph', and `Fabric::verdict` refuses to return a
runnable verdict for any fabric with `native_placement: false` (both D-Wave entries) precisely
because placement is a question it cannot answer. `cloud::hitachi` REFUSES a program whose
couplings are not already King-adjacent under the row-major layout rather than embedding it ('a
driver that placed it for you would be choosing an embedding you did not see').
`Fabric::scale_to_fit` solves the coefficient-RANGE problem (largest positive factor landing
every coupling and field inside the fabric's `Range`, with integrality handled by walking
candidate numerators down from the ceiling), which is a different problem. `src/compile.rs` is a
VARIATIONAL compiler — it fits a target conditional P(y|x) with a device-native Boltzmann kernel
restricted to a hardware patch's actual edges (`Kernel`, `patch_kernel`, `fit`, exact KL and
exact positive/negative-phase gradients) — useful, but it is not placement of an arbitrary QUBO.
Against minorminer/Ocean or Fixstars this is the single largest missing layer.

> /Users/dcharlot/vibe-coding/bmi-concept/research/ferrotherm/src/fabric.rs (Caveat::NeedsEmbedding 228-260, verdict 726-741, scale_to_fit 398-489); cloud/src/hitachi.rs:95-131; src/compile.rs (variational, not placement)

### 7. Samplers / solvers (algorithms, CPU/GPU/hardware) — partial

CPU, broad: `gibbs::Sampler` — chromatic block-Gibbs with `sweep`, `sweep_par(threads)`
(std::thread::scope over disjoint index chunks within a colour class, bit-reproducible for fixed
(seed, threads)), `clamp`/`unclamp`, `read_all`/`read_subset`. `tempering::{anneal,
anneal_scheduled, parallel_tempering, geometric_ladder}` with replica-swap diagnostics.
`sbm::run` — Toshiba's ballistic and discrete simulated bifurcation, symplectic, best-so-far
readout. `het::HetSampler` — heterogeneous factor-graph Gibbs over mixed spin/categorical nodes
with arbitrary-arity energy tables, the general engine `gibbs` is the fast special case of.
`exact::Elimination` — variable elimination giving exact ground state (min-sum) and exact log Z
(sum-product) with min-fill ordering and the induced `width()` reported UP FRONT.
`oracle::{Exhaustive, SteepestDescent(restarts), RandomGuess, Annealer}`. Plus `tla` (OU SPD
solve), `lrw` (ternary-increment SDE), `mppi` + closed-form `Lqr`. All updates funnel through
one `kernel::p_up(field, beta)` — the module doc records that the equation used to be written
six times in three spellings of beta, one of which had no beta. HARSH: no GPU sampler is
callable from Rust, Python, Julia, Zig, HTTP or MCP. `wgsl::sweep_shader()` emits WGSL TEXT
which only the browser pages execute (`docs/ide.html`, `web/gibbs_bench.html`); there is no
`impl Device` for a GPU, so `conform` cannot even score the GPU path. `hdl::FixedFabric` emits
Verilog plus a cycle-exact Q.8 emulator, and the `silicon` crate does real 7-series
bitstream/JTAG, but `PtV2::run` returns an error saying the sequential fabric is not assembled.
No tabu search, no branch-and-bound, no dual/LP/SDP bound, no population annealing.

> /Users/dcharlot/vibe-coding/bmi-concept/research/ferrotherm/src/gibbs.rs:91-150; src/tempering.rs; src/sbm.rs; src/het.rs; src/exact.rs:184-194; src/oracle.rs; src/kernel.rs:26; src/wgsl.rs:90; silicon/src/lib.rs:123-135

### 8. Device abstraction (one interface over multiple vendors, capability declaration) — **yes**

The most differentiated layer. `fabric::Device` trait = `fabric()` / `program(&Program) ->
Vec<Unsupported>` / `run(&Schedule, seed)` / `ledger()`. `Fabric` declares topology
(`AllToAll|Degree(n)|Named(&str)|Unconstrained`), max_spins, max_degree, max_arity,
supports_field, `Precision::{Exact, Unstated, Fixed{bits}, Float{mantissa}}` separately for
couplings and fields, `Range{lo, hi, integral}` for each (so J=0.5 fits D-Wave and no integer
machine), `uniform_couplings` (a fabric that COUNTS neighbours rather than weighting them — a
spin glass cannot be expressed at all), `native_placement`, and `unstated: &'static [&'static
str]` for what the vendor does not publish. `Fabric::check` returns EVERY violation
(`Unsupported::{TooManySpins, TooHighDegree, ArityTooHigh, NoFieldSupport, CouplingPrecision,
NonUniformCouplings, Unplaceable, OutOfRange}`) with remedies named in the message;
`Fabric::verdict` returns `Verdict{caveats}` and refuses to say 'runnable' when placement needs
embedding or a limit is unpublished. `requantize` performs quantisation and RETURNS the relative
error it introduced. Declared fabrics with cited vendor provenance: `dwave_advantage`,
`dwave_advantage2`, `fujitsu_da3`, `toshiba_sqbm`, `toshiba_sqbm_pubo`, `qboson_cpqc`.
Implemented backends: `fabric::Cpu`, `cloud::hitachi::Hitachi` (real HTTPS POST to annealing-
cloud.com/api/v2/solve, King-graph layout, 4-bit coefficients, sign inversion measured not
assumed), `silicon::device::PtV2` (declares 5 neighbours and unweighted couplings because a LUT6
spends one input on the random bit). HARSH: the six vendor entries are DECLARATIONS with no
submission path — you can ask what rules your program out, you cannot run it. The whole device
layer is unreachable from HTTP, MCP, Python, Julia and Zig (`api::dispatch` is
sample/anneal/energy/verify/solve/capabilities only). No async/job-status/batch model.
`Hitachi::run` charges `samples += spins` — one per spin per solve, an invented count, not the
machine's actual update count.

> /Users/dcharlot/vibe-coding/bmi-concept/research/ferrotherm/src/fabric.rs (Fabric 40-89, Precision 100-161, Unsupported 296-362, declared fabrics 496-688, verdict/check 726-877, trait 931-948); cloud/src/hitachi.rs:264-345; silicon/src/lib.rs:36-135

### 9. Verification (certificates, effective temperature, ESS, TV vs exact, conformance) — **yes**

The best layer in the stack and, per its own module doc, an empty lane in the field.
`certify::certify(&Graph, beta_requested, &samples, &trace) -> Certificate` computed FROM
SAMPLES ALONE: `beta_eff` by pseudolikelihood MLE (bisection, not Newton — the comment records
Newton diverging and pinning uniform noise at the clamp), a 95% Fisher CI inflated by sqrt(2τ);
`tau_int` by Sokal automatic windowing taken as the WORSE of the energy trace and the
magnetisation trace (an ordered lattice has fast-jittering energy in a frozen configuration);
`ess = draws/(2τ)`; a Geweke-style early-vs-late drift test; and for n ≤ 20 the TV from the
exact Boltzmann distribution ALWAYS reported beside an explicit `noise_floor =
0.5·sqrt(2^n/ESS)` — 'a distance below the floor is agreement, not accuracy'. Failures are a
`Vec<Finding>` in prose (`BetaMismatch`, `Undermixed`, `AboveNoiseFloor`, `NotConverged`,
`TooFewSamples`); an empty list is the only pass. `conform::run(&mut dyn Device) -> Report`
scores any backend on 7 cases (ferromagnet, frustration, planted optimum, agreement with
variable elimination, determinism, REJECTING a deliberately-bad run, sampling fidelity) and
includes test devices `AlwaysUp` and `Narrow` that the suite must FAIL — 'a suite that cannot
reject anything certifies nothing'. Ground truth beyond enumeration: `exact::Elimination` (exact
ground state and log Z with the induced width reported), `planted::{frustrated_loops, wishart}`
(optimum known at any size by construction), `oracle::RandomGuess` as a null control every
quality test is run against. Format conformance: `spec/ftp-v1.md` is normative and standalone
('where this document and any implementation disagree, this document is correct') with its test
vectors transcribed into `tests/spec_conformance.rs`. HARSH: the noise floor is a heuristic
scaling, not a calibrated bound; TV only to 20 spins; `beta_eff` presumes the device targets a
Boltzmann distribution, so it is meaningless for D-Wave or SBM (the docs say so); no multi-chain
R-hat; and there is no optimality certificate (no dual bound) on the optimisation path.

> /Users/dcharlot/vibe-coding/bmi-concept/research/ferrotherm/src/certify.rs (Finding 31-44, fit_beta 128-182, tau_int 189-213, certify 219-345); src/conform.rs (run 78-, AlwaysUp 250, Narrow 279); src/exact.rs; src/planted.rs; spec/ftp-v1.md; tests/spec_conformance.rs

### 10. Energy / cost accounting (joules or price per operation) — **yes** (0.9.0)

**Amended 2026-08-20 — the row is still `yes`, and it was measuring the wrong half.** Every figure
above is *joules above idle divided by work done*. That prices a machine kept busy, and the case a
sampling substrate is supposed to win is the opposite one: intermittent, low-duty work where the
machine spends most of its life waiting, and where subtracting idle discards where the joules
actually went. `src/duty.rs` (0.17.0) prices the wait — `E over one period = marginal × t_run +
idle × period` — and inverts it into `standby budget = idle + marginal × duty`, the standby power a
challenger must beat *granting it free computation*. As the cadence slackens the budget collapses
onto the incumbent's idle draw and nothing about sampling remains in it. **`Prices` has no standby
term because no vendor in this survey publishes one**, so that comparison is one number away from
being decidable and nobody can decide it; `Machine::beaten_by` takes the number as an argument so
the party that knows it can. Also amended: `Meter::idle` now refuses a baseline above a 1-minute
load average of 2, after a complete and plausible table was produced here on a machine at load 82.
Foreign load *inflates* a baseline, so that contamination overstated the idle share — flattering
this project's own argument, which is the dangerous direction.

**Updated 2026-08-15.** All three findings closed, and the first was worse than reported. (a) `writes` is charged: `Device::program` IS the write, the trait now says so, and both the CPU and Hitachi implementations charge one per node. A demonstration run shows the write at 100% of the projected energy, which is the module's own thesis and was invisible while the term sat at zero. (b) `Prices` carries a `source` naming what the numbers describe, and `Prices::UNSTATED` exists: Hitachi and the CPU declare it rather than borrowing Z1's SPICE estimates, and `Ledger::joules` returns `Option<f64>` — `None`, not zero, because a device nobody has characterised does not cost nothing. The HTTP surface reported Z1 joules for every run including a plain CPU sample; it now reports exact COUNTS always, joules as null when unpriced, and a `priced_as` field generated from the prices rather than hardcoded. (c) `reflash_hz_cap` feeds `Ledger::reflash_seconds`: a workload that reflashes faster than the device sustains is unphysical, and pricing it describes a run that could not have happened. **Closed 2026-08-15:** `ferrotherm-meter` measures wall power and derives per-operation energy from it. macOS backend via `macmon`; std-only, no dependencies. Measured on an Apple M5 Max: **4.261e-7 J per node update**, whole-system above idle, from an 8.29 s window with 75 power readings — against `Z1_SPICE`'s 7.09e-15 J estimate, a ratio of 6.0e7. Those measure different things (a general-purpose CPU at the wall versus a per-device SPICE estimate for unfabricated silicon), so the ratio is the size of the prize being claimed rather than a measured speedup — but one side of it is now measured. Jetson/Linux INA3221 rails are the equivalent and are NOT implemented: the Jetson on this tailnet has been offline for a week, and a backend nobody can run is a backend nobody has tested.

`ledger::Ledger{samples, reads, writes}` with `joules(&Prices)` and `shares()`, and
`Prices{e_sample, e_read, e_write, reflash_hz_cap}`. Every sampler takes `Option<&mut Ledger>`
and charges samples; `Sampler::read_all`/`read_subset`, `compile::Kernel::sample` and the
Hitachi driver charge reads. Provenance labelling is genuinely careful: `Z1_SPICE` is documented
as SPICE estimates for uncharacterised silicon, with the note that 'measured' in the source
paper's prose is a misnomer its own appendix contradicts. HARSH, and this is the module's own
thesis failing: (a) `writes` is NEVER incremented by any library code path — `grep` finds
increments only in `examples/z1_ledger.rs` and `examples/reach_on_z1.rs`, so the term the whole
design rests on ('a write costs ~21,700 samples; any honest account of this hardware class is an
I/O story') is the one term a caller must meter by hand. (b) Exactly ONE `Prices` constant
exists in the entire tree, and every fabric declares `prices: Z1_SPICE` — including Hitachi's
CMOS annealing ASIC and the Alchitry FPGA, so a Hitachi run's reported joules are Extropic's
pre-silicon numbers applied to a different vendor's machine. (c) `reflash_hz_cap` is declared
and read by nothing. (d) No currency/price-per-solve, no wall-clock power measurement, no
measured energy API — the measured GPU/CPU flips-per-second and package-watt figures live in
README prose, not in code. The HTTP/MCP reply does label the field `joules_z1_spice` and states
the provenance inline, which is the right instinct.

> /Users/dcharlot/vibe-coding/bmi-concept/research/ferrotherm/src/ledger.rs (whole file, 67 lines); src/gibbs.rs:153,161; cloud/src/hitachi.rs:287 (prices: Z1_SPICE), :339; examples/z1_ledger.rs:48,71,77,94; serve/src/api.rs:172-191

### 11. Language surfaces (which languages, native vs FFI vs subprocess) — **yes**

Nine surfaces, and — unusually — Python/Zig/Julia reach the MODELLING layer, not just the
sampler. Rust: native, `[dependencies]` is EMPTY, std-only, wasm-clean. C: `src/ffi.rs` (~90
`#[no_mangle] extern "C"` functions incl. the whole `ft_model_*` family — declare, constrain,
objective incl. `ft_model_objective_product` for higher-order, compile, solve, read by handle,
violations with amounts, `ft_model_ftp`) plus a hand-written `include/ferrotherm.h`. wasm:
`crate-type=["rlib","cdylib"]`, `docs/ferrotherm.wasm` (317 KB). Python:
`python/ferrotherm/__init__.py` (991 lines) via CTYPES over the C ABI — FFI, no build step,
ships a prebuilt wheel; exposes `Problem/Variable/Literal/Term/Answer` with operator overloading
and `Sim.certify()`. Zig: `zig/ferrotherm.zig` (803 lines) via `@cImport` of the header — FFI,
converts sentinel returns into a Zig error set. Julia: `julia/Ferrotherm/src/Ferrotherm.jl` (973
lines) via `ccall` — FFI, with a `ferrotherm_jll` artifact and two package extensions, including
a real `QUBODrivers`/MOI driver so any JuMP model `ToQUBO.jl` can reformulate becomes a
workload, tested against `QUBODrivers.ExactSampler` on random instances to catch the sign
inversion. HARSH: no C++, no npm/JS package, no Go, no MATLAB. Python is ctypes-only — no native
extension, so no zero-copy numpy beyond a manual `.numpy()`. The `Encoding` choice, the
`Fabric`/`Device` layer, `reduce`, `exact`'s log Z, `conform`, `het`, `program` and `dtm` are
ALL Rust-only. ECOSYSTEM.md's architecture diagram draws `dimod · OMMX · QUBODrivers · MOI` as
the adapter row, but only QUBODrivers/MOI exist — ROADMAP.md defers dimod and ommx to 'later',
so the diagram exceeds the code.

> /Users/dcharlot/vibe-coding/bmi-concept/research/ferrotherm/Cargo.toml; src/ffi.rs; include/ferrotherm.h; python/ferrotherm/__init__.py; zig/ferrotherm.zig; julia/Ferrotherm/src/Ferrotherm.jl; julia/Ferrotherm/ext/FerrothermQUBODriversExt.jl; ECOSYSTEM.md:55 vs ROADMAP.md:445

### 12. Agent / AI surfaces (MCP, HTTP API, structured tool schemas) — **yes**

MCP over stdio, JSON-RPC 2.0 line-delimited, protocol `2025-06-18` with
`2025-03-26`/`2024-11-05` fallbacks: SIX tools with full JSON Schema — `ferrotherm_sample`,
`_anneal`, `_energy`, `_verify`, `_solve`, `_capabilities`. HTTP: `GET /v1/health`, `GET
/v1/capabilities`, `POST /v1/{op}`, both transports dispatching through the SAME `api::dispatch`
so 'HTTP and MCP cannot drift apart', with a test asserting every advertised tool name is
routable. The schemas are unusually well written for an agent audience: `ferrotherm_solve`'s
description tells the model to prefer it over `_anneal`, to check `feasible` before trusting
values, that a false means either `did_not_decode` or `violated`, that a penalty makes a
constraint expensive rather than impossible, and to lengthen the schedule rather than raise the
penalty when a model stays infeasible. `api::capabilities()` self-describes operations, graph
spec, limits (`MAX_NODES`, `MAX_NODE_UPDATES`, exact-verification cap) and determinism.
`AGENTS.md` and `llms.txt` exist for agent onboarding. Errors come back as `isError` tool
results, correctable rather than fatal. HARSH: no tool reaches the `Device`/`Fabric` layer,
`reduce`, `exact`'s log Z, `conform`, `.ftp` round-trip, or the encoding choice — an agent gets
the sampler and the modelling layer and nothing else. No streaming or long-running job model; a
request is synchronous under a node-update ceiling. `solve` exposes only the unit-weighted
constraint vocabulary of layer 4.

> /Users/dcharlot/vibe-coding/bmi-concept/research/ferrotherm/serve/src/mcp.rs (tools() 50-155, dispatch 221-231); serve/src/http.rs:53-58; serve/src/api.rs (dispatch 818-830, capabilities 471-523)

### 13. Training (energy-based model training, gradient estimators) — **yes**

Three independent lines, each verified against exact enumeration before the sampled path is
trusted. `program.rs` — stochastic differentiable programs (typed binary/real wires,
parametrised stochastic `Gate`s) with THREE cross-validated gradient routes: `reinforce_grad`
(score function, exact trajectory log-density including through a full Gibbs kernel — every spin
update is a Bernoulli with known probability, so no approximation), `pshift_grad_pnot`
(parameter shift for sigmoid-mixture gates with common random numbers), `ebm_kernel_grad`, and
`fd_grad` as the finite-difference referee all three must agree with (`examples/grad_check.rs`:
-0.1922 / -0.1922 / -0.1926). `compile.rs` — exact positive/negative-phase gradients of
KL/cross-entropy for a device-native Boltzmann kernel (`ce_grad`, `ce_grad_onehot` with the
hidden-only positive phase, `apply_grad`, `fit`, `factor_eps`), plus
`exact_conditional`/`kl_from_target` on enumerable kernels. `dtm.rs` — Denoising Thermodynamic
Models: chain of shallow BMs over a closed-form forward noising process, `Ebm` with chromatic
Gibbs, `Dtm::train_step` contrastive updates, `exact_log_cond`/`exact_nll` for verification,
ACP. The DTM doc records catching a printed SIGN ERROR in the source paper's Eq. D1 by a keep-
probability test. HARSH: no autodiff framework and no optimizer library — `apply_grad` is plain
SGD with a scalar lr; the exact routes cap at ~13 free bits (2^n_free enumeration); no
minibatching infrastructure, no GPU training, no PyTorch/JAX interop, and none of this is
reachable from Python/Julia/Zig/HTTP/MCP.

> /Users/dcharlot/vibe-coding/bmi-concept/research/ferrotherm/src/program.rs (reinforce_grad 170, pshift_grad_pnot 203, ebm_kernel_grad 235, fd_grad 276); src/compile.rs (ce_grad 112, ce_grad_onehot 259, apply_grad 352, fit 480); src/dtm.rs (train_step 271, exact_nll 390)

### 14. Visual / graph programming — **yes**

`docs/graph.html` — a 48 KB self-contained node-graph editor (no CDN, no build step) that drives
the wasm build's `ft_model_*` C ABI directly in the browser. Node palette from `const TYPES`:
variables (`categorical`, `integer`, `binary`), constraints (`notequal`, `equal`, `fix`,
`cardinality`, `atmost`, `atleast`), objectives (`prefer`, `agree`, `together` — variadic, i.e.
a genuine higher-order term the compiler lowers with ancillas), and `solve` (variadic, with
tries and the full beta ladder), `certify`, `report`. Ports are TYPED and a wire may only join
ports of the same kind, so 'a graph that connects is a model that compiles'; a node whose call
was refused gets `.bad` styling and the library's own sentence via `ft_model_error`. The panel
shows the compiled `.ftp` text (`ft_model_ftp`), spin count, effective penalty, ancilla count,
feasibility and violations. Graphs save/load as JSON (`graph.ferrotherm.json`), deliberately
text 'a person can diff and an agent can author'. Playwright tests in `web-tests/`
(editor/workbench/live, 21 tests). A second page `docs/ide.html` is a workbench that also runs
the emitted WGSL on the visitor's real GPU via `navigator.gpu`. HARSH: no
`exactly_one`/`at_most_one` nodes (the two cheapest constraints, which need no slack); no
encoding choice; no device/fabric node, so the graph can only ever run on the local wasm
sampler; and the counting nodes are fixed at four inputs — only `together` and `solve` grow
ports.

> /Users/dcharlot/vibe-coding/bmi-concept/research/ferrotherm/docs/graph.html (TYPES 173-215, wasm binding 419-680, export 753); docs/ide.html:788-824; web-tests/

### 15. Licence, openness, and whether hardware is required — **yes**

Apache-2.0 throughout (`LICENSE`, full 202-line text; `license = "Apache-2.0"` in Cargo.toml,
serve, cloud, silicon), single public repo github.com/dcharlot-physicalai-bmi/ferrotherm,
published to crates.io with docs.rs. Dependency posture is genuinely unusual: the core crate's
`[dependencies]` section is EMPTY (std only), `serve` is also zero-dependency down to its own
JSON parser in `serve/src/json.rs`, `cloud` adds only `ureq` for TLS and is explicitly deletable
('nothing here is named by the core'), `silicon` gates FTDI behind a `flash` feature. NO
HARDWARE IS REQUIRED for anything: CPU is the default, wasm + WebGPU run in a browser tab with
no install, and every hardware path fails honestly rather than silently — `PtV2::run` returns
'the sequential fabric is not yet assembled: flip-flops and clock are pending', and Hitachi
needs only a free Annealing Cloud Web token read from `ACW_TOKEN` (never committed).
Reproducibility is enforced: PCG RNG, deterministic per (seed, thread count), and the HTTP
capabilities endpoint states it. Evidence it runs: I executed `cargo test --lib` in-tree — 263
passed, 0 failed, 497 s. HARSH on doc drift: README's Verification section still claims 'cargo
test — 6/6' against 263 actual library tests, and claims 'a 44 KB .wasm' while
`docs/ferrotherm.wasm` is 324,936 bytes (317 KB). ECOSYSTEM.md's fabric matrix marks WebGPU '✅
done' although no `impl Device` exists for it, so `conform` cannot score it, and it lists QBoson
twice (once 'declared', once 'planned').

> /Users/dcharlot/vibe-coding/bmi-concept/research/ferrotherm/LICENSE; Cargo.toml; cloud/src/lib.rs:1-6; silicon/src/lib.rs:123-135; README.md:56-100; ECOSYSTEM.md:99-125; measured: cargo test --lib → 263 passed

**Notes.** LICENCE: Apache-2.0, uniformly, across all four workspace crates; single public repo; published
on crates.io/docs.rs. LANGUAGE: Rust (edition 2021) for everything; the core crate has an EMPTY
[dependencies] table (std-only, wasm-clean), serve is also zero-dependency including its own
JSON parser, cloud adds only `ureq`, silicon gates FTDI behind a feature. Every non-Rust surface
(C, Python, Zig, Julia, wasm) is FFI over one hand-written C ABI in src/ffi.rs — no subprocess
anywhere, no Python native extension (ctypes only). HARDWARE: not required for any capability.
CPU is the default path; browser wasm and WebGPU need no install; Hitachi needs only a free
Annealing Cloud Web token; the FPGA path refuses honestly when no board is attached. EVIDENCE: I
ran `cargo test --lib` in-tree — 263 passed, 0 failed, 497 s — so the claims above are backed by
a green suite, not just by reading.  SCALE, for calibration against a decade-old stack: ~15.0k
lines in src/, ~2.7k in silicon/, ~2.4k in serve/, ~1.0k Python, ~0.8k Zig, ~1.0k Julia. This is
one person-scale work at high density, not a decade of accumulated modelling features.  WHERE IT
WINS relative to the field it is being compared against: (9) verification — a sampler
certificate computed from samples alone, with beta_eff, ESS, TV beside an explicit noise floor,
and a conformance suite containing devices it MUST fail; (8) capability declaration — Precision
as a type (Exact/Unstated/Fixed/Float), Range with an integrality flag, uniform_couplings,
native_placement and an explicit `unstated` list, with a Verdict type that refuses to promise a
run; (11) surface breadth — nine surfaces, of which Python, Zig, Julia, wasm, HTTP, MCP and the
node graph all reach the MODELLING layer rather than only the sampler; (15) zero dependencies
and no hardware requirement.  WHERE IT IS THIN, stated plainly: (6) no embedder at all — the
single biggest gap, and it is admitted in the code rather than papered over; (4) no all-
different, no weighted linear constraints, no reification, no logical/global constraints; (7) no
GPU sampler callable from the library (WGSL is emitted text the browser runs) and no `impl
Device` for GPU, so the conformance suite cannot score it; (10) the ledger's `writes` channel —
the term the module's entire thesis rests on — is never charged by library code, and one price
table (Extropic's Z1 SPICE estimates) is applied to every fabric including Hitachi's ASIC and an
Alchitry FPGA; (2) the encoding choice exists only in the Rust API and `Model::compile` discards
`Slot::add_penalty`'s exactness flag; (1) no continuous variables in the modelling layer; (8)
the six vendor fabrics are declarations with no submission path, and the device layer is
unreachable from every non-Rust surface.  DOCS EXCEEDING CODE (three, all minor and none load-
bearing): ECOSYSTEM.md:55 draws `dimod · OMMX` in the adapter row while ROADMAP.md:445 defers
them and neither exists; ECOSYSTEM.md's matrix marks WebGPU '✅ done' though no Device impl
exists for it; README.md still says 'cargo test — 6/6' (actual: 263) and 'a 44 KB .wasm'
(actual: 317 KB). The docs' general register is the opposite of inflated — the crate repeatedly
records its own retracted claims in comments, marks vendor figures as vendor figures,
distinguishes 'declared' from 'running' in its own support matrix, and writes 'this review did
not locate' rather than 'does not exist'.
