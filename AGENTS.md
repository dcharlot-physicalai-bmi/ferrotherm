# ferrotherm — guidance for AI agents

This crate is designed to be driven by coding agents. Everything below is enforced by tests, so
you can trust it without reading the source first.

## What this library is

Pure-Rust thermodynamic computing: sparse energy-based models, chromatic block-Gibbs sampling,
simulated annealing and parallel tempering, thermodynamic linear algebra (OU-network solves),
stochastic differentiable programs (score-function + parameter-shift gradients), a variational
compiler that fits conditional kernels onto hardware graph topologies, and a joules ledger that
prices every sample, read, and write against swappable device models. Zero dependencies, std-only,
compiles unchanged to wasm32-unknown-unknown, deterministic for a fixed seed.

## Invariants you can rely on

1. **Every module is verified against exact physics or closed forms.** The Gibbs sampler
   reproduces exact Boltzmann distributions on enumerable systems and Onsager's 2D solution;
   parallel tempering finds exhaustively-verified ground states; the TLA solver matches Gaussian
   elimination; all three gradient estimators agree with finite differences; the compiler's exact
   gradients match FD to 1e-6. If you change something and `cargo test` passes, these still hold.
2. **Determinism**: same seed, same draws, every platform. Never use wall-clock or OS randomness
   inside the library.
3. **The ledger is honest**: `ledger::Prices` describes a DEVICE MODEL. `Z1_SPICE` is a
   pre-silicon vendor estimate (arXiv:2608.01615 Table IV) and must stay labelled as such.
4. **Idle is part of the bill, and a busy machine has no idle.** Energy figures in this tree were
   all joules ABOVE IDLE divided by work done, which prices a machine kept busy and silently
   assumes the case a sampling substrate is worst at. `duty` prices the wait. And before taking any
   power baseline, `Meter::idle` checks the load average and refuses above 2 runnable threads:
   foreign load INFLATES a baseline, so contamination overstates the idle share — the direction
   that flatters this project's own argument. A contaminated measurement that supports the thesis
   is the dangerous one. Withdraw and re-measure; do not publish with a caveat.
5. **No dependencies.** Do not add any. The zero-dep property is a product feature (auditability,
   wasm size, longevity), not an accident.
6. **A value is what the modeller wrote.** Slot indices exist only inside `Model::linearise`, which
   is the one place that knows a domain's layout. Every public surface takes and returns the
   domain's own values. If you add a domain, give it its own arm in `Domain::values`, `index_of`
   and `Compiled::reify` — a catch-all is how a spin variable came to report 0 and 1 to the reader
   while the decoder handed back -1 and +1.
7. **A sample set knows what it may be asked.** `samples::SampleSet` carries the distribution its
   states came from, and expectation values are REFUSED where there is none. Averaging spins over
   the states a tabu search visited yields a number of exactly the shape of `<s_i>` and estimates
   nothing. If you add a producer, give it the right `Provenance` — mislabelling one as `Chain`
   makes `tau_int`, the drift check and every error bar downstream believe an order that is not
   there. And every estimate's error bar is `sqrt(var/ess)`, never `sqrt(var/N)`; the difference is
   measured against exact enumeration in `examples/interval_calibration.rs`, where the naive
   interval covers 24% while announcing 95%.
8. **Reading a state costs the device.** Take draws through `Sampler::collect` (or `collect_par`),
   never by cloning `smp.s` in your own loop. On a Z1-class device a read is 1.692 pJ per node
   against 7.09 fJ per Gibbs cycle — one read is worth 239 updates — and five hand-written
   collection loops in this repository each reported their readback as exactly zero, which on the
   HTTP endpoint was 98.9% of the answer's energy.
9. **A default is not a fallback.** An input the code cannot understand is an error naming what
   was actually sent. Substituting a default for an unreadable value is how `"maximize": 1`
   silently minimised and `"value": "13"` silently pinned a variable to 0.

## How to do common tasks

- **Solve a problem, rather than sample a graph**: `model::Model` — declare variables
  (`categorical`, `integer`, `binary`, `spin`), state constraints (`not_equal`, `equal`, `fix`,
  `cardinality`, `at_most`, `at_least`), add objective terms with arithmetic
  (`5.0 * x.is(2) + 2.0 * y.is(1)`), then `compile()?.solve_best_of(n)` and read values BY NAME.
  This is the layer to reach for first; everything else in this list works in spins.
  - Values are the modeller's own. An integer over `10..=20` takes the value 13; passing 3 is an
    error naming the range, not a slot index. Same on every binding.
  - `objective()` ACCUMULATES and folds its sense in per term, so writing one term per option in a
    loop works and a minimising term does not re-interpret the maximising ones. `set_objective()`
    replaces.
  - Check `feasible()` before reading. It means every variable decoded AND every constraint holds.
    When false, `violated` names the constraints the objective outbid and `invalid` names the
    variables whose encoding lost. A penalty makes a constraint expensive, not impossible: raise
    `fixed_penalty`, or lengthen the ladder with `solve_best_with`.
- **Sample an Ising model**: `ising::lattice2d` or `ising::ring` → `gibbs::Sampler::new` →
  `sweeps(n, Some(&mut ledger))`. For device topology: `device::z1_grid`.
- **Optimize (ground states / QUBO-like)**: `tempering::parallel_tempering` with
  `geometric_ladder`; check `swap_rates` are in ~[0.2, 0.6].
- **Solve SPD linear systems thermodynamically**: `tla::solve_spd`; verify against
  `tla::solve_exact` when the size allows.
- **Train a stochastic program**: build `program::Program` from `Gate`s, use `reinforce_grad`
  (any gate) or `pshift_grad_pnot` (flip gates), and always cross-check a new gate's gradient
  against `fd_grad` before trusting training results.
- **Compile a conditional onto a device graph**: `compile::patch_kernel` (roles: input/output/
  hidden on a device patch) → `fit` / `ce_grad_onehot` → check `factor_eps`; per-factor KLs sum
  to the chain bound (readout KL ≤ Σ ε).
- **Drive from wasm / another language**: the `ffi` module (`ft_*` functions, C ABI), declared in
  `include/ferrotherm.h`. The whole modelling layer is there as `ft_model_*`. Build with
  `cargo build --release --lib --target wasm32-unknown-unknown`.
- **Drive from Python**: `python/ferrotherm` — `Problem`, `Variable.is_(v)`, arithmetic objectives,
  `solve()`, `certify()`. It loads the built cdylib, so `cargo build --release` first.
- **Drive as an agent**: `ferrotherm-serve` speaks MCP over stdio and HTTP. Six tools; use
  `ferrotherm_solve` unless you specifically want spins. `serve/tests/agent_harness.rs` is the
  reference for what a client should be able to discover from the protocol alone.
- **Drive the node editor**: `window.ferrotherm` in `docs/graph.html` — `await ready`, then
  `types()`, `add`, `set`, `connect`, `find`, `run()`. `web-tests/` drives exactly that surface.

## House rules for contributions

- New algorithm modules must ship with a verification test against an exact result (enumeration,
  closed form, or an independently-computed reference) in the same PR. A benchmark is not a test.
- A test must be able to FAIL on the bug it names. `scripts/mutation-check.sh` does this: it breaks
  the code on purpose and requires a named test to notice. Several tests here passed while the thing
  they described was broken, because the case they chose made the right and wrong answers coincide —
  and three more were found blind because they never reached the branch they were named after.
  - **Commit before mutating.** The script restores from git. Twice in one day an uncommitted
    afternoon was destroyed this way, once by hand and once by the script's own cleanup.
  - Read its verdicts literally. "MUTATION DID NOT APPLY" and "NO TEST MATCHED" are not passes;
    they are the two ways a mutation check has silently lied here before.
- A new capability lands on every surface it belongs on, or the gap is written down. The matrix is
  Rust / C header / Python / Zig / Julia / node editor / HTTP / MCP.
  - **Declaring a symbol a binding never calls is gaming the gate, not passing it.** `check-parity.sh`
    greps for the name; three `ft_hubo_*` entry points were "reachable" from Python and Julia only
    because signatures had been declared for them and nothing called them. The honest move was to
    delete the declarations and write three EXEMPT lines saying why a borrowed state pointer, a
    pending-list count and a fixed-arity node-graph form belong to the wasm path and not to a
    high-level binding.
  `scripts/check-parity.sh` enforces the four that hang off the C ABI, and a gap passes only with a
  reason in its EXEMPT table. It is in CI. Written-down means written there, not remembered here --
  the rule was broken twice while it lived only in this file, and its first run found sixteen real
  gaps including the ancilla count, which is the number that says whether sampling a solved model is
  sound at all.
  - **`hubo` reaches seven of the eight, and the eighth is written down here.** Rust, C header,
    Python, Zig, Julia, HTTP, MCP and the browser workbench all take a higher-order model. The NODE
    EDITOR does not, and should not as it stands: it is an editor for the MODEL layer -- named
    variables, domains, constraints -- and `hubo` is spins and terms. Wiring a spins-and-terms node
    into a graph whose whole value is that answers come back in the modeller's own words would be
    putting two vocabularies on one canvas. The honest form would be a model-layer node that
    compiles to `hubo` instead of to the reduction, which is a compiler change and not an editor
    one. Until that exists, the gap is this paragraph.
  - **The node editor is a surface, and check-parity.sh does not look at it.** That exemption was
    not deliberate and it cost three constraints: the model layer had nine, the C ABI reached all
    nine through `ft_model_close`'s kind codes, and the editor called kinds 0, 1 and 2 — so
    `all_different` was unsayable in a picture for as long as nothing compared the lists.
    `scripts/check-editor-parity.sh` now does, with the same EXEMPT-with-a-reason rule, and
    `scripts/check-editor-model.sh` puts one model through the API and through `fromModel` and
    compares the COMPILED size — because vocabulary parity passes an editor that silently drops a
    `k`, which still draws every node type, still runs, and answers a different question. Both are
    in CI and both carry a `--selftest` that damages the thing under test and demands a failure. Run
    it: the first version of the parity self-test PASSED an editor with `all_different` cut out,
    because an unconditional assignment at the top of the script overwrote the path the self-test
    was pointing at it.
- State provenance on every number in docs: measured / simulated / projected, and on what.
- Follow the failure-analysis discipline: if a method underperforms, measure WHERE it fails
  (train vs held-out vs on-policy; per-factor vs end-to-end) before concluding anything.

## Ecosystem

- Live instrument + teaching page: https://energy.physicalai-bmi.org/thermo
- Program context (Energy First Architecture): https://efa.physicalai-bmi.org
- Institute: https://physicalai-bmi.org
