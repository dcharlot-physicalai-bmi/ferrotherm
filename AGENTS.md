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
4. **No dependencies.** Do not add any. The zero-dep property is a product feature (auditability,
   wasm size, longevity), not an accident.

## How to do common tasks

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
- **Drive from wasm / another language**: the `ffi` module (`ft_*` functions, C ABI). Build with
  `cargo build --release --lib --target wasm32-unknown-unknown`.

## House rules for contributions

- New algorithm modules must ship with a verification test against an exact result (enumeration,
  closed form, or an independently-computed reference) in the same PR. A benchmark is not a test.
- State provenance on every number in docs: measured / simulated / projected, and on what.
- Follow the failure-analysis discipline: if a method underperforms, measure WHERE it fails
  (train vs held-out vs on-policy; per-factor vs end-to-end) before concluding anything.

## Ecosystem

- Live instrument + teaching page: https://energy.physicalai-bmi.org/thermo
- Program context (Energy First Architecture): https://efa.physicalai-bmi.org
- Institute: https://physicalai-bmi.org
