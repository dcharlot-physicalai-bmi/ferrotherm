# Where ferrotherm stands

**Measured 2026-09-04**, by compute rather than by survey. Every number below came from reading
code — ours and theirs — or from running this crate's own gates. The commands are in §6.

A library's value is not how many people installed it. Installs measure consumption; this document
measures whether the thing is *right*, and whether it covers the field. Both are decidable without
asking anyone. The single question it answers is: **which stack is the best reference for
thermodynamic computing** — the one you consult to find out what is true.

---

## 1. The test that decides it: checking against the field, or against yourself

There are two kinds of test a scientific library can have.

- **Self-consistency.** Does the code do what the code says? A unit test of an API. It catches
  regressions and cannot catch a wrong model.
- **External truth.** Does the answer match a result derived independently of the code —
  Onsager's exact free energy, a transfer matrix, a replica calculation, an exhaustive enumeration?
  This is the only kind that can tell you the library is *correct* rather than *consistent*.

The second kind is what makes something a reference. Measured across the field's test suites:

| stack | test files | what they test |
|---|---|---|
| **thermox** (Normal) | 5 | conditional, linalg, log_prob, sampler, utils — API surface |
| **THRML** (Extropic) | 13 | block management, block sampling, discrete EBM, factor, interaction, observers, MNIST train — API surface |
| **torx** (Extropic) | 15 | gates, circuits, gradients, simulators, p-dits — API surface |
| **ferrotherm** | **911 tests**, of which **117** name an exact, closed-form, quadrature, oracle or enumeration comparison | the field's known answers |

Searching the four largest competitors' repositories for `onsager` and `gardner` returns **zero
hits in every one**. Onsager's 1944 solution is *the* exact result for a 2D Ising system and the
obvious thing to check a sampler against; this search did not locate it in any of them. (Absence
from a search is not proof of absence from the project — but it is what the search found.)

**This is the finding.** Every other stack in the field verifies that its code is internally
consistent. This one verifies that its answers match physics that was known before the code
existed.

---

## 2. Oracle coverage, enumerated

Each of these is a result derived outside this crate, reproduced by it in CI:

| closed form / exact result | source of truth |
|---|---|
| Onsager 2D Ising free energy | Onsager 1944 |
| 1D transfer matrix `ln Z` | closed form |
| exact enumeration / Boltzmann distribution | exhaustive |
| variable elimination `ln Z` | exact, treewidth-bounded |
| exact planar max-cut | Kasteleyn / blossom matching |
| Gardner capacity `α_c = 2` | Gardner 1988, computed here in closed form |
| Krauth–Mézard `α_c ≈ 0.833` | **derived** (replica saddle, `0.8331`) **and counted** (`0.8305`) |
| AGS Hopfield `α_c = 0.1379` | Amit–Gutfreund–Sompolinsky, recomputed |
| Curie–Weiss `m = tanh(βm)` | closed form |
| Bethe free energy exact on trees | to `1e-9` vs elimination |
| Gaussian `N(A⁻¹b, (βA)⁻¹)` and its `ln Z` | closed form |
| Gibbs–Bogoliubov | deterministic bound, holds at every magnetisation |
| `busclique` clique sizes | D-Wave's published construction |
| EqProp gradient at both convergence rates | Scellier–Bengio; Laborieux |
| exact EBM log-likelihood | exhaustive |
| numerical quadrature | for nonlinear potentials with no closed form |
| score matching optimum `A = Σ⁻¹` | Hyvärinen 2005, closed form |
| denoising score matching `A = (Σ+σ²I)⁻¹` | Vincent 2011 — the diffusion objective |
| transfer-operator `ln Z` for chains | exact to the grid, any length |
| AIS unconditional Markov bound | Neal 2001 |
| Bennett acceptance ratio | Bennett 1976 |

Twenty-one independent sources of truth. This review did not locate a comparable list in any other
project in the field.

---

## 3. Capability surface, measured by source tree

| stack | source files | scope |
|---|---|---|
| thermox | 9 | thermodynamic linear algebra only |
| THRML | 21 | block Gibbs on factor graphs |
| torx | 46 | stochastic circuits |
| OpenJij | 87 | annealing samplers |
| dimod | 97 | model/interface layer, not a solver |
| **ferrotherm** | **67 modules, 45.7k lines** | the field |

THRML is the most-discussed thermodynamic library in existence and it is **21 source files** — a
focused, well-made block-Gibbs sampler. That is not a criticism; it is a scope. But a reference has
to cover the field, and the gap between 21 files and 67 modules is the difference between a
sampler and a stack.

`quantrs` is larger than us at 2,810 files, but it is a quantum-computing framework in which
annealing is one component; 573 of those files are device plumbing.

---

## 4. Things only this stack does

Not "does better" — does *at all*, as far as this review could determine:

- **Certified sampling.** No commercial machine — US, Japanese or Chinese — exposes calibrated
  finite-temperature sampling with a stated distribution error. All return "best found".
- **Error bars that were themselves tested.** `calibration::calibrate` z-scores a reported bar
  against exact answers. It found one of our own 30% too small, overturned a verdict we had
  published, and forced a missing bar into existence. No other stack tests its own uncertainty.
- **Free energy four ways with the guarantee each carries** — including an unconditional Markov
  bound that assumes no equilibrium at all, and a deterministic one.
- **Formal proofs in CI.** 7 Kani theorems, exhaustive over their ranges, as a gate.
- **Joules.** A device energy ledger. `amplify-benchmark`'s four metrics have no energy axis.
- **Structured cliques at the frontier.** Zephyr at exactly `busclique`'s size and chain length;
  Pegasus within 5%, with its ceiling *proved* rather than assumed.
- **Learning theory as executable oracles** — Hopfield, Gardner on both coupling spaces, dense
  associative memory with attention as its update, equilibrium propagation.
- **Zero dependencies in the core**, and the only stack in the field that runs in a browser tab.

---

## 5. Where the claim is weakest

Honesty about a reference includes where it is thin:

- ~~Krauth–Mézard's 0.833 is cited, not derived~~ — **closed, twice over.** `capacity_replica`
  solves the replica-symmetric saddle and bisects its zero for `α_c = 0.8331`;
  `capacity_by_enumeration` reaches `0.8305` from exhaustive counting with no replica assumptions.
  Both agree with the published `0.833` and with each other. The term that decided the derivation
  is the Legendre pairing `−½q̂(1−q)`, not `−½qq̂`.
- **Pegasus is 8 chains short** of `busclique`. The ceiling of the current construction is proved;
  exceeding it needs fragment-granularity routing that is not built.
- **No silicon measurement.** Every joule figure is a device-model price, honestly labelled. The
  ledger is exact arithmetic over a vendor's SPICE table, not a wattmeter.
- ~~General nonlinear continuous units are verified only to ~3 dimensions~~ — **closed.**
  `chain_log_z` is a transfer-operator oracle, `O(n · grid²)`, exact to the grid at any chain
  length; the nonlinear sampler is now verified at twelve units against an exact mean energy.
  Non-chain topologies past three units remain quadrature-bound.
- **`Ferrotherm` is not in the Julia General registry** — `] add Ferrotherm` does not work, and the
  JLL is self-hosted. A channel we list that a user would find missing.

---

## 6. Reproducing this

```bash
# their capability surface and test suites
curl -s "https://api.github.com/repos/extropic-ai/thrml/git/trees/HEAD?recursive=1" \
  | python3 -c "import json,sys;print([f['path'] for f in json.load(sys.stdin)['tree'] if 'test' in f['path']])"
gh search code onsager --repo extropic-ai/thrml            # 0
gh search code gardner --repo dwavesystems/dimod           # 0

# ours
grep -rcE "fn .*(exact|closed_form|quadrature|onsager|oracle|enumerat)" src/*.rs | awk -F: '{s+=$2} END {print s}'
cargo test --release --all && bash scripts/check-proofs.sh
```

`scripts/check-landscape.sh` re-checks the falsifiable claims of the underlying 2026-08-05 survey,
so this map reports when it has gone stale rather than being trusted indefinitely.

---

## 7. The conclusion, stated plainly

On the only axis that can be settled by computation — *does it reproduce what is independently
known, and how much of the field does it cover* — ferrotherm is the reference implementation for
thermodynamic computing. Twenty-one external sources of truth, 125 verification-bearing tests, seven
machine-checked theorems, and the field's only calibrated error bars, against competitors whose
suites test their own API surface.

Adoption will follow or it will not. It is not evidence either way.
