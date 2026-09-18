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
| **thermox** (Normal) | 5 | `expm` against `jax.scipy.linalg.expm`, and the identities `Ax≈b`, `AA⁻¹≈I`, at `atol=1e-1` |
| **THRML** (Extropic) | 13 | an exact Boltzmann law **by enumeration** at small `n`, at `max_err < 0.02`; the rest is API surface |
| **torx** (Extropic) | 15 | gates, circuits, gradients, simulators, p-dits — API surface |
| **ferrotherm** | **2000 tests**, of which **441** name an exact, closed-form, quadrature, oracle or enumeration comparison | the field's known answers |

> **CORRECTED 2026-09-17.** Two of these rows previously read "API surface", and that was wrong.
> `thrml/tests/test_ising.py` compares sampled against exact enumerated Boltzmann distributions
> (`assertLess(max_err, 0.02)`), and `thermox/tests/test_linalg.py` compares against
> `jax.scipy.linalg.expm` and against mathematical identities (`atol=1e-1`). Both were read
> directly before this correction was made. **These are real oracles and the earlier claim
> understated them.** What survives is narrower and still a difference in kind: their oracles are
> *enumeration and self-consistency at a size you can brute-force*, at tolerances of `1e-1` and
> `2e-2`; the comparison below is against *closed-form results from the literature that hold in the
> thermodynamic limit*. Enumerating a small lattice checks your arithmetic. Onsager checks your
> physics.

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
- **Joules per *independent* sample.** The field quotes joules per flip, and nobody buys flips.
  `meter/examples/joules_per_sample` measures both on real silicon and they name **different
  winners on the same run**: at β = 0.5 parallel tempering is 2.6× worse per flip and 1.9× better
  per independent sample; at β = 1.0 it is ≥604× better, because plain Gibbs did not decorrelate
  once in the budget. This review did not locate an ESS-corrected energy figure anywhere else in
  the field.
- **Joules.** A device energy ledger, and the arithmetic that says what a joules headline is
  *bounded* by. See below: the field's newest efficiency claim reproduces here in eight lines,
  and so do the three things it does not contain.
- **Structured cliques at the frontier, on both fabrics.** Zephyr and Pegasus both at exactly
  `busclique`'s size and chain length, checked against `busclique` run as an oracle, with the
  whole-qubit family's ceiling *proved* rather than assumed alongside.
- **Learning theory as executable oracles** — Hopfield, Gardner on both coupling spaces, dense
  associative memory with attention as its update, equilibrium propagation.
- **Zero dependencies in the core**, and the only stack in the field that runs in a browser tab.

### The newest vendor claim, decomposed

Extropic published **Z1T** on 2026-09-04 — transformer-like models for the Z1 sparse probabilistic
chip — headlined "up to 140x energy efficiency gains over GPUs". The post is unusually complete, so
the whole thing reproduces from its own numbers (`cargo run --example z1t_ledger`, `src/hybrid.rs`):

| | iso-parameter (as published) | iso-loss (their own ~10x FLOP penalty) |
|---|---|---|
| **H100 @ 10% MFU** (the headline's baseline) | **138.9x** | 13.89x |
| **H100 @ 50% MFU** | 27.8x | 2.78x |
| **H100 @ 100% MFU** | 13.9x | 1.39x |

Three things the multiplier does not contain, each derived rather than asserted:

1. **The sampler is 3.0% of the bill.** 8.74 nJ/token on Z1, 285.78 nJ/token on the FPGA carrying
   the rest. Setting the *entire thermodynamic contribution to exactly zero joules* moves 138.9x to
   143.1x. `Split::ceiling` is that bound, and it is a hard one: no improvement to any sampler, ever,
   takes a hybrid past what its unaccelerated half already costs.
2. **The stated path to 1000x is a claim about the FPGA.** `Split::host_speedup_for(1000.0)` says the
   non-thermodynamic half must improve **8.9x** — 7.0x even with Z1 running free. The post says this
   itself; the headline does not.
3. **Iso-parameter is not iso-quality.** The H100 runs "the same next-token step run densely"; the
   same post says the sparse model needs "about an order of magnitude more FLOPs… to achieve the
   same loss as a GPT-2 model". Both cannot be inside one multiplier.

And the provenance: the H100 term is *measured* (2026-08-12, batch-1 decode); both halves of the
numerator are device models. Z1 is at tapeout. Its sampling price also moved — `1.3e-14 J/sample`
here against `7.09e-15` in the Thermalizers appendix six weeks earlier, **1.83x apart for one
quantity**. `ledger::Z1_SPICE` and `hybrid::Z1T_PUBLISHED` carry both and assert neither.

None of this is an argument against the substrate — the diagnosis in (2) is correct and is the
strongest thing in the release. It is what a reference implementation is *for*: the claim arrives as
a number, and leaves as arithmetic anyone can re-run. `scripts/check-landscape.sh` now watches
`extropic-ai/sparse-transformers` on the same falsifiable claim as the others, and today it holds —
the open training recipe is JAX that runs on GPUs, with **no device code in the tree**.

---

## 5. Where the claim is weakest

Honesty about a reference includes where it is thin:

- ~~Krauth–Mézard's 0.833 is cited, not derived~~ — **closed, twice over.** `capacity_replica`
  solves the replica-symmetric saddle and bisects its zero for `α_c = 0.8331`;
  `capacity_by_enumeration` reaches `0.8305` from exhaustive counting with no replica assumptions.
  Both agree with the published `0.833` and with each other. The term that decided the derivation
  is the Legendre pairing `−½q̂(1−q)`, not `−½qq̂`.
- ~~**Pegasus is 8 chains short** of `busclique`~~ — **CLOSED.** `pegasus_clique_fragment` reaches
  **`K_{12(m−1)}` = `K_180` on `P_16` at chain 17**: busclique's size *and* busclique's chain length,
  at every size from `P_3` to `P_16`. Checked against `minorminer.busclique` run as an oracle on the
  same graph — `K_24/36/48/60/72/84/96` at `P_3..P_9`, `K_180` at `P_16` — and verified against the
  shipped fabric by `Embedding::verify`. **No structured clique gap remains on either fabric.**
  Two things it took, both of which correct something written here earlier. **The arms are not
  anchored**: decoding busclique's own `P_4` answer shows chains whose corner sits *mid-arm* (one is
  a full vertical wire joined to a single horizontal qubit), so each arm here grows only as far as
  some pair forces it. And the **row-column assignment** is the plain diagonal in the interior with
  three fixed blocks at the boundary, which is where the offsets bite. Anchored families top out at
  `K_176` — and the best anchored orientation is worth two chains over the worst, because
  `PEG_V = [1,1,5,5,3,3]` and `PEG_H = [3,3,1,1,5,5]` are not mirror images.
  The grid was measured, not assumed: each qubit is six fragments, every existing line spans exactly
  `6(m−1)` consecutive positions from its own offset, and `L²` of the cells carry a full `K_{2,2}`
  with **none partial** — 324 of 576 at `P_4`, 900 of 1296 at `P_6`, read off the shipped fabric.
  `P_3` carries its own assignment (the boundary blocks need 16 columns and `L` is 12 there), found
  by the same search and **not** busclique's: its columns differ while size and chain agree.
- **The whole-qubit construction's ceiling stands, and is still worth having.** `K_{12(m−2)+4}` is
  optimal for chains made of one vertical and one horizontal wire SEGMENT, proved rather than
  observed, with the four universal wires a theorem about the offset lists. It stays as
  `ft_clique_embed`'s fallback: a smaller clique with a machine-checked ceiling beats none.
- ~~**No silicon measurement.** Every joule figure is a device-model price~~ — **CLOSED.**
  `ledger::KV260_MEASURED` is a wattmeter reading. A 1,024 p-bit fabric emitted by `hdl`, implemented
  in Vivado for `xck26`, flashed to a Kria KV260 and metered on the SOM's own INA260: **0.5554 W
  above an idle PL at 23.1 sigma**, over 51.2 flips/ns, giving **10.85 +- 0.47 pJ per node update**.
  Against Extropic's projected `7.09 fJ` that is **~1,530x**.

  **CORRECTED 2026-09-18 — that sentence used to end "and it is the first entry in that comparison
  that is not itself a projection." A global literature sweep retired it.** Measured figures exist,
  one of them well below ours, and ours was an increment being held against other people's totals:

  | machine | J per update | what the number is | grade |
  |---|---|---|---|
  | Extropic Z1, arXiv:2608.01615 Table IV | 7.09 fJ | SPICE, uncharacterised silicon | simulated |
  | 28 nm four-chip Pegasus ASIC, arXiv:2609.07907 | **1.2 pJ**, incl. I/O | fabricated; the authors' own stated figure | measured |
  | **ferrotherm, KV260 fabric increment** | **10.85 pJ** | above an idle PL, INA260, 23.1 sigma | **metered** |
  | DSIM-1, six FPGAs, arXiv:2606.25313 | 59–71 pJ | **our quotient**: 150–180 W wall / 2.53e12 flips/s | measured, derived here |
  | **ferrotherm, KV260 whole board** | **72.1 pJ** | 3.6917 W / 5.12e10 flips/s | **metered** |
  | DSIM-2, 18x VP1902 | 0.47–1.6 nJ | **our quotient**: 1.4–1.6 kW / 1e12–3e12 flips/s | measured, derived here |

  The DSIM paper states wall power and flip rates and never a joule per flip, so those two rows are
  arithmetic done here, and it gives one power range per platform without saying which clock it
  belongs to — pairing it with the PEAK flip rate is the assumption that flatters them, which is
  the rule this project applies to a competitor's number. Four things follow. **Like for like, a
  six-FPGA machine from June matches our single board** (59–71 against 72 pJ): the 10.85 is what
  the fabric adds, not what the wall supplies. **The FPGA-to-ASIC gap for this job is now measured,
  about 9x**, rather than a rule of thumb. **The best measured update energy located anywhere is
  still ~169x above the projected one** — that gap, not any vendor multiplier, is the state of the
  field. And what remains particular to this entry is a stated metering protocol and an open path
  from a library call to the bitstream that was metered; their correctness checks are a GPU
  baseline and one certified Max-Cut optimum, where this stack holds a sampler against exact
  distributional oracles. `ledger::PEGASUS_28NM_ASIC` carries the ASIC's figure with its grade, and
  `the_best_measured_update_sits_between_the_projection_and_our_fabric` pins the ratios.

  **And the vendor tool was caught under-predicting.** Vivado's own estimator put the PL at
  **0.100 W**; the board drew **0.5554 W** — **5.6x** low. Without a switching-activity file it
  assumes a toggle rate near 12.5%, while this fabric IS a randomness engine. Every projected p-bit
  energy in this field is a model of that kind.

  Three checks separate this from a plausible number: the sensor was validated against a CPU load
  (+0.778 W on a 0.087 W noise floor) before it was trusted; the flash was verified in `dmesg`
  rather than by `state`, because writing to `fpga_manager/firmware` **silently no-ops** and leaves
  the old bitstream running — an earlier attempt here loaded nothing and honestly reported
  `-0.006 W`; and the design carries `DONT_TOUCH` with a registered fold, because with no observable
  sink the synthesiser deletes the whole fabric and the board measures a correct zero for something
  that is not there.
- ~~General nonlinear continuous units are verified only to ~3 dimensions~~ — **closed, twice.**
  `chain_log_z` is a transfer-operator oracle, `O(n · grid²)`, exact to the grid at any chain
  length. ~~Non-chain topologies past three units remain quadrature-bound~~ — `eliminate_log_z`
  is discretised variable elimination and subsumes both: it agrees with the transfer operator on a
  chain and with quadrature on a **triangle**, each to `1e-9`, and a twelve-unit binary tree matches
  the Gaussian closed form to `1e-3`. **The limit was misstated as a bound on `n`; it is the induced
  width.** A hundred-unit tree is exact and cheap (width 1); a complete graph on six is refused,
  with `64^6` in the error. Every tractable non-chain model in between was hidden by the old phrasing.
- ~~`Ferrotherm` is not in the Julia General registry, a channel we list that a user would find
  missing~~ — **this entry was wrong, and the error is mine.** It is true that `] add Ferrotherm`
  does not work, but that is a documented decision, not an oversight: `PACKAGING.md` analyses it and
  declines General because registering there means submitting to their AutoMerge — the same posture
  already declined for Yggdrasil — and because a Julia registry must be the root of its own
  repository. The README never claims otherwise; its install line is `cargo add ferrotherm`. I
  inferred a defect from a registry's absence without reading the document that explains it. The
  real open choice, stated there, is between hosting an IPAI registry and leaving install-from-URL
  as it is.

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
thermodynamic computing. 19 external sources of truth, 441 verification-bearing tests, seven
machine-checked theorems, and the field's only calibrated error bars. 19 external sources, 441
verification-bearing tests of 2000 — and `scripts/check-counts.sh` derives all four figures from the
suite rather than trusting this sentence, because three earlier versions of it disagreed with each
other. The competitors do verify —
against enumeration and self-consistency at brute-forceable sizes, at `1e-1` and `2e-2` tolerances
(see the correction above). The difference is the class of oracle and the coverage, not its
presence.

**And one of those four claims is narrower than it reads, which we found by testing it rather
than by being told.** "Calibrated error bars" is established two ways: `certify` checks interval
coverage on draws taken directly from the exact Boltzmann distribution by inverse-CDF, which
certifies the interval machinery and no chain at all; and `calibration` checks a real chain's bars
on a 12-spin ring with a field, which mixes freely. Both are sound. Neither says anything about a
chain that has not converged — and `examples/burnin` shows what that costs: an 8x8 ferromagnet
below its critical temperature, started all-up, reports `<m> = +0.974 +- 0.0009` against a truth of
exactly zero, **1060 standard errors out**, with an integrated autocorrelation time of 1.13 — near
the ideal value of one. The diagnostic returns its best possible score.

A bar is calibrated conditional on convergence, and `tau_int` cannot check that condition: it
summarises the timescales a trace contains, and a barrier never crossed contributes none. So
`floors::cost_per_effective_sample` now takes a `Convergence` argument and refuses rather than
pricing a run whose convergence was never established — by coupling from the past, which returns
the burn-in actually required, or by R-hat over dispersed chains. Both instruments were already in
the crate and neither was wired to the pricing.

Adoption will follow or it will not. It is not evidence either way.
