# Where ferrotherm stands

**Measured 2026-09-04.** Every number here came from a registry or repository API on that date and
is reproducible by re-running the commands in the last section. Where a number is an estimate or a
vendor claim rather than a measurement, it says so. Opinions are labelled as opinions.

This is a scorecard, not a pitch. The uncomfortable numbers are in the first table.

---

## 1. The uncomfortable part: reach

| project | language | GitHub stars | package downloads | first release |
|---|---|---|---|---|
| **THRML** (Extropic) | Python / JAX | **1,151** | 581/month (PyPI) | 2026 |
| **dimod** (D-Wave) | Python / C++ | 142 | **175,051/month** (PyPI) | 2017 |
| **OpenJij** | C++ / Python | 133 | — | 2018 |
| **thermox** (Normal) | Python / JAX | 69 | — | 2024 |
| **torx** (Extropic) | Python | 68 | — | 2026 |
| **ommx** (Jij) | **Rust** / Python | 53 | 7,895/month (PyPI), 47k total (crates.io) | 2024 |
| **quantrs2-anneal** | **Rust** | 16 | 3,882 total (crates.io) | 2025 |
| **ferrotherm** | **Rust** | **1** | 1,266 total (crates.io), 1 month old | 2026-08-05 |

**Read that honestly.** By adoption we are last. THRML has a thousand times our stars. dimod has a
hundred and thirty times our downloads, monthly, and has had eight years to get them. Nothing below
changes this row, and no amount of technical depth substitutes for it.

Two things make it less bleak than it looks. Our crate is **one month old** and its 1,266 downloads
are all from that month, against THRML's 581/month at 1,151 stars — the interest per unit of
existence is not the problem. And the *reach* number that matters for a library nobody has heard of
is not stars but whether the install works in one line, which §3 scores.

---

## 2. The Rust lane: essentially uncontested, and that is checkable

The claim "own the Rust lane" needs testing, not asserting. Searching crates.io for the field's
vocabulary returns:

| crate | what it is | last release | recent downloads | verdict |
|---|---|---|---|---|
| **ferrotherm** | this | 2026-09-04 | 1,266 | the only hit for `thermodynamic-computing` |
| `th-rust` | "framework for thermodynamic and probabilistic computing" | **2023-04-28** | 9 | **abandoned** — one release, 3½ years cold |
| `qmc` | quantum Monte Carlo (SSE) | 2026-01-07 | 311 | active, **GPL-3.0**, different workload |
| `quantrs2-anneal` | annealing inside a quantum framework | 2026-08-30 | 207 | **live competitor**, Apache-2.0 |
| `hercules` | QUBO heuristics toolbox | 2025-09-09 | 24 | narrow, slow |
| `annealers` | bindings to vendor annealers | 2023-06-03 | 82 | abandoned |
| `ernst` | 2D spin-glass simulation | 2024-03-23 | 9 | abandoned, GPL-3.0 |
| `ommx` | **interchange format**, not a sampler | 2026-09-02 | 7,895/mo | **thriving, complementary** |

The nearest-named competitor is dead. The only live sampling rival, `quantrs2-anneal`, is a
component of a quantum-computing framework rather than a thermodynamic stack, and is two orders of
magnitude smaller than the Python incumbents. `qmc` has 50× our downloads but is GPL-3.0, which
excludes it from commercial use that Apache-2.0 does not.

**The genuinely important row is `ommx`.** Jij is building the field's interchange layer *in Rust*,
actively, with real adoption. That is the one place a Rust-lane claim could be contested — and it is
complementary, not competing: OMMX describes a *problem*, we describe a *program*. We already adopt
it rather than inventing a rival, which remains the right call.

**Verdict: the lane is open, and the reason is not that we are good — it is that nobody else showed
up.** That is a fact with an expiry date.

---

## 3. Ease of use, measured as install friction

The only honest way to score "ease of use" without focus-grouping it is to count what a new user must
do before their first sample.

| stack | one-line install | runtime deps | works in a browser | notebook-first docs |
|---|---|---|---|---|
| THRML | `pip install thrml` | **JAX** (+ CUDA for GPU) | no | **yes** |
| thermox | `pip install thermox` | JAX | no | yes |
| Ocean / dimod | `pip install dwave-ocean-sdk` | large SDK | no | yes |
| OpenJij | `pip install openjij` | C++ toolchain for source builds | no | yes |
| Amplify | `pip install amplify` | proprietary licence | no | yes |
| **ferrotherm** | `cargo add ferrotherm` / `pip install ferrotherm` | **zero in the core** | **yes, WebGPU** | **no** |

**Where we win:** the core has **zero dependencies**. Not "few" — zero. THRML cannot run without
JAX, and JAX with GPU is the single largest install-friction item in this field. Our Python wheel
has one dependency. And we are the only stack in the table that runs in a browser tab with no
install at all.

**Where we lose, and it is the same weakness twice:** we have **no notebook-first documentation**.
Every incumbent teaches through a tutorial notebook; we teach through 43 examples that must be
compiled and 67 module docs that must be read. For a researcher evaluating three libraries in an
afternoon, that is the difference between being tried and being skipped. This is the highest-value
gap in this document and it is not a technical one.

---

## 4. Deployment

| stack | channels |
|---|---|
| THRML / thermox / Ocean / OpenJij | PyPI (Ocean also conda) |
| ommx | crates.io + PyPI |
| **ferrotherm** | crates.io ×6, PyPI, Julia JLL, C header, Zig, wasm/browser, HTTP service, MCP |

Eight surfaces against one is the widest deployment surface in the field, and it is not close. The
C ABI is 211 symbols with a parity gate that fails CI if any surface falls behind.

**One measured gap:** `Ferrotherm` is **not in the Julia General registry**. `] add Ferrotherm`
does not work; the JLL is self-hosted on GitHub releases. That is a one-line claim of "Julia
support" that a Julia user would find false on first contact, and it should be registered or the
claim softened.

---

## 5. Where we are actually ahead

These are the roadmap's differentiators, and unlike the rows above they are not contested by anyone:

- **Certified sampling.** No commercial machine — US, Japanese or Chinese — exposes calibrated
  finite-temperature sampling with a stated distribution error. They return "best found". We return
  a certificate with the achieved β, effective sample size and noise floor.
- **Error bars that were checked.** `calibration::calibrate` tests whether a reported bar is honest
  by z-scoring against exact answers. It found one of our own bars 30% too small, corrected a
  published verdict, and forced a missing bar into existence. No other stack in this field tests its
  own uncertainty at all.
- **Free energy, four ways, with the guarantee each carries** — including an *unconditional* Markov
  lower bound and a deterministic Gibbs–Bogoliubov one.
- **Formal proofs in CI.** 7 Kani theorems, exhaustive over their ranges, as the 14th gate.
- **Joules.** A device energy ledger. `amplify-benchmark`'s four metrics have no energy axis.
- **Structured cliques at the frontier**: Zephyr at exactly busclique's size and chain length,
  Pegasus within 5% with its ceiling *proved*.
- **Breadth**: 67 modules, 45.7k lines, 911 tests, and the learning-theory oracles (Hopfield,
  Gardner both couplings, dense associative memory, equilibrium propagation) that no other
  thermodynamic stack carries.

---

## 6. What to do about it, in order

1. **A notebook-first quickstart.** The single highest-leverage item in this document. Browser-based,
   no install, first sample in under a minute — we are uniquely able to do this because we run in a
   tab and nobody else does.
2. **Register `Ferrotherm` in the Julia General registry**, or stop listing Julia as a channel.
3. **Publish one benchmark others can cite**, reporting joules per independent sample — the axis
   nobody reports, on the workload (`THRML` notebook 02's Pegasus benchmark) they chose themselves.
4. **A `dimod.Sampler` shim.** dimod is 175k downloads a month. Conforming to its interface is how
   OpenJij — a competitor — got reach, and it costs us a small adapter.
5. Keep the technical lead compounding, but understand that items 1–4 are worth more per hour than
   any further depth right now.

---

## 7. Reproducing this

```bash
# reach and activity
curl -s https://api.github.com/repos/extropic-ai/thrml | jq '{stars:.stargazers_count, pushed:.pushed_at}'
curl -s https://pypistats.org/api/packages/thrml/recent
curl -s -A "you@example.com" 'https://crates.io/api/v1/crates?q=qubo&sort=downloads'
curl -s -A "you@example.com" https://crates.io/api/v1/crates/th-rust | jq '.crate.updated_at'

# our own profile
grep -c '^pub mod' src/lib.rs && cat src/*.rs | wc -l
bash scripts/check-parity.sh && bash scripts/check-proofs.sh
```

`scripts/check-landscape.sh` re-checks the falsifiable claims of the 2026-08-05 survey this builds
on, so the map can say when it has gone stale instead of being trusted indefinitely.
