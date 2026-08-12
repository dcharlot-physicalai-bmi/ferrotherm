# ferrotherm as the universal stack

Institute for Physical AI @ BMI. The development and implementation plan for making ferrotherm the
single open ecosystem that programs, verifies and prices every thermodynamic compute fabric.

Companion documents: [`ROADMAP.md`](ROADMAP.md) for the engineering phases already under way,
[`WORKLOADS.md`](WORKLOADS.md) for what runs on it.

---

## 1. The thesis

Thermodynamic computing has a hardware problem and a software problem, and only the second one is
ours to fix.

**The field is fragmented at every layer.** Six or more physically distinct fabrics — CMOS
stochastic units, fibre-loop optics, oscillator arrays, memristive crossbars, sMTJ p-bits, digital
annealer ASICs — each arriving with its own closed SDK, its own problem format, and its own
benchmark. Above them sit at least eight incompatible software stacks in four languages. Between
them: nothing. No shared IR, no shared program format, no shared notion of what a correct sample is.

**And there is no open device layer anywhere.** Every "open" repository in this field is a
simulator. Grep `thrml`, `torx`, `thermox`, `posteriors`, `kaiwu-pytorch-plugin`, `SANTA`,
`AOCoptimizer.jl` for `pcie|usb|/dev|ioctl|fpga|driver|firmware` and you get zero hits across all of
them. The only open stack that drives real sampling silicon belongs to D-Wave.

That is the shape of a missing middle, and it is the same shape Modular saw in AI compute: many
accelerators below, many frameworks above, an unmaintainable cross product between. The difference
is what we do with the position. **Modular owns the middle and closes it. We own the middle and open
it** — the Institute exists to build a commons, not to become a toll booth. The parallel is to the
architectural role, not to the business model.

### Why this is winnable now

- **Nobody has proved value, so nobody has lock-in.** Zero performance or energy claims in this
  field have been independently verified. Where outsiders got hardware access, the machine lost.
  There is no incumbent to displace, only a vacuum to fill.
- **The vendors are not going to do it.** A fabric company's incentive is to make its own silicon
  look good, which is precisely why none of them ship a neutral benchmark or an honest baseline. A
  neutral layer has to come from outside, and it has to be open or nobody will trust it.
- **We already run everywhere the users are.** A stack that requires exotic silicon reaches nobody.
  ferrotherm runs on a CPU, in a browser tab, on a GPU, and onto an FPGA today. Exotic fabric is an
  *acceleration*, never the entry ticket.

---

## 2. Architecture

Seven layers, all Rust, all ours.

```
    Rust  C  Python  Zig  Julia  Node graph  MCP  HTTP  wasm        surfaces
                              │                                      all nine reach the modelling
                              │                                      layer, not just the sampler
                    adapters ─┤ dimod · OMMX · QUBODrivers · MOI     (edge, deletable)
                              │
                     lowering passes                                 gates→factors, categorical→spins,
                              │                                      dense→sparse, higher-order→pairwise
                              ▼
           Model × Blocks × Kernels × Schedule × Observers            THE IR  (.ftp)
                              │
                              ▼
                Sampler trait ── Certificate ── Ledger                proof + price, always
                              │
                              ▼
                      Device trait                                    ← the layer nobody has
                              │
   CPU · WebGPU · FPGA · D-Wave · Fujitsu · Hitachi · Toshiba · QBoson · future fabric
```

**Declared means the fabric's limits are stated in code and a program is checked against them
before it is submitted** — topology, degree, arity, coefficient range, precision, and whether
variables place one-to-one or need minor embedding. It does not mean we drive the silicon; today
only the Hitachi part and our own FPGA are driven end to end. Declaring is most of the value on its
own, because a caller can ask what rules their program out before buying time on a machine.

QBoson is declared **partially**, which is a state this now has a way to express. Its Kaiwu SDK
documentation gives the number that matters — *"the CIM machine only supports 8-bit INT space
[-128, 127]"*, by far the hardest coefficient limit here — and does not give the machine's size or
its connectivity. Those are listed in `unstated`, and `Fabric::verdict` therefore refuses to promise
a run however cleanly a program checks.

An earlier version of this file said no vendor material gave the precision. That was wrong: it is in
the SDK documentation rather than a datasheet, which is not where this review had looked. Worth
recording, because it is exactly why an absence is written as *"this review did not locate X"*
rather than *"X does not exist"* — the first survives being wrong.

Three of these are strategic assets rather than engineering conveniences, and they are what make
this an ecosystem rather than a library.

| Asset | Why it is the lever |
|---|---|
| **`.ftp`, the program format** | `(J, h, coloring, schedule)` is the field's de facto interchange and **nobody specified it**. It lives as scattered `.mat` files. Being the format is durable in a way that conforming to someone else's never is. |
| **The Certificate** | Nothing anywhere reports the temperature a sampler achieved, its effective sample size, or a noise floor. Make it the definition of a correct sample and it becomes the **conformance test** — the way a standard wins without asking permission. |
| **The Device trait** | The only open device code in the field is D-Wave's. Owning an open, multi-vendor device layer is the moat, and it is the one thing a vendor cannot casually replicate without opening their own stack. |

---

## 3. The fabric support matrix

"Universal" has to mean something checkable. It means: every fabric a person can actually reach,
reachable through one program.

| Fabric | Access route | Status | Phase |
|---|---|---|---|
| **CPU** | native | ✅ shipping | done |
| **WebGPU** | browser, no install | ✅ correct, crossover measured | done |
| **FPGA — Alchitry Pt V2** | our own JTAG + bitstream stack | ◐ fabric emitted, sequential pending | E1 |
| **Hitachi CMOS annealer** | free public web API | ✅ **running on the real ASIC** | E2 |
| **D-Wave** | Ocean / Leap | **declared** — Advantage, Advantage2 | E2 |
| **Fujitsu Digital Annealer** | cloud API | **declared** — DA3 | E3 |
| **Toshiba SQBM+** | AWS Marketplace | **declared** — QUBO solver | E3 |
| **QBoson** | Kaiwu SDK | **declared** — CPQC, precision only | — |
| **QBoson CPQC** | cloud | planned | E3 |
| **Extropic / Normal silicon** | no external access exists | trait ready, blocked on them | — |
| **Oscillator / memristive** | academic only, no company | trait ready | — |

Two things this table says that matter.

**Hitachi's annealer is a fabricated Ising ASIC callable from a free public API, and essentially
nobody has ever used it** — two papers in all of OpenAlex mention it. That is the cheapest real
silicon in the world to support and the easiest first win.

**The bottom two rows are not failures, they are the point.** We cannot reach Extropic or Normal
silicon because *nobody outside those companies can*. A stack that already speaks to eight fabrics
is the natural thing to plug a ninth into, and the trait is ready when they open up.

---

## 4. Phases

Continuing the numbering in `ROADMAP.md`, which carries Phases 0–5. These are the ecosystem phases.

### E0 — Make the middle real *(prerequisite, mostly done)*

The IR, the certificate and the oracles all exist. What remains before a device layer makes sense:

- **`Sampler` trait as the single execution seam.** Every backend implements it; nothing else
  executes. `Accept:` the CPU, WebGPU and FPGA paths all reach a graph only through this trait.
- **`.ftp` v1 frozen and specified** as a standalone document, not only a module doc.
  `Accept:` a third implementation, written from the spec alone by someone who has not read our
  code, round-trips our test corpus. Two implementations already agree; a third from the spec is
  what makes it a format.

### E1 — The device layer *(the moat)*

- **`Device` trait**: `program(&Ftp) -> Handle`, `run(Handle, Schedule) -> Samples`, `readback`,
  `capabilities() -> Fabric`. Capabilities are *declared and checked*, not assumed: topology, degree
  limit, coupling precision in bits, whether it can hold a field, reprogramming cost.
- **Precision is first-class.** QBoson's int8 coupling limit is the binding constraint on that whole
  platform and appears nowhere in its documentation — a third party had to discover it. Every device
  in our matrix declares its precision, and the compiler refuses or requantises rather than silently
  degrading.
- **Complete the Pt V2**: flip-flops, clock, a sampler running and read back.
  `Accept:` one `.ftp` runs on CPU, browser and FPGA and the three agree within the certificate's
  own noise floor. Published with the bitstream. **This is the first open library-to-silicon path in
  the field.**

### E2 — First external fabrics

- **Hitachi** via the free public API, then **D-Wave** via Ocean.
- **The embedding layer** becomes load-bearing here: COPY-gate sparsification, DSATUR colouring, and
  2D adaptive parallel tempering over (β, W₀) so nobody hand-tunes copy strength — the field's
  admitted open wound, whose only implementation is one unadopted MATLAB file.
- `Accept:` the *same* `.ftp` produces certified samples on three physically different machines, and
  the certificate catches it when one of them is wrong.

### E3 — The conformance suite

This is the phase that converts a library into a standard.

- **`ferrotherm-conform`**: a published suite any fabric can run — planted instances with known
  optima, exact-enumerable models, Onsager, the SK transition, and the certificate's own checks.
- It reports **sampling fidelity**, which nothing in this field currently reports at all: TV distance
  to a known Boltzmann target with its noise floor, effective sample size per joule, integrated
  autocorrelation time, and bias from device non-idealities.
- **Run it on ourselves first and publish the results including the bad ones.** A conformance suite
  authored by someone who exempts themselves is worthless.
- `Accept:` one fabric we do not own has run it and we have published the result unedited.

### E4 — The commons

- **Package everywhere**: crates.io ✅, PyPI, Julia General, and a Zig package. The bindings
  themselves are complete — every surface states problems, not just spins — so what remains is
  distribution rather than capability.
- **Governance that survives us**: a specification repository separate from the implementation, with
  a documented process for adding a fabric. A format controlled by one implementation is a library
  with pretensions.
- **The fabric registry**: an open, cited database of every fabric, its real capabilities, its
  declared precision, and its independently verified results — with vendor claims and verified
  results in *separate columns*.

---

## 5. What we do not do

**We do not benchmark ourselves into a corner.** Every headline multiplier in this sector — 10,000×,
1000×, 350×, 100× — is a projection, a self-comparison, or a comparison against an untuned baseline.
The honest measured p-bit advantage over a tuned GPU sampler is 5–18×. We publish against **tuned**
baselines at **equal solution quality** or we do not publish.

**We do not chase routing, scheduling or portfolio optimisation.** MILP in a QUBO costume; they lose
to Gurobi.

**We do not close anything.** No closed binary wheel with a plug hole where the sampler goes. If a
fabric vendor requires an NDA to talk to their silicon, we write the trait and wait.

**We do not claim universality we have not demonstrated.** The matrix in §3 is the claim, it is
checkable, and rows move only when a certified sample comes back.

---

## 6. Risks, honestly

| Risk | Reading |
|---|---|
| **The fabrics never become worth using** | Real. Fujitsu's independently audited verdict is "competitive with 1999–2004 heuristics". If that is the ceiling, the *sampling* substrate still matters for EBM training and control, which is why `WORKLOADS.md` leads with those rather than with optimisation. |
| **A vendor bundles a competing layer** | They will, and it will be closed and single-fabric. That is the argument for us, not against us. |
| **Jij gets there first** | The closest architectural comparator: Rust-first, Apache-2.0, funded, expanding. They have modelling and interchange; they do not have a device layer, a certificate, or a browser. Watch closely and interoperate rather than duplicate — OMMX is theirs and we read it. |
| **We spread too thin** | The real one. Nine fabrics is a lot of surface for a small team. Mitigation: the `Device` trait means a fabric is one implementation, not a fork; and a fabric with no user gets frozen, not maintained. |

---

## 7. Progress

| # | Step | State |
|---|---|---|
| 1 | Freeze `.ftp` v1 as a standalone spec | ✅ `spec/ftp-v1.md`, 13 test vectors, 14 conformance tests binding the implementation |
| 2 | `Device` trait with declared capabilities and precision | ✅ `src/fabric.rs`; FPGA ported onto it |
| 3 | Finish the Pt V2 | ◐ limits now **declared and checkable without a board**; sequential fabric blocked on hardware access |
| 4 | Hitachi over its free public API | ✅ **`ferrotherm-cloud`, verified on the ASIC** — see below |
| 5 | `ferrotherm-conform`, run on ourselves first | ✅ `src/conform.rs`, 7 cases, **7/7 on our own backend** |

*Porting the FPGA onto the trait immediately earned its keep.* The Pt V2 cell counts active
neighbours rather than weighting them and spends one of its six LUT inputs on the random bit, so it
supports **five** neighbours and **unweighted** couplings — a spin glass cannot be expressed on it at
all. That is a property of the cell rather than the placement, no LUT budget changes it, and it was
undeclared until the trait made declaring it mandatory. Same class of defect as QBoson's
undocumented int8; we found ours on our own hardware first.

### The first external fabric, running

`ferrotherm-cloud` drives Hitachi's CMOS annealing ASIC — real fabricated Ising silicon that two
papers in all of OpenAlex mention — through the same `Device` trait as our CPU:

```
fabric        hitachi-cmos-asic | king-graph | 147456 sites | degree 8 | coupling 4 bits
program       1156 spins declared, 24 couplings, digest 3a5b2d8652af72bf
machine energy (their sign) [-24.0]
execution     275.511 ms on the ASIC
our energy    -24
checkerboard  yes - every bond satisfied
```

Two conventions were **measured rather than read**, on the first call, because a mistake in either
produces plausible output that is wrong on every problem:

- **The sign is inverted.** Their energy is `Σ pᵢⱼ sᵢsⱼ` *minimised*, so a positive coefficient is
  antiferromagnetic. Four positive couplings on a 2×2 block came back as a checkerboard at −24.
  That the two energies now agree exactly is the proof the mapping is right; a wrong sign gives +24.
- **The topology is a King's graph** — neighbours are orthogonal *and diagonal*, and coupling
  non-adjacent coordinates is an error rather than an ignored term.

And the ASIC stores coefficients in **four bits**, `-7 ≤ p ≤ 7`. That is the binding constraint on
the machine, it is declared in the `Fabric`, and a program exceeding it is refused before submission
rather than silently quantised. A model that does not fit the grid is refused too, with the reason:
*a driver that placed it for you would be choosing an embedding you did not see.*

**Our own conformance result, unedited:**

```
PASS ferromagnet          ground energy -12, exact -12
PASS frustration          ground energy -3, exact -3 (one bond must break)
PASS planted optimum      2.08% above a planted optimum of -192
PASS exact agreement      -59 against variable elimination's exact -59
PASS determinism          same seed reproduces: true
PASS rejects a bad run    caught it: 300 draws are worth about 3 independent samples
PASS sampling fidelity    beta_eff 0.4978 (asked 0.5), ess 2734,
                          tv 0.1523 against a 0.3060 noise floor
7/7 cases
```

The sixth case is the one that makes the suite mean anything. A fabric that always returns the same
low-energy state passes every "did it find the optimum" test ever written, so the suite asks for a
deliberately bad run and **fails the fabric if the certificate blesses it**. A test proves this
works by handing the suite a device that answers everything with all-spins-up: it passes the
ferromagnet, because all-up genuinely *is* that optimum, and is caught by frustration and the
planted instance.

---

## 8. The first five things

1. **Freeze `.ftp` v1 as a standalone spec** and get a third implementation written from it.
2. **Land the `Device` trait** with declared capabilities and precision, and port the FPGA path onto
   it.
3. **Finish the Pt V2** — flip-flops, clock, readback — and publish the first open
   library-to-silicon path in this field, bitstream included.
4. **Hitachi over its free public API**, because it is real fabricated silicon nobody has used.
5. **Publish `ferrotherm-conform` and run it on ourselves first**, results unedited.

The order is deliberate. Each step makes the next cheaper, and the first three are entirely within
our control — no vendor, no NDA, no permission.
