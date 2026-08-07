# The `.ftp` program format, version 1

**Status:** stable. **Date:** 2026-08-06.
**Editor:** Institute for Physical AI @ BMI. **Licence:** Apache-2.0.

This document is normative and standalone. It is written so that a conforming implementation can be
produced **without reading any ferrotherm source code**, and the reference implementation is bound by
this document rather than the reverse.

The key words MUST, MUST NOT, SHOULD, SHOULD NOT and MAY are to be interpreted as in RFC 2119.

---

## 1. Scope and rationale

`.ftp` describes a **program** for a thermodynamic or Ising sampling fabric: the model, which
variables update together, at what temperatures, with which penalties ramping, what to observe, and
what an operation costs.

It exists because `(J, h, coloring, schedule)` is the de facto interchange of this entire field and
has never been specified. It circulates as MATLAB `.mat` files beside a `colorMap.csv`, and every
vendor accepts a different upload shape for the same matrix.

`.ftp` is deliberately **not** an instance format. An instance format describes a *problem*:
variables, an objective, constraints. Formats for that already exist and are adequate — implementers
wanting one SHOULD use OMMX or LP/MPS rather than this. What no existing format carries is how the
problem is to be *run*, which is the part a sampling fabric needs and the part this specifies.

### 1.1 Design constraints

- **Text, not binary.** The audience currently exchanges files that only the sending lab can read. A
  format that survives `grep`, `diff`, code review and an email is worth more here than saved bytes.
- **Lossless numerics.** A format that silently alters the last bit of a coupling alters the model.
- **No required dependency.** A conforming reader MUST be implementable with a language's standard
  library alone.

---

## 2. Lexical structure

A `.ftp` document is a sequence of Unicode characters encoded as UTF-8. Implementations MUST NOT
require a byte-order mark and MUST accept one if present, discarding it.

The document is line-oriented. Lines are separated by LF (`U+000A`); a conforming reader MUST also
accept CRLF and MUST treat the CR as part of the separator rather than as content.

Within a line:

- A `#` character begins a **comment**, which extends to the end of the line. A `#` has no special
  meaning inside no other construct, because no construct in this version contains a `#`.
- Leading and trailing whitespace is insignificant.
- Fields are separated by one or more space (`U+0020`) or tab (`U+0009`) characters.
- A line that is empty after comment removal and trimming is **ignored**.

A non-ignored line is a **directive**: a keyword followed by zero or more fields.

Comments and blank lines MUST NOT affect the meaning of a document. Two documents differing only in
comments and blank lines are equivalent, and their digests (§7) MUST be equal.

---

## 3. Directives

The `ftp` directive MUST be the first non-ignored line. All other directives MAY appear in any
order, except as constrained in §3.3.

| Directive | Fields | Cardinality |
|---|---|---|
| `ftp` | *version* | exactly 1, first |
| `name` | *token* | 0 or 1 |
| `spins` | *count* | exactly 1 |
| `bias` | *index* *value* | 0 or more |
| `factor` | *weight* *index*… | 0 or more |
| `color` | *class* *index*… | 0 or more |
| `encode` | *base* *k* *kind* | 0 or more |
| `stage` | *beta* *sweeps* *dw* *copy* | 0 or more, **ordered** |
| `observe` | *token* | 0 or more |
| `target` | *token* | 0 or 1 |
| `price` | *token* | 0 or 1 |

### 3.1 Definitions

**`ftp` *version*** — a non-negative integer. A reader encountering a version it does not implement
MUST reject the document and MUST report the version found. It MUST NOT attempt partial
interpretation.

**`spins` *count*** — the number of spins, a non-negative integer. Spin indices are **0-based** and
range over `[0, count)`.

**`bias` *i* *h*** — external field `h` on spin `i`. Repeated directives for the same spin
**accumulate**; the resulting field is their sum. A biases of zero MAY be omitted and a writer
SHOULD omit them.

**`factor` *w* *v*…** — an energy term contributing `-w · Π s_v`. Arity MUST be at least 1. A
variable MUST NOT appear more than once within one factor: `s·s = 1`, so a repeated variable silently
changes the factor's order, and a reader MUST reject such a factor rather than interpret it.

**`color` *c* *i*…** — the spins in colour class `c`, which may be updated simultaneously. Colour
indices are 0-based. A conforming reader MUST NOT assume the classes partition the spins, and MUST
NOT assume they are disjoint; see §5.2.

**`encode` *base* *k* *kind*** — provenance: the spins beginning at *base* hold one *k*-valued
variable, spelled *kind*, which MUST be one of `onehot`, `binary`, `domainwall`. *k* MUST be at
least 2. The width occupied is `k` for `onehot`, `ceil(log2 k)` for `binary`, and `k-1` for
`domainwall`. This directive is **descriptive**: it does not create constraints, and the penalty
terms enforcing the encoding MUST appear as ordinary `factor` and `bias` directives.

**`stage` *beta* *sweeps* *dw* *copy*** — one rung of the schedule: run *sweeps* sweeps at inverse
temperature *beta*, with the domain-wall penalty at strength *dw* and the copy-agreement penalty at
*copy*. Stages execute in the order they appear, and that order is significant.

**`observe` *token*** — a quantity to reduce during sampling. This version defines `energy` and
`magnetization`; other tokens are permitted and a reader that does not recognise one MUST ignore it
rather than reject the document.

**`target` *token*** — the intended backend or device topology. Advisory.

**`price` *token*** — the price table used to interpret the energy ledger. Advisory.

### 3.2 Numbers

*index*, *count*, *class*, *base*, *k* and *sweeps* are non-negative decimal integers.

*value*, *weight*, *beta*, *dw* and *copy* are IEEE 754 binary64 values written in decimal.

A writer MUST emit the shortest decimal representation that round-trips to the identical binary64
value. A reader MUST parse to the nearest binary64. Together these guarantee that a parse of a write
is bit-identical to the original and a write of a parse is byte-identical to the input.

`NaN` and the infinities MUST NOT appear. A reader encountering a non-finite value MUST reject the
document, because a non-finite coupling poisons every energy it touches.

### 3.3 Ordering

`spins` MUST precede any directive naming a spin index. This is the only ordering constraint besides
`ftp` being first, and it exists so a reader can validate indices in a single pass without buffering.

---

## 4. Semantics

The energy of a state `s ∈ {-1,+1}^n` is

```
E(s) = - Σ_factors w · Π_{v ∈ factor} s_v  - Σ_i h_i · s_i
```

A **positive** weight therefore prefers the product of its variables to be `+1`, and a
ferromagnetic bond is low energy when aligned. This sign convention is normative. Implementations
bridging to a format with the opposite convention MUST negate at the boundary and SHOULD test that
mapping against an independent exact solver, because a sign error produces entirely plausible output
that is wrong on every problem.

At inverse temperature `beta`, the target distribution is `p(s) ∝ exp(-beta · E(s))`.

An arity-2 factor `w i j` is identical in meaning to the coupling `J_ij = w`. A conforming
implementation MUST NOT treat them differently.

---

## 5. Conformance

### 5.1 Readers

A conforming reader MUST:

1. reject a document whose first non-ignored directive is not `ftp`;
2. reject an unimplemented version, reporting the version found;
3. reject a spin index outside `[0, spins)`;
4. reject a factor containing a repeated variable;
5. reject a non-finite number;
6. reject an unknown *directive keyword*;
7. accept an unknown `observe` token, ignoring it;
8. report the **line number** with every rejection.

Requirement 8 is not cosmetic. A format nobody can debug is a format nobody adopts.

### 5.2 The colouring is a hint, and MUST be verified

A `color` directive asserts that its members share no factor. A conforming reader that intends to
update a class simultaneously MUST **verify** this against the factors, and MUST NOT trust the
assertion.

An incorrect colouring does not produce an error; it produces a sampler that draws from the wrong
distribution while appearing to work. A reader MAY ignore the supplied colouring and compute its own.

### 5.3 Writers

A conforming writer MUST emit `ftp` first and `spins` before any spin reference, MUST use the
round-tripping numeric form of §3.2, and MUST NOT emit a construct it would itself reject.

### 5.4 Claiming conformance

An implementation claiming conformance MUST pass the test vectors of §8 and SHOULD state which
optional directives it interprets rather than merely tolerates.

---

## 6. Example

```
ftp 1
name frustrated-ring
spins 5
factor -1 0 1
factor -1 1 2
factor -1 2 3
factor -1 3 4
factor -1 4 0
stage 0.05 40 1 1
stage 4 40 1 1
observe energy
target cpu
price z1_spice
```

An odd antiferromagnetic ring cannot be two-coloured, so exactly one bond must remain unsatisfied
and the ground energy is `-3`. An implementation may use this as a first end-to-end check.

---

## 7. Digest

A digest identifies a program across runs and machines. It is computed as FNV-1a (64-bit) over the
document's **canonical serialisation** — the output a conforming writer produces for the parsed
program — and therefore ignores comments, blank lines and whitespace.

```
h = 0xcbf29ce484222325
for each byte b:  h = (h XOR b) * 0x100000001b3   (mod 2^64)
```

This is an identity function, not a security function, and MUST NOT be used as one.

---

## 8. Test vectors

A conforming implementation MUST produce these results.

| # | Input | Required outcome |
|---|---|---|
| 1 | `spins 4` alone | reject: `ftp` not first, line 1 |
| 2 | `ftp 99` | reject: unsupported version 99, line 1 |
| 3 | `ftp 1` then `bias 0 1` | reject: `spins` must precede a spin reference |
| 4 | `ftp 1`, `spins 4`, `factor 1 0 9` | reject: index 9 out of range, line 3 |
| 5 | `ftp 1`, `spins 4`, `factor 1 0 0` | reject: repeated variable, line 3 |
| 6 | `ftp 1`, `spins 4`, `wobble 3` | reject: unknown directive, line 3 |
| 7 | `ftp 1`, `spins 4`, `encode 0 3 trinary` | reject: unknown encoding, line 3 |
| 8 | `ftp 1`, `spins 4`, `observe entropy` | **accept**, ignoring the token |
| 9 | `ftp 1` alone | reject: missing `spins` |
| 10 | §6 example | accept; ground energy `-3` |
| 11 | `bias 2 0.1` twice | accumulates to `0.2` |
| 12 | couplings `0.1`, `1/3`, `1e-300`, `1e300`, `π` | round-trip **bit-identical** |
| 13 | §6 example with comments and blank lines inserted | same digest as §6 |

---

## 9. Versioning

The version integer increments when a document valid under version *n* would be misread by a reader
implementing version *n+1*, or vice versa. Adding a directive that older readers must reject is a
version increment; adding an `observe` token is not, because §5.1(7) already requires tolerance.

A future version MUST NOT redefine the meaning of an existing directive.

## 10. Reference implementation

`ferrotherm`, Apache-2.0, `src/ftp.rs`. Two independent implementations exist as of this writing:
the Rust reference and a JavaScript writer in the in-browser workbench, whose real output is
committed as a cross-implementation test fixture.

**Where this document and any implementation disagree, this document is correct.**
