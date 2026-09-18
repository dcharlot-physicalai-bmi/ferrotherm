# Absorbing the dynamical-system programme

**Status:** five modules shipped and under test — `kuramoto`, `nonrev`, `dsisa`, `precision`,
`phasegen`. Every figure below is printed by `cargo run --release --example absorption`; nothing
here is transcribed by hand. If the document and the example disagree, the document is wrong.

---

## 0. The thesis, stated so it can be attacked

Three claims underwrite this work, and only the third is contentious.

1. The dominant compute fabric is the wrong hardware, inefficient by design rather than by
   accident. It was built for a different application space and was never asked to be efficient at
   this one.
2. Efficiency is now the binding constraint rather than a refinement.
3. The bio-inspired camp is wrong that biology is efficient. The better path is engineered on
   purpose from what is actually efficient, and the best solution will look like nothing the living
   world has produced.

The third claim is usually argued by assertion. It can be argued by arithmetic instead. Landauer's
floor at 300 K is 2.87 zJ per erased bit. A synaptic event costs on the order of 10⁷ of those. A
Z1-class Gibbs cycle, on the most favourable published pre-silicon estimate, costs about 10⁶. The
KV260 p-bit fabric this project metered costs 3.8 × 10⁹. So biology is not efficient — it is seven
orders above the floor — and neither is anything anyone has built. **Every engineered stochastic
substrate on the table today is worse than the brain per primitive event, on paper, including
ours.** That is the honest form of claim 3, and it is much stronger than the rhetorical form,
because it says where the unoccupied ground is: the 10⁰–10⁵ kT band, which is the reversible and
adiabatic regime, not the noisy one. Neither p-bits nor oscillators live there. The Periodic Stack
and the reversible reconfigurable array do.

Against that background, the coupled-oscillator programme is a competitor for the *middle* of the
range, and the question this document answers is what thermodynamic computing is missing in order to
contain it entirely.

The answer turned out to be five things, of which one was a genuine gap in the theory, three were
missing constructions, and one was an accounting question nobody in the field had asked.

---

## 1. What the rival actually is

Stripped of framing, the published generative model is:

```
  dtheta_i/dt = omega_i + sum_j K_ij sin(theta_j - theta_i)
```

with a **dense** learned `K` that is **not symmetric**, learned `omega`, one to ten explicit Euler
steps from uniformly random initial phases, a snapshot at fixed `T`, a `(cos, sin)` readout, and a
conventional convolutional decoder of roughly 37 M parameters turning that readout into pixels.
Trained end to end on a distribution-matching loss over pretrained visual features. Their
instruction set is nine instructions in five phases — connect, load, lock, evolve, store — with a
label-and-trigger discipline and a cascade pattern for multi-stage computation.

Three structural facts follow immediately and are worth naming before any analysis.

Noise enters **once**, at `t = 0`. Everything after is deterministic. So the object is a
*pushforward* of the uniform measure on the torus — the same information-theoretic animal as a GAN
generator or a one-step flow map — and **not a sampler of any distribution the system defines**. It
has no invariant measure, no temperature, and no likelihood. `kuramoto::Pushforward` implements it
and says so in its own documentation.

`K` is dense. At `n = 16384` that is 2,147,483,648 bytes of couplings in `f64`
(`kuramoto::dense_coupling_bytes`), and no fabric of physically coupled oscillators implements
all-to-all. Whatever is taped out will be sparse, and the embedding tax this crate already measures
will apply to it.

The decoder is digital and is not small.

---

## 2. The reduction: their state space is already ours

Set `omega = 0` and symmetrise `K`. Then the drift is exactly `-grad E` for

```
  E(theta) = -sum_{i<j} S_ij cos(theta_i - theta_j),        S = (K + K^T)/2
```

which is the **XY model**. Restrict the phases to the `q`-point grid `theta_i = 2 pi a_i / q` and
`cos(theta_i - theta_j) = cos(2 pi (a_i - a_j)/q)`, which is the pair term of the **clock model**
already in `potts::Interaction::Clock`. This is not an approximation on the grid: `to_clock(q)`
reproduces the XY energy to `1e-12`, asserted over every state of a 12³ enumeration.

What the grid costs is a bound, not a hope. Rounding moves each phase by at most `pi/q`, each phase
difference by at most `2 pi / q`, and the cosine is 1-Lipschitz, so

```
  |E(theta) - E(round_q theta)|  <=  (2 pi / q) sum_{i<j} |S_ij|  =:  epsilon(q)
```

and two Boltzmann measures whose energies differ by `epsilon` pointwise satisfy
`|ln Z1 - ln Z2| <= beta epsilon` and `KL <= 2 beta epsilon`. Since `epsilon` falls like `1/q =
2^-b`, **one extra bit of phase precision halves the KL**. That is the exchange rate, and it turns
"their variable is continuous and yours is discrete" from an objection into a price.

Measured on an eight-oscillator all-to-all system at `beta = 1`:

| KL budget | grid `q` | bits/phase | achieved | read floor |
|---|---|---|---|---|
| 1 nat | 256 | 8 | 0.687 | 3.26e-15 J |
| 0.1 nats | 2,048 | 11 | 0.086 | 2.09e-13 J |
| 0.01 nats | 32,768 | 15 | 0.005 | 5.34e-11 J |

---

## 3. The one real gap in the theory, and the correction

Drop the symmetry assumption and the Lyapunov function goes with it. Write `A = (K - K^T)/2`. The
drift on the torus decomposes into three pieces that cannot be converted into one another — a Hodge
split:

| component | source | is it a gradient? | test |
|---|---|---|---|
| gradient | `S` | yes, of the XY energy | Jacobian symmetric |
| solenoidal | `A` | no — pure curl | Jacobian antisymmetric |
| harmonic | `omega` | curl-free but no *periodic* potential | constant field, winding |

The membership test is exact and pointwise: `dF_i/dtheta_j = K_ij cos(theta_j - theta_i)` and
`dF_j/dtheta_i = K_ji cos(theta_j - theta_i)` share the cosine, so the drift is a gradient field
**iff `K` is symmetric**. `kuramoto::jacobian_asymmetry` measures the violation and the test checks
both directions.

Here is the part that is easy to get wrong, and which I did get wrong on the first pass — the test
suite caught it. It is tempting to read the antisymmetric part as *non-reversible acceleration*, the
known trick where a divergence-free drift added to Langevin dynamics preserves the target and
strictly improves mixing (Hwang, Hwang & Sheu 1993). It is not that trick.

The `A` part **is** divergence-free: `div F_A = -sum_ij A_ij cos(theta_j - theta_i) = 0`, because an
antisymmetric matrix contracts to zero against a symmetric one. Measured over a 24 × 24 phase scan,
`max |div F_A| = 0.000e0` — identically zero, not small. But preserving the **Boltzmann** measure
requires `div(pi F) = 0`, which expands to `pi (div F - beta grad E . F)`, and the surviving term
does not vanish. On the same scan, `max |grad E . F_A| = 0.3633` against a coupling asymmetry of
0.375.

**So an asymmetrically-coupled oscillator network has no stationary distribution anyone can name.**
Not the Boltzmann measure of its own symmetric part, not anything else in closed form. Its speed is
real and its correctness is undefined. (A related trap: the *full* drift is not divergence-free
either, since the `S` contraction survives — `kuramoto::divergence` and
`kuramoto::solenoidal_divergence` are deliberately separate functions returning separate numbers,
because conflating them is exactly how an asymmetric coupling gets mistaken for a measure-preserving
perturbation.)

The construction that does work is `g = A grad E` with `A` a constant antisymmetric matrix, because
then two identities hold: `div g = tr(A H) = 0` (antisymmetric against the symmetric Hessian) and
`grad E . A grad E = 0` (an antisymmetric quadratic form). `nonrev` machine-checks both rather than
asserting the theorem. This matters physically: `A grad E` is a linear map applied to the field each
node already computes, so **a fabric that can evaluate `grad E` locally can evaluate `A grad E`
locally**. Non-reversible acceleration is available to a thermodynamic fabric for one extra coupling
matrix, and it keeps an invariant measure the fabric can still certify.

The discrete image of the same thing is **lifting**: one direction bit per site, a directed
proposal, and a reversal on rejection. Exactly `pi`-invariant, provably, with the case analysis
written out in `nonrev::LiftedClock`. Measured exactly from the transition matrices of a `q = 16`
clock site in a field:

| chain | stationary defect | balance defect | exact `tau_int` |
|---|---|---|---|
| lifted | 8.3e-17 | **0.1034** | 3.265 |
| reversible, same proposal | 1.7e-16 | 4.2e-17 | 13.109 |

**4.01× faster, same proposal, same target, invariant measure intact and detailed balance broken on
purpose.** Both numbers computed from the matrices, not estimated from runs. This is the capability
the rival architecture is reaching for and cannot certify; it is now here, and it is certifiable.

---

## 4. The accounting question nobody asked

An analogue variable is said to carry what many bits would. Both halves of that claim are checkable.

The information half is §2: `b` bits buy the phase to within `2 beta epsilon(2^b)` nats, falling
exponentially in `b`.

The energy half is `kT/C` sampling noise. A full-scale sinusoid on swing `V` has signal power
`V²/8`; sampled noise on capacitance `C` is `kT/C`; an ideal `b`-bit quantiser needs SNR
`(3/2) 2^{2b}`. Equate and solve: `C = 12 kT 2^{2b} / V²`, so the energy to drive the node over its
swing, `C V²`, is **`12 kT 2^{2b}`, and the voltage cancels**. No supply scaling, no device scaling
and no architectural cleverness moves it, because it is the noise of the bath.

Against Landauer's `b kT ln 2` for the same `b` bits digitally:

| bits | analogue floor | Landauer | ratio |
|---|---|---|---|
| 1 | 1.99e-19 J | 2.87e-21 J | **69.2** |
| 2 | 7.95e-19 J | 5.74e-21 J | 138.5 |
| 4 | 1.27e-17 J | 1.15e-20 J | 1,108 |
| 8 | 3.26e-15 J | 2.30e-20 J | 141,823 |
| 16 | 2.14e-10 J | 4.59e-20 J | 4.6e9 |

**An analogue variable never undercuts the Landauer floor, at any depth, and the gap grows like
`2^{2b}/b`.** Asserted for every `b` from 1 to 32. So analogue is not the cheap way to carry
information; it is the cheap way to carry *very little* information — and at the depth where it is
cheapest it is worth a handful of bits, which is what a handful of stochastic bits already provides
for less. That is the emulation theorem in its quantitative form, and §2 supplies its constructive
half.

---

## 5. Where the joules actually are

Price the published architecture end to end: 16,384 oscillators, ten Euler steps, eight-bit phase
readout at the thermal floor, 37 M decoder parameters. Oscillator updates charged at the Z1-class
7.09 fJ — the most favourable published figure in the field, and generous to a device nobody has
characterised. Decoder MACs at 1 pJ, a competent digital accelerator.

```
  dynamics   1.162e-9 J    0.0031 %
  readout    5.337e-11 J   0.0001 %
  decoder    3.700e-5 J   99.9967 %
  ---------------------------------
  total      3.700e-5 J
```

**Make the entire substrate free and the total improves by a factor of 1.00003.** Amdahl's law
applied to a thousand-fold claim, and the answer is three parts in a hundred thousand. This is the
same finding as this project's Z1T decomposition reached from the other side — there the sampler was
3% of the system's energy and the fabric around it was the rest.

One nuance worth stating because it cuts against our own usual result: **readout dominance does not
transfer to this architecture.** At the thermal floor an eight-bit conversion is 3.26 fJ, under half
of one published Gibbs update, so the break-even per-update energy at ten steps is 3.26e-16 J and
the dynamics are the larger half at every step count from one upward. Readout dominance is a
property of *our* fabric, which takes thousands of updates between reads, not of an architecture
that reads after ten. (A real converter at ~30× the floor does bring the two halves level again.)

---

## 6. Their instruction set, lowered and priced

| DS-ISA | ferrotherm | ledger |
|---|---|---|
| `Connect i j w` | `GraphBuilder::couple` | one write per endpoint |
| `Disconnect i j` | dropped before `build` | one write per endpoint |
| `Bias i h` | `GraphBuilder::bias` | one write |
| `Load i v` | initial state | one write |
| `Lock i v` | `Sampler::clamp`, deferred to next `Evolve` | one write |
| `Unlock i` | `Sampler::unclamp`, deferred | one write |
| `Evolve t` | `Sampler::sweeps` or `Kuramoto::step_euler` | one sample per moving node per step |
| `Store i` | `Sampler::read_subset` | one read |
| `Barrier` | synchronisation point | nothing |

Nine instructions, nine rows, every row ending in a ledger charge. The published ISA states no
energy for any instruction; lowered here, every DS-ISA program acquires a joules figure on any
device model in `ledger`. `dsisa` asserts the lowering is not a reinterpretation: a program and the
hand-written crate code it lowers to produce identical state **and identical ledgers**, operation
for operation.

Two things the lowering exposes.

**Connect is a write, and writes are the expensive operation.** A programming model whose first
phase is "dynamically configure connectivity" spends its budget before the dynamics begin. Measured
on a four-node ring at Z1 prices:

| evolution | configuration share |
|---|---|
| 1 sweep | 99.86 % |
| 1,000 sweeps | 97.61 % |
| 100,000 sweeps | 30.22 % |
| **390,000 sweeps** | **10.00 %** |
| 1,000,000 sweeps | 4.15 % |

Eight writes at 153.6 pJ is 1.23 nJ, and a four-node fabric needs about 390,000 sweeps before
configuration falls under a tenth of the bill. The label-and-trigger discipline is the right
instinct — batch the configuration, release once — and it is what this crate's schedules already do.

**`Evolve` has no temperature, and therefore no invariant measure.** Ours carries a `beta`, which is
what makes the result a distribution with a certificate rather than a point. Set `beta = infinity`
and their semantics come back exactly — zero-temperature relaxation to a local minimum, asserted in
`the_zero_temperature_limit_reproduces_relaxation`. **Their execution model is a special case of
ours, in code, not in argument.** And the same assembled program runs on both the spin sampler and
the oscillator integrator through one interface, charging the same writes, the same reads and the
same samples — the "same trait as a CPU" property extended to their ISA.

---

## 7. The two lanes

**Lane A — replication, priced.** `kuramoto::Pushforward` is their architecture exactly: seeded
uniform phases, `n` Euler steps, `(cos, sin)` readout, external decoder. Reproducible bit for bit
from a seed, every step and readout on the ledger, and the frozen-random-`K` ablation directly
runnable. This is the comparator, and it is what makes §5 a measurement rather than a model.

**Lane B — absorption, certified.** `phasegen` is the same physics with the decoder deleted. Outputs
become Gaussian units inside the same energy function:

```
  E(x, theta) = 1/2 x^T A x - b^T x - x^T C u(theta) - sum_{i<j} S_ij cos(theta_i - theta_j)
```

Both conditionals are exactly samplable, which is the whole design. Outputs given phases are
Gaussian — one Cholesky, one triangular solve. A phase given everything else is **von Mises**:
collecting the terms in `theta_i` gives `-R_i cos(theta_i - phi_i)`, so the conditional is
`vM(phi_i, beta R_i)`, sampled exactly by Best & Fisher's method with `2 pi I_0(beta R)` as its
normaliser in closed form. A von Mises conditional is the circular heat bath, and on the `q`-point
grid it is *exactly* the clock-model heat bath this crate already samples — so the continuous and
categorical lanes are one sampler at two resolutions, with §2's covering bound as the stated
distance between them.

What this buys that the decoder architecture cannot have: the outputs integrate out in closed form,
leaving `E_eff(theta) = -1/2 (b + Cu)^T A^-1 (b + Cu) - sum S_ij cos(theta_i - theta_j)`, which is
enumerable on a grid. So `enumerate_grid` returns **the exact distribution and exact `ln Z` of a
complete generative model** — and the block-Gibbs sampler is scored against it.

Which `ln Z`, though, is worth one sentence, because the obvious reading is wrong. `log_z` is the
partition function of the model **as read on the grid**: 7.330681 at `q = 16` and 8.716975 at
`q = 32`, rising by `n ln 2` for every doubling, because the grid sum carries a factor of one cell
volume per phase. The phase integrand is smooth and periodic, so that sum is the torus integral to
machine precision and the two differ by exactly `n ln(q / 2 pi)`. `log_z_continuum` divides it out
and returns **5.461258** at every `q`, which independent quadrature over the torus confirms. The
model's number is that one; the grid's number is a property of the readout.

No published generative model in this field has an exact partition function,
because a convolutional decoder has none. This one does, because every block was chosen so that it
would.

---

## 8. What is shipped, and what is not

Shipped, tested, clippy-clean at `-D warnings`, 48 new tests, no regressions across the existing
1,650:

- `kuramoto` — general asymmetric Kuramoto, Hodge split, gradient/solenoidal/harmonic separation
  with exact membership tests, XY and clock reductions, covering and KL bounds, two-oscillator
  closed-form oracle, the pushforward lane.
- `nonrev` — skew-drift Langevin with both invariance identities machine-checked; lifted clock
  sampler with exact matrix-level stationarity, balance-violation and autocorrelation comparison.
- `dsisa` — the nine instructions, assembly, two substrates behind one interface, lowering
  equivalence, configuration-share pricing.
- `precision` — the `kT/C` bound, the analogue/Landauer ratio, the emulation cost, the generator
  budget and Amdahl analysis, an oscillator-fabric price floor that refuses to invent an update
  energy.
- `phasegen` — the in-fabric generator, von Mises conditional, exact Gaussian marginalisation, grid
  enumeration oracle.

Not yet built, in the order I would build them:

1. ~~**Sparse `K`.**~~ **Answered, with one input still missing.** `kuramoto::truncate_to_degree`
   and `Truncated::kl_bound` price the third mismatch in the same unit as the first two — dropping a
   set of couplings moves the energy pointwise by at most their total mass, so `KL <= 2 beta *
   dropped_mass`, and the grid gap and the sparsity gap simply add. `examples/sparse_k` measures the
   truncation exactly, by enumerating both laws.

   The answer is that **it depends entirely on the trained weights, and not at all on the
   architecture**. On a heavy-tailed coupling, truncating to degree 3 of a possible 4 keeps 95.6% of
   the coupling mass and costs `0.0007` nats. On a flat coupling of the same size and degree it
   keeps 60% and costs `0.0505` — seventy times more. Against that, keeping every coupling exactly
   by copy-splitting costs a factor of **4,096 in oscillators** at the published `n = 16384` on a
   degree-6 fabric: 67,108,864 of them for a model of 16,384.

   So the measurement to demand of any dense oscillator architecture is one number: what fraction of
   the coupling mass sits in the heaviest `d` couplings per oscillator. If it is concentrated the
   fabric is buildable and truncation is nearly free; if it is flat the architecture owes four
   orders of magnitude in silicon. **What is still missing is that number for a real trained `K`,
   because the published models ship a decoder rather than a coupling matrix.**
2. ~~**Pathwise gradients through the dynamics.**~~ **Built, and it turned up a second gap.**
   `pathwise::euler_adjoint` is the exact reverse-mode gradient through the Euler map, checked
   entry by entry against central differences to `7e-10` — and, because a gradient check validates
   the derivative of *whatever* was implemented, the forward map is pinned to the two-oscillator
   closed form before any gradient is taken. `examples/pathwise` fits a coupling from zero with it,
   to machine-zero feature error.

   The second gap is the sim-to-hardware one, and it is now measurable rather than asserted.
   `pathwise::integrator_gap` compares the Euler map against a fourth-order reference in **both**
   the trajectory and the gradient, each error relative to the size of the quantity it belongs to.
   Over five systems and five step sizes the gradient's relative error exceeds the trajectory's in
   **22 of 25** cases, by as much as **32x** — largest on the strongly coupled and larger systems,
   which is where a real model lives. So a step size chosen by watching the trajectory is not one
   the gradient agrees with, and `dt` is a trained parameter in disguise: a model fitted through a
   coarse Euler map has absorbed that map's error into its weights, and silicon has no such error
   to cancel.

   The three exceptions are pinned in a test of their own, because the temptation is to state this
   as a theorem and it is not one. They are the smallest, most weakly coupled fixture at its finest
   steps, where both errors are already under a percent. What remains true without exception is the
   weaker and more useful claim: **the two errors are not proxies for one another.**
3. **A measured ADC line.** The read price in `precision::oscillator_fabric_floor` is a thermal
   bound. A real converter's energy, metered, turns §5's readout row from a floor into a fact.
4. ~~**Scale.**~~ **Answered on planar couplings, and the premise turned out to be wrong.** The
   sentence that used to sit here said everything is verified at `n` in the single digits, "where
   exact oracles exist". That carries a premise — that enumeration is the only exact oracle — and
   `pfaffian` refutes it. The Kac–Ward determinant returns `ln Z` and `<E>` for **any** planar
   graph at **any** beta in `O((2E)^3)`: polynomial, not exponential, and exact rather than
   estimated. The crate had it all along and used it to verify nothing but itself.

   `examples/scale` wires it to the cluster sampler and walks a ladder to `n = 900`, where the
   exact answer takes 29 seconds and enumerating it would take `10^271` terms. All eight rungs'
   95% intervals cover the exact energy, and the determinant's phase residual stays at `3.8e-15`
   — zero in exact arithmetic, and a free check that the embedding stayed sound.

   **The finding is the other way round from the assumption.** Every rung is also asked a question
   it must answer *no* to: does the same interval also cover a lattice one percent colder? At
   `n = 16, 36, 64` it does. There, agreeing with the exact answer separates almost nothing.
   Resolving power rises monotonically with size — `1.0, 1.7, 2.3, 3.6, 5.0, 5.5, 7.1, 8.7` sigma
   across the ladder — at least as fast as `sqrt(n)`, because an intensive observable's error
   shrinks as it is averaged over more spins while the gap it must resolve does not shrink with it.
   So along *this* ladder a small lattice is where an exact oracle is cheapest and where it
   discriminates least, and verifying only at single-digit `n` is not the conservative choice it
   looks like.

   **That is a statement about this ladder, not a law, and the distinction was measured rather than
   reasoned about.** The ladder holds the statistic and the budget fixed and varies only `n`, so `n`
   is the only thing that can move the power. Injecting one real defect into the Gibbs update
   instead — a `beta` scaled by `d` — and asking which test notices gives identical answers from a
   4-spin test and a 125-spin one: both survive `d = 1.02` and `1.05`, both catch `1.10` and `1.20`.
   Resolving power is a property of the **statistic** and the **budget**; size moves it only when
   those are held still. The large oracle's real advantage is **reach** — `cftp` referees a model
   where enumeration (`2^125`) and elimination (`2^25` per node) both fail outright — and
   `pfaffian::tests::an_exact_check_discriminates_better_as_the_lattice_grows` pins both halves:
   blind at `n = 16`, resolving at `n = 144`, from one fixed budget.

   **What this does not settle.** Kac–Ward is a planarity argument, and their coupling is dense and
   asymmetric — exactly the structure it refuses. Computing `Z` exactly for a dense non-planar
   coupling at `n = 16384` is `#P`-hard, so this is not a gap in the crate but a fact about the
   problem. What is now available at scale is an exact referee for the *planar* case, which is the
   one a real oscillator fabric with nearest-neighbour coupling would actually build.

---

## 9. The one-paragraph version

Their programme is a Kuramoto system whose coupling is dense and asymmetric, integrated for ten
deterministic steps from random initial conditions, read out through a digital decoder that is
99.997% of the energy. The symmetric part of that system is the XY model, which on a `q`-point grid
is the clock model this crate already samples exactly, at a KL cost that halves with every bit. The
asymmetric part is not the non-reversible acceleration it resembles: it is divergence-free but does
not preserve any Boltzmann measure, so the machine is fast and its output distribution is undefined
— while the construction that *does* preserve the measure runs on the same fabric and is now
implemented, with a measured 4× speed-up and detailed balance broken on purpose. Their instruction
set lowers onto ours in nine rows, executes identically, and acquires the joules figure it never had;
their `Evolve` is our `Evolve` at `beta = infinity`. And an analogue variable at `b` bits costs
`12 kT 2^{2b}` against Landauer's `b kT ln 2`, so it is never the cheap way to carry information at
any depth. What remains of their programme after that is a decoder we can delete, which `phasegen`
does — yielding a generative model with an exact partition function, which nothing in the field
currently has.
