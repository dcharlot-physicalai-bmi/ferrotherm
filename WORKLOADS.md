# What to program on a thermodynamic fabric

Institute for Physical AI @ BMI. Every entry names its **oracle** — the thing that says what the
right answer is — and reports what was **measured against it**, including where the method stops.

A workload without an oracle is a demo. This field has a long history of reporting results against
whatever the last paper achieved, and that is how fifteen years of speedup claims got walked back.
Everything below is scored against something that cannot be argued with: an exact solution, a
closed form, or a planted optimum that was chosen before the problem was built.

---

## The rule that governs this file

**Report the rate, not the mean.** Four separate times while building these, an average concealed
the real behaviour: the planted-instance hardness peak, the Wishart difficulty, a seed-sensitive
autocorrelation, and a mixing threshold. In one case the mean said a family was easy when its solve
rate said the opposite.

**And a test that cannot fail is not a test.** Two of the workloads below shipped a first version
whose assertion was vacuously true. Both are recorded in place rather than quietly fixed.

---

## 1. Sampling-based control — `src/mppi.rs`

**Why it belongs here.** MPPI weights sampled trajectories by `exp(-cost/λ)`. That *is* Boltzmann
weighting with λ as temperature, so a machine whose native operation is drawing Boltzmann samples
performs the expensive part of the algorithm directly. It is also the workload that connects this
stack to a robot, and no thermodynamic vendor is pursuing it.

**Oracle.** On a linear system with quadratic cost, the optimal controller is known in closed form.
The Riccati solution is itself checked twice before anything is scored against it: its residual, and
a perturbation test confirming the gain is a minimum in both directions.

**Measured.** **7.1% above the provable optimum** on a stable plant, at horizon 5 with 10 refinement
passes — **over a 200-step run**, and that last clause is load-bearing. See below.

**Where it stops**, which is the more useful half:

| plant | horizon | iters | excess @200 | @100 | @800 |
|---|---|---|---|---|---|
| stable, `a = 0.9` | 5 | 10 | **7.1%** | 3.4% | 22.6% |
| stable, `a = 0.9` | 5 | 1 | 28.7% | 26.0% | 43.2% |
| stable, `a = 0.9` | 15 | 10 | 19.9% | 10.6% | 78.8% |
| unstable, `a = 1.1` | 10 | 30 | 15.1% | 7.2% | 61.2% |
| unstable, `a = 1.1` | 30 | 30 | **1446%** | 733.5% | 5400% |

**The step count is part of every number in that table, and it was missing.** Excess over the
provable optimum is not a property of the method: MPPI injects `sigma` noise at every step forever,
while the LQR oracle's `cost_to_go` is a finite infinite-horizon cost from `x0 = 1`, so the ratio
grows without bound in the horizon it is measured over. The flagship 7.1% is **1.0% at 25 steps and
22.6% at 800**. It is a coordinate — a number plus the run length it was taken over — and it was
published as though it were a property.

**And the 729% that used to sit in the last row was wrong.** At 200 steps, where all three stable
rows reproduce to the printed digit, horizon 30 gives 1446%. 729% is what horizon 30 gives at *100*
steps — but at 100 steps the row above it reads 7.2%, not the 15.7% that was published beside it.
No single run produced both numbers.

One refinement pass — the textbook receding-horizon form — is not converged. A longer horizon makes
it *worse*, because rollouts are open-loop and noise compounds instead of being corrected inside
them. An unstable plant fails outright at horizon 30, since the state grows like `a^H` inside every
rollout; practical MPPI stabilises rollouts around a base policy, which is not implemented here.

---

## 2. Categorical optimisation — `src/categorical.rs`

**Why it belongs here.** Real problems have variables that take one of `k` values, and how those are
spelled in spins is a compiler decision with measurable consequences.

**Oracle.** Feasibility is exactly decidable: a state either decodes to a valid codeword or it does
not. The workload has no objective at all, so nothing competes with the constraint and any failure
belongs to the encoding.

**Measured.** At an adequate penalty, **both encodings are perfectly feasible** — 1.0 at every `k`
up to 32. There is no gap there, and this file's first test found one only vacuously.

The real difference is how weak a penalty each tolerates. Smallest penalty reaching 0.99 feasible:

| k | domain wall | one-hot | ratio |
|---|---|---|---|
| 4 | 0.074 | 0.358 | 4.8× |
| 8 | 0.163 | 0.466 | 2.9× |
| 16 | 0.212 | 0.606 | 2.9× |
| 32 | 0.276 | 0.787 | 2.9× |

Domain wall needs roughly **three times weaker** a penalty. That matters more than the spin count: a
penalty is *added* to whatever objective the model encodes, so a large one distorts it.

---

### A hardware p-trit's sampling rule, held to the exact law — `examples/sps_exact`

Rhee, Jang, … K. M. Kim (KAIST), *Advanced Science* 2026, e76754 (doi:10.1002/advs.76754) report a
NbOx Mott-memristor **p-trit** — the first device-level counterpart to the p-trit this crate emits
as RTL — and drive it with **Segmented Probabilistic Sampling**: one scalar input per node, and in
each 2π/3 phase interval only the two ADJACENT states get probability. The paper calls the rule
approximate and validates it by solution quality on Max-3-Cut. It never asks what the chain
samples. On all 3⁶ states, by a direct solve, against the enumerated law (heat-bath control: TV
`1e-15`):

| rule, as decomposed here | β = 0.5 | β = 1 | β = 2 | β = 4 |
|---|---|---|---|---|
| third state excluded, exact pair odds — TV at the programmed β | 0.466 | 0.393 | 0.230 | 0.111 |
| … TV at the NEAREST β, and that β | 0.297 @ 1.22 | 0.230 @ 1.73 | 0.140 @ 2.83 | 0.092 @ 4.81 |
| ground-state mass, against Boltzmann's at the same β | 0.186 / 0.058 | 0.324 / 0.146 | 0.586 / 0.417 | 0.842 / 0.790 |

Three things follow, none of which depends on how the paper's one ambiguous equation is read.
**It is not a Boltzmann sampler at the temperature it is programmed for.** **It samples COLDER** —
the nearest β is always above the nominal one, because the state it excludes is the costliest —
which hands optimisation more ground-state mass for free and is a bias for the inference and
learning uses the paper proposes. And **an ordinal device (off–osc–on, no on↔off path) finds the
optima and cannot share them**: with the omitted interval clamped to the nearer end, ground-state
mass is 0.998, yet TV stays above 0.42 at every sharpness, and the whole of it is in the split
among symmetry-related ground states — the cyclic rule puts 0.0832 on each of twelve, exactly
1/12, where the ordinal one puts 0.16–0.18 on a few and **zero** on most. The look-up-table remap
the authors offer as a cure is what restores the symmetry.

What is read from the paper and what is ours is stated at the top of the example. The ordinal
rows are a BRACKET, not a claim about their machine: a saturating voltage map collapses
ground-state mass to ~0.5, which their own reported convergence rules out, so that reading is
shown and rejected. Their headline comparison deserves the same care — "~33% of the overhead" is
one operation against three because the binary baseline solves Max-3-Cut by three recursive
bisections, which is not how three states are put on binary hardware; one-hot and domain-wall are,
and the table above this section measures them. Their prototype's fourteen nodes are
microcontrollers emulating the measured device, and 2.4 nJ per bit is an estimate for the
oscillator alone.

## 3. Thermodynamic linear algebra — `src/tla.rs`

**Why it belongs here.** An Ornstein–Uhlenbeck network's stationary distribution is
`N(A⁻¹b, β⁻¹A⁻¹)`, so equilibrating one *solves* a linear system and its covariance *is* the
inverse. This is the workload Normal Computing's programme is built around.

**Oracle.** Gaussian elimination with partial pivoting, in the same file.

**Measured.** The exact-transition integrator is unbiased, and the Euler–Maruyama chain's
per-eigenmode covariance bias follows the predicted `2/(2 - dt·α)` law — so the bias is not merely
observed, it is *predicted and confirmed*. Sampled covariance recovers `A⁻¹`.

### ⛔ AND IT LOSES, ON A DENSE SYSTEM, AT BOTH MOMENTS

This entry used to stop above, and stopping there was selling the workload. `tla::dominance`
measures it against our own code on the fair work axis — one matrix-vector product per
conjugate-gradient iteration, one per Euler–Maruyama step, so both methods are billed in the same
unit.

| matrix-vector products | conjugate gradient | the OU mean |
|---|---|---|
| ~40 (= `n`) | **`1e-15`** | `1.0` |
| 10,000 | — | `1.2e-1` |

**250× the work for fourteen orders of magnitude worse accuracy.** The critique holds and it is not
close.

The obvious defence is that the sampler returns a covariance a solve does not. **On a dense system
that defence does not survive either, and this part goes past what the paper claims.** The cheap
integrator's covariance bias does not shrink with steps — it shrinks with `dt`, and shrinking `dt`
costs steps, so the two fight; at 200,000 matvecs the best error available is about **7%**. The
unbiased integrator reaches 0.8%, but it opens with an `O(n³)` eigendecomposition, which already
costs more than inverting the matrix. Against an exact inverse at `n` matvec-equivalents and zero
error, both lose.

**What we do NOT claim, because we have not measured it.** The large-sparse case, where `O(n³)` is
unaffordable and a selective-inversion route has its own fill-in problem, is a genuinely open
comparison. It is the only place this workload's argument can still live, and until it is measured
here this entry is a record of a method that loses on the case we did test.

**Why it stays in the catalogue.** Because it is the honest state of a workload this field is built
around, and because a catalogue that only listed wins would not be a measurement of anything.

### The adjacent paper, and a correction to our own citation of it

arXiv:2608.09743 (Kirsten et al., **Signaloid**, 2026-08-10 — one author also at Cambridge) is the
nearest published result, and until this entry was written we cited it, from a summary rather than
the paper, as *"the mean of the OU dynamics is preconditioned gradient descent"*. **That is not what
it says.** Its theorem is about the **covariance** dynamics of matrix inversion, with `b` set to
zero: *"to a first-order approximation, the covariance dynamics are mathematically identical to
preconditioned gradient descent on the Frobenius norm of the residual"*. The measurement above is
about the **mean** route to `Ax = b`, which is a different quantity — so our result complements
theirs rather than repeating it, and the earlier citation was wrong in the moment it named.

Read at the source, their numbers also carry caveats worth passing on, because they are the kind
this catalogue exists to surface. The 100,000-fold speedup is Python/NumPy wall-clock on one
laptop, with the sampler **held at a fixed 1,000,000 samples** while the solver stops at a
tolerance — not equal budgets. Their gradient descent is handed precomputed eigenvalues, which the
authors say *"defeats the purpose of iterative methods in practice"*. And their own Table 4 has
that gradient descent 20–100× **slower** than Newton–Schulz once the condition number reaches 100.
They also fence their headline themselves: the redundancy is *"specific to problems that involve a
convex quadratic potential with a single global minimum"*, and for non-convex landscapes the noise
*"could remain algorithmically essential to escape local minima and cross energy barriers"* —
which is the ground every other workload in this file stands on.

---

## 4. Spin-glass physics — `src/ising.rs`, `src/planted.rs`

**Why it belongs here.** The substrate *is* the model. Nothing is being emulated.

**Oracle.** Onsager's exact solution for the 2D Ising model, and planted instances whose optimum was
chosen before the couplings were built.

**Measured.** Magnetisation agrees with Onsager to **at most 0.0086** across β = 0.45 to 0.7,
usually under 0.003 — *when annealed in*.

**A finding worth carrying.** Quenching a random 64×64 lattice straight to a cold β traps it in a
two-domain striped state: |m| = **0.029** at β = 0.7 where Onsager says **0.990**. The sampler is
not wrong; the chain never left its initial condition. The certificate reports exactly that, so the
same quench produces a result the machinery refuses to bless.

**Planted difficulty is not monotonic.** Frustrated loops show an easy–hard–easy transition peaking
near four loops per edge, where greedy solves 4 of 16 seed pairs against 16 of 16 at both extremes.
The Wishart ensemble is monotonic and hard below α = 1 — and the two families **fail differently**: a
lattice miss can be 17% above the optimum, a Wishart miss under 2%. Any benchmark reporting mean
excess calls Wishart easy when it is not.

⚠ 2D spin-glass ground states in no field are polynomial-time computable, so nothing here is hard in
the complexity sense. These are benchmarks for *heuristics* and must be described that way.

---

## 5. Energy-based model training — `src/dtm.rs`

**Why it belongs here.** A chain of energy-based models trained by contrastive divergence is the
flagship workload of the thermodynamic-computing literature, and the negative phase is exactly what
a sampler is for.

**Oracle.** The data's own statistics, against an untrained-noise baseline.

**Measured** at the published flagship configuration — 70×70 G12, 8 chained EBMs, 4,900 nodes, 784
visible sites, 247,904 parameters, real binarised Fashion-MNIST: per-pixel MAE **0.128** against a
noise baseline of **0.474**, so samples land 72.9% closer to the data than noise.

⛔ **That figure is not reproducible, and the reason is a defect in how it was taken.**
`examples/dtm_scale` trained inside `while start.elapsed() < budget`, defaulting to **120 seconds**
— so the quality it reported was a function of how fast the machine was and what else was running
on it. A faster box takes more gradient steps and gets a better number from the identical command.
That is a division by wall-clock time reported as a property of the method, which is exactly what
this repository's `host` and `ledger` documentation warns against everywhere else. Neither the step
count nor the machine was recorded, so **0.128 cannot be reproduced or refuted**.

The example is step-bounded now (`dtm_scale <images> [steps]`, default 2000) and prints the step
count, grid, layer count, image count and learning rate beside the MAE. The wall clock survives only
as a safety stop, and a run it truncates says so loudly rather than reporting a quality figure as
though the run had finished. **Regenerating this row needs the dataset and a real training run; the
number above stands as an unreproducible historical claim until then, and should not be quoted.**

**`K_mix`, certified at the paper's site count.** `examples/kmix_exact` measured the sweeps a
denoising step needs exactly at `n = 9` and `12` (at most 6, against the paper's 250) and said what
it could not reach. `cftp::Bounding` — perfect sampling for models with couplings of both signs —
reaches it: `examples/kmix_certified` reproduces those two sizes (certificate 8 beside an exact 6)
and then certifies the all-visible nearest-neighbour grid up to **70×70 = 4,900 sites: worst
coalescence 64 sweeps, no refusal in 10,240 exact draws**, flat from 576 sites up. The constant
covers the site count by about four on this family; the forty-fold margin seen at twelve spins
does not transfer, and the flagship configuration (latents, G12, eight steps) is unmeasured.

**On the flagship SHAPE a fixed penalty buys `K_mix` by not learning — `examples/kmix_flagship`.**
G12 wiring, 16% of sites visible at random, eight steps, `dtm_scale`'s settings, a frustrated `+-J`
grid as data. Two referees: the bounding-chain certificate, and R-hat over dispersed chains traced
*after* the paper's 250 sweeps, which can say 250 is not enough.

| L (sites) | penalty | learned @ 32k steps | worst coalescence | refused | R-hat after 250 |
|---|---|---|---|---|---|
| 10 (100) | 0.35 | 0.13 | 8 | 0 | 1.006 |
| 10 (100) | off | **0.80** | cap (8,192) | 27 of 144 | 1.010 |
| 20 (400) | 0.35 | 0.03 | 16 | 0 | 1.004 |
| 20 (400) | off | **0.59** | — | **144 of 144** | 1.007 |
| 28 (784) | 0.35 | 0.03 | 16 | 0 | 1.002 |
| 28 (784) | off | **0.40** (0.45 at 8k) | — | **144 of 144** | **1.055** (1.137 at 8k) |

"Learned" is the share of the data's nearest-neighbour correlations that *generated* samples
reproduce; sampling noise caps it near 0.9. The fixed penalty keeps every layer certifiably fast
and the model learns almost nothing; without it the model learns and the conditionals leave the
fast-mixing regime, worse with size. **This indicts a FIXED penalty, which is this crate's
simplification. The paper adapts it per layer, lowering a layer's penalty when its conditional's
autocorrelation at lag `K` is near zero and raising it otherwise — and `dtm::acp_update`, this
crate's version of that controller, had until this measurement never been called by anything but
its own test.** (Its constants are documented as the paper's; two automated reads of the paper did
not locate them, so they are flagged unverified at the function.) The data here is synthetic.

**The certified frontier shrinks with size, and neither controller finds it
(`kmix_flagship <L> frontier`).** Sweeping the fixed penalty, the best setting that still certifies
after 32,000 steps learns **0.39** at `L = 10` (penalty 0.10) and **0.17** at `L = 20`; everything
that learns more fails a referee. A bounding chain that fails to coalesce is a sufficient condition
failing, not proof of slow mixing, so that frontier is a lower bound. A controller of the form the
paper describes — `acp_update` fed each layer's per-site spin autocorrelation at lag `K`, which is
*our* choice of an observable the paper does not name — drove the penalty to **zero** at both
sizes: on the resulting `L = 20` model it reads **0.023 on its worst layer**, under its own 0.03
threshold, while R-hat over dispersed chains reads **1.92** after 250 sweeps and the certificate
refuses 144 of 144. That is `examples/burnin` inside a training loop — a mean-subtracted
single-chain statistic reads a stuck chain as decorrelated. **The repair we tried did not work
either:** the same law driven by R-hat pulled it to 1.009 and the certificate still refused every
draw, because a multiplicative law that decays to its floor early needs ~38 updates to climb back
and the run has 40.

**A third controller, driven by the certificate, is a partial success (`C-ACP`).** Its sensor is
the certificate at the deployment budget — does a bounding chain coalesce within `K_mix` on four
draws — and it doubles a layer's penalty on a refusal, easing it by a tenth when all four finish
inside half the budget. It is the only controller that keeps exact sampling possible: **no draw
refused at any size**, where the other two end at 144 of 144. At `L = 10` it certifies and learns
**0.50** against the uniform frontier's 0.39, by penalising one layer heavily and the rest barely
(0.001–0.122). It does not hold the budget at size — slowest draw 512 sweeps at `L = 20`, 2,048 at
`L = 28`, where one layer's penalty has run away to 4.1 without fixing it. That is an actuator
mismatch: the penalty acts on *correlations*, and a bounding chain's coalescence is governed by
coupling *magnitudes*, so a large coupling with almost no correlation is invisible to the one and
poison to the other.

**The matched actuator closes part of the gap, and the rest is statistical (`PROJECT`).** Train
with no penalty; whenever a layer fails the certificate, scale its couplings down until it passes.

| L (sites) | uniform frontier (penalty) | `C-ACP` | `PROJECT` | `QPROJ` |
|---|---|---|---|---|
| 10 (100) | 0.39, certified (0.10) | **0.50, certified** | 0.66; slowest draw 256, R-hat 1.016 | 0.47, certified |
| 20 (400) | 0.17, certified (0.10) | 0.23; slowest draw 512 | **0.25, certified** | 0.14, certified |
| 28 (784) | 0.087, certified (0.20) | 0.22; slowest draw 2,048 | 0.11; slowest draw 512 | 0.063, certified |

`QPROJ` is the projection with a DERIVED margin: coalescence from the past is submultiplicative, so
a layer passing eight of eight draws at an eighth of the budget has a failure rate under
`(3/8)^8 = 3.9e-4` a draw at the full one. That bound is per clamp context and the draws are spread
over four, so the theorem motivates the margin and does not guarantee it. It **certifies in all
nine cells** (three sizes by three training lengths; worst coalescence 64, 32, 32) — the one
prediction of three that held — and it is over-conservative, its worst draw sitting four to eight
times inside the budget. **The frontier falls with size, 0.39 → 0.17 → 0.087, while the penalty
needed to certify rises, 0.10 → 0.20; nothing measured here beats it at size while certified.**
`QPROJ`'s advantage is that it finds its operating point by itself, where a uniform penalty needs
a size-dependent strength known in advance.

Learned structure at 32,000 steps, with why a cell is not certified. Two predictions written before
these runs — that each new arm would hold the budget everywhere — were both wrong, and are
recorded as wrong in `dtm.rs`. What remains is not tuning: the sensor tests eight draws, scoring
takes the maximum over 144 across fresh clamp contexts, and coalescence time is heavy-tailed, so
controlling a small sample's maximum does not bound a large one's. **What every certificate-aware
arm does achieve is that no draw is ever refused — exact sampling stays possible — where the
paper-style controller and no penalty end at 144 of 144 refused.**

**The total-correlation penalty is load-bearing, measured both ways.** Without it `|J|` grows
linearly and never settles — an unmixed negative phase underestimating the model's own correlations,
which looks like learning and is not. With it the increments decelerate and settle.

⛔ **CORRECTED 2026-09-18 — that measurement was made with the penalty's sign REVERSED, and could
not have noticed.** `examples/dtm_scale` applied the term as `J += lr·λ·c_ab`, where `c_ab` is the
model's connected correlation; `Dtm::train_step` applies `J -= lr·λ·c_ab`. Opposite signs, and
nothing pinned either. Both updates have a fixed point, so "the increments decelerate and settle"
is true of both and discriminates nothing — the reversed one settles at a model *more* correlated
than the data, `⟨ss⟩ = (⟨ss⟩_data − λ·m_a·m_b)/(1 − λ)`, which is the opposite of what a penalty
meant to keep sampling easy is for. The direction is now decided by the quantity the term is named
for — and under BOTH of its readings, because the paper's Eq. 15 is `D(∏ P(s_i|x) ‖ P(s|x))`,
product of marginals to joint, which is not the same number as the more familiar `Σ H(p_i) − H(p)`.
On an enumerable conditional the paper's functional is **0.1839** nats, falling to **0.1704** along
`train_step`'s step and rising to **0.1981** along the reversed one; the other reads 0.1734, 0.1617
and 0.1856. Same ordering, and the step shrinks every one of the eight couplings toward zero. The example is fixed, a test pins the sign,
and a mutation row reverses it. **The 72.9% figure above was taken with the reversed sign as well
as the wall-clock defect — one more reason not to quote it.**

⚠ **Calibrate this one.** This is *not* the published FID ≈ 28. Reaching that needs K ≈ 1000 and
≥100 epochs: roughly 2,170 CPU-hours, or ~14 hours on the WebGPU path.

⛔ **And the metric orders models backwards — measured, not argued.** This row used to warn that
"per-pixel marginals are a weak metric: a model can match them without capturing structure". That
warning was an *assertion*, written because the argument is obvious rather than because anyone had
measured it. `examples/metric_calibration` measures it, on datasets small enough for the exact
log-likelihood to be computed beside the marginal MAE.

Against a **bias-only model** — nine pixels, no hidden units, no couplings, so matching the
marginals is the whole of what it can do — on bars-only images, which are made entirely of
correlation:

| arm | per-pixel MAE vs noise | actually learned |
|---|---|---|
| marginals-only | **87.3% closer** | **2.1%** |
| wide (12 hidden) | −39.8% closer, i.e. worse than noise | **95.4%** |

The model that learned almost nothing wins the metric by a wide margin; the model that learned
nearly everything scores *worse than noise* on it. On the symmetric bars-and-stripes set it is worse
still — every true marginal is exactly zero, so a model that has learned **nothing** scores a
perfect 0.0000.

The mechanism is ordinary, not exotic: a maximum-likelihood fit *would* match first moments, since
moment matching is the gradient's fixed point. Contrastive divergence is a biased gradient by
construction, and hidden units give a model somewhere else to spend capacity — so the metric rewards
the model that optimises *it*. What is worth knowing is the size of that effect, and here it is
large enough to flip the ranking.

**This does not make the 72.9% above wrong** — it is what it says it is, and it was measured. It
means the number cannot carry the weight a reader would put on it. Read a per-pixel figure as a
**floor** (a model failing it has certainly not learned) and never as evidence that one has.

---

## 6. Higher-order models on pairwise hardware — `src/reduce.rs`

Every fabric in this repository declares `max_arity: 2`, and plenty of real problems are not
pairwise: a three-body constraint, a parity check, a term saying *these three agree*. Toshiba's
SQBM+ has a PUBO solver taking order 4 and it is the exception, not the rule.

`reduce::to_pairwise` lowers any of it. It introduces an ancilla spin equal to the product of two
existing ones, substitutes it wherever that pair appears, and repeats — paying for each definition
with a penalty larger than the whole model can afford to break it. The pair chosen each round is the
one appearing in the most wide terms, so one ancilla can serve several.

It goes through binary because in spin space *"t equals s_a·s_b"* is itself a three-body statement,
which is the problem being solved. In binary it is Rosenberg's `3y + x_a·x_b − 2x_a·y − 2x_b·y`,
zero exactly when `y = x_a·x_b` and quadratic throughout.

| model | ancillas | check |
|---|---|---|
| one 3-body term | 1 | every state, enumerated |
| one 4-body term | 2 | every state, enumerated |
| three 3-body terms sharing a pair | **1** | every state, enumerated |

**The guarantee is about optimisation, and the tests are exhaustive rather than sampled.** For every
assignment of the original spins, the reduced energy minimised over the ancillas equals the original
plus one constant — so no state is reordered and the ground states correspond exactly. The ancillas
add states, so the Boltzmann distribution over the original variables is *not* preserved at finite
temperature; the penalty makes a violation expensive, not impossible.

Five mutations of the pass were each required to turn the enumeration red. One appeared not to and
was the mutation failing to apply rather than the check failing to see, which is why the mutation
script now refuses to run when its pattern does not match.

## 7. Attention as an energy — `src/dense_memory.rs`

Softmax attention and the modern Hopfield network are the same object. That is the bridge the whole
"thermodynamic computing for AI" thesis rests on, and it is usually asserted rather than checked, so
here it is checked and then pushed until it breaks.

**The identity, held to central differences.** Write the energy

```text
    E(ξ) = −lse(β, Xᵀξ) + ½ ξᵀξ,      lse(β, z) = β⁻¹ ln Σ_μ exp(β z_μ)
```

with the stored patterns as the columns of `X`. Then `T(ξ) = ξ − ∇E(ξ)` is exactly softmax
attention: `X softmax(β Xᵀξ)`. `lse_energy` never calls `attention_update` and the difference
quotient never sees a softmax, so the two sides are independent code. At `h = 1e-6`, K = 8, d = 16,
β = 2, over 100 queries and every one of 16 coordinates, the worst disagreement is below 1e-6.

Attention is therefore **one gradient step, at step size exactly 1**, on a function we can write
down. Three mutations of the energy — dropping `½ξᵀξ`, flipping the `lse` sign, forgetting the
`β⁻¹` — each turn it red.

### ⛔ AND THE OBVIOUS NEXT STEP IS FALSE

The step everyone takes from there is: *attention is an energy-based model, so a machine that
relaxes to that energy computes attention.* It does not. Relaxation returns the energy's minimiser,
`T^∞(ξ)`. Attention returns `T(ξ)`. They are the same vector only in one corner.

Iterating `T` to its fixed point and measuring `‖T(ξ) − T^∞(ξ)‖ / ‖T^∞(ξ)‖`, mean over 20 draws:

| query | β = 0.25 | β = 1 | β = 4 |
|---|---|---|---|
| sits on a stored pattern | 0.818 | 0.0088 | **0.0** |
| diffuse | 0.572 | 0.781 | 0.412 |

**One-step retrieval needs both conditions.** A query already on a pattern *and* a high β — which
is Ramsauer et al.'s separation hypothesis, and is exactly what their one-step theorem assumes. Drop
either one and the fixed point is somewhere else: at β = 0.25 the same pattern query is still 82%
away, and a diffuse query is **41–90% away at every β measured, never small** — and a diffuse query
is what a real attention head has, since a head that already sat on its answer would not need to run.

The descent is real: 0 energy increases across 300 iterations, which is the concave-convex guarantee
holding. It just takes up to 68 iterations, not one. So the substitution is not a speedup with a
constant factor to argue about; in the regime that matters it computes a different function.

**Three of the five mutations against this entry survived the first version of its tests**, and each
survivor was a way the test measured its own bookkeeping instead of the code:

- Deleting the mixed-query regime entirely — every draw at one scale — still satisfied a
  *"at least 20 draws were blended"* count, because the remaining family drifted under the bar. The
  two families now have to **partition**: every diffuse draw below 0.9 and every pattern draw above
  0.99, with exact counts.
- Writing `0.0` into the recorded softmax weights satisfied both of those bounds. A softmax over K
  terms cannot have a largest weight below `1/K`, and that line now says so.
- Substituting the expected value for the measured energy was exactly right, because the expected
  value was `2·flips` — a closed form in the loop index. The references are now squared distances of
  a randomly drawn vector, which no expression in the index reproduces.

### What the omitted constants are for

The published energy carries `β⁻¹ ln P + ½M²` on top, and they are not bookkeeping: they put the
floor at zero, since `lse(β, Xᵀξ) ≤ max_μ x_μ·ξ + β⁻¹ ln P ≤ ‖ξ‖M + β⁻¹ ln P` gives
`E ≥ ½(‖ξ‖ − M)² ≥ 0`. Without them the energy is negative on more than half of a 200-draw scan.

**That bound is sound but slack, and measuring it says by how much.** `lse ≤ max + β⁻¹ ln P` is tight
only when all P logits are *equal*, so a separated pattern set — where one logit dominates, which is
the whole point of a memory — leaves precisely `β⁻¹ ln P` on the table. Measured infimum over 200
draws: **1.0399, against ln(8)/2 = 1.0397.** Zero is attained only in the degenerate case of P
identical patterns, where `lse` is exact and `E(ξ) = ½‖ξ − x‖²` on the nose.

---

## 8. Does reusing a p-bit change the answer? — `src/autocorr.rs`, `Kernel::TickRandom`

Onizawa & Hanyu (arXiv:2604.01564, 2026) introduce a **time-multiplexing reuse factor `c`** — *"the
number of logical p-bits that are sequentially mapped onto a single physical p-bit"* — and their
synchronous **tick-random** policy selects each spin on each tick with probability `p_flip = 1/c`,
updating every selected spin from the previous state. Reusing hardware this way is the paper's route
to a large cost saving, and the paper argues it is free:

> time-multiplexed reuse corresponds to a temporal rescaling of the underlying Markov process rather
> than a change in its transition kernel.

> Such time-thinning arguments are well established in stochastic simulation theory and imply that
> only the convergence speed, not the stationary distribution, is affected.

The Conclusion restates it: *"the effective update rate can be reduced without altering the target
stationary distribution"*. **So `TV(π_{1/c}, π_1)` should be zero for every `c`.** The oracle here is
the exact stationary law of each kernel, solved rather than sampled, so the column below is
arithmetic and not a statistic:

| fixture | β | c = 1.25 | c = 1.5 | c = 2 | **c = 3** | c = 10 |
|---|---|---|---|---|---|---|
| 5×2 grid (bipartite) | 0.5 | 0.287 | 0.376 | 0.450 | **0.503** | 0.558 |
| 5×2 grid | 1 | 0.492 | 0.564 | 0.619 | **0.659** | 0.700 |
| 5×2 grid | 2 | 0.620 | 0.650 | 0.671 | **0.684** | 0.697 |
| 5×2 grid | 3 | 0.609 | 0.614 | 0.620 | **0.624** | 0.628 |
| 10-ring + 2 chords (frustrated) | 0.5 | 0.271 | 0.360 | 0.434 | **0.486** | 0.539 |
| 10-ring + 2 chords | 1 | 0.371 | 0.431 | 0.476 | **0.507** | 0.538 |
| 10-ring + 2 chords | 2 | 0.199 | 0.242 | 0.280 | **0.306** | 0.332 |
| 10-ring + 2 chords | 3 | 0.136 | 0.176 | 0.211 | **0.235** | 0.260 |

**At `c = 3` the two laws disagree on 23.5% to 68.4% of the probability mass.** `c = 3` is not a
corner: it is the reuse factor the paper's own prose headlines, and in both its cost tables the
tick-random row at `c = 3` is the top-scoring synchronous entry. This is not a temporal rescaling.

### Why the thinning argument holds for their other branch and not this one

Thinning is sound for a continuous-time chain in which at most one site moves at a time — their
**Poisson/asynchronous** policy, where the argument is correct and we make no claim against it. A
Bernoulli mask is not a thinning of that chain: it leaves probability `p²` on two **adjacent** sites
moving together from the same stale state, and that is the term that breaks detailed balance. It
lives inside the per-site conditional, so changing `c` changes the kernel, not the clock.

**The uncoupled control is the proof of mechanism, not a smoke test.** Take the same fields and
delete every coupling: now no adjacent pair exists, and the law stops depending on `p` altogether —
worst TV from Boltzmann `3.9e-15` over every `p` and every β. The movement above is the interaction
term, measured.

Two more facts pin the ends. `TickRandom { p: 1.0 }` is **bit-identical** to `Kernel::Synchronous` —
total variation exactly `0.0`, in all eight cells, not merely close. (Held against that kernel rather
than against `peretto`: the closed form carries its own conditioning, and the same comparison against
`peretto` needs a `1e-5` tolerance at β = 3 on twelve spins where this one is exact.) And `p = 0` is
the identity map, which has no unique invariant law — `stationary_solved` returns `Reducible` rather
than whatever a singular solve leaves behind. In between, the movement is **first order in `p` with
no intercept**: `TV/p` = 0.1439, 0.1447, 0.1473, 0.1520 at `p` = 0.01, 0.02, 0.05, 0.1.

### ⚠ What this entry deliberately does NOT assert

**The direction.** On the 10-spin fixtures the distance from Boltzmann falls monotonically as `p`
falls, in all eight cells. On the 12-spin frustrated ring of `examples/tick_random_exact.rs` it does
**not**: at β = 2 it runs `0.150, 0.0015, 0.0048, 0.0047, 0.0031, 0.0009` — down, up, and down again.
An independent replica of this kernel, built to check us, reported monotonicity in 24 of 24 cells on
*its own* coupling draws and so would have licensed an assertion that our own fixtures refute. That
the law **moves** is robust across every fixture, β and `c` we have run; **which way it moves is
not**, and no test here claims it.

The scope is also stated rather than implied: this is the synchronous tick-random branch. It says
nothing about the asynchronous branch, and nothing about whether the paper's cut-quality results
hold — a kernel can sample the wrong law and still anneal to good cuts, which is the separation
`WORKLOADS.md` keeps throughout between *sampling* and *optimisation*.

`examples/tick_random_exact.rs` prints the full twelve-spin table (about seven minutes: the dense
solve is `O(8^n)` in the spin count).

---

## What we deliberately do not do

**Routing, scheduling and portfolio optimisation.** They are MILP in a QUBO costume and they lose to
Gurobi. Chasing them is how this field burned its credibility, and a stack that reports its own
noise floors should not spend that credibility on problems it cannot win.

---

## Claims not to repeat

Every headline multiplier in this field deserves the treatment above. In particular: Extropic's
"~10,000× less energy" was revised down roughly tenfold by their own later SPICE table; Normal's
"up to 1000×" appears in a chip paper containing zero watts and no GPU comparison; and
`QUBODrivers.ExactSampler`, the JuMP ecosystem's own correctness oracle, is 2ⁿ brute force.

We hold ourselves to the same standard, in public, including when it costs a number we would rather
quote — as it does in entries 1 and 5 above, and in entry 3, where the workload loses.

**Added 2026-09-21, each read at the source rather than in summary.**

- **arXiv:2608.06803's "130× lower energy" on a real CMOS Ising chip** compares against a classical
  baseline whose energy is not measured: *"the CPU energy is computed using a 15 W per-core power
  estimate obtained by normalizing the CPU TDP by the number of cores"*. Its own chip figure is an
  assumed operating point — *"We use 50 μs solve time and 9 mW chip power"* — rather than a metering
  result. A projection against a projection, on both sides of the ratio.
- **arXiv:2608.00754 (CN101) reports no energy figure at all.** The chip is fabricated and the
  functional results are silicon, but *"the energy and latency targets that motivate the programme
  are deferred to later chips"*. Any joules-per-something attributed to CN101 did not come from that
  paper.
- **arXiv:2410.14093's Ising machine on a vehicle** measures itself honestly — 284 µs, 3.4 W — but
  states no timing for any other solver on the same assignment problem, so it is an absolute figure
  and not a comparison. The digital bar for sampling-based control is instead arXiv:2601.17231, which
  does name its baselines: 2.33 ms and 14.90 mJ per control step on an FPGA against 7.24 ms and
  37.44 mJ on a Jetson Orin Nano.

**Added 2026-09-22, read at the source in both versions, main text and supplement.**

- **arXiv:2603.27996's Table I row labelled "FPGA (experimental)" is arithmetic, not a measurement.**
  The row reads `15000 Gsample/s, 25 W, 600 Gsamples/J, ~10^2` improvement over the GPU. The
  supplement's own first word about those numbers is *derived*: *"The FPGA numbers reported in Table I
  are derived from the p-computer architecture described in Section II. To isolate the random number
  generation capability, we strip away all logic, LUT, and MAC unit, and retain only the 32-bit
  LFSR-based pseudo-random number generators. The architecture supports 10^5 independent p-bits
  (Table S1), each containing its own LFSR, clocked at 150 MHz ... yielding an aggregate throughput of
  10^5 x 150 x 10^6 = 1.5 x 10^13 samples/s"*. Every cell is a product of three Table S1 parameters,
  and `1.5e13 / 25 W` is the efficiency exactly. **The asymmetry is what makes it worth recording:**
  one paragraph earlier the GPU side of the same table states instrument, sampling rate and averaging
  window — *"asynchronously sampling the on-board power sensors at 100 ms intervals, and the reported
  power corresponds to the time-averaged draw over the kernel execution window"* — so a single label,
  "(experimental)", spans a metered figure and a projected one. The supplement's own Table S1 does not
  repeat the label: it attaches *"experimental values are from our A100 LFSR microbenchmark"* and
  lists the FPGA rows as plain "FPGA". **This review did not locate any sentence, in either version,
  stating that the 25 W board power was measured, on what instrument, at what utilisation, or against
  what idle baseline.** A real Alveo U250 was really programmed — but the supplement says it was used
  *"to generate equilibrium training configurations via Gibbs sampling"*, not to benchmark RNG
  throughput. On our own ladder that row is `Derived`; the GPU row, lacking an idle baseline and a
  reproduced control, is `Measured` rather than `Metered`.
  The authors fence the number twice themselves, and both fences are worth passing on: *"a 'sample'
  refers to one 32-bit random number produced by the underlying stochastic hardware primitive, not a
  full Ising configuration or a complete Gibbs sweep"*, and *"these efficiency figures refer to raw
  stochastic sample generation, not full end-to-end diffusion inference"*. Their "sample" is one LFSR
  output from a design with the LUT and MAC stripped out; a `Ledger` sample here is a full single-node
  Gibbs update including the field sum. **The two are not the same operation and the ratio does not
  transfer.**
- **The same paper's sampling cost, which does check out, is worth quoting because it is large.**
  *"For T = 100 and Nchains = 10, this corresponds to 49,500 Gibbs sweeps per generated sample"* —
  confirmed independently in the supplement as `10 x 4950`, with the authors themselves flagging the
  off-by-one (*"t - 1 sweeps per chain, not t - 1 chains"*, and the final reverse step costs none). On
  the 1000-spin 3D spin glass they name, and given that a sweep *"updates every spin once in a fixed
  order"*, that is **4.95e7 single-spin updates per generated sample**. Each chain's sweeps are
  sequential, so that figure cannot be divided by a p-bit count to get a wall-clock.

### The general form: an advantage that does not charge readout is bounded by readout

Every entry above is a particular case of one rule, and `ledger::advantage_after_readout` and
`ledger::read_budget_for_advantage` compute it. An energy-advantage claim for a physical computer is
almost always a ratio of two **dynamics** figures — what the digital baseline burns computing,
against what the device dissipates evolving. When the ratio is quoted, the device's answer is still
inside the device. **Getting it out is a third term, and it is charged nowhere.**

Inverting the ratio gives a bound that needs no device and no agreement about one: *for this claim
to hold, every value read out must cost no more than N times its own `kT ln 2`.* When `N` comes out
near 1, the claim is not a hardware target — it is a statement that readout is thermodynamically
free.

- **arXiv:2506.15121 (Whitelam, LBNL), "more than 10^11".** Both of its numbers are the paper's own
  and both are dynamics: *"the order-of-magnitude energy budget of denoising using a neural network
  is not less than 5×10^{14} k_BT"*, against *"Over 1000 independent denoising trajectories of the
  trained computer we calculate a mean heat emission of ⟨Q⟩=2.9×10^{3} k_BT"*. The heat is defined
  as *"Q=V(x(0))-V(x(t_f))"* — the potential at the two ends of a trajectory — so obtaining the
  paper's own reported quantity means reading the full state twice, across
  `N_v + N_h = 784 + 512 = 1296` units: **2,592 values per trajectory, 2,592,000 over the 1000 the
  mean is taken over.**

  | per-value readout cost | advantage that survives |
  |---|---|
  | free | **1.72e11** — the paper's ratio, reproduced exactly |
  | 1.18 × the Landauer floor | 1e11 — the last point the headline holds |
  | 10 × the floor | 2.4e10 |
  | 26.4 × the floor | 1e10 |
  | one metered KV260 AXI read | **1.38** |

  **For "more than 10^11" to survive, each of those 2,592 values must be read out for no more than
  1.18 times `kT ln 2`** — at the floor for reading one bit, with nothing left over for a wire, an
  amplifier or a converter. The free-readout row is the control: priced at zero, the same arithmetic
  returns the paper's own ratio, so the collapse is the readout and not our bookkeeping. The
  conclusion does not turn on which temperature is used — 300 K (this crate's constant) and 301.8 K
  (what the paper's own `kT` implies) are both asserted.

  **This is a bound on any readout, not a complaint about one board.** The KV260 row is there because
  it is the only read price in `ledger::CATALOGUE` graded `Metered`, and it shows where a real
  single-beat AXI4-Lite read of one node sits: about 2.0e11 times its own Landauer floor. The paper
  claims a physical implementation only prospectively — *"Realized physically ... such systems could
  perform autonomous generative computation"* — so nothing here says the measurement it did report is
  wrong. What it says is that the quoted ratio is a ratio of dynamics, and the term it omits is the
  one that decides the answer.
