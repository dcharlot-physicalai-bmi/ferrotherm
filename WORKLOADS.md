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

## 3. Thermodynamic linear algebra — `src/tla.rs`

**Why it belongs here.** An Ornstein–Uhlenbeck network's stationary distribution is
`N(A⁻¹b, β⁻¹A⁻¹)`, so equilibrating one *solves* a linear system and its covariance *is* the
inverse. This is the workload Normal Computing's programme is built around.

**Oracle.** Gaussian elimination with partial pivoting, in the same file.

**Measured.** The exact-transition integrator is unbiased, and the Euler–Maruyama chain's
per-eigenmode covariance bias follows the predicted `2/(2 - dt·α)` law — so the bias is not merely
observed, it is *predicted and confirmed*. Sampled covariance recovers `A⁻¹`.

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

**The total-correlation penalty is load-bearing, measured both ways.** Without it `|J|` grows
linearly and never settles — an unmixed negative phase underestimating the model's own correlations,
which looks like learning and is not. With it the increments decelerate and settle.

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
quote — as it does in entries 1 and 5 above.
